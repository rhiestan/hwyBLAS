// Kernels for the banded and packed Level 2 routines:
//
//   GBMV         y := alpha * op(A) * x + beta * y, A general banded
//   SBMV, HBMV   y := alpha * A * x + beta * y, A symmetric/Hermitian banded
//   SPMV, HPMV   the same with A packed
//   TBMV, TPMV   x := op(A) * x, A triangular banded or packed
//   TBSV, TPSV   solves op(A) * x = b, A triangular banded or packed
//   SPR, HPR, SPR2, HPR2   rank-1 and rank-2 updates of a packed matrix
//
// In both storage formats, the stored elements of each column are contiguous
// in memory (see ColumnLayout). As in the dense routines, the matrix is
// processed in strips of 4 columns, using the same vector kernels with the
// columns given as pointers:
//
//   - For triangular and symmetric matrices, a strip [j, j + 4) splits into
//     a small diagonal block, done element by element, and the stored rows
//     outside the strip (below it for lower, above it for upper).
//   - The rows outside the strip that all 4 columns store form a rectangle
//     for the vector kernels: axpy form (op = N), dot form (op = T, C), the
//     fused axpy + dot form of SYMV, or the rank update. In band storage, the
//     columns of a strip end (lower) or start (upper) one row apart, and the
//     few "ragged" elements beyond the rectangle are done one by one.
//
// Square matrices with fewer than kMinStripOffDiagonals off-diagonals are
// instead processed column by column (see UseStrips); GBMV and the packed
// rank updates always use strips.
//
// x and y are contiguous; callers handle argument checking, strides and
// quick returns.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_BAND_PACKED_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_BAND_PACKED_INL_H_
#undef HWYBLAS_SRC_BAND_PACKED_INL_H_
#else
#define HWYBLAS_SRC_BAND_PACKED_INL_H_
#endif

#include <stddef.h>
#include <string.h>

#include "common.h"
#include "hwy/highway.h"
#include "src/band_packed.h"
#include "src/gemv-inl.h"
#include "src/level1-inl.h"
#include "src/rank_update-inl.h"
#include "src/symv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

// Columns per strip.
constexpr size_t kBandStrip = kGemvCols;

// Rows [lo[c], hi[c]) of the columns j + c of a strip, with lo and hi
// non-decreasing in c, split into the rectangle [begin, end) that all
// columns contain and the "ragged" rows of each column outside it:
// [lo[c], RaggedEnd1(c)) above and [RaggedBegin2(c), hi[c]) below.
template <size_t kCols>
struct StripRows {
  size_t lo[kCols], hi[kCols];
  size_t begin = 0, end = 0;

  void Finish() {
    begin = lo[kCols - 1];
    end = HWY_MAX(begin, hi[0]);
  }
  size_t RaggedEnd1(size_t c) const { return HWY_MIN(begin, hi[c]); }
  size_t RaggedBegin2(size_t c) const { return HWY_MAX(end, lo[c]); }
};

// Stored rows of a strip's columns [j, j + kCols) outside the strip: the
// rows below it (lower) or above it (upper) of a square matrix. If
// `general`, all stored rows (GBMV).
template <size_t kCols>
HWY_INLINE StripRows<kCols> OffStripRows(const ColumnLayout& layout,
                                         bool general, bool lower, size_t j,
                                         ColumnLayout::Column* col) {
  StripRows<kCols> rows;
  for (size_t c = 0; c < kCols; ++c) {
    col[c] = layout.At(j + c);
    if (general) {
      rows.lo[c] = col[c].begin;
      rows.hi[c] = col[c].end;
    } else if (lower) {
      rows.lo[c] = j + kCols;
      rows.hi[c] = HWY_MAX(j + kCols, col[c].end);
    } else {
      rows.lo[c] = HWY_MIN(j, col[c].begin);
      rows.hi[c] = j;
    }
  }
  rows.Finish();
  return rows;
}

// Element (i, j) of a column with layout `col`.
template <typename T>
HWY_INLINE T* Element(T* a, const ColumnLayout::Column& col, size_t i) {
  return a + col.offset + (i - col.begin);
}

// Pointers to row rows.begin of the strip's columns (only used if the
// rectangle is not empty).
template <size_t kCols, typename T>
HWY_INLINE void RectangleColumns(T* a, const ColumnLayout::Column* col,
                                 const StripRows<kCols>& rows,
                                 T* (&ptrs)[kCols]) {
  for (size_t c = 0; c < kCols; ++c) {
    ptrs[c] = (rows.end > rows.begin) ? Element(a, col[c], rows.begin) : a;
  }
}

// Calls f(c, i) for every ragged row i of every column c of the strip.
// The lambdas passed here only capture scalars and pointers, which keeps
// vectors out of memory; they need HWY_ATTR like any function using the
// target's instructions.
template <size_t kCols, class Func>
HWY_INLINE void ForEachRagged(const StripRows<kCols>& rows, const Func& f) {
  for (size_t c = 0; c < kCols; ++c) {
    for (size_t i = rows.lo[c]; i < rows.RaggedEnd1(c); ++i) f(c, i);
    for (size_t i = rows.RaggedBegin2(c); i < rows.hi[c]; ++i) f(c, i);
  }
}

// Calls f(width, j) for the strips of an n-column matrix, in increasing
// order of j if `forward`, otherwise decreasing: full strips of kBandStrip
// columns, then single columns.
template <class Func>
HWY_INLINE void ForEachStrip(size_t n, bool forward, const Func& f) {
  const size_t full = n / kBandStrip;
  const size_t num = full + (n - full * kBandStrip);
  for (size_t step = 0; step < num; ++step) {
    const size_t s = forward ? step : num - 1 - step;
    if (s < full) {
      f(kBandStrip, s * kBandStrip);
    } else {
      f(size_t{1}, full * kBandStrip + (s - full));
    }
  }
}

// Whether to process a square band or packed matrix in strips. Strips share
// the loads of x and y between columns, but their bookkeeping (diagonal
// block, ragged rows) only pays off for long columns: below about 128
// off-diagonals (measured on AVX2), the matrix is processed column by column
// with the contiguous Level 1 kernels instead.
constexpr size_t kMinStripOffDiagonals = 128;

HWY_INLINE bool UseStrips(const ColumnLayout& layout) {
  return layout.OffDiagonals() >= kMinStripOffDiagonals;
}

// Column kernels for the column-by-column path.

// y[0:n] += t * x[0:n].
template <typename T>
HWY_INLINE void ColumnAxpy(size_t n, T t, const T* x, T* y) {
  if (n != 0) AxpyContiguous(n, t, x, y);
}

// Returns sum op(a_i) * x_i over n contiguous elements, op conjugating if
// conj.
template <typename T>
HWY_INLINE T ColumnDot(bool conj, size_t n, const T* a, const T* x) {
  if (n == 0) return T(0);
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    const R* ar = reinterpret_cast<const R*>(a);
    const R* xr = reinterpret_cast<const R*>(x);
    return conj ? DotComplex<true>(n, ar, xr) : DotComplex<false>(n, ar, xr);
  } else {
    return DotReal<T>(n, a, x);
  }
}

// y[0:n] += t * a[0:n] and returns sum op(a_i) * x_i, in one pass over a
// (op conjugates if kConj).
template <bool kConj, typename T>
HWY_INLINE T ColumnAxpyDot(size_t n, const T* a, T t, const T* x, T* y) {
  T dot = T(0);
  if (n != 0) AxpyDotCols<1, kConj>(n, &a, &t, x, y, &dot);
  return dot;
}

// Off-diagonal stored rows of column j of a square matrix: (j, end) for
// lower, [begin, j) for upper.
HWY_INLINE void OffDiagonalRows(const ColumnLayout::Column& col, bool lower,
                                size_t j, size_t* begin, size_t* end) {
  *begin = lower ? j + 1 : col.begin;
  *end = lower ? col.end : j;
}

// ---------------------------------------------------------------------------
// GBMV: y := alpha * op(A) * x + beta * y, A is m x n with the layout's
// band. Strips pay off for all bandwidths here.

template <size_t kCols, typename T>
HWY_INLINE void BandGemvStrip(const ColumnLayout& layout, Op op, size_t j,
                              T alpha, const T* a, const T* x, T* y) {
  ColumnLayout::Column col[kCols];
  const StripRows<kCols> rows =
      OffStripRows<kCols>(layout, /*general=*/true, false, j, col);
  const T* ptrs[kCols];
  RectangleColumns(a, col, rows, ptrs);
  const size_t len = rows.end - rows.begin;
  if (op == Op::kNoTrans) {
    T t[kCols];
    for (size_t c = 0; c < kCols; ++c) t[c] = alpha * x[j + c];
    AxpyCols<kCols>(len, ptrs, t, y + rows.begin);
    ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
      y[i] += *Element(a, col[c], i) * t[c];
    });
  } else {
    const bool conj = (op == Op::kConjTrans);
    if (conj) {
      DotCols<kCols, true>(len, ptrs, alpha, x + rows.begin, y + j);
    } else {
      DotCols<kCols, false>(len, ptrs, alpha, x + rows.begin, y + j);
    }
    ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
      const T v = *Element(a, col[c], i);
      y[j + c] += alpha * (conj ? Conj(v) : v) * x[i];
    });
  }
}

template <typename T>
void BandGemv(const ColumnLayout& layout, Op op, size_t m, size_t n, T alpha,
              const T* a, const T* x, T beta, T* y) {
  ScaleVector(op == Op::kNoTrans ? m : n, beta, y);
  if (IsZero(alpha)) return;
  ForEachStrip(n, true, [&](size_t width, size_t j) HWY_ATTR {
    if (width == kBandStrip) {
      BandGemvStrip<kBandStrip>(layout, op, j, alpha, a, x, y);
    } else {
      BandGemvStrip<1>(layout, op, j, alpha, a, x, y);
    }
  });
}

// ---------------------------------------------------------------------------
// The diagonal block of a strip [j, j + kCols) of a square matrix: calls
// f(i, jj, a_ij) for the stored elements (i, jj) of the `lower` or upper
// triangle with i, jj in the strip.

template <size_t kCols, typename T, class Func>
HWY_INLINE void ForEachInBlock(const ColumnLayout::Column* col, bool lower,
                               size_t j, T* a, const Func& f) {
  for (size_t c = 0; c < kCols; ++c) {
    const size_t jj = j + c;
    const size_t first = lower ? jj : HWY_MAX(j, col[c].begin);
    const size_t last = lower ? HWY_MIN(j + kCols, col[c].end) : jj + 1;
    for (size_t i = first; i < last; ++i) f(i, jj, Element(a, col[c], i));
  }
}

// ---------------------------------------------------------------------------
// SBMV, HBMV, SPMV, HPMV: y := alpha * A * x + beta * y for the n x n
// symmetric or Hermitian A whose `lower` or upper triangle the layout
// describes. Each stored off-diagonal element A(i, j) also stands for A(j, i)
// (conjugated if Hermitian).

template <size_t kCols, bool kHermitian, typename T>
HWY_INLINE void SymmetricStrip(const ColumnLayout& layout, bool lower, size_t j,
                               T alpha, const T* a, const T* x, T* y) {
  ColumnLayout::Column col[kCols];
  const StripRows<kCols> rows =
      OffStripRows<kCols>(layout, false, lower, j, col);
  ForEachInBlock<kCols>(
      col, lower, j, a, [&](size_t i, size_t jj, const T* p) HWY_ATTR {
        if (i == jj) {
          y[i] += alpha * (kHermitian ? T(std::real(*p)) : *p) * x[i];
        } else {
          y[i] += alpha * *p * x[jj];
          y[jj] += alpha * (kHermitian ? Conj(*p) : *p) * x[i];
        }
      });

  const T* ptrs[kCols];
  RectangleColumns(a, col, rows, ptrs);
  T t[kCols], dots[kCols];
  for (size_t c = 0; c < kCols; ++c) t[c] = alpha * x[j + c];
  const size_t len = rows.end - rows.begin;
  if (len != 0) {
    AxpyDotCols<kCols, kHermitian>(len, ptrs, t, x + rows.begin, y + rows.begin,
                                   dots);
    for (size_t c = 0; c < kCols; ++c) y[j + c] += alpha * dots[c];
  }
  ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
    const T v = *Element(a, col[c], i);
    y[i] += v * t[c];
    y[j + c] += alpha * (kHermitian ? Conj(v) : v) * x[i];
  });
}

template <bool kHermitian, typename T>
HWY_INLINE void SymmetricByColumn(const ColumnLayout& layout, bool lower,
                                  size_t n, T alpha, const T* a, const T* x,
                                  T* y) {
  for (size_t j = 0; j < n; ++j) {
    const ColumnLayout::Column col = layout.At(j);
    T diag = *Element(a, col, j);
    if (kHermitian) diag = T(std::real(diag));
    size_t begin, end;
    OffDiagonalRows(col, lower, j, &begin, &end);
    const T dot = ColumnAxpyDot<kHermitian>(end - begin, Element(a, col, begin),
                                            alpha * x[j], x + begin, y + begin);
    y[j] += alpha * (diag * x[j] + dot);
  }
}

template <bool kHermitian, typename T>
void SymmetricColumnsMv(const ColumnLayout& layout, bool lower, size_t n,
                        T alpha, const T* a, const T* x, T beta, T* y) {
  ScaleVector(n, beta, y);
  if (IsZero(alpha)) return;
  if (!UseStrips(layout)) {
    return SymmetricByColumn<kHermitian>(layout, lower, n, alpha, a, x, y);
  }
  ForEachStrip(n, true, [&](size_t width, size_t j) HWY_ATTR {
    if (width == kBandStrip) {
      SymmetricStrip<kBandStrip, kHermitian>(layout, lower, j, alpha, a, x, y);
    } else {
      SymmetricStrip<1, kHermitian>(layout, lower, j, alpha, a, x, y);
    }
  });
}

// ---------------------------------------------------------------------------
// TBMV, TPMV: x := op(A) * x for the n x n triangular A whose `lower` or
// upper triangle the layout describes. The product is accumulated in a
// zeroed temporary y and copied back.

template <size_t kCols, typename T>
HWY_INLINE void TriangularMvStrip(const ColumnLayout& layout, bool lower, Op op,
                                  bool unit, size_t j, const T* a, const T* x,
                                  T* y) {
  ColumnLayout::Column col[kCols];
  const StripRows<kCols> rows =
      OffStripRows<kCols>(layout, false, lower, j, col);
  const bool conj = (op == Op::kConjTrans);
  ForEachInBlock<kCols>(col, lower, j, a,
                        [&](size_t i, size_t jj, const T* p) HWY_ATTR {
                          const T v = (i == jj && unit) ? T(1) : *p;
                          if (op == Op::kNoTrans) {
                            y[i] += v * x[jj];
                          } else {
                            y[jj] += (conj ? Conj(v) : v) * x[i];
                          }
                        });

  const T* ptrs[kCols];
  RectangleColumns(a, col, rows, ptrs);
  const size_t len = rows.end - rows.begin;
  if (op == Op::kNoTrans) {
    AxpyCols<kCols>(len, ptrs, x + j, y + rows.begin);
    ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
      y[i] += *Element(a, col[c], i) * x[j + c];
    });
  } else {
    if (conj) {
      DotCols<kCols, true>(len, ptrs, T(1), x + rows.begin, y + j);
    } else {
      DotCols<kCols, false>(len, ptrs, T(1), x + rows.begin, y + j);
    }
    ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
      const T v = *Element(a, col[c], i);
      y[j + c] += (conj ? Conj(v) : v) * x[i];
    });
  }
}

template <typename T>
HWY_INLINE void TriangularMvByColumn(const ColumnLayout& layout, bool lower,
                                     Op op, bool unit, size_t n, const T* a,
                                     const T* x, T* y) {
  const bool conj = (op == Op::kConjTrans);
  for (size_t j = 0; j < n; ++j) {
    const ColumnLayout::Column col = layout.At(j);
    T diag = unit ? T(1) : *Element(a, col, j);
    if (conj) diag = Conj(diag);
    size_t begin, end;
    OffDiagonalRows(col, lower, j, &begin, &end);
    const T* off = Element(a, col, begin);
    if (op == Op::kNoTrans) {
      ColumnAxpy(end - begin, x[j], off, y + begin);
      y[j] += diag * x[j];
    } else {
      y[j] += diag * x[j] + ColumnDot(conj, end - begin, off, x + begin);
    }
  }
}

template <typename T>
void TriangularColumnsMv(const ColumnLayout& layout, bool lower, Op op,
                         bool unit, size_t n, const T* a, T* HWY_RESTRICT x) {
  T* HWY_RESTRICT y =
      static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
  for (size_t i = 0; i < n; ++i) y[i] = T(0);
  if (!UseStrips(layout)) {
    TriangularMvByColumn(layout, lower, op, unit, n, a, x, y);
    memcpy(x, y, n * sizeof(T));
    return;
  }
  ForEachStrip(n, true, [&](size_t width, size_t j) HWY_ATTR {
    if (width == kBandStrip) {
      TriangularMvStrip<kBandStrip>(layout, lower, op, unit, j, a, x, y);
    } else {
      TriangularMvStrip<1>(layout, lower, op, unit, j, a, x, y);
    }
  });
  memcpy(x, y, n * sizeof(T));
}

// ---------------------------------------------------------------------------
// TBSV, TPSV: solves op(A) * x = b (x holds b on entry) for the n x n
// triangular A whose `lower` or upper triangle the layout describes.
//
// Strips are visited in the order in which the unknowns are solved:
// increasing if op(A) is lower triangular, decreasing otherwise.
//   - op = N (axpy form): after solving the strip's unknowns, subtract their
//     columns times them from the unknowns not yet solved.
//   - op = T, C (dot form): before solving the strip's unknowns, subtract the
//     dot products of their columns with the unknowns already solved.

template <size_t kCols, typename T>
HWY_INLINE void TriangularSvStrip(const ColumnLayout& layout, bool lower, Op op,
                                  bool unit, size_t j, const T* a, T* x) {
  ColumnLayout::Column col[kCols];
  const StripRows<kCols> rows =
      OffStripRows<kCols>(layout, false, lower, j, col);
  const bool conj = (op == Op::kConjTrans);
  const bool forward = (lower == (op == Op::kNoTrans));
  const T* ptrs[kCols];
  RectangleColumns(a, col, rows, ptrs);
  const size_t len = rows.end - rows.begin;

  if (op != Op::kNoTrans) {
    if (conj) {
      DotCols<kCols, true>(len, ptrs, T(-1), x + rows.begin, x + j);
    } else {
      DotCols<kCols, false>(len, ptrs, T(-1), x + rows.begin, x + j);
    }
    ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
      const T v = *Element(a, col[c], i);
      x[j + c] -= (conj ? Conj(v) : v) * x[i];
    });
  }

  // Diagonal block: substitution in solve order. op(A)(p, q) for p, q in the
  // strip is A(p, q) (op = N) or op(A(q, p)), stored if within the band.
  const auto op_a = [&](size_t p, size_t q) HWY_ATTR {
    const bool n_op = (op == Op::kNoTrans);
    const size_t row = n_op ? p : q, c = (n_op ? q : p) - j;
    if (row < col[c].begin || row >= col[c].end) return T(0);
    const T v = *Element(a, col[c], row);
    return conj ? Conj(v) : v;
  };
  for (size_t step = 0; step < kCols; ++step) {
    const size_t p = forward ? j + step : j + kCols - 1 - step;
    T sum = x[p];
    if (forward) {
      for (size_t q = j; q < p; ++q) sum -= op_a(p, q) * x[q];
    } else {
      for (size_t q = p + 1; q < j + kCols; ++q) sum -= op_a(p, q) * x[q];
    }
    x[p] = unit ? sum : sum / op_a(p, p);
  }

  if (op == Op::kNoTrans) {
    T t[kCols];
    for (size_t c = 0; c < kCols; ++c) t[c] = -x[j + c];
    AxpyCols<kCols>(len, ptrs, t, x + rows.begin);
    ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
      x[i] += *Element(a, col[c], i) * t[c];
    });
  }
}

template <typename T>
HWY_INLINE void TriangularSvByColumn(const ColumnLayout& layout, bool lower,
                                     Op op, bool unit, size_t n, const T* a,
                                     T* x) {
  const bool conj = (op == Op::kConjTrans);
  const bool forward = (lower == (op == Op::kNoTrans));
  for (size_t step = 0; step < n; ++step) {
    const size_t j = forward ? step : n - 1 - step;
    const ColumnLayout::Column col = layout.At(j);
    T diag = unit ? T(1) : *Element(a, col, j);
    if (conj) diag = Conj(diag);
    size_t begin, end;
    OffDiagonalRows(col, lower, j, &begin, &end);
    const T* off = Element(a, col, begin);
    if (op == Op::kNoTrans) {
      if (!unit) x[j] /= diag;
      ColumnAxpy(end - begin, -x[j], off, x + begin);
    } else {
      const T v = x[j] - ColumnDot(conj, end - begin, off, x + begin);
      x[j] = unit ? v : v / diag;
    }
  }
}

template <typename T>
void TriangularColumnsSv(const ColumnLayout& layout, bool lower, Op op,
                         bool unit, size_t n, const T* a, T* HWY_RESTRICT x) {
  if (!UseStrips(layout)) {
    return TriangularSvByColumn(layout, lower, op, unit, n, a, x);
  }
  const bool forward = (lower == (op == Op::kNoTrans));
  ForEachStrip(n, forward, [&](size_t width, size_t j) HWY_ATTR {
    if (width == kBandStrip) {
      TriangularSvStrip<kBandStrip>(layout, lower, op, unit, j, a, x);
    } else {
      TriangularSvStrip<1>(layout, lower, op, unit, j, a, x);
    }
  });
}

// ---------------------------------------------------------------------------
// SPR, SPR2, HPR, HPR2: A += u0 * s0^T (+ u1 * s1^T) for the packed n x n
// matrix whose `lower` or upper triangle the layout describes; s0, s1
// include alpha and any conjugation (see rank_update-inl.h). For Hermitian
// A, the imaginary parts of the diagonal are set to zero.

template <size_t kCols, size_t kTerms, typename T>
HWY_INLINE void RankUpdateStrip(const ColumnLayout& layout, bool lower,
                                bool hermitian, size_t j, T* a, const T* u0,
                                const T* s0, const T* u1, const T* s1) {
  ColumnLayout::Column col[kCols];
  const StripRows<kCols> rows =
      OffStripRows<kCols>(layout, false, lower, j, col);
  ForEachInBlock<kCols>(col, lower, j, a,
                        [&](size_t i, size_t jj, T* p) HWY_ATTR {
                          T v = u0[i] * s0[jj];
                          if (kTerms == 2) v += u1[i] * s1[jj];
                          if constexpr (IsComplex<T>::value) {
                            if (hermitian && i == jj) {
                              *p = T(p->real() + v.real(), 0);
                              return;
                            }
                          }
                          *p += v;
                        });

  T* ptrs[kCols];
  RectangleColumns(a, col, rows, ptrs);
  const size_t len = rows.end - rows.begin;
  if (len != 0) {
    RankUpdateCols<kCols, kTerms>(len, ptrs, u0 + rows.begin, s0 + j,
                                  u1 + rows.begin, s1 + j);
  }
  ForEachRagged(rows, [&](size_t c, size_t i) HWY_ATTR {
    T v = u0[i] * s0[j + c];
    if (kTerms == 2) v += u1[i] * s1[j + c];
    *Element(a, col[c], i) += v;
  });
}

template <size_t kTerms, typename T>
void PackedRankUpdate(const ColumnLayout& layout, bool lower, bool hermitian,
                      size_t n, T* a, const T* u0, const T* s0, const T* u1,
                      const T* s1) {
  ForEachStrip(n, true, [&](size_t width, size_t j) HWY_ATTR {
    if (width == kBandStrip) {
      RankUpdateStrip<kBandStrip, kTerms>(layout, lower, hermitian, j, a, u0,
                                          s0, u1, s1);
    } else {
      RankUpdateStrip<1, kTerms>(layout, lower, hermitian, j, a, u0, s0, u1,
                                 s1);
    }
  });
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_BAND_PACKED_INL_H_
