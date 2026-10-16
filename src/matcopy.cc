// Matrix copies with scaling and transposition (OpenBLAS extensions):
//
//   ?OMATCOPY  B := alpha * op(A)   (out of place)
//   ?IMATCOPY  A := alpha * op(A)   (in place; the result may have another
//                                    leading dimension)
//
// op is N (none), T (transpose), R (conjugate, no transpose) or C (conjugate
// transpose); R and C are N and T for real types. The matrices are column-
// or row-major (ORDER = 'C' or 'R'); a row-major matrix is the column-major
// storage of its transpose, so row-major copies are column-major copies with
// rows and cols swapped.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/matcopy.cc"
#include <string.h>

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/level1-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

// Rows and columns of the blocks in which transposed copies are done, so
// that the parts of A and B in use stay in cache. Tuned on Zen 3: the
// vectorized real copies prefer large blocks, the element-wise complex
// copies small ones.
template <typename T>
constexpr size_t TransposeBlock() {
  return IsComplex<T>::value ? 16 : 256;
}

// alpha * v (conjugated if kConj), without the NaN/Inf recovery of
// std::complex multiplication, which slows down the transposed copy.
template <bool kConj, typename T>
HWY_INLINE T ScaleElement(T alpha, T v) {
  if constexpr (IsComplex<T>::value) {
    const auto vr = v.real(), vi = kConj ? -v.imag() : v.imag();
    return T(alpha.real() * vr - alpha.imag() * vi,
             alpha.real() * vi + alpha.imag() * vr);
  } else {
    return alpha * v;
  }
}

// B(j, i) = alpha * op(A(i, j)) for i in [i0, i1), j in [j0, j1), element by
// element.
template <bool kConj, typename T>
HWY_INLINE void TransposeElements(size_t i0, size_t i1, size_t j0, size_t j1,
                                  T alpha, const T* a, size_t lda, T* b,
                                  size_t ldb) {
  for (size_t i = i0; i < i1; ++i) {
    T* HWY_RESTRICT bi = b + i * ldb;  // column i of B
    for (size_t j = j0; j < j1; ++j) {
      bi[j] = ScaleElement<kConj>(alpha, a[i + j * lda]);
    }
  }
}

// As TransposeElements for real T, with vectors: N rows of 4 columns of A
// are loaded as 4 vectors and interleaved, which yields the N rows of A as
// contiguous groups of 4 elements, i.e. 4 elements of N columns of B.
template <typename T>
HWY_INLINE void TransposeVectors(size_t i0, size_t i1, size_t j0, size_t j1,
                                 T alpha, const T* a, size_t lda, T* b,
                                 size_t ldb) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  HWY_ALIGN T rows4[4 * hn::MaxLanes(d)];
  const V va = hn::Set(d, alpha);
  size_t j = j0;
  for (; j + 4 <= j1; j += 4) {
    size_t i = i0;
    for (; i + N <= i1; i += N) {
      const V v0 = hn::Mul(va, hn::LoadU(d, a + i + (j + 0) * lda));
      const V v1 = hn::Mul(va, hn::LoadU(d, a + i + (j + 1) * lda));
      const V v2 = hn::Mul(va, hn::LoadU(d, a + i + (j + 2) * lda));
      const V v3 = hn::Mul(va, hn::LoadU(d, a + i + (j + 3) * lda));
      hn::StoreInterleaved4(v0, v1, v2, v3, d, rows4);
      for (size_t r = 0; r < N; ++r) {
        memcpy(b + j + (i + r) * ldb, rows4 + 4 * r, 4 * sizeof(T));
      }
    }
    TransposeElements<false>(i, i1, j, j + 4, alpha, a, lda, b, ldb);
  }
  TransposeElements<false>(i0, i1, j, j1, alpha, a, lda, b, ldb);
}

// B(j, i) = alpha * op(A(i, j)) for all i < rows, j < cols, block by block.
template <bool kConj, typename T>
void TransposeCopy(size_t rows, size_t cols, T alpha, const T* a, size_t lda,
                   T* b, size_t ldb) {
  for (size_t i0 = 0; i0 < rows; i0 += TransposeBlock<T>()) {
    const size_t i1 = HWY_MIN(rows, i0 + TransposeBlock<T>());
    for (size_t j0 = 0; j0 < cols; j0 += TransposeBlock<T>()) {
      const size_t j1 = HWY_MIN(cols, j0 + TransposeBlock<T>());
      if constexpr (IsComplex<T>::value) {
        TransposeElements<kConj>(i0, i1, j0, j1, alpha, a, lda, b, ldb);
      } else {
        TransposeVectors(i0, i1, j0, j1, alpha, a, lda, b, ldb);
      }
    }
  }
}

// Column-major B := alpha * op(A) for a rows x cols matrix A; op transposes
// if `trans` and conjugates if `conj`. alpha != 0.
template <typename T>
void OmatcopyColMajor(bool trans, bool conj, size_t rows, size_t cols, T alpha,
                      const T* a, size_t lda, T* b, size_t ldb) {
  if (!trans) {
    for (size_t j = 0; j < cols; ++j) {
      if (conj) {
        AxpbyContiguous<true, false>(rows, alpha, a + j * lda, T(0),
                                     b + j * ldb);
      } else {
        AxpbyContiguous<false, false>(rows, alpha, a + j * lda, T(0),
                                      b + j * ldb);
      }
    }
    return;
  }
  if (conj) {
    TransposeCopy<true>(rows, cols, alpha, a, lda, b, ldb);
  } else {
    TransposeCopy<false>(rows, cols, alpha, a, lda, b, ldb);
  }
}

#define HWYBLAS_MATCOPY(SUFFIX, T)                                           \
  void Omatcopy##SUFFIX(bool trans, bool conj, size_t rows, size_t cols,     \
                        T alpha, const T* a, size_t lda, T* b, size_t ldb) { \
    OmatcopyColMajor(trans, conj, rows, cols, alpha, a, lda, b, ldb);        \
  }                                                                          \
  void Scale##SUFFIX(size_t n, T alpha, T* x) { ScalContiguous(n, alpha, x); }
HWYBLAS_MATCOPY(F32, float)
HWYBLAS_MATCOPY(F64, double)
HWYBLAS_MATCOPY(C32, c32)
HWYBLAS_MATCOPY(C64, c64)
#undef HWYBLAS_MATCOPY

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include <string.h>

#include "src/api.h"

namespace hwyblas {
namespace {

HWYBLAS_EXPORT_BY_TYPE(Omatcopy);
HWYBLAS_EXPORT_BY_TYPE(Scale);

// Decoded and checked arguments of ?OMATCOPY and ?IMATCOPY, mapped to a
// column-major copy of a rows x cols matrix.
struct MatcopyArgs {
  bool trans = false, conj = false;
  size_t rows = 0, cols = 0;  // of A, column-major
  size_t lda = 0, ldb = 0;
};

// Returns false after reporting an invalid argument. ldb_position is the
// position of LDB (9 for OMATCOPY, 8 for IMATCOPY).
bool DecodeMatcopy(const char* name, char order, char trans, blasint rows,
                   blasint cols, blasint lda, blasint ldb, int ldb_position,
                   MatcopyArgs* args) {
  order = ToUpper(order);
  trans = ToUpper(trans);
  const bool col_major = (order == 'C');
  blasint info = 0;
  if (order != 'C' && order != 'R') {
    info = 1;
  } else if (trans != 'N' && trans != 'T' && trans != 'R' && trans != 'C') {
    info = 2;
  } else if (rows < 0) {
    info = 3;
  } else if (cols < 0) {
    info = 4;
  } else if (lda < HWY_MAX(1, col_major ? rows : cols)) {
    info = 7;
  } else {
    // B has the dimensions of op(A).
    const bool transposed = (trans == 'T' || trans == 'C');
    const blasint b_rows = (col_major != transposed) ? rows : cols;
    if (ldb < HWY_MAX(1, b_rows)) info = ldb_position;
  }
  if (info != 0) {
    ReportError(name, info);
    return false;
  }
  args->trans = (trans == 'T' || trans == 'C');
  args->conj = (trans == 'R' || trans == 'C');
  // Row-major: the column-major storage of A^T (cols x rows).
  args->rows = static_cast<size_t>(col_major ? rows : cols);
  args->cols = static_cast<size_t>(col_major ? cols : rows);
  args->lda = static_cast<size_t>(lda);
  args->ldb = static_cast<size_t>(ldb);
  return true;
}

// Sets the m x n column-major matrix b to zero.
template <typename T>
void SetZero(size_t m, size_t n, T* b, size_t ldb) {
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) b[i + j * ldb] = T(0);
  }
}

template <typename T>
void Omatcopy(const char* name, const char* order, const char* trans,
              const blasint* rows, const blasint* cols, const T* alpha_ptr,
              const T* a, const blasint* lda, T* b, const blasint* ldb) {
  MatcopyArgs g;
  if (!DecodeMatcopy(name, *order, *trans, *rows, *cols, *lda, *ldb, 9, &g)) {
    return;
  }
  if (g.rows == 0 || g.cols == 0) return;
  const T alpha = *alpha_ptr;
  // B is rows x cols, or cols x rows if transposed.
  const size_t b_rows = g.trans ? g.cols : g.rows;
  const size_t b_cols = g.trans ? g.rows : g.cols;
  if (IsZero(alpha)) return SetZero(b_rows, b_cols, b, g.ldb);
  HWYBLAS_DISPATCH_BY_TYPE(T, Omatcopy, g.trans, g.conj, g.rows, g.cols, alpha,
                           a, g.lda, b, g.ldb);
}

template <typename T>
void Imatcopy(const char* name, const char* order, const char* trans,
              const blasint* rows, const blasint* cols, const T* alpha_ptr,
              T* a, const blasint* lda, const blasint* ldb) {
  MatcopyArgs g;
  if (!DecodeMatcopy(name, *order, *trans, *rows, *cols, *lda, *ldb, 8, &g)) {
    return;
  }
  if (g.rows == 0 || g.cols == 0) return;
  const T alpha = *alpha_ptr;
  const size_t b_rows = g.trans ? g.cols : g.rows;
  const size_t b_cols = g.trans ? g.rows : g.cols;
  if (IsZero(alpha)) return SetZero(b_rows, b_cols, a, g.ldb);

  // Same shape and leading dimension, no conjugation: scale in place.
  if (!g.trans && !g.conj && g.lda == g.ldb) {
    for (size_t j = 0; j < g.cols; ++j) {
      HWYBLAS_DISPATCH_BY_TYPE(T, Scale, g.rows, alpha, a + j * g.lda);
    }
    return;
  }
  // Otherwise via a contiguous temporary holding alpha * op(A).
  T* tmp =
      static_cast<T*>(ScratchBuffer(kScratchTemp, b_rows * b_cols * sizeof(T)));
  HWYBLAS_DISPATCH_BY_TYPE(T, Omatcopy, g.trans, g.conj, g.rows, g.cols, alpha,
                           a, g.lda, tmp, b_rows);
  for (size_t j = 0; j < b_cols; ++j) {
    memcpy(a + j * g.ldb, tmp + j * b_rows, b_rows * sizeof(T));
  }
}

}  // namespace
}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

HWYBLAS_API void somatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const float* alpha, const float* a,
                            const blasint* lda, float* b, const blasint* ldb) {
  hwyblas::Omatcopy("SOMATCOPY", order, trans, rows, cols, alpha, a, lda, b,
                    ldb);
}
HWYBLAS_API void domatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const double* alpha, const double* a,
                            const blasint* lda, double* b, const blasint* ldb) {
  hwyblas::Omatcopy("DOMATCOPY", order, trans, rows, cols, alpha, a, lda, b,
                    ldb);
}
HWYBLAS_API void comatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const float* alpha, const float* a,
                            const blasint* lda, float* b, const blasint* ldb) {
  hwyblas::Omatcopy("COMATCOPY", order, trans, rows, cols, C32(alpha), C32(a),
                    lda, C32(b), ldb);
}
HWYBLAS_API void zomatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const double* alpha, const double* a,
                            const blasint* lda, double* b, const blasint* ldb) {
  hwyblas::Omatcopy("ZOMATCOPY", order, trans, rows, cols, C64(alpha), C64(a),
                    lda, C64(b), ldb);
}

HWYBLAS_API void simatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const float* alpha, float* a, const blasint* lda,
                            const blasint* ldb) {
  hwyblas::Imatcopy("SIMATCOPY", order, trans, rows, cols, alpha, a, lda, ldb);
}
HWYBLAS_API void dimatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const double* alpha, double* a, const blasint* lda,
                            const blasint* ldb) {
  hwyblas::Imatcopy("DIMATCOPY", order, trans, rows, cols, alpha, a, lda, ldb);
}
HWYBLAS_API void cimatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const float* alpha, float* a, const blasint* lda,
                            const blasint* ldb) {
  hwyblas::Imatcopy("CIMATCOPY", order, trans, rows, cols, C32(alpha), C32(a),
                    lda, ldb);
}
HWYBLAS_API void zimatcopy_(const char* order, const char* trans,
                            const blasint* rows, const blasint* cols,
                            const double* alpha, double* a, const blasint* lda,
                            const blasint* ldb) {
  hwyblas::Imatcopy("ZIMATCOPY", order, trans, rows, cols, C64(alpha), C64(a),
                    lda, ldb);
}

}  // extern "C"
#endif  // HWY_ONCE
