// TRSM base kernels: triangular solves with a small triangle (at most
// kTrsmBase rows). The recursive algorithm in trsm.cc reduces every TRSM to
// these kernels plus GEMM updates.
//
// The core kernel solves X T = B (T on the right). Rows of X are independent
// there, so a chunk of rows of B is processed in vector registers: each
// column of X is computed in "dot" form from the columns solved before it,
//
//   x_j = (b_j - sum_q x_q * t(q, j)) / t(j, j),
//
// with q < j for upper and q > j for lower triangles. B's columns are
// contiguous, so all loads and stores are full vectors. For real types two
// columns are computed per pass, sharing the loads of the solved columns.
// T is read in place: the kernel only broadcasts its elements.
//
// The left side (T X = B) is solved through X^T T^T = B^T on a transposed
// copy of a block of columns of B.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_TRSM_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_TRSM_INL_H_
#undef HWYBLAS_SRC_TRSM_INL_H_
#else
#define HWYBLAS_SRC_TRSM_INL_H_
#endif

#include <stddef.h>

#include "common.h"
#include "gemm.h"
#include "hwy/highway.h"
#include "src/level1-inl.h"
#include "src/transpose-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// A small triangle: the off-diagonal elements are read in place (the
// kernels only broadcast them, so any strides will do); inv[j] holds
// 1 / t(j, j) (1 if unit).
template <typename T>
struct SmallTriangle {
  const T* data;
  ptrdiff_t rs, cs;  // element (i, j) is data[i * rs + j * cs]
  size_t r;
  bool lower, unit, conj;
  T inv[kTrsmBase];

  // Element (i, j), not conjugated.
  const T& At(size_t i, size_t j) const {
    return data[static_cast<ptrdiff_t>(i) * rs +
                static_cast<ptrdiff_t>(j) * cs];
  }
};

template <typename T>
void LoadSmallTriangle(const Operand<T>& tri, size_t r, SmallTriangle<T>* s) {
  s->data = tri.data;
  s->rs = tri.rs;
  s->cs = tri.cs;
  s->r = r;
  s->lower = tri.lower;
  s->unit = tri.unit_diag;
  s->conj = tri.conj;
  for (size_t j = 0; j < r; ++j) {
    const T diag = tri.conj ? Conj(s->At(j, j)) : s->At(j, j);
    s->inv[j] = tri.unit_diag ? T(1) : T(1) / diag;
  }
}

// Loads and stores of vector v of a chunk of a column; if !kFull, only
// `count` reals of the chunk are valid (masked).
template <bool kFull, class D>
struct ChunkAccess {
  using R = hn::TFromD<D>;
  using V = hn::Vec<D>;
  D d;
  size_t count;

  HWY_INLINE V Load(const R* p, size_t v) const {
    const size_t N = hn::Lanes(d);
    if (kFull) return hn::LoadU(d, p + v * N);
    const size_t start = HWY_MIN(v * N, count);
    return hn::LoadN(d, p + v * N, HWY_MIN(N, count - start));
  }
  HWY_INLINE void Store(V x, R* p, size_t v) const {
    const size_t N = hn::Lanes(d);
    if (kFull) return hn::StoreU(x, d, p + v * N);
    const size_t start = HWY_MIN(v * N, count);
    hn::StoreN(x, d, p + v * N, HWY_MIN(N, count - start));
  }
};

// Real T: solves X T = B for `kV` vectors starting at row i of every column
// of B (leading dimension ld).
//
// Columns are solved two at a time, j0 and then j1 (in solve order), so that
// each vector of an already solved column x_q, loaded once, updates both:
//
//   b_j0 -= x_q * t(q, j0),  b_j1 -= x_q * t(q, j1)   for solved q
//   x_j0  = b_j0 / t(j0, j0)
//   x_j1  = (b_j1 - x_j0 * t(j0, j1)) / t(j1, j1)
//
// which also gives 2 * kV independent FMA chains.
template <size_t kV, bool kFull, typename T>
HWY_INLINE void SolveRowChunkReal(const SmallTriangle<T>& tri, T* b, size_t ld,
                                  size_t i, size_t count) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const ChunkAccess<kFull, decltype(d)> access{d, count};
  const size_t r = tri.r;
  // Column solved at step s.
  const auto col = [&](size_t s) { return tri.lower ? r - 1 - s : s; };
  const auto finish = [&](V& x, size_t j) HWY_ATTR {
    if (!tri.unit) x = hn::Mul(x, hn::Set(d, tri.inv[j]));
  };

  size_t step = 0;
  for (; step + 2 <= r; step += 2) {
    const size_t j0 = col(step), j1 = col(step + 1);
    T* b0 = b + j0 * ld + i;
    T* b1 = b + j1 * ld + i;
    V acc0[kV], acc1[kV];
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) {
      acc0[v] = access.Load(b0, v);
      acc1[v] = access.Load(b1, v);
    }
    for (size_t s = 0; s < step; ++s) {
      const size_t q = col(s);
      const V t0 = hn::Set(d, tri.At(q, j0));
      const V t1 = hn::Set(d, tri.At(q, j1));
      const T* xq = b + q * ld + i;
      HWY_UNROLL(16)
      for (size_t v = 0; v < kV; ++v) {
        const V x = access.Load(xq, v);
        acc0[v] = hn::NegMulAdd(x, t0, acc0[v]);
        acc1[v] = hn::NegMulAdd(x, t1, acc1[v]);
      }
    }
    const V t01 = hn::Set(d, tri.At(j0, j1));
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) {
      finish(acc0[v], j0);
      access.Store(acc0[v], b0, v);
      acc1[v] = hn::NegMulAdd(acc0[v], t01, acc1[v]);
      finish(acc1[v], j1);
      access.Store(acc1[v], b1, v);
    }
  }
  if (step < r) {  // odd r: the last column
    const size_t j = col(step);
    T* bj = b + j * ld + i;
    V acc[kV];
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) acc[v] = access.Load(bj, v);
    for (size_t s = 0; s < step; ++s) {
      const size_t q = col(s);
      const V t = hn::Set(d, tri.At(q, j));
      const T* xq = b + q * ld + i;
      HWY_UNROLL(16)
      for (size_t v = 0; v < kV; ++v) {
        acc[v] = hn::NegMulAdd(access.Load(xq, v), t, acc[v]);
      }
    }
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) {
      finish(acc[v], j);
      access.Store(acc[v], bj, v);
    }
  }
}

// Complex T: solves X T = B for `kV` vectors of reals starting at offset i
// (in reals) of every column of B (b and ld_reals in reals). Each column j
// is computed in dot form from the columns solved before it.
template <size_t kV, bool kFull, typename T>
HWY_INLINE void SolveRowChunkComplex(const SmallTriangle<T>& tri, RealT<T>* b,
                                     size_t ld_reals, size_t i, size_t count) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const ChunkAccess<kFull, decltype(d)> access{d, count};
  const size_t r = tri.r;

  for (size_t step = 0; step < r; ++step) {
    const size_t j = tri.lower ? r - 1 - step : step;
    R* bj = b + j * ld_reals + i;
    // Already solved columns: q < j (upper) or q > j (lower).
    const size_t q_begin = tri.lower ? j + 1 : 0;
    const size_t q_end = tri.lower ? r : j;

    // Sum of x_q * t(q, j) as products with re(t) and im(t), combined
    // once: x * t = AddSub(x * re(t), Reverse2(x * im(t))).
    V sum_re[kV], sum_im[kV];
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) sum_re[v] = sum_im[v] = hn::Zero(d);
    for (size_t q = q_begin; q < q_end; ++q) {
      const T t = tri.At(q, j);
      const V t_re = hn::Set(d, t.real());
      const V t_im = hn::Set(d, tri.conj ? -t.imag() : t.imag());
      const R* xq = b + q * ld_reals + i;
      HWY_UNROLL(16)
      for (size_t v = 0; v < kV; ++v) {
        const V x = access.Load(xq, v);
        sum_re[v] = hn::MulAdd(x, t_re, sum_re[v]);
        sum_im[v] = hn::MulAdd(x, t_im, sum_im[v]);
      }
    }
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) {
      V x = hn::Sub(access.Load(bj, v),
                    hn::AddSub(sum_re[v], hn::Reverse2(d, sum_im[v])));
      if (!tri.unit) {
        x = MulComplex(d, x, tri.inv[j].real(), tri.inv[j].imag());
      }
      access.Store(x, bj, v);
    }
  }
}

// Solves X T = B for `kV` vectors of reals starting at offset i (in reals) of
// every column of B. If !kFull, only `count` reals are valid (masked).
template <size_t kV, bool kFull, typename T>
HWY_INLINE void SolveRowChunk(const SmallTriangle<T>& tri, RealT<T>* b,
                              size_t ld_reals, size_t i, size_t count) {
  if constexpr (IsComplex<T>::value) {
    SolveRowChunkComplex<kV, kFull>(tri, b, ld_reals, i, count);
  } else {
    SolveRowChunkReal<kV, kFull>(tri, b, ld_reals, i, count);
  }
}

// Solves X T = B in place, where B has `rows` rows and tri.r columns
// (column-major, leading dimension ldb).
template <typename T>
HWY_NOINLINE void TrsmRight(const SmallTriangle<T>& tri, size_t rows, T* b,
                            size_t ldb) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  const size_t N = hn::Lanes(d);
  constexpr size_t kV = 4;  // vectors per chunk: independent accumulators
  constexpr size_t kReals = IsComplex<T>::value ? 2 : 1;
  R* br = reinterpret_cast<R*>(b);
  const size_t len = rows * kReals, ld = ldb * kReals;

  size_t i = 0;
  for (; i + kV * N <= len; i += kV * N) {
    SolveRowChunk<kV, true>(tri, br, ld, i, kV * N);
  }
  // Remaining full vectors in one chunk, then a masked partial vector.
  static_assert(kV == 4, "Update the cases below");
  switch ((len - i) / N) {
    case 3:
      SolveRowChunk<3, true>(tri, br, ld, i, 3 * N);
      i += 3 * N;
      break;
    case 2:
      SolveRowChunk<2, true>(tri, br, ld, i, 2 * N);
      i += 2 * N;
      break;
    case 1:
      SolveRowChunk<1, true>(tri, br, ld, i, N);
      i += N;
      break;
    default:
      break;
  }
  if (i < len) SolveRowChunk<1, false>(tri, br, ld, i, len - i);
}

// Solves T X = B (left) or X T = B (right) in place, where T is the r x r
// triangular operand `tri` (r <= kTrsmBase) and B has `other` columns (left)
// or rows (right), column-major with leading dimension ldb.
template <typename T>
HWY_NOINLINE void TrsmBase(bool left, const Operand<T>& tri, size_t r,
                           size_t other, T* HWY_RESTRICT b, size_t ldb) {
  SmallTriangle<T> small;
  if (!left) {
    LoadSmallTriangle(tri, r, &small);
    // The kernel works on all r columns of a chunk of rows at once. If the
    // columns are a multiple of 1 KiB apart, they map to only a few L1 sets
    // and evict each other; then solve on compact copies of blocks of rows.
    constexpr size_t kRowBlock = 512 / sizeof(T);  // 16 KiB for r = 32
    if (other <= kRowBlock || (ldb * sizeof(T)) % 1024 != 0) {
      TrsmRight(small, other, b, ldb);
      return;
    }
    T* copy = static_cast<T*>(
        ScratchBuffer(kScratchTemp, kRowBlock * kTrsmBase * sizeof(T)));
    for (size_t i0 = 0; i0 < other; i0 += kRowBlock) {
      const size_t rows = HWY_MIN(kRowBlock, other - i0);
      for (size_t j = 0; j < r; ++j) {
        memcpy(copy + j * rows, b + i0 + j * ldb, rows * sizeof(T));
      }
      TrsmRight(small, rows, copy, rows);
      for (size_t j = 0; j < r; ++j) {
        memcpy(b + i0 + j * ldb, copy + j * rows, rows * sizeof(T));
      }
    }
    return;
  }

  // T X = B  <=>  X^T T^T = B^T. Solve on transposed blocks of columns of B
  // (small enough to stay in cache).
  Operand<T> tri_t = tri;
  tri_t.rs = tri.cs;
  tri_t.cs = tri.rs;
  tri_t.lower = !tri.lower;
  LoadSmallTriangle(tri_t, r, &small);

  constexpr size_t kBlock = 64;  // columns of B per block: fits in L1
  T* bt = static_cast<T*>(
      ScratchBuffer(kScratchTemp, kBlock * kTrsmBase * sizeof(T)));
  for (size_t j0 = 0; j0 < other; j0 += kBlock) {
    const size_t cols = HWY_MIN(kBlock, other - j0);
    T* block = b + j0 * ldb;
    // bt (cols x r, leading dimension cols) = B[0:r, j0:j0+cols]^T.
    Transpose(r, cols, block, ldb, bt, cols);
    TrsmRight(small, cols, bt, cols);
    Transpose(cols, r, bt, cols, block, ldb);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_TRSM_INL_H_
