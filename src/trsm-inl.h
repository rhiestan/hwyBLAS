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
// contiguous, so all loads and stores are full vectors.
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

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// A small triangle in dense form: t[i + j * r] holds element (i, j) of the
// triangle (conjugated if requested), inv[j] = 1 / t(j, j) (1 if unit).
template <typename T>
struct SmallTriangle {
  size_t r;
  bool lower, unit;
  T t[kTrsmBase * kTrsmBase];
  T inv[kTrsmBase];
};

template <typename T>
void LoadSmallTriangle(const Operand<T>& tri, size_t r, SmallTriangle<T>* s) {
  s->r = r;
  s->lower = tri.lower;
  s->unit = tri.unit_diag;
  const auto at = [&tri](size_t i, size_t j) {
    const T v = tri.data[static_cast<ptrdiff_t>(i) * tri.rs +
                         static_cast<ptrdiff_t>(j) * tri.cs];
    return tri.conj ? Conj(v) : v;
  };
  for (size_t j = 0; j < r; ++j) {
    for (size_t i = 0; i < r; ++i) {
      if (tri.lower ? (i > j) : (i < j)) s->t[i + j * r] = at(i, j);
    }
    s->inv[j] = tri.unit_diag ? T(1) : T(1) / at(j, j);
  }
}

// Solves X T = B for `kV` vectors of reals starting at offset i (in reals) of
// every column of B. If !kFull, only `count` reals are valid (masked).
template <size_t kV, bool kFull, typename T>
HWY_INLINE void SolveRowChunk(const SmallTriangle<T>& tri, RealT<T>* b,
                              size_t ld_reals, size_t i, size_t count) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const size_t r = tri.r;

  const auto load = [&](const R* p, size_t v) HWY_ATTR {
    if (kFull) return hn::LoadU(d, p + v * N);
    const size_t start = HWY_MIN(v * N, count);
    return hn::LoadN(d, p + v * N, HWY_MIN(N, count - start));
  };
  const auto store = [&](V x, R* p, size_t v) HWY_ATTR {
    if (kFull) return hn::StoreU(x, d, p + v * N);
    const size_t start = HWY_MIN(v * N, count);
    hn::StoreN(x, d, p + v * N, HWY_MIN(N, count - start));
  };

  for (size_t step = 0; step < r; ++step) {
    const size_t j = tri.lower ? r - 1 - step : step;
    R* bj = b + j * ld_reals + i;
    // Already solved columns: q < j (upper) or q > j (lower).
    const size_t q_begin = tri.lower ? j + 1 : 0;
    const size_t q_end = tri.lower ? r : j;

    V acc[kV];
    HWY_UNROLL(16)
    for (size_t v = 0; v < kV; ++v) acc[v] = load(bj, v);

    if constexpr (!IsComplex<T>::value) {
      for (size_t q = q_begin; q < q_end; ++q) {
        const V tq = hn::Set(d, tri.t[q + j * r]);
        const R* xq = b + q * ld_reals + i;
        HWY_UNROLL(16)
        for (size_t v = 0; v < kV; ++v) {
          acc[v] = hn::NegMulAdd(load(xq, v), tq, acc[v]);
        }
      }
      const V inv = hn::Set(d, tri.inv[j]);
      HWY_UNROLL(16)
      for (size_t v = 0; v < kV; ++v) {
        store(tri.unit ? acc[v] : hn::Mul(acc[v], inv), bj, v);
      }
    } else {
      // Sum of x_q * t(q, j) as products with re(t) and im(t), combined
      // once: x * t = AddSub(x * re(t), Reverse2(x * im(t))).
      V sum_re[kV], sum_im[kV];
      HWY_UNROLL(16)
      for (size_t v = 0; v < kV; ++v) sum_re[v] = sum_im[v] = hn::Zero(d);
      for (size_t q = q_begin; q < q_end; ++q) {
        const T t = tri.t[q + j * r];
        const V t_re = hn::Set(d, t.real()), t_im = hn::Set(d, t.imag());
        const R* xq = b + q * ld_reals + i;
        HWY_UNROLL(16)
        for (size_t v = 0; v < kV; ++v) {
          const V x = load(xq, v);
          sum_re[v] = hn::MulAdd(x, t_re, sum_re[v]);
          sum_im[v] = hn::MulAdd(x, t_im, sum_im[v]);
        }
      }
      HWY_UNROLL(16)
      for (size_t v = 0; v < kV; ++v) {
        V x =
            hn::Sub(acc[v], hn::AddSub(sum_re[v], hn::Reverse2(d, sum_im[v])));
        if (!tri.unit) {
          x = MulComplex(d, x, tri.inv[j].real(), tri.inv[j].imag());
        }
        store(x, bj, v);
      }
    }
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
  for (; i < len; i += N) {
    SolveRowChunk<1, false>(tri, br, ld, i, HWY_MIN(N, len - i));
  }
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
    TrsmRight(small, other, b, ldb);
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
    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < r; ++i) bt[j + i * cols] = block[i + j * ldb];
    }
    TrsmRight(small, cols, bt, cols);
    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < r; ++i) block[i + j * ldb] = bt[j + i * cols];
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_TRSM_INL_H_
