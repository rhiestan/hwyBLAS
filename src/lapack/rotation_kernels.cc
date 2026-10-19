// Kernel for the sequences of plane rotations of xLASR: see
// rotation_kernels.h.
//
// A sweep of rotations in the planes (k, k+1) is a recurrence along each
// row (side 'R') or column (side 'L') of A: with x the current value of
// element k, as updated by rotation k-1, rotation k sets element k to its
// final value from x and element k+1, which becomes the new x. LAPACK
// applies each rotation to two whole columns (or rows) in turn; here the
// recurrence is carried in registers instead, for SIMD vectors of
// independent rows or columns, so that each element of A is loaded and
// stored once per sweep.
//
// For side 'R', the rows of a column are contiguous, and a vector holds
// consecutive rows. For side 'L', a vector holds an element of consecutive
// columns: tiles of K rows of K columns are transposed in registers after
// loading and before storing.
//
// The operations are those of LAPACK, as separate multiplications and
// additions (not FMA), in the same order for each element. The rotations
// are real; for complex T, the real and imaginary parts are independent
// recurrences with the same operations.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/lapack/rotation_kernels.cc"
#include "src/lapack/rotation_kernels.h"

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// A rotation (c, s), broadcast to the lanes of D.
template <class D>
struct Rotation {
  using V = hn::Vec<D>;
  Rotation(D d, hn::TFromD<D> cs, hn::TFromD<D> sn)
      : identity(cs == 1 && sn == 0), c(hn::Set(d, cs)), s(hn::Set(d, sn)) {}
  bool identity;
  V c, s;
};

// One step of the recurrence: rotation r of the element t, given the
// current value x of the element before it (kForward) or after it.
// Returns the final value of the element before (after) t and updates x to
// the new value of t. As LAPACK, skips identity rotations (which would turn
// infinities into NaN).
template <bool kForward, class D, class V = hn::Vec<D>>
HWY_INLINE V RotateStep(const Rotation<D>& r, V t, V& x) {
  if (r.identity) {
    const V out = x;
    x = t;
    return out;
  }
  if constexpr (kForward) {
    // A(k) := s*A(k+1) + c*A(k), A(k+1) := c*A(k+1) - s*A(k).
    const V out = hn::Add(hn::Mul(r.s, t), hn::Mul(r.c, x));
    x = hn::Sub(hn::Mul(r.c, t), hn::Mul(r.s, x));
    return out;
  } else {
    // A(k+1) := c*A(k+1) - s*A(k), A(k) := s*A(k+1) + c*A(k), with A(k+1)
    // the current value x and A(k) = t.
    const V out = hn::Sub(hn::Mul(r.c, x), hn::Mul(r.s, t));
    x = hn::Add(hn::Mul(r.s, x), hn::Mul(r.c, t));
    return out;
  }
}

// Side 'R': rotates the columns 0:n-1 (stride lda) of the kVecs * Lanes(d)
// consecutive rows at a.
template <bool kForward, size_t kVecs, class D, typename R = hn::TFromD<D>>
HWY_INLINE void RotateColumns(D d, size_t n, const R* HWY_RESTRICT c,
                              const R* HWY_RESTRICT s, R* HWY_RESTRICT a,
                              size_t lda) {
  using V = hn::Vec<D>;
  const size_t N = hn::Lanes(d);
  V x[kVecs];
  const size_t first = kForward ? 0 : n - 1;
  for (size_t v = 0; v < kVecs; ++v) {
    x[v] = hn::LoadU(d, a + first * lda + v * N);
  }
  for (size_t k = 0; k + 1 < n; ++k) {
    // Rotation j, in the plane of the columns j and j+1; t is the column
    // that it brings into the recurrence, out the one that it completes.
    const size_t j = kForward ? k : n - 2 - k;
    const R* t = a + (kForward ? j + 1 : j) * lda;
    R* out = a + (kForward ? j : j + 1) * lda;
    const Rotation<D> r(d, c[j], s[j]);
    for (size_t v = 0; v < kVecs; ++v) {
      const V tv = hn::LoadU(d, t + v * N);
      hn::StoreU(RotateStep<kForward>(r, tv, x[v]), d, out + v * N);
    }
  }
  const size_t last = kForward ? n - 1 : 0;
  for (size_t v = 0; v < kVecs; ++v) {
    hn::StoreU(x[v], d, a + last * lda + v * N);
  }
}

// Side 'R' for the len rows (of R elements) at a.
template <bool kForward, typename R>
HWY_INLINE void RotateColumnsAll(size_t len, size_t n, const R* HWY_RESTRICT c,
                                 const R* HWY_RESTRICT s, R* HWY_RESTRICT a,
                                 size_t lda) {
  const hn::ScalableTag<R> d;
  const size_t N = hn::Lanes(d);
  constexpr size_t kVecs = 4;
  size_t i = 0;
  for (; i + kVecs * N <= len; i += kVecs * N) {
    RotateColumns<kForward, kVecs>(d, n, c, s, a + i, lda);
  }
  for (; i + N <= len; i += N) {
    RotateColumns<kForward, 1>(d, n, c, s, a + i, lda);
  }
  // The remaining rows (fewer than N), with single-lane vectors.
  const hn::CappedTag<R, 1> d1;
  for (; i + 4 <= len; i += 4) {
    RotateColumns<kForward, 4>(d1, n, c, s, a + i, lda);
  }
  for (; i < len; ++i) RotateColumns<kForward, 1>(d1, n, c, s, a + i, lda);
}

// Side 'L' for one column of len elements (real and imaginary parts of
// kParts = 2 interleaved for complex T), from the state x0 (one value per
// part): forward, the elements from `from` on, with the rotations from
// rotation from / kParts - 1 on; backward, the elements before `from`,
// with the rotations from rotation from / kParts - 1 down to 0. Stores the
// final state.
template <bool kForward, size_t kParts, typename R>
HWY_INLINE void RotateColumnTail(size_t len, size_t from,
                                 const R* HWY_RESTRICT c,
                                 const R* HWY_RESTRICT s, R* HWY_RESTRICT col,
                                 const R* x0) {
  const hn::CappedTag<R, 1> d1;
  using D1 = decltype(d1);
  using V1 = hn::Vec<D1>;
  V1 x[kParts];
  for (size_t p = 0; p < kParts; ++p) x[p] = hn::Set(d1, x0[p]);
  if constexpr (kForward) {
    for (size_t i = from; i < len; i += kParts) {
      const Rotation<D1> r(d1, c[i / kParts - 1], s[i / kParts - 1]);
      for (size_t p = 0; p < kParts; ++p) {
        const V1 t = hn::LoadU(d1, col + i + p);
        hn::StoreU(RotateStep<true>(r, t, x[p]), d1, col + i + p - kParts);
      }
    }
    for (size_t p = 0; p < kParts; ++p) {
      hn::StoreU(x[p], d1, col + len - kParts + p);
    }
  } else {
    for (size_t i = from; i != 0;) {
      i -= kParts;
      const Rotation<D1> r(d1, c[i / kParts], s[i / kParts]);
      for (size_t p = 0; p < kParts; ++p) {
        const V1 t = hn::LoadU(d1, col + i + p);
        hn::StoreU(RotateStep<false>(r, t, x[p]), d1, col + i + p + kParts);
      }
    }
    for (size_t p = 0; p < kParts; ++p) hn::StoreU(x[p], d1, col + p);
  }
}

#if HWY_TARGET != HWY_SCALAR

// Transposes the K x K tile v[0:K] of a vector type with K lanes, K = 2 or
// 4. A vector of 4 doubles has two 128-bit blocks, within which Interleave*
// operate; a vector of 4 floats (or of 2 elements) has one.
template <size_t K, class D, class V = hn::Vec<D>>
HWY_INLINE void TransposeTile(D d, V* v) {
  if constexpr (K == 2) {
    const V t0 = hn::InterleaveLower(d, v[0], v[1]);
    v[1] = hn::InterleaveUpper(d, v[0], v[1]);
    v[0] = t0;
  } else if constexpr (sizeof(hn::TFromD<D>) == 8) {
    const V t0 = hn::InterleaveLower(d, v[0], v[1]);  // 00 10 | 02 12
    const V t1 = hn::InterleaveUpper(d, v[0], v[1]);  // 01 11 | 03 13
    const V t2 = hn::InterleaveLower(d, v[2], v[3]);  // 20 30 | 22 32
    const V t3 = hn::InterleaveUpper(d, v[2], v[3]);  // 21 31 | 23 33
    v[0] = hn::ConcatLowerLower(d, t2, t0);
    v[1] = hn::ConcatLowerLower(d, t3, t1);
    v[2] = hn::ConcatUpperUpper(d, t2, t0);
    v[3] = hn::ConcatUpperUpper(d, t3, t1);
  } else {
    const hn::Repartition<uint64_t, D> d64;
    const V t0 = hn::InterleaveLower(d, v[0], v[1]);  // 00 10 01 11
    const V t1 = hn::InterleaveUpper(d, v[0], v[1]);  // 02 12 03 13
    const V t2 = hn::InterleaveLower(d, v[2], v[3]);  // 20 30 21 31
    const V t3 = hn::InterleaveUpper(d, v[2], v[3]);  // 22 32 23 33
    const auto pairs = [d, d64](V lo, V hi, bool upper) HWY_ATTR {
      const auto l = hn::BitCast(d64, lo), h = hn::BitCast(d64, hi);
      return hn::BitCast(d, upper ? hn::InterleaveUpper(d64, l, h)
                                  : hn::InterleaveLower(d64, l, h));
    };
    v[0] = pairs(t0, t2, false);
    v[1] = pairs(t0, t2, true);
    v[2] = pairs(t1, t3, false);
    v[3] = pairs(t1, t3, true);
  }
}

// Side 'L' for the kGroups * K columns (stride lda) of len elements at a,
// with d a tag of K lanes. Tiles of K elements of K columns are rotated as
// K vectors of one element of each of the K columns. Each group of K
// columns is an independent recurrence; several hide the latency of one.
template <bool kForward, size_t kGroups, size_t kParts, class D,
          typename R = hn::TFromD<D>>
HWY_INLINE void RotateRows(D d, size_t len, const R* HWY_RESTRICT c,
                           const R* HWY_RESTRICT s, R* HWY_RESTRICT a,
                           size_t lda) {
  using V = hn::Vec<D>;
  constexpr size_t K = hn::MaxLanes(D());
  static_assert(K % kParts == 0);
  const auto column = [=](size_t g, size_t k) {
    return a + (g * K + k) * lda;
  };
  // The state of each recurrence: x[g][p] holds part p of the current
  // element of the columns of group g.
  V x[kGroups][kParts];
  HWY_ALIGN R lanes[K];
  const size_t first = kForward ? 0 : len - kParts;
  for (size_t g = 0; g < kGroups; ++g) {
    for (size_t p = 0; p < kParts; ++p) {
      for (size_t k = 0; k < K; ++k) lanes[k] = column(g, k)[first + p];
      x[g][p] = hn::Load(d, lanes);
    }
  }
  // Each tile holds the elements i:i+K, those that K / kParts rotations
  // bring into the recurrence; it is stored kParts elements before
  // (forward) or after them, with the elements that they complete.
  size_t i = kForward ? kParts : len - kParts;
  while (kForward ? i + K <= len : i >= K) {
    if constexpr (!kForward) i -= K;
    V tile[kGroups][K];
    for (size_t g = 0; g < kGroups; ++g) {
      for (size_t k = 0; k < K; ++k) {
        tile[g][k] = hn::LoadU(d, column(g, k) + i);
      }
      TransposeTile<K>(d, tile[g]);
    }
    for (size_t q = 0; q < K / kParts; ++q) {
      const size_t e = (kForward ? q : K / kParts - 1 - q) * kParts;
      const size_t j = (i + e) / kParts - (kForward ? 1 : 0);
      const Rotation<D> r(d, c[j], s[j]);
      for (size_t g = 0; g < kGroups; ++g) {
        for (size_t p = 0; p < kParts; ++p) {
          tile[g][e + p] = RotateStep<kForward>(r, tile[g][e + p], x[g][p]);
        }
      }
    }
    const size_t to = kForward ? i - kParts : i + kParts;
    for (size_t g = 0; g < kGroups; ++g) {
      TransposeTile<K>(d, tile[g]);
      for (size_t k = 0; k < K; ++k) {
        hn::StoreU(tile[g][k], d, column(g, k) + to);
      }
    }
    if constexpr (kForward) i += K;
  }
  // The remaining elements (fewer than K), one column at a time.
  HWY_ALIGN R state[kParts][K];
  for (size_t g = 0; g < kGroups; ++g) {
    for (size_t p = 0; p < kParts; ++p) hn::Store(x[g][p], d, state[p]);
    for (size_t k = 0; k < K; ++k) {
      R x0[kParts];
      for (size_t p = 0; p < kParts; ++p) x0[p] = state[p][k];
      RotateColumnTail<kForward, kParts>(len, i, c, s, column(g, k), x0);
    }
  }
}

#endif  // HWY_TARGET != HWY_SCALAR

#ifndef HWYBLAS_LASR_GROUPS
#define HWYBLAS_LASR_GROUPS 2
#endif

// Side 'L' for the n columns of len elements at a.
template <bool kForward, size_t kParts, typename R>
HWY_INLINE void RotateRowsAll(size_t len, size_t n, const R* HWY_RESTRICT c,
                              const R* HWY_RESTRICT s, R* HWY_RESTRICT a,
                              size_t lda) {
  size_t i = 0;
#if HWY_TARGET != HWY_SCALAR
  // Tiles of 4 x 4 if a vector holds 4 elements, then 2 x 2.
  const auto groups = [&](auto d) HWY_ATTR {
    constexpr size_t K = hn::MaxLanes(decltype(d)());
    if constexpr (K % kParts == 0) {
      if (hn::Lanes(d) != K) return;
      constexpr size_t kGroups = HWYBLAS_LASR_GROUPS;
      for (; i + kGroups * K <= n; i += kGroups * K) {
        RotateRows<kForward, kGroups, kParts>(d, len, c, s, a + i * lda, lda);
      }
      for (; i + K <= n; i += K) {
        RotateRows<kForward, 1, kParts>(d, len, c, s, a + i * lda, lda);
      }
    }
  };
  const hn::CappedTag<R, 4> d4;
  const hn::CappedTag<R, 2> d2;
  if constexpr (hn::MaxLanes(decltype(d4)()) == 4) groups(d4);
  if constexpr (hn::MaxLanes(decltype(d2)()) == 2) groups(d2);
#endif
  // The remaining columns, one at a time.
  for (; i < n; ++i) {
    R* HWY_RESTRICT col = a + i * lda;
    R x0[kParts];
    const size_t first = kForward ? 0 : len - kParts;
    for (size_t p = 0; p < kParts; ++p) x0[p] = col[first + p];
    RotateColumnTail<kForward, kParts>(len, kForward ? kParts : len - kParts,
                                       c, s, col, x0);
  }
}

template <bool kForward, typename T, typename R = RealT<T>>
void LasrVariableImpl(bool left, size_t m, size_t n, const R* HWY_RESTRICT c,
                      const R* HWY_RESTRICT s, T* HWY_RESTRICT a, size_t lda) {
  R* HWY_RESTRICT ar = reinterpret_cast<R*>(a);
  if (left) {
    if (m < 2) return;
    constexpr size_t kParts = RealCount<T>(1);
    RotateRowsAll<kForward, kParts>(RealCount<T>(m), n, c, s, ar,
                                    RealCount<T>(lda));
  } else {
    if (n < 2) return;
    RotateColumnsAll<kForward>(RealCount<T>(m), n, c, s, ar,
                               RealCount<T>(lda));
  }
}

#define HWYBLAS_ROTATION_KERNELS(SUFFIX, T)                                  \
  void LasrVariable##SUFFIX(bool left, bool forward, size_t m, size_t n,     \
                            const RealT<T>* c, const RealT<T>* s, T* a,      \
                            size_t lda) {                                    \
    if (forward) {                                                           \
      LasrVariableImpl<true>(left, m, n, c, s, a, lda);                      \
    } else {                                                                 \
      LasrVariableImpl<false>(left, m, n, c, s, a, lda);                     \
    }                                                                        \
  }
HWYBLAS_ROTATION_KERNELS(F32, float)
HWYBLAS_ROTATION_KERNELS(F64, double)
HWYBLAS_ROTATION_KERNELS(C32, c32)
HWYBLAS_ROTATION_KERNELS(C64, c64)
#undef HWYBLAS_ROTATION_KERNELS

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(LasrVariable);

namespace lapack {

template <typename T>
void LasrVariable(bool left, bool forward, blasint m, blasint n,
                  const RealT<T>* c, const RealT<T>* s, T* a, blasint lda) {
  HWYBLAS_DISPATCH_BY_TYPE(T, LasrVariable, left, forward, m, n, c, s, a, lda);
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                   \
  template void LasrVariable<T>(bool, bool, blasint, blasint,           \
                                const RealT<T>*, const RealT<T>*, T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE(float)
HWYBLAS_LAPACK_INSTANTIATE(double)
HWYBLAS_LAPACK_INSTANTIATE(c32)
HWYBLAS_LAPACK_INSTANTIATE(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas
#endif  // HWY_ONCE
