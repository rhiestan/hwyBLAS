// Level 1 kernels on contiguous vectors, also used as building blocks by
// level 2 and 3 routines.
//
// Complex numbers are interleaved (re, im); a complex scalar times a vector is
// computed as AddSub(v * re(s), Reverse2(v) * im(s)) (see gemm_kernel-inl.h).
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_LEVEL1_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_LEVEL1_INL_H_
#undef HWYBLAS_SRC_LEVEL1_INL_H_
#else
#define HWYBLAS_SRC_LEVEL1_INL_H_
#endif

#include <stddef.h>

#include <cmath>
#include <complex>

#include "common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Returns v * s for each complex (re, im) lane pair of v.
template <class D, typename R>
HWY_INLINE hn::Vec<D> MulComplex(D d, hn::Vec<D> v, R re, R im) {
  return hn::AddSub(hn::Mul(v, hn::Set(d, re)),
                    hn::Mul(hn::Reverse2(d, v), hn::Set(d, im)));
}

// y[0:n] += alpha * x[0:n].
template <typename T>
HWY_INLINE void AxpyContiguous(size_t n, T alpha, const T* HWY_RESTRICT x,
                               T* HWY_RESTRICT y) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  const size_t N = hn::Lanes(d);
  const R* HWY_RESTRICT xr = reinterpret_cast<const R*>(x);
  R* HWY_RESTRICT yr = reinterpret_cast<R*>(y);
  const size_t len = IsComplex<T>::value ? 2 * n : n;  // in reals

  // Returns alpha * v.
  const auto scale = [&](hn::Vec<decltype(d)> v) HWY_ATTR {
    if constexpr (IsComplex<T>::value) {
      return MulComplex(d, v, alpha.real(), alpha.imag());
    } else {
      return hn::Mul(v, hn::Set(d, alpha));
    }
  };

  size_t i = 0;
  if constexpr (!IsComplex<T>::value) {
    const auto va = hn::Set(d, alpha);
    for (; i + 2 * N <= len; i += 2 * N) {
      const auto y0 =
          hn::MulAdd(va, hn::LoadU(d, xr + i), hn::LoadU(d, yr + i));
      const auto y1 =
          hn::MulAdd(va, hn::LoadU(d, xr + i + N), hn::LoadU(d, yr + i + N));
      hn::StoreU(y0, d, yr + i);
      hn::StoreU(y1, d, yr + i + N);
    }
  }
  for (; i + N <= len; i += N) {
    hn::StoreU(hn::Add(hn::LoadU(d, yr + i), scale(hn::LoadU(d, xr + i))), d,
               yr + i);
  }
  if (i < len) {
    const size_t count = len - i;
    hn::StoreN(hn::Add(hn::LoadN(d, yr + i, count),
                       scale(hn::LoadN(d, xr + i, count))),
               d, yr + i, count);
  }
}

// v * alpha for a real alpha (broadcast in re), or for each complex lane
// pair of v and a complex alpha (broadcasts re and im).
template <bool kComplex, class D>
HWY_INLINE hn::Vec<D> ScaleByBroadcast(D d, hn::Vec<D> v, hn::Vec<D> re,
                                       hn::Vec<D> im) {
  if constexpr (kComplex) {
    return hn::AddSub(hn::Mul(v, re), hn::Mul(hn::Reverse2(d, v), im));
  } else {
    (void)d;
    (void)im;
    return hn::Mul(v, re);
  }
}

// x[0:n] *= alpha.
template <typename T>
HWY_INLINE void ScalContiguous(size_t n, T alpha, T* HWY_RESTRICT x) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  R* HWY_RESTRICT xr = reinterpret_cast<R*>(x);
  const size_t len = RealCount<T>(n);

  // Broadcasts of alpha; plain vectors rather than a lambda capturing alpha
  // by reference, which would prevent keeping them in registers.
  R re, im = 0;
  if constexpr (IsComplex<T>::value) {
    re = alpha.real();
    im = alpha.imag();
  } else {
    re = alpha;
  }
  const V vre = hn::Set(d, re), vim = hn::Set(d, im);
  constexpr bool kComplex = IsComplex<T>::value;

  size_t i = 0;
  for (; i + 4 * N <= len; i += 4 * N) {
    const V v0 = hn::LoadU(d, xr + i), v1 = hn::LoadU(d, xr + i + N);
    const V v2 = hn::LoadU(d, xr + i + 2 * N),
            v3 = hn::LoadU(d, xr + i + 3 * N);
    hn::StoreU(ScaleByBroadcast<kComplex>(d, v0, vre, vim), d, xr + i);
    hn::StoreU(ScaleByBroadcast<kComplex>(d, v1, vre, vim), d, xr + i + N);
    hn::StoreU(ScaleByBroadcast<kComplex>(d, v2, vre, vim), d, xr + i + 2 * N);
    hn::StoreU(ScaleByBroadcast<kComplex>(d, v3, vre, vim), d, xr + i + 3 * N);
  }
  for (; i < len; i += N) {
    const size_t count = HWY_MIN(N, len - i);
    const V v = hn::LoadN(d, xr + i, count);
    hn::StoreN(ScaleByBroadcast<kComplex>(d, v, vre, vim), d, xr + i, count);
  }
}

// y[0:n] := alpha * op(x[0:n]) + beta * y[0:n], where op conjugates if
// kConj (complex T only). If !kReadY, y is not read and beta is ignored
// (y := alpha * op(x), for beta == 0).
template <bool kConj, bool kReadY, typename T>
HWY_INLINE void AxpbyContiguous(size_t n, T alpha, const T* HWY_RESTRICT x,
                                T beta, T* HWY_RESTRICT y) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const R* HWY_RESTRICT xr = reinterpret_cast<const R*>(x);
  R* HWY_RESTRICT yr = reinterpret_cast<R*>(y);
  const size_t len = RealCount<T>(n);
  constexpr bool kComplex = IsComplex<T>::value;

  R a_re, a_im = 0, b_re, b_im = 0;
  if constexpr (kComplex) {
    a_re = alpha.real();
    a_im = alpha.imag();
    b_re = beta.real();
    b_im = beta.imag();
  } else {
    a_re = alpha;
    b_re = beta;
  }
  const V va_re = hn::Set(d, a_re), va_im = hn::Set(d, a_im);
  const V vb_re = hn::Set(d, b_re), vb_im = hn::Set(d, b_im);

  size_t i = 0;
  for (; i < len; i += N) {
    const size_t count = HWY_MIN(N, len - i);
    V xv = (count == N) ? hn::LoadU(d, xr + i) : hn::LoadN(d, xr + i, count);
    if constexpr (kComplex && kConj) {
      xv = hn::OddEven(hn::Neg(xv), xv);  // negate the imaginary parts
    }
    V r = ScaleByBroadcast<kComplex>(d, xv, va_re, va_im);
    if constexpr (kReadY) {
      const V yv =
          (count == N) ? hn::LoadU(d, yr + i) : hn::LoadN(d, yr + i, count);
      r = hn::Add(r, ScaleByBroadcast<kComplex>(d, yv, vb_re, vb_im));
    }
    if (count == N) {
      hn::StoreU(r, d, yr + i);
    } else {
      hn::StoreN(r, d, yr + i, count);
    }
  }
}

// ---------------------------------------------------------------------------
// Reductions. They use four independent accumulators so that the FMA latency
// does not limit throughput, and work on the underlying reals (a complex
// vector of n elements is 2n reals).

// Returns the sum of f(LoadU(x + i)) over all vectors of a contiguous array
// of `len` reals, where f(v, acc) adds the contribution of vector v to acc.
// Lanes past the end are loaded as zero.
template <class D, class Func>
HWY_INLINE hn::Vec<D> SumOverVectors(D d, size_t len,
                                     const hn::TFromD<D>* HWY_RESTRICT x,
                                     const Func& f) {
  using V = hn::Vec<D>;
  const size_t N = hn::Lanes(d);
  V s0 = hn::Zero(d), s1 = hn::Zero(d), s2 = hn::Zero(d), s3 = hn::Zero(d);
  size_t i = 0;
  for (; i + 4 * N <= len; i += 4 * N) {
    s0 = f(hn::LoadU(d, x + i), s0);
    s1 = f(hn::LoadU(d, x + i + N), s1);
    s2 = f(hn::LoadU(d, x + i + 2 * N), s2);
    s3 = f(hn::LoadU(d, x + i + 3 * N), s3);
  }
  for (; i + N <= len; i += N) s0 = f(hn::LoadU(d, x + i), s0);
  if (i < len) s1 = f(hn::LoadN(d, x + i, len - i), s1);
  return hn::Add(hn::Add(s0, s1), hn::Add(s2, s3));
}

// Sum of |x_i| over len contiguous reals (ASUM; for complex vectors this is
// the BLAS definition sum |re| + |im|).
template <typename R>
HWY_INLINE R AbsSum(size_t len, const R* HWY_RESTRICT x) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const V sum = SumOverVectors(
      d, len, x, [](V v, V acc) HWY_ATTR { return hn::Add(hn::Abs(v), acc); });
  return hn::ReduceSum(d, sum);
}

// Sum of x_i^2 over len contiguous reals, accumulated in double. For float
// input this cannot overflow or underflow; for double input the caller
// checks the range (see Nrm2).
template <typename R>
HWY_INLINE double SumSquares(size_t len, const R* HWY_RESTRICT x) {
  const hn::ScalableTag<double> dd;
  using VD = hn::Vec<decltype(dd)>;
  if constexpr (sizeof(R) == 8) {
    const VD sum = SumOverVectors(dd, len, x, [](VD v, VD acc) HWY_ATTR {
      return hn::MulAdd(v, v, acc);
    });
    return hn::ReduceSum(dd, sum);
  } else {
    // Load half-width float vectors and promote them to double.
    const hn::Rebind<float, decltype(dd)> df;
    const size_t N = hn::Lanes(dd);
    VD s0 = hn::Zero(dd), s1 = hn::Zero(dd);
    size_t i = 0;
    for (; i + 2 * N <= len; i += 2 * N) {
      const VD v0 = hn::PromoteTo(dd, hn::LoadU(df, x + i));
      const VD v1 = hn::PromoteTo(dd, hn::LoadU(df, x + i + N));
      s0 = hn::MulAdd(v0, v0, s0);
      s1 = hn::MulAdd(v1, v1, s1);
    }
    for (; i < len; i += N) {
      const VD v = hn::PromoteTo(dd, hn::LoadN(df, x + i, HWY_MIN(N, len - i)));
      s0 = hn::MulAdd(v, v, s0);
    }
    return hn::ReduceSum(dd, hn::Add(s0, s1));
  }
}

// Real dot product sum x_i * y_i over n contiguous elements, accumulated in
// precision A (float or double; double for DSDOT and SDSDOT).
template <typename A, typename R>
HWY_INLINE A DotReal(size_t n, const R* HWY_RESTRICT x,
                     const R* HWY_RESTRICT y) {
  const hn::ScalableTag<A> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  // Loads N elements of R as a vector of A (promoting float to double).
  const auto load = [](const R* p, size_t count) HWY_ATTR {
    const hn::ScalableTag<A> da;
    if constexpr (sizeof(A) == sizeof(R)) {
      return hn::LoadN(da, p, count);
    } else {
      const hn::Rebind<R, decltype(da)> dr;
      return hn::PromoteTo(da, hn::LoadN(dr, p, count));
    }
  };
  const auto load_full = [](const R* p) HWY_ATTR {
    const hn::ScalableTag<A> da;
    if constexpr (sizeof(A) == sizeof(R)) {
      return hn::LoadU(da, p);
    } else {
      const hn::Rebind<R, decltype(da)> dr;
      return hn::PromoteTo(da, hn::LoadU(dr, p));
    }
  };

  V s0 = hn::Zero(d), s1 = hn::Zero(d), s2 = hn::Zero(d), s3 = hn::Zero(d);
  size_t i = 0;
  for (; i + 4 * N <= n; i += 4 * N) {
    s0 = hn::MulAdd(load_full(x + i), load_full(y + i), s0);
    s1 = hn::MulAdd(load_full(x + i + N), load_full(y + i + N), s1);
    s2 = hn::MulAdd(load_full(x + i + 2 * N), load_full(y + i + 2 * N), s2);
    s3 = hn::MulAdd(load_full(x + i + 3 * N), load_full(y + i + 3 * N), s3);
  }
  for (; i < n; i += N) {
    const size_t count = HWY_MIN(N, n - i);
    s0 = hn::MulAdd(load(x + i, count), load(y + i, count), s0);
  }
  return hn::ReduceSum(d, hn::Add(hn::Add(s0, s1), hn::Add(s2, s3)));
}

// Complex dot product sum op(x_i) * y_i over n contiguous complex elements,
// op conjugating if kConj (DOTC) or not (DOTU). With P = x * y and
// Q = x * Reverse2(y) (lane-wise), as in gemv-inl.h:
//   sum x * y       = (sum P_even - sum P_odd, sum Q)
//   sum conj(x) * y = (sum P, sum Q_even - sum Q_odd)
template <bool kConj, typename R>
HWY_INLINE std::complex<R> DotComplex(size_t n, const R* HWY_RESTRICT x,
                                      const R* HWY_RESTRICT y) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const size_t len = 2 * n;
  // Eight independent accumulator chains keep both FMA units busy.
  V p0 = hn::Zero(d), p1 = p0, p2 = p0, p3 = p0;
  V q0 = hn::Zero(d), q1 = q0, q2 = q0, q3 = q0;
  size_t i = 0;
  for (; i + 4 * N <= len; i += 4 * N) {
    const V x0 = hn::LoadU(d, x + i), y0 = hn::LoadU(d, y + i);
    const V x1 = hn::LoadU(d, x + i + N), y1 = hn::LoadU(d, y + i + N);
    const V x2 = hn::LoadU(d, x + i + 2 * N), y2 = hn::LoadU(d, y + i + 2 * N);
    const V x3 = hn::LoadU(d, x + i + 3 * N), y3 = hn::LoadU(d, y + i + 3 * N);
    p0 = hn::MulAdd(x0, y0, p0);
    q0 = hn::MulAdd(x0, hn::Reverse2(d, y0), q0);
    p1 = hn::MulAdd(x1, y1, p1);
    q1 = hn::MulAdd(x1, hn::Reverse2(d, y1), q1);
    p2 = hn::MulAdd(x2, y2, p2);
    q2 = hn::MulAdd(x2, hn::Reverse2(d, y2), q2);
    p3 = hn::MulAdd(x3, y3, p3);
    q3 = hn::MulAdd(x3, hn::Reverse2(d, y3), q3);
  }
  p0 = hn::Add(p0, p2);
  p1 = hn::Add(p1, p3);
  q0 = hn::Add(q0, q2);
  q1 = hn::Add(q1, q3);
  for (; i < len; i += N) {
    const size_t count = HWY_MIN(N, len - i);
    const V xv = hn::LoadN(d, x + i, count), yv = hn::LoadN(d, y + i, count);
    p0 = hn::MulAdd(xv, yv, p0);
    q0 = hn::MulAdd(xv, hn::Reverse2(d, yv), q0);
  }
  const V p = hn::Add(p0, p1), q = hn::Add(q0, q1);
  // +1 in even lanes, -1 in odd lanes.
  const V alternate = hn::OddEven(hn::Set(d, R(-1)), hn::Set(d, R(1)));
  if (kConj) {
    return {hn::ReduceSum(d, p), hn::ReduceSum(d, hn::Mul(q, alternate))};
  }
  return {hn::ReduceSum(d, hn::Mul(p, alternate)), hn::ReduceSum(d, q)};
}

// Index (0-based) of the first element with the largest (kMax) or smallest
// (!kMax) magnitude among n contiguous elements, n >= 1. The magnitude is
// |x| for real and |re| + |im| for complex elements. As in the reference
// BLAS, NaNs never compare larger (or smaller), so they are only returned
// if x[0] is NaN.
//
// Two passes: the first finds the extremal magnitude (four independent
// compare-and-select chains), the second stops at the first element with
// that magnitude. This is faster than tracking indices in a single pass.
template <bool kMax, typename T>
HWY_INLINE size_t ArgAbsExtremum(size_t n, const T* HWY_RESTRICT x_typed) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  constexpr size_t kReals = IsComplex<T>::value ? 2 : 1;
  const R* HWY_RESTRICT x = reinterpret_cast<const R*>(x_typed);
  const size_t len = RealCount<T>(n);

  const R first =
      IsComplex<T>::value ? std::abs(x[0]) + std::abs(x[1]) : std::abs(x[0]);
  if (std::isnan(first)) return 0;

  // Magnitudes of the elements in one vector. For complex elements both
  // lanes of a pair hold |re| + |im|.
  const auto magnitude = [&](V v) HWY_ATTR {
    const V a = hn::Abs(v);
    if constexpr (IsComplex<T>::value) {
      return hn::Add(a, hn::Reverse2(d, a));
    } else {
      return a;
    }
  };
  // Keeps the better of m and best; comparisons with NaN are false, so NaNs
  // are skipped.
  const auto keep = [](V m, V best) HWY_ATTR {
    return hn::IfThenElse(kMax ? hn::Gt(m, best) : hn::Lt(m, best), m, best);
  };

  // Pass 1: the extremal magnitude. Lanes past the end count as `first`.
  V b0 = hn::Set(d, first), b1 = b0, b2 = b0, b3 = b0;
  size_t i = 0;
  for (; i + 4 * N <= len; i += 4 * N) {
    b0 = keep(magnitude(hn::LoadU(d, x + i)), b0);
    b1 = keep(magnitude(hn::LoadU(d, x + i + N)), b1);
    b2 = keep(magnitude(hn::LoadU(d, x + i + 2 * N)), b2);
    b3 = keep(magnitude(hn::LoadU(d, x + i + 3 * N)), b3);
  }
  for (; i < len; i += N) {
    const size_t count = HWY_MIN(N, len - i);
    const V m = hn::IfThenElse(hn::FirstN(d, count),
                               magnitude(hn::LoadN(d, x + i, count)),
                               hn::Set(d, first));
    b0 = keep(m, b0);
  }
  const V best = keep(keep(b1, b0), keep(b3, b2));
  const V target =
      hn::Set(d, kMax ? hn::ReduceMax(d, best) : hn::ReduceMin(d, best));

  // Pass 2: the first element with that magnitude; whole blocks of four
  // vectors are tested at once.
  for (i = 0; i + 4 * N <= len; i += 4 * N) {
    const auto e0 = hn::Eq(magnitude(hn::LoadU(d, x + i)), target);
    const auto e1 = hn::Eq(magnitude(hn::LoadU(d, x + i + N)), target);
    const auto e2 = hn::Eq(magnitude(hn::LoadU(d, x + i + 2 * N)), target);
    const auto e3 = hn::Eq(magnitude(hn::LoadU(d, x + i + 3 * N)), target);
    if (HWY_UNLIKELY(
            !hn::AllFalse(d, hn::Or(hn::Or(e0, e1), hn::Or(e2, e3))))) {
      break;  // found in this block; locate it below
    }
  }
  for (; i < len; i += N) {
    const size_t count = HWY_MIN(N, len - i);
    const auto eq =
        hn::And(hn::Eq(magnitude(hn::LoadN(d, x + i, count)), target),
                hn::FirstN(d, count));
    const intptr_t lane = hn::FindFirstTrue(d, eq);
    if (lane >= 0) return (i + static_cast<size_t>(lane)) / kReals;
  }
  return 0;  // not reached: the target occurs in x
}

// ---------------------------------------------------------------------------
// Element-wise operations on contiguous vectors (in reals).

// Applies the 2 x 2 transformation [x; y] := [h11 h12; h21 h22] [x; y] to
// len contiguous reals of x and y (ROT and ROTM).
template <typename R>
HWY_INLINE void Rotate(size_t len, R* HWY_RESTRICT x, R* HWY_RESTRICT y, R h11,
                       R h12, R h21, R h22) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const V v11 = hn::Set(d, h11), v12 = hn::Set(d, h12);
  const V v21 = hn::Set(d, h21), v22 = hn::Set(d, h22);
  size_t i = 0;
  for (; i + N <= len; i += N) {
    const V xv = hn::LoadU(d, x + i), yv = hn::LoadU(d, y + i);
    hn::StoreU(hn::MulAdd(xv, v11, hn::Mul(yv, v12)), d, x + i);
    hn::StoreU(hn::MulAdd(xv, v21, hn::Mul(yv, v22)), d, y + i);
  }
  if (i < len) {
    const size_t count = len - i;
    const V xv = hn::LoadN(d, x + i, count), yv = hn::LoadN(d, y + i, count);
    hn::StoreN(hn::MulAdd(xv, v11, hn::Mul(yv, v12)), d, x + i, count);
    hn::StoreN(hn::MulAdd(xv, v21, hn::Mul(yv, v22)), d, y + i, count);
  }
}

// Exchanges len contiguous reals of x and y.
template <typename R>
HWY_INLINE void Swap(size_t len, R* HWY_RESTRICT x, R* HWY_RESTRICT y) {
  const hn::ScalableTag<R> d;
  const size_t N = hn::Lanes(d);
  size_t i = 0;
  for (; i + N <= len; i += N) {
    const auto xv = hn::LoadU(d, x + i), yv = hn::LoadU(d, y + i);
    hn::StoreU(yv, d, x + i);
    hn::StoreU(xv, d, y + i);
  }
  for (; i < len; ++i) {
    const R t = x[i];
    x[i] = y[i];
    y[i] = t;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_LEVEL1_INL_H_
