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

// x[0:n] *= alpha.
template <typename T>
HWY_INLINE void ScalContiguous(size_t n, T alpha, T* HWY_RESTRICT x) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  const size_t N = hn::Lanes(d);
  R* HWY_RESTRICT xr = reinterpret_cast<R*>(x);
  const size_t len = IsComplex<T>::value ? 2 * n : n;  // in reals

  const auto scale = [&](hn::Vec<decltype(d)> v) HWY_ATTR {
    if constexpr (IsComplex<T>::value) {
      return MulComplex(d, v, alpha.real(), alpha.imag());
    } else {
      return hn::Mul(v, hn::Set(d, alpha));
    }
  };

  size_t i = 0;
  for (; i + N <= len; i += N) {
    hn::StoreU(scale(hn::LoadU(d, xr + i)), d, xr + i);
  }
  if (i < len) {
    const size_t count = len - i;
    hn::StoreN(scale(hn::LoadN(d, xr + i, count)), d, xr + i, count);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_LEVEL1_INL_H_
