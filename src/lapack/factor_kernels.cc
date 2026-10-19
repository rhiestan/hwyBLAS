// Kernels for the leaves of the recursive LU and Cholesky factorizations
// (xGETRF2, xPOTRF2): see factor_kernels.h.
//
// Both are the right-looking unblocked algorithms (as in xGETF2 and
// xPOTF2): at step j, the pivot column is scaled and the trailing matrix
// receives a rank-1 update, one contiguous column (AXPY) at a time. The whole
// leaf stays in L1, and nothing is dispatched per column, which is what made
// the recursion down to single columns slow for small matrices: each level
// called xTRSM and xGEMM / xHERK on blocks of a few elements.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/lapack/factor_kernels.cc"
#include "src/lapack/factor_kernels.h"

#include <utility>

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/level1-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// The column operations below work on full vectors only: masked stores
// (Highway's StoreN on AVX2) are slow, and the columns of a leaf are short,
// so most of them would end in one. A column of at least one vector ends
// instead with a vector that overlaps the one before it; it is computed
// from the original values before anything is stored, so that the
// overlapping elements are stored twice with the same value. Shorter
// columns are processed element by element.

// y[0:n] += alpha * x[0:n] (x and y do not overlap).
template <typename T>
HWY_INLINE void AxpyColumn(size_t n, T alpha, const T* HWY_RESTRICT x,
                           T* HWY_RESTRICT y) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const size_t len = RealCount<T>(n);
  if (len < N) {
    for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
    return;
  }
  const R* HWY_RESTRICT xr = reinterpret_cast<const R*>(x);
  R* HWY_RESTRICT yr = reinterpret_cast<R*>(y);
  // y + alpha * x is MulAdd(x, re(alpha), y) for real T; for complex T,
  // the imaginary part adds Reverse2(x) * (-im, im) (see MulComplex).
  R re, im = 0;
  if constexpr (IsComplex<T>::value) {
    re = alpha.real();
    im = alpha.imag();
  } else {
    re = alpha;
  }
  const V vre = hn::Set(d, re);
  const V vim = hn::OddEven(hn::Set(d, im), hn::Set(d, -im));
  const auto update = [=](V xv, V yv) HWY_ATTR {
    const V t = hn::MulAdd(xv, vre, yv);
    if constexpr (IsComplex<T>::value) {
      return hn::MulAdd(hn::Reverse2(d, xv), vim, t);
    } else {
      return t;
    }
  };
  const V last = update(hn::LoadU(d, xr + len - N), hn::LoadU(d, yr + len - N));
  for (size_t i = 0; i + N <= len; i += N) {
    hn::StoreU(update(hn::LoadU(d, xr + i), hn::LoadU(d, yr + i)), d, yr + i);
  }
  hn::StoreU(last, d, yr + len - N);
}

// x[0:n] *= alpha.
template <typename T>
HWY_INLINE void ScaleColumn(size_t n, T alpha, T* HWY_RESTRICT x) {
  using R = RealT<T>;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const size_t len = RealCount<T>(n);
  if (len < N) {
    for (size_t i = 0; i < n; ++i) x[i] *= alpha;
    return;
  }
  R* HWY_RESTRICT xr = reinterpret_cast<R*>(x);
  R re, im = 0;
  if constexpr (IsComplex<T>::value) {
    re = alpha.real();
    im = alpha.imag();
  } else {
    re = alpha;
  }
  const V vre = hn::Set(d, re), vim = hn::Set(d, im);
  constexpr bool kComplex = IsComplex<T>::value;
  const V last =
      ScaleByBroadcast<kComplex>(d, hn::LoadU(d, xr + len - N), vre, vim);
  for (size_t i = 0; i + N <= len; i += N) {
    hn::StoreU(ScaleByBroadcast<kComplex>(d, hn::LoadU(d, xr + i), vre, vim), d,
               xr + i);
  }
  hn::StoreU(last, d, xr + len - N);
}

// LU factorization with partial pivoting of the m x n column-major matrix a
// (0-based); see GetrfLeaf. sfmin is the safe minimum (xLAMCH('S')).
template <typename T>
blasint GetrfLeafImpl(size_t m, size_t n, T* HWY_RESTRICT a, size_t lda,
                      blasint* HWY_RESTRICT ipiv, RealT<T> sfmin) {
  const auto at = [a, lda](size_t i, size_t j) -> T& { return a[i + j * lda]; };
  blasint info = 0;
  for (size_t j = 0; j < HWY_MIN(m, n); ++j) {
    // Find the pivot (as xIAMAX) and test for singularity.
    const size_t p = j + ArgAbsExtremum<true>(m - j, &at(j, j));
    ipiv[j] = static_cast<blasint>(p + 1);
    const T pivot = at(p, j);
    if (pivot != T(0)) {
      // Apply the interchange to the columns of the leaf.
      if (p != j) {
        for (size_t k = 0; k < n; ++k) std::swap(at(j, k), at(p, k));
      }
      // Compute elements j+1:m of column j.
      if (std::abs(pivot) >= sfmin) {
        ScaleColumn(m - j - 1, T(1) / pivot, &at(j + 1, j));
      } else {
        for (size_t i = j + 1; i < m; ++i) at(i, j) = at(i, j) / pivot;
      }
    } else if (info == 0) {
      info = static_cast<blasint>(j + 1);
    }
    // Update the trailing submatrix.
    for (size_t k = j + 1; k < n; ++k) {
      AxpyColumn(m - j - 1, -at(j, k), &at(j + 1, j), &at(j + 1, k));
    }
  }
  return info;
}

// Cholesky factorization of the n x n column-major matrix a (0-based); see
// PotrfLeaf. n <= kFactorLeafMax.
template <typename T>
blasint PotrfLeafImpl(bool upper, size_t n, T* HWY_RESTRICT a, size_t lda) {
  using R = RealT<T>;
  const auto at = [a, lda](size_t i, size_t j) -> T& { return a[i + j * lda]; };
  // Row j of U, conjugated, as a contiguous vector.
  HWY_ALIGN T row[kFactorLeafMax];
  for (size_t j = 0; j < n; ++j) {
    // Test for non-positive-definiteness.
    R ajj = std::real(at(j, j));
    if (ajj <= R(0) || std::isnan(ajj)) return static_cast<blasint>(j + 1);
    ajj = std::sqrt(ajj);
    at(j, j) = T(ajj);
    const R rinv = R(1) / ajj;
    if (upper) {
      // Compute elements j+1:n of row j, then update the trailing triangle:
      // A(j+1:k, k) -= conj(U(j, j+1:k)) * U(j, k).
      for (size_t k = j + 1; k < n; ++k) {
        at(j, k) *= rinv;
        row[k] = Conj(at(j, k));
      }
      for (size_t k = j + 1; k < n; ++k) {
        AxpyColumn(k - j, -at(j, k), &row[j + 1], &at(j + 1, k));
      }
    } else {
      // Compute elements j+1:n of column j, then update the trailing
      // triangle: A(k:n, k) -= L(k:n, j) * conj(L(k, j)).
      ScaleColumn(RealCount<T>(n - j - 1), rinv,
                  reinterpret_cast<R*>(&at(j + 1, j)));
      for (size_t k = j + 1; k < n; ++k) {
        AxpyColumn(n - k, -Conj(at(k, j)), &at(k, j), &at(k, k));
      }
    }
  }
  return 0;
}

#define HWYBLAS_FACTOR_KERNELS(SUFFIX, T)                             \
  blasint GetrfLeaf##SUFFIX(size_t m, size_t n, T* a, size_t lda,     \
                            blasint* ipiv, RealT<T> sfmin) {          \
    return GetrfLeafImpl(m, n, a, lda, ipiv, sfmin);                  \
  }                                                                   \
  blasint PotrfLeaf##SUFFIX(bool upper, size_t n, T* a, size_t lda) { \
    return PotrfLeafImpl(upper, n, a, lda);                           \
  }
HWYBLAS_FACTOR_KERNELS(F32, float)
HWYBLAS_FACTOR_KERNELS(F64, double)
HWYBLAS_FACTOR_KERNELS(C32, c32)
HWYBLAS_FACTOR_KERNELS(C64, c64)
#undef HWYBLAS_FACTOR_KERNELS

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(GetrfLeaf);
HWYBLAS_EXPORT_BY_TYPE(PotrfLeaf);

namespace lapack {

template <typename T>
blasint GetrfLeaf(blasint m, blasint n, T* a, blasint lda, blasint* ipiv) {
  return HWYBLAS_DISPATCH_BY_TYPE(T, GetrfLeaf, m, n, a, lda, ipiv,
                                  Lamch<RealT<T>>('S'));
}

template <typename T>
blasint PotrfLeaf(bool upper, blasint n, T* a, blasint lda) {
  return HWYBLAS_DISPATCH_BY_TYPE(T, PotrfLeaf, upper, n, a, lda);
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                     \
  template blasint GetrfLeaf<T>(blasint, blasint, T*, blasint, blasint*); \
  template blasint PotrfLeaf<T>(bool, blasint, T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE(float)
HWYBLAS_LAPACK_INSTANTIATE(double)
HWYBLAS_LAPACK_INSTANTIATE(c32)
HWYBLAS_LAPACK_INSTANTIATE(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas
#endif  // HWY_ONCE
