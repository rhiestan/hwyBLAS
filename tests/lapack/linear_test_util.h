// Shared helpers of the tests of the linear systems routines (general,
// triangular, symmetric and Hermitian): reference matrices and inverses,
// the backward error of the refinement routines and the LAPACKE layouts.
#ifndef HWYBLAS_TESTS_LAPACK_LINEAR_TEST_UTIL_H_
#define HWYBLAS_TESTS_LAPACK_LINEAR_TEST_UTIL_H_

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {

// |re| + |im| (|x| for real x), the absolute value of CABS1.
template <typename X>
double Cabs1Ref(X x) {
  return std::abs(static_cast<double>(std::real(x))) +
         std::abs(static_cast<double>(std::imag(x)));
}

// A power of 2 that scales a right hand side so that the denominators of
// the backward errors of the refinement routines (xGERFS, xTRRFS, xPORFS,
// xSYRFS, ...) fall below SAFE2 (where SAFE1 is added).
template <typename T>
double NearUnderflow() {
  return std::is_same_v<RealT<T>, float> ? std::ldexp(1.0, -110)
                                         : std::ldexp(1.0, -982);
}

// The componentwise backward error of the refinement routines for the
// residual r and the denominators denom = (abs(op(A))*abs(X) + abs(B))(i),
// with SAFE1 added where denom(i) <= SAFE2.
template <typename T>
double BackwardError(const std::vector<double>& r,
                     const std::vector<double>& denom) {
  using R = RealT<T>;
  const double nz = static_cast<double>(r.size() + 1);
  const double safe1 = nz * std::numeric_limits<R>::min();
  const double safe2 = safe1 / LapackEps<T>();
  double s = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    s = std::max(s, denom[i] > safe2 ? r[i] / denom[i]
                                     : (r[i] + safe1) / (denom[i] + safe1));
  }
  return s;
}

// op(d) for op = 'N', 'T' or 'C'.
template <typename T>
Dense<T> Op(char op, const Dense<T>& d) {
  if (op == 'N') return d;
  Dense<T> t(d.cols, d.rows);
  for (size_t j = 0; j < d.cols; ++j) {
    for (size_t i = 0; i < d.rows; ++i) {
      t(j, i) = op == 'C' ? Conj(d(i, j)) : d(i, j);
    }
  }
  return t;
}

// The infinity norm (maximum row sum) of a dense matrix.
template <typename T>
double NormInf(const Dense<T>& a) {
  double norm = 0;
  for (size_t i = 0; i < a.rows; ++i) {
    double sum = 0;
    for (size_t j = 0; j < a.cols; ++j) sum += std::abs(a(i, j));
    norm = std::max(norm, sum);
  }
  return norm;
}

// The inverse of the n x n matrix d by Gauss-Jordan elimination with
// partial pivoting, in the reference precision.
template <typename T>
Dense<T> Inverse(Dense<T> d) {
  const size_t n = d.rows;
  Dense<T> inv = Identity<T>(n);
  for (size_t k = 0; k < n; ++k) {
    size_t p = k;
    for (size_t i = k + 1; i < n; ++i) {
      if (std::abs(d(i, k)) > std::abs(d(p, k))) p = i;
    }
    for (size_t j = 0; j < n; ++j) {
      std::swap(d(k, j), d(p, j));
      std::swap(inv(k, j), inv(p, j));
    }
    const RefT<T> pivot = d(k, k);
    for (size_t j = 0; j < n; ++j) {
      d(k, j) /= pivot;
      inv(k, j) /= pivot;
    }
    for (size_t i = 0; i < n; ++i) {
      if (i == k) continue;
      const RefT<T> f = d(i, k);
      for (size_t j = 0; j < n; ++j) {
        d(i, j) -= f * d(k, j);
        inv(i, j) -= f * inv(k, j);
      }
    }
  }
  return inv;
}

// The value stored outside the triangle uplo of a symmetric (Hermitian)
// matrix, which the routines must not change.
template <typename T>
T Sentinel() {
  return MakeScalar<T>(7.0, -3.0);
}

// Whether element (i, j) is in the triangle uplo ('U' or 'L').
inline bool InTriangle(char uplo, size_t i, size_t j) {
  return uplo == 'U' ? i <= j : i >= j;
}

// A random symmetric (Hermitian) positive definite n x n matrix, stored in
// the triangle uplo of a (leading dimension lda), with Sentinel() elsewhere:
// B * B**H + n * I for random B, with a real diagonal.
template <typename T>
std::vector<T> PositiveDefinite(char uplo, size_t n, size_t lda,
                                Random<T>& rng) {
  const std::vector<T> b = rng.Vector(n * n);
  Dense<T> full = Mul(ToDense(b, n, n, n), Adjoint(ToDense(b, n, n, n)));
  std::vector<T> a(lda * n, Sentinel<T>());
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      if (!InTriangle(uplo, i, j)) continue;
      RefT<T> v = full(i, j);
      if (i == j) v = std::real(v) + static_cast<double>(n);
      At(a, i, j, lda) = static_cast<T>(v);
    }
  }
  return a;
}

// The Hermitian matrix of the triangle uplo of a (real diagonal).
template <typename T>
Dense<T> HermitianOf(char uplo, const std::vector<T>& a, size_t n, size_t lda) {
  Dense<T> d(n, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      const RefT<T> v = InTriangle(uplo, i, j)
                            ? static_cast<RefT<T>>(At(a, i, j, lda))
                            : Conj(static_cast<RefT<T>>(At(a, j, i, lda)));
      d(i, j) = (i == j) ? RefT<T>(std::real(v)) : v;
    }
  }
  return d;
}

// The symmetric matrix of the triangle uplo of a (complex symmetric for
// complex T).
template <typename T>
Dense<T> SymmetricOf(char uplo, const std::vector<T>& a, size_t n, size_t lda) {
  Dense<T> d(n, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      d(i, j) = InTriangle(uplo, i, j) ? static_cast<RefT<T>>(At(a, i, j, lda))
                                       : static_cast<RefT<T>>(At(a, j, i, lda));
    }
  }
  return d;
}

// Calls LAPACKE_s/d/c/z##NAME(args...) according to T.
#define LAPACKE_CALL(T, NAME, ...)                                 \
  [&]() {                                                          \
    if constexpr (std::is_same_v<T, float>) {                      \
      return LAPACKE_s##NAME(__VA_ARGS__);                         \
    } else if constexpr (std::is_same_v<T, double>) {              \
      return LAPACKE_d##NAME(__VA_ARGS__);                         \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      return LAPACKE_c##NAME(__VA_ARGS__);                         \
    } else {                                                       \
      return LAPACKE_z##NAME(__VA_ARGS__);                         \
    }                                                              \
  }()

// The m x n column-major matrix col (leading dimension ldc) in row-major
// layout (leading dimension ldr); the elements beyond n in each row are 5.
template <typename T>
std::vector<T> RowMajor(const std::vector<T>& col, size_t m, size_t n,
                        size_t ldc, size_t ldr) {
  std::vector<T> row(ldr * std::max<size_t>(m, 1), T(5));
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) row[i * ldr + j] = At(col, i, j, ldc);
  }
  return row;
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_LAPACK_LINEAR_TEST_UTIL_H_
