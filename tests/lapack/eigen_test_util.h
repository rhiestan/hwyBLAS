// Helpers of the tests of the symmetric (Hermitian) eigenvalue routines.
#ifndef HWYBLAS_TESTS_LAPACK_EIGEN_TEST_UTIL_H_
#define HWYBLAS_TESTS_LAPACK_EIGEN_TEST_UTIL_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {

// Calls the real routine `rname` or the complex routine `cname`.
#define CALL_RC(T, rname, cname, ...)                              \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      s##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, double>) {              \
      d##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      c##cname##_(__VA_ARGS__);                                    \
    } else {                                                       \
      z##cname##_(__VA_ARGS__);                                    \
    }                                                              \
  } while (0)

// A random symmetric (Hermitian) n x n matrix in the triangle uplo of a
// (leading dimension lda), with NaN elsewhere and in the imaginary parts of
// the diagonal.
template <typename T>
std::vector<T> RandomHermitian(char uplo, hwyblas_int n, hwyblas_int lda,
                               Random<T>& rng) {
  return StructuredMatrix(
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric, uplo, n, lda,
      rng);
}

template <typename T>
Dense<T> HermitianDense(char uplo, const std::vector<T>& a, hwyblas_int n,
                        hwyblas_int lda) {
  return DenseStructured(
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric, uplo, 'N', a,
      n, lda);
}

// The tridiagonal matrix (d, e) as a dense matrix.
template <typename T>
Dense<T> Tridiagonal(const std::vector<RealT<T>>& d,
                     const std::vector<RealT<T>>& e) {
  const size_t n = d.size();
  Dense<T> t(n, n);
  for (size_t i = 0; i < n; ++i) {
    t(i, i) = d[i];
    if (i + 1 < n) t(i + 1, i) = t(i, i + 1) = e[i];
  }
  return t;
}

// ||A * Z - Z * diag(w)||_1 / (n ||A||_1 eps) (LAPACK's xSYT21 ratio, for
// eigenvectors in z).
template <typename T>
double EigenResidual(const Dense<T>& a, const Dense<T>& z,
                     const std::vector<RealT<T>>& w) {
  Dense<T> zw = z;
  for (size_t j = 0; j < z.cols; ++j) {
    for (size_t i = 0; i < z.rows; ++i) zw(i, j) *= static_cast<double>(w[j]);
  }
  const double anorm = Norm1(a);
  const double resid = Norm1(Sub(Mul(a, z), zw));
  if (anorm == 0) return resid / LapackEps<T>();
  return resid / static_cast<double>(a.rows) / anorm / LapackEps<T>();
}

// Eigenvalues in ascending order.
template <typename R>
void ExpectAscending(const std::vector<R>& w) {
  for (size_t i = 1; i < w.size(); ++i) ASSERT_LE(w[i - 1], w[i]) << i;
}

// A symmetric tridiagonal matrix of one of several kinds: random, with
// zeros on the off-diagonal (it splits), graded, with clustered
// eigenvalues (the Wilkinson matrix), or scaled near the overflow or the
// underflow threshold (xSTEQR and xSTERF then scale it).
template <typename R>
void TestTridiagonal(int kind, hwyblas_int n, Random<R>& rng, std::vector<R>& d,
                     std::vector<R>& e) {
  d.resize(n);
  e.resize(std::max<hwyblas_int>(0, n - 1));
  for (auto& x : d) x = rng.Next();
  for (auto& x : e) x = rng.Next();
  switch (kind) {
    case 1:  // splits
      for (hwyblas_int i = 0; i + 1 < n; i += 3) e[i] = 0;
      break;
    case 2:  // graded
      for (hwyblas_int i = 0; i < n; ++i) {
        d[i] *= static_cast<R>(std::pow(10.0, -i % 8));
        if (i + 1 < n) e[i] *= static_cast<R>(std::pow(10.0, -i % 8));
      }
      break;
    case 3: {  // Wilkinson: d = |mid - i|, e = 1
      const hwyblas_int mid = n / 2;
      for (hwyblas_int i = 0; i < n; ++i) d[i] = std::abs(R(mid - i));
      for (auto& x : e) x = 1;
      break;
    }
    case 4:    // near overflow
    case 5: {  // near underflow
      const R scale = kind == 4 ? std::sqrt(std::numeric_limits<R>::max()) * 4
                                : std::sqrt(std::numeric_limits<R>::min()) / 4;
      for (auto& x : d) x *= scale;
      for (auto& x : e) x *= scale;
      break;
    }
    default:
      break;
  }
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_LAPACK_EIGEN_TEST_UTIL_H_
