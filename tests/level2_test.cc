// Tests for SYMV, HEMV, TRMV, TRSV and the rank updates (GER, SYR, HER,
// SYR2, HER2) against references computed in double precision. Structured
// matrices have NaN wherever BLAS must not read.
#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <complex>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class Level2Test : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Level2Test, ScalarTypes);

// ---------------------------------------------------------------------------
// SYMV / HEMV

template <typename T>
void CheckSymv(Shape shape, char uplo, int n, int incx, int incy, T alpha,
               T beta, uint64_t seed) {
  const int lda = n + 3;
  Random<T> rng(seed);
  const std::vector<T> a = StructuredMatrix(shape, uplo, n, lda, rng);
  StridedVector<T> x(n, incx, rng), y(n, incy, rng);
  if (beta == T(0)) {
    for (int i = 0; i < n; ++i) y[i] = NaN<T>();
  }
  std::vector<T> y0(n);
  for (int i = 0; i < n; ++i) y0[i] = y[i];

  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  ReferenceProduct(DenseStructured(shape, uplo, 'N', a, n, lda), x.ToDense(),
                   alpha, beta, y0, n, &expected, &bound);

  if (shape == Shape::kHermitian) {
    HWYBLAS_CALL_COMPLEX(T, hemv, &uplo, &n, P(&alpha), P(a.data()), &lda,
                         P(x.data.data()), &incx, P(&beta), P(y.data.data()),
                         &incy);
  } else {
    HWYBLAS_CALL_REAL(T, symv, &uplo, &n, P(&alpha), P(a.data()), &lda,
                      P(x.data.data()), &incx, P(&beta), P(y.data.data()),
                      &incy);
  }
  const double tol = 4.0 * (n + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(y.Elements(), expected, bound, tol, n, 1, n))
      << "uplo=" << uplo << " n=" << n << " incx=" << incx << " incy=" << incy;
}

TYPED_TEST(Level2Test, SymvHemv) {
  using T = TypeParam;
  // SYMV exists for real types, HEMV for complex ones.
  static constexpr Shape kShape =
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char uplo : {'L', 'U'}) {
      for (int n : {0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 64, 333}) {
        CheckSymv<T>(kShape, uplo, n, 1, 1, MakeScalar<T>(1.5, -0.5),
                     MakeScalar<T>(0.5, 0.25), seed++);
      }
      for (int inc : {2, -1, -3}) {
        CheckSymv<T>(kShape, uplo, 100, inc, -inc, MakeScalar<T>(-1.0, 0.5),
                     MakeScalar<T>(0.0, 0.0), seed++);
      }
      CheckSymv<T>(kShape, uplo, 50, 1, 2, MakeScalar<T>(0.0, 0.0),
                   MakeScalar<T>(2.0, 0.0), seed++);
    }
  });
}

// ---------------------------------------------------------------------------
// TRMV

template <typename T>
void CheckTrmv(char uplo, char trans, char diag, int n, int incx,
               uint64_t seed) {
  const int lda = n + 1;
  const Shape shape =
      (diag == 'U') ? Shape::kUnitTriangular : Shape::kTriangular;
  Random<T> rng(seed);
  const std::vector<T> a = StructuredMatrix(shape, uplo, n, lda, rng);
  StridedVector<T> x(n, incx, rng);
  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  ReferenceProduct(DenseStructured(shape, uplo, trans, a, n, lda), x.ToDense(),
                   T(1), T(0), std::vector<T>(n), n, &expected, &bound);

  HWYBLAS_CALL(T, trmv, &uplo, &trans, &diag, &n, P(a.data()), &lda,
               P(x.data.data()), &incx);
  const double tol = 4.0 * (n + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(x.Elements(), expected, bound, tol, n, 1, n))
      << "uplo=" << uplo << " trans=" << trans << " diag=" << diag << " n=" << n
      << " incx=" << incx;
}

TYPED_TEST(Level2Test, Trmv) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char uplo : {'L', 'U'}) {
      for (char trans : Ops<T>()) {
        for (char diag : {'N', 'U'}) {
          for (int n : {0, 1, 2, 3, 4, 5, 9, 31, 64, 257}) {
            CheckTrmv<T>(uplo, trans, diag, n, 1, seed++);
          }
          CheckTrmv<T>(uplo, trans, diag, 77, -2, seed++);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// TRSV

template <typename T>
void CheckTrsv(char uplo, char trans, char diag, int n, int incx,
               uint64_t seed) {
  const int lda = n + 2;
  const Shape shape =
      (diag == 'U') ? Shape::kUnitTriangular : Shape::kTriangular;
  Random<T> rng(seed);
  // Scaled off-diagonal part: random triangular matrices are otherwise
  // exponentially ill-conditioned.
  const std::vector<T> a =
      StructuredMatrix(shape, uplo, n, lda, rng, 1.0 / std::max(n, 1));
  StridedVector<T> x(n, incx, rng);
  std::vector<T> b(n);
  for (int i = 0; i < n; ++i) b[i] = x[i];

  HWYBLAS_CALL(T, trsv, &uplo, &trans, &diag, &n, P(a.data()), &lda,
               P(x.data.data()), &incx);

  // Residual check: op(A) * x against b.
  std::vector<RefT<T>> product;
  std::vector<double> bound;
  ReferenceProduct(DenseStructured(shape, uplo, trans, a, n, lda), x.ToDense(),
                   T(1), T(0), std::vector<T>(n), n, &product, &bound);
  for (int i = 0; i < n; ++i) bound[i] += Abs(b[i]);
  const double tol = 8.0 * (n + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(b, product, bound, tol, n, 1, n))
      << "uplo=" << uplo << " trans=" << trans << " diag=" << diag << " n=" << n
      << " incx=" << incx;
  x.Elements();  // checks the gaps
}

TYPED_TEST(Level2Test, Trsv) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char uplo : {'L', 'U'}) {
      for (char trans : Ops<T>()) {
        for (char diag : {'N', 'U'}) {
          for (int n : {0, 1, 2, 3, 4, 5, 9, 31, 64, 257}) {
            CheckTrsv<T>(uplo, trans, diag, n, 1, seed++);
          }
          CheckTrsv<T>(uplo, trans, diag, 77, -2, seed++);
          CheckTrsv<T>(uplo, trans, diag, 30, 3, seed++);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Rank-1 and rank-2 updates

// Checks the m x n matrix `a` (leading dimension lda) after an update of
// `a0`: elements where `updated(i, j)` must match `expected` within
// tol * bound, all others (the other triangle, padding rows) must be
// unchanged bit for bit.
template <typename T, class Updated>
::testing::AssertionResult CheckUpdated(const std::vector<T>& a,
                                        const std::vector<T>& a0,
                                        const std::vector<RefT<T>>& expected,
                                        const std::vector<double>& bound,
                                        double tol, size_t n, size_t lda,
                                        const Updated& updated) {
  size_t failures = 0;
  std::string msg;
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < lda; ++i) {
      const size_t idx = i + j * lda;
      bool ok;
      if (updated(i, j)) {
        const double err =
            std::abs(static_cast<RefT<T>>(a[idx]) - expected[idx]);
        ok = err <= tol * bound[idx];  // false for NaN
      } else {
        ok = memcmp(&a[idx], &a0[idx], sizeof(T)) == 0;
      }
      if (!ok && failures++ < 5) {
        std::ostringstream os;
        os << " (" << i << "," << j << "): got " << a[idx] << " initial "
           << a0[idx] << (updated(i, j) ? " updated" : " untouched");
        msg += os.str();
      }
    }
  }
  if (failures == 0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << failures << " mismatches:" << msg;
}

// GER / GERU (conj = false) and GERC (conj = true).
template <typename T>
void CheckGer(bool conj, int m, int n, int incx, int incy, T alpha,
              uint64_t seed) {
  const int lda = m + 3;
  Random<T> rng(seed);
  std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = m; i < lda; ++i) a[i + j * lda] = NaN<T>();  // padding
  }
  const std::vector<T> a0 = a;
  StridedVector<T> x(m, incx, rng), y(n, incy, rng);

  std::vector<RefT<T>> expected(a.size());
  std::vector<double> bound(a.size());
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      const size_t idx = i + j * lda;
      const RefT<T> xi = x[i], yj = conj ? Conj(y[j]) : y[j];
      expected[idx] =
          static_cast<RefT<T>>(a0[idx]) + static_cast<RefT<T>>(alpha) * xi * yj;
      bound[idx] = Abs(a0[idx]) + Abs(alpha) * std::abs(xi) * std::abs(yj);
    }
  }

  if constexpr (IsComplex<T>::value) {
    if (conj) {
      HWYBLAS_CALL_COMPLEX(T, gerc, &m, &n, P(&alpha), P(x.data.data()), &incx,
                           P(y.data.data()), &incy, P(a.data()), &lda);
    } else {
      HWYBLAS_CALL_COMPLEX(T, geru, &m, &n, P(&alpha), P(x.data.data()), &incx,
                           P(y.data.data()), &incy, P(a.data()), &lda);
    }
  } else {
    HWYBLAS_CALL_REAL(T, ger, &m, &n, P(&alpha), P(x.data.data()), &incx,
                      P(y.data.data()), &incy, P(a.data()), &lda);
  }
  const bool any = !(alpha == T(0));
  EXPECT_TRUE(CheckUpdated(
      a, a0, expected, bound, 4 * Eps<T>(), n, lda,
      [&](size_t i, size_t) { return any && i < static_cast<size_t>(m); }))
      << "conj=" << conj << " m=" << m << " n=" << n << " incx=" << incx
      << " incy=" << incy;
  x.Elements();
  y.Elements();
}

TYPED_TEST(Level2Test, Ger) {
  using T = TypeParam;
  const std::vector<bool> conjs = IsComplex<T>::value
                                      ? std::vector<bool>{false, true}
                                      : std::vector<bool>{false};
  ForEachTarget([&] {
    uint64_t seed = 1;
    for (bool conj : conjs) {
      for (int m : {0, 1, 3, 4, 7, 8, 9, 33, 100}) {
        for (int n : {0, 1, 3, 4, 5, 9, 17}) {
          CheckGer<T>(conj, m, n, 1, 1, MakeScalar<T>(1.5, -0.5), seed++);
        }
      }
      CheckGer<T>(conj, 50, 13, 2, -3, MakeScalar<T>(-0.5, 2.0), seed++);
      CheckGer<T>(conj, 21, 30, -1, 2, MakeScalar<T>(0.75, 0.0), seed++);
      CheckGer<T>(conj, 20, 10, 1, 1, MakeScalar<T>(0.0, 0.0), seed++);
    }
  });
}

// SYR / HER (terms = 1) and SYR2 / HER2 (terms = 2). For complex T, the
// Hermitian variants are tested (there is no complex SYR or SYR2 in BLAS).
template <typename T>
void CheckSyr(int terms, char uplo, int n, int incx, int incy, T alpha,
              uint64_t seed) {
  constexpr bool kHermitian = IsComplex<T>::value;
  const int lda = n + 2;
  Random<T> rng(seed);
  // NaN in the other triangle, the padding rows and (Hermitian) the
  // imaginary parts of the diagonal: none of them may be read.
  std::vector<T> a = StructuredMatrix(
      kHermitian ? Shape::kHermitian : Shape::kSymmetric, uplo, n, lda, rng);
  const std::vector<T> a0 = a;
  StridedVector<T> x(n, incx, rng), y(n, incy, rng);
  // SYR and HER have a real alpha.
  if (terms == 1) alpha = T(std::real(alpha));

  const auto in_triangle = [&](size_t i, size_t j) {
    return (uplo == 'L') ? (i >= j && i < static_cast<size_t>(n)) : (i <= j);
  };
  std::vector<RefT<T>> expected(a.size());
  std::vector<double> bound(a.size());
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (!in_triangle(i, j)) continue;
      const size_t idx = i + j * lda;
      const RefT<T> xi = x[i], xj = x[j], yi = y[i], yj = y[j];
      const RefT<T> al = static_cast<RefT<T>>(alpha);
      RefT<T> update = al * xi * Conj(xj);
      double scale = Abs(alpha) * std::abs(xi) * std::abs(xj);
      if (terms == 2) {
        update = al * xi * Conj(yj) + Conj(al) * yi * Conj(xj);
        scale = Abs(alpha) *
                (std::abs(xi) * std::abs(yj) + std::abs(yi) * std::abs(xj));
      }
      RefT<T> a_ij = static_cast<RefT<T>>(a0[idx]);
      if (kHermitian && i == j) {
        a_ij = std::real(a_ij);
        update = std::real(update);
      }
      expected[idx] = a_ij + update;
      bound[idx] = std::abs(a_ij) + scale;
    }
  }

  if (terms == 1) {
    const RealT<T> real_alpha = std::real(alpha);
    if constexpr (kHermitian) {
      HWYBLAS_CALL_COMPLEX(T, her, &uplo, &n, &real_alpha, P(x.data.data()),
                           &incx, P(a.data()), &lda);
    } else {
      HWYBLAS_CALL_REAL(T, syr, &uplo, &n, &real_alpha, P(x.data.data()), &incx,
                        P(a.data()), &lda);
    }
  } else {
    if constexpr (kHermitian) {
      HWYBLAS_CALL_COMPLEX(T, her2, &uplo, &n, P(&alpha), P(x.data.data()),
                           &incx, P(y.data.data()), &incy, P(a.data()), &lda);
    } else {
      HWYBLAS_CALL_REAL(T, syr2, &uplo, &n, P(&alpha), P(x.data.data()), &incx,
                        P(y.data.data()), &incy, P(a.data()), &lda);
    }
  }
  const bool any = !(alpha == T(0));
  EXPECT_TRUE(CheckUpdated(
      a, a0, expected, bound, 4 * Eps<T>(), n, lda,
      [&](size_t i, size_t j) { return any && in_triangle(i, j); }))
      << "terms=" << terms << " uplo=" << uplo << " n=" << n << " incx=" << incx
      << " incy=" << incy;
  if constexpr (kHermitian) {
    // The imaginary parts of the diagonal are set to zero exactly.
    for (int j = 0; any && j < n; ++j) {
      EXPECT_EQ(std::imag(a[j + j * lda]), 0) << "j=" << j;
    }
  }
  x.Elements();
  y.Elements();
}

TYPED_TEST(Level2Test, SyrHerSyr2Her2) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int terms : {1, 2}) {
      for (char uplo : {'L', 'U'}) {
        for (int n : {0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 64, 101}) {
          CheckSyr<T>(terms, uplo, n, 1, 1, MakeScalar<T>(1.5, -0.5), seed++);
        }
        CheckSyr<T>(terms, uplo, 40, 2, -1, MakeScalar<T>(-0.5, 2.0), seed++);
        CheckSyr<T>(terms, uplo, 23, -3, 3, MakeScalar<T>(0.75, 0.25), seed++);
        CheckSyr<T>(terms, uplo, 10, 1, 1, MakeScalar<T>(0.0, 0.0), seed++);
      }
    }
  });
}

}  // namespace
}  // namespace hwyblas_test
