// Tests of the Cholesky factorization: xPOTRF, xPOTRF2 and their LAPACKE
// functions.
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "lapack/linear_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class PotrfTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PotrfTest, ScalarTypes);

template <typename T>
hwyblas_int Potrf(bool recursive, char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda) {
  hwyblas_int info = -99;
  if (recursive) {
    HWYBLAS_CALL(T, potrf2, &uplo, &n, P(a.data()), &lda, &info);
  } else {
    HWYBLAS_CALL(T, potrf, &uplo, &n, P(a.data()), &lda, &info);
  }
  return info;
}

// The test ratio of LAPACK's xPOT01: ||U**H * U - A||_1 / (n ||A||_1 eps)
// (or with L * L**H), for the factor in the triangle uplo of `factor`.
template <typename T>
double CholeskyResidual(char uplo, const std::vector<T>& a,
                        const std::vector<T>& factor, size_t n, size_t lda) {
  Dense<T> f(n, n);  // U or L
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      const bool stored = (uplo == 'U') ? (i <= j) : (i >= j);
      if (stored) f(i, j) = static_cast<RefT<T>>(At(factor, i, j, lda));
    }
  }
  const Dense<T> prod = (uplo == 'U') ? Mul(Adjoint(f), f) : Mul(f, Adjoint(f));
  return ResidualRatio(HermitianOf(uplo, a, n, lda), prod,
                       static_cast<double>(n));
}

TYPED_TEST(PotrfTest, Factorization) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(11);
    for (const bool recursive : {false, true}) {
      for (const hwyblas_int nb : BlockSizes()) {
        if (recursive && nb != 0) continue;  // xPOTRF2 does not block
        const ScopedBlockSize block_size(nb);
        for (const char uplo : {'U', 'L'}) {
          // Including sizes around the leaf size of xPOTRF2 (32).
          for (const hwyblas_int n :
               {1, 2, 3, 7, 16, 31, 32, 33, 64, 65, 130}) {
            for (const hwyblas_int pad : {0, 3}) {
              const hwyblas_int lda = n + pad;
              SCOPED_TRACE(::testing::Message()
                           << (recursive ? "potrf2" : "potrf")
                           << " uplo=" << uplo << " n=" << n << " lda=" << lda
                           << " nb=" << nb);
              const std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
              std::vector<T> f = a;
              ASSERT_EQ(Potrf(recursive, uplo, n, f, lda), 0);
              EXPECT_LT(CholeskyResidual(uplo, a, f, n, lda), kLapackThreshold);
              // The factor's diagonal is real and positive, and nothing
              // outside the triangle changes.
              for (hwyblas_int j = 0; j < n; ++j) {
                const T d = At(f, j, j, lda);
                EXPECT_GT(std::real(d), 0);
                EXPECT_EQ(std::imag(d), 0);
                for (hwyblas_int i = 0; i < lda; ++i) {
                  const bool stored =
                      i < n && ((uplo == 'U') ? (i <= j) : (i >= j));
                  if (!stored) {
                    ASSERT_EQ(At(f, i, j, lda), At(a, i, j, lda));
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

// A leading minor of order k that is not positive (here: A(k, k) <= 0 with
// zeros beside it, or NaN) gives INFO = k; the first k - 1 columns are then
// factored. n = 70 also has k in the recursion beyond the leaf.
TYPED_TEST(PotrfTest, NotPositiveDefinite) {
  using T = TypeParam;
  for (const hwyblas_int n : {9, 70}) {
    for (const bool recursive : {false, true}) {
      for (const hwyblas_int nb : BlockSizes()) {
        if (recursive && nb != 0) continue;
        const ScopedBlockSize block_size(nb);
        for (const char uplo : {'U', 'L'}) {
          for (const hwyblas_int k : {1, 2, 4, 9, 33, 70}) {
            if (k > n) continue;
            for (const double value :
                 {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
              SCOPED_TRACE(::testing::Message()
                           << (recursive ? "potrf2" : "potrf")
                           << " uplo=" << uplo << " n=" << n << " k=" << k
                           << " value=" << value << " nb=" << nb);
              std::vector<T> a(n * n);
              for (hwyblas_int i = 0; i < n; ++i) At(a, i, i, n) = T(4);
              At(a, k - 1, k - 1, n) = T(static_cast<RealT<T>>(value));
              EXPECT_EQ(Potrf(recursive, uplo, n, a, n), k);
              for (hwyblas_int i = 0; i < k - 1; ++i) {
                EXPECT_EQ(At(a, i, i, n), T(2));
              }
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(PotrfTest, QuickReturn) {
  using T = TypeParam;
  std::vector<T> a(1, T(5));
  for (const bool recursive : {false, true}) {
    EXPECT_EQ(Potrf(recursive, 'U', 0, a, 1), 0);
    EXPECT_EQ(a[0], T(5));
  }
}

TYPED_TEST(PotrfTest, InvalidArguments) {
  using T = TypeParam;
  std::vector<T> a(16, T(1));
  for (const bool recursive : {false, true}) {
    const std::string name = Name<T>(recursive ? "POTRF2" : "POTRF");
    ErrorCapture errors;
    EXPECT_EQ(Potrf(recursive, 'X', 4, a, 4), -1);
    EXPECT_EQ(Potrf(recursive, 'U', -1, a, 4), -2);
    EXPECT_EQ(Potrf(recursive, 'L', 4, a, 3), -4);
    EXPECT_EQ(errors.Take(), (std::vector<std::string>{name + ":1", name + ":2",
                                                       name + ":4"}));
  }
  EXPECT_EQ(a, std::vector<T>(16, T(1)));
}

template <typename T>
lapack_int LapackePotrf(bool recursive, bool work, int layout, char uplo,
                        lapack_int n, T* a, lapack_int lda) {
  if constexpr (std::is_same_v<T, float>) {
    if (recursive) {
      return work ? LAPACKE_spotrf2_work(layout, uplo, n, a, lda)
                  : LAPACKE_spotrf2(layout, uplo, n, a, lda);
    }
    return work ? LAPACKE_spotrf_work(layout, uplo, n, a, lda)
                : LAPACKE_spotrf(layout, uplo, n, a, lda);
  } else if constexpr (std::is_same_v<T, double>) {
    if (recursive) {
      return work ? LAPACKE_dpotrf2_work(layout, uplo, n, a, lda)
                  : LAPACKE_dpotrf2(layout, uplo, n, a, lda);
    }
    return work ? LAPACKE_dpotrf_work(layout, uplo, n, a, lda)
                : LAPACKE_dpotrf(layout, uplo, n, a, lda);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    if (recursive) {
      return work ? LAPACKE_cpotrf2_work(layout, uplo, n, a, lda)
                  : LAPACKE_cpotrf2(layout, uplo, n, a, lda);
    }
    return work ? LAPACKE_cpotrf_work(layout, uplo, n, a, lda)
                : LAPACKE_cpotrf(layout, uplo, n, a, lda);
  } else {
    if (recursive) {
      return work ? LAPACKE_zpotrf2_work(layout, uplo, n, a, lda)
                  : LAPACKE_zpotrf2(layout, uplo, n, a, lda);
    }
    return work ? LAPACKE_zpotrf_work(layout, uplo, n, a, lda)
                : LAPACKE_zpotrf(layout, uplo, n, a, lda);
  }
}

// Row-major A gives the same factor as column-major A, in row-major
// storage; only the triangle uplo is copied in and out.
TYPED_TEST(PotrfTest, LapackeLayouts) {
  using T = TypeParam;
  Random<T> rng(5);
  const lapack_int n = 21, ld = 24;
  for (const bool recursive : {false, true}) {
    for (const bool work : {false, true}) {
      for (const char uplo : {'U', 'L'}) {
        SCOPED_TRACE(::testing::Message()
                     << "recursive=" << recursive << " work=" << work
                     << " uplo=" << uplo);
        const std::vector<T> a = PositiveDefinite<T>(uplo, n, ld, rng);
        std::vector<T> col = a;
        ASSERT_EQ(LapackePotrf(recursive, work, LAPACK_COL_MAJOR, uplo, n,
                               col.data(), ld),
                  0);
        // The same matrix, row-major: element (i, j) at [i * ld + j].
        std::vector<T> row(a.size(), Sentinel<T>());
        for (lapack_int j = 0; j < n; ++j) {
          for (lapack_int i = 0; i < n; ++i) row[i * ld + j] = At(a, i, j, ld);
        }
        ASSERT_EQ(LapackePotrf(recursive, work, LAPACK_ROW_MAJOR, uplo, n,
                               row.data(), ld),
                  0);
        for (lapack_int j = 0; j < n; ++j) {
          for (lapack_int i = 0; i < n; ++i) {
            const bool stored = (uplo == 'U') ? (i <= j) : (i >= j);
            ASSERT_EQ(row[i * ld + j],
                      stored ? At(col, i, j, ld) : At(a, i, j, ld))
                << i << ", " << j;
          }
        }
      }
    }
  }
}

TYPED_TEST(PotrfTest, LapackeErrors) {
  using T = TypeParam;
  std::vector<T> a(16, T(0));
  for (int i = 0; i < 4; ++i) a[i * 5] = T(1);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackePotrf<T>(false, false, 0, 'U', 4, a.data(), 4), -1);
  EXPECT_EQ(LapackePotrf<T>(false, true, 0, 'U', 4, a.data(), 4), -1);
  EXPECT_EQ(LapackePotrf<T>(true, true, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3),
            -5);
  const std::string out = ::testing::internal::GetCapturedStdout();
  EXPECT_EQ(out, "Wrong parameter 1 in " + LapackeName<T>("potrf") +
                     "\nWrong parameter 1 in " + LapackeName<T>("potrf_work") +
                     "\nWrong parameter 5 in " + LapackeName<T>("potrf2_work") +
                     "\n");
  {
    // An invalid argument of xPOTRF itself, shifted by one.
    ErrorCapture errors;
    EXPECT_EQ(
        LapackePotrf<T>(false, false, LAPACK_COL_MAJOR, 'X', 4, a.data(), 4),
        -2);
    EXPECT_EQ(errors.Take(), std::vector<std::string>{Name<T>("POTRF") + ":1"});
  }

  // NaN check: -4 (argument a) for a NaN in the triangle uplo only.
  a[4] = NaN<T>();  // (0, 1): in the upper triangle (column-major)
  EXPECT_EQ(
      LapackePotrf<T>(false, false, LAPACK_COL_MAJOR, 'U', 4, a.data(), 4), -4);
  std::vector<T> b = a;
  EXPECT_EQ(
      LapackePotrf<T>(false, false, LAPACK_COL_MAJOR, 'L', 4, b.data(), 4), 0);
  b = a;
  EXPECT_EQ(LapackePotrf<T>(true, false, LAPACK_ROW_MAJOR, 'L', 4, b.data(), 4),
            -4);
  LAPACKE_set_nancheck(0);
  b = a;
  EXPECT_NE(
      LapackePotrf<T>(false, false, LAPACK_COL_MAJOR, 'U', 4, b.data(), 4), -4);
  LAPACKE_set_nancheck(1);
}

}  // namespace
}  // namespace hwyblas_test
