// Tests of the LU factorization: xGETRF, xGETRF2, xLASWP and their LAPACKE
// functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class GetrfTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GetrfTest, ScalarTypes);

template <typename T>
hwyblas_int Getrf(bool recursive, hwyblas_int m, hwyblas_int n,
                  std::vector<T>& a, hwyblas_int lda,
                  std::vector<hwyblas_int>& ipiv) {
  hwyblas_int info = -99;
  if (recursive) {
    HWYBLAS_CALL(T, getrf2, &m, &n, P(a.data()), &lda, ipiv.data(), &info);
  } else {
    HWYBLAS_CALL(T, getrf, &m, &n, P(a.data()), &lda, ipiv.data(), &info);
  }
  return info;
}

// The test ratio of LAPACK's xGET01: ||P * L * U - A||_1 / (n ||A||_1 eps)
// for the factorization `lu` (with pivots ipiv) of the m x n matrix a.
template <typename T>
double LuResidual(const std::vector<T>& a, const std::vector<T>& lu,
                  const std::vector<hwyblas_int>& ipiv, size_t m, size_t n,
                  size_t lda) {
  const size_t k = std::min(m, n);
  Dense<T> prod(m, n);  // L * U
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      RefT<T> sum = 0;
      for (size_t l = 0; l <= std::min({i, j, k - 1}); ++l) {
        const RefT<T> lil =
            (l == i) ? RefT<T>(1) : static_cast<RefT<T>>(At(lu, i, l, lda));
        sum += lil * static_cast<RefT<T>>(At(lu, l, j, lda));
      }
      prod(i, j) = sum;
    }
  }
  // P * (L * U): undo the interchanges, last one first.
  for (size_t i = k; i-- > 0;) {
    const size_t ip = static_cast<size_t>(ipiv[i] - 1);
    for (size_t j = 0; j < n; ++j) std::swap(prod(i, j), prod(ip, j));
  }
  Dense<T> diff(m, n), orig(m, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      orig(i, j) = static_cast<RefT<T>>(At(a, i, j, lda));
      diff(i, j) = prod(i, j) - orig(i, j);
    }
  }
  const double eps = Eps<T>() / 2;  // xLAMCH('Epsilon')
  const double anorm = Norm1(orig);
  const double resid = Norm1(diff);
  if (anorm == 0) return resid == 0 ? 0 : 1 / eps;
  return resid / static_cast<double>(n) / anorm / eps;
}

// Checks the factorization of a: the residual, the pivot indices and the
// bound on L that partial pivoting guarantees: |L(i, j)| <= 1, or sqrt(2)
// for complex matrices, whose pivots are chosen by |re| + |im| (plus
// rounding errors of the scaling by the pivot's reciprocal).
template <typename T>
void CheckFactorization(const std::vector<T>& a, const std::vector<T>& lu,
                        const std::vector<hwyblas_int>& ipiv, size_t m,
                        size_t n, size_t lda) {
  for (size_t i = 0; i < std::min(m, n); ++i) {
    ASSERT_GE(ipiv[i], static_cast<hwyblas_int>(i + 1));
    ASSERT_LE(ipiv[i], static_cast<hwyblas_int>(m));
  }
  const double bound =
      (IsComplex<T>::value ? std::sqrt(2.0) : 1.0) * (1 + 4 * Eps<T>());
  for (size_t j = 0; j < std::min(m, n); ++j) {
    for (size_t i = j + 1; i < m; ++i) {
      ASSERT_LE(Abs(At(lu, i, j, lda)), bound) << "L(" << i << ", " << j << ")";
    }
  }
  EXPECT_LT(LuResidual(a, lu, ipiv, m, n, lda), kLapackThreshold);
}

struct Shape {
  hwyblas_int m, n;
};
// Including shapes around the leaf size of xGETRF2 (16).
const Shape kShapes[] = {{1, 1},    {1, 6},    {6, 1},    {2, 2},   {3, 7},
                         {7, 3},    {10, 10},  {16, 16},  {16, 40}, {40, 16},
                         {17, 17},  {33, 17},  {17, 33},  {64, 64}, {65, 70},
                         {90, 150}, {150, 90}, {200, 200}};

TYPED_TEST(GetrfTest, Factorization) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(7);
    for (const bool recursive : {false, true}) {
      for (const hwyblas_int nb : BlockSizes()) {
        if (recursive && nb != 0) continue;  // xGETRF2 does not block
        const ScopedBlockSize block_size(nb);
        for (const Shape& s : kShapes) {
          for (const hwyblas_int pad : {0, 3}) {
            const hwyblas_int lda = s.m + pad;
            SCOPED_TRACE(::testing::Message()
                         << (recursive ? "getrf2" : "getrf") << " m=" << s.m
                         << " n=" << s.n << " lda=" << lda << " nb=" << nb);
            const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * s.n);
            std::vector<T> lu = a;
            std::vector<hwyblas_int> ipiv(std::min(s.m, s.n));
            EXPECT_EQ(Getrf(recursive, s.m, s.n, lu, lda, ipiv), 0);
            CheckFactorization(a, lu, ipiv, s.m, s.n, lda);
            // Rows lda > m are not touched.
            for (hwyblas_int j = 0; j < s.n; ++j) {
              for (hwyblas_int i = s.m; i < lda; ++i) {
                ASSERT_EQ(At(lu, i, j, lda), At(a, i, j, lda));
              }
            }
          }
        }
      }
    }
  });
}

// A zero column k makes U(k, k) exactly zero: INFO = k is the first such
// column, and the factorization is still completed.
TYPED_TEST(GetrfTest, ExactlySingular) {
  using T = TypeParam;
  Random<T> rng(11);
  const hwyblas_int m = 40, n = 30, lda = m;
  for (const bool recursive : {false, true}) {
    for (const hwyblas_int nb : {0, 1, 8}) {
      const ScopedBlockSize block_size(nb);
      for (const hwyblas_int k : {1, 2, 8, 9, 17, 30}) {
        SCOPED_TRACE(::testing::Message() << (recursive ? "getrf2" : "getrf")
                                          << " nb=" << nb << " k=" << k);
        std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * n);
        for (hwyblas_int i = 0; i < m; ++i) At(a, i, k - 1, lda) = T(0);
        if (k < n) {  // a second zero column must not change INFO
          for (hwyblas_int i = 0; i < m; ++i) At(a, i, n - 1, lda) = T(0);
        }
        std::vector<T> lu = a;
        std::vector<hwyblas_int> ipiv(n);
        EXPECT_EQ(Getrf(recursive, m, n, lu, lda, ipiv), k);
        EXPECT_EQ(At(lu, k - 1, k - 1, lda), T(0));
        CheckFactorization(a, lu, ipiv, m, n, lda);
      }
    }
  }
}

// Pivots below the safe minimum: their reciprocal would overflow, so the
// column is divided by them instead. The first column is
// tiny * (1, 1/2, 1/4, ...), which gives L(i, 1) = 2**-i exactly; the other
// columns have a dominant diagonal, so that no later interchange moves these
// rows of L.
TYPED_TEST(GetrfTest, TinyPivots) {
  using T = TypeParam;
  using R = RealT<T>;
  for (const bool recursive : {false, true}) {
    for (const hwyblas_int n : {5, 40}) {
      SCOPED_TRACE(::testing::Message()
                   << (recursive ? "getrf2" : "getrf") << " n=" << n);
      const hwyblas_int m = n, lda = m;
      std::vector<T> a(static_cast<size_t>(lda) * n);
      const R tiny = std::numeric_limits<R>::min() / 4;
      const auto power = [](hwyblas_int i) { return -std::min<int>(i, 20); };
      for (hwyblas_int i = 0; i < m; ++i) {
        At(a, i, 0, lda) = T(std::ldexp(tiny, power(i)));
      }
      for (hwyblas_int k = 1; k < n; ++k) {
        At(a, 0, k, lda) = T(1);
        At(a, k, k, lda) = T(4);
      }
      std::vector<T> lu = a;
      std::vector<hwyblas_int> ipiv(n);
      EXPECT_EQ(Getrf(recursive, m, n, lu, lda, ipiv), 0);
      for (hwyblas_int i = 0; i < n; ++i) EXPECT_EQ(ipiv[i], i + 1);
      for (hwyblas_int i = 1; i < m; ++i) {
        ASSERT_EQ(At(lu, i, 0, lda), T(std::ldexp(R(1), power(i))))
            << "L(" << i << ", 0)";
      }
    }
  }
}

TYPED_TEST(GetrfTest, QuickReturns) {
  using T = TypeParam;
  for (const bool recursive : {false, true}) {
    for (const Shape s : {Shape{0, 0}, Shape{0, 5}, Shape{5, 0}}) {
      std::vector<T> a(25, T(3));
      std::vector<hwyblas_int> ipiv(5, -7);
      EXPECT_EQ(Getrf(recursive, s.m, s.n, a, 5, ipiv), 0);
      for (const T& x : a) EXPECT_EQ(x, T(3));
      for (const hwyblas_int p : ipiv) EXPECT_EQ(p, -7);
    }
  }
}

TYPED_TEST(GetrfTest, InvalidArguments) {
  using T = TypeParam;
  for (const bool recursive : {false, true}) {
    const std::string name = Name<T>(recursive ? "GETRF2" : "GETRF");
    std::vector<T> a(16, T(3));
    std::vector<hwyblas_int> ipiv(4, -7);
    ErrorCapture errors;
    EXPECT_EQ(Getrf(recursive, -1, 4, a, 4, ipiv), -1);
    EXPECT_EQ(Getrf(recursive, 4, -1, a, 4, ipiv), -2);
    EXPECT_EQ(Getrf(recursive, 4, 4, a, 3, ipiv), -4);
    EXPECT_EQ(Getrf(recursive, 0, 4, a, 0, ipiv), -4);  // lda >= max(1, m)
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{name + ":1", name + ":2", name + ":4",
                                        name + ":4"}));
    for (const T& x : a) EXPECT_EQ(x, T(3));
    for (const hwyblas_int p : ipiv) EXPECT_EQ(p, -7);
  }
}

// xLASWP against a direct implementation, for both directions, increments
// other than 1 and column counts around its block of 32.
TYPED_TEST(GetrfTest, Laswp) {
  using T = TypeParam;
  Random<T> rng(3);
  std::mt19937 row_rng(3);
  const hwyblas_int lda = 12;
  for (const hwyblas_int n : {0, 1, 31, 32, 33, 70}) {
    for (const hwyblas_int incx : {1, -1, 2, -2, 0}) {
      const hwyblas_int k1 = 2, k2 = 9;
      // IPIV(k1 + (i - k1) * |incx|) for i = k1 ... k2: rows in 1 ... lda.
      std::vector<hwyblas_int> ipiv(k1 +
                                    (k2 - k1) * std::max(1, std::abs(incx)));
      for (hwyblas_int& p : ipiv) {
        p = std::uniform_int_distribution<hwyblas_int>(1, lda)(row_rng);
      }
      const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * n);
      std::vector<T> expected = a;
      if (incx != 0) {
        const hwyblas_int inc = std::abs(incx);
        const auto swap_row = [&](hwyblas_int i) {
          const hwyblas_int ip = ipiv[k1 - 1 + (i - k1) * inc];
          for (hwyblas_int j = 0; j < n; ++j) {
            std::swap(At(expected, i - 1, j, lda),
                      At(expected, ip - 1, j, lda));
          }
        };
        if (incx > 0) {
          for (hwyblas_int i = k1; i <= k2; ++i) swap_row(i);
        } else {
          for (hwyblas_int i = k2; i >= k1; --i) swap_row(i);
        }
      }
      std::vector<T> actual = a;
      const hwyblas_int* ipiv_arg = ipiv.data();
      HWYBLAS_CALL(T, laswp, &n, P(actual.data()), &lda, &k1, &k2, ipiv_arg,
                   &incx);
      EXPECT_EQ(actual, expected) << "n=" << n << " incx=" << incx;
    }
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
lapack_int LapackeGetrf(bool work, int layout, lapack_int m, lapack_int n, T* a,
                        lapack_int lda, lapack_int* ipiv) {
  if constexpr (std::is_same_v<T, float>) {
    return work ? LAPACKE_sgetrf_work(layout, m, n, a, lda, ipiv)
                : LAPACKE_sgetrf(layout, m, n, a, lda, ipiv);
  } else if constexpr (std::is_same_v<T, double>) {
    return work ? LAPACKE_dgetrf_work(layout, m, n, a, lda, ipiv)
                : LAPACKE_dgetrf(layout, m, n, a, lda, ipiv);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return work ? LAPACKE_cgetrf_work(layout, m, n, a, lda, ipiv)
                : LAPACKE_cgetrf(layout, m, n, a, lda, ipiv);
  } else {
    return work ? LAPACKE_zgetrf_work(layout, m, n, a, lda, ipiv)
                : LAPACKE_zgetrf(layout, m, n, a, lda, ipiv);
  }
}

// Row-major LAPACKE calls factor the same matrix as column-major ones: the
// results agree exactly.
TYPED_TEST(GetrfTest, LapackeLayouts) {
  using T = TypeParam;
  Random<T> rng(5);
  for (const bool work : {false, true}) {
    for (const Shape s : {Shape{7, 5}, Shape{5, 7}, Shape{70, 70}}) {
      const lapack_int lda_col = s.m + 2, lda_row = s.n + 1;
      const std::vector<T> a = rng.Vector(static_cast<size_t>(lda_col) * s.n);
      std::vector<T> col = a;
      std::vector<T> row(static_cast<size_t>(s.m) * lda_row, T(-5));
      for (lapack_int i = 0; i < s.m; ++i) {
        for (lapack_int j = 0; j < s.n; ++j) {
          row[i * lda_row + j] = At(a, i, j, lda_col);
        }
      }
      std::vector<lapack_int> ipiv_col(std::min(s.m, s.n)),
          ipiv_row(ipiv_col.size());
      EXPECT_EQ(LapackeGetrf(work, LAPACK_COL_MAJOR, s.m, s.n, col.data(),
                             lda_col, ipiv_col.data()),
                0);
      EXPECT_EQ(LapackeGetrf(work, LAPACK_ROW_MAJOR, s.m, s.n, row.data(),
                             lda_row, ipiv_row.data()),
                0);
      EXPECT_EQ(ipiv_row, ipiv_col);
      for (lapack_int i = 0; i < s.m; ++i) {
        for (lapack_int j = 0; j < s.n; ++j) {
          ASSERT_EQ(row[i * lda_row + j], At(col, i, j, lda_col));
        }
        // Padding of row-major rows is kept.
        ASSERT_EQ(row[i * lda_row + s.n], T(-5));
      }
      CheckFactorization(a, col, ipiv_col, s.m, s.n, lda_col);
    }
  }
}

TYPED_TEST(GetrfTest, LapackeErrors) {
  using T = TypeParam;
  std::vector<T> a(16, T(1));
  std::vector<lapack_int> ipiv(4);
  // Invalid layout, and the arguments of xGETRF shifted by one.
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeGetrf<T>(false, 0, 4, 4, a.data(), 4, ipiv.data()), -1);
  EXPECT_EQ(LapackeGetrf<T>(true, 0, 4, 4, a.data(), 4, ipiv.data()), -1);
  EXPECT_EQ(
      LapackeGetrf<T>(true, LAPACK_ROW_MAJOR, 4, 4, a.data(), 3, ipiv.data()),
      -5);
  const std::string out = ::testing::internal::GetCapturedStdout();
  const std::string prefix = LapackeName<T>("getrf");
  EXPECT_EQ(out, "Wrong parameter 1 in " + prefix + "\nWrong parameter 1 in " +
                     prefix + "_work\nWrong parameter 5 in " + prefix +
                     "_work\n");
  {
    ErrorCapture errors;
    EXPECT_EQ(LapackeGetrf<T>(true, LAPACK_COL_MAJOR, -1, 4, a.data(), 4,
                              ipiv.data()),
              -2);
    EXPECT_EQ(errors.Take(), std::vector<std::string>{Name<T>("GETRF") + ":1"});
  }

  // NaN check: -4 (argument a) unless switched off.
  a[5] = NaN<T>();
  EXPECT_EQ(
      LapackeGetrf<T>(false, LAPACK_COL_MAJOR, 4, 4, a.data(), 4, ipiv.data()),
      -4);
  LAPACKE_set_nancheck(0);
  EXPECT_EQ(LAPACKE_get_nancheck(), 0);
  EXPECT_GE(
      LapackeGetrf<T>(false, LAPACK_COL_MAJOR, 4, 4, a.data(), 4, ipiv.data()),
      0);
  LAPACKE_set_nancheck(1);
}

}  // namespace
}  // namespace hwyblas_test
