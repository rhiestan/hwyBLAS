// Tests of the auxiliary routines: xLACPY, xLACGV, xLAPY2, xLAPY3, xLADIV,
// ILAxLC, ILAxLR, and the LAPACKE functions of the auxiliary routines and
// of xLASWP.
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class AuxiliaryTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(AuxiliaryTest, ScalarTypes);

template <typename R>
class RealAuxiliaryTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RealAuxiliaryTest, RealTypes);

TYPED_TEST(AuxiliaryTest, Lacpy) {
  using T = TypeParam;
  Random<T> rng(1);
  const hwyblas_int m = 5, n = 7, lda = 6, ldb = 8;
  const std::vector<T> a = rng.Vector(lda * n);
  for (const char uplo : {'U', 'l', 'A', 'x'}) {
    std::vector<T> b(ldb * n, T(9));
    HWYBLAS_CALL(T, lacpy, &uplo, &m, &n, P(a.data()), &lda, P(b.data()), &ldb);
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = 0; i < ldb; ++i) {
        const bool copied = i < m && ((uplo == 'U')   ? i <= j
                                      : (uplo == 'l') ? i >= j
                                                      : true);
        EXPECT_EQ(At(b, i, j, ldb), copied ? At(a, i, j, lda) : T(9))
            << uplo << " " << i << ", " << j;
      }
    }
  }
}

TYPED_TEST(AuxiliaryTest, Lacgv) {
  using T = TypeParam;
  if constexpr (IsComplex<T>::value) {
    Random<T> rng(2);
    for (const hwyblas_int incx : {1, 3, -2}) {
      const hwyblas_int n = 5;
      const std::vector<T> x0 = rng.Vector(n * std::abs(incx));
      std::vector<T> x = x0;
      HWYBLAS_CALL_COMPLEX(T, lacgv, &n, P(x.data()), &incx);
      for (size_t i = 0; i < x.size(); ++i) {
        const bool element = i % std::abs(incx) == 0;
        EXPECT_EQ(x[i], element ? std::conj(x0[i]) : x0[i]) << incx << " " << i;
      }
    }
  }
}

template <typename R>
R Lapy2(R x, R y) {
  if constexpr (std::is_same_v<R, float>) {
    return slapy2_(&x, &y);
  } else {
    return dlapy2_(&x, &y);
  }
}

template <typename R>
R Lapy3(R x, R y, R z) {
  if constexpr (std::is_same_v<R, float>) {
    return slapy3_(&x, &y, &z);
  } else {
    return dlapy3_(&x, &y, &z);
  }
}

TYPED_TEST(RealAuxiliaryTest, Lapy) {
  using R = TypeParam;
  const R big = std::numeric_limits<R>::max() / 4;
  const R tiny = std::numeric_limits<R>::min();
  const R nan = std::numeric_limits<R>::quiet_NaN();
  const R inf = std::numeric_limits<R>::infinity();
  EXPECT_EQ(Lapy2<R>(3, -4), R(5));
  EXPECT_EQ(Lapy2<R>(0, -2), R(2));
  // No overflow or underflow in the intermediate results.
  EXPECT_NEAR(Lapy2<R>(big, big) / big, std::sqrt(R(2)), 4 * Eps<R>());
  EXPECT_NEAR(Lapy2<R>(tiny, tiny) / tiny, std::sqrt(R(2)), 4 * Eps<R>());
  EXPECT_TRUE(std::isnan(Lapy2<R>(nan, 1)));
  EXPECT_TRUE(std::isnan(Lapy2<R>(1, nan)));
  EXPECT_EQ(Lapy2<R>(inf, 1), inf);

  EXPECT_NEAR(Lapy3<R>(2, -3, 6), R(7), 4 * Eps<R>() * 7);
  EXPECT_EQ(Lapy3<R>(0, 0, 0), R(0));
  EXPECT_NEAR(Lapy3<R>(big, big, big) / big, std::sqrt(R(3)), 4 * Eps<R>());
  EXPECT_NEAR(Lapy3<R>(tiny, 0, tiny) / tiny, std::sqrt(R(2)), 4 * Eps<R>());
  EXPECT_TRUE(std::isnan(Lapy3<R>(0, nan, 0)));
  EXPECT_EQ(Lapy3<R>(inf, 1, 0), inf);
}

// xLADIV divides without overflow where the textbook formula overflows,
// and accurately.
TYPED_TEST(RealAuxiliaryTest, Ladiv) {
  using R = TypeParam;
  const auto ladiv = [](R a, R b, R c, R d) {
    R p, q;
    if constexpr (std::is_same_v<R, float>) {
      sladiv_(&a, &b, &c, &d, &p, &q);
    } else {
      dladiv_(&a, &b, &c, &d, &p, &q);
    }
    return std::complex<R>(p, q);
  };
  const R big = std::numeric_limits<R>::max() / 2;
  const R tiny = std::numeric_limits<R>::min() * 4;
  struct Case {
    R a, b, c, d;
  };
  const Case cases[] = {
      {1, 2, 3, 4},       {-5, 1, 0, 2},      {7, 0, 2, 0},
      {big, big, 1, 1},   {1, 1, big, big},   {big, 1, big, 1},
      {tiny, tiny, 1, 1}, {1, 1, tiny, tiny}, {1, 2, -3, 0.5}};
  for (const Case& t : cases) {
    SCOPED_TRACE(::testing::Message()
                 << t.a << " " << t.b << " " << t.c << " " << t.d);
    const std::complex<R> z = ladiv(t.a, t.b, t.c, t.d);
    // Reference in double precision with scaling: (a + ib)/(c + id)
    // = ((a + ib)/s) / ((c + id)/s), where |(c + id)/s| is about 1 and the
    // division accurate to a few units in the last place.
    const double s = std::max(std::fabs(static_cast<double>(t.c)),
                              std::fabs(static_cast<double>(t.d)));
    const std::complex<double> num(t.a / s, t.b / s), den(t.c / s, t.d / s);
    const std::complex<double> expected = num / den;
    const double scale = std::abs(expected);
    ASSERT_TRUE(std::isfinite(z.real()) && std::isfinite(z.imag()));
    EXPECT_LE(std::abs(std::complex<double>(z.real(), z.imag()) - expected),
              8 * Eps<R>() * scale);
  }
}

TYPED_TEST(AuxiliaryTest, LastNonzero) {
  using T = TypeParam;
  const auto ilalc = [](hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
                        hwyblas_int lda) {
    hwyblas_int r = -1;
    if constexpr (std::is_same_v<T, float>) {
      r = ilaslc_(&m, &n, a.data(), &lda);
    } else if constexpr (std::is_same_v<T, double>) {
      r = iladlc_(&m, &n, a.data(), &lda);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      r = ilaclc_(&m, &n, P(a.data()), &lda);
    } else {
      r = ilazlc_(&m, &n, P(a.data()), &lda);
    }
    return r;
  };
  const auto ilalr = [](hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
                        hwyblas_int lda) {
    hwyblas_int r = -1;
    if constexpr (std::is_same_v<T, float>) {
      r = ilaslr_(&m, &n, a.data(), &lda);
    } else if constexpr (std::is_same_v<T, double>) {
      r = iladlr_(&m, &n, a.data(), &lda);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      r = ilaclr_(&m, &n, P(a.data()), &lda);
    } else {
      r = ilazlr_(&m, &n, P(a.data()), &lda);
    }
    return r;
  };
  const hwyblas_int m = 5, n = 6, lda = 7;
  std::vector<T> a(lda * n, T(0));
  // Padding rows below m are not part of A.
  for (hwyblas_int j = 0; j < n; ++j) At(a, 6, j, lda) = T(1);
  EXPECT_EQ(ilalc(m, n, a, lda), 0);
  EXPECT_EQ(ilalr(m, n, a, lda), 0);
  EXPECT_EQ(ilalc(m, 0, a, lda), 0);
  EXPECT_EQ(ilalr(0, n, a, lda), 0);
  At(a, 2, 3, lda) = MakeScalar<T>(0, 1);  // imaginary part only (complex)
  if (IsComplex<T>::value || std::real(At(a, 2, 3, lda)) != 0) {
    EXPECT_EQ(ilalc(m, n, a, lda), 4);
    EXPECT_EQ(ilalr(m, n, a, lda), 3);
  }
  At(a, 2, 3, lda) = T(2);
  At(a, 1, 1, lda) = T(-1);
  EXPECT_EQ(ilalc(m, n, a, lda), 4);
  EXPECT_EQ(ilalr(m, n, a, lda), 3);
  // The corners.
  At(a, 0, n - 1, lda) = T(1);
  EXPECT_EQ(ilalc(m, n, a, lda), n);
  At(a, m - 1, 0, lda) = T(1);
  EXPECT_EQ(ilalr(m, n, a, lda), m);
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
lapack_int LapackeLacpy(int layout, char uplo, lapack_int m, lapack_int n,
                        const T* a, lapack_int lda, T* b, lapack_int ldb) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_slacpy(layout, uplo, m, n, a, lda, b, ldb);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dlacpy(layout, uplo, m, n, a, lda, b, ldb);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_clacpy(layout, uplo, m, n, a, lda, b, ldb);
  } else {
    return LAPACKE_zlacpy(layout, uplo, m, n, a, lda, b, ldb);
  }
}

// In row-major layout, the triangle uplo of the row-major matrix is copied,
// and (unlike in the reference LAPACKE) the rest of B is kept.
TYPED_TEST(AuxiliaryTest, LapackeLacpy) {
  using T = TypeParam;
  Random<T> rng(4);
  const lapack_int m = 4, n = 6, ld = 7;
  const std::vector<T> a = rng.Vector(m * ld);
  for (const char uplo : {'U', 'L', 'A'}) {
    std::vector<T> b(m * ld, T(9));
    ASSERT_EQ(
        LapackeLacpy(LAPACK_ROW_MAJOR, uplo, m, n, a.data(), ld, b.data(), ld),
        0);
    for (lapack_int i = 0; i < m; ++i) {
      for (lapack_int j = 0; j < ld; ++j) {
        const bool copied = j < n && ((uplo == 'U')   ? i <= j
                                      : (uplo == 'L') ? i >= j
                                                      : true);
        EXPECT_EQ(b[i * ld + j], copied ? a[i * ld + j] : T(9))
            << uplo << " " << i << ", " << j;
      }
    }
  }
  std::vector<T> b(m * ld);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(
      LapackeLacpy(LAPACK_ROW_MAJOR, 'A', m, n, a.data(), 5, b.data(), ld), -6);
  EXPECT_EQ(
      LapackeLacpy(LAPACK_ROW_MAJOR, 'A', m, n, a.data(), ld, b.data(), 5), -8);
  EXPECT_EQ(LapackeLacpy(7, 'A', m, n, a.data(), ld, b.data(), ld), -1);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 6 in " + LapackeName<T>("lacpy_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("lacpy_work") +
                "\nWrong parameter 1 in " + LapackeName<T>("lacpy") + "\n");
  std::vector<T> a_nan = a;
  a_nan[3] = NaN<T>();
  EXPECT_EQ(
      LapackeLacpy(LAPACK_ROW_MAJOR, 'A', m, n, a_nan.data(), ld, b.data(), ld),
      -5);
}

TYPED_TEST(AuxiliaryTest, LapackeLaswp) {
  using T = TypeParam;
  Random<T> rng(6);
  const lapack_int n = 5, ld = 9;
  const std::vector<lapack_int> ipiv = {3, 7, 3, 4};
  const std::vector<T> a = rng.Vector(ld * ld);
  for (const lapack_int incx : {1, -1}) {
    // Column-major: rows of the 7 x 5 matrix; row-major: the same rows.
    std::vector<T> col = a, row(ld * ld);
    for (lapack_int i = 0; i < ld; ++i) {
      for (lapack_int j = 0; j < n; ++j) row[i * ld + j] = At(a, i, j, ld);
    }
    lapack_int r_col, r_row;
    if constexpr (std::is_same_v<T, float>) {
      r_col = LAPACKE_slaswp(LAPACK_COL_MAJOR, n, col.data(), ld, 1, 4,
                             ipiv.data(), incx);
      r_row = LAPACKE_slaswp(LAPACK_ROW_MAJOR, n, row.data(), ld, 1, 4,
                             ipiv.data(), incx);
    } else if constexpr (std::is_same_v<T, double>) {
      r_col = LAPACKE_dlaswp(LAPACK_COL_MAJOR, n, col.data(), ld, 1, 4,
                             ipiv.data(), incx);
      r_row = LAPACKE_dlaswp(LAPACK_ROW_MAJOR, n, row.data(), ld, 1, 4,
                             ipiv.data(), incx);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      r_col = LAPACKE_claswp(LAPACK_COL_MAJOR, n, col.data(), ld, 1, 4,
                             ipiv.data(), incx);
      r_row = LAPACKE_claswp(LAPACK_ROW_MAJOR, n, row.data(), ld, 1, 4,
                             ipiv.data(), incx);
    } else {
      r_col = LAPACKE_zlaswp(LAPACK_COL_MAJOR, n, col.data(), ld, 1, 4,
                             ipiv.data(), incx);
      r_row = LAPACKE_zlaswp(LAPACK_ROW_MAJOR, n, row.data(), ld, 1, 4,
                             ipiv.data(), incx);
    }
    EXPECT_EQ(r_col, 0);
    EXPECT_EQ(r_row, 0);
    // Rows 1 .. 7 (the largest pivot) are swapped.
    for (lapack_int i = 0; i < 7; ++i) {
      for (lapack_int j = 0; j < n; ++j) {
        ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << incx << " " << i;
      }
    }
  }
}

TYPED_TEST(AuxiliaryTest, LapackeVectors) {
  using T = TypeParam;
  Random<T> rng(8);
  // xLARFG: the same as the Fortran interface; NaN checks of alpha and x.
  std::vector<T> x = rng.Vector(4), x2 = x;
  T alpha = rng.Next(), alpha2 = alpha, tau, tau2;
  lapack_int r = -99;
  const hwyblas_int n = 5, one = 1;
  HWYBLAS_CALL(T, larfg, &n, P(&alpha2), P(x2.data()), &one, P(&tau2));
  if constexpr (std::is_same_v<T, float>) {
    r = LAPACKE_slarfg(n, &alpha, x.data(), 1, &tau);
  } else if constexpr (std::is_same_v<T, double>) {
    r = LAPACKE_dlarfg(n, &alpha, x.data(), 1, &tau);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    r = LAPACKE_clarfg(n, &alpha, x.data(), 1, &tau);
  } else {
    r = LAPACKE_zlarfg(n, &alpha, x.data(), 1, &tau);
  }
  EXPECT_EQ(r, 0);
  EXPECT_EQ(alpha, alpha2);
  EXPECT_EQ(tau, tau2);
  EXPECT_EQ(x, x2);
  T nan = NaN<T>();
  x[2] = NaN<T>();
  if constexpr (std::is_same_v<T, double>) {
    EXPECT_EQ(LAPACKE_dlarfg(n, &nan, x2.data(), 1, &tau), -2);
    EXPECT_EQ(LAPACKE_dlarfg(n, &alpha, x.data(), 1, &tau), -3);
    EXPECT_TRUE(std::isnan(LAPACKE_dlapy2(1, std::nan(""))) ||
                LAPACKE_dlapy2(1, std::nan("")) == -2);
    EXPECT_EQ(LAPACKE_dlapy2(std::nan(""), 1), -1);
    EXPECT_EQ(LAPACKE_dlapy3(1, 2, std::nan("")), -3);
    EXPECT_EQ(LAPACKE_dlapy3(2, 3, 6), 7);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    EXPECT_EQ(LAPACKE_zlarfg(n, &nan, x2.data(), 1, &tau), -2);
    EXPECT_EQ(LAPACKE_zlarfg(n, &alpha, x.data(), 1, &tau), -3);
    std::vector<T> v = {T(1, 2), T(3, -4)};
    EXPECT_EQ(LAPACKE_zlacgv(2, v.data(), 1), 0);
    EXPECT_EQ(v, (std::vector<T>{T(1, -2), T(3, 4)}));
    v[1] = NaN<T>();
    EXPECT_EQ(LAPACKE_zlacgv(2, v.data(), 1), -2);
  } else if constexpr (std::is_same_v<T, float>) {
    EXPECT_EQ(LAPACKE_slapy2(3, 4), 5);
    EXPECT_EQ(LAPACKE_slapy2(std::nanf(""), 4), -1);
  }
}

}  // namespace
}  // namespace hwyblas_test
