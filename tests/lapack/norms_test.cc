// Tests of the norms and of scaling and setting matrices: xLASSQ, xLANGE,
// xLANSY, xLANHE, xLASCL, xLASET, and their LAPACKE functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class NormsTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(NormsTest, ScalarTypes);

template <typename T>
void Lassq(hwyblas_int n, const std::vector<T>& x, hwyblas_int incx,
           RealT<T>* scale, RealT<T>* sumsq) {
  HWYBLAS_CALL(T, lassq, &n, P(x.data()), &incx, scale, sumsq);
}

// sqrt(sum of |x(i)|^2) with scaling, in double precision.
template <typename T>
double ReferenceNorm(const std::vector<T>& x) {
  double big = 0;
  for (const T& e : x) big = std::max(big, Abs(e));
  if (big == 0) return 0;
  double sum = 0;
  for (const T& e : x) sum += std::norm(static_cast<RefT<T>>(e) / big);
  return big * std::sqrt(sum);
}

// scale * sqrt(sumsq) is the norm of x, also for elements near the
// overflow and the underflow threshold, and accumulates onto the scale and
// sum of squares given on entry.
TYPED_TEST(NormsTest, Lassq) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(59);
  const R tiny = std::numeric_limits<R>::min();
  const R huge = std::numeric_limits<R>::max();
  for (const hwyblas_int incx : {1, 2, -1}) {
    for (const double scale_of_x :
         {1.0, static_cast<double>(huge) / 4, static_cast<double>(tiny) * 4,
          static_cast<double>(tiny) / 64 /* subnormal */}) {
      for (const bool mixed : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "incx=" << incx << " scale="
                                          << scale_of_x << " mixed=" << mixed);
        const hwyblas_int n = 17;
        std::vector<T> x(n * std::abs(incx));
        for (size_t i = 0; i < x.size(); ++i) {
          x[i] = rng.Next() * static_cast<R>(scale_of_x);
          // Mixed: also values of other magnitudes (1 and subnormal).
          if (mixed && i % 3 == 1) x[i] = rng.Next();
          if (mixed && i % 3 == 2) x[i] = T(tiny / 8);
        }
        std::vector<T> used(n);
        for (hwyblas_int i = 0; i < n; ++i) used[i] = x[i * std::abs(incx)];
        R scale = 0, sumsq = 1;
        Lassq(n, x, incx, &scale, &sumsq);
        const double expected = ReferenceNorm(used);
        // (Compared as scale / expected * sqrt(sumsq): near the overflow
        // threshold, the norm itself may not be representable.)
        const auto relative = [&] {
          return static_cast<double>(scale) / expected *
                 std::sqrt(static_cast<double>(sumsq));
        };
        EXPECT_NEAR(relative(), 1.0, 8 * n * Eps<T>());
        // Accumulate the same vector again: sqrt(2) times the norm.
        Lassq(n, x, incx, &scale, &sumsq);
        EXPECT_NEAR(relative(), std::sqrt(2.0), 16 * n * Eps<T>());
      }
    }
  }
  // Values on both sides of Blue's thresholds tbig = 2^floor((maxexponent -
  // digits + 1) / 2) and tsml = 2^ceil((minexponent - 1) / 2), so that two
  // accumulators contribute comparable amounts.
  using L = std::numeric_limits<R>;
  const int big_exp = (L::max_exponent - L::digits + 1) / 2;  // positive
  const int small_exp = -((1 - L::min_exponent) / 2);         // ceiling
  for (const int e : {big_exp, small_exp}) {
    const R t = std::ldexp(R(1), e);
    const std::vector<T> y = {T(t * R(1.5)), T(t * R(0.75)), T(t * R(-1.25)),
                              T(t * R(0.5))};
    R scale = 0, sumsq = 1;
    Lassq(4, y, 1, &scale, &sumsq);
    const double expected = static_cast<double>(t) * std::sqrt(4.625);
    EXPECT_NEAR(static_cast<double>(scale) / expected *
                    std::sqrt(static_cast<double>(sumsq)),
                1.0, 8 * Eps<T>())
        << "threshold 2^" << e;
  }

  // NaN propagates; empty x leaves scale and sumsq (with sumsq = 0 or
  // scale = 0 replaced by the empty sum).
  std::vector<T> x = {T(1), NaN<T>(), T(2)};
  R scale = 1, sumsq = 0;
  Lassq(3, x, 1, &scale, &sumsq);
  EXPECT_TRUE(std::isnan(sumsq) || std::isnan(scale));
  scale = 3;
  sumsq = 2;
  Lassq(0, x, 1, &scale, &sumsq);
  EXPECT_EQ(scale, 3);
  EXPECT_EQ(sumsq, 2);
  scale = 0;
  sumsq = 5;
  Lassq(0, x, 1, &scale, &sumsq);
  EXPECT_EQ(scale, 1);
  EXPECT_EQ(sumsq, 0);
}

template <typename T>
RealT<T> Lange(char norm, hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
               hwyblas_int lda) {
  std::vector<RealT<T>> work(std::max<hwyblas_int>(1, m), NAN);
  RealT<T> r = -1;
  if constexpr (std::is_same_v<T, float>) {
    r = slange_(&norm, &m, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, double>) {
    r = dlange_(&norm, &m, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    r = clange_(&norm, &m, &n, P(a.data()), &lda, work.data());
  } else {
    r = zlange_(&norm, &m, &n, P(a.data()), &lda, work.data());
  }
  return r;
}

// The four norms of the dense matrix d.
template <typename T>
double DenseNorm(char norm, const Dense<T>& d) {
  double value = 0;
  if (norm == 'M') {
    for (const auto& e : d.v) value = std::max(value, std::abs(e));
  } else if (norm == '1') {
    value = Norm1(d);
  } else if (norm == 'I') {
    value = Norm1(Adjoint(d));
  } else {
    double sum = 0;
    for (const auto& e : d.v) sum += std::norm(e);
    value = std::sqrt(sum);
  }
  return value;
}

TYPED_TEST(NormsTest, Lange) {
  using T = TypeParam;
  Random<T> rng(61);
  const hwyblas_int m = 13, n = 9, lda = 15;
  std::vector<T> a = rng.Vector(lda * n);
  for (hwyblas_int j = 0; j < n; ++j) At(a, m, j, lda) = NaN<T>();  // padding
  const Dense<T> d = ToDense(a, m, n, lda);
  for (const char norm : {'M', 'm', '1', 'O', 'o', 'I', 'i', 'F', 'f', 'E'}) {
    const char base = norm == 'm'                    ? 'M'
                      : (norm == 'O' || norm == 'o') ? '1'
                      : norm == 'i'                  ? 'I'
                      : (norm == 'f' || norm == 'E') ? 'F'
                                                     : norm;
    const double expected = DenseNorm(base, d);
    EXPECT_NEAR(Lange(norm, m, n, a, lda), expected,
                4 * (m + n) * Eps<T>() * expected)
        << norm;
  }
  EXPECT_EQ(Lange('M', 0, n, a, lda), 0);
  EXPECT_EQ(Lange('F', m, 0, a, lda), 0);
  // NaN propagates for every norm, wherever it is.
  for (const char norm : {'M', '1', 'I', 'F'}) {
    std::vector<T> b = a;
    At(b, 5, 3, lda) = NaN<T>();
    EXPECT_TRUE(std::isnan(Lange(norm, m, n, b, lda))) << norm;
  }
}

// xLANSY (hermitian = false) or xLANHE.
template <typename T>
RealT<T> Lansy(bool hermitian, char norm, char uplo, hwyblas_int n,
               const std::vector<T>& a, hwyblas_int lda) {
  std::vector<RealT<T>> work(std::max<hwyblas_int>(1, n), NAN);
  RealT<T> r = -1;
  if constexpr (std::is_same_v<T, float>) {
    r = slansy_(&norm, &uplo, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, double>) {
    r = dlansy_(&norm, &uplo, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    r = hermitian ? clanhe_(&norm, &uplo, &n, P(a.data()), &lda, work.data())
                  : clansy_(&norm, &uplo, &n, P(a.data()), &lda, work.data());
  } else {
    r = hermitian ? zlanhe_(&norm, &uplo, &n, P(a.data()), &lda, work.data())
                  : zlansy_(&norm, &uplo, &n, P(a.data()), &lda, work.data());
  }
  return r;
}

// The norms of symmetric and Hermitian matrices are those of the full
// matrices; the other triangle (and for xLANHE the imaginary parts of the
// diagonal) are not referenced.
TYPED_TEST(NormsTest, LansyLanhe) {
  using T = TypeParam;
  Random<T> rng(67);
  for (const bool hermitian : {false, true}) {
    if (hermitian && !IsComplex<T>::value) continue;
    const Shape shape = hermitian ? Shape::kHermitian : Shape::kSymmetric;
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {1, 2, 11}) {
        const hwyblas_int lda = n + 2;
        const std::vector<T> a = StructuredMatrix(shape, uplo, n, lda, rng);
        const Dense<T> d = DenseStructured(shape, uplo, 'N', a, n, lda);
        for (const char norm : {'M', '1', 'O', 'I', 'F', 'E'}) {
          const char base = norm == 'O' ? '1' : norm == 'E' ? 'F' : norm;
          const double expected = DenseNorm(base, d);
          EXPECT_NEAR(Lansy(hermitian, norm, uplo, n, a, lda), expected,
                      8 * n * Eps<T>() * expected)
              << hermitian << " " << uplo << " " << n << " " << norm;
        }
      }
    }
    EXPECT_EQ(Lansy(hermitian, 'F', 'U', 0, std::vector<T>(1), 1), 0);
  }
}

template <typename T>
hwyblas_int Lascl(char type, hwyblas_int kl, hwyblas_int ku, RealT<T> cfrom,
                  RealT<T> cto, hwyblas_int m, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, lascl, &type, &kl, &ku, &cfrom, &cto, &m, &n, P(a.data()),
               &lda, &info);
  return info;
}

// Whether xLASCL scales element (i, j) (0-based) of the storage of A.
bool Scaled(char type, hwyblas_int kl, hwyblas_int ku, hwyblas_int m,
            hwyblas_int n, hwyblas_int i, hwyblas_int j) {
  const hwyblas_int i1 = i + 1, j1 = j + 1;  // 1-based
  switch (type) {
    case 'G':
      return i < m;
    case 'L':
      return i1 >= j1 && i < m;
    case 'U':
      return i1 <= std::min(j1, m);
    case 'H':
      return i1 <= std::min(j1 + 1, m);
    case 'B':
      return i1 <= std::min(kl + 1, n + 1 - j1);
    case 'Q':
      return i1 >= std::max<hwyblas_int>(ku + 2 - j1, 1) && i1 <= ku + 1;
    default:  // 'Z'
      return i1 >= std::max(kl + ku + 2 - j1, kl + 1) &&
             i1 <= std::min(2 * kl + ku + 1, kl + ku + 1 + m - j1);
  }
}

// xLASCL scales exactly the elements of its storage type, by cto / cfrom,
// also when cto / cfrom itself over- or underflows.
TYPED_TEST(NormsTest, Lascl) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(71);
  const R big = std::numeric_limits<R>::max() / 8;
  const R small = std::numeric_limits<R>::min() * 8;
  struct Factor {
    R cfrom, cto;
  };
  const Factor factors[] = {{2, 3},       {-4, 1}, {small, big},
                            {big, small}, {1, 0},  {3, 3}};
  for (const char type : {'G', 'L', 'U', 'H', 'B', 'Q', 'Z'}) {
    const hwyblas_int m = 7, n = (type == 'B' || type == 'Q') ? 7 : 6;
    const hwyblas_int kl = 2, ku = (type == 'B' || type == 'Q') ? 2 : 1;
    const hwyblas_int lda = (type == 'Z') ? 2 * kl + ku + 2 : m + 1;
    for (const Factor& f : factors) {
      SCOPED_TRACE(::testing::Message()
                   << type << " " << f.cfrom << " -> " << f.cto);
      // Values whose scaled values are representable.
      const bool up = std::abs(f.cto) > std::abs(f.cfrom) * 1e3;
      std::vector<T> a = rng.Vector(lda * n);
      for (T& x : a) x = x * (up ? small : f.cfrom > 1e3 ? big : R(1));
      const std::vector<T> a0 = a;
      ASSERT_EQ(Lascl(type, kl, ku, f.cfrom, f.cto, m, n, a, lda), 0);
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < lda; ++i) {
          const T x0 = At(a0, i, j, lda), x = At(a, i, j, lda);
          if (!Scaled(type, kl, ku, m, n, i, j)) {
            ASSERT_EQ(x, x0) << i << ", " << j;
            continue;
          }
          const RefT<T> expected = static_cast<RefT<T>>(x0) /
                                   static_cast<double>(f.cfrom) *
                                   static_cast<double>(f.cto);
          ASSERT_LE(
              std::abs(static_cast<RefT<T>>(x) - expected),
              4 * Eps<T>() * std::abs(expected) +
                  static_cast<double>(std::numeric_limits<R>::denorm_min()))
              << i << ", " << j;
        }
      }
    }
  }
  // Invalid arguments.
  std::vector<T> a(64, T(1));
  const std::string name = Name<T>("LASCL");
  ErrorCapture errors;
  EXPECT_EQ(Lascl<T>('X', 0, 0, 1, 2, 4, 4, a, 4), -1);
  EXPECT_EQ(Lascl<T>('G', 0, 0, 0, 2, 4, 4, a, 4), -4);
  EXPECT_EQ(Lascl<T>('G', 0, 0, NAN, 2, 4, 4, a, 4), -4);
  EXPECT_EQ(Lascl<T>('G', 0, 0, 1, NAN, 4, 4, a, 4), -5);
  EXPECT_EQ(Lascl<T>('G', 0, 0, 1, 2, -1, 4, a, 4), -6);
  EXPECT_EQ(Lascl<T>('B', 1, 1, 1, 2, 4, 3, a, 4), -7);
  EXPECT_EQ(Lascl<T>('G', 0, 0, 1, 2, 4, 4, a, 3), -9);
  EXPECT_EQ(Lascl<T>('B', 4, 4, 1, 2, 4, 4, a, 8), -2);
  EXPECT_EQ(Lascl<T>('Q', 1, 2, 1, 2, 4, 4, a, 8), -3);
  EXPECT_EQ(Lascl<T>('Z', 1, 1, 1, 2, 4, 4, a, 3), -9);
  EXPECT_EQ(
      errors.Take(),
      (std::vector<std::string>{
          name + ":1", name + ":4", name + ":4", name + ":5", name + ":6",
          name + ":7", name + ":9", name + ":2", name + ":3", name + ":9"}));
  EXPECT_EQ(a, std::vector<T>(64, T(1)));
}

TYPED_TEST(NormsTest, Laset) {
  using T = TypeParam;
  const hwyblas_int m = 4, n = 6, lda = 5;
  const T alpha = MakeScalar<T>(2, -1), beta = MakeScalar<T>(-3, 0.5);
  for (const char uplo : {'U', 'L', 'A'}) {
    std::vector<T> a(lda * n, T(9));
    HWYBLAS_CALL(T, laset, &uplo, &m, &n, P(&alpha), P(&beta), P(a.data()),
                 &lda);
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = 0; i < lda; ++i) {
        T expected = T(9);
        if (i < m) {
          if (i == j) {
            expected = beta;
          } else if (uplo == 'U' ? i < j : uplo == 'L' ? i > j : true) {
            expected = alpha;
          }
        }
        EXPECT_EQ(At(a, i, j, lda), expected) << uplo << " " << i << ", " << j;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
RealT<T> LapackeLange(int layout, char norm, lapack_int m, lapack_int n,
                      const T* a, lapack_int lda) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_slange(layout, norm, m, n, a, lda);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dlange(layout, norm, m, n, a, lda);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_clange(layout, norm, m, n, a, lda);
  } else {
    return LAPACKE_zlange(layout, norm, m, n, a, lda);
  }
}

template <typename T>
RealT<T> LapackeLansy(bool hermitian, int layout, char norm, char uplo,
                      lapack_int n, const T* a, lapack_int lda) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_slansy(layout, norm, uplo, n, a, lda);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dlansy(layout, norm, uplo, n, a, lda);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return hermitian ? LAPACKE_clanhe(layout, norm, uplo, n, a, lda)
                     : LAPACKE_clansy(layout, norm, uplo, n, a, lda);
  } else {
    return hermitian ? LAPACKE_zlanhe(layout, norm, uplo, n, a, lda)
                     : LAPACKE_zlansy(layout, norm, uplo, n, a, lda);
  }
}

// Row-major matrices have the norms of the same matrices in column-major
// layout; the scaling and setting functions change the same elements.
TYPED_TEST(NormsTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(73);
  const lapack_int m = 8, n = 5, ld = 9;
  const std::vector<T> col = rng.Vector(ld * ld);
  std::vector<T> row(col.size(), NaN<T>());
  for (lapack_int j = 0; j < ld; ++j) {
    for (lapack_int i = 0; i < ld; ++i) row[i * ld + j] = At(col, i, j, ld);
  }
  for (const char norm : {'M', '1', 'O', 'I', 'F'}) {
    // (The Frobenius norm of the transpose sums in another order.)
    const R c = LapackeLange(LAPACK_COL_MAJOR, norm, m, n, col.data(), ld);
    EXPECT_NEAR(LapackeLange(LAPACK_ROW_MAJOR, norm, m, n, row.data(), ld), c,
                4 * (m + n) * Eps<T>() * c)
        << norm;
    for (const bool hermitian : {false, true}) {
      if (hermitian && !IsComplex<T>::value) continue;
      for (const char uplo : {'U', 'L'}) {
        // The hermitian norms read only real parts of the diagonal, which
        // are the same in both layouts.
        const R r = LapackeLansy(hermitian, LAPACK_ROW_MAJOR, norm, uplo, n,
                                 row.data(), ld);
        const R c = LapackeLansy(hermitian, LAPACK_COL_MAJOR, norm, uplo, n,
                                 col.data(), ld);
        EXPECT_NEAR(r, c, 4 * n * Eps<T>() * c) << hermitian << uplo << norm;
      }
    }
  }

  // xLASCL and xLASET: the same changes in both layouts.
  for (const char type : {'G', 'L', 'U', 'H'}) {
    std::vector<T> c = col, r = row;
    lapack_int info_c, info_r;
    if constexpr (std::is_same_v<T, double>) {
      info_c = LAPACKE_dlascl(LAPACK_COL_MAJOR, type, 0, 0, 3, 5, m, n,
                              c.data(), ld);
      info_r = LAPACKE_dlascl(LAPACK_ROW_MAJOR, type, 0, 0, 3, 5, m, n,
                              r.data(), ld);
    } else if constexpr (std::is_same_v<T, float>) {
      info_c = LAPACKE_slascl(LAPACK_COL_MAJOR, type, 0, 0, 3, 5, m, n,
                              c.data(), ld);
      info_r = LAPACKE_slascl(LAPACK_ROW_MAJOR, type, 0, 0, 3, 5, m, n,
                              r.data(), ld);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      info_c = LAPACKE_clascl(LAPACK_COL_MAJOR, type, 0, 0, 3, 5, m, n,
                              c.data(), ld);
      info_r = LAPACKE_clascl(LAPACK_ROW_MAJOR, type, 0, 0, 3, 5, m, n,
                              r.data(), ld);
    } else {
      info_c = LAPACKE_zlascl(LAPACK_COL_MAJOR, type, 0, 0, 3, 5, m, n,
                              c.data(), ld);
      info_r = LAPACKE_zlascl(LAPACK_ROW_MAJOR, type, 0, 0, 3, 5, m, n,
                              r.data(), ld);
    }
    ASSERT_EQ(info_c, 0);
    ASSERT_EQ(info_r, 0);
    for (lapack_int j = 0; j < n; ++j) {
      for (lapack_int i = 0; i < m; ++i) {
        ASSERT_EQ(r[i * ld + j], At(c, i, j, ld)) << type << i << ", " << j;
      }
    }
  }
  for (const char uplo : {'U', 'L', 'A'}) {
    std::vector<T> c = col, r = row;
    const T alpha = T(2), beta = T(-1);
    if constexpr (std::is_same_v<T, double>) {
      LAPACKE_dlaset(LAPACK_COL_MAJOR, uplo, m, n, alpha, beta, c.data(), ld);
      LAPACKE_dlaset(LAPACK_ROW_MAJOR, uplo, m, n, alpha, beta, r.data(), ld);
    } else if constexpr (std::is_same_v<T, float>) {
      LAPACKE_slaset(LAPACK_COL_MAJOR, uplo, m, n, alpha, beta, c.data(), ld);
      LAPACKE_slaset(LAPACK_ROW_MAJOR, uplo, m, n, alpha, beta, r.data(), ld);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      LAPACKE_claset(LAPACK_COL_MAJOR, uplo, m, n, alpha, beta, c.data(), ld);
      LAPACKE_claset(LAPACK_ROW_MAJOR, uplo, m, n, alpha, beta, r.data(), ld);
    } else {
      LAPACKE_zlaset(LAPACK_COL_MAJOR, uplo, m, n, alpha, beta, c.data(), ld);
      LAPACKE_zlaset(LAPACK_ROW_MAJOR, uplo, m, n, alpha, beta, r.data(), ld);
    }
    for (lapack_int j = 0; j < n; ++j) {
      for (lapack_int i = 0; i < m; ++i) {
        ASSERT_EQ(r[i * ld + j], At(c, i, j, ld)) << uplo << i << ", " << j;
      }
    }
  }
}

TYPED_TEST(NormsTest, LapackeErrors) {
  using T = TypeParam;
  std::vector<T> a(25, T(1));
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeLange<T>(0, 'M', 5, 5, a.data(), 5), -1);
  EXPECT_EQ(LapackeLange<T>(LAPACK_ROW_MAJOR, 'M', 5, 5, a.data(), 4), -6);
  EXPECT_EQ(LapackeLansy<T>(false, LAPACK_ROW_MAJOR, 'M', 'U', 5, a.data(), 4),
            -6);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("lange") +
                "\nWrong parameter 6 in " + LapackeName<T>("lange_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("lansy_work") +
                "\n");
  // NaN: in A for the norms (only in the triangle for xLANSY), and in the
  // scaled part for xLASCL.
  a[1] = NaN<T>();  // (1, 0): lower triangle in column-major layout
  EXPECT_EQ(LapackeLange<T>(LAPACK_COL_MAJOR, 'M', 5, 5, a.data(), 5), -5);
  EXPECT_EQ(LapackeLansy<T>(false, LAPACK_COL_MAJOR, 'M', 'L', 5, a.data(), 5),
            -5);
  EXPECT_EQ(LapackeLansy<T>(false, LAPACK_COL_MAJOR, 'M', 'U', 5, a.data(), 5),
            1);
  lapack_int upper = -99, lower = -99;
  std::vector<T> b = a;
  if constexpr (std::is_same_v<T, double>) {
    upper =
        LAPACKE_dlascl(LAPACK_COL_MAJOR, 'U', 0, 0, 1, 2, 5, 5, b.data(), 5);
    lower =
        LAPACKE_dlascl(LAPACK_COL_MAJOR, 'L', 0, 0, 1, 2, 5, 5, b.data(), 5);
  } else if constexpr (std::is_same_v<T, float>) {
    upper =
        LAPACKE_slascl(LAPACK_COL_MAJOR, 'U', 0, 0, 1, 2, 5, 5, b.data(), 5);
    lower =
        LAPACKE_slascl(LAPACK_COL_MAJOR, 'L', 0, 0, 1, 2, 5, 5, b.data(), 5);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    upper =
        LAPACKE_clascl(LAPACK_COL_MAJOR, 'U', 0, 0, 1, 2, 5, 5, b.data(), 5);
    lower =
        LAPACKE_clascl(LAPACK_COL_MAJOR, 'L', 0, 0, 1, 2, 5, 5, b.data(), 5);
  } else {
    upper =
        LAPACKE_zlascl(LAPACK_COL_MAJOR, 'U', 0, 0, 1, 2, 5, 5, b.data(), 5);
    lower =
        LAPACKE_zlascl(LAPACK_COL_MAJOR, 'L', 0, 0, 1, 2, 5, 5, b.data(), 5);
  }
  EXPECT_EQ(upper, 0);
  EXPECT_EQ(lower, -9);
}

}  // namespace
}  // namespace hwyblas_test
