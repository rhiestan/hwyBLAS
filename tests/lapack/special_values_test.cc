// Drivers on matrices with NaN and Inf: they must return, and stay within
// their arrays. The reference LAPACK reads or writes out of bounds in these
// cases (see "Fixed reference bugs" in doc/deviations.md); the guard elements
// after each output array catch it also without a sanitizer, which finds it
// where it happens (the sanitizer CI job).
#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename R>
class SpecialValuesTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SpecialValuesTest, RealTypes);

// An n x n matrix of numbers in (-1, 1) with `value` at (n / 2, n / 2).
template <typename R>
std::vector<R> MatrixWith(hwyblas_int n, R value) {
  std::vector<R> a(static_cast<size_t>(n) * n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      a[i + j * n] = static_cast<R>(std::sin(1.7 * i + 2.3 * j + 0.5));
    }
  }
  a[(n / 2) + (n / 2) * n] = value;
  return a;
}

// Array of n elements followed by 4 guard elements.
template <typename R>
std::vector<R> Guarded(hwyblas_int n, R fill = R(0)) {
  std::vector<R> v(static_cast<size_t>(n) + 4, fill);
  for (size_t i = static_cast<size_t>(n); i < v.size(); ++i) v[i] = R(12345);
  return v;
}

template <typename R>
void ExpectGuardsIntact(const std::vector<R>& v, hwyblas_int n,
                        const char* what) {
  for (size_t i = static_cast<size_t>(n); i < v.size(); ++i) {
    EXPECT_EQ(v[i], R(12345)) << what << ": element " << i + 1 << " of " << n;
  }
}

// xGEBAL leaves ILO and IHI undefined when it finds a NaN (and returns
// INFO = -3); they are the limits found so far.
TYPED_TEST(SpecialValuesTest, GebalNanSetsIloIhi) {
  using R = TypeParam;
  ErrorCapture errors;
  for (hwyblas_int n = 1; n <= 5; ++n) {
    for (const char job : {'S', 'B'}) {
      std::vector<R> a = MatrixWith<R>(n, std::numeric_limits<R>::quiet_NaN());
      std::vector<R> scale(static_cast<size_t>(n));
      hwyblas_int ilo = -77, ihi = -77, info = -99;
      HWYBLAS_CALL_REAL(R, gebal, &job, &n, a.data(), &n, &ilo, &ihi,
                        scale.data(), &info);
      // -3: the NaN was found while scaling; 0: the permutation isolated all
      // eigenvalues before.
      EXPECT_TRUE(info == -3 || info == 0) << "n=" << n << " job=" << job;
      EXPECT_GE(ilo, 1) << "n=" << n << " job=" << job;
      EXPECT_LE(ihi, n) << "n=" << n << " job=" << job;
      EXPECT_LE(ilo, ihi + 1) << "n=" << n << " job=" << job;
    }
  }
}

// xGEEV, xGEEVX: a NaN in A is no illegal argument (INFO < 0), and the
// eigenvalues are not rescaled out of bounds.
TYPED_TEST(SpecialValuesTest, GeevWithNanOrInf) {
  using R = TypeParam;
  ErrorCapture errors;
  for (const R value : {std::numeric_limits<R>::quiet_NaN(),
                        std::numeric_limits<R>::infinity()}) {
    for (hwyblas_int n = 1; n <= 5; ++n) {
      std::vector<R> a = MatrixWith<R>(n, value);
      std::vector<R> wr = Guarded<R>(n), wi = Guarded<R>(n);
      const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
      std::vector<R> vl(nn), vr(nn);
      std::vector<R> work(static_cast<size_t>(8) * n + 64);
      hwyblas_int lwork = static_cast<hwyblas_int>(work.size()), info = -99;
      const char v = 'V';
      HWYBLAS_CALL_REAL(R, geev, &v, &v, &n, a.data(), &n, wr.data(), wi.data(),
                        vl.data(), &n, vr.data(), &n, work.data(), &lwork,
                        &info);
      EXPECT_GE(info, 0) << "geev n=" << n << " value=" << value;
      ExpectGuardsIntact(wr, n, "geev wr");
      ExpectGuardsIntact(wi, n, "geev wi");

      a = MatrixWith<R>(n, value);
      wr = Guarded<R>(n);
      wi = Guarded<R>(n);
      std::vector<R> scale = Guarded<R>(n), rconde = Guarded<R>(n),
                     rcondv = Guarded<R>(n);
      std::vector<R> work2(nn + static_cast<size_t>(6) * n + 64);
      std::vector<hwyblas_int> iwork(static_cast<size_t>(2) * n + 8);
      hwyblas_int ilo = 0, ihi = 0;
      R abnrm = 0;
      const char balanc = 'B', sense = 'N';
      lwork = static_cast<hwyblas_int>(work2.size());
      HWYBLAS_CALL_REAL(R, geevx, &balanc, &v, &v, &sense, &n, a.data(), &n,
                        wr.data(), wi.data(), vl.data(), &n, vr.data(), &n,
                        &ilo, &ihi, scale.data(), &abnrm, rconde.data(),
                        rcondv.data(), work2.data(), &lwork, iwork.data(),
                        &info);
      EXPECT_GE(info, 0) << "geevx n=" << n << " value=" << value;
      ExpectGuardsIntact(wr, n, "geevx wr");
      ExpectGuardsIntact(wi, n, "geevx wi");
    }
  }
}

// xSYEVX and xSTEVX-based range 'I': for a matrix with Inf, INFO > 0 from
// xSTEBZ (up to 4) made the rescaling of W cover more than its N elements.
TYPED_TEST(SpecialValuesTest, SyevxRescalesOnlyN) {
  using R = TypeParam;
  ErrorCapture errors;
  for (hwyblas_int n = 1; n <= 4; ++n) {
    for (const char jobz : {'N', 'V'}) {
      for (const char uplo : {'U', 'L'}) {
        // Inf everywhere (INFO = 4 from xSTEBZ), as in the case that overran.
        std::vector<R> a(static_cast<size_t>(n) * static_cast<size_t>(n),
                         std::numeric_limits<R>::infinity());
        std::vector<R> w = Guarded<R>(n);
        std::vector<R> z(static_cast<size_t>(n) * n);
        std::vector<R> work(static_cast<size_t>(8) * n + 64);
        std::vector<hwyblas_int> iwork(static_cast<size_t>(5) * n + 8),
            ifail(static_cast<size_t>(n) + 4);
        const char range = 'I';
        const hwyblas_int il = 1, iu = 1,
                          lwork = static_cast<hwyblas_int>(work.size());
        const R vl = 0, vu = 0, abstol = 0;
        hwyblas_int m = 0, info = -99;
        HWYBLAS_CALL_REAL(R, syevx, &jobz, &range, &uplo, &n, a.data(), &n, &vl,
                          &vu, &il, &iu, &abstol, &m, w.data(), z.data(), &n,
                          work.data(), &lwork, iwork.data(), ifail.data(),
                          &info);
        ExpectGuardsIntact(w, n, "syevx w");
      }
    }
  }
}

// xGETC2 finds no pivot in a matrix of NaNs: no swap with an undefined index.
TYPED_TEST(SpecialValuesTest, Getc2OfNans) {
  using R = TypeParam;
  ErrorCapture errors;
  for (hwyblas_int n = 1; n <= 5; ++n) {
    std::vector<R> a(static_cast<size_t>(n) * n,
                     std::numeric_limits<R>::quiet_NaN());
    std::vector<hwyblas_int> ipiv(static_cast<size_t>(n)),
        jpiv(static_cast<size_t>(n));
    hwyblas_int info = -99;
    HWYBLAS_CALL_REAL(R, getc2, &n, a.data(), &n, ipiv.data(), jpiv.data(),
                      &info);
    for (hwyblas_int i = 0; i < n; ++i) {
      EXPECT_GE(ipiv[i], 1) << "n=" << n;
      EXPECT_LE(ipiv[i], n) << "n=" << n;
      EXPECT_GE(jpiv[i], 1) << "n=" << n;
      EXPECT_LE(jpiv[i], n) << "n=" << n;
    }
  }
}

// xGGEV: a NaN in ALPHAI is a real eigenvalue, not the first of a complex
// pair: the normalization does not touch column N + 1 of VL and VR.
TYPED_TEST(SpecialValuesTest, GgevNormalizesNanAsReal) {
  using R = TypeParam;
  ErrorCapture errors;
  for (hwyblas_int n = 1; n <= 4; ++n) {
    std::vector<R> a = MatrixWith<R>(n, std::numeric_limits<R>::quiet_NaN());
    std::vector<R> b = MatrixWith<R>(n, std::numeric_limits<R>::quiet_NaN());
    std::vector<R> alphar = Guarded<R>(n), alphai = Guarded<R>(n),
                   beta = Guarded<R>(n);
    // One column beyond N, which stays untouched.
    std::vector<R> vl(static_cast<size_t>(n) * (n + 1), R(12345)),
        vr(static_cast<size_t>(n) * (n + 1), R(12345));
    std::vector<R> work(static_cast<size_t>(8) * n + 64);
    const hwyblas_int lwork = static_cast<hwyblas_int>(work.size());
    hwyblas_int info = -99;
    const char v = 'V';
    HWYBLAS_CALL_REAL(R, ggev, &v, &v, &n, a.data(), &n, b.data(), &n,
                      alphar.data(), alphai.data(), beta.data(), vl.data(), &n,
                      vr.data(), &n, work.data(), &lwork, &info);
    for (hwyblas_int i = 0; i < n; ++i) {
      EXPECT_EQ(vl[static_cast<size_t>(n) * n + i], R(12345)) << "vl n=" << n;
      EXPECT_EQ(vr[static_cast<size_t>(n) * n + i], R(12345)) << "vr n=" << n;
    }
    ExpectGuardsIntact(alphar, n, "ggev alphar");
    ExpectGuardsIntact(alphai, n, "ggev alphai");
    ExpectGuardsIntact(beta, n, "ggev beta");
  }
}

}  // namespace
}  // namespace hwyblas_test
