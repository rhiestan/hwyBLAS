// Tests for the CBLAS interface.
//
// The Fortran-style routines are tested elsewhere; these tests check the
// mapping onto them. For every routine and precision:
//   - a column-major CBLAS call must give exactly the result of the Fortran
//     routine on the same data;
//   - a row-major CBLAS call on the row-major storage of the same logical
//     matrices must give the same result up to rounding. The row-major
//     storage of a matrix is the column-major storage of its transpose; for
//     band and packed matrices that of the transpose with the other
//     triangle (or kl and ku swapped), as defined by the reference CBLAS.
// Invalid arguments must be reported with their CBLAS argument positions.
#include "hwyblas/cblas.h"

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class CblasTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CblasTest, ScalarTypes);

// Calls cblas_<s/d/c/z>NAME(args...) according to T.
#define CBLAS_CALL(T, NAME, ...)                                   \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      cblas_s##NAME(__VA_ARGS__);                                  \
    } else if constexpr (std::is_same_v<T, double>) {              \
      cblas_d##NAME(__VA_ARGS__);                                  \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      cblas_c##NAME(__VA_ARGS__);                                  \
    } else {                                                       \
      cblas_z##NAME(__VA_ARGS__);                                  \
    }                                                              \
  } while (0)

// Calls REAL_NAME for real T, COMPLEX_NAME for complex T (e.g. symv, hemv),
// for both Fortran and CBLAS routines.
#define CALL_REAL_OR_COMPLEX(T, REAL_MACRO, REAL_NAME, COMPLEX_MACRO, \
                             COMPLEX_NAME, ...)                       \
  do {                                                                \
    if constexpr (IsComplex<T>::value) {                              \
      COMPLEX_MACRO(T, COMPLEX_NAME, __VA_ARGS__);                    \
    } else {                                                          \
      REAL_MACRO(T, REAL_NAME, __VA_ARGS__);                          \
    }                                                                 \
  } while (0)

#define CBLAS_CALL_REAL(T, NAME, ...)                 \
  do {                                                \
    if constexpr (std::is_same_v<T, float>) {         \
      cblas_s##NAME(__VA_ARGS__);                     \
    } else if constexpr (std::is_same_v<T, double>) { \
      cblas_d##NAME(__VA_ARGS__);                     \
    }                                                 \
  } while (0)

#define CBLAS_CALL_COMPLEX(T, NAME, ...)                            \
  do {                                                              \
    if constexpr (std::is_same_v<T, std::complex<float>>) {         \
      cblas_c##NAME(__VA_ARGS__);                                   \
    } else if constexpr (std::is_same_v<T, std::complex<double>>) { \
      cblas_z##NAME(__VA_ARGS__);                                   \
    }                                                               \
  } while (0)

// CBLAS passes real scalars by value and complex ones by pointer.
template <typename T>
auto S(const T& v) {
  if constexpr (IsComplex<T>::value) {
    return static_cast<const void*>(&v);
  } else {
    return v;
  }
}

// ---------------------------------------------------------------------------
// Logical matrices and their storage

// A dense rows x cols matrix (column-major, no padding). NaN marks elements
// that are not stored.
template <typename T>
struct Mat {
  int rows = 0, cols = 0;
  std::vector<T> v;
  Mat(int r, int c, T fill = T(0))
      : rows(r), cols(c), v(static_cast<size_t>(r) * c, fill) {}
  T& operator()(int i, int j) { return v[i + static_cast<size_t>(j) * rows]; }
  const T& operator()(int i, int j) const {
    return v[i + static_cast<size_t>(j) * rows];
  }
};

template <typename T>
Mat<T> RandomMat(int rows, int cols, Random<T>& rng) {
  Mat<T> m(rows, cols);
  for (T& x : m.v) x = rng.Next();
  return m;
}

template <typename T>
Mat<T> Transpose(const Mat<T>& a) {
  Mat<T> t(a.cols, a.rows);
  for (int j = 0; j < a.cols; ++j) {
    for (int i = 0; i < a.rows; ++i) t(j, i) = a(i, j);
  }
  return t;
}

// A structured n x n matrix (see StructuredMatrix) as a Mat.
template <typename T>
Mat<T> Structured(Shape shape, char uplo, int n, Random<T>& rng,
                  double offdiag_scale = 1.0) {
  Mat<T> m(n, n);
  m.v = StructuredMatrix(shape, uplo, n, std::max(n, 1), rng, offdiag_scale);
  return m;
}

char FlipUplo(char uplo) { return uplo == 'L' ? 'U' : 'L'; }

// Column-major storage with leading dimension ld (padding: sentinel).
template <typename T>
std::vector<T> General(const Mat<T>& a, int ld) {
  std::vector<T> s(static_cast<size_t>(ld) * std::max(a.cols, 1),
                   MakeScalar<T>(7, 7));
  for (int j = 0; j < a.cols; ++j) {
    for (int i = 0; i < a.rows; ++i)
      s[i + static_cast<size_t>(j) * ld] = a(i, j);
  }
  return s;
}

// The matrix stored in General(a, ld) (rows x cols).
template <typename T>
Mat<T> FromGeneral(const std::vector<T>& s, int rows, int cols, int ld) {
  Mat<T> a(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) a(i, j) = s[i + static_cast<size_t>(j) * ld];
  }
  return a;
}

// Band storage with kl sub- and ku super-diagonals, leading dimension ldab.
template <typename T>
std::vector<T> Band(const Mat<T>& a, int kl, int ku, int ldab) {
  std::vector<T> s(static_cast<size_t>(ldab) * std::max(a.cols, 1), NaN<T>());
  for (int j = 0; j < a.cols; ++j) {
    for (int i = std::max(0, j - ku); i < std::min(a.rows, j + kl + 1); ++i) {
      s[(ku + i - j) + static_cast<size_t>(j) * ldab] = a(i, j);
    }
  }
  return s;
}

// Packed storage of the uplo triangle of the n x n matrix a.
template <typename T>
std::vector<T> Packed(const Mat<T>& a, char uplo) {
  std::vector<T> s;
  for (int j = 0; j < a.cols; ++j) {
    for (int i = (uplo == 'L') ? j : 0; i < ((uplo == 'L') ? a.rows : j + 1);
         ++i) {
      s.push_back(a(i, j));
    }
  }
  return s;
}

// The uplo triangle of the n x n matrix in packed storage (NaN elsewhere).
template <typename T>
Mat<T> FromPacked(const std::vector<T>& s, int n, char uplo) {
  Mat<T> a(n, n, NaN<T>());
  size_t k = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = (uplo == 'L') ? j : 0; i < ((uplo == 'L') ? n : j + 1); ++i) {
      a(i, j) = s[k++];
    }
  }
  return a;
}

// Sets the elements outside the band to zero.
template <typename T>
Mat<T> InBand(Mat<T> a, int kl, int ku) {
  for (int j = 0; j < a.cols; ++j) {
    for (int i = 0; i < a.rows; ++i) {
      if (i - j > kl || j - i > ku) a(i, j) = T(0);
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// Comparisons (NaN equals NaN: elements that must stay untouched)

template <typename T>
bool BothNaN(T a, T b) {
  return std::isnan(std::real(a)) && std::isnan(std::real(b));
}

template <typename T>
void ExpectIdentical(const std::vector<T>& actual,
                     const std::vector<T>& expected, const std::string& what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(memcmp(&actual[i], &expected[i], sizeof(T)), 0)
        << what << ": element " << i << " is " << actual[i] << ", expected "
        << expected[i];
  }
}

template <typename T>
void ExpectNear(const std::vector<T>& actual, const std::vector<T>& expected,
                const std::string& what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  const double tol = 1000 * Eps<T>();
  size_t failures = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (BothNaN(actual[i], expected[i])) continue;
    const double err = Abs(actual[i] - expected[i]);
    if (!(err <= tol * (1 + Abs(expected[i]))) && failures++ < 5) {
      ADD_FAILURE() << what << ": element " << i << " is " << actual[i]
                    << ", expected " << expected[i];
    }
  }
}

// The 0-based CBLAS index for a 1-based Fortran index (0 for "none").
size_t ZeroBased(int i) { return i > 0 ? static_cast<size_t>(i - 1) : 0; }

// Random vector with increment inc (gaps hold the StridedVector sentinel).
template <typename T>
std::vector<T> Vec(int n, int inc, Random<T>& rng) {
  return StridedVector<T>(n, inc, rng).data;
}

CBLAS_TRANSPOSE CblasOp(char t) {
  return t == 'N' ? CblasNoTrans : t == 'T' ? CblasTrans : CblasConjTrans;
}
CBLAS_UPLO CblasUplo(char u) { return u == 'U' ? CblasUpper : CblasLower; }
CBLAS_DIAG CblasDiag(char d) { return d == 'U' ? CblasUnit : CblasNonUnit; }
CBLAS_SIDE CblasSide(char s) { return s == 'L' ? CblasLeft : CblasRight; }

// ---------------------------------------------------------------------------
// Level 1

TYPED_TEST(CblasTest, Level1) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(1);
  const int n = 13;
  for (int inc : {1, -2}) {
    const std::vector<T> x = Vec(n, inc, rng), y = Vec(n, 3, rng);
    const int incy = 3;
    const T alpha = MakeScalar<T>(0.75, -0.5);

    // axpy, scal, copy, swap: same as the Fortran routines.
    std::vector<T> y1 = y, y2 = y;
    CBLAS_CALL(T, axpy, n, S(alpha), x.data(), inc, y1.data(), incy);
    HWYBLAS_CALL(T, axpy, &n, P(&alpha), P(x.data()), &inc, P(y2.data()),
                 &incy);
    ExpectIdentical(y1, y2, "axpy");
    std::vector<T> x1 = x, x2 = x;
    CBLAS_CALL(T, scal, n, S(alpha), x1.data(), inc);
    HWYBLAS_CALL(T, scal, &n, P(&alpha), P(x2.data()), &inc);
    ExpectIdentical(x1, x2, "scal");
    y1 = y;
    y2 = y;
    CBLAS_CALL(T, copy, n, x.data(), inc, y1.data(), incy);
    HWYBLAS_CALL(T, copy, &n, P(x.data()), &inc, P(y2.data()), &incy);
    ExpectIdentical(y1, y2, "copy");
    x1 = x;
    x2 = x;
    y1 = y;
    y2 = y;
    CBLAS_CALL(T, swap, n, x1.data(), inc, y1.data(), incy);
    HWYBLAS_CALL(T, swap, &n, P(x2.data()), &inc, P(y2.data()), &incy);
    ExpectIdentical(x1, x2, "swap x");
    ExpectIdentical(y1, y2, "swap y");

    // Reductions.
    const T* xd = x.data();
    if constexpr (IsComplex<T>::value) {
      using C =
          std::conditional_t<std::is_same_v<R, float>, hwyblas_complex_float,
                             hwyblas_complex_double>;
      C u1, u2, c1, c2, sub;
      if constexpr (std::is_same_v<R, float>) {
        u1 = cblas_cdotu(n, xd, inc, y.data(), incy);
        c1 = cblas_cdotc(n, xd, inc, y.data(), incy);
        u2 = cdotu_(&n, P(xd), &inc, P(y.data()), &incy);
        c2 = cdotc_(&n, P(xd), &inc, P(y.data()), &incy);
        cblas_cdotu_sub(n, xd, inc, y.data(), incy, &sub);
        EXPECT_EQ(sub.real, u2.real);
        cblas_cdotc_sub(n, xd, inc, y.data(), incy, &sub);
        EXPECT_EQ(sub.imag, c2.imag);
        EXPECT_EQ(cblas_scnrm2(n, xd, inc), scnrm2_(&n, P(xd), &inc));
        EXPECT_EQ(cblas_scasum(n, xd, inc), scasum_(&n, P(xd), &inc));
        EXPECT_EQ(cblas_icamax(n, xd, inc),
                  ZeroBased(icamax_(&n, P(xd), &inc)));
      } else {
        u1 = cblas_zdotu(n, xd, inc, y.data(), incy);
        c1 = cblas_zdotc(n, xd, inc, y.data(), incy);
        u2 = zdotu_(&n, P(xd), &inc, P(y.data()), &incy);
        c2 = zdotc_(&n, P(xd), &inc, P(y.data()), &incy);
        cblas_zdotu_sub(n, xd, inc, y.data(), incy, &sub);
        EXPECT_EQ(sub.real, u2.real);
        cblas_zdotc_sub(n, xd, inc, y.data(), incy, &sub);
        EXPECT_EQ(sub.imag, c2.imag);
        EXPECT_EQ(cblas_dznrm2(n, xd, inc), dznrm2_(&n, P(xd), &inc));
        EXPECT_EQ(cblas_dzasum(n, xd, inc), dzasum_(&n, P(xd), &inc));
        EXPECT_EQ(cblas_izamax(n, xd, inc),
                  ZeroBased(izamax_(&n, P(xd), &inc)));
      }
      EXPECT_EQ(u1.real, u2.real);
      EXPECT_EQ(u1.imag, u2.imag);
      EXPECT_EQ(c1.real, c2.real);
      EXPECT_EQ(c1.imag, c2.imag);
    } else if constexpr (std::is_same_v<T, float>) {
      EXPECT_EQ(cblas_sdot(n, xd, inc, y.data(), incy),
                sdot_(&n, xd, &inc, y.data(), &incy));
      const float sb = 0.25f;
      EXPECT_EQ(cblas_sdsdot(n, sb, xd, inc, y.data(), incy),
                sdsdot_(&n, &sb, xd, &inc, y.data(), &incy));
      EXPECT_EQ(cblas_dsdot(n, xd, inc, y.data(), incy),
                dsdot_(&n, xd, &inc, y.data(), &incy));
      EXPECT_EQ(cblas_snrm2(n, xd, inc), snrm2_(&n, xd, &inc));
      EXPECT_EQ(cblas_sasum(n, xd, inc), sasum_(&n, xd, &inc));
      EXPECT_EQ(cblas_isamax(n, xd, inc), ZeroBased(isamax_(&n, xd, &inc)));
    } else {
      EXPECT_EQ(cblas_ddot(n, xd, inc, y.data(), incy),
                ddot_(&n, xd, &inc, y.data(), &incy));
      EXPECT_EQ(cblas_dnrm2(n, xd, inc), dnrm2_(&n, xd, &inc));
      EXPECT_EQ(cblas_dasum(n, xd, inc), dasum_(&n, xd, &inc));
      EXPECT_EQ(cblas_idamax(n, xd, inc), ZeroBased(idamax_(&n, xd, &inc)));
    }

    // Rotations and real scaling.
    const R c = R(0.6), s = R(0.8);
    x1 = x;
    x2 = x;
    y1 = y;
    y2 = y;
    if constexpr (std::is_same_v<T, float>) {
      cblas_srot(n, x1.data(), inc, y1.data(), incy, c, s);
      srot_(&n, x2.data(), &inc, y2.data(), &incy, &c, &s);
      const float p[5] = {-1, 0.5f, -0.25f, 2, 1.5f};
      cblas_srotm(n, x1.data(), inc, y1.data(), incy, p);
      srotm_(&n, x2.data(), &inc, y2.data(), &incy, p);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_drot(n, x1.data(), inc, y1.data(), incy, c, s);
      drot_(&n, x2.data(), &inc, y2.data(), &incy, &c, &s);
      const double p[5] = {-1, 0.5, -0.25, 2, 1.5};
      cblas_drotm(n, x1.data(), inc, y1.data(), incy, p);
      drotm_(&n, x2.data(), &inc, y2.data(), &incy, p);
    } else if constexpr (std::is_same_v<R, float>) {
      cblas_csrot(n, x1.data(), inc, y1.data(), incy, c, s);
      csrot_(&n, P(x2.data()), &inc, P(y2.data()), &incy, &c, &s);
      cblas_csscal(n, c, x1.data(), inc);
      csscal_(&n, &c, P(x2.data()), &inc);
    } else {
      cblas_zdrot(n, x1.data(), inc, y1.data(), incy, c, s);
      zdrot_(&n, P(x2.data()), &inc, P(y2.data()), &incy, &c, &s);
      cblas_zdscal(n, c, x1.data(), inc);
      zdscal_(&n, &c, P(x2.data()), &inc);
    }
    ExpectIdentical(x1, x2, "rot x");
    ExpectIdentical(y1, y2, "rot y");
  }

  // rotg and rotmg.
  T a1 = MakeScalar<T>(3, 1), b1 = MakeScalar<T>(-4, 2), s1, s2;
  T a2 = a1, b2 = b1;
  R c1, c2;
  if constexpr (std::is_same_v<T, float>) {
    cblas_srotg(&a1, &b1, &c1, &s1);
    srotg_(&a2, &b2, &c2, &s2);
    float d1[2] = {2, 2}, d2[2] = {3, 3}, x1[2] = {0.5f, 0.5f}, p1[5] = {},
          p2[5] = {};
    cblas_srotmg(&d1[0], &d2[0], &x1[0], 0.75f, p1);
    const float y1 = 0.75f;
    srotmg_(&d1[1], &d2[1], &x1[1], &y1, p2);
    EXPECT_EQ(memcmp(p1, p2, sizeof(p1)), 0);
    EXPECT_EQ(x1[0], x1[1]);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_drotg(&a1, &b1, &c1, &s1);
    drotg_(&a2, &b2, &c2, &s2);
    double d1[2] = {2, 2}, d2[2] = {3, 3}, x1[2] = {0.5, 0.5}, p1[5] = {},
           p2[5] = {};
    cblas_drotmg(&d1[0], &d2[0], &x1[0], 0.75, p1);
    const double y1 = 0.75;
    drotmg_(&d1[1], &d2[1], &x1[1], &y1, p2);
    EXPECT_EQ(memcmp(p1, p2, sizeof(p1)), 0);
    EXPECT_EQ(x1[0], x1[1]);
  } else if constexpr (std::is_same_v<R, float>) {
    cblas_crotg(&a1, &b1, &c1, &s1);
    crotg_(P(&a2), P(&b2), &c2, P(&s2));
  } else {
    cblas_zrotg(&a1, &b1, &c1, &s1);
    zrotg_(P(&a2), P(&b2), &c2, P(&s2));
  }
  EXPECT_EQ(a1, a2);
  EXPECT_EQ(c1, c2);
  EXPECT_EQ(s1, s2);
}

// ---------------------------------------------------------------------------
// GEMV, GBMV

template <typename T>
void CheckGemv(bool band, char trans, int m, int n, int kl, int ku, int incx,
               int incy, uint64_t seed) {
  Random<T> rng(seed);
  const Mat<T> a =
      band ? InBand(RandomMat<T>(m, n, rng), kl, ku) : RandomMat<T>(m, n, rng);
  const Mat<T> at = Transpose(a);
  const int len_x = (trans == 'N') ? n : m, len_y = (trans == 'N') ? m : n;
  const std::vector<T> x = Vec(len_x, incx, rng), y0 = Vec(len_y, incy, rng);
  const T alpha = MakeScalar<T>(1.5, -0.5), beta = MakeScalar<T>(0.5, 0.25);
  const CBLAS_TRANSPOSE ct = CblasOp(trans);
  std::vector<T> y_col = y0, y_fortran = y0, y_row = y0;
  if (band) {
    const int lda = kl + ku + 2;
    const std::vector<T> ac = Band(a, kl, ku, lda), ar = Band(at, ku, kl, lda);
    CBLAS_CALL(T, gbmv, CblasColMajor, ct, m, n, kl, ku, S(alpha), ac.data(),
               lda, x.data(), incx, S(beta), y_col.data(), incy);
    HWYBLAS_CALL(T, gbmv, &trans, &m, &n, &kl, &ku, P(&alpha), P(ac.data()),
                 &lda, P(x.data()), &incx, P(&beta), P(y_fortran.data()),
                 &incy);
    CBLAS_CALL(T, gbmv, CblasRowMajor, ct, m, n, kl, ku, S(alpha), ar.data(),
               lda, x.data(), incx, S(beta), y_row.data(), incy);
  } else {
    const int ldc = m + 1, ldr = n + 2;
    const std::vector<T> ac = General(a, ldc), ar = General(at, ldr);
    CBLAS_CALL(T, gemv, CblasColMajor, ct, m, n, S(alpha), ac.data(), ldc,
               x.data(), incx, S(beta), y_col.data(), incy);
    HWYBLAS_CALL(T, gemv, &trans, &m, &n, P(&alpha), P(ac.data()), &ldc,
                 P(x.data()), &incx, P(&beta), P(y_fortran.data()), &incy);
    CBLAS_CALL(T, gemv, CblasRowMajor, ct, m, n, S(alpha), ar.data(), ldr,
               x.data(), incx, S(beta), y_row.data(), incy);
  }
  const std::string what = std::string(band ? "gbmv" : "gemv") +
                           " trans=" + trans + " incx=" + std::to_string(incx);
  ExpectIdentical(y_col, y_fortran, what);
  ExpectNear(y_row, y_col, what + " row-major");
}

TYPED_TEST(CblasTest, GemvGbmv) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (char trans : Ops<T>()) {
    for (int inc : {1, -2}) {
      CheckGemv<T>(false, trans, 7, 5, 0, 0, inc, -inc, seed++);
      CheckGemv<T>(true, trans, 7, 6, 2, 1, inc, -inc, seed++);
      CheckGemv<T>(true, trans, 5, 8, 0, 3, inc, 1, seed++);
    }
  }
}

// ---------------------------------------------------------------------------
// Triangular: TRMV, TRSV, TBMV, TBSV, TPMV, TPSV

enum class Storage { kFull, kBand, kPacked };

template <typename T>
void CheckTriangular(Storage storage, bool solve, char uplo, char trans,
                     char diag, int incx, uint64_t seed) {
  const int n = 9, k = 3;
  Random<T> rng(seed);
  const Shape shape =
      (diag == 'U') ? Shape::kUnitTriangular : Shape::kTriangular;
  Mat<T> a = Structured<T>(shape, uplo, n, rng, 1.0 / n);
  if (storage == Storage::kBand) {
    a = InBand(a, uplo == 'L' ? k : 0, uplo == 'L' ? 0 : k);
  }
  const Mat<T> at = Transpose(a);
  const char uplo_t = FlipUplo(uplo);
  const std::vector<T> x0 = Vec(n, incx, rng);
  std::vector<T> x_col = x0, x_fortran = x0, x_row = x0;
  const CBLAS_UPLO cu = CblasUplo(uplo);
  const CBLAS_TRANSPOSE ct = CblasOp(trans);
  const CBLAS_DIAG cd = CblasDiag(diag);
  const int lda = (storage == Storage::kBand) ? k + 2 : n + 1;
  std::vector<T> ac, ar;
  switch (storage) {
    case Storage::kFull:
      ac = General(a, lda);
      ar = General(at, lda);
      break;
    case Storage::kBand:
      ac = Band(a, uplo == 'L' ? k : 0, uplo == 'L' ? 0 : k, lda);
      ar = Band(at, uplo_t == 'L' ? k : 0, uplo_t == 'L' ? 0 : k, lda);
      break;
    case Storage::kPacked:
      ac = Packed(a, uplo);
      ar = Packed(at, uplo_t);
      break;
  }
  const T* pc = ac.data();
  const T* pr = ar.data();
#define CHECK_TRIANGULAR(NAME, ...)                                    \
  do {                                                                 \
    CBLAS_CALL(T, NAME, CblasColMajor, cu, ct, cd, n, __VA_ARGS__(pc), \
               x_col.data(), incx);                                    \
    CBLAS_CALL(T, NAME, CblasRowMajor, cu, ct, cd, n, __VA_ARGS__(pr), \
               x_row.data(), incx);                                    \
  } while (0)
#define FULL_ARGS(p) p, lda
#define BAND_ARGS(p) k, p, lda
#define PACKED_ARGS(p) p
  switch (storage) {
    case Storage::kFull:
      if (solve) {
        CHECK_TRIANGULAR(trsv, FULL_ARGS);
        HWYBLAS_CALL(T, trsv, &uplo, &trans, &diag, &n, P(pc), &lda,
                     P(x_fortran.data()), &incx);
      } else {
        CHECK_TRIANGULAR(trmv, FULL_ARGS);
        HWYBLAS_CALL(T, trmv, &uplo, &trans, &diag, &n, P(pc), &lda,
                     P(x_fortran.data()), &incx);
      }
      break;
    case Storage::kBand:
      if (solve) {
        CHECK_TRIANGULAR(tbsv, BAND_ARGS);
        HWYBLAS_CALL(T, tbsv, &uplo, &trans, &diag, &n, &k, P(pc), &lda,
                     P(x_fortran.data()), &incx);
      } else {
        CHECK_TRIANGULAR(tbmv, BAND_ARGS);
        HWYBLAS_CALL(T, tbmv, &uplo, &trans, &diag, &n, &k, P(pc), &lda,
                     P(x_fortran.data()), &incx);
      }
      break;
    case Storage::kPacked:
      if (solve) {
        CHECK_TRIANGULAR(tpsv, PACKED_ARGS);
        HWYBLAS_CALL(T, tpsv, &uplo, &trans, &diag, &n, P(pc),
                     P(x_fortran.data()), &incx);
      } else {
        CHECK_TRIANGULAR(tpmv, PACKED_ARGS);
        HWYBLAS_CALL(T, tpmv, &uplo, &trans, &diag, &n, P(pc),
                     P(x_fortran.data()), &incx);
      }
      break;
  }
#undef CHECK_TRIANGULAR
#undef FULL_ARGS
#undef BAND_ARGS
#undef PACKED_ARGS
  const std::string what = std::string("storage=") +
                           std::to_string(static_cast<int>(storage)) +
                           (solve ? " solve" : " product") + " uplo=" + uplo +
                           " trans=" + trans + " diag=" + diag;
  ExpectIdentical(x_col, x_fortran, what);
  ExpectNear(x_row, x_col, what + " row-major");
}

TYPED_TEST(CblasTest, Triangular) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (Storage storage : {Storage::kFull, Storage::kBand, Storage::kPacked}) {
    for (bool solve : {false, true}) {
      for (char uplo : {'L', 'U'}) {
        for (char trans : Ops<T>()) {
          for (char diag : {'N', 'U'}) {
            CheckTriangular<T>(storage, solve, uplo, trans, diag,
                               (seed % 2) ? 1 : -2, seed);
            ++seed;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SYMV, SBMV, SPMV (HEMV, HBMV, HPMV)

template <typename T>
void CheckSymmetricMv(Storage storage, char uplo, int incx, uint64_t seed) {
  const int n = 9, k = 2;
  Random<T> rng(seed);
  const Shape shape =
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric;
  Mat<T> a = Structured<T>(shape, uplo, n, rng);
  if (storage == Storage::kBand) a = InBand(a, k, k);
  const Mat<T> at = Transpose(a);
  const char uplo_t = FlipUplo(uplo);
  const std::vector<T> x = Vec(n, incx, rng), y0 = Vec(n, -incx, rng);
  const int incy = -incx;
  const T alpha = MakeScalar<T>(1.5, -0.5), beta = MakeScalar<T>(0.5, 0.25);
  std::vector<T> y_col = y0, y_fortran = y0, y_row = y0;
  const CBLAS_UPLO cu = CblasUplo(uplo);
  const int lda = (storage == Storage::kBand) ? k + 2 : n + 1;
  std::vector<T> ac, ar;
  switch (storage) {
    case Storage::kFull:
      ac = General(a, lda);
      ar = General(at, lda);
      break;
    case Storage::kBand:
      ac = Band(a, uplo == 'L' ? k : 0, uplo == 'L' ? 0 : k, lda);
      ar = Band(at, uplo_t == 'L' ? k : 0, uplo_t == 'L' ? 0 : k, lda);
      break;
    case Storage::kPacked:
      ac = Packed(a, uplo);
      ar = Packed(at, uplo_t);
      break;
  }
  const T* pc = ac.data();
  const T* pr = ar.data();
  switch (storage) {
    case Storage::kFull:
      for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
        const bool row = (layout == CblasRowMajor);
        CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, symv, CBLAS_CALL_COMPLEX, hemv,
                             layout, cu, n, S(alpha), row ? pr : pc, lda,
                             x.data(), incx, S(beta),
                             (row ? y_row : y_col).data(), incy);
      }
      CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, symv, HWYBLAS_CALL_COMPLEX,
                           hemv, &uplo, &n, P(&alpha), P(pc), &lda, P(x.data()),
                           &incx, P(&beta), P(y_fortran.data()), &incy);
      break;
    case Storage::kBand:
      for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
        const bool row = (layout == CblasRowMajor);
        CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, sbmv, CBLAS_CALL_COMPLEX, hbmv,
                             layout, cu, n, k, S(alpha), row ? pr : pc, lda,
                             x.data(), incx, S(beta),
                             (row ? y_row : y_col).data(), incy);
      }
      CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, sbmv, HWYBLAS_CALL_COMPLEX,
                           hbmv, &uplo, &n, &k, P(&alpha), P(pc), &lda,
                           P(x.data()), &incx, P(&beta), P(y_fortran.data()),
                           &incy);
      break;
    case Storage::kPacked:
      for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
        const bool row = (layout == CblasRowMajor);
        CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, spmv, CBLAS_CALL_COMPLEX, hpmv,
                             layout, cu, n, S(alpha), row ? pr : pc, x.data(),
                             incx, S(beta), (row ? y_row : y_col).data(), incy);
      }
      CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, spmv, HWYBLAS_CALL_COMPLEX,
                           hpmv, &uplo, &n, P(&alpha), P(pc), P(x.data()),
                           &incx, P(&beta), P(y_fortran.data()), &incy);
      break;
  }
  const std::string what = std::string("symmetric storage=") +
                           std::to_string(static_cast<int>(storage)) +
                           " uplo=" + uplo;
  ExpectIdentical(y_col, y_fortran, what);
  ExpectNear(y_row, y_col, what + " row-major");
}

TYPED_TEST(CblasTest, SymmetricMv) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (Storage storage : {Storage::kFull, Storage::kBand, Storage::kPacked}) {
    for (char uplo : {'L', 'U'}) {
      for (int inc : {1, -2}) CheckSymmetricMv<T>(storage, uplo, inc, seed++);
    }
  }
}

// ---------------------------------------------------------------------------
// GER, GERU, GERC

template <typename T>
void CheckGer(bool conj, int incx, uint64_t seed) {
  const int m = 7, n = 5;
  Random<T> rng(seed);
  const Mat<T> a = RandomMat<T>(m, n, rng);
  const std::vector<T> x = Vec(m, incx, rng), y = Vec(n, 2, rng);
  const int incy = 2, ldc = m + 1, ldr = n + 3;
  const T alpha = MakeScalar<T>(1.5, -0.5);
  std::vector<T> a_col = General(a, ldc), a_fortran = a_col;
  std::vector<T> a_row = General(Transpose(a), ldr);
  for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
    const bool row = (layout == CblasRowMajor);
    T* out = row ? a_row.data() : a_col.data();
    const int ld = row ? ldr : ldc;
    if constexpr (IsComplex<T>::value) {
      if (conj) {
        CBLAS_CALL_COMPLEX(T, gerc, layout, m, n, S(alpha), x.data(), incx,
                           y.data(), incy, out, ld);
      } else {
        CBLAS_CALL_COMPLEX(T, geru, layout, m, n, S(alpha), x.data(), incx,
                           y.data(), incy, out, ld);
      }
    } else {
      CBLAS_CALL_REAL(T, ger, layout, m, n, S(alpha), x.data(), incx, y.data(),
                      incy, out, ld);
    }
  }
  if constexpr (IsComplex<T>::value) {
    if (conj) {
      HWYBLAS_CALL_COMPLEX(T, gerc, &m, &n, P(&alpha), P(x.data()), &incx,
                           P(y.data()), &incy, P(a_fortran.data()), &ldc);
    } else {
      HWYBLAS_CALL_COMPLEX(T, geru, &m, &n, P(&alpha), P(x.data()), &incx,
                           P(y.data()), &incy, P(a_fortran.data()), &ldc);
    }
  } else {
    HWYBLAS_CALL_REAL(T, ger, &m, &n, P(&alpha), P(x.data()), &incx,
                      P(y.data()), &incy, P(a_fortran.data()), &ldc);
  }
  const std::string what = std::string("ger conj=") + (conj ? "1" : "0");
  ExpectIdentical(a_col, a_fortran, what);
  ExpectNear(Transpose(FromGeneral(a_row, n, m, ldr)).v,
             FromGeneral(a_col, m, n, ldc).v, what + " row-major");
}

TYPED_TEST(CblasTest, Ger) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (bool conj : {false, true}) {
    if (conj && !IsComplex<T>::value) continue;
    for (int inc : {1, -2}) CheckGer<T>(conj, inc, seed++);
  }
}

// ---------------------------------------------------------------------------
// SYR, SPR, SYR2, SPR2 (HER, HPR, HER2, HPR2)

template <typename T>
void CheckSymmetricUpdate(bool packed, int terms, char uplo, int incx,
                          uint64_t seed) {
  const int n = 8, lda = n + 1;
  Random<T> rng(seed);
  const Shape shape =
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric;
  const Mat<T> a = Structured<T>(shape, uplo, n, rng);
  const char uplo_t = FlipUplo(uplo);
  const std::vector<T> x = Vec(n, incx, rng), y = Vec(n, 1, rng);
  const int incy = 1;
  const T alpha = MakeScalar<T>(0.75, terms == 1 ? 0.0 : 0.5);
  const RealT<T> real_alpha = std::real(alpha);
  std::vector<T> a_col = packed ? Packed(a, uplo) : General(a, lda);
  std::vector<T> a_fortran = a_col;
  std::vector<T> a_row =
      packed ? Packed(Transpose(a), uplo_t) : General(Transpose(a), lda);
  const CBLAS_UPLO cu = CblasUplo(uplo);
  for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
    T* out = (layout == CblasRowMajor) ? a_row.data() : a_col.data();
    if (terms == 1 && packed) {
      CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, spr, CBLAS_CALL_COMPLEX, hpr,
                           layout, cu, n, real_alpha, x.data(), incx, out);
    } else if (terms == 1) {
      CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, syr, CBLAS_CALL_COMPLEX, her,
                           layout, cu, n, real_alpha, x.data(), incx, out, lda);
    } else if (packed) {
      CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, spr2, CBLAS_CALL_COMPLEX, hpr2,
                           layout, cu, n, S(alpha), x.data(), incx, y.data(),
                           incy, out);
    } else {
      CALL_REAL_OR_COMPLEX(T, CBLAS_CALL_REAL, syr2, CBLAS_CALL_COMPLEX, her2,
                           layout, cu, n, S(alpha), x.data(), incx, y.data(),
                           incy, out, lda);
    }
  }
  T* f = a_fortran.data();
  if (terms == 1 && packed) {
    CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, spr, HWYBLAS_CALL_COMPLEX, hpr,
                         &uplo, &n, &real_alpha, P(x.data()), &incx, P(f));
  } else if (terms == 1) {
    CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, syr, HWYBLAS_CALL_COMPLEX, her,
                         &uplo, &n, &real_alpha, P(x.data()), &incx, P(f),
                         &lda);
  } else if (packed) {
    CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, spr2, HWYBLAS_CALL_COMPLEX, hpr2,
                         &uplo, &n, P(&alpha), P(x.data()), &incx, P(y.data()),
                         &incy, P(f));
  } else {
    CALL_REAL_OR_COMPLEX(T, HWYBLAS_CALL_REAL, syr2, HWYBLAS_CALL_COMPLEX, her2,
                         &uplo, &n, P(&alpha), P(x.data()), &incx, P(y.data()),
                         &incy, P(f), &lda);
  }
  const std::string what = std::string(packed ? "packed" : "full") +
                           " update terms=" + std::to_string(terms) +
                           " uplo=" + uplo;
  ExpectIdentical(a_col, a_fortran, what);
  const Mat<T> col =
      packed ? FromPacked(a_col, n, uplo) : FromGeneral(a_col, n, n, lda);
  const Mat<T> row = Transpose(packed ? FromPacked(a_row, n, uplo_t)
                                      : FromGeneral(a_row, n, n, lda));
  ExpectNear(row.v, col.v, what + " row-major");
}

TYPED_TEST(CblasTest, SymmetricUpdate) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (bool packed : {false, true}) {
    for (int terms : {1, 2}) {
      for (char uplo : {'L', 'U'}) {
        for (int inc : {1, -2}) {
          CheckSymmetricUpdate<T>(packed, terms, uplo, inc, seed++);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Level 3

// Dimensions of op(X) for a stored rows x cols matrix X.
template <typename T>
Mat<T> RandomOp(char op, int rows, int cols, Random<T>& rng) {
  return (op == 'N') ? RandomMat<T>(rows, cols, rng)
                     : RandomMat<T>(cols, rows, rng);
}

// Checks C (m x n) from the three calls.
template <typename T>
void ExpectLevel3(const std::vector<T>& c_col, const std::vector<T>& c_fortran,
                  const std::vector<T>& c_row, int m, int n, int ldc, int ldr,
                  const std::string& what) {
  ExpectIdentical(c_col, c_fortran, what);
  ExpectNear(Transpose(FromGeneral(c_row, n, m, ldr)).v,
             FromGeneral(c_col, m, n, ldc).v, what + " row-major");
}

template <typename T>
void CheckGemm(bool triangle, char uplo, char ta, char tb, uint64_t seed) {
  const int m = 6, n = triangle ? 6 : 5, k = 4;
  Random<T> rng(seed);
  // a is the stored matrix: op(A) is m x k.
  const Mat<T> a =
      (ta == 'N') ? RandomMat<T>(m, k, rng) : RandomMat<T>(k, m, rng);
  const Mat<T> b =
      (tb == 'N') ? RandomMat<T>(k, n, rng) : RandomMat<T>(n, k, rng);
  const Mat<T> c = RandomMat<T>(m, n, rng);
  const int lda_c = a.rows + 1, ldb_c = b.rows + 2, ldc_c = m + 1;
  const int lda_r = a.cols + 1, ldb_r = b.cols + 2, ldc_r = n + 3;
  const std::vector<T> ac = General(a, lda_c), bc = General(b, ldb_c);
  const std::vector<T> ar = General(Transpose(a), lda_r);
  const std::vector<T> br = General(Transpose(b), ldb_r);
  std::vector<T> c_col = General(c, ldc_c), c_fortran = c_col;
  std::vector<T> c_row = General(Transpose(c), ldc_r);
  const T alpha = MakeScalar<T>(1.5, -0.5), beta = MakeScalar<T>(0.5, 0.25);
  const CBLAS_TRANSPOSE cta = CblasOp(ta), ctb = CblasOp(tb);
  if (triangle) {
    const CBLAS_UPLO cu = CblasUplo(uplo);
    CBLAS_CALL(T, gemmtr, CblasColMajor, cu, cta, ctb, n, k, S(alpha),
               ac.data(), lda_c, bc.data(), ldb_c, S(beta), c_col.data(),
               ldc_c);
    HWYBLAS_CALL(T, gemmtr, &uplo, &ta, &tb, &n, &k, P(&alpha), P(ac.data()),
                 &lda_c, P(bc.data()), &ldb_c, P(&beta), P(c_fortran.data()),
                 &ldc_c);
    CBLAS_CALL(T, gemmt, CblasRowMajor, cu, cta, ctb, n, k, S(alpha), ar.data(),
               lda_r, br.data(), ldb_r, S(beta), c_row.data(), ldc_r);
  } else {
    CBLAS_CALL(T, gemm, CblasColMajor, cta, ctb, m, n, k, S(alpha), ac.data(),
               lda_c, bc.data(), ldb_c, S(beta), c_col.data(), ldc_c);
    HWYBLAS_CALL(T, gemm, &ta, &tb, &m, &n, &k, P(&alpha), P(ac.data()), &lda_c,
                 P(bc.data()), &ldb_c, P(&beta), P(c_fortran.data()), &ldc_c);
    CBLAS_CALL(T, gemm, CblasRowMajor, cta, ctb, m, n, k, S(alpha), ar.data(),
               lda_r, br.data(), ldb_r, S(beta), c_row.data(), ldc_r);
  }
  ExpectLevel3(c_col, c_fortran, c_row, m, n, ldc_c, ldc_r,
               std::string(triangle ? "gemmtr" : "gemm") + " uplo=" + uplo +
                   " trans=" + ta + tb);
}

TYPED_TEST(CblasTest, GemmGemmtr) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (char ta : Ops<T>()) {
    for (char tb : Ops<T>()) {
      CheckGemm<T>(false, 'L', ta, tb, seed++);
      for (char uplo : {'L', 'U'}) CheckGemm<T>(true, uplo, ta, tb, seed++);
    }
  }
}

template <typename T>
void CheckSymm(bool hermitian, char side, char uplo, uint64_t seed) {
  const int m = 6, n = 5, ka = (side == 'L') ? m : n;
  Random<T> rng(seed);
  const Mat<T> a = Structured<T>(
      hermitian ? Shape::kHermitian : Shape::kSymmetric, uplo, ka, rng);
  const Mat<T> b = RandomMat<T>(m, n, rng), c = RandomMat<T>(m, n, rng);
  const int lda = ka + 1, ldb_c = m + 2, ldc_c = m + 1, ldb_r = n + 2,
            ldc_r = n + 1;
  const std::vector<T> ac = General(a, lda), ar = General(Transpose(a), lda);
  const std::vector<T> bc = General(b, ldb_c),
                       br = General(Transpose(b), ldb_r);
  std::vector<T> c_col = General(c, ldc_c), c_fortran = c_col;
  std::vector<T> c_row = General(Transpose(c), ldc_r);
  const T alpha = MakeScalar<T>(1.5, -0.5), beta = MakeScalar<T>(0.5, 0.25);
  const CBLAS_SIDE cs = CblasSide(side);
  const CBLAS_UPLO cu = CblasUplo(uplo);
  if (hermitian) {
    CBLAS_CALL_COMPLEX(T, hemm, CblasColMajor, cs, cu, m, n, S(alpha),
                       ac.data(), lda, bc.data(), ldb_c, S(beta), c_col.data(),
                       ldc_c);
    HWYBLAS_CALL_COMPLEX(T, hemm, &side, &uplo, &m, &n, P(&alpha), P(ac.data()),
                         &lda, P(bc.data()), &ldb_c, P(&beta),
                         P(c_fortran.data()), &ldc_c);
    CBLAS_CALL_COMPLEX(T, hemm, CblasRowMajor, cs, cu, m, n, S(alpha),
                       ar.data(), lda, br.data(), ldb_r, S(beta), c_row.data(),
                       ldc_r);
  } else {
    CBLAS_CALL(T, symm, CblasColMajor, cs, cu, m, n, S(alpha), ac.data(), lda,
               bc.data(), ldb_c, S(beta), c_col.data(), ldc_c);
    HWYBLAS_CALL(T, symm, &side, &uplo, &m, &n, P(&alpha), P(ac.data()), &lda,
                 P(bc.data()), &ldb_c, P(&beta), P(c_fortran.data()), &ldc_c);
    CBLAS_CALL(T, symm, CblasRowMajor, cs, cu, m, n, S(alpha), ar.data(), lda,
               br.data(), ldb_r, S(beta), c_row.data(), ldc_r);
  }
  ExpectLevel3(c_col, c_fortran, c_row, m, n, ldc_c, ldc_r,
               std::string(hermitian ? "hemm" : "symm") + " side=" + side +
                   " uplo=" + uplo);
}

TYPED_TEST(CblasTest, SymmHemm) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (bool hermitian : {false, true}) {
    if (hermitian && !IsComplex<T>::value) continue;
    for (char side : {'L', 'R'}) {
      for (char uplo : {'L', 'U'}) CheckSymm<T>(hermitian, side, uplo, seed++);
    }
  }
}

// SYRK, HERK (terms = 1) and SYR2K, HER2K (terms = 2).
template <typename T>
void CheckRankK(bool hermitian, int terms, char uplo, char trans,
                uint64_t seed) {
  const int n = 6, k = 4;
  Random<T> rng(seed);
  const Mat<T> a =
      (trans == 'N') ? RandomMat<T>(n, k, rng) : RandomMat<T>(k, n, rng);
  const Mat<T> b =
      (trans == 'N') ? RandomMat<T>(n, k, rng) : RandomMat<T>(k, n, rng);
  Mat<T> c = Structured<T>(hermitian ? Shape::kHermitian : Shape::kSymmetric,
                           uplo, n, rng);
  const int lda_c = a.rows + 1, lda_r = a.cols + 1, ldc = n + 2;
  const std::vector<T> ac = General(a, lda_c), bc = General(b, lda_c);
  const std::vector<T> ar = General(Transpose(a), lda_r);
  const std::vector<T> br = General(Transpose(b), lda_r);
  std::vector<T> c_col = General(c, ldc), c_fortran = c_col;
  std::vector<T> c_row = General(Transpose(c), ldc);
  const T alpha = MakeScalar<T>(1.5, (hermitian && terms == 1) ? 0.0 : -0.5);
  const T beta = MakeScalar<T>(0.5, hermitian ? 0.0 : 0.25);
  const RealT<T> ra = std::real(alpha), rb = std::real(beta);
  const CBLAS_UPLO cu = CblasUplo(uplo);
  const CBLAS_TRANSPOSE ct = CblasOp(trans);
  for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
    const bool row = (layout == CblasRowMajor);
    const T* pa = row ? ar.data() : ac.data();
    const T* pb = row ? br.data() : bc.data();
    const int lda = row ? lda_r : lda_c;
    T* out = row ? c_row.data() : c_col.data();
    if (hermitian && terms == 1) {
      CBLAS_CALL_COMPLEX(T, herk, layout, cu, ct, n, k, ra, pa, lda, rb, out,
                         ldc);
    } else if (hermitian) {
      CBLAS_CALL_COMPLEX(T, her2k, layout, cu, ct, n, k, S(alpha), pa, lda, pb,
                         lda, rb, out, ldc);
    } else if (terms == 1) {
      CBLAS_CALL(T, syrk, layout, cu, ct, n, k, S(alpha), pa, lda, S(beta), out,
                 ldc);
    } else {
      CBLAS_CALL(T, syr2k, layout, cu, ct, n, k, S(alpha), pa, lda, pb, lda,
                 S(beta), out, ldc);
    }
  }
  T* f = c_fortran.data();
  if (hermitian && terms == 1) {
    HWYBLAS_CALL_COMPLEX(T, herk, &uplo, &trans, &n, &k, &ra, P(ac.data()),
                         &lda_c, &rb, P(f), &ldc);
  } else if (hermitian) {
    HWYBLAS_CALL_COMPLEX(T, her2k, &uplo, &trans, &n, &k, P(&alpha),
                         P(ac.data()), &lda_c, P(bc.data()), &lda_c, &rb, P(f),
                         &ldc);
  } else if (terms == 1) {
    HWYBLAS_CALL(T, syrk, &uplo, &trans, &n, &k, P(&alpha), P(ac.data()),
                 &lda_c, P(&beta), P(f), &ldc);
  } else {
    HWYBLAS_CALL(T, syr2k, &uplo, &trans, &n, &k, P(&alpha), P(ac.data()),
                 &lda_c, P(bc.data()), &lda_c, P(&beta), P(f), &ldc);
  }
  ExpectLevel3(c_col, c_fortran, c_row, n, n, ldc, ldc,
               std::string(hermitian ? "her" : "syr") +
                   (terms == 2 ? "2k" : "k") + " uplo=" + uplo +
                   " trans=" + trans);
}

TYPED_TEST(CblasTest, RankK) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (bool hermitian : {false, true}) {
    if (hermitian && !IsComplex<T>::value) continue;
    // Allowed: N and T (and C for real types) for SYRK, N and C for HERK.
    std::vector<char> transes = {'N', hermitian ? 'C' : 'T'};
    if (!IsComplex<T>::value) transes.push_back('C');
    for (int terms : {1, 2}) {
      for (char uplo : {'L', 'U'}) {
        for (char trans : transes) {
          CheckRankK<T>(hermitian, terms, uplo, trans, seed++);
        }
      }
    }
  }
}

template <typename T>
void CheckTrmm(bool solve, char side, char uplo, char trans, char diag,
               uint64_t seed) {
  const int m = 6, n = 5, ka = (side == 'L') ? m : n;
  Random<T> rng(seed);
  const Mat<T> a =
      Structured<T>(diag == 'U' ? Shape::kUnitTriangular : Shape::kTriangular,
                    uplo, ka, rng, 1.0 / ka);
  const Mat<T> b = RandomMat<T>(m, n, rng);
  const int lda = ka + 1, ldb_c = m + 1, ldb_r = n + 2;
  const std::vector<T> ac = General(a, lda), ar = General(Transpose(a), lda);
  std::vector<T> b_col = General(b, ldb_c), b_fortran = b_col;
  std::vector<T> b_row = General(Transpose(b), ldb_r);
  const T alpha = MakeScalar<T>(1.5, -0.5);
  const CBLAS_SIDE cs = CblasSide(side);
  const CBLAS_UPLO cu = CblasUplo(uplo);
  const CBLAS_TRANSPOSE ct = CblasOp(trans);
  const CBLAS_DIAG cd = CblasDiag(diag);
  if (solve) {
    CBLAS_CALL(T, trsm, CblasColMajor, cs, cu, ct, cd, m, n, S(alpha),
               ac.data(), lda, b_col.data(), ldb_c);
    HWYBLAS_CALL(T, trsm, &side, &uplo, &trans, &diag, &m, &n, P(&alpha),
                 P(ac.data()), &lda, P(b_fortran.data()), &ldb_c);
    CBLAS_CALL(T, trsm, CblasRowMajor, cs, cu, ct, cd, m, n, S(alpha),
               ar.data(), lda, b_row.data(), ldb_r);
  } else {
    CBLAS_CALL(T, trmm, CblasColMajor, cs, cu, ct, cd, m, n, S(alpha),
               ac.data(), lda, b_col.data(), ldb_c);
    HWYBLAS_CALL(T, trmm, &side, &uplo, &trans, &diag, &m, &n, P(&alpha),
                 P(ac.data()), &lda, P(b_fortran.data()), &ldb_c);
    CBLAS_CALL(T, trmm, CblasRowMajor, cs, cu, ct, cd, m, n, S(alpha),
               ar.data(), lda, b_row.data(), ldb_r);
  }
  ExpectLevel3(b_col, b_fortran, b_row, m, n, ldb_c, ldb_r,
               std::string(solve ? "trsm" : "trmm") + " side=" + side +
                   " uplo=" + uplo + " trans=" + trans + " diag=" + diag);
}

TYPED_TEST(CblasTest, TrmmTrsm) {
  using T = TypeParam;
  uint64_t seed = 1;
  for (bool solve : {false, true}) {
    for (char side : {'L', 'R'}) {
      for (char uplo : {'L', 'U'}) {
        for (char trans : Ops<T>()) {
          for (char diag : {'N', 'U'}) {
            CheckTrmm<T>(solve, side, uplo, trans, diag, seed++);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Errors are reported with CBLAS argument positions.

std::string ErrorOf(void (*call)()) {
  ::testing::internal::CaptureStderr();
  call();
  return ::testing::internal::GetCapturedStderr();
}

TEST(CblasErrors, Positions) {
  static double a[16], b[16], c[16], x[4], y[4];
  // Invalid layout: argument 1.
  EXPECT_NE(ErrorOf([] {
              cblas_dgemm(static_cast<CBLAS_LAYOUT>(0), CblasNoTrans,
                          CblasNoTrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
            }).find("Parameter 1 to routine cblas_dgemm"),
            std::string::npos);
  // m < 0 is argument 4 in both layouts (it maps to the Fortran n for
  // row-major storage).
  for (CBLAS_LAYOUT layout : {CblasColMajor, CblasRowMajor}) {
    static CBLAS_LAYOUT l;
    l = layout;
    EXPECT_NE(ErrorOf([] {
                cblas_dgemm(l, CblasNoTrans, CblasNoTrans, -1, 2, 2, 1.0, a, 2,
                            b, 2, 0.0, c, 2);
              }).find("Parameter 4 to routine cblas_dgemm"),
              std::string::npos);
    // Invalid transb: argument 3.
    EXPECT_NE(ErrorOf([] {
                cblas_dgemm(l, CblasNoTrans, static_cast<CBLAS_TRANSPOSE>(0), 2,
                            2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
              }).find("Parameter 3 to routine cblas_dgemm"),
              std::string::npos);
    // incy == 0 in GER: argument 8.
    EXPECT_NE(ErrorOf([] {
                cblas_dger(l, 2, 2, 1.0, x, 1, y, 0, a, 2);
              }).find("Parameter 8 to routine cblas_dger"),
              std::string::npos);
    // Invalid uplo in TRSV: argument 2.
    EXPECT_NE(ErrorOf([] {
                cblas_dtrsv(l, static_cast<CBLAS_UPLO>(0), CblasNoTrans,
                            CblasNonUnit, 2, a, 2, x, 1);
              }).find("Parameter 2 to routine cblas_dtrsv"),
              std::string::npos);
  }
  // lda too small for row-major storage: lda >= n (argument 9).
  EXPECT_NE(ErrorOf([] {
              cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 3, 2,
                          1.0, a, 1, b, 3, 0.0, c, 3);
            }).find("Parameter 9 to routine cblas_dgemm"),
            std::string::npos);
  // The Fortran routines keep reporting through xerbla_ with their own
  // names afterwards.
  EXPECT_NE(ErrorOf([] {
              const int m = -1, n = 2;
              const double one = 1;
              dger_(&m, &n, &one, x, &n, y, &n, a, &n);
            }).find("DGER"),
            std::string::npos);
}

// An installed handler replaces xerbla_ and cblas_xerbla, for both the
// Fortran-style routines and the CBLAS functions.
std::vector<std::string> g_errors;
void RecordError(const char* name, hwyblas_int position) {
  g_errors.push_back(std::string(name) + ":" + std::to_string(position));
}

TEST(CblasErrors, Handler) {
  static double a[16], b[16], c[16];
  g_errors.clear();
  EXPECT_EQ(hwyblas_set_error_handler(RecordError), nullptr);
  const std::string printed = ErrorOf([] {
    const int m = -1, n = 2;
    const double one = 1;
    dgemm_("N", "N", &m, &n, &n, &one, a, &n, b, &n, &one, c, &n);
    cblas_dgemm(static_cast<CBLAS_LAYOUT>(0), CblasNoTrans, CblasNoTrans, 2, 2,
                2, 1.0, a, 2, b, 2, 0.0, c, 2);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 3, 2, 1.0, a, 1,
                b, 3, 0.0, c, 3);
  });
  EXPECT_EQ(hwyblas_set_error_handler(nullptr), &RecordError);
  EXPECT_EQ(printed, "");  // nothing went to the default handlers
  EXPECT_EQ(g_errors, (std::vector<std::string>{"DGEMM:3", "cblas_dgemm:1",
                                                "cblas_dgemm:9"}));
  // The default handlers are back.
  EXPECT_NE(ErrorOf([] {
              cblas_dgemm(static_cast<CBLAS_LAYOUT>(0), CblasNoTrans,
                          CblasNoTrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
            }).find("Parameter 1 to routine cblas_dgemm"),
            std::string::npos);
}

// Row-major complex calls that conjugate the output vector around the
// Fortran call must report invalid arguments before touching it: the vector
// may be shorter than the dimensions say. The handler checks the vector at
// the time of the report.
using z64 = std::complex<double>;
z64 g_vector[2];
std::vector<std::string> g_reports;
void CheckVectorUntouched(const char* name, hwyblas_int position) {
  const bool untouched = g_vector[0] == z64(1, 2) && g_vector[1] == z64(3, 4);
  g_reports.push_back(std::string(name) + ":" + std::to_string(position) +
                      (untouched ? "" : " touched"));
}

TEST(CblasErrors, InvalidCallsLeaveVectorsUntouched) {
  static const z64 a[4], x[2], one = 1;
  g_vector[0] = z64(1, 2);
  g_vector[1] = z64(3, 4);
  g_reports.clear();
  hwyblas_set_error_handler(CheckVectorUntouched);
  // lda < n (k < 0 for the band matrices).
  cblas_zgemv(CblasRowMajor, CblasConjTrans, 2, 2, &one, a, 1, x, 1, &one,
              g_vector, 1);
  cblas_zgbmv(CblasRowMajor, CblasConjTrans, 2, 2, -1, 0, &one, a, 1, x, 1,
              &one, g_vector, 1);
  cblas_ztrmv(CblasRowMajor, CblasUpper, CblasConjTrans, CblasNonUnit, 2, a, 1,
              g_vector, 1);
  cblas_ztbsv(CblasRowMajor, CblasUpper, CblasConjTrans, CblasNonUnit, 2, -1,
              a, 1, g_vector, 1);
  cblas_zhemv(CblasRowMajor, CblasUpper, 2, &one, a, 1, x, 1, &one, g_vector,
              1);
  cblas_zhbmv(CblasRowMajor, CblasUpper, 2, -1, &one, a, 1, x, 1, &one,
              g_vector, 1);
  // incx == 0.
  cblas_zhpmv(CblasRowMajor, CblasUpper, 2, &one, a, x, 0, &one, g_vector, 1);
  hwyblas_set_error_handler(nullptr);
  EXPECT_EQ(g_reports, (std::vector<std::string>{
                           "cblas_zgemv:7", "cblas_zgbmv:5", "cblas_ztrmv:7",
                           "cblas_ztbsv:6", "cblas_zhemv:6", "cblas_zhbmv:4",
                           "cblas_zhpmv:7"}));
}

}  // namespace
}  // namespace hwyblas_test
