// Tests for the OpenBLAS-style extensions ?axpby, i?amin, ?omatcopy and
// ?imatcopy, in their Fortran and CBLAS forms.
#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/cblas.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class ExtensionsTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ExtensionsTest, ScalarTypes);

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

// CBLAS passes real scalars by value and complex ones by pointer.
template <typename T>
auto S(const T& v) {
  if constexpr (IsComplex<T>::value) {
    return static_cast<const void*>(&v);
  } else {
    return v;
  }
}

template <typename T>
bool Identical(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// ---------------------------------------------------------------------------
// AXPBY

template <typename T>
void CheckAxpby(int n, int incx, int incy, T alpha, T beta, uint64_t seed) {
  Random<T> rng(seed);
  StridedVector<T> x(n, incx, rng), y(n, incy, rng);
  // Inputs that must not be read are NaN.
  if (alpha == T(0)) {
    for (int i = 0; i < n; ++i) x[i] = NaN<T>();
  }
  if (beta == T(0)) {
    for (int i = 0; i < n; ++i) y[i] = NaN<T>();
  }
  std::vector<RefT<T>> expected(n);
  std::vector<double> bound(n);
  for (int i = 0; i < n; ++i) {
    RefT<T> e = 0;
    double b = 0;
    if (!(alpha == T(0))) {
      e += static_cast<RefT<T>>(alpha) * static_cast<RefT<T>>(x[i]);
      b += Abs(alpha) * Abs(x[i]);
    }
    if (!(beta == T(0))) {
      e += static_cast<RefT<T>>(beta) * static_cast<RefT<T>>(y[i]);
      b += Abs(beta) * Abs(y[i]);
    }
    expected[i] = e;
    bound[i] = b + std::numeric_limits<RealT<T>>::min();
  }
  StridedVector<T> y_cblas = y;

  HWYBLAS_CALL(T, axpby, &n, P(&alpha), P(x.data.data()), &incx, P(&beta),
               P(y.data.data()), &incy);
  CBLAS_CALL(T, axpby, n, S(alpha), x.data.data(), incx, S(beta),
             y_cblas.data.data(), incy);

  EXPECT_TRUE(
      NearWithBound(y.Elements(), expected, bound, 4 * Eps<T>(), n, 1, n))
      << "n=" << n << " incx=" << incx << " incy=" << incy << " alpha=" << alpha
      << " beta=" << beta;
  EXPECT_TRUE(Identical(y.data, y_cblas.data)) << "cblas n=" << n;
  x.Elements();  // x unchanged in the gaps
}

TYPED_TEST(ExtensionsTest, Axpby) {
  using T = TypeParam;
  const T a = MakeScalar<T>(1.5, -0.5), b = MakeScalar<T>(-0.75, 0.25);
  ForEachTarget([&] {
    uint64_t seed = 1;
    for (int n : {0, 1, 3, 4, 7, 8, 9, 31, 100}) {
      for (const auto& ab : {std::pair<T, T>{a, b},
                             {T(0), b},
                             {a, T(0)},
                             {T(0), T(0)},
                             {a, T(1)}}) {
        CheckAxpby<T>(n, 1, 1, ab.first, ab.second, seed++);
      }
    }
    for (const auto& inc : {std::pair<int, int>{2, -3}, {-1, 1}, {1, 2}}) {
      CheckAxpby<T>(37, inc.first, inc.second, a, b, seed++);
      CheckAxpby<T>(37, inc.first, inc.second, a, T(0), seed++);
      CheckAxpby<T>(37, inc.first, inc.second, T(0), b, seed++);
    }
  });
}

// ---------------------------------------------------------------------------
// I?AMIN

// Reference: 1-based index of the first smallest |re| + |im|; NaNs never
// compare smaller.
template <typename T>
int ReferenceIamin(StridedVector<T>& x) {
  const auto mag = [](T v) {
    return std::abs(std::real(v)) + std::abs(std::imag(v));
  };
  int best = 1;
  RealT<T> best_value = mag(x[0]);
  for (size_t i = 1; i < x.n; ++i) {
    const RealT<T> v = mag(x[i]);
    if (v < best_value) {
      best = static_cast<int>(i) + 1;
      best_value = v;
    }
  }
  return best;
}

template <typename T>
int Iamin(int n, const T* x, int incx) {
  if constexpr (std::is_same_v<T, float>) return isamin_(&n, x, &incx);
  if constexpr (std::is_same_v<T, double>) return idamin_(&n, x, &incx);
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return icamin_(&n, P(x), &incx);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    return izamin_(&n, P(x), &incx);
  }
}

template <typename T>
size_t CblasIamin(int n, const T* x, int incx) {
  if constexpr (std::is_same_v<T, float>) return cblas_isamin(n, x, incx);
  if constexpr (std::is_same_v<T, double>) return cblas_idamin(n, x, incx);
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return cblas_icamin(n, x, incx);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    return cblas_izamin(n, x, incx);
  }
}

TYPED_TEST(ExtensionsTest, Iamin) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int inc : {1, 2}) {
      for (int n : {1, 2, 3, 7, 8, 9, 31, 64, 100, 1000}) {
        for (int variant = 0; variant < 4; ++variant) {
          Random<T> rng(seed++);
          StridedVector<T> x(n, inc, rng);
          if (variant == 1) {  // ties: the first one wins
            x[n / 2] = MakeScalar<T>(1e-3, 0);
            x[n - 1] = MakeScalar<T>(-1e-3, 0);
          } else if (variant == 2) {  // NaNs are skipped...
            x[n / 3] = NaN<T>();
            x[n - 1] = MakeScalar<T>(0, 0);
          } else if (variant == 3) {  // ... unless the first element is NaN
            x[0] = NaN<T>();
          }
          const int expected = ReferenceIamin(x);
          EXPECT_EQ(Iamin(n, x.data.data(), inc), expected)
              << "n=" << n << " inc=" << inc << " variant=" << variant;
          EXPECT_EQ(CblasIamin(n, x.data.data(), inc), size_t(expected - 1));
        }
      }
    }
  });
  // n < 1 or incx <= 0: 0 (CBLAS: 0 as well).
  std::vector<T> x(4, T(1));
  EXPECT_EQ(Iamin(0, x.data(), 1), 0);
  EXPECT_EQ(Iamin(4, x.data(), 0), 0);
  EXPECT_EQ(Iamin(4, x.data(), -1), 0);
  EXPECT_EQ(CblasIamin(4, x.data(), -1), size_t{0});
}

// ---------------------------------------------------------------------------
// OMATCOPY, IMATCOPY

// Element (i, j) of a rows x cols matrix with the given order.
size_t Index(char order, int i, int j, int ld) {
  return order == 'C' ? i + size_t(j) * ld : size_t(i) * ld + j;
}

// The transposition options of ?omatcopy_ for T (R and C conjugate).
const char kMatcopyTrans[] = {'N', 'T', 'R', 'C'};

CBLAS_TRANSPOSE CblasMatcopyTrans(char t) {
  switch (t) {
    case 'N':
      return CblasNoTrans;
    case 'T':
      return CblasTrans;
    case 'R':
      return CblasConjNoTrans;
    default:
      return CblasConjTrans;
  }
}

template <typename T>
void CheckMatcopy(bool in_place, char order, char trans, int rows, int cols,
                  T alpha, uint64_t seed) {
  const bool transposed = (trans == 'T' || trans == 'C');
  const bool conj = (trans == 'R' || trans == 'C');
  // B = op(A) is b_rows x b_cols.
  const int b_rows = transposed ? cols : rows,
            b_cols = transposed ? rows : cols;
  // Leading dimensions with padding (for the given order).
  const int lda = (order == 'C' ? rows : cols) + 2;
  const int ldb = (order == 'C' ? b_rows : b_cols) + 3;
  const size_t a_size = size_t(lda) * std::max(order == 'C' ? cols : rows, 1);
  const size_t b_size =
      size_t(ldb) * std::max(order == 'C' ? b_cols : b_rows, 1);

  Random<T> rng(seed);
  std::vector<T> a = rng.Vector(a_size);
  if (alpha == T(0)) a.assign(a_size, NaN<T>());  // must not be read
  const T sentinel = MakeScalar<T>(7, 7);
  const char ord[2] = {order, 0}, tr[2] = {trans, 0};
  const CBLAS_LAYOUT layout = (order == 'C') ? CblasColMajor : CblasRowMajor;
  const CBLAS_TRANSPOSE ct = CblasMatcopyTrans(trans);

  std::vector<T> b, b_cblas;
  if (in_place) {
    b.assign(std::max(a_size, b_size), sentinel);
    std::copy(a.begin(), a.end(), b.begin());
    b_cblas = b;
    HWYBLAS_CALL(T, imatcopy, ord, tr, &rows, &cols, P(&alpha), P(b.data()),
                 &lda, &ldb);
    CBLAS_CALL(T, imatcopy, layout, ct, rows, cols, S(alpha), b_cblas.data(),
               lda, ldb);
  } else {
    b.assign(b_size, sentinel);
    b_cblas = b;
    HWYBLAS_CALL(T, omatcopy, ord, tr, &rows, &cols, P(&alpha), P(a.data()),
                 &lda, P(b.data()), &ldb);
    CBLAS_CALL(T, omatcopy, layout, ct, rows, cols, S(alpha), a.data(), lda,
               b_cblas.data(), ldb);
  }
  const std::string what = std::string(in_place ? "imatcopy" : "omatcopy") +
                           " order=" + order + " trans=" + trans +
                           " rows=" + std::to_string(rows) +
                           " cols=" + std::to_string(cols);
  EXPECT_TRUE(Identical(b, b_cblas)) << what << " (cblas)";

  // B(i, j) = alpha * op(A)(i, j); for out-of-place copies, the rest of b
  // (padding) must be untouched.
  std::vector<bool> in_b(b.size(), false);
  size_t failures = 0;
  for (int j = 0; j < b_cols; ++j) {
    for (int i = 0; i < b_rows; ++i) {
      const size_t bi = Index(order, i, j, ldb);
      in_b[bi] = true;
      const T aij =
          transposed ? a[Index(order, j, i, lda)] : a[Index(order, i, j, lda)];
      const RefT<T> expected =
          (alpha == T(0)) ? RefT<T>(0)
                          : static_cast<RefT<T>>(alpha) *
                                static_cast<RefT<T>>(conj ? Conj(aij) : aij);
      const double err = std::abs(static_cast<RefT<T>>(b[bi]) - expected);
      // alpha == 0: exactly zero (A, all NaN, must not be read).
      const double tol =
          (alpha == T(0)) ? 0.0 : 4 * Eps<T>() * Abs(alpha) * Abs(aij);
      if (!(err <= tol)) {
        if (failures++ < 3) {
          ADD_FAILURE() << what << " B(" << i << "," << j << ") = " << b[bi]
                        << ", expected " << expected;
        }
      }
    }
  }
  if (!in_place) {
    for (size_t k = 0; k < b.size(); ++k) {
      if (!in_b[k] && !(b[k] == sentinel) && failures++ < 3) {
        ADD_FAILURE() << what << ": padding element " << k << " modified";
      }
    }
  }
}

TYPED_TEST(ExtensionsTest, Matcopy) {
  using T = TypeParam;
  const int shapes[][2] = {{0, 3}, {3, 0},   {1, 1},  {5, 7},
                           {7, 5}, {40, 33}, {65, 70}};
  ForEachTarget([&] {
    uint64_t seed = 1;
    for (bool in_place : {false, true}) {
      for (char order : {'C', 'R'}) {
        for (char trans : kMatcopyTrans) {
          for (const auto& s : shapes) {
            CheckMatcopy<T>(in_place, order, trans, s[0], s[1],
                            MakeScalar<T>(1.5, -0.5), seed++);
          }
          CheckMatcopy<T>(in_place, order, trans, 9, 4, T(1), seed++);
          CheckMatcopy<T>(in_place, order, trans, 9, 4, T(0), seed++);
        }
      }
    }
  });
}

// In place with the same leading dimension (the direct scaling path).
TYPED_TEST(ExtensionsTest, ImatcopySameLeadingDimension) {
  using T = TypeParam;
  const int rows = 6, cols = 5, ld = 8;
  Random<T> rng(3);
  std::vector<T> a = rng.Vector(size_t(ld) * cols);
  const std::vector<T> a0 = a;
  const T alpha = MakeScalar<T>(-2.0, 0.5);
  HWYBLAS_CALL(T, imatcopy, "C", "N", &rows, &cols, P(&alpha), P(a.data()), &ld,
               &ld);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < ld; ++i) {
      const size_t k = i + size_t(j) * ld;
      if (i < rows) {
        EXPECT_NEAR(Abs(a[k] - alpha * a0[k]), 0, 4 * Eps<T>() * 4);
      } else {
        EXPECT_EQ(a[k], a0[k]) << "padding";
      }
    }
  }
}

std::string ErrorOf(void (*call)()) {
  ::testing::internal::CaptureStderr();
  call();
  return ::testing::internal::GetCapturedStderr();
}

TEST(ExtensionsErrors, Matcopy) {
  static double a[64], b[64];
  // ldb too small (argument 9): column-major transposed B is 3 x 2.
  EXPECT_NE(ErrorOf([] {
              const int rows = 2, cols = 3, lda = 2, ldb = 2;
              const double one = 1;
              domatcopy_("C", "T", &rows, &cols, &one, a, &lda, b, &ldb);
            }).find("DOMATCOPY"),
            std::string::npos);
  EXPECT_NE(ErrorOf([] {
              cblas_domatcopy(CblasRowMajor, CblasNoTrans, 2, 3, 1.0, a, 3, b,
                              2);
            }).find("Parameter 9 to routine cblas_domatcopy"),
            std::string::npos);
  EXPECT_NE(ErrorOf([] {
              cblas_dimatcopy(static_cast<CBLAS_LAYOUT>(0), CblasNoTrans, 2, 3,
                              1.0, a, 3, 3);
            }).find("Parameter 1 to routine cblas_dimatcopy"),
            std::string::npos);
  EXPECT_NE(ErrorOf([] {
              cblas_dimatcopy(CblasColMajor, CblasTrans, 2, 3, 1.0, a, 2, 2);
            }).find("Parameter 8 to routine cblas_dimatcopy"),
            std::string::npos);
}

}  // namespace
}  // namespace hwyblas_test
