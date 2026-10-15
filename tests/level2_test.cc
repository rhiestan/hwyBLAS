// Tests for SYMV, HEMV, TRMV and AXPY against references computed in double
// precision. Structured matrices have NaN wherever BLAS must not read.
#include <stddef.h>

#include <complex>
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

// Strided vector with n elements and increment inc (inc < 0: stored
// backwards). Gaps between elements hold a sentinel to detect stray writes.
template <typename T>
struct StridedVector {
  size_t n;
  int inc;
  std::vector<T> data;

  StridedVector(size_t n_, int inc_, Random<T>& rng)
      : n(n_),
        inc(inc_),
        data(n_ == 0 ? 0 : 1 + (n_ - 1) * std::abs(inc_), Sentinel()) {
    for (size_t i = 0; i < n; ++i) (*this)[i] = rng.Next();
  }
  static T Sentinel() { return MakeScalar<T>(7.0, 7.0); }
  T& operator()(size_t i) { return (*this)[i]; }
  T& operator[](size_t i) {
    return data[inc > 0 ? i * inc : (n - 1 - i) * static_cast<size_t>(-inc)];
  }
  // Dense n x 1 copy in double precision.
  Dense<T> ToDense() {
    Dense<T> d(n, 1);
    for (size_t i = 0; i < n; ++i) d(i, 0) = static_cast<RefT<T>>((*this)[i]);
    return d;
  }
  // Checks that only the elements were modified, then returns them.
  std::vector<T> Elements() {
    std::vector<bool> is_element(data.size(), false);
    for (size_t i = 0; i < n; ++i) is_element[&(*this)[i] - data.data()] = true;
    for (size_t i = 0; i < data.size(); ++i) {
      if (!is_element[i]) EXPECT_EQ(data[i], Sentinel()) << "gap " << i;
    }
    std::vector<T> e(n);
    for (size_t i = 0; i < n; ++i) e[i] = (*this)[i];
    return e;
  }
};

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
// AXPY

TYPED_TEST(Level2Test, Axpy) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : {0, 1, 3, 8, 17, 1000}) {
      for (int inc : {1, 2, -3}) {
        Random<T> rng(seed++);
        StridedVector<T> x(n, inc, rng), y(n, -inc, rng);
        const T alpha = MakeScalar<T>(-1.5, 0.75);
        std::vector<RefT<T>> expected(n);
        std::vector<double> bound(n);
        for (int i = 0; i < n; ++i) {
          expected[i] =
              static_cast<RefT<T>>(alpha) * static_cast<RefT<T>>(x[i]) +
              static_cast<RefT<T>>(y[i]);
          bound[i] = Abs(alpha) * Abs(x[i]) + Abs(y[i]);
        }
        const int incy = -inc;
        HWYBLAS_CALL(T, axpy, &n, P(&alpha), P(x.data.data()), &inc,
                     P(y.data.data()), &incy);
        EXPECT_TRUE(
            NearWithBound(y.Elements(), expected, bound, 4 * Eps<T>(), n, 1, n))
            << "n=" << n << " inc=" << inc;
      }
    }
  });
}

}  // namespace
}  // namespace hwyblas_test
