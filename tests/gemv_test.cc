// Tests for sgemv_, dgemv_, cgemv_ and zgemv_ against a reference
// implementation computed in double precision.
#include <stddef.h>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

// Calls the BLAS routine for T.
template <typename T>
void Gemv(char trans, int m, int n, T alpha, const T* a, int lda, const T* x,
          int incx, T beta, T* y, int incy) {
  using R = RealT<T>;
  const auto in = [](const T* p) { return reinterpret_cast<const R*>(p); };
  R* out = reinterpret_cast<R*>(y);
  if constexpr (std::is_same_v<T, float>) {
    sgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else if constexpr (std::is_same_v<T, double>) {
    dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else if constexpr (std::is_same_v<R, float>) {
    cgemv_(&trans, &m, &n, in(&alpha), in(a), &lda, in(x), &incx, in(&beta),
           out, &incy);
  } else {
    zgemv_(&trans, &m, &n, in(&alpha), in(a), &lda, in(x), &incx, in(&beta),
           out, &incy);
  }
}

struct GemvCase {
  char trans;
  int m, n;
  int incx, incy;
  int pad;  // extra rows in lda
};

// Element i of a BLAS vector with increment inc (inc < 0: stored backwards).
size_t VectorIndex(size_t i, size_t len, int inc) {
  return inc > 0 ? i * inc : (len - 1 - i) * static_cast<size_t>(-inc);
}

template <typename T>
void CheckGemv(const GemvCase& tc, T alpha, T beta, uint64_t seed) {
  using R = RefT<T>;
  const int lda = std::max(1, tc.m + tc.pad);
  const size_t len_x = tc.trans == 'N' ? tc.n : tc.m;
  const size_t len_y = tc.trans == 'N' ? tc.m : tc.n;
  const size_t size_x = len_x == 0 ? 0 : 1 + (len_x - 1) * std::abs(tc.incx);
  const size_t size_y = len_y == 0 ? 0 : 1 + (len_y - 1) * std::abs(tc.incy);

  Random<T> rng(seed);
  const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * tc.n);
  const std::vector<T> x = rng.Vector(size_x);
  std::vector<T> y = rng.Vector(size_y);
  if (beta == T(0)) {
    // beta == 0 must not read y: poison the elements of the vector.
    for (size_t i = 0; i < len_y; ++i) {
      y[VectorIndex(i, len_y, tc.incy)] =
          T(std::numeric_limits<RealT<T>>::quiet_NaN());
    }
  }
  const std::vector<T> y_orig = y;

  // Expected values, stored densely by vector element index.
  std::vector<R> expected(len_y);
  std::vector<double> bound(len_y);
  for (size_t i = 0; i < len_y; ++i) {
    R sum = 0;
    double abs_sum = 0;
    for (size_t p = 0; p < len_x; ++p) {
      const R ap =
          static_cast<R>(tc.trans == 'N' ? OpAt('N', a, i, p, lda)
                                         : OpAt(tc.trans, a, i, p, lda));
      const R xp = static_cast<R>(x[VectorIndex(p, len_x, tc.incx)]);
      sum += ap * xp;
      abs_sum += std::abs(ap) * std::abs(xp);
    }
    const T y_old = y_orig[VectorIndex(i, len_y, tc.incy)];
    if (tc.m == 0 || tc.n == 0) {
      // Reference BLAS returns immediately: y is not even scaled by beta.
      expected[i] = static_cast<R>(y_old);
      bound[i] = 1.0;
      continue;
    }
    R result = static_cast<R>(alpha) * sum;
    double scale = Abs(alpha) * abs_sum;
    if (beta != T(0)) {
      result += static_cast<R>(beta) * static_cast<R>(y_old);
      scale += Abs(beta) * Abs(y_old);
    }
    expected[i] = result;
    bound[i] = scale + std::numeric_limits<RealT<T>>::min();
  }

  Gemv<T>(tc.trans, tc.m, tc.n, alpha, a.data(), lda, x.data(), tc.incx, beta,
          y.data(), tc.incy);

  std::vector<T> actual(len_y);
  for (size_t i = 0; i < len_y; ++i) {
    actual[i] = y[VectorIndex(i, len_y, tc.incy)];
  }
  const double tol = 4.0 * (len_x + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(actual, expected, bound, tol, len_y, 1, len_y))
      << "trans=" << tc.trans << " m=" << tc.m << " n=" << tc.n
      << " incx=" << tc.incx << " incy=" << tc.incy;

  // Elements of y between the strided entries must be untouched.
  std::vector<bool> is_element(size_y, false);
  for (size_t i = 0; i < len_y; ++i) {
    is_element[VectorIndex(i, len_y, tc.incy)] = true;
  }
  for (size_t i = 0; i < size_y; ++i) {
    if (!is_element[i]) ASSERT_EQ(y[i], y_orig[i]) << "gap modified at " << i;
  }
}

template <typename T>
class GemvTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GemvTest, ScalarTypes);

// All small shapes: covers vector tails and the column remainder loops.
TYPED_TEST(GemvTest, SmallShapes) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char trans : Ops<T>()) {
      for (int m = 0; m <= 21; ++m) {
        for (int n = 0; n <= 11; ++n) {
          CheckGemv<T>({trans, m, n, 1, 1, 0}, MakeScalar<T>(1.5, -0.5),
                       MakeScalar<T>(0.75, 0.25), seed++);
        }
      }
    }
  });
}

TYPED_TEST(GemvTest, LargeAndStrided) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 100;
    for (char trans : Ops<T>()) {
      for (int inc : {1, 2, -1, -3}) {
        CheckGemv<T>({trans, 517, 263, inc, -inc, 3}, MakeScalar<T>(-0.5, 2.0),
                     MakeScalar<T>(1.0, 0.0), seed++);
        CheckGemv<T>({trans, 31, 1000, 1, inc, 0}, MakeScalar<T>(2.0, 0.0),
                     MakeScalar<T>(0.0, 0.0), seed++);
      }
    }
  });
}

TYPED_TEST(GemvTest, AlphaBetaSpecialCases) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1000;
    const double values[] = {0.0, 1.0, -1.0, 0.5};
    for (char trans : Ops<T>()) {
      for (double alpha : values) {
        for (double beta : values) {
          CheckGemv<T>({trans, 37, 23, 1, 2, 1}, MakeScalar<T>(alpha, 0.0),
                       MakeScalar<T>(beta, 0.0), seed++);
        }
      }
    }
  });
}

TYPED_TEST(GemvTest, InvalidArguments) {
  using T = TypeParam;
  const T one = MakeScalar<T>(1.0, 0.0);
  T a[4] = {one, one, one, one}, x[2] = {one, one};
  T y[2] = {T(7), T(7)};
  Gemv<T>('X', 2, 2, one, a, 2, x, 1, one, y, 1);   // bad trans
  Gemv<T>('N', -1, 2, one, a, 2, x, 1, one, y, 1);  // m < 0
  Gemv<T>('N', 2, 2, one, a, 1, x, 1, one, y, 1);   // lda < m
  Gemv<T>('N', 2, 2, one, a, 2, x, 0, one, y, 1);   // incx == 0
  Gemv<T>('N', 2, 2, one, a, 2, x, 1, one, y, 0);   // incy == 0
  for (const T& v : y) EXPECT_EQ(v, T(7));
}

}  // namespace
}  // namespace hwyblas_test
