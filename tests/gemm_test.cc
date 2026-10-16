// Tests for sgemm_, dgemm_, cgemm_ and zgemm_ against a reference
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
void Gemm(char ta, char tb, int m, int n, int k, T alpha, const T* a, int lda,
          const T* b, int ldb, T beta, T* c, int ldc) {
  if constexpr (std::is_same_v<T, float>) {
    sgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  } else if constexpr (std::is_same_v<T, double>) {
    dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  } else {
    // Complex routines take pointers to interleaved (re, im) reals.
    using R = RealT<T>;
    const auto in = [](const T* p) { return reinterpret_cast<const R*>(p); };
    if constexpr (std::is_same_v<R, float>) {
      cgemm_(&ta, &tb, &m, &n, &k, in(&alpha), in(a), &lda, in(b), &ldb,
             in(&beta), reinterpret_cast<R*>(c), &ldc);
    } else {
      zgemm_(&ta, &tb, &m, &n, &k, in(&alpha), in(a), &lda, in(b), &ldb,
             in(&beta), reinterpret_cast<R*>(c), &ldc);
    }
  }
}

struct GemmCase {
  char ta, tb;
  int m, n, k;
  // Extra rows in each leading dimension, to catch lda/ldb/ldc mix-ups.
  int pad;
};

// Runs one GEMM case and compares against the reference. The reference also
// computes, per element, the sum of absolute values of all terms, which
// scales the allowed rounding error.
template <typename T>
void CheckGemm(const GemmCase& tc, T alpha, T beta, uint64_t seed) {
  using R = RefT<T>;
  const int rows_a = tc.ta == 'N' ? tc.m : tc.k;
  const int cols_a = tc.ta == 'N' ? tc.k : tc.m;
  const int rows_b = tc.tb == 'N' ? tc.k : tc.n;
  const int cols_b = tc.tb == 'N' ? tc.n : tc.k;
  const int lda = std::max(1, rows_a + tc.pad);
  const int ldb = std::max(1, rows_b + tc.pad);
  const int ldc = std::max(1, tc.m + tc.pad);

  Random<T> rng(seed);
  const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * cols_a);
  const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * cols_b);
  std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) * tc.n);
  if (beta == T(0)) {
    // beta == 0 must not read C: poison it.
    for (T& x : c) x = T(std::numeric_limits<RealT<T>>::quiet_NaN());
  }
  const std::vector<T> c_orig = c;

  std::vector<R> expected(c.size());
  std::vector<double> bound(c.size(), 0.0);
  for (int j = 0; j < tc.n; ++j) {
    for (int i = 0; i < tc.m; ++i) {
      R sum = 0;
      double abs_sum = 0;
      for (int p = 0; p < tc.k; ++p) {
        const R ap = static_cast<R>(OpAt(tc.ta, a, i, p, lda));
        const R bp = static_cast<R>(OpAt(tc.tb, b, p, j, ldb));
        sum += ap * bp;
        abs_sum += std::abs(ap) * std::abs(bp);
      }
      const size_t idx = i + static_cast<size_t>(j) * ldc;
      R result = static_cast<R>(alpha) * sum;
      double scale = Abs(alpha) * abs_sum;
      if (beta != T(0)) {
        result += static_cast<R>(beta) * static_cast<R>(c_orig[idx]);
        scale += Abs(beta) * Abs(c_orig[idx]);
      }
      expected[idx] = result;
      // Floor protects elements whose exact value is (close to) zero.
      bound[idx] = scale + std::numeric_limits<RealT<T>>::min();
    }
  }

  Gemm<T>(tc.ta, tc.tb, tc.m, tc.n, tc.k, alpha, a.data(), lda, b.data(), ldb,
          beta, c.data(), ldc);

  // Standard bound: |error| <= gamma_k * scale with gamma_k ~ k * eps; a
  // small constant covers complex arithmetic and the alpha/beta updates.
  const double tol = 4.0 * (tc.k + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(c, expected, bound, tol, tc.m, tc.n, ldc))
      << "ta=" << tc.ta << " tb=" << tc.tb << " m=" << tc.m << " n=" << tc.n
      << " k=" << tc.k << " pad=" << tc.pad;

  // Padding rows of C must be untouched.
  for (int j = 0; j < tc.n; ++j) {
    for (int i = tc.m; i < ldc; ++i) {
      const size_t idx = i + static_cast<size_t>(j) * ldc;
      ASSERT_TRUE(c[idx] == c_orig[idx] || (std::isnan(std::real(c[idx])) &&
                                            std::isnan(std::real(c_orig[idx]))))
          << "padding of C modified at (" << i << ", " << j << ")";
    }
  }
}

template <typename T>
class GemmTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GemmTest, ScalarTypes);

// All small shapes: exercises every partial-tile combination.
TYPED_TEST(GemmTest, SmallShapes) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char ta : Ops<T>()) {
      for (char tb : Ops<T>()) {
        for (int m = 1; m <= 19; m += 3) {
          for (int n = 1; n <= 15; n += 2) {
            for (int k : {1, 2, 7, 16}) {
              CheckGemm<T>({ta, tb, m, n, k, 0}, MakeScalar<T>(1.5, -0.5),
                           MakeScalar<T>(0.75, 0.25), seed++);
            }
          }
        }
      }
    }
  });
}

// Shapes spanning several cache blocks (MC, KC, NC) with padded strides.
TYPED_TEST(GemmTest, BlockedShapes) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 100;
    const GemmCase shapes[] = {
        {'N', 'N', 64, 64, 64, 0},   {'N', 'N', 150, 35, 300, 3},
        {'N', 'N', 333, 17, 513, 1}, {'N', 'N', 31, 4100, 9, 0},
        {'N', 'N', 257, 129, 1, 2},  {'N', 'N', 1, 300, 600, 0},
    };
    for (char ta : Ops<T>()) {
      for (char tb : Ops<T>()) {
        for (GemmCase tc : shapes) {
          tc.ta = ta;
          tc.tb = tb;
          CheckGemm<T>(tc, MakeScalar<T>(-0.5, 2.0), MakeScalar<T>(1.0, 0.0),
                       seed++);
        }
      }
    }
  });
}

// Special values of alpha and beta take dedicated code paths.
TYPED_TEST(GemmTest, AlphaBetaSpecialCases) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1000;
    const double values[] = {0.0, 1.0, -1.0, 0.5};
    for (double alpha : values) {
      for (double beta : values) {
        for (int k : {0, 5, 300}) {
          CheckGemm<T>({'N', 'T', 37, 23, k, 2}, MakeScalar<T>(alpha, 0.0),
                       MakeScalar<T>(beta, 0.0), seed++);
        }
      }
    }
    if constexpr (IsComplex<T>::value) {
      // Purely imaginary alpha/beta.
      CheckGemm<T>({'C', 'N', 21, 13, 40, 1}, MakeScalar<T>(0.0, 1.0),
                   MakeScalar<T>(0.0, -1.0), seed++);
    }
  });
}

// m == 0 or n == 0 must not touch C at all.
TYPED_TEST(GemmTest, EmptyDimensions) {
  using T = TypeParam;
  const T alpha = MakeScalar<T>(1.0, 0.0), beta = MakeScalar<T>(0.0, 0.0);
  T a[4] = {}, b[4] = {};
  T c[4] = {T(7), T(7), T(7), T(7)};
  Gemm<T>('N', 'N', 0, 2, 2, alpha, a, 1, b, 2, beta, c, 1);
  Gemm<T>('N', 'N', 2, 0, 2, alpha, a, 2, b, 2, beta, c, 2);
  for (const T& x : c) EXPECT_EQ(x, T(7));
}

// Invalid arguments must be rejected without modifying C.
TYPED_TEST(GemmTest, InvalidArguments) {
  using T = TypeParam;
  const T one = MakeScalar<T>(1.0, 0.0);
  T a[4] = {one, one, one, one}, b[4] = {one, one, one, one};
  T c[4] = {T(7), T(7), T(7), T(7)};
  Gemm<T>('X', 'N', 2, 2, 2, one, a, 2, b, 2, one, c, 2);   // bad transa
  Gemm<T>('N', 'N', -1, 2, 2, one, a, 2, b, 2, one, c, 2);  // m < 0
  Gemm<T>('N', 'N', 2, 2, 2, one, a, 1, b, 2, one, c, 2);   // lda < m
  Gemm<T>('N', 'N', 2, 2, 2, one, a, 2, b, 2, one, c, 1);   // ldc < m
  for (const T& x : c) EXPECT_EQ(x, T(7));
}

}  // namespace
}  // namespace hwyblas_test
