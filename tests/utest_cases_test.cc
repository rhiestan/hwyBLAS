// Edge cases ported from OpenBLAS's unit tests (utest/ and
// utest/test_extensions/, Copyright (c) The OpenBLAS Project, BSD 3-Clause
// License): zero increments, NaN/Inf propagation, extreme values, and
// reference results of the rotation constructors (from the reference BLAS
// and gonum, as given there). Cases for OpenBLAS-only functions are omitted.
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

template <typename T>
class UtestCases : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(UtestCases, ScalarTypes);

// The tolerances of the OpenBLAS tests, relative to max(1, |expected|).
template <typename T>
double Tol() {
  return std::is_same_v<RealT<T>, float> ? 1e-4 : 1e-13;
}

template <typename T>
void ExpectNear(double expected, double got, const char* what) {
  EXPECT_NEAR(got, expected, Tol<T>() * std::max(1.0, std::abs(expected)))
      << what;
}

// Builds a vector of T from interleaved (re, im) pairs for complex T, or
// from plain values for real T.
template <typename T>
std::vector<T> Make(std::initializer_list<double> values) {
  std::vector<T> v;
  const std::vector<double> d(values);
  if constexpr (IsComplex<T>::value) {
    for (size_t i = 0; i + 1 < d.size(); i += 2) {
      v.push_back(T(RealT<T>(d[i]), RealT<T>(d[i + 1])));
    }
  } else {
    for (double x : d) v.push_back(T(x));
  }
  return v;
}

template <typename T>
void ExpectVector(const std::vector<T>& expected, const std::vector<T>& got,
                  const char* what) {
  ASSERT_EQ(expected.size(), got.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ExpectNear<T>(std::real(expected[i]), std::real(got[i]), what);
    ExpectNear<T>(std::imag(expected[i]), std::imag(got[i]), what);
  }
}

// ---------------------------------------------------------------------------
// Zero increments: every iteration of the reference loop updates the same
// element (test_axpy.c, test_swap.c, test_rot.c, test_axpby.c).

TYPED_TEST(UtestCases, AxpyZeroIncrements) {
  using T = TypeParam;
  ForEachTarget([] {
    int zero = 0, one = 1;
    if constexpr (IsComplex<T>::value) {
      int n = 4;
      const T a(RealT<T>(0.25), RealT<T>(0.5));
      std::vector<T> x = Make<T>({1, 3, 5, 7, 1, 3, 5, 7});
      std::vector<T> y = Make<T>({2, 4, 6, 8, 2, 4, 6, 8});
      const std::vector<T> x0 = x, y0 = y;
      HWYBLAS_CALL(T, axpy, &n, P(&a), P(x.data()), &zero, P(y.data()), &zero);
      ExpectVector(x0, x, "x, incx = incy = 0");
      ExpectVector(Make<T>({-3, 9, 6, 8, 2, 4, 6, 8}), y, "incx = incy = 0");
      y = y0;
      HWYBLAS_CALL(T, axpy, &n, P(&a), P(x.data()), &zero, P(y.data()), &one);
      ExpectVector(Make<T>({0.75, 5.25, 4.75, 9.25, 0.75, 5.25, 4.75, 9.25}), y,
                   "incx = 0");
    } else {
      int n = 8;
      const T a(0.25);
      std::vector<T> x = Make<T>({1, 3, 5, 7, 1, 3, 5, 7});
      std::vector<T> y = Make<T>({2, 4, 6, 8, 2, 4, 6, 8});
      const std::vector<T> x0 = x;
      HWYBLAS_CALL(T, axpy, &n, P(&a), P(x.data()), &zero, P(y.data()), &zero);
      ExpectVector(x0, x, "x, incx = incy = 0");
      ExpectVector(Make<T>({4, 4, 6, 8, 2, 4, 6, 8}), y, "incx = incy = 0");
    }
  });
}

TYPED_TEST(UtestCases, SwapZeroIncrements) {
  using T = TypeParam;
  ForEachTarget([] {
    // Four swaps of the same pair: no change.
    int n = 4, zero = 0;
    std::vector<T> x = Make<T>({1, 3, 5, 7, 1, 3, 5, 7});
    std::vector<T> y = Make<T>({2, 4, 6, 8, 2, 4, 6, 8});
    const std::vector<T> x0 = x, y0 = y;
    HWYBLAS_CALL(T, swap, &n, P(x.data()), &zero, P(y.data()), &zero);
    ExpectVector(x0, x, "x");
    ExpectVector(y0, y, "y");
  });
}

TYPED_TEST(UtestCases, RotZeroIncrements) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    int n = 4, zero = 0;
    const R c(0.25), s(0.5);
    std::vector<T> x = Make<T>({1, 3, 5, 7, 1, 3, 5, 7});
    std::vector<T> y = Make<T>({2, 4, 6, 8, 2, 4, 6, 8});
    if constexpr (std::is_same_v<T, float>) {
      srot_(&n, x.data(), &zero, y.data(), &zero, &c, &s);
    } else if constexpr (std::is_same_v<T, double>) {
      drot_(&n, x.data(), &zero, y.data(), &zero, &c, &s);
    } else if constexpr (std::is_same_v<R, float>) {
      csrot_(&n, P(x.data()), &zero, P(y.data()), &zero, &c, &s);
    } else {
      zdrot_(&n, P(x.data()), &zero, P(y.data()), &zero, &c, &s);
    }
    if constexpr (IsComplex<T>::value) {
      ExpectVector(Make<T>({-0.21484375, -0.45703125, 5, 7, 1, 3, 5, 7}), x,
                   "x");
      ExpectVector(Make<T>({0.0390625, 0.171875, 6, 8, 2, 4, 6, 8}), y, "y");
    } else {
      ExpectVector(Make<T>({-0.21484375, 3, 5, 7, 1, 3, 5, 7}), x, "x");
      ExpectVector(Make<T>({0.0390625, 4, 6, 8, 2, 4, 6, 8}), y, "y");
    }
  });
}

TYPED_TEST(UtestCases, AxpbyZeroIncrements) {
  using T = TypeParam;
  ForEachTarget([] {
    // y(1) := beta * y(1) + alpha * x(1), nine times.
    int n = 9, zero = 0;
    std::vector<T> x =
        Make<T>({1, 3, 5, 7, 1, 3, 5, 7, 9, 1, 3, 5, 7, 1, 3, 5, 7, 9});
    std::vector<T> y =
        Make<T>({2, 4, 6, 8, 2, 4, 6, 8, 10, 2, 4, 6, 8, 2, 4, 6, 8, 10});
    std::vector<T> expected = y;
    if constexpr (IsComplex<T>::value) {
      const T alpha(1, 2), beta(2, 1);
      HWYBLAS_CALL(T, axpby, &n, P(&alpha), P(x.data()), &zero, P(&beta),
                   P(y.data()), &zero);
      expected[0] = T(9355, -8865);
    } else {
      const T alpha(1), beta(2);
      HWYBLAS_CALL(T, axpby, &n, &alpha, x.data(), &zero, &beta, y.data(),
                   &zero);
      expected[0] = T(1535);
    }
    ExpectVector(expected, y, "y");
  });
}

// ---------------------------------------------------------------------------
// Dot products (test_dsdot.c, test_dotu.c)

TEST(UtestCases, DsdotSingleElement) {
  const float x = 0.172555164F, y = -0.0138700781F;
  const int n = 1, inc = 1;
  // The product of two floats is exact in double.
  EXPECT_NEAR(dsdot_(&n, &x, &inc, &y, &inc), -0.00239335360107, 1e-13);
}

TYPED_TEST(UtestCases, DotuSingleElement) {
  using T = TypeParam;
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    const int n = 1, inc = 1;
    const auto dotu = [&](const R* x, const R* y) {
      if constexpr (std::is_same_v<R, float>) {
        const auto r = cdotu_(&n, x, &inc, y, &inc);
        return std::complex<double>(r.real, r.imag);
      } else {
        const auto r = zdotu_(&n, x, &inc, y, &inc);
        return std::complex<double>(r.real, r.imag);
      }
    };
    const R x1[] = {1, 1}, y1[] = {1, 2};
    const std::complex<double> r1 = dotu(x1, y1);
    ExpectNear<T>(-1, r1.real(), "re");
    ExpectNear<T>(3, r1.imag(), "im");
    // A complex number starting at an odd real offset.
    const R x2[] = {1, 2, 3, 4}, y2[] = {5, 6, 7, 8};
    const std::complex<double> r2 = dotu(x2 + 1, y2 + 1);
    ExpectNear<T>(-9, r2.real(), "re, offset");
    ExpectNear<T>(32, r2.imag(), "im, offset");
  }
}

// ---------------------------------------------------------------------------
// NRM2 (test_dnrm2.c)

template <typename T>
double Nrm2(int n, const T* x, int incx) {
  if constexpr (std::is_same_v<T, float>) {
    return snrm2_(&n, x, &incx);
  } else if constexpr (std::is_same_v<T, double>) {
    return dnrm2_(&n, x, &incx);
  } else if constexpr (std::is_same_v<RealT<T>, float>) {
    return scnrm2_(&n, P(x), &incx);
  } else {
    return dznrm2_(&n, P(x), &incx);
  }
}

TYPED_TEST(UtestCases, Nrm2Extremes) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    const int n = 28;
    // An infinite element among zeros.
    std::vector<T> x(n, T(0));
    x[10] = T(-std::numeric_limits<R>::infinity());
    EXPECT_EQ(Nrm2(n, x.data(), 1), std::numeric_limits<double>::infinity());
    // Subnormal elements: the result is sqrt(28) * value, not 0. (The
    // elements have few significant bits, hence the loose tolerance.)
    const R tiny = std::numeric_limits<R>::denorm_min() * R(1000);
    x.assign(n, T(tiny));
    const double expected = std::sqrt(double(n)) * double(tiny);
    EXPECT_NEAR(Nrm2(n, x.data(), 1), expected, 1e-2 * expected);
    // Negative increment.
    x.assign(5, T(10));
    ExpectNear<T>(std::sqrt(500.0), Nrm2(5, x.data(), -1), "incx = -1");
  });
}

// ---------------------------------------------------------------------------
// SCAL with NaN and Inf (test_zscal.c): the reference multiplies, so
// 0 * NaN and 0 * Inf give NaN, and complex products follow the plain
// formula (ar * xr - ai * xi, ar * xi + ai * xr).

TYPED_TEST(UtestCases, ScalNanInf) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    const R nan = std::numeric_limits<R>::quiet_NaN();
    const R inf = std::numeric_limits<R>::infinity();
    for (int inc : {1, 2}) {
      const int n = 9;
      // Scales n elements at stride inc of value x by alpha; returns them.
      const auto scal = [&](T alpha, T x) {
        std::vector<T> v(n * inc, x);
        HWYBLAS_CALL(T, scal, &n, P(&alpha), P(v.data()), &inc);
        return v;
      };
      if constexpr (!IsComplex<T>::value) {
        const R pairs[][2] = {{0, nan}, {nan, 0},   {0, inf},
                              {inf, 0}, {nan, inf}, {inf, nan}};
        for (const auto& p : pairs) {
          const std::vector<T> v = scal(p[0], p[1]);
          EXPECT_TRUE(std::isnan(v[0]) && std::isnan(v[(n - 1) * inc]))
              << "alpha=" << p[0] << " x=" << p[1] << " inc=" << inc;
        }
      } else {
        struct Case {
          T alpha, x;
          bool re_nan, re_inf, im_nan, im_inf;
        };
        const T i(0, 1);
        const Case cases[] = {
            {i, T(nan, 0), true, false, true, false},
            {T(nan, 0), i, true, false, true, false},
            {i, T(inf, 0), true, false, false, true},
            {T(inf, 0), i, true, false, false, true},
            {i, T(0, inf), false, true, true, false},
            {T(0, 0), T(nan, 0), true, false, true, false},
            {T(0, 0), T(0, nan), true, false, true, false},
        };
        for (const Case& c : cases) {
          for (const T& r :
               {scal(c.alpha, c.x)[0], scal(c.alpha, c.x)[(n - 1) * inc]}) {
            EXPECT_EQ(std::isnan(r.real()), c.re_nan) << r << " inc=" << inc;
            EXPECT_EQ(std::isinf(r.real()), c.re_inf) << r << " inc=" << inc;
            EXPECT_EQ(std::isnan(r.imag()), c.im_nan) << r << " inc=" << inc;
            EXPECT_EQ(std::isinf(r.imag()), c.im_inf) << r << " inc=" << inc;
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// GEMV with NaN and Inf in y (test_gemv.c): beta = 0 sets y to zero without
// reading it; any other beta multiplies.

TYPED_TEST(UtestCases, GemvNanInfInY) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    const R nan = std::numeric_limits<R>::quiet_NaN();
    const R inf = std::numeric_limits<R>::infinity();
    const int n = 17;
    const std::vector<T> a(n * n, T(0)), x(n, T(0));
    for (int incy : {1, 2}) {
      // y alternates NaN and Inf (complex: with imaginary part 1).
      const auto run = [&](T beta) {
        std::vector<T> y(n * incy, T(0));
        for (int i = 0; i < n; ++i) {
          const R v = (i % 2 == 0) ? nan : inf;
          y[i * incy] = IsComplex<T>::value ? MakeScalar<T>(v, 1) : T(v);
        }
        const T alpha(0);
        const int inc = 1;
        HWYBLAS_CALL(T, gemv, "N", &n, &n, P(&alpha), P(a.data()), &n,
                     P(x.data()), &inc, P(&beta), P(y.data()), &incy);
        return y;
      };
      const std::vector<T> y0 = run(T(0));
      for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(y0[i * incy] == T(0)) << "beta = 0, incy=" << incy;
      }
      if constexpr (IsComplex<T>::value) {
        // beta = 2i: (NaN, 1) -> (NaN, NaN), (Inf, 1) -> (NaN, Inf).
        const std::vector<T> y1 = run(T(0, 2));
        // beta = 2: (NaN, 1) -> (NaN, NaN), (Inf, 1) -> (Inf, NaN).
        const std::vector<T> y2 = run(T(2, 0));
        for (int i = 0; i < n; ++i) {
          const T v1 = y1[i * incy], v2 = y2[i * incy];
          if (i % 2 == 0) {
            EXPECT_TRUE(std::isnan(v1.real()) && std::isnan(v1.imag())) << v1;
            EXPECT_TRUE(std::isnan(v2.real()) && std::isnan(v2.imag())) << v2;
          } else {
            EXPECT_TRUE(std::isnan(v1.real()) && std::isinf(v1.imag())) << v1;
            EXPECT_TRUE(std::isinf(v2.real()) && std::isnan(v2.imag())) << v2;
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// ROTG for complex arguments (test_extensions/test_zrotg.c)

TYPED_TEST(UtestCases, ComplexRotg) {
  using T = TypeParam;
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    struct Case {
      double a[2], b[2];
      double c, s[2], r[2];
    };
    const Case cases[] = {
        {{0, 0},
         {1, 1},
         0,
         {0.70710678118655, -0.70710678118655},
         {1.41421356237310, 0}},
        {{1, 1}, {0, 0}, 1, {0, 0}, {1, 1}},
        {{0, 1},
         {0, 1},
         0.70710678118654,
         {0.70710678118654, 0},
         {0, 1.41421356237309}},
        {{3, 4},
         {4, 6},
         0.56980288229818,
         {0.82051615050939, -0.04558423058385},
         {5.26497863243527, 7.01997150991369}},
        {{-3, 4},
         {-4, 6},
         0.56980288229818,
         {0.82051615050939, 0.04558423058385},
         {-5.26497863243527, 7.01997150991369}},
        {{3, -4},
         {4, -6},
         0.56980288229818,
         {0.82051615050939, 0.04558423058385},
         {5.26497863243527, -7.01997150991369}},
        {{-3, -4},
         {-4, -6},
         0.56980288229818,
         {0.82051615050939, -0.04558423058385},
         {-5.26497863243527, -7.01997150991369}},
    };
    for (const Case& t : cases) {
      T a(R(t.a[0]), R(t.a[1])), b(R(t.b[0]), R(t.b[1])), s;
      R c;
      HWYBLAS_CALL_COMPLEX(T, rotg, P(&a), P(&b), &c, P(&s));
      ExpectNear<T>(t.c, c, "c");
      ExpectNear<T>(t.s[0], s.real(), "re(s)");
      ExpectNear<T>(t.s[1], s.imag(), "im(s)");
      ExpectNear<T>(t.r[0], a.real(), "re(r)");
      ExpectNear<T>(t.r[1], a.imag(), "im(r)");
    }
  }
}

// ---------------------------------------------------------------------------
// ROTMG (test_rotmg.c, test_extensions/test_drotmg.c): d1, d2, x1, y1 and
// all of param, including the elements the flag leaves unchanged.

TYPED_TEST(UtestCases, Rotmg) {
  using T = TypeParam;
  if constexpr (!IsComplex<T>::value) {
    struct Case {
      const char* name;
      double d1, d2, x1, y1, param_in[5];
      double d1_out, d2_out, x1_out, y1_out, param_out[5];
    };
    const Case cases[] = {
        // Rescaling bug of an older implementation (reference BLAS values).
        {"libGoto",
         0.21149573940783739,
         0.046892057172954082,
         -0.42272687517106533,
         0.42211309121921659,
         {0, 0, 0, 0, 0},
         0.17320483687975,
         0.03840233915037,
         -0.51618034832329,
         0.42211309121922,
         {0, 0, 0.99854803659786, -0.22139439665872, 0}},
        // OpenBLAS issue #1452 (gonum values).
        {"issue1452",
         5.9e-8,
         5.960464e-8,
         1.0,
         150.0,
         {0, 0, 0, 1. / 4096., 0},
         0.99995592822897,
         0.98981219860583,
         0.03662270484346,
         150.0,
         {-1, 0.00000161109346, -0.00024414062500, 0.00024414062500,
          0.00000162760417}},
        {"d1 == d2, x1 == y1",
         2,
         2,
         8,
         8,
         {0, 0, 0, 0, 0},
         1,
         1,
         16,
         8,
         {1, 1, 0, 0, 1}},
        // Large d1, d2: rescaled (gonum values).
        {"large d",
         1600000000.,
         800000000.,
         8,
         7,
         {1, 4096, -4096, 1, 4096},
         68.96627824858757,
         34.483139124293785,
         45312,
         7,
         {-1, 4096, -3584, 1792, 4096}},
        {"y1 == 0", 2, 2, 8, 0, {0, 0, 0, 0, 0}, 2, 2, 8, 0, {-2, 0, 0, 0, 0}},
        {"d1 < 0", -1, 2, 8, 8, {0, 0, 0, 0, 0}, 0, 0, 0, 8, {-1, 0, 0, 0, 0}},
        {"x1 == 0", 2, 2, 0, 8, {0, 0, 0, 0, 0}, 2, 2, 8, 8, {1, 0, 0, 0, 0}},
        {"d2 * y1^2 < 0, larger than d1 * x1^2",
         1,
         -2,
         8,
         8,
         {0, 0, 0, 0, 0},
         0,
         0,
         0,
         8,
         {-1, 0, 0, 0, 0}},
    };
    for (const Case& t : cases) {
      T d1 = T(t.d1), d2 = T(t.d2), x1 = T(t.x1), param[5];
      const T y1 = T(t.y1);
      for (int i = 0; i < 5; ++i) param[i] = T(t.param_in[i]);
      HWYBLAS_CALL_REAL(T, rotmg, &d1, &d2, &x1, &y1, param);
      ExpectNear<T>(t.d1_out, d1, t.name);
      ExpectNear<T>(t.d2_out, d2, t.name);
      ExpectNear<T>(t.x1_out, x1, t.name);
      ExpectNear<T>(t.y1_out, y1, t.name);
      for (int i = 0; i < 5; ++i)
        ExpectNear<T>(t.param_out[i], param[i], t.name);
    }
  }
}

// ---------------------------------------------------------------------------
// I?AMIN (test_ismin.c, test_extensions/test_i?amin.c)

TYPED_TEST(UtestCases, Iamin) {
  using T = TypeParam;
  ForEachTarget([] {
    const int n = 50;
    for (int inc : {1, 2}) {
      // Values 1000, 1001, ... (complex: in both parts), with a zero at
      // element 9 or at the last element.
      for (int at : {8, n - 1}) {
        std::vector<T> x(n * inc);
        for (int i = 0; i < n * inc; ++i) {
          x[i] = MakeScalar<T>(1000 + 2 * i, 1001 + 2 * i);
        }
        x[at * inc] = T(0);
        int index;
        HWYBLAS_CALL_RESULT(T, index, amin, &n, P(x.data()), &inc);
        EXPECT_EQ(index, at + 1) << "inc=" << inc;
      }
    }
    // A zero increment returns 0, as for i?amax.
    std::vector<T> x(n, T(1));
    x[8] = T(0);
    int zero = 0, index;
    HWYBLAS_CALL_RESULT(T, index, amin, &n, P(x.data()), &zero);
    EXPECT_EQ(index, 0);
  });
}

}  // namespace
}  // namespace hwyblas_test
