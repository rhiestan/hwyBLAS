// Tests for the level 1 routines against references computed in double
// precision (or, for the rotation constructors, against their defining
// properties).
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
class Level1Test : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Level1Test, ScalarTypes);

// Lengths covering empty, scalar tails and several vectors.
const int kLengths[] = {0, 1, 2, 3, 7, 8, 17, 64, 1001};
// Increments: unit, strided and backwards.
const int kIncs[] = {1, 2, -1, -3};

// ---------------------------------------------------------------------------
// AXPY, SCAL, COPY, SWAP

TYPED_TEST(Level1Test, Axpy) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : kIncs) {
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

// SCAL, and CSSCAL/ZDSCAL (real alpha) for complex types.
TYPED_TEST(Level1Test, Scal) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : {1, 3}) {
        for (bool real_alpha : {false, true}) {
          if (real_alpha && !IsComplex<T>::value) continue;
          Random<T> rng(seed++);
          StridedVector<T> x(n, inc, rng);
          const std::vector<T> x0 = x.Elements();
          const T alpha = real_alpha ? MakeScalar<T>(-1.25, 0.0)
                                     : MakeScalar<T>(-1.25, 0.5);
          if (real_alpha) {
            const R ra = std::real(alpha);
            if constexpr (std::is_same_v<T, std::complex<float>>) {
              csscal_(&n, &ra, P(x.data.data()), &inc);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
              zdscal_(&n, &ra, P(x.data.data()), &inc);
            }
          } else {
            HWYBLAS_CALL(T, scal, &n, P(&alpha), P(x.data.data()), &inc);
          }
          std::vector<RefT<T>> expected(n);
          std::vector<double> bound(n);
          for (int i = 0; i < n; ++i) {
            expected[i] =
                static_cast<RefT<T>>(alpha) * static_cast<RefT<T>>(x0[i]);
            bound[i] = Abs(alpha) * Abs(x0[i]);
          }
          EXPECT_TRUE(NearWithBound(x.Elements(), expected, bound, 4 * Eps<T>(),
                                    n, 1, n))
              << "n=" << n << " inc=" << inc << " real_alpha=" << real_alpha;
        }
      }
    }
  });
}

TYPED_TEST(Level1Test, ScalSpecialCases) {
  using T = TypeParam;
  // incx <= 0 is a no-op.
  T x[3] = {T(1), T(2), T(3)};
  const T two(2);
  int n = 3, inc = 0;
  HWYBLAS_CALL(T, scal, &n, P(&two), P(x), &inc);
  inc = -1;
  HWYBLAS_CALL(T, scal, &n, P(&two), P(x), &inc);
  EXPECT_EQ(x[0], T(1));
  EXPECT_EQ(x[2], T(3));
  // As in the reference BLAS, alpha = 0 multiplies: NaN stays NaN.
  x[1] = NaN<T>();
  const T zero(0);
  inc = 1;
  HWYBLAS_CALL(T, scal, &n, P(&zero), P(x), &inc);
  EXPECT_EQ(x[0], T(0));
  EXPECT_TRUE(std::isnan(std::real(x[1])));
}

TYPED_TEST(Level1Test, CopySwap) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : kIncs) {
        Random<T> rng(seed++);
        StridedVector<T> x(n, inc, rng), y(n, inc == 1 ? 1 : -2, rng);
        const std::vector<T> x0 = x.Elements(), y0 = y.Elements();
        HWYBLAS_CALL(T, swap, &n, P(x.data.data()), &x.inc, P(y.data.data()),
                     &y.inc);
        EXPECT_EQ(x.Elements(), y0) << "swap n=" << n << " inc=" << inc;
        EXPECT_EQ(y.Elements(), x0) << "swap n=" << n << " inc=" << inc;
        HWYBLAS_CALL(T, copy, &n, P(x.data.data()), &x.inc, P(y.data.data()),
                     &y.inc);
        EXPECT_EQ(y.Elements(), y0) << "copy n=" << n << " inc=" << inc;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// DOT

// Reference dot product and its error scale.
template <typename T>
RefT<T> RefDot(bool conj, StridedVector<T>& x, StridedVector<T>& y,
               double* scale) {
  RefT<T> sum = 0;
  *scale = 0;
  for (size_t i = 0; i < x.n; ++i) {
    RefT<T> xi = static_cast<RefT<T>>(x[i]);
    if (conj) xi = Conj(xi);
    sum += xi * static_cast<RefT<T>>(y[i]);
    *scale += Abs(x[i]) * Abs(y[i]);
  }
  return sum;
}

TYPED_TEST(Level1Test, Dot) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : kIncs) {
        Random<T> rng(seed++);
        StridedVector<T> x(n, inc, rng), y(n, inc == 1 ? 1 : 2, rng);
        const double tol = 2.0 * (n + 2) * Eps<T>();
        for (bool conj : {false, true}) {
          if (conj && !IsComplex<T>::value) continue;
          double scale;
          const RefT<T> expected = RefDot(conj, x, y, &scale);
          RefT<T> got;
          if constexpr (std::is_same_v<T, float>) {
            got = sdot_(&n, x.data.data(), &x.inc, y.data.data(), &y.inc);
          } else if constexpr (std::is_same_v<T, double>) {
            got = ddot_(&n, x.data.data(), &x.inc, y.data.data(), &y.inc);
          } else if constexpr (std::is_same_v<R, float>) {
            const auto r = conj ? cdotc_(&n, P(x.data.data()), &x.inc,
                                         P(y.data.data()), &y.inc)
                                : cdotu_(&n, P(x.data.data()), &x.inc,
                                         P(y.data.data()), &y.inc);
            got = RefT<T>(r.real, r.imag);
          } else {
            const auto r = conj ? zdotc_(&n, P(x.data.data()), &x.inc,
                                         P(y.data.data()), &y.inc)
                                : zdotu_(&n, P(x.data.data()), &x.inc,
                                         P(y.data.data()), &y.inc);
            got = RefT<T>(r.real, r.imag);
          }
          EXPECT_LE(std::abs(got - expected), tol * scale + 1e-300)
              << "n=" << n << " inc=" << inc << " conj=" << conj;
        }
        if constexpr (std::is_same_v<T, float>) {
          // Accumulated in double: error of the order of double precision
          // plus the final rounding to float.
          double scale;
          const double expected = std::real(RefDot(false, x, y, &scale));
          const double wide =
              dsdot_(&n, x.data.data(), &x.inc, y.data.data(), &y.inc);
          EXPECT_LE(std::abs(wide - expected), 1e-12 * (scale + 1));
          const float sb = 0.25f;
          const float with_sb =
              sdsdot_(&n, &sb, x.data.data(), &x.inc, y.data.data(), &y.inc);
          EXPECT_EQ(with_sb, static_cast<float>(0.25 + wide));
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// NRM2, ASUM

// Calls ?nrm2_ (scnrm2_/dznrm2_ for complex types).
template <typename T>
double Nrm2(int n, StridedVector<T>& x) {
  int inc = x.inc;
  if constexpr (std::is_same_v<T, float>)
    return snrm2_(&n, x.data.data(), &inc);
  if constexpr (std::is_same_v<T, double>)
    return dnrm2_(&n, x.data.data(), &inc);
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return scnrm2_(&n, P(x.data.data()), &inc);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    return dznrm2_(&n, P(x.data.data()), &inc);
  }
}

// Reference norm with scaling (no overflow or underflow in double).
template <typename T>
double RefNorm(StridedVector<T>& x) {
  double big = 0;
  for (size_t i = 0; i < x.n; ++i) big = std::max(big, Abs(x[i]));
  if (big == 0) return 0;
  double sum = 0;
  for (size_t i = 0; i < x.n; ++i) {
    const double re = std::real(x[i]) / big, im = std::imag(x[i]) / big;
    sum += re * re + im * im;
  }
  return big * std::sqrt(sum);
}

TYPED_TEST(Level1Test, Nrm2) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    uint64_t seed = 1;
    // Scales near the limits of the type: plain sums of squares would
    // overflow (or underflow to zero).
    const R lim_big = std::numeric_limits<R>::max() / R(64);
    const R lim_small = std::numeric_limits<R>::min() * R(64);
    for (int n : kLengths) {
      for (int inc : {1, 3, -1, -2}) {
        for (R scale : {R(1), lim_big, lim_small}) {
          Random<T> rng(seed++);
          StridedVector<T> x(n, inc, rng);
          for (int i = 0; i < n; ++i) x[i] *= scale;
          const double expected = RefNorm(x);
          const double got = Nrm2(n, x);
          EXPECT_LE(std::abs(got - expected),
                    2.0 * (n + 2) * Eps<T>() * expected)
              << "n=" << n << " inc=" << inc << " scale=" << scale;
        }
      }
    }
    // Mixed magnitudes, and a zero increment (x[0] repeated n times).
    Random<T> rng(seed);
    StridedVector<T> x(5, 1, rng);
    x[0] = T(lim_big);
    x[3] = T(lim_small);
    EXPECT_NEAR(Nrm2(5, x) / RefNorm(x), 1.0, 4 * Eps<T>());
    x.inc = 0;
    EXPECT_NEAR(Nrm2(5, x) / (std::sqrt(5.0) * Abs(x.data[0])), 1.0,
                4 * Eps<T>());
    if constexpr (std::is_same_v<R, double>) {
      // Sums of squares below the range of the fast double path: medium
      // elements together with tiny ones, or medium ones only.
      StridedVector<T> y(2, 1, rng);
      for (double tiny : {1e-160, 0.0}) {
        y[0] = T(1e-152);
        y[1] = T(tiny);
        EXPECT_NEAR(Nrm2(2, y) / RefNorm(y), 1.0, 4 * Eps<T>()) << tiny;
      }
    }
  });
}

TYPED_TEST(Level1Test, Asum) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : {1, 2}) {
        Random<T> rng(seed++);
        StridedVector<T> x(n, inc, rng);
        double expected = 0;
        for (int i = 0; i < n; ++i) {
          expected += std::abs(std::real(x[i])) + std::abs(std::imag(x[i]));
        }
        double got;
        if constexpr (std::is_same_v<T, float>) {
          got = sasum_(&n, x.data.data(), &inc);
        } else if constexpr (std::is_same_v<T, double>) {
          got = dasum_(&n, x.data.data(), &inc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
          got = scasum_(&n, P(x.data.data()), &inc);
        } else {
          got = dzasum_(&n, P(x.data.data()), &inc);
        }
        EXPECT_LE(std::abs(got - expected), 2.0 * (n + 2) * Eps<T>() * expected)
            << "n=" << n << " inc=" << inc;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// I?AMAX

template <typename T>
int Iamax(int n, const T* x, int inc) {
  int result = 0;
  HWYBLAS_CALL_RESULT(T, result, amax, &n, P(x), &inc);
  return result;
}

// Reference: first index (1-based) of the largest |re| + |im|, NaNs never
// compare larger.
template <typename T>
int RefIamax(const std::vector<T>& x) {
  if (x.empty()) return 0;
  int best = 1;
  double best_value = std::abs(std::real(x[0])) + std::abs(std::imag(x[0]));
  for (size_t i = 1; i < x.size(); ++i) {
    const double v = std::abs(std::real(x[i])) + std::abs(std::imag(x[i]));
    if (v > best_value) {
      best = static_cast<int>(i) + 1;
      best_value = v;
    }
  }
  return best;
}

TYPED_TEST(Level1Test, Iamax) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : {1, 2}) {
        Random<T> rng(seed++);
        StridedVector<T> x(n, inc, rng);
        EXPECT_EQ(Iamax(n, x.data.data(), inc), RefIamax(x.Elements()))
            << "n=" << n << " inc=" << inc;
        if (n >= 4) {
          // Ties: the first occurrence wins. NaN (not first) is ignored.
          const T big = MakeScalar<T>(5.0, 0.0);
          x[n - 1] = big;
          x[1] = -big;
          x[2] = NaN<T>();
          EXPECT_EQ(Iamax(n, x.data.data(), inc), 2) << "n=" << n;
          // NaN in the first position is returned.
          x[0] = NaN<T>();
          EXPECT_EQ(Iamax(n, x.data.data(), inc), 1) << "n=" << n;
        }
      }
    }
    T one[1] = {T(1)};
    EXPECT_EQ(Iamax(1, one, 0), 0);  // incx <= 0
  });
}

// ---------------------------------------------------------------------------
// ROT, ROTM

TYPED_TEST(Level1Test, Rot) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int n : kLengths) {
      for (int inc : kIncs) {
        Random<T> rng(seed++);
        StridedVector<T> x(n, inc, rng), y(n, inc == 1 ? 1 : 2, rng);
        const std::vector<T> x0 = x.Elements(), y0 = y.Elements();
        const R c = R(0.6), s = R(-0.8);
        if constexpr (std::is_same_v<T, float>) {
          srot_(&n, x.data.data(), &x.inc, y.data.data(), &y.inc, &c, &s);
        } else if constexpr (std::is_same_v<T, double>) {
          drot_(&n, x.data.data(), &x.inc, y.data.data(), &y.inc, &c, &s);
        } else if constexpr (std::is_same_v<R, float>) {
          csrot_(&n, P(x.data.data()), &x.inc, P(y.data.data()), &y.inc, &c,
                 &s);
        } else {
          zdrot_(&n, P(x.data.data()), &x.inc, P(y.data.data()), &y.inc, &c,
                 &s);
        }
        const std::vector<T> x1 = x.Elements(), y1 = y.Elements();
        for (int i = 0; i < n; ++i) {
          const RefT<T> xi = static_cast<RefT<T>>(x0[i]);
          const RefT<T> yi = static_cast<RefT<T>>(y0[i]);
          const double tol = 4 * Eps<T>() * (Abs(x0[i]) + Abs(y0[i]));
          EXPECT_LE(std::abs(static_cast<RefT<T>>(x1[i]) -
                             (double(c) * xi + double(s) * yi)),
                    tol);
          EXPECT_LE(std::abs(static_cast<RefT<T>>(y1[i]) -
                             (double(c) * yi - double(s) * xi)),
                    tol);
        }
      }
    }
  });
}

TYPED_TEST(Level1Test, Rotm) {
  using T = TypeParam;
  if constexpr (!IsComplex<T>::value) {
    ForEachTarget([] {
      uint64_t seed = 1;
      for (T flag : {T(-2), T(-1), T(0), T(1)}) {
        for (int n : kLengths) {
          for (int inc : kIncs) {
            Random<T> rng(seed++);
            StridedVector<T> x(n, inc, rng), y(n, -inc, rng);
            const std::vector<T> x0 = x.Elements(), y0 = y.Elements();
            const T param[5] = {flag, T(0.5), T(-0.25), T(2), T(1.5)};
            // The full H implied by flag.
            double h11 = 1, h12 = 0, h21 = 0, h22 = 1;
            if (flag == -1) {
              h11 = param[1], h21 = param[2], h12 = param[3], h22 = param[4];
            } else if (flag == 0) {
              h21 = param[2], h12 = param[3];
            } else if (flag == 1) {
              h11 = param[1], h12 = 1, h21 = -1, h22 = param[4];
            }
            HWYBLAS_CALL_REAL(T, rotm, &n, x.data.data(), &x.inc, y.data.data(),
                              &y.inc, param);
            const std::vector<T> x1 = x.Elements(), y1 = y.Elements();
            for (int i = 0; i < n; ++i) {
              const double tol = 8 * Eps<T>() * (Abs(x0[i]) + Abs(y0[i]));
              EXPECT_NEAR(x1[i], h11 * x0[i] + h12 * y0[i], tol);
              EXPECT_NEAR(y1[i], h21 * x0[i] + h22 * y0[i], tol);
            }
          }
        }
      }
    });
  }
}

// ---------------------------------------------------------------------------
// ROTG, ROTMG, CABS1 (scalar)

TYPED_TEST(Level1Test, Rotg) {
  using T = TypeParam;
  using R = RealT<T>;
  const R big = std::numeric_limits<R>::max() / R(8);
  const R small = std::numeric_limits<R>::min() * R(8);
  // Near the square roots of the limits: the unscaled algorithm with
  // |f| << |g| and |f|^2 below its thresholds; with big, the scaled one with
  // a tiny |f / u|^2.
  const R sqrt_max = std::sqrt(std::numeric_limits<R>::max());
  const R sqrt_big = sqrt_max / R(64);
  const R sqrt_small = std::sqrt(std::numeric_limits<R>::min()) * R(64);
  const R values[] = {0,     1,      -2,       R(0.5),   big,       -big,
                      small, -small, sqrt_max, sqrt_big, sqrt_small};
  for (R are : values) {
    for (R bre : values) {
      for (R aim : {R(0), R(0.75)}) {
        for (R bim : {R(0), R(-3)}) {
          if (!IsComplex<T>::value && (aim != 0 || bim != 0)) continue;
          const T a0 = MakeScalar<T>(are, aim), b0 = MakeScalar<T>(bre, bim);
          T a = a0, b = b0, s;
          R c;
          if constexpr (IsComplex<T>::value) {
            HWYBLAS_CALL_COMPLEX(T, rotg, P(&a), P(&b), &c, P(&s));
          } else {
            HWYBLAS_CALL_REAL(T, rotg, &a, &b, &c, &s);
          }
          // [c s; -conj(s) c] [a0; b0] = [r; 0] with r = a, c^2 + |s|^2 = 1.
          // Compare in units of the input size to avoid overflow.
          using C = std::complex<double>;
          const double unit = std::max(Abs(a0), Abs(b0));
          if (unit == 0) {
            EXPECT_EQ(c, R(1));
            continue;
          }
          const C ca = C(std::real(a0), std::imag(a0)) / unit;
          const C cb = C(std::real(b0), std::imag(b0)) / unit;
          const C cs(std::real(s), std::imag(s));
          const C r = C(std::real(a), std::imag(a)) / unit;
          const double tol = 16 * Eps<T>();
          EXPECT_LE(std::abs(double(c) * ca + cs * cb - r), tol)
              << "a=" << are << "," << aim << " b=" << bre << "," << bim;
          EXPECT_LE(std::abs(-std::conj(cs) * ca + double(c) * cb), tol)
              << "a=" << are << "," << aim << " b=" << bre << "," << bim;
          EXPECT_NEAR(double(c) * c + std::norm(cs), 1.0, tol);
          if constexpr (!IsComplex<T>::value) {
            // b holds z, from which c and s can be reconstructed.
            const double z = b;
            double c2, s2;
            if (z == 1) {
              c2 = 0, s2 = 1;
            } else if (std::abs(z) < 1) {
              c2 = std::sqrt(1 - z * z), s2 = z;
            } else {
              c2 = 1 / z, s2 = std::sqrt(1 - c2 * c2);
            }
            EXPECT_NEAR(c2, c, tol);
            EXPECT_NEAR(s2, s, tol);
          }
        }
      }
    }
  }
}

TYPED_TEST(Level1Test, Rotmg) {
  using T = TypeParam;
  if constexpr (!IsComplex<T>::value) {
    Random<T> rng(1);
    for (int iter = 0; iter < 625; ++iter) {
      // d1 and d2 over many orders of magnitude, independently, so that the
      // rescaling of either or both runs, possibly several times.
      const int e1 = iter % 25 - 12, e2 = iter / 25 - 12;
      const T d1_in = std::abs(rng.Next()) * std::pow(T(10), T(e1));
      const T d2_in = rng.Next() * std::pow(T(10), T(e2));
      const T x1_in = rng.Next(), y1 = rng.Next();
      T d1 = d1_in, d2 = d2_in, x1 = x1_in, param[5] = {};
      HWYBLAS_CALL_REAL(T, rotmg, &d1, &d2, &x1, &y1, param);
      const T flag = param[0];
      if (flag == -2) continue;  // identity: nothing to check
      double h11 = 1, h12 = 0, h21 = 0, h22 = 1;
      if (flag == -1) {
        h11 = param[1], h21 = param[2], h12 = param[3], h22 = param[4];
      } else if (flag == 0) {
        h21 = param[2], h12 = param[3];
      } else {
        h11 = param[1], h12 = 1, h21 = -1, h22 = param[4];
      }
      if (d1 == 0 && d2 == 0 && x1 == 0) continue;  // the d2 * y1^2 < 0 case
      // H [x1_in; y1] = [x1; 0].
      const double tol =
          64 * Eps<T>() *
          (std::abs(h11 * x1_in) + std::abs(h12 * y1) + std::abs(x1));
      EXPECT_NEAR(h11 * x1_in + h12 * y1, x1, tol) << iter;
      EXPECT_NEAR(h21 * x1_in + h22 * y1, 0.0,
                  64 * Eps<T>() * (std::abs(h21 * x1_in) + std::abs(h22 * y1)))
          << iter;
      // The weights are preserved: d1 x1^2 = d1_in x1_in^2 + d2_in y1^2.
      const double lhs = double(d1) * x1 * x1;
      const double rhs =
          double(d1_in) * x1_in * x1_in + double(d2_in) * y1 * y1;
      EXPECT_NEAR(lhs, rhs, 64 * Eps<T>() * (std::abs(lhs) + std::abs(rhs)))
          << iter;
    }
  }
}

TEST(Level1Test, Cabs1) {
  const float zf[2] = {-1.5f, 2.0f};
  const double zd[2] = {3.0, -0.25};
  EXPECT_EQ(scabs1_(zf), 3.5f);
  EXPECT_EQ(dcabs1_(zd), 3.25);
}

}  // namespace
}  // namespace hwyblas_test
