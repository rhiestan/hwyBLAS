// Tests of the elementary reflectors: xLARFG, xLARF, xLARF1F, xLARF1L,
// xLARFT and xLARFB, against reflectors formed explicitly.
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class HouseholderTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HouseholderTest, ScalarTypes);

// TRANS for H**H: 'T' for real T, 'C' for complex T.
template <typename T>
char ConjTrans() {
  return IsComplex<T>::value ? 'C' : 'T';
}

// H = I - tau * v * v**H of order v.rows; with `bound`, the bound
// I + |tau| * |v| * |v|**T on the elements of |H| instead.
template <typename T>
Dense<T> Reflector(const Dense<T>& v, RefT<T> tau, bool bound = false) {
  Dense<T> h = Identity<T>(v.rows);
  for (size_t j = 0; j < v.rows; ++j) {
    for (size_t i = 0; i < v.rows; ++i) {
      if (bound) {
        h(i, j) += std::abs(tau) * std::abs(v(i, 0)) * std::abs(v(j, 0));
      } else {
        h(i, j) -= tau * v(i, 0) * Conj(v(j, 0));
      }
    }
  }
  return h;
}

// ||expected - actual||_1 / (order * ||h_bound||_1 * ||c||_1 * eps): the
// error of applying a reflector to c, with the bound h_bound on the
// magnitude of its elements (see Reflector). With random tau and v, the
// reflector is not orthogonal / unitary, and its elements can be much
// smaller than the terms whose rounding errors the result has.
template <typename T>
double ApplyRatio(const Dense<T>& expected, const Dense<T>& actual,
                  const Dense<T>& h_bound, const Dense<T>& c, size_t order) {
  const double scale = Norm1(h_bound) * Norm1(c);
  const double resid = Norm1(Sub(expected, actual));
  if (scale == 0) return resid / LapackEps<T>();
  return resid / static_cast<double>(order) / scale / LapackEps<T>();
}

// Element i (0-based) of the vector x with n elements and increment incx,
// with the BLAS convention for incx < 0.
template <typename T>
T& Element(std::vector<T>& x, size_t n, hwyblas_int incx, size_t i) {
  const size_t inc = static_cast<size_t>(std::abs(incx));
  return x[(incx > 0 ? i : n - 1 - i) * inc];
}

template <typename T>
void Larfg(hwyblas_int n, T* alpha, std::vector<T>& x, hwyblas_int incx,
           T* tau) {
  HWYBLAS_CALL(T, larfg, &n, P(alpha), P(x.data()), &incx, P(tau));
}

// Checks that the reflector of xLARFG (on input alpha0, x0) maps (alpha0;
// x0) to (beta; 0) with real beta, |beta| = ||(alpha0; x0)||, and that tau
// is in the documented range.
template <typename T>
void CheckLarfg(hwyblas_int n, T alpha0, const std::vector<T>& x0, T beta,
                const std::vector<T>& x, hwyblas_int incx, T tau) {
  const size_t m = static_cast<size_t>(n);
  Dense<T> v(m, 1), y(m, 1);
  v(0, 0) = 1;
  y(0, 0) = static_cast<RefT<T>>(alpha0);
  std::vector<T> xv = x, x0v = x0;
  for (size_t i = 1; i < m; ++i) {
    v(i, 0) = static_cast<RefT<T>>(Element(xv, m - 1, incx, i - 1));
    y(i, 0) = static_cast<RefT<T>>(Element(x0v, m - 1, incx, i - 1));
  }
  // ||y||_2, scaled (y may be near the underflow threshold).
  double ymax = 0, sum = 0;
  for (size_t i = 0; i < m; ++i) ymax = std::max(ymax, std::abs(y(i, 0)));
  for (size_t i = 0; i < m && ymax > 0; ++i) {
    sum += std::norm(y(i, 0) / ymax);
  }
  const double norm = ymax * std::sqrt(sum);

  EXPECT_EQ(std::imag(beta), 0);
  const RefT<T> t = static_cast<RefT<T>>(tau);
  if (t != RefT<T>(0)) {
    EXPECT_GE(std::real(t), 1 - 4 * Eps<T>());
    EXPECT_LE(std::real(t), 2 + 4 * Eps<T>());
    EXPECT_LE(std::abs(t - RefT<T>(1)), 1 + 4 * Eps<T>());
  }
  // H**H * (alpha0; x0) = (beta; 0).
  const Dense<T> hy = Mul(Adjoint(Reflector(v, t)), y);
  Dense<T> expected(m, 1);
  expected(0, 0) = static_cast<RefT<T>>(beta);
  const double tol = 30 * m * LapackEps<T>() * std::max(norm, 1e-300);
  for (size_t i = 0; i < m; ++i) {
    EXPECT_LE(std::abs(hy(i, 0) - expected(i, 0)), tol) << "row " << i;
  }
  EXPECT_NEAR(std::abs(static_cast<RefT<T>>(beta)), norm, tol);
}

TYPED_TEST(HouseholderTest, Larfg) {
  using T = TypeParam;
  Random<T> rng(3);
  for (const hwyblas_int n : {1, 2, 3, 10, 41}) {
    // (LAPACK calls xLARFG with incx > 0 only: xSCAL does nothing for
    // incx < 0.)
    for (const hwyblas_int incx : {1, 2}) {
      // Random, and scaled below xLAMCH('S') / xLAMCH('E') (2e-31 for
      // float, 2e-292 for double), where xLARFG rescales x and recomputes
      // beta.
      for (const double scale : {1.0, 1e-32, 1e-300}) {
        if (scale < 1e4 * std::numeric_limits<RealT<T>>::min()) {
          continue;  // would underflow T
        }
        SCOPED_TRACE(::testing::Message()
                     << "n=" << n << " incx=" << incx << " scale=" << scale);
        const auto s = static_cast<RealT<T>>(scale);
        const T alpha0 = rng.Next() * s;
        std::vector<T> x0(static_cast<size_t>(std::max(n - 1, 0)) *
                              static_cast<size_t>(std::abs(incx)) +
                          1);
        for (T& e : x0) e = rng.Next() * s;
        T alpha = alpha0, tau;
        std::vector<T> x = x0;
        Larfg(n, &alpha, x, incx, &tau);
        // Real n = 1: H = I.
        if (!IsComplex<T>::value && n == 1) {
          EXPECT_EQ(tau, T(0));
          EXPECT_EQ(alpha, alpha0);
        }
        CheckLarfg(n, alpha0, x0, alpha, x, incx, tau);
      }
    }
  }
}

// x = 0: H = I (tau = 0) for real alpha; for complex alpha with an
// imaginary part, H makes it real.
TYPED_TEST(HouseholderTest, LarfgZeroVector) {
  using T = TypeParam;
  for (const hwyblas_int n : {1, 4}) {
    std::vector<T> x(3, T(0));
    T alpha = T(-2.5), tau = T(9);
    Larfg(n, &alpha, x, 1, &tau);
    EXPECT_EQ(tau, T(0));
    EXPECT_EQ(alpha, T(-2.5));
    if constexpr (IsComplex<T>::value) {
      const T alpha0 = T(3, 4);
      alpha = alpha0;
      Larfg(n, &alpha, x, 1, &tau);
      EXPECT_NE(tau, T(0));
      CheckLarfg(n, alpha0, std::vector<T>(3, T(0)), alpha, x, 1, tau);
      EXPECT_NEAR(std::real(alpha), -5, 1e-5);
    }
  }
  // n = 0: tau = 0.
  T alpha = T(1), tau = T(9);
  std::vector<T> x(1, T(1));
  Larfg(0, &alpha, x, 1, &tau);
  EXPECT_EQ(tau, T(0));
}

// Which of the three routines that apply one reflector.
enum class Apply { kLarf, kLarf1f, kLarf1l };

template <typename T>
void ApplyReflector(Apply which, char side, hwyblas_int m, hwyblas_int n,
                    const std::vector<T>& v, hwyblas_int incv, T tau,
                    std::vector<T>& c, hwyblas_int ldc, std::vector<T>& work) {
  switch (which) {
    case Apply::kLarf:
      HWYBLAS_CALL(T, larf, &side, &m, &n, P(v.data()), &incv, P(&tau),
                   P(c.data()), &ldc, P(work.data()));
      break;
    case Apply::kLarf1f:
      HWYBLAS_CALL(T, larf1f, &side, &m, &n, P(v.data()), &incv, P(&tau),
                   P(c.data()), &ldc, P(work.data()));
      break;
    case Apply::kLarf1l:
      HWYBLAS_CALL(T, larf1l, &side, &m, &n, P(v.data()), &incv, P(&tau),
                   P(c.data()), &ldc, P(work.data()));
      break;
  }
}

// H * C or C * H for v with zeros at its end (xLARF, xLARF1F) or start
// (xLARF1L), C with zero rows and columns at its end, and tau = 0.
TYPED_TEST(HouseholderTest, ApplyOne) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(17);
    for (const Apply which : {Apply::kLarf, Apply::kLarf1f, Apply::kLarf1l}) {
      for (const char side : {'L', 'R'}) {
        for (const hwyblas_int m : {1, 2, 7, 20}) {
          for (const hwyblas_int n : {1, 3, 9}) {
            for (const hwyblas_int incv : {1, 2}) {
              for (const int zeros : {0, 1, 100}) {  // zeros in v (capped)
                for (const bool zero_c : {false, true}) {
                  for (const bool zero_tau : {false, true}) {
                    const hwyblas_int order = (side == 'L') ? m : n;
                    const int nz = std::min(zeros, order - 1);
                    SCOPED_TRACE(::testing::Message()
                                 << "routine=" << static_cast<int>(which)
                                 << " side=" << side << " m=" << m << " n=" << n
                                 << " incv=" << incv << " zeros=" << nz
                                 << " zero_c=" << zero_c
                                 << " zero_tau=" << zero_tau);
                    // v: logical elements 0 .. order-1 at index i*incv.
                    std::vector<T> v(static_cast<size_t>(order) * incv,
                                     NaN<T>());
                    Dense<T> vd(order, 1);
                    for (hwyblas_int i = 0; i < order; ++i) {
                      T e = rng.Next();
                      const bool leading = which == Apply::kLarf1l;
                      if (leading ? (i < nz) : (i >= order - nz)) e = T(0);
                      // The implicit unit element is not referenced.
                      const bool unit =
                          (which == Apply::kLarf1f && i == 0) ||
                          (which == Apply::kLarf1l && i == order - 1);
                      if (unit) {
                        vd(i, 0) = 1;
                        continue;
                      }
                      v[static_cast<size_t>(i) * incv] = e;
                      vd(i, 0) = static_cast<RefT<T>>(e);
                    }
                    const T tau = zero_tau ? T(0) : rng.Next() + T(1);
                    const hwyblas_int ldc = m + 1;
                    std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) * n);
                    if (zero_c) {
                      // The last rows (side L) or columns (side R) of C
                      // that v multiplies are zero, and some others.
                      for (hwyblas_int j = 0; j < n; ++j) {
                        for (hwyblas_int i = 0; i < m; ++i) {
                          if (i >= m / 2 || j >= n / 2) At(c, i, j, ldc) = T(0);
                        }
                      }
                    }
                    const std::vector<T> c0 = c;
                    std::vector<T> work(std::max(m, n), NaN<T>());
                    ApplyReflector(which, side, m, n, v, incv, tau, c, ldc,
                                   work);
                    const Dense<T> h = Reflector(vd, static_cast<RefT<T>>(tau));
                    const Dense<T> c0d = ToDense(c0, m, n, ldc);
                    const Dense<T> expected =
                        (side == 'L') ? Mul(h, c0d) : Mul(c0d, h);
                    EXPECT_LT(
                        ApplyRatio(
                            expected, ToDense(c, m, n, ldc),
                            Reflector(vd, static_cast<RefT<T>>(tau), true), c0d,
                            order),
                        kLapackThreshold);
                    for (hwyblas_int j = 0; j < n; ++j) {
                      ASSERT_EQ(At(c, m, j, ldc), At(c0, m, j, ldc));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

// The k reflectors of order n stored in v as xLARFT expects (direct =
// 'F'/'B', storev = 'C'/'R'), with NaN in the elements that are 1 or 0 by
// definition; returns their vectors as the columns of an n x k matrix.
template <typename T>
Dense<T> StoredReflectors(char direct, char storev, size_t n, size_t k,
                          std::vector<T>& v, size_t ldv, Random<T>& rng) {
  const bool col = storev == 'C';
  v.assign(ldv * (col ? k : n), NaN<T>());
  Dense<T> vecs(n, k);
  for (size_t i = 0; i < k; ++i) {
    // The position of the unit element of reflector i.
    const size_t unit = (direct == 'F') ? i : n - k + i;
    for (size_t r = 0; r < n; ++r) {
      RefT<T> e;
      if (r == unit) {
        e = 1;
      } else if ((direct == 'F') ? r < unit : r > unit) {
        e = 0;
      } else {
        const T x = rng.Next();
        if (col) {
          At(v, r, i, ldv) = x;
          e = static_cast<RefT<T>>(x);
        } else {
          // Row storage holds conj(v): H = I - V**H * T * V.
          At(v, i, r, ldv) = x;
          e = Conj(static_cast<RefT<T>>(x));
        }
      }
      vecs(r, i) = e;
    }
  }
  return vecs;
}

// H = H(1) H(2) . . . H(k) (direct = 'F') or H(k) . . . H(1) ('B'); with
// `bound`, the product of the bounds of the reflectors (see Reflector).
template <typename T>
Dense<T> BlockReflector(char direct, const Dense<T>& vecs,
                        const std::vector<T>& tau, bool bound = false) {
  const size_t n = vecs.rows, k = vecs.cols;
  Dense<T> h = Identity<T>(n);
  for (size_t s = 0; s < k; ++s) {
    const size_t i = (direct == 'F') ? s : k - 1 - s;
    Dense<T> vi(n, 1);
    for (size_t r = 0; r < n; ++r) vi(r, 0) = vecs(r, i);
    h = Mul(h, Reflector(vi, static_cast<RefT<T>>(tau[i]), bound));
  }
  return h;
}

template <typename T>
void Larft(char direct, char storev, hwyblas_int n, hwyblas_int k,
           const std::vector<T>& v, hwyblas_int ldv, const std::vector<T>& tau,
           std::vector<T>& t, hwyblas_int ldt) {
  HWYBLAS_CALL(T, larft, &direct, &storev, &n, &k, P(v.data()), &ldv,
               P(tau.data()), P(t.data()), &ldt);
}

// The triangular factor of xLARFT: I - V * T * V**H (storev = 'C') or
// I - V**H * T * V ('R') is the product of the reflectors.
TYPED_TEST(HouseholderTest, Larft) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(23);
    for (const char direct : {'F', 'B'}) {
      for (const char storev : {'C', 'R'}) {
        for (const hwyblas_int n : {1, 2, 5, 17, 40}) {
          for (const hwyblas_int k : {1, 2, 3, 8, 17}) {
            if (k > n) continue;
            SCOPED_TRACE(::testing::Message()
                         << "direct=" << direct << " storev=" << storev
                         << " n=" << n << " k=" << k);
            const hwyblas_int ldv = (storev == 'C' ? n : k) + 2;
            std::vector<T> v;
            const Dense<T> vecs =
                StoredReflectors<T>(direct, storev, n, k, v, ldv, rng);
            std::vector<T> tau(k);
            for (T& x : tau) x = rng.Next() + T(1);
            const hwyblas_int ldt = k + 1;
            const T sentinel = MakeScalar<T>(5, 5);
            std::vector<T> t(static_cast<size_t>(ldt) * k, sentinel);
            Larft(direct, storev, n, k, v, ldv, tau, t, ldt);

            Dense<T> td(k, k);
            for (hwyblas_int j = 0; j < k; ++j) {
              for (hwyblas_int i = 0; i < ldt; ++i) {
                const bool in_triangle =
                    i < k && ((direct == 'F') ? i <= j : i >= j);
                if (in_triangle) {
                  td(i, j) = static_cast<RefT<T>>(At(t, i, j, ldt));
                } else {
                  ASSERT_EQ(At(t, i, j, ldt), sentinel) << i << ", " << j;
                }
              }
            }
            const Dense<T> h =
                Sub(Identity<T>(n), Mul(Mul(vecs, td), Adjoint(vecs)));
            EXPECT_LT(ResidualRatio(BlockReflector(direct, vecs, tau), h, n),
                      kLapackThreshold);
          }
        }
      }
    }
  });
}

// xLARFB applies H or H**H from either side, with T from xLARFT.
TYPED_TEST(HouseholderTest, Larfb) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(29);
    for (const char direct : {'F', 'B'}) {
      for (const char storev : {'C', 'R'}) {
        for (const char side : {'L', 'R'}) {
          for (const char trans : {'N', ConjTrans<T>()}) {
            for (const hwyblas_int m : {1, 4, 19}) {
              for (const hwyblas_int n : {1, 6, 23}) {
                for (const hwyblas_int k : {1, 3, 6}) {
                  const hwyblas_int order = (side == 'L') ? m : n;
                  if (k > order) continue;
                  SCOPED_TRACE(::testing::Message()
                               << "direct=" << direct << " storev=" << storev
                               << " side=" << side << " trans=" << trans
                               << " m=" << m << " n=" << n << " k=" << k);
                  const hwyblas_int ldv = (storev == 'C' ? order : k) + 1;
                  std::vector<T> v;
                  const Dense<T> vecs = StoredReflectors<T>(
                      direct, storev, order, k, v, ldv, rng);
                  std::vector<T> tau(k);
                  for (T& x : tau) x = rng.Next() + T(1);
                  const hwyblas_int ldt = k;
                  std::vector<T> t(static_cast<size_t>(ldt) * k, T(0));
                  Larft(direct, storev, order, k, v, ldv, tau, t, ldt);

                  const hwyblas_int ldc = m + 2;
                  std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) * n);
                  const std::vector<T> c0 = c;
                  const hwyblas_int ldwork = (side == 'L') ? n : m;
                  std::vector<T> work(static_cast<size_t>(ldwork) * k);
                  HWYBLAS_CALL(T, larfb, &side, &trans, &direct, &storev, &m,
                               &n, &k, P(v.data()), &ldv, P(t.data()), &ldt,
                               P(c.data()), &ldc, P(work.data()), &ldwork);

                  Dense<T> h = BlockReflector(direct, vecs, tau);
                  if (trans != 'N') h = Adjoint(h);
                  const Dense<T> c0d = ToDense(c0, m, n, ldc);
                  const Dense<T> expected =
                      (side == 'L') ? Mul(h, c0d) : Mul(c0d, h);
                  EXPECT_LT(ApplyRatio(expected, ToDense(c, m, n, ldc),
                                       BlockReflector(direct, vecs, tau, true),
                                       c0d, order),
                            kLapackThreshold);
                  for (hwyblas_int j = 0; j < n; ++j) {
                    for (hwyblas_int i = m; i < ldc; ++i) {
                      ASSERT_EQ(At(c, i, j, ldc), At(c0, i, j, ldc));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

}  // namespace
}  // namespace hwyblas_test
