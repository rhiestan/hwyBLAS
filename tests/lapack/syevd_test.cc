// Tests of the divide and conquer symmetric (Hermitian) eigensolvers:
// xSTEDC and its xLAED* routines, xSYEVD / xHEEVD, xSTEVD, the product
// with the Q of the tridiagonal reduction (xORMTR / xUNMTR, xORMQL /
// xUNMQL, xORM2L / xUNM2L), the generalized problems (xSYGST / xHEGST,
// xSYGS2 / xHEGS2, xSYGV / xHEGV, xSYGVD / xHEGVD), and LAPACKE.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/eigen_test_util.h"
#include "lapack/lapack_test_util.h"
#include "lapack/linear_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class SyevdTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SyevdTest, ScalarTypes);

template <typename R>
class SyevdRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SyevdRealTest, RealTypes);

// xSTEDC with the workspaces of a workspace query (which are the minimal
// ones), or with the given ones if lwork > 0.
template <typename T>
hwyblas_int Stedc(char compz, hwyblas_int n, std::vector<RealT<T>>& d,
                  std::vector<RealT<T>>& e, std::vector<T>& z, hwyblas_int ldz,
                  hwyblas_int lwork = -1, hwyblas_int lrwork = -1,
                  hwyblas_int liwork = -1) {
  using R = RealT<T>;
  hwyblas_int info = -99;
  if (lwork == -1 || lrwork == -1 || liwork == -1) {
    std::vector<T> wq(1);
    std::vector<R> rq(1);
    std::vector<hwyblas_int> iq(1);
    const hwyblas_int m1 = -1;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, stedc, &compz, &n, d.data(), e.data(),
                           P(z.data()), &ldz, P(wq.data()), &m1, rq.data(), &m1,
                           iq.data(), &m1, &info);
      if (lrwork == -1) lrwork = QueryResult(rq[0]);
    } else {
      HWYBLAS_CALL_REAL(T, stedc, &compz, &n, d.data(), e.data(), z.data(),
                        &ldz, wq.data(), &m1, iq.data(), &m1, &info);
    }
    if (info != 0) return info;
    if (lwork == -1) lwork = QueryResult(wq[0]);
    if (liwork == -1) liwork = iq[0];
  }
  std::vector<T> work(std::max<hwyblas_int>(1, lwork), NaN<T>());
  std::vector<R> rwork(std::max<hwyblas_int>(1, lrwork), NAN);
  std::vector<hwyblas_int> iwork(std::max<hwyblas_int>(1, liwork), -77);
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, stedc, &compz, &n, d.data(), e.data(), P(z.data()),
                         &ldz, P(work.data()), &lwork, rwork.data(), &lrwork,
                         iwork.data(), &liwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, stedc, &compz, &n, d.data(), e.data(), z.data(), &ldz,
                      work.data(), &lwork, iwork.data(), &liwork, &info);
  }
  return info;
}

// A random orthogonal (unitary) n x n matrix (leading dimension ld): the Q
// of a QR factorization of a random matrix.
template <typename T>
std::vector<T> RandomUnitary(hwyblas_int n, hwyblas_int ld, Random<T>& rng) {
  std::vector<T> q = rng.Vector(ld * n);
  std::vector<T> tau(std::max<hwyblas_int>(1, n));
  hwyblas_int info = -99, lwork = 64 * std::max<hwyblas_int>(1, n);
  std::vector<T> work(lwork);
  HWYBLAS_CALL(T, geqrf, &n, &n, P(q.data()), &ld, P(tau.data()),
               P(work.data()), &lwork, &info);
  CALL_RC(T, orgqr, ungqr, &n, &n, &n, P(q.data()), &ld, P(tau.data()),
          P(work.data()), &lwork, &info);
  return q;
}

// The subproblem sizes of the tests: LAPACK's (25) and 3, which splits small
// matrices into several levels of divide and conquer.
const hwyblas_int kSubproblemSizes[] = {0, 3};

// xSTEDC ('I', 'V', 'N'): eigenvalues in ascending order, eigenvectors
// with small residuals that are orthonormal, the same eigenvalues for all
// COMPZ and as xSTEQR (up to rounding), on tridiagonal matrices that split,
// are graded, have clustered eigenvalues or are scaled.
TYPED_TEST(SyevdTest, Stedc) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<R> rng(211);
  Random<T> crng(223);
  for (const hwyblas_int smlsiz : kSubproblemSizes) {
    const ScopedSubproblemSize subproblems(smlsiz);
    for (const int kind : {0, 1, 2, 3, 4, 5}) {
      for (const hwyblas_int n : {1, 2, 3, 4, 7, 10, 33, 70}) {
        SCOPED_TRACE(::testing::Message()
                     << "smlsiz=" << smlsiz << " kind=" << kind << " n=" << n);
        std::vector<R> d0, e0;
        TestTridiagonal(kind, n, rng, d0, e0);
        // Dense T in scaled form (the residual ratios are scale invariant).
        R anorm = 0;
        for (R x : d0) anorm = std::max(anorm, std::abs(x));
        for (R x : e0) anorm = std::max(anorm, std::abs(x));
        std::vector<R> ds = d0, es = e0;
        for (auto& x : ds) x /= anorm;
        for (auto& x : es) x /= anorm;
        const Dense<T> t = Tridiagonal<T>(ds, es);

        // 'I': eigenvectors of T.
        std::vector<R> d = d0, e = e0;
        e.resize(std::max<hwyblas_int>(1, n - 1));
        const hwyblas_int ldz = n + 1;
        std::vector<T> z(ldz * n, NaN<T>());
        ASSERT_EQ(Stedc('I', n, d, e, z, ldz), 0);
        ExpectAscending(d);
        std::vector<R> w = d;
        for (auto& x : w) x /= anorm;
        const Dense<T> zd = ToDense(z, n, n, ldz);
        EXPECT_LT(OrthogonalityRatio(zd), kLapackThreshold);
        EXPECT_LT(EigenResidual(t, zd, w), kLapackThreshold);

        // 'V': Z = Q * (eigenvectors of T) for orthogonal (unitary) Q.
        std::vector<T> q = RandomUnitary(n, ldz, crng);
        const Dense<T> qd = ToDense(q, n, n, ldz);
        std::vector<R> dv = d0, ev = e0;
        ev.resize(std::max<hwyblas_int>(1, n - 1));
        ASSERT_EQ(Stedc('V', n, dv, ev, q, ldz), 0);
        ExpectAscending(dv);
        const Dense<T> qz = ToDense(q, n, n, ldz);
        // Q**H * (Q * Z) are the eigenvectors of T.
        std::vector<R> wv = dv;
        for (auto& x : wv) x /= anorm;
        EXPECT_LT(EigenResidual(t, Mul(Adjoint(qd), qz), wv), kLapackThreshold);
        EXPECT_LT(OrthogonalityRatio(qz), kLapackThreshold);

        // 'N' (xSTERF) and xSTEQR: the same eigenvalues (up to rounding).
        std::vector<R> dn = d0, en = e0;
        en.resize(std::max<hwyblas_int>(1, n - 1));
        std::vector<T> dummy(1);
        ASSERT_EQ(Stedc('N', n, dn, en, dummy, 1), 0);
        std::vector<R> dq = d0, eq = e0;
        std::vector<T> zq(ldz * n);
        std::vector<R> work(std::max<hwyblas_int>(1, 2 * n - 2));
        hwyblas_int info = -99;
        const char compz = 'I';
        HWYBLAS_CALL(T, steqr, &compz, &n, dq.data(), eq.data(), P(zq.data()),
                     &ldz, work.data(), &info);
        ASSERT_EQ(info, 0);
        const double tol =
            30.0 * n * LapackEps<T>() * static_cast<double>(anorm);
        for (hwyblas_int i = 0; i < n; ++i) {
          EXPECT_NEAR(dv[i], d[i], tol) << i;
          EXPECT_NEAR(dn[i], d[i], tol) << i;
          EXPECT_NEAR(dq[i], d[i], tol) << i;
        }
      }
    }
  }
}

// xSYEVD / xHEEVD with the minimal workspaces (minimal) or the optimal ones
// of a workspace query.
template <typename T>
hwyblas_int Heevd(char jobz, char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda, std::vector<RealT<T>>& w, bool minimal) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  hwyblas_int lwork, lrwork, liwork, info = -99;
  if (minimal) {
    const bool v = jobz == 'V';
    lwork = n <= 1     ? 1
            : kComplex ? (v ? 2 * n + n * n : n + 1)
                       : (v ? 1 + 6 * n + 2 * n * n : 2 * n + 1);
    lrwork = n <= 1 ? 1 : (v ? 1 + 5 * n + 2 * n * n : n);
    liwork = n <= 1 || !v ? 1 : 3 + 5 * n;
  } else {
    std::vector<T> wq(1);
    std::vector<R> rq(1);
    std::vector<hwyblas_int> iq(1);
    const hwyblas_int m1 = -1;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevd, &jobz, &uplo, &n, P(a.data()), &lda,
                           w.data(), P(wq.data()), &m1, rq.data(), &m1,
                           iq.data(), &m1, &info);
    } else {
      HWYBLAS_CALL_REAL(T, syevd, &jobz, &uplo, &n, a.data(), &lda, w.data(),
                        wq.data(), &m1, iq.data(), &m1, &info);
    }
    if (info != 0) return info;
    lwork = QueryResult(wq[0]);
    lrwork = QueryResult(rq[0]);
    liwork = iq[0];
  }
  std::vector<T> work(lwork, NaN<T>());
  std::vector<R> rwork(std::max<hwyblas_int>(1, lrwork), NAN);
  std::vector<hwyblas_int> iwork(liwork, -77);
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heevd, &jobz, &uplo, &n, P(a.data()), &lda,
                         w.data(), P(work.data()), &lwork, rwork.data(),
                         &lrwork, iwork.data(), &liwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, syevd, &jobz, &uplo, &n, a.data(), &lda, w.data(),
                      work.data(), &lwork, iwork.data(), &liwork, &info);
  }
  return info;
}

struct Blocking {
  hwyblas_int nb, nx;
};

// xSYEVD / xHEEVD: A * V = V * diag(w) with orthonormal V and ascending w,
// the same w without eigenvectors and as xSYEV / xHEEV (up to rounding),
// also for matrices that the driver scales (norm near the underflow or
// overflow threshold), with blocked and unblocked reductions, several
// levels of divide and conquer and minimal workspaces.
TYPED_TEST(SyevdTest, Heevd) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(227);
    const double rmin = std::sqrt(static_cast<double>(
        std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon()));
    for (const Blocking& b : {Blocking{0, 0}, Blocking{3, 5}}) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const hwyblas_int smlsiz : kSubproblemSizes) {
        const ScopedSubproblemSize subproblems(smlsiz);
        for (const char uplo : {'U', 'L'}) {
          for (const hwyblas_int n : {1, 2, 5, 17, 40}) {
            for (const double scale : {1.0, rmin * 1e-3, 1 / (rmin * 1e-3)}) {
              for (const bool minimal : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "uplo=" << uplo << " n=" << n << " nb=" << b.nb
                             << " smlsiz=" << smlsiz << " scale=" << scale
                             << " minimal=" << minimal);
                const hwyblas_int lda = n + 1;
                std::vector<T> a = RandomHermitian(uplo, n, lda, rng);
                const Dense<T> ad = HermitianDense(uplo, a, n, lda);
                for (T& x : a) x = x * static_cast<R>(scale);
                std::vector<T> v = a;
                std::vector<R> w(n);
                ASSERT_EQ(Heevd('V', uplo, n, v, lda, w, minimal), 0);
                ExpectAscending(w);
                for (R& x : w) x = static_cast<R>(x / scale);
                const Dense<T> vd = ToDense(v, n, n, lda);
                EXPECT_LT(OrthogonalityRatio(vd), kLapackThreshold);
                EXPECT_LT(EigenResidual(ad, vd, w), kLapackThreshold);

                std::vector<T> an = a;
                std::vector<R> wn(n);
                ASSERT_EQ(Heevd('N', uplo, n, an, lda, wn, minimal), 0);
                double wmax = 0;
                for (R x : w) {
                  wmax = std::max(wmax, std::abs(static_cast<double>(x)));
                }
                for (hwyblas_int i = 0; i < n; ++i) {
                  EXPECT_NEAR(wn[i] / scale, w[i], 30.0 * n * Eps<T>() * wmax)
                      << i;
                }
              }
            }
          }
        }
      }
    }
  });
}

// xSTEVD: as xSTEDC with COMPZ = 'I' and 'N', also for tridiagonal matrices
// that it scales.
TYPED_TEST(SyevdRealTest, Stevd) {
  using R = TypeParam;
  Random<R> rng(229);
  for (const hwyblas_int smlsiz : kSubproblemSizes) {
    const ScopedSubproblemSize subproblems(smlsiz);
    for (const int kind : {0, 1, 2, 3, 4, 5}) {
      for (const hwyblas_int n : {0, 1, 2, 9, 40}) {
        for (const bool minimal : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "smlsiz=" << smlsiz << " kind=" << kind << " n=" << n
                       << " minimal=" << minimal);
          std::vector<R> d0, e0;
          TestTridiagonal(kind, n, rng, d0, e0);
          R anorm = 0;
          for (R x : d0) anorm = std::max(anorm, std::abs(x));
          for (R x : e0) anorm = std::max(anorm, std::abs(x));
          std::vector<R> ds = d0, es = e0;
          for (auto& x : ds) x /= anorm;
          for (auto& x : es) x /= anorm;
          const Dense<R> t = Tridiagonal<R>(ds, es);
          for (const char jobz : {'V', 'N'}) {
            std::vector<R> d = d0, e = e0;
            e.resize(std::max<hwyblas_int>(1, n - 1));
            const hwyblas_int ldz = std::max<hwyblas_int>(1, n);
            std::vector<R> z(ldz * std::max<hwyblas_int>(1, n), NAN);
            hwyblas_int lwork = 1, liwork = 1, info = -99;
            if (n > 1 && jobz == 'V') {
              lwork = 1 + 4 * n + n * n;
              liwork = 3 + 5 * n;
            }
            if (!minimal) {
              R wq = 0;
              hwyblas_int iq = 0;
              const hwyblas_int m1 = -1;
              HWYBLAS_CALL_REAL(R, stevd, &jobz, &n, d.data(), e.data(),
                                z.data(), &ldz, &wq, &m1, &iq, &m1, &info);
              ASSERT_EQ(info, 0);
              ASSERT_EQ(QueryResult(wq), lwork);
              ASSERT_EQ(iq, liwork);
            }
            std::vector<R> work(lwork, NAN);
            std::vector<hwyblas_int> iwork(liwork, -77);
            HWYBLAS_CALL_REAL(R, stevd, &jobz, &n, d.data(), e.data(), z.data(),
                              &ldz, work.data(), &lwork, iwork.data(), &liwork,
                              &info);
            ASSERT_EQ(info, 0);
            ExpectAscending(d);
            if (jobz == 'V' && n > 0) {
              std::vector<R> w = d;
              for (auto& x : w) x /= anorm;
              const Dense<R> zd = ToDense(z, n, n, ldz);
              EXPECT_LT(OrthogonalityRatio(zd), kLapackThreshold);
              EXPECT_LT(EigenResidual(t, zd, w), kLapackThreshold);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The secular equation

// xLAED4 (and xLAED5 for n = 2): the eigenvalues of diag(d) + rho * z * z**T
// interlace with d, and for n >= 3 they are roots of the secular equation
// 1 + rho * sum(z(j)**2 / (d(j) - lambda)), whose differences d(j) - lambda
// DELTA holds; for n = 2, DELTA is the eigenvector, and for n = 1 one.
TYPED_TEST(SyevdRealTest, Laed4) {
  using R = TypeParam;
  Random<R> rng(241);
  for (const hwyblas_int n : {1, 2, 3, 5, 20}) {
    for (const R rho : {R(0.125), R(1), R(10)}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " rho=" << rho);
      // Distinct ascending d, z of norm 1.
      std::vector<R> d(n), z(n);
      R sum = 0, znorm = 0;
      for (hwyblas_int j = 0; j < n; ++j) {
        sum += R(0.1) + std::abs(rng.Next());
        d[j] = sum - R(n) / 2;
        z[j] = rng.Next();
        znorm += z[j] * z[j];
      }
      for (R& x : z) x /= std::sqrt(znorm);
      for (hwyblas_int i = 1; i <= n; ++i) {
        std::vector<R> delta(n, NAN);
        R dlam = NAN;
        hwyblas_int info = -99;
        HWYBLAS_CALL_REAL(R, laed4, &n, &i, d.data(), z.data(), delta.data(),
                          &rho, &dlam, &info);
        ASSERT_EQ(info, 0) << i;
        EXPECT_GT(dlam, d[i - 1]) << i;
        // The largest eigenvalue is at most d(n) + rho * |z|**2 (equal for
        // n = 1).
        if (i < n) {
          EXPECT_LT(dlam, d[i]) << i;
        } else {
          EXPECT_LE(dlam, d[n - 1] + rho) << i;
        }
        if (n == 1) {
          EXPECT_EQ(delta[0], 1);
        } else if (n == 2) {
          // (diag(d) + rho * z * z**T) * delta = dlam * delta, |delta| = 1.
          const R zd = z[0] * delta[0] + z[1] * delta[1];
          for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(d[j] * delta[j] + rho * z[j] * zd, dlam * delta[j],
                        8 * Eps<R>() * (std::abs(d[j]) + rho + std::abs(dlam)));
          }
          EXPECT_NEAR(delta[0] * delta[0] + delta[1] * delta[1], 1,
                      8 * Eps<R>());
          // xLAED5 directly gives the same.
          std::vector<R> delta5(2, NAN);
          R dlam5 = NAN;
          HWYBLAS_CALL_REAL(R, laed5, &i, d.data(), z.data(), delta5.data(),
                            &rho, &dlam5);
          EXPECT_EQ(dlam5, dlam);
          EXPECT_EQ(delta5, delta);
        } else {
          double f = 1, fabs = 1;
          for (hwyblas_int j = 0; j < n; ++j) {
            EXPECT_NEAR(delta[j], d[j] - dlam,
                        8 * Eps<R>() * (std::abs(d[j]) + std::abs(dlam)))
                << j;
            const double t = static_cast<double>(rho) * z[j] * z[j] / delta[j];
            f += t;
            fabs += std::abs(t);
          }
          EXPECT_LT(std::abs(f), 30 * n * Eps<R>() * fabs) << i;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The products with Q

// Q (nq x nq) of a reduction to tridiagonal form by xSYTRD / xHETRD of the
// random Hermitian a (triangle uplo), generated by xORGTR / xUNGTR; a
// returns the reflectors and tau their scalars.
template <typename T>
Dense<T> TridiagonalQ(char uplo, hwyblas_int nq, std::vector<T>& a,
                      hwyblas_int lda, std::vector<T>& tau) {
  using R = RealT<T>;
  std::vector<R> d(std::max<hwyblas_int>(1, nq)),
      e(std::max<hwyblas_int>(1, nq));
  tau.assign(std::max<hwyblas_int>(1, nq), T(0));
  hwyblas_int info = -99, lwork = 64 * std::max<hwyblas_int>(1, nq);
  std::vector<T> work(lwork);
  CALL_RC(T, sytrd, hetrd, &uplo, &nq, P(a.data()), &lda, d.data(), e.data(),
          P(tau.data()), P(work.data()), &lwork, &info);
  EXPECT_EQ(info, 0);
  std::vector<T> q = a;
  CALL_RC(T, orgtr, ungtr, &uplo, &nq, P(q.data()), &lda, P(tau.data()),
          P(work.data()), &lwork, &info);
  EXPECT_EQ(info, 0);
  return ToDense(q, nq, nq, lda);
}

// op(Q) * C (side 'L') or C * op(Q) (side 'R').
template <typename T>
Dense<T> ApplyQ(char side, char trans, const Dense<T>& q, const Dense<T>& c) {
  const Dense<T> op = trans == 'N' ? q : Adjoint(q);
  return side == 'L' ? Mul(op, c) : Mul(c, op);
}

// ||result - expected||_1 / (max(1, nq) * eps * ||C||_1).
template <typename T>
double ProductRatio(const Dense<T>& result, const Dense<T>& expected,
                    const Dense<T>& c, hwyblas_int nq) {
  const double cnorm = std::max(Norm1(c), 1e-300);
  return Norm1(Sub(result, expected)) /
         (std::max<double>(1, static_cast<double>(nq)) * LapackEps<T>() *
          cnorm);
}

// xORMTR / xUNMTR: op(Q) * C and C * op(Q) for the Q of xSYTRD / xHETRD,
// equal to the products with the Q of xORGTR / xUNGTR, through the blocked
// (several blocks) and the unblocked xORMQL / xUNMQL and xORMQR / xUNMQR,
// with minimal workspaces; the rows of C beyond M are not touched.
TYPED_TEST(SyevdTest, Unmtr) {
  using T = TypeParam;
  Random<T> rng(233);
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  for (const hwyblas_int nb : {0, 1, 3}) {
    const ScopedBlockSize block_size(nb);
    for (const char uplo : {'U', 'L'}) {
      for (const char side : {'L', 'R'}) {
        for (const char trans : {'N', ctrans}) {
          for (const hwyblas_int nq : {1, 2, 3, 7, 20}) {
            for (const hwyblas_int other : {1, 3, 5}) {
              for (const bool minimal : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "nb=" << nb << " uplo=" << uplo << " side="
                             << side << " trans=" << trans << " nq=" << nq
                             << " other=" << other << " minimal=" << minimal);
                const hwyblas_int lda = nq + 1;
                std::vector<T> a = RandomHermitian(uplo, nq, lda, rng);
                std::vector<T> tau;
                const Dense<T> q = TridiagonalQ(uplo, nq, a, lda, tau);
                const hwyblas_int m = side == 'L' ? nq : other;
                const hwyblas_int n = side == 'L' ? other : nq;
                const hwyblas_int ldc = m + 2;
                std::vector<T> c = rng.Vector(ldc * n);
                const std::vector<T> c0 = c;
                const Dense<T> cd = ToDense(c, m, n, ldc);
                hwyblas_int lwork =
                    std::max<hwyblas_int>(1, side == 'L' ? n : m);
                hwyblas_int info = -99;
                if (!minimal) {
                  T query = T(0);
                  const hwyblas_int m1 = -1;
                  CALL_RC(T, ormtr, unmtr, &side, &uplo, &trans, &m, &n,
                          P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                          P(&query), &m1, &info);
                  ASSERT_EQ(info, 0);
                  lwork = QueryResult(query);
                }
                std::vector<T> work(lwork, NaN<T>());
                CALL_RC(T, ormtr, unmtr, &side, &uplo, &trans, &m, &n,
                        P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                        P(work.data()), &lwork, &info);
                ASSERT_EQ(info, 0);
                EXPECT_LT(ProductRatio(ToDense(c, m, n, ldc),
                                       ApplyQ(side, trans, q, cd), cd, nq),
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
}

// xORMQL / xUNMQL (blocked and unblocked) and xORM2L / xUNM2L: the products
// with Q = H(k) . . . H(1) of K reflectors from xLARFG, equal to those with
// the Q of xORGQL / xUNGQL, without referencing the elements of A below the
// unit elements of the reflectors (NaN here).
TYPED_TEST(SyevdTest, Unmql) {
  using T = TypeParam;
  Random<T> rng(251);
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  for (const hwyblas_int nb : {0, 1, 2, 3}) {
    const ScopedBlockSize block_size(nb);
    for (const char side : {'L', 'R'}) {
      for (const char trans : {'N', ctrans}) {
        for (const hwyblas_int nq : {1, 3, 8, 20}) {
          for (const hwyblas_int k : {0, 1, nq / 2, nq}) {
            SCOPED_TRACE(::testing::Message()
                         << "nb=" << nb << " side=" << side
                         << " trans=" << trans << " nq=" << nq << " k=" << k);
            // The reflectors: column i has its unit element in row
            // nq-k+i (1-based).
            const hwyblas_int lda = nq + 1;
            std::vector<T> a(lda * std::max<hwyblas_int>(1, k), NaN<T>());
            std::vector<T> tau(std::max<hwyblas_int>(1, k));
            for (hwyblas_int i = 1; i <= k; ++i) {
              hwyblas_int len = nq - k + i;
              const hwyblas_int inc = 1;
              for (hwyblas_int r = 0; r < len; ++r) {
                At(a, r, i - 1, lda) = rng.Next();
              }
              HWYBLAS_CALL(T, larfg, &len, P(&At(a, len - 1, i - 1, lda)),
                           P(&At(a, 0, i - 1, lda)), &inc, P(&tau[i - 1]));
            }
            // Q of order nq: the reflectors in its last k columns.
            std::vector<T> qa(lda * std::max<hwyblas_int>(1, nq), T(0));
            for (hwyblas_int i = 0; i < k; ++i) {
              for (hwyblas_int r = 0; r < nq; ++r) {
                At(qa, r, nq - k + i, lda) = At(a, r, i, lda);
              }
            }
            hwyblas_int info = -99, lq = 64 * std::max<hwyblas_int>(1, nq);
            std::vector<T> wq(lq);
            {
              const ScopedBlockSize unblocked(1);
              CALL_RC(T, orgql, ungql, &nq, &nq, &k, P(qa.data()), &lda,
                      P(tau.data()), P(wq.data()), &lq, &info);
            }
            ASSERT_EQ(info, 0);
            const Dense<T> q = ToDense(qa, nq, nq, lda);
            const hwyblas_int other = 4;
            const hwyblas_int m = side == 'L' ? nq : other;
            const hwyblas_int n = side == 'L' ? other : nq;
            const hwyblas_int ldc = m + 1;
            const std::vector<T> c0 = rng.Vector(ldc * n);
            const Dense<T> cd = ToDense(c0, m, n, ldc);
            const Dense<T> expected = ApplyQ(side, trans, q, cd);
            for (const bool minimal : {false, true}) {
              std::vector<T> c = c0;
              hwyblas_int lwork =
                  minimal ? std::max<hwyblas_int>(1, side == 'L' ? n : m) : -1;
              if (lwork == -1) {
                T query = T(0);
                CALL_RC(T, ormql, unmql, &side, &trans, &m, &n, &k, P(a.data()),
                        &lda, P(tau.data()), P(c.data()), &ldc, P(&query),
                        &lwork, &info);
                ASSERT_EQ(info, 0);
                lwork = QueryResult(query);
              }
              std::vector<T> work(lwork, NaN<T>());
              CALL_RC(T, ormql, unmql, &side, &trans, &m, &n, &k, P(a.data()),
                      &lda, P(tau.data()), P(c.data()), &ldc, P(work.data()),
                      &lwork, &info);
              ASSERT_EQ(info, 0);
              EXPECT_LT(ProductRatio(ToDense(c, m, n, ldc), expected, cd, nq),
                        kLapackThreshold);
            }
            std::vector<T> c = c0;
            std::vector<T> work(std::max<hwyblas_int>(1, other), NaN<T>());
            CALL_RC(T, orm2l, unm2l, &side, &trans, &m, &n, &k, P(a.data()),
                    &lda, P(tau.data()), P(c.data()), &ldc, P(work.data()),
                    &info);
            ASSERT_EQ(info, 0);
            EXPECT_LT(ProductRatio(ToDense(c, m, n, ldc), expected, cd, nq),
                      kLapackThreshold);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The generalized problem

// The triangular factor F of B = F**H * F (uplo 'U') or F * F**H ('L') from
// xPOTRF in b.
template <typename T>
Dense<T> Factor(char uplo, const std::vector<T>& b, hwyblas_int n,
                hwyblas_int ldb) {
  Dense<T> f(n, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      if (InTriangle(uplo, i, j)) f(i, j) = At(b, i, j, ldb);
    }
  }
  return f;
}

// xSYGST / xHEGST (blocked with several blocks and unblocked) and xSYGS2 /
// xHEGS2: the reduced matrix C satisfies F**H * C * F = A or F * C * F**H =
// A (ITYPE = 1) or is F * A * F**H or F**H * A * F (ITYPE = 2, 3), and B
// and the other triangle of A are unchanged.
TYPED_TEST(SyevdTest, Hegst) {
  using T = TypeParam;
  Random<T> rng(239);
  for (const hwyblas_int nb : {0, 1, 3}) {
    const ScopedBlockSize block_size(nb);
    for (const bool unblocked : {false, true}) {
      for (const hwyblas_int itype : {1, 2, 3}) {
        for (const char uplo : {'U', 'L'}) {
          for (const hwyblas_int n : {0, 1, 2, 5, 13}) {
            SCOPED_TRACE(::testing::Message()
                         << "nb=" << nb << " unblocked=" << unblocked
                         << " itype=" << itype << " uplo=" << uplo
                         << " n=" << n);
            const hwyblas_int lda = n + 1;
            const std::vector<T> a0 = RandomHermitian(uplo, n, lda, rng);
            std::vector<T> b = PositiveDefinite(uplo, n, lda, rng);
            const Dense<T> ad = HermitianOf(uplo, a0, n, lda);
            const Dense<T> bd = HermitianOf(uplo, b, n, lda);
            hwyblas_int info = -99;
            HWYBLAS_CALL(T, potrf, &uplo, &n, P(b.data()), &lda, &info);
            ASSERT_EQ(info, 0);
            const std::vector<T> b0 = b;
            std::vector<T> a = a0;
            if (unblocked) {
              CALL_RC(T, sygs2, hegs2, &itype, &uplo, &n, P(a.data()), &lda,
                      P(b.data()), &lda, &info);
            } else {
              CALL_RC(T, sygst, hegst, &itype, &uplo, &n, P(a.data()), &lda,
                      P(b.data()), &lda, &info);
            }
            ASSERT_EQ(info, 0);
            for (size_t i = 0; i < b.size(); ++i) {
              ASSERT_EQ(b[i], b0[i]) << i;
            }
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = 0; i < n; ++i) {
                if (!InTriangle(uplo, i, j)) {
                  ASSERT_TRUE(std::isnan(std::real(At(a, i, j, lda))));
                }
              }
            }
            if (n == 0) continue;
            const Dense<T> c = HermitianOf(uplo, a, n, lda);
            const Dense<T> f = Factor(uplo, b, n, lda);
            const bool upper = uplo == 'U';
            const Dense<T> lhs = itype != 1 ? c
                                 : upper    ? Mul(Mul(Adjoint(f), c), f)
                                            : Mul(Mul(f, c), Adjoint(f));
            const Dense<T> rhs = itype == 1 ? ad
                                 : upper    ? Mul(Mul(f, ad), Adjoint(f))
                                            : Mul(Mul(Adjoint(f), ad), f);
            const double ratio =
                Norm1(Sub(lhs, rhs)) /
                (n * LapackEps<T>() * Norm1(ad) * std::max(1.0, Norm1(bd)));
            EXPECT_LT(ratio, kLapackThreshold);
          }
        }
      }
    }
  }
}

// xSYGV / xHEGV (divide_and_conquer false) or xSYGVD / xHEGVD with the
// minimal workspaces (minimal) or the optimal ones of a workspace query.
template <typename T>
hwyblas_int Hegv(bool divide_and_conquer, hwyblas_int itype, char jobz,
                 char uplo, hwyblas_int n, std::vector<T>& a, hwyblas_int lda,
                 std::vector<T>& b, hwyblas_int ldb, std::vector<RealT<T>>& w,
                 bool minimal) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const bool v = jobz == 'V';
  hwyblas_int lwork, lrwork, liwork, info = -99;
  if (!divide_and_conquer) {
    lwork = std::max<hwyblas_int>(1, kComplex ? 2 * n - 1 : 3 * n - 1);
    lrwork = std::max<hwyblas_int>(1, 3 * n - 2);
    liwork = 1;
  } else {
    lwork = n <= 1     ? 1
            : kComplex ? (v ? 2 * n + n * n : n + 1)
                       : (v ? 1 + 6 * n + 2 * n * n : 2 * n + 1);
    lrwork = n <= 1 ? 1 : (v ? 1 + 5 * n + 2 * n * n : n);
    liwork = n <= 1 || !v ? 1 : 3 + 5 * n;
  }
  if (!minimal) {
    std::vector<T> wq(1);
    std::vector<R> rq(std::max<hwyblas_int>(1, lrwork));
    std::vector<hwyblas_int> iq(1);
    const hwyblas_int m1 = -1;
    if (!divide_and_conquer) {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hegv, &itype, &jobz, &uplo, &n, P(a.data()),
                             &lda, P(b.data()), &ldb, w.data(), P(wq.data()),
                             &m1, rq.data(), &info);
      } else {
        HWYBLAS_CALL_REAL(T, sygv, &itype, &jobz, &uplo, &n, a.data(), &lda,
                          b.data(), &ldb, w.data(), wq.data(), &m1, &info);
      }
    } else {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hegvd, &itype, &jobz, &uplo, &n, P(a.data()),
                             &lda, P(b.data()), &ldb, w.data(), P(wq.data()),
                             &m1, rq.data(), &m1, iq.data(), &m1, &info);
        lrwork = QueryResult(rq[0]);
      } else {
        HWYBLAS_CALL_REAL(T, sygvd, &itype, &jobz, &uplo, &n, a.data(), &lda,
                          b.data(), &ldb, w.data(), wq.data(), &m1, iq.data(),
                          &m1, &info);
      }
      liwork = iq[0];
    }
    if (info != 0) return info;
    lwork = QueryResult(wq[0]);
  }
  std::vector<T> work(lwork, NaN<T>());
  std::vector<R> rwork(std::max<hwyblas_int>(1, lrwork), NAN);
  std::vector<hwyblas_int> iwork(liwork, -77);
  if (!divide_and_conquer) {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, hegv, &itype, &jobz, &uplo, &n, P(a.data()), &lda,
                           P(b.data()), &ldb, w.data(), P(work.data()), &lwork,
                           rwork.data(), &info);
    } else {
      HWYBLAS_CALL_REAL(T, sygv, &itype, &jobz, &uplo, &n, a.data(), &lda,
                        b.data(), &ldb, w.data(), work.data(), &lwork, &info);
    }
  } else {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, hegvd, &itype, &jobz, &uplo, &n, P(a.data()),
                           &lda, P(b.data()), &ldb, w.data(), P(work.data()),
                           &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
                           &info);
    } else {
      HWYBLAS_CALL_REAL(T, sygvd, &itype, &jobz, &uplo, &n, a.data(), &lda,
                        b.data(), &ldb, w.data(), work.data(), &lwork,
                        iwork.data(), &liwork, &info);
    }
  }
  return info;
}

// xSYGV / xHEGV and xSYGVD / xHEGVD: for all three ITYPEs, the eigenvectors
// Z satisfy A*Z = B*Z*D (1), A*B*Z = Z*D (2) or B*A*Z = Z*D (3) and are
// normalized (Z**H*B*Z = I or Z**H*inv(B)*Z = I), the eigenvalues ascend
// and are the same without eigenvectors and for both drivers (up to
// rounding); B returns its Cholesky factor. INFO = N+i if B is not
// positive definite.
TYPED_TEST(SyevdTest, Hegv) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(257);
  for (const bool dc : {false, true}) {
    for (const hwyblas_int nb : {0, 3}) {
      const ScopedBlockSize block_size(nb);
      for (const hwyblas_int smlsiz : kSubproblemSizes) {
        const ScopedSubproblemSize subproblems(smlsiz);
        for (const hwyblas_int itype : {1, 2, 3}) {
          for (const char uplo : {'U', 'L'}) {
            for (const hwyblas_int n : {1, 2, 6, 17}) {
              for (const bool minimal : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "dc=" << dc << " nb=" << nb << " smlsiz="
                             << smlsiz << " itype=" << itype << " uplo=" << uplo
                             << " n=" << n << " minimal=" << minimal);
                const hwyblas_int lda = n + 1;
                const std::vector<T> a0 = RandomHermitian(uplo, n, lda, rng);
                const std::vector<T> b0 = PositiveDefinite(uplo, n, lda, rng);
                const Dense<T> ad = HermitianOf(uplo, a0, n, lda);
                const Dense<T> bd = HermitianOf(uplo, b0, n, lda);
                std::vector<T> z = a0, b = b0;
                std::vector<R> w(n);
                ASSERT_EQ(
                    Hegv(dc, itype, 'V', uplo, n, z, lda, b, lda, w, minimal),
                    0);
                ExpectAscending(w);
                // B returns its Cholesky factor.
                std::vector<T> f = b0;
                hwyblas_int info = -99;
                HWYBLAS_CALL(T, potrf, &uplo, &n, P(f.data()), &lda, &info);
                for (hwyblas_int j = 0; j < n; ++j) {
                  for (hwyblas_int i = 0; i < n; ++i) {
                    if (InTriangle(uplo, i, j)) {
                      ASSERT_EQ(At(b, i, j, lda), At(f, i, j, lda));
                    }
                  }
                }

                const Dense<T> zd = ToDense(z, n, n, lda);
                Dense<T> zw = zd;
                double wmax = 0;
                for (hwyblas_int j = 0; j < n; ++j) {
                  wmax = std::max(wmax, std::abs(static_cast<double>(w[j])));
                  for (hwyblas_int i = 0; i < n; ++i) {
                    zw(i, j) *= static_cast<double>(w[j]);
                  }
                }
                const double anorm = Norm1(ad), bnorm = Norm1(bd),
                             znorm = Norm1(zd);
                const Dense<T> resid =
                    itype == 1   ? Sub(Mul(ad, zd), Mul(bd, zw))
                    : itype == 2 ? Sub(Mul(ad, Mul(bd, zd)), zw)
                                 : Sub(Mul(bd, Mul(ad, zd)), zw);
                const double scale = itype == 1
                                         ? (anorm + bnorm * wmax) * znorm
                                         : anorm * bnorm * znorm + wmax * znorm;
                EXPECT_LT(Norm1(resid) / (n * LapackEps<T>() * scale),
                          kLapackThreshold);
                // Normalization: Z**H * B * Z = I (itype 1, 2) or
                // Z**H * inv(B) * Z = I (itype 3: Z = F**H*Y or L*Y with
                // orthonormal Y, so that Y = inv(F**H) Z is orthonormal).
                if (itype != 3) {
                  const Dense<T> g = Mul(Adjoint(zd), Mul(bd, zd));
                  EXPECT_LT(Norm1(Sub(g, Identity<T>(n))) /
                                (n * LapackEps<T>() * bnorm * znorm * znorm),
                            kLapackThreshold);
                }

                std::vector<T> an = a0, bn = b0;
                std::vector<R> wn(n);
                ASSERT_EQ(Hegv(dc, itype, 'N', uplo, n, an, lda, bn, lda, wn,
                               minimal),
                          0);
                std::vector<T> ao = a0, bo = b0;
                std::vector<R> wo(n);
                ASSERT_EQ(Hegv(!dc, itype, 'N', uplo, n, ao, lda, bo, lda, wo,
                               minimal),
                          0);
                for (hwyblas_int i = 0; i < n; ++i) {
                  EXPECT_NEAR(wn[i], w[i], 30.0 * n * Eps<T>() * wmax) << i;
                  EXPECT_NEAR(wo[i], w[i], 30.0 * n * Eps<T>() * wmax) << i;
                }
              }
            }
          }
        }
      }
    }
  }

  // B not positive definite: INFO = N + the order of the failing minor.
  for (const bool dc : {false, true}) {
    const hwyblas_int n = 5, lda = 5;
    std::vector<T> a = RandomHermitian('L', n, lda, rng);
    std::vector<T> b = PositiveDefinite('L', n, lda, rng);
    At(b, 2, 2, lda) = T(-1);
    std::vector<R> w(n);
    EXPECT_EQ(Hegv(dc, 1, 'V', 'L', n, a, lda, b, lda, w, false), n + 3);
  }
}

// ---------------------------------------------------------------------------
// Workspace queries and errors

TYPED_TEST(SyevdTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const ScopedBlockSize block_size(8);
  const hwyblas_int n = 40;  // > SMLSIZ = 25: divide and conquer
  std::vector<R> d(n, 1), e(n, 0);
  std::vector<T> z(n * n);
  // xSTEDC: the minimal sizes (lgn = 6 for n = 40).
  const hwyblas_int lgn = 6;
  for (const char compz : {'I', 'V', 'N'}) {
    SCOPED_TRACE(compz);
    T wq = T(0);
    R rq = 0;
    hwyblas_int iq = 0, info = -99;
    const hwyblas_int m1 = -1;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, stedc, &compz, &n, d.data(), e.data(),
                           P(z.data()), &n, P(&wq), &m1, &rq, &m1, &iq, &m1,
                           &info);
    } else {
      HWYBLAS_CALL_REAL(T, stedc, &compz, &n, d.data(), e.data(), z.data(), &n,
                        &wq, &m1, &iq, &m1, &info);
    }
    ASSERT_EQ(info, 0);
    hwyblas_int lwork = 1, lrwork = 1, liwork = 1;
    if (compz == 'I') {
      lwork = kComplex ? 1 : 1 + 4 * n + n * n;
      lrwork = 1 + 4 * n + 2 * n * n;
      liwork = 3 + 5 * n;
    } else if (compz == 'V') {
      const hwyblas_int big = 1 + 3 * n + 2 * n * lgn + 4 * n * n;
      lwork = kComplex ? n * n : big;
      lrwork = big;
      liwork = 6 + 6 * n + 5 * n * lgn;
    }
    EXPECT_EQ(QueryResult(wq), lwork);
    if (kComplex) EXPECT_EQ(QueryResult(rq), lrwork);
    EXPECT_EQ(iq, liwork);
  }
  // xSYEVD / xHEEVD: the optimal LWORK with NB = 8.
  for (const char jobz : {'V', 'N'}) {
    std::vector<T> a(n * n);
    std::vector<R> w(n);
    T wq = T(0);
    R rq = 0;
    hwyblas_int iq = 0, info = -99;
    const hwyblas_int m1 = -1;
    const char uplo = 'L';
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevd, &jobz, &uplo, &n, P(a.data()), &n,
                           w.data(), P(&wq), &m1, &rq, &m1, &iq, &m1, &info);
      EXPECT_EQ(QueryResult(wq),
                std::max(jobz == 'V' ? 2 * n + n * n : n + 1, n + n * 8));
      EXPECT_EQ(QueryResult(rq), jobz == 'V' ? 1 + 5 * n + 2 * n * n : n);
    } else {
      HWYBLAS_CALL_REAL(T, syevd, &jobz, &uplo, &n, a.data(), &n, w.data(), &wq,
                        &m1, &iq, &m1, &info);
      EXPECT_EQ(QueryResult(wq),
                std::max(jobz == 'V' ? 1 + 6 * n + 2 * n * n : 2 * n + 1,
                         2 * n + n * 8));
    }
    EXPECT_EQ(info, 0);
    EXPECT_EQ(iq, jobz == 'V' ? 3 + 5 * n : 1);
  }
  // xSYGV / xHEGV, xORMTR / xUNMTR, xORMQL / xUNMQL.
  {
    std::vector<T> a(n * n), b(n * n), c(n * n), tau(n);
    std::vector<R> w(n), rwork(3 * n);
    T wq = T(0);
    hwyblas_int info = -99;
    const hwyblas_int m1 = -1, one = 1;
    const char jobz = 'V', uplo = 'U';
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, hegv, &one, &jobz, &uplo, &n, P(a.data()), &n,
                           P(b.data()), &n, w.data(), P(&wq), &m1, rwork.data(),
                           &info);
      EXPECT_EQ(QueryResult(wq), (8 + 1) * n);
    } else {
      HWYBLAS_CALL_REAL(T, sygv, &one, &jobz, &uplo, &n, a.data(), &n, b.data(),
                        &n, w.data(), &wq, &m1, &info);
      EXPECT_EQ(QueryResult(wq), (8 + 2) * n);
    }
    EXPECT_EQ(info, 0);
    const char side = 'L', trans = 'N';
    const hwyblas_int nc = 3;
    CALL_RC(T, ormtr, unmtr, &side, &uplo, &trans, &n, &nc, P(a.data()), &n,
            P(tau.data()), P(c.data()), &n, P(&wq), &m1, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(wq), nc * 8);
    CALL_RC(T, ormql, unmql, &side, &trans, &n, &nc, &n, P(a.data()), &n,
            P(tau.data()), P(c.data()), &n, P(&wq), &m1, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(wq), nc * 8 + 65 * 64);
  }

  // Invalid arguments.
  ErrorCapture errors;
  const hwyblas_int m = 4;
  std::vector<T> a(16), b(16), c(16), tau(4), work(128);
  std::vector<R> w(4), rwork(128);
  std::vector<hwyblas_int> iwork(32);
  const auto stedc = [&](char compz, hwyblas_int nn, hwyblas_int ldz,
                         hwyblas_int lwork, hwyblas_int lrwork,
                         hwyblas_int liwork) {
    hwyblas_int info = -99;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, stedc, &compz, &nn, w.data(), rwork.data(),
                           P(a.data()), &ldz, P(work.data()), &lwork,
                           rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    } else {
      HWYBLAS_CALL_REAL(T, stedc, &compz, &nn, w.data(), rwork.data(), a.data(),
                        &ldz, work.data(), &lwork, iwork.data(), &liwork,
                        &info);
    }
    return info;
  };
  EXPECT_EQ(stedc('X', m, m, 99, 99, 99), -1);
  EXPECT_EQ(stedc('I', -1, m, 99, 99, 99), -2);
  EXPECT_EQ(stedc('I', m, m - 1, 99, 99, 99), -6);
  EXPECT_EQ(stedc('N', m, 0, 99, 99, 99), -6);
  // n = 4 <= SMLSIZ: LWORK (real) or LRWORK (complex) >= 2*(n-1).
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(stedc('I', m, m, 1, 2 * (m - 1) - 1, 1), -10);
    EXPECT_EQ(stedc('I', 41, 41, 1, 9999, 1), -12);
  } else {
    EXPECT_EQ(stedc('I', m, m, 2 * (m - 1) - 1, 1, 1), -8);
    EXPECT_EQ(stedc('I', 41, 41, 9999, 1, 1), -10);
  }
  const std::string stedc_name = Name<T>("STEDC");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{
                stedc_name + ":1", stedc_name + ":2", stedc_name + ":6",
                stedc_name + ":6", stedc_name + (kComplex ? ":10" : ":8"),
                stedc_name + (kComplex ? ":12" : ":10")}));

  const auto heevd = [&](char jobz, char uplo, hwyblas_int nn, hwyblas_int lda,
                         hwyblas_int lwork, hwyblas_int lrwork,
                         hwyblas_int liwork) {
    hwyblas_int info = -99;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevd, &jobz, &uplo, &nn, P(a.data()), &lda,
                           w.data(), P(work.data()), &lwork, rwork.data(),
                           &lrwork, iwork.data(), &liwork, &info);
    } else {
      HWYBLAS_CALL_REAL(T, syevd, &jobz, &uplo, &nn, a.data(), &lda, w.data(),
                        work.data(), &lwork, iwork.data(), &liwork, &info);
    }
    return info;
  };
  EXPECT_EQ(heevd('X', 'L', m, m, 99, 99, 99), -1);
  EXPECT_EQ(heevd('V', 'X', m, m, 99, 99, 99), -2);
  EXPECT_EQ(heevd('V', 'L', -1, m, 99, 99, 99), -3);
  EXPECT_EQ(heevd('V', 'L', m, m - 1, 99, 99, 99), -5);
  EXPECT_EQ(heevd('N', 'L', m, m, kComplex ? m : 2 * m, 99, 99), -8);
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(heevd('N', 'L', m, m, 99, m - 1, 99), -10);
    EXPECT_EQ(heevd('V', 'L', m, m, 99, 99, 3 + 5 * m - 1), -12);
  } else {
    EXPECT_EQ(heevd('V', 'L', m, m, 99, 0, 3 + 5 * m - 1), -10);
  }
  const std::string ev = Name<T>(kComplex ? "HEEVD" : "SYEVD");
  std::vector<std::string> expected = {ev + ":1", ev + ":2", ev + ":3",
                                       ev + ":5", ev + ":8"};
  if (kComplex) expected.push_back(ev + ":10");
  expected.push_back(ev + (kComplex ? ":12" : ":10"));
  EXPECT_EQ(errors.Take(), expected);

  const auto hegst = [&](hwyblas_int itype, char uplo, hwyblas_int nn,
                         hwyblas_int lda, hwyblas_int ldb, bool unblocked) {
    hwyblas_int info = -99;
    if (unblocked) {
      CALL_RC(T, sygs2, hegs2, &itype, &uplo, &nn, P(a.data()), &lda,
              P(b.data()), &ldb, &info);
    } else {
      CALL_RC(T, sygst, hegst, &itype, &uplo, &nn, P(a.data()), &lda,
              P(b.data()), &ldb, &info);
    }
    return info;
  };
  for (const bool unblocked : {false, true}) {
    EXPECT_EQ(hegst(0, 'U', m, m, m, unblocked), -1);
    EXPECT_EQ(hegst(4, 'U', m, m, m, unblocked), -1);
    EXPECT_EQ(hegst(1, 'X', m, m, m, unblocked), -2);
    EXPECT_EQ(hegst(1, 'U', -1, m, m, unblocked), -3);
    EXPECT_EQ(hegst(1, 'U', m, m - 1, m, unblocked), -5);
    EXPECT_EQ(hegst(1, 'U', m, m, m - 1, unblocked), -7);
    const std::string gst = Name<T>(unblocked ? (kComplex ? "HEGS2" : "SYGS2")
                                              : (kComplex ? "HEGST" : "SYGST"));
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{gst + ":1", gst + ":1", gst + ":2",
                                        gst + ":3", gst + ":5", gst + ":7"}));
  }

  for (const bool dc : {false, true}) {
    const auto hegv = [&](hwyblas_int itype, char jobz, char uplo,
                          hwyblas_int nn, hwyblas_int lda, hwyblas_int ldb,
                          hwyblas_int lwork, hwyblas_int lrwork,
                          hwyblas_int liwork) {
      hwyblas_int info = -99;
      if (!dc) {
        if constexpr (IsComplex<T>::value) {
          HWYBLAS_CALL_COMPLEX(T, hegv, &itype, &jobz, &uplo, &nn, P(a.data()),
                               &lda, P(b.data()), &ldb, w.data(),
                               P(work.data()), &lwork, rwork.data(), &info);
        } else {
          HWYBLAS_CALL_REAL(T, sygv, &itype, &jobz, &uplo, &nn, a.data(), &lda,
                            b.data(), &ldb, w.data(), work.data(), &lwork,
                            &info);
        }
      } else {
        if constexpr (IsComplex<T>::value) {
          HWYBLAS_CALL_COMPLEX(T, hegvd, &itype, &jobz, &uplo, &nn, P(a.data()),
                               &lda, P(b.data()), &ldb, w.data(),
                               P(work.data()), &lwork, rwork.data(), &lrwork,
                               iwork.data(), &liwork, &info);
        } else {
          HWYBLAS_CALL_REAL(T, sygvd, &itype, &jobz, &uplo, &nn, a.data(), &lda,
                            b.data(), &ldb, w.data(), work.data(), &lwork,
                            iwork.data(), &liwork, &info);
        }
      }
      return info;
    };
    EXPECT_EQ(hegv(0, 'V', 'U', m, m, m, 99, 99, 99), -1);
    EXPECT_EQ(hegv(1, 'X', 'U', m, m, m, 99, 99, 99), -2);
    EXPECT_EQ(hegv(1, 'V', 'X', m, m, m, 99, 99, 99), -3);
    EXPECT_EQ(hegv(1, 'V', 'U', -1, m, m, 99, 99, 99), -4);
    EXPECT_EQ(hegv(1, 'V', 'U', m, m - 1, m, 99, 99, 99), -6);
    EXPECT_EQ(hegv(1, 'V', 'U', m, m, m - 1, 99, 99, 99), -8);
    const std::string gv = Name<T>(dc ? (kComplex ? "HEGVD" : "SYGVD")
                                      : (kComplex ? "HEGV" : "SYGV"));
    std::vector<std::string> gv_expected = {gv + ":1", gv + ":2", gv + ":3",
                                            gv + ":4", gv + ":6", gv + ":8"};
    if (!dc) {
      EXPECT_EQ(
          hegv(1, 'V', 'U', m, m, m, kComplex ? 2 * m - 2 : 3 * m - 2, 99, 99),
          -11);
      gv_expected.push_back(gv + ":11");
    } else {
      EXPECT_EQ(hegv(1, 'N', 'U', m, m, m, kComplex ? m : 2 * m, 99, 99), -11);
      gv_expected.push_back(gv + ":11");
      if constexpr (IsComplex<T>::value) {
        EXPECT_EQ(hegv(1, 'N', 'U', m, m, m, 99, m - 1, 99), -13);
        EXPECT_EQ(hegv(1, 'V', 'U', m, m, m, 99, 99, 3 + 5 * m - 1), -15);
        gv_expected.push_back(gv + ":13");
        gv_expected.push_back(gv + ":15");
      } else {
        EXPECT_EQ(hegv(1, 'V', 'U', m, m, m, 99, 0, 3 + 5 * m - 1), -13);
        gv_expected.push_back(gv + ":13");
      }
    }
    EXPECT_EQ(errors.Take(), gv_expected);
  }

  const char ctrans = kComplex ? 'C' : 'T';
  const auto unmtr = [&](char side, char uplo, char trans, hwyblas_int mm,
                         hwyblas_int nn, hwyblas_int lda, hwyblas_int ldc,
                         hwyblas_int lwork) {
    hwyblas_int info = -99;
    CALL_RC(T, ormtr, unmtr, &side, &uplo, &trans, &mm, &nn, P(a.data()), &lda,
            P(tau.data()), P(c.data()), &ldc, P(work.data()), &lwork, &info);
    return info;
  };
  EXPECT_EQ(unmtr('X', 'U', 'N', m, m, m, m, 99), -1);
  EXPECT_EQ(unmtr('L', 'X', 'N', m, m, m, m, 99), -2);
  EXPECT_EQ(unmtr('L', 'U', kComplex ? 'T' : 'C', m, m, m, m, 99), -3);
  EXPECT_EQ(unmtr('L', 'U', 'N', -1, m, m, m, 99), -4);
  EXPECT_EQ(unmtr('L', 'U', 'N', m, -1, m, m, 99), -5);
  EXPECT_EQ(unmtr('R', 'U', ctrans, m, m, m - 1, m, 99), -7);
  EXPECT_EQ(unmtr('L', 'U', 'N', m, m, m, m - 1, 99), -10);
  EXPECT_EQ(unmtr('L', 'U', 'N', m, m, m, m, m - 1), -12);
  const std::string mtr = Name<T>(kComplex ? "UNMTR" : "ORMTR");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{mtr + ":1", mtr + ":2", mtr + ":3",
                                      mtr + ":4", mtr + ":5", mtr + ":7",
                                      mtr + ":10", mtr + ":12"}));
  for (const bool unblocked : {false, true}) {
    const auto unmql = [&](char side, char trans, hwyblas_int mm,
                           hwyblas_int nn, hwyblas_int k, hwyblas_int lda,
                           hwyblas_int ldc, hwyblas_int lwork) {
      hwyblas_int info = -99;
      if (unblocked) {
        CALL_RC(T, orm2l, unm2l, &side, &trans, &mm, &nn, &k, P(a.data()), &lda,
                P(tau.data()), P(c.data()), &ldc, P(work.data()), &info);
      } else {
        CALL_RC(T, ormql, unmql, &side, &trans, &mm, &nn, &k, P(a.data()), &lda,
                P(tau.data()), P(c.data()), &ldc, P(work.data()), &lwork,
                &info);
      }
      return info;
    };
    EXPECT_EQ(unmql('X', 'N', m, m, m, m, m, 99), -1);
    EXPECT_EQ(unmql('L', kComplex ? 'T' : 'C', m, m, m, m, m, 99), -2);
    EXPECT_EQ(unmql('L', 'N', -1, m, m, m, m, 99), -3);
    EXPECT_EQ(unmql('L', 'N', m, -1, m, m, m, 99), -4);
    EXPECT_EQ(unmql('L', 'N', m, 2, m + 1, m, m, 99), -5);
    EXPECT_EQ(unmql('R', ctrans, 2, m, m, m - 1, m, 99), -7);
    EXPECT_EQ(unmql('L', 'N', m, m, m, m, m - 1, 99), -10);
    std::vector<std::string> ql_expected;
    const std::string ql = Name<T>(unblocked ? (kComplex ? "UNM2L" : "ORM2L")
                                             : (kComplex ? "UNMQL" : "ORMQL"));
    for (const char* arg : {":1", ":2", ":3", ":4", ":5", ":7", ":10"}) {
      ql_expected.push_back(ql + arg);
    }
    if (!unblocked) {
      EXPECT_EQ(unmql('L', 'N', m, m, m, m, m, m - 1), -12);
      ql_expected.push_back(ql + ":12");
    }
    EXPECT_EQ(errors.Take(), ql_expected);
  }

  if constexpr (!kComplex) {
    const auto stevd = [&](char jobz, hwyblas_int nn, hwyblas_int ldz,
                           hwyblas_int lwork, hwyblas_int liwork) {
      hwyblas_int info = -99;
      HWYBLAS_CALL_REAL(T, stevd, &jobz, &nn, w.data(), rwork.data(), a.data(),
                        &ldz, work.data(), &lwork, iwork.data(), &liwork,
                        &info);
      return info;
    };
    EXPECT_EQ(stevd('X', m, m, 99, 99), -1);
    EXPECT_EQ(stevd('V', -1, m, 99, 99), -2);
    EXPECT_EQ(stevd('V', m, m - 1, 99, 99), -6);
    EXPECT_EQ(stevd('V', m, m, 4 * m + m * m, 99), -8);
    EXPECT_EQ(stevd('V', m, m, 99, 2 + 5 * m), -10);
    const std::string st = Name<T>("STEVD");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{st + ":1", st + ":2", st + ":6",
                                        st + ":8", st + ":10"}));
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

// Row-major results are the column-major ones of the transposed matrices,
// bitwise, for all the functions (with their workspace queries).
TYPED_TEST(SyevdTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(263);
  const lapack_int n = 30, ld = 32;  // n > SMLSIZ = 25
  const auto expect_same = [&](const std::vector<T>& row,
                               const std::vector<T>& col, lapack_int rows,
                               lapack_int cols) {
    for (lapack_int i = 0; i < rows; ++i) {
      for (lapack_int j = 0; j < cols; ++j) {
        ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << i << ", " << j;
      }
    }
  };
  for (const char uplo : {'U', 'L'}) {
    SCOPED_TRACE(uplo);
    // ld x ld arrays (row-major copies have ld rows): the matrices with 3
    // elsewhere (also outside the triangle uplo).
    std::vector<T> a(ld * ld, T(3)), b(ld * ld, T(3));
    {
      const std::vector<T> h = RandomHermitian(uplo, n, ld, rng);
      const std::vector<T> p = PositiveDefinite(uplo, n, ld, rng);
      for (size_t i = 0; i < h.size(); ++i) {
        if (!std::isnan(std::real(h[i]))) a[i] = h[i];
        b[i] = p[i];
      }
      for (lapack_int i = 0; i < n; ++i) {
        At(a, i, i, ld) = T(std::real(At(a, i, i, ld)));
      }
    }
    for (const char jobz : {'V', 'N'}) {
      std::vector<T> col = a, row = Transposed(a, ld);
      std::vector<R> wc(n), wr(n);
      const lapack_int ic =
          LAPACKE_CALL_RC(T, syevd, heevd, LAPACK_COL_MAJOR, jobz, uplo, n,
                          col.data(), ld, wc.data());
      const lapack_int ir =
          LAPACKE_CALL_RC(T, syevd, heevd, LAPACK_ROW_MAJOR, jobz, uplo, n,
                          row.data(), ld, wr.data());
      ASSERT_EQ(ic, 0);
      ASSERT_EQ(ir, 0);
      EXPECT_EQ(wr, wc);
      if (jobz == 'V') expect_same(row, col, n, n);

      for (const lapack_int itype : {1, 2, 3}) {
        for (const bool dc : {false, true}) {
          std::vector<T> ac = a, ar = Transposed(a, ld);
          std::vector<T> bc = b, br = Transposed(b, ld);
          std::vector<R> gc(n), gr(n);
          lapack_int jc, jr;
          if (dc) {
            jc = LAPACKE_CALL_RC(T, sygvd, hegvd, LAPACK_COL_MAJOR, itype, jobz,
                                 uplo, n, ac.data(), ld, bc.data(), ld,
                                 gc.data());
            jr = LAPACKE_CALL_RC(T, sygvd, hegvd, LAPACK_ROW_MAJOR, itype, jobz,
                                 uplo, n, ar.data(), ld, br.data(), ld,
                                 gr.data());
          } else {
            jc = LAPACKE_CALL_RC(T, sygv, hegv, LAPACK_COL_MAJOR, itype, jobz,
                                 uplo, n, ac.data(), ld, bc.data(), ld,
                                 gc.data());
            jr = LAPACKE_CALL_RC(T, sygv, hegv, LAPACK_ROW_MAJOR, itype, jobz,
                                 uplo, n, ar.data(), ld, br.data(), ld,
                                 gr.data());
          }
          ASSERT_EQ(jc, 0);
          ASSERT_EQ(jr, 0);
          EXPECT_EQ(gr, gc);
          expect_same(br, bc, n, n);
          if (jobz == 'V') expect_same(ar, ac, n, n);
        }
      }
    }
    // xSYGST / xHEGST with the Cholesky factor of B.
    std::vector<T> f = b;
    lapack_int info =
        LAPACKE_CALL(T, potrf, LAPACK_COL_MAJOR, uplo, n, f.data(), ld);
    ASSERT_EQ(info, 0);
    for (const lapack_int itype : {1, 2, 3}) {
      std::vector<T> col = a, row = Transposed(a, ld);
      const std::vector<T> frow = Transposed(f, ld);
      ASSERT_EQ(LAPACKE_CALL_RC(T, sygst, hegst, LAPACK_COL_MAJOR, itype, uplo,
                                n, col.data(), ld, f.data(), ld),
                0);
      ASSERT_EQ(LAPACKE_CALL_RC(T, sygst, hegst, LAPACK_ROW_MAJOR, itype, uplo,
                                n, row.data(), ld, frow.data(), ld),
                0);
      for (lapack_int i = 0; i < n; ++i) {
        for (lapack_int j = 0; j < n; ++j) {
          if (InTriangle(uplo, i, j)) {
            ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << i << ", " << j;
          }
        }
      }
    }
    // xORMTR / xUNMTR and xORMQL / xUNMQL with the reflectors of xSYTRD /
    // xHETRD.
    std::vector<T> red = a;
    std::vector<R> d(n), e(n);
    std::vector<T> tau(n);
    ASSERT_EQ(LAPACKE_CALL_RC(T, sytrd, hetrd, LAPACK_COL_MAJOR, uplo, n,
                              red.data(), ld, d.data(), e.data(), tau.data()),
              0);
    const std::vector<T> redrow = Transposed(red, ld);
    const char ctrans = IsComplex<T>::value ? 'C' : 'T';
    for (const char side : {'L', 'R'}) {
      for (const char trans : {'N', ctrans}) {
        const lapack_int m = side == 'L' ? n : 7, nc = side == 'L' ? 7 : n;
        const std::vector<T> c = rng.Vector(ld * ld);
        std::vector<T> col = c, row = Transposed(c, ld);
        ASSERT_EQ(LAPACKE_CALL_RC(T, ormtr, unmtr, LAPACK_COL_MAJOR, side, uplo,
                                  trans, m, nc, red.data(), ld, tau.data(),
                                  col.data(), ld),
                  0);
        ASSERT_EQ(LAPACKE_CALL_RC(T, ormtr, unmtr, LAPACK_ROW_MAJOR, side, uplo,
                                  trans, m, nc, redrow.data(), ld, tau.data(),
                                  row.data(), ld),
                  0);
        expect_same(row, col, m, nc);
        if (uplo == 'U') {
          // The reflectors of uplo = 'U' are those of a QL factorization
          // in columns 2..n.
          const lapack_int k = n - 1, mq = side == 'L' ? k : 7,
                           nq = side == 'L' ? 7 : k;
          std::vector<T> qc = c, qr = Transposed(c, ld);
          ASSERT_EQ(LAPACKE_CALL_RC(T, ormql, unmql, LAPACK_COL_MAJOR, side,
                                    trans, mq, nq, k, &At(red, 0, 1, ld), ld,
                                    tau.data(), qc.data(), ld),
                    0);
          ASSERT_EQ(LAPACKE_CALL_RC(T, ormql, unmql, LAPACK_ROW_MAJOR, side,
                                    trans, mq, nq, k, &redrow[1], ld,
                                    tau.data(), qr.data(), ld),
                    0);
          expect_same(qr, qc, mq, nq);
        }
      }
    }
  }
  // xSTEDC ('I', 'V', 'N') and xSTEVD.
  Random<R> rrng(269);
  const std::vector<R> d0 = rrng.Vector(n), e0 = rrng.Vector(n - 1);
  for (const char compz : {'I', 'V', 'N'}) {
    SCOPED_TRACE(compz);
    std::vector<R> dc = d0, ec = e0, dr = d0, er = e0;
    std::vector<T> zc(ld * ld, T(0));
    for (lapack_int i = 0; i < n; ++i) At(zc, i, i, ld) = T(1);
    if (compz == 'V') zc = rng.Vector(ld * ld);
    std::vector<T> zr = Transposed(zc, ld);
    ASSERT_EQ(LAPACKE_CALL(T, stedc, LAPACK_COL_MAJOR, compz, n, dc.data(),
                           ec.data(), zc.data(), ld),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, stedc, LAPACK_ROW_MAJOR, compz, n, dr.data(),
                           er.data(), zr.data(), ld),
              0);
    EXPECT_EQ(dr, dc);
    if (compz != 'N') expect_same(zr, zc, n, n);
  }
  if constexpr (!IsComplex<T>::value) {
    for (const char jobz : {'V', 'N'}) {
      std::vector<R> dc = d0, ec = e0, dr = d0, er = e0;
      std::vector<R> zc(ld * ld, NAN), zr(ld * ld, NAN);
      ASSERT_EQ(LAPACKE_CALL_REAL(T, stevd, LAPACK_COL_MAJOR, jobz, n,
                                  dc.data(), ec.data(), zc.data(), ld),
                0);
      ASSERT_EQ(LAPACKE_CALL_REAL(T, stevd, LAPACK_ROW_MAJOR, jobz, n,
                                  dr.data(), er.data(), zr.data(), ld),
                0);
      EXPECT_EQ(dr, dc);
      if (jobz == 'V') expect_same(zr, zc, n, n);
    }
  }
}

// Invalid layouts and leading dimensions, and the NaN checks.
TYPED_TEST(SyevdTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const lapack_int n = 4;
  std::vector<T> a(16, T(0)), b(16, T(0)), c(16, T(0)), tau(4, T(0));
  for (lapack_int i = 0; i < n; ++i) {
    At(a, i, i, n) = T(1);
    At(b, i, i, n) = T(2);
  }
  std::vector<R> w(4), d(4, 1), e(3, 0);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(
      LAPACKE_CALL_RC(T, syevd, heevd, 0, 'V', 'U', n, a.data(), n, w.data()),
      -1);
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevd, heevd, LAPACK_ROW_MAJOR, 'V', 'U', n,
                            a.data(), n - 1, w.data()),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, stedc, LAPACK_ROW_MAJOR, 'I', n, d.data(), e.data(),
                         a.data(), n - 1),
            -7);
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygv, hegv, LAPACK_ROW_MAJOR, 1, 'V', 'U', n,
                            a.data(), n, b.data(), n - 1, w.data()),
            -9);
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygvd, hegvd, LAPACK_ROW_MAJOR, 1, 'V', 'U', n,
                            a.data(), n - 1, b.data(), n, w.data()),
            -7);
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygst, hegst, LAPACK_ROW_MAJOR, 1, 'U', n,
                            a.data(), n, b.data(), n - 1),
            -8);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormtr, unmtr, LAPACK_ROW_MAJOR, 'L', 'U', 'N', n,
                            n, a.data(), n - 1, tau.data(), c.data(), n),
            -8);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormql, unmql, LAPACK_ROW_MAJOR, 'L', 'N', n, n,
                            n, a.data(), n, tau.data(), c.data(), n - 1),
            -11);
  const std::string ev = kComplex ? "heevd" : "syevd";
  const std::string gv = kComplex ? "hegv" : "sygv";
  const std::string gvd = kComplex ? "hegvd" : "sygvd";
  const std::string gst = kComplex ? "hegst" : "sygst";
  const std::string mtr = kComplex ? "unmtr" : "ormtr";
  const std::string mql = kComplex ? "unmql" : "ormql";
  EXPECT_EQ(
      ::testing::internal::GetCapturedStdout(),
      "Wrong parameter 1 in " + LapackeName<T>(ev.c_str()) +
          "\nWrong parameter 6 in " + LapackeName<T>((ev + "_work").c_str()) +
          "\nWrong parameter 7 in " + LapackeName<T>("stedc_work") +
          "\nWrong parameter 9 in " + LapackeName<T>((gv + "_work").c_str()) +
          "\nWrong parameter 7 in " + LapackeName<T>((gvd + "_work").c_str()) +
          "\nWrong parameter 8 in " + LapackeName<T>((gst + "_work").c_str()) +
          "\nWrong parameter 8 in " + LapackeName<T>((mtr + "_work").c_str()) +
          "\nWrong parameter 11 in " + LapackeName<T>((mql + "_work").c_str()) +
          "\n");

  // NaN checks.
  a[1] = NaN<T>();  // (1, 0) column-major: lower triangle
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevd, heevd, LAPACK_COL_MAJOR, 'N', 'L', n,
                            a.data(), n, w.data()),
            -5);
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygv, hegv, LAPACK_COL_MAJOR, 1, 'N', 'L', n,
                            a.data(), n, b.data(), n, w.data()),
            -6);
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygst, hegst, LAPACK_COL_MAJOR, 1, 'L', n,
                            a.data(), n, b.data(), n),
            -5);
  // ?ormtr checks the triangle uplo of A, ?unmtr all of A.
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormtr, unmtr, LAPACK_COL_MAJOR, 'L', 'U', 'N', n,
                            n, a.data(), n, tau.data(), c.data(), n),
            kComplex ? -7 : 0);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormql, unmql, LAPACK_COL_MAJOR, 'L', 'N', n, n,
                            1, a.data(), n, tau.data(), c.data(), n),
            -7);
  a[1] = T(0);
  b[1] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygvd, hegvd, LAPACK_COL_MAJOR, 1, 'N', 'L', n,
                            a.data(), n, b.data(), n, w.data()),
            -8);
  EXPECT_EQ(LAPACKE_CALL_RC(T, sygvd, hegvd, LAPACK_COL_MAJOR, 1, 'N', 'U', n,
                            a.data(), n, b.data(), n, w.data()),
            0);
  tau[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormql, unmql, LAPACK_COL_MAJOR, 'L', 'N', n, n,
                            3, a.data(), n, tau.data(), c.data(), n),
            -9);
  d[2] = NAN;
  EXPECT_EQ(LAPACKE_CALL(T, stedc, LAPACK_COL_MAJOR, 'N', n, d.data(), e.data(),
                         a.data(), n),
            -4);
  d[2] = 1;
  e[1] = NAN;
  EXPECT_EQ(LAPACKE_CALL(T, stedc, LAPACK_COL_MAJOR, 'N', n, d.data(), e.data(),
                         a.data(), n),
            -5);
  e[1] = 0;
  std::vector<T> z(16, NaN<T>());
  EXPECT_EQ(LAPACKE_CALL(T, stedc, LAPACK_COL_MAJOR, 'V', n, d.data(), e.data(),
                         z.data(), n),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, stedc, LAPACK_COL_MAJOR, 'I', n, d.data(), e.data(),
                         z.data(), n),
            0);
}

}  // namespace
}  // namespace hwyblas_test
