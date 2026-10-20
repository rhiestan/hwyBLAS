// Tests of the symmetric (Hermitian) positive definite linear systems
// routines: xPOTRS, xPOSV, xLAUUM, xLAUU2, xPOTRI, xPOCON, xPOEQU, xLAQSY,
// xLAQHE, xPORFS, xPOSVX, the pivoted Cholesky factorization xPSTRF and
// xPSTF2, and their LAPACKE functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "lapack/linear_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class PosvTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PosvTest, ScalarTypes);

template <typename T>
hwyblas_int Potrf(char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, potrf, &uplo, &n, P(a.data()), &lda, &info);
  return info;
}

template <typename T>
hwyblas_int Potrs(char uplo, hwyblas_int n, hwyblas_int nrhs,
                  const std::vector<T>& a, hwyblas_int lda, std::vector<T>& b,
                  hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, potrs, &uplo, &n, &nrhs, P(a.data()), &lda, P(b.data()), &ldb,
               &info);
  return info;
}

template <typename T>
hwyblas_int Posv(char uplo, hwyblas_int n, hwyblas_int nrhs, std::vector<T>& a,
                 hwyblas_int lda, std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, posv, &uplo, &n, &nrhs, P(a.data()), &lda, P(b.data()), &ldb,
               &info);
  return info;
}

// xLAUUM, or xLAUU2 if unblocked.
template <typename T>
hwyblas_int Lauum(bool unblocked, char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda) {
  hwyblas_int info = -99;
  if (unblocked) {
    HWYBLAS_CALL(T, lauu2, &uplo, &n, P(a.data()), &lda, &info);
  } else {
    HWYBLAS_CALL(T, lauum, &uplo, &n, P(a.data()), &lda, &info);
  }
  return info;
}

template <typename T>
hwyblas_int Potri(char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, potri, &uplo, &n, P(a.data()), &lda, &info);
  return info;
}

// xPOCON with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Pocon(char uplo, hwyblas_int n, const std::vector<T>& a,
                  hwyblas_int lda, RealT<T> anorm, RealT<T>* rcond) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    HWYBLAS_CALL_COMPLEX(T, pocon, &uplo, &n, P(a.data()), &lda, &anorm, rcond,
                         P(work.data()), rwork.data(), &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, pocon, &uplo, &n, a.data(), &lda, &anorm, rcond,
                      work.data(), iwork.data(), &info);
  }
  return info;
}

template <typename T>
hwyblas_int Poequ(hwyblas_int n, const std::vector<T>& a, hwyblas_int lda,
                  std::vector<RealT<T>>& s, RealT<T>* scond, RealT<T>* amax) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, poequ, &n, P(a.data()), &lda, s.data(), scond, amax, &info);
  return info;
}

// xLAQSY, or xLAQHE (complex T) if herm.
template <typename T>
char Laq(bool herm, char uplo, hwyblas_int n, std::vector<T>& a,
         hwyblas_int lda, const std::vector<RealT<T>>& s, RealT<T> scond,
         RealT<T> amax) {
  char equed = '?';
  if (herm) {
    HWYBLAS_CALL_COMPLEX(T, laqhe, &uplo, &n, P(a.data()), &lda, s.data(),
                         &scond, &amax, &equed);
  } else {
    HWYBLAS_CALL(T, laqsy, &uplo, &n, P(a.data()), &lda, s.data(), &scond,
                 &amax, &equed);
  }
  return equed;
}

// xPORFS with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Porfs(char uplo, hwyblas_int n, hwyblas_int nrhs,
                  const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& af, hwyblas_int ldaf,
                  const std::vector<T>& b, hwyblas_int ldb, std::vector<T>& x,
                  hwyblas_int ldx, std::vector<RealT<T>>& ferr,
                  std::vector<RealT<T>>& berr) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    HWYBLAS_CALL_COMPLEX(T, porfs, &uplo, &n, &nrhs, P(a.data()), &lda,
                         P(af.data()), &ldaf, P(b.data()), &ldb, P(x.data()),
                         &ldx, ferr.data(), berr.data(), P(work.data()),
                         rwork.data(), &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, porfs, &uplo, &n, &nrhs, a.data(), &lda, af.data(),
                      &ldaf, b.data(), &ldb, x.data(), &ldx, ferr.data(),
                      berr.data(), work.data(), iwork.data(), &info);
  }
  return info;
}

// The arguments and results of xPOSVX.
template <typename T>
struct PosvxCall {
  using R = RealT<T>;
  char fact = 'N', uplo = 'U', equed = 'N';
  hwyblas_int n = 0, nrhs = 0, lda = 1, ldaf = 1, ldb = 1, ldx = 1;
  std::vector<T> a, af, b, x;
  std::vector<R> s, ferr, berr;
  R rcond = -1;

  // Calls xPOSVX with workspaces of the minimal sizes; returns INFO.
  hwyblas_int Run() {
    const size_t n1 = std::max<hwyblas_int>(1, n);
    af.resize(static_cast<size_t>(ldaf) * n1);
    x.resize(static_cast<size_t>(ldx) * std::max<hwyblas_int>(1, nrhs));
    s.resize(n1);
    ferr.assign(std::max<hwyblas_int>(1, nrhs), -1);
    berr.assign(std::max<hwyblas_int>(1, nrhs), -1);
    hwyblas_int info = -99;
    if constexpr (IsComplex<T>::value) {
      std::vector<T> work(2 * n1);
      std::vector<R> rwork(n1);
      HWYBLAS_CALL_COMPLEX(T, posvx, &fact, &uplo, &n, &nrhs, P(a.data()), &lda,
                           P(af.data()), &ldaf, &equed, s.data(), P(b.data()),
                           &ldb, P(x.data()), &ldx, &rcond, ferr.data(),
                           berr.data(), P(work.data()), rwork.data(), &info);
    } else {
      std::vector<T> work(3 * n1);
      std::vector<hwyblas_int> iwork(n1);
      HWYBLAS_CALL_REAL(T, posvx, &fact, &uplo, &n, &nrhs, a.data(), &lda,
                        af.data(), &ldaf, &equed, s.data(), b.data(), &ldb,
                        x.data(), &ldx, &rcond, ferr.data(), berr.data(),
                        work.data(), iwork.data(), &info);
    }
    return info;
  }
};

// xPSTRF, or xPSTF2 if unblocked.
template <typename T>
hwyblas_int Pstrf(bool unblocked, char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda, std::vector<hwyblas_int>& piv,
                  hwyblas_int* rank, RealT<T> tol) {
  hwyblas_int info = -99;
  std::vector<RealT<T>> work(2 * std::max<hwyblas_int>(1, n));
  piv.resize(std::max<hwyblas_int>(1, n));
  if (unblocked) {
    HWYBLAS_CALL(T, pstf2, &uplo, &n, P(a.data()), &lda, piv.data(), rank, &tol,
                 work.data(), &info);
  } else {
    HWYBLAS_CALL(T, pstrf, &uplo, &n, P(a.data()), &lda, piv.data(), rank, &tol,
                 work.data(), &info);
  }
  return info;
}

// The test ratio of LAPACK's xPOT02: ||A * X - B||_1 / (n ||A||_1 ||X||_1
// eps), for the Hermitian A of the triangle uplo of a.
template <typename T>
double SolveRatio(char uplo, const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& x, hwyblas_int ldx,
                  const std::vector<T>& b, hwyblas_int ldb, hwyblas_int n,
                  hwyblas_int nrhs) {
  const Dense<T> da = HermitianOf(uplo, a, n, lda);
  const Dense<T> dx = ToDense(x, n, nrhs, ldx);
  const double resid = Norm1(Sub(Mul(da, dx), ToDense(b, n, nrhs, ldb)));
  return resid / std::max<hwyblas_int>(1, n) / Norm1(da) / Norm1(dx) /
         LapackEps<T>();
}

// Checks that the elements of a outside the triangle uplo (and below row n)
// are those of a0.
template <typename T>
void ExpectOutsideUnchanged(char uplo, const std::vector<T>& a,
                            const std::vector<T>& a0, hwyblas_int n,
                            hwyblas_int lda) {
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < lda; ++i) {
      if (i < n && InTriangle(uplo, i, j)) continue;
      ASSERT_EQ(At(a, i, j, lda), At(a0, i, j, lda)) << i << ", " << j;
    }
  }
}

// A positive definite matrix with a badly scaled diagonal (D * A * D with
// D = diag(2**k), k up to 8) for equilibration.
template <typename T>
std::vector<T> BadlyScaled(char uplo, Random<T>& rng, hwyblas_int n,
                           hwyblas_int lda) {
  std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      if (!InTriangle(uplo, i, j)) continue;
      At(a, i, j, lda) *=
          static_cast<RealT<T>>(std::ldexp(1.0, 2 * (i % 5) + 2 * (j % 5)));
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// Solution

// xPOTRS after xPOTRF, and xPOSV, which gives the same factor and solution.
TYPED_TEST(PosvTest, Solve) {
  using T = TypeParam;
  Random<T> rng(7);
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int n : {0, 1, 2, 5, 16, 33, 70}) {
      for (const hwyblas_int nrhs : {1, 3}) {
        SCOPED_TRACE(::testing::Message()
                     << uplo << " n=" << n << " nrhs=" << nrhs);
        const hwyblas_int lda = n + 2, ldb = n + 1;
        const std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
        const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
        std::vector<T> f = a, x = b;
        ASSERT_EQ(Potrf(uplo, n, f, lda), 0);
        ASSERT_EQ(Potrs(uplo, n, nrhs, f, lda, x, ldb), 0);
        if (n > 0) {
          EXPECT_LT(SolveRatio(uplo, a, lda, x, ldb, b, ldb, n, nrhs),
                    kLapackThreshold);
        }
        std::vector<T> a2 = a, x2 = b;
        ASSERT_EQ(Posv(uplo, n, nrhs, a2, lda, x2, ldb), 0);
        EXPECT_EQ(a2, f);
        EXPECT_EQ(x2, x);
      }
    }
  }
}

// xPOSV stops at a leading minor that is not positive: INFO = k, and B is
// not changed.
TYPED_TEST(PosvTest, SolveNotPositiveDefinite) {
  using T = TypeParam;
  Random<T> rng(9);
  const hwyblas_int n = 7;
  for (const char uplo : {'U', 'L'}) {
    std::vector<T> a = PositiveDefinite<T>(uplo, n, n, rng);
    At(a, 4, 4, n) = T(-1);
    const std::vector<T> b = rng.Vector(n);
    std::vector<T> x = b;
    EXPECT_EQ(Posv(uplo, n, 1, a, n, x, n), 5);
    EXPECT_EQ(x, b);
  }
}

// ---------------------------------------------------------------------------
// Inverse

// The triangle uplo of U * U**H (L**H * L) of xLAUUM (blocked, several block
// sizes) and xLAUU2 for a triangular matrix with a real diagonal; nothing
// outside the triangle changes.
TYPED_TEST(PosvTest, TriangularProduct) {
  using T = TypeParam;
  Random<T> rng(11);
  for (const bool unblocked : {false, true}) {
    for (const hwyblas_int nb : BlockSizes()) {
      if (unblocked && nb != 0) continue;
      const ScopedBlockSize block_size(nb);
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {0, 1, 2, 7, 16, 33, 70}) {
          SCOPED_TRACE(::testing::Message()
                       << (unblocked ? "lauu2 " : "lauum ") << uplo
                       << " n=" << n << " nb=" << nb);
          const hwyblas_int lda = n + 3;
          std::vector<T> a(
              static_cast<size_t>(lda) * std::max<hwyblas_int>(n, 1),
              Sentinel<T>());
          Dense<T> f(n, n);
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              if (!InTriangle(uplo, i, j)) continue;
              T v = rng.Next();
              if (i == j) v = T(std::abs(std::real(v)) + 1);
              At(a, i, j, lda) = v;
              f(i, j) = static_cast<RefT<T>>(v);
            }
          }
          const std::vector<T> a0 = a;
          ASSERT_EQ(Lauum(unblocked, uplo, n, a, lda), 0);
          if (n == 0) {
            EXPECT_EQ(a, a0);
            continue;
          }
          const Dense<T> prod =
              uplo == 'U' ? Mul(f, Adjoint(f)) : Mul(Adjoint(f), f);
          EXPECT_LT(ResidualRatio(prod, HermitianOf(uplo, a, n, lda),
                                  static_cast<double>(n)),
                    kLapackThreshold);
          for (hwyblas_int i = 0; i < n; ++i) {
            EXPECT_EQ(std::imag(At(a, i, i, lda)), 0);
          }
          ExpectOutsideUnchanged(uplo, a, a0, n, lda);
        }
      }
    }
  }
}

// The test ratio of LAPACK's xPOT03: ||I - A * inv(A)||_1 /
// (n ||A||_1 ||inv(A)||_1 eps) for xPOTRI after xPOTRF, with block sizes
// that select the blocked and unblocked code of xLAUUM.
TYPED_TEST(PosvTest, Inverse) {
  using T = TypeParam;
  Random<T> rng(13);
  for (const hwyblas_int nb : BlockSizes()) {
    const ScopedBlockSize block_size(nb);
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {0, 1, 2, 9, 33, 70}) {
        SCOPED_TRACE(::testing::Message()
                     << uplo << " n=" << n << " nb=" << nb);
        const hwyblas_int lda = n + 2;
        const std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
        std::vector<T> inv = a;
        ASSERT_EQ(Potrf(uplo, n, inv, lda), 0);
        ASSERT_EQ(Potri(uplo, n, inv, lda), 0);
        ExpectOutsideUnchanged(uplo, inv, a, n, lda);
        if (n == 0) continue;
        const Dense<T> da = HermitianOf(uplo, a, n, lda);
        const Dense<T> di = HermitianOf(uplo, inv, n, lda);
        const double resid = Norm1(Sub(Identity<T>(n), Mul(da, di)));
        EXPECT_LT(resid / n / Norm1(da) / Norm1(di) / LapackEps<T>(),
                  kLapackThreshold);
      }
    }
  }
}

// A singular factor (U(k, k) = 0): xPOTRI returns INFO = k from xTRTRI.
TYPED_TEST(PosvTest, InverseSingular) {
  using T = TypeParam;
  const hwyblas_int n = 5;
  for (const char uplo : {'U', 'L'}) {
    std::vector<T> a(n * n, T(0));
    for (hwyblas_int i = 0; i < n; ++i) At(a, i, i, n) = T(2);
    At(a, 2, 2, n) = T(0);
    EXPECT_EQ(Potri(uplo, n, a, n), 3);
  }
}

// ---------------------------------------------------------------------------
// Condition number

// The estimate of xPOCON against the exact reciprocal condition number
// (1-norm): the ratio of LAPACK's xGET06 (the larger of the two over the
// smaller).
TYPED_TEST(PosvTest, ConditionNumber) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(17);
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int n : {1, 2, 7, 25, 50}) {
      SCOPED_TRACE(::testing::Message() << uplo << " n=" << n);
      const hwyblas_int lda = n + 1;
      std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
      // Make it worse conditioned: D * A * D with D = diag(2**-(i/4)).
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < n; ++i) {
          if (InTriangle(uplo, i, j)) {
            At(a, i, j, lda) *=
                static_cast<R>(std::ldexp(1.0, -(i / 4) - (j / 4)));
          }
        }
      }
      const Dense<T> da = HermitianOf(uplo, a, n, lda);
      std::vector<T> f = a;
      ASSERT_EQ(Potrf(uplo, n, f, lda), 0);
      R rcond = -1;
      ASSERT_EQ(Pocon(uplo, n, f, lda, static_cast<R>(Norm1(da)), &rcond), 0);
      const double rcondc = 1 / (Norm1(da) * Norm1(Inverse(da)));
      ASSERT_GT(rcond, 0);
      EXPECT_LT(std::max(rcond / rcondc, rcondc / rcond), kLapackThreshold)
          << rcond << " " << rcondc;
    }
  }
}

// n = 0 gives RCOND = 1, ANORM = 0 gives RCOND = 0.
TYPED_TEST(PosvTest, ConditionNumberSpecialCases) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(4, T(1));
  R rcond = -1;
  EXPECT_EQ(Pocon('U', 0, a, 1, R(1), &rcond), 0);
  EXPECT_EQ(rcond, 1);
  EXPECT_EQ(Pocon('L', 2, a, 2, R(0), &rcond), 0);
  EXPECT_EQ(rcond, 0);
}

// ---------------------------------------------------------------------------
// Equilibration

// xPOEQU computes exactly the scalings of its definition (recomputed here
// with the same operations), and xLAQSY / xLAQHE apply them as far as
// worthwhile.
TYPED_TEST(PosvTest, Equilibration) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(19);
  const hwyblas_int n = 9, lda = 11;
  for (const char uplo : {'U', 'L'}) {
    for (const bool scaled : {false, true}) {
      SCOPED_TRACE(::testing::Message() << uplo << " scaled=" << scaled);
      const std::vector<T> a = scaled ? BadlyScaled<T>(uplo, rng, n, lda)
                                      : PositiveDefinite<T>(uplo, n, lda, rng);
      std::vector<R> s(n);
      R scond = -1, amax = -1;
      ASSERT_EQ(Poequ(n, a, lda, s, &scond, &amax), 0);
      R smin = std::real(a[0]), smax = smin;
      for (hwyblas_int i = 0; i < n; ++i) {
        const R d = std::real(At(a, i, i, lda));
        smin = std::min(smin, d);
        smax = std::max(smax, d);
        EXPECT_EQ(s[i], R(1) / std::sqrt(d));
      }
      EXPECT_EQ(amax, smax);
      EXPECT_EQ(scond, std::sqrt(smin) / std::sqrt(smax));
      EXPECT_EQ(scond < R(0.1), scaled);
      for (const bool herm : {false, true}) {
        if (herm && !IsComplex<T>::value) continue;
        std::vector<T> e = a;
        const char equed = Laq(herm, uplo, n, e, lda, s, scond, amax);
        EXPECT_EQ(equed, scaled ? 'Y' : 'N');
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) {
            if (!InTriangle(uplo, i, j)) continue;
            T want = At(a, i, j, lda);
            if (scaled) {
              want = (i == j && herm) ? T(s[j] * s[j] * std::real(want))
                                      : s[j] * s[i] * want;
            }
            EXPECT_EQ(At(e, i, j, lda), want) << i << ", " << j;
          }
        }
        ExpectOutsideUnchanged(uplo, e, a, n, lda);
      }
    }
  }
  // A diagonal element <= 0 gives INFO = its index; n = 0 SCOND = 1 and
  // AMAX = 0.
  std::vector<T> d(16, T(0));
  for (hwyblas_int i = 0; i < 4; ++i) At(d, i, i, 4) = T(1);
  At(d, 2, 2, 4) = T(-2);
  std::vector<R> s(4);
  R scond = -1, amax = -1;
  EXPECT_EQ(Poequ(4, d, 4, s, &scond, &amax), 3);
  EXPECT_EQ(Poequ(0, d, 1, s, &scond, &amax), 0);
  EXPECT_EQ(scond, 1);
  EXPECT_EQ(amax, 0);
  // AMAX close to underflow or overflow equilibrates even with SCOND = 1.
  for (const R big_or_small :
       {std::numeric_limits<R>::min(), std::numeric_limits<R>::max()}) {
    std::vector<T> e = {T(1)};
    const std::vector<R> one = {R(0.5)};
    EXPECT_EQ(Laq(false, 'U', 1, e, 1, one, R(1), big_or_small), 'Y');
    EXPECT_EQ(e[0], T(R(0.25)));
  }
  std::vector<T> e = {T(1)};
  EXPECT_EQ(Laq(false, 'U', 0, e, 1, s, R(0), R(1)), 'N');
}

// ---------------------------------------------------------------------------
// Iterative refinement

// Solutions perturbed by a known relative error: xPORFS refines them (the
// error decreases to about the rounding error) and bounds the remaining
// error; the backward error is that of the refined solution.
TYPED_TEST(PosvTest, Refinement) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(23);
  const double delta = std::is_same_v<R, float> ? 1e-2 : 1e-6;
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int n : {1, 6, 40}) {
      for (const double bscale : {1.0, NearUnderflow<T>()}) {
        SCOPED_TRACE(::testing::Message()
                     << uplo << " n=" << n << " " << bscale);
        const hwyblas_int nrhs = 2, lda = n + 1, ldb = n + 2, ldx = n + 3;
        const std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
        std::vector<T> af = a;
        ASSERT_EQ(Potrf(uplo, n, af, lda), 0);
        std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
        for (T& e : b) e *= static_cast<R>(bscale);
        const Dense<T> da = HermitianOf(uplo, a, n, lda);
        const Dense<T> xt = Mul(Inverse(da), ToDense(b, n, nrhs, ldb));
        std::vector<T> x(static_cast<size_t>(ldx) * nrhs);
        for (hwyblas_int j = 0; j < nrhs; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) {
            At(x, i, j, ldx) =
                static_cast<T>(xt(i, j) * (1 + delta * std::real(rng.Next())));
          }
        }
        std::vector<R> ferr(nrhs, -1), berr(nrhs, -1);
        ASSERT_EQ(
            Porfs(uplo, n, nrhs, a, lda, af, lda, b, ldb, x, ldx, ferr, berr),
            0);
        for (hwyblas_int j = 0; j < nrhs; ++j) {
          double err = 0, xnorm = 0;
          for (hwyblas_int i = 0; i < n; ++i) {
            err = std::max(
                err,
                Cabs1Ref(static_cast<RefT<T>>(At(x, i, j, ldx)) - xt(i, j)));
            xnorm = std::max(xnorm, Cabs1Ref(At(x, i, j, ldx)));
          }
          EXPECT_LT(err / xnorm, 100 * n * LapackEps<T>());
          EXPECT_GE(ferr[j], 0.3 * err / xnorm);
          // The backward error of the refined solution, with the diagonal
          // as the routine sees it (abs(Re(A(k, k))), the same for real T).
          std::vector<double> resid(n), denom(n);
          for (hwyblas_int i = 0; i < n; ++i) {
            RefT<T> r = -static_cast<RefT<T>>(At(b, i, j, ldb));
            denom[i] = Cabs1Ref(At(b, i, j, ldb));
            for (hwyblas_int k = 0; k < n; ++k) {
              r += da(i, k) * static_cast<RefT<T>>(At(x, k, j, ldx));
              denom[i] += Cabs1Ref(da(i, k)) * Cabs1Ref(At(x, k, j, ldx));
            }
            resid[i] = Cabs1Ref(r);
          }
          const double berr_c = BackwardError<T>(resid, denom);
          if (bscale == 1) {
            EXPECT_LT(berr[j], 4 * (n + 1) * LapackEps<T>());
            EXPECT_LT(ferr[j], 1e3 * n * LapackEps<T>());
          } else {
            // Near underflow, SAFE1 dominates the backward error.
            EXPECT_NEAR(berr[j], berr_c, 0.01 * berr_c);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Expert driver

// xPOSVX with FACT = 'N' and 'E' (on well and badly scaled matrices), and
// 'F' with the factorization and equilibration of an 'E' call, which gives
// the same solution. The solution solves the original system; RCOND is that
// of xPOCON for the (equilibrated) matrix.
TYPED_TEST(PosvTest, ExpertDriver) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(29);
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int n : {0, 1, 5, 30}) {
      for (const bool scaled : {false, true}) {
        for (const char fact : {'N', 'E'}) {
          SCOPED_TRACE(::testing::Message()
                       << uplo << " n=" << n << " " << scaled << " " << fact);
          PosvxCall<T> c;
          c.fact = fact;
          c.uplo = uplo;
          c.n = n;
          c.nrhs = 2;
          c.lda = n + 1;
          c.ldaf = n + 2;
          c.ldb = n + 3;
          c.ldx = n + 4;
          const std::vector<T> a0 =
              scaled ? BadlyScaled<T>(uplo, rng, n, c.lda)
                     : PositiveDefinite<T>(uplo, n, c.lda, rng);
          const std::vector<T> b0 =
              rng.Vector(static_cast<size_t>(c.ldb) * c.nrhs);
          c.a = a0;
          c.b = b0;
          const hwyblas_int info = c.Run();
          ASSERT_EQ(info, 0);
          const bool equilibrated = fact == 'E' && scaled && n > 1;
          EXPECT_EQ(c.equed, equilibrated ? 'Y' : 'N');
          if (!equilibrated) {
            EXPECT_EQ(c.a, a0);
            EXPECT_EQ(c.b, b0);
          }
          if (n == 0) {
            EXPECT_EQ(c.rcond, 1);
            continue;
          }
          EXPECT_LT(
              SolveRatio(uplo, a0, c.lda, c.x, c.ldx, b0, c.ldb, n, c.nrhs),
              kLapackThreshold);
          // RCOND of the matrix that was factored (in c.a).
          const Dense<T> da = HermitianOf(uplo, c.a, n, c.lda);
          R rcond = -1;
          ASSERT_EQ(
              Pocon(uplo, n, c.af, c.ldaf, static_cast<R>(Norm1(da)), &rcond),
              0);
          EXPECT_NEAR(c.rcond, rcond, 1e-3 * rcond);
          // FERR bounds the error of the solution of the original system
          // (for EQUED = 'Y' after the division by SCOND).
          const Dense<T> xt =
              Mul(Inverse(HermitianOf(uplo, a0, n, c.lda)),
                  ToDense(b0, n, c.nrhs, c.ldb));
          for (hwyblas_int j = 0; j < c.nrhs; ++j) {
            double err = 0, xnorm = 0;
            for (hwyblas_int i = 0; i < n; ++i) {
              err = std::max(err, Cabs1Ref(static_cast<RefT<T>>(
                                               At(c.x, i, j, c.ldx)) -
                                           xt(i, j)));
              xnorm = std::max(xnorm, Cabs1Ref(At(c.x, i, j, c.ldx)));
            }
            EXPECT_GE(c.ferr[j], 0.3 * err / xnorm);
            EXPECT_LT(c.berr[j], 4 * (n + 1) * LapackEps<T>());
          }
          // For EQUED = 'Y': the FERR of the equilibrated system (FACT =
          // 'F', EQUED = 'N', with the scaled B that xPOSVX returns, which
          // computes the same X before the scaling back), divided by SCOND.
          if (equilibrated) {
            PosvxCall<T> g = c;
            g.fact = 'F';
            g.equed = 'N';
            g.b = c.b;
            ASSERT_EQ(g.Run(), 0);
            const R smin = *std::min_element(c.s.begin(), c.s.begin() + n);
            const R smax = *std::max_element(c.s.begin(), c.s.begin() + n);
            const double scond = static_cast<double>(smin) / smax;
            for (hwyblas_int j = 0; j < c.nrhs; ++j) {
              EXPECT_NEAR(c.ferr[j], g.ferr[j] / scond,
                          1e-4 * g.ferr[j] / scond);
            }
          }
          // FACT = 'F' with the factorization (and scaling) of this call.
          PosvxCall<T> f = c;
          f.fact = 'F';
          // xPOSVX scales B (again) for EQUED = 'Y', so it takes the
          // original right hand side; A is the equilibrated matrix.
          f.a = c.a;
          f.b = b0;
          f.x.assign(f.x.size(), T(0));
          ASSERT_EQ(f.Run(), 0);
          EXPECT_EQ(f.x, c.x);
          EXPECT_EQ(f.rcond, c.rcond);
        }
      }
    }
  }
}

// A matrix that is not positive definite: INFO = k, RCOND = 0, no solution.
// A matrix singular to working precision: INFO = n + 1 with a solution.
TYPED_TEST(PosvTest, ExpertDriverSingular) {
  using T = TypeParam;
  Random<T> rng(31);
  const hwyblas_int n = 6;
  PosvxCall<T> c;
  c.n = n;
  c.nrhs = 1;
  c.lda = c.ldaf = c.ldb = c.ldx = n;
  c.a = PositiveDefinite<T>('U', n, n, rng);
  At(c.a, 3, 3, n) = T(-1);
  c.b = rng.Vector(n);
  EXPECT_EQ(c.Run(), 4);
  EXPECT_EQ(c.rcond, 0);
  EXPECT_EQ(c.ferr[0], -1);  // not computed

  // diag(1, ..., 1, eps**2).
  PosvxCall<T> d;
  d.n = n;
  d.nrhs = 1;
  d.uplo = 'L';
  d.lda = d.ldaf = d.ldb = d.ldx = n;
  d.a.assign(n * n, T(0));
  for (hwyblas_int i = 0; i < n; ++i) At(d.a, i, i, n) = T(1);
  At(d.a, n - 1, n - 1, n) = T(static_cast<RealT<T>>(Eps<T>() * Eps<T>()));
  d.b = rng.Vector(n);
  EXPECT_EQ(d.Run(), n + 1);
  EXPECT_LT(d.rcond, LapackEps<T>());
  EXPECT_GE(d.ferr[0], 0);
}

// ---------------------------------------------------------------------------
// Pivoted Cholesky factorization

// A random positive semidefinite matrix of rank r (B * B**H for an n x r
// matrix B), in the triangle uplo, with Sentinel() elsewhere.
template <typename T>
std::vector<T> Semidefinite(char uplo, hwyblas_int n, hwyblas_int r,
                            hwyblas_int lda, Random<T>& rng) {
  const std::vector<T> b = rng.Vector(static_cast<size_t>(n) * r);
  const Dense<T> full = Mul(ToDense(b, n, r, n), Adjoint(ToDense(b, n, r, n)));
  std::vector<T> a(static_cast<size_t>(lda) * std::max<hwyblas_int>(n, 1),
                   Sentinel<T>());
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      if (InTriangle(uplo, i, j)) {
        At(a, i, j, lda) = static_cast<T>(
            i == j ? RefT<T>(std::real(full(i, j))) : full(i, j));
      }
    }
  }
  return a;
}

// The test ratio of LAPACK's xPST01: ||P**T * A * P - U**H * U||_1 /
// (n ||A||_1 eps) (or with L * L**H), with the rows of U (columns of L)
// beyond the rank set to zero.
template <typename T>
double PivotedResidual(char uplo, const std::vector<T>& a,
                       const std::vector<T>& f,
                       const std::vector<hwyblas_int>& piv, hwyblas_int n,
                       hwyblas_int lda, hwyblas_int rank) {
  const Dense<T> da = HermitianOf(uplo, a, n, lda);
  Dense<T> pap(n, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) pap(i, j) = da(piv[i] - 1, piv[j] - 1);
  }
  Dense<T> u(n, n);  // U, or L**H
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i <= j; ++i) {
      if (i >= rank) continue;
      u(i, j) = uplo == 'U' ? static_cast<RefT<T>>(At(f, i, j, lda))
                            : Conj(static_cast<RefT<T>>(At(f, j, i, lda)));
    }
  }
  return ResidualRatio(pap, Mul(Adjoint(u), u), static_cast<double>(n));
}

// Full rank and rank-deficient semidefinite matrices, with the default and
// a given tolerance, for xPSTF2 and xPSTRF with several block sizes: the
// rank, a permutation in PIV and the residual.
TYPED_TEST(PosvTest, PivotedCholesky) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(37);
  for (const hwyblas_int nb : BlockSizes()) {
    const ScopedBlockSize block_size(nb);
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {1, 2, 9, 40}) {
        for (const hwyblas_int r : {n, n / 2, hwyblas_int{1}}) {
          if (r < 1) continue;
          for (const bool unblocked : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << (unblocked ? "pstf2 " : "pstrf ") << uplo
                         << " n=" << n << " r=" << r << " nb=" << nb);
            const hwyblas_int lda = n + 2;
            // Full rank: a well-conditioned matrix and the default
            // tolerance (TOL < 0: n * eps * max(A(k, k))). Rank r < n: the
            // rounding errors of the remaining diagonal can exceed the
            // default tolerance, so TOL is sqrt(eps) * max(A(k, k)).
            const std::vector<T> a =
                r == n ? PositiveDefinite<T>(uplo, n, lda, rng)
                       : Semidefinite<T>(uplo, n, r, lda, rng);
            R tol = -1;
            if (r < n) {
              R amax = 0;
              for (hwyblas_int i = 0; i < n; ++i) {
                amax = std::max(amax, std::real(At(a, i, i, lda)));
              }
              tol = std::sqrt(static_cast<R>(LapackEps<T>())) * amax;
            }
            std::vector<T> f = a;
            std::vector<hwyblas_int> piv;
            hwyblas_int rank = -1;
            const hwyblas_int info =
                Pstrf(unblocked, uplo, n, f, lda, piv, &rank, tol);
            EXPECT_EQ(info, rank < n ? 1 : 0);
            EXPECT_EQ(rank, r);
            std::vector<hwyblas_int> sorted(piv.begin(), piv.begin() + n);
            std::sort(sorted.begin(), sorted.end());
            std::vector<hwyblas_int> perm(n);
            std::iota(perm.begin(), perm.end(), 1);
            EXPECT_EQ(sorted, perm);
            EXPECT_LT(PivotedResidual(uplo, a, f, piv, n, lda, rank),
                      kLapackThreshold);
            for (hwyblas_int k = 0; k < rank; ++k) {
              EXPECT_GT(std::real(At(f, k, k, lda)), 0);
              EXPECT_EQ(std::imag(At(f, k, k, lda)), 0);
            }
            ExpectOutsideUnchanged(uplo, f, a, n, lda);
          }
        }
      }
    }
  }
  // The default tolerance is relative to the largest diagonal element:
  // diag(16, 8 * eps) has rank 1 (8 * eps < n * eps * 16).
  for (const bool unblocked : {false, true}) {
    std::vector<T> d(4, T(0));
    d[0] = T(16);
    d[3] = T(static_cast<R>(8 * LapackEps<T>()));
    std::vector<hwyblas_int> piv;
    hwyblas_int rank = -1;
    EXPECT_EQ(Pstrf(unblocked, 'L', 2, d, 2, piv, &rank, R(-1)), 1);
    EXPECT_EQ(rank, 1);
  }
  // A given tolerance stops at the first diagonal element below it:
  // diag(16, 9, 4, 1) with TOL = 2 gives rank 3.
  for (const bool unblocked : {false, true}) {
    std::vector<T> d(16, T(0));
    for (hwyblas_int i = 0; i < 4; ++i) At(d, i, i, 4) = T(R(1 + i) * (1 + i));
    std::vector<hwyblas_int> piv;
    hwyblas_int rank = -1;
    EXPECT_EQ(Pstrf(unblocked, 'U', 4, d, 4, piv, &rank, R(2)), 1);
    EXPECT_EQ(rank, 3);
    EXPECT_EQ(piv, (std::vector<hwyblas_int>{4, 3, 2, 1}));
    EXPECT_EQ(At(d, 0, 0, 4), T(4));
    EXPECT_EQ(At(d, 3, 3, 4), T(1));  // the stopping value
  }
}

// n = 0, a first pivot that is not positive, and a NaN on the diagonal: the
// real routines find the first pivot with a loop that stops at A(1, 1) =
// NaN, the complex ones with MAXLOC, which skips NaNs; both stop with
// INFO = 1 when only NaNs are left.
TYPED_TEST(PosvTest, PivotedCholeskySpecialCases) {
  using T = TypeParam;
  using R = RealT<T>;
  for (const bool unblocked : {false, true}) {
    SCOPED_TRACE(unblocked);
    std::vector<T> a(1, T(3));
    std::vector<hwyblas_int> piv;
    hwyblas_int rank = -1;
    EXPECT_EQ(Pstrf(unblocked, 'U', 0, a, 1, piv, &rank, R(-1)), 0);
    EXPECT_EQ(rank, -1);
    EXPECT_EQ(a[0], T(3));
    std::vector<T> z(9, T(0));
    At(z, 1, 1, 3) = T(-1);
    EXPECT_EQ(Pstrf(unblocked, 'L', 3, z, 3, piv, &rank, R(-1)), 1);
    EXPECT_EQ(rank, 0);
    std::vector<T> n3(9, T(0));
    At(n3, 0, 0, 3) = NaN<T>();
    At(n3, 1, 1, 3) = T(4);
    At(n3, 2, 2, 3) = T(9);
    EXPECT_EQ(Pstrf(unblocked, 'U', 3, n3, 3, piv, &rank, R(-1)), 1);
    if constexpr (IsComplex<T>::value) {
      EXPECT_EQ(rank, 2);
      EXPECT_EQ(piv, (std::vector<hwyblas_int>{3, 2, 1}));
    } else {
      EXPECT_EQ(rank, 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Errors

TYPED_TEST(PosvTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16, T(1));
  std::vector<R> s(4, 1), ferr(4), berr(4);
  R rcond, scond, amax;
  ErrorCapture errors;
  EXPECT_EQ(Potrs('X', 4, 1, a, 4, b, 4), -1);
  EXPECT_EQ(Potrs('U', -1, 1, a, 4, b, 4), -2);
  EXPECT_EQ(Potrs('U', 4, -1, a, 4, b, 4), -3);
  EXPECT_EQ(Potrs('U', 4, 1, a, 3, b, 4), -5);
  EXPECT_EQ(Potrs('U', 4, 1, a, 4, b, 3), -7);
  const std::string trs = Name<T>("POTRS");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{trs + ":1", trs + ":2", trs + ":3",
                                      trs + ":5", trs + ":7"}));
  EXPECT_EQ(Posv('X', 4, 1, a, 4, b, 4), -1);
  EXPECT_EQ(Posv('U', -1, 1, a, 4, b, 4), -2);
  EXPECT_EQ(Posv('U', 4, -1, a, 4, b, 4), -3);
  EXPECT_EQ(Posv('U', 4, 1, a, 3, b, 4), -5);
  EXPECT_EQ(Posv('U', 4, 1, a, 4, b, 3), -7);
  const std::string sv = Name<T>("POSV");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{sv + ":1", sv + ":2", sv + ":3",
                                      sv + ":5", sv + ":7"}));
  for (const bool unblocked : {false, true}) {
    EXPECT_EQ(Lauum(unblocked, 'X', 4, a, 4), -1);
    EXPECT_EQ(Lauum(unblocked, 'U', -1, a, 4), -2);
    EXPECT_EQ(Lauum(unblocked, 'U', 4, a, 3), -4);
    const std::string uum = Name<T>(unblocked ? "LAUU2" : "LAUUM");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{uum + ":1", uum + ":2", uum + ":4"}));
  }
  EXPECT_EQ(Potri('X', 4, a, 4), -1);
  EXPECT_EQ(Potri('U', -1, a, 4), -2);
  EXPECT_EQ(Potri('U', 4, a, 3), -4);
  const std::string tri = Name<T>("POTRI");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{tri + ":1", tri + ":2", tri + ":4"}));
  EXPECT_EQ(Pocon('X', 4, a, 4, R(1), &rcond), -1);
  EXPECT_EQ(Pocon('U', -1, a, 4, R(1), &rcond), -2);
  EXPECT_EQ(Pocon('U', 4, a, 3, R(1), &rcond), -4);
  EXPECT_EQ(Pocon('U', 4, a, 4, R(-1), &rcond), -5);
  const std::string con = Name<T>("POCON");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{con + ":1", con + ":2",
                                                     con + ":4", con + ":5"}));
  EXPECT_EQ(Poequ(-1, a, 4, s, &scond, &amax), -1);
  EXPECT_EQ(Poequ(4, a, 3, s, &scond, &amax), -3);
  const std::string equ = Name<T>("POEQU");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{equ + ":1", equ + ":3"}));
  EXPECT_EQ(Porfs('X', 4, 1, a, 4, a, 4, b, 4, b, 4, ferr, berr), -1);
  EXPECT_EQ(Porfs('U', -1, 1, a, 4, a, 4, b, 4, b, 4, ferr, berr), -2);
  EXPECT_EQ(Porfs('U', 4, -1, a, 4, a, 4, b, 4, b, 4, ferr, berr), -3);
  EXPECT_EQ(Porfs('U', 4, 1, a, 3, a, 4, b, 4, b, 4, ferr, berr), -5);
  EXPECT_EQ(Porfs('U', 4, 1, a, 4, a, 3, b, 4, b, 4, ferr, berr), -7);
  EXPECT_EQ(Porfs('U', 4, 1, a, 4, a, 4, b, 3, b, 4, ferr, berr), -9);
  EXPECT_EQ(Porfs('U', 4, 1, a, 4, a, 4, b, 4, b, 3, ferr, berr), -11);
  const std::string rfs = Name<T>("PORFS");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               rfs + ":1", rfs + ":2", rfs + ":3", rfs + ":5",
                               rfs + ":7", rfs + ":9", rfs + ":11"}));
  const auto posvx = [&](char fact, char uplo, hwyblas_int n, hwyblas_int nrhs,
                         hwyblas_int lda, hwyblas_int ldaf, char equed, R sv,
                         hwyblas_int ldb, hwyblas_int ldx) {
    PosvxCall<T> call;
    call.fact = fact;
    call.uplo = uplo;
    call.n = n;
    call.nrhs = nrhs;
    call.lda = lda;
    call.ldaf = ldaf;
    call.ldb = ldb;
    call.ldx = ldx;
    call.equed = equed;
    call.a.assign(16, T(1));
    call.b.assign(16, T(1));
    call.af.assign(16, T(1));
    call.s.assign(4, sv);
    return call.Run();
  };
  EXPECT_EQ(posvx('X', 'U', 4, 1, 4, 4, 'N', 1, 4, 4), -1);
  EXPECT_EQ(posvx('N', 'X', 4, 1, 4, 4, 'N', 1, 4, 4), -2);
  EXPECT_EQ(posvx('N', 'U', -1, 1, 4, 4, 'N', 1, 4, 4), -3);
  EXPECT_EQ(posvx('N', 'U', 4, -1, 4, 4, 'N', 1, 4, 4), -4);
  EXPECT_EQ(posvx('N', 'U', 4, 1, 3, 4, 'N', 1, 4, 4), -6);
  EXPECT_EQ(posvx('N', 'U', 4, 1, 4, 3, 'N', 1, 4, 4), -8);
  EXPECT_EQ(posvx('F', 'U', 4, 1, 4, 4, 'X', 1, 4, 4), -9);
  EXPECT_EQ(posvx('F', 'U', 4, 1, 4, 4, 'Y', 0, 4, 4), -10);
  EXPECT_EQ(posvx('N', 'U', 4, 1, 4, 4, 'N', 1, 3, 4), -12);
  EXPECT_EQ(posvx('N', 'U', 4, 1, 4, 4, 'N', 1, 4, 3), -14);
  const std::string svx = Name<T>("POSVX");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               svx + ":1", svx + ":2", svx + ":3", svx + ":4",
                               svx + ":6", svx + ":8", svx + ":9", svx + ":10",
                               svx + ":12", svx + ":14"}));
  for (const bool unblocked : {false, true}) {
    std::vector<hwyblas_int> piv;
    hwyblas_int rank;
    EXPECT_EQ(Pstrf(unblocked, 'X', 4, a, 4, piv, &rank, R(-1)), -1);
    EXPECT_EQ(Pstrf(unblocked, 'U', -1, a, 4, piv, &rank, R(-1)), -2);
    EXPECT_EQ(Pstrf(unblocked, 'U', 4, a, 3, piv, &rank, R(-1)), -4);
    const std::string ps = Name<T>(unblocked ? "PSTF2" : "PSTRF");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{ps + ":1", ps + ":2", ps + ":4"}));
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

// The row-major functions give the results of the column-major ones (the
// same arithmetic on transposed copies of the triangle uplo); elements
// outside the triangle are not changed.
TYPED_TEST(PosvTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(41);
  const lapack_int n = 8, nrhs = 3, lda = 10, ldb = 9;
  for (const char uplo : {'U', 'L'}) {
    SCOPED_TRACE(uplo);
    const std::vector<T> a = BadlyScaled<T>(uplo, rng, n, lda);
    const std::vector<T> ar = RowMajor(a, n, n, lda, lda);
    const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * n);
    const std::vector<T> br = RowMajor(b, n, nrhs, ldb, ldb);
    const auto same = [&](const std::vector<T>& row, const std::vector<T>& col,
                          lapack_int rows, lapack_int cols, lapack_int ldr,
                          lapack_int ldc) {
      for (lapack_int i = 0; i < rows; ++i) {
        for (lapack_int j = 0; j < cols; ++j) {
          ASSERT_EQ(row[i * ldr + j], At(col, i, j, ldc)) << i << ", " << j;
        }
      }
    };
    // xPOSV and xPOTRS.
    std::vector<T> fc = a, fr = ar, xc = b, xr = br;
    ASSERT_EQ(LAPACKE_CALL(T, posv, LAPACK_COL_MAJOR, uplo, n, nrhs, fc.data(),
                           lda, xc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, posv, LAPACK_ROW_MAJOR, uplo, n, nrhs, fr.data(),
                           lda, xr.data(), ldb),
              0);
    same(fr, fc, n, n, lda, lda);
    same(xr, xc, n, nrhs, ldb, ldb);
    std::vector<T> yc = b, yr = br;
    ASSERT_EQ(LAPACKE_CALL(T, potrs, LAPACK_COL_MAJOR, uplo, n, nrhs, fc.data(),
                           lda, yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, potrs, LAPACK_ROW_MAJOR, uplo, n, nrhs, fr.data(),
                           lda, yr.data(), ldb),
              0);
    same(yr, yc, n, nrhs, ldb, ldb);
    // xPOCON, xPORFS.
    R rc = -1, rr = -2;
    ASSERT_EQ(LAPACKE_CALL(T, pocon, LAPACK_COL_MAJOR, uplo, n, fc.data(), lda,
                           R(3), &rc),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, pocon, LAPACK_ROW_MAJOR, uplo, n, fr.data(), lda,
                           R(3), &rr),
              0);
    EXPECT_EQ(rr, rc);
    std::vector<R> f1(nrhs), e1(nrhs), f2(nrhs), e2(nrhs);
    ASSERT_EQ(LAPACKE_CALL(T, porfs, LAPACK_COL_MAJOR, uplo, n, nrhs, a.data(),
                           lda, fc.data(), lda, b.data(), ldb, yc.data(), ldb,
                           f1.data(), e1.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, porfs, LAPACK_ROW_MAJOR, uplo, n, nrhs, ar.data(),
                           lda, fr.data(), lda, br.data(), ldb, yr.data(), ldb,
                           f2.data(), e2.data()),
              0);
    same(yr, yc, n, nrhs, ldb, ldb);
    EXPECT_EQ(f2, f1);
    EXPECT_EQ(e2, e1);
    // xPOTRI, xLAUUM.
    std::vector<T> ic = fc, ir = fr;
    ASSERT_EQ(LAPACKE_CALL(T, potri, LAPACK_COL_MAJOR, uplo, n, ic.data(), lda),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, potri, LAPACK_ROW_MAJOR, uplo, n, ir.data(), lda),
              0);
    same(ir, ic, n, n, lda, lda);
    ASSERT_EQ(LAPACKE_CALL(T, lauum, LAPACK_COL_MAJOR, uplo, n, ic.data(), lda),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, lauum, LAPACK_ROW_MAJOR, uplo, n, ir.data(), lda),
              0);
    same(ir, ic, n, n, lda, lda);
    // xPOEQU.
    std::vector<R> sc(n), sr(n);
    R scc, scr, amc, amr;
    ASSERT_EQ(LAPACKE_CALL(T, poequ, LAPACK_COL_MAJOR, n, a.data(), lda,
                           sc.data(), &scc, &amc),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, poequ, LAPACK_ROW_MAJOR, n, ar.data(), lda,
                           sr.data(), &scr, &amr),
              0);
    EXPECT_EQ(sr, sc);
    EXPECT_EQ(scr, scc);
    EXPECT_EQ(amr, amc);
    // xPSTRF of a semidefinite matrix.
    const std::vector<T> sd = Semidefinite<T>(uplo, n, 5, lda, rng);
    std::vector<T> pc = sd, pr = RowMajor(sd, n, n, lda, lda);
    std::vector<lapack_int> pivc(n), pivr(n);
    lapack_int rankc = -1, rankr = -2;
    ASSERT_EQ(LAPACKE_CALL(T, pstrf, LAPACK_COL_MAJOR, uplo, n, pc.data(), lda,
                           pivc.data(), &rankc, R(-1)),
              1);
    ASSERT_EQ(LAPACKE_CALL(T, pstrf, LAPACK_ROW_MAJOR, uplo, n, pr.data(), lda,
                           pivr.data(), &rankr, R(-1)),
              1);
    EXPECT_EQ(rankc, 5);
    EXPECT_EQ(rankr, rankc);
    EXPECT_EQ(pivr, pivc);
    same(pr, pc, n, n, lda, lda);
    // xPOSVX: A (equilibrated), AF, B (scaled) and X are those of the
    // column-major call.
    for (const char fact : {'N', 'E'}) {
      SCOPED_TRACE(fact);
      std::vector<T> ac = a, arr = ar, afc(lda * n), afr(lda * n);
      std::vector<T> bcc = b, brw = br;
      std::vector<T> x1(ldb * nrhs), x2(ldb * n);
      std::vector<R> s1(n), s2(n), g1(nrhs), h1(nrhs), g2(nrhs), h2(nrhs);
      char eq1 = 'N', eq2 = 'N';
      R rc1, rc2;
      ASSERT_EQ(LAPACKE_CALL(T, posvx, LAPACK_COL_MAJOR, fact, uplo, n, nrhs,
                             ac.data(), lda, afc.data(), lda, &eq1, s1.data(),
                             bcc.data(), ldb, x1.data(), ldb, &rc1, g1.data(),
                             h1.data()),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, posvx, LAPACK_ROW_MAJOR, fact, uplo, n, nrhs,
                             arr.data(), lda, afr.data(), lda, &eq2, s2.data(),
                             brw.data(), ldb, x2.data(), ldb, &rc2, g2.data(),
                             h2.data()),
                0);
      EXPECT_EQ(eq2, eq1);
      EXPECT_EQ(eq1, fact == 'E' ? 'Y' : 'N');
      same(arr, ac, n, n, lda, lda);
      // Only the triangle of AF is set.
      for (lapack_int i = 0; i < n; ++i) {
        for (lapack_int j = 0; j < n; ++j) {
          if (InTriangle(uplo, i, j)) {
            ASSERT_EQ(afr[i * lda + j], At(afc, i, j, lda)) << i << ", " << j;
          }
        }
      }
      same(brw, bcc, n, nrhs, ldb, ldb);
      same(x2, x1, n, nrhs, ldb, ldb);
      EXPECT_EQ(s2, s1);
      EXPECT_EQ(rc2, rc1);
      EXPECT_EQ(g2, g1);
      EXPECT_EQ(h2, h1);
    }
  }
}

TYPED_TEST(PosvTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16, T(1)), x(16);
  std::vector<lapack_int> piv(4);
  std::vector<R> s(4, 1), ferr(4), berr(4);
  R rcond = 0, scond, amax;
  lapack_int rank;
  char equed = 'N';
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LAPACKE_CALL(T, posv, 0, 'U', 4, 1, a.data(), 4, b.data(), 4), -1);
  EXPECT_EQ(LAPACKE_CALL(T, posv, LAPACK_ROW_MAJOR, 'U', 4, 2, a.data(), 4,
                         b.data(), 1),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, potrs, LAPACK_ROW_MAJOR, 'U', 4, 1, a.data(), 3,
                         b.data(), 1),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, potri, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3), -5);
  EXPECT_EQ(LAPACKE_CALL(T, lauum, LAPACK_ROW_MAJOR, 'L', 4, a.data(), 3), -5);
  EXPECT_EQ(LAPACKE_CALL(T, pocon, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3, R(1),
                         &rcond),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, poequ, LAPACK_ROW_MAJOR, 4, a.data(), 3, s.data(),
                         &scond, &amax),
            -4);
  EXPECT_EQ(
      LAPACKE_CALL(T, porfs, LAPACK_ROW_MAJOR, 'U', 4, 2, a.data(), 4, a.data(),
                   4, b.data(), 4, x.data(), 1, ferr.data(), berr.data()),
      -12);
  EXPECT_EQ(LAPACKE_CALL(T, posvx, LAPACK_ROW_MAJOR, 'N', 'U', 4, 2, a.data(),
                         4, a.data(), 4, &equed, s.data(), b.data(), 1,
                         x.data(), 4, &rcond, ferr.data(), berr.data()),
            -13);
  EXPECT_EQ(LAPACKE_CALL(T, pstrf, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3,
                         piv.data(), &rank, R(-1)),
            -5);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("posv") +
                "\nWrong parameter 8 in " + LapackeName<T>("posv_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("potrs_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("potri_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("lauum_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("pocon_work") +
                "\nWrong parameter 4 in " + LapackeName<T>("poequ_work") +
                "\nWrong parameter 12 in " + LapackeName<T>("porfs_work") +
                "\nWrong parameter 13 in " + LapackeName<T>("posvx_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("pstrf_work") +
                "\n");
  // NaN checks: only the triangle uplo of A (and AF) is checked.
  a[1] = NaN<T>();  // A(2, 1): in the lower triangle
  EXPECT_EQ(LAPACKE_CALL(T, potri, LAPACK_COL_MAJOR, 'L', 4, a.data(), 4), -4);
  std::vector<T> au = a;
  for (lapack_int i = 0; i < 4; ++i) au[i * 5] = T(4);
  EXPECT_EQ(LAPACKE_CALL(T, potri, LAPACK_COL_MAJOR, 'U', 4, au.data(), 4), 0);
  b[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, posv, LAPACK_COL_MAJOR, 'U', 4, 1, au.data(), 4,
                         b.data(), 4),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, pocon, LAPACK_COL_MAJOR, 'U', 4, au.data(), 4,
                         std::numeric_limits<R>::quiet_NaN(), &rcond),
            -6);
  EXPECT_EQ(
      LAPACKE_CALL(T, pstrf, LAPACK_COL_MAJOR, 'U', 4, au.data(), 4, piv.data(),
                   &rank, std::numeric_limits<R>::quiet_NaN()),
      -8);
  EXPECT_EQ(LAPACKE_CALL(T, poequ, LAPACK_COL_MAJOR, 4, a.data(), 4, s.data(),
                         &scond, &amax),
            -3);  // all of A
  // xPOSVX checks S only for FACT = 'F' and EQUED = 'Y'.
  b[2] = T(1);
  s[1] = std::numeric_limits<R>::quiet_NaN();
  equed = 'Y';
  EXPECT_EQ(LAPACKE_CALL(T, posvx, LAPACK_COL_MAJOR, 'F', 'U', 4, 1, au.data(),
                         4, au.data(), 4, &equed, s.data(), b.data(), 4,
                         x.data(), 4, &rcond, ferr.data(), berr.data()),
            -11);
  equed = 'N';
  EXPECT_NE(LAPACKE_CALL(T, posvx, LAPACK_COL_MAJOR, 'F', 'U', 4, 1, au.data(),
                         4, au.data(), 4, &equed, s.data(), b.data(), 4,
                         x.data(), 4, &rcond, ferr.data(), berr.data()),
            -11);
}

}  // namespace
}  // namespace hwyblas_test
