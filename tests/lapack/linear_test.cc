// Tests of the general linear systems routines: xGETRS, xGESV, xGETRI,
// xGECON, xGEEQU, xLAQGE, xGERFS, xGESVX and their LAPACKE functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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
class LinearTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LinearTest, ScalarTypes);

// CABS1 in the working precision, as the routines compute it.
template <typename T>
RealT<T> Cabs1W(T x) {
  return std::abs(std::real(x)) + std::abs(std::imag(x));
}

// A random n x n matrix (leading dimension lda) with a dominant diagonal
// (well-conditioned).
template <typename T>
std::vector<T> WellConditioned(Random<T>& rng, size_t n, size_t lda) {
  std::vector<T> a = rng.Vector(lda * std::max<size_t>(n, 1));
  for (size_t i = 0; i < n; ++i) {
    At(a, i, i, lda) += T(static_cast<RealT<T>>(n) / 2);
  }
  return a;
}

template <typename T>
hwyblas_int Getrf(hwyblas_int n, std::vector<T>& a, hwyblas_int lda,
                  std::vector<hwyblas_int>& ipiv) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, getrf, &n, &n, P(a.data()), &lda, ipiv.data(), &info);
  return info;
}

template <typename T>
hwyblas_int Getrs(char trans, hwyblas_int n, hwyblas_int nrhs,
                  const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<hwyblas_int>& ipiv, std::vector<T>& b,
                  hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, getrs, &trans, &n, &nrhs, P(a.data()), &lda, ipiv.data(),
               P(b.data()), &ldb, &info);
  return info;
}

template <typename T>
hwyblas_int Gesv(hwyblas_int n, hwyblas_int nrhs, std::vector<T>& a,
                 hwyblas_int lda, std::vector<hwyblas_int>& ipiv,
                 std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gesv, &n, &nrhs, P(a.data()), &lda, ipiv.data(), P(b.data()),
               &ldb, &info);
  return info;
}

template <typename T>
hwyblas_int Getri(hwyblas_int n, std::vector<T>& a, hwyblas_int lda,
                  const std::vector<hwyblas_int>& ipiv, std::vector<T>& work,
                  hwyblas_int lwork) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, getri, &n, P(a.data()), &lda, ipiv.data(), P(work.data()),
               &lwork, &info);
  return info;
}

// xGECON with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Gecon(char norm, hwyblas_int n, const std::vector<T>& a,
                  hwyblas_int lda, RealT<T> anorm, RealT<T>* rcond) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(2 * n1);
    HWYBLAS_CALL_COMPLEX(T, gecon, &norm, &n, P(a.data()), &lda, &anorm, rcond,
                         P(work.data()), rwork.data(), &info);
  } else {
    std::vector<T> work(4 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, gecon, &norm, &n, a.data(), &lda, &anorm, rcond,
                      work.data(), iwork.data(), &info);
  }
  return info;
}

template <typename T>
RealT<T> Lange(char norm, hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
               hwyblas_int lda) {
  std::vector<RealT<T>> work(std::max<hwyblas_int>(1, m));
  if constexpr (std::is_same_v<T, float>) {
    return slange_(&norm, &m, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, double>) {
    return dlange_(&norm, &m, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return clange_(&norm, &m, &n, P(a.data()), &lda, work.data());
  } else {
    return zlange_(&norm, &m, &n, P(a.data()), &lda, work.data());
  }
}

template <typename T>
RealT<T> LantrUpperMax(hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
                       hwyblas_int lda) {
  const char norm = 'M', uplo = 'U', diag = 'N';
  RealT<T> work[1];
  if constexpr (std::is_same_v<T, float>) {
    return slantr_(&norm, &uplo, &diag, &m, &n, a.data(), &lda, work);
  } else if constexpr (std::is_same_v<T, double>) {
    return dlantr_(&norm, &uplo, &diag, &m, &n, a.data(), &lda, work);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return clantr_(&norm, &uplo, &diag, &m, &n, P(a.data()), &lda, work);
  } else {
    return zlantr_(&norm, &uplo, &diag, &m, &n, P(a.data()), &lda, work);
  }
}

template <typename T>
hwyblas_int Geequ(hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
                  hwyblas_int lda, std::vector<RealT<T>>& r,
                  std::vector<RealT<T>>& c, RealT<T>* rowcnd, RealT<T>* colcnd,
                  RealT<T>* amax) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, geequ, &m, &n, P(a.data()), &lda, r.data(), c.data(), rowcnd,
               colcnd, amax, &info);
  return info;
}

// xGERFS with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Gerfs(char trans, hwyblas_int n, hwyblas_int nrhs,
                  const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& af, hwyblas_int ldaf,
                  const std::vector<hwyblas_int>& ipiv, const std::vector<T>& b,
                  hwyblas_int ldb, std::vector<T>& x, hwyblas_int ldx,
                  std::vector<RealT<T>>& ferr, std::vector<RealT<T>>& berr) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    HWYBLAS_CALL_COMPLEX(T, gerfs, &trans, &n, &nrhs, P(a.data()), &lda,
                         P(af.data()), &ldaf, ipiv.data(), P(b.data()), &ldb,
                         P(x.data()), &ldx, ferr.data(), berr.data(),
                         P(work.data()), rwork.data(), &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, gerfs, &trans, &n, &nrhs, a.data(), &lda, af.data(),
                      &ldaf, ipiv.data(), b.data(), &ldb, x.data(), &ldx,
                      ferr.data(), berr.data(), work.data(), iwork.data(),
                      &info);
  }
  return info;
}

// The arguments and results of xGESVX.
template <typename T>
struct GesvxCall {
  using R = RealT<T>;
  char fact = 'N', trans = 'N', equed = 'N';
  hwyblas_int n = 0, nrhs = 0, lda = 1, ldaf = 1, ldb = 1, ldx = 1;
  std::vector<T> a, af, b, x;
  std::vector<hwyblas_int> ipiv;
  std::vector<R> r, c, ferr, berr;
  R rcond = -1, rpvgrw = -1;

  // Calls xGESVX with workspaces of the minimal sizes; returns INFO.
  hwyblas_int Run() {
    const size_t n1 = std::max<hwyblas_int>(1, n);
    af.resize(static_cast<size_t>(ldaf) * n1);
    x.resize(static_cast<size_t>(ldx) * std::max<hwyblas_int>(1, nrhs));
    ipiv.resize(n1);
    r.resize(n1);
    c.resize(n1);
    ferr.assign(std::max<hwyblas_int>(1, nrhs), -1);
    berr.assign(std::max<hwyblas_int>(1, nrhs), -1);
    hwyblas_int info = -99;
    if constexpr (IsComplex<T>::value) {
      std::vector<T> work(2 * n1);
      std::vector<R> rwork(2 * n1);
      HWYBLAS_CALL_COMPLEX(T, gesvx, &fact, &trans, &n, &nrhs, P(a.data()),
                           &lda, P(af.data()), &ldaf, ipiv.data(), &equed,
                           r.data(), c.data(), P(b.data()), &ldb, P(x.data()),
                           &ldx, &rcond, ferr.data(), berr.data(),
                           P(work.data()), rwork.data(), &info);
      rpvgrw = rwork[0];
    } else {
      std::vector<T> work(4 * n1);
      std::vector<hwyblas_int> iwork(n1);
      HWYBLAS_CALL_REAL(T, gesvx, &fact, &trans, &n, &nrhs, a.data(), &lda,
                        af.data(), &ldaf, ipiv.data(), &equed, r.data(),
                        c.data(), b.data(), &ldb, x.data(), &ldx, &rcond,
                        ferr.data(), berr.data(), work.data(), iwork.data(),
                        &info);
      rpvgrw = work[0];
    }
    return info;
  }
};

// The test ratio of LAPACK's xGET02: ||op(A) * X - B||_1 /
// (n ||op(A)||_1 ||X||_1 eps).
template <typename T>
double SolveRatio(char trans, const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& x, hwyblas_int ldx,
                  const std::vector<T>& b, hwyblas_int ldb, hwyblas_int n,
                  hwyblas_int nrhs) {
  const Dense<T> opa = Op(trans, ToDense(a, n, n, lda));
  const Dense<T> dx = ToDense(x, n, nrhs, ldx);
  const double resid = Norm1(Sub(Mul(opa, dx), ToDense(b, n, nrhs, ldb)));
  return resid / n / Norm1(opa) / Norm1(dx) / LapackEps<T>();
}

// ---------------------------------------------------------------------------
// Solution and inverse

TYPED_TEST(LinearTest, Solve) {
  using T = TypeParam;
  Random<T> rng(7);
  for (const hwyblas_int n : {0, 1, 2, 5, 16, 33, 70}) {
    for (const hwyblas_int nrhs : {0, 1, 3}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " nrhs=" << nrhs);
      const hwyblas_int lda = n + 2, ldb = n + 1;
      const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * n + 1);
      const std::vector<T> b =
          rng.Vector(static_cast<size_t>(ldb) * std::max(nrhs, 1));
      // xGESV.
      std::vector<T> lu = a, x = b;
      std::vector<hwyblas_int> ipiv(std::max(n, 1));
      ASSERT_EQ(Gesv(n, nrhs, lu, lda, ipiv, x, ldb), 0);
      if (n == 0 || nrhs == 0) {
        EXPECT_EQ(x, b);
        continue;
      }
      EXPECT_LT(SolveRatio('N', a, lda, x, ldb, b, ldb, n, nrhs),
                kLapackThreshold);
      // xGETRS with the factorization, for each op.
      for (const char trans : Ops<T>()) {
        std::vector<T> xt = b;
        ASSERT_EQ(Getrs(trans, n, nrhs, lu, lda, ipiv, xt, ldb), 0);
        EXPECT_LT(SolveRatio(trans, a, lda, xt, ldb, b, ldb, n, nrhs),
                  kLapackThreshold)
            << trans;
        if (trans == 'N') EXPECT_EQ(xt, x);
      }
    }
  }
  // An exactly singular matrix: INFO = k for U(k, k) = 0, B unchanged.
  const hwyblas_int n = 5;
  std::vector<T> a = rng.Vector(n * n);
  for (hwyblas_int i = 0; i < n; ++i) At(a, i, 2, n) = T(0);
  std::vector<T> b = rng.Vector(n), b0 = b;
  std::vector<hwyblas_int> ipiv(n);
  EXPECT_EQ(Gesv(n, 1, a, n, ipiv, b, n), 3);
  EXPECT_EQ(b, b0);
}

// The test ratio of LAPACK's xGET03: ||I - A * inv(A)||_1 /
// (n ||A||_1 ||inv(A)||_1 eps), with block sizes and workspaces that select
// the blocked code, the unblocked code, and smaller blocks.
TYPED_TEST(LinearTest, Inverse) {
  using T = TypeParam;
  Random<T> rng(11);
  for (const hwyblas_int nb : BlockSizes()) {
    const ScopedBlockSize block_size(nb);
    for (const hwyblas_int n : {0, 1, 2, 5, 16, 33, 70}) {
      const hwyblas_int lda = n + 1;
      std::vector<T> a = WellConditioned(rng, n, lda);
      std::vector<T> lu = a;
      std::vector<hwyblas_int> ipiv(std::max(n, 1));
      ASSERT_EQ(Getrf(n, lu, lda, ipiv), 0);
      // Workspace query.
      std::vector<T> query(1);
      ASSERT_EQ(Getri(n, lu, lda, ipiv, query, -1), 0);
      const hwyblas_int lwkopt = static_cast<hwyblas_int>(std::real(query[0]));
      if (nb > 0) EXPECT_EQ(lwkopt, std::max<hwyblas_int>(1, n * nb));
      for (const hwyblas_int lwork : {lwkopt, std::max<hwyblas_int>(1, n),
                                      std::max<hwyblas_int>(1, 2 * n)}) {
        SCOPED_TRACE(::testing::Message()
                     << "n=" << n << " nb=" << nb << " lwork=" << lwork);
        std::vector<T> inv = lu;
        std::vector<T> work(lwork);
        ASSERT_EQ(Getri(n, inv, lda, ipiv, work, lwork), 0);
        if (n == 0) continue;
        const Dense<T> da = ToDense(a, n, n, lda), di = ToDense(inv, n, n, lda);
        const double resid = Norm1(Sub(Identity<T>(n), Mul(da, di)));
        EXPECT_LT(resid / n / Norm1(da) / Norm1(di) / LapackEps<T>(),
                  kLapackThreshold);
      }
    }
  }
  // U(k, k) = 0: INFO = k.
  const hwyblas_int n = 4;
  std::vector<T> lu(n * n, T(1));
  At(lu, 2, 2, n) = T(0);
  std::vector<hwyblas_int> ipiv = {1, 2, 3, 4};
  std::vector<T> work(n);
  EXPECT_EQ(Getri(n, lu, n, ipiv, work, n), 3);
}

// ---------------------------------------------------------------------------
// Condition number

// The estimate of xGECON against the exact reciprocal condition number: the
// ratio of LAPACK's xGET06 (the larger of the two over the smaller).
TYPED_TEST(LinearTest, ConditionNumber) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(13);
  for (const char norm : {'1', 'O', 'I'}) {
    for (const hwyblas_int n : {1, 2, 7, 25, 50}) {
      SCOPED_TRACE(::testing::Message() << norm << " n=" << n);
      const hwyblas_int lda = n + 2;
      const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * n);
      std::vector<T> lu = a;
      std::vector<hwyblas_int> ipiv(n);
      ASSERT_EQ(Getrf(n, lu, lda, ipiv), 0);
      const R anorm = Lange(norm, n, n, a, lda);
      R rcond = -1;
      ASSERT_EQ(Gecon(norm, n, lu, lda, anorm, &rcond), 0);
      const Dense<T> da = ToDense(a, n, n, lda);
      const Dense<T> inv = Inverse(da);
      const double rcondc = norm == 'I' ? 1 / (NormInf(da) * NormInf(inv))
                                        : 1 / (Norm1(da) * Norm1(inv));
      ASSERT_GT(rcond, 0);
      EXPECT_LT(std::max(rcond / rcondc, rcondc / rcond), kLapackThreshold)
          << rcond << " " << rcondc;
    }
  }
}

// The special cases of xGECON: n = 0 (RCOND = 1), ANORM = 0 (RCOND = 0),
// ANORM NaN (RCOND = NaN, INFO = -5, not reported as an error) or infinite
// (INFO = -5), a singular factor (RCOND = 0), and negative ANORM (an
// error).
TYPED_TEST(LinearTest, ConditionNumberSpecialCases) {
  using T = TypeParam;
  using R = RealT<T>;
  const hwyblas_int n = 3;
  std::vector<T> lu = {T(2), T(0), T(0), T(1), T(3), T(0), T(1), T(1), T(4)};
  R rcond = -1;
  ErrorCapture errors;
  EXPECT_EQ(Gecon('1', 0, lu, 1, R(1), &rcond), 0);
  EXPECT_EQ(rcond, 1);
  EXPECT_EQ(Gecon('1', n, lu, n, R(0), &rcond), 0);
  EXPECT_EQ(rcond, 0);
  EXPECT_EQ(Gecon('1', n, lu, n, std::numeric_limits<R>::quiet_NaN(), &rcond),
            -5);
  EXPECT_TRUE(std::isnan(rcond));
  EXPECT_EQ(Gecon('1', n, lu, n, std::numeric_limits<R>::infinity(), &rcond),
            -5);
  EXPECT_EQ(rcond, 0);
  EXPECT_TRUE(errors.Take().empty());
  std::vector<T> singular = lu;
  At(singular, 1, 1, n) = T(0);
  EXPECT_EQ(Gecon('I', n, singular, n, R(5), &rcond), 0);
  EXPECT_EQ(rcond, 0);
  EXPECT_EQ(Gecon('X', n, lu, n, R(1), &rcond), -1);
  EXPECT_EQ(Gecon('1', -1, lu, n, R(1), &rcond), -2);
  EXPECT_EQ(Gecon('1', n, lu, n - 1, R(1), &rcond), -4);
  EXPECT_EQ(Gecon('1', n, lu, n, R(-1), &rcond), -5);
  const std::string con = Name<T>("GECON");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{con + ":1", con + ":2",
                                                     con + ":4", con + ":5"}));
}

// ---------------------------------------------------------------------------
// Equilibration

// xGEEQU computes exactly the scalings of its definition (recomputed here
// with the same operations), and xLAQGE applies them as far as worthwhile.
TYPED_TEST(LinearTest, Equilibration) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(17);
  const R smlnum = std::numeric_limits<R>::min();
  const R bignum = 1 / smlnum;
  const hwyblas_int m = 7, n = 5, lda = 8;
  // Rows scaled by 2**(3i), columns by 2**(-2j): ROWCND and COLCND are
  // small.
  std::vector<T> a = rng.Vector(lda * n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      At(a, i, j, lda) *= static_cast<R>(std::ldexp(1.0, 3 * i - 2 * j));
    }
  }
  std::vector<R> r(m), c(n);
  R rowcnd = -1, colcnd = -1, amax = -1;
  ASSERT_EQ(Geequ(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax), 0);
  std::vector<R> r_c(m, 0), c_c(n, 0);
  R amax_c = 0;
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      r_c[i] = std::max(r_c[i], Cabs1W(At(a, i, j, lda)));
    }
  }
  R rmin = bignum, rmax = 0;
  for (hwyblas_int i = 0; i < m; ++i) {
    rmin = std::min(rmin, r_c[i]);
    rmax = std::max(rmax, r_c[i]);
    amax_c = std::max(amax_c, r_c[i]);
    r_c[i] = 1 / std::min(std::max(r_c[i], smlnum), bignum);
  }
  EXPECT_EQ(r, r_c);
  EXPECT_EQ(amax, amax_c);
  EXPECT_EQ(rowcnd, std::max(rmin, smlnum) / std::min(rmax, bignum));
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      c_c[j] = std::max(c_c[j], Cabs1W(At(a, i, j, lda)) * r_c[i]);
    }
  }
  R cmin = bignum, cmax = 0;
  for (hwyblas_int j = 0; j < n; ++j) {
    cmin = std::min(cmin, c_c[j]);
    cmax = std::max(cmax, c_c[j]);
    c_c[j] = 1 / std::min(std::max(c_c[j], smlnum), bignum);
  }
  EXPECT_EQ(c, c_c);
  EXPECT_EQ(colcnd, std::max(cmin, smlnum) / std::min(cmax, bignum));
  EXPECT_LT(rowcnd, 0.1);
  EXPECT_LT(colcnd, 0.1);

  // xLAQGE: ROWCND and COLCND < 0.1 scale rows and columns, otherwise only
  // what is badly scaled; AMAX close to overflow forces row scaling.
  const auto laqge = [&](R rcnd, R ccnd, R am, std::vector<T>& x) {
    char equed = '?';
    HWYBLAS_CALL(T, laqge, &m, &n, P(x.data()), &lda, r.data(), c.data(), &rcnd,
                 &ccnd, &am, &equed);
    return equed;
  };
  std::vector<T> x = a;
  EXPECT_EQ(laqge(rowcnd, colcnd, amax, x), 'B');
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      ASSERT_EQ(At(x, i, j, lda), c[j] * r[i] * At(a, i, j, lda));
    }
  }
  x = a;
  EXPECT_EQ(laqge(R(0.5), colcnd, R(1), x), 'C');
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      ASSERT_EQ(At(x, i, j, lda), c[j] * At(a, i, j, lda));
    }
  }
  x = a;
  EXPECT_EQ(laqge(rowcnd, R(0.5), R(1), x), 'R');
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      ASSERT_EQ(At(x, i, j, lda), r[i] * At(a, i, j, lda));
    }
  }
  x = a;
  EXPECT_EQ(laqge(R(0.5), R(0.5), R(1), x), 'N');
  EXPECT_EQ(x, a);
  EXPECT_EQ(laqge(R(0.5), R(0.5), std::numeric_limits<R>::max(), x), 'R');

  // A zero row i gives INFO = i, a zero column j INFO = m + j.
  std::vector<T> z = a;
  for (hwyblas_int j = 0; j < n; ++j) At(z, 4, j, lda) = T(0);
  EXPECT_EQ(Geequ(m, n, z, lda, r, c, &rowcnd, &colcnd, &amax), 5);
  z = a;
  for (hwyblas_int i = 0; i < m; ++i) At(z, i, 3, lda) = T(0);
  EXPECT_EQ(Geequ(m, n, z, lda, r, c, &rowcnd, &colcnd, &amax), m + 4);
  // Empty matrix.
  EXPECT_EQ(Geequ(0, n, a, 1, r, c, &rowcnd, &colcnd, &amax), 0);
  EXPECT_EQ(rowcnd, 1);
  EXPECT_EQ(colcnd, 1);
  EXPECT_EQ(amax, 0);
}

// ---------------------------------------------------------------------------
// Iterative refinement

// Solutions perturbed by a known relative error: xGERFS refines them (the
// error decreases to about the rounding error) and bounds the remaining
// error; the backward error is that of the refined solution.
TYPED_TEST(LinearTest, Refinement) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(19);
  const double delta = std::is_same_v<R, float> ? 1e-2 : 1e-6;
  for (const char trans : Ops<T>()) {
    for (const hwyblas_int n : {1, 6, 40}) {
      for (const double bscale : {1.0, NearUnderflow<T>()}) {
        SCOPED_TRACE(::testing::Message()
                     << trans << " n=" << n << " " << bscale);
        const hwyblas_int nrhs = 2, lda = n + 1, ldb = n + 2, ldx = n + 3;
        const std::vector<T> a = WellConditioned(rng, n, lda);
        std::vector<T> af = a;
        std::vector<hwyblas_int> ipiv(n);
        ASSERT_EQ(Getrf(n, af, lda, ipiv), 0);
        std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
        for (T& e : b) e *= static_cast<RealT<T>>(bscale);
        const Dense<T> opa = Op(trans, ToDense(a, n, n, lda));
        const Dense<T> xt = Mul(Inverse(opa), ToDense(b, n, nrhs, ldb));
        std::vector<T> x(static_cast<size_t>(ldx) * nrhs);
        for (hwyblas_int j = 0; j < nrhs; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) {
            At(x, i, j, ldx) =
                static_cast<T>(xt(i, j) * (1 + delta * std::real(rng.Next())));
          }
        }
        std::vector<R> ferr(nrhs, -1), berr(nrhs, -1);
        ASSERT_EQ(Gerfs(trans, n, nrhs, a, lda, af, lda, ipiv, b, ldb, x, ldx,
                        ferr, berr),
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
          if (bscale == 1) {
            EXPECT_LT(berr[j], 4 * (n + 1) * LapackEps<T>());
            EXPECT_LT(ferr[j], 1e3 * n * LapackEps<T>());
          } else {
            // Near underflow, SAFE1 dominates the backward error.
            std::vector<double> resid(n), denom(n);
            for (hwyblas_int i = 0; i < n; ++i) {
              RefT<T> r = -static_cast<RefT<T>>(At(b, i, j, ldb));
              denom[i] = Cabs1Ref(At(b, i, j, ldb));
              for (hwyblas_int k = 0; k < n; ++k) {
                r += opa(i, k) * static_cast<RefT<T>>(At(x, k, j, ldx));
                denom[i] += Cabs1Ref(opa(i, k)) * Cabs1Ref(At(x, k, j, ldx));
              }
              resid[i] = Cabs1Ref(r);
            }
            const double berr_c = BackwardError<T>(resid, denom);
            EXPECT_NEAR(berr[j], berr_c, 0.01 * berr_c);
            // W >= SAFE1 in the bound.
            const double safe1 = (n + 1) * std::numeric_limits<R>::min();
            EXPECT_GE(ferr[j], 0.3 * safe1 / (NormInf(opa) * xnorm));
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Expert driver

// A badly scaled matrix (rows scaled by up to 2**6, columns by up to 2**-8) for
// equilibration, otherwise well-conditioned.
template <typename T>
std::vector<T> BadlyScaled(Random<T>& rng, hwyblas_int n, hwyblas_int lda) {
  std::vector<T> a = WellConditioned(rng, n, lda);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      At(a, i, j, lda) *=
          static_cast<RealT<T>>(std::ldexp(1.0, 2 * (i % 4) - 2 * (j % 5)));
    }
  }
  return a;
}

// xGESVX with FACT = 'N' and 'E' (on well and badly scaled matrices), and
// 'F' with the factorization and equilibration of an 'E' call, which gives
// the same solution. The solution solves the original system; RCOND and
// the pivot growth factor are those of xGECON, xLANGE and xLANTR.
TYPED_TEST(LinearTest, ExpertDriver) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(23);
  for (const bool scaled : {false, true}) {
    for (const char trans : Ops<T>()) {
      for (const hwyblas_int n : {1, 4, 30}) {
        SCOPED_TRACE(::testing::Message()
                     << (scaled ? "scaled " : "") << trans << " n=" << n);
        GesvxCall<T> call;
        call.trans = trans;
        call.n = n;
        call.nrhs = 3;
        call.lda = n + 1;
        call.ldaf = n + 2;
        call.ldb = n + 3;
        call.ldx = n + 4;
        const std::vector<T> a0 = scaled ? BadlyScaled(rng, n, call.lda)
                                         : WellConditioned(rng, n, call.lda);
        const std::vector<T> b0 =
            rng.Vector(static_cast<size_t>(call.ldb) * call.nrhs);
        for (const char fact : {'N', 'E'}) {
          GesvxCall<T> c = call;
          c.fact = fact;
          c.a = a0;
          c.b = b0;
          ASSERT_EQ(c.Run(), 0) << fact;
          EXPECT_LT(
              SolveRatio(trans, a0, c.lda, c.x, c.ldx, b0, c.ldb, n, c.nrhs),
              kLapackThreshold)
              << fact;
          for (hwyblas_int j = 0; j < c.nrhs; ++j) {
            EXPECT_LT(c.berr[j], 4 * (n + 1) * LapackEps<T>());
            EXPECT_GE(c.ferr[j], 0);
          }
          const bool rowequ = c.equed == 'R' || c.equed == 'B';
          const bool colequ = c.equed == 'C' || c.equed == 'B';
          if (fact == 'N' || !scaled) {
            EXPECT_EQ(c.equed, 'N');
            EXPECT_EQ(c.a, a0);
          } else if (n > 4) {
            EXPECT_NE(c.equed, 'N');
          }
          // A returns the equilibrated matrix, B the scaled right hand side.
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              T e = At(a0, i, j, c.lda);
              if (rowequ && colequ) {
                e = c.c[j] * c.r[i] * e;
              } else if (rowequ) {
                e = c.r[i] * e;
              } else if (colequ) {
                e = c.c[j] * e;
              }
              ASSERT_EQ(At(c.a, i, j, c.lda), e);
            }
          }
          for (hwyblas_int j = 0; j < c.nrhs; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              T e = At(b0, i, j, c.ldb);
              if (trans == 'N' && rowequ) e = c.r[i] * e;
              if (trans != 'N' && colequ) e = c.c[i] * e;
              ASSERT_EQ(At(c.b, i, j, c.ldb), e);
            }
          }
          // RCOND of the factored (equilibrated) matrix, the pivot growth.
          const char norm = trans == 'N' ? '1' : 'I';
          R rcond = -1;
          ASSERT_EQ(Gecon(norm, n, c.af, c.ldaf, Lange(norm, n, n, c.a, c.lda),
                          &rcond),
                    0);
          EXPECT_EQ(c.rcond, rcond);
          EXPECT_EQ(c.rpvgrw, Lange('M', n, n, c.a, c.lda) /
                                  LantrUpperMax(n, n, c.af, c.ldaf));
          if (fact != 'E') continue;
          // FACT = 'F' with the results of the 'E' call.
          GesvxCall<T> f = c;
          f.fact = 'F';
          f.b = b0;
          f.x.assign(f.x.size(), T(0));
          ASSERT_EQ(f.Run(), 0);
          EXPECT_EQ(f.x, c.x);
          EXPECT_EQ(f.rcond, c.rcond);
          // FERR is divided by ROWCND or COLCND, which FACT = 'F'
          // computes from R and C rather than from the row and column
          // maxima (with other roundings).
          for (hwyblas_int j = 0; j < c.nrhs; ++j) {
            EXPECT_NEAR(f.ferr[j], c.ferr[j], 4 * Eps<T>() * c.ferr[j]);
          }
          EXPECT_EQ(f.berr, c.berr);
          EXPECT_EQ(f.a, c.a);
          EXPECT_EQ(f.af, c.af);
        }
      }
    }
  }
}

// An exactly singular matrix: INFO = k (U(k, k) = 0), RCOND = 0, the pivot
// growth factor of the first k columns, and no solution. A matrix singular
// to working precision: INFO = n + 1 with a solution.
TYPED_TEST(LinearTest, ExpertDriverSingular) {
  using T = TypeParam;
  Random<T> rng(29);
  const hwyblas_int n = 6;
  GesvxCall<T> c;
  c.n = n;
  c.nrhs = 1;
  c.lda = c.ldaf = c.ldb = c.ldx = n;
  c.a = rng.Vector(n * n);
  for (hwyblas_int i = 0; i < n; ++i) At(c.a, i, 3, n) = T(0);
  c.b = rng.Vector(n);
  const std::vector<T> a0 = c.a;
  EXPECT_EQ(c.Run(), 4);
  EXPECT_EQ(c.rcond, 0);
  EXPECT_EQ(c.rpvgrw,
            Lange('M', n, 4, a0, n) / LantrUpperMax(4, 4, c.af, c.ldaf));
  EXPECT_EQ(c.ferr[0], -1);  // not computed

  // An upper triangular matrix with a unit diagonal, except for the last
  // element eps**2.
  GesvxCall<T> d;
  d.n = n;
  d.nrhs = 1;
  d.lda = d.ldaf = d.ldb = d.ldx = n;
  d.a.assign(n * n, T(0));
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < j; ++i)
      At(d.a, i, j, n) = rng.Next() / RealT<T>(2);
    At(d.a, j, j, n) = T(1);
  }
  At(d.a, n - 1, n - 1, n) = T(static_cast<RealT<T>>(Eps<T>() * Eps<T>()));
  d.b = rng.Vector(n);
  EXPECT_EQ(d.Run(), n + 1);
  EXPECT_LT(d.rcond, LapackEps<T>());
  EXPECT_GE(d.ferr[0], 0);
}

// ---------------------------------------------------------------------------
// Errors

TYPED_TEST(LinearTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16), work(16);
  std::vector<hwyblas_int> ipiv = {1, 2, 3, 4};
  std::vector<R> ferr(4), berr(4);
  ErrorCapture errors;
  EXPECT_EQ(Getrs('X', 4, 1, a, 4, ipiv, b, 4), -1);
  EXPECT_EQ(Getrs('N', -1, 1, a, 4, ipiv, b, 4), -2);
  EXPECT_EQ(Getrs('N', 4, -1, a, 4, ipiv, b, 4), -3);
  EXPECT_EQ(Getrs('N', 4, 1, a, 3, ipiv, b, 4), -5);
  EXPECT_EQ(Getrs('N', 4, 1, a, 4, ipiv, b, 3), -8);
  const std::string trs = Name<T>("GETRS");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{trs + ":1", trs + ":2", trs + ":3",
                                      trs + ":5", trs + ":8"}));
  EXPECT_EQ(Gesv(-1, 1, a, 4, ipiv, b, 4), -1);
  EXPECT_EQ(Gesv(4, -1, a, 4, ipiv, b, 4), -2);
  EXPECT_EQ(Gesv(4, 1, a, 3, ipiv, b, 4), -4);
  EXPECT_EQ(Gesv(4, 1, a, 4, ipiv, b, 3), -7);
  const std::string sv = Name<T>("GESV");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{sv + ":1", sv + ":2",
                                                     sv + ":4", sv + ":7"}));
  EXPECT_EQ(Getri(-1, a, 4, ipiv, work, 4), -1);
  EXPECT_EQ(Getri(4, a, 3, ipiv, work, 4), -3);
  EXPECT_EQ(Getri(4, a, 4, ipiv, work, 3), -6);
  const std::string tri = Name<T>("GETRI");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{tri + ":1", tri + ":3", tri + ":6"}));
  std::vector<R> r(4), c(4);
  R rowcnd, colcnd, amax;
  EXPECT_EQ(Geequ(-1, 4, a, 4, r, c, &rowcnd, &colcnd, &amax), -1);
  EXPECT_EQ(Geequ(4, -1, a, 4, r, c, &rowcnd, &colcnd, &amax), -2);
  EXPECT_EQ(Geequ(4, 4, a, 3, r, c, &rowcnd, &colcnd, &amax), -4);
  const std::string equ = Name<T>("GEEQU");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{equ + ":1", equ + ":2", equ + ":4"}));
  EXPECT_EQ(Gerfs('X', 4, 1, a, 4, a, 4, ipiv, b, 4, b, 4, ferr, berr), -1);
  EXPECT_EQ(Gerfs('N', -1, 1, a, 4, a, 4, ipiv, b, 4, b, 4, ferr, berr), -2);
  EXPECT_EQ(Gerfs('N', 4, -1, a, 4, a, 4, ipiv, b, 4, b, 4, ferr, berr), -3);
  EXPECT_EQ(Gerfs('N', 4, 1, a, 3, a, 4, ipiv, b, 4, b, 4, ferr, berr), -5);
  EXPECT_EQ(Gerfs('N', 4, 1, a, 4, a, 3, ipiv, b, 4, b, 4, ferr, berr), -7);
  EXPECT_EQ(Gerfs('N', 4, 1, a, 4, a, 4, ipiv, b, 3, b, 4, ferr, berr), -10);
  EXPECT_EQ(Gerfs('N', 4, 1, a, 4, a, 4, ipiv, b, 4, b, 3, ferr, berr), -12);
  const std::string rfs = Name<T>("GERFS");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               rfs + ":1", rfs + ":2", rfs + ":3", rfs + ":5",
                               rfs + ":7", rfs + ":10", rfs + ":12"}));
  const auto gesvx = [&](char fact, char trans, hwyblas_int n, hwyblas_int nrhs,
                         hwyblas_int lda, hwyblas_int ldaf, char equed, R rc,
                         hwyblas_int ldb, hwyblas_int ldx) {
    GesvxCall<T> call;
    call.fact = fact;
    call.trans = trans;
    call.n = n;
    call.nrhs = nrhs;
    call.lda = lda;
    call.ldaf = ldaf;
    call.ldb = ldb;
    call.ldx = ldx;
    call.equed = equed;
    call.a.assign(16, T(1));
    call.b.assign(16, T(1));
    call.r.assign(4, rc);
    call.c.assign(4, rc);
    call.af.assign(16, T(1));
    return call.Run();
  };
  EXPECT_EQ(gesvx('X', 'N', 4, 1, 4, 4, 'N', 1, 4, 4), -1);
  EXPECT_EQ(gesvx('N', 'X', 4, 1, 4, 4, 'N', 1, 4, 4), -2);
  EXPECT_EQ(gesvx('N', 'N', -1, 1, 4, 4, 'N', 1, 4, 4), -3);
  EXPECT_EQ(gesvx('N', 'N', 4, -1, 4, 4, 'N', 1, 4, 4), -4);
  EXPECT_EQ(gesvx('N', 'N', 4, 1, 3, 4, 'N', 1, 4, 4), -6);
  EXPECT_EQ(gesvx('N', 'N', 4, 1, 4, 3, 'N', 1, 4, 4), -8);
  EXPECT_EQ(gesvx('F', 'N', 4, 1, 4, 4, 'X', 1, 4, 4), -10);
  EXPECT_EQ(gesvx('F', 'N', 4, 1, 4, 4, 'R', 0, 4, 4), -11);
  EXPECT_EQ(gesvx('F', 'N', 4, 1, 4, 4, 'C', 0, 4, 4), -12);
  EXPECT_EQ(gesvx('N', 'N', 4, 1, 4, 4, 'N', 1, 3, 4), -14);
  EXPECT_EQ(gesvx('N', 'N', 4, 1, 4, 4, 'N', 1, 4, 3), -16);
  const std::string svx = Name<T>("GESVX");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               svx + ":1", svx + ":2", svx + ":3", svx + ":4",
                               svx + ":6", svx + ":8", svx + ":10", svx + ":11",
                               svx + ":12", svx + ":14", svx + ":16"}));
}

// ---------------------------------------------------------------------------
// LAPACKE

// The row-major functions give the results of the column-major ones (the
// same arithmetic on transposed copies).
TYPED_TEST(LinearTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(31);
  const lapack_int n = 8, nrhs = 3, lda = 10, ldb = 9;
  const std::vector<T> a = BadlyScaled(rng, n, lda);
  const std::vector<T> ar = RowMajor(a, n, n, lda, lda);
  const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * n);
  const auto same = [&](const std::vector<T>& row, const std::vector<T>& col,
                        lapack_int rows, lapack_int cols, lapack_int ldr,
                        lapack_int ldc) {
    for (lapack_int i = 0; i < rows; ++i) {
      for (lapack_int j = 0; j < cols; ++j) {
        ASSERT_EQ(row[i * ldr + j], At(col, i, j, ldc)) << i << ", " << j;
      }
    }
  };
  // xGESV, xGETRS, xGETRI, xGECON, xGERFS.
  std::vector<T> lc = a, lr = ar, xc = b, xr = RowMajor(b, n, nrhs, ldb, ldb);
  std::vector<lapack_int> pc(n), pr(n);
  ASSERT_EQ(LAPACKE_CALL(T, gesv, LAPACK_COL_MAJOR, n, nrhs, lc.data(), lda,
                         pc.data(), xc.data(), ldb),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, gesv, LAPACK_ROW_MAJOR, n, nrhs, lr.data(), lda,
                         pr.data(), xr.data(), ldb),
            0);
  EXPECT_EQ(pr, pc);
  same(lr, lc, n, n, lda, lda);
  same(xr, xc, n, nrhs, ldb, ldb);
  for (const char trans : Ops<T>()) {
    std::vector<T> yc = b, yr = RowMajor(b, n, nrhs, ldb, ldb);
    ASSERT_EQ(LAPACKE_CALL(T, getrs, LAPACK_COL_MAJOR, trans, n, nrhs,
                           lc.data(), lda, pc.data(), yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, getrs, LAPACK_ROW_MAJOR, trans, n, nrhs,
                           lr.data(), lda, pr.data(), yr.data(), ldb),
              0);
    same(yr, yc, n, nrhs, ldb, ldb);
    std::vector<R> fc(nrhs), bc(nrhs), fr(nrhs), brr(nrhs);
    const std::vector<T> b_row = RowMajor(b, n, nrhs, ldb, ldb);
    ASSERT_EQ(LAPACKE_CALL(T, gerfs, LAPACK_COL_MAJOR, trans, n, nrhs, a.data(),
                           lda, lc.data(), lda, pc.data(), b.data(), ldb,
                           yc.data(), ldb, fc.data(), bc.data()),
              0);
    ASSERT_EQ(
        LAPACKE_CALL(T, gerfs, LAPACK_ROW_MAJOR, trans, n, nrhs, ar.data(), lda,
                     lr.data(), lda, pr.data(), b_row.data(), ldb, yr.data(),
                     ldb, fr.data(), brr.data()),
        0);
    same(yr, yc, n, nrhs, ldb, ldb);
    EXPECT_EQ(fr, fc);
    EXPECT_EQ(brr, bc);
  }
  for (const char norm : {'1', 'I'}) {
    R rc = -1, rr = -2;
    ASSERT_EQ(LAPACKE_CALL(T, gecon, LAPACK_COL_MAJOR, norm, n, lc.data(), lda,
                           R(3), &rc),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gecon, LAPACK_ROW_MAJOR, norm, n, lr.data(), lda,
                           R(3), &rr),
              0);
    EXPECT_EQ(rr, rc);
  }
  ASSERT_EQ(
      LAPACKE_CALL(T, getri, LAPACK_COL_MAJOR, n, lc.data(), lda, pc.data()),
      0);
  ASSERT_EQ(
      LAPACKE_CALL(T, getri, LAPACK_ROW_MAJOR, n, lr.data(), lda, pr.data()),
      0);
  same(lr, lc, n, n, lda, lda);
  // xGEEQU of an m x n matrix.
  const lapack_int m = 6;
  std::vector<R> rc(m), cc(n), rr(m), cr(n);
  R rowc, colc, amc, rowr, colr, amr;
  ASSERT_EQ(LAPACKE_CALL(T, geequ, LAPACK_COL_MAJOR, m, n, a.data(), lda,
                         rc.data(), cc.data(), &rowc, &colc, &amc),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, geequ, LAPACK_ROW_MAJOR, m, n,
                         RowMajor(a, m, n, lda, lda).data(), lda, rr.data(),
                         cr.data(), &rowr, &colr, &amr),
            0);
  EXPECT_EQ(rr, rc);
  EXPECT_EQ(cr, cc);
  EXPECT_EQ(rowr, rowc);
  EXPECT_EQ(colr, colc);
  EXPECT_EQ(amr, amc);
  // xGESVX: A (equilibrated), AF, B (scaled, also for FACT = 'E': a
  // deviation from the reference LAPACKE) and X are those of the
  // column-major call.
  for (const char fact : {'N', 'E'}) {
    for (const char trans : Ops<T>()) {
      SCOPED_TRACE(::testing::Message() << fact << trans);
      std::vector<T> ac = a, arr = ar, afc(lda * n), afr(lda * n);
      std::vector<T> bcc = b, brw = RowMajor(b, n, nrhs, ldb, ldb);
      std::vector<T> x1(ldb * nrhs), x2(ldb * n);
      std::vector<lapack_int> p1(n), p2(n);
      std::vector<R> r1(n), c1(n), r2(n), c2(n), f1(nrhs), e1(nrhs), f2(nrhs),
          e2(nrhs);
      char eq1 = 'N', eq2 = 'N';
      R rc1, rc2, g1, g2;
      ASSERT_EQ(LAPACKE_CALL(T, gesvx, LAPACK_COL_MAJOR, fact, trans, n, nrhs,
                             ac.data(), lda, afc.data(), lda, p1.data(), &eq1,
                             r1.data(), c1.data(), bcc.data(), ldb, x1.data(),
                             ldb, &rc1, f1.data(), e1.data(), &g1),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, gesvx, LAPACK_ROW_MAJOR, fact, trans, n, nrhs,
                             arr.data(), lda, afr.data(), lda, p2.data(), &eq2,
                             r2.data(), c2.data(), brw.data(), ldb, x2.data(),
                             ldb, &rc2, f2.data(), e2.data(), &g2),
                0);
      EXPECT_EQ(eq2, eq1);
      if (fact == 'E') EXPECT_NE(eq1, 'N');
      same(arr, ac, n, n, lda, lda);
      same(afr, afc, n, n, lda, lda);
      same(brw, bcc, n, nrhs, ldb, ldb);
      same(x2, x1, n, nrhs, ldb, ldb);
      EXPECT_EQ(rc2, rc1);
      EXPECT_EQ(g2, g1);
      EXPECT_EQ(f2, f1);
      EXPECT_EQ(e2, e1);
    }
  }
}

TYPED_TEST(LinearTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16, T(1)), x(16);
  std::vector<lapack_int> ipiv = {1, 2, 3, 4};
  std::vector<R> r(4, 1), c(4, 1), ferr(4), berr(4);
  R rcond = 0, rpivot = 0, rowcnd, colcnd, amax;
  char equed = 'N';
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(
      LAPACKE_CALL(T, gesv, 0, 4, 1, a.data(), 4, ipiv.data(), b.data(), 4),
      -1);
  EXPECT_EQ(LAPACKE_CALL(T, gesv, LAPACK_ROW_MAJOR, 4, 2, a.data(), 4,
                         ipiv.data(), b.data(), 1),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, getrs, LAPACK_ROW_MAJOR, 'N', 4, 1, a.data(), 3,
                         ipiv.data(), b.data(), 1),
            -6);
  EXPECT_EQ(
      LAPACKE_CALL(T, getri, LAPACK_ROW_MAJOR, 4, a.data(), 3, ipiv.data()),
      -4);
  EXPECT_EQ(LAPACKE_CALL(T, gecon, LAPACK_ROW_MAJOR, '1', 4, a.data(), 3, R(1),
                         &rcond),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, geequ, LAPACK_ROW_MAJOR, 2, 4, a.data(), 3,
                         r.data(), c.data(), &rowcnd, &colcnd, &amax),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, gerfs, LAPACK_ROW_MAJOR, 'N', 4, 2, a.data(), 4,
                         a.data(), 4, ipiv.data(), b.data(), 4, x.data(), 1,
                         ferr.data(), berr.data()),
            -13);
  EXPECT_EQ(LAPACKE_CALL(T, gesvx, LAPACK_ROW_MAJOR, 'N', 'N', 4, 2, a.data(),
                         4, a.data(), 4, ipiv.data(), &equed, r.data(),
                         c.data(), b.data(), 1, x.data(), 4, &rcond,
                         ferr.data(), berr.data(), &rpivot),
            -15);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("gesv") +
                "\nWrong parameter 8 in " + LapackeName<T>("gesv_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("getrs_work") +
                "\nWrong parameter 4 in " + LapackeName<T>("getri_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("gecon_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("geequ_work") +
                "\nWrong parameter 13 in " + LapackeName<T>("gerfs_work") +
                "\nWrong parameter 15 in " + LapackeName<T>("gesvx_work") +
                "\n");
  // NaN checks.
  b[1] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, gesv, LAPACK_COL_MAJOR, 4, 1, a.data(), 4,
                         ipiv.data(), b.data(), 4),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, gerfs, LAPACK_COL_MAJOR, 'N', 4, 1, a.data(), 4,
                         a.data(), 4, ipiv.data(), a.data(), 4, b.data(), 4,
                         ferr.data(), berr.data()),
            -12);
  EXPECT_EQ(LAPACKE_CALL(T, gecon, LAPACK_COL_MAJOR, '1', 4, a.data(), 4,
                         std::numeric_limits<R>::quiet_NaN(), &rcond),
            -6);
  // xGESVX checks R and C only if they are used (FACT = 'F', EQUED).
  b[1] = T(1);
  r[2] = std::numeric_limits<R>::quiet_NaN();
  equed = 'C';
  EXPECT_NE(LAPACKE_CALL(T, gesvx, LAPACK_COL_MAJOR, 'F', 'N', 4, 1, a.data(),
                         4, a.data(), 4, ipiv.data(), &equed, r.data(),
                         c.data(), b.data(), 4, x.data(), 4, &rcond,
                         ferr.data(), berr.data(), &rpivot),
            -12);
  equed = 'R';
  EXPECT_EQ(LAPACKE_CALL(T, gesvx, LAPACK_COL_MAJOR, 'F', 'N', 4, 1, a.data(),
                         4, a.data(), 4, ipiv.data(), &equed, r.data(),
                         c.data(), b.data(), 4, x.data(), 4, &rcond,
                         ferr.data(), berr.data(), &rpivot),
            -12);
  a[5] = NaN<T>();
  EXPECT_EQ(
      LAPACKE_CALL(T, getri, LAPACK_COL_MAJOR, 4, a.data(), 4, ipiv.data()),
      -3);
}

}  // namespace
}  // namespace hwyblas_test
