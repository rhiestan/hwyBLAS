// Tests of the singular value decomposition by divide and conquer: xGESDD,
// the bidiagonal SVD xBDSDC and its steps (xLASDQ, xLASD0 - xLASD8, xLASDA,
// xLASDT, xLAMRG, xLAED6), xLACP2 / xLACRM / xLARCM, and LAPACKE.
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
#include "lapack/svd_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class GesddTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GesddTest, ScalarTypes);

template <typename R>
class GesddRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GesddRealTest, RealTypes);

template <typename T>
class GesddComplexTest : public ::testing::Test {};

using ComplexTypes =
    ::testing::Types<std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GesddComplexTest, ComplexTypes);

// Forces the maximum size SMLSIZ of the subproblems of divide and conquer
// (ILAENV ispec = 9) while it exists, so that small matrices are split into
// several levels.
class ScopedSubproblemSize {
 public:
  explicit ScopedSubproblemSize(hwyblas_int smlsiz) {
    smlsiz_ = smlsiz;
    previous_ = hwyblas_set_ilaenv(smlsiz > 0 ? &Ilaenv : nullptr);
  }
  ~ScopedSubproblemSize() {
    hwyblas_set_ilaenv(previous_);
    smlsiz_ = 0;
  }
  ScopedSubproblemSize(const ScopedSubproblemSize&) = delete;
  ScopedSubproblemSize& operator=(const ScopedSubproblemSize&) = delete;

 private:
  static hwyblas_int Ilaenv(const hwyblas_int* ispec, const char* name,
                            const char* opts, const hwyblas_int* n1,
                            const hwyblas_int* n2, const hwyblas_int* n3,
                            const hwyblas_int* n4, size_t name_len,
                            size_t opts_len) {
    if (*ispec == 9) return smlsiz_;
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }

  static inline hwyblas_int smlsiz_ = 0;
  hwyblas_ilaenv_function previous_ = nullptr;
};

// ---------------------------------------------------------------------------
// The steps of xBDSDC

// xLAMRG: the permutation that merges two sorted lists, each in ascending or
// descending order.
TYPED_TEST(GesddRealTest, Lamrg) {
  using R = TypeParam;
  Random<R> rng(181);
  for (const hwyblas_int n1 : {0, 1, 4, 9}) {
    for (const hwyblas_int n2 : {0, 1, 5}) {
      for (const hwyblas_int dtrd1 : {1, -1}) {
        for (const hwyblas_int dtrd2 : {1, -1}) {
          SCOPED_TRACE(::testing::Message()
                       << n1 << " " << n2 << " " << dtrd1 << " " << dtrd2);
          std::vector<R> a = rng.Vector(n1 + n2);
          a.push_back(0);
          // Ties between the lists: equal elements.
          if (n1 > 1 && n2 > 1) a[n1] = a[0];
          std::sort(a.begin(), a.begin() + n1);
          std::sort(a.begin() + n1, a.begin() + n1 + n2);
          if (dtrd1 < 0) std::reverse(a.begin(), a.begin() + n1);
          if (dtrd2 < 0) std::reverse(a.begin() + n1, a.begin() + n1 + n2);
          std::vector<hwyblas_int> index(n1 + n2 + 1, -1);
          HWYBLAS_CALL_REAL(R, lamrg, &n1, &n2, a.data(), &dtrd1, &dtrd2,
                            index.data());
          std::vector<bool> seen(n1 + n2, false);
          for (hwyblas_int i = 0; i < n1 + n2; ++i) {
            ASSERT_GE(index[i], 1);
            ASSERT_LE(index[i], n1 + n2);
            ASSERT_FALSE(seen[index[i] - 1]);
            seen[index[i] - 1] = true;
            if (i > 0) ASSERT_LE(a[index[i - 1] - 1], a[index[i] - 1]) << i;
          }
          EXPECT_EQ(index[n1 + n2], -1);
        }
      }
    }
  }
}

// xLASDT: the tree of subproblems, with nd = 2**lvl - 1 nodes whose children
// split them around their center row, and leaves of at most msub rows on
// either side.
TYPED_TEST(GesddRealTest, Lasdt) {
  using R = TypeParam;
  for (const hwyblas_int msub : {3, 25}) {
    for (const hwyblas_int n : {4, 5, 7, 26, 27, 51, 100, 1000}) {
      // (xLASDT is used for n > msub only.)
      if (n <= msub) continue;
      SCOPED_TRACE(::testing::Message() << "n=" << n << " msub=" << msub);
      hwyblas_int lvl = -1, nd = -1;
      std::vector<hwyblas_int> inode(n), ndiml(n), ndimr(n);
      HWYBLAS_CALL_REAL(R, lasdt, &n, &lvl, &nd, inode.data(), ndiml.data(),
                        ndimr.data(), &msub);
      ASSERT_EQ(nd, (hwyblas_int{1} << lvl) - 1);
      ASSERT_LE(nd, n);
      EXPECT_EQ(inode[0], n / 2 + 1);
      EXPECT_EQ(ndiml[0] + ndimr[0] + 1, n);
      for (hwyblas_int i = 1; i <= nd / 2; ++i) {
        const hwyblas_int l = 2 * i, r = 2 * i + 1;
        EXPECT_EQ(ndiml[l - 1] + ndimr[l - 1] + 1, ndiml[i - 1]) << i;
        EXPECT_EQ(ndiml[r - 1] + ndimr[r - 1] + 1, ndimr[i - 1]) << i;
        EXPECT_EQ(inode[l - 1] + ndimr[l - 1] + 1, inode[i - 1]) << i;
        EXPECT_EQ(inode[r - 1] - ndiml[r - 1] - 1, inode[i - 1]) << i;
      }
      for (hwyblas_int i = (nd + 1) / 2; i <= nd; ++i) {
        EXPECT_LE(ndiml[i - 1], msub) << i;
        EXPECT_LE(ndimr[i - 1], msub) << i;
      }
    }
  }
}

// xLASD4 (with xLASD5 for n = 2 and xLAED6): the roots sigma of the secular
// equation are the square roots of the eigenvalues of diag(d)**2 +
// rho * z * z**T (from xSYEV), with delta(j) = d(j) - sigma and
// work(j) = d(j) + sigma, also for d(1) = 0 and close d's.
TYPED_TEST(GesddRealTest, Lasd4) {
  using R = TypeParam;
  Random<R> rng(191);
  for (const int kind : {0, 1, 2}) {
    for (const hwyblas_int n : {1, 2, 3, 4, 9, 30}) {
      SCOPED_TRACE(::testing::Message() << "kind=" << kind << " n=" << n);
      std::vector<R> d = rng.Vector(n), z = rng.Vector(n);
      for (R& x : d) x = std::abs(x);
      std::sort(d.begin(), d.end());
      if (kind == 1) d[0] = 0;
      if (kind == 2) {
        for (hwyblas_int i = 1; i < n; ++i) d[i] = d[i - 1] + R(1e-3) * d[i];
      }
      for (hwyblas_int i = 1; i < n; ++i) {
        // Distinct poles, as xLASD4 requires.
        if (d[i] <= d[i - 1]) d[i] = d[i - 1] + R(0.01);
      }
      R znorm = 0;
      for (R x : z) znorm += x * x;
      for (R& x : z) x /= std::sqrt(znorm);
      const R rho = R(0.75);
      // The eigenvalues of diag(d)**2 + rho * z * z**T.
      std::vector<R> m(n * n, R(0)), w(n);
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < n; ++i) m[i + j * n] = rho * z[i] * z[j];
        m[j + j * n] += d[j] * d[j];
      }
      const char jobz = 'N', uplo = 'U';
      hwyblas_int lwork = 3 * n, info = -99;
      std::vector<R> work(lwork);
      HWYBLAS_CALL_REAL(R, syev, &jobz, &uplo, &n, m.data(), &n, w.data(),
                        work.data(), &lwork, &info);
      ASSERT_EQ(info, 0);
      for (hwyblas_int i = 1; i <= n; ++i) {
        std::vector<R> delta(n, NAN), wk(n, NAN);
        R sigma = NAN;
        HWYBLAS_CALL_REAL(R, lasd4, &n, &i, d.data(), z.data(), delta.data(),
                          &rho, &sigma, wk.data(), &info);
        ASSERT_EQ(info, 0) << i;
        EXPECT_NEAR(sigma * sigma, w[i - 1], 30 * n * Eps<R>() * w[n - 1]) << i;
        // (For n = 1, DELTA(1) = WORK(1) = 1.)
        for (hwyblas_int j = 0; j < n && n > 1; ++j) {
          EXPECT_NEAR(delta[j], d[j] - sigma, 30 * Eps<R>() * (d[j] + sigma))
              << i << " " << j;
          EXPECT_NEAR(wk[j], d[j] + sigma, 30 * Eps<R>() * (d[j] + sigma))
              << i << " " << j;
        }
      }
    }
  }
}

template <typename R>
hwyblas_int Bdsdc(char uplo, char compq, hwyblas_int n, std::vector<R>& d,
                  std::vector<R>& e, std::vector<R>& u, hwyblas_int ldu,
                  std::vector<R>& vt, hwyblas_int ldvt, std::vector<R>& q,
                  std::vector<hwyblas_int>& iq) {
  std::vector<R> work(std::max<hwyblas_int>(1, 3 * n * n + 4 * n), NAN);
  std::vector<hwyblas_int> iwork(std::max<hwyblas_int>(1, 8 * n), -1);
  hwyblas_int info = -99;
  HWYBLAS_CALL_REAL(R, bdsdc, &uplo, &compq, &n, d.data(), e.data(), u.data(),
                    &ldu, vt.data(), &ldvt, q.data(), iq.data(), work.data(),
                    iwork.data(), &info);
  return info;
}

// xBDSDC: B = U * S * VT for upper and lower B with orthonormal U and VT
// (COMPQ = 'I'), singular values in decreasing order, and the same singular
// values without vectors ('N', with xLASDQ) and with the vectors in compact
// form ('P', with xLASDA), for matrices that are split into several levels
// of subproblems (SMLSIZ = 3) or not.
TYPED_TEST(GesddRealTest, Bdsdc) {
  using R = TypeParam;
  Random<R> rng(193);
  for (const hwyblas_int smlsiz : {0, 3}) {
    const ScopedSubproblemSize subproblems(smlsiz);
    for (const int kind : {0, 1, 2, 3, 4, 5, 6}) {
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {1, 2, 3, 4, 10, 26, 41, 90}) {
          SCOPED_TRACE(::testing::Message()
                       << "smlsiz=" << smlsiz << " kind=" << kind
                       << " uplo=" << uplo << " n=" << n);
          std::vector<R> d0, e0;
          TestBidiagonal(kind, n, rng, d0, e0);
          R bnorm = 0;
          for (R x : d0) bnorm = std::max(bnorm, std::abs(x));
          for (R x : e0) bnorm = std::max(bnorm, std::abs(x));
          std::vector<R> ds = d0, es = e0;
          if (bnorm > 0) {
            for (auto& x : ds) x /= bnorm;
            for (auto& x : es) x /= bnorm;
          }
          const Dense<R> bd = Bidiagonal<R>(ds, es, uplo == 'U');

          const hwyblas_int ld = n + 2;
          std::vector<R> d = d0, e = e0, u(ld * n, NAN), vt(ld * n, NAN);
          std::vector<R> q(1, NAN);
          std::vector<hwyblas_int> iq(1, -1);
          ASSERT_EQ(Bdsdc<R>(uplo, 'I', n, d, e, u, ld, vt, ld, q, iq), 0);
          ExpectDecreasing(d);
          std::vector<R> sv = d;
          if (bnorm > 0) {
            for (auto& x : sv) x /= bnorm;
          }
          const Dense<R> ud = ToDense(u, n, n, ld), vtd = ToDense(vt, n, n, ld);
          EXPECT_LT(OrthogonalityRatio(ud), kLapackThreshold);
          EXPECT_LT(OrthogonalityRatio(Adjoint(vtd)), kLapackThreshold);
          EXPECT_LT(ResidualRatio(bd, Mul(Mul(ud, Diagonal<R>(sv, n)), vtd), n),
                    kLapackThreshold);

          // Values only, and the compact form.
          std::vector<R> dn = d0, en = e0, dummy(1, NAN);
          ASSERT_EQ(Bdsdc<R>(uplo, 'N', n, dn, en, dummy, 1, dummy, 1, q, iq),
                    0);
          ExpectDecreasing(dn);
          std::vector<R> dp = d0, ep = e0;
          // The sizes of Q and IQ that xBDSDC uses (larger than those it
          // documents) with MLVL levels, and for n <= SMLSIZ (U and VT in Q)
          // 2 * n**2 + 5 * n.
          const hwyblas_int sml = smlsiz > 0 ? smlsiz : 25;
          const hwyblas_int mlvl = std::max<hwyblas_int>(
              1, static_cast<hwyblas_int>(
                     std::log(static_cast<double>(n) / (sml + 1)) /
                     std::log(2.0)) +
                     1);
          std::vector<R> qp(
              std::max(n * (7 + 2 * sml + 8 * mlvl), 2 * n * n + 5 * n), NAN);
          std::vector<hwyblas_int> iqp(static_cast<size_t>(n) * (3 + 3 * mlvl),
                                       -1);
          ASSERT_EQ(Bdsdc<R>(uplo, 'P', n, dp, ep, dummy, 1, dummy, 1, qp, iqp),
                    0);
          ExpectDecreasing(dp);
          for (hwyblas_int i = 0; i < n; ++i) {
            EXPECT_NEAR(dn[i], d[i], 30 * n * Eps<R>() * bnorm) << i;
            EXPECT_NEAR(dp[i], d[i], 30 * n * Eps<R>() * bnorm) << i;
          }
          // IQ(N) = 1 for upper and 0 for lower B ('P').
          if (n > 1) EXPECT_EQ(iqp[n - 1], uplo == 'U' ? 1 : 0);
        }
      }
    }
  }
}

// xLASDQ: the SVD of the n x (n+sqre) upper and the (n+sqre) x n lower
// bidiagonal matrix, with the singular values in ascending order, and the
// rotations applied to VT, U and C.
TYPED_TEST(GesddRealTest, Lasdq) {
  using R = TypeParam;
  Random<R> rng(197);
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int sqre : {0, 1}) {
      for (const hwyblas_int n : {1, 2, 5, 17}) {
        SCOPED_TRACE(::testing::Message()
                     << "uplo=" << uplo << " sqre=" << sqre << " n=" << n);
        const hwyblas_int m = n + sqre;  // B is n x m (upper) or m x n
        std::vector<R> d0 = rng.Vector(n), e0 = rng.Vector(n);
        // Dense B.
        const hwyblas_int rows = uplo == 'U' ? n : m,
                          cols = uplo == 'U' ? m : n;
        Dense<R> bd(rows, cols);
        for (hwyblas_int i = 0; i < n; ++i) {
          bd(i, i) = d0[i];
          if (uplo == 'U' && i + 1 < m) bd(i, i + 1) = e0[i];
          if (uplo == 'L' && i + 1 < m) bd(i + 1, i) = e0[i];
        }
        // VT (cols x cols), U (rows x rows) and C (rows x 3) start as I and
        // a random C; VT := P**T * VT, U := U * Q, C := Q**T * C.
        const hwyblas_int ldvt = cols + 1, ldu = rows + 1, ldc = rows + 2;
        std::vector<R> vt(ldvt * cols, R(0)), u(ldu * rows, R(0));
        for (hwyblas_int i = 0; i < cols; ++i) vt[i + i * ldvt] = 1;
        for (hwyblas_int i = 0; i < rows; ++i) u[i + i * ldu] = 1;
        const std::vector<R> c0 = rng.Vector(ldc * 3);
        std::vector<R> c = c0, d = d0, e = e0, work(4 * m, NAN);
        const hwyblas_int ncc = 3;
        hwyblas_int info = -99;
        HWYBLAS_CALL_REAL(R, lasdq, &uplo, &sqre, &n, &cols, &rows, &ncc,
                          d.data(), e.data(), vt.data(), &ldvt, u.data(), &ldu,
                          c.data(), &ldc, work.data(), &info);
        ASSERT_EQ(info, 0);
        for (hwyblas_int i = 0; i < n; ++i) {
          ASSERT_GE(d[i], R(0));
          if (i > 0) ASSERT_LE(d[i - 1], d[i]) << i;
        }
        // B = U(:, 1:n) * diag(d) * VT(1:n, :) (the first n columns and
        // rows).
        const Dense<R> ud = ToDense(u, rows, rows, ldu);
        const Dense<R> vtd = ToDense(vt, cols, cols, ldvt);
        EXPECT_LT(OrthogonalityRatio(ud), kLapackThreshold);
        EXPECT_LT(OrthogonalityRatio(Adjoint(vtd)), kLapackThreshold);
        Dense<R> usv(rows, cols);
        for (hwyblas_int j = 0; j < cols; ++j) {
          for (hwyblas_int i = 0; i < rows; ++i) {
            double sum = 0;
            for (hwyblas_int k = 0; k < n; ++k)
              sum += ud(i, k) * d[k] * vtd(k, j);
            usv(i, j) = static_cast<R>(sum);
          }
        }
        EXPECT_LT(ResidualRatio(bd, usv, m), kLapackThreshold);
        // C = U**T * C0.
        EXPECT_LT(ResidualRatio(Mul(Adjoint(ud), ToDense(c0, rows, 3, ldc)),
                                ToDense(c, rows, 3, ldc), rows),
                  kLapackThreshold);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// xGESDD

template <typename T>
hwyblas_int MinimumGesddLwork(char jobz, hwyblas_int m, hwyblas_int n) {
  const hwyblas_int mn = std::min(m, n), mx = std::max(m, n);
  if (mn == 0) return 1;
  if (IsComplex<T>::value) {
    switch (jobz) {
      case 'N':
        return 2 * mn + mx;
      case 'O':
        return 2 * mn * mn + 2 * mn + mx;
      case 'S':
        return mn * mn + 3 * mn;
      default:
        return mn * mn + 2 * mn + mx;
    }
  }
  switch (jobz) {
    case 'N':
      return 3 * mn + std::max(mx, 7 * mn);
    case 'O':
      return 3 * mn + std::max(mx, 5 * mn * mn + 4 * mn);
    case 'S':
      return 4 * mn * mn + 7 * mn;
    default:
      return 4 * mn * mn + 6 * mn + mx;
  }
}

// The documented minimum of RWORK (complex T).
inline hwyblas_int MinimumGesddRwork(char jobz, hwyblas_int m, hwyblas_int n) {
  const hwyblas_int mn = std::min(m, n), mx = std::max(m, n);
  if (jobz == 'N') return std::max<hwyblas_int>(1, 5 * mn);
  return std::max<hwyblas_int>(
      {1, 5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn});
}

// The result of xGESDD.
template <typename T>
struct Sdd {
  std::vector<T> a, u, vt, work;
  std::vector<RealT<T>> s, rwork;
  std::vector<hwyblas_int> iwork;
  hwyblas_int info = -99;
};

// xGESDD of the m x n matrix a (leading dimension lda); lwork = -1 uses the
// optimal workspace from a query.
template <typename T>
Sdd<T> Gesdd(char jobz, hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
             hwyblas_int lda, hwyblas_int ldu, hwyblas_int ldvt,
             hwyblas_int lwork) {
  Sdd<T> r;
  r.a = a;
  const hwyblas_int mn = std::min(m, n);
  r.s.assign(std::max<hwyblas_int>(1, mn), NAN);
  r.u.assign(std::max<hwyblas_int>(1, ldu * m), NaN<T>());
  r.vt.assign(std::max<hwyblas_int>(1, ldvt * n), NaN<T>());
  r.rwork.assign(MinimumGesddRwork(jobz, m, n), NAN);
  r.iwork.assign(std::max<hwyblas_int>(1, 8 * mn), -1);
  const auto call = [&](T* work, hwyblas_int lw) {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gesdd, &jobz, &m, &n, P(r.a.data()), &lda,
                           r.s.data(), P(r.u.data()), &ldu, P(r.vt.data()),
                           &ldvt, P(work), &lw, r.rwork.data(), r.iwork.data(),
                           &r.info);
    } else {
      HWYBLAS_CALL_REAL(T, gesdd, &jobz, &m, &n, P(r.a.data()), &lda,
                        r.s.data(), P(r.u.data()), &ldu, P(r.vt.data()), &ldvt,
                        P(work), &lw, r.iwork.data(), &r.info);
    }
  };
  if (lwork == -1) {
    T query = NaN<T>();
    call(&query, -1);
    if (r.info != 0) return r;
    lwork = static_cast<hwyblas_int>(std::real(query));
  }
  r.work.assign(std::max<hwyblas_int>(1, lwork), NaN<T>());
  call(r.work.data(), lwork);
  return r;
}

// The smallest LWORK that xGESDD accepts (at most the documented minimum).
template <typename T>
hwyblas_int AcceptedGesddLwork(char jobz, hwyblas_int m, hwyblas_int n,
                               const std::vector<T>& a, hwyblas_int lda) {
  ErrorCapture errors;
  const auto accepted = [&](hwyblas_int lwork) {
    // (A call that is accepted computes the SVD.)
    return Gesdd<T>(jobz, m, n, a, lda, m, n, lwork).info != -12;
  };
  hwyblas_int lo = 1, hi = MinimumGesddLwork<T>(jobz, m, n);
  EXPECT_TRUE(accepted(hi));
  while (lo < hi) {
    const hwyblas_int mid = lo + (hi - lo) / 2;
    if (accepted(mid)) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  errors.Take();
  return lo;
}

// xGESDD for all JOBZ, matrices of all paths (m much larger than n, m
// larger, m = n, m smaller, m much smaller; the crossovers are 11/6 *
// min(m, n) for real T and 17/9 and 5/3 * min(m, n) for complex T), with
// several levels of divide and conquer (SMLSIZ = 3) or none, and
// workspaces from the smallest accepted one to the optimal one: A = U * S *
// VT with orthonormal U and VT, decreasing S, and the singular values and
// vectors (up to unit factors) of xGESVD.
TYPED_TEST(GesddTest, Gesdd) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(199);
    for (const hwyblas_int smlsiz : {0, 3}) {
      const ScopedSubproblemSize subproblems(smlsiz);
      for (const Shape2 sh : std::vector<Shape2>{{1, 1},
                                                 {1, 6},
                                                 {6, 1},
                                                 {12, 12},
                                                 {15, 11},
                                                 {18, 11},
                                                 {20, 11},
                                                 {40, 11},
                                                 {11, 15},
                                                 {11, 18},
                                                 {11, 20},
                                                 {11, 40},
                                                 {57, 31},
                                                 {31, 57}}) {
        const hwyblas_int m = sh.m, n = sh.n, mn = std::min(m, n);
        const hwyblas_int lda = m + 2;
        const std::vector<T> a = rng.Vector(lda * n);
        const Dense<T> ad = ToDense(a, m, n, lda);
        // The reference: xGESVD's singular values and vectors.
        std::vector<T> ar = a, uref_v(m * m), vref_v(n * n);
        std::vector<R> sref(mn), rwork(5 * mn + 1);
        {
          const char job = 'A';
          hwyblas_int lwork = -1, info = -99;
          T query;
          const auto gesvd = [&](T* work) {
            if constexpr (IsComplex<T>::value) {
              HWYBLAS_CALL_COMPLEX(T, gesvd, &job, &job, &m, &n, P(ar.data()),
                                   &lda, sref.data(), P(uref_v.data()), &m,
                                   P(vref_v.data()), &n, P(work), &lwork,
                                   rwork.data(), &info);
            } else {
              HWYBLAS_CALL_REAL(T, gesvd, &job, &job, &m, &n, P(ar.data()),
                                &lda, sref.data(), P(uref_v.data()), &m,
                                P(vref_v.data()), &n, P(work), &lwork, &info);
            }
          };
          gesvd(&query);
          lwork = static_cast<hwyblas_int>(std::real(query));
          std::vector<T> work(lwork);
          gesvd(work.data());
          ASSERT_EQ(info, 0);
        }
        const Dense<T> uref = ToDense(uref_v, m, mn, m);
        const Dense<T> vref = Adjoint(ToDense(vref_v, mn, n, n));
        for (const char jobz : {'A', 'S', 'O', 'N'}) {
          const hwyblas_int smallest =
              AcceptedGesddLwork<T>(jobz, m, n, a, lda);
          const hwyblas_int documented = MinimumGesddLwork<T>(jobz, m, n);
          const Sdd<T> opt = Gesdd<T>(jobz, m, n, a, lda, m + 1, n + 1, -1);
          ASSERT_EQ(opt.info, 0);
          const hwyblas_int optimal = static_cast<hwyblas_int>(opt.work.size());
          for (const hwyblas_int lwork :
               {smallest, documented, (smallest + optimal) / 2,
                hwyblas_int{-1}}) {
            SCOPED_TRACE(::testing::Message()
                         << "smlsiz=" << smlsiz << " m=" << m << " n=" << n
                         << " jobz=" << jobz << " lwork=" << lwork);
            const hwyblas_int ldu = m + 1, ldvt = n + 1;
            const Sdd<T> r = Gesdd<T>(jobz, m, n, a, lda, ldu, ldvt, lwork);
            ASSERT_EQ(r.info, 0);
            std::vector<R> s(r.s.begin(), r.s.begin() + mn);
            ExpectDecreasing(s);
            for (hwyblas_int i = 0; i < mn; ++i) {
              EXPECT_NEAR(s[i], sref[i], 30 * std::max(m, n) * Eps<T>()) << i;
            }
            if (jobz == 'N') continue;
            // The vectors: in U and VT, or one of them in A ('O').
            Dense<T> ud(0, 0), vd(0, 0);
            const bool u_in_a = jobz == 'O' && m >= n;
            const bool vt_in_a = jobz == 'O' && m < n;
            if (u_in_a) {
              ud = ToDense(r.a, m, n, lda);
            } else {
              const hwyblas_int ucols = jobz == 'S' ? mn : m;
              const Dense<T> uf = ToDense(r.u, m, ucols, ldu);
              EXPECT_LT(OrthogonalityRatio(uf), kLapackThreshold);
              ud = Columns(uf, mn);
            }
            if (vt_in_a) {
              vd = Adjoint(ToDense(r.a, m, n, lda));
            } else {
              const hwyblas_int vrows = jobz == 'S' ? mn : n;
              const Dense<T> vf = Adjoint(ToDense(r.vt, vrows, n, ldvt));
              EXPECT_LT(OrthogonalityRatio(vf), kLapackThreshold);
              vd = Columns(vf, mn);
            }
            EXPECT_LT(OrthogonalityRatio(ud), kLapackThreshold);
            EXPECT_LT(OrthogonalityRatio(vd), kLapackThreshold);
            EXPECT_LT(VectorsRatio(ud, uref, sref), kLapackThreshold);
            EXPECT_LT(VectorsRatio(vd, vref, sref), kLapackThreshold);
            const Dense<T> usv = Mul(Mul(ud, Diagonal<T>(s, mn)), Adjoint(vd));
            EXPECT_LT(ResidualRatio(ad, usv, std::max(m, n)), kLapackThreshold);
          }
        }
      }
    }
  });
}

// xGESDD of matrices that it scales (norm near the underflow or overflow
// threshold), of special matrices (zero, rank one, empty), and of a matrix
// with a NaN (INFO = -4, without an error message).
TYPED_TEST(GesddTest, GesddScalingAndSpecialMatrices) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(211);
  const hwyblas_int m = 15, n = 9;
  const std::vector<T> a = rng.Vector(m * n);
  const Sdd<T> ref = Gesdd<T>('S', m, n, a, m, m, n, -1);
  ASSERT_EQ(ref.info, 0);
  const R tiny = std::numeric_limits<R>::min() * 16;
  const R huge = std::numeric_limits<R>::max() / 64;
  for (const R scale : {tiny, huge}) {
    SCOPED_TRACE(::testing::Message() << "scale=" << scale);
    std::vector<T> as = a;
    for (T& x : as) x *= scale;
    const Sdd<T> r = Gesdd<T>('S', m, n, as, m, m, n, -1);
    ASSERT_EQ(r.info, 0);
    for (hwyblas_int i = 0; i < n; ++i) {
      EXPECT_NEAR(r.s[i] / scale, ref.s[i], 30 * m * Eps<T>()) << i;
    }
    EXPECT_LT(
        VectorsRatio(ToDense(r.u, m, n, m), ToDense(ref.u, m, n, m), ref.s),
        kLapackThreshold);
  }
  // Zero matrix: zero singular values, U and VT orthonormal.
  const Sdd<T> z =
      Gesdd<T>('A', m, n, std::vector<T>(m * n, T(0)), m, m, n, -1);
  ASSERT_EQ(z.info, 0);
  for (hwyblas_int i = 0; i < n; ++i) EXPECT_EQ(z.s[i], R(0));
  EXPECT_LT(OrthogonalityRatio(ToDense(z.u, m, m, m)), kLapackThreshold);
  EXPECT_LT(OrthogonalityRatio(ToDense(z.vt, n, n, n)), kLapackThreshold);
  // Rank one: x * y**H has the single singular value ||x|| ||y||.
  std::vector<T> x = rng.Vector(m), y = rng.Vector(n), r1(m * n);
  double nx = 0, ny = 0;
  for (const T& v : x) nx += std::norm(v);
  for (const T& v : y) ny += std::norm(v);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) r1[i + j * m] = x[i] * Conj(y[j]);
  }
  const Sdd<T> one = Gesdd<T>('N', m, n, r1, m, 1, 1, -1);
  ASSERT_EQ(one.info, 0);
  EXPECT_NEAR(one.s[0], std::sqrt(nx * ny), 30 * m * Eps<T>());
  for (hwyblas_int i = 1; i < n; ++i) {
    EXPECT_LT(one.s[i], 30 * m * Eps<T>()) << i;
  }
  // Empty matrices.
  EXPECT_EQ(Gesdd<T>('A', 0, 3, a, 1, 1, 3, -1).info, 0);
  EXPECT_EQ(Gesdd<T>('A', 3, 0, a, 3, 3, 1, -1).info, 0);
  // A NaN.
  ErrorCapture errors;
  std::vector<T> an = a;
  an[7] = NaN<T>();
  EXPECT_EQ(Gesdd<T>('S', m, n, an, m, m, n, -1).info, -4);
  EXPECT_TRUE(errors.Take().empty());
}

TYPED_TEST(GesddTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  const hwyblas_int m = 10, n = 6;
  std::vector<T> a(m * n, T(1));
  // The query returns at least the minimum, and 1 for empty matrices.
  for (const char jobz : {'N', 'O', 'S', 'A'}) {
    const Sdd<T> q = Gesdd<T>(jobz, m, n, a, m, m, n, -1);
    EXPECT_EQ(q.info, 0);
    EXPECT_GE(static_cast<hwyblas_int>(q.work.size()),
              AcceptedGesddLwork<T>(jobz, m, n, a, m));
    const Sdd<T> qt = Gesdd<T>(jobz, n, m, a, n, n, m, -1);
    EXPECT_EQ(qt.info, 0);
    EXPECT_EQ(Gesdd<T>(jobz, 0, n, a, 1, 1, n, -1).work.size(), 1u);
  }

  // Invalid arguments.
  ErrorCapture errors;
  const std::string sdd = Name<T>("GESDD");
  const hwyblas_int mw = MinimumGesddLwork<T>('A', m, n);
  EXPECT_EQ(Gesdd<T>('X', m, n, a, m, m, n, mw).info, -1);
  EXPECT_EQ(Gesdd<T>('N', -1, n, a, m, m, n, mw).info, -2);
  EXPECT_EQ(Gesdd<T>('N', m, -1, a, m, m, n, mw).info, -3);
  EXPECT_EQ(Gesdd<T>('N', m, n, a, m - 1, m, n, mw).info, -5);
  EXPECT_EQ(Gesdd<T>('N', m, n, a, m, 0, n, mw).info, -8);
  EXPECT_EQ(Gesdd<T>('S', m, n, a, m, m - 1, n, mw).info, -8);
  EXPECT_EQ(Gesdd<T>('O', n, m, a, n, n - 1, m, mw).info, -8);
  EXPECT_EQ(Gesdd<T>('N', m, n, a, m, m, 0, mw).info, -10);
  EXPECT_EQ(Gesdd<T>('A', m, n, a, m, m, n - 1, mw).info, -10);
  EXPECT_EQ(Gesdd<T>('S', n, m, a, n, n, n - 1, mw).info, -10);
  EXPECT_EQ(Gesdd<T>('O', m, n, a, m, m, n - 1, mw).info, -10);
  EXPECT_EQ(Gesdd<T>('N', m, n, a, m, m, n, 1).info, -12);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{sdd + ":1", sdd + ":2", sdd + ":3",
                                      sdd + ":5", sdd + ":8", sdd + ":8",
                                      sdd + ":8", sdd + ":10", sdd + ":10",
                                      sdd + ":10", sdd + ":10", sdd + ":12"}));
}

TYPED_TEST(GesddRealTest, BdsdcErrors) {
  using R = TypeParam;
  const hwyblas_int n = 5;
  std::vector<R> d(n, 1), e(n, 0), u(n * n), vt(n * n), q(1);
  std::vector<hwyblas_int> iq(1);
  ErrorCapture errors;
  EXPECT_EQ(Bdsdc<R>('X', 'N', n, d, e, u, n, vt, n, q, iq), -1);
  EXPECT_EQ(Bdsdc<R>('U', 'X', n, d, e, u, n, vt, n, q, iq), -2);
  EXPECT_EQ(Bdsdc<R>('U', 'N', -1, d, e, u, n, vt, n, q, iq), -3);
  EXPECT_EQ(Bdsdc<R>('U', 'N', n, d, e, u, 0, vt, n, q, iq), -7);
  EXPECT_EQ(Bdsdc<R>('U', 'I', n, d, e, u, n - 1, vt, n, q, iq), -7);
  EXPECT_EQ(Bdsdc<R>('U', 'N', n, d, e, u, n, vt, 0, q, iq), -9);
  EXPECT_EQ(Bdsdc<R>('U', 'I', n, d, e, u, n, vt, n - 1, q, iq), -9);
  const std::string bdc = Name<R>("BDSDC");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               bdc + ":1", bdc + ":2", bdc + ":3", bdc + ":7",
                               bdc + ":7", bdc + ":9", bdc + ":9"}));
  // n = 0 and 1.
  EXPECT_EQ(Bdsdc<R>('U', 'I', 0, d, e, u, 1, vt, 1, q, iq), 0);
  d[0] = -2;
  EXPECT_EQ(Bdsdc<R>('L', 'I', 1, d, e, u, 1, vt, 1, q, iq), 0);
  EXPECT_EQ(d[0], R(2));
  EXPECT_EQ(u[0], R(-1));
  EXPECT_EQ(vt[0], R(1));
}

// ---------------------------------------------------------------------------
// Complex matrices with real ones

// xLACP2 (all, upper and lower parts), xLACRM and xLARCM: the products of
// the real and complex matrices.
TYPED_TEST(GesddComplexTest, RealComplexProducts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(223);
  Random<R> rrng(227);
  const hwyblas_int m = 7, n = 5, lda = m + 1;
  const std::vector<R> ar = rrng.Vector(lda * n);
  for (const char uplo : {'A', 'U', 'L'}) {
    std::vector<T> b(m * n, T(9));
    HWYBLAS_CALL_COMPLEX(T, lacp2, &uplo, &m, &n, ar.data(), &lda, P(b.data()),
                         &m);
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = 0; i < m; ++i) {
        const bool in = uplo == 'U' ? i <= j : (uplo == 'L' ? i >= j : true);
        EXPECT_EQ(b[i + j * m], in ? T(ar[i + j * lda]) : T(9))
            << i << " " << j;
      }
    }
  }
  // xLACRM: C (m x n) = A (complex m x n) * B (real n x n).
  const std::vector<T> ac = rng.Vector(m * n);
  const std::vector<R> br = rrng.Vector(n * n);
  std::vector<T> c(m * n, NaN<T>());
  std::vector<R> rwork(2 * m * n, NAN);
  HWYBLAS_CALL_COMPLEX(T, lacrm, &m, &n, P(ac.data()), &m, br.data(), &n,
                       P(c.data()), &m, rwork.data());
  Dense<T> bd(n, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) bd(i, j) = br[i + j * n];
  }
  EXPECT_LT(
      ResidualRatio(Mul(ToDense(ac, m, n, m), bd), ToDense(c, m, n, m), n),
      kLapackThreshold);
  // xLARCM: C (m x n) = A (real m x m) * B (complex m x n).
  const std::vector<R> am = rrng.Vector(m * m);
  std::vector<T> c2(m * n, NaN<T>());
  HWYBLAS_CALL_COMPLEX(T, larcm, &m, &n, am.data(), &m, P(ac.data()), &m,
                       P(c2.data()), &m, rwork.data());
  Dense<T> amd(m, m);
  for (hwyblas_int j = 0; j < m; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) amd(i, j) = am[i + j * m];
  }
  EXPECT_LT(
      ResidualRatio(Mul(amd, ToDense(ac, m, n, m)), ToDense(c2, m, n, m), m),
      kLapackThreshold);
}

// ---------------------------------------------------------------------------
// LAPACKE

// LAPACKE_xlacp2, xlacrm and xlarcm in row-major layout give the results of
// column-major layout (xlacp2 keeps B outside the triangle), with their
// errors and NaN checks.
TYPED_TEST(GesddComplexTest, LapackeRealComplexProducts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(239);
  Random<R> rrng(241);
  const lapack_int m = 6, n = 4;
  const std::vector<R> a = rrng.Vector(m * n), sq = rrng.Vector(m * m),
                       bn = rrng.Vector(n * n);
  const std::vector<T> b0 = rng.Vector(m * n);
  for (const char uplo : {'A', 'U', 'L'}) {
    std::vector<T> bc = b0, br = ToRowMajor(b0, m, n, m, n + 1);
    ASSERT_EQ(LAPACKE_CALL(T, lacp2, lacp2, LAPACK_COL_MAJOR, uplo, m, n,
                           a.data(), m, bc.data(), m),
              0);
    ASSERT_EQ(
        LAPACKE_CALL(T, lacp2, lacp2, LAPACK_ROW_MAJOR, uplo, m, n,
                     ToRowMajor(a, m, n, m, n).data(), n, br.data(), n + 1),
        0);
    ExpectSameRowMajor(br, bc, m, n, n + 1, m);
  }
  std::vector<T> cc(m * n), cr((m + 1) * (n + 1), NaN<T>());
  ASSERT_EQ(LAPACKE_CALL(T, lacrm, lacrm, LAPACK_COL_MAJOR, m, n, b0.data(), m,
                         bn.data(), n, cc.data(), m),
            0);
  ASSERT_EQ(
      LAPACKE_CALL(T, lacrm, lacrm, LAPACK_ROW_MAJOR, m, n,
                   ToRowMajor(b0, m, n, m, n).data(), n,
                   ToRowMajor(bn, n, n, n, n).data(), n, cr.data(), n + 1),
      0);
  ExpectSameRowMajor(cr, cc, m, n, n + 1, m);
  ASSERT_EQ(LAPACKE_CALL(T, larcm, larcm, LAPACK_COL_MAJOR, m, n, sq.data(), m,
                         b0.data(), m, cc.data(), m),
            0);
  ASSERT_EQ(
      LAPACKE_CALL(T, larcm, larcm, LAPACK_ROW_MAJOR, m, n,
                   ToRowMajor(sq, m, m, m, m).data(), m,
                   ToRowMajor(b0, m, n, m, n).data(), n, cr.data(), n + 1),
      0);
  ExpectSameRowMajor(cr, cc, m, n, n + 1, m);

  // Errors and NaN checks.
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(
      LAPACKE_CALL(T, lacp2, lacp2, 0, 'A', m, n, a.data(), m, cc.data(), m),
      -1);
  EXPECT_EQ(LAPACKE_CALL(T, lacp2, lacp2, LAPACK_ROW_MAJOR, 'A', m, n, a.data(),
                         n - 1, cc.data(), n),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, lacp2, lacp2, LAPACK_ROW_MAJOR, 'A', m, n, a.data(),
                         n, cc.data(), n - 1),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, lacrm, lacrm, LAPACK_ROW_MAJOR, m, n, b0.data(),
                         n - 1, bn.data(), n, cc.data(), n),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, lacrm, lacrm, LAPACK_ROW_MAJOR, m, n, b0.data(), n,
                         bn.data(), n - 1, cc.data(), n),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, lacrm, lacrm, LAPACK_ROW_MAJOR, m, n, b0.data(), n,
                         bn.data(), n, cc.data(), n - 1),
            -9);
  EXPECT_EQ(LAPACKE_CALL(T, larcm, larcm, LAPACK_ROW_MAJOR, m, n, sq.data(),
                         m - 1, b0.data(), n, cc.data(), n),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, larcm, larcm, 0, m, n, sq.data(), m, b0.data(), m,
                         cc.data(), m),
            -1);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("lacp2") +
                "\nWrong parameter 6 in " + LapackeName<T>("lacp2_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("lacp2_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("lacrm_work") +
                "\nWrong parameter 7 in " + LapackeName<T>("lacrm_work") +
                "\nWrong parameter 9 in " + LapackeName<T>("lacrm_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("larcm_work") +
                "\nWrong parameter 1 in " + LapackeName<T>("larcm") + "\n");
  std::vector<R> an = a;
  an[3] = NAN;
  std::vector<T> b1 = b0;
  b1[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, lacp2, lacp2, LAPACK_COL_MAJOR, 'A', m, n,
                         an.data(), m, cc.data(), m),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, lacrm, lacrm, LAPACK_COL_MAJOR, m, n, b1.data(), m,
                         bn.data(), n, cc.data(), m),
            -4);
  EXPECT_EQ(LAPACKE_CALL(T, lacrm, lacrm, LAPACK_COL_MAJOR, m, n, b0.data(), m,
                         an.data(), n, cc.data(), m),
            -6);
  std::vector<R> sqn = sq;
  sqn[3] = NAN;
  EXPECT_EQ(LAPACKE_CALL(T, larcm, larcm, LAPACK_COL_MAJOR, m, n, sqn.data(), m,
                         b0.data(), m, cc.data(), m),
            -4);
  EXPECT_EQ(LAPACKE_CALL(T, larcm, larcm, LAPACK_COL_MAJOR, m, n, sq.data(), m,
                         b1.data(), m, cc.data(), m),
            -6);
}

template <typename T>
lapack_int LapackeGesdd(int layout, char jobz, lapack_int m, lapack_int n, T* a,
                        lapack_int lda, RealT<T>* s, T* u, lapack_int ldu,
                        T* vt, lapack_int ldvt) {
  return LAPACKE_CALL(T, gesdd, gesdd, layout, jobz, m, n, a, lda, s, u, ldu,
                      vt, ldvt);
}

template <typename R>
lapack_int LapackeBdsdc(int layout, char uplo, char compq, lapack_int n, R* d,
                        R* e, R* u, lapack_int ldu, R* vt, lapack_int ldvt,
                        R* q, lapack_int* iq) {
  if constexpr (std::is_same_v<R, float>) {
    return LAPACKE_sbdsdc(layout, uplo, compq, n, d, e, u, ldu, vt, ldvt, q,
                          iq);
  } else {
    return LAPACKE_dbdsdc(layout, uplo, compq, n, d, e, u, ldu, vt, ldvt, q,
                          iq);
  }
}

// LAPACKE_xgesdd and xbdsdc in row-major layout give the results of
// column-major layout.
TYPED_TEST(GesddTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(229);
  for (const Shape2 sh : std::vector<Shape2>{{13, 7}, {7, 13}, {30, 6}}) {
    const lapack_int m = sh.m, n = sh.n, mn = std::min(m, n);
    const lapack_int lda = m + 1, ldr = n + 2;
    const std::vector<T> a = rng.Vector(lda * n);
    for (const char jobz : {'A', 'S', 'O', 'N'}) {
      SCOPED_TRACE(::testing::Message()
                   << "m=" << m << " n=" << n << " " << jobz);
      std::vector<T> ac = a, ar = ToRowMajor(a, m, n, lda, ldr);
      std::vector<T> uc(m * m, NaN<T>()), ur(m * (m + 1), NaN<T>());
      std::vector<T> vc(n * n, NaN<T>()), vr(n * (n + 1), NaN<T>());
      std::vector<R> sc(mn), sr(mn);
      ASSERT_EQ(LapackeGesdd(LAPACK_COL_MAJOR, jobz, m, n, ac.data(), lda,
                             sc.data(), uc.data(), m, vc.data(), n),
                0);
      ASSERT_EQ(LapackeGesdd(LAPACK_ROW_MAJOR, jobz, m, n, ar.data(), ldr,
                             sr.data(), ur.data(), m + 1, vr.data(), n + 1),
                0);
      EXPECT_EQ(sr, sc);
      const bool u_out = jobz == 'A' || jobz == 'S' || (jobz == 'O' && m < n);
      const bool vt_out = jobz == 'A' || jobz == 'S' || (jobz == 'O' && m >= n);
      const lapack_int ucols = (jobz == 'S') ? mn : m;
      const lapack_int vrows = (jobz == 'S') ? mn : n;
      if (u_out) ExpectSameRowMajor(ur, uc, m, ucols, m + 1, m);
      if (vt_out) ExpectSameRowMajor(vr, vc, vrows, n, n + 1, n);
      ExpectSameRowMajor(ar, ac, m, n, ldr, lda);
    }
  }
  if constexpr (!IsComplex<T>::value) {
    // xBDSDC: U and VT (COMPQ = 'I') are transposed, Q and IQ ('P') are
    // not.
    for (const char uplo : {'U', 'L'}) {
      for (const char compq : {'I', 'N', 'P'}) {
        SCOPED_TRACE(::testing::Message() << uplo << compq);
        const lapack_int n = 30;
        Random<R> rrng(233);
        const std::vector<R> d0 = rrng.Vector(n), e0 = rrng.Vector(n);
        std::vector<R> d1 = d0, e1 = e0, d2 = d0, e2 = e0;
        std::vector<R> u1(n * n, NAN), v1(n * n, NAN), u2(n * (n + 1), NAN),
            v2(n * (n + 1), NAN);
        std::vector<R> q1(n * (n + 40), NAN), q2 = q1;
        std::vector<lapack_int> iq1(static_cast<size_t>(n) * 10, -1), iq2 = iq1;
        ASSERT_EQ(LapackeBdsdc<R>(LAPACK_COL_MAJOR, uplo, compq, n, d1.data(),
                                  e1.data(), u1.data(), n, v1.data(), n,
                                  q1.data(), iq1.data()),
                  0);
        ASSERT_EQ(LapackeBdsdc<R>(LAPACK_ROW_MAJOR, uplo, compq, n, d2.data(),
                                  e2.data(), u2.data(), n + 1, v2.data(), n + 1,
                                  q2.data(), iq2.data()),
                  0);
        EXPECT_EQ(d1, d2);
        if (compq == 'I') {
          ExpectSameRowMajor(u2, u1, n, n, n + 1, n);
          ExpectSameRowMajor(v2, v1, n, n, n + 1, n);
        }
        if (compq == 'P') {
          EXPECT_EQ(iq1, iq2);
          for (size_t i = 0; i < q1.size(); ++i) {
            if (!std::isnan(q1[i]) || !std::isnan(q2[i])) {
              ASSERT_EQ(q1[i], q2[i]) << i;
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(GesddTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), u(16), vt(16);
  std::vector<R> s(4);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeGesdd<T>(0, 'A', 4, 4, a.data(), 4, s.data(), u.data(), 4,
                            vt.data(), 4),
            -1);
  EXPECT_EQ(LapackeGesdd<T>(LAPACK_ROW_MAJOR, 'A', 4, 4, a.data(), 3, s.data(),
                            u.data(), 4, vt.data(), 4),
            -6);
  EXPECT_EQ(LapackeGesdd<T>(LAPACK_ROW_MAJOR, 'A', 4, 4, a.data(), 4, s.data(),
                            u.data(), 3, vt.data(), 4),
            -9);
  // As in the reference: ldvt >= n also without VT.
  EXPECT_EQ(LapackeGesdd<T>(LAPACK_ROW_MAJOR, 'N', 4, 4, a.data(), 4, s.data(),
                            u.data(), 1, vt.data(), 3),
            -11);
  std::string expected =
      "Wrong parameter 1 in " + LapackeName<T>("gesdd") +
      "\nWrong parameter 6 in " + LapackeName<T>("gesdd_work") +
      "\nWrong parameter 9 in " + LapackeName<T>("gesdd_work") +
      "\nWrong parameter 11 in " + LapackeName<T>("gesdd_work") + "\n";
  std::vector<R> d(4, 1), e(4, 0), q(4);
  std::vector<lapack_int> iq(4);
  if constexpr (!IsComplex<T>::value) {
    EXPECT_EQ(LapackeBdsdc<R>(0, 'U', 'I', 4, d.data(), e.data(), u.data(), 4,
                              vt.data(), 4, q.data(), iq.data()),
              -1);
    EXPECT_EQ(LapackeBdsdc<R>(LAPACK_ROW_MAJOR, 'U', 'I', 4, d.data(), e.data(),
                              u.data(), 3, vt.data(), 4, q.data(), iq.data()),
              -8);
    EXPECT_EQ(LapackeBdsdc<R>(LAPACK_ROW_MAJOR, 'U', 'I', 4, d.data(), e.data(),
                              u.data(), 4, vt.data(), 3, q.data(), iq.data()),
              -10);
    expected += "Wrong parameter 1 in " + LapackeName<R>("bdsdc") +
                "\nWrong parameter 8 in " + LapackeName<R>("bdsdc_work") +
                "\nWrong parameter 10 in " + LapackeName<R>("bdsdc_work") +
                "\n";
  }
  EXPECT_EQ(::testing::internal::GetCapturedStdout(), expected);
  // NaN checks.
  a[5] = NaN<T>();
  EXPECT_EQ(LapackeGesdd<T>(LAPACK_COL_MAJOR, 'N', 4, 4, a.data(), 4, s.data(),
                            u.data(), 1, vt.data(), 1),
            -5);
  if constexpr (!IsComplex<T>::value) {
    d[2] = NAN;
    EXPECT_EQ(LapackeBdsdc<R>(LAPACK_COL_MAJOR, 'U', 'N', 4, d.data(), e.data(),
                              u.data(), 1, vt.data(), 1, q.data(), iq.data()),
              -5);
    d[2] = 1;
    e[1] = NAN;
    EXPECT_EQ(LapackeBdsdc<R>(LAPACK_COL_MAJOR, 'U', 'N', 4, d.data(), e.data(),
                              u.data(), 1, vt.data(), 1, q.data(), iq.data()),
              -6);
  }
}

}  // namespace
}  // namespace hwyblas_test
