// Tests of the MRRR, bisection and inverse iteration symmetric (Hermitian)
// eigensolvers: xSTEMR and its xLARR* routines (xLANEG, xLARRC), xSTEBZ
// (xLAEBZ), xSTEIN (xLAGTF, xLAGTS), xSYEVR / xHEEVR, xSTEVR, xSYEVX /
// xHEEVX, xSTEVX, the random number generators xLARUV and xLARNV, and
// LAPACKE.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
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
class SyevrTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SyevrTest, ScalarTypes);

template <typename R>
class SyevrRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SyevrRealTest, RealTypes);

// Forces ILAENV's block sizes (ISPEC = 1 and 3; nb = 0: ILAENV's own) and,
// for nan_trusted = false, reports that NaN arithmetic is not trusted
// (ISPEC = 10), so that xSYEVR / xHEEVR and xSTEVR compute all eigenvalues
// by xSTEBZ and xSTEIN instead of xSTERF or xSTEMR.
class ScopedConfig {
 public:
  ScopedConfig(hwyblas_int nb, hwyblas_int nx, bool nan_trusted) {
    nb_ = nb;
    nx_ = nx;
    nan_trusted_ = nan_trusted;
    previous_ = hwyblas_set_ilaenv(&Ilaenv);
  }
  ~ScopedConfig() { hwyblas_set_ilaenv(previous_); }
  ScopedConfig(const ScopedConfig&) = delete;
  ScopedConfig& operator=(const ScopedConfig&) = delete;

 private:
  static hwyblas_int Ilaenv(const hwyblas_int* ispec, const char* name,
                            const char* opts, const hwyblas_int* n1,
                            const hwyblas_int* n2, const hwyblas_int* n3,
                            const hwyblas_int* n4, size_t name_len,
                            size_t opts_len) {
    if (nb_ > 0 && *ispec == 1) return nb_;
    if (nb_ > 0 && *ispec == 3) return nx_;
    if (!nan_trusted_ && *ispec == 10) return 0;
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }

  static inline hwyblas_int nb_ = 0;
  static inline hwyblas_int nx_ = 0;
  static inline bool nan_trusted_ = true;
  hwyblas_ilaenv_function previous_ = nullptr;
};

struct Config {
  hwyblas_int nb, nx;
  bool nan_trusted;
};

// The bound of the residual, orthogonality and eigenvalue error ratios of
// the results of xSTEMR (MRRR), whose constants are larger than those of
// xSTEQR, divide and conquer or inverse iteration (kLapackThreshold): on
// the graded matrices of these tests, LAPACK's DSTEMR gives an
// orthogonality ratio of 82, and on a matrix near the underflow threshold
// (RANGE = 'V') a residual ratio of 99 and an eigenvalue error of 754 eps
// ||T|| (hwyBLAS's results are bitwise the same).
constexpr double kMrrrThreshold = 200.0;

// The configurations of the driver tests: ILAENV's, blocked with small
// blocks, and with NaN arithmetic not trusted.
const Config kConfigs[] = {{0, 0, true}, {3, 5, true}, {0, 0, false}};

// The eigenvalues of the symmetric tridiagonal matrix (d, e) in ascending
// order, from xSTERF.
template <typename R>
std::vector<R> SterfEigenvalues(std::vector<R> d, std::vector<R> e) {
  const hwyblas_int n = static_cast<hwyblas_int>(d.size());
  e.resize(std::max<size_t>(1, d.size()));
  hwyblas_int info = -99;
  HWYBLAS_CALL_REAL(R, sterf, &n, d.data(), e.data(), &info);
  EXPECT_EQ(info, 0);
  return d;
}

template <typename R>
R MaxAbs(const std::vector<R>& x, const std::vector<R>& y = {}) {
  R m = 0;
  for (R v : x) m = std::max(m, std::abs(v));
  for (R v : y) m = std::max(m, std::abs(v));
  return m;
}

// TestTridiagonal's kinds 0 to 5 (random, split, graded, clustered, near
// overflow, near underflow) and 6: scaled diagonally dominant with a
// graded diagonal, which defines its eigenvalues to high relative accuracy
// (xSTEMR keeps TRYRAC).
template <typename R>
void MrrrTridiagonal(int kind, hwyblas_int n, Random<R>& rng, std::vector<R>& d,
                     std::vector<R>& e) {
  if (kind != 6) {
    TestTridiagonal(kind, n, rng, d, e);
    return;
  }
  d.resize(n);
  e.resize(std::max<hwyblas_int>(0, n - 1));
  for (hwyblas_int i = 0; i < n; ++i) {
    d[i] = (R(1.5) + rng.Next() / 2) * static_cast<R>(std::pow(10.0, -(i % 6)));
    if (i % 3 == 1) d[i] = -d[i];
  }
  for (hwyblas_int i = 0; i + 1 < n; ++i) {
    e[i] = R(0.2) * rng.Next() * std::sqrt(std::abs(d[i] * d[i + 1]));
  }
}

// A set of wanted eigenvalues: RANGE with VL, VU or IL, IU, and the indices
// first .. first+count-1 (0-based) of the wanted ones among all in
// ascending order.
template <typename R>
struct Subset {
  char range;
  R vl, vu;
  hwyblas_int il, iu;
  hwyblas_int first, count;
};

// Subsets for the ascending eigenvalues ref: all; by index all, the first
// third, one in the middle and the last quarter; by value all, from or
// between gaps of the spectrum, and an empty interval inside a gap.
template <typename R>
std::vector<Subset<R>> Subsets(const std::vector<R>& ref) {
  const hwyblas_int n = static_cast<hwyblas_int>(ref.size());
  std::vector<Subset<R>> s = {{'A', 0, 0, 0, 0, 0, n}};
  if (n == 0) return s;
  const hwyblas_int third = std::max<hwyblas_int>(1, n / 3);
  const hwyblas_int mid = n / 2 + 1, last = n - n / 4;
  s.push_back({'I', 0, 0, 1, n, 0, n});
  s.push_back({'I', 0, 0, 1, third, 0, third});
  s.push_back({'I', 0, 0, mid, mid, mid - 1, 1});
  s.push_back({'I', 0, 0, last, n, last - 1, n - last + 1});
  const R big = MaxAbs(ref);
  const R outside = 2 * big + 1;
  s.push_back({'V', -outside, outside, 0, 0, 0, n});
  // The index i in [1, n-1] nearest to target with a clear gap between
  // ref[i-1] and ref[i], or 0.
  const R gap = static_cast<R>(1000.0 * n * Eps<R>()) * big;
  const auto split = [&](hwyblas_int target) {
    for (hwyblas_int k = 0; k < n; ++k) {
      for (const hwyblas_int i : {target - k, target + k}) {
        if (i >= 1 && i <= n - 1 && ref[i] - ref[i - 1] > gap) return i;
      }
    }
    return hwyblas_int{0};
  };
  const auto between = [&](hwyblas_int i) { return (ref[i - 1] + ref[i]) / 2; };
  const hwyblas_int lo = split(n / 4), hi = split((3 * n) / 4);
  if (lo > 0) {
    s.push_back({'V', between(lo), outside, 0, 0, lo, n - lo});
    s.push_back({'V', -outside, between(lo), 0, 0, 0, lo});
    const R g = ref[lo] - ref[lo - 1];
    s.push_back(
        {'V', ref[lo - 1] + g / 3, ref[lo - 1] + 2 * g / 3, 0, 0, lo, 0});
  }
  if (lo > 0 && hi > lo) {
    s.push_back({'V', between(lo), between(hi), 0, 0, lo, hi - lo});
  }
  return s;
}

template <typename R>
::testing::Message Trace(const Subset<R>& s) {
  ::testing::Message msg;
  msg << "range=" << s.range;
  if (s.range == 'I') msg << " il=" << s.il << " iu=" << s.iu;
  if (s.range == 'V') msg << " vl=" << s.vl << " vu=" << s.vu;
  return msg;
}

// The wanted eigenvalues w[0..m) match ref[first..first+m) within tol and
// ascend.
template <typename R>
void ExpectEigenvalues(const std::vector<R>& w, hwyblas_int m,
                       const std::vector<R>& ref, const Subset<R>& s,
                       double tol) {
  ASSERT_EQ(m, s.count);
  for (hwyblas_int k = 0; k < m; ++k) {
    EXPECT_NEAR(w[k], ref[s.first + k], tol) << k;
    if (k > 0) {
      EXPECT_LE(w[k - 1], w[k]) << k;
    }
  }
}

// The m eigenvectors in z (leading dimension ldz) of the n x n matrix a
// (scaled by 1/scale, with w): orthonormal and with small residuals (both
// ratios below threshold).
template <typename T>
void ExpectEigenvectors(const Dense<T>& a, const std::vector<T>& z,
                        hwyblas_int n, hwyblas_int m, hwyblas_int ldz,
                        const std::vector<RealT<T>>& w, double scale,
                        double threshold) {
  if (m == 0) return;
  const Dense<T> zd = ToDense(z, n, m, ldz);
  EXPECT_LT(OrthogonalityRatio(zd), threshold);
  std::vector<RealT<T>> ws(w.begin(), w.begin() + m);
  for (auto& x : ws) x = static_cast<RealT<T>>(x / scale);
  EXPECT_LT(EigenResidual(a, zd, ws), threshold);
}

// ---------------------------------------------------------------------------
// MRRR

// xSTEMR with the minimal workspaces (minimal) or those of a workspace
// query. d and e are copies; e gets N elements (E(N) is workspace).
template <typename T>
hwyblas_int Stemr(char jobz, const Subset<RealT<T>>& s, std::vector<RealT<T>> d,
                  std::vector<RealT<T>> e, hwyblas_int* m,
                  std::vector<RealT<T>>& w, std::vector<T>& z, hwyblas_int ldz,
                  hwyblas_int nzc, std::vector<hwyblas_int>& isuppz,
                  hwyblas_int* tryrac, bool minimal) {
  using R = RealT<T>;
  const hwyblas_int n = static_cast<hwyblas_int>(d.size());
  e.resize(std::max<hwyblas_int>(1, n));
  w.assign(std::max<hwyblas_int>(1, n), NAN);
  isuppz.assign(2 * static_cast<size_t>(std::max<hwyblas_int>(1, n)), -77);
  hwyblas_int lwork, liwork, info = -99;
  if (minimal) {
    lwork = (jobz == 'V' ? 18 : 12) * n;
    liwork = (jobz == 'V' ? 10 : 8) * n;
  } else {
    R wq = 0;
    hwyblas_int iq = 0;
    const hwyblas_int m1 = -1;
    HWYBLAS_CALL(T, stemr, &jobz, &s.range, &n, d.data(), e.data(), &s.vl,
                 &s.vu, &s.il, &s.iu, m, w.data(), P(z.data()), &ldz, &nzc,
                 isuppz.data(), tryrac, &wq, &m1, &iq, &m1, &info);
    if (info != 0) return info;
    lwork = QueryResult(wq);
    liwork = iq;
  }
  std::vector<R> work(std::max<hwyblas_int>(1, lwork), NAN);
  std::vector<hwyblas_int> iwork(std::max<hwyblas_int>(1, liwork), -77);
  HWYBLAS_CALL(T, stemr, &jobz, &s.range, &n, d.data(), e.data(), &s.vl, &s.vu,
               &s.il, &s.iu, m, w.data(), P(z.data()), &ldz, &nzc,
               isuppz.data(), tryrac, work.data(), &lwork, iwork.data(),
               &liwork, &info);
  return info;
}

// xSTEMR: for all, index and value ranges, the wanted eigenvalues (as from
// xSTERF), orthonormal eigenvectors with small residuals that are zero
// outside their supports ISUPPZ (real for complex T), on tridiagonal
// matrices that split, are graded, have clustered eigenvalues (child
// representations), are scaled, or define their eigenvalues to high
// relative accuracy (TRYRAC stays set, the eigenvalues are refined by
// xLARRJ); with and without TRYRAC and with minimal workspaces.
TYPED_TEST(SyevrTest, Stemr) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<R> rng(307);
  for (const int kind : {0, 1, 2, 3, 4, 5, 6}) {
    for (const hwyblas_int n : {1, 2, 3, 4, 10, 33, 70}) {
      std::vector<R> d0, e0;
      MrrrTridiagonal(kind, n, rng, d0, e0);
      const std::vector<R> ref = SterfEigenvalues(d0, e0);
      // Dense T in scaled form (the residual ratios are scale invariant).
      // (Not zero: the Wilkinson matrix of order 1 is zero.)
      const R anorm = std::max(MaxAbs(d0, e0), std::numeric_limits<R>::min());
      std::vector<R> ds = d0, es = e0;
      for (auto& x : ds) x /= anorm;
      for (auto& x : es) x /= anorm;
      const Dense<T> t = Tridiagonal<T>(ds, es);
      const double tol =
          kMrrrThreshold * n * Eps<T>() * static_cast<double>(anorm);
      for (const Subset<R>& s : Subsets(ref)) {
        for (const char jobz : {'V', 'N'}) {
          for (const hwyblas_int rac : {1, 0}) {
            SCOPED_TRACE(::testing::Message()
                         << "kind=" << kind << " n=" << n << " " << Trace(s)
                         << " jobz=" << jobz << " tryrac=" << rac);
            const hwyblas_int ldz = n + 1;
            std::vector<T> z(ldz * n, NaN<T>());
            std::vector<R> w;
            std::vector<hwyblas_int> isuppz;
            hwyblas_int m = -1, tryrac = rac;
            ASSERT_EQ(Stemr<T>(jobz, s, d0, e0, &m, w, z, ldz, n, isuppz,
                               &tryrac, rac == 0),
                      0);
            ExpectEigenvalues(w, m, ref, s, tol);
            if (rac == 0) {
              EXPECT_EQ(tryrac, 0);
            }
            if (kind == 6 && n > 2 && rac == 1) {
              EXPECT_EQ(tryrac, 1);
            }
            if (jobz != 'V') continue;
            ExpectEigenvectors(t, z, n, m, ldz, w, anorm, kMrrrThreshold);
            // For N = 2, LAPACK's xSTEMR sets the support (2, 2) whenever
            // the sine of the rotation is zero (E(1) = 0), also for the
            // eigenvector (1, 0).
            if (n == 2 && e0[0] == 0) continue;
            for (hwyblas_int j = 0; j < m; ++j) {
              const hwyblas_int lo = isuppz[2 * static_cast<size_t>(j)] - 1;
              const hwyblas_int hi = isuppz[2 * static_cast<size_t>(j) + 1] - 1;
              ASSERT_LE(0, lo) << j;
              ASSERT_LE(lo, hi) << j;
              ASSERT_LT(hi, n) << j;
              for (hwyblas_int i = 0; i < n; ++i) {
                const T zij = z[i + j * ldz];
                EXPECT_EQ(std::imag(zij), 0) << i << ", " << j;
                if (i < lo || i > hi) {
                  EXPECT_EQ(zij, T(0)) << i << ", " << j;
                }
              }
            }
          }
        }
      }
    }
  }
}

// Sturm counts: xLANEG of L D L**T - SIGMA*I for every twist index, and
// xLARRC on T and on L D L**T, give the numbers of eigenvalues below SIGMA
// and in (VL, VU] (the eigenvalues of T = L D L**T from xSTERF).
TYPED_TEST(SyevrRealTest, SturmCounts) {
  using R = TypeParam;
  Random<R> rng(317);
  const hwyblas_int n = 12;
  std::vector<R> d(n), l(n - 1), lld(n - 1), td(n), te(n - 1);
  for (hwyblas_int i = 0; i < n; ++i) {
    d[i] = (R(1) + rng.Next() / 2) * (i % 4 == 2 ? -1 : 1);
  }
  for (auto& x : l) x = rng.Next();
  for (hwyblas_int i = 0; i < n; ++i) {
    td[i] = d[i] + (i > 0 ? l[i - 1] * l[i - 1] * d[i - 1] : R(0));
    if (i + 1 < n) {
      te[i] = l[i] * d[i];
      lld[i] = l[i] * l[i] * d[i];
    }
  }
  const std::vector<R> ref = SterfEigenvalues(td, te);
  for (hwyblas_int i = 1; i < n; ++i) ASSERT_GT(ref[i] - ref[i - 1], R(1e-3));
  // sigma[k]: k eigenvalues below it.
  std::vector<R> sigma(n + 1);
  sigma[0] = ref[0] - 1;
  sigma[n] = ref[n - 1] + 1;
  for (hwyblas_int k = 1; k < n; ++k) sigma[k] = (ref[k - 1] + ref[k]) / 2;
  const R pivmin = std::numeric_limits<R>::min();
  for (hwyblas_int k = 0; k <= n; ++k) {
    for (hwyblas_int r = 1; r <= n; ++r) {
      hwyblas_int count;
      if constexpr (std::is_same_v<R, float>) {
        count = slaneg_(&n, d.data(), lld.data(), &sigma[k], &pivmin, &r);
      } else {
        count = dlaneg_(&n, d.data(), lld.data(), &sigma[k], &pivmin, &r);
      }
      EXPECT_EQ(count, k) << "k=" << k << " r=" << r;
    }
  }
  for (const char jobt : {'T', 'L'}) {
    for (hwyblas_int a = 0; a <= n; a += 3) {
      for (hwyblas_int b = a; b <= n; b += 2) {
        hwyblas_int eigcnt = -1, lcnt = -1, rcnt = -1, info = -99;
        std::vector<R>& dd = jobt == 'T' ? td : d;
        std::vector<R>& ee = jobt == 'T' ? te : l;
        HWYBLAS_CALL_REAL(R, larrc, &jobt, &n, &sigma[a], &sigma[b], dd.data(),
                          ee.data(), &pivmin, &eigcnt, &lcnt, &rcnt, &info);
        EXPECT_EQ(info, 0);
        EXPECT_EQ(lcnt, a) << jobt << " " << a << " " << b;
        EXPECT_EQ(rcnt, b) << jobt << " " << a << " " << b;
        EXPECT_EQ(eigcnt, b - a) << jobt << " " << a << " " << b;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Bisection and inverse iteration

// xSTEBZ (all ranges, ORDER = 'B' and 'E', ABSTOL = 0 and 2*safmin, with
// the scalar and the vector loops of xLAEBZ): the wanted eigenvalues (as
// from xSTERF), consistent blocks; then xSTEIN: orthonormal eigenvectors
// with small residuals, all converged. Not near overflow or underflow:
// xSTEBZ does not scale, and the squares of E overflow or are subnormal
// (the drivers scale first).
TYPED_TEST(SyevrTest, StebzStein) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<R> rng(311);
  for (const hwyblas_int nb : {0, 3}) {  // 3: xLAEBZ's vector loop
    const ScopedConfig config(nb, nb, true);
    for (const int kind : {0, 1, 2, 3}) {
      for (const hwyblas_int n : {1, 2, 5, 17, 40}) {
        std::vector<R> d, e;
        TestTridiagonal(kind, n, rng, d, e);
        e.resize(std::max<hwyblas_int>(1, n));
        const std::vector<R> ref = SterfEigenvalues(d, e);
        // (Not zero: the Wilkinson matrix of order 1 is zero.)
        const R anorm = std::max(MaxAbs(d, e), std::numeric_limits<R>::min());
        std::vector<R> ds = d, es = e;
        for (auto& x : ds) x /= anorm;
        for (auto& x : es) x /= anorm;
        es.resize(std::max<hwyblas_int>(0, n - 1));
        const Dense<T> t = Tridiagonal<T>(ds, es);
        const double tol = 30.0 * n * Eps<T>() * static_cast<double>(anorm);
        for (const Subset<R>& s : Subsets(ref)) {
          for (const char order : {'B', 'E'}) {
            for (const R abstol : {R(0), 2 * std::numeric_limits<R>::min()}) {
              SCOPED_TRACE(::testing::Message()
                           << "nb=" << nb << " kind=" << kind << " n=" << n
                           << " " << Trace(s) << " order=" << order
                           << " abstol=" << abstol);
              hwyblas_int m = -1, nsplit = -1, info = -99;
              std::vector<R> w(n, NAN), work(4 * n, NAN);
              std::vector<hwyblas_int> iblock(n, -77), isplit(n, -77),
                  iwork(3 * static_cast<size_t>(n), -77);
              HWYBLAS_CALL_REAL(R, stebz, &s.range, &order, &n, &s.vl, &s.vu,
                                &s.il, &s.iu, &abstol, d.data(), e.data(), &m,
                                &nsplit, w.data(), iblock.data(), isplit.data(),
                                work.data(), iwork.data(), &info);
              ASSERT_EQ(info, 0);
              ASSERT_EQ(m, s.count);
              ASSERT_GE(nsplit, 1);
              ASSERT_LE(nsplit, n);
              EXPECT_EQ(isplit[nsplit - 1], n);
              for (hwyblas_int k = 1; k < nsplit; ++k) {
                EXPECT_LT(isplit[k - 1], isplit[k]) << k;
              }
              for (hwyblas_int k = 0; k < m; ++k) {
                EXPECT_GE(iblock[k], 1) << k;
                EXPECT_LE(iblock[k], nsplit) << k;
                if (order == 'B' && k > 0) {
                  EXPECT_LE(iblock[k - 1], iblock[k]) << k;
                }
              }
              std::vector<R> sorted(w.begin(), w.begin() + m);
              if (order == 'E') {
                ExpectEigenvalues(w, m, ref, s, tol);
              } else {
                std::sort(sorted.begin(), sorted.end());
                ExpectEigenvalues(sorted, m, ref, s, tol);
              }
              if (order != 'B' || m == 0) continue;
              // xSTEIN with the eigenvalues by block.
              const hwyblas_int ldz = n + 1;
              std::vector<T> z(ldz * m, NaN<T>());
              std::vector<R> swork(5 * n, NAN);
              std::vector<hwyblas_int> siwork(n, -77), ifail(m, -77);
              HWYBLAS_CALL(T, stein, &n, d.data(), e.data(), &m, w.data(),
                           iblock.data(), isplit.data(), P(z.data()), &ldz,
                           swork.data(), siwork.data(), ifail.data(), &info);
              ASSERT_EQ(info, 0);
              for (hwyblas_int k = 0; k < m; ++k) EXPECT_EQ(ifail[k], 0) << k;
              ExpectEigenvectors(t, z, n, m, ldz, w, anorm, kLapackThreshold);
            }
          }
        }
      }
    }
  }
}

// xLAGTF and xLAGTS: (T - LAMBDA*I) x = y (JOB = 1, -1) and its transpose
// (JOB = 2, -2) for tridiagonal T, with partial pivoting; JOB < 0 gives the
// same x when no pivot is small. A zero pivot is reported in IN(N), makes
// JOB > 0 fail with INFO = its index, and is perturbed for JOB < 0.
TYPED_TEST(SyevrRealTest, Lagts) {
  using R = TypeParam;
  Random<R> rng(331);
  for (const hwyblas_int n : {1, 2, 3, 8, 25}) {
    std::vector<R> a(n), b(std::max<hwyblas_int>(1, n - 1)),
        c(std::max<hwyblas_int>(1, n - 1));
    for (auto& x : a) x = rng.Next();
    for (auto& x : b) x = rng.Next();
    for (auto& x : c) x = rng.Next();
    const R lambda = R(0.3);
    Dense<R> tl(n, n);
    for (hwyblas_int i = 0; i < n; ++i) {
      tl(i, i) = a[i] - lambda;
      if (i + 1 < n) {
        tl(i, i + 1) = b[i];
        tl(i + 1, i) = c[i];
      }
    }
    // Factor.
    std::vector<R> fa = a, fb = b, fc = c, fd(std::max<hwyblas_int>(1, n - 2));
    std::vector<hwyblas_int> in(n, -77);
    hwyblas_int info = -99;
    const R tol = 0;
    HWYBLAS_CALL_REAL(R, lagtf, &n, fa.data(), &lambda, fb.data(), fc.data(),
                      &tol, fd.data(), in.data(), &info);
    ASSERT_EQ(info, 0);
    EXPECT_EQ(in[n - 1], 0);
    for (const hwyblas_int job : {1, -1, 2, -2}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " job=" << job);
      std::vector<R> y(n);
      for (auto& x : y) x = rng.Next();
      std::vector<R> x = y;
      R jtol = 0;
      HWYBLAS_CALL_REAL(R, lagts, &job, &n, fa.data(), fb.data(), fc.data(),
                        fd.data(), in.data(), x.data(), &jtol, &info);
      ASSERT_EQ(info, 0);
      if (job < 0) {
        EXPECT_GT(jtol, 0);
      }
      const bool trans = job == 2 || job == -2;
      double resid = 0, norm = 0;
      for (hwyblas_int i = 0; i < n; ++i) {
        double s = 0;
        for (hwyblas_int j = 0; j < n; ++j) {
          s += (trans ? tl(j, i) : tl(i, j)) * static_cast<double>(x[j]);
        }
        resid = std::max(resid, std::abs(s - static_cast<double>(y[i])));
        norm = std::max(norm, std::abs(static_cast<double>(x[i])));
      }
      EXPECT_LT(resid, 1000 * n * Eps<R>() * std::max(1.0, norm));
    }
  }
  // A diagonal T with a(2) = LAMBDA: the second pivot is zero.
  const hwyblas_int n = 4;
  std::vector<R> a = {1, 2, 3, 4}, b(3, 0), c(3, 0), d(2);
  std::vector<hwyblas_int> in(n, -77);
  hwyblas_int info = -99;
  const R lambda = 2, tol = 0;
  HWYBLAS_CALL_REAL(R, lagtf, &n, a.data(), &lambda, b.data(), c.data(), &tol,
                    d.data(), in.data(), &info);
  ASSERT_EQ(info, 0);
  EXPECT_EQ(in[n - 1], 2);
  std::vector<R> y = {1, 1, 1, 1};
  R jtol = 0;
  const hwyblas_int job1 = 1, jobm1 = -1;
  HWYBLAS_CALL_REAL(R, lagts, &job1, &n, a.data(), b.data(), c.data(), d.data(),
                    in.data(), y.data(), &jtol, &info);
  EXPECT_EQ(info, 2);
  y = {1, 1, 1, 1};
  HWYBLAS_CALL_REAL(R, lagts, &jobm1, &n, a.data(), b.data(), c.data(),
                    d.data(), in.data(), y.data(), &jtol, &info);
  EXPECT_EQ(info, 0);
  for (const R x : y) EXPECT_TRUE(std::isfinite(x));
}

// ---------------------------------------------------------------------------
// The drivers

// The eigenvalues of the symmetric (Hermitian) n x n matrix a (triangle
// uplo) from xSYEV / xHEEV.
template <typename T>
std::vector<RealT<T>> ReferenceEigenvalues(char uplo, hwyblas_int n,
                                           std::vector<T> a, hwyblas_int lda) {
  using R = RealT<T>;
  std::vector<R> w(n), rwork(std::max<hwyblas_int>(1, 3 * n - 2));
  hwyblas_int lwork = std::max<hwyblas_int>(1, 64 * n), info = -99;
  std::vector<T> work(lwork);
  const char jobz = 'N';
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heev, &jobz, &uplo, &n, P(a.data()), &lda, w.data(),
                         P(work.data()), &lwork, rwork.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, syev, &jobz, &uplo, &n, a.data(), &lda, w.data(),
                      work.data(), &lwork, &info);
  }
  EXPECT_EQ(info, 0);
  return w;
}

// xSYEVR / xHEEVR with the minimal workspaces (minimal) or those of a
// workspace query; a is a copy.
template <typename T>
hwyblas_int Heevr(char jobz, const Subset<RealT<T>>& s, char uplo,
                  hwyblas_int n, std::vector<T> a, hwyblas_int lda,
                  RealT<T> abstol, hwyblas_int* m, std::vector<RealT<T>>& w,
                  std::vector<T>& z, hwyblas_int ldz,
                  std::vector<hwyblas_int>& isuppz, bool minimal) {
  using R = RealT<T>;
  w.assign(std::max<hwyblas_int>(1, n), NAN);
  isuppz.assign(2 * static_cast<size_t>(std::max<hwyblas_int>(1, n)), -77);
  hwyblas_int lwork, lrwork, liwork, info = -99;
  if (minimal) {
    lwork = n <= 1 ? 1 : (IsComplex<T>::value ? 2 * n : 26 * n);
    lrwork = n <= 1 ? 1 : 24 * n;
    liwork = n <= 1 ? 1 : 10 * n;
  } else {
    T wq = T(0);
    R rq = 0;
    hwyblas_int iq = 0;
    const hwyblas_int m1 = -1;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevr, &jobz, &s.range, &uplo, &n, P(a.data()),
                           &lda, &s.vl, &s.vu, &s.il, &s.iu, &abstol, m,
                           w.data(), P(z.data()), &ldz, isuppz.data(), P(&wq),
                           &m1, &rq, &m1, &iq, &m1, &info);
    } else {
      HWYBLAS_CALL_REAL(T, syevr, &jobz, &s.range, &uplo, &n, a.data(), &lda,
                        &s.vl, &s.vu, &s.il, &s.iu, &abstol, m, w.data(),
                        z.data(), &ldz, isuppz.data(), &wq, &m1, &iq, &m1,
                        &info);
    }
    if (info != 0) return info;
    lwork = QueryResult(wq);
    lrwork = QueryResult(rq);
    liwork = iq;
  }
  std::vector<T> work(lwork, NaN<T>());
  std::vector<R> rwork(std::max<hwyblas_int>(1, lrwork), NAN);
  std::vector<hwyblas_int> iwork(liwork, -77);
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heevr, &jobz, &s.range, &uplo, &n, P(a.data()),
                         &lda, &s.vl, &s.vu, &s.il, &s.iu, &abstol, m, w.data(),
                         P(z.data()), &ldz, isuppz.data(), P(work.data()),
                         &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
                         &info);
  } else {
    HWYBLAS_CALL_REAL(T, syevr, &jobz, &s.range, &uplo, &n, a.data(), &lda,
                      &s.vl, &s.vu, &s.il, &s.iu, &abstol, m, w.data(),
                      z.data(), &ldz, isuppz.data(), work.data(), &lwork,
                      iwork.data(), &liwork, &info);
  }
  return info;
}

// xSYEVR / xHEEVR: for all, index and value ranges, the wanted eigenvalues
// (as from xSYEV / xHEEV) and orthonormal eigenvectors with small
// residuals, also for matrices that the driver scales (norm near the
// underflow or overflow threshold), with blocked and unblocked reductions,
// through xSTEMR (with and without TRYRAC: ABSTOL) and, with NaN arithmetic
// not trusted, through xSTEBZ and xSTEIN, and with minimal workspaces.
TYPED_TEST(SyevrTest, Heevr) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(313);
    const double rmin = std::sqrt(static_cast<double>(
        std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon()));
    for (const Config& c : kConfigs) {
      const ScopedConfig config(c.nb, c.nx, c.nan_trusted);
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {1, 2, 5, 17, 40}) {
          for (const double scale : {1.0, rmin * 1e-3, 1 / (rmin * 1e-3)}) {
            const hwyblas_int lda = n + 1;
            std::vector<T> a = RandomHermitian(uplo, n, lda, rng);
            // xHEEVR's norm (xLANSY) reads the imaginary parts of the
            // diagonal: zero, not NaN, so that it scales.
            for (hwyblas_int i = 0; i < n; ++i) {
              a[i + i * lda] = T(std::real(a[i + i * lda]));
            }
            const Dense<T> ad = HermitianDense(uplo, a, n, lda);
            for (T& x : a) x = x * static_cast<R>(scale);
            const std::vector<R> ref = ReferenceEigenvalues(uplo, n, a, lda);
            for (const Subset<R>& s : Subsets(ref)) {
              // All eigenvalues through xSTEMR, a subset through xSTEBZ
              // and xSTEIN.
              const bool all =
                  s.range == 'A' || (s.range == 'I' && s.il == 1 && s.iu == n);
              const double threshold =
                  all && c.nan_trusted ? kMrrrThreshold : kLapackThreshold;
              const double tol =
                  threshold * n * Eps<T>() * static_cast<double>(MaxAbs(ref));
              for (const char jobz : {'V', 'N'}) {
                for (const R abstol : {R(0), R(1)}) {
                  // A large ABSTOL only through xSTEMR (all eigenvalues),
                  // where it turns TRYRAC off; xSTEBZ would compute the
                  // eigenvalues to that accuracy.
                  if (abstol != 0 && !(all && c.nan_trusted)) continue;
                  const bool minimal = (n + (jobz == 'V')) % 2 == 0;
                  SCOPED_TRACE(::testing::Message()
                               << "nb=" << c.nb << " nan=" << c.nan_trusted
                               << " uplo=" << uplo << " n=" << n
                               << " scale=" << scale << " " << Trace(s)
                               << " jobz=" << jobz << " abstol=" << abstol
                               << " minimal=" << minimal);
                  const hwyblas_int ldz = n + 2;
                  std::vector<T> z(ldz * n, NaN<T>());
                  std::vector<R> w;
                  std::vector<hwyblas_int> isuppz;
                  hwyblas_int m = -1;
                  ASSERT_EQ(Heevr(jobz, s, uplo, n, a, lda, abstol, &m, w, z,
                                  ldz, isuppz, minimal),
                            0);
                  ExpectEigenvalues(w, m, ref, s, tol);
                  if (jobz == 'V') {
                    ExpectEigenvectors(ad, z, n, m, ldz, w, scale, threshold);
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

// xSTEVR: as xSTEMR (through xSTEMR or, with NaN arithmetic not trusted,
// xSTEBZ and xSTEIN), also for tridiagonal matrices that it scales, with
// minimal workspaces.
TYPED_TEST(SyevrRealTest, Stevr) {
  using R = TypeParam;
  Random<R> rng(337);
  for (const bool nan_trusted : {true, false}) {
    const ScopedConfig config(0, 0, nan_trusted);
    for (const int kind : {0, 1, 2, 3, 4, 5, 6}) {
      for (const hwyblas_int n : {0, 1, 2, 9, 40}) {
        std::vector<R> d0, e0;
        MrrrTridiagonal(kind, n, rng, d0, e0);
        const std::vector<R> ref = SterfEigenvalues(d0, e0);
        const R anorm = std::max(MaxAbs(d0, e0), std::numeric_limits<R>::min());
        std::vector<R> ds = d0, es = e0;
        for (auto& x : ds) x /= anorm;
        for (auto& x : es) x /= anorm;
        const Dense<R> t = Tridiagonal<R>(ds, es);
        for (const Subset<R>& s : Subsets(ref)) {
          const bool all =
              s.range == 'A' || (s.range == 'I' && s.il == 1 && s.iu == n);
          const double threshold =
              all && nan_trusted ? kMrrrThreshold : kLapackThreshold;
          const double tol =
              threshold * n * Eps<R>() * static_cast<double>(anorm);
          for (const char jobz : {'V', 'N'}) {
            for (const bool minimal : {false, true}) {
              SCOPED_TRACE(::testing::Message()
                           << "nan=" << nan_trusted << " kind=" << kind
                           << " n=" << n << " " << Trace(s) << " jobz=" << jobz
                           << " minimal=" << minimal);
              std::vector<R> d = d0, e = e0;
              e.resize(std::max<hwyblas_int>(1, n));
              const hwyblas_int ldz = std::max<hwyblas_int>(1, n);
              std::vector<R> z(ldz * std::max<hwyblas_int>(1, n), NAN);
              std::vector<R> w(std::max<hwyblas_int>(1, n), NAN);
              std::vector<hwyblas_int> isuppz(
                  2 * static_cast<size_t>(std::max<hwyblas_int>(1, n)));
              hwyblas_int lwork = std::max<hwyblas_int>(1, 20 * n);
              hwyblas_int liwork = std::max<hwyblas_int>(1, 10 * n);
              hwyblas_int m = -1, info = -99;
              const R abstol = 0;
              if (!minimal) {
                R wq = 0;
                hwyblas_int iq = 0;
                const hwyblas_int m1 = -1;
                HWYBLAS_CALL_REAL(R, stevr, &jobz, &s.range, &n, d.data(),
                                  e.data(), &s.vl, &s.vu, &s.il, &s.iu, &abstol,
                                  &m, w.data(), z.data(), &ldz, isuppz.data(),
                                  &wq, &m1, &iq, &m1, &info);
                ASSERT_EQ(info, 0);
                ASSERT_EQ(QueryResult(wq), lwork);
                ASSERT_EQ(iq, liwork);
              }
              std::vector<R> work(lwork, NAN);
              std::vector<hwyblas_int> iwork(liwork, -77);
              HWYBLAS_CALL_REAL(R, stevr, &jobz, &s.range, &n, d.data(),
                                e.data(), &s.vl, &s.vu, &s.il, &s.iu, &abstol,
                                &m, w.data(), z.data(), &ldz, isuppz.data(),
                                work.data(), &lwork, iwork.data(), &liwork,
                                &info);
              ASSERT_EQ(info, 0);
              ExpectEigenvalues(w, m, ref, s, tol);
              if (jobz == 'V') {
                ExpectEigenvectors(t, z, n, m, ldz, w, anorm, threshold);
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Bisection and inverse iteration drivers

// xSYEVX / xHEEVX with the minimal workspaces (minimal) or that of a
// workspace query, which WORK(1) must give again on exit; a is a copy.
// IFAIL is set to -77 first.
template <typename T>
hwyblas_int Heevx(char jobz, const Subset<RealT<T>>& s, char uplo,
                  hwyblas_int n, std::vector<T> a, hwyblas_int lda,
                  RealT<T> abstol, hwyblas_int* m, std::vector<RealT<T>>& w,
                  std::vector<T>& z, hwyblas_int ldz,
                  std::vector<hwyblas_int>& ifail, bool minimal) {
  using R = RealT<T>;
  w.assign(std::max<hwyblas_int>(1, n), NAN);
  ifail.assign(std::max<hwyblas_int>(1, n), -77);
  std::vector<R> rwork(std::max<hwyblas_int>(1, 7 * n), NAN);
  std::vector<hwyblas_int> iwork(std::max<hwyblas_int>(1, 5 * n), -77);
  hwyblas_int lwork, info = -99;
  if (minimal) {
    lwork = n <= 1 ? 1 : (IsComplex<T>::value ? 2 * n : 8 * n);
  } else {
    T wq = T(0);
    const hwyblas_int m1 = -1;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevx, &jobz, &s.range, &uplo, &n, P(a.data()),
                           &lda, &s.vl, &s.vu, &s.il, &s.iu, &abstol, m,
                           w.data(), P(z.data()), &ldz, P(&wq), &m1,
                           rwork.data(), iwork.data(), ifail.data(), &info);
    } else {
      HWYBLAS_CALL_REAL(T, syevx, &jobz, &s.range, &uplo, &n, a.data(), &lda,
                        &s.vl, &s.vu, &s.il, &s.iu, &abstol, m, w.data(),
                        z.data(), &ldz, &wq, &m1, iwork.data(), ifail.data(),
                        &info);
    }
    if (info != 0) return info;
    lwork = QueryResult(wq);
  }
  std::vector<T> work(lwork, NaN<T>());
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heevx, &jobz, &s.range, &uplo, &n, P(a.data()),
                         &lda, &s.vl, &s.vu, &s.il, &s.iu, &abstol, m, w.data(),
                         P(z.data()), &ldz, P(work.data()), &lwork,
                         rwork.data(), iwork.data(), ifail.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, syevx, &jobz, &s.range, &uplo, &n, a.data(), &lda,
                      &s.vl, &s.vu, &s.il, &s.iu, &abstol, m, w.data(),
                      z.data(), &ldz, work.data(), &lwork, iwork.data(),
                      ifail.data(), &info);
  }
  if (!minimal && info == 0) {
    EXPECT_EQ(QueryResult(work[0]), lwork);
  }
  return info;
}

// The first m elements of IFAIL are zero (no eigenvector failed).
void ExpectNoFailures(const std::vector<hwyblas_int>& ifail, hwyblas_int m) {
  for (hwyblas_int k = 0; k < m; ++k) EXPECT_EQ(ifail[k], 0) << k;
}

// xSYEVX / xHEEVX: for all, index and value ranges, the wanted eigenvalues
// (as from xSYEV / xHEEV) and orthonormal eigenvectors with small
// residuals, also for matrices that the driver scales (norm near the
// underflow or overflow threshold), with blocked and unblocked reductions,
// through xSTERF / xSTEQR (all eigenvalues, ABSTOL = 0) and xSTEBZ and
// xSTEIN (a subset, or ABSTOL > 0), and with minimal workspaces; also for
// block diagonal matrices, whose tridiagonal form splits (xSTEBZ orders the
// eigenvalues by block, the driver sorts them). The imaginary parts of the
// diagonal (NaN) are not referenced.
TYPED_TEST(SyevrTest, Heevx) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(353);
    const double rmin = std::sqrt(static_cast<double>(
        std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon()));
    for (const hwyblas_int nb : {0, 3}) {
      const ScopedConfig config(nb, nb + 2, true);
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {1, 2, 5, 17, 40}) {
          for (const double scale : {1.0, rmin * 1e-3, 1 / (rmin * 1e-3)}) {
            for (const bool blocks : {false, true}) {
              if (blocks && (n < 5 || scale != 1.0)) continue;
              const hwyblas_int lda = n + 1;
              std::vector<T> a = RandomHermitian(uplo, n, lda, rng);
              if (blocks) {
                // Two diagonal blocks of orders h and n-h, the second shifted
                // into the middle of the first's spectrum.
                const hwyblas_int h = n / 2;
                for (hwyblas_int j = 0; j < n; ++j) {
                  for (hwyblas_int i = 0; i < n; ++i) {
                    if ((i < h) != (j < h)) a[i + j * lda] = T(0);
                  }
                }
                for (hwyblas_int i = h; i < n; ++i) a[i + i * lda] += T(R(0.1));
              }
              const Dense<T> ad = HermitianDense(uplo, a, n, lda);
              for (T& x : a) x = x * static_cast<R>(scale);
              const std::vector<R> ref = ReferenceEigenvalues(uplo, n, a, lda);
              const double tol = kLapackThreshold * n * Eps<T>() *
                                 static_cast<double>(MaxAbs(ref));
              for (const Subset<R>& s : Subsets(ref)) {
                for (const char jobz : {'V', 'N'}) {
                  for (const R abstol :
                       {R(0), 2 * std::numeric_limits<R>::min()}) {
                    const bool minimal = (n + (jobz == 'V')) % 2 == 0;
                    SCOPED_TRACE(::testing::Message()
                                 << "nb=" << nb << " uplo=" << uplo << " n="
                                 << n << " scale=" << scale << " " << Trace(s)
                                 << " jobz=" << jobz << " abstol=" << abstol
                                 << " minimal=" << minimal);
                    const hwyblas_int ldz = n + 2;
                    std::vector<T> z(ldz * n, NaN<T>());
                    std::vector<R> w;
                    std::vector<hwyblas_int> ifail;
                    hwyblas_int m = -1;
                    ASSERT_EQ(Heevx(jobz, s, uplo, n, a, lda, abstol, &m, w, z,
                                    ldz, ifail, minimal),
                              0);
                    ExpectEigenvalues(w, m, ref, s, tol);
                    if (jobz == 'V') {
                      if (n > 1) ExpectNoFailures(ifail, m);
                      ExpectEigenvectors(ad, z, n, m, ldz, w, scale,
                                         kLapackThreshold);
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

// xSTEVX: as xSYEVX for tridiagonal matrices (all eigenvectors by xSTEQR),
// also for those that it scales and those whose eigenvalues are graded or
// clustered, with the documented workspaces (5*N).
TYPED_TEST(SyevrRealTest, Stevx) {
  using R = TypeParam;
  Random<R> rng(359);
  for (const int kind : {0, 1, 2, 3, 4, 5, 6}) {
    for (const hwyblas_int n : {0, 1, 2, 9, 40}) {
      std::vector<R> d0, e0;
      MrrrTridiagonal(kind, n, rng, d0, e0);
      const std::vector<R> ref = SterfEigenvalues(d0, e0);
      const R anorm = std::max(MaxAbs(d0, e0), std::numeric_limits<R>::min());
      std::vector<R> ds = d0, es = e0;
      for (auto& x : ds) x /= anorm;
      for (auto& x : es) x /= anorm;
      const Dense<R> t = Tridiagonal<R>(ds, es);
      const double tol =
          kLapackThreshold * n * Eps<R>() * static_cast<double>(anorm);
      for (const Subset<R>& s : Subsets(ref)) {
        for (const char jobz : {'V', 'N'}) {
          for (const R abstol : {R(0), 2 * std::numeric_limits<R>::min()}) {
            SCOPED_TRACE(::testing::Message()
                         << "kind=" << kind << " n=" << n << " " << Trace(s)
                         << " jobz=" << jobz << " abstol=" << abstol);
            std::vector<R> d = d0, e = e0;
            e.resize(std::max<hwyblas_int>(1, n));
            const hwyblas_int ldz = std::max<hwyblas_int>(1, n);
            std::vector<R> z(ldz * std::max<hwyblas_int>(1, n), NAN);
            std::vector<R> w(std::max<hwyblas_int>(1, n), NAN);
            std::vector<R> work(std::max<hwyblas_int>(1, 5 * n), NAN);
            std::vector<hwyblas_int> iwork(std::max<hwyblas_int>(1, 5 * n),
                                           -77),
                ifail(std::max<hwyblas_int>(1, n), -77);
            hwyblas_int m = -1, info = -99;
            HWYBLAS_CALL_REAL(R, stevx, &jobz, &s.range, &n, d.data(), e.data(),
                              &s.vl, &s.vu, &s.il, &s.iu, &abstol, &m, w.data(),
                              z.data(), &ldz, work.data(), iwork.data(),
                              ifail.data(), &info);
            ASSERT_EQ(info, 0);
            ExpectEigenvalues(w, m, ref, s, tol);
            if (jobz == 'V') {
              if (n > 1) ExpectNoFailures(ifail, m);
              ExpectEigenvectors(t, z, n, m, ldz, w, anorm, kLapackThreshold);
            }
          }
        }
      }
    }
  }
}

// For N = 1, xSYEVX / xHEEVX and xSTEVX return the eigenvalue for RANGE =
// 'V' only if it lies in the half-open interval (VL, VU].
TYPED_TEST(SyevrTest, HeevxOrder1ValueRange) {
  using T = TypeParam;
  using R = RealT<T>;
  const hwyblas_int n = 1;
  const R a11 = 2;
  struct Interval {
    R vl, vu;
    hwyblas_int count;
  };
  for (const auto& [vl, vu, count] :
       {Interval{R(1), R(2), 1}, Interval{R(2), R(3), 0},
        Interval{R(0), R(1), 0}}) {
    SCOPED_TRACE(::testing::Message() << vl << " " << vu);
    const Subset<R> s = {'V', vl, vu, 0, 0, 0, count};
    std::vector<T> z(1, NaN<T>());
    std::vector<R> w;
    std::vector<hwyblas_int> ifail;
    hwyblas_int m = -1;
    ASSERT_EQ(Heevx('V', s, 'U', n, std::vector<T>{T(a11)}, n, R(0), &m, w, z,
                    n, ifail, false),
              0);
    ASSERT_EQ(m, count);
    if (count == 1) {
      EXPECT_EQ(w[0], a11);
    }
    EXPECT_EQ(z[0], T(1));
    if constexpr (!IsComplex<T>::value) {
      std::vector<R> d = {a11}, e = {R(0)}, work(5), zz(1, NAN);
      std::vector<hwyblas_int> iwork(5), fail(1);
      const char jobz = 'V', range = 'V';
      const R abstol = 0;
      hwyblas_int info = -99;
      m = -1;
      HWYBLAS_CALL_REAL(R, stevx, &jobz, &range, &n, d.data(), e.data(), &vl,
                        &vu, &n, &n, &abstol, &m, w.data(), zz.data(), &n,
                        work.data(), iwork.data(), fail.data(), &info);
      ASSERT_EQ(info, 0);
      ASSERT_EQ(m, count);
      if (count == 1) {
        EXPECT_EQ(w[0], a11);
      }
      EXPECT_EQ(zz[0], R(1));
    }
  }
}

// The workspace queries of xSYEVX / xHEEVX (max(8*N, (NB+3)*N) real,
// (NB+1)*N complex; 1 for N <= 1) and their and xSTEVX's argument checks.
TYPED_TEST(SyevrTest, HeevxWorkspaceQueriesAndErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const hwyblas_int n = 40;
  std::vector<T> a(n * n), z(n * n);
  std::vector<R> w(n), d(n, 1), e(n, 0), rwork(7 * n);
  std::vector<hwyblas_int> iwork(5 * static_cast<size_t>(n)), ifail(n);
  const hwyblas_int m1 = -1, one = 1;
  const R vl = 0, vu = 2, abstol = 0;
  hwyblas_int m = -1;
  const auto heevx = [&](char jobz, char range, char uplo, hwyblas_int nn,
                         hwyblas_int lda, R lo, R hi, hwyblas_int il,
                         hwyblas_int iu, hwyblas_int ldz, T* work,
                         hwyblas_int lwork) {
    hwyblas_int inf = -99;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevx, &jobz, &range, &uplo, &nn, P(a.data()),
                           &lda, &lo, &hi, &il, &iu, &abstol, &m, w.data(),
                           P(z.data()), &ldz, P(work), &lwork, rwork.data(),
                           iwork.data(), ifail.data(), &inf);
    } else {
      HWYBLAS_CALL_REAL(T, syevx, &jobz, &range, &uplo, &nn, a.data(), &lda,
                        &lo, &hi, &il, &iu, &abstol, &m, w.data(), z.data(),
                        &ldz, work, &lwork, iwork.data(), ifail.data(), &inf);
    }
    return inf;
  };
  for (const hwyblas_int nb : {30, 1}) {
    const ScopedBlockSize block_size(nb);
    for (const hwyblas_int nn : {0, 1, 2, n}) {
      SCOPED_TRACE(::testing::Message() << "nb=" << nb << " n=" << nn);
      T wq = T(0);
      EXPECT_EQ(heevx('V', 'A', 'L', nn, n, vl, vu, one, one, n, &wq, m1), 0);
      const hwyblas_int expected = nn <= 1    ? 1
                                   : kComplex ? (nb + 1) * nn
                                              : std::max((nb + 3) * nn, 8 * nn);
      EXPECT_EQ(QueryResult(wq), expected);
    }
  }

  // Invalid arguments.
  ErrorCapture errors;
  const hwyblas_int k = 4, big = 999;
  std::vector<T> work(1024);
  const auto call = [&](char jobz, char range, char uplo, hwyblas_int nn,
                        hwyblas_int lda, R lo, R hi, hwyblas_int il,
                        hwyblas_int iu, hwyblas_int ldz, hwyblas_int lwork) {
    return heevx(jobz, range, uplo, nn, lda, lo, hi, il, iu, ldz, work.data(),
                 lwork);
  };
  EXPECT_EQ(call('X', 'A', 'L', k, k, 0, 1, 1, 1, k, big), -1);
  EXPECT_EQ(call('V', 'X', 'L', k, k, 0, 1, 1, 1, k, big), -2);
  EXPECT_EQ(call('V', 'A', 'X', k, k, 0, 1, 1, 1, k, big), -3);
  EXPECT_EQ(call('V', 'A', 'L', -1, k, 0, 1, 1, 1, k, big), -4);
  EXPECT_EQ(call('V', 'A', 'L', k, k - 1, 0, 1, 1, 1, k, big), -6);
  EXPECT_EQ(call('V', 'V', 'L', k, k, 1, 1, 1, 1, k, big), -8);
  EXPECT_EQ(call('V', 'I', 'L', k, k, 0, 1, 0, 1, k, big), -9);
  EXPECT_EQ(call('V', 'I', 'L', k, k, 0, 1, 2, 1, k, big), -10);
  EXPECT_EQ(call('V', 'I', 'L', k, k, 0, 1, 1, k + 1, k, big), -10);
  EXPECT_EQ(call('V', 'A', 'L', k, k, 0, 1, 1, 1, k - 1, big), -15);
  EXPECT_EQ(call('N', 'A', 'L', k, k, 0, 1, 1, 1, 0, big), -15);
  EXPECT_EQ(
      call('V', 'A', 'L', k, k, 0, 1, 1, 1, k, (kComplex ? 2 : 8) * k - 1),
      -17);
  // Valid: N = 0 with any VL, VU; LDZ = 1 for JOBZ = 'N'; minimal LWORK.
  EXPECT_EQ(call('V', 'V', 'L', 0, 1, 1, 1, 1, 1, 1, 1), 0);
  EXPECT_EQ(call('N', 'A', 'L', k, k, 0, 1, 1, 1, 1, (kComplex ? 2 : 8) * k),
            0);
  const std::string ev = Name<T>(kComplex ? "HEEVX" : "SYEVX");
  std::vector<std::string> expected;
  for (const char* pos :
       {"1", "2", "3", "4", "6", "8", "9", "10", "10", "15", "15", "17"}) {
    expected.push_back(ev + ":" + pos);
  }
  EXPECT_EQ(errors.Take(), expected);

  if constexpr (!kComplex) {
    const auto stevx = [&](char jobz, char range, hwyblas_int nn, R lo, R hi,
                           hwyblas_int il, hwyblas_int iu, hwyblas_int ldz) {
      hwyblas_int inf = -99;
      std::vector<R> rw(5 * k);
      HWYBLAS_CALL_REAL(T, stevx, &jobz, &range, &nn, d.data(), e.data(), &lo,
                        &hi, &il, &iu, &abstol, &m, w.data(), z.data(), &ldz,
                        rw.data(), iwork.data(), ifail.data(), &inf);
      return inf;
    };
    EXPECT_EQ(stevx('X', 'A', k, 0, 1, 1, 1, k), -1);
    EXPECT_EQ(stevx('V', 'X', k, 0, 1, 1, 1, k), -2);
    EXPECT_EQ(stevx('V', 'A', -1, 0, 1, 1, 1, k), -3);
    EXPECT_EQ(stevx('V', 'V', k, 1, 1, 1, 1, k), -7);
    EXPECT_EQ(stevx('V', 'I', k, 0, 1, 0, 1, k), -8);
    EXPECT_EQ(stevx('V', 'I', k, 0, 1, 2, 1, k), -9);
    EXPECT_EQ(stevx('V', 'A', k, 0, 1, 1, 1, k - 1), -14);
    EXPECT_EQ(stevx('N', 'A', k, 0, 1, 1, 1, 0), -14);
    EXPECT_EQ(stevx('N', 'A', k, 0, 1, 1, 1, 1), 0);
    const std::string vx = Name<T>("STEVX");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{vx + ":1", vx + ":2", vx + ":3",
                                        vx + ":7", vx + ":8", vx + ":9",
                                        vx + ":14", vx + ":14"}));
  }
}

// ---------------------------------------------------------------------------
// Random numbers

// x * y mod 2**48 for x, y < 2**48.
uint64_t MulMod48(uint64_t x, uint64_t y) {
  const uint64_t m24 = (uint64_t{1} << 24) - 1;
  const uint64_t xl = x & m24, xh = x >> 24, yl = y & m24, yh = y >> 24;
  const uint64_t cross = (xh * yl + xl * yh) & m24;
  return (xl * yl + (cross << 24)) & ((uint64_t{1} << 48) - 1);
}

// The 48-bit integer of the seed (4 base-4096 digits) and back.
uint64_t SeedValue(const hwyblas_int* iseed) {
  uint64_t s = 0;
  for (int i = 0; i < 4; ++i) s = s * 4096 + static_cast<uint64_t>(iseed[i]);
  return s;
}
void SetSeed(uint64_t s, hwyblas_int* iseed) {
  for (int i = 3; i >= 0; --i) {
    iseed[i] = static_cast<hwyblas_int>(s % 4096);
    s /= 4096;
  }
}

template <typename R>
void Laruv(hwyblas_int* iseed, hwyblas_int n, R* x) {
  if constexpr (std::is_same_v<R, float>) {
    slaruv_(iseed, &n, x);
  } else {
    dlaruv_(iseed, &n, x);
  }
}

// xLARUV: the i-th number is the seed times the i-th power of the multiplier
// 33952834046453 modulo 2**48, as a fraction of 2**48 (from its base-4096
// digits, as xLARUV rounds it), and the new seed is the last of them.
TYPED_TEST(SyevrRealTest, Laruv) {
  using R = TypeParam;
  const uint64_t a = 33952834046453;
  const R r = R(1) / 4096;
  for (const hwyblas_int n : {1, 7, 128}) {
    for (const uint64_t seed :
         {uint64_t{1}, uint64_t{0x123456789AB}, (uint64_t{1} << 48) - 1}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " seed=" << seed);
      hwyblas_int iseed[4];
      SetSeed(seed, iseed);
      std::vector<R> x(n);
      Laruv(iseed, n, x.data());
      // p = seed * a**i mod 2**48.
      uint64_t p = seed;
      for (hwyblas_int i = 0; i < n; ++i) {
        p = MulMod48(p, a);
        hwyblas_int it[4];
        SetSeed(p, it);
        const R expected =
            r * (R(it[0]) + r * (R(it[1]) + r * (R(it[2]) + r * R(it[3]))));
        ASSERT_EQ(x[i], expected) << i;
      }
      EXPECT_EQ(SeedValue(iseed), p);
    }
  }
}

// xLARNV: the uniform numbers of xLARUV in chunks of 64 (real IDIST = 1, 2)
// or 128 per 64 results (real IDIST = 3, complex), transformed to the
// distribution IDIST.
TYPED_TEST(SyevrTest, Larnv) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const double twopi = 6.28318530717958647692528676655900576839;
  const hwyblas_int n = 150;
  for (const hwyblas_int idist : {1, 2, 3, 4, 5}) {
    if (!kComplex && idist > 3) continue;
    SCOPED_TRACE(idist);
    hwyblas_int iseed[4] = {1, 2, 3, 5}, useed[4] = {1, 2, 3, 5};
    std::vector<T> x(n);
    HWYBLAS_CALL(T, larnv, &idist, iseed, &n, P(x.data()));
    // The uniform numbers per chunk of 64 results.
    const bool pairs = kComplex || idist == 3;
    for (hwyblas_int iv = 0; iv < n; iv += 64) {
      const hwyblas_int il = std::min<hwyblas_int>(64, n - iv);
      std::vector<R> u(2 * il);
      Laruv(useed, pairs ? 2 * il : il, u.data());
      for (hwyblas_int i = 0; i < il; ++i) {
        const T got = x[iv + i];
        if (!pairs) {
          const R expected = idist == 1 ? u[i] : 2 * u[i] - 1;
          ASSERT_EQ(got, T(expected)) << iv + i;
          continue;
        }
        const double u1 = u[2 * i], u2 = u[2 * i + 1];
        const double tol = 64 * Eps<R>();
        if (!kComplex) {
          const double expected =
              std::sqrt(-2 * std::log(u1)) * std::cos(twopi * u2);
          ASSERT_NEAR(std::real(got), expected, tol * (1 + std::abs(expected)))
              << iv + i;
          continue;
        }
        std::complex<double> expected;
        const std::complex<double> phase =
            std::exp(std::complex<double>(0, twopi * u2));
        switch (idist) {
          case 1:
            expected = {u1, u2};
            break;
          case 2:
            expected = {2 * u1 - 1, 2 * u2 - 1};
            break;
          case 3:
            expected = std::sqrt(-2 * std::log(u1)) * phase;
            break;
          case 4:
            expected = std::sqrt(u1) * phase;
            break;
          default:
            expected = phase;
            break;
        }
        ASSERT_NEAR(std::real(got), expected.real(),
                    tol * (1 + std::abs(expected)))
            << iv + i;
        ASSERT_NEAR(std::imag(got), expected.imag(),
                    tol * (1 + std::abs(expected)))
            << iv + i;
      }
    }
    for (int k = 0; k < 4; ++k) EXPECT_EQ(iseed[k], useed[k]) << k;
  }
}

// ---------------------------------------------------------------------------
// Workspace queries and errors

TYPED_TEST(SyevrTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const hwyblas_int n = 40;
  std::vector<T> a(n * n), z(n * n);
  std::vector<R> w(n), d(n, 1), e(n, 0);
  std::vector<hwyblas_int> isuppz(2 * static_cast<size_t>(n));
  const hwyblas_int m1 = -1, one = 1;
  const R vl = 0, vu = 2, abstol = 0;
  hwyblas_int m = -1, info = -99;
  // xSYEVR / xHEEVR: the optimal LWORK with NB = 30.
  {
    const ScopedBlockSize block_size(30);
    T wq = T(0);
    R rq = 0;
    hwyblas_int iq = 0;
    const char jobz = 'V', range = 'A', uplo = 'L';
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevr, &jobz, &range, &uplo, &n, P(a.data()), &n,
                           &vl, &vu, &one, &n, &abstol, &m, w.data(),
                           P(z.data()), &n, isuppz.data(), P(&wq), &m1, &rq,
                           &m1, &iq, &m1, &info);
      EXPECT_EQ(QueryResult(rq), 24 * n);
    } else {
      HWYBLAS_CALL_REAL(T, syevr, &jobz, &range, &uplo, &n, a.data(), &n, &vl,
                        &vu, &one, &n, &abstol, &m, w.data(), z.data(), &n,
                        isuppz.data(), &wq, &m1, &iq, &m1, &info);
    }
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(wq), (30 + 1) * n);
    EXPECT_EQ(iq, 10 * n);
  }
  // xSTEMR: LWORK, LIWORK and the number of columns of Z (NZC = -1).
  for (const char jobz : {'V', 'N'}) {
    for (const char range : {'A', 'V', 'I'}) {
      SCOPED_TRACE(::testing::Message() << jobz << " " << range);
      std::vector<R> dd(n), ee(n, R(0.5));
      for (hwyblas_int i = 0; i < n; ++i) dd[i] = R(i);
      R wq = 0;
      hwyblas_int iq = 0, tryrac = 1;
      const hwyblas_int il = 3, iu = 9;
      // (VL, VU] between eigenvalues, containing the 2nd to 4th.
      const std::vector<R> ref = SterfEigenvalues(
          dd, std::vector<R>(ee.begin(), ee.begin() + (n - 1)));
      const R lo = (ref[0] + ref[1]) / 2, hi = (ref[3] + ref[4]) / 2;
      const hwyblas_int count = 3;
      HWYBLAS_CALL(T, stemr, &jobz, &range, &n, dd.data(), ee.data(), &lo, &hi,
                   &il, &iu, &m, w.data(), P(z.data()), &n, &m1, isuppz.data(),
                   &tryrac, &wq, &m1, &iq, &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), (jobz == 'V' ? 18 : 12) * n);
      EXPECT_EQ(iq, (jobz == 'V' ? 10 : 8) * n);
      const hwyblas_int nzc = jobz == 'N'    ? 0
                              : range == 'A' ? n
                              : range == 'I' ? iu - il + 1
                                             : count;
      EXPECT_EQ(QueryResult(z[0]), nzc);
    }
  }
  // xSTEVR.
  if constexpr (!kComplex) {
    for (const hwyblas_int nn : {0, 1, n}) {
      R wq = 0;
      hwyblas_int iq = 0;
      const char jobz = 'V', range = 'A';
      HWYBLAS_CALL_REAL(T, stevr, &jobz, &range, &nn, d.data(), e.data(), &vl,
                        &vu, &one, &one, &abstol, &m, w.data(), z.data(), &n,
                        isuppz.data(), &wq, &m1, &iq, &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), std::max<hwyblas_int>(1, 20 * nn));
      EXPECT_EQ(iq, std::max<hwyblas_int>(1, 10 * nn));
    }
  }

  // Invalid arguments.
  ErrorCapture errors;
  const hwyblas_int k = 4;
  std::vector<T> work(1024);
  std::vector<R> rwork(1024);
  std::vector<hwyblas_int> iwork(1024);
  const auto heevr = [&](char jobz, char range, char uplo, hwyblas_int nn,
                         hwyblas_int lda, R lo, R hi, hwyblas_int il,
                         hwyblas_int iu, hwyblas_int ldz, hwyblas_int lwork,
                         hwyblas_int lrwork, hwyblas_int liwork) {
    hwyblas_int inf = -99;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevr, &jobz, &range, &uplo, &nn, P(a.data()),
                           &lda, &lo, &hi, &il, &iu, &abstol, &m, w.data(),
                           P(z.data()), &ldz, isuppz.data(), P(work.data()),
                           &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
                           &inf);
    } else {
      HWYBLAS_CALL_REAL(T, syevr, &jobz, &range, &uplo, &nn, a.data(), &lda,
                        &lo, &hi, &il, &iu, &abstol, &m, w.data(), z.data(),
                        &ldz, isuppz.data(), work.data(), &lwork, iwork.data(),
                        &liwork, &inf);
    }
    return inf;
  };
  const hwyblas_int big = 999;
  EXPECT_EQ(heevr('X', 'A', 'L', k, k, 0, 1, 1, 1, k, big, big, big), -1);
  EXPECT_EQ(heevr('V', 'X', 'L', k, k, 0, 1, 1, 1, k, big, big, big), -2);
  EXPECT_EQ(heevr('V', 'A', 'X', k, k, 0, 1, 1, 1, k, big, big, big), -3);
  EXPECT_EQ(heevr('V', 'A', 'L', -1, k, 0, 1, 1, 1, k, big, big, big), -4);
  EXPECT_EQ(heevr('V', 'A', 'L', k, k - 1, 0, 1, 1, 1, k, big, big, big), -6);
  EXPECT_EQ(heevr('V', 'V', 'L', k, k, 1, 1, 1, 1, k, big, big, big), -8);
  EXPECT_EQ(heevr('V', 'I', 'L', k, k, 0, 1, 0, 1, k, big, big, big), -9);
  EXPECT_EQ(heevr('V', 'I', 'L', k, k, 0, 1, 2, 1, k, big, big, big), -10);
  EXPECT_EQ(heevr('V', 'A', 'L', k, k, 0, 1, 1, 1, k - 1, big, big, big), -15);
  EXPECT_EQ(heevr('V', 'A', 'L', k, k, 0, 1, 1, 1, k,
                  (kComplex ? 2 : 26) * k - 1, big, big),
            -18);
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(heevr('V', 'A', 'L', k, k, 0, 1, 1, 1, k, big, 24 * k - 1, big),
              -20);
    EXPECT_EQ(heevr('V', 'A', 'L', k, k, 0, 1, 1, 1, k, big, big, 10 * k - 1),
              -22);
  } else {
    EXPECT_EQ(heevr('V', 'A', 'L', k, k, 0, 1, 1, 1, k, big, big, 10 * k - 1),
              -20);
  }
  const std::string ev = Name<T>(kComplex ? "HEEVR" : "SYEVR");
  std::vector<std::string> expected;
  for (const char* pos :
       {"1", "2", "3", "4", "6", "8", "9", "10", "15", "18", "20"}) {
    expected.push_back(ev + ":" + pos);
  }
  if (kComplex) expected.push_back(ev + ":22");
  EXPECT_EQ(errors.Take(), expected);

  const auto stemr = [&](char jobz, char range, hwyblas_int nn, R lo, R hi,
                         hwyblas_int il, hwyblas_int iu, hwyblas_int ldz,
                         hwyblas_int nzc, hwyblas_int lwork,
                         hwyblas_int liwork) {
    hwyblas_int inf = -99, tryrac = 1;
    HWYBLAS_CALL(T, stemr, &jobz, &range, &nn, d.data(), e.data(), &lo, &hi,
                 &il, &iu, &m, w.data(), P(z.data()), &ldz, &nzc, isuppz.data(),
                 &tryrac, rwork.data(), &lwork, iwork.data(), &liwork, &inf);
    return inf;
  };
  EXPECT_EQ(stemr('X', 'A', k, 0, 1, 1, 1, k, k, big, big), -1);
  EXPECT_EQ(stemr('V', 'X', k, 0, 1, 1, 1, k, k, big, big), -2);
  EXPECT_EQ(stemr('V', 'A', -1, 0, 1, 1, 1, k, k, big, big), -3);
  EXPECT_EQ(stemr('V', 'V', k, 1, 1, 1, 1, k, k, big, big), -7);
  EXPECT_EQ(stemr('V', 'I', k, 0, 1, 0, 1, k, k, big, big), -8);
  EXPECT_EQ(stemr('V', 'I', k, 0, 1, 2, 1, k, k, big, big), -9);
  EXPECT_EQ(stemr('V', 'A', k, 0, 1, 1, 1, k - 1, k, big, big), -13);
  EXPECT_EQ(stemr('N', 'A', k, 0, 1, 1, 1, 0, k, big, big), -13);
  // As LAPACK: -17 for LWORK (argument 18) and -19 for LIWORK (20).
  EXPECT_EQ(stemr('V', 'A', k, 0, 1, 1, 1, k, k, 18 * k - 1, big), -17);
  EXPECT_EQ(stemr('V', 'A', k, 0, 1, 1, 1, k, k, big, 10 * k - 1), -19);
  EXPECT_EQ(stemr('V', 'A', k, 0, 1, 1, 1, k, k - 1, big, big), -14);
  const std::string mr = Name<T>("STEMR");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               mr + ":1", mr + ":2", mr + ":3", mr + ":7",
                               mr + ":8", mr + ":9", mr + ":13", mr + ":13",
                               mr + ":17", mr + ":19", mr + ":14"}));

  const auto stein = [&](hwyblas_int nn, hwyblas_int mm, std::vector<R> ws,
                         std::vector<hwyblas_int> iblock, hwyblas_int ldz) {
    hwyblas_int inf = -99;
    // xSTEIN sets IFAIL(1..M) before it checks M.
    std::vector<hwyblas_int> isplit = {2, 4}, ifail(8);
    HWYBLAS_CALL(T, stein, &nn, d.data(), e.data(), &mm, ws.data(),
                 iblock.data(), isplit.data(), P(z.data()), &ldz, rwork.data(),
                 iwork.data(), ifail.data(), &inf);
    return inf;
  };
  EXPECT_EQ(stein(-1, 0, {1}, {1}, 1), -1);
  EXPECT_EQ(stein(k, k + 1, {1, 2, 3, 4, 5}, {1, 1, 2, 2, 2}, k), -4);
  EXPECT_EQ(stein(k, 2, {1, 2}, {1, 1}, k - 1), -9);
  EXPECT_EQ(stein(k, 3, {1, 2, 3}, {1, 2, 1}, k), -6);
  EXPECT_EQ(stein(k, 3, {1, 2, 1}, {1, 2, 2}, k), -5);
  const std::string si = Name<T>("STEIN");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{si + ":1", si + ":4", si + ":9",
                                      si + ":6", si + ":5"}));

  if constexpr (!kComplex) {
    const auto stebz = [&](char range, char order, hwyblas_int nn, R lo, R hi,
                           hwyblas_int il, hwyblas_int iu) {
      hwyblas_int inf = -99, nsplit = 0;
      HWYBLAS_CALL_REAL(T, stebz, &range, &order, &nn, &lo, &hi, &il, &iu,
                        &abstol, d.data(), e.data(), &m, &nsplit, w.data(),
                        iwork.data(), iwork.data() + 100, rwork.data(),
                        iwork.data() + 200, &inf);
      return inf;
    };
    EXPECT_EQ(stebz('X', 'E', k, 0, 1, 1, 1), -1);
    EXPECT_EQ(stebz('A', 'X', k, 0, 1, 1, 1), -2);
    EXPECT_EQ(stebz('A', 'E', -1, 0, 1, 1, 1), -3);
    EXPECT_EQ(stebz('V', 'E', k, 1, 1, 1, 1), -5);
    EXPECT_EQ(stebz('I', 'E', k, 0, 1, 0, 1), -6);
    EXPECT_EQ(stebz('I', 'E', k, 0, 1, 2, 1), -7);
    const auto stevr = [&](char jobz, char range, hwyblas_int nn, R lo, R hi,
                           hwyblas_int il, hwyblas_int iu, hwyblas_int ldz,
                           hwyblas_int lwork, hwyblas_int liwork) {
      hwyblas_int inf = -99;
      HWYBLAS_CALL_REAL(T, stevr, &jobz, &range, &nn, d.data(), e.data(), &lo,
                        &hi, &il, &iu, &abstol, &m, w.data(), z.data(), &ldz,
                        isuppz.data(), rwork.data(), &lwork, iwork.data(),
                        &liwork, &inf);
      return inf;
    };
    EXPECT_EQ(stevr('X', 'A', k, 0, 1, 1, 1, k, big, big), -1);
    EXPECT_EQ(stevr('V', 'X', k, 0, 1, 1, 1, k, big, big), -2);
    EXPECT_EQ(stevr('V', 'A', -1, 0, 1, 1, 1, k, big, big), -3);
    EXPECT_EQ(stevr('V', 'V', k, 1, 1, 1, 1, k, big, big), -7);
    EXPECT_EQ(stevr('V', 'I', k, 0, 1, 0, 1, k, big, big), -8);
    EXPECT_EQ(stevr('V', 'I', k, 0, 1, 2, 1, k, big, big), -9);
    EXPECT_EQ(stevr('V', 'A', k, 0, 1, 1, 1, k - 1, big, big), -14);
    EXPECT_EQ(stevr('V', 'A', k, 0, 1, 1, 1, k, 20 * k - 1, big), -17);
    EXPECT_EQ(stevr('V', 'A', k, 0, 1, 1, 1, k, big, 10 * k - 1), -19);
    hwyblas_int inf = -99;
    const hwyblas_int minus1 = -1, job3 = 3;
    const R lambda = 0, tol = 0;
    std::vector<R> y(4);
    R ytol = 0;
    HWYBLAS_CALL_REAL(T, lagtf, &minus1, d.data(), &lambda, e.data(), e.data(),
                      &tol, rwork.data(), iwork.data(), &inf);
    EXPECT_EQ(inf, -1);
    HWYBLAS_CALL_REAL(T, lagts, &job3, &k, d.data(), e.data(), e.data(),
                      rwork.data(), iwork.data(), y.data(), &ytol, &inf);
    EXPECT_EQ(inf, -1);
    const hwyblas_int job1 = 1;
    HWYBLAS_CALL_REAL(T, lagts, &job1, &minus1, d.data(), e.data(), e.data(),
                      rwork.data(), iwork.data(), y.data(), &ytol, &inf);
    EXPECT_EQ(inf, -2);
    const std::string bz = Name<T>("STEBZ"), vr = Name<T>("STEVR");
    EXPECT_EQ(
        errors.Take(),
        (std::vector<std::string>{
            bz + ":1", bz + ":2", bz + ":3", bz + ":5", bz + ":6", bz + ":7",
            vr + ":1", vr + ":2", vr + ":3", vr + ":7", vr + ":8", vr + ":9",
            vr + ":14", vr + ":17", vr + ":19", Name<T>("LAGTF") + ":1",
            Name<T>("LAGTS") + ":1", Name<T>("LAGTS") + ":2"}));
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

// Row-major results are the column-major ones of the transposed matrices,
// bitwise, for all the functions; ?stebz and ?larnv (without layout) give
// the results of the Fortran routines.
TYPED_TEST(SyevrTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(347);
  Random<R> rrng(349);
  const lapack_int n = 30, ld = 32;
  for (const char uplo : {'U', 'L'}) {
    std::vector<T> a(ld * ld, T(3));
    {
      const std::vector<T> h = RandomHermitian(uplo, n, ld, rng);
      for (size_t i = 0; i < h.size(); ++i) {
        if (!std::isnan(std::real(h[i]))) a[i] = h[i];
      }
      for (lapack_int i = 0; i < n; ++i) {
        At(a, i, i, ld) = T(std::real(At(a, i, i, ld)));
      }
    }
    const std::vector<R> ref =
        ReferenceEigenvalues(uplo, n, std::vector<T>(a), ld);
    for (const Subset<R>& s : Subsets(ref)) {
      for (const char jobz : {'V', 'N'}) {
        SCOPED_TRACE(::testing::Message()
                     << uplo << " " << Trace(s) << " " << jobz);
        std::vector<T> ac = a, ar = Transposed(a, ld);
        std::vector<T> zc(ld * ld, NaN<T>()), zr(ld * ld, NaN<T>());
        std::vector<R> wc(n), wr(n);
        std::vector<lapack_int> sc(2 * static_cast<size_t>(n), -1),
            sr(2 * static_cast<size_t>(n), -1);
        lapack_int mc = -1, mr = -1;
        const lapack_int ic =
            LAPACKE_CALL_RC(T, syevr, heevr, LAPACK_COL_MAJOR, jobz, s.range,
                            uplo, n, ac.data(), ld, s.vl, s.vu, s.il, s.iu,
                            R(0), &mc, wc.data(), zc.data(), ld, sc.data());
        const lapack_int ir =
            LAPACKE_CALL_RC(T, syevr, heevr, LAPACK_ROW_MAJOR, jobz, s.range,
                            uplo, n, ar.data(), ld, s.vl, s.vu, s.il, s.iu,
                            R(0), &mr, wr.data(), zr.data(), ld, sr.data());
        ASSERT_EQ(ic, 0);
        ASSERT_EQ(ir, 0);
        ASSERT_EQ(mc, s.count);
        ASSERT_EQ(mr, mc);
        EXPECT_EQ(wr, wc);
        if (jobz == 'V') {
          EXPECT_EQ(sr, sc);
          for (lapack_int i = 0; i < n; ++i) {
            for (lapack_int j = 0; j < mc; ++j) {
              ASSERT_EQ(zr[i * ld + j], At(zc, i, j, ld)) << i << ", " << j;
            }
          }
        }
        // ?syevx / ?heevx.
        ac = a;
        ar = Transposed(a, ld);
        std::vector<lapack_int> fc(n, -1), fr(n, -1);
        const lapack_int xc =
            LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_COL_MAJOR, jobz, s.range,
                            uplo, n, ac.data(), ld, s.vl, s.vu, s.il, s.iu,
                            R(0), &mc, wc.data(), zc.data(), ld, fc.data());
        const lapack_int xr =
            LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_ROW_MAJOR, jobz, s.range,
                            uplo, n, ar.data(), ld, s.vl, s.vu, s.il, s.iu,
                            R(0), &mr, wr.data(), zr.data(), ld, fr.data());
        ASSERT_EQ(xc, 0);
        ASSERT_EQ(xr, 0);
        ASSERT_EQ(mc, s.count);
        ASSERT_EQ(mr, mc);
        EXPECT_EQ(wr, wc);
        if (jobz == 'V') {
          EXPECT_EQ(fr, fc);
          for (lapack_int i = 0; i < n; ++i) {
            for (lapack_int j = 0; j < mc; ++j) {
              ASSERT_EQ(zr[i * ld + j], At(zc, i, j, ld)) << i << ", " << j;
            }
          }
        }
        // The reduced matrix in the triangle uplo of A (copied back).
        for (lapack_int i = 0; i < n; ++i) {
          for (lapack_int j = 0; j < n; ++j) {
            if (uplo == 'U' ? i <= j : i >= j) {
              ASSERT_EQ(ar[i * ld + j], At(ac, i, j, ld)) << i << ", " << j;
            }
          }
        }
      }
    }
  }

  // The tridiagonal routines.
  std::vector<R> d0(n), e0(n);
  for (auto& x : d0) x = rrng.Next();
  for (auto& x : e0) x = rrng.Next();
  const std::vector<R> ref =
      SterfEigenvalues(d0, std::vector<R>(e0.begin(), e0.end() - 1));
  for (const Subset<R>& s : Subsets(ref)) {
    for (const char jobz : {'V', 'N'}) {
      SCOPED_TRACE(::testing::Message() << Trace(s) << " " << jobz);
      // ?stemr.
      {
        std::vector<R> dc = d0, ec = e0, dr = d0, er = e0, wc(n), wr(n);
        std::vector<T> zc(ld * ld, NaN<T>()), zr(ld * ld, NaN<T>());
        std::vector<lapack_int> sc(2 * static_cast<size_t>(n), -1),
            sr(2 * static_cast<size_t>(n), -1);
        lapack_int mc = -1, mr = -1;
        lapack_logical tc = 1, tr = 1;
        const lapack_int ic =
            LAPACKE_CALL(T, stemr, LAPACK_COL_MAJOR, jobz, s.range, n,
                         dc.data(), ec.data(), s.vl, s.vu, s.il, s.iu, &mc,
                         wc.data(), zc.data(), ld, n, sc.data(), &tc);
        const lapack_int ir =
            LAPACKE_CALL(T, stemr, LAPACK_ROW_MAJOR, jobz, s.range, n,
                         dr.data(), er.data(), s.vl, s.vu, s.il, s.iu, &mr,
                         wr.data(), zr.data(), ld, n, sr.data(), &tr);
        ASSERT_EQ(ic, 0);
        ASSERT_EQ(ir, 0);
        ASSERT_EQ(mc, s.count);
        ASSERT_EQ(mr, mc);
        EXPECT_EQ(tr, tc);
        EXPECT_EQ(wr, wc);
        if (jobz == 'V') {
          EXPECT_EQ(sr, sc);
          for (lapack_int i = 0; i < n; ++i) {
            for (lapack_int j = 0; j < mc; ++j) {
              ASSERT_EQ(zr[i * ld + j], At(zc, i, j, ld)) << i << ", " << j;
            }
          }
        }
      }
      if constexpr (!IsComplex<T>::value) {
        // ?stevr.
        std::vector<R> dc = d0, ec = e0, dr = d0, er = e0, wc(n), wr(n);
        std::vector<R> zc(ld * ld, NAN), zr(ld * ld, NAN);
        std::vector<lapack_int> sc(2 * static_cast<size_t>(n), -1),
            sr(2 * static_cast<size_t>(n), -1);
        lapack_int mc = -1, mr = -1;
        const lapack_int ic =
            LAPACKE_CALL_REAL(T, stevr, LAPACK_COL_MAJOR, jobz, s.range, n,
                              dc.data(), ec.data(), s.vl, s.vu, s.il, s.iu,
                              R(0), &mc, wc.data(), zc.data(), ld, sc.data());
        const lapack_int ir =
            LAPACKE_CALL_REAL(T, stevr, LAPACK_ROW_MAJOR, jobz, s.range, n,
                              dr.data(), er.data(), s.vl, s.vu, s.il, s.iu,
                              R(0), &mr, wr.data(), zr.data(), ld, sr.data());
        ASSERT_EQ(ic, 0);
        ASSERT_EQ(ir, 0);
        ASSERT_EQ(mr, mc);
        EXPECT_EQ(wr, wc);
        if (jobz == 'V') {
          for (lapack_int i = 0; i < n; ++i) {
            for (lapack_int j = 0; j < mc; ++j) {
              ASSERT_EQ(zr[i * ld + j], At(zc, i, j, ld)) << i << ", " << j;
            }
          }
        }
        // ?stevx.
        dc = d0;
        ec = e0;
        dr = d0;
        er = e0;
        std::vector<lapack_int> fc(n, -1), fr(n, -1);
        const lapack_int xc =
            LAPACKE_CALL_REAL(T, stevx, LAPACK_COL_MAJOR, jobz, s.range, n,
                              dc.data(), ec.data(), s.vl, s.vu, s.il, s.iu,
                              R(0), &mc, wc.data(), zc.data(), ld, fc.data());
        const lapack_int xr =
            LAPACKE_CALL_REAL(T, stevx, LAPACK_ROW_MAJOR, jobz, s.range, n,
                              dr.data(), er.data(), s.vl, s.vu, s.il, s.iu,
                              R(0), &mr, wr.data(), zr.data(), ld, fr.data());
        ASSERT_EQ(xc, 0);
        ASSERT_EQ(xr, 0);
        ASSERT_EQ(mc, s.count);
        ASSERT_EQ(mr, mc);
        EXPECT_EQ(wr, wc);
        if (jobz == 'V') {
          EXPECT_EQ(fr, fc);
          for (lapack_int i = 0; i < n; ++i) {
            for (lapack_int j = 0; j < mc; ++j) {
              ASSERT_EQ(zr[i * ld + j], At(zc, i, j, ld)) << i << ", " << j;
            }
          }
        }
      }
    }
    // ?stebz (as the Fortran routine), then ?stein.
    lapack_int m = -1, nsplit = -1;
    std::vector<R> w(n);
    std::vector<lapack_int> iblock(n), isplit(n);
    lapack_int info;
    if constexpr (std::is_same_v<R, float>) {
      info = LAPACKE_sstebz(s.range, 'B', n, s.vl, s.vu, s.il, s.iu, R(0),
                            d0.data(), e0.data(), &m, &nsplit, w.data(),
                            iblock.data(), isplit.data());
    } else {
      info = LAPACKE_dstebz(s.range, 'B', n, s.vl, s.vu, s.il, s.iu, R(0),
                            d0.data(), e0.data(), &m, &nsplit, w.data(),
                            iblock.data(), isplit.data());
    }
    ASSERT_EQ(info, 0);
    {
      lapack_int fm = -1, fnsplit = -1, finfo = -99;
      std::vector<R> fw(n), work(4 * n);
      std::vector<lapack_int> fiblock(n), fisplit(n),
          iwork(3 * static_cast<size_t>(n));
      const char order = 'B';
      const R abstol = 0;
      HWYBLAS_CALL_REAL(R, stebz, &s.range, &order, &n, &s.vl, &s.vu, &s.il,
                        &s.iu, &abstol, d0.data(), e0.data(), &fm, &fnsplit,
                        fw.data(), fiblock.data(), fisplit.data(), work.data(),
                        iwork.data(), &finfo);
      EXPECT_EQ(finfo, 0);
      EXPECT_EQ(m, fm);
      EXPECT_EQ(nsplit, fnsplit);
      EXPECT_EQ(w, fw);
      EXPECT_EQ(iblock, fiblock);
    }
    if (m == 0) continue;
    std::vector<T> zc(ld * ld, NaN<T>()), zr(ld * ld, NaN<T>());
    std::vector<lapack_int> fc(n, -1), fr(n, -1);
    const lapack_int ic = LAPACKE_CALL(T, stein, LAPACK_COL_MAJOR, n, d0.data(),
                                       e0.data(), m, w.data(), iblock.data(),
                                       isplit.data(), zc.data(), ld, fc.data());
    const lapack_int ir = LAPACKE_CALL(T, stein, LAPACK_ROW_MAJOR, n, d0.data(),
                                       e0.data(), m, w.data(), iblock.data(),
                                       isplit.data(), zr.data(), ld, fr.data());
    ASSERT_EQ(ic, 0);
    ASSERT_EQ(ir, 0);
    for (lapack_int i = 0; i < n; ++i) {
      for (lapack_int j = 0; j < m; ++j) {
        ASSERT_EQ(zr[i * ld + j], At(zc, i, j, ld)) << i << ", " << j;
      }
    }
  }

  // ?larnv: as the Fortran routine.
  for (const lapack_int idist : {1, 2, 3}) {
    lapack_int s1[4] = {5, 6, 7, 9}, s2[4] = {5, 6, 7, 9};
    std::vector<T> x1(100), x2(100);
    const lapack_int nn = 100;
    EXPECT_EQ(LAPACKE_CALL(T, larnv, idist, s1, nn, x1.data()), 0);
    HWYBLAS_CALL(T, larnv, &idist, s2, &nn, P(x2.data()));
    EXPECT_EQ(x1, x2);
    for (int k = 0; k < 4; ++k) EXPECT_EQ(s1[k], s2[k]);
  }
}

// Invalid layouts and leading dimensions, and the NaN checks.
TYPED_TEST(SyevrTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const lapack_int n = 4;
  std::vector<T> a(16, T(0)), z(16, T(0));
  for (lapack_int i = 0; i < n; ++i) At(a, i, i, n) = T(1);
  std::vector<R> w(4), d(4, 1), e(4, 0);
  std::vector<lapack_int> isuppz(8), iblock = {1, 1, 1, 1}, isplit = {4},
                                     ifail(4);
  lapack_int m = -1;
  lapack_logical tryrac = 1;
  const R zero = 0, nan = NAN;
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevr, heevr, 0, 'V', 'A', 'U', n, a.data(), n,
                            zero, zero, 1, 1, zero, &m, w.data(), z.data(), n,
                            isuppz.data()),
            -1);
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevr, heevr, LAPACK_ROW_MAJOR, 'V', 'A', 'U', n,
                            a.data(), n - 1, zero, zero, 1, 1, zero, &m,
                            w.data(), z.data(), n, isuppz.data()),
            -7);
  // Row-major Z has iu-il+1 columns for RANGE = 'I'.
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevr, heevr, LAPACK_ROW_MAJOR, 'V', 'I', 'U', n,
                            a.data(), n, zero, zero, 1, 3, zero, &m, w.data(),
                            z.data(), 2, isuppz.data()),
            -16);
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevr, heevr, LAPACK_ROW_MAJOR, 'V', 'I', 'U', n,
                            a.data(), n, zero, zero, 1, 3, zero, &m, w.data(),
                            z.data(), 3, isuppz.data()),
            0);
  EXPECT_EQ(LAPACKE_CALL(T, stemr, LAPACK_ROW_MAJOR, 'V', 'A', n, d.data(),
                         e.data(), zero, zero, 1, 1, &m, w.data(), z.data(),
                         n - 1, n, isuppz.data(), &tryrac),
            -14);
  EXPECT_EQ(LAPACKE_CALL(T, stemr, LAPACK_ROW_MAJOR, 'N', 'A', n, d.data(),
                         e.data(), zero, zero, 1, 1, &m, w.data(), z.data(), 0,
                         n, isuppz.data(), &tryrac),
            -14);
  EXPECT_EQ(LAPACKE_CALL(T, stein, LAPACK_ROW_MAJOR, n, d.data(), e.data(), 3,
                         w.data(), iblock.data(), isplit.data(), z.data(), 2,
                         ifail.data()),
            -10);
  EXPECT_EQ(
      LAPACKE_CALL(T, stein, 0, n, d.data(), e.data(), 3, w.data(),
                   iblock.data(), isplit.data(), z.data(), n, ifail.data()),
      -1);
  std::string expected =
      "Wrong parameter 1 in " + LapackeName<T>(kComplex ? "heevr" : "syevr") +
      "\nWrong parameter 7 in " +
      LapackeName<T>(kComplex ? "heevr_work" : "syevr_work") +
      "\nWrong parameter 16 in " +
      LapackeName<T>(kComplex ? "heevr_work" : "syevr_work") +
      "\nWrong parameter 14 in " + LapackeName<T>("stemr_work") +
      "\nWrong parameter 14 in " + LapackeName<T>("stemr_work") +
      "\nWrong parameter 10 in " + LapackeName<T>("stein_work") +
      "\nWrong parameter 1 in " + LapackeName<T>("stein") + "\n";
  if constexpr (!kComplex) {
    // ?stevr checks LDZ against the columns of Z also for JOBZ = 'N'.
    EXPECT_EQ(LAPACKE_CALL_REAL(T, stevr, LAPACK_ROW_MAJOR, 'N', 'A', n,
                                d.data(), e.data(), zero, zero, 1, 1, zero, &m,
                                w.data(), z.data(), n - 1, isuppz.data()),
              -15);
    expected += "Wrong parameter 15 in " + LapackeName<T>("stevr_work") + "\n";
  }
  // ?syevx / ?heevx and ?stevx (as ?syevr and ?stevr).
  const char* evx = kComplex ? "heevx" : "syevx";
  const char* evx_work = kComplex ? "heevx_work" : "syevx_work";
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevx, heevx, 0, 'V', 'A', 'U', n, a.data(), n,
                            zero, zero, 1, 1, zero, &m, w.data(), z.data(), n,
                            ifail.data()),
            -1);
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_ROW_MAJOR, 'V', 'A', 'U', n,
                            a.data(), n - 1, zero, zero, 1, 1, zero, &m,
                            w.data(), z.data(), n, ifail.data()),
            -7);
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_ROW_MAJOR, 'V', 'I', 'U', n,
                            a.data(), n, zero, zero, 1, 3, zero, &m, w.data(),
                            z.data(), 2, ifail.data()),
            -16);
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_ROW_MAJOR, 'V', 'I', 'U', n,
                            a.data(), n, zero, zero, 1, 3, zero, &m, w.data(),
                            z.data(), 3, ifail.data()),
            0);
  // LDZ = 1 is enough for JOBZ = 'N'.
  EXPECT_EQ(LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_ROW_MAJOR, 'N', 'A', 'U', n,
                            a.data(), n, zero, zero, 1, 1, zero, &m, w.data(),
                            z.data(), 1, ifail.data()),
            0);
  expected += "Wrong parameter 1 in " + LapackeName<T>(evx) +
              "\nWrong parameter 7 in " + LapackeName<T>(evx_work) +
              "\nWrong parameter 16 in " + LapackeName<T>(evx_work) + "\n";
  if constexpr (!kComplex) {
    EXPECT_EQ(LAPACKE_CALL_REAL(T, stevx, LAPACK_ROW_MAJOR, 'N', 'A', n,
                                d.data(), e.data(), zero, zero, 1, 1, zero, &m,
                                w.data(), z.data(), n - 1, ifail.data()),
              -15);
    EXPECT_EQ(LAPACKE_CALL_REAL(T, stevx, 0, 'N', 'A', n, d.data(), e.data(),
                                zero, zero, 1, 1, zero, &m, w.data(), z.data(),
                                n, ifail.data()),
              -1);
    expected += "Wrong parameter 15 in " + LapackeName<T>("stevx_work") +
                "\nWrong parameter 1 in " + LapackeName<T>("stevx") + "\n";
  }
  EXPECT_EQ(::testing::internal::GetCapturedStdout(), expected);

  // A workspace query of ?syevx_work / ?heevx_work in row-major layout
  // gives that of the Fortran routine.
  {
    T lapacke_query = T(0), fortran_query = T(0);
    std::vector<R> rwork(7 * n);
    std::vector<lapack_int> iwork(5 * static_cast<size_t>(n));
    const lapack_int m1 = -1, ldz = n, il = 1, iu = 1;
    lapack_int info = -99;
    const char jobz = 'V', range = 'A', uplo = 'U';
    if constexpr (kComplex) {
      EXPECT_EQ(
          LAPACKE_CALL_RC(T, syevx_work, heevx_work, LAPACK_ROW_MAJOR, jobz,
                          range, uplo, n, a.data(), n, zero, zero, il, iu, zero,
                          &m, w.data(), z.data(), n, &lapacke_query, m1,
                          rwork.data(), iwork.data(), ifail.data()),
          0);
      HWYBLAS_CALL_COMPLEX(T, heevx, &jobz, &range, &uplo, &n, P(a.data()), &n,
                           &zero, &zero, &il, &iu, &zero, &m, w.data(),
                           P(z.data()), &ldz, P(&fortran_query), &m1,
                           rwork.data(), iwork.data(), ifail.data(), &info);
    } else {
      EXPECT_EQ(LAPACKE_CALL_RC(T, syevx_work, heevx_work, LAPACK_ROW_MAJOR,
                                jobz, range, uplo, n, a.data(), n, zero, zero,
                                il, iu, zero, &m, w.data(), z.data(), n,
                                &lapacke_query, m1, iwork.data(), ifail.data()),
                0);
      HWYBLAS_CALL_REAL(T, syevx, &jobz, &range, &uplo, &n, a.data(), &n, &zero,
                        &zero, &il, &iu, &zero, &m, w.data(), z.data(), &ldz,
                        &fortran_query, &m1, iwork.data(), ifail.data(), &info);
    }
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(lapacke_query), QueryResult(fortran_query));
  }

  // NaN checks.
  const auto syevr = [&](char range, R vl, R vu, R abstol) {
    return LAPACKE_CALL_RC(T, syevr, heevr, LAPACK_COL_MAJOR, 'N', range, 'U',
                           n, a.data(), n, vl, vu, 1, 1, abstol, &m, w.data(),
                           z.data(), n, isuppz.data());
  };
  const auto syevx = [&](char range, R vl, R vu, R abstol) {
    return LAPACKE_CALL_RC(T, syevx, heevx, LAPACK_COL_MAJOR, 'N', range, 'U',
                           n, a.data(), n, vl, vu, 1, 1, abstol, &m, w.data(),
                           z.data(), n, ifail.data());
  };
  EXPECT_EQ(syevr('A', nan, nan, zero), 0);
  EXPECT_EQ(syevr('V', nan, R(1), zero), -8);
  EXPECT_EQ(syevr('V', zero, nan, zero), -9);
  EXPECT_EQ(syevr('A', zero, zero, nan), -12);
  EXPECT_EQ(syevx('A', nan, nan, zero), 0);
  EXPECT_EQ(syevx('V', nan, R(1), zero), -8);
  EXPECT_EQ(syevx('V', zero, nan, zero), -9);
  EXPECT_EQ(syevx('A', zero, zero, nan), -12);
  a[1] = NaN<T>();  // (1, 0): lower triangle
  EXPECT_EQ(syevr('A', zero, zero, zero), 0);
  EXPECT_EQ(syevx('A', zero, zero, zero), 0);
  a[n] = NaN<T>();  // (0, 1): upper triangle
  EXPECT_EQ(syevr('A', zero, zero, zero), -6);
  EXPECT_EQ(syevx('A', zero, zero, zero), -6);
  const auto stemr = [&](R vl, R vu) {
    return LAPACKE_CALL(T, stemr, LAPACK_COL_MAJOR, 'N', 'A', n, d.data(),
                        e.data(), vl, vu, 1, 1, &m, w.data(), z.data(), n, n,
                        isuppz.data(), &tryrac);
  };
  EXPECT_EQ(stemr(nan, zero), -7);  // also for RANGE = 'A'
  EXPECT_EQ(stemr(zero, nan), -8);
  const auto stein = [&]() {
    return LAPACKE_CALL(T, stein, LAPACK_COL_MAJOR, n, d.data(), e.data(), 1,
                        w.data(), iblock.data(), isplit.data(), z.data(), n,
                        ifail.data());
  };
  w = {1, 1, 1, NAN};
  EXPECT_EQ(stein(), -6);
  w = {1, 1, 1, 1};
  e[2] = NAN;
  EXPECT_EQ(stein(), -4);
  EXPECT_EQ(stemr(zero, zero), -6);
  e[2] = 0;
  e[3] = NAN;  // not checked (n-1 elements)
  EXPECT_EQ(stemr(zero, zero), 0);
  e[3] = 0;
  d[3] = NAN;
  EXPECT_EQ(stein(), -3);
  EXPECT_EQ(stemr(zero, zero), -5);
  d[3] = 1;
  if constexpr (!kComplex) {
    const auto stevr = [&](char range, R vl, R vu, R abstol) {
      return LAPACKE_CALL_REAL(T, stevr, LAPACK_COL_MAJOR, 'N', range, n,
                               d.data(), e.data(), vl, vu, 1, 1, abstol, &m,
                               w.data(), z.data(), n, isuppz.data());
    };
    const auto stebz = [&](char range, R vl, R vu, R abstol) {
      lapack_int nsplit;
      std::vector<lapack_int> blocks(n), splits(n);
      return LAPACKE_CALL_REAL(T, stebz, range, 'E', n, vl, vu, 1, 1, abstol,
                               d.data(), e.data(), &m, &nsplit, w.data(),
                               blocks.data(), splits.data());
    };
    const auto stevx = [&](char range, R vl, R vu, R abstol) {
      return LAPACKE_CALL_REAL(T, stevx, LAPACK_COL_MAJOR, 'N', range, n,
                               d.data(), e.data(), vl, vu, 1, 1, abstol, &m,
                               w.data(), z.data(), n, ifail.data());
    };
    EXPECT_EQ(stevr('A', nan, nan, zero), 0);
    EXPECT_EQ(stevr('V', nan, R(1), zero), -7);
    EXPECT_EQ(stevr('V', zero, nan, zero), -8);
    EXPECT_EQ(stevr('A', zero, zero, nan), -11);
    EXPECT_EQ(stevx('A', nan, nan, zero), 0);
    EXPECT_EQ(stevx('V', nan, R(1), zero), -7);
    EXPECT_EQ(stevx('V', zero, nan, zero), -8);
    EXPECT_EQ(stevx('A', zero, zero, nan), -11);
    EXPECT_EQ(stebz('A', nan, nan, zero), 0);
    EXPECT_EQ(stebz('V', nan, R(1), zero), -4);
    EXPECT_EQ(stebz('V', zero, nan, zero), -5);
    EXPECT_EQ(stebz('A', zero, zero, nan), -8);
    d[0] = NAN;
    EXPECT_EQ(stevr('A', zero, zero, zero), -5);
    EXPECT_EQ(stevx('A', zero, zero, zero), -5);
    EXPECT_EQ(stebz('A', zero, zero, zero), -9);
    d[0] = 1;
    e[0] = NAN;
    EXPECT_EQ(stevr('A', zero, zero, zero), -6);
    EXPECT_EQ(stevx('A', zero, zero, zero), -6);
    EXPECT_EQ(stebz('A', zero, zero, zero), -10);
  }
}

}  // namespace
}  // namespace hwyblas_test
