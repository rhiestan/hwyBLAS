// Tests of the symmetric (Hermitian) eigenvalue routines: xSYEV / xHEEV,
// the reduction to tridiagonal form (xSYTRD / xHETRD, xSYTD2 / xHETD2,
// xLATRD) and its Q (xORGTR / xUNGTR, xORGQL / xUNGQL, xORG2L / xUNG2L),
// the tridiagonal solvers xSTEQR and xSTERF and their auxiliary routines
// (xLAE2, xLAEV2, xLARTG, xLASR, xLANST, xLASRT), and LAPACKE.
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
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class SyevTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SyevTest, ScalarTypes);

template <typename R>
class SyevRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SyevRealTest, RealTypes);

// Calls the real routine `rname` or the complex routine `cname`.
#define CALL_RC(T, rname, cname, ...)                              \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      s##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, double>) {              \
      d##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      c##cname##_(__VA_ARGS__);                                    \
    } else {                                                       \
      z##cname##_(__VA_ARGS__);                                    \
    }                                                              \
  } while (0)

// A random symmetric (Hermitian) n x n matrix in the triangle uplo of a
// (leading dimension lda), with NaN elsewhere and in the imaginary parts of
// the diagonal.
template <typename T>
std::vector<T> RandomHermitian(char uplo, hwyblas_int n, hwyblas_int lda,
                               Random<T>& rng) {
  return StructuredMatrix(
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric, uplo, n, lda,
      rng);
}

template <typename T>
Dense<T> HermitianDense(char uplo, const std::vector<T>& a, hwyblas_int n,
                        hwyblas_int lda) {
  return DenseStructured(
      IsComplex<T>::value ? Shape::kHermitian : Shape::kSymmetric, uplo, 'N', a,
      n, lda);
}

// The tridiagonal matrix (d, e) as a dense matrix.
template <typename T>
Dense<T> Tridiagonal(const std::vector<RealT<T>>& d,
                     const std::vector<RealT<T>>& e) {
  const size_t n = d.size();
  Dense<T> t(n, n);
  for (size_t i = 0; i < n; ++i) {
    t(i, i) = d[i];
    if (i + 1 < n) t(i + 1, i) = t(i, i + 1) = e[i];
  }
  return t;
}

// ||A * Z - Z * diag(w)||_1 / (n ||A||_1 eps) (LAPACK's xSYT21 ratio, for
// eigenvectors in z).
template <typename T>
double EigenResidual(const Dense<T>& a, const Dense<T>& z,
                     const std::vector<RealT<T>>& w) {
  Dense<T> zw = z;
  for (size_t j = 0; j < z.cols; ++j) {
    for (size_t i = 0; i < z.rows; ++i) zw(i, j) *= static_cast<double>(w[j]);
  }
  const double anorm = Norm1(a);
  const double resid = Norm1(Sub(Mul(a, z), zw));
  if (anorm == 0) return resid / LapackEps<T>();
  return resid / static_cast<double>(a.rows) / anorm / LapackEps<T>();
}

// Eigenvalues in ascending order.
template <typename R>
void ExpectAscending(const std::vector<R>& w) {
  for (size_t i = 1; i < w.size(); ++i) ASSERT_LE(w[i - 1], w[i]) << i;
}

struct Blocking {
  hwyblas_int nb, nx;
};
const Blocking kBlockings[] = {{0, 0}, {1, 0}, {2, 0}, {3, 5}, {8, 0}, {8, 6}};

// xSYTRD / xHETRD with xORGTR / xUNGTR: A = Q * T * Q**H with orthogonal
// (unitary) Q, with blocked and unblocked code and reduced workspaces.
TYPED_TEST(SyevTest, Tridiagonalization) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(79);
    for (const Blocking& b : kBlockings) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {1, 2, 3, 9, 20, 41}) {
          for (const bool minimal : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "uplo=" << uplo << " n=" << n << " nb=" << b.nb
                         << " nx=" << b.nx << " minimal=" << minimal);
            const hwyblas_int lda = n + 2;
            const std::vector<T> a = RandomHermitian(uplo, n, lda, rng);
            std::vector<T> af = a;
            std::vector<R> d(n), e(std::max<hwyblas_int>(1, n - 1));
            std::vector<T> tau(std::max<hwyblas_int>(1, n - 1));
            hwyblas_int lwork = minimal ? 1 : n * 64, info = -99;
            std::vector<T> work(lwork, NaN<T>());
            CALL_RC(T, sytrd, hetrd, &uplo, &n, P(af.data()), &lda, d.data(),
                    e.data(), P(tau.data()), P(work.data()), &lwork, &info);
            ASSERT_EQ(info, 0);
            e.resize(n - 1);
            // The diagonal and off-diagonal are stored in A too. (Of the
            // diagonal, the real parts: as in LAPACK, the blocked upper
            // reduction leaves the imaginary part of A(n, n), which is
            // assumed to be zero, as it was.)
            for (hwyblas_int i = 0; i < n; ++i) {
              ASSERT_EQ(std::real(At(af, i, i, lda)), d[i]);
              if (i + 1 < n) {
                const T off =
                    uplo == 'U' ? At(af, i, i + 1, lda) : At(af, i + 1, i, lda);
                ASSERT_EQ(off, T(e[i]));
              }
            }
            // Q, with the minimum workspace or the optimal one.
            lwork = minimal ? std::max<hwyblas_int>(1, n - 1) : n * 64;
            work.assign(lwork, NaN<T>());
            CALL_RC(T, orgtr, ungtr, &uplo, &n, P(af.data()), &lda,
                    P(tau.data()), P(work.data()), &lwork, &info);
            ASSERT_EQ(info, 0);
            const Dense<T> q = ToDense(af, n, n, lda);
            EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);
            const Dense<T> qtq = Mul(Mul(q, Tridiagonal<T>(d, e)), Adjoint(q));
            EXPECT_LT(ResidualRatio(HermitianDense(uplo, a, n, lda), qtq, n),
                      kLapackThreshold);
          }
        }
      }
    }
  });
}

template <typename T>
hwyblas_int Steqr(char compz, hwyblas_int n, std::vector<RealT<T>>& d,
                  std::vector<RealT<T>>& e, std::vector<T>& z,
                  hwyblas_int ldz) {
  std::vector<RealT<T>> work(std::max<hwyblas_int>(1, 2 * n - 2), NAN);
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, steqr, &compz, &n, d.data(), e.data(), P(z.data()), &ldz,
               work.data(), &info);
  return info;
}

// A symmetric tridiagonal matrix of one of several kinds: random, with
// zeros on the off-diagonal (it splits), graded, with clustered
// eigenvalues (the Wilkinson matrix), or scaled near the overflow or the
// underflow threshold (xSTEQR and xSTERF then scale it).
template <typename R>
void TestTridiagonal(int kind, hwyblas_int n, Random<R>& rng, std::vector<R>& d,
                     std::vector<R>& e) {
  d.resize(n);
  e.resize(std::max<hwyblas_int>(0, n - 1));
  for (auto& x : d) x = rng.Next();
  for (auto& x : e) x = rng.Next();
  switch (kind) {
    case 1:  // splits
      for (hwyblas_int i = 0; i + 1 < n; i += 3) e[i] = 0;
      break;
    case 2:  // graded
      for (hwyblas_int i = 0; i < n; ++i) {
        d[i] *= static_cast<R>(std::pow(10.0, -i % 8));
        if (i + 1 < n) e[i] *= static_cast<R>(std::pow(10.0, -i % 8));
      }
      break;
    case 3: {  // Wilkinson: d = |mid - i|, e = 1
      const hwyblas_int mid = n / 2;
      for (hwyblas_int i = 0; i < n; ++i) d[i] = std::abs(R(mid - i));
      for (auto& x : e) x = 1;
      break;
    }
    case 4:    // near overflow
    case 5: {  // near underflow
      const R scale = kind == 4 ? std::sqrt(std::numeric_limits<R>::max()) * 4
                                : std::sqrt(std::numeric_limits<R>::min()) / 4;
      for (auto& x : d) x *= scale;
      for (auto& x : e) x *= scale;
      break;
    }
    default:
      break;
  }
}

// xSTEQR ('I', 'V', 'N') and xSTERF: eigenvalues in ascending order,
// eigenvectors with small residuals that are orthonormal, and the same
// eigenvalues without eigenvectors.
TYPED_TEST(SyevTest, Steqr) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<R> rng(83);
  Random<T> crng(89);
  for (const int kind : {0, 1, 2, 3, 4, 5}) {
    for (const hwyblas_int n : {1, 2, 3, 10, 33}) {
      SCOPED_TRACE(::testing::Message() << "kind=" << kind << " n=" << n);
      std::vector<R> d0, e0;
      TestTridiagonal(kind, n, rng, d0, e0);
      // Dense T in scaled form (the residual ratios are scale invariant).
      std::vector<R> ds = d0, es = e0;
      R anorm = 0;
      for (R x : d0) anorm = std::max(anorm, std::abs(x));
      for (R x : e0) anorm = std::max(anorm, std::abs(x));
      for (auto& x : ds) x /= anorm;
      for (auto& x : es) x /= anorm;
      const Dense<T> t = Tridiagonal<T>(ds, es);

      // 'I': eigenvectors of T.
      std::vector<R> d = d0, e = e0;
      const hwyblas_int ldz = n + 1;
      std::vector<T> z(ldz * n, NaN<T>());
      ASSERT_EQ(Steqr('I', n, d, e, z, ldz), 0);
      ExpectAscending(d);
      std::vector<R> w = d;
      for (auto& x : w) x /= anorm;
      const Dense<T> zd = ToDense(z, n, n, ldz);
      EXPECT_LT(OrthogonalityRatio(zd), kLapackThreshold);
      EXPECT_LT(EigenResidual(t, zd, w), kLapackThreshold);

      // 'V': Z = Q * (eigenvectors of T) for orthogonal (unitary) Q, here
      // the Q of a QR factorization of a random matrix.
      std::vector<T> q = crng.Vector(ldz * n);
      std::vector<T> tau(n);
      hwyblas_int info = -99, lwork = 64 * n;
      std::vector<T> work(lwork);
      HWYBLAS_CALL(T, geqrf, &n, &n, P(q.data()), &ldz, P(tau.data()),
                   P(work.data()), &lwork, &info);
      CALL_RC(T, orgqr, ungqr, &n, &n, &n, P(q.data()), &ldz, P(tau.data()),
              P(work.data()), &lwork, &info);
      const Dense<T> qd = ToDense(q, n, n, ldz);
      std::vector<R> dv = d0, ev = e0;
      ASSERT_EQ(Steqr('V', n, dv, ev, q, ldz), 0);
      EXPECT_EQ(dv, d);
      const Dense<T> qz = ToDense(q, n, n, ldz);
      // Q**H * (Q * Z) are the eigenvectors of T.
      EXPECT_LT(EigenResidual(t, Mul(Adjoint(qd), qz), w), kLapackThreshold);
      EXPECT_LT(OrthogonalityRatio(qz), kLapackThreshold);

      // 'N' and xSTERF: the same eigenvalues (up to rounding).
      std::vector<R> dn = d0, en = e0, df = d0, ef = e0;
      std::vector<T> dummy(1);
      ASSERT_EQ(Steqr('N', n, dn, en, dummy, 1), 0);
      if constexpr (!IsComplex<T>::value) {
        hwyblas_int sinfo = -99;
        HWYBLAS_CALL_REAL(T, sterf, &n, df.data(), ef.data(), &sinfo);
        ASSERT_EQ(sinfo, 0);
      } else {
        df = dn;
      }
      for (hwyblas_int i = 0; i < n; ++i) {
        const double tol =
            30.0 * n * LapackEps<T>() * static_cast<double>(anorm);
        EXPECT_NEAR(dn[i], d[i], tol) << i;
        EXPECT_NEAR(df[i], d[i], tol) << i;
      }
    }
  }
}

template <typename T>
hwyblas_int Heev(char jobz, char uplo, hwyblas_int n, std::vector<T>& a,
                 hwyblas_int lda, std::vector<RealT<T>>& w, hwyblas_int lwork) {
  std::vector<RealT<T>> rwork(std::max<hwyblas_int>(1, 3 * n - 2), NAN);
  hwyblas_int info = -99;
  if (lwork == -1) {
    std::vector<T> query(1);
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heev, &jobz, &uplo, &n, P(a.data()), &lda,
                           w.data(), P(query.data()), &lwork, rwork.data(),
                           &info);
    } else {
      HWYBLAS_CALL_REAL(T, syev, &jobz, &uplo, &n, P(a.data()), &lda, w.data(),
                        P(query.data()), &lwork, &info);
    }
    if (info != 0) return info;
    lwork = static_cast<hwyblas_int>(std::real(query[0]));
  }
  std::vector<T> work(std::max<hwyblas_int>(1, lwork), NaN<T>());
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heev, &jobz, &uplo, &n, P(a.data()), &lda, w.data(),
                         P(work.data()), &lwork, rwork.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, syev, &jobz, &uplo, &n, P(a.data()), &lda, w.data(),
                      P(work.data()), &lwork, &info);
  }
  return info;
}

template <typename T>
hwyblas_int MinimumHeevLwork(hwyblas_int n) {
  return std::max<hwyblas_int>(1, IsComplex<T>::value ? 2 * n - 1 : 3 * n - 1);
}

// xSYEV / xHEEV: A * V = V * diag(w) with orthonormal V and ascending w,
// the same w without eigenvectors, also for matrices that the driver
// scales (norm near the underflow or overflow threshold).
TYPED_TEST(SyevTest, Heev) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(97);
    const double rmin = std::sqrt(static_cast<double>(
        std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon()));
    for (const Blocking& b : {Blocking{0, 0}, Blocking{3, 5}}) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {1, 2, 5, 17, 40}) {
          for (const double scale : {1.0, rmin * 1e-3, 1 / (rmin * 1e-3)}) {
            for (const bool minimal : {false, true}) {
              SCOPED_TRACE(::testing::Message()
                           << "uplo=" << uplo << " n=" << n << " nb=" << b.nb
                           << " scale=" << scale << " minimal=" << minimal);
              const hwyblas_int lda = n + 1;
              std::vector<T> a = RandomHermitian(uplo, n, lda, rng);
              const Dense<T> ad = HermitianDense(uplo, a, n, lda);
              for (T& x : a) x = x * static_cast<R>(scale);
              const hwyblas_int lwork = minimal ? MinimumHeevLwork<T>(n) : -1;
              std::vector<T> v = a;
              std::vector<R> w(n);
              ASSERT_EQ(Heev('V', uplo, n, v, lda, w, lwork), 0);
              ExpectAscending(w);
              for (R& x : w) x = static_cast<R>(x / scale);
              const Dense<T> vd = ToDense(v, n, n, lda);
              EXPECT_LT(OrthogonalityRatio(vd), kLapackThreshold);
              EXPECT_LT(EigenResidual(ad, vd, w), kLapackThreshold);

              std::vector<T> an = a;
              std::vector<R> wn(n);
              ASSERT_EQ(Heev('N', uplo, n, an, lda, wn, lwork), 0);
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
  });
}

TYPED_TEST(SyevTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  const ScopedBlockSize block_size(8);
  const hwyblas_int n = 10;
  std::vector<T> a(n * n);
  std::vector<R> w(n);
  hwyblas_int lwork = -1, info = -99;
  std::vector<T> work(1);
  std::vector<R> rwork(3 * n);
  const char jobz = 'V', uplo = 'L';
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heev, &jobz, &uplo, &n, P(a.data()), &n, w.data(),
                         P(work.data()), &lwork, rwork.data(), &info);
    EXPECT_EQ(std::real(work[0]), (8 + 1) * n);
  } else {
    HWYBLAS_CALL_REAL(T, syev, &jobz, &uplo, &n, P(a.data()), &n, w.data(),
                      P(work.data()), &lwork, &info);
    EXPECT_EQ(work[0], (8 + 2) * n);
  }
  EXPECT_EQ(info, 0);
  std::vector<R> d(n), e(n);
  std::vector<T> tau(n);
  CALL_RC(T, sytrd, hetrd, &uplo, &n, P(a.data()), &n, d.data(), e.data(),
          P(tau.data()), P(work.data()), &lwork, &info);
  EXPECT_EQ(std::real(work[0]), n * 8);
  CALL_RC(T, orgtr, ungtr, &uplo, &n, P(a.data()), &n, P(tau.data()),
          P(work.data()), &lwork, &info);
  EXPECT_EQ(std::real(work[0]), (n - 1) * 8);

  // Invalid arguments.
  ErrorCapture errors;
  const std::string ev = Name<T>(IsComplex<T>::value ? "HEEV" : "SYEV");
  std::vector<R> w1(n);
  EXPECT_EQ(Heev<T>('X', 'L', n, a, n, w1, 64), -1);
  EXPECT_EQ(Heev<T>('V', 'X', n, a, n, w1, 64), -2);
  EXPECT_EQ(Heev<T>('V', 'L', -1, a, n, w1, 64), -3);
  EXPECT_EQ(Heev<T>('V', 'L', n, a, n - 1, w1, 64), -5);
  EXPECT_EQ(Heev<T>('V', 'L', n, a, n, w1, MinimumHeevLwork<T>(n) - 1), -8);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{ev + ":1", ev + ":2", ev + ":3",
                                      ev + ":5", ev + ":8"}));
  std::vector<R> dd(n), ee(n);
  std::vector<T> z(n * n);
  EXPECT_EQ(Steqr<T>('X', n, dd, ee, z, n), -1);
  EXPECT_EQ(Steqr<T>('I', -1, dd, ee, z, n), -2);
  EXPECT_EQ(Steqr<T>('I', n, dd, ee, z, n - 1), -6);
  EXPECT_EQ(Steqr<T>('N', n, dd, ee, z, 0), -6);
  const std::string steqr = Name<T>("STEQR");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{steqr + ":1", steqr + ":2", steqr + ":6",
                                      steqr + ":6"}));
}

// ---------------------------------------------------------------------------
// Auxiliary routines

TYPED_TEST(SyevRealTest, Lae2Laev2) {
  using R = TypeParam;
  Random<R> rng(101);
  for (int t = 0; t < 200; ++t) {
    R a = rng.Next(), b = rng.Next(), c = rng.Next();
    if (t % 10 == 1) b = 0;
    if (t % 10 == 2) c = a;
    if (t % 10 == 3) c = -a;
    R rt1, rt2, rt1v, rt2v, cs, sn;
    if constexpr (std::is_same_v<R, float>) {
      slae2_(&a, &b, &c, &rt1, &rt2);
      slaev2_(&a, &b, &c, &rt1v, &rt2v, &cs, &sn);
    } else {
      dlae2_(&a, &b, &c, &rt1, &rt2);
      dlaev2_(&a, &b, &c, &rt1v, &rt2v, &cs, &sn);
    }
    EXPECT_EQ(rt1, rt1v);
    EXPECT_EQ(rt2, rt2v);
    EXPECT_GE(std::abs(rt1), std::abs(rt2));
    // Exact eigenvalues.
    const double da = a, db = b, dc = c;
    const double disc = std::sqrt((da - dc) * (da - dc) / 4 + db * db);
    const double e1 = (da + dc) / 2 + disc, e2 = (da + dc) / 2 - disc;
    const double big = std::abs(e1) >= std::abs(e2) ? e1 : e2;
    const double scale = std::max({std::abs(da), std::abs(db), std::abs(dc)});
    EXPECT_NEAR(rt1, big, 4 * Eps<R>() * scale);
    EXPECT_NEAR(static_cast<double>(rt1) + rt2, da + dc, 8 * Eps<R>() * scale);
    // (cs, sn) is a unit eigenvector for rt1.
    EXPECT_NEAR(static_cast<double>(cs) * cs + static_cast<double>(sn) * sn,
                1.0, 4 * Eps<R>());
    EXPECT_NEAR(da * cs + db * sn, static_cast<double>(rt1) * cs,
                8 * Eps<R>() * scale);
    EXPECT_NEAR(db * cs + dc * sn, static_cast<double>(rt1) * sn,
                8 * Eps<R>() * scale);
  }
}

TYPED_TEST(SyevRealTest, Lartg) {
  using R = TypeParam;
  const R big = std::numeric_limits<R>::max() / 4;
  const R tiny = std::numeric_limits<R>::min() * 4;
  struct Case {
    R f, g;
  };
  const Case cases[] = {{3, 4},       {-3, 4},  {3, -4},   {0, 2},
                        {0, -2},      {5, 0},   {-5, 0},   {big, big},
                        {tiny, tiny}, {big, 1}, {1, tiny}, {tiny, big}};
  for (const Case& t : cases) {
    SCOPED_TRACE(::testing::Message() << t.f << " " << t.g);
    R c, s, r;
    R f = t.f, g = t.g;
    if constexpr (std::is_same_v<R, float>) {
      slartg_(&f, &g, &c, &s, &r);
    } else {
      dlartg_(&f, &g, &c, &s, &r);
    }
    EXPECT_GE(c, 0);
    EXPECT_NEAR(static_cast<double>(c) * c + static_cast<double>(s) * s, 1.0,
                4 * Eps<R>());
    // [c s; -s c] [f; g] = [r; 0], in scaled form.
    const double scale = std::max(std::abs(double(t.f)), std::abs(double(t.g)));
    const double fs = t.f / scale, gs = t.g / scale;
    EXPECT_NEAR(c * fs + s * gs, r / scale, 4 * Eps<R>());
    EXPECT_NEAR(-s * fs + c * gs, 0.0, 4 * Eps<R>());
    if (t.f != 0) EXPECT_EQ(std::signbit(r), std::signbit(t.f));
  }
}

TYPED_TEST(SyevRealTest, LanstLasrt) {
  using R = TypeParam;
  Random<R> rng(103);
  const hwyblas_int n = 9;
  std::vector<R> d = rng.Vector(n), e = rng.Vector(n - 1);
  const Dense<R> t = Tridiagonal<R>(d, e);
  const auto lanst = [&](char norm) {
    if constexpr (std::is_same_v<R, float>) {
      return slanst_(&norm, &n, d.data(), e.data());
    } else {
      return dlanst_(&norm, &n, d.data(), e.data());
    }
  };
  double maxabs = 0, frob = 0;
  for (const auto& x : t.v) {
    maxabs = std::max(maxabs, std::abs(x));
    frob += x * x;
  }
  EXPECT_NEAR(lanst('M'), maxabs, 2 * Eps<R>() * maxabs);
  EXPECT_NEAR(lanst('1'), Norm1(t), 4 * n * Eps<R>() * Norm1(t));
  EXPECT_NEAR(lanst('I'), Norm1(t), 4 * n * Eps<R>() * Norm1(t));
  EXPECT_NEAR(lanst('F'), std::sqrt(frob), 4 * n * Eps<R>() * std::sqrt(frob));

  // Sorting: insertion sort (n <= 20) and quick sort, with duplicates.
  for (const hwyblas_int m : {0, 1, 2, 7, 20, 21, 100, 1000}) {
    for (const char id : {'I', 'D'}) {
      std::vector<R> x = rng.Vector(std::max<hwyblas_int>(1, m));
      x.resize(m);
      for (hwyblas_int i = 0; i < m; i += 7) x[i] = R(0.5);  // duplicates
      std::vector<R> sorted = x;
      std::sort(sorted.begin(), sorted.end());
      if (id == 'D') std::reverse(sorted.begin(), sorted.end());
      hwyblas_int info = -99;
      HWYBLAS_CALL_REAL(R, lasrt, &id, &m, x.data(), &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(x, sorted) << m << " " << id;
    }
  }
}

// xLASR against the product of the rotations formed explicitly.
TYPED_TEST(SyevTest, Lasr) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(107);
  Random<R> rrng(109);
  for (const char side : {'L', 'R'}) {
    for (const char pivot : {'V', 'T', 'B'}) {
      for (const char direct : {'F', 'B'}) {
        const hwyblas_int m = 6, n = 5, lda = 7;
        const hwyblas_int z = side == 'L' ? m : n;
        std::vector<R> c(z - 1), s(z - 1);
        for (hwyblas_int k = 0; k < z - 1; ++k) {
          const R angle = rrng.Next() * R(3);
          c[k] = std::cos(angle);
          s[k] = std::sin(angle);
        }
        c[1] = 1;  // an identity rotation, which xLASR skips
        s[1] = 0;
        // P = P(z-1) ... P(1) (forward) or P(1) ... P(z-1) (backward).
        Dense<T> p = Identity<T>(z);
        for (hwyblas_int k = 0; k < z - 1; ++k) {
          // The plane of rotation k.
          hwyblas_int i1 = k, i2 = k + 1;
          if (pivot == 'T') i1 = 0, i2 = k + 1;
          if (pivot == 'B') i1 = k, i2 = z - 1;
          Dense<T> pk = Identity<T>(z);
          // P(k) = [c s; -s c] in the plane (i1, i2).
          pk(i1, i1) = c[k];
          pk(i2, i2) = c[k];
          pk(i1, i2) = s[k];
          pk(i2, i1) = -s[k];
          p = direct == 'F' ? Mul(pk, p) : Mul(p, pk);
        }
        std::vector<T> a = rng.Vector(lda * n);
        const Dense<T> a0 = ToDense(a, m, n, lda);
        HWYBLAS_CALL(T, lasr, &side, &pivot, &direct, &m, &n, c.data(),
                     s.data(), P(a.data()), &lda);
        const Dense<T> expected =
            side == 'L' ? Mul(p, a0) : Mul(a0, Adjoint(p));
        EXPECT_LT(ResidualRatio(expected, ToDense(a, m, n, lda), z),
                  kLapackThreshold)
            << side << pivot << direct;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
lapack_int LapackeHeev(int layout, char jobz, char uplo, lapack_int n, T* a,
                       lapack_int lda, RealT<T>* w) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_ssyev(layout, jobz, uplo, n, a, lda, w);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dsyev(layout, jobz, uplo, n, a, lda, w);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_cheev(layout, jobz, uplo, n, a, lda, w);
  } else {
    return LAPACKE_zheev(layout, jobz, uplo, n, a, lda, w);
  }
}

template <typename T>
lapack_int LapackeSteqr(int layout, char compz, lapack_int n, RealT<T>* d,
                        RealT<T>* e, T* z, lapack_int ldz) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_ssteqr(layout, compz, n, d, e, z, ldz);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dsteqr(layout, compz, n, d, e, z, ldz);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_csteqr(layout, compz, n, d, e, z, ldz);
  } else {
    return LAPACKE_zsteqr(layout, compz, n, d, e, z, ldz);
  }
}

// Row-major LAPACKE calls give the same results as column-major ones (the
// same arithmetic, only with other leading dimensions); for jobz = 'N', A
// keeps the elements outside the triangle uplo.
// xLASR from the left processes the columns in blocks (see there): the
// result is identical to rotating the columns of A**T from the right,
// which applies the same operations to each element in the same order.
// Sizes with a partial block, and identity rotations.
TYPED_TEST(SyevTest, LasrLeftIsRightOfTranspose) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(111);
  Random<R> rrng(113);
  for (const char pivot : {'V', 'T', 'B'}) {
    for (const char direct : {'F', 'B'}) {
      for (const hwyblas_int n : {1, 7, 8, 21}) {
        const hwyblas_int m = 13, lda = m + 2, ldt = n + 1;
        std::vector<R> c(m - 1), s(m - 1);
        for (hwyblas_int k = 0; k < m - 1; ++k) {
          const R angle = rrng.Next() * R(3);
          c[k] = std::cos(angle);
          s[k] = std::sin(angle);
        }
        c[2] = 1;
        s[2] = 0;
        std::vector<T> a = rng.Vector(lda * n), at(ldt * m);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < m; ++i) {
            At(at, j, i, ldt) = At(a, i, j, lda);
          }
        }
        const char left = 'L', right = 'R';
        HWYBLAS_CALL(T, lasr, &left, &pivot, &direct, &m, &n, c.data(),
                     s.data(), P(a.data()), &lda);
        HWYBLAS_CALL(T, lasr, &right, &pivot, &direct, &n, &m, c.data(),
                     s.data(), P(at.data()), &ldt);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < m; ++i) {
            ASSERT_EQ(At(a, i, j, lda), At(at, j, i, ldt))
                << pivot << direct << " n=" << n << " (" << i << ", " << j
                << ")";
          }
        }
      }
    }
  }
}

TYPED_TEST(SyevTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(113);
  const lapack_int n = 12, ld = 14;
  for (const char jobz : {'V', 'N'}) {
    for (const char uplo : {'U', 'L'}) {
      SCOPED_TRACE(::testing::Message() << jobz << uplo);
      std::vector<T> col = rng.Vector(ld * ld);
      for (lapack_int i = 0; i < n; ++i) {
        At(col, i, i, ld) = T(std::real(At(col, i, i, ld)));
      }
      // The same matrix, row-major: element (i, j) at [i * ld + j].
      std::vector<T> row(col.size());
      for (lapack_int j = 0; j < ld; ++j) {
        for (lapack_int i = 0; i < ld; ++i) row[i * ld + j] = At(col, i, j, ld);
      }
      const std::vector<T> row0 = row;
      std::vector<R> wc(n), wr(n);
      ASSERT_EQ(LapackeHeev(LAPACK_COL_MAJOR, jobz, uplo, n, col.data(), ld,
                            wc.data()),
                0);
      ASSERT_EQ(LapackeHeev(LAPACK_ROW_MAJOR, jobz, uplo, n, row.data(), ld,
                            wr.data()),
                0);
      EXPECT_EQ(wr, wc);
      for (lapack_int i = 0; i < n; ++i) {
        for (lapack_int j = 0; j < n; ++j) {
          const bool in = uplo == 'U' ? i <= j : i >= j;
          if (jobz == 'V') {
            ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << i << ", " << j;
          } else if (!in) {
            ASSERT_EQ(row[i * ld + j], row0[i * ld + j]) << i << ", " << j;
          }
        }
      }
    }
  }

  // xSTEQR with Z = I ('V'), compared with 'I'.
  std::vector<R> d = Random<R>(5).Vector(n), e = Random<R>(7).Vector(n - 1);
  std::vector<R> d2 = d, e2 = e;
  std::vector<T> zi(ld * ld, NaN<T>()), zv(ld * ld, T(0));
  for (lapack_int i = 0; i < n; ++i) zv[i * ld + i] = T(1);
  ASSERT_EQ(
      LapackeSteqr(LAPACK_ROW_MAJOR, 'I', n, d.data(), e.data(), zi.data(), ld),
      0);
  ASSERT_EQ(LapackeSteqr(LAPACK_ROW_MAJOR, 'V', n, d2.data(), e2.data(),
                         zv.data(), ld),
            0);
  EXPECT_EQ(d, d2);
  for (lapack_int i = 0; i < n; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      ASSERT_EQ(zi[i * ld + j], zv[i * ld + j]) << i << ", " << j;
    }
  }
  std::vector<R> d3 = Random<R>(5).Vector(n), e3 = Random<R>(7).Vector(n - 1);
  ASSERT_EQ(LapackeSteqr<T>(LAPACK_ROW_MAJOR, 'N', n, d3.data(), e3.data(),
                            nullptr, n),
            0);
  for (lapack_int i = 0; i < n; ++i)
    EXPECT_NEAR(d3[i], d[i], 30 * n * Eps<T>());
}

TYPED_TEST(SyevTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1));
  std::vector<R> w(4);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeHeev<T>(0, 'V', 'U', 4, a.data(), 4, w.data()), -1);
  EXPECT_EQ(
      LapackeHeev<T>(LAPACK_ROW_MAJOR, 'V', 'U', 4, a.data(), 3, w.data()), -6);
  std::vector<R> d(4, 1), e(3, 0);
  EXPECT_EQ(LapackeSteqr<T>(LAPACK_ROW_MAJOR, 'I', 4, d.data(), e.data(),
                            a.data(), 3),
            -7);
  const std::string ev = IsComplex<T>::value ? "heev" : "syev";
  EXPECT_EQ(
      ::testing::internal::GetCapturedStdout(),
      "Wrong parameter 1 in " + LapackeName<T>(ev.c_str()) +
          "\nWrong parameter 6 in " + LapackeName<T>((ev + "_work").c_str()) +
          "\nWrong parameter 7 in " + LapackeName<T>("steqr_work") + "\n");
  // NaN checks: in the triangle of A, in d, e and (for 'V') in Z.
  a[1] = NaN<T>();  // (1, 0) column-major: lower triangle
  EXPECT_EQ(
      LapackeHeev<T>(LAPACK_COL_MAJOR, 'N', 'L', 4, a.data(), 4, w.data()), -5);
  EXPECT_EQ(
      LapackeHeev<T>(LAPACK_COL_MAJOR, 'N', 'U', 4, a.data(), 4, w.data()), 0);
  d[2] = NAN;
  EXPECT_EQ(
      LapackeSteqr<T>(LAPACK_COL_MAJOR, 'N', 4, d.data(), e.data(), nullptr, 1),
      -4);
  d[2] = 1;
  e[1] = NAN;
  EXPECT_EQ(
      LapackeSteqr<T>(LAPACK_COL_MAJOR, 'N', 4, d.data(), e.data(), nullptr, 1),
      -5);
  e[1] = 0;
  std::vector<T> z(16, NaN<T>());
  EXPECT_EQ(LapackeSteqr<T>(LAPACK_COL_MAJOR, 'V', 4, d.data(), e.data(),
                            z.data(), 4),
            -6);
  EXPECT_EQ(LapackeSteqr<T>(LAPACK_COL_MAJOR, 'I', 4, d.data(), e.data(),
                            z.data(), 4),
            0);
  if constexpr (!IsComplex<T>::value) {
    std::vector<R> x = {3, 1, 2};
    if constexpr (std::is_same_v<T, double>) {
      EXPECT_EQ(LAPACKE_dlasrt('D', 3, x.data()), 0);
      EXPECT_EQ(x, (std::vector<R>{3, 2, 1}));
      x[1] = NAN;
      EXPECT_EQ(LAPACKE_dlasrt('D', 3, x.data()), -3);
      EXPECT_EQ(LAPACKE_dsterf(4, d.data(), e.data()), 0);
    } else {
      EXPECT_EQ(LAPACKE_slasrt('I', 3, x.data()), 0);
      EXPECT_EQ(x, (std::vector<R>{1, 2, 3}));
    }
  }
}

}  // namespace
}  // namespace hwyblas_test
