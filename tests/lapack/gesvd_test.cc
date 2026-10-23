// Tests of the singular value decomposition: xGESVD, the reduction to
// bidiagonal form (xGEBRD, xGEBD2, xLABRD) and its Q and P (xORGBR /
// xUNGBR, xORMBR / xUNMBR), the bidiagonal SVD xBDSQR with xLAS2 and xLASV2,
// the dqds routines (xLASQ1 - xLASQ6), and LAPACKE.
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
class GesvdTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GesvdTest, ScalarTypes);

template <typename R>
class GesvdRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GesvdRealTest, RealTypes);

struct Blocking {
  hwyblas_int nb, nx;
};
const Blocking kBlockings[] = {{0, 0}, {1, 0}, {2, 0}, {3, 5}, {8, 0}};

// ---------------------------------------------------------------------------
// Bidiagonal reduction

template <typename T>
hwyblas_int Gebrd(hwyblas_int m, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda, std::vector<RealT<T>>& d,
                  std::vector<RealT<T>>& e, std::vector<T>& tauq,
                  std::vector<T>& taup, hwyblas_int lwork) {
  const hwyblas_int k = std::max<hwyblas_int>(1, std::min(m, n));
  d.assign(k, NAN);
  e.assign(k, NAN);
  tauq.assign(k, NaN<T>());
  taup.assign(k, NaN<T>());
  std::vector<T> work(std::max<hwyblas_int>(1, lwork), NaN<T>());
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gebrd, &m, &n, P(a.data()), &lda, d.data(), e.data(),
               P(tauq.data()), P(taup.data()), P(work.data()), &lwork, &info);
  return info;
}

template <typename T>
hwyblas_int Ungbr(char vect, hwyblas_int m, hwyblas_int n, hwyblas_int k,
                  std::vector<T>& a, hwyblas_int lda, const std::vector<T>& tau,
                  hwyblas_int lwork) {
  std::vector<T> work(std::max<hwyblas_int>(1, lwork), NaN<T>());
  hwyblas_int info = -99;
  CALL_RC(T, orgbr, ungbr, &vect, &m, &n, &k, P(a.data()), &lda, P(tau.data()),
          P(work.data()), &lwork, &info);
  return info;
}

// A copy of the reflectors in a (m x n, leading dimension lda) in a matrix
// with `cols` columns (for the full Q or P**H, which overwrite it).
template <typename T>
std::vector<T> Widen(const std::vector<T>& a, hwyblas_int m, hwyblas_int n,
                     hwyblas_int lda, hwyblas_int rows, hwyblas_int cols) {
  std::vector<T> w(static_cast<size_t>(rows) * cols, NaN<T>());
  for (hwyblas_int j = 0; j < std::min(n, cols); ++j) {
    for (hwyblas_int i = 0; i < std::min(m, rows); ++i) {
      At(w, i, j, rows) = At(a, i, j, lda);
    }
  }
  return w;
}

// xGEBRD with xORGBR / xUNGBR: A = Q * B * P**H with B upper bidiagonal for
// m >= n and lower bidiagonal for m < n, orthonormal columns of Q and rows
// of P**H (also the full square Q and P**H), with blocked and unblocked
// code and the minimum or optimal workspaces.
TYPED_TEST(GesvdTest, Bidiagonalization) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(131);
    for (const Blocking& b : kBlockings) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const Shape2 s : std::vector<Shape2>{{1, 1},
                                                {1, 4},
                                                {4, 1},
                                                {5, 3},
                                                {3, 5},
                                                {20, 13},
                                                {13, 20},
                                                {24, 24},
                                                {31, 7},
                                                {7, 31}}) {
        for (const bool minimal : {false, true}) {
          const hwyblas_int m = s.m, n = s.n, k = std::min(m, n);
          SCOPED_TRACE(::testing::Message()
                       << "m=" << m << " n=" << n << " nb=" << b.nb
                       << " nx=" << b.nx << " minimal=" << minimal);
          const hwyblas_int lda = m + 3;
          const std::vector<T> a = rng.Vector(lda * n);
          std::vector<T> af = a, tauq, taup;
          std::vector<R> d, e;
          ASSERT_EQ(Gebrd(m, n, af, lda, d, e, tauq, taup,
                          minimal ? std::max(m, n) : (m + n) * 64),
                    0);
          d.resize(k);
          e.resize(k - 1);
          // The diagonal and off-diagonal are stored in A too.
          for (hwyblas_int i = 0; i < k; ++i) {
            ASSERT_EQ(At(af, i, i, lda), T(d[i]));
            if (i + 1 < k) {
              ASSERT_EQ(m >= n ? At(af, i, i + 1, lda) : At(af, i + 1, i, lda),
                        T(e[i]));
            }
          }
          // TAUP(k) (m >= n) or TAUQ(k) (m < n) is zero.
          ASSERT_EQ(m >= n ? taup[k - 1] : tauq[k - 1], T(0));

          // Q (m x k) and P**H (k x n).
          std::vector<T> q = af, pt = af;
          const hwyblas_int lw = minimal ? k : 64 * std::max(m, n);
          ASSERT_EQ(Ungbr('Q', m, k, n, q, lda, tauq, lw), 0);
          ASSERT_EQ(Ungbr('P', k, n, m, pt, lda, taup, lw), 0);
          const Dense<T> qd = ToDense(q, m, k, lda);
          const Dense<T> ptd = ToDense(pt, k, n, lda);
          EXPECT_LT(OrthogonalityRatio(qd), kLapackThreshold);
          EXPECT_LT(OrthogonalityRatio(Adjoint(ptd)), kLapackThreshold);
          const Dense<T> qbp = Mul(Mul(qd, Bidiagonal<T>(d, e, m >= n)), ptd);
          EXPECT_LT(ResidualRatio(ToDense(a, m, n, lda), qbp, std::max(m, n)),
                    kLapackThreshold);

          // The full m x m Q and n x n P**H, whose leading parts are the
          // ones above.
          std::vector<T> qf = Widen(af, m, n, lda, m, m);
          std::vector<T> pf = Widen(af, m, n, lda, n, n);
          ASSERT_EQ(Ungbr('Q', m, m, n, qf, m, tauq, minimal ? m : 64 * m), 0);
          ASSERT_EQ(Ungbr('P', n, n, m, pf, n, taup, minimal ? n : 64 * n), 0);
          const Dense<T> qfd = ToDense(qf, m, m, m);
          const Dense<T> pfd = ToDense(pf, n, n, n);
          EXPECT_LT(OrthogonalityRatio(qfd), kLapackThreshold);
          EXPECT_LT(OrthogonalityRatio(pfd), kLapackThreshold);
          EXPECT_LT(ResidualRatio(qd, Columns(qfd, k), 1), kLapackThreshold);
          EXPECT_LT(ResidualRatio(ptd, Rows(pfd, k), 1), kLapackThreshold);
        }
      }
    }
  });
}

// xLABRD on its own, with the update A := A - V * Y**H - X * U**H of the
// trailing matrix, gives the same B as xGEBD2 of the whole matrix.
TYPED_TEST(GesvdTest, Labrd) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(137);
  for (const Shape2 s : std::vector<Shape2>{{9, 6}, {6, 9}, {7, 7}}) {
    const hwyblas_int m = s.m, n = s.n, k = std::min(m, n), nb = 3;
    SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n);
    const hwyblas_int lda = m + 1;
    const std::vector<T> a = rng.Vector(lda * n);
    std::vector<T> a1 = a, a2 = a;
    std::vector<R> d1(k), e1(k), d2(k), e2(k);
    std::vector<T> tq1(k), tp1(k), tq2(k), tp2(k);
    std::vector<T> work(std::max(m, n));
    hwyblas_int info = -99;
    HWYBLAS_CALL(T, gebd2, &m, &n, P(a1.data()), &lda, d1.data(), e1.data(),
                 P(tq1.data()), P(tp1.data()), P(work.data()), &info);
    ASSERT_EQ(info, 0);
    std::vector<T> x(m * nb, NaN<T>()), y(n * nb, NaN<T>());
    HWYBLAS_CALL(T, labrd, &m, &n, &nb, P(a2.data()), &lda, d2.data(),
                 e2.data(), P(tq2.data()), P(tp2.data()), P(x.data()), &m,
                 P(y.data()), &n);
    // Update the trailing matrix as xGEBRD does, then reduce it.
    const hwyblas_int mr = m - nb, nr = n - nb;
    const T one = T(1), mone = T(-1);
    const char nt = 'N', ct = IsComplex<T>::value ? 'C' : 'T';
    HWYBLAS_CALL(T, gemm, &nt, &ct, &mr, &nr, &nb, P(&mone),
                 P(&At(a2, nb, 0, lda)), &lda, P(&y[nb]), &n, P(&one),
                 P(&At(a2, nb, nb, lda)), &lda);
    HWYBLAS_CALL(T, gemm, &nt, &nt, &mr, &nr, &nb, P(&mone), P(&x[nb]), &m,
                 P(&At(a2, 0, nb, lda)), &lda, P(&one), P(&At(a2, nb, nb, lda)),
                 &lda);
    HWYBLAS_CALL(T, gebd2, &mr, &nr, P(&At(a2, nb, nb, lda)), &lda, &d2[nb],
                 &e2[nb], P(&tq2[nb]), P(&tp2[nb]), P(work.data()), &info);
    ASSERT_EQ(info, 0);
    const double tol = 50 * k * Eps<T>();
    for (hwyblas_int i = 0; i < k; ++i) {
      EXPECT_NEAR(d2[i], d1[i], tol) << i;
      if (i + 1 < k) {
        EXPECT_NEAR(e2[i], e1[i], tol) << i;
      }
      EXPECT_NEAR(Abs(tq2[i] - tq1[i]), 0, tol) << i;
      EXPECT_NEAR(Abs(tp2[i] - tp1[i]), 0, tol) << i;
    }
  }
}

// xORMBR / xUNMBR: Q * C, Q**H * C, C * Q, C * Q**H and the same for P, for
// nq >= k and nq < k, against the explicit Q and P from xORGBR / xUNGBR.
TYPED_TEST(GesvdTest, Unmbr) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(139);
    const char ct = IsComplex<T>::value ? 'C' : 'T';
    for (const Blocking& b : {Blocking{0, 0}, Blocking{2, 0}}) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const char vect : {'Q', 'P'}) {
        for (const char side : {'L', 'R'}) {
          for (const char trans : {'N', ct}) {
            for (const hwyblas_int k : {4, 9, 13}) {
              const hwyblas_int m = 9, n = 7;
              const hwyblas_int nq = side == 'L' ? m : n;
              SCOPED_TRACE(::testing::Message()
                           << vect << side << trans << " k=" << k
                           << " nb=" << b.nb);
              // Reduce an nq x k (Q) or k x nq (P) matrix.
              const hwyblas_int am = vect == 'Q' ? nq : k;
              const hwyblas_int an = vect == 'Q' ? k : nq;
              const hwyblas_int lda = am + 2;
              std::vector<T> a = rng.Vector(lda * an), tauq, taup;
              std::vector<R> d, e;
              ASSERT_EQ(Gebrd(am, an, a, lda, d, e, tauq, taup, 64 * (m + n)),
                        0);
              const std::vector<T>& tau = vect == 'Q' ? tauq : taup;
              // The explicit nq x nq Q or P**H.
              std::vector<T> qf = Widen(a, am, an, lda, nq, nq);
              ASSERT_EQ(Ungbr(vect, nq, nq, k, qf, nq, tau, 64 * nq), 0);
              Dense<T> op = ToDense(qf, nq, nq, nq);  // Q or P**H
              if (vect == 'P') op = Adjoint(op);      // P
              if (trans != 'N') op = Adjoint(op);
              const hwyblas_int ldc = m + 1;
              const std::vector<T> c = rng.Vector(ldc * n);
              std::vector<T> cc = c;
              const std::vector<T> a0 = a;
              hwyblas_int lwork = 64 * (m + n), info = -99;
              std::vector<T> work(lwork, NaN<T>());
              CALL_RC(T, ormbr, unmbr, &vect, &side, &trans, &m, &n, &k,
                      P(a.data()), &lda, P(tau.data()), P(cc.data()), &ldc,
                      P(work.data()), &lwork, &info);
              ASSERT_EQ(info, 0);
              EXPECT_EQ(a, a0);  // restored after the conjugations
              const Dense<T> cd = ToDense(c, m, n, ldc);
              const Dense<T> expected = side == 'L' ? Mul(op, cd) : Mul(cd, op);
              EXPECT_LT(ResidualRatio(expected, ToDense(cc, m, n, ldc), nq),
                        kLapackThreshold);
              // The minimum workspace.
              std::vector<T> cm = c;
              lwork = side == 'L' ? n : m;
              CALL_RC(T, ormbr, unmbr, &vect, &side, &trans, &m, &n, &k,
                      P(a.data()), &lda, P(tau.data()), P(cm.data()), &ldc,
                      P(work.data()), &lwork, &info);
              ASSERT_EQ(info, 0);
              EXPECT_LT(ResidualRatio(expected, ToDense(cm, m, n, ldc), nq),
                        kLapackThreshold);
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// The bidiagonal SVD

template <typename T>
hwyblas_int Bdsqr(char uplo, hwyblas_int n, hwyblas_int ncvt, hwyblas_int nru,
                  hwyblas_int ncc, std::vector<RealT<T>>& d,
                  std::vector<RealT<T>>& e, std::vector<T>& vt,
                  hwyblas_int ldvt, std::vector<T>& u, hwyblas_int ldu,
                  std::vector<T>& c, hwyblas_int ldc) {
  std::vector<RealT<T>> work(std::max<hwyblas_int>(1, 4 * n), NAN);
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, bdsqr, &uplo, &n, &ncvt, &nru, &ncc, d.data(), e.data(),
               P(vt.data()), &ldvt, P(u.data()), &ldu, P(c.data()), &ldc,
               work.data(), &info);
  return info;
}

// xBDSQR: B = U * S * VT for upper and lower B with orthonormal U and VT, C
// = U**H for C = I, singular values in decreasing order, and (to high
// relative accuracy) the same singular values without vectors (dqds,
// xLASQ1).
TYPED_TEST(GesvdTest, Bdsqr) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<R> rng(149);
  for (const int kind : {0, 1, 2, 3, 4, 5, 6}) {
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {1, 2, 3, 10, 37}) {
        SCOPED_TRACE(::testing::Message()
                     << "kind=" << kind << " uplo=" << uplo << " n=" << n);
        std::vector<R> d0, e0;
        TestBidiagonal(kind, n, rng, d0, e0);
        R bnorm = 0;
        for (R x : d0) bnorm = std::max(bnorm, std::abs(x));
        for (R x : e0) bnorm = std::max(bnorm, std::abs(x));
        // Dense B in scaled form (the residual ratios are scale invariant).
        std::vector<R> ds = d0, es = e0;
        if (bnorm > 0) {
          for (auto& x : ds) x /= bnorm;
          for (auto& x : es) x /= bnorm;
        }
        const Dense<T> bd = Bidiagonal<T>(ds, es, uplo == 'U');

        const hwyblas_int ld = n + 2;
        std::vector<T> vt(ld * n, T(0)), u(ld * n, T(0)), c(ld * n, T(0));
        for (hwyblas_int i = 0; i < n; ++i) {
          At(vt, i, i, ld) = At(u, i, i, ld) = At(c, i, i, ld) = T(1);
        }
        std::vector<R> d = d0, e = e0;
        ASSERT_EQ(Bdsqr<T>(uplo, n, n, n, n, d, e, vt, ld, u, ld, c, ld), 0);
        ExpectDecreasing(d);
        std::vector<R> sv = d;
        if (bnorm > 0) {
          for (auto& x : sv) x /= bnorm;
        }
        const Dense<T> ud = ToDense(u, n, n, ld), vtd = ToDense(vt, n, n, ld);
        EXPECT_LT(OrthogonalityRatio(ud), kLapackThreshold);
        EXPECT_LT(OrthogonalityRatio(Adjoint(vtd)), kLapackThreshold);
        EXPECT_LT(ResidualRatio(bd, Mul(Mul(ud, Diagonal<T>(sv, n)), vtd), n),
                  kLapackThreshold);
        EXPECT_LT(ResidualRatio(Adjoint(ud), ToDense(c, n, n, ld), n),
                  kLapackThreshold);

        // Only VT (ncvt > 0, nru = ncc = 0), and no vectors (dqds).
        std::vector<R> d1 = d0, e1 = e0, dn = d0, en = e0;
        std::vector<T> vt1(ld * n, T(0)), dummy(1);
        for (hwyblas_int i = 0; i < n; ++i) At(vt1, i, i, ld) = T(1);
        ASSERT_EQ(
            Bdsqr<T>(uplo, n, n, 0, 0, d1, e1, vt1, ld, dummy, 1, dummy, 1), 0);
        ASSERT_EQ(
            Bdsqr<T>(uplo, n, 0, 0, 0, dn, en, dummy, 1, dummy, 1, dummy, 1),
            0);
        ExpectDecreasing(dn);
        for (hwyblas_int i = 0; i < n; ++i) {
          EXPECT_EQ(d1[i], d[i]) << i;
          // Both algorithms are accurate relative to each singular value.
          EXPECT_NEAR(dn[i], d[i],
                      60 * n * Eps<T>() * (d[i] + Eps<T>() * bnorm))
              << i;
        }
        EXPECT_EQ(vt1, vt);
      }
    }
  }
}

// xLAS2 and xLASV2: the singular values of [f g; 0 h], and with xLASV2 the
// rotations that diagonalize it, also for zeros and widely differing
// elements.
TYPED_TEST(GesvdRealTest, Las2Lasv2) {
  using R = TypeParam;
  Random<R> rng(151);
  const R big = std::numeric_limits<R>::max() / 16;
  for (int t = 0; t < 400; ++t) {
    R f = rng.Next(), g = rng.Next(), h = rng.Next();
    switch (t % 12) {
      case 1:
        g = 0;
        break;
      case 2:
        f = 0;
        break;
      case 3:
        h = 0;
        break;
      case 4:
        f = h = 0;
        break;
      case 5:
        g *= static_cast<R>(1 / Eps<R>());  // g dominates: fa / ga < eps
        break;
      case 6:
        h *= 1e8f;  // swap (ha > fa)
        break;
      case 7:
        f = h;  // equal diagonal (l = 0)
        break;
      case 8:
        f = big, g = big, h = 1;
        break;
      case 9:
        f *= std::numeric_limits<R>::min(), g = 1;
        break;
      default:
        break;
    }
    SCOPED_TRACE(::testing::Message() << "f=" << f << " g=" << g << " h=" << h);
    R ssmin, ssmax, ssmin2, ssmax2, snr, csr, snl, csl;
    if constexpr (std::is_same_v<R, float>) {
      slas2_(&f, &g, &h, &ssmin, &ssmax);
      slasv2_(&f, &g, &h, &ssmin2, &ssmax2, &snr, &csr, &snl, &csl);
    } else {
      dlas2_(&f, &g, &h, &ssmin, &ssmax);
      dlasv2_(&f, &g, &h, &ssmin2, &ssmax2, &snr, &csr, &snl, &csl);
    }
    // xLAS2 returns non-negative values; xLASV2 signed ones, with the same
    // magnitudes up to rounding.
    EXPECT_GE(ssmin, 0);
    EXPECT_GE(ssmax, ssmin);
    const double scale = std::max({std::abs(static_cast<double>(f)),
                                   std::abs(static_cast<double>(g)),
                                   std::abs(static_cast<double>(h))});
    const double tol = 8 * Eps<R>();
    EXPECT_NEAR(std::abs(ssmax2) / scale, ssmax / scale, tol);
    EXPECT_NEAR(std::abs(ssmin2) / scale, ssmin / scale, tol);
    // |ssmax| * |ssmin| = |f * h| and ssmax**2 + ssmin**2 = f**2 + g**2 +
    // h**2 (relative to the scale).
    const double fs = f / scale, gs = g / scale, hs = h / scale;
    const double smax = ssmax / scale, smin = ssmin / scale;
    EXPECT_NEAR(smax * smin, std::abs(fs * hs), tol);
    EXPECT_NEAR(smax * smax + smin * smin, fs * fs + gs * gs + hs * hs,
                4 * tol);
    // [csl snl; -snl csl] * [f g; 0 h] * [csr -snr; snr csr] =
    // diag(ssmax, ssmin), with unit rotations.
    EXPECT_NEAR(static_cast<double>(csl) * csl + static_cast<double>(snl) * snl,
                1.0, tol);
    EXPECT_NEAR(static_cast<double>(csr) * csr + static_cast<double>(snr) * snr,
                1.0, tol);
    const double b11 = csl * fs, b12 = csl * gs + snl * hs;
    const double b21 = -snl * fs, b22 = -snl * gs + csl * hs;
    EXPECT_NEAR(b11 * csr + b12 * snr, ssmax2 / scale, 4 * tol);
    EXPECT_NEAR(-b11 * snr + b12 * csr, 0, 4 * tol);
    EXPECT_NEAR(b21 * csr + b22 * snr, 0, 4 * tol);
    EXPECT_NEAR(-b21 * snr + b22 * csr, ssmin2 / scale, 4 * tol);
  }
}

template <typename R>
hwyblas_int Lasq1(hwyblas_int n, std::vector<R>& d, std::vector<R>& e) {
  std::vector<R> work(std::max<hwyblas_int>(1, 4 * n), NAN);
  hwyblas_int info = -99;
  HWYBLAS_CALL_REAL(R, lasq1, &n, d.data(), e.data(), work.data(), &info);
  return info;
}

template <typename R>
hwyblas_int Lasq2(hwyblas_int n, std::vector<R>& z) {
  hwyblas_int info = -99;
  HWYBLAS_CALL_REAL(R, lasq2, &n, z.data(), &info);
  return info;
}

// xLASQ1 and xLASQ2: the singular values of bidiagonal matrices (as xBDSQR
// with vectors computes them), the special cases (n <= 2, e = 0, a zero
// matrix), and xLASQ2's statistics and error returns.
TYPED_TEST(GesvdRealTest, Lasq) {
  using R = TypeParam;
  Random<R> rng(157);
  for (const int kind : {0, 1, 2, 3, 4, 5, 6}) {
    for (const hwyblas_int n : {1, 2, 3, 4, 17, 60}) {
      SCOPED_TRACE(::testing::Message() << "kind=" << kind << " n=" << n);
      std::vector<R> d0, e0;
      TestBidiagonal(kind, n, rng, d0, e0);
      std::vector<R> d = d0, e = e0, dq = d0, eq = e0;
      std::vector<R> vt(n * n, 0), dummy(1);
      for (hwyblas_int i = 0; i < n; ++i) vt[i * n + i] = 1;
      ASSERT_EQ(Bdsqr<R>('U', n, n, 0, 0, d, e, vt, n, dummy, 1, dummy, 1), 0);
      eq.push_back(0);
      ASSERT_EQ(Lasq1(n, dq, eq), 0);
      R bnorm = 0;
      for (R x : d) bnorm = std::max(bnorm, x);
      for (hwyblas_int i = 0; i < n; ++i) {
        EXPECT_NEAR(dq[i], d[i], 60 * n * Eps<R>() * (d[i] + Eps<R>() * bnorm))
            << i;
      }
    }
  }
  // e = 0: the sorted absolute values.
  std::vector<R> d = {1, -3, 2, 0.5}, e = {0, 0, 0, 0};
  ASSERT_EQ(Lasq1<R>(4, d, e), 0);
  EXPECT_EQ(d, (std::vector<R>{3, 2, 1, 0.5}));
  // A zero matrix.
  d = {0, 0, 0};
  e = {0, 0, 0};
  ASSERT_EQ(Lasq1<R>(3, d, e), 0);
  EXPECT_EQ(d, (std::vector<R>{0, 0, 0}));
  // n = 1 and 2.
  d = {-2};
  ASSERT_EQ(Lasq1<R>(1, d, e), 0);
  EXPECT_EQ(d[0], 2);
  d = {3, 0};
  e = {4};
  ASSERT_EQ(Lasq1<R>(2, d, e), 0);
  EXPECT_EQ(d, (std::vector<R>{5, 0}));
  ASSERT_EQ(Lasq1<R>(0, d, e), 0);

  // xLASQ2: the eigenvalues of the qd array of [2 1; 0 1]**H * [2 1; 0 1]
  // etc.: q = d**2, e = e**2. Z(2n+1) is the trace and Z(2n+2) the sum of
  // the eigenvalues.
  const hwyblas_int n = 30;
  std::vector<R> dd = Random<R>(5).Vector(n), ee = Random<R>(7).Vector(n - 1);
  std::vector<R> z(4 * n, NAN);
  for (hwyblas_int i = 0; i < n; ++i) {
    z[2 * i] = dd[i] * dd[i];
    if (i + 1 < n) z[2 * i + 1] = ee[i] * ee[i];
  }
  double trace = 0;
  for (hwyblas_int i = 0; i < 2 * n - 1; ++i) trace += z[i];
  ASSERT_EQ(Lasq2(n, z), 0);
  for (hwyblas_int i = 1; i < n; ++i) ASSERT_GE(z[i - 1], z[i]);
  EXPECT_NEAR(z[2 * n], trace, 4 * n * Eps<R>() * trace);
  EXPECT_NEAR(z[2 * n + 1], trace, 4 * n * Eps<R>() * trace);
  EXPECT_GE(z[2 * n + 2], 1);  // iterations
  // The singular values are the square roots.
  std::vector<R> d2 = dd, e2 = ee;
  e2.push_back(0);
  ASSERT_EQ(Lasq1(n, d2, e2), 0);
  for (hwyblas_int i = 0; i < n; ++i) {
    EXPECT_NEAR(std::sqrt(z[i]), d2[i], 20 * n * Eps<R>() * d2[i]);
  }

  // Errors: a negative n or element (INFO = -(200 + i); XERBLA gets 1 and
  // 2), and xLASQ1's negative n.
  ErrorCapture errors;
  std::vector<R> zn = {1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(Lasq2<R>(-1, zn), -1);
  EXPECT_EQ(Lasq2<R>(3, zn), -202);
  zn = {-1, 0, 0, 0};
  EXPECT_EQ(Lasq2<R>(1, zn), -201);
  zn = {1, 1, -1, 0, 0, 0, 0, 0};
  EXPECT_EQ(Lasq2<R>(2, zn), -203);
  std::vector<R> d3(2), e3(2);
  EXPECT_EQ(Lasq1<R>(-1, d3, e3), -1);
  const std::string q2 = Name<R>("LASQ2"), q1 = Name<R>("LASQ1");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{q2 + ":1", q2 + ":2", q2 + ":2",
                                      q2 + ":2", q1 + ":1"}));
}

// ---------------------------------------------------------------------------
// xGESVD

template <typename T>
hwyblas_int MinimumGesvdLwork(hwyblas_int m, hwyblas_int n) {
  const hwyblas_int mn = std::min(m, n), mx = std::max(m, n);
  if (IsComplex<T>::value) return std::max<hwyblas_int>(1, 2 * mn + mx);
  return std::max<hwyblas_int>({1, 3 * mn + mx, 5 * mn});
}

// The result of xGESVD.
template <typename T>
struct Svd {
  std::vector<T> a, u, vt;
  std::vector<RealT<T>> s, rwork;
  std::vector<T> work;
  hwyblas_int info = -99;
};

// xGESVD of the m x n matrix a (leading dimension lda); lwork = -1 uses the
// optimal workspace from a query.
template <typename T>
Svd<T> Gesvd(char jobu, char jobvt, hwyblas_int m, hwyblas_int n,
             const std::vector<T>& a, hwyblas_int lda, hwyblas_int ldu,
             hwyblas_int ldvt, hwyblas_int lwork) {
  Svd<T> r;
  r.a = a;
  const hwyblas_int mn = std::min(m, n);
  r.s.assign(std::max<hwyblas_int>(1, mn), NAN);
  r.u.assign(std::max<hwyblas_int>(1, ldu * m), NaN<T>());
  r.vt.assign(std::max<hwyblas_int>(1, ldvt * n), NaN<T>());
  r.rwork.assign(std::max<hwyblas_int>(1, 5 * mn), NAN);
  if (lwork == -1) {
    T query = NaN<T>();
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gesvd, &jobu, &jobvt, &m, &n, P(r.a.data()), &lda,
                           r.s.data(), P(r.u.data()), &ldu, P(r.vt.data()),
                           &ldvt, P(&query), &lwork, r.rwork.data(), &r.info);
    } else {
      HWYBLAS_CALL_REAL(T, gesvd, &jobu, &jobvt, &m, &n, P(r.a.data()), &lda,
                        r.s.data(), P(r.u.data()), &ldu, P(r.vt.data()), &ldvt,
                        P(&query), &lwork, &r.info);
    }
    if (r.info != 0) return r;
    lwork = static_cast<hwyblas_int>(std::real(query));
  }
  r.work.assign(std::max<hwyblas_int>(1, lwork), NaN<T>());
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, gesvd, &jobu, &jobvt, &m, &n, P(r.a.data()), &lda,
                         r.s.data(), P(r.u.data()), &ldu, P(r.vt.data()), &ldvt,
                         P(r.work.data()), &lwork, r.rwork.data(), &r.info);
  } else {
    HWYBLAS_CALL_REAL(T, gesvd, &jobu, &jobvt, &m, &n, P(r.a.data()), &lda,
                      r.s.data(), P(r.u.data()), &ldu, P(r.vt.data()), &ldvt,
                      P(r.work.data()), &lwork, &r.info);
  }
  return r;
}

// xGESVD for all JOBU and JOBVT, matrices of all paths (m much larger than
// n, m larger, m = n, m smaller, m much smaller; the crossover is 1.6 *
// min(m, n)) and workspaces from the minimum (with the slow paths) to the
// optimal one: A = U * S * VT with orthonormal U and VT, decreasing S, and
// the same S and vectors (up to unit factors) for all jobs.
TYPED_TEST(GesvdTest, Gesvd) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(163);
    const char jobs[] = {'A', 'S', 'O', 'N'};
    for (const Shape2 sh : std::vector<Shape2>{{1, 1},
                                               {1, 6},
                                               {6, 1},
                                               {12, 12},
                                               {18, 11},
                                               {40, 11},
                                               {11, 18},
                                               {11, 40},
                                               {57, 31}}) {
      const hwyblas_int m = sh.m, n = sh.n, mn = std::min(m, n);
      const hwyblas_int lda = m + 2;
      const std::vector<T> a = rng.Vector(lda * n);
      const Dense<T> ad = ToDense(a, m, n, lda);
      // The reference: all vectors with the optimal workspace.
      const Svd<T> ref = Gesvd<T>('A', 'A', m, n, a, lda, m, n, -1);
      ASSERT_EQ(ref.info, 0);
      const Dense<T> uref = ToDense(ref.u, m, mn, m);
      const Dense<T> vref = Adjoint(ToDense(ref.vt, mn, n, n));
      const hwyblas_int minimum = MinimumGesvdLwork<T>(m, n);
      for (const hwyblas_int lwork :
           {minimum, minimum + mn * mn, minimum + 2 * mn * mn,
            minimum + 2 * mn * mn + mn * std::max(m, n), hwyblas_int{-1}}) {
        for (const char jobu : jobs) {
          for (const char jobvt : jobs) {
            if (jobu == 'O' && jobvt == 'O') continue;
            SCOPED_TRACE(::testing::Message()
                         << "m=" << m << " n=" << n << " jobu=" << jobu
                         << " jobvt=" << jobvt << " lwork=" << lwork);
            const hwyblas_int ldu = m + 1, ldvt = n + 1;
            const Svd<T> r =
                Gesvd<T>(jobu, jobvt, m, n, a, lda, ldu, ldvt, lwork);
            ASSERT_EQ(r.info, 0);
            std::vector<R> s(r.s.begin(), r.s.begin() + mn);
            ExpectDecreasing(s);
            for (hwyblas_int i = 0; i < mn; ++i) {
              EXPECT_NEAR(s[i], ref.s[i], 30 * std::max(m, n) * Eps<T>()) << i;
            }
            // The vectors: in U, VT or A, or none.
            const hwyblas_int ucols = jobu == 'A' ? m : mn;
            const hwyblas_int vrows = jobvt == 'A' ? n : mn;
            Dense<T> ud(0, 0), vd(0, 0);
            if (jobu == 'A' || jobu == 'S') {
              const Dense<T> uf = ToDense(r.u, m, ucols, ldu);
              EXPECT_LT(OrthogonalityRatio(uf), kLapackThreshold);
              ud = Columns(uf, mn);
            } else if (jobu == 'O') {
              ud = ToDense(r.a, m, mn, lda);
              EXPECT_LT(OrthogonalityRatio(ud), kLapackThreshold);
            }
            if (jobvt == 'A' || jobvt == 'S') {
              const Dense<T> vf = Adjoint(ToDense(r.vt, vrows, n, ldvt));
              EXPECT_LT(OrthogonalityRatio(vf), kLapackThreshold);
              vd = Columns(vf, mn);
            } else if (jobvt == 'O') {
              vd = Adjoint(ToDense(r.a, mn, n, lda));
              EXPECT_LT(OrthogonalityRatio(vd), kLapackThreshold);
            }
            if (ud.cols > 0) {
              EXPECT_LT(VectorsRatio(ud, uref, ref.s), kLapackThreshold);
            }
            if (vd.cols > 0) {
              EXPECT_LT(VectorsRatio(vd, vref, ref.s), kLapackThreshold);
            }
            if (ud.cols > 0 && vd.cols > 0) {
              const Dense<T> usv =
                  Mul(Mul(ud, Diagonal<T>(s, mn)), Adjoint(vd));
              EXPECT_LT(ResidualRatio(ad, usv, std::max(m, n)),
                        kLapackThreshold);
            }
            // Full U and VT: the complement is orthogonal to the range.
            if (jobu == 'A' && m > mn) {
              const Dense<T> uf = ToDense(r.u, m, m, ldu);
              EXPECT_LT(OrthogonalityRatio(uf), kLapackThreshold);
            }
          }
        }
      }
    }
  });
}

// xGESVD of matrices that it scales (norm near the underflow or overflow
// threshold) and of special matrices (zero, rank one, empty).
TYPED_TEST(GesvdTest, GesvdScalingAndSpecialMatrices) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(167);
  const hwyblas_int m = 15, n = 9;
  const std::vector<T> a = rng.Vector(m * n);
  const Svd<T> ref = Gesvd<T>('S', 'S', m, n, a, m, m, n, -1);
  ASSERT_EQ(ref.info, 0);
  const R tiny = std::numeric_limits<R>::min() * 16;
  const R huge = std::numeric_limits<R>::max() / 64;
  for (const R scale : {tiny, huge}) {
    SCOPED_TRACE(::testing::Message() << "scale=" << scale);
    std::vector<T> as = a;
    for (T& x : as) x *= scale;
    const Svd<T> r = Gesvd<T>('S', 'S', m, n, as, m, m, n, -1);
    ASSERT_EQ(r.info, 0);
    for (hwyblas_int i = 0; i < n; ++i) {
      EXPECT_NEAR(r.s[i] / scale, ref.s[i], 30 * m * Eps<T>()) << i;
    }
    EXPECT_LT(
        VectorsRatio(ToDense(r.u, m, n, m), ToDense(ref.u, m, n, m), ref.s),
        kLapackThreshold);
  }
  // Zero matrix: zero singular values, U and VT orthonormal.
  const Svd<T> z =
      Gesvd<T>('A', 'A', m, n, std::vector<T>(m * n, T(0)), m, m, n, -1);
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
  const Svd<T> one = Gesvd<T>('N', 'N', m, n, r1, m, 1, 1, -1);
  ASSERT_EQ(one.info, 0);
  EXPECT_NEAR(one.s[0], std::sqrt(nx * ny), 30 * m * Eps<T>());
  for (hwyblas_int i = 1; i < n; ++i) {
    EXPECT_LT(one.s[i], 30 * m * Eps<T>()) << i;
  }
  // Empty matrices.
  EXPECT_EQ(Gesvd<T>('A', 'A', 0, 3, a, 1, 1, 3, -1).info, 0);
  EXPECT_EQ(Gesvd<T>('A', 'A', 3, 0, a, 3, 3, 1, -1).info, 0);
}

// NaN and Inf inputs end in finite time, with the results of the reference
// (which depend on Fortran's MAX(NaN, x) = x: with std::max, xBDSQR's
// THRESH was NaN and it looped forever).
TYPED_TEST(GesvdTest, NonFiniteInput) {
  using T = TypeParam;
  using R = RealT<T>;
  const hwyblas_int n = 10;
  // As xBDSDC passes a bidiagonal with an Inf, scaled by its norm.
  std::vector<R> d(n, R(0)), e(n - 1, R(0));
  d[0] = std::numeric_limits<R>::quiet_NaN();
  std::vector<T> u(n * n, T(0)), vt(n * n, T(0)), c(1);
  for (hwyblas_int i = 0; i < n; ++i) u[i * (n + 1)] = vt[i * (n + 1)] = T(1);
  EXPECT_EQ(Bdsqr<T>('U', n, n, n, 0, d, e, vt, n, u, n, c, 1), 0);
  EXPECT_EQ(d[0], R(0));
  // One Inf in the first column.
  const R inf = std::numeric_limits<R>::infinity();
  std::vector<T> a(n * n, T(0));
  a[3] = T(inf);
  EXPECT_EQ(Gesvd<T>('A', 'A', n, n, a, n, n, n, -1).info, 0);
  // An Inf matrix.
  a.assign(n * n, T(inf));
  EXPECT_GE(Gesvd<T>('A', 'A', n, n, a, n, n, n, -1).info, 0);
}

TYPED_TEST(GesvdTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  const ScopedBlockSize block_size(8);
  const hwyblas_int m = 10, n = 6;
  // xGEBRD: (m + n) * nb; xORGBR / xUNGBR and xORMBR / xUNMBR: from xORGQR
  // etc. (at least min(m, n)) and n * nb or m * nb.
  std::vector<T> a(m * n), tauq, taup;
  std::vector<R> d, e;
  hwyblas_int lwork = -1, info = -99;
  std::vector<T> work(1);
  HWYBLAS_CALL(T, gebrd, &m, &n, P(a.data()), &m, d.data(), e.data(),
               P(a.data()), P(a.data()), P(work.data()), &lwork, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(std::real(work[0]), (m + n) * 8);
  const char q = 'Q', l = 'L', nt = 'N';
  CALL_RC(T, orgbr, ungbr, &q, &m, &n, &n, P(a.data()), &m, P(a.data()),
          P(work.data()), &lwork, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(std::real(work[0]), n * 8);
  std::vector<T> c(m * n);
  CALL_RC(T, ormbr, unmbr, &q, &l, &nt, &m, &n, &n, P(a.data()), &m,
          P(a.data()), P(c.data()), &m, P(work.data()), &lwork, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(std::real(work[0]), n * 8);
  // xGESVD: at least the minimum.
  const Svd<T> query = Gesvd<T>('A', 'A', m, n, a, m, m, n, -1);
  EXPECT_EQ(query.info, 0);
  EXPECT_GE(static_cast<hwyblas_int>(query.work.size()),
            MinimumGesvdLwork<T>(m, n));

  // Invalid arguments.
  ErrorCapture errors;
  const std::string sv = Name<T>("GESVD");
  const hwyblas_int mw = MinimumGesvdLwork<T>(m, n);
  EXPECT_EQ(Gesvd<T>('X', 'N', m, n, a, m, m, n, mw).info, -1);
  EXPECT_EQ(Gesvd<T>('N', 'X', m, n, a, m, m, n, mw).info, -2);
  EXPECT_EQ(Gesvd<T>('O', 'O', m, n, a, m, m, n, mw).info, -2);
  EXPECT_EQ(Gesvd<T>('N', 'N', -1, n, a, m, m, n, mw).info, -3);
  EXPECT_EQ(Gesvd<T>('N', 'N', m, -1, a, m, m, n, mw).info, -4);
  EXPECT_EQ(Gesvd<T>('N', 'N', m, n, a, m - 1, m, n, mw).info, -6);
  EXPECT_EQ(Gesvd<T>('A', 'N', m, n, a, m, m - 1, n, mw).info, -9);
  EXPECT_EQ(Gesvd<T>('N', 'N', m, n, a, m, 0, n, mw).info, -9);
  EXPECT_EQ(Gesvd<T>('N', 'A', m, n, a, m, m, n - 1, mw).info, -11);
  EXPECT_EQ(Gesvd<T>('N', 'S', n, m, a, n, n, n - 1, mw).info, -11);
  EXPECT_EQ(Gesvd<T>('N', 'N', m, n, a, m, m, n, 1).info, -13);
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               sv + ":1", sv + ":2", sv + ":2", sv + ":3",
                               sv + ":4", sv + ":6", sv + ":9", sv + ":9",
                               sv + ":11", sv + ":11", sv + ":13"}));
  // xGEBRD, xORGBR / xUNGBR, xORMBR / xUNMBR and xBDSQR.
  std::vector<T> tq(n), tp(n);
  d.resize(n);
  e.resize(n);
  EXPECT_EQ(Gebrd<T>(m, n, a, m - 1, d, e, tq, tp, m + n), -4);
  EXPECT_EQ(Gebrd<T>(m, n, a, m, d, e, tq, tp, m - 1), -10);
  EXPECT_EQ(Ungbr<T>('X', m, n, n, a, m, tq, m), -1);
  EXPECT_EQ(Ungbr<T>('Q', m, m + 1, n, a, m, tq, m), -3);
  EXPECT_EQ(Ungbr<T>('P', n + 1, n, n, a, m, tq, m), -3);
  EXPECT_EQ(Ungbr<T>('Q', m, n, -1, a, m, tq, m), -4);
  EXPECT_EQ(Ungbr<T>('Q', m, n, n, a, m - 1, tq, m), -6);
  EXPECT_EQ(Ungbr<T>('Q', m, n, n, a, m, tq, n - 1), -9);
  const std::string brd = Name<T>("GEBRD");
  const std::string gbr = Name<T>(IsComplex<T>::value ? "UNGBR" : "ORGBR");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{brd + ":4", brd + ":10", gbr + ":1",
                                      gbr + ":3", gbr + ":3", gbr + ":4",
                                      gbr + ":6", gbr + ":9"}));
  std::vector<T> dummy(1);
  std::vector<R> dd(n), ee(n);
  EXPECT_EQ(Bdsqr<T>('X', n, 0, 0, 0, dd, ee, dummy, 1, dummy, 1, dummy, 1),
            -1);
  EXPECT_EQ(Bdsqr<T>('U', -1, 0, 0, 0, dd, ee, dummy, 1, dummy, 1, dummy, 1),
            -2);
  EXPECT_EQ(Bdsqr<T>('U', n, -1, 0, 0, dd, ee, dummy, 1, dummy, 1, dummy, 1),
            -3);
  EXPECT_EQ(Bdsqr<T>('U', n, 0, -1, 0, dd, ee, dummy, 1, dummy, 1, dummy, 1),
            -4);
  EXPECT_EQ(Bdsqr<T>('U', n, 0, 0, -1, dd, ee, dummy, 1, dummy, 1, dummy, 1),
            -5);
  EXPECT_EQ(Bdsqr<T>('U', n, 1, 0, 0, dd, ee, dummy, n - 1, dummy, 1, dummy, 1),
            -9);
  EXPECT_EQ(Bdsqr<T>('U', n, 0, 2, 0, dd, ee, dummy, 1, dummy, 1, dummy, 1),
            -11);
  EXPECT_EQ(Bdsqr<T>('U', n, 0, 0, 1, dd, ee, dummy, 1, dummy, 1, dummy, n - 1),
            -13);
  const std::string bsq = Name<T>("BDSQR");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{bsq + ":1", bsq + ":2", bsq + ":3",
                                      bsq + ":4", bsq + ":5", bsq + ":9",
                                      bsq + ":11", bsq + ":13"}));
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
lapack_int LapackeGesvd(int layout, char jobu, char jobvt, lapack_int m,
                        lapack_int n, T* a, lapack_int lda, RealT<T>* s, T* u,
                        lapack_int ldu, T* vt, lapack_int ldvt,
                        RealT<T>* superb) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_sgesvd(layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                          ldvt, superb);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dgesvd(layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                          ldvt, superb);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_cgesvd(layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                          ldvt, superb);
  } else {
    return LAPACKE_zgesvd(layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                          ldvt, superb);
  }
}

// LAPACKE_xgesvd, xgebrd, xorgbr / xungbr, xormbr / xunmbr and xbdsqr in
// row-major layout give the results of column-major layout.
TYPED_TEST(GesvdTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(173);
  const char jobs[] = {'A', 'S', 'O', 'N'};
  for (const Shape2 sh : std::vector<Shape2>{{13, 7}, {7, 13}, {30, 6}}) {
    const lapack_int m = sh.m, n = sh.n, mn = std::min(m, n);
    const lapack_int lda = m + 1, ldr = n + 2;
    const std::vector<T> a = rng.Vector(lda * n);
    for (const char jobu : jobs) {
      for (const char jobvt : jobs) {
        if (jobu == 'O' && jobvt == 'O') continue;
        SCOPED_TRACE(::testing::Message()
                     << "m=" << m << " n=" << n << " " << jobu << jobvt);
        std::vector<T> ac = a, ar = ToRowMajor(a, m, n, lda, ldr);
        std::vector<T> uc(m * m, NaN<T>()), ur(m * (m + 1), NaN<T>());
        std::vector<T> vc(n * n, NaN<T>()), vr(n * (n + 1), NaN<T>());
        std::vector<R> sc(mn), sr(mn), bc(mn), br(mn);
        ASSERT_EQ(
            LapackeGesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, ac.data(), lda,
                         sc.data(), uc.data(), m, vc.data(), n, bc.data()),
            0);
        ASSERT_EQ(LapackeGesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, ar.data(),
                               ldr, sr.data(), ur.data(), m + 1, vr.data(),
                               n + 1, br.data()),
                  0);
        EXPECT_EQ(sr, sc);
        const lapack_int ucols = jobu == 'A' ? m : mn;
        const lapack_int vrows = jobvt == 'A' ? n : mn;
        if (jobu == 'A' || jobu == 'S') {
          ExpectSameRowMajor(ur, uc, m, ucols, m + 1, m);
        }
        if (jobvt == 'A' || jobvt == 'S') {
          ExpectSameRowMajor(vr, vc, vrows, n, n + 1, n);
        }
        if (jobu == 'O' || jobvt == 'O') {
          ExpectSameRowMajor(ar, ac, m, n, ldr, lda);
        }
      }
    }
    // xGEBRD, then Q and P**H (xORGBR / xUNGBR), and xORMBR / xUNMBR with
    // them, which for complex T and vect = 'P' with nq > k needs all of the
    // k x nq reflectors.
    std::vector<T> ac = a, ar = ToRowMajor(a, m, n, lda, ldr);
    std::vector<R> dc(mn), ec(mn), dr(mn), er(mn);
    std::vector<T> tqc(mn), tpc(mn), tqr(mn), tpr(mn);
    ASSERT_EQ(LAPACKE_CALL(T, gebrd, gebrd, LAPACK_COL_MAJOR, m, n, ac.data(),
                           lda, dc.data(), ec.data(), tqc.data(), tpc.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gebrd, gebrd, LAPACK_ROW_MAJOR, m, n, ar.data(),
                           ldr, dr.data(), er.data(), tqr.data(), tpr.data()),
              0);
    EXPECT_EQ(dr, dc);
    EXPECT_EQ(tqr, tqc);
    EXPECT_EQ(tpr, tpc);
    ExpectSameRowMajor(ar, ac, m, n, ldr, lda);
    for (const char vect : {'Q', 'P'}) {
      for (const char side : {'L', 'R'}) {
        for (const char trans : {'N', IsComplex<T>::value ? 'C' : 'T'}) {
          SCOPED_TRACE(::testing::Message() << vect << side << trans);
          // The reduced matrix is m x n = nq x k for Q and k x nq for P;
          // C is nq x 5 (side 'L') or 5 x nq ('R').
          const lapack_int nq = vect == 'Q' ? m : n;
          const lapack_int k = vect == 'Q' ? n : m;
          const lapack_int cm = side == 'L' ? nq : 5;
          const lapack_int cn = side == 'L' ? 5 : nq;
          const std::vector<T> c = rng.Vector(cm * cn);
          std::vector<T> cc = c, cr = ToRowMajor(c, cm, cn, cm, cn + 1);
          const std::vector<T>& tau = vect == 'Q' ? tqc : tpc;
          ASSERT_EQ(LAPACKE_CALL(T, ormbr, unmbr, LAPACK_COL_MAJOR, vect, side,
                                 trans, cm, cn, k, ac.data(), lda, tau.data(),
                                 cc.data(), cm),
                    0);
          ASSERT_EQ(LAPACKE_CALL(T, ormbr, unmbr, LAPACK_ROW_MAJOR, vect, side,
                                 trans, cm, cn, k, ar.data(), ldr, tau.data(),
                                 cr.data(), cn + 1),
                    0);
          ExpectSameRowMajor(cr, cc, cm, cn, cn + 1, cm);
        }
      }
    }
    std::vector<T> qc = ac, qr = ar;
    ASSERT_EQ(LAPACKE_CALL(T, orgbr, ungbr, LAPACK_COL_MAJOR, 'Q', m, mn, n,
                           qc.data(), lda, tqc.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, orgbr, ungbr, LAPACK_ROW_MAJOR, 'Q', m, mn, n,
                           qr.data(), ldr, tqc.data()),
              0);
    ExpectSameRowMajor(qr, qc, m, mn, ldr, lda);
    std::vector<T> pc = ac, pr = ar;
    ASSERT_EQ(LAPACKE_CALL(T, orgbr, ungbr, LAPACK_COL_MAJOR, 'P', mn, n, m,
                           pc.data(), lda, tpc.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, orgbr, ungbr, LAPACK_ROW_MAJOR, 'P', mn, n, m,
                           pr.data(), ldr, tpc.data()),
              0);
    ExpectSameRowMajor(pr, pc, mn, n, ldr, lda);

    // xBDSQR with U, VT and C.
    const char uplo = m >= n ? 'U' : 'L';
    std::vector<R> d1 = dc, e1 = ec, d2 = dc, e2 = ec;
    std::vector<T> u1 = qc, u2 = qr, v1 = pc, v2 = pr;
    const std::vector<T> c = rng.Vector(mn * 3);
    std::vector<T> c1 = c, c2 = ToRowMajor(c, mn, 3, mn, 4);
    ASSERT_EQ(LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_COL_MAJOR, uplo, mn, n, m, 3,
                           d1.data(), e1.data(), v1.data(), lda, u1.data(), lda,
                           c1.data(), mn),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_ROW_MAJOR, uplo, mn, n, m, 3,
                           d2.data(), e2.data(), v2.data(), ldr, u2.data(), ldr,
                           c2.data(), 4),
              0);
    EXPECT_EQ(d1, d2);
    ExpectSameRowMajor(u2, u1, m, mn, ldr, lda);
    ExpectSameRowMajor(v2, v1, mn, n, ldr, lda);
    ExpectSameRowMajor(c2, c1, mn, 3, 4, mn);
    // The SVD from these is the one of xGESVD.
    std::vector<T> as = a, us(m * mn), vs(mn * n);
    std::vector<R> ss(mn), superb(mn);
    ASSERT_EQ(
        LapackeGesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, as.data(), lda,
                     ss.data(), us.data(), m, vs.data(), mn, superb.data()),
        0);
    for (lapack_int i = 0; i < mn; ++i) {
      EXPECT_NEAR(d1[i], ss[i], 30 * m * Eps<T>()) << i;
    }
  }
}

TYPED_TEST(GesvdTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), u(16), vt(16);
  std::vector<R> s(4), superb(4);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeGesvd<T>(0, 'A', 'A', 4, 4, a.data(), 4, s.data(), u.data(),
                            4, vt.data(), 4, superb.data()),
            -1);
  EXPECT_EQ(LapackeGesvd<T>(LAPACK_ROW_MAJOR, 'A', 'A', 4, 4, a.data(), 3,
                            s.data(), u.data(), 4, vt.data(), 4, superb.data()),
            -7);
  EXPECT_EQ(LapackeGesvd<T>(LAPACK_ROW_MAJOR, 'A', 'A', 4, 4, a.data(), 4,
                            s.data(), u.data(), 3, vt.data(), 4, superb.data()),
            -10);
  EXPECT_EQ(LapackeGesvd<T>(LAPACK_ROW_MAJOR, 'A', 'S', 4, 4, a.data(), 4,
                            s.data(), u.data(), 4, vt.data(), 3, superb.data()),
            -12);
  std::vector<R> d(4, 1), e(3, 0);
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_ROW_MAJOR, 'U', 4, 4, 4, 0, d.data(),
                   e.data(), vt.data(), 3, u.data(), 4, a.data(), 1),
      -10);
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_ROW_MAJOR, 'U', 4, 4, 4, 0, d.data(),
                   e.data(), vt.data(), 4, u.data(), 3, a.data(), 1),
      -12);
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_ROW_MAJOR, 'U', 4, 0, 0, 2, d.data(),
                   e.data(), vt.data(), 4, u.data(), 4, a.data(), 1),
      -14);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("gesvd") +
                "\nWrong parameter 7 in " + LapackeName<T>("gesvd_work") +
                "\nWrong parameter 10 in " + LapackeName<T>("gesvd_work") +
                "\nWrong parameter 12 in " + LapackeName<T>("gesvd_work") +
                "\nWrong parameter 10 in " + LapackeName<T>("bdsqr_work") +
                "\nWrong parameter 12 in " + LapackeName<T>("bdsqr_work") +
                "\nWrong parameter 14 in " + LapackeName<T>("bdsqr_work") +
                "\n");
  // NaN checks.
  a[5] = NaN<T>();
  EXPECT_EQ(LapackeGesvd<T>(LAPACK_COL_MAJOR, 'N', 'N', 4, 4, a.data(), 4,
                            s.data(), u.data(), 1, vt.data(), 1, superb.data()),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, gebrd, gebrd, LAPACK_COL_MAJOR, 4, 4, a.data(), 4,
                         d.data(), e.data(), u.data(), vt.data()),
            -4);
  std::vector<T> tau(4, T(0));
  EXPECT_EQ(LAPACKE_CALL(T, orgbr, ungbr, LAPACK_COL_MAJOR, 'Q', 4, 4, 4,
                         a.data(), 4, tau.data()),
            -6);
  std::vector<T> c(16, T(1));
  EXPECT_EQ(LAPACKE_CALL(T, ormbr, unmbr, LAPACK_COL_MAJOR, 'Q', 'L', 'N', 4, 4,
                         4, a.data(), 4, tau.data(), c.data(), 4),
            -8);
  a[5] = T(1);
  tau[1] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, ormbr, unmbr, LAPACK_COL_MAJOR, 'Q', 'L', 'N', 4, 4,
                         4, a.data(), 4, tau.data(), c.data(), 4),
            -10);
  EXPECT_EQ(LAPACKE_CALL(T, orgbr, ungbr, LAPACK_COL_MAJOR, 'Q', 4, 4, 4,
                         a.data(), 4, tau.data()),
            -8);
  c[3] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, ormbr, unmbr, LAPACK_COL_MAJOR, 'Q', 'L', 'N', 4, 4,
                         4, a.data(), 4, tau.data(), c.data(), 4),
            -11);
  d[2] = NAN;
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_COL_MAJOR, 'U', 4, 0, 0, 0, d.data(),
                   e.data(), vt.data(), 1, u.data(), 1, a.data(), 1),
      -7);
  d[2] = 1;
  e[1] = NAN;
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_COL_MAJOR, 'U', 4, 0, 0, 0, d.data(),
                   e.data(), vt.data(), 1, u.data(), 1, a.data(), 1),
      -8);
  e[1] = 0;
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_COL_MAJOR, 'U', 4, 0, 0, 4, d.data(),
                   e.data(), vt.data(), 1, u.data(), 1, c.data(), 4),
      -13);
  u[0] = NaN<T>();
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_COL_MAJOR, 'U', 4, 0, 4, 0, d.data(),
                   e.data(), vt.data(), 1, u.data(), 4, a.data(), 1),
      -11);
  vt[0] = NaN<T>();
  EXPECT_EQ(
      LAPACKE_CALL(T, bdsqr, bdsqr, LAPACK_COL_MAJOR, 'U', 4, 4, 0, 0, d.data(),
                   e.data(), vt.data(), 4, u.data(), 1, a.data(), 1),
      -9);
}

}  // namespace
}  // namespace hwyblas_test
