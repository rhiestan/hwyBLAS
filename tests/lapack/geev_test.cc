// Tests of the eigenvectors and condition numbers of the nonsymmetric
// eigenproblem: xTREVC3 (eigenvectors of the Schur form), xTRSNA (their
// condition numbers), xLAQTR, the drivers xGEEV, xGEEVX and xGEESX, and
// LAPACKE.
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
class GeevTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GeevTest, ScalarTypes);

template <typename R>
class GeevRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GeevRealTest, RealTypes);

// The eigenvalues, as complex numbers also for real T.
template <typename T>
using Eig = std::complex<RealT<T>>;

using C64 = std::complex<double>;

size_t Sz(hwyblas_int n) {
  return static_cast<size_t>(std::max<hwyblas_int>(1, n));
}

size_t Sz2(hwyblas_int m, hwyblas_int n) { return Sz(m) * Sz(n); }

// ---------------------------------------------------------------------------
// Test matrices

enum class Kind {
  kRandom,
  kZero,
  kIdentity,
  kJordan,      // a Jordan block for 1 (defective)
  kCyclic,      // the cyclic permutation (eigenvalues: roots of unity)
  kTriangular,  // upper triangular
  kGraded,      // random, A(i,j) scaled by 2**(j - i)
  kPairs,       // block diagonal with 2 x 2 rotations (complex pairs)
};
const Kind kKinds[] = {Kind::kRandom, Kind::kZero,   Kind::kIdentity,
                       Kind::kJordan, Kind::kCyclic, Kind::kTriangular,
                       Kind::kGraded, Kind::kPairs};

const char* KindName(Kind kind) {
  switch (kind) {
    case Kind::kRandom:
      return "random";
    case Kind::kZero:
      return "zero";
    case Kind::kIdentity:
      return "identity";
    case Kind::kJordan:
      return "jordan";
    case Kind::kCyclic:
      return "cyclic";
    case Kind::kTriangular:
      return "triangular";
    case Kind::kGraded:
      return "graded";
    case Kind::kPairs:
      return "pairs";
  }
  return "";
}

template <typename T>
std::vector<T> TestMatrix(Kind kind, hwyblas_int n, hwyblas_int lda,
                          Random<T>& rng) {
  using R = RealT<T>;
  std::vector<T> a = rng.Vector(Sz2(lda, n));
  const T zero = T(0);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      T& x = At(a, i, j, lda);
      switch (kind) {
        case Kind::kRandom:
          break;
        case Kind::kZero:
          x = zero;
          break;
        case Kind::kIdentity:
          x = T(R(i == j ? 1 : 0));
          break;
        case Kind::kJordan:
          x = T(R(i == j || i + 1 == j ? 1 : 0));
          break;
        case Kind::kCyclic:
          x = T(R(i == j + 1 || (i == 0 && j == n - 1) ? 1 : 0));
          break;
        case Kind::kTriangular:
          if (i > j) x = zero;
          break;
        case Kind::kGraded:
          x *= static_cast<R>(std::ldexp(1.0, j - i));
          break;
        case Kind::kPairs: {
          const hwyblas_int bi = i / 2, bj = j / 2;
          if (bi != bj) {
            x = zero;
          } else if (i == j) {
            x = T(static_cast<R>(bi + 1));
          } else {
            x = T(static_cast<R>(i < j ? 2 : -3));
          }
          break;
        }
      }
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// Calls with vectors; the eigenvalues as complex numbers.

template <typename R>
hwyblas_int NeverSelect(const R* /*wr*/, const R* /*wi*/) {
  return 0;
}
template <typename R>
hwyblas_int NeverSelectC(const R* /*w*/) {
  return 0;
}
template <typename R>
hwyblas_int PositiveReal(const R* wr, const R* /*wi*/) {
  return *wr > 0 ? 1 : 0;
}
template <typename R>
hwyblas_int PositiveRealC(const R* w) {
  return w[0] > 0 ? 1 : 0;
}

// xGEES with JOBVS = 'V' and SORT = 'N': the Schur form T (in a) and the
// Schur vectors Z of a.
template <typename T>
void Schur(hwyblas_int n, std::vector<T>& a, std::vector<T>& z,
           std::vector<Eig<T>>& w) {
  using R = RealT<T>;
  const hwyblas_int lwork = 64 * std::max<hwyblas_int>(1, n);
  std::vector<T> work(Sz(lwork));
  std::vector<hwyblas_int> bwork(Sz(n));
  z.assign(Sz2(n, n), T(0));
  w.assign(Sz(n), Eig<T>());
  hwyblas_int sdim = 0, info = -99;
  const char v = 'V', s = 'N';
  const hwyblas_int ld = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<R> rwork(Sz(n));
    HWYBLAS_CALL_COMPLEX(T, gees, &v, &s, &NeverSelectC<R>, &n, P(a.data()),
                         &ld, &sdim, P(w.data()), P(z.data()), &ld,
                         P(work.data()), &lwork, rwork.data(), bwork.data(),
                         &info);
  } else {
    std::vector<R> wr(Sz(n)), wi(Sz(n));
    HWYBLAS_CALL_REAL(T, gees, &v, &s, &NeverSelect<R>, &n, a.data(), &ld,
                      &sdim, wr.data(), wi.data(), z.data(), &ld, work.data(),
                      &lwork, bwork.data(), &info);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
  }
  ASSERT_EQ(info, 0);
  w.resize(static_cast<size_t>(n));
}

// A random n x n matrix of the given kind (in a), its Schur form t, Schur
// vectors z and eigenvalues w.
template <typename T>
void RandomSchur(Kind kind, hwyblas_int n, Random<T>& rng, std::vector<T>& a,
                 std::vector<T>& t, std::vector<T>& z, std::vector<Eig<T>>& w) {
  a = TestMatrix<T>(kind, n, n, rng);
  t = a;
  Schur(n, t, z, w);
}

// xTREVC3; *opt returns WORK(1) (the optimal LWORK, also of a query).
template <typename T>
hwyblas_int Trevc3(char side, char howmny, std::vector<hwyblas_int>& select,
                   hwyblas_int n, std::vector<T>& t, hwyblas_int ldt,
                   std::vector<T>& vl, hwyblas_int ldvl, std::vector<T>& vr,
                   hwyblas_int ldvr, hwyblas_int mm, hwyblas_int* m,
                   hwyblas_int lwork, hwyblas_int* opt = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    const hwyblas_int lrwork = lwork == -1 ? -1 : std::max<hwyblas_int>(1, n);
    std::vector<R> rwork(Sz(n), NAN);
    HWYBLAS_CALL_COMPLEX(T, trevc3, &side, &howmny, select.data(), &n,
                         P(t.data()), &ldt, P(vl.data()), &ldvl, P(vr.data()),
                         &ldvr, &mm, m, P(work.data()), &lwork, rwork.data(),
                         &lrwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, trevc3, &side, &howmny, select.data(), &n, t.data(),
                      &ldt, vl.data(), &ldvl, vr.data(), &ldvr, &mm, m,
                      work.data(), &lwork, &info);
  }
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

// xTRSNA with LDWORK = max(1, n).
template <typename T>
hwyblas_int Trsna(char job, char howmny, const std::vector<hwyblas_int>& select,
                  hwyblas_int n, const std::vector<T>& t, hwyblas_int ldt,
                  const std::vector<T>& vl, hwyblas_int ldvl,
                  const std::vector<T>& vr, hwyblas_int ldvr,
                  std::vector<RealT<T>>& s, std::vector<RealT<T>>& sep,
                  hwyblas_int mm, hwyblas_int* m) {
  using R = RealT<T>;
  const hwyblas_int ldwork = std::max<hwyblas_int>(1, n);
  std::vector<T> work(Sz(ldwork) * Sz(n + 6), NaN<T>());
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    std::vector<R> rwork(Sz(n), NAN);
    HWYBLAS_CALL_COMPLEX(T, trsna, &job, &howmny, select.data(), &n,
                         P(t.data()), &ldt, P(vl.data()), &ldvl, P(vr.data()),
                         &ldvr, s.data(), sep.data(), &mm, m, P(work.data()),
                         &ldwork, rwork.data(), &info);
  } else {
    std::vector<hwyblas_int> iwork(Sz(2 * (n - 1)), -7);
    HWYBLAS_CALL_REAL(T, trsna, &job, &howmny, select.data(), &n, t.data(),
                      &ldt, vl.data(), &ldvl, vr.data(), &ldvr, s.data(),
                      sep.data(), &mm, m, work.data(), &ldwork, iwork.data(),
                      &info);
  }
  return info;
}

// xGEEV; for lwork = -1, w[0] returns the optimal LWORK.
template <typename T>
hwyblas_int Geev(char jobvl, char jobvr, hwyblas_int n, std::vector<T>& a,
                 hwyblas_int lda, std::vector<Eig<T>>& w, std::vector<T>& vl,
                 hwyblas_int ldvl, std::vector<T>& vr, hwyblas_int ldvr,
                 hwyblas_int lwork) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  hwyblas_int info = -99;
  w.assign(Sz(n), Eig<T>(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    std::vector<R> rwork(Sz(2 * n), NAN);
    HWYBLAS_CALL_COMPLEX(T, geev, &jobvl, &jobvr, &n, P(a.data()), &lda,
                         P(w.data()), P(vl.data()), &ldvl, P(vr.data()), &ldvr,
                         P(work.data()), &lwork, rwork.data(), &info);
  } else {
    std::vector<R> wr(Sz(n), NAN), wi(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, geev, &jobvl, &jobvr, &n, a.data(), &lda, wr.data(),
                      wi.data(), vl.data(), &ldvl, vr.data(), &ldvr,
                      work.data(), &lwork, &info);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
  }
  w.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (lwork == -1) w.assign(1, Eig<T>(std::real(work[0]), 0));
  return info;
}

// The outputs of xGEEVX besides the eigenvalues and vectors.
template <typename R>
struct GeevxOut {
  hwyblas_int ilo = -1, ihi = -1;
  std::vector<R> scale, rconde, rcondv;
  R abnrm = -1;
  R work0 = -1;  // WORK(1)
};

template <typename T>
hwyblas_int Geevx(char balanc, char jobvl, char jobvr, char sense,
                  hwyblas_int n, std::vector<T>& a, hwyblas_int lda,
                  std::vector<Eig<T>>& w, std::vector<T>& vl, hwyblas_int ldvl,
                  std::vector<T>& vr, hwyblas_int ldvr, GeevxOut<RealT<T>>& out,
                  hwyblas_int lwork) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  out.scale.assign(Sz(n), NAN);
  out.rconde.assign(Sz(n), NAN);
  out.rcondv.assign(Sz(n), NAN);
  hwyblas_int info = -99;
  w.assign(Sz(n), Eig<T>(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    std::vector<R> rwork(Sz(2 * n), NAN);
    HWYBLAS_CALL_COMPLEX(
        T, geevx, &balanc, &jobvl, &jobvr, &sense, &n, P(a.data()), &lda,
        P(w.data()), P(vl.data()), &ldvl, P(vr.data()), &ldvr, &out.ilo,
        &out.ihi, out.scale.data(), &out.abnrm, out.rconde.data(),
        out.rcondv.data(), P(work.data()), &lwork, rwork.data(), &info);
  } else {
    std::vector<R> wr(Sz(n), NAN), wi(Sz(n), NAN);
    std::vector<hwyblas_int> iwork(Sz(2 * n - 2), -7);
    HWYBLAS_CALL_REAL(T, geevx, &balanc, &jobvl, &jobvr, &sense, &n, a.data(),
                      &lda, wr.data(), wi.data(), vl.data(), &ldvl, vr.data(),
                      &ldvr, &out.ilo, &out.ihi, out.scale.data(), &out.abnrm,
                      out.rconde.data(), out.rcondv.data(), work.data(), &lwork,
                      iwork.data(), &info);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
  }
  w.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  out.work0 = std::real(work[0]);
  return info;
}

// xGEESX with SORT = 'S' selecting the eigenvalues with a positive real
// part (or SORT = 'N'); *work0 and *iwork0 return WORK(1) and IWORK(1).
template <typename T>
hwyblas_int Geesx(char jobvs, char sort, char sense, hwyblas_int n,
                  std::vector<T>& a, hwyblas_int lda, hwyblas_int* sdim,
                  std::vector<Eig<T>>& w, std::vector<T>& vs, hwyblas_int ldvs,
                  RealT<T>* rconde, RealT<T>* rcondv, hwyblas_int lwork,
                  hwyblas_int liwork, hwyblas_int* work0 = nullptr,
                  hwyblas_int* iwork0 = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  std::vector<hwyblas_int> bwork(Sz(n), -7);
  hwyblas_int info = -99;
  w.assign(Sz(n), Eig<T>(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    (void)liwork;
    std::vector<R> rwork(Sz(n), NAN);
    HWYBLAS_CALL_COMPLEX(T, geesx, &jobvs, &sort, &PositiveRealC<R>, &sense, &n,
                         P(a.data()), &lda, sdim, P(w.data()), P(vs.data()),
                         &ldvs, rconde, rcondv, P(work.data()), &lwork,
                         rwork.data(), bwork.data(), &info);
    if (iwork0 != nullptr) *iwork0 = -1;
  } else {
    std::vector<R> wr(Sz(n), NAN), wi(Sz(n), NAN);
    std::vector<hwyblas_int> iwork(Sz(liwork), -7);
    HWYBLAS_CALL_REAL(T, geesx, &jobvs, &sort, &PositiveReal<R>, &sense, &n,
                      a.data(), &lda, sdim, wr.data(), wi.data(), vs.data(),
                      &ldvs, rconde, rcondv, work.data(), &lwork, iwork.data(),
                      &liwork, bwork.data(), &info);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
    if (iwork0 != nullptr) *iwork0 = iwork[0];
  }
  w.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (work0 != nullptr) *work0 = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

// ---------------------------------------------------------------------------
// Checks

// Eigenvector j of the matrix V whose columns are the eigenvectors of the
// eigenvalues w (as returned for HOWMNY = 'A' or 'B', or by xGEEV), as a
// complex vector: for real T, the columns j and j+1 of a complex conjugate
// pair hold the real and imaginary part of the vector of w(j) (Im w(j) > 0)
// and its conjugate is the vector of w(j+1).
template <typename T>
std::vector<C64> Eigvec(const std::vector<T>& v, hwyblas_int n, hwyblas_int ld,
                        const std::vector<Eig<T>>& w, hwyblas_int j) {
  std::vector<C64> x(static_cast<size_t>(n));
  for (hwyblas_int k = 0; k < n; ++k) {
    if (IsComplex<T>::value || w[j].imag() == 0) {
      x[k] = static_cast<C64>(At(v, k, j, ld));
    } else if (w[j].imag() > 0) {
      x[k] = C64(std::real(At(v, k, j, ld)), std::real(At(v, k, j + 1, ld)));
    } else {
      x[k] = C64(std::real(At(v, k, j - 1, ld)), -std::real(At(v, k, j, ld)));
    }
  }
  return x;
}

double VecNorm1(const std::vector<C64>& x) {
  double s = 0;
  for (const C64& v : x) s += std::abs(v);
  return s;
}

double Norm2(const std::vector<C64>& x) {
  double s = 0;
  for (const C64& v : x) s += std::norm(v);
  return std::sqrt(s);
}

// The residual ratio of the eigenpair (lambda, x) of the n x n matrix a:
// ||A x - lambda x||_1 (right) or ||x**H A - lambda x**H||_1 (left) /
// (n ||A||_1 ||x||_1 eps), as in LAPACK's xGET22.
template <typename T>
double EigenpairRatio(const std::vector<T>& a, hwyblas_int n, hwyblas_int lda,
                      Eig<T> lambda, const std::vector<C64>& x, bool left) {
  const C64 lam = static_cast<C64>(lambda);
  double resid = 0;
  for (hwyblas_int i = 0; i < n; ++i) {
    C64 r = 0;
    for (hwyblas_int k = 0; k < n; ++k) {
      if (left) {
        // (A**H x)_i - conj(lambda) x_i
        r += std::conj(static_cast<C64>(At(a, k, i, lda))) * x[k];
      } else {
        r += static_cast<C64>(At(a, i, k, lda)) * x[k];
      }
    }
    r -= (left ? std::conj(lam) : lam) * x[i];
    resid += std::abs(r);
  }
  const double anorm = std::max(Norm1(a, n, n, lda), 1e-300);
  const double xnorm = VecNorm1(x);
  if (xnorm == 0) return std::numeric_limits<double>::infinity();
  return resid / anorm / xnorm / std::max<double>(1, n) / LapackEps<T>();
}

// Checks all eigenpairs (w(j), column j of V) of a: right (VR) or left (VL)
// vectors.
template <typename T>
void ExpectEigenvectors(const std::vector<T>& a, hwyblas_int n, hwyblas_int lda,
                        const std::vector<Eig<T>>& w, const std::vector<T>& v,
                        hwyblas_int ldv, bool left) {
  for (hwyblas_int j = 0; j < n; ++j) {
    EXPECT_LT(EigenpairRatio(a, n, lda, w[j], Eigvec(v, n, ldv, w, j), left),
              kLapackThreshold)
        << (left ? "left " : "right ") << j;
  }
}

// xTREVC3's normalization: the largest |Re| + |Im| of a vector is 1.
void ExpectTrevcNormalized(const std::vector<C64>& x, double eps) {
  double big = 0;
  for (const C64& v : x)
    big = std::max(big, std::abs(v.real()) + std::abs(v.imag()));
  EXPECT_NEAR(big, 1.0, 4 * eps);
}

// xGEEV's normalization: Euclidean norm 1, and the largest component real.
void ExpectGeevNormalized(const std::vector<C64>& x, double eps) {
  if (x.empty()) return;
  const double n = static_cast<double>(x.size());
  EXPECT_NEAR(Norm2(x), 1.0, 4 * n * eps);
  double big = 0;
  for (const C64& v : x) big = std::max(big, std::abs(v));
  bool real_max = false;
  for (const C64& v : x) {
    real_max =
        real_max || (v.imag() == 0 && std::abs(v) >= big * (1 - 8 * eps));
  }
  EXPECT_TRUE(real_max);
}

// Real T: complex conjugate pairs are consecutive, with the positive
// imaginary part first.
template <typename T>
void ExpectPairs(const std::vector<Eig<T>>& w) {
  if constexpr (!IsComplex<T>::value) {
    for (size_t i = 0; i < w.size(); ++i) {
      if (w[i].imag() > 0) {
        ASSERT_LT(i + 1, w.size());
        EXPECT_EQ(w[i + 1], std::conj(w[i])) << i;
        ++i;
      } else {
        EXPECT_EQ(w[i].imag(), 0) << i;
      }
    }
  }
}

// The inverse of the m x m complex matrix a by Gauss-Jordan elimination with
// partial pivoting (false if singular).
bool Invert(std::vector<C64>& a, size_t m) {
  std::vector<C64> inv(m * m, C64(0));
  for (size_t i = 0; i < m; ++i) inv[i + i * m] = 1;
  for (size_t k = 0; k < m; ++k) {
    size_t p = k;
    for (size_t i = k + 1; i < m; ++i) {
      if (std::abs(a[i + k * m]) > std::abs(a[p + k * m])) p = i;
    }
    if (a[p + k * m] == C64(0)) return false;
    for (size_t j = 0; j < m; ++j) {
      std::swap(a[k + j * m], a[p + j * m]);
      std::swap(inv[k + j * m], inv[p + j * m]);
    }
    const C64 d = a[k + k * m];
    for (size_t j = 0; j < m; ++j) {
      a[k + j * m] /= d;
      inv[k + j * m] /= d;
    }
    for (size_t i = 0; i < m; ++i) {
      if (i == k) continue;
      const C64 f = a[i + k * m];
      if (f == C64(0)) continue;
      for (size_t j = 0; j < m; ++j) {
        a[i + j * m] -= f * a[k + j * m];
        inv[i + j * m] -= f * inv[k + j * m];
      }
    }
  }
  a = inv;
  return true;
}

// ||inv(T22 - lambda I)||_F, where T = Q [lambda *; 0 T22] Q**H with the
// first column of the unitary Q along the right eigenvector x of lambda
// (Householder): unitarily invariant, so it bounds the SEP of xTRSNA.
template <typename T>
double InverseSepFrobenius(const std::vector<T>& t, hwyblas_int n,
                           hwyblas_int ldt, Eig<T> lambda, std::vector<C64> x) {
  const size_t nn = static_cast<size_t>(n);
  const double xn = Norm2(x);
  for (C64& v : x) v /= xn;
  // Householder H = I - 2 u u**H / (u**H u), with H e1 a multiple of x.
  const double ax = std::abs(x[0]);
  const C64 phase = ax == 0 ? C64(1) : x[0] / ax;
  std::vector<C64> u = x;
  u[0] += phase;
  const double uu = Norm2(u) * Norm2(u);
  const auto h = [&](size_t i, size_t j) {
    return C64(i == j ? 1 : 0) - 2.0 * u[i] * std::conj(u[j]) / uu;
  };
  // M = H(:,2:n)**H (T - lambda I) H(:,2:n).
  std::vector<C64> tl(nn * nn);
  for (size_t j = 0; j < nn; ++j) {
    for (size_t i = 0; i < nn; ++i) {
      tl[i + j * nn] = static_cast<C64>(At(t, i, j, ldt)) -
                       (i == j ? static_cast<C64>(lambda) : C64(0));
    }
  }
  std::vector<C64> th(nn * nn, C64(0));  // (T - lambda I) H
  for (size_t j = 0; j < nn; ++j) {
    for (size_t k = 0; k < nn; ++k) {
      const C64 hkj = h(k, j);
      for (size_t i = 0; i < nn; ++i) th[i + j * nn] += tl[i + k * nn] * hkj;
    }
  }
  const size_t m = nn - 1;
  std::vector<C64> mm(m * m, C64(0));
  for (size_t j = 0; j < m; ++j) {
    for (size_t i = 0; i < m; ++i) {
      C64 s = 0;
      for (size_t k = 0; k < nn; ++k) {
        s += std::conj(h(k, i + 1)) * th[k + (j + 1) * nn];
      }
      mm[i + j * m] = s;
    }
  }
  if (!Invert(mm, m)) return std::numeric_limits<double>::infinity();
  double f = 0;
  for (const C64& v : mm) f += std::norm(v);
  return std::sqrt(f);
}

// Multiplies each eigenvector of V (as in Eigvec) by exp(i theta): for real
// T, rotates the real and imaginary columns of the complex pairs (real
// eigenvectors stay). The eigenvectors stay eigenvectors.
template <typename T>
void RotatePhases(std::vector<T>& v, hwyblas_int n, hwyblas_int ld,
                  const std::vector<Eig<T>>& w, double theta) {
  using R = RealT<T>;
  const R c = static_cast<R>(std::cos(theta));
  const R s = static_cast<R>(std::sin(theta));
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int k = 0; k < n; ++k) {
      if constexpr (IsComplex<T>::value) {
        At(v, k, j, ld) *= T(c, s);
      } else if (w[j].imag() > 0) {
        const T re = At(v, k, j, ld), im = At(v, k, j + 1, ld);
        At(v, k, j, ld) = c * re - s * im;
        At(v, k, j + 1, ld) = s * re + c * im;
      }
    }
    if (!IsComplex<T>::value && w[j].imag() > 0) ++j;
  }
}

// ---------------------------------------------------------------------------
// xTREVC3

// The right (left) eigenvectors of T for HOWMNY = 'A' are the eigenvectors
// of T, zero below (above) the eigenvalue's row and normalized; with
// HOWMNY = 'B' the back-transformed ones are those of A = Z T Z**H. The
// unblocked (LWORK = 3N, xGEMV) and blocked (xGEMM in blocks of NB = 8
// vectors, or the optimal LWORK) back-transforms agree, and HOWMNY = 'S'
// gives the selected columns of HOWMNY = 'A'.
TYPED_TEST(GeevTest, Trevc3) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(201);
    const double eps = LapackEps<T>();
    for (const hwyblas_int n : {0, 1, 2, 3, 6, 17, 40}) {
      for (const Kind kind : kKinds) {
        SCOPED_TRACE(::testing::Message()
                     << "n=" << n << " kind=" << KindName(kind));
        std::vector<T> a, t, z;
        std::vector<Eig<T>> w;
        ASSERT_NO_FATAL_FAILURE(RandomSchur(kind, n, rng, a, t, z, w));
        const hwyblas_int ld = std::max<hwyblas_int>(1, n);
        const hwyblas_int nbmin_lwork = n + 2 * n * 8;
        std::vector<hwyblas_int> none(Sz(n), 0);
        // HOWMNY = 'A', both sides.
        std::vector<T> vl_all(Sz2(n, n), NaN<T>()), vr_all = vl_all;
        hwyblas_int m = -1;
        std::vector<T> tc = t;
        ASSERT_EQ(Trevc3<T>('B', 'A', none, n, tc, ld, vl_all, ld, vr_all, ld,
                            n, &m, std::max<hwyblas_int>(1, 3 * n)),
                  0);
        EXPECT_EQ(m, n);
        EXPECT_EQ(tc, t);  // restored (complex T) or not referenced
        ExpectEigenvectors(t, n, ld, w, vr_all, ld, false);
        ExpectEigenvectors(t, n, ld, w, vl_all, ld, true);
        for (hwyblas_int j = 0; j < n; ++j) {
          ExpectTrevcNormalized(Eigvec(vr_all, n, ld, w, j), eps);
          ExpectTrevcNormalized(Eigvec(vl_all, n, ld, w, j), eps);
          // Zero below the (last row of the) eigenvalue's block for right
          // vectors, above its first row for left ones.
          const bool second = !IsComplex<T>::value && w[j].imag() < 0;
          const bool first = !IsComplex<T>::value && w[j].imag() > 0;
          const hwyblas_int last = first ? j + 1 : j;
          const hwyblas_int top = second ? j - 1 : j;
          for (hwyblas_int k = last + 1; k < n; ++k) {
            EXPECT_EQ(At(vr_all, k, j, ld), T(0)) << k << ", " << j;
          }
          for (hwyblas_int k = 0; k < top; ++k) {
            EXPECT_EQ(At(vl_all, k, j, ld), T(0)) << k << ", " << j;
          }
        }
        // HOWMNY = 'B' with the Schur vectors, unblocked, blocked (NB = 8)
        // and with the optimal workspace.
        hwyblas_int opt = -1;
        std::vector<T> dummy(1);
        ASSERT_EQ(Trevc3<T>('B', 'B', none, n, tc, ld, dummy, ld, dummy, ld, n,
                            &m, -1, &opt),
                  0);
        EXPECT_GE(opt, std::max<hwyblas_int>(1, nbmin_lwork));
        std::vector<T> vl_ref, vr_ref;
        for (const hwyblas_int lwork :
             {std::max<hwyblas_int>(1, 3 * n), nbmin_lwork, opt}) {
          SCOPED_TRACE(::testing::Message() << "lwork=" << lwork);
          std::vector<T> vl = z, vr = z;
          ASSERT_EQ(Trevc3<T>('B', 'B', none, n, tc, ld, vl, ld, vr, ld, n, &m,
                              std::max<hwyblas_int>(1, lwork)),
                    0);
          EXPECT_EQ(m, n);
          ExpectEigenvectors(a, n, ld, w, vr, ld, false);
          ExpectEigenvectors(a, n, ld, w, vl, ld, true);
          for (hwyblas_int j = 0; j < n; ++j) {
            ExpectTrevcNormalized(Eigvec(vr, n, ld, w, j), eps);
            ExpectTrevcNormalized(Eigvec(vl, n, ld, w, j), eps);
          }
          if (vl_ref.empty()) {
            vl_ref = vl;
            vr_ref = vr;
          } else {
            // The same vectors, but for rounding.
            for (size_t i = 0; i < vl.size(); ++i) {
              EXPECT_NEAR(Abs(vl[i] - vl_ref[i]), 0, 64 * n * eps) << i;
              EXPECT_NEAR(Abs(vr[i] - vr_ref[i]), 0, 64 * n * eps) << i;
            }
          }
        }
        // HOWMNY = 'S': every third eigenvalue (both of a real pair if
        // either is selected); one side at a time.
        std::vector<hwyblas_int> select(Sz(n));
        hwyblas_int expected = 0;
        std::vector<hwyblas_int> columns;  // the selected eigenvalues
        for (hwyblas_int j = 0; j < n; ++j) select[j] = (j % 3 == 1);
        for (hwyblas_int j = 0; j < n; ++j) {
          const bool first = !IsComplex<T>::value && w[j].imag() > 0;
          if (first) {
            if (select[j] || select[j + 1]) {
              columns.push_back(j);
              columns.push_back(j + 1);
              expected += 2;
            }
            ++j;
          } else if (select[j]) {
            columns.push_back(j);
            ++expected;
          }
        }
        for (const char side : {'R', 'L'}) {
          std::vector<hwyblas_int> sel = select;
          std::vector<T> vl(Sz2(n, n), NaN<T>()), vr = vl;
          ASSERT_EQ(Trevc3<T>(side, 'S', sel, n, tc, ld, vl, ld, vr, ld, n, &m,
                              std::max<hwyblas_int>(1, 3 * n)),
                    0);
          ASSERT_EQ(m, expected);
          const std::vector<T>& v = side == 'R' ? vr : vl;
          const std::vector<T>& v_all = side == 'R' ? vr_all : vl_all;
          for (size_t c = 0; c < columns.size(); ++c) {
            for (hwyblas_int k = 0; k < n; ++k) {
              EXPECT_EQ(At(v, k, c, ld), At(v_all, k, columns[c], ld))
                  << side << " " << k << ", " << c;
            }
          }
          if constexpr (!IsComplex<T>::value) {
            // SELECT is normalized: the first of a pair is selected.
            for (hwyblas_int j = 0; j < n; ++j) {
              if (w[j].imag() > 0) {
                EXPECT_EQ(sel[j] != 0, select[j] || select[j + 1]) << j;
                EXPECT_EQ(sel[j + 1], 0) << j;
                ++j;
              }
            }
          }
        }
      }
    }
  });
}

// xTREVC3 for triangular systems that need scaling against overflow: a
// diagonal that is nearly constant (tiny differences of eigenvalues) and
// huge off-diagonal elements.
TYPED_TEST(GeevTest, Trevc3Scaling) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(203);
  const hwyblas_int n = 30;
  for (const R big : {R(1), std::sqrt(std::numeric_limits<R>::max()) / 64,
                      std::numeric_limits<R>::max() / 64}) {
    SCOPED_TRACE(::testing::Message() << "big=" << big);
    std::vector<T> t(Sz2(n, n), T(0));
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = 0; i < j; ++i) At(t, i, j, n) = rng.Next() * big;
      At(t, j, j, n) = T(R(1) + R(j) * std::numeric_limits<R>::epsilon());
    }
    std::vector<Eig<T>> w(n);
    for (hwyblas_int j = 0; j < n; ++j) w[j] = Eig<T>(At(t, j, j, n));
    std::vector<hwyblas_int> none(n, 0);
    std::vector<T> vl(Sz2(n, n)), vr(Sz2(n, n));
    hwyblas_int m = -1;
    std::vector<T> tc = t;
    ASSERT_EQ(Trevc3<T>('B', 'A', none, n, tc, n, vl, n, vr, n, n, &m, 3 * n),
              0);
    for (const T& x : vl) ASSERT_TRUE(std::isfinite(std::abs(x)));
    for (const T& x : vr) ASSERT_TRUE(std::isfinite(std::abs(x)));
    for (hwyblas_int j = 0; j < n; ++j) {
      ExpectTrevcNormalized(Eigvec(vr, n, n, w, j), LapackEps<T>());
      ExpectTrevcNormalized(Eigvec(vl, n, n, w, j), LapackEps<T>());
    }
    if (big == R(1)) {
      ExpectEigenvectors(t, n, n, w, vr, n, false);
      ExpectEigenvectors(t, n, n, w, vl, n, true);
    }
  }
}

// Eigenvalues closer than SMIN = max(ulp |lambda|, SMLNUM), SMLNUM =
// unfl * (N / ulp): the difference is replaced by SMIN. T = s [1 1; 0 1+d]
// with d < ulp: the right eigenvector of 1+d is (-s / SMIN, 1), normalized
// (with SMIN = SMLNUM for a tiny s).
TYPED_TEST(GeevTest, Trevc3Perturbation) {
  using T = TypeParam;
  using R = RealT<T>;
  const R ulp = std::numeric_limits<R>::epsilon();
  const R unfl = std::numeric_limits<R>::min();
  for (const R scale : {R(1), R(4) * unfl}) {
    SCOPED_TRACE(::testing::Message() << "scale=" << scale);
    const hwyblas_int n = 2;
    const R lambda = scale * (1 + ulp / 4);
    std::vector<T> t = {T(scale), T(0), T(scale), T(lambda)};
    std::vector<hwyblas_int> none(n, 0);
    std::vector<T> vl(4), vr(4);
    hwyblas_int m = -1;
    ASSERT_EQ(Trevc3<T>('R', 'A', none, n, t, n, vl, n, vr, n, n, &m, 3 * n),
              0);
    const R smin = std::max(ulp * std::abs(lambda), unfl * (R(n) / ulp));
    const R x1 = -scale / smin;
    const R big = std::max(std::abs(x1), R(1));
    EXPECT_NEAR(std::real(vr[2]), x1 / big, 4 * ulp * std::abs(x1 / big));
    EXPECT_NEAR(std::real(vr[3]), 1 / big, 4 * ulp / big);
    EXPECT_EQ(std::imag(vr[2]), 0);
  }
}

// ---------------------------------------------------------------------------
// xTRSNA

// S(j) = |y**H x| / (||x|| ||y||) of the eigenvectors of xTREVC3, and SEP(j)
// within the bounds that ||inv(T22 - lambda I)||_F gives; HOWMNY = 'S'
// gives the selected values of HOWMNY = 'A'; JOB = 'E' and 'V' the parts
// of JOB = 'B'.
TYPED_TEST(GeevTest, Trsna) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(211);
  for (const hwyblas_int n : {1, 2, 3, 5, 12}) {
    for (const Kind kind :
         {Kind::kRandom, Kind::kTriangular, Kind::kGraded, Kind::kPairs}) {
      SCOPED_TRACE(::testing::Message()
                   << "n=" << n << " kind=" << KindName(kind));
      std::vector<T> a, t, z;
      std::vector<Eig<T>> w;
      ASSERT_NO_FATAL_FAILURE(RandomSchur(kind, n, rng, a, t, z, w));
      std::vector<hwyblas_int> all(Sz(n), 0);
      std::vector<T> vl(Sz2(n, n)), vr(Sz2(n, n));
      hwyblas_int m = -1;
      std::vector<T> tc = t;
      ASSERT_EQ(Trevc3<T>('B', 'A', all, n, tc, n, vl, n, vr, n, n, &m, 3 * n),
                0);
      std::vector<R> s(Sz(n), NAN), sep(Sz(n), NAN);
      ASSERT_EQ(Trsna<T>('B', 'A', all, n, t, n, vl, n, vr, n, s, sep, n, &m),
                0);
      EXPECT_EQ(m, n);
      const double eps = LapackEps<T>();
      for (hwyblas_int j = 0; j < n; ++j) {
        const std::vector<C64> x = Eigvec(vr, n, n, w, j);
        const std::vector<C64> y = Eigvec(vl, n, n, w, j);
        C64 yx = 0;
        for (hwyblas_int k = 0; k < n; ++k) yx += std::conj(y[k]) * x[k];
        const double sref = std::abs(yx) / (Norm2(x) * Norm2(y));
        EXPECT_NEAR(s[j], sref, 64 * n * eps) << j;
        if (n == 1) {
          EXPECT_EQ(sep[j], std::abs(At(t, 0, 0, 1)));
          continue;
        }
        const double f = InverseSepFrobenius(t, n, n, w[j], x);
        const double dim = static_cast<double>(n - 1);
        EXPECT_GE(sep[j], 1 / (std::sqrt(2 * dim) * f) / 1.01) << j;
        EXPECT_LE(sep[j], 10 * std::sqrt(2.0) * dim / f) << j;
      }
      // S does not depend on the phases of the eigenvectors (for real T,
      // whose xTREVC3 vectors of a complex pair have orthogonal real and
      // imaginary parts on the 2 x 2 block, the phases make all terms of
      // the condition number nonzero).
      {
        std::vector<T> vlp = vl, vrp = vr;
        RotatePhases(vlp, n, n, w, 0.7);
        RotatePhases(vrp, n, n, w, -1.9);
        std::vector<R> sp(Sz(n), NAN), sepp(Sz(n), NAN);
        ASSERT_EQ(
            Trsna<T>('E', 'A', all, n, t, n, vlp, n, vrp, n, sp, sepp, n, &m),
            0);
        for (hwyblas_int j = 0; j < n; ++j) {
          EXPECT_NEAR(sp[j], s[j], 64 * n * eps) << "phases " << j;
        }
      }
      // JOB = 'E' and 'V'.
      std::vector<R> s2(Sz(n), NAN), sep2(Sz(n), NAN);
      ASSERT_EQ(Trsna<T>('E', 'A', all, n, t, n, vl, n, vr, n, s2, sep2, n, &m),
                0);
      for (hwyblas_int j = 0; j < n; ++j) {
        EXPECT_EQ(s2[j], s[j]);
        EXPECT_TRUE(std::isnan(sep2[j]));
      }
      std::vector<T> none(1);
      ASSERT_EQ(
          Trsna<T>('V', 'A', all, n, t, n, none, 1, none, 1, s2, sep2, n, &m),
          0);
      for (hwyblas_int j = 0; j < n; ++j) EXPECT_EQ(sep2[j], sep[j]);
      // HOWMNY = 'S': the vectors of the selected eigenvalues (xTREVC3 with
      // the same SELECT).
      std::vector<hwyblas_int> select(Sz(n));
      for (hwyblas_int j = 0; j < n; ++j) select[j] = (j % 2 == 0);
      std::vector<hwyblas_int> sel = select;
      std::vector<T> vls(Sz2(n, n)), vrs(Sz2(n, n));
      hwyblas_int ms = -1;
      ASSERT_EQ(
          Trevc3<T>('B', 'S', sel, n, tc, n, vls, n, vrs, n, n, &ms, 3 * n), 0);
      std::vector<R> s3(Sz(n), NAN), sep3(Sz(n), NAN);
      ASSERT_EQ(
          Trsna<T>('B', 'S', select, n, t, n, vls, n, vrs, n, s3, sep3, n, &m),
          0);
      EXPECT_EQ(m, ms);
      hwyblas_int c = 0;
      for (hwyblas_int j = 0; j < n; ++j) {
        const bool first = !IsComplex<T>::value && w[j].imag() > 0;
        const bool chosen =
            select[j] != 0 || (first && select[j + 1] != 0) ||
            (!IsComplex<T>::value && w[j].imag() < 0 && select[j - 1] != 0);
        if (!chosen) continue;
        EXPECT_EQ(s3[c], s[j]) << j;
        EXPECT_EQ(sep3[c], sep[j]) << j;
        ++c;
      }
      EXPECT_EQ(c, m);
    }
  }
}

// ---------------------------------------------------------------------------
// xLAQTR

// A random n x n upper quasi-triangular matrix with 2 x 2 blocks at rows
// 2-3 and n-2 - n-1 (0-based; the latter if they do not overlap; the
// first block is 1 x 1, as xLAQTR needs for LREAL = false).
template <typename R>
std::vector<R> QuasiTriangular(hwyblas_int n, Random<R>& rng) {
  std::vector<R> t = rng.Vector(Sz2(n, n));
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j + 1; i < n; ++i) At(t, i, j, n) = 0;
    At(t, j, j, n) += 2 * (j % 2 == 0 ? 1 : -1);
  }
  for (const hwyblas_int b : {hwyblas_int{2}, n - 2}) {
    if (b >= 1 && b + 1 < n && (b == 2 || b >= 4)) {
      At(t, b + 1, b, n) = -1.5;
      At(t, b + 1, b + 1, n) = At(t, b, b, n);
      At(t, b, b + 1, n) = 2;
    }
  }
  return t;
}

// op(T) p = scale c (LREAL) and op(T + iB)(p + iq) = scale (c + id), with
// B = [b(1) b(2) ... b(n); w I in rows 2:n]: the residual is small and
// 0 < SCALE <= 1; SCALE < 1 for huge c. For LTRAN, op is the transpose of
// T, and the conjugate transpose of T + iB (T**T - iB**T, which the
// documentation calls (T + iB)**T).
TYPED_TEST(GeevRealTest, Laqtr) {
  using R = TypeParam;
  Random<R> rng(223);
  const double eps = LapackEps<R>();
  for (const hwyblas_int n : {1, 2, 5, 9}) {
    const std::vector<R> t = QuasiTriangular<R>(n, rng);
    const std::vector<R> b = rng.Vector(Sz(n));
    const R w = R(0.75);
    for (const hwyblas_int ltran : {0, 1}) {
      for (const hwyblas_int lreal : {1, 0}) {
        for (const R factor : {R(1), std::numeric_limits<R>::max() / 4}) {
          SCOPED_TRACE(::testing::Message()
                       << "n=" << n << " ltran=" << ltran << " lreal=" << lreal
                       << " factor=" << factor);
          const hwyblas_int nx = lreal ? n : 2 * n;
          std::vector<R> x0 = rng.Vector(Sz(nx));
          for (R& v : x0) v *= factor;
          std::vector<R> x = x0, work(Sz(n));
          R scale = -1;
          hwyblas_int info = -99;
          HWYBLAS_CALL_REAL(R, laqtr, &ltran, &lreal, &n, t.data(), &n,
                            b.data(), &w, &scale, x.data(), work.data(), &info);
          ASSERT_EQ(info, 0);
          EXPECT_GT(scale, 0);
          EXPECT_LE(scale, 1);
          if (factor > 1) {
            EXPECT_LT(scale, 1);
          }
          // M = T + iB (complex), op(M) z = scale * (c + id).
          std::vector<C64> mat(static_cast<size_t>(n * n));
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              double im = 0;
              if (!lreal) {
                if (i == 0) im = b[j];
                if (i == j && i > 0) im = w;
              }
              mat[i + j * n] = C64(At(t, i, j, n), im);
            }
          }
          // ||op(M) z - scale (c + id)||_1 /
          // (n eps (||M||_1 ||z||_1 + scale ||c + id||_1)).
          double resid = 0, znorm = 0, cnorm = 0, mnorm = 0;
          for (hwyblas_int i = 0; i < n; ++i) {
            C64 r = 0;
            double col = 0;
            for (hwyblas_int k = 0; k < n; ++k) {
              const C64 zk(x[k], lreal ? 0.0 : x[n + k]);
              r += (ltran ? std::conj(mat[k + i * n]) : mat[i + k * n]) * zk;
              col += std::abs(mat[k + i * n]) + std::abs(mat[i + k * n]);
            }
            const C64 ci(x0[i], lreal ? 0.0 : x0[n + i]);
            r -= static_cast<double>(scale) * ci;
            resid += std::abs(r);
            znorm += std::abs(C64(x[i], lreal ? 0.0 : x[n + i]));
            cnorm += std::abs(ci);
            mnorm = std::max(mnorm, col);
          }
          EXPECT_LT(resid / (n * eps * (mnorm * znorm + scale * cnorm)),
                    kLapackThreshold);
        }
      }
    }
  }
  // A zero diagonal element is perturbed (INFO = 1).
  const hwyblas_int n = 3;
  std::vector<R> t = {1, 0, 0, 2, 0, 0, 3, 4, 5};
  std::vector<R> b(n, 0), x = {1, 1, 1}, work(n);
  const hwyblas_int one = 1;
  const R w = 0;
  R scale = -1;
  hwyblas_int info = -99;
  HWYBLAS_CALL_REAL(R, laqtr, &one, &one, &n, t.data(), &n, b.data(), &w,
                    &scale, x.data(), work.data(), &info);
  EXPECT_EQ(info, 1);
}

// ---------------------------------------------------------------------------
// xGEEV

// The eigenpairs of A (both sides), normalized, the complex conjugate pairs
// of real T consecutive; the eigenvalues of JOBVL = JOBVR = 'N' and of the
// other options the same up to rounding.
TYPED_TEST(GeevTest, Geev) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(227);
    const double eps = LapackEps<T>();
    for (const hwyblas_int n : {0, 1, 2, 3, 7, 16, 33}) {
      for (const Kind kind : kKinds) {
        for (const int variant : {0, 1}) {  // optimal / minimal workspace
          const hwyblas_int lda = n + 2;
          const std::vector<T> a = TestMatrix<T>(kind, n, lda, rng);
          std::vector<Eig<T>> w_ref;
          for (const char jobvl : {'V', 'N'}) {
            for (const char jobvr : {'V', 'N'}) {
              SCOPED_TRACE(::testing::Message()
                           << "n=" << n << " kind=" << KindName(kind) << " "
                           << jobvl << jobvr << " variant=" << variant);
              const bool vectors = jobvl == 'V' || jobvr == 'V';
              hwyblas_int lwork = std::max<hwyblas_int>(
                  1, (IsComplex<T>::value ? 2 : (vectors ? 4 : 3)) * n);
              std::vector<T> ac = a, vl(Sz2(lda, n), NaN<T>()), vr = vl;
              std::vector<Eig<T>> w;
              if (variant == 0) {
                ASSERT_EQ(
                    Geev<T>(jobvl, jobvr, n, ac, lda, w, vl, lda, vr, lda, -1),
                    0);
                const hwyblas_int opt = static_cast<hwyblas_int>(w[0].real());
                EXPECT_GE(opt, lwork);
                lwork = opt;
              }
              ASSERT_EQ(
                  Geev<T>(jobvl, jobvr, n, ac, lda, w, vl, lda, vr, lda, lwork),
                  0);
              ASSERT_NO_FATAL_FAILURE(ExpectPairs<T>(w));
              if (jobvr == 'V') {
                ExpectEigenvectors(a, n, lda, w, vr, lda, false);
                for (hwyblas_int j = 0; j < n; ++j) {
                  ExpectGeevNormalized(Eigvec(vr, n, lda, w, j), eps);
                }
              }
              if (jobvl == 'V') {
                ExpectEigenvectors(a, n, lda, w, vl, lda, true);
                for (hwyblas_int j = 0; j < n; ++j) {
                  ExpectGeevNormalized(Eigvec(vl, n, lda, w, j), eps);
                }
              }
              if (w_ref.empty()) {
                w_ref = w;
              } else if (kind != Kind::kJordan && kind != Kind::kCyclic) {
                // The same eigenvalues as with both vector sets (as a set:
                // xHSEQR's 'E' and 'S' may order them differently).
                double anorm = Norm1(a, n, n, lda);
                for (hwyblas_int i = 0; i < n; ++i) {
                  double d = std::numeric_limits<double>::infinity();
                  for (hwyblas_int k = 0; k < n; ++k) {
                    d = std::min(d, std::abs(static_cast<C64>(w[i]) -
                                             static_cast<C64>(w_ref[k])));
                  }
                  EXPECT_LE(d, 1e3 * n * eps * std::max(1.0, anorm)) << i;
                }
              }
            }
          }
        }
      }
    }
  });
}

// xGEEV on matrices with tiny or huge elements, scaled before and after.
TYPED_TEST(GeevTest, GeevScaling) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(229);
  const R tiny = std::sqrt(std::numeric_limits<R>::min()) *
                 std::numeric_limits<R>::epsilon() / 64;
  for (const R factor : {tiny, 1 / tiny}) {
    for (const hwyblas_int n : {4, 12}) {
      SCOPED_TRACE(::testing::Message() << "factor=" << factor << " n=" << n);
      std::vector<T> a = TestMatrix<T>(Kind::kRandom, n, n, rng);
      for (T& x : a) x *= factor;
      std::vector<T> ac = a, vl(Sz2(n, n)), vr(Sz2(n, n));
      std::vector<Eig<T>> w;
      ASSERT_EQ(Geev<T>('V', 'V', n, ac, n, w, vl, n, vr, n, 64 * n), 0);
      ExpectEigenvectors(a, n, n, w, vr, n, false);
      ExpectEigenvectors(a, n, n, w, vl, n, true);
      for (const Eig<T>& e : w) {
        EXPECT_GT(std::abs(e), factor / 1000);
        EXPECT_LT(std::abs(e), factor * 1000);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// xGEEVX

// For each balancing and SENSE option: the eigenpairs of A; ILO, IHI and
// SCALE as xGEBAL's and ABNRM the 1-norm of the balanced matrix; RCONDE
// the reciprocal condition numbers of the eigenvalues of the balanced
// matrix (from the returned vectors, transformed back); RCONDV the SEP of
// xTRSNA for the Schur form in A.
TYPED_TEST(GeevTest, Geevx) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(233);
    const double eps = LapackEps<T>();
    for (const hwyblas_int n : {0, 1, 2, 5, 13}) {
      for (const Kind kind :
           {Kind::kRandom, Kind::kTriangular, Kind::kGraded, Kind::kPairs}) {
        const hwyblas_int lda = n + 1;
        std::vector<T> a = TestMatrix<T>(kind, n, lda, rng);
        if (kind == Kind::kRandom && n > 2) {
          // Rows and columns of very different sizes, and a zero below the
          // diagonal of the first column (permutation).
          for (hwyblas_int i = 0; i < n; ++i) {
            At(a, i, 1, lda) *= R(1024);
            At(a, 1, i, lda) /= R(1024);
            if (i > 0) At(a, i, 0, lda) = T(0);
          }
        }
        for (const char balanc : {'N', 'P', 'S', 'B'}) {
          for (const char sense : {'N', 'E', 'V', 'B'}) {
            for (const char jobvl : {'V', 'N'}) {
              for (const char jobvr : {'V', 'N'}) {
                if ((sense == 'E' || sense == 'B') &&
                    (jobvl != 'V' || jobvr != 'V')) {
                  continue;
                }
                SCOPED_TRACE(::testing::Message()
                             << "n=" << n << " kind=" << KindName(kind) << " "
                             << balanc << sense << jobvl << jobvr);
                std::vector<T> ac = a, vl(Sz2(lda, n), NaN<T>()), vr = vl;
                std::vector<Eig<T>> w;
                GeevxOut<R> out;
                // Workspace query, then the call with that workspace.
                ASSERT_EQ(Geevx<T>(balanc, jobvl, jobvr, sense, n, ac, lda, w,
                                   vl, lda, vr, lda, out, -1),
                          0);
                const hwyblas_int lwork = static_cast<hwyblas_int>(out.work0);
                ASSERT_EQ(Geevx<T>(balanc, jobvl, jobvr, sense, n, ac, lda, w,
                                   vl, lda, vr, lda, out, lwork),
                          0);
                ASSERT_NO_FATAL_FAILURE(ExpectPairs<T>(w));
                if (jobvr == 'V')
                  ExpectEigenvectors(a, n, lda, w, vr, lda, false);
                if (jobvl == 'V')
                  ExpectEigenvectors(a, n, lda, w, vl, lda, true);
                if (n == 0) continue;  // (quick return before balancing)
                // xGEBAL of A.
                std::vector<T> bal = a;
                std::vector<R> scale(Sz(n));
                hwyblas_int ilo = -1, ihi = -1, info = -99;
                HWYBLAS_CALL(T, gebal, &balanc, &n, P(bal.data()), &lda, &ilo,
                             &ihi, scale.data(), &info);
                ASSERT_EQ(info, 0);
                EXPECT_EQ(out.ilo, ilo);
                EXPECT_EQ(out.ihi, ihi);
                for (hwyblas_int i = 0; i < n; ++i) {
                  EXPECT_EQ(out.scale[i], scale[i]) << i;
                }
                const double bnorm = Norm1(bal, n, n, lda);
                EXPECT_NEAR(out.abnrm, bnorm, 4 * n * eps * bnorm);
                if (sense == 'V' || sense == 'B') {
                  // A holds the Schur form of the balanced matrix.
                  std::vector<hwyblas_int> all(Sz(n), 0);
                  std::vector<R> s(Sz(n)), sep(Sz(n));
                  std::vector<T> none(1);
                  hwyblas_int m = -1;
                  ASSERT_EQ(Trsna<T>('V', 'A', all, n, ac, lda, none, 1, none,
                                     1, s, sep, n, &m),
                            0);
                  for (hwyblas_int j = 0; j < n; ++j) {
                    EXPECT_EQ(out.rcondv[j], sep[j]) << j;
                  }
                }
                if (sense == 'E' || sense == 'B') {
                  // X = P D (xGEBAK('R') of I), X_L = P inv(D): the vectors
                  // of the balanced matrix are inv(X) x = X_L**T x and
                  // X**T y.
                  std::vector<T> xr(Sz2(n, n), T(0)), xl = xr;
                  for (hwyblas_int i = 0; i < n; ++i) {
                    At(xr, i, i, n) = At(xl, i, i, n) = T(1);
                  }
                  const char r = 'R', l = 'L';
                  HWYBLAS_CALL(T, gebak, &balanc, &r, &n, &ilo, &ihi,
                               scale.data(), &n, P(xr.data()), &n, &info);
                  HWYBLAS_CALL(T, gebak, &balanc, &l, &n, &ilo, &ihi,
                               scale.data(), &n, P(xl.data()), &n, &info);
                  for (hwyblas_int j = 0; j < n; ++j) {
                    const std::vector<C64> x = Eigvec(vr, n, lda, w, j);
                    const std::vector<C64> y = Eigvec(vl, n, lda, w, j);
                    std::vector<C64> xb(n), yb(n);
                    C64 yx = 0;
                    for (hwyblas_int i = 0; i < n; ++i) {
                      yx += std::conj(y[i]) * x[i];
                      for (hwyblas_int k = 0; k < n; ++k) {
                        xb[i] += static_cast<C64>(At(xl, k, i, n)) * x[k];
                        yb[i] += static_cast<C64>(At(xr, k, i, n)) * y[k];
                      }
                    }
                    const double sref = std::abs(yx) / (Norm2(xb) * Norm2(yb));
                    EXPECT_NEAR(out.rconde[j], sref, 256 * n * eps) << j;
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

// xGEEVX on a scaled matrix (tiny or huge, scaled before and after):
// RCONDE does not change, RCONDV (a separation of eigenvalues) scales; the
// eigenvalues are matched, as their order may change.
TYPED_TEST(GeevTest, GeevxScaling) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(235);
  const hwyblas_int n = 7;
  const std::vector<T> a0 = TestMatrix<T>(Kind::kRandom, n, n, rng);
  const auto run = [&](R factor, GeevxOut<R>& out, std::vector<Eig<T>>& w) {
    std::vector<T> a = a0, vl(Sz2(n, n)), vr(Sz2(n, n));
    for (T& x : a) x *= factor;
    ASSERT_EQ(Geevx<T>('N', 'V', 'V', 'B', n, a, n, w, vl, n, vr, n, out,
                       n * n + 8 * n),
              0);
  };
  GeevxOut<R> ref;
  std::vector<Eig<T>> wref;
  ASSERT_NO_FATAL_FAILURE(run(R(1), ref, wref));
  const int e = std::numeric_limits<R>::max_exponent - 20;
  for (const R factor : {std::ldexp(R(1), -e), std::ldexp(R(1), e)}) {
    SCOPED_TRACE(::testing::Message() << "factor=" << factor);
    GeevxOut<R> out;
    std::vector<Eig<T>> w;
    ASSERT_NO_FATAL_FAILURE(run(factor, out, w));
    for (hwyblas_int j = 0; j < n; ++j) {
      hwyblas_int k = 0;
      for (hwyblas_int i = 1; i < n; ++i) {
        if (std::abs(static_cast<C64>(w[j]) / static_cast<double>(factor) -
                     static_cast<C64>(wref[i])) <
            std::abs(static_cast<C64>(w[j]) / static_cast<double>(factor) -
                     static_cast<C64>(wref[k]))) {
          k = i;
        }
      }
      EXPECT_NEAR(out.rconde[j], ref.rconde[k], 1e-3 * ref.rconde[k]) << j;
      // RCONDV is an estimate, which depends on the order of the
      // eigenvalues in the Schur form: the same within a small factor.
      const double ratio = out.rcondv[j] / factor / ref.rcondv[k];
      EXPECT_GT(ratio, 0.1) << j;
      EXPECT_LT(ratio, 10) << j;
    }
    EXPECT_NEAR(out.abnrm / factor, ref.abnrm, 1e-3 * ref.abnrm);
  }
}

// ---------------------------------------------------------------------------
// xGEESX

// xGEESX gives the factorization of xGEES (bitwise), and RCONDE / RCONDV
// are those of xTRSEN for the same selection.
TYPED_TEST(GeevTest, Geesx) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(239);
    for (const hwyblas_int n : {0, 1, 2, 3, 8, 19}) {
      for (const Kind kind :
           {Kind::kRandom, Kind::kTriangular, Kind::kPairs, Kind::kIdentity}) {
        const hwyblas_int lda = n + 1;
        const std::vector<T> a = TestMatrix<T>(kind, n, lda, rng);
        for (const char jobvs : {'V', 'N'}) {
          for (const char sort : {'N', 'S'}) {
            for (const char sense : {'N', 'E', 'V', 'B'}) {
              if (sort == 'N' && sense != 'N') continue;
              SCOPED_TRACE(::testing::Message()
                           << "n=" << n << " kind=" << KindName(kind) << " "
                           << jobvs << sort << sense);
              // The reference: xGEES without sorting, then xTRSEN.
              std::vector<T> t0 = a, vs0(Sz2(lda, n), T(0));
              std::vector<Eig<T>> w0;
              hwyblas_int sdim0 = -1;
              R e0 = -1, v0 = -1;
              ASSERT_EQ(
                  Geesx<T>(jobvs, 'N', 'N', n, t0, lda, &sdim0, w0, vs0, lda,
                           &e0, &v0, 64 * std::max<hwyblas_int>(1, n), 1),
                  0);
              // The workspace query.
              std::vector<T> t = a, vs(Sz2(lda, n), T(0));
              std::vector<Eig<T>> w;
              hwyblas_int sdim = -1, lwork = 0, liwork = 0;
              R rconde = -1, rcondv = -1;
              ASSERT_EQ(
                  Geesx<T>(jobvs, sort, sense, n, t, lda, &sdim, w, vs, lda,
                           &rconde, &rcondv, -1, -1, &lwork, &liwork),
                  0);
              const hwyblas_int nn = std::max<hwyblas_int>(1, n);
              EXPECT_GE(lwork, (IsComplex<T>::value ? 2 : 3) * n);
              if (sense != 'N') {
                EXPECT_GE(lwork, (IsComplex<T>::value ? 0 : n) + n * n / 2);
              }
              if (!IsComplex<T>::value) {
                EXPECT_EQ(liwork, (sense == 'V' || sense == 'B') && n > 0
                                      ? n * n / 4
                                      : 1);
              }
              ASSERT_EQ(Geesx<T>(jobvs, sort, sense, n, t, lda, &sdim, w, vs,
                                 lda, &rconde, &rcondv, std::max(nn, lwork),
                                 std::max<hwyblas_int>(1, liwork)),
                        0);
              if (sort == 'N') {
                EXPECT_EQ(t, t0);
                EXPECT_EQ(w, w0);
                if (jobvs == 'V') {
                  EXPECT_EQ(vs, vs0);
                }
                EXPECT_EQ(sdim, 0);
                continue;
              }
              if (n == 0) continue;  // (quick return before xTRSEN)
              // xTRSEN on the unsorted factorization (vs0 holds the Schur
              // vectors if jobvs = 'V').
              std::vector<hwyblas_int> select(Sz(n));
              for (hwyblas_int i = 0; i < n; ++i) select[i] = w0[i].real() > 0;
              hwyblas_int m = -1, info = -99;
              R s = -1, sep = -1;
              std::vector<T> work(Sz(n * n + 4 * n));
              const hwyblas_int lw = static_cast<hwyblas_int>(work.size());
              std::vector<Eig<T>> w1(Sz(n));
              if constexpr (IsComplex<T>::value) {
                HWYBLAS_CALL_COMPLEX(T, trsen, &sense, &jobvs, select.data(),
                                     &n, P(t0.data()), &lda, P(vs0.data()),
                                     &lda, P(w1.data()), &m, &s, &sep,
                                     P(work.data()), &lw, &info);
              } else {
                std::vector<R> wr(Sz(n)), wi(Sz(n));
                std::vector<hwyblas_int> iwork(Sz(n * n));
                const hwyblas_int liw = static_cast<hwyblas_int>(iwork.size());
                HWYBLAS_CALL_REAL(T, trsen, &sense, &jobvs, select.data(), &n,
                                  t0.data(), &lda, vs0.data(), &lda, wr.data(),
                                  wi.data(), &m, &s, &sep, work.data(), &lw,
                                  iwork.data(), &liw, &info);
                for (hwyblas_int i = 0; i < n; ++i)
                  w1[i] = Eig<T>(wr[i], wi[i]);
              }
              ASSERT_EQ(info, 0);
              w1.resize(static_cast<size_t>(n));
              EXPECT_EQ(t, t0);
              EXPECT_EQ(w, w1);
              if (jobvs == 'V') {
                EXPECT_EQ(vs, vs0);
              }
              EXPECT_EQ(sdim, m);
              if (sense == 'E' || sense == 'B') {
                EXPECT_EQ(rconde, s);
              }
              if (sense == 'V' || sense == 'B') {
                EXPECT_EQ(rcondv, sep);
              }
            }
          }
        }
      }
    }
  });
}

// After the factorization, WORK(1) of xGEESX is at least the workspace
// that xTRSEN needed for SDIM: N + 2*SDIM*(N-SDIM) (real), 2*SDIM*(N-SDIM)
// (complex), and IWORK(1) = SDIM*(N-SDIM) (real, SENSE = 'V' or 'B').
TYPED_TEST(GeevTest, GeesxWorkSizes) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(243);
  const hwyblas_int n = 100;
  std::vector<T> a = TestMatrix<T>(Kind::kTriangular, n, n, rng);
  for (hwyblas_int i = 0; i < n; ++i) At(a, i, i, n) = T(R(i) - R(49.5));
  for (const char sense : {'E', 'V', 'B'}) {
    SCOPED_TRACE(sense);
    std::vector<T> t = a, vs(Sz2(n, n));
    std::vector<Eig<T>> w;
    hwyblas_int sdim = -1, work0 = 0, iwork0 = 0;
    R e = -1, v = -1;
    ASSERT_EQ(Geesx<T>('V', 'S', sense, n, t, n, &sdim, w, vs, n, &e, &v, n * n,
                       n * n, &work0, &iwork0),
              0);
    ASSERT_EQ(sdim, 50);
    const hwyblas_int trsen = 2 * sdim * (n - sdim);
    EXPECT_GE(work0, (IsComplex<T>::value ? 0 : n) + trsen);
    if (!IsComplex<T>::value) {
      EXPECT_EQ(iwork0, sense == 'E' ? 1 : sdim * (n - sdim));
    }
  }
}

// xGEESX with too little workspace for the condition numbers of SDIM
// selected eigenvalues: the factorization is complete, INFO = -16 / -18
// (real) or -15 (complex), and xTRSEN reports the error.
TYPED_TEST(GeevTest, GeesxWorkspaceForSdim) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(241);
  const hwyblas_int n = 12;
  // Eigenvalues 1..12 (half of them positive after the shift).
  std::vector<T> a = TestMatrix<T>(Kind::kTriangular, n, n, rng);
  for (hwyblas_int i = 0; i < n; ++i) At(a, i, i, n) = T(R(i) - R(5.5));
  ErrorCapture errors;
  std::vector<T> t = a, vs(Sz2(n, n));
  std::vector<Eig<T>> w;
  hwyblas_int sdim = -1;
  R e = -1, v = -1;
  const hwyblas_int minwork = (IsComplex<T>::value ? 2 : 3) * n;
  EXPECT_EQ(
      Geesx<T>('V', 'S', 'B', n, t, n, &sdim, w, vs, n, &e, &v, minwork, n * n),
      IsComplex<T>::value ? -15 : -16);
  EXPECT_EQ(sdim, 6);
  // The first Schur vector is an eigenvector (the eigenvalues are real).
  std::vector<C64> z1(n);
  for (hwyblas_int i = 0; i < n; ++i) z1[i] = static_cast<C64>(vs[i]);
  EXPECT_LT(EigenpairRatio(a, n, n, w[0], z1, false), kLapackThreshold);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{Name<T>("TRSEN") + ":" +
                                      (IsComplex<T>::value ? "14" : "15")}));
  if constexpr (!IsComplex<T>::value) {
    t = a;
    EXPECT_EQ(Geesx<T>('V', 'S', 'V', n, t, n, &sdim, w, vs, n, &e, &v,
                       n * n + 4 * n, 1),
              -18);
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{Name<T>("TRSEN") + ":17"}));
  }
}

// ---------------------------------------------------------------------------
// Errors

TYPED_TEST(GeevTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const hwyblas_int n = 6;
  std::vector<T> a(Sz2(n, n), T(1)), vl(Sz2(n, n)), vr(Sz2(n, n));
  std::vector<Eig<T>> w;
  ErrorCapture errors;
  const std::string geev = Name<T>("GEEV");
  const hwyblas_int big = 64 * n;
  EXPECT_EQ(Geev<T>('X', 'N', n, a, n, w, vl, n, vr, n, big), -1);
  EXPECT_EQ(Geev<T>('N', 'X', n, a, n, w, vl, n, vr, n, big), -2);
  EXPECT_EQ(Geev<T>('N', 'N', -1, a, n, w, vl, n, vr, n, big), -3);
  EXPECT_EQ(Geev<T>('N', 'N', n, a, n - 1, w, vl, n, vr, n, big), -5);
  const hwyblas_int vl_pos = kComplex ? 8 : 9, vr_pos = kComplex ? 10 : 11;
  const hwyblas_int lw_pos = kComplex ? 12 : 13;
  EXPECT_EQ(Geev<T>('V', 'N', n, a, n, w, vl, n - 1, vr, n, big), -vl_pos);
  EXPECT_EQ(Geev<T>('N', 'N', n, a, n, w, vl, 0, vr, n, big), -vl_pos);
  EXPECT_EQ(Geev<T>('N', 'V', n, a, n, w, vl, n, vr, n - 1, big), -vr_pos);
  EXPECT_EQ(
      Geev<T>('N', 'V', n, a, n, w, vl, n, vr, n, (kComplex ? 2 : 4) * n - 1),
      -lw_pos);
  EXPECT_EQ(
      Geev<T>('N', 'N', n, a, n, w, vl, n, vr, n, (kComplex ? 2 : 3) * n - 1),
      -lw_pos);
  const auto at = [](const std::string& name, hwyblas_int pos) {
    return name + ":" + std::to_string(pos);
  };
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{at(geev, 1), at(geev, 2), at(geev, 3),
                                      at(geev, 5), at(geev, vl_pos),
                                      at(geev, vl_pos), at(geev, vr_pos),
                                      at(geev, lw_pos), at(geev, lw_pos)}));

  // xGEEVX.
  const std::string geevx = Name<T>("GEEVX");
  GeevxOut<R> out;
  EXPECT_EQ(Geevx<T>('X', 'N', 'N', 'N', n, a, n, w, vl, n, vr, n, out, big),
            -1);
  EXPECT_EQ(Geevx<T>('N', 'X', 'N', 'N', n, a, n, w, vl, n, vr, n, out, big),
            -2);
  EXPECT_EQ(Geevx<T>('N', 'N', 'X', 'N', n, a, n, w, vl, n, vr, n, out, big),
            -3);
  EXPECT_EQ(Geevx<T>('N', 'N', 'N', 'X', n, a, n, w, vl, n, vr, n, out, big),
            -4);
  EXPECT_EQ(Geevx<T>('N', 'V', 'N', 'E', n, a, n, w, vl, n, vr, n, out, big),
            -4);
  EXPECT_EQ(Geevx<T>('N', 'N', 'N', 'N', -1, a, n, w, vl, n, vr, n, out, big),
            -5);
  EXPECT_EQ(
      Geevx<T>('N', 'N', 'N', 'N', n, a, n - 1, w, vl, n, vr, n, out, big), -7);
  const hwyblas_int xvl = kComplex ? 10 : 11, xvr = kComplex ? 12 : 13;
  const hwyblas_int xlw = kComplex ? 20 : 21;
  EXPECT_EQ(
      Geevx<T>('N', 'V', 'N', 'N', n, a, n, w, vl, n - 1, vr, n, out, big),
      -xvl);
  EXPECT_EQ(
      Geevx<T>('N', 'N', 'V', 'N', n, a, n, w, vl, n, vr, n - 1, out, big),
      -xvr);
  EXPECT_EQ(Geevx<T>('N', 'N', 'N', 'V', n, a, n, w, vl, n, vr, n, out,
                     n * n + (kComplex ? 2 : 6) * n - 1),
            -xlw);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{at(geevx, 1), at(geevx, 2), at(geevx, 3),
                                      at(geevx, 4), at(geevx, 4), at(geevx, 5),
                                      at(geevx, 7), at(geevx, xvl),
                                      at(geevx, xvr), at(geevx, xlw)}));

  // xGEESX.
  const std::string geesx = Name<T>("GEESX");
  hwyblas_int sdim = 0;
  R e = 0, v = 0;
  EXPECT_EQ(Geesx<T>('X', 'N', 'N', n, a, n, &sdim, w, vl, n, &e, &v, big, 1),
            -1);
  EXPECT_EQ(Geesx<T>('V', 'X', 'N', n, a, n, &sdim, w, vl, n, &e, &v, big, 1),
            -2);
  EXPECT_EQ(Geesx<T>('V', 'S', 'X', n, a, n, &sdim, w, vl, n, &e, &v, big, 1),
            -4);
  EXPECT_EQ(Geesx<T>('V', 'N', 'E', n, a, n, &sdim, w, vl, n, &e, &v, big, 1),
            -4);
  EXPECT_EQ(Geesx<T>('V', 'N', 'N', -1, a, n, &sdim, w, vl, n, &e, &v, big, 1),
            -5);
  EXPECT_EQ(
      Geesx<T>('V', 'N', 'N', n, a, n - 1, &sdim, w, vl, n, &e, &v, big, 1),
      -7);
  const hwyblas_int svs = kComplex ? 11 : 12, slw = kComplex ? 15 : 16;
  EXPECT_EQ(
      Geesx<T>('V', 'N', 'N', n, a, n, &sdim, w, vl, n - 1, &e, &v, big, 1),
      -svs);
  EXPECT_EQ(Geesx<T>('V', 'N', 'N', n, a, n, &sdim, w, vl, n, &e, &v,
                     (kComplex ? 2 : 3) * n - 1, 1),
            -slw);
  std::vector<std::string> expected = {
      at(geesx, 1), at(geesx, 2), at(geesx, 4),   at(geesx, 4),
      at(geesx, 5), at(geesx, 7), at(geesx, svs), at(geesx, slw)};
  if constexpr (!kComplex) {
    EXPECT_EQ(Geesx<T>('V', 'N', 'N', n, a, n, &sdim, w, vl, n, &e, &v, big, 0),
              -18);
    expected.push_back(at(geesx, 18));
  }
  EXPECT_EQ(errors.Take(), expected);

  // xTREVC3.
  const std::string trevc = Name<T>("TREVC3");
  std::vector<hwyblas_int> sel(Sz(n), 1);
  hwyblas_int m = 0;
  std::vector<T> t(Sz2(n, n), T(0));
  EXPECT_EQ(Trevc3<T>('X', 'A', sel, n, t, n, vl, n, vr, n, n, &m, big), -1);
  EXPECT_EQ(Trevc3<T>('R', 'X', sel, n, t, n, vl, n, vr, n, n, &m, big), -2);
  EXPECT_EQ(Trevc3<T>('R', 'A', sel, -1, t, n, vl, n, vr, n, n, &m, big), -4);
  EXPECT_EQ(Trevc3<T>('R', 'A', sel, n, t, n - 1, vl, n, vr, n, n, &m, big),
            -6);
  EXPECT_EQ(Trevc3<T>('L', 'A', sel, n, t, n, vl, n - 1, vr, n, n, &m, big),
            -8);
  EXPECT_EQ(Trevc3<T>('R', 'A', sel, n, t, n, vl, 0, vr, n, n, &m, big), -8);
  EXPECT_EQ(Trevc3<T>('R', 'A', sel, n, t, n, vl, n, vr, n - 1, n, &m, big),
            -10);
  EXPECT_EQ(Trevc3<T>('R', 'S', sel, n, t, n, vl, n, vr, n, n - 1, &m, big),
            -11);
  EXPECT_EQ(Trevc3<T>('R', 'A', sel, n, t, n, vl, n, vr, n, n, &m,
                      (kComplex ? 2 : 3) * n - 1),
            -14);
  std::vector<std::string> trevc_errors = {
      at(trevc, 1), at(trevc, 2),  at(trevc, 4),  at(trevc, 6), at(trevc, 8),
      at(trevc, 8), at(trevc, 10), at(trevc, 11), at(trevc, 14)};
  if constexpr (kComplex) {
    // LRWORK < N.
    std::vector<T> work(big);
    std::vector<R> rwork(n);
    const char r = 'R', all = 'A';
    const hwyblas_int lrwork = n - 1;
    hwyblas_int info = 0;
    HWYBLAS_CALL_COMPLEX(T, trevc3, &r, &all, sel.data(), &n, P(t.data()), &n,
                         P(vl.data()), &n, P(vr.data()), &n, &n, &m,
                         P(work.data()), &big, rwork.data(), &lrwork, &info);
    EXPECT_EQ(info, -16);
    trevc_errors.push_back(at(trevc, 16));
  }
  EXPECT_EQ(errors.Take(), trevc_errors);

  // xTRSNA.
  const std::string trsna = Name<T>("TRSNA");
  std::vector<R> s(n), sep(n);
  EXPECT_EQ(Trsna<T>('X', 'A', sel, n, t, n, vl, n, vr, n, s, sep, n, &m), -1);
  EXPECT_EQ(Trsna<T>('B', 'X', sel, n, t, n, vl, n, vr, n, s, sep, n, &m), -2);
  EXPECT_EQ(Trsna<T>('B', 'A', sel, -1, t, n, vl, n, vr, n, s, sep, n, &m), -4);
  EXPECT_EQ(Trsna<T>('B', 'A', sel, n, t, n - 1, vl, n, vr, n, s, sep, n, &m),
            -6);
  EXPECT_EQ(Trsna<T>('E', 'A', sel, n, t, n, vl, n - 1, vr, n, s, sep, n, &m),
            -8);
  EXPECT_EQ(Trsna<T>('E', 'A', sel, n, t, n, vl, n, vr, n - 1, s, sep, n, &m),
            -10);
  EXPECT_EQ(Trsna<T>('V', 'A', sel, n, t, n, vl, 1, vr, 1, s, sep, n, &m), 0);
  EXPECT_EQ(Trsna<T>('B', 'A', sel, n, t, n, vl, n, vr, n, s, sep, n - 1, &m),
            -13);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{at(trsna, 1), at(trsna, 2), at(trsna, 4),
                                      at(trsna, 6), at(trsna, 8), at(trsna, 10),
                                      at(trsna, 13)}));
}

// The workspace queries return the reference's sizes for N = 0 and enough
// for the routines otherwise.
TYPED_TEST(GeevTest, WorkspaceQueries) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(1), vl(1), vr(1);
  std::vector<Eig<T>> w;
  ASSERT_EQ(Geev<T>('V', 'V', 0, a, 1, w, vl, 1, vr, 1, -1), 0);
  EXPECT_EQ(w[0].real(), 1);
  GeevxOut<R> out;
  ASSERT_EQ(Geevx<T>('B', 'V', 'V', 'B', 0, a, 1, w, vl, 1, vr, 1, out, -1), 0);
  EXPECT_EQ(out.work0, 1);
  hwyblas_int sdim = -1, lwork = 0, liwork = 0;
  R e = 0, v = 0;
  ASSERT_EQ(Geesx<T>('V', 'S', 'B', 0, a, 1, &sdim, w, vl, 1, &e, &v, -1, -1,
                     &lwork, &liwork),
            0);
  EXPECT_EQ(lwork, 1);
  if (!IsComplex<T>::value) {
    EXPECT_EQ(liwork, 1);
  }
  // xTREVC3: N + 2*N*NB with ILAENV's NB.
  std::vector<hwyblas_int> sel(1);
  hwyblas_int m = 0, opt = 0;
  const hwyblas_int n = 10;
  std::vector<T> t(Sz2(n, n));
  ASSERT_EQ(Trevc3<T>('B', 'B', sel, n, t, n, vl, n, vr, n, n, &m, -1, &opt),
            0);
  EXPECT_GE(opt, n + 2 * n * 8);
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
std::vector<T> ToRowMajor(const std::vector<T>& col, lapack_int ld) {
  std::vector<T> row(col.size());
  for (lapack_int j = 0; j < ld; ++j) {
    for (lapack_int i = 0; i < ld; ++i) row[i * ld + j] = col[i + j * ld];
  }
  return row;
}

template <typename T>
void ExpectSameRowMajor(const std::vector<T>& row, const std::vector<T>& col,
                        lapack_int m, lapack_int n, lapack_int ld) {
  for (lapack_int i = 0; i < m; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      ASSERT_EQ(row[i * ld + j], col[i + j * ld]) << i << ", " << j;
    }
  }
}

// LAPACKE_?geev, ?geevx, ?geesx with the eigenvalues as complex numbers.
template <typename T>
lapack_int LapackeGeev(int layout, char jobvl, char jobvr, lapack_int n, T* a,
                       lapack_int lda, std::vector<Eig<T>>& w, T* vl,
                       lapack_int ldvl, T* vr, lapack_int ldvr) {
  using R = RealT<T>;
  w.assign(Sz(n), Eig<T>());
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_cgeev(layout, jobvl, jobvr, n, a, lda, w.data(), vl, ldvl,
                         vr, ldvr);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    return LAPACKE_zgeev(layout, jobvl, jobvr, n, a, lda, w.data(), vl, ldvl,
                         vr, ldvr);
  } else {
    std::vector<R> wr(Sz(n)), wi(Sz(n));
    lapack_int info;
    if constexpr (std::is_same_v<T, float>) {
      info = LAPACKE_sgeev(layout, jobvl, jobvr, n, a, lda, wr.data(),
                           wi.data(), vl, ldvl, vr, ldvr);
    } else {
      info = LAPACKE_dgeev(layout, jobvl, jobvr, n, a, lda, wr.data(),
                           wi.data(), vl, ldvl, vr, ldvr);
    }
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
    return info;
  }
}

template <typename T>
lapack_int LapackeGeevx(int layout, char balanc, char jobvl, char jobvr,
                        char sense, lapack_int n, T* a, lapack_int lda,
                        std::vector<Eig<T>>& w, T* vl, lapack_int ldvl, T* vr,
                        lapack_int ldvr, GeevxOut<RealT<T>>& out) {
  using R = RealT<T>;
  w.assign(Sz(n), Eig<T>());
  out.scale.assign(Sz(n), 0);
  out.rconde.assign(Sz(n), 0);
  out.rcondv.assign(Sz(n), 0);
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_cgeevx(layout, balanc, jobvl, jobvr, sense, n, a, lda,
                          w.data(), vl, ldvl, vr, ldvr, &out.ilo, &out.ihi,
                          out.scale.data(), &out.abnrm, out.rconde.data(),
                          out.rcondv.data());
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    return LAPACKE_zgeevx(layout, balanc, jobvl, jobvr, sense, n, a, lda,
                          w.data(), vl, ldvl, vr, ldvr, &out.ilo, &out.ihi,
                          out.scale.data(), &out.abnrm, out.rconde.data(),
                          out.rcondv.data());
  } else {
    std::vector<R> wr(Sz(n)), wi(Sz(n));
    lapack_int info;
    if constexpr (std::is_same_v<T, float>) {
      info = LAPACKE_sgeevx(layout, balanc, jobvl, jobvr, sense, n, a, lda,
                            wr.data(), wi.data(), vl, ldvl, vr, ldvr, &out.ilo,
                            &out.ihi, out.scale.data(), &out.abnrm,
                            out.rconde.data(), out.rcondv.data());
    } else {
      info = LAPACKE_dgeevx(layout, balanc, jobvl, jobvr, sense, n, a, lda,
                            wr.data(), wi.data(), vl, ldvl, vr, ldvr, &out.ilo,
                            &out.ihi, out.scale.data(), &out.abnrm,
                            out.rconde.data(), out.rcondv.data());
    }
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
    return info;
  }
}

template <typename T>
lapack_int LapackeGeesx(int layout, char jobvs, char sort, char sense,
                        lapack_int n, T* a, lapack_int lda, lapack_int* sdim,
                        std::vector<Eig<T>>& w, T* vs, lapack_int ldvs,
                        RealT<T>* rconde, RealT<T>* rcondv) {
  using R = RealT<T>;
  w.assign(Sz(n), Eig<T>());
  if constexpr (IsComplex<T>::value) {
    lapack_logical (*sel)(const T*) = [](const T* x) -> lapack_logical {
      return x->real() > 0;
    };
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      return LAPACKE_cgeesx(layout, jobvs, sort, sel, sense, n, a, lda, sdim,
                            w.data(), vs, ldvs, rconde, rcondv);
    } else {
      return LAPACKE_zgeesx(layout, jobvs, sort, sel, sense, n, a, lda, sdim,
                            w.data(), vs, ldvs, rconde, rcondv);
    }
  } else {
    lapack_logical (*sel)(const R*, const R*) =
        [](const R* re, const R*) -> lapack_logical { return *re > 0; };
    std::vector<R> wr(Sz(n)), wi(Sz(n));
    lapack_int info;
    if constexpr (std::is_same_v<T, float>) {
      info = LAPACKE_sgeesx(layout, jobvs, sort, sel, sense, n, a, lda, sdim,
                            wr.data(), wi.data(), vs, ldvs, rconde, rcondv);
    } else {
      info = LAPACKE_dgeesx(layout, jobvs, sort, sel, sense, n, a, lda, sdim,
                            wr.data(), wi.data(), vs, ldvs, rconde, rcondv);
    }
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
    return info;
  }
}

template <typename T>
lapack_int LapackeTrsna(int layout, char job, char howmny,
                        const lapack_logical* select, lapack_int n, const T* t,
                        lapack_int ldt, const T* vl, lapack_int ldvl,
                        const T* vr, lapack_int ldvr, RealT<T>* s,
                        RealT<T>* sep, lapack_int mm, lapack_int* m) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_strsna(layout, job, howmny, select, n, t, ldt, vl, ldvl, vr,
                          ldvr, s, sep, mm, m);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dtrsna(layout, job, howmny, select, n, t, ldt, vl, ldvl, vr,
                          ldvr, s, sep, mm, m);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_ctrsna(layout, job, howmny, select, n, t, ldt, vl, ldvl, vr,
                          ldvr, s, sep, mm, m);
  } else {
    return LAPACKE_ztrsna(layout, job, howmny, select, n, t, ldt, vl, ldvl, vr,
                          ldvr, s, sep, mm, m);
  }
}

// Row-major results equal the column-major ones, and both those of the
// LAPACK routines.
TYPED_TEST(GeevTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(251);
  const lapack_int n = 9, ld = 11;
  const std::vector<T> a0 = rng.Vector(static_cast<size_t>(ld) * ld);
  const std::vector<T> ar0 = ToRowMajor(a0, ld);
  const size_t size = static_cast<size_t>(ld) * ld;
  // ?geev.
  for (const char jobvl : {'V', 'N'}) {
    for (const char jobvr : {'V', 'N'}) {
      SCOPED_TRACE(::testing::Message() << "geev " << jobvl << jobvr);
      std::vector<T> col = a0, row = ar0;
      std::vector<T> lc(size), rc(size), lr(size), rr(size);
      std::vector<Eig<T>> wc, wr;
      ASSERT_EQ(LapackeGeev<T>(LAPACK_COL_MAJOR, jobvl, jobvr, n, col.data(),
                               ld, wc, lc.data(), ld, rc.data(), ld),
                0);
      ASSERT_EQ(LapackeGeev<T>(LAPACK_ROW_MAJOR, jobvl, jobvr, n, row.data(),
                               ld, wr, lr.data(), ld, rr.data(), ld),
                0);
      EXPECT_EQ(wr, wc);
      ExpectSameRowMajor(row, col, n, n, ld);
      if (jobvl == 'V') ExpectSameRowMajor(lr, lc, n, n, ld);
      if (jobvr == 'V') ExpectSameRowMajor(rr, rc, n, n, ld);
      // The LAPACK routine (with the optimal workspace).
      std::vector<T> ref = a0, lref(size), rref(size);
      std::vector<Eig<T>> wref;
      std::vector<T> q = a0;
      ASSERT_EQ(Geev<T>(jobvl, jobvr, n, q, ld, wref, lref, ld, rref, ld, -1),
                0);
      const hwyblas_int lwork = static_cast<hwyblas_int>(wref[0].real());
      ASSERT_EQ(
          Geev<T>(jobvl, jobvr, n, ref, ld, wref, lref, ld, rref, ld, lwork),
          0);
      EXPECT_EQ(wc, wref);
      if (jobvr == 'V') {
        for (lapack_int j = 0; j < n; ++j) {
          for (lapack_int i = 0; i < n; ++i) {
            ASSERT_EQ(At(rc, i, j, ld), At(rref, i, j, ld));
          }
        }
      }
    }
  }
  // ?geevx.
  for (const char sense : {'N', 'E', 'V', 'B'}) {
    SCOPED_TRACE(::testing::Message() << "geevx " << sense);
    std::vector<T> col = a0, row = ar0;
    std::vector<T> lc(size), rc(size), lr(size), rr(size);
    std::vector<Eig<T>> wc, wr;
    GeevxOut<R> oc, orow;
    ASSERT_EQ(
        LapackeGeevx<T>(LAPACK_COL_MAJOR, 'B', 'V', 'V', sense, n, col.data(),
                        ld, wc, lc.data(), ld, rc.data(), ld, oc),
        0);
    ASSERT_EQ(
        LapackeGeevx<T>(LAPACK_ROW_MAJOR, 'B', 'V', 'V', sense, n, row.data(),
                        ld, wr, lr.data(), ld, rr.data(), ld, orow),
        0);
    EXPECT_EQ(wr, wc);
    ExpectSameRowMajor(row, col, n, n, ld);
    ExpectSameRowMajor(lr, lc, n, n, ld);
    ExpectSameRowMajor(rr, rc, n, n, ld);
    EXPECT_EQ(orow.ilo, oc.ilo);
    EXPECT_EQ(orow.ihi, oc.ihi);
    EXPECT_EQ(orow.scale, oc.scale);
    EXPECT_EQ(orow.abnrm, oc.abnrm);
    if (sense == 'E' || sense == 'B') {
      EXPECT_EQ(orow.rconde, oc.rconde);
    }
    if (sense == 'V' || sense == 'B') {
      EXPECT_EQ(orow.rcondv, oc.rcondv);
    }
  }
  // ?geesx.
  for (const char sense : {'N', 'E', 'V', 'B'}) {
    for (const char sort : {'N', 'S'}) {
      if (sort == 'N' && sense != 'N') continue;
      SCOPED_TRACE(::testing::Message() << "geesx " << sort << sense);
      std::vector<T> col = a0, row = ar0, vc(size), vr(size);
      std::vector<Eig<T>> wc, wr;
      lapack_int sc = -1, sr = -1;
      R ec = -1, er = -1, vcn = -1, vrn = -1;
      ASSERT_EQ(
          LapackeGeesx<T>(LAPACK_COL_MAJOR, 'V', sort, sense, n, col.data(), ld,
                          &sc, wc, vc.data(), ld, &ec, &vcn),
          0);
      ASSERT_EQ(
          LapackeGeesx<T>(LAPACK_ROW_MAJOR, 'V', sort, sense, n, row.data(), ld,
                          &sr, wr, vr.data(), ld, &er, &vrn),
          0);
      EXPECT_EQ(wr, wc);
      EXPECT_EQ(sr, sc);
      ExpectSameRowMajor(row, col, n, n, ld);
      ExpectSameRowMajor(vr, vc, n, n, ld);
      if (sense == 'E' || sense == 'B') {
        EXPECT_EQ(er, ec);
      }
      if (sense == 'V' || sense == 'B') {
        EXPECT_EQ(vrn, vcn);
      }
    }
  }
  // ?trsna on the Schur form of A.
  std::vector<T> t = a0, z;
  std::vector<Eig<T>> w;
  t.resize(static_cast<size_t>(n) * n);
  for (lapack_int j = 0; j < n; ++j) {
    for (lapack_int i = 0; i < n; ++i) t[i + j * n] = a0[i + j * ld];
  }
  Schur(n, t, z, w);
  std::vector<hwyblas_int> none(n, 0);
  std::vector<T> vl(static_cast<size_t>(n) * n), vr(vl.size());
  hwyblas_int m = 0;
  std::vector<T> tc = t;
  ASSERT_EQ(Trevc3<T>('B', 'A', none, n, tc, n, vl, n, vr, n, n, &m, 3 * n), 0);
  // VL and VR: n x mm (mm = n + 1 columns, the last unused).
  const lapack_int mm = n + 1, ldv = mm + 1;
  std::vector<T> vlc(static_cast<size_t>(ldv) * mm), vrc(vlc.size());
  std::vector<T> vlr(static_cast<size_t>(n) * ldv), vrr(vlr.size());
  for (lapack_int j = 0; j < n; ++j) {
    for (lapack_int i = 0; i < n; ++i) {
      vlc[i + j * ldv] = vlr[i * ldv + j] = vl[i + j * n];
      vrc[i + j * ldv] = vrr[i * ldv + j] = vr[i + j * n];
    }
  }
  const std::vector<T> tr = ToRowMajor(t, n);
  for (const char job : {'E', 'V', 'B'}) {
    SCOPED_TRACE(::testing::Message() << "trsna " << job);
    std::vector<R> sc(mm, -1), sepc(mm, -1), sr(mm, -1), sepr(mm, -1);
    lapack_int mc = -1, mr = -1;
    std::vector<lapack_logical> sel(n, 1);
    ASSERT_EQ(LapackeTrsna<T>(LAPACK_COL_MAJOR, job, 'A', sel.data(), n,
                              t.data(), n, vlc.data(), ldv, vrc.data(), ldv,
                              sc.data(), sepc.data(), mm, &mc),
              0);
    ASSERT_EQ(LapackeTrsna<T>(LAPACK_ROW_MAJOR, job, 'A', sel.data(), n,
                              tr.data(), n, vlr.data(), ldv, vrr.data(), ldv,
                              sr.data(), sepr.data(), mm, &mr),
              0);
    EXPECT_EQ(mr, mc);
    EXPECT_EQ(sr, sc);
    EXPECT_EQ(sepr, sepc);
  }
}

TYPED_TEST(GeevTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  std::vector<T> a(16, T(1)), vl(16), vr(16);
  std::vector<Eig<T>> w;
  GeevxOut<R> out;
  lapack_int sdim = 0, m = 0;
  R e = 0, v = 0;
  std::vector<R> s(4), sep(4);
  std::vector<lapack_logical> sel(4, 1);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeGeev<T>(0, 'V', 'V', 4, a.data(), 4, w, vl.data(), 4,
                           vr.data(), 4),
            -1);
  // Row-major leading dimensions: LDVL / LDVR >= N only if wanted.
  EXPECT_EQ(LapackeGeev<T>(LAPACK_ROW_MAJOR, 'V', 'V', 4, a.data(), 3, w,
                           vl.data(), 4, vr.data(), 4),
            -6);
  const lapack_int gvl = kComplex ? 9 : 10, gvr = kComplex ? 11 : 12;
  EXPECT_EQ(LapackeGeev<T>(LAPACK_ROW_MAJOR, 'V', 'N', 4, a.data(), 4, w,
                           vl.data(), 3, vr.data(), 1),
            -gvl);
  EXPECT_EQ(LapackeGeev<T>(LAPACK_ROW_MAJOR, 'N', 'V', 4, a.data(), 4, w,
                           vl.data(), 1, vr.data(), 3),
            -gvr);
  EXPECT_EQ(LapackeGeev<T>(LAPACK_ROW_MAJOR, 'N', 'N', 4, a.data(), 4, w,
                           vl.data(), 0, vr.data(), 1),
            -gvl);
  const lapack_int xvl = kComplex ? 11 : 12;
  EXPECT_EQ(LapackeGeevx<T>(LAPACK_ROW_MAJOR, 'N', 'V', 'N', 'N', 4, a.data(),
                            4, w, vl.data(), 3, vr.data(), 1, out),
            -xvl);
  const lapack_int svs = kComplex ? 12 : 13;
  EXPECT_EQ(LapackeGeesx<T>(LAPACK_ROW_MAJOR, 'N', 'N', 'N', 4, a.data(), 4,
                            &sdim, w, vl.data(), 3, &e, &v),
            -svs);
  EXPECT_EQ(
      LapackeTrsna<T>(LAPACK_ROW_MAJOR, 'V', 'A', sel.data(), 4, a.data(), 4,
                      vl.data(), 3, vr.data(), 4, s.data(), sep.data(), 4, &m),
      -9);
  const std::string out_text = ::testing::internal::GetCapturedStdout();
  const std::string geev = LapackeName<T>("geev");
  EXPECT_EQ(
      out_text,
      "Wrong parameter 1 in " + geev + "\nWrong parameter 6 in " + geev +
          "_work\nWrong parameter " + std::to_string(gvl) + " in " + geev +
          "_work\nWrong parameter " + std::to_string(gvr) + " in " + geev +
          "_work\nWrong parameter " + std::to_string(gvl) + " in " + geev +
          "_work\nWrong parameter " + std::to_string(xvl) + " in " +
          LapackeName<T>("geevx_work") + "\nWrong parameter " +
          std::to_string(svs) + " in " + LapackeName<T>("geesx_work") +
          "\nWrong parameter 9 in " + LapackeName<T>("trsna_work") + "\n");
  // NaN checks.
  a[5] = NaN<T>();
  EXPECT_EQ(LapackeGeev<T>(LAPACK_COL_MAJOR, 'N', 'N', 4, a.data(), 4, w,
                           vl.data(), 4, vr.data(), 4),
            -5);
  EXPECT_EQ(LapackeGeevx<T>(LAPACK_COL_MAJOR, 'N', 'N', 'N', 'N', 4, a.data(),
                            4, w, vl.data(), 4, vr.data(), 4, out),
            -7);
  EXPECT_EQ(LapackeGeesx<T>(LAPACK_COL_MAJOR, 'N', 'N', 'N', 4, a.data(), 4,
                            &sdim, w, vl.data(), 4, &e, &v),
            -7);
  EXPECT_EQ(
      LapackeTrsna<T>(LAPACK_COL_MAJOR, 'V', 'A', sel.data(), 4, a.data(), 4,
                      vl.data(), 4, vr.data(), 4, s.data(), sep.data(), 4, &m),
      -6);
  std::vector<T> t(16, T(0));
  for (int i = 0; i < 4; ++i) t[i * 5] = T(R(i + 1));
  vl.assign(16, T(1));
  vr.assign(16, T(1));
  vr[3] = NaN<T>();
  // VL and VR are checked for JOB = 'E' or 'B' only.
  EXPECT_EQ(
      LapackeTrsna<T>(LAPACK_COL_MAJOR, 'E', 'A', sel.data(), 4, t.data(), 4,
                      vl.data(), 4, vr.data(), 4, s.data(), sep.data(), 4, &m),
      -10);
  EXPECT_EQ(
      LapackeTrsna<T>(LAPACK_COL_MAJOR, 'V', 'A', sel.data(), 4, t.data(), 4,
                      vl.data(), 4, vr.data(), 4, s.data(), sep.data(), 4, &m),
      0);
  vl[0] = NaN<T>();
  EXPECT_EQ(
      LapackeTrsna<T>(LAPACK_COL_MAJOR, 'B', 'A', sel.data(), 4, t.data(), 4,
                      vl.data(), 4, vr.data(), 4, s.data(), sep.data(), 4, &m),
      -8);
  // ?geesx for n = 1 and SENSE = 'V' (the reference fails: LIWORK = 0).
  std::vector<T> one = {T(2)}, vs1(1);
  EXPECT_EQ(LapackeGeesx<T>(LAPACK_COL_MAJOR, 'V', 'S', 'V', 1, one.data(), 1,
                            &sdim, w, vs1.data(), 1, &e, &v),
            0);
  EXPECT_EQ(sdim, 1);
}

}  // namespace
}  // namespace hwyblas_test
