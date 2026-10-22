// Tests of the generalized nonsymmetric eigenproblem (xGGBAL, xGGBAK,
// xGGHRD, xHGEQZ, xTGEVC, xTGEXC, xTGSEN, xTGSYL and their helpers, the
// drivers xGGES and xGGEV), the generalized singular value decomposition
// (xGGSVD3, xGGSVP3, xTGSJA, xLAGS2, xLAPLL, xLAPMT), xLANHS, xORGR2 /
// xUNGR2, and LAPACKE.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/eigen_test_util.h"
#include "lapack/lapack_test_util.h"
#include "lapack/svd_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class GgevTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GgevTest, ScalarTypes);

template <typename R>
class GgevRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GgevRealTest, RealTypes);

using C64 = std::complex<double>;

size_t Sz(hwyblas_int n) {
  return static_cast<size_t>(std::max<hwyblas_int>(1, n));
}

size_t Sz2(hwyblas_int m, hwyblas_int n) { return Sz(m) * Sz(n); }

// Whether x and y hold the same values, NaNs (the padding of the leading
// dimensions) included.
template <typename T>
bool SameValues(const std::vector<T>& x, const std::vector<T>& y) {
  using R = RealT<T>;
  const auto same = [](R u, R v) {
    return u == v || (std::isnan(u) && std::isnan(v));
  };
  if (x.size() != y.size()) return false;
  for (size_t i = 0; i < x.size(); ++i) {
    if (!same(std::real(x[i]), std::real(y[i])) ||
        !same(std::imag(x[i]), std::imag(y[i]))) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Test pairs

enum class Kind {
  kRandom,
  kSingularB,   // B with a zero column and row: infinite eigenvalues
  kTriangular,  // A and B upper triangular
  kPermutable,  // block triangular after permutations (xGGBAL 'P')
  kGraded,      // random, (i,j) scaled by 4**(j - i) (xGGBAL 'S')
  kLarge,       // random times a huge number (the drivers scale A and B)
  kSmall,       // random times a tiny number
  kPairs,       // A block diagonal with rotations, B = I (complex pairs)
};
const Kind kKinds[] = {Kind::kRandom,     Kind::kSingularB, Kind::kTriangular,
                       Kind::kPermutable, Kind::kGraded,    Kind::kLarge,
                       Kind::kSmall,      Kind::kPairs};

const char* KindName(Kind kind) {
  switch (kind) {
    case Kind::kRandom:
      return "random";
    case Kind::kSingularB:
      return "singular B";
    case Kind::kTriangular:
      return "triangular";
    case Kind::kPermutable:
      return "permutable";
    case Kind::kGraded:
      return "graded";
    case Kind::kLarge:
      return "large";
    case Kind::kSmall:
      return "small";
    case Kind::kPairs:
      return "pairs";
  }
  return "";
}

// A pair (A, B) of order n (leading dimension ld).
template <typename T>
void TestPair(Kind kind, hwyblas_int n, hwyblas_int ld, Random<T>& rng,
              std::vector<T>& a, std::vector<T>& b) {
  using R = RealT<T>;
  a = rng.Vector(Sz2(ld, n));
  b = rng.Vector(Sz2(ld, n));
  const T zero = T(0);
  // Huge and tiny factors near sqrt of the overflow / underflow
  // thresholds, beyond the drivers' scaling thresholds.
  const R huge = std::sqrt(std::numeric_limits<R>::max()) * R(0.25);
  const R tiny = std::sqrt(std::numeric_limits<R>::min()) * R(4);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      T& x = At(a, i, j, ld);
      T& y = At(b, i, j, ld);
      switch (kind) {
        case Kind::kRandom:
          break;
        case Kind::kSingularB:
          if (i == n / 2 || j == n / 2) y = zero;
          break;
        case Kind::kTriangular:
          if (i > j) x = y = zero;
          break;
        case Kind::kPermutable:
          // Rows 0 and n-1 of (A, B) have one nonzero, in columns n-1 and
          // 0 (isolated eigenvalues at both ends after permutation).
          if (i == 0 && j != n - 1) x = y = zero;
          if (j == 0 && i != n - 1) x = y = zero;
          break;
        case Kind::kGraded: {
          const R f = static_cast<R>(std::ldexp(1.0, 2 * (j - i)));
          x *= f;
          y *= f;
          break;
        }
        case Kind::kLarge:
          x *= huge;
          y *= huge;
          break;
        case Kind::kSmall:
          x *= tiny;
          y *= tiny;
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
          y = T(R(i == j ? 1 : 0));
          break;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Calls

template <typename T>
hwyblas_int Ggbal(char job, hwyblas_int n, std::vector<T>& a, hwyblas_int lda,
                  std::vector<T>& b, hwyblas_int ldb, hwyblas_int* ilo,
                  hwyblas_int* ihi, std::vector<RealT<T>>& lscale,
                  std::vector<RealT<T>>& rscale) {
  using R = RealT<T>;
  std::vector<R> work(Sz(6 * n), NAN);
  lscale.assign(Sz(n), NAN);
  rscale.assign(Sz(n), NAN);
  hwyblas_int info = -99;
  CALL_RC(T, ggbal, ggbal, &job, &n, P(a.data()), &lda, P(b.data()), &ldb, ilo,
          ihi, lscale.data(), rscale.data(), work.data(), &info);
  return info;
}

template <typename T>
hwyblas_int Ggbak(char job, char side, hwyblas_int n, hwyblas_int ilo,
                  hwyblas_int ihi, const std::vector<RealT<T>>& lscale,
                  const std::vector<RealT<T>>& rscale, hwyblas_int m,
                  std::vector<T>& v, hwyblas_int ldv) {
  hwyblas_int info = -99;
  CALL_RC(T, ggbak, ggbak, &job, &side, &n, &ilo, &ihi, lscale.data(),
          rscale.data(), &m, P(v.data()), &ldv, &info);
  return info;
}

template <typename T>
hwyblas_int Gghrd(char compq, char compz, hwyblas_int n, hwyblas_int ilo,
                  hwyblas_int ihi, std::vector<T>& a, hwyblas_int lda,
                  std::vector<T>& b, hwyblas_int ldb, std::vector<T>& q,
                  hwyblas_int ldq, std::vector<T>& z, hwyblas_int ldz) {
  hwyblas_int info = -99;
  CALL_RC(T, gghrd, gghrd, &compq, &compz, &n, &ilo, &ihi, P(a.data()), &lda,
          P(b.data()), &ldb, P(q.data()), &ldq, P(z.data()), &ldz, &info);
  return info;
}

// xHGEQZ with the eigenvalues as alpha (complex) and beta; *opt returns
// WORK(1).
template <typename T>
hwyblas_int Hgeqz(char job, char compq, char compz, hwyblas_int n,
                  hwyblas_int ilo, hwyblas_int ihi, std::vector<T>& h,
                  hwyblas_int ldh, std::vector<T>& t, hwyblas_int ldt,
                  std::vector<C64>& alpha, std::vector<C64>& beta,
                  std::vector<T>& q, hwyblas_int ldq, std::vector<T>& z,
                  hwyblas_int ldz, hwyblas_int lwork,
                  hwyblas_int* opt = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  hwyblas_int info = -99;
  alpha.assign(Sz(n), C64(NAN, NAN));
  beta.assign(Sz(n), C64(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n), NaN<T>()), be(Sz(n), NaN<T>());
    std::vector<R> rwork(Sz(n), NAN);
    HWYBLAS_CALL_COMPLEX(T, hgeqz, &job, &compq, &compz, &n, &ilo, &ihi,
                         P(h.data()), &ldh, P(t.data()), &ldt, P(al.data()),
                         P(be.data()), P(q.data()), &ldq, P(z.data()), &ldz,
                         P(work.data()), &lwork, rwork.data(), &info);
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n), NAN), ai(Sz(n), NAN), be(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, hgeqz, &job, &compq, &compz, &n, &ilo, &ihi, h.data(),
                      &ldh, t.data(), &ldt, ar.data(), ai.data(), be.data(),
                      q.data(), &ldq, z.data(), &ldz, work.data(), &lwork,
                      &info);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  alpha.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  beta.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

template <typename T>
hwyblas_int Tgevc(char side, char howmny, const std::vector<hwyblas_int>& sel,
                  hwyblas_int n, const std::vector<T>& s, hwyblas_int lds,
                  const std::vector<T>& p, hwyblas_int ldp, std::vector<T>& vl,
                  hwyblas_int ldvl, std::vector<T>& vr, hwyblas_int ldvr,
                  hwyblas_int mm, hwyblas_int* m) {
  using R = RealT<T>;
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(Sz(2 * n), NaN<T>());
    std::vector<R> rwork(Sz(2 * n), NAN);
    HWYBLAS_CALL_COMPLEX(T, tgevc, &side, &howmny, sel.data(), &n, P(s.data()),
                         &lds, P(p.data()), &ldp, P(vl.data()), &ldvl,
                         P(vr.data()), &ldvr, &mm, m, P(work.data()),
                         rwork.data(), &info);
  } else {
    std::vector<T> work(Sz(6 * n), NaN<T>());
    HWYBLAS_CALL_REAL(T, tgevc, &side, &howmny, sel.data(), &n, s.data(), &lds,
                      p.data(), &ldp, vl.data(), &ldvl, vr.data(), &ldvr, &mm,
                      m, work.data(), &info);
  }
  return info;
}

// Selection functions of xGGES: the eigenvalues with a negative real part.
template <typename R>
hwyblas_int NegativeReal3(const R* ar, const R* /*ai*/, const R* b) {
  return *ar * *b < 0 ? 1 : 0;
}
template <typename R>
hwyblas_int NegativeReal2(const R* a, const R* b) {
  // Re(alpha * conj(beta)) < 0.
  return a[0] * b[0] + a[1] * b[1] < 0 ? 1 : 0;
}

// xGGES (SORT = 'S' selects the eigenvalues with a negative real part) or
// xGGEV (vsl / vsr are VL / VR); for lwork = -1, *opt returns WORK(1).
template <typename T>
hwyblas_int Gges(char jobvsl, char jobvsr, char sort, hwyblas_int n,
                 std::vector<T>& a, hwyblas_int lda, std::vector<T>& b,
                 hwyblas_int ldb, hwyblas_int* sdim, std::vector<C64>& alpha,
                 std::vector<C64>& beta, std::vector<T>& vsl, hwyblas_int ldvsl,
                 std::vector<T>& vsr, hwyblas_int ldvsr, hwyblas_int lwork,
                 hwyblas_int* opt = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  std::vector<hwyblas_int> bwork(Sz(n), -7);
  hwyblas_int info = -99;
  alpha.assign(Sz(n), C64(NAN, NAN));
  beta.assign(Sz(n), C64(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n), NaN<T>()), be(Sz(n), NaN<T>());
    std::vector<R> rwork(Sz(8 * n), NAN);
    HWYBLAS_CALL_COMPLEX(T, gges, &jobvsl, &jobvsr, &sort, &NegativeReal2<R>,
                         &n, P(a.data()), &lda, P(b.data()), &ldb, sdim,
                         P(al.data()), P(be.data()), P(vsl.data()), &ldvsl,
                         P(vsr.data()), &ldvsr, P(work.data()), &lwork,
                         rwork.data(), bwork.data(), &info);
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n), NAN), ai(Sz(n), NAN), be(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, gges, &jobvsl, &jobvsr, &sort, &NegativeReal3<R>, &n,
                      a.data(), &lda, b.data(), &ldb, sdim, ar.data(),
                      ai.data(), be.data(), vsl.data(), &ldvsl, vsr.data(),
                      &ldvsr, work.data(), &lwork, bwork.data(), &info);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  alpha.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  beta.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

template <typename T>
hwyblas_int Ggev(char jobvl, char jobvr, hwyblas_int n, std::vector<T>& a,
                 hwyblas_int lda, std::vector<T>& b, hwyblas_int ldb,
                 std::vector<C64>& alpha, std::vector<C64>& beta,
                 std::vector<T>& vl, hwyblas_int ldvl, std::vector<T>& vr,
                 hwyblas_int ldvr, hwyblas_int lwork,
                 hwyblas_int* opt = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  hwyblas_int info = -99;
  alpha.assign(Sz(n), C64(NAN, NAN));
  beta.assign(Sz(n), C64(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n), NaN<T>()), be(Sz(n), NaN<T>());
    std::vector<R> rwork(Sz(8 * n), NAN);
    HWYBLAS_CALL_COMPLEX(T, ggev, &jobvl, &jobvr, &n, P(a.data()), &lda,
                         P(b.data()), &ldb, P(al.data()), P(be.data()),
                         P(vl.data()), &ldvl, P(vr.data()), &ldvr,
                         P(work.data()), &lwork, rwork.data(), &info);
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n), NAN), ai(Sz(n), NAN), be(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, ggev, &jobvl, &jobvr, &n, a.data(), &lda, b.data(),
                      &ldb, ar.data(), ai.data(), be.data(), vl.data(), &ldvl,
                      vr.data(), &ldvr, work.data(), &lwork, &info);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  alpha.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  beta.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

// The outputs of xTGSEN besides the reordered pair.
template <typename R>
struct TgsenOut {
  hwyblas_int m = -1;
  R pl = -1, pr = -1;
  R dif[2] = {-1, -1};
  hwyblas_int work0 = -1, iwork0 = -1;
};

template <typename T>
hwyblas_int Tgsen(hwyblas_int ijob, bool wantq, bool wantz,
                  const std::vector<hwyblas_int>& sel, hwyblas_int n,
                  std::vector<T>& a, hwyblas_int lda, std::vector<T>& b,
                  hwyblas_int ldb, std::vector<C64>& alpha,
                  std::vector<C64>& beta, std::vector<T>& q, hwyblas_int ldq,
                  std::vector<T>& z, hwyblas_int ldz, TgsenOut<RealT<T>>& out,
                  hwyblas_int lwork, hwyblas_int liwork) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  std::vector<hwyblas_int> iwork(Sz(liwork), -7);
  const hwyblas_int wq = wantq ? 1 : 0, wz = wantz ? 1 : 0;
  hwyblas_int info = -99;
  alpha.assign(Sz(n), C64(NAN, NAN));
  beta.assign(Sz(n), C64(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n), NaN<T>()), be(Sz(n), NaN<T>());
    HWYBLAS_CALL_COMPLEX(T, tgsen, &ijob, &wq, &wz, sel.data(), &n, P(a.data()),
                         &lda, P(b.data()), &ldb, P(al.data()), P(be.data()),
                         P(q.data()), &ldq, P(z.data()), &ldz, &out.m, &out.pl,
                         &out.pr, out.dif, P(work.data()), &lwork, iwork.data(),
                         &liwork, &info);
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n), NAN), ai(Sz(n), NAN), be(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, tgsen, &ijob, &wq, &wz, sel.data(), &n, a.data(), &lda,
                      b.data(), &ldb, ar.data(), ai.data(), be.data(), q.data(),
                      &ldq, z.data(), &ldz, &out.m, &out.pl, &out.pr, out.dif,
                      work.data(), &lwork, iwork.data(), &liwork, &info);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  alpha.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  beta.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  out.work0 = static_cast<hwyblas_int>(std::real(work[0]));
  out.iwork0 = iwork[0];
  return info;
}

// xTGEXC; for complex T, *ifst is passed by value.
template <typename T>
hwyblas_int Tgexc(bool wantq, bool wantz, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda, std::vector<T>& b, hwyblas_int ldb,
                  std::vector<T>& q, hwyblas_int ldq, std::vector<T>& z,
                  hwyblas_int ldz, hwyblas_int* ifst, hwyblas_int* ilst) {
  const hwyblas_int wq = wantq ? 1 : 0, wz = wantz ? 1 : 0;
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, tgexc, &wq, &wz, &n, P(a.data()), &lda, P(b.data()),
                         &ldb, P(q.data()), &ldq, P(z.data()), &ldz, ifst, ilst,
                         &info);
  } else {
    const hwyblas_int lwork = std::max<hwyblas_int>(1, 4 * n + 16);
    std::vector<T> work(Sz(lwork), NaN<T>());
    HWYBLAS_CALL_REAL(T, tgexc, &wq, &wz, &n, a.data(), &lda, b.data(), &ldb,
                      q.data(), &ldq, z.data(), &ldz, ifst, ilst, work.data(),
                      &lwork, &info);
  }
  return info;
}

template <typename T>
hwyblas_int Tgsyl(char trans, hwyblas_int ijob, hwyblas_int m, hwyblas_int n,
                  const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& b, hwyblas_int ldb, std::vector<T>& c,
                  hwyblas_int ldc, const std::vector<T>& d, hwyblas_int ldd,
                  const std::vector<T>& e, hwyblas_int lde, std::vector<T>& f,
                  hwyblas_int ldf, RealT<T>* scale, RealT<T>* dif,
                  hwyblas_int lwork, hwyblas_int* opt = nullptr) {
  std::vector<T> work(Sz(lwork), NaN<T>());
  std::vector<hwyblas_int> iwork(Sz(m + n + 6), -7);
  hwyblas_int info = -99;
  CALL_RC(T, tgsyl, tgsyl, &trans, &ijob, &m, &n, P(a.data()), &lda,
          P(b.data()), &ldb, P(c.data()), &ldc, P(d.data()), &ldd, P(e.data()),
          &lde, P(f.data()), &ldf, scale, dif, P(work.data()), &lwork,
          iwork.data(), &info);
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

// The outputs of xGGSVD3 besides U, V, Q and the overwritten A and B.
template <typename R>
struct Ggsvd3Out {
  hwyblas_int k = -1, l = -1;
  std::vector<R> alpha, beta;
  std::vector<hwyblas_int> iwork;
};

template <typename T>
hwyblas_int Ggsvd3(char jobu, char jobv, char jobq, hwyblas_int m,
                   hwyblas_int n, hwyblas_int p, std::vector<T>& a,
                   hwyblas_int lda, std::vector<T>& b, hwyblas_int ldb,
                   std::vector<T>& u, hwyblas_int ldu, std::vector<T>& v,
                   hwyblas_int ldv, std::vector<T>& q, hwyblas_int ldq,
                   Ggsvd3Out<RealT<T>>& out, hwyblas_int lwork,
                   hwyblas_int* opt = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  out.alpha.assign(Sz(n), NAN);
  out.beta.assign(Sz(n), NAN);
  out.iwork.assign(Sz(n), -7);
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    std::vector<R> rwork(Sz(2 * n), NAN);
    HWYBLAS_CALL_COMPLEX(T, ggsvd3, &jobu, &jobv, &jobq, &m, &n, &p, &out.k,
                         &out.l, P(a.data()), &lda, P(b.data()), &ldb,
                         out.alpha.data(), out.beta.data(), P(u.data()), &ldu,
                         P(v.data()), &ldv, P(q.data()), &ldq, P(work.data()),
                         &lwork, rwork.data(), out.iwork.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, ggsvd3, &jobu, &jobv, &jobq, &m, &n, &p, &out.k,
                      &out.l, a.data(), &lda, b.data(), &ldb, out.alpha.data(),
                      out.beta.data(), u.data(), &ldu, v.data(), &ldv, q.data(),
                      &ldq, work.data(), &lwork, out.iwork.data(), &info);
  }
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

template <typename T>
hwyblas_int Ggsvp3(char jobu, char jobv, char jobq, hwyblas_int m,
                   hwyblas_int p, hwyblas_int n, std::vector<T>& a,
                   hwyblas_int lda, std::vector<T>& b, hwyblas_int ldb,
                   RealT<T> tola, RealT<T> tolb, hwyblas_int* k, hwyblas_int* l,
                   std::vector<T>& u, hwyblas_int ldu, std::vector<T>& v,
                   hwyblas_int ldv, std::vector<T>& q, hwyblas_int ldq,
                   hwyblas_int lwork, hwyblas_int* opt = nullptr) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>()), tau(Sz(n), NaN<T>());
  std::vector<hwyblas_int> iwork(Sz(n), -7);
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    std::vector<R> rwork(Sz(2 * n), NAN);
    HWYBLAS_CALL_COMPLEX(T, ggsvp3, &jobu, &jobv, &jobq, &m, &p, &n,
                         P(a.data()), &lda, P(b.data()), &ldb, &tola, &tolb, k,
                         l, P(u.data()), &ldu, P(v.data()), &ldv, P(q.data()),
                         &ldq, iwork.data(), rwork.data(), P(tau.data()),
                         P(work.data()), &lwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, ggsvp3, &jobu, &jobv, &jobq, &m, &p, &n, a.data(),
                      &lda, b.data(), &ldb, &tola, &tolb, k, l, u.data(), &ldu,
                      v.data(), &ldv, q.data(), &ldq, iwork.data(), tau.data(),
                      work.data(), &lwork, &info);
  }
  if (opt != nullptr) *opt = static_cast<hwyblas_int>(std::real(work[0]));
  return info;
}

// ---------------------------------------------------------------------------
// Checks

// The n x n identity (leading dimension ld).
template <typename T>
std::vector<T> IdentityMatrix(hwyblas_int n, hwyblas_int ld) {
  std::vector<T> x(Sz2(ld, n), T(0));
  for (hwyblas_int i = 0; i < n; ++i) At(x, i, i, ld) = T(1);
  return x;
}

// ||A - Q * S * Z**H||_1 / (n ||A||_1 eps) (or the residual ratio for A
// = 0), as LAPACK's xGET51.
template <typename T>
double DecompositionRatio(const std::vector<T>& a, const std::vector<T>& q,
                          const std::vector<T>& s, const std::vector<T>& z,
                          hwyblas_int n, hwyblas_int ld) {
  const Dense<T> qd = ToDense(q, n, n, ld), zd = ToDense(z, n, n, ld);
  return ResidualRatio(ToDense(a, n, n, ld),
                       Mul(Mul(qd, ToDense(s, n, n, ld)), Adjoint(zd)),
                       std::max<double>(1, n));
}

// Whether the n x n matrix s is upper triangular (real T and
// quasi = true: quasi-upper triangular, without consecutive nonzero
// subdiagonal elements).
template <typename T>
bool IsUpperTriangular(const std::vector<T>& s, hwyblas_int n, hwyblas_int ld,
                       bool quasi) {
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j + 1; i < n; ++i) {
      if (At(s, i, j, ld) == T(0)) continue;
      if (!quasi || i != j + 1) return false;
      if (j > 0 && At(s, j, j - 1, ld) != T(0)) return false;
    }
  }
  return true;
}

// The eigenvector of the eigenvalue j in the columns of V (as returned by
// xTGEVC with HOWMNY = 'A' or 'B', or by xGGEV): for real T, a complex pair
// (alpha(j) with a positive imaginary part first) takes the columns j and
// j+1 (the real and imaginary part of the vector of alpha(j)).
template <typename T>
std::vector<C64> Eigvec(const std::vector<T>& v, hwyblas_int n, hwyblas_int ld,
                        const std::vector<C64>& alpha, hwyblas_int j) {
  std::vector<C64> x(static_cast<size_t>(n));
  for (hwyblas_int k = 0; k < n; ++k) {
    if (IsComplex<T>::value || alpha[j].imag() == 0) {
      x[k] = static_cast<C64>(At(v, k, j, ld));
    } else if (alpha[j].imag() > 0) {
      x[k] = C64(std::real(At(v, k, j, ld)), std::real(At(v, k, j + 1, ld)));
    } else {
      x[k] = C64(std::real(At(v, k, j - 1, ld)), -std::real(At(v, k, j, ld)));
    }
  }
  return x;
}

// The test ratio of LAPACK's xGET52 for the eigenpair (alpha / beta, x) of
// the pair (A, B): ||(beta A - alpha B) x||_1 (right) or ||x**H (beta A -
// alpha B)||_1 (left), with (alpha, beta) normalized to a maximum of 1, /
// ((|beta| ||A||_1 + |alpha| ||B||_1) ||x||_1 n eps).
template <typename T>
double EigenpairRatio(const std::vector<T>& a, const std::vector<T>& b,
                      hwyblas_int n, hwyblas_int ld, C64 alpha, C64 beta,
                      const std::vector<C64>& x, bool left) {
  const double s = std::max(std::abs(alpha), std::abs(beta));
  if (s == 0) return 0;
  const C64 al = alpha / s, be = beta / s;
  double resid = 0, xnorm = 0;
  for (hwyblas_int i = 0; i < n; ++i) {
    C64 r = 0;
    for (hwyblas_int k = 0; k < n; ++k) {
      if (left) {
        r += std::conj(be * static_cast<C64>(At(a, k, i, ld)) -
                       al * static_cast<C64>(At(b, k, i, ld))) *
             x[k];
      } else {
        r += (be * static_cast<C64>(At(a, i, k, ld)) -
              al * static_cast<C64>(At(b, i, k, ld))) *
             x[k];
      }
    }
    resid += std::abs(r);
    xnorm += std::abs(x[i]);
  }
  const double scale =
      std::abs(be) * Norm1(a, n, n, ld) + std::abs(al) * Norm1(b, n, n, ld);
  if (xnorm == 0) return std::numeric_limits<double>::infinity();
  return resid / std::max(scale, 1e-300) / xnorm / std::max<double>(1, n) /
         LapackEps<T>();
}

// Checks the eigenvectors in V (left or right) of all eigenvalues of (A,
// B): their test ratios and their normalization (largest |re| + |im| = 1).
template <typename T>
void ExpectEigenvectors(const std::vector<T>& a, const std::vector<T>& b,
                        hwyblas_int n, hwyblas_int ld,
                        const std::vector<C64>& alpha,
                        const std::vector<C64>& beta, const std::vector<T>& v,
                        hwyblas_int ldv, bool left) {
  for (hwyblas_int j = 0; j < n; ++j) {
    const std::vector<C64> x = Eigvec(v, n, ldv, alpha, j);
    EXPECT_LT(EigenpairRatio(a, b, n, ld, alpha[j], beta[j], x, left),
              kLapackThreshold)
        << (left ? "left " : "right ") << j;
    double big = 0;
    for (const C64& c : x)
      big = std::max(big, std::abs(c.real()) + std::abs(c.imag()));
    EXPECT_NEAR(big, 1.0, 100 * LapackEps<T>()) << j;
  }
}

// Checks the generalized Schur form (S, P) of (A, B) = (Q S Z**H, Q P
// Z**H) and its eigenvalues: S (quasi-)upper triangular, P upper
// triangular with a nonnegative real diagonal (real T: the 2-by-2 diagonal
// blocks of P diagonal), alpha / beta the eigenvalues of the 1-by-1 blocks
// (exactly the diagonal elements if `exact`, as xHGEQZ returns them) and
// complex conjugate pairs for the 2-by-2 blocks.
template <typename T>
void ExpectSchurForm(const std::vector<T>& a, const std::vector<T>& b,
                     const std::vector<T>& s, const std::vector<T>& p,
                     const std::vector<T>& q, const std::vector<T>& z,
                     hwyblas_int n, hwyblas_int ld,
                     const std::vector<C64>& alpha,
                     const std::vector<C64>& beta, bool exact = true) {
  constexpr bool kComplex = IsComplex<T>::value;
  EXPECT_TRUE(IsUpperTriangular(s, n, ld, !kComplex));
  EXPECT_TRUE(IsUpperTriangular(p, n, ld, false));
  EXPECT_LT(DecompositionRatio(a, q, s, z, n, ld), kLapackThreshold);
  EXPECT_LT(DecompositionRatio(b, q, p, z, n, ld), kLapackThreshold);
  EXPECT_LT(OrthogonalityRatio(ToDense(q, n, n, ld)), kLapackThreshold);
  EXPECT_LT(OrthogonalityRatio(ToDense(z, n, n, ld)), kLapackThreshold);
  for (hwyblas_int j = 0; j < n; ++j) {
    const T d = At(p, j, j, ld);
    EXPECT_EQ(std::imag(d), 0) << j;
    const bool pair = !kComplex && j + 1 < n && At(s, j + 1, j, ld) != T(0);
    if (pair) {
      // A diagonal 2-by-2 block of P. (Its elements can be negative after
      // reordering by xTGSEN, also in the reference LAPACK, although xGGES
      // documents positive ones.)
      EXPECT_EQ(At(p, j, j + 1, ld), T(0)) << j;
      EXPECT_GT(alpha[j].imag(), 0) << j;
      // The two eigenvalues are conjugate (alpha and beta are scaled
      // differently).
      const C64 l1 = alpha[j] / beta[j], l2 = alpha[j + 1] / beta[j + 1];
      EXPECT_LE(std::abs(l2 - std::conj(l1)),
                100 * LapackEps<T>() * std::abs(l1))
          << j;
      ++j;
    } else {
      EXPECT_GE(std::real(d), 0) << j;
    }
    if (!pair && exact) {
      EXPECT_EQ(alpha[j], static_cast<C64>(At(s, j, j, ld))) << j;
      EXPECT_EQ(beta[j], static_cast<C64>(d)) << j;
    }
  }
}

// The eigenvalues alpha / beta as (sorted) complex numbers, infinite ones
// as (inf, 0).
std::vector<C64> Lambdas(const std::vector<C64>& alpha,
                         const std::vector<C64>& beta) {
  std::vector<C64> l;
  l.reserve(alpha.size());
  for (size_t j = 0; j < alpha.size(); ++j) {
    l.push_back(beta[j] == 0.0 ? C64(INFINITY, 0) : alpha[j] / beta[j]);
  }
  std::sort(l.begin(), l.end(), [](C64 x, C64 y) {
    return x.real() != y.real() ? x.real() < y.real() : x.imag() < y.imag();
  });
  return l;
}

// Expects the eigenvalues l and m to agree as sets, to a relative
// tolerance (each of l matched with the nearest unmatched one of m).
template <typename T>
void ExpectSameLambdas(const std::vector<C64>& l, const std::vector<C64>& m) {
  ASSERT_EQ(l.size(), m.size());
  std::vector<bool> used(m.size(), false);
  for (size_t j = 0; j < l.size(); ++j) {
    size_t best = m.size();
    double dist = INFINITY;
    for (size_t k = 0; k < m.size(); ++k) {
      if (used[k]) continue;
      const double d =
          std::isinf(l[j].real()) || std::isinf(m[k].real())
              ? (std::isinf(l[j].real()) && std::isinf(m[k].real()) ? 0.0
                                                                    : INFINITY)
              : std::abs(l[j] - m[k]);
      if (best == m.size() || d < dist) {
        best = k;
        dist = d;
      }
    }
    used[best] = true;
    if (std::isinf(l[j].real())) {
      EXPECT_EQ(dist, 0) << j;
      continue;
    }
    EXPECT_LE(dist, 1e3 * LapackEps<T>() * std::max(1.0, std::abs(l[j])))
        << j << ": " << l[j] << " " << m[best];
  }
}

// A pair in generalized Schur form: (A, B) = (Q S Z**H, Q P Z**H) from
// xGGES without sorting (Q and Z in vsl / vsr).
template <typename T>
void SchurPair(Kind kind, hwyblas_int n, hwyblas_int ld, Random<T>& rng,
               std::vector<T>& a, std::vector<T>& b, std::vector<T>& s,
               std::vector<T>& p, std::vector<T>& q, std::vector<T>& z,
               std::vector<C64>& alpha, std::vector<C64>& beta) {
  TestPair(kind, n, ld, rng, a, b);
  s = a;
  p = b;
  q.assign(Sz2(ld, n), NaN<T>());
  z.assign(Sz2(ld, n), NaN<T>());
  hwyblas_int sdim = -1;
  ASSERT_EQ(Gges<T>('V', 'V', 'N', n, s, ld, p, ld, &sdim, alpha, beta, q, ld,
                    z, ld, 64 * (n + 2)),
            0);
}

// ---------------------------------------------------------------------------
// xGGBAL, xGGBAK

TYPED_TEST(GgevTest, BalanceAndBackTransform) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(11);
  for (const Kind kind : {Kind::kRandom, Kind::kPermutable, Kind::kGraded,
                          Kind::kTriangular, Kind::kSingularB}) {
    for (const hwyblas_int n : {0, 1, 2, 5, 9}) {
      for (const char job : {'N', 'P', 'S', 'B'}) {
        SCOPED_TRACE(::testing::Message()
                     << KindName(kind) << " n=" << n << " job=" << job);
        const hwyblas_int ld = n + 2;
        std::vector<T> a, b;
        TestPair(kind, n, ld, rng, a, b);
        std::vector<T> ab = a, bb = b;
        hwyblas_int ilo = -1, ihi = -1;
        std::vector<R> ls, rs;
        ASSERT_EQ(Ggbal<T>(job, n, ab, ld, bb, ld, &ilo, &ihi, ls, rs), 0);
        if (job == 'N' || job == 'S') EXPECT_EQ(ilo, 1);
        if (n == 0) {
          EXPECT_EQ(ihi, 0);
          continue;
        }
        if (job == 'N' || job == 'S') EXPECT_EQ(ihi, n);
        if (job == 'N') {
          EXPECT_TRUE(SameValues(ab, a));
          EXPECT_TRUE(SameValues(bb, b));
        }
        if (kind == Kind::kPermutable && n > 2 && (job == 'P' || job == 'B')) {
          EXPECT_LT(ihi - ilo + 1, n);
        }
        // Mr = Pr * Dr and Ml = Pl**T * Dl from xGGBAK applied to I; the
        // balanced pair is Ml**T * (A, B) * Mr.
        std::vector<T> mr = IdentityMatrix<T>(n, ld), ml = mr;
        ASSERT_EQ(Ggbak<T>(job, 'R', n, ilo, ihi, ls, rs, n, mr, ld), 0);
        ASSERT_EQ(Ggbak<T>(job, 'L', n, ilo, ihi, ls, rs, n, ml, ld), 0);
        const Dense<T> mlt = Adjoint(ToDense(ml, n, n, ld));
        const Dense<T> mrd = ToDense(mr, n, n, ld);
        EXPECT_LT(ResidualRatio(Mul(Mul(mlt, ToDense(a, n, n, ld)), mrd),
                                ToDense(ab, n, n, ld), n),
                  kLapackThreshold);
        EXPECT_LT(ResidualRatio(Mul(Mul(mlt, ToDense(b, n, n, ld)), mrd),
                                ToDense(bb, n, n, ld), n),
                  kLapackThreshold);
        // Rows and columns outside ILO:IHI are triangular.
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = j + 1; i < n; ++i) {
            if (j >= ilo - 1 && i < ihi) continue;
            EXPECT_EQ(At(ab, i, j, ld), T(0)) << i << "," << j;
            EXPECT_EQ(At(bb, i, j, ld), T(0)) << i << "," << j;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// xGGHRD

TYPED_TEST(GgevTest, GghrdHessenbergTriangular) {
  using T = TypeParam;
  Random<T> rng(12);
  for (const hwyblas_int n : {0, 1, 2, 3, 6, 11}) {
    for (const bool partial : {false, true}) {
      if (partial && n < 4) continue;
      const hwyblas_int ld = n + 1;
      const hwyblas_int ilo = partial ? 2 : 1, ihi = partial ? n - 1 : n;
      SCOPED_TRACE(::testing::Message()
                   << "n=" << n << " ilo=" << ilo << " ihi=" << ihi);
      std::vector<T> a = rng.Vector(Sz2(ld, n)), b = rng.Vector(Sz2(ld, n));
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = j + 1; i < n; ++i) {
          At(b, i, j, ld) = T(0);
          if (j < ilo - 1 || i > ihi - 1) At(a, i, j, ld) = T(0);
        }
      }
      // COMPQ = COMPZ = 'I'.
      std::vector<T> h = a, t = b, q(Sz2(ld, n), NaN<T>()),
                     z(Sz2(ld, n), NaN<T>());
      ASSERT_EQ(Gghrd<T>('I', 'I', n, ilo, ihi, h, ld, t, ld, q, ld, z, ld), 0);
      EXPECT_TRUE(IsUpperTriangular(t, n, ld, false));
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = j + 2; i < n; ++i) {
          EXPECT_EQ(At(h, i, j, ld), T(0)) << i << "," << j;
        }
      }
      EXPECT_LT(DecompositionRatio(a, q, h, z, n, ld), kLapackThreshold);
      EXPECT_LT(DecompositionRatio(b, q, t, z, n, ld), kLapackThreshold);
      EXPECT_LT(OrthogonalityRatio(ToDense(q, n, n, ld)), kLapackThreshold);
      EXPECT_LT(OrthogonalityRatio(ToDense(z, n, n, ld)), kLapackThreshold);
      // COMPQ = COMPZ = 'N': the same H and T.
      std::vector<T> h2 = a, t2 = b, q1(1), z1(1);
      ASSERT_EQ(Gghrd<T>('N', 'N', n, ilo, ihi, h2, ld, t2, ld, q1, 1, z1, 1),
                0);
      EXPECT_TRUE(SameValues(h2, h));
      EXPECT_TRUE(SameValues(t2, t));
      // COMPQ = COMPZ = 'V': Q1 * Q and Z1 * Z.
      const std::vector<T> q0 = rng.Vector(Sz2(ld, n)),
                           z0 = rng.Vector(Sz2(ld, n));
      std::vector<T> h3 = a, t3 = b, q3 = q0, z3 = z0;
      ASSERT_EQ(Gghrd<T>('V', 'V', n, ilo, ihi, h3, ld, t3, ld, q3, ld, z3, ld),
                0);
      EXPECT_TRUE(SameValues(h3, h));
      EXPECT_TRUE(SameValues(t3, t));
      EXPECT_LT(ResidualRatio(Mul(ToDense(q0, n, n, ld), ToDense(q, n, n, ld)),
                              ToDense(q3, n, n, ld), n),
                kLapackThreshold);
      EXPECT_LT(ResidualRatio(Mul(ToDense(z0, n, n, ld), ToDense(z, n, n, ld)),
                              ToDense(z3, n, n, ld), n),
                kLapackThreshold);
    }
  }
}

// ---------------------------------------------------------------------------
// xHGEQZ, xTGEVC

// (A, B) with B upper triangular, its generalized Hessenberg form (H, T)
// with Q and Z (xGGHRD), and the generalized Schur form (S, P) in h and t
// with Q and Z updated (xHGEQZ, JOB = 'S').
template <typename T>
void HessenbergSchur(Kind kind, hwyblas_int n, hwyblas_int ld, Random<T>& rng,
                     std::vector<T>& a, std::vector<T>& b, std::vector<T>& h,
                     std::vector<T>& t, std::vector<T>& q, std::vector<T>& z,
                     std::vector<C64>& alpha, std::vector<C64>& beta) {
  TestPair(kind, n, ld, rng, a, b);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j + 1; i < n; ++i) At(b, i, j, ld) = T(0);
  }
  h = a;
  t = b;
  q.assign(Sz2(ld, n), NaN<T>());
  z.assign(Sz2(ld, n), NaN<T>());
  ASSERT_EQ(Gghrd<T>('I', 'I', n, 1, n, h, ld, t, ld, q, ld, z, ld), 0);
  hwyblas_int opt = -1;
  std::vector<T> h0 = h, t0 = t, q0 = q, z0 = z;
  ASSERT_EQ(Hgeqz<T>('S', 'V', 'V', n, 1, n, h0, ld, t0, ld, alpha, beta, q0,
                     ld, z0, ld, -1, &opt),
            0);
  EXPECT_EQ(opt, std::max<hwyblas_int>(1, n));
  ASSERT_EQ(Hgeqz<T>('S', 'V', 'V', n, 1, n, h, ld, t, ld, alpha, beta, q, ld,
                     z, ld, std::max<hwyblas_int>(1, n), &opt),
            0);
  EXPECT_EQ(opt, std::max<hwyblas_int>(1, n));
}

TYPED_TEST(GgevTest, HgeqzSchurForm) {
  using T = TypeParam;
  Random<T> rng(13);
  for (const Kind kind : kKinds) {
    for (const hwyblas_int n : {0, 1, 2, 3, 7, 16}) {
      SCOPED_TRACE(::testing::Message() << KindName(kind) << " n=" << n);
      const hwyblas_int ld = n + 2;
      std::vector<T> a, b, s, p, q, z;
      std::vector<C64> alpha, beta;
      HessenbergSchur(kind, n, ld, rng, a, b, s, p, q, z, alpha, beta);
      ExpectSchurForm(a, b, s, p, q, z, n, ld, alpha, beta);
      // JOB = 'E': the same eigenvalues (to rounding).
      std::vector<T> h = a, t = b, q1(1), z1(1);
      std::vector<T> qh(Sz2(ld, n)), zh(Sz2(ld, n));
      ASSERT_EQ(Gghrd<T>('N', 'N', n, 1, n, h, ld, t, ld, q1, 1, z1, 1), 0);
      std::vector<C64> alpha2, beta2;
      ASSERT_EQ(Hgeqz<T>('E', 'N', 'N', n, 1, n, h, ld, t, ld, alpha2, beta2,
                         q1, 1, z1, 1, std::max<hwyblas_int>(1, n)),
                0);
      if (kind != Kind::kSingularB) {
        ExpectSameLambdas<T>(Lambdas(alpha, beta), Lambdas(alpha2, beta2));
      }
    }
  }
}

// Eigenvalues in ILO:IHI only: the parts outside are already triangular.
TYPED_TEST(GgevTest, HgeqzActiveBlock) {
  using T = TypeParam;
  Random<T> rng(14);
  const hwyblas_int n = 9, ld = 10, ilo = 3, ihi = 7;
  std::vector<T> a = rng.Vector(Sz2(ld, n)), b = rng.Vector(Sz2(ld, n));
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j + 1; i < n; ++i) {
      At(b, i, j, ld) = T(0);
      if (j < ilo - 1 || i > ihi - 1) At(a, i, j, ld) = T(0);
    }
  }
  // Negative diagonal elements of B outside ILO:IHI (made positive).
  At(b, 0, 0, ld) = T(-2);
  At(b, n - 1, n - 1, ld) = T(-3);
  std::vector<T> h = a, t = b, q(Sz2(ld, n)), z(Sz2(ld, n));
  ASSERT_EQ(Gghrd<T>('I', 'I', n, ilo, ihi, h, ld, t, ld, q, ld, z, ld), 0);
  std::vector<C64> alpha, beta;
  ASSERT_EQ(Hgeqz<T>('S', 'V', 'V', n, ilo, ihi, h, ld, t, ld, alpha, beta, q,
                     ld, z, ld, n),
            0);
  ExpectSchurForm(a, b, h, t, q, z, n, ld, alpha, beta);
}

TYPED_TEST(GgevTest, TgevcEigenvectors) {
  using T = TypeParam;
  Random<T> rng(15);
  for (const Kind kind : {Kind::kRandom, Kind::kSingularB, Kind::kPairs,
                          Kind::kTriangular, Kind::kGraded}) {
    for (const hwyblas_int n : {1, 2, 5, 12}) {
      SCOPED_TRACE(::testing::Message() << KindName(kind) << " n=" << n);
      const hwyblas_int ld = n + 1;
      std::vector<T> a, b, s, p, q, z;
      std::vector<C64> alpha, beta;
      HessenbergSchur(kind, n, ld, rng, a, b, s, p, q, z, alpha, beta);
      std::vector<hwyblas_int> sel(Sz(n), 0);
      // HOWMNY = 'A': the eigenvectors of (S, P).
      std::vector<T> vl(Sz2(ld, n), NaN<T>()), vr(Sz2(ld, n), NaN<T>());
      hwyblas_int m = -1;
      ASSERT_EQ(Tgevc<T>('B', 'A', sel, n, s, ld, p, ld, vl, ld, vr, ld, n, &m),
                0);
      EXPECT_EQ(m, n);
      ExpectEigenvectors(s, p, n, ld, alpha, beta, vr, ld, false);
      ExpectEigenvectors(s, p, n, ld, alpha, beta, vl, ld, true);
      // SIDE = 'R' and 'L' alone: the same vectors.
      std::vector<T> vr2(Sz2(ld, n), NaN<T>()), vl2(Sz2(ld, n), NaN<T>());
      std::vector<T> dummy(1);
      ASSERT_EQ(
          Tgevc<T>('R', 'A', sel, n, s, ld, p, ld, dummy, 1, vr2, ld, n, &m),
          0);
      ASSERT_EQ(
          Tgevc<T>('L', 'A', sel, n, s, ld, p, ld, vl2, ld, dummy, 1, n, &m),
          0);
      EXPECT_TRUE(SameValues(vr2, vr));
      EXPECT_TRUE(SameValues(vl2, vl));
      // HOWMNY = 'B': back-transformed with Q and Z, the eigenvectors of
      // (A, B).
      std::vector<T> vlb = q, vrb = z;
      ASSERT_EQ(
          Tgevc<T>('B', 'B', sel, n, s, ld, p, ld, vlb, ld, vrb, ld, n, &m), 0);
      ExpectEigenvectors(a, b, n, ld, alpha, beta, vrb, ld, false);
      ExpectEigenvectors(a, b, n, ld, alpha, beta, vlb, ld, true);
      // HOWMNY = 'S': the selected columns of HOWMNY = 'A' (a complex pair
      // takes two columns if either of its eigenvalues is selected).
      std::vector<hwyblas_int> cols;
      for (hwyblas_int j = 0; j < n; ++j) {
        sel[j] = j % 3 == 1 ? 1 : 0;
        const bool pair = !IsComplex<T>::value && alpha[j].imag() > 0;
        if (pair) {
          if (sel[j] || (j % 3 == 0)) {
            cols.push_back(j);
            cols.push_back(j + 1);
          }
          if (j % 3 == 0) sel[j + 1] = 1;
          ++j;
        } else if (sel[j]) {
          cols.push_back(j);
        }
      }
      const hwyblas_int mm = static_cast<hwyblas_int>(cols.size());
      std::vector<T> vls(Sz2(ld, std::max<hwyblas_int>(1, mm)), NaN<T>()),
          vrs(Sz2(ld, std::max<hwyblas_int>(1, mm)), NaN<T>());
      ASSERT_EQ(
          Tgevc<T>('B', 'S', sel, n, s, ld, p, ld, vls, ld, vrs, ld, mm, &m),
          0);
      ASSERT_EQ(m, mm);
      for (hwyblas_int c = 0; c < mm; ++c) {
        for (hwyblas_int i = 0; i < n; ++i) {
          EXPECT_EQ(At(vrs, i, c, ld), At(vr, i, cols[c], ld)) << i << "," << c;
          EXPECT_EQ(At(vls, i, c, ld), At(vl, i, cols[c], ld)) << i << "," << c;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// xGGEV, xGGES

TYPED_TEST(GgevTest, GgevEigenpairs) {
  using T = TypeParam;
  constexpr bool kComplex = IsComplex<T>::value;
  Random<T> rng(16);
  for (const Kind kind : kKinds) {
    for (const hwyblas_int n : {0, 1, 2, 4, 11, 25}) {
      const hwyblas_int ld = n + 2;
      std::vector<T> a, b;
      TestPair(kind, n, ld, rng, a, b);
      const hwyblas_int minwork =
          std::max<hwyblas_int>(1, (kComplex ? 2 : 8) * n);
      for (const char jobvl : {'V', 'N'}) {
        for (const char jobvr : {'V', 'N'}) {
          for (const bool minimal : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << KindName(kind) << " n=" << n << " " << jobvl
                         << jobvr << (minimal ? " minimal" : ""));
            std::vector<T> af = a, bf = b, vl(Sz2(ld, n), NaN<T>()),
                           vr(Sz2(ld, n), NaN<T>());
            std::vector<C64> alpha, beta;
            hwyblas_int lwork = minwork;
            if (!minimal) {
              ASSERT_EQ(Ggev<T>(jobvl, jobvr, n, af, ld, bf, ld, alpha, beta,
                                vl, ld, vr, ld, -1, &lwork),
                        0);
              EXPECT_GE(lwork, minwork);
            }
            hwyblas_int opt = -1;
            ASSERT_EQ(Ggev<T>(jobvl, jobvr, n, af, ld, bf, ld, alpha, beta, vl,
                              ld, vr, ld, lwork, &opt),
                      0);
            if (!minimal) EXPECT_EQ(opt, lwork);
            if (jobvr == 'V') {
              ExpectEigenvectors(a, b, n, ld, alpha, beta, vr, ld, false);
            }
            if (jobvl == 'V') {
              ExpectEigenvectors(a, b, n, ld, alpha, beta, vl, ld, true);
            }
            if (kind == Kind::kSingularB && n > 0) {
              // The infinite eigenvalue.
              bool infinite = false;
              for (const C64& x : beta) infinite |= x == 0.0;
              EXPECT_TRUE(infinite);
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(GgevTest, GgesSchurFormSorted) {
  using T = TypeParam;
  constexpr bool kComplex = IsComplex<T>::value;
  Random<T> rng(17);
  for (const Kind kind : kKinds) {
    for (const hwyblas_int n : {0, 1, 2, 5, 12, 24}) {
      const hwyblas_int ld = n + 1;
      std::vector<T> a, b;
      TestPair(kind, n, ld, rng, a, b);
      const hwyblas_int minwork =
          n == 0
              ? 1
              : (kComplex ? 2 * n : std::max<hwyblas_int>(8 * n, 6 * n + 16));
      for (const char sort : {'N', 'S'}) {
        for (const bool minimal : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << KindName(kind) << " n=" << n << " sort=" << sort
                       << (minimal ? " minimal" : ""));
          std::vector<T> s = a, p = b, q(Sz2(ld, n), NaN<T>()),
                         z(Sz2(ld, n), NaN<T>());
          std::vector<C64> alpha, beta;
          hwyblas_int sdim = -1, lwork = minwork;
          if (!minimal) {
            ASSERT_EQ(Gges<T>('V', 'V', sort, n, s, ld, p, ld, &sdim, alpha,
                              beta, q, ld, z, ld, -1, &lwork),
                      0);
            EXPECT_GE(lwork, minwork);
          }
          const hwyblas_int info =
              Gges<T>('V', 'V', sort, n, s, ld, p, ld, &sdim, alpha, beta, q,
                      ld, z, ld, lwork);
          // INFO = N+2: after reordering, roundoff changed an eigenvalue so
          // that the selection changed (possible for ill-conditioned pairs).
          const bool changed = info == n + 2 && sort == 'S' &&
                               (kind == Kind::kGraded || kind == Kind::kLarge ||
                                kind == Kind::kSmall);
          ASSERT_TRUE(info == 0 || changed) << info;
          ExpectSchurForm(a, b, s, p, q, z, n, ld, alpha, beta, false);
          if (changed) continue;
          if (sort == 'N') {
            EXPECT_EQ(sdim, 0);
            continue;
          }
          // The selected eigenvalues (negative real part) first.
          hwyblas_int count = 0;
          for (hwyblas_int j = 0; j < n; ++j) {
            const bool selected = (alpha[j] * std::conj(beta[j])).real() < 0;
            if (j < sdim) {
              EXPECT_TRUE(selected) << j;
            } else {
              EXPECT_FALSE(selected) << j;
            }
            count += selected ? 1 : 0;
          }
          EXPECT_EQ(sdim, count);
        }
      }
    }
  }
}

// JOBVSL / JOBVSR = 'N': the same Schur form.
TYPED_TEST(GgevTest, GgesWithoutVectors) {
  using T = TypeParam;
  Random<T> rng(18);
  const hwyblas_int n = 10, ld = 10;
  std::vector<T> a, b;
  TestPair(Kind::kRandom, n, ld, rng, a, b);
  std::vector<T> s = a, p = b, q(Sz2(ld, n)), z(Sz2(ld, n));
  std::vector<C64> alpha, beta, alpha2, beta2;
  hwyblas_int sdim = -1;
  ASSERT_EQ(Gges<T>('V', 'V', 'S', n, s, ld, p, ld, &sdim, alpha, beta, q, ld,
                    z, ld, 400),
            0);
  for (const char jl : {'N', 'V'}) {
    for (const char jr : {'N', 'V'}) {
      std::vector<T> s2 = a, p2 = b, q2(Sz2(ld, n)), z2(Sz2(ld, n));
      hwyblas_int sdim2 = -1;
      ASSERT_EQ(Gges<T>(jl, jr, 'S', n, s2, ld, p2, ld, &sdim2, alpha2, beta2,
                        q2, ld, z2, ld, 400),
                0);
      EXPECT_EQ(sdim2, sdim);
      ExpectSameLambdas<T>(Lambdas(alpha2, beta2), Lambdas(alpha, beta));
    }
  }
}

// ---------------------------------------------------------------------------
// xTGSEN, xTGEXC

TYPED_TEST(GgevTest, TgsenReorders) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  Random<T> rng(19);
  for (const hwyblas_int n : {1, 2, 6, 13}) {
    for (const int pattern : {0, 1, 2}) {
      const hwyblas_int ld = n + 2;
      std::vector<T> a, b, s0, p0, q0, z0;
      std::vector<C64> alpha0, beta0;
      SchurPair(Kind::kRandom, n, ld, rng, a, b, s0, p0, q0, z0, alpha0, beta0);
      // The selection: none (pattern 0), every third (1) or all (2); a
      // complex pair counts twice if either of it is selected.
      std::vector<hwyblas_int> sel(Sz(n), 0);
      std::vector<C64> selected;
      selected.reserve(static_cast<size_t>(n));
      for (hwyblas_int j = 0; j < n; ++j) {
        sel[j] = pattern == 2 || (pattern == 1 && j % 3 == 1) ? 1 : 0;
        const bool pair = !kComplex && alpha0[j].imag() > 0;
        if (pair) {
          if (sel[j]) {
            selected.push_back(alpha0[j] / beta0[j]);
            selected.push_back(alpha0[j + 1] / beta0[j + 1]);
          }
          ++j;
        } else if (sel[j]) {
          selected.push_back(alpha0[j] / beta0[j]);
        }
      }
      std::vector<T> s_ref, p_ref, q_ref, z_ref;
      R dif_f[2] = {}, dif_1[2] = {};
      bool difs_set = false;
      for (const hwyblas_int ijob : {0, 1, 2, 3, 4, 5}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " pattern=" << pattern
                                          << " ijob=" << ijob);
        std::vector<T> s = s0, p = p0, q = q0, z = z0;
        std::vector<C64> alpha, beta;
        TgsenOut<R> out;
        // Workspace query.
        ASSERT_EQ(Tgsen<T>(ijob, true, true, sel, n, s, ld, p, ld, alpha, beta,
                           q, ld, z, ld, out, -1, -1),
                  0);
        const hwyblas_int lwork = out.work0, liwork = out.iwork0;
        EXPECT_GE(lwork, 1);
        EXPECT_GE(liwork, 1);
        ASSERT_EQ(Tgsen<T>(ijob, true, true, sel, n, s, ld, p, ld, alpha, beta,
                           q, ld, z, ld, out, lwork, liwork),
                  0);
        const hwyblas_int m = static_cast<hwyblas_int>(selected.size());
        EXPECT_EQ(out.m, m);
        ExpectSchurForm(a, b, s, p, q, z, n, ld, alpha, beta, false);
        // The selected eigenvalues come first.
        std::vector<C64> first(static_cast<size_t>(m));
        for (hwyblas_int j = 0; j < m; ++j) first[j] = alpha[j] / beta[j];
        const std::vector<C64> ones(first.size(), C64(1, 0));
        ExpectSameLambdas<T>(Lambdas(first, ones), Lambdas(selected, ones));
        if (ijob == 1 || ijob == 4 || ijob == 5) {
          EXPECT_GT(out.pl, 0);
          EXPECT_LE(out.pl, 1 + 10 * LapackEps<T>());
          EXPECT_GT(out.pr, 0);
          EXPECT_LE(out.pr, 1 + 10 * LapackEps<T>());
        }
        if (ijob >= 2) {
          EXPECT_GT(out.dif[0], 0);
          EXPECT_GT(out.dif[1], 0);
          EXPECT_TRUE(std::isfinite(out.dif[0]) && std::isfinite(out.dif[1]));
          const bool proper = out.m > 0 && out.m < n;
          if (ijob == 2 && proper) std::copy(out.dif, out.dif + 2, dif_f);
          if (ijob == 3 && proper) {
            std::copy(out.dif, out.dif + 2, dif_1);
            difs_set = true;
          }
        }
        // The reordering does not depend on IJOB.
        if (ijob == 0) {
          s_ref = s;
          p_ref = p;
          q_ref = q;
          z_ref = z;
        } else {
          EXPECT_TRUE(SameValues(s, s_ref));
          EXPECT_TRUE(SameValues(p, p_ref));
          EXPECT_TRUE(SameValues(q, q_ref));
          EXPECT_TRUE(SameValues(z, z_ref));
        }
      }
      // The two estimates of Difu and Difl (Frobenius norm, IJOB = 2, and
      // 1-norm, IJOB = 3) agree up to a modest factor.
      if (difs_set) {
        for (int k = 0; k < 2; ++k) {
          const double ratio = double(dif_f[k]) / double(dif_1[k]);
          EXPECT_GT(ratio, 0.05) << k;
          EXPECT_LT(ratio, 20) << k;
        }
      }
      // WANTQ = WANTZ = false: the same pair.
      std::vector<T> s = s0, p = p0, q1(1), z1(1);
      std::vector<C64> alpha, beta;
      TgsenOut<R> out;
      ASSERT_EQ(Tgsen<T>(0, false, false, sel, n, s, ld, p, ld, alpha, beta, q1,
                         1, z1, 1, out, 4 * n + 16, n + 6),
                0);
      EXPECT_TRUE(SameValues(s, s_ref));
      EXPECT_TRUE(SameValues(p, p_ref));
    }
  }
}

TYPED_TEST(GgevTest, TgexcMovesBlocks) {
  using T = TypeParam;
  constexpr bool kComplex = IsComplex<T>::value;
  Random<T> rng(20);
  const hwyblas_int n = 9, ld = 10;
  for (const auto& move : std::vector<std::pair<hwyblas_int, hwyblas_int>>{
           {1, n}, {n, 1}, {3, 6}, {6, 2}, {4, 4}}) {
    SCOPED_TRACE(::testing::Message() << move.first << " -> " << move.second);
    std::vector<T> a, b, s, p, q, z;
    std::vector<C64> alpha, beta;
    SchurPair(Kind::kRandom, n, ld, rng, a, b, s, p, q, z, alpha, beta);
    hwyblas_int ifst = move.first, ilst = move.second;
    // The eigenvalue at IFST (the first row of its block).
    hwyblas_int first = ifst - 1;
    if (!kComplex && first > 0 && At(s, first, first - 1, ld) != T(0)) --first;
    const C64 lambda = alpha[first] / beta[first];
    const hwyblas_int target = ilst;
    ASSERT_EQ(Tgexc<T>(true, true, n, s, ld, p, ld, q, ld, z, ld, &ifst, &ilst),
              0);
    if (kComplex) {
      // As in the reference: the final position minus one after a move
      // down.
      EXPECT_EQ(ilst, move.first < target ? target - 1 : target);
      ilst = target;
    }
    // Its new position.
    hwyblas_int last = ilst - 1;
    if (!kComplex && last > 0 && At(s, last, last - 1, ld) != T(0)) --last;
    const T s_ll = At(s, last, last, ld), p_ll = At(p, last, last, ld);
    if (kComplex || last + 1 == n || At(s, last + 1, last, ld) == T(0)) {
      const C64 l2 = static_cast<C64>(s_ll) / static_cast<C64>(p_ll);
      EXPECT_LE(std::abs(l2 - lambda), 1e3 * LapackEps<T>() * std::abs(lambda));
    }
    EXPECT_TRUE(IsUpperTriangular(s, n, ld, !kComplex));
    EXPECT_TRUE(IsUpperTriangular(p, n, ld, false));
    EXPECT_LT(DecompositionRatio(a, q, s, z, n, ld), kLapackThreshold);
    EXPECT_LT(DecompositionRatio(b, q, p, z, n, ld), kLapackThreshold);
  }
}

// ||Z**(-1)||_1 for the operator of the generalized Sylvester equation
// (A * R - L * B, D * R - L * E) with A, D of order m and B, E of order n,
// acting on (vec(R), vec(L)): Z is formed and inverted (Gauss-Jordan with
// partial pivoting, in double precision).
template <typename T>
double SylvesterInverseNorm1(const Dense<T>& a, const Dense<T>& b,
                             const Dense<T>& d, const Dense<T>& e) {
  const size_t m = a.rows, n = b.rows, k = 2 * m * n;
  std::vector<C64> z(k * k), inv(k * k);
  const auto at = [k](std::vector<C64>& x, size_t i, size_t j) -> C64& {
    return x[i + j * k];
  };
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      const size_t row = i + j * m;
      // (A R)(i,j) and (D R)(i,j).
      for (size_t l = 0; l < m; ++l) {
        at(z, row, l + j * m) += static_cast<C64>(a(i, l));
        at(z, m * n + row, l + j * m) += static_cast<C64>(d(i, l));
      }
      // -(L B)(i,j) and -(L E)(i,j).
      for (size_t l = 0; l < n; ++l) {
        at(z, row, m * n + i + l * m) -= static_cast<C64>(b(l, j));
        at(z, m * n + row, m * n + i + l * m) -= static_cast<C64>(e(l, j));
      }
    }
  }
  for (size_t i = 0; i < k; ++i) at(inv, i, i) = 1;
  for (size_t c = 0; c < k; ++c) {
    size_t piv = c;
    for (size_t i = c + 1; i < k; ++i) {
      if (std::abs(at(z, i, c)) > std::abs(at(z, piv, c))) piv = i;
    }
    for (size_t j = 0; j < k; ++j) {
      std::swap(at(z, c, j), at(z, piv, j));
      std::swap(at(inv, c, j), at(inv, piv, j));
    }
    const C64 p = at(z, c, c);
    for (size_t j = 0; j < k; ++j) {
      at(z, c, j) /= p;
      at(inv, c, j) /= p;
    }
    for (size_t i = 0; i < k; ++i) {
      if (i == c) continue;
      const C64 f = at(z, i, c);
      if (f == 0.0) continue;
      for (size_t j = 0; j < k; ++j) {
        at(z, i, j) -= f * at(z, c, j);
        at(inv, i, j) -= f * at(inv, c, j);
      }
    }
  }
  double norm = 0;
  for (size_t j = 0; j < k; ++j) {
    double col = 0;
    for (size_t i = 0; i < k; ++i) col += std::abs(at(inv, i, j));
    norm = std::max(norm, col);
  }
  return norm;
}

// The leading (first = true) or trailing part of order m of the n x n
// matrix x.
template <typename T>
Dense<T> Part(const std::vector<T>& x, hwyblas_int n, hwyblas_int ld,
              hwyblas_int m, bool first) {
  const hwyblas_int o = first ? 0 : n - m;
  Dense<T> d(static_cast<size_t>(m), static_cast<size_t>(m));
  for (hwyblas_int j = 0; j < m; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      d(i, j) = static_cast<RefT<T>>(At(x, o + i, o + j, ld));
    }
  }
  return d;
}

// xTGSEN's 1-norm-based estimates DIF(1) = 1 / ||Zu**(-1)||_1 and DIF(2)
// = 1 / ||Zl**(-1)||_1 (IJOB = 3; Zu: the operator of (A11, A22), (B11,
// B22), Zl: of (A22, A11), (B22, B11)): xLACN2's estimates are lower bounds
// of the norms, and here close to them.
TYPED_TEST(GgevTest, TgsenDifEstimates) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(33);
  for (const hwyblas_int n : {3, 5, 6}) {
    for (int trial = 0; trial < 4; ++trial) {
      const hwyblas_int ld = n;
      std::vector<T> a, b, s, p, q, z;
      std::vector<C64> alpha, beta;
      SchurPair(Kind::kRandom, n, ld, rng, a, b, s, p, q, z, alpha, beta);
      // Select the first eigenvalue (and its pair).
      std::vector<hwyblas_int> sel(Sz(n), 0);
      sel[trial % 2 == 0 ? 0 : n - 1] = 1;
      TgsenOut<R> out;
      ASSERT_EQ(Tgsen<T>(3, true, true, sel, n, s, ld, p, ld, alpha, beta, q,
                         ld, z, ld, out, 4 * n * n + 16, 2 * n * n + 6),
                0);
      const hwyblas_int m = out.m;
      ASSERT_GT(m, 0);
      ASSERT_LT(m, n);
      const Dense<T> a11 = Part(s, n, ld, m, true),
                     a22 = Part(s, n, ld, n - m, false),
                     b11 = Part(p, n, ld, m, true),
                     b22 = Part(p, n, ld, n - m, false);
      const double exact[2] = {SylvesterInverseNorm1(a11, a22, b11, b22),
                               SylvesterInverseNorm1(a22, a11, b22, b11)};
      for (int k = 0; k < 2; ++k) {
        const double estimate = 1 / double(out.dif[k]);
        SCOPED_TRACE(::testing::Message() << "n=" << n << " trial=" << trial
                                          << " DIF(" << k + 1 << ")");
        EXPECT_LE(estimate, exact[k] * (1 + 1e3 * LapackEps<T>()));
        EXPECT_GE(estimate, exact[k] / 3);
      }
    }
  }
}

// xTGEXC (real T): a 2-by-2 block moved down to the last row (a 1-by-1
// block) ends at row N-1, which ILST returns.
TYPED_TEST(GgevRealTest, TgexcMovesPairDown) {
  using R = TypeParam;
  Random<R> rng(32);
  const hwyblas_int n = 7, ld = 7;
  // A = diag(rotation, 3, ..., 7) + strictly upper random, B = I + strictly
  // upper random: one complex pair and five real eigenvalues.
  std::vector<R> a = rng.Vector(Sz2(ld, n)), b = rng.Vector(Sz2(ld, n));
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j; i < n; ++i) {
      At(a, i, j, ld) = i == j ? R(j < 2 ? 1 : j + 1) : R(0);
      At(b, i, j, ld) = i == j ? R(1) : R(0);
    }
  }
  At(a, 1, 0, ld) = R(-2);
  At(a, 0, 1, ld) = R(2);
  std::vector<R> s = a, p = b, q(Sz2(ld, n)), z(Sz2(ld, n));
  std::vector<C64> alpha, beta;
  hwyblas_int sdim = -1;
  ASSERT_EQ(Gges<R>('V', 'V', 'N', n, s, ld, p, ld, &sdim, alpha, beta, q, ld,
                    z, ld, 64 * n),
            0);
  hwyblas_int pair = -1;
  for (hwyblas_int j = 0; j + 1 < n; ++j) {
    if (At(s, j + 1, j, ld) != R(0)) pair = j;
  }
  ASSERT_GE(pair, 0);
  ASSERT_LT(pair, n - 2);
  const C64 lambda = alpha[pair] / beta[pair];
  hwyblas_int ifst = pair + 1, ilst = n;
  ASSERT_EQ(Tgexc<R>(true, true, n, s, ld, p, ld, q, ld, z, ld, &ifst, &ilst),
            0);
  EXPECT_EQ(ilst, n - 1);
  EXPECT_NE(At(s, n - 1, n - 2, ld), R(0));
  // The eigenvalues of the moved block (with xLAG2) are those of the pair.
  R s1, s2, wr1, wr2, wi;
  const R sm = 100 * std::numeric_limits<R>::min();
  const hwyblas_int l2 = ld;
  HWYBLAS_CALL_REAL(R, lag2, &At(s, n - 2, n - 2, ld), &l2,
                    &At(p, n - 2, n - 2, ld), &l2, &sm, &s1, &s2, &wr1, &wr2,
                    &wi);
  const C64 moved(wr1 / s1, std::abs(wi) / s1);
  EXPECT_LE(std::abs(moved - C64(lambda.real(), std::abs(lambda.imag()))),
            1e3 * LapackEps<R>() * std::abs(lambda));
  EXPECT_LT(DecompositionRatio(a, q, s, z, n, ld), kLapackThreshold);
  EXPECT_LT(DecompositionRatio(b, q, p, z, n, ld), kLapackThreshold);
}

// ---------------------------------------------------------------------------
// xTGSYL

// Forces the block sizes MB and NB of xTGSYL (ILAENV ispec 2 and 5) while
// it exists.
class ScopedTgsylBlocks {
 public:
  ScopedTgsylBlocks(hwyblas_int mb, hwyblas_int nb) {
    mb_ = mb;
    nb_ = nb;
    previous_ = hwyblas_set_ilaenv(mb > 0 ? &Ilaenv : nullptr);
  }
  ~ScopedTgsylBlocks() { hwyblas_set_ilaenv(previous_); }
  ScopedTgsylBlocks(const ScopedTgsylBlocks&) = delete;
  ScopedTgsylBlocks& operator=(const ScopedTgsylBlocks&) = delete;

 private:
  static hwyblas_int Ilaenv(const hwyblas_int* ispec, const char* name,
                            const char* opts, const hwyblas_int* n1,
                            const hwyblas_int* n2, const hwyblas_int* n3,
                            const hwyblas_int* n4, size_t name_len,
                            size_t opts_len) {
    if (*ispec == 2) return mb_;
    if (*ispec == 5) return nb_;
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }
  static inline hwyblas_int mb_ = 0, nb_ = 0;
  hwyblas_ilaenv_function previous_ = nullptr;
};

TYPED_TEST(GgevTest, TgsylSolves) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const char ct = kComplex ? 'C' : 'T';
  Random<T> rng(21);
  for (const hwyblas_int m : {1, 2, 5, 8}) {
    for (const hwyblas_int n : {1, 3, 6}) {
      const hwyblas_int ld = 9;
      // (A, D) and (B, E) in generalized Schur form.
      std::vector<T> a0, d0, a, d, qa, za, b0, e0, b, e, qb, zb;
      std::vector<C64> al, be;
      SchurPair(Kind::kRandom, m, ld, rng, a0, d0, a, d, qa, za, al, be);
      SchurPair(Kind::kRandom, n, ld, rng, b0, e0, b, e, qb, zb, al, be);
      const std::vector<T> c0 = rng.Vector(Sz2(ld, n)),
                           f0 = rng.Vector(Sz2(ld, n));
      const Dense<T> ad = ToDense(a, m, m, ld), dd = ToDense(d, m, m, ld),
                     bd = ToDense(b, n, n, ld), ed = ToDense(e, n, n, ld);
      for (const auto& blocks :
           std::vector<std::pair<hwyblas_int, hwyblas_int>>{
               {0, 0}, {1, 1}, {3, 2}, {2, 4}}) {
        ScopedTgsylBlocks scoped(blocks.first, blocks.second);
        for (const char trans : {'N', ct}) {
          SCOPED_TRACE(::testing::Message()
                       << "m=" << m << " n=" << n << " blocks " << blocks.first
                       << "," << blocks.second << " " << trans);
          std::vector<T> c = c0, f = f0;
          R scale = -1, dif = -1;
          hwyblas_int opt = -1;
          ASSERT_EQ(Tgsyl<T>(trans, 0, m, n, a, ld, b, ld, c, ld, d, ld, e, ld,
                             f, ld, &scale, &dif, -1, &opt),
                    0);
          EXPECT_EQ(opt, 1);
          ASSERT_EQ(Tgsyl<T>(trans, 0, m, n, a, ld, b, ld, c, ld, d, ld, e, ld,
                             f, ld, &scale, &dif, 1),
                    0);
          EXPECT_GT(scale, 0);
          EXPECT_LE(scale, 1);
          const Dense<T> r = ToDense(c, m, n, ld), l = ToDense(f, m, n, ld);
          Dense<T> sc = ToDense(c0, m, n, ld), sf = ToDense(f0, m, n, ld);
          for (auto& x : sc.v) x *= scale;
          for (auto& x : sf.v) x *= scale;
          Dense<T> lhs1(m, n), lhs2(m, n);
          if (trans == 'N') {
            // A * R - L * B = scale * C, D * R - L * E = scale * F.
            lhs1 = Sub(Mul(ad, r), Mul(l, bd));
            lhs2 = Sub(Mul(dd, r), Mul(l, ed));
          } else {
            // A**H * R + D**H * L = scale * C,
            // R * B**H + L * E**H = -scale * F.
            lhs1 = Mul(Adjoint(ad), r);
            const Dense<T> t1 = Mul(Adjoint(dd), l);
            for (size_t i = 0; i < lhs1.v.size(); ++i) lhs1.v[i] += t1.v[i];
            lhs2 = Mul(r, Adjoint(bd));
            const Dense<T> t2 = Mul(l, Adjoint(ed));
            for (size_t i = 0; i < lhs2.v.size(); ++i) {
              lhs2.v[i] = -(lhs2.v[i] + t2.v[i]);
            }
          }
          const double scale_rl =
              (Norm1(ad) + Norm1(bd) + Norm1(dd) + Norm1(ed)) *
              (Norm1(r) + Norm1(l));
          for (const auto& pr :
               {std::make_pair(&lhs1, &sc), std::make_pair(&lhs2, &sf)}) {
            double resid = 0;
            for (size_t i = 0; i < pr.first->v.size(); ++i) {
              resid += std::abs(pr.first->v[i] - pr.second->v[i]);
            }
            EXPECT_LT(resid / std::max(scale_rl, 1e-300) /
                          std::max<hwyblas_int>(m, n) / LapackEps<T>(),
                      kLapackThreshold);
          }
          if (trans != 'N') continue;
          // The Dif-estimates (IJOB 1, 2: with the solution; 3, 4: only the
          // estimate, the same values).
          R difs[5] = {};
          for (const hwyblas_int ijob : {1, 2, 3, 4}) {
            std::vector<T> c2 = c0, f2 = f0;
            R scale2 = -1;
            ASSERT_EQ(Tgsyl<T>('N', ijob, m, n, a, ld, b, ld, c2, ld, d, ld, e,
                               ld, f2, ld, &scale2, &difs[ijob], -1, &opt),
                      0);
            EXPECT_EQ(opt, ijob <= 2 ? 2 * m * n : 1);
            ASSERT_EQ(Tgsyl<T>('N', ijob, m, n, a, ld, b, ld, c2, ld, d, ld, e,
                               ld, f2, ld, &scale2, &difs[ijob], 2 * m * n),
                      0);
            EXPECT_GT(difs[ijob], 0);
            EXPECT_TRUE(std::isfinite(difs[ijob]));
            if (ijob <= 2) {
              // The solution of IJOB = 0.
              EXPECT_EQ(scale2, scale);
              EXPECT_TRUE(SameValues(c2, c));
              EXPECT_TRUE(SameValues(f2, f));
            }
          }
          EXPECT_EQ(difs[1], difs[3]);
          EXPECT_EQ(difs[2], difs[4]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// xGGSVD3, xGGSVP3

// Checks the GSVD of the m x n A and the p x n B (a0, b0) as LAPACK's
// xGSVTS3 does: U**H A Q - D1 R and V**H B Q - D2 R, the orthogonality of
// U, V, Q, ALPHA**2 + BETA**2 = 1 and the order of ALPHA (with IWORK).
template <typename T>
void ExpectGsvd(const std::vector<T>& a0, const std::vector<T>& b0,
                const std::vector<T>& af, const std::vector<T>& bf,
                hwyblas_int m, hwyblas_int p, hwyblas_int n, hwyblas_int ld,
                const std::vector<T>& u, const std::vector<T>& v,
                const std::vector<T>& q, const Ggsvd3Out<RealT<T>>& out) {
  using Ref = RefT<T>;
  const hwyblas_int k = out.k, l = out.l;
  ASSERT_GE(k, 0);
  ASSERT_GE(l, 0);
  ASSERT_LE(k + l, n);
  // R ((k+l) x (k+l), upper triangular).
  Dense<T> r(static_cast<size_t>(k + l), static_cast<size_t>(k + l));
  for (hwyblas_int i = 0; i < std::min(k + l, m); ++i) {
    for (hwyblas_int j = i; j < k + l; ++j) {
      r(i, j) = static_cast<Ref>(At(af, i, n - k - l + j, ld));
    }
  }
  if (m - k - l < 0) {
    for (hwyblas_int i = m; i < k + l; ++i) {
      for (hwyblas_int j = i; j < k + l; ++j) {
        r(i, j) = static_cast<Ref>(At(bf, i - k, n - k - l + j, ld));
      }
    }
  }
  const Dense<T> qd = ToDense(q, n, n, ld);
  // U**H * A * Q - D1 * (0 R).
  Dense<T> x =
      Mul(Mul(Adjoint(ToDense(u, m, m, ld)), ToDense(a0, m, n, ld)), qd);
  for (hwyblas_int i = 0; i < k; ++i) {
    for (hwyblas_int j = i; j < k + l; ++j) x(i, n - k - l + j) -= r(i, j);
  }
  for (hwyblas_int i = 0; i < std::min(l, m - k); ++i) {
    for (hwyblas_int j = i; j < l; ++j) {
      x(k + i, n - l + j) -=
          static_cast<Ref>(out.alpha[k + i]) * r(k + i, k + j);
    }
  }
  const double anorm = std::max(Norm1(ToDense(a0, m, n, ld)), 1e-300);
  if (m > 0 && n > 0) {
    EXPECT_LT(Norm1(x) / anorm / std::max<hwyblas_int>(m, n) / LapackEps<T>(),
              kLapackThreshold);
  }
  // V**H * B * Q - D2 * (0 R).
  Dense<T> y =
      Mul(Mul(Adjoint(ToDense(v, p, p, ld)), ToDense(b0, p, n, ld)), qd);
  for (hwyblas_int i = 0; i < l; ++i) {
    for (hwyblas_int j = i; j < l; ++j) {
      y(i, n - l + j) -= static_cast<Ref>(out.beta[k + i]) * r(k + i, k + j);
    }
  }
  const double bnorm = std::max(Norm1(ToDense(b0, p, n, ld)), 1e-300);
  if (p > 0 && n > 0) {
    EXPECT_LT(Norm1(y) / bnorm / std::max<hwyblas_int>(p, n) / LapackEps<T>(),
              kLapackThreshold);
  }
  if (m > 0) {
    EXPECT_LT(OrthogonalityRatio(ToDense(u, m, m, ld)), kLapackThreshold);
  }
  if (p > 0) {
    EXPECT_LT(OrthogonalityRatio(ToDense(v, p, p, ld)), kLapackThreshold);
  }
  if (n > 0) {
    EXPECT_LT(OrthogonalityRatio(qd), kLapackThreshold);
  }
  for (hwyblas_int i = k; i < k + l; ++i) {
    const double c = out.alpha[i], s = out.beta[i];
    EXPECT_NEAR(c * c + s * s, 1.0, 100 * LapackEps<T>()) << i;
  }
  for (hwyblas_int i = 0; i < k; ++i) {
    EXPECT_EQ(out.alpha[i], 1) << i;
    EXPECT_EQ(out.beta[i], 0) << i;
  }
  // ALPHA sorted with IWORK: swap ALPHA(i) and ALPHA(IWORK(i)) for i = k+1
  // ... min(m, k+l) gives a nonincreasing sequence.
  std::vector<double> sorted(out.alpha.begin(), out.alpha.begin() + n);
  for (hwyblas_int i = k; i < std::min(m, k + l); ++i) {
    std::swap(sorted[i], sorted[out.iwork[i] - 1]);
  }
  for (hwyblas_int i = k + 1; i < std::min(m, k + l); ++i) {
    EXPECT_GE(sorted[i - 1], sorted[i]) << i;
  }
}

TYPED_TEST(GgevTest, Ggsvd3Decomposition) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(22);
  struct Shape {
    hwyblas_int m, p, n;
  };
  for (const Shape sh :
       {Shape{0, 0, 0}, Shape{1, 1, 1}, Shape{5, 4, 3}, Shape{3, 4, 6},
        Shape{6, 2, 4}, Shape{2, 6, 5}, Shape{8, 7, 8}, Shape{0, 3, 2},
        Shape{3, 0, 2}, Shape{12, 10, 9}}) {
    for (const int rank : {0, 1}) {
      const hwyblas_int m = sh.m, p = sh.p, n = sh.n, ld = 13;
      SCOPED_TRACE(::testing::Message() << "m=" << m << " p=" << p << " n=" << n
                                        << (rank ? " deficient" : ""));
      std::vector<T> a = rng.Vector(Sz2(ld, n)), b = rng.Vector(Sz2(ld, n));
      if (rank == 1) {
        // Rank deficiency: A's last row = its first, B's column 1 = column 0
        // and A's too.
        for (hwyblas_int j = 0; j < n; ++j) {
          if (m > 1) At(a, m - 1, j, ld) = At(a, 0, j, ld);
        }
        if (n > 1) {
          for (hwyblas_int i = 0; i < ld; ++i) {
            At(a, i, 1, ld) = At(a, i, 0, ld);
            At(b, i, 1, ld) = At(b, i, 0, ld);
          }
        }
      }
      std::vector<T> af = a, bf = b, u(Sz2(ld, m), NaN<T>()),
                     v(Sz2(ld, p), NaN<T>()), q(Sz2(ld, n), NaN<T>());
      Ggsvd3Out<R> out;
      hwyblas_int lwork = -1;
      ASSERT_EQ(Ggsvd3<T>('U', 'V', 'Q', m, n, p, af, ld, bf, ld, u, ld, v, ld,
                          q, ld, out, -1, &lwork),
                0);
      ASSERT_GE(lwork, 1);
      ASSERT_EQ(Ggsvd3<T>('U', 'V', 'Q', m, n, p, af, ld, bf, ld, u, ld, v, ld,
                          q, ld, out, lwork),
                0);
      ExpectGsvd(a, b, af, bf, m, p, n, ld, u, v, q, out);
      // Without U, V, Q: the same results.
      std::vector<T> af2 = a, bf2 = b, d1(1), d2(1), d3(1);
      Ggsvd3Out<R> out2;
      ASSERT_EQ(Ggsvd3<T>('N', 'N', 'N', m, n, p, af2, ld, bf2, ld, d1, 1, d2,
                          1, d3, 1, out2, lwork),
                0);
      EXPECT_EQ(out2.k, out.k);
      EXPECT_EQ(out2.l, out.l);
      EXPECT_TRUE(SameValues(out2.alpha, out.alpha));
      EXPECT_TRUE(SameValues(out2.beta, out.beta));
      EXPECT_TRUE(SameValues(af2, af));
      EXPECT_TRUE(SameValues(bf2, bf));
    }
  }
}

// xGGSVP3 alone: U**H A Q = ( 0 A12 A13 ; 0 0 A23 ; 0 0 0 ) and V**H B Q
// = ( 0 0 B13 ; 0 0 0 ), with A12 (k x k) and B13 (l x l) upper triangular.
TYPED_TEST(GgevTest, Ggsvp3Preprocessing) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(23);
  const hwyblas_int m = 6, p = 4, n = 5, ld = 7;
  const std::vector<T> a = rng.Vector(Sz2(ld, n)), b = rng.Vector(Sz2(ld, n));
  const R tol = R(10) * std::numeric_limits<R>::epsilon();
  std::vector<T> af = a, bf = b, u(Sz2(ld, m)), v(Sz2(ld, p)), q(Sz2(ld, n));
  hwyblas_int k = -1, l = -1, lwork = -1;
  ASSERT_EQ(Ggsvp3<T>('U', 'V', 'Q', m, p, n, af, ld, bf, ld, tol, tol, &k, &l,
                      u, ld, v, ld, q, ld, -1, &lwork),
            0);
  ASSERT_EQ(Ggsvp3<T>('U', 'V', 'Q', m, p, n, af, ld, bf, ld, tol, tol, &k, &l,
                      u, ld, v, ld, q, ld, lwork),
            0);
  EXPECT_EQ(k + l, n);
  EXPECT_EQ(l, p);
  // A = U * A' * Q**H, B = V * B' * Q**H.
  const Dense<T> qd = ToDense(q, n, n, ld);
  EXPECT_LT(ResidualRatio(ToDense(a, m, n, ld),
                          Mul(Mul(ToDense(u, m, m, ld), ToDense(af, m, n, ld)),
                              Adjoint(qd)),
                          n),
            kLapackThreshold);
  EXPECT_LT(ResidualRatio(ToDense(b, p, n, ld),
                          Mul(Mul(ToDense(v, p, p, ld), ToDense(bf, p, n, ld)),
                              Adjoint(qd)),
                          n),
            kLapackThreshold);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      // A12 starts at column n-k-l, upper triangular; A23 at row k.
      const bool zero =
          j < n - k - l || (i < k && j - (n - k - l) < i) ||
          (i >= k && (j < n - l || i >= k + l || j - (n - l) < i - k));
      if (zero) EXPECT_EQ(At(af, i, j, ld), T(0)) << i << "," << j;
    }
    for (hwyblas_int i = 0; i < p; ++i) {
      if (j < n - l || i >= l || j - (n - l) < i) {
        EXPECT_EQ(At(bf, i, j, ld), T(0)) << i << "," << j;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Auxiliary routines

template <typename T>
RealT<T> Lanhs(char norm, hwyblas_int n, const std::vector<T>& a,
               hwyblas_int lda, std::vector<RealT<T>>& work) {
  if constexpr (std::is_same_v<T, float>) {
    return slanhs_(&norm, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, double>) {
    return dlanhs_(&norm, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return clanhs_(&norm, &n, P(a.data()), &lda, work.data());
  } else {
    return zlanhs_(&norm, &n, P(a.data()), &lda, work.data());
  }
}

TYPED_TEST(GgevTest, Lanhs) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(24);
  for (const hwyblas_int n : {0, 1, 2, 5, 9}) {
    const hwyblas_int ld = n + 1;
    std::vector<T> a = rng.Vector(Sz2(ld, n));
    // Not referenced: below the subdiagonal.
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = j + 2; i < n; ++i) At(a, i, j, ld) = NaN<T>();
    }
    double maxabs = 0, one = 0, inf = 0, fro = 0;
    std::vector<double> rows(Sz(n), 0);
    for (hwyblas_int j = 0; j < n; ++j) {
      double col = 0;
      for (hwyblas_int i = 0; i <= std::min(j + 1, n - 1); ++i) {
        const double x = std::abs(At(a, i, j, ld));
        maxabs = std::max(maxabs, x);
        col += x;
        rows[i] += x;
        fro += x * x;
      }
      one = std::max(one, col);
    }
    for (const double r : rows) inf = std::max(inf, r);
    fro = std::sqrt(fro);
    std::vector<R> work(Sz(n), NAN);
    for (const auto& c : std::vector<std::pair<char, double>>{{'M', maxabs},
                                                              {'1', one},
                                                              {'O', one},
                                                              {'I', inf},
                                                              {'F', fro},
                                                              {'E', fro}}) {
      const char norm = c.first;
      const R value = Lanhs<T>(norm, n, a, ld, work);
      EXPECT_NEAR(value, c.second, 10 * LapackEps<T>() * (c.second + 1))
          << norm << " n=" << n;
    }
    // A NaN in the Hessenberg part propagates.
    if (n > 0) {
      At(a, n - 1, n - 1, ld) = NaN<T>();
      for (const char norm : {'M', '1', 'I', 'F'}) {
        const R value = Lanhs<T>(norm, n, a, ld, work);
        EXPECT_TRUE(std::isnan(value)) << norm;
      }
    }
  }
}

TYPED_TEST(GgevTest, Lapmt) {
  using T = TypeParam;
  Random<T> rng(25);
  const hwyblas_int m = 4, n = 6, ld = 5;
  const std::vector<T> x0 = rng.Vector(Sz2(ld, n));
  const std::vector<hwyblas_int> k0 = {3, 1, 6, 2, 5, 4};
  for (const hwyblas_int forwrd : {1, 0}) {
    std::vector<T> x = x0;
    std::vector<hwyblas_int> k = k0;
    CALL_RC(T, lapmt, lapmt, &forwrd, &m, &n, P(x.data()), &ld, k.data());
    EXPECT_EQ(k, k0);
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = 0; i < m; ++i) {
        if (forwrd) {
          EXPECT_EQ(At(x, i, j, ld), At(x0, i, k0[j] - 1, ld));
        } else {
          EXPECT_EQ(At(x, i, k0[j] - 1, ld), At(x0, i, j, ld));
        }
      }
    }
  }
}

TYPED_TEST(GgevTest, Lapll) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(26);
  for (const hwyblas_int n : {1, 2, 3, 7}) {
    for (const hwyblas_int inc : {1, 2}) {
      std::vector<T> x = rng.Vector(Sz(n * inc)), y = rng.Vector(Sz(n * inc));
      if (n == 3) {
        // Nearly parallel.
        for (hwyblas_int i = 0; i < n; ++i) {
          y[i * inc] = x[i * inc] * T(R(2)) + T(R(1e-3));
        }
      }
      // The smallest singular value of ( x y ) from its Gram matrix.
      C64 xx = 0, xy = 0, yy = 0;
      for (hwyblas_int i = 0; i < n; ++i) {
        const C64 xi = static_cast<C64>(x[i * inc]);
        const C64 yi = static_cast<C64>(y[i * inc]);
        xx += std::conj(xi) * xi;
        xy += std::conj(xi) * yi;
        yy += std::conj(yi) * yi;
      }
      const double tr = (xx + yy).real();
      const double det = (xx * yy).real() - std::norm(xy);
      const double smin =
          n == 1 ? 0
                 : std::sqrt(std::max(
                       0.0,
                       (tr - std::sqrt(std::max(0.0, tr * tr - 4 * det))) / 2));
      R ssmin = -1;
      CALL_RC(T, lapll, lapll, &n, P(x.data()), &inc, P(y.data()), &inc,
              &ssmin);
      EXPECT_NEAR(ssmin, smin, 1e3 * LapackEps<T>() * std::sqrt(tr))
          << "n=" << n << " inc=" << inc;
    }
  }
}

TYPED_TEST(GgevTest, Lags2ZeroesElements) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(27);
  Random<R> rr(28);
  for (int trial = 0; trial < 50; ++trial) {
    for (const hwyblas_int upper : {1, 0}) {
      const R a1 = rr.Next(), a3 = rr.Next(), b1 = rr.Next(), b3 = rr.Next();
      const T a2 = rng.Next(), b2 = rng.Next();
      R csu, csv, csq;
      T snu, snv, snq;
      CALL_RC(T, lags2, lags2, &upper, &a1, P(&a2), &a3, &b1, P(&b2), &b3, &csu,
              P(&snu), &csv, P(&snv), &csq, P(&snq));
      // U = ( CSU SNU ; -SNU**H CSU ) and V, Q likewise.
      const auto rot = [](R c, T s) {
        Dense<T> x(2, 2);
        x(0, 0) = c;
        x(0, 1) = static_cast<RefT<T>>(s);
        x(1, 0) = -Conj(static_cast<RefT<T>>(s));
        x(1, 1) = c;
        return x;
      };
      const Dense<T> u = rot(csu, snu), v = rot(csv, snv), q = rot(csq, snq);
      EXPECT_LT(OrthogonalityRatio(u), kLapackThreshold);
      EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);
      Dense<T> am(2, 2), bm(2, 2);
      am(0, 0) = a1;
      am(1, 1) = a3;
      bm(0, 0) = b1;
      bm(1, 1) = b3;
      if (upper) {
        am(0, 1) = static_cast<RefT<T>>(a2);
        bm(0, 1) = static_cast<RefT<T>>(b2);
      } else {
        am(1, 0) = static_cast<RefT<T>>(a2);
        bm(1, 0) = static_cast<RefT<T>>(b2);
      }
      const Dense<T> ua = Mul(Mul(Adjoint(u), am), q);
      const Dense<T> vb = Mul(Mul(Adjoint(v), bm), q);
      // UPPER: the (1,2) elements vanish; lower: the (2,1) elements.
      const double tol = 100 * LapackEps<T>();
      if (upper) {
        EXPECT_LE(std::abs(ua(0, 1)), tol * (Norm1(am) + 1));
        EXPECT_LE(std::abs(vb(0, 1)), tol * (Norm1(bm) + 1));
      } else {
        EXPECT_LE(std::abs(ua(1, 0)), tol * (Norm1(am) + 1));
        EXPECT_LE(std::abs(vb(1, 0)), tol * (Norm1(bm) + 1));
      }
    }
  }
}

TYPED_TEST(GgevTest, Getc2Gesc2) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(29);
  for (const hwyblas_int n : {1, 2, 5, 8}) {
    for (const bool singular : {false, true}) {
      const hwyblas_int ld = n + 1;
      std::vector<T> a = rng.Vector(Sz2(ld, n));
      if (singular) {
        for (hwyblas_int i = 0; i < n; ++i) At(a, i, n - 1, ld) = T(0);
      }
      std::vector<T> lu = a;
      std::vector<hwyblas_int> ipiv(Sz(n), -1), jpiv(Sz(n), -1);
      hwyblas_int info = -99;
      CALL_RC(T, getc2, getc2, &n, P(lu.data()), &ld, ipiv.data(), jpiv.data(),
              &info);
      if (singular) {
        EXPECT_EQ(info, n);
        continue;
      }
      EXPECT_EQ(info, 0);
      const std::vector<T> rhs0 = rng.Vector(Sz(n));
      std::vector<T> x = rhs0;
      R scale = -1;
      CALL_RC(T, gesc2, gesc2, &n, P(lu.data()), &ld, P(x.data()), ipiv.data(),
              jpiv.data(), &scale);
      EXPECT_GT(scale, 0);
      // A * x = scale * rhs.
      double resid = 0, xn = 0;
      for (hwyblas_int i = 0; i < n; ++i) {
        C64 r = -static_cast<double>(scale) * static_cast<C64>(rhs0[i]);
        for (hwyblas_int j = 0; j < n; ++j) {
          r += static_cast<C64>(At(a, i, j, ld)) * static_cast<C64>(x[j]);
        }
        resid += std::abs(r);
        xn += std::abs(static_cast<C64>(x[i]));
      }
      EXPECT_LT(resid / (Norm1(a, n, n, ld) * xn) / n / LapackEps<T>(),
                kLapackThreshold);
    }
  }
}

// xORGR2 / xUNGR2: the Q of an RQ factorization (xGERQ2), A = R * Q.
TYPED_TEST(GgevTest, Ungr2) {
  using T = TypeParam;
  Random<T> rng(30);
  for (const auto& mn : std::vector<std::pair<hwyblas_int, hwyblas_int>>{
           {0, 0}, {1, 1}, {3, 5}, {4, 4}, {5, 8}}) {
    const hwyblas_int m = mn.first, n = mn.second, ld = n + 1;
    SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n);
    const std::vector<T> a = rng.Vector(Sz2(ld, n));
    std::vector<T> rq = a, tau(Sz(m)), work(Sz(n));
    hwyblas_int info = -99;
    CALL_RC(T, gerq2, gerq2, &m, &n, P(rq.data()), &ld, P(tau.data()),
            P(work.data()), &info);
    ASSERT_EQ(info, 0);
    std::vector<T> q = rq;
    CALL_RC(T, orgr2, ungr2, &m, &n, &m, P(q.data()), &ld, P(tau.data()),
            P(work.data()), &info);
    ASSERT_EQ(info, 0);
    if (m == 0) continue;
    // Orthonormal rows.
    EXPECT_LT(OrthogonalityRatio(Adjoint(ToDense(q, m, n, ld))),
              kLapackThreshold);
    // R in RQ(1:m, n-m+1:n), upper triangular.
    Dense<T> r(static_cast<size_t>(m), static_cast<size_t>(m));
    for (hwyblas_int j = 0; j < m; ++j) {
      for (hwyblas_int i = 0; i <= j; ++i) {
        r(i, j) = static_cast<RefT<T>>(At(rq, i, n - m + j, ld));
      }
    }
    EXPECT_LT(
        ResidualRatio(ToDense(a, m, n, ld), Mul(r, ToDense(q, m, n, ld)), n),
        kLapackThreshold);
    // K < M: the last K reflectors, the first M-K rows of the identity's
    // pattern for the others.
    std::vector<T> q2 = rq;
    const hwyblas_int k = m - 1;
    CALL_RC(T, orgr2, ungr2, &m, &n, &k, P(q2.data()), &ld, P(tau.data() + 1),
            P(work.data()), &info);
    ASSERT_EQ(info, 0);
    EXPECT_LT(OrthogonalityRatio(Adjoint(ToDense(q2, m, n, ld))),
              kLapackThreshold);
  }
}

TYPED_TEST(GgevRealTest, Lag2Lagv2) {
  using R = TypeParam;
  Random<R> rng(31);
  const R safmin = std::numeric_limits<R>::min();
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<R> a = rng.Vector(4), b = rng.Vector(4);
    b[1] = 0;  // B upper triangular (column-major: b[1] = B(2,1))
    const hwyblas_int ld = 2;
    R scale1, scale2, wr1, wr2, wi;
    const R sm = safmin * 100;
    HWYBLAS_CALL_REAL(R, lag2, a.data(), &ld, b.data(), &ld, &sm, &scale1,
                      &scale2, &wr1, &wr2, &wi);
    // det(A - lambda B) = 0 for lambda = (wr + i wi) / scale.
    const auto det = [&](C64 lambda) {
      const C64 m11 = double(a[0]) - lambda * double(b[0]);
      const C64 m12 = double(a[2]) - lambda * double(b[2]);
      const C64 m21 = double(a[1]);
      const C64 m22 = double(a[3]) - lambda * double(b[3]);
      return std::abs(m11 * m22 - m12 * m21);
    };
    const double tol = 1e3 * LapackEps<R>();
    const C64 l1(wr1 / scale1, wi / scale1);
    const C64 l2(wr2 / scale2, -wi / scale2);
    for (const C64 l : {l1, l2}) {
      EXPECT_LE(det(l), tol * (1 + std::abs(l)) * (1 + std::abs(l))) << l;
    }
    // xLAGV2: (A, B) := Ql**T (A, B) Qr in generalized Schur form.
    std::vector<R> as = a, bs = b, ar(2), ai(2), be(2);
    R csl, snl, csr, snr;
    HWYBLAS_CALL_REAL(R, lagv2, as.data(), &ld, bs.data(), &ld, ar.data(),
                      ai.data(), be.data(), &csl, &snl, &csr, &snr);
    Dense<R> ql(2, 2), qr(2, 2);
    ql(0, 0) = csl;
    ql(0, 1) = -snl;
    ql(1, 0) = snl;
    ql(1, 1) = csl;
    qr(0, 0) = csr;
    qr(0, 1) = -snr;
    qr(1, 0) = snr;
    qr(1, 1) = csr;
    EXPECT_LT(ResidualRatio(Mul(Mul(Adjoint(ql), ToDense(a, 2, 2, 2)), qr),
                            ToDense(as, 2, 2, 2), 2),
              kLapackThreshold);
    EXPECT_LT(ResidualRatio(Mul(Mul(Adjoint(ql), ToDense(b, 2, 2, 2)), qr),
                            ToDense(bs, 2, 2, 2), 2),
              kLapackThreshold);
    EXPECT_EQ(bs[1], 0);
    if (ai[0] != 0) {
      // A complex pair: B diagonal, A a 2 x 2 block. (LAPACK documents a
      // positive diagonal of B, which the reference xLAGV2 does not
      // ensure either.)
      EXPECT_EQ(bs[2], 0);
      EXPECT_EQ(ai[1], -ai[0]);
    } else {
      EXPECT_EQ(as[1], 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Error exits

TYPED_TEST(GgevTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  const hwyblas_int c = IsComplex<T>::value ? 1 : 0;
  std::vector<T> a(64, T(1)), b(64, T(1)), q(64), z(64);
  std::vector<R> ls(8), rs(8);
  std::vector<C64> al, be;
  std::vector<hwyblas_int> sel(8, 1);
  hwyblas_int ilo, ihi, m, sdim;
  R scale, dif;
  ErrorCapture errors;
  const auto check = [&](const char* name, const std::vector<int>& args) {
    std::vector<std::string> expected;
    expected.reserve(args.size());
    for (const int arg : args) {
      expected.push_back(Name<T>(name) + ":" + std::to_string(arg));
    }
    EXPECT_EQ(errors.Take(), expected) << name;
  };
  EXPECT_EQ(Ggbal<T>('X', 4, a, 4, b, 4, &ilo, &ihi, ls, rs), -1);
  EXPECT_EQ(Ggbal<T>('B', -1, a, 4, b, 4, &ilo, &ihi, ls, rs), -2);
  EXPECT_EQ(Ggbal<T>('B', 4, a, 3, b, 4, &ilo, &ihi, ls, rs), -4);
  EXPECT_EQ(Ggbal<T>('B', 4, a, 4, b, 3, &ilo, &ihi, ls, rs), -6);
  check("GGBAL", {1, 2, 4, 6});
  EXPECT_EQ(Ggbak<T>('X', 'R', 4, 1, 4, ls, rs, 4, a, 4), -1);
  EXPECT_EQ(Ggbak<T>('B', 'X', 4, 1, 4, ls, rs, 4, a, 4), -2);
  EXPECT_EQ(Ggbak<T>('B', 'R', -1, 1, 4, ls, rs, 4, a, 4), -3);
  EXPECT_EQ(Ggbak<T>('B', 'R', 4, 0, 4, ls, rs, 4, a, 4), -4);
  EXPECT_EQ(Ggbak<T>('B', 'R', 0, 2, 0, ls, rs, 4, a, 4), -4);
  EXPECT_EQ(Ggbak<T>('B', 'R', 4, 2, 1, ls, rs, 4, a, 4), -5);
  EXPECT_EQ(Ggbak<T>('B', 'R', 0, 1, 1, ls, rs, 4, a, 4), -5);
  EXPECT_EQ(Ggbak<T>('B', 'R', 4, 1, 4, ls, rs, -1, a, 4), -8);
  EXPECT_EQ(Ggbak<T>('B', 'R', 4, 1, 4, ls, rs, 4, a, 3), -10);
  check("GGBAK", {1, 2, 3, 4, 4, 5, 5, 8, 10});
  EXPECT_EQ(Gghrd<T>('X', 'N', 4, 1, 4, a, 4, b, 4, q, 4, z, 4), -1);
  EXPECT_EQ(Gghrd<T>('N', 'X', 4, 1, 4, a, 4, b, 4, q, 4, z, 4), -2);
  EXPECT_EQ(Gghrd<T>('N', 'N', -1, 1, 4, a, 4, b, 4, q, 4, z, 4), -3);
  EXPECT_EQ(Gghrd<T>('N', 'N', 4, 0, 4, a, 4, b, 4, q, 4, z, 4), -4);
  EXPECT_EQ(Gghrd<T>('N', 'N', 4, 1, 5, a, 4, b, 4, q, 4, z, 4), -5);
  EXPECT_EQ(Gghrd<T>('N', 'N', 4, 1, 4, a, 3, b, 4, q, 4, z, 4), -7);
  EXPECT_EQ(Gghrd<T>('N', 'N', 4, 1, 4, a, 4, b, 3, q, 4, z, 4), -9);
  EXPECT_EQ(Gghrd<T>('V', 'N', 4, 1, 4, a, 4, b, 4, q, 3, z, 4), -11);
  EXPECT_EQ(Gghrd<T>('N', 'I', 4, 1, 4, a, 4, b, 4, q, 4, z, 3), -13);
  check("GGHRD", {1, 2, 3, 4, 5, 7, 9, 11, 13});
  EXPECT_EQ(Hgeqz<T>('X', 'N', 'N', 4, 1, 4, a, 4, b, 4, al, be, q, 4, z, 4, 4),
            -1);
  EXPECT_EQ(Hgeqz<T>('E', 'X', 'N', 4, 1, 4, a, 4, b, 4, al, be, q, 4, z, 4, 4),
            -2);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'X', 4, 1, 4, a, 4, b, 4, al, be, q, 4, z, 4, 4),
            -3);
  EXPECT_EQ(
      Hgeqz<T>('E', 'N', 'N', -1, 1, 4, a, 4, b, 4, al, be, q, 4, z, 4, 4), -4);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'N', 4, 0, 4, a, 4, b, 4, al, be, q, 4, z, 4, 4),
            -5);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'N', 4, 1, 5, a, 4, b, 4, al, be, q, 4, z, 4, 4),
            -6);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'N', 4, 1, 4, a, 3, b, 4, al, be, q, 4, z, 4, 4),
            -8);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'N', 4, 1, 4, a, 4, b, 3, al, be, q, 4, z, 4, 4),
            -10);
  EXPECT_EQ(Hgeqz<T>('E', 'I', 'N', 4, 1, 4, a, 4, b, 4, al, be, q, 3, z, 4, 4),
            -15 + c);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'V', 4, 1, 4, a, 4, b, 4, al, be, q, 4, z, 3, 4),
            -17 + c);
  EXPECT_EQ(Hgeqz<T>('E', 'N', 'N', 4, 1, 4, a, 4, b, 4, al, be, q, 4, z, 4, 3),
            -19 + c);
  check("HGEQZ", {1, 2, 3, 4, 5, 6, 8, 10, 15 - c, 17 - c, 19 - c});
  std::vector<T> s = IdentityMatrix<T>(4, 4), p = s;
  EXPECT_EQ(Tgevc<T>('X', 'A', sel, 4, s, 4, p, 4, q, 4, z, 4, 4, &m), -1);
  EXPECT_EQ(Tgevc<T>('R', 'X', sel, 4, s, 4, p, 4, q, 4, z, 4, 4, &m), -2);
  EXPECT_EQ(Tgevc<T>('R', 'A', sel, -1, s, 4, p, 4, q, 4, z, 4, 4, &m), -4);
  EXPECT_EQ(Tgevc<T>('R', 'A', sel, 4, s, 3, p, 4, q, 4, z, 4, 4, &m), -6);
  EXPECT_EQ(Tgevc<T>('R', 'A', sel, 4, s, 4, p, 3, q, 4, z, 4, 4, &m), -8);
  EXPECT_EQ(Tgevc<T>('L', 'A', sel, 4, s, 4, p, 4, q, 3, z, 4, 4, &m), -10);
  EXPECT_EQ(Tgevc<T>('R', 'A', sel, 4, s, 4, p, 4, q, 4, z, 3, 4, &m), -12);
  EXPECT_EQ(Tgevc<T>('R', 'A', sel, 4, s, 4, p, 4, q, 4, z, 4, 3, &m), -13);
  check("TGEVC", {1, 2, 4, 6, 8, 10, 12, 13});
  if constexpr (!IsComplex<T>::value) {
    // Illegal 2-by-2 blocks: consecutive nonzero subdiagonal elements of S;
    // a nonzero P(j,j+1) in a block of P.
    std::vector<T> s2 = s, p2 = p;
    At(s2, 1, 0, 4) = At(s2, 2, 1, 4) = T(1);
    EXPECT_EQ(Tgevc<T>('R', 'A', sel, 4, s2, 4, p, 4, q, 4, z, 4, 4, &m), -5);
    std::vector<T> s3 = s;
    At(s3, 1, 0, 4) = T(1);
    At(p2, 0, 1, 4) = T(1);
    EXPECT_EQ(Tgevc<T>('R', 'A', sel, 4, s3, 4, p2, 4, q, 4, z, 4, 4, &m), -7);
    check("TGEVC", {5, 7});
  }
  hwyblas_int ifst = 1, ilst = 2;
  EXPECT_EQ(Tgexc<T>(true, true, -1, a, 4, b, 4, q, 4, z, 4, &ifst, &ilst), -3);
  EXPECT_EQ(Tgexc<T>(true, true, 4, a, 3, b, 4, q, 4, z, 4, &ifst, &ilst), -5);
  EXPECT_EQ(Tgexc<T>(true, true, 4, a, 4, b, 3, q, 4, z, 4, &ifst, &ilst), -7);
  EXPECT_EQ(Tgexc<T>(true, true, 4, a, 4, b, 4, q, 3, z, 4, &ifst, &ilst), -9);
  EXPECT_EQ(Tgexc<T>(true, true, 4, a, 4, b, 4, q, 4, z, 3, &ifst, &ilst), -11);
  ifst = 5;
  EXPECT_EQ(Tgexc<T>(true, true, 4, a, 4, b, 4, q, 4, z, 4, &ifst, &ilst), -12);
  ifst = 1;
  ilst = 0;
  EXPECT_EQ(Tgexc<T>(true, true, 4, a, 4, b, 4, q, 4, z, 4, &ifst, &ilst), -13);
  check("TGEXC", {3, 5, 7, 9, 11, 12, 13});
  TgsenOut<R> out;
  EXPECT_EQ(Tgsen<T>(6, true, true, sel, 4, a, 4, b, 4, al, be, q, 4, z, 4, out,
                     64, 64),
            -1);
  EXPECT_EQ(Tgsen<T>(0, true, true, sel, -1, a, 4, b, 4, al, be, q, 4, z, 4,
                     out, 64, 64),
            -5);
  EXPECT_EQ(Tgsen<T>(0, true, true, sel, 4, a, 3, b, 4, al, be, q, 4, z, 4, out,
                     64, 64),
            -7);
  EXPECT_EQ(Tgsen<T>(0, true, true, sel, 4, a, 4, b, 3, al, be, q, 4, z, 4, out,
                     64, 64),
            -9);
  EXPECT_EQ(Tgsen<T>(0, true, true, sel, 4, a, 4, b, 4, al, be, q, 3, z, 4, out,
                     64, 64),
            -14 + c);
  EXPECT_EQ(Tgsen<T>(0, true, true, sel, 4, a, 4, b, 4, al, be, q, 4, z, 3, out,
                     64, 64),
            -16 + c);
  // M = 2 (two of the four eigenvalues of the triangular pair selected).
  std::vector<T> st = IdentityMatrix<T>(4, 4), pt = st;
  std::vector<hwyblas_int> sel2 = {1, 0, 1, 0};
  EXPECT_EQ(Tgsen<T>(3, true, true, sel2, 4, st, 4, pt, 4, al, be, q, 4, z, 4,
                     out, 15, 64),
            -22 + c);
  EXPECT_EQ(Tgsen<T>(3, true, true, sel2, 4, st, 4, pt, 4, al, be, q, 4, z, 4,
                     out, 64, 7),
            -24 + c);
  check("TGSEN", {1, 5, 7, 9, 14 - c, 16 - c, 22 - c, 24 - c});
  EXPECT_EQ(Tgsyl<T>('X', 0, 4, 4, a, 4, b, 4, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -1);
  EXPECT_EQ(Tgsyl<T>('N', 5, 4, 4, a, 4, b, 4, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -2);
  EXPECT_EQ(Tgsyl<T>('N', 0, 0, 4, a, 4, b, 4, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -3);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 0, a, 4, b, 4, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -4);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 4, a, 3, b, 4, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -6);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 4, a, 4, b, 3, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -8);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 4, a, 4, b, 4, q, 3, s, 4, p, 4, z, 4, &scale,
                     &dif, 1),
            -10);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 4, a, 4, b, 4, q, 4, s, 3, p, 4, z, 4, &scale,
                     &dif, 1),
            -12);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 4, a, 4, b, 4, q, 4, s, 4, p, 3, z, 4, &scale,
                     &dif, 1),
            -14);
  EXPECT_EQ(Tgsyl<T>('N', 0, 4, 4, a, 4, b, 4, q, 4, s, 4, p, 4, z, 3, &scale,
                     &dif, 1),
            -16);
  EXPECT_EQ(Tgsyl<T>('N', 1, 4, 4, a, 4, b, 4, q, 4, s, 4, p, 4, z, 4, &scale,
                     &dif, 31),
            -20);
  check("TGSYL", {1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20});
  EXPECT_EQ(
      Gges<T>('X', 'N', 'N', 4, a, 4, b, 4, &sdim, al, be, q, 4, z, 4, 64), -1);
  EXPECT_EQ(
      Gges<T>('N', 'X', 'N', 4, a, 4, b, 4, &sdim, al, be, q, 4, z, 4, 64), -2);
  EXPECT_EQ(
      Gges<T>('N', 'N', 'X', 4, a, 4, b, 4, &sdim, al, be, q, 4, z, 4, 64), -3);
  EXPECT_EQ(
      Gges<T>('N', 'N', 'N', -1, a, 4, b, 4, &sdim, al, be, q, 4, z, 4, 64),
      -5);
  EXPECT_EQ(
      Gges<T>('N', 'N', 'N', 4, a, 3, b, 4, &sdim, al, be, q, 4, z, 4, 64), -7);
  EXPECT_EQ(
      Gges<T>('N', 'N', 'N', 4, a, 4, b, 3, &sdim, al, be, q, 4, z, 4, 64), -9);
  EXPECT_EQ(
      Gges<T>('V', 'N', 'N', 4, a, 4, b, 4, &sdim, al, be, q, 3, z, 4, 64),
      -15 + c);
  EXPECT_EQ(
      Gges<T>('N', 'V', 'N', 4, a, 4, b, 4, &sdim, al, be, q, 4, z, 3, 64),
      -17 + c);
  EXPECT_EQ(Gges<T>('N', 'N', 'N', 4, a, 4, b, 4, &sdim, al, be, q, 4, z, 4,
                    c ? 7 : 39),
            -19 + c);
  check("GGES", {1, 2, 3, 5, 7, 9, 15 - c, 17 - c, 19 - c});
  EXPECT_EQ(Ggev<T>('X', 'N', 4, a, 4, b, 4, al, be, q, 4, z, 4, 64), -1);
  EXPECT_EQ(Ggev<T>('N', 'X', 4, a, 4, b, 4, al, be, q, 4, z, 4, 64), -2);
  EXPECT_EQ(Ggev<T>('N', 'N', -1, a, 4, b, 4, al, be, q, 4, z, 4, 64), -3);
  EXPECT_EQ(Ggev<T>('N', 'N', 4, a, 3, b, 4, al, be, q, 4, z, 4, 64), -5);
  EXPECT_EQ(Ggev<T>('N', 'N', 4, a, 4, b, 3, al, be, q, 4, z, 4, 64), -7);
  EXPECT_EQ(Ggev<T>('V', 'N', 4, a, 4, b, 4, al, be, q, 3, z, 4, 64), -12 + c);
  EXPECT_EQ(Ggev<T>('N', 'V', 4, a, 4, b, 4, al, be, q, 4, z, 3, 64), -14 + c);
  EXPECT_EQ(Ggev<T>('N', 'N', 4, a, 4, b, 4, al, be, q, 4, z, 4, c ? 7 : 31),
            -16 + c);
  check("GGEV", {1, 2, 3, 5, 7, 12 - c, 14 - c, 16 - c});
  Ggsvd3Out<R> gout;
  EXPECT_EQ(
      Ggsvd3<T>('X', 'N', 'N', 4, 4, 4, a, 4, b, 4, q, 4, z, 4, s, 4, gout, 64),
      -1);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'X', 'N', 4, 4, 4, a, 4, b, 4, q, 4, z, 4, s, 4, gout, 64),
      -2);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'N', 'X', 4, 4, 4, a, 4, b, 4, q, 4, z, 4, s, 4, gout, 64),
      -3);
  EXPECT_EQ(Ggsvd3<T>('N', 'N', 'N', -1, 4, 4, a, 4, b, 4, q, 4, z, 4, s, 4,
                      gout, 64),
            -4);
  EXPECT_EQ(Ggsvd3<T>('N', 'N', 'N', 4, -1, 4, a, 4, b, 4, q, 4, z, 4, s, 4,
                      gout, 64),
            -5);
  EXPECT_EQ(Ggsvd3<T>('N', 'N', 'N', 4, 4, -1, a, 4, b, 4, q, 4, z, 4, s, 4,
                      gout, 64),
            -6);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'N', 'N', 4, 4, 4, a, 3, b, 4, q, 4, z, 4, s, 4, gout, 64),
      -10);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'N', 'N', 4, 4, 4, a, 4, b, 3, q, 4, z, 4, s, 4, gout, 64),
      -12);
  EXPECT_EQ(
      Ggsvd3<T>('U', 'N', 'N', 4, 4, 4, a, 4, b, 4, q, 3, z, 4, s, 4, gout, 64),
      -16);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'V', 'N', 4, 4, 4, a, 4, b, 4, q, 4, z, 3, s, 4, gout, 64),
      -18);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'N', 'Q', 4, 4, 4, a, 4, b, 4, q, 4, z, 4, s, 3, gout, 64),
      -20);
  EXPECT_EQ(
      Ggsvd3<T>('N', 'N', 'N', 4, 4, 4, a, 4, b, 4, q, 4, z, 4, s, 4, gout, 0),
      -24);
  check("GGSVD3", {1, 2, 3, 4, 5, 6, 10, 12, 16, 18, 20, 24});
  hwyblas_int k, l;
  EXPECT_EQ(Ggsvp3<T>('X', 'N', 'N', 4, 4, 4, a, 4, b, 4, R(0), R(0), &k, &l, q,
                      4, z, 4, s, 4, 64),
            -1);
  EXPECT_EQ(Ggsvp3<T>('N', 'N', 'N', 4, 4, 4, a, 3, b, 4, R(0), R(0), &k, &l, q,
                      4, z, 4, s, 4, 64),
            -8);
  EXPECT_EQ(Ggsvp3<T>('N', 'N', 'N', 4, 4, 4, a, 4, b, 3, R(0), R(0), &k, &l, q,
                      4, z, 4, s, 4, 64),
            -10);
  EXPECT_EQ(Ggsvp3<T>('N', 'N', 'Q', 4, 4, 4, a, 4, b, 4, R(0), R(0), &k, &l, q,
                      4, z, 4, s, 3, 64),
            -20);
  EXPECT_EQ(Ggsvp3<T>('N', 'N', 'N', 4, 4, 4, a, 4, b, 4, R(0), R(0), &k, &l, q,
                      4, z, 4, s, 4, 0),
            -24);
  check("GGSVP3", {1, 8, 10, 20, 24});
}

// ---------------------------------------------------------------------------
// LAPACKE

// LAPACKE_?ggev / ?gges / ?hgeqz / ?tgsen with the eigenvalues as complex
// numbers.
template <typename T>
lapack_int LapackeGgev(int layout, char jobvl, char jobvr, lapack_int n, T* a,
                       lapack_int lda, T* b, lapack_int ldb,
                       std::vector<C64>& alpha, std::vector<C64>& beta, T* vl,
                       lapack_int ldvl, T* vr, lapack_int ldvr) {
  using R = RealT<T>;
  alpha.assign(Sz(n), C64());
  beta.assign(Sz(n), C64());
  lapack_int info;
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n)), be(Sz(n));
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      info = LAPACKE_cggev(layout, jobvl, jobvr, n, a, lda, b, ldb, al.data(),
                           be.data(), vl, ldvl, vr, ldvr);
    } else {
      info = LAPACKE_zggev(layout, jobvl, jobvr, n, a, lda, b, ldb, al.data(),
                           be.data(), vl, ldvl, vr, ldvr);
    }
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n)), ai(Sz(n)), be(Sz(n));
    info =
        LAPACKE_CALL_REAL(T, ggev, layout, jobvl, jobvr, n, a, lda, b, ldb,
                          ar.data(), ai.data(), be.data(), vl, ldvl, vr, ldvr);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  return info;
}

template <typename R>
lapack_logical NegativeRealC3(const R* ar, const R* /*ai*/, const R* b) {
  return *ar * *b < 0 ? 1 : 0;
}
template <typename T>
lapack_logical NegativeRealC2(const T* a, const T* b) {
  return (*a * std::conj(*b)).real() < 0 ? 1 : 0;
}

template <typename T>
lapack_int LapackeGges(int layout, char jobvsl, char jobvsr, char sort,
                       lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                       lapack_int* sdim, std::vector<C64>& alpha,
                       std::vector<C64>& beta, T* vsl, lapack_int ldvsl, T* vsr,
                       lapack_int ldvsr) {
  using R = RealT<T>;
  alpha.assign(Sz(n), C64());
  beta.assign(Sz(n), C64());
  lapack_int info;
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n)), be(Sz(n));
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      info = LAPACKE_cgges(layout, jobvsl, jobvsr, sort, &NegativeRealC2<T>, n,
                           a, lda, b, ldb, sdim, al.data(), be.data(), vsl,
                           ldvsl, vsr, ldvsr);
    } else {
      info = LAPACKE_zgges(layout, jobvsl, jobvsr, sort, &NegativeRealC2<T>, n,
                           a, lda, b, ldb, sdim, al.data(), be.data(), vsl,
                           ldvsl, vsr, ldvsr);
    }
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n)), ai(Sz(n)), be(Sz(n));
    info = LAPACKE_CALL_REAL(
        T, gges, layout, jobvsl, jobvsr, sort, &NegativeRealC3<R>, n, a, lda, b,
        ldb, sdim, ar.data(), ai.data(), be.data(), vsl, ldvsl, vsr, ldvsr);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  return info;
}

template <typename T>
lapack_int LapackeHgeqz(int layout, char job, char compq, char compz,
                        lapack_int n, lapack_int ilo, lapack_int ihi, T* h,
                        lapack_int ldh, T* t, lapack_int ldt,
                        std::vector<C64>& alpha, std::vector<C64>& beta, T* q,
                        lapack_int ldq, T* z, lapack_int ldz) {
  using R = RealT<T>;
  alpha.assign(Sz(n), C64());
  beta.assign(Sz(n), C64());
  lapack_int info;
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n)), be(Sz(n));
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      info = LAPACKE_chgeqz(layout, job, compq, compz, n, ilo, ihi, h, ldh, t,
                            ldt, al.data(), be.data(), q, ldq, z, ldz);
    } else {
      info = LAPACKE_zhgeqz(layout, job, compq, compz, n, ilo, ihi, h, ldh, t,
                            ldt, al.data(), be.data(), q, ldq, z, ldz);
    }
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]);
      beta[i] = static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n)), ai(Sz(n)), be(Sz(n));
    info = LAPACKE_CALL_REAL(T, hgeqz, layout, job, compq, compz, n, ilo, ihi,
                             h, ldh, t, ldt, ar.data(), ai.data(), be.data(), q,
                             ldq, z, ldz);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]);
      beta[i] = C64(be[i], 0);
    }
  }
  return info;
}

template <typename T>
lapack_int LapackeTgsen(int layout, lapack_int ijob,
                        const lapack_logical* select, lapack_int n, T* a,
                        lapack_int lda, T* b, lapack_int ldb,
                        std::vector<C64>& alpha, T* q, lapack_int ldq, T* z,
                        lapack_int ldz, lapack_int* m, RealT<T>* pl,
                        RealT<T>* pr, RealT<T>* dif) {
  using R = RealT<T>;
  alpha.assign(Sz(n), C64());
  lapack_int info;
  if constexpr (IsComplex<T>::value) {
    std::vector<T> al(Sz(n)), be(Sz(n));
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      info =
          LAPACKE_ctgsen(layout, ijob, 1, 1, select, n, a, lda, b, ldb,
                         al.data(), be.data(), q, ldq, z, ldz, m, pl, pr, dif);
    } else {
      info =
          LAPACKE_ztgsen(layout, ijob, 1, 1, select, n, a, lda, b, ldb,
                         al.data(), be.data(), q, ldq, z, ldz, m, pl, pr, dif);
    }
    for (size_t i = 0; i < al.size(); ++i) {
      alpha[i] = static_cast<C64>(al[i]) / static_cast<C64>(be[i]);
    }
  } else {
    std::vector<R> ar(Sz(n)), ai(Sz(n)), be(Sz(n));
    info = LAPACKE_CALL_REAL(T, tgsen, layout, ijob, 1, 1, select, n, a, lda, b,
                             ldb, ar.data(), ai.data(), be.data(), q, ldq, z,
                             ldz, m, pl, pr, dif);
    for (size_t i = 0; i < ar.size(); ++i) {
      alpha[i] = C64(ar[i], ai[i]) / C64(be[i], 0);
    }
  }
  return info;
}

// The row-major results of a LAPACKE function equal the column-major ones.
TYPED_TEST(GgevTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(41);
  const lapack_int n = 7, ld = 9;
  std::vector<T> a0, b0;
  TestPair(Kind::kGraded, n, ld, rng, a0, b0);
  const std::vector<T> ar0 = ToRowMajor(a0, n, n, ld, ld),
                       br0 = ToRowMajor(b0, n, n, ld, ld);
  const size_t size = static_cast<size_t>(ld) * ld;
  // ?ggbal, ?ggbak.
  for (const char job : {'N', 'P', 'S', 'B'}) {
    SCOPED_TRACE(::testing::Message() << "ggbal " << job);
    std::vector<T> ac = a0, bc = b0, ar = ar0, br = br0;
    lapack_int iloc, ihic, ilor, ihir;
    std::vector<R> lc(n), rc(n), lr(n), rr(n);
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, ggbal, ggbal, LAPACK_COL_MAJOR, job, n, ac.data(),
                        ld, bc.data(), ld, &iloc, &ihic, lc.data(), rc.data()),
        0);
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, ggbal, ggbal, LAPACK_ROW_MAJOR, job, n, ar.data(),
                        ld, br.data(), ld, &ilor, &ihir, lr.data(), rr.data()),
        0);
    EXPECT_EQ(iloc, ilor);
    EXPECT_EQ(ihic, ihir);
    EXPECT_EQ(lc, lr);
    EXPECT_EQ(rc, rr);
    ExpectSameRowMajor(ar, ac, n, n, ld, ld);
    ExpectSameRowMajor(br, bc, n, n, ld, ld);
    const lapack_int mv = 3;
    std::vector<T> vc = rng.Vector(size);
    std::vector<T> vr = ToRowMajor(vc, n, mv, ld, ld);
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, ggbak, ggbak, LAPACK_COL_MAJOR, job, 'L', n, iloc,
                        ihic, lc.data(), rc.data(), mv, vc.data(), ld),
        0);
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, ggbak, ggbak, LAPACK_ROW_MAJOR, job, 'L', n, iloc,
                        ihic, lc.data(), rc.data(), mv, vr.data(), ld),
        0);
    ExpectSameRowMajor(vr, vc, n, mv, ld, ld);
  }
  // ?gghrd, ?hgeqz, ?tgevc on (A, triangular B).
  std::vector<T> bt0 = b0;
  for (lapack_int j = 0; j < n; ++j) {
    for (lapack_int i = j + 1; i < n; ++i) At(bt0, i, j, ld) = T(0);
  }
  const std::vector<T> btr0 = ToRowMajor(bt0, n, n, ld, ld);
  std::vector<T> hc = a0, tc = bt0, qc(size), zc(size);
  std::vector<T> hr = ar0, tr = btr0, qr(size), zr(size);
  ASSERT_EQ(LAPACKE_CALL_RC(T, gghrd, gghrd, LAPACK_COL_MAJOR, 'I', 'I', n, 1,
                            n, hc.data(), ld, tc.data(), ld, qc.data(), ld,
                            zc.data(), ld),
            0);
  ASSERT_EQ(LAPACKE_CALL_RC(T, gghrd, gghrd, LAPACK_ROW_MAJOR, 'I', 'I', n, 1,
                            n, hr.data(), ld, tr.data(), ld, qr.data(), ld,
                            zr.data(), ld),
            0);
  ExpectSameRowMajor(hr, hc, n, n, ld, ld);
  ExpectSameRowMajor(tr, tc, n, n, ld, ld);
  ExpectSameRowMajor(qr, qc, n, n, ld, ld);
  ExpectSameRowMajor(zr, zc, n, n, ld, ld);
  std::vector<C64> alc, bec, alr, ber;
  ASSERT_EQ(
      LapackeHgeqz<T>(LAPACK_COL_MAJOR, 'S', 'V', 'V', n, 1, n, hc.data(), ld,
                      tc.data(), ld, alc, bec, qc.data(), ld, zc.data(), ld),
      0);
  ASSERT_EQ(
      LapackeHgeqz<T>(LAPACK_ROW_MAJOR, 'S', 'V', 'V', n, 1, n, hr.data(), ld,
                      tr.data(), ld, alr, ber, qr.data(), ld, zr.data(), ld),
      0);
  EXPECT_EQ(alc, alr);
  EXPECT_EQ(bec, ber);
  ExpectSameRowMajor(hr, hc, n, n, ld, ld);
  ExpectSameRowMajor(tr, tc, n, n, ld, ld);
  ExpectSameRowMajor(qr, qc, n, n, ld, ld);
  ExpectSameRowMajor(zr, zc, n, n, ld, ld);
  {
    std::vector<lapack_logical> sel(n, 1);
    std::vector<T> vlc = qc, vrc = zc, vlr = qr, vrr = zr;
    lapack_int mc = -1, mr = -1;
    ASSERT_EQ(LAPACKE_CALL_RC(T, tgevc, tgevc, LAPACK_COL_MAJOR, 'B', 'B',
                              sel.data(), n, hc.data(), ld, tc.data(), ld,
                              vlc.data(), ld, vrc.data(), ld, n, &mc),
              0);
    ASSERT_EQ(LAPACKE_CALL_RC(T, tgevc, tgevc, LAPACK_ROW_MAJOR, 'B', 'B',
                              sel.data(), n, hr.data(), ld, tr.data(), ld,
                              vlr.data(), ld, vrr.data(), ld, n, &mr),
              0);
    EXPECT_EQ(mc, mr);
    ExpectSameRowMajor(vlr, vlc, n, n, ld, ld);
    ExpectSameRowMajor(vrr, vrc, n, n, ld, ld);
  }
  // ?ggev.
  for (const char jobvl : {'V', 'N'}) {
    for (const char jobvr : {'V', 'N'}) {
      SCOPED_TRACE(::testing::Message() << "ggev " << jobvl << jobvr);
      std::vector<T> ac = a0, bc = b0, ar = ar0, br = br0;
      std::vector<T> lc(size), rc(size), lr(size), rr(size);
      std::vector<C64> alpha_c, beta_c, alpha_r, beta_r;
      ASSERT_EQ(LapackeGgev<T>(LAPACK_COL_MAJOR, jobvl, jobvr, n, ac.data(), ld,
                               bc.data(), ld, alpha_c, beta_c, lc.data(), ld,
                               rc.data(), ld),
                0);
      ASSERT_EQ(LapackeGgev<T>(LAPACK_ROW_MAJOR, jobvl, jobvr, n, ar.data(), ld,
                               br.data(), ld, alpha_r, beta_r, lr.data(), ld,
                               rr.data(), ld),
                0);
      EXPECT_EQ(alpha_c, alpha_r);
      EXPECT_EQ(beta_c, beta_r);
      if (jobvl == 'V') ExpectSameRowMajor(lr, lc, n, n, ld, ld);
      if (jobvr == 'V') ExpectSameRowMajor(rr, rc, n, n, ld, ld);
      // The LAPACK routine.
      std::vector<T> af = a0, bf = b0, vl(size), vr(size);
      std::vector<C64> alpha, beta;
      ASSERT_EQ(Ggev<T>(jobvl, jobvr, n, af, ld, bf, ld, alpha, beta, vl, ld,
                        vr, ld, 64 * n),
                0);
      ExpectSameLambdas<T>(Lambdas(alpha, beta), Lambdas(alpha_c, beta_c));
    }
  }
  // ?gges, then ?tgsen and ?tgexc on its Schur form.
  std::vector<T> sc = a0, pc = b0, sr = ar0, pr = br0, vlc(size), vrc(size),
                 vlr(size), vrr(size);
  {
    lapack_int sdc = -1, sdr = -1;
    std::vector<C64> alpha_c, beta_c, alpha_r, beta_r;
    ASSERT_EQ(LapackeGges<T>(LAPACK_COL_MAJOR, 'V', 'V', 'S', n, sc.data(), ld,
                             pc.data(), ld, &sdc, alpha_c, beta_c, vlc.data(),
                             ld, vrc.data(), ld),
              0);
    ASSERT_EQ(LapackeGges<T>(LAPACK_ROW_MAJOR, 'V', 'V', 'S', n, sr.data(), ld,
                             pr.data(), ld, &sdr, alpha_r, beta_r, vlr.data(),
                             ld, vrr.data(), ld),
              0);
    EXPECT_EQ(sdc, sdr);
    EXPECT_EQ(alpha_c, alpha_r);
    EXPECT_EQ(beta_c, beta_r);
    ExpectSameRowMajor(sr, sc, n, n, ld, ld);
    ExpectSameRowMajor(pr, pc, n, n, ld, ld);
    ExpectSameRowMajor(vlr, vlc, n, n, ld, ld);
    ExpectSameRowMajor(vrr, vrc, n, n, ld, ld);
  }
  for (const lapack_int ijob : {0, 1, 2, 3, 4, 5}) {
    SCOPED_TRACE(::testing::Message() << "tgsen " << ijob);
    std::vector<lapack_logical> sel(n, 0);
    sel[n - 1] = 1;
    sel[n - 3] = 1;
    std::vector<T> s1 = sc, p1 = pc, q1 = vlc, z1 = vrc;
    std::vector<T> s2 = sr, p2 = pr, q2 = vlr, z2 = vrr;
    std::vector<C64> w1, w2;
    lapack_int m1 = -1, m2 = -1;
    R pl1 = 0, pr1 = 0, pl2 = 0, pr2 = 0, dif1[2] = {}, dif2[2] = {};
    ASSERT_EQ(LapackeTgsen<T>(LAPACK_COL_MAJOR, ijob, sel.data(), n, s1.data(),
                              ld, p1.data(), ld, w1, q1.data(), ld, z1.data(),
                              ld, &m1, &pl1, &pr1, dif1),
              0);
    ASSERT_EQ(LapackeTgsen<T>(LAPACK_ROW_MAJOR, ijob, sel.data(), n, s2.data(),
                              ld, p2.data(), ld, w2, q2.data(), ld, z2.data(),
                              ld, &m2, &pl2, &pr2, dif2),
              0);
    EXPECT_EQ(m1, m2);
    EXPECT_EQ(w1, w2);
    EXPECT_EQ(pl1, pl2);
    EXPECT_EQ(pr1, pr2);
    EXPECT_EQ(dif1[0], dif2[0]);
    EXPECT_EQ(dif1[1], dif2[1]);
    ExpectSameRowMajor(s2, s1, n, n, ld, ld);
    ExpectSameRowMajor(p2, p1, n, n, ld, ld);
    ExpectSameRowMajor(q2, q1, n, n, ld, ld);
    ExpectSameRowMajor(z2, z1, n, n, ld, ld);
  }
  {
    std::vector<T> s1 = sc, p1 = pc, q1 = vlc, z1 = vrc;
    std::vector<T> s2 = sr, p2 = pr, q2 = vlr, z2 = vrr;
    lapack_int info1, info2;
    if constexpr (IsComplex<T>::value) {
      info1 = LAPACKE_CALL_RC(T, tgexc, tgexc, LAPACK_COL_MAJOR, 1, 1, n,
                              s1.data(), ld, p1.data(), ld, q1.data(), ld,
                              z1.data(), ld, 1, n);
      info2 = LAPACKE_CALL_RC(T, tgexc, tgexc, LAPACK_ROW_MAJOR, 1, 1, n,
                              s2.data(), ld, p2.data(), ld, q2.data(), ld,
                              z2.data(), ld, 1, n);
    } else {
      lapack_int f1 = 1, l1 = n, f2 = 1, l2 = n;
      info1 = LAPACKE_CALL_REAL(T, tgexc, LAPACK_COL_MAJOR, 1, 1, n, s1.data(),
                                ld, p1.data(), ld, q1.data(), ld, z1.data(), ld,
                                &f1, &l1);
      info2 = LAPACKE_CALL_REAL(T, tgexc, LAPACK_ROW_MAJOR, 1, 1, n, s2.data(),
                                ld, p2.data(), ld, q2.data(), ld, z2.data(), ld,
                                &f2, &l2);
      EXPECT_EQ(l1, l2);
    }
    EXPECT_EQ(info1, 0);
    EXPECT_EQ(info2, 0);
    ExpectSameRowMajor(s2, s1, n, n, ld, ld);
    ExpectSameRowMajor(p2, p1, n, n, ld, ld);
    ExpectSameRowMajor(q2, q1, n, n, ld, ld);
    ExpectSameRowMajor(z2, z1, n, n, ld, ld);
  }
  // ?tgsyl with (A, D) = the Schur form (S, P) and (B, E) = (S + 5 P, P)
  // in its leading 4 x 4 part (eigenvalues shifted by 5).
  std::vector<T> bc5 = sc, br5 = sr;
  for (size_t i = 0; i < bc5.size(); ++i) bc5[i] += T(R(5)) * pc[i];
  for (size_t i = 0; i < br5.size(); ++i) br5[i] += T(R(5)) * pr[i];
  for (const lapack_int ijob : {0, 1, 3}) {
    SCOPED_TRACE(::testing::Message() << "tgsyl " << ijob);
    const lapack_int mm = n, nn = 4;
    const std::vector<T> c0 = rng.Vector(size), f0 = rng.Vector(size);
    std::vector<T> c1 = c0, f1 = f0;
    std::vector<T> c2 = ToRowMajor(c0, mm, nn, ld, ld),
                   f2 = ToRowMajor(f0, mm, nn, ld, ld);
    R scale1 = 0, dif1 = 0, scale2 = 0, dif2 = 0;
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, tgsyl, tgsyl, LAPACK_COL_MAJOR, 'N', ijob, mm, nn,
                        sc.data(), ld, bc5.data(), ld, c1.data(), ld, pc.data(),
                        ld, pc.data(), ld, f1.data(), ld, &scale1, &dif1),
        0);
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, tgsyl, tgsyl, LAPACK_ROW_MAJOR, 'N', ijob, mm, nn,
                        sr.data(), ld, br5.data(), ld, c2.data(), ld, pr.data(),
                        ld, pr.data(), ld, f2.data(), ld, &scale2, &dif2),
        0);
    EXPECT_EQ(scale1, scale2);
    EXPECT_EQ(dif1, dif2);
    ExpectSameRowMajor(c2, c1, mm, nn, ld, ld);
    ExpectSameRowMajor(f2, f1, mm, nn, ld, ld);
  }
  // ?ggsvd3, ?ggsvp3, ?tgsja (5 x 7 and 4 x 7).
  {
    const lapack_int m = 5, p = 4, nn = 7;
    const std::vector<T> ga = rng.Vector(size), gb = rng.Vector(size);
    std::vector<T> a1 = ga, b1 = gb, u1(size), vb1(size), q1(size);
    std::vector<T> a2 = ToRowMajor(ga, m, nn, ld, ld),
                   b2 = ToRowMajor(gb, p, nn, ld, ld), u2(size), v2(size),
                   q2(size);
    lapack_int k1, l1, k2, l2;
    std::vector<R> al1(nn), be1(nn), al2(nn), be2(nn);
    std::vector<lapack_int> iw1(nn), iw2(nn);
    ASSERT_EQ(LAPACKE_CALL_RC(T, ggsvd3, ggsvd3, LAPACK_COL_MAJOR, 'U', 'V',
                              'Q', m, nn, p, &k1, &l1, a1.data(), ld, b1.data(),
                              ld, al1.data(), be1.data(), u1.data(), ld,
                              vb1.data(), ld, q1.data(), ld, iw1.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL_RC(T, ggsvd3, ggsvd3, LAPACK_ROW_MAJOR, 'U', 'V',
                              'Q', m, nn, p, &k2, &l2, a2.data(), ld, b2.data(),
                              ld, al2.data(), be2.data(), u2.data(), ld,
                              v2.data(), ld, q2.data(), ld, iw2.data()),
              0);
    EXPECT_EQ(k1, k2);
    EXPECT_EQ(l1, l2);
    EXPECT_EQ(al1, al2);
    EXPECT_EQ(be1, be2);
    EXPECT_EQ(iw1, iw2);
    ExpectSameRowMajor(a2, a1, m, nn, ld, ld);
    ExpectSameRowMajor(b2, b1, p, nn, ld, ld);
    ExpectSameRowMajor(u2, u1, m, m, ld, ld);
    ExpectSameRowMajor(v2, vb1, p, p, ld, ld);
    ExpectSameRowMajor(q2, q1, nn, nn, ld, ld);
    // ?ggsvp3 then ?tgsja ('U', 'V', 'Q': update the matrices).
    const R tol = R(1e-3);
    std::vector<T> a3 = ga, b3 = gb, u3(size), v3(size), q3(size);
    std::vector<T> a4 = ToRowMajor(ga, m, nn, ld, ld),
                   b4 = ToRowMajor(gb, p, nn, ld, ld), u4(size), v4(size),
                   q4(size);
    lapack_int k3, l3, k4, l4;
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, ggsvp3, ggsvp3, LAPACK_COL_MAJOR, 'U', 'V', 'Q', m,
                        p, nn, a3.data(), ld, b3.data(), ld, tol, tol, &k3, &l3,
                        u3.data(), ld, v3.data(), ld, q3.data(), ld),
        0);
    ASSERT_EQ(
        LAPACKE_CALL_RC(T, ggsvp3, ggsvp3, LAPACK_ROW_MAJOR, 'U', 'V', 'Q', m,
                        p, nn, a4.data(), ld, b4.data(), ld, tol, tol, &k4, &l4,
                        u4.data(), ld, v4.data(), ld, q4.data(), ld),
        0);
    EXPECT_EQ(k3, k4);
    EXPECT_EQ(l3, l4);
    ExpectSameRowMajor(a4, a3, m, nn, ld, ld);
    ExpectSameRowMajor(b4, b3, p, nn, ld, ld);
    ExpectSameRowMajor(u4, u3, m, m, ld, ld);
    ExpectSameRowMajor(q4, q3, nn, nn, ld, ld);
    std::vector<R> al3(nn), be3(nn), al4(nn), be4(nn);
    lapack_int nc3 = -1, nc4 = -1;
    ASSERT_EQ(LAPACKE_CALL_RC(T, tgsja, tgsja, LAPACK_COL_MAJOR, 'U', 'V', 'Q',
                              m, p, nn, k3, l3, a3.data(), ld, b3.data(), ld,
                              tol, tol, al3.data(), be3.data(), u3.data(), ld,
                              v3.data(), ld, q3.data(), ld, &nc3),
              0);
    ASSERT_EQ(LAPACKE_CALL_RC(T, tgsja, tgsja, LAPACK_ROW_MAJOR, 'U', 'V', 'Q',
                              m, p, nn, k4, l4, a4.data(), ld, b4.data(), ld,
                              tol, tol, al4.data(), be4.data(), u4.data(), ld,
                              v4.data(), ld, q4.data(), ld, &nc4),
              0);
    EXPECT_EQ(nc3, nc4);
    EXPECT_EQ(al3, al4);
    EXPECT_EQ(be3, be4);
    ExpectSameRowMajor(a4, a3, m, nn, ld, ld);
    ExpectSameRowMajor(u4, u3, m, m, ld, ld);
    ExpectSameRowMajor(v4, v3, p, p, ld, ld);
    ExpectSameRowMajor(q4, q3, nn, nn, ld, ld);
  }
  // ?lapmt.
  {
    const std::vector<T> x0 = rng.Vector(size);
    std::vector<T> x1 = x0, x2 = ToRowMajor(x0, 3, n, ld, ld);
    std::vector<lapack_int> k1 = {2, 1, 4, 3, 6, 7, 5}, k2 = k1;
    ASSERT_EQ(LAPACKE_CALL_RC(T, lapmt, lapmt, LAPACK_COL_MAJOR, 1, 3, n,
                              x1.data(), ld, k1.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL_RC(T, lapmt, lapmt, LAPACK_ROW_MAJOR, 1, 3, n,
                              x2.data(), ld, k2.data()),
              0);
    ExpectSameRowMajor(x2, x1, 3, n, ld, ld);
  }
}

TYPED_TEST(GgevTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  std::vector<T> a(64, T(1)), b(64, T(1)), q(64), z(64);
  std::vector<C64> al, be;
  std::vector<R> ls(8, 1), rs(8, 1);
  lapack_int ilo, ihi, sdim;
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeGgev<T>(0, 'V', 'V', 4, a.data(), 4, b.data(), 4, al, be,
                           q.data(), 4, z.data(), 4),
            -1);
  // Row-major leading dimensions.
  EXPECT_EQ(LapackeGgev<T>(LAPACK_ROW_MAJOR, 'V', 'V', 4, a.data(), 3, b.data(),
                           4, al, be, q.data(), 4, z.data(), 4),
            -6);
  EXPECT_EQ(LapackeGgev<T>(LAPACK_ROW_MAJOR, 'V', 'V', 4, a.data(), 4, b.data(),
                           3, al, be, q.data(), 4, z.data(), 4),
            -8);
  EXPECT_EQ(LapackeGgev<T>(LAPACK_ROW_MAJOR, 'V', 'N', 4, a.data(), 4, b.data(),
                           4, al, be, q.data(), 3, z.data(), 1),
            -13 + c);
  EXPECT_EQ(LapackeGgev<T>(LAPACK_ROW_MAJOR, 'N', 'V', 4, a.data(), 4, b.data(),
                           4, al, be, q.data(), 1, z.data(), 3),
            -15 + c);
  EXPECT_EQ(
      LapackeGges<T>(LAPACK_ROW_MAJOR, 'V', 'V', 'N', 4, a.data(), 4, b.data(),
                     4, &sdim, al, be, q.data(), 3, z.data(), 4),
      -16 + c);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ggbal, ggbal, LAPACK_ROW_MAJOR, 'B', 4, a.data(),
                            3, b.data(), 4, &ilo, &ihi, ls.data(), rs.data()),
            -5);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ggbak, ggbak, LAPACK_ROW_MAJOR, 'B', 'R', 4, 1,
                            4, ls.data(), rs.data(), 4, a.data(), 3),
            -11);
  EXPECT_EQ(
      LAPACKE_CALL_RC(T, gghrd, gghrd, LAPACK_ROW_MAJOR, 'N', 'N', 4, 1, 4,
                      a.data(), 4, b.data(), 4, q.data(), 3, z.data(), 4),
      -12);
  // NaN checks.
  a[5] = NaN<T>();
  EXPECT_EQ(LapackeGgev<T>(LAPACK_COL_MAJOR, 'V', 'V', 4, a.data(), 4, b.data(),
                           4, al, be, q.data(), 4, z.data(), 4),
            -5);
  EXPECT_EQ(
      LapackeGges<T>(LAPACK_COL_MAJOR, 'V', 'V', 'N', 4, a.data(), 4, b.data(),
                     4, &sdim, al, be, q.data(), 4, z.data(), 4),
      -7);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ggbal, ggbal, LAPACK_COL_MAJOR, 'B', 4, a.data(),
                            4, b.data(), 4, &ilo, &ihi, ls.data(), rs.data()),
            -4);
  // ... but not for JOB = 'N' (A is not referenced).
  EXPECT_EQ(LAPACKE_CALL_RC(T, ggbal, ggbal, LAPACK_COL_MAJOR, 'N', 4, a.data(),
                            4, b.data(), 4, &ilo, &ihi, ls.data(), rs.data()),
            0);
  EXPECT_EQ(
      LAPACKE_CALL_RC(T, gghrd, gghrd, LAPACK_COL_MAJOR, 'N', 'N', 4, 1, 4,
                      a.data(), 4, b.data(), 4, q.data(), 4, z.data(), 4),
      -7);
  ls[1] = NAN;
  EXPECT_EQ(LAPACKE_CALL_RC(T, ggbak, ggbak, LAPACK_COL_MAJOR, 'B', 'R', 4, 1,
                            4, ls.data(), rs.data(), 4, b.data(), 4),
            -7);
  const R nan = NAN;
  lapack_int k, l;
  EXPECT_EQ(LAPACKE_CALL_RC(T, ggsvp3, ggsvp3, LAPACK_COL_MAJOR, 'N', 'N', 'N',
                            4, 4, 4, b.data(), 4, b.data(), 4, nan, R(1), &k,
                            &l, q.data(), 4, z.data(), 4, q.data(), 4),
            -12);
  ::testing::internal::GetCapturedStdout();
}

}  // namespace
}  // namespace hwyblas_test
