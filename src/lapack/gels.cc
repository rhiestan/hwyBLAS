// Linear least squares problems: by QR or LQ factorization for full-rank
// matrices (xGELS), by the singular value decomposition (xGELSS), and by
// the complete orthogonal factorization (xGELSY), with the incremental
// condition estimator xLAIC1.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <complex>
#include <type_traits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// DLAIC1 (real T).
template <typename T>
void Laic1Real(blasint job, blasint j, const T* x, T sest, const T* w, T gamma,
               T& sestpr, T& s, T& c) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = R(0.5);
  const R four = R(4);
  R absalp, absest, absgam, alpha, b, cosine, eps, norma, s1, s2, sine, t, test,
      tmp, zeta1, zeta2;
  eps = Lamch<R>('E');
  alpha = Dotu(j, x, 1, w, 1);
  absalp = std::abs(alpha);
  absgam = std::abs(gamma);
  absest = std::abs(sest);
  if (job == 1) {
    // Estimating largest singular value
    // special cases
    if (sest == zero) {
      s1 = Max(absgam, absalp);
      if (s1 == zero) {
        s = zero;
        c = one;
        sestpr = zero;
      } else {
        s = alpha / s1;
        c = gamma / s1;
        tmp = std::sqrt(s * s + c * c);
        s = s / tmp;
        c = c / tmp;
        sestpr = s1 * tmp;
      }
      return;
    } else if (absgam <= eps * absest) {
      s = one;
      c = zero;
      tmp = Max(absest, absalp);
      s1 = absest / tmp;
      s2 = absalp / tmp;
      sestpr = tmp * std::sqrt(s1 * s1 + s2 * s2);
      return;
    } else if (absalp <= eps * absest) {
      s1 = absgam;
      s2 = absest;
      if (s1 <= s2) {
        s = one;
        c = zero;
        sestpr = s2;
      } else {
        s = zero;
        c = one;
        sestpr = s1;
      }
      return;
    } else if (absest <= eps * absalp || absest <= eps * absgam) {
      s1 = absgam;
      s2 = absalp;
      if (s1 <= s2) {
        tmp = s1 / s2;
        s = std::sqrt(one + tmp * tmp);
        sestpr = s2 * s;
        c = (gamma / s2) / s;
        s = std::copysign(one, alpha) / s;
      } else {
        tmp = s2 / s1;
        c = std::sqrt(one + tmp * tmp);
        sestpr = s1 * c;
        s = (alpha / s1) / c;
        c = std::copysign(one, gamma) / c;
      }
      return;
    } else {
      // normal case
      zeta1 = alpha / absest;
      zeta2 = gamma / absest;
      b = (one - zeta1 * zeta1 - zeta2 * zeta2) * half;
      c = zeta1 * zeta1;
      if (b > zero) {
        t = c / (b + std::sqrt(b * b + c));
      } else {
        t = std::sqrt(b * b + c) - b;
      }
      sine = -zeta1 / t;
      cosine = -zeta2 / (one + t);
      tmp = std::sqrt(sine * sine + cosine * cosine);
      s = sine / tmp;
      c = cosine / tmp;
      sestpr = std::sqrt(t + one) * absest;
      return;
    }
  } else if (job == 2) {
    // Estimating smallest singular value
    // special cases
    if (sest == zero) {
      sestpr = zero;
      if (Max(absgam, absalp) == zero) {
        sine = one;
        cosine = zero;
      } else {
        sine = -gamma;
        cosine = alpha;
      }
      s1 = Max(std::abs(sine), std::abs(cosine));
      s = sine / s1;
      c = cosine / s1;
      tmp = std::sqrt(s * s + c * c);
      s = s / tmp;
      c = c / tmp;
      return;
    } else if (absgam <= eps * absest) {
      s = zero;
      c = one;
      sestpr = absgam;
      return;
    } else if (absalp <= eps * absest) {
      s1 = absgam;
      s2 = absest;
      if (s1 <= s2) {
        s = zero;
        c = one;
        sestpr = s1;
      } else {
        s = one;
        c = zero;
        sestpr = s2;
      }
      return;
    } else if (absest <= eps * absalp || absest <= eps * absgam) {
      s1 = absgam;
      s2 = absalp;
      if (s1 <= s2) {
        tmp = s1 / s2;
        c = std::sqrt(one + tmp * tmp);
        sestpr = absest * (tmp / c);
        s = -(gamma / s2) / c;
        c = std::copysign(one, alpha) / c;
      } else {
        tmp = s2 / s1;
        s = std::sqrt(one + tmp * tmp);
        sestpr = absest / s;
        c = (alpha / s1) / s;
        s = -std::copysign(one, gamma) / s;
      }
      return;
    } else {
      // normal case
      zeta1 = alpha / absest;
      zeta2 = gamma / absest;
      norma = Max(one + zeta1 * zeta1 + std::abs(zeta1 * zeta2),
                  std::abs(zeta1 * zeta2) + zeta2 * zeta2);
      // See if root is closer to zero or to ONE
      test = one + two * (zeta1 - zeta2) * (zeta1 + zeta2);
      if (test >= zero) {
        // root is close to zero, compute directly
        b = (zeta1 * zeta1 + zeta2 * zeta2 + one) * half;
        c = zeta2 * zeta2;
        t = c / (b + std::sqrt(std::abs(b * b - c)));
        sine = zeta1 / (one - t);
        cosine = -zeta2 / t;
        sestpr = std::sqrt(t + four * eps * eps * norma) * absest;
      } else {
        // root is closer to ONE, shift by that amount
        b = (zeta2 * zeta2 + zeta1 * zeta1 - one) * half;
        c = zeta1 * zeta1;
        if (b >= zero) {
          t = -c / (b + std::sqrt(b * b + c));
        } else {
          t = b - std::sqrt(b * b + c);
        }
        sine = -zeta1 / t;
        cosine = -zeta2 / (one + t);
        sestpr = std::sqrt(one + t + four * eps * eps * norma) * absest;
      }
      tmp = std::sqrt(sine * sine + cosine * cosine);
      s = sine / tmp;
      c = cosine / tmp;
      return;
    }
  }
}

// ZLAIC1 (complex T). As the Fortran, the output C holds intermediate
// values of the normal case, whose square roots are complex.
template <typename T>
void Laic1Complex(blasint job, blasint j, const T* x, RealT<T> sest, const T* w,
                  T gamma, RealT<T>& sestpr, T& s, T& c) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = R(0.5);
  const R four = R(4);
  R absalp, absest, absgam, b, eps, norma, s1, s2, scl, t, test, tmp, zeta1,
      zeta2;
  T alpha, cosine, sine;
  eps = Lamch<R>('E');
  alpha = Dotc(j, x, 1, w, 1);
  absalp = std::abs(alpha);
  absgam = std::abs(gamma);
  absest = std::abs(sest);
  if (job == 1) {
    // Estimating largest singular value
    // special cases
    if (sest == zero) {
      s1 = Max(absgam, absalp);
      if (s1 == zero) {
        s = zero;
        c = one;
        sestpr = zero;
      } else {
        s = alpha / s1;
        c = gamma / s1;
        tmp = std::real(std::sqrt(s * Conj(s) + c * Conj(c)));
        s = s / tmp;
        c = c / tmp;
        sestpr = s1 * tmp;
      }
      return;
    } else if (absgam <= eps * absest) {
      s = one;
      c = zero;
      tmp = Max(absest, absalp);
      s1 = absest / tmp;
      s2 = absalp / tmp;
      sestpr = tmp * std::sqrt(s1 * s1 + s2 * s2);
      return;
    } else if (absalp <= eps * absest) {
      s1 = absgam;
      s2 = absest;
      if (s1 <= s2) {
        s = one;
        c = zero;
        sestpr = s2;
      } else {
        s = zero;
        c = one;
        sestpr = s1;
      }
      return;
    } else if (absest <= eps * absalp || absest <= eps * absgam) {
      s1 = absgam;
      s2 = absalp;
      if (s1 <= s2) {
        tmp = s1 / s2;
        scl = std::sqrt(one + tmp * tmp);
        sestpr = s2 * scl;
        s = (alpha / s2) / scl;
        c = (gamma / s2) / scl;
      } else {
        tmp = s2 / s1;
        scl = std::sqrt(one + tmp * tmp);
        sestpr = s1 * scl;
        s = (alpha / s1) / scl;
        c = (gamma / s1) / scl;
      }
      return;
    } else {
      // normal case
      zeta1 = absalp / absest;
      zeta2 = absgam / absest;
      b = (one - zeta1 * zeta1 - zeta2 * zeta2) * half;
      c = zeta1 * zeta1;
      if (b > zero) {
        t = std::real(c / (b + std::sqrt(b * b + c)));
      } else {
        t = std::real(std::sqrt(b * b + c) - b);
      }
      sine = -(alpha / absest) / t;
      cosine = -(gamma / absest) / (one + t);
      tmp = std::real(std::sqrt(sine * Conj(sine) + cosine * Conj(cosine)));
      s = sine / tmp;
      c = cosine / tmp;
      sestpr = std::sqrt(t + one) * absest;
      return;
    }
  } else if (job == 2) {
    // Estimating smallest singular value
    // special cases
    if (sest == zero) {
      sestpr = zero;
      if (Max(absgam, absalp) == zero) {
        sine = one;
        cosine = zero;
      } else {
        sine = -Conj(gamma);
        cosine = Conj(alpha);
      }
      s1 = Max(std::abs(sine), std::abs(cosine));
      s = sine / s1;
      c = cosine / s1;
      tmp = std::real(std::sqrt(s * Conj(s) + c * Conj(c)));
      s = s / tmp;
      c = c / tmp;
      return;
    } else if (absgam <= eps * absest) {
      s = zero;
      c = one;
      sestpr = absgam;
      return;
    } else if (absalp <= eps * absest) {
      s1 = absgam;
      s2 = absest;
      if (s1 <= s2) {
        s = zero;
        c = one;
        sestpr = s1;
      } else {
        s = one;
        c = zero;
        sestpr = s2;
      }
      return;
    } else if (absest <= eps * absalp || absest <= eps * absgam) {
      s1 = absgam;
      s2 = absalp;
      if (s1 <= s2) {
        tmp = s1 / s2;
        scl = std::sqrt(one + tmp * tmp);
        sestpr = absest * (tmp / scl);
        s = -(Conj(gamma) / s2) / scl;
        c = (Conj(alpha) / s2) / scl;
      } else {
        tmp = s2 / s1;
        scl = std::sqrt(one + tmp * tmp);
        sestpr = absest / scl;
        s = -(Conj(gamma) / s1) / scl;
        c = (Conj(alpha) / s1) / scl;
      }
      return;
    } else {
      // normal case
      zeta1 = absalp / absest;
      zeta2 = absgam / absest;
      norma = Max(one + zeta1 * zeta1 + zeta1 * zeta2,
                  zeta1 * zeta2 + zeta2 * zeta2);
      // See if root is closer to zero or to ONE
      test = one + two * (zeta1 - zeta2) * (zeta1 + zeta2);
      if (test >= zero) {
        // root is close to zero, compute directly
        b = (zeta1 * zeta1 + zeta2 * zeta2 + one) * half;
        c = zeta2 * zeta2;
        t = std::real(c / (b + std::sqrt(std::abs(b * b - c))));
        sine = (alpha / absest) / (one - t);
        cosine = -(gamma / absest) / t;
        sestpr = std::sqrt(t + four * eps * eps * norma) * absest;
      } else {
        // root is closer to ONE, shift by that amount
        b = (zeta2 * zeta2 + zeta1 * zeta1 - one) * half;
        c = zeta1 * zeta1;
        if (b >= zero) {
          t = std::real(-c / (b + std::sqrt(b * b + c)));
        } else {
          t = std::real(b - std::sqrt(b * b + c));
        }
        sine = -(alpha / absest) / t;
        cosine = -(gamma / absest) / (one + t);
        sestpr = std::sqrt(one + t + four * eps * eps * norma) * absest;
      }
      tmp = std::real(std::sqrt(sine * Conj(sine) + cosine * Conj(cosine)));
      s = sine / tmp;
      c = cosine / tmp;
      return;
    }
  }
}

// DGELSS (real T).
template <typename T>
blasint GelssReal(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
                  blasint ldb, T* s, T rcond, blasint* rank, T* work,
                  blasint lwork) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Matrix<R> B(b, ldb);
  R dum[1];
  const Vector<R> DUM(dum);
  const Vector<R> S(s);
  const Vector<R> WORK(work);
  blasint bdspac, bl, chunk, i, iascl, ibscl, ie, il, itau, itaup, itauq, iwork,
      ldwork, lwork_dgebrd, lwork_dgelqf, lwork_dgeqrf, lwork_dorgbr,
      lwork_dormbr, lwork_dormlq, lwork_dormqr, maxmn, maxwrk, minmn, minwrk,
      mm, mnthr;
  bool lquery;
  R anrm, bignum, bnrm, eps, sfmin, smlnum, thr;
  blasint info = 0;
  minmn = Min<blasint>(m, n);
  maxmn = Max<blasint>(m, n);
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, maxmn)) {
    info = -7;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV.)
  if (info == 0) {
    minwrk = 1;
    maxwrk = 1;
    if (minmn > 0) {
      mm = m;
      mnthr = Ilaenv<T>(6, "GELSS", " ", m, n, nrhs, -1);
      if (m >= n && m >= mnthr) {
        // Path 1a - overdetermined, with many more rows than
        // columns
        // Compute space needed for DGEQRF
        info = Geqrf(m, n, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
        lwork_dgeqrf = static_cast<blasint>(DUM(1));
        // Compute space needed for DORMQR
        info = Unmqr('L', 'T', m, nrhs, n, a, lda, DUM.ptr(1), b, ldb,
                     DUM.ptr(1), -1);
        lwork_dormqr = static_cast<blasint>(DUM(1));
        mm = n;
        maxwrk = Max<blasint>(maxwrk, n + lwork_dgeqrf);
        maxwrk = Max<blasint>(maxwrk, n + lwork_dormqr);
      }
      if (m >= n) {
        // Path 1 - overdetermined or exactly determined
        // Compute workspace needed for DBDSQR
        bdspac = Max<blasint>(1, 5 * n);
        // Compute space needed for DGEBRD
        info = Gebrd(mm, n, a, lda, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1),
                     DUM.ptr(1), -1);
        lwork_dgebrd = static_cast<blasint>(DUM(1));
        // Compute space needed for DORMBR
        info = Unmbr('Q', 'L', 'T', mm, nrhs, n, a, lda, DUM.ptr(1), b, ldb,
                     DUM.ptr(1), -1);
        lwork_dormbr = static_cast<blasint>(DUM(1));
        // Compute space needed for DORGBR
        info = Ungbr('P', n, n, n, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
        lwork_dorgbr = static_cast<blasint>(DUM(1));
        // Compute total workspace needed
        maxwrk = Max<blasint>(maxwrk, 3 * n + lwork_dgebrd);
        maxwrk = Max<blasint>(maxwrk, 3 * n + lwork_dormbr);
        maxwrk = Max<blasint>(maxwrk, 3 * n + lwork_dorgbr);
        maxwrk = Max<blasint>(maxwrk, bdspac);
        maxwrk = Max<blasint>(maxwrk, n * nrhs);
        minwrk = Max<blasint>(3 * n + mm, 3 * n + nrhs, bdspac);
        maxwrk = Max<blasint>(minwrk, maxwrk);
      }
      if (n > m) {
        // Compute workspace needed for DBDSQR
        bdspac = Max<blasint>(1, 5 * m);
        minwrk = Max<blasint>(3 * m + nrhs, 3 * m + n, bdspac);
        if (n >= mnthr) {
          // Path 2a - underdetermined, with many more columns
          // than rows
          // Compute space needed for DGELQF
          // (SGELSS takes its block size from ILAENV below.)
          if constexpr (std::is_same_v<T, double>) {
            info = Gelqf(m, n, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
            lwork_dgelqf = static_cast<blasint>(DUM(1));
          }
          // Compute space needed for DGEBRD
          info = Gebrd(m, m, a, lda, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1),
                       DUM.ptr(1), -1);
          lwork_dgebrd = static_cast<blasint>(DUM(1));
          // Compute space needed for DORMBR
          info = Unmbr('Q', 'L', 'T', m, nrhs, n, a, lda, DUM.ptr(1), b, ldb,
                       DUM.ptr(1), -1);
          lwork_dormbr = static_cast<blasint>(DUM(1));
          // Compute space needed for DORGBR
          info = Ungbr('P', m, m, m, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
          lwork_dorgbr = static_cast<blasint>(DUM(1));
          // Compute space needed for DORMLQ
          info = Unmlq('L', 'T', n, nrhs, m, a, lda, DUM.ptr(1), b, ldb,
                       DUM.ptr(1), -1);
          lwork_dormlq = static_cast<blasint>(DUM(1));
          // Compute total workspace needed
          if constexpr (std::is_same_v<T, double>) {
            maxwrk = m + lwork_dgelqf;
          } else {
            maxwrk = m + m * Ilaenv<T>(1, "GELQF", " ", m, n, -1, -1);
          }
          maxwrk = Max<blasint>(maxwrk, m * m + 4 * m + lwork_dgebrd);
          maxwrk = Max<blasint>(maxwrk, m * m + 4 * m + lwork_dormbr);
          maxwrk = Max<blasint>(maxwrk, m * m + 4 * m + lwork_dorgbr);
          maxwrk = Max<blasint>(maxwrk, m * m + m + bdspac);
          if (nrhs > 1) {
            maxwrk = Max<blasint>(maxwrk, m * m + m + m * nrhs);
          } else {
            maxwrk = Max<blasint>(maxwrk, m * m + 2 * m);
          }
          maxwrk = Max<blasint>(maxwrk, m + lwork_dormlq);
        } else {
          // Path 2 - underdetermined
          // Compute space needed for DGEBRD
          info = Gebrd(m, n, a, lda, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1),
                       DUM.ptr(1), -1);
          lwork_dgebrd = static_cast<blasint>(DUM(1));
          // Compute space needed for DORMBR
          info = Unmbr('Q', 'L', 'T', m, nrhs, m, a, lda, DUM.ptr(1), b, ldb,
                       DUM.ptr(1), -1);
          lwork_dormbr = static_cast<blasint>(DUM(1));
          // Compute space needed for DORGBR
          info = Ungbr('P', m, n, m, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
          lwork_dorgbr = static_cast<blasint>(DUM(1));
          maxwrk = 3 * m + lwork_dgebrd;
          maxwrk = Max<blasint>(maxwrk, 3 * m + lwork_dormbr);
          maxwrk = Max<blasint>(maxwrk, 3 * m + lwork_dorgbr);
          maxwrk = Max<blasint>(maxwrk, bdspac);
          maxwrk = Max<blasint>(maxwrk, n * nrhs);
        }
      }
      maxwrk = Max<blasint>(minwrk, maxwrk);
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) info = -12;
  }
  if (info != 0) {
    Xerbla<T>("GELSS", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (m == 0 || n == 0) {
    *rank = 0;
    return info;
  }
  // Get machine parameters
  eps = Lamch<R>('P');
  sfmin = Lamch<R>('S');
  smlnum = sfmin / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', m, n, a, lda, work);
  iascl = 0;
  if (anrm > zero && anrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
    iascl = 1;
  } else if (anrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
    iascl = 2;
  } else if (anrm == zero) {
    // Matrix all zero. Return zero solution.
    Laset('F', Max<blasint>(m, n), nrhs, zero, zero, b, ldb);
    Laset('F', minmn, 1, zero, zero, s, minmn);
    *rank = 0;
    goto L70;
  }
  // Scale B if max element outside range [SMLNUM,BIGNUM]
  bnrm = Lange('M', m, nrhs, b, ldb, work);
  ibscl = 0;
  if (bnrm > zero && bnrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, bnrm, smlnum, m, nrhs, b, ldb);
    ibscl = 1;
  } else if (bnrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, bnrm, bignum, m, nrhs, b, ldb);
    ibscl = 2;
  }
  // Overdetermined case
  if (m >= n) {
    // Path 1 - overdetermined or exactly determined
    mm = m;
    if (m >= mnthr) {
      // Path 1a - overdetermined, with many more rows than columns
      mm = n;
      itau = 1;
      iwork = itau + n;
      // Compute A=Q*R
      // (Workspace: need 2*N, prefer N+N*NB)
      info = Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                   lwork - iwork + 1);
      // Multiply B by transpose(Q)
      // (Workspace: need N+NRHS, prefer N+NRHS*NB)
      info = Unmqr('L', 'T', m, nrhs, n, a, lda, WORK.ptr(itau), b, ldb,
                   WORK.ptr(iwork), lwork - iwork + 1);
      // Zero out below R
      if (n > 1) Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
    }
    ie = 1;
    itauq = ie + n;
    itaup = itauq + n;
    iwork = itaup + n;
    // Bidiagonalize R in A
    // (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB)
    info = Gebrd(mm, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of R
    // (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB)
    info = Unmbr('Q', 'L', 'T', mm, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(iwork), lwork - iwork + 1);
    // Generate right bidiagonalizing vectors of R in A
    // (Workspace: need 4*N-1, prefer 3*N+(N-1)*NB)
    info = Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                 lwork - iwork + 1);
    iwork = ie + n;
    // Perform bidiagonal QR iteration
    // multiply B by transpose of left singular vectors
    // compute right singular vectors in A
    // (Workspace: need BDSPAC)
    info = Bdsqr('U', n, n, 0, nrhs, s, WORK.ptr(ie), a, lda, dum, 1, b, ldb,
                 WORK.ptr(iwork));
    if (info != 0) goto L70;
    // Multiply B by reciprocals of singular values
    thr = Max(rcond * S(1), sfmin);
    if (rcond < zero) thr = Max(eps * S(1), sfmin);
    *rank = 0;
    for (i = 1; i <= n; ++i) {
      if (S(i) > thr) {
        Rscl(nrhs, S(i), B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      } else {
        Laset('F', 1, nrhs, zero, zero, B.ptr(i, 1), ldb);
      }
    }
    // Multiply B by right singular vectors
    // (Workspace: need N, prefer N*NRHS)
    if (lwork >= ldb * nrhs && nrhs > 1) {
      Gemm('T', 'N', n, nrhs, n, one, a, lda, b, ldb, zero, work, ldb);
      Lacpy('G', n, nrhs, work, ldb, b, ldb);
    } else if (nrhs > 1) {
      chunk = lwork / n;
      for (i = 1; chunk > 0 ? i <= nrhs : i >= nrhs; i += chunk) {
        bl = Min<blasint>(nrhs - i + 1, chunk);
        Gemm('T', 'N', n, bl, n, one, a, lda, B.ptr(1, i), ldb, zero, work, n);
        Lacpy('G', n, bl, work, n, B.ptr(1, i), ldb);
      }
    } else if (nrhs == 1) {
      Gemv('T', n, n, one, a, lda, b, 1, zero, work, 1);
      Copy(n, work, 1, b, 1);
    }
  } else if (n >= mnthr &&
             lwork >=
                 4 * m + m * m + Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m)) {
    // Path 2a - underdetermined, with many more columns than rows
    // and sufficient workspace for an efficient algorithm
    ldwork = m;
    if (lwork >= Max<blasint>(4 * m + m * lda +
                                  Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m),
                              m * lda + m + m * nrhs))
      ldwork = lda;
    itau = 1;
    iwork = m + 1;
    // Compute A=L*Q
    // (Workspace: need 2*M, prefer M+M*NB)
    info =
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork), lwork - iwork + 1);
    il = iwork;
    // Copy L to WORK(IL), zeroing out above it
    Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwork);
    Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(il + ldwork), ldwork);
    ie = il + ldwork * m;
    itauq = ie + m;
    itaup = itauq + m;
    iwork = itaup + m;
    // Bidiagonalize L in WORK(IL)
    // (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB)
    info = Gebrd(m, m, WORK.ptr(il), ldwork, s, WORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of L
    // (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
    info = Unmbr('Q', 'L', 'T', m, nrhs, m, WORK.ptr(il), ldwork,
                 WORK.ptr(itauq), b, ldb, WORK.ptr(iwork), lwork - iwork + 1);
    // Generate right bidiagonalizing vectors of R in WORK(IL)
    // (Workspace: need M*M+5*M-1, prefer M*M+4*M+(M-1)*NB)
    info = Ungbr('P', m, m, m, WORK.ptr(il), ldwork, WORK.ptr(itaup),
                 WORK.ptr(iwork), lwork - iwork + 1);
    iwork = ie + m;
    // Perform bidiagonal QR iteration,
    // computing right singular vectors of L in WORK(IL) and
    // multiplying B by transpose of left singular vectors
    // (Workspace: need M*M+M+BDSPAC)
    info = Bdsqr('U', m, m, 0, nrhs, s, WORK.ptr(ie), WORK.ptr(il), ldwork, a,
                 lda, b, ldb, WORK.ptr(iwork));
    if (info != 0) goto L70;
    // Multiply B by reciprocals of singular values
    thr = Max(rcond * S(1), sfmin);
    if (rcond < zero) thr = Max(eps * S(1), sfmin);
    *rank = 0;
    for (i = 1; i <= m; ++i) {
      if (S(i) > thr) {
        Rscl(nrhs, S(i), B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      } else {
        Laset('F', 1, nrhs, zero, zero, B.ptr(i, 1), ldb);
      }
    }
    iwork = ie;
    // Multiply B by right singular vectors of L in WORK(IL)
    // (Workspace: need M*M+2*M, prefer M*M+M+M*NRHS)
    if (lwork >= ldb * nrhs + iwork - 1 && nrhs > 1) {
      Gemm('T', 'N', m, nrhs, m, one, WORK.ptr(il), ldwork, b, ldb, zero,
           WORK.ptr(iwork), ldb);
      Lacpy('G', m, nrhs, WORK.ptr(iwork), ldb, b, ldb);
    } else if (nrhs > 1) {
      chunk = (lwork - iwork + 1) / m;
      for (i = 1; chunk > 0 ? i <= nrhs : i >= nrhs; i += chunk) {
        bl = Min<blasint>(nrhs - i + 1, chunk);
        Gemm('T', 'N', m, bl, m, one, WORK.ptr(il), ldwork, B.ptr(1, i), ldb,
             zero, WORK.ptr(iwork), m);
        Lacpy('G', m, bl, WORK.ptr(iwork), m, B.ptr(1, i), ldb);
      }
    } else if (nrhs == 1) {
      Gemv('T', m, m, one, WORK.ptr(il), ldwork, B.ptr(1, 1), 1, zero,
           WORK.ptr(iwork), 1);
      Copy(m, WORK.ptr(iwork), 1, B.ptr(1, 1), 1);
    }
    // Zero out below first M rows of B
    Laset('F', n - m, nrhs, zero, zero, B.ptr(m + 1, 1), ldb);
    iwork = itau + m;
    // Multiply transpose(Q) by B
    // (Workspace: need M+NRHS, prefer M+NRHS*NB)
    info = Unmlq('L', 'T', n, nrhs, m, a, lda, WORK.ptr(itau), b, ldb,
                 WORK.ptr(iwork), lwork - iwork + 1);
  } else {
    // Path 2 - remaining underdetermined cases
    ie = 1;
    itauq = ie + m;
    itaup = itauq + m;
    iwork = itaup + m;
    // Bidiagonalize A
    // (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
    info = Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors
    // (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB)
    info = Unmbr('Q', 'L', 'T', m, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(iwork), lwork - iwork + 1);
    // Generate right bidiagonalizing vectors in A
    // (Workspace: need 4*M, prefer 3*M+M*NB)
    info = Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                 lwork - iwork + 1);
    iwork = ie + m;
    // Perform bidiagonal QR iteration,
    // computing right singular vectors of A in A and
    // multiplying B by transpose of left singular vectors
    // (Workspace: need BDSPAC)
    info = Bdsqr('L', m, n, 0, nrhs, s, WORK.ptr(ie), a, lda, dum, 1, b, ldb,
                 WORK.ptr(iwork));
    if (info != 0) goto L70;
    // Multiply B by reciprocals of singular values
    thr = Max(rcond * S(1), sfmin);
    if (rcond < zero) thr = Max(eps * S(1), sfmin);
    *rank = 0;
    for (i = 1; i <= m; ++i) {
      if (S(i) > thr) {
        Rscl(nrhs, S(i), B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      } else {
        Laset('F', 1, nrhs, zero, zero, B.ptr(i, 1), ldb);
      }
    }
    // Multiply B by right singular vectors of A
    // (Workspace: need N, prefer N*NRHS)
    if (lwork >= ldb * nrhs && nrhs > 1) {
      Gemm('T', 'N', n, nrhs, m, one, a, lda, b, ldb, zero, work, ldb);
      Lacpy('F', n, nrhs, work, ldb, b, ldb);
    } else if (nrhs > 1) {
      chunk = lwork / n;
      for (i = 1; chunk > 0 ? i <= nrhs : i >= nrhs; i += chunk) {
        bl = Min<blasint>(nrhs - i + 1, chunk);
        Gemm('T', 'N', n, bl, m, one, a, lda, B.ptr(1, i), ldb, zero, work, n);
        Lacpy('F', n, bl, work, n, B.ptr(1, i), ldb);
      }
    } else if (nrhs == 1) {
      Gemv('T', m, n, one, a, lda, b, 1, zero, work, 1);
      Copy(n, work, 1, b, 1);
    }
  }
  // Undo scaling
  if (iascl == 1) {
    info = Lascl('G', 0, 0, anrm, smlnum, n, nrhs, b, ldb);
    info = Lascl('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
  } else if (iascl == 2) {
    info = Lascl('G', 0, 0, anrm, bignum, n, nrhs, b, ldb);
    info = Lascl('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
  }
  if (ibscl == 1) {
    info = Lascl('G', 0, 0, smlnum, bnrm, n, nrhs, b, ldb);
  } else if (ibscl == 2) {
    info = Lascl('G', 0, 0, bignum, bnrm, n, nrhs, b, ldb);
  }
L70:
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// ZGELSS (complex T).
template <typename T>
blasint GelssComplex(blasint m, blasint n, blasint nrhs, T* a, blasint lda,
                     T* b, blasint ldb, RealT<T>* s, RealT<T> rcond,
                     blasint* rank, T* work, blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  T dum[1];
  const Vector<T> DUM(dum);
  const Vector<R> RWORK(rwork);
  const Vector<R> S(s);
  const Vector<T> WORK(work);
  blasint bl, chunk, i, iascl, ibscl, ie, il, irwork, itau, itaup, itauq, iwork,
      ldwork, lwork_zgebrd, lwork_zgelqf, lwork_zgeqrf, lwork_zungbr,
      lwork_zunmbr, lwork_zunmlq, lwork_zunmqr, maxmn, maxwrk, minmn, minwrk,
      mm, mnthr;
  bool lquery;
  R anrm, bignum, bnrm, eps, sfmin, smlnum, thr;
  blasint info = 0;
  minmn = Min<blasint>(m, n);
  maxmn = Max<blasint>(m, n);
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, maxmn)) {
    info = -7;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // CWorkspace refers to complex workspace, and RWorkspace refers
  // to real workspace. NB refers to the optimal block size for the
  // immediately following subroutine, as returned by ILAENV.)
  if (info == 0) {
    minwrk = 1;
    maxwrk = 1;
    if (minmn > 0) {
      mm = m;
      mnthr = Ilaenv<T>(6, "GELSS", " ", m, n, nrhs, -1);
      if (m >= n && m >= mnthr) {
        // Path 1a - overdetermined, with many more rows than
        // columns
        // Compute space needed for ZGEQRF
        info = Geqrf(m, n, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
        lwork_zgeqrf = static_cast<blasint>(std::real(DUM(1)));
        // Compute space needed for ZUNMQR
        info = Unmqr('L', 'C', m, nrhs, n, a, lda, DUM.ptr(1), b, ldb,
                     DUM.ptr(1), -1);
        lwork_zunmqr = static_cast<blasint>(std::real(DUM(1)));
        // ZGELSS queries ZGEQRF and ZUNMQR but sizes their workspace from
        // ILAENV.
        (void)lwork_zgeqrf;
        (void)lwork_zunmqr;
        mm = n;
        maxwrk = Max<blasint>(maxwrk,
                              n + n * Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1));
        maxwrk = Max<blasint>(
            maxwrk, n + nrhs * Ilaenv<T>(1, "UNMQR", "LC", m, nrhs, n, -1));
      }
      if (m >= n) {
        // Path 1 - overdetermined or exactly determined
        // Compute space needed for ZGEBRD
        info =
            Gebrd(mm, n, a, lda, s, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), -1);
        lwork_zgebrd = static_cast<blasint>(std::real(DUM(1)));
        // Compute space needed for ZUNMBR
        info = Unmbr('Q', 'L', 'C', mm, nrhs, n, a, lda, DUM.ptr(1), b, ldb,
                     DUM.ptr(1), -1);
        lwork_zunmbr = static_cast<blasint>(std::real(DUM(1)));
        // Compute space needed for ZUNGBR
        info = Ungbr('P', n, n, n, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
        lwork_zungbr = static_cast<blasint>(std::real(DUM(1)));
        // Compute total workspace needed
        maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zgebrd);
        maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr);
        maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr);
        maxwrk = Max<blasint>(maxwrk, n * nrhs);
        minwrk = 2 * n + Max<blasint>(nrhs, m);
      }
      if (n > m) {
        minwrk = 2 * m + Max<blasint>(nrhs, n);
        if (n >= mnthr) {
          // Path 2a - underdetermined, with many more columns
          // than rows
          // Compute space needed for ZGELQF
          info = Gelqf(m, n, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
          lwork_zgelqf = static_cast<blasint>(std::real(DUM(1)));
          // Compute space needed for ZGEBRD
          info =
              Gebrd(m, m, a, lda, s, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), -1);
          lwork_zgebrd = static_cast<blasint>(std::real(DUM(1)));
          // Compute space needed for ZUNMBR
          info = Unmbr('Q', 'L', 'C', m, nrhs, n, a, lda, DUM.ptr(1), b, ldb,
                       DUM.ptr(1), -1);
          lwork_zunmbr = static_cast<blasint>(std::real(DUM(1)));
          // Compute space needed for ZUNGBR
          info = Ungbr('P', m, m, m, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
          lwork_zungbr = static_cast<blasint>(std::real(DUM(1)));
          // Compute space needed for ZUNMLQ
          info = Unmlq('L', 'C', n, nrhs, m, a, lda, DUM.ptr(1), b, ldb,
                       DUM.ptr(1), -1);
          lwork_zunmlq = static_cast<blasint>(std::real(DUM(1)));
          // Compute total workspace needed
          maxwrk = m + lwork_zgelqf;
          maxwrk = Max<blasint>(maxwrk, 3 * m + m * m + lwork_zgebrd);
          maxwrk = Max<blasint>(maxwrk, 3 * m + m * m + lwork_zunmbr);
          maxwrk = Max<blasint>(maxwrk, 3 * m + m * m + lwork_zungbr);
          if (nrhs > 1) {
            maxwrk = Max<blasint>(maxwrk, m * m + m + m * nrhs);
          } else {
            maxwrk = Max<blasint>(maxwrk, m * m + 2 * m);
          }
          maxwrk = Max<blasint>(maxwrk, m + lwork_zunmlq);
        } else {
          // Path 2 - underdetermined
          // Compute space needed for ZGEBRD
          info =
              Gebrd(m, n, a, lda, s, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), -1);
          lwork_zgebrd = static_cast<blasint>(std::real(DUM(1)));
          // Compute space needed for ZUNMBR
          info = Unmbr('Q', 'L', 'C', m, nrhs, m, a, lda, DUM.ptr(1), b, ldb,
                       DUM.ptr(1), -1);
          lwork_zunmbr = static_cast<blasint>(std::real(DUM(1)));
          // Compute space needed for ZUNGBR
          info = Ungbr('P', m, n, m, a, lda, DUM.ptr(1), DUM.ptr(1), -1);
          lwork_zungbr = static_cast<blasint>(std::real(DUM(1)));
          maxwrk = 2 * m + lwork_zgebrd;
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr);
          maxwrk = Max<blasint>(maxwrk, n * nrhs);
        }
      }
      maxwrk = Max<blasint>(minwrk, maxwrk);
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) info = -12;
  }
  if (info != 0) {
    Xerbla<T>("GELSS", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (m == 0 || n == 0) {
    *rank = 0;
    return info;
  }
  // Get machine parameters
  eps = Lamch<R>('P');
  sfmin = Lamch<R>('S');
  smlnum = sfmin / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', m, n, a, lda, rwork);
  iascl = 0;
  if (anrm > zero && anrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
    iascl = 1;
  } else if (anrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
    iascl = 2;
  } else if (anrm == zero) {
    // Matrix all zero. Return zero solution.
    Laset('F', Max<blasint>(m, n), nrhs, czero, czero, b, ldb);
    Laset<R>('F', minmn, 1, zero, zero, s, minmn);
    *rank = 0;
    goto L70;
  }
  // Scale B if max element outside range [SMLNUM,BIGNUM]
  bnrm = Lange('M', m, nrhs, b, ldb, rwork);
  ibscl = 0;
  if (bnrm > zero && bnrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, bnrm, smlnum, m, nrhs, b, ldb);
    ibscl = 1;
  } else if (bnrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, bnrm, bignum, m, nrhs, b, ldb);
    ibscl = 2;
  }
  // Overdetermined case
  if (m >= n) {
    // Path 1 - overdetermined or exactly determined
    mm = m;
    if (m >= mnthr) {
      // Path 1a - overdetermined, with many more rows than columns
      mm = n;
      itau = 1;
      iwork = itau + n;
      // Compute A=Q*R
      // (CWorkspace: need 2*N, prefer N+N*NB)
      // (RWorkspace: none)
      info = Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                   lwork - iwork + 1);
      // Multiply B by transpose(Q)
      // (CWorkspace: need N+NRHS, prefer N+NRHS*NB)
      // (RWorkspace: none)
      info = Unmqr('L', 'C', m, nrhs, n, a, lda, WORK.ptr(itau), b, ldb,
                   WORK.ptr(iwork), lwork - iwork + 1);
      // Zero out below R
      if (n > 1) Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
    }
    ie = 1;
    itauq = 1;
    itaup = itauq + n;
    iwork = itaup + n;
    // Bidiagonalize R in A
    // (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
    // (RWorkspace: need N)
    info = Gebrd(mm, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of R
    // (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
    // (RWorkspace: none)
    info = Unmbr('Q', 'L', 'C', mm, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(iwork), lwork - iwork + 1);
    // Generate right bidiagonalizing vectors of R in A
    // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    // (RWorkspace: none)
    info = Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                 lwork - iwork + 1);
    irwork = ie + n;
    // Perform bidiagonal QR iteration
    // multiply B by transpose of left singular vectors
    // compute right singular vectors in A
    // (CWorkspace: none)
    // (RWorkspace: need BDSPAC)
    info = Bdsqr('U', n, n, 0, nrhs, s, RWORK.ptr(ie), a, lda, dum, 1, b, ldb,
                 RWORK.ptr(irwork));
    if (info != 0) goto L70;
    // Multiply B by reciprocals of singular values
    thr = Max(rcond * S(1), sfmin);
    if (rcond < zero) thr = Max(eps * S(1), sfmin);
    *rank = 0;
    for (i = 1; i <= n; ++i) {
      if (S(i) > thr) {
        Rscl(nrhs, S(i), B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      } else {
        Laset('F', 1, nrhs, czero, czero, B.ptr(i, 1), ldb);
      }
    }
    // Multiply B by right singular vectors
    // (CWorkspace: need N, prefer N*NRHS)
    // (RWorkspace: none)
    if (lwork >= ldb * nrhs && nrhs > 1) {
      Gemm('C', 'N', n, nrhs, n, cone, a, lda, b, ldb, czero, work, ldb);
      Lacpy('G', n, nrhs, work, ldb, b, ldb);
    } else if (nrhs > 1) {
      chunk = lwork / n;
      for (i = 1; chunk > 0 ? i <= nrhs : i >= nrhs; i += chunk) {
        bl = Min<blasint>(nrhs - i + 1, chunk);
        Gemm('C', 'N', n, bl, n, cone, a, lda, B.ptr(1, i), ldb, czero, work,
             n);
        Lacpy('G', n, bl, work, n, B.ptr(1, i), ldb);
      }
    } else if (nrhs == 1) {
      Gemv('C', n, n, cone, a, lda, b, 1, czero, work, 1);
      Copy(n, work, 1, b, 1);
    }
  } else if (n >= mnthr &&
             lwork >= 3 * m + m * m + Max<blasint>(m, nrhs, n - 2 * m)) {
    // Underdetermined case, M much less than N
    // Path 2a - underdetermined, with many more columns than rows
    // and sufficient workspace for an efficient algorithm
    ldwork = m;
    if (lwork >= 3 * m + m * lda + Max<blasint>(m, nrhs, n - 2 * m))
      ldwork = lda;
    itau = 1;
    iwork = m + 1;
    // Compute A=L*Q
    // (CWorkspace: need 2*M, prefer M+M*NB)
    // (RWorkspace: none)
    info =
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork), lwork - iwork + 1);
    il = iwork;
    // Copy L to WORK(IL), zeroing out above it
    Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwork);
    Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(il + ldwork), ldwork);
    ie = 1;
    itauq = il + ldwork * m;
    itaup = itauq + m;
    iwork = itaup + m;
    // Bidiagonalize L in WORK(IL)
    // (CWorkspace: need M*M+4*M, prefer M*M+3*M+2*M*NB)
    // (RWorkspace: need M)
    info = Gebrd(m, m, WORK.ptr(il), ldwork, s, RWORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of L
    // (CWorkspace: need M*M+3*M+NRHS, prefer M*M+3*M+NRHS*NB)
    // (RWorkspace: none)
    info = Unmbr('Q', 'L', 'C', m, nrhs, m, WORK.ptr(il), ldwork,
                 WORK.ptr(itauq), b, ldb, WORK.ptr(iwork), lwork - iwork + 1);
    // Generate right bidiagonalizing vectors of R in WORK(IL)
    // (CWorkspace: need M*M+4*M-1, prefer M*M+3*M+(M-1)*NB)
    // (RWorkspace: none)
    info = Ungbr('P', m, m, m, WORK.ptr(il), ldwork, WORK.ptr(itaup),
                 WORK.ptr(iwork), lwork - iwork + 1);
    irwork = ie + m;
    // Perform bidiagonal QR iteration, computing right singular
    // vectors of L in WORK(IL) and multiplying B by transpose of
    // left singular vectors
    // (CWorkspace: need M*M)
    // (RWorkspace: need BDSPAC)
    info = Bdsqr('U', m, m, 0, nrhs, s, RWORK.ptr(ie), WORK.ptr(il), ldwork, a,
                 lda, b, ldb, RWORK.ptr(irwork));
    if (info != 0) goto L70;
    // Multiply B by reciprocals of singular values
    thr = Max(rcond * S(1), sfmin);
    if (rcond < zero) thr = Max(eps * S(1), sfmin);
    *rank = 0;
    for (i = 1; i <= m; ++i) {
      if (S(i) > thr) {
        Rscl(nrhs, S(i), B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      } else {
        Laset('F', 1, nrhs, czero, czero, B.ptr(i, 1), ldb);
      }
    }
    iwork = il + m * ldwork;
    // Multiply B by right singular vectors of L in WORK(IL)
    // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NRHS)
    // (RWorkspace: none)
    if (lwork >= ldb * nrhs + iwork - 1 && nrhs > 1) {
      Gemm('C', 'N', m, nrhs, m, cone, WORK.ptr(il), ldwork, b, ldb, czero,
           WORK.ptr(iwork), ldb);
      Lacpy('G', m, nrhs, WORK.ptr(iwork), ldb, b, ldb);
    } else if (nrhs > 1) {
      chunk = (lwork - iwork + 1) / m;
      for (i = 1; chunk > 0 ? i <= nrhs : i >= nrhs; i += chunk) {
        bl = Min<blasint>(nrhs - i + 1, chunk);
        Gemm('C', 'N', m, bl, m, cone, WORK.ptr(il), ldwork, B.ptr(1, i), ldb,
             czero, WORK.ptr(iwork), m);
        Lacpy('G', m, bl, WORK.ptr(iwork), m, B.ptr(1, i), ldb);
      }
    } else if (nrhs == 1) {
      Gemv('C', m, m, cone, WORK.ptr(il), ldwork, B.ptr(1, 1), 1, czero,
           WORK.ptr(iwork), 1);
      Copy(m, WORK.ptr(iwork), 1, B.ptr(1, 1), 1);
    }
    // Zero out below first M rows of B
    Laset('F', n - m, nrhs, czero, czero, B.ptr(m + 1, 1), ldb);
    iwork = itau + m;
    // Multiply transpose(Q) by B
    // (CWorkspace: need M+NRHS, prefer M+NHRS*NB)
    // (RWorkspace: none)
    info = Unmlq('L', 'C', n, nrhs, m, a, lda, WORK.ptr(itau), b, ldb,
                 WORK.ptr(iwork), lwork - iwork + 1);
  } else {
    // Path 2 - remaining underdetermined cases
    ie = 1;
    itauq = 1;
    itaup = itauq + m;
    iwork = itaup + m;
    // Bidiagonalize A
    // (CWorkspace: need 3*M, prefer 2*M+(M+N)*NB)
    // (RWorkspace: need N)
    info = Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors
    // (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
    // (RWorkspace: none)
    info = Unmbr('Q', 'L', 'C', m, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(iwork), lwork - iwork + 1);
    // Generate right bidiagonalizing vectors in A
    // (CWorkspace: need 3*M, prefer 2*M+M*NB)
    // (RWorkspace: none)
    info = Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                 lwork - iwork + 1);
    irwork = ie + m;
    // Perform bidiagonal QR iteration,
    // computing right singular vectors of A in A and
    // multiplying B by transpose of left singular vectors
    // (CWorkspace: none)
    // (RWorkspace: need BDSPAC)
    info = Bdsqr('L', m, n, 0, nrhs, s, RWORK.ptr(ie), a, lda, dum, 1, b, ldb,
                 RWORK.ptr(irwork));
    if (info != 0) goto L70;
    // Multiply B by reciprocals of singular values
    thr = Max(rcond * S(1), sfmin);
    if (rcond < zero) thr = Max(eps * S(1), sfmin);
    *rank = 0;
    for (i = 1; i <= m; ++i) {
      if (S(i) > thr) {
        Rscl(nrhs, S(i), B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      } else {
        Laset('F', 1, nrhs, czero, czero, B.ptr(i, 1), ldb);
      }
    }
    // Multiply B by right singular vectors of A
    // (CWorkspace: need N, prefer N*NRHS)
    // (RWorkspace: none)
    if (lwork >= ldb * nrhs && nrhs > 1) {
      Gemm('C', 'N', n, nrhs, m, cone, a, lda, b, ldb, czero, work, ldb);
      Lacpy('G', n, nrhs, work, ldb, b, ldb);
    } else if (nrhs > 1) {
      chunk = lwork / n;
      for (i = 1; chunk > 0 ? i <= nrhs : i >= nrhs; i += chunk) {
        bl = Min<blasint>(nrhs - i + 1, chunk);
        Gemm('C', 'N', n, bl, m, cone, a, lda, B.ptr(1, i), ldb, czero, work,
             n);
        Lacpy('F', n, bl, work, n, B.ptr(1, i), ldb);
      }
    } else if (nrhs == 1) {
      Gemv('C', m, n, cone, a, lda, b, 1, czero, work, 1);
      Copy(n, work, 1, b, 1);
    }
  }
  // Undo scaling
  if (iascl == 1) {
    info = Lascl('G', 0, 0, anrm, smlnum, n, nrhs, b, ldb);
    info = Lascl<R>('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
  } else if (iascl == 2) {
    info = Lascl('G', 0, 0, anrm, bignum, n, nrhs, b, ldb);
    info = Lascl<R>('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
  }
  if (ibscl == 1) {
    info = Lascl('G', 0, 0, smlnum, bnrm, n, nrhs, b, ldb);
  } else if (ibscl == 2) {
    info = Lascl('G', 0, 0, bignum, bnrm, n, nrhs, b, ldb);
  }
L70:
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

}  // namespace

// xLAIC1: one step of incremental condition estimation. Given the
// estimate SEST of the largest (JOB = 1) or smallest (JOB = 2) singular
// value of the J-by-J lower triangular matrix L and an approximate
// singular vector x (X(J), ||x|| = 1), it estimates the singular value
// SESTPR of the (J+1)-by-(J+1) matrix ( L 0; w**H gamma ) with the vector
// ( s*x; c ), returning S and C.
template <typename T>
void Laic1(blasint job, blasint j, const T* x, RealT<T> sest, const T* w,
           T gamma, RealT<T>* sestpr, T* s, T* c) {
  if constexpr (IsComplex<T>::value) {
    Laic1Complex(job, j, x, sest, w, gamma, *sestpr, *s, *c);
  } else {
    Laic1Real(job, j, x, sest, w, gamma, *sestpr, *s, *c);
  }
}

// xGELS: solves overdetermined or underdetermined systems with the M-by-N
// matrix A of full rank or its (conjugate) transpose (TRANS = 'N' or 'T' /
// 'C'): the least squares solution of an overdetermined system and the
// minimum norm solution of an underdetermined one, by a QR (M >= N) or LQ
// (M < N) factorization of A. B (LDB >= max(1, M, N)) holds the NRHS
// right-hand sides on entry and the solutions on exit (with the residual
// sums of squares in the rows below them for overdetermined systems; as
// LAPACK, in scaled units if A or B was scaled). A
// holds the factorization on exit. LWORK >= max(1, MN + max(MN, NRHS));
// the optimal LWORK is MN + max(MN, NRHS)*NB. LWORK = -1 is a workspace
// query. Returns INFO: 0, -i for an illegal argument i, or i > 0 if the
// i-th diagonal element of the triangular factor is zero (A not of full
// rank).
template <typename T>
blasint Gels(char trans, blasint m, blasint n, blasint nrhs, T* a, blasint lda,
             T* b, blasint ldb, T* work, blasint lwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  R rwork[1];
  const Vector<T> WORK(work);
  blasint brow, i, iascl, ibscl, j, mn, nb, scllen, wsize;
  bool lquery, tpsd;
  R anrm, bignum, bnrm, smlnum;
  blasint info = 0;
  mn = Min<blasint>(m, n);
  lquery = (lwork == -1);
  if (!(Lsame(trans, 'N') || Lsame(trans, ConjTrans<T>()))) {
    info = -1;
  } else if (m < 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (nrhs < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, m)) {
    info = -6;
  } else if (ldb < Max<blasint>(1, m, n)) {
    info = -8;
  } else if (lwork < Max<blasint>(1, mn + Max<blasint>(mn, nrhs)) && !lquery) {
    info = -10;
  }
  // Figure out optimal block size
  if (info == 0 || info == -10) {
    tpsd = true;
    if (Lsame(trans, 'N')) tpsd = false;
    const char* ormqr = RealOrComplex<T>("ORMQR", "UNMQR");
    const char* ormlq = RealOrComplex<T>("ORMLQ", "UNMLQ");
    const char lc[3] = {'L', ConjTrans<T>(), '\0'};
    if (m >= n) {
      nb = Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1);
      if (tpsd) {
        nb = Max(nb, Ilaenv<T>(1, ormqr, "LN", m, nrhs, n, -1));
      } else {
        nb = Max(nb, Ilaenv<T>(1, ormqr, lc, m, nrhs, n, -1));
      }
    } else {
      nb = Ilaenv<T>(1, "GELQF", " ", m, n, -1, -1);
      if (tpsd) {
        nb = Max(nb, Ilaenv<T>(1, ormlq, lc, n, nrhs, m, -1));
      } else {
        nb = Max(nb, Ilaenv<T>(1, ormlq, "LN", n, nrhs, m, -1));
      }
    }
    wsize = Max<blasint>(1, mn + Max<blasint>(mn, nrhs) * nb);
    WORK(1) = WorkSize<T>(wsize);
  }
  if (info != 0) {
    Xerbla<T>("GELS", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (Min<blasint>(m, n, nrhs) == 0) {
    Laset('F', Max<blasint>(m, n), nrhs, czero, czero, b, ldb);
    return info;
  }
  // Get machine parameters
  smlnum = Lamch<R>('S') / Lamch<R>('P');
  bignum = one / smlnum;
  // Scale A, B if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', m, n, a, lda, rwork);
  iascl = 0;
  if (anrm > zero && anrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
    iascl = 1;
  } else if (anrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
    iascl = 2;
  } else if (anrm == zero) {
    // Matrix all zero. Return zero solution.
    Laset('F', Max<blasint>(m, n), nrhs, czero, czero, b, ldb);
    goto L50;
  }
  brow = m;
  if (tpsd) brow = n;
  bnrm = Lange('M', brow, nrhs, b, ldb, rwork);
  ibscl = 0;
  if (bnrm > zero && bnrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, bnrm, smlnum, brow, nrhs, b, ldb);
    ibscl = 1;
  } else if (bnrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, bnrm, bignum, brow, nrhs, b, ldb);
    ibscl = 2;
  }
  if (m >= n) {
    // compute QR factorization of A
    info = Geqrf(m, n, a, lda, WORK.ptr(1), WORK.ptr(mn + 1), lwork - mn);
    // workspace at least N, optimally N*NB
    if (!tpsd) {
      // Least-Squares Problem min || A * X - B ||
      // B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS)
      info = Unmqr('L', ConjTrans<T>(), m, nrhs, n, a, lda, WORK.ptr(1), b, ldb,
                   WORK.ptr(mn + 1), lwork - mn);
      // workspace at least NRHS, optimally NRHS*NB
      // B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
      info = Trtrs('U', 'N', 'N', n, nrhs, a, lda, b, ldb);
      if (info > 0) {
        return info;
      }
      scllen = n;
    } else {
      // Underdetermined system of equations A**T * X = B
      // B(1:N,1:NRHS) := inv(R**H) * B(1:N,1:NRHS)
      info = Trtrs('U', ConjTrans<T>(), 'N', n, nrhs, a, lda, b, ldb);
      if (info > 0) {
        return info;
      }
      // B(N+1:M,1:NRHS) = ZERO
      for (j = 1; j <= nrhs; ++j) {
        for (i = n + 1; i <= m; ++i) {
          B(i, j) = czero;
        }
      }
      // B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
      info = Unmqr('L', 'N', m, nrhs, n, a, lda, WORK.ptr(1), b, ldb,
                   WORK.ptr(mn + 1), lwork - mn);
      // workspace at least NRHS, optimally NRHS*NB
      scllen = m;
    }
  } else {
    // Compute LQ factorization of A
    info = Gelqf(m, n, a, lda, WORK.ptr(1), WORK.ptr(mn + 1), lwork - mn);
    // workspace at least M, optimally M*NB.
    if (!tpsd) {
      // underdetermined system of equations A * X = B
      // B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS)
      info = Trtrs('L', 'N', 'N', m, nrhs, a, lda, b, ldb);
      if (info > 0) {
        return info;
      }
      // B(M+1:N,1:NRHS) = 0
      for (j = 1; j <= nrhs; ++j) {
        for (i = m + 1; i <= n; ++i) {
          B(i, j) = czero;
        }
      }
      // B(1:N,1:NRHS) := Q(1:N,:)**H * B(1:M,1:NRHS)
      info = Unmlq('L', ConjTrans<T>(), n, nrhs, m, a, lda, WORK.ptr(1), b, ldb,
                   WORK.ptr(mn + 1), lwork - mn);
      // workspace at least NRHS, optimally NRHS*NB
      scllen = n;
    } else {
      // overdetermined system min || A**H * X - B ||
      // B(1:N,1:NRHS) := Q * B(1:N,1:NRHS)
      info = Unmlq('L', 'N', n, nrhs, m, a, lda, WORK.ptr(1), b, ldb,
                   WORK.ptr(mn + 1), lwork - mn);
      // workspace at least NRHS, optimally NRHS*NB
      // B(1:M,1:NRHS) := inv(L**H) * B(1:M,1:NRHS)
      info = Trtrs('L', ConjTrans<T>(), 'N', m, nrhs, a, lda, b, ldb);
      if (info > 0) {
        return info;
      }
      scllen = m;
    }
  }
  // Undo scaling
  if (iascl == 1) {
    info = Lascl('G', 0, 0, anrm, smlnum, scllen, nrhs, b, ldb);
  } else if (iascl == 2) {
    info = Lascl('G', 0, 0, anrm, bignum, scllen, nrhs, b, ldb);
  }
  if (ibscl == 1) {
    info = Lascl('G', 0, 0, smlnum, bnrm, scllen, nrhs, b, ldb);
  } else if (ibscl == 2) {
    info = Lascl('G', 0, 0, bignum, bnrm, scllen, nrhs, b, ldb);
  }
L50:
  WORK(1) = WorkSize<T>(wsize);
  return info;
}

// xGELSS: computes the minimum norm solution of the least squares problem
// min || B - A*X || for the M-by-N matrix A of any rank and the NRHS
// right-hand sides in B (LDB >= max(1, M, N)), by the singular value
// decomposition A = U * S * V**H (bidiagonalization and xBDSQR). Singular
// values S(i) <= RCOND*S(1) count as zero (RCOND < 0: machine precision);
// RANK is the effective rank. On exit, A holds the right singular vectors
// (its first min(M,N) rows; but for M < N with an LQ factorization first,
// which needs N >= MNTHR and a large enough LWORK, that factorization, which
// LAPACK's documentation does not say), S the singular values in decreasing
// order, B the solutions. The real and complex routines use different
// workspaces:
//   real T:    LWORK >= 3*min(M,N) + max(2*min(M,N), max(M,N), NRHS);
//              RWORK is not referenced.
//   complex T: LWORK >= 2*min(M,N) + max(M,N,NRHS); RWORK has 5*min(M,N)
//              elements.
// LWORK = -1 is a workspace query. Returns INFO: 0, -i for an illegal
// argument i, or i > 0 if xBDSQR failed to converge (i off-diagonal
// elements did not converge to zero). SGELSS sizes the workspace of
// xGELQF from ILAENV where DGELSS asks xGELQF.
template <typename T>
blasint Gelss(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
              blasint ldb, RealT<T>* s, RealT<T> rcond, blasint* rank, T* work,
              blasint lwork, RealT<T>* rwork) {
  if constexpr (IsComplex<T>::value) {
    return GelssComplex(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork,
                        rwork);
  } else {
    return GelssReal(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork);
  }
}

// xGELSY: computes the minimum norm solution of the least squares problem
// min || B - A*X || for the M-by-N matrix A of any rank by a complete
// orthogonal factorization: a QR factorization with column pivoting
// (xGEQP3), whose rank RANK is the largest leading triangle R11 with an
// estimated condition number (xLAIC1) below 1/RCOND, then R12 is
// annihilated (xTZRZF). JPVT gives the columns to keep in front (nonzero
// on entry) and the permutation on exit. B (LDB >= max(1, M, N)) holds the
// right-hand sides on entry and the solutions on exit; A the factorization.
// LWORK >= MN + max(2*MN, N+1, MN+NRHS) (MN = min(M,N); for real T, 1 if
// MN = 0 or NRHS = 0); RWORK (complex T only) has 2*N elements. LWORK = -1
// is a workspace query. Returns INFO: 0 or -i for an illegal argument i.
// As LAPACK, the real routine accepts that LWORK, but its xGEQP3 needs the
// documented max(MN+3*N+1, 2*MN+NRHS) (else xGEQP3 reports its LWORK).
template <typename T>
blasint Gelsy(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
              blasint ldb, blasint* jpvt, RealT<T> rcond, blasint* rank,
              T* work, blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;

  const blasint imax = 1;
  const blasint imin = 2;
  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<blasint> JPVT(jpvt);
  const Vector<R> RWORK(rwork);
  const Vector<T> WORK(work);
  blasint i, iascl, ibscl, ismax, ismin, j, lwkmin, lwkopt, mn, nb, nb1, nb2,
      nb3, nb4;
  bool lquery;
  R anrm, bignum, bnrm, smax, smaxpr, smin, sminpr, smlnum, wsize;
  T c1, c2, s1, s2;
  mn = Min<blasint>(m, n);
  ismin = mn + 1;
  ismax = 2 * mn + 1;
  // Test the input arguments.
  blasint info = 0;
  const char* ormqr = RealOrComplex<T>("ORMQR", "UNMQR");
  const char* ormrq = RealOrComplex<T>("ORMRQ", "UNMRQ");
  lwkopt = 1;
  if constexpr (IsComplex<T>::value) {
    nb1 = Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1);
    nb2 = Ilaenv<T>(1, "GERQF", " ", m, n, -1, -1);
    nb3 = Ilaenv<T>(1, ormqr, " ", m, n, nrhs, -1);
    nb4 = Ilaenv<T>(1, ormrq, " ", m, n, nrhs, -1);
    nb = Max(nb1, nb2, nb3, nb4);
    lwkopt = Max<blasint>(1, mn + 2 * n + nb * (n + 1), 2 * mn + nb * nrhs);
    WORK(1) = WorkSize<T>(lwkopt);
  }
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, m, n)) {
    info = -7;
  } else if (IsComplex<T>::value &&
             lwork < (mn + Max<blasint>(2 * mn, n + 1, mn + nrhs)) && !lquery) {
    info = -12;
  }

  if constexpr (!IsComplex<T>::value) {
    if (info == 0) {
      if (mn == 0 || nrhs == 0) {
        lwkmin = 1;
        lwkopt = 1;
      } else {
        nb1 = Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1);
        nb2 = Ilaenv<T>(1, "GERQF", " ", m, n, -1, -1);
        nb3 = Ilaenv<T>(1, ormqr, " ", m, n, nrhs, -1);
        nb4 = Ilaenv<T>(1, ormrq, " ", m, n, nrhs, -1);
        nb = Max(nb1, nb2, nb3, nb4);
        lwkmin = mn + Max<blasint>(2 * mn, n + 1, mn + nrhs);
        lwkopt =
            Max<blasint>(lwkmin, mn + 2 * n + nb * (n + 1), 2 * mn + nb * nrhs);
      }
      WORK(1) = WorkSize<T>(lwkopt);

      if (lwork < lwkmin && !lquery) info = -12;
    }
  }

  if (info != 0) {
    Xerbla<T>("GELSY", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (Min<blasint>(m, n, nrhs) == 0) {
    *rank = 0;
    return info;
  }
  // Get machine parameters
  smlnum = Lamch<R>('S') / Lamch<R>('P');
  bignum = one / smlnum;
  // Scale A, B if max entries outside range [SMLNUM,BIGNUM]
  R* lange_work;
  if constexpr (IsComplex<T>::value) {
    lange_work = rwork;
  } else {
    lange_work = work;
  }
  anrm = Lange('M', m, n, a, lda, lange_work);
  iascl = 0;
  if (anrm > zero && anrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
    iascl = 1;
  } else if (anrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
    iascl = 2;
  } else if (anrm == zero) {
    // Matrix all zero. Return zero solution.
    Laset('F', Max<blasint>(m, n), nrhs, czero, czero, b, ldb);
    *rank = 0;
    goto L70;
  }
  bnrm = Lange('M', m, nrhs, b, ldb, lange_work);
  ibscl = 0;
  if (bnrm > zero && bnrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, bnrm, smlnum, m, nrhs, b, ldb);
    ibscl = 1;
  } else if (bnrm > bignum) {
    // Scale matrix norm down to BIGNUM
    info = Lascl('G', 0, 0, bnrm, bignum, m, nrhs, b, ldb);
    ibscl = 2;
  }
  // Compute QR factorization with column pivoting of A:
  // A * P = Q * R
  info = Geqp3(m, n, a, lda, jpvt, WORK.ptr(1), WORK.ptr(mn + 1), lwork - mn,
               rwork);
  wsize = mn + std::real(WORK(mn + 1));
  // complex workspace: MN+NB*(N+1). real workspace 2*N.
  // Details of Householder rotations stored in WORK(1:MN).
  // Determine RANK using incremental condition estimation
  WORK(ismin) = cone;
  WORK(ismax) = cone;
  smax = std::abs(A(1, 1));
  smin = smax;
  if (std::abs(A(1, 1)) == zero) {
    *rank = 0;
    Laset('F', Max<blasint>(m, n), nrhs, czero, czero, b, ldb);
    goto L70;
  } else {
    *rank = 1;
  }
L10:
  if (*rank < mn) {
    i = *rank + 1;
    Laic1(imin, *rank, WORK.ptr(ismin), smin, A.ptr(1, i), A(i, i), &sminpr,
          &s1, &c1);
    Laic1(imax, *rank, WORK.ptr(ismax), smax, A.ptr(1, i), A(i, i), &smaxpr,
          &s2, &c2);
    if (smaxpr * rcond <= sminpr) {
      for (i = 1; i <= *rank; ++i) {
        WORK(ismin + i - 1) = s1 * WORK(ismin + i - 1);
        WORK(ismax + i - 1) = s2 * WORK(ismax + i - 1);
      }
      WORK(ismin + *rank) = c1;
      WORK(ismax + *rank) = c2;
      smin = sminpr;
      smax = smaxpr;
      *rank = *rank + 1;
      goto L10;
    }
  }
  // complex workspace: 3*MN.
  // Logically partition R = [ R11 R12 ]
  // [  0  R22 ]
  // where R11 = R(1:RANK,1:RANK)
  // [R11,R12] = [ T11, 0 ] * Y
  if (*rank < n)
    info = Tzrzf(*rank, n, a, lda, WORK.ptr(mn + 1), WORK.ptr(2 * mn + 1),
                 lwork - 2 * mn);
  // complex workspace: 2*MN.
  // Details of Householder rotations stored in WORK(MN+1:2*MN)
  // B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS)
  info = Unmqr('L', ConjTrans<T>(), m, nrhs, mn, a, lda, WORK.ptr(1), b, ldb,
               WORK.ptr(2 * mn + 1), lwork - 2 * mn);
  wsize = Max(wsize, 2 * mn + std::real(WORK(2 * mn + 1)));
  // complex workspace: 2*MN+NB*NRHS.
  // B(1:RANK,1:NRHS) := inv(T11) * B(1:RANK,1:NRHS)
  Trsm('L', 'U', 'N', 'N', *rank, nrhs, cone, a, lda, b, ldb);
  for (j = 1; j <= nrhs; ++j) {
    for (i = *rank + 1; i <= n; ++i) {
      B(i, j) = czero;
    }
  }
  // B(1:N,1:NRHS) := Y**H * B(1:N,1:NRHS)
  if (*rank < n) {
    info =
        Unmrz('L', ConjTrans<T>(), n, nrhs, *rank, n - *rank, a, lda,
              WORK.ptr(mn + 1), b, ldb, WORK.ptr(2 * mn + 1), lwork - 2 * mn);
  }
  // complex workspace: 2*MN+NRHS.
  // B(1:N,1:NRHS) := P * B(1:N,1:NRHS)
  for (j = 1; j <= nrhs; ++j) {
    for (i = 1; i <= n; ++i) {
      WORK(JPVT(i)) = B(i, j);
    }
    Copy(n, WORK.ptr(1), 1, B.ptr(1, j), 1);
  }
  // complex workspace: N.
  // Undo scaling
  if (iascl == 1) {
    info = Lascl('G', 0, 0, anrm, smlnum, n, nrhs, b, ldb);
    info = Lascl('U', 0, 0, smlnum, anrm, *rank, *rank, a, lda);
  } else if (iascl == 2) {
    info = Lascl('G', 0, 0, anrm, bignum, n, nrhs, b, ldb);
    info = Lascl('U', 0, 0, bignum, anrm, *rank, *rank, a, lda);
  }
  if (ibscl == 1) {
    info = Lascl('G', 0, 0, smlnum, bnrm, n, nrhs, b, ldb);
  } else if (ibscl == 2) {
    info = Lascl('G', 0, 0, bignum, bnrm, n, nrhs, b, ldb);
  }
L70:
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                        \
  template void Laic1<T>(blasint, blasint, const T*, RealT<T>, const T*, T,  \
                         RealT<T>*, T*, T*);                                 \
  template blasint Gels<T>(char, blasint, blasint, blasint, T*, blasint, T*, \
                           blasint, T*, blasint);                            \
  template blasint Gelss<T>(blasint, blasint, blasint, T*, blasint, T*,      \
                            blasint, RealT<T>*, RealT<T>, blasint*, T*,      \
                            blasint, RealT<T>*);                             \
  template blasint Gelsy<T>(blasint, blasint, blasint, T*, blasint, T*,      \
                            blasint, blasint*, RealT<T>, blasint*, T*,       \
                            blasint, RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GELS(prefix, T, R)                                \
  HWYBLAS_API void prefix##gels_(                                        \
      const char* trans, const hwyblas_int* m, const hwyblas_int* n,     \
      const hwyblas_int* nrhs, R* a, const hwyblas_int* lda, R* b,       \
      const hwyblas_int* ldb, R* work, const hwyblas_int* lwork,         \
      hwyblas_int* info) {                                               \
    *info = hwyblas::lapack::Gels(*trans, *m, *n, *nrhs, As<T>(a), *lda, \
                                  As<T>(b), *ldb, As<T>(work), *lwork);  \
  }

HWYBLAS_LAPACK_GELS(s, float, float)
HWYBLAS_LAPACK_GELS(d, double, double)
HWYBLAS_LAPACK_GELS(c, c32, float)
HWYBLAS_LAPACK_GELS(z, c64, double)

#undef HWYBLAS_LAPACK_GELS

#define HWYBLAS_LAPACK_GELS_REAL(prefix, R)                                    \
  HWYBLAS_API void prefix##laic1_(                                             \
      const hwyblas_int* job, const hwyblas_int* j, const R* x, const R* sest, \
      const R* w, const R* gamma, R* sestpr, R* s, R* c) {                     \
    hwyblas::lapack::Laic1(*job, *j, x, *sest, w, *gamma, sestpr, s, c);       \
  }                                                                            \
  HWYBLAS_API void prefix##gelss_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nrhs,     \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* s,        \
      const R* rcond, hwyblas_int* rank, R* work, const hwyblas_int* lwork,    \
      hwyblas_int* info) {                                                     \
    *info =                                                                    \
        hwyblas::lapack::Gelss(*m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond,     \
                               rank, work, *lwork, static_cast<R*>(nullptr));  \
  }                                                                            \
  HWYBLAS_API void prefix##gelsy_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nrhs,     \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb,              \
      hwyblas_int* jpvt, const R* rcond, hwyblas_int* rank, R* work,           \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info =                                                                    \
        hwyblas::lapack::Gelsy(*m, *n, *nrhs, a, *lda, b, *ldb, jpvt, *rcond,  \
                               rank, work, *lwork, static_cast<R*>(nullptr));  \
  }

HWYBLAS_LAPACK_GELS_REAL(s, float)
HWYBLAS_LAPACK_GELS_REAL(d, double)

#undef HWYBLAS_LAPACK_GELS_REAL

#define HWYBLAS_LAPACK_GELS_COMPLEX(prefix, T, R)                              \
  HWYBLAS_API void prefix##laic1_(                                             \
      const hwyblas_int* job, const hwyblas_int* j, const R* x, const R* sest, \
      const R* w, const R* gamma, R* sestpr, R* s, R* c) {                     \
    hwyblas::lapack::Laic1(*job, *j, As<const T>(x), *sest, As<const T>(w),    \
                           *As<const T>(gamma), sestpr, As<T>(s), As<T>(c));   \
  }                                                                            \
  HWYBLAS_API void prefix##gelss_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nrhs,     \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* s,        \
      const R* rcond, hwyblas_int* rank, R* work, const hwyblas_int* lwork,    \
      R* rwork, hwyblas_int* info) {                                           \
    *info =                                                                    \
        hwyblas::lapack::Gelss(*m, *n, *nrhs, As<T>(a), *lda, As<T>(b), *ldb,  \
                               s, *rcond, rank, As<T>(work), *lwork, rwork);   \
  }                                                                            \
  HWYBLAS_API void prefix##gelsy_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nrhs,     \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb,              \
      hwyblas_int* jpvt, const R* rcond, hwyblas_int* rank, R* work,           \
      const hwyblas_int* lwork, R* rwork, hwyblas_int* info) {                 \
    *info = hwyblas::lapack::Gelsy(*m, *n, *nrhs, As<T>(a), *lda, As<T>(b),    \
                                   *ldb, jpvt, *rcond, rank, As<T>(work),      \
                                   *lwork, rwork);                             \
  }

HWYBLAS_LAPACK_GELS_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GELS_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GELS_COMPLEX

}  // extern "C"
