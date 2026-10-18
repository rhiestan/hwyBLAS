// The singular value decomposition of a general matrix: xGESVD.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// The real and the complex routine follow the same paths, but differ in
// their workspaces throughout (the complex routine keeps the bidiagonal's
// off-diagonal and xBDSQR's workspace in RWORK), so they are ported as two
// functions.

// SGESVD / DGESVD.
template <typename T>
// NOLINTNEXTLINE(google-readability-function-size,readability-function-size)
blasint GesvdReal(char jobu, char jobvt, blasint m, blasint n, T* a,
                  blasint lda, T* s, T* u, blasint ldu, T* vt, blasint ldvt,
                  T* work, blasint lwork) {
  using R = T;
  const char opts[3] = {jobu, jobvt, '\0'};

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  R dum[1];
  const Vector<R> S(s);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldvt);
  const Vector<R> WORK(work);
  blasint bdspac, blk, chunk, i, ie, ir, iscl, itau, itaup, itauq, iu, iwork,
      ldwrkr, ldwrku, lwork_dgebrd, lwork_dgelqf, lwork_dgeqrf, lwork_dorgbr_p,
      lwork_dorgbr_q, lwork_dorglq_m, lwork_dorglq_n, lwork_dorgqr_m,
      lwork_dorgqr_n, maxwrk, minmn, minwrk, mnthr, ncu, ncvt, nru, nrvt, wrkbl;
  bool lquery, wntua, wntuas, wntun, wntuo, wntus, wntva, wntvas, wntvn, wntvo,
      wntvs;
  R anrm, bignum, eps, smlnum;
  blasint info = 0;
  minmn = std::min<blasint>(m, n);
  wntua = Lsame(jobu, 'A');
  wntus = Lsame(jobu, 'S');
  wntuas = wntua || wntus;
  wntuo = Lsame(jobu, 'O');
  wntun = Lsame(jobu, 'N');
  wntva = Lsame(jobvt, 'A');
  wntvs = Lsame(jobvt, 'S');
  wntvas = wntva || wntvs;
  wntvo = Lsame(jobvt, 'O');
  wntvn = Lsame(jobvt, 'N');
  lquery = (lwork == -1);
  if (!(wntua || wntus || wntuo || wntun)) {
    info = -1;
  } else if (!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo)) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < std::max<blasint>(1, m)) {
    info = -6;
  } else if (ldu < 1 || (wntuas && ldu < m)) {
    info = -9;
  } else if (ldvt < 1 || (wntva && ldvt < n) || (wntvs && ldvt < minmn)) {
    info = -11;
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
    if (m >= n && minmn > 0) {
      // Compute space needed for DBDSQR
      mnthr = Ilaenv<T>(6, "GESVD", opts, m, n, 0, 0);
      bdspac = 5 * n;
      // Compute space needed for DGEQRF
      Geqrf(m, n, a, lda, dum, dum, -1);
      lwork_dgeqrf = static_cast<blasint>(dum[0]);
      // Compute space needed for DORGQR
      Ungqr(m, n, n, a, lda, dum, dum, -1);
      lwork_dorgqr_n = static_cast<blasint>(dum[0]);
      Ungqr(m, m, n, a, lda, dum, dum, -1);
      lwork_dorgqr_m = static_cast<blasint>(dum[0]);
      // Compute space needed for DGEBRD
      Gebrd(n, n, a, lda, s, dum, dum, dum, dum, -1);
      lwork_dgebrd = static_cast<blasint>(dum[0]);
      // Compute space needed for DORGBR P
      Ungbr('P', n, n, n, a, lda, dum, dum, -1);
      lwork_dorgbr_p = static_cast<blasint>(dum[0]);
      // Compute space needed for DORGBR Q
      Ungbr('Q', n, n, n, a, lda, dum, dum, -1);
      lwork_dorgbr_q = static_cast<blasint>(dum[0]);
      if (m >= mnthr) {
        if (wntun) {
          // Path 1 (M much larger than N, JOBU='N')
          maxwrk = n + lwork_dgeqrf;
          maxwrk = std::max<blasint>(maxwrk, 3 * n + lwork_dgebrd);
          if (wntvo || wntvas)
            maxwrk = std::max<blasint>(maxwrk, 3 * n + lwork_dorgbr_p);
          maxwrk = std::max<blasint>(maxwrk, bdspac);
          minwrk = std::max<blasint>(4 * n, bdspac);
        } else if (wntuo && wntvn) {
          // Path 2 (M much larger than N, JOBU='O', JOBVT='N')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = std::max<blasint>(n * n + wrkbl, n * n + m * n + n);
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntuo && wntvas) {
          // Path 3 (M much larger than N, JOBU='O', JOBVT='S' or
          // 'A')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = std::max<blasint>(n * n + wrkbl, n * n + m * n + n);
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntus && wntvn) {
          // Path 4 (M much larger than N, JOBU='S', JOBVT='N')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntus && wntvo) {
          // Path 5 (M much larger than N, JOBU='S', JOBVT='O')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = 2 * n * n + wrkbl;
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntus && wntvas) {
          // Path 6 (M much larger than N, JOBU='S', JOBVT='S' or
          // 'A')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntua && wntvn) {
          // Path 7 (M much larger than N, JOBU='A', JOBVT='N')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntua && wntvo) {
          // Path 8 (M much larger than N, JOBU='A', JOBVT='O')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = 2 * n * n + wrkbl;
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        } else if (wntua && wntvas) {
          // Path 9 (M much larger than N, JOBU='A', JOBVT='S' or
          // 'A')
          wrkbl = n + lwork_dgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_dorgqr_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, 3 * n + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = n * n + wrkbl;
          minwrk = std::max<blasint>(3 * n + m, bdspac);
        }
      } else {
        // Path 10 (M at least N, but not much larger)
        Gebrd(m, n, a, lda, s, dum, dum, dum, dum, -1);
        lwork_dgebrd = static_cast<blasint>(dum[0]);
        maxwrk = 3 * n + lwork_dgebrd;
        if (wntus || wntuo) {
          Ungbr('Q', m, n, n, a, lda, dum, dum, -1);
          lwork_dorgbr_q = static_cast<blasint>(dum[0]);
          maxwrk = std::max<blasint>(maxwrk, 3 * n + lwork_dorgbr_q);
        }
        if (wntua) {
          Ungbr('Q', m, m, n, a, lda, dum, dum, -1);
          lwork_dorgbr_q = static_cast<blasint>(dum[0]);
          maxwrk = std::max<blasint>(maxwrk, 3 * n + lwork_dorgbr_q);
        }
        if (!wntvn) {
          maxwrk = std::max<blasint>(maxwrk, 3 * n + lwork_dorgbr_p);
        }
        maxwrk = std::max<blasint>(maxwrk, bdspac);
        minwrk = std::max<blasint>(3 * n + m, bdspac);
      }
    } else if (minmn > 0) {
      // Compute space needed for DBDSQR
      mnthr = Ilaenv<T>(6, "GESVD", opts, m, n, 0, 0);
      bdspac = 5 * m;
      // Compute space needed for DGELQF
      Gelqf(m, n, a, lda, dum, dum, -1);
      lwork_dgelqf = static_cast<blasint>(dum[0]);
      // Compute space needed for DORGLQ
      Unglq(n, n, m, dum, n, dum, dum, -1);
      lwork_dorglq_n = static_cast<blasint>(dum[0]);
      Unglq(m, n, m, a, lda, dum, dum, -1);
      lwork_dorglq_m = static_cast<blasint>(dum[0]);
      // Compute space needed for DGEBRD
      Gebrd(m, m, a, lda, s, dum, dum, dum, dum, -1);
      lwork_dgebrd = static_cast<blasint>(dum[0]);
      // Compute space needed for DORGBR P
      Ungbr('P', m, m, m, a, n, dum, dum, -1);
      lwork_dorgbr_p = static_cast<blasint>(dum[0]);
      // Compute space needed for DORGBR Q
      Ungbr('Q', m, m, m, a, n, dum, dum, -1);
      lwork_dorgbr_q = static_cast<blasint>(dum[0]);
      if (n >= mnthr) {
        if (wntvn) {
          // Path 1t(N much larger than M, JOBVT='N')
          maxwrk = m + lwork_dgelqf;
          maxwrk = std::max<blasint>(maxwrk, 3 * m + lwork_dgebrd);
          if (wntuo || wntuas)
            maxwrk = std::max<blasint>(maxwrk, 3 * m + lwork_dorgbr_q);
          maxwrk = std::max<blasint>(maxwrk, bdspac);
          minwrk = std::max<blasint>(4 * m, bdspac);
        } else if (wntvo && wntun) {
          // Path 2t(N much larger than M, JOBU='N', JOBVT='O')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = std::max<blasint>(m * m + wrkbl, m * m + m * n + m);
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntvo && wntuas) {
          // Path 3t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='O')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = std::max<blasint>(m * m + wrkbl, m * m + m * n + m);
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntvs && wntun) {
          // Path 4t(N much larger than M, JOBU='N', JOBVT='S')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntvs && wntuo) {
          // Path 5t(N much larger than M, JOBU='O', JOBVT='S')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = 2 * m * m + wrkbl;
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntvs && wntuas) {
          // Path 6t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='S')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_m);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntva && wntun) {
          // Path 7t(N much larger than M, JOBU='N', JOBVT='A')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntva && wntuo) {
          // Path 8t(N much larger than M, JOBU='O', JOBVT='A')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = 2 * m * m + wrkbl;
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        } else if (wntva && wntuas) {
          // Path 9t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='A')
          wrkbl = m + lwork_dgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_dorglq_n);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dgebrd);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_p);
          wrkbl = std::max<blasint>(wrkbl, 3 * m + lwork_dorgbr_q);
          wrkbl = std::max<blasint>(wrkbl, bdspac);
          maxwrk = m * m + wrkbl;
          minwrk = std::max<blasint>(3 * m + n, bdspac);
        }
      } else {
        // Path 10t(N greater than M, but not much larger)
        Gebrd(m, n, a, lda, s, dum, dum, dum, dum, -1);
        lwork_dgebrd = static_cast<blasint>(dum[0]);
        maxwrk = 3 * m + lwork_dgebrd;
        if (wntvs || wntvo) {
          // Compute space needed for DORGBR P
          Ungbr('P', m, n, m, a, n, dum, dum, -1);
          lwork_dorgbr_p = static_cast<blasint>(dum[0]);
          maxwrk = std::max<blasint>(maxwrk, 3 * m + lwork_dorgbr_p);
        }
        if (wntva) {
          Ungbr('P', n, n, m, a, n, dum, dum, -1);
          lwork_dorgbr_p = static_cast<blasint>(dum[0]);
          maxwrk = std::max<blasint>(maxwrk, 3 * m + lwork_dorgbr_p);
        }
        if (!wntun) {
          maxwrk = std::max<blasint>(maxwrk, 3 * m + lwork_dorgbr_q);
        }
        maxwrk = std::max<blasint>(maxwrk, bdspac);
        minwrk = std::max<blasint>(3 * m + n, bdspac);
      }
    }
    maxwrk = std::max<blasint>(maxwrk, minwrk);
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -13;
    }
  }
  if (info != 0) {
    Xerbla<T>("GESVD", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (m == 0 || n == 0) {
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = std::sqrt(Lamch<R>('S')) / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', m, n, a, lda, dum);
  iscl = 0;
  if (anrm > zero && anrm < smlnum) {
    iscl = 1;
    Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
  } else if (anrm > bignum) {
    iscl = 1;
    Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
  }
  if (m >= n) {
    // A has at least as many rows as columns. If A has sufficiently
    // more rows than columns, first reduce using the QR
    // decomposition (if sufficient workspace available)
    if (m >= mnthr) {
      if (wntun) {
        // Path 1 (M much larger than N, JOBU='N')
        // No left singular vectors to be computed
        itau = 1;
        iwork = itau + n;
        // Compute A=Q*R
        // (Workspace: need 2*N, prefer N + N*NB)
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork), lwork - iwork + 1);
        // Zero out below R
        if (n > 1) {
          Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
        }
        ie = 1;
        itauq = ie + n;
        itaup = itauq + n;
        iwork = itaup + n;
        // Bidiagonalize R in A
        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
        Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(iwork), lwork - iwork + 1);
        ncvt = 0;
        if (wntvo || wntvas) {
          // If right singular vectors desired, generate P'.
          // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
          Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          ncvt = n;
        }
        iwork = ie + n;
        // Perform bidiagonal QR iteration, computing right
        // singular vectors of A in A if desired
        // (Workspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, 0, 0, s, WORK.ptr(ie), a, lda, dum, 1, dum,
                     1, WORK.ptr(iwork));
        // If right singular vectors desired in VT, copy them there
        if (wntvas) Lacpy('F', n, n, a, lda, vt, ldvt);
      } else if (wntuo && wntvn) {
        // Path 2 (M much larger than N, JOBU='O', JOBVT='N')
        // N left singular vectors to be overwritten on A and
        // no right singular vectors to be computed
        if (lwork >= n * n + std::max<blasint>(4 * n, bdspac)) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n + n) + lda * n) {
            // WORK(IU) is LDA by N, WORK(IR) is LDA by N
            ldwrku = lda;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n + n) + n * n) {
            // WORK(IU) is LDA by N, WORK(IR) is N by N
            ldwrku = lda;
            ldwrkr = n;
          } else {
            // WORK(IU) is LDWRKU by N, WORK(IR) is N by N
            ldwrku = (lwork - n * n - n) / n;
            ldwrkr = n;
          }
          itau = ir + ldwrkr * n;
          iwork = itau + n;
          // Compute A=Q*R
          // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
          Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy R to WORK(IR) and zero out below it
          Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
          Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(ir + 1), ldwrkr);
          // Generate Q in A
          // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
          Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize R in WORK(IR)
          // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
          Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing R
          // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
          Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                WORK.ptr(iwork), lwork - iwork + 1);
          iwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IR)
          // (Workspace: need N*N + BDSPAC)
          info = Bdsqr('U', n, 0, n, 0, s, WORK.ptr(ie), dum, 1, WORK.ptr(ir),
                       ldwrkr, dum, 1, WORK.ptr(iwork));
          iu = ie + n;
          // Multiply Q in A by left singular vectors of R in
          // WORK(IR), storing result in WORK(IU) and copying to A
          // (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
          for (i = 1; i <= m; i += ldwrku) {
            chunk = std::min<blasint>(m - i + 1, ldwrku);
            Gemm('N', 'N', chunk, n, n, one, A.ptr(i, 1), lda, WORK.ptr(ir),
                 ldwrkr, zero, WORK.ptr(iu), ldwrku);
            Lacpy('F', chunk, n, WORK.ptr(iu), ldwrku, A.ptr(i, 1), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          ie = 1;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize A
          // (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
          Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing A
          // (Workspace: need 4*N, prefer 3*N + N*NB)
          Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
          iwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in A
          // (Workspace: need BDSPAC)
          info = Bdsqr('U', n, 0, m, 0, s, WORK.ptr(ie), dum, 1, a, lda, dum, 1,
                       WORK.ptr(iwork));
        }
      } else if (wntuo && wntvas) {
        // Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
        // N left singular vectors to be overwritten on A and
        // N right singular vectors to be computed in VT
        if (lwork >= n * n + std::max<blasint>(4 * n, bdspac)) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n + n) + lda * n) {
            // WORK(IU) is LDA by N and WORK(IR) is LDA by N
            ldwrku = lda;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n + n) + n * n) {
            // WORK(IU) is LDA by N and WORK(IR) is N by N
            ldwrku = lda;
            ldwrkr = n;
          } else {
            // WORK(IU) is LDWRKU by N and WORK(IR) is N by N
            ldwrku = (lwork - n * n - n) / n;
            ldwrkr = n;
          }
          itau = ir + ldwrkr * n;
          iwork = itau + n;
          // Compute A=Q*R
          // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
          Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy R to VT, zeroing out below it
          Lacpy('U', n, n, a, lda, vt, ldvt);
          if (n > 1) Laset('L', n - 1, n - 1, zero, zero, VT.ptr(2, 1), ldvt);
          // Generate Q in A
          // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
          Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize R in VT, copying result to WORK(IR)
          // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
          Gebrd(n, n, vt, ldvt, s, WORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          Lacpy('L', n, n, vt, ldvt, WORK.ptr(ir), ldwrkr);
          // Generate left vectors bidiagonalizing R in WORK(IR)
          // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
          Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing R in VT
          // (Workspace: need N*N + 4*N-1, prefer N*N + 3*N + (N-1)*NB)
          Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          iwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IR) and computing right
          // singular vectors of R in VT
          // (Workspace: need N*N + BDSPAC)
          info = Bdsqr('U', n, n, n, 0, s, WORK.ptr(ie), vt, ldvt, WORK.ptr(ir),
                       ldwrkr, dum, 1, WORK.ptr(iwork));
          iu = ie + n;
          // Multiply Q in A by left singular vectors of R in
          // WORK(IR), storing result in WORK(IU) and copying to A
          // (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
          for (i = 1; i <= m; i += ldwrku) {
            chunk = std::min<blasint>(m - i + 1, ldwrku);
            Gemm('N', 'N', chunk, n, n, one, A.ptr(i, 1), lda, WORK.ptr(ir),
                 ldwrkr, zero, WORK.ptr(iu), ldwrku);
            Lacpy('F', chunk, n, WORK.ptr(iu), ldwrku, A.ptr(i, 1), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          itau = 1;
          iwork = itau + n;
          // Compute A=Q*R
          // (Workspace: need 2*N, prefer N + N*NB)
          Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy R to VT, zeroing out below it
          Lacpy('U', n, n, a, lda, vt, ldvt);
          if (n > 1) Laset('L', n - 1, n - 1, zero, zero, VT.ptr(2, 1), ldvt);
          // Generate Q in A
          // (Workspace: need 2*N, prefer N + N*NB)
          Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = itau;
          itauq = ie + n;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize R in VT
          // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
          Gebrd(n, n, vt, ldvt, s, WORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Multiply Q in A by left vectors bidiagonalizing R
          // (Workspace: need 3*N + M, prefer 3*N + M*NB)
          Unmbr('Q', 'R', 'N', m, n, n, vt, ldvt, WORK.ptr(itauq), a, lda,
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing R in VT
          // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
          Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          iwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in A and computing right
          // singular vectors of A in VT
          // (Workspace: need BDSPAC)
          info = Bdsqr('U', n, n, m, 0, s, WORK.ptr(ie), vt, ldvt, a, lda, dum,
                       1, WORK.ptr(iwork));
        }
      } else if (wntus) {
        if (wntvn) {
          // Path 4 (M much larger than N, JOBU='S', JOBVT='N')
          // N left singular vectors to be computed in U and
          // no right singular vectors to be computed
          if (lwork >= n * n + std::max<blasint>(4 * n, bdspac)) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IR) is LDA by N
              ldwrkr = lda;
            } else {
              // WORK(IR) is N by N
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R
            // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IR), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(ir + 1), ldwrkr);
            // Generate Q in A
            // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
            Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IR)
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
            Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left vectors bidiagonalizing R in WORK(IR)
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
            Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IR)
            // (Workspace: need N*N + BDSPAC)
            info = Bdsqr('U', n, 0, n, 0, s, WORK.ptr(ie), dum, 1, WORK.ptr(ir),
                         ldwrkr, dum, 1, WORK.ptr(iwork));
            // Multiply Q in A by left singular vectors of R in
            // WORK(IR), storing result in U
            // (Workspace: need N*N)
            Gemm('N', 'N', m, n, n, one, a, lda, WORK.ptr(ir), ldwrkr, zero, u,
                 ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N, prefer N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need 2*N, prefer N + N*NB)
            Ungqr(m, n, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
            Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left vectors bidiagonalizing R
            // (Workspace: need 3*N + M, prefer 3*N + M*NB)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', n, 0, m, 0, s, WORK.ptr(ie), dum, 1, u, ldu, dum,
                         1, WORK.ptr(iwork));
          }
        } else if (wntvo) {
          // Path 5 (M much larger than N, JOBU='S', JOBVT='O')
          // N left singular vectors to be computed in U and
          // N right singular vectors to be overwritten on A
          if (lwork >= 2 * n * n + std::max<blasint>(4 * n, bdspac)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * n) {
              // WORK(IU) is LDA by N and WORK(IR) is LDA by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + n) * n) {
              // WORK(IU) is LDA by N and WORK(IR) is N by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            } else {
              // WORK(IU) is N by N and WORK(IR) is N by N
              ldwrku = n;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R
            // (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(iu + 1), ldwrku);
            // Generate Q in A
            // (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
            Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to
            // WORK(IR)
            // (Workspace: need 2*N*N + 4*N,
            // prefer 2*N*N+3*N+2*N*NB)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in WORK(IR)
            // (Workspace: need 2*N*N + 4*N-1,
            // prefer 2*N*N+3*N+(N-1)*NB)
            Ungbr('P', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in WORK(IR)
            // (Workspace: need 2*N*N + BDSPAC)
            info = Bdsqr('U', n, n, n, 0, s, WORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                         WORK.ptr(iu), ldwrku, dum, 1, WORK.ptr(iwork));
            // Multiply Q in A by left singular vectors of R in
            // WORK(IU), storing result in U
            // (Workspace: need N*N)
            Gemm('N', 'N', m, n, n, one, a, lda, WORK.ptr(iu), ldwrku, zero, u,
                 ldu);
            // Copy right singular vectors of R to A
            // (Workspace: need N*N)
            Lacpy('F', n, n, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N, prefer N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need 2*N, prefer N + N*NB)
            Ungqr(m, n, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
            Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left vectors bidiagonalizing R
            // (Workspace: need 3*N + M, prefer 3*N + M*NB)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right vectors bidiagonalizing R in A
            // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
            Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in A
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, WORK.ptr(ie), a, lda, u, ldu, dum,
                         1, WORK.ptr(iwork));
          }
        } else if (wntvas) {
          // Path 6 (M much larger than N, JOBU='S', JOBVT='S'
          // or 'A')
          // N left singular vectors to be computed in U and
          // N right singular vectors to be computed in VT
          if (lwork >= n * n + std::max<blasint>(4 * n, bdspac)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IU) is LDA by N
              ldwrku = lda;
            } else {
              // WORK(IU) is N by N
              ldwrku = n;
            }
            itau = iu + ldwrku * n;
            iwork = itau + n;
            // Compute A=Q*R
            // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(iu + 1), ldwrku);
            // Generate Q in A
            // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
            Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to VT
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, vt, ldvt);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (Workspace: need N*N + 4*N-1,
            // prefer N*N+3*N+(N-1)*NB)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in VT
            // (Workspace: need N*N + BDSPAC)
            info = Bdsqr('U', n, n, n, 0, s, WORK.ptr(ie), vt, ldvt,
                         WORK.ptr(iu), ldwrku, dum, 1, WORK.ptr(iwork));
            // Multiply Q in A by left singular vectors of R in
            // WORK(IU), storing result in U
            // (Workspace: need N*N)
            Gemm('N', 'N', m, n, n, one, a, lda, WORK.ptr(iu), ldwrku, zero, u,
                 ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N, prefer N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need 2*N, prefer N + N*NB)
            Ungqr(m, n, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to VT, zeroing out below it
            Lacpy('U', n, n, a, lda, vt, ldvt);
            if (n > 1) Laset('L', n - 1, n - 1, zero, zero, VT.ptr(2, 1), ldvt);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in VT
            // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
            Gebrd(n, n, vt, ldvt, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in VT
            // (Workspace: need 3*N + M, prefer 3*N + M*NB)
            Unmbr('Q', 'R', 'N', m, n, n, vt, ldvt, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, WORK.ptr(ie), vt, ldvt, u, ldu,
                         dum, 1, WORK.ptr(iwork));
          }
        }
      } else if (wntua) {
        if (wntvn) {
          // Path 7 (M much larger than N, JOBU='A', JOBVT='N')
          // M left singular vectors to be computed in U and
          // no right singular vectors to be computed
          if (lwork >= n * n + std::max<blasint>({n + m, 4 * n, bdspac})) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IR) is LDA by N
              ldwrkr = lda;
            } else {
              // WORK(IR) is N by N
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Copy R to WORK(IR), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(ir + 1), ldwrkr);
            // Generate Q in U
            // (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IR)
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
            Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in WORK(IR)
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
            Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IR)
            // (Workspace: need N*N + BDSPAC)
            info = Bdsqr('U', n, 0, n, 0, s, WORK.ptr(ie), dum, 1, WORK.ptr(ir),
                         ldwrkr, dum, 1, WORK.ptr(iwork));
            // Multiply Q in U by left singular vectors of R in
            // WORK(IR), storing result in A
            // (Workspace: need N*N)
            Gemm('N', 'N', m, n, n, one, u, ldu, WORK.ptr(ir), ldwrkr, zero, a,
                 lda);
            // Copy left singular vectors of A from A to U
            Lacpy('F', m, n, a, lda, u, ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N, prefer N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need N + M, prefer N + M*NB)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
            Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in A
            // (Workspace: need 3*N + M, prefer 3*N + M*NB)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', n, 0, m, 0, s, WORK.ptr(ie), dum, 1, u, ldu, dum,
                         1, WORK.ptr(iwork));
          }
        } else if (wntvo) {
          // Path 8 (M much larger than N, JOBU='A', JOBVT='O')
          // M left singular vectors to be computed in U and
          // N right singular vectors to be overwritten on A
          if (lwork >= 2 * n * n + std::max<blasint>({n + m, 4 * n, bdspac})) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * n) {
              // WORK(IU) is LDA by N and WORK(IR) is LDA by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + n) * n) {
              // WORK(IU) is LDA by N and WORK(IR) is N by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            } else {
              // WORK(IU) is N by N and WORK(IR) is N by N
              ldwrku = n;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need 2*N*N + N + M, prefer 2*N*N + N + M*NB)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(iu + 1), ldwrku);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to
            // WORK(IR)
            // (Workspace: need 2*N*N + 4*N,
            // prefer 2*N*N+3*N+2*N*NB)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in WORK(IR)
            // (Workspace: need 2*N*N + 4*N-1,
            // prefer 2*N*N+3*N+(N-1)*NB)
            Ungbr('P', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in WORK(IR)
            // (Workspace: need 2*N*N + BDSPAC)
            info = Bdsqr('U', n, n, n, 0, s, WORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                         WORK.ptr(iu), ldwrku, dum, 1, WORK.ptr(iwork));
            // Multiply Q in U by left singular vectors of R in
            // WORK(IU), storing result in A
            // (Workspace: need N*N)
            Gemm('N', 'N', m, n, n, one, u, ldu, WORK.ptr(iu), ldwrku, zero, a,
                 lda);
            // Copy left singular vectors of A from A to U
            Lacpy('F', m, n, a, lda, u, ldu);
            // Copy right singular vectors of R from WORK(IR) to A
            Lacpy('F', n, n, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N, prefer N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need N + M, prefer N + M*NB)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
            Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in A
            // (Workspace: need 3*N + M, prefer 3*N + M*NB)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in A
            // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
            Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in A
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, WORK.ptr(ie), a, lda, u, ldu, dum,
                         1, WORK.ptr(iwork));
          }
        } else if (wntvas) {
          // Path 9 (M much larger than N, JOBU='A', JOBVT='S'
          // or 'A')
          // M left singular vectors to be computed in U and
          // N right singular vectors to be computed in VT
          if (lwork >= n * n + std::max<blasint>({n + m, 4 * n, bdspac})) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IU) is LDA by N
              ldwrku = lda;
            } else {
              // WORK(IU) is N by N
              ldwrku = n;
            }
            itau = iu + ldwrku * n;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(iu + 1), ldwrku);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to VT
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, vt, ldvt);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (Workspace: need N*N + 4*N-1,
            // prefer N*N+3*N+(N-1)*NB)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in VT
            // (Workspace: need N*N + BDSPAC)
            info = Bdsqr('U', n, n, n, 0, s, WORK.ptr(ie), vt, ldvt,
                         WORK.ptr(iu), ldwrku, dum, 1, WORK.ptr(iwork));
            // Multiply Q in U by left singular vectors of R in
            // WORK(IU), storing result in A
            // (Workspace: need N*N)
            Gemm('N', 'N', m, n, n, one, u, ldu, WORK.ptr(iu), ldwrku, zero, a,
                 lda);
            // Copy left singular vectors of A from A to U
            Lacpy('F', m, n, a, lda, u, ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (Workspace: need 2*N, prefer N + N*NB)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (Workspace: need N + M, prefer N + M*NB)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R from A to VT, zeroing out below it
            Lacpy('U', n, n, a, lda, vt, ldvt);
            if (n > 1) Laset('L', n - 1, n - 1, zero, zero, VT.ptr(2, 1), ldvt);
            ie = itau;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in VT
            // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
            Gebrd(n, n, vt, ldvt, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in VT
            // (Workspace: need 3*N + M, prefer 3*N + M*NB)
            Unmbr('Q', 'R', 'N', m, n, n, vt, ldvt, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, WORK.ptr(ie), vt, ldvt, u, ldu,
                         dum, 1, WORK.ptr(iwork));
          }
        }
      }
    } else {
      // M .LT. MNTHR
      // Path 10 (M at least N, but not much larger)
      // Reduce to bidiagonal form without QR decomposition
      ie = 1;
      itauq = ie + n;
      itaup = itauq + n;
      iwork = itaup + n;
      // Bidiagonalize A
      // (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
      Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(iwork), lwork - iwork + 1);
      if (wntuas) {
        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // (Workspace: need 3*N + NCU, prefer 3*N + NCU*NB)
        Lacpy('L', m, n, a, lda, u, ldu);
        if (wntus) ncu = n;
        if (wntua) ncu = m;
        Ungbr('Q', m, ncu, n, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvas) {
        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
        Lacpy('U', n, n, a, lda, vt, ldvt);
        Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntuo) {
        // If left singular vectors desired in A, generate left
        // bidiagonalizing vectors in A
        // (Workspace: need 4*N, prefer 3*N + N*NB)
        Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvo) {
        // If right singular vectors desired in A, generate right
        // bidiagonalizing vectors in A
        // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
        Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      iwork = ie + n;
      if (wntuas || wntuo) nru = m;
      if (wntun) nru = 0;
      if (wntvas || wntvo) ncvt = n;
      if (wntvn) ncvt = 0;
      if ((!wntuo) && (!wntvo)) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in VT
        // (Workspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, nru, 0, s, WORK.ptr(ie), vt, ldvt, u, ldu,
                     dum, 1, WORK.ptr(iwork));
      } else if ((!wntuo) && wntvo) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in A
        // (Workspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, nru, 0, s, WORK.ptr(ie), a, lda, u, ldu, dum,
                     1, WORK.ptr(iwork));
      } else {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in A and computing right singular
        // vectors in VT
        // (Workspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, nru, 0, s, WORK.ptr(ie), vt, ldvt, a, lda,
                     dum, 1, WORK.ptr(iwork));
      }
    }
  } else {
    // A has more columns than rows. If A has sufficiently more
    // columns than rows, first reduce using the LQ decomposition (if
    // sufficient workspace available)
    if (n >= mnthr) {
      if (wntvn) {
        // Path 1t(N much larger than M, JOBVT='N')
        // No right singular vectors to be computed
        itau = 1;
        iwork = itau + m;
        // Compute A=L*Q
        // (Workspace: need 2*M, prefer M + M*NB)
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork), lwork - iwork + 1);
        // Zero out above L
        Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
        ie = 1;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;
        // Bidiagonalize L in A
        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
        Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(iwork), lwork - iwork + 1);
        if (wntuo || wntuas) {
          // If left singular vectors desired, generate Q
          // (Workspace: need 4*M, prefer 3*M + M*NB)
          Ungbr('Q', m, m, m, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
        }
        iwork = ie + m;
        nru = 0;
        if (wntuo || wntuas) nru = m;
        // Perform bidiagonal QR iteration, computing left singular
        // vectors of A in A if desired
        // (Workspace: need BDSPAC)
        info = Bdsqr('U', m, 0, nru, 0, s, WORK.ptr(ie), dum, 1, a, lda, dum, 1,
                     WORK.ptr(iwork));
        // If left singular vectors desired in U, copy them there
        if (wntuas) Lacpy('F', m, m, a, lda, u, ldu);
      } else if (wntvo && wntun) {
        // Path 2t(N much larger than M, JOBU='N', JOBVT='O')
        // M right singular vectors to be overwritten on A and
        // no left singular vectors to be computed
        if (lwork >= m * m + std::max<blasint>(4 * m, bdspac)) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n + m) + lda * m) {
            // WORK(IU) is LDA by N and WORK(IR) is LDA by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n + m) + m * m) {
            // WORK(IU) is LDA by N and WORK(IR) is M by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = m;
          } else {
            // WORK(IU) is M by CHUNK and WORK(IR) is M by M
            ldwrku = m;
            chunk = (lwork - m * m - m) / m;
            ldwrkr = m;
          }
          itau = ir + ldwrkr * m;
          iwork = itau + m;
          // Compute A=L*Q
          // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
          Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy L to WORK(IR) and zero out above it
          Lacpy('L', m, m, a, lda, WORK.ptr(ir), ldwrkr);
          Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(ir + ldwrkr), ldwrkr);
          // Generate Q in A
          // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
          Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize L in WORK(IR)
          // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
          Gebrd(m, m, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing L
          // (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
          Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          iwork = ie + m;
          // Perform bidiagonal QR iteration, computing right
          // singular vectors of L in WORK(IR)
          // (Workspace: need M*M + BDSPAC)
          info = Bdsqr('U', m, m, 0, 0, s, WORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                       dum, 1, dum, 1, WORK.ptr(iwork));
          iu = ie + m;
          // Multiply right singular vectors of L in WORK(IR) by Q
          // in A, storing result in WORK(IU) and copying to A
          // (Workspace: need M*M + 2*M, prefer M*M + M*N + M)
          for (i = 1; i <= n; i += chunk) {
            blk = std::min<blasint>(n - i + 1, chunk);
            Gemm('N', 'N', m, blk, m, one, WORK.ptr(ir), ldwrkr, A.ptr(1, i),
                 lda, zero, WORK.ptr(iu), ldwrku);
            Lacpy('F', m, blk, WORK.ptr(iu), ldwrku, A.ptr(1, i), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          ie = 1;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize A
          // (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
          Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing A
          // (Workspace: need 4*M, prefer 3*M + M*NB)
          Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          iwork = ie + m;
          // Perform bidiagonal QR iteration, computing right
          // singular vectors of A in A
          // (Workspace: need BDSPAC)
          info = Bdsqr('L', m, n, 0, 0, s, WORK.ptr(ie), a, lda, dum, 1, dum, 1,
                       WORK.ptr(iwork));
        }
      } else if (wntvo && wntuas) {
        // Path 3t(N much larger than M, JOBU='S' or 'A', JOBVT='O')
        // M right singular vectors to be overwritten on A and
        // M left singular vectors to be computed in U
        if (lwork >= m * m + std::max<blasint>(4 * m, bdspac)) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n + m) + lda * m) {
            // WORK(IU) is LDA by N and WORK(IR) is LDA by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n + m) + m * m) {
            // WORK(IU) is LDA by N and WORK(IR) is M by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = m;
          } else {
            // WORK(IU) is M by CHUNK and WORK(IR) is M by M
            ldwrku = m;
            chunk = (lwork - m * m - m) / m;
            ldwrkr = m;
          }
          itau = ir + ldwrkr * m;
          iwork = itau + m;
          // Compute A=L*Q
          // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
          Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy L to U, zeroing about above it
          Lacpy('L', m, m, a, lda, u, ldu);
          Laset('U', m - 1, m - 1, zero, zero, U.ptr(1, 2), ldu);
          // Generate Q in A
          // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
          Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize L in U, copying result to WORK(IR)
          // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
          Gebrd(m, m, u, ldu, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          Lacpy('U', m, m, u, ldu, WORK.ptr(ir), ldwrkr);
          // Generate right vectors bidiagonalizing L in WORK(IR)
          // (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
          Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing L in U
          // (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
          Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
          iwork = ie + m;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in U, and computing right
          // singular vectors of L in WORK(IR)
          // (Workspace: need M*M + BDSPAC)
          info = Bdsqr('U', m, m, m, 0, s, WORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                       u, ldu, dum, 1, WORK.ptr(iwork));
          iu = ie + m;
          // Multiply right singular vectors of L in WORK(IR) by Q
          // in A, storing result in WORK(IU) and copying to A
          // (Workspace: need M*M + 2*M, prefer M*M + M*N + M))
          for (i = 1; i <= n; i += chunk) {
            blk = std::min<blasint>(n - i + 1, chunk);
            Gemm('N', 'N', m, blk, m, one, WORK.ptr(ir), ldwrkr, A.ptr(1, i),
                 lda, zero, WORK.ptr(iu), ldwrku);
            Lacpy('F', m, blk, WORK.ptr(iu), ldwrku, A.ptr(1, i), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          itau = 1;
          iwork = itau + m;
          // Compute A=L*Q
          // (Workspace: need 2*M, prefer M + M*NB)
          Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy L to U, zeroing out above it
          Lacpy('L', m, m, a, lda, u, ldu);
          Laset('U', m - 1, m - 1, zero, zero, U.ptr(1, 2), ldu);
          // Generate Q in A
          // (Workspace: need 2*M, prefer M + M*NB)
          Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize L in U
          // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
          Gebrd(m, m, u, ldu, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Multiply right vectors bidiagonalizing L by Q in A
          // (Workspace: need 3*M + N, prefer 3*M + N*NB)
          Unmbr('P', 'L', 'T', m, n, m, u, ldu, WORK.ptr(itaup), a, lda,
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing L in U
          // (Workspace: need 4*M, prefer 3*M + M*NB)
          Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
          iwork = ie + m;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in A
          // (Workspace: need BDSPAC)
          info = Bdsqr('U', m, n, m, 0, s, WORK.ptr(ie), a, lda, u, ldu, dum, 1,
                       WORK.ptr(iwork));
        }
      } else if (wntvs) {
        if (wntun) {
          // Path 4t(N much larger than M, JOBU='N', JOBVT='S')
          // M right singular vectors to be computed in VT and
          // no left singular vectors to be computed
          if (lwork >= m * m + std::max<blasint>(4 * m, bdspac)) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IR) is LDA by M
              ldwrkr = lda;
            } else {
              // WORK(IR) is M by M
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q
            // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IR), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(ir + ldwrkr), ldwrkr);
            // Generate Q in A
            // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
            Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IR)
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
            Gebrd(m, m, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right vectors bidiagonalizing L in
            // WORK(IR)
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
            Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of L in WORK(IR)
            // (Workspace: need M*M + BDSPAC)
            info = Bdsqr('U', m, m, 0, 0, s, WORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                         dum, 1, dum, 1, WORK.ptr(iwork));
            // Multiply right singular vectors of L in WORK(IR) by
            // Q in A, storing result in VT
            // (Workspace: need M*M)
            Gemm('N', 'N', m, n, m, one, WORK.ptr(ir), ldwrkr, a, lda, zero, vt,
                 ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q
            // (Workspace: need 2*M, prefer M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy result to VT
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Unglq(m, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
            Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right vectors bidiagonalizing L by Q in VT
            // (Workspace: need 3*M + N, prefer 3*M + N*NB)
            Unmbr('P', 'L', 'T', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', m, n, 0, 0, s, WORK.ptr(ie), vt, ldvt, dum, 1,
                         dum, 1, WORK.ptr(iwork));
          }
        } else if (wntuo) {
          // Path 5t(N much larger than M, JOBU='O', JOBVT='S')
          // M right singular vectors to be computed in VT and
          // M left singular vectors to be overwritten on A
          if (lwork >= 2 * m * m + std::max<blasint>(4 * m, bdspac)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * m) {
              // WORK(IU) is LDA by M and WORK(IR) is LDA by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + m) * m) {
              // WORK(IU) is LDA by M and WORK(IR) is M by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            } else {
              // WORK(IU) is M by M and WORK(IR) is M by M
              ldwrku = m;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q
            // (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out below it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(iu + ldwrku), ldwrku);
            // Generate Q in A
            // (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
            Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to
            // WORK(IR)
            // (Workspace: need 2*M*M + 4*M,
            // prefer 2*M*M+3*M+2*M*NB)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (Workspace: need 2*M*M + 4*M-1,
            // prefer 2*M*M+3*M+(M-1)*NB)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in WORK(IR)
            // (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
            Ungbr('Q', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in WORK(IR) and computing
            // right singular vectors of L in WORK(IU)
            // (Workspace: need 2*M*M + BDSPAC)
            info = Bdsqr('U', m, m, m, 0, s, WORK.ptr(ie), WORK.ptr(iu), ldwrku,
                         WORK.ptr(ir), ldwrkr, dum, 1, WORK.ptr(iwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in A, storing result in VT
            // (Workspace: need M*M)
            Gemm('N', 'N', m, n, m, one, WORK.ptr(iu), ldwrku, a, lda, zero, vt,
                 ldvt);
            // Copy left singular vectors of L to A
            // (Workspace: need M*M)
            Lacpy('F', m, m, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Unglq(m, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
            Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right vectors bidiagonalizing L by Q in VT
            // (Workspace: need 3*M + N, prefer 3*M + N*NB)
            Unmbr('P', 'L', 'T', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors of L in A
            // (Workspace: need 4*M, prefer 3*M + M*NB)
            Ungbr('Q', m, m, m, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, compute left
            // singular vectors of A in A and compute right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, WORK.ptr(ie), vt, ldvt, a, lda,
                         dum, 1, WORK.ptr(iwork));
          }
        } else if (wntuas) {
          // Path 6t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='S')
          // M right singular vectors to be computed in VT and
          // M left singular vectors to be computed in U
          if (lwork >= m * m + std::max<blasint>(4 * m, bdspac)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IU) is LDA by N
              ldwrku = lda;
            } else {
              // WORK(IU) is LDA by M
              ldwrku = m;
            }
            itau = iu + ldwrku * m;
            iwork = itau + m;
            // Compute A=L*Q
            // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(iu + ldwrku), ldwrku);
            // Generate Q in A
            // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
            Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to U
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, u, ldu);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (Workspace: need M*M + 4*M-1,
            // prefer M*M+3*M+(M-1)*NB)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in U and computing right
            // singular vectors of L in WORK(IU)
            // (Workspace: need M*M + BDSPAC)
            info = Bdsqr('U', m, m, m, 0, s, WORK.ptr(ie), WORK.ptr(iu), ldwrku,
                         u, ldu, dum, 1, WORK.ptr(iwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in A, storing result in VT
            // (Workspace: need M*M)
            Gemm('N', 'N', m, n, m, one, WORK.ptr(iu), ldwrku, a, lda, zero, vt,
                 ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Unglq(m, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to U, zeroing out above it
            Lacpy('L', m, m, a, lda, u, ldu);
            Laset('U', m - 1, m - 1, zero, zero, U.ptr(1, 2), ldu);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in U
            // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
            Gebrd(m, m, u, ldu, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in U by Q
            // in VT
            // (Workspace: need 3*M + N, prefer 3*M + N*NB)
            Unmbr('P', 'L', 'T', m, n, m, u, ldu, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (Workspace: need 4*M, prefer 3*M + M*NB)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, WORK.ptr(ie), vt, ldvt, u, ldu,
                         dum, 1, WORK.ptr(iwork));
          }
        }
      } else if (wntva) {
        if (wntun) {
          // Path 7t(N much larger than M, JOBU='N', JOBVT='A')
          // N right singular vectors to be computed in VT and
          // no left singular vectors to be computed
          if (lwork >= m * m + std::max<blasint>({n + m, 4 * m, bdspac})) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IR) is LDA by M
              ldwrkr = lda;
            } else {
              // WORK(IR) is M by M
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Copy L to WORK(IR), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(ir + ldwrkr), ldwrkr);
            // Generate Q in VT
            // (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IR)
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
            Gebrd(m, m, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in WORK(IR)
            // (Workspace: need M*M + 4*M-1,
            // prefer M*M+3*M+(M-1)*NB)
            Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of L in WORK(IR)
            // (Workspace: need M*M + BDSPAC)
            info = Bdsqr('U', m, m, 0, 0, s, WORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                         dum, 1, dum, 1, WORK.ptr(iwork));
            // Multiply right singular vectors of L in WORK(IR) by
            // Q in VT, storing result in A
            // (Workspace: need M*M)
            Gemm('N', 'N', m, n, m, one, WORK.ptr(ir), ldwrkr, vt, ldvt, zero,
                 a, lda);
            // Copy right singular vectors of A from A to VT
            Lacpy('F', m, n, a, lda, vt, ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need M + N, prefer M + N*NB)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
            Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in A by Q
            // in VT
            // (Workspace: need 3*M + N, prefer 3*M + N*NB)
            Unmbr('P', 'L', 'T', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', m, n, 0, 0, s, WORK.ptr(ie), vt, ldvt, dum, 1,
                         dum, 1, WORK.ptr(iwork));
          }
        } else if (wntuo) {
          // Path 8t(N much larger than M, JOBU='O', JOBVT='A')
          // N right singular vectors to be computed in VT and
          // M left singular vectors to be overwritten on A
          if (lwork >= 2 * m * m + std::max<blasint>({n + m, 4 * m, bdspac})) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * m) {
              // WORK(IU) is LDA by M and WORK(IR) is LDA by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + m) * m) {
              // WORK(IU) is LDA by M and WORK(IR) is M by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            } else {
              // WORK(IU) is M by M and WORK(IR) is M by M
              ldwrku = m;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need 2*M*M + M + N, prefer 2*M*M + M + N*NB)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(iu + ldwrku), ldwrku);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to
            // WORK(IR)
            // (Workspace: need 2*M*M + 4*M,
            // prefer 2*M*M+3*M+2*M*NB)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (Workspace: need 2*M*M + 4*M-1,
            // prefer 2*M*M+3*M+(M-1)*NB)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in WORK(IR)
            // (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
            Ungbr('Q', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in WORK(IR) and computing
            // right singular vectors of L in WORK(IU)
            // (Workspace: need 2*M*M + BDSPAC)
            info = Bdsqr('U', m, m, m, 0, s, WORK.ptr(ie), WORK.ptr(iu), ldwrku,
                         WORK.ptr(ir), ldwrkr, dum, 1, WORK.ptr(iwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in VT, storing result in A
            // (Workspace: need M*M)
            Gemm('N', 'N', m, n, m, one, WORK.ptr(iu), ldwrku, vt, ldvt, zero,
                 a, lda);
            // Copy right singular vectors of A from A to VT
            Lacpy('F', m, n, a, lda, vt, ldvt);
            // Copy left singular vectors of A from WORK(IR) to A
            Lacpy('F', m, m, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need M + N, prefer M + N*NB)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
            Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in A by Q
            // in VT
            // (Workspace: need 3*M + N, prefer 3*M + N*NB)
            Unmbr('P', 'L', 'T', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in A
            // (Workspace: need 4*M, prefer 3*M + M*NB)
            Ungbr('Q', m, m, m, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in A and computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, WORK.ptr(ie), vt, ldvt, a, lda,
                         dum, 1, WORK.ptr(iwork));
          }
        } else if (wntuas) {
          // Path 9t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='A')
          // N right singular vectors to be computed in VT and
          // M left singular vectors to be computed in U
          if (lwork >= m * m + std::max<blasint>({n + m, 4 * m, bdspac})) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IU) is LDA by M
              ldwrku = lda;
            } else {
              // WORK(IU) is M by M
              ldwrku = m;
            }
            itau = iu + ldwrku * m;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(iu + ldwrku), ldwrku);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to U
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, u, ldu);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in U and computing right
            // singular vectors of L in WORK(IU)
            // (Workspace: need M*M + BDSPAC)
            info = Bdsqr('U', m, m, m, 0, s, WORK.ptr(ie), WORK.ptr(iu), ldwrku,
                         u, ldu, dum, 1, WORK.ptr(iwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in VT, storing result in A
            // (Workspace: need M*M)
            Gemm('N', 'N', m, n, m, one, WORK.ptr(iu), ldwrku, vt, ldvt, zero,
                 a, lda);
            // Copy right singular vectors of A from A to VT
            Lacpy('F', m, n, a, lda, vt, ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (Workspace: need 2*M, prefer M + M*NB)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (Workspace: need M + N, prefer M + N*NB)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to U, zeroing out above it
            Lacpy('L', m, m, a, lda, u, ldu);
            Laset('U', m - 1, m - 1, zero, zero, U.ptr(1, 2), ldu);
            ie = itau;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in U
            // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
            Gebrd(m, m, u, ldu, s, WORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in U by Q
            // in VT
            // (Workspace: need 3*M + N, prefer 3*M + N*NB)
            Unmbr('P', 'L', 'T', m, n, m, u, ldu, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (Workspace: need 4*M, prefer 3*M + M*NB)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            iwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (Workspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, WORK.ptr(ie), vt, ldvt, u, ldu,
                         dum, 1, WORK.ptr(iwork));
          }
        }
      }
    } else {
      // N .LT. MNTHR
      // Path 10t(N greater than M, but not much larger)
      // Reduce to bidiagonal form without LQ decomposition
      ie = 1;
      itauq = ie + m;
      itaup = itauq + m;
      iwork = itaup + m;
      // Bidiagonalize A
      // (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
      Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(iwork), lwork - iwork + 1);
      if (wntuas) {
        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
        Lacpy('L', m, m, a, lda, u, ldu);
        Ungbr('Q', m, m, n, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvas) {
        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // (Workspace: need 3*M + NRVT, prefer 3*M + NRVT*NB)
        Lacpy('U', m, n, a, lda, vt, ldvt);
        if (wntva) nrvt = n;
        if (wntvs) nrvt = m;
        Ungbr('P', nrvt, n, m, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntuo) {
        // If left singular vectors desired in A, generate left
        // bidiagonalizing vectors in A
        // (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
        Ungbr('Q', m, m, n, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvo) {
        // If right singular vectors desired in A, generate right
        // bidiagonalizing vectors in A
        // (Workspace: need 4*M, prefer 3*M + M*NB)
        Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      iwork = ie + m;
      if (wntuas || wntuo) nru = m;
      if (wntun) nru = 0;
      if (wntvas || wntvo) ncvt = n;
      if (wntvn) ncvt = 0;
      if ((!wntuo) && (!wntvo)) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in VT
        // (Workspace: need BDSPAC)
        info = Bdsqr('L', m, ncvt, nru, 0, s, WORK.ptr(ie), vt, ldvt, u, ldu,
                     dum, 1, WORK.ptr(iwork));
      } else if ((!wntuo) && wntvo) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in A
        // (Workspace: need BDSPAC)
        info = Bdsqr('L', m, ncvt, nru, 0, s, WORK.ptr(ie), a, lda, u, ldu, dum,
                     1, WORK.ptr(iwork));
      } else {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in A and computing right singular
        // vectors in VT
        // (Workspace: need BDSPAC)
        info = Bdsqr('L', m, ncvt, nru, 0, s, WORK.ptr(ie), vt, ldvt, a, lda,
                     dum, 1, WORK.ptr(iwork));
      }
    }
  }
  // If DBDSQR failed to converge, copy unconverged superdiagonals
  // to WORK( 2:MINMN )
  if (info != 0) {
    if (ie > 2) {
      for (i = 1; i <= minmn - 1; ++i) {
        WORK(i + 1) = WORK(i + ie - 1);
      }
    }
    if (ie < 2) {
      for (i = minmn - 1; i >= 1; --i) {
        WORK(i + 1) = WORK(i + ie - 1);
      }
    }
  }
  // Undo scaling if necessary
  if (iscl == 1) {
    if (anrm > bignum) Lascl('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
    if (info != 0 && anrm > bignum)
      Lascl('G', 0, 0, bignum, anrm, minmn - 1, 1, WORK.ptr(2), minmn);
    if (anrm < smlnum) Lascl('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
    if (info != 0 && anrm < smlnum)
      Lascl('G', 0, 0, smlnum, anrm, minmn - 1, 1, WORK.ptr(2), minmn);
  }
  // Return optimal workspace in WORK(1)
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// CGESVD / ZGESVD.
template <typename T>
// NOLINTNEXTLINE(google-readability-function-size,readability-function-size)
blasint GesvdComplex(char jobu, char jobvt, blasint m, blasint n, T* a,
                     blasint lda, RealT<T>* s, T* u, blasint ldu, T* vt,
                     blasint ldvt, T* work, blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;
  const char opts[3] = {jobu, jobvt, '\0'};

  const T czero = T(0);
  const T cone = T(1);
  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> A(a, lda);
  T cdum[1];
  R dum[1];
  const Vector<R> RWORK(rwork);
  const Vector<R> S(s);
  const Matrix<T> U(u, ldu);
  const Matrix<T> VT(vt, ldvt);
  const Vector<T> WORK(work);
  blasint blk, chunk, i, ie, ir, irwork, iscl, itau, itaup, itauq, iu, iwork,
      ldwrkr, ldwrku, lwork_zgebrd, lwork_zgelqf, lwork_zgeqrf, lwork_zungbr_p,
      lwork_zungbr_q, lwork_zunglq_m, lwork_zunglq_n, lwork_zungqr_m,
      lwork_zungqr_n, maxwrk, minmn, minwrk, mnthr, ncu, ncvt, nru, nrvt, wrkbl;
  bool lquery, wntua, wntuas, wntun, wntuo, wntus, wntva, wntvas, wntvn, wntvo,
      wntvs;
  R anrm, bignum, eps, smlnum;
  blasint info = 0;
  minmn = std::min<blasint>(m, n);
  wntua = Lsame(jobu, 'A');
  wntus = Lsame(jobu, 'S');
  wntuas = wntua || wntus;
  wntuo = Lsame(jobu, 'O');
  wntun = Lsame(jobu, 'N');
  wntva = Lsame(jobvt, 'A');
  wntvs = Lsame(jobvt, 'S');
  wntvas = wntva || wntvs;
  wntvo = Lsame(jobvt, 'O');
  wntvn = Lsame(jobvt, 'N');
  lquery = (lwork == -1);
  if (!(wntua || wntus || wntuo || wntun)) {
    info = -1;
  } else if (!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo)) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < std::max<blasint>(1, m)) {
    info = -6;
  } else if (ldu < 1 || (wntuas && ldu < m)) {
    info = -9;
  } else if (ldvt < 1 || (wntva && ldvt < n) || (wntvs && ldvt < minmn)) {
    info = -11;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // CWorkspace refers to complex workspace, and RWorkspace to
  // real workspace. NB refers to the optimal block size for the
  // immediately following subroutine, as returned by ILAENV.)
  if (info == 0) {
    minwrk = 1;
    maxwrk = 1;
    if (m >= n && minmn > 0) {
      // Space needed for ZBDSQR is BDSPAC = 5*N
      mnthr = Ilaenv<T>(6, "GESVD", opts, m, n, 0, 0);
      // Compute space needed for ZGEQRF
      Geqrf(m, n, a, lda, cdum, cdum, -1);
      lwork_zgeqrf = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZUNGQR
      Ungqr(m, n, n, a, lda, cdum, cdum, -1);
      lwork_zungqr_n = static_cast<blasint>(std::real(cdum[0]));
      Ungqr(m, m, n, a, lda, cdum, cdum, -1);
      lwork_zungqr_m = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZGEBRD
      Gebrd(n, n, a, lda, s, dum, cdum, cdum, cdum, -1);
      lwork_zgebrd = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZUNGBR
      Ungbr('P', n, n, n, a, lda, cdum, cdum, -1);
      lwork_zungbr_p = static_cast<blasint>(std::real(cdum[0]));
      Ungbr('Q', n, n, n, a, lda, cdum, cdum, -1);
      lwork_zungbr_q = static_cast<blasint>(std::real(cdum[0]));
      if (m >= mnthr) {
        if (wntun) {
          // Path 1 (M much larger than N, JOBU='N')
          maxwrk = n + lwork_zgeqrf;
          maxwrk = std::max<blasint>(maxwrk, 2 * n + lwork_zgebrd);
          if (wntvo || wntvas)
            maxwrk = std::max<blasint>(maxwrk, 2 * n + lwork_zungbr_p);
          minwrk = 3 * n;
        } else if (wntuo && wntvn) {
          // Path 2 (M much larger than N, JOBU='O', JOBVT='N')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          maxwrk = std::max<blasint>(n * n + wrkbl, n * n + m * n);
          minwrk = 2 * n + m;
        } else if (wntuo && wntvas) {
          // Path 3 (M much larger than N, JOBU='O', JOBVT='S' or
          // 'A')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_p);
          maxwrk = std::max<blasint>(n * n + wrkbl, n * n + m * n);
          minwrk = 2 * n + m;
        } else if (wntus && wntvn) {
          // Path 4 (M much larger than N, JOBU='S', JOBVT='N')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          maxwrk = n * n + wrkbl;
          minwrk = 2 * n + m;
        } else if (wntus && wntvo) {
          // Path 5 (M much larger than N, JOBU='S', JOBVT='O')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_p);
          maxwrk = 2 * n * n + wrkbl;
          minwrk = 2 * n + m;
        } else if (wntus && wntvas) {
          // Path 6 (M much larger than N, JOBU='S', JOBVT='S' or
          // 'A')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_p);
          maxwrk = n * n + wrkbl;
          minwrk = 2 * n + m;
        } else if (wntua && wntvn) {
          // Path 7 (M much larger than N, JOBU='A', JOBVT='N')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          maxwrk = n * n + wrkbl;
          minwrk = 2 * n + m;
        } else if (wntua && wntvo) {
          // Path 8 (M much larger than N, JOBU='A', JOBVT='O')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_p);
          maxwrk = 2 * n * n + wrkbl;
          minwrk = 2 * n + m;
        } else if (wntua && wntvas) {
          // Path 9 (M much larger than N, JOBU='A', JOBVT='S' or
          // 'A')
          wrkbl = n + lwork_zgeqrf;
          wrkbl = std::max<blasint>(wrkbl, n + lwork_zungqr_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_q);
          wrkbl = std::max<blasint>(wrkbl, 2 * n + lwork_zungbr_p);
          maxwrk = n * n + wrkbl;
          minwrk = 2 * n + m;
        }
      } else {
        // Path 10 (M at least N, but not much larger)
        Gebrd(m, n, a, lda, s, dum, cdum, cdum, cdum, -1);
        lwork_zgebrd = static_cast<blasint>(std::real(cdum[0]));
        maxwrk = 2 * n + lwork_zgebrd;
        if (wntus || wntuo) {
          Ungbr('Q', m, n, n, a, lda, cdum, cdum, -1);
          lwork_zungbr_q = static_cast<blasint>(std::real(cdum[0]));
          maxwrk = std::max<blasint>(maxwrk, 2 * n + lwork_zungbr_q);
        }
        if (wntua) {
          Ungbr('Q', m, m, n, a, lda, cdum, cdum, -1);
          lwork_zungbr_q = static_cast<blasint>(std::real(cdum[0]));
          maxwrk = std::max<blasint>(maxwrk, 2 * n + lwork_zungbr_q);
        }
        if (!wntvn) {
          maxwrk = std::max<blasint>(maxwrk, 2 * n + lwork_zungbr_p);
        }
        minwrk = 2 * n + m;
      }
    } else if (minmn > 0) {
      // Space needed for ZBDSQR is BDSPAC = 5*M
      mnthr = Ilaenv<T>(6, "GESVD", opts, m, n, 0, 0);
      // Compute space needed for ZGELQF
      Gelqf(m, n, a, lda, cdum, cdum, -1);
      lwork_zgelqf = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZUNGLQ
      Unglq(n, n, m, cdum, n, cdum, cdum, -1);
      lwork_zunglq_n = static_cast<blasint>(std::real(cdum[0]));
      Unglq(m, n, m, a, lda, cdum, cdum, -1);
      lwork_zunglq_m = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZGEBRD
      Gebrd(m, m, a, lda, s, dum, cdum, cdum, cdum, -1);
      lwork_zgebrd = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZUNGBR P
      Ungbr('P', m, m, m, a, n, cdum, cdum, -1);
      lwork_zungbr_p = static_cast<blasint>(std::real(cdum[0]));
      // Compute space needed for ZUNGBR Q
      Ungbr('Q', m, m, m, a, n, cdum, cdum, -1);
      lwork_zungbr_q = static_cast<blasint>(std::real(cdum[0]));
      if (n >= mnthr) {
        if (wntvn) {
          // Path 1t(N much larger than M, JOBVT='N')
          maxwrk = m + lwork_zgelqf;
          maxwrk = std::max<blasint>(maxwrk, 2 * m + lwork_zgebrd);
          if (wntuo || wntuas)
            maxwrk = std::max<blasint>(maxwrk, 2 * m + lwork_zungbr_q);
          minwrk = 3 * m;
        } else if (wntvo && wntun) {
          // Path 2t(N much larger than M, JOBU='N', JOBVT='O')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          maxwrk = std::max<blasint>(m * m + wrkbl, m * m + m * n);
          minwrk = 2 * m + n;
        } else if (wntvo && wntuas) {
          // Path 3t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='O')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_q);
          maxwrk = std::max<blasint>(m * m + wrkbl, m * m + m * n);
          minwrk = 2 * m + n;
        } else if (wntvs && wntun) {
          // Path 4t(N much larger than M, JOBU='N', JOBVT='S')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          maxwrk = m * m + wrkbl;
          minwrk = 2 * m + n;
        } else if (wntvs && wntuo) {
          // Path 5t(N much larger than M, JOBU='O', JOBVT='S')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_q);
          maxwrk = 2 * m * m + wrkbl;
          minwrk = 2 * m + n;
        } else if (wntvs && wntuas) {
          // Path 6t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='S')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_m);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_q);
          maxwrk = m * m + wrkbl;
          minwrk = 2 * m + n;
        } else if (wntva && wntun) {
          // Path 7t(N much larger than M, JOBU='N', JOBVT='A')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          maxwrk = m * m + wrkbl;
          minwrk = 2 * m + n;
        } else if (wntva && wntuo) {
          // Path 8t(N much larger than M, JOBU='O', JOBVT='A')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_q);
          maxwrk = 2 * m * m + wrkbl;
          minwrk = 2 * m + n;
        } else if (wntva && wntuas) {
          // Path 9t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='A')
          wrkbl = m + lwork_zgelqf;
          wrkbl = std::max<blasint>(wrkbl, m + lwork_zunglq_n);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zgebrd);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_p);
          wrkbl = std::max<blasint>(wrkbl, 2 * m + lwork_zungbr_q);
          maxwrk = m * m + wrkbl;
          minwrk = 2 * m + n;
        }
      } else {
        // Path 10t(N greater than M, but not much larger)
        Gebrd(m, n, a, lda, s, dum, cdum, cdum, cdum, -1);
        lwork_zgebrd = static_cast<blasint>(std::real(cdum[0]));
        maxwrk = 2 * m + lwork_zgebrd;
        if (wntvs || wntvo) {
          // Compute space needed for ZUNGBR P
          Ungbr('P', m, n, m, a, n, cdum, cdum, -1);
          lwork_zungbr_p = static_cast<blasint>(std::real(cdum[0]));
          maxwrk = std::max<blasint>(maxwrk, 2 * m + lwork_zungbr_p);
        }
        if (wntva) {
          Ungbr('P', n, n, m, a, n, cdum, cdum, -1);
          lwork_zungbr_p = static_cast<blasint>(std::real(cdum[0]));
          maxwrk = std::max<blasint>(maxwrk, 2 * m + lwork_zungbr_p);
        }
        if (!wntun) {
          maxwrk = std::max<blasint>(maxwrk, 2 * m + lwork_zungbr_q);
        }
        minwrk = 2 * m + n;
      }
    }
    maxwrk = std::max<blasint>(maxwrk, minwrk);
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -13;
    }
  }
  if (info != 0) {
    Xerbla<T>("GESVD", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (m == 0 || n == 0) {
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = std::sqrt(Lamch<R>('S')) / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', m, n, a, lda, dum);
  iscl = 0;
  if (anrm > zero && anrm < smlnum) {
    iscl = 1;
    Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
  } else if (anrm > bignum) {
    iscl = 1;
    Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
  }
  if (m >= n) {
    // A has at least as many rows as columns. If A has sufficiently
    // more rows than columns, first reduce using the QR
    // decomposition (if sufficient workspace available)
    if (m >= mnthr) {
      if (wntun) {
        // Path 1 (M much larger than N, JOBU='N')
        // No left singular vectors to be computed
        itau = 1;
        iwork = itau + n;
        // Compute A=Q*R
        // (CWorkspace: need 2*N, prefer N+N*NB)
        // (RWorkspace: need 0)
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork), lwork - iwork + 1);
        // Zero out below R
        if (n > 1) {
          Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
        }
        ie = 1;
        itauq = 1;
        itaup = itauq + n;
        iwork = itaup + n;
        // Bidiagonalize R in A
        // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
        // (RWorkspace: need N)
        Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(iwork), lwork - iwork + 1);
        ncvt = 0;
        if (wntvo || wntvas) {
          // If right singular vectors desired, generate P'.
          // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
          // (RWorkspace: 0)
          Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          ncvt = n;
        }
        irwork = ie + n;
        // Perform bidiagonal QR iteration, computing right
        // singular vectors of A in A if desired
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, 0, 0, s, RWORK.ptr(ie), a, lda, cdum, 1,
                     cdum, 1, RWORK.ptr(irwork));
        // If right singular vectors desired in VT, copy them there
        if (wntvas) Lacpy('F', n, n, a, lda, vt, ldvt);
      } else if (wntuo && wntvn) {
        // Path 2 (M much larger than N, JOBU='O', JOBVT='N')
        // N left singular vectors to be overwritten on A and
        // no right singular vectors to be computed
        if (lwork >= n * n + 3 * n) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n) + lda * n) {
            // WORK(IU) is LDA by N, WORK(IR) is LDA by N
            ldwrku = lda;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n) + n * n) {
            // WORK(IU) is LDA by N, WORK(IR) is N by N
            ldwrku = lda;
            ldwrkr = n;
          } else {
            // WORK(IU) is LDWRKU by N, WORK(IR) is N by N
            ldwrku = (lwork - n * n) / n;
            ldwrkr = n;
          }
          itau = ir + ldwrkr * n;
          iwork = itau + n;
          // Compute A=Q*R
          // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
          // (RWorkspace: 0)
          Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy R to WORK(IR) and zero out below it
          Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
          Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(ir + 1), ldwrkr);
          // Generate Q in A
          // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
          // (RWorkspace: 0)
          Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = 1;
          itauq = itau;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize R in WORK(IR)
          // (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
          // (RWorkspace: need N)
          Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing R
          // (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
          // (RWorkspace: need 0)
          Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                WORK.ptr(iwork), lwork - iwork + 1);
          irwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IR)
          // (CWorkspace: need N*N)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', n, 0, n, 0, s, RWORK.ptr(ie), cdum, 1, WORK.ptr(ir),
                       ldwrkr, cdum, 1, RWORK.ptr(irwork));
          iu = itauq;
          // Multiply Q in A by left singular vectors of R in
          // WORK(IR), storing result in WORK(IU) and copying to A
          // (CWorkspace: need N*N+N, prefer N*N+M*N)
          // (RWorkspace: 0)
          for (i = 1; i <= m; i += ldwrku) {
            chunk = std::min<blasint>(m - i + 1, ldwrku);
            Gemm('N', 'N', chunk, n, n, cone, A.ptr(i, 1), lda, WORK.ptr(ir),
                 ldwrkr, czero, WORK.ptr(iu), ldwrku);
            Lacpy('F', chunk, n, WORK.ptr(iu), ldwrku, A.ptr(i, 1), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          ie = 1;
          itauq = 1;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize A
          // (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
          // (RWorkspace: N)
          Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing A
          // (CWorkspace: need 3*N, prefer 2*N+N*NB)
          // (RWorkspace: 0)
          Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
          irwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in A
          // (CWorkspace: need 0)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', n, 0, m, 0, s, RWORK.ptr(ie), cdum, 1, a, lda, cdum,
                       1, RWORK.ptr(irwork));
        }
      } else if (wntuo && wntvas) {
        // Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
        // N left singular vectors to be overwritten on A and
        // N right singular vectors to be computed in VT
        if (lwork >= n * n + 3 * n) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n) + lda * n) {
            // WORK(IU) is LDA by N and WORK(IR) is LDA by N
            ldwrku = lda;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n) + n * n) {
            // WORK(IU) is LDA by N and WORK(IR) is N by N
            ldwrku = lda;
            ldwrkr = n;
          } else {
            // WORK(IU) is LDWRKU by N and WORK(IR) is N by N
            ldwrku = (lwork - n * n) / n;
            ldwrkr = n;
          }
          itau = ir + ldwrkr * n;
          iwork = itau + n;
          // Compute A=Q*R
          // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
          // (RWorkspace: 0)
          Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy R to VT, zeroing out below it
          Lacpy('U', n, n, a, lda, vt, ldvt);
          if (n > 1) Laset('L', n - 1, n - 1, czero, czero, VT.ptr(2, 1), ldvt);
          // Generate Q in A
          // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
          // (RWorkspace: 0)
          Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = 1;
          itauq = itau;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize R in VT, copying result to WORK(IR)
          // (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
          // (RWorkspace: need N)
          Gebrd(n, n, vt, ldvt, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          Lacpy('L', n, n, vt, ldvt, WORK.ptr(ir), ldwrkr);
          // Generate left vectors bidiagonalizing R in WORK(IR)
          // (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
          // (RWorkspace: 0)
          Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing R in VT
          // (CWorkspace: need N*N+3*N-1, prefer N*N+2*N+(N-1)*NB)
          // (RWorkspace: 0)
          Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          irwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IR) and computing right
          // singular vectors of R in VT
          // (CWorkspace: need N*N)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', n, n, n, 0, s, RWORK.ptr(ie), vt, ldvt,
                       WORK.ptr(ir), ldwrkr, cdum, 1, RWORK.ptr(irwork));
          iu = itauq;
          // Multiply Q in A by left singular vectors of R in
          // WORK(IR), storing result in WORK(IU) and copying to A
          // (CWorkspace: need N*N+N, prefer N*N+M*N)
          // (RWorkspace: 0)
          for (i = 1; i <= m; i += ldwrku) {
            chunk = std::min<blasint>(m - i + 1, ldwrku);
            Gemm('N', 'N', chunk, n, n, cone, A.ptr(i, 1), lda, WORK.ptr(ir),
                 ldwrkr, czero, WORK.ptr(iu), ldwrku);
            Lacpy('F', chunk, n, WORK.ptr(iu), ldwrku, A.ptr(i, 1), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          itau = 1;
          iwork = itau + n;
          // Compute A=Q*R
          // (CWorkspace: need 2*N, prefer N+N*NB)
          // (RWorkspace: 0)
          Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy R to VT, zeroing out below it
          Lacpy('U', n, n, a, lda, vt, ldvt);
          if (n > 1) Laset('L', n - 1, n - 1, czero, czero, VT.ptr(2, 1), ldvt);
          // Generate Q in A
          // (CWorkspace: need 2*N, prefer N+N*NB)
          // (RWorkspace: 0)
          Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = 1;
          itauq = itau;
          itaup = itauq + n;
          iwork = itaup + n;
          // Bidiagonalize R in VT
          // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
          // (RWorkspace: N)
          Gebrd(n, n, vt, ldvt, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Multiply Q in A by left vectors bidiagonalizing R
          // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
          // (RWorkspace: 0)
          Unmbr('Q', 'R', 'N', m, n, n, vt, ldvt, WORK.ptr(itauq), a, lda,
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing R in VT
          // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
          // (RWorkspace: 0)
          Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          irwork = ie + n;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in A and computing right
          // singular vectors of A in VT
          // (CWorkspace: 0)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', n, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, a, lda,
                       cdum, 1, RWORK.ptr(irwork));
        }
      } else if (wntus) {
        if (wntvn) {
          // Path 4 (M much larger than N, JOBU='S', JOBVT='N')
          // N left singular vectors to be computed in U and
          // no right singular vectors to be computed
          if (lwork >= n * n + 3 * n) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IR) is LDA by N
              ldwrkr = lda;
            } else {
              // WORK(IR) is N by N
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R
            // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IR), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(ir + 1), ldwrkr);
            // Generate Q in A
            // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
            // (RWorkspace: 0)
            Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IR)
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left vectors bidiagonalizing R in WORK(IR)
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
            // (RWorkspace: 0)
            Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IR)
            // (CWorkspace: need N*N)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, 0, n, 0, s, RWORK.ptr(ie), cdum, 1,
                         WORK.ptr(ir), ldwrkr, cdum, 1, RWORK.ptr(irwork));
            // Multiply Q in A by left singular vectors of R in
            // WORK(IR), storing result in U
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, n, cone, a, lda, WORK.ptr(ir), ldwrkr, czero,
                 u, ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Ungqr(m, n, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left vectors bidiagonalizing R
            // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
            // (RWorkspace: 0)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, 0, m, 0, s, RWORK.ptr(ie), cdum, 1, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntvo) {
          // Path 5 (M much larger than N, JOBU='S', JOBVT='O')
          // N left singular vectors to be computed in U and
          // N right singular vectors to be overwritten on A
          if (lwork >= 2 * n * n + 3 * n) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * n) {
              // WORK(IU) is LDA by N and WORK(IR) is LDA by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + n) * n) {
              // WORK(IU) is LDA by N and WORK(IR) is N by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            } else {
              // WORK(IU) is N by N and WORK(IR) is N by N
              ldwrku = n;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R
            // (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(iu + 1), ldwrku);
            // Generate Q in A
            // (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
            // (RWorkspace: 0)
            Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to
            // WORK(IR)
            // (CWorkspace: need   2*N*N+3*N,
            // prefer 2*N*N+2*N+2*N*NB)
            // (RWorkspace: need   N)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
            // (RWorkspace: 0)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in WORK(IR)
            // (CWorkspace: need   2*N*N+3*N-1,
            // prefer 2*N*N+2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in WORK(IR)
            // (CWorkspace: need 2*N*N)
            // (RWorkspace: need BDSPAC)
            info =
                Bdsqr('U', n, n, n, 0, s, RWORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                      WORK.ptr(iu), ldwrku, cdum, 1, RWORK.ptr(irwork));
            // Multiply Q in A by left singular vectors of R in
            // WORK(IU), storing result in U
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, n, cone, a, lda, WORK.ptr(iu), ldwrku, czero,
                 u, ldu);
            // Copy right singular vectors of R to A
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Lacpy('F', n, n, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Ungqr(m, n, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left vectors bidiagonalizing R
            // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
            // (RWorkspace: 0)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right vectors bidiagonalizing R in A
            // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in A
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, RWORK.ptr(ie), a, lda, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntvas) {
          // Path 6 (M much larger than N, JOBU='S', JOBVT='S'
          // or 'A')
          // N left singular vectors to be computed in U and
          // N right singular vectors to be computed in VT
          if (lwork >= n * n + 3 * n) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IU) is LDA by N
              ldwrku = lda;
            } else {
              // WORK(IU) is N by N
              ldwrku = n;
            }
            itau = iu + ldwrku * n;
            iwork = itau + n;
            // Compute A=Q*R
            // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(iu + 1), ldwrku);
            // Generate Q in A
            // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
            // (RWorkspace: 0)
            Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to VT
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, vt, ldvt);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
            // (RWorkspace: 0)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (CWorkspace: need   N*N+3*N-1,
            // prefer N*N+2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in VT
            // (CWorkspace: need N*N)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, n, n, 0, s, RWORK.ptr(ie), vt, ldvt,
                         WORK.ptr(iu), ldwrku, cdum, 1, RWORK.ptr(irwork));
            // Multiply Q in A by left singular vectors of R in
            // WORK(IU), storing result in U
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, n, cone, a, lda, WORK.ptr(iu), ldwrku, czero,
                 u, ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Ungqr(m, n, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to VT, zeroing out below it
            Lacpy('U', n, n, a, lda, vt, ldvt);
            if (n > 1)
              Laset('L', n - 1, n - 1, czero, czero, VT.ptr(2, 1), ldvt);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in VT
            // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, vt, ldvt, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in VT
            // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
            // (RWorkspace: 0)
            Unmbr('Q', 'R', 'N', m, n, n, vt, ldvt, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        }
      } else if (wntua) {
        if (wntvn) {
          // Path 7 (M much larger than N, JOBU='A', JOBVT='N')
          // M left singular vectors to be computed in U and
          // no right singular vectors to be computed
          if (lwork >= n * n + std::max<blasint>(n + m, 3 * n)) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IR) is LDA by N
              ldwrkr = lda;
            } else {
              // WORK(IR) is N by N
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Copy R to WORK(IR), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(ir + 1), ldwrkr);
            // Generate Q in U
            // (CWorkspace: need N*N+N+M, prefer N*N+N+M*NB)
            // (RWorkspace: 0)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IR)
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in WORK(IR)
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
            // (RWorkspace: 0)
            Ungbr('Q', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IR)
            // (CWorkspace: need N*N)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, 0, n, 0, s, RWORK.ptr(ie), cdum, 1,
                         WORK.ptr(ir), ldwrkr, cdum, 1, RWORK.ptr(irwork));
            // Multiply Q in U by left singular vectors of R in
            // WORK(IR), storing result in A
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, n, cone, u, ldu, WORK.ptr(ir), ldwrkr, czero,
                 a, lda);
            // Copy left singular vectors of A from A to U
            Lacpy('F', m, n, a, lda, u, ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need N+M, prefer N+M*NB)
            // (RWorkspace: 0)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in A
            // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
            // (RWorkspace: 0)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, 0, m, 0, s, RWORK.ptr(ie), cdum, 1, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntvo) {
          // Path 8 (M much larger than N, JOBU='A', JOBVT='O')
          // M left singular vectors to be computed in U and
          // N right singular vectors to be overwritten on A
          if (lwork >= 2 * n * n + std::max<blasint>(n + m, 3 * n)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * n) {
              // WORK(IU) is LDA by N and WORK(IR) is LDA by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + n) * n) {
              // WORK(IU) is LDA by N and WORK(IR) is N by N
              ldwrku = lda;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            } else {
              // WORK(IU) is N by N and WORK(IR) is N by N
              ldwrku = n;
              ir = iu + ldwrku * n;
              ldwrkr = n;
            }
            itau = ir + ldwrkr * n;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N*N+2*N, prefer 2*N*N+N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need 2*N*N+N+M, prefer 2*N*N+N+M*NB)
            // (RWorkspace: 0)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(iu + 1), ldwrku);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to
            // WORK(IR)
            // (CWorkspace: need   2*N*N+3*N,
            // prefer 2*N*N+2*N+2*N*NB)
            // (RWorkspace: need   N)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
            // (RWorkspace: 0)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in WORK(IR)
            // (CWorkspace: need   2*N*N+3*N-1,
            // prefer 2*N*N+2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in WORK(IR)
            // (CWorkspace: need 2*N*N)
            // (RWorkspace: need BDSPAC)
            info =
                Bdsqr('U', n, n, n, 0, s, RWORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                      WORK.ptr(iu), ldwrku, cdum, 1, RWORK.ptr(irwork));
            // Multiply Q in U by left singular vectors of R in
            // WORK(IU), storing result in A
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, n, cone, u, ldu, WORK.ptr(iu), ldwrku, czero,
                 a, lda);
            // Copy left singular vectors of A from A to U
            Lacpy('F', m, n, a, lda, u, ldu);
            // Copy right singular vectors of R from WORK(IR) to A
            Lacpy('F', n, n, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need N+M, prefer N+M*NB)
            // (RWorkspace: 0)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Zero out below R in A
            if (n > 1) {
              Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
            }
            // Bidiagonalize R in A
            // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in A
            // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
            // (RWorkspace: 0)
            Unmbr('Q', 'R', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in A
            // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in A
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, RWORK.ptr(ie), a, lda, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntvas) {
          // Path 9 (M much larger than N, JOBU='A', JOBVT='S'
          // or 'A')
          // M left singular vectors to be computed in U and
          // N right singular vectors to be computed in VT
          if (lwork >= n * n + std::max<blasint>(n + m, 3 * n)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * n) {
              // WORK(IU) is LDA by N
              ldwrku = lda;
            } else {
              // WORK(IU) is N by N
              ldwrku = n;
            }
            itau = iu + ldwrku * n;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need N*N+N+M, prefer N*N+N+M*NB)
            // (RWorkspace: 0)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R to WORK(IU), zeroing out below it
            Lacpy('U', n, n, a, lda, WORK.ptr(iu), ldwrku);
            Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(iu + 1), ldwrku);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in WORK(IU), copying result to VT
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('U', n, n, WORK.ptr(iu), ldwrku, vt, ldvt);
            // Generate left bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
            // (RWorkspace: 0)
            Ungbr('Q', n, n, n, WORK.ptr(iu), ldwrku, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (CWorkspace: need   N*N+3*N-1,
            // prefer N*N+2*N+(N-1)*NB)
            // (RWorkspace: need   0)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IU) and computing
            // right singular vectors of R in VT
            // (CWorkspace: need N*N)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, n, n, 0, s, RWORK.ptr(ie), vt, ldvt,
                         WORK.ptr(iu), ldwrku, cdum, 1, RWORK.ptr(irwork));
            // Multiply Q in U by left singular vectors of R in
            // WORK(IU), storing result in A
            // (CWorkspace: need N*N)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, n, cone, u, ldu, WORK.ptr(iu), ldwrku, czero,
                 a, lda);
            // Copy left singular vectors of A from A to U
            Lacpy('F', m, n, a, lda, u, ldu);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + n;
            // Compute A=Q*R, copying result to U
            // (CWorkspace: need 2*N, prefer N+N*NB)
            // (RWorkspace: 0)
            Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('L', m, n, a, lda, u, ldu);
            // Generate Q in U
            // (CWorkspace: need N+M, prefer N+M*NB)
            // (RWorkspace: 0)
            Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy R from A to VT, zeroing out below it
            Lacpy('U', n, n, a, lda, vt, ldvt);
            if (n > 1)
              Laset('L', n - 1, n - 1, czero, czero, VT.ptr(2, 1), ldvt);
            ie = 1;
            itauq = itau;
            itaup = itauq + n;
            iwork = itaup + n;
            // Bidiagonalize R in VT
            // (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
            // (RWorkspace: need N)
            Gebrd(n, n, vt, ldvt, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply Q in U by left bidiagonalizing vectors
            // in VT
            // (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
            // (RWorkspace: 0)
            Unmbr('Q', 'R', 'N', m, n, n, vt, ldvt, WORK.ptr(itauq), u, ldu,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in VT
            // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + n;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', n, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        }
      }
    } else {
      // M .LT. MNTHR
      // Path 10 (M at least N, but not much larger)
      // Reduce to bidiagonal form without QR decomposition
      ie = 1;
      itauq = 1;
      itaup = itauq + n;
      iwork = itaup + n;
      // Bidiagonalize A
      // (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
      // (RWorkspace: need N)
      Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(iwork), lwork - iwork + 1);
      if (wntuas) {
        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // (CWorkspace: need 2*N+NCU, prefer 2*N+NCU*NB)
        // (RWorkspace: 0)
        Lacpy('L', m, n, a, lda, u, ldu);
        if (wntus) ncu = n;
        if (wntua) ncu = m;
        Ungbr('Q', m, ncu, n, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvas) {
        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
        // (RWorkspace: 0)
        Lacpy('U', n, n, a, lda, vt, ldvt);
        Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntuo) {
        // If left singular vectors desired in A, generate left
        // bidiagonalizing vectors in A
        // (CWorkspace: need 3*N, prefer 2*N+N*NB)
        // (RWorkspace: 0)
        Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvo) {
        // If right singular vectors desired in A, generate right
        // bidiagonalizing vectors in A
        // (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
        // (RWorkspace: 0)
        Ungbr('P', n, n, n, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      irwork = ie + n;
      if (wntuas || wntuo) nru = m;
      if (wntun) nru = 0;
      if (wntvas || wntvo) ncvt = n;
      if (wntvn) ncvt = 0;
      if ((!wntuo) && (!wntvo)) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in VT
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, nru, 0, s, RWORK.ptr(ie), vt, ldvt, u, ldu,
                     cdum, 1, RWORK.ptr(irwork));
      } else if ((!wntuo) && wntvo) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in A
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, nru, 0, s, RWORK.ptr(ie), a, lda, u, ldu,
                     cdum, 1, RWORK.ptr(irwork));
      } else {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in A and computing right singular
        // vectors in VT
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('U', n, ncvt, nru, 0, s, RWORK.ptr(ie), vt, ldvt, a, lda,
                     cdum, 1, RWORK.ptr(irwork));
      }
    }
  } else {
    // A has more columns than rows. If A has sufficiently more
    // columns than rows, first reduce using the LQ decomposition (if
    // sufficient workspace available)
    if (n >= mnthr) {
      if (wntvn) {
        // Path 1t(N much larger than M, JOBVT='N')
        // No right singular vectors to be computed
        itau = 1;
        iwork = itau + m;
        // Compute A=L*Q
        // (CWorkspace: need 2*M, prefer M+M*NB)
        // (RWorkspace: 0)
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork), lwork - iwork + 1);
        // Zero out above L
        Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
        ie = 1;
        itauq = 1;
        itaup = itauq + m;
        iwork = itaup + m;
        // Bidiagonalize L in A
        // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
        // (RWorkspace: need M)
        Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(iwork), lwork - iwork + 1);
        if (wntuo || wntuas) {
          // If left singular vectors desired, generate Q
          // (CWorkspace: need 3*M, prefer 2*M+M*NB)
          // (RWorkspace: 0)
          Ungbr('Q', m, m, m, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
        }
        irwork = ie + m;
        nru = 0;
        if (wntuo || wntuas) nru = m;
        // Perform bidiagonal QR iteration, computing left singular
        // vectors of A in A if desired
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('U', m, 0, nru, 0, s, RWORK.ptr(ie), cdum, 1, a, lda, cdum,
                     1, RWORK.ptr(irwork));
        // If left singular vectors desired in U, copy them there
        if (wntuas) Lacpy('F', m, m, a, lda, u, ldu);
      } else if (wntvo && wntun) {
        // Path 2t(N much larger than M, JOBU='N', JOBVT='O')
        // M right singular vectors to be overwritten on A and
        // no left singular vectors to be computed
        if (lwork >= m * m + 3 * m) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n) + lda * m) {
            // WORK(IU) is LDA by N and WORK(IR) is LDA by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n) + m * m) {
            // WORK(IU) is LDA by N and WORK(IR) is M by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = m;
          } else {
            // WORK(IU) is M by CHUNK and WORK(IR) is M by M
            ldwrku = m;
            chunk = (lwork - m * m) / m;
            ldwrkr = m;
          }
          itau = ir + ldwrkr * m;
          iwork = itau + m;
          // Compute A=L*Q
          // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
          // (RWorkspace: 0)
          Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy L to WORK(IR) and zero out above it
          Lacpy('L', m, m, a, lda, WORK.ptr(ir), ldwrkr);
          Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(ir + ldwrkr), ldwrkr);
          // Generate Q in A
          // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
          // (RWorkspace: 0)
          Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = 1;
          itauq = itau;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize L in WORK(IR)
          // (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
          // (RWorkspace: need M)
          Gebrd(m, m, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing L
          // (CWorkspace: need M*M+3*M-1, prefer M*M+2*M+(M-1)*NB)
          // (RWorkspace: 0)
          Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          irwork = ie + m;
          // Perform bidiagonal QR iteration, computing right
          // singular vectors of L in WORK(IR)
          // (CWorkspace: need M*M)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', m, m, 0, 0, s, RWORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                       cdum, 1, cdum, 1, RWORK.ptr(irwork));
          iu = itauq;
          // Multiply right singular vectors of L in WORK(IR) by Q
          // in A, storing result in WORK(IU) and copying to A
          // (CWorkspace: need M*M+M, prefer M*M+M*N)
          // (RWorkspace: 0)
          for (i = 1; i <= n; i += chunk) {
            blk = std::min<blasint>(n - i + 1, chunk);
            Gemm('N', 'N', m, blk, m, cone, WORK.ptr(ir), ldwrkr, A.ptr(1, i),
                 lda, czero, WORK.ptr(iu), ldwrku);
            Lacpy('F', m, blk, WORK.ptr(iu), ldwrku, A.ptr(1, i), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          ie = 1;
          itauq = 1;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize A
          // (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
          // (RWorkspace: need M)
          Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Generate right vectors bidiagonalizing A
          // (CWorkspace: need 3*M, prefer 2*M+M*NB)
          // (RWorkspace: 0)
          Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
                lwork - iwork + 1);
          irwork = ie + m;
          // Perform bidiagonal QR iteration, computing right
          // singular vectors of A in A
          // (CWorkspace: 0)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('L', m, n, 0, 0, s, RWORK.ptr(ie), a, lda, cdum, 1, cdum,
                       1, RWORK.ptr(irwork));
        }
      } else if (wntvo && wntuas) {
        // Path 3t(N much larger than M, JOBU='S' or 'A', JOBVT='O')
        // M right singular vectors to be overwritten on A and
        // M left singular vectors to be computed in U
        if (lwork >= m * m + 3 * m) {
          // Sufficient workspace for a fast algorithm
          ir = 1;
          if (lwork >= std::max<blasint>(wrkbl, lda * n) + lda * m) {
            // WORK(IU) is LDA by N and WORK(IR) is LDA by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = lda;
          } else if (lwork >= std::max<blasint>(wrkbl, lda * n) + m * m) {
            // WORK(IU) is LDA by N and WORK(IR) is M by M
            ldwrku = lda;
            chunk = n;
            ldwrkr = m;
          } else {
            // WORK(IU) is M by CHUNK and WORK(IR) is M by M
            ldwrku = m;
            chunk = (lwork - m * m) / m;
            ldwrkr = m;
          }
          itau = ir + ldwrkr * m;
          iwork = itau + m;
          // Compute A=L*Q
          // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
          // (RWorkspace: 0)
          Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy L to U, zeroing about above it
          Lacpy('L', m, m, a, lda, u, ldu);
          Laset('U', m - 1, m - 1, czero, czero, U.ptr(1, 2), ldu);
          // Generate Q in A
          // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
          // (RWorkspace: 0)
          Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = 1;
          itauq = itau;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize L in U, copying result to WORK(IR)
          // (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
          // (RWorkspace: need M)
          Gebrd(m, m, u, ldu, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          Lacpy('U', m, m, u, ldu, WORK.ptr(ir), ldwrkr);
          // Generate right vectors bidiagonalizing L in WORK(IR)
          // (CWorkspace: need M*M+3*M-1, prefer M*M+2*M+(M-1)*NB)
          // (RWorkspace: 0)
          Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing L in U
          // (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
          // (RWorkspace: 0)
          Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
          irwork = ie + m;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in U, and computing right
          // singular vectors of L in WORK(IR)
          // (CWorkspace: need M*M)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', m, m, m, 0, s, RWORK.ptr(ie), WORK.ptr(ir), ldwrkr,
                       u, ldu, cdum, 1, RWORK.ptr(irwork));
          iu = itauq;
          // Multiply right singular vectors of L in WORK(IR) by Q
          // in A, storing result in WORK(IU) and copying to A
          // (CWorkspace: need M*M+M, prefer M*M+M*N))
          // (RWorkspace: 0)
          for (i = 1; i <= n; i += chunk) {
            blk = std::min<blasint>(n - i + 1, chunk);
            Gemm('N', 'N', m, blk, m, cone, WORK.ptr(ir), ldwrkr, A.ptr(1, i),
                 lda, czero, WORK.ptr(iu), ldwrku);
            Lacpy('F', m, blk, WORK.ptr(iu), ldwrku, A.ptr(1, i), lda);
          }
        } else {
          // Insufficient workspace for a fast algorithm
          itau = 1;
          iwork = itau + m;
          // Compute A=L*Q
          // (CWorkspace: need 2*M, prefer M+M*NB)
          // (RWorkspace: 0)
          Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          // Copy L to U, zeroing out above it
          Lacpy('L', m, m, a, lda, u, ldu);
          Laset('U', m - 1, m - 1, czero, czero, U.ptr(1, 2), ldu);
          // Generate Q in A
          // (CWorkspace: need 2*M, prefer M+M*NB)
          // (RWorkspace: 0)
          Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                lwork - iwork + 1);
          ie = 1;
          itauq = itau;
          itaup = itauq + m;
          iwork = itaup + m;
          // Bidiagonalize L in U
          // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
          // (RWorkspace: need M)
          Gebrd(m, m, u, ldu, s, RWORK.ptr(ie), WORK.ptr(itauq),
                WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
          // Multiply right vectors bidiagonalizing L by Q in A
          // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
          // (RWorkspace: 0)
          Unmbr('P', 'L', 'C', m, n, m, u, ldu, WORK.ptr(itaup), a, lda,
                WORK.ptr(iwork), lwork - iwork + 1);
          // Generate left vectors bidiagonalizing L in U
          // (CWorkspace: need 3*M, prefer 2*M+M*NB)
          // (RWorkspace: 0)
          Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                lwork - iwork + 1);
          irwork = ie + m;
          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in A
          // (CWorkspace: 0)
          // (RWorkspace: need BDSPAC)
          info = Bdsqr('U', m, n, m, 0, s, RWORK.ptr(ie), a, lda, u, ldu, cdum,
                       1, RWORK.ptr(irwork));
        }
      } else if (wntvs) {
        if (wntun) {
          // Path 4t(N much larger than M, JOBU='N', JOBVT='S')
          // M right singular vectors to be computed in VT and
          // no left singular vectors to be computed
          if (lwork >= m * m + 3 * m) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IR) is LDA by M
              ldwrkr = lda;
            } else {
              // WORK(IR) is M by M
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q
            // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IR), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(ir + ldwrkr),
                  ldwrkr);
            // Generate Q in A
            // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
            // (RWorkspace: 0)
            Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IR)
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right vectors bidiagonalizing L in
            // WORK(IR)
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+(M-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of L in WORK(IR)
            // (CWorkspace: need M*M)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, m, 0, 0, s, RWORK.ptr(ie), WORK.ptr(ir),
                         ldwrkr, cdum, 1, cdum, 1, RWORK.ptr(irwork));
            // Multiply right singular vectors of L in WORK(IR) by
            // Q in A, storing result in VT
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, m, cone, WORK.ptr(ir), ldwrkr, a, lda, czero,
                 vt, ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy result to VT
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Unglq(m, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right vectors bidiagonalizing L by Q in VT
            // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
            // (RWorkspace: 0)
            Unmbr('P', 'L', 'C', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, n, 0, 0, s, RWORK.ptr(ie), vt, ldvt, cdum, 1,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntuo) {
          // Path 5t(N much larger than M, JOBU='O', JOBVT='S')
          // M right singular vectors to be computed in VT and
          // M left singular vectors to be overwritten on A
          if (lwork >= 2 * m * m + 3 * m) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * m) {
              // WORK(IU) is LDA by M and WORK(IR) is LDA by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + m) * m) {
              // WORK(IU) is LDA by M and WORK(IR) is M by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            } else {
              // WORK(IU) is M by M and WORK(IR) is M by M
              ldwrku = m;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q
            // (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out below it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(iu + ldwrku),
                  ldwrku);
            // Generate Q in A
            // (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
            // (RWorkspace: 0)
            Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to
            // WORK(IR)
            // (CWorkspace: need   2*M*M+3*M,
            // prefer 2*M*M+2*M+2*M*NB)
            // (RWorkspace: need   M)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need   2*M*M+3*M-1,
            // prefer 2*M*M+2*M+(M-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in WORK(IR)
            // (CWorkspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in WORK(IR) and computing
            // right singular vectors of L in WORK(IU)
            // (CWorkspace: need 2*M*M)
            // (RWorkspace: need BDSPAC)
            info =
                Bdsqr('U', m, m, m, 0, s, RWORK.ptr(ie), WORK.ptr(iu), ldwrku,
                      WORK.ptr(ir), ldwrkr, cdum, 1, RWORK.ptr(irwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in A, storing result in VT
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, m, cone, WORK.ptr(iu), ldwrku, a, lda, czero,
                 vt, ldvt);
            // Copy left singular vectors of L to A
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Lacpy('F', m, m, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Unglq(m, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right vectors bidiagonalizing L by Q in VT
            // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
            // (RWorkspace: 0)
            Unmbr('P', 'L', 'C', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors of L in A
            // (CWorkspace: need 3*M, prefer 2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in A and computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, a, lda,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntuas) {
          // Path 6t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='S')
          // M right singular vectors to be computed in VT and
          // M left singular vectors to be computed in U
          if (lwork >= m * m + 3 * m) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IU) is LDA by N
              ldwrku = lda;
            } else {
              // WORK(IU) is LDA by M
              ldwrku = m;
            }
            itau = iu + ldwrku * m;
            iwork = itau + m;
            // Compute A=L*Q
            // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(iu + ldwrku),
                  ldwrku);
            // Generate Q in A
            // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
            // (RWorkspace: 0)
            Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to U
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, u, ldu);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need   M*M+3*M-1,
            // prefer M*M+2*M+(M-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in U and computing right
            // singular vectors of L in WORK(IU)
            // (CWorkspace: need M*M)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, m, m, 0, s, RWORK.ptr(ie), WORK.ptr(iu),
                         ldwrku, u, ldu, cdum, 1, RWORK.ptr(irwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in A, storing result in VT
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, m, cone, WORK.ptr(iu), ldwrku, a, lda, czero,
                 vt, ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Unglq(m, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to U, zeroing out above it
            Lacpy('L', m, m, a, lda, u, ldu);
            Laset('U', m - 1, m - 1, czero, czero, U.ptr(1, 2), ldu);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in U
            // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, u, ldu, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in U by Q
            // in VT
            // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
            // (RWorkspace: 0)
            Unmbr('P', 'L', 'C', m, n, m, u, ldu, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (CWorkspace: need 3*M, prefer 2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        }
      } else if (wntva) {
        if (wntun) {
          // Path 7t(N much larger than M, JOBU='N', JOBVT='A')
          // N right singular vectors to be computed in VT and
          // no left singular vectors to be computed
          if (lwork >= m * m + std::max<blasint>(n + m, 3 * m)) {
            // Sufficient workspace for a fast algorithm
            ir = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IR) is LDA by M
              ldwrkr = lda;
            } else {
              // WORK(IR) is M by M
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Copy L to WORK(IR), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(ir), ldwrkr);
            Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(ir + ldwrkr),
                  ldwrkr);
            // Generate Q in VT
            // (CWorkspace: need M*M+M+N, prefer M*M+M+N*NB)
            // (RWorkspace: 0)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IR)
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Generate right bidiagonalizing vectors in WORK(IR)
            // (CWorkspace: need   M*M+3*M-1,
            // prefer M*M+2*M+(M-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of L in WORK(IR)
            // (CWorkspace: need M*M)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, m, 0, 0, s, RWORK.ptr(ie), WORK.ptr(ir),
                         ldwrkr, cdum, 1, cdum, 1, RWORK.ptr(irwork));
            // Multiply right singular vectors of L in WORK(IR) by
            // Q in VT, storing result in A
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, m, cone, WORK.ptr(ir), ldwrkr, vt, ldvt, czero,
                 a, lda);
            // Copy right singular vectors of A from A to VT
            Lacpy('F', m, n, a, lda, vt, ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need M+N, prefer M+N*NB)
            // (RWorkspace: 0)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in A by Q
            // in VT
            // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
            // (RWorkspace: 0)
            Unmbr('P', 'L', 'C', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, n, 0, 0, s, RWORK.ptr(ie), vt, ldvt, cdum, 1,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntuo) {
          // Path 8t(N much larger than M, JOBU='O', JOBVT='A')
          // N right singular vectors to be computed in VT and
          // M left singular vectors to be overwritten on A
          if (lwork >= 2 * m * m + std::max<blasint>(n + m, 3 * m)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + 2 * lda * m) {
              // WORK(IU) is LDA by M and WORK(IR) is LDA by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = lda;
            } else if (lwork >= wrkbl + (lda + m) * m) {
              // WORK(IU) is LDA by M and WORK(IR) is M by M
              ldwrku = lda;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            } else {
              // WORK(IU) is M by M and WORK(IR) is M by M
              ldwrku = m;
              ir = iu + ldwrku * m;
              ldwrkr = m;
            }
            itau = ir + ldwrkr * m;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need 2*M*M+2*M, prefer 2*M*M+M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need 2*M*M+M+N, prefer 2*M*M+M+N*NB)
            // (RWorkspace: 0)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(iu + ldwrku),
                  ldwrku);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to
            // WORK(IR)
            // (CWorkspace: need   2*M*M+3*M,
            // prefer 2*M*M+2*M+2*M*NB)
            // (RWorkspace: need   M)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, WORK.ptr(ir), ldwrkr);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need   2*M*M+3*M-1,
            // prefer 2*M*M+2*M+(M-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in WORK(IR)
            // (CWorkspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
                  WORK.ptr(iwork), lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in WORK(IR) and computing
            // right singular vectors of L in WORK(IU)
            // (CWorkspace: need 2*M*M)
            // (RWorkspace: need BDSPAC)
            info =
                Bdsqr('U', m, m, m, 0, s, RWORK.ptr(ie), WORK.ptr(iu), ldwrku,
                      WORK.ptr(ir), ldwrkr, cdum, 1, RWORK.ptr(irwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in VT, storing result in A
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, m, cone, WORK.ptr(iu), ldwrku, vt, ldvt, czero,
                 a, lda);
            // Copy right singular vectors of A from A to VT
            Lacpy('F', m, n, a, lda, vt, ldvt);
            // Copy left singular vectors of A from WORK(IR) to A
            Lacpy('F', m, m, WORK.ptr(ir), ldwrkr, a, lda);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need M+N, prefer M+N*NB)
            // (RWorkspace: 0)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Zero out above L in A
            Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
            // Bidiagonalize L in A
            // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in A by Q
            // in VT
            // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
            // (RWorkspace: 0)
            Unmbr('P', 'L', 'C', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in A
            // (CWorkspace: need 3*M, prefer 2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in A and computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, a, lda,
                         cdum, 1, RWORK.ptr(irwork));
          }
        } else if (wntuas) {
          // Path 9t(N much larger than M, JOBU='S' or 'A',
          // JOBVT='A')
          // N right singular vectors to be computed in VT and
          // M left singular vectors to be computed in U
          if (lwork >= m * m + std::max<blasint>(n + m, 3 * m)) {
            // Sufficient workspace for a fast algorithm
            iu = 1;
            if (lwork >= wrkbl + lda * m) {
              // WORK(IU) is LDA by M
              ldwrku = lda;
            } else {
              // WORK(IU) is M by M
              ldwrku = m;
            }
            itau = iu + ldwrku * m;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need M*M+M+N, prefer M*M+M+N*NB)
            // (RWorkspace: 0)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to WORK(IU), zeroing out above it
            Lacpy('L', m, m, a, lda, WORK.ptr(iu), ldwrku);
            Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(iu + ldwrku),
                  ldwrku);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in WORK(IU), copying result to U
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, WORK.ptr(iu), ldwrku, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            Lacpy('L', m, m, WORK.ptr(iu), ldwrku, u, ldu);
            // Generate right bidiagonalizing vectors in WORK(IU)
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+(M-1)*NB)
            // (RWorkspace: 0)
            Ungbr('P', m, m, m, WORK.ptr(iu), ldwrku, WORK.ptr(itaup),
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of L in U and computing right
            // singular vectors of L in WORK(IU)
            // (CWorkspace: need M*M)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, m, m, 0, s, RWORK.ptr(ie), WORK.ptr(iu),
                         ldwrku, u, ldu, cdum, 1, RWORK.ptr(irwork));
            // Multiply right singular vectors of L in WORK(IU) by
            // Q in VT, storing result in A
            // (CWorkspace: need M*M)
            // (RWorkspace: 0)
            Gemm('N', 'N', m, n, m, cone, WORK.ptr(iu), ldwrku, vt, ldvt, czero,
                 a, lda);
            // Copy right singular vectors of A from A to VT
            Lacpy('F', m, n, a, lda, vt, ldvt);
          } else {
            // Insufficient workspace for a fast algorithm
            itau = 1;
            iwork = itau + m;
            // Compute A=L*Q, copying result to VT
            // (CWorkspace: need 2*M, prefer M+M*NB)
            // (RWorkspace: 0)
            Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            Lacpy('U', m, n, a, lda, vt, ldvt);
            // Generate Q in VT
            // (CWorkspace: need M+N, prefer M+N*NB)
            // (RWorkspace: 0)
            Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(iwork),
                  lwork - iwork + 1);
            // Copy L to U, zeroing out above it
            Lacpy('L', m, m, a, lda, u, ldu);
            Laset('U', m - 1, m - 1, czero, czero, U.ptr(1, 2), ldu);
            ie = 1;
            itauq = itau;
            itaup = itauq + m;
            iwork = itaup + m;
            // Bidiagonalize L in U
            // (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
            // (RWorkspace: need M)
            Gebrd(m, m, u, ldu, s, RWORK.ptr(ie), WORK.ptr(itauq),
                  WORK.ptr(itaup), WORK.ptr(iwork), lwork - iwork + 1);
            // Multiply right bidiagonalizing vectors in U by Q
            // in VT
            // (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
            // (RWorkspace: 0)
            Unmbr('P', 'L', 'C', m, n, m, u, ldu, WORK.ptr(itaup), vt, ldvt,
                  WORK.ptr(iwork), lwork - iwork + 1);
            // Generate left bidiagonalizing vectors in U
            // (CWorkspace: need 3*M, prefer 2*M+M*NB)
            // (RWorkspace: 0)
            Ungbr('Q', m, m, m, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
                  lwork - iwork + 1);
            irwork = ie + m;
            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in U and computing right
            // singular vectors of A in VT
            // (CWorkspace: 0)
            // (RWorkspace: need BDSPAC)
            info = Bdsqr('U', m, n, m, 0, s, RWORK.ptr(ie), vt, ldvt, u, ldu,
                         cdum, 1, RWORK.ptr(irwork));
          }
        }
      }
    } else {
      // N .LT. MNTHR
      // Path 10t(N greater than M, but not much larger)
      // Reduce to bidiagonal form without LQ decomposition
      ie = 1;
      itauq = 1;
      itaup = itauq + m;
      iwork = itaup + m;
      // Bidiagonalize A
      // (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
      // (RWorkspace: M)
      Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(iwork), lwork - iwork + 1);
      if (wntuas) {
        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // (CWorkspace: need 3*M-1, prefer 2*M+(M-1)*NB)
        // (RWorkspace: 0)
        Lacpy('L', m, m, a, lda, u, ldu);
        Ungbr('Q', m, m, n, u, ldu, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvas) {
        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // (CWorkspace: need 2*M+NRVT, prefer 2*M+NRVT*NB)
        // (RWorkspace: 0)
        Lacpy('U', m, n, a, lda, vt, ldvt);
        if (wntva) nrvt = n;
        if (wntvs) nrvt = m;
        Ungbr('P', nrvt, n, m, vt, ldvt, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntuo) {
        // If left singular vectors desired in A, generate left
        // bidiagonalizing vectors in A
        // (CWorkspace: need 3*M-1, prefer 2*M+(M-1)*NB)
        // (RWorkspace: 0)
        Ungbr('Q', m, m, n, a, lda, WORK.ptr(itauq), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      if (wntvo) {
        // If right singular vectors desired in A, generate right
        // bidiagonalizing vectors in A
        // (CWorkspace: need 3*M, prefer 2*M+M*NB)
        // (RWorkspace: 0)
        Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(iwork),
              lwork - iwork + 1);
      }
      irwork = ie + m;
      if (wntuas || wntuo) nru = m;
      if (wntun) nru = 0;
      if (wntvas || wntvo) ncvt = n;
      if (wntvn) ncvt = 0;
      if ((!wntuo) && (!wntvo)) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in VT
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('L', m, ncvt, nru, 0, s, RWORK.ptr(ie), vt, ldvt, u, ldu,
                     cdum, 1, RWORK.ptr(irwork));
      } else if ((!wntuo) && wntvo) {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in A
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('L', m, ncvt, nru, 0, s, RWORK.ptr(ie), a, lda, u, ldu,
                     cdum, 1, RWORK.ptr(irwork));
      } else {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in A and computing right singular
        // vectors in VT
        // (CWorkspace: 0)
        // (RWorkspace: need BDSPAC)
        info = Bdsqr('L', m, ncvt, nru, 0, s, RWORK.ptr(ie), vt, ldvt, a, lda,
                     cdum, 1, RWORK.ptr(irwork));
      }
    }
  }
  // Undo scaling if necessary
  if (iscl == 1) {
    if (anrm > bignum) Lascl<R>('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
    if (info != 0 && anrm > bignum)
      Lascl<R>('G', 0, 0, bignum, anrm, minmn - 1, 1, RWORK.ptr(ie), minmn);
    if (anrm < smlnum) Lascl<R>('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
    if (info != 0 && anrm < smlnum)
      Lascl<R>('G', 0, 0, smlnum, anrm, minmn - 1, 1, RWORK.ptr(ie), minmn);
  }
  // Return optimal workspace in WORK(1)
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

}  // namespace

// xGESVD: computes the singular value decomposition A = U * SIGMA * V**H of
// the M-by-N matrix A, optionally with the left and/or right singular
// vectors. SIGMA is zero except for its min(M,N) diagonal elements, the
// singular values, which are real, non-negative and returned in S in
// decreasing order; the first min(M,N) columns of U and rows of V**H are
// the left and right singular vectors. The routine returns V**H, not V.
//
// JOBU: 'A' all M columns of U are returned in U; 'S' the first min(M,N)
// columns; 'O' they overwrite A; 'N' no columns of U are computed.
// JOBVT: likewise for the rows of V**H (in VT), not both 'O'. A is
// destroyed unless JOBU or JOBVT is 'O'.
//
// LWORK >= max(1, 3*min(M,N) + max(M,N), 5*min(M,N)) for real T and >=
// max(1, 2*min(M,N) + max(M,N)) for complex T; larger for good performance
// (LWORK = -1 is a workspace query). RWORK (5*min(M,N) elements) is used
// for complex T only.
//
// Returns INFO: 0, -i for an illegal argument i, or i > 0 if xBDSQR did not
// converge, with i superdiagonals of an intermediate bidiagonal form B that
// did not converge to zero; they are returned in WORK(2:min(M,N)) for real
// T and in RWORK(1:min(M,N)-1) for complex T.
template <typename T>
blasint Gesvd(char jobu, char jobvt, blasint m, blasint n, T* a, blasint lda,
              RealT<T>* s, T* u, blasint ldu, T* vt, blasint ldvt, T* work,
              blasint lwork, RealT<T>* rwork) {
  if constexpr (IsComplex<T>::value) {
    return GesvdComplex(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work,
                        lwork, rwork);
  } else {
    return GesvdReal(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work,
                     lwork);
  }
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Gesvd<T>(char, char, blasint, blasint, T*, blasint,        \
                            RealT<T>*, T*, blasint, T*, blasint, T*, blasint, \
                            RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

HWYBLAS_API void sgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* s, float* u,
                         const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info) {
  *info =
      hwyblas::lapack::Gesvd(*jobu, *jobvt, *m, *n, a, *lda, s, u, *ldu, vt,
                             *ldvt, work, *lwork, static_cast<float*>(nullptr));
}
HWYBLAS_API void dgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gesvd(*jobu, *jobvt, *m, *n, a, *lda, s, u, *ldu, vt,
                                 *ldvt, work, *lwork,
                                 static_cast<double*>(nullptr));
}
HWYBLAS_API void cgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* s, float* u,
                         const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Gesvd(*jobu, *jobvt, *m, *n, As<c32>(a), *lda, s,
                                 As<c32>(u), *ldu, As<c32>(vt), *ldvt,
                                 As<c32>(work), *lwork, rwork);
}
HWYBLAS_API void zgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Gesvd(*jobu, *jobvt, *m, *n, As<c64>(a), *lda, s,
                                 As<c64>(u), *ldu, As<c64>(vt), *ldvt,
                                 As<c64>(work), *lwork, rwork);
}

}  // extern "C"
