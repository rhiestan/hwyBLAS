// The singular value decomposition of a general matrix by divide and
// conquer: xGESDD.
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

// The real and the complex routine differ in their paths (the complex one
// has the thresholds MNTHR1 and MNTHR2 and multiplies xBDSDC's real
// singular vectors into the complex ones with xLACRM / xLARCM) and in their
// workspaces throughout, so they are ported as two functions.

// SGESDD / DGESDD.
template <typename T>
// NOLINTNEXTLINE(google-readability-function-size,readability-function-size)
blasint GesddReal(char jobz, blasint m, blasint n, T* a, blasint lda, T* s,
                  T* u, blasint ldu, T* vt, blasint ldvt, T* work,
                  blasint lwork, blasint* iwork) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  R dum[1];
  const Vector<R> DUM(dum);
  blasint idum[1];
  const Vector<blasint> IWORK(iwork);
  const Vector<R> S(s);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldvt);
  const Vector<R> WORK(work);
  blasint bdspac, blk, chunk, i, ie, il, ir, iscl, itau, itaup, itauq, iu, ivt,
      ldwkvt, ldwrkl, ldwrkr, ldwrku, lwork_dgebrd_mm, lwork_dgebrd_mn,
      lwork_dgebrd_nn, lwork_dgelqf_mn, lwork_dgeqrf_mn, lwork_dorglq_mn,
      lwork_dorglq_nn, lwork_dorgqr_mm, lwork_dorgqr_mn, lwork_dormbr_prt_mm,
      lwork_dormbr_prt_mn, lwork_dormbr_prt_nn, lwork_dormbr_qln_mm,
      lwork_dormbr_qln_mn, lwork_dormbr_qln_nn, maxwrk, minmn, minwrk, mnthr,
      nwork, wrkbl;
  bool lquery, wntqa, wntqas, wntqn, wntqo, wntqs;
  R anrm, bignum, eps, smlnum;
  blasint info = 0;
  minmn = Min<blasint>(m, n);
  wntqa = Lsame(jobz, 'A');
  wntqs = Lsame(jobz, 'S');
  wntqas = wntqa || wntqs;
  wntqo = Lsame(jobz, 'O');
  wntqn = Lsame(jobz, 'N');
  lquery = (lwork == -1);
  if (!(wntqa || wntqs || wntqo || wntqn)) {
    info = -1;
  } else if (m < 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldu < 1 || (wntqas && ldu < m) || (wntqo && m < n && ldu < m)) {
    info = -8;
  } else if (ldvt < 1 || (wntqa && ldvt < n) || (wntqs && ldvt < minmn) ||
             (wntqo && m >= n && ldvt < n)) {
    info = -10;
  }
  // Compute workspace
  // Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace allocated at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV.
  if (info == 0) {
    minwrk = 1;
    maxwrk = 1;
    bdspac = 0;
    mnthr = static_cast<blasint>(minmn * R(11) / R(6));
    if (m >= n && minmn > 0) {
      // Compute space needed for DBDSDC
      if (wntqn) {
        // dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
        // keep 7*N for backwards compatibility.
        bdspac = 7 * n;
      } else {
        bdspac = 3 * n * n + 4 * n;
      }
      // Compute space preferred for each routine
      Gebrd(m, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), DUM.ptr(1),
            DUM.ptr(1), -1);
      lwork_dgebrd_mn = static_cast<blasint>(DUM(1));
      Gebrd(n, n, DUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), DUM.ptr(1),
            DUM.ptr(1), -1);
      lwork_dgebrd_nn = static_cast<blasint>(DUM(1));
      Geqrf(m, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dgeqrf_mn = static_cast<blasint>(DUM(1));
      Ungbr('Q', n, n, n, DUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), -1);
      // (DGESDD keeps this size in LWORK_DORGBR_Q_NN, but never uses it.)
      Ungqr(m, m, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dorgqr_mm = static_cast<blasint>(DUM(1));
      Ungqr(m, n, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dorgqr_mn = static_cast<blasint>(DUM(1));
      Unmbr('P', 'R', 'T', n, n, n, DUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), n,
            DUM.ptr(1), -1);
      lwork_dormbr_prt_nn = static_cast<blasint>(DUM(1));
      Unmbr('Q', 'L', 'N', n, n, n, DUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), n,
            DUM.ptr(1), -1);
      lwork_dormbr_qln_nn = static_cast<blasint>(DUM(1));
      Unmbr('Q', 'L', 'N', m, n, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), m,
            DUM.ptr(1), -1);
      lwork_dormbr_qln_mn = static_cast<blasint>(DUM(1));
      Unmbr('Q', 'L', 'N', m, m, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), m,
            DUM.ptr(1), -1);
      lwork_dormbr_qln_mm = static_cast<blasint>(DUM(1));
      if (m >= mnthr) {
        if (wntqn) {
          // Path 1 (M >> N, JOBZ='N')
          wrkbl = n + lwork_dgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dgebrd_nn);
          maxwrk = Max<blasint>(wrkbl, bdspac + n);
          minwrk = bdspac + n;
        } else if (wntqo) {
          // Path 2 (M >> N, JOBZ='O')
          wrkbl = n + lwork_dgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, n + lwork_dorgqr_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dgebrd_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_qln_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_prt_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + bdspac);
          maxwrk = wrkbl + 2 * n * n;
          minwrk = bdspac + 2 * n * n + 3 * n;
        } else if (wntqs) {
          // Path 3 (M >> N, JOBZ='S')
          wrkbl = n + lwork_dgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, n + lwork_dorgqr_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dgebrd_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_qln_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_prt_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + bdspac);
          maxwrk = wrkbl + n * n;
          minwrk = bdspac + n * n + 3 * n;
        } else if (wntqa) {
          // Path 4 (M >> N, JOBZ='A')
          wrkbl = n + lwork_dgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, n + lwork_dorgqr_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dgebrd_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_qln_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_prt_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + bdspac);
          maxwrk = wrkbl + n * n;
          minwrk = n * n + Max<blasint>(3 * n + bdspac, n + m);
        }
      } else {
        // Path 5 (M >= N, but not much larger)
        wrkbl = 3 * n + lwork_dgebrd_mn;
        if (wntqn) {
          // Path 5n (M >= N, jobz='N')
          maxwrk = Max<blasint>(wrkbl, 3 * n + bdspac);
          minwrk = 3 * n + Max<blasint>(m, bdspac);
        } else if (wntqo) {
          // Path 5o (M >= N, jobz='O')
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_prt_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_qln_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + bdspac);
          maxwrk = wrkbl + m * n;
          minwrk = 3 * n + Max<blasint>(m, n * n + bdspac);
        } else if (wntqs) {
          // Path 5s (M >= N, jobz='S')
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_qln_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_prt_nn);
          maxwrk = Max<blasint>(wrkbl, 3 * n + bdspac);
          minwrk = 3 * n + Max<blasint>(m, bdspac);
        } else if (wntqa) {
          // Path 5a (M >= N, jobz='A')
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * n + lwork_dormbr_prt_nn);
          maxwrk = Max<blasint>(wrkbl, 3 * n + bdspac);
          minwrk = 3 * n + Max<blasint>(m, bdspac);
        }
      }
    } else if (minmn > 0) {
      // Compute space needed for DBDSDC
      if (wntqn) {
        // dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
        // keep 7*N for backwards compatibility.
        bdspac = 7 * m;
      } else {
        bdspac = 3 * m * m + 4 * m;
      }
      // Compute space preferred for each routine
      Gebrd(m, n, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), DUM.ptr(1),
            DUM.ptr(1), -1);
      lwork_dgebrd_mn = static_cast<blasint>(DUM(1));
      Gebrd(m, m, a, m, s, DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dgebrd_mm = static_cast<blasint>(DUM(1));
      Gelqf(m, n, a, m, DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dgelqf_mn = static_cast<blasint>(DUM(1));
      Unglq(n, n, m, DUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dorglq_nn = static_cast<blasint>(DUM(1));
      Unglq(m, n, m, a, m, DUM.ptr(1), DUM.ptr(1), -1);
      lwork_dorglq_mn = static_cast<blasint>(DUM(1));
      Ungbr('P', m, m, m, a, n, DUM.ptr(1), DUM.ptr(1), -1);
      // (DGESDD keeps this size in LWORK_DORGBR_P_MM, but never uses it.)
      Unmbr('P', 'R', 'T', m, m, m, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), m,
            DUM.ptr(1), -1);
      lwork_dormbr_prt_mm = static_cast<blasint>(DUM(1));
      Unmbr('P', 'R', 'T', m, n, m, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), m,
            DUM.ptr(1), -1);
      lwork_dormbr_prt_mn = static_cast<blasint>(DUM(1));
      Unmbr('P', 'R', 'T', n, n, m, DUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), n,
            DUM.ptr(1), -1);
      lwork_dormbr_prt_nn = static_cast<blasint>(DUM(1));
      Unmbr('Q', 'L', 'N', m, m, m, DUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), m,
            DUM.ptr(1), -1);
      lwork_dormbr_qln_mm = static_cast<blasint>(DUM(1));
      if (n >= mnthr) {
        if (wntqn) {
          // Path 1t (N >> M, JOBZ='N')
          wrkbl = m + lwork_dgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dgebrd_mm);
          maxwrk = Max<blasint>(wrkbl, bdspac + m);
          minwrk = bdspac + m;
        } else if (wntqo) {
          // Path 2t (N >> M, JOBZ='O')
          wrkbl = m + lwork_dgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, m + lwork_dorglq_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dgebrd_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_prt_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + bdspac);
          maxwrk = wrkbl + 2 * m * m;
          minwrk = bdspac + 2 * m * m + 3 * m;
        } else if (wntqs) {
          // Path 3t (N >> M, JOBZ='S')
          wrkbl = m + lwork_dgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, m + lwork_dorglq_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dgebrd_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_prt_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + bdspac);
          maxwrk = wrkbl + m * m;
          minwrk = bdspac + m * m + 3 * m;
        } else if (wntqa) {
          // Path 4t (N >> M, JOBZ='A')
          wrkbl = m + lwork_dgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, m + lwork_dorglq_nn);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dgebrd_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_prt_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + bdspac);
          maxwrk = wrkbl + m * m;
          minwrk = m * m + Max<blasint>(3 * m + bdspac, m + n);
        }
      } else {
        // Path 5t (N > M, but not much larger)
        wrkbl = 3 * m + lwork_dgebrd_mn;
        if (wntqn) {
          // Path 5tn (N > M, jobz='N')
          maxwrk = Max<blasint>(wrkbl, 3 * m + bdspac);
          minwrk = 3 * m + Max<blasint>(n, bdspac);
        } else if (wntqo) {
          // Path 5to (N > M, jobz='O')
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_prt_mn);
          wrkbl = Max<blasint>(wrkbl, 3 * m + bdspac);
          maxwrk = wrkbl + m * n;
          minwrk = 3 * m + Max<blasint>(n, m * m + bdspac);
        } else if (wntqs) {
          // Path 5ts (N > M, jobz='S')
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_prt_mn);
          maxwrk = Max<blasint>(wrkbl, 3 * m + bdspac);
          minwrk = 3 * m + Max<blasint>(n, bdspac);
        } else if (wntqa) {
          // Path 5ta (N > M, jobz='A')
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 3 * m + lwork_dormbr_prt_nn);
          maxwrk = Max<blasint>(wrkbl, 3 * m + bdspac);
          minwrk = 3 * m + Max<blasint>(n, bdspac);
        }
      }
    }
    maxwrk = Max<blasint>(maxwrk, minwrk);
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -12;
    }
  }
  if (info != 0) {
    Xerbla<T>("GESDD", -info);
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
  if (std::isnan(anrm)) {
    info = -4;
    return info;
  }
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
      if (wntqn) {
        // Path 1 (M >> N, JOBZ='N')
        // No singular vectors to be computed
        itau = 1;
        nwork = itau + n;
        // Compute A=Q*R
        // Workspace: need   N [tau] + N    [work]
        // Workspace: prefer N [tau] + N*NB [work]
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Zero out below R
        Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
        ie = 1;
        itauq = ie + n;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in A
        // Workspace: need   3*N [e, tauq, taup] + N      [work]
        // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [work]
        Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        nwork = ie + n;
        // Perform bidiagonal SVD, computing singular values only
        // Workspace: need   N [e] + BDSPAC
        info = Bdsdc('U', 'N', n, s, WORK.ptr(ie), dum, 1, dum, 1, dum, idum,
                     WORK.ptr(nwork), iwork);
      } else if (wntqo) {
        // Path 2 (M >> N, JOBZ = 'O')
        // N left singular vectors to be overwritten on A and
        // N right singular vectors to be computed in VT
        ir = 1;
        // WORK(IR) is LDWRKR by N
        if (lwork >= lda * n + n * n + 3 * n + bdspac) {
          ldwrkr = lda;
        } else {
          ldwrkr = (lwork - n * n - 3 * n - bdspac) / n;
        }
        itau = ir + ldwrkr * n;
        nwork = itau + n;
        // Compute A=Q*R
        // Workspace: need   N*N [R] + N [tau] + N    [work]
        // Workspace: prefer N*N [R] + N [tau] + N*NB [work]
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy R to WORK(IR), zeroing out below it
        Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
        Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(ir + 1), ldwrkr);
        // Generate Q in A
        // Workspace: need   N*N [R] + N [tau] + N    [work]
        // Workspace: prefer N*N [R] + N [tau] + N*NB [work]
        Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = itau;
        itauq = ie + n;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in WORK(IR)
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
        // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
        Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // WORK(IU) is N by N
        iu = nwork;
        nwork = iu + n * n;
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in WORK(IU) and computing right
        // singular vectors of bidiagonal matrix in VT
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + BDSPAC
        info = Bdsdc('U', 'I', n, s, WORK.ptr(ie), WORK.ptr(iu), n, vt, ldvt,
                     dum, idum, WORK.ptr(nwork), iwork);
        // Overwrite WORK(IU) by left singular vectors of R
        // and VT by right singular vectors of R
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N [work]
        // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N*NB
        // [work]
        Unmbr('Q', 'L', 'N', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
              WORK.ptr(iu), n, WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup), vt,
              ldvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply Q in A by left singular vectors of R in
        // WORK(IU), storing result in WORK(IR) and copying to A
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U]
        // Workspace: prefer M*N [R] + 3*N [e, tauq, taup] + N*N [U]
        for (i = 1; i <= m; i += ldwrkr) {
          chunk = Min<blasint>(m - i + 1, ldwrkr);
          Gemm('N', 'N', chunk, n, n, one, A.ptr(i, 1), lda, WORK.ptr(iu), n,
               zero, WORK.ptr(ir), ldwrkr);
          Lacpy('F', chunk, n, WORK.ptr(ir), ldwrkr, A.ptr(i, 1), lda);
        }
      } else if (wntqs) {
        // Path 3 (M >> N, JOBZ='S')
        // N left singular vectors to be computed in U and
        // N right singular vectors to be computed in VT
        ir = 1;
        // WORK(IR) is N by N
        ldwrkr = n;
        itau = ir + ldwrkr * n;
        nwork = itau + n;
        // Compute A=Q*R
        // Workspace: need   N*N [R] + N [tau] + N    [work]
        // Workspace: prefer N*N [R] + N [tau] + N*NB [work]
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy R to WORK(IR), zeroing out below it
        Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
        Laset('L', n - 1, n - 1, zero, zero, WORK.ptr(ir + 1), ldwrkr);
        // Generate Q in A
        // Workspace: need   N*N [R] + N [tau] + N    [work]
        // Workspace: prefer N*N [R] + N [tau] + N*NB [work]
        Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = itau;
        itauq = ie + n;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in WORK(IR)
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
        // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
        Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, WORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagoal matrix in U and computing right singular
        // vectors of bidiagonal matrix in VT
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + BDSPAC
        info = Bdsdc('U', 'I', n, s, WORK.ptr(ie), u, ldu, vt, ldvt, dum, idum,
                     WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of R and VT
        // by right singular vectors of R
        // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [work]
        // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [work]
        Unmbr('Q', 'L', 'N', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq), u,
              ldu, WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup), vt,
              ldvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply Q in A by left singular vectors of R in
        // WORK(IR), storing result in U
        // Workspace: need   N*N [R]
        Lacpy('F', n, n, u, ldu, WORK.ptr(ir), ldwrkr);
        Gemm('N', 'N', m, n, n, one, a, lda, WORK.ptr(ir), ldwrkr, zero, u,
             ldu);
      } else if (wntqa) {
        // Path 4 (M >> N, JOBZ='A')
        // M left singular vectors to be computed in U and
        // N right singular vectors to be computed in VT
        iu = 1;
        // WORK(IU) is N by N
        ldwrku = n;
        itau = iu + ldwrku * n;
        nwork = itau + n;
        // Compute A=Q*R, copying result to U
        // Workspace: need   N*N [U] + N [tau] + N    [work]
        // Workspace: prefer N*N [U] + N [tau] + N*NB [work]
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        Lacpy('L', m, n, a, lda, u, ldu);
        // Generate Q in U
        // Workspace: need   N*N [U] + N [tau] + M    [work]
        // Workspace: prefer N*N [U] + N [tau] + M*NB [work]
        Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Produce R in A, zeroing out other entries
        Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
        ie = itau;
        itauq = ie + n;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in A
        // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [work]
        // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [work]
        Gebrd(n, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in WORK(IU) and computing right
        // singular vectors of bidiagonal matrix in VT
        // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + BDSPAC
        info = Bdsdc('U', 'I', n, s, WORK.ptr(ie), WORK.ptr(iu), n, vt, ldvt,
                     dum, idum, WORK.ptr(nwork), iwork);
        // Overwrite WORK(IU) by left singular vectors of R and VT
        // by right singular vectors of R
        // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [work]
        // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [work]
        Unmbr('Q', 'L', 'N', n, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iu),
              ldwrku, WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply Q in U by left singular vectors of R in
        // WORK(IU), storing result in A
        // Workspace: need   N*N [U]
        Gemm('N', 'N', m, n, n, one, u, ldu, WORK.ptr(iu), ldwrku, zero, a,
             lda);
        // Copy left singular vectors of A from A to U
        Lacpy('F', m, n, a, lda, u, ldu);
      }
    } else {
      // M .LT. MNTHR
      // Path 5 (M >= N, but not much larger)
      // Reduce to bidiagonal form without QR decomposition
      ie = 1;
      itauq = ie + n;
      itaup = itauq + n;
      nwork = itaup + n;
      // Bidiagonalize A
      // Workspace: need   3*N [e, tauq, taup] + M        [work]
      // Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [work]
      Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(nwork), lwork - nwork + 1);
      if (wntqn) {
        // Path 5n (M >= N, JOBZ='N')
        // Perform bidiagonal SVD, only computing singular values
        // Workspace: need   3*N [e, tauq, taup] + BDSPAC
        info = Bdsdc('U', 'N', n, s, WORK.ptr(ie), dum, 1, dum, 1, dum, idum,
                     WORK.ptr(nwork), iwork);
      } else if (wntqo) {
        // Path 5o (M >= N, JOBZ='O')
        iu = nwork;
        if (lwork >= m * n + 3 * n + bdspac) {
          // WORK( IU ) is M by N
          ldwrku = m;
          nwork = iu + ldwrku * n;
          Laset('F', m, n, zero, zero, WORK.ptr(iu), ldwrku);
          // IR is unused; silence compile warnings
          ir = -1;
        } else {
          // WORK( IU ) is N by N
          ldwrku = n;
          nwork = iu + ldwrku * n;
          // WORK(IR) is LDWRKR by N
          ir = nwork;
          ldwrkr = (lwork - n * n - 3 * n) / n;
        }
        nwork = iu + ldwrku * n;
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in WORK(IU) and computing right
        // singular vectors of bidiagonal matrix in VT
        // Workspace: need   3*N [e, tauq, taup] + N*N [U] + BDSPAC
        info = Bdsdc('U', 'I', n, s, WORK.ptr(ie), WORK.ptr(iu), ldwrku, vt,
                     ldvt, dum, idum, WORK.ptr(nwork), iwork);
        // Overwrite VT by right singular vectors of A
        // Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
        // Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
        Unmbr('P', 'R', 'T', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
        if (lwork >= m * n + 3 * n + bdspac) {
          // Path 5o-fast
          // Overwrite WORK(IU) by left singular vectors of A
          // Workspace: need   3*N [e, tauq, taup] + M*N [U] + N    [work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*N [U] + N*NB [work]
          Unmbr('Q', 'L', 'N', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iu),
                ldwrku, WORK.ptr(nwork), lwork - nwork + 1);
          // Copy left singular vectors of A from WORK(IU) to A
          Lacpy('F', m, n, WORK.ptr(iu), ldwrku, a, lda);
        } else {
          // Path 5o-slow
          // Generate Q in A
          // Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
          // Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
          Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(nwork),
                lwork - nwork + 1);
          // Multiply Q in A by left singular vectors of
          // bidiagonal matrix in WORK(IU), storing result in
          // WORK(IR) and copying to A
          // Workspace: need   3*N [e, tauq, taup] + N*N [U] + NB*N [R]
          // Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + M*N  [R]
          for (i = 1; i <= m; i += ldwrkr) {
            chunk = Min<blasint>(m - i + 1, ldwrkr);
            Gemm('N', 'N', chunk, n, n, one, A.ptr(i, 1), lda, WORK.ptr(iu),
                 ldwrku, zero, WORK.ptr(ir), ldwrkr);
            Lacpy('F', chunk, n, WORK.ptr(ir), ldwrkr, A.ptr(i, 1), lda);
          }
        }
      } else if (wntqs) {
        // Path 5s (M >= N, JOBZ='S')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in VT
        // Workspace: need   3*N [e, tauq, taup] + BDSPAC
        Laset('F', m, n, zero, zero, u, ldu);
        info = Bdsdc('U', 'I', n, s, WORK.ptr(ie), u, ldu, vt, ldvt, dum, idum,
                     WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of A and VT
        // by right singular vectors of A
        // Workspace: need   3*N [e, tauq, taup] + N    [work]
        // Workspace: prefer 3*N [e, tauq, taup] + N*NB [work]
        Unmbr('Q', 'L', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      } else if (wntqa) {
        // Path 5a (M >= N, JOBZ='A')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in VT
        // Workspace: need   3*N [e, tauq, taup] + BDSPAC
        Laset('F', m, m, zero, zero, u, ldu);
        info = Bdsdc('U', 'I', n, s, WORK.ptr(ie), u, ldu, vt, ldvt, dum, idum,
                     WORK.ptr(nwork), iwork);
        // Set the right corner of U to identity matrix
        if (m > n) {
          Laset('F', m - n, m - n, zero, one, U.ptr(n + 1, n + 1), ldu);
        }
        // Overwrite U by left singular vectors of A and VT
        // by right singular vectors of A
        // Workspace: need   3*N [e, tauq, taup] + M    [work]
        // Workspace: prefer 3*N [e, tauq, taup] + M*NB [work]
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', n, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      }
    }
  } else {
    // A has more columns than rows. If A has sufficiently more
    // columns than rows, first reduce using the LQ decomposition (if
    // sufficient workspace available)
    if (n >= mnthr) {
      if (wntqn) {
        // Path 1t (N >> M, JOBZ='N')
        // No singular vectors to be computed
        itau = 1;
        nwork = itau + m;
        // Compute A=L*Q
        // Workspace: need   M [tau] + M [work]
        // Workspace: prefer M [tau] + M*NB [work]
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Zero out above L
        Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
        ie = 1;
        itauq = ie + m;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in A
        // Workspace: need   3*M [e, tauq, taup] + M      [work]
        // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [work]
        Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        nwork = ie + m;
        // Perform bidiagonal SVD, computing singular values only
        // Workspace: need   M [e] + BDSPAC
        info = Bdsdc('U', 'N', m, s, WORK.ptr(ie), dum, 1, dum, 1, dum, idum,
                     WORK.ptr(nwork), iwork);
      } else if (wntqo) {
        // Path 2t (N >> M, JOBZ='O')
        // M right singular vectors to be overwritten on A and
        // M left singular vectors to be computed in U
        ivt = 1;
        // WORK(IVT) is M by M
        // WORK(IL)  is M by M; it is later resized to M by chunk for gemm
        il = ivt + m * m;
        if (lwork >= m * n + m * m + 3 * m + bdspac) {
          ldwrkl = m;
          chunk = n;
        } else {
          ldwrkl = m;
          chunk = (lwork - m * m) / m;
        }
        itau = il + ldwrkl * m;
        nwork = itau + m;
        // Compute A=L*Q
        // Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
        // Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy L to WORK(IL), zeroing about above it
        Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwrkl);
        Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(il + ldwrkl), ldwrkl);
        // Generate Q in A
        // Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
        // Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
        Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = itau;
        itauq = ie + m;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in WORK(IL)
        // Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M [work]
        // Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + 2*M*NB
        // [work]
        Gebrd(m, m, WORK.ptr(il), ldwrkl, s, WORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U, and computing right singular
        // vectors of bidiagonal matrix in WORK(IVT)
        // Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + BDSPAC
        info = Bdsdc('U', 'I', m, s, WORK.ptr(ie), u, ldu, WORK.ptr(ivt), m,
                     dum, idum, WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of L and WORK(IVT)
        // by right singular vectors of L
        // Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M [work]
        // Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M*NB
        // [work]
        Unmbr('Q', 'L', 'N', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itauq), u,
              ldu, WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itaup),
              WORK.ptr(ivt), m, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply right singular vectors of L in WORK(IVT) by Q
        // in A, storing result in WORK(IL) and copying to A
        // Workspace: need   M*M [VT] + M*M [L]
        // Workspace: prefer M*M [VT] + M*N [L]
        // At this point, L is resized as M by chunk.
        for (i = 1; i <= n; i += chunk) {
          blk = Min<blasint>(n - i + 1, chunk);
          Gemm('N', 'N', m, blk, m, one, WORK.ptr(ivt), m, A.ptr(1, i), lda,
               zero, WORK.ptr(il), ldwrkl);
          Lacpy('F', m, blk, WORK.ptr(il), ldwrkl, A.ptr(1, i), lda);
        }
      } else if (wntqs) {
        // Path 3t (N >> M, JOBZ='S')
        // M right singular vectors to be computed in VT and
        // M left singular vectors to be computed in U
        il = 1;
        // WORK(IL) is M by M
        ldwrkl = m;
        itau = il + ldwrkl * m;
        nwork = itau + m;
        // Compute A=L*Q
        // Workspace: need   M*M [L] + M [tau] + M    [work]
        // Workspace: prefer M*M [L] + M [tau] + M*NB [work]
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy L to WORK(IL), zeroing out above it
        Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwrkl);
        Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(il + ldwrkl), ldwrkl);
        // Generate Q in A
        // Workspace: need   M*M [L] + M [tau] + M    [work]
        // Workspace: prefer M*M [L] + M [tau] + M*NB [work]
        Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = itau;
        itauq = ie + m;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in WORK(IU).
        // Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M      [work]
        // Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
        Gebrd(m, m, WORK.ptr(il), ldwrkl, s, WORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in VT
        // Workspace: need   M*M [L] + 3*M [e, tauq, taup] + BDSPAC
        info = Bdsdc('U', 'I', m, s, WORK.ptr(ie), u, ldu, vt, ldvt, dum, idum,
                     WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of L and VT
        // by right singular vectors of L
        // Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M    [work]
        // Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
        Unmbr('Q', 'L', 'N', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itauq), u,
              ldu, WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itaup), vt,
              ldvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply right singular vectors of L in WORK(IL) by
        // Q in A, storing result in VT
        // Workspace: need   M*M [L]
        Lacpy('F', m, m, vt, ldvt, WORK.ptr(il), ldwrkl);
        Gemm('N', 'N', m, n, m, one, WORK.ptr(il), ldwrkl, a, lda, zero, vt,
             ldvt);
      } else if (wntqa) {
        // Path 4t (N >> M, JOBZ='A')
        // N right singular vectors to be computed in VT and
        // M left singular vectors to be computed in U
        ivt = 1;
        // WORK(IVT) is M by M
        ldwkvt = m;
        itau = ivt + ldwkvt * m;
        nwork = itau + m;
        // Compute A=L*Q, copying result to VT
        // Workspace: need   M*M [VT] + M [tau] + M    [work]
        // Workspace: prefer M*M [VT] + M [tau] + M*NB [work]
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        Lacpy('U', m, n, a, lda, vt, ldvt);
        // Generate Q in VT
        // Workspace: need   M*M [VT] + M [tau] + N    [work]
        // Workspace: prefer M*M [VT] + M [tau] + N*NB [work]
        Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Produce L in A, zeroing out other entries
        Laset('U', m - 1, m - 1, zero, zero, A.ptr(1, 2), lda);
        ie = itau;
        itauq = ie + m;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in A
        // Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + M      [work]
        // Workspace: prefer M*M [VT] + 3*M [e, tauq, taup] + 2*M*NB [work]
        Gebrd(m, m, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in WORK(IVT)
        // Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + BDSPAC
        info = Bdsdc('U', 'I', m, s, WORK.ptr(ie), u, ldu, WORK.ptr(ivt),
                     ldwkvt, dum, idum, WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of L and WORK(IVT)
        // by right singular vectors of L
        // Workspace: need   M*M [VT] + 3*M [e, tauq, taup]+ M    [work]
        // Workspace: prefer M*M [VT] + 3*M [e, tauq, taup]+ M*NB [work]
        Unmbr('Q', 'L', 'N', m, m, m, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', m, m, m, a, lda, WORK.ptr(itaup), WORK.ptr(ivt),
              ldwkvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply right singular vectors of L in WORK(IVT) by
        // Q in VT, storing result in A
        // Workspace: need   M*M [VT]
        Gemm('N', 'N', m, n, m, one, WORK.ptr(ivt), ldwkvt, vt, ldvt, zero, a,
             lda);
        // Copy right singular vectors of A from A to VT
        Lacpy('F', m, n, a, lda, vt, ldvt);
      }
    } else {
      // N .LT. MNTHR
      // Path 5t (N > M, but not much larger)
      // Reduce to bidiagonal form without LQ decomposition
      ie = 1;
      itauq = ie + m;
      itaup = itauq + m;
      nwork = itaup + m;
      // Bidiagonalize A
      // Workspace: need   3*M [e, tauq, taup] + N        [work]
      // Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [work]
      Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(nwork), lwork - nwork + 1);
      if (wntqn) {
        // Path 5tn (N > M, JOBZ='N')
        // Perform bidiagonal SVD, only computing singular values
        // Workspace: need   3*M [e, tauq, taup] + BDSPAC
        info = Bdsdc('L', 'N', m, s, WORK.ptr(ie), dum, 1, dum, 1, dum, idum,
                     WORK.ptr(nwork), iwork);
      } else if (wntqo) {
        // Path 5to (N > M, JOBZ='O')
        ldwkvt = m;
        ivt = nwork;
        if (lwork >= m * n + 3 * m + bdspac) {
          // WORK( IVT ) is M by N
          Laset('F', m, n, zero, zero, WORK.ptr(ivt), ldwkvt);
          nwork = ivt + ldwkvt * n;
          // IL is unused; silence compile warnings
          il = -1;
        } else {
          // WORK( IVT ) is M by M
          nwork = ivt + ldwkvt * m;
          il = nwork;
          // WORK(IL) is M by CHUNK
          chunk = (lwork - m * m - 3 * m) / m;
        }
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in WORK(IVT)
        // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + BDSPAC
        info = Bdsdc('L', 'I', m, s, WORK.ptr(ie), u, ldu, WORK.ptr(ivt),
                     ldwkvt, dum, idum, WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of A
        // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
        // Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        if (lwork >= m * n + 3 * m + bdspac) {
          // Path 5to-fast
          // Overwrite WORK(IVT) by left singular vectors of A
          // Workspace: need   3*M [e, tauq, taup] + M*N [VT] + M    [work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*N [VT] + M*NB [work]
          Unmbr('P', 'R', 'T', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(ivt),
                ldwkvt, WORK.ptr(nwork), lwork - nwork + 1);
          // Copy right singular vectors of A from WORK(IVT) to A
          Lacpy('F', m, n, WORK.ptr(ivt), ldwkvt, a, lda);
        } else {
          // Path 5to-slow
          // Generate P**T in A
          // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
          Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(nwork),
                lwork - nwork + 1);
          // Multiply Q in A by right singular vectors of
          // bidiagonal matrix in WORK(IVT), storing result in
          // WORK(IL) and copying to A
          // Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M*NB [L]
          // Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*N  [L]
          for (i = 1; i <= n; i += chunk) {
            blk = Min<blasint>(n - i + 1, chunk);
            Gemm('N', 'N', m, blk, m, one, WORK.ptr(ivt), ldwkvt, A.ptr(1, i),
                 lda, zero, WORK.ptr(il), m);
            Lacpy('F', m, blk, WORK.ptr(il), m, A.ptr(1, i), lda);
          }
        }
      } else if (wntqs) {
        // Path 5ts (N > M, JOBZ='S')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in VT
        // Workspace: need   3*M [e, tauq, taup] + BDSPAC
        Laset('F', m, n, zero, zero, vt, ldvt);
        info = Bdsdc('L', 'I', m, s, WORK.ptr(ie), u, ldu, vt, ldvt, dum, idum,
                     WORK.ptr(nwork), iwork);
        // Overwrite U by left singular vectors of A and VT
        // by right singular vectors of A
        // Workspace: need   3*M [e, tauq, taup] + M    [work]
        // Workspace: prefer 3*M [e, tauq, taup] + M*NB [work]
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      } else if (wntqa) {
        // Path 5ta (N > M, JOBZ='A')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in U and computing right singular
        // vectors of bidiagonal matrix in VT
        // Workspace: need   3*M [e, tauq, taup] + BDSPAC
        Laset('F', n, n, zero, zero, vt, ldvt);
        info = Bdsdc('L', 'I', m, s, WORK.ptr(ie), u, ldu, vt, ldvt, dum, idum,
                     WORK.ptr(nwork), iwork);
        // Set the right corner of VT to identity matrix
        if (n > m) {
          Laset('F', n - m, n - m, zero, one, VT.ptr(m + 1, m + 1), ldvt);
        }
        // Overwrite U by left singular vectors of A and VT
        // by right singular vectors of A
        // Workspace: need   3*M [e, tauq, taup] + N    [work]
        // Workspace: prefer 3*M [e, tauq, taup] + N*NB [work]
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        Unmbr('P', 'R', 'T', n, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      }
    }
  }
  // Undo scaling if necessary
  if (iscl == 1) {
    if (anrm > bignum) Lascl('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
    if (anrm < smlnum) Lascl('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
  }
  // Return optimal workspace in WORK(1)
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// CGESDD / ZGESDD.
template <typename T>
// NOLINTNEXTLINE(google-readability-function-size,readability-function-size)
blasint GesddComplex(char jobz, blasint m, blasint n, T* a, blasint lda,
                     RealT<T>* s, T* u, blasint ldu, T* vt, blasint ldvt,
                     T* work, blasint lwork, RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;

  const T czero = T(0);
  const T cone = T(1);
  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> A(a, lda);
  T cdum[1];
  const Vector<T> CDUM(cdum);
  R dum[1];
  const Vector<R> DUM(dum);
  blasint idum[1];
  const Vector<blasint> IWORK(iwork);
  const Vector<R> RWORK(rwork);
  const Vector<R> S(s);
  const Matrix<T> U(u, ldu);
  const Matrix<T> VT(vt, ldvt);
  const Vector<T> WORK(work);
  blasint blk, chunk, i, ie, il, ir, iru, irvt, iscl, itau, itaup, itauq, iu,
      ivt, ldwkvt, ldwrkl, ldwrkr, ldwrku, lwork_zgebrd_mm, lwork_zgebrd_mn,
      lwork_zgebrd_nn, lwork_zgelqf_mn, lwork_zgeqrf_mn, lwork_zungbr_p_mn,
      lwork_zungbr_p_nn, lwork_zungbr_q_mm, lwork_zungbr_q_mn, lwork_zunglq_mn,
      lwork_zunglq_nn, lwork_zungqr_mm, lwork_zungqr_mn, lwork_zunmbr_prc_mm,
      lwork_zunmbr_prc_mn, lwork_zunmbr_prc_nn, lwork_zunmbr_qln_mm,
      lwork_zunmbr_qln_mn, lwork_zunmbr_qln_nn, maxwrk, minmn, minwrk, mnthr1,
      mnthr2, nrwork, nwork, wrkbl;
  bool lquery, wntqa, wntqas, wntqn, wntqo, wntqs;
  R anrm, bignum, eps, smlnum;
  blasint info = 0;
  minmn = Min<blasint>(m, n);
  mnthr1 = static_cast<blasint>(minmn * R(17) / R(9));
  mnthr2 = static_cast<blasint>(minmn * R(5) / R(3));
  wntqa = Lsame(jobz, 'A');
  wntqs = Lsame(jobz, 'S');
  wntqas = wntqa || wntqs;
  wntqo = Lsame(jobz, 'O');
  wntqn = Lsame(jobz, 'N');
  lquery = (lwork == -1);
  minwrk = 1;
  maxwrk = 1;
  if (!(wntqa || wntqs || wntqo || wntqn)) {
    info = -1;
  } else if (m < 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldu < 1 || (wntqas && ldu < m) || (wntqo && m < n && ldu < m)) {
    info = -8;
  } else if (ldvt < 1 || (wntqa && ldvt < n) || (wntqs && ldvt < minmn) ||
             (wntqo && m >= n && ldvt < n)) {
    info = -10;
  }
  // Compute workspace
  // Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace allocated at that point in the code,
  // as well as the preferred amount for good performance.
  // CWorkspace refers to complex workspace, and RWorkspace to
  // real workspace. NB refers to the optimal block size for the
  // immediately following subroutine, as returned by ILAENV.)
  if (info == 0) {
    minwrk = 1;
    maxwrk = 1;
    if (m >= n && minmn > 0) {
      // There is no complex work space needed for bidiagonal SVD
      // The real work space needed for bidiagonal SVD (dbdsdc) is
      // BDSPAC = 3*N*N + 4*N for singular values and vectors;
      // BDSPAC = 4*N         for singular values only;
      // not including e, RU, and RVT matrices.
      // Compute space preferred for each routine
      Gebrd(m, n, CDUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), CDUM.ptr(1),
            CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zgebrd_mn = static_cast<blasint>(std::real(CDUM(1)));
      Gebrd(n, n, CDUM.ptr(1), n, DUM.ptr(1), DUM.ptr(1), CDUM.ptr(1),
            CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zgebrd_nn = static_cast<blasint>(std::real(CDUM(1)));
      Geqrf(m, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zgeqrf_mn = static_cast<blasint>(std::real(CDUM(1)));
      Ungbr('P', n, n, n, CDUM.ptr(1), n, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungbr_p_nn = static_cast<blasint>(std::real(CDUM(1)));
      Ungbr('Q', m, m, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungbr_q_mm = static_cast<blasint>(std::real(CDUM(1)));
      Ungbr('Q', m, n, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungbr_q_mn = static_cast<blasint>(std::real(CDUM(1)));
      Ungqr(m, m, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungqr_mm = static_cast<blasint>(std::real(CDUM(1)));
      Ungqr(m, n, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungqr_mn = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('P', 'R', 'C', n, n, n, CDUM.ptr(1), n, CDUM.ptr(1), CDUM.ptr(1), n,
            CDUM.ptr(1), -1);
      lwork_zunmbr_prc_nn = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('Q', 'L', 'N', m, m, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), m,
            CDUM.ptr(1), -1);
      lwork_zunmbr_qln_mm = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('Q', 'L', 'N', m, n, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), m,
            CDUM.ptr(1), -1);
      lwork_zunmbr_qln_mn = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('Q', 'L', 'N', n, n, n, CDUM.ptr(1), n, CDUM.ptr(1), CDUM.ptr(1), n,
            CDUM.ptr(1), -1);
      lwork_zunmbr_qln_nn = static_cast<blasint>(std::real(CDUM(1)));
      if (m >= mnthr1) {
        if (wntqn) {
          // Path 1 (M >> N, JOBZ='N')
          maxwrk = n + lwork_zgeqrf_mn;
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zgebrd_nn);
          minwrk = 3 * n;
        } else if (wntqo) {
          // Path 2 (M >> N, JOBZ='O')
          wrkbl = n + lwork_zgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, n + lwork_zungqr_mn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zgebrd_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zunmbr_qln_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zunmbr_prc_nn);
          maxwrk = m * n + n * n + wrkbl;
          minwrk = 2 * n * n + 3 * n;
        } else if (wntqs) {
          // Path 3 (M >> N, JOBZ='S')
          wrkbl = n + lwork_zgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, n + lwork_zungqr_mn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zgebrd_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zunmbr_qln_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zunmbr_prc_nn);
          maxwrk = n * n + wrkbl;
          minwrk = n * n + 3 * n;
        } else if (wntqa) {
          // Path 4 (M >> N, JOBZ='A')
          wrkbl = n + lwork_zgeqrf_mn;
          wrkbl = Max<blasint>(wrkbl, n + lwork_zungqr_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zgebrd_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zunmbr_qln_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * n + lwork_zunmbr_prc_nn);
          maxwrk = n * n + wrkbl;
          minwrk = n * n + Max<blasint>(3 * n, n + m);
        }
      } else if (m >= mnthr2) {
        // Path 5 (M >> N, but not as much as MNTHR1)
        maxwrk = 2 * n + lwork_zgebrd_mn;
        minwrk = 2 * n + m;
        if (wntqo) {
          // Path 5o (M >> N, JOBZ='O')
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr_p_nn);
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr_q_mn);
          maxwrk = maxwrk + m * n;
          minwrk = minwrk + n * n;
        } else if (wntqs) {
          // Path 5s (M >> N, JOBZ='S')
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr_p_nn);
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr_q_mn);
        } else if (wntqa) {
          // Path 5a (M >> N, JOBZ='A')
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr_p_nn);
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zungbr_q_mm);
        }
      } else {
        // Path 6 (M >= N, but not much larger)
        maxwrk = 2 * n + lwork_zgebrd_mn;
        minwrk = 2 * n + m;
        if (wntqo) {
          // Path 6o (M >= N, JOBZ='O')
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr_prc_nn);
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr_qln_mn);
          maxwrk = maxwrk + m * n;
          minwrk = minwrk + n * n;
        } else if (wntqs) {
          // Path 6s (M >= N, JOBZ='S')
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr_qln_mn);
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr_prc_nn);
        } else if (wntqa) {
          // Path 6a (M >= N, JOBZ='A')
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr_qln_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * n + lwork_zunmbr_prc_nn);
        }
      }
    } else if (minmn > 0) {
      // There is no complex work space needed for bidiagonal SVD
      // The real work space needed for bidiagonal SVD (dbdsdc) is
      // BDSPAC = 3*M*M + 4*M for singular values and vectors;
      // BDSPAC = 4*M         for singular values only;
      // not including e, RU, and RVT matrices.
      // Compute space preferred for each routine
      Gebrd(m, n, CDUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), CDUM.ptr(1),
            CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zgebrd_mn = static_cast<blasint>(std::real(CDUM(1)));
      Gebrd(m, m, CDUM.ptr(1), m, DUM.ptr(1), DUM.ptr(1), CDUM.ptr(1),
            CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zgebrd_mm = static_cast<blasint>(std::real(CDUM(1)));
      Gelqf(m, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zgelqf_mn = static_cast<blasint>(std::real(CDUM(1)));
      Ungbr('P', m, n, m, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungbr_p_mn = static_cast<blasint>(std::real(CDUM(1)));
      Ungbr('P', n, n, m, CDUM.ptr(1), n, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungbr_p_nn = static_cast<blasint>(std::real(CDUM(1)));
      Ungbr('Q', m, m, n, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zungbr_q_mm = static_cast<blasint>(std::real(CDUM(1)));
      Unglq(m, n, m, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zunglq_mn = static_cast<blasint>(std::real(CDUM(1)));
      Unglq(n, n, m, CDUM.ptr(1), n, CDUM.ptr(1), CDUM.ptr(1), -1);
      lwork_zunglq_nn = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('P', 'R', 'C', m, m, m, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), m,
            CDUM.ptr(1), -1);
      lwork_zunmbr_prc_mm = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('P', 'R', 'C', m, n, m, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), m,
            CDUM.ptr(1), -1);
      lwork_zunmbr_prc_mn = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('P', 'R', 'C', n, n, m, CDUM.ptr(1), n, CDUM.ptr(1), CDUM.ptr(1), n,
            CDUM.ptr(1), -1);
      lwork_zunmbr_prc_nn = static_cast<blasint>(std::real(CDUM(1)));
      Unmbr('Q', 'L', 'N', m, m, m, CDUM.ptr(1), m, CDUM.ptr(1), CDUM.ptr(1), m,
            CDUM.ptr(1), -1);
      lwork_zunmbr_qln_mm = static_cast<blasint>(std::real(CDUM(1)));
      if (n >= mnthr1) {
        if (wntqn) {
          // Path 1t (N >> M, JOBZ='N')
          maxwrk = m + lwork_zgelqf_mn;
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zgebrd_mm);
          minwrk = 3 * m;
        } else if (wntqo) {
          // Path 2t (N >> M, JOBZ='O')
          wrkbl = m + lwork_zgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, m + lwork_zunglq_mn);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zgebrd_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zunmbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zunmbr_prc_mm);
          maxwrk = m * n + m * m + wrkbl;
          minwrk = 2 * m * m + 3 * m;
        } else if (wntqs) {
          // Path 3t (N >> M, JOBZ='S')
          wrkbl = m + lwork_zgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, m + lwork_zunglq_mn);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zgebrd_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zunmbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zunmbr_prc_mm);
          maxwrk = m * m + wrkbl;
          minwrk = m * m + 3 * m;
        } else if (wntqa) {
          // Path 4t (N >> M, JOBZ='A')
          wrkbl = m + lwork_zgelqf_mn;
          wrkbl = Max<blasint>(wrkbl, m + lwork_zunglq_nn);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zgebrd_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zunmbr_qln_mm);
          wrkbl = Max<blasint>(wrkbl, 2 * m + lwork_zunmbr_prc_mm);
          maxwrk = m * m + wrkbl;
          minwrk = m * m + Max<blasint>(3 * m, m + n);
        }
      } else if (n >= mnthr2) {
        // Path 5t (N >> M, but not as much as MNTHR1)
        maxwrk = 2 * m + lwork_zgebrd_mn;
        minwrk = 2 * m + n;
        if (wntqo) {
          // Path 5to (N >> M, JOBZ='O')
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr_q_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr_p_mn);
          maxwrk = maxwrk + m * n;
          minwrk = minwrk + m * m;
        } else if (wntqs) {
          // Path 5ts (N >> M, JOBZ='S')
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr_q_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr_p_mn);
        } else if (wntqa) {
          // Path 5ta (N >> M, JOBZ='A')
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr_q_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zungbr_p_nn);
        }
      } else {
        // Path 6t (N > M, but not much larger)
        maxwrk = 2 * m + lwork_zgebrd_mn;
        minwrk = 2 * m + n;
        if (wntqo) {
          // Path 6to (N > M, JOBZ='O')
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr_qln_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr_prc_mn);
          maxwrk = maxwrk + m * n;
          minwrk = minwrk + m * m;
        } else if (wntqs) {
          // Path 6ts (N > M, JOBZ='S')
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr_qln_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr_prc_mn);
        } else if (wntqa) {
          // Path 6ta (N > M, JOBZ='A')
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr_qln_mm);
          maxwrk = Max<blasint>(maxwrk, 2 * m + lwork_zunmbr_prc_nn);
        }
      }
    }
    maxwrk = Max<blasint>(maxwrk, minwrk);
  }
  if (info == 0) {
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -12;
    }
  }
  if (info != 0) {
    Xerbla<T>("GESDD", -info);
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
  if (std::isnan(anrm)) {
    info = -4;
    return info;
  }
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
    if (m >= mnthr1) {
      if (wntqn) {
        // Path 1 (M >> N, JOBZ='N')
        // No singular vectors to be computed
        itau = 1;
        nwork = itau + n;
        // Compute A=Q*R
        // CWorkspace: need   N [tau] + N    [work]
        // CWorkspace: prefer N [tau] + N*NB [work]
        // RWorkspace: need   0
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Zero out below R
        Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
        ie = 1;
        itauq = 1;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in A
        // CWorkspace: need   2*N [tauq, taup] + N      [work]
        // CWorkspace: prefer 2*N [tauq, taup] + 2*N*NB [work]
        // RWorkspace: need   N [e]
        Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        nrwork = ie + n;
        // Perform bidiagonal SVD, compute singular values only
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + BDSPAC
        info = Bdsdc<R>('U', 'N', n, s, RWORK.ptr(ie), dum, 1, dum, 1, dum,
                        idum, RWORK.ptr(nrwork), iwork);
      } else if (wntqo) {
        // Path 2 (M >> N, JOBZ='O')
        // N left singular vectors to be overwritten on A and
        // N right singular vectors to be computed in VT
        iu = 1;
        // WORK(IU) is N by N
        ldwrku = n;
        ir = iu + ldwrku * n;
        if (lwork >= m * n + n * n + 3 * n) {
          // WORK(IR) is M by N
          ldwrkr = m;
        } else {
          ldwrkr = (lwork - n * n - 3 * n) / n;
        }
        itau = ir + ldwrkr * n;
        nwork = itau + n;
        // Compute A=Q*R
        // CWorkspace: need   N*N [U] + N*N [R] + N [tau] + N    [work]
        // CWorkspace: prefer N*N [U] + N*N [R] + N [tau] + N*NB [work]
        // RWorkspace: need   0
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy R to WORK( IR ), zeroing out below it
        Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
        Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(ir + 1), ldwrkr);
        // Generate Q in A
        // CWorkspace: need   N*N [U] + N*N [R] + N [tau] + N    [work]
        // CWorkspace: prefer N*N [U] + N*N [R] + N [tau] + N*NB [work]
        // RWorkspace: need   0
        Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = 1;
        itauq = itau;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in WORK(IR)
        // CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N [work]
        // CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + 2*N*NB
        // [work] RWorkspace: need   N [e]
        Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of R in WORK(IRU) and computing right singular vectors
        // of R in WORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        iru = ie + n;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
        // Overwrite WORK(IU) by the left singular vectors of R
        // CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacp2('F', n, n, RWORK.ptr(iru), n, WORK.ptr(iu), ldwrku);
        Unmbr('Q', 'L', 'N', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq),
              WORK.ptr(iu), ldwrku, WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by the right singular vectors of R
        // CWorkspace: need   N*N [U] + N*N [R] + 2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer N*N [U] + N*N [R] + 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacp2('F', n, n, RWORK.ptr(irvt), n, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup), vt,
              ldvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply Q in A by left singular vectors of R in
        // WORK(IU), storing result in WORK(IR) and copying to A
        // CWorkspace: need   N*N [U] + N*N [R]
        // CWorkspace: prefer N*N [U] + M*N [R]
        // RWorkspace: need   0
        for (i = 1; i <= m; i += ldwrkr) {
          chunk = Min<blasint>(m - i + 1, ldwrkr);
          Gemm('N', 'N', chunk, n, n, cone, A.ptr(i, 1), lda, WORK.ptr(iu),
               ldwrku, czero, WORK.ptr(ir), ldwrkr);
          Lacpy('F', chunk, n, WORK.ptr(ir), ldwrkr, A.ptr(i, 1), lda);
        }
      } else if (wntqs) {
        // Path 3 (M >> N, JOBZ='S')
        // N left singular vectors to be computed in U and
        // N right singular vectors to be computed in VT
        ir = 1;
        // WORK(IR) is N by N
        ldwrkr = n;
        itau = ir + ldwrkr * n;
        nwork = itau + n;
        // Compute A=Q*R
        // CWorkspace: need   N*N [R] + N [tau] + N    [work]
        // CWorkspace: prefer N*N [R] + N [tau] + N*NB [work]
        // RWorkspace: need   0
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy R to WORK(IR), zeroing out below it
        Lacpy('U', n, n, a, lda, WORK.ptr(ir), ldwrkr);
        Laset('L', n - 1, n - 1, czero, czero, WORK.ptr(ir + 1), ldwrkr);
        // Generate Q in A
        // CWorkspace: need   N*N [R] + N [tau] + N    [work]
        // CWorkspace: prefer N*N [R] + N [tau] + N*NB [work]
        // RWorkspace: need   0
        Ungqr(m, n, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = 1;
        itauq = itau;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in WORK(IR)
        // CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N      [work]
        // CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + 2*N*NB [work]
        // RWorkspace: need   N [e]
        Gebrd(n, n, WORK.ptr(ir), ldwrkr, s, RWORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        iru = ie + n;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of R
        // CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacp2('F', n, n, RWORK.ptr(iru), n, u, ldu);
        Unmbr('Q', 'L', 'N', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itauq), u,
              ldu, WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of R
        // CWorkspace: need   N*N [R] + 2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer N*N [R] + 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacp2('F', n, n, RWORK.ptr(irvt), n, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, n, WORK.ptr(ir), ldwrkr, WORK.ptr(itaup), vt,
              ldvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply Q in A by left singular vectors of R in
        // WORK(IR), storing result in U
        // CWorkspace: need   N*N [R]
        // RWorkspace: need   0
        Lacpy('F', n, n, u, ldu, WORK.ptr(ir), ldwrkr);
        Gemm('N', 'N', m, n, n, cone, a, lda, WORK.ptr(ir), ldwrkr, czero, u,
             ldu);
      } else if (wntqa) {
        // Path 4 (M >> N, JOBZ='A')
        // M left singular vectors to be computed in U and
        // N right singular vectors to be computed in VT
        iu = 1;
        // WORK(IU) is N by N
        ldwrku = n;
        itau = iu + ldwrku * n;
        nwork = itau + n;
        // Compute A=Q*R, copying result to U
        // CWorkspace: need   N*N [U] + N [tau] + N    [work]
        // CWorkspace: prefer N*N [U] + N [tau] + N*NB [work]
        // RWorkspace: need   0
        Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        Lacpy('L', m, n, a, lda, u, ldu);
        // Generate Q in U
        // CWorkspace: need   N*N [U] + N [tau] + M    [work]
        // CWorkspace: prefer N*N [U] + N [tau] + M*NB [work]
        // RWorkspace: need   0
        Ungqr(m, m, n, u, ldu, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Produce R in A, zeroing out below it
        Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
        ie = 1;
        itauq = itau;
        itaup = itauq + n;
        nwork = itaup + n;
        // Bidiagonalize R in A
        // CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N      [work]
        // CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + 2*N*NB [work]
        // RWorkspace: need   N [e]
        Gebrd(n, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        iru = ie + n;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
        // Overwrite WORK(IU) by left singular vectors of R
        // CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacp2('F', n, n, RWORK.ptr(iru), n, WORK.ptr(iu), ldwrku);
        Unmbr('Q', 'L', 'N', n, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iu),
              ldwrku, WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of R
        // CWorkspace: need   N*N [U] + 2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer N*N [U] + 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacp2('F', n, n, RWORK.ptr(irvt), n, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply Q in U by left singular vectors of R in
        // WORK(IU), storing result in A
        // CWorkspace: need   N*N [U]
        // RWorkspace: need   0
        Gemm('N', 'N', m, n, n, cone, u, ldu, WORK.ptr(iu), ldwrku, czero, a,
             lda);
        // Copy left singular vectors of A from A to U
        Lacpy('F', m, n, a, lda, u, ldu);
      }
    } else if (m >= mnthr2) {
      // MNTHR2 <= M < MNTHR1
      // Path 5 (M >> N, but not as much as MNTHR1)
      // Reduce to bidiagonal form without QR decomposition, use
      // ZUNGBR and matrix multiplication to compute singular vectors
      ie = 1;
      nrwork = ie + n;
      itauq = 1;
      itaup = itauq + n;
      nwork = itaup + n;
      // Bidiagonalize A
      // CWorkspace: need   2*N [tauq, taup] + M        [work]
      // CWorkspace: prefer 2*N [tauq, taup] + (M+N)*NB [work]
      // RWorkspace: need   N [e]
      Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(nwork), lwork - nwork + 1);
      if (wntqn) {
        // Path 5n (M >> N, JOBZ='N')
        // Compute singular values only
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + BDSPAC
        info = Bdsdc<R>('U', 'N', n, s, RWORK.ptr(ie), dum, 1, dum, 1, dum,
                        idum, RWORK.ptr(nrwork), iwork);
      } else if (wntqo) {
        iu = nwork;
        iru = nrwork;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        // Path 5o (M >> N, JOBZ='O')
        // Copy A to VT, generate P**H
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacpy('U', n, n, a, lda, vt, ldvt);
        Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Generate Q in A
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(nwork),
              lwork - nwork + 1);
        if (lwork >= m * n + 3 * n) {
          // WORK( IU ) is M by N
          ldwrku = m;
        } else {
          // WORK(IU) is LDWRKU by N
          ldwrku = (lwork - 3 * n) / n;
        }
        nwork = iu + ldwrku * n;
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Multiply real matrix RWORK(IRVT) by P**H in VT,
        // storing the result in WORK(IU), copying to VT
        // CWorkspace: need   2*N [tauq, taup] + N*N [U]
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
        Larcm(n, n, RWORK.ptr(irvt), n, vt, ldvt, WORK.ptr(iu), ldwrku,
              RWORK.ptr(nrwork));
        Lacpy('F', n, n, WORK.ptr(iu), ldwrku, vt, ldvt);
        // Multiply Q in A by real matrix RWORK(IRU), storing the
        // result in WORK(IU), copying to A
        // CWorkspace: need   2*N [tauq, taup] + N*N [U]
        // CWorkspace: prefer 2*N [tauq, taup] + M*N [U]
        // RWorkspace: need   N [e] + N*N [RU] + 2*N*N [rwork]
        // RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since
        // M < 2*N here
        nrwork = irvt;
        for (i = 1; i <= m; i += ldwrku) {
          chunk = Min<blasint>(m - i + 1, ldwrku);
          Lacrm(chunk, n, A.ptr(i, 1), lda, RWORK.ptr(iru), n, WORK.ptr(iu),
                ldwrku, RWORK.ptr(nrwork));
          Lacpy('F', chunk, n, WORK.ptr(iu), ldwrku, A.ptr(i, 1), lda);
        }
      } else if (wntqs) {
        // Path 5s (M >> N, JOBZ='S')
        // Copy A to VT, generate P**H
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacpy('U', n, n, a, lda, vt, ldvt);
        Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Copy A to U, generate Q
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacpy('L', m, n, a, lda, u, ldu);
        Ungbr('Q', m, n, n, u, ldu, WORK.ptr(itauq), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        iru = nrwork;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Multiply real matrix RWORK(IRVT) by P**H in VT,
        // storing the result in A, copying to VT
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
        Larcm(n, n, RWORK.ptr(irvt), n, vt, ldvt, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', n, n, a, lda, vt, ldvt);
        // Multiply Q in U by real matrix RWORK(IRU), storing the
        // result in A, copying to U
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since
        // M < 2*N here
        nrwork = irvt;
        Lacrm(m, n, u, ldu, RWORK.ptr(iru), n, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', m, n, a, lda, u, ldu);
      } else {
        // Path 5a (M >> N, JOBZ='A')
        // Copy A to VT, generate P**H
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacpy('U', n, n, a, lda, vt, ldvt);
        Ungbr('P', n, n, n, vt, ldvt, WORK.ptr(itaup), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Copy A to U, generate Q
        // CWorkspace: need   2*N [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacpy('L', m, n, a, lda, u, ldu);
        Ungbr('Q', m, m, n, u, ldu, WORK.ptr(itauq), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        iru = nrwork;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Multiply real matrix RWORK(IRVT) by P**H in VT,
        // storing the result in A, copying to VT
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + 2*N*N [rwork]
        Larcm(n, n, RWORK.ptr(irvt), n, vt, ldvt, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', n, n, a, lda, vt, ldvt);
        // Multiply Q in U by real matrix RWORK(IRU), storing the
        // result in A, copying to U
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N since
        // M < 2*N here
        nrwork = irvt;
        Lacrm(m, n, u, ldu, RWORK.ptr(iru), n, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', m, n, a, lda, u, ldu);
      }
    } else {
      // M .LT. MNTHR2
      // Path 6 (M >= N, but not much larger)
      // Reduce to bidiagonal form without QR decomposition
      // Use ZUNMBR to compute singular vectors
      ie = 1;
      nrwork = ie + n;
      itauq = 1;
      itaup = itauq + n;
      nwork = itaup + n;
      // Bidiagonalize A
      // CWorkspace: need   2*N [tauq, taup] + M        [work]
      // CWorkspace: prefer 2*N [tauq, taup] + (M+N)*NB [work]
      // RWorkspace: need   N [e]
      Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(nwork), lwork - nwork + 1);
      if (wntqn) {
        // Path 6n (M >= N, JOBZ='N')
        // Compute singular values only
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + BDSPAC
        info = Bdsdc<R>('U', 'N', n, s, RWORK.ptr(ie), dum, 1, dum, 1, dum,
                        idum, RWORK.ptr(nrwork), iwork);
      } else if (wntqo) {
        iu = nwork;
        iru = nrwork;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        if (lwork >= m * n + 3 * n) {
          // WORK( IU ) is M by N
          ldwrku = m;
        } else {
          // WORK( IU ) is LDWRKU by N
          ldwrku = (lwork - 3 * n) / n;
        }
        nwork = iu + ldwrku * n;
        // Path 6o (M >= N, JOBZ='O')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of A
        // CWorkspace: need   2*N [tauq, taup] + N*N [U] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*N [U] + N*NB [work]
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
        Lacp2('F', n, n, RWORK.ptr(irvt), n, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
        if (lwork >= m * n + 3 * n) {
          // Path 6o-fast
          // Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
          // Overwrite WORK(IU) by left singular vectors of A, copying
          // to A
          // CWorkspace: need   2*N [tauq, taup] + M*N [U] + N    [work]
          // CWorkspace: prefer 2*N [tauq, taup] + M*N [U] + N*NB [work]
          // RWorkspace: need   N [e] + N*N [RU]
          Laset('F', m, n, czero, czero, WORK.ptr(iu), ldwrku);
          Lacp2('F', n, n, RWORK.ptr(iru), n, WORK.ptr(iu), ldwrku);
          Unmbr('Q', 'L', 'N', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(iu),
                ldwrku, WORK.ptr(nwork), lwork - nwork + 1);
          Lacpy('F', m, n, WORK.ptr(iu), ldwrku, a, lda);
        } else {
          // Path 6o-slow
          // Generate Q in A
          // CWorkspace: need   2*N [tauq, taup] + N*N [U] + N    [work]
          // CWorkspace: prefer 2*N [tauq, taup] + N*N [U] + N*NB [work]
          // RWorkspace: need   0
          Ungbr('Q', m, n, n, a, lda, WORK.ptr(itauq), WORK.ptr(nwork),
                lwork - nwork + 1);
          // Multiply Q in A by real matrix RWORK(IRU), storing the
          // result in WORK(IU), copying to A
          // CWorkspace: need   2*N [tauq, taup] + N*N [U]
          // CWorkspace: prefer 2*N [tauq, taup] + M*N [U]
          // RWorkspace: need   N [e] + N*N [RU] + 2*N*N [rwork]
          // RWorkspace: prefer N [e] + N*N [RU] + 2*M*N [rwork] < N + 5*N*N
          // since M < 2*N here
          nrwork = irvt;
          for (i = 1; i <= m; i += ldwrku) {
            chunk = Min<blasint>(m - i + 1, ldwrku);
            Lacrm(chunk, n, A.ptr(i, 1), lda, RWORK.ptr(iru), n, WORK.ptr(iu),
                  ldwrku, RWORK.ptr(nrwork));
            Lacpy('F', chunk, n, WORK.ptr(iu), ldwrku, A.ptr(i, 1), lda);
          }
        }
      } else if (wntqs) {
        // Path 6s (M >= N, JOBZ='S')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        iru = nrwork;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of A
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
        Laset('F', m, n, czero, czero, u, ldu);
        Lacp2('F', n, n, RWORK.ptr(iru), n, u, ldu);
        Unmbr('Q', 'L', 'N', m, n, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of A
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
        Lacp2('F', n, n, RWORK.ptr(irvt), n, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      } else {
        // Path 6a (M >= N, JOBZ='A')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT] + BDSPAC
        iru = nrwork;
        irvt = iru + n * n;
        nrwork = irvt + n * n;
        info =
            Bdsdc<R>('U', 'I', n, s, RWORK.ptr(ie), RWORK.ptr(iru), n,
                     RWORK.ptr(irvt), n, dum, idum, RWORK.ptr(nrwork), iwork);
        // Set the right corner of U to identity matrix
        Laset('F', m, m, czero, czero, u, ldu);
        if (m > n) {
          Laset('F', m - n, m - n, czero, cone, U.ptr(n + 1, n + 1), ldu);
        }
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of A
        // CWorkspace: need   2*N [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + M*NB [work]
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
        Lacp2('F', n, n, RWORK.ptr(iru), n, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of A
        // CWorkspace: need   2*N [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*N [tauq, taup] + N*NB [work]
        // RWorkspace: need   N [e] + N*N [RU] + N*N [RVT]
        Lacp2('F', n, n, RWORK.ptr(irvt), n, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, n, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      }
    }
  } else {
    // A has more columns than rows. If A has sufficiently more
    // columns than rows, first reduce using the LQ decomposition (if
    // sufficient workspace available)
    if (n >= mnthr1) {
      if (wntqn) {
        // Path 1t (N >> M, JOBZ='N')
        // No singular vectors to be computed
        itau = 1;
        nwork = itau + m;
        // Compute A=L*Q
        // CWorkspace: need   M [tau] + M    [work]
        // CWorkspace: prefer M [tau] + M*NB [work]
        // RWorkspace: need   0
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Zero out above L
        Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
        ie = 1;
        itauq = 1;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in A
        // CWorkspace: need   2*M [tauq, taup] + M      [work]
        // CWorkspace: prefer 2*M [tauq, taup] + 2*M*NB [work]
        // RWorkspace: need   M [e]
        Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        nrwork = ie + m;
        // Perform bidiagonal SVD, compute singular values only
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + BDSPAC
        info = Bdsdc<R>('U', 'N', m, s, RWORK.ptr(ie), dum, 1, dum, 1, dum,
                        idum, RWORK.ptr(nrwork), iwork);
      } else if (wntqo) {
        // Path 2t (N >> M, JOBZ='O')
        // M right singular vectors to be overwritten on A and
        // M left singular vectors to be computed in U
        ivt = 1;
        ldwkvt = m;
        // WORK(IVT) is M by M
        il = ivt + ldwkvt * m;
        if (lwork >= m * n + m * m + 3 * m) {
          // WORK(IL) M by N
          ldwrkl = m;
          chunk = n;
        } else {
          // WORK(IL) is M by CHUNK
          ldwrkl = m;
          chunk = (lwork - m * m - 3 * m) / m;
        }
        itau = il + ldwrkl * chunk;
        nwork = itau + m;
        // Compute A=L*Q
        // CWorkspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
        // CWorkspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
        // RWorkspace: need   0
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy L to WORK(IL), zeroing about above it
        Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwrkl);
        Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(il + ldwrkl), ldwrkl);
        // Generate Q in A
        // CWorkspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
        // CWorkspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
        // RWorkspace: need   0
        Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = 1;
        itauq = itau;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in WORK(IL)
        // CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M [work]
        // CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + 2*M*NB
        // [work] RWorkspace: need   M [e]
        Gebrd(m, m, WORK.ptr(il), ldwrkl, s, RWORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
        iru = ie + m;
        irvt = iru + m * m;
        nrwork = irvt + m * m;
        info =
            Bdsdc<R>('U', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
        // Overwrite WORK(IU) by the left singular vectors of L
        // CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M [work]
        // CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB
        // [work] RWorkspace: need   0
        Lacp2('F', m, m, RWORK.ptr(iru), m, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itauq), u,
              ldu, WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
        // Overwrite WORK(IVT) by the right singular vectors of L
        // CWorkspace: need   M*M [VT] + M*M [L] + 2*M [tauq, taup] + M [work]
        // CWorkspace: prefer M*M [VT] + M*M [L] + 2*M [tauq, taup] + M*NB
        // [work] RWorkspace: need   0
        Lacp2('F', m, m, RWORK.ptr(irvt), m, WORK.ptr(ivt), ldwkvt);
        Unmbr('P', 'R', 'C', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itaup),
              WORK.ptr(ivt), ldwkvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply right singular vectors of L in WORK(IL) by Q
        // in A, storing result in WORK(IL) and copying to A
        // CWorkspace: need   M*M [VT] + M*M [L]
        // CWorkspace: prefer M*M [VT] + M*N [L]
        // RWorkspace: need   0
        for (i = 1; i <= n; i += chunk) {
          blk = Min<blasint>(n - i + 1, chunk);
          Gemm('N', 'N', m, blk, m, cone, WORK.ptr(ivt), m, A.ptr(1, i), lda,
               czero, WORK.ptr(il), ldwrkl);
          Lacpy('F', m, blk, WORK.ptr(il), ldwrkl, A.ptr(1, i), lda);
        }
      } else if (wntqs) {
        // Path 3t (N >> M, JOBZ='S')
        // M right singular vectors to be computed in VT and
        // M left singular vectors to be computed in U
        il = 1;
        // WORK(IL) is M by M
        ldwrkl = m;
        itau = il + ldwrkl * m;
        nwork = itau + m;
        // Compute A=L*Q
        // CWorkspace: need   M*M [L] + M [tau] + M    [work]
        // CWorkspace: prefer M*M [L] + M [tau] + M*NB [work]
        // RWorkspace: need   0
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        // Copy L to WORK(IL), zeroing out above it
        Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwrkl);
        Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(il + ldwrkl), ldwrkl);
        // Generate Q in A
        // CWorkspace: need   M*M [L] + M [tau] + M    [work]
        // CWorkspace: prefer M*M [L] + M [tau] + M*NB [work]
        // RWorkspace: need   0
        Unglq(m, n, m, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        ie = 1;
        itauq = itau;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in WORK(IL)
        // CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M      [work]
        // CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + 2*M*NB [work]
        // RWorkspace: need   M [e]
        Gebrd(m, m, WORK.ptr(il), ldwrkl, s, RWORK.ptr(ie), WORK.ptr(itauq),
              WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
        iru = ie + m;
        irvt = iru + m * m;
        nrwork = irvt + m * m;
        info =
            Bdsdc<R>('U', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of L
        // CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacp2('F', m, m, RWORK.ptr(iru), m, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itauq), u,
              ldu, WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by left singular vectors of L
        // CWorkspace: need   M*M [L] + 2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer M*M [L] + 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacp2('F', m, m, RWORK.ptr(irvt), m, vt, ldvt);
        Unmbr('P', 'R', 'C', m, m, m, WORK.ptr(il), ldwrkl, WORK.ptr(itaup), vt,
              ldvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Copy VT to WORK(IL), multiply right singular vectors of L
        // in WORK(IL) by Q in A, storing result in VT
        // CWorkspace: need   M*M [L]
        // RWorkspace: need   0
        Lacpy('F', m, m, vt, ldvt, WORK.ptr(il), ldwrkl);
        Gemm('N', 'N', m, n, m, cone, WORK.ptr(il), ldwrkl, a, lda, czero, vt,
             ldvt);
      } else if (wntqa) {
        // Path 4t (N >> M, JOBZ='A')
        // N right singular vectors to be computed in VT and
        // M left singular vectors to be computed in U
        ivt = 1;
        // WORK(IVT) is M by M
        ldwkvt = m;
        itau = ivt + ldwkvt * m;
        nwork = itau + m;
        // Compute A=L*Q, copying result to VT
        // CWorkspace: need   M*M [VT] + M [tau] + M    [work]
        // CWorkspace: prefer M*M [VT] + M [tau] + M*NB [work]
        // RWorkspace: need   0
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
        Lacpy('U', m, n, a, lda, vt, ldvt);
        // Generate Q in VT
        // CWorkspace: need   M*M [VT] + M [tau] + N    [work]
        // CWorkspace: prefer M*M [VT] + M [tau] + N*NB [work]
        // RWorkspace: need   0
        Unglq(n, n, m, vt, ldvt, WORK.ptr(itau), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Produce L in A, zeroing out above it
        Laset('U', m - 1, m - 1, czero, czero, A.ptr(1, 2), lda);
        ie = 1;
        itauq = itau;
        itaup = itauq + m;
        nwork = itaup + m;
        // Bidiagonalize L in A
        // CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M      [work]
        // CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + 2*M*NB [work]
        // RWorkspace: need   M [e]
        Gebrd(m, m, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
              WORK.ptr(nwork), lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RU] + M*M [RVT] + BDSPAC
        iru = ie + m;
        irvt = iru + m * m;
        nrwork = irvt + m * m;
        info =
            Bdsdc<R>('U', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of L
        // CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacp2('F', m, m, RWORK.ptr(iru), m, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, m, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
        // Overwrite WORK(IVT) by right singular vectors of L
        // CWorkspace: need   M*M [VT] + 2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer M*M [VT] + 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacp2('F', m, m, RWORK.ptr(irvt), m, WORK.ptr(ivt), ldwkvt);
        Unmbr('P', 'R', 'C', m, m, m, a, lda, WORK.ptr(itaup), WORK.ptr(ivt),
              ldwkvt, WORK.ptr(nwork), lwork - nwork + 1);
        // Multiply right singular vectors of L in WORK(IVT) by
        // Q in VT, storing result in A
        // CWorkspace: need   M*M [VT]
        // RWorkspace: need   0
        Gemm('N', 'N', m, n, m, cone, WORK.ptr(ivt), ldwkvt, vt, ldvt, czero, a,
             lda);
        // Copy right singular vectors of A from A to VT
        Lacpy('F', m, n, a, lda, vt, ldvt);
      }
    } else if (n >= mnthr2) {
      // MNTHR2 <= N < MNTHR1
      // Path 5t (N >> M, but not as much as MNTHR1)
      // Reduce to bidiagonal form without QR decomposition, use
      // ZUNGBR and matrix multiplication to compute singular vectors
      ie = 1;
      nrwork = ie + m;
      itauq = 1;
      itaup = itauq + m;
      nwork = itaup + m;
      // Bidiagonalize A
      // CWorkspace: need   2*M [tauq, taup] + N        [work]
      // CWorkspace: prefer 2*M [tauq, taup] + (M+N)*NB [work]
      // RWorkspace: need   M [e]
      Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(nwork), lwork - nwork + 1);
      if (wntqn) {
        // Path 5tn (N >> M, JOBZ='N')
        // Compute singular values only
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + BDSPAC
        info = Bdsdc<R>('L', 'N', m, s, RWORK.ptr(ie), dum, 1, dum, 1, dum,
                        idum, RWORK.ptr(nrwork), iwork);
      } else if (wntqo) {
        irvt = nrwork;
        iru = irvt + m * m;
        nrwork = iru + m * m;
        ivt = nwork;
        // Path 5to (N >> M, JOBZ='O')
        // Copy A to U, generate Q
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacpy('L', m, m, a, lda, u, ldu);
        Ungbr('Q', m, m, n, u, ldu, WORK.ptr(itauq), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Generate P**H in A
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(nwork),
              lwork - nwork + 1);
        ldwkvt = m;
        if (lwork >= m * n + 3 * m) {
          // WORK( IVT ) is M by N
          nwork = ivt + ldwkvt * n;
          chunk = n;
        } else {
          // WORK( IVT ) is M by CHUNK
          chunk = (lwork - 3 * m) / m;
          nwork = ivt + ldwkvt * chunk;
        }
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
        info =
            Bdsdc<R>('L', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Multiply Q in U by real matrix RWORK(IRVT)
        // storing the result in WORK(IVT), copying to U
        // CWorkspace: need   2*M [tauq, taup] + M*M [VT]
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
        Lacrm(m, m, u, ldu, RWORK.ptr(iru), m, WORK.ptr(ivt), ldwkvt,
              RWORK.ptr(nrwork));
        Lacpy('F', m, m, WORK.ptr(ivt), ldwkvt, u, ldu);
        // Multiply RWORK(IRVT) by P**H in A, storing the
        // result in WORK(IVT), copying to A
        // CWorkspace: need   2*M [tauq, taup] + M*M [VT]
        // CWorkspace: prefer 2*M [tauq, taup] + M*N [VT]
        // RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [rwork]
        // RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M
        // since N < 2*M here
        nrwork = iru;
        for (i = 1; i <= n; i += chunk) {
          blk = Min<blasint>(n - i + 1, chunk);
          Larcm(m, blk, RWORK.ptr(irvt), m, A.ptr(1, i), lda, WORK.ptr(ivt),
                ldwkvt, RWORK.ptr(nrwork));
          Lacpy('F', m, blk, WORK.ptr(ivt), ldwkvt, A.ptr(1, i), lda);
        }
      } else if (wntqs) {
        // Path 5ts (N >> M, JOBZ='S')
        // Copy A to U, generate Q
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacpy('L', m, m, a, lda, u, ldu);
        Ungbr('Q', m, m, n, u, ldu, WORK.ptr(itauq), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Copy A to VT, generate P**H
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacpy('U', m, n, a, lda, vt, ldvt);
        Ungbr('P', m, n, m, vt, ldvt, WORK.ptr(itaup), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
        irvt = nrwork;
        iru = irvt + m * m;
        nrwork = iru + m * m;
        info =
            Bdsdc<R>('L', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Multiply Q in U by real matrix RWORK(IRU), storing the
        // result in A, copying to U
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
        Lacrm(m, m, u, ldu, RWORK.ptr(iru), m, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', m, m, a, lda, u, ldu);
        // Multiply real matrix RWORK(IRVT) by P**H in VT,
        // storing the result in A, copying to VT
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M
        // since N < 2*M here
        nrwork = iru;
        Larcm(m, n, RWORK.ptr(irvt), m, vt, ldvt, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', m, n, a, lda, vt, ldvt);
      } else {
        // Path 5ta (N >> M, JOBZ='A')
        // Copy A to U, generate Q
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   0
        Lacpy('L', m, m, a, lda, u, ldu);
        Ungbr('Q', m, m, n, u, ldu, WORK.ptr(itauq), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Copy A to VT, generate P**H
        // CWorkspace: need   2*M [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + N*NB [work]
        // RWorkspace: need   0
        Lacpy('U', m, n, a, lda, vt, ldvt);
        Ungbr('P', n, n, m, vt, ldvt, WORK.ptr(itaup), WORK.ptr(nwork),
              lwork - nwork + 1);
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
        irvt = nrwork;
        iru = irvt + m * m;
        nrwork = iru + m * m;
        info =
            Bdsdc<R>('L', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Multiply Q in U by real matrix RWORK(IRU), storing the
        // result in A, copying to U
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + 2*M*M [rwork]
        Lacrm(m, m, u, ldu, RWORK.ptr(iru), m, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', m, m, a, lda, u, ldu);
        // Multiply real matrix RWORK(IRVT) by P**H in VT,
        // storing the result in A, copying to VT
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M
        // since N < 2*M here
        nrwork = iru;
        Larcm(m, n, RWORK.ptr(irvt), m, vt, ldvt, a, lda, RWORK.ptr(nrwork));
        Lacpy('F', m, n, a, lda, vt, ldvt);
      }
    } else {
      // N .LT. MNTHR2
      // Path 6t (N > M, but not much larger)
      // Reduce to bidiagonal form without LQ decomposition
      // Use ZUNMBR to compute singular vectors
      ie = 1;
      nrwork = ie + m;
      itauq = 1;
      itaup = itauq + m;
      nwork = itaup + m;
      // Bidiagonalize A
      // CWorkspace: need   2*M [tauq, taup] + N        [work]
      // CWorkspace: prefer 2*M [tauq, taup] + (M+N)*NB [work]
      // RWorkspace: need   M [e]
      Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq), WORK.ptr(itaup),
            WORK.ptr(nwork), lwork - nwork + 1);
      if (wntqn) {
        // Path 6tn (N > M, JOBZ='N')
        // Compute singular values only
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + BDSPAC
        info = Bdsdc<R>('L', 'N', m, s, RWORK.ptr(ie), dum, 1, dum, 1, dum,
                        idum, RWORK.ptr(nrwork), iwork);
      } else if (wntqo) {
        // Path 6to (N > M, JOBZ='O')
        ldwkvt = m;
        ivt = nwork;
        if (lwork >= m * n + 3 * m) {
          // WORK( IVT ) is M by N
          Laset('F', m, n, czero, czero, WORK.ptr(ivt), ldwkvt);
          nwork = ivt + ldwkvt * n;
        } else {
          // WORK( IVT ) is M by CHUNK
          chunk = (lwork - 3 * m) / m;
          nwork = ivt + ldwkvt * chunk;
        }
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
        irvt = nrwork;
        iru = irvt + m * m;
        nrwork = iru + m * m;
        info =
            Bdsdc<R>('L', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of A
        // CWorkspace: need   2*M [tauq, taup] + M*M [VT] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*M [VT] + M*NB [work]
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
        Lacp2('F', m, m, RWORK.ptr(iru), m, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        if (lwork >= m * n + 3 * m) {
          // Path 6to-fast
          // Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
          // Overwrite WORK(IVT) by right singular vectors of A,
          // copying to A
          // CWorkspace: need   2*M [tauq, taup] + M*N [VT] + M    [work]
          // CWorkspace: prefer 2*M [tauq, taup] + M*N [VT] + M*NB [work]
          // RWorkspace: need   M [e] + M*M [RVT]
          Lacp2('F', m, m, RWORK.ptr(irvt), m, WORK.ptr(ivt), ldwkvt);
          Unmbr('P', 'R', 'C', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(ivt),
                ldwkvt, WORK.ptr(nwork), lwork - nwork + 1);
          Lacpy('F', m, n, WORK.ptr(ivt), ldwkvt, a, lda);
        } else {
          // Path 6to-slow
          // Generate P**H in A
          // CWorkspace: need   2*M [tauq, taup] + M*M [VT] + M    [work]
          // CWorkspace: prefer 2*M [tauq, taup] + M*M [VT] + M*NB [work]
          // RWorkspace: need   0
          Ungbr('P', m, n, m, a, lda, WORK.ptr(itaup), WORK.ptr(nwork),
                lwork - nwork + 1);
          // Multiply Q in A by real matrix RWORK(IRU), storing the
          // result in WORK(IU), copying to A
          // CWorkspace: need   2*M [tauq, taup] + M*M [VT]
          // CWorkspace: prefer 2*M [tauq, taup] + M*N [VT]
          // RWorkspace: need   M [e] + M*M [RVT] + 2*M*M [rwork]
          // RWorkspace: prefer M [e] + M*M [RVT] + 2*M*N [rwork] < M + 5*M*M
          // since N < 2*M here
          nrwork = iru;
          for (i = 1; i <= n; i += chunk) {
            blk = Min<blasint>(n - i + 1, chunk);
            Larcm(m, blk, RWORK.ptr(irvt), m, A.ptr(1, i), lda, WORK.ptr(ivt),
                  ldwkvt, RWORK.ptr(nrwork));
            Lacpy('F', m, blk, WORK.ptr(ivt), ldwkvt, A.ptr(1, i), lda);
          }
        }
      } else if (wntqs) {
        // Path 6ts (N > M, JOBZ='S')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
        irvt = nrwork;
        iru = irvt + m * m;
        nrwork = iru + m * m;
        info =
            Bdsdc<R>('L', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of A
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
        Lacp2('F', m, m, RWORK.ptr(iru), m, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of A
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   M [e] + M*M [RVT]
        Laset('F', m, n, czero, czero, vt, ldvt);
        Lacp2('F', m, m, RWORK.ptr(irvt), m, vt, ldvt);
        Unmbr('P', 'R', 'C', m, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
      } else {
        // Path 6ta (N > M, JOBZ='A')
        // Perform bidiagonal SVD, computing left singular vectors
        // of bidiagonal matrix in RWORK(IRU) and computing right
        // singular vectors of bidiagonal matrix in RWORK(IRVT)
        // CWorkspace: need   0
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU] + BDSPAC
        irvt = nrwork;
        iru = irvt + m * m;
        nrwork = iru + m * m;
        info =
            Bdsdc<R>('L', 'I', m, s, RWORK.ptr(ie), RWORK.ptr(iru), m,
                     RWORK.ptr(irvt), m, dum, idum, RWORK.ptr(nrwork), iwork);
        // Copy real matrix RWORK(IRU) to complex matrix U
        // Overwrite U by left singular vectors of A
        // CWorkspace: need   2*M [tauq, taup] + M    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + M*NB [work]
        // RWorkspace: need   M [e] + M*M [RVT] + M*M [RU]
        Lacp2('F', m, m, RWORK.ptr(iru), m, u, ldu);
        Unmbr('Q', 'L', 'N', m, m, n, a, lda, WORK.ptr(itauq), u, ldu,
              WORK.ptr(nwork), lwork - nwork + 1);
        // Set all of VT to identity matrix
        Laset('F', n, n, czero, cone, vt, ldvt);
        // Copy real matrix RWORK(IRVT) to complex matrix VT
        // Overwrite VT by right singular vectors of A
        // CWorkspace: need   2*M [tauq, taup] + N    [work]
        // CWorkspace: prefer 2*M [tauq, taup] + N*NB [work]
        // RWorkspace: need   M [e] + M*M [RVT]
        Lacp2('F', m, m, RWORK.ptr(irvt), m, vt, ldvt);
        Unmbr('P', 'R', 'C', n, n, m, a, lda, WORK.ptr(itaup), vt, ldvt,
              WORK.ptr(nwork), lwork - nwork + 1);
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

// xGESDD: computes the singular value decomposition A = U * SIGMA * V**H of
// the M-by-N matrix A by divide and conquer (xBDSDC), optionally with the
// left and right singular vectors. SIGMA is zero except for its min(M,N)
// diagonal elements, the singular values, which are real, non-negative and
// returned in S in decreasing order; the first min(M,N) columns of U and
// rows of V**H are the left and right singular vectors. The routine returns
// V**H, not V.
//
// JOBZ: 'A' all M columns of U and all N rows of V**H are returned in U and
// VT; 'S' the first min(M,N) columns and rows; 'O' they overwrite A for
// M >= N (the rows of V**H for M < N), the others are returned in U (VT);
// 'N' no vectors. A is destroyed unless JOBZ = 'O'.
//
// With mn = min(M,N) and mx = max(M,N), LWORK (LWORK = -1 is a workspace
// query; larger values give better performance) must be at least, for
// real T: 3*mn + max(mx, 7*mn) (JOBZ = 'N'), 3*mn + max(mx, 5*mn*mn+4*mn)
// ('O'), 4*mn*mn + 7*mn ('S') or 4*mn*mn + 6*mn + mx ('A'); for complex T:
// 2*mn + mx ('N'), 2*mn*mn + 2*mn + mx ('O'), mn*mn + 3*mn ('S') or
// mn*mn + 2*mn + mx ('A'). RWORK is used for complex T only and has at
// least 5*mn (JOBZ = 'N'), 5*mn*mn + 5*mn (mx >> mn) or
// max(5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn) elements. IWORK has 8*mn
// elements.
//
// Returns INFO: 0, -i for an illegal argument i (-4 if A has a NaN), or
// > 0 if xBDSDC did not converge.
template <typename T>
blasint Gesdd(char jobz, blasint m, blasint n, T* a, blasint lda, RealT<T>* s,
              T* u, blasint ldu, T* vt, blasint ldvt, T* work, blasint lwork,
              RealT<T>* rwork, blasint* iwork) {
  if constexpr (IsComplex<T>::value) {
    return GesddComplex(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                        rwork, iwork);
  } else {
    (void)rwork;
    return GesddReal(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                     iwork);
  }
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Gesdd<T>(char, blasint, blasint, T*, blasint, RealT<T>*,   \
                            T*, blasint, T*, blasint, T*, blasint, RealT<T>*, \
                            blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

HWYBLAS_API void sgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* s, float* u, const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* info) {
  *info =
      hwyblas::lapack::Gesdd(*jobz, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt,
                             work, *lwork, static_cast<float*>(nullptr), iwork);
}
HWYBLAS_API void dgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Gesdd(*jobz, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt,
                                 work, *lwork, static_cast<double*>(nullptr),
                                 iwork);
}
HWYBLAS_API void cgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* s, float* u, const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* iwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gesdd(*jobz, *m, *n, As<c32>(a), *lda, s, As<c32>(u),
                                 *ldu, As<c32>(vt), *ldvt, As<c32>(work),
                                 *lwork, rwork, iwork);
}
HWYBLAS_API void zgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* iwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gesdd(*jobz, *m, *n, As<c64>(a), *lda, s, As<c64>(u),
                                 *ldu, As<c64>(vt), *ldvt, As<c64>(work),
                                 *lwork, rwork, iwork);
}

}  // extern "C"
