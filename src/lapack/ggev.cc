// The generalized nonsymmetric eigenproblem of a pair (A, B): the
// generalized Schur factorization xGGES (A = Q*S*Z**H, B = Q*T*Z**H, with
// optional ordering of the eigenvalues) and the generalized eigenvalues and
// eigenvectors xGGEV. Both balance (permute) the pair with xGGBAL, reduce
// B to triangular form by a QR factorization and the pair to generalized
// Hessenberg form with xGGHRD, and run the QZ iteration xHGEQZ.
//
// The real and the complex routines differ in the eigenvalues (ALPHAR and
// ALPHAI for ALPHA), the normalization of the complex eigenvectors and
// the workspaces, so they are two overloads each.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xGGES (real T): computes the generalized eigenvalues (ALPHAR(j) +
// i*ALPHAI(j)) / BETA(j) of the N-by-N pair (A, B), its generalized real
// Schur form (S, T) in A and B (S quasi-upper triangular with 2-by-2
// blocks for complex conjugate pairs, T upper triangular with diagonal
// 2-by-2 blocks; as in the reference, their elements can be negative after
// sorting, although LAPACK documents positive ones), and for JOBVSL /
// JOBVSR = 'V' the
// left / right Schur vectors in VSL / VSR (A = VSL*S*VSR**T, B =
// VSL*T*VSR**T). SORT = 'S': the eigenvalues for which SELCTG(ALPHAR(j),
// ALPHAI(j), BETA(j)) is true are moved to the top left (SDIM returns
// their number, a complex pair counting 2); SORT = 'N': no ordering. WORK
// has LWORK >= max(8*N, 6*N+16) elements (N > 0; 1 for N = 0; LWORK = -1:
// workspace query); BWORK has N elements (SORT = 'S').
//
// Returns INFO: 0, -i for an illegal argument i, 1..N if the QZ iteration
// failed (ALPHAR(j), ALPHAI(j), BETA(j) are correct for j = INFO+1..N),
// N+1 for other failures in xHGEQZ, N+2 if, after reordering, roundoff
// changed the eigenvalues so that the leading ones no longer satisfy
// SELCTG, or N+3 if the reordering failed in xTGSEN.
template <typename R>
blasint Gges(char jobvsl, char jobvsr, char sort, GgesSelectReal<R> selctg,
             blasint n, R* a, blasint lda, R* b, blasint ldb, blasint* sdim,
             R* alphar, R* alphai, R* beta, R* vsl, blasint ldvsl, R* vsr,
             blasint ldvsr, R* work, blasint lwork, blasint* bwork) {
  using T = R;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Vector<R> ALPHAI(alphai);
  const Vector<R> ALPHAR(alphar);
  const Matrix<R> B(b, ldb);
  const Vector<R> BETA(beta);
  const Vector<blasint> BWORK(bwork);
  R dif[2];
  blasint idum[1];
  const Matrix<R> VSL(vsl, ldvsl);
  const Matrix<R> VSR(vsr, ldvsr);
  const Vector<R> WORK(work);
  blasint i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, ip, iright, irows,
      itau, iwrk, maxwrk, minwrk;
  bool cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, lst2sl, wantst;
  R anrm, anrmto, bignum, bnrm, bnrmto, eps, pvsl, pvsr, safmax, safmin, smlnum;
  if (Lsame(jobvsl, 'N')) {
    ijobvl = 1;
    ilvsl = false;
  } else if (Lsame(jobvsl, 'V')) {
    ijobvl = 2;
    ilvsl = true;
  } else {
    ijobvl = -1;
    ilvsl = false;
  }
  if (Lsame(jobvsr, 'N')) {
    ijobvr = 1;
    ilvsr = false;
  } else if (Lsame(jobvsr, 'V')) {
    ijobvr = 2;
    ilvsr = true;
  } else {
    ijobvr = -1;
    ilvsr = false;
  }
  wantst = Lsame(sort, 'S');
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  if (ijobvl <= 0) {
    info = -1;
  } else if (ijobvr <= 0) {
    info = -2;
  } else if ((!wantst) && (!Lsame(sort, 'N'))) {
    info = -3;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldvsl < 1 || (ilvsl && ldvsl < n)) {
    info = -15;
  } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
    info = -17;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV.)
  if (info == 0) {
    if (n > 0) {
      minwrk = Max<blasint>(8 * n, 6 * n + 16);
      maxwrk = minwrk - n + n * Ilaenv<T>(1, "GEQRF", " ", n, 1, n, 0);
      maxwrk = Max<blasint>(
          maxwrk, minwrk - n + n * Ilaenv<T>(1, "ORMQR", " ", n, 1, n, -1));
      if (ilvsl) {
        maxwrk = Max<blasint>(
            maxwrk, minwrk - n + n * Ilaenv<T>(1, "ORGQR", " ", n, 1, n, -1));
      }
    } else {
      minwrk = 1;
      maxwrk = 1;
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) info = -19;
  }
  if (info != 0) {
    Xerbla<T>("GGES", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n == 0) {
    *sdim = 0;
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  smlnum = std::sqrt(safmin) / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', n, n, a, lda, work);
  ilascl = false;
  if (anrm > zero && anrm < smlnum) {
    anrmto = smlnum;
    ilascl = true;
  } else if (anrm > bignum) {
    anrmto = bignum;
    ilascl = true;
  }
  if (ilascl) ierr = Lascl('G', 0, 0, anrm, anrmto, n, n, a, lda);
  // Scale B if max element outside range [SMLNUM,BIGNUM]
  bnrm = Lange('M', n, n, b, ldb, work);
  ilbscl = false;
  if (bnrm > zero && bnrm < smlnum) {
    bnrmto = smlnum;
    ilbscl = true;
  } else if (bnrm > bignum) {
    bnrmto = bignum;
    ilbscl = true;
  }
  if (ilbscl) ierr = Lascl('G', 0, 0, bnrm, bnrmto, n, n, b, ldb);
  // Permute the matrix to make it more nearly triangular
  // (Workspace: need 6*N + 2*N space for storing balancing factors)
  ileft = 1;
  iright = n + 1;
  iwrk = iright + n;
  ierr = Ggbal('P', n, a, lda, b, ldb, &ilo, &ihi, WORK.ptr(ileft),
               WORK.ptr(iright), WORK.ptr(iwrk));
  // Reduce B to triangular form (QR decomposition of B)
  // (Workspace: need N, prefer N*NB)
  irows = ihi + 1 - ilo;
  icols = n + 1 - ilo;
  itau = iwrk;
  iwrk = itau + irows;
  ierr = Geqrf(irows, icols, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
               WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Apply the orthogonal transformation to matrix A
  // (Workspace: need N, prefer N*NB)
  ierr =
      Unmqr('L', 'T', irows, icols, irows, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
            A.ptr(ilo, ilo), lda, WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Initialize VSL
  // (Workspace: need N, prefer N*NB)
  if (ilvsl) {
    Laset('F', n, n, zero, one, vsl, ldvsl);
    if (irows > 1) {
      Lacpy('L', irows - 1, irows - 1, B.ptr(ilo + 1, ilo), ldb,
            VSL.ptr(ilo + 1, ilo), ldvsl);
    }
    ierr = Ungqr(irows, irows, irows, VSL.ptr(ilo, ilo), ldvsl, WORK.ptr(itau),
                 WORK.ptr(iwrk), lwork + 1 - iwrk);
  }
  // Initialize VSR
  if (ilvsr) Laset('F', n, n, zero, one, vsr, ldvsr);
  // Reduce to generalized Hessenberg form
  // (Workspace: none needed)
  ierr = Gghrd(jobvsl, jobvsr, n, ilo, ihi, a, lda, b, ldb, vsl, ldvsl, vsr,
               ldvsr);
  // Perform QZ algorithm, computing Schur vectors if desired
  // (Workspace: need N)
  iwrk = itau;
  ierr = Hgeqz('S', jobvsl, jobvsr, n, ilo, ihi, a, lda, b, ldb, alphar, alphai,
               beta, vsl, ldvsl, vsr, ldvsr, WORK.ptr(iwrk), lwork + 1 - iwrk);
  if (ierr != 0) {
    if (ierr > 0 && ierr <= n) {
      info = ierr;
    } else if (ierr > n && ierr <= 2 * n) {
      info = ierr - n;
    } else {
      info = n + 1;
    }
    goto L50;
  }
  // Sort eigenvalues ALPHA/BETA if desired
  // (Workspace: need 4*N+16 )
  *sdim = 0;
  if (wantst) {
    // Undo scaling on eigenvalues before SELCTGing
    if (ilascl) {
      ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alphar, n);
      ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alphai, n);
    }
    if (ilbscl) ierr = Lascl('G', 0, 0, bnrmto, bnrm, n, 1, beta, n);
    // Select eigenvalues
    for (i = 1; i <= n; ++i) {
      BWORK(i) = selctg(ALPHAR.ptr(i), ALPHAI.ptr(i), BETA.ptr(i));
    }
    ierr = Tgsen(0, ilvsl, ilvsr, bwork, n, a, lda, b, ldb, alphar, alphai,
                 beta, vsl, ldvsl, vsr, ldvsr, sdim, &pvsl, &pvsr, dif,
                 WORK.ptr(iwrk), lwork - iwrk + 1, idum, 1);
    if (ierr == 1) info = n + 3;
  }
  // Apply back-permutation to VSL and VSR
  // (Workspace: none needed)
  if (ilvsl)
    ierr = Ggbak('P', 'L', n, ilo, ihi, WORK.ptr(ileft), WORK.ptr(iright), n,
                 vsl, ldvsl);
  if (ilvsr)
    ierr = Ggbak('P', 'R', n, ilo, ihi, WORK.ptr(ileft), WORK.ptr(iright), n,
                 vsr, ldvsr);
  // Check if unscaling would cause over/underflow, if so, rescale
  // (ALPHAR(I),ALPHAI(I),BETA(I)) so BETA(I) is on the order of
  // B(I,I) and ALPHAR(I) and ALPHAI(I) are on the order of A(I,I)
  if (ilascl) {
    for (i = 1; i <= n; ++i) {
      if (ALPHAI(i) != zero) {
        if ((ALPHAR(i) / safmax) > (anrmto / anrm) ||
            (safmin / ALPHAR(i)) > (anrm / anrmto)) {
          WORK(1) = std::abs(A(i, i) / ALPHAR(i));
          BETA(i) = BETA(i) * WORK(1);
          ALPHAR(i) = ALPHAR(i) * WORK(1);
          ALPHAI(i) = ALPHAI(i) * WORK(1);
        } else if ((ALPHAI(i) / safmax) > (anrmto / anrm) ||
                   (safmin / ALPHAI(i)) > (anrm / anrmto)) {
          WORK(1) = std::abs(A(i, i + 1) / ALPHAI(i));
          BETA(i) = BETA(i) * WORK(1);
          ALPHAR(i) = ALPHAR(i) * WORK(1);
          ALPHAI(i) = ALPHAI(i) * WORK(1);
        }
      }
    }
  }
  if (ilbscl) {
    for (i = 1; i <= n; ++i) {
      if (ALPHAI(i) != zero) {
        if ((BETA(i) / safmax) > (bnrmto / bnrm) ||
            (safmin / BETA(i)) > (bnrm / bnrmto)) {
          WORK(1) = std::abs(B(i, i) / BETA(i));
          BETA(i) = BETA(i) * WORK(1);
          ALPHAR(i) = ALPHAR(i) * WORK(1);
          ALPHAI(i) = ALPHAI(i) * WORK(1);
        }
      }
    }
  }
  // Undo scaling
  if (ilascl) {
    ierr = Lascl('H', 0, 0, anrmto, anrm, n, n, a, lda);
    ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alphar, n);
    ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alphai, n);
  }
  if (ilbscl) {
    ierr = Lascl('U', 0, 0, bnrmto, bnrm, n, n, b, ldb);
    ierr = Lascl('G', 0, 0, bnrmto, bnrm, n, 1, beta, n);
  }
  if (wantst) {
    // Check if reordering is correct
    lastsl = true;
    lst2sl = true;
    *sdim = 0;
    ip = 0;
    for (i = 1; i <= n; ++i) {
      cursl = selctg(ALPHAR.ptr(i), ALPHAI.ptr(i), BETA.ptr(i)) != 0;
      if (ALPHAI(i) == zero) {
        if (cursl) *sdim = *sdim + 1;
        ip = 0;
        if (cursl && !lastsl) info = n + 2;
      } else {
        if (ip == 1) {
          // Last eigenvalue of conjugate pair
          cursl = cursl || lastsl;
          lastsl = cursl;
          if (cursl) *sdim = *sdim + 2;
          ip = -1;
          if (cursl && !lst2sl) info = n + 2;
        } else {
          // First eigenvalue of conjugate pair
          ip = 1;
        }
      }
      lst2sl = lastsl;
      lastsl = cursl;
    }
  }
L50:
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// xGGES (complex T): as the real routine for the upper triangular
// generalized Schur form (T with a real nonnegative diagonal) and the
// eigenvalues ALPHA(j) / BETA(j), selected by SELCTG(ALPHA(j), BETA(j)).
// WORK has LWORK >= max(1, 2*N) elements, RWORK 8*N.
template <typename T>
blasint Gges(char jobvsl, char jobvsr, char sort,
             GgesSelectComplex<RealT<T>> selctg, blasint n, T* a, blasint lda,
             T* b, blasint ldb, blasint* sdim, T* alpha, T* beta, T* vsl,
             blasint ldvsl, T* vsr, blasint ldvsr, T* work, blasint lwork,
             RealT<T>* rwork, blasint* bwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<T> ALPHA(alpha);
  const Matrix<T> B(b, ldb);
  const Vector<T> BETA(beta);
  const Vector<blasint> BWORK(bwork);
  R dif[2];
  blasint idum[1];
  const Vector<R> RWORK(rwork);
  const Matrix<T> VSL(vsl, ldvsl);
  const Matrix<T> VSR(vsr, ldvsr);
  const Vector<T> WORK(work);
  blasint i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, iright, irows, irwrk,
      itau, iwrk, lwkmin, lwkopt;
  bool cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, wantst;
  R anrm, anrmto, bignum, bnrm, bnrmto, eps, pvsl, pvsr, smlnum;
  if (Lsame(jobvsl, 'N')) {
    ijobvl = 1;
    ilvsl = false;
  } else if (Lsame(jobvsl, 'V')) {
    ijobvl = 2;
    ilvsl = true;
  } else {
    ijobvl = -1;
    ilvsl = false;
  }
  if (Lsame(jobvsr, 'N')) {
    ijobvr = 1;
    ilvsr = false;
  } else if (Lsame(jobvsr, 'V')) {
    ijobvr = 2;
    ilvsr = true;
  } else {
    ijobvr = -1;
    ilvsr = false;
  }
  wantst = Lsame(sort, 'S');
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  if (ijobvl <= 0) {
    info = -1;
  } else if (ijobvr <= 0) {
    info = -2;
  } else if ((!wantst) && (!Lsame(sort, 'N'))) {
    info = -3;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldvsl < 1 || (ilvsl && ldvsl < n)) {
    info = -14;
  } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
    info = -16;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV.)
  if (info == 0) {
    lwkmin = Max<blasint>(1, 2 * n);
    lwkopt = Max<blasint>(1, n + n * Ilaenv<T>(1, "GEQRF", " ", n, 1, n, 0));
    lwkopt =
        Max<blasint>(lwkopt, n + n * Ilaenv<T>(1, "UNMQR", " ", n, 1, n, -1));
    if (ilvsl) {
      lwkopt =
          Max<blasint>(lwkopt, n + n * Ilaenv<T>(1, "UNGQR", " ", n, 1, n, -1));
    }
    WORK(1) = WorkSize<T>(lwkopt);
    if (lwork < lwkmin && !lquery) info = -18;
  }
  if (info != 0) {
    Xerbla<T>("GGES", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n == 0) {
    *sdim = 0;
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S');
  bignum = one / smlnum;
  smlnum = std::sqrt(smlnum) / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', n, n, a, lda, rwork);
  ilascl = false;
  if (anrm > zero && anrm < smlnum) {
    anrmto = smlnum;
    ilascl = true;
  } else if (anrm > bignum) {
    anrmto = bignum;
    ilascl = true;
  }
  if (ilascl) ierr = Lascl('G', 0, 0, anrm, anrmto, n, n, a, lda);
  // Scale B if max element outside range [SMLNUM,BIGNUM]
  bnrm = Lange('M', n, n, b, ldb, rwork);
  ilbscl = false;
  if (bnrm > zero && bnrm < smlnum) {
    bnrmto = smlnum;
    ilbscl = true;
  } else if (bnrm > bignum) {
    bnrmto = bignum;
    ilbscl = true;
  }
  if (ilbscl) ierr = Lascl('G', 0, 0, bnrm, bnrmto, n, n, b, ldb);
  // Permute the matrix to make it more nearly triangular
  // (Real Workspace: need 6*N)
  ileft = 1;
  iright = n + 1;
  irwrk = iright + n;
  ierr = Ggbal('P', n, a, lda, b, ldb, &ilo, &ihi, RWORK.ptr(ileft),
               RWORK.ptr(iright), RWORK.ptr(irwrk));
  // Reduce B to triangular form (QR decomposition of B)
  // (Complex Workspace: need N, prefer N*NB)
  irows = ihi + 1 - ilo;
  icols = n + 1 - ilo;
  itau = 1;
  iwrk = itau + irows;
  ierr = Geqrf(irows, icols, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
               WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Apply the orthogonal transformation to matrix A
  // (Complex Workspace: need N, prefer N*NB)
  ierr =
      Unmqr('L', 'C', irows, icols, irows, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
            A.ptr(ilo, ilo), lda, WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Initialize VSL
  // (Complex Workspace: need N, prefer N*NB)
  if (ilvsl) {
    Laset('F', n, n, czero, cone, vsl, ldvsl);
    if (irows > 1) {
      Lacpy('L', irows - 1, irows - 1, B.ptr(ilo + 1, ilo), ldb,
            VSL.ptr(ilo + 1, ilo), ldvsl);
    }
    ierr = Ungqr(irows, irows, irows, VSL.ptr(ilo, ilo), ldvsl, WORK.ptr(itau),
                 WORK.ptr(iwrk), lwork + 1 - iwrk);
  }
  // Initialize VSR
  if (ilvsr) Laset('F', n, n, czero, cone, vsr, ldvsr);
  // Reduce to generalized Hessenberg form
  // (Workspace: none needed)
  ierr = Gghrd(jobvsl, jobvsr, n, ilo, ihi, a, lda, b, ldb, vsl, ldvsl, vsr,
               ldvsr);
  *sdim = 0;
  // Perform QZ algorithm, computing Schur vectors if desired
  // (Complex Workspace: need N)
  // (Real Workspace: need N)
  iwrk = itau;
  ierr = Hgeqz('S', jobvsl, jobvsr, n, ilo, ihi, a, lda, b, ldb, alpha, beta,
               vsl, ldvsl, vsr, ldvsr, WORK.ptr(iwrk), lwork + 1 - iwrk,
               RWORK.ptr(irwrk));
  if (ierr != 0) {
    if (ierr > 0 && ierr <= n) {
      info = ierr;
    } else if (ierr > n && ierr <= 2 * n) {
      info = ierr - n;
    } else {
      info = n + 1;
    }
    goto L30;
  }
  // Sort eigenvalues ALPHA/BETA if desired
  // (Workspace: none needed)
  if (wantst) {
    // Undo scaling on eigenvalues before selecting
    if (ilascl) ierr = Lascl('G', 0, 0, anrm, anrmto, n, 1, alpha, n);
    if (ilbscl) ierr = Lascl('G', 0, 0, bnrm, bnrmto, n, 1, beta, n);
    // Select eigenvalues
    for (i = 1; i <= n; ++i) {
      BWORK(i) = selctg(reinterpret_cast<const R*>(ALPHA.ptr(i)),
                        reinterpret_cast<const R*>(BETA.ptr(i)));
    }
    ierr = Tgsen(0, ilvsl, ilvsr, bwork, n, a, lda, b, ldb, alpha, beta, vsl,
                 ldvsl, vsr, ldvsr, sdim, &pvsl, &pvsr, dif, WORK.ptr(iwrk),
                 lwork - iwrk + 1, idum, 1);
    if (ierr == 1) info = n + 3;
  }
  // Apply back-permutation to VSL and VSR
  // (Workspace: none needed)
  if (ilvsl)
    ierr = Ggbak('P', 'L', n, ilo, ihi, RWORK.ptr(ileft), RWORK.ptr(iright), n,
                 vsl, ldvsl);
  if (ilvsr)
    ierr = Ggbak('P', 'R', n, ilo, ihi, RWORK.ptr(ileft), RWORK.ptr(iright), n,
                 vsr, ldvsr);
  // Undo scaling
  if (ilascl) {
    ierr = Lascl('U', 0, 0, anrmto, anrm, n, n, a, lda);
    ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alpha, n);
  }
  if (ilbscl) {
    ierr = Lascl('U', 0, 0, bnrmto, bnrm, n, n, b, ldb);
    ierr = Lascl('G', 0, 0, bnrmto, bnrm, n, 1, beta, n);
  }
  if (wantst) {
    // Check if reordering is correct
    lastsl = true;
    *sdim = 0;
    for (i = 1; i <= n; ++i) {
      cursl = selctg(reinterpret_cast<const R*>(ALPHA.ptr(i)),
                     reinterpret_cast<const R*>(BETA.ptr(i))) != 0;
      if (cursl) *sdim = *sdim + 1;
      if (cursl && !lastsl) info = n + 2;
      lastsl = cursl;
    }
  }
L30:
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xGGEV (real T): computes the generalized eigenvalues (ALPHAR(j) +
// i*ALPHAI(j)) / BETA(j) of the N-by-N pair (A, B) (destroyed) and, for
// JOBVL / JOBVR = 'V', the left / right generalized eigenvectors: u(j)**H
// * A = lambda(j) * u(j)**H * B in VL, A * v(j) = lambda(j) * B * v(j) in
// VR. Complex conjugate eigenvalue pairs are consecutive, the one with
// positive imaginary part first; the eigenvector of the pair is then
// VR(:,j) + i*VR(:,j+1) (and its conjugate). Each eigenvector is scaled
// so that its largest component has |re| + |im| = 1. WORK has LWORK >=
// max(1, 8*N) elements (LWORK = -1: workspace query).
//
// Returns INFO: 0, -i for an illegal argument i, 1..N if the QZ iteration
// failed (no eigenvectors; ALPHAR(j), ALPHAI(j), BETA(j) are correct for j
// = INFO+1..N), N+1 for other failures in xHGEQZ, or N+2 if xTGEVC failed.
template <typename R>
blasint Ggev(char jobvl, char jobvr, blasint n, R* a, blasint lda, R* b,
             blasint ldb, R* alphar, R* alphai, R* beta, R* vl, blasint ldvl,
             R* vr, blasint ldvr, R* work, blasint lwork) {
  using T = R;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Vector<R> ALPHAI(alphai);
  const Vector<R> ALPHAR(alphar);
  const Matrix<R> B(b, ldb);
  const Vector<R> BETA(beta);
  blasint ldumma[1] = {0};
  const Matrix<R> VL(vl, ldvl);
  const Matrix<R> VR(vr, ldvr);
  const Vector<R> WORK(work);
  blasint icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, in, iright, irows, itau,
      iwrk, jc, jr, maxwrk, minwrk;
  bool ilascl, ilbscl, ilv, ilvl, ilvr, lquery;
  R anrm, anrmto, bignum, bnrm, bnrmto, eps, smlnum, temp;
  char chtemp;
  if (Lsame(jobvl, 'N')) {
    ijobvl = 1;
    ilvl = false;
  } else if (Lsame(jobvl, 'V')) {
    ijobvl = 2;
    ilvl = true;
  } else {
    ijobvl = -1;
    ilvl = false;
  }
  if (Lsame(jobvr, 'N')) {
    ijobvr = 1;
    ilvr = false;
  } else if (Lsame(jobvr, 'V')) {
    ijobvr = 2;
    ilvr = true;
  } else {
    ijobvr = -1;
    ilvr = false;
  }
  ilv = ilvl || ilvr;
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  if (ijobvl <= 0) {
    info = -1;
  } else if (ijobvr <= 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldvl < 1 || (ilvl && ldvl < n)) {
    info = -12;
  } else if (ldvr < 1 || (ilvr && ldvr < n)) {
    info = -14;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV. The workspace is
  // computed assuming ILO = 1 and IHI = N, the worst case.)
  if (info == 0) {
    minwrk = Max<blasint>(1, 8 * n);
    maxwrk = Max<blasint>(1, n * (7 + Ilaenv<T>(1, "GEQRF", " ", n, 1, n, 0)));
    maxwrk =
        Max<blasint>(maxwrk, n * (7 + Ilaenv<T>(1, "ORMQR", " ", n, 1, n, 0)));
    if (ilvl) {
      maxwrk = Max<blasint>(maxwrk,
                            n * (7 + Ilaenv<T>(1, "ORGQR", " ", n, 1, n, -1)));
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) info = -16;
  }
  if (info != 0) {
    Xerbla<T>("GGEV", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S');
  bignum = one / smlnum;
  smlnum = std::sqrt(smlnum) / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', n, n, a, lda, work);
  ilascl = false;
  if (anrm > zero && anrm < smlnum) {
    anrmto = smlnum;
    ilascl = true;
  } else if (anrm > bignum) {
    anrmto = bignum;
    ilascl = true;
  }
  if (ilascl) ierr = Lascl('G', 0, 0, anrm, anrmto, n, n, a, lda);
  // Scale B if max element outside range [SMLNUM,BIGNUM]
  bnrm = Lange('M', n, n, b, ldb, work);
  ilbscl = false;
  if (bnrm > zero && bnrm < smlnum) {
    bnrmto = smlnum;
    ilbscl = true;
  } else if (bnrm > bignum) {
    bnrmto = bignum;
    ilbscl = true;
  }
  if (ilbscl) ierr = Lascl('G', 0, 0, bnrm, bnrmto, n, n, b, ldb);
  // Permute the matrices A, B to isolate eigenvalues if possible
  // (Workspace: need 6*N)
  ileft = 1;
  iright = n + 1;
  iwrk = iright + n;
  ierr = Ggbal('P', n, a, lda, b, ldb, &ilo, &ihi, WORK.ptr(ileft),
               WORK.ptr(iright), WORK.ptr(iwrk));
  // Reduce B to triangular form (QR decomposition of B)
  // (Workspace: need N, prefer N*NB)
  irows = ihi + 1 - ilo;
  if (ilv) {
    icols = n + 1 - ilo;
  } else {
    icols = irows;
  }
  itau = iwrk;
  iwrk = itau + irows;
  ierr = Geqrf(irows, icols, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
               WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Apply the orthogonal transformation to matrix A
  // (Workspace: need N, prefer N*NB)
  ierr =
      Unmqr('L', 'T', irows, icols, irows, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
            A.ptr(ilo, ilo), lda, WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Initialize VL
  // (Workspace: need N, prefer N*NB)
  if (ilvl) {
    Laset('F', n, n, zero, one, vl, ldvl);
    if (irows > 1) {
      Lacpy('L', irows - 1, irows - 1, B.ptr(ilo + 1, ilo), ldb,
            VL.ptr(ilo + 1, ilo), ldvl);
    }
    ierr = Ungqr(irows, irows, irows, VL.ptr(ilo, ilo), ldvl, WORK.ptr(itau),
                 WORK.ptr(iwrk), lwork + 1 - iwrk);
  }
  // Initialize VR
  if (ilvr) Laset('F', n, n, zero, one, vr, ldvr);
  // Reduce to generalized Hessenberg form
  // (Workspace: none needed)
  if (ilv) {
    // Eigenvectors requested -- work on whole matrix.
    ierr = Gghrd(jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr);
  } else {
    ierr = Gghrd('N', 'N', irows, 1, irows, A.ptr(ilo, ilo), lda,
                 B.ptr(ilo, ilo), ldb, vl, ldvl, vr, ldvr);
  }
  // Perform QZ algorithm (Compute eigenvalues, and optionally, the
  // Schur forms and Schur vectors)
  // (Workspace: need N)
  iwrk = itau;
  if (ilv) {
    chtemp = 'S';
  } else {
    chtemp = 'E';
  }
  ierr =
      Hgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, alphar, alphai,
            beta, vl, ldvl, vr, ldvr, WORK.ptr(iwrk), lwork + 1 - iwrk);
  if (ierr != 0) {
    if (ierr > 0 && ierr <= n) {
      info = ierr;
    } else if (ierr > n && ierr <= 2 * n) {
      info = ierr - n;
    } else {
      info = n + 1;
    }
    goto L110;
  }
  // Compute Eigenvectors
  // (Workspace: need 6*N)
  if (ilv) {
    if (ilvl) {
      if (ilvr) {
        chtemp = 'B';
      } else {
        chtemp = 'L';
      }
    } else {
      chtemp = 'R';
    }
    ierr = Tgevc(chtemp, 'B', ldumma, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, n,
                 &in, WORK.ptr(iwrk));
    if (ierr != 0) {
      info = n + 2;
      goto L110;
    }
    // Undo balancing on VL and VR and normalization
    // (Workspace: none needed)
    if (ilvl) {
      ierr = Ggbak('P', 'L', n, ilo, ihi, WORK.ptr(ileft), WORK.ptr(iright), n,
                   vl, ldvl);
      for (jc = 1; jc <= n; ++jc) {
        if (ALPHAI(jc) < zero) goto L50;
        temp = zero;
        if (ALPHAI(jc) == zero) {
          for (jr = 1; jr <= n; ++jr) {
            temp = Max(temp, std::abs(VL(jr, jc)));
          }
        } else {
          for (jr = 1; jr <= n; ++jr) {
            temp = Max(temp, std::abs(VL(jr, jc)) + std::abs(VL(jr, jc + 1)));
          }
        }
        if (temp < smlnum) goto L50;
        temp = one / temp;
        if (ALPHAI(jc) == zero) {
          for (jr = 1; jr <= n; ++jr) {
            VL(jr, jc) = VL(jr, jc) * temp;
          }
        } else {
          for (jr = 1; jr <= n; ++jr) {
            VL(jr, jc) = VL(jr, jc) * temp;
            VL(jr, jc + 1) = VL(jr, jc + 1) * temp;
          }
        }
      L50:;
      }
    }
    if (ilvr) {
      ierr = Ggbak('P', 'R', n, ilo, ihi, WORK.ptr(ileft), WORK.ptr(iright), n,
                   vr, ldvr);
      for (jc = 1; jc <= n; ++jc) {
        if (ALPHAI(jc) < zero) goto L100;
        temp = zero;
        if (ALPHAI(jc) == zero) {
          for (jr = 1; jr <= n; ++jr) {
            temp = Max(temp, std::abs(VR(jr, jc)));
          }
        } else {
          for (jr = 1; jr <= n; ++jr) {
            temp = Max(temp, std::abs(VR(jr, jc)) + std::abs(VR(jr, jc + 1)));
          }
        }
        if (temp < smlnum) goto L100;
        temp = one / temp;
        if (ALPHAI(jc) == zero) {
          for (jr = 1; jr <= n; ++jr) {
            VR(jr, jc) = VR(jr, jc) * temp;
          }
        } else {
          for (jr = 1; jr <= n; ++jr) {
            VR(jr, jc) = VR(jr, jc) * temp;
            VR(jr, jc + 1) = VR(jr, jc + 1) * temp;
          }
        }
      L100:;
      }
    }
    // End of eigenvector calculation
  }
  // Undo scaling if necessary
L110:
  if (ilascl) {
    ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alphar, n);
    ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alphai, n);
  }
  if (ilbscl) {
    ierr = Lascl('G', 0, 0, bnrmto, bnrm, n, 1, beta, n);
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// xGGEV (complex T): as the real routine for the eigenvalues ALPHA(j) /
// BETA(j). WORK has LWORK >= max(1, 2*N) elements, RWORK 8*N.
template <typename T>
blasint Ggev(char jobvl, char jobvr, blasint n, T* a, blasint lda, T* b,
             blasint ldb, T* alpha, T* beta, T* vl, blasint ldvl, T* vr,
             blasint ldvr, T* work, blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<T> ALPHA(alpha);
  const Matrix<T> B(b, ldb);
  const Vector<T> BETA(beta);
  blasint ldumma[1] = {0};
  const Vector<R> RWORK(rwork);
  const Matrix<T> VL(vl, ldvl);
  const Matrix<T> VR(vr, ldvr);
  const Vector<T> WORK(work);
  blasint icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, in, iright, irows,
      irwrk, itau, iwrk, jc, jr, lwkmin, lwkopt;
  bool ilascl, ilbscl, ilv, ilvl, ilvr, lquery;
  R anrm, anrmto, bignum, bnrm, bnrmto, eps, smlnum, temp;
  char chtemp;
  if (Lsame(jobvl, 'N')) {
    ijobvl = 1;
    ilvl = false;
  } else if (Lsame(jobvl, 'V')) {
    ijobvl = 2;
    ilvl = true;
  } else {
    ijobvl = -1;
    ilvl = false;
  }
  if (Lsame(jobvr, 'N')) {
    ijobvr = 1;
    ilvr = false;
  } else if (Lsame(jobvr, 'V')) {
    ijobvr = 2;
    ilvr = true;
  } else {
    ijobvr = -1;
    ilvr = false;
  }
  ilv = ilvl || ilvr;
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  if (ijobvl <= 0) {
    info = -1;
  } else if (ijobvr <= 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldvl < 1 || (ilvl && ldvl < n)) {
    info = -11;
  } else if (ldvr < 1 || (ilvr && ldvr < n)) {
    info = -13;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV. The workspace is
  // computed assuming ILO = 1 and IHI = N, the worst case.)
  if (info == 0) {
    lwkmin = Max<blasint>(1, 2 * n);
    lwkopt = Max<blasint>(1, n + n * Ilaenv<T>(1, "GEQRF", " ", n, 1, n, 0));
    lwkopt =
        Max<blasint>(lwkopt, n + n * Ilaenv<T>(1, "UNMQR", " ", n, 1, n, 0));
    if (ilvl) {
      lwkopt =
          Max<blasint>(lwkopt, n + n * Ilaenv<T>(1, "UNGQR", " ", n, 1, n, -1));
    }
    WORK(1) = WorkSize<T>(lwkopt);
    if (lwork < lwkmin && !lquery) info = -15;
  }
  if (info != 0) {
    Xerbla<T>("GGEV", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  // Get machine constants
  eps = Lamch<R>('E') * Lamch<R>('B');
  smlnum = Lamch<R>('S');
  bignum = one / smlnum;
  smlnum = std::sqrt(smlnum) / eps;
  bignum = one / smlnum;
  // Scale A if max element outside range [SMLNUM,BIGNUM]
  anrm = Lange('M', n, n, a, lda, rwork);
  ilascl = false;
  if (anrm > zero && anrm < smlnum) {
    anrmto = smlnum;
    ilascl = true;
  } else if (anrm > bignum) {
    anrmto = bignum;
    ilascl = true;
  }
  if (ilascl) ierr = Lascl('G', 0, 0, anrm, anrmto, n, n, a, lda);
  // Scale B if max element outside range [SMLNUM,BIGNUM]
  bnrm = Lange('M', n, n, b, ldb, rwork);
  ilbscl = false;
  if (bnrm > zero && bnrm < smlnum) {
    bnrmto = smlnum;
    ilbscl = true;
  } else if (bnrm > bignum) {
    bnrmto = bignum;
    ilbscl = true;
  }
  if (ilbscl) ierr = Lascl('G', 0, 0, bnrm, bnrmto, n, n, b, ldb);
  // Permute the matrices A, B to isolate eigenvalues if possible
  // (Real Workspace: need 6*N)
  ileft = 1;
  iright = n + 1;
  irwrk = iright + n;
  ierr = Ggbal('P', n, a, lda, b, ldb, &ilo, &ihi, RWORK.ptr(ileft),
               RWORK.ptr(iright), RWORK.ptr(irwrk));
  // Reduce B to triangular form (QR decomposition of B)
  // (Complex Workspace: need N, prefer N*NB)
  irows = ihi + 1 - ilo;
  if (ilv) {
    icols = n + 1 - ilo;
  } else {
    icols = irows;
  }
  itau = 1;
  iwrk = itau + irows;
  ierr = Geqrf(irows, icols, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
               WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Apply the orthogonal transformation to matrix A
  // (Complex Workspace: need N, prefer N*NB)
  ierr =
      Unmqr('L', 'C', irows, icols, irows, B.ptr(ilo, ilo), ldb, WORK.ptr(itau),
            A.ptr(ilo, ilo), lda, WORK.ptr(iwrk), lwork + 1 - iwrk);
  // Initialize VL
  // (Complex Workspace: need N, prefer N*NB)
  if (ilvl) {
    Laset('F', n, n, czero, cone, vl, ldvl);
    if (irows > 1) {
      Lacpy('L', irows - 1, irows - 1, B.ptr(ilo + 1, ilo), ldb,
            VL.ptr(ilo + 1, ilo), ldvl);
    }
    ierr = Ungqr(irows, irows, irows, VL.ptr(ilo, ilo), ldvl, WORK.ptr(itau),
                 WORK.ptr(iwrk), lwork + 1 - iwrk);
  }
  // Initialize VR
  if (ilvr) Laset('F', n, n, czero, cone, vr, ldvr);
  // Reduce to generalized Hessenberg form
  if (ilv) {
    // Eigenvectors requested -- work on whole matrix.
    ierr = Gghrd(jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, vl, ldvl, vr, ldvr);
  } else {
    ierr = Gghrd('N', 'N', irows, 1, irows, A.ptr(ilo, ilo), lda,
                 B.ptr(ilo, ilo), ldb, vl, ldvl, vr, ldvr);
  }
  // Perform QZ algorithm (Compute eigenvalues, and optionally, the
  // Schur form and Schur vectors)
  // (Complex Workspace: need N)
  // (Real Workspace: need N)
  iwrk = itau;
  if (ilv) {
    chtemp = 'S';
  } else {
    chtemp = 'E';
  }
  ierr =
      Hgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, lda, b, ldb, alpha, beta, vl,
            ldvl, vr, ldvr, WORK.ptr(iwrk), lwork + 1 - iwrk, RWORK.ptr(irwrk));
  if (ierr != 0) {
    if (ierr > 0 && ierr <= n) {
      info = ierr;
    } else if (ierr > n && ierr <= 2 * n) {
      info = ierr - n;
    } else {
      info = n + 1;
    }
    goto L70;
  }
  // Compute Eigenvectors
  // (Real Workspace: need 2*N)
  // (Complex Workspace: need 2*N)
  if (ilv) {
    if (ilvl) {
      if (ilvr) {
        chtemp = 'B';
      } else {
        chtemp = 'L';
      }
    } else {
      chtemp = 'R';
    }
    ierr = Tgevc(chtemp, 'B', ldumma, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, n,
                 &in, WORK.ptr(iwrk), RWORK.ptr(irwrk));
    if (ierr != 0) {
      info = n + 2;
      goto L70;
    }
    // Undo balancing on VL and VR and normalization
    // (Workspace: none needed)
    if (ilvl) {
      ierr = Ggbak('P', 'L', n, ilo, ihi, RWORK.ptr(ileft), RWORK.ptr(iright),
                   n, vl, ldvl);
      for (jc = 1; jc <= n; ++jc) {
        temp = zero;
        for (jr = 1; jr <= n; ++jr) {
          temp = Max(temp, Cabs1(VL(jr, jc)));
        }
        if (temp < smlnum) goto L30;
        temp = one / temp;
        for (jr = 1; jr <= n; ++jr) {
          VL(jr, jc) = VL(jr, jc) * temp;
        }
      L30:;
      }
    }
    if (ilvr) {
      ierr = Ggbak('P', 'R', n, ilo, ihi, RWORK.ptr(ileft), RWORK.ptr(iright),
                   n, vr, ldvr);
      for (jc = 1; jc <= n; ++jc) {
        temp = zero;
        for (jr = 1; jr <= n; ++jr) {
          temp = Max(temp, Cabs1(VR(jr, jc)));
        }
        if (temp < smlnum) goto L60;
        temp = one / temp;
        for (jr = 1; jr <= n; ++jr) {
          VR(jr, jc) = VR(jr, jc) * temp;
        }
      L60:;
      }
    }
  }
  // Undo scaling if necessary
L70:
  if (ilascl) ierr = Lascl('G', 0, 0, anrmto, anrm, n, 1, alpha, n);
  if (ilbscl) ierr = Lascl('G', 0, 0, bnrmto, bnrm, n, 1, beta, n);
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                    \
  template blasint Gges<R>(char, char, char, GgesSelectReal<R>, blasint, R*,  \
                           blasint, R*, blasint, blasint*, R*, R*, R*, R*,    \
                           blasint, R*, blasint, R*, blasint, blasint*);      \
  template blasint Ggev<R>(char, char, blasint, R*, blasint, R*, blasint, R*, \
                           R*, R*, R*, blasint, R*, blasint, R*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                 \
  template blasint Gges<T>(char, char, char, GgesSelectComplex<RealT<T>>,     \
                           blasint, T*, blasint, T*, blasint, blasint*, T*,   \
                           T*, T*, blasint, T*, blasint, T*, blasint,         \
                           RealT<T>*, blasint*);                              \
  template blasint Ggev<T>(char, char, blasint, T*, blasint, T*, blasint, T*, \
                           T*, T*, blasint, T*, blasint, T*, blasint,         \
                           RealT<T>*);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GGEV_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##ggev_(                                             \
      const char* jobvl, const char* jobvr, const hwyblas_int* n, R* a,       \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* alphar,        \
      R* alphai, R* beta, R* vl, const hwyblas_int* ldvl, R* vr,              \
      const hwyblas_int* ldvr, R* work, const hwyblas_int* lwork,             \
      hwyblas_int* info) {                                                    \
    *info = hwyblas::lapack::Ggev(*jobvl, *jobvr, *n, a, *lda, b, *ldb,       \
                                  alphar, alphai, beta, vl, *ldvl, vr, *ldvr, \
                                  work, *lwork);                              \
  }

HWYBLAS_LAPACK_GGEV_REAL(s, float)
HWYBLAS_LAPACK_GGEV_REAL(d, double)

#undef HWYBLAS_LAPACK_GGEV_REAL

#define HWYBLAS_LAPACK_GGEV_COMPLEX(prefix, T, R)                              \
  HWYBLAS_API void prefix##ggev_(                                              \
      const char* jobvl, const char* jobvr, const hwyblas_int* n, R* a,        \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* alpha, R* beta, \
      R* vl, const hwyblas_int* ldvl, R* vr, const hwyblas_int* ldvr, R* work, \
      const hwyblas_int* lwork, R* rwork, hwyblas_int* info) {                 \
    *info = hwyblas::lapack::Ggev(*jobvl, *jobvr, *n, As<T>(a), *lda,          \
                                  As<T>(b), *ldb, As<T>(alpha), As<T>(beta),   \
                                  As<T>(vl), *ldvl, As<T>(vr), *ldvr,          \
                                  As<T>(work), *lwork, rwork);                 \
  }

HWYBLAS_LAPACK_GGEV_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GGEV_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GGEV_COMPLEX

// SELCTG is a Fortran LOGICAL FUNCTION (nonzero is true) of ALPHAR, ALPHAI
// and BETA, or of ALPHA and BETA (pointers to their real and imaginary
// parts); BWORK is a LOGICAL array.
HWYBLAS_API void sgges_(const char* jobvsl, const char* jobvsr,
                        const char* sort, hwyblas_select3_float selctg,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* b, const hwyblas_int* ldb, hwyblas_int* sdim,
                        float* alphar, float* alphai, float* beta, float* vsl,
                        const hwyblas_int* ldvsl, float* vsr,
                        const hwyblas_int* ldvsr, float* work,
                        const hwyblas_int* lwork, hwyblas_int* bwork,
                        hwyblas_int* info) {
  *info = hwyblas::lapack::Gges(*jobvsl, *jobvsr, *sort, selctg, *n, a, *lda, b,
                                *ldb, sdim, alphar, alphai, beta, vsl, *ldvsl,
                                vsr, *ldvsr, work, *lwork, bwork);
}
HWYBLAS_API void dgges_(const char* jobvsl, const char* jobvsr,
                        const char* sort, hwyblas_select3_double selctg,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* b, const hwyblas_int* ldb, hwyblas_int* sdim,
                        double* alphar, double* alphai, double* beta,
                        double* vsl, const hwyblas_int* ldvsl, double* vsr,
                        const hwyblas_int* ldvsr, double* work,
                        const hwyblas_int* lwork, hwyblas_int* bwork,
                        hwyblas_int* info) {
  *info = hwyblas::lapack::Gges(*jobvsl, *jobvsr, *sort, selctg, *n, a, *lda, b,
                                *ldb, sdim, alphar, alphai, beta, vsl, *ldvsl,
                                vsr, *ldvsr, work, *lwork, bwork);
}
HWYBLAS_API void cgges_(const char* jobvsl, const char* jobvsr,
                        const char* sort, hwyblas_select2_float selctg,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* b, const hwyblas_int* ldb, hwyblas_int* sdim,
                        float* alpha, float* beta, float* vsl,
                        const hwyblas_int* ldvsl, float* vsr,
                        const hwyblas_int* ldvsr, float* work,
                        const hwyblas_int* lwork, float* rwork,
                        hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gges(
      *jobvsl, *jobvsr, *sort, selctg, *n, As<c32>(a), *lda, As<c32>(b), *ldb,
      sdim, As<c32>(alpha), As<c32>(beta), As<c32>(vsl), *ldvsl, As<c32>(vsr),
      *ldvsr, As<c32>(work), *lwork, rwork, bwork);
}
HWYBLAS_API void zgges_(const char* jobvsl, const char* jobvsr,
                        const char* sort, hwyblas_select2_double selctg,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* b, const hwyblas_int* ldb, hwyblas_int* sdim,
                        double* alpha, double* beta, double* vsl,
                        const hwyblas_int* ldvsl, double* vsr,
                        const hwyblas_int* ldvsr, double* work,
                        const hwyblas_int* lwork, double* rwork,
                        hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gges(
      *jobvsl, *jobvsr, *sort, selctg, *n, As<c64>(a), *lda, As<c64>(b), *ldb,
      sdim, As<c64>(alpha), As<c64>(beta), As<c64>(vsl), *ldvsl, As<c64>(vsr),
      *ldvsr, As<c64>(work), *lwork, rwork, bwork);
}

}  // extern "C"
