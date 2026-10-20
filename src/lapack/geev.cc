// The eigenvalues and eigenvectors of a general matrix: xGEEV, the expert
// driver xGEEVX (balancing options and condition numbers), and the Schur
// factorization with condition numbers of a cluster of eigenvalues, xGEESX.
//
// The real and the complex routines differ in the eigenvalues (WR and WI
// for W), the normalization of the complex eigenvectors and the
// workspaces, so they are ported as two overloads each.
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

// xGEEV (real T): computes the eigenvalues WR + i*WI of the N-by-N matrix A
// (destroyed) and, for JOBVL = 'V' / JOBVR = 'V', its left / right
// eigenvectors (JOBVL, JOBVR = 'N': none): u(j)**H * A = lambda(j) *
// u(j)**H in VL, A * v(j) = lambda(j) * v(j) in VR. Complex conjugate
// eigenvalue pairs are consecutive, the one with positive imaginary part
// first; the eigenvector of the pair is then VR(:,j) + i*VR(:,j+1) (and
// its conjugate). Each eigenvector has Euclidean norm 1 and a largest
// component that is real. WORK has LWORK >= max(1, 3*N) elements, 4*N if
// eigenvectors are wanted (LWORK = -1: workspace query).
//
// Returns INFO: 0, -i for an illegal argument i, or i > 0 if the QR
// algorithm failed: no eigenvectors were computed, and WR(i+1:N),
// WI(i+1:N) hold the eigenvalues that converged.
template <typename R>
blasint Geev(char jobvl, char jobvr, blasint n, R* a, blasint lda, R* wr, R* wi,
             R* vl, blasint ldvl, R* vr, blasint ldvr, R* work, blasint lwork) {
  using T = R;
  const R zero = R(0);
  const R one = R(1);
  R dum[1];
  blasint select[1];
  const Matrix<R> VL(vl, ldvl);
  const Matrix<R> VR(vr, ldvr);
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  blasint hswork, i, ibal, ihi, ilo, itau, iwrk, k, lwork_trevc, maxwrk, minwrk,
      nout;
  bool lquery, scalea, wantvl, wantvr;
  R anrm, bignum, cs, cscale, eps, r, scl, smlnum, sn;
  // Set wherever it is read (eigenvectors are wanted); initialized for the
  // compiler.
  char side = 'N';
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  wantvl = Lsame(jobvl, 'V');
  wantvr = Lsame(jobvr, 'V');
  if ((!wantvl) && (!Lsame(jobvl, 'N'))) {
    info = -1;
  } else if ((!wantvr) && (!Lsame(jobvr, 'N'))) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldvl < 1 || (wantvl && ldvl < n)) {
    info = -9;
  } else if (ldvr < 1 || (wantvr && ldvr < n)) {
    info = -11;
  }
  // Compute workspace
  //  (Note: Comments in the code beginning "Workspace:" describe the
  //   minimal amount of workspace needed at that point in the code,
  //   as well as the preferred amount for good performance.
  //   NB refers to the optimal block size for the immediately
  //   following subroutine, as returned by ILAENV.
  //   HSWORK refers to the workspace preferred by DHSEQR, as
  //   calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //   the worst case.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = 2 * n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      if (wantvl) {
        minwrk = 4 * n;
        maxwrk = Max<blasint>(
            maxwrk, 2 * n + (n - 1) * Ilaenv<T>(1, "ORGHR", " ", n, 1, n, -1));
        info = Hseqr('S', 'V', n, 1, n, a, lda, wr, wi, vl, ldvl, work, -1);
        hswork = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + 1, n + hswork);
        Trevc3('L', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1);
        lwork_trevc = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + lwork_trevc);
        maxwrk = Max<blasint>(maxwrk, 4 * n);
      } else if (wantvr) {
        minwrk = 4 * n;
        maxwrk = Max<blasint>(
            maxwrk, 2 * n + (n - 1) * Ilaenv<T>(1, "ORGHR", " ", n, 1, n, -1));
        info = Hseqr('S', 'V', n, 1, n, a, lda, wr, wi, vr, ldvr, work, -1);
        hswork = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + 1, n + hswork);
        Trevc3('R', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1);
        lwork_trevc = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + lwork_trevc);
        maxwrk = Max<blasint>(maxwrk, 4 * n);
      } else {
        minwrk = 3 * n;
        info = Hseqr('E', 'N', n, 1, n, a, lda, wr, wi, vr, ldvr, work, -1);
        hswork = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + 1, n + hswork);
      }
      maxwrk = Max<blasint>(maxwrk, minwrk);
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -13;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEEV", -info);
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
  anrm = Lange('M', n, n, a, lda, dum);
  scalea = false;
  if (anrm > zero && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) Lascl('G', 0, 0, anrm, cscale, n, n, a, lda);
  // Balance the matrix
  // (Workspace: need N)
  ibal = 1;
  Gebal('B', n, a, lda, &ilo, &ihi, WORK.ptr(ibal));
  // Reduce to upper Hessenberg form
  // (Workspace: need 3*N, prefer 2*N+N*NB)
  itau = ibal + n;
  iwrk = itau + n;
  Gehrd(n, ilo, ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk), lwork - iwrk + 1);
  if (wantvl) {
    // Want left eigenvectors
    // Copy Householder vectors to VL
    side = 'L';
    Lacpy('L', n, n, a, lda, vl, ldvl);
    // Generate orthogonal matrix in VL
    // (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    Unghr(n, ilo, ihi, vl, ldvl, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VL
    // (Workspace: need N+1, prefer N+HSWORK (see comments) )
    iwrk = itau;
    info = Hseqr('S', 'V', n, ilo, ihi, a, lda, wr, wi, vl, ldvl,
                 WORK.ptr(iwrk), lwork - iwrk + 1);
    if (wantvr) {
      // Want left and right eigenvectors
      // Copy Schur vectors to VR
      side = 'B';
      Lacpy('F', n, n, vl, ldvl, vr, ldvr);
    }
  } else if (wantvr) {
    // Want right eigenvectors
    // Copy Householder vectors to VR
    side = 'R';
    Lacpy('L', n, n, a, lda, vr, ldvr);
    // Generate orthogonal matrix in VR
    // (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    Unghr(n, ilo, ihi, vr, ldvr, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VR
    // (Workspace: need N+1, prefer N+HSWORK (see comments) )
    iwrk = itau;
    info = Hseqr('S', 'V', n, ilo, ihi, a, lda, wr, wi, vr, ldvr,
                 WORK.ptr(iwrk), lwork - iwrk + 1);
  } else {
    // Compute eigenvalues only
    // (Workspace: need N+1, prefer N+HSWORK (see comments) )
    iwrk = itau;
    info = Hseqr('E', 'N', n, ilo, ihi, a, lda, wr, wi, vr, ldvr,
                 WORK.ptr(iwrk), lwork - iwrk + 1);
  }
  // If INFO .NE. 0 from DHSEQR, then quit
  if (info != 0) goto L50;
  if (wantvl || wantvr) {
    // Compute left and/or right eigenvectors
    // (Workspace: need 4*N, prefer N + N + 2*N*NB)
    Trevc3(side, 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout,
           WORK.ptr(iwrk), lwork - iwrk + 1);
  }
  if (wantvl) {
    // Undo balancing of left eigenvectors
    // (Workspace: need N)
    Gebak('B', 'L', n, ilo, ihi, WORK.ptr(ibal), n, vl, ldvl);
    // Normalize left eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      if (WI(i) == zero) {
        scl = one / Nrm2(n, VL.ptr(1, i), 1);
        Rscal(n, scl, VL.ptr(1, i), 1);
      } else if (WI(i) > zero) {
        scl =
            one / Lapy2(Nrm2(n, VL.ptr(1, i), 1), Nrm2(n, VL.ptr(1, i + 1), 1));
        Rscal(n, scl, VL.ptr(1, i), 1);
        Rscal(n, scl, VL.ptr(1, i + 1), 1);
        for (k = 1; k <= n; ++k) {
          WORK(iwrk + k - 1) =
              (VL(k, i) * VL(k, i)) + (VL(k, i + 1) * VL(k, i + 1));
        }
        k = Iamax(n, WORK.ptr(iwrk), 1);
        Lartg(VL(k, i), VL(k, i + 1), &cs, &sn, &r);
        Rot(n, VL.ptr(1, i), 1, VL.ptr(1, i + 1), 1, cs, sn);
        VL(k, i + 1) = zero;
      }
    }
  }
  if (wantvr) {
    // Undo balancing of right eigenvectors
    // (Workspace: need N)
    Gebak('B', 'R', n, ilo, ihi, WORK.ptr(ibal), n, vr, ldvr);
    // Normalize right eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      if (WI(i) == zero) {
        scl = one / Nrm2(n, VR.ptr(1, i), 1);
        Rscal(n, scl, VR.ptr(1, i), 1);
      } else if (WI(i) > zero) {
        scl =
            one / Lapy2(Nrm2(n, VR.ptr(1, i), 1), Nrm2(n, VR.ptr(1, i + 1), 1));
        Rscal(n, scl, VR.ptr(1, i), 1);
        Rscal(n, scl, VR.ptr(1, i + 1), 1);
        for (k = 1; k <= n; ++k) {
          WORK(iwrk + k - 1) =
              (VR(k, i) * VR(k, i)) + (VR(k, i + 1) * VR(k, i + 1));
        }
        k = Iamax(n, WORK.ptr(iwrk), 1);
        Lartg(VR(k, i), VR(k, i + 1), &cs, &sn, &r);
        Rot(n, VR.ptr(1, i), 1, VR.ptr(1, i + 1), 1, cs, sn);
        VR(k, i + 1) = zero;
      }
    }
  }
  // Undo scaling if necessary
L50:
  if (scalea) {
    Lascl('G', 0, 0, cscale, anrm, n - info, 1, WR.ptr(info + 1),
          Max<blasint>(n - info, 1));
    Lascl('G', 0, 0, cscale, anrm, n - info, 1, WI.ptr(info + 1),
          Max<blasint>(n - info, 1));
    if (info > 0) {
      Lascl('G', 0, 0, cscale, anrm, ilo - 1, 1, wr, n);
      Lascl('G', 0, 0, cscale, anrm, ilo - 1, 1, wi, n);
    }
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// xGEEV (complex T): as the real routine, with the eigenvalues W and one
// column of VL / VR per eigenvector. WORK has LWORK >= max(1, 2*N)
// elements, RWORK 2*N.
template <typename T>
blasint Geev(char jobvl, char jobvr, blasint n, T* a, blasint lda, T* w, T* vl,
             blasint ldvl, T* vr, blasint ldvr, T* work, blasint lwork,
             RealT<T>* rwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const R one = R(1);
  R dum[1];
  const Vector<R> RWORK(rwork);
  blasint select[1];
  const Matrix<T> VL(vl, ldvl);
  const Matrix<T> VR(vr, ldvr);
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  blasint hswork, i, ibal, ihi, ilo, itau, iwrk, k, lwork_trevc, maxwrk, minwrk,
      nout;
  // Set wherever it is read (eigenvectors are wanted); initialized for the
  // compiler.
  blasint irwork = 0;
  bool lquery, scalea, wantvl, wantvr;
  R anrm, bignum, cscale, eps, scl, smlnum;
  T tmp;
  // Set wherever it is read (eigenvectors are wanted); initialized for the
  // compiler.
  char side = 'N';
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  wantvl = Lsame(jobvl, 'V');
  wantvr = Lsame(jobvr, 'V');
  if ((!wantvl) && (!Lsame(jobvl, 'N'))) {
    info = -1;
  } else if ((!wantvr) && (!Lsame(jobvr, 'N'))) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldvl < 1 || (wantvl && ldvl < n)) {
    info = -8;
  } else if (ldvr < 1 || (wantvr && ldvr < n)) {
    info = -10;
  }
  // Compute workspace
  //  (Note: Comments in the code beginning "Workspace:" describe the
  //   minimal amount of workspace needed at that point in the code,
  //   as well as the preferred amount for good performance.
  //   CWorkspace refers to complex workspace, and RWorkspace to real
  //   workspace. NB refers to the optimal block size for the
  //   immediately following subroutine, as returned by ILAENV.
  //   HSWORK refers to the workspace preferred by ZHSEQR, as
  //   calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //   the worst case.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      minwrk = 2 * n;
      if (wantvl) {
        maxwrk = Max<blasint>(
            maxwrk, n + (n - 1) * Ilaenv<T>(1, "UNGHR", " ", n, 1, n, -1));
        Trevc3('L', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1, rwork, -1);
        lwork_trevc = static_cast<blasint>(std::real(WORK(1)));
        maxwrk = Max<blasint>(maxwrk, n + lwork_trevc);
        info = Hseqr('S', 'V', n, 1, n, a, lda, w, vl, ldvl, work, -1);
      } else if (wantvr) {
        maxwrk = Max<blasint>(
            maxwrk, n + (n - 1) * Ilaenv<T>(1, "UNGHR", " ", n, 1, n, -1));
        Trevc3('R', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1, rwork, -1);
        lwork_trevc = static_cast<blasint>(std::real(WORK(1)));
        maxwrk = Max<blasint>(maxwrk, n + lwork_trevc);
        info = Hseqr('S', 'V', n, 1, n, a, lda, w, vr, ldvr, work, -1);
      } else {
        info = Hseqr('E', 'N', n, 1, n, a, lda, w, vr, ldvr, work, -1);
      }
      hswork = static_cast<blasint>(std::real(WORK(1)));
      maxwrk = Max<blasint>(maxwrk, hswork, minwrk);
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -12;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEEV", -info);
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
  anrm = Lange('M', n, n, a, lda, dum);
  scalea = false;
  if (anrm > zero && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) Lascl('G', 0, 0, anrm, cscale, n, n, a, lda);
  // Balance the matrix
  // (CWorkspace: none)
  // (RWorkspace: need N)
  ibal = 1;
  Gebal('B', n, a, lda, &ilo, &ihi, RWORK.ptr(ibal));
  // Reduce to upper Hessenberg form
  // (CWorkspace: need 2*N, prefer N+N*NB)
  // (RWorkspace: none)
  itau = 1;
  iwrk = itau + n;
  Gehrd(n, ilo, ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk), lwork - iwrk + 1);
  if (wantvl) {
    // Want left eigenvectors
    // Copy Householder vectors to VL
    side = 'L';
    Lacpy('L', n, n, a, lda, vl, ldvl);
    // Generate unitary matrix in VL
    // (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
    // (RWorkspace: none)
    Unghr(n, ilo, ihi, vl, ldvl, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VL
    // (CWorkspace: need 1, prefer HSWORK (see comments) )
    // (RWorkspace: none)
    iwrk = itau;
    info = Hseqr('S', 'V', n, ilo, ihi, a, lda, w, vl, ldvl, WORK.ptr(iwrk),
                 lwork - iwrk + 1);
    if (wantvr) {
      // Want left and right eigenvectors
      // Copy Schur vectors to VR
      side = 'B';
      Lacpy('F', n, n, vl, ldvl, vr, ldvr);
    }
  } else if (wantvr) {
    // Want right eigenvectors
    // Copy Householder vectors to VR
    side = 'R';
    Lacpy('L', n, n, a, lda, vr, ldvr);
    // Generate unitary matrix in VR
    // (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
    // (RWorkspace: none)
    Unghr(n, ilo, ihi, vr, ldvr, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VR
    // (CWorkspace: need 1, prefer HSWORK (see comments) )
    // (RWorkspace: none)
    iwrk = itau;
    info = Hseqr('S', 'V', n, ilo, ihi, a, lda, w, vr, ldvr, WORK.ptr(iwrk),
                 lwork - iwrk + 1);
  } else {
    // Compute eigenvalues only
    // (CWorkspace: need 1, prefer HSWORK (see comments) )
    // (RWorkspace: none)
    iwrk = itau;
    info = Hseqr('E', 'N', n, ilo, ihi, a, lda, w, vr, ldvr, WORK.ptr(iwrk),
                 lwork - iwrk + 1);
  }
  // If INFO .NE. 0 from ZHSEQR, then quit
  if (info != 0) goto L50;
  if (wantvl || wantvr) {
    // Compute left and/or right eigenvectors
    // (CWorkspace: need 2*N, prefer N + 2*N*NB)
    // (RWorkspace: need 2*N)
    irwork = ibal + n;
    Trevc3(side, 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout,
           WORK.ptr(iwrk), lwork - iwrk + 1, RWORK.ptr(irwork), n);
  }
  if (wantvl) {
    // Undo balancing of left eigenvectors
    // (CWorkspace: none)
    // (RWorkspace: need N)
    Gebak('B', 'L', n, ilo, ihi, RWORK.ptr(ibal), n, vl, ldvl);
    // Normalize left eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      scl = one / Nrm2(n, VL.ptr(1, i), 1);
      Rscal(n, scl, VL.ptr(1, i), 1);
      for (k = 1; k <= n; ++k) {
        RWORK(irwork + k - 1) = (std::real(VL(k, i)) * std::real(VL(k, i))) +
                                (std::imag(VL(k, i)) * std::imag(VL(k, i)));
      }
      k = Iamax(n, RWORK.ptr(irwork), 1);
      tmp = Conj(VL(k, i)) / std::sqrt(RWORK(irwork + k - 1));
      Scal(n, tmp, VL.ptr(1, i), 1);
      VL(k, i) = T(std::real(VL(k, i)), zero);
    }
  }
  if (wantvr) {
    // Undo balancing of right eigenvectors
    // (CWorkspace: none)
    // (RWorkspace: need N)
    Gebak('B', 'R', n, ilo, ihi, RWORK.ptr(ibal), n, vr, ldvr);
    // Normalize right eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      scl = one / Nrm2(n, VR.ptr(1, i), 1);
      Rscal(n, scl, VR.ptr(1, i), 1);
      for (k = 1; k <= n; ++k) {
        RWORK(irwork + k - 1) = (std::real(VR(k, i)) * std::real(VR(k, i))) +
                                (std::imag(VR(k, i)) * std::imag(VR(k, i)));
      }
      k = Iamax(n, RWORK.ptr(irwork), 1);
      tmp = Conj(VR(k, i)) / std::sqrt(RWORK(irwork + k - 1));
      Scal(n, tmp, VR.ptr(1, i), 1);
      VR(k, i) = T(std::real(VR(k, i)), zero);
    }
  }
  // Undo scaling if necessary
L50:
  if (scalea) {
    Lascl('G', 0, 0, cscale, anrm, n - info, 1, W.ptr(info + 1),
          Max<blasint>(n - info, 1));
    if (info > 0) {
      Lascl('G', 0, 0, cscale, anrm, ilo - 1, 1, w, n);
    }
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// xGEEVX (real T): as xGEEV, with the balancing BALANC ('N': none, 'P':
// permute, 'S': scale, 'B': both; ILO, IHI and SCALE describe it, as in
// xGEBAL) and the reciprocal condition numbers of the eigenvalues (SENSE =
// 'E', in RCONDE) and/or of the right eigenvectors ('V', in RCONDV; 'B':
// both; 'N': none). SENSE = 'E' or 'B' needs both JOBVL and JOBVR = 'V'.
// ABNRM is the 1-norm of the balanced matrix. WORK has LWORK >= max(1,
// 2*N) elements, 3*N if eigenvectors are wanted, N*N+6*N for SENSE = 'V'
// or 'B'; IWORK has 2*N-2 elements.
//
// Returns INFO as xGEEV (for INFO > 0, no condition numbers either).
template <typename R>
blasint Geevx(char balanc, char jobvl, char jobvr, char sense, blasint n, R* a,
              blasint lda, R* wr, R* wi, R* vl, blasint ldvl, R* vr,
              blasint ldvr, blasint* ilo, blasint* ihi, R* scale, R* abnrm,
              R* rconde, R* rcondv, R* work, blasint lwork, blasint* iwork) {
  using T = R;
  const R zero = R(0);
  const R one = R(1);
  R dum[1];
  const Vector<R> DUM(dum);
  blasint select[1];
  const Matrix<R> VL(vl, ldvl);
  const Matrix<R> VR(vr, ldvr);
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  blasint hswork, i, icond, itau, iwrk, k, lwork_trevc, maxwrk, minwrk, nout;
  bool lquery, scalea, wantvl, wantvr, wntsnb, wntsne, wntsnn, wntsnv;
  R anrm, bignum, cs, cscale, eps, r, scl, smlnum, sn;
  char job;
  // Set wherever it is read (eigenvectors are wanted); initialized for the
  // compiler.
  char side = 'N';
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  wantvl = Lsame(jobvl, 'V');
  wantvr = Lsame(jobvr, 'V');
  wntsnn = Lsame(sense, 'N');
  wntsne = Lsame(sense, 'E');
  wntsnv = Lsame(sense, 'V');
  wntsnb = Lsame(sense, 'B');
  if (!(Lsame(balanc, 'N') || Lsame(balanc, 'S') || Lsame(balanc, 'P') ||
        Lsame(balanc, 'B'))) {
    info = -1;
  } else if ((!wantvl) && (!Lsame(jobvl, 'N'))) {
    info = -2;
  } else if ((!wantvr) && (!Lsame(jobvr, 'N'))) {
    info = -3;
  } else if (!(wntsnn || wntsne || wntsnb || wntsnv) ||
             ((wntsne || wntsnb) && !(wantvl && wantvr))) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldvl < 1 || (wantvl && ldvl < n)) {
    info = -11;
  } else if (ldvr < 1 || (wantvr && ldvr < n)) {
    info = -13;
  }
  // Compute workspace
  //  (Note: Comments in the code beginning "Workspace:" describe the
  //   minimal amount of workspace needed at that point in the code,
  //   as well as the preferred amount for good performance.
  //   NB refers to the optimal block size for the immediately
  //   following subroutine, as returned by ILAENV.
  //   HSWORK refers to the workspace preferred by DHSEQR, as
  //   calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //   the worst case.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      if (wantvl) {
        Trevc3('L', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1);
        lwork_trevc = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + lwork_trevc);
        info = Hseqr('S', 'V', n, 1, n, a, lda, wr, wi, vl, ldvl, work, -1);
      } else if (wantvr) {
        Trevc3('R', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1);
        lwork_trevc = static_cast<blasint>(WORK(1));
        maxwrk = Max<blasint>(maxwrk, n + lwork_trevc);
        info = Hseqr('S', 'V', n, 1, n, a, lda, wr, wi, vr, ldvr, work, -1);
      } else {
        if (wntsnn) {
          info = Hseqr('E', 'N', n, 1, n, a, lda, wr, wi, vr, ldvr, work, -1);
        } else {
          info = Hseqr('S', 'N', n, 1, n, a, lda, wr, wi, vr, ldvr, work, -1);
        }
      }
      hswork = static_cast<blasint>(WORK(1));
      if ((!wantvl) && (!wantvr)) {
        minwrk = 2 * n;
        if (!wntsnn) minwrk = Max<blasint>(minwrk, n * n + 6 * n);
        maxwrk = Max<blasint>(maxwrk, hswork);
        if (!wntsnn) maxwrk = Max<blasint>(maxwrk, n * n + 6 * n);
      } else {
        minwrk = 3 * n;
        if ((!wntsnn) && (!wntsne))
          minwrk = Max<blasint>(minwrk, n * n + 6 * n);
        maxwrk = Max<blasint>(maxwrk, hswork);
        maxwrk = Max<blasint>(
            maxwrk, n + (n - 1) * Ilaenv<T>(1, "ORGHR", " ", n, 1, n, -1));
        if ((!wntsnn) && (!wntsne))
          maxwrk = Max<blasint>(maxwrk, n * n + 6 * n);
        maxwrk = Max<blasint>(maxwrk, 3 * n);
      }
      maxwrk = Max<blasint>(maxwrk, minwrk);
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -21;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEEVX", -info);
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
  icond = 0;
  anrm = Lange('M', n, n, a, lda, dum);
  scalea = false;
  if (anrm > zero && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) Lascl('G', 0, 0, anrm, cscale, n, n, a, lda);
  // Balance the matrix and compute ABNRM
  Gebal(balanc, n, a, lda, ilo, ihi, scale);
  *abnrm = Lange('1', n, n, a, lda, dum);
  if (scalea) {
    DUM(1) = *abnrm;
    Lascl('G', 0, 0, cscale, anrm, 1, 1, dum, 1);
    *abnrm = DUM(1);
  }
  // Reduce to upper Hessenberg form
  // (Workspace: need 2*N, prefer N+N*NB)
  itau = 1;
  iwrk = itau + n;
  Gehrd(n, *ilo, *ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk),
        lwork - iwrk + 1);
  if (wantvl) {
    // Want left eigenvectors
    // Copy Householder vectors to VL
    side = 'L';
    Lacpy('L', n, n, a, lda, vl, ldvl);
    // Generate orthogonal matrix in VL
    // (Workspace: need 2*N-1, prefer N+(N-1)*NB)
    Unghr(n, *ilo, *ihi, vl, ldvl, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VL
    // (Workspace: need 1, prefer HSWORK (see comments) )
    iwrk = itau;
    info = Hseqr('S', 'V', n, *ilo, *ihi, a, lda, wr, wi, vl, ldvl,
                 WORK.ptr(iwrk), lwork - iwrk + 1);
    if (wantvr) {
      // Want left and right eigenvectors
      // Copy Schur vectors to VR
      side = 'B';
      Lacpy('F', n, n, vl, ldvl, vr, ldvr);
    }
  } else if (wantvr) {
    // Want right eigenvectors
    // Copy Householder vectors to VR
    side = 'R';
    Lacpy('L', n, n, a, lda, vr, ldvr);
    // Generate orthogonal matrix in VR
    // (Workspace: need 2*N-1, prefer N+(N-1)*NB)
    Unghr(n, *ilo, *ihi, vr, ldvr, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VR
    // (Workspace: need 1, prefer HSWORK (see comments) )
    iwrk = itau;
    info = Hseqr('S', 'V', n, *ilo, *ihi, a, lda, wr, wi, vr, ldvr,
                 WORK.ptr(iwrk), lwork - iwrk + 1);
  } else {
    // Compute eigenvalues only
    // If condition numbers desired, compute Schur form
    if (wntsnn) {
      job = 'E';
    } else {
      job = 'S';
    }
    // (Workspace: need 1, prefer HSWORK (see comments) )
    iwrk = itau;
    info = Hseqr(job, 'N', n, *ilo, *ihi, a, lda, wr, wi, vr, ldvr,
                 WORK.ptr(iwrk), lwork - iwrk + 1);
  }
  // If INFO .NE. 0 from DHSEQR, then quit
  if (info != 0) goto L50;
  if (wantvl || wantvr) {
    // Compute left and/or right eigenvectors
    // (Workspace: need 3*N, prefer N + 2*N*NB)
    Trevc3(side, 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout,
           WORK.ptr(iwrk), lwork - iwrk + 1);
  }
  // Compute condition numbers if desired
  // (Workspace: need N*N+6*N unless SENSE = 'E')
  if (!wntsnn) {
    icond = Trsna(sense, 'A', select, n, a, lda, vl, ldvl, vr, ldvr, rconde,
                  rcondv, n, &nout, WORK.ptr(iwrk), n, iwork);
  }
  if (wantvl) {
    // Undo balancing of left eigenvectors
    Gebak(balanc, 'L', n, *ilo, *ihi, scale, n, vl, ldvl);
    // Normalize left eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      if (WI(i) == zero) {
        scl = one / Nrm2(n, VL.ptr(1, i), 1);
        Rscal(n, scl, VL.ptr(1, i), 1);
      } else if (WI(i) > zero) {
        scl =
            one / Lapy2(Nrm2(n, VL.ptr(1, i), 1), Nrm2(n, VL.ptr(1, i + 1), 1));
        Rscal(n, scl, VL.ptr(1, i), 1);
        Rscal(n, scl, VL.ptr(1, i + 1), 1);
        for (k = 1; k <= n; ++k) {
          WORK(k) = (VL(k, i) * VL(k, i)) + (VL(k, i + 1) * VL(k, i + 1));
        }
        k = Iamax(n, work, 1);
        Lartg(VL(k, i), VL(k, i + 1), &cs, &sn, &r);
        Rot(n, VL.ptr(1, i), 1, VL.ptr(1, i + 1), 1, cs, sn);
        VL(k, i + 1) = zero;
      }
    }
  }
  if (wantvr) {
    // Undo balancing of right eigenvectors
    Gebak(balanc, 'R', n, *ilo, *ihi, scale, n, vr, ldvr);
    // Normalize right eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      if (WI(i) == zero) {
        scl = one / Nrm2(n, VR.ptr(1, i), 1);
        Rscal(n, scl, VR.ptr(1, i), 1);
      } else if (WI(i) > zero) {
        scl =
            one / Lapy2(Nrm2(n, VR.ptr(1, i), 1), Nrm2(n, VR.ptr(1, i + 1), 1));
        Rscal(n, scl, VR.ptr(1, i), 1);
        Rscal(n, scl, VR.ptr(1, i + 1), 1);
        for (k = 1; k <= n; ++k) {
          WORK(k) = (VR(k, i) * VR(k, i)) + (VR(k, i + 1) * VR(k, i + 1));
        }
        k = Iamax(n, work, 1);
        Lartg(VR(k, i), VR(k, i + 1), &cs, &sn, &r);
        Rot(n, VR.ptr(1, i), 1, VR.ptr(1, i + 1), 1, cs, sn);
        VR(k, i + 1) = zero;
      }
    }
  }
  // Undo scaling if necessary
L50:
  if (scalea) {
    Lascl('G', 0, 0, cscale, anrm, n - info, 1, WR.ptr(info + 1),
          Max<blasint>(n - info, 1));
    Lascl('G', 0, 0, cscale, anrm, n - info, 1, WI.ptr(info + 1),
          Max<blasint>(n - info, 1));
    if (info == 0) {
      if ((wntsnv || wntsnb) && icond == 0)
        Lascl('G', 0, 0, cscale, anrm, n, 1, rcondv, n);
    } else {
      Lascl('G', 0, 0, cscale, anrm, *ilo - 1, 1, wr, n);
      Lascl('G', 0, 0, cscale, anrm, *ilo - 1, 1, wi, n);
    }
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// xGEEVX (complex T): as the real routine, with the eigenvalues W and one
// column of VL / VR per eigenvector. WORK has LWORK >= max(1, 2*N)
// elements, N*N+2*N for SENSE = 'V' or 'B'; RWORK has 2*N.
template <typename T>
blasint Geevx(char balanc, char jobvl, char jobvr, char sense, blasint n, T* a,
              blasint lda, T* w, T* vl, blasint ldvl, T* vr, blasint ldvr,
              blasint* ilo, blasint* ihi, RealT<T>* scale, RealT<T>* abnrm,
              RealT<T>* rconde, RealT<T>* rcondv, T* work, blasint lwork,
              RealT<T>* rwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const R one = R(1);
  R dum[1];
  const Vector<R> DUM(dum);
  const Vector<R> RWORK(rwork);
  blasint select[1];
  const Matrix<T> VL(vl, ldvl);
  const Matrix<T> VR(vr, ldvr);
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  blasint hswork, i, icond, itau, iwrk, k, lwork_trevc, maxwrk, minwrk, nout;
  bool lquery, scalea, wantvl, wantvr, wntsnb, wntsne, wntsnn, wntsnv;
  R anrm, bignum, cscale, eps, scl, smlnum;
  T tmp;
  char job;
  // Set wherever it is read (eigenvectors are wanted); initialized for the
  // compiler.
  char side = 'N';
  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  wantvl = Lsame(jobvl, 'V');
  wantvr = Lsame(jobvr, 'V');
  wntsnn = Lsame(sense, 'N');
  wntsne = Lsame(sense, 'E');
  wntsnv = Lsame(sense, 'V');
  wntsnb = Lsame(sense, 'B');
  if (!(Lsame(balanc, 'N') || Lsame(balanc, 'S') || Lsame(balanc, 'P') ||
        Lsame(balanc, 'B'))) {
    info = -1;
  } else if ((!wantvl) && (!Lsame(jobvl, 'N'))) {
    info = -2;
  } else if ((!wantvr) && (!Lsame(jobvr, 'N'))) {
    info = -3;
  } else if (!(wntsnn || wntsne || wntsnb || wntsnv) ||
             ((wntsne || wntsnb) && !(wantvl && wantvr))) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldvl < 1 || (wantvl && ldvl < n)) {
    info = -10;
  } else if (ldvr < 1 || (wantvr && ldvr < n)) {
    info = -12;
  }
  // Compute workspace
  //  (Note: Comments in the code beginning "Workspace:" describe the
  //   minimal amount of workspace needed at that point in the code,
  //   as well as the preferred amount for good performance.
  //   CWorkspace refers to complex workspace, and RWorkspace to real
  //   workspace. NB refers to the optimal block size for the
  //   immediately following subroutine, as returned by ILAENV.
  //   HSWORK refers to the workspace preferred by ZHSEQR, as
  //   calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //   the worst case.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      if (wantvl) {
        Trevc3('L', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1, rwork, -1);
        lwork_trevc = static_cast<blasint>(std::real(WORK(1)));
        maxwrk = Max<blasint>(maxwrk, lwork_trevc);
        info = Hseqr('S', 'V', n, 1, n, a, lda, w, vl, ldvl, work, -1);
      } else if (wantvr) {
        Trevc3('R', 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout, work,
               -1, rwork, -1);
        lwork_trevc = static_cast<blasint>(std::real(WORK(1)));
        maxwrk = Max<blasint>(maxwrk, lwork_trevc);
        info = Hseqr('S', 'V', n, 1, n, a, lda, w, vr, ldvr, work, -1);
      } else {
        if (wntsnn) {
          info = Hseqr('E', 'N', n, 1, n, a, lda, w, vr, ldvr, work, -1);
        } else {
          info = Hseqr('S', 'N', n, 1, n, a, lda, w, vr, ldvr, work, -1);
        }
      }
      hswork = static_cast<blasint>(std::real(WORK(1)));
      if ((!wantvl) && (!wantvr)) {
        minwrk = 2 * n;
        if (!(wntsnn || wntsne)) minwrk = Max<blasint>(minwrk, n * n + 2 * n);
        maxwrk = Max<blasint>(maxwrk, hswork);
        if (!(wntsnn || wntsne)) maxwrk = Max<blasint>(maxwrk, n * n + 2 * n);
      } else {
        minwrk = 2 * n;
        if (!(wntsnn || wntsne)) minwrk = Max<blasint>(minwrk, n * n + 2 * n);
        maxwrk = Max<blasint>(maxwrk, hswork);
        maxwrk = Max<blasint>(
            maxwrk, n + (n - 1) * Ilaenv<T>(1, "UNGHR", " ", n, 1, n, -1));
        if (!(wntsnn || wntsne)) maxwrk = Max<blasint>(maxwrk, n * n + 2 * n);
        maxwrk = Max<blasint>(maxwrk, 2 * n);
      }
      maxwrk = Max<blasint>(maxwrk, minwrk);
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -20;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEEVX", -info);
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
  icond = 0;
  anrm = Lange('M', n, n, a, lda, dum);
  scalea = false;
  if (anrm > zero && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) Lascl('G', 0, 0, anrm, cscale, n, n, a, lda);
  // Balance the matrix and compute ABNRM
  Gebal(balanc, n, a, lda, ilo, ihi, scale);
  *abnrm = Lange('1', n, n, a, lda, dum);
  if (scalea) {
    DUM(1) = *abnrm;
    Lascl<R>('G', 0, 0, cscale, anrm, 1, 1, dum, 1);
    *abnrm = DUM(1);
  }
  // Reduce to upper Hessenberg form
  // (CWorkspace: need 2*N, prefer N+N*NB)
  // (RWorkspace: none)
  itau = 1;
  iwrk = itau + n;
  Gehrd(n, *ilo, *ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk),
        lwork - iwrk + 1);
  if (wantvl) {
    // Want left eigenvectors
    // Copy Householder vectors to VL
    side = 'L';
    Lacpy('L', n, n, a, lda, vl, ldvl);
    // Generate unitary matrix in VL
    // (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
    // (RWorkspace: none)
    Unghr(n, *ilo, *ihi, vl, ldvl, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VL
    // (CWorkspace: need 1, prefer HSWORK (see comments) )
    // (RWorkspace: none)
    iwrk = itau;
    info = Hseqr('S', 'V', n, *ilo, *ihi, a, lda, w, vl, ldvl, WORK.ptr(iwrk),
                 lwork - iwrk + 1);
    if (wantvr) {
      // Want left and right eigenvectors
      // Copy Schur vectors to VR
      side = 'B';
      Lacpy('F', n, n, vl, ldvl, vr, ldvr);
    }
  } else if (wantvr) {
    // Want right eigenvectors
    // Copy Householder vectors to VR
    side = 'R';
    Lacpy('L', n, n, a, lda, vr, ldvr);
    // Generate unitary matrix in VR
    // (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
    // (RWorkspace: none)
    Unghr(n, *ilo, *ihi, vr, ldvr, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
    // Perform QR iteration, accumulating Schur vectors in VR
    // (CWorkspace: need 1, prefer HSWORK (see comments) )
    // (RWorkspace: none)
    iwrk = itau;
    info = Hseqr('S', 'V', n, *ilo, *ihi, a, lda, w, vr, ldvr, WORK.ptr(iwrk),
                 lwork - iwrk + 1);
  } else {
    // Compute eigenvalues only
    // If condition numbers desired, compute Schur form
    if (wntsnn) {
      job = 'E';
    } else {
      job = 'S';
    }
    // (CWorkspace: need 1, prefer HSWORK (see comments) )
    // (RWorkspace: none)
    iwrk = itau;
    info = Hseqr(job, 'N', n, *ilo, *ihi, a, lda, w, vr, ldvr, WORK.ptr(iwrk),
                 lwork - iwrk + 1);
  }
  // If INFO .NE. 0 from ZHSEQR, then quit
  if (info != 0) goto L50;
  if (wantvl || wantvr) {
    // Compute left and/or right eigenvectors
    // (CWorkspace: need 2*N, prefer N + 2*N*NB)
    // (RWorkspace: need N)
    Trevc3(side, 'B', select, n, a, lda, vl, ldvl, vr, ldvr, n, &nout,
           WORK.ptr(iwrk), lwork - iwrk + 1, rwork, n);
  }
  // Compute condition numbers if desired
  // (CWorkspace: need N*N+2*N unless SENSE = 'E')
  // (RWorkspace: need 2*N unless SENSE = 'E')
  if (!wntsnn) {
    icond = Trsna(sense, 'A', select, n, a, lda, vl, ldvl, vr, ldvr, rconde,
                  rcondv, n, &nout, WORK.ptr(iwrk), n, rwork);
  }
  if (wantvl) {
    // Undo balancing of left eigenvectors
    Gebak(balanc, 'L', n, *ilo, *ihi, scale, n, vl, ldvl);
    // Normalize left eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      scl = one / Nrm2(n, VL.ptr(1, i), 1);
      Rscal(n, scl, VL.ptr(1, i), 1);
      for (k = 1; k <= n; ++k) {
        RWORK(k) = (std::real(VL(k, i)) * std::real(VL(k, i))) +
                   (std::imag(VL(k, i)) * std::imag(VL(k, i)));
      }
      k = Iamax(n, rwork, 1);
      tmp = Conj(VL(k, i)) / std::sqrt(RWORK(k));
      Scal(n, tmp, VL.ptr(1, i), 1);
      VL(k, i) = T(std::real(VL(k, i)), zero);
    }
  }
  if (wantvr) {
    // Undo balancing of right eigenvectors
    Gebak(balanc, 'R', n, *ilo, *ihi, scale, n, vr, ldvr);
    // Normalize right eigenvectors and make largest component real
    for (i = 1; i <= n; ++i) {
      scl = one / Nrm2(n, VR.ptr(1, i), 1);
      Rscal(n, scl, VR.ptr(1, i), 1);
      for (k = 1; k <= n; ++k) {
        RWORK(k) = (std::real(VR(k, i)) * std::real(VR(k, i))) +
                   (std::imag(VR(k, i)) * std::imag(VR(k, i)));
      }
      k = Iamax(n, rwork, 1);
      tmp = Conj(VR(k, i)) / std::sqrt(RWORK(k));
      Scal(n, tmp, VR.ptr(1, i), 1);
      VR(k, i) = T(std::real(VR(k, i)), zero);
    }
  }
  // Undo scaling if necessary
L50:
  if (scalea) {
    Lascl('G', 0, 0, cscale, anrm, n - info, 1, W.ptr(info + 1),
          Max<blasint>(n - info, 1));
    if (info == 0) {
      if ((wntsnv || wntsnb) && icond == 0)
        Lascl<R>('G', 0, 0, cscale, anrm, n, 1, rcondv, n);
    } else {
      Lascl('G', 0, 0, cscale, anrm, *ilo - 1, 1, w, n);
    }
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

// xGEESX (real T): as xGEES (the Schur factorization with optional
// ordering), and for SORT = 'S' the reciprocal condition numbers of the
// average of the selected eigenvalues (SENSE = 'E', RCONDE) and/or of
// their right invariant subspace ('V', RCONDV; 'B': both; 'N': none;
// anything but 'N' needs SORT = 'S'). WORK has LWORK >= max(1, 3*N)
// elements, N+2*SDIM*(N-SDIM) for SENSE = 'E', 'V' or 'B' (the estimate
// N+N*N/2 suffices); IWORK has LIWORK >= 1 elements, SDIM*(N-SDIM) for
// SENSE = 'V' or 'B' (LWORK = -1 or LIWORK = -1: workspace query; WORK(1)
// and IWORK(1) return the sizes, the ones that SDIM needs after a
// factorization).
//
// Returns INFO as xGEES, or -16 / -18 if LWORK / LIWORK is too small for
// SDIM (the factorization is then complete but not the condition numbers).
template <typename R>
blasint Geesx(char jobvs, char sort, GeesSelectReal<R> select, char sense,
              blasint n, R* a, blasint lda, blasint* sdim, R* wr, R* wi, R* vs,
              blasint ldvs, R* rconde, R* rcondv, R* work, blasint lwork,
              blasint* iwork, blasint liwork, blasint* bwork) {
  using T = R;
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Vector<blasint> BWORK(bwork);
  R dum[1];
  const Vector<R> DUM(dum);
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> VS(vs, ldvs);
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  blasint hswork, i, i1, i2, ibal, icond, ieval, ihi, ilo, inxt, ip, itau, iwrk,
      liwrk, lwrk, minwrk;
  // Set for N > 0, the only case in which it is read; initialized for the
  // compiler.
  blasint maxwrk = 1;
  bool cursl, lastsl, lquery, lst2sl, scalea, wantsb, wantse, wantsn, wantst,
      wantsv, wantvs;
  R anrm, bignum, cscale, eps, smlnum;
  // Test the input arguments
  blasint info = 0;
  wantvs = Lsame(jobvs, 'V');
  wantst = Lsame(sort, 'S');
  wantsn = Lsame(sense, 'N');
  wantse = Lsame(sense, 'E');
  wantsv = Lsame(sense, 'V');
  wantsb = Lsame(sense, 'B');
  lquery = (lwork == -1 || liwork == -1);
  if ((!wantvs) && (!Lsame(jobvs, 'N'))) {
    info = -1;
  } else if ((!wantst) && (!Lsame(sort, 'N'))) {
    info = -2;
  } else if (!(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn)) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldvs < 1 || (wantvs && ldvs < n)) {
    info = -12;
  }
  // Compute workspace
  //  (Note: Comments in the code beginning "RWorkspace:" describe the
  //   minimal amount of real workspace needed at that point in the
  //   code, as well as the preferred amount for good performance.
  //   IWorkspace refers to integer workspace.
  //   NB refers to the optimal block size for the immediately
  //   following subroutine, as returned by ILAENV.
  //   HSWORK refers to the workspace preferred by DHSEQR, as
  //   calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //   the worst case.
  //   If SENSE = 'E', 'V' or 'B', then the amount of workspace needed
  //   depends on SDIM, which is computed by the routine DTRSEN later
  //   in the code.)
  if (info == 0) {
    liwrk = 1;
    if (n == 0) {
      minwrk = 1;
      lwrk = 1;
    } else {
      maxwrk = 2 * n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      minwrk = 3 * n;
      ieval = Hseqr('S', jobvs, n, 1, n, a, lda, wr, wi, vs, ldvs, work, -1);
      hswork = static_cast<blasint>(WORK(1));
      if (!wantvs) {
        maxwrk = Max<blasint>(maxwrk, n + hswork);
      } else {
        maxwrk = Max<blasint>(
            maxwrk, 2 * n + (n - 1) * Ilaenv<T>(1, "ORGHR", " ", n, 1, n, -1));
        maxwrk = Max<blasint>(maxwrk, n + hswork);
      }
      lwrk = maxwrk;
      if (!wantsn) lwrk = Max<blasint>(lwrk, n + (n * n) / 2);
      if (wantsv || wantsb) liwrk = (n * n) / 4;
    }
    IWORK(1) = liwrk;
    WORK(1) = WorkSize<T>(lwrk);
    if (lwork < minwrk && !lquery) {
      info = -16;
    } else if (liwork < 1 && !lquery) {
      info = -18;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEESX", -info);
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
  anrm = Lange('M', n, n, a, lda, dum);
  scalea = false;
  if (anrm > zero && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) Lascl('G', 0, 0, anrm, cscale, n, n, a, lda);
  // Permute the matrix to make it more nearly triangular
  // (RWorkspace: need N)
  ibal = 1;
  Gebal('P', n, a, lda, &ilo, &ihi, WORK.ptr(ibal));
  // Reduce to upper Hessenberg form
  // (RWorkspace: need 3*N, prefer 2*N+N*NB)
  itau = n + ibal;
  iwrk = n + itau;
  Gehrd(n, ilo, ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk), lwork - iwrk + 1);
  if (wantvs) {
    // Copy Householder vectors to VS
    Lacpy('L', n, n, a, lda, vs, ldvs);
    // Generate orthogonal matrix in VS
    // (RWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    Unghr(n, ilo, ihi, vs, ldvs, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
  }
  *sdim = 0;
  // Perform QR iteration, accumulating Schur vectors in VS if desired
  // (RWorkspace: need N+1, prefer N+HSWORK (see comments) )
  iwrk = itau;
  ieval = Hseqr('S', jobvs, n, ilo, ihi, a, lda, wr, wi, vs, ldvs,
                WORK.ptr(iwrk), lwork - iwrk + 1);
  if (ieval > 0) info = ieval;
  // Sort eigenvalues if desired
  if (wantst && info == 0) {
    if (scalea) {
      Lascl('G', 0, 0, cscale, anrm, n, 1, wr, n);
      Lascl('G', 0, 0, cscale, anrm, n, 1, wi, n);
    }
    for (i = 1; i <= n; ++i) {
      BWORK(i) = select(WR.ptr(i), WI.ptr(i));
    }
    // Reorder eigenvalues, transform Schur vectors, and compute
    // reciprocal condition numbers
    // (RWorkspace: if SENSE is not 'N', need N+2*SDIM*(N-SDIM)
    //              otherwise, need N )
    // (IWorkspace: if SENSE is 'V' or 'B', need SDIM*(N-SDIM)
    //              otherwise, need 0 )
    icond =
        Trsen(sense, jobvs, bwork, n, a, lda, vs, ldvs, wr, wi, sdim, rconde,
              rcondv, WORK.ptr(iwrk), lwork - iwrk + 1, iwork, liwork);
    if (!wantsn) maxwrk = Max<blasint>(maxwrk, n + 2 * *sdim * (n - *sdim));
    if (icond == -15) {
      // Not enough real workspace
      info = -16;
    } else if (icond == -17) {
      // Not enough integer workspace
      info = -18;
    } else if (icond > 0) {
      // DTRSEN failed to reorder or to restore standard Schur form
      info = icond + n;
    }
  }
  if (wantvs) {
    // Undo balancing
    // (RWorkspace: need N)
    Gebak('P', 'R', n, ilo, ihi, WORK.ptr(ibal), n, vs, ldvs);
  }
  if (scalea) {
    // Undo scaling for the Schur form of A
    Lascl('H', 0, 0, cscale, anrm, n, n, a, lda);
    Copy(n, a, lda + 1, wr, 1);
    if ((wantsv || wantsb) && info == 0) {
      DUM(1) = *rcondv;
      Lascl('G', 0, 0, cscale, anrm, 1, 1, dum, 1);
      *rcondv = DUM(1);
    }
    if (cscale == smlnum) {
      // If scaling back towards underflow, adjust WI if an
      // offdiagonal element of a 2-by-2 block in the Schur form
      // underflows.
      if (ieval > 0) {
        i1 = ieval + 1;
        i2 = ihi - 1;
        Lascl('G', 0, 0, cscale, anrm, ilo - 1, 1, wi, n);
      } else if (wantst) {
        i1 = 1;
        i2 = n - 1;
      } else {
        i1 = ilo;
        i2 = ihi - 1;
      }
      inxt = i1 - 1;
      for (i = i1; i <= i2; ++i) {
        if (i < inxt) continue;
        if (WI(i) == zero) {
          inxt = i + 1;
        } else {
          if (A(i + 1, i) == zero) {
            WI(i) = zero;
            WI(i + 1) = zero;
          } else if (A(i + 1, i) != zero && A(i, i + 1) == zero) {
            WI(i) = zero;
            WI(i + 1) = zero;
            if (i > 1) Swap(i - 1, A.ptr(1, i), 1, A.ptr(1, i + 1), 1);
            if (n > i + 1)
              Swap(n - i - 1, A.ptr(i, i + 2), lda, A.ptr(i + 1, i + 2), lda);
            if (wantvs) {
              Swap(n, VS.ptr(1, i), 1, VS.ptr(1, i + 1), 1);
            }
            A(i, i + 1) = A(i + 1, i);
            A(i + 1, i) = zero;
          }
          inxt = i + 2;
        }
      }
    }
    Lascl('G', 0, 0, cscale, anrm, n - ieval, 1, WI.ptr(ieval + 1),
          Max<blasint>(n - ieval, 1));
  }
  if (wantst && info == 0) {
    // Check if reordering successful
    lastsl = true;
    lst2sl = true;
    *sdim = 0;
    ip = 0;
    for (i = 1; i <= n; ++i) {
      cursl = select(WR.ptr(i), WI.ptr(i)) != 0;
      if (WI(i) == zero) {
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
  WORK(1) = WorkSize<T>(maxwrk);
  if (wantsv || wantsb) {
    IWORK(1) = Max<blasint>(1, *sdim * (n - *sdim));
  } else {
    IWORK(1) = 1;
  }
  return info;
}

// xGEESX (complex T): as the real routine, with the eigenvalues W and the
// upper triangular Schur form; SELECT(W(j)) selects. WORK has LWORK >=
// max(1, 2*N) elements, 2*SDIM*(N-SDIM) for SENSE = 'E', 'V' or 'B' (the
// estimate N*N/2 suffices); RWORK has N. Returns -15 if LWORK is too small
// for SDIM; INFO = N+2 does not occur.
template <typename T>
blasint Geesx(char jobvs, char sort, GeesSelectComplex<RealT<T>> select,
              char sense, blasint n, T* a, blasint lda, blasint* sdim, T* w,
              T* vs, blasint ldvs, RealT<T>* rconde, RealT<T>* rcondv, T* work,
              blasint lwork, RealT<T>* rwork, blasint* bwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const R one = R(1);
  const Vector<blasint> BWORK(bwork);
  R dum[1];
  const Vector<R> DUM(dum);
  const Vector<R> RWORK(rwork);
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  blasint hswork, i, ibal, icond, ieval, ihi, ilo, itau, iwrk, lwrk, minwrk;
  // Set for N > 0, the only case in which it is read; initialized for the
  // compiler.
  blasint maxwrk = 1;
  bool lquery, scalea, wantsb, wantse, wantsn, wantst, wantsv, wantvs;
  R anrm, bignum, cscale, eps, smlnum;
  // Test the input arguments
  blasint info = 0;
  wantvs = Lsame(jobvs, 'V');
  wantst = Lsame(sort, 'S');
  wantsn = Lsame(sense, 'N');
  wantse = Lsame(sense, 'E');
  wantsv = Lsame(sense, 'V');
  wantsb = Lsame(sense, 'B');
  lquery = (lwork == -1);
  if ((!wantvs) && (!Lsame(jobvs, 'N'))) {
    info = -1;
  } else if ((!wantst) && (!Lsame(sort, 'N'))) {
    info = -2;
  } else if (!(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn)) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldvs < 1 || (wantvs && ldvs < n)) {
    info = -11;
  }
  // Compute workspace
  //  (Note: Comments in the code beginning "Workspace:" describe the
  //   minimal amount of real workspace needed at that point in the
  //   code, as well as the preferred amount for good performance.
  //   CWorkspace refers to complex workspace, and RWorkspace to real
  //   workspace. NB refers to the optimal block size for the
  //   immediately following subroutine, as returned by ILAENV.
  //   HSWORK refers to the workspace preferred by ZHSEQR, as
  //   calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  //   the worst case.
  //   If SENSE = 'E', 'V' or 'B', then the amount of workspace needed
  //   depends on SDIM, which is computed by the routine ZTRSEN later
  //   in the code.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      lwrk = 1;
    } else {
      maxwrk = n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      minwrk = 2 * n;
      ieval = Hseqr('S', jobvs, n, 1, n, a, lda, w, vs, ldvs, work, -1);
      hswork = static_cast<blasint>(std::real(WORK(1)));
      if (!wantvs) {
        maxwrk = Max<blasint>(maxwrk, hswork);
      } else {
        maxwrk = Max<blasint>(
            maxwrk, n + (n - 1) * Ilaenv<T>(1, "UNGHR", " ", n, 1, n, -1));
        maxwrk = Max<blasint>(maxwrk, hswork);
      }
      lwrk = maxwrk;
      if (!wantsn) lwrk = Max<blasint>(lwrk, (n * n) / 2);
    }
    WORK(1) = WorkSize<T>(lwrk);
    if (lwork < minwrk && !lquery) {
      info = -15;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEESX", -info);
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
  anrm = Lange('M', n, n, a, lda, dum);
  scalea = false;
  if (anrm > zero && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) Lascl('G', 0, 0, anrm, cscale, n, n, a, lda);
  // Permute the matrix to make it more nearly triangular
  // (CWorkspace: none)
  // (RWorkspace: need N)
  ibal = 1;
  Gebal('P', n, a, lda, &ilo, &ihi, RWORK.ptr(ibal));
  // Reduce to upper Hessenberg form
  // (CWorkspace: need 2*N, prefer N+N*NB)
  // (RWorkspace: none)
  itau = 1;
  iwrk = n + itau;
  Gehrd(n, ilo, ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk), lwork - iwrk + 1);
  if (wantvs) {
    // Copy Householder vectors to VS
    Lacpy('L', n, n, a, lda, vs, ldvs);
    // Generate unitary matrix in VS
    // (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
    // (RWorkspace: none)
    Unghr(n, ilo, ihi, vs, ldvs, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
  }
  *sdim = 0;
  // Perform QR iteration, accumulating Schur vectors in VS if desired
  // (CWorkspace: need 1, prefer HSWORK (see comments) )
  // (RWorkspace: none)
  iwrk = itau;
  ieval = Hseqr('S', jobvs, n, ilo, ihi, a, lda, w, vs, ldvs, WORK.ptr(iwrk),
                lwork - iwrk + 1);
  if (ieval > 0) info = ieval;
  // Sort eigenvalues if desired
  if (wantst && info == 0) {
    if (scalea) Lascl('G', 0, 0, cscale, anrm, n, 1, w, n);
    for (i = 1; i <= n; ++i) {
      BWORK(i) = select(reinterpret_cast<const R*>(W.ptr(i)));
    }
    // Reorder eigenvalues, transform Schur vectors, and compute
    // reciprocal condition numbers
    // (CWorkspace: if SENSE is not 'N', need 2*SDIM*(N-SDIM)
    //              otherwise, need none )
    // (RWorkspace: none)
    icond = Trsen(sense, jobvs, bwork, n, a, lda, vs, ldvs, w, sdim, rconde,
                  rcondv, WORK.ptr(iwrk), lwork - iwrk + 1);
    if (!wantsn) maxwrk = Max<blasint>(maxwrk, 2 * *sdim * (n - *sdim));
    if (icond == -14) {
      // Not enough complex workspace
      info = -15;
    }
  }
  if (wantvs) {
    // Undo balancing
    // (CWorkspace: none)
    // (RWorkspace: need N)
    Gebak('P', 'R', n, ilo, ihi, RWORK.ptr(ibal), n, vs, ldvs);
  }
  if (scalea) {
    // Undo scaling for the Schur form of A
    Lascl('U', 0, 0, cscale, anrm, n, n, a, lda);
    Copy(n, a, lda + 1, w, 1);
    if ((wantsv || wantsb) && info == 0) {
      DUM(1) = *rcondv;
      Lascl<R>('G', 0, 0, cscale, anrm, 1, 1, dum, 1);
      *rcondv = DUM(1);
    }
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                    \
  template blasint Geev<R>(char, char, blasint, R*, blasint, R*, R*, R*,      \
                           blasint, R*, blasint, R*, blasint);                \
  template blasint Geevx<R>(char, char, char, char, blasint, R*, blasint, R*, \
                            R*, R*, blasint, R*, blasint, blasint*, blasint*, \
                            R*, R*, R*, R*, R*, blasint, blasint*);           \
  template blasint Geesx<R>(char, char, GeesSelectReal<R>, char, blasint, R*, \
                            blasint, blasint*, R*, R*, R*, blasint, R*, R*,   \
                            R*, blasint, blasint*, blasint, blasint*);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                 \
  template blasint Geev<T>(char, char, blasint, T*, blasint, T*, T*, blasint, \
                           T*, blasint, T*, blasint, RealT<T>*);              \
  template blasint Geevx<T>(char, char, char, char, blasint, T*, blasint, T*, \
                            T*, blasint, T*, blasint, blasint*, blasint*,     \
                            RealT<T>*, RealT<T>*, RealT<T>*, RealT<T>*, T*,   \
                            blasint, RealT<T>*);                              \
  template blasint Geesx<T>(char, char, GeesSelectComplex<RealT<T>>, char,    \
                            blasint, T*, blasint, blasint*, T*, T*, blasint,  \
                            RealT<T>*, RealT<T>*, T*, blasint, RealT<T>*,     \
                            blasint*);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GEEV_REAL(prefix, R)                                  \
  HWYBLAS_API void prefix##geev_(                                            \
      const char* jobvl, const char* jobvr, const hwyblas_int* n, R* a,      \
      const hwyblas_int* lda, R* wr, R* wi, R* vl, const hwyblas_int* ldvl,  \
      R* vr, const hwyblas_int* ldvr, R* work, const hwyblas_int* lwork,     \
      hwyblas_int* info) {                                                   \
    *info = hwyblas::lapack::Geev(*jobvl, *jobvr, *n, a, *lda, wr, wi, vl,   \
                                  *ldvl, vr, *ldvr, work, *lwork);           \
  }                                                                          \
  HWYBLAS_API void prefix##geevx_(                                           \
      const char* balanc, const char* jobvl, const char* jobvr,              \
      const char* sense, const hwyblas_int* n, R* a, const hwyblas_int* lda, \
      R* wr, R* wi, R* vl, const hwyblas_int* ldvl, R* vr,                   \
      const hwyblas_int* ldvr, hwyblas_int* ilo, hwyblas_int* ihi, R* scale, \
      R* abnrm, R* rconde, R* rcondv, R* work, const hwyblas_int* lwork,     \
      hwyblas_int* iwork, hwyblas_int* info) {                               \
    *info = hwyblas::lapack::Geevx(                                          \
        *balanc, *jobvl, *jobvr, *sense, *n, a, *lda, wr, wi, vl, *ldvl, vr, \
        *ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, *lwork, iwork); \
  }

HWYBLAS_LAPACK_GEEV_REAL(s, float)
HWYBLAS_LAPACK_GEEV_REAL(d, double)

#undef HWYBLAS_LAPACK_GEEV_REAL

#define HWYBLAS_LAPACK_GEEV_COMPLEX(prefix, T, R)                            \
  HWYBLAS_API void prefix##geev_(                                            \
      const char* jobvl, const char* jobvr, const hwyblas_int* n, R* a,      \
      const hwyblas_int* lda, R* w, R* vl, const hwyblas_int* ldvl, R* vr,   \
      const hwyblas_int* ldvr, R* work, const hwyblas_int* lwork, R* rwork,  \
      hwyblas_int* info) {                                                   \
    *info = hwyblas::lapack::Geev(*jobvl, *jobvr, *n, As<T>(a), *lda,        \
                                  As<T>(w), As<T>(vl), *ldvl, As<T>(vr),     \
                                  *ldvr, As<T>(work), *lwork, rwork);        \
  }                                                                          \
  HWYBLAS_API void prefix##geevx_(                                           \
      const char* balanc, const char* jobvl, const char* jobvr,              \
      const char* sense, const hwyblas_int* n, R* a, const hwyblas_int* lda, \
      R* w, R* vl, const hwyblas_int* ldvl, R* vr, const hwyblas_int* ldvr,  \
      hwyblas_int* ilo, hwyblas_int* ihi, R* scale, R* abnrm, R* rconde,     \
      R* rcondv, R* work, const hwyblas_int* lwork, R* rwork,                \
      hwyblas_int* info) {                                                   \
    *info = hwyblas::lapack::Geevx(                                          \
        *balanc, *jobvl, *jobvr, *sense, *n, As<T>(a), *lda, As<T>(w),       \
        As<T>(vl), *ldvl, As<T>(vr), *ldvr, ilo, ihi, scale, abnrm, rconde,  \
        rcondv, As<T>(work), *lwork, rwork);                                 \
  }

HWYBLAS_LAPACK_GEEV_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GEEV_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GEEV_COMPLEX

// SELECT is a Fortran LOGICAL FUNCTION (nonzero is true) of WR and WI, or
// of W (a pointer to its real and imaginary part); BWORK is a LOGICAL
// array.
HWYBLAS_API void sgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select2_float select, const char* sense,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* sdim, float* wr, float* wi, float* vs,
                         const hwyblas_int* ldvs, float* rconde, float* rcondv,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Geesx(*jobvs, *sort, select, *sense, *n, a, *lda,
                                 sdim, wr, wi, vs, *ldvs, rconde, rcondv, work,
                                 *lwork, iwork, *liwork, bwork);
}
HWYBLAS_API void dgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select2_double select, const char* sense,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* sdim, double* wr,
                         double* wi, double* vs, const hwyblas_int* ldvs,
                         double* rconde, double* rcondv, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* bwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Geesx(*jobvs, *sort, select, *sense, *n, a, *lda,
                                 sdim, wr, wi, vs, *ldvs, rconde, rcondv, work,
                                 *lwork, iwork, *liwork, bwork);
}
HWYBLAS_API void cgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select1_float select, const char* sense,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* sdim, float* w, float* vs,
                         const hwyblas_int* ldvs, float* rconde, float* rcondv,
                         float* work, const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Geesx(
      *jobvs, *sort, select, *sense, *n, As<c32>(a), *lda, sdim, As<c32>(w),
      As<c32>(vs), *ldvs, rconde, rcondv, As<c32>(work), *lwork, rwork, bwork);
}
HWYBLAS_API void zgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select1_double select, const char* sense,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* sdim, double* w,
                         double* vs, const hwyblas_int* ldvs, double* rconde,
                         double* rcondv, double* work, const hwyblas_int* lwork,
                         double* rwork, hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Geesx(
      *jobvs, *sort, select, *sense, *n, As<c64>(a), *lda, sdim, As<c64>(w),
      As<c64>(vs), *ldvs, rconde, rcondv, As<c64>(work), *lwork, rwork, bwork);
}

}  // extern "C"
