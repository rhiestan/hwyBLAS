// The Schur factorization of a general matrix, A = Z*T*Z**H, with optional
// ordering of the eigenvalues: xGEES.
//
// The real and the complex routine differ in the Schur form (2-by-2 blocks
// for complex conjugate pairs in the real one), the eigenvalues (WR and WI
// for W) and the workspace, so they are ported as two overloads.
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

// xGEES (real T): computes the eigenvalues WR + i*WI, the real Schur form T
// (in A; upper quasi-triangular, with 2-by-2 diagonal blocks in standard
// form for the complex conjugate eigenvalue pairs) and, for JOBVS = 'V',
// the Schur vectors Z (in VS; JOBVS = 'N': none) of the N-by-N matrix A.
// For SORT = 'S', the eigenvalues for which SELECT(WR(j), WI(j)) is true
// (a complex conjugate pair if either is) are moved to the top left of T;
// SDIM is their number (SORT = 'N': no ordering, SDIM = 0). WORK has
// LWORK >= max(1, 3*N) elements (LWORK = -1: workspace query); BWORK has N
// elements and is used for SORT = 'S'.
//
// Returns INFO: 0, -i for an illegal argument i, i in 1:N if the QR
// algorithm failed (WR and WI hold eigenvalues i+1:N), N+1 if the
// eigenvalues could not be reordered (too close to swap), or N+2 if after
// reordering, rounding changed some complex eigenvalues so that the
// leading eigenvalues no longer all satisfy SELECT.
template <typename R>
blasint Gees(char jobvs, char sort, GeesSelectReal<R> select, blasint n, R* a,
             blasint lda, blasint* sdim, R* wr, R* wi, R* vs, blasint ldvs,
             R* work, blasint lwork, blasint* bwork) {
  using T = R;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Vector<blasint> BWORK(bwork);
  R dum[1];
  blasint idum[1];
  const Matrix<R> VS(vs, ldvs);
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  blasint hswork, i, i1, i2, ibal, icond, ieval, ihi, ilo, inxt, ip, itau, iwrk,
      maxwrk, minwrk;
  bool cursl, lastsl, lquery, lst2sl, scalea, wantst, wantvs;
  R anrm, bignum, cscale, eps, s, sep, smlnum;
  blasint info = 0;
  lquery = (lwork == -1);
  wantvs = Lsame(jobvs, 'V');
  wantst = Lsame(sort, 'S');
  if ((!wantvs) && (!Lsame(jobvs, 'N'))) {
    info = -1;
  } else if ((!wantst) && (!Lsame(sort, 'N'))) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (lda < std::max<blasint>(1, n)) {
    info = -6;
  } else if (ldvs < 1 || (wantvs && ldvs < n)) {
    info = -11;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV.
  // HSWORK refers to the workspace preferred by DHSEQR, as
  // calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  // the worst case.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = 2 * n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      minwrk = 3 * n;
      ieval = Hseqr('S', jobvs, n, 1, n, a, lda, wr, wi, vs, ldvs, work, -1);
      hswork = static_cast<blasint>(WORK(1));
      if (!wantvs) {
        maxwrk = std::max<blasint>(maxwrk, n + hswork);
      } else {
        maxwrk = std::max<blasint>(
            maxwrk, 2 * n + (n - 1) * Ilaenv<T>(1, "ORGHR", " ", n, 1, n, -1));
        maxwrk = std::max<blasint>(maxwrk, n + hswork);
      }
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -13;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEES", -info);
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
  // (Workspace: need N)
  ibal = 1;
  Gebal('P', n, a, lda, &ilo, &ihi, WORK.ptr(ibal));
  // Reduce to upper Hessenberg form
  // (Workspace: need 3*N, prefer 2*N+N*NB)
  itau = n + ibal;
  iwrk = n + itau;
  Gehrd(n, ilo, ihi, a, lda, WORK.ptr(itau), WORK.ptr(iwrk), lwork - iwrk + 1);
  if (wantvs) {
    // Copy Householder vectors to VS
    Lacpy('L', n, n, a, lda, vs, ldvs);
    // Generate orthogonal matrix in VS
    // (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
    Unghr(n, ilo, ihi, vs, ldvs, WORK.ptr(itau), WORK.ptr(iwrk),
          lwork - iwrk + 1);
  }
  *sdim = 0;
  // Perform QR iteration, accumulating Schur vectors in VS if desired
  // (Workspace: need N+1, prefer N+HSWORK (see comments) )
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
    // Reorder eigenvalues and transform Schur vectors
    // (Workspace: none needed)
    icond = Trsen('N', jobvs, bwork, n, a, lda, vs, ldvs, wr, wi, sdim, &s,
                  &sep, WORK.ptr(iwrk), lwork - iwrk + 1, idum, 1);
    if (icond > 0) info = n + icond;
  }
  if (wantvs) {
    // Undo balancing
    // (Workspace: need N)
    Gebak('P', 'R', n, ilo, ihi, WORK.ptr(ibal), n, vs, ldvs);
  }
  if (scalea) {
    // Undo scaling for the Schur form of A
    Lascl('H', 0, 0, cscale, anrm, n, n, a, lda);
    Copy(n, a, lda + 1, wr, 1);
    if (cscale == smlnum) {
      // If scaling back towards underflow, adjust WI if an
      // offdiagonal element of a 2-by-2 block in the Schur form
      // underflows.
      if (ieval > 0) {
        i1 = ieval + 1;
        i2 = ihi - 1;
        Lascl('G', 0, 0, cscale, anrm, ilo - 1, 1, wi,
              std::max<blasint>(ilo - 1, 1));
      } else if (wantst) {
        i1 = 1;
        i2 = n - 1;
      } else {
        i1 = ilo;
        i2 = ihi - 1;
      }
      inxt = i1 - 1;
      for (i = i1; i <= i2; ++i) {
        if (i < inxt) goto L20;
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
      L20:;
      }
    }
    // Undo scaling for the imaginary part of the eigenvalues
    Lascl('G', 0, 0, cscale, anrm, n - ieval, 1, WI.ptr(ieval + 1),
          std::max<blasint>(n - ieval, 1));
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
  return info;
}

// xGEES (complex T): as the real routine, with the eigenvalues W and the
// upper triangular Schur form; SELECT(W(j)) selects. RWORK has N elements.
// INFO = N+2 does not occur.
template <typename T>
blasint Gees(char jobvs, char sort, GeesSelectComplex<RealT<T>> select,
             blasint n, T* a, blasint lda, blasint* sdim, T* w, T* vs,
             blasint ldvs, T* work, blasint lwork, RealT<T>* rwork,
             blasint* bwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> BWORK(bwork);
  R dum[1];
  const Vector<R> RWORK(rwork);
  const Matrix<T> VS(vs, ldvs);
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  blasint hswork, i, ibal, ieval, ihi, ilo, itau, iwrk, maxwrk, minwrk;
  bool lquery, scalea, wantst, wantvs;
  R anrm, bignum, cscale, eps, s, sep, smlnum;
  blasint info = 0;
  lquery = (lwork == -1);
  wantvs = Lsame(jobvs, 'V');
  wantst = Lsame(sort, 'S');
  if ((!wantvs) && (!Lsame(jobvs, 'N'))) {
    info = -1;
  } else if ((!wantst) && (!Lsame(sort, 'N'))) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (lda < std::max<blasint>(1, n)) {
    info = -6;
  } else if (ldvs < 1 || (wantvs && ldvs < n)) {
    info = -10;
  }
  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // CWorkspace refers to complex workspace, and RWorkspace to real
  // workspace. NB refers to the optimal block size for the
  // immediately following subroutine, as returned by ILAENV.
  // HSWORK refers to the workspace preferred by ZHSEQR, as
  // calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
  // the worst case.)
  if (info == 0) {
    if (n == 0) {
      minwrk = 1;
      maxwrk = 1;
    } else {
      maxwrk = n + n * Ilaenv<T>(1, "GEHRD", " ", n, 1, n, 0);
      minwrk = 2 * n;
      ieval = Hseqr('S', jobvs, n, 1, n, a, lda, w, vs, ldvs, work, -1);
      hswork = static_cast<blasint>(std::real(WORK(1)));
      if (!wantvs) {
        maxwrk = std::max<blasint>(maxwrk, hswork);
      } else {
        maxwrk = std::max<blasint>(
            maxwrk, n + (n - 1) * Ilaenv<T>(1, "UNGHR", " ", n, 1, n, -1));
        maxwrk = std::max<blasint>(maxwrk, hswork);
      }
    }
    WORK(1) = WorkSize<T>(maxwrk);
    if (lwork < minwrk && !lquery) {
      info = -12;
    }
  }
  if (info != 0) {
    Xerbla<T>("GEES", -info);
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
    // Reorder eigenvalues and transform Schur vectors
    // (CWorkspace: none)
    // (RWorkspace: none)
    // (ZGEES ignores the INFO of ZTRSEN, which is 0.)
    Trsen('N', jobvs, bwork, n, a, lda, vs, ldvs, w, sdim, &s, &sep,
          WORK.ptr(iwrk), lwork - iwrk + 1);
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
  }
  WORK(1) = WorkSize<T>(maxwrk);
  return info;
}

template blasint Gees<float>(char, char, GeesSelectReal<float>, blasint, float*,
                             blasint, blasint*, float*, float*, float*, blasint,
                             float*, blasint, blasint*);
template blasint Gees<double>(char, char, GeesSelectReal<double>, blasint,
                              double*, blasint, blasint*, double*, double*,
                              double*, blasint, double*, blasint, blasint*);
template blasint Gees<c32>(char, char, GeesSelectComplex<float>, blasint, c32*,
                           blasint, blasint*, c32*, c32*, blasint, c32*,
                           blasint, float*, blasint*);
template blasint Gees<c64>(char, char, GeesSelectComplex<double>, blasint, c64*,
                           blasint, blasint*, c64*, c64*, blasint, c64*,
                           blasint, double*, blasint*);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// SELECT is a Fortran LOGICAL FUNCTION (nonzero is true) of WR and WI, or
// of W (a pointer to its real and imaginary part); BWORK is a LOGICAL
// array.
HWYBLAS_API void sgees_(const char* jobvs, const char* sort,
                        hwyblas_select2_float select, const hwyblas_int* n,
                        float* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        float* wr, float* wi, float* vs,
                        const hwyblas_int* ldvs, float* work,
                        const hwyblas_int* lwork, hwyblas_int* bwork,
                        hwyblas_int* info) {
  *info = hwyblas::lapack::Gees(*jobvs, *sort, select, *n, a, *lda, sdim, wr,
                                wi, vs, *ldvs, work, *lwork, bwork);
}
HWYBLAS_API void dgees_(const char* jobvs, const char* sort,
                        hwyblas_select2_double select, const hwyblas_int* n,
                        double* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        double* wr, double* wi, double* vs,
                        const hwyblas_int* ldvs, double* work,
                        const hwyblas_int* lwork, hwyblas_int* bwork,
                        hwyblas_int* info) {
  *info = hwyblas::lapack::Gees(*jobvs, *sort, select, *n, a, *lda, sdim, wr,
                                wi, vs, *ldvs, work, *lwork, bwork);
}
HWYBLAS_API void cgees_(const char* jobvs, const char* sort,
                        hwyblas_select1_float select, const hwyblas_int* n,
                        float* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        float* w, float* vs, const hwyblas_int* ldvs,
                        float* work, const hwyblas_int* lwork, float* rwork,
                        hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gees(*jobvs, *sort, select, *n, As<c32>(a), *lda,
                                sdim, As<c32>(w), As<c32>(vs), *ldvs,
                                As<c32>(work), *lwork, rwork, bwork);
}
HWYBLAS_API void zgees_(const char* jobvs, const char* sort,
                        hwyblas_select1_double select, const hwyblas_int* n,
                        double* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        double* w, double* vs, const hwyblas_int* ldvs,
                        double* work, const hwyblas_int* lwork, double* rwork,
                        hwyblas_int* bwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Gees(*jobvs, *sort, select, *n, As<c64>(a), *lda,
                                sdim, As<c64>(w), As<c64>(vs), *ldvs,
                                As<c64>(work), *lwork, rwork, bwork);
}

}  // extern "C"
