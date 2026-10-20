// Selected eigenvalues and eigenvectors of symmetric (Hermitian) and
// symmetric tridiagonal matrices by the MRRR algorithm: xSYEVR / xHEEVR and
// xSTEVR. All eigenvalues are computed by xSTERF or xSTEMR; a subset, or
// all if xSTEMR fails (or NaN arithmetic is not trusted), by xSTEBZ
// (bisection) and xSTEIN (inverse iteration).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <type_traits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// DSYEVR (real T).
template <typename T>
blasint SyevrReal(char jobz, char range, char uplo, blasint n, T* a,
                  blasint lda, T vl, T vu, blasint il, blasint iu, T abstol,
                  blasint* m, T* w, T* z, blasint ldz, blasint* isuppz, T* work,
                  blasint lwork, blasint* iwork, blasint liwork) {
  using R = T;
  const char opts[2] = {uplo, '\0'};

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Matrix<R> A(a, lda);
  const Vector<blasint> ISUPPZ(isuppz);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint i, ieeeok, imax, indd, inddd, inde, indee, indibl, indifl, indisp,
      indiwo, indtau, indwk, indwkn, iscale, j, jj, liwmin, llwork, llwrkn,
      lwkopt, lwmin, nb, nsplit;
  bool alleig, indeig, lower, lquery, tryrac, valeig, wantz;
  R abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1;
  // Set for RANGE = 'V' only (xSTEBZ references them only then).
  R vll = zero, vuu = zero;
  char order;
  ieeeok = Ilaenv<T>(10, "SYEVR", "N", 1, 2, 3, 4);
  lower = Lsame(uplo, 'L');
  wantz = Lsame(jobz, 'V');
  alleig = Lsame(range, 'A');
  valeig = Lsame(range, 'V');
  indeig = Lsame(range, 'I');
  lquery = ((lwork == -1) || (liwork == -1));
  if (n <= 1) {
    lwmin = 1;
    liwmin = 1;
  } else {
    lwmin = 26 * n;
    liwmin = 10 * n;
  }
  blasint info = 0;
  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (!(alleig || valeig || indeig)) {
    info = -2;
  } else if (!(lower || Lsame(uplo, 'U'))) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  } else {
    if (valeig) {
      if (n > 0 && vu <= vl) info = -8;
    } else if (indeig) {
      if (il < 1 || il > Max<blasint>(1, n)) {
        info = -9;
      } else if (iu < Min<blasint>(n, il) || iu > n) {
        info = -10;
      }
    }
  }
  if (info == 0) {
    if (ldz < 1 || (wantz && ldz < n)) {
      info = -15;
    } else if (std::is_same_v<T, double> && lwork < lwmin && !lquery) {
      info = -18;
    } else if (std::is_same_v<T, double> && liwork < liwmin && !lquery) {
      info = -20;
    }
  }
  if (info == 0) {
    nb = Ilaenv<T>(1, "SYTRD", opts, n, -1, -1, -1);
    nb = Max(nb, Ilaenv<T>(1, "ORMTR", opts, n, -1, -1, -1));
    lwkopt = Max<blasint>((nb + 1) * n, lwmin);
    WORK(1) = WorkSize<T>(lwkopt);
    IWORK(1) = liwmin;
    if constexpr (std::is_same_v<T, float>) {
      if (lwork < lwmin && !lquery) {
        info = -18;
      } else if (liwork < liwmin && !lquery) {
        info = -20;
      }
    }
  }
  if (info != 0) {
    Xerbla<T>("SYEVR", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  *m = 0;
  if (n == 0) {
    WORK(1) = 1;
    return info;
  }
  if (n == 1) {
    // SSYEVR returns 26 (as LAPACK), DSYEVR 1.
    WORK(1) = std::is_same_v<T, float> ? 26 : 1;
    if (alleig || indeig) {
      *m = 1;
      W(1) = A(1, 1);
    } else {
      if (vl < A(1, 1) && vu >= A(1, 1)) {
        *m = 1;
        W(1) = A(1, 1);
      }
    }
    if (wantz) {
      Z(1, 1) = one;
      ISUPPZ(1) = 1;
      ISUPPZ(2) = 1;
    }
    return info;
  }
  // Get machine constants.
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  smlnum = safmin / eps;
  bignum = one / smlnum;
  rmin = std::sqrt(smlnum);
  rmax = Min(std::sqrt(bignum), one / std::sqrt(std::sqrt(safmin)));
  // Scale matrix to allowable range, if necessary.
  iscale = 0;
  abstll = abstol;
  if (valeig) {
    vll = vl;
    vuu = vu;
  }
  anrm = Lansy('M', uplo, n, a, lda, work);
  if (anrm > zero && anrm < rmin) {
    iscale = 1;
    sigma = rmin / anrm;
  } else if (anrm > rmax) {
    iscale = 1;
    sigma = rmax / anrm;
  }
  if (iscale == 1) {
    if (lower) {
      for (j = 1; j <= n; ++j) {
        Rscal(n - j + 1, sigma, A.ptr(j, j), 1);
      }
    } else {
      for (j = 1; j <= n; ++j) {
        Rscal(j, sigma, A.ptr(1, j), 1);
      }
    }
    if (abstol > 0) abstll = abstol * sigma;
    if (valeig) {
      vll = vl * sigma;
      vuu = vu * sigma;
    }
  }
  // Initialize indices into workspaces.  Note: The IWORK indices are
  // used only if DSTERF or DSTEMR fail.
  // WORK(INDTAU:INDTAU+N-1) stores the scalar factors of the
  // elementary reflectors used in DSYTRD.
  indtau = 1;
  // WORK(INDD:INDD+N-1) stores the tridiagonal's diagonal entries.
  indd = indtau + n;
  // WORK(INDE:INDE+N-1) stores the off-diagonal entries of the
  // tridiagonal matrix from DSYTRD.
  inde = indd + n;
  // WORK(INDDD:INDDD+N-1) is a copy of the diagonal entries over
  // -written by DSTEMR (the DSTERF path copies the diagonal to W).
  inddd = inde + n;
  // WORK(INDEE:INDEE+N-1) is a copy of the off-diagonal entries over
  // -written while computing the eigenvalues in DSTERF and DSTEMR.
  indee = inddd + n;
  // INDWK is the starting offset of the left-over workspace, and
  // LLWORK is the remaining workspace size.
  indwk = indee + n;
  llwork = lwork - indwk + 1;
  // IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
  // stores the block indices of each of the M<=N eigenvalues.
  indibl = 1;
  // IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
  // stores the starting and finishing indices of each block.
  indisp = indibl + n;
  // IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
  // that corresponding to eigenvectors that fail to converge in
  // DSTEIN.  This information is discarded; if any fail, the driver
  // returns INFO > 0.
  indifl = indisp + n;
  // INDIWO is the offset of the remaining integer workspace.
  indiwo = indifl + n;
  // Call DSYTRD to reduce symmetric matrix to tridiagonal form.
  Hetrd(uplo, n, a, lda, WORK.ptr(indd), WORK.ptr(inde), WORK.ptr(indtau),
        WORK.ptr(indwk), llwork);
  // If all eigenvalues are desired
  // then call DSTERF or DSTEMR and DORMTR.
  if ((alleig || (indeig && il == 1 && iu == n)) && ieeeok == 1) {
    if (!wantz) {
      Copy(n, WORK.ptr(indd), 1, w, 1);
      Copy(n - 1, WORK.ptr(inde), 1, WORK.ptr(indee), 1);
      info = Sterf(n, w, WORK.ptr(indee));
    } else {
      Copy(n - 1, WORK.ptr(inde), 1, WORK.ptr(indee), 1);
      Copy(n, WORK.ptr(indd), 1, WORK.ptr(inddd), 1);
      if (abstol <= two * n * eps) {
        tryrac = true;  // NOLINT(readability-simplify-boolean-expr): as LAPACK
      } else {
        tryrac = false;
      }
      info = Stemr(jobz, 'A', n, WORK.ptr(inddd), WORK.ptr(indee), vl, vu, il,
                   iu, m, w, z, ldz, n, isuppz, &tryrac, WORK.ptr(indwk), lwork,
                   iwork, liwork);
      // Apply orthogonal matrix used in reduction to tridiagonal
      // form to eigenvectors returned by DSTEMR.
      if (wantz && info == 0) {
        indwkn = inde;
        llwrkn = lwork - indwkn + 1;
        Unmtr('L', uplo, 'N', n, *m, a, lda, WORK.ptr(indtau), z, ldz,
              WORK.ptr(indwkn), llwrkn);
      }
    }
    if (info == 0) {
      // Everything worked.  Skip DSTEBZ/DSTEIN.  IWORK(:) are
      // undefined.
      *m = n;
      goto L30;
    }
    info = 0;
  }
  // Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN.
  // Also call DSTEBZ and DSTEIN if DSTEMR fails.
  if (wantz) {
    order = 'B';
  } else {
    order = 'E';
  }
  info = Stebz(range, order, n, vll, vuu, il, iu, abstll, WORK.ptr(indd),
               WORK.ptr(inde), m, &nsplit, w, IWORK.ptr(indibl),
               IWORK.ptr(indisp), WORK.ptr(indwk), IWORK.ptr(indiwo));
  if (wantz) {
    info = Stein(n, WORK.ptr(indd), WORK.ptr(inde), *m, w, IWORK.ptr(indibl),
                 IWORK.ptr(indisp), z, ldz, WORK.ptr(indwk), IWORK.ptr(indiwo),
                 IWORK.ptr(indifl));
    // Apply orthogonal matrix used in reduction to tridiagonal
    // form to eigenvectors returned by DSTEIN.
    indwkn = inde;
    llwrkn = lwork - indwkn + 1;
    Unmtr('L', uplo, 'N', n, *m, a, lda, WORK.ptr(indtau), z, ldz,
          WORK.ptr(indwkn), llwrkn);
  }
  // If matrix was scaled, then rescale eigenvalues appropriately.
  // Jump here if DSTEMR/DSTEIN succeeded.
L30:
  if (iscale == 1) {
    if (info == 0) {
      imax = *m;
    } else {
      imax = info - 1;
    }
    Rscal(imax, one / sigma, w, 1);
  }
  // If eigenvalues are not in order, then sort them, along with
  // eigenvectors.  Note: We do not sort the IFAIL portion of IWORK.
  // It may not be initialized (if DSTEMR/DSTEIN succeeded), and we do
  // not return this detailed information to the user.
  if (wantz) {
    for (j = 1; j <= *m - 1; ++j) {
      i = 0;
      tmp1 = W(j);
      for (jj = j + 1; jj <= *m; ++jj) {
        if (W(jj) < tmp1) {
          i = jj;
          tmp1 = W(jj);
        }
      }
      if (i != 0) {
        W(i) = W(j);
        W(j) = tmp1;
        Swap(n, Z.ptr(1, i), 1, Z.ptr(1, j), 1);
      }
    }
  }
  // Set WORK(1) to optimal workspace size.
  WORK(1) = WorkSize<T>(lwkopt);
  IWORK(1) = liwmin;
  return info;
}

// ZHEEVR (complex T).
template <typename T>
blasint HeevrComplex(char jobz, char range, char uplo, blasint n, T* a,
                     blasint lda, RealT<T> vl, RealT<T> vu, blasint il,
                     blasint iu, RealT<T> abstol, blasint* m, RealT<T>* w, T* z,
                     blasint ldz, blasint* isuppz, T* work, blasint lwork,
                     RealT<T>* rwork, blasint lrwork, blasint* iwork,
                     blasint liwork) {
  using R = RealT<T>;
  const char opts[2] = {uplo, '\0'};

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Matrix<T> A(a, lda);
  const Vector<blasint> ISUPPZ(isuppz);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> RWORK(rwork);
  const Vector<R> W(w);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint i, ieeeok, imax, indibl, indifl, indisp, indiwo, indrd, indrdd, indre,
      indree, indrwk, indtau, indwk, indwkn, iscale, itmp1, j, jj, liwmin,
      llrwork, llwork, llwrkn, lrwmin, lwkopt, lwmin, nb, nsplit;
  bool alleig, indeig, lower, lquery, test, tryrac, valeig, wantz;
  R abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1;
  // Set for RANGE = 'V' only (xSTEBZ references them only then).
  R vll = zero, vuu = zero;
  char order;
  ieeeok = Ilaenv<T>(10, "HEEVR", "N", 1, 2, 3, 4);
  lower = Lsame(uplo, 'L');
  wantz = Lsame(jobz, 'V');
  alleig = Lsame(range, 'A');
  valeig = Lsame(range, 'V');
  indeig = Lsame(range, 'I');
  lquery = ((lwork == -1) || (lrwork == -1) || (liwork == -1));
  if (n <= 1) {
    lwmin = 1;
    lrwmin = 1;
    liwmin = 1;
  } else {
    lwmin = 2 * n;
    lrwmin = 24 * n;
    liwmin = 10 * n;
  }
  blasint info = 0;
  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (!(alleig || valeig || indeig)) {
    info = -2;
  } else if (!(lower || Lsame(uplo, 'U'))) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  } else {
    if (valeig) {
      if (n > 0 && vu <= vl) info = -8;
    } else if (indeig) {
      if (il < 1 || il > Max<blasint>(1, n)) {
        info = -9;
      } else if (iu < Min<blasint>(n, il) || iu > n) {
        info = -10;
      }
    }
  }
  if (info == 0) {
    if (ldz < 1 || (wantz && ldz < n)) {
      info = -15;
    }
  }
  if (info == 0) {
    nb = Ilaenv<T>(1, "HETRD", opts, n, -1, -1, -1);
    nb = Max(nb, Ilaenv<T>(1, "UNMTR", opts, n, -1, -1, -1));
    lwkopt = Max<blasint>((nb + 1) * n, lwmin);
    WORK(1) = WorkSize<T>(lwkopt);
    RWORK(1) = WorkSize<R>(lrwmin);
    IWORK(1) = liwmin;
    if (lwork < lwmin && !lquery) {
      info = -18;
    } else if (lrwork < lrwmin && !lquery) {
      info = -20;
    } else if (liwork < liwmin && !lquery) {
      info = -22;
    }
  }
  if (info != 0) {
    Xerbla<T>("HEEVR", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  *m = 0;
  if (n == 0) {
    WORK(1) = 1;
    return info;
  }
  if (n == 1) {
    WORK(1) = 1;
    if (alleig || indeig) {
      *m = 1;
      W(1) = std::real(A(1, 1));
    } else {
      if (vl < std::real(A(1, 1)) && vu >= std::real(A(1, 1))) {
        *m = 1;
        W(1) = std::real(A(1, 1));
      }
    }
    if (wantz) {
      Z(1, 1) = T(one);
      ISUPPZ(1) = 1;
      ISUPPZ(2) = 1;
    }
    return info;
  }
  // Get machine constants.
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  smlnum = safmin / eps;
  bignum = one / smlnum;
  rmin = std::sqrt(smlnum);
  rmax = Min(std::sqrt(bignum), one / std::sqrt(std::sqrt(safmin)));
  // Scale matrix to allowable range, if necessary.
  iscale = 0;
  abstll = abstol;
  if (valeig) {
    vll = vl;
    vuu = vu;
  }
  anrm = Lansy('M', uplo, n, a, lda, rwork);
  if (anrm > zero && anrm < rmin) {
    iscale = 1;
    sigma = rmin / anrm;
  } else if (anrm > rmax) {
    iscale = 1;
    sigma = rmax / anrm;
  }
  if (iscale == 1) {
    if (lower) {
      for (j = 1; j <= n; ++j) {
        Rscal(n - j + 1, sigma, A.ptr(j, j), 1);
      }
    } else {
      for (j = 1; j <= n; ++j) {
        Rscal(j, sigma, A.ptr(1, j), 1);
      }
    }
    if (abstol > 0) abstll = abstol * sigma;
    if (valeig) {
      vll = vl * sigma;
      vuu = vu * sigma;
    }
  }
  // Initialize indices into workspaces.  Note: The IWORK indices are
  // used only if DSTERF or ZSTEMR fail.
  // WORK(INDTAU:INDTAU+N-1) stores the complex scalar factors of the
  // elementary reflectors used in ZHETRD.
  indtau = 1;
  // INDWK is the starting offset of the remaining complex workspace,
  // and LLWORK is the remaining complex workspace size.
  indwk = indtau + n;
  llwork = lwork - indwk + 1;
  // RWORK(INDRD:INDRD+N-1) stores the real tridiagonal's diagonal
  // entries.
  indrd = 1;
  // RWORK(INDRE:INDRE+N-1) stores the off-diagonal entries of the
  // tridiagonal matrix from ZHETRD.
  indre = indrd + n;
  // RWORK(INDRDD:INDRDD+N-1) is a copy of the diagonal entries over
  // -written by ZSTEMR (the DSTERF path copies the diagonal to W).
  indrdd = indre + n;
  // RWORK(INDREE:INDREE+N-1) is a copy of the off-diagonal entries over
  // -written while computing the eigenvalues in DSTERF and ZSTEMR.
  indree = indrdd + n;
  // INDRWK is the starting offset of the left-over real workspace, and
  // LLRWORK is the remaining workspace size.
  indrwk = indree + n;
  llrwork = lrwork - indrwk + 1;
  // IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
  // stores the block indices of each of the M<=N eigenvalues.
  indibl = 1;
  // IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
  // stores the starting and finishing indices of each block.
  indisp = indibl + n;
  // IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
  // that corresponding to eigenvectors that fail to converge in
  // DSTEIN.  This information is discarded; if any fail, the driver
  // returns INFO > 0.
  indifl = indisp + n;
  // INDIWO is the offset of the remaining integer workspace.
  indiwo = indifl + n;
  // Call ZHETRD to reduce Hermitian matrix to tridiagonal form.
  Hetrd(uplo, n, a, lda, RWORK.ptr(indrd), RWORK.ptr(indre), WORK.ptr(indtau),
        WORK.ptr(indwk), llwork);
  // If all eigenvalues are desired
  // then call DSTERF or ZSTEMR and ZUNMTR.
  test = false;
  if (indeig) {
    if (il == 1 && iu == n) {
      test = true;
    }
  }
  if ((alleig || test) && (ieeeok == 1)) {
    if (!wantz) {
      Copy(n, RWORK.ptr(indrd), 1, w, 1);
      Copy(n - 1, RWORK.ptr(indre), 1, RWORK.ptr(indree), 1);
      info = Sterf(n, w, RWORK.ptr(indree));
    } else {
      Copy(n - 1, RWORK.ptr(indre), 1, RWORK.ptr(indree), 1);
      Copy(n, RWORK.ptr(indrd), 1, RWORK.ptr(indrdd), 1);
      if (abstol <= two * n * eps) {
        tryrac = true;  // NOLINT(readability-simplify-boolean-expr): as LAPACK
      } else {
        tryrac = false;
      }
      info = Stemr(jobz, 'A', n, RWORK.ptr(indrdd), RWORK.ptr(indree), vl, vu,
                   il, iu, m, w, z, ldz, n, isuppz, &tryrac, RWORK.ptr(indrwk),
                   llrwork, iwork, liwork);
      // Apply unitary matrix used in reduction to tridiagonal
      // form to eigenvectors returned by ZSTEMR.
      if (wantz && info == 0) {
        indwkn = indwk;
        llwrkn = lwork - indwkn + 1;
        Unmtr('L', uplo, 'N', n, *m, a, lda, WORK.ptr(indtau), z, ldz,
              WORK.ptr(indwkn), llwrkn);
      }
    }
    if (info == 0) {
      *m = n;
      goto L30;
    }
    info = 0;
  }
  // Otherwise, call DSTEBZ and, if eigenvectors are desired, ZSTEIN.
  // Also call DSTEBZ and ZSTEIN if ZSTEMR fails.
  if (wantz) {
    order = 'B';
  } else {
    order = 'E';
  }
  info = Stebz(range, order, n, vll, vuu, il, iu, abstll, RWORK.ptr(indrd),
               RWORK.ptr(indre), m, &nsplit, w, IWORK.ptr(indibl),
               IWORK.ptr(indisp), RWORK.ptr(indrwk), IWORK.ptr(indiwo));
  if (wantz) {
    info = Stein(n, RWORK.ptr(indrd), RWORK.ptr(indre), *m, w,
                 IWORK.ptr(indibl), IWORK.ptr(indisp), z, ldz,
                 RWORK.ptr(indrwk), IWORK.ptr(indiwo), IWORK.ptr(indifl));
    // Apply unitary matrix used in reduction to tridiagonal
    // form to eigenvectors returned by ZSTEIN.
    indwkn = indwk;
    llwrkn = lwork - indwkn + 1;
    Unmtr('L', uplo, 'N', n, *m, a, lda, WORK.ptr(indtau), z, ldz,
          WORK.ptr(indwkn), llwrkn);
  }
  // If matrix was scaled, then rescale eigenvalues appropriately.
L30:
  if (iscale == 1) {
    if (info == 0) {
      imax = *m;
    } else {
      imax = info - 1;
    }
    Scal(imax, one / sigma, w, 1);
  }
  // If eigenvalues are not in order, then sort them, along with
  // eigenvectors.
  if (wantz) {
    for (j = 1; j <= *m - 1; ++j) {
      i = 0;
      tmp1 = W(j);
      for (jj = j + 1; jj <= *m; ++jj) {
        if (W(jj) < tmp1) {
          i = jj;
          tmp1 = W(jj);
        }
      }
      if (i != 0) {
        itmp1 = IWORK(indibl + i - 1);
        W(i) = W(j);
        IWORK(indibl + i - 1) = IWORK(indibl + j - 1);
        W(j) = tmp1;
        IWORK(indibl + j - 1) = itmp1;
        Swap(n, Z.ptr(1, i), 1, Z.ptr(1, j), 1);
      }
    }
  }
  // Set WORK(1) to optimal workspace size.
  WORK(1) = WorkSize<T>(lwkopt);
  RWORK(1) = WorkSize<R>(lrwmin);
  IWORK(1) = liwmin;
  return info;
}

}  // namespace

// xSYEVR / xHEEVR: selected eigenvalues and, optionally (JOBZ = 'V'),
// eigenvectors of the N-by-N real symmetric (complex Hermitian) matrix A,
// of which the triangle UPLO is referenced (destroyed on exit): all (RANGE
// = 'A'), those in (VL, VU] ('V') or the IL-th through IU-th ('I'). A is
// reduced to tridiagonal form (xSYTRD / xHETRD); all eigenvalues are
// computed by xSTERF or xSTEMR (MRRR; with high relative accuracy if
// ABSTOL <= 2*N*EPS), a subset (or all, if xSTEMR fails or NaN arithmetic
// is not trusted) by xSTEBZ to the absolute accuracy ABSTOL and xSTEIN;
// xORMTR / xUNMTR transforms the eigenvectors back. Returns M eigenvalues W
// in ascending order, the eigenvectors in Z and (xSTEMR only) their
// supports ISUPPZ.
//
// The real and the complex routine use different workspaces (N > 1):
//   real T:    LWORK >= 26*N, LIWORK >= 10*N; WORK holds TAU, the
//              tridiagonal matrix and its copies; RWORK and LRWORK are not
//              referenced.
//   complex T: LWORK >= 2*N, LRWORK >= 24*N, LIWORK >= 10*N; RWORK holds
//              the tridiagonal matrix.
// LWORK, LRWORK or LIWORK = -1 is a workspace query; the optimal LWORK is
// max(LWORK minimum, (NB+1)*N) with the block size NB of xSYTRD / xORMTR
// (xHETRD / xUNMTR). The real routine passes all of LWORK to xSTEMR,
// although its WORK starts at an offset (as LAPACK; xSTEMR uses 18*N).
// Returns INFO: 0, -i for an illegal argument i, or > 0 for an internal
// error (xSTEBZ, xSTEIN: see there). The norm that decides the scaling
// is xLANSY's also for complex T (as ZHEEVR), which reads the imaginary
// parts of the diagonal. As LAPACK's SSYEVR, the single
// precision routine sets WORK(1) and IWORK(1) also when LWORK or LIWORK is
// too small, and WORK(1) = 26 for N = 1.
template <typename T>
blasint Heevr(char jobz, char range, char uplo, blasint n, T* a, blasint lda,
              RealT<T> vl, RealT<T> vu, blasint il, blasint iu, RealT<T> abstol,
              blasint* m, RealT<T>* w, T* z, blasint ldz, blasint* isuppz,
              T* work, blasint lwork, RealT<T>* rwork, blasint lrwork,
              blasint* iwork, blasint liwork) {
  if constexpr (IsComplex<T>::value) {
    return HeevrComplex(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m,
                        w, z, ldz, isuppz, work, lwork, rwork, lrwork, iwork,
                        liwork);
  } else {
    return SyevrReal(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w,
                     z, ldz, isuppz, work, lwork, iwork, liwork);
  }
}

// xSTEVR: selected eigenvalues and, optionally (JOBZ = 'V'), eigenvectors
// of the real symmetric tridiagonal matrix with diagonal D and
// off-diagonal E (N; E is destroyed, D too if xSTEBZ is used): as
// xSYEVR for a tridiagonal matrix. LWORK >= max(1, 20*N), LIWORK >= max(1,
// 10*N); -1 is a workspace query. Returns INFO: 0, -i for an illegal
// argument i, or > 0 for an internal error. Real R only.
template <typename R>
blasint Stevr(char jobz, char range, blasint n, R* d, R* e, R vl, R vu,
              blasint il, blasint iu, R abstol, blasint* m, R* w, R* z,
              blasint ldz, blasint* isuppz, R* work, blasint lwork,
              blasint* iwork, blasint liwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> ISUPPZ(isuppz);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint i, ieeeok, imax, indibl, indifl, indisp, indiwo, iscale, itmp1, j, jj,
      liwmin, lwmin, nsplit;
  bool alleig, indeig, lquery, test, tryrac, valeig, wantz;
  R bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1, tnrm;
  // Set for RANGE = 'V' only (xSTEBZ references them only then).
  R vll = zero, vuu = zero;
  char order;
  ieeeok = Ilaenv<R>(10, "STEVR", "N", 1, 2, 3, 4);
  wantz = Lsame(jobz, 'V');
  alleig = Lsame(range, 'A');
  valeig = Lsame(range, 'V');
  indeig = Lsame(range, 'I');
  lquery = ((lwork == -1) || (liwork == -1));
  lwmin = Max<blasint>(1, 20 * n);
  liwmin = Max<blasint>(1, 10 * n);
  blasint info = 0;
  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (!(alleig || valeig || indeig)) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else {
    if (valeig) {
      if (n > 0 && vu <= vl) info = -7;
    } else if (indeig) {
      if (il < 1 || il > Max<blasint>(1, n)) {
        info = -8;
      } else if (iu < Min<blasint>(n, il) || iu > n) {
        info = -9;
      }
    }
  }
  if (info == 0) {
    if (ldz < 1 || (wantz && ldz < n)) {
      info = -14;
    }
  }
  if (info == 0) {
    WORK(1) = WorkSize<R>(lwmin);
    IWORK(1) = liwmin;
    if (lwork < lwmin && !lquery) {
      info = -17;
    } else if (liwork < liwmin && !lquery) {
      info = -19;
    }
  }
  if (info != 0) {
    Xerbla<R>("STEVR", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  *m = 0;
  if (n == 0) return info;
  if (n == 1) {
    if (alleig || indeig) {
      *m = 1;
      W(1) = D(1);
    } else {
      if (vl < D(1) && vu >= D(1)) {
        *m = 1;
        W(1) = D(1);
      }
    }
    if (wantz) Z(1, 1) = one;
    return info;
  }
  // Get machine constants.
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  smlnum = safmin / eps;
  bignum = one / smlnum;
  rmin = std::sqrt(smlnum);
  rmax = Min(std::sqrt(bignum), one / std::sqrt(std::sqrt(safmin)));
  // Scale matrix to allowable range, if necessary.
  iscale = 0;
  if (valeig) {
    vll = vl;
    vuu = vu;
  }
  tnrm = Lanst('M', n, d, e);
  if (tnrm > zero && tnrm < rmin) {
    iscale = 1;
    sigma = rmin / tnrm;
  } else if (tnrm > rmax) {
    iscale = 1;
    sigma = rmax / tnrm;
  }
  if (iscale == 1) {
    Scal(n, sigma, d, 1);
    Scal(n - 1, sigma, E.ptr(1), 1);
    if (valeig) {
      vll = vl * sigma;
      vuu = vu * sigma;
    }
  }
  // Initialize indices into workspaces.  Note: These indices are used only
  // if DSTERF or DSTEMR fail.
  // IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
  // stores the block indices of each of the M<=N eigenvalues.
  indibl = 1;
  // IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
  // stores the starting and finishing indices of each block.
  indisp = indibl + n;
  // IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
  // that corresponding to eigenvectors that fail to converge in
  // DSTEIN.  This information is discarded; if any fail, the driver
  // returns INFO > 0.
  indifl = indisp + n;
  // INDIWO is the offset of the remaining integer workspace.
  indiwo = indisp + n;
  // If all eigenvalues are desired, then
  // call DSTERF or DSTEMR.  If this fails for some eigenvalue, then
  // try DSTEBZ.
  test = false;
  if (indeig) {
    if (il == 1 && iu == n) {
      test = true;
    }
  }
  if ((alleig || test) && ieeeok == 1) {
    Copy(n - 1, E.ptr(1), 1, WORK.ptr(1), 1);
    if (!wantz) {
      Copy(n, d, 1, w, 1);
      info = Sterf(n, w, work);
    } else {
      Copy(n, d, 1, WORK.ptr(n + 1), 1);
      if (abstol <= two * n * eps) {
        tryrac = true;  // NOLINT(readability-simplify-boolean-expr): as LAPACK
      } else {
        tryrac = false;
      }
      info = Stemr(jobz, 'A', n, WORK.ptr(n + 1), work, vl, vu, il, iu, m, w, z,
                   ldz, n, isuppz, &tryrac, WORK.ptr(2 * n + 1), lwork - 2 * n,
                   iwork, liwork);
    }
    if (info == 0) {
      *m = n;
      goto L10;
    }
    info = 0;
  }
  // Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN.
  if (wantz) {
    order = 'B';
  } else {
    order = 'E';
  }
  info = Stebz(range, order, n, vll, vuu, il, iu, abstol, d, e, m, &nsplit, w,
               IWORK.ptr(indibl), IWORK.ptr(indisp), work, IWORK.ptr(indiwo));
  if (wantz) {
    info = Stein(n, d, e, *m, w, IWORK.ptr(indibl), IWORK.ptr(indisp), z, ldz,
                 work, IWORK.ptr(indiwo), IWORK.ptr(indifl));
  }
  // If matrix was scaled, then rescale eigenvalues appropriately.
L10:
  if (iscale == 1) {
    if (info == 0) {
      imax = *m;
    } else {
      imax = info - 1;
    }
    Scal(imax, one / sigma, w, 1);
  }
  // If eigenvalues are not in order, then sort them, along with
  // eigenvectors.
  if (wantz) {
    for (j = 1; j <= *m - 1; ++j) {
      i = 0;
      tmp1 = W(j);
      for (jj = j + 1; jj <= *m; ++jj) {
        if (W(jj) < tmp1) {
          i = jj;
          tmp1 = W(jj);
        }
      }
      if (i != 0) {
        W(i) = W(j);
        W(j) = tmp1;
        // DSTEVR also swaps the block numbers of xSTEBZ in IWORK (which are
        // not returned), SSTEVR does not.
        if constexpr (std::is_same_v<R, double>) {
          itmp1 = IWORK(i);
          IWORK(i) = IWORK(j);
          IWORK(j) = itmp1;
        }
        Swap(n, Z.ptr(1, i), 1, Z.ptr(1, j), 1);
      }
    }
  }
  // Causes problems with tests 19 & 20:
  // IF (wantz .and. INDEIG ) Z( 1,1) = Z(1,1) / 1.002 + .002
  WORK(1) = WorkSize<R>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Heevr<T>(char, char, char, blasint, T*, blasint, RealT<T>, \
                            RealT<T>, blasint, blasint, RealT<T>, blasint*,   \
                            RealT<T>*, T*, blasint, blasint*, T*, blasint,    \
                            RealT<T>*, blasint, blasint*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE
template blasint Stevr<float>(char, char, blasint, float*, float*, float, float,
                              blasint, blasint, float, blasint*, float*, float*,
                              blasint, blasint*, float*, blasint, blasint*,
                              blasint);
template blasint Stevr<double>(char, char, blasint, double*, double*, double,
                               double, blasint, blasint, double, blasint*,
                               double*, double*, blasint, blasint*, double*,
                               blasint, blasint*, blasint);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_SYEVR_REAL(p, R)                                       \
  HWYBLAS_API void p##syevr_(                                                 \
      const char* jobz, const char* range, const char* uplo,                  \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, const R* vl,        \
      const R* vu, const hwyblas_int* il, const hwyblas_int* iu,              \
      const R* abstol, hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,    \
      hwyblas_int* isuppz, R* work, const hwyblas_int* lwork,                 \
      hwyblas_int* iwork, const hwyblas_int* liwork, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Heevr(                                           \
        *jobz, *range, *uplo, *n, a, *lda, *vl, *vu, *il, *iu, *abstol, m, w, \
        z, *ldz, isuppz, work, *lwork, static_cast<R*>(nullptr), 0, iwork,    \
        *liwork);                                                             \
  }                                                                           \
  HWYBLAS_API void p##stevr_(                                                 \
      const char* jobz, const char* range, const hwyblas_int* n, R* d, R* e,  \
      const R* vl, const R* vu, const hwyblas_int* il, const hwyblas_int* iu, \
      const R* abstol, hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,    \
      hwyblas_int* isuppz, R* work, const hwyblas_int* lwork,                 \
      hwyblas_int* iwork, const hwyblas_int* liwork, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Stevr(*jobz, *range, *n, d, e, *vl, *vu, *il,    \
                                   *iu, *abstol, m, w, z, *ldz, isuppz, work, \
                                   *lwork, iwork, *liwork);                   \
  }

HWYBLAS_LAPACK_SYEVR_REAL(s, float)
HWYBLAS_LAPACK_SYEVR_REAL(d, double)

#undef HWYBLAS_LAPACK_SYEVR_REAL

#define HWYBLAS_LAPACK_HEEVR(p, T, R)                                        \
  HWYBLAS_API void p##heevr_(                                                \
      const char* jobz, const char* range, const char* uplo,                 \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, const R* vl,       \
      const R* vu, const hwyblas_int* il, const hwyblas_int* iu,             \
      const R* abstol, hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,   \
      hwyblas_int* isuppz, R* work, const hwyblas_int* lwork, R* rwork,      \
      const hwyblas_int* lrwork, hwyblas_int* iwork,                         \
      const hwyblas_int* liwork, hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Heevr(*jobz, *range, *uplo, *n, As<T>(a), *lda, \
                                   *vl, *vu, *il, *iu, *abstol, m, w,        \
                                   As<T>(z), *ldz, isuppz, As<T>(work),      \
                                   *lwork, rwork, *lrwork, iwork, *liwork);  \
  }

HWYBLAS_LAPACK_HEEVR(c, c32, float)
HWYBLAS_LAPACK_HEEVR(z, c64, double)

#undef HWYBLAS_LAPACK_HEEVR

}  // extern "C"
