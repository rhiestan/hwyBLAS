// Selected eigenvalues and eigenvectors of symmetric (Hermitian) and
// symmetric tridiagonal matrices by bisection and inverse iteration: xSYEVX
// / xHEEVX and xSTEVX. All eigenvalues are computed by xSTERF or xSTEQR if
// ABSTOL <= 0; a subset, or all if those fail, by xSTEBZ (bisection) and
// xSTEIN (inverse iteration).
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

// xSYEVX / xHEEVX: selected eigenvalues and, optionally (JOBZ = 'V'),
// eigenvectors of the N-by-N real symmetric (complex Hermitian) matrix A,
// of which the triangle UPLO is referenced (destroyed on exit): all (RANGE
// = 'A'), those in (VL, VU] ('V') or the IL-th through IU-th ('I'). A is
// reduced to tridiagonal form (xSYTRD / xHETRD); all eigenvalues are
// computed by xSTERF or xSTEQR (after xORGTR / xUNGTR) if ABSTOL <= 0, a
// subset (or all, if those fail) by xSTEBZ to the absolute accuracy ABSTOL
// and xSTEIN; xORMTR / xUNMTR transforms the eigenvectors back. Returns M
// eigenvalues W in ascending order, the eigenvectors in Z and, if INFO >
// 0, the indices of those that failed to converge in IFAIL (JOBZ = 'V').
//
// The real and the complex routine use different workspaces (N > 1):
//   real T:    LWORK >= 8*N, optimal max(8*N, (NB+3)*N); WORK holds TAU,
//              the tridiagonal matrix and the workspace of the calls; RWORK
//              is not referenced.
//   complex T: LWORK >= 2*N, optimal (NB+1)*N; WORK holds TAU, RWORK (7*N
//              elements) the tridiagonal matrix and the workspace of the
//              real calls.
// NB is the block size of xSYTRD / xORMTR (xHETRD / xUNMTR); IWORK has 5*N
// elements. LWORK = -1 is a workspace query. Returns INFO: 0, -i for an
// illegal argument i, or i > 0 if i eigenvectors failed to converge.
template <typename T>
blasint Heevx(char jobz, char range, char uplo, blasint n, T* a, blasint lda,
              RealT<T> vl, RealT<T> vu, blasint il, blasint iu, RealT<T> abstol,
              blasint* m, RealT<T>* w, T* z, blasint ldz, T* work,
              blasint lwork, RealT<T>* rwork, blasint* iwork, blasint* ifail) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IFAIL(ifail);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  const char opts[2] = {uplo, '\0'};
  bool alleig, indeig, lower, lquery, test, valeig, wantz;
  char order;
  blasint i, imax, indd, inde, indee, indisp, indiwk, indrwk, indtau, indwrk,
      iscale, itmp1, j, jj, llwork, lwkmin, lwkopt, nb, nsplit;
  // Set before xSTEBZ; the sort reads it also after xSTEQR (unset in LAPACK),
  // whose eigenvalues are in order, so that it swaps nothing.
  blasint indibl = 1;
  R abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1;
  // Set for RANGE = 'V' only (xSTEBZ references them only then).
  R vll = zero, vuu = zero;

  // Test the input parameters.
  lower = Lsame(uplo, 'L');
  wantz = Lsame(jobz, 'V');
  alleig = Lsame(range, 'A');
  valeig = Lsame(range, 'V');
  indeig = Lsame(range, 'I');
  lquery = (lwork == -1);

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

  lwkopt = 1;
  if (info == 0) {
    if (n <= 1) {
      lwkmin = 1;
      lwkopt = 1;
    } else {
      nb =
          Ilaenv<T>(1, RealOrComplex<T>("SYTRD", "HETRD"), opts, n, -1, -1, -1);
      nb = Max(nb, Ilaenv<T>(1, RealOrComplex<T>("ORMTR", "UNMTR"), opts, n, -1,
                             -1, -1));
      if constexpr (kComplex) {
        lwkmin = 2 * n;
        lwkopt = Max<blasint>(1, (nb + 1) * n);
      } else {
        lwkmin = 8 * n;
        lwkopt = Max(lwkmin, (nb + 3) * n);
      }
    }
    WORK(1) = WorkSize<T>(lwkopt);
    if (lwork < lwkmin && !lquery) info = -17;
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYEVX", "HEEVX"), -info);
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
      W(1) = std::real(A(1, 1));
    } else if (valeig) {
      if (vl < std::real(A(1, 1)) && vu >= std::real(A(1, 1))) {
        *m = 1;
        W(1) = std::real(A(1, 1));
      }
    }
    if (wantz) Z(1, 1) = T(one);
    return info;
  }

  // Get machine constants.
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  smlnum = safmin / eps;
  bignum = one / smlnum;
  rmin = std::sqrt(smlnum);
  rmax = Min(std::sqrt(bignum), one / std::sqrt(std::sqrt(safmin)));

  // The tridiagonal matrix and the workspace of the real calls: in RWORK
  // for complex T, in WORK for real T.
  R* rw;
  if constexpr (kComplex) {
    rw = rwork;
  } else {
    rw = work;
  }
  const Vector<R> RWORK(rw);

  // Scale matrix to allowable range, if necessary.
  iscale = 0;
  abstll = abstol;
  if (valeig) {
    vll = vl;
    vuu = vu;
  }
  if constexpr (kComplex) {
    anrm = Lanhe('M', uplo, n, a, lda, rw);
  } else {
    anrm = Lansy('M', uplo, n, a, lda, rw);
  }
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

  // Call xSYTRD / xHETRD to reduce the symmetric (Hermitian) matrix to
  // tridiagonal form. The real routine keeps TAU, E, D and its workspace in
  // WORK (INDRWK = INDWRK), the complex one D, E and the real workspace in
  // RWORK.
  indtau = 1;
  if constexpr (kComplex) {
    indd = 1;
    inde = indd + n;
    indrwk = inde + n;
    indwrk = indtau + n;
  } else {
    inde = indtau + n;
    indd = inde + n;
    indwrk = indd + n;
    indrwk = indwrk;
  }
  llwork = lwork - indwrk + 1;
  Hetrd(uplo, n, a, lda, RWORK.ptr(indd), RWORK.ptr(inde), WORK.ptr(indtau),
        WORK.ptr(indwrk), llwork);

  // If all eigenvalues are desired and ABSTOL is less than or equal to
  // zero, then call xSTERF or xUNGTR and xSTEQR. If this fails for some
  // eigenvalue, then try xSTEBZ.
  test = false;
  if (indeig) {
    if (il == 1 && iu == n) {
      test = true;
    }
  }
  if ((alleig || test) && (abstol <= zero)) {
    Copy(n, RWORK.ptr(indd), 1, w, 1);
    indee = indrwk + 2 * n;
    if (!wantz) {
      Copy(n - 1, RWORK.ptr(inde), 1, RWORK.ptr(indee), 1);
      info = Sterf(n, w, RWORK.ptr(indee));
    } else {
      Lacpy('A', n, n, a, lda, z, ldz);
      Ungtr(uplo, n, z, ldz, WORK.ptr(indtau), WORK.ptr(indwrk), llwork);
      Copy(n - 1, RWORK.ptr(inde), 1, RWORK.ptr(indee), 1);
      info = Steqr(jobz, n, w, RWORK.ptr(indee), z, ldz, RWORK.ptr(indrwk));
      if (info == 0) {
        for (i = 1; i <= n; ++i) {
          IFAIL(i) = 0;
        }
      }
    }
    if (info == 0) {
      *m = n;
      goto L40;
    }
    info = 0;
  }

  // Otherwise, call xSTEBZ and, if eigenvectors are desired, xSTEIN.
  if (wantz) {
    order = 'B';
  } else {
    order = 'E';
  }
  indibl = 1;
  indisp = indibl + n;
  indiwk = indisp + n;
  info = Stebz(range, order, n, vll, vuu, il, iu, abstll, RWORK.ptr(indd),
               RWORK.ptr(inde), m, &nsplit, w, IWORK.ptr(indibl),
               IWORK.ptr(indisp), RWORK.ptr(indrwk), IWORK.ptr(indiwk));

  if (wantz) {
    info = Stein(n, RWORK.ptr(indd), RWORK.ptr(inde), *m, w, IWORK.ptr(indibl),
                 IWORK.ptr(indisp), z, ldz, RWORK.ptr(indrwk),
                 IWORK.ptr(indiwk), ifail);

    // Apply orthogonal (unitary) matrix used in reduction to tridiagonal
    // form to eigenvectors returned by xSTEIN. The real routine's workspace
    // starts at E (INDWKN = INDE, LLWRKN = LWORK - INDWKN + 1), which is no
    // longer needed.
    if constexpr (kComplex) {
      Unmtr('L', uplo, 'N', n, *m, a, lda, WORK.ptr(indtau), z, ldz,
            WORK.ptr(indwrk), llwork);
    } else {
      const blasint indwkn = inde;
      const blasint llwrkn = lwork - indwkn + 1;
      Unmtr('L', uplo, 'N', n, *m, a, lda, WORK.ptr(indtau), z, ldz,
            WORK.ptr(indwkn), llwrkn);
    }
  }

  // If matrix was scaled, then rescale eigenvalues appropriately.
L40:
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
        if (info != 0) {
          itmp1 = IFAIL(i);
          IFAIL(i) = IFAIL(j);
          IFAIL(j) = itmp1;
        }
      }
    }
  }

  // Set WORK(1) to optimal workspace size.
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xSTEVX: selected eigenvalues and, optionally (JOBZ = 'V'), eigenvectors
// of the real symmetric tridiagonal matrix with diagonal D and
// off-diagonal E (N; both may be scaled on exit, as LAPACK): as xSYEVX for
// a tridiagonal matrix (xSTEQR computes the eigenvectors of the whole
// matrix if ABSTOL <= 0). WORK has 5*N elements, IWORK 5*N. Returns INFO:
// 0, -i for an illegal argument i, or i > 0 if i eigenvectors failed to
// converge. Real R only.
template <typename R>
blasint Stevx(char jobz, char range, blasint n, R* d, R* e, R vl, R vu,
              blasint il, blasint iu, R abstol, blasint* m, R* w, R* z,
              blasint ldz, R* work, blasint* iwork, blasint* ifail) {
  const R zero = R(0);
  const R one = R(1);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IFAIL(ifail);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  bool alleig, indeig, test, valeig, wantz;
  char order;
  blasint i, imax, indisp, indiwo, indwrk, iscale, itmp1, j, jj, nsplit;
  R bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1, tnrm, vll, vuu;

  // Test the input parameters.
  wantz = Lsame(jobz, 'V');
  alleig = Lsame(range, 'A');
  valeig = Lsame(range, 'V');
  indeig = Lsame(range, 'I');

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
    if (ldz < 1 || (wantz && ldz < n)) info = -14;
  }

  if (info != 0) {
    Xerbla<R>("STEVX", -info);
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
  } else {
    vll = zero;
    vuu = zero;
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

  // If all eigenvalues are desired and ABSTOL is less than zero, then call
  // xSTERF or xSTEQR. If this fails for some eigenvalue, then try xSTEBZ.
  test = false;
  if (indeig) {
    if (il == 1 && iu == n) {
      test = true;
    }
  }
  if ((alleig || test) && (abstol <= zero)) {
    Copy(n, d, 1, w, 1);
    Copy(n - 1, E.ptr(1), 1, WORK.ptr(1), 1);
    indwrk = n + 1;
    if (!wantz) {
      info = Sterf(n, w, work);
    } else {
      info = Steqr('I', n, w, work, z, ldz, WORK.ptr(indwrk));
      if (info == 0) {
        for (i = 1; i <= n; ++i) {
          IFAIL(i) = 0;
        }
      }
    }
    if (info == 0) {
      *m = n;
      goto L20;
    }
    info = 0;
  }

  // Otherwise, call xSTEBZ and, if eigenvectors are desired, xSTEIN.
  if (wantz) {
    order = 'B';
  } else {
    order = 'E';
  }
  indwrk = 1;
  indisp = 1 + n;
  indiwo = indisp + n;
  info = Stebz(range, order, n, vll, vuu, il, iu, abstol, d, e, m, &nsplit, w,
               IWORK.ptr(1), IWORK.ptr(indisp), WORK.ptr(indwrk),
               IWORK.ptr(indiwo));

  if (wantz) {
    info = Stein(n, d, e, *m, w, IWORK.ptr(1), IWORK.ptr(indisp), z, ldz,
                 WORK.ptr(indwrk), IWORK.ptr(indiwo), ifail);
  }

  // If matrix was scaled, then rescale eigenvalues appropriately.
L20:
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
        itmp1 = IWORK(1 + i - 1);
        W(i) = W(j);
        IWORK(1 + i - 1) = IWORK(1 + j - 1);
        W(j) = tmp1;
        IWORK(1 + j - 1) = itmp1;
        Swap(n, Z.ptr(1, i), 1, Z.ptr(1, j), 1);
        if (info != 0) {
          itmp1 = IFAIL(i);
          IFAIL(i) = IFAIL(j);
          IFAIL(j) = itmp1;
        }
      }
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Heevx<T>(char, char, char, blasint, T*, blasint, RealT<T>, \
                            RealT<T>, blasint, blasint, RealT<T>, blasint*,   \
                            RealT<T>*, T*, blasint, T*, blasint, RealT<T>*,   \
                            blasint*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE
template blasint Stevx<float>(char, char, blasint, float*, float*, float, float,
                              blasint, blasint, float, blasint*, float*, float*,
                              blasint, float*, blasint*, blasint*);
template blasint Stevx<double>(char, char, blasint, double*, double*, double,
                               double, blasint, blasint, double, blasint*,
                               double*, double*, blasint, double*, blasint*,
                               blasint*);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_SYEVX_REAL(p, R)                                       \
  HWYBLAS_API void p##syevx_(                                                 \
      const char* jobz, const char* range, const char* uplo,                  \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, const R* vl,        \
      const R* vu, const hwyblas_int* il, const hwyblas_int* iu,              \
      const R* abstol, hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,    \
      R* work, const hwyblas_int* lwork, hwyblas_int* iwork,                  \
      hwyblas_int* ifail, hwyblas_int* info) {                                \
    *info = hwyblas::lapack::Heevx(                                           \
        *jobz, *range, *uplo, *n, a, *lda, *vl, *vu, *il, *iu, *abstol, m, w, \
        z, *ldz, work, *lwork, static_cast<R*>(nullptr), iwork, ifail);       \
  }                                                                           \
  HWYBLAS_API void p##stevx_(                                                 \
      const char* jobz, const char* range, const hwyblas_int* n, R* d, R* e,  \
      const R* vl, const R* vu, const hwyblas_int* il, const hwyblas_int* iu, \
      const R* abstol, hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,    \
      R* work, hwyblas_int* iwork, hwyblas_int* ifail, hwyblas_int* info) {   \
    *info =                                                                   \
        hwyblas::lapack::Stevx(*jobz, *range, *n, d, e, *vl, *vu, *il, *iu,   \
                               *abstol, m, w, z, *ldz, work, iwork, ifail);   \
  }

HWYBLAS_LAPACK_SYEVX_REAL(s, float)
HWYBLAS_LAPACK_SYEVX_REAL(d, double)

#undef HWYBLAS_LAPACK_SYEVX_REAL

#define HWYBLAS_LAPACK_HEEVX(p, T, R)                                          \
  HWYBLAS_API void p##heevx_(                                                  \
      const char* jobz, const char* range, const char* uplo,                   \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, const R* vl,         \
      const R* vu, const hwyblas_int* il, const hwyblas_int* iu,               \
      const R* abstol, hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,     \
      R* work, const hwyblas_int* lwork, R* rwork, hwyblas_int* iwork,         \
      hwyblas_int* ifail, hwyblas_int* info) {                                 \
    *info = hwyblas::lapack::Heevx(                                            \
        *jobz, *range, *uplo, *n, As<T>(a), *lda, *vl, *vu, *il, *iu, *abstol, \
        m, w, As<T>(z), *ldz, As<T>(work), *lwork, rwork, iwork, ifail);       \
  }

HWYBLAS_LAPACK_HEEVX(c, c32, float)
HWYBLAS_LAPACK_HEEVX(z, c64, double)

#undef HWYBLAS_LAPACK_HEEVX

}  // extern "C"
