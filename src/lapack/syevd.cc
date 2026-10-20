// Eigenvalues and eigenvectors of symmetric (Hermitian) and symmetric
// tridiagonal matrices by divide and conquer: xSYEVD / xHEEVD and xSTEVD.
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

// xSYEVD / xHEEVD: computes all eigenvalues and, optionally (JOBZ = 'V'),
// eigenvectors of the N-by-N real symmetric (complex Hermitian) matrix A,
// of which the triangle UPLO is referenced: A is reduced to tridiagonal
// form (xSYTRD / xHETRD), whose eigenvalues are computed with xSTERF, or
// its eigenvectors with xSTEDC (divide and conquer), which xORMTR /
// xUNMTR multiplies by the orthogonal (unitary) matrix of the reduction.
//
// On exit, W holds the eigenvalues in ascending order, and A the
// orthonormal eigenvectors (JOBZ = 'V'; destroyed for JOBZ = 'N'). Returns
// INFO: 0, -i for an illegal argument i, or i > 0 if the algorithm failed
// to converge (JOBZ = 'N': i off-diagonal elements of an intermediate
// tridiagonal form did not converge to zero; JOBZ = 'V': the eigenvalue
// i / (N+1) through mod(i, N+1) could not be computed).
//
// The real and the complex routine use different workspaces (N > 1):
//   real T:    LWORK >= 2*N+1 (JOBZ = 'N') or 1+6*N+2*N**2 (JOBZ = 'V'),
//              LIWORK >= 1 or 3+5*N; WORK holds the off-diagonal, TAU and
//              the eigenvectors of the tridiagonal matrix; RWORK and LRWORK
//              are not referenced.
//   complex T: LWORK >= N+1 or 2*N+N**2, LRWORK >= N or 1+5*N+2*N**2,
//              LIWORK >= 1 or 3+5*N; RWORK holds the off-diagonal.
// LWORK, LRWORK or LIWORK = -1 is a workspace query. The optimal LWORK is
// max(LWORK minimum, (2 or 1)*N + N*NB) with the block size NB of xSYTRD /
// xHETRD; the real routine rescales all N eigenvalues also when xSTEDC
// failed, the complex routine only the first INFO-1 (as LAPACK).
template <typename T>
blasint Heevd(char jobz, char uplo, blasint n, T* a, blasint lda, RealT<T>* w,
              T* work, blasint lwork, RealT<T>* rwork, blasint lrwork,
              blasint* iwork, blasint liwork) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const R zero = R(0), one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<R> W(w);
  const Vector<T> WORK(work);
  const Vector<R> RWORK(rwork);
  const Vector<blasint> IWORK(iwork);
  const char opts[2] = {uplo, '\0'};

  // Test the input parameters.
  const bool wantz = Lsame(jobz, 'V');
  const bool lower = Lsame(uplo, 'L');
  bool lquery = (lwork == -1 || liwork == -1);
  if constexpr (kComplex) lquery = lquery || lrwork == -1;

  blasint info = 0;
  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (!(lower || Lsame(uplo, 'U'))) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  }

  blasint lwmin = 1, lrwmin = 1, liwmin = 1, lopt = 1, lropt = 1, liopt = 1;
  if (info == 0) {
    if (n <= 1) {
      liwmin = 1;
      lwmin = 1;
      lrwmin = 1;
      lopt = lwmin;
      lropt = lrwmin;
      liopt = liwmin;
    } else {
      if (wantz) {
        liwmin = 3 + 5 * n;
        if constexpr (kComplex) {
          lwmin = 2 * n + n * n;
          lrwmin = 1 + 5 * n + 2 * (n * n);
        } else {
          lwmin = 1 + 6 * n + 2 * (n * n);
        }
      } else {
        liwmin = 1;
        if constexpr (kComplex) {
          lwmin = n + 1;
          lrwmin = n;
        } else {
          lwmin = 2 * n + 1;
        }
      }
      const blasint nb =
          Ilaenv<T>(1, RealOrComplex<T>("SYTRD", "HETRD"), opts, n, -1, -1, -1);
      lopt = Max(lwmin, (kComplex ? 1 : 2) * n + n * nb);
      lropt = lrwmin;
      liopt = liwmin;
    }
    WORK(1) = WorkSize<T>(lopt);
    if constexpr (kComplex) RWORK(1) = WorkSize<R>(lropt);
    IWORK(1) = liopt;

    if (lwork < lwmin && !lquery) {
      info = -8;
    } else if (kComplex && lrwork < lrwmin && !lquery) {
      info = -10;
    } else if (liwork < liwmin && !lquery) {
      info = kComplex ? -12 : -10;
    }
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYEVD", "HEEVD"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) return info;

  if (n == 1) {
    W(1) = std::real(A(1, 1));
    if (wantz) A(1, 1) = T(one);
    return info;
  }

  // Get machine constants.
  const R safmin = Lamch<R>('S');
  const R eps = Lamch<R>('P');
  const R smlnum = safmin / eps;
  const R bignum = one / smlnum;
  const R rmin = std::sqrt(smlnum);
  const R rmax = std::sqrt(bignum);

  // The off-diagonal elements (and the norm's workspace): in RWORK for
  // complex T, at the start of WORK for real T.
  R* rw;
  if constexpr (kComplex) {
    rw = rwork;
  } else {
    rw = work;
  }
  const Vector<R> RW(rw);

  // Scale matrix to allowable range, if necessary.
  R anrm;
  if constexpr (kComplex) {
    anrm = Lanhe('M', uplo, n, a, lda, rw);
  } else {
    anrm = Lansy('M', uplo, n, a, lda, rw);
  }
  bool iscale = false;
  R sigma = one;
  if (anrm > zero && anrm < rmin) {
    iscale = true;
    sigma = rmin / anrm;
  } else if (anrm > rmax) {
    iscale = true;
    sigma = rmax / anrm;
  }
  if (iscale) Lascl(uplo, 0, 0, one, sigma, n, n, a, lda);

  // Call xSYTRD / xHETRD to reduce the symmetric (Hermitian) matrix to
  // tridiagonal form.
  const blasint inde = 1;
  const blasint indtau = kComplex ? 1 : inde + n;
  const blasint indwrk = indtau + n;
  const blasint indrwk = inde + n;  // Complex T only.
  const blasint indwk2 = indwrk + n * n;
  const blasint llwork = lwork - indwrk + 1;
  const blasint llwrk2 = lwork - indwk2 + 1;
  const blasint llrwk = lrwork - indrwk + 1;  // Complex T only.

  Hetrd(uplo, n, a, lda, w, RW.ptr(inde), WORK.ptr(indtau), WORK.ptr(indwrk),
        llwork);

  // For eigenvalues only, call xSTERF. For eigenvectors, first call xSTEDC
  // to generate the eigenvector matrix, WORK(INDWRK), of the tridiagonal
  // matrix, then call xORMTR / xUNMTR to multiply it by the Householder
  // transformations stored in A.
  if (!wantz) {
    info = Sterf(n, w, RW.ptr(inde));
  } else {
    if constexpr (kComplex) {
      info =
          Stedc('I', n, w, RW.ptr(inde), WORK.ptr(indwrk), n, WORK.ptr(indwk2),
                llwrk2, RWORK.ptr(indrwk), llrwk, iwork, liwork);
    } else {
      info = Stedc('I', n, w, RW.ptr(inde), WORK.ptr(indwrk), n,
                   WORK.ptr(indwk2), llwrk2, iwork, liwork);
    }
    Unmtr('L', uplo, 'N', n, n, a, lda, WORK.ptr(indtau), WORK.ptr(indwrk), n,
          WORK.ptr(indwk2), llwrk2);
    Lacpy('A', n, n, WORK.ptr(indwrk), n, a, lda);
  }

  // If matrix was scaled, then rescale eigenvalues appropriately.
  if (iscale) {
    blasint imax = n;
    if constexpr (kComplex) imax = (info == 0) ? n : info - 1;
    Scal<R>(imax, one / sigma, w, 1);
  }

  WORK(1) = WorkSize<T>(lopt);
  if constexpr (kComplex) RWORK(1) = WorkSize<R>(lropt);
  IWORK(1) = liopt;
  return info;
}

// xSTEVD: computes all eigenvalues and, optionally (JOBZ = 'V'),
// eigenvectors of the real symmetric tridiagonal matrix with diagonal D (N)
// and off-diagonal E (N-1; destroyed): xSTERF, or xSTEDC (divide and
// conquer) with COMPZ = 'I'. On exit, D holds the eigenvalues in ascending
// order and Z the orthonormal eigenvectors (JOBZ = 'V'). LWORK >= 1 and
// LIWORK >= 1, for JOBZ = 'V' and N > 1 LWORK >= 1+4*N+N**2 and LIWORK >=
// 3+5*N; -1 is a workspace query. Returns INFO: 0, -i for an illegal
// argument i, or i > 0 if the algorithm failed to converge. Real R only.
template <typename R>
blasint Stevd(char jobz, blasint n, R* d, R* e, R* z, blasint ldz, R* work,
              blasint lwork, blasint* iwork, blasint liwork) {
  const R zero = R(0), one = R(1);
  const Matrix<R> Z(z, ldz);
  const Vector<R> WORK(work);
  const Vector<blasint> IWORK(iwork);

  const bool wantz = Lsame(jobz, 'V');
  const bool lquery = (lwork == -1 || liwork == -1);

  blasint info = 0;
  blasint liwmin = 1;
  blasint lwmin = 1;
  if (n > 1 && wantz) {
    lwmin = 1 + 4 * n + (n * n);
    liwmin = 3 + 5 * n;
  }

  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldz < 1 || (wantz && ldz < n)) {
    info = -6;
  }

  if (info == 0) {
    WORK(1) = WorkSize<R>(lwmin);
    IWORK(1) = liwmin;

    if (lwork < lwmin && !lquery) {
      info = -8;
    } else if (liwork < liwmin && !lquery) {
      info = -10;
    }
  }

  if (info != 0) {
    Xerbla<R>("STEVD", -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) return info;

  if (n == 1) {
    if (wantz) Z(1, 1) = one;
    return info;
  }

  // Get machine constants.
  const R safmin = Lamch<R>('S');
  const R eps = Lamch<R>('P');
  const R smlnum = safmin / eps;
  const R bignum = one / smlnum;
  const R rmin = std::sqrt(smlnum);
  const R rmax = std::sqrt(bignum);

  // Scale matrix to allowable range, if necessary.
  bool iscale = false;
  R sigma = one;
  const R tnrm = Lanst('M', n, d, e);
  if (tnrm > zero && tnrm < rmin) {
    iscale = true;
    sigma = rmin / tnrm;
  } else if (tnrm > rmax) {
    iscale = true;
    sigma = rmax / tnrm;
  }
  if (iscale) {
    Scal(n, sigma, d, 1);
    Scal(n - 1, sigma, e, 1);
  }

  // For eigenvalues only, call xSTERF. For eigenvalues and eigenvectors,
  // call xSTEDC.
  if (!wantz) {
    info = Sterf(n, d, e);
  } else {
    info = Stedc('I', n, d, e, z, ldz, work, lwork, iwork, liwork);
  }

  // If matrix was scaled, then rescale eigenvalues appropriately.
  if (iscale) Scal(n, one / sigma, d, 1);

  WORK(1) = WorkSize<R>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                        \
  template blasint Heevd<T>(char, char, blasint, T*, blasint, RealT<T>*, T*, \
                            blasint, RealT<T>*, blasint, blasint*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE
template blasint Stevd<float>(char, blasint, float*, float*, float*, blasint,
                              float*, blasint, blasint*, blasint);
template blasint Stevd<double>(char, blasint, double*, double*, double*,
                               blasint, double*, blasint, blasint*, blasint);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_SYEVD_REAL(p, R)                                      \
  HWYBLAS_API void p##syevd_(                                                \
      const char* jobz, const char* uplo, const hwyblas_int* n, R* a,        \
      const hwyblas_int* lda, R* w, R* work, const hwyblas_int* lwork,       \
      hwyblas_int* iwork, const hwyblas_int* liwork, hwyblas_int* info) {    \
    *info =                                                                  \
        hwyblas::lapack::Heevd(*jobz, *uplo, *n, a, *lda, w, work, *lwork,   \
                               static_cast<R*>(nullptr), 0, iwork, *liwork); \
  }                                                                          \
  HWYBLAS_API void p##stevd_(const char* jobz, const hwyblas_int* n, R* d,   \
                             R* e, R* z, const hwyblas_int* ldz, R* work,    \
                             const hwyblas_int* lwork, hwyblas_int* iwork,   \
                             const hwyblas_int* liwork, hwyblas_int* info) { \
    *info = hwyblas::lapack::Stevd(*jobz, *n, d, e, z, *ldz, work, *lwork,   \
                                   iwork, *liwork);                          \
  }

HWYBLAS_LAPACK_SYEVD_REAL(s, float)
HWYBLAS_LAPACK_SYEVD_REAL(d, double)

#undef HWYBLAS_LAPACK_SYEVD_REAL

#define HWYBLAS_LAPACK_HEEVD(p, T, R)                                          \
  HWYBLAS_API void p##heevd_(                                                  \
      const char* jobz, const char* uplo, const hwyblas_int* n, R* a,          \
      const hwyblas_int* lda, R* w, R* work, const hwyblas_int* lwork,         \
      R* rwork, const hwyblas_int* lrwork, hwyblas_int* iwork,                 \
      const hwyblas_int* liwork, hwyblas_int* info) {                          \
    *info = hwyblas::lapack::Heevd(*jobz, *uplo, *n, As<T>(a), *lda, w,        \
                                   As<T>(work), *lwork, rwork, *lrwork, iwork, \
                                   *liwork);                                   \
  }

HWYBLAS_LAPACK_HEEVD(c, c32, float)
HWYBLAS_LAPACK_HEEVD(z, c64, double)

#undef HWYBLAS_LAPACK_HEEVD

}  // extern "C"
