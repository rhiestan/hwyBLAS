// Eigenvalues and eigenvectors of symmetric (Hermitian) matrices: xSYEV /
// xHEEV.
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

// xSYEV / xHEEV: computes all eigenvalues and, optionally (JOBZ = 'V'),
// eigenvectors of the N-by-N real symmetric (complex Hermitian) matrix A,
// of which the triangle UPLO is referenced: A is reduced to tridiagonal
// form (xSYTRD / xHETRD), whose eigenvalues are computed with xSTERF, or
// its eigenvectors with xORGTR / xUNGTR and xSTEQR.
//
// On exit, W holds the eigenvalues in ascending order, and A the
// orthonormal eigenvectors (JOBZ = 'V'; destroyed for JOBZ = 'N'). Returns
// INFO: 0, -i for an illegal argument i, or i > 0 if the algorithm failed
// to converge (i off-diagonal elements of an intermediate tridiagonal form
// did not converge to zero).
//
// The real and the complex routine use different workspaces:
//   real T:    LWORK >= max(1, 3*N-1), optimal (NB+2)*N; WORK holds the
//              off-diagonal and TAU, then the workspace of the calls;
//              RWORK is not referenced.
//   complex T: LWORK >= max(1, 2*N-1), optimal (NB+1)*N; RWORK (max(1,
//              3*N-2) elements) holds the off-diagonal and the workspace of
//              xSTEQR.
// LWORK = -1 is a workspace query.
template <typename T>
blasint Heev(char jobz, char uplo, blasint n, T* a, blasint lda, RealT<T>* w,
             T* work, blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const R zero = R(0), one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<R> W(w);
  const Vector<T> WORK(work);
  const char opts[2] = {uplo, '\0'};

  // Test the input parameters.
  const bool wantz = Lsame(jobz, 'V');
  const bool lower = Lsame(uplo, 'L');
  const bool lquery = (lwork == -1);

  blasint info = 0;
  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (!(lower || Lsame(uplo, 'U'))) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < std::max<blasint>(1, n)) {
    info = -5;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    const blasint nb =
        Ilaenv<T>(1, RealOrComplex<T>("SYTRD", "HETRD"), opts, n, -1, -1, -1);
    lwkopt = std::max<blasint>(1, (nb + (kComplex ? 1 : 2)) * n);
    WORK(1) = WorkSize<T>(lwkopt);

    const blasint minimum = kComplex ? 2 * n - 1 : 3 * n - 1;
    if (lwork < std::max<blasint>(1, minimum) && !lquery) info = -8;
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYEV", "HEEV"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) return info;

  if (n == 1) {
    W(1) = std::real(A(1, 1));
    WORK(1) = T(kComplex ? 1 : 2);
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

  // The off-diagonal elements (and in the real routine the norm's
  // workspace): in RWORK for complex T, at the start of WORK for real T.
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
  blasint indwrk = indtau + n;
  const blasint llwork = lwork - indwrk + 1;
  Hetrd(uplo, n, a, lda, w, RW.ptr(inde), WORK.ptr(indtau), WORK.ptr(indwrk),
        llwork);

  // For eigenvalues only, call xSTERF. For eigenvectors, first call xORGTR
  // / xUNGTR to generate the orthogonal (unitary) matrix, then call
  // xSTEQR.
  if (!wantz) {
    info = Sterf(n, w, RW.ptr(inde));
  } else {
    Ungtr(uplo, n, a, lda, WORK.ptr(indtau), WORK.ptr(indwrk), llwork);
    R* steqr_work;
    if constexpr (kComplex) {
      indwrk = inde + n;
      steqr_work = RW.ptr(indwrk);
    } else {
      steqr_work = RW.ptr(indtau);
    }
    info = Steqr(jobz, n, w, RW.ptr(inde), a, lda, steqr_work);
  }

  // If matrix was scaled, then rescale eigenvalues appropriately.
  if (iscale) {
    const blasint imax = (info == 0) ? n : info - 1;
    Scal<R>(imax, one / sigma, w, 1);
  }

  // Set WORK(1) to optimal workspace size.
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template blasint Heev<T>(char, char, blasint, T*, blasint, RealT<T>*, T*, \
                           blasint, RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

HWYBLAS_API void ssyev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* w, float* work, const hwyblas_int* lwork,
                        hwyblas_int* info) {
  *info = hwyblas::lapack::Heev(*jobz, *uplo, *n, a, *lda, w, work, *lwork,
                                static_cast<float*>(nullptr));
}
HWYBLAS_API void dsyev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* w, double* work, const hwyblas_int* lwork,
                        hwyblas_int* info) {
  *info = hwyblas::lapack::Heev(*jobz, *uplo, *n, a, *lda, w, work, *lwork,
                                static_cast<double*>(nullptr));
}
HWYBLAS_API void cheev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* w, float* work, const hwyblas_int* lwork,
                        float* rwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Heev(*jobz, *uplo, *n, As<c32>(a), *lda, w,
                                As<c32>(work), *lwork, rwork);
}
HWYBLAS_API void zheev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* w, double* work, const hwyblas_int* lwork,
                        double* rwork, hwyblas_int* info) {
  *info = hwyblas::lapack::Heev(*jobz, *uplo, *n, As<c64>(a), *lda, w,
                                As<c64>(work), *lwork, rwork);
}

}  // extern "C"
