// Cholesky factorization of a symmetric (Hermitian) positive definite
// matrix: xPOTRF (blocked) and xPOTRF2 (recursive).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/factor_kernels.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xPOTRF2: computes the Cholesky factorization of a real symmetric or
// complex Hermitian positive definite matrix A of order N:
//
//     A = U**H * U  (UPLO = 'U'),  or  A = L * L**H  (UPLO = 'L'),
//
// where U is upper and L is lower triangular. Only the triangle UPLO of A is
// referenced, and overwritten with the factor.
//
// This is the recursive version of the algorithm. It divides the matrix
// into four submatrices:
//
//         [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
//     A = [ -----|----- ]  with n1 = n/2
//         [  A21 | A22  ]       n2 = n-n1
//
// The subroutine calls itself to factor A11. Update and scale A21 or A12,
// update A22 then call itself to factor A22.
//
// Deviation: the recursion stops at matrices of order kPotrfLeaf or less,
// which a kernel factors with the unblocked algorithm (see
// factor_kernels.h): same results up to rounding, several times faster for
// small matrices. The reference's case N = 1 is therefore not reached; it
// is kept for the comparison with the reference.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the leading principal minor of order i is not positive (or NaN),
// and the factorization could not be completed.
template <typename T>
blasint Potrf2(char uplo, blasint n, T* a, blasint lda) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("POTRF2", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Deviation: small matrices are factored by a kernel.
  if (n <= kPotrfLeaf) return PotrfLeaf(upper, n, a, lda);

  if (n == 1) {
    // N=1 case: test for non-positive-definiteness.
    const R ajj = std::real(A(1, 1));
    if (ajj <= zero || std::isnan(ajj)) return 1;
    // Factor.
    A(1, 1) = T(std::sqrt(ajj));
  } else {
    // Use recursive code.
    const blasint n1 = n / 2;
    const blasint n2 = n - n1;

    // Factor A11.
    blasint iinfo = Potrf2(uplo, n1, A.ptr(1, 1), lda);
    if (iinfo != 0) return iinfo;

    if (upper) {
      // Compute the Cholesky factorization A = U**H*U.

      // Update and scale A12.
      Trsm('L', 'U', 'C', 'N', n1, n2, cone, A.ptr(1, 1), lda, A.ptr(1, n1 + 1),
           lda);

      // Update and factor A22.
      Herk(uplo, 'C', n2, n1, -one, A.ptr(1, n1 + 1), lda, one,
           A.ptr(n1 + 1, n1 + 1), lda);
      iinfo = Potrf2(uplo, n2, A.ptr(n1 + 1, n1 + 1), lda);
      if (iinfo != 0) return iinfo + n1;
    } else {
      // Compute the Cholesky factorization A = L*L**H.

      // Update and scale A21.
      Trsm('R', 'L', 'C', 'N', n2, n1, cone, A.ptr(1, 1), lda, A.ptr(n1 + 1, 1),
           lda);

      // Update and factor A22.
      Herk(uplo, 'N', n2, n1, -one, A.ptr(n1 + 1, 1), lda, one,
           A.ptr(n1 + 1, n1 + 1), lda);
      iinfo = Potrf2(uplo, n2, A.ptr(n1 + 1, n1 + 1), lda);
      if (iinfo != 0) return iinfo + n1;
    }
  }
  return info;
}

// xPOTRF: computes the Cholesky factorization of a real symmetric or
// complex Hermitian positive definite matrix A, like xPOTRF2 (see there for
// the result and INFO).
//
// This is the block version of the algorithm, calling Level 3 BLAS: it
// factors diagonal blocks of NB columns (from ILAENV) with xPOTRF2 after
// updating them with xSYRK / xHERK, and the blocks beside them with xGEMM
// and xTRSM.
template <typename T>
blasint Potrf(char uplo, blasint n, T* a, blasint lda) {
  using R = RealT<T>;
  const R one = R(1);
  const T cone = T(1);
  const Matrix<T> A(a, lda);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("POTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Determine the block size for this environment.
  const char opts[2] = {uplo, '\0'};
  const blasint nb = Ilaenv<T>(1, "POTRF", opts, n, -1, -1, -1);
  if (nb <= 1 || nb >= n) {
    // Use unblocked code.
    return Potrf2(uplo, n, a, lda);
  }

  // Use blocked code.
  if (upper) {
    // Compute the Cholesky factorization A = U**H*U.
    for (blasint j = 1; j <= n; j += nb) {
      // Update and factorize the current diagonal block and test for
      // non-positive-definiteness.
      const blasint jb = Min(nb, n - j + 1);
      Herk('U', 'C', jb, j - 1, -one, A.ptr(1, j), lda, one, A.ptr(j, j), lda);
      info = Potrf2('U', jb, A.ptr(j, j), lda);
      if (info != 0) return info + j - 1;
      if (j + jb <= n) {
        // Compute the current block row.
        Gemm('C', 'N', jb, n - j - jb + 1, j - 1, -cone, A.ptr(1, j), lda,
             A.ptr(1, j + jb), lda, cone, A.ptr(j, j + jb), lda);
        Trsm('L', 'U', 'C', 'N', jb, n - j - jb + 1, cone, A.ptr(j, j), lda,
             A.ptr(j, j + jb), lda);
      }
    }
  } else {
    // Compute the Cholesky factorization A = L*L**H.
    for (blasint j = 1; j <= n; j += nb) {
      // Update and factorize the current diagonal block and test for
      // non-positive-definiteness.
      const blasint jb = Min(nb, n - j + 1);
      Herk('L', 'N', jb, j - 1, -one, A.ptr(j, 1), lda, one, A.ptr(j, j), lda);
      info = Potrf2('L', jb, A.ptr(j, j), lda);
      if (info != 0) return info + j - 1;
      if (j + jb <= n) {
        // Compute the current block column.
        Gemm('N', 'C', n - j - jb + 1, jb, j - 1, -cone, A.ptr(j + jb, 1), lda,
             A.ptr(j, 1), lda, cone, A.ptr(j + jb, j), lda);
        Trsm('R', 'L', 'C', 'N', n - j - jb + 1, jb, cone, A.ptr(j, j), lda,
             A.ptr(j + jb, j), lda);
      }
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                     \
  template blasint Potrf2<T>(char, blasint, T*, blasint); \
  template blasint Potrf<T>(char, blasint, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_POTRF(prefix, T, R)                                 \
  HWYBLAS_API void prefix##potrf_(const char* uplo, const hwyblas_int* n,  \
                                  R* a, const hwyblas_int* lda,            \
                                  hwyblas_int* info) {                     \
    *info = hwyblas::lapack::Potrf(*uplo, *n, As<T>(a), *lda);             \
  }                                                                        \
  HWYBLAS_API void prefix##potrf2_(const char* uplo, const hwyblas_int* n, \
                                   R* a, const hwyblas_int* lda,           \
                                   hwyblas_int* info) {                    \
    *info = hwyblas::lapack::Potrf2(*uplo, *n, As<T>(a), *lda);            \
  }

HWYBLAS_LAPACK_POTRF(s, float, float)
HWYBLAS_LAPACK_POTRF(d, double, double)
HWYBLAS_LAPACK_POTRF(c, c32, float)
HWYBLAS_LAPACK_POTRF(z, c64, double)

#undef HWYBLAS_LAPACK_POTRF

}  // extern "C"
