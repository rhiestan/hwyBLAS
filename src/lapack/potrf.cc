// Cholesky factorization of a symmetric (Hermitian) positive definite
// matrix: xPOTRF (blocked), xPOTRF2 (recursive) and xPOTF2 (unblocked),
// and of a positive semidefinite one with complete pivoting: xPSTRF
// (blocked) and xPSTF2.
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

// xPOTF2: computes the Cholesky factorization of a real symmetric or
// complex Hermitian positive definite matrix A, like xPOTRF2 (see there for
// the result and INFO), with the unblocked algorithm (Level 2 BLAS): it
// computes U row by row (L column by column). (xPOTRF uses xPOTRF2; xPOTF2
// factors the diagonal blocks of xPBTRF.)
template <typename T>
blasint Potf2(char uplo, blasint n, T* a, blasint lda) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  blasint j;
  bool upper;
  R ajj;

  // Test the input parameters.
  blasint info = 0;
  upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("POTF2", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  if (upper) {
    // Compute the Cholesky factorization A = U**H *U.
    for (j = 1; j <= n; ++j) {
      // Compute U(J,J) and test for non-positive-definiteness.
      ajj = std::real(A(j, j)) -
            std::real(Dotc(j - 1, A.ptr(1, j), 1, A.ptr(1, j), 1));
      if (ajj <= zero || std::isnan(ajj)) {
        A(j, j) = ajj;
        goto L30;
      }
      ajj = std::sqrt(ajj);
      A(j, j) = ajj;

      // Compute elements J+1:N of row J.
      if (j < n) {
        Lacgv(j - 1, A.ptr(1, j), 1);
        Gemv('T', j - 1, n - j, -cone, A.ptr(1, j + 1), lda, A.ptr(1, j), 1,
             cone, A.ptr(j, j + 1), lda);
        Lacgv(j - 1, A.ptr(1, j), 1);
        Rscal(n - j, one / ajj, A.ptr(j, j + 1), lda);
      }
    }
  } else {
    // Compute the Cholesky factorization A = L*L**H.
    for (j = 1; j <= n; ++j) {
      // Compute L(J,J) and test for non-positive-definiteness.
      ajj = std::real(A(j, j)) -
            std::real(Dotc(j - 1, A.ptr(j, 1), lda, A.ptr(j, 1), lda));
      if (ajj <= zero || std::isnan(ajj)) {
        A(j, j) = ajj;
        goto L30;
      }
      ajj = std::sqrt(ajj);
      A(j, j) = ajj;

      // Compute elements J+1:N of column J.
      if (j < n) {
        Lacgv(j - 1, A.ptr(j, 1), lda);
        Gemv('N', n - j, j - 1, -cone, A.ptr(j + 1, 1), lda, A.ptr(j, 1), lda,
             cone, A.ptr(j + 1, j), 1);
        Lacgv(j - 1, A.ptr(j, 1), lda);
        Rscal(n - j, one / ajj, A.ptr(j + 1, j), 1);
      }
    }
  }
  return info;

L30:
  info = j;
  return info;
}

// The pivot search of xPSTF2 and xPSTRF at the start: PVT = the index of the
// largest diagonal element of A, AJJ = A(PVT, PVT). The real routines search
// with a loop, the complex ones with MAXLOC of the real parts in WORK(1:N);
// they differ only if a diagonal element is NaN.
template <typename T>
void PstrfFirstPivot(blasint n, const Matrix<T>& A,
                     const Vector<RealT<T>>& WORK, blasint* pvt,
                     RealT<T>* ajj) {
  if constexpr (IsComplex<T>::value) {
    for (blasint i = 1; i <= n; ++i) WORK(i) = std::real(A(i, i));
    *pvt = Maxloc(n, WORK.ptr(1));
    *ajj = std::real(A(*pvt, *pvt));
  } else {
    *pvt = 1;
    *ajj = A(*pvt, *pvt);
    for (blasint i = 2; i <= n; ++i) {
      if (A(i, i) > *ajj) {
        *pvt = i;
        *ajj = A(*pvt, *pvt);
      }
    }
  }
}

// xPSTF2: computes the Cholesky factorization with complete pivoting of a
// real symmetric or complex Hermitian positive semidefinite matrix A of
// order N:
//
//     P**T * A * P = U**H * U  (UPLO = 'U'),  or
//     P**T * A * P = L * L**H  (UPLO = 'L'),
//
// where U is upper and L lower triangular, and P is stored in PIV: column
// PIV(k) of A is column k of A * P. The algorithm stops when the largest
// remaining diagonal element is not larger than TOL (if TOL < 0,
// N * eps * max(A(k, k))); RANK returns the number of steps done, and the
// trailing part of the triangle UPLO is left unfactored. This is the
// unblocked version (Level 2 BLAS). WORK has 2*N elements.
//
// The real routine swaps the part of the pivot row and column between J and
// PVT with xSWAP, the complex one with a loop that also conjugates; for
// real T both give the same.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or 1
// if A is rank deficient (RANK < N) or not positive semidefinite (as far
// as detected; also for a NaN).
template <typename T>
blasint Pstf2(char uplo, blasint n, T* a, blasint lda, blasint* piv,
              blasint* rank, RealT<T> tol, RealT<T>* work) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> PIV(piv);
  const Vector<R> WORK(work);

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
    Xerbla<T>("PSTF2", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Initialize PIV.
  for (blasint i = 1; i <= n; ++i) PIV(i) = i;

  // Compute stopping value.
  blasint pvt;
  R ajj;
  PstrfFirstPivot(n, A, WORK, &pvt, &ajj);
  if (ajj <= zero || std::isnan(ajj)) {
    *rank = 0;
    info = 1;
    return info;
  }

  // Compute stopping value if not supplied.
  const R dstop = tol < zero ? n * Lamch<R>('E') * ajj : tol;

  // Set first half of WORK to zero, holds dot products.
  for (blasint i = 1; i <= n; ++i) WORK(i) = 0;

  blasint j;
  if (upper) {
    // Compute the Cholesky factorization P**T * A * P = U**H * U.
    for (j = 1; j <= n; ++j) {
      // Find pivot, test for exit, else swap rows and columns. Update dot
      // products, compute possible pivots which are stored in the second
      // half of WORK.
      for (blasint i = j; i <= n; ++i) {
        if (j > 1) {
          WORK(i) = WORK(i) + std::real(Conj(A(j - 1, i)) * A(j - 1, i));
        }
        WORK(n + i) = std::real(A(i, i)) - WORK(i);
      }

      if (j > 1) {
        const blasint itemp = Maxloc(n - j + 1, WORK.ptr(n + j));
        pvt = itemp + j - 1;
        ajj = WORK(n + pvt);
        if (ajj <= dstop || std::isnan(ajj)) {
          A(j, j) = ajj;
          goto L190;
        }
      }

      if (j != pvt) {
        // Pivot OK, so can now swap pivot rows and columns.
        A(pvt, pvt) = A(j, j);
        Swap(j - 1, A.ptr(1, j), 1, A.ptr(1, pvt), 1);
        if (pvt < n) {
          Swap(n - pvt, A.ptr(j, pvt + 1), lda, A.ptr(pvt, pvt + 1), lda);
        }
        for (blasint i = j + 1; i <= pvt - 1; ++i) {
          const T ztemp = Conj(A(j, i));
          A(j, i) = Conj(A(i, pvt));
          A(i, pvt) = ztemp;
        }
        A(j, pvt) = Conj(A(j, pvt));

        // Swap dot products and PIV.
        const R dtemp = WORK(j);
        WORK(j) = WORK(pvt);
        WORK(pvt) = dtemp;
        const blasint itemp = PIV(pvt);
        PIV(pvt) = PIV(j);
        PIV(j) = itemp;
      }

      ajj = std::sqrt(ajj);
      A(j, j) = ajj;

      // Compute elements J+1:N of row J.
      if (j < n) {
        Lacgv(j - 1, A.ptr(1, j), 1);
        Gemv('T', j - 1, n - j, -cone, A.ptr(1, j + 1), lda, A.ptr(1, j), 1,
             cone, A.ptr(j, j + 1), lda);
        Lacgv(j - 1, A.ptr(1, j), 1);
        Rscal(n - j, one / ajj, A.ptr(j, j + 1), lda);
      }
    }
  } else {
    // Compute the Cholesky factorization P**T * A * P = L * L**H.
    for (j = 1; j <= n; ++j) {
      // Find pivot, test for exit, else swap rows and columns. Update dot
      // products, compute possible pivots which are stored in the second
      // half of WORK.
      for (blasint i = j; i <= n; ++i) {
        if (j > 1) {
          WORK(i) = WORK(i) + std::real(Conj(A(i, j - 1)) * A(i, j - 1));
        }
        WORK(n + i) = std::real(A(i, i)) - WORK(i);
      }

      if (j > 1) {
        const blasint itemp = Maxloc(n - j + 1, WORK.ptr(n + j));
        pvt = itemp + j - 1;
        ajj = WORK(n + pvt);
        if (ajj <= dstop || std::isnan(ajj)) {
          A(j, j) = ajj;
          goto L190;
        }
      }

      if (j != pvt) {
        // Pivot OK, so can now swap pivot rows and columns.
        A(pvt, pvt) = A(j, j);
        Swap(j - 1, A.ptr(j, 1), lda, A.ptr(pvt, 1), lda);
        if (pvt < n) {
          Swap(n - pvt, A.ptr(pvt + 1, j), 1, A.ptr(pvt + 1, pvt), 1);
        }
        for (blasint i = j + 1; i <= pvt - 1; ++i) {
          const T ztemp = Conj(A(i, j));
          A(i, j) = Conj(A(pvt, i));
          A(pvt, i) = ztemp;
        }
        A(pvt, j) = Conj(A(pvt, j));

        // Swap dot products and PIV.
        const R dtemp = WORK(j);
        WORK(j) = WORK(pvt);
        WORK(pvt) = dtemp;
        const blasint itemp = PIV(pvt);
        PIV(pvt) = PIV(j);
        PIV(j) = itemp;
      }

      ajj = std::sqrt(ajj);
      A(j, j) = ajj;

      // Compute elements J+1:N of column J.
      if (j < n) {
        Lacgv(j - 1, A.ptr(j, 1), lda);
        Gemv('N', n - j, j - 1, -cone, A.ptr(j + 1, 1), lda, A.ptr(j, 1), lda,
             cone, A.ptr(j + 1, j), 1);
        Lacgv(j - 1, A.ptr(j, 1), lda);
        Rscal(n - j, one / ajj, A.ptr(j + 1, j), 1);
      }
    }
  }

  // Ran to completion, A has full rank.
  *rank = n;
  return info;

L190:
  // Rank is number of steps completed. Set INFO = 1 to signal that the
  // factorization cannot be used to solve a system.
  *rank = j - 1;
  info = 1;
  return info;
}

// xPSTRF: computes the Cholesky factorization with complete pivoting of a
// real symmetric or complex Hermitian positive semidefinite matrix A, like
// xPSTF2 (see there for the arguments and INFO). This is the blocked
// version (Level 3 BLAS): it updates the dot products and factors NB
// columns at a time (NB = ILAENV's block size for xPOTRF), then updates
// the trailing matrix with xSYRK / xHERK.
template <typename T>
blasint Pstrf(char uplo, blasint n, T* a, blasint lda, blasint* piv,
              blasint* rank, RealT<T> tol, RealT<T>* work) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> PIV(piv);
  const Vector<R> WORK(work);

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
    Xerbla<T>("PSTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Get block size.
  const char opts[2] = {uplo, '\0'};
  const blasint nb = Ilaenv<T>(1, "POTRF", opts, n, -1, -1, -1);
  if (nb <= 1 || nb >= n) {
    // Use unblocked code.
    return Pstf2(uplo, n, A.ptr(1, 1), lda, piv, rank, tol, work);
  }

  // Initialize PIV.
  for (blasint i = 1; i <= n; ++i) PIV(i) = i;

  // Compute stopping value.
  blasint pvt;
  R ajj;
  PstrfFirstPivot(n, A, WORK, &pvt, &ajj);
  if (ajj <= zero || std::isnan(ajj)) {
    *rank = 0;
    info = 1;
    return info;
  }

  // Compute stopping value if not supplied.
  const R dstop = tol < zero ? n * Lamch<R>('E') * ajj : tol;

  blasint j;
  if (upper) {
    // Compute the Cholesky factorization P**T * A * P = U**H * U.
    for (blasint k = 1; k <= n; k += nb) {
      // Account for last block not being NB wide.
      const blasint jb = Min(nb, n - k + 1);

      // Set relevant part of first half of WORK to zero, holds dot
      // products.
      for (blasint i = k; i <= n; ++i) WORK(i) = 0;

      for (j = k; j <= k + jb - 1; ++j) {
        // Find pivot, test for exit, else swap rows and columns. Update dot
        // products, compute possible pivots which are stored in the second
        // half of WORK.
        for (blasint i = j; i <= n; ++i) {
          if (j > k) {
            WORK(i) = WORK(i) + std::real(Conj(A(j - 1, i)) * A(j - 1, i));
          }
          WORK(n + i) = std::real(A(i, i)) - WORK(i);
        }

        if (j > 1) {
          const blasint itemp = Maxloc(n - j + 1, WORK.ptr(n + j));
          pvt = itemp + j - 1;
          ajj = WORK(n + pvt);
          if (ajj <= dstop || std::isnan(ajj)) {
            A(j, j) = ajj;
            goto L220;
          }
        }

        if (j != pvt) {
          // Pivot OK, so can now swap pivot rows and columns.
          A(pvt, pvt) = A(j, j);
          Swap(j - 1, A.ptr(1, j), 1, A.ptr(1, pvt), 1);
          if (pvt < n) {
            Swap(n - pvt, A.ptr(j, pvt + 1), lda, A.ptr(pvt, pvt + 1), lda);
          }
          for (blasint i = j + 1; i <= pvt - 1; ++i) {
            const T ztemp = Conj(A(j, i));
            A(j, i) = Conj(A(i, pvt));
            A(i, pvt) = ztemp;
          }
          A(j, pvt) = Conj(A(j, pvt));

          // Swap dot products and PIV.
          const R dtemp = WORK(j);
          WORK(j) = WORK(pvt);
          WORK(pvt) = dtemp;
          const blasint itemp = PIV(pvt);
          PIV(pvt) = PIV(j);
          PIV(j) = itemp;
        }

        ajj = std::sqrt(ajj);
        A(j, j) = ajj;

        // Compute elements J+1:N of row J.
        if (j < n) {
          Lacgv(j - 1, A.ptr(1, j), 1);
          Gemv('T', j - k, n - j, -cone, A.ptr(k, j + 1), lda, A.ptr(k, j), 1,
               cone, A.ptr(j, j + 1), lda);
          Lacgv(j - 1, A.ptr(1, j), 1);
          Rscal(n - j, one / ajj, A.ptr(j, j + 1), lda);
        }
      }

      // Update trailing matrix, J already incremented.
      if (k + jb <= n) {
        Herk('U', 'C', n - j + 1, jb, -one, A.ptr(k, j), lda, one, A.ptr(j, j),
             lda);
      }
    }
  } else {
    // Compute the Cholesky factorization P**T * A * P = L * L**H.
    for (blasint k = 1; k <= n; k += nb) {
      // Account for last block not being NB wide.
      const blasint jb = Min(nb, n - k + 1);

      // Set relevant part of first half of WORK to zero, holds dot
      // products.
      for (blasint i = k; i <= n; ++i) WORK(i) = 0;

      for (j = k; j <= k + jb - 1; ++j) {
        // Find pivot, test for exit, else swap rows and columns. Update dot
        // products, compute possible pivots which are stored in the second
        // half of WORK.
        for (blasint i = j; i <= n; ++i) {
          if (j > k) {
            WORK(i) = WORK(i) + std::real(Conj(A(i, j - 1)) * A(i, j - 1));
          }
          WORK(n + i) = std::real(A(i, i)) - WORK(i);
        }

        if (j > 1) {
          const blasint itemp = Maxloc(n - j + 1, WORK.ptr(n + j));
          pvt = itemp + j - 1;
          ajj = WORK(n + pvt);
          if (ajj <= dstop || std::isnan(ajj)) {
            A(j, j) = ajj;
            goto L220;
          }
        }

        if (j != pvt) {
          // Pivot OK, so can now swap pivot rows and columns.
          A(pvt, pvt) = A(j, j);
          Swap(j - 1, A.ptr(j, 1), lda, A.ptr(pvt, 1), lda);
          if (pvt < n) {
            Swap(n - pvt, A.ptr(pvt + 1, j), 1, A.ptr(pvt + 1, pvt), 1);
          }
          for (blasint i = j + 1; i <= pvt - 1; ++i) {
            const T ztemp = Conj(A(i, j));
            A(i, j) = Conj(A(pvt, i));
            A(pvt, i) = ztemp;
          }
          A(pvt, j) = Conj(A(pvt, j));

          // Swap dot products and PIV.
          const R dtemp = WORK(j);
          WORK(j) = WORK(pvt);
          WORK(pvt) = dtemp;
          const blasint itemp = PIV(pvt);
          PIV(pvt) = PIV(j);
          PIV(j) = itemp;
        }

        ajj = std::sqrt(ajj);
        A(j, j) = ajj;

        // Compute elements J+1:N of column J.
        if (j < n) {
          Lacgv(j - 1, A.ptr(j, 1), lda);
          Gemv('N', n - j, j - k, -cone, A.ptr(j + 1, k), lda, A.ptr(j, k), lda,
               cone, A.ptr(j + 1, j), 1);
          Lacgv(j - 1, A.ptr(j, 1), lda);
          Rscal(n - j, one / ajj, A.ptr(j + 1, j), 1);
        }
      }

      // Update trailing matrix, J already incremented.
      if (k + jb <= n) {
        Herk('L', 'N', n - j + 1, jb, -one, A.ptr(j, k), lda, one, A.ptr(j, j),
             lda);
      }
    }
  }

  // Ran to completion, A has full rank.
  *rank = n;
  return info;

L220:
  // Rank is the number of steps completed. Set INFO = 1 to signal that the
  // factorization cannot be used to solve a system.
  *rank = j - 1;
  info = 1;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template blasint Potrf2<T>(char, blasint, T*, blasint);                   \
  template blasint Potrf<T>(char, blasint, T*, blasint);                      template blasint Potf2<T>(char, blasint, T*, blasint);                    \
  template blasint Pstf2<T>(char, blasint, T*, blasint, blasint*, blasint*, \
                            RealT<T>, RealT<T>*);                           \
  template blasint Pstrf<T>(char, blasint, T*, blasint, blasint*, blasint*, \
                            RealT<T>, RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_POTRF(prefix, T, R)                                     \
  HWYBLAS_API void prefix##potrf_(const char* uplo, const hwyblas_int* n,      \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Potrf(*uplo, *n, As<T>(a), *lda);                 \
  }                                                                            \
  HWYBLAS_API void prefix##potrf2_(const char* uplo, const hwyblas_int* n,     \
                                   R* a, const hwyblas_int* lda,               \
                                   hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Potrf2(*uplo, *n, As<T>(a), *lda);                \
  }                                                                            \
  HWYBLAS_API void prefix##potf2_(const char* uplo, const hwyblas_int* n,                                        R* a, const hwyblas_int* lda,                                                  hwyblas_int* info) {                             *info = hwyblas::lapack::Potf2(*uplo, *n, As<T>(a), *lda);                   }                                                                              HWYBLAS_API void prefix##pstf2_(const char* uplo, const hwyblas_int* n,      \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* piv, hwyblas_int* rank,         \
                                  const R* tol, R* work, hwyblas_int* info) {  \
    *info = hwyblas::lapack::Pstf2(*uplo, *n, As<T>(a), *lda, piv, rank, *tol, \
                                   work);                                      \
  }                                                                            \
  HWYBLAS_API void prefix##pstrf_(const char* uplo, const hwyblas_int* n,      \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* piv, hwyblas_int* rank,         \
                                  const R* tol, R* work, hwyblas_int* info) {  \
    *info = hwyblas::lapack::Pstrf(*uplo, *n, As<T>(a), *lda, piv, rank, *tol, \
                                   work);                                      \
  }

HWYBLAS_LAPACK_POTRF(s, float, float)
HWYBLAS_LAPACK_POTRF(d, double, double)
HWYBLAS_LAPACK_POTRF(c, c32, float)
HWYBLAS_LAPACK_POTRF(z, c64, double)

#undef HWYBLAS_LAPACK_POTRF

}  // extern "C"
