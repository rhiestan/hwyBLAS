// Symmetric (xSY*) and Hermitian (xHE*, complex T only) indefinite linear
// systems with the Bunch-Kaufman factorization of xSYTRF / xHETRF: the
// solvers xSYTRS / xHETRS (Level 2 BLAS) and xSYTRS2 / xHETRS2 (Level 3
// BLAS, with the conversion xSYCONV), the inverse (xSYTRI, xHETRI), the
// condition number estimate (xSYCON, xHECON), iterative refinement with
// error bounds (xSYRFS, xHERFS) and the drivers xSYSV / xHESV and the
// expert drivers xSYSVX / xHESVX.
//
// The real routines differ from the complex symmetric ones only in |.|
// against CABS1 (except where noted), and in the workspaces of xSYCON,
// xSYRFS and xSYSVX, which are handled as in gesvx.cc (rwork for complex T
// only, iwork for real T only). Where a symmetric and a Hermitian routine
// differ only in the routines they call (xSYTRS2, xSYCON, xSYRFS, xSYSV,
// xSYSVX and their Hermitian counterparts), one template with the
// parameter kHerm serves both.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <cstdlib>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xSYTRS: solves A * X = B for the N-by-NRHS matrix B (overwritten with X),
// with the factorization A = U * D * U**T or L * D * L**T (UPLO) of the
// symmetric matrix A computed by xSYTRF (A and IPIV), column by column
// (Level 2 BLAS).
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Sytrs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const blasint* ipiv, T* b, blasint ldb) {
  const T one = T(1);
  const Matrix<const T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<const blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("SYTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  if (upper) {
    // Solve A*X = B, where A = U*D*U**T.
    //
    // First solve U*D*X = B, overwriting B with X.
    //
    // K is the main loop index, decreasing from N to 1 in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = n;
    while (k >= 1) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(U(K)), where U(K) is the transformation stored in
        // column K of A.
        Geru(k - 1, nrhs, -one, A.ptr(1, k), 1, B.ptr(k, 1), ldb, B.ptr(1, 1),
             ldb);

        // Multiply by the inverse of the diagonal block.
        Scal(nrhs, one / A(k, k), B.ptr(k, 1), ldb);
        k = k - 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K-1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k - 1) Swap(nrhs, B.ptr(k - 1, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(U(K)), where U(K) is the transformation stored in
        // columns K-1 and K of A.
        Geru(k - 2, nrhs, -one, A.ptr(1, k), 1, B.ptr(k, 1), ldb, B.ptr(1, 1),
             ldb);
        Geru(k - 2, nrhs, -one, A.ptr(1, k - 1), 1, B.ptr(k - 1, 1), ldb,
             B.ptr(1, 1), ldb);

        // Multiply by the inverse of the diagonal block.
        const T akm1k = A(k - 1, k);
        const T akm1 = A(k - 1, k - 1) / akm1k;
        const T ak = A(k, k) / akm1k;
        const T denom = akm1 * ak - one;
        for (blasint j = 1; j <= nrhs; ++j) {
          const T bkm1 = B(k - 1, j) / akm1k;
          const T bk = B(k, j) / akm1k;
          B(k - 1, j) = (ak * bkm1 - bk) / denom;
          B(k, j) = (akm1 * bk - bkm1) / denom;
        }
        k = k - 2;
      }
    }

    // Next solve U**T *X = B, overwriting B with X.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    k = 1;
    while (k <= n) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Multiply by inv(U**T(K)), where U(K) is the transformation stored
        // in column K of A.
        Gemv('T', k - 1, nrhs, -one, b, ldb, A.ptr(1, k), 1, one, B.ptr(k, 1),
             ldb);

        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k + 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Multiply by inv(U**T(K+1)), where U(K+1) is the transformation
        // stored in columns K and K+1 of A.
        Gemv('T', k - 1, nrhs, -one, b, ldb, A.ptr(1, k), 1, one, B.ptr(k, 1),
             ldb);
        Gemv('T', k - 1, nrhs, -one, b, ldb, A.ptr(1, k + 1), 1, one,
             B.ptr(k + 1, 1), ldb);

        // Interchange rows K and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k + 2;
      }
    }
  } else {
    // Solve A*X = B, where A = L*D*L**T.
    //
    // First solve L*D*X = B, overwriting B with X.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = 1;
    while (k <= n) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(L(K)), where L(K) is the transformation stored in
        // column K of A.
        if (k < n) {
          Geru(n - k, nrhs, -one, A.ptr(k + 1, k), 1, B.ptr(k, 1), ldb,
               B.ptr(k + 1, 1), ldb);
        }

        // Multiply by the inverse of the diagonal block.
        Scal(nrhs, one / A(k, k), B.ptr(k, 1), ldb);
        k = k + 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K+1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k + 1) Swap(nrhs, B.ptr(k + 1, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(L(K)), where L(K) is the transformation stored in
        // columns K and K+1 of A.
        if (k < n - 1) {
          Geru(n - k - 1, nrhs, -one, A.ptr(k + 2, k), 1, B.ptr(k, 1), ldb,
               B.ptr(k + 2, 1), ldb);
          Geru(n - k - 1, nrhs, -one, A.ptr(k + 2, k + 1), 1, B.ptr(k + 1, 1),
               ldb, B.ptr(k + 2, 1), ldb);
        }

        // Multiply by the inverse of the diagonal block.
        const T akm1k = A(k + 1, k);
        const T akm1 = A(k, k) / akm1k;
        const T ak = A(k + 1, k + 1) / akm1k;
        const T denom = akm1 * ak - one;
        for (blasint j = 1; j <= nrhs; ++j) {
          const T bkm1 = B(k, j) / akm1k;
          const T bk = B(k + 1, j) / akm1k;
          B(k, j) = (ak * bkm1 - bk) / denom;
          B(k + 1, j) = (akm1 * bk - bkm1) / denom;
        }
        k = k + 2;
      }
    }

    // Next solve L**T *X = B, overwriting B with X.
    //
    // K is the main loop index, decreasing from N to 1 in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    k = n;
    while (k >= 1) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Multiply by inv(L**T(K)), where L(K) is the transformation stored
        // in column K of A.
        if (k < n) {
          Gemv('T', n - k, nrhs, -one, B.ptr(k + 1, 1), ldb, A.ptr(k + 1, k), 1,
               one, B.ptr(k, 1), ldb);
        }

        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k - 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Multiply by inv(L**T(K-1)), where L(K-1) is the transformation
        // stored in columns K-1 and K of A.
        if (k < n) {
          Gemv('T', n - k, nrhs, -one, B.ptr(k + 1, 1), ldb, A.ptr(k + 1, k), 1,
               one, B.ptr(k, 1), ldb);
          Gemv('T', n - k, nrhs, -one, B.ptr(k + 1, 1), ldb,
               A.ptr(k + 1, k - 1), 1, one, B.ptr(k - 1, 1), ldb);
        }

        // Interchange rows K and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k - 2;
      }
    }
  }
  return info;
}

// xHETRS (complex T only): solves A * X = B with the factorization
// A = U * D * U**H or L * D * L**H of the Hermitian matrix A computed by
// xHETRF, like xSYTRS (see there).
template <typename T>
blasint Hetrs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const blasint* ipiv, T* b, blasint ldb) {
  using R = RealT<T>;
  const T one = T(1);
  const Matrix<const T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<const blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("HETRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  if (upper) {
    // Solve A*X = B, where A = U*D*U**H.
    //
    // First solve U*D*X = B, overwriting B with X.
    //
    // K is the main loop index, decreasing from N to 1 in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = n;
    while (k >= 1) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(U(K)), where U(K) is the transformation stored in
        // column K of A.
        Geru(k - 1, nrhs, -one, A.ptr(1, k), 1, B.ptr(k, 1), ldb, B.ptr(1, 1),
             ldb);

        // Multiply by the inverse of the diagonal block.
        const R s = std::real(one) / std::real(A(k, k));
        Rscal(nrhs, s, B.ptr(k, 1), ldb);
        k = k - 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K-1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k - 1) Swap(nrhs, B.ptr(k - 1, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(U(K)), where U(K) is the transformation stored in
        // columns K-1 and K of A.
        Geru(k - 2, nrhs, -one, A.ptr(1, k), 1, B.ptr(k, 1), ldb, B.ptr(1, 1),
             ldb);
        Geru(k - 2, nrhs, -one, A.ptr(1, k - 1), 1, B.ptr(k - 1, 1), ldb,
             B.ptr(1, 1), ldb);

        // Multiply by the inverse of the diagonal block.
        const T akm1k = A(k - 1, k);
        const T akm1 = A(k - 1, k - 1) / akm1k;
        const T ak = A(k, k) / Conj(akm1k);
        const T denom = akm1 * ak - one;
        for (blasint j = 1; j <= nrhs; ++j) {
          const T bkm1 = B(k - 1, j) / akm1k;
          const T bk = B(k, j) / Conj(akm1k);
          B(k - 1, j) = (ak * bkm1 - bk) / denom;
          B(k, j) = (akm1 * bk - bkm1) / denom;
        }
        k = k - 2;
      }
    }

    // Next solve U**H *X = B, overwriting B with X.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    k = 1;
    while (k <= n) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Multiply by inv(U**H(K)), where U(K) is the transformation stored
        // in column K of A.
        if (k > 1) {
          Lacgv(nrhs, B.ptr(k, 1), ldb);
          Gemv('C', k - 1, nrhs, -one, b, ldb, A.ptr(1, k), 1, one, B.ptr(k, 1),
               ldb);
          Lacgv(nrhs, B.ptr(k, 1), ldb);
        }

        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k + 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Multiply by inv(U**H(K+1)), where U(K+1) is the transformation
        // stored in columns K and K+1 of A.
        if (k > 1) {
          Lacgv(nrhs, B.ptr(k, 1), ldb);
          Gemv('C', k - 1, nrhs, -one, b, ldb, A.ptr(1, k), 1, one, B.ptr(k, 1),
               ldb);
          Lacgv(nrhs, B.ptr(k, 1), ldb);

          Lacgv(nrhs, B.ptr(k + 1, 1), ldb);
          Gemv('C', k - 1, nrhs, -one, b, ldb, A.ptr(1, k + 1), 1, one,
               B.ptr(k + 1, 1), ldb);
          Lacgv(nrhs, B.ptr(k + 1, 1), ldb);
        }

        // Interchange rows K and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k + 2;
      }
    }
  } else {
    // Solve A*X = B, where A = L*D*L**H.
    //
    // First solve L*D*X = B, overwriting B with X.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = 1;
    while (k <= n) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(L(K)), where L(K) is the transformation stored in
        // column K of A.
        if (k < n) {
          Geru(n - k, nrhs, -one, A.ptr(k + 1, k), 1, B.ptr(k, 1), ldb,
               B.ptr(k + 1, 1), ldb);
        }

        // Multiply by the inverse of the diagonal block.
        const R s = std::real(one) / std::real(A(k, k));
        Rscal(nrhs, s, B.ptr(k, 1), ldb);
        k = k + 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K+1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k + 1) Swap(nrhs, B.ptr(k + 1, 1), ldb, B.ptr(kp, 1), ldb);

        // Multiply by inv(L(K)), where L(K) is the transformation stored in
        // columns K and K+1 of A.
        if (k < n - 1) {
          Geru(n - k - 1, nrhs, -one, A.ptr(k + 2, k), 1, B.ptr(k, 1), ldb,
               B.ptr(k + 2, 1), ldb);
          Geru(n - k - 1, nrhs, -one, A.ptr(k + 2, k + 1), 1, B.ptr(k + 1, 1),
               ldb, B.ptr(k + 2, 1), ldb);
        }

        // Multiply by the inverse of the diagonal block.
        const T akm1k = A(k + 1, k);
        const T akm1 = A(k, k) / Conj(akm1k);
        const T ak = A(k + 1, k + 1) / akm1k;
        const T denom = akm1 * ak - one;
        for (blasint j = 1; j <= nrhs; ++j) {
          const T bkm1 = B(k, j) / Conj(akm1k);
          const T bk = B(k + 1, j) / akm1k;
          B(k, j) = (ak * bkm1 - bk) / denom;
          B(k + 1, j) = (akm1 * bk - bkm1) / denom;
        }
        k = k + 2;
      }
    }

    // Next solve L**H *X = B, overwriting B with X.
    //
    // K is the main loop index, decreasing from N to 1 in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    k = n;
    while (k >= 1) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Multiply by inv(L**H(K)), where L(K) is the transformation stored
        // in column K of A.
        if (k < n) {
          Lacgv(nrhs, B.ptr(k, 1), ldb);
          Gemv('C', n - k, nrhs, -one, B.ptr(k + 1, 1), ldb, A.ptr(k + 1, k), 1,
               one, B.ptr(k, 1), ldb);
          Lacgv(nrhs, B.ptr(k, 1), ldb);
        }

        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k - 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Multiply by inv(L**H(K-1)), where L(K-1) is the transformation
        // stored in columns K-1 and K of A.
        if (k < n) {
          Lacgv(nrhs, B.ptr(k, 1), ldb);
          Gemv('C', n - k, nrhs, -one, B.ptr(k + 1, 1), ldb, A.ptr(k + 1, k), 1,
               one, B.ptr(k, 1), ldb);
          Lacgv(nrhs, B.ptr(k, 1), ldb);

          Lacgv(nrhs, B.ptr(k - 1, 1), ldb);
          Gemv('C', n - k, nrhs, -one, B.ptr(k + 1, 1), ldb,
               A.ptr(k + 1, k - 1), 1, one, B.ptr(k - 1, 1), ldb);
          Lacgv(nrhs, B.ptr(k - 1, 1), ldb);
        }

        // Interchange rows K and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k - 2;
      }
    }
  }
  return info;
}

// xSYCONV: converts the factorization of xSYTRF (or xHETRF) in A to the
// form used by xSYTRS2 and back (WAY = 'C' convert, 'R' revert): moves the
// off-diagonal elements of the 2-by-2 blocks of D from A to E (N elements)
// and applies the interchanges of IPIV to the multipliers, so that A holds
// the unit triangular factor U or L (UPLO) and the diagonal of D.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Syconv(char uplo, char way, blasint n, T* a, blasint lda,
               const blasint* ipiv, T* e) {
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<T> E(e);
  const Vector<const blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool convert = Lsame(way, 'C');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!convert && !Lsame(way, 'R')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("SYCONV", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  if (upper) {
    // A is UPPER.
    if (convert) {
      // Convert A (A is upper).
      //
      // Convert VALUE.
      blasint i = n;
      E(1) = zero;
      while (i > 1) {
        if (IPIV(i) < 0) {
          E(i) = A(i - 1, i);
          E(i - 1) = zero;
          A(i - 1, i) = zero;
          i = i - 1;
        } else {
          E(i) = zero;
        }
        i = i - 1;
      }

      // Convert PERMUTATIONS.
      i = n;
      while (i >= 1) {
        if (IPIV(i) > 0) {
          const blasint ip = IPIV(i);
          if (i < n) {
            for (blasint j = i + 1; j <= n; ++j) {
              const T temp = A(ip, j);
              A(ip, j) = A(i, j);
              A(i, j) = temp;
            }
          }
        } else {
          const blasint ip = -IPIV(i);
          if (i < n) {
            for (blasint j = i + 1; j <= n; ++j) {
              const T temp = A(ip, j);
              A(ip, j) = A(i - 1, j);
              A(i - 1, j) = temp;
            }
          }
          i = i - 1;
        }
        i = i - 1;
      }
    } else {
      // Revert A (A is upper).
      //
      // Revert PERMUTATIONS.
      blasint i = 1;
      while (i <= n) {
        if (IPIV(i) > 0) {
          const blasint ip = IPIV(i);
          if (i < n) {
            for (blasint j = i + 1; j <= n; ++j) {
              const T temp = A(ip, j);
              A(ip, j) = A(i, j);
              A(i, j) = temp;
            }
          }
        } else {
          const blasint ip = -IPIV(i);
          i = i + 1;
          if (i < n) {
            for (blasint j = i + 1; j <= n; ++j) {
              const T temp = A(ip, j);
              A(ip, j) = A(i - 1, j);
              A(i - 1, j) = temp;
            }
          }
        }
        i = i + 1;
      }

      // Revert VALUE.
      i = n;
      while (i > 1) {
        if (IPIV(i) < 0) {
          A(i - 1, i) = E(i);
          i = i - 1;
        }
        i = i - 1;
      }
    }
  } else {
    // A is LOWER.
    if (convert) {
      // Convert A (A is lower).
      //
      // Convert VALUE.
      blasint i = 1;
      E(n) = zero;
      while (i <= n) {
        if (i < n && IPIV(i) < 0) {
          E(i) = A(i + 1, i);
          E(i + 1) = zero;
          A(i + 1, i) = zero;
          i = i + 1;
        } else {
          E(i) = zero;
        }
        i = i + 1;
      }

      // Convert PERMUTATIONS.
      i = 1;
      while (i <= n) {
        if (IPIV(i) > 0) {
          const blasint ip = IPIV(i);
          if (i > 1) {
            for (blasint j = 1; j <= i - 1; ++j) {
              const T temp = A(ip, j);
              A(ip, j) = A(i, j);
              A(i, j) = temp;
            }
          }
        } else {
          const blasint ip = -IPIV(i);
          if (i > 1) {
            for (blasint j = 1; j <= i - 1; ++j) {
              const T temp = A(ip, j);
              A(ip, j) = A(i + 1, j);
              A(i + 1, j) = temp;
            }
          }
          i = i + 1;
        }
        i = i + 1;
      }
    } else {
      // Revert A (A is lower).
      //
      // Revert PERMUTATIONS.
      blasint i = n;
      while (i >= 1) {
        if (IPIV(i) > 0) {
          const blasint ip = IPIV(i);
          if (i > 1) {
            for (blasint j = 1; j <= i - 1; ++j) {
              const T temp = A(i, j);
              A(i, j) = A(ip, j);
              A(ip, j) = temp;
            }
          }
        } else {
          const blasint ip = -IPIV(i);
          i = i - 1;
          if (i > 1) {
            for (blasint j = 1; j <= i - 1; ++j) {
              const T temp = A(i + 1, j);
              A(i + 1, j) = A(ip, j);
              A(ip, j) = temp;
            }
          }
        }
        i = i - 1;
      }

      // Revert VALUE.
      i = 1;
      while (i <= n - 1) {
        if (IPIV(i) < 0) {
          A(i + 1, i) = E(i);
          i = i + 1;
        }
        i = i + 1;
      }
    }
  }
  return info;
}

// xSYTRS2 (kHerm = false) and xHETRS2 (kHerm = true): solve A * X = B with
// the factorization of xSYTRF / xHETRF like xSYTRS / xHETRS, but with Level
// 3 BLAS: A is converted by xSYCONV (and reverted at the end), so that the
// triangular solves can use xTRSM. WORK (N elements) holds the
// off-diagonal elements of D. The Hermitian routine scales by the real
// 1 / D(i, i), divides by conj(D(i+1, i)) where xSYTRS2 divides by
// D(i+1, i), and solves with U**H (L**H).
template <bool kHerm, typename T>
blasint Sytrs2Impl(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
                   const blasint* ipiv, T* b, blasint ldb, T* work) {
  using R = RealT<T>;
  const T one = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<const blasint> IPIV(ipiv);
  const Vector<T> WORK(work);
  // The conjugate of x for xHETRS2, x for xSYTRS2.
  const auto cj = [](T x) {
    if constexpr (kHerm) {
      return Conj(x);
    } else {
      return x;
    }
  };
  // Multiplies row I of B by 1 / D(i, i).
  const auto scale_row = [&](blasint i) {
    if constexpr (kHerm) {
      const R s = std::real(one) / std::real(A(i, i));
      Rscal(nrhs, s, B.ptr(i, 1), ldb);
    } else {
      Scal(nrhs, one / A(i, i), B.ptr(i, 1), ldb);
    }
  };
  const char trans = kHerm ? 'C' : 'T';

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>(kHerm ? "HETRS2" : "SYTRS2", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  // Convert A.
  Syconv(uplo, 'C', n, a, lda, ipiv, work);

  if (upper) {
    // Solve A*X = B, where A = U*D*U**T (U**H).
    //
    // P**T * B
    blasint k = n;
    while (k >= 1) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k - 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K-1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (kp == -IPIV(k - 1)) {
          Swap(nrhs, B.ptr(k - 1, 1), ldb, B.ptr(kp, 1), ldb);
        }
        k = k - 2;
      }
    }

    // Compute (U \P**T * B) -> B    [ (U \P**T * B) ]
    Trsm('L', 'U', 'N', 'U', n, nrhs, one, a, lda, b, ldb);

    // Compute D \ B -> B   [ D \ (U \P**T * B) ]
    blasint i = n;
    while (i >= 1) {
      if (IPIV(i) > 0) {
        scale_row(i);
      } else if (i > 1) {
        if (IPIV(i - 1) == IPIV(i)) {
          const T akm1k = WORK(i);
          const T akm1 = A(i - 1, i - 1) / akm1k;
          const T ak = A(i, i) / cj(akm1k);
          const T denom = akm1 * ak - one;
          for (blasint j = 1; j <= nrhs; ++j) {
            const T bkm1 = B(i - 1, j) / akm1k;
            const T bk = B(i, j) / cj(akm1k);
            B(i - 1, j) = (ak * bkm1 - bk) / denom;
            B(i, j) = (akm1 * bk - bkm1) / denom;
          }
          i = i - 1;
        }
      }
      i = i - 1;
    }

    // Compute (U**T \ B) -> B   [ U**T \ (D \ (U \P**T * B) ) ]
    Trsm('L', 'U', trans, 'U', n, nrhs, one, a, lda, b, ldb);

    // P * B  [ P * (U**T \ (D \ (U \P**T * B) )) ]
    k = 1;
    while (k <= n) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k + 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K-1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (k < n && kp == -IPIV(k + 1)) {
          Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        }
        k = k + 2;
      }
    }
  } else {
    // Solve A*X = B, where A = L*D*L**T (L**H).
    //
    // P**T * B
    blasint k = 1;
    while (k <= n) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k + 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K and -IPIV(K+1).
        const blasint kp = -IPIV(k + 1);
        if (kp == -IPIV(k)) {
          Swap(nrhs, B.ptr(k + 1, 1), ldb, B.ptr(kp, 1), ldb);
        }
        k = k + 2;
      }
    }

    // Compute (L \P**T * B) -> B    [ (L \P**T * B) ]
    Trsm('L', 'L', 'N', 'U', n, nrhs, one, a, lda, b, ldb);

    // Compute D \ B -> B   [ D \ (L \P**T * B) ]
    blasint i = 1;
    while (i <= n) {
      if (IPIV(i) > 0) {
        scale_row(i);
      } else {
        const T akm1k = WORK(i);
        const T akm1 = A(i, i) / cj(akm1k);
        const T ak = A(i + 1, i + 1) / akm1k;
        const T denom = akm1 * ak - one;
        for (blasint j = 1; j <= nrhs; ++j) {
          const T bkm1 = B(i, j) / cj(akm1k);
          const T bk = B(i + 1, j) / akm1k;
          B(i, j) = (ak * bkm1 - bk) / denom;
          B(i + 1, j) = (akm1 * bk - bkm1) / denom;
        }
        i = i + 1;
      }
      i = i + 1;
    }

    // Compute (L**T \ B) -> B   [ L**T \ (D \ (L \P**T * B) ) ]
    Trsm('L', 'L', trans, 'U', n, nrhs, one, a, lda, b, ldb);

    // P * B  [ P * (L**T \ (D \ (L \P**T * B) )) ]
    k = n;
    while (k >= 1) {
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Interchange rows K and IPIV(K).
        const blasint kp = IPIV(k);
        if (kp != k) Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        k = k - 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Interchange rows K-1 and -IPIV(K).
        const blasint kp = -IPIV(k);
        if (k > 1 && kp == -IPIV(k - 1)) {
          Swap(nrhs, B.ptr(k, 1), ldb, B.ptr(kp, 1), ldb);
        }
        k = k - 2;
      }
    }
  }

  // Revert A.
  Syconv(uplo, 'R', n, a, lda, ipiv, work);
  return info;
}

// xSYTRS2: solves A * X = B with the factorization of xSYTRF, with Level 3
// BLAS (see Sytrs2Impl). A is changed, but restored on exit.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Sytrs2(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
               const blasint* ipiv, T* b, blasint ldb, T* work) {
  return Sytrs2Impl<false>(uplo, n, nrhs, a, lda, ipiv, b, ldb, work);
}

// xHETRS2 (complex T only): solves A * X = B with the factorization of
// xHETRF, like xSYTRS2.
template <typename T>
blasint Hetrs2(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
               const blasint* ipiv, T* b, blasint ldb, T* work) {
  return Sytrs2Impl<true>(uplo, n, nrhs, a, lda, ipiv, b, ldb, work);
}

// xSYTRI: computes the inverse of the symmetric matrix A of order N from
// its factorization by xSYTRF (A and IPIV), in place (the triangle UPLO).
// WORK has N elements.
//
// The real routine scales the 2-by-2 blocks of D by T = abs(D(k+1, k)),
// the complex one by T = D(k+1, k); the template keeps both.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if D(i, i) is exactly zero (A is singular; the inverse was not
// computed).
template <typename T>
blasint Sytri(char uplo, blasint n, T* a, blasint lda, const blasint* ipiv,
              T* work) {
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const blasint> IPIV(ipiv);
  // The scaling of a 2-by-2 block of D with off-diagonal element x.
  const auto scaling = [](T x) {
    if constexpr (IsComplex<T>::value) {
      return x;
    } else {
      return std::abs(x);
    }
  };

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
    Xerbla<T>("SYTRI", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Check that the diagonal matrix D is nonsingular.
  if (upper) {
    // Upper triangular storage: examine D from bottom to top.
    for (info = n; info >= 1; --info) {
      if (IPIV(info) > 0 && A(info, info) == zero) return info;
    }
  } else {
    // Lower triangular storage: examine D from top to bottom.
    for (info = 1; info <= n; ++info) {
      if (IPIV(info) > 0 && A(info, info) == zero) return info;
    }
  }
  info = 0;

  if (upper) {
    // Compute inv(A) from the factorization A = U*D*U**T.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = 1;
    while (k <= n) {
      blasint kstep;
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Invert the diagonal block.
        A(k, k) = one / A(k, k);

        // Compute column K of the inverse.
        if (k > 1) {
          Copy(k - 1, A.ptr(1, k), 1, work, 1);
          Symv(uplo, k - 1, -one, a, lda, work, 1, zero, A.ptr(1, k), 1);
          A(k, k) = A(k, k) - Dotu(k - 1, work, 1, A.ptr(1, k), 1);
        }
        kstep = 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Invert the diagonal block.
        const T t = scaling(A(k, k + 1));
        const T ak = A(k, k) / t;
        const T akp1 = A(k + 1, k + 1) / t;
        const T akkp1 = A(k, k + 1) / t;
        const T d = t * (ak * akp1 - one);
        A(k, k) = akp1 / d;
        A(k + 1, k + 1) = ak / d;
        A(k, k + 1) = -akkp1 / d;

        // Compute columns K and K+1 of the inverse.
        if (k > 1) {
          Copy(k - 1, A.ptr(1, k), 1, work, 1);
          Symv(uplo, k - 1, -one, a, lda, work, 1, zero, A.ptr(1, k), 1);
          A(k, k) = A(k, k) - Dotu(k - 1, work, 1, A.ptr(1, k), 1);
          A(k, k + 1) =
              A(k, k + 1) - Dotu(k - 1, A.ptr(1, k), 1, A.ptr(1, k + 1), 1);
          Copy(k - 1, A.ptr(1, k + 1), 1, work, 1);
          Symv(uplo, k - 1, -one, a, lda, work, 1, zero, A.ptr(1, k + 1), 1);
          A(k + 1, k + 1) =
              A(k + 1, k + 1) - Dotu(k - 1, work, 1, A.ptr(1, k + 1), 1);
        }
        kstep = 2;
      }

      const blasint kp = std::abs(IPIV(k));
      if (kp != k) {
        // Interchange rows and columns K and KP in the leading submatrix
        // A(1:k+1,1:k+1).
        Swap(kp - 1, A.ptr(1, k), 1, A.ptr(1, kp), 1);
        Swap(k - kp - 1, A.ptr(kp + 1, k), 1, A.ptr(kp, kp + 1), lda);
        T temp = A(k, k);
        A(k, k) = A(kp, kp);
        A(kp, kp) = temp;
        if (kstep == 2) {
          temp = A(k, k + 1);
          A(k, k + 1) = A(kp, k + 1);
          A(kp, k + 1) = temp;
        }
      }

      k = k + kstep;
    }
  } else {
    // Compute inv(A) from the factorization A = L*D*L**T.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = n;
    while (k >= 1) {
      blasint kstep;
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Invert the diagonal block.
        A(k, k) = one / A(k, k);

        // Compute column K of the inverse.
        if (k < n) {
          Copy(n - k, A.ptr(k + 1, k), 1, work, 1);
          Symv(uplo, n - k, -one, A.ptr(k + 1, k + 1), lda, work, 1, zero,
               A.ptr(k + 1, k), 1);
          A(k, k) = A(k, k) - Dotu(n - k, work, 1, A.ptr(k + 1, k), 1);
        }
        kstep = 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Invert the diagonal block.
        const T t = scaling(A(k, k - 1));
        const T ak = A(k - 1, k - 1) / t;
        const T akp1 = A(k, k) / t;
        const T akkp1 = A(k, k - 1) / t;
        const T d = t * (ak * akp1 - one);
        A(k - 1, k - 1) = akp1 / d;
        A(k, k) = ak / d;
        A(k, k - 1) = -akkp1 / d;

        // Compute columns K-1 and K of the inverse.
        if (k < n) {
          Copy(n - k, A.ptr(k + 1, k), 1, work, 1);
          Symv(uplo, n - k, -one, A.ptr(k + 1, k + 1), lda, work, 1, zero,
               A.ptr(k + 1, k), 1);
          A(k, k) = A(k, k) - Dotu(n - k, work, 1, A.ptr(k + 1, k), 1);
          A(k, k - 1) = A(k, k - 1) -
                        Dotu(n - k, A.ptr(k + 1, k), 1, A.ptr(k + 1, k - 1), 1);
          Copy(n - k, A.ptr(k + 1, k - 1), 1, work, 1);
          Symv(uplo, n - k, -one, A.ptr(k + 1, k + 1), lda, work, 1, zero,
               A.ptr(k + 1, k - 1), 1);
          A(k - 1, k - 1) =
              A(k - 1, k - 1) - Dotu(n - k, work, 1, A.ptr(k + 1, k - 1), 1);
        }
        kstep = 2;
      }

      const blasint kp = std::abs(IPIV(k));
      if (kp != k) {
        // Interchange rows and columns K and KP in the trailing submatrix
        // A(k-1:n,k-1:n).
        if (kp < n) Swap(n - kp, A.ptr(kp + 1, k), 1, A.ptr(kp + 1, kp), 1);
        Swap(kp - k - 1, A.ptr(k + 1, k), 1, A.ptr(kp, k + 1), lda);
        T temp = A(k, k);
        A(k, k) = A(kp, kp);
        A(kp, kp) = temp;
        if (kstep == 2) {
          temp = A(k, k - 1);
          A(k, k - 1) = A(kp, k - 1);
          A(kp, k - 1) = temp;
        }
      }

      k = k - kstep;
    }
  }
  return info;
}

// xHETRI (complex T only): computes the inverse of the Hermitian matrix A
// from its factorization by xHETRF, like xSYTRI (see there).
template <typename T>
blasint Hetri(char uplo, blasint n, T* a, blasint lda, const blasint* ipiv,
              T* work) {
  using R = RealT<T>;
  const R one = R(1);
  const T cone = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const blasint> IPIV(ipiv);

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
    Xerbla<T>("HETRI", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Check that the diagonal matrix D is nonsingular.
  if (upper) {
    // Upper triangular storage: examine D from bottom to top.
    for (info = n; info >= 1; --info) {
      if (IPIV(info) > 0 && A(info, info) == zero) return info;
    }
  } else {
    // Lower triangular storage: examine D from top to bottom.
    for (info = 1; info <= n; ++info) {
      if (IPIV(info) > 0 && A(info, info) == zero) return info;
    }
  }
  info = 0;

  if (upper) {
    // Compute inv(A) from the factorization A = U*D*U**H.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = 1;
    while (k <= n) {
      blasint kstep;
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Invert the diagonal block.
        A(k, k) = one / std::real(A(k, k));

        // Compute column K of the inverse.
        if (k > 1) {
          Copy(k - 1, A.ptr(1, k), 1, work, 1);
          Hemv(uplo, k - 1, -cone, a, lda, work, 1, zero, A.ptr(1, k), 1);
          A(k, k) = A(k, k) - std::real(Dotc(k - 1, work, 1, A.ptr(1, k), 1));
        }
        kstep = 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Invert the diagonal block.
        const R t = std::abs(A(k, k + 1));
        const R ak = std::real(A(k, k)) / t;
        const R akp1 = std::real(A(k + 1, k + 1)) / t;
        const T akkp1 = A(k, k + 1) / t;
        const R d = t * (ak * akp1 - one);
        A(k, k) = akp1 / d;
        A(k + 1, k + 1) = ak / d;
        A(k, k + 1) = -akkp1 / d;

        // Compute columns K and K+1 of the inverse.
        if (k > 1) {
          Copy(k - 1, A.ptr(1, k), 1, work, 1);
          Hemv(uplo, k - 1, -cone, a, lda, work, 1, zero, A.ptr(1, k), 1);
          A(k, k) = A(k, k) - std::real(Dotc(k - 1, work, 1, A.ptr(1, k), 1));
          A(k, k + 1) =
              A(k, k + 1) - Dotc(k - 1, A.ptr(1, k), 1, A.ptr(1, k + 1), 1);
          Copy(k - 1, A.ptr(1, k + 1), 1, work, 1);
          Hemv(uplo, k - 1, -cone, a, lda, work, 1, zero, A.ptr(1, k + 1), 1);
          A(k + 1, k + 1) = A(k + 1, k + 1) -
                            std::real(Dotc(k - 1, work, 1, A.ptr(1, k + 1), 1));
        }
        kstep = 2;
      }

      const blasint kp = std::abs(IPIV(k));
      if (kp != k) {
        // Interchange rows and columns K and KP in the leading submatrix
        // A(1:k+1,1:k+1).
        Swap(kp - 1, A.ptr(1, k), 1, A.ptr(1, kp), 1);
        for (blasint j = kp + 1; j <= k - 1; ++j) {
          const T temp = Conj(A(j, k));
          A(j, k) = Conj(A(kp, j));
          A(kp, j) = temp;
        }
        A(kp, k) = Conj(A(kp, k));
        T temp = A(k, k);
        A(k, k) = A(kp, kp);
        A(kp, kp) = temp;
        if (kstep == 2) {
          temp = A(k, k + 1);
          A(k, k + 1) = A(kp, k + 1);
          A(kp, k + 1) = temp;
        }
      }

      k = k + kstep;
    }
  } else {
    // Compute inv(A) from the factorization A = L*D*L**H.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2,
    // depending on the size of the diagonal blocks.
    blasint k = n;
    while (k >= 1) {
      blasint kstep;
      if (IPIV(k) > 0) {
        // 1 x 1 diagonal block.
        //
        // Invert the diagonal block.
        A(k, k) = one / std::real(A(k, k));

        // Compute column K of the inverse.
        if (k < n) {
          Copy(n - k, A.ptr(k + 1, k), 1, work, 1);
          Hemv(uplo, n - k, -cone, A.ptr(k + 1, k + 1), lda, work, 1, zero,
               A.ptr(k + 1, k), 1);
          A(k, k) =
              A(k, k) - std::real(Dotc(n - k, work, 1, A.ptr(k + 1, k), 1));
        }
        kstep = 1;
      } else {
        // 2 x 2 diagonal block.
        //
        // Invert the diagonal block.
        const R t = std::abs(A(k, k - 1));
        const R ak = std::real(A(k - 1, k - 1)) / t;
        const R akp1 = std::real(A(k, k)) / t;
        const T akkp1 = A(k, k - 1) / t;
        const R d = t * (ak * akp1 - one);
        A(k - 1, k - 1) = akp1 / d;
        A(k, k) = ak / d;
        A(k, k - 1) = -akkp1 / d;

        // Compute columns K-1 and K of the inverse.
        if (k < n) {
          Copy(n - k, A.ptr(k + 1, k), 1, work, 1);
          Hemv(uplo, n - k, -cone, A.ptr(k + 1, k + 1), lda, work, 1, zero,
               A.ptr(k + 1, k), 1);
          A(k, k) =
              A(k, k) - std::real(Dotc(n - k, work, 1, A.ptr(k + 1, k), 1));
          A(k, k - 1) = A(k, k - 1) -
                        Dotc(n - k, A.ptr(k + 1, k), 1, A.ptr(k + 1, k - 1), 1);
          Copy(n - k, A.ptr(k + 1, k - 1), 1, work, 1);
          Hemv(uplo, n - k, -cone, A.ptr(k + 1, k + 1), lda, work, 1, zero,
               A.ptr(k + 1, k - 1), 1);
          A(k - 1, k - 1) =
              A(k - 1, k - 1) -
              std::real(Dotc(n - k, work, 1, A.ptr(k + 1, k - 1), 1));
        }
        kstep = 2;
      }

      const blasint kp = std::abs(IPIV(k));
      if (kp != k) {
        // Interchange rows and columns K and KP in the trailing submatrix
        // A(k-1:n,k-1:n).
        if (kp < n) Swap(n - kp, A.ptr(kp + 1, k), 1, A.ptr(kp + 1, kp), 1);
        for (blasint j = k + 1; j <= kp - 1; ++j) {
          const T temp = Conj(A(j, k));
          A(j, k) = Conj(A(kp, j));
          A(kp, j) = temp;
        }
        A(kp, k) = Conj(A(kp, k));
        T temp = A(k, k);
        A(k, k) = A(kp, kp);
        A(kp, kp) = temp;
        if (kstep == 2) {
          temp = A(k, k - 1);
          A(k, k - 1) = A(kp, k - 1);
          A(kp, k - 1) = temp;
        }
      }

      k = k - kstep;
    }
  }
  return info;
}

// xSYCON (kHerm = false) and xHECON (kHerm = true): estimate the
// reciprocal of the condition number (1-norm) of the symmetric (Hermitian)
// matrix A of order N from its factorization by xSYTRF / xHETRF (A, IPIV)
// and the 1-norm ANORM of the original matrix: RCOND =
// 1 / (ANORM * norm(inv(A))), with norm(inv(A)) estimated by xLACN2. WORK
// has 2*N elements; IWORK (N) is used for real T only.
template <bool kHerm, typename T>
blasint SyconImpl(char uplo, blasint n, const T* a, blasint lda,
                  const blasint* ipiv, RealT<T> anorm, RealT<T>* rcond, T* work,
                  blasint* iwork) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> A(a, lda);
  const Vector<const blasint> IPIV(ipiv);
  const Vector<T> WORK(work);
  blasint isave[3];

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  } else if (anorm < zero) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>(kHerm ? "HECON" : "SYCON", -info);
    return info;
  }

  // Quick return if possible.
  *rcond = zero;
  if (n == 0) {
    *rcond = one;
    return info;
  } else if (anorm <= zero) {
    return info;
  }

  // Check that the diagonal matrix D is nonsingular.
  if (upper) {
    // Upper triangular storage: examine D from bottom to top.
    for (blasint i = n; i >= 1; --i) {
      if (IPIV(i) > 0 && A(i, i) == zero) return info;
    }
  } else {
    // Lower triangular storage: examine D from top to bottom.
    for (blasint i = 1; i <= n; ++i) {
      if (IPIV(i) > 0 && A(i, i) == zero) return info;
    }
  }

  // Estimate the 1-norm of the inverse.
  R ainvnm = zero;
  blasint kase = 0;
  for (;;) {
    if constexpr (IsComplex<T>::value) {
      Lacn2(n, WORK.ptr(n + 1), work, &ainvnm, &kase, isave);
    } else {
      Lacn2(n, WORK.ptr(n + 1), work, iwork, &ainvnm, &kase, isave);
    }
    if (kase == 0) break;

    // Multiply by inv(L*D*L**T) or inv(U*D*U**T) (**H for xHECON).
    if constexpr (kHerm) {
      Hetrs(uplo, n, 1, a, lda, ipiv, work, n);
    } else {
      Sytrs(uplo, n, 1, a, lda, ipiv, work, n);
    }
  }

  // Compute the estimate of the reciprocal condition number.
  if (ainvnm != zero) *rcond = (one / ainvnm) / anorm;
  return info;
}

// xSYCON: estimates the reciprocal condition number of the symmetric matrix
// A from its factorization by xSYTRF (see SyconImpl). iwork (N elements)
// is used for real T only.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Sycon(char uplo, blasint n, const T* a, blasint lda,
              const blasint* ipiv, RealT<T> anorm, RealT<T>* rcond, T* work,
              blasint* iwork) {
  return SyconImpl<false>(uplo, n, a, lda, ipiv, anorm, rcond, work, iwork);
}

// xHECON (complex T only): like xSYCON, from the factorization by xHETRF.
template <typename T>
blasint Hecon(char uplo, blasint n, const T* a, blasint lda,
              const blasint* ipiv, RealT<T> anorm, RealT<T>* rcond, T* work) {
  return SyconImpl<true>(uplo, n, a, lda, ipiv, anorm, rcond, work,
                         static_cast<blasint*>(nullptr));
}

// xSYRFS (kHerm = false) and xHERFS (kHerm = true): improve the computed
// solution X (N-by-NRHS) of A * X = B for the symmetric (Hermitian) A
// (triangle UPLO) by iterative refinement, with the factorization AF, IPIV
// by xSYTRF / xHETRF, and compute error bounds: FERR(j) bounds the relative
// error of column j of X, BERR(j) is its componentwise relative backward
// error.
//
// Workspaces as in xGERFS: the real routine keeps W = abs(A)*abs(X) +
// abs(B) in WORK(1:N), the residual in WORK(N+1:2*N) and xLACN2's V in
// WORK(2*N+1:3*N) (with IWORK), the complex ones W in RWORK and the
// residual and V in WORK(1:2*N). The views below have the complex
// routine's names. The Hermitian routine takes abs(Re(A(k, k))) for the
// diagonal, the symmetric one CABS1(A(k, k)).
template <bool kHerm, typename T>
blasint SyrfsImpl(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
                  const T* af, blasint ldaf, const blasint* ipiv, const T* b,
                  blasint ldb, T* x, blasint ldx, RealT<T>* ferr,
                  RealT<T>* berr, T* work, RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;
  constexpr blasint itmax = 5;
  const R zero = R(0), two = R(2), three = R(3);
  const T one = T(1);
  const Matrix<const T> A(a, lda);
  const Matrix<const T> B(b, ldb);
  const Vector<R> BERR(berr);
  const Vector<R> FERR(ferr);
  const Matrix<T> X(x, ldx);
  T* wk;
  R* rw;
  if constexpr (IsComplex<T>::value) {
    wk = work;
    rw = rwork;
  } else {
    wk = work + n;
    rw = work;
  }
  const Vector<T> WORK(wk);
  const Vector<R> RWORK(rw);
  blasint isave[3];
  // abs(A(k, k)) as the routine computes it.
  const auto abs_diag = [&](blasint k) {
    if constexpr (kHerm) {
      return std::abs(std::real(A(k, k)));
    } else {
      return Cabs1(A(k, k));
    }
  };
  // Solves with the factorization (xSYTRS or xHETRS).
  const auto solve = [&](T* rhs) {
    if constexpr (kHerm) {
      Hetrs(uplo, n, 1, af, ldaf, ipiv, rhs, n);
    } else {
      Sytrs(uplo, n, 1, af, ldaf, ipiv, rhs, n);
    }
  };

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldaf < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -10;
  } else if (ldx < Max<blasint>(1, n)) {
    info = -12;
  }
  if (info != 0) {
    Xerbla<T>(kHerm ? "HERFS" : "SYRFS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) {
    for (blasint j = 1; j <= nrhs; ++j) {
      FERR(j) = zero;
      BERR(j) = zero;
    }
    return info;
  }

  // NZ = maximum number of nonzero elements in each row of A, plus 1.
  const blasint nz = n + 1;
  const R eps = Lamch<R>('E');
  const R safmin = Lamch<R>('S');
  const R safe1 = nz * safmin;
  const R safe2 = safe1 / eps;

  // Do for each right hand side.
  for (blasint j = 1; j <= nrhs; ++j) {
    blasint count = 1;
    R lstres = three;
    R s;
    for (;;) {
      // Loop until stopping criterion is satisfied.
      //
      // Compute residual R = B - A * X.
      Copy(n, B.ptr(1, j), 1, wk, 1);
      if constexpr (kHerm) {
        Hemv(uplo, n, -one, a, lda, X.ptr(1, j), 1, one, wk, 1);
      } else {
        Symv(uplo, n, -one, a, lda, X.ptr(1, j), 1, one, wk, 1);
      }

      // Compute componentwise relative backward error from formula
      //
      // max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
      //
      // where abs(Z) is the componentwise absolute value of the matrix or
      // vector Z. If the i-th component of the denominator is less than
      // SAFE2, then SAFE1 is added to the i-th components of the numerator
      // and denominator before dividing.
      for (blasint i = 1; i <= n; ++i) RWORK(i) = Cabs1(B(i, j));

      // Compute abs(A)*abs(X) + abs(B).
      if (upper) {
        for (blasint k = 1; k <= n; ++k) {
          s = zero;
          const R xk = Cabs1(X(k, j));
          for (blasint i = 1; i <= k - 1; ++i) {
            RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
            s = s + Cabs1(A(i, k)) * Cabs1(X(i, j));
          }
          RWORK(k) = RWORK(k) + abs_diag(k) * xk + s;
        }
      } else {
        for (blasint k = 1; k <= n; ++k) {
          s = zero;
          const R xk = Cabs1(X(k, j));
          RWORK(k) = RWORK(k) + abs_diag(k) * xk;
          for (blasint i = k + 1; i <= n; ++i) {
            RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
            s = s + Cabs1(A(i, k)) * Cabs1(X(i, j));
          }
          RWORK(k) = RWORK(k) + s;
        }
      }
      s = zero;
      for (blasint i = 1; i <= n; ++i) {
        if (RWORK(i) > safe2) {
          s = Max(s, Cabs1(WORK(i)) / RWORK(i));
        } else {
          s = Max(s, (Cabs1(WORK(i)) + safe1) / (RWORK(i) + safe1));
        }
      }
      BERR(j) = s;

      // Test stopping criterion. Continue iterating if
      //    1) The residual BERR(J) is larger than machine epsilon, and
      //    2) BERR(J) decreased by at least a factor of 2 during the last
      //       iteration, and
      //    3) At most ITMAX iterations tried.
      if (BERR(j) > eps && two * BERR(j) <= lstres && count <= itmax) {
        // Update solution and try again.
        solve(wk);
        Axpy(n, one, wk, 1, X.ptr(1, j), 1);
        lstres = BERR(j);
        count = count + 1;
        continue;
      }
      break;
    }

    // Bound error from formula
    //
    // norm(X - XTRUE) / norm(X) .le. FERR =
    // norm( abs(inv(A))*
    //    ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X)
    //
    // where
    //   norm(Z) is the magnitude of the largest component of Z
    //   inv(A) is the inverse of A
    //   abs(Z) is the componentwise absolute value of the matrix or vector Z
    //   NZ is the maximum number of nonzeros in any row of A, plus 1
    //   EPS is machine epsilon
    //
    // The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B)) is
    // incremented by SAFE1 if the i-th component of abs(A)*abs(X) + abs(B)
    // is less than SAFE2.
    //
    // Use xLACN2 to estimate the infinity-norm of the matrix
    //    inv(A) * diag(W),
    // where W = abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) )))
    for (blasint i = 1; i <= n; ++i) {
      if (RWORK(i) > safe2) {
        RWORK(i) = Cabs1(WORK(i)) + nz * eps * RWORK(i);
      } else {
        RWORK(i) = Cabs1(WORK(i)) + nz * eps * RWORK(i) + safe1;
      }
    }

    blasint kase = 0;
    for (;;) {
      if constexpr (IsComplex<T>::value) {
        Lacn2(n, WORK.ptr(n + 1), wk, FERR.ptr(j), &kase, isave);
      } else {
        Lacn2(n, WORK.ptr(n + 1), wk, iwork, FERR.ptr(j), &kase, isave);
      }
      if (kase == 0) break;
      if (kase == 1) {
        // Multiply by diag(W)*inv(A**T) (A**H).
        solve(wk);
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
      } else if (kase == 2) {
        // Multiply by inv(A)*diag(W).
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
        solve(wk);
      }
    }

    // Normalize error.
    lstres = zero;
    for (blasint i = 1; i <= n; ++i) lstres = Max(lstres, Cabs1(X(i, j)));
    if (lstres != zero) FERR(j) = FERR(j) / lstres;
  }
  return info;
}

// xSYRFS: iterative refinement and error bounds for A * X = B with the
// symmetric A and its factorization by xSYTRF (see SyrfsImpl). WORK has
// 3*N elements and IWORK N for real T, WORK 2*N and RWORK N for complex T.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Syrfs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* ferr, RealT<T>* berr,
              T* work, RealT<T>* rwork, blasint* iwork) {
  return SyrfsImpl<false>(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx,
                          ferr, berr, work, rwork, iwork);
}

// xHERFS (complex T only): like xSYRFS, for the Hermitian A and its
// factorization by xHETRF.
template <typename T>
blasint Herfs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* ferr, RealT<T>* berr,
              T* work, RealT<T>* rwork) {
  return SyrfsImpl<true>(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx,
                         ferr, berr, work, rwork,
                         static_cast<blasint*>(nullptr));
}

// xSYSV (kHerm = false) and xHESV (kHerm = true): solve A * X = B for the
// symmetric (Hermitian) matrix A of order N (triangle UPLO) and the
// N-by-NRHS matrix B: factor A with xSYTRF / xHETRF (A and IPIV return the
// factorization) and solve with xSYTRS2 / xHETRS2 (or xSYTRS / xHETRS if
// LWORK < N). LWORK >= 1; LWORK = -1 is a workspace query. xSYSV queries
// xSYTRF for the optimal size, xHESV computes it from ILAENV's block size
// (N*NB; 1 for N = 0).
template <bool kHerm, typename T>
blasint SysvImpl(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
                 blasint* ipiv, T* b, blasint ldb, T* work, blasint lwork) {
  const Vector<T> WORK(work);

  // Test the input parameters.
  blasint info = 0;
  const bool lquery = lwork == -1;
  if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  } else if (lwork < 1 && !lquery) {
    info = -10;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    if (n == 0) {
      lwkopt = 1;
    } else if constexpr (kHerm) {
      const char opts[2] = {uplo, '\0'};
      const blasint nb = Ilaenv<T>(1, "HETRF", opts, n, -1, -1, -1);
      lwkopt = n * nb;
    } else {
      info = Sytrf(uplo, n, a, lda, ipiv, work, -1);
      lwkopt = static_cast<blasint>(std::real(WORK(1)));
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(kHerm ? "HESV" : "SYSV", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Compute the factorization A = U*D*U**T (U**H) or A = L*D*L**T (L**H).
  if constexpr (kHerm) {
    info = Hetrf(uplo, n, a, lda, ipiv, work, lwork);
  } else {
    info = Sytrf(uplo, n, a, lda, ipiv, work, lwork);
  }
  if (info == 0) {
    // Solve the system A*X = B, overwriting B with X.
    if (lwork < n) {
      // Solve with TRS ( Use Level BLAS 2).
      if constexpr (kHerm) {
        info = Hetrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
      } else {
        info = Sytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
      }
    } else {
      // Solve with TRS2 ( Use Level BLAS 3).
      if constexpr (kHerm) {
        info = Hetrs2(uplo, n, nrhs, a, lda, ipiv, b, ldb, work);
      } else {
        info = Sytrs2(uplo, n, nrhs, a, lda, ipiv, b, ldb, work);
      }
    }
  }

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xSYSV: solves A * X = B for the symmetric matrix A with the factorization
// of xSYTRF (see SysvImpl).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if D(i, i) is exactly zero (the factorization has been completed,
// but D is singular and the solution was not computed).
template <typename T>
blasint Sysv(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
             blasint* ipiv, T* b, blasint ldb, T* work, blasint lwork) {
  return SysvImpl<false>(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork);
}

// xHESV (complex T only): like xSYSV, for the Hermitian matrix A with the
// factorization of xHETRF.
template <typename T>
blasint Hesv(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
             blasint* ipiv, T* b, blasint ldb, T* work, blasint lwork) {
  return SysvImpl<true>(uplo, n, nrhs, a, lda, ipiv, b, ldb, work, lwork);
}

// xSYSVX (kHerm = false) and xHESVX (kHerm = true): solve A * X = B for the
// symmetric (Hermitian) matrix A of order N (triangle UPLO) and N-by-NRHS
// matrices X and B with the factorization of xSYTRF / xHETRF, and return
// error bounds and an estimate of the condition number. FACT = 'F': AF and
// IPIV contain the factorization; 'N': it is computed. X is refined with
// xSYRFS / xHERFS. RCOND estimates the reciprocal condition number of A.
//
// The real routine needs LWORK >= max(1, 3*N) and IWORK (N elements), the
// complex ones LWORK >= max(1, 2*N) and RWORK (N); LWORK = -1 is a
// workspace query. The real routine passes WORK to xLANSY, the complex ones
// RWORK to xLANSY / xLANHE.
template <bool kHerm, typename T>
blasint SysvxImpl(char fact, char uplo, blasint n, blasint nrhs, const T* a,
                  blasint lda, T* af, blasint ldaf, blasint* ipiv, const T* b,
                  blasint ldb, T* x, blasint ldx, RealT<T>* rcond,
                  RealT<T>* ferr, RealT<T>* berr, T* work, blasint lwork,
                  RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const Vector<T> WORK(work);
  R* rw;
  if constexpr (IsComplex<T>::value) {
    rw = rwork;
  } else {
    rw = work;
  }

  // Test the input parameters.
  blasint info = 0;
  const bool nofact = Lsame(fact, 'N');
  const bool lquery = lwork == -1;
  const blasint lwkmin =
      IsComplex<T>::value ? Max<blasint>(1, 2 * n) : Max<blasint>(1, 3 * n);
  if (!nofact && !Lsame(fact, 'F')) {
    info = -1;
  } else if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (nrhs < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldaf < Max<blasint>(1, n)) {
    info = -8;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -11;
  } else if (ldx < Max<blasint>(1, n)) {
    info = -13;
  } else if (lwork < lwkmin && !lquery) {
    info = -18;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    lwkopt = lwkmin;
    if (nofact) {
      const char opts[2] = {uplo, '\0'};
      const blasint nb =
          Ilaenv<T>(1, kHerm ? "HETRF" : "SYTRF", opts, n, -1, -1, -1);
      lwkopt = Max(lwkopt, n * nb);
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(kHerm ? "HESVX" : "SYSVX", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  if (nofact) {
    // Compute the factorization A = U*D*U**T (U**H) or A = L*D*L**T
    // (L**H).
    Lacpy(uplo, n, n, a, lda, af, ldaf);
    if constexpr (kHerm) {
      info = Hetrf(uplo, n, af, ldaf, ipiv, work, lwork);
    } else {
      info = Sytrf(uplo, n, af, ldaf, ipiv, work, lwork);
    }

    // Return if INFO is non-zero.
    if (info > 0) {
      *rcond = zero;
      return info;
    }
  }

  // Compute the norm of the matrix A.
  R anorm;
  if constexpr (kHerm) {
    anorm = Lanhe('I', uplo, n, a, lda, rw);
  } else {
    anorm = Lansy('I', uplo, n, a, lda, rw);
  }

  // Compute the reciprocal of the condition number of A.
  SyconImpl<kHerm>(uplo, n, af, ldaf, ipiv, anorm, rcond, work, iwork);

  // Compute the solution vectors X.
  Lacpy('F', n, nrhs, b, ldb, x, ldx);
  if constexpr (kHerm) {
    info = Hetrs(uplo, n, nrhs, af, ldaf, ipiv, x, ldx);
  } else {
    info = Sytrs(uplo, n, nrhs, af, ldaf, ipiv, x, ldx);
  }

  // Use iterative refinement to improve the computed solutions and compute
  // error bounds and backward error estimates for them.
  SyrfsImpl<kHerm>(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr,
                   berr, work, rwork, iwork);

  // Set INFO = N+1 if the matrix is singular to working precision.
  if (*rcond < Lamch<R>('E')) info = n + 1;

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xSYSVX: expert driver for the symmetric matrix A (see SysvxImpl).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, i <= N
// if D(i, i) is exactly zero (the factorization has been completed, but
// the solution and error bounds were not computed; RCOND = 0), or N + 1 if
// RCOND is less than the unit roundoff (the matrix is singular to working
// precision; the solution and error bounds have been computed).
template <typename T>
blasint Sysvx(char fact, char uplo, blasint n, blasint nrhs, const T* a,
              blasint lda, T* af, blasint ldaf, blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* rcond, RealT<T>* ferr,
              RealT<T>* berr, T* work, blasint lwork, RealT<T>* rwork,
              blasint* iwork) {
  return SysvxImpl<false>(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb,
                          x, ldx, rcond, ferr, berr, work, lwork, rwork, iwork);
}

// xHESVX (complex T only): expert driver for the Hermitian matrix A, like
// xSYSVX.
template <typename T>
blasint Hesvx(char fact, char uplo, blasint n, blasint nrhs, const T* a,
              blasint lda, T* af, blasint ldaf, blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* rcond, RealT<T>* ferr,
              RealT<T>* berr, T* work, blasint lwork, RealT<T>* rwork) {
  return SysvxImpl<true>(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x,
                         ldx, rcond, ferr, berr, work, lwork, rwork,
                         static_cast<blasint*>(nullptr));
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template blasint Sytrs<T>(char, blasint, blasint, const T*, blasint,         \
                            const blasint*, T*, blasint);                      \
  template blasint Syconv<T>(char, char, blasint, T*, blasint, const blasint*, \
                             T*);                                              \
  template blasint Sytrs2<T>(char, blasint, blasint, T*, blasint,              \
                             const blasint*, T*, blasint, T*);                 \
  template blasint Sytri<T>(char, blasint, T*, blasint, const blasint*, T*);   \
  template blasint Sycon<T>(char, blasint, const T*, blasint, const blasint*,  \
                            RealT<T>, RealT<T>*, T*, blasint*);                \
  template blasint Syrfs<T>(char, blasint, blasint, const T*, blasint,         \
                            const T*, blasint, const blasint*, const T*,       \
                            blasint, T*, blasint, RealT<T>*, RealT<T>*, T*,    \
                            RealT<T>*, blasint*);                              \
  template blasint Sysv<T>(char, blasint, blasint, T*, blasint, blasint*, T*,  \
                           blasint, T*, blasint);                              \
  template blasint Sysvx<T>(char, char, blasint, blasint, const T*, blasint,   \
                            T*, blasint, blasint*, const T*, blasint, T*,      \
                            blasint, RealT<T>*, RealT<T>*, RealT<T>*, T*,      \
                            blasint, RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Hetrs<T>(char, blasint, blasint, const T*, blasint,        \
                            const blasint*, T*, blasint);                     \
  template blasint Hetrs2<T>(char, blasint, blasint, T*, blasint,             \
                             const blasint*, T*, blasint, T*);                \
  template blasint Hetri<T>(char, blasint, T*, blasint, const blasint*, T*);  \
  template blasint Hecon<T>(char, blasint, const T*, blasint, const blasint*, \
                            RealT<T>, RealT<T>*, T*);                         \
  template blasint Herfs<T>(char, blasint, blasint, const T*, blasint,        \
                            const T*, blasint, const blasint*, const T*,      \
                            blasint, T*, blasint, RealT<T>*, RealT<T>*, T*,   \
                            RealT<T>*);                                       \
  template blasint Hesv<T>(char, blasint, blasint, T*, blasint, blasint*, T*, \
                           blasint, T*, blasint);                             \
  template blasint Hesvx<T>(char, char, blasint, blasint, const T*, blasint,  \
                            T*, blasint, blasint*, const T*, blasint, T*,     \
                            blasint, RealT<T>*, RealT<T>*, RealT<T>*, T*,     \
                            blasint, RealT<T>*);
HWYBLAS_LAPACK_INSTANTIATE(c32)
HWYBLAS_LAPACK_INSTANTIATE(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_SYSV(prefix, T, R)                                      \
  HWYBLAS_API void prefix##sytrs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const hwyblas_int* ipiv, R* b,       \
      const hwyblas_int* ldb, hwyblas_int* info) {                             \
    *info = hwyblas::lapack::Sytrs(*uplo, *n, *nrhs, As<const T>(a), *lda,     \
                                   ipiv, As<T>(b), *ldb);                      \
  }                                                                            \
  HWYBLAS_API void prefix##syconv_(                                            \
      const char* uplo, const char* way, const hwyblas_int* n, R* a,           \
      const hwyblas_int* lda, const hwyblas_int* ipiv, R* e,                   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Syconv(*uplo, *way, *n, As<T>(a), *lda, ipiv,     \
                                    As<T>(e));                                 \
  }                                                                            \
  HWYBLAS_API void prefix##sytrs2_(                                            \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs, R* a,   \
      const hwyblas_int* lda, const hwyblas_int* ipiv, R* b,                   \
      const hwyblas_int* ldb, R* work, hwyblas_int* info) {                    \
    *info = hwyblas::lapack::Sytrs2(*uplo, *n, *nrhs, As<T>(a), *lda, ipiv,    \
                                    As<T>(b), *ldb, As<T>(work));              \
  }                                                                            \
  HWYBLAS_API void prefix##sytri_(                                             \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,    \
      const hwyblas_int* ipiv, R* work, hwyblas_int* info) {                   \
    *info =                                                                    \
        hwyblas::lapack::Sytri(*uplo, *n, As<T>(a), *lda, ipiv, As<T>(work));  \
  }                                                                            \
  HWYBLAS_API void prefix##sysv_(                                              \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs, R* a,   \
      const hwyblas_int* lda, hwyblas_int* ipiv, R* b, const hwyblas_int* ldb, \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Sysv(*uplo, *n, *nrhs, As<T>(a), *lda, ipiv,      \
                                  As<T>(b), *ldb, As<T>(work), *lwork);        \
  }

HWYBLAS_LAPACK_SYSV(s, float, float)
HWYBLAS_LAPACK_SYSV(d, double, double)
HWYBLAS_LAPACK_SYSV(c, c32, float)
HWYBLAS_LAPACK_SYSV(z, c64, double)

#undef HWYBLAS_LAPACK_SYSV

// xSYCON, xSYRFS and xSYSVX take IWORK (real) or RWORK (complex, except
// xSYCON, which takes neither).
#define HWYBLAS_LAPACK_SYSVX_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##sycon_(                                             \
      const char* uplo, const hwyblas_int* n, const R* a,                      \
      const hwyblas_int* lda, const hwyblas_int* ipiv, const R* anorm,         \
      R* rcond, R* work, hwyblas_int* iwork, hwyblas_int* info) {              \
    *info = hwyblas::lapack::Sycon(*uplo, *n, a, *lda, ipiv, *anorm, rcond,    \
                                   work, iwork);                               \
  }                                                                            \
  HWYBLAS_API void prefix##syrfs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const hwyblas_int* ipiv, const R* b,            \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* ferr, R* berr,  \
      R* work, hwyblas_int* iwork, hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Syrfs(*uplo, *n, *nrhs, a, *lda, af, *ldaf, ipiv, \
                                   b, *ldb, x, *ldx, ferr, berr, work,         \
                                   static_cast<R*>(nullptr), iwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##sysvx_(                                             \
      const char* fact, const char* uplo, const hwyblas_int* n,                \
      const hwyblas_int* nrhs, const R* a, const hwyblas_int* lda, R* af,      \
      const hwyblas_int* ldaf, hwyblas_int* ipiv, const R* b,                  \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond, R* ferr, \
      R* berr, R* work, const hwyblas_int* lwork, hwyblas_int* iwork,          \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Sysvx(                                            \
        *fact, *uplo, *n, *nrhs, a, *lda, af, *ldaf, ipiv, b, *ldb, x, *ldx,   \
        rcond, ferr, berr, work, *lwork, static_cast<R*>(nullptr), iwork);     \
  }

#define HWYBLAS_LAPACK_SYSVX_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##sycon_(const char* uplo, const hwyblas_int* n,      \
                                  const R* a, const hwyblas_int* lda,          \
                                  const hwyblas_int* ipiv, const R* anorm,     \
                                  R* rcond, R* work, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Sycon(*uplo, *n, As<const T>(a), *lda, ipiv,      \
                                   *anorm, rcond, As<T>(work),                 \
                                   static_cast<hwyblas_int*>(nullptr));        \
  }                                                                            \
  HWYBLAS_API void prefix##syrfs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const hwyblas_int* ipiv, const R* b,            \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* ferr, R* berr,  \
      R* work, R* rwork, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Syrfs(                                            \
        *uplo, *n, *nrhs, As<const T>(a), *lda, As<const T>(af), *ldaf, ipiv,  \
        As<const T>(b), *ldb, As<T>(x), *ldx, ferr, berr, As<T>(work), rwork,  \
        static_cast<hwyblas_int*>(nullptr));                                   \
  }                                                                            \
  HWYBLAS_API void prefix##sysvx_(                                             \
      const char* fact, const char* uplo, const hwyblas_int* n,                \
      const hwyblas_int* nrhs, const R* a, const hwyblas_int* lda, R* af,      \
      const hwyblas_int* ldaf, hwyblas_int* ipiv, const R* b,                  \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond, R* ferr, \
      R* berr, R* work, const hwyblas_int* lwork, R* rwork,                    \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Sysvx(                                            \
        *fact, *uplo, *n, *nrhs, As<const T>(a), *lda, As<T>(af), *ldaf, ipiv, \
        As<const T>(b), *ldb, As<T>(x), *ldx, rcond, ferr, berr, As<T>(work),  \
        *lwork, rwork, static_cast<hwyblas_int*>(nullptr));                    \
  }                                                                            \
  HWYBLAS_API void prefix##hetrs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const hwyblas_int* ipiv, R* b,       \
      const hwyblas_int* ldb, hwyblas_int* info) {                             \
    *info = hwyblas::lapack::Hetrs(*uplo, *n, *nrhs, As<const T>(a), *lda,     \
                                   ipiv, As<T>(b), *ldb);                      \
  }                                                                            \
  HWYBLAS_API void prefix##hetrs2_(                                            \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs, R* a,   \
      const hwyblas_int* lda, const hwyblas_int* ipiv, R* b,                   \
      const hwyblas_int* ldb, R* work, hwyblas_int* info) {                    \
    *info = hwyblas::lapack::Hetrs2(*uplo, *n, *nrhs, As<T>(a), *lda, ipiv,    \
                                    As<T>(b), *ldb, As<T>(work));              \
  }                                                                            \
  HWYBLAS_API void prefix##hetri_(                                             \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,    \
      const hwyblas_int* ipiv, R* work, hwyblas_int* info) {                   \
    *info =                                                                    \
        hwyblas::lapack::Hetri(*uplo, *n, As<T>(a), *lda, ipiv, As<T>(work));  \
  }                                                                            \
  HWYBLAS_API void prefix##hecon_(const char* uplo, const hwyblas_int* n,      \
                                  const R* a, const hwyblas_int* lda,          \
                                  const hwyblas_int* ipiv, const R* anorm,     \
                                  R* rcond, R* work, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Hecon(*uplo, *n, As<const T>(a), *lda, ipiv,      \
                                   *anorm, rcond, As<T>(work));                \
  }                                                                            \
  HWYBLAS_API void prefix##herfs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const hwyblas_int* ipiv, const R* b,            \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* ferr, R* berr,  \
      R* work, R* rwork, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Herfs(                                            \
        *uplo, *n, *nrhs, As<const T>(a), *lda, As<const T>(af), *ldaf, ipiv,  \
        As<const T>(b), *ldb, As<T>(x), *ldx, ferr, berr, As<T>(work), rwork); \
  }                                                                            \
  HWYBLAS_API void prefix##hesv_(                                              \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs, R* a,   \
      const hwyblas_int* lda, hwyblas_int* ipiv, R* b, const hwyblas_int* ldb, \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Hesv(*uplo, *n, *nrhs, As<T>(a), *lda, ipiv,      \
                                  As<T>(b), *ldb, As<T>(work), *lwork);        \
  }                                                                            \
  HWYBLAS_API void prefix##hesvx_(                                             \
      const char* fact, const char* uplo, const hwyblas_int* n,                \
      const hwyblas_int* nrhs, const R* a, const hwyblas_int* lda, R* af,      \
      const hwyblas_int* ldaf, hwyblas_int* ipiv, const R* b,                  \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond, R* ferr, \
      R* berr, R* work, const hwyblas_int* lwork, R* rwork,                    \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Hesvx(                                            \
        *fact, *uplo, *n, *nrhs, As<const T>(a), *lda, As<T>(af), *ldaf, ipiv, \
        As<const T>(b), *ldb, As<T>(x), *ldx, rcond, ferr, berr, As<T>(work),  \
        *lwork, rwork);                                                        \
  }

HWYBLAS_LAPACK_SYSVX_REAL(s, float)
HWYBLAS_LAPACK_SYSVX_REAL(d, double)
HWYBLAS_LAPACK_SYSVX_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_SYSVX_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_SYSVX_REAL
#undef HWYBLAS_LAPACK_SYSVX_COMPLEX

}  // extern "C"
