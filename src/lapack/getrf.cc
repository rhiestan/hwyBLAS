// LU factorization with partial pivoting: xGETRF (blocked), xGETRF2
// (recursive) and xLASWP (row interchanges); solution of linear systems
// with it (xGETRS, xGESV) and the inverse (xGETRI).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>
#include <utility>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/factor_kernels.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLASWP: performs a series of row interchanges on the matrix A (N columns).
// One row interchange is initiated for each of rows K1 through K2 of A: row
// I is interchanged with row IPIV(K1 + (I - K1) * |INCX|). For INCX < 0 the
// interchanges are applied in reverse order; INCX = 0 does nothing.
//
// The interchanges are applied to blocks of 32 columns at a time.
template <typename T>
void Laswp(blasint n, T* a, blasint lda, blasint k1, blasint k2,
           const blasint* ipiv, blasint incx) {
  const Matrix<T> A(a, lda);
  const Vector<const blasint> IPIV(ipiv);

  // Interchange row I with row IPIV(K1 + (I - K1) * ABS(INCX)) for each of
  // rows K1 through K2.
  blasint i1, i2, inc, ix0;
  if (incx > 0) {
    ix0 = k1;
    i1 = k1;
    i2 = k2;
    inc = 1;
  } else if (incx < 0) {
    ix0 = k1 + (k1 - k2) * incx;
    i1 = k2;
    i2 = k1;
    inc = -1;
  } else {
    return;
  }
  // DO I = I1, I2, INC.
  const auto in_range = [&](blasint i) { return inc > 0 ? i <= i2 : i >= i2; };

  blasint n32 = (n / 32) * 32;
  if (n32 != 0) {
    for (blasint j = 1; j <= n32; j += 32) {
      blasint ix = ix0;
      for (blasint i = i1; in_range(i); i += inc) {
        const blasint ip = IPIV(ix);
        if (ip != i) {
          for (blasint k = j; k <= j + 31; ++k) {
            std::swap(A(i, k), A(ip, k));
          }
        }
        ix = ix + incx;
      }
    }
  }
  if (n32 != n) {
    n32 = n32 + 1;
    blasint ix = ix0;
    for (blasint i = i1; in_range(i); i += inc) {
      const blasint ip = IPIV(ix);
      if (ip != i) {
        for (blasint k = n32; k <= n; ++k) {
          std::swap(A(i, k), A(ip, k));
        }
      }
      ix = ix + incx;
    }
  }
}

// xGETRF2: computes an LU factorization of a general M-by-N matrix A using
// partial pivoting with row interchanges: A = P * L * U, where P is a
// permutation matrix, L is lower triangular with unit diagonal elements
// (lower trapezoidal if m > n), and U is upper triangular (upper trapezoidal
// if m < n). On exit, A holds L (without its unit diagonal) and U, and
// IPIV(i) (1 <= i <= min(m, n)) is the row interchanged with row i.
//
// This is the recursive version of the algorithm. It divides the matrix
// into four submatrices:
//
//         [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
//     A = [ -----|----- ]  with n1 = min(m,n)/2
//         [  A21 | A22  ]       n2 = n-n1
//
// The routine calls itself to factor [A11; A21], does the swaps on
// [A12; A22], solves A12, updates A22, then calls itself to factor A22 and
// does the swaps on A21.
//
// Deviation: the recursion stops at matrices with kGetrfLeaf rows or
// columns or fewer, which a kernel factors with the unblocked algorithm
// (see factor_kernels.h): the same pivots (chosen as by xIAMAX) and results
// up to rounding, several times faster for small matrices and panels. The
// reference's cases of one row or column are therefore not reached; they
// are kept for the comparison with the reference.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero. The factorization has then been
// completed, but U is exactly singular, and division by zero will occur if
// it is used to solve a system of equations.
template <typename T>
blasint Getrf2(blasint m, blasint n, T* a, blasint lda, blasint* ipiv) {
  using R = RealT<T>;
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("GETRF2", -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) return info;

  // Deviation: matrices with few rows or columns are factored by a kernel.
  if (Min(m, n) <= kGetrfLeaf) {
    return GetrfLeaf(m, n, a, lda, ipiv);
  }

  if (m == 1) {
    // Use unblocked code for one row case. Just need to handle IPIV and
    // INFO.
    IPIV(1) = 1;
    if (A(1, 1) == zero) info = 1;
  } else if (n == 1) {
    // Use unblocked code for one column case.

    // Compute machine safe minimum.
    const R sfmin = Lamch<R>('S');

    // Find pivot and test for singularity.
    const blasint i = Iamax(m, A.ptr(1, 1), 1);
    IPIV(1) = i;
    if (A(i, 1) != zero) {
      // Apply the interchange.
      if (i != 1) std::swap(A(1, 1), A(i, 1));

      // Compute elements 2:M of the column.
      if (std::abs(A(1, 1)) >= sfmin) {
        Scal(m - 1, one / A(1, 1), A.ptr(2, 1), 1);
      } else {
        for (blasint k = 1; k <= m - 1; ++k) {
          A(1 + k, 1) = A(1 + k, 1) / A(1, 1);
        }
      }
    } else {
      info = 1;
    }
  } else {
    // Use recursive code.
    const blasint n1 = Min(m, n) / 2;
    const blasint n2 = n - n1;

    //        [ A11 ]
    // Factor [ --- ]
    //        [ A21 ]
    blasint iinfo = Getrf2(m, n1, A.ptr(1, 1), lda, IPIV.ptr(1));
    if (info == 0 && iinfo > 0) info = iinfo;

    //                       [ A12 ]
    // Apply interchanges to [ --- ]
    //                       [ A22 ]
    Laswp(n2, A.ptr(1, n1 + 1), lda, 1, n1, IPIV.ptr(1), 1);

    // Solve A12.
    Trsm('L', 'L', 'N', 'U', n1, n2, one, A.ptr(1, 1), lda, A.ptr(1, n1 + 1),
         lda);

    // Update A22.
    Gemm('N', 'N', m - n1, n2, n1, -one, A.ptr(n1 + 1, 1), lda,
         A.ptr(1, n1 + 1), lda, one, A.ptr(n1 + 1, n1 + 1), lda);

    // Factor A22.
    iinfo = Getrf2(m - n1, n2, A.ptr(n1 + 1, n1 + 1), lda, IPIV.ptr(n1 + 1));

    // Adjust INFO and the pivot indices.
    if (info == 0 && iinfo > 0) info = iinfo + n1;
    for (blasint i = n1 + 1; i <= Min(m, n); ++i) {
      IPIV(i) = IPIV(i) + n1;
    }

    // Apply interchanges to A21.
    Laswp(n1, A.ptr(1, 1), lda, n1 + 1, Min(m, n), IPIV.ptr(1), 1);
  }
  return info;
}

// xGETRF: computes an LU factorization of a general M-by-N matrix A using
// partial pivoting with row interchanges, like xGETRF2 (see there for the
// result and INFO).
//
// This is the right-looking Level 3 BLAS version of the algorithm: it
// factors panels of NB columns (from ILAENV) with xGETRF2 and updates the
// rest of the matrix with xTRSM and xGEMM.
template <typename T>
blasint Getrf(blasint m, blasint n, T* a, blasint lda, blasint* ipiv) {
  const T one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("GETRF", -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) return info;

  // Determine the block size for this environment.
  const blasint nb = Ilaenv<T>(1, "GETRF", " ", m, n, -1, -1);
  if (nb <= 1 || nb >= Min(m, n)) {
    // Use unblocked code.
    info = Getrf2(m, n, a, lda, ipiv);
  } else {
    // Use blocked code.
    for (blasint j = 1; j <= Min(m, n); j += nb) {
      const blasint jb = Min(Min(m, n) - j + 1, nb);

      // Factor diagonal and subdiagonal blocks and test for exact
      // singularity.
      const blasint iinfo =
          Getrf2(m - j + 1, jb, A.ptr(j, j), lda, IPIV.ptr(j));

      // Adjust INFO and the pivot indices.
      if (info == 0 && iinfo > 0) info = iinfo + j - 1;
      for (blasint i = j; i <= Min(m, j + jb - 1); ++i) {
        IPIV(i) = j - 1 + IPIV(i);
      }

      // Apply interchanges to columns 1:J-1.
      Laswp(j - 1, a, lda, j, j + jb - 1, ipiv, 1);

      if (j + jb <= n) {
        // Apply interchanges to columns J+JB:N.
        Laswp(n - j - jb + 1, A.ptr(1, j + jb), lda, j, j + jb - 1, ipiv, 1);

        // Compute block row of U.
        Trsm('L', 'L', 'N', 'U', jb, n - j - jb + 1, one, A.ptr(j, j), lda,
             A.ptr(j, j + jb), lda);
        if (j + jb <= m) {
          // Update trailing submatrix.
          Gemm('N', 'N', m - j - jb + 1, n - j - jb + 1, jb, -one,
               A.ptr(j + jb, j), lda, A.ptr(j, j + jb), lda, one,
               A.ptr(j + jb, j + jb), lda);
        }
      }
    }
  }
  return info;
}

// xGETRS: solves A * X = B, A**T * X = B or A**H * X = B (TRANS = 'N', 'T'
// or 'C') for the N-by-NRHS matrix X, with the LU factorization of the
// N-by-N matrix A computed by xGETRF (A and IPIV). X overwrites B.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Getrs(char trans, blasint n, blasint nrhs, const T* a, blasint lda,
              const blasint* ipiv, T* b, blasint ldb) {
  const T one = T(1);

  // Test the input parameters.
  blasint info = 0;
  const bool notran = Lsame(trans, 'N');
  if (!notran && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
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
    Xerbla<T>("GETRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  if (notran) {
    // Solve A * X = B.

    // Apply row interchanges to the right hand sides.
    Laswp(nrhs, b, ldb, 1, n, ipiv, 1);

    // Solve L*X = B, overwriting B with X.
    Trsm('L', 'L', 'N', 'U', n, nrhs, one, a, lda, b, ldb);

    // Solve U*X = B, overwriting B with X.
    Trsm('L', 'U', 'N', 'N', n, nrhs, one, a, lda, b, ldb);
  } else {
    // Solve A**T * X = B or A**H * X = B. (The real routine passes 'T'
    // where this passes TRANS, which the BLAS treat alike for real T.)

    // Solve U**T *X = B or U**H *X = B, overwriting B with X.
    Trsm('L', 'U', trans, 'N', n, nrhs, one, a, lda, b, ldb);

    // Solve L**T *X = B, or L**H *X = B overwriting B with X.
    Trsm('L', 'L', trans, 'U', n, nrhs, one, a, lda, b, ldb);

    // Apply row interchanges to the solution vectors.
    Laswp(nrhs, b, ldb, 1, n, ipiv, -1);
  }
  return info;
}

// xGESV: solves A * X = B for the N-by-N matrix A and the N-by-NRHS
// matrices X and B, with the LU factorization of A (xGETRF), which
// overwrites A (with IPIV). X overwrites B.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero: the factorization has been completed,
// but U is exactly singular, so the solution could not be computed.
template <typename T>
blasint Gesv(blasint n, blasint nrhs, T* a, blasint lda, blasint* ipiv, T* b,
             blasint ldb) {
  // Test the input parameters.
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (nrhs < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("GESV", -info);
    return info;
  }

  // Compute the LU factorization of A.
  info = Getrf(n, n, a, lda, ipiv);
  if (info == 0) {
    // Solve the system A*X = B, overwriting B with X.
    info = Getrs('N', n, nrhs, a, lda, ipiv, b, ldb);
  }
  return info;
}

// xGETRI: computes the inverse of the N-by-N matrix A from its LU
// factorization by xGETRF (A and IPIV), which it overwrites. The method
// inverts U and then solves inv(A)*L = inv(U) for inv(A), in blocks of NB
// columns (from ILAENV) if LWORK >= N*NB, else unblocked (LWORK >= N).
// LWORK = -1 is a workspace query: the optimal LWORK is returned in
// WORK(1).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero; the matrix is singular and its inverse
// could not be computed.
template <typename T>
blasint Getri(blasint n, T* a, blasint lda, const blasint* ipiv, T* work,
              blasint lwork) {
  const T zero = T(0), one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<const blasint> IPIV(ipiv);
  const Vector<T> WORK(work);

  // Test the input parameters.
  blasint info = 0;
  blasint nb = Ilaenv<T>(1, "GETRI", " ", n, -1, -1, -1);
  const blasint lwkopt = Max<blasint>(1, n * nb);
  WORK(1) = WorkSize<T>(lwkopt);
  const bool lquery = (lwork == -1);
  if (n < 0) {
    info = -1;
  } else if (lda < Max<blasint>(1, n)) {
    info = -3;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("GETRI", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Form inv(U). If INFO > 0 from xTRTRI, then U is singular, and the
  // inverse is not computed.
  info = Trtri('U', 'N', n, a, lda);
  if (info > 0) return info;

  blasint nbmin = 2;
  const blasint ldwork = n;
  blasint iws;
  if (nb > 1 && nb < n) {
    iws = Max<blasint>(ldwork * nb, 1);
    if (lwork < iws) {
      nb = lwork / ldwork;
      nbmin = Max<blasint>(2, Ilaenv<T>(2, "GETRI", " ", n, -1, -1, -1));
    }
  } else {
    iws = n;
  }

  // Solve the equation inv(A)*L = inv(U) for inv(A).
  if (nb < nbmin || nb >= n) {
    // Use unblocked code.
    for (blasint j = n; j >= 1; --j) {
      // Copy current column of L to WORK and replace with zeros.
      for (blasint i = j + 1; i <= n; ++i) {
        WORK(i) = A(i, j);
        A(i, j) = zero;
      }

      // Compute current column of inv(A).
      if (j < n) {
        Gemv('N', n, n - j, -one, A.ptr(1, j + 1), lda, WORK.ptr(j + 1), 1, one,
             A.ptr(1, j), 1);
      }
    }
  } else {
    // Use blocked code.
    const blasint nn = ((n - 1) / nb) * nb + 1;
    for (blasint j = nn; j >= 1; j -= nb) {
      const blasint jb = Min(nb, n - j + 1);

      // Copy current block column of L to WORK and replace with zeros.
      for (blasint jj = j; jj <= j + jb - 1; ++jj) {
        for (blasint i = jj + 1; i <= n; ++i) {
          WORK(i + (jj - j) * ldwork) = A(i, jj);
          A(i, jj) = zero;
        }
      }

      // Compute current block column of inv(A).
      if (j + jb <= n) {
        Gemm('N', 'N', n, jb, n - j - jb + 1, -one, A.ptr(1, j + jb), lda,
             WORK.ptr(j + jb), ldwork, one, A.ptr(1, j), lda);
      }
      Trsm('R', 'L', 'N', 'U', n, jb, one, WORK.ptr(j), ldwork, A.ptr(1, j),
           lda);
    }
  }

  // Apply column interchanges.
  for (blasint j = n - 1; j >= 1; --j) {
    const blasint jp = IPIV(j);
    if (jp != j) Swap(n, A.ptr(1, j), 1, A.ptr(1, jp), 1);
  }

  WORK(1) = WorkSize<T>(iws);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                   \
  template void Laswp<T>(blasint, T*, blasint, blasint, blasint,        \
                         const blasint*, blasint);                      \
  template blasint Getrf2<T>(blasint, blasint, T*, blasint, blasint*);  \
  template blasint Getrf<T>(blasint, blasint, T*, blasint, blasint*);   \
  template blasint Getrs<T>(char, blasint, blasint, const T*, blasint,  \
                            const blasint*, T*, blasint);               \
  template blasint Gesv<T>(blasint, blasint, T*, blasint, blasint*, T*, \
                           blasint);                                    \
  template blasint Getri<T>(blasint, T*, blasint, const blasint*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GETRF(prefix, T, R)                                     \
  HWYBLAS_API void prefix##getrf_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* ipiv, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Getrf(*m, *n, As<T>(a), *lda, ipiv);              \
  }                                                                            \
  HWYBLAS_API void prefix##getrf2_(const hwyblas_int* m, const hwyblas_int* n, \
                                   R* a, const hwyblas_int* lda,               \
                                   hwyblas_int* ipiv, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Getrf2(*m, *n, As<T>(a), *lda, ipiv);             \
  }                                                                            \
  HWYBLAS_API void prefix##laswp_(                                             \
      const hwyblas_int* n, R* a, const hwyblas_int* lda,                      \
      const hwyblas_int* k1, const hwyblas_int* k2, const hwyblas_int* ipiv,   \
      const hwyblas_int* incx) {                                               \
    hwyblas::lapack::Laswp(*n, As<T>(a), *lda, *k1, *k2, ipiv, *incx);         \
  }                                                                            \
  HWYBLAS_API void prefix##getrs_(                                             \
      const char* trans, const hwyblas_int* n, const hwyblas_int* nrhs,        \
      const R* a, const hwyblas_int* lda, const hwyblas_int* ipiv, R* b,       \
      const hwyblas_int* ldb, hwyblas_int* info) {                             \
    *info = hwyblas::lapack::Getrs(*trans, *n, *nrhs, As<const T>(a), *lda,    \
                                   ipiv, As<T>(b), *ldb);                      \
  }                                                                            \
  HWYBLAS_API void prefix##gesv_(                                              \
      const hwyblas_int* n, const hwyblas_int* nrhs, R* a,                     \
      const hwyblas_int* lda, hwyblas_int* ipiv, R* b, const hwyblas_int* ldb, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Gesv(*n, *nrhs, As<T>(a), *lda, ipiv, As<T>(b),   \
                                  *ldb);                                       \
  }                                                                            \
  HWYBLAS_API void prefix##getri_(                                             \
      const hwyblas_int* n, R* a, const hwyblas_int* lda,                      \
      const hwyblas_int* ipiv, R* work, const hwyblas_int* lwork,              \
      hwyblas_int* info) {                                                     \
    *info =                                                                    \
        hwyblas::lapack::Getri(*n, As<T>(a), *lda, ipiv, As<T>(work), *lwork); \
  }

HWYBLAS_LAPACK_GETRF(s, float, float)
HWYBLAS_LAPACK_GETRF(d, double, double)
HWYBLAS_LAPACK_GETRF(c, c32, float)
HWYBLAS_LAPACK_GETRF(z, c64, double)

#undef HWYBLAS_LAPACK_GETRF

}  // extern "C"
