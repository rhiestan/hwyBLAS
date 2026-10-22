// Symmetric (Hermitian) positive definite band matrices: the Cholesky
// factorization (xPBTF2 unblocked, xPBTRF blocked) and the solvers xPBTRS
// and xPBSV.
//
// The band matrix with KD super- or subdiagonals is stored as in LAPACK:
// for UPLO = 'U', A(i, j) in AB(KD+1+i-j, j) for max(1, j-KD) <= i <= j;
// for UPLO = 'L', A(i, j) in AB(1+i-j, j) for j <= i <= min(N, j+KD).
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

// xPBTF2: computes the Cholesky factorization A = U**H * U (UPLO = 'U') or
// A = L * L**H ('L') of the symmetric (Hermitian) positive definite band
// matrix A of order N with KD super- or subdiagonals (band storage in AB,
// LDAB >= KD+1, overwritten with the factor), with the unblocked algorithm
// (Level 2 BLAS).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the leading principal minor of order i is not positive, and the
// factorization could not be completed.
template <typename T>
blasint Pbtf2(char uplo, blasint n, blasint kd, T* ab, blasint ldab) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<T> AB(ab, ldab);
  blasint j, kld, kn;
  bool upper;
  R ajj;

  // Test the input parameters.
  blasint info = 0;
  upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kd < 0) {
    info = -3;
  } else if (ldab < kd + 1) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("PBTF2", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  kld = Max<blasint>(1, ldab - 1);

  if (upper) {
    // Compute the Cholesky factorization A = U**H * U.
    for (j = 1; j <= n; ++j) {
      // Compute U(J,J) and test for non-positive-definiteness. (The real
      // routine does not store AJJ, which is AB(KD+1,J) there.)
      ajj = std::real(AB(kd + 1, j));
      if (ajj <= zero) {
        AB(kd + 1, j) = ajj;
        goto L30;
      }
      ajj = std::sqrt(ajj);
      AB(kd + 1, j) = ajj;

      // Compute elements J+1:J+KN of row J and update the trailing
      // submatrix within the band.
      kn = Min(kd, n - j);
      if (kn > 0) {
        Rscal(kn, one / ajj, AB.ptr(kd, j + 1), kld);
        Lacgv(kn, AB.ptr(kd, j + 1), kld);
        Her('U', kn, -one, AB.ptr(kd, j + 1), kld, AB.ptr(kd + 1, j + 1),
            kld);
        Lacgv(kn, AB.ptr(kd, j + 1), kld);
      }
    }
  } else {
    // Compute the Cholesky factorization A = L*L**H.
    for (j = 1; j <= n; ++j) {
      // Compute L(J,J) and test for non-positive-definiteness.
      ajj = std::real(AB(1, j));
      if (ajj <= zero) {
        AB(1, j) = ajj;
        goto L30;
      }
      ajj = std::sqrt(ajj);
      AB(1, j) = ajj;

      // Compute elements J+1:J+KN of column J and update the trailing
      // submatrix within the band.
      kn = Min(kd, n - j);
      if (kn > 0) {
        Rscal(kn, one / ajj, AB.ptr(2, j), 1);
        Her('L', kn, -one, AB.ptr(2, j), 1, AB.ptr(1, j + 1), kld);
      }
    }
  }
  return info;

L30:
  info = j;
  return info;
}

// xPBTRF: computes the Cholesky factorization of the symmetric (Hermitian)
// positive definite band matrix A as xPBTF2, with the blocked algorithm
// (Level 3 BLAS, diagonal blocks factored by xPOTF2) if the ILAENV block
// size NB is > 1 and <= KD.
//
// Returns INFO as Pbtf2.
template <typename T>
blasint Pbtrf(char uplo, blasint n, blasint kd, T* ab, blasint ldab) {
  using R = RealT<T>;
  const R one = R(1);
  const T zero = T(0), cone = T(1);
  constexpr blasint nbmax = 32, ldwork = nbmax + 1;
  const Matrix<T> AB(ab, ldab);
  // The local array WORK(LDWORK, NBMAX).
  T work[ldwork * nbmax] = {};
  const Matrix<T> WORK(work, ldwork);
  blasint i, i2, i3, ib, ii, j, jj, nb;

  // Test the input parameters.
  blasint info = 0;
  if ((!Lsame(uplo, 'U')) && (!Lsame(uplo, 'L'))) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kd < 0) {
    info = -3;
  } else if (ldab < kd + 1) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("PBTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Determine the block size for this environment.
  const char opts[2] = {uplo, '\0'};
  nb = Ilaenv<T>(1, "PBTRF", opts, n, kd, -1, -1);

  // The block size must not exceed the semi-bandwidth KD, and must not
  // exceed the limit set by the size of the local array WORK.
  nb = Min(nb, nbmax);

  if (nb <= 1 || nb > kd) {
    // Use unblocked code.
    info = Pbtf2(uplo, n, kd, ab, ldab);
  } else {
    // Use blocked code.
    if (Lsame(uplo, 'U')) {
      // Compute the Cholesky factorization of a symmetric (Hermitian) band
      // matrix, given the upper triangle of the matrix in band storage.

      // Zero the upper triangle of the work array.
      for (j = 1; j <= nb; ++j) {
        for (i = 1; i <= j - 1; ++i) WORK(i, j) = zero;
      }

      // Process the band matrix one diagonal block at a time.
      for (i = 1; i <= n; i += nb) {
        ib = Min(nb, n - i + 1);

        // Factorize the diagonal block.
        ii = Potf2(uplo, ib, AB.ptr(kd + 1, i), ldab - 1);
        if (ii != 0) {
          info = i + ii - 1;
          return info;
        }
        if (i + ib <= n) {
          // Update the relevant part of the trailing submatrix. If A11
          // denotes the diagonal block which has just been factorized, then
          // we need to update the remaining blocks in the diagram:
          //
          //    A11   A12   A13
          //          A22   A23
          //                A33
          //
          // The numbers of rows and columns in the partitioning are IB,
          // I2, I3 respectively. The blocks A12, A22 and A23 are empty if
          // IB = KD. The upper triangle of A13 lies outside the band.
          i2 = Min(kd - ib, n - i - ib + 1);
          i3 = Min(ib, n - i - kd + 1);

          if (i2 > 0) {
            // Update A12.
            Trsm('L', 'U', 'C', 'N', ib, i2, cone, AB.ptr(kd + 1, i),
                 ldab - 1, AB.ptr(kd + 1 - ib, i + ib), ldab - 1);

            // Update A22.
            Herk('U', 'C', i2, ib, -one, AB.ptr(kd + 1 - ib, i + ib),
                 ldab - 1, one, AB.ptr(kd + 1, i + ib), ldab - 1);
          }

          if (i3 > 0) {
            // Copy the lower triangle of A13 into the work array.
            for (jj = 1; jj <= i3; ++jj) {
              for (ii = jj; ii <= ib; ++ii) {
                WORK(ii, jj) = AB(ii - jj + 1, jj + i + kd - 1);
              }
            }

            // Update A13 (in the work array).
            Trsm('L', 'U', 'C', 'N', ib, i3, cone, AB.ptr(kd + 1, i),
                 ldab - 1, work, ldwork);

            // Update A23.
            if (i2 > 0) {
              Gemm('C', 'N', i2, i3, ib, -cone, AB.ptr(kd + 1 - ib, i + ib),
                   ldab - 1, work, ldwork, cone, AB.ptr(1 + ib, i + kd),
                   ldab - 1);
            }

            // Update A33.
            Herk('U', 'C', i3, ib, -one, work, ldwork, one,
                 AB.ptr(kd + 1, i + kd), ldab - 1);

            // Copy the lower triangle of A13 back into place.
            for (jj = 1; jj <= i3; ++jj) {
              for (ii = jj; ii <= ib; ++ii) {
                AB(ii - jj + 1, jj + i + kd - 1) = WORK(ii, jj);
              }
            }
          }
        }
      }
    } else {
      // Compute the Cholesky factorization of a symmetric (Hermitian) band
      // matrix, given the lower triangle of the matrix in band storage.

      // Zero the lower triangle of the work array.
      for (j = 1; j <= nb; ++j) {
        for (i = j + 1; i <= nb; ++i) WORK(i, j) = zero;
      }

      // Process the band matrix one diagonal block at a time.
      for (i = 1; i <= n; i += nb) {
        ib = Min(nb, n - i + 1);

        // Factorize the diagonal block.
        ii = Potf2(uplo, ib, AB.ptr(1, i), ldab - 1);
        if (ii != 0) {
          info = i + ii - 1;
          return info;
        }
        if (i + ib <= n) {
          // Update the relevant part of the trailing submatrix. If A11
          // denotes the diagonal block which has just been factorized, then
          // we need to update the remaining blocks in the diagram:
          //
          //    A11
          //    A21   A22
          //    A31   A32   A33
          //
          // The numbers of rows and columns in the partitioning are IB,
          // I2, I3 respectively. The blocks A21, A22 and A32 are empty if
          // IB = KD. The lower triangle of A31 lies outside the band.
          i2 = Min(kd - ib, n - i - ib + 1);
          i3 = Min(ib, n - i - kd + 1);

          if (i2 > 0) {
            // Update A21.
            Trsm('R', 'L', 'C', 'N', i2, ib, cone, AB.ptr(1, i), ldab - 1,
                 AB.ptr(1 + ib, i), ldab - 1);

            // Update A22.
            Herk('L', 'N', i2, ib, -one, AB.ptr(1 + ib, i), ldab - 1, one,
                 AB.ptr(1, i + ib), ldab - 1);
          }

          if (i3 > 0) {
            // Copy the upper triangle of A31 into the work array.
            for (jj = 1; jj <= ib; ++jj) {
              for (ii = 1; ii <= Min(jj, i3); ++ii) {
                WORK(ii, jj) = AB(kd + 1 - jj + ii, jj + i - 1);
              }
            }

            // Update A31 (in the work array).
            Trsm('R', 'L', 'C', 'N', i3, ib, cone, AB.ptr(1, i), ldab - 1,
                 work, ldwork);

            // Update A32.
            if (i2 > 0) {
              Gemm('N', 'C', i3, i2, ib, -cone, work, ldwork,
                   AB.ptr(1 + ib, i), ldab - 1, cone,
                   AB.ptr(1 + kd - ib, i + ib), ldab - 1);
            }

            // Update A33.
            Herk('L', 'N', i3, ib, -one, work, ldwork, one,
                 AB.ptr(1, i + kd), ldab - 1);

            // Copy the upper triangle of A31 back into place.
            for (jj = 1; jj <= ib; ++jj) {
              for (ii = 1; ii <= Min(jj, i3); ++ii) {
                AB(kd + 1 - jj + ii, jj + i - 1) = WORK(ii, jj);
              }
            }
          }
        }
      }
    }
  }
  return info;
}

// xPBTRS: solves A * X = B for the N-by-NRHS matrix B (overwritten with X),
// with the Cholesky factorization A = U**H * U or A = L * L**H (UPLO) of the
// symmetric (Hermitian) positive definite band matrix A computed by xPBTRF.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Pbtrs(char uplo, blasint n, blasint kd, blasint nrhs, const T* ab,
              blasint ldab, T* b, blasint ldb) {
  const Matrix<T> B(b, ldb);
  blasint j;
  bool upper;

  // Test the input parameters.
  blasint info = 0;
  upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kd < 0) {
    info = -3;
  } else if (nrhs < 0) {
    info = -4;
  } else if (ldab < kd + 1) {
    info = -6;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("PBTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  if (upper) {
    // Solve A*X = B where A = U**H *U.
    for (j = 1; j <= nrhs; ++j) {
      // Solve U**H *X = B, overwriting B with X.
      Tbsv('U', 'C', 'N', n, kd, ab, ldab, B.ptr(1, j), 1);

      // Solve U*X = B, overwriting B with X.
      Tbsv('U', 'N', 'N', n, kd, ab, ldab, B.ptr(1, j), 1);
    }
  } else {
    // Solve A*X = B where A = L*L**H.
    for (j = 1; j <= nrhs; ++j) {
      // Solve L*X = B, overwriting B with X.
      Tbsv('L', 'N', 'N', n, kd, ab, ldab, B.ptr(1, j), 1);

      // Solve L**H *X = B, overwriting B with X.
      Tbsv('L', 'C', 'N', n, kd, ab, ldab, B.ptr(1, j), 1);
    }
  }
  return info;
}

// xPBSV: solves A * X = B for the symmetric (Hermitian) positive definite
// band matrix A of order N with KD super- or subdiagonals (triangle UPLO
// in AB, overwritten with its Cholesky factor by xPBTRF) and the N-by-NRHS
// matrix B, overwritten with X.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the leading principal minor of order i is not positive (the
// solution has not been computed).
template <typename T>
blasint Pbsv(char uplo, blasint n, blasint kd, blasint nrhs, T* ab,
             blasint ldab, T* b, blasint ldb) {
  // Test the input parameters.
  blasint info = 0;
  if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kd < 0) {
    info = -3;
  } else if (nrhs < 0) {
    info = -4;
  } else if (ldab < kd + 1) {
    info = -6;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("PBSV", -info);
    return info;
  }

  // Compute the Cholesky factorization A = U**H *U or A = L*L**H.
  info = Pbtrf(uplo, n, kd, ab, ldab);
  if (info == 0) {
    // Solve the system A*X = B, overwriting B with X.
    info = Pbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb);
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Pbtf2<T>(char, blasint, blasint, T*, blasint);             \
  template blasint Pbtrf<T>(char, blasint, blasint, T*, blasint);             \
  template blasint Pbtrs<T>(char, blasint, blasint, blasint, const T*,        \
                            blasint, T*, blasint);                            \
  template blasint Pbsv<T>(char, blasint, blasint, blasint, T*, blasint, T*,  \
                           blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_PBSV(prefix, T, R)                                      \
  HWYBLAS_API void prefix##pbtf2_(const char* uplo, const hwyblas_int* n,      \
                                  const hwyblas_int* kd, R* ab,                \
                                  const hwyblas_int* ldab,                     \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Pbtf2(*uplo, *n, *kd, As<T>(ab), *ldab);          \
  }                                                                            \
  HWYBLAS_API void prefix##pbtrf_(const char* uplo, const hwyblas_int* n,      \
                                  const hwyblas_int* kd, R* ab,                \
                                  const hwyblas_int* ldab,                     \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Pbtrf(*uplo, *n, *kd, As<T>(ab), *ldab);          \
  }                                                                            \
  HWYBLAS_API void prefix##pbtrs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* kd,           \
      const hwyblas_int* nrhs, const R* ab, const hwyblas_int* ldab, R* b,     \
      const hwyblas_int* ldb, hwyblas_int* info) {                             \
    *info = hwyblas::lapack::Pbtrs(*uplo, *n, *kd, *nrhs, As<const T>(ab),     \
                                   *ldab, As<T>(b), *ldb);                     \
  }                                                                            \
  HWYBLAS_API void prefix##pbsv_(const char* uplo, const hwyblas_int* n,       \
                                 const hwyblas_int* kd,                        \
                                 const hwyblas_int* nrhs, R* ab,               \
                                 const hwyblas_int* ldab, R* b,                \
                                 const hwyblas_int* ldb, hwyblas_int* info) {  \
    *info = hwyblas::lapack::Pbsv(*uplo, *n, *kd, *nrhs, As<T>(ab), *ldab,     \
                                  As<T>(b), *ldb);                             \
  }

HWYBLAS_LAPACK_PBSV(s, float, float)
HWYBLAS_LAPACK_PBSV(d, double, double)
HWYBLAS_LAPACK_PBSV(c, c32, float)
HWYBLAS_LAPACK_PBSV(z, c64, double)

#undef HWYBLAS_LAPACK_PBSV

}  // extern "C"
