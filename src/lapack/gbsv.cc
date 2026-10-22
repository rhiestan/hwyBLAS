// General band matrices: LU factorization with partial pivoting (xGBTF2
// unblocked, xGBTRF blocked), the solvers xGBTRS and xGBSV, and the
// condition number estimate xGBCON with the triangular band solver xLATBS
// (scaling against overflow).
//
// A band matrix with KL subdiagonals and KU superdiagonals is stored as in
// LAPACK: A(i, j) in AB(KV+1+i-j, j) with KV = KU for xGBTRS and xGBCON's
// input (rows 1 to KL+KU+1 of the factor U, rows KV+2 to KV+KL+1 the
// multipliers) and KV = KL + KU for xGBTRF (the first KL rows are fill-in).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <vector>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xGBTF2: computes the LU factorization A = P * L * U of the M-by-N band
// matrix A with KL subdiagonals and KU superdiagonals, with partial
// pivoting and row interchanges, with the unblocked algorithm (Level 2
// BLAS). AB (LDAB >= 2*KL+KU+1) holds A in rows KL+1 to 2*KL+KU+1 on entry;
// on exit, U in rows 1 to KL+KU+1 and the multipliers of L in the rows
// below. IPIV(i) is the row interchanged with row i.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero (the factorization is completed).
template <typename T>
blasint Gbtf2(blasint m, blasint n, blasint kl, blasint ku, T* ab,
              blasint ldab, blasint* ipiv) {
  const T one = T(1), zero = T(0);
  const Matrix<T> AB(ab, ldab);
  const Vector<blasint> IPIV(ipiv);
  blasint i, j, jp, ju, km, kv;

  // KV is the number of superdiagonals in the factor U, allowing for
  // fill-in.
  kv = ku + kl;

  // Test the input parameters.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kl < 0) {
    info = -3;
  } else if (ku < 0) {
    info = -4;
  } else if (ldab < kl + kv + 1) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("GBTF2", -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) return info;

  // Gaussian elimination with partial pivoting.

  // Set fill-in elements in columns KU+2 to KV to zero.
  for (j = ku + 2; j <= Min(kv, n); ++j) {
    for (i = kv - j + 2; i <= kl; ++i) AB(i, j) = zero;
  }

  // JU is the index of the last column affected by the current stage of
  // the factorization.
  ju = 1;

  for (j = 1; j <= Min(m, n); ++j) {
    // Set fill-in elements in column J+KV to zero.
    if (j + kv <= n) {
      for (i = 1; i <= kl; ++i) AB(i, j + kv) = zero;
    }

    // Find pivot and test for singularity. KM is the number of
    // subdiagonal elements in the current column.
    km = Min(kl, m - j);
    jp = Iamax(km + 1, AB.ptr(kv + 1, j), 1);
    IPIV(j) = jp + j - 1;
    if (AB(kv + jp, j) != zero) {
      ju = Max(ju, Min(j + ku + jp - 1, n));

      // Apply interchange to columns J to JU.
      if (jp != 1) {
        Swap(ju - j + 1, AB.ptr(kv + jp, j), ldab - 1, AB.ptr(kv + 1, j),
             ldab - 1);
      }

      if (km > 0) {
        // Compute multipliers.
        Scal(km, one / AB(kv + 1, j), AB.ptr(kv + 2, j), 1);

        // Update trailing submatrix within the band.
        if (ju > j) {
          Geru(km, ju - j, -one, AB.ptr(kv + 2, j), 1, AB.ptr(kv, j + 1),
               ldab - 1, AB.ptr(kv + 1, j + 1), ldab - 1);
        }
      }
    } else {
      // If pivot is zero, set INFO to the index of the pivot unless a zero
      // pivot has already been found.
      if (info == 0) info = j;
    }
  }
  return info;
}

// xGBTRF: computes the LU factorization of the band matrix A as xGBTF2,
// with the blocked algorithm (Level 3 BLAS) if the ILAENV block size NB
// is > 1 and <= KL.
//
// Returns INFO as Gbtf2.
template <typename T>
blasint Gbtrf(blasint m, blasint n, blasint kl, blasint ku, T* ab,
              blasint ldab, blasint* ipiv) {
  const T one = T(1), zero = T(0);
  const blasint nbmax = 64, ldwork = nbmax + 1;
  const Matrix<T> AB(ab, ldab);
  const Vector<blasint> IPIV(ipiv);
  blasint i, i2, i3, ii, ip, j, j2, j3, jb, jj, jm, jp, ju, k2, km, kv, nb,
      nw;
  T temp;

  // KV is the number of superdiagonals in the factor U, allowing for
  // fill-in.
  kv = ku + kl;

  // Test the input parameters.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kl < 0) {
    info = -3;
  } else if (ku < 0) {
    info = -4;
  } else if (ldab < kl + kv + 1) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("GBTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) return info;

  // Determine the block size for this environment.
  nb = Ilaenv<T>(1, "GBTRF", " ", m, n, kl, ku);

  // The block size must not exceed the limit set by the size of the local
  // arrays WORK13 and WORK31.
  nb = Min(nb, nbmax);

  if (nb <= 1 || nb > kl) {
    // Use unblocked code.
    info = Gbtf2(m, n, kl, ku, ab, ldab, ipiv);
  } else {
    // Use blocked code.

    // WORK13(LDWORK, NBMAX) and WORK31(LDWORK, NBMAX), local arrays of the
    // reference (on the heap here: 133 KB for complex T).
    std::vector<T> work13(ldwork * nbmax), work31(ldwork * nbmax);
    const Matrix<T> WORK13(work13.data(), ldwork);
    const Matrix<T> WORK31(work31.data(), ldwork);

    // Zero the superdiagonal elements of the work array WORK13.
    for (j = 1; j <= nb; ++j) {
      for (i = 1; i <= j - 1; ++i) WORK13(i, j) = zero;
    }

    // Zero the subdiagonal elements of the work array WORK31.
    for (j = 1; j <= nb; ++j) {
      for (i = j + 1; i <= nb; ++i) WORK31(i, j) = zero;
    }

    // Gaussian elimination with partial pivoting.

    // Set fill-in elements in columns KU+2 to KV to zero.
    for (j = ku + 2; j <= Min(kv, n); ++j) {
      for (i = kv - j + 2; i <= kl; ++i) AB(i, j) = zero;
    }

    // JU is the index of the last column affected by the current stage of
    // the factorization.
    ju = 1;

    for (j = 1; j <= Min(m, n); j += nb) {
      jb = Min(nb, Min(m, n) - j + 1);

      // The active part of the matrix is partitioned
      //
      //    A11   A12   A13
      //    A21   A22   A23
      //    A31   A32   A33
      //
      // Here A11, A21 and A31 denote the current block of JB columns which
      // is about to be factorized. The number of rows in the partitioning
      // are JB, I2, I3 respectively, and the numbers of columns are JB,
      // J2, J3. The superdiagonal elements of A13 and the subdiagonal
      // elements of A31 lie outside the band.
      i2 = Min(kl - jb, m - j - jb + 1);
      i3 = Min(jb, m - j - kl + 1);

      // J2 and J3 are computed after JU has been updated.

      // Factorize the current block of JB columns.
      for (jj = j; jj <= j + jb - 1; ++jj) {
        // Set fill-in elements in column JJ+KV to zero.
        if (jj + kv <= n) {
          for (i = 1; i <= kl; ++i) AB(i, jj + kv) = zero;
        }

        // Find pivot and test for singularity. KM is the number of
        // subdiagonal elements in the current column.
        km = Min(kl, m - jj);
        jp = Iamax(km + 1, AB.ptr(kv + 1, jj), 1);
        IPIV(jj) = jp + jj - j;
        if (AB(kv + jp, jj) != zero) {
          ju = Max(ju, Min(jj + ku + jp - 1, n));
          if (jp != 1) {
            // Apply interchange to columns J to J+JB-1.
            if (jp + jj - 1 < j + kl) {
              Swap(jb, AB.ptr(kv + 1 + jj - j, j), ldab - 1,
                   AB.ptr(kv + jp + jj - j, j), ldab - 1);
            } else {
              // The interchange affects columns J to JJ-1 of A31 which are
              // stored in the work array WORK31.
              Swap(jj - j, AB.ptr(kv + 1 + jj - j, j), ldab - 1,
                   WORK31.ptr(jp + jj - j - kl, 1), ldwork);
              Swap(j + jb - jj, AB.ptr(kv + 1, jj), ldab - 1,
                   AB.ptr(kv + jp, jj), ldab - 1);
            }
          }

          // Compute multipliers.
          Scal(km, one / AB(kv + 1, jj), AB.ptr(kv + 2, jj), 1);

          // Update trailing submatrix within the band and within the
          // current block. JM is the index of the last column which needs
          // to be updated.
          jm = Min(ju, j + jb - 1);
          if (jm > jj) {
            Geru(km, jm - jj, -one, AB.ptr(kv + 2, jj), 1, AB.ptr(kv, jj + 1),
                 ldab - 1, AB.ptr(kv + 1, jj + 1), ldab - 1);
          }
        } else {
          // If pivot is zero, set INFO to the index of the pivot unless a
          // zero pivot has already been found.
          if (info == 0) info = jj;
        }

        // Copy current column of A31 into the work array WORK31.
        nw = Min(jj - j + 1, i3);
        if (nw > 0) {
          Copy(nw, AB.ptr(kv + kl + 1 - jj + j, jj), 1,
               WORK31.ptr(1, jj - j + 1), 1);
        }
      }
      if (j + jb <= n) {
        // Apply the row interchanges to the other blocks.
        j2 = Min(ju - j + 1, kv) - jb;
        j3 = Max<blasint>(0, ju - j - kv + 1);

        // Use xLASWP to apply the row interchanges to A12, A22, and A32.
        Laswp(j2, AB.ptr(kv + 1 - jb, j + jb), ldab - 1, 1, jb, IPIV.ptr(j),
              1);

        // Adjust the pivot indices.
        for (i = j; i <= j + jb - 1; ++i) IPIV(i) = IPIV(i) + j - 1;

        // Apply the row interchanges to A13, A23, and A33 columnwise.
        k2 = j - 1 + jb + j2;
        for (i = 1; i <= j3; ++i) {
          jj = k2 + i;
          for (ii = j + i - 1; ii <= j + jb - 1; ++ii) {
            ip = IPIV(ii);
            if (ip != ii) {
              temp = AB(kv + 1 + ii - jj, jj);
              AB(kv + 1 + ii - jj, jj) = AB(kv + 1 + ip - jj, jj);
              AB(kv + 1 + ip - jj, jj) = temp;
            }
          }
        }

        // Update the relevant part of the trailing submatrix.
        if (j2 > 0) {
          // Update A12.
          Trsm('L', 'L', 'N', 'U', jb, j2, one, AB.ptr(kv + 1, j), ldab - 1,
               AB.ptr(kv + 1 - jb, j + jb), ldab - 1);

          if (i2 > 0) {
            // Update A22.
            Gemm('N', 'N', i2, j2, jb, -one, AB.ptr(kv + 1 + jb, j),
                 ldab - 1, AB.ptr(kv + 1 - jb, j + jb), ldab - 1, one,
                 AB.ptr(kv + 1, j + jb), ldab - 1);
          }

          if (i3 > 0) {
            // Update A32.
            Gemm('N', 'N', i3, j2, jb, -one, WORK31.ptr(1, 1), ldwork,
                 AB.ptr(kv + 1 - jb, j + jb), ldab - 1, one,
                 AB.ptr(kv + kl + 1 - jb, j + jb), ldab - 1);
          }
        }

        if (j3 > 0) {
          // Copy the lower triangle of A13 into the work array WORK13.
          for (jj = 1; jj <= j3; ++jj) {
            for (ii = jj; ii <= jb; ++ii) {
              WORK13(ii, jj) = AB(ii - jj + 1, jj + j + kv - 1);
            }
          }

          // Update A13 in the work array.
          Trsm('L', 'L', 'N', 'U', jb, j3, one, AB.ptr(kv + 1, j), ldab - 1,
               WORK13.ptr(1, 1), ldwork);

          if (i2 > 0) {
            // Update A23.
            Gemm('N', 'N', i2, j3, jb, -one, AB.ptr(kv + 1 + jb, j),
                 ldab - 1, WORK13.ptr(1, 1), ldwork, one,
                 AB.ptr(1 + jb, j + kv), ldab - 1);
          }

          if (i3 > 0) {
            // Update A33.
            Gemm('N', 'N', i3, j3, jb, -one, WORK31.ptr(1, 1), ldwork,
                 WORK13.ptr(1, 1), ldwork, one, AB.ptr(1 + kl, j + kv),
                 ldab - 1);
          }

          // Copy the lower triangle of A13 back into place.
          for (jj = 1; jj <= j3; ++jj) {
            for (ii = jj; ii <= jb; ++ii) {
              AB(ii - jj + 1, jj + j + kv - 1) = WORK13(ii, jj);
            }
          }
        }
      } else {
        // Adjust the pivot indices.
        for (i = j; i <= j + jb - 1; ++i) IPIV(i) = IPIV(i) + j - 1;
      }

      // Partially undo the interchanges in the current block to restore
      // the upper triangular form of A31 and copy the upper triangle of A31
      // back into place.
      for (jj = j + jb - 1; jj >= j; --jj) {
        jp = IPIV(jj) - jj + 1;
        if (jp != 1) {
          // Apply interchange to columns J to JJ-1.
          if (jp + jj - 1 < j + kl) {
            // The interchange does not affect A31.
            Swap(jj - j, AB.ptr(kv + 1 + jj - j, j), ldab - 1,
                 AB.ptr(kv + jp + jj - j, j), ldab - 1);
          } else {
            // The interchange does affect A31.
            Swap(jj - j, AB.ptr(kv + 1 + jj - j, j), ldab - 1,
                 WORK31.ptr(jp + jj - j - kl, 1), ldwork);
          }
        }

        // Copy the current column of A31 back into place.
        nw = Min(i3, jj - j + 1);
        if (nw > 0) {
          Copy(nw, WORK31.ptr(1, jj - j + 1), 1,
               AB.ptr(kv + kl + 1 - jj + j, jj), 1);
        }
      }
    }
  }
  return info;
}

// xGBTRS: solves A * X = B, A**T * X = B or A**H * X = B (TRANS = 'N', 'T'
// or 'C') for the N-by-NRHS matrix B (overwritten with X), with the LU
// factorization of the N-by-N band matrix A computed by xGBTRF (AB, LDAB >=
// 2*KL+KU+1, and IPIV).
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Gbtrs(char trans, blasint n, blasint kl, blasint ku, blasint nrhs,
              const T* ab, blasint ldab, const blasint* ipiv, T* b,
              blasint ldb) {
  const T one = T(1);
  const Matrix<const T> AB(ab, ldab);
  const Matrix<T> B(b, ldb);
  const Vector<const blasint> IPIV(ipiv);
  blasint i, j, kd, l, lm;
  bool lnoti, notran;

  // Test the input parameters.
  blasint info = 0;
  notran = Lsame(trans, 'N');
  if (!notran && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kl < 0) {
    info = -3;
  } else if (ku < 0) {
    info = -4;
  } else if (nrhs < 0) {
    info = -5;
  } else if (ldab < (2 * kl + ku + 1)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -10;
  }
  if (info != 0) {
    Xerbla<T>("GBTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  kd = ku + kl + 1;
  lnoti = kl > 0;

  if (notran) {
    // Solve A*X = B.
    //
    // Solve L*X = B, overwriting B with X.
    //
    // L is represented as a product of permutations and unit lower
    // triangular matrices L = P(1) * L(1) * ... * P(n-1) * L(n-1), where
    // each transformation L(i) is a rank-one modification of the identity
    // matrix.
    if (lnoti) {
      for (j = 1; j <= n - 1; ++j) {
        lm = Min(kl, n - j);
        l = IPIV(j);
        if (l != j) Swap(nrhs, B.ptr(l, 1), ldb, B.ptr(j, 1), ldb);
        Geru(lm, nrhs, -one, AB.ptr(kd + 1, j), 1, B.ptr(j, 1), ldb,
             B.ptr(j + 1, 1), ldb);
      }
    }

    for (i = 1; i <= nrhs; ++i) {
      // Solve U*X = B, overwriting B with X.
      Tbsv('U', 'N', 'N', n, kl + ku, ab, ldab, B.ptr(1, i), 1);
    }
  } else if (!IsComplex<T>::value || Lsame(trans, 'T')) {
    // Solve A**T * X = B (also for TRANS = 'C' if T is real).
    for (i = 1; i <= nrhs; ++i) {
      // Solve U**T * X = B, overwriting B with X.
      Tbsv('U', 'T', 'N', n, kl + ku, ab, ldab, B.ptr(1, i), 1);
    }

    // Solve L**T * X = B, overwriting B with X.
    if (lnoti) {
      for (j = n - 1; j >= 1; --j) {
        lm = Min(kl, n - j);
        Gemv('T', lm, nrhs, -one, B.ptr(j + 1, 1), ldb, AB.ptr(kd + 1, j), 1,
             one, B.ptr(j, 1), ldb);
        l = IPIV(j);
        if (l != j) Swap(nrhs, B.ptr(l, 1), ldb, B.ptr(j, 1), ldb);
      }
    }
  } else {
    // Solve A**H * X = B.
    for (i = 1; i <= nrhs; ++i) {
      // Solve U**H * X = B, overwriting B with X.
      Tbsv('U', 'C', 'N', n, kl + ku, ab, ldab, B.ptr(1, i), 1);
    }

    // Solve L**H * X = B, overwriting B with X.
    if (lnoti) {
      for (j = n - 1; j >= 1; --j) {
        lm = Min(kl, n - j);
        Lacgv(nrhs, B.ptr(j, 1), ldb);
        Gemv('C', lm, nrhs, -one, B.ptr(j + 1, 1), ldb, AB.ptr(kd + 1, j), 1,
             one, B.ptr(j, 1), ldb);
        Lacgv(nrhs, B.ptr(j, 1), ldb);
        l = IPIV(j);
        if (l != j) Swap(nrhs, B.ptr(l, 1), ldb, B.ptr(j, 1), ldb);
      }
    }
  }
  return info;
}

// xGBSV: solves A * X = B for the N-by-N band matrix A (KL subdiagonals,
// KU superdiagonals, in rows KL+1 to 2*KL+KU+1 of AB) and the N-by-NRHS
// matrix B, with the LU factorization of xGBTRF (overwriting AB and IPIV);
// X overwrites B.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero (the solution has not been computed).
template <typename T>
blasint Gbsv(blasint n, blasint kl, blasint ku, blasint nrhs, T* ab,
             blasint ldab, blasint* ipiv, T* b, blasint ldb) {
  // Test the input parameters.
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (kl < 0) {
    info = -2;
  } else if (ku < 0) {
    info = -3;
  } else if (nrhs < 0) {
    info = -4;
  } else if (ldab < 2 * kl + ku + 1) {
    info = -6;
  } else if (ldb < Max<blasint>(n, 1)) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<T>("GBSV", -info);
    return info;
  }

  // Compute the LU factorization of the band matrix A.
  info = Gbtrf(n, n, kl, ku, ab, ldab, ipiv);
  if (info == 0) {
    // Solve the system A*X = B, overwriting B with X.
    info = Gbtrs('N', n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  }
  return info;
}

// xLATBS for real T (DLATBS / SLATBS); see Latbs.
template <typename T>
blasint LatbsReal(char uplo, char trans, char diag, char normin, blasint n,
                  blasint kd, const T* ab, blasint ldab, T* x, T* scale,
                  T* cnorm) {
  using R = T;
  const R zero = R(0), half = R(0.5), one = R(1);
  const Matrix<const R> AB(ab, ldab);
  const Vector<R> CNORM(cnorm);
  const Vector<R> X(x);
  blasint i, imax, j, jfirst, jinc, jlast, jlen, maind;
  bool notran, nounit, upper;
  R bignum, grow, rec, smlnum, sumj, tjj, tmax, tscal, uscal, xbnd, xj, xmax;
  // Set wherever it is read (USCAL differs from TSCAL only after a division
  // by TJJS); initialized for the compiler.
  R tjjs = zero;

  blasint info = 0;
  upper = Lsame(uplo, 'U');
  notran = Lsame(trans, 'N');
  nounit = Lsame(diag, 'N');

  // Test the input parameters.
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!notran && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
    info = -2;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -3;
  } else if (!Lsame(normin, 'Y') && !Lsame(normin, 'N')) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (kd < 0) {
    info = -6;
  } else if (ldab < kd + 1) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<R>("LATBS", -info);
    return info;
  }

  // Quick return if possible.
  *scale = one;
  if (n == 0) return info;

  // Determine machine dependent parameters to control overflow.
  smlnum = Lamch<R>('S') / Lamch<R>('P');
  bignum = one / smlnum;

  if (Lsame(normin, 'N')) {
    // Compute the 1-norm of each column, not including the diagonal.
    if (upper) {
      // A is upper triangular.
      for (j = 1; j <= n; ++j) {
        jlen = Min(kd, j - 1);
        CNORM(j) = Asum(jlen, AB.ptr(kd + 1 - jlen, j), 1);
      }
    } else {
      // A is lower triangular.
      for (j = 1; j <= n; ++j) {
        jlen = Min(kd, n - j);
        if (jlen > 0) {
          CNORM(j) = Asum(jlen, AB.ptr(2, j), 1);
        } else {
          CNORM(j) = zero;
        }
      }
    }
  }

  // Scale the column norms by TSCAL if the maximum element in CNORM is
  // greater than BIGNUM.
  imax = Iamax(n, cnorm, 1);
  tmax = CNORM(imax);
  if (tmax <= bignum) {
    tscal = one;
  } else {
    tscal = one / (smlnum * tmax);
    Scal(n, tscal, cnorm, 1);
  }

  // Compute a bound on the computed solution vector to see if the Level 2
  // BLAS routine xTBSV can be used.
  j = Iamax(n, x, 1);
  xmax = std::abs(X(j));
  xbnd = xmax;
  if (notran) {
    // Compute the growth in A * x = b.
    if (upper) {
      jfirst = n;
      jlast = 1;
      jinc = -1;
      maind = kd + 1;
    } else {
      jfirst = 1;
      jlast = n;
      jinc = 1;
      maind = 1;
    }

    if (tscal != one) {
      grow = zero;
      goto L50;
    }

    if (nounit) {
      // A is non-unit triangular.
      //
      // Compute GROW = 1/G(j) and XBND = 1/M(j).
      // Initially, G(0) = max{x(i), i=1,...,n}.
      grow = one / Max(xbnd, smlnum);
      xbnd = grow;
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L50;

        // M(j) = G(j-1) / abs(A(j,j))
        tjj = std::abs(AB(maind, j));
        xbnd = Min(xbnd, Min(one, tjj) * grow);
        if (tjj + CNORM(j) >= smlnum) {
          // G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) )
          grow = grow * (tjj / (tjj + CNORM(j)));
        } else {
          // G(j) could overflow, set GROW to 0.
          grow = zero;
        }
      }
      grow = xbnd;
    } else {
      // A is unit triangular.
      //
      // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
      grow = Min(one, one / Max(xbnd, smlnum));
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L50;

        // G(j) = G(j-1)*( 1 + CNORM(j) )
        grow = grow * (one / (one + CNORM(j)));
      }
    }
  L50:;
  } else {
    // Compute the growth in A**T * x = b.
    if (upper) {
      jfirst = 1;
      jlast = n;
      jinc = 1;
      maind = kd + 1;
    } else {
      jfirst = n;
      jlast = 1;
      jinc = -1;
      maind = 1;
    }

    if (tscal != one) {
      grow = zero;
      goto L80;
    }

    if (nounit) {
      // A is non-unit triangular.
      //
      // Compute GROW = 1/G(j) and XBND = 1/M(j).
      // Initially, M(0) = max{x(i), i=1,...,n}.
      grow = one / Max(xbnd, smlnum);
      xbnd = grow;
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L80;

        // G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
        xj = one + CNORM(j);
        grow = Min(grow, xbnd / xj);

        // M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
        tjj = std::abs(AB(maind, j));
        if (xj > tjj) xbnd = xbnd * (tjj / xj);
      }
      grow = Min(grow, xbnd);
    } else {
      // A is unit triangular.
      //
      // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
      grow = Min(one, one / Max(xbnd, smlnum));
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L80;

        // G(j) = ( 1 + CNORM(j) )*G(j-1)
        xj = one + CNORM(j);
        grow = grow / xj;
      }
    }
  L80:;
  }

  if ((grow * tscal) > smlnum) {
    // Use the Level 2 BLAS solve if the reciprocal of the bound on elements
    // of X is not too small.
    Tbsv(uplo, trans, diag, n, kd, ab, ldab, x, 1);
  } else {
    // Use a Level 1 BLAS solve, scaling intermediate results.
    if (xmax > bignum) {
      // Scale X so that its components are less than or equal to BIGNUM in
      // absolute value.
      *scale = bignum / xmax;
      Scal(n, *scale, x, 1);
      xmax = bignum;
    }

    if (notran) {
      // Solve A * x = b
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Compute x(j) = b(j) / A(j,j), scaling x if necessary.
        xj = std::abs(X(j));
        if (nounit) {
          tjjs = AB(maind, j) * tscal;
        } else {
          tjjs = tscal;
          if (tscal == one) goto L100;
        }
        tjj = std::abs(tjjs);
        if (tjj > smlnum) {
          // abs(A(j,j)) > SMLNUM:
          if (tjj < one) {
            if (xj > tjj * bignum) {
              // Scale x by 1/b(j).
              rec = one / xj;
              Scal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          X(j) = X(j) / tjjs;
          xj = std::abs(X(j));
        } else if (tjj > zero) {
          // 0 < abs(A(j,j)) <= SMLNUM:
          if (xj > tjj * bignum) {
            // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM to avoid overflow
            // when dividing by A(j,j).
            rec = (tjj * bignum) / xj;
            if (CNORM(j) > one) {
              // Scale by 1/CNORM(j) to avoid overflow when multiplying x(j)
              // times column j.
              rec = rec / CNORM(j);
            }
            Scal(n, rec, x, 1);
            *scale = *scale * rec;
            xmax = xmax * rec;
          }
          X(j) = X(j) / tjjs;
          xj = std::abs(X(j));
        } else {
          // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and scale = 0, and
          // compute a solution to A*x = 0.
          for (i = 1; i <= n; ++i) X(i) = zero;
          X(j) = one;
          xj = one;
          *scale = zero;
          xmax = zero;
        }
      L100:

        // Scale x if necessary to avoid overflow when adding a multiple of
        // column j of A.
        if (xj > one) {
          rec = one / xj;
          if (CNORM(j) > (bignum - xmax) * rec) {
            // Scale x by 1/(2*abs(x(j))).
            rec = rec * half;
            Scal(n, rec, x, 1);
            *scale = *scale * rec;
          }
        } else if (xj * CNORM(j) > (bignum - xmax)) {
          // Scale x by 1/2.
          Scal(n, half, x, 1);
          *scale = *scale * half;
        }

        if (upper) {
          if (j > 1) {
            // Compute the update
            //    x(max(1,j-kd):j-1) := x(max(1,j-kd):j-1) -
            //                          x(j)* A(max(1,j-kd):j-1,j)
            jlen = Min(kd, j - 1);
            Axpy(jlen, -X(j) * tscal, AB.ptr(kd + 1 - jlen, j), 1,
                 X.ptr(j - jlen), 1);
            i = Iamax(j - 1, x, 1);
            xmax = std::abs(X(i));
          }
        } else if (j < n) {
          // Compute the update
          //    x(j+1:min(j+kd,n)) := x(j+1:min(j+kd,n)) -
          //                          x(j) * A(j+1:min(j+kd,n),j)
          jlen = Min(kd, n - j);
          if (jlen > 0) {
            Axpy(jlen, -X(j) * tscal, AB.ptr(2, j), 1, X.ptr(j + 1), 1);
          }
          i = j + Iamax(n - j, X.ptr(j + 1), 1);
          xmax = std::abs(X(i));
        }
      }
    } else {
      // Solve A**T * x = b
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Compute x(j) = b(j) - sum A(k,j)*x(k).
        //                       k<>j
        xj = std::abs(X(j));
        uscal = tscal;
        rec = one / Max(xmax, one);
        if (CNORM(j) > (bignum - xj) * rec) {
          // If x(j) could overflow, scale x by 1/(2*XMAX).
          rec = rec * half;
          if (nounit) {
            tjjs = AB(maind, j) * tscal;
          } else {
            tjjs = tscal;
          }
          tjj = std::abs(tjjs);
          if (tjj > one) {
            // Divide by A(j,j) when scaling x if A(j,j) > 1.
            rec = Min(one, rec * tjj);
            uscal = uscal / tjjs;
          }
          if (rec < one) {
            Scal(n, rec, x, 1);
            *scale = *scale * rec;
            xmax = xmax * rec;
          }
        }

        sumj = zero;
        if (uscal == one) {
          // If the scaling needed for A in the dot product is 1, call xDOT
          // to perform the dot product.
          if (upper) {
            jlen = Min(kd, j - 1);
            sumj = Dotc(jlen, AB.ptr(kd + 1 - jlen, j), 1, X.ptr(j - jlen), 1);
          } else {
            jlen = Min(kd, n - j);
            if (jlen > 0) sumj = Dotc(jlen, AB.ptr(2, j), 1, X.ptr(j + 1), 1);
          }
        } else {
          // Otherwise, use in-line code for the dot product.
          if (upper) {
            jlen = Min(kd, j - 1);
            for (i = 1; i <= jlen; ++i) {
              sumj = sumj + (AB(kd + i - jlen, j) * uscal) * X(j - jlen - 1 + i);
            }
          } else {
            jlen = Min(kd, n - j);
            for (i = 1; i <= jlen; ++i) {
              sumj = sumj + (AB(i + 1, j) * uscal) * X(j + i);
            }
          }
        }

        if (uscal == tscal) {
          // Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j) was not
          // used to scale the dotproduct.
          X(j) = X(j) - sumj;
          xj = std::abs(X(j));
          if (nounit) {
            // Compute x(j) = x(j) / A(j,j), scaling if necessary.
            tjjs = AB(maind, j) * tscal;
          } else {
            tjjs = tscal;
            if (tscal == one) goto L150;
          }
          tjj = std::abs(tjjs);
          if (tjj > smlnum) {
            // abs(A(j,j)) > SMLNUM:
            if (tjj < one) {
              if (xj > tjj * bignum) {
                // Scale X by 1/abs(x(j)).
                rec = one / xj;
                Scal(n, rec, x, 1);
                *scale = *scale * rec;
                xmax = xmax * rec;
              }
            }
            X(j) = X(j) / tjjs;
          } else if (tjj > zero) {
            // 0 < abs(A(j,j)) <= SMLNUM:
            if (xj > tjj * bignum) {
              // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
              rec = (tjj * bignum) / xj;
              Scal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
            X(j) = X(j) / tjjs;
          } else {
            // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and scale = 0, and
            // compute a solution to A**T*x = 0.
            for (i = 1; i <= n; ++i) X(i) = zero;
            X(j) = one;
            *scale = zero;
            xmax = zero;
          }
        L150:;
        } else {
          // Compute x(j) := x(j) / A(j,j) - sumj if the dot product has
          // already been divided by 1/A(j,j).
          X(j) = X(j) / tjjs - sumj;
        }
        xmax = Max(xmax, std::abs(X(j)));
      }
    }
    *scale = *scale / tscal;
  }

  // Scale the column norms by 1/TSCAL for return.
  if (tscal != one) Scal(n, one / tscal, cnorm, 1);

  return info;
}

// xLATBS for complex T (CLATBS / ZLATBS); see Latbs. It differs from the
// real routine as xLATRS does: in its scaling (by BIGNUM/2, with |Re| + |Im|
// as the absolute value) and in its separate branches for A**T and A**H.
//
// Deviation: in the lower triangular case of these two branches, the
// reference computes the dot product with xDOTU / xDOTC only if JLEN > 1
// (DLATBS: JLEN > 0), so that for JLEN = 1 (KD = 1, or J = N-1) it leaves
// out the element A(j+1, j) and returns a wrong x whenever this scaled
// solve runs with USCAL = 1. The port tests JLEN > 0 as DLATBS.
template <typename T>
blasint LatbsComplex(char uplo, char trans, char diag, char normin, blasint n,
                     blasint kd, const T* ab, blasint ldab, T* x,
                     RealT<T>* scale, RealT<T>* cnorm) {
  using R = RealT<T>;
  const R zero = R(0), half = R(0.5), one = R(1), two = R(2);
  const Matrix<const T> AB(ab, ldab);
  const Vector<R> CNORM(cnorm);
  const Vector<T> X(x);
  blasint i, imax, j, jfirst, jinc, jlast, jlen, maind;
  bool notran, nounit, upper;
  R bignum, grow, rec, smlnum, tjj, tmax, tscal, xbnd, xj, xmax;
  T csumj, tjjs, uscal;

  blasint info = 0;
  upper = Lsame(uplo, 'U');
  notran = Lsame(trans, 'N');
  nounit = Lsame(diag, 'N');

  // Test the input parameters.
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!notran && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
    info = -2;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -3;
  } else if (!Lsame(normin, 'Y') && !Lsame(normin, 'N')) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (kd < 0) {
    info = -6;
  } else if (ldab < kd + 1) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("LATBS", -info);
    return info;
  }

  // Quick return if possible.
  *scale = one;
  if (n == 0) return info;

  // Determine machine dependent parameters to control overflow.
  smlnum = Lamch<R>('S') / Lamch<R>('P');
  bignum = one / smlnum;

  if (Lsame(normin, 'N')) {
    // Compute the 1-norm of each column, not including the diagonal.
    if (upper) {
      // A is upper triangular.
      for (j = 1; j <= n; ++j) {
        jlen = Min(kd, j - 1);
        CNORM(j) = Asum(jlen, AB.ptr(kd + 1 - jlen, j), 1);
      }
    } else {
      // A is lower triangular.
      for (j = 1; j <= n; ++j) {
        jlen = Min(kd, n - j);
        if (jlen > 0) {
          CNORM(j) = Asum(jlen, AB.ptr(2, j), 1);
        } else {
          CNORM(j) = zero;
        }
      }
    }
  }

  // Scale the column norms by TSCAL if the maximum element in CNORM is
  // greater than BIGNUM/2.
  imax = Iamax(n, cnorm, 1);
  tmax = CNORM(imax);
  if (tmax <= bignum * half) {
    tscal = one;
  } else {
    tscal = half / (smlnum * tmax);
    Scal(n, tscal, cnorm, 1);
  }

  // Compute a bound on the computed solution vector to see if the Level 2
  // BLAS routine xTBSV can be used.
  xmax = zero;
  for (j = 1; j <= n; ++j) xmax = Max(xmax, Cabs2(X(j)));
  xbnd = xmax;
  if (notran) {
    // Compute the growth in A * x = b.
    if (upper) {
      jfirst = n;
      jlast = 1;
      jinc = -1;
      maind = kd + 1;
    } else {
      jfirst = 1;
      jlast = n;
      jinc = 1;
      maind = 1;
    }

    if (tscal != one) {
      grow = zero;
      goto L60;
    }

    if (nounit) {
      // A is non-unit triangular.
      //
      // Compute GROW = 1/G(j) and XBND = 1/M(j).
      // Initially, G(0) = max{x(i), i=1,...,n}.
      grow = half / Max(xbnd, smlnum);
      xbnd = grow;
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L60;

        tjjs = AB(maind, j);
        tjj = Cabs1(tjjs);

        if (tjj >= smlnum) {
          // M(j) = G(j-1) / abs(A(j,j))
          xbnd = Min(xbnd, Min(one, tjj) * grow);
        } else {
          // M(j) could overflow, set XBND to 0.
          xbnd = zero;
        }

        if (tjj + CNORM(j) >= smlnum) {
          // G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) )
          grow = grow * (tjj / (tjj + CNORM(j)));
        } else {
          // G(j) could overflow, set GROW to 0.
          grow = zero;
        }
      }
      grow = xbnd;
    } else {
      // A is unit triangular.
      //
      // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
      grow = Min(one, half / Max(xbnd, smlnum));
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L60;

        // G(j) = G(j-1)*( 1 + CNORM(j) )
        grow = grow * (one / (one + CNORM(j)));
      }
    }
  L60:;
  } else {
    // Compute the growth in A**T * x = b  or  A**H * x = b.
    if (upper) {
      jfirst = 1;
      jlast = n;
      jinc = 1;
      maind = kd + 1;
    } else {
      jfirst = n;
      jlast = 1;
      jinc = -1;
      maind = 1;
    }

    if (tscal != one) {
      grow = zero;
      goto L90;
    }

    if (nounit) {
      // A is non-unit triangular.
      //
      // Compute GROW = 1/G(j) and XBND = 1/M(j).
      // Initially, M(0) = max{x(i), i=1,...,n}.
      grow = half / Max(xbnd, smlnum);
      xbnd = grow;
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L90;

        // G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
        xj = one + CNORM(j);
        grow = Min(grow, xbnd / xj);

        tjjs = AB(maind, j);
        tjj = Cabs1(tjjs);

        if (tjj >= smlnum) {
          // M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
          if (xj > tjj) xbnd = xbnd * (tjj / xj);
        } else {
          // M(j) could overflow, set XBND to 0.
          xbnd = zero;
        }
      }
      grow = Min(grow, xbnd);
    } else {
      // A is unit triangular.
      //
      // Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
      grow = Min(one, half / Max(xbnd, smlnum));
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Exit the loop if the growth factor is too small.
        if (grow <= smlnum) goto L90;

        // G(j) = ( 1 + CNORM(j) )*G(j-1)
        xj = one + CNORM(j);
        grow = grow / xj;
      }
    }
  L90:;
  }

  if ((grow * tscal) > smlnum) {
    // Use the Level 2 BLAS solve if the reciprocal of the bound on elements
    // of X is not too small.
    Tbsv(uplo, trans, diag, n, kd, ab, ldab, x, 1);
  } else {
    // Use a Level 1 BLAS solve, scaling intermediate results.
    if (xmax > bignum * half) {
      // Scale X so that its components are less than or equal to BIGNUM in
      // absolute value.
      *scale = (bignum * half) / xmax;
      Rscal(n, *scale, x, 1);
      xmax = bignum;
    } else {
      xmax = xmax * two;
    }

    if (notran) {
      // Solve A * x = b
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Compute x(j) = b(j) / A(j,j), scaling x if necessary.
        xj = Cabs1(X(j));
        if (nounit) {
          tjjs = AB(maind, j) * tscal;
        } else {
          tjjs = tscal;
          if (tscal == one) goto L110;
        }
        tjj = Cabs1(tjjs);
        if (tjj > smlnum) {
          // abs(A(j,j)) > SMLNUM:
          if (tjj < one) {
            if (xj > tjj * bignum) {
              // Scale x by 1/b(j).
              rec = one / xj;
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          X(j) = Ladiv(X(j), tjjs);
          xj = Cabs1(X(j));
        } else if (tjj > zero) {
          // 0 < abs(A(j,j)) <= SMLNUM:
          if (xj > tjj * bignum) {
            // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM to avoid overflow
            // when dividing by A(j,j).
            rec = (tjj * bignum) / xj;
            if (CNORM(j) > one) {
              // Scale by 1/CNORM(j) to avoid overflow when multiplying x(j)
              // times column j.
              rec = rec / CNORM(j);
            }
            Rscal(n, rec, x, 1);
            *scale = *scale * rec;
            xmax = xmax * rec;
          }
          X(j) = Ladiv(X(j), tjjs);
          xj = Cabs1(X(j));
        } else {
          // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and scale = 0, and
          // compute a solution to A*x = 0.
          for (i = 1; i <= n; ++i) X(i) = zero;
          X(j) = one;
          xj = one;
          *scale = zero;
          xmax = zero;
        }
      L110:

        // Scale x if necessary to avoid overflow when adding a multiple of
        // column j of A.
        if (xj > one) {
          rec = one / xj;
          if (CNORM(j) > (bignum - xmax) * rec) {
            // Scale x by 1/(2*abs(x(j))).
            rec = rec * half;
            Rscal(n, rec, x, 1);
            *scale = *scale * rec;
          }
        } else if (xj * CNORM(j) > (bignum - xmax)) {
          // Scale x by 1/2.
          Rscal(n, half, x, 1);
          *scale = *scale * half;
        }

        if (upper) {
          if (j > 1) {
            // Compute the update
            //    x(max(1,j-kd):j-1) := x(max(1,j-kd):j-1) -
            //                          x(j)* A(max(1,j-kd):j-1,j)
            jlen = Min(kd, j - 1);
            Axpy(jlen, -X(j) * tscal, AB.ptr(kd + 1 - jlen, j), 1,
                 X.ptr(j - jlen), 1);
            i = Iamax(j - 1, x, 1);
            xmax = Cabs1(X(i));
          }
        } else if (j < n) {
          // Compute the update
          //    x(j+1:min(j+kd,n)) := x(j+1:min(j+kd,n)) -
          //                          x(j) * A(j+1:min(j+kd,n),j)
          jlen = Min(kd, n - j);
          if (jlen > 0) {
            Axpy(jlen, -X(j) * tscal, AB.ptr(2, j), 1, X.ptr(j + 1), 1);
          }
          i = j + Iamax(n - j, X.ptr(j + 1), 1);
          xmax = Cabs1(X(i));
        }
      }
    } else if (Lsame(trans, 'T')) {
      // Solve A**T * x = b
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Compute x(j) = b(j) - sum A(k,j)*x(k).
        //                       k<>j
        xj = Cabs1(X(j));
        uscal = tscal;
        rec = one / Max(xmax, one);
        if (CNORM(j) > (bignum - xj) * rec) {
          // If x(j) could overflow, scale x by 1/(2*XMAX).
          rec = rec * half;
          if (nounit) {
            tjjs = AB(maind, j) * tscal;
          } else {
            tjjs = tscal;
          }
          tjj = Cabs1(tjjs);
          if (tjj > one) {
            // Divide by A(j,j) when scaling x if A(j,j) > 1.
            rec = Min(one, rec * tjj);
            uscal = Ladiv(uscal, tjjs);
          }
          if (rec < one) {
            Rscal(n, rec, x, 1);
            *scale = *scale * rec;
            xmax = xmax * rec;
          }
        }

        csumj = zero;
        if (uscal == T(one)) {
          // If the scaling needed for A in the dot product is 1, call xDOTU
          // to perform the dot product.
          if (upper) {
            jlen = Min(kd, j - 1);
            csumj = Dotu(jlen, AB.ptr(kd + 1 - jlen, j), 1, X.ptr(j - jlen), 1);
          } else {
            jlen = Min(kd, n - j);
            // JLEN > 1 in the reference (see above).
            if (jlen > 0) csumj = Dotu(jlen, AB.ptr(2, j), 1, X.ptr(j + 1), 1);
          }
        } else {
          // Otherwise, use in-line code for the dot product.
          if (upper) {
            jlen = Min(kd, j - 1);
            for (i = 1; i <= jlen; ++i) {
              csumj = csumj +
                      (AB(kd + i - jlen, j) * uscal) * X(j - jlen - 1 + i);
            }
          } else {
            jlen = Min(kd, n - j);
            for (i = 1; i <= jlen; ++i) {
              csumj = csumj + (AB(i + 1, j) * uscal) * X(j + i);
            }
          }
        }

        if (uscal == T(tscal)) {
          // Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) was not
          // used to scale the dotproduct.
          X(j) = X(j) - csumj;
          xj = Cabs1(X(j));
          if (nounit) {
            // Compute x(j) = x(j) / A(j,j), scaling if necessary.
            tjjs = AB(maind, j) * tscal;
          } else {
            tjjs = tscal;
            if (tscal == one) goto L160;
          }
          tjj = Cabs1(tjjs);
          if (tjj > smlnum) {
            // abs(A(j,j)) > SMLNUM:
            if (tjj < one) {
              if (xj > tjj * bignum) {
                // Scale X by 1/abs(x(j)).
                rec = one / xj;
                Rscal(n, rec, x, 1);
                *scale = *scale * rec;
                xmax = xmax * rec;
              }
            }
            X(j) = Ladiv(X(j), tjjs);
          } else if (tjj > zero) {
            // 0 < abs(A(j,j)) <= SMLNUM:
            if (xj > tjj * bignum) {
              // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
              rec = (tjj * bignum) / xj;
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
            X(j) = Ladiv(X(j), tjjs);
          } else {
            // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and scale = 0 and
            // compute a solution to A**T *x = 0.
            for (i = 1; i <= n; ++i) X(i) = zero;
            X(j) = one;
            *scale = zero;
            xmax = zero;
          }
        L160:;
        } else {
          // Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot product has
          // already been divided by 1/A(j,j).
          X(j) = Ladiv(X(j), tjjs) - csumj;
        }
        xmax = Max(xmax, Cabs1(X(j)));
      }
    } else {
      // Solve A**H * x = b
      for (j = jfirst; jinc > 0 ? j <= jlast : j >= jlast; j += jinc) {
        // Compute x(j) = b(j) - sum A(k,j)*x(k).
        //                       k<>j
        xj = Cabs1(X(j));
        uscal = tscal;
        rec = one / Max(xmax, one);
        if (CNORM(j) > (bignum - xj) * rec) {
          // If x(j) could overflow, scale x by 1/(2*XMAX).
          rec = rec * half;
          if (nounit) {
            tjjs = std::conj(AB(maind, j)) * tscal;
          } else {
            tjjs = tscal;
          }
          tjj = Cabs1(tjjs);
          if (tjj > one) {
            // Divide by A(j,j) when scaling x if A(j,j) > 1.
            rec = Min(one, rec * tjj);
            uscal = Ladiv(uscal, tjjs);
          }
          if (rec < one) {
            Rscal(n, rec, x, 1);
            *scale = *scale * rec;
            xmax = xmax * rec;
          }
        }

        csumj = zero;
        if (uscal == T(one)) {
          // If the scaling needed for A in the dot product is 1, call xDOTC
          // to perform the dot product.
          if (upper) {
            jlen = Min(kd, j - 1);
            csumj = Dotc(jlen, AB.ptr(kd + 1 - jlen, j), 1, X.ptr(j - jlen), 1);
          } else {
            jlen = Min(kd, n - j);
            // JLEN > 1 in the reference (see above).
            if (jlen > 0) csumj = Dotc(jlen, AB.ptr(2, j), 1, X.ptr(j + 1), 1);
          }
        } else {
          // Otherwise, use in-line code for the dot product.
          if (upper) {
            jlen = Min(kd, j - 1);
            for (i = 1; i <= jlen; ++i) {
              csumj = csumj + (std::conj(AB(kd + i - jlen, j)) * uscal) *
                                  X(j - jlen - 1 + i);
            }
          } else {
            jlen = Min(kd, n - j);
            for (i = 1; i <= jlen; ++i) {
              csumj = csumj + (std::conj(AB(i + 1, j)) * uscal) * X(j + i);
            }
          }
        }

        if (uscal == T(tscal)) {
          // Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) was not
          // used to scale the dotproduct.
          X(j) = X(j) - csumj;
          xj = Cabs1(X(j));
          if (nounit) {
            // Compute x(j) = x(j) / A(j,j), scaling if necessary.
            tjjs = std::conj(AB(maind, j)) * tscal;
          } else {
            tjjs = tscal;
            if (tscal == one) goto L210;
          }
          tjj = Cabs1(tjjs);
          if (tjj > smlnum) {
            // abs(A(j,j)) > SMLNUM:
            if (tjj < one) {
              if (xj > tjj * bignum) {
                // Scale X by 1/abs(x(j)).
                rec = one / xj;
                Rscal(n, rec, x, 1);
                *scale = *scale * rec;
                xmax = xmax * rec;
              }
            }
            X(j) = Ladiv(X(j), tjjs);
          } else if (tjj > zero) {
            // 0 < abs(A(j,j)) <= SMLNUM:
            if (xj > tjj * bignum) {
              // Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
              rec = (tjj * bignum) / xj;
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
            X(j) = Ladiv(X(j), tjjs);
          } else {
            // A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and scale = 0 and
            // compute a solution to A**H *x = 0.
            for (i = 1; i <= n; ++i) X(i) = zero;
            X(j) = one;
            *scale = zero;
            xmax = zero;
          }
        L210:;
        } else {
          // Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot product has
          // already been divided by 1/A(j,j).
          X(j) = Ladiv(X(j), tjjs) - csumj;
        }
        xmax = Max(xmax, Cabs1(X(j)));
      }
    }
    *scale = *scale / tscal;
  }

  // Scale the column norms by 1/TSCAL for return.
  if (tscal != one) Scal(n, one / tscal, cnorm, 1);

  return info;
}

// xLATBS: solves A * x = s*b, A**T * x = s*b or A**H * x = s*b (TRANS = 'N',
// 'T' or 'C') with the upper or lower (UPLO) triangular N-by-N band matrix
// A with KD super- or subdiagonals (band storage in AB, LDAB >= KD+1; unit
// diagonal for DIAG = 'U'), b and x of N elements (x overwrites b), and a
// scale factor s <= 1 (SCALE) chosen so that the components of x do not
// overflow. If the unscaled problem would not cause overflow, it calls the
// Level 2 BLAS xTBSV; otherwise it solves column by column, scaling x.
// CNORM(j) holds the 1-norms of the off-diagonal parts of the columns of
// A: computed here for NORMIN = 'N', given for NORMIN = 'Y'. If A is
// singular (A(j, j) = 0 for some j), s is 0 and x is a nontrivial solution
// of A * x = 0.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Latbs(char uplo, char trans, char diag, char normin, blasint n,
              blasint kd, const T* ab, blasint ldab, T* x, RealT<T>* scale,
              RealT<T>* cnorm) {
  if constexpr (IsComplex<T>::value) {
    return LatbsComplex(uplo, trans, diag, normin, n, kd, ab, ldab, x, scale,
                        cnorm);
  } else {
    return LatbsReal(uplo, trans, diag, normin, n, kd, ab, ldab, x, scale,
                     cnorm);
  }
}

// xGBCON: estimates the reciprocal of the condition number of the N-by-N
// band matrix A in the 1-norm (NORM = '1' or 'O') or the infinity norm
// ('I'), with its LU factorization computed by xGBTRF (AB, LDAB >=
// 2*KL+KU+1, and IPIV) and ANORM, the norm of A: RCOND = 1 / (ANORM *
// norm(inv(A))), with norm(inv(A)) estimated by xLACN2.
//
// The real routine uses WORK (3*N elements) and IWORK (N), the complex one
// WORK (2*N) and RWORK (N). As in Gecon, RWORK names the real routine's
// WORK(2*N+1:3*N) for real T; rwork is used for complex T only, iwork for
// real T only.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Gbcon(char norm, blasint n, blasint kl, blasint ku, const T* ab,
              blasint ldab, const blasint* ipiv, RealT<T> anorm,
              RealT<T>* rcond, T* work, RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> AB(ab, ldab);
  const Vector<const blasint> IPIV(ipiv);
  const Vector<T> WORK(work);
  R* rw;
  if constexpr (IsComplex<T>::value) {
    rw = rwork;
  } else {
    rw = WORK.ptr(2 * n + 1);
  }
  blasint isave[3];
  blasint ix, j, jp, kase, kase1, kd, lm;
  bool lnoti, onenrm;
  R ainvnm, scale, smlnum;
  T t;
  char normin;

  // Test the input parameters.
  blasint info = 0;
  onenrm = norm == '1' || Lsame(norm, 'O');
  if (!onenrm && !Lsame(norm, 'I')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (kl < 0) {
    info = -3;
  } else if (ku < 0) {
    info = -4;
  } else if (ldab < 2 * kl + ku + 1) {
    info = -6;
  } else if (anorm < zero) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("GBCON", -info);
    return info;
  }

  // Quick return if possible.
  *rcond = zero;
  if (n == 0) {
    *rcond = one;
    return info;
  } else if (anorm == zero) {
    return info;
  }

  smlnum = Lamch<R>('S');

  // Estimate the norm of inv(A).
  ainvnm = zero;
  normin = 'N';
  if (onenrm) {
    kase1 = 1;
  } else {
    kase1 = 2;
  }
  kd = kl + ku + 1;
  lnoti = kl > 0;
  kase = 0;
  for (;;) {
    if constexpr (IsComplex<T>::value) {
      Lacn2(n, WORK.ptr(n + 1), work, &ainvnm, &kase, isave);
    } else {
      Lacn2(n, WORK.ptr(n + 1), work, iwork, &ainvnm, &kase, isave);
    }
    if (kase == 0) break;
    if (kase == kase1) {
      // Multiply by inv(L).
      if (lnoti) {
        for (j = 1; j <= n - 1; ++j) {
          lm = Min(kl, n - j);
          jp = IPIV(j);
          t = WORK(jp);
          if (jp != j) {
            WORK(jp) = WORK(j);
            WORK(j) = t;
          }
          Axpy(lm, -t, AB.ptr(kd + 1, j), 1, WORK.ptr(j + 1), 1);
        }
      }

      // Multiply by inv(U).
      Latbs('U', 'N', 'N', normin, n, kl + ku, ab, ldab, work, &scale, rw);
    } else {
      // Multiply by inv(U**T) or inv(U**H).
      Latbs('U', ConjTrans<T>(), 'N', normin, n, kl + ku, ab, ldab, work,
            &scale, rw);

      // Multiply by inv(L**T) or inv(L**H).
      if (lnoti) {
        for (j = n - 1; j >= 1; --j) {
          lm = Min(kl, n - j);
          WORK(j) = WORK(j) - Dotc(lm, AB.ptr(kd + 1, j), 1, WORK.ptr(j + 1), 1);
          jp = IPIV(j);
          if (jp != j) {
            t = WORK(jp);
            WORK(jp) = WORK(j);
            WORK(j) = t;
          }
        }
      }
    }

    // Divide X by 1/SCALE if doing so will not cause overflow.
    normin = 'Y';
    if (scale != one) {
      ix = Iamax(n, work, 1);
      if (scale < Cabs1(WORK(ix)) * smlnum || scale == zero) return info;
      Rscl(n, scale, work, 1);
    }
  }

  // Compute the estimate of the reciprocal condition number.
  if (ainvnm != zero) *rcond = (one / ainvnm) / anorm;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Gbtf2<T>(blasint, blasint, blasint, blasint, T*, blasint,  \
                            blasint*);                                        \
  template blasint Gbtrf<T>(blasint, blasint, blasint, blasint, T*, blasint,  \
                            blasint*);                                        \
  template blasint Gbtrs<T>(char, blasint, blasint, blasint, blasint,         \
                            const T*, blasint, const blasint*, T*, blasint);  \
  template blasint Gbsv<T>(blasint, blasint, blasint, blasint, T*, blasint,   \
                           blasint*, T*, blasint);                            \
  template blasint Latbs<T>(char, char, char, char, blasint, blasint,         \
                            const T*, blasint, T*, RealT<T>*, RealT<T>*);     \
  template blasint Gbcon<T>(char, blasint, blasint, blasint, const T*,        \
                            blasint, const blasint*, RealT<T>, RealT<T>*, T*, \
                            RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GBSV(prefix, T, R)                                      \
  HWYBLAS_API void prefix##gbtf2_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  const hwyblas_int* kl,                       \
                                  const hwyblas_int* ku, R* ab,                \
                                  const hwyblas_int* ldab, hwyblas_int* ipiv,  \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Gbtf2(*m, *n, *kl, *ku, As<T>(ab), *ldab, ipiv);  \
  }                                                                            \
  HWYBLAS_API void prefix##gbtrf_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  const hwyblas_int* kl,                       \
                                  const hwyblas_int* ku, R* ab,                \
                                  const hwyblas_int* ldab, hwyblas_int* ipiv,  \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Gbtrf(*m, *n, *kl, *ku, As<T>(ab), *ldab, ipiv);  \
  }                                                                            \
  HWYBLAS_API void prefix##gbtrs_(                                             \
      const char* trans, const hwyblas_int* n, const hwyblas_int* kl,          \
      const hwyblas_int* ku, const hwyblas_int* nrhs, const R* ab,             \
      const hwyblas_int* ldab, const hwyblas_int* ipiv, R* b,                  \
      const hwyblas_int* ldb, hwyblas_int* info) {                             \
    *info = hwyblas::lapack::Gbtrs(*trans, *n, *kl, *ku, *nrhs,                \
                                   As<const T>(ab), *ldab, ipiv, As<T>(b),     \
                                   *ldb);                                      \
  }                                                                            \
  HWYBLAS_API void prefix##gbsv_(                                              \
      const hwyblas_int* n, const hwyblas_int* kl, const hwyblas_int* ku,      \
      const hwyblas_int* nrhs, R* ab, const hwyblas_int* ldab,                 \
      hwyblas_int* ipiv, R* b, const hwyblas_int* ldb, hwyblas_int* info) {    \
    *info = hwyblas::lapack::Gbsv(*n, *kl, *ku, *nrhs, As<T>(ab), *ldab, ipiv, \
                                  As<T>(b), *ldb);                             \
  }                                                                            \
  HWYBLAS_API void prefix##latbs_(                                             \
      const char* uplo, const char* trans, const char* diag,                   \
      const char* normin, const hwyblas_int* n, const hwyblas_int* kd,         \
      const R* ab, const hwyblas_int* ldab, R* x, R* scale, R* cnorm,          \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Latbs(*uplo, *trans, *diag, *normin, *n, *kd,     \
                                   As<const T>(ab), *ldab, As<T>(x), scale,    \
                                   cnorm);                                     \
  }

HWYBLAS_LAPACK_GBSV(s, float, float)
HWYBLAS_LAPACK_GBSV(d, double, double)
HWYBLAS_LAPACK_GBSV(c, c32, float)
HWYBLAS_LAPACK_GBSV(z, c64, double)

#undef HWYBLAS_LAPACK_GBSV

// xGBCON takes IWORK (real) or RWORK (complex).
#define HWYBLAS_LAPACK_GBCON_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##gbcon_(                                             \
      const char* norm, const hwyblas_int* n, const hwyblas_int* kl,           \
      const hwyblas_int* ku, const R* ab, const hwyblas_int* ldab,             \
      const hwyblas_int* ipiv, const R* anorm, R* rcond, R* work,              \
      hwyblas_int* iwork, hwyblas_int* info) {                                 \
    *info = hwyblas::lapack::Gbcon(*norm, *n, *kl, *ku, ab, *ldab, ipiv,       \
                                   *anorm, rcond, work,                        \
                                   static_cast<R*>(nullptr), iwork);           \
  }

#define HWYBLAS_LAPACK_GBCON_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##gbcon_(                                             \
      const char* norm, const hwyblas_int* n, const hwyblas_int* kl,           \
      const hwyblas_int* ku, const R* ab, const hwyblas_int* ldab,             \
      const hwyblas_int* ipiv, const R* anorm, R* rcond, R* work, R* rwork,    \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Gbcon(*norm, *n, *kl, *ku, As<const T>(ab),       \
                                   *ldab, ipiv, *anorm, rcond, As<T>(work),    \
                                   rwork, static_cast<hwyblas_int*>(nullptr)); \
  }

HWYBLAS_LAPACK_GBCON_REAL(s, float)
HWYBLAS_LAPACK_GBCON_REAL(d, double)
HWYBLAS_LAPACK_GBCON_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GBCON_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GBCON_REAL
#undef HWYBLAS_LAPACK_GBCON_COMPLEX

}  // extern "C"
