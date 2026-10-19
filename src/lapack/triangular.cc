// Triangular matrices: inverse (xTRTI2 unblocked, xTRTRI blocked),
// solution of triangular systems (xTRTRS with the BLAS, xLATRS with scaling
// against overflow), condition number (xTRCON) and error bounds (xTRRFS).
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

// xTRTI2: computes the inverse of the upper or lower (UPLO) triangular
// N-by-N matrix A, with unit diagonal for DIAG = 'U', in place, with the
// unblocked algorithm (Level 2 BLAS).
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Trti2(char uplo, char diag, blasint n, T* a, blasint lda) {
  const T one = T(1);
  const Matrix<T> A(a, lda);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool nounit = Lsame(diag, 'N');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("TRTI2", -info);
    return info;
  }

  if (upper) {
    // Compute inverse of upper triangular matrix.
    for (blasint j = 1; j <= n; ++j) {
      T ajj;
      if (nounit) {
        A(j, j) = one / A(j, j);
        ajj = -A(j, j);
      } else {
        ajj = -one;
      }

      // Compute elements 1:j-1 of j-th column.
      Trmv('U', 'N', diag, j - 1, a, lda, A.ptr(1, j), 1);
      Scal(j - 1, ajj, A.ptr(1, j), 1);
    }
  } else {
    // Compute inverse of lower triangular matrix.
    for (blasint j = n; j >= 1; --j) {
      T ajj;
      if (nounit) {
        A(j, j) = one / A(j, j);
        ajj = -A(j, j);
      } else {
        ajj = -one;
      }
      if (j < n) {
        // Compute elements j+1:n of j-th column.
        Trmv('L', 'N', diag, n - j, A.ptr(j + 1, j + 1), lda, A.ptr(j + 1, j),
             1);
        Scal(n - j, ajj, A.ptr(j + 1, j), 1);
      }
    }
  }
  return info;
}

// xTRTRI: computes the inverse of the upper or lower (UPLO) triangular
// N-by-N matrix A, with unit diagonal for DIAG = 'U', in place, in blocks of
// NB columns (from ILAENV; Level 3 BLAS) and xTRTI2 for the diagonal
// blocks.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if A(i, i) is exactly zero: the matrix is singular and its inverse
// is not computed.
template <typename T>
blasint Trtri(char uplo, char diag, blasint n, T* a, blasint lda) {
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool nounit = Lsame(diag, 'N');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("TRTRI", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Check for singularity if non-unit.
  if (nounit) {
    for (info = 1; info <= n; ++info) {
      if (A(info, info) == zero) return info;
    }
    info = 0;
  }

  // Determine the block size for this environment.
  const char opts[3] = {uplo, diag, '\0'};
  const blasint nb = Ilaenv<T>(1, "TRTRI", opts, n, -1, -1, -1);
  if (nb <= 1 || nb >= n) {
    // Use unblocked code.
    info = Trti2(uplo, diag, n, a, lda);
  } else {
    // Use blocked code.
    if (upper) {
      // Compute inverse of upper triangular matrix.
      for (blasint j = 1; j <= n; j += nb) {
        const blasint jb = Min(nb, n - j + 1);

        // Compute rows 1:j-1 of current block column.
        Trmm('L', 'U', 'N', diag, j - 1, jb, one, a, lda, A.ptr(1, j), lda);
        Trsm('R', 'U', 'N', diag, j - 1, jb, -one, A.ptr(j, j), lda,
             A.ptr(1, j), lda);

        // Compute inverse of current diagonal block.
        info = Trti2('U', diag, jb, A.ptr(j, j), lda);
      }
    } else {
      // Compute inverse of lower triangular matrix.
      const blasint nn = ((n - 1) / nb) * nb + 1;
      for (blasint j = nn; j >= 1; j -= nb) {
        const blasint jb = Min(nb, n - j + 1);
        if (j + jb <= n) {
          // Compute rows j+jb:n of current block column.
          Trmm('L', 'L', 'N', diag, n - j - jb + 1, jb, one,
               A.ptr(j + jb, j + jb), lda, A.ptr(j + jb, j), lda);
          Trsm('R', 'L', 'N', diag, n - j - jb + 1, jb, -one, A.ptr(j, j), lda,
               A.ptr(j + jb, j), lda);
        }

        // Compute inverse of current diagonal block.
        info = Trti2('L', diag, jb, A.ptr(j, j), lda);
      }
    }
  }
  return info;
}

// xTRTRS: solves A * X = B, A**T * X = B or A**H * X = B (TRANS = 'N', 'T'
// or 'C') for the N-by-NRHS matrix X, with the upper or lower (UPLO)
// triangular N-by-N matrix A, with unit diagonal for DIAG = 'U'. X
// overwrites B. A check for singularity precedes the solution; there is no
// scaling against overflow (see xLATRS).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if A(i, i) is exactly zero: A is singular and the solutions have not
// been computed.
template <typename T>
blasint Trtrs(char uplo, char trans, char diag, blasint n, blasint nrhs,
              const T* a, blasint lda, T* b, blasint ldb) {
  const T zero = T(0), one = T(1);
  const Matrix<const T> A(a, lda);

  // Test the input parameters.
  blasint info = 0;
  const bool nounit = Lsame(diag, 'N');
  if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!Lsame(trans, 'N') && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
    info = -2;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (nrhs < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<T>("TRTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Check for singularity.
  if (nounit) {
    for (info = 1; info <= n; ++info) {
      if (A(info, info) == zero) return info;
    }
  }
  info = 0;

  // Solve A * x = b, A**T * x = b, or A**H * x = b.
  Trsm('L', uplo, trans, diag, n, nrhs, one, a, lda, b, ldb);
  return info;
}

// xLATRS for real T (DLATRS / SLATRS); see Latrs.
template <typename T>
blasint LatrsReal(char uplo, char trans, char diag, char normin, blasint n,
                  const T* a, blasint lda, T* x, T* scale, T* cnorm) {
  using R = T;
  const R zero = R(0), half = R(0.5), one = R(1);
  const Matrix<const R> A(a, lda);
  const Vector<R> CNORM(cnorm);
  R work[1];
  const Vector<R> X(x);
  blasint i, imax, j, jfirst, jinc, jlast;
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
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<R>("LATRS", -info);
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
      for (j = 1; j <= n; ++j) CNORM(j) = Asum(j - 1, A.ptr(1, j), 1);
    } else {
      // A is lower triangular.
      for (j = 1; j <= n - 1; ++j) CNORM(j) = Asum(n - j, A.ptr(j + 1, j), 1);
      CNORM(n) = zero;
    }
  }

  // Scale the column norms by TSCAL if the maximum element in CNORM is
  // greater than BIGNUM.
  imax = Iamax(n, cnorm, 1);
  tmax = CNORM(imax);
  if (tmax <= bignum) {
    tscal = one;
  } else {
    // Avoid NaN generation if entries in CNORM exceed the overflow
    // threshold.
    if (tmax <= Lamch<R>('O')) {
      // Case 1: All entries in CNORM are valid floating-point numbers.
      tscal = one / (smlnum * tmax);
      Scal(n, tscal, cnorm, 1);
    } else {
      // Case 2: At least one column norm of A cannot be represented as
      // floating-point number. Find the offdiagonal entry A(I, J) with the
      // largest absolute value. If this entry is not +/- Infinity, use this
      // value as TSCAL.
      tmax = zero;
      if (upper) {
        // A is upper triangular.
        for (j = 2; j <= n; ++j) {
          tmax = Max(Lange('M', j - 1, 1, A.ptr(1, j), 1, work), tmax);
        }
      } else {
        // A is lower triangular.
        for (j = 1; j <= n - 1; ++j) {
          tmax = Max(Lange('M', n - j, 1, A.ptr(j + 1, j), 1, work), tmax);
        }
      }

      if (tmax <= Lamch<R>('O')) {
        tscal = one / (smlnum * tmax);
        for (j = 1; j <= n; ++j) {
          if (CNORM(j) <= Lamch<R>('O')) {
            CNORM(j) = CNORM(j) * tscal;
          } else {
            // Recompute the 1-norm without introducing Infinity in the
            // summation.
            CNORM(j) = zero;
            if (upper) {
              for (i = 1; i <= j - 1; ++i) {
                CNORM(j) = CNORM(j) + tscal * std::abs(A(i, j));
              }
            } else {
              for (i = j + 1; i <= n; ++i) {
                CNORM(j) = CNORM(j) + tscal * std::abs(A(i, j));
              }
            }
          }
        }
      } else {
        // At least one entry of A is not a valid floating-point entry. Rely
        // on TRSV to propagate Inf and NaN.
        Trsv(uplo, trans, diag, n, a, lda, x, 1);
        return info;
      }
    }
  }

  // Compute a bound on the computed solution vector to see if the Level 2
  // BLAS routine DTRSV can be used.
  j = Iamax(n, x, 1);
  xmax = std::abs(X(j));
  xbnd = xmax;
  if (notran) {
    // Compute the growth in A * x = b.
    if (upper) {
      jfirst = n;
      jlast = 1;
      jinc = -1;
    } else {
      jfirst = 1;
      jlast = n;
      jinc = 1;
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
        tjj = std::abs(A(j, j));
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
    } else {
      jfirst = n;
      jlast = 1;
      jinc = -1;
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
        tjj = std::abs(A(j, j));
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
    Trsv(uplo, trans, diag, n, a, lda, x, 1);
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
          tjjs = A(j, j) * tscal;
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
          // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0, and
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
            //    x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
            Axpy(j - 1, -X(j) * tscal, A.ptr(1, j), 1, x, 1);
            i = Iamax(j - 1, x, 1);
            xmax = std::abs(X(i));
          }
        } else {
          if (j < n) {
            // Compute the update
            //    x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
            Axpy(n - j, -X(j) * tscal, A.ptr(j + 1, j), 1, X.ptr(j + 1), 1);
            i = j + Iamax(n - j, X.ptr(j + 1), 1);
            xmax = std::abs(X(i));
          }
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
            tjjs = A(j, j) * tscal;
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
          // If the scaling needed for A in the dot product is 1, call DDOT
          // to perform the dot product.
          if (upper) {
            sumj = Dotu(j - 1, A.ptr(1, j), 1, x, 1);
          } else if (j < n) {
            sumj = Dotu(n - j, A.ptr(j + 1, j), 1, X.ptr(j + 1), 1);
          }
        } else {
          // Otherwise, use in-line code for the dot product.
          if (upper) {
            for (i = 1; i <= j - 1; ++i) {
              sumj = sumj + (A(i, j) * uscal) * X(i);
            }
          } else if (j < n) {
            for (i = j + 1; i <= n; ++i) {
              sumj = sumj + (A(i, j) * uscal) * X(i);
            }
          }
        }

        if (uscal == tscal) {
          // Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j) was not
          // used to scale the dotproduct.
          X(j) = X(j) - sumj;
          xj = std::abs(X(j));
          if (nounit) {
            tjjs = A(j, j) * tscal;
          } else {
            tjjs = tscal;
            if (tscal == one) goto L150;
          }

          // Compute x(j) = x(j) / A(j,j), scaling if necessary.
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
            // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0, and
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

// CABS2(z) = |Re(z) / 2| + |Im(z) / 2|, the statement function of xLATRS.
template <typename T>
RealT<T> Cabs2(T z) {
  using R = RealT<T>;
  return std::abs(std::real(z) / R(2)) + std::abs(std::imag(z) / R(2));
}

// xLATRS for complex T (CLATRS / ZLATRS); see Latrs. It differs from the
// real routine in its scaling (by BIGNUM/2, with |Re| + |Im| as the
// absolute value) and in its separate branches for A**T and A**H.
template <typename T>
blasint LatrsComplex(char uplo, char trans, char diag, char normin, blasint n,
                     const T* a, blasint lda, T* x, RealT<T>* scale,
                     RealT<T>* cnorm) {
  using R = RealT<T>;
  const R zero = R(0), half = R(0.5), one = R(1), two = R(2);
  const Matrix<const T> A(a, lda);
  const Vector<R> CNORM(cnorm);
  const Vector<T> X(x);
  blasint i, imax, j, jfirst, jinc, jlast;
  bool notran, nounit, upper;
  R bignum, grow, rec, smlnum, tjj, tmax, tscal, xbnd, xj, xmax;
  T csumj, uscal;
  // Set wherever it is read (USCAL differs from TSCAL only after a division
  // by TJJS); initialized for the compiler.
  T tjjs = T(zero);

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
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("LATRS", -info);
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
      for (j = 1; j <= n; ++j) CNORM(j) = Asum(j - 1, A.ptr(1, j), 1);
    } else {
      // A is lower triangular.
      for (j = 1; j <= n - 1; ++j) CNORM(j) = Asum(n - j, A.ptr(j + 1, j), 1);
      CNORM(n) = zero;
    }
  }

  // Scale the column norms by TSCAL if the maximum element in CNORM is
  // greater than BIGNUM/2.
  imax = Iamax(n, cnorm, 1);
  tmax = CNORM(imax);
  if (tmax <= bignum * half) {
    tscal = one;
  } else {
    // Avoid NaN generation if entries in CNORM exceed the overflow
    // threshold.
    if (tmax <= Lamch<R>('O')) {
      // Case 1: All entries in CNORM are valid floating-point numbers.
      tscal = half / (smlnum * tmax);
      Scal(n, tscal, cnorm, 1);
    } else {
      // Case 2: At least one column norm of A cannot be represented as a
      // floating-point number. Find the maximum offdiagonal absolute value
      // max( |Re(A(I,J))|, |Im(A(I,J)| ). If this entry is not +/-
      // Infinity, use this value as TSCAL.
      tmax = zero;
      if (upper) {
        // A is upper triangular.
        for (j = 2; j <= n; ++j) {
          for (i = 1; i <= j - 1; ++i) {
            tmax = Max(tmax, std::abs(std::real(A(i, j))),
                       std::abs(std::imag(A(i, j))));
          }
        }
      } else {
        // A is lower triangular.
        for (j = 1; j <= n - 1; ++j) {
          for (i = j + 1; i <= n; ++i) {
            tmax = Max(tmax, std::abs(std::real(A(i, j))),
                       std::abs(std::imag(A(i, j))));
          }
        }
      }

      if (tmax <= Lamch<R>('O')) {
        tscal = one / (smlnum * tmax);
        for (j = 1; j <= n; ++j) {
          if (CNORM(j) <= Lamch<R>('O')) {
            CNORM(j) = CNORM(j) * tscal;
          } else {
            // Recompute the 1-norm of each column without introducing
            // Infinity in the summation.
            tscal = two * tscal;
            CNORM(j) = zero;
            if (upper) {
              for (i = 1; i <= j - 1; ++i) {
                CNORM(j) = CNORM(j) + tscal * Cabs2(A(i, j));
              }
            } else {
              for (i = j + 1; i <= n; ++i) {
                CNORM(j) = CNORM(j) + tscal * Cabs2(A(i, j));
              }
            }
            tscal = tscal * half;
          }
        }
      } else {
        // At least one entry of A is not a valid floating-point entry. Rely
        // on TRSV to propagate Inf and NaN.
        Trsv(uplo, trans, diag, n, a, lda, x, 1);
        return info;
      }
    }
  }

  // Compute a bound on the computed solution vector to see if the Level 2
  // BLAS routine ZTRSV can be used.
  xmax = zero;
  for (j = 1; j <= n; ++j) xmax = Max(xmax, Cabs2(X(j)));
  xbnd = xmax;

  if (notran) {
    // Compute the growth in A * x = b.
    if (upper) {
      jfirst = n;
      jlast = 1;
      jinc = -1;
    } else {
      jfirst = 1;
      jlast = n;
      jinc = 1;
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

        tjjs = A(j, j);
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
    // Compute the growth in A**T * x = b or A**H * x = b.
    if (upper) {
      jfirst = 1;
      jlast = n;
      jinc = 1;
    } else {
      jfirst = n;
      jlast = 1;
      jinc = -1;
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

        tjjs = A(j, j);
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
    Trsv(uplo, trans, diag, n, a, lda, x, 1);
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
          tjjs = A(j, j) * tscal;
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
          // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0, and
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
            //    x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
            Axpy(j - 1, -X(j) * tscal, A.ptr(1, j), 1, x, 1);
            i = Iamax(j - 1, x, 1);
            xmax = Cabs1(X(i));
          }
        } else {
          if (j < n) {
            // Compute the update
            //    x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
            Axpy(n - j, -X(j) * tscal, A.ptr(j + 1, j), 1, X.ptr(j + 1), 1);
            i = j + Iamax(n - j, X.ptr(j + 1), 1);
            xmax = Cabs1(X(i));
          }
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
            tjjs = A(j, j) * tscal;
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
          // If the scaling needed for A in the dot product is 1, call ZDOTU
          // to perform the dot product.
          if (upper) {
            csumj = Dotu(j - 1, A.ptr(1, j), 1, x, 1);
          } else if (j < n) {
            csumj = Dotu(n - j, A.ptr(j + 1, j), 1, X.ptr(j + 1), 1);
          }
        } else {
          // Otherwise, use in-line code for the dot product.
          if (upper) {
            for (i = 1; i <= j - 1; ++i) {
              csumj = csumj + (A(i, j) * uscal) * X(i);
            }
          } else if (j < n) {
            for (i = j + 1; i <= n; ++i) {
              csumj = csumj + (A(i, j) * uscal) * X(i);
            }
          }
        }

        if (uscal == T(tscal)) {
          // Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) was not
          // used to scale the dotproduct.
          X(j) = X(j) - csumj;
          xj = Cabs1(X(j));
          if (nounit) {
            tjjs = A(j, j) * tscal;
          } else {
            tjjs = tscal;
            if (tscal == one) goto L160;
          }

          // Compute x(j) = x(j) / A(j,j), scaling if necessary.
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
            // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0 and
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
            tjjs = std::conj(A(j, j)) * tscal;
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
          // If the scaling needed for A in the dot product is 1, call ZDOTC
          // to perform the dot product.
          if (upper) {
            csumj = Dotc(j - 1, A.ptr(1, j), 1, x, 1);
          } else if (j < n) {
            csumj = Dotc(n - j, A.ptr(j + 1, j), 1, X.ptr(j + 1), 1);
          }
        } else {
          // Otherwise, use in-line code for the dot product.
          if (upper) {
            for (i = 1; i <= j - 1; ++i) {
              csumj = csumj + (std::conj(A(i, j)) * uscal) * X(i);
            }
          } else if (j < n) {
            for (i = j + 1; i <= n; ++i) {
              csumj = csumj + (std::conj(A(i, j)) * uscal) * X(i);
            }
          }
        }

        if (uscal == T(tscal)) {
          // Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) was not
          // used to scale the dotproduct.
          X(j) = X(j) - csumj;
          xj = Cabs1(X(j));
          if (nounit) {
            tjjs = std::conj(A(j, j)) * tscal;
          } else {
            tjjs = tscal;
            if (tscal == one) goto L210;
          }

          // Compute x(j) = x(j) / A(j,j), scaling if necessary.
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
            // A(j,j) = 0: Set x(1:n) = 0, x(j) = 1, and scale = 0 and
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

// xLATRS: solves A * x = s*b, A**T * x = s*b or A**H * x = s*b (TRANS = 'N',
// 'T' or 'C') with the upper or lower (UPLO) triangular N-by-N matrix A
// (unit diagonal for DIAG = 'U'), b and x of N elements (x overwrites b),
// and a scale factor s <= 1 (SCALE) chosen so that the components of x do
// not overflow. If the unscaled problem would not cause overflow, it calls
// the Level 2 BLAS xTRSV; otherwise it solves column by column, scaling
// x. CNORM(j) holds the 1-norms of the off-diagonal parts of the columns
// of A: computed here for NORMIN = 'N', given for NORMIN = 'Y'. If A is
// singular (A(j, j) = 0 for some j), s is 0 and x is a nontrivial solution
// of A * x = 0.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Latrs(char uplo, char trans, char diag, char normin, blasint n,
              const T* a, blasint lda, T* x, RealT<T>* scale, RealT<T>* cnorm) {
  if constexpr (IsComplex<T>::value) {
    return LatrsComplex(uplo, trans, diag, normin, n, a, lda, x, scale, cnorm);
  } else {
    return LatrsReal(uplo, trans, diag, normin, n, a, lda, x, scale, cnorm);
  }
}

// xTRCON: estimates the reciprocal of the condition number of the upper or
// lower (UPLO) triangular N-by-N matrix A (unit diagonal for DIAG = 'U') in
// the 1-norm (NORM = '1' or 'O') or the infinity norm ('I'): RCOND =
// 1 / (norm(A) * norm(inv(A))), with norm(inv(A)) estimated by xLACN2.
//
// The real routine uses WORK (3*N elements) and IWORK (N), the complex one
// WORK (2*N) and RWORK (N). Here RWORK names the real routine's WORK(2*N+1:3*N)
// for real T (the column norms for xLATRS); rwork is used for complex T only,
// iwork for real T only.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Trcon(char norm, char uplo, char diag, blasint n, const T* a,
              blasint lda, RealT<T>* rcond, T* work, RealT<T>* rwork,
              blasint* iwork) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Vector<T> WORK(work);
  R* rw;
  if constexpr (IsComplex<T>::value) {
    rw = rwork;
  } else {
    rw = WORK.ptr(2 * n + 1);
  }
  const Vector<R> RWORK(rw);
  blasint isave[3];

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool onenrm = norm == '1' || Lsame(norm, 'O');
  const bool nounit = Lsame(diag, 'N');
  if (!onenrm && !Lsame(norm, 'I')) {
    info = -1;
  } else if (!upper && !Lsame(uplo, 'L')) {
    info = -2;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("TRCON", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) {
    *rcond = one;
    return info;
  }

  *rcond = zero;
  const R smlnum = Lamch<R>('S') * R(Max<blasint>(1, n));

  // Compute the norm of the triangular matrix A. (The real routine passes
  // WORK as Lantr's workspace, which it uses before the rest.)
  R* lantr_work;
  if constexpr (IsComplex<T>::value) {
    lantr_work = rwork;
  } else {
    lantr_work = work;
  }
  const R anorm = Lantr(norm, uplo, diag, n, n, a, lda, lantr_work);

  // Continue only if ANORM > 0.
  if (anorm > zero) {
    // Estimate the norm of the inverse of A.
    R ainvnm = zero;
    char normin = 'N';
    const blasint kase1 = onenrm ? 1 : 2;
    blasint kase = 0;
    for (;;) {
      if constexpr (IsComplex<T>::value) {
        Lacn2(n, WORK.ptr(n + 1), work, &ainvnm, &kase, isave);
      } else {
        Lacn2(n, WORK.ptr(n + 1), work, iwork, &ainvnm, &kase, isave);
      }
      if (kase == 0) break;
      R scale;
      if (kase == kase1) {
        // Multiply by inv(A).
        Latrs(uplo, 'N', diag, normin, n, a, lda, work, &scale, rw);
      } else {
        // Multiply by inv(A**T) or inv(A**H).
        Latrs(uplo, ConjTrans<T>(), diag, normin, n, a, lda, work, &scale, rw);
      }
      normin = 'Y';

      // Multiply by 1/SCALE if doing so will not cause overflow.
      if (scale != one) {
        const blasint ix = Iamax(n, work, 1);
        const R xnorm = Cabs1(WORK(ix));
        if (scale < xnorm * smlnum || scale == zero) return info;
        Rscl(n, scale, work, 1);
      }
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != zero) *rcond = (one / anorm) / ainvnm;
  }
  return info;
}

// xTRRFS: error bounds and backward error estimates for the solution X
// (N-by-NRHS) of A * X = B, A**T * X = B or A**H * X = B (TRANS) with the
// upper or lower (UPLO) triangular N-by-N matrix A (unit diagonal for
// DIAG = 'U'), as computed by xTRTRS or otherwise: FERR(j) bounds the
// relative error of column j of X, BERR(j) is its componentwise relative
// backward error. X is not changed (there is no refinement).
//
// The real routine keeps W = abs(op(A))*abs(X) + abs(B) in WORK(1:N), the
// residual in WORK(N+1:2*N) and xLACN2's V in WORK(2*N+1:3*N) (with IWORK),
// the complex one W in RWORK and the residual and V in WORK(1:2*N). The
// views below have the complex routine's names; rwork is used for complex T
// only, iwork for real T only. (Where the real routine passes 'T' to xTRSV,
// this passes 'C', which the BLAS treat alike for real T.)
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Trrfs(char uplo, char trans, char diag, blasint n, blasint nrhs,
              const T* a, blasint lda, const T* b, blasint ldb, const T* x,
              blasint ldx, RealT<T>* ferr, RealT<T>* berr, T* work,
              RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const T one = T(1);
  const Matrix<const T> A(a, lda);
  const Matrix<const T> B(b, ldb);
  const Vector<R> BERR(berr);
  const Vector<R> FERR(ferr);
  const Matrix<const T> X(x, ldx);
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

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool notran = Lsame(trans, 'N');
  const bool nounit = Lsame(diag, 'N');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (!notran && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
    info = -2;
  } else if (!nounit && !Lsame(diag, 'U')) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (nrhs < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldx < Max<blasint>(1, n)) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>("TRRFS", -info);
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

  const char transn = notran ? 'N' : 'C';
  const char transt = notran ? 'C' : 'N';

  // NZ = maximum number of nonzero elements in each row of A, plus 1.
  const blasint nz = n + 1;
  const R eps = Lamch<R>('E');
  const R safmin = Lamch<R>('S');
  const R safe1 = nz * safmin;
  const R safe2 = safe1 / eps;

  // Do for each right hand side.
  for (blasint j = 1; j <= nrhs; ++j) {
    // Compute residual R = B - op(A) * X, where op(A) = A, A**T, or A**H,
    // depending on TRANS.
    Copy(n, X.ptr(1, j), 1, wk, 1);
    Trmv(uplo, trans, diag, n, a, lda, wk, 1);
    Axpy(n, -one, B.ptr(1, j), 1, wk, 1);

    // Compute componentwise relative backward error from formula
    //
    // max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
    //
    // where abs(Z) is the componentwise absolute value of the matrix or
    // vector Z. If the i-th component of the denominator is less than
    // SAFE2, then SAFE1 is added to the i-th components of the numerator
    // and denominator before dividing.
    for (blasint i = 1; i <= n; ++i) RWORK(i) = Cabs1(B(i, j));

    R s, xk;
    if (notran) {
      // Compute abs(A)*abs(X) + abs(B).
      if (upper) {
        if (nounit) {
          for (blasint k = 1; k <= n; ++k) {
            xk = Cabs1(X(k, j));
            for (blasint i = 1; i <= k; ++i) {
              RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
            }
          }
        } else {
          for (blasint k = 1; k <= n; ++k) {
            xk = Cabs1(X(k, j));
            for (blasint i = 1; i <= k - 1; ++i) {
              RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
            }
            RWORK(k) = RWORK(k) + xk;
          }
        }
      } else {
        if (nounit) {
          for (blasint k = 1; k <= n; ++k) {
            xk = Cabs1(X(k, j));
            for (blasint i = k; i <= n; ++i) {
              RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
            }
          }
        } else {
          for (blasint k = 1; k <= n; ++k) {
            xk = Cabs1(X(k, j));
            for (blasint i = k + 1; i <= n; ++i) {
              RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
            }
            RWORK(k) = RWORK(k) + xk;
          }
        }
      }
    } else {
      // Compute abs(A**T)*abs(X) + abs(B) (abs(A**H)*abs(X) + abs(B)).
      if (upper) {
        if (nounit) {
          for (blasint k = 1; k <= n; ++k) {
            s = zero;
            for (blasint i = 1; i <= k; ++i) {
              s = s + Cabs1(A(i, k)) * Cabs1(X(i, j));
            }
            RWORK(k) = RWORK(k) + s;
          }
        } else {
          for (blasint k = 1; k <= n; ++k) {
            s = Cabs1(X(k, j));
            for (blasint i = 1; i <= k - 1; ++i) {
              s = s + Cabs1(A(i, k)) * Cabs1(X(i, j));
            }
            RWORK(k) = RWORK(k) + s;
          }
        }
      } else {
        if (nounit) {
          for (blasint k = 1; k <= n; ++k) {
            s = zero;
            for (blasint i = k; i <= n; ++i) {
              s = s + Cabs1(A(i, k)) * Cabs1(X(i, j));
            }
            RWORK(k) = RWORK(k) + s;
          }
        } else {
          for (blasint k = 1; k <= n; ++k) {
            s = Cabs1(X(k, j));
            for (blasint i = k + 1; i <= n; ++i) {
              s = s + Cabs1(A(i, k)) * Cabs1(X(i, j));
            }
            RWORK(k) = RWORK(k) + s;
          }
        }
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

    // Bound error from formula
    //
    // norm(X - XTRUE) / norm(X) .le. FERR =
    // norm( abs(inv(op(A)))*
    //    ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
    //
    // where
    //   norm(Z) is the magnitude of the largest component of Z
    //   inv(op(A)) is the inverse of op(A)
    //   abs(Z) is the componentwise absolute value of the matrix or vector Z
    //   NZ is the maximum number of nonzeros in any row of A, plus 1
    //   EPS is machine epsilon
    //
    // The i-th component of abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B)) is
    // incremented by SAFE1 if the i-th component of abs(op(A))*abs(X) +
    // abs(B) is less than SAFE2.
    //
    // Use xLACN2 to estimate the infinity-norm of the matrix
    //    inv(op(A)) * diag(W),
    // where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) )))
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
        // Multiply by diag(W)*inv(op(A)**H).
        Trsv(uplo, transt, diag, n, a, lda, wk, 1);
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
      } else {
        // Multiply by inv(op(A))*diag(W).
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
        Trsv(uplo, transn, diag, n, a, lda, wk, 1);
      }
    }

    // Normalize error.
    R lstres = zero;
    for (blasint i = 1; i <= n; ++i) lstres = Max(lstres, Cabs1(X(i, j)));
    if (lstres != zero) FERR(j) = FERR(j) / lstres;
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                      \
  template blasint Trti2<T>(char, char, blasint, T*, blasint);             \
  template blasint Trtri<T>(char, char, blasint, T*, blasint);             \
  template blasint Trtrs<T>(char, char, char, blasint, blasint, const T*,  \
                            blasint, T*, blasint);                         \
  template blasint Latrs<T>(char, char, char, char, blasint, const T*,     \
                            blasint, T*, RealT<T>*, RealT<T>*);            \
  template blasint Trcon<T>(char, char, char, blasint, const T*, blasint,  \
                            RealT<T>*, T*, RealT<T>*, blasint*);           \
  template blasint Trrfs<T>(char, char, char, blasint, blasint, const T*,  \
                            blasint, const T*, blasint, const T*, blasint, \
                            RealT<T>*, RealT<T>*, T*, RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_TRIANGULAR(prefix, T, R)                                \
  HWYBLAS_API void prefix##trti2_(const char* uplo, const char* diag,          \
                                  const hwyblas_int* n, R* a,                  \
                                  const hwyblas_int* lda, hwyblas_int* info) { \
    *info = hwyblas::lapack::Trti2(*uplo, *diag, *n, As<T>(a), *lda);          \
  }                                                                            \
  HWYBLAS_API void prefix##trtri_(const char* uplo, const char* diag,          \
                                  const hwyblas_int* n, R* a,                  \
                                  const hwyblas_int* lda, hwyblas_int* info) { \
    *info = hwyblas::lapack::Trtri(*uplo, *diag, *n, As<T>(a), *lda);          \
  }                                                                            \
  HWYBLAS_API void prefix##trtrs_(const char* uplo, const char* trans,         \
                                  const char* diag, const hwyblas_int* n,      \
                                  const hwyblas_int* nrhs, const R* a,         \
                                  const hwyblas_int* lda, R* b,                \
                                  const hwyblas_int* ldb, hwyblas_int* info) { \
    *info = hwyblas::lapack::Trtrs(*uplo, *trans, *diag, *n, *nrhs,            \
                                   As<const T>(a), *lda, As<T>(b), *ldb);      \
  }                                                                            \
  HWYBLAS_API void prefix##latrs_(                                             \
      const char* uplo, const char* trans, const char* diag,                   \
      const char* normin, const hwyblas_int* n, const R* a,                    \
      const hwyblas_int* lda, R* x, R* scale, R* cnorm, hwyblas_int* info) {   \
    *info =                                                                    \
        hwyblas::lapack::Latrs(*uplo, *trans, *diag, *normin, *n,              \
                               As<const T>(a), *lda, As<T>(x), scale, cnorm);  \
  }

HWYBLAS_LAPACK_TRIANGULAR(s, float, float)
HWYBLAS_LAPACK_TRIANGULAR(d, double, double)
HWYBLAS_LAPACK_TRIANGULAR(c, c32, float)
HWYBLAS_LAPACK_TRIANGULAR(z, c64, double)

#undef HWYBLAS_LAPACK_TRIANGULAR

// xTRCON and xTRRFS take IWORK (real) or RWORK (complex).
#define HWYBLAS_LAPACK_TRIANGULAR_REAL(prefix, R)                             \
  HWYBLAS_API void prefix##trcon_(                                            \
      const char* norm, const char* uplo, const char* diag,                   \
      const hwyblas_int* n, const R* a, const hwyblas_int* lda, R* rcond,     \
      R* work, hwyblas_int* iwork, hwyblas_int* info) {                       \
    *info = hwyblas::lapack::Trcon(*norm, *uplo, *diag, *n, a, *lda, rcond,   \
                                   work, static_cast<R*>(nullptr), iwork);    \
  }                                                                           \
  HWYBLAS_API void prefix##trrfs_(                                            \
      const char* uplo, const char* trans, const char* diag,                  \
      const hwyblas_int* n, const hwyblas_int* nrhs, const R* a,              \
      const hwyblas_int* lda, const R* b, const hwyblas_int* ldb, const R* x, \
      const hwyblas_int* ldx, R* ferr, R* berr, R* work, hwyblas_int* iwork,  \
      hwyblas_int* info) {                                                    \
    *info = hwyblas::lapack::Trrfs(*uplo, *trans, *diag, *n, *nrhs, a, *lda,  \
                                   b, *ldb, x, *ldx, ferr, berr, work,        \
                                   static_cast<R*>(nullptr), iwork);          \
  }

#define HWYBLAS_LAPACK_TRIANGULAR_COMPLEX(prefix, T, R)                        \
  HWYBLAS_API void prefix##trcon_(                                             \
      const char* norm, const char* uplo, const char* diag,                    \
      const hwyblas_int* n, const R* a, const hwyblas_int* lda, R* rcond,      \
      R* work, R* rwork, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Trcon(*norm, *uplo, *diag, *n, As<const T>(a),    \
                                   *lda, rcond, As<T>(work), rwork,            \
                                   static_cast<hwyblas_int*>(nullptr));        \
  }                                                                            \
  HWYBLAS_API void prefix##trrfs_(                                             \
      const char* uplo, const char* trans, const char* diag,                   \
      const hwyblas_int* n, const hwyblas_int* nrhs, const R* a,               \
      const hwyblas_int* lda, const R* b, const hwyblas_int* ldb, const R* x,  \
      const hwyblas_int* ldx, R* ferr, R* berr, R* work, R* rwork,             \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Trrfs(                                            \
        *uplo, *trans, *diag, *n, *nrhs, As<const T>(a), *lda, As<const T>(b), \
        *ldb, As<const T>(x), *ldx, ferr, berr, As<T>(work), rwork,            \
        static_cast<hwyblas_int*>(nullptr));                                   \
  }

HWYBLAS_LAPACK_TRIANGULAR_REAL(s, float)
HWYBLAS_LAPACK_TRIANGULAR_REAL(d, double)
HWYBLAS_LAPACK_TRIANGULAR_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_TRIANGULAR_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_TRIANGULAR_REAL
#undef HWYBLAS_LAPACK_TRIANGULAR_COMPLEX

}  // extern "C"
