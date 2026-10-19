// The expert driver of general linear systems, xGESVX, and its parts: the
// condition number estimate (xGECON), equilibration (xGEEQU, xLAQGE) and
// iterative refinement with error bounds (xGERFS).
//
// The real and the complex routines differ in their workspaces: the real
// ones use WORK and IWORK, the complex ones WORK and RWORK, with the real
// quantities (the column norms of xLATRS, abs(op(A))*abs(X) + abs(B), the
// pivot growth factor) in RWORK. One template serves both; its views are
// named as in the complex routine, and point into the real routine's WORK
// for real T (see each routine). rwork is used for complex T only, iwork
// for real T only.
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

// xGECON: estimates the reciprocal of the condition number of the N-by-N
// matrix A in the 1-norm (NORM = '1' or 'O') or the infinity norm ('I'),
// from its LU factorization by xGETRF (A) and the norm ANORM of the original
// matrix: RCOND = 1 / (ANORM * norm(inv(A))), with norm(inv(A)) estimated
// by xLACN2.
//
// The real routine uses WORK (4*N elements; the column norms of xLATRS in
// WORK(2*N+1:4*N)) and IWORK (N), the complex one WORK (2*N) and RWORK
// (2*N; the column norms).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value (also
// -5 if ANORM is NaN or infinite), or 1 if RCOND is NaN or infinite, or
// zero (as computed; the matrix is singular).
template <typename T>
blasint Gecon(char norm, blasint n, const T* a, blasint lda, RealT<T> anorm,
              RealT<T>* rcond, T* work, RealT<T>* rwork, blasint* iwork) {
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

  const R hugeval = Lamch<R>('O');

  // Test the input parameters.
  blasint info = 0;
  const bool onenrm = norm == '1' || Lsame(norm, 'O');
  if (!onenrm && !Lsame(norm, 'I')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  } else if (anorm < zero) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("GECON", -info);
    return info;
  }

  // Quick return if possible.
  *rcond = zero;
  if (n == 0) {
    *rcond = one;
    return info;
  } else if (anorm == zero) {
    return info;
  } else if (std::isnan(anorm)) {
    *rcond = anorm;
    info = -5;
    return info;
  } else if (anorm > hugeval) {
    info = -5;
    return info;
  }

  const R smlnum = Lamch<R>('S');

  // Estimate the norm of inv(A).
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
    R sl, su;
    if (kase == kase1) {
      // Multiply by inv(L).
      Latrs('L', 'N', 'U', normin, n, a, lda, work, &sl, rw);

      // Multiply by inv(U).
      Latrs('U', 'N', 'N', normin, n, a, lda, work, &su, RWORK.ptr(n + 1));
    } else {
      // Multiply by inv(U**T) or inv(U**H).
      Latrs('U', ConjTrans<T>(), 'N', normin, n, a, lda, work, &su,
            RWORK.ptr(n + 1));

      // Multiply by inv(L**T) or inv(L**H).
      Latrs('L', ConjTrans<T>(), 'U', normin, n, a, lda, work, &sl, rw);
    }

    // Divide X by 1/(SL*SU) if doing so will not cause overflow.
    const R scale = sl * su;
    normin = 'Y';
    if (scale != one) {
      const blasint ix = Iamax(n, work, 1);
      if (scale < Cabs1(WORK(ix)) * smlnum || scale == zero) return info;
      Rscl(n, scale, work, 1);
    }
  }

  // Compute the estimate of the reciprocal condition number.
  if (ainvnm != zero) {
    *rcond = (one / ainvnm) / anorm;
  } else {
    info = 1;
    return info;
  }

  // Check for NaNs and Infs.
  if (std::isnan(*rcond) || *rcond > hugeval) info = 1;
  return info;
}

// xGEEQU: computes row and column scalings R (M elements) and C (N) that
// equilibrate the M-by-N matrix A: the largest element of each row and
// column of diag(R) * A * diag(C) has absolute value 1 (|Re| + |Im| for
// complex T). R(i) and C(j) are restricted to the range [SMLNUM, BIGNUM].
// ROWCND = min(R(i)) / max(R(i)) and COLCND = min(C(j)) / max(C(j)) tell
// whether scaling is worthwhile (not if >= 0.1); AMAX is the largest
// absolute value in A.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0: row i of A is exactly zero (i <= M), or column i - M is exactly
// zero (i > M).
template <typename T>
blasint Geequ(blasint m, blasint n, const T* a, blasint lda, RealT<T>* r,
              RealT<T>* c, RealT<T>* rowcnd, RealT<T>* colcnd, RealT<T>* amax) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> A(a, lda);
  const Vector<R> C(c);
  // The Fortran array R, renamed to keep R the real type.
  const Vector<R> Rs(r);

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
    Xerbla<T>("GEEQU", -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) {
    *rowcnd = one;
    *colcnd = one;
    *amax = zero;
    return info;
  }

  // Get machine constants.
  const R smlnum = Lamch<R>('S');
  const R bignum = one / smlnum;

  // Compute row scale factors.
  for (blasint i = 1; i <= m; ++i) Rs(i) = zero;

  // Find the maximum element in each row.
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) Rs(i) = Max(Rs(i), Cabs1(A(i, j)));
  }

  // Find the maximum and minimum scale factors.
  R rcmin = bignum;
  R rcmax = zero;
  for (blasint i = 1; i <= m; ++i) {
    rcmax = Max(rcmax, Rs(i));
    rcmin = Min(rcmin, Rs(i));
  }
  *amax = rcmax;

  if (rcmin == zero) {
    // Find the first zero scale factor and return an error code.
    for (blasint i = 1; i <= m; ++i) {
      if (Rs(i) == zero) {
        info = i;
        return info;
      }
    }
  } else {
    // Invert the scale factors.
    for (blasint i = 1; i <= m; ++i) {
      Rs(i) = one / Min(Max(Rs(i), smlnum), bignum);
    }

    // Compute ROWCND = min(R(I)) / max(R(I)).
    *rowcnd = Max(rcmin, smlnum) / Min(rcmax, bignum);
  }

  // Compute column scale factors.
  for (blasint j = 1; j <= n; ++j) C(j) = zero;

  // Find the maximum element in each column, assuming the row scaling
  // computed above.
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      C(j) = Max(C(j), Cabs1(A(i, j)) * Rs(i));
    }
  }

  // Find the maximum and minimum scale factors.
  rcmin = bignum;
  rcmax = zero;
  for (blasint j = 1; j <= n; ++j) {
    rcmin = Min(rcmin, C(j));
    rcmax = Max(rcmax, C(j));
  }

  if (rcmin == zero) {
    // Find the first zero scale factor and return an error code.
    for (blasint j = 1; j <= n; ++j) {
      if (C(j) == zero) {
        info = m + j;
        return info;
      }
    }
  } else {
    // Invert the scale factors.
    for (blasint j = 1; j <= n; ++j) {
      C(j) = one / Min(Max(C(j), smlnum), bignum);
    }

    // Compute COLCND = min(C(J)) / max(C(J)).
    *colcnd = Max(rcmin, smlnum) / Min(rcmax, bignum);
  }
  return info;
}

// xLAQGE: equilibrates the M-by-N matrix A with the row and column scalings
// R and C (from xGEEQU), as far as worthwhile: rows if ROWCND < 0.1 or AMAX
// is close to underflow or overflow, columns if COLCND < 0.1. EQUED returns
// what was done: 'N' nothing, 'R' A := diag(R) * A, 'C' A := A * diag(C),
// 'B' A := diag(R) * A * diag(C).
template <typename T>
void Laqge(blasint m, blasint n, T* a, blasint lda, const RealT<T>* r,
           const RealT<T>* c, RealT<T> rowcnd, RealT<T> colcnd, RealT<T> amax,
           char* equed) {
  using R = RealT<T>;
  const R one = R(1), thresh = R(0.1);
  const Matrix<T> A(a, lda);
  const Vector<const R> C(c);
  // The Fortran array R, renamed to keep R the real type.
  const Vector<const R> Rs(r);

  // Quick return if possible.
  if (m <= 0 || n <= 0) {
    *equed = 'N';
    return;
  }

  // Initialize LARGE and SMALL.
  const R small = Lamch<R>('S') / Lamch<R>('P');
  const R large = one / small;

  if (rowcnd >= thresh && amax >= small && amax <= large) {
    // No row scaling.
    if (colcnd >= thresh) {
      // No column scaling.
      *equed = 'N';
    } else {
      // Column scaling.
      for (blasint j = 1; j <= n; ++j) {
        const R cj = C(j);
        for (blasint i = 1; i <= m; ++i) A(i, j) = cj * A(i, j);
      }
      *equed = 'C';
    }
  } else if (colcnd >= thresh) {
    // Row scaling, no column scaling.
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = 1; i <= m; ++i) A(i, j) = Rs(i) * A(i, j);
    }
    *equed = 'R';
  } else {
    // Row and column scaling.
    for (blasint j = 1; j <= n; ++j) {
      const R cj = C(j);
      for (blasint i = 1; i <= m; ++i) A(i, j) = cj * Rs(i) * A(i, j);
    }
    *equed = 'B';
  }
}

// xGERFS: improves the computed solution X (N-by-NRHS) of A * X = B,
// A**T * X = B or A**H * X = B (TRANS) by iterative refinement, with the LU
// factorization of A by xGETRF (AF and IPIV), and computes error bounds:
// FERR(j) bounds the relative error of column j of X, BERR(j) is its
// componentwise relative backward error.
//
// The real routine keeps W = abs(op(A))*abs(X) + abs(B) in WORK(1:N), the
// residual in WORK(N+1:2*N) and xLACN2's V in WORK(2*N+1:3*N) (with IWORK),
// the complex one W in RWORK and the residual and V in WORK(1:2*N). The
// views below have the complex routine's names. (Where the real routine
// passes 'T' to xGETRS, this passes 'C', which xGETRS treats alike for
// real T.)
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Gerfs(char trans, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* ferr, RealT<T>* berr,
              T* work, RealT<T>* rwork, blasint* iwork) {
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
  } else if (ldaf < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -10;
  } else if (ldx < Max<blasint>(1, n)) {
    info = -12;
  }
  if (info != 0) {
    Xerbla<T>("GERFS", -info);
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
    blasint count = 1;
    R lstres = three;
    R s;
    for (;;) {
      // Loop until stopping criterion is satisfied.
      //
      // Compute residual R = B - op(A) * X, where op(A) = A, A**T, or A**H,
      // depending on TRANS.
      Copy(n, B.ptr(1, j), 1, wk, 1);
      Gemv(trans, n, n, -one, a, lda, X.ptr(1, j), 1, one, wk, 1);

      // Compute componentwise relative backward error from formula
      //
      // max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
      //
      // where abs(Z) is the componentwise absolute value of the matrix or
      // vector Z. If the i-th component of the denominator is less than
      // SAFE2, then SAFE1 is added to the i-th components of the numerator
      // and denominator before dividing.
      for (blasint i = 1; i <= n; ++i) RWORK(i) = Cabs1(B(i, j));

      // Compute abs(op(A))*abs(X) + abs(B).
      if (notran) {
        for (blasint k = 1; k <= n; ++k) {
          const R xk = Cabs1(X(k, j));
          for (blasint i = 1; i <= n; ++i) {
            RWORK(i) = RWORK(i) + Cabs1(A(i, k)) * xk;
          }
        }
      } else {
        for (blasint k = 1; k <= n; ++k) {
          s = zero;
          for (blasint i = 1; i <= n; ++i) {
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
        Getrs(trans, n, 1, af, ldaf, ipiv, wk, n);
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
        Getrs(transt, n, 1, af, ldaf, ipiv, wk, n);
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
      } else {
        // Multiply by inv(op(A))*diag(W).
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
        Getrs(transn, n, 1, af, ldaf, ipiv, wk, n);
      }
    }

    // Normalize error.
    lstres = zero;
    for (blasint i = 1; i <= n; ++i) lstres = Max(lstres, Cabs1(X(i, j)));
    if (lstres != zero) FERR(j) = FERR(j) / lstres;
  }
  return info;
}

// xGESVX: solves A * X = B, A**T * X = B or A**H * X = B (TRANS) for the
// N-by-N matrix A and N-by-NRHS matrices X and B with the LU factorization
// of A, and returns error bounds and an estimate of the condition number.
// FACT selects whether the factorization AF, IPIV is given ('F'; with the
// equilibration EQUED, R, C that it is of), computed ('N'), or computed
// after equilibrating A (and B) if worthwhile ('E'; A then returns the
// equilibrated matrix, EQUED what was done). X is refined with xGERFS and
// returned for the original system. RCOND estimates the reciprocal
// condition number of the (equilibrated) A.
//
// The real routine uses WORK (4*N elements) and IWORK (N), the complex one
// WORK (2*N) and RWORK (2*N). The reciprocal pivot growth factor
// norm(A) / norm(U) ('M' norm, of the leading INFO columns if INFO > 0) is
// returned in WORK(1) for real T and RWORK(1) for complex T, where the
// real routine also passes WORK to xLANGE and xLANTR. The view RWORK is the
// real routine's WORK for real T.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, i <= N
// if U(i, i) is exactly zero (the factorization has been completed, but
// the solution and error bounds have not been computed; RCOND = 0), or
// N + 1 if U is nonsingular but RCOND is less than the unit roundoff (the
// matrix is singular to working precision; the solution and error bounds
// have been computed).
template <typename T>
blasint Gesvx(char fact, char trans, blasint n, blasint nrhs, T* a, blasint lda,
              T* af, blasint ldaf, blasint* ipiv, char* equed, RealT<T>* r,
              RealT<T>* c, T* b, blasint ldb, T* x, blasint ldx,
              RealT<T>* rcond, RealT<T>* ferr, RealT<T>* berr, T* work,
              RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const Matrix<T> B(b, ldb);
  const Vector<R> C(c);
  const Vector<R> FERR(ferr);
  // The Fortran array R, renamed to keep R the real type.
  const Vector<R> Rs(r);
  const Matrix<T> X(x, ldx);
  R* rw;
  if constexpr (IsComplex<T>::value) {
    rw = rwork;
  } else {
    rw = work;
  }
  const Vector<R> RWORK(rw);

  R rowcnd = zero, colcnd = zero, smlnum = zero, bignum = zero;
  bool rowequ, colequ;
  blasint info = 0;
  const bool nofact = Lsame(fact, 'N');
  const bool equil = Lsame(fact, 'E');
  const bool notran = Lsame(trans, 'N');
  if (nofact || equil) {
    *equed = 'N';
    rowequ = false;
    colequ = false;
  } else {
    rowequ = Lsame(*equed, 'R') || Lsame(*equed, 'B');
    colequ = Lsame(*equed, 'C') || Lsame(*equed, 'B');
    smlnum = Lamch<R>('S');
    bignum = one / smlnum;
  }

  // Test the input parameters.
  if (!nofact && !equil && !Lsame(fact, 'F')) {
    info = -1;
  } else if (!notran && !Lsame(trans, 'T') && !Lsame(trans, 'C')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (nrhs < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldaf < Max<blasint>(1, n)) {
    info = -8;
  } else if (Lsame(fact, 'F') && !(rowequ || colequ || Lsame(*equed, 'N'))) {
    info = -10;
  } else {
    if (rowequ) {
      R rcmin = bignum;
      R rcmax = zero;
      for (blasint j = 1; j <= n; ++j) {
        rcmin = Min(rcmin, Rs(j));
        rcmax = Max(rcmax, Rs(j));
      }
      if (rcmin <= zero) {
        info = -11;
      } else if (n > 0) {
        rowcnd = Max(rcmin, smlnum) / Min(rcmax, bignum);
      } else {
        rowcnd = one;
      }
    }
    if (colequ && info == 0) {
      R rcmin = bignum;
      R rcmax = zero;
      for (blasint j = 1; j <= n; ++j) {
        rcmin = Min(rcmin, C(j));
        rcmax = Max(rcmax, C(j));
      }
      if (rcmin <= zero) {
        info = -12;
      } else if (n > 0) {
        colcnd = Max(rcmin, smlnum) / Min(rcmax, bignum);
      } else {
        colcnd = one;
      }
    }
    if (info == 0) {
      if (ldb < Max<blasint>(1, n)) {
        info = -14;
      } else if (ldx < Max<blasint>(1, n)) {
        info = -16;
      }
    }
  }

  if (info != 0) {
    Xerbla<T>("GESVX", -info);
    return info;
  }

  if (equil) {
    // Compute row and column scalings to equilibrate the matrix A.
    R amax;
    const blasint infequ = Geequ(n, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
    if (infequ == 0) {
      // Equilibrate the matrix.
      Laqge(n, n, a, lda, r, c, rowcnd, colcnd, amax, equed);
      rowequ = Lsame(*equed, 'R') || Lsame(*equed, 'B');
      colequ = Lsame(*equed, 'C') || Lsame(*equed, 'B');
    }
  }

  // Scale the right hand side.
  if (notran) {
    if (rowequ) {
      for (blasint j = 1; j <= nrhs; ++j) {
        for (blasint i = 1; i <= n; ++i) B(i, j) = Rs(i) * B(i, j);
      }
    }
  } else if (colequ) {
    for (blasint j = 1; j <= nrhs; ++j) {
      for (blasint i = 1; i <= n; ++i) B(i, j) = C(i) * B(i, j);
    }
  }

  if (nofact || equil) {
    // Compute the LU factorization of A.
    Lacpy('F', n, n, a, lda, af, ldaf);
    info = Getrf(n, n, af, ldaf, ipiv);

    // Return if INFO is non-zero.
    if (info > 0) {
      // Compute the reciprocal pivot growth factor of the leading
      // rank-deficient INFO columns of A.
      R rpvgrw = Lantr('M', 'U', 'N', info, info, af, ldaf, rw);
      if (rpvgrw == zero) {
        rpvgrw = one;
      } else {
        rpvgrw = Lange('M', n, info, a, lda, rw) / rpvgrw;
      }
      RWORK(1) = rpvgrw;
      *rcond = zero;
      return info;
    }
  }

  // Compute the norm of the matrix A and the reciprocal pivot growth factor
  // RPVGRW.
  const char norm = notran ? '1' : 'I';
  const R anorm = Lange(norm, n, n, a, lda, rw);
  R rpvgrw = Lantr('M', 'U', 'N', n, n, af, ldaf, rw);
  if (rpvgrw == zero) {
    rpvgrw = one;
  } else {
    rpvgrw = Lange('M', n, n, a, lda, rw) / rpvgrw;
  }

  // Compute the reciprocal of the condition number of A.
  Gecon(norm, n, af, ldaf, anorm, rcond, work, rwork, iwork);

  // Compute the solution matrix X.
  Lacpy('F', n, nrhs, b, ldb, x, ldx);
  info = Getrs(trans, n, nrhs, af, ldaf, ipiv, x, ldx);

  // Use iterative refinement to improve the computed solution and compute
  // error bounds and backward error estimates for it.
  Gerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr,
        work, rwork, iwork);

  // Transform the solution matrix X to a solution of the original system.
  if (notran) {
    if (colequ) {
      for (blasint j = 1; j <= nrhs; ++j) {
        for (blasint i = 1; i <= n; ++i) X(i, j) = C(i) * X(i, j);
      }
      for (blasint j = 1; j <= nrhs; ++j) FERR(j) = FERR(j) / colcnd;
    }
  } else if (rowequ) {
    for (blasint j = 1; j <= nrhs; ++j) {
      for (blasint i = 1; i <= n; ++i) X(i, j) = Rs(i) * X(i, j);
    }
    for (blasint j = 1; j <= nrhs; ++j) FERR(j) = FERR(j) / rowcnd;
  }

  // Set INFO = N+1 if the matrix is singular to working precision.
  if (*rcond < Lamch<R>('E')) info = n + 1;

  RWORK(1) = rpvgrw;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template blasint Gecon<T>(char, blasint, const T*, blasint, RealT<T>,     \
                            RealT<T>*, T*, RealT<T>*, blasint*);            \
  template blasint Geequ<T>(blasint, blasint, const T*, blasint, RealT<T>*, \
                            RealT<T>*, RealT<T>*, RealT<T>*, RealT<T>*);    \
  template void Laqge<T>(blasint, blasint, T*, blasint, const RealT<T>*,    \
                         const RealT<T>*, RealT<T>, RealT<T>, RealT<T>,     \
                         char*);                                            \
  template blasint Gerfs<T>(char, blasint, blasint, const T*, blasint,      \
                            const T*, blasint, const blasint*, const T*,    \
                            blasint, T*, blasint, RealT<T>*, RealT<T>*, T*, \
                            RealT<T>*, blasint*);                           \
  template blasint Gesvx<T>(char, char, blasint, blasint, T*, blasint, T*,  \
                            blasint, blasint*, char*, RealT<T>*, RealT<T>*, \
                            T*, blasint, T*, blasint, RealT<T>*, RealT<T>*, \
                            RealT<T>*, T*, RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GESVX(prefix, T, R)                                     \
  HWYBLAS_API void prefix##geequ_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  const R* a, const hwyblas_int* lda, R* r,    \
                                  R* c, R* rowcnd, R* colcnd, R* amax,         \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Geequ(*m, *n, As<const T>(a), *lda, r, c, rowcnd, \
                                   colcnd, amax);                              \
  }                                                                            \
  HWYBLAS_API void prefix##laqge_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, R* a,                        \
      const hwyblas_int* lda, const R* r, const R* c, const R* rowcnd,         \
      const R* colcnd, const R* amax, char* equed) {                           \
    hwyblas::lapack::Laqge(*m, *n, As<T>(a), *lda, r, c, *rowcnd, *colcnd,     \
                           *amax, equed);                                      \
  }

HWYBLAS_LAPACK_GESVX(s, float, float)
HWYBLAS_LAPACK_GESVX(d, double, double)
HWYBLAS_LAPACK_GESVX(c, c32, float)
HWYBLAS_LAPACK_GESVX(z, c64, double)

#undef HWYBLAS_LAPACK_GESVX

// xGECON, xGERFS and xGESVX take IWORK (real) or RWORK (complex).
#define HWYBLAS_LAPACK_GESVX_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##gecon_(const char* norm, const hwyblas_int* n,      \
                                  const R* a, const hwyblas_int* lda,          \
                                  const R* anorm, R* rcond, R* work,           \
                                  hwyblas_int* iwork, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Gecon(*norm, *n, a, *lda, *anorm, rcond, work,    \
                                   static_cast<R*>(nullptr), iwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##gerfs_(                                             \
      const char* trans, const hwyblas_int* n, const hwyblas_int* nrhs,        \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const hwyblas_int* ipiv, const R* b,            \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* ferr, R* berr,  \
      R* work, hwyblas_int* iwork, hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Gerfs(*trans, *n, *nrhs, a, *lda, af, *ldaf,      \
                                   ipiv, b, *ldb, x, *ldx, ferr, berr, work,   \
                                   static_cast<R*>(nullptr), iwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##gesvx_(                                             \
      const char* fact, const char* trans, const hwyblas_int* n,               \
      const hwyblas_int* nrhs, R* a, const hwyblas_int* lda, R* af,            \
      const hwyblas_int* ldaf, hwyblas_int* ipiv, char* equed, R* r, R* c,     \
      R* b, const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond,    \
      R* ferr, R* berr, R* work, hwyblas_int* iwork, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Gesvx(*fact, *trans, *n, *nrhs, a, *lda, af,      \
                                   *ldaf, ipiv, equed, r, c, b, *ldb, x, *ldx, \
                                   rcond, ferr, berr, work,                    \
                                   static_cast<R*>(nullptr), iwork);           \
  }

#define HWYBLAS_LAPACK_GESVX_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##gecon_(const char* norm, const hwyblas_int* n,      \
                                  const R* a, const hwyblas_int* lda,          \
                                  const R* anorm, R* rcond, R* work, R* rwork, \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Gecon(*norm, *n, As<const T>(a), *lda, *anorm,    \
                                   rcond, As<T>(work), rwork,                  \
                                   static_cast<hwyblas_int*>(nullptr));        \
  }                                                                            \
  HWYBLAS_API void prefix##gerfs_(                                             \
      const char* trans, const hwyblas_int* n, const hwyblas_int* nrhs,        \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const hwyblas_int* ipiv, const R* b,            \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* ferr, R* berr,  \
      R* work, R* rwork, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Gerfs(                                            \
        *trans, *n, *nrhs, As<const T>(a), *lda, As<const T>(af), *ldaf, ipiv, \
        As<const T>(b), *ldb, As<T>(x), *ldx, ferr, berr, As<T>(work), rwork,  \
        static_cast<hwyblas_int*>(nullptr));                                   \
  }                                                                            \
  HWYBLAS_API void prefix##gesvx_(                                             \
      const char* fact, const char* trans, const hwyblas_int* n,               \
      const hwyblas_int* nrhs, R* a, const hwyblas_int* lda, R* af,            \
      const hwyblas_int* ldaf, hwyblas_int* ipiv, char* equed, R* r, R* c,     \
      R* b, const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond,    \
      R* ferr, R* berr, R* work, R* rwork, hwyblas_int* info) {                \
    *info = hwyblas::lapack::Gesvx(                                            \
        *fact, *trans, *n, *nrhs, As<T>(a), *lda, As<T>(af), *ldaf, ipiv,      \
        equed, r, c, As<T>(b), *ldb, As<T>(x), *ldx, rcond, ferr, berr,        \
        As<T>(work), rwork, static_cast<hwyblas_int*>(nullptr));               \
  }

HWYBLAS_LAPACK_GESVX_REAL(s, float)
HWYBLAS_LAPACK_GESVX_REAL(d, double)
HWYBLAS_LAPACK_GESVX_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GESVX_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GESVX_REAL
#undef HWYBLAS_LAPACK_GESVX_COMPLEX

}  // extern "C"
