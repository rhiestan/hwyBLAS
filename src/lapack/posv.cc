// Symmetric (Hermitian) positive definite linear systems with the Cholesky
// factorization of xPOTRF: the solvers xPOTRS and xPOSV, the inverse
// (xPOTRI with xLAUUM, xLAUU2), the condition number estimate (xPOCON),
// equilibration (xPOEQU, xLAQSY, xLAQHE), iterative refinement with error
// bounds (xPORFS) and the expert driver xPOSVX.
//
// As in gesvx.cc, the real and the complex routines that differ only in
// their workspaces (xPOCON, xPORFS, xPOSVX) share one template, whose views
// are named as in the complex routine and point into the real routine's
// WORK for real T; rwork is used for complex T only, iwork for real T only.
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

// xPOTRS: solves A * X = B for the N-by-NRHS matrix B (overwritten with X),
// with the Cholesky factorization A = U**H * U or A = L * L**H (UPLO) of
// the symmetric (Hermitian) positive definite matrix A computed by xPOTRF.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Potrs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda, T* b,
              blasint ldb) {
  const T one = T(1);

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
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("POTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  if (upper) {
    // Solve A*X = B where A = U**H *U.

    // Solve U**H *X = B, overwriting B with X.
    Trsm('L', 'U', 'C', 'N', n, nrhs, one, a, lda, b, ldb);

    // Solve U*X = B, overwriting B with X.
    Trsm('L', 'U', 'N', 'N', n, nrhs, one, a, lda, b, ldb);
  } else {
    // Solve A*X = B where A = L*L**H.

    // Solve L*X = B, overwriting B with X.
    Trsm('L', 'L', 'N', 'N', n, nrhs, one, a, lda, b, ldb);

    // Solve L**H *X = B, overwriting B with X.
    Trsm('L', 'L', 'C', 'N', n, nrhs, one, a, lda, b, ldb);
  }
  return info;
}

// xPOSV: solves A * X = B for a symmetric (Hermitian) positive definite
// matrix A of order N (triangle UPLO) and the N-by-NRHS matrix B: factors
// A = U**H * U or L * L**H with xPOTRF (A returns the factor) and solves
// with xPOTRS (B returns X).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the leading principal minor of order i of A is not positive
// (the factorization could not be completed and X was not computed).
template <typename T>
blasint Posv(char uplo, blasint n, blasint nrhs, T* a, blasint lda, T* b,
             blasint ldb) {
  // Test the input parameters.
  blasint info = 0;
  if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("POSV", -info);
    return info;
  }

  // Compute the Cholesky factorization A = U**H *U or A = L*L**H.
  info = Potrf(uplo, n, a, lda);
  if (info == 0) {
    // Solve the system A*X = B, overwriting B with X.
    info = Potrs(uplo, n, nrhs, a, lda, b, ldb);
  }
  return info;
}

// xLAUU2: computes the product U * U**H (UPLO = 'U') or L**H * L ('L') of
// the upper or lower triangular matrix of order N in the triangle UPLO of
// A, which it overwrites. This is the unblocked version (Level 2 BLAS).
//
// The real routine computes the new diagonal element as one dot product
// that includes A(I, I)**2, the complex one adds AII*AII to the dot product
// of the rest; the template keeps both.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Lauu2(char uplo, blasint n, T* a, blasint lda) {
  using R = RealT<T>;
  const T one = T(1);
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
    Xerbla<T>("LAUU2", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  if (upper) {
    // Compute the product U * U**H.
    for (blasint i = 1; i <= n; ++i) {
      const R aii = std::real(A(i, i));
      if (i < n) {
        if constexpr (IsComplex<T>::value) {
          A(i, i) = aii * aii + std::real(Dotc(n - i, A.ptr(i, i + 1), lda,
                                               A.ptr(i, i + 1), lda));
        } else {
          A(i, i) = Dotc(n - i + 1, A.ptr(i, i), lda, A.ptr(i, i), lda);
        }
        Lacgv(n - i, A.ptr(i, i + 1), lda);
        Gemv('N', i - 1, n - i, one, A.ptr(1, i + 1), lda, A.ptr(i, i + 1), lda,
             T(aii), A.ptr(1, i), 1);
        Lacgv(n - i, A.ptr(i, i + 1), lda);
      } else {
        Rscal(i, aii, A.ptr(1, i), 1);
      }
    }
  } else {
    // Compute the product L**H * L.
    for (blasint i = 1; i <= n; ++i) {
      const R aii = std::real(A(i, i));
      if (i < n) {
        if constexpr (IsComplex<T>::value) {
          A(i, i) = aii * aii + std::real(Dotc(n - i, A.ptr(i + 1, i), 1,
                                               A.ptr(i + 1, i), 1));
        } else {
          A(i, i) = Dotc(n - i + 1, A.ptr(i, i), 1, A.ptr(i, i), 1);
        }
        Lacgv(i - 1, A.ptr(i, 1), lda);
        Gemv('C', n - i, i - 1, one, A.ptr(i + 1, 1), lda, A.ptr(i + 1, i), 1,
             T(aii), A.ptr(i, 1), lda);
        Lacgv(i - 1, A.ptr(i, 1), lda);
      } else {
        Rscal(i, aii, A.ptr(i, 1), lda);
      }
    }
  }
  return info;
}

// xLAUUM: computes the product U * U**H or L**H * L like xLAUU2 (see
// there), blocked: NB columns at a time (NB from ILAENV), with xLAUU2 for
// the diagonal blocks and Level 3 BLAS for the rest.
template <typename T>
blasint Lauum(char uplo, blasint n, T* a, blasint lda) {
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
    Xerbla<T>("LAUUM", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Determine the block size for this environment.
  const char opts[2] = {uplo, '\0'};
  const blasint nb = Ilaenv<T>(1, "LAUUM", opts, n, -1, -1, -1);

  if (nb <= 1 || nb >= n) {
    // Use unblocked code.
    info = Lauu2(uplo, n, a, lda);
  } else {
    // Use blocked code.
    if (upper) {
      // Compute the product U * U**H.
      for (blasint i = 1; i <= n; i += nb) {
        const blasint ib = Min(nb, n - i + 1);
        Trmm('R', 'U', 'C', 'N', i - 1, ib, cone, A.ptr(i, i), lda, A.ptr(1, i),
             lda);
        info = Lauu2('U', ib, A.ptr(i, i), lda);
        if (i + ib <= n) {
          Gemm('N', 'C', i - 1, ib, n - i - ib + 1, cone, A.ptr(1, i + ib), lda,
               A.ptr(i, i + ib), lda, cone, A.ptr(1, i), lda);
          Herk('U', 'N', ib, n - i - ib + 1, one, A.ptr(i, i + ib), lda, one,
               A.ptr(i, i), lda);
        }
      }
    } else {
      // Compute the product L**H * L.
      for (blasint i = 1; i <= n; i += nb) {
        const blasint ib = Min(nb, n - i + 1);
        Trmm('L', 'L', 'C', 'N', ib, i - 1, cone, A.ptr(i, i), lda, A.ptr(i, 1),
             lda);
        info = Lauu2('L', ib, A.ptr(i, i), lda);
        if (i + ib <= n) {
          Gemm('C', 'N', ib, i - 1, n - i - ib + 1, cone, A.ptr(i + ib, i), lda,
               A.ptr(i + ib, 1), lda, cone, A.ptr(i, 1), lda);
          Herk('L', 'C', ib, n - i - ib + 1, one, A.ptr(i + ib, i), lda, one,
               A.ptr(i, i), lda);
        }
      }
    }
  }
  return info;
}

// xPOTRI: computes the inverse of the symmetric (Hermitian) positive
// definite matrix A of order N from its Cholesky factorization by xPOTRF
// in the triangle UPLO of A: inverts the factor (xTRTRI) and forms
// inv(U) * inv(U)**H or inv(L)**H * inv(L) (xLAUUM), the triangle UPLO of
// inv(A), in place.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the factor's element (i, i) is zero (A is singular).
template <typename T>
blasint Potri(char uplo, blasint n, T* a, blasint lda) {
  // Test the input parameters.
  blasint info = 0;
  if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("POTRI", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Invert the triangular Cholesky factor U or L.
  info = Trtri(uplo, 'N', n, a, lda);
  if (info > 0) return info;

  // Form inv(U) * inv(U)**H or inv(L)**H * inv(L).
  info = Lauum(uplo, n, a, lda);
  return info;
}

// xPOCON: estimates the reciprocal of the condition number (1-norm) of the
// symmetric (Hermitian) positive definite matrix A of order N from its
// Cholesky factorization by xPOTRF (A, triangle UPLO) and the 1-norm ANORM
// of the original matrix: RCOND = 1 / (ANORM * norm(inv(A))), with
// norm(inv(A)) estimated by xLACN2.
//
// The real routine uses WORK (3*N elements; the column norms of xLATRS in
// WORK(2*N+1:3*N)) and IWORK (N), the complex one WORK (2*N) and RWORK
// (N; the column norms).
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Pocon(char uplo, blasint n, const T* a, blasint lda, RealT<T> anorm,
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
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("POCON", -info);
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

  const R smlnum = Lamch<R>('S');

  // Estimate the 1-norm of inv(A).
  R ainvnm = zero;
  char normin = 'N';
  blasint kase = 0;
  for (;;) {
    if constexpr (IsComplex<T>::value) {
      Lacn2(n, WORK.ptr(n + 1), work, &ainvnm, &kase, isave);
    } else {
      Lacn2(n, WORK.ptr(n + 1), work, iwork, &ainvnm, &kase, isave);
    }
    if (kase == 0) break;
    R scalel, scaleu;
    if (upper) {
      // Multiply by inv(U**H).
      Latrs('U', ConjTrans<T>(), 'N', normin, n, a, lda, work, &scalel, rw);
      normin = 'Y';

      // Multiply by inv(U).
      Latrs('U', 'N', 'N', normin, n, a, lda, work, &scaleu, rw);
    } else {
      // Multiply by inv(L).
      Latrs('L', 'N', 'N', normin, n, a, lda, work, &scalel, rw);
      normin = 'Y';

      // Multiply by inv(L**H).
      Latrs('L', ConjTrans<T>(), 'N', normin, n, a, lda, work, &scaleu, rw);
    }

    // Multiply by 1/SCALE if doing so will not cause overflow.
    const R scale = scalel * scaleu;
    if (scale != one) {
      const blasint ix = Iamax(n, work, 1);
      if (scale < Cabs1(WORK(ix)) * smlnum || scale == zero) return info;
      Rscl(n, scale, work, 1);
    }
  }

  // Compute the estimate of the reciprocal condition number.
  if (ainvnm != zero) *rcond = (one / ainvnm) / anorm;
  return info;
}

// xPOEQU: computes scalings S(i) = 1 / sqrt(A(i, i)) (N elements) that
// equilibrate the symmetric (Hermitian) positive definite matrix A of
// order N: diag(S) * A * diag(S) has ones on the diagonal. SCOND =
// min(S(i)) / max(S(i)) tells whether scaling is worthwhile (not if
// >= 0.1); AMAX is the largest diagonal element of A. Only the diagonal of
// A is referenced (its real parts, for complex T).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if A(i, i) is the first non-positive diagonal element.
template <typename T>
blasint Poequ(blasint n, const T* a, blasint lda, RealT<T>* s, RealT<T>* scond,
              RealT<T>* amax) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const Matrix<const T> A(a, lda);
  const Vector<R> S(s);

  // Test the input parameters.
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (lda < Max<blasint>(1, n)) {
    info = -3;
  }
  if (info != 0) {
    Xerbla<T>("POEQU", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) {
    *scond = one;
    *amax = zero;
    return info;
  }

  // Find the minimum and maximum diagonal elements.
  S(1) = std::real(A(1, 1));
  R smin = S(1);
  *amax = S(1);
  for (blasint i = 2; i <= n; ++i) {
    S(i) = std::real(A(i, i));
    smin = Min(smin, S(i));
    *amax = Max(*amax, S(i));
  }

  if (smin <= zero) {
    // Find the first non-positive diagonal element and return.
    for (blasint i = 1; i <= n; ++i) {
      if (S(i) <= zero) {
        info = i;
        return info;
      }
    }
  } else {
    // Set the scale factors to the reciprocals of the diagonal elements.
    for (blasint i = 1; i <= n; ++i) S(i) = one / std::sqrt(S(i));

    // Compute SCOND = min(S(I)) / max(S(I)).
    *scond = std::sqrt(smin) / std::sqrt(*amax);
  }
  return info;
}

// xLAQSY: equilibrates the symmetric matrix A of order N (triangle UPLO)
// with the scaling S (from xPOEQU or xSYEQUB), A := diag(S) * A * diag(S),
// if worthwhile: if SCOND < 0.1 or AMAX is close to underflow or overflow.
// EQUED returns 'Y' if A was scaled, 'N' if not. (Complex T: complex
// symmetric A, CLAQSY / ZLAQSY.)
template <typename T>
void Laqsy(char uplo, blasint n, T* a, blasint lda, const RealT<T>* s,
           RealT<T> scond, RealT<T> amax, char* equed) {
  using R = RealT<T>;
  const R one = R(1), thresh = R(0.1);
  const Matrix<T> A(a, lda);
  const Vector<const R> S(s);

  // Quick return if possible.
  if (n <= 0) {
    *equed = 'N';
    return;
  }

  // Initialize LARGE and SMALL.
  const R small = Lamch<R>('S') / Lamch<R>('P');
  const R large = one / small;

  if (scond >= thresh && amax >= small && amax <= large) {
    // No equilibration.
    *equed = 'N';
  } else {
    // Replace A by diag(S) * A * diag(S).
    if (Lsame(uplo, 'U')) {
      // Upper triangle of A is stored.
      for (blasint j = 1; j <= n; ++j) {
        const R cj = S(j);
        for (blasint i = 1; i <= j; ++i) A(i, j) = cj * S(i) * A(i, j);
      }
    } else {
      // Lower triangle of A is stored.
      for (blasint j = 1; j <= n; ++j) {
        const R cj = S(j);
        for (blasint i = j; i <= n; ++i) A(i, j) = cj * S(i) * A(i, j);
      }
    }
    *equed = 'Y';
  }
}

// xLAQHE (complex T only): equilibrates the Hermitian matrix A like xLAQSY
// (see there); the diagonal elements become real.
template <typename T>
void Laqhe(char uplo, blasint n, T* a, blasint lda, const RealT<T>* s,
           RealT<T> scond, RealT<T> amax, char* equed) {
  using R = RealT<T>;
  const R one = R(1), thresh = R(0.1);
  const Matrix<T> A(a, lda);
  const Vector<const R> S(s);

  // Quick return if possible.
  if (n <= 0) {
    *equed = 'N';
    return;
  }

  // Initialize LARGE and SMALL.
  const R small = Lamch<R>('S') / Lamch<R>('P');
  const R large = one / small;

  if (scond >= thresh && amax >= small && amax <= large) {
    // No equilibration.
    *equed = 'N';
  } else {
    // Replace A by diag(S) * A * diag(S).
    if (Lsame(uplo, 'U')) {
      // Upper triangle of A is stored.
      for (blasint j = 1; j <= n; ++j) {
        const R cj = S(j);
        for (blasint i = 1; i <= j - 1; ++i) A(i, j) = cj * S(i) * A(i, j);
        A(j, j) = cj * cj * std::real(A(j, j));
      }
    } else {
      // Lower triangle of A is stored.
      for (blasint j = 1; j <= n; ++j) {
        const R cj = S(j);
        A(j, j) = cj * cj * std::real(A(j, j));
        for (blasint i = j + 1; i <= n; ++i) A(i, j) = cj * S(i) * A(i, j);
      }
    }
    *equed = 'Y';
  }
}

// xPORFS: improves the computed solution X (N-by-NRHS) of A * X = B for the
// symmetric (Hermitian) positive definite A (triangle UPLO) by iterative
// refinement, with the Cholesky factorization AF by xPOTRF, and computes
// error bounds: FERR(j) bounds the relative error of column j of X, BERR(j)
// is its componentwise relative backward error.
//
// Workspaces as in xGERFS: the real routine keeps W = abs(A)*abs(X) +
// abs(B) in WORK(1:N), the residual in WORK(N+1:2*N) and xLACN2's V in
// WORK(2*N+1:3*N) (with IWORK), the complex one W in RWORK and the
// residual and V in WORK(1:2*N). The views below have the complex
// routine's names.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Porfs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const T* b, blasint ldb, T* x,
              blasint ldx, RealT<T>* ferr, RealT<T>* berr, T* work,
              RealT<T>* rwork, blasint* iwork) {
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
    info = -9;
  } else if (ldx < Max<blasint>(1, n)) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>("PORFS", -info);
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
      Hemv(uplo, n, -one, a, lda, X.ptr(1, j), 1, one, wk, 1);

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
          RWORK(k) = RWORK(k) + std::abs(std::real(A(k, k))) * xk + s;
        }
      } else {
        for (blasint k = 1; k <= n; ++k) {
          s = zero;
          const R xk = Cabs1(X(k, j));
          RWORK(k) = RWORK(k) + std::abs(std::real(A(k, k))) * xk;
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
        Potrs(uplo, n, 1, af, ldaf, wk, n);
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
        // Multiply by diag(W)*inv(A**H).
        Potrs(uplo, n, 1, af, ldaf, wk, n);
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
      } else if (kase == 2) {
        // Multiply by inv(A)*diag(W).
        for (blasint i = 1; i <= n; ++i) WORK(i) = RWORK(i) * WORK(i);
        Potrs(uplo, n, 1, af, ldaf, wk, n);
      }
    }

    // Normalize error.
    lstres = zero;
    for (blasint i = 1; i <= n; ++i) lstres = Max(lstres, Cabs1(X(i, j)));
    if (lstres != zero) FERR(j) = FERR(j) / lstres;
  }
  return info;
}

// xPOSVX: solves A * X = B for the symmetric (Hermitian) positive definite
// matrix A of order N (triangle UPLO) and N-by-NRHS matrices X and B with
// the Cholesky factorization of A, and returns error bounds and an estimate
// of the condition number. FACT selects whether the factorization AF is
// given ('F'; with the equilibration EQUED, S that it is of), computed
// ('N'), or computed after equilibrating A (and B) if worthwhile ('E'; A
// then returns the equilibrated matrix, EQUED what was done: 'N' or 'Y').
// X is refined with xPORFS and returned for the original system. RCOND
// estimates the reciprocal condition number of the (equilibrated) A.
//
// The real routine uses WORK (3*N elements) and IWORK (N), the complex one
// WORK (2*N) and RWORK (N). The real routine passes WORK to xLANSY, the
// complex one RWORK to xLANHE; the view RWORK is the real routine's WORK
// for real T.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, i <= N
// if the leading principal minor of order i of A is not positive (the
// factorization could not be completed, and the solution and error bounds
// were not computed; RCOND = 0), or N + 1 if RCOND is less than the unit
// roundoff (the matrix is singular to working precision; the solution and
// error bounds have been computed).
template <typename T>
blasint Posvx(char fact, char uplo, blasint n, blasint nrhs, T* a, blasint lda,
              T* af, blasint ldaf, char* equed, RealT<T>* s, T* b, blasint ldb,
              T* x, blasint ldx, RealT<T>* rcond, RealT<T>* ferr,
              RealT<T>* berr, T* work, RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const Matrix<T> B(b, ldb);
  const Vector<R> FERR(ferr);
  const Vector<R> S(s);
  const Matrix<T> X(x, ldx);
  R* rw;
  if constexpr (IsComplex<T>::value) {
    rw = rwork;
  } else {
    rw = work;
  }

  R scond = zero, smlnum = zero, bignum = zero;
  bool rcequ;
  blasint info = 0;
  const bool nofact = Lsame(fact, 'N');
  const bool equil = Lsame(fact, 'E');
  if (nofact || equil) {
    *equed = 'N';
    rcequ = false;
  } else {
    rcequ = Lsame(*equed, 'Y');
    smlnum = Lamch<R>('S');
    bignum = one / smlnum;
  }

  // Test the input parameters.
  if (!nofact && !equil && !Lsame(fact, 'F')) {
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
  } else if (Lsame(fact, 'F') && !(rcequ || Lsame(*equed, 'N'))) {
    info = -9;
  } else {
    if (rcequ) {
      R smin = bignum;
      R smax = zero;
      for (blasint j = 1; j <= n; ++j) {
        smin = Min(smin, S(j));
        smax = Max(smax, S(j));
      }
      if (smin <= zero) {
        info = -10;
      } else if (n > 0) {
        scond = Max(smin, smlnum) / Min(smax, bignum);
      } else {
        scond = one;
      }
    }
    if (info == 0) {
      if (ldb < Max<blasint>(1, n)) {
        info = -12;
      } else if (ldx < Max<blasint>(1, n)) {
        info = -14;
      }
    }
  }

  if (info != 0) {
    Xerbla<T>("POSVX", -info);
    return info;
  }

  if (equil) {
    // Compute row and column scalings to equilibrate the matrix A.
    R amax;
    const blasint infequ = Poequ(n, a, lda, s, &scond, &amax);
    if (infequ == 0) {
      // Equilibrate the matrix.
      if constexpr (IsComplex<T>::value) {
        Laqhe(uplo, n, a, lda, s, scond, amax, equed);
      } else {
        Laqsy(uplo, n, a, lda, s, scond, amax, equed);
      }
      rcequ = Lsame(*equed, 'Y');
    }
  }

  // Scale the right hand side.
  if (rcequ) {
    for (blasint j = 1; j <= nrhs; ++j) {
      for (blasint i = 1; i <= n; ++i) B(i, j) = S(i) * B(i, j);
    }
  }

  if (nofact || equil) {
    // Compute the Cholesky factorization A = U**H *U or A = L*L**H.
    Lacpy(uplo, n, n, a, lda, af, ldaf);
    info = Potrf(uplo, n, af, ldaf);

    // Return if INFO is non-zero.
    if (info > 0) {
      *rcond = zero;
      return info;
    }
  }

  // Compute the norm of the matrix A.
  R anorm;
  if constexpr (IsComplex<T>::value) {
    anorm = Lanhe('1', uplo, n, a, lda, rw);
  } else {
    anorm = Lansy('1', uplo, n, a, lda, rw);
  }

  // Compute the reciprocal of the condition number of A.
  Pocon(uplo, n, af, ldaf, anorm, rcond, work, rwork, iwork);

  // Compute the solution matrix X.
  Lacpy('F', n, nrhs, b, ldb, x, ldx);
  info = Potrs(uplo, n, nrhs, af, ldaf, x, ldx);

  // Use iterative refinement to improve the computed solution and compute
  // error bounds and backward error estimates for it.
  Porfs(uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x, ldx, ferr, berr, work,
        rwork, iwork);

  // Transform the solution matrix X to a solution of the original system.
  if (rcequ) {
    for (blasint j = 1; j <= nrhs; ++j) {
      for (blasint i = 1; i <= n; ++i) X(i, j) = S(i) * X(i, j);
    }
    for (blasint j = 1; j <= nrhs; ++j) FERR(j) = FERR(j) / scond;
  }

  // Set INFO = N+1 if the matrix is singular to working precision.
  if (*rcond < Lamch<R>('E')) info = n + 1;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template blasint Potrs<T>(char, blasint, blasint, const T*, blasint, T*,     \
                            blasint);                                          \
  template blasint Posv<T>(char, blasint, blasint, T*, blasint, T*, blasint);  \
  template blasint Lauu2<T>(char, blasint, T*, blasint);                       \
  template blasint Lauum<T>(char, blasint, T*, blasint);                       \
  template blasint Potri<T>(char, blasint, T*, blasint);                       \
  template blasint Pocon<T>(char, blasint, const T*, blasint, RealT<T>,        \
                            RealT<T>*, T*, RealT<T>*, blasint*);               \
  template blasint Poequ<T>(blasint, const T*, blasint, RealT<T>*, RealT<T>*,  \
                            RealT<T>*);                                        \
  template void Laqsy<T>(char, blasint, T*, blasint, const RealT<T>*,          \
                         RealT<T>, RealT<T>, char*);                           \
  template blasint Porfs<T>(char, blasint, blasint, const T*, blasint,         \
                            const T*, blasint, const T*, blasint, T*, blasint, \
                            RealT<T>*, RealT<T>*, T*, RealT<T>*, blasint*);    \
  template blasint Posvx<T>(char, char, blasint, blasint, T*, blasint, T*,     \
                            blasint, char*, RealT<T>*, T*, blasint, T*,        \
                            blasint, RealT<T>*, RealT<T>*, RealT<T>*, T*,      \
                            RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

template void Laqhe<c32>(char, blasint, c32*, blasint, const float*, float,
                         float, char*);
template void Laqhe<c64>(char, blasint, c64*, blasint, const double*, double,
                         double, char*);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_POSV(prefix, T, R)                                      \
  HWYBLAS_API void prefix##potrs_(const char* uplo, const hwyblas_int* n,      \
                                  const hwyblas_int* nrhs, const R* a,         \
                                  const hwyblas_int* lda, R* b,                \
                                  const hwyblas_int* ldb, hwyblas_int* info) { \
    *info = hwyblas::lapack::Potrs(*uplo, *n, *nrhs, As<const T>(a), *lda,     \
                                   As<T>(b), *ldb);                            \
  }                                                                            \
  HWYBLAS_API void prefix##posv_(const char* uplo, const hwyblas_int* n,       \
                                 const hwyblas_int* nrhs, R* a,                \
                                 const hwyblas_int* lda, R* b,                 \
                                 const hwyblas_int* ldb, hwyblas_int* info) {  \
    *info = hwyblas::lapack::Posv(*uplo, *n, *nrhs, As<T>(a), *lda, As<T>(b),  \
                                  *ldb);                                       \
  }                                                                            \
  HWYBLAS_API void prefix##lauu2_(const char* uplo, const hwyblas_int* n,      \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Lauu2(*uplo, *n, As<T>(a), *lda);                 \
  }                                                                            \
  HWYBLAS_API void prefix##lauum_(const char* uplo, const hwyblas_int* n,      \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Lauum(*uplo, *n, As<T>(a), *lda);                 \
  }                                                                            \
  HWYBLAS_API void prefix##potri_(const char* uplo, const hwyblas_int* n,      \
                                  R* a, const hwyblas_int* lda,                \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Potri(*uplo, *n, As<T>(a), *lda);                 \
  }                                                                            \
  HWYBLAS_API void prefix##poequ_(const hwyblas_int* n, const R* a,            \
                                  const hwyblas_int* lda, R* s, R* scond,      \
                                  R* amax, hwyblas_int* info) {                \
    *info = hwyblas::lapack::Poequ(*n, As<const T>(a), *lda, s, scond, amax);  \
  }                                                                            \
  HWYBLAS_API void prefix##laqsy_(                                             \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,    \
      const R* s, const R* scond, const R* amax, char* equed) {                \
    hwyblas::lapack::Laqsy(*uplo, *n, As<T>(a), *lda, s, *scond, *amax,        \
                           equed);                                             \
  }

HWYBLAS_LAPACK_POSV(s, float, float)
HWYBLAS_LAPACK_POSV(d, double, double)
HWYBLAS_LAPACK_POSV(c, c32, float)
HWYBLAS_LAPACK_POSV(z, c64, double)

#undef HWYBLAS_LAPACK_POSV

// xPOCON, xPORFS and xPOSVX take IWORK (real) or RWORK (complex).
#define HWYBLAS_LAPACK_POSVX_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##pocon_(const char* uplo, const hwyblas_int* n,      \
                                  const R* a, const hwyblas_int* lda,          \
                                  const R* anorm, R* rcond, R* work,           \
                                  hwyblas_int* iwork, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Pocon(*uplo, *n, a, *lda, *anorm, rcond, work,    \
                                   static_cast<R*>(nullptr), iwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##porfs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const R* b, const hwyblas_int* ldb, R* x,       \
      const hwyblas_int* ldx, R* ferr, R* berr, R* work, hwyblas_int* iwork,   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Porfs(*uplo, *n, *nrhs, a, *lda, af, *ldaf, b,    \
                                   *ldb, x, *ldx, ferr, berr, work,            \
                                   static_cast<R*>(nullptr), iwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##posvx_(                                             \
      const char* fact, const char* uplo, const hwyblas_int* n,                \
      const hwyblas_int* nrhs, R* a, const hwyblas_int* lda, R* af,            \
      const hwyblas_int* ldaf, char* equed, R* s, R* b,                        \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond, R* ferr, \
      R* berr, R* work, hwyblas_int* iwork, hwyblas_int* info) {               \
    *info = hwyblas::lapack::Posvx(                                            \
        *fact, *uplo, *n, *nrhs, a, *lda, af, *ldaf, equed, s, b, *ldb, x,     \
        *ldx, rcond, ferr, berr, work, static_cast<R*>(nullptr), iwork);       \
  }

#define HWYBLAS_LAPACK_POSVX_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##pocon_(const char* uplo, const hwyblas_int* n,      \
                                  const R* a, const hwyblas_int* lda,          \
                                  const R* anorm, R* rcond, R* work, R* rwork, \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Pocon(*uplo, *n, As<const T>(a), *lda, *anorm,    \
                                   rcond, As<T>(work), rwork,                  \
                                   static_cast<hwyblas_int*>(nullptr));        \
  }                                                                            \
  HWYBLAS_API void prefix##porfs_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nrhs,         \
      const R* a, const hwyblas_int* lda, const R* af,                         \
      const hwyblas_int* ldaf, const R* b, const hwyblas_int* ldb, R* x,       \
      const hwyblas_int* ldx, R* ferr, R* berr, R* work, R* rwork,             \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Porfs(                                            \
        *uplo, *n, *nrhs, As<const T>(a), *lda, As<const T>(af), *ldaf,        \
        As<const T>(b), *ldb, As<T>(x), *ldx, ferr, berr, As<T>(work), rwork,  \
        static_cast<hwyblas_int*>(nullptr));                                   \
  }                                                                            \
  HWYBLAS_API void prefix##posvx_(                                             \
      const char* fact, const char* uplo, const hwyblas_int* n,                \
      const hwyblas_int* nrhs, R* a, const hwyblas_int* lda, R* af,            \
      const hwyblas_int* ldaf, char* equed, R* s, R* b,                        \
      const hwyblas_int* ldb, R* x, const hwyblas_int* ldx, R* rcond, R* ferr, \
      R* berr, R* work, R* rwork, hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Posvx(                                            \
        *fact, *uplo, *n, *nrhs, As<T>(a), *lda, As<T>(af), *ldaf, equed, s,   \
        As<T>(b), *ldb, As<T>(x), *ldx, rcond, ferr, berr, As<T>(work), rwork, \
        static_cast<hwyblas_int*>(nullptr));                                   \
  }                                                                            \
  HWYBLAS_API void prefix##laqhe_(                                             \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,    \
      const R* s, const R* scond, const R* amax, char* equed) {                \
    hwyblas::lapack::Laqhe(*uplo, *n, As<T>(a), *lda, s, *scond, *amax,        \
                           equed);                                             \
  }

HWYBLAS_LAPACK_POSVX_REAL(s, float)
HWYBLAS_LAPACK_POSVX_REAL(d, double)
HWYBLAS_LAPACK_POSVX_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_POSVX_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_POSVX_REAL
#undef HWYBLAS_LAPACK_POSVX_COMPLEX

}  // extern "C"
