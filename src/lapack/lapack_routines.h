// The ported LAPACK routines, as templates over the four precisions (float,
// double, c32, c64), for the routines that call each other and for LAPACKE.
// Each one is defined (and explicitly instantiated for the four types) in
// the .cc file of its family; the comment there documents it.
//
// Routines return INFO, and take pointers where Fortran passes arrays and
// values where it passes scalars that it does not change. A real routine and
// its complex counterpart share one template, named after the complex
// routine where the names differ (e.g. Ungqr is xORGQR for real T and xUNGQR
// for complex T).
//
// Derived from LAPACK, Copyright (c) 1992-2023 The University of Tennessee
// and The University of Tennessee Research Foundation, The University of
// California Berkeley, and The University of Colorado Denver. Modified BSD
// license, see LICENSE-LAPACK.
#ifndef HWYBLAS_SRC_LAPACK_LAPACK_ROUTINES_H_
#define HWYBLAS_SRC_LAPACK_LAPACK_ROUTINES_H_

#include "src/common.h"

namespace hwyblas {
namespace lapack {

// Calls `macro(T)` for the four precisions, e.g. for explicit
// instantiations.
#define HWYBLAS_LAPACK_FOR_EACH_TYPE(macro) \
  macro(float) macro(double) macro(c32) macro(c64)

// ---------------------------------------------------------------------------
// Auxiliary routines (auxiliary.cc)

// xLACGV: conjugates the vector x (nothing for real T).
template <typename T>
void Lacgv(blasint n, T* x, blasint incx);

// xLACPY: copies all or the upper ('U') or lower ('L') triangle of A to B.
template <typename T>
void Lacpy(char uplo, blasint m, blasint n, const T* a, blasint lda, T* b,
           blasint ldb);

// xLAPY2: sqrt(x**2 + y**2), avoiding unnecessary overflow and underflow.
template <typename R>
R Lapy2(R x, R y);

// xLAPY3: sqrt(x**2 + y**2 + z**2), avoiding unnecessary overflow and
// underflow.
template <typename R>
R Lapy3(R x, R y, R z);

// xLADIV: p + i*q = (a + i*b) / (c + i*d), robustly.
template <typename R>
void Ladiv(R a, R b, R c, R d, R* p, R* q);

// CLADIV / ZLADIV: x / y, robustly.
template <typename T>
T Ladiv(T x, T y);

// xLASET: off-diagonal elements alpha and diagonal beta.
template <typename T>
void Laset(char uplo, blasint m, blasint n, T alpha, T beta, T* a, blasint lda);

// xLASCL: A := A * (cto / cfrom), safely.
template <typename T>
blasint Lascl(char type, blasint kl, blasint ku, RealT<T> cfrom, RealT<T> cto,
              blasint m, blasint n, T* a, blasint lda);

// xRSCL (real T), CSRSCL / ZDRSCL (complex T): x := x / sa, safely.
template <typename T>
void Rscl(blasint n, RealT<T> sa, T* sx, blasint incx);

// ILAxLC: the last non-zero column of A, or 0 if A is zero.
template <typename T>
blasint Ilalc(blasint m, blasint n, const T* a, blasint lda);

// ILAxLR: the last non-zero row of A, or 0 if A is zero.
template <typename T>
blasint Ilalr(blasint m, blasint n, const T* a, blasint lda);

// ---------------------------------------------------------------------------
// Norms (norms.cc)

template <typename T>
void Lassq(blasint n, const T* x, blasint incx, RealT<T>* scale,
           RealT<T>* sumsq);

template <typename T>
RealT<T> Lange(char norm, blasint m, blasint n, const T* a, blasint lda,
               RealT<T>* work);

template <typename T>
RealT<T> Lansy(char norm, char uplo, blasint n, const T* a, blasint lda,
               RealT<T>* work);

// Complex T only.
template <typename T>
RealT<T> Lanhe(char norm, char uplo, blasint n, const T* a, blasint lda,
               RealT<T>* work);

template <typename T>
RealT<T> Lantr(char norm, char uplo, char diag, blasint m, blasint n,
               const T* a, blasint lda, RealT<T>* work);

// ---------------------------------------------------------------------------
// Householder reflectors (householder.cc)

template <typename T>
void Larfg(blasint n, T* alpha, T* x, blasint incx, T* tau);

template <typename T>
void Larf(char side, blasint m, blasint n, const T* v, blasint incv, T tau,
          T* c, blasint ldc, T* work);

template <typename T>
void Larf1f(char side, blasint m, blasint n, const T* v, blasint incv, T tau,
            T* c, blasint ldc, T* work);

template <typename T>
void Larf1l(char side, blasint m, blasint n, const T* v, blasint incv, T tau,
            T* c, blasint ldc, T* work);

template <typename T>
void Larft(char direct, char storev, blasint n, blasint k, const T* v,
           blasint ldv, const T* tau, T* t, blasint ldt);

template <typename T>
void Larfb(char side, char trans, char direct, char storev, blasint m,
           blasint n, blasint k, const T* v, blasint ldv, const T* t,
           blasint ldt, T* c, blasint ldc, T* work, blasint ldwork);

// ---------------------------------------------------------------------------
// LU factorization (getrf.cc)

template <typename T>
void Laswp(blasint n, T* a, blasint lda, blasint k1, blasint k2,
           const blasint* ipiv, blasint incx);

template <typename T>
blasint Getrf2(blasint m, blasint n, T* a, blasint lda, blasint* ipiv);

template <typename T>
blasint Getrf(blasint m, blasint n, T* a, blasint lda, blasint* ipiv);

template <typename T>
blasint Getrs(char trans, blasint n, blasint nrhs, const T* a, blasint lda,
              const blasint* ipiv, T* b, blasint ldb);

template <typename T>
blasint Gesv(blasint n, blasint nrhs, T* a, blasint lda, blasint* ipiv, T* b,
             blasint ldb);

template <typename T>
blasint Getri(blasint n, T* a, blasint lda, const blasint* ipiv, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// Triangular matrices (triangular.cc)

template <typename T>
blasint Trti2(char uplo, char diag, blasint n, T* a, blasint lda);

template <typename T>
blasint Trtri(char uplo, char diag, blasint n, T* a, blasint lda);

template <typename T>
blasint Trtrs(char uplo, char trans, char diag, blasint n, blasint nrhs,
              const T* a, blasint lda, T* b, blasint ldb);

template <typename T>
blasint Latrs(char uplo, char trans, char diag, char normin, blasint n,
              const T* a, blasint lda, T* x, RealT<T>* scale, RealT<T>* cnorm);

// The workspaces of the real and the complex routine: WORK (3 * n
// elements) and IWORK (n) for real T, WORK (2 * n) and RWORK (n) for
// complex T; rwork is used for complex T only, iwork for real T only.
template <typename T>
blasint Trcon(char norm, char uplo, char diag, blasint n, const T* a,
              blasint lda, RealT<T>* rcond, T* work, RealT<T>* rwork,
              blasint* iwork);

// Workspaces as for Trcon.
template <typename T>
blasint Trrfs(char uplo, char trans, char diag, blasint n, blasint nrhs,
              const T* a, blasint lda, const T* b, blasint ldb, const T* x,
              blasint ldx, RealT<T>* ferr, RealT<T>* berr, T* work,
              RealT<T>* rwork, blasint* iwork);

// ---------------------------------------------------------------------------
// Expert driver of general linear systems: condition number, equilibration,
// iterative refinement (gesvx.cc)

// WORK (4 * n elements) and IWORK (n) for real T, WORK (2 * n) and RWORK
// (2 * n) for complex T; rwork is used for complex T only, iwork for real T
// only.
template <typename T>
blasint Gecon(char norm, blasint n, const T* a, blasint lda, RealT<T> anorm,
              RealT<T>* rcond, T* work, RealT<T>* rwork, blasint* iwork);

template <typename T>
blasint Geequ(blasint m, blasint n, const T* a, blasint lda, RealT<T>* r,
              RealT<T>* c, RealT<T>* rowcnd, RealT<T>* colcnd, RealT<T>* amax);

template <typename T>
void Laqge(blasint m, blasint n, T* a, blasint lda, const RealT<T>* r,
           const RealT<T>* c, RealT<T> rowcnd, RealT<T> colcnd, RealT<T> amax,
           char* equed);

// WORK (3 * n elements) and IWORK (n) for real T, WORK (2 * n) and RWORK
// (n) for complex T; rwork is used for complex T only, iwork for real T
// only.
template <typename T>
blasint Gerfs(char trans, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* ferr, RealT<T>* berr,
              T* work, RealT<T>* rwork, blasint* iwork);

// Workspaces as for Gecon. The reciprocal pivot growth factor is returned
// in WORK(1) for real T and RWORK(1) for complex T.
template <typename T>
blasint Gesvx(char fact, char trans, blasint n, blasint nrhs, T* a, blasint lda,
              T* af, blasint ldaf, blasint* ipiv, char* equed, RealT<T>* r,
              RealT<T>* c, T* b, blasint ldb, T* x, blasint ldx,
              RealT<T>* rcond, RealT<T>* ferr, RealT<T>* berr, T* work,
              RealT<T>* rwork, blasint* iwork);

// ---------------------------------------------------------------------------
// Cholesky factorization (potrf.cc)

template <typename T>
blasint Potrf2(char uplo, blasint n, T* a, blasint lda);

template <typename T>
blasint Potrf(char uplo, blasint n, T* a, blasint lda);

// WORK has 2 * n elements.
template <typename T>
blasint Pstf2(char uplo, blasint n, T* a, blasint lda, blasint* piv,
              blasint* rank, RealT<T> tol, RealT<T>* work);

template <typename T>
blasint Pstrf(char uplo, blasint n, T* a, blasint lda, blasint* piv,
              blasint* rank, RealT<T> tol, RealT<T>* work);

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) positive definite systems (posv.cc)

template <typename T>
blasint Potrs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda, T* b,
              blasint ldb);

template <typename T>
blasint Posv(char uplo, blasint n, blasint nrhs, T* a, blasint lda, T* b,
             blasint ldb);

template <typename T>
blasint Lauu2(char uplo, blasint n, T* a, blasint lda);

template <typename T>
blasint Lauum(char uplo, blasint n, T* a, blasint lda);

template <typename T>
blasint Potri(char uplo, blasint n, T* a, blasint lda);

// WORK (3 * n elements) and IWORK (n) for real T, WORK (2 * n) and RWORK
// (n) for complex T; rwork is used for complex T only, iwork for real T
// only.
template <typename T>
blasint Pocon(char uplo, blasint n, const T* a, blasint lda, RealT<T> anorm,
              RealT<T>* rcond, T* work, RealT<T>* rwork, blasint* iwork);

template <typename T>
blasint Poequ(blasint n, const T* a, blasint lda, RealT<T>* s, RealT<T>* scond,
              RealT<T>* amax);

// xLAQSY: symmetric A (complex symmetric for complex T).
template <typename T>
void Laqsy(char uplo, blasint n, T* a, blasint lda, const RealT<T>* s,
           RealT<T> scond, RealT<T> amax, char* equed);

// Complex T only.
template <typename T>
void Laqhe(char uplo, blasint n, T* a, blasint lda, const RealT<T>* s,
           RealT<T> scond, RealT<T> amax, char* equed);

// Workspaces as for Pocon.
template <typename T>
blasint Porfs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const T* b, blasint ldb, T* x,
              blasint ldx, RealT<T>* ferr, RealT<T>* berr, T* work,
              RealT<T>* rwork, blasint* iwork);

// Workspaces as for Pocon.
template <typename T>
blasint Posvx(char fact, char uplo, blasint n, blasint nrhs, T* a, blasint lda,
              T* af, blasint ldaf, char* equed, RealT<T>* s, T* b, blasint ldb,
              T* x, blasint ldx, RealT<T>* rcond, RealT<T>* ferr,
              RealT<T>* berr, T* work, RealT<T>* rwork, blasint* iwork);

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) indefinite systems (sytrf.cc, sysv.cc). The xHE*
// routines are for complex T only.

// CSYMV / ZSYMV and CSYR / ZSYR for complex T (complex symmetric A); the
// BLAS's xSYMV and xSYR for real T.
template <typename T>
void Symv(char uplo, blasint n, T alpha, const T* a, blasint lda, const T* x,
          blasint incx, T beta, T* y, blasint incy);

template <typename T>
void Syr(char uplo, blasint n, T alpha, const T* x, blasint incx, T* a,
         blasint lda);

template <typename T>
blasint Sytf2(char uplo, blasint n, T* a, blasint lda, blasint* ipiv);

template <typename T>
blasint Lasyf(char uplo, blasint n, blasint nb, blasint* kb, T* a, blasint lda,
              blasint* ipiv, T* w, blasint ldw);

template <typename T>
blasint Sytrf(char uplo, blasint n, T* a, blasint lda, blasint* ipiv, T* work,
              blasint lwork);

template <typename T>
blasint Hetf2(char uplo, blasint n, T* a, blasint lda, blasint* ipiv);

template <typename T>
blasint Lahef(char uplo, blasint n, blasint nb, blasint* kb, T* a, blasint lda,
              blasint* ipiv, T* w, blasint ldw);

template <typename T>
blasint Hetrf(char uplo, blasint n, T* a, blasint lda, blasint* ipiv, T* work,
              blasint lwork);

template <typename T>
blasint Sytrs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const blasint* ipiv, T* b, blasint ldb);

template <typename T>
blasint Hetrs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const blasint* ipiv, T* b, blasint ldb);

template <typename T>
blasint Syconv(char uplo, char way, blasint n, T* a, blasint lda,
               const blasint* ipiv, T* e);

// A is changed, but restored on exit.
template <typename T>
blasint Sytrs2(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
               const blasint* ipiv, T* b, blasint ldb, T* work);

// A is changed, but restored on exit.
template <typename T>
blasint Hetrs2(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
               const blasint* ipiv, T* b, blasint ldb, T* work);

template <typename T>
blasint Sytri(char uplo, blasint n, T* a, blasint lda, const blasint* ipiv,
              T* work);

template <typename T>
blasint Hetri(char uplo, blasint n, T* a, blasint lda, const blasint* ipiv,
              T* work);

// WORK (2 * n elements); IWORK (n) is used for real T only.
template <typename T>
blasint Sycon(char uplo, blasint n, const T* a, blasint lda,
              const blasint* ipiv, RealT<T> anorm, RealT<T>* rcond, T* work,
              blasint* iwork);

template <typename T>
blasint Hecon(char uplo, blasint n, const T* a, blasint lda,
              const blasint* ipiv, RealT<T> anorm, RealT<T>* rcond, T* work);

// WORK (3 * n elements) and IWORK (n) for real T, WORK (2 * n) and RWORK
// (n) for complex T; rwork is used for complex T only, iwork for real T
// only.
template <typename T>
blasint Syrfs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* ferr, RealT<T>* berr,
              T* work, RealT<T>* rwork, blasint* iwork);

template <typename T>
blasint Herfs(char uplo, blasint n, blasint nrhs, const T* a, blasint lda,
              const T* af, blasint ldaf, const blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* ferr, RealT<T>* berr,
              T* work, RealT<T>* rwork);

template <typename T>
blasint Sysv(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
             blasint* ipiv, T* b, blasint ldb, T* work, blasint lwork);

template <typename T>
blasint Hesv(char uplo, blasint n, blasint nrhs, T* a, blasint lda,
             blasint* ipiv, T* b, blasint ldb, T* work, blasint lwork);

// Workspaces as for Syrfs (WORK has LWORK elements).
template <typename T>
blasint Sysvx(char fact, char uplo, blasint n, blasint nrhs, const T* a,
              blasint lda, T* af, blasint ldaf, blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* rcond, RealT<T>* ferr,
              RealT<T>* berr, T* work, blasint lwork, RealT<T>* rwork,
              blasint* iwork);

template <typename T>
blasint Hesvx(char fact, char uplo, blasint n, blasint nrhs, const T* a,
              blasint lda, T* af, blasint ldaf, blasint* ipiv, const T* b,
              blasint ldb, T* x, blasint ldx, RealT<T>* rcond, RealT<T>* ferr,
              RealT<T>* berr, T* work, blasint lwork, RealT<T>* rwork);

// ---------------------------------------------------------------------------
// QR, LQ and RQ factorizations (geqrf.cc)

template <typename T>
blasint Geqr2(blasint m, blasint n, T* a, blasint lda, T* tau, T* work);

template <typename T>
blasint Geqrf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork);

template <typename T>
blasint Gelq2(blasint m, blasint n, T* a, blasint lda, T* tau, T* work);

template <typename T>
blasint Gelqf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork);

template <typename T>
blasint Gerq2(blasint m, blasint n, T* a, blasint lda, T* tau, T* work);

template <typename T>
blasint Gerqf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// QR factorization with column pivoting (geqp3.cc)

template <typename T>
void Laqp2(blasint m, blasint n, blasint offset, T* a, blasint lda,
           blasint* jpvt, T* tau, RealT<T>* vn1, RealT<T>* vn2, T* work);

// Returns KB, the number of columns factored.
template <typename T>
blasint Laqps(blasint m, blasint n, blasint offset, blasint nb, T* a,
              blasint lda, blasint* jpvt, T* tau, RealT<T>* vn1, RealT<T>* vn2,
              T* auxv, T* f, blasint ldf);

// rwork (2 * n elements) is used for complex T only.
template <typename T>
blasint Geqp3(blasint m, blasint n, T* a, blasint lda, blasint* jpvt, T* tau,
              T* work, blasint lwork, RealT<T>* rwork);

// ---------------------------------------------------------------------------
// Orthogonal / unitary matrices from QR, LQ and RQ factorizations (ungqr.cc)

template <typename T>
blasint Ung2r(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work);

template <typename T>
blasint Ungqr(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork);

template <typename T>
blasint Ungl2(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work);

template <typename T>
blasint Unglq(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork);

template <typename T>
blasint Unm2r(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc,
              T* work);

template <typename T>
blasint Unmqr(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

// A is changed (conjugated) for complex T, but restored on exit.
template <typename T>
blasint Unml2(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work);

// A is changed (conjugated) for complex T, but restored on exit.
template <typename T>
blasint Unmr2(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work);

template <typename T>
blasint Unmrq(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

template <typename T>
blasint Unmlq(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// RZ factorization of a trapezoidal matrix and its Z (tzrzf.cc). The complex
// xLARZB, xLARZT and xUNMRZ conjugate V (rows of A) while they run.

template <typename T>
void Larz(char side, blasint m, blasint n, blasint l, const T* v, blasint incv,
          T tau, T* c, blasint ldc, T* work);
template <typename T>
void Larzb(char side, char trans, char direct, char storev, blasint m,
           blasint n, blasint k, blasint l, T* v, blasint ldv, T* t,
           blasint ldt, T* c, blasint ldc, T* work, blasint ldwork);
template <typename T>
void Larzt(char direct, char storev, blasint n, blasint k, T* v, blasint ldv,
           const T* tau, T* t, blasint ldt);
template <typename T>
void Latrz(blasint m, blasint n, blasint l, T* a, blasint lda, T* tau, T* work);
template <typename T>
blasint Tzrzf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork);
template <typename T>
blasint Unmr3(char side, char trans, blasint m, blasint n, blasint k, blasint l,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc,
              T* work);
template <typename T>
blasint Unmrz(char side, char trans, blasint m, blasint n, blasint k, blasint l,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// QL factorization: Q (ungtr.cc)

template <typename T>
blasint Ung2l(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work);

template <typename T>
blasint Ungql(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork);

template <typename T>
blasint Unm2l(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc,
              T* work);

template <typename T>
blasint Unmql(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

template <typename T>
blasint Unmtr(char side, char uplo, char trans, blasint m, blasint n,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) eigenproblems (tridiagonal.cc, ungtr.cc, steqr.cc,
// syev.cc)

template <typename T>
blasint Hetd2(char uplo, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tau);

template <typename T>
void Latrd(char uplo, blasint n, blasint nb, T* a, blasint lda, RealT<T>* e,
           T* tau, T* w, blasint ldw);

template <typename T>
blasint Hetrd(char uplo, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tau, T* work, blasint lwork);

template <typename T>
blasint Ungtr(char uplo, blasint n, T* a, blasint lda, const T* tau, T* work,
              blasint lwork);

// Real R only.
template <typename R>
void Lae2(R a, R b, R c, R* rt1, R* rt2);

template <typename R>
void Laev2(R a, R b, R c, R* rt1, R* rt2, R* cs1, R* sn1);

template <typename R>
void Lartg(R f, R g, R* c, R* s, R* r);

template <typename R>
R Lanst(char norm, blasint n, const R* d, const R* e);

template <typename R>
blasint Lasrt(char id, blasint n, R* d);

template <typename R>
blasint Sterf(blasint n, R* d, R* e);

// The rotations are real also for complex T.
template <typename T>
void Lasr(char side, char pivot, char direct, blasint m, blasint n,
          const RealT<T>* c, const RealT<T>* s, T* a, blasint lda);

template <typename T>
blasint Steqr(char compz, blasint n, RealT<T>* d, RealT<T>* e, T* z,
              blasint ldz, RealT<T>* work);

// rwork (max(1, 3 * n - 2) elements) is used for complex T only.
template <typename T>
blasint Heev(char jobz, char uplo, blasint n, T* a, blasint lda, RealT<T>* w,
             T* work, blasint lwork, RealT<T>* rwork);

// ---------------------------------------------------------------------------
// Divide and conquer symmetric eigensolvers (stedc.cc, syevd.cc, sygv.cc)

// The xLAED* routines are real (R); xSTEDC, xLAED0, xLAED7 and xLAED8 have
// complex overloads (ZSTEDC, ZLAED0, ZLAED7, ZLAED8).
template <typename R>
void Laed5(blasint i, R* d, R* z, R* delta, R rho, R* dlam);

template <typename R>
blasint Laed4(blasint n, blasint i, R* d, R* z, R* delta, R rho, R* dlam);

template <typename R>
blasint Laed3(blasint k, blasint n, blasint n1, R* d, R* q, blasint ldq, R rho,
              R* dlambda, R* q2, blasint* indx, blasint* ctot, R* w, R* s);

template <typename R>
blasint Laed2(blasint* k, blasint n, blasint n1, R* d, R* q, blasint ldq,
              blasint* indxq, R* rho, R* z, R* dlambda, R* w, R* q2,
              blasint* indx, blasint* indxc, blasint* indxp, blasint* coltyp);

template <typename R>
blasint Laed1(blasint n, R* d, R* q, blasint ldq, blasint* indxq, R* rho,
              blasint cutpnt, R* work, blasint* iwork);

template <typename R>
blasint Laeda(blasint n, blasint tlvls, blasint curlvl, blasint curpbm,
              blasint* prmptr, blasint* perm, blasint* givptr, blasint* givcol,
              R* givnum, R* q, blasint* qptr, R* z, R* ztemp);

template <typename R>
blasint Laed9(blasint k, blasint kstart, blasint kstop, blasint n, R* d, R* q,
              blasint ldq, R rho, R* dlambda, R* w, R* s, blasint lds);

template <typename R>
blasint Laed8(blasint icompq, blasint* k, blasint n, blasint qsiz, R* d, R* q,
              blasint ldq, blasint* indxq, R* rho, blasint cutpnt, R* z,
              R* dlambda, R* q2, blasint ldq2, R* w, blasint* perm,
              blasint* givptr, blasint* givcol, R* givnum, blasint* indxp,
              blasint* indx);
template <typename T>
blasint Laed8(blasint* k, blasint n, blasint qsiz, T* q, blasint ldq,
              RealT<T>* d, RealT<T>* rho, blasint cutpnt, RealT<T>* z,
              RealT<T>* dlambda, T* q2, blasint ldq2, RealT<T>* w,
              blasint* indxp, blasint* indx, blasint* indxq, blasint* perm,
              blasint* givptr, blasint* givcol, RealT<T>* givnum);

template <typename R>
blasint Laed7(blasint icompq, blasint n, blasint qsiz, blasint tlvls,
              blasint curlvl, blasint curpbm, R* d, R* q, blasint ldq,
              blasint* indxq, R* rho, blasint cutpnt, R* qstore, blasint* qptr,
              blasint* prmptr, blasint* perm, blasint* givptr, blasint* givcol,
              R* givnum, R* work, blasint* iwork);
template <typename T>
blasint Laed7(blasint n, blasint cutpnt, blasint qsiz, blasint tlvls,
              blasint curlvl, blasint curpbm, RealT<T>* d, T* q, blasint ldq,
              RealT<T>* rho, blasint* indxq, RealT<T>* qstore, blasint* qptr,
              blasint* prmptr, blasint* perm, blasint* givptr, blasint* givcol,
              RealT<T>* givnum, T* work, RealT<T>* rwork, blasint* iwork);

template <typename R>
blasint Laed0(blasint icompq, blasint qsiz, blasint n, R* d, R* e, R* q,
              blasint ldq, R* qstore, blasint ldqs, R* work, blasint* iwork);
template <typename T>
blasint Laed0(blasint qsiz, blasint n, RealT<T>* d, RealT<T>* e, T* q,
              blasint ldq, T* qstore, blasint ldqs, RealT<T>* rwork,
              blasint* iwork);

template <typename R>
blasint Stedc(char compz, blasint n, R* d, R* e, R* z, blasint ldz, R* work,
              blasint lwork, blasint* iwork, blasint liwork);
template <typename T>
blasint Stedc(char compz, blasint n, RealT<T>* d, RealT<T>* e, T* z,
              blasint ldz, T* work, blasint lwork, RealT<T>* rwork,
              blasint lrwork, blasint* iwork, blasint liwork);

// rwork and lrwork are used for complex T only.
template <typename T>
blasint Heevd(char jobz, char uplo, blasint n, T* a, blasint lda, RealT<T>* w,
              T* work, blasint lwork, RealT<T>* rwork, blasint lrwork,
              blasint* iwork, blasint liwork);

// B is restored on exit (complex T conjugates parts of it in between).
template <typename T>
blasint Hegs2(blasint itype, char uplo, blasint n, T* a, blasint lda, T* b,
              blasint ldb);

template <typename T>
blasint Hegst(blasint itype, char uplo, blasint n, T* a, blasint lda, T* b,
              blasint ldb);

// rwork is used for complex T only (max(1, 3 * n - 2) elements).
template <typename T>
blasint Hegv(blasint itype, char jobz, char uplo, blasint n, T* a, blasint lda,
             T* b, blasint ldb, RealT<T>* w, T* work, blasint lwork,
             RealT<T>* rwork);

// rwork and lrwork are used for complex T only.
template <typename T>
blasint Hegvd(blasint itype, char jobz, char uplo, blasint n, T* a, blasint lda,
              T* b, blasint ldb, RealT<T>* w, T* work, blasint lwork,
              RealT<T>* rwork, blasint lrwork, blasint* iwork, blasint liwork);

// Real R only.
template <typename R>
blasint Stevd(char jobz, blasint n, R* d, R* e, R* z, blasint ldz, R* work,
              blasint lwork, blasint* iwork, blasint liwork);

// ---------------------------------------------------------------------------
// Bisection, inverse iteration and MRRR symmetric eigensolvers (stebz.cc,
// stemr.cc, syevr.cc, syevx.cc)

// Real R only.
template <typename R>
blasint Laebz(blasint ijob, blasint nitmax, blasint n, blasint mmax,
              blasint minp, blasint nbmin, R abstol, R reltol, R pivmin, R* d,
              R* e, R* e2, blasint* nval, R* ab, R* c, blasint* mout,
              blasint* nab, R* work, blasint* iwork);
template <typename R>
blasint Stebz(char range, char order, blasint n, R vl, R vu, blasint il,
              blasint iu, R abstol, R* d, R* e, blasint* m, blasint* nsplit,
              R* w, blasint* iblock, blasint* isplit, R* work, blasint* iwork);
template <typename R>
blasint Lagtf(blasint n, R* a, R lambda, R* b, R* c, R tol, R* d, blasint* in);
template <typename R>
blasint Lagts(blasint job, blasint n, R* a, R* b, R* c, R* d, blasint* in, R* y,
              R* tol);
template <typename R>
void Laruv(blasint* iseed, blasint n, R* x);

template <typename T>
void Larnv(blasint idist, blasint* iseed, blasint n, T* x);

// The eigenvectors are real; for complex T they are stored in the complex
// Z.
template <typename T>
blasint Stein(blasint n, RealT<T>* d, RealT<T>* e, blasint m, RealT<T>* w,
              blasint* iblock, blasint* isplit, T* z, blasint ldz,
              RealT<T>* work, blasint* iwork, blasint* ifail);

// Real R only.
template <typename R>
blasint Laneg(blasint n, R* d, R* lld, R sigma, R pivmin, blasint r);
template <typename R>
blasint Larra(blasint n, R* d, R* e, R* e2, R spltol, R tnrm, blasint* nsplit,
              blasint* isplit);
template <typename R>
blasint Larrb(blasint n, R* d, R* lld, blasint ifirst, blasint ilast, R rtol1,
              R rtol2, blasint offset, R* w, R* wgap, R* werr, R* work,
              blasint* iwork, R pivmin, R spdiam, blasint twist);
template <typename R>
blasint Larrc(char jobt, blasint n, R vl, R vu, R* d, R* e, R pivmin,
              blasint* eigcnt, blasint* lcnt, blasint* rcnt);
template <typename R>
blasint Larrd(char range, char order, blasint n, R vl, R vu, blasint il,
              blasint iu, R* gers, R reltol, R* d, R* e, R* e2, R pivmin,
              blasint nsplit, blasint* isplit, blasint* m, R* w, R* werr, R* wl,
              R* wu, blasint* iblock, blasint* indexw, R* work, blasint* iwork);
template <typename R>
blasint Larre(char range, blasint n, R* vl, R* vu, blasint il, blasint iu, R* d,
              R* e, R* e2, R rtol1, R rtol2, R spltol, blasint* nsplit,
              blasint* isplit, blasint* m, R* w, R* werr, R* wgap,
              blasint* iblock, blasint* indexw, R* gers, R* pivmin, R* work,
              blasint* iwork);
template <typename R>
blasint Larrf(blasint n, R* d, R* l, R* ld, blasint clstrt, blasint clend, R* w,
              R* wgap, R* werr, R spdiam, R clgapl, R clgapr, R pivmin,
              R* sigma, R* dplus, R* lplus, R* work);
template <typename R>
blasint Larrj(blasint n, R* d, R* e2, blasint ifirst, blasint ilast, R rtol,
              blasint offset, R* w, R* werr, R* work, blasint* iwork, R pivmin,
              R spdiam);
template <typename R>
blasint Larrk(blasint n, blasint iw, R gl, R gu, R* d, R* e2, R pivmin,
              R reltol, R* w, R* werr);
template <typename R>
blasint Larrr(blasint n, R* d, R* e);

// The eigenvectors are real; for complex T they are stored in the complex
// Z.
template <typename T>
void Lar1v(blasint n, blasint b1, blasint bn, RealT<T> lambda, RealT<T>* d,
           RealT<T>* l, RealT<T>* ld, RealT<T>* lld, RealT<T> pivmin,
           RealT<T> gaptol, T* z, bool wantnc, blasint* negcnt, RealT<T>* ztz,
           RealT<T>* mingma, blasint* r, blasint* isuppz, RealT<T>* nrminv,
           RealT<T>* resid, RealT<T>* rqcorr, RealT<T>* work);
template <typename T>
blasint Larrv(blasint n, RealT<T> vl, RealT<T> vu, RealT<T>* d, RealT<T>* l,
              RealT<T> pivmin, blasint* isplit, blasint m, blasint dol,
              blasint dou, RealT<T> minrgp, RealT<T>* rtol1, RealT<T>* rtol2,
              RealT<T>* w, RealT<T>* werr, RealT<T>* wgap, blasint* iblock,
              blasint* indexw, RealT<T>* gers, T* z, blasint ldz,
              blasint* isuppz, RealT<T>* work, blasint* iwork);
template <typename T>
blasint Stemr(char jobz, char range, blasint n, RealT<T>* d, RealT<T>* e,
              RealT<T> vl, RealT<T> vu, blasint il, blasint iu, blasint* m,
              RealT<T>* w, T* z, blasint ldz, blasint nzc, blasint* isuppz,
              bool* tryrac, RealT<T>* work, blasint lwork, blasint* iwork,
              blasint liwork);

// RWORK and LRWORK are referenced for complex T only.
template <typename T>
blasint Heevr(char jobz, char range, char uplo, blasint n, T* a, blasint lda,
              RealT<T> vl, RealT<T> vu, blasint il, blasint iu, RealT<T> abstol,
              blasint* m, RealT<T>* w, T* z, blasint ldz, blasint* isuppz,
              T* work, blasint lwork, RealT<T>* rwork, blasint lrwork,
              blasint* iwork, blasint liwork);

// Real R only.
template <typename R>
blasint Stevr(char jobz, char range, blasint n, R* d, R* e, R vl, R vu,
              blasint il, blasint iu, R abstol, blasint* m, R* w, R* z,
              blasint ldz, blasint* isuppz, R* work, blasint lwork,
              blasint* iwork, blasint liwork);

// RWORK is referenced for complex T only.
template <typename T>
blasint Heevx(char jobz, char range, char uplo, blasint n, T* a, blasint lda,
              RealT<T> vl, RealT<T> vu, blasint il, blasint iu, RealT<T> abstol,
              blasint* m, RealT<T>* w, T* z, blasint ldz, T* work,
              blasint lwork, RealT<T>* rwork, blasint* iwork, blasint* ifail);

// Real R only.
template <typename R>
blasint Stevx(char jobz, char range, blasint n, R* d, R* e, R vl, R vu,
              blasint il, blasint iu, R abstol, blasint* m, R* w, R* z,
              blasint ldz, R* work, blasint* iwork, blasint* ifail);

// ---------------------------------------------------------------------------
// Linear least squares (gels.cc, gelsd.cc). RWORK is referenced for complex
// T only.

template <typename T>
void Laic1(blasint job, blasint j, const T* x, RealT<T> sest, const T* w,
           T gamma, RealT<T>* sestpr, T* s, T* c);
template <typename T>
blasint Gels(char trans, blasint m, blasint n, blasint nrhs, T* a, blasint lda,
             T* b, blasint ldb, T* work, blasint lwork);
template <typename T>
blasint Gelss(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
              blasint ldb, RealT<T>* s, RealT<T> rcond, blasint* rank, T* work,
              blasint lwork, RealT<T>* rwork);
template <typename T>
blasint Gelsy(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
              blasint ldb, blasint* jpvt, RealT<T> rcond, blasint* rank,
              T* work, blasint lwork, RealT<T>* rwork);

// The singular vectors are real; for complex T, B and BX (and WORK of
// xLALSD) are complex, the workspace of xLALS0 and xLALSA is real.
template <typename T>
blasint Lals0(blasint icompq, blasint nl, blasint nr, blasint sqre,
              blasint nrhs, T* b, blasint ldb, T* bx, blasint ldbx,
              blasint* perm, blasint givptr, blasint* givcol, blasint ldgcol,
              RealT<T>* givnum, blasint ldgnum, RealT<T>* poles, RealT<T>* difl,
              RealT<T>* difr, RealT<T>* z, blasint k, RealT<T> c, RealT<T> s,
              RealT<T>* work);
template <typename T>
blasint Lalsa(blasint icompq, blasint smlsiz, blasint n, blasint nrhs, T* b,
              blasint ldb, T* bx, blasint ldbx, RealT<T>* u, blasint ldu,
              RealT<T>* vt, blasint* k, RealT<T>* difl, RealT<T>* difr,
              RealT<T>* z, RealT<T>* poles, blasint* givptr, blasint* givcol,
              blasint ldgcol, blasint* perm, RealT<T>* givnum, RealT<T>* c,
              RealT<T>* s, RealT<T>* work, blasint* iwork);
template <typename T>
blasint Lalsd(char uplo, blasint smlsiz, blasint n, blasint nrhs, RealT<T>* d,
              RealT<T>* e, T* b, blasint ldb, RealT<T> rcond, blasint* rank,
              T* work, RealT<T>* rwork, blasint* iwork);
template <typename T>
blasint Gelsd(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
              blasint ldb, RealT<T>* s, RealT<T> rcond, blasint* rank, T* work,
              blasint lwork, RealT<T>* rwork, blasint* iwork);

// ---------------------------------------------------------------------------
// Generalized linear least squares (gglse.cc)

template <typename T>
blasint Ggqrf(blasint n, blasint m, blasint p, T* a, blasint lda, T* taua, T* b,
              blasint ldb, T* taub, T* work, blasint lwork);
template <typename T>
blasint Ggrqf(blasint m, blasint p, blasint n, T* a, blasint lda, T* taua, T* b,
              blasint ldb, T* taub, T* work, blasint lwork);
template <typename T>
blasint Gglse(blasint m, blasint n, blasint p, T* a, blasint lda, T* b,
              blasint ldb, T* c, T* d, T* x, T* work, blasint lwork);
template <typename T>
blasint Ggglm(blasint n, blasint m, blasint p, T* a, blasint lda, T* b,
              blasint ldb, T* d, T* x, T* y, T* work, blasint lwork);

// ---------------------------------------------------------------------------
// Bidiagonal reduction (bidiagonal.cc)

template <typename T>
blasint Gebd2(blasint m, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tauq, T* taup, T* work);

template <typename T>
void Labrd(blasint m, blasint n, blasint nb, T* a, blasint lda, RealT<T>* d,
           RealT<T>* e, T* tauq, T* taup, T* x, blasint ldx, T* y, blasint ldy);

template <typename T>
blasint Gebrd(blasint m, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tauq, T* taup, T* work, blasint lwork);

template <typename T>
blasint Ungbr(char vect, blasint m, blasint n, blasint k, T* a, blasint lda,
              const T* tau, T* work, blasint lwork);

// A is changed (conjugated) for complex T and VECT = 'P', but restored on
// exit.
template <typename T>
blasint Unmbr(char vect, char side, char trans, blasint m, blasint n, blasint k,
              T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// Singular value decomposition (bdsqr.cc, lasq.cc, gesvd.cc)

// Real R only.
template <typename R>
void Las2(R f, R g, R h, R* ssmin, R* ssmax);

template <typename R>
void Lasv2(R f, R g, R h, R* ssmin, R* ssmax, R* snr, R* csr, R* snl, R* csl);

template <typename R>
void Lasq6(blasint i0, blasint n0, R* z, blasint pp, R* dmin, R* dmin1,
           R* dmin2, R* dn, R* dnm1, R* dnm2);

template <typename R>
void Lasq5(blasint i0, blasint n0, R* z, blasint pp, R* tau, R sigma, R* dmin,
           R* dmin1, R* dmin2, R* dn, R* dnm1, R* dnm2, bool ieee, R eps);

template <typename R>
void Lasq4(blasint i0, blasint n0, const R* z, blasint pp, blasint n0in, R dmin,
           R dmin1, R dmin2, R dn, R dn1, R dn2, R* tau, blasint* ttype, R* g);

template <typename R>
void Lasq3(blasint i0, blasint* n0, R* z, blasint* pp, R* dmin, R* sigma,
           R* desig, R* qmax, blasint* nfail, blasint* iter, blasint* ndiv,
           bool ieee, blasint* ttype, R* dmin1, R* dmin2, R* dn, R* dn1, R* dn2,
           R* g, R* tau);

template <typename R>
blasint Lasq2(blasint n, R* z);

template <typename R>
blasint Lasq1(blasint n, R* d, R* e, R* work);

// The rotations are real also for complex T.
template <typename T>
blasint Bdsqr(char uplo, blasint n, blasint ncvt, blasint nru, blasint ncc,
              RealT<T>* d, RealT<T>* e, T* vt, blasint ldvt, T* u, blasint ldu,
              T* c, blasint ldc, RealT<T>* work);

// rwork (5 * min(m, n) elements) is used for complex T only.
template <typename T>
blasint Gesvd(char jobu, char jobvt, blasint m, blasint n, T* a, blasint lda,
              RealT<T>* s, T* u, blasint ldu, T* vt, blasint ldvt, T* work,
              blasint lwork, RealT<T>* rwork);

// ---------------------------------------------------------------------------
// Divide and conquer singular value decomposition (bdsdc.cc, gesdd.cc)

// Real R only.
template <typename R>
void Lamrg(blasint n1, blasint n2, const R* a, blasint dtrd1, blasint dtrd2,
           blasint* index);

template <typename R>
void Lasdt(blasint n, blasint* lvl, blasint* nd, blasint* inode, blasint* ndiml,
           blasint* ndimr, blasint msub);

template <typename R>
blasint Laed6(blasint kniter, bool orgati, R rho, const R* d, const R* z,
              R finit, R* tau);

template <typename R>
void Lasd5(blasint i, const R* d, const R* z, R* delta, R rho, R* dsigma,
           R* work);

template <typename R>
blasint Lasd4(blasint n, blasint i, const R* d, const R* z, R* delta, R rho,
              R* sigma, R* work);

template <typename R>
blasint Lasd8(blasint icompq, blasint k, R* d, R* z, R* vf, R* vl, R* difl,
              R* difr, blasint lddifr, R* dsigma, R* work);

template <typename R>
blasint Lasd7(blasint icompq, blasint nl, blasint nr, blasint sqre, blasint* k,
              R* d, R* z, R* zw, R* vf, R* vfw, R* vl, R* vlw, R alpha, R beta,
              R* dsigma, blasint* idx, blasint* idxp, blasint* idxq,
              blasint* perm, blasint* givptr, blasint* givcol, blasint ldgcol,
              R* givnum, blasint ldgnum, R* c, R* s);

template <typename R>
blasint Lasd6(blasint icompq, blasint nl, blasint nr, blasint sqre, R* d, R* vf,
              R* vl, R* alpha, R* beta, blasint* idxq, blasint* perm,
              blasint* givptr, blasint* givcol, blasint ldgcol, R* givnum,
              blasint ldgnum, R* poles, R* difl, R* difr, R* z, blasint* k,
              R* c, R* s, R* work, blasint* iwork);

template <typename R>
blasint Lasd3(blasint nl, blasint nr, blasint sqre, blasint k, R* d, R* q,
              blasint ldq, R* dsigma, R* u, blasint ldu, R* u2, blasint ldu2,
              R* vt, blasint ldvt, R* vt2, blasint ldvt2, const blasint* idxc,
              const blasint* ctot, R* z);

template <typename R>
blasint Lasd2(blasint nl, blasint nr, blasint sqre, blasint* k, R* d, R* z,
              R alpha, R beta, R* u, blasint ldu, R* vt, blasint ldvt,
              R* dsigma, R* u2, blasint ldu2, R* vt2, blasint ldvt2,
              blasint* idxp, blasint* idx, blasint* idxc, blasint* idxq,
              blasint* coltyp);

template <typename R>
blasint Lasd1(blasint nl, blasint nr, blasint sqre, R* d, R* alpha, R* beta,
              R* u, blasint ldu, R* vt, blasint ldvt, blasint* idxq,
              blasint* iwork, R* work);

template <typename R>
blasint Lasdq(char uplo, blasint sqre, blasint n, blasint ncvt, blasint nru,
              blasint ncc, R* d, R* e, R* vt, blasint ldvt, R* u, blasint ldu,
              R* c, blasint ldc, R* work);

template <typename R>
blasint Lasd0(blasint n, blasint sqre, R* d, R* e, R* u, blasint ldu, R* vt,
              blasint ldvt, blasint smlsiz, blasint* iwork, R* work);

template <typename R>
blasint Lasda(blasint icompq, blasint smlsiz, blasint n, blasint sqre, R* d,
              R* e, R* u, blasint ldu, R* vt, blasint* k, R* difl, R* difr,
              R* z, R* poles, blasint* givptr, blasint* givcol, blasint ldgcol,
              blasint* perm, R* givnum, R* c, R* s, R* work, blasint* iwork);

template <typename R>
blasint Bdsdc(char uplo, char compq, blasint n, R* d, R* e, R* u, blasint ldu,
              R* vt, blasint ldvt, R* q, blasint* iq, R* work, blasint* iwork);

// Complex T only: B := A for real A (xLACP2), C := A * B for complex A and
// real B (xLACRM), C := A * B for real A and complex B (xLARCM).
template <typename T>
void Lacp2(char uplo, blasint m, blasint n, const RealT<T>* a, blasint lda,
           T* b, blasint ldb);

template <typename T>
void Lacrm(blasint m, blasint n, const T* a, blasint lda, const RealT<T>* b,
           blasint ldb, T* c, blasint ldc, RealT<T>* rwork);

template <typename T>
void Larcm(blasint m, blasint n, const RealT<T>* a, blasint lda, const T* b,
           blasint ldb, T* c, blasint ldc, RealT<T>* rwork);

// rwork is used for complex T only (see xGESDD for its size); iwork has
// 8 * min(m, n) elements.
template <typename T>
blasint Gesdd(char jobz, blasint m, blasint n, T* a, blasint lda, RealT<T>* s,
              T* u, blasint ldu, T* vt, blasint ldvt, T* work, blasint lwork,
              RealT<T>* rwork, blasint* iwork);

// ---------------------------------------------------------------------------
// Nonsymmetric eigenproblem: balancing and Hessenberg reduction
// (hessenberg.cc), Hessenberg QR (hseqr.cc), reordering of the Schur form
// (trsen.cc), Schur factorization (gees.cc). Where the real and the
// complex routine have different arguments (WR and WI for W, pairs of real
// shifts for complex ones), they are two overloads of the same name.

template <typename T>
blasint Gebal(char job, blasint n, T* a, blasint lda, blasint* ilo,
              blasint* ihi, RealT<T>* scale);

template <typename T>
blasint Gebak(char job, char side, blasint n, blasint ilo, blasint ihi,
              const RealT<T>* scale, blasint m, T* v, blasint ldv);

template <typename T>
blasint Gehd2(blasint n, blasint ilo, blasint ihi, T* a, blasint lda, T* tau,
              T* work);

template <typename T>
void Lahr2(blasint n, blasint k, blasint nb, T* a, blasint lda, T* tau, T* t,
           blasint ldt, T* y, blasint ldy);

template <typename T>
blasint Gehrd(blasint n, blasint ilo, blasint ihi, T* a, blasint lda, T* tau,
              T* work, blasint lwork);

// xORGHR for real T, xUNGHR for complex T.
template <typename T>
blasint Unghr(blasint n, blasint ilo, blasint ihi, T* a, blasint lda,
              const T* tau, T* work, blasint lwork);

// xORMHR for real T, xUNMHR for complex T.
template <typename T>
blasint Unmhr(char side, char trans, blasint m, blasint n, blasint ilo,
              blasint ihi, const T* a, blasint lda, const T* tau, T* c,
              blasint ldc, T* work, blasint lwork);

// Real T only.
template <typename R>
void Lanv2(R* a, R* b, R* c, R* d, R* rt1r, R* rt1i, R* rt2r, R* rt2i, R* cs,
           R* sn);

template <typename R>
void Laqr1(blasint n, const R* h, blasint ldh, R sr1, R si1, R sr2, R si2,
           R* v);
template <typename T>
void Laqr1(blasint n, const T* h, blasint ldh, T s1, T s2, T* v);

template <typename R>
blasint Lahqr(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, blasint iloz, blasint ihiz, R* z,
              blasint ldz);
template <typename T>
blasint Lahqr(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, blasint iloz, blasint ihiz, T* z, blasint ldz);

template <typename R>
void Laqr5(bool wantt, bool wantz, blasint kacc22, blasint n, blasint ktop,
           blasint kbot, blasint nshfts, R* sr, R* si, R* h, blasint ldh,
           blasint iloz, blasint ihiz, R* z, blasint ldz, R* v, blasint ldv,
           R* u, blasint ldu, blasint nv, R* wv, blasint ldwv, blasint nh,
           R* wh, blasint ldwh);
template <typename T>
void Laqr5(bool wantt, bool wantz, blasint kacc22, blasint n, blasint ktop,
           blasint kbot, blasint nshfts, T* s, T* h, blasint ldh, blasint iloz,
           blasint ihiz, T* z, blasint ldz, T* v, blasint ldv, T* u,
           blasint ldu, blasint nv, T* wv, blasint ldwv, blasint nh, T* wh,
           blasint ldwh);

template <typename R>
void Laqr2(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, R* h, blasint ldh, blasint iloz, blasint ihiz, R* z,
           blasint ldz, blasint* ns, blasint* nd, R* sr, R* si, R* v,
           blasint ldv, blasint nh, R* t, blasint ldt, blasint nv, R* wv,
           blasint ldwv, R* work, blasint lwork);
template <typename T>
void Laqr2(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, T* h, blasint ldh, blasint iloz, blasint ihiz, T* z,
           blasint ldz, blasint* ns, blasint* nd, T* sh, T* v, blasint ldv,
           blasint nh, T* t, blasint ldt, blasint nv, T* wv, blasint ldwv,
           T* work, blasint lwork);

template <typename R>
void Laqr3(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, R* h, blasint ldh, blasint iloz, blasint ihiz, R* z,
           blasint ldz, blasint* ns, blasint* nd, R* sr, R* si, R* v,
           blasint ldv, blasint nh, R* t, blasint ldt, blasint nv, R* wv,
           blasint ldwv, R* work, blasint lwork);
template <typename T>
void Laqr3(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, T* h, blasint ldh, blasint iloz, blasint ihiz, T* z,
           blasint ldz, blasint* ns, blasint* nd, T* sh, T* v, blasint ldv,
           blasint nh, T* t, blasint ldt, blasint nv, T* wv, blasint ldwv,
           T* work, blasint lwork);

template <typename R>
blasint Laqr4(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, blasint iloz, blasint ihiz, R* z,
              blasint ldz, R* work, blasint lwork);
template <typename T>
blasint Laqr4(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, blasint iloz, blasint ihiz, T* z, blasint ldz,
              T* work, blasint lwork);

template <typename R>
blasint Laqr0(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, blasint iloz, blasint ihiz, R* z,
              blasint ldz, R* work, blasint lwork);
template <typename T>
blasint Laqr0(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, blasint iloz, blasint ihiz, T* z, blasint ldz,
              T* work, blasint lwork);

template <typename R>
blasint Hseqr(char job, char compz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, R* z, blasint ldz, R* work,
              blasint lwork);
template <typename T>
blasint Hseqr(char job, char compz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, T* z, blasint ldz, T* work, blasint lwork);

// xLARFX: applies H = I - tau * v * v**H to C, unrolled for orders <= 10.
template <typename T>
void Larfx(char side, blasint m, blasint n, const T* v, T tau, T* c,
           blasint ldc, T* work);

// Complex T only: xLARTG for complex F and G (real C), xROT with complex S
// (CROT / ZROT; the real-S rotation is the BLAS's, see blas.h), ICMAX1 /
// IZMAX1 and SCSUM1 / DZSUM1 (true absolute values).
template <typename T>
std::enable_if_t<IsComplex<T>::value> Lartg(T f, T g, RealT<T>* c, T* s, T* r);

template <typename T>
std::enable_if_t<IsComplex<T>::value> Rot(blasint n, T* cx, blasint incx, T* cy,
                                          blasint incy, RealT<T> c, T s);

template <typename T>
blasint Imax1(blasint n, const T* zx, blasint incx);

template <typename T>
RealT<T> Sum1(blasint n, const T* cx, blasint incx);

// Real T only.
template <typename R>
blasint Lasy2(bool ltranl, bool ltranr, blasint isgn, blasint n1, blasint n2,
              const R* tl, blasint ldtl, const R* tr, blasint ldtr, const R* b,
              blasint ldb, R* scale, R* x, blasint ldx, R* xnorm);

template <typename R>
blasint Laln2(bool ltrans, blasint na, blasint nw, R smin, R ca, const R* a,
              blasint lda, R d1, R d2, const R* b, blasint ldb, R wr, R wi,
              R* x, blasint ldx, R* scale, R* xnorm);

template <typename R>
blasint Laexc(bool wantq, blasint n, R* t, blasint ldt, R* q, blasint ldq,
              blasint j1, blasint n1, blasint n2, R* work);

template <typename R>
void Lacn2(blasint n, R* v, R* x, blasint* isgn, R* est, blasint* kase,
           blasint* isave);
template <typename T>
void Lacn2(blasint n, T* v, T* x, RealT<T>* est, blasint* kase, blasint* isave);

template <typename T>
blasint Trsyl(char trana, char tranb, blasint isgn, blasint m, blasint n,
              const T* a, blasint lda, const T* b, blasint ldb, T* c,
              blasint ldc, RealT<T>* scale);

template <typename R>
blasint Trexc(char compq, blasint n, R* t, blasint ldt, R* q, blasint ldq,
              blasint* ifst, blasint* ilst, R* work);
template <typename T>
blasint Trexc(char compq, blasint n, T* t, blasint ldt, T* q, blasint ldq,
              blasint ifst, blasint ilst);

// SELECT is a LOGICAL array (nonzero is true).
template <typename R>
blasint Trsen(char job, char compq, const blasint* select, blasint n, R* t,
              blasint ldt, R* q, blasint ldq, R* wr, R* wi, blasint* m, R* s,
              R* sep, R* work, blasint lwork, blasint* iwork, blasint liwork);
template <typename T>
blasint Trsen(char job, char compq, const blasint* select, blasint n, T* t,
              blasint ldt, T* q, blasint ldq, T* w, blasint* m, RealT<T>* s,
              RealT<T>* sep, T* work, blasint lwork);

// The SELECT functions of xGEES, with the Fortran calling convention:
// SELECT(WR, WI) for real T, SELECT(W) for complex T (W points to the real
// and the imaginary part). Nonzero is true.
template <typename R>
using GeesSelectReal = blasint (*)(const R*, const R*);
template <typename R>
using GeesSelectComplex = blasint (*)(const R*);

// BWORK is a LOGICAL array of N elements (used if SORT = 'S').
template <typename R>
blasint Gees(char jobvs, char sort, GeesSelectReal<R> select, blasint n, R* a,
             blasint lda, blasint* sdim, R* wr, R* wi, R* vs, blasint ldvs,
             R* work, blasint lwork, blasint* bwork);
template <typename T>
blasint Gees(char jobvs, char sort, GeesSelectComplex<RealT<T>> select,
             blasint n, T* a, blasint lda, blasint* sdim, T* w, T* vs,
             blasint ldvs, T* work, blasint lwork, RealT<T>* rwork,
             blasint* bwork);

// ---------------------------------------------------------------------------
// Eigenvectors and condition numbers of a general matrix (trevc.cc,
// trsna.cc, geev.cc). SELECT is a LOGICAL array (nonzero is true); xTREVC3
// normalizes the real one (in/out).

template <typename R>
blasint Trevc3(char side, char howmny, blasint* select, blasint n, const R* t,
               blasint ldt, R* vl, blasint ldvl, R* vr, blasint ldvr,
               blasint mm, blasint* m, R* work, blasint lwork);
template <typename T>
blasint Trevc3(char side, char howmny, const blasint* select, blasint n, T* t,
               blasint ldt, T* vl, blasint ldvl, T* vr, blasint ldvr,
               blasint mm, blasint* m, T* work, blasint lwork, RealT<T>* rwork,
               blasint lrwork);

// Real T only.
template <typename R>
blasint Laqtr(bool ltran, bool lreal, blasint n, const R* t, blasint ldt,
              const R* b, R w, R* scale, R* x, R* work);

template <typename R>
blasint Trsna(char job, char howmny, const blasint* select, blasint n,
              const R* t, blasint ldt, const R* vl, blasint ldvl, const R* vr,
              blasint ldvr, R* s, R* sep, blasint mm, blasint* m, R* work,
              blasint ldwork, blasint* iwork);
template <typename T>
blasint Trsna(char job, char howmny, const blasint* select, blasint n,
              const T* t, blasint ldt, const T* vl, blasint ldvl, const T* vr,
              blasint ldvr, RealT<T>* s, RealT<T>* sep, blasint mm, blasint* m,
              T* work, blasint ldwork, RealT<T>* rwork);

template <typename R>
blasint Geev(char jobvl, char jobvr, blasint n, R* a, blasint lda, R* wr, R* wi,
             R* vl, blasint ldvl, R* vr, blasint ldvr, R* work, blasint lwork);
template <typename T>
blasint Geev(char jobvl, char jobvr, blasint n, T* a, blasint lda, T* w, T* vl,
             blasint ldvl, T* vr, blasint ldvr, T* work, blasint lwork,
             RealT<T>* rwork);

template <typename R>
blasint Geevx(char balanc, char jobvl, char jobvr, char sense, blasint n, R* a,
              blasint lda, R* wr, R* wi, R* vl, blasint ldvl, R* vr,
              blasint ldvr, blasint* ilo, blasint* ihi, R* scale, R* abnrm,
              R* rconde, R* rcondv, R* work, blasint lwork, blasint* iwork);
template <typename T>
blasint Geevx(char balanc, char jobvl, char jobvr, char sense, blasint n, T* a,
              blasint lda, T* w, T* vl, blasint ldvl, T* vr, blasint ldvr,
              blasint* ilo, blasint* ihi, RealT<T>* scale, RealT<T>* abnrm,
              RealT<T>* rconde, RealT<T>* rcondv, T* work, blasint lwork,
              RealT<T>* rwork);

// BWORK is a LOGICAL array of N elements (used if SORT = 'S').
template <typename R>
blasint Geesx(char jobvs, char sort, GeesSelectReal<R> select, char sense,
              blasint n, R* a, blasint lda, blasint* sdim, R* wr, R* wi, R* vs,
              blasint ldvs, R* rconde, R* rcondv, R* work, blasint lwork,
              blasint* iwork, blasint liwork, blasint* bwork);
template <typename T>
blasint Geesx(char jobvs, char sort, GeesSelectComplex<RealT<T>> select,
              char sense, blasint n, T* a, blasint lda, blasint* sdim, T* w,
              T* vs, blasint ldvs, RealT<T>* rconde, RealT<T>* rcondv, T* work,
              blasint lwork, RealT<T>* rwork, blasint* bwork);

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_LAPACK_ROUTINES_H_
