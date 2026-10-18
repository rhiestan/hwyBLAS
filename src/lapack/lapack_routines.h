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

// ---------------------------------------------------------------------------
// Cholesky factorization (potrf.cc)

template <typename T>
blasint Potrf2(char uplo, blasint n, T* a, blasint lda);

template <typename T>
blasint Potrf(char uplo, blasint n, T* a, blasint lda);

// ---------------------------------------------------------------------------
// QR and LQ factorizations (geqrf.cc)

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
// Orthogonal / unitary matrices from QR and LQ factorizations (ungqr.cc)

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
blasint Unmlq(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork);

// ---------------------------------------------------------------------------
// QL factorization: Q (ungtr.cc)

template <typename T>
blasint Ung2l(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work);

template <typename T>
blasint Ungql(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork);

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

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_LAPACK_ROUTINES_H_
