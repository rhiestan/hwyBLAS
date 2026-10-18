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

// ILAxLC: the last non-zero column of A, or 0 if A is zero.
template <typename T>
blasint Ilalc(blasint m, blasint n, const T* a, blasint lda);

// ILAxLR: the last non-zero row of A, or 0 if A is zero.
template <typename T>
blasint Ilalr(blasint m, blasint n, const T* a, blasint lda);

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

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_LAPACK_ROUTINES_H_
