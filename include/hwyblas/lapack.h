/*
 * hwyBLAS - LAPACK routines (Fortran interface).
 *
 * The routines are ported from the reference implementation (LAPACK 3.12.1)
 * and have its Fortran calling convention: lower-case names with a trailing
 * underscore, every argument passed by pointer, column-major storage. As in
 * hwyblas.h, complex arguments are passed as float* / double* pointing to
 * interleaved (real, imag) pairs. Only the first character of CHARACTER
 * arguments is used, so the hidden length arguments that Fortran passes are
 * not declared, except where the whole string matters (ILAENV, LSAME and
 * xLAMCH).
 *
 * Invalid arguments are reported like those of the BLAS routines (xerbla_
 * or the handler installed with hwyblas_set_error_handler), with INFO set
 * to minus the argument's position.
 *
 * Derived from LAPACK, Copyright (c) 1992-2023 The University of Tennessee
 * and The University of Tennessee Research Foundation, The University of
 * California Berkeley, and The University of Colorado Denver. Modified BSD
 * license, see LICENSE-LAPACK.
 */
#ifndef HWYBLAS_LAPACK_H_
#define HWYBLAS_LAPACK_H_

#include "hwyblas/hwyblas.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------ */
/* Environment and machine parameters                                       */

/* ILAENV(ispec, name, opts, n1, n2, n3, n4): tuning parameters of the LAPACK
 * routines, e.g. the block size (ispec = 1) of the routine `name`. */
HWYBLAS_API hwyblas_int ilaenv_(const hwyblas_int* ispec, const char* name,
                                const char* opts, const hwyblas_int* n1,
                                const hwyblas_int* n2, const hwyblas_int* n3,
                                const hwyblas_int* n4, size_t name_len,
                                size_t opts_len);

/* Parameters of the nonsymmetric eigenvalue routines (ILAENV ispec 12-17). */
HWYBLAS_API hwyblas_int iparmq_(const hwyblas_int* ispec, const char* name,
                                const char* opts, const hwyblas_int* n,
                                const hwyblas_int* ilo, const hwyblas_int* ihi,
                                const hwyblas_int* lwork, size_t name_len,
                                size_t opts_len);

/* 1 if infinity (ispec = 0) or also NaN (ispec = 1) arithmetic works as in
 * IEEE 754, else 0. */
HWYBLAS_API hwyblas_int ieeeck_(const hwyblas_int* ispec, const float* zero,
                                const float* one);

/* Fortran LOGICAL (1 or 0): whether ca and cb are the same letter, ignoring
 * case. */
HWYBLAS_API hwyblas_int lsame_(const char* ca, const char* cb, size_t ca_len,
                               size_t cb_len);

/* Machine parameters: 'E' unit roundoff (half the machine epsilon), 'S'
 * safe minimum, 'B' base, 'P' eps * base, 'N' mantissa digits, 'R' 1 (the
 * rounding mode), 'M' minimum exponent, 'U' underflow threshold, 'L' maximum
 * exponent, 'O' overflow threshold. */
HWYBLAS_API float slamch_(const char* cmach, size_t cmach_len);
HWYBLAS_API double dlamch_(const char* cmach, size_t cmach_len);
HWYBLAS_API float slamc3_(const float* a, const float* b);
HWYBLAS_API double dlamc3_(const double* a, const double* b);

/* The signature of ilaenv_. */
typedef hwyblas_int (*hwyblas_ilaenv_function)(
    const hwyblas_int* ispec, const char* name, const char* opts,
    const hwyblas_int* n1, const hwyblas_int* n2, const hwyblas_int* n3,
    const hwyblas_int* n4, size_t name_len, size_t opts_len);

/* Installs a function that the LAPACK routines call instead of ilaenv_, and
 * returns the previous one; NULL restores ilaenv_. As with xerbla_, a DLL's
 * ilaenv_ cannot be replaced at link time: this lets, e.g., LAPACK's test
 * programs set the block sizes. Not thread-safe with respect to running
 * LAPACK routines. */
HWYBLAS_API hwyblas_ilaenv_function
hwyblas_set_ilaenv(hwyblas_ilaenv_function function);

/* ------------------------------------------------------------------------ */
/* LU factorization                                                         */

/* A = P * L * U with partial pivoting (row interchanges); A is m x n, L unit
 * lower triangular (lower trapezoidal if m > n), U upper triangular (upper
 * trapezoidal if m < n). ipiv(i) is the row interchanged with row i. info > 0:
 * U(info, info) is exactly zero (the factorization is complete, but U is
 * singular). xGETRF is blocked, xGETRF2 recursive. */
HWYBLAS_API void sgetrf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void dgetrf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void cgetrf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void zgetrf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void sgetrf2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                          const hwyblas_int* lda, hwyblas_int* ipiv,
                          hwyblas_int* info);
HWYBLAS_API void dgetrf2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                          const hwyblas_int* lda, hwyblas_int* ipiv,
                          hwyblas_int* info);
HWYBLAS_API void cgetrf2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                          const hwyblas_int* lda, hwyblas_int* ipiv,
                          hwyblas_int* info);
HWYBLAS_API void zgetrf2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                          const hwyblas_int* lda, hwyblas_int* ipiv,
                          hwyblas_int* info);

/* Applies the row interchanges ipiv(k1 ... k2) (with increment incx; in
 * reverse order for incx < 0) to the n columns of A. */
HWYBLAS_API void slaswp_(const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const hwyblas_int* k1, const hwyblas_int* k2,
                         const hwyblas_int* ipiv, const hwyblas_int* incx);
HWYBLAS_API void dlaswp_(const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* k1,
                         const hwyblas_int* k2, const hwyblas_int* ipiv,
                         const hwyblas_int* incx);
HWYBLAS_API void claswp_(const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const hwyblas_int* k1, const hwyblas_int* k2,
                         const hwyblas_int* ipiv, const hwyblas_int* incx);
HWYBLAS_API void zlaswp_(const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* k1,
                         const hwyblas_int* k2, const hwyblas_int* ipiv,
                         const hwyblas_int* incx);

/* ------------------------------------------------------------------------ */
/* Auxiliary routines                                                       */

/* Conjugates the vector x (n elements, increment incx). */
HWYBLAS_API void clacgv_(const hwyblas_int* n, float* x,
                         const hwyblas_int* incx);
HWYBLAS_API void zlacgv_(const hwyblas_int* n, double* x,
                         const hwyblas_int* incx);

/* Copies all of the m x n matrix A to B (uplo other than 'U' or 'L'), or
 * its upper ('U') or lower ('L') triangle or trapezoid. */
HWYBLAS_API void slacpy_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void dlacpy_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void clacpy_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void zlacpy_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb);

/* sqrt(x^2 + y^2) and sqrt(x^2 + y^2 + z^2), avoiding unnecessary
 * overflow and underflow. */
HWYBLAS_API float slapy2_(const float* x, const float* y);
HWYBLAS_API double dlapy2_(const double* x, const double* y);
HWYBLAS_API float slapy3_(const float* x, const float* y, const float* z);
HWYBLAS_API double dlapy3_(const double* x, const double* y, const double* z);

/* p + i*q = (a + i*b) / (c + i*d), robustly (Baudin and Smith). The
 * complex functions CLADIV and ZLADIV are not exported: Fortran compilers
 * return complex function values in different ways. */
HWYBLAS_API void sladiv_(const float* a, const float* b, const float* c,
                         const float* d, float* p, float* q);
HWYBLAS_API void dladiv_(const double* a, const double* b, const double* c,
                         const double* d, double* p, double* q);

/* The last non-zero column (ila?lc) or row (ila?lr) of the m x n matrix
 * A, or 0 if A is zero. */
HWYBLAS_API hwyblas_int ilaslc_(const hwyblas_int* m, const hwyblas_int* n,
                                const float* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int iladlc_(const hwyblas_int* m, const hwyblas_int* n,
                                const double* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int ilaclc_(const hwyblas_int* m, const hwyblas_int* n,
                                const float* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int ilazlc_(const hwyblas_int* m, const hwyblas_int* n,
                                const double* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int ilaslr_(const hwyblas_int* m, const hwyblas_int* n,
                                const float* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int iladlr_(const hwyblas_int* m, const hwyblas_int* n,
                                const double* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int ilaclr_(const hwyblas_int* m, const hwyblas_int* n,
                                const float* a, const hwyblas_int* lda);
HWYBLAS_API hwyblas_int ilazlr_(const hwyblas_int* m, const hwyblas_int* n,
                                const double* a, const hwyblas_int* lda);

/* ------------------------------------------------------------------------ */
/* Householder reflectors                                                   */

/* Generates an elementary reflector H = I - tau * v * v^H of order n with
 * H^H * (alpha; x) = (beta; 0) and real beta: alpha is overwritten with
 * beta and x (n - 1 elements) with v(2:n); v(1) = 1. */
HWYBLAS_API void slarfg_(const hwyblas_int* n, float* alpha, float* x,
                         const hwyblas_int* incx, float* tau);
HWYBLAS_API void dlarfg_(const hwyblas_int* n, double* alpha, double* x,
                         const hwyblas_int* incx, double* tau);
HWYBLAS_API void clarfg_(const hwyblas_int* n, float* alpha, float* x,
                         const hwyblas_int* incx, float* tau);
HWYBLAS_API void zlarfg_(const hwyblas_int* n, double* alpha, double* x,
                         const hwyblas_int* incx, double* tau);

/* Applies H = I - tau * v * v^H to the m x n matrix C from the left
 * (side = 'L') or the right ('R'); work has n ('L') or m ('R')
 * elements. ?larf1f takes v(1) = 1 and ?larf1l the last element = 1,
 * without reading it. */
HWYBLAS_API void slarf_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const float* v,
                        const hwyblas_int* incv, const float* tau, float* c,
                        const hwyblas_int* ldc, float* work);
HWYBLAS_API void dlarf_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const double* v,
                        const hwyblas_int* incv, const double* tau, double* c,
                        const hwyblas_int* ldc, double* work);
HWYBLAS_API void clarf_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const float* v,
                        const hwyblas_int* incv, const float* tau, float* c,
                        const hwyblas_int* ldc, float* work);
HWYBLAS_API void zlarf_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const double* v,
                        const hwyblas_int* incv, const double* tau, double* c,
                        const hwyblas_int* ldc, double* work);
HWYBLAS_API void slarf1f_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const float* v,
                          const hwyblas_int* incv, const float* tau, float* c,
                          const hwyblas_int* ldc, float* work);
HWYBLAS_API void dlarf1f_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const double* v,
                          const hwyblas_int* incv, const double* tau, double* c,
                          const hwyblas_int* ldc, double* work);
HWYBLAS_API void clarf1f_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const float* v,
                          const hwyblas_int* incv, const float* tau, float* c,
                          const hwyblas_int* ldc, float* work);
HWYBLAS_API void zlarf1f_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const double* v,
                          const hwyblas_int* incv, const double* tau, double* c,
                          const hwyblas_int* ldc, double* work);
HWYBLAS_API void slarf1l_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const float* v,
                          const hwyblas_int* incv, const float* tau, float* c,
                          const hwyblas_int* ldc, float* work);
HWYBLAS_API void dlarf1l_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const double* v,
                          const hwyblas_int* incv, const double* tau, double* c,
                          const hwyblas_int* ldc, double* work);
HWYBLAS_API void clarf1l_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const float* v,
                          const hwyblas_int* incv, const float* tau, float* c,
                          const hwyblas_int* ldc, float* work);
HWYBLAS_API void zlarf1l_(const char* side, const hwyblas_int* m,
                          const hwyblas_int* n, const double* v,
                          const hwyblas_int* incv, const double* tau, double* c,
                          const hwyblas_int* ldc, double* work);

/* Forms the triangular factor T of the block reflector of k reflectors of
 * order n, stored in the columns (storev = 'C') or rows ('R') of V, in
 * forward ('F') or backward ('B') order (direct). */
HWYBLAS_API void slarft_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const float* v, const hwyblas_int* ldv,
                         const float* tau, float* t, const hwyblas_int* ldt);
HWYBLAS_API void dlarft_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const double* v, const hwyblas_int* ldv,
                         const double* tau, double* t, const hwyblas_int* ldt);
HWYBLAS_API void clarft_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const float* v, const hwyblas_int* ldv,
                         const float* tau, float* t, const hwyblas_int* ldt);
HWYBLAS_API void zlarft_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const double* v, const hwyblas_int* ldv,
                         const double* tau, double* t, const hwyblas_int* ldt);

/* Applies the block reflector H of ?larft (trans = 'N') or H^H ('C', or
 * 'T' for real) to the m x n matrix C from the left (side = 'L') or the
 * right ('R'); work is ldwork x k. */
HWYBLAS_API void slarfb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* v,
                         const hwyblas_int* ldv, const float* t,
                         const hwyblas_int* ldt, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void dlarfb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* v,
                         const hwyblas_int* ldv, const double* t,
                         const hwyblas_int* ldt, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void clarfb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* v,
                         const hwyblas_int* ldv, const float* t,
                         const hwyblas_int* ldt, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void zlarfb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* v,
                         const hwyblas_int* ldv, const double* t,
                         const hwyblas_int* ldt, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* ldwork);

/* ------------------------------------------------------------------------ */
/* Cholesky factorization                                                   */

/* A = U^H * U (uplo = 'U') or A = L * L^H ('L') for symmetric (Hermitian)
 * positive definite A of order n; only the triangle uplo is referenced
 * and overwritten with the factor. info > 0: the leading minor of order
 * info is not positive. ?potrf is blocked, ?potrf2 recursive. */
HWYBLAS_API void spotrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void dpotrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void cpotrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void zpotrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void spotrf2_(const char* uplo, const hwyblas_int* n, float* a,
                          const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void dpotrf2_(const char* uplo, const hwyblas_int* n, double* a,
                          const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void cpotrf2_(const char* uplo, const hwyblas_int* n, float* a,
                          const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void zpotrf2_(const char* uplo, const hwyblas_int* n, double* a,
                          const hwyblas_int* lda, hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* QR and LQ factorizations                                                 */

/* A = Q * R (?geqrf) or A = L * Q (?gelqf) for the m x n matrix A. Q is
 * returned as min(m, n) reflectors, in A below (QR) or right of (LQ)
 * the diagonal, and in tau. lwork >= max(1, n) (QR) or max(1, m) (LQ),
 * or -1 for a workspace query (the optimal lwork in work[0]). ?geqr2
 * and ?gelq2 are unblocked; their work has n (QR) or m (LQ) elements. */
HWYBLAS_API void sgeqrf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgeqrf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgeqrf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgeqrf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sgeqr2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dgeqr2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cgeqr2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zgeqr2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);
HWYBLAS_API void sgelqf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgelqf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgelqf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgelqf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sgelq2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dgelq2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cgelq2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zgelq2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Orthogonal / unitary matrices of QR and LQ factorizations                */

/* Generate the m x n matrix Q with orthonormal columns (?orgqr, ?ungqr;
 * m >= n >= k) or rows (?orglq, ?unglq; n >= m >= k) from the k
 * reflectors of ?geqrf or ?gelqf. lwork >= max(1, n) (QR) or max(1, m)
 * (LQ), or -1 for a workspace query. ?org2r etc. are unblocked. */
HWYBLAS_API void sorgqr_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dorgqr_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cungqr_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zungqr_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sorg2r_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work, hwyblas_int* info);
HWYBLAS_API void dorg2r_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, hwyblas_int* info);
HWYBLAS_API void cung2r_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work, hwyblas_int* info);
HWYBLAS_API void zung2r_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, hwyblas_int* info);
HWYBLAS_API void sorglq_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dorglq_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cunglq_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zunglq_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sorgl2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work, hwyblas_int* info);
HWYBLAS_API void dorgl2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, hwyblas_int* info);
HWYBLAS_API void cungl2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work, hwyblas_int* info);
HWYBLAS_API void zungl2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, hwyblas_int* info);

/* C := Q * C, Q^H * C (side = 'L'), C * Q or C * Q^H ('R'), with
 * trans = 'N', or 'T' (real) or 'C' (complex) for Q^H, for the Q of the
 * k reflectors of ?geqrf (?ormqr, ?unmqr) or ?gelqf (?ormlq, ?unmlq).
 * lwork >= max(1, n) (side = 'L') or max(1, m) ('R'), or -1 for a
 * workspace query. ?orm2r etc. are unblocked. The complex LQ routines
 * conjugate rows of A while they run, and restore them on exit. */
HWYBLAS_API void sormqr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dormqr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cunmqr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zunmqr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sorm2r_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dorm2r_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cunm2r_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zunm2r_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void sormlq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dormlq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cunmlq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zunmlq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sorml2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dorml2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cunml2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zunml2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Norms and scaling                                                        */

/* Norms of matrices: norm = 'M' (max |a(i, j)|), '1' or 'O' (maximum
 * column sum), 'I' (maximum row sum), 'F' or 'E' (Frobenius). ?lange:
 * m x n general A (work: m elements, for 'I'); ?lansy: symmetric, ?lanhe:
 * Hermitian A of order n, triangle uplo (work: n elements, for 'I', 'O',
 * '1'). */
HWYBLAS_API float slange_(const char* norm, const hwyblas_int* m,
                          const hwyblas_int* n, const float* a,
                          const hwyblas_int* lda, float* work);
HWYBLAS_API double dlange_(const char* norm, const hwyblas_int* m,
                           const hwyblas_int* n, const double* a,
                           const hwyblas_int* lda, double* work);
HWYBLAS_API float clange_(const char* norm, const hwyblas_int* m,
                          const hwyblas_int* n, const float* a,
                          const hwyblas_int* lda, float* work);
HWYBLAS_API double zlange_(const char* norm, const hwyblas_int* m,
                           const hwyblas_int* n, const double* a,
                           const hwyblas_int* lda, double* work);
HWYBLAS_API float slansy_(const char* norm, const char* uplo,
                          const hwyblas_int* n, const float* a,
                          const hwyblas_int* lda, float* work);
HWYBLAS_API double dlansy_(const char* norm, const char* uplo,
                           const hwyblas_int* n, const double* a,
                           const hwyblas_int* lda, double* work);
HWYBLAS_API float clansy_(const char* norm, const char* uplo,
                          const hwyblas_int* n, const float* a,
                          const hwyblas_int* lda, float* work);
HWYBLAS_API double zlansy_(const char* norm, const char* uplo,
                           const hwyblas_int* n, const double* a,
                           const hwyblas_int* lda, double* work);
HWYBLAS_API float clanhe_(const char* norm, const char* uplo,
                          const hwyblas_int* n, const float* a,
                          const hwyblas_int* lda, float* work);
HWYBLAS_API double zlanhe_(const char* norm, const char* uplo,
                           const hwyblas_int* n, const double* a,
                           const hwyblas_int* lda, double* work);

/* Updates scale and sumsq so that scale^2 * sumsq = sum |x(i)|^2 + (the
 * previous scale^2 * sumsq), without unnecessary overflow and underflow. */
HWYBLAS_API void slassq_(const hwyblas_int* n, const float* x,
                         const hwyblas_int* incx, float* scale, float* sumsq);
HWYBLAS_API void dlassq_(const hwyblas_int* n, const double* x,
                         const hwyblas_int* incx, double* scale, double* sumsq);
HWYBLAS_API void classq_(const hwyblas_int* n, const float* x,
                         const hwyblas_int* incx, float* scale, float* sumsq);
HWYBLAS_API void zlassq_(const hwyblas_int* n, const double* x,
                         const hwyblas_int* incx, double* scale, double* sumsq);

/* A := A * (cto / cfrom) without over- or underflow; type = 'G' (full),
 * 'L' / 'U' (triangle), 'H' (upper Hessenberg), 'B' / 'Q' (lower / upper
 * half of a symmetric band matrix), 'Z' (band matrix of ?gbtrf). */
HWYBLAS_API void slascl_(const char* type, const hwyblas_int* kl,
                         const hwyblas_int* ku, const float* cfrom,
                         const float* cto, const hwyblas_int* m,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* info);
HWYBLAS_API void dlascl_(const char* type, const hwyblas_int* kl,
                         const hwyblas_int* ku, const double* cfrom,
                         const double* cto, const hwyblas_int* m,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void clascl_(const char* type, const hwyblas_int* kl,
                         const hwyblas_int* ku, const float* cfrom,
                         const float* cto, const hwyblas_int* m,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* info);
HWYBLAS_API void zlascl_(const char* type, const hwyblas_int* kl,
                         const hwyblas_int* ku, const double* cfrom,
                         const double* cto, const hwyblas_int* m,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);

/* Sets the off-diagonal elements of A (the strictly upper ('U') or lower
 * ('L') triangle, or all) to alpha and the diagonal to beta. */
HWYBLAS_API void slaset_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const float* alpha,
                         const float* beta, float* a, const hwyblas_int* lda);
HWYBLAS_API void dlaset_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const double* alpha,
                         const double* beta, double* a, const hwyblas_int* lda);
HWYBLAS_API void claset_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const float* alpha,
                         const float* beta, float* a, const hwyblas_int* lda);
HWYBLAS_API void zlaset_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const double* alpha,
                         const double* beta, double* a, const hwyblas_int* lda);

/* ------------------------------------------------------------------------ */
/* QR factorization with column pivoting                                    */

/* A * P = Q * R. On entry, jpvt(j) != 0 makes column j a leading column
 * (moved to the front), 0 a free one; on exit, jpvt(j) = k if column j
 * of A * P was column k of A. lwork >= 3 * n + 1 (real) or n + 1
 * (complex), or -1 for a workspace query; rwork has 2 * n elements.
 * ?laqps (returning kb columns factored) and ?laqp2 factor a block
 * of columns. */
HWYBLAS_API void sgeqp3_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, float* tau,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dgeqp3_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cgeqp3_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, float* tau,
                         float* work, const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgeqp3_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, double* tau,
                         double* work, const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* info);
HWYBLAS_API void slaqps_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, const hwyblas_int* nb,
                         hwyblas_int* kb, float* a, const hwyblas_int* lda,
                         hwyblas_int* jpvt, float* tau, float* vn1, float* vn2,
                         float* auxv, float* f, const hwyblas_int* ldf);
HWYBLAS_API void dlaqps_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, const hwyblas_int* nb,
                         hwyblas_int* kb, double* a, const hwyblas_int* lda,
                         hwyblas_int* jpvt, double* tau, double* vn1,
                         double* vn2, double* auxv, double* f,
                         const hwyblas_int* ldf);
HWYBLAS_API void slaqp2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, float* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, float* tau,
                         float* vn1, float* vn2, float* work);
HWYBLAS_API void dlaqp2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, double* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, double* tau,
                         double* vn1, double* vn2, double* work);
HWYBLAS_API void claqps_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, const hwyblas_int* nb,
                         hwyblas_int* kb, float* a, const hwyblas_int* lda,
                         hwyblas_int* jpvt, float* tau, float* vn1, float* vn2,
                         float* auxv, float* f, const hwyblas_int* ldf);
HWYBLAS_API void zlaqps_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, const hwyblas_int* nb,
                         hwyblas_int* kb, double* a, const hwyblas_int* lda,
                         hwyblas_int* jpvt, double* tau, double* vn1,
                         double* vn2, double* auxv, double* f,
                         const hwyblas_int* ldf);
HWYBLAS_API void claqp2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, float* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, float* tau,
                         float* vn1, float* vn2, float* work);
HWYBLAS_API void zlaqp2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* offset, double* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, double* tau,
                         double* vn1, double* vn2, double* work);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) eigenproblems                                      */

/* All eigenvalues (w, ascending) and, for jobz = 'V', the orthonormal
 * eigenvectors (overwriting A) of the symmetric (?syev) or Hermitian
 * (?heev) matrix A of order n, triangle uplo. lwork >= max(1, 3*n-1)
 * (real) or max(1, 2*n-1) (complex), or -1 for a workspace query;
 * rwork: max(1, 3*n-2) elements. info > 0: no convergence. */
HWYBLAS_API void ssyev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* w, float* work, const hwyblas_int* lwork,
                        hwyblas_int* info);
HWYBLAS_API void dsyev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* w, double* work, const hwyblas_int* lwork,
                        hwyblas_int* info);
HWYBLAS_API void cheev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* w, float* work, const hwyblas_int* lwork,
                        float* rwork, hwyblas_int* info);
HWYBLAS_API void zheev_(const char* jobz, const char* uplo,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* w, double* work, const hwyblas_int* lwork,
                        double* rwork, hwyblas_int* info);

/* Reduction to real symmetric tridiagonal form T = Q^H * A * Q (d: the
 * diagonal, e: the off-diagonal, Q as n-1 reflectors in A and tau).
 * ?sytrd / ?hetrd: blocked, lwork >= 1 or -1 (query); ?sytd2 / ?hetd2:
 * unblocked; ?latrd: one block of nb columns, with the n x nb matrix W
 * for the update A := A - V * W^H - W * V^H. */
HWYBLAS_API void ssytrd_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e, float* tau,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dsytrd_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tau, double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void chetrd_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e, float* tau,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zhetrd_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tau, double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void ssytd2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e, float* tau,
                         hwyblas_int* info);
HWYBLAS_API void dsytd2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tau, hwyblas_int* info);
HWYBLAS_API void chetd2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e, float* tau,
                         hwyblas_int* info);
HWYBLAS_API void zhetd2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tau, hwyblas_int* info);
HWYBLAS_API void slatrd_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, float* a,
                         const hwyblas_int* lda, float* e, float* tau, float* w,
                         const hwyblas_int* ldw);
HWYBLAS_API void dlatrd_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, double* a,
                         const hwyblas_int* lda, double* e, double* tau,
                         double* w, const hwyblas_int* ldw);
HWYBLAS_API void clatrd_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, float* a,
                         const hwyblas_int* lda, float* e, float* tau, float* w,
                         const hwyblas_int* ldw);
HWYBLAS_API void zlatrd_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, double* a,
                         const hwyblas_int* lda, double* e, double* tau,
                         double* w, const hwyblas_int* ldw);

/* The n x n matrix Q of ?sytrd / ?hetrd (same uplo). lwork >=
 * max(1, n-1), or -1 for a workspace query. */
HWYBLAS_API void sorgtr_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dorgtr_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cungtr_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zungtr_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);

/* The m x n matrix Q with orthonormal columns (the last n columns of k
 * reflectors of a QL factorization). lwork >= max(1, n), or -1 for a
 * workspace query. ?org2l / ?ung2l are unblocked. */
HWYBLAS_API void sorgql_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dorgql_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cungql_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zungql_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sorg2l_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work, hwyblas_int* info);
HWYBLAS_API void dorg2l_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, hwyblas_int* info);
HWYBLAS_API void cung2l_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* work, hwyblas_int* info);
HWYBLAS_API void zung2l_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, hwyblas_int* info);

/* Eigenvalues (d, ascending) and, for compz = 'V' or 'I', eigenvectors
 * (z: those of the original matrix for 'V', of the tridiagonal matrix
 * for 'I') of the symmetric tridiagonal matrix (d, e) with the implicit
 * QL / QR method; work: max(1, 2*n-2) real elements. ?sterf:
 * eigenvalues only, root-free. */
HWYBLAS_API void ssteqr_(const char* compz, const hwyblas_int* n, float* d,
                         float* e, float* z, const hwyblas_int* ldz,
                         float* work, hwyblas_int* info);
HWYBLAS_API void dsteqr_(const char* compz, const hwyblas_int* n, double* d,
                         double* e, double* z, const hwyblas_int* ldz,
                         double* work, hwyblas_int* info);
HWYBLAS_API void csteqr_(const char* compz, const hwyblas_int* n, float* d,
                         float* e, float* z, const hwyblas_int* ldz,
                         float* work, hwyblas_int* info);
HWYBLAS_API void zsteqr_(const char* compz, const hwyblas_int* n, double* d,
                         double* e, double* z, const hwyblas_int* ldz,
                         double* work, hwyblas_int* info);
HWYBLAS_API void ssterf_(const hwyblas_int* n, float* d, float* e,
                         hwyblas_int* info);
HWYBLAS_API void dsterf_(const hwyblas_int* n, double* d, double* e,
                         hwyblas_int* info);

/* Applies the plane rotations (c(k), s(k)) (real, also for complex A) to
 * the m x n matrix A from the left (side = 'L') or right ('R'), in the
 * planes (k, k+1) (pivot = 'V'), (1, k+1) ('T') or (k, z) ('B'), in
 * forward ('F') or backward ('B') order. */
HWYBLAS_API void slasr_(const char* side, const char* pivot, const char* direct,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const float* c, const float* s, float* a,
                        const hwyblas_int* lda);
HWYBLAS_API void dlasr_(const char* side, const char* pivot, const char* direct,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const double* c, const double* s, double* a,
                        const hwyblas_int* lda);
HWYBLAS_API void clasr_(const char* side, const char* pivot, const char* direct,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const float* c, const float* s, float* a,
                        const hwyblas_int* lda);
HWYBLAS_API void zlasr_(const char* side, const char* pivot, const char* direct,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const double* c, const double* s, double* a,
                        const hwyblas_int* lda);

/* Eigenvalues of [a b; b c] (rt1 of larger absolute value), and with
 * ?laev2 the eigenvector (cs1, sn1) of rt1; ?lartg: a plane rotation
 * with [c s; -s c] * [f; g] = [r; 0]; ?lanst: the norm of the symmetric
 * tridiagonal matrix (d, e); ?lasrt: sorts d in increasing ('I') or
 * decreasing ('D') order. */
HWYBLAS_API void slae2_(const float* a, const float* b, const float* c,
                        float* rt1, float* rt2);
HWYBLAS_API void dlae2_(const double* a, const double* b, const double* c,
                        double* rt1, double* rt2);
HWYBLAS_API void slaev2_(const float* a, const float* b, const float* c,
                         float* rt1, float* rt2, float* cs1, float* sn1);
HWYBLAS_API void dlaev2_(const double* a, const double* b, const double* c,
                         double* rt1, double* rt2, double* cs1, double* sn1);
HWYBLAS_API void slartg_(const float* f, const float* g, float* c, float* s,
                         float* r);
HWYBLAS_API void dlartg_(const double* f, const double* g, double* c, double* s,
                         double* r);
HWYBLAS_API float slanst_(const char* norm, const hwyblas_int* n,
                          const float* d, const float* e);
HWYBLAS_API double dlanst_(const char* norm, const hwyblas_int* n,
                           const double* d, const double* e);
HWYBLAS_API void slasrt_(const char* id, const hwyblas_int* n, float* d,
                         hwyblas_int* info);
HWYBLAS_API void dlasrt_(const char* id, const hwyblas_int* n, double* d,
                         hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Singular value decomposition                                             */

/* The singular values (s, decreasing) and, for jobu / jobvt = 'A', 'S'
 * or 'O', all, the first min(m, n) or those (overwriting A) left
 * singular vectors (U) / right singular vectors (VT = V^H) of the m x n
 * matrix A = U * SIGMA * V^H (jobu and jobvt not both 'O'; 'N': none).
 * lwork >= max(1, 3*min(m,n) + max(m,n), 5*min(m,n)) (real) or max(1,
 * 2*min(m,n) + max(m,n)) (complex), or -1 for a workspace query; rwork:
 * 5*min(m,n) elements. info > 0: ?bdsqr did not converge. */
HWYBLAS_API void sgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* s, float* u,
                         const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* s, float* u,
                         const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgesvd_(const char* jobu, const char* jobvt,
                         const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* info);

/* Reduction to real bidiagonal form B = Q^H * A * P (upper for m >= n,
 * lower otherwise; d: the diagonal, e: the off-diagonal; Q and P as
 * reflectors in A, tauq and taup). ?gebrd: blocked, lwork >= max(1, m,
 * n) or -1 (query); ?gebd2: unblocked; ?labrd: one block of nb rows and
 * columns, with the matrices X and Y for the update A := A - V * Y^H -
 * X * U^H. */
HWYBLAS_API void sgebrd_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e,
                         float* tauq, float* taup, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgebrd_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tauq, double* taup, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgebrd_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e,
                         float* tauq, float* taup, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgebrd_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tauq, double* taup, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sgebd2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e,
                         float* tauq, float* taup, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dgebd2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tauq, double* taup, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cgebd2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* d, float* e,
                         float* tauq, float* taup, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zgebd2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tauq, double* taup, double* work,
                         hwyblas_int* info);
HWYBLAS_API void slabrd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nb, float* a,
                         const hwyblas_int* lda, float* d, float* e,
                         float* tauq, float* taup, float* x,
                         const hwyblas_int* ldx, float* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void dlabrd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nb, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tauq, double* taup, double* x,
                         const hwyblas_int* ldx, double* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void clabrd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nb, float* a,
                         const hwyblas_int* lda, float* d, float* e,
                         float* tauq, float* taup, float* x,
                         const hwyblas_int* ldx, float* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void zlabrd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nb, double* a,
                         const hwyblas_int* lda, double* d, double* e,
                         double* tauq, double* taup, double* x,
                         const hwyblas_int* ldx, double* y,
                         const hwyblas_int* ldy);

/* Q (vect = 'Q') or P^H ('P') of ?gebrd: ?orgbr / ?ungbr generates it
 * (lwork >= max(1, min(m, n))), ?ormbr / ?unmbr applies it to C from
 * the left or right, transposed or not (lwork >= max(1, n) for side =
 * 'L', max(1, m) for 'R'); lwork = -1 is a workspace query. */
HWYBLAS_API void sorgbr_(const char* vect, const hwyblas_int* m,
                         const hwyblas_int* n, const hwyblas_int* k, float* a,
                         const hwyblas_int* lda, const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dorgbr_(const char* vect, const hwyblas_int* m,
                         const hwyblas_int* n, const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cungbr_(const char* vect, const hwyblas_int* m,
                         const hwyblas_int* n, const hwyblas_int* k, float* a,
                         const hwyblas_int* lda, const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zungbr_(const char* vect, const hwyblas_int* m,
                         const hwyblas_int* n, const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sormbr_(const char* vect, const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dormbr_(const char* vect, const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cunmbr_(const char* vect, const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zunmbr_(const char* vect, const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);

/* The SVD B = Q * S * P^H of the real n x n upper ('U') or lower ('L')
 * bidiagonal matrix (d, e) with the implicit zero-shift QR algorithm:
 * d returns the singular values (decreasing); VT := P^H * VT (n x
 * ncvt), U := U * Q (nru x n), C := Q^H * C (n x ncc); work (rwork for
 * complex): 4*(n-1) real elements. info > 0: no convergence. */
HWYBLAS_API void sbdsqr_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* ncvt, const hwyblas_int* nru,
                         const hwyblas_int* ncc, float* d, float* e, float* vt,
                         const hwyblas_int* ldvt, float* u,
                         const hwyblas_int* ldu, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dbdsqr_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* ncvt, const hwyblas_int* nru,
                         const hwyblas_int* ncc, double* d, double* e,
                         double* vt, const hwyblas_int* ldvt, double* u,
                         const hwyblas_int* ldu, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cbdsqr_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* ncvt, const hwyblas_int* nru,
                         const hwyblas_int* ncc, float* d, float* e, float* vt,
                         const hwyblas_int* ldvt, float* u,
                         const hwyblas_int* ldu, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zbdsqr_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* ncvt, const hwyblas_int* nru,
                         const hwyblas_int* ncc, double* d, double* e,
                         double* vt, const hwyblas_int* ldvt, double* u,
                         const hwyblas_int* ldu, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);

/* The singular values of [f g; 0 h] (?las2), and with ?lasv2 the left
 * (csl, snl) and right (csr, snr) singular vectors of the larger one. */
HWYBLAS_API void slas2_(const float* f, const float* g, const float* h,
                        float* ssmin, float* ssmax);
HWYBLAS_API void dlas2_(const double* f, const double* g, const double* h,
                        double* ssmin, double* ssmax);
HWYBLAS_API void slasv2_(const float* f, const float* g, const float* h,
                         float* ssmin, float* ssmax, float* snr, float* csr,
                         float* snl, float* csl);
HWYBLAS_API void dlasv2_(const double* f, const double* g, const double* h,
                         double* ssmin, double* ssmax, double* snr, double* csr,
                         double* snl, double* csl);

/* The dqds algorithm: ?lasq1 computes the singular values of the
 * bidiagonal matrix (d, e) (work: 4*n elements), ?lasq2 the eigenvalues
 * of the qd array z (4*n elements); ?lasq3 - ?lasq6 are its steps (ieee
 * is a Fortran LOGICAL, 1 or 0). */
HWYBLAS_API void slasq1_(const hwyblas_int* n, float* d, float* e, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dlasq1_(const hwyblas_int* n, double* d, double* e,
                         double* work, hwyblas_int* info);
HWYBLAS_API void slasq2_(const hwyblas_int* n, float* z, hwyblas_int* info);
HWYBLAS_API void dlasq2_(const hwyblas_int* n, double* z, hwyblas_int* info);
HWYBLAS_API void slasq3_(const hwyblas_int* i0, hwyblas_int* n0, float* z,
                         hwyblas_int* pp, float* dmin, float* sigma,
                         float* desig, float* qmax, hwyblas_int* nfail,
                         hwyblas_int* iter, hwyblas_int* ndiv,
                         const hwyblas_int* ieee, hwyblas_int* ttype,
                         float* dmin1, float* dmin2, float* dn, float* dn1,
                         float* dn2, float* g, float* tau);
HWYBLAS_API void dlasq3_(const hwyblas_int* i0, hwyblas_int* n0, double* z,
                         hwyblas_int* pp, double* dmin, double* sigma,
                         double* desig, double* qmax, hwyblas_int* nfail,
                         hwyblas_int* iter, hwyblas_int* ndiv,
                         const hwyblas_int* ieee, hwyblas_int* ttype,
                         double* dmin1, double* dmin2, double* dn, double* dn1,
                         double* dn2, double* g, double* tau);
HWYBLAS_API void slasq4_(const hwyblas_int* i0, const hwyblas_int* n0,
                         const float* z, const hwyblas_int* pp,
                         const hwyblas_int* n0in, const float* dmin,
                         const float* dmin1, const float* dmin2,
                         const float* dn, const float* dn1, const float* dn2,
                         float* tau, hwyblas_int* ttype, float* g);
HWYBLAS_API void dlasq4_(const hwyblas_int* i0, const hwyblas_int* n0,
                         const double* z, const hwyblas_int* pp,
                         const hwyblas_int* n0in, const double* dmin,
                         const double* dmin1, const double* dmin2,
                         const double* dn, const double* dn1, const double* dn2,
                         double* tau, hwyblas_int* ttype, double* g);
HWYBLAS_API void slasq5_(const hwyblas_int* i0, const hwyblas_int* n0, float* z,
                         const hwyblas_int* pp, float* tau, const float* sigma,
                         float* dmin, float* dmin1, float* dmin2, float* dn,
                         float* dnm1, float* dnm2, const hwyblas_int* ieee,
                         const float* eps);
HWYBLAS_API void dlasq5_(const hwyblas_int* i0, const hwyblas_int* n0,
                         double* z, const hwyblas_int* pp, double* tau,
                         const double* sigma, double* dmin, double* dmin1,
                         double* dmin2, double* dn, double* dnm1, double* dnm2,
                         const hwyblas_int* ieee, const double* eps);
HWYBLAS_API void slasq6_(const hwyblas_int* i0, const hwyblas_int* n0, float* z,
                         const hwyblas_int* pp, float* dmin, float* dmin1,
                         float* dmin2, float* dn, float* dnm1, float* dnm2);
HWYBLAS_API void dlasq6_(const hwyblas_int* i0, const hwyblas_int* n0,
                         double* z, const hwyblas_int* pp, double* dmin,
                         double* dmin1, double* dmin2, double* dn, double* dnm1,
                         double* dnm2);

/* ------------------------------------------------------------------------ */
/* Singular value decomposition by divide and conquer                       */

/* The singular values (s, decreasing) and, for jobz = 'A', 'S' or 'O',
 * all, the first min(m, n) or those left and right singular vectors
 * (U, VT = V^H; for 'O' the first min(m, n) overwrite A) of the m x n
 * matrix A = U * SIGMA * V^H by divide and conquer ('N': none). See
 * LAPACK's ?gesdd for the sizes of work (lwork = -1: query) and rwork;
 * iwork: 8*min(m,n) elements. info = -4: A has a NaN; info > 0: ?bdsdc
 * did not converge. */
HWYBLAS_API void sgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* s, float* u, const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void cgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* s, float* u, const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zgesdd_(const char* jobz, const hwyblas_int* m,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* s, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);

/* The SVD B = U * S * VT of the real n x n upper ('U') or lower ('L')
 * bidiagonal matrix (d, e) by divide and conquer: d returns the
 * singular values (decreasing); compq = 'N': values only, 'P': also
 * the vectors in compact form (q, iq), 'I': U and VT. work: 4*n ('N'),
 * 6*n ('P') or 3*n^2 + 4*n ('I') elements; iwork: 8*n. */
HWYBLAS_API void sbdsdc_(const char* uplo, const char* compq,
                         const hwyblas_int* n, float* d, float* e, float* u,
                         const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* q, hwyblas_int* iq,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dbdsdc_(const char* uplo, const char* compq,
                         const hwyblas_int* n, double* d, double* e, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* q, hwyblas_int* iq,
                         double* work, hwyblas_int* iwork, hwyblas_int* info);

/* The steps of ?bdsdc: ?lasdq (the SVD of an n x (n+sqre) bidiagonal
 * matrix with ?bdsqr), ?lasd0 / ?lasda (divide and conquer with / without
 * the explicit singular vectors), ?lasd1 / ?lasd6 (merging two
 * subproblems), ?lasd2 / ?lasd7 (deflation), ?lasd3 / ?lasd8 (the secular
 * equation and the vectors), ?lasd4 / ?lasd5 (one root of the secular
 * equation), ?laed6 (one Newton step; orgati is a Fortran LOGICAL),
 * ?lasdt (the tree of subproblems) and ?lamrg (merging sorted lists). */
HWYBLAS_API void slasdq_(const char* uplo, const hwyblas_int* sqre,
                         const hwyblas_int* n, const hwyblas_int* ncvt,
                         const hwyblas_int* nru, const hwyblas_int* ncc,
                         float* d, float* e, float* vt, const hwyblas_int* ldvt,
                         float* u, const hwyblas_int* ldu, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dlasdq_(const char* uplo, const hwyblas_int* sqre,
                         const hwyblas_int* n, const hwyblas_int* ncvt,
                         const hwyblas_int* nru, const hwyblas_int* ncc,
                         double* d, double* e, double* vt,
                         const hwyblas_int* ldvt, double* u,
                         const hwyblas_int* ldu, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void slasd0_(const hwyblas_int* n, const hwyblas_int* sqre,
                         float* d, float* e, float* u, const hwyblas_int* ldu,
                         float* vt, const hwyblas_int* ldvt,
                         const hwyblas_int* smlsiz, hwyblas_int* iwork,
                         float* work, hwyblas_int* info);
HWYBLAS_API void dlasd0_(const hwyblas_int* n, const hwyblas_int* sqre,
                         double* d, double* e, double* u,
                         const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, const hwyblas_int* smlsiz,
                         hwyblas_int* iwork, double* work, hwyblas_int* info);
HWYBLAS_API void slasda_(const hwyblas_int* icompq, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* sqre,
                         float* d, float* e, float* u, const hwyblas_int* ldu,
                         float* vt, hwyblas_int* k, float* difl, float* difr,
                         float* z, float* poles, hwyblas_int* givptr,
                         hwyblas_int* givcol, const hwyblas_int* ldgcol,
                         hwyblas_int* perm, float* givnum, float* c, float* s,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlasda_(const hwyblas_int* icompq, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* sqre,
                         double* d, double* e, double* u,
                         const hwyblas_int* ldu, double* vt, hwyblas_int* k,
                         double* difl, double* difr, double* z, double* poles,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, hwyblas_int* perm,
                         double* givnum, double* c, double* s, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slasd1_(const hwyblas_int* nl, const hwyblas_int* nr,
                         const hwyblas_int* sqre, float* d, float* alpha,
                         float* beta, float* u, const hwyblas_int* ldu,
                         float* vt, const hwyblas_int* ldvt, hwyblas_int* idxq,
                         hwyblas_int* iwork, float* work, hwyblas_int* info);
HWYBLAS_API void dlasd1_(const hwyblas_int* nl, const hwyblas_int* nr,
                         const hwyblas_int* sqre, double* d, double* alpha,
                         double* beta, double* u, const hwyblas_int* ldu,
                         double* vt, const hwyblas_int* ldvt, hwyblas_int* idxq,
                         hwyblas_int* iwork, double* work, hwyblas_int* info);
HWYBLAS_API void slasd2_(const hwyblas_int* nl, const hwyblas_int* nr,
                         const hwyblas_int* sqre, hwyblas_int* k, float* d,
                         float* z, const float* alpha, const float* beta,
                         float* u, const hwyblas_int* ldu, float* vt,
                         const hwyblas_int* ldvt, float* dsigma, float* u2,
                         const hwyblas_int* ldu2, float* vt2,
                         const hwyblas_int* ldvt2, hwyblas_int* idxp,
                         hwyblas_int* idx, hwyblas_int* idxc, hwyblas_int* idxq,
                         hwyblas_int* coltyp, hwyblas_int* info);
HWYBLAS_API void dlasd2_(const hwyblas_int* nl, const hwyblas_int* nr,
                         const hwyblas_int* sqre, hwyblas_int* k, double* d,
                         double* z, const double* alpha, const double* beta,
                         double* u, const hwyblas_int* ldu, double* vt,
                         const hwyblas_int* ldvt, double* dsigma, double* u2,
                         const hwyblas_int* ldu2, double* vt2,
                         const hwyblas_int* ldvt2, hwyblas_int* idxp,
                         hwyblas_int* idx, hwyblas_int* idxc, hwyblas_int* idxq,
                         hwyblas_int* coltyp, hwyblas_int* info);
HWYBLAS_API void slasd3_(const hwyblas_int* nl, const hwyblas_int* nr,
                         const hwyblas_int* sqre, const hwyblas_int* k,
                         float* d, float* q, const hwyblas_int* ldq,
                         float* dsigma, float* u, const hwyblas_int* ldu,
                         float* u2, const hwyblas_int* ldu2, float* vt,
                         const hwyblas_int* ldvt, float* vt2,
                         const hwyblas_int* ldvt2, const hwyblas_int* idxc,
                         const hwyblas_int* ctot, float* z, hwyblas_int* info);
HWYBLAS_API void dlasd3_(const hwyblas_int* nl, const hwyblas_int* nr,
                         const hwyblas_int* sqre, const hwyblas_int* k,
                         double* d, double* q, const hwyblas_int* ldq,
                         double* dsigma, double* u, const hwyblas_int* ldu,
                         double* u2, const hwyblas_int* ldu2, double* vt,
                         const hwyblas_int* ldvt, double* vt2,
                         const hwyblas_int* ldvt2, const hwyblas_int* idxc,
                         const hwyblas_int* ctot, double* z, hwyblas_int* info);
HWYBLAS_API void slasd4_(const hwyblas_int* n, const hwyblas_int* i,
                         const float* d, const float* z, float* delta,
                         const float* rho, float* sigma, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dlasd4_(const hwyblas_int* n, const hwyblas_int* i,
                         const double* d, const double* z, double* delta,
                         const double* rho, double* sigma, double* work,
                         hwyblas_int* info);
HWYBLAS_API void slasd5_(const hwyblas_int* i, const float* d, const float* z,
                         float* delta, const float* rho, float* dsigma,
                         float* work);
HWYBLAS_API void dlasd5_(const hwyblas_int* i, const double* d, const double* z,
                         double* delta, const double* rho, double* dsigma,
                         double* work);
HWYBLAS_API void slasd6_(const hwyblas_int* icompq, const hwyblas_int* nl,
                         const hwyblas_int* nr, const hwyblas_int* sqre,
                         float* d, float* vf, float* vl, float* alpha,
                         float* beta, hwyblas_int* idxq, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, float* givnum,
                         const hwyblas_int* ldgnum, float* poles, float* difl,
                         float* difr, float* z, hwyblas_int* k, float* c,
                         float* s, float* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dlasd6_(const hwyblas_int* icompq, const hwyblas_int* nl,
                         const hwyblas_int* nr, const hwyblas_int* sqre,
                         double* d, double* vf, double* vl, double* alpha,
                         double* beta, hwyblas_int* idxq, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, double* givnum,
                         const hwyblas_int* ldgnum, double* poles, double* difl,
                         double* difr, double* z, hwyblas_int* k, double* c,
                         double* s, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void slasd7_(const hwyblas_int* icompq, const hwyblas_int* nl,
                         const hwyblas_int* nr, const hwyblas_int* sqre,
                         hwyblas_int* k, float* d, float* z, float* zw,
                         float* vf, float* vfw, float* vl, float* vlw,
                         const float* alpha, const float* beta, float* dsigma,
                         hwyblas_int* idx, hwyblas_int* idxp, hwyblas_int* idxq,
                         hwyblas_int* perm, hwyblas_int* givptr,
                         hwyblas_int* givcol, const hwyblas_int* ldgcol,
                         float* givnum, const hwyblas_int* ldgnum, float* c,
                         float* s, hwyblas_int* info);
HWYBLAS_API void dlasd7_(
    const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr,
    const hwyblas_int* sqre, hwyblas_int* k, double* d, double* z, double* zw,
    double* vf, double* vfw, double* vl, double* vlw, const double* alpha,
    const double* beta, double* dsigma, hwyblas_int* idx, hwyblas_int* idxp,
    hwyblas_int* idxq, hwyblas_int* perm, hwyblas_int* givptr,
    hwyblas_int* givcol, const hwyblas_int* ldgcol, double* givnum,
    const hwyblas_int* ldgnum, double* c, double* s, hwyblas_int* info);
HWYBLAS_API void slasd8_(const hwyblas_int* icompq, const hwyblas_int* k,
                         float* d, float* z, float* vf, float* vl, float* difl,
                         float* difr, const hwyblas_int* lddifr, float* dsigma,
                         float* work, hwyblas_int* info);
HWYBLAS_API void dlasd8_(const hwyblas_int* icompq, const hwyblas_int* k,
                         double* d, double* z, double* vf, double* vl,
                         double* difl, double* difr, const hwyblas_int* lddifr,
                         double* dsigma, double* work, hwyblas_int* info);
HWYBLAS_API void slaed6_(const hwyblas_int* kniter, const hwyblas_int* orgati,
                         const float* rho, const float* d, const float* z,
                         const float* finit, float* tau, hwyblas_int* info);
HWYBLAS_API void dlaed6_(const hwyblas_int* kniter, const hwyblas_int* orgati,
                         const double* rho, const double* d, const double* z,
                         const double* finit, double* tau, hwyblas_int* info);
HWYBLAS_API void slasdt_(const hwyblas_int* n, hwyblas_int* lvl,
                         hwyblas_int* nd, hwyblas_int* inode,
                         hwyblas_int* ndiml, hwyblas_int* ndimr,
                         const hwyblas_int* msub);
HWYBLAS_API void dlasdt_(const hwyblas_int* n, hwyblas_int* lvl,
                         hwyblas_int* nd, hwyblas_int* inode,
                         hwyblas_int* ndiml, hwyblas_int* ndimr,
                         const hwyblas_int* msub);
HWYBLAS_API void slamrg_(const hwyblas_int* n1, const hwyblas_int* n2,
                         const float* a, const hwyblas_int* dtrd1,
                         const hwyblas_int* dtrd2, hwyblas_int* index);
HWYBLAS_API void dlamrg_(const hwyblas_int* n1, const hwyblas_int* n2,
                         const double* a, const hwyblas_int* dtrd1,
                         const hwyblas_int* dtrd2, hwyblas_int* index);

/* Complex matrices with real ones: B := A for a real m x n A (?lacp2,
 * all or the upper / lower part like ?lacpy), C := A * B for a complex m
 * x n A and a real n x n B (?lacrm), C := A * B for a real m x m A and a
 * complex m x n B (?larcm); rwork: 2*m*n elements. */
HWYBLAS_API void clacp2_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void zlacp2_(const char* uplo, const hwyblas_int* m,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void clacrm_(const hwyblas_int* m, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, float* c,
                         const hwyblas_int* ldc, float* rwork);
HWYBLAS_API void zlacrm_(const hwyblas_int* m, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* b, const hwyblas_int* ldb, double* c,
                         const hwyblas_int* ldc, double* rwork);
HWYBLAS_API void clarcm_(const hwyblas_int* m, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, float* c,
                         const hwyblas_int* ldc, float* rwork);
HWYBLAS_API void zlarcm_(const hwyblas_int* m, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* b, const hwyblas_int* ldb, double* c,
                         const hwyblas_int* ldc, double* rwork);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACK_H_ */
