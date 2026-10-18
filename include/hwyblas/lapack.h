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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACK_H_ */
