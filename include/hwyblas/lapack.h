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

/* Solves op(A) * X = B (trans = 'N', 'T' or 'C') with the LU
 * factorization of the n x n matrix A by ?getrf (a and ipiv); X overwrites
 * the n x nrhs matrix B. */
HWYBLAS_API void sgetrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dgetrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cgetrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zgetrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);

/* Solves A * X = B with the LU factorization of the n x n matrix A
 * (?getrf), which overwrites A and ipiv; X overwrites B. info > 0:
 * U(info, info) is exactly zero, and X has not been computed. */
HWYBLAS_API void sgesv_(const hwyblas_int* n, const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dgesv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* a, const hwyblas_int* lda, hwyblas_int* ipiv,
                        double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cgesv_(const hwyblas_int* n, const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zgesv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* a, const hwyblas_int* lda, hwyblas_int* ipiv,
                        double* b, const hwyblas_int* ldb, hwyblas_int* info);

/* The inverse of A from its LU factorization by ?getrf (a and
 * ipiv), which it overwrites. lwork >= max(1, n) (optimal: n * the block
 * size; -1: workspace query). info > 0: U(info, info) is exactly zero. */
HWYBLAS_API void sgetri_(const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const hwyblas_int* ipiv, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgetri_(const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cgetri_(const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const hwyblas_int* ipiv, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgetri_(const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);

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

/* x := x / sa (real sa) without overflow or underflow if x / sa
 * does not: ?rscl (real x), csrscl / zdrscl (complex x). */
HWYBLAS_API void srscl_(const hwyblas_int* n, const float* sa, float* sx,
                        const hwyblas_int* incx);
HWYBLAS_API void drscl_(const hwyblas_int* n, const double* sa, double* sx,
                        const hwyblas_int* incx);
HWYBLAS_API void csrscl_(const hwyblas_int* n, const float* sa, float* sx,
                         const hwyblas_int* incx);
HWYBLAS_API void zdrscl_(const hwyblas_int* n, const double* sa, double* sx,
                         const hwyblas_int* incx);

/* Fortran LOGICAL (1 or 0): whether din is NaN (?isnan), whether
 * din1 != din2 (?laisnan). */
HWYBLAS_API hwyblas_int sisnan_(const float* din);
HWYBLAS_API hwyblas_int disnan_(const double* din);
HWYBLAS_API hwyblas_int slaisnan_(const float* din1, const float* din2);
HWYBLAS_API hwyblas_int dlaisnan_(const double* din1, const double* din2);

/* ------------------------------------------------------------------------ */
/* Triangular matrices                                                      */

/* The inverse of the upper ('U') or lower ('L') triangular n x n
 * matrix A, unit diagonal for diag = 'U', in place: ?trti2 unblocked, ?trtri
 * blocked. ?trtri: info > 0: A(info, info) is exactly zero (A is singular;
 * not inverted). */
HWYBLAS_API void strti2_(const char* uplo, const char* diag,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* info);
HWYBLAS_API void dtrti2_(const char* uplo, const char* diag,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void ctrti2_(const char* uplo, const char* diag,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* info);
HWYBLAS_API void ztrti2_(const char* uplo, const char* diag,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void strtri_(const char* uplo, const char* diag,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* info);
HWYBLAS_API void dtrtri_(const char* uplo, const char* diag,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void ctrtri_(const char* uplo, const char* diag,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* info);
HWYBLAS_API void ztrtri_(const char* uplo, const char* diag,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);

/* Solves op(A) * X = B (trans = 'N', 'T' or 'C') with the
 * triangular matrix A (uplo, diag); X overwrites the n x nrhs matrix B.
 * info > 0: A(info, info) is exactly zero, and X has not been computed. */
HWYBLAS_API void strtrs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dtrtrs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void ctrtrs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void ztrtrs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);

/* Solves op(A) * x = scale * b with the triangular matrix A
 * (uplo, diag), choosing scale <= 1 so that x does not overflow; x
 * overwrites b. cnorm: the 1-norms of the off-diagonal columns of A
 * (computed for normin = 'N', given for 'Y'). */
HWYBLAS_API void slatrs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda, float* x,
                         float* scale, float* cnorm, hwyblas_int* info);
HWYBLAS_API void dlatrs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda, double* x,
                         double* scale, double* cnorm, hwyblas_int* info);
HWYBLAS_API void clatrs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda, float* x,
                         float* scale, float* cnorm, hwyblas_int* info);
HWYBLAS_API void zlatrs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda, double* x,
                         double* scale, double* cnorm, hwyblas_int* info);

/* Estimates the reciprocal condition number of the triangular
 * matrix A (uplo, diag) in the 1-norm (norm = '1' or 'O') or the infinity
 * norm ('I'). work: 3n elements and iwork: n (real), work: 2n and rwork: n
 * (complex). */
HWYBLAS_API void strcon_(const char* norm, const char* uplo, const char* diag,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* rcond, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dtrcon_(const char* norm, const char* uplo, const char* diag,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* rcond, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void ctrcon_(const char* norm, const char* uplo, const char* diag,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* rcond, float* work,
                         float* rwork, hwyblas_int* info);
HWYBLAS_API void ztrcon_(const char* norm, const char* uplo, const char* diag,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* rcond, double* work,
                         double* rwork, hwyblas_int* info);

/* Error bounds (ferr) and backward errors (berr) of the solution
 * X of op(A) * X = B with the triangular matrix A (uplo, diag). work: 3n
 * elements and iwork: n (real), work: 2n and rwork: n (complex). */
HWYBLAS_API void strrfs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dtrrfs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda,
                         const double* b, const hwyblas_int* ldb,
                         const double* x, const hwyblas_int* ldx, double* ferr,
                         double* berr, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void ctrrfs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, float* rwork, hwyblas_int* info);
HWYBLAS_API void ztrrfs_(const char* uplo, const char* trans, const char* diag,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda,
                         const double* b, const hwyblas_int* ldb,
                         const double* x, const hwyblas_int* ldx, double* ferr,
                         double* berr, double* work, double* rwork,
                         hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* General linear systems: condition, equilibration, refinement             */

/* Estimates the reciprocal condition number of A in the 1-norm
 * (norm = '1' or 'O') or the infinity norm ('I') from its LU factorization
 * by ?getrf (a) and the norm anorm of A. work: 4n elements and iwork: n
 * (real), work: 2n and rwork: 2n (complex). */
HWYBLAS_API void sgecon_(const char* norm, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const float* anorm,
                         float* rcond, float* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dgecon_(const char* norm, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* anorm, double* rcond, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void cgecon_(const char* norm, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const float* anorm,
                         float* rcond, float* work, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgecon_(const char* norm, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* anorm, double* rcond, double* work,
                         double* rwork, hwyblas_int* info);

/* Row and column scalings r (m elements) and c (n) that
 * equilibrate the m x n matrix A, with rowcnd, colcnd (ratios of the
 * smallest to the largest scaling) and amax (the largest |a(i, j)|). info
 * > 0: row info (info <= m) or column info - m of A is exactly zero. */
HWYBLAS_API void sgeequ_(const hwyblas_int* m, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda, float* r,
                         float* c, float* rowcnd, float* colcnd, float* amax,
                         hwyblas_int* info);
HWYBLAS_API void dgeequ_(const hwyblas_int* m, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda, double* r,
                         double* c, double* rowcnd, double* colcnd,
                         double* amax, hwyblas_int* info);
HWYBLAS_API void cgeequ_(const hwyblas_int* m, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda, float* r,
                         float* c, float* rowcnd, float* colcnd, float* amax,
                         hwyblas_int* info);
HWYBLAS_API void zgeequ_(const hwyblas_int* m, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda, double* r,
                         double* c, double* rowcnd, double* colcnd,
                         double* amax, hwyblas_int* info);

/* Equilibrates A with the scalings of ?geequ where worthwhile;
 * equed returns 'N' (none), 'R' (rows), 'C' (columns) or 'B' (both). */
HWYBLAS_API void slaqge_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* r, const float* c,
                         const float* rowcnd, const float* colcnd,
                         const float* amax, char* equed);
HWYBLAS_API void dlaqge_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* r,
                         const double* c, const double* rowcnd,
                         const double* colcnd, const double* amax, char* equed);
HWYBLAS_API void claqge_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* r, const float* c,
                         const float* rowcnd, const float* colcnd,
                         const float* amax, char* equed);
HWYBLAS_API void zlaqge_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* r,
                         const double* c, const double* rowcnd,
                         const double* colcnd, const double* amax, char* equed);

/* Improves the solution X of op(A) * X = B by iterative refinement
 * with the LU factorization of A by ?getrf (af, ipiv), and computes error
 * bounds (ferr) and backward errors (berr). work: 3n elements and iwork: n
 * (real), work: 2n and rwork: n (complex). */
HWYBLAS_API void sgerfs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dgerfs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void cgerfs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, float* rwork, hwyblas_int* info);
HWYBLAS_API void zgerfs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, double* rwork, hwyblas_int* info);

/* Solves op(A) * X = B with the LU factorization of A (given for
 * fact = 'F', computed for 'N', computed after equilibration for 'E'),
 * with iterative refinement, error bounds and the reciprocal condition
 * number. work: 4n elements and iwork: n (real), work: 2n and rwork: 2n
 * (complex); work(1) (real) or rwork(1) (complex) returns the reciprocal
 * pivot growth factor. info = n + 1: A is singular to working precision. */
HWYBLAS_API void sgesvx_(const char* fact, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         char* equed, float* r, float* c, float* b,
                         const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dgesvx_(const char* fact, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         char* equed, double* r, double* c, double* b,
                         const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void cgesvx_(const char* fact, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         char* equed, float* r, float* c, float* b,
                         const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgesvx_(const char* fact, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         char* equed, double* r, double* c, double* b,
                         const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, double* rwork,
                         hwyblas_int* info);

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

/* Cholesky factorization with complete pivoting of the symmetric
 * (Hermitian) positive semidefinite A of order n: P^T*A*P = U^H*U or L*L^H;
 * piv: P (column piv(k) of A is column k of A*P); rank: the number of steps
 * done, stopping at a diagonal element <= tol (tol < 0: n*eps*max(A(k,k)));
 * work: 2n. info = 1: A is rank deficient (rank < n). ?pstrf is blocked,
 * ?pstf2 unblocked. */
HWYBLAS_API void spstrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const float* tol, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dpstrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const double* tol, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cpstrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const float* tol, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zpstrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const double* tol, double* work,
                         hwyblas_int* info);
HWYBLAS_API void spstf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const float* tol, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dpstf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const double* tol, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cpstf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const float* tol, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zpstf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* piv,
                         hwyblas_int* rank, const double* tol, double* work,
                         hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) positive definite systems                          */

/* Solves A*X = B with the Cholesky factorization of ?potrf (a);
 * b: n x nrhs, overwritten with X. */
HWYBLAS_API void spotrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dpotrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cpotrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zpotrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);

/* Solves A*X = B for symmetric (Hermitian) positive definite A:
 * ?potrf (a returns the factor), then ?potrs. info > 0: the leading minor of
 * order info is not positive. */
HWYBLAS_API void sposv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dposv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, double* a,
                        const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cposv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zposv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, double* a,
                        const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb, hwyblas_int* info);

/* U*U^H (uplo = 'U') or L^H*L ('L') of the triangular factor in the
 * triangle uplo of a, in place. ?lauum is blocked, ?lauu2 unblocked. */
HWYBLAS_API void slauum_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void dlauum_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void clauum_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void zlauum_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void slauu2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void dlauu2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void clauu2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void zlauu2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);

/* The inverse of A from its Cholesky factorization (?potrf), in
 * place (triangle uplo). info > 0: the factor is singular. */
HWYBLAS_API void spotri_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void dpotri_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void cpotri_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void zpotri_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);

/* Estimates the reciprocal 1-norm condition number of A from its
 * Cholesky factorization (a) and anorm = norm1(A). work: 3n (real) or 2n
 * (complex); iwork: n; rwork: n. */
HWYBLAS_API void spocon_(const char* uplo, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const float* anorm,
                         float* rcond, float* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dpocon_(const char* uplo, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* anorm, double* rcond, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void cpocon_(const char* uplo, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const float* anorm,
                         float* rcond, float* work, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zpocon_(const char* uplo, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* anorm, double* rcond, double* work,
                         double* rwork, hwyblas_int* info);

/* Scalings s(i) = 1/sqrt(A(i,i)) that equilibrate A; scond =
 * min(s)/max(s), amax = max(A(i,i)). info > 0: A(info,info) <= 0. */
HWYBLAS_API void spoequ_(const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* s, float* scond,
                         float* amax, hwyblas_int* info);
HWYBLAS_API void dpoequ_(const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* s, double* scond,
                         double* amax, hwyblas_int* info);
HWYBLAS_API void cpoequ_(const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, float* s, float* scond,
                         float* amax, hwyblas_int* info);
HWYBLAS_API void zpoequ_(const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, double* s, double* scond,
                         double* amax, hwyblas_int* info);

/* A := diag(s)*A*diag(s) if worthwhile (scond < 0.1 or amax
 * near underflow or overflow); equed = 'Y' if done, else 'N'. ?laqsy:
 * symmetric (complex symmetric for c, z), ?laqhe: Hermitian. */
HWYBLAS_API void slaqsy_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* s,
                         const float* scond, const float* amax, char* equed);
HWYBLAS_API void dlaqsy_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* s,
                         const double* scond, const double* amax, char* equed);
HWYBLAS_API void claqsy_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* s,
                         const float* scond, const float* amax, char* equed);
HWYBLAS_API void zlaqsy_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* s,
                         const double* scond, const double* amax, char* equed);
HWYBLAS_API void claqhe_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const float* s,
                         const float* scond, const float* amax, char* equed);
HWYBLAS_API void zlaqhe_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* s,
                         const double* scond, const double* amax, char* equed);

/* Iterative refinement of the solution x of A*X = B with the
 * Cholesky factorization af, with forward (ferr) and backward (berr) error
 * bounds. work: 3n (real) or 2n (complex); iwork: n; rwork: n. */
HWYBLAS_API void sporfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const float* b,
                         const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dporfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const double* b,
                         const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void cporfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const float* b,
                         const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, float* rwork, hwyblas_int* info);
HWYBLAS_API void zporfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const double* b,
                         const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, double* rwork, hwyblas_int* info);

/* Expert driver: solves A*X = B with the Cholesky factorization
 * (fact = 'F': given in af with equed, s; 'N': computed; 'E': after
 * equilibration if worthwhile), refines X and returns rcond, ferr and berr.
 * Workspaces as for ?porfs. info = n+1: rcond < eps. */
HWYBLAS_API void sposvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, char* equed, float* s,
                         float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dposvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, char* equed, double* s,
                         double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void cposvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, char* equed, float* s,
                         float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zposvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, char* equed, double* s,
                         double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, double* rwork,
                         hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) indefinite systems                                 */

/* Bunch-Kaufman factorization A = U*D*U^T or L*D*L^T (?sytrf;
 * complex symmetric for c, z) and A = U*D*U^H or L*D*L^H (?hetrf) with D
 * block diagonal (1x1 and 2x2 blocks) and the interchanges in ipiv. work:
 * lwork >= 1, n*nb for the optimal block size (lwork = -1: query). info > 0:
 * D(info,info) is zero. ?lasyf / ?lahef factor a panel of nb columns (kb:
 * the number factored; w: ldw x nb), ?sytf2 / ?hetf2 are unblocked. */
HWYBLAS_API void ssytrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dsytrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void csytrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zsytrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void chetrf_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zhetrf_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void slasyf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, hwyblas_int* kb, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, float* w,
                         const hwyblas_int* ldw, hwyblas_int* info);
HWYBLAS_API void dlasyf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, hwyblas_int* kb, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, double* w,
                         const hwyblas_int* ldw, hwyblas_int* info);
HWYBLAS_API void clasyf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, hwyblas_int* kb, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, float* w,
                         const hwyblas_int* ldw, hwyblas_int* info);
HWYBLAS_API void zlasyf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, hwyblas_int* kb, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, double* w,
                         const hwyblas_int* ldw, hwyblas_int* info);
HWYBLAS_API void clahef_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, hwyblas_int* kb, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, float* w,
                         const hwyblas_int* ldw, hwyblas_int* info);
HWYBLAS_API void zlahef_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nb, hwyblas_int* kb, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv, double* w,
                         const hwyblas_int* ldw, hwyblas_int* info);
HWYBLAS_API void ssytf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void dsytf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void csytf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void zsytf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void chetf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void zhetf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ipiv,
                         hwyblas_int* info);

/* Solve A*X = B with the factorization of ?sytrf / ?hetrf; b: n x
 * nrhs, overwritten with X. ?sytrs2 / ?hetrs2 use Level 3 BLAS (work: n;
 * a is changed and restored). */
HWYBLAS_API void ssytrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dsytrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void csytrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zsytrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void chetrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zhetrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void ssytrs2_(const char* uplo, const hwyblas_int* n,
                          const hwyblas_int* nrhs, float* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          float* b, const hwyblas_int* ldb, float* work,
                          hwyblas_int* info);
HWYBLAS_API void dsytrs2_(const char* uplo, const hwyblas_int* n,
                          const hwyblas_int* nrhs, double* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          double* b, const hwyblas_int* ldb, double* work,
                          hwyblas_int* info);
HWYBLAS_API void csytrs2_(const char* uplo, const hwyblas_int* n,
                          const hwyblas_int* nrhs, float* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          float* b, const hwyblas_int* ldb, float* work,
                          hwyblas_int* info);
HWYBLAS_API void zsytrs2_(const char* uplo, const hwyblas_int* n,
                          const hwyblas_int* nrhs, double* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          double* b, const hwyblas_int* ldb, double* work,
                          hwyblas_int* info);
HWYBLAS_API void chetrs2_(const char* uplo, const hwyblas_int* n,
                          const hwyblas_int* nrhs, float* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          float* b, const hwyblas_int* ldb, float* work,
                          hwyblas_int* info);
HWYBLAS_API void zhetrs2_(const char* uplo, const hwyblas_int* n,
                          const hwyblas_int* nrhs, double* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          double* b, const hwyblas_int* ldb, double* work,
                          hwyblas_int* info);

/* Converts the factorization of ?sytrf (way = 'C') to the form of
 * ?sytrs2 (the off-diagonal elements of D in e: n) or back ('R'). */
HWYBLAS_API void ssyconv_(const char* uplo, const char* way,
                          const hwyblas_int* n, float* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          float* e, hwyblas_int* info);
HWYBLAS_API void dsyconv_(const char* uplo, const char* way,
                          const hwyblas_int* n, double* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          double* e, hwyblas_int* info);
HWYBLAS_API void csyconv_(const char* uplo, const char* way,
                          const hwyblas_int* n, float* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          float* e, hwyblas_int* info);
HWYBLAS_API void zsyconv_(const char* uplo, const char* way,
                          const hwyblas_int* n, double* a,
                          const hwyblas_int* lda, const hwyblas_int* ipiv,
                          double* e, hwyblas_int* info);

/* The inverse of A from the factorization of ?sytrf / ?hetrf, in
 * place (triangle uplo); work: n. info > 0: D(info,info) is zero. */
HWYBLAS_API void ssytri_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* work, hwyblas_int* info);
HWYBLAS_API void dsytri_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* work, hwyblas_int* info);
HWYBLAS_API void csytri_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* work, hwyblas_int* info);
HWYBLAS_API void zsytri_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* work, hwyblas_int* info);
HWYBLAS_API void chetri_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         float* work, hwyblas_int* info);
HWYBLAS_API void zhetri_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         double* work, hwyblas_int* info);

/* Estimate the reciprocal 1-norm condition number of A from the
 * factorization of ?sytrf / ?hetrf and anorm = norm1(A). work: 2n; iwork:
 * n. */
HWYBLAS_API void ssycon_(const char* uplo, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         const float* anorm, float* rcond, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dsycon_(const char* uplo, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const hwyblas_int* ipiv, const double* anorm,
                         double* rcond, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void csycon_(const char* uplo, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         const float* anorm, float* rcond, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zsycon_(const char* uplo, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const hwyblas_int* ipiv, const double* anorm,
                         double* rcond, double* work, hwyblas_int* info);
HWYBLAS_API void checon_(const char* uplo, const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const hwyblas_int* ipiv,
                         const float* anorm, float* rcond, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zhecon_(const char* uplo, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const hwyblas_int* ipiv, const double* anorm,
                         double* rcond, double* work, hwyblas_int* info);

/* Iterative refinement of the solution x of A*X = B with the
 * factorization af, ipiv, with forward (ferr) and backward (berr) error
 * bounds. work: 3n (real) or 2n (complex); iwork: n; rwork: n. */
HWYBLAS_API void ssyrfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dsyrfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void csyrfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, float* rwork, hwyblas_int* info);
HWYBLAS_API void zsyrfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, double* rwork, hwyblas_int* info);
HWYBLAS_API void cherfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* a,
                         const hwyblas_int* lda, const float* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* ferr, float* berr,
                         float* work, float* rwork, hwyblas_int* info);
HWYBLAS_API void zherfs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* a,
                         const hwyblas_int* lda, const double* af,
                         const hwyblas_int* ldaf, const hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* ferr, double* berr,
                         double* work, double* rwork, hwyblas_int* info);

/* Solve A*X = B for symmetric (Hermitian) A: ?sytrf / ?hetrf (a,
 * ipiv return the factorization), then ?sytrs2 / ?hetrs2 (?sytrs / ?hetrs
 * if lwork < n). lwork >= 1 (-1: query). info > 0: D(info,info) is zero. */
HWYBLAS_API void ssysv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, float* b,
                        const hwyblas_int* ldb, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dsysv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, double* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, double* b,
                        const hwyblas_int* ldb, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void csysv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, float* b,
                        const hwyblas_int* ldb, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zsysv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, double* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, double* b,
                        const hwyblas_int* ldb, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void chesv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, float* b,
                        const hwyblas_int* ldb, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zhesv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* nrhs, double* a,
                        const hwyblas_int* lda, hwyblas_int* ipiv, double* b,
                        const hwyblas_int* ldb, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);

/* Expert drivers: solve A*X = B with the factorization (fact =
 * 'F': given in af, ipiv; 'N': computed), refine X and return rcond, ferr
 * and berr. work: lwork >= max(1, 3n) (real) or max(1, 2n) (complex)
 * (-1: query); iwork: n; rwork: n. info = n+1: rcond < eps. */
HWYBLAS_API void ssysvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dsysvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void csysvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, const hwyblas_int* lwork,
                         float* rwork, hwyblas_int* info);
HWYBLAS_API void zsysvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, const hwyblas_int* lwork,
                         double* rwork, hwyblas_int* info);
HWYBLAS_API void chesvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* a, const hwyblas_int* lda, float* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         const float* b, const hwyblas_int* ldb, float* x,
                         const hwyblas_int* ldx, float* rcond, float* ferr,
                         float* berr, float* work, const hwyblas_int* lwork,
                         float* rwork, hwyblas_int* info);
HWYBLAS_API void zhesvx_(const char* fact, const char* uplo,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* a, const hwyblas_int* lda, double* af,
                         const hwyblas_int* ldaf, hwyblas_int* ipiv,
                         const double* b, const hwyblas_int* ldb, double* x,
                         const hwyblas_int* ldx, double* rcond, double* ferr,
                         double* berr, double* work, const hwyblas_int* lwork,
                         double* rwork, hwyblas_int* info);

/* LAPACK's complex symmetric (not Hermitian) y := alpha*A*x +
 * beta*y and A := alpha*x*x^T + A (the BLAS has the real and Hermitian
 * ones). */
HWYBLAS_API void csymv_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zsymv_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* x,
                        const hwyblas_int* incx, const double* beta, double* y,
                        const hwyblas_int* incy);
HWYBLAS_API void csyr_(const char* uplo, const hwyblas_int* n,
                       const float* alpha, const float* x,
                       const hwyblas_int* incx, float* a,
                       const hwyblas_int* lda);
HWYBLAS_API void zsyr_(const char* uplo, const hwyblas_int* n,
                       const double* alpha, const double* x,
                       const hwyblas_int* incx, double* a,
                       const hwyblas_int* lda);

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

/* The norm (see above) of the m x n upper ('U') or lower ('L')
 * trapezoidal matrix A (triangular if m = n), with unit diagonal for
 * diag = 'U' (not referenced); work: m elements, for 'I'. */
HWYBLAS_API float slantr_(const char* norm, const char* uplo, const char* diag,
                          const hwyblas_int* m, const hwyblas_int* n,
                          const float* a, const hwyblas_int* lda, float* work);
HWYBLAS_API double dlantr_(const char* norm, const char* uplo, const char* diag,
                           const hwyblas_int* m, const hwyblas_int* n,
                           const double* a, const hwyblas_int* lda,
                           double* work);
HWYBLAS_API float clantr_(const char* norm, const char* uplo, const char* diag,
                          const hwyblas_int* m, const hwyblas_int* n,
                          const float* a, const hwyblas_int* lda, float* work);
HWYBLAS_API double zlantr_(const char* norm, const char* uplo, const char* diag,
                           const hwyblas_int* m, const hwyblas_int* n,
                           const double* a, const hwyblas_int* lda,
                           double* work);

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
/* Symmetric (Hermitian) eigenproblems by divide and conquer                */

/* All eigenvalues (w, ascending) and, for jobz = 'V', the orthonormal
 * eigenvectors (in a) of the symmetric (Hermitian) n x n matrix A (triangle
 * uplo): ?sytrd / ?hetrd, then ?sterf or ?stedc and ?ormtr / ?unmtr. n > 1:
 * lwork >= 2n+1 / 1+6n+2n^2 (real, jobz 'N' / 'V') or n+1 / 2n+n^2 (complex),
 * lrwork >= n / 1+5n+2n^2, liwork >= 1 / 3+5n (-1: query). info > 0: no
 * convergence. */
HWYBLAS_API void ssyevd_(const char* jobz, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* w, float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void dsyevd_(const char* jobz, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* w, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void cheevd_(const char* jobz, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* w, float* work, const hwyblas_int* lwork,
                         float* rwork, const hwyblas_int* lrwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void zheevd_(const char* jobz, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* w, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);

/* All eigenvalues (d, ascending) and, for jobz = 'V', the eigenvectors
 * (z) of the symmetric tridiagonal matrix (d, e; e is destroyed): ?sterf or
 * ?stedc. jobz = 'V', n > 1: lwork >= 1+4n+n^2, liwork >= 3+5n (-1: query). */
HWYBLAS_API void sstevd_(const char* jobz, const hwyblas_int* n, float* d,
                         float* e, float* z, const hwyblas_int* ldz,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void dstevd_(const char* jobz, const hwyblas_int* n, double* d,
                         double* e, double* z, const hwyblas_int* ldz,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);

/* Eigenvalues (d, ascending) and, for compz = 'I', the eigenvectors of
 * the symmetric tridiagonal matrix (d, e) by divide and conquer, or for
 * compz = 'V' the product of z with them; 'N': eigenvalues only (?sterf).
 * Workspaces: see the reference LAPACK (-1: query). info > 0: an eigenvalue
 * did not converge. */
HWYBLAS_API void sstedc_(const char* compz, const hwyblas_int* n, float* d,
                         float* e, float* z, const hwyblas_int* ldz,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void dstedc_(const char* compz, const hwyblas_int* n, double* d,
                         double* e, double* z, const hwyblas_int* ldz,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void cstedc_(const char* compz, const hwyblas_int* n, float* d,
                         float* e, float* z, const hwyblas_int* ldz,
                         float* work, const hwyblas_int* lwork, float* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void zstedc_(const char* compz, const hwyblas_int* n, double* d,
                         double* e, double* z, const hwyblas_int* ldz,
                         double* work, const hwyblas_int* lwork, double* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);

/* The steps of ?stedc: ?laed0 (the tree of subproblems), ?laed1 /
 * ?laed7 (merge of two subproblems, with ?laed2 / ?laed8 for the deflation
 * and ?laed3 / ?laed9 for the secular equation and the eigenvectors), ?laeda
 * (the vector z of a merge of ?laed7), ?laed4 / ?laed5 (one root of the
 * secular equation). Real, except ?laed0, ?laed7 and ?laed8, which also
 * exist for complex Q (c, z; different arguments). */
HWYBLAS_API void slaed0_(const hwyblas_int* icompq, const hwyblas_int* qsiz,
                         const hwyblas_int* n, float* d, float* e, float* q,
                         const hwyblas_int* ldq, float* qstore,
                         const hwyblas_int* ldqs, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlaed0_(const hwyblas_int* icompq, const hwyblas_int* qsiz,
                         const hwyblas_int* n, double* d, double* e, double* q,
                         const hwyblas_int* ldq, double* qstore,
                         const hwyblas_int* ldqs, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void claed0_(const hwyblas_int* qsiz, const hwyblas_int* n,
                         float* d, float* e, float* q, const hwyblas_int* ldq,
                         float* qstore, const hwyblas_int* ldqs, float* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zlaed0_(const hwyblas_int* qsiz, const hwyblas_int* n,
                         double* d, double* e, double* q,
                         const hwyblas_int* ldq, double* qstore,
                         const hwyblas_int* ldqs, double* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slaed1_(const hwyblas_int* n, float* d, float* q,
                         const hwyblas_int* ldq, hwyblas_int* indxq, float* rho,
                         const hwyblas_int* cutpnt, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlaed1_(const hwyblas_int* n, double* d, double* q,
                         const hwyblas_int* ldq, hwyblas_int* indxq,
                         double* rho, const hwyblas_int* cutpnt, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slaed2_(hwyblas_int* k, const hwyblas_int* n,
                         const hwyblas_int* n1, float* d, float* q,
                         const hwyblas_int* ldq, hwyblas_int* indxq, float* rho,
                         float* z, float* dlambda, float* w, float* q2,
                         hwyblas_int* indx, hwyblas_int* indxc,
                         hwyblas_int* indxp, hwyblas_int* coltyp,
                         hwyblas_int* info);
HWYBLAS_API void dlaed2_(hwyblas_int* k, const hwyblas_int* n,
                         const hwyblas_int* n1, double* d, double* q,
                         const hwyblas_int* ldq, hwyblas_int* indxq,
                         double* rho, double* z, double* dlambda, double* w,
                         double* q2, hwyblas_int* indx, hwyblas_int* indxc,
                         hwyblas_int* indxp, hwyblas_int* coltyp,
                         hwyblas_int* info);
HWYBLAS_API void slaed3_(const hwyblas_int* k, const hwyblas_int* n,
                         const hwyblas_int* n1, float* d, float* q,
                         const hwyblas_int* ldq, const float* rho,
                         float* dlambda, float* q2, hwyblas_int* indx,
                         hwyblas_int* ctot, float* w, float* s,
                         hwyblas_int* info);
HWYBLAS_API void dlaed3_(const hwyblas_int* k, const hwyblas_int* n,
                         const hwyblas_int* n1, double* d, double* q,
                         const hwyblas_int* ldq, const double* rho,
                         double* dlambda, double* q2, hwyblas_int* indx,
                         hwyblas_int* ctot, double* w, double* s,
                         hwyblas_int* info);
HWYBLAS_API void slaed4_(const hwyblas_int* n, const hwyblas_int* i, float* d,
                         float* z, float* delta, const float* rho, float* dlam,
                         hwyblas_int* info);
HWYBLAS_API void dlaed4_(const hwyblas_int* n, const hwyblas_int* i, double* d,
                         double* z, double* delta, const double* rho,
                         double* dlam, hwyblas_int* info);
HWYBLAS_API void slaed5_(const hwyblas_int* i, float* d, float* z, float* delta,
                         const float* rho, float* dlam);
HWYBLAS_API void dlaed5_(const hwyblas_int* i, double* d, double* z,
                         double* delta, const double* rho, double* dlam);
HWYBLAS_API void slaed7_(const hwyblas_int* icompq, const hwyblas_int* n,
                         const hwyblas_int* qsiz, const hwyblas_int* tlvls,
                         const hwyblas_int* curlvl, const hwyblas_int* curpbm,
                         float* d, float* q, const hwyblas_int* ldq,
                         hwyblas_int* indxq, float* rho,
                         const hwyblas_int* cutpnt, float* qstore,
                         hwyblas_int* qptr, hwyblas_int* prmptr,
                         hwyblas_int* perm, hwyblas_int* givptr,
                         hwyblas_int* givcol, float* givnum, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlaed7_(const hwyblas_int* icompq, const hwyblas_int* n,
                         const hwyblas_int* qsiz, const hwyblas_int* tlvls,
                         const hwyblas_int* curlvl, const hwyblas_int* curpbm,
                         double* d, double* q, const hwyblas_int* ldq,
                         hwyblas_int* indxq, double* rho,
                         const hwyblas_int* cutpnt, double* qstore,
                         hwyblas_int* qptr, hwyblas_int* prmptr,
                         hwyblas_int* perm, hwyblas_int* givptr,
                         hwyblas_int* givcol, double* givnum, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void claed7_(const hwyblas_int* n, const hwyblas_int* cutpnt,
                         const hwyblas_int* qsiz, const hwyblas_int* tlvls,
                         const hwyblas_int* curlvl, const hwyblas_int* curpbm,
                         float* d, float* q, const hwyblas_int* ldq, float* rho,
                         hwyblas_int* indxq, float* qstore, hwyblas_int* qptr,
                         hwyblas_int* prmptr, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         float* givnum, float* work, float* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zlaed7_(const hwyblas_int* n, const hwyblas_int* cutpnt,
                         const hwyblas_int* qsiz, const hwyblas_int* tlvls,
                         const hwyblas_int* curlvl, const hwyblas_int* curpbm,
                         double* d, double* q, const hwyblas_int* ldq,
                         double* rho, hwyblas_int* indxq, double* qstore,
                         hwyblas_int* qptr, hwyblas_int* prmptr,
                         hwyblas_int* perm, hwyblas_int* givptr,
                         hwyblas_int* givcol, double* givnum, double* work,
                         double* rwork, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slaed8_(const hwyblas_int* icompq, hwyblas_int* k,
                         const hwyblas_int* n, const hwyblas_int* qsiz,
                         float* d, float* q, const hwyblas_int* ldq,
                         hwyblas_int* indxq, float* rho,
                         const hwyblas_int* cutpnt, float* z, float* dlambda,
                         float* q2, const hwyblas_int* ldq2, float* w,
                         hwyblas_int* perm, hwyblas_int* givptr,
                         hwyblas_int* givcol, float* givnum, hwyblas_int* indxp,
                         hwyblas_int* indx, hwyblas_int* info);
HWYBLAS_API void dlaed8_(
    const hwyblas_int* icompq, hwyblas_int* k, const hwyblas_int* n,
    const hwyblas_int* qsiz, double* d, double* q, const hwyblas_int* ldq,
    hwyblas_int* indxq, double* rho, const hwyblas_int* cutpnt, double* z,
    double* dlambda, double* q2, const hwyblas_int* ldq2, double* w,
    hwyblas_int* perm, hwyblas_int* givptr, hwyblas_int* givcol, double* givnum,
    hwyblas_int* indxp, hwyblas_int* indx, hwyblas_int* info);
HWYBLAS_API void claed8_(hwyblas_int* k, const hwyblas_int* n,
                         const hwyblas_int* qsiz, float* q,
                         const hwyblas_int* ldq, float* d, float* rho,
                         const hwyblas_int* cutpnt, float* z, float* dlambda,
                         float* q2, const hwyblas_int* ldq2, float* w,
                         hwyblas_int* indxp, hwyblas_int* indx,
                         hwyblas_int* indxq, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         float* givnum, hwyblas_int* info);
HWYBLAS_API void zlaed8_(hwyblas_int* k, const hwyblas_int* n,
                         const hwyblas_int* qsiz, double* q,
                         const hwyblas_int* ldq, double* d, double* rho,
                         const hwyblas_int* cutpnt, double* z, double* dlambda,
                         double* q2, const hwyblas_int* ldq2, double* w,
                         hwyblas_int* indxp, hwyblas_int* indx,
                         hwyblas_int* indxq, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         double* givnum, hwyblas_int* info);
HWYBLAS_API void slaed9_(const hwyblas_int* k, const hwyblas_int* kstart,
                         const hwyblas_int* kstop, const hwyblas_int* n,
                         float* d, float* q, const hwyblas_int* ldq,
                         const float* rho, float* dlambda, float* w, float* s,
                         const hwyblas_int* lds, hwyblas_int* info);
HWYBLAS_API void dlaed9_(const hwyblas_int* k, const hwyblas_int* kstart,
                         const hwyblas_int* kstop, const hwyblas_int* n,
                         double* d, double* q, const hwyblas_int* ldq,
                         const double* rho, double* dlambda, double* w,
                         double* s, const hwyblas_int* lds, hwyblas_int* info);
HWYBLAS_API void slaeda_(const hwyblas_int* n, const hwyblas_int* tlvls,
                         const hwyblas_int* curlvl, const hwyblas_int* curpbm,
                         hwyblas_int* prmptr, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         float* givnum, float* q, hwyblas_int* qptr, float* z,
                         float* ztemp, hwyblas_int* info);
HWYBLAS_API void dlaeda_(const hwyblas_int* n, const hwyblas_int* tlvls,
                         const hwyblas_int* curlvl, const hwyblas_int* curpbm,
                         hwyblas_int* prmptr, hwyblas_int* perm,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         double* givnum, double* q, hwyblas_int* qptr,
                         double* z, double* ztemp, hwyblas_int* info);

/* C := op(Q)*C (side 'L') or C*op(Q) (side 'R'), op = 'N' or 'T' /
 * 'C', for the Q of ?sytrd / ?hetrd (same uplo; the reflectors in a, tau).
 * lwork >= max(1, n) (side 'L') or max(1, m) (side 'R') (-1: query). */
HWYBLAS_API void sormtr_(const char* side, const char* uplo, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dormtr_(const char* side, const char* uplo, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cunmtr_(const char* side, const char* uplo, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zunmtr_(const char* side, const char* uplo, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);

/* C := op(Q)*C or C*op(Q) for the Q = H(k)...H(1) of a QL factorization
 * (k reflectors in the last columns of a, tau): ?ormql / ?unmql blocked
 * (lwork >= max(1, n) or max(1, m), -1: query), ?orm2l / ?unm2l unblocked
 * (work: n or m). */
HWYBLAS_API void sormql_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dormql_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cunmql_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zunmql_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sorm2l_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dorm2l_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cunm2l_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zunm2l_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);

/* Reduces the generalized problem with symmetric (Hermitian) A and the
 * Cholesky factor of B (?potrf, same uplo) to standard form: itype = 1:
 * inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H); 2, 3: U*A*U^H or L^H*A*L (in a).
 * ?sygst / ?hegst blocked, ?sygs2 / ?hegs2 unblocked. */
HWYBLAS_API void ssygst_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const float* b, const hwyblas_int* ldb,
                         hwyblas_int* info);
HWYBLAS_API void dsygst_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void chegst_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const float* b, const hwyblas_int* ldb,
                         hwyblas_int* info);
HWYBLAS_API void zhegst_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void ssygs2_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dsygs2_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void chegs2_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zhegs2_(const hwyblas_int* itype, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);

/* All eigenvalues (w, ascending) and, for jobz = 'V', the eigenvectors
 * (in a, B-orthonormal) of the generalized problem A*x = lambda*B*x
 * (itype 1), A*B*x = lambda*x (2) or B*A*x = lambda*x (3) with symmetric
 * (Hermitian) A and positive definite B (b returns its Cholesky factor):
 * ?sygv / ?hegv with ?syev / ?heev (lwork >= max(1, 3n-1) real or
 * max(1, 2n-1) complex; rwork: max(1, 3n-2)), ?sygvd / ?hegvd with ?syevd /
 * ?heevd (workspaces as there). info > n: the leading minor of order
 * info-n of B is not positive. */
HWYBLAS_API void ssygv_(const hwyblas_int* itype, const char* jobz,
                        const char* uplo, const hwyblas_int* n, float* a,
                        const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb, float* w, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dsygv_(const hwyblas_int* itype, const char* jobz,
                        const char* uplo, const hwyblas_int* n, double* a,
                        const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb, double* w, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void chegv_(const hwyblas_int* itype, const char* jobz,
                        const char* uplo, const hwyblas_int* n, float* a,
                        const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb, float* w, float* work,
                        const hwyblas_int* lwork, float* rwork,
                        hwyblas_int* info);
HWYBLAS_API void zhegv_(const hwyblas_int* itype, const char* jobz,
                        const char* uplo, const hwyblas_int* n, double* a,
                        const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb, double* w, double* work,
                        const hwyblas_int* lwork, double* rwork,
                        hwyblas_int* info);
HWYBLAS_API void ssygvd_(const hwyblas_int* itype, const char* jobz,
                         const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, float* w, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void dsygvd_(const hwyblas_int* itype, const char* jobz,
                         const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* w, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void chegvd_(const hwyblas_int* itype, const char* jobz,
                         const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, float* w, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void zhegvd_(const hwyblas_int* itype, const char* jobz,
                         const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* w, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) eigenproblems by MRRR, bisection and inverse iteration
 */

/* Selected eigenvalues (w, ascending; m of them) and, for jobz = 'V',
 * eigenvectors (z) of the symmetric (Hermitian) n x n matrix A (triangle
 * uplo; destroyed): all (range = 'A'), in (vl, vu] ('V') or the il-th
 * through iu-th ('I'). ?sytrd / ?hetrd, then ?sterf or ?stemr (MRRR) for all
 * eigenvalues, ?stebz and ?stein (bisection, inverse iteration) for a
 * subset or if ?stemr fails, and ?ormtr / ?unmtr. isuppz: the supports of
 * the eigenvectors (2*max(1, m); ?stemr only). n > 1: lwork >= 26n (real)
 * or 2n (complex), lrwork >= 24n, liwork >= 10n (-1: query). info > 0:
 * internal error. */
HWYBLAS_API void ssyevr_(const char* jobz, const char* range, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, hwyblas_int* m, float* w,
                         float* z, const hwyblas_int* ldz, hwyblas_int* isuppz,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void dsyevr_(
    const char* jobz, const char* range, const char* uplo, const hwyblas_int* n,
    double* a, const hwyblas_int* lda, const double* vl, const double* vu,
    const hwyblas_int* il, const hwyblas_int* iu, const double* abstol,
    hwyblas_int* m, double* w, double* z, const hwyblas_int* ldz,
    hwyblas_int* isuppz, double* work, const hwyblas_int* lwork,
    hwyblas_int* iwork, const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void cheevr_(const char* jobz, const char* range, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, hwyblas_int* m, float* w,
                         float* z, const hwyblas_int* ldz, hwyblas_int* isuppz,
                         float* work, const hwyblas_int* lwork, float* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void zheevr_(const char* jobz, const char* range, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* vl,
                         const double* vu, const hwyblas_int* il,
                         const hwyblas_int* iu, const double* abstol,
                         hwyblas_int* m, double* w, double* z,
                         const hwyblas_int* ldz, hwyblas_int* isuppz,
                         double* work, const hwyblas_int* lwork, double* rwork,
                         const hwyblas_int* lrwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);

/* Selected eigenvalues and, for jobz = 'V', eigenvectors of the
 * symmetric tridiagonal matrix (d, e): as ?syevr. lwork >= max(1, 20n),
 * liwork >= max(1, 10n) (-1: query). */
HWYBLAS_API void sstevr_(const char* jobz, const char* range,
                         const hwyblas_int* n, float* d, float* e,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, hwyblas_int* m, float* w,
                         float* z, const hwyblas_int* ldz, hwyblas_int* isuppz,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);
HWYBLAS_API void dstevr_(const char* jobz, const char* range,
                         const hwyblas_int* n, double* d, double* e,
                         const double* vl, const double* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const double* abstol, hwyblas_int* m, double* w,
                         double* z, const hwyblas_int* ldz, hwyblas_int* isuppz,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* info);

/* Selected eigenvalues (w, ascending; m of them) and, for jobz = 'V',
 * eigenvectors (z) of the symmetric (Hermitian) n x n matrix A (triangle
 * uplo; destroyed), as ?syevr, by ?sytrd / ?hetrd, then ?sterf or
 * ?steqr for all eigenvalues if abstol <= 0, ?stebz and ?stein
 * (bisection, inverse iteration) for a subset or if those fail, and
 * ?ormtr / ?unmtr. ifail: the eigenvectors that failed to converge
 * (info > 0: their number). n > 1: lwork >= 8n (real) or 2n (complex)
 * (-1: query), rwork 7n, iwork 5n. */
HWYBLAS_API void ssyevx_(const char* jobz, const char* range, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, hwyblas_int* m, float* w,
                         float* z, const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* ifail, hwyblas_int* info);
HWYBLAS_API void dsyevx_(const char* jobz, const char* range, const char* uplo,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, const double* vl,
                         const double* vu, const hwyblas_int* il,
                         const hwyblas_int* iu, const double* abstol,
                         hwyblas_int* m, double* w, double* z,
                         const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* ifail, hwyblas_int* info);
HWYBLAS_API void cheevx_(const char* jobz, const char* range, const char* uplo,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, hwyblas_int* m, float* w,
                         float* z, const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);
HWYBLAS_API void zheevx_(
    const char* jobz, const char* range, const char* uplo, const hwyblas_int* n,
    double* a, const hwyblas_int* lda, const double* vl, const double* vu,
    const hwyblas_int* il, const hwyblas_int* iu, const double* abstol,
    hwyblas_int* m, double* w, double* z, const hwyblas_int* ldz, double* work,
    const hwyblas_int* lwork, double* rwork, hwyblas_int* iwork,
    hwyblas_int* ifail, hwyblas_int* info);

/* Selected eigenvalues and, for jobz = 'V', eigenvectors of the
 * symmetric tridiagonal matrix (d, e): as ?syevx (?steqr for all
 * eigenvectors if abstol <= 0). work 5n, iwork 5n. */
HWYBLAS_API void sstevx_(const char* jobz, const char* range,
                         const hwyblas_int* n, float* d, float* e,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, hwyblas_int* m, float* w,
                         float* z, const hwyblas_int* ldz, float* work,
                         hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);
HWYBLAS_API void dstevx_(const char* jobz, const char* range,
                         const hwyblas_int* n, double* d, double* e,
                         const double* vl, const double* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const double* abstol, hwyblas_int* m, double* w,
                         double* z, const hwyblas_int* ldz, double* work,
                         hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);

/* Selected eigenvalues (w, ascending) and, for jobz = 'V', eigenvectors
 * (z, nzc columns; nzc = -1: query for their number, in z[0]) of the
 * symmetric tridiagonal matrix (d, e; destroyed) by MRRR. tryrac: on entry
 * whether to try for high relative accuracy, on exit whether it was done
 * (LOGICAL). lwork >= 18n (12n for jobz = 'N'), liwork >= 10n (8n) (-1:
 * query); the workspaces are real also for complex z. */
HWYBLAS_API void sstemr_(const char* jobz, const char* range,
                         const hwyblas_int* n, float* d, float* e,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         hwyblas_int* m, float* w, float* z,
                         const hwyblas_int* ldz, const hwyblas_int* nzc,
                         hwyblas_int* isuppz, hwyblas_int* tryrac, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void dstemr_(const char* jobz, const char* range,
                         const hwyblas_int* n, double* d, double* e,
                         const double* vl, const double* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         hwyblas_int* m, double* w, double* z,
                         const hwyblas_int* ldz, const hwyblas_int* nzc,
                         hwyblas_int* isuppz, hwyblas_int* tryrac, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void cstemr_(const char* jobz, const char* range,
                         const hwyblas_int* n, float* d, float* e,
                         const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         hwyblas_int* m, float* w, float* z,
                         const hwyblas_int* ldz, const hwyblas_int* nzc,
                         hwyblas_int* isuppz, hwyblas_int* tryrac, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void zstemr_(const char* jobz, const char* range,
                         const hwyblas_int* n, double* d, double* e,
                         const double* vl, const double* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         hwyblas_int* m, double* w, double* z,
                         const hwyblas_int* ldz, const hwyblas_int* nzc,
                         hwyblas_int* isuppz, hwyblas_int* tryrac, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);

/* The steps of ?stemr: ?larre (root representations and eigenvalue
 * approximations), ?larrv (eigenvectors by the representation tree), ?lar1v
 * (one eigenvector by a twisted factorization), ?larra (splitting), ?larrb,
 * ?larrd, ?larrj, ?larrk (bisection), ?larrc (Sturm counts), ?larrf (child
 * representation), ?larrr (relative accuracy test), ?laneg (Sturm count of
 * L D L^T - sigma I). Real, except ?lar1v and ?larrv, which also exist for
 * complex Z (c, z). */
HWYBLAS_API hwyblas_int slaneg_(const hwyblas_int* n, float* d, float* lld,
                                const float* sigma, const float* pivmin,
                                const hwyblas_int* r);
HWYBLAS_API hwyblas_int dlaneg_(const hwyblas_int* n, double* d, double* lld,
                                const double* sigma, const double* pivmin,
                                const hwyblas_int* r);
HWYBLAS_API void slar1v_(const hwyblas_int* n, const hwyblas_int* b1,
                         const hwyblas_int* bn, const float* lambda, float* d,
                         float* l, float* ld, float* lld, const float* pivmin,
                         const float* gaptol, float* z,
                         const hwyblas_int* wantnc, hwyblas_int* negcnt,
                         float* ztz, float* mingma, hwyblas_int* r,
                         hwyblas_int* isuppz, float* nrminv, float* resid,
                         float* rqcorr, float* work);
HWYBLAS_API void dlar1v_(const hwyblas_int* n, const hwyblas_int* b1,
                         const hwyblas_int* bn, const double* lambda, double* d,
                         double* l, double* ld, double* lld,
                         const double* pivmin, const double* gaptol, double* z,
                         const hwyblas_int* wantnc, hwyblas_int* negcnt,
                         double* ztz, double* mingma, hwyblas_int* r,
                         hwyblas_int* isuppz, double* nrminv, double* resid,
                         double* rqcorr, double* work);
HWYBLAS_API void clar1v_(const hwyblas_int* n, const hwyblas_int* b1,
                         const hwyblas_int* bn, const float* lambda, float* d,
                         float* l, float* ld, float* lld, const float* pivmin,
                         const float* gaptol, float* z,
                         const hwyblas_int* wantnc, hwyblas_int* negcnt,
                         float* ztz, float* mingma, hwyblas_int* r,
                         hwyblas_int* isuppz, float* nrminv, float* resid,
                         float* rqcorr, float* work);
HWYBLAS_API void zlar1v_(const hwyblas_int* n, const hwyblas_int* b1,
                         const hwyblas_int* bn, const double* lambda, double* d,
                         double* l, double* ld, double* lld,
                         const double* pivmin, const double* gaptol, double* z,
                         const hwyblas_int* wantnc, hwyblas_int* negcnt,
                         double* ztz, double* mingma, hwyblas_int* r,
                         hwyblas_int* isuppz, double* nrminv, double* resid,
                         double* rqcorr, double* work);
HWYBLAS_API void slarra_(const hwyblas_int* n, float* d, float* e, float* e2,
                         const float* spltol, const float* tnrm,
                         hwyblas_int* nsplit, hwyblas_int* isplit,
                         hwyblas_int* info);
HWYBLAS_API void dlarra_(const hwyblas_int* n, double* d, double* e, double* e2,
                         const double* spltol, const double* tnrm,
                         hwyblas_int* nsplit, hwyblas_int* isplit,
                         hwyblas_int* info);
HWYBLAS_API void slarrb_(const hwyblas_int* n, float* d, float* lld,
                         const hwyblas_int* ifirst, const hwyblas_int* ilast,
                         const float* rtol1, const float* rtol2,
                         const hwyblas_int* offset, float* w, float* wgap,
                         float* werr, float* work, hwyblas_int* iwork,
                         const float* pivmin, const float* spdiam,
                         const hwyblas_int* twist, hwyblas_int* info);
HWYBLAS_API void dlarrb_(const hwyblas_int* n, double* d, double* lld,
                         const hwyblas_int* ifirst, const hwyblas_int* ilast,
                         const double* rtol1, const double* rtol2,
                         const hwyblas_int* offset, double* w, double* wgap,
                         double* werr, double* work, hwyblas_int* iwork,
                         const double* pivmin, const double* spdiam,
                         const hwyblas_int* twist, hwyblas_int* info);
HWYBLAS_API void slarrc_(const char* jobt, const hwyblas_int* n,
                         const float* vl, const float* vu, float* d, float* e,
                         const float* pivmin, hwyblas_int* eigcnt,
                         hwyblas_int* lcnt, hwyblas_int* rcnt,
                         hwyblas_int* info);
HWYBLAS_API void dlarrc_(const char* jobt, const hwyblas_int* n,
                         const double* vl, const double* vu, double* d,
                         double* e, const double* pivmin, hwyblas_int* eigcnt,
                         hwyblas_int* lcnt, hwyblas_int* rcnt,
                         hwyblas_int* info);
HWYBLAS_API void slarrd_(const char* range, const char* order,
                         const hwyblas_int* n, const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         float* gers, const float* reltol, float* d, float* e,
                         float* e2, const float* pivmin,
                         const hwyblas_int* nsplit, hwyblas_int* isplit,
                         hwyblas_int* m, float* w, float* werr, float* wl,
                         float* wu, hwyblas_int* iblock, hwyblas_int* indexw,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlarrd_(const char* range, const char* order,
                         const hwyblas_int* n, const double* vl,
                         const double* vu, const hwyblas_int* il,
                         const hwyblas_int* iu, double* gers,
                         const double* reltol, double* d, double* e, double* e2,
                         const double* pivmin, const hwyblas_int* nsplit,
                         hwyblas_int* isplit, hwyblas_int* m, double* w,
                         double* werr, double* wl, double* wu,
                         hwyblas_int* iblock, hwyblas_int* indexw, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slarre_(const char* range, const hwyblas_int* n, float* vl,
                         float* vu, const hwyblas_int* il,
                         const hwyblas_int* iu, float* d, float* e, float* e2,
                         const float* rtol1, const float* rtol2,
                         const float* spltol, hwyblas_int* nsplit,
                         hwyblas_int* isplit, hwyblas_int* m, float* w,
                         float* werr, float* wgap, hwyblas_int* iblock,
                         hwyblas_int* indexw, float* gers, float* pivmin,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlarre_(const char* range, const hwyblas_int* n, double* vl,
                         double* vu, const hwyblas_int* il,
                         const hwyblas_int* iu, double* d, double* e,
                         double* e2, const double* rtol1, const double* rtol2,
                         const double* spltol, hwyblas_int* nsplit,
                         hwyblas_int* isplit, hwyblas_int* m, double* w,
                         double* werr, double* wgap, hwyblas_int* iblock,
                         hwyblas_int* indexw, double* gers, double* pivmin,
                         double* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slarrf_(const hwyblas_int* n, float* d, float* l, float* ld,
                         const hwyblas_int* clstrt, const hwyblas_int* clend,
                         float* w, float* wgap, float* werr,
                         const float* spdiam, const float* clgapl,
                         const float* clgapr, const float* pivmin, float* sigma,
                         float* dplus, float* lplus, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dlarrf_(const hwyblas_int* n, double* d, double* l, double* ld,
                         const hwyblas_int* clstrt, const hwyblas_int* clend,
                         double* w, double* wgap, double* werr,
                         const double* spdiam, const double* clgapl,
                         const double* clgapr, const double* pivmin,
                         double* sigma, double* dplus, double* lplus,
                         double* work, hwyblas_int* info);
HWYBLAS_API void slarrj_(const hwyblas_int* n, float* d, float* e2,
                         const hwyblas_int* ifirst, const hwyblas_int* ilast,
                         const float* rtol, const hwyblas_int* offset, float* w,
                         float* werr, float* work, hwyblas_int* iwork,
                         const float* pivmin, const float* spdiam,
                         hwyblas_int* info);
HWYBLAS_API void dlarrj_(const hwyblas_int* n, double* d, double* e2,
                         const hwyblas_int* ifirst, const hwyblas_int* ilast,
                         const double* rtol, const hwyblas_int* offset,
                         double* w, double* werr, double* work,
                         hwyblas_int* iwork, const double* pivmin,
                         const double* spdiam, hwyblas_int* info);
HWYBLAS_API void slarrk_(const hwyblas_int* n, const hwyblas_int* iw,
                         const float* gl, const float* gu, float* d, float* e2,
                         const float* pivmin, const float* reltol, float* w,
                         float* werr, hwyblas_int* info);
HWYBLAS_API void dlarrk_(const hwyblas_int* n, const hwyblas_int* iw,
                         const double* gl, const double* gu, double* d,
                         double* e2, const double* pivmin, const double* reltol,
                         double* w, double* werr, hwyblas_int* info);
HWYBLAS_API void slarrr_(const hwyblas_int* n, float* d, float* e,
                         hwyblas_int* info);
HWYBLAS_API void dlarrr_(const hwyblas_int* n, double* d, double* e,
                         hwyblas_int* info);
HWYBLAS_API void slarrv_(const hwyblas_int* n, const float* vl, const float* vu,
                         float* d, float* l, const float* pivmin,
                         hwyblas_int* isplit, const hwyblas_int* m,
                         const hwyblas_int* dol, const hwyblas_int* dou,
                         const float* minrgp, float* rtol1, float* rtol2,
                         float* w, float* werr, float* wgap,
                         hwyblas_int* iblock, hwyblas_int* indexw, float* gers,
                         float* z, const hwyblas_int* ldz, hwyblas_int* isuppz,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlarrv_(const hwyblas_int* n, const double* vl,
                         const double* vu, double* d, double* l,
                         const double* pivmin, hwyblas_int* isplit,
                         const hwyblas_int* m, const hwyblas_int* dol,
                         const hwyblas_int* dou, const double* minrgp,
                         double* rtol1, double* rtol2, double* w, double* werr,
                         double* wgap, hwyblas_int* iblock, hwyblas_int* indexw,
                         double* gers, double* z, const hwyblas_int* ldz,
                         hwyblas_int* isuppz, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void clarrv_(const hwyblas_int* n, const float* vl, const float* vu,
                         float* d, float* l, const float* pivmin,
                         hwyblas_int* isplit, const hwyblas_int* m,
                         const hwyblas_int* dol, const hwyblas_int* dou,
                         const float* minrgp, float* rtol1, float* rtol2,
                         float* w, float* werr, float* wgap,
                         hwyblas_int* iblock, hwyblas_int* indexw, float* gers,
                         float* z, const hwyblas_int* ldz, hwyblas_int* isuppz,
                         float* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zlarrv_(const hwyblas_int* n, const double* vl,
                         const double* vu, double* d, double* l,
                         const double* pivmin, hwyblas_int* isplit,
                         const hwyblas_int* m, const hwyblas_int* dol,
                         const hwyblas_int* dou, const double* minrgp,
                         double* rtol1, double* rtol2, double* w, double* werr,
                         double* wgap, hwyblas_int* iblock, hwyblas_int* indexw,
                         double* gers, double* z, const hwyblas_int* ldz,
                         hwyblas_int* isuppz, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);

/* Selected eigenvalues (w; m of them) of the symmetric tridiagonal
 * matrix (d, e) by bisection: all (range = 'A'), in (vl, vu] or the il-th
 * through iu-th, to the absolute accuracy abstol, ordered by block (order =
 * 'B': iblock, isplit; nsplit blocks) or overall ('E'). work: 4n, iwork: 3n
 * elements. ?laebz is its bisection kernel. */
HWYBLAS_API void sstebz_(const char* range, const char* order,
                         const hwyblas_int* n, const float* vl, const float* vu,
                         const hwyblas_int* il, const hwyblas_int* iu,
                         const float* abstol, float* d, float* e,
                         hwyblas_int* m, hwyblas_int* nsplit, float* w,
                         hwyblas_int* iblock, hwyblas_int* isplit, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dstebz_(const char* range, const char* order,
                         const hwyblas_int* n, const double* vl,
                         const double* vu, const hwyblas_int* il,
                         const hwyblas_int* iu, const double* abstol, double* d,
                         double* e, hwyblas_int* m, hwyblas_int* nsplit,
                         double* w, hwyblas_int* iblock, hwyblas_int* isplit,
                         double* work, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slaebz_(const hwyblas_int* ijob, const hwyblas_int* nitmax,
                         const hwyblas_int* n, const hwyblas_int* mmax,
                         const hwyblas_int* minp, const hwyblas_int* nbmin,
                         const float* abstol, const float* reltol,
                         const float* pivmin, float* d, float* e, float* e2,
                         hwyblas_int* nval, float* ab, float* c,
                         hwyblas_int* mout, hwyblas_int* nab, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlaebz_(const hwyblas_int* ijob, const hwyblas_int* nitmax,
                         const hwyblas_int* n, const hwyblas_int* mmax,
                         const hwyblas_int* minp, const hwyblas_int* nbmin,
                         const double* abstol, const double* reltol,
                         const double* pivmin, double* d, double* e, double* e2,
                         hwyblas_int* nval, double* ab, double* c,
                         hwyblas_int* mout, hwyblas_int* nab, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);

/* The eigenvectors (z) of the symmetric tridiagonal matrix (d, e) for
 * the m eigenvalues w, given by block (iblock, isplit as from ?stebz with
 * order = 'B'), by inverse iteration; ifail: those that did not converge.
 * work: 5n, iwork: n elements; real also for complex z. ?lagtf and ?lagts
 * factor and solve the shifted tridiagonal systems. */
HWYBLAS_API void sstein_(const hwyblas_int* n, float* d, float* e,
                         const hwyblas_int* m, float* w, hwyblas_int* iblock,
                         hwyblas_int* isplit, float* z, const hwyblas_int* ldz,
                         float* work, hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);
HWYBLAS_API void dstein_(const hwyblas_int* n, double* d, double* e,
                         const hwyblas_int* m, double* w, hwyblas_int* iblock,
                         hwyblas_int* isplit, double* z, const hwyblas_int* ldz,
                         double* work, hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);
HWYBLAS_API void cstein_(const hwyblas_int* n, float* d, float* e,
                         const hwyblas_int* m, float* w, hwyblas_int* iblock,
                         hwyblas_int* isplit, float* z, const hwyblas_int* ldz,
                         float* work, hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);
HWYBLAS_API void zstein_(const hwyblas_int* n, double* d, double* e,
                         const hwyblas_int* m, double* w, hwyblas_int* iblock,
                         hwyblas_int* isplit, double* z, const hwyblas_int* ldz,
                         double* work, hwyblas_int* iwork, hwyblas_int* ifail,
                         hwyblas_int* info);
HWYBLAS_API void slagtf_(const hwyblas_int* n, float* a, const float* lambda,
                         float* b, float* c, const float* tol, float* d,
                         hwyblas_int* in, hwyblas_int* info);
HWYBLAS_API void dlagtf_(const hwyblas_int* n, double* a, const double* lambda,
                         double* b, double* c, const double* tol, double* d,
                         hwyblas_int* in, hwyblas_int* info);
HWYBLAS_API void slagts_(const hwyblas_int* job, const hwyblas_int* n, float* a,
                         float* b, float* c, float* d, hwyblas_int* in,
                         float* y, float* tol, hwyblas_int* info);
HWYBLAS_API void dlagts_(const hwyblas_int* job, const hwyblas_int* n,
                         double* a, double* b, double* c, double* d,
                         hwyblas_int* in, double* y, double* tol,
                         hwyblas_int* info);

/* n random numbers (x) from the seed iseed (4 integers in [0, 4095],
 * iseed[3] odd; updated): ?larnv with the distribution idist (1: uniform
 * (0, 1), 2: uniform (-1, 1), 3: normal (0, 1); complex: 4: uniform in the
 * unit disk, 5: on the unit circle), ?laruv uniform (0, 1) (n <= 128). */
HWYBLAS_API void slarnv_(const hwyblas_int* idist, hwyblas_int* iseed,
                         const hwyblas_int* n, float* x);
HWYBLAS_API void dlarnv_(const hwyblas_int* idist, hwyblas_int* iseed,
                         const hwyblas_int* n, double* x);
HWYBLAS_API void clarnv_(const hwyblas_int* idist, hwyblas_int* iseed,
                         const hwyblas_int* n, float* x);
HWYBLAS_API void zlarnv_(const hwyblas_int* idist, hwyblas_int* iseed,
                         const hwyblas_int* n, double* x);
HWYBLAS_API void slaruv_(hwyblas_int* iseed, const hwyblas_int* n, float* x);
HWYBLAS_API void dlaruv_(hwyblas_int* iseed, const hwyblas_int* n, double* x);

/* ------------------------------------------------------------------------ */
/* Band and tridiagonal systems                                             */

/* A = P * L * U with partial pivoting for the m x n band matrix A with kl
 * subdiagonals and ku superdiagonals (?gbtrf blocked, ?gbtf2 unblocked).
 * Band storage: A(i, j) in ab(kl+ku+1+i-j, j), ldab >= 2*kl+ku+1 (the
 * first kl rows are for fill-in); on exit U in rows 1 to kl+ku+1 and the
 * multipliers below. info > 0: U(info, info) is exactly zero. */
HWYBLAS_API void sgbtrf_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         float* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void dgbtrf_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         double* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void cgbtrf_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         float* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void zgbtrf_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         double* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void sgbtf2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         float* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void dgbtf2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         double* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void cgbtf2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         float* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void zgbtf2_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         double* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                         hwyblas_int* info);

/* Solves op(A) * X = B (trans = 'N', 'T' or 'C') with the LU factorization
 * of ?gbtrf (?gbtrs), or A * X = B, factoring A first (?gbsv); B is n x
 * nrhs. Estimates the reciprocal condition number of A in the 1-norm
 * (norm = '1' or 'O') or the infinity norm ('I') from its LU
 * factorization and anorm (?gbcon); work: 3n (real) or 2n (complex),
 * iwork / rwork: n. */
HWYBLAS_API void sgbtrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const hwyblas_int* nrhs, const float* ab,
                         const hwyblas_int* ldab, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dgbtrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const hwyblas_int* nrhs, const double* ab,
                         const hwyblas_int* ldab, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cgbtrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const hwyblas_int* nrhs, const float* ab,
                         const hwyblas_int* ldab, const hwyblas_int* ipiv,
                         float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zgbtrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const hwyblas_int* nrhs, const double* ab,
                         const hwyblas_int* ldab, const hwyblas_int* ipiv,
                         double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void sgbsv_(const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const hwyblas_int* nrhs,
                        float* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                        float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dgbsv_(const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const hwyblas_int* nrhs,
                        double* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                        double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cgbsv_(const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const hwyblas_int* nrhs,
                        float* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                        float* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zgbsv_(const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const hwyblas_int* nrhs,
                        double* ab, const hwyblas_int* ldab, hwyblas_int* ipiv,
                        double* b, const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void sgbcon_(const char* norm, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const float* ab, const hwyblas_int* ldab,
                         const hwyblas_int* ipiv, const float* anorm,
                         float* rcond, float* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dgbcon_(const char* norm, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const double* ab, const hwyblas_int* ldab,
                         const hwyblas_int* ipiv, const double* anorm,
                         double* rcond, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void cgbcon_(const char* norm, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const float* ab, const hwyblas_int* ldab,
                         const hwyblas_int* ipiv, const float* anorm,
                         float* rcond, float* work, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgbcon_(const char* norm, const hwyblas_int* n,
                         const hwyblas_int* kl, const hwyblas_int* ku,
                         const double* ab, const hwyblas_int* ldab,
                         const hwyblas_int* ipiv, const double* anorm,
                         double* rcond, double* work, double* rwork,
                         hwyblas_int* info);

/* Solves op(A) * x = scale * b with the triangular band matrix A (kd
 * super- or subdiagonals, band storage), scaling against overflow as
 * ?latrs; cnorm: the column norms (computed for normin = 'N'). */
HWYBLAS_API void slatbs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const hwyblas_int* kd, const float* ab,
                         const hwyblas_int* ldab, float* x, float* scale,
                         float* cnorm, hwyblas_int* info);
HWYBLAS_API void dlatbs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const hwyblas_int* kd, const double* ab,
                         const hwyblas_int* ldab, double* x, double* scale,
                         double* cnorm, hwyblas_int* info);
HWYBLAS_API void clatbs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const hwyblas_int* kd, const float* ab,
                         const hwyblas_int* ldab, float* x, float* scale,
                         float* cnorm, hwyblas_int* info);
HWYBLAS_API void zlatbs_(const char* uplo, const char* trans, const char* diag,
                         const char* normin, const hwyblas_int* n,
                         const hwyblas_int* kd, const double* ab,
                         const hwyblas_int* ldab, double* x, double* scale,
                         double* cnorm, hwyblas_int* info);

/* A = U^H * U (uplo = 'U') or A = L * L^H ('L') for the symmetric
 * (Hermitian) positive definite band matrix A of order n with kd super- or
 * subdiagonals (?pbtrf blocked, ?pbtf2 unblocked; ?potf2: the unblocked
 * Cholesky factorization of a full matrix). Band storage: uplo = 'U',
 * A(i, j) in ab(kd+1+i-j, j); 'L', in ab(1+i-j, j); ldab >= kd+1. Solves
 * A * X = B with the factorization (?pbtrs), or factoring A first
 * (?pbsv). info > 0: the leading minor of order info is not positive. */
HWYBLAS_API void spbtrf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, float* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void dpbtrf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, double* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void cpbtrf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, float* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void zpbtrf_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, double* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void spbtf2_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, float* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void dpbtf2_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, double* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void cpbtf2_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, float* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void zpbtf2_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, double* ab,
                         const hwyblas_int* ldab, hwyblas_int* info);
HWYBLAS_API void spotf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void dpotf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void cpotf2_(const char* uplo, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void zpotf2_(const char* uplo, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* info);
HWYBLAS_API void spbtrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, const hwyblas_int* nrhs,
                         const float* ab, const hwyblas_int* ldab, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dpbtrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, const hwyblas_int* nrhs,
                         const double* ab, const hwyblas_int* ldab, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cpbtrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, const hwyblas_int* nrhs,
                         const float* ab, const hwyblas_int* ldab, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zpbtrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* kd, const hwyblas_int* nrhs,
                         const double* ab, const hwyblas_int* ldab, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void spbsv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* kd, const hwyblas_int* nrhs,
                        float* ab, const hwyblas_int* ldab, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dpbsv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* kd, const hwyblas_int* nrhs,
                        double* ab, const hwyblas_int* ldab, double* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cpbsv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* kd, const hwyblas_int* nrhs,
                        float* ab, const hwyblas_int* ldab, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zpbsv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* kd, const hwyblas_int* nrhs,
                        double* ab, const hwyblas_int* ldab, double* b,
                        const hwyblas_int* ldb, hwyblas_int* info);

/* The tridiagonal matrix A with subdiagonal dl (n-1), diagonal d (n) and
 * superdiagonal du (n-1): A = L * U with partial pivoting (?gttrf; du2:
 * the second superdiagonal of U, n-2); solves op(A) * X = B with it
 * (?gttrs; ?gtts2 without argument checks, itrans = 0, 1 or 2 for 'N',
 * 'T', 'C'), or factoring A first (?gtsv). info > 0: U(info, info) is
 * exactly zero. */
HWYBLAS_API void sgttrf_(const hwyblas_int* n, float* dl, float* d, float* du,
                         float* du2, hwyblas_int* ipiv, hwyblas_int* info);
HWYBLAS_API void dgttrf_(const hwyblas_int* n, double* dl, double* d,
                         double* du, double* du2, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void cgttrf_(const hwyblas_int* n, float* dl, float* d, float* du,
                         float* du2, hwyblas_int* ipiv, hwyblas_int* info);
HWYBLAS_API void zgttrf_(const hwyblas_int* n, double* dl, double* d,
                         double* du, double* du2, hwyblas_int* ipiv,
                         hwyblas_int* info);
HWYBLAS_API void sgttrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* dl,
                         const float* d, const float* du, const float* du2,
                         const hwyblas_int* ipiv, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dgttrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* dl,
                         const double* d, const double* du, const double* du2,
                         const hwyblas_int* ipiv, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cgttrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* dl,
                         const float* d, const float* du, const float* du2,
                         const hwyblas_int* ipiv, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zgttrs_(const char* trans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* dl,
                         const double* d, const double* du, const double* du2,
                         const hwyblas_int* ipiv, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void sgtts2_(const hwyblas_int* itrans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* dl,
                         const float* d, const float* du, const float* du2,
                         const hwyblas_int* ipiv, float* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void dgtts2_(const hwyblas_int* itrans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* dl,
                         const double* d, const double* du, const double* du2,
                         const hwyblas_int* ipiv, double* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void cgtts2_(const hwyblas_int* itrans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* dl,
                         const float* d, const float* du, const float* du2,
                         const hwyblas_int* ipiv, float* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void zgtts2_(const hwyblas_int* itrans, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* dl,
                         const double* d, const double* du, const double* du2,
                         const hwyblas_int* ipiv, double* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void sgtsv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        float* dl, float* d, float* du, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dgtsv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* dl, double* d, double* du, double* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cgtsv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        float* dl, float* d, float* du, float* b,
                        const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void zgtsv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* dl, double* d, double* du, double* b,
                        const hwyblas_int* ldb, hwyblas_int* info);

/* The symmetric (Hermitian) positive definite tridiagonal matrix A with
 * real diagonal d (n) and subdiagonal e (n-1): A = L * D * L^H (?pttrf);
 * solves A * X = B with it (?pttrs; ?ptts2 without argument checks; the
 * complex routines take uplo, 'U' if e is the superdiagonal of U = L^H,
 * or iuplo = 1 or 0), or factoring A first (?ptsv). info > 0: the leading
 * minor of order info is not positive. */
HWYBLAS_API void spttrf_(const hwyblas_int* n, float* d, float* e,
                         hwyblas_int* info);
HWYBLAS_API void dpttrf_(const hwyblas_int* n, double* d, double* e,
                         hwyblas_int* info);
HWYBLAS_API void cpttrf_(const hwyblas_int* n, float* d, float* e,
                         hwyblas_int* info);
HWYBLAS_API void zpttrf_(const hwyblas_int* n, double* d, double* e,
                         hwyblas_int* info);
HWYBLAS_API void spttrs_(const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* d, const float* e, float* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void dpttrs_(const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* d, const double* e, double* b,
                         const hwyblas_int* ldb, hwyblas_int* info);
HWYBLAS_API void cpttrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* d,
                         const float* e, float* b, const hwyblas_int* ldb,
                         hwyblas_int* info);
HWYBLAS_API void zpttrs_(const char* uplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* d,
                         const double* e, double* b, const hwyblas_int* ldb,
                         hwyblas_int* info);
HWYBLAS_API void sptts2_(const hwyblas_int* n, const hwyblas_int* nrhs,
                         const float* d, const float* e, float* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void dptts2_(const hwyblas_int* n, const hwyblas_int* nrhs,
                         const double* d, const double* e, double* b,
                         const hwyblas_int* ldb);
HWYBLAS_API void cptts2_(const hwyblas_int* iuplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const float* d,
                         const float* e, float* b, const hwyblas_int* ldb);
HWYBLAS_API void zptts2_(const hwyblas_int* iuplo, const hwyblas_int* n,
                         const hwyblas_int* nrhs, const double* d,
                         const double* e, double* b, const hwyblas_int* ldb);
HWYBLAS_API void sptsv_(const hwyblas_int* n, const hwyblas_int* nrhs, float* d,
                        float* e, float* b, const hwyblas_int* ldb,
                        hwyblas_int* info);
HWYBLAS_API void dptsv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* d, double* e, double* b, const hwyblas_int* ldb,
                        hwyblas_int* info);
HWYBLAS_API void cptsv_(const hwyblas_int* n, const hwyblas_int* nrhs, float* d,
                        float* e, float* b, const hwyblas_int* ldb,
                        hwyblas_int* info);
HWYBLAS_API void zptsv_(const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* d, double* e, double* b, const hwyblas_int* ldb,
                        hwyblas_int* info);

/* ------------------------------------------------------------------------ */
/* Least squares                                                            */

/* A = R * Q (?gerqf, blocked; ?gerq2 unblocked) for the m x n matrix A: R
 * in the last min(m, n) columns (upper triangle / trapezoid), Q as
 * min(m, n) reflectors in the rest of A and tau. lwork >= max(1, m)
 * (-1: query); ?gerq2's work has m elements. */
HWYBLAS_API void sgerqf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgerqf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgerqf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgerqf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sgerq2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dgerq2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cgerq2_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zgerq2_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);

/* Multiplies the m x n matrix C by Q of ?gerqf (k reflectors in the rows
 * of A, k x m for side = 'L', k x n for 'R'): Q*C, Q^H*C, C*Q or C*Q^H
 * (?ormrq / ?unmrq blocked, lwork >= max(1, n) or max(1, m), -1: query;
 * ?ormr2 / ?unmr2 unblocked). The complex routines conjugate rows of A
 * while they run and restore them. */
HWYBLAS_API void sormrq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dormrq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cunmrq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zunmrq_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sormr2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dormr2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cunmr2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const float* a,
                         const hwyblas_int* lda, const float* tau, float* c,
                         const hwyblas_int* ldc, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zunmr2_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const double* a,
                         const hwyblas_int* lda, const double* tau, double* c,
                         const hwyblas_int* ldc, double* work,
                         hwyblas_int* info);

/* RZ factorization of the m x n (m <= n) upper trapezoidal matrix A =
 * ( R 0 ) * Z (?tzrzf blocked, lwork >= max(1, m), -1: query; ?latrz
 * unblocked for A(1:m, [1:m, n-l+1:n]), work m), the reflectors of Z
 * (?larz: one, from the left or right; ?larzt: the triangular factor of
 * k of them (direct = 'B', storev = 'R'); ?larzb: the block reflector),
 * and multiplying by Z (?ormrz / ?unmrz blocked, ?ormr3 / ?unmr3
 * unblocked; l: the length of the reflectors' vectors). */
HWYBLAS_API void stzrzf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dtzrzf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void ctzrzf_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void ztzrzf_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void slatrz_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* l, float* a, const hwyblas_int* lda,
                         float* tau, float* work);
HWYBLAS_API void dlatrz_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* l, double* a,
                         const hwyblas_int* lda, double* tau, double* work);
HWYBLAS_API void clatrz_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* l, float* a, const hwyblas_int* lda,
                         float* tau, float* work);
HWYBLAS_API void zlatrz_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* l, double* a,
                         const hwyblas_int* lda, double* tau, double* work);
HWYBLAS_API void slarz_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* l,
                        const float* v, const hwyblas_int* incv,
                        const float* tau, float* c, const hwyblas_int* ldc,
                        float* work);
HWYBLAS_API void dlarz_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* l,
                        const double* v, const hwyblas_int* incv,
                        const double* tau, double* c, const hwyblas_int* ldc,
                        double* work);
HWYBLAS_API void clarz_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* l,
                        const float* v, const hwyblas_int* incv,
                        const float* tau, float* c, const hwyblas_int* ldc,
                        float* work);
HWYBLAS_API void zlarz_(const char* side, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* l,
                        const double* v, const hwyblas_int* incv,
                        const double* tau, double* c, const hwyblas_int* ldc,
                        double* work);
HWYBLAS_API void slarzt_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k, float* v,
                         const hwyblas_int* ldv, const float* tau, float* t,
                         const hwyblas_int* ldt);
HWYBLAS_API void dlarzt_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k, double* v,
                         const hwyblas_int* ldv, const double* tau, double* t,
                         const hwyblas_int* ldt);
HWYBLAS_API void clarzt_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k, float* v,
                         const hwyblas_int* ldv, const float* tau, float* t,
                         const hwyblas_int* ldt);
HWYBLAS_API void zlarzt_(const char* direct, const char* storev,
                         const hwyblas_int* n, const hwyblas_int* k, double* v,
                         const hwyblas_int* ldv, const double* tau, double* t,
                         const hwyblas_int* ldt);
HWYBLAS_API void slarzb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l, float* v,
                         const hwyblas_int* ldv, float* t,
                         const hwyblas_int* ldt, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void dlarzb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l, double* v,
                         const hwyblas_int* ldv, double* t,
                         const hwyblas_int* ldt, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void clarzb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l, float* v,
                         const hwyblas_int* ldv, float* t,
                         const hwyblas_int* ldt, float* c,
                         const hwyblas_int* ldc, float* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void zlarzb_(const char* side, const char* trans,
                         const char* direct, const char* storev,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l, double* v,
                         const hwyblas_int* ldv, double* t,
                         const hwyblas_int* ldt, double* c,
                         const hwyblas_int* ldc, double* work,
                         const hwyblas_int* ldwork);
HWYBLAS_API void sormrz_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dormrz_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cunmrz_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zunmrz_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sormr3_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, hwyblas_int* info);
HWYBLAS_API void dormr3_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, hwyblas_int* info);
HWYBLAS_API void cunmr3_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, hwyblas_int* info);
HWYBLAS_API void zunmr3_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* k, const hwyblas_int* l,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, hwyblas_int* info);

/* Minimum norm least squares solutions of A * X = B or A^H * X = B
 * (trans = 'N' or 'T' / 'C') for the m x n matrix A of full rank, by QR
 * or LQ factorization (?gels), and for A of any rank (rank: its effective
 * rank, with singular values s(i) <= rcond*s(1) as zero): by the SVD
 * (?gelss), by the SVD with divide and conquer (?gelsd, with ?lalsd for
 * the bidiagonal problem and ?lalsa / ?lals0 applying its singular
 * vectors), by a complete orthogonal factorization (?gelsy, with column
 * pivots jpvt and the incremental condition estimator ?laic1). b has
 * max(m, n) rows (ldb). lwork = -1: query (?gelsd also returns the minimal
 * liwork in iwork[0] and, complex, lrwork in rwork[0]); rwork (complex):
 * 5*min(m, n) (?gelss), 2*n (?gelsy). info > 0: A is not of full rank
 * (?gels) or the SVD did not converge. */
HWYBLAS_API void sgels_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgels_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* a, const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgels_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* nrhs, float* a,
                        const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgels_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* nrhs,
                        double* a, const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sgelss_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, float* s, const float* rcond,
                         hwyblas_int* rank, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgelss_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* s, const double* rcond,
                         hwyblas_int* rank, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgelss_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, float* s, const float* rcond,
                         hwyblas_int* rank, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgelss_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* s, const double* rcond,
                         hwyblas_int* rank, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* info);
HWYBLAS_API void sgelsd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, float* s, const float* rcond,
                         hwyblas_int* rank, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void dgelsd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* s, const double* rcond,
                         hwyblas_int* rank, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void cgelsd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, float* s, const float* rcond,
                         hwyblas_int* rank, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zgelsd_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* s, const double* rcond,
                         hwyblas_int* rank, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void sgelsy_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, hwyblas_int* jpvt,
                         const float* rcond, hwyblas_int* rank, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgelsy_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* jpvt,
                         const double* rcond, hwyblas_int* rank, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgelsy_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, float* a,
                         const hwyblas_int* lda, float* b,
                         const hwyblas_int* ldb, hwyblas_int* jpvt,
                         const float* rcond, hwyblas_int* rank, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgelsy_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* nrhs, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, hwyblas_int* jpvt,
                         const double* rcond, hwyblas_int* rank, double* work,
                         const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* info);
HWYBLAS_API void slaic1_(const hwyblas_int* job, const hwyblas_int* j,
                         const float* x, const float* sest, const float* w,
                         const float* gamma, float* sestpr, float* s, float* c);
HWYBLAS_API void dlaic1_(const hwyblas_int* job, const hwyblas_int* j,
                         const double* x, const double* sest, const double* w,
                         const double* gamma, double* sestpr, double* s,
                         double* c);
HWYBLAS_API void claic1_(const hwyblas_int* job, const hwyblas_int* j,
                         const float* x, const float* sest, const float* w,
                         const float* gamma, float* sestpr, float* s, float* c);
HWYBLAS_API void zlaic1_(const hwyblas_int* job, const hwyblas_int* j,
                         const double* x, const double* sest, const double* w,
                         const double* gamma, double* sestpr, double* s,
                         double* c);
HWYBLAS_API void slalsd_(const char* uplo, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* d, float* e, float* b, const hwyblas_int* ldb,
                         const float* rcond, hwyblas_int* rank, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlalsd_(const char* uplo, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* d, double* e, double* b,
                         const hwyblas_int* ldb, const double* rcond,
                         hwyblas_int* rank, double* work, hwyblas_int* iwork,
                         hwyblas_int* info);
HWYBLAS_API void clalsd_(const char* uplo, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* d, float* e, float* b, const hwyblas_int* ldb,
                         const float* rcond, hwyblas_int* rank, float* work,
                         float* rwork, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zlalsd_(const char* uplo, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* d, double* e, double* b,
                         const hwyblas_int* ldb, const double* rcond,
                         hwyblas_int* rank, double* work, double* rwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slalsa_(const hwyblas_int* icompq, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* b, const hwyblas_int* ldb, float* bx,
                         const hwyblas_int* ldbx, float* u,
                         const hwyblas_int* ldu, float* vt, hwyblas_int* k,
                         float* difl, float* difr, float* z, float* poles,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, hwyblas_int* perm,
                         float* givnum, float* c, float* s, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dlalsa_(const hwyblas_int* icompq, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* b, const hwyblas_int* ldb, double* bx,
                         const hwyblas_int* ldbx, double* u,
                         const hwyblas_int* ldu, double* vt, hwyblas_int* k,
                         double* difl, double* difr, double* z, double* poles,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, hwyblas_int* perm,
                         double* givnum, double* c, double* s, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void clalsa_(const hwyblas_int* icompq, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         float* b, const hwyblas_int* ldb, float* bx,
                         const hwyblas_int* ldbx, float* u,
                         const hwyblas_int* ldu, float* vt, hwyblas_int* k,
                         float* difl, float* difr, float* z, float* poles,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, hwyblas_int* perm,
                         float* givnum, float* c, float* s, float* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void zlalsa_(const hwyblas_int* icompq, const hwyblas_int* smlsiz,
                         const hwyblas_int* n, const hwyblas_int* nrhs,
                         double* b, const hwyblas_int* ldb, double* bx,
                         const hwyblas_int* ldbx, double* u,
                         const hwyblas_int* ldu, double* vt, hwyblas_int* k,
                         double* difl, double* difr, double* z, double* poles,
                         hwyblas_int* givptr, hwyblas_int* givcol,
                         const hwyblas_int* ldgcol, hwyblas_int* perm,
                         double* givnum, double* c, double* s, double* work,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void slals0_(
    const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr,
    const hwyblas_int* sqre, const hwyblas_int* nrhs, float* b,
    const hwyblas_int* ldb, float* bx, const hwyblas_int* ldbx,
    hwyblas_int* perm, const hwyblas_int* givptr, hwyblas_int* givcol,
    const hwyblas_int* ldgcol, float* givnum, const hwyblas_int* ldgnum,
    float* poles, float* difl, float* difr, float* z, const hwyblas_int* k,
    const float* c, const float* s, float* work, hwyblas_int* info);
HWYBLAS_API void dlals0_(
    const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr,
    const hwyblas_int* sqre, const hwyblas_int* nrhs, double* b,
    const hwyblas_int* ldb, double* bx, const hwyblas_int* ldbx,
    hwyblas_int* perm, const hwyblas_int* givptr, hwyblas_int* givcol,
    const hwyblas_int* ldgcol, double* givnum, const hwyblas_int* ldgnum,
    double* poles, double* difl, double* difr, double* z, const hwyblas_int* k,
    const double* c, const double* s, double* work, hwyblas_int* info);
HWYBLAS_API void clals0_(
    const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr,
    const hwyblas_int* sqre, const hwyblas_int* nrhs, float* b,
    const hwyblas_int* ldb, float* bx, const hwyblas_int* ldbx,
    hwyblas_int* perm, const hwyblas_int* givptr, hwyblas_int* givcol,
    const hwyblas_int* ldgcol, float* givnum, const hwyblas_int* ldgnum,
    float* poles, float* difl, float* difr, float* z, const hwyblas_int* k,
    const float* c, const float* s, float* work, hwyblas_int* info);
HWYBLAS_API void zlals0_(
    const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr,
    const hwyblas_int* sqre, const hwyblas_int* nrhs, double* b,
    const hwyblas_int* ldb, double* bx, const hwyblas_int* ldbx,
    hwyblas_int* perm, const hwyblas_int* givptr, hwyblas_int* givcol,
    const hwyblas_int* ldgcol, double* givnum, const hwyblas_int* ldgnum,
    double* poles, double* difl, double* difr, double* z, const hwyblas_int* k,
    const double* c, const double* s, double* work, hwyblas_int* info);

/* Generalized least squares: ?gglse minimizes ||c - A*x|| subject to B*x
 * = d (A m x n, B p x n, p <= n <= m+p); ?ggglm minimizes ||y|| subject
 * to d = A*x + B*y (A n x m, B n x p, m <= n <= m+p); with the generalized
 * QR (?ggqrf: A = Q*R, B = Q*T*Z) and RQ (?ggrqf: A = R*Q, B = Z*T*Q)
 * factorizations. lwork >= max(1, m+n+p) (?gglse, ?ggglm) or max(1, n, m,
 * p) (-1: query). info = 1 or 2: a triangular factor is singular. */
HWYBLAS_API void sgglse_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* p, float* a, const hwyblas_int* lda,
                         float* b, const hwyblas_int* ldb, float* c, float* d,
                         float* x, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dgglse_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* p, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* c, double* d,
                         double* x, double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cgglse_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* p, float* a, const hwyblas_int* lda,
                         float* b, const hwyblas_int* ldb, float* c, float* d,
                         float* x, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zgglse_(const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* p, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* c, double* d,
                         double* x, double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sggglm_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, float* a, const hwyblas_int* lda,
                         float* b, const hwyblas_int* ldb, float* d, float* x,
                         float* y, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dggglm_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* d, double* x,
                         double* y, double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cggglm_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, float* a, const hwyblas_int* lda,
                         float* b, const hwyblas_int* ldb, float* d, float* x,
                         float* y, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zggglm_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, double* a,
                         const hwyblas_int* lda, double* b,
                         const hwyblas_int* ldb, double* d, double* x,
                         double* y, double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sggqrf_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, float* a, const hwyblas_int* lda,
                         float* taua, float* b, const hwyblas_int* ldb,
                         float* taub, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dggqrf_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, double* a,
                         const hwyblas_int* lda, double* taua, double* b,
                         const hwyblas_int* ldb, double* taub, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cggqrf_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, float* a, const hwyblas_int* lda,
                         float* taua, float* b, const hwyblas_int* ldb,
                         float* taub, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zggqrf_(const hwyblas_int* n, const hwyblas_int* m,
                         const hwyblas_int* p, double* a,
                         const hwyblas_int* lda, double* taua, double* b,
                         const hwyblas_int* ldb, double* taub, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sggrqf_(const hwyblas_int* m, const hwyblas_int* p,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* taua, float* b, const hwyblas_int* ldb,
                         float* taub, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dggrqf_(const hwyblas_int* m, const hwyblas_int* p,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* taua, double* b,
                         const hwyblas_int* ldb, double* taub, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cggrqf_(const hwyblas_int* m, const hwyblas_int* p,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* taua, float* b, const hwyblas_int* ldb,
                         float* taub, float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zggrqf_(const hwyblas_int* m, const hwyblas_int* p,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, double* taua, double* b,
                         const hwyblas_int* ldb, double* taub, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);

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

/* ------------------------------------------------------------------------ */
/* Nonsymmetric eigenproblem: Schur factorization                           */

/* The SELECT functions of ?gees: Fortran LOGICAL FUNCTIONs (nonzero is
 * true) of the eigenvalue wr + i*wi (real routines) or w (complex
 * routines; w points to the real and the imaginary part). */
typedef hwyblas_int (*hwyblas_select2_float)(const float* wr, const float* wi);
typedef hwyblas_int (*hwyblas_select2_double)(const double* wr,
                                              const double* wi);
typedef hwyblas_int (*hwyblas_select1_float)(const float* w);
typedef hwyblas_int (*hwyblas_select1_double)(const double* w);

/* The Schur factorization A = VS * T * VS^H of the n x n matrix A: the
 * eigenvalues (wr + i*wi, or w), the Schur form T in A (real routines:
 * quasi-triangular, with 2 x 2 blocks for complex conjugate pairs)
 * and for jobvs = 'V' the Schur vectors VS. sort = 'S' moves the
 * eigenvalues for which select is true to the top left (sdim: their
 * number). work: lwork >= max(1, 3n) (real) or max(1, 2n) (complex)
 * elements (lwork = -1: query); rwork: n; bwork: n (sort = 'S').
 * info = i <= n: the QR algorithm failed; n+1: reordering failed;
 * n+2: rounding changed the selected eigenvalues (real routines). */
HWYBLAS_API void sgees_(const char* jobvs, const char* sort,
                        hwyblas_select2_float select, const hwyblas_int* n,
                        float* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        float* wr, float* wi, float* vs,
                        const hwyblas_int* ldvs, float* work,
                        const hwyblas_int* lwork, hwyblas_int* bwork,
                        hwyblas_int* info);
HWYBLAS_API void dgees_(const char* jobvs, const char* sort,
                        hwyblas_select2_double select, const hwyblas_int* n,
                        double* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        double* wr, double* wi, double* vs,
                        const hwyblas_int* ldvs, double* work,
                        const hwyblas_int* lwork, hwyblas_int* bwork,
                        hwyblas_int* info);
HWYBLAS_API void cgees_(const char* jobvs, const char* sort,
                        hwyblas_select1_float select, const hwyblas_int* n,
                        float* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        float* w, float* vs, const hwyblas_int* ldvs,
                        float* work, const hwyblas_int* lwork, float* rwork,
                        hwyblas_int* bwork, hwyblas_int* info);
HWYBLAS_API void zgees_(const char* jobvs, const char* sort,
                        hwyblas_select1_double select, const hwyblas_int* n,
                        double* a, const hwyblas_int* lda, hwyblas_int* sdim,
                        double* w, double* vs, const hwyblas_int* ldvs,
                        double* work, const hwyblas_int* lwork, double* rwork,
                        hwyblas_int* bwork, hwyblas_int* info);

/* The eigenvalues (wr + i*wi, or w) of the n x n matrix A
 * (destroyed) and for jobvl / jobvr = 'V' its left / right eigenvectors
 * (vl, vr; real routines: a complex conjugate pair j, j+1 has the vectors
 * v(:,j) +- i*v(:,j+1)), normalized to norm 1 with a real largest
 * component. work: lwork >= max(1, 3n), 4n with eigenvectors (real), or
 * max(1, 2n) (complex) elements (lwork = -1: query); rwork: 2n. info > 0:
 * the QR algorithm failed, eigenvalues info+1:n have converged. */
HWYBLAS_API void sgeev_(const char* jobvl, const char* jobvr,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* wr, float* wi, float* vl,
                        const hwyblas_int* ldvl, float* vr,
                        const hwyblas_int* ldvr, float* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgeev_(const char* jobvl, const char* jobvr,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* wr, double* wi, double* vl,
                        const hwyblas_int* ldvl, double* vr,
                        const hwyblas_int* ldvr, double* work,
                        const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgeev_(const char* jobvl, const char* jobvr,
                        const hwyblas_int* n, float* a, const hwyblas_int* lda,
                        float* w, float* vl, const hwyblas_int* ldvl, float* vr,
                        const hwyblas_int* ldvr, float* work,
                        const hwyblas_int* lwork, float* rwork,
                        hwyblas_int* info);
HWYBLAS_API void zgeev_(const char* jobvl, const char* jobvr,
                        const hwyblas_int* n, double* a, const hwyblas_int* lda,
                        double* w, double* vl, const hwyblas_int* ldvl,
                        double* vr, const hwyblas_int* ldvr, double* work,
                        const hwyblas_int* lwork, double* rwork,
                        hwyblas_int* info);

/* ?geev with balancing (balanc = 'N', 'P', 'S' or 'B'; ilo,
 * ihi and scale as for ?gebal; abnrm: the 1-norm of the balanced matrix)
 * and reciprocal condition numbers of the eigenvalues (sense = 'E',
 * rconde; needs both eigenvector sets) and/or right eigenvectors ('V',
 * rcondv; 'B': both). work: lwork >= max(1, 2n), 3n with eigenvectors,
 * n*n + 6n for sense = 'V' or 'B' (real); max(1, 2n), n*n + 2n for 'V' or
 * 'B' (complex); iwork: 2n - 2; rwork: 2n. */
HWYBLAS_API void sgeevx_(
    const char* balanc, const char* jobvl, const char* jobvr, const char* sense,
    const hwyblas_int* n, float* a, const hwyblas_int* lda, float* wr,
    float* wi, float* vl, const hwyblas_int* ldvl, float* vr,
    const hwyblas_int* ldvr, hwyblas_int* ilo, hwyblas_int* ihi, float* scale,
    float* abnrm, float* rconde, float* rcondv, float* work,
    const hwyblas_int* lwork, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dgeevx_(
    const char* balanc, const char* jobvl, const char* jobvr, const char* sense,
    const hwyblas_int* n, double* a, const hwyblas_int* lda, double* wr,
    double* wi, double* vl, const hwyblas_int* ldvl, double* vr,
    const hwyblas_int* ldvr, hwyblas_int* ilo, hwyblas_int* ihi, double* scale,
    double* abnrm, double* rconde, double* rcondv, double* work,
    const hwyblas_int* lwork, hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void cgeevx_(const char* balanc, const char* jobvl,
                         const char* jobvr, const char* sense,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         float* w, float* vl, const hwyblas_int* ldvl,
                         float* vr, const hwyblas_int* ldvr, hwyblas_int* ilo,
                         hwyblas_int* ihi, float* scale, float* abnrm,
                         float* rconde, float* rcondv, float* work,
                         const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void zgeevx_(
    const char* balanc, const char* jobvl, const char* jobvr, const char* sense,
    const hwyblas_int* n, double* a, const hwyblas_int* lda, double* w,
    double* vl, const hwyblas_int* ldvl, double* vr, const hwyblas_int* ldvr,
    hwyblas_int* ilo, hwyblas_int* ihi, double* scale, double* abnrm,
    double* rconde, double* rcondv, double* work, const hwyblas_int* lwork,
    double* rwork, hwyblas_int* info);

/* ?gees with the reciprocal condition numbers of the average of
 * the selected eigenvalues (sense = 'E', rconde) and/or of their right
 * invariant subspace ('V', rcondv; 'B': both; needs sort = 'S'). work:
 * lwork >= max(1, 3n) (real), max(1, 2n) (complex), plus 2*sdim*(n-sdim)
 * for sense != 'N' (n*n/2 suffices); iwork: liwork >= sdim*(n-sdim) for
 * 'V' or 'B' (lwork, liwork = -1: query). info = -16 / -18 (real), -15
 * (complex): workspace too small for sdim. */
HWYBLAS_API void sgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select2_float select, const char* sense,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* sdim, float* wr, float* wi, float* vs,
                         const hwyblas_int* ldvs, float* rconde, float* rcondv,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* iwork, const hwyblas_int* liwork,
                         hwyblas_int* bwork, hwyblas_int* info);
HWYBLAS_API void dgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select2_double select, const char* sense,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* sdim, double* wr,
                         double* wi, double* vs, const hwyblas_int* ldvs,
                         double* rconde, double* rcondv, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* bwork,
                         hwyblas_int* info);
HWYBLAS_API void cgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select1_float select, const char* sense,
                         const hwyblas_int* n, float* a, const hwyblas_int* lda,
                         hwyblas_int* sdim, float* w, float* vs,
                         const hwyblas_int* ldvs, float* rconde, float* rcondv,
                         float* work, const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* bwork, hwyblas_int* info);
HWYBLAS_API void zgeesx_(const char* jobvs, const char* sort,
                         hwyblas_select1_double select, const char* sense,
                         const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* sdim, double* w,
                         double* vs, const hwyblas_int* ldvs, double* rconde,
                         double* rcondv, double* work, const hwyblas_int* lwork,
                         double* rwork, hwyblas_int* bwork, hwyblas_int* info);

/* Balancing (?gebal: job = 'N', 'P' permute, 'S' scale or 'B' both;
 * scale: n elements) and back-transformation of eigenvectors (?gebak,
 * side = 'R' or 'L'). */
HWYBLAS_API void sgebal_(const char* job, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ilo,
                         hwyblas_int* ihi, float* scale, hwyblas_int* info);
HWYBLAS_API void dgebal_(const char* job, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ilo,
                         hwyblas_int* ihi, double* scale, hwyblas_int* info);
HWYBLAS_API void cgebal_(const char* job, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* ilo,
                         hwyblas_int* ihi, float* scale, hwyblas_int* info);
HWYBLAS_API void zgebal_(const char* job, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* ilo,
                         hwyblas_int* ihi, double* scale, hwyblas_int* info);
HWYBLAS_API void sgebak_(const char* job, const char* side,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, const float* scale,
                         const hwyblas_int* m, float* v, const hwyblas_int* ldv,
                         hwyblas_int* info);
HWYBLAS_API void dgebak_(const char* job, const char* side,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, const double* scale,
                         const hwyblas_int* m, double* v,
                         const hwyblas_int* ldv, hwyblas_int* info);
HWYBLAS_API void cgebak_(const char* job, const char* side,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, const float* scale,
                         const hwyblas_int* m, float* v, const hwyblas_int* ldv,
                         hwyblas_int* info);
HWYBLAS_API void zgebak_(const char* job, const char* side,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, const double* scale,
                         const hwyblas_int* m, double* v,
                         const hwyblas_int* ldv, hwyblas_int* info);

/* Reduction to upper Hessenberg form Q^H * A * Q = H in rows and columns
 * ilo:ihi: ?gehrd (blocked; lwork = -1: query), ?gehd2 (unblocked; work:
 * n), ?lahr2 (one block); and Q: ?orghr / ?unghr (generate),
 * ?ormhr / ?unmhr (apply). */
HWYBLAS_API void sgehrd_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dgehrd_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void cgehrd_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zgehrd_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void sgehd2_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dgehd2_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);
HWYBLAS_API void cgehd2_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* a,
                         const hwyblas_int* lda, float* tau, float* work,
                         hwyblas_int* info);
HWYBLAS_API void zgehd2_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* a,
                         const hwyblas_int* lda, double* tau, double* work,
                         hwyblas_int* info);
HWYBLAS_API void slahr2_(const hwyblas_int* n, const hwyblas_int* k,
                         const hwyblas_int* nb, float* a,
                         const hwyblas_int* lda, float* tau, float* t,
                         const hwyblas_int* ldt, float* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void dlahr2_(const hwyblas_int* n, const hwyblas_int* k,
                         const hwyblas_int* nb, double* a,
                         const hwyblas_int* lda, double* tau, double* t,
                         const hwyblas_int* ldt, double* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void clahr2_(const hwyblas_int* n, const hwyblas_int* k,
                         const hwyblas_int* nb, float* a,
                         const hwyblas_int* lda, float* tau, float* t,
                         const hwyblas_int* ldt, float* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void zlahr2_(const hwyblas_int* n, const hwyblas_int* k,
                         const hwyblas_int* nb, double* a,
                         const hwyblas_int* lda, double* tau, double* t,
                         const hwyblas_int* ldt, double* y,
                         const hwyblas_int* ldy);
HWYBLAS_API void sorghr_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* a,
                         const hwyblas_int* lda, const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dorghr_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cunghr_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* a,
                         const hwyblas_int* lda, const float* tau, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zunghr_(const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* a,
                         const hwyblas_int* lda, const double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void sormhr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* ilo, const hwyblas_int* ihi,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void dormhr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* ilo, const hwyblas_int* ihi,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void cunmhr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* ilo, const hwyblas_int* ihi,
                         const float* a, const hwyblas_int* lda,
                         const float* tau, float* c, const hwyblas_int* ldc,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info);
HWYBLAS_API void zunmhr_(const char* side, const char* trans,
                         const hwyblas_int* m, const hwyblas_int* n,
                         const hwyblas_int* ilo, const hwyblas_int* ihi,
                         const double* a, const hwyblas_int* lda,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info);

/* The eigenvalues and the Schur form (job = 'S'; 'E': eigenvalues only)
 * of an upper Hessenberg matrix H, and Schur vectors (compz = 'I': Z =
 * Q, 'V': Z := Z * Q, 'N': none): ?hseqr (lwork = -1: query). wantt
 * and wantz of the steps are Fortran LOGICALs: ?laqr0 / ?laqr4 (the
 * multishift QR algorithm with aggressive early deflation), ?laqr2 /
 * ?laqr3 (aggressive early deflation), ?laqr5 (a multishift QR sweep),
 * ?laqr1 (the first column of the shift polynomial), ?lahqr (the
 * double-shift QR algorithm for small matrices) and ?lanv2 (the Schur
 * factorization of a real 2 x 2 matrix). */
HWYBLAS_API void shseqr_(const char* job, const char* compz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* wr, float* wi, float* z,
                         const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dhseqr_(const char* job, const char* compz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* wr, double* wi,
                         double* z, const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void chseqr_(const char* job, const char* compz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* w, float* z,
                         const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zhseqr_(const char* job, const char* compz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* w, double* z,
                         const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void slaqr0_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* wr, float* wi,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         float* z, const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dlaqr0_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* wr, double* wi,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         double* z, const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void claqr0_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* w,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         float* z, const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zlaqr0_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* w,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         double* z, const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void slaqr4_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* wr, float* wi,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         float* z, const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void dlaqr4_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* wr, double* wi,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         double* z, const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void claqr4_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* w,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         float* z, const hwyblas_int* ldz, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void zlaqr4_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* w,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         double* z, const hwyblas_int* ldz, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void slaqr2_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    float* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, float* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, float* sr, float* si, float* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, float* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, float* wv, const hwyblas_int* ldwv, float* work,
    const hwyblas_int* lwork);
HWYBLAS_API void dlaqr2_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    double* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, double* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, double* sr, double* si, double* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, double* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, double* wv, const hwyblas_int* ldwv, double* work,
    const hwyblas_int* lwork);
HWYBLAS_API void claqr2_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    float* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, float* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, float* sh, float* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, float* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, float* wv, const hwyblas_int* ldwv, float* work,
    const hwyblas_int* lwork);
HWYBLAS_API void zlaqr2_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    double* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, double* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, double* sh, double* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, double* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, double* wv, const hwyblas_int* ldwv, double* work,
    const hwyblas_int* lwork);
HWYBLAS_API void slaqr3_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    float* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, float* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, float* sr, float* si, float* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, float* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, float* wv, const hwyblas_int* ldwv, float* work,
    const hwyblas_int* lwork);
HWYBLAS_API void dlaqr3_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    double* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, double* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, double* sr, double* si, double* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, double* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, double* wv, const hwyblas_int* ldwv, double* work,
    const hwyblas_int* lwork);
HWYBLAS_API void claqr3_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    float* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, float* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, float* sh, float* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, float* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, float* wv, const hwyblas_int* ldwv, float* work,
    const hwyblas_int* lwork);
HWYBLAS_API void zlaqr3_(
    const hwyblas_int* wantt, const hwyblas_int* wantz, const hwyblas_int* n,
    const hwyblas_int* ktop, const hwyblas_int* kbot, const hwyblas_int* nw,
    double* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, double* z, const hwyblas_int* ldz, hwyblas_int* ns,
    hwyblas_int* nd, double* sh, double* v, const hwyblas_int* ldv,
    const hwyblas_int* nh, double* t, const hwyblas_int* ldt,
    const hwyblas_int* nv, double* wv, const hwyblas_int* ldwv, double* work,
    const hwyblas_int* lwork);
HWYBLAS_API void slaqr5_(
    const hwyblas_int* wantt, const hwyblas_int* wantz,
    const hwyblas_int* kacc22, const hwyblas_int* n, const hwyblas_int* ktop,
    const hwyblas_int* kbot, const hwyblas_int* nshfts, float* sr, float* si,
    float* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, float* z, const hwyblas_int* ldz, float* v,
    const hwyblas_int* ldv, float* u, const hwyblas_int* ldu,
    const hwyblas_int* nv, float* wv, const hwyblas_int* ldwv,
    const hwyblas_int* nh, float* wh, const hwyblas_int* ldwh);
HWYBLAS_API void dlaqr5_(
    const hwyblas_int* wantt, const hwyblas_int* wantz,
    const hwyblas_int* kacc22, const hwyblas_int* n, const hwyblas_int* ktop,
    const hwyblas_int* kbot, const hwyblas_int* nshfts, double* sr, double* si,
    double* h, const hwyblas_int* ldh, const hwyblas_int* iloz,
    const hwyblas_int* ihiz, double* z, const hwyblas_int* ldz, double* v,
    const hwyblas_int* ldv, double* u, const hwyblas_int* ldu,
    const hwyblas_int* nv, double* wv, const hwyblas_int* ldwv,
    const hwyblas_int* nh, double* wh, const hwyblas_int* ldwh);
HWYBLAS_API void claqr5_(
    const hwyblas_int* wantt, const hwyblas_int* wantz,
    const hwyblas_int* kacc22, const hwyblas_int* n, const hwyblas_int* ktop,
    const hwyblas_int* kbot, const hwyblas_int* nshfts, float* s, float* h,
    const hwyblas_int* ldh, const hwyblas_int* iloz, const hwyblas_int* ihiz,
    float* z, const hwyblas_int* ldz, float* v, const hwyblas_int* ldv,
    float* u, const hwyblas_int* ldu, const hwyblas_int* nv, float* wv,
    const hwyblas_int* ldwv, const hwyblas_int* nh, float* wh,
    const hwyblas_int* ldwh);
HWYBLAS_API void zlaqr5_(
    const hwyblas_int* wantt, const hwyblas_int* wantz,
    const hwyblas_int* kacc22, const hwyblas_int* n, const hwyblas_int* ktop,
    const hwyblas_int* kbot, const hwyblas_int* nshfts, double* s, double* h,
    const hwyblas_int* ldh, const hwyblas_int* iloz, const hwyblas_int* ihiz,
    double* z, const hwyblas_int* ldz, double* v, const hwyblas_int* ldv,
    double* u, const hwyblas_int* ldu, const hwyblas_int* nv, double* wv,
    const hwyblas_int* ldwv, const hwyblas_int* nh, double* wh,
    const hwyblas_int* ldwh);
HWYBLAS_API void slaqr1_(const hwyblas_int* n, const float* h,
                         const hwyblas_int* ldh, const float* sr1,
                         const float* si1, const float* sr2, const float* si2,
                         float* v);
HWYBLAS_API void dlaqr1_(const hwyblas_int* n, const double* h,
                         const hwyblas_int* ldh, const double* sr1,
                         const double* si1, const double* sr2,
                         const double* si2, double* v);
HWYBLAS_API void claqr1_(const hwyblas_int* n, const float* h,
                         const hwyblas_int* ldh, const float* s1,
                         const float* s2, float* v);
HWYBLAS_API void zlaqr1_(const hwyblas_int* n, const double* h,
                         const hwyblas_int* ldh, const double* s1,
                         const double* s2, double* v);
HWYBLAS_API void slahqr_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* wr, float* wi,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         float* z, const hwyblas_int* ldz, hwyblas_int* info);
HWYBLAS_API void dlahqr_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* wr, double* wi,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         double* z, const hwyblas_int* ldz, hwyblas_int* info);
HWYBLAS_API void clahqr_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, float* h,
                         const hwyblas_int* ldh, float* w,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         float* z, const hwyblas_int* ldz, hwyblas_int* info);
HWYBLAS_API void zlahqr_(const hwyblas_int* wantt, const hwyblas_int* wantz,
                         const hwyblas_int* n, const hwyblas_int* ilo,
                         const hwyblas_int* ihi, double* h,
                         const hwyblas_int* ldh, double* w,
                         const hwyblas_int* iloz, const hwyblas_int* ihiz,
                         double* z, const hwyblas_int* ldz, hwyblas_int* info);
HWYBLAS_API void slanv2_(float* a, float* b, float* c, float* d, float* rt1r,
                         float* rt1i, float* rt2r, float* rt2i, float* cs,
                         float* sn);
HWYBLAS_API void dlanv2_(double* a, double* b, double* c, double* d,
                         double* rt1r, double* rt1i, double* rt2r, double* rt2i,
                         double* cs, double* sn);

/* Reordering of the Schur factorization: ?trsen (moves the eigenvalues
 * selected by the LOGICAL array select to the top and estimates the
 * condition numbers s (job = 'E' or 'B') and sep ('V' or 'B'); lwork,
 * liwork = -1: query), ?trexc (moves one diagonal block from ifst to
 * ilst; work: n), ?laexc (swaps two adjacent blocks; wantq is a LOGICAL),
 * ?trsyl (the Sylvester equation op(A)*X + isgn*X*op(B) = scale*C),
 * ?lasy2 (a 1 x 1 to 2 x 2 Sylvester equation; ltranl and ltranr are
 * LOGICALs), ?laln2 (a scaled 1 x 1 or 2 x 2 system with a perturbed
 * diagonal; ltrans is a LOGICAL), ?lacn2 (1-norm estimation by reverse
 * communication) and ?larfx (an elementary reflector, unrolled for
 * orders <= 10). */
HWYBLAS_API void strsen_(const char* job, const char* compq,
                         const hwyblas_int* select, const hwyblas_int* n,
                         float* t, const hwyblas_int* ldt, float* q,
                         const hwyblas_int* ldq, float* wr, float* wi,
                         hwyblas_int* m, float* s, float* sep, float* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void dtrsen_(const char* job, const char* compq,
                         const hwyblas_int* select, const hwyblas_int* n,
                         double* t, const hwyblas_int* ldt, double* q,
                         const hwyblas_int* ldq, double* wr, double* wi,
                         hwyblas_int* m, double* s, double* sep, double* work,
                         const hwyblas_int* lwork, hwyblas_int* iwork,
                         const hwyblas_int* liwork, hwyblas_int* info);
HWYBLAS_API void ctrsen_(const char* job, const char* compq,
                         const hwyblas_int* select, const hwyblas_int* n,
                         float* t, const hwyblas_int* ldt, float* q,
                         const hwyblas_int* ldq, float* w, hwyblas_int* m,
                         float* s, float* sep, float* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void ztrsen_(const char* job, const char* compq,
                         const hwyblas_int* select, const hwyblas_int* n,
                         double* t, const hwyblas_int* ldt, double* q,
                         const hwyblas_int* ldq, double* w, hwyblas_int* m,
                         double* s, double* sep, double* work,
                         const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void strexc_(const char* compq, const hwyblas_int* n, float* t,
                         const hwyblas_int* ldt, float* q,
                         const hwyblas_int* ldq, hwyblas_int* ifst,
                         hwyblas_int* ilst, float* work, hwyblas_int* info);
HWYBLAS_API void dtrexc_(const char* compq, const hwyblas_int* n, double* t,
                         const hwyblas_int* ldt, double* q,
                         const hwyblas_int* ldq, hwyblas_int* ifst,
                         hwyblas_int* ilst, double* work, hwyblas_int* info);
HWYBLAS_API void ctrexc_(const char* compq, const hwyblas_int* n, float* t,
                         const hwyblas_int* ldt, float* q,
                         const hwyblas_int* ldq, const hwyblas_int* ifst,
                         const hwyblas_int* ilst, hwyblas_int* info);
HWYBLAS_API void ztrexc_(const char* compq, const hwyblas_int* n, double* t,
                         const hwyblas_int* ldt, double* q,
                         const hwyblas_int* ldq, const hwyblas_int* ifst,
                         const hwyblas_int* ilst, hwyblas_int* info);
HWYBLAS_API void slaexc_(const hwyblas_int* wantq, const hwyblas_int* n,
                         float* t, const hwyblas_int* ldt, float* q,
                         const hwyblas_int* ldq, const hwyblas_int* j1,
                         const hwyblas_int* n1, const hwyblas_int* n2,
                         float* work, hwyblas_int* info);
HWYBLAS_API void dlaexc_(const hwyblas_int* wantq, const hwyblas_int* n,
                         double* t, const hwyblas_int* ldt, double* q,
                         const hwyblas_int* ldq, const hwyblas_int* j1,
                         const hwyblas_int* n1, const hwyblas_int* n2,
                         double* work, hwyblas_int* info);
HWYBLAS_API void strsyl_(const char* trana, const char* tranb,
                         const hwyblas_int* isgn, const hwyblas_int* m,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, float* c,
                         const hwyblas_int* ldc, float* scale,
                         hwyblas_int* info);
HWYBLAS_API void dtrsyl_(const char* trana, const char* tranb,
                         const hwyblas_int* isgn, const hwyblas_int* m,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, double* c,
                         const hwyblas_int* ldc, double* scale,
                         hwyblas_int* info);
HWYBLAS_API void ctrsyl_(const char* trana, const char* tranb,
                         const hwyblas_int* isgn, const hwyblas_int* m,
                         const hwyblas_int* n, const float* a,
                         const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, float* c,
                         const hwyblas_int* ldc, float* scale,
                         hwyblas_int* info);
HWYBLAS_API void ztrsyl_(const char* trana, const char* tranb,
                         const hwyblas_int* isgn, const hwyblas_int* m,
                         const hwyblas_int* n, const double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, double* c,
                         const hwyblas_int* ldc, double* scale,
                         hwyblas_int* info);
HWYBLAS_API void slasy2_(const hwyblas_int* ltranl, const hwyblas_int* ltranr,
                         const hwyblas_int* isgn, const hwyblas_int* n1,
                         const hwyblas_int* n2, const float* tl,
                         const hwyblas_int* ldtl, const float* tr,
                         const hwyblas_int* ldtr, const float* b,
                         const hwyblas_int* ldb, float* scale, float* x,
                         const hwyblas_int* ldx, float* xnorm,
                         hwyblas_int* info);
HWYBLAS_API void dlasy2_(const hwyblas_int* ltranl, const hwyblas_int* ltranr,
                         const hwyblas_int* isgn, const hwyblas_int* n1,
                         const hwyblas_int* n2, const double* tl,
                         const hwyblas_int* ldtl, const double* tr,
                         const hwyblas_int* ldtr, const double* b,
                         const hwyblas_int* ldb, double* scale, double* x,
                         const hwyblas_int* ldx, double* xnorm,
                         hwyblas_int* info);
HWYBLAS_API void slaln2_(const hwyblas_int* ltrans, const hwyblas_int* na,
                         const hwyblas_int* nw, const float* smin,
                         const float* ca, const float* a,
                         const hwyblas_int* lda, const float* d1,
                         const float* d2, const float* b,
                         const hwyblas_int* ldb, const float* wr,
                         const float* wi, float* x, const hwyblas_int* ldx,
                         float* scale, float* xnorm, hwyblas_int* info);
HWYBLAS_API void dlaln2_(const hwyblas_int* ltrans, const hwyblas_int* na,
                         const hwyblas_int* nw, const double* smin,
                         const double* ca, const double* a,
                         const hwyblas_int* lda, const double* d1,
                         const double* d2, const double* b,
                         const hwyblas_int* ldb, const double* wr,
                         const double* wi, double* x, const hwyblas_int* ldx,
                         double* scale, double* xnorm, hwyblas_int* info);
HWYBLAS_API void slacn2_(const hwyblas_int* n, float* v, float* x,
                         hwyblas_int* isgn, float* est, hwyblas_int* kase,
                         hwyblas_int* isave);
HWYBLAS_API void dlacn2_(const hwyblas_int* n, double* v, double* x,
                         hwyblas_int* isgn, double* est, hwyblas_int* kase,
                         hwyblas_int* isave);
HWYBLAS_API void clacn2_(const hwyblas_int* n, float* v, float* x, float* est,
                         hwyblas_int* kase, hwyblas_int* isave);
HWYBLAS_API void zlacn2_(const hwyblas_int* n, double* v, double* x,
                         double* est, hwyblas_int* kase, hwyblas_int* isave);
HWYBLAS_API void slarfx_(const char* side, const hwyblas_int* m,
                         const hwyblas_int* n, const float* v, const float* tau,
                         float* c, const hwyblas_int* ldc, float* work);
HWYBLAS_API void dlarfx_(const char* side, const hwyblas_int* m,
                         const hwyblas_int* n, const double* v,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work);
HWYBLAS_API void clarfx_(const char* side, const hwyblas_int* m,
                         const hwyblas_int* n, const float* v, const float* tau,
                         float* c, const hwyblas_int* ldc, float* work);
HWYBLAS_API void zlarfx_(const char* side, const hwyblas_int* m,
                         const hwyblas_int* n, const double* v,
                         const double* tau, double* c, const hwyblas_int* ldc,
                         double* work);

/* Eigenvectors of the Schur form T (?hseqr): right (side = 'R'),
 * left ('L') or both ('B'); all of them (howmny = 'A'), all back-transformed
 * with the Schur vectors in vl / vr ('B'), or the ones selected by the
 * LOGICAL array select ('S'; real routines: normalizes select for complex
 * pairs). mm: columns of vl / vr; m: columns used. work: lwork >= max(1,
 * 3n) (real) or max(1, 2n) (complex), more for the blocked back-transform
 * (lwork = -1: query); rwork: lrwork >= max(1, n). */
HWYBLAS_API void strevc3_(const char* side, const char* howmny,
                          hwyblas_int* select, const hwyblas_int* n,
                          const float* t, const hwyblas_int* ldt, float* vl,
                          const hwyblas_int* ldvl, float* vr,
                          const hwyblas_int* ldvr, const hwyblas_int* mm,
                          hwyblas_int* m, float* work, const hwyblas_int* lwork,
                          hwyblas_int* info);
HWYBLAS_API void dtrevc3_(const char* side, const char* howmny,
                          hwyblas_int* select, const hwyblas_int* n,
                          const double* t, const hwyblas_int* ldt, double* vl,
                          const hwyblas_int* ldvl, double* vr,
                          const hwyblas_int* ldvr, const hwyblas_int* mm,
                          hwyblas_int* m, double* work,
                          const hwyblas_int* lwork, hwyblas_int* info);
HWYBLAS_API void ctrevc3_(const char* side, const char* howmny,
                          const hwyblas_int* select, const hwyblas_int* n,
                          float* t, const hwyblas_int* ldt, float* vl,
                          const hwyblas_int* ldvl, float* vr,
                          const hwyblas_int* ldvr, const hwyblas_int* mm,
                          hwyblas_int* m, float* work, const hwyblas_int* lwork,
                          float* rwork, const hwyblas_int* lrwork,
                          hwyblas_int* info);
HWYBLAS_API void ztrevc3_(const char* side, const char* howmny,
                          const hwyblas_int* select, const hwyblas_int* n,
                          double* t, const hwyblas_int* ldt, double* vl,
                          const hwyblas_int* ldvl, double* vr,
                          const hwyblas_int* ldvr, const hwyblas_int* mm,
                          hwyblas_int* m, double* work,
                          const hwyblas_int* lwork, double* rwork,
                          const hwyblas_int* lrwork, hwyblas_int* info);

/* Reciprocal condition numbers of eigenvalues (job = 'E', s) and/or
 * right eigenvectors ('V', sep; 'B': both) of the Schur form T, for all
 * (howmny = 'A') or the selected ('S') eigenpairs; vl / vr: their
 * eigenvectors (?trevc3). work: ldwork x (n+6) (real) or (n+1) (complex);
 * iwork: 2(n-1); rwork: n. */
HWYBLAS_API void strsna_(const char* job, const char* howmny,
                         const hwyblas_int* select, const hwyblas_int* n,
                         const float* t, const hwyblas_int* ldt,
                         const float* vl, const hwyblas_int* ldvl,
                         const float* vr, const hwyblas_int* ldvr, float* s,
                         float* sep, const hwyblas_int* mm, hwyblas_int* m,
                         float* work, const hwyblas_int* ldwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void dtrsna_(const char* job, const char* howmny,
                         const hwyblas_int* select, const hwyblas_int* n,
                         const double* t, const hwyblas_int* ldt,
                         const double* vl, const hwyblas_int* ldvl,
                         const double* vr, const hwyblas_int* ldvr, double* s,
                         double* sep, const hwyblas_int* mm, hwyblas_int* m,
                         double* work, const hwyblas_int* ldwork,
                         hwyblas_int* iwork, hwyblas_int* info);
HWYBLAS_API void ctrsna_(const char* job, const char* howmny,
                         const hwyblas_int* select, const hwyblas_int* n,
                         const float* t, const hwyblas_int* ldt,
                         const float* vl, const hwyblas_int* ldvl,
                         const float* vr, const hwyblas_int* ldvr, float* s,
                         float* sep, const hwyblas_int* mm, hwyblas_int* m,
                         float* work, const hwyblas_int* ldwork, float* rwork,
                         hwyblas_int* info);
HWYBLAS_API void ztrsna_(const char* job, const char* howmny,
                         const hwyblas_int* select, const hwyblas_int* n,
                         const double* t, const hwyblas_int* ldt,
                         const double* vl, const hwyblas_int* ldvl,
                         const double* vr, const hwyblas_int* ldvr, double* s,
                         double* sep, const hwyblas_int* mm, hwyblas_int* m,
                         double* work, const hwyblas_int* ldwork, double* rwork,
                         hwyblas_int* info);

/* Solves op(T)*p = scale*c (lreal) or op(T + iB)*(p+iq) =
 * scale*(c+id) in real arithmetic with the quasi-triangular T (ltran:
 * op(T) = T**T; ltran and lreal are LOGICALs; B = diag(w, b(2:n))); x
 * holds c (and d) and returns the solution; work: n. */
HWYBLAS_API void slaqtr_(const hwyblas_int* ltran, const hwyblas_int* lreal,
                         const hwyblas_int* n, const float* t,
                         const hwyblas_int* ldt, const float* b, const float* w,
                         float* scale, float* x, float* work,
                         hwyblas_int* info);
HWYBLAS_API void dlaqtr_(const hwyblas_int* ltran, const hwyblas_int* lreal,
                         const hwyblas_int* n, const double* t,
                         const hwyblas_int* ldt, const double* b,
                         const double* w, double* scale, double* x,
                         double* work, hwyblas_int* info);

/* Complex auxiliary routines: ?lartg (a plane rotation with real c and
 * complex s), ?rot (applies one; the BLAS's csrot / zdrot have a real
 * s), icmax1 / izmax1 (the index of the largest |x_i|) and scsum1 /
 * dzsum1 (the sum of the |x_i|). */
HWYBLAS_API void clartg_(const float* f, const float* g, float* c, float* s,
                         float* r);
HWYBLAS_API void zlartg_(const double* f, const double* g, double* c, double* s,
                         double* r);
HWYBLAS_API void crot_(const hwyblas_int* n, float* cx, const hwyblas_int* incx,
                       float* cy, const hwyblas_int* incy, const float* c,
                       const float* s);
HWYBLAS_API void zrot_(const hwyblas_int* n, double* cx,
                       const hwyblas_int* incx, double* cy,
                       const hwyblas_int* incy, const double* c,
                       const double* s);
HWYBLAS_API hwyblas_int icmax1_(const hwyblas_int* n, const float* zx,
                                const hwyblas_int* incx);
HWYBLAS_API hwyblas_int izmax1_(const hwyblas_int* n, const double* zx,
                                const hwyblas_int* incx);
HWYBLAS_API float scsum1_(const hwyblas_int* n, const float* cx,
                          const hwyblas_int* incx);
HWYBLAS_API double dzsum1_(const hwyblas_int* n, const double* cx,
                           const hwyblas_int* incx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACK_H_ */
