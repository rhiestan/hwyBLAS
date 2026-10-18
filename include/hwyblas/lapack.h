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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACK_H_ */
