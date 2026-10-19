/*
 * hwyBLAS - LAPACKE, the C interface to the LAPACK routines.
 *
 * Compatible with the reference LAPACKE (LAPACK 3.12.1): the same functions,
 * types and conventions, and the same behavior (see below), so that programs
 * written for it, such as Eigen with EIGEN_USE_LAPACKE, can use hwyBLAS.
 *
 * Each routine xyyzzz has two functions:
 *  - LAPACKE_xyyzzz(matrix_layout, ...) checks the input matrices for NaN
 *    (unless switched off, see LAPACKE_set_nancheck), allocates the
 *    workspace and calls LAPACKE_xyyzzz_work.
 *  - LAPACKE_xyyzzz_work(matrix_layout, ...) takes the workspace from the
 *    caller.
 * matrix_layout is LAPACK_COL_MAJOR or LAPACK_ROW_MAJOR (row-major matrices
 * are transposed into temporary column-major copies). The functions return
 * INFO; an invalid argument i gives -i (counting matrix_layout as argument
 * 1) and is reported with LAPACKE_xerbla. A NaN in input argument i gives
 * -i without a report.
 *
 * Types: lapack_int is int (LP64, like hwyBLAS); define lapack_complex_float
 * and lapack_complex_double before including this header to use your own
 * complex types (with the memory layout of two floats or doubles). The
 * defaults are std::complex in C++, and float _Complex (_Fcomplex with
 * MSVC) in C.
 *
 * Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
 * LAPACK, see LICENSE-LAPACK.
 */
#ifndef HWYBLAS_LAPACKE_H_
#define HWYBLAS_LAPACKE_H_

#include "hwyblas/hwyblas.h"

#ifndef lapack_int
#define lapack_int hwyblas_int
#endif
#ifndef lapack_logical
#define lapack_logical lapack_int
#endif

#ifdef __cplusplus
#include <complex>
#ifndef lapack_complex_float
#define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
#define lapack_complex_double std::complex<double>
#endif
#else
#include <complex.h>
#ifndef lapack_complex_float
#if defined(_MSC_VER)
#define lapack_complex_float _Fcomplex
#else
#define lapack_complex_float float _Complex
#endif
#endif
#ifndef lapack_complex_double
#if defined(_MSC_VER)
#define lapack_complex_double _Dcomplex
#else
#define lapack_complex_double double _Complex
#endif
#endif
#endif

/* Macros, as in the reference LAPACKE (programs test them with #ifdef). */
/* NOLINTBEGIN(modernize-macro-to-enum) */
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

#define LAPACK_WORK_MEMORY_ERROR -1010
#define LAPACK_TRANSPOSE_MEMORY_ERROR -1011
/* NOLINTEND(modernize-macro-to-enum) */

/* The SELECT functions of LAPACKE_?gees (nonzero is true), as in the
 * reference LAPACK. */
typedef lapack_logical (*LAPACK_S_SELECT2)(const float*, const float*);
typedef lapack_logical (*LAPACK_D_SELECT2)(const double*, const double*);
typedef lapack_logical (*LAPACK_C_SELECT1)(const lapack_complex_float*);
typedef lapack_logical (*LAPACK_Z_SELECT1)(const lapack_complex_double*);

#ifdef __cplusplus
extern "C" {
#endif

/* Prints an error message: "Wrong parameter <-info> in <name>" (or an
 * out-of-memory message for LAPACK_WORK_MEMORY_ERROR and
 * LAPACK_TRANSPOSE_MEMORY_ERROR). */
HWYBLAS_API void LAPACKE_xerbla(const char* name, lapack_int info);

/* Whether the LAPACKE_xyyzzz functions check their input matrices for NaN.
 * The default is on, unless the environment variable LAPACKE_NANCHECK is set
 * to 0 when it is first needed. */
HWYBLAS_API void LAPACKE_set_nancheck(int flag);
HWYBLAS_API int LAPACKE_get_nancheck(void);

/* ------------------------------------------------------------------------ */
/* LU factorization (see lapack.h)                                          */

HWYBLAS_API lapack_int LAPACKE_sgetrf(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgetrf(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgetrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgetrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgetrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgetrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgetrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgetrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgetrf2(int matrix_layout, lapack_int m,
                                       lapack_int n, float* a, lapack_int lda,
                                       lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgetrf2(int matrix_layout, lapack_int m,
                                       lapack_int n, double* a, lapack_int lda,
                                       lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgetrf2(int matrix_layout, lapack_int m,
                                       lapack_int n, lapack_complex_float* a,
                                       lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgetrf2(int matrix_layout, lapack_int m,
                                       lapack_int n, lapack_complex_double* a,
                                       lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgetrf2_work(int matrix_layout, lapack_int m,
                                            lapack_int n, float* a,
                                            lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgetrf2_work(int matrix_layout, lapack_int m,
                                            lapack_int n, double* a,
                                            lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgetrf2_work(int matrix_layout, lapack_int m,
                                            lapack_int n,
                                            lapack_complex_float* a,
                                            lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgetrf2_work(int matrix_layout, lapack_int m,
                                            lapack_int n,
                                            lapack_complex_double* a,
                                            lapack_int lda, lapack_int* ipiv);

/* No NaN check: the number of rows of A is not known. */
HWYBLAS_API lapack_int LAPACKE_slaswp(int matrix_layout, lapack_int n, float* a,
                                      lapack_int lda, lapack_int k1,
                                      lapack_int k2, const lapack_int* ipiv,
                                      lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_dlaswp(int matrix_layout, lapack_int n,
                                      double* a, lapack_int lda, lapack_int k1,
                                      lapack_int k2, const lapack_int* ipiv,
                                      lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_claswp(int matrix_layout, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_int k1, lapack_int k2,
                                      const lapack_int* ipiv, lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_zlaswp(int matrix_layout, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_int k1, lapack_int k2,
                                      const lapack_int* ipiv, lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_slaswp_work(int matrix_layout, lapack_int n,
                                           float* a, lapack_int lda,
                                           lapack_int k1, lapack_int k2,
                                           const lapack_int* ipiv,
                                           lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_dlaswp_work(int matrix_layout, lapack_int n,
                                           double* a, lapack_int lda,
                                           lapack_int k1, lapack_int k2,
                                           const lapack_int* ipiv,
                                           lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_claswp_work(
    int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda,
    lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_zlaswp_work(
    int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda,
    lapack_int k1, lapack_int k2, const lapack_int* ipiv, lapack_int incx);

/* ------------------------------------------------------------------------ */
/* Cholesky factorization (see lapack.h)                                    */

HWYBLAS_API lapack_int LAPACKE_spotrf(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dpotrf(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_cpotrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zpotrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_spotrf_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dpotrf_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_cpotrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zpotrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_spotrf2(int matrix_layout, char uplo,
                                       lapack_int n, float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dpotrf2(int matrix_layout, char uplo,
                                       lapack_int n, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_cpotrf2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_complex_float* a,
                                       lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zpotrf2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_complex_double* a,
                                       lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_spotrf2_work(int matrix_layout, char uplo,
                                            lapack_int n, float* a,
                                            lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dpotrf2_work(int matrix_layout, char uplo,
                                            lapack_int n, double* a,
                                            lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_cpotrf2_work(int matrix_layout, char uplo,
                                            lapack_int n,
                                            lapack_complex_float* a,
                                            lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zpotrf2_work(int matrix_layout, char uplo,
                                            lapack_int n,
                                            lapack_complex_double* a,
                                            lapack_int lda);

/* ------------------------------------------------------------------------ */
/* QR and LQ factorizations (see lapack.h)                                  */

HWYBLAS_API lapack_int LAPACKE_sgeqrf(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* tau);
HWYBLAS_API lapack_int LAPACKE_dgeqrf(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* tau);
HWYBLAS_API lapack_int LAPACKE_cgeqrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zgeqrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sgeqrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgeqrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* tau,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgeqrf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgeqrf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sgelqf(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* tau);
HWYBLAS_API lapack_int LAPACKE_dgelqf(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* tau);
HWYBLAS_API lapack_int LAPACKE_cgelqf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zgelqf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sgelqf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgelqf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* tau,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgelqf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgelqf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sgeqr2(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* tau);
HWYBLAS_API lapack_int LAPACKE_dgeqr2(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* tau);
HWYBLAS_API lapack_int LAPACKE_cgeqr2(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zgeqr2(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sgeqr2_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* tau,
                                           float* work);
HWYBLAS_API lapack_int LAPACKE_dgeqr2_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* tau,
                                           double* work);
HWYBLAS_API lapack_int LAPACKE_cgeqr2_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zgeqr2_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_sgelq2(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* tau);
HWYBLAS_API lapack_int LAPACKE_dgelq2(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* tau);
HWYBLAS_API lapack_int LAPACKE_cgelq2(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zgelq2(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sgelq2_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* tau,
                                           float* work);
HWYBLAS_API lapack_int LAPACKE_dgelq2_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* tau,
                                           double* work);
HWYBLAS_API lapack_int LAPACKE_cgelq2_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zgelq2_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work);

/* ------------------------------------------------------------------------ */
/* Orthogonal / unitary matrices (see lapack.h)                             */

HWYBLAS_API lapack_int LAPACKE_sorgqr(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, float* a,
                                      lapack_int lda, const float* tau);
HWYBLAS_API lapack_int LAPACKE_dorgqr(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, double* a,
                                      lapack_int lda, const double* tau);
HWYBLAS_API lapack_int LAPACKE_cungqr(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zungqr(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sorgqr_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int k, float* a,
                                           lapack_int lda, const float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dorgqr_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int k,
                                           double* a, lapack_int lda,
                                           const double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cungqr_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zungqr_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sorglq(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, float* a,
                                      lapack_int lda, const float* tau);
HWYBLAS_API lapack_int LAPACKE_dorglq(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, double* a,
                                      lapack_int lda, const double* tau);
HWYBLAS_API lapack_int LAPACKE_cunglq(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zunglq(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sorglq_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int k, float* a,
                                           lapack_int lda, const float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dorglq_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int k,
                                           double* a, lapack_int lda,
                                           const double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunglq_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunglq_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sormqr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const float* a, lapack_int lda,
                                      const float* tau, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormqr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const double* a, lapack_int lda,
                                      const double* tau, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmqr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmqr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormqr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const float* a, lapack_int lda, const float* tau, float* c,
    lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormqr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const double* a, lapack_int lda, const double* tau, double* c,
    lapack_int ldc, double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmqr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmqr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sormlq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const float* a, lapack_int lda,
                                      const float* tau, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormlq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const double* a, lapack_int lda,
                                      const double* tau, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmlq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmlq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormlq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const float* a, lapack_int lda, const float* tau, float* c,
    lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormlq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const double* a, lapack_int lda, const double* tau, double* c,
    lapack_int ldc, double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmlq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmlq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);

/* ------------------------------------------------------------------------ */
/* Auxiliary routines (see lapack.h)                                        */

/* In row-major layout, the elements of B outside the triangle uplo are
 * kept (the reference LAPACKE 3.12.1 leaves them undefined). */
HWYBLAS_API lapack_int LAPACKE_slacpy(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      const float* a, lapack_int lda, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dlacpy(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      const double* a, lapack_int lda,
                                      double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_clacpy(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, lapack_complex_float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zlacpy(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, lapack_complex_double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_slacpy_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           const float* a, lapack_int lda,
                                           float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dlacpy_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           const double* a, lapack_int lda,
                                           double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_clacpy_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           const lapack_complex_float* a,
                                           lapack_int lda,
                                           lapack_complex_float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zlacpy_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           const lapack_complex_double* a,
                                           lapack_int lda,
                                           lapack_complex_double* b,
                                           lapack_int ldb);

HWYBLAS_API lapack_int LAPACKE_slarfg(lapack_int n, float* alpha, float* x,
                                      lapack_int incx, float* tau);
HWYBLAS_API lapack_int LAPACKE_dlarfg(lapack_int n, double* alpha, double* x,
                                      lapack_int incx, double* tau);
HWYBLAS_API lapack_int LAPACKE_clarfg(lapack_int n, lapack_complex_float* alpha,
                                      lapack_complex_float* x, lapack_int incx,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zlarfg(lapack_int n,
                                      lapack_complex_double* alpha,
                                      lapack_complex_double* x, lapack_int incx,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_slarfg_work(lapack_int n, float* alpha, float* x,
                                           lapack_int incx, float* tau);
HWYBLAS_API lapack_int LAPACKE_dlarfg_work(lapack_int n, double* alpha,
                                           double* x, lapack_int incx,
                                           double* tau);
HWYBLAS_API lapack_int LAPACKE_clarfg_work(lapack_int n,
                                           lapack_complex_float* alpha,
                                           lapack_complex_float* x,
                                           lapack_int incx,
                                           lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zlarfg_work(lapack_int n,
                                           lapack_complex_double* alpha,
                                           lapack_complex_double* x,
                                           lapack_int incx,
                                           lapack_complex_double* tau);

HWYBLAS_API lapack_int LAPACKE_slarft(int matrix_layout, char direct,
                                      char storev, lapack_int n, lapack_int k,
                                      const float* v, lapack_int ldv,
                                      const float* tau, float* t,
                                      lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_dlarft(int matrix_layout, char direct,
                                      char storev, lapack_int n, lapack_int k,
                                      const double* v, lapack_int ldv,
                                      const double* tau, double* t,
                                      lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_clarft(int matrix_layout, char direct,
                                      char storev, lapack_int n, lapack_int k,
                                      const lapack_complex_float* v,
                                      lapack_int ldv,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* t, lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_zlarft(int matrix_layout, char direct,
                                      char storev, lapack_int n, lapack_int k,
                                      const lapack_complex_double* v,
                                      lapack_int ldv,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* t, lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_slarft_work(int matrix_layout, char direct,
                                           char storev, lapack_int n,
                                           lapack_int k, const float* v,
                                           lapack_int ldv, const float* tau,
                                           float* t, lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_dlarft_work(int matrix_layout, char direct,
                                           char storev, lapack_int n,
                                           lapack_int k, const double* v,
                                           lapack_int ldv, const double* tau,
                                           double* t, lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_clarft_work(
    int matrix_layout, char direct, char storev, lapack_int n, lapack_int k,
    const lapack_complex_float* v, lapack_int ldv,
    const lapack_complex_float* tau, lapack_complex_float* t, lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_zlarft_work(
    int matrix_layout, char direct, char storev, lapack_int n, lapack_int k,
    const lapack_complex_double* v, lapack_int ldv,
    const lapack_complex_double* tau, lapack_complex_double* t, lapack_int ldt);
HWYBLAS_API lapack_int LAPACKE_slarfb(int matrix_layout, char side, char trans,
                                      char direct, char storev, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      const float* v, lapack_int ldv,
                                      const float* t, lapack_int ldt, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dlarfb(int matrix_layout, char side, char trans,
                                      char direct, char storev, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      const double* v, lapack_int ldv,
                                      const double* t, lapack_int ldt,
                                      double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_clarfb(
    int matrix_layout, char side, char trans, char direct, char storev,
    lapack_int m, lapack_int n, lapack_int k, const lapack_complex_float* v,
    lapack_int ldv, const lapack_complex_float* t, lapack_int ldt,
    lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zlarfb(
    int matrix_layout, char side, char trans, char direct, char storev,
    lapack_int m, lapack_int n, lapack_int k, const lapack_complex_double* v,
    lapack_int ldv, const lapack_complex_double* t, lapack_int ldt,
    lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_slarfb_work(
    int matrix_layout, char side, char trans, char direct, char storev,
    lapack_int m, lapack_int n, lapack_int k, const float* v, lapack_int ldv,
    const float* t, lapack_int ldt, float* c, lapack_int ldc, float* work,
    lapack_int ldwork);
HWYBLAS_API lapack_int LAPACKE_dlarfb_work(
    int matrix_layout, char side, char trans, char direct, char storev,
    lapack_int m, lapack_int n, lapack_int k, const double* v, lapack_int ldv,
    const double* t, lapack_int ldt, double* c, lapack_int ldc, double* work,
    lapack_int ldwork);
HWYBLAS_API lapack_int LAPACKE_clarfb_work(
    int matrix_layout, char side, char trans, char direct, char storev,
    lapack_int m, lapack_int n, lapack_int k, const lapack_complex_float* v,
    lapack_int ldv, const lapack_complex_float* t, lapack_int ldt,
    lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work,
    lapack_int ldwork);
HWYBLAS_API lapack_int LAPACKE_zlarfb_work(
    int matrix_layout, char side, char trans, char direct, char storev,
    lapack_int m, lapack_int n, lapack_int k, const lapack_complex_double* v,
    lapack_int ldv, const lapack_complex_double* t, lapack_int ldt,
    lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work,
    lapack_int ldwork);

HWYBLAS_API lapack_int LAPACKE_clacgv(lapack_int n, lapack_complex_float* x,
                                      lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_zlacgv(lapack_int n, lapack_complex_double* x,
                                      lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_clacgv_work(lapack_int n,
                                           lapack_complex_float* x,
                                           lapack_int incx);
HWYBLAS_API lapack_int LAPACKE_zlacgv_work(lapack_int n,
                                           lapack_complex_double* x,
                                           lapack_int incx);

/* A NaN argument gives minus its position (as a floating-point value). */
HWYBLAS_API float LAPACKE_slapy2(float x, float y);
HWYBLAS_API double LAPACKE_dlapy2(double x, double y);
HWYBLAS_API float LAPACKE_slapy2_work(float x, float y);
HWYBLAS_API double LAPACKE_dlapy2_work(double x, double y);
HWYBLAS_API float LAPACKE_slapy3(float x, float y, float z);
HWYBLAS_API double LAPACKE_dlapy3(double x, double y, double z);
HWYBLAS_API float LAPACKE_slapy3_work(float x, float y, float z);
HWYBLAS_API double LAPACKE_dlapy3_work(double x, double y, double z);

/* ------------------------------------------------------------------------ */
/* QR factorization with column pivoting (see lapack.h)                     */

HWYBLAS_API lapack_int LAPACKE_sgeqp3(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      lapack_int* jpvt, float* tau);
HWYBLAS_API lapack_int LAPACKE_dgeqp3(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      lapack_int* jpvt, double* tau);
HWYBLAS_API lapack_int LAPACKE_cgeqp3(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int* jpvt,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zgeqp3(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, lapack_int* jpvt,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sgeqp3_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, lapack_int* jpvt,
                                           float* tau, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgeqp3_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, lapack_int* jpvt,
                                           double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgeqp3_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_int* jpvt, lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgeqp3_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_int* jpvt, lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork, double* rwork);

/* ------------------------------------------------------------------------ */
/* Norms, scaling and setting (see lapack.h)                                */

/* The norms return a negative argument position (as a floating-point
 * value) for an invalid argument or NaN. */
HWYBLAS_API float LAPACKE_slange(int matrix_layout, char norm, lapack_int m,
                                 lapack_int n, const float* a, lapack_int lda);
HWYBLAS_API double LAPACKE_dlange(int matrix_layout, char norm, lapack_int m,
                                  lapack_int n, const double* a,
                                  lapack_int lda);
HWYBLAS_API float LAPACKE_clange(int matrix_layout, char norm, lapack_int m,
                                 lapack_int n, const lapack_complex_float* a,
                                 lapack_int lda);
HWYBLAS_API double LAPACKE_zlange(int matrix_layout, char norm, lapack_int m,
                                  lapack_int n, const lapack_complex_double* a,
                                  lapack_int lda);
HWYBLAS_API float LAPACKE_slange_work(int matrix_layout, char norm,
                                      lapack_int m, lapack_int n,
                                      const float* a, lapack_int lda,
                                      float* work);
HWYBLAS_API double LAPACKE_dlange_work(int matrix_layout, char norm,
                                       lapack_int m, lapack_int n,
                                       const double* a, lapack_int lda,
                                       double* work);
HWYBLAS_API float LAPACKE_clange_work(int matrix_layout, char norm,
                                      lapack_int m, lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* work);
HWYBLAS_API double LAPACKE_zlange_work(int matrix_layout, char norm,
                                       lapack_int m, lapack_int n,
                                       const lapack_complex_double* a,
                                       lapack_int lda, double* work);
HWYBLAS_API float LAPACKE_slansy(int matrix_layout, char norm, char uplo,
                                 lapack_int n, const float* a, lapack_int lda);
HWYBLAS_API double LAPACKE_dlansy(int matrix_layout, char norm, char uplo,
                                  lapack_int n, const double* a,
                                  lapack_int lda);
HWYBLAS_API float LAPACKE_clansy(int matrix_layout, char norm, char uplo,
                                 lapack_int n, const lapack_complex_float* a,
                                 lapack_int lda);
HWYBLAS_API double LAPACKE_zlansy(int matrix_layout, char norm, char uplo,
                                  lapack_int n, const lapack_complex_double* a,
                                  lapack_int lda);
HWYBLAS_API float LAPACKE_slansy_work(int matrix_layout, char norm, char uplo,
                                      lapack_int n, const float* a,
                                      lapack_int lda, float* work);
HWYBLAS_API double LAPACKE_dlansy_work(int matrix_layout, char norm, char uplo,
                                       lapack_int n, const double* a,
                                       lapack_int lda, double* work);
HWYBLAS_API float LAPACKE_clansy_work(int matrix_layout, char norm, char uplo,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* work);
HWYBLAS_API double LAPACKE_zlansy_work(int matrix_layout, char norm, char uplo,
                                       lapack_int n,
                                       const lapack_complex_double* a,
                                       lapack_int lda, double* work);
HWYBLAS_API float LAPACKE_clanhe(int matrix_layout, char norm, char uplo,
                                 lapack_int n, const lapack_complex_float* a,
                                 lapack_int lda);
HWYBLAS_API double LAPACKE_zlanhe(int matrix_layout, char norm, char uplo,
                                  lapack_int n, const lapack_complex_double* a,
                                  lapack_int lda);
HWYBLAS_API float LAPACKE_clanhe_work(int matrix_layout, char norm, char uplo,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* work);
HWYBLAS_API double LAPACKE_zlanhe_work(int matrix_layout, char norm, char uplo,
                                       lapack_int n,
                                       const lapack_complex_double* a,
                                       lapack_int lda, double* work);

HWYBLAS_API lapack_int LAPACKE_slascl(int matrix_layout, char type,
                                      lapack_int kl, lapack_int ku, float cfrom,
                                      float cto, lapack_int m, lapack_int n,
                                      float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dlascl(int matrix_layout, char type,
                                      lapack_int kl, lapack_int ku,
                                      double cfrom, double cto, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_clascl(int matrix_layout, char type,
                                      lapack_int kl, lapack_int ku, float cfrom,
                                      float cto, lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zlascl(int matrix_layout, char type,
                                      lapack_int kl, lapack_int ku,
                                      double cfrom, double cto, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_slascl_work(int matrix_layout, char type,
                                           lapack_int kl, lapack_int ku,
                                           float cfrom, float cto, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dlascl_work(int matrix_layout, char type,
                                           lapack_int kl, lapack_int ku,
                                           double cfrom, double cto,
                                           lapack_int m, lapack_int n,
                                           double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_clascl_work(int matrix_layout, char type,
                                           lapack_int kl, lapack_int ku,
                                           float cfrom, float cto, lapack_int m,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zlascl_work(int matrix_layout, char type,
                                           lapack_int kl, lapack_int ku,
                                           double cfrom, double cto,
                                           lapack_int m, lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda);

HWYBLAS_API lapack_int LAPACKE_slaset(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n, float alpha,
                                      float beta, float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dlaset(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n, double alpha,
                                      double beta, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_claset(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_float alpha,
                                      lapack_complex_float beta,
                                      lapack_complex_float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zlaset(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_double alpha,
                                      lapack_complex_double beta,
                                      lapack_complex_double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_slaset_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           float alpha, float beta, float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dlaset_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           double alpha, double beta, double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_claset_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           lapack_complex_float alpha,
                                           lapack_complex_float beta,
                                           lapack_complex_float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zlaset_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           lapack_complex_double alpha,
                                           lapack_complex_double beta,
                                           lapack_complex_double* a,
                                           lapack_int lda);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) eigenproblems (see lapack.h)                       */

HWYBLAS_API lapack_int LAPACKE_ssyev(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, float* a, lapack_int lda,
                                     float* w);
HWYBLAS_API lapack_int LAPACKE_dsyev(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, double* a, lapack_int lda,
                                     double* w);
HWYBLAS_API lapack_int LAPACKE_cheev(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, lapack_complex_float* a,
                                     lapack_int lda, float* w);
HWYBLAS_API lapack_int LAPACKE_zheev(int matrix_layout, char jobz, char uplo,
                                     lapack_int n, lapack_complex_double* a,
                                     lapack_int lda, double* w);
HWYBLAS_API lapack_int LAPACKE_ssyev_work(int matrix_layout, char jobz,
                                          char uplo, lapack_int n, float* a,
                                          lapack_int lda, float* w, float* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dsyev_work(int matrix_layout, char jobz,
                                          char uplo, lapack_int n, double* a,
                                          lapack_int lda, double* w,
                                          double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cheev_work(int matrix_layout, char jobz,
                                          char uplo, lapack_int n,
                                          lapack_complex_float* a,
                                          lapack_int lda, float* w,
                                          lapack_complex_float* work,
                                          lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zheev_work(int matrix_layout, char jobz,
                                          char uplo, lapack_int n,
                                          lapack_complex_double* a,
                                          lapack_int lda, double* w,
                                          lapack_complex_double* work,
                                          lapack_int lwork, double* rwork);

HWYBLAS_API lapack_int LAPACKE_ssytrd(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* d, float* e, float* tau);
HWYBLAS_API lapack_int LAPACKE_dsytrd(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* d, double* e, double* tau);
HWYBLAS_API lapack_int LAPACKE_chetrd(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, float* d, float* e,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zhetrd(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, double* d, double* e,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_ssytrd_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* d, float* e,
                                           float* tau, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dsytrd_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* d, double* e,
                                           double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_chetrd_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a,
    lapack_int lda, float* d, float* e, lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zhetrd_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a,
    lapack_int lda, double* d, double* e, lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sorgtr(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda,
                                      const float* tau);
HWYBLAS_API lapack_int LAPACKE_dorgtr(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda,
                                      const double* tau);
HWYBLAS_API lapack_int LAPACKE_cungtr(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zungtr(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sorgtr_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda, const float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dorgtr_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda, const double* tau,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cungtr_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a,
    lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zungtr_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a,
    lapack_int lda, const lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sorgql(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, float* a,
                                      lapack_int lda, const float* tau);
HWYBLAS_API lapack_int LAPACKE_dorgql(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k, double* a,
                                      lapack_int lda, const double* tau);
HWYBLAS_API lapack_int LAPACKE_cungql(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zungql(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int k,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sorgql_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int k, float* a,
                                           lapack_int lda, const float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dorgql_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int k,
                                           double* a, lapack_int lda,
                                           const double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cungql_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zungql_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_ssteqr(int matrix_layout, char compz,
                                      lapack_int n, float* d, float* e,
                                      float* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_dsteqr(int matrix_layout, char compz,
                                      lapack_int n, double* d, double* e,
                                      double* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_csteqr(int matrix_layout, char compz,
                                      lapack_int n, float* d, float* e,
                                      lapack_complex_float* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_zsteqr(int matrix_layout, char compz,
                                      lapack_int n, double* d, double* e,
                                      lapack_complex_double* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_ssteqr_work(int matrix_layout, char compz,
                                           lapack_int n, float* d, float* e,
                                           float* z, lapack_int ldz,
                                           float* work);
HWYBLAS_API lapack_int LAPACKE_dsteqr_work(int matrix_layout, char compz,
                                           lapack_int n, double* d, double* e,
                                           double* z, lapack_int ldz,
                                           double* work);
HWYBLAS_API lapack_int LAPACKE_csteqr_work(int matrix_layout, char compz,
                                           lapack_int n, float* d, float* e,
                                           lapack_complex_float* z,
                                           lapack_int ldz, float* work);
HWYBLAS_API lapack_int LAPACKE_zsteqr_work(int matrix_layout, char compz,
                                           lapack_int n, double* d, double* e,
                                           lapack_complex_double* z,
                                           lapack_int ldz, double* work);

HWYBLAS_API lapack_int LAPACKE_ssterf(lapack_int n, float* d, float* e);
HWYBLAS_API lapack_int LAPACKE_dsterf(lapack_int n, double* d, double* e);
HWYBLAS_API lapack_int LAPACKE_ssterf_work(lapack_int n, float* d, float* e);
HWYBLAS_API lapack_int LAPACKE_dsterf_work(lapack_int n, double* d, double* e);
HWYBLAS_API lapack_int LAPACKE_slasrt(char id, lapack_int n, float* d);
HWYBLAS_API lapack_int LAPACKE_dlasrt(char id, lapack_int n, double* d);
HWYBLAS_API lapack_int LAPACKE_slasrt_work(char id, lapack_int n, float* d);
HWYBLAS_API lapack_int LAPACKE_dlasrt_work(char id, lapack_int n, double* d);

/* ------------------------------------------------------------------------ */
/* Singular value decomposition (see lapack.h)                              */

/* superb (min(m,n)-1 elements) returns the unconverged superdiagonal
 * elements of the bidiagonal matrix (info > 0). */
HWYBLAS_API lapack_int LAPACKE_sgesvd(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n, float* a,
                                      lapack_int lda, float* s, float* u,
                                      lapack_int ldu, float* vt,
                                      lapack_int ldvt, float* superb);
HWYBLAS_API lapack_int LAPACKE_dgesvd(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n, double* a,
                                      lapack_int lda, double* s, double* u,
                                      lapack_int ldu, double* vt,
                                      lapack_int ldvt, double* superb);
HWYBLAS_API lapack_int LAPACKE_cgesvd(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      float* s, lapack_complex_float* u,
                                      lapack_int ldu, lapack_complex_float* vt,
                                      lapack_int ldvt, float* superb);
HWYBLAS_API lapack_int LAPACKE_zgesvd(int matrix_layout, char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      double* s, lapack_complex_double* u,
                                      lapack_int ldu, lapack_complex_double* vt,
                                      lapack_int ldvt, double* superb);
HWYBLAS_API lapack_int LAPACKE_sgesvd_work(
    int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n,
    float* a, lapack_int lda, float* s, float* u, lapack_int ldu, float* vt,
    lapack_int ldvt, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgesvd_work(
    int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n,
    double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt,
    lapack_int ldvt, double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgesvd_work(
    int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n,
    lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u,
    lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt,
    lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgesvd_work(
    int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double* s,
    lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt,
    lapack_int ldvt, lapack_complex_double* work, lapack_int lwork,
    double* rwork);

HWYBLAS_API lapack_int LAPACKE_sgebrd(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* d, float* e, float* tauq,
                                      float* taup);
HWYBLAS_API lapack_int LAPACKE_dgebrd(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* d, double* e, double* tauq,
                                      double* taup);
HWYBLAS_API lapack_int LAPACKE_cgebrd(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, float* d, float* e,
                                      lapack_complex_float* tauq,
                                      lapack_complex_float* taup);
HWYBLAS_API lapack_int LAPACKE_zgebrd(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, double* d, double* e,
                                      lapack_complex_double* tauq,
                                      lapack_complex_double* taup);
HWYBLAS_API lapack_int LAPACKE_sgebrd_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* d, float* e,
                                           float* tauq, float* taup,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgebrd_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* d, double* e,
                                           double* tauq, double* taup,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgebrd_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, float* d, float* e, lapack_complex_float* tauq,
    lapack_complex_float* taup, lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgebrd_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, double* d, double* e, lapack_complex_double* tauq,
    lapack_complex_double* taup, lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sorgbr(int matrix_layout, char vect,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      float* a, lapack_int lda,
                                      const float* tau);
HWYBLAS_API lapack_int LAPACKE_dorgbr(int matrix_layout, char vect,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      double* a, lapack_int lda,
                                      const double* tau);
HWYBLAS_API lapack_int LAPACKE_cungbr(int matrix_layout, char vect,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zungbr(int matrix_layout, char vect,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sorgbr_work(int matrix_layout, char vect,
                                           lapack_int m, lapack_int n,
                                           lapack_int k, float* a,
                                           lapack_int lda, const float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dorgbr_work(int matrix_layout, char vect,
                                           lapack_int m, lapack_int n,
                                           lapack_int k, double* a,
                                           lapack_int lda, const double* tau,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cungbr_work(
    int matrix_layout, char vect, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zungbr_work(
    int matrix_layout, char vect, lapack_int m, lapack_int n, lapack_int k,
    lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sormbr(int matrix_layout, char vect, char side,
                                      char trans, lapack_int m, lapack_int n,
                                      lapack_int k, const float* a,
                                      lapack_int lda, const float* tau,
                                      float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormbr(int matrix_layout, char vect, char side,
                                      char trans, lapack_int m, lapack_int n,
                                      lapack_int k, const double* a,
                                      lapack_int lda, const double* tau,
                                      double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmbr(
    int matrix_layout, char vect, char side, char trans, lapack_int m,
    lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmbr(
    int matrix_layout, char vect, char side, char trans, lapack_int m,
    lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormbr_work(
    int matrix_layout, char vect, char side, char trans, lapack_int m,
    lapack_int n, lapack_int k, const float* a, lapack_int lda,
    const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormbr_work(int matrix_layout, char vect,
                                           char side, char trans, lapack_int m,
                                           lapack_int n, lapack_int k,
                                           const double* a, lapack_int lda,
                                           const double* tau, double* c,
                                           lapack_int ldc, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmbr_work(
    int matrix_layout, char vect, char side, char trans, lapack_int m,
    lapack_int n, lapack_int k, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmbr_work(
    int matrix_layout, char vect, char side, char trans, lapack_int m,
    lapack_int n, lapack_int k, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sbdsqr(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int ncvt,
                                      lapack_int nru, lapack_int ncc, float* d,
                                      float* e, float* vt, lapack_int ldvt,
                                      float* u, lapack_int ldu, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dbdsqr(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int ncvt,
                                      lapack_int nru, lapack_int ncc, double* d,
                                      double* e, double* vt, lapack_int ldvt,
                                      double* u, lapack_int ldu, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cbdsqr(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int ncvt,
                                      lapack_int nru, lapack_int ncc, float* d,
                                      float* e, lapack_complex_float* vt,
                                      lapack_int ldvt, lapack_complex_float* u,
                                      lapack_int ldu, lapack_complex_float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zbdsqr(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int ncvt,
                                      lapack_int nru, lapack_int ncc, double* d,
                                      double* e, lapack_complex_double* vt,
                                      lapack_int ldvt, lapack_complex_double* u,
                                      lapack_int ldu, lapack_complex_double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sbdsqr_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, lapack_int nru,
    lapack_int ncc, float* d, float* e, float* vt, lapack_int ldvt, float* u,
    lapack_int ldu, float* c, lapack_int ldc, float* work);
HWYBLAS_API lapack_int LAPACKE_dbdsqr_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, lapack_int nru,
    lapack_int ncc, double* d, double* e, double* vt, lapack_int ldvt,
    double* u, lapack_int ldu, double* c, lapack_int ldc, double* work);
HWYBLAS_API lapack_int LAPACKE_cbdsqr_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, lapack_int nru,
    lapack_int ncc, float* d, float* e, lapack_complex_float* vt,
    lapack_int ldvt, lapack_complex_float* u, lapack_int ldu,
    lapack_complex_float* c, lapack_int ldc, float* work);
HWYBLAS_API lapack_int LAPACKE_zbdsqr_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int ncvt, lapack_int nru,
    lapack_int ncc, double* d, double* e, lapack_complex_double* vt,
    lapack_int ldvt, lapack_complex_double* u, lapack_int ldu,
    lapack_complex_double* c, lapack_int ldc, double* work);

/* ------------------------------------------------------------------------ */
/* Singular value decomposition by divide and conquer (see lapack.h)        */

HWYBLAS_API lapack_int LAPACKE_sgesdd(int matrix_layout, char jobz,
                                      lapack_int m, lapack_int n, float* a,
                                      lapack_int lda, float* s, float* u,
                                      lapack_int ldu, float* vt,
                                      lapack_int ldvt);
HWYBLAS_API lapack_int LAPACKE_dgesdd(int matrix_layout, char jobz,
                                      lapack_int m, lapack_int n, double* a,
                                      lapack_int lda, double* s, double* u,
                                      lapack_int ldu, double* vt,
                                      lapack_int ldvt);
HWYBLAS_API lapack_int LAPACKE_cgesdd(int matrix_layout, char jobz,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      float* s, lapack_complex_float* u,
                                      lapack_int ldu, lapack_complex_float* vt,
                                      lapack_int ldvt);
HWYBLAS_API lapack_int LAPACKE_zgesdd(int matrix_layout, char jobz,
                                      lapack_int m, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      double* s, lapack_complex_double* u,
                                      lapack_int ldu, lapack_complex_double* vt,
                                      lapack_int ldvt);
HWYBLAS_API lapack_int LAPACKE_sgesdd_work(int matrix_layout, char jobz,
                                           lapack_int m, lapack_int n, float* a,
                                           lapack_int lda, float* s, float* u,
                                           lapack_int ldu, float* vt,
                                           lapack_int ldvt, float* work,
                                           lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgesdd_work(
    int matrix_layout, char jobz, lapack_int m, lapack_int n, double* a,
    lapack_int lda, double* s, double* u, lapack_int ldu, double* vt,
    lapack_int ldvt, double* work, lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cgesdd_work(
    int matrix_layout, char jobz, lapack_int m, lapack_int n,
    lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u,
    lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt,
    lapack_complex_float* work, lapack_int lwork, float* rwork,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_zgesdd_work(
    int matrix_layout, char jobz, lapack_int m, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double* s,
    lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt,
    lapack_int ldvt, lapack_complex_double* work, lapack_int lwork,
    double* rwork, lapack_int* iwork);

HWYBLAS_API lapack_int LAPACKE_sbdsdc(int matrix_layout, char uplo, char compq,
                                      lapack_int n, float* d, float* e,
                                      float* u, lapack_int ldu, float* vt,
                                      lapack_int ldvt, float* q,
                                      lapack_int* iq);
HWYBLAS_API lapack_int LAPACKE_dbdsdc(int matrix_layout, char uplo, char compq,
                                      lapack_int n, double* d, double* e,
                                      double* u, lapack_int ldu, double* vt,
                                      lapack_int ldvt, double* q,
                                      lapack_int* iq);
HWYBLAS_API lapack_int LAPACKE_sbdsdc_work(int matrix_layout, char uplo,
                                           char compq, lapack_int n, float* d,
                                           float* e, float* u, lapack_int ldu,
                                           float* vt, lapack_int ldvt, float* q,
                                           lapack_int* iq, float* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dbdsdc_work(int matrix_layout, char uplo,
                                           char compq, lapack_int n, double* d,
                                           double* e, double* u, lapack_int ldu,
                                           double* vt, lapack_int ldvt,
                                           double* q, lapack_int* iq,
                                           double* work, lapack_int* iwork);

/* Complex matrices with real ones (see lapack.h). */
HWYBLAS_API lapack_int LAPACKE_clacp2(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      const float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zlacp2(int matrix_layout, char uplo,
                                      lapack_int m, lapack_int n,
                                      const double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_clacp2_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           const float* a, lapack_int lda,
                                           lapack_complex_float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zlacp2_work(int matrix_layout, char uplo,
                                           lapack_int m, lapack_int n,
                                           const double* a, lapack_int lda,
                                           lapack_complex_double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_clacrm(int matrix_layout, lapack_int m,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, const float* b,
                                      lapack_int ldb, lapack_complex_float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zlacrm(int matrix_layout, lapack_int m,
                                      lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, const double* b,
                                      lapack_int ldb, lapack_complex_double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_clacrm_work(
    int matrix_layout, lapack_int m, lapack_int n,
    const lapack_complex_float* a, lapack_int lda, const float* b,
    lapack_int ldb, lapack_complex_float* c, lapack_int ldc, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zlacrm_work(
    int matrix_layout, lapack_int m, lapack_int n,
    const lapack_complex_double* a, lapack_int lda, const double* b,
    lapack_int ldb, lapack_complex_double* c, lapack_int ldc, double* rwork);
HWYBLAS_API lapack_int LAPACKE_clarcm(int matrix_layout, lapack_int m,
                                      lapack_int n, const float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* b,
                                      lapack_int ldb, lapack_complex_float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zlarcm(int matrix_layout, lapack_int m,
                                      lapack_int n, const double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* b,
                                      lapack_int ldb, lapack_complex_double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_clarcm_work(
    int matrix_layout, lapack_int m, lapack_int n, const float* a,
    lapack_int lda, const lapack_complex_float* b, lapack_int ldb,
    lapack_complex_float* c, lapack_int ldc, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zlarcm_work(
    int matrix_layout, lapack_int m, lapack_int n, const double* a,
    lapack_int lda, const lapack_complex_double* b, lapack_int ldb,
    lapack_complex_double* c, lapack_int ldc, double* rwork);

/* ------------------------------------------------------------------------ */
/* Nonsymmetric eigenproblem (see lapack.h)                                 */

HWYBLAS_API lapack_int LAPACKE_sgees(int matrix_layout, char jobvs, char sort,
                                     LAPACK_S_SELECT2 select, lapack_int n,
                                     float* a, lapack_int lda, lapack_int* sdim,
                                     float* wr, float* wi, float* vs,
                                     lapack_int ldvs);
HWYBLAS_API lapack_int LAPACKE_dgees(int matrix_layout, char jobvs, char sort,
                                     LAPACK_D_SELECT2 select, lapack_int n,
                                     double* a, lapack_int lda,
                                     lapack_int* sdim, double* wr, double* wi,
                                     double* vs, lapack_int ldvs);
HWYBLAS_API lapack_int LAPACKE_cgees(int matrix_layout, char jobvs, char sort,
                                     LAPACK_C_SELECT1 select, lapack_int n,
                                     lapack_complex_float* a, lapack_int lda,
                                     lapack_int* sdim, lapack_complex_float* w,
                                     lapack_complex_float* vs, lapack_int ldvs);
HWYBLAS_API lapack_int LAPACKE_zgees(int matrix_layout, char jobvs, char sort,
                                     LAPACK_Z_SELECT1 select, lapack_int n,
                                     lapack_complex_double* a, lapack_int lda,
                                     lapack_int* sdim, lapack_complex_double* w,
                                     lapack_complex_double* vs,
                                     lapack_int ldvs);
HWYBLAS_API lapack_int LAPACKE_sgees_work(
    int matrix_layout, char jobvs, char sort, LAPACK_S_SELECT2 select,
    lapack_int n, float* a, lapack_int lda, lapack_int* sdim, float* wr,
    float* wi, float* vs, lapack_int ldvs, float* work, lapack_int lwork,
    lapack_logical* bwork);
HWYBLAS_API lapack_int LAPACKE_dgees_work(
    int matrix_layout, char jobvs, char sort, LAPACK_D_SELECT2 select,
    lapack_int n, double* a, lapack_int lda, lapack_int* sdim, double* wr,
    double* wi, double* vs, lapack_int ldvs, double* work, lapack_int lwork,
    lapack_logical* bwork);
HWYBLAS_API lapack_int LAPACKE_cgees_work(
    int matrix_layout, char jobvs, char sort, LAPACK_C_SELECT1 select,
    lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* sdim,
    lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs,
    lapack_complex_float* work, lapack_int lwork, float* rwork,
    lapack_logical* bwork);
HWYBLAS_API lapack_int LAPACKE_zgees_work(
    int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select,
    lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int* sdim,
    lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs,
    lapack_complex_double* work, lapack_int lwork, double* rwork,
    lapack_logical* bwork);

HWYBLAS_API lapack_int LAPACKE_sgebal(int matrix_layout, char job, lapack_int n,
                                      float* a, lapack_int lda, lapack_int* ilo,
                                      lapack_int* ihi, float* scale);
HWYBLAS_API lapack_int LAPACKE_dgebal(int matrix_layout, char job, lapack_int n,
                                      double* a, lapack_int lda,
                                      lapack_int* ilo, lapack_int* ihi,
                                      double* scale);
HWYBLAS_API lapack_int LAPACKE_cgebal(int matrix_layout, char job, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_int* ilo, lapack_int* ihi,
                                      float* scale);
HWYBLAS_API lapack_int LAPACKE_zgebal(int matrix_layout, char job, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_int* ilo, lapack_int* ihi,
                                      double* scale);
HWYBLAS_API lapack_int LAPACKE_sgebal_work(int matrix_layout, char job,
                                           lapack_int n, float* a,
                                           lapack_int lda, lapack_int* ilo,
                                           lapack_int* ihi, float* scale);
HWYBLAS_API lapack_int LAPACKE_dgebal_work(int matrix_layout, char job,
                                           lapack_int n, double* a,
                                           lapack_int lda, lapack_int* ilo,
                                           lapack_int* ihi, double* scale);
HWYBLAS_API lapack_int LAPACKE_cgebal_work(int matrix_layout, char job,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda, lapack_int* ilo,
                                           lapack_int* ihi, float* scale);
HWYBLAS_API lapack_int LAPACKE_zgebal_work(int matrix_layout, char job,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda, lapack_int* ilo,
                                           lapack_int* ihi, double* scale);

HWYBLAS_API lapack_int LAPACKE_sgebak(int matrix_layout, char job, char side,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, const float* scale,
                                      lapack_int m, float* v, lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_dgebak(int matrix_layout, char job, char side,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, const double* scale,
                                      lapack_int m, double* v, lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_cgebak(int matrix_layout, char job, char side,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, const float* scale,
                                      lapack_int m, lapack_complex_float* v,
                                      lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_zgebak(int matrix_layout, char job, char side,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, const double* scale,
                                      lapack_int m, lapack_complex_double* v,
                                      lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_sgebak_work(int matrix_layout, char job,
                                           char side, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           const float* scale, lapack_int m,
                                           float* v, lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_dgebak_work(int matrix_layout, char job,
                                           char side, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           const double* scale, lapack_int m,
                                           double* v, lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_cgebak_work(int matrix_layout, char job,
                                           char side, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           const float* scale, lapack_int m,
                                           lapack_complex_float* v,
                                           lapack_int ldv);
HWYBLAS_API lapack_int LAPACKE_zgebak_work(int matrix_layout, char job,
                                           char side, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           const double* scale, lapack_int m,
                                           lapack_complex_double* v,
                                           lapack_int ldv);

HWYBLAS_API lapack_int LAPACKE_sgehrd(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi, float* a,
                                      lapack_int lda, float* tau);
HWYBLAS_API lapack_int LAPACKE_dgehrd(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi, double* a,
                                      lapack_int lda, double* tau);
HWYBLAS_API lapack_int LAPACKE_cgehrd(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zgehrd(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sgehrd_work(int matrix_layout, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           float* a, lapack_int lda, float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgehrd_work(int matrix_layout, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           double* a, lapack_int lda,
                                           double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgehrd_work(
    int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgehrd_work(
    int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sorghr(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi, float* a,
                                      lapack_int lda, const float* tau);
HWYBLAS_API lapack_int LAPACKE_dorghr(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi, double* a,
                                      lapack_int lda, const double* tau);
HWYBLAS_API lapack_int LAPACKE_cunghr(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_zunghr(int matrix_layout, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_sorghr_work(int matrix_layout, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           float* a, lapack_int lda,
                                           const float* tau, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dorghr_work(int matrix_layout, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           double* a, lapack_int lda,
                                           const double* tau, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunghr_work(
    int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi,
    lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunghr_work(
    int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi,
    lapack_complex_double* a, lapack_int lda, const lapack_complex_double* tau,
    lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_sormhr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      const float* a, lapack_int lda,
                                      const float* tau, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormhr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      const double* a, lapack_int lda,
                                      const double* tau, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmhr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      const lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmhr(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n,
                                      lapack_int ilo, lapack_int ihi,
                                      const lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormhr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int ilo, lapack_int ihi, const float* a, lapack_int lda,
    const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormhr_work(int matrix_layout, char side,
                                           char trans, lapack_int m,
                                           lapack_int n, lapack_int ilo,
                                           lapack_int ihi, const double* a,
                                           lapack_int lda, const double* tau,
                                           double* c, lapack_int ldc,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmhr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int ilo, lapack_int ihi, const lapack_complex_float* a,
    lapack_int lda, const lapack_complex_float* tau, lapack_complex_float* c,
    lapack_int ldc, lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmhr_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int ilo, lapack_int ihi, const lapack_complex_double* a,
    lapack_int lda, const lapack_complex_double* tau, lapack_complex_double* c,
    lapack_int ldc, lapack_complex_double* work, lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_shseqr(int matrix_layout, char job, char compz,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, float* h, lapack_int ldh,
                                      float* wr, float* wi, float* z,
                                      lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_dhseqr(int matrix_layout, char job, char compz,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, double* h, lapack_int ldh,
                                      double* wr, double* wi, double* z,
                                      lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_chseqr(int matrix_layout, char job, char compz,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, lapack_complex_float* h,
                                      lapack_int ldh, lapack_complex_float* w,
                                      lapack_complex_float* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_zhseqr(int matrix_layout, char job, char compz,
                                      lapack_int n, lapack_int ilo,
                                      lapack_int ihi, lapack_complex_double* h,
                                      lapack_int ldh, lapack_complex_double* w,
                                      lapack_complex_double* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_shseqr_work(int matrix_layout, char job,
                                           char compz, lapack_int n,
                                           lapack_int ilo, lapack_int ihi,
                                           float* h, lapack_int ldh, float* wr,
                                           float* wi, float* z, lapack_int ldz,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dhseqr_work(
    int matrix_layout, char job, char compz, lapack_int n, lapack_int ilo,
    lapack_int ihi, double* h, lapack_int ldh, double* wr, double* wi,
    double* z, lapack_int ldz, double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_chseqr_work(
    int matrix_layout, char job, char compz, lapack_int n, lapack_int ilo,
    lapack_int ihi, lapack_complex_float* h, lapack_int ldh,
    lapack_complex_float* w, lapack_complex_float* z, lapack_int ldz,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zhseqr_work(
    int matrix_layout, char job, char compz, lapack_int n, lapack_int ilo,
    lapack_int ihi, lapack_complex_double* h, lapack_int ldh,
    lapack_complex_double* w, lapack_complex_double* z, lapack_int ldz,
    lapack_complex_double* work, lapack_int lwork);

/* ?trexc: ifst and ilst are pointers for real, values for complex types. */
HWYBLAS_API lapack_int LAPACKE_strexc(int matrix_layout, char compq,
                                      lapack_int n, float* t, lapack_int ldt,
                                      float* q, lapack_int ldq,
                                      lapack_int* ifst, lapack_int* ilst);
HWYBLAS_API lapack_int LAPACKE_dtrexc(int matrix_layout, char compq,
                                      lapack_int n, double* t, lapack_int ldt,
                                      double* q, lapack_int ldq,
                                      lapack_int* ifst, lapack_int* ilst);
HWYBLAS_API lapack_int LAPACKE_ctrexc(int matrix_layout, char compq,
                                      lapack_int n, lapack_complex_float* t,
                                      lapack_int ldt, lapack_complex_float* q,
                                      lapack_int ldq, lapack_int ifst,
                                      lapack_int ilst);
HWYBLAS_API lapack_int LAPACKE_ztrexc(int matrix_layout, char compq,
                                      lapack_int n, lapack_complex_double* t,
                                      lapack_int ldt, lapack_complex_double* q,
                                      lapack_int ldq, lapack_int ifst,
                                      lapack_int ilst);
HWYBLAS_API lapack_int LAPACKE_strexc_work(int matrix_layout, char compq,
                                           lapack_int n, float* t,
                                           lapack_int ldt, float* q,
                                           lapack_int ldq, lapack_int* ifst,
                                           lapack_int* ilst, float* work);
HWYBLAS_API lapack_int LAPACKE_dtrexc_work(int matrix_layout, char compq,
                                           lapack_int n, double* t,
                                           lapack_int ldt, double* q,
                                           lapack_int ldq, lapack_int* ifst,
                                           lapack_int* ilst, double* work);
HWYBLAS_API lapack_int LAPACKE_ctrexc_work(
    int matrix_layout, char compq, lapack_int n, lapack_complex_float* t,
    lapack_int ldt, lapack_complex_float* q, lapack_int ldq, lapack_int ifst,
    lapack_int ilst);
HWYBLAS_API lapack_int LAPACKE_ztrexc_work(
    int matrix_layout, char compq, lapack_int n, lapack_complex_double* t,
    lapack_int ldt, lapack_complex_double* q, lapack_int ldq, lapack_int ifst,
    lapack_int ilst);

HWYBLAS_API lapack_int LAPACKE_strsen(int matrix_layout, char job, char compq,
                                      const lapack_logical* select,
                                      lapack_int n, float* t, lapack_int ldt,
                                      float* q, lapack_int ldq, float* wr,
                                      float* wi, lapack_int* m, float* s,
                                      float* sep);
HWYBLAS_API lapack_int LAPACKE_dtrsen(int matrix_layout, char job, char compq,
                                      const lapack_logical* select,
                                      lapack_int n, double* t, lapack_int ldt,
                                      double* q, lapack_int ldq, double* wr,
                                      double* wi, lapack_int* m, double* s,
                                      double* sep);
HWYBLAS_API lapack_int LAPACKE_ctrsen(int matrix_layout, char job, char compq,
                                      const lapack_logical* select,
                                      lapack_int n, lapack_complex_float* t,
                                      lapack_int ldt, lapack_complex_float* q,
                                      lapack_int ldq, lapack_complex_float* w,
                                      lapack_int* m, float* s, float* sep);
HWYBLAS_API lapack_int LAPACKE_ztrsen(int matrix_layout, char job, char compq,
                                      const lapack_logical* select,
                                      lapack_int n, lapack_complex_double* t,
                                      lapack_int ldt, lapack_complex_double* q,
                                      lapack_int ldq, lapack_complex_double* w,
                                      lapack_int* m, double* s, double* sep);
HWYBLAS_API lapack_int LAPACKE_strsen_work(
    int matrix_layout, char job, char compq, const lapack_logical* select,
    lapack_int n, float* t, lapack_int ldt, float* q, lapack_int ldq, float* wr,
    float* wi, lapack_int* m, float* s, float* sep, float* work,
    lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dtrsen_work(
    int matrix_layout, char job, char compq, const lapack_logical* select,
    lapack_int n, double* t, lapack_int ldt, double* q, lapack_int ldq,
    double* wr, double* wi, lapack_int* m, double* s, double* sep, double* work,
    lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_ctrsen_work(
    int matrix_layout, char job, char compq, const lapack_logical* select,
    lapack_int n, lapack_complex_float* t, lapack_int ldt,
    lapack_complex_float* q, lapack_int ldq, lapack_complex_float* w,
    lapack_int* m, float* s, float* sep, lapack_complex_float* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_ztrsen_work(
    int matrix_layout, char job, char compq, const lapack_logical* select,
    lapack_int n, lapack_complex_double* t, lapack_int ldt,
    lapack_complex_double* q, lapack_int ldq, lapack_complex_double* w,
    lapack_int* m, double* s, double* sep, lapack_complex_double* work,
    lapack_int lwork);

HWYBLAS_API lapack_int LAPACKE_strsyl(int matrix_layout, char trana, char tranb,
                                      lapack_int isgn, lapack_int m,
                                      lapack_int n, const float* a,
                                      lapack_int lda, const float* b,
                                      lapack_int ldb, float* c, lapack_int ldc,
                                      float* scale);
HWYBLAS_API lapack_int LAPACKE_dtrsyl(int matrix_layout, char trana, char tranb,
                                      lapack_int isgn, lapack_int m,
                                      lapack_int n, const double* a,
                                      lapack_int lda, const double* b,
                                      lapack_int ldb, double* c, lapack_int ldc,
                                      double* scale);
HWYBLAS_API lapack_int
LAPACKE_ctrsyl(int matrix_layout, char trana, char tranb, lapack_int isgn,
               lapack_int m, lapack_int n, const lapack_complex_float* a,
               lapack_int lda, const lapack_complex_float* b, lapack_int ldb,
               lapack_complex_float* c, lapack_int ldc, float* scale);
HWYBLAS_API lapack_int
LAPACKE_ztrsyl(int matrix_layout, char trana, char tranb, lapack_int isgn,
               lapack_int m, lapack_int n, const lapack_complex_double* a,
               lapack_int lda, const lapack_complex_double* b, lapack_int ldb,
               lapack_complex_double* c, lapack_int ldc, double* scale);
HWYBLAS_API lapack_int LAPACKE_strsyl_work(
    int matrix_layout, char trana, char tranb, lapack_int isgn, lapack_int m,
    lapack_int n, const float* a, lapack_int lda, const float* b,
    lapack_int ldb, float* c, lapack_int ldc, float* scale);
HWYBLAS_API lapack_int LAPACKE_dtrsyl_work(
    int matrix_layout, char trana, char tranb, lapack_int isgn, lapack_int m,
    lapack_int n, const double* a, lapack_int lda, const double* b,
    lapack_int ldb, double* c, lapack_int ldc, double* scale);
HWYBLAS_API lapack_int LAPACKE_ctrsyl_work(
    int matrix_layout, char trana, char tranb, lapack_int isgn, lapack_int m,
    lapack_int n, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c,
    lapack_int ldc, float* scale);
HWYBLAS_API lapack_int LAPACKE_ztrsyl_work(
    int matrix_layout, char trana, char tranb, lapack_int isgn, lapack_int m,
    lapack_int n, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* c,
    lapack_int ldc, double* scale);

HWYBLAS_API lapack_int LAPACKE_slarfx(int matrix_layout, char side,
                                      lapack_int m, lapack_int n,
                                      const float* v, float tau, float* c,
                                      lapack_int ldc, float* work);
HWYBLAS_API lapack_int LAPACKE_dlarfx(int matrix_layout, char side,
                                      lapack_int m, lapack_int n,
                                      const double* v, double tau, double* c,
                                      lapack_int ldc, double* work);
HWYBLAS_API lapack_int LAPACKE_clarfx(int matrix_layout, char side,
                                      lapack_int m, lapack_int n,
                                      const lapack_complex_float* v,
                                      lapack_complex_float tau,
                                      lapack_complex_float* c, lapack_int ldc,
                                      lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zlarfx(int matrix_layout, char side,
                                      lapack_int m, lapack_int n,
                                      const lapack_complex_double* v,
                                      lapack_complex_double tau,
                                      lapack_complex_double* c, lapack_int ldc,
                                      lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_slarfx_work(int matrix_layout, char side,
                                           lapack_int m, lapack_int n,
                                           const float* v, float tau, float* c,
                                           lapack_int ldc, float* work);
HWYBLAS_API lapack_int LAPACKE_dlarfx_work(int matrix_layout, char side,
                                           lapack_int m, lapack_int n,
                                           const double* v, double tau,
                                           double* c, lapack_int ldc,
                                           double* work);
HWYBLAS_API lapack_int LAPACKE_clarfx_work(
    int matrix_layout, char side, lapack_int m, lapack_int n,
    const lapack_complex_float* v, lapack_complex_float tau,
    lapack_complex_float* c, lapack_int ldc, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zlarfx_work(
    int matrix_layout, char side, lapack_int m, lapack_int n,
    const lapack_complex_double* v, lapack_complex_double tau,
    lapack_complex_double* c, lapack_int ldc, lapack_complex_double* work);

HWYBLAS_API lapack_int LAPACKE_slacn2(lapack_int n, float* v, float* x,
                                      lapack_int* isgn, float* est,
                                      lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_dlacn2(lapack_int n, double* v, double* x,
                                      lapack_int* isgn, double* est,
                                      lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_slacn2_work(lapack_int n, float* v, float* x,
                                           lapack_int* isgn, float* est,
                                           lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_dlacn2_work(lapack_int n, double* v, double* x,
                                           lapack_int* isgn, double* est,
                                           lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_clacn2(lapack_int n, lapack_complex_float* v,
                                      lapack_complex_float* x, float* est,
                                      lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_zlacn2(lapack_int n, lapack_complex_double* v,
                                      lapack_complex_double* x, double* est,
                                      lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_clacn2_work(lapack_int n,
                                           lapack_complex_float* v,
                                           lapack_complex_float* x, float* est,
                                           lapack_int* kase, lapack_int* isave);
HWYBLAS_API lapack_int LAPACKE_zlacn2_work(lapack_int n,
                                           lapack_complex_double* v,
                                           lapack_complex_double* x,
                                           double* est, lapack_int* kase,
                                           lapack_int* isave);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACKE_H_ */
