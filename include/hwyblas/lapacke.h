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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACKE_H_ */
