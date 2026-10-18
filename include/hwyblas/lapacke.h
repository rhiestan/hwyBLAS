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
/* LU factorization (see xGETRF in lapack.h)                                */

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_LAPACKE_H_ */
