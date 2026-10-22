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
/* Linear systems (see lapack.h)                                            */

HWYBLAS_API lapack_int LAPACKE_sgetrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda,
                                      const lapack_int* ipiv, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgetrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      const lapack_int* ipiv, double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgetrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgetrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgetrs_work(int matrix_layout, char trans,
                                           lapack_int n, lapack_int nrhs,
                                           const float* a, lapack_int lda,
                                           const lapack_int* ipiv, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgetrs_work(int matrix_layout, char trans,
                                           lapack_int n, lapack_int nrhs,
                                           const double* a, lapack_int lda,
                                           const lapack_int* ipiv, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgetrs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgetrs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgesv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, float* a, lapack_int lda,
                                     lapack_int* ipiv, float* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgesv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, double* a, lapack_int lda,
                                     lapack_int* ipiv, double* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgesv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float* a,
                                     lapack_int lda, lapack_int* ipiv,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgesv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double* a,
                                     lapack_int lda, lapack_int* ipiv,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgesv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, float* a,
                                          lapack_int lda, lapack_int* ipiv,
                                          float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgesv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, double* a,
                                          lapack_int lda, lapack_int* ipiv,
                                          double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgesv_work(
    int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_float* a,
    lapack_int lda, lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgesv_work(
    int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* a,
    lapack_int lda, lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgetri(int matrix_layout, lapack_int n, float* a,
                                      lapack_int lda, const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgetri(int matrix_layout, lapack_int n,
                                      double* a, lapack_int lda,
                                      const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgetri(int matrix_layout, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgetri(int matrix_layout, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgetri_work(int matrix_layout, lapack_int n,
                                           float* a, lapack_int lda,
                                           const lapack_int* ipiv, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgetri_work(int matrix_layout, lapack_int n,
                                           double* a, lapack_int lda,
                                           const lapack_int* ipiv, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgetri_work(
    int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda,
    const lapack_int* ipiv, lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgetri_work(
    int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda,
    const lapack_int* ipiv, lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sgecon(int matrix_layout, char norm,
                                      lapack_int n, const float* a,
                                      lapack_int lda, float anorm,
                                      float* rcond);
HWYBLAS_API lapack_int LAPACKE_dgecon(int matrix_layout, char norm,
                                      lapack_int n, const double* a,
                                      lapack_int lda, double anorm,
                                      double* rcond);
HWYBLAS_API lapack_int LAPACKE_cgecon(int matrix_layout, char norm,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float anorm,
                                      float* rcond);
HWYBLAS_API lapack_int LAPACKE_zgecon(int matrix_layout, char norm,
                                      lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, double anorm,
                                      double* rcond);
HWYBLAS_API lapack_int LAPACKE_sgecon_work(int matrix_layout, char norm,
                                           lapack_int n, const float* a,
                                           lapack_int lda, float anorm,
                                           float* rcond, float* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgecon_work(int matrix_layout, char norm,
                                           lapack_int n, const double* a,
                                           lapack_int lda, double anorm,
                                           double* rcond, double* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int
LAPACKE_cgecon_work(int matrix_layout, char norm, lapack_int n,
                    const lapack_complex_float* a, lapack_int lda, float anorm,
                    float* rcond, lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgecon_work(
    int matrix_layout, char norm, lapack_int n, const lapack_complex_double* a,
    lapack_int lda, double anorm, double* rcond, lapack_complex_double* work,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_sgeequ(int matrix_layout, lapack_int m,
                                      lapack_int n, const float* a,
                                      lapack_int lda, float* r, float* c,
                                      float* rowcnd, float* colcnd,
                                      float* amax);
HWYBLAS_API lapack_int LAPACKE_dgeequ(int matrix_layout, lapack_int m,
                                      lapack_int n, const double* a,
                                      lapack_int lda, double* r, double* c,
                                      double* rowcnd, double* colcnd,
                                      double* amax);
HWYBLAS_API lapack_int LAPACKE_cgeequ(int matrix_layout, lapack_int m,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* r, float* c,
                                      float* rowcnd, float* colcnd,
                                      float* amax);
HWYBLAS_API lapack_int LAPACKE_zgeequ(int matrix_layout, lapack_int m,
                                      lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, double* r, double* c,
                                      double* rowcnd, double* colcnd,
                                      double* amax);
HWYBLAS_API lapack_int LAPACKE_sgeequ_work(int matrix_layout, lapack_int m,
                                           lapack_int n, const float* a,
                                           lapack_int lda, float* r, float* c,
                                           float* rowcnd, float* colcnd,
                                           float* amax);
HWYBLAS_API lapack_int LAPACKE_dgeequ_work(int matrix_layout, lapack_int m,
                                           lapack_int n, const double* a,
                                           lapack_int lda, double* r, double* c,
                                           double* rowcnd, double* colcnd,
                                           double* amax);
HWYBLAS_API lapack_int LAPACKE_cgeequ_work(int matrix_layout, lapack_int m,
                                           lapack_int n,
                                           const lapack_complex_float* a,
                                           lapack_int lda, float* r, float* c,
                                           float* rowcnd, float* colcnd,
                                           float* amax);
HWYBLAS_API lapack_int LAPACKE_zgeequ_work(int matrix_layout, lapack_int m,
                                           lapack_int n,
                                           const lapack_complex_double* a,
                                           lapack_int lda, double* r, double* c,
                                           double* rowcnd, double* colcnd,
                                           double* amax);
HWYBLAS_API lapack_int LAPACKE_sgerfs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda,
                                      const float* af, lapack_int ldaf,
                                      const lapack_int* ipiv, const float* b,
                                      lapack_int ldb, float* x, lapack_int ldx,
                                      float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_dgerfs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      const double* af, lapack_int ldaf,
                                      const lapack_int* ipiv, const double* b,
                                      lapack_int ldb, double* x, lapack_int ldx,
                                      double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_cgerfs(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zgerfs(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_sgerfs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const float* a, lapack_int lda, const float* af, lapack_int ldaf,
    const lapack_int* ipiv, const float* b, lapack_int ldb, float* x,
    lapack_int ldx, float* ferr, float* berr, float* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgerfs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const double* a, lapack_int lda, const double* af, lapack_int ldaf,
    const lapack_int* ipiv, const double* b, lapack_int ldb, double* x,
    lapack_int ldx, double* ferr, double* berr, double* work,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cgerfs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work,
    float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgerfs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work,
    double* rwork);

/* In row-major layout, B is also copied back for fact = 'E' (the
 * reference LAPACKE copies it back only for fact = 'F', although it is
 * scaled for 'E' too). rpivot: the reciprocal pivot growth factor. */
HWYBLAS_API lapack_int LAPACKE_sgesvx(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    float* a, lapack_int lda, float* af, lapack_int ldaf, lapack_int* ipiv,
    char* equed, float* r, float* c, float* b, lapack_int ldb, float* x,
    lapack_int ldx, float* rcond, float* ferr, float* berr, float* rpivot);
HWYBLAS_API lapack_int LAPACKE_dgesvx(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    double* a, lapack_int lda, double* af, lapack_int ldaf, lapack_int* ipiv,
    char* equed, double* r, double* c, double* b, lapack_int ldb, double* x,
    lapack_int ldx, double* rcond, double* ferr, double* berr, double* rpivot);
HWYBLAS_API lapack_int LAPACKE_cgesvx(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, float* c,
    lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* rcond, float* ferr, float* berr, float* rpivot);
HWYBLAS_API lapack_int LAPACKE_zgesvx(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, double* c,
    lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* rcond, double* ferr, double* berr, double* rpivot);
HWYBLAS_API lapack_int LAPACKE_sgesvx_work(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    float* a, lapack_int lda, float* af, lapack_int ldaf, lapack_int* ipiv,
    char* equed, float* r, float* c, float* b, lapack_int ldb, float* x,
    lapack_int ldx, float* rcond, float* ferr, float* berr, float* work,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgesvx_work(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    double* a, lapack_int lda, double* af, lapack_int ldaf, lapack_int* ipiv,
    char* equed, double* r, double* c, double* b, lapack_int ldb, double* x,
    lapack_int ldx, double* rcond, double* ferr, double* berr, double* work,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cgesvx_work(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, lapack_int* ipiv, char* equed, float* r, float* c,
    lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* rcond, float* ferr, float* berr,
    lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgesvx_work(
    int matrix_layout, char fact, char trans, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, lapack_int* ipiv, char* equed, double* r, double* c,
    lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* rcond, double* ferr, double* berr,
    lapack_complex_double* work, double* rwork);

HWYBLAS_API lapack_int LAPACKE_strtri(int matrix_layout, char uplo, char diag,
                                      lapack_int n, float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dtrtri(int matrix_layout, char uplo, char diag,
                                      lapack_int n, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_ctrtri(int matrix_layout, char uplo, char diag,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_ztrtri(int matrix_layout, char uplo, char diag,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_strtri_work(int matrix_layout, char uplo,
                                           char diag, lapack_int n, float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dtrtri_work(int matrix_layout, char uplo,
                                           char diag, lapack_int n, double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_ctrtri_work(int matrix_layout, char uplo,
                                           char diag, lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_ztrtri_work(int matrix_layout, char uplo,
                                           char diag, lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_strtrs(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ctrtrs(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float* a,
                                      lapack_int lda, lapack_complex_float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ztrtrs(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double* a,
                                      lapack_int lda, lapack_complex_double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_strtrs_work(int matrix_layout, char uplo,
                                           char trans, char diag, lapack_int n,
                                           lapack_int nrhs, const float* a,
                                           lapack_int lda, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dtrtrs_work(int matrix_layout, char uplo,
                                           char trans, char diag, lapack_int n,
                                           lapack_int nrhs, const double* a,
                                           lapack_int lda, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ctrtrs_work(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const lapack_complex_float* a, lapack_int lda,
    lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ztrtrs_work(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const lapack_complex_double* a, lapack_int lda,
    lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_strcon(int matrix_layout, char norm, char uplo,
                                      char diag, lapack_int n, const float* a,
                                      lapack_int lda, float* rcond);
HWYBLAS_API lapack_int LAPACKE_dtrcon(int matrix_layout, char norm, char uplo,
                                      char diag, lapack_int n, const double* a,
                                      lapack_int lda, double* rcond);
HWYBLAS_API lapack_int LAPACKE_ctrcon(int matrix_layout, char norm, char uplo,
                                      char diag, lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* rcond);
HWYBLAS_API lapack_int LAPACKE_ztrcon(int matrix_layout, char norm, char uplo,
                                      char diag, lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, double* rcond);
HWYBLAS_API lapack_int LAPACKE_strcon_work(int matrix_layout, char norm,
                                           char uplo, char diag, lapack_int n,
                                           const float* a, lapack_int lda,
                                           float* rcond, float* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dtrcon_work(int matrix_layout, char norm,
                                           char uplo, char diag, lapack_int n,
                                           const double* a, lapack_int lda,
                                           double* rcond, double* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_ctrcon_work(int matrix_layout, char norm,
                                           char uplo, char diag, lapack_int n,
                                           const lapack_complex_float* a,
                                           lapack_int lda, float* rcond,
                                           lapack_complex_float* work,
                                           float* rwork);
HWYBLAS_API lapack_int LAPACKE_ztrcon_work(int matrix_layout, char norm,
                                           char uplo, char diag, lapack_int n,
                                           const lapack_complex_double* a,
                                           lapack_int lda, double* rcond,
                                           lapack_complex_double* work,
                                           double* rwork);
HWYBLAS_API lapack_int LAPACKE_strrfs(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda,
                                      const float* b, lapack_int ldb,
                                      const float* x, lapack_int ldx,
                                      float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_dtrrfs(int matrix_layout, char uplo, char trans,
                                      char diag, lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      const double* b, lapack_int ldb,
                                      const double* x, lapack_int ldx,
                                      double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_ctrrfs(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* b, lapack_int ldb,
    const lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_ztrrfs(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* b, lapack_int ldb,
    const lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_strrfs_work(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const float* a, lapack_int lda, const float* b,
    lapack_int ldb, const float* x, lapack_int ldx, float* ferr, float* berr,
    float* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dtrrfs_work(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const double* a, lapack_int lda, const double* b,
    lapack_int ldb, const double* x, lapack_int ldx, double* ferr, double* berr,
    double* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_ctrrfs_work(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* b, lapack_int ldb,
    const lapack_complex_float* x, lapack_int ldx, float* ferr, float* berr,
    lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int LAPACKE_ztrrfs_work(
    int matrix_layout, char uplo, char trans, char diag, lapack_int n,
    lapack_int nrhs, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* b, lapack_int ldb,
    const lapack_complex_double* x, lapack_int ldx, double* ferr, double* berr,
    lapack_complex_double* work, double* rwork);

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
/* Symmetric (Hermitian) positive definite systems (see lapack.h)           */

HWYBLAS_API lapack_int LAPACKE_spotrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpotrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cpotrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float* a,
                                      lapack_int lda, lapack_complex_float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zpotrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double* a,
                                      lapack_int lda, lapack_complex_double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_spotrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const float* a, lapack_int lda,
                                           float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpotrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const double* a, lapack_int lda,
                                           double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cpotrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const lapack_complex_float* a,
                                           lapack_int lda,
                                           lapack_complex_float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zpotrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const lapack_complex_double* a,
                                           lapack_int lda,
                                           lapack_complex_double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sposv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, float* a, lapack_int lda,
                                     float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dposv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, double* a, lapack_int lda,
                                     double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cposv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float* a,
                                     lapack_int lda, lapack_complex_float* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zposv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double* a,
                                     lapack_int lda, lapack_complex_double* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sposv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int nrhs,
                                          float* a, lapack_int lda, float* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dposv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int nrhs,
                                          double* a, lapack_int lda, double* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cposv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int nrhs,
                                          lapack_complex_float* a,
                                          lapack_int lda,
                                          lapack_complex_float* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zposv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int nrhs,
                                          lapack_complex_double* a,
                                          lapack_int lda,
                                          lapack_complex_double* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_slauum(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dlauum(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_clauum(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zlauum(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_slauum_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dlauum_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_clauum_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zlauum_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_spotri(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dpotri(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_cpotri(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zpotri(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_spotri_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_dpotri_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_cpotri_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_zpotri_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda);
HWYBLAS_API lapack_int LAPACKE_spocon(int matrix_layout, char uplo,
                                      lapack_int n, const float* a,
                                      lapack_int lda, float anorm,
                                      float* rcond);
HWYBLAS_API lapack_int LAPACKE_dpocon(int matrix_layout, char uplo,
                                      lapack_int n, const double* a,
                                      lapack_int lda, double anorm,
                                      double* rcond);
HWYBLAS_API lapack_int LAPACKE_cpocon(int matrix_layout, char uplo,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float anorm,
                                      float* rcond);
HWYBLAS_API lapack_int LAPACKE_zpocon(int matrix_layout, char uplo,
                                      lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, double anorm,
                                      double* rcond);
HWYBLAS_API lapack_int LAPACKE_spocon_work(int matrix_layout, char uplo,
                                           lapack_int n, const float* a,
                                           lapack_int lda, float anorm,
                                           float* rcond, float* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dpocon_work(int matrix_layout, char uplo,
                                           lapack_int n, const double* a,
                                           lapack_int lda, double anorm,
                                           double* rcond, double* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int
LAPACKE_cpocon_work(int matrix_layout, char uplo, lapack_int n,
                    const lapack_complex_float* a, lapack_int lda, float anorm,
                    float* rcond, lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zpocon_work(
    int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a,
    lapack_int lda, double anorm, double* rcond, lapack_complex_double* work,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_spoequ(int matrix_layout, lapack_int n,
                                      const float* a, lapack_int lda, float* s,
                                      float* scond, float* amax);
HWYBLAS_API lapack_int LAPACKE_dpoequ(int matrix_layout, lapack_int n,
                                      const double* a, lapack_int lda,
                                      double* s, double* scond, double* amax);
HWYBLAS_API lapack_int LAPACKE_cpoequ(int matrix_layout, lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* s, float* scond,
                                      float* amax);
HWYBLAS_API lapack_int LAPACKE_zpoequ(int matrix_layout, lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, double* s, double* scond,
                                      double* amax);
HWYBLAS_API lapack_int LAPACKE_spoequ_work(int matrix_layout, lapack_int n,
                                           const float* a, lapack_int lda,
                                           float* s, float* scond, float* amax);
HWYBLAS_API lapack_int LAPACKE_dpoequ_work(int matrix_layout, lapack_int n,
                                           const double* a, lapack_int lda,
                                           double* s, double* scond,
                                           double* amax);
HWYBLAS_API lapack_int LAPACKE_cpoequ_work(int matrix_layout, lapack_int n,
                                           const lapack_complex_float* a,
                                           lapack_int lda, float* s,
                                           float* scond, float* amax);
HWYBLAS_API lapack_int LAPACKE_zpoequ_work(int matrix_layout, lapack_int n,
                                           const lapack_complex_double* a,
                                           lapack_int lda, double* s,
                                           double* scond, double* amax);
HWYBLAS_API lapack_int LAPACKE_sporfs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda,
                                      const float* af, lapack_int ldaf,
                                      const float* b, lapack_int ldb, float* x,
                                      lapack_int ldx, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_dporfs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      const double* af, lapack_int ldaf,
                                      const double* b, lapack_int ldb,
                                      double* x, lapack_int ldx, double* ferr,
                                      double* berr);
HWYBLAS_API lapack_int LAPACKE_cporfs(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zporfs(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_sporfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, const float* a,
    lapack_int lda, const float* af, lapack_int ldaf, const float* b,
    lapack_int ldb, float* x, lapack_int ldx, float* ferr, float* berr,
    float* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dporfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const double* a, lapack_int lda, const double* af, lapack_int ldaf,
    const double* b, lapack_int ldb, double* x, lapack_int ldx, double* ferr,
    double* berr, double* work, lapack_int* iwork);
HWYBLAS_API lapack_int
LAPACKE_cporfs_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                    const lapack_complex_float* a, lapack_int lda,
                    const lapack_complex_float* af, lapack_int ldaf,
                    const lapack_complex_float* b, lapack_int ldb,
                    lapack_complex_float* x, lapack_int ldx, float* ferr,
                    float* berr, lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int
LAPACKE_zporfs_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                    const lapack_complex_double* a, lapack_int lda,
                    const lapack_complex_double* af, lapack_int ldaf,
                    const lapack_complex_double* b, lapack_int ldb,
                    lapack_complex_double* x, lapack_int ldx, double* ferr,
                    double* berr, lapack_complex_double* work, double* rwork);
HWYBLAS_API lapack_int LAPACKE_sposvx(int matrix_layout, char fact, char uplo,
                                      lapack_int n, lapack_int nrhs, float* a,
                                      lapack_int lda, float* af,
                                      lapack_int ldaf, char* equed, float* s,
                                      float* b, lapack_int ldb, float* x,
                                      lapack_int ldx, float* rcond, float* ferr,
                                      float* berr);
HWYBLAS_API lapack_int LAPACKE_dposvx(int matrix_layout, char fact, char uplo,
                                      lapack_int n, lapack_int nrhs, double* a,
                                      lapack_int lda, double* af,
                                      lapack_int ldaf, char* equed, double* s,
                                      double* b, lapack_int ldb, double* x,
                                      lapack_int ldx, double* rcond,
                                      double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_cposvx(int matrix_layout, char fact, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* af, lapack_int ldaf,
                                      char* equed, float* s,
                                      lapack_complex_float* b, lapack_int ldb,
                                      lapack_complex_float* x, lapack_int ldx,
                                      float* rcond, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zposvx(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, char* equed, double* s, lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond,
    double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_sposvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    float* a, lapack_int lda, float* af, lapack_int ldaf, char* equed, float* s,
    float* b, lapack_int ldb, float* x, lapack_int ldx, float* rcond,
    float* ferr, float* berr, float* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dposvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    double* a, lapack_int lda, double* af, lapack_int ldaf, char* equed,
    double* s, double* b, lapack_int ldb, double* x, lapack_int ldx,
    double* rcond, double* ferr, double* berr, double* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cposvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, char* equed, float* s, lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond,
    float* ferr, float* berr, lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zposvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, char* equed, double* s, lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond,
    double* ferr, double* berr, lapack_complex_double* work, double* rwork);
HWYBLAS_API lapack_int LAPACKE_spstrf(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda,
                                      lapack_int* piv, lapack_int* rank,
                                      float tol);
HWYBLAS_API lapack_int LAPACKE_dpstrf(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda,
                                      lapack_int* piv, lapack_int* rank,
                                      double tol);
HWYBLAS_API lapack_int LAPACKE_cpstrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int* piv,
                                      lapack_int* rank, float tol);
HWYBLAS_API lapack_int LAPACKE_zpstrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, lapack_int* piv,
                                      lapack_int* rank, double tol);
HWYBLAS_API lapack_int LAPACKE_spstrf_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda, lapack_int* piv,
                                           lapack_int* rank, float tol,
                                           float* work);
HWYBLAS_API lapack_int LAPACKE_dpstrf_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda, lapack_int* piv,
                                           lapack_int* rank, double tol,
                                           double* work);
HWYBLAS_API lapack_int LAPACKE_cpstrf_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_int* piv, lapack_int* rank, float tol, float* work);
HWYBLAS_API lapack_int LAPACKE_zpstrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda, lapack_int* piv,
                                           lapack_int* rank, double tol,
                                           double* work);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) indefinite systems (see lapack.h)                  */

HWYBLAS_API lapack_int LAPACKE_ssytrf(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dsytrf(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_csytrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zsytrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_ssytrf_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda, lapack_int* ipiv,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dsytrf_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda, lapack_int* ipiv,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_csytrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda, lapack_int* ipiv,
                                           lapack_complex_float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zsytrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda, lapack_int* ipiv,
                                           lapack_complex_double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_chetrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zhetrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_chetrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda, lapack_int* ipiv,
                                           lapack_complex_float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zhetrf_work(int matrix_layout, char uplo,
                                           lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda, lapack_int* ipiv,
                                           lapack_complex_double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_ssytrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda,
                                      const lapack_int* ipiv, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dsytrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      const lapack_int* ipiv, double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_csytrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zsytrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ssytrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const float* a, lapack_int lda,
                                           const lapack_int* ipiv, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dsytrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const double* a, lapack_int lda,
                                           const lapack_int* ipiv, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_csytrs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zsytrs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_chetrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zhetrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_chetrs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zhetrs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ssytrs2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_int nrhs,
                                       const float* a, lapack_int lda,
                                       const lapack_int* ipiv, float* b,
                                       lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dsytrs2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_int nrhs,
                                       const double* a, lapack_int lda,
                                       const lapack_int* ipiv, double* b,
                                       lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_csytrs2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_int nrhs,
                                       const lapack_complex_float* a,
                                       lapack_int lda, const lapack_int* ipiv,
                                       lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zsytrs2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_int nrhs,
                                       const lapack_complex_double* a,
                                       lapack_int lda, const lapack_int* ipiv,
                                       lapack_complex_double* b,
                                       lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ssytrs2_work(int matrix_layout, char uplo,
                                            lapack_int n, lapack_int nrhs,
                                            const float* a, lapack_int lda,
                                            const lapack_int* ipiv, float* b,
                                            lapack_int ldb, float* work);
HWYBLAS_API lapack_int LAPACKE_dsytrs2_work(int matrix_layout, char uplo,
                                            lapack_int n, lapack_int nrhs,
                                            const double* a, lapack_int lda,
                                            const lapack_int* ipiv, double* b,
                                            lapack_int ldb, double* work);
HWYBLAS_API lapack_int LAPACKE_csytrs2_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_float* b, lapack_int ldb, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zsytrs2_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_double* b, lapack_int ldb, lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_chetrs2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_int nrhs,
                                       const lapack_complex_float* a,
                                       lapack_int lda, const lapack_int* ipiv,
                                       lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zhetrs2(int matrix_layout, char uplo,
                                       lapack_int n, lapack_int nrhs,
                                       const lapack_complex_double* a,
                                       lapack_int lda, const lapack_int* ipiv,
                                       lapack_complex_double* b,
                                       lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_chetrs2_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_float* b, lapack_int ldb, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zhetrs2_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv,
    lapack_complex_double* b, lapack_int ldb, lapack_complex_double* work);

/* In row-major layout, ?syconv transposes the n x n matrix A.
 * (The reference transposes lda rows, reading and writing past the n rows
 * of the array if lda > n.) */
HWYBLAS_API lapack_int LAPACKE_ssyconv(int matrix_layout, char uplo, char way,
                                       lapack_int n, float* a, lapack_int lda,
                                       const lapack_int* ipiv, float* e);
HWYBLAS_API lapack_int LAPACKE_dsyconv(int matrix_layout, char uplo, char way,
                                       lapack_int n, double* a, lapack_int lda,
                                       const lapack_int* ipiv, double* e);
HWYBLAS_API lapack_int LAPACKE_csyconv(int matrix_layout, char uplo, char way,
                                       lapack_int n, lapack_complex_float* a,
                                       lapack_int lda, const lapack_int* ipiv,
                                       lapack_complex_float* e);
HWYBLAS_API lapack_int LAPACKE_zsyconv(int matrix_layout, char uplo, char way,
                                       lapack_int n, lapack_complex_double* a,
                                       lapack_int lda, const lapack_int* ipiv,
                                       lapack_complex_double* e);
HWYBLAS_API lapack_int LAPACKE_ssyconv_work(int matrix_layout, char uplo,
                                            char way, lapack_int n, float* a,
                                            lapack_int lda,
                                            const lapack_int* ipiv, float* e);
HWYBLAS_API lapack_int LAPACKE_dsyconv_work(int matrix_layout, char uplo,
                                            char way, lapack_int n, double* a,
                                            lapack_int lda,
                                            const lapack_int* ipiv, double* e);
HWYBLAS_API lapack_int LAPACKE_csyconv_work(int matrix_layout, char uplo,
                                            char way, lapack_int n,
                                            lapack_complex_float* a,
                                            lapack_int lda,
                                            const lapack_int* ipiv,
                                            lapack_complex_float* e);
HWYBLAS_API lapack_int LAPACKE_zsyconv_work(int matrix_layout, char uplo,
                                            char way, lapack_int n,
                                            lapack_complex_double* a,
                                            lapack_int lda,
                                            const lapack_int* ipiv,
                                            lapack_complex_double* e);

HWYBLAS_API lapack_int LAPACKE_ssytri(int matrix_layout, char uplo,
                                      lapack_int n, float* a, lapack_int lda,
                                      const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dsytri(int matrix_layout, char uplo,
                                      lapack_int n, double* a, lapack_int lda,
                                      const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_csytri(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zsytri(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_ssytri_work(int matrix_layout, char uplo,
                                           lapack_int n, float* a,
                                           lapack_int lda,
                                           const lapack_int* ipiv, float* work);
HWYBLAS_API lapack_int LAPACKE_dsytri_work(int matrix_layout, char uplo,
                                           lapack_int n, double* a,
                                           lapack_int lda,
                                           const lapack_int* ipiv,
                                           double* work);
HWYBLAS_API lapack_int LAPACKE_csytri_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a,
    lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zsytri_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a,
    lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_chetri(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zhetri(int matrix_layout, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_chetri_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a,
    lapack_int lda, const lapack_int* ipiv, lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zhetri_work(
    int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a,
    lapack_int lda, const lapack_int* ipiv, lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_ssycon(int matrix_layout, char uplo,
                                      lapack_int n, const float* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      float anorm, float* rcond);
HWYBLAS_API lapack_int LAPACKE_dsycon(int matrix_layout, char uplo,
                                      lapack_int n, const double* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      double anorm, double* rcond);
HWYBLAS_API lapack_int LAPACKE_csycon(int matrix_layout, char uplo,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      float anorm, float* rcond);
HWYBLAS_API lapack_int LAPACKE_zsycon(int matrix_layout, char uplo,
                                      lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      double anorm, double* rcond);
HWYBLAS_API lapack_int LAPACKE_ssycon_work(int matrix_layout, char uplo,
                                           lapack_int n, const float* a,
                                           lapack_int lda,
                                           const lapack_int* ipiv, float anorm,
                                           float* rcond, float* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dsycon_work(int matrix_layout, char uplo,
                                           lapack_int n, const double* a,
                                           lapack_int lda,
                                           const lapack_int* ipiv, double anorm,
                                           double* rcond, double* work,
                                           lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_csycon_work(
    int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a,
    lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond,
    lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zsycon_work(
    int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a,
    lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond,
    lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_checon(int matrix_layout, char uplo,
                                      lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      float anorm, float* rcond);
HWYBLAS_API lapack_int LAPACKE_zhecon(int matrix_layout, char uplo,
                                      lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda, const lapack_int* ipiv,
                                      double anorm, double* rcond);
HWYBLAS_API lapack_int LAPACKE_checon_work(
    int matrix_layout, char uplo, lapack_int n, const lapack_complex_float* a,
    lapack_int lda, const lapack_int* ipiv, float anorm, float* rcond,
    lapack_complex_float* work);
HWYBLAS_API lapack_int LAPACKE_zhecon_work(
    int matrix_layout, char uplo, lapack_int n, const lapack_complex_double* a,
    lapack_int lda, const lapack_int* ipiv, double anorm, double* rcond,
    lapack_complex_double* work);
HWYBLAS_API lapack_int LAPACKE_ssyrfs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda,
                                      const float* af, lapack_int ldaf,
                                      const lapack_int* ipiv, const float* b,
                                      lapack_int ldb, float* x, lapack_int ldx,
                                      float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_dsyrfs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda,
                                      const double* af, lapack_int ldaf,
                                      const lapack_int* ipiv, const double* b,
                                      lapack_int ldb, double* x, lapack_int ldx,
                                      double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_csyrfs(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zsyrfs(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_ssyrfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, const float* a,
    lapack_int lda, const float* af, lapack_int ldaf, const lapack_int* ipiv,
    const float* b, lapack_int ldb, float* x, lapack_int ldx, float* ferr,
    float* berr, float* work, lapack_int* iwork);
HWYBLAS_API lapack_int
LAPACKE_dsyrfs_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                    const double* a, lapack_int lda, const double* af,
                    lapack_int ldaf, const lapack_int* ipiv, const double* b,
                    lapack_int ldb, double* x, lapack_int ldx, double* ferr,
                    double* berr, double* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_csyrfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work,
    float* rwork);
HWYBLAS_API lapack_int LAPACKE_zsyrfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_cherfs(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zherfs(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_cherfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* x,
    lapack_int ldx, float* ferr, float* berr, lapack_complex_float* work,
    float* rwork);
HWYBLAS_API lapack_int LAPACKE_zherfs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* af, lapack_int ldaf, const lapack_int* ipiv,
    const lapack_complex_double* b, lapack_int ldb, lapack_complex_double* x,
    lapack_int ldx, double* ferr, double* berr, lapack_complex_double* work,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_ssysv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, float* a, lapack_int lda,
                                     lapack_int* ipiv, float* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dsysv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, double* a, lapack_int lda,
                                     lapack_int* ipiv, double* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_csysv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float* a,
                                     lapack_int lda, lapack_int* ipiv,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zsysv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double* a,
                                     lapack_int lda, lapack_int* ipiv,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ssysv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int nrhs,
                                          float* a, lapack_int lda,
                                          lapack_int* ipiv, float* b,
                                          lapack_int ldb, float* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dsysv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int nrhs,
                                          double* a, lapack_int lda,
                                          lapack_int* ipiv, double* b,
                                          lapack_int ldb, double* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int
LAPACKE_csysv_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                   lapack_complex_float* a, lapack_int lda, lapack_int* ipiv,
                   lapack_complex_float* b, lapack_int ldb,
                   lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int
LAPACKE_zsysv_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                   lapack_complex_double* a, lapack_int lda, lapack_int* ipiv,
                   lapack_complex_double* b, lapack_int ldb,
                   lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_chesv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float* a,
                                     lapack_int lda, lapack_int* ipiv,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zhesv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double* a,
                                     lapack_int lda, lapack_int* ipiv,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int
LAPACKE_chesv_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                   lapack_complex_float* a, lapack_int lda, lapack_int* ipiv,
                   lapack_complex_float* b, lapack_int ldb,
                   lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int
LAPACKE_zhesv_work(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                   lapack_complex_double* a, lapack_int lda, lapack_int* ipiv,
                   lapack_complex_double* b, lapack_int ldb,
                   lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_ssysvx(int matrix_layout, char fact, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda, float* af,
                                      lapack_int ldaf, lapack_int* ipiv,
                                      const float* b, lapack_int ldb, float* x,
                                      lapack_int ldx, float* rcond, float* ferr,
                                      float* berr);
HWYBLAS_API lapack_int LAPACKE_dsysvx(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const double* a, lapack_int lda, double* af, lapack_int ldaf,
    lapack_int* ipiv, const double* b, lapack_int ldb, double* x,
    lapack_int ldx, double* rcond, double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_csysvx(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond,
    float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zsysvx(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond,
    double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_ssysvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const float* a, lapack_int lda, float* af, lapack_int ldaf,
    lapack_int* ipiv, const float* b, lapack_int ldb, float* x, lapack_int ldx,
    float* rcond, float* ferr, float* berr, float* work, lapack_int lwork,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dsysvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const double* a, lapack_int lda, double* af, lapack_int ldaf,
    lapack_int* ipiv, const double* b, lapack_int ldb, double* x,
    lapack_int ldx, double* rcond, double* ferr, double* berr, double* work,
    lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_csysvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond,
    float* ferr, float* berr, lapack_complex_float* work, lapack_int lwork,
    float* rwork);
HWYBLAS_API lapack_int LAPACKE_zsysvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond,
    double* ferr, double* berr, lapack_complex_double* work, lapack_int lwork,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_chesvx(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond,
    float* ferr, float* berr);
HWYBLAS_API lapack_int LAPACKE_zhesvx(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond,
    double* ferr, double* berr);
HWYBLAS_API lapack_int LAPACKE_chesvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* a, lapack_int lda, lapack_complex_float* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* x, lapack_int ldx, float* rcond,
    float* ferr, float* berr, lapack_complex_float* work, lapack_int lwork,
    float* rwork);
HWYBLAS_API lapack_int LAPACKE_zhesvx_work(
    int matrix_layout, char fact, char uplo, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* a, lapack_int lda, lapack_complex_double* af,
    lapack_int ldaf, lapack_int* ipiv, const lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* x, lapack_int ldx, double* rcond,
    double* ferr, double* berr, lapack_complex_double* work, lapack_int lwork,
    double* rwork);

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

HWYBLAS_API float LAPACKE_slantr(int matrix_layout, char norm, char uplo,
                                 char diag, lapack_int m, lapack_int n,
                                 const float* a, lapack_int lda);
HWYBLAS_API double LAPACKE_dlantr(int matrix_layout, char norm, char uplo,
                                  char diag, lapack_int m, lapack_int n,
                                  const double* a, lapack_int lda);
HWYBLAS_API float LAPACKE_clantr(int matrix_layout, char norm, char uplo,
                                 char diag, lapack_int m, lapack_int n,
                                 const lapack_complex_float* a, lapack_int lda);
HWYBLAS_API double LAPACKE_zlantr(int matrix_layout, char norm, char uplo,
                                  char diag, lapack_int m, lapack_int n,
                                  const lapack_complex_double* a,
                                  lapack_int lda);
HWYBLAS_API float LAPACKE_slantr_work(int matrix_layout, char norm, char uplo,
                                      char diag, lapack_int m, lapack_int n,
                                      const float* a, lapack_int lda,
                                      float* work);
HWYBLAS_API double LAPACKE_dlantr_work(int matrix_layout, char norm, char uplo,
                                       char diag, lapack_int m, lapack_int n,
                                       const double* a, lapack_int lda,
                                       double* work);
HWYBLAS_API float LAPACKE_clantr_work(int matrix_layout, char norm, char uplo,
                                      char diag, lapack_int m, lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda, float* work);
HWYBLAS_API double LAPACKE_zlantr_work(int matrix_layout, char norm, char uplo,
                                       char diag, lapack_int m, lapack_int n,
                                       const lapack_complex_double* a,
                                       lapack_int lda, double* work);

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
/* Symmetric (Hermitian) eigenproblems by divide and conquer (see lapack.h) */

HWYBLAS_API lapack_int LAPACKE_ssyevd(int matrix_layout, char jobz, char uplo,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* w);
HWYBLAS_API lapack_int LAPACKE_dsyevd(int matrix_layout, char jobz, char uplo,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* w);
HWYBLAS_API lapack_int LAPACKE_ssyevd_work(int matrix_layout, char jobz,
                                           char uplo, lapack_int n, float* a,
                                           lapack_int lda, float* w,
                                           float* work, lapack_int lwork,
                                           lapack_int* iwork,
                                           lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dsyevd_work(int matrix_layout, char jobz,
                                           char uplo, lapack_int n, double* a,
                                           lapack_int lda, double* w,
                                           double* work, lapack_int lwork,
                                           lapack_int* iwork,
                                           lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_cheevd(int matrix_layout, char jobz, char uplo,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, float* w);
HWYBLAS_API lapack_int LAPACKE_zheevd(int matrix_layout, char jobz, char uplo,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, double* w);
HWYBLAS_API lapack_int
LAPACKE_cheevd_work(int matrix_layout, char jobz, char uplo, lapack_int n,
                    lapack_complex_float* a, lapack_int lda, float* w,
                    lapack_complex_float* work, lapack_int lwork, float* rwork,
                    lapack_int lrwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_zheevd_work(
    int matrix_layout, char jobz, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double* w,
    lapack_complex_double* work, lapack_int lwork, double* rwork,
    lapack_int lrwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_sstevd(int matrix_layout, char jobz,
                                      lapack_int n, float* d, float* e,
                                      float* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_dstevd(int matrix_layout, char jobz,
                                      lapack_int n, double* d, double* e,
                                      double* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_sstevd_work(int matrix_layout, char jobz,
                                           lapack_int n, float* d, float* e,
                                           float* z, lapack_int ldz,
                                           float* work, lapack_int lwork,
                                           lapack_int* iwork,
                                           lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dstevd_work(int matrix_layout, char jobz,
                                           lapack_int n, double* d, double* e,
                                           double* z, lapack_int ldz,
                                           double* work, lapack_int lwork,
                                           lapack_int* iwork,
                                           lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_sstedc(int matrix_layout, char compz,
                                      lapack_int n, float* d, float* e,
                                      float* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_dstedc(int matrix_layout, char compz,
                                      lapack_int n, double* d, double* e,
                                      double* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_cstedc(int matrix_layout, char compz,
                                      lapack_int n, float* d, float* e,
                                      lapack_complex_float* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_zstedc(int matrix_layout, char compz,
                                      lapack_int n, double* d, double* e,
                                      lapack_complex_double* z, lapack_int ldz);
HWYBLAS_API lapack_int LAPACKE_sstedc_work(int matrix_layout, char compz,
                                           lapack_int n, float* d, float* e,
                                           float* z, lapack_int ldz,
                                           float* work, lapack_int lwork,
                                           lapack_int* iwork,
                                           lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dstedc_work(int matrix_layout, char compz,
                                           lapack_int n, double* d, double* e,
                                           double* z, lapack_int ldz,
                                           double* work, lapack_int lwork,
                                           lapack_int* iwork,
                                           lapack_int liwork);
HWYBLAS_API lapack_int
LAPACKE_cstedc_work(int matrix_layout, char compz, lapack_int n, float* d,
                    float* e, lapack_complex_float* z, lapack_int ldz,
                    lapack_complex_float* work, lapack_int lwork, float* rwork,
                    lapack_int lrwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_zstedc_work(
    int matrix_layout, char compz, lapack_int n, double* d, double* e,
    lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work,
    lapack_int lwork, double* rwork, lapack_int lrwork, lapack_int* iwork,
    lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_sormtr(int matrix_layout, char side, char uplo,
                                      char trans, lapack_int m, lapack_int n,
                                      const float* a, lapack_int lda,
                                      const float* tau, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormtr(int matrix_layout, char side, char uplo,
                                      char trans, lapack_int m, lapack_int n,
                                      const double* a, lapack_int lda,
                                      const double* tau, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormtr_work(int matrix_layout, char side,
                                           char uplo, char trans, lapack_int m,
                                           lapack_int n, const float* a,
                                           lapack_int lda, const float* tau,
                                           float* c, lapack_int ldc,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormtr_work(int matrix_layout, char side,
                                           char uplo, char trans, lapack_int m,
                                           lapack_int n, const double* a,
                                           lapack_int lda, const double* tau,
                                           double* c, lapack_int ldc,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmtr(int matrix_layout, char side, char uplo,
                                      char trans, lapack_int m, lapack_int n,
                                      const lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmtr(int matrix_layout, char side, char uplo,
                                      char trans, lapack_int m, lapack_int n,
                                      const lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmtr_work(
    int matrix_layout, char side, char uplo, char trans, lapack_int m,
    lapack_int n, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmtr_work(
    int matrix_layout, char side, char uplo, char trans, lapack_int m,
    lapack_int n, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sormql(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const float* a, lapack_int lda,
                                      const float* tau, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormql(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const double* a, lapack_int lda,
                                      const double* tau, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormql_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const float* a, lapack_int lda, const float* tau, float* c,
    lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormql_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const double* a, lapack_int lda, const double* tau, double* c,
    lapack_int ldc, double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmql(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmql(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmql_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmql_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_ssygst(int matrix_layout, lapack_int itype,
                                      char uplo, lapack_int n, float* a,
                                      lapack_int lda, const float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dsygst(int matrix_layout, lapack_int itype,
                                      char uplo, lapack_int n, double* a,
                                      lapack_int lda, const double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ssygst_work(int matrix_layout, lapack_int itype,
                                           char uplo, lapack_int n, float* a,
                                           lapack_int lda, const float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dsygst_work(int matrix_layout, lapack_int itype,
                                           char uplo, lapack_int n, double* a,
                                           lapack_int lda, const double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_chegst(int matrix_layout, lapack_int itype,
                                      char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zhegst(int matrix_layout, lapack_int itype,
                                      char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_chegst_work(int matrix_layout, lapack_int itype,
                                           char uplo, lapack_int n,
                                           lapack_complex_float* a,
                                           lapack_int lda,
                                           const lapack_complex_float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zhegst_work(int matrix_layout, lapack_int itype,
                                           char uplo, lapack_int n,
                                           lapack_complex_double* a,
                                           lapack_int lda,
                                           const lapack_complex_double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_ssygv(int matrix_layout, lapack_int itype,
                                     char jobz, char uplo, lapack_int n,
                                     float* a, lapack_int lda, float* b,
                                     lapack_int ldb, float* w);
HWYBLAS_API lapack_int LAPACKE_dsygv(int matrix_layout, lapack_int itype,
                                     char jobz, char uplo, lapack_int n,
                                     double* a, lapack_int lda, double* b,
                                     lapack_int ldb, double* w);
HWYBLAS_API lapack_int LAPACKE_ssygv_work(int matrix_layout, lapack_int itype,
                                          char jobz, char uplo, lapack_int n,
                                          float* a, lapack_int lda, float* b,
                                          lapack_int ldb, float* w, float* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dsygv_work(int matrix_layout, lapack_int itype,
                                          char jobz, char uplo, lapack_int n,
                                          double* a, lapack_int lda, double* b,
                                          lapack_int ldb, double* w,
                                          double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_chegv(int matrix_layout, lapack_int itype,
                                     char jobz, char uplo, lapack_int n,
                                     lapack_complex_float* a, lapack_int lda,
                                     lapack_complex_float* b, lapack_int ldb,
                                     float* w);
HWYBLAS_API lapack_int LAPACKE_zhegv(int matrix_layout, lapack_int itype,
                                     char jobz, char uplo, lapack_int n,
                                     lapack_complex_double* a, lapack_int lda,
                                     lapack_complex_double* b, lapack_int ldb,
                                     double* w);
HWYBLAS_API lapack_int
LAPACKE_chegv_work(int matrix_layout, lapack_int itype, char jobz, char uplo,
                   lapack_int n, lapack_complex_float* a, lapack_int lda,
                   lapack_complex_float* b, lapack_int ldb, float* w,
                   lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zhegv_work(
    int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, double* w, lapack_complex_double* work, lapack_int lwork,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_ssygvd(int matrix_layout, lapack_int itype,
                                      char jobz, char uplo, lapack_int n,
                                      float* a, lapack_int lda, float* b,
                                      lapack_int ldb, float* w);
HWYBLAS_API lapack_int LAPACKE_dsygvd(int matrix_layout, lapack_int itype,
                                      char jobz, char uplo, lapack_int n,
                                      double* a, lapack_int lda, double* b,
                                      lapack_int ldb, double* w);
HWYBLAS_API lapack_int LAPACKE_ssygvd_work(
    int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n,
    float* a, lapack_int lda, float* b, lapack_int ldb, float* w, float* work,
    lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dsygvd_work(
    int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n,
    double* a, lapack_int lda, double* b, lapack_int ldb, double* w,
    double* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_chegvd(int matrix_layout, lapack_int itype,
                                      char jobz, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb,
                                      float* w);
HWYBLAS_API lapack_int LAPACKE_zhegvd(int matrix_layout, lapack_int itype,
                                      char jobz, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb,
                                      double* w);
HWYBLAS_API lapack_int LAPACKE_chegvd_work(
    int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* b,
    lapack_int ldb, float* w, lapack_complex_float* work, lapack_int lwork,
    float* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_zhegvd_work(
    int matrix_layout, lapack_int itype, char jobz, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, double* w, lapack_complex_double* work, lapack_int lwork,
    double* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork);

/* ------------------------------------------------------------------------ */
/* Symmetric (Hermitian) eigenproblems by MRRR, bisection and inverse iteration
 * (see lapack.h) */

HWYBLAS_API lapack_int LAPACKE_ssyevr(int matrix_layout, char jobz, char range,
                                      char uplo, lapack_int n, float* a,
                                      lapack_int lda, float vl, float vu,
                                      lapack_int il, lapack_int iu,
                                      float abstol, lapack_int* m, float* w,
                                      float* z, lapack_int ldz,
                                      lapack_int* isuppz);
HWYBLAS_API lapack_int LAPACKE_dsyevr(int matrix_layout, char jobz, char range,
                                      char uplo, lapack_int n, double* a,
                                      lapack_int lda, double vl, double vu,
                                      lapack_int il, lapack_int iu,
                                      double abstol, lapack_int* m, double* w,
                                      double* z, lapack_int ldz,
                                      lapack_int* isuppz);
HWYBLAS_API lapack_int LAPACKE_ssyevr_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n, float* a,
    lapack_int lda, float vl, float vu, lapack_int il, lapack_int iu,
    float abstol, lapack_int* m, float* w, float* z, lapack_int ldz,
    lapack_int* isuppz, float* work, lapack_int lwork, lapack_int* iwork,
    lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dsyevr_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    double* a, lapack_int lda, double vl, double vu, lapack_int il,
    lapack_int iu, double abstol, lapack_int* m, double* w, double* z,
    lapack_int ldz, lapack_int* isuppz, double* work, lapack_int lwork,
    lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_cheevr(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_float* a, lapack_int lda, float vl, float vu, lapack_int il,
    lapack_int iu, float abstol, lapack_int* m, float* w,
    lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz);
HWYBLAS_API lapack_int LAPACKE_zheevr(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double vl, double vu,
    lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* w,
    lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz);
HWYBLAS_API lapack_int LAPACKE_cheevr_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_float* a, lapack_int lda, float vl, float vu, lapack_int il,
    lapack_int iu, float abstol, lapack_int* m, float* w,
    lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz,
    lapack_complex_float* work, lapack_int lwork, float* rwork,
    lapack_int lrwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_zheevr_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double vl, double vu,
    lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* w,
    lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz,
    lapack_complex_double* work, lapack_int lwork, double* rwork,
    lapack_int lrwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_sstevr(int matrix_layout, char jobz, char range,
                                      lapack_int n, float* d, float* e,
                                      float vl, float vu, lapack_int il,
                                      lapack_int iu, float abstol,
                                      lapack_int* m, float* w, float* z,
                                      lapack_int ldz, lapack_int* isuppz);
HWYBLAS_API lapack_int LAPACKE_dstevr(int matrix_layout, char jobz, char range,
                                      lapack_int n, double* d, double* e,
                                      double vl, double vu, lapack_int il,
                                      lapack_int iu, double abstol,
                                      lapack_int* m, double* w, double* z,
                                      lapack_int ldz, lapack_int* isuppz);
HWYBLAS_API lapack_int LAPACKE_sstevr_work(
    int matrix_layout, char jobz, char range, lapack_int n, float* d, float* e,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* m, float* w, float* z, lapack_int ldz, lapack_int* isuppz,
    float* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int
LAPACKE_dstevr_work(int matrix_layout, char jobz, char range, lapack_int n,
                    double* d, double* e, double vl, double vu, lapack_int il,
                    lapack_int iu, double abstol, lapack_int* m, double* w,
                    double* z, lapack_int ldz, lapack_int* isuppz, double* work,
                    lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_ssyevx(int matrix_layout, char jobz, char range,
                                      char uplo, lapack_int n, float* a,
                                      lapack_int lda, float vl, float vu,
                                      lapack_int il, lapack_int iu,
                                      float abstol, lapack_int* m, float* w,
                                      float* z, lapack_int ldz,
                                      lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_dsyevx(int matrix_layout, char jobz, char range,
                                      char uplo, lapack_int n, double* a,
                                      lapack_int lda, double vl, double vu,
                                      lapack_int il, lapack_int iu,
                                      double abstol, lapack_int* m, double* w,
                                      double* z, lapack_int ldz,
                                      lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_ssyevx_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n, float* a,
    lapack_int lda, float vl, float vu, lapack_int il, lapack_int iu,
    float abstol, lapack_int* m, float* w, float* z, lapack_int ldz,
    float* work, lapack_int lwork, lapack_int* iwork, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_dsyevx_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    double* a, lapack_int lda, double vl, double vu, lapack_int il,
    lapack_int iu, double abstol, lapack_int* m, double* w, double* z,
    lapack_int ldz, double* work, lapack_int lwork, lapack_int* iwork,
    lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_cheevx(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_float* a, lapack_int lda, float vl, float vu, lapack_int il,
    lapack_int iu, float abstol, lapack_int* m, float* w,
    lapack_complex_float* z, lapack_int ldz, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_zheevx(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double vl, double vu,
    lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* w,
    lapack_complex_double* z, lapack_int ldz, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_cheevx_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_float* a, lapack_int lda, float vl, float vu, lapack_int il,
    lapack_int iu, float abstol, lapack_int* m, float* w,
    lapack_complex_float* z, lapack_int ldz, lapack_complex_float* work,
    lapack_int lwork, float* rwork, lapack_int* iwork, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_zheevx_work(
    int matrix_layout, char jobz, char range, char uplo, lapack_int n,
    lapack_complex_double* a, lapack_int lda, double vl, double vu,
    lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* w,
    lapack_complex_double* z, lapack_int ldz, lapack_complex_double* work,
    lapack_int lwork, double* rwork, lapack_int* iwork, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_sstevx(int matrix_layout, char jobz, char range,
                                      lapack_int n, float* d, float* e,
                                      float vl, float vu, lapack_int il,
                                      lapack_int iu, float abstol,
                                      lapack_int* m, float* w, float* z,
                                      lapack_int ldz, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_dstevx(int matrix_layout, char jobz, char range,
                                      lapack_int n, double* d, double* e,
                                      double vl, double vu, lapack_int il,
                                      lapack_int iu, double abstol,
                                      lapack_int* m, double* w, double* z,
                                      lapack_int ldz, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_sstevx_work(
    int matrix_layout, char jobz, char range, lapack_int n, float* d, float* e,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* m, float* w, float* z, lapack_int ldz, float* work,
    lapack_int* iwork, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_dstevx_work(
    int matrix_layout, char jobz, char range, lapack_int n, double* d,
    double* e, double vl, double vu, lapack_int il, lapack_int iu,
    double abstol, lapack_int* m, double* w, double* z, lapack_int ldz,
    double* work, lapack_int* iwork, lapack_int* ifail);
HWYBLAS_API lapack_int LAPACKE_sstemr(int matrix_layout, char jobz, char range,
                                      lapack_int n, float* d, float* e,
                                      float vl, float vu, lapack_int il,
                                      lapack_int iu, lapack_int* m, float* w,
                                      float* z, lapack_int ldz, lapack_int nzc,
                                      lapack_int* isuppz,
                                      lapack_logical* tryrac);
HWYBLAS_API lapack_int LAPACKE_dstemr(int matrix_layout, char jobz, char range,
                                      lapack_int n, double* d, double* e,
                                      double vl, double vu, lapack_int il,
                                      lapack_int iu, lapack_int* m, double* w,
                                      double* z, lapack_int ldz, lapack_int nzc,
                                      lapack_int* isuppz,
                                      lapack_logical* tryrac);
HWYBLAS_API lapack_int LAPACKE_cstemr(int matrix_layout, char jobz, char range,
                                      lapack_int n, float* d, float* e,
                                      float vl, float vu, lapack_int il,
                                      lapack_int iu, lapack_int* m, float* w,
                                      lapack_complex_float* z, lapack_int ldz,
                                      lapack_int nzc, lapack_int* isuppz,
                                      lapack_logical* tryrac);
HWYBLAS_API lapack_int LAPACKE_zstemr(int matrix_layout, char jobz, char range,
                                      lapack_int n, double* d, double* e,
                                      double vl, double vu, lapack_int il,
                                      lapack_int iu, lapack_int* m, double* w,
                                      lapack_complex_double* z, lapack_int ldz,
                                      lapack_int nzc, lapack_int* isuppz,
                                      lapack_logical* tryrac);
HWYBLAS_API lapack_int LAPACKE_sstemr_work(
    int matrix_layout, char jobz, char range, lapack_int n, float* d, float* e,
    float vl, float vu, lapack_int il, lapack_int iu, lapack_int* m, float* w,
    float* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz,
    lapack_logical* tryrac, float* work, lapack_int lwork, lapack_int* iwork,
    lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_dstemr_work(
    int matrix_layout, char jobz, char range, lapack_int n, double* d,
    double* e, double vl, double vu, lapack_int il, lapack_int iu,
    lapack_int* m, double* w, double* z, lapack_int ldz, lapack_int nzc,
    lapack_int* isuppz, lapack_logical* tryrac, double* work, lapack_int lwork,
    lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_cstemr_work(
    int matrix_layout, char jobz, char range, lapack_int n, float* d, float* e,
    float vl, float vu, lapack_int il, lapack_int iu, lapack_int* m, float* w,
    lapack_complex_float* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz,
    lapack_logical* tryrac, float* work, lapack_int lwork, lapack_int* iwork,
    lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_zstemr_work(
    int matrix_layout, char jobz, char range, lapack_int n, double* d,
    double* e, double vl, double vu, lapack_int il, lapack_int iu,
    lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz,
    lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac, double* work,
    lapack_int lwork, lapack_int* iwork, lapack_int liwork);
HWYBLAS_API lapack_int LAPACKE_sstebz(
    char range, char order, lapack_int n, float vl, float vu, lapack_int il,
    lapack_int iu, float abstol, const float* d, const float* e, lapack_int* m,
    lapack_int* nsplit, float* w, lapack_int* iblock, lapack_int* isplit);
HWYBLAS_API lapack_int LAPACKE_dstebz(char range, char order, lapack_int n,
                                      double vl, double vu, lapack_int il,
                                      lapack_int iu, double abstol,
                                      const double* d, const double* e,
                                      lapack_int* m, lapack_int* nsplit,
                                      double* w, lapack_int* iblock,
                                      lapack_int* isplit);
HWYBLAS_API lapack_int LAPACKE_sstebz_work(
    char range, char order, lapack_int n, float vl, float vu, lapack_int il,
    lapack_int iu, float abstol, const float* d, const float* e, lapack_int* m,
    lapack_int* nsplit, float* w, lapack_int* iblock, lapack_int* isplit,
    float* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dstebz_work(
    char range, char order, lapack_int n, double vl, double vu, lapack_int il,
    lapack_int iu, double abstol, const double* d, const double* e,
    lapack_int* m, lapack_int* nsplit, double* w, lapack_int* iblock,
    lapack_int* isplit, double* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_sstein(int matrix_layout, lapack_int n,
                                      const float* d, const float* e,
                                      lapack_int m, const float* w,
                                      const lapack_int* iblock,
                                      const lapack_int* isplit, float* z,
                                      lapack_int ldz, lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_dstein(int matrix_layout, lapack_int n,
                                      const double* d, const double* e,
                                      lapack_int m, const double* w,
                                      const lapack_int* iblock,
                                      const lapack_int* isplit, double* z,
                                      lapack_int ldz, lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_cstein(int matrix_layout, lapack_int n,
                                      const float* d, const float* e,
                                      lapack_int m, const float* w,
                                      const lapack_int* iblock,
                                      const lapack_int* isplit,
                                      lapack_complex_float* z, lapack_int ldz,
                                      lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_zstein(int matrix_layout, lapack_int n,
                                      const double* d, const double* e,
                                      lapack_int m, const double* w,
                                      const lapack_int* iblock,
                                      const lapack_int* isplit,
                                      lapack_complex_double* z, lapack_int ldz,
                                      lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_sstein_work(
    int matrix_layout, lapack_int n, const float* d, const float* e,
    lapack_int m, const float* w, const lapack_int* iblock,
    const lapack_int* isplit, float* z, lapack_int ldz, float* work,
    lapack_int* iwork, lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_dstein_work(
    int matrix_layout, lapack_int n, const double* d, const double* e,
    lapack_int m, const double* w, const lapack_int* iblock,
    const lapack_int* isplit, double* z, lapack_int ldz, double* work,
    lapack_int* iwork, lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_cstein_work(
    int matrix_layout, lapack_int n, const float* d, const float* e,
    lapack_int m, const float* w, const lapack_int* iblock,
    const lapack_int* isplit, lapack_complex_float* z, lapack_int ldz,
    float* work, lapack_int* iwork, lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_zstein_work(
    int matrix_layout, lapack_int n, const double* d, const double* e,
    lapack_int m, const double* w, const lapack_int* iblock,
    const lapack_int* isplit, lapack_complex_double* z, lapack_int ldz,
    double* work, lapack_int* iwork, lapack_int* ifailv);
HWYBLAS_API lapack_int LAPACKE_slarnv(lapack_int idist, lapack_int* iseed,
                                      lapack_int n, float* x);
HWYBLAS_API lapack_int LAPACKE_dlarnv(lapack_int idist, lapack_int* iseed,
                                      lapack_int n, double* x);
HWYBLAS_API lapack_int LAPACKE_clarnv(lapack_int idist, lapack_int* iseed,
                                      lapack_int n, lapack_complex_float* x);
HWYBLAS_API lapack_int LAPACKE_zlarnv(lapack_int idist, lapack_int* iseed,
                                      lapack_int n, lapack_complex_double* x);
HWYBLAS_API lapack_int LAPACKE_slarnv_work(lapack_int idist, lapack_int* iseed,
                                           lapack_int n, float* x);
HWYBLAS_API lapack_int LAPACKE_dlarnv_work(lapack_int idist, lapack_int* iseed,
                                           lapack_int n, double* x);
HWYBLAS_API lapack_int LAPACKE_clarnv_work(lapack_int idist, lapack_int* iseed,
                                           lapack_int n,
                                           lapack_complex_float* x);
HWYBLAS_API lapack_int LAPACKE_zlarnv_work(lapack_int idist, lapack_int* iseed,
                                           lapack_int n,
                                           lapack_complex_double* x);

/* ------------------------------------------------------------------------ */
/* Band and tridiagonal systems (see lapack.h). In row-major layout, a band
 * matrix is the transpose of the band storage: ab[i*ldab + j], ldab >= n.
 * As in the reference LAPACKE, ?gbsv's NaN check includes the first kl
 * rows of ab, which ?gbsv uses as workspace. ?gttrf and ?pttrf take no
 * matrix_layout. */
HWYBLAS_API lapack_int LAPACKE_sgbtrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, float* ab, lapack_int ldab,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgbtrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int kl,
                                           lapack_int ku, float* ab,
                                           lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgbtrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, double* ab,
                                      lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgbtrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int kl,
                                           lapack_int ku, double* ab,
                                           lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgbtrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, lapack_complex_float* ab,
                                      lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgbtrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int kl,
                                           lapack_int ku,
                                           lapack_complex_float* ab,
                                           lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgbtrf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, lapack_complex_double* ab,
                                      lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgbtrf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int kl,
                                           lapack_int ku,
                                           lapack_complex_double* ab,
                                           lapack_int ldab, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgbtrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, lapack_int nrhs,
                                      const float* ab, lapack_int ldab,
                                      const lapack_int* ipiv, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgbtrs_work(int matrix_layout, char trans,
                                           lapack_int n, lapack_int kl,
                                           lapack_int ku, lapack_int nrhs,
                                           const float* ab, lapack_int ldab,
                                           const lapack_int* ipiv, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgbtrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, lapack_int nrhs,
                                      const double* ab, lapack_int ldab,
                                      const lapack_int* ipiv, double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgbtrs_work(int matrix_layout, char trans,
                                           lapack_int n, lapack_int kl,
                                           lapack_int ku, lapack_int nrhs,
                                           const double* ab, lapack_int ldab,
                                           const lapack_int* ipiv, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgbtrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, lapack_int nrhs,
                                      const lapack_complex_float* ab,
                                      lapack_int ldab, const lapack_int* ipiv,
                                      lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgbtrs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int kl, lapack_int ku,
    lapack_int nrhs, const lapack_complex_float* ab, lapack_int ldab,
    const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgbtrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, lapack_int nrhs,
                                      const lapack_complex_double* ab,
                                      lapack_int ldab, const lapack_int* ipiv,
                                      lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgbtrs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int kl, lapack_int ku,
    lapack_int nrhs, const lapack_complex_double* ab, lapack_int ldab,
    const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgbsv(int matrix_layout, lapack_int n,
                                     lapack_int kl, lapack_int ku,
                                     lapack_int nrhs, float* ab,
                                     lapack_int ldab, lapack_int* ipiv,
                                     float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgbsv_work(int matrix_layout, lapack_int n,
                                          lapack_int kl, lapack_int ku,
                                          lapack_int nrhs, float* ab,
                                          lapack_int ldab, lapack_int* ipiv,
                                          float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgbsv(int matrix_layout, lapack_int n,
                                     lapack_int kl, lapack_int ku,
                                     lapack_int nrhs, double* ab,
                                     lapack_int ldab, lapack_int* ipiv,
                                     double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgbsv_work(int matrix_layout, lapack_int n,
                                          lapack_int kl, lapack_int ku,
                                          lapack_int nrhs, double* ab,
                                          lapack_int ldab, lapack_int* ipiv,
                                          double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgbsv(int matrix_layout, lapack_int n,
                                     lapack_int kl, lapack_int ku,
                                     lapack_int nrhs, lapack_complex_float* ab,
                                     lapack_int ldab, lapack_int* ipiv,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgbsv_work(
    int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku,
    lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab,
    lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgbsv(int matrix_layout, lapack_int n,
                                     lapack_int kl, lapack_int ku,
                                     lapack_int nrhs, lapack_complex_double* ab,
                                     lapack_int ldab, lapack_int* ipiv,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgbsv_work(
    int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku,
    lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab,
    lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgbcon(int matrix_layout, char norm,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, const float* ab,
                                      lapack_int ldab, const lapack_int* ipiv,
                                      float anorm, float* rcond);
HWYBLAS_API lapack_int LAPACKE_sgbcon_work(
    int matrix_layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,
    const float* ab, lapack_int ldab, const lapack_int* ipiv, float anorm,
    float* rcond, float* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgbcon(int matrix_layout, char norm,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku, const double* ab,
                                      lapack_int ldab, const lapack_int* ipiv,
                                      double anorm, double* rcond);
HWYBLAS_API lapack_int LAPACKE_dgbcon_work(
    int matrix_layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,
    const double* ab, lapack_int ldab, const lapack_int* ipiv, double anorm,
    double* rcond, double* work, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cgbcon(int matrix_layout, char norm,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku,
                                      const lapack_complex_float* ab,
                                      lapack_int ldab, const lapack_int* ipiv,
                                      float anorm, float* rcond);
HWYBLAS_API lapack_int LAPACKE_cgbcon_work(
    int matrix_layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,
    const lapack_complex_float* ab, lapack_int ldab, const lapack_int* ipiv,
    float anorm, float* rcond, lapack_complex_float* work, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgbcon(int matrix_layout, char norm,
                                      lapack_int n, lapack_int kl,
                                      lapack_int ku,
                                      const lapack_complex_double* ab,
                                      lapack_int ldab, const lapack_int* ipiv,
                                      double anorm, double* rcond);
HWYBLAS_API lapack_int LAPACKE_zgbcon_work(
    int matrix_layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,
    const lapack_complex_double* ab, lapack_int ldab, const lapack_int* ipiv,
    double anorm, double* rcond, lapack_complex_double* work, double* rwork);
HWYBLAS_API lapack_int LAPACKE_spbtrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd, float* ab,
                                      lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_spbtrf_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int kd,
                                           float* ab, lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_dpbtrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd, double* ab,
                                      lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_dpbtrf_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int kd,
                                           double* ab, lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_cpbtrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd,
                                      lapack_complex_float* ab,
                                      lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_cpbtrf_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int kd,
                                           lapack_complex_float* ab,
                                           lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_zpbtrf(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd,
                                      lapack_complex_double* ab,
                                      lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_zpbtrf_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int kd,
                                           lapack_complex_double* ab,
                                           lapack_int ldab);
HWYBLAS_API lapack_int LAPACKE_spbtrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd,
                                      lapack_int nrhs, const float* ab,
                                      lapack_int ldab, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_spbtrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int kd,
                                           lapack_int nrhs, const float* ab,
                                           lapack_int ldab, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpbtrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd,
                                      lapack_int nrhs, const double* ab,
                                      lapack_int ldab, double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpbtrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int kd,
                                           lapack_int nrhs, const double* ab,
                                           lapack_int ldab, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cpbtrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd,
                                      lapack_int nrhs,
                                      const lapack_complex_float* ab,
                                      lapack_int ldab, lapack_complex_float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int
LAPACKE_cpbtrs_work(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                    lapack_int nrhs, const lapack_complex_float* ab,
                    lapack_int ldab, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zpbtrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int kd,
                                      lapack_int nrhs,
                                      const lapack_complex_double* ab,
                                      lapack_int ldab, lapack_complex_double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int
LAPACKE_zpbtrs_work(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                    lapack_int nrhs, const lapack_complex_double* ab,
                    lapack_int ldab, lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_spbsv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int kd, lapack_int nrhs, float* ab,
                                     lapack_int ldab, float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_spbsv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int kd,
                                          lapack_int nrhs, float* ab,
                                          lapack_int ldab, float* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpbsv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int kd, lapack_int nrhs, double* ab,
                                     lapack_int ldab, double* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpbsv_work(int matrix_layout, char uplo,
                                          lapack_int n, lapack_int kd,
                                          lapack_int nrhs, double* ab,
                                          lapack_int ldab, double* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cpbsv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int kd, lapack_int nrhs,
                                     lapack_complex_float* ab, lapack_int ldab,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int
LAPACKE_cpbsv_work(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                   lapack_int nrhs, lapack_complex_float* ab, lapack_int ldab,
                   lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zpbsv(int matrix_layout, char uplo, lapack_int n,
                                     lapack_int kd, lapack_int nrhs,
                                     lapack_complex_double* ab, lapack_int ldab,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int
LAPACKE_zpbsv_work(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                   lapack_int nrhs, lapack_complex_double* ab, lapack_int ldab,
                   lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgttrf(lapack_int n, float* dl, float* d,
                                      float* du, float* du2, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgttrf_work(lapack_int n, float* dl, float* d,
                                           float* du, float* du2,
                                           lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgttrf(lapack_int n, double* dl, double* d,
                                      double* du, double* du2,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_dgttrf_work(lapack_int n, double* dl, double* d,
                                           double* du, double* du2,
                                           lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgttrf(lapack_int n, lapack_complex_float* dl,
                                      lapack_complex_float* d,
                                      lapack_complex_float* du,
                                      lapack_complex_float* du2,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_cgttrf_work(
    lapack_int n, lapack_complex_float* dl, lapack_complex_float* d,
    lapack_complex_float* du, lapack_complex_float* du2, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgttrf(lapack_int n, lapack_complex_double* dl,
                                      lapack_complex_double* d,
                                      lapack_complex_double* du,
                                      lapack_complex_double* du2,
                                      lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_zgttrf_work(
    lapack_int n, lapack_complex_double* dl, lapack_complex_double* d,
    lapack_complex_double* du, lapack_complex_double* du2, lapack_int* ipiv);
HWYBLAS_API lapack_int LAPACKE_sgttrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const float* dl, const float* d,
                                      const float* du, const float* du2,
                                      const lapack_int* ipiv, float* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgttrs_work(int matrix_layout, char trans,
                                           lapack_int n, lapack_int nrhs,
                                           const float* dl, const float* d,
                                           const float* du, const float* du2,
                                           const lapack_int* ipiv, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgttrs(int matrix_layout, char trans,
                                      lapack_int n, lapack_int nrhs,
                                      const double* dl, const double* d,
                                      const double* du, const double* du2,
                                      const lapack_int* ipiv, double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgttrs_work(int matrix_layout, char trans,
                                           lapack_int n, lapack_int nrhs,
                                           const double* dl, const double* d,
                                           const double* du, const double* du2,
                                           const lapack_int* ipiv, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int
LAPACKE_cgttrs(int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
               const lapack_complex_float* dl, const lapack_complex_float* d,
               const lapack_complex_float* du, const lapack_complex_float* du2,
               const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgttrs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_float* dl, const lapack_complex_float* d,
    const lapack_complex_float* du, const lapack_complex_float* du2,
    const lapack_int* ipiv, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgttrs(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* dl, const lapack_complex_double* d,
    const lapack_complex_double* du, const lapack_complex_double* du2,
    const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgttrs_work(
    int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
    const lapack_complex_double* dl, const lapack_complex_double* d,
    const lapack_complex_double* du, const lapack_complex_double* du2,
    const lapack_int* ipiv, lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgtsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, float* dl, float* d,
                                     float* du, float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgtsv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, float* dl, float* d,
                                          float* du, float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgtsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, double* dl, double* d,
                                     double* du, double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgtsv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, double* dl,
                                          double* d, double* du, double* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgtsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float* dl,
                                     lapack_complex_float* d,
                                     lapack_complex_float* du,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgtsv_work(
    int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_float* dl,
    lapack_complex_float* d, lapack_complex_float* du, lapack_complex_float* b,
    lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgtsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double* dl,
                                     lapack_complex_double* d,
                                     lapack_complex_double* du,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgtsv_work(
    int matrix_layout, lapack_int n, lapack_int nrhs, lapack_complex_double* dl,
    lapack_complex_double* d, lapack_complex_double* du,
    lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_spttrf(lapack_int n, float* d, float* e);
HWYBLAS_API lapack_int LAPACKE_spttrf_work(lapack_int n, float* d, float* e);
HWYBLAS_API lapack_int LAPACKE_dpttrf(lapack_int n, double* d, double* e);
HWYBLAS_API lapack_int LAPACKE_dpttrf_work(lapack_int n, double* d, double* e);
HWYBLAS_API lapack_int LAPACKE_cpttrf(lapack_int n, float* d,
                                      lapack_complex_float* e);
HWYBLAS_API lapack_int LAPACKE_cpttrf_work(lapack_int n, float* d,
                                           lapack_complex_float* e);
HWYBLAS_API lapack_int LAPACKE_zpttrf(lapack_int n, double* d,
                                      lapack_complex_double* e);
HWYBLAS_API lapack_int LAPACKE_zpttrf_work(lapack_int n, double* d,
                                           lapack_complex_double* e);
HWYBLAS_API lapack_int LAPACKE_spttrs(int matrix_layout, lapack_int n,
                                      lapack_int nrhs, const float* d,
                                      const float* e, float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_spttrs_work(int matrix_layout, lapack_int n,
                                           lapack_int nrhs, const float* d,
                                           const float* e, float* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpttrs(int matrix_layout, lapack_int n,
                                      lapack_int nrhs, const double* d,
                                      const double* e, double* b,
                                      lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dpttrs_work(int matrix_layout, lapack_int n,
                                           lapack_int nrhs, const double* d,
                                           const double* e, double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cpttrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const float* d,
                                      const lapack_complex_float* e,
                                      lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cpttrs_work(
    int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, const float* d,
    const lapack_complex_float* e, lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zpttrs(int matrix_layout, char uplo,
                                      lapack_int n, lapack_int nrhs,
                                      const double* d,
                                      const lapack_complex_double* e,
                                      lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zpttrs_work(int matrix_layout, char uplo,
                                           lapack_int n, lapack_int nrhs,
                                           const double* d,
                                           const lapack_complex_double* e,
                                           lapack_complex_double* b,
                                           lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sptsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, float* d, float* e,
                                     float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sptsv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, float* d, float* e,
                                          float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dptsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, double* d, double* e,
                                     double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dptsv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, double* d, double* e,
                                          double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cptsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, float* d,
                                     lapack_complex_float* e,
                                     lapack_complex_float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cptsv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, float* d,
                                          lapack_complex_float* e,
                                          lapack_complex_float* b,
                                          lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zptsv(int matrix_layout, lapack_int n,
                                     lapack_int nrhs, double* d,
                                     lapack_complex_double* e,
                                     lapack_complex_double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zptsv_work(int matrix_layout, lapack_int n,
                                          lapack_int nrhs, double* d,
                                          lapack_complex_double* e,
                                          lapack_complex_double* b,
                                          lapack_int ldb);

/* ------------------------------------------------------------------------ */
/* Least squares (see lapack.h). ?ormrq / ?unmrq and ?ormrz / ?unmrz take A
 * as k x n for side = 'R' (the reference LAPACKE 3.12.1: k x m). */
HWYBLAS_API lapack_int LAPACKE_sgels(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, float* a, lapack_int lda,
                                     float* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_sgels_work(int matrix_layout, char trans,
                                          lapack_int m, lapack_int n,
                                          lapack_int nrhs, float* a,
                                          lapack_int lda, float* b,
                                          lapack_int ldb, float* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgels(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, double* a, lapack_int lda,
                                     double* b, lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_dgels_work(int matrix_layout, char trans,
                                          lapack_int m, lapack_int n,
                                          lapack_int nrhs, double* a,
                                          lapack_int lda, double* b,
                                          lapack_int ldb, double* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgels(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, lapack_complex_float* a,
                                     lapack_int lda, lapack_complex_float* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_cgels_work(
    int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgels(int matrix_layout, char trans,
                                     lapack_int m, lapack_int n,
                                     lapack_int nrhs, lapack_complex_double* a,
                                     lapack_int lda, lapack_complex_double* b,
                                     lapack_int ldb);
HWYBLAS_API lapack_int LAPACKE_zgels_work(
    int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sgelss(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs, float* a,
                                      lapack_int lda, float* b, lapack_int ldb,
                                      float* s, float rcond, lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_sgelss_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int nrhs,
                                           float* a, lapack_int lda, float* b,
                                           lapack_int ldb, float* s,
                                           float rcond, lapack_int* rank,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgelss(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs, double* a,
                                      lapack_int lda, double* b, lapack_int ldb,
                                      double* s, double rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_dgelss_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int nrhs,
                                           double* a, lapack_int lda, double* b,
                                           lapack_int ldb, double* s,
                                           double rcond, lapack_int* rank,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgelss(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb,
                                      float* s, float rcond, lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_cgelss_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* b,
    lapack_int ldb, float* s, float rcond, lapack_int* rank,
    lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgelss(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb,
                                      double* s, double rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_zgelss_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, double* s, double rcond, lapack_int* rank,
    lapack_complex_double* work, lapack_int lwork, double* rwork);
HWYBLAS_API lapack_int LAPACKE_sgelsd(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs, float* a,
                                      lapack_int lda, float* b, lapack_int ldb,
                                      float* s, float rcond, lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_sgelsd_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, float* a,
    lapack_int lda, float* b, lapack_int ldb, float* s, float rcond,
    lapack_int* rank, float* work, lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgelsd(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs, double* a,
                                      lapack_int lda, double* b, lapack_int ldb,
                                      double* s, double rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_dgelsd_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs, double* a,
    lapack_int lda, double* b, lapack_int ldb, double* s, double rcond,
    lapack_int* rank, double* work, lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cgelsd(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb,
                                      float* s, float rcond, lapack_int* rank);
HWYBLAS_API lapack_int
LAPACKE_cgelsd_work(int matrix_layout, lapack_int m, lapack_int n,
                    lapack_int nrhs, lapack_complex_float* a, lapack_int lda,
                    lapack_complex_float* b, lapack_int ldb, float* s,
                    float rcond, lapack_int* rank, lapack_complex_float* work,
                    lapack_int lwork, float* rwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_zgelsd(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb,
                                      double* s, double rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int
LAPACKE_zgelsd_work(int matrix_layout, lapack_int m, lapack_int n,
                    lapack_int nrhs, lapack_complex_double* a, lapack_int lda,
                    lapack_complex_double* b, lapack_int ldb, double* s,
                    double rcond, lapack_int* rank, lapack_complex_double* work,
                    lapack_int lwork, double* rwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_sgelsy(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs, float* a,
                                      lapack_int lda, float* b, lapack_int ldb,
                                      lapack_int* jpvt, float rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_sgelsy_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int nrhs,
                                           float* a, lapack_int lda, float* b,
                                           lapack_int ldb, lapack_int* jpvt,
                                           float rcond, lapack_int* rank,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgelsy(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs, double* a,
                                      lapack_int lda, double* b, lapack_int ldb,
                                      lapack_int* jpvt, double rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_dgelsy_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int nrhs,
                                           double* a, lapack_int lda, double* b,
                                           lapack_int ldb, lapack_int* jpvt,
                                           double rcond, lapack_int* rank,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgelsy(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb,
                                      lapack_int* jpvt, float rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_cgelsy_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* b,
    lapack_int ldb, lapack_int* jpvt, float rcond, lapack_int* rank,
    lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgelsy(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int nrhs,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb,
                                      lapack_int* jpvt, double rcond,
                                      lapack_int* rank);
HWYBLAS_API lapack_int LAPACKE_zgelsy_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, lapack_int* jpvt, double rcond, lapack_int* rank,
    lapack_complex_double* work, lapack_int lwork, double* rwork);
HWYBLAS_API lapack_int LAPACKE_sgglse(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int p, float* a,
                                      lapack_int lda, float* b, lapack_int ldb,
                                      float* c, float* d, float* x);
HWYBLAS_API lapack_int LAPACKE_sgglse_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int p, float* a,
                                           lapack_int lda, float* b,
                                           lapack_int ldb, float* c, float* d,
                                           float* x, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgglse(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int p, double* a,
                                      lapack_int lda, double* b, lapack_int ldb,
                                      double* c, double* d, double* x);
HWYBLAS_API lapack_int LAPACKE_dgglse_work(int matrix_layout, lapack_int m,
                                           lapack_int n, lapack_int p,
                                           double* a, lapack_int lda, double* b,
                                           lapack_int ldb, double* c, double* d,
                                           double* x, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgglse(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int p,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb,
                                      lapack_complex_float* c,
                                      lapack_complex_float* d,
                                      lapack_complex_float* x);
HWYBLAS_API lapack_int LAPACKE_cgglse_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int p,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* c, lapack_complex_float* d,
    lapack_complex_float* x, lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgglse(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_int p,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb,
                                      lapack_complex_double* c,
                                      lapack_complex_double* d,
                                      lapack_complex_double* x);
HWYBLAS_API lapack_int LAPACKE_zgglse_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_int p,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* c, lapack_complex_double* d,
    lapack_complex_double* x, lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sggglm(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p, float* a,
                                      lapack_int lda, float* b, lapack_int ldb,
                                      float* d, float* x, float* y);
HWYBLAS_API lapack_int LAPACKE_sggglm_work(int matrix_layout, lapack_int n,
                                           lapack_int m, lapack_int p, float* a,
                                           lapack_int lda, float* b,
                                           lapack_int ldb, float* d, float* x,
                                           float* y, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dggglm(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p, double* a,
                                      lapack_int lda, double* b, lapack_int ldb,
                                      double* d, double* x, double* y);
HWYBLAS_API lapack_int LAPACKE_dggglm_work(int matrix_layout, lapack_int n,
                                           lapack_int m, lapack_int p,
                                           double* a, lapack_int lda, double* b,
                                           lapack_int ldb, double* d, double* x,
                                           double* y, double* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cggglm(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb,
                                      lapack_complex_float* d,
                                      lapack_complex_float* x,
                                      lapack_complex_float* y);
HWYBLAS_API lapack_int LAPACKE_cggglm_work(
    int matrix_layout, lapack_int n, lapack_int m, lapack_int p,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* b,
    lapack_int ldb, lapack_complex_float* d, lapack_complex_float* x,
    lapack_complex_float* y, lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zggglm(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb,
                                      lapack_complex_double* d,
                                      lapack_complex_double* x,
                                      lapack_complex_double* y);
HWYBLAS_API lapack_int LAPACKE_zggglm_work(
    int matrix_layout, lapack_int n, lapack_int m, lapack_int p,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* b,
    lapack_int ldb, lapack_complex_double* d, lapack_complex_double* x,
    lapack_complex_double* y, lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sggqrf(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p, float* a,
                                      lapack_int lda, float* taua, float* b,
                                      lapack_int ldb, float* taub);
HWYBLAS_API lapack_int LAPACKE_sggqrf_work(int matrix_layout, lapack_int n,
                                           lapack_int m, lapack_int p, float* a,
                                           lapack_int lda, float* taua,
                                           float* b, lapack_int ldb,
                                           float* taub, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dggqrf(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p, double* a,
                                      lapack_int lda, double* taua, double* b,
                                      lapack_int ldb, double* taub);
HWYBLAS_API lapack_int LAPACKE_dggqrf_work(int matrix_layout, lapack_int n,
                                           lapack_int m, lapack_int p,
                                           double* a, lapack_int lda,
                                           double* taua, double* b,
                                           lapack_int ldb, double* taub,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cggqrf(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* taua,
                                      lapack_complex_float* b, lapack_int ldb,
                                      lapack_complex_float* taub);
HWYBLAS_API lapack_int LAPACKE_cggqrf_work(
    int matrix_layout, lapack_int n, lapack_int m, lapack_int p,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* taua,
    lapack_complex_float* b, lapack_int ldb, lapack_complex_float* taub,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zggqrf(int matrix_layout, lapack_int n,
                                      lapack_int m, lapack_int p,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* taua,
                                      lapack_complex_double* b, lapack_int ldb,
                                      lapack_complex_double* taub);
HWYBLAS_API lapack_int LAPACKE_zggqrf_work(
    int matrix_layout, lapack_int n, lapack_int m, lapack_int p,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* taua,
    lapack_complex_double* b, lapack_int ldb, lapack_complex_double* taub,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sggrqf(int matrix_layout, lapack_int m,
                                      lapack_int p, lapack_int n, float* a,
                                      lapack_int lda, float* taua, float* b,
                                      lapack_int ldb, float* taub);
HWYBLAS_API lapack_int LAPACKE_sggrqf_work(int matrix_layout, lapack_int m,
                                           lapack_int p, lapack_int n, float* a,
                                           lapack_int lda, float* taua,
                                           float* b, lapack_int ldb,
                                           float* taub, float* work,
                                           lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dggrqf(int matrix_layout, lapack_int m,
                                      lapack_int p, lapack_int n, double* a,
                                      lapack_int lda, double* taua, double* b,
                                      lapack_int ldb, double* taub);
HWYBLAS_API lapack_int LAPACKE_dggrqf_work(int matrix_layout, lapack_int m,
                                           lapack_int p, lapack_int n,
                                           double* a, lapack_int lda,
                                           double* taua, double* b,
                                           lapack_int ldb, double* taub,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cggrqf(int matrix_layout, lapack_int m,
                                      lapack_int p, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* taua,
                                      lapack_complex_float* b, lapack_int ldb,
                                      lapack_complex_float* taub);
HWYBLAS_API lapack_int LAPACKE_cggrqf_work(
    int matrix_layout, lapack_int m, lapack_int p, lapack_int n,
    lapack_complex_float* a, lapack_int lda, lapack_complex_float* taua,
    lapack_complex_float* b, lapack_int ldb, lapack_complex_float* taub,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zggrqf(int matrix_layout, lapack_int m,
                                      lapack_int p, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* taua,
                                      lapack_complex_double* b, lapack_int ldb,
                                      lapack_complex_double* taub);
HWYBLAS_API lapack_int LAPACKE_zggrqf_work(
    int matrix_layout, lapack_int m, lapack_int p, lapack_int n,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* taua,
    lapack_complex_double* b, lapack_int ldb, lapack_complex_double* taub,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sgerqf(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* tau);
HWYBLAS_API lapack_int LAPACKE_sgerqf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgerqf(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* tau);
HWYBLAS_API lapack_int LAPACKE_dgerqf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* tau,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cgerqf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_cgerqf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zgerqf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_zgerqf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_stzrzf(int matrix_layout, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda,
                                      float* tau);
HWYBLAS_API lapack_int LAPACKE_stzrzf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, float* tau,
                                           float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dtzrzf(int matrix_layout, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda,
                                      double* tau);
HWYBLAS_API lapack_int LAPACKE_dtzrzf_work(int matrix_layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, double* tau,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_ctzrzf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda,
                                      lapack_complex_float* tau);
HWYBLAS_API lapack_int LAPACKE_ctzrzf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a,
    lapack_int lda, lapack_complex_float* tau, lapack_complex_float* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_ztzrzf(int matrix_layout, lapack_int m,
                                      lapack_int n, lapack_complex_double* a,
                                      lapack_int lda,
                                      lapack_complex_double* tau);
HWYBLAS_API lapack_int LAPACKE_ztzrzf_work(
    int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    lapack_int lda, lapack_complex_double* tau, lapack_complex_double* work,
    lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sormrq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const float* a, lapack_int lda,
                                      const float* tau, float* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormrq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const float* a, lapack_int lda, const float* tau, float* c,
    lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormrq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const double* a, lapack_int lda,
                                      const double* tau, double* c,
                                      lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormrq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const double* a, lapack_int lda, const double* tau, double* c,
    lapack_int ldc, double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmrq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_float* a,
                                      lapack_int lda,
                                      const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmrq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmrq(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      const lapack_complex_double* a,
                                      lapack_int lda,
                                      const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmrq_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_sormrz(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      lapack_int l, const float* a,
                                      lapack_int lda, const float* tau,
                                      float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_sormrz_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, lapack_int l, const float* a, lapack_int lda,
    const float* tau, float* c, lapack_int ldc, float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dormrz(int matrix_layout, char side, char trans,
                                      lapack_int m, lapack_int n, lapack_int k,
                                      lapack_int l, const double* a,
                                      lapack_int lda, const double* tau,
                                      double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_dormrz_work(int matrix_layout, char side,
                                           char trans, lapack_int m,
                                           lapack_int n, lapack_int k,
                                           lapack_int l, const double* a,
                                           lapack_int lda, const double* tau,
                                           double* c, lapack_int ldc,
                                           double* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_cunmrz(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, lapack_int l, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_cunmrz_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, lapack_int l, const lapack_complex_float* a, lapack_int lda,
    const lapack_complex_float* tau, lapack_complex_float* c, lapack_int ldc,
    lapack_complex_float* work, lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_zunmrz(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, lapack_int l, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc);
HWYBLAS_API lapack_int LAPACKE_zunmrz_work(
    int matrix_layout, char side, char trans, lapack_int m, lapack_int n,
    lapack_int k, lapack_int l, const lapack_complex_double* a, lapack_int lda,
    const lapack_complex_double* tau, lapack_complex_double* c, lapack_int ldc,
    lapack_complex_double* work, lapack_int lwork);

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

HWYBLAS_API lapack_int LAPACKE_sgeev(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, float* a, lapack_int lda,
                                     float* wr, float* wi, float* vl,
                                     lapack_int ldvl, float* vr,
                                     lapack_int ldvr);
HWYBLAS_API lapack_int LAPACKE_dgeev(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, double* a, lapack_int lda,
                                     double* wr, double* wi, double* vl,
                                     lapack_int ldvl, double* vr,
                                     lapack_int ldvr);
HWYBLAS_API lapack_int LAPACKE_cgeev(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, lapack_complex_float* a,
                                     lapack_int lda, lapack_complex_float* w,
                                     lapack_complex_float* vl, lapack_int ldvl,
                                     lapack_complex_float* vr, lapack_int ldvr);
HWYBLAS_API lapack_int LAPACKE_zgeev(int matrix_layout, char jobvl, char jobvr,
                                     lapack_int n, lapack_complex_double* a,
                                     lapack_int lda, lapack_complex_double* w,
                                     lapack_complex_double* vl, lapack_int ldvl,
                                     lapack_complex_double* vr,
                                     lapack_int ldvr);
HWYBLAS_API lapack_int LAPACKE_sgeev_work(int matrix_layout, char jobvl,
                                          char jobvr, lapack_int n, float* a,
                                          lapack_int lda, float* wr, float* wi,
                                          float* vl, lapack_int ldvl, float* vr,
                                          lapack_int ldvr, float* work,
                                          lapack_int lwork);
HWYBLAS_API lapack_int LAPACKE_dgeev_work(
    int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a,
    lapack_int lda, double* wr, double* wi, double* vl, lapack_int ldvl,
    double* vr, lapack_int ldvr, double* work, lapack_int lwork);
HWYBLAS_API lapack_int
LAPACKE_cgeev_work(int matrix_layout, char jobvl, char jobvr, lapack_int n,
                   lapack_complex_float* a, lapack_int lda,
                   lapack_complex_float* w, lapack_complex_float* vl,
                   lapack_int ldvl, lapack_complex_float* vr, lapack_int ldvr,
                   lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgeev_work(
    int matrix_layout, char jobvl, char jobvr, lapack_int n,
    lapack_complex_double* a, lapack_int lda, lapack_complex_double* w,
    lapack_complex_double* vl, lapack_int ldvl, lapack_complex_double* vr,
    lapack_int ldvr, lapack_complex_double* work, lapack_int lwork,
    double* rwork);
HWYBLAS_API lapack_int LAPACKE_sgeevx(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, float* a, lapack_int lda, float* wr, float* wi, float* vl,
    lapack_int ldvl, float* vr, lapack_int ldvr, lapack_int* ilo,
    lapack_int* ihi, float* scale, float* abnrm, float* rconde, float* rcondv);
HWYBLAS_API lapack_int
LAPACKE_dgeevx(int matrix_layout, char balanc, char jobvl, char jobvr,
               char sense, lapack_int n, double* a, lapack_int lda, double* wr,
               double* wi, double* vl, lapack_int ldvl, double* vr,
               lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, double* scale,
               double* abnrm, double* rconde, double* rcondv);
HWYBLAS_API lapack_int LAPACKE_cgeevx(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, lapack_complex_float* a, lapack_int lda,
    lapack_complex_float* w, lapack_complex_float* vl, lapack_int ldvl,
    lapack_complex_float* vr, lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
    float* scale, float* abnrm, float* rconde, float* rcondv);
HWYBLAS_API lapack_int LAPACKE_zgeevx(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, lapack_complex_double* a, lapack_int lda,
    lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl,
    lapack_complex_double* vr, lapack_int ldvr, lapack_int* ilo,
    lapack_int* ihi, double* scale, double* abnrm, double* rconde,
    double* rcondv);
HWYBLAS_API lapack_int LAPACKE_sgeevx_work(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, float* a, lapack_int lda, float* wr, float* wi, float* vl,
    lapack_int ldvl, float* vr, lapack_int ldvr, lapack_int* ilo,
    lapack_int* ihi, float* scale, float* abnrm, float* rconde, float* rcondv,
    float* work, lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dgeevx_work(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, double* a, lapack_int lda, double* wr, double* wi, double* vl,
    lapack_int ldvl, double* vr, lapack_int ldvr, lapack_int* ilo,
    lapack_int* ihi, double* scale, double* abnrm, double* rconde,
    double* rcondv, double* work, lapack_int lwork, lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_cgeevx_work(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, lapack_complex_float* a, lapack_int lda,
    lapack_complex_float* w, lapack_complex_float* vl, lapack_int ldvl,
    lapack_complex_float* vr, lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
    float* scale, float* abnrm, float* rconde, float* rcondv,
    lapack_complex_float* work, lapack_int lwork, float* rwork);
HWYBLAS_API lapack_int LAPACKE_zgeevx_work(
    int matrix_layout, char balanc, char jobvl, char jobvr, char sense,
    lapack_int n, lapack_complex_double* a, lapack_int lda,
    lapack_complex_double* w, lapack_complex_double* vl, lapack_int ldvl,
    lapack_complex_double* vr, lapack_int ldvr, lapack_int* ilo,
    lapack_int* ihi, double* scale, double* abnrm, double* rconde,
    double* rcondv, lapack_complex_double* work, lapack_int lwork,
    double* rwork);

/* LAPACKE_?geesx (real) uses LIWORK >= 1: the reference passes
 * the IWORK(1) of the workspace query, which is 0 for n = 1 and sense = 'V'
 * or 'B', and ?geesx rejects it. */
HWYBLAS_API lapack_int LAPACKE_sgeesx(int matrix_layout, char jobvs, char sort,
                                      LAPACK_S_SELECT2 select, char sense,
                                      lapack_int n, float* a, lapack_int lda,
                                      lapack_int* sdim, float* wr, float* wi,
                                      float* vs, lapack_int ldvs, float* rconde,
                                      float* rcondv);
HWYBLAS_API lapack_int LAPACKE_dgeesx(int matrix_layout, char jobvs, char sort,
                                      LAPACK_D_SELECT2 select, char sense,
                                      lapack_int n, double* a, lapack_int lda,
                                      lapack_int* sdim, double* wr, double* wi,
                                      double* vs, lapack_int ldvs,
                                      double* rconde, double* rcondv);
HWYBLAS_API lapack_int LAPACKE_cgeesx(int matrix_layout, char jobvs, char sort,
                                      LAPACK_C_SELECT1 select, char sense,
                                      lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int* sdim,
                                      lapack_complex_float* w,
                                      lapack_complex_float* vs, lapack_int ldvs,
                                      float* rconde, float* rcondv);
HWYBLAS_API lapack_int LAPACKE_zgeesx(
    int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select,
    char sense, lapack_int n, lapack_complex_double* a, lapack_int lda,
    lapack_int* sdim, lapack_complex_double* w, lapack_complex_double* vs,
    lapack_int ldvs, double* rconde, double* rcondv);
HWYBLAS_API lapack_int LAPACKE_sgeesx_work(
    int matrix_layout, char jobvs, char sort, LAPACK_S_SELECT2 select,
    char sense, lapack_int n, float* a, lapack_int lda, lapack_int* sdim,
    float* wr, float* wi, float* vs, lapack_int ldvs, float* rconde,
    float* rcondv, float* work, lapack_int lwork, lapack_int* iwork,
    lapack_int liwork, lapack_logical* bwork);
HWYBLAS_API lapack_int LAPACKE_dgeesx_work(
    int matrix_layout, char jobvs, char sort, LAPACK_D_SELECT2 select,
    char sense, lapack_int n, double* a, lapack_int lda, lapack_int* sdim,
    double* wr, double* wi, double* vs, lapack_int ldvs, double* rconde,
    double* rcondv, double* work, lapack_int lwork, lapack_int* iwork,
    lapack_int liwork, lapack_logical* bwork);
HWYBLAS_API lapack_int LAPACKE_cgeesx_work(
    int matrix_layout, char jobvs, char sort, LAPACK_C_SELECT1 select,
    char sense, lapack_int n, lapack_complex_float* a, lapack_int lda,
    lapack_int* sdim, lapack_complex_float* w, lapack_complex_float* vs,
    lapack_int ldvs, float* rconde, float* rcondv, lapack_complex_float* work,
    lapack_int lwork, float* rwork, lapack_logical* bwork);
HWYBLAS_API lapack_int LAPACKE_zgeesx_work(
    int matrix_layout, char jobvs, char sort, LAPACK_Z_SELECT1 select,
    char sense, lapack_int n, lapack_complex_double* a, lapack_int lda,
    lapack_int* sdim, lapack_complex_double* w, lapack_complex_double* vs,
    lapack_int ldvs, double* rconde, double* rcondv,
    lapack_complex_double* work, lapack_int lwork, double* rwork,
    lapack_logical* bwork);

HWYBLAS_API lapack_int LAPACKE_strsna(int matrix_layout, char job, char howmny,
                                      const lapack_logical* select,
                                      lapack_int n, const float* t,
                                      lapack_int ldt, const float* vl,
                                      lapack_int ldvl, const float* vr,
                                      lapack_int ldvr, float* s, float* sep,
                                      lapack_int mm, lapack_int* m);
HWYBLAS_API lapack_int LAPACKE_dtrsna(int matrix_layout, char job, char howmny,
                                      const lapack_logical* select,
                                      lapack_int n, const double* t,
                                      lapack_int ldt, const double* vl,
                                      lapack_int ldvl, const double* vr,
                                      lapack_int ldvr, double* s, double* sep,
                                      lapack_int mm, lapack_int* m);
HWYBLAS_API lapack_int LAPACKE_ctrsna(
    int matrix_layout, char job, char howmny, const lapack_logical* select,
    lapack_int n, const lapack_complex_float* t, lapack_int ldt,
    const lapack_complex_float* vl, lapack_int ldvl,
    const lapack_complex_float* vr, lapack_int ldvr, float* s, float* sep,
    lapack_int mm, lapack_int* m);
HWYBLAS_API lapack_int LAPACKE_ztrsna(
    int matrix_layout, char job, char howmny, const lapack_logical* select,
    lapack_int n, const lapack_complex_double* t, lapack_int ldt,
    const lapack_complex_double* vl, lapack_int ldvl,
    const lapack_complex_double* vr, lapack_int ldvr, double* s, double* sep,
    lapack_int mm, lapack_int* m);
HWYBLAS_API lapack_int LAPACKE_strsna_work(
    int matrix_layout, char job, char howmny, const lapack_logical* select,
    lapack_int n, const float* t, lapack_int ldt, const float* vl,
    lapack_int ldvl, const float* vr, lapack_int ldvr, float* s, float* sep,
    lapack_int mm, lapack_int* m, float* work, lapack_int ldwork,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_dtrsna_work(
    int matrix_layout, char job, char howmny, const lapack_logical* select,
    lapack_int n, const double* t, lapack_int ldt, const double* vl,
    lapack_int ldvl, const double* vr, lapack_int ldvr, double* s, double* sep,
    lapack_int mm, lapack_int* m, double* work, lapack_int ldwork,
    lapack_int* iwork);
HWYBLAS_API lapack_int LAPACKE_ctrsna_work(
    int matrix_layout, char job, char howmny, const lapack_logical* select,
    lapack_int n, const lapack_complex_float* t, lapack_int ldt,
    const lapack_complex_float* vl, lapack_int ldvl,
    const lapack_complex_float* vr, lapack_int ldvr, float* s, float* sep,
    lapack_int mm, lapack_int* m, lapack_complex_float* work, lapack_int ldwork,
    float* rwork);
HWYBLAS_API lapack_int LAPACKE_ztrsna_work(
    int matrix_layout, char job, char howmny, const lapack_logical* select,
    lapack_int n, const lapack_complex_double* t, lapack_int ldt,
    const lapack_complex_double* vl, lapack_int ldvl,
    const lapack_complex_double* vr, lapack_int ldvr, double* s, double* sep,
    lapack_int mm, lapack_int* m, lapack_complex_double* work,
    lapack_int ldwork, double* rwork);

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
