/*
 * hwyBLAS CBLAS interface: the C API of the reference CBLAS (Netlib), with
 * row- and column-major storage. Also declares cblas_?gemmt (OpenBLAS and
 * MKL name of cblas_?gemmtr) and the by-value complex dot products
 * cblas_?dotu/dotc (OpenBLAS).
 *
 * Unlike the reference cblas_xerbla, the default cblas_xerbla prints a
 * message but does not terminate the program.
 *
 * Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.
 */
#ifndef HWYBLAS_CBLAS_H_
#define HWYBLAS_CBLAS_H_

#include <stddef.h>

#include "hwyblas/hwyblas.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Index type of cblas_i?amax (0-based). */
#define CBLAS_INDEX size_t

typedef enum CBLAS_LAYOUT {
  CblasRowMajor = 101,
  CblasColMajor = 102
} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;
/* Older name of CBLAS_LAYOUT. */
#define CBLAS_ORDER CBLAS_LAYOUT

/* Called with the 1-based position p of an invalid argument (the layout is
 * argument 1) and the routine name; prints "Parameter p to routine rout was
 * incorrect" and then `form`, a printf format for the remaining arguments. */
HWYBLAS_API void cblas_xerbla(hwyblas_int p, const char* rout, const char* form,
                              ...);

/* ------------------------------------------------------------------------ */
/* Level 1                                                                  */

/* Dot product. */
HWYBLAS_API float cblas_sdot(const hwyblas_int n, const float* x,
                             const hwyblas_int incx, const float* y,
                             const hwyblas_int incy);
HWYBLAS_API double cblas_ddot(const hwyblas_int n, const double* x,
                              const hwyblas_int incx, const double* y,
                              const hwyblas_int incy);
/* alpha + dot product, accumulated in double. */
HWYBLAS_API float cblas_sdsdot(const hwyblas_int n, const float alpha,
                               const float* x, const hwyblas_int incx,
                               const float* y, const hwyblas_int incy);
/* Dot product of float vectors, accumulated and returned in double. */
HWYBLAS_API double cblas_dsdot(const hwyblas_int n, const float* x,
                               const hwyblas_int incx, const float* y,
                               const hwyblas_int incy);
/* Complex dot product sum x_i * y_i (OpenBLAS extension: by value). */
HWYBLAS_API hwyblas_complex_float cblas_cdotu(const hwyblas_int n,
                                              const void* x,
                                              const hwyblas_int incx,
                                              const void* y,
                                              const hwyblas_int incy);
/* Complex dot product sum x_i * y_i, written to *ret. */
HWYBLAS_API void cblas_cdotu_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret);
/* Complex dot product sum conj(x_i) * y_i (OpenBLAS extension: by value). */
HWYBLAS_API hwyblas_complex_float cblas_cdotc(const hwyblas_int n,
                                              const void* x,
                                              const hwyblas_int incx,
                                              const void* y,
                                              const hwyblas_int incy);
/* Complex dot product sum conj(x_i) * y_i, written to *ret. */
HWYBLAS_API void cblas_cdotc_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret);
/* Complex dot product sum x_i * y_i (OpenBLAS extension: by value). */
HWYBLAS_API hwyblas_complex_double cblas_zdotu(const hwyblas_int n,
                                               const void* x,
                                               const hwyblas_int incx,
                                               const void* y,
                                               const hwyblas_int incy);
/* Complex dot product sum x_i * y_i, written to *ret. */
HWYBLAS_API void cblas_zdotu_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret);
/* Complex dot product sum conj(x_i) * y_i (OpenBLAS extension: by value). */
HWYBLAS_API hwyblas_complex_double cblas_zdotc(const hwyblas_int n,
                                               const void* x,
                                               const hwyblas_int incx,
                                               const void* y,
                                               const hwyblas_int incy);
/* Complex dot product sum conj(x_i) * y_i, written to *ret. */
HWYBLAS_API void cblas_zdotc_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret);
/* Euclidean norm. */
HWYBLAS_API float cblas_snrm2(const hwyblas_int n, const float* x,
                              const hwyblas_int incx);
HWYBLAS_API double cblas_dnrm2(const hwyblas_int n, const double* x,
                               const hwyblas_int incx);
HWYBLAS_API float cblas_scnrm2(const hwyblas_int n, const void* x,
                               const hwyblas_int incx);
HWYBLAS_API double cblas_dznrm2(const hwyblas_int n, const void* x,
                                const hwyblas_int incx);
/* Sum of |x_i| (|re| + |im| for complex). */
HWYBLAS_API float cblas_sasum(const hwyblas_int n, const float* x,
                              const hwyblas_int incx);
HWYBLAS_API double cblas_dasum(const hwyblas_int n, const double* x,
                               const hwyblas_int incx);
HWYBLAS_API float cblas_scasum(const hwyblas_int n, const void* x,
                               const hwyblas_int incx);
HWYBLAS_API double cblas_dzasum(const hwyblas_int n, const void* x,
                                const hwyblas_int incx);
/* 0-based index of the first element of largest |x_i| (|re| + |im|
 * for complex); 0 if n <= 0 or incx <= 0. */
HWYBLAS_API CBLAS_INDEX cblas_isamax(const hwyblas_int n, const float* x,
                                     const hwyblas_int incx);
HWYBLAS_API CBLAS_INDEX cblas_idamax(const hwyblas_int n, const double* x,
                                     const hwyblas_int incx);
HWYBLAS_API CBLAS_INDEX cblas_icamax(const hwyblas_int n, const void* x,
                                     const hwyblas_int incx);
HWYBLAS_API CBLAS_INDEX cblas_izamax(const hwyblas_int n, const void* x,
                                     const hwyblas_int incx);
/* x <-> y. */
HWYBLAS_API void cblas_sswap(const hwyblas_int n, float* x,
                             const hwyblas_int incx, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_dswap(const hwyblas_int n, double* x,
                             const hwyblas_int incx, double* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_cswap(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_zswap(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy);
/* y := x. */
HWYBLAS_API void cblas_scopy(const hwyblas_int n, const float* x,
                             const hwyblas_int incx, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_dcopy(const hwyblas_int n, const double* x,
                             const hwyblas_int incx, double* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_ccopy(const hwyblas_int n, const void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_zcopy(const hwyblas_int n, const void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy);
/* y := alpha * x + y. */
HWYBLAS_API void cblas_saxpy(const hwyblas_int n, const float alpha,
                             const float* x, const hwyblas_int incx, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_daxpy(const hwyblas_int n, const double alpha,
                             const double* x, const hwyblas_int incx, double* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_caxpy(const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx, void* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_zaxpy(const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx, void* y,
                             const hwyblas_int incy);
/* x := alpha * x. */
HWYBLAS_API void cblas_sscal(const hwyblas_int n, const float alpha, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dscal(const hwyblas_int n, const double alpha, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_cscal(const hwyblas_int n, const void* alpha, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_zscal(const hwyblas_int n, const void* alpha, void* x,
                             const hwyblas_int incx);
/* x := alpha * x, alpha real. */
HWYBLAS_API void cblas_csscal(const hwyblas_int n, const float alpha, void* x,
                              const hwyblas_int incx);
HWYBLAS_API void cblas_zdscal(const hwyblas_int n, const double alpha, void* x,
                              const hwyblas_int incx);
/* Plane rotation: x := c x + s y, y := c y - s x. */
HWYBLAS_API void cblas_srot(const hwyblas_int n, float* x,
                            const hwyblas_int incx, float* y,
                            const hwyblas_int incy, const float c,
                            const float s);
HWYBLAS_API void cblas_drot(const hwyblas_int n, double* x,
                            const hwyblas_int incx, double* y,
                            const hwyblas_int incy, const double c,
                            const double s);
HWYBLAS_API void cblas_csrot(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy, const float c,
                             const float s);
HWYBLAS_API void cblas_zdrot(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy, const double c,
                             const double s);
/* Constructs a Givens rotation. */
HWYBLAS_API void cblas_srotg(float* a, float* b, float* c, float* s);
HWYBLAS_API void cblas_drotg(double* a, double* b, double* c, double* s);
/* Constructs a complex Givens rotation. */
HWYBLAS_API void cblas_crotg(void* a, const void* b, float* c, void* s);
HWYBLAS_API void cblas_zrotg(void* a, const void* b, double* c, void* s);
/* Modified Givens transformation; p = (flag, h11, h21, h12, h22). */
HWYBLAS_API void cblas_srotm(const hwyblas_int n, float* x,
                             const hwyblas_int incx, float* y,
                             const hwyblas_int incy, const float* p);
HWYBLAS_API void cblas_drotm(const hwyblas_int n, double* x,
                             const hwyblas_int incx, double* y,
                             const hwyblas_int incy, const double* p);
/* Constructs a modified Givens transformation. */
HWYBLAS_API void cblas_srotmg(float* d1, float* d2, float* b1, const float b2,
                              float* p);
HWYBLAS_API void cblas_drotmg(double* d1, double* d2, double* b1,
                              const double b2, double* p);

/* ------------------------------------------------------------------------ */
/* Level 2                                                                  */

/* y := alpha * op(A) * x + beta * y. */
HWYBLAS_API void cblas_sgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* x, const hwyblas_int incx,
                             const float beta, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_dgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* x, const hwyblas_int incx,
                             const double beta, double* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_cgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y, const hwyblas_int incy);
HWYBLAS_API void cblas_zgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y, const hwyblas_int incy);

/* y := alpha * op(A) * x + beta * y, A banded. */
HWYBLAS_API void cblas_sgbmv(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, const hwyblas_int m,
    const hwyblas_int n, const hwyblas_int kl, const hwyblas_int ku,
    const float alpha, const float* a, const hwyblas_int lda, const float* x,
    const hwyblas_int incx, const float beta, float* y, const hwyblas_int incy);
HWYBLAS_API void cblas_dgbmv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int kl,
                             const hwyblas_int ku, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* x, const hwyblas_int incx,
                             const double beta, double* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_cgbmv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int kl,
                             const hwyblas_int ku, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y, const hwyblas_int incy);
HWYBLAS_API void cblas_zgbmv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int kl,
                             const hwyblas_int ku, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y, const hwyblas_int incy);

/* x := op(A) * x, A triangular. */
HWYBLAS_API void cblas_strmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* a,
                             const hwyblas_int lda, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dtrmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* a,
                             const hwyblas_int lda, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ctrmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ztrmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx);

/* Solves op(A) * x = b, A triangular. */
HWYBLAS_API void cblas_strsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* a,
                             const hwyblas_int lda, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dtrsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* a,
                             const hwyblas_int lda, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ctrsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ztrsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx);

/* x := op(A) * x, A triangular banded. */
HWYBLAS_API void cblas_stbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const float* a, const hwyblas_int lda, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dtbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const double* a, const hwyblas_int lda, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ctbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ztbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx);

/* Solves op(A) * x = b, A triangular banded. */
HWYBLAS_API void cblas_stbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const float* a, const hwyblas_int lda, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dtbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const double* a, const hwyblas_int lda, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ctbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ztbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx);

/* x := op(A) * x, A triangular packed. */
HWYBLAS_API void cblas_stpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* ap, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dtpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* ap, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ctpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ztpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx);

/* Solves op(A) * x = b, A triangular packed. */
HWYBLAS_API void cblas_stpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* ap, float* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_dtpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* ap, double* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ctpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx);
HWYBLAS_API void cblas_ztpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx);

/* y := alpha * A * x + beta * y, A symmetric. */
HWYBLAS_API void cblas_ssymv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* x, const hwyblas_int incx,
                             const float beta, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_dsymv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* x, const hwyblas_int incx,
                             const double beta, double* y,
                             const hwyblas_int incy);

/* y := alpha * A * x + beta * y, A Hermitian. */
HWYBLAS_API void cblas_chemv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y, const hwyblas_int incy);
HWYBLAS_API void cblas_zhemv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y, const hwyblas_int incy);

/* y := alpha * A * x + beta * y, A symmetric banded. */
HWYBLAS_API void cblas_ssbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const float alpha, const float* a,
                             const hwyblas_int lda, const float* x,
                             const hwyblas_int incx, const float beta, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_dsbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const double alpha, const double* a,
                             const hwyblas_int lda, const double* x,
                             const hwyblas_int incx, const double beta,
                             double* y, const hwyblas_int incy);

/* y := alpha * A * x + beta * y, A Hermitian banded. */
HWYBLAS_API void cblas_chbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_zhbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy);

/* y := alpha * A * x + beta * y, A symmetric packed. */
HWYBLAS_API void cblas_sspmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* ap, const float* x,
                             const hwyblas_int incx, const float beta, float* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_dspmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* ap, const double* x,
                             const hwyblas_int incx, const double beta,
                             double* y, const hwyblas_int incy);

/* y := alpha * A * x + beta * y, A Hermitian packed. */
HWYBLAS_API void cblas_chpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* ap, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy);
HWYBLAS_API void cblas_zhpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* ap, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy);

/* A := alpha * x * y^T + A. */
HWYBLAS_API void cblas_sger(const CBLAS_LAYOUT layout, const hwyblas_int m,
                            const hwyblas_int n, const float alpha,
                            const float* x, const hwyblas_int incx,
                            const float* y, const hwyblas_int incy, float* a,
                            const hwyblas_int lda);
HWYBLAS_API void cblas_dger(const CBLAS_LAYOUT layout, const hwyblas_int m,
                            const hwyblas_int n, const double alpha,
                            const double* x, const hwyblas_int incx,
                            const double* y, const hwyblas_int incy, double* a,
                            const hwyblas_int lda);

/* A := alpha * x * y^T + A. */
HWYBLAS_API void cblas_cgeru(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda);
HWYBLAS_API void cblas_zgeru(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda);

/* A := alpha * x * y^H + A. */
HWYBLAS_API void cblas_cgerc(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda);
HWYBLAS_API void cblas_zgerc(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda);

/* A := alpha * x * x^T + A, A symmetric. */
HWYBLAS_API void cblas_ssyr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const float* x, const hwyblas_int incx, float* a,
                            const hwyblas_int lda);
HWYBLAS_API void cblas_dsyr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const double* x, const hwyblas_int incx, double* a,
                            const hwyblas_int lda);

/* A := alpha * x * x^H + A, A Hermitian, alpha real. */
HWYBLAS_API void cblas_cher(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const void* x, const hwyblas_int incx, void* a,
                            const hwyblas_int lda);
HWYBLAS_API void cblas_zher(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const void* x, const hwyblas_int incx, void* a,
                            const hwyblas_int lda);

/* A := alpha * x * x^T + A, A symmetric packed. */
HWYBLAS_API void cblas_sspr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const float* x, const hwyblas_int incx, float* ap);
HWYBLAS_API void cblas_dspr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const double* x, const hwyblas_int incx,
                            double* ap);

/* A := alpha * x * x^H + A, A Hermitian packed, alpha real. */
HWYBLAS_API void cblas_chpr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const void* x, const hwyblas_int incx, void* ap);
HWYBLAS_API void cblas_zhpr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const void* x, const hwyblas_int incx, void* ap);

/* A := alpha * x * y^T + alpha * y * x^T + A, A symmetric. */
HWYBLAS_API void cblas_ssyr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* x, const hwyblas_int incx,
                             const float* y, const hwyblas_int incy, float* a,
                             const hwyblas_int lda);
HWYBLAS_API void cblas_dsyr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* x, const hwyblas_int incx,
                             const double* y, const hwyblas_int incy, double* a,
                             const hwyblas_int lda);

/* A := alpha * x * y^H + conj(alpha) * y * x^H + A, A Hermitian. */
HWYBLAS_API void cblas_cher2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda);
HWYBLAS_API void cblas_zher2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda);

/* A := alpha * x * y^T + alpha * y * x^T + A, A symmetric packed. */
HWYBLAS_API void cblas_sspr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* x, const hwyblas_int incx,
                             const float* y, const hwyblas_int incy, float* ap);
HWYBLAS_API void cblas_dspr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* x, const hwyblas_int incx,
                             const double* y, const hwyblas_int incy,
                             double* ap);

/* A := alpha * x * y^H + conj(alpha) * y * x^H + A, A Hermitian packed. */
HWYBLAS_API void cblas_chpr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* ap);
HWYBLAS_API void cblas_zhpr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* ap);

/* ------------------------------------------------------------------------ */
/* Level 3                                                                  */

/* C := alpha * op(A) * op(B) + beta * C. */
HWYBLAS_API void cblas_sgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const float alpha, const float* a,
                             const hwyblas_int lda, const float* b,
                             const hwyblas_int ldb, const float beta, float* c,
                             const hwyblas_int ldc);
HWYBLAS_API void cblas_dgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const double alpha, const double* a,
                             const hwyblas_int lda, const double* b,
                             const hwyblas_int ldb, const double beta,
                             double* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_cgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* b,
                             const hwyblas_int ldb, const void* beta, void* c,
                             const hwyblas_int ldc);
HWYBLAS_API void cblas_zgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* b,
                             const hwyblas_int ldb, const void* beta, void* c,
                             const hwyblas_int ldc);

/* As ?gemm for the uplo triangle of the n x n matrix C only. */
HWYBLAS_API void cblas_sgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const float alpha, const float* a,
                               const hwyblas_int lda, const float* b,
                               const hwyblas_int ldb, const float beta,
                               float* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_dgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const double alpha, const double* a,
                               const hwyblas_int lda, const double* b,
                               const hwyblas_int ldb, const double beta,
                               double* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_cgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const void* alpha, const void* a,
                               const hwyblas_int lda, const void* b,
                               const hwyblas_int ldb, const void* beta, void* c,
                               const hwyblas_int ldc);
HWYBLAS_API void cblas_zgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const void* alpha, const void* a,
                               const hwyblas_int lda, const void* b,
                               const hwyblas_int ldb, const void* beta, void* c,
                               const hwyblas_int ldc);

/* Same as cblas_?gemmtr (the name used by OpenBLAS and MKL). */
HWYBLAS_API void cblas_sgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const float alpha,
                              const float* a, const hwyblas_int lda,
                              const float* b, const hwyblas_int ldb,
                              const float beta, float* c,
                              const hwyblas_int ldc);
HWYBLAS_API void cblas_dgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const double alpha,
                              const double* a, const hwyblas_int lda,
                              const double* b, const hwyblas_int ldb,
                              const double beta, double* c,
                              const hwyblas_int ldc);
HWYBLAS_API void cblas_cgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c, const hwyblas_int ldc);

/* C := alpha * A * B + beta * C (side L) or alpha * B * A + beta * C, A
 * symmetric. */
HWYBLAS_API void cblas_ssymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* b, const hwyblas_int ldb,
                             const float beta, float* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_dsymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* b, const hwyblas_int ldb,
                             const double beta, double* c,
                             const hwyblas_int ldc);
HWYBLAS_API void cblas_csymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zsymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc);

/* As cblas_?symm, A Hermitian. */
HWYBLAS_API void cblas_chemm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zhemm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc);

/* C := alpha * op(A) * op(A)^T + beta * C, uplo triangle of C only. */
HWYBLAS_API void cblas_ssyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float beta, float* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_dsyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double beta, double* c,
                             const hwyblas_int ldc);
HWYBLAS_API void cblas_csyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zsyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* beta, void* c, const hwyblas_int ldc);

/* C := alpha * op(A) * op(A)^H + beta * C, alpha and beta real. */
HWYBLAS_API void cblas_cherk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const float alpha,
                             const void* a, const hwyblas_int lda,
                             const float beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zherk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const double alpha,
                             const void* a, const hwyblas_int lda,
                             const double beta, void* c, const hwyblas_int ldc);

/* C := alpha * op(A) * op(B)^T + alpha * op(B) * op(A)^T + beta * C. */
HWYBLAS_API void cblas_ssyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const float alpha,
                              const float* a, const hwyblas_int lda,
                              const float* b, const hwyblas_int ldb,
                              const float beta, float* c,
                              const hwyblas_int ldc);
HWYBLAS_API void cblas_dsyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const double alpha,
                              const double* a, const hwyblas_int lda,
                              const double* b, const hwyblas_int ldb,
                              const double beta, double* c,
                              const hwyblas_int ldc);
HWYBLAS_API void cblas_csyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zsyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c, const hwyblas_int ldc);

/* C := alpha * op(A) * op(B)^H + conj(alpha) * op(B) * op(A)^H + beta * C,
 * beta real. */
HWYBLAS_API void cblas_cher2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const float beta, void* c, const hwyblas_int ldc);
HWYBLAS_API void cblas_zher2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const double beta, void* c,
                              const hwyblas_int ldc);

/* B := alpha * op(A) * B (side L) or alpha * B * op(A), A triangular. */
HWYBLAS_API void cblas_strmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda, float* b,
                             const hwyblas_int ldb);
HWYBLAS_API void cblas_dtrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda, double* b,
                             const hwyblas_int ldb);
HWYBLAS_API void cblas_ctrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb);
HWYBLAS_API void cblas_ztrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb);

/* Solves op(A) * X = alpha * B (side L) or X * op(A) = alpha * B. */
HWYBLAS_API void cblas_strsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda, float* b,
                             const hwyblas_int ldb);
HWYBLAS_API void cblas_dtrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda, double* b,
                             const hwyblas_int ldb);
HWYBLAS_API void cblas_ctrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb);
HWYBLAS_API void cblas_ztrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_CBLAS_H_ */
