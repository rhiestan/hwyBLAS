/*
 * hwyBLAS - a BLAS implementation built on Google Highway.
 *
 * Public C interface. All routines use the classic Fortran-77 BLAS calling
 * convention (lower-case name with a trailing underscore, every argument
 * passed by pointer, column-major storage), which is what Eigen expects when
 * compiled with EIGEN_USE_BLAS.
 *
 * Complex arguments are passed as float* / double* pointing to interleaved
 * (real, imag) pairs, i.e. the memory layout of std::complex<T>, T _Complex
 * and Fortran COMPLEX. The declarations match Eigen's Eigen/src/misc/blas.h,
 * so both headers can be included in the same translation unit.
 *
 * Fortran passes the length of CHARACTER arguments as hidden trailing
 * arguments. hwyBLAS only looks at the first character, so these hidden
 * arguments are neither declared nor needed.
 */
#ifndef HWYBLAS_HWYBLAS_H_
#define HWYBLAS_HWYBLAS_H_

#include <stddef.h>
#include <stdint.h>

/* Exported symbols. Consumers need no __declspec(dllimport) (functions are
 * resolved through the import library), which keeps these declarations
 * compatible with other BLAS headers such as Eigen's. */
#if defined(_WIN32)
#if defined(HWYBLAS_BUILDING) && defined(HWYBLAS_SHARED_BUILD)
#define HWYBLAS_API __declspec(dllexport)
#else
#define HWYBLAS_API
#endif
#else
#define HWYBLAS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Integer type of all dimension, stride and increment arguments (LP64). */
typedef int hwyblas_int;

/* Complex function results (layout-compatible with std::complex and
 * C99 _Complex; returned by value like gfortran and OpenBLAS do). */
typedef struct {
  float real, imag;
} hwyblas_complex_float;
typedef struct {
  double real, imag;
} hwyblas_complex_double;

/* ------------------------------------------------------------------------ */
/* Level 1                                                                  */

/* y := alpha * x + y. */
HWYBLAS_API void saxpy_(const hwyblas_int* n, const float* alpha,
                        const float* x, const hwyblas_int* incx, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void daxpy_(const hwyblas_int* n, const double* alpha,
                        const double* x, const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy);
HWYBLAS_API void caxpy_(const hwyblas_int* n, const float* alpha,
                        const float* x, const hwyblas_int* incx, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zaxpy_(const hwyblas_int* n, const double* alpha,
                        const double* x, const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy);

/* x := alpha * x (alpha real for csscal_/zdscal_). As in the reference
 * BLAS, incx <= 0 is a no-op. */
HWYBLAS_API void sscal_(const hwyblas_int* n, const float* alpha, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dscal_(const hwyblas_int* n, const double* alpha, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void cscal_(const hwyblas_int* n, const float* alpha, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void zscal_(const hwyblas_int* n, const double* alpha, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void csscal_(const hwyblas_int* n, const float* alpha, float* x,
                         const hwyblas_int* incx);
HWYBLAS_API void zdscal_(const hwyblas_int* n, const double* alpha, double* x,
                         const hwyblas_int* incx);

/* y := x. */
HWYBLAS_API void scopy_(const hwyblas_int* n, const float* x,
                        const hwyblas_int* incx, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void dcopy_(const hwyblas_int* n, const double* x,
                        const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy);
HWYBLAS_API void ccopy_(const hwyblas_int* n, const float* x,
                        const hwyblas_int* incx, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zcopy_(const hwyblas_int* n, const double* x,
                        const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy);

/* x <-> y. */
HWYBLAS_API void sswap_(const hwyblas_int* n, float* x, const hwyblas_int* incx,
                        float* y, const hwyblas_int* incy);
HWYBLAS_API void dswap_(const hwyblas_int* n, double* x,
                        const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy);
HWYBLAS_API void cswap_(const hwyblas_int* n, float* x, const hwyblas_int* incx,
                        float* y, const hwyblas_int* incy);
HWYBLAS_API void zswap_(const hwyblas_int* n, double* x,
                        const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy);

/* Dot products sum x_i * y_i (dotc: sum conj(x_i) * y_i). sdsdot_ returns
 * sb + the dot product and dsdot_ the dot product, both accumulated in
 * double precision. Complex results are returned by value. */
HWYBLAS_API float sdot_(const hwyblas_int* n, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy);
HWYBLAS_API double ddot_(const hwyblas_int* n, const double* x,
                         const hwyblas_int* incx, const double* y,
                         const hwyblas_int* incy);
HWYBLAS_API float sdsdot_(const hwyblas_int* n, const float* sb, const float* x,
                          const hwyblas_int* incx, const float* y,
                          const hwyblas_int* incy);
HWYBLAS_API double dsdot_(const hwyblas_int* n, const float* x,
                          const hwyblas_int* incx, const float* y,
                          const hwyblas_int* incy);
HWYBLAS_API hwyblas_complex_float cdotu_(const hwyblas_int* n, const float* x,
                                         const hwyblas_int* incx,
                                         const float* y,
                                         const hwyblas_int* incy);
HWYBLAS_API hwyblas_complex_float cdotc_(const hwyblas_int* n, const float* x,
                                         const hwyblas_int* incx,
                                         const float* y,
                                         const hwyblas_int* incy);
HWYBLAS_API hwyblas_complex_double zdotu_(const hwyblas_int* n, const double* x,
                                          const hwyblas_int* incx,
                                          const double* y,
                                          const hwyblas_int* incy);
HWYBLAS_API hwyblas_complex_double zdotc_(const hwyblas_int* n, const double* x,
                                          const hwyblas_int* incx,
                                          const double* y,
                                          const hwyblas_int* incy);

/* Euclidean norm, without harmful overflow or underflow. */
HWYBLAS_API float snrm2_(const hwyblas_int* n, const float* x,
                         const hwyblas_int* incx);
HWYBLAS_API double dnrm2_(const hwyblas_int* n, const double* x,
                          const hwyblas_int* incx);
HWYBLAS_API float scnrm2_(const hwyblas_int* n, const float* x,
                          const hwyblas_int* incx);
HWYBLAS_API double dznrm2_(const hwyblas_int* n, const double* x,
                           const hwyblas_int* incx);

/* Sum of |x_i| (|re| + |im| for complex vectors). */
HWYBLAS_API float sasum_(const hwyblas_int* n, const float* x,
                         const hwyblas_int* incx);
HWYBLAS_API double dasum_(const hwyblas_int* n, const double* x,
                          const hwyblas_int* incx);
HWYBLAS_API float scasum_(const hwyblas_int* n, const float* x,
                          const hwyblas_int* incx);
HWYBLAS_API double dzasum_(const hwyblas_int* n, const double* x,
                           const hwyblas_int* incx);

/* 1-based index of the first element of largest |x_i| (|re| + |im| for
 * complex vectors); 0 if n < 1 or incx <= 0. */
HWYBLAS_API hwyblas_int isamax_(const hwyblas_int* n, const float* x,
                                const hwyblas_int* incx);
HWYBLAS_API hwyblas_int idamax_(const hwyblas_int* n, const double* x,
                                const hwyblas_int* incx);
HWYBLAS_API hwyblas_int icamax_(const hwyblas_int* n, const float* x,
                                const hwyblas_int* incx);
HWYBLAS_API hwyblas_int izamax_(const hwyblas_int* n, const double* x,
                                const hwyblas_int* incx);

/* Plane rotation: x := c x + s y, y := c y - s x (c, s real). */
HWYBLAS_API void srot_(const hwyblas_int* n, float* x, const hwyblas_int* incx,
                       float* y, const hwyblas_int* incy, const float* c,
                       const float* s);
HWYBLAS_API void drot_(const hwyblas_int* n, double* x, const hwyblas_int* incx,
                       double* y, const hwyblas_int* incy, const double* c,
                       const double* s);
HWYBLAS_API void csrot_(const hwyblas_int* n, float* x, const hwyblas_int* incx,
                        float* y, const hwyblas_int* incy, const float* c,
                        const float* s);
HWYBLAS_API void zdrot_(const hwyblas_int* n, double* x,
                        const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy, const double* c,
                        const double* s);

/* Modified Givens transformation; param = (flag, h11, h21, h12, h22). */
HWYBLAS_API void srotm_(const hwyblas_int* n, float* x, const hwyblas_int* incx,
                        float* y, const hwyblas_int* incy, const float* param);
HWYBLAS_API void drotm_(const hwyblas_int* n, double* x,
                        const hwyblas_int* incx, double* y,
                        const hwyblas_int* incy, const double* param);

/* Constructs a Givens rotation such that [c s; -conj(s) c] [a; b] = [r; 0];
 * a is overwritten by r (and, for real types, b by a reconstruction value). */
HWYBLAS_API void srotg_(float* a, float* b, float* c, float* s);
HWYBLAS_API void drotg_(double* a, double* b, double* c, double* s);
HWYBLAS_API void crotg_(float* a, const float* b, float* c, float* s);
HWYBLAS_API void zrotg_(double* a, const double* b, double* c, double* s);

/* Constructs a modified Givens transformation. */
HWYBLAS_API void srotmg_(float* d1, float* d2, float* x1, const float* y1,
                         float* param);
HWYBLAS_API void drotmg_(double* d1, double* d2, double* x1, const double* y1,
                         double* param);

/* |re(z)| + |im(z)|. */
HWYBLAS_API float scabs1_(const float* z);
HWYBLAS_API double dcabs1_(const double* z);

/* ------------------------------------------------------------------------ */
/* Level 2                                                                  */

/* y := alpha * op(A) * x + beta * y, where op(A) is A, A^T or A^H. */
HWYBLAS_API void sgemv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void dgemv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* x, const hwyblas_int* incx,
                        const double* beta, double* y, const hwyblas_int* incy);
HWYBLAS_API void cgemv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zgemv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* x, const hwyblas_int* incx,
                        const double* beta, double* y, const hwyblas_int* incy);

/* y := alpha * A * x + beta * y, A symmetric (one triangle stored). */
HWYBLAS_API void ssymv_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void dsymv_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* x,
                        const hwyblas_int* incx, const double* beta, double* y,
                        const hwyblas_int* incy);

/* y := alpha * A * x + beta * y, A Hermitian (one triangle stored). */
HWYBLAS_API void chemv_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zhemv_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* x,
                        const hwyblas_int* incx, const double* beta, double* y,
                        const hwyblas_int* incy);

/* x := op(A) * x, A triangular. */
HWYBLAS_API void strmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* a,
                        const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dtrmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* a,
                        const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ctrmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* a,
                        const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ztrmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* a,
                        const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);

/* Solves op(A) * x = b for x, A triangular; x overwrites b. */
HWYBLAS_API void strsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* a,
                        const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dtrsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* a,
                        const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ctrsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* a,
                        const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ztrsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* a,
                        const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);

/* A := alpha * x * y^T + A (m x n). */
HWYBLAS_API void sger_(const hwyblas_int* m, const hwyblas_int* n,
                       const float* alpha, const float* x,
                       const hwyblas_int* incx, const float* y,
                       const hwyblas_int* incy, float* a,
                       const hwyblas_int* lda);
HWYBLAS_API void dger_(const hwyblas_int* m, const hwyblas_int* n,
                       const double* alpha, const double* x,
                       const hwyblas_int* incx, const double* y,
                       const hwyblas_int* incy, double* a,
                       const hwyblas_int* lda);

/* A := alpha * x * y^T + A (m x n). */
HWYBLAS_API void cgeru_(const hwyblas_int* m, const hwyblas_int* n,
                        const float* alpha, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy, float* a,
                        const hwyblas_int* lda);
HWYBLAS_API void zgeru_(const hwyblas_int* m, const hwyblas_int* n,
                        const double* alpha, const double* x,
                        const hwyblas_int* incx, const double* y,
                        const hwyblas_int* incy, double* a,
                        const hwyblas_int* lda);

/* A := alpha * x * y^H + A (m x n). */
HWYBLAS_API void cgerc_(const hwyblas_int* m, const hwyblas_int* n,
                        const float* alpha, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy, float* a,
                        const hwyblas_int* lda);
HWYBLAS_API void zgerc_(const hwyblas_int* m, const hwyblas_int* n,
                        const double* alpha, const double* x,
                        const hwyblas_int* incx, const double* y,
                        const hwyblas_int* incy, double* a,
                        const hwyblas_int* lda);

/* A := alpha * x * x^T + A, A symmetric (one triangle stored). */
HWYBLAS_API void ssyr_(const char* uplo, const hwyblas_int* n,
                       const float* alpha, const float* x,
                       const hwyblas_int* incx, float* a,
                       const hwyblas_int* lda);
HWYBLAS_API void dsyr_(const char* uplo, const hwyblas_int* n,
                       const double* alpha, const double* x,
                       const hwyblas_int* incx, double* a,
                       const hwyblas_int* lda);

/* A := alpha * x * x^H + A, A Hermitian (one triangle stored), alpha real.
 * The imaginary parts of the diagonal are set to zero. */
HWYBLAS_API void cher_(const char* uplo, const hwyblas_int* n,
                       const float* alpha, const float* x,
                       const hwyblas_int* incx, float* a,
                       const hwyblas_int* lda);
HWYBLAS_API void zher_(const char* uplo, const hwyblas_int* n,
                       const double* alpha, const double* x,
                       const hwyblas_int* incx, double* a,
                       const hwyblas_int* lda);

/* A := alpha * x * y^T + alpha * y * x^T + A, A symmetric (one triangle
 * stored). */
HWYBLAS_API void ssyr2_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy, float* a,
                        const hwyblas_int* lda);
HWYBLAS_API void dsyr2_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* x,
                        const hwyblas_int* incx, const double* y,
                        const hwyblas_int* incy, double* a,
                        const hwyblas_int* lda);

/* A := alpha * x * y^H + conj(alpha) * y * x^H + A, A Hermitian (one
 * triangle stored). The imaginary parts of the diagonal are set to zero. */
HWYBLAS_API void cher2_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy, float* a,
                        const hwyblas_int* lda);
HWYBLAS_API void zher2_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* x,
                        const hwyblas_int* incx, const double* y,
                        const hwyblas_int* incy, double* a,
                        const hwyblas_int* lda);

/* Banded and packed matrices. */

/* y := alpha * op(A) * x + beta * y, A m x n banded with kl sub- and ku
 * super-diagonals. */
HWYBLAS_API void sgbmv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void dgbmv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* x, const hwyblas_int* incx,
                        const double* beta, double* y, const hwyblas_int* incy);
HWYBLAS_API void cgbmv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zgbmv_(const char* trans, const hwyblas_int* m,
                        const hwyblas_int* n, const hwyblas_int* kl,
                        const hwyblas_int* ku, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* x, const hwyblas_int* incx,
                        const double* beta, double* y, const hwyblas_int* incy);

/* y := alpha * A * x + beta * y, A symmetric banded with k off-diagonals
 * (one triangle stored). */
HWYBLAS_API void ssbmv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* k, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void dsbmv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* k, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* x, const hwyblas_int* incx,
                        const double* beta, double* y, const hwyblas_int* incy);

/* y := alpha * A * x + beta * y, A Hermitian banded with k off-diagonals
 * (one triangle stored). */
HWYBLAS_API void chbmv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* k, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zhbmv_(const char* uplo, const hwyblas_int* n,
                        const hwyblas_int* k, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* x, const hwyblas_int* incx,
                        const double* beta, double* y, const hwyblas_int* incy);

/* y := alpha * A * x + beta * y, A symmetric packed. */
HWYBLAS_API void sspmv_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* ap, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void dspmv_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* ap, const double* x,
                        const hwyblas_int* incx, const double* beta, double* y,
                        const hwyblas_int* incy);

/* y := alpha * A * x + beta * y, A Hermitian packed. */
HWYBLAS_API void chpmv_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* ap, const float* x,
                        const hwyblas_int* incx, const float* beta, float* y,
                        const hwyblas_int* incy);
HWYBLAS_API void zhpmv_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* ap, const double* x,
                        const hwyblas_int* incx, const double* beta, double* y,
                        const hwyblas_int* incy);

/* x := op(A) * x, A triangular banded with k off-diagonals. */
HWYBLAS_API void stbmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* a, const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dtbmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* a, const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ctbmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* a, const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ztbmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* a, const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);

/* Solves op(A) * x = b, A triangular banded with k off-diagonals. */
HWYBLAS_API void stbsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* a, const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dtbsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* a, const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ctbsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* a, const hwyblas_int* lda, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ztbsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* a, const hwyblas_int* lda, double* x,
                        const hwyblas_int* incx);

/* x := op(A) * x, A triangular packed. */
HWYBLAS_API void stpmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* ap, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dtpmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* ap, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ctpmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* ap, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ztpmv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* ap, double* x,
                        const hwyblas_int* incx);

/* Solves op(A) * x = b, A triangular packed. */
HWYBLAS_API void stpsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* ap, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void dtpsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* ap, double* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ctpsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const float* ap, float* x,
                        const hwyblas_int* incx);
HWYBLAS_API void ztpsv_(const char* uplo, const char* trans, const char* diag,
                        const hwyblas_int* n, const double* ap, double* x,
                        const hwyblas_int* incx);

/* A := alpha * x * x^T + A, A symmetric packed. */
HWYBLAS_API void sspr_(const char* uplo, const hwyblas_int* n,
                       const float* alpha, const float* x,
                       const hwyblas_int* incx, float* ap);
HWYBLAS_API void dspr_(const char* uplo, const hwyblas_int* n,
                       const double* alpha, const double* x,
                       const hwyblas_int* incx, double* ap);

/* A := alpha * x * x^H + A, A Hermitian packed, alpha real. The imaginary
 * parts of the diagonal are set to zero. */
HWYBLAS_API void chpr_(const char* uplo, const hwyblas_int* n,
                       const float* alpha, const float* x,
                       const hwyblas_int* incx, float* ap);
HWYBLAS_API void zhpr_(const char* uplo, const hwyblas_int* n,
                       const double* alpha, const double* x,
                       const hwyblas_int* incx, double* ap);

/* A := alpha * x * y^T + alpha * y * x^T + A, A symmetric packed. */
HWYBLAS_API void sspr2_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy, float* ap);
HWYBLAS_API void dspr2_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* x,
                        const hwyblas_int* incx, const double* y,
                        const hwyblas_int* incy, double* ap);

/* A := alpha * x * y^H + conj(alpha) * y * x^H + A, A Hermitian packed. The
 * imaginary parts of the diagonal are set to zero. */
HWYBLAS_API void chpr2_(const char* uplo, const hwyblas_int* n,
                        const float* alpha, const float* x,
                        const hwyblas_int* incx, const float* y,
                        const hwyblas_int* incy, float* ap);
HWYBLAS_API void zhpr2_(const char* uplo, const hwyblas_int* n,
                        const double* alpha, const double* x,
                        const hwyblas_int* incx, const double* y,
                        const hwyblas_int* incy, double* ap);

/* ------------------------------------------------------------------------ */
/* Level 3                                                                  */

/* C := alpha * op(A) * op(B) + beta * C, where op(X) is X, X^T or X^H. */
HWYBLAS_API void sgemm_(const char* transa, const char* transb,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const hwyblas_int* k, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* b,
                        const hwyblas_int* ldb, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void dgemm_(const char* transa, const char* transb,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const hwyblas_int* k, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* b, const hwyblas_int* ldb,
                        const double* beta, double* c, const hwyblas_int* ldc);
HWYBLAS_API void cgemm_(const char* transa, const char* transb,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const hwyblas_int* k, const float* alpha,
                        const float* a, const hwyblas_int* lda, const float* b,
                        const hwyblas_int* ldb, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void zgemm_(const char* transa, const char* transb,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const hwyblas_int* k, const double* alpha,
                        const double* a, const hwyblas_int* lda,
                        const double* b, const hwyblas_int* ldb,
                        const double* beta, double* c, const hwyblas_int* ldc);

/* C := alpha * A * B + beta * C (side L) or alpha * B * A + beta * C
 * (side R), A symmetric. */
HWYBLAS_API void ssymm_(const char* side, const char* uplo,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* b,
                        const hwyblas_int* ldb, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void dsymm_(const char* side, const char* uplo,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* b,
                        const hwyblas_int* ldb, const double* beta, double* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void csymm_(const char* side, const char* uplo,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* b,
                        const hwyblas_int* ldb, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void zsymm_(const char* side, const char* uplo,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* b,
                        const hwyblas_int* ldb, const double* beta, double* c,
                        const hwyblas_int* ldc);

/* As ?symm_, A Hermitian. */
HWYBLAS_API void chemm_(const char* side, const char* uplo,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* b,
                        const hwyblas_int* ldb, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void zhemm_(const char* side, const char* uplo,
                        const hwyblas_int* m, const hwyblas_int* n,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* b,
                        const hwyblas_int* ldb, const double* beta, double* c,
                        const hwyblas_int* ldc);

/* C := alpha * A * A^T + beta * C (trans N) or alpha * A^T * A + beta * C;
 * only the uplo triangle of C is referenced. */
HWYBLAS_API void ssyrk_(const char* uplo, const char* trans,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void dsyrk_(const char* uplo, const char* trans,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* beta, double* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void csyrk_(const char* uplo, const char* trans,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void zsyrk_(const char* uplo, const char* trans,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* beta, double* c,
                        const hwyblas_int* ldc);

/* As ?syrk_ with A^H instead of A^T; alpha and beta are real. */
HWYBLAS_API void cherk_(const char* uplo, const char* trans,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const float* alpha, const float* a,
                        const hwyblas_int* lda, const float* beta, float* c,
                        const hwyblas_int* ldc);
HWYBLAS_API void zherk_(const char* uplo, const char* trans,
                        const hwyblas_int* n, const hwyblas_int* k,
                        const double* alpha, const double* a,
                        const hwyblas_int* lda, const double* beta, double* c,
                        const hwyblas_int* ldc);

/* C := alpha * A * B^T + alpha * B * A^T + beta * C (trans N) or
 * alpha * A^T * B + alpha * B^T * A + beta * C; only the uplo triangle of C
 * is referenced. */
HWYBLAS_API void ssyr2k_(const char* uplo, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const float* alpha, const float* a,
                         const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* beta, float* c,
                         const hwyblas_int* ldc);
HWYBLAS_API void dsyr2k_(const char* uplo, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const double* alpha, const double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, const double* beta, double* c,
                         const hwyblas_int* ldc);
HWYBLAS_API void csyr2k_(const char* uplo, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const float* alpha, const float* a,
                         const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* beta, float* c,
                         const hwyblas_int* ldc);
HWYBLAS_API void zsyr2k_(const char* uplo, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const double* alpha, const double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, const double* beta, double* c,
                         const hwyblas_int* ldc);

/* As ?syr2k_ with ^H instead of ^T and conj(alpha) for the second term;
 * beta is real. */
HWYBLAS_API void cher2k_(const char* uplo, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const float* alpha, const float* a,
                         const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* beta, float* c,
                         const hwyblas_int* ldc);
HWYBLAS_API void zher2k_(const char* uplo, const char* trans,
                         const hwyblas_int* n, const hwyblas_int* k,
                         const double* alpha, const double* a,
                         const hwyblas_int* lda, const double* b,
                         const hwyblas_int* ldb, const double* beta, double* c,
                         const hwyblas_int* ldc);

/* C := alpha * op(A) * op(B) + beta * C for the uplo triangle of the n x n
 * matrix C only. ?gemmt_ is the same routine under the name used by
 * OpenBLAS and MKL. */
HWYBLAS_API void sgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const hwyblas_int* n,
                          const hwyblas_int* k, const float* alpha,
                          const float* a, const hwyblas_int* lda,
                          const float* b, const hwyblas_int* ldb,
                          const float* beta, float* c, const hwyblas_int* ldc);
HWYBLAS_API void dgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const hwyblas_int* n,
                          const hwyblas_int* k, const double* alpha,
                          const double* a, const hwyblas_int* lda,
                          const double* b, const hwyblas_int* ldb,
                          const double* beta, double* c,
                          const hwyblas_int* ldc);
HWYBLAS_API void cgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const hwyblas_int* n,
                          const hwyblas_int* k, const float* alpha,
                          const float* a, const hwyblas_int* lda,
                          const float* b, const hwyblas_int* ldb,
                          const float* beta, float* c, const hwyblas_int* ldc);
HWYBLAS_API void zgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const hwyblas_int* n,
                          const hwyblas_int* k, const double* alpha,
                          const double* a, const hwyblas_int* lda,
                          const double* b, const hwyblas_int* ldb,
                          const double* beta, double* c,
                          const hwyblas_int* ldc);
HWYBLAS_API void sgemmt_(const char* uplo, const char* transa,
                         const char* transb, const hwyblas_int* n,
                         const hwyblas_int* k, const float* alpha,
                         const float* a, const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* beta, float* c,
                         const hwyblas_int* ldc);
HWYBLAS_API void dgemmt_(const char* uplo, const char* transa,
                         const char* transb, const hwyblas_int* n,
                         const hwyblas_int* k, const double* alpha,
                         const double* a, const hwyblas_int* lda,
                         const double* b, const hwyblas_int* ldb,
                         const double* beta, double* c, const hwyblas_int* ldc);
HWYBLAS_API void cgemmt_(const char* uplo, const char* transa,
                         const char* transb, const hwyblas_int* n,
                         const hwyblas_int* k, const float* alpha,
                         const float* a, const hwyblas_int* lda, const float* b,
                         const hwyblas_int* ldb, const float* beta, float* c,
                         const hwyblas_int* ldc);
HWYBLAS_API void zgemmt_(const char* uplo, const char* transa,
                         const char* transb, const hwyblas_int* n,
                         const hwyblas_int* k, const double* alpha,
                         const double* a, const hwyblas_int* lda,
                         const double* b, const hwyblas_int* ldb,
                         const double* beta, double* c, const hwyblas_int* ldc);

/* B := alpha * op(A) * B (side L) or alpha * B * op(A) (side R),
 * A triangular. */
HWYBLAS_API void strmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const float* alpha,
                        const float* a, const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb);
HWYBLAS_API void dtrmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const double* alpha,
                        const double* a, const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb);
HWYBLAS_API void ctrmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const float* alpha,
                        const float* a, const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb);
HWYBLAS_API void ztrmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const double* alpha,
                        const double* a, const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb);

/* Solves op(A) * X = alpha * B (side L) or X * op(A) = alpha * B (side R)
 * for X, A triangular; X overwrites B. */
HWYBLAS_API void strsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const float* alpha,
                        const float* a, const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb);
HWYBLAS_API void dtrsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const double* alpha,
                        const double* a, const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb);
HWYBLAS_API void ctrsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const float* alpha,
                        const float* a, const hwyblas_int* lda, float* b,
                        const hwyblas_int* ldb);
HWYBLAS_API void ztrsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const hwyblas_int* m,
                        const hwyblas_int* n, const double* alpha,
                        const double* a, const hwyblas_int* lda, double* b,
                        const hwyblas_int* ldb);

/* ------------------------------------------------------------------------ */
/* Error handling                                                            */

/* Called with the routine name and the 1-based position of the first invalid
 * argument. The default implementation prints a message to stderr and
 * returns; the offending routine then returns without doing anything. */
HWYBLAS_API void xerbla_(const char* srname, const hwyblas_int* info,
                         size_t srname_len);

/* ------------------------------------------------------------------------ */
/* Introspection                                                             */

/* Name of the SIMD target selected at runtime, e.g. "AVX2". */
HWYBLAS_API const char* hwyblas_target_name(void);

/* Bitmask of Highway targets (HWY_AVX2 etc.) that are compiled into the
 * library and supported by this CPU. */
HWYBLAS_API int64_t hwyblas_available_targets(void);

/* Restricts runtime dispatch to `target` (one bit of
 * hwyblas_available_targets()), e.g. to test or benchmark a specific code
 * path. 0 restores automatic selection. Not thread-safe. */
HWYBLAS_API void hwyblas_force_target(int64_t target);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HWYBLAS_HWYBLAS_H_ */
