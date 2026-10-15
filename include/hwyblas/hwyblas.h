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
