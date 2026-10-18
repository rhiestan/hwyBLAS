// BLAS calls of the LAPACK routines: overloads for the four precisions with
// the argument order of the Fortran calls they replace, e.g.
//   CALL DGEMM('N', 'N', M, N, K, -ONE, A(I, J), LDA, B, LDB, ONE, C, LDC)
// becomes
//   Gemm('N', 'N', m, n, k, -one, A.ptr(i, j), lda, b, ldb, one, c, ldc);
// Scalars are passed by value. Like the BLAS, the wrappers take 'C'
// (conjugate transpose) also for real T, where it means 'T', so that one
// template serves both the real and the complex LAPACK routine. The calls go
// through hwyBLAS's Fortran-style routines (including their argument checks,
// which LAPACK relies on not failing).
#ifndef HWYBLAS_SRC_LAPACK_BLAS_H_
#define HWYBLAS_SRC_LAPACK_BLAS_H_

#include "src/common.h"

namespace hwyblas {
namespace lapack {
namespace blas_detail {

// The pointer types of hwyBLAS's Fortran-style routines: float or double
// pairs for complex T.
template <typename T>
auto Raw(T* p) {
  if constexpr (std::is_const_v<T>) {
    return reinterpret_cast<const RealT<std::remove_const_t<T>>*>(p);
  } else {
    return reinterpret_cast<RealT<T>*>(p);
  }
}

}  // namespace blas_detail

// Selects the routine of T's precision among the four given ones and calls
// it with `args`.
#define HWYBLAS_LAPACK_BLAS_CALL(T, s, d, c, z, ...)  \
  do {                                                \
    if constexpr (std::is_same_v<T, float>) {         \
      s(__VA_ARGS__);                                 \
    } else if constexpr (std::is_same_v<T, double>) { \
      d(__VA_ARGS__);                                 \
    } else if constexpr (std::is_same_v<T, c32>) {    \
      c(__VA_ARGS__);                                 \
    } else {                                          \
      z(__VA_ARGS__);                                 \
    }                                                 \
  } while (false)

// C := alpha * op(A) * op(B) + beta * C.
template <typename T>
void Gemm(char transa, char transb, blasint m, blasint n, blasint k, T alpha,
          const T* a, blasint lda, const T* b, blasint ldb, T beta, T* c,
          blasint ldc) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sgemm_, dgemm_, cgemm_, zgemm_, &transa, &transb,
                           &m, &n, &k, Raw(&alpha), Raw(a), &lda, Raw(b), &ldb,
                           Raw(&beta), Raw(c), &ldc);
}

// Solves op(A) * X = alpha * B or X * op(A) = alpha * B; X overwrites B.
template <typename T>
void Trsm(char side, char uplo, char transa, char diag, blasint m, blasint n,
          T alpha, const T* a, blasint lda, T* b, blasint ldb) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, strsm_, dtrsm_, ctrsm_, ztrsm_, &side, &uplo,
                           &transa, &diag, &m, &n, Raw(&alpha), Raw(a), &lda,
                           Raw(b), &ldb);
}

// C := alpha * op(A) * op(B) + beta * C, C triangular (uplo).
// xSYRK for real T and xHERK for complex T: C := alpha * A * A**H + beta * C
// (trans = 'N') or alpha * A**H * A + beta * C (trans = 'C', or 'T' for real
// T), with real alpha and beta.
template <typename T>
void Herk(char uplo, char trans, blasint n, blasint k, RealT<T> alpha,
          const T* a, blasint lda, RealT<T> beta, T* c, blasint ldc) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, ssyrk_, dsyrk_, cherk_, zherk_, &uplo, &trans, &n,
                           &k, &alpha, Raw(a), &lda, &beta, Raw(c), &ldc);
}

// B := alpha * op(A) * B or alpha * B * op(A), A triangular.
template <typename T>
void Trmm(char side, char uplo, char transa, char diag, blasint m, blasint n,
          T alpha, const T* a, blasint lda, T* b, blasint ldb) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, strmm_, dtrmm_, ctrmm_, ztrmm_, &side, &uplo,
                           &transa, &diag, &m, &n, Raw(&alpha), Raw(a), &lda,
                           Raw(b), &ldb);
}

// y := alpha * op(A) * x + beta * y.
template <typename T>
void Gemv(char trans, blasint m, blasint n, T alpha, const T* a, blasint lda,
          const T* x, blasint incx, T beta, T* y, blasint incy) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sgemv_, dgemv_, cgemv_, zgemv_, &trans, &m, &n,
                           Raw(&alpha), Raw(a), &lda, Raw(x), &incx, Raw(&beta),
                           Raw(y), &incy);
}

// A := alpha * x * y**H + A: xGER for real T, xGERC for complex T.
template <typename T>
void Gerc(blasint m, blasint n, T alpha, const T* x, blasint incx, const T* y,
          blasint incy, T* a, blasint lda) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sger_, dger_, cgerc_, zgerc_, &m, &n, Raw(&alpha),
                           Raw(x), &incx, Raw(y), &incy, Raw(a), &lda);
}

// y := alpha * A * x + beta * y for symmetric (xSYMV, real T) or Hermitian
// (xHEMV, complex T) A, triangle uplo.
template <typename T>
void Hemv(char uplo, blasint n, T alpha, const T* a, blasint lda, const T* x,
          blasint incx, T beta, T* y, blasint incy) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, ssymv_, dsymv_, chemv_, zhemv_, &uplo, &n,
                           Raw(&alpha), Raw(a), &lda, Raw(x), &incx, Raw(&beta),
                           Raw(y), &incy);
}

// A := alpha * x * y**H + conj(alpha) * y * x**H + A, triangle uplo: xSYR2
// for real T, xHER2 for complex T.
template <typename T>
void Her2(char uplo, blasint n, T alpha, const T* x, blasint incx, const T* y,
          blasint incy, T* a, blasint lda) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, ssyr2_, dsyr2_, cher2_, zher2_, &uplo, &n,
                           Raw(&alpha), Raw(x), &incx, Raw(y), &incy, Raw(a),
                           &lda);
}

// C := alpha * A * B**H + conj(alpha) * B * A**H + beta * C (trans = 'N'),
// or with A**H * B etc., with real beta: xSYR2K for real T, xHER2K for
// complex T.
template <typename T>
void Her2k(char uplo, char trans, blasint n, blasint k, T alpha, const T* a,
           blasint lda, const T* b, blasint ldb, RealT<T> beta, T* c,
           blasint ldc) {
  using blas_detail::Raw;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_LAPACK_BLAS_CALL(T, ssyr2k_, dsyr2k_, cher2k_, zher2k_, &uplo,
                             &trans, &n, &k, Raw(&alpha), Raw(a), &lda, Raw(b),
                             &ldb, &beta, Raw(c), &ldc);
  } else {
    T beta_t = beta;
    HWYBLAS_LAPACK_BLAS_CALL(T, ssyr2k_, dsyr2k_, cher2k_, zher2k_, &uplo,
                             &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta_t,
                             c, &ldc);
  }
}

// x**H * y: xDOT for real T, xDOTC for complex T.
template <typename T>
T Dotc(blasint n, const T* x, blasint incx, const T* y, blasint incy) {
  using blas_detail::Raw;
  if constexpr (std::is_same_v<T, float>) {
    return sdot_(&n, x, &incx, y, &incy);
  } else if constexpr (std::is_same_v<T, double>) {
    return ddot_(&n, x, &incx, y, &incy);
  } else if constexpr (std::is_same_v<T, c32>) {
    const hwyblas_complex_float r = cdotc_(&n, Raw(x), &incx, Raw(y), &incy);
    return T(r.real, r.imag);
  } else {
    const hwyblas_complex_double r = zdotc_(&n, Raw(x), &incx, Raw(y), &incy);
    return T(r.real, r.imag);
  }
}

// y := alpha * x + y.
template <typename T>
void Axpy(blasint n, T alpha, const T* x, blasint incx, T* y, blasint incy) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, saxpy_, daxpy_, caxpy_, zaxpy_, &n, Raw(&alpha),
                           Raw(x), &incx, Raw(y), &incy);
}

// y := x.
template <typename T>
void Copy(blasint n, const T* x, blasint incx, T* y, blasint incy) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, scopy_, dcopy_, ccopy_, zcopy_, &n, Raw(x), &incx,
                           Raw(y), &incy);
}

// x <-> y.
template <typename T>
void Swap(blasint n, T* x, blasint incx, T* y, blasint incy) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sswap_, dswap_, cswap_, zswap_, &n, Raw(x), &incx,
                           Raw(y), &incy);
}

// x := alpha * x.
template <typename T>
void Scal(blasint n, T alpha, T* x, blasint incx) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sscal_, dscal_, cscal_, zscal_, &n, Raw(&alpha),
                           Raw(x), &incx);
}

// x := alpha * x with real alpha: xSCAL for real T, CSSCAL / ZDSCAL for
// complex T.
template <typename T>
void Rscal(blasint n, RealT<T> alpha, T* x, blasint incx) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sscal_, dscal_, csscal_, zdscal_, &n, &alpha,
                           Raw(x), &incx);
}

// Euclidean norm of x: xNRM2, SCNRM2 or DZNRM2.
template <typename T>
RealT<T> Nrm2(blasint n, const T* x, blasint incx) {
  using blas_detail::Raw;
  if constexpr (std::is_same_v<T, float>) {
    return snrm2_(&n, Raw(x), &incx);
  } else if constexpr (std::is_same_v<T, double>) {
    return dnrm2_(&n, Raw(x), &incx);
  } else if constexpr (std::is_same_v<T, c32>) {
    return scnrm2_(&n, Raw(x), &incx);
  } else {
    return dznrm2_(&n, Raw(x), &incx);
  }
}

// IxAMAX: 1-based index of the first element of largest |x_i| (|re| + |im|
// for complex x), or 0 if n < 1.
template <typename T>
blasint Iamax(blasint n, const T* x, blasint incx) {
  using blas_detail::Raw;
  if constexpr (std::is_same_v<T, float>) {
    return isamax_(&n, Raw(x), &incx);
  } else if constexpr (std::is_same_v<T, double>) {
    return idamax_(&n, Raw(x), &incx);
  } else if constexpr (std::is_same_v<T, c32>) {
    return icamax_(&n, Raw(x), &incx);
  } else {
    return izamax_(&n, Raw(x), &incx);
  }
}

#undef HWYBLAS_LAPACK_BLAS_CALL

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_BLAS_H_
