// BLAS calls of the LAPACK routines: overloads for the four precisions with
// the argument order of the Fortran calls they replace, e.g.
//   CALL DGEMM('N', 'N', M, N, K, -ONE, A(I, J), LDA, B, LDB, ONE, C, LDC)
// becomes
//   Gemm('N', 'N', m, n, k, -one, A.ptr(i, j), lda, b, ldb, one, c, ldc);
// Scalars are passed by value. The calls go through hwyBLAS's Fortran-style
// routines (including their argument checks, which LAPACK relies on not
// failing).
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

// x := alpha * x.
template <typename T>
void Scal(blasint n, T alpha, T* x, blasint incx) {
  using blas_detail::Raw;
  HWYBLAS_LAPACK_BLAS_CALL(T, sscal_, dscal_, cscal_, zscal_, &n, Raw(&alpha),
                           Raw(x), &incx);
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
