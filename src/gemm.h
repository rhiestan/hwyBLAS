// Target-independent description of a GEMM problem.
#ifndef HWYBLAS_SRC_GEMM_H_
#define HWYBLAS_SRC_GEMM_H_

#include <stddef.h>

namespace hwyblas {

// C := alpha * op(A) * op(B) + beta * C with operands described by strides.
//
// op(A) is m x k with element (i, p) at a[i * a_rs + p * a_cs]; op(B) is
// k x n with element (p, j) at b[p * b_rs + j * b_cs]. Transposition is thus
// expressed by swapping strides and conjugation by the conj_* flags. C is
// column-major with leading dimension ldc.
template <typename T>
struct GemmProblem {
  size_t m, n, k;
  T alpha;
  const T* a;
  ptrdiff_t a_rs, a_cs;
  bool conj_a;
  const T* b;
  ptrdiff_t b_rs, b_cs;
  bool conj_b;
  T beta;
  T* c;
  size_t ldc;
};

}  // namespace hwyblas

#endif  // HWYBLAS_SRC_GEMM_H_
