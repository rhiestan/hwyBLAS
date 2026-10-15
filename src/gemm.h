// Target-independent description of a GEMM problem, shared by all level 3
// routines (GEMM, SYMM/HEMM, SYRK/HERK, TRMM, and the updates inside TRSM).
#ifndef HWYBLAS_SRC_GEMM_H_
#define HWYBLAS_SRC_GEMM_H_

#include <stddef.h>

#include <complex>

namespace hwyblas {

// How the elements of an operand are obtained from memory.
enum class Structure {
  kGeneral,     // every element is stored
  kSymmetric,   // only one triangle is stored; X(i, p) = X(p, i)
  kHermitian,   // only one triangle is stored; X(i, p) = conj(X(p, i)), and
                // the imaginary parts of the diagonal are taken as zero
  kTriangular,  // elements outside the triangle are zero
};

// An operand op(X) of a matrix product, as seen by the GEMM driver.
//
// Element (i, p) of op(X) is stored at data[i * rs + p * cs]; transposition
// is expressed by swapping the strides. If conj, the stored values are
// conjugated. For structured operands, `lower` tells which triangle of op(X)
// is stored (or non-zero), and unit_diag (triangular only) that the diagonal
// is taken as all ones without being read.
template <typename T>
struct Operand {
  const T* data = nullptr;
  ptrdiff_t rs = 1, cs = 1;
  bool conj = false;
  Structure structure = Structure::kGeneral;
  bool lower = false;
  bool unit_diag = false;
};

// Which part of C is updated.
enum class Part { kAll, kLower, kUpper };

// C := alpha * op(A) * op(B) + beta * C, where op(A) is m x k, op(B) is
// k x n and C is column-major with leading dimension ldc.
//
// If `part` is kLower or kUpper, only that triangle of C (which must be
// square) is read and written. If additionally `real_diagonal`, the imaginary
// parts of the diagonal of C are set to zero (HERK semantics).
template <typename T>
struct GemmProblem {
  size_t m = 0, n = 0, k = 0;
  T alpha = T(1);
  Operand<T> a, b;
  T beta = T(0);
  T* c = nullptr;
  size_t ldc = 1;
  Part part = Part::kAll;
  bool real_diagonal = false;
};

// Runs a GEMM problem on the best available SIMD target. Requires
// m, n, k > 0 and alpha != 0; the caller handles trivial cases.
template <typename T>
void RunGemm(const GemmProblem<T>& g);

// C := beta * C for an m x n column-major matrix (or one triangle of it).
// beta == 0 sets C to zero without reading it, so NaN/Inf in C do not
// propagate. With real_diagonal, imaginary parts of the diagonal are zeroed.
template <typename T>
void ScaleMatrix(size_t m, size_t n, T beta, T* c, size_t ldc,
                 Part part = Part::kAll, bool real_diagonal = false);

}  // namespace hwyblas

#endif  // HWYBLAS_SRC_GEMM_H_
