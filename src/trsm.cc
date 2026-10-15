// TRSM: solves op(A) X = alpha B or X op(A) = alpha B, overwriting B with X.
//
// Recursive algorithm: the triangle is split in two halves,
//
//   T = [T11  0 ]  (lower)        T X = B:  X1 = T11 \ B1
//       [T21 T22]                           B2 -= T21 X1      (GEMM)
//                                           X2 = T22 \ B2
//
// and similarly for upper triangles and for the right side. Almost all work
// ends up in large GEMM updates; triangles of at most kTrsmBase rows are
// solved by the substitution kernel in trsm-inl.h.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/trsm.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/trsm-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

#define HWYBLAS_TRSM_BASE(SUFFIX, T)                                \
  void TrsmBase##SUFFIX(bool left, const Operand<T>& tri, size_t r, \
                        size_t other, T* b, size_t ldb) {           \
    TrsmBase(left, tri, r, other, b, ldb);                          \
  }
HWYBLAS_TRSM_BASE(F32, float)
HWYBLAS_TRSM_BASE(F64, double)
HWYBLAS_TRSM_BASE(C32, c32)
HWYBLAS_TRSM_BASE(C64, c64)
#undef HWYBLAS_TRSM_BASE

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(TrsmBase);

// Returns the sub-block of operand x starting at (i0, j0); `structure`
// replaces the structure of x (off-diagonal blocks are general).
template <typename T>
Operand<T> SubOperand(const Operand<T>& x, size_t i0, size_t j0,
                      Structure structure) {
  Operand<T> s = x;
  s.data +=
      static_cast<ptrdiff_t>(i0) * x.rs + static_cast<ptrdiff_t>(j0) * x.cs;
  s.structure = structure;
  return s;
}

// A block of B (column-major, leading dimension ldb) as a GEMM operand.
template <typename T>
Operand<T> ColumnMajor(const T* block, size_t ldb) {
  Operand<T> o;
  o.data = block;
  o.rs = 1;
  o.cs = static_cast<ptrdiff_t>(ldb);
  return o;
}

// B -= op1 * op2 for an m x n block of B (column-major, leading dimension
// ldb), with op1 m x k and op2 k x n.
template <typename T>
void SubtractProduct(size_t m, size_t n, size_t k, const Operand<T>& op1,
                     const Operand<T>& op2, T* b, size_t ldb) {
  GemmProblem<T> g;
  g.m = m;
  g.n = n;
  g.k = k;
  g.alpha = T(-1);
  g.a = op1;
  g.b = op2;
  g.beta = T(1);
  g.c = b;
  g.ldc = ldb;
  RunGemm(g);
}

// Solves T X = B (left, B is r x other) or X T = B (right, B is other x r)
// in place; T is the r x r triangular operand `tri`.
template <typename T>
void TrsmRecursive(bool left, const Operand<T>& tri, size_t r, size_t other,
                   T* b, size_t ldb) {
  if (r <= kTrsmBase) {
    HWYBLAS_DISPATCH_BY_TYPE(T, TrsmBase, left, tri, r, other, b, ldb);
    return;
  }
  // Split near the middle, at a multiple of the base size.
  const size_t r1 = HWY_MAX(kTrsmBase, (r / 2) / kTrsmBase * kTrsmBase);
  const size_t r2 = r - r1;
  const Operand<T> t11 = tri;
  const Operand<T> t22 = SubOperand(tri, r1, r1, Structure::kTriangular);
  // The off-diagonal block: T21 (lower) or T12 (upper).
  const Operand<T> t_off = tri.lower
                               ? SubOperand(tri, r1, 0, Structure::kGeneral)
                               : SubOperand(tri, 0, r1, Structure::kGeneral);

  if (left) {
    // B1 = rows [0, r1) of B, B2 = rows [r1, r).
    T* b1 = b;
    T* b2 = b + r1;
    if (tri.lower) {
      TrsmRecursive(left, t11, r1, other, b1, ldb);
      SubtractProduct(r2, other, r1, t_off, ColumnMajor(b1, ldb), b2,
                      ldb);  // T21 X1
      TrsmRecursive(left, t22, r2, other, b2, ldb);
    } else {
      TrsmRecursive(left, t22, r2, other, b2, ldb);
      SubtractProduct(r1, other, r2, t_off, ColumnMajor(b2, ldb), b1,
                      ldb);  // T12 X2
      TrsmRecursive(left, t11, r1, other, b1, ldb);
    }
  } else {
    // B1 = columns [0, r1) of B, B2 = columns [r1, r).
    T* b1 = b;
    T* b2 = b + r1 * ldb;
    if (tri.lower) {
      TrsmRecursive(left, t22, r2, other, b2, ldb);
      SubtractProduct(other, r1, r2, ColumnMajor(b2, ldb), t_off, b1,
                      ldb);  // X2 T21
      TrsmRecursive(left, t11, r1, other, b1, ldb);
    } else {
      TrsmRecursive(left, t11, r1, other, b1, ldb);
      SubtractProduct(other, r2, r1, ColumnMajor(b1, ldb), t_off, b2,
                      ldb);  // X1 T12
      TrsmRecursive(left, t22, r2, other, b2, ldb);
    }
  }
}

template <typename T>
void Trsm(const char* name, const char* side_ptr, const char* uplo_ptr,
          const char* transa_ptr, const char* diag_ptr, const blasint* m_ptr,
          const blasint* n_ptr, const T* alpha_ptr, const T* a,
          const blasint* lda_ptr, T* b, const blasint* ldb_ptr) {
  const blasint m = *m_ptr, n = *n_ptr, lda = *lda_ptr, ldb = *ldb_ptr;
  bool left, lower, unit;
  Op op;
  const bool valid_side = DecodeSide(*side_ptr, &left);
  const bool valid_uplo = DecodeUplo(*uplo_ptr, &lower);
  const bool valid_trans = DecodeOp(*transa_ptr, &op);
  const bool valid_diag = DecodeDiag(*diag_ptr, &unit);
  const blasint rows_a = left ? m : n;

  blasint info = 0;
  if (!valid_side) {
    info = 1;
  } else if (!valid_uplo) {
    info = 2;
  } else if (!valid_trans) {
    info = 3;
  } else if (!valid_diag) {
    info = 4;
  } else if (m < 0) {
    info = 5;
  } else if (n < 0) {
    info = 6;
  } else if (lda < HWY_MAX(1, rows_a)) {
    info = 9;
  } else if (ldb < HWY_MAX(1, m)) {
    info = 11;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr;
  if (m == 0 || n == 0) return;
  // B := alpha * B first; then solve with alpha = 1.
  if (!IsOne(alpha)) ScaleMatrix(m, n, alpha, b, ldb);
  if (IsZero(alpha)) return;

  const Operand<T> tri =
      StructuredOperand(Structure::kTriangular, lower, op, a, lda, unit);
  TrsmRecursive(left, tri, left ? m : n, left ? n : m, b, ldb);
}

}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

HWYBLAS_API void strsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const float* alpha, const float* a, const blasint* lda,
                        float* b, const blasint* ldb) {
  hwyblas::Trsm("STRSM", side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void dtrsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, double* b, const blasint* ldb) {
  hwyblas::Trsm("DTRSM", side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void ctrsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const float* alpha, const float* a, const blasint* lda,
                        float* b, const blasint* ldb) {
  hwyblas::Trsm("CTRSM", side, uplo, transa, diag, m, n, C32(alpha), C32(a),
                lda, C32(b), ldb);
}
HWYBLAS_API void ztrsm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, double* b, const blasint* ldb) {
  hwyblas::Trsm("ZTRSM", side, uplo, transa, diag, m, n, C64(alpha), C64(a),
                lda, C64(b), ldb);
}

}  // extern "C"
#endif  // HWY_ONCE
