// Level 3 routines that map directly onto the GEMM driver:
//
//   SYMM, HEMM  C := alpha * A * B + beta * C   (or B * A), A symmetric or
//               Hermitian, read from one triangle while packing.
//   SYRK, HERK  C := alpha * A * A^T + beta * C (or A^H, or A^T * A), only
//               one triangle of C is computed.
//   SYR2K, HER2K  C := alpha * A * B^T + alpha * B * A^T + beta * C (or with
//               ^H and conj(alpha) for the second term), one triangle of C.
//   GEMMTR      C := alpha * op(A) * op(B) + beta * C, one triangle of C.
//   TRMM        B := alpha * op(A) * B          (or B * op(A)), A triangular.
//
// Each routine validates its arguments like the reference BLAS, handles the
// trivial cases and describes the remaining work as a GemmProblem.
#include <string.h>

#include "src/api.h"
#include "src/common.h"
#include "src/gemm.h"

namespace hwyblas {
namespace {

// ---------------------------------------------------------------------------
// SYMM / HEMM

template <typename T>
void Symm(const char* name, Structure structure, const char* side_ptr,
          const char* uplo_ptr, const blasint* m_ptr, const blasint* n_ptr,
          const T* alpha_ptr, const T* a, const blasint* lda_ptr, const T* b,
          const blasint* ldb_ptr, const T* beta_ptr, T* c,
          const blasint* ldc_ptr) {
  const blasint m = *m_ptr, n = *n_ptr;
  const blasint lda = *lda_ptr, ldb = *ldb_ptr, ldc = *ldc_ptr;
  bool left, lower;
  const bool valid_side = DecodeSide(*side_ptr, &left);
  const bool valid_uplo = DecodeUplo(*uplo_ptr, &lower);
  const blasint rows_a = left ? m : n;  // A is rows_a x rows_a

  blasint info = 0;
  if (!valid_side) {
    info = 1;
  } else if (!valid_uplo) {
    info = 2;
  } else if (m < 0) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (lda < HWY_MAX(1, rows_a)) {
    info = 7;
  } else if (ldb < HWY_MAX(1, m)) {
    info = 9;
  } else if (ldc < HWY_MAX(1, m)) {
    info = 12;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr, beta = *beta_ptr;
  if (m == 0 || n == 0 || (IsZero(alpha) && IsOne(beta))) return;
  if (IsZero(alpha)) return ScaleMatrix(m, n, beta, c, ldc);

  const Operand<T> sym =
      StructuredOperand(structure, lower, Op::kNoTrans, a, lda);
  const Operand<T> general = GeneralOperand(Op::kNoTrans, b, ldb);
  GemmProblem<T> g;
  g.m = m;
  g.n = n;
  g.k = rows_a;
  g.alpha = alpha;
  g.beta = beta;
  g.a = left ? sym : general;
  g.b = left ? general : sym;
  g.c = c;
  g.ldc = ldc;
  RunGemm(g);
}

// ---------------------------------------------------------------------------
// SYRK / HERK
//
// C := alpha * op(A) * op(A)' + beta * C where ' is ^T (SYRK) or ^H (HERK);
// for trans = 'N' op(A) = A (n x k), otherwise op(A) = A' (A is k x n).
// For HERK, alpha and beta are real (S is the type of alpha and beta).

template <typename T, typename S>
void Syrk(const char* name, bool hermitian, const char* uplo_ptr,
          const char* trans_ptr, const blasint* n_ptr, const blasint* k_ptr,
          const S* alpha_ptr, const T* a, const blasint* lda_ptr,
          const S* beta_ptr, T* c, const blasint* ldc_ptr) {
  const blasint n = *n_ptr, k = *k_ptr, lda = *lda_ptr, ldc = *ldc_ptr;
  bool lower;
  const bool valid_uplo = DecodeUplo(*uplo_ptr, &lower);
  Op op;
  bool valid_trans = DecodeOp(*trans_ptr, &op);
  // Allowed: N and C for HERK; N and T for complex SYRK; N, T, C for real.
  if (IsComplex<T>::value) {
    valid_trans &= (op != (hermitian ? Op::kTrans : Op::kConjTrans));
  }
  const bool no_trans = (op == Op::kNoTrans);
  const blasint rows_a = no_trans ? n : k;

  blasint info = 0;
  if (!valid_uplo) {
    info = 1;
  } else if (!valid_trans) {
    info = 2;
  } else if (n < 0) {
    info = 3;
  } else if (k < 0) {
    info = 4;
  } else if (lda < HWY_MAX(1, rows_a)) {
    info = 7;
  } else if (ldc < HWY_MAX(1, n)) {
    info = 10;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = T(*alpha_ptr), beta = T(*beta_ptr);
  const Part part = lower ? Part::kLower : Part::kUpper;
  if (n == 0 || ((IsZero(alpha) || k == 0) && IsOne(beta))) return;
  if (IsZero(alpha) || k == 0) {
    return ScaleMatrix(n, n, beta, c, ldc, part, hermitian);
  }

  // The transposition of the second factor (T or C) for the first one.
  const Op second = hermitian ? Op::kConjTrans : Op::kTrans;
  GemmProblem<T> g;
  g.m = n;
  g.n = n;
  g.k = k;
  g.alpha = alpha;
  g.beta = beta;
  g.a = GeneralOperand(no_trans ? Op::kNoTrans : second, a, lda);
  g.b = GeneralOperand(no_trans ? second : Op::kNoTrans, a, lda);
  g.c = c;
  g.ldc = ldc;
  g.part = part;
  g.real_diagonal = hermitian;
  RunGemm(g);
}

// ---------------------------------------------------------------------------
// SYR2K / HER2K
//
// C := alpha * op(A) * op(B)' + alpha2 * op(B) * op(A)' + beta * C, where ' is
// ^T and alpha2 = alpha (SYR2K), or ' is ^H and alpha2 = conj(alpha) (HER2K,
// beta real: S is the type of beta). For trans = 'N' op(X) = X (n x k),
// otherwise op(X) = X' (X is k x n).
//
// Computed as two GEMM passes over the triangle; the second adds to the
// result of the first. For HER2K both passes drop the imaginary parts of the
// diagonal, which cancel in the exact sum.

template <typename T, typename S>
void Syr2k(const char* name, bool hermitian, const char* uplo_ptr,
           const char* trans_ptr, const blasint* n_ptr, const blasint* k_ptr,
           const T* alpha_ptr, const T* a, const blasint* lda_ptr, const T* b,
           const blasint* ldb_ptr, const S* beta_ptr, T* c,
           const blasint* ldc_ptr) {
  const blasint n = *n_ptr, k = *k_ptr;
  const blasint lda = *lda_ptr, ldb = *ldb_ptr, ldc = *ldc_ptr;
  bool lower;
  const bool valid_uplo = DecodeUplo(*uplo_ptr, &lower);
  Op op;
  bool valid_trans = DecodeOp(*trans_ptr, &op);
  // Allowed: N and C for HER2K; N and T for complex SYR2K; N, T, C for real.
  if (IsComplex<T>::value) {
    valid_trans &= (op != (hermitian ? Op::kTrans : Op::kConjTrans));
  }
  const bool no_trans = (op == Op::kNoTrans);
  const blasint rows = no_trans ? n : k;  // rows of A and B

  blasint info = 0;
  if (!valid_uplo) {
    info = 1;
  } else if (!valid_trans) {
    info = 2;
  } else if (n < 0) {
    info = 3;
  } else if (k < 0) {
    info = 4;
  } else if (lda < HWY_MAX(1, rows)) {
    info = 7;
  } else if (ldb < HWY_MAX(1, rows)) {
    info = 9;
  } else if (ldc < HWY_MAX(1, n)) {
    info = 12;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr, beta = T(*beta_ptr);
  const Part part = lower ? Part::kLower : Part::kUpper;
  if (n == 0 || ((IsZero(alpha) || k == 0) && IsOne(beta))) return;
  if (IsZero(alpha) || k == 0) {
    return ScaleMatrix(n, n, beta, c, ldc, part, hermitian);
  }

  const Op second = hermitian ? Op::kConjTrans : Op::kTrans;
  const Op op_first = no_trans ? Op::kNoTrans : second;
  const Op op_second = no_trans ? second : Op::kNoTrans;
  GemmProblem<T> g;
  g.m = n;
  g.n = n;
  g.k = k;
  g.c = c;
  g.ldc = ldc;
  g.part = part;
  g.real_diagonal = hermitian;
  // alpha * op(A) * op(B)' + beta * C
  g.alpha = alpha;
  g.beta = beta;
  g.a = GeneralOperand(op_first, a, lda);
  g.b = GeneralOperand(op_second, b, ldb);
  RunGemm(g);
  // + alpha2 * op(B) * op(A)'
  g.alpha = hermitian ? Conj(alpha) : alpha;
  g.beta = T(1);
  g.a = GeneralOperand(op_first, b, ldb);
  g.b = GeneralOperand(op_second, a, lda);
  RunGemm(g);
}

// ---------------------------------------------------------------------------
// GEMMTR: C := alpha * op(A) * op(B) + beta * C, where op(A) is n x k and
// op(B) is k x n, updating only the uplo triangle of C.

template <typename T>
void Gemmtr(const char* name, const char* uplo_ptr, const char* transa_ptr,
            const char* transb_ptr, const blasint* n_ptr, const blasint* k_ptr,
            const T* alpha_ptr, const T* a, const blasint* lda_ptr, const T* b,
            const blasint* ldb_ptr, const T* beta_ptr, T* c,
            const blasint* ldc_ptr) {
  const blasint n = *n_ptr, k = *k_ptr;
  const blasint lda = *lda_ptr, ldb = *ldb_ptr, ldc = *ldc_ptr;
  bool lower;
  Op op_a, op_b;
  const bool valid_uplo = DecodeUplo(*uplo_ptr, &lower);
  const bool valid_a = DecodeOp(*transa_ptr, &op_a);
  const bool valid_b = DecodeOp(*transb_ptr, &op_b);
  const blasint rows_a = (op_a == Op::kNoTrans) ? n : k;
  const blasint rows_b = (op_b == Op::kNoTrans) ? k : n;

  blasint info = 0;
  if (!valid_uplo) {
    info = 1;
  } else if (!valid_a) {
    info = 2;
  } else if (!valid_b) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (k < 0) {
    info = 5;
  } else if (lda < HWY_MAX(1, rows_a)) {
    info = 8;
  } else if (ldb < HWY_MAX(1, rows_b)) {
    info = 10;
  } else if (ldc < HWY_MAX(1, n)) {
    info = 13;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr, beta = *beta_ptr;
  const Part part = lower ? Part::kLower : Part::kUpper;
  if (n == 0 || ((IsZero(alpha) || k == 0) && IsOne(beta))) return;
  if (IsZero(alpha) || k == 0) return ScaleMatrix(n, n, beta, c, ldc, part);

  GemmProblem<T> g;
  g.m = n;
  g.n = n;
  g.k = k;
  g.alpha = alpha;
  g.beta = beta;
  g.a = GeneralOperand(op_a, a, lda);
  g.b = GeneralOperand(op_b, b, ldb);
  g.c = c;
  g.ldc = ldc;
  g.part = part;
  RunGemm(g);
}

// ---------------------------------------------------------------------------
// TRMM

template <typename T>
void Trmm(const char* name, const char* side_ptr, const char* uplo_ptr,
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
  if (IsZero(alpha)) return ScaleMatrix(m, n, T(0), b, ldb);

  // B is both input and output: compute from a copy of the input.
  T* copy = static_cast<T*>(
      ScratchBuffer(kScratchTemp, static_cast<size_t>(m) * n * sizeof(T)));
  for (blasint j = 0; j < n; ++j) {
    memcpy(copy + static_cast<size_t>(j) * m, b + static_cast<size_t>(j) * ldb,
           m * sizeof(T));
  }

  const Operand<T> tri =
      StructuredOperand(Structure::kTriangular, lower, op, a, lda, unit);
  const Operand<T> general =
      GeneralOperand(Op::kNoTrans, static_cast<const T*>(copy), m);
  GemmProblem<T> g;
  g.m = m;
  g.n = n;
  g.k = rows_a;
  g.alpha = alpha;
  g.beta = T(0);
  g.a = left ? tri : general;
  g.b = left ? general : tri;
  g.c = b;
  g.ldc = ldb;
  RunGemm(g);
}

}  // namespace
}  // namespace hwyblas

// ---------------------------------------------------------------------------
// Fortran-style entry points

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;
using hwyblas::Structure;

extern "C" {

// SYMM

HWYBLAS_API void ssymm_(const char* side, const char* uplo, const blasint* m,
                        const blasint* n, const float* alpha, const float* a,
                        const blasint* lda, const float* b, const blasint* ldb,
                        const float* beta, float* c, const blasint* ldc) {
  hwyblas::Symm("SSYMM", Structure::kSymmetric, side, uplo, m, n, alpha, a, lda,
                b, ldb, beta, c, ldc);
}
HWYBLAS_API void dsymm_(const char* side, const char* uplo, const blasint* m,
                        const blasint* n, const double* alpha, const double* a,
                        const blasint* lda, const double* b, const blasint* ldb,
                        const double* beta, double* c, const blasint* ldc) {
  hwyblas::Symm("DSYMM", Structure::kSymmetric, side, uplo, m, n, alpha, a, lda,
                b, ldb, beta, c, ldc);
}
HWYBLAS_API void csymm_(const char* side, const char* uplo, const blasint* m,
                        const blasint* n, const float* alpha, const float* a,
                        const blasint* lda, const float* b, const blasint* ldb,
                        const float* beta, float* c, const blasint* ldc) {
  hwyblas::Symm("CSYMM", Structure::kSymmetric, side, uplo, m, n, C32(alpha),
                C32(a), lda, C32(b), ldb, C32(beta), C32(c), ldc);
}
HWYBLAS_API void zsymm_(const char* side, const char* uplo, const blasint* m,
                        const blasint* n, const double* alpha, const double* a,
                        const blasint* lda, const double* b, const blasint* ldb,
                        const double* beta, double* c, const blasint* ldc) {
  hwyblas::Symm("ZSYMM", Structure::kSymmetric, side, uplo, m, n, C64(alpha),
                C64(a), lda, C64(b), ldb, C64(beta), C64(c), ldc);
}

// HEMM

HWYBLAS_API void chemm_(const char* side, const char* uplo, const blasint* m,
                        const blasint* n, const float* alpha, const float* a,
                        const blasint* lda, const float* b, const blasint* ldb,
                        const float* beta, float* c, const blasint* ldc) {
  hwyblas::Symm("CHEMM", Structure::kHermitian, side, uplo, m, n, C32(alpha),
                C32(a), lda, C32(b), ldb, C32(beta), C32(c), ldc);
}
HWYBLAS_API void zhemm_(const char* side, const char* uplo, const blasint* m,
                        const blasint* n, const double* alpha, const double* a,
                        const blasint* lda, const double* b, const blasint* ldb,
                        const double* beta, double* c, const blasint* ldc) {
  hwyblas::Symm("ZHEMM", Structure::kHermitian, side, uplo, m, n, C64(alpha),
                C64(a), lda, C64(b), ldb, C64(beta), C64(c), ldc);
}

// SYRK

HWYBLAS_API void ssyrk_(const char* uplo, const char* trans, const blasint* n,
                        const blasint* k, const float* alpha, const float* a,
                        const blasint* lda, const float* beta, float* c,
                        const blasint* ldc) {
  hwyblas::Syrk("SSYRK", false, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
HWYBLAS_API void dsyrk_(const char* uplo, const char* trans, const blasint* n,
                        const blasint* k, const double* alpha, const double* a,
                        const blasint* lda, const double* beta, double* c,
                        const blasint* ldc) {
  hwyblas::Syrk("DSYRK", false, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
HWYBLAS_API void csyrk_(const char* uplo, const char* trans, const blasint* n,
                        const blasint* k, const float* alpha, const float* a,
                        const blasint* lda, const float* beta, float* c,
                        const blasint* ldc) {
  hwyblas::Syrk("CSYRK", false, uplo, trans, n, k, C32(alpha), C32(a), lda,
                C32(beta), C32(c), ldc);
}
HWYBLAS_API void zsyrk_(const char* uplo, const char* trans, const blasint* n,
                        const blasint* k, const double* alpha, const double* a,
                        const blasint* lda, const double* beta, double* c,
                        const blasint* ldc) {
  hwyblas::Syrk("ZSYRK", false, uplo, trans, n, k, C64(alpha), C64(a), lda,
                C64(beta), C64(c), ldc);
}

// HERK: alpha and beta are real.

HWYBLAS_API void cherk_(const char* uplo, const char* trans, const blasint* n,
                        const blasint* k, const float* alpha, const float* a,
                        const blasint* lda, const float* beta, float* c,
                        const blasint* ldc) {
  hwyblas::Syrk("CHERK", true, uplo, trans, n, k, alpha, C32(a), lda, beta,
                C32(c), ldc);
}
HWYBLAS_API void zherk_(const char* uplo, const char* trans, const blasint* n,
                        const blasint* k, const double* alpha, const double* a,
                        const blasint* lda, const double* beta, double* c,
                        const blasint* ldc) {
  hwyblas::Syrk("ZHERK", true, uplo, trans, n, k, alpha, C64(a), lda, beta,
                C64(c), ldc);
}

// SYR2K

HWYBLAS_API void ssyr2k_(const char* uplo, const char* trans, const blasint* n,
                         const blasint* k, const float* alpha, const float* a,
                         const blasint* lda, const float* b, const blasint* ldb,
                         const float* beta, float* c, const blasint* ldc) {
  hwyblas::Syr2k("SSYR2K", false, uplo, trans, n, k, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}
HWYBLAS_API void dsyr2k_(const char* uplo, const char* trans, const blasint* n,
                         const blasint* k, const double* alpha, const double* a,
                         const blasint* lda, const double* b,
                         const blasint* ldb, const double* beta, double* c,
                         const blasint* ldc) {
  hwyblas::Syr2k("DSYR2K", false, uplo, trans, n, k, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}
HWYBLAS_API void csyr2k_(const char* uplo, const char* trans, const blasint* n,
                         const blasint* k, const float* alpha, const float* a,
                         const blasint* lda, const float* b, const blasint* ldb,
                         const float* beta, float* c, const blasint* ldc) {
  hwyblas::Syr2k("CSYR2K", false, uplo, trans, n, k, C32(alpha), C32(a), lda,
                 C32(b), ldb, C32(beta), C32(c), ldc);
}
HWYBLAS_API void zsyr2k_(const char* uplo, const char* trans, const blasint* n,
                         const blasint* k, const double* alpha, const double* a,
                         const blasint* lda, const double* b,
                         const blasint* ldb, const double* beta, double* c,
                         const blasint* ldc) {
  hwyblas::Syr2k("ZSYR2K", false, uplo, trans, n, k, C64(alpha), C64(a), lda,
                 C64(b), ldb, C64(beta), C64(c), ldc);
}

// HER2K: beta is real.

HWYBLAS_API void cher2k_(const char* uplo, const char* trans, const blasint* n,
                         const blasint* k, const float* alpha, const float* a,
                         const blasint* lda, const float* b, const blasint* ldb,
                         const float* beta, float* c, const blasint* ldc) {
  hwyblas::Syr2k("CHER2K", true, uplo, trans, n, k, C32(alpha), C32(a), lda,
                 C32(b), ldb, beta, C32(c), ldc);
}
HWYBLAS_API void zher2k_(const char* uplo, const char* trans, const blasint* n,
                         const blasint* k, const double* alpha, const double* a,
                         const blasint* lda, const double* b,
                         const blasint* ldb, const double* beta, double* c,
                         const blasint* ldc) {
  hwyblas::Syr2k("ZHER2K", true, uplo, trans, n, k, C64(alpha), C64(a), lda,
                 C64(b), ldb, beta, C64(c), ldc);
}

// GEMMTR, and GEMMT: the name used by OpenBLAS and MKL for the same
// routine.

HWYBLAS_API void sgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const blasint* n,
                          const blasint* k, const float* alpha, const float* a,
                          const blasint* lda, const float* b,
                          const blasint* ldb, const float* beta, float* c,
                          const blasint* ldc) {
  hwyblas::Gemmtr("SGEMMTR", uplo, transa, transb, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}
HWYBLAS_API void dgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const blasint* n,
                          const blasint* k, const double* alpha,
                          const double* a, const blasint* lda, const double* b,
                          const blasint* ldb, const double* beta, double* c,
                          const blasint* ldc) {
  hwyblas::Gemmtr("DGEMMTR", uplo, transa, transb, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}
HWYBLAS_API void cgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const blasint* n,
                          const blasint* k, const float* alpha, const float* a,
                          const blasint* lda, const float* b,
                          const blasint* ldb, const float* beta, float* c,
                          const blasint* ldc) {
  hwyblas::Gemmtr("CGEMMTR", uplo, transa, transb, n, k, C32(alpha), C32(a),
                  lda, C32(b), ldb, C32(beta), C32(c), ldc);
}
HWYBLAS_API void zgemmtr_(const char* uplo, const char* transa,
                          const char* transb, const blasint* n,
                          const blasint* k, const double* alpha,
                          const double* a, const blasint* lda, const double* b,
                          const blasint* ldb, const double* beta, double* c,
                          const blasint* ldc) {
  hwyblas::Gemmtr("ZGEMMTR", uplo, transa, transb, n, k, C64(alpha), C64(a),
                  lda, C64(b), ldb, C64(beta), C64(c), ldc);
}
HWYBLAS_API void sgemmt_(const char* uplo, const char* transa,
                         const char* transb, const blasint* n, const blasint* k,
                         const float* alpha, const float* a, const blasint* lda,
                         const float* b, const blasint* ldb, const float* beta,
                         float* c, const blasint* ldc) {
  hwyblas::Gemmtr("SGEMMT", uplo, transa, transb, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}
HWYBLAS_API void dgemmt_(const char* uplo, const char* transa,
                         const char* transb, const blasint* n, const blasint* k,
                         const double* alpha, const double* a,
                         const blasint* lda, const double* b,
                         const blasint* ldb, const double* beta, double* c,
                         const blasint* ldc) {
  hwyblas::Gemmtr("DGEMMT", uplo, transa, transb, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}
HWYBLAS_API void cgemmt_(const char* uplo, const char* transa,
                         const char* transb, const blasint* n, const blasint* k,
                         const float* alpha, const float* a, const blasint* lda,
                         const float* b, const blasint* ldb, const float* beta,
                         float* c, const blasint* ldc) {
  hwyblas::Gemmtr("CGEMMT", uplo, transa, transb, n, k, C32(alpha), C32(a), lda,
                  C32(b), ldb, C32(beta), C32(c), ldc);
}
HWYBLAS_API void zgemmt_(const char* uplo, const char* transa,
                         const char* transb, const blasint* n, const blasint* k,
                         const double* alpha, const double* a,
                         const blasint* lda, const double* b,
                         const blasint* ldb, const double* beta, double* c,
                         const blasint* ldc) {
  hwyblas::Gemmtr("ZGEMMT", uplo, transa, transb, n, k, C64(alpha), C64(a), lda,
                  C64(b), ldb, C64(beta), C64(c), ldc);
}

// TRMM

HWYBLAS_API void strmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const float* alpha, const float* a, const blasint* lda,
                        float* b, const blasint* ldb) {
  hwyblas::Trmm("STRMM", side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void dtrmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, double* b, const blasint* ldb) {
  hwyblas::Trmm("DTRMM", side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void ctrmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const float* alpha, const float* a, const blasint* lda,
                        float* b, const blasint* ldb) {
  hwyblas::Trmm("CTRMM", side, uplo, transa, diag, m, n, C32(alpha), C32(a),
                lda, C32(b), ldb);
}
HWYBLAS_API void ztrmm_(const char* side, const char* uplo, const char* transa,
                        const char* diag, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, double* b, const blasint* ldb) {
  hwyblas::Trmm("ZTRMM", side, uplo, transa, diag, m, n, C64(alpha), C64(a),
                lda, C64(b), ldb);
}

}  // extern "C"
