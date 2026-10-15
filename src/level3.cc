// Level 3 routines that map directly onto the GEMM driver:
//
//   SYMM, HEMM  C := alpha * A * B + beta * C   (or B * A), A symmetric or
//               Hermitian, read from one triangle while packing.
//   SYRK, HERK  C := alpha * A * A^T + beta * C (or A^H, or A^T * A), only
//               one triangle of C is computed.
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
