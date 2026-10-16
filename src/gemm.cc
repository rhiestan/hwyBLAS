// GEMM: per-target instantiation, runtime dispatch and the Fortran-style API.
//
// This file is compiled once per SIMD target by Highway's foreach_target.h;
// the code outside HWY_ONCE is the per-target part.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/gemm.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/gemm-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

void GemmF32(const GemmProblem<float>& g) { BlockedGemm(g); }
void GemmF64(const GemmProblem<double>& g) { BlockedGemm(g); }
void GemmC32(const GemmProblem<c32>& g) { BlockedGemm(g); }
void GemmC64(const GemmProblem<c64>& g) { BlockedGemm(g); }

const char* TargetName() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(Gemm);
HWY_EXPORT(TargetName);

template <typename T>
void RunGemm(const GemmProblem<T>& g) {
  HWYBLAS_DISPATCH_BY_TYPE(T, Gemm, g);
}
template void RunGemm(const GemmProblem<float>&);
template void RunGemm(const GemmProblem<double>&);
template void RunGemm(const GemmProblem<std::complex<float>>&);
template void RunGemm(const GemmProblem<std::complex<double>>&);

template <typename T>
void ScaleMatrix(size_t m, size_t n, T beta, T* c, size_t ldc, Part part,
                 bool real_diagonal) {
  for (size_t j = 0; j < n; ++j) {
    T* cj = c + j * ldc;
    // Row range of column j that belongs to `part`.
    const size_t begin = (part == Part::kLower) ? HWY_MIN(j, m) : 0;
    const size_t end = (part == Part::kUpper) ? HWY_MIN(j + 1, m) : m;
    // HERK: the imaginary part of the diagonal is neither read nor kept.
    if (real_diagonal && j < m) cj[j] = T(std::real(cj[j]));
    if (IsZero(beta)) {
      for (size_t i = begin; i < end; ++i) cj[i] = T(0);
    } else {
      for (size_t i = begin; i < end; ++i) cj[i] *= beta;
    }
  }
}
template void ScaleMatrix(size_t, size_t, float, float*, size_t, Part, bool);
template void ScaleMatrix(size_t, size_t, double, double*, size_t, Part, bool);
template void ScaleMatrix(size_t, size_t, std::complex<float>,
                          std::complex<float>*, size_t, Part, bool);
template void ScaleMatrix(size_t, size_t, std::complex<double>,
                          std::complex<double>*, size_t, Part, bool);

// Validates the Fortran arguments, handles trivial cases and dispatches.
template <typename T>
void Gemm(const char* name, const char* transa, const char* transb,
          const blasint* m_ptr, const blasint* n_ptr, const blasint* k_ptr,
          const T* alpha_ptr, const T* a, const blasint* lda_ptr, const T* b,
          const blasint* ldb_ptr, const T* beta_ptr, T* c,
          const blasint* ldc_ptr) {
  const blasint m = *m_ptr, n = *n_ptr, k = *k_ptr;
  const blasint lda = *lda_ptr, ldb = *ldb_ptr, ldc = *ldc_ptr;
  Op op_a, op_b;
  const bool valid_a = DecodeOp(*transa, &op_a);
  const bool valid_b = DecodeOp(*transb, &op_b);

  // Stored dimensions: A is m x k unless transposed (then k x m).
  const blasint rows_a = (op_a == Op::kNoTrans) ? m : k;
  const blasint rows_b = (op_b == Op::kNoTrans) ? k : n;

  blasint info = 0;
  if (!valid_a) {
    info = 1;
  } else if (!valid_b) {
    info = 2;
  } else if (m < 0) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (k < 0) {
    info = 5;
  } else if (lda < HWY_MAX(1, rows_a)) {
    info = 8;
  } else if (ldb < HWY_MAX(1, rows_b)) {
    info = 10;
  } else if (ldc < HWY_MAX(1, m)) {
    info = 13;
  }
  if (info != 0) {
    ReportError(name, info);
    return;
  }

  const T alpha = *alpha_ptr;
  const T beta = *beta_ptr;
  if (m == 0 || n == 0) return;
  if ((k == 0 || IsZero(alpha)) && IsOne(beta)) return;
  if (k == 0 || IsZero(alpha)) {
    ScaleMatrix(m, n, beta, c, ldc);
    return;
  }

  GemmProblem<T> g;
  g.m = m;
  g.n = n;
  g.k = k;
  g.alpha = alpha;
  g.beta = beta;
  g.c = c;
  g.ldc = ldc;
  g.a = GeneralOperand(op_a, a, lda);
  g.b = GeneralOperand(op_b, b, ldb);
  RunGemm(g);
}

}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

HWYBLAS_API void sgemm_(const char* transa, const char* transb,
                        const blasint* m, const blasint* n, const blasint* k,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* b, const blasint* ldb, const float* beta,
                        float* c, const blasint* ldc) {
  hwyblas::Gemm("SGEMM", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                c, ldc);
}

HWYBLAS_API void dgemm_(const char* transa, const char* transb,
                        const blasint* m, const blasint* n, const blasint* k,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* b, const blasint* ldb,
                        const double* beta, double* c, const blasint* ldc) {
  hwyblas::Gemm("DGEMM", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                c, ldc);
}

HWYBLAS_API void cgemm_(const char* transa, const char* transb,
                        const blasint* m, const blasint* n, const blasint* k,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* b, const blasint* ldb, const float* beta,
                        float* c, const blasint* ldc) {
  hwyblas::Gemm("CGEMM", transa, transb, m, n, k, C32(alpha), C32(a), lda,
                C32(b), ldb, C32(beta), C32(c), ldc);
}

HWYBLAS_API void zgemm_(const char* transa, const char* transb,
                        const blasint* m, const blasint* n, const blasint* k,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* b, const blasint* ldb,
                        const double* beta, double* c, const blasint* ldc) {
  hwyblas::Gemm("ZGEMM", transa, transb, m, n, k, C64(alpha), C64(a), lda,
                C64(b), ldb, C64(beta), C64(c), ldc);
}

HWYBLAS_API const char* hwyblas_target_name(void) {
  return HWY_DYNAMIC_DISPATCH(hwyblas::TargetName)();
}

}  // extern "C"
#endif  // HWY_ONCE
