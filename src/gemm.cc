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
void GemmC32(const GemmProblem<std::complex<float>>& g) { BlockedGemm(g); }
void GemmC64(const GemmProblem<std::complex<double>>& g) { BlockedGemm(g); }

const char* TargetName() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwyblas {

HWY_EXPORT(GemmF32);
HWY_EXPORT(GemmF64);
HWY_EXPORT(GemmC32);
HWY_EXPORT(GemmC64);
HWY_EXPORT(TargetName);

template <typename T>
void Dispatch(const GemmProblem<T>& g) {
  if constexpr (std::is_same_v<T, float>) {
    HWY_DYNAMIC_DISPATCH(GemmF32)(g);
  } else if constexpr (std::is_same_v<T, double>) {
    HWY_DYNAMIC_DISPATCH(GemmF64)(g);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    HWY_DYNAMIC_DISPATCH(GemmC32)(g);
  } else {
    HWY_DYNAMIC_DISPATCH(GemmC64)(g);
  }
}

// C := beta * C for an m x n column-major matrix. beta == 0 sets C to zero
// without reading it (so NaN/Inf in C are not propagated).
template <typename T>
void ScaleMatrix(size_t m, size_t n, T beta, T* c, size_t ldc) {
  for (size_t j = 0; j < n; ++j) {
    T* cj = c + j * ldc;
    if (IsZero(beta)) {
      for (size_t i = 0; i < m; ++i) cj[i] = T(0);
    } else {
      for (size_t i = 0; i < m; ++i) cj[i] *= beta;
    }
  }
}

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
  g.a = a;
  g.b = b;
  g.c = c;
  g.ldc = ldc;
  // op(A)(i, p): A(i, p) = a[i + p * lda], or A(p, i) = a[p + i * lda].
  g.a_rs = (op_a == Op::kNoTrans) ? 1 : lda;
  g.a_cs = (op_a == Op::kNoTrans) ? lda : 1;
  g.conj_a = (op_a == Op::kConjTrans);
  // op(B)(p, j): B(p, j) = b[p + j * ldb], or B(j, p) = b[j + p * ldb].
  g.b_rs = (op_b == Op::kNoTrans) ? 1 : ldb;
  g.b_cs = (op_b == Op::kNoTrans) ? ldb : 1;
  g.conj_b = (op_b == Op::kConjTrans);
  Dispatch(g);
}

}  // namespace hwyblas

using hwyblas::blasint;
using c32 = std::complex<float>;
using c64 = std::complex<double>;

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
  hwyblas::Gemm(
      "CGEMM", transa, transb, m, n, k, reinterpret_cast<const c32*>(alpha),
      reinterpret_cast<const c32*>(a), lda, reinterpret_cast<const c32*>(b),
      ldb, reinterpret_cast<const c32*>(beta), reinterpret_cast<c32*>(c), ldc);
}

HWYBLAS_API void zgemm_(const char* transa, const char* transb,
                        const blasint* m, const blasint* n, const blasint* k,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* b, const blasint* ldb,
                        const double* beta, double* c, const blasint* ldc) {
  hwyblas::Gemm(
      "ZGEMM", transa, transb, m, n, k, reinterpret_cast<const c64*>(alpha),
      reinterpret_cast<const c64*>(a), lda, reinterpret_cast<const c64*>(b),
      ldb, reinterpret_cast<const c64*>(beta), reinterpret_cast<c64*>(c), ldc);
}

HWYBLAS_API const char* hwyblas_target_name(void) {
  return HWY_DYNAMIC_DISPATCH(hwyblas::TargetName)();
}

}  // extern "C"
#endif  // HWY_ONCE
