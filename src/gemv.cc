// GEMV: per-target instantiation, runtime dispatch and the Fortran-style API.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/gemv.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/gemv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

#define HWYBLAS_GEMV(SUFFIX, T)                                     \
  void Gemv##SUFFIX(Op op, size_t m, size_t n, T alpha, const T* a, \
                    size_t lda, const T* x, T beta, T* y) {         \
    GemvContiguous(op, m, n, alpha, a, lda, x, beta, y);            \
  }
HWYBLAS_GEMV(F32, float)
HWYBLAS_GEMV(F64, double)
HWYBLAS_GEMV(C32, c32)
HWYBLAS_GEMV(C64, c64)
#undef HWYBLAS_GEMV

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"
#include "src/level2.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(Gemv);

// Validates the Fortran arguments, handles trivial cases, makes x and y
// contiguous if needed and dispatches.
template <typename T>
void Gemv(const char* name, const char* trans, const blasint* m_ptr,
          const blasint* n_ptr, const T* alpha_ptr, const T* a,
          const blasint* lda_ptr, const T* x, const blasint* incx_ptr,
          const T* beta_ptr, T* y, const blasint* incy_ptr) {
  const blasint m = *m_ptr, n = *n_ptr, lda = *lda_ptr;
  const blasint incx = *incx_ptr, incy = *incy_ptr;
  Op op;
  blasint info = 0;
  if (!DecodeOp(*trans, &op)) {
    info = 1;
  } else if (m < 0) {
    info = 2;
  } else if (n < 0) {
    info = 3;
  } else if (lda < HWY_MAX(1, m)) {
    info = 6;
  } else if (incx == 0) {
    info = 8;
  } else if (incy == 0) {
    info = 11;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr, beta = *beta_ptr;
  if (m == 0 || n == 0 || (IsZero(alpha) && IsOne(beta))) return;

  const size_t len_x = (op == Op::kNoTrans) ? n : m;
  const size_t len_y = (op == Op::kNoTrans) ? m : n;
  const T* xc = ContiguousInput(x, len_x, incx, kScratchX);
  // beta == 0: y must not be read.
  T* yc = BeginOutput(y, len_y, incy, kScratchY, !IsZero(beta));
  HWYBLAS_DISPATCH_BY_TYPE(T, Gemv, op, m, n, alpha, a, lda, xc, beta, yc);
  FinishOutput(yc, y, len_y, incy);
}

}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

HWYBLAS_API void sgemv_(const char* trans, const blasint* m, const blasint* n,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* x, const blasint* incx, const float* beta,
                        float* y, const blasint* incy) {
  hwyblas::Gemv("SGEMV", trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

HWYBLAS_API void dgemv_(const char* trans, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Gemv("DGEMV", trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

HWYBLAS_API void cgemv_(const char* trans, const blasint* m, const blasint* n,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* x, const blasint* incx, const float* beta,
                        float* y, const blasint* incy) {
  hwyblas::Gemv("CGEMV", trans, m, n, C32(alpha), C32(a), lda, C32(x), incx,
                C32(beta), C32(y), incy);
}

HWYBLAS_API void zgemv_(const char* trans, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Gemv("ZGEMV", trans, m, n, C64(alpha), C64(a), lda, C64(x), incx,
                C64(beta), C64(y), incy);
}

}  // extern "C"
#endif  // HWY_ONCE
