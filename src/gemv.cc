// GEMV: per-target instantiation, runtime dispatch and the Fortran-style API.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/gemv.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/gemv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

#define HWYBLAS_GEMV(NAME, T)                                           \
  void NAME(Op op, size_t m, size_t n, T alpha, const T* a, size_t lda, \
            const T* x, T beta, T* y) {                                 \
    GemvContiguous(op, m, n, alpha, a, lda, x, beta, y);                \
  }
HWYBLAS_GEMV(GemvF32, float)
HWYBLAS_GEMV(GemvF64, double)
HWYBLAS_GEMV(GemvC32, std::complex<float>)
HWYBLAS_GEMV(GemvC64, std::complex<double>)
#undef HWYBLAS_GEMV

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwyblas {

HWY_EXPORT(GemvF32);
HWY_EXPORT(GemvF64);
HWY_EXPORT(GemvC32);
HWY_EXPORT(GemvC64);

template <typename T>
void DispatchGemv(Op op, size_t m, size_t n, T alpha, const T* a, size_t lda,
                  const T* x, T beta, T* y) {
  if constexpr (std::is_same_v<T, float>) {
    HWY_DYNAMIC_DISPATCH(GemvF32)(op, m, n, alpha, a, lda, x, beta, y);
  } else if constexpr (std::is_same_v<T, double>) {
    HWY_DYNAMIC_DISPATCH(GemvF64)(op, m, n, alpha, a, lda, x, beta, y);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    HWY_DYNAMIC_DISPATCH(GemvC32)(op, m, n, alpha, a, lda, x, beta, y);
  } else {
    HWY_DYNAMIC_DISPATCH(GemvC64)(op, m, n, alpha, a, lda, x, beta, y);
  }
}

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
  if (info != 0) {
    ReportError(name, info);
    return;
  }

  const T alpha = *alpha_ptr, beta = *beta_ptr;
  if (m == 0 || n == 0 || (IsZero(alpha) && IsOne(beta))) return;

  const size_t len_x = (op == Op::kNoTrans) ? n : m;
  const size_t len_y = (op == Op::kNoTrans) ? m : n;

  // Gather a strided x into a contiguous buffer.
  const T* x_start = VectorStart(x, len_x, incx);
  const T* x_contig = x_start;
  if (incx != 1) {
    T* buf = static_cast<T*>(ScratchBuffer(kScratchX, len_x * sizeof(T)));
    for (size_t i = 0; i < len_x; ++i) buf[i] = x_start[i * incx];
    x_contig = buf;
  }

  if (incy == 1) {
    DispatchGemv(op, m, n, alpha, a, lda, x_contig, beta, y);
    return;
  }

  // Strided y: compute into a contiguous copy, then scatter back.
  T* y_start = VectorStart(y, len_y, incy);
  T* buf = static_cast<T*>(ScratchBuffer(kScratchY, len_y * sizeof(T)));
  if (!IsZero(beta)) {  // beta == 0: y must not be read
    for (size_t i = 0; i < len_y; ++i) buf[i] = y_start[i * incy];
  }
  DispatchGemv(op, m, n, alpha, a, lda, x_contig, beta, buf);
  for (size_t i = 0; i < len_y; ++i) y_start[i * incy] = buf[i];
}

}  // namespace hwyblas

using hwyblas::blasint;
using c32 = std::complex<float>;
using c64 = std::complex<double>;

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
  hwyblas::Gemv("CGEMV", trans, m, n, reinterpret_cast<const c32*>(alpha),
                reinterpret_cast<const c32*>(a), lda,
                reinterpret_cast<const c32*>(x), incx,
                reinterpret_cast<const c32*>(beta), reinterpret_cast<c32*>(y),
                incy);
}

HWYBLAS_API void zgemv_(const char* trans, const blasint* m, const blasint* n,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Gemv("ZGEMV", trans, m, n, reinterpret_cast<const c64*>(alpha),
                reinterpret_cast<const c64*>(a), lda,
                reinterpret_cast<const c64*>(x), incx,
                reinterpret_cast<const c64*>(beta), reinterpret_cast<c64*>(y),
                incy);
}

}  // extern "C"
#endif  // HWY_ONCE
