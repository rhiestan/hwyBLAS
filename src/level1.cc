// Level 1 routines: AXPY.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/level1.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/level1-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

#define HWYBLAS_LEVEL1(SUFFIX, T)                          \
  void Axpy##SUFFIX(size_t n, T alpha, const T* x, T* y) { \
    AxpyContiguous(n, alpha, x, y);                        \
  }
HWYBLAS_LEVEL1(F32, float)
HWYBLAS_LEVEL1(F64, double)
HWYBLAS_LEVEL1(C32, c32)
HWYBLAS_LEVEL1(C64, c64)
#undef HWYBLAS_LEVEL1

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(Axpy);

// y := alpha * x + y. Like the reference BLAS, there are no error checks:
// n <= 0 is a no-op.
template <typename T>
void Axpy(const blasint* n_ptr, const T* alpha_ptr, const T* x,
          const blasint* incx_ptr, T* y, const blasint* incy_ptr) {
  const blasint n = *n_ptr, incx = *incx_ptr, incy = *incy_ptr;
  const T alpha = *alpha_ptr;
  if (n <= 0 || IsZero(alpha)) return;
  if (incx == 1 && incy == 1) {
    HWYBLAS_DISPATCH_BY_TYPE(T, Axpy, n, alpha, x, y);
    return;
  }
  const T* xs = VectorStart(x, n, incx);
  T* ys = VectorStart(y, n, incy);
  for (blasint i = 0; i < n; ++i) {
    ys[static_cast<ptrdiff_t>(i) * incy] +=
        alpha * xs[static_cast<ptrdiff_t>(i) * incx];
  }
}

}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

HWYBLAS_API void saxpy_(const blasint* n, const float* alpha, const float* x,
                        const blasint* incx, float* y, const blasint* incy) {
  hwyblas::Axpy(n, alpha, x, incx, y, incy);
}
HWYBLAS_API void daxpy_(const blasint* n, const double* alpha, const double* x,
                        const blasint* incx, double* y, const blasint* incy) {
  hwyblas::Axpy(n, alpha, x, incx, y, incy);
}
HWYBLAS_API void caxpy_(const blasint* n, const float* alpha, const float* x,
                        const blasint* incx, float* y, const blasint* incy) {
  hwyblas::Axpy(n, C32(alpha), C32(x), incx, C32(y), incy);
}
HWYBLAS_API void zaxpy_(const blasint* n, const double* alpha, const double* x,
                        const blasint* incx, double* y, const blasint* incy) {
  hwyblas::Axpy(n, C64(alpha), C64(x), incx, C64(y), incy);
}

}  // extern "C"
#endif  // HWY_ONCE
