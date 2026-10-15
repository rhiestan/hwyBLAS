// Level 2 routines besides GEMV: SYMV, HEMV and TRMV.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/level2.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/symv-inl.h"
#include "src/trmv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

#define HWYBLAS_LEVEL2(SUFFIX, T)                                              \
  void Symv##SUFFIX(bool hermitian, bool lower, size_t n, T alpha, const T* a, \
                    size_t lda, const T* x, T beta, T* y) {                    \
    if (hermitian) {                                                           \
      SymvContiguous<true>(lower, n, alpha, a, lda, x, beta, y);               \
    } else {                                                                   \
      SymvContiguous<false>(lower, n, alpha, a, lda, x, beta, y);              \
    }                                                                          \
  }                                                                            \
  void Trmv##SUFFIX(bool lower, Op op, bool unit, size_t n, const T* a,        \
                    size_t lda, T* x) {                                        \
    TrmvContiguous(lower, op, unit, n, a, lda, x);                             \
  }
HWYBLAS_LEVEL2(F32, float)
HWYBLAS_LEVEL2(F64, double)
HWYBLAS_LEVEL2(C32, c32)
HWYBLAS_LEVEL2(C64, c64)
#undef HWYBLAS_LEVEL2

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"
#include "src/level2.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(Symv);
HWYBLAS_EXPORT_BY_TYPE(Trmv);

// SYMV (hermitian = false) and HEMV (hermitian = true):
// y := alpha * A * x + beta * y.
template <typename T>
void Symv(const char* name, bool hermitian, const char* uplo_ptr,
          const blasint* n_ptr, const T* alpha_ptr, const T* a,
          const blasint* lda_ptr, const T* x, const blasint* incx_ptr,
          const T* beta_ptr, T* y, const blasint* incy_ptr) {
  const blasint n = *n_ptr, lda = *lda_ptr;
  const blasint incx = *incx_ptr, incy = *incy_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo_ptr, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (lda < HWY_MAX(1, n)) {
    info = 5;
  } else if (incx == 0) {
    info = 7;
  } else if (incy == 0) {
    info = 10;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr, beta = *beta_ptr;
  if (n == 0 || (IsZero(alpha) && IsOne(beta))) return;

  const T* xc = ContiguousInput(x, n, incx, kScratchX);
  T* yc = BeginOutput(y, n, incy, kScratchY, !IsZero(beta));
  HWYBLAS_DISPATCH_BY_TYPE(T, Symv, hermitian, lower, n, alpha, a, lda, xc,
                           beta, yc);
  FinishOutput(yc, y, n, incy);
}

// TRMV: x := op(A) * x.
template <typename T>
void Trmv(const char* name, const char* uplo_ptr, const char* trans_ptr,
          const char* diag_ptr, const blasint* n_ptr, const T* a,
          const blasint* lda_ptr, T* x, const blasint* incx_ptr) {
  const blasint n = *n_ptr, lda = *lda_ptr, incx = *incx_ptr;
  bool lower, unit;
  Op op;
  blasint info = 0;
  if (!DecodeUplo(*uplo_ptr, &lower)) {
    info = 1;
  } else if (!DecodeOp(*trans_ptr, &op)) {
    info = 2;
  } else if (!DecodeDiag(*diag_ptr, &unit)) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (lda < HWY_MAX(1, n)) {
    info = 6;
  } else if (incx == 0) {
    info = 8;
  }
  if (info != 0) return ReportError(name, info);
  if (n == 0) return;

  T* xc = BeginOutput(x, n, incx, kScratchX, /*read=*/true);
  HWYBLAS_DISPATCH_BY_TYPE(T, Trmv, lower, op, unit, n, a, lda, xc);
  FinishOutput(xc, x, n, incx);
}

}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

// SYMV (real) and HEMV (complex)

HWYBLAS_API void ssymv_(const char* uplo, const blasint* n, const float* alpha,
                        const float* a, const blasint* lda, const float* x,
                        const blasint* incx, const float* beta, float* y,
                        const blasint* incy) {
  hwyblas::Symv("SSYMV", false, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void dsymv_(const char* uplo, const blasint* n, const double* alpha,
                        const double* a, const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Symv("DSYMV", false, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void chemv_(const char* uplo, const blasint* n, const float* alpha,
                        const float* a, const blasint* lda, const float* x,
                        const blasint* incx, const float* beta, float* y,
                        const blasint* incy) {
  hwyblas::Symv("CHEMV", true, uplo, n, C32(alpha), C32(a), lda, C32(x), incx,
                C32(beta), C32(y), incy);
}
HWYBLAS_API void zhemv_(const char* uplo, const blasint* n, const double* alpha,
                        const double* a, const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Symv("ZHEMV", true, uplo, n, C64(alpha), C64(a), lda, C64(x), incx,
                C64(beta), C64(y), incy);
}

// TRMV

HWYBLAS_API void strmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* a, const blasint* lda,
                        float* x, const blasint* incx) {
  hwyblas::Trmv("STRMV", uplo, trans, diag, n, a, lda, x, incx);
}
HWYBLAS_API void dtrmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* a, const blasint* lda,
                        double* x, const blasint* incx) {
  hwyblas::Trmv("DTRMV", uplo, trans, diag, n, a, lda, x, incx);
}
HWYBLAS_API void ctrmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* a, const blasint* lda,
                        float* x, const blasint* incx) {
  hwyblas::Trmv("CTRMV", uplo, trans, diag, n, C32(a), lda, C32(x), incx);
}
HWYBLAS_API void ztrmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* a, const blasint* lda,
                        double* x, const blasint* incx) {
  hwyblas::Trmv("ZTRMV", uplo, trans, diag, n, C64(a), lda, C64(x), incx);
}

}  // extern "C"
#endif  // HWY_ONCE
