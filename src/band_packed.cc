// Banded and packed Level 2 routines: GBMV, SBMV, HBMV, SPMV, HPMV, TBMV,
// TBSV, TPMV, TPSV, SPR, HPR, SPR2 and HPR2. The kernels are in
// band_packed-inl.h.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/band_packed.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/band_packed-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

#define HWYBLAS_BAND_PACKED(SUFFIX, T)                                         \
  void BandGemv##SUFFIX(const ColumnLayout& layout, Op op, size_t m, size_t n, \
                        T alpha, const T* a, const T* x, T beta, T* y) {       \
    BandGemv(layout, op, m, n, alpha, a, x, beta, y);                          \
  }                                                                            \
  void SymmetricMv##SUFFIX(const ColumnLayout& layout, bool lower, size_t n,   \
                           T alpha, const T* a, const T* x, T beta, T* y) {    \
    SymmetricColumnsMv<IsComplex<T>::value>(layout, lower, n, alpha, a, x,     \
                                            beta, y);                          \
  }                                                                            \
  void TriangularMv##SUFFIX(bool solve, const ColumnLayout& layout,            \
                            bool lower, Op op, bool unit, size_t n,            \
                            const T* a, T* x) {                                \
    if (solve) {                                                               \
      TriangularColumnsSv(layout, lower, op, unit, n, a, x);                   \
    } else {                                                                   \
      TriangularColumnsMv(layout, lower, op, unit, n, a, x);                   \
    }                                                                          \
  }                                                                            \
  void PackedUpdate##SUFFIX(size_t terms, const ColumnLayout& layout,          \
                            bool lower, size_t n, T* a, const T* u0,           \
                            const T* s0, const T* u1, const T* s1) {           \
    constexpr bool kHermitian = IsComplex<T>::value;                           \
    if (terms == 1) {                                                          \
      PackedRankUpdate<1>(layout, lower, kHermitian, n, a, u0, s0, u0, s0);    \
    } else {                                                                   \
      PackedRankUpdate<2>(layout, lower, kHermitian, n, a, u0, s0, u1, s1);    \
    }                                                                          \
  }
HWYBLAS_BAND_PACKED(F32, float)
HWYBLAS_BAND_PACKED(F64, double)
HWYBLAS_BAND_PACKED(C32, c32)
HWYBLAS_BAND_PACKED(C64, c64)
#undef HWYBLAS_BAND_PACKED

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "src/api.h"
#include "src/band_packed.h"
#include "src/level2.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(BandGemv);
HWYBLAS_EXPORT_BY_TYPE(SymmetricMv);
HWYBLAS_EXPORT_BY_TYPE(TriangularMv);
HWYBLAS_EXPORT_BY_TYPE(PackedUpdate);

// GBMV: y := alpha * op(A) * x + beta * y, A m x n with kl sub- and ku
// super-diagonals.
template <typename T>
void Gbmv(const char* name, const char* trans, const blasint* m_ptr,
          const blasint* n_ptr, const blasint* kl_ptr, const blasint* ku_ptr,
          const T* alpha_ptr, const T* a, const blasint* lda_ptr, const T* x,
          const blasint* incx_ptr, const T* beta_ptr, T* y,
          const blasint* incy_ptr) {
  const blasint m = *m_ptr, n = *n_ptr, kl = *kl_ptr, ku = *ku_ptr;
  const blasint lda = *lda_ptr, incx = *incx_ptr, incy = *incy_ptr;
  Op op;
  blasint info = 0;
  if (!DecodeOp(*trans, &op)) {
    info = 1;
  } else if (m < 0) {
    info = 2;
  } else if (n < 0) {
    info = 3;
  } else if (kl < 0) {
    info = 4;
  } else if (ku < 0) {
    info = 5;
  } else if (lda < kl + ku + 1) {
    info = 8;
  } else if (incx == 0) {
    info = 10;
  } else if (incy == 0) {
    info = 13;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr, beta = *beta_ptr;
  if (m == 0 || n == 0 || (IsZero(alpha) && IsOne(beta))) return;

  const size_t len_x = (op == Op::kNoTrans) ? n : m;
  const size_t len_y = (op == Op::kNoTrans) ? m : n;
  // The layout of op(A)'s columns is that of A's; the kernel visits the n
  // columns of A.
  const ColumnLayout layout = ColumnLayout::Band(m, kl, ku, lda);
  const T* xc = ContiguousInput(x, len_x, incx, kScratchX);
  T* yc = BeginOutput(y, len_y, incy, kScratchY, !IsZero(beta));
  HWYBLAS_DISPATCH_BY_TYPE(T, BandGemv, layout, op, m, n, alpha, a, xc, beta,
                           yc);
  FinishOutput(yc, y, len_y, incy);
}

// y := alpha * A * x + beta * y for a symmetric or Hermitian A with the given
// layout (shared by SBMV, HBMV, SPMV and HPMV after argument checking).
template <typename T>
void SymmetricMv(const ColumnLayout& layout, bool lower, blasint n, T alpha,
                 const T* a, const T* x, blasint incx, T beta, T* y,
                 blasint incy) {
  if (n == 0 || (IsZero(alpha) && IsOne(beta))) return;
  const T* xc = ContiguousInput(x, n, incx, kScratchX);
  T* yc = BeginOutput(y, n, incy, kScratchY, !IsZero(beta));
  HWYBLAS_DISPATCH_BY_TYPE(T, SymmetricMv, layout, lower, n, alpha, a, xc, beta,
                           yc);
  FinishOutput(yc, y, n, incy);
}

// SBMV (real T), HBMV (complex T): A has k off-diagonals.
template <typename T>
void Sbmv(const char* name, const char* uplo, const blasint* n_ptr,
          const blasint* k_ptr, const T* alpha_ptr, const T* a,
          const blasint* lda_ptr, const T* x, const blasint* incx_ptr,
          const T* beta_ptr, T* y, const blasint* incy_ptr) {
  const blasint n = *n_ptr, k = *k_ptr, lda = *lda_ptr;
  const blasint incx = *incx_ptr, incy = *incy_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (k < 0) {
    info = 3;
  } else if (lda < k + 1) {
    info = 6;
  } else if (incx == 0) {
    info = 8;
  } else if (incy == 0) {
    info = 11;
  }
  if (info != 0) return ReportError(name, info);
  const ColumnLayout layout =
      ColumnLayout::Band(n, lower ? k : 0, lower ? 0 : k, lda);
  SymmetricMv(layout, lower, n, *alpha_ptr, a, x, incx, *beta_ptr, y, incy);
}

// SPMV (real T), HPMV (complex T).
template <typename T>
void Spmv(const char* name, const char* uplo, const blasint* n_ptr,
          const T* alpha_ptr, const T* ap, const T* x, const blasint* incx_ptr,
          const T* beta_ptr, T* y, const blasint* incy_ptr) {
  const blasint n = *n_ptr, incx = *incx_ptr, incy = *incy_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx == 0) {
    info = 6;
  } else if (incy == 0) {
    info = 9;
  }
  if (info != 0) return ReportError(name, info);
  SymmetricMv(ColumnLayout::Packed(lower, n), lower, n, *alpha_ptr, ap, x, incx,
              *beta_ptr, y, incy);
}

// x := op(A) * x (solve = false) or solves op(A) * x = b (solve = true) for
// a triangular A with the given layout (shared by TBMV, TBSV, TPMV, TPSV).
template <typename T>
void TriangularMv(bool solve, const ColumnLayout& layout, bool lower, Op op,
                  bool unit, blasint n, const T* a, T* x, blasint incx) {
  if (n == 0) return;
  T* xc = BeginOutput(x, n, incx, kScratchX, /*read=*/true);
  HWYBLAS_DISPATCH_BY_TYPE(T, TriangularMv, solve, layout, lower, op, unit, n,
                           a, xc);
  FinishOutput(xc, x, n, incx);
}

// Decodes UPLO, TRANS and DIAG; returns the 1-based position of the first
// invalid one, or 0.
inline blasint DecodeTriangular(const char* uplo, const char* trans,
                                const char* diag, bool* lower, Op* op,
                                bool* unit) {
  if (!DecodeUplo(*uplo, lower)) return 1;
  if (!DecodeOp(*trans, op)) return 2;
  if (!DecodeDiag(*diag, unit)) return 3;
  return 0;
}

// TBMV, TBSV: A has k off-diagonals.
template <typename T>
void Tbmv(const char* name, bool solve, const char* uplo, const char* trans,
          const char* diag, const blasint* n_ptr, const blasint* k_ptr,
          const T* a, const blasint* lda_ptr, T* x, const blasint* incx_ptr) {
  const blasint n = *n_ptr, k = *k_ptr, lda = *lda_ptr, incx = *incx_ptr;
  bool lower, unit;
  Op op;
  blasint info = DecodeTriangular(uplo, trans, diag, &lower, &op, &unit);
  if (info != 0) {
  } else if (n < 0) {
    info = 4;
  } else if (k < 0) {
    info = 5;
  } else if (lda < k + 1) {
    info = 7;
  } else if (incx == 0) {
    info = 9;
  }
  if (info != 0) return ReportError(name, info);
  const ColumnLayout layout =
      ColumnLayout::Band(n, lower ? k : 0, lower ? 0 : k, lda);
  TriangularMv(solve, layout, lower, op, unit, n, a, x, incx);
}

// TPMV, TPSV.
template <typename T>
void Tpmv(const char* name, bool solve, const char* uplo, const char* trans,
          const char* diag, const blasint* n_ptr, const T* ap, T* x,
          const blasint* incx_ptr) {
  const blasint n = *n_ptr, incx = *incx_ptr;
  bool lower, unit;
  Op op;
  blasint info = DecodeTriangular(uplo, trans, diag, &lower, &op, &unit);
  if (info != 0) {
  } else if (n < 0) {
    info = 4;
  } else if (incx == 0) {
    info = 7;
  }
  if (info != 0) return ReportError(name, info);
  TriangularMv(solve, ColumnLayout::Packed(lower, n), lower, op, unit, n, ap, x,
               incx);
}

// SPR (real T): A := alpha * x * x^T + A, and HPR (complex T, real alpha):
// A := alpha * x * x^H + A, A packed.
template <typename T>
void Spr(const char* name, const char* uplo, const blasint* n_ptr,
         const RealT<T>* alpha_ptr, const T* x, const blasint* incx_ptr,
         T* ap) {
  const blasint n = *n_ptr, incx = *incx_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx == 0) {
    info = 5;
  }
  if (info != 0) return ReportError(name, info);

  const RealT<T> alpha = *alpha_ptr;
  if (n == 0 || alpha == 0) return;
  const T* xc = ContiguousInput(x, n, incx, kScratchX);
  // Row scales s = alpha * conj(x) (Conj is the identity for real T).
  T* s = static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
  for (blasint j = 0; j < n; ++j) s[j] = alpha * Conj(xc[j]);
  HWYBLAS_DISPATCH_BY_TYPE(T, PackedUpdate, 1, ColumnLayout::Packed(lower, n),
                           lower, n, ap, xc, s, xc, s);
}

// SPR2 (real T): A := alpha * x * y^T + alpha * y * x^T + A, and HPR2
// (complex T): A := alpha * x * y^H + conj(alpha) * y * x^H + A, A packed.
template <typename T>
void Spr2(const char* name, const char* uplo, const blasint* n_ptr,
          const T* alpha_ptr, const T* x, const blasint* incx_ptr, const T* y,
          const blasint* incy_ptr, T* ap) {
  const blasint n = *n_ptr, incx = *incx_ptr, incy = *incy_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx == 0) {
    info = 5;
  } else if (incy == 0) {
    info = 7;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr;
  if (n == 0 || IsZero(alpha)) return;
  const T* xc = ContiguousInput(x, n, incx, kScratchX);
  const T* yc = ContiguousInput(y, n, incy, kScratchY);
  // Row scales s0 = alpha * conj(y) and s1 = conj(alpha * x).
  T* s0 = static_cast<T*>(ScratchBuffer(kScratchTemp, 2 * n * sizeof(T)));
  T* s1 = s0 + n;
  for (blasint j = 0; j < n; ++j) {
    s0[j] = alpha * Conj(yc[j]);
    s1[j] = Conj(alpha * xc[j]);
  }
  HWYBLAS_DISPATCH_BY_TYPE(T, PackedUpdate, 2, ColumnLayout::Packed(lower, n),
                           lower, n, ap, xc, s0, yc, s1);
}

}  // namespace hwyblas

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

extern "C" {

// GBMV

HWYBLAS_API void sgbmv_(const char* trans, const blasint* m, const blasint* n,
                        const blasint* kl, const blasint* ku,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* x, const blasint* incx, const float* beta,
                        float* y, const blasint* incy) {
  hwyblas::Gbmv("SGBMV", trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                incy);
}
HWYBLAS_API void dgbmv_(const char* trans, const blasint* m, const blasint* n,
                        const blasint* kl, const blasint* ku,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Gbmv("DGBMV", trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                incy);
}
HWYBLAS_API void cgbmv_(const char* trans, const blasint* m, const blasint* n,
                        const blasint* kl, const blasint* ku,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* x, const blasint* incx, const float* beta,
                        float* y, const blasint* incy) {
  hwyblas::Gbmv("CGBMV", trans, m, n, kl, ku, C32(alpha), C32(a), lda, C32(x),
                incx, C32(beta), C32(y), incy);
}
HWYBLAS_API void zgbmv_(const char* trans, const blasint* m, const blasint* n,
                        const blasint* kl, const blasint* ku,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Gbmv("ZGBMV", trans, m, n, kl, ku, C64(alpha), C64(a), lda, C64(x),
                incx, C64(beta), C64(y), incy);
}

// SBMV

HWYBLAS_API void ssbmv_(const char* uplo, const blasint* n, const blasint* k,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* x, const blasint* incx, const float* beta,
                        float* y, const blasint* incy) {
  hwyblas::Sbmv("SSBMV", uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void dsbmv_(const char* uplo, const blasint* n, const blasint* k,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Sbmv("DSBMV", uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

// HBMV

HWYBLAS_API void chbmv_(const char* uplo, const blasint* n, const blasint* k,
                        const float* alpha, const float* a, const blasint* lda,
                        const float* x, const blasint* incx, const float* beta,
                        float* y, const blasint* incy) {
  hwyblas::Sbmv("CHBMV", uplo, n, k, C32(alpha), C32(a), lda, C32(x), incx,
                C32(beta), C32(y), incy);
}
HWYBLAS_API void zhbmv_(const char* uplo, const blasint* n, const blasint* k,
                        const double* alpha, const double* a,
                        const blasint* lda, const double* x,
                        const blasint* incx, const double* beta, double* y,
                        const blasint* incy) {
  hwyblas::Sbmv("ZHBMV", uplo, n, k, C64(alpha), C64(a), lda, C64(x), incx,
                C64(beta), C64(y), incy);
}

// SPMV

HWYBLAS_API void sspmv_(const char* uplo, const blasint* n, const float* alpha,
                        const float* ap, const float* x, const blasint* incx,
                        const float* beta, float* y, const blasint* incy) {
  hwyblas::Spmv("SSPMV", uplo, n, alpha, ap, x, incx, beta, y, incy);
}
HWYBLAS_API void dspmv_(const char* uplo, const blasint* n, const double* alpha,
                        const double* ap, const double* x, const blasint* incx,
                        const double* beta, double* y, const blasint* incy) {
  hwyblas::Spmv("DSPMV", uplo, n, alpha, ap, x, incx, beta, y, incy);
}

// HPMV

HWYBLAS_API void chpmv_(const char* uplo, const blasint* n, const float* alpha,
                        const float* ap, const float* x, const blasint* incx,
                        const float* beta, float* y, const blasint* incy) {
  hwyblas::Spmv("CHPMV", uplo, n, C32(alpha), C32(ap), C32(x), incx, C32(beta),
                C32(y), incy);
}
HWYBLAS_API void zhpmv_(const char* uplo, const blasint* n, const double* alpha,
                        const double* ap, const double* x, const blasint* incx,
                        const double* beta, double* y, const blasint* incy) {
  hwyblas::Spmv("ZHPMV", uplo, n, C64(alpha), C64(ap), C64(x), incx, C64(beta),
                C64(y), incy);
}

// TBMV

HWYBLAS_API void stbmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const float* a,
                        const blasint* lda, float* x, const blasint* incx) {
  hwyblas::Tbmv("STBMV", false, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void dtbmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const double* a,
                        const blasint* lda, double* x, const blasint* incx) {
  hwyblas::Tbmv("DTBMV", false, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void ctbmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const float* a,
                        const blasint* lda, float* x, const blasint* incx) {
  hwyblas::Tbmv("CTBMV", false, uplo, trans, diag, n, k, C32(a), lda, C32(x),
                incx);
}
HWYBLAS_API void ztbmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const double* a,
                        const blasint* lda, double* x, const blasint* incx) {
  hwyblas::Tbmv("ZTBMV", false, uplo, trans, diag, n, k, C64(a), lda, C64(x),
                incx);
}

// TBSV

HWYBLAS_API void stbsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const float* a,
                        const blasint* lda, float* x, const blasint* incx) {
  hwyblas::Tbmv("STBSV", true, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void dtbsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const double* a,
                        const blasint* lda, double* x, const blasint* incx) {
  hwyblas::Tbmv("DTBSV", true, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void ctbsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const float* a,
                        const blasint* lda, float* x, const blasint* incx) {
  hwyblas::Tbmv("CTBSV", true, uplo, trans, diag, n, k, C32(a), lda, C32(x),
                incx);
}
HWYBLAS_API void ztbsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const blasint* k, const double* a,
                        const blasint* lda, double* x, const blasint* incx) {
  hwyblas::Tbmv("ZTBSV", true, uplo, trans, diag, n, k, C64(a), lda, C64(x),
                incx);
}

// TPMV

HWYBLAS_API void stpmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* ap, float* x,
                        const blasint* incx) {
  hwyblas::Tpmv("STPMV", false, uplo, trans, diag, n, ap, x, incx);
}
HWYBLAS_API void dtpmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* ap, double* x,
                        const blasint* incx) {
  hwyblas::Tpmv("DTPMV", false, uplo, trans, diag, n, ap, x, incx);
}
HWYBLAS_API void ctpmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* ap, float* x,
                        const blasint* incx) {
  hwyblas::Tpmv("CTPMV", false, uplo, trans, diag, n, C32(ap), C32(x), incx);
}
HWYBLAS_API void ztpmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* ap, double* x,
                        const blasint* incx) {
  hwyblas::Tpmv("ZTPMV", false, uplo, trans, diag, n, C64(ap), C64(x), incx);
}

// TPSV

HWYBLAS_API void stpsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* ap, float* x,
                        const blasint* incx) {
  hwyblas::Tpmv("STPSV", true, uplo, trans, diag, n, ap, x, incx);
}
HWYBLAS_API void dtpsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* ap, double* x,
                        const blasint* incx) {
  hwyblas::Tpmv("DTPSV", true, uplo, trans, diag, n, ap, x, incx);
}
HWYBLAS_API void ctpsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* ap, float* x,
                        const blasint* incx) {
  hwyblas::Tpmv("CTPSV", true, uplo, trans, diag, n, C32(ap), C32(x), incx);
}
HWYBLAS_API void ztpsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* ap, double* x,
                        const blasint* incx) {
  hwyblas::Tpmv("ZTPSV", true, uplo, trans, diag, n, C64(ap), C64(x), incx);
}

// SPR

HWYBLAS_API void sspr_(const char* uplo, const blasint* n, const float* alpha,
                       const float* x, const blasint* incx, float* ap) {
  hwyblas::Spr("SSPR", uplo, n, alpha, x, incx, ap);
}
HWYBLAS_API void dspr_(const char* uplo, const blasint* n, const double* alpha,
                       const double* x, const blasint* incx, double* ap) {
  hwyblas::Spr("DSPR", uplo, n, alpha, x, incx, ap);
}

// HPR

HWYBLAS_API void chpr_(const char* uplo, const blasint* n, const float* alpha,
                       const float* x, const blasint* incx, float* ap) {
  hwyblas::Spr("CHPR", uplo, n, alpha, C32(x), incx, C32(ap));
}
HWYBLAS_API void zhpr_(const char* uplo, const blasint* n, const double* alpha,
                       const double* x, const blasint* incx, double* ap) {
  hwyblas::Spr("ZHPR", uplo, n, alpha, C64(x), incx, C64(ap));
}

// SPR2

HWYBLAS_API void sspr2_(const char* uplo, const blasint* n, const float* alpha,
                        const float* x, const blasint* incx, const float* y,
                        const blasint* incy, float* ap) {
  hwyblas::Spr2("SSPR2", uplo, n, alpha, x, incx, y, incy, ap);
}
HWYBLAS_API void dspr2_(const char* uplo, const blasint* n, const double* alpha,
                        const double* x, const blasint* incx, const double* y,
                        const blasint* incy, double* ap) {
  hwyblas::Spr2("DSPR2", uplo, n, alpha, x, incx, y, incy, ap);
}

// HPR2

HWYBLAS_API void chpr2_(const char* uplo, const blasint* n, const float* alpha,
                        const float* x, const blasint* incx, const float* y,
                        const blasint* incy, float* ap) {
  hwyblas::Spr2("CHPR2", uplo, n, C32(alpha), C32(x), incx, C32(y), incy,
                C32(ap));
}
HWYBLAS_API void zhpr2_(const char* uplo, const blasint* n, const double* alpha,
                        const double* x, const blasint* incx, const double* y,
                        const blasint* incy, double* ap) {
  hwyblas::Spr2("ZHPR2", uplo, n, C64(alpha), C64(x), incx, C64(y), incy,
                C64(ap));
}

}  // extern "C"
#endif  // HWY_ONCE
