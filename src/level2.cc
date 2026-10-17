// Level 2 routines besides GEMV: SYMV, HEMV, TRMV, TRSV and the rank-1 and
// rank-2 updates GER, SYR, HER, SYR2 and HER2.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/level2.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/rank_update-inl.h"
#include "src/symv-inl.h"
#include "src/trmv-inl.h"
#include "src/trsv-inl.h"

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
  }                                                                            \
  void Trsv##SUFFIX(bool lower, Op op, bool unit, size_t n, const T* a,        \
                    size_t lda, T* x) {                                        \
    TrsvContiguous(lower, op, unit, n, a, lda, x);                             \
  }                                                                            \
  void Ger##SUFFIX(size_t m, size_t n, T* a, size_t lda, const T* u,           \
                   const T* s) {                                               \
    RankUpdate<1>(m, n, a, lda, u, s, u, s);                                   \
  }                                                                            \
  void Syr##SUFFIX(size_t terms, bool lower, size_t n, T* a, size_t lda,       \
                   const T* u0, const T* s0, const T* u1, const T* s1) {       \
    constexpr bool kHermitian = IsComplex<T>::value;                           \
    if (terms == 1) {                                                          \
      TriangularRankUpdate<1>(lower, kHermitian, n, a, lda, u0, s0, u0, s0);   \
    } else {                                                                   \
      TriangularRankUpdate<2>(lower, kHermitian, n, a, lda, u0, s0, u1, s1);   \
    }                                                                          \
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
HWYBLAS_EXPORT_BY_TYPE(Trsv);
HWYBLAS_EXPORT_BY_TYPE(Ger);
HWYBLAS_EXPORT_BY_TYPE(Syr);

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

// TRMV (solve = false): x := op(A) * x, and
// TRSV (solve = true): solves op(A) * x = b, x overwrites b.
template <typename T>
void Triangular(const char* name, bool solve, const char* uplo_ptr,
                const char* trans_ptr, const char* diag_ptr,
                const blasint* n_ptr, const T* a, const blasint* lda_ptr, T* x,
                const blasint* incx_ptr) {
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
  if (solve) {
    HWYBLAS_DISPATCH_BY_TYPE(T, Trsv, lower, op, unit, n, a, lda, xc);
  } else {
    HWYBLAS_DISPATCH_BY_TYPE(T, Trmv, lower, op, unit, n, a, lda, xc);
  }
  FinishOutput(xc, x, n, incx);
}

// GER, GERU (conj = false) and GERC (conj = true):
// A := alpha * x * y^T + A, or alpha * x * y^H + A.
template <typename T>
void Ger(const char* name, bool conj, const blasint* m_ptr,
         const blasint* n_ptr, const T* alpha_ptr, const T* x,
         const blasint* incx_ptr, const T* y, const blasint* incy_ptr, T* a,
         const blasint* lda_ptr) {
  const blasint m = *m_ptr, n = *n_ptr, lda = *lda_ptr;
  const blasint incx = *incx_ptr, incy = *incy_ptr;
  blasint info = 0;
  if (m < 0) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx == 0) {
    info = 5;
  } else if (incy == 0) {
    info = 7;
  } else if (lda < HWY_MAX(1, m)) {
    info = 9;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr;
  if (m == 0 || n == 0 || IsZero(alpha)) return;

  const T* xc = ContiguousInput(x, m, incx, kScratchX);
  // Row scales s = alpha * op(y).
  T* s = static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
  const T* y0 = VectorStart(y, n, incy);
  for (blasint j = 0; j < n; ++j) {
    const T yj = y0[static_cast<ptrdiff_t>(j) * incy];
    s[j] = alpha * (conj ? Conj(yj) : yj);
  }
  HWYBLAS_DISPATCH_BY_TYPE(T, Ger, m, n, a, lda, xc, s);
}

// SYR (real T): A := alpha * x * x^T + A, and
// HER (complex T, real alpha): A := alpha * x * x^H + A.
template <typename T>
void Syr(const char* name, const char* uplo_ptr, const blasint* n_ptr,
         const RealT<T>* alpha_ptr, const T* x, const blasint* incx_ptr, T* a,
         const blasint* lda_ptr) {
  const blasint n = *n_ptr, lda = *lda_ptr, incx = *incx_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo_ptr, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx == 0) {
    info = 5;
  } else if (lda < HWY_MAX(1, n)) {
    info = 7;
  }
  if (info != 0) return ReportError(name, info);

  const RealT<T> alpha = *alpha_ptr;
  if (n == 0 || alpha == 0) return;

  const T* xc = ContiguousInput(x, n, incx, kScratchX);
  // Row scales s = alpha * conj(x) (Conj is the identity for real T).
  T* s = static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
  for (blasint j = 0; j < n; ++j) s[j] = alpha * Conj(xc[j]);
  HWYBLAS_DISPATCH_BY_TYPE(T, Syr, 1, lower, n, a, lda, xc, s, xc, s);
}

// SYR2 (real T): A := alpha * x * y^T + alpha * y * x^T + A, and
// HER2 (complex T): A := alpha * x * y^H + conj(alpha) * y * x^H + A.
template <typename T>
void Syr2(const char* name, const char* uplo_ptr, const blasint* n_ptr,
          const T* alpha_ptr, const T* x, const blasint* incx_ptr, const T* y,
          const blasint* incy_ptr, T* a, const blasint* lda_ptr) {
  const blasint n = *n_ptr, lda = *lda_ptr;
  const blasint incx = *incx_ptr, incy = *incy_ptr;
  bool lower;
  blasint info = 0;
  if (!DecodeUplo(*uplo_ptr, &lower)) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx == 0) {
    info = 5;
  } else if (incy == 0) {
    info = 7;
  } else if (lda < HWY_MAX(1, n)) {
    info = 9;
  }
  if (info != 0) return ReportError(name, info);

  const T alpha = *alpha_ptr;
  if (n == 0 || IsZero(alpha)) return;

  const T* xc = ContiguousInput(x, n, incx, kScratchX);
  const T* yc = ContiguousInput(y, n, incy, kScratchY);
  // Row scales s0 = alpha * conj(y) and s1 = conj(alpha * x).
  T* s0 = static_cast<T*>(ScratchBuffer(kScratchTemp, 2 * sizeof(T) * n));
  T* s1 = s0 + n;
  for (blasint j = 0; j < n; ++j) {
    s0[j] = alpha * Conj(yc[j]);
    s1[j] = Conj(alpha * xc[j]);
  }
  HWYBLAS_DISPATCH_BY_TYPE(T, Syr, 2, lower, n, a, lda, xc, s0, yc, s1);
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
  hwyblas::Triangular("STRMV", false, uplo, trans, diag, n, a, lda, x, incx);
}
HWYBLAS_API void dtrmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* a, const blasint* lda,
                        double* x, const blasint* incx) {
  hwyblas::Triangular("DTRMV", false, uplo, trans, diag, n, a, lda, x, incx);
}
HWYBLAS_API void ctrmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* a, const blasint* lda,
                        float* x, const blasint* incx) {
  hwyblas::Triangular("CTRMV", false, uplo, trans, diag, n, C32(a), lda, C32(x),
                      incx);
}
HWYBLAS_API void ztrmv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* a, const blasint* lda,
                        double* x, const blasint* incx) {
  hwyblas::Triangular("ZTRMV", false, uplo, trans, diag, n, C64(a), lda, C64(x),
                      incx);
}

// TRSV

HWYBLAS_API void strsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* a, const blasint* lda,
                        float* x, const blasint* incx) {
  hwyblas::Triangular("STRSV", true, uplo, trans, diag, n, a, lda, x, incx);
}
HWYBLAS_API void dtrsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* a, const blasint* lda,
                        double* x, const blasint* incx) {
  hwyblas::Triangular("DTRSV", true, uplo, trans, diag, n, a, lda, x, incx);
}
HWYBLAS_API void ctrsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const float* a, const blasint* lda,
                        float* x, const blasint* incx) {
  hwyblas::Triangular("CTRSV", true, uplo, trans, diag, n, C32(a), lda, C32(x),
                      incx);
}
HWYBLAS_API void ztrsv_(const char* uplo, const char* trans, const char* diag,
                        const blasint* n, const double* a, const blasint* lda,
                        double* x, const blasint* incx) {
  hwyblas::Triangular("ZTRSV", true, uplo, trans, diag, n, C64(a), lda, C64(x),
                      incx);
}

// GER (real), GERU and GERC (complex)

HWYBLAS_API void sger_(const blasint* m, const blasint* n, const float* alpha,
                       const float* x, const blasint* incx, const float* y,
                       const blasint* incy, float* a, const blasint* lda) {
  hwyblas::Ger("SGER", false, m, n, alpha, x, incx, y, incy, a, lda);
}
HWYBLAS_API void dger_(const blasint* m, const blasint* n, const double* alpha,
                       const double* x, const blasint* incx, const double* y,
                       const blasint* incy, double* a, const blasint* lda) {
  hwyblas::Ger("DGER", false, m, n, alpha, x, incx, y, incy, a, lda);
}
HWYBLAS_API void cgeru_(const blasint* m, const blasint* n, const float* alpha,
                        const float* x, const blasint* incx, const float* y,
                        const blasint* incy, float* a, const blasint* lda) {
  hwyblas::Ger("CGERU", false, m, n, C32(alpha), C32(x), incx, C32(y), incy,
               C32(a), lda);
}
HWYBLAS_API void cgerc_(const blasint* m, const blasint* n, const float* alpha,
                        const float* x, const blasint* incx, const float* y,
                        const blasint* incy, float* a, const blasint* lda) {
  hwyblas::Ger("CGERC", true, m, n, C32(alpha), C32(x), incx, C32(y), incy,
               C32(a), lda);
}
HWYBLAS_API void zgeru_(const blasint* m, const blasint* n, const double* alpha,
                        const double* x, const blasint* incx, const double* y,
                        const blasint* incy, double* a, const blasint* lda) {
  hwyblas::Ger("ZGERU", false, m, n, C64(alpha), C64(x), incx, C64(y), incy,
               C64(a), lda);
}
HWYBLAS_API void zgerc_(const blasint* m, const blasint* n, const double* alpha,
                        const double* x, const blasint* incx, const double* y,
                        const blasint* incy, double* a, const blasint* lda) {
  hwyblas::Ger("ZGERC", true, m, n, C64(alpha), C64(x), incx, C64(y), incy,
               C64(a), lda);
}

// SYR (real) and HER (complex)

HWYBLAS_API void ssyr_(const char* uplo, const blasint* n, const float* alpha,
                       const float* x, const blasint* incx, float* a,
                       const blasint* lda) {
  hwyblas::Syr("SSYR", uplo, n, alpha, x, incx, a, lda);
}
HWYBLAS_API void dsyr_(const char* uplo, const blasint* n, const double* alpha,
                       const double* x, const blasint* incx, double* a,
                       const blasint* lda) {
  hwyblas::Syr("DSYR", uplo, n, alpha, x, incx, a, lda);
}
HWYBLAS_API void cher_(const char* uplo, const blasint* n, const float* alpha,
                       const float* x, const blasint* incx, float* a,
                       const blasint* lda) {
  hwyblas::Syr("CHER", uplo, n, alpha, C32(x), incx, C32(a), lda);
}
HWYBLAS_API void zher_(const char* uplo, const blasint* n, const double* alpha,
                       const double* x, const blasint* incx, double* a,
                       const blasint* lda) {
  hwyblas::Syr("ZHER", uplo, n, alpha, C64(x), incx, C64(a), lda);
}

// SYR2 (real) and HER2 (complex)

HWYBLAS_API void ssyr2_(const char* uplo, const blasint* n, const float* alpha,
                        const float* x, const blasint* incx, const float* y,
                        const blasint* incy, float* a, const blasint* lda) {
  hwyblas::Syr2("SSYR2", uplo, n, alpha, x, incx, y, incy, a, lda);
}
HWYBLAS_API void dsyr2_(const char* uplo, const blasint* n, const double* alpha,
                        const double* x, const blasint* incx, const double* y,
                        const blasint* incy, double* a, const blasint* lda) {
  hwyblas::Syr2("DSYR2", uplo, n, alpha, x, incx, y, incy, a, lda);
}
HWYBLAS_API void cher2_(const char* uplo, const blasint* n, const float* alpha,
                        const float* x, const blasint* incx, const float* y,
                        const blasint* incy, float* a, const blasint* lda) {
  hwyblas::Syr2("CHER2", uplo, n, C32(alpha), C32(x), incx, C32(y), incy,
                C32(a), lda);
}
HWYBLAS_API void zher2_(const char* uplo, const blasint* n, const double* alpha,
                        const double* x, const blasint* incx, const double* y,
                        const blasint* incy, double* a, const blasint* lda) {
  hwyblas::Syr2("ZHER2", uplo, n, C64(alpha), C64(x), incx, C64(y), incy,
                C64(a), lda);
}

}  // extern "C"
#endif  // HWY_ONCE
