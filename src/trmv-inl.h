// TRMV kernel: x := op(A) * x for a triangular A, x contiguous.
//
// The product is accumulated into a zeroed temporary y and copied back. The
// matrix is processed in strips of 4 columns: a small diagonal block, done
// element by element, and the rectangle below (lower) or above (upper) it,
// done by the GEMV kernels (axpy form for op = N, dot form for T and C).
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_TRMV_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_TRMV_INL_H_
#undef HWYBLAS_SRC_TRMV_INL_H_
#else
#define HWYBLAS_SRC_TRMV_INL_H_
#endif

#include <stddef.h>
#include <string.h>

#include "common.h"
#include "hwy/highway.h"
#include "src/gemv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

// y[0:m] += R * t, where R is m x cols (axpy form).
template <typename T>
HWY_INLINE void RectAxpy(size_t m, size_t cols, const T* r, size_t lda,
                         const T* t, T* y) {
  if (m == 0) return;
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    GemvNComplex(m, cols, reinterpret_cast<const R*>(r), lda,
                 reinterpret_cast<const R*>(t), reinterpret_cast<R*>(y));
  } else {
    GemvNReal(m, cols, r, lda, t, y);
  }
}

// y[0:cols] += alpha * op(R)^T * x[0:m], where R is m x cols (dot form)
// and op conjugates if conj.
template <typename T>
HWY_INLINE void RectDot(bool conj, size_t m, size_t cols, T alpha, const T* r,
                        size_t lda, const T* x, T* y) {
  if (m == 0) return;
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    const R* rr = reinterpret_cast<const R*>(r);
    const R* xr = reinterpret_cast<const R*>(x);
    R* yr = reinterpret_cast<R*>(y);
    if (conj) {
      GemvTComplex<true>(m, cols, alpha, rr, lda, xr, yr);
    } else {
      GemvTComplex<false>(m, cols, alpha, rr, lda, xr, yr);
    }
  } else {
    GemvTReal(m, cols, alpha, r, lda, x, y);
  }
}

template <typename T>
void TrmvContiguous(bool lower, Op op, bool unit, size_t n, const T* a,
                    size_t lda, T* HWY_RESTRICT x) {
  constexpr size_t kStrip = 4;
  T* HWY_RESTRICT y =
      static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
  for (size_t i = 0; i < n; ++i) y[i] = T(0);
  const bool conj = (op == Op::kConjTrans);

  for (size_t j = 0; j < n; j += kStrip) {
    const size_t cols = HWY_MIN(kStrip, n - j);

    // Diagonal block: element (ii, jj) of A for ii, jj in [j, j + cols).
    for (size_t jj = j; jj < j + cols; ++jj) {
      for (size_t ii = j; ii < j + cols; ++ii) {
        if (lower ? (ii < jj) : (ii > jj)) continue;
        T v = (ii == jj && unit) ? T(1) : a[ii + jj * lda];
        if (op == Op::kNoTrans) {
          y[ii] += v * x[jj];
        } else {
          y[jj] += (conj ? Conj(v) : v) * x[ii];
        }
      }
    }

    // Rectangle of rows [row0, row0 + m) in columns [j, j + cols).
    const size_t row0 = lower ? j + cols : 0;
    const size_t m = lower ? n - row0 : j;
    const T* r = a + row0 + j * lda;
    if (op == Op::kNoTrans) {
      RectAxpy(m, cols, r, lda, x + j, y + row0);
    } else {
      RectDot(conj, m, cols, T(1), r, lda, x + row0, y + j);
    }
  }
  memcpy(x, y, n * sizeof(T));
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_TRMV_INL_H_
