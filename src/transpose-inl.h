// Transposed copy of a small column-major matrix, used to pack and to solve
// on transposed blocks.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_TRANSPOSE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_TRANSPOSE_INL_H_
#undef HWYBLAS_SRC_TRANSPOSE_INL_H_
#else
#define HWYBLAS_SRC_TRANSPOSE_INL_H_
#endif

#include <stddef.h>
#include <string.h>

#include "common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// dst(j, i) = src(i, j) for a rows x cols column-major matrix src.
template <typename T>
HWY_INLINE void Transpose(size_t rows, size_t cols, const T* HWY_RESTRICT src,
                          size_t lds, T* HWY_RESTRICT dst, size_t ldd) {
  if constexpr (IsComplex<T>::value) {
    // Element by element. The inner loop runs contiguously through the
    // matrix with the larger leading dimension, whose strided accesses would
    // be more likely to conflict in the cache.
    if (lds >= ldd) {
      for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) dst[j + i * ldd] = src[i + j * lds];
      }
    } else {
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) dst[j + i * ldd] = src[i + j * lds];
      }
    }
  } else {
    // Edges: element by element for i >= i0, j in [j0, j1).
    const auto elements = [&](size_t i0, size_t j0, size_t j1) {
      for (size_t i = i0; i < rows; ++i) {
        for (size_t j = j0; j < j1; ++j) dst[j + i * ldd] = src[i + j * lds];
      }
    };
    // N rows of 4 columns of src are loaded as 4 vectors and interleaved,
    // which yields each row's 4 elements contiguously: 4 elements of a
    // column of dst.
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    HWY_ALIGN T rows4[4 * hn::MaxLanes(d)];
    size_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      const T* s = src + j * lds;
      size_t i = 0;
      for (; i + N <= rows; i += N) {
        const V v0 = hn::LoadU(d, s + i);
        const V v1 = hn::LoadU(d, s + i + lds);
        const V v2 = hn::LoadU(d, s + i + 2 * lds);
        const V v3 = hn::LoadU(d, s + i + 3 * lds);
        hn::StoreInterleaved4(v0, v1, v2, v3, d, rows4);
        for (size_t k = 0; k < N; ++k) {
          memcpy(dst + j + (i + k) * ldd, rows4 + 4 * k, 4 * sizeof(T));
        }
      }
      elements(i, j, j + 4);
    }
    elements(0, j, cols);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_TRANSPOSE_INL_H_
