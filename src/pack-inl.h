// Packing of matrix blocks into the contiguous panel layout consumed by the
// GEMM micro-kernels.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_PACK_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_PACK_INL_H_
#undef HWYBLAS_SRC_PACK_INL_H_
#else
#define HWYBLAS_SRC_PACK_INL_H_
#endif

#include <stddef.h>
#include <string.h>

#include "common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

// Packs a `rows` x `kc` block of a strided matrix into panels of kR rows.
//
// Source element (i, p) is at src[i * rs + p * ks]. Panel q holds rows
// [q * kR, q * kR + kR) stored "k-major": kc consecutive groups of kR
// elements, one group per p. Rows beyond `rows` in the last panel are filled
// with zeros, so micro-kernels can always process full panels.
//
//   panel q, step p, row r  ->  dst[q * kR * kc + p * kR + r]
//
// The same routine packs both GEMM operands:
//   - A (M x K): panel rows are rows of A,    kR = MR.
//   - B (K x N): panel rows are columns of B, kR = NR (i.e. it packs B^T).
//
// If kConj, complex elements are conjugated while packing so that the
// micro-kernels never need to handle conjugation.
template <size_t kR, bool kConj, typename T>
HWY_NOINLINE void PackPanels(size_t rows, size_t kc, const T* HWY_RESTRICT src,
                             ptrdiff_t rs, ptrdiff_t ks, T* HWY_RESTRICT dst) {
  const auto load = [](const T* p) { return kConj ? Conj(*p) : *p; };

  for (size_t i0 = 0; i0 < rows; i0 += kR) {
    const size_t r = HWY_MIN(kR, rows - i0);
    const T* HWY_RESTRICT panel_src = src + static_cast<ptrdiff_t>(i0) * rs;

    if (r == kR && rs == 1) {
      // Rows are contiguous in memory: copy kR elements per step.
      for (size_t p = 0; p < kc; ++p) {
        const T* HWY_RESTRICT s = panel_src + static_cast<ptrdiff_t>(p) * ks;
        if constexpr (kConj) {
          for (size_t i = 0; i < kR; ++i) dst[i] = load(s + i);
        } else {
          memcpy(dst, s, kR * sizeof(T));
        }
        dst += kR;
      }
    } else if (r == kR && ks == 1) {
      // Each row is contiguous along k: gather one element from each row.
      for (size_t p = 0; p < kc; ++p) {
        for (size_t i = 0; i < kR; ++i) {
          dst[i] = load(panel_src + static_cast<ptrdiff_t>(i) * rs + p);
        }
        dst += kR;
      }
    } else {
      // General strides and/or partial panel (zero padded).
      for (size_t p = 0; p < kc; ++p) {
        const T* HWY_RESTRICT s = panel_src + static_cast<ptrdiff_t>(p) * ks;
        size_t i = 0;
        for (; i < r; ++i) dst[i] = load(s + static_cast<ptrdiff_t>(i) * rs);
        for (; i < kR; ++i) dst[i] = T(0);
        dst += kR;
      }
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_PACK_INL_H_
