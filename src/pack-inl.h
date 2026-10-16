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
#include "gemm.h"
#include "hwy/highway.h"
#include "src/transpose-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

// Packed layout
// -------------
// A block of `rows` x kc elements is packed into panels of kR rows. Each
// panel is stored "k-major": kc consecutive groups of kR elements, one group
// per k index. Rows beyond `rows` in the last panel are zero, so micro-kernels
// can always process full panels:
//
//   panel q, step p, row r  ->  dst[q * kR * kc + p * kR + r]
//
// The same layout serves both GEMM operands:
//   - A (M x K): panel rows are rows of op(A),    kR = MR.
//   - B (K x N): panel rows are columns of op(B), kR = NR (i.e. it packs
//     op(B)^T).
// Conjugation is applied while packing, so micro-kernels never need to handle
// it.

// Packs one panel of r <= kR rows from a general strided matrix: element
// (i, p) is at src[i * rs + p * ks].
template <size_t kR, bool kConj, typename T>
HWY_INLINE void PackPanel(size_t r, size_t kc, const T* HWY_RESTRICT src,
                          ptrdiff_t rs, ptrdiff_t ks, T* HWY_RESTRICT dst) {
  const auto load = [](const T* p) { return kConj ? Conj(*p) : *p; };

  if (r == kR && rs == 1) {
    // Rows are contiguous in memory: copy kR elements per step.
    for (size_t p = 0; p < kc; ++p) {
      const T* HWY_RESTRICT s = src + static_cast<ptrdiff_t>(p) * ks;
      if constexpr (kConj) {
        for (size_t i = 0; i < kR; ++i) dst[i] = load(s + i);
      } else {
        memcpy(dst, s, kR * sizeof(T));
      }
      dst += kR;
    }
  } else if (r == kR && ks == 1) {
    // Each row is contiguous along k: the panel is the transpose of the
    // kc x kR column-major matrix with leading dimension rs. The vectorized
    // transpose handles 4 rows at a time and only pays off without leftover
    // rows (e.g. not for 6-row panels of B).
    if constexpr (!kConj && kR % 4 == 0) {
      if (rs > 0) {
        Transpose(kc, kR, src, static_cast<size_t>(rs), dst, kR);
        return;
      }
    }
    for (size_t p = 0; p < kc; ++p) {
      for (size_t i = 0; i < kR; ++i) {
        dst[i] = load(src + static_cast<ptrdiff_t>(i) * rs + p);
      }
      dst += kR;
    }
  } else {
    // General strides and/or partial panel (zero padded).
    for (size_t p = 0; p < kc; ++p) {
      const T* HWY_RESTRICT s = src + static_cast<ptrdiff_t>(p) * ks;
      size_t i = 0;
      for (; i < r; ++i) dst[i] = load(s + static_cast<ptrdiff_t>(i) * rs);
      for (; i < kR; ++i) dst[i] = T(0);
      dst += kR;
    }
  }
}

template <size_t kR, typename T>
HWY_INLINE void PackPanelMaybeConj(bool conj, size_t r, size_t kc, const T* src,
                                   ptrdiff_t rs, ptrdiff_t ks, T* dst) {
  if (IsComplex<T>::value && conj) {
    PackPanel<kR, true>(r, kc, src, rs, ks, dst);
  } else {
    PackPanel<kR, false>(r, kc, src, rs, ks, dst);
  }
}

// Returns element (i, p) of a structured operand. Only used for panels that
// cross the diagonal, so it favors clarity over speed.
template <typename T>
T StructuredElement(const Operand<T>& x, size_t i, size_t p) {
  const auto at = [&x](size_t row, size_t col) {
    const T v = x.data[static_cast<ptrdiff_t>(row) * x.rs +
                       static_cast<ptrdiff_t>(col) * x.cs];
    return x.conj ? Conj(v) : v;
  };
  if (i == p) {
    if (x.structure == Structure::kTriangular && x.unit_diag) return T(1);
    if (x.structure == Structure::kHermitian) return T(std::real(at(i, i)));
    return at(i, i);
  }
  const bool stored = x.lower ? (i > p) : (i < p);
  if (stored) return at(i, p);
  switch (x.structure) {
    case Structure::kSymmetric:
      return at(p, i);
    case Structure::kHermitian:
      return Conj(at(p, i));
    default:  // kTriangular
      return T(0);
  }
}

// Packs rows [row0, row0 + rows) x columns [p0, p0 + kc) of op(X) into panels
// of kR rows (see "Packed layout" above).
//
// For structured operands each panel is classified against the diagonal:
// panels entirely inside the stored triangle are packed directly, panels
// entirely in the other triangle are packed from the mirrored (transposed)
// storage, or as zeros for triangular operands, and only panels crossing the
// diagonal are packed element by element.
template <size_t kR, typename T>
HWY_NOINLINE void PackOperand(const Operand<T>& x, size_t row0, size_t rows,
                              size_t p0, size_t kc, T* HWY_RESTRICT dst) {
  for (size_t q = 0; q < rows; q += kR, dst += kR * kc) {
    const size_t r = HWY_MIN(kR, rows - q);
    const size_t i_begin = row0 + q, i_last = i_begin + r - 1;
    const size_t p_last = p0 + kc - 1;
    const T* direct = x.data + static_cast<ptrdiff_t>(i_begin) * x.rs +
                      static_cast<ptrdiff_t>(p0) * x.cs;

    if (x.structure == Structure::kGeneral) {
      PackPanelMaybeConj<kR>(x.conj, r, kc, direct, x.rs, x.cs, dst);
      continue;
    }

    // Entirely in the stored triangle (diagonal included)?
    const bool in_stored = x.lower ? (i_begin >= p_last) : (i_last <= p0);
    // Entirely strictly in the other triangle?
    const bool in_other = x.lower ? (i_last < p0) : (i_begin > p_last);
    const bool has_diagonal = i_begin <= p_last && p0 <= i_last;

    if (in_stored && !has_diagonal) {
      PackPanelMaybeConj<kR>(x.conj, r, kc, direct, x.rs, x.cs, dst);
    } else if (in_other && x.structure == Structure::kTriangular) {
      memset(static_cast<void*>(dst), 0, kR * kc * sizeof(T));
    } else if (in_other) {
      // Mirrored: element (i, p) is stored at (p, i).
      const T* mirrored = x.data + static_cast<ptrdiff_t>(i_begin) * x.cs +
                          static_cast<ptrdiff_t>(p0) * x.rs;
      const bool conj = x.conj != (x.structure == Structure::kHermitian);
      PackPanelMaybeConj<kR>(conj, r, kc, mirrored, x.cs, x.rs, dst);
    } else {
      for (size_t p = 0; p < kc; ++p) {
        size_t i = 0;
        for (; i < r; ++i) {
          dst[p * kR + i] = StructuredElement(x, i_begin + i, p0 + p);
        }
        for (; i < kR; ++i) dst[p * kR + i] = T(0);
      }
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_PACK_INL_H_
