// TRSV kernel: solves op(A) * x = b for a triangular A, x contiguous (x holds
// b on entry and the solution on exit).
//
// The matrix is processed in strips of 4 columns, in the order in which the
// unknowns are solved. Each strip consists of a small diagonal block, solved
// element by element, and the rectangle below (lower) or above (upper) it,
// which is applied with the GEMV kernels:
//   - op = N (axpy form): after solving the block's unknowns, subtract the
//     rectangle times them from the unknowns not yet solved;
//   - op = T, C (dot form): before solving the block's unknowns, subtract
//     the dot products of the rectangle with the unknowns already solved.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_TRSV_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_TRSV_INL_H_
#undef HWYBLAS_SRC_TRSV_INL_H_
#else
#define HWYBLAS_SRC_TRSV_INL_H_
#endif

#include <stddef.h>

#include "common.h"
#include "hwy/highway.h"
#include "src/trmv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

template <typename T>
void TrsvContiguous(bool lower, Op op, bool unit, size_t n, const T* a,
                    size_t lda, T* HWY_RESTRICT x) {
  constexpr size_t kStrip = 4;
  const bool conj = (op == Op::kConjTrans);
  // op(A) is lower triangular: solve for increasing indices.
  const bool forward = (lower == (op == Op::kNoTrans));
  // Element (p, q) of op(A).
  const auto op_a = [&](size_t p, size_t q) {
    if (op == Op::kNoTrans) return a[p + q * lda];
    const T v = a[q + p * lda];
    return conj ? Conj(v) : v;
  };

  const size_t num_strips = (n + kStrip - 1) / kStrip;
  for (size_t s = 0; s < num_strips; ++s) {
    const size_t j = (forward ? s : num_strips - 1 - s) * kStrip;
    const size_t cols = HWY_MIN(kStrip, n - j);
    // Rectangle of A: rows [row0, row0 + m) in columns [j, j + cols).
    const size_t row0 = lower ? j + cols : 0;
    const size_t m = lower ? n - row0 : j;
    const T* r = a + row0 + j * lda;

    if (op != Op::kNoTrans) {
      RectDot(conj, m, cols, T(-1), r, lda, x + row0, x + j);
    }
    // Diagonal block: substitution in solve order.
    for (size_t step = 0; step < cols; ++step) {
      const size_t p = forward ? j + step : j + cols - 1 - step;
      T sum = x[p];
      if (forward) {
        for (size_t q = j; q < p; ++q) sum -= op_a(p, q) * x[q];
      } else {
        for (size_t q = p + 1; q < j + cols; ++q) sum -= op_a(p, q) * x[q];
      }
      x[p] = unit ? sum : sum / op_a(p, p);
    }
    if (op == Op::kNoTrans) {
      T t[kStrip];
      for (size_t c = 0; c < cols; ++c) t[c] = -x[j + c];
      RectAxpy(m, cols, r, lda, t, x + row0);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_TRSV_INL_H_
