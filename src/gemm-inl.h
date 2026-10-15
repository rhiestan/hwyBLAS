// Blocked GEMM driver (the Goto / BLIS algorithm).
//
//   C := alpha * op(A) * op(B) + beta * C
//
// Loop structure, from outermost to innermost:
//
//   for jc in N step NC:                      (B block kc x nc -> L3)
//     for pc in K step KC:
//       pack B[pc:pc+kc, jc:jc+nc] into NR-column panels
//       for ic in M step MC:                  (A block mc x kc -> L2)
//         pack A[ic:ic+mc, pc:pc+kc] into MR-row panels
//         for jr in nc step NR:               (B panel kc x NR -> L1)
//           for ir in mc step MR:
//             micro-kernel: C[ir:ir+MR, jr:jr+NR] += A_panel * B_panel
//
// beta is applied during the first pass over K only (pc == 0); later passes
// accumulate into C with beta = 1.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_GEMM_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_GEMM_INL_H_
#undef HWYBLAS_SRC_GEMM_INL_H_
#else
#define HWYBLAS_SRC_GEMM_INL_H_
#endif

#include <stddef.h>

#include "common.h"
#include "gemm.h"
#include "hwy/highway.h"
#include "src/gemm_kernel-inl.h"
#include "src/pack-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

template <size_t kR, typename T>
HWY_INLINE void PackPanelsMaybeConj(bool conj, size_t rows, size_t kc,
                                    const T* src, ptrdiff_t rs, ptrdiff_t ks,
                                    T* dst) {
  if (IsComplex<T>::value && conj) {
    PackPanels<kR, true>(rows, kc, src, rs, ks, dst);
  } else {
    PackPanels<kR, false>(rows, kc, src, rs, ks, dst);
  }
}

// Requires m, n, k > 0 and alpha != 0 (callers handle the trivial cases).
template <typename T>
void BlockedGemm(const GemmProblem<T>& g) {
  using K = GemmKernel<T>;
  constexpr size_t MR = K::kMR, NR = K::kNR;
  constexpr size_t MC = K::kMC, KC = K::kKC, NC = K::kNC;
  static_assert(MC % MR == 0 && NC % NR == 0, "Blocks must be whole panels");

  // Packed buffers, sized for full blocks rounded up to whole panels.
  const size_t kc_max = HWY_MIN(KC, g.k);
  const size_t mc_max = HWY_MIN(MC, hwy::RoundUpTo(g.m, MR));
  const size_t nc_max = HWY_MIN(NC, hwy::RoundUpTo(g.n, NR));
  T* HWY_RESTRICT packed_a = static_cast<T*>(
      ScratchBuffer(kScratchPackA, mc_max * kc_max * sizeof(T)));
  T* HWY_RESTRICT packed_b = static_cast<T*>(
      ScratchBuffer(kScratchPackB, nc_max * kc_max * sizeof(T)));

  for (size_t jc = 0; jc < g.n; jc += NC) {
    const size_t nc = HWY_MIN(NC, g.n - jc);

    for (size_t pc = 0; pc < g.k; pc += KC) {
      const size_t kc = HWY_MIN(KC, g.k - pc);
      const T beta = (pc == 0) ? g.beta : T(1);

      // B[pc:pc+kc, jc:jc+nc]: panel "rows" are columns of B.
      PackPanelsMaybeConj<NR>(g.conj_b, nc, kc, g.b + pc * g.b_rs + jc * g.b_cs,
                              g.b_cs, g.b_rs, packed_b);

      for (size_t ic = 0; ic < g.m; ic += MC) {
        const size_t mc = HWY_MIN(MC, g.m - ic);

        // A[ic:ic+mc, pc:pc+kc].
        PackPanelsMaybeConj<MR>(g.conj_a, mc, kc,
                                g.a + ic * g.a_rs + pc * g.a_cs, g.a_rs, g.a_cs,
                                packed_a);

        for (size_t jr = 0; jr < nc; jr += NR) {
          const size_t nr = HWY_MIN(NR, nc - jr);
          const T* panel_b = packed_b + jr * kc;
          T* c_col = g.c + (jc + jr) * g.ldc + ic;

          for (size_t ir = 0; ir < mc; ir += MR) {
            const size_t mr = HWY_MIN(MR, mc - ir);
            K::Run(kc, packed_a + ir * kc, panel_b, g.alpha, beta, c_col + ir,
                   g.ldc, mr, nr);
          }
        }
      }
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_GEMM_INL_H_
