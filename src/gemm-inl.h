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
// When m is small, B is not packed: the micro-kernels read op(B) in place
// (see UseDirectB), which makes small products much cheaper.
//
// The same driver serves the other level 3 routines (see GemmProblem):
//   - Symmetric/Hermitian operands (SYMM, HEMM) only change the packing.
//   - Triangular operands (TRMM) also restrict, per micro-tile, the range of
//     k to the part where the operand is non-zero, skipping about half of
//     the work.
//   - Triangular C (SYRK, HERK) skips micro-tiles outside the triangle;
//     tiles crossing the diagonal are computed into a temporary and merged
//     element-wise.
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

// Returns op(B)^T as an operand, so that B can be packed with the same
// routine as A (panel rows are then the columns of op(B)).
template <typename T>
Operand<T> Transposed(Operand<T> x) {
  const ptrdiff_t rs = x.rs;
  x.rs = x.cs;
  x.cs = rs;
  x.lower = !x.lower;
  return x;
}

// Range [begin, end) of k for which the product of rows [i0, i0 + mr) of
// op(A) and columns [j0, j0 + nr) of op(B) can be non-zero. This is all of k
// unless an operand is triangular.
template <typename T>
HWY_INLINE void NonZeroRange(const GemmProblem<T>& g, size_t i0, size_t mr,
                             size_t j0, size_t nr, size_t* begin, size_t* end) {
  *begin = 0;
  *end = g.k;
  if (g.a.structure == Structure::kTriangular) {
    // Row i of a lower triangular op(A) is zero beyond column i.
    if (g.a.lower) {
      *end = HWY_MIN(*end, i0 + mr);
    } else {
      *begin = HWY_MAX(*begin, i0);
    }
  }
  if (g.b.structure == Structure::kTriangular) {
    // Column j of a lower triangular op(B) is zero above row j.
    if (g.b.lower) {
      *begin = HWY_MAX(*begin, j0);
    } else {
      *end = HWY_MIN(*end, j0 + nr);
    }
  }
}

// How a micro-tile of C relates to the updated part of C. Tiles containing
// a diagonal element always count as crossing, so that real_diagonal can be
// applied element-wise.
enum class TileCoverage { kOutside, kInside, kCrossing };

HWY_INLINE TileCoverage Coverage(Part part, size_t i0, size_t mr, size_t j0,
                                 size_t nr) {
  const size_t i_last = i0 + mr - 1, j_last = j0 + nr - 1;
  switch (part) {
    case Part::kLower:  // i >= j
      if (i_last < j0) return TileCoverage::kOutside;
      return (i0 > j_last) ? TileCoverage::kInside : TileCoverage::kCrossing;
    case Part::kUpper:  // i <= j
      if (i0 > j_last) return TileCoverage::kOutside;
      return (i_last < j0) ? TileCoverage::kInside : TileCoverage::kCrossing;
    default:
      return TileCoverage::kInside;
  }
}

// Merges a tile computed into `tmp` (column-major, leading dimension ld_tmp,
// already scaled by alpha) into the triangle `part` of C, applying beta.
template <typename T>
void MergeTile(const T* tmp, size_t ld_tmp, Part part, bool real_diagonal,
               T beta, T* c, size_t ldc, size_t i0, size_t mr, size_t j0,
               size_t nr) {
  for (size_t j = 0; j < nr; ++j) {
    for (size_t i = 0; i < mr; ++i) {
      const size_t gi = i0 + i, gj = j0 + j;
      if (part == Part::kLower ? (gi < gj) : (gi > gj)) continue;
      T& cij = c[i + j * ldc];
      T v = tmp[i + j * ld_tmp];
      if (real_diagonal && gi == gj) {
        // HERK: the imaginary part of the diagonal is neither read nor kept.
        v = T(std::real(v));
        if (!IsZero(beta)) v += beta * T(std::real(cij));
      } else if (!IsZero(beta)) {
        v += beta * cij;
      }
      cij = v;
    }
  }
}

// Whether the micro-kernels read op(B) in place instead of packing it.
//
// Packing B costs O(k * n) and pays off over the m / MR micro-tiles that
// reuse each panel, so it is skipped when m is small. Without packing, a
// k step of the kernel reads one element from each of NR columns of op(B):
// with a unit k stride (op(B) = B) these are NR sequential streams, which
// the caches handle well even for large m. Otherwise each k step reads a
// different row of B; with a power-of-two leading dimension those rows map
// to few L1 sets and evict each other, so this is limited to smaller m.
// The kernels do not conjugate, so a conjugated op(B) is always packed.
template <typename T>
HWY_INLINE bool UseDirectB(const GemmProblem<T>& g) {
  if (g.b.structure != Structure::kGeneral || g.part != Part::kAll ||
      (IsComplex<T>::value && g.b.conj)) {
    return false;
  }
  return g.m <= (g.b.rs == 1 ? 256 : 64);
}

// The driver; kDirectB selects whether op(B) is read in place (see
// UseDirectB). Separate instantiations keep each one's code small.
template <bool kDirectB, typename T>
HWY_NOINLINE void BlockedGemmImpl(const GemmProblem<T>& g) {
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
  T* HWY_RESTRICT packed_b =
      kDirectB ? nullptr
               : static_cast<T*>(
                     ScratchBuffer(kScratchPackB, nc_max * kc_max * sizeof(T)));
  const Operand<T> bt = Transposed(g.b);
  HWY_ALIGN T tmp[MR * NR];  // for micro-tiles crossing the diagonal of C

  for (size_t jc = 0; jc < g.n; jc += NC) {
    const size_t nc = HWY_MIN(NC, g.n - jc);

    for (size_t pc = 0; pc < g.k; pc += KC) {
      const size_t kc = HWY_MIN(KC, g.k - pc);
      const T beta = (pc == 0) ? g.beta : T(1);

      // op(B)[pc:pc+kc, jc:jc+nc]: panel rows are columns of op(B).
      if constexpr (!kDirectB) PackOperand<NR>(bt, jc, nc, pc, kc, packed_b);

      for (size_t ic = 0; ic < g.m; ic += MC) {
        const size_t mc = HWY_MIN(MC, g.m - ic);

        // op(A)[ic:ic+mc, pc:pc+kc].
        PackOperand<MR>(g.a, ic, mc, pc, kc, packed_a);

        for (size_t jr = 0; jr < nc; jr += NR) {
          const size_t nr = HWY_MIN(NR, nc - jr);
          const size_t j0 = jc + jr;  // global column
          const T* panel_b = kDirectB ? nullptr : packed_b + jr * kc;
          T* c_col = g.c + j0 * g.ldc + ic;

          for (size_t ir = 0; ir < mc; ir += MR) {
            const size_t mr = HWY_MIN(MR, mc - ir);
            const size_t i0 = ic + ir;  // global row
            const TileCoverage coverage = Coverage(g.part, i0, mr, j0, nr);
            if (coverage == TileCoverage::kOutside) continue;

            // Restrict k to [pc, pc + kc) intersected with the non-zero
            // range; an empty range still applies beta.
            size_t k_begin, k_end;
            NonZeroRange(g, i0, mr, j0, nr, &k_begin, &k_end);
            k_begin = HWY_MAX(k_begin, pc) - pc;
            k_end = HWY_MIN(k_end, pc + kc);
            k_end = (k_end > pc + k_begin) ? k_end - pc : k_begin;
            if (k_end == k_begin && pc != 0) continue;  // nothing to add

            const T* a_panel = packed_a + ir * kc + k_begin * MR;
            const T* b_panel = kDirectB ? nullptr : panel_b + k_begin * NR;
            const size_t k_len = k_end - k_begin;
            if constexpr (kDirectB) {  // implies coverage == kInside
              const T* b_block = g.b.data +
                                 static_cast<ptrdiff_t>(pc + k_begin) * g.b.rs +
                                 static_cast<ptrdiff_t>(j0) * g.b.cs;
              K::RunDirectB(k_len, a_panel, b_block, g.b.rs, g.b.cs, g.alpha,
                            beta, c_col + ir, g.ldc, mr, nr);
            } else if (coverage == TileCoverage::kInside) {
              K::Run(k_len, a_panel, b_panel, g.alpha, beta, c_col + ir, g.ldc,
                     mr, nr);
            } else {
              K::Run(k_len, a_panel, b_panel, g.alpha, T(0), tmp, MR, MR, NR);
              MergeTile(tmp, MR, g.part, g.real_diagonal, beta, c_col + ir,
                        g.ldc, i0, mr, j0, nr);
            }
          }
        }
      }
    }
  }
}

// Requires m, n, k > 0 and alpha != 0 (callers handle the trivial cases).
template <typename T>
void BlockedGemm(const GemmProblem<T>& g) {
  if (UseDirectB(g)) {
    BlockedGemmImpl<true>(g);
  } else {
    BlockedGemmImpl<false>(g);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_GEMM_INL_H_
