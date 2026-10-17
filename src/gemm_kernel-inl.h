// GEMM micro-kernels: compute one MR x NR tile of C from packed panels.
//
//   C[0:m, 0:n] = alpha * (A_panel * B_panel) + beta * C[0:m, 0:n]
//
// where A_panel is MR x kc and B_panel is kc x NR, both packed by PackPanels
// (zero padded, so the kernels always compute a full MR x NR tile in
// registers; only the final write-back is restricted to m <= MR, n <= NR).
//
// Vectors run along M (down the columns of C), which matches column-major
// storage of C: each column of the tile is kMV vectors, loaded and stored
// with plain unaligned loads/stores. Elements of B are broadcast.
//
// The accumulators are small arrays of vectors indexed by loops with
// compile-time trip counts; these loops are fully unrolled so that every
// accumulator lives in its own register.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_GEMM_KERNEL_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_GEMM_KERNEL_INL_H_
#undef HWYBLAS_SRC_GEMM_KERNEL_INL_H_
#else
#define HWYBLAS_SRC_GEMM_KERNEL_INL_H_
#endif

#include <stddef.h>

#include <complex>
#include <type_traits>
#include <utility>

#include "common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// The tile shapes below rely on vector lengths known at compile time.
static_assert(!HWY_HAVE_SCALABLE, "Scalable vector targets are not supported");

// Number of architectural vector registers. The tile shape is chosen so that
// the accumulators plus a few temporaries fit without spilling.
#if (HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3) || HWY_ARCH_ARM
constexpr size_t kVectorRegs = 32;
#else
constexpr size_t kVectorRegs = 16;
#endif

// ---------------------------------------------------------------------------
// Real micro-kernel (float, double)
//
// Tile: MR = kMV vectors (along M) x NR columns. Per k step it loads kMV
// vectors of A, broadcasts NR elements of B and issues kMV * NR FMAs.
//
// Edge tiles with fewer than MR rows or NR columns run a smaller
// instantiation of the same loop, with only as many row vectors and columns
// as needed, so that small matrices do not pay for a padded full tile.

template <typename T>
struct RealGemmKernel {
  using D = hn::ScalableTag<T>;
  using V = hn::Vec<D>;
  static constexpr size_t kN = hn::MaxLanes(D());  // lanes per vector
  static constexpr size_t kMV = 2;                 // vectors per column
  static constexpr size_t kMR = kMV * kN;
  // Accumulators: kMV * kNR registers (12 of 16, or 24 of 32).
  static constexpr size_t kNR = kVectorRegs >= 32 ? 12 : 6;

  // Cache blocking (in elements): the kc x NR panel of B stays in L1, the
  // mc x kc block of A in L2, and the kc x nc block of B in L3.
  static constexpr size_t kKC = sizeof(T) == 4 ? 384 : 256;
  static constexpr size_t kMC = (sizeof(T) == 4 ? 144 : 96) / kMR * kMR;
  static constexpr size_t kNC = 4080 / kNR * kNR;

  static HWY_INLINE void Run(size_t kc, const T* HWY_RESTRICT a,
                             const T* HWY_RESTRICT b, T alpha, T beta,
                             T* HWY_RESTRICT c, size_t ldc, size_t m,
                             size_t n) {
    Dispatch<false>(kc, a, b, 0, 0, alpha, beta, c, ldc, m, n);
  }

  // As Run, but B is read in place instead of from a packed panel: element
  // (p, j) of the kc x n block of op(B) is b[p * b_ks + j * b_cs].
  static HWY_INLINE void RunDirectB(size_t kc, const T* HWY_RESTRICT a,
                                    const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                                    ptrdiff_t b_cs, T alpha, T beta,
                                    T* HWY_RESTRICT c, size_t ldc, size_t m,
                                    size_t n) {
    // A unit k stride is common (op(B) = B) and worth a constant.
    if (b_ks == 1) {
      Dispatch<true>(kc, a, b, 1, b_cs, alpha, beta, c, ldc, m, n);
    } else {
      Dispatch<true>(kc, a, b, b_ks, b_cs, alpha, beta, c, ldc, m, n);
    }
  }

 private:
  // Runs the instantiation for an m x n tile (m <= kMR, n <= kNR).
  template <bool kDirectB>
  static HWY_INLINE void Dispatch(size_t kc, const T* HWY_RESTRICT a,
                                  const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                                  ptrdiff_t b_cs, T alpha, T beta,
                                  T* HWY_RESTRICT c, size_t ldc, size_t m,
                                  size_t n) {
    static_assert(kMV == 2, "Dispatch assumes two vectors per column");
    const auto cols = std::make_index_sequence<kNR - 1>();
    if (HWY_LIKELY(m > kN)) {
      DispatchCols<kDirectB, 2>(cols, kc, a, b, b_ks, b_cs, alpha, beta, c, ldc,
                                m, n);
    } else {
      DispatchCols<kDirectB, 1>(cols, kc, a, b, b_ks, b_cs, alpha, beta, c, ldc,
                                m, n);
    }
  }

  // Runs Tile<..., kCols = n>; kJ + 1 enumerates the partial column counts.
  template <bool kDirectB, size_t kRowV, size_t... kJ>
  static HWY_INLINE void DispatchCols(std::index_sequence<kJ...> /*cols*/,
                                      size_t kc, const T* HWY_RESTRICT a,
                                      const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                                      ptrdiff_t b_cs, T alpha, T beta,
                                      T* HWY_RESTRICT c, size_t ldc, size_t m,
                                      size_t n) {
    if (HWY_LIKELY(n == kNR)) {
      return Tile<kDirectB, kRowV, kNR>(kc, a, b, b_ks, b_cs, alpha, beta, c,
                                        ldc, m);
    }
    (void)((n == kJ + 1 && (Tile<kDirectB, kRowV, kJ + 1>(
                                kc, a, b, b_ks, b_cs, alpha, beta, c, ldc, m),
                            true)) ||
           ...);
  }

  // Computes an m x kCols tile with kRowV vectors per column, where
  // (kRowV - 1) * kN < m <= kRowV * kN. Panels of A have kMR rows; only the
  // first kRowV vectors of each step are read.
  template <bool kDirectB, size_t kRowV, size_t kCols>
  static HWY_INLINE void Tile(size_t kc, const T* HWY_RESTRICT a,
                              const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                              ptrdiff_t b_cs, T alpha, T beta,
                              T* HWY_RESTRICT c, size_t ldc, size_t m) {
    const D d;

    // Direct B: one pointer per column.
    const T* HWY_RESTRICT b_col[kCols];
    if constexpr (kDirectB) {
      HWY_UNROLL(16)
      for (size_t j = 0; j < kCols; ++j) {
        b_col[j] = b + static_cast<ptrdiff_t>(j) * b_cs;
      }
    }

    V acc[kCols][kRowV];
    HWY_UNROLL(16)
    for (size_t j = 0; j < kCols; ++j) {
      HWY_UNROLL(16)
      for (size_t v = 0; v < kRowV; ++v) acc[j][v] = hn::Zero(d);
    }

    for (size_t p = 0; p < kc; ++p) {
      V va[kRowV];
      HWY_UNROLL(16)
      for (size_t v = 0; v < kRowV; ++v) va[v] = hn::Load(d, a + v * kN);
      HWY_UNROLL(16)
      for (size_t j = 0; j < kCols; ++j) {
        const V vb = hn::Set(
            d, kDirectB ? b_col[j][static_cast<ptrdiff_t>(p) * b_ks] : b[j]);
        HWY_UNROLL(16)
        for (size_t v = 0; v < kRowV; ++v) {
          acc[j][v] = hn::MulAdd(va[v], vb, acc[j][v]);
        }
      }
      a += kMR;
      if constexpr (!kDirectB) b += kNR;
    }

    Store(d, acc, alpha, beta, c, ldc, m);
  }

  // Writes C = alpha * acc + beta * C for m rows. beta == 0 must not read C.
  template <size_t kRowV, size_t kCols>
  static HWY_INLINE void Store(D d, const V (&acc)[kCols][kRowV], T alpha,
                               T beta, T* HWY_RESTRICT c, size_t ldc,
                               size_t m) {
    const V valpha = hn::Set(d, alpha);
    const V vbeta = hn::Set(d, beta);
    const bool beta_zero = IsZero(beta);
    const bool beta_one = IsOne(beta);

    if (HWY_LIKELY(m == kRowV * kN)) {
      HWY_UNROLL(16)
      for (size_t j = 0; j < kCols; ++j) {
        T* HWY_RESTRICT cj = c + j * ldc;
        HWY_UNROLL(16)
        for (size_t v = 0; v < kRowV; ++v) {
          V r = hn::Mul(acc[j][v], valpha);
          if (beta_one) {
            r = hn::Add(r, hn::LoadU(d, cj + v * kN));
          } else if (!beta_zero) {
            r = hn::MulAdd(hn::LoadU(d, cj + v * kN), vbeta, r);
          }
          hn::StoreU(r, d, cj + v * kN);
        }
      }
      return;
    }

    // Partial last vector: masked loads/stores.
    HWY_UNROLL(16)
    for (size_t j = 0; j < kCols; ++j) {
      T* HWY_RESTRICT cj = c + j * ldc;
      HWY_UNROLL(16)
      for (size_t v = 0; v < kRowV; ++v) {
        const size_t row = v * kN;
        const size_t count = HWY_MIN(kN, m - row);
        V r = hn::Mul(acc[j][v], valpha);
        if (!beta_zero) {
          r = hn::MulAdd(hn::LoadN(d, cj + row, count), vbeta, r);
        }
        hn::StoreN(r, d, cj + row, count);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Complex micro-kernel (std::complex<float>, std::complex<double>)
//
// Complex numbers are stored interleaved (re, im), so a vector of kN reals
// holds kN / 2 complex numbers. For each complex b we accumulate separately
//
//   acc_re += a * re(b)   -> lanes (ar*br, ai*br)
//   acc_im += a * im(b)   -> lanes (ar*bi, ai*bi)
//
// and combine once at the end:
//
//   a*b = AddSub(acc_re, Reverse2(acc_im)) = (ar*br - ai*bi, ai*br + ar*bi)
//
// Conjugation of A or B is applied while packing, so this kernel only
// computes plain products.

template <typename T>  // T = std::complex<float> or std::complex<double>
struct ComplexGemmKernel {
  using R = RealT<T>;
  using D = hn::ScalableTag<R>;
  using V = hn::Vec<D>;
  static constexpr size_t kN = hn::MaxLanes(D());  // real lanes per vector
  static_assert(kN >= 2, "Complex kernel requires at least two lanes");
  static constexpr size_t kMV = 2;             // vectors per column
  static constexpr size_t kMR = kMV * kN / 2;  // complex rows per tile
  // Accumulators: 2 * kMV * kNR registers (12 of 16, or 24 of 32).
  static constexpr size_t kNR = kVectorRegs >= 32 ? 6 : 3;

  static constexpr size_t kKC = 256;
  static constexpr size_t kMC = (sizeof(R) == 4 ? 96 : 64) / kMR * kMR;
  static constexpr size_t kNC = 2040 / kNR * kNR;

  static HWY_INLINE void Run(size_t kc, const T* HWY_RESTRICT a,
                             const T* HWY_RESTRICT b, T alpha, T beta,
                             T* HWY_RESTRICT c, size_t ldc, size_t m,
                             size_t n) {
    Dispatch<false>(kc, a, b, 0, 0, alpha, beta, c, ldc, m, n);
  }

  // As Run, but B is read in place instead of from a packed panel: element
  // (p, j) of the kc x n block of op(B) is b[p * b_ks + j * b_cs]. op(B) must
  // not be conjugated.
  static HWY_INLINE void RunDirectB(size_t kc, const T* HWY_RESTRICT a,
                                    const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                                    ptrdiff_t b_cs, T alpha, T beta,
                                    T* HWY_RESTRICT c, size_t ldc, size_t m,
                                    size_t n) {
    // A unit k stride is common (op(B) = B) and worth a constant.
    if (b_ks == 1) {
      Dispatch<true>(kc, a, b, 1, b_cs, alpha, beta, c, ldc, m, n);
    } else {
      Dispatch<true>(kc, a, b, b_ks, b_cs, alpha, beta, c, ldc, m, n);
    }
  }

 private:
  // Runs the instantiation for an m x n tile (m <= kMR, n <= kNR).
  template <bool kDirectB>
  static HWY_INLINE void Dispatch(size_t kc, const T* HWY_RESTRICT a,
                                  const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                                  ptrdiff_t b_cs, T alpha, T beta,
                                  T* HWY_RESTRICT c, size_t ldc, size_t m,
                                  size_t n) {
    static_assert(kMV == 2, "Dispatch assumes two vectors per column");
    const auto cols = std::make_index_sequence<kNR - 1>();
    if (HWY_LIKELY(2 * m > kN)) {
      DispatchCols<kDirectB, 2>(cols, kc, a, b, b_ks, b_cs, alpha, beta, c, ldc,
                                m, n);
    } else {
      DispatchCols<kDirectB, 1>(cols, kc, a, b, b_ks, b_cs, alpha, beta, c, ldc,
                                m, n);
    }
  }

  // Runs Tile<..., kCols = n>; kJ + 1 enumerates the partial column counts.
  template <bool kDirectB, size_t kRowV, size_t... kJ>
  static HWY_INLINE void DispatchCols(std::index_sequence<kJ...> /*cols*/,
                                      size_t kc, const T* HWY_RESTRICT a,
                                      const T* HWY_RESTRICT b, ptrdiff_t b_ks,
                                      ptrdiff_t b_cs, T alpha, T beta,
                                      T* HWY_RESTRICT c, size_t ldc, size_t m,
                                      size_t n) {
    if (HWY_LIKELY(n == kNR)) {
      return Tile<kDirectB, kRowV, kNR>(kc, a, b, b_ks, b_cs, alpha, beta, c,
                                        ldc, m);
    }
    (void)((n == kJ + 1 && (Tile<kDirectB, kRowV, kJ + 1>(
                                kc, a, b, b_ks, b_cs, alpha, beta, c, ldc, m),
                            true)) ||
           ...);
  }

  // Computes an m x kCols tile with kRowV vectors per column (m complex rows
  // span at most kRowV vectors). Panels of A have kMR rows; only the first
  // kRowV vectors of each step are read.
  template <bool kDirectB, size_t kRowV, size_t kCols>
  static HWY_INLINE void Tile(size_t kc, const T* HWY_RESTRICT a_complex,
                              const T* HWY_RESTRICT b_complex, ptrdiff_t b_ks,
                              ptrdiff_t b_cs, T alpha, T beta,
                              T* HWY_RESTRICT c_complex, size_t ldc, size_t m) {
    const D d;
    // Work on the underlying reals: a k step of a panel spans 2 * kMR reals.
    const R* HWY_RESTRICT a = reinterpret_cast<const R*>(a_complex);
    const R* HWY_RESTRICT b = reinterpret_cast<const R*>(b_complex);

    // Direct B: one pointer per column; strides in reals.
    const R* HWY_RESTRICT b_col[kCols];
    if constexpr (kDirectB) {
      HWY_UNROLL(16)
      for (size_t j = 0; j < kCols; ++j) {
        b_col[j] = b + 2 * static_cast<ptrdiff_t>(j) * b_cs;
      }
    }
    // Returns the real (part 0) or imaginary (part 1) part of B(p, j).
    const auto b_at = [&](size_t p, size_t j, size_t part) HWY_ATTR {
      return kDirectB ? b_col[j][2 * static_cast<ptrdiff_t>(p) * b_ks + part]
                      : b[2 * j + part];
    };

    V acc_re[kCols][kRowV];
    V acc_im[kCols][kRowV];
    HWY_UNROLL(16)
    for (size_t j = 0; j < kCols; ++j) {
      HWY_UNROLL(16)
      for (size_t v = 0; v < kRowV; ++v) {
        acc_re[j][v] = hn::Zero(d);
        acc_im[j][v] = hn::Zero(d);
      }
    }

    for (size_t p = 0; p < kc; ++p) {
      V va[kRowV];
      HWY_UNROLL(16)
      for (size_t v = 0; v < kRowV; ++v) va[v] = hn::Load(d, a + v * kN);
      // Real and imaginary parts are processed one after the other so that
      // only one broadcast register is live: with 16 registers the tile
      // uses 12 accumulators + kMV for A + 1 for B.
      HWY_UNROLL(16)
      for (size_t j = 0; j < kCols; ++j) {
        const V b_re = hn::Set(d, b_at(p, j, 0));
        HWY_UNROLL(16)
        for (size_t v = 0; v < kRowV; ++v) {
          acc_re[j][v] = hn::MulAdd(va[v], b_re, acc_re[j][v]);
        }
        const V b_im = hn::Set(d, b_at(p, j, 1));
        HWY_UNROLL(16)
        for (size_t v = 0; v < kRowV; ++v) {
          acc_im[j][v] = hn::MulAdd(va[v], b_im, acc_im[j][v]);
        }
      }
      a += 2 * kMR;
      if constexpr (!kDirectB) b += 2 * kNR;
    }

    Store(d, acc_re, acc_im, alpha, beta, reinterpret_cast<R*>(c_complex), ldc,
          m);
  }

  // Returns the complex product of each (re, im) lane pair of v with the
  // complex scalar s.
  static HWY_INLINE V MulScalar(D d, V v, T s) {
    return hn::AddSub(hn::Mul(v, hn::Set(d, s.real())),
                      hn::Mul(hn::Reverse2(d, v), hn::Set(d, s.imag())));
  }

  // Writes C = alpha * (A * B) + beta * C for m rows, where c points to the
  // reals of C (column j starts at c + 2 * j * ldc). beta == 0 must not read
  // C.
  template <size_t kRowV, size_t kCols>
  static HWY_INLINE void Store(D d, const V (&acc_re)[kCols][kRowV],
                               const V (&acc_im)[kCols][kRowV], T alpha, T beta,
                               R* HWY_RESTRICT c, size_t ldc, size_t m) {
    const bool beta_zero = IsZero(beta);
    const bool beta_one = IsOne(beta);
    const size_t m_reals = 2 * m;

    HWY_UNROLL(16)
    for (size_t j = 0; j < kCols; ++j) {
      R* HWY_RESTRICT cj = c + 2 * j * ldc;
      HWY_UNROLL(16)
      for (size_t v = 0; v < kRowV; ++v) {
        const size_t row = v * kN;  // in reals
        const size_t count = HWY_MIN(kN, m_reals - row);
        const V ab = hn::AddSub(acc_re[j][v], hn::Reverse2(d, acc_im[j][v]));
        V r = MulScalar(d, ab, alpha);
        if (HWY_LIKELY(count == kN)) {
          if (beta_one) {
            r = hn::Add(r, hn::LoadU(d, cj + row));
          } else if (!beta_zero) {
            r = hn::Add(r, MulScalar(d, hn::LoadU(d, cj + row), beta));
          }
          hn::StoreU(r, d, cj + row);
        } else {
          if (!beta_zero) {
            r = hn::Add(r, MulScalar(d, hn::LoadN(d, cj + row, count), beta));
          }
          hn::StoreN(r, d, cj + row, count);
        }
      }
    }
  }
};

// Selects the micro-kernel for element type T.
template <typename T>
using GemmKernel = std::conditional_t<IsComplex<T>::value, ComplexGemmKernel<T>,
                                      RealGemmKernel<T>>;

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_GEMM_KERNEL_INL_H_
