// Rank-1 and rank-2 update kernels (GER, SYR, SYR2, HER, HER2):
//
//   A[0:m, 0:n] += u0 * s0^T  (+ u1 * s1^T)
//
// where u0, u1 are contiguous column vectors and s0, s1 are row vectors that
// already include alpha and any conjugation. Like GEMV, these updates read
// and write every element of A once and are bound by memory bandwidth, so
// the kernels stream columns of A with full-width vectors and load each
// vector of u once per 4 columns.
//
// For complex numbers, a += u * s with s = (sr, si) is computed as
//   a += u * sr + Reverse2(u) * (-si, si),
// i.e. two multiply-adds per vector and term.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_RANK_UPDATE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_RANK_UPDATE_INL_H_
#undef HWYBLAS_SRC_RANK_UPDATE_INL_H_
#else
#define HWYBLAS_SRC_RANK_UPDATE_INL_H_
#endif

#include <stddef.h>

#include "common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Columns updated together.
constexpr size_t kRankUpdateCols = 4;

// Columns [0, kCols) of A[0:m, :] += sum over t < kTerms of u_t * s_t^T.
template <size_t kCols, size_t kTerms, typename T>
HWY_INLINE void RankUpdateCols(size_t m, T* HWY_RESTRICT a, size_t lda,
                               const T* HWY_RESTRICT u0, const T* s0,
                               const T* HWY_RESTRICT u1, const T* s1) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);
  const size_t len = RealCount<T>(m);

  const R* HWY_RESTRICT u[2] = {reinterpret_cast<const R*>(u0),
                                reinterpret_cast<const R*>(u1)};
  R* HWY_RESTRICT cols[kCols];
  // Broadcast scales: s_re = re(s) (or s), s_im = (-im(s), im(s)) pairs.
  V s_re[kTerms][kCols], s_im[kTerms][kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    cols[c] = reinterpret_cast<R*>(a + c * lda);
    HWY_UNROLL(16)
    for (size_t t = 0; t < kTerms; ++t) {
      const T st = (t == 0 ? s0 : s1)[c];
      if constexpr (kComplex) {
        s_re[t][c] = hn::Set(d, st.real());
        s_im[t][c] = hn::OddEven(hn::Set(d, st.imag()), hn::Set(d, -st.imag()));
      } else {
        s_re[t][c] = hn::Set(d, st);
        s_im[t][c] = s_re[t][c];  // unused
      }
    }
  }

  size_t i = 0;
  for (; i + N <= len; i += N) {
    V uv[kTerms], us[kTerms];
    HWY_UNROLL(16)
    for (size_t t = 0; t < kTerms; ++t) {
      uv[t] = hn::LoadU(d, u[t] + i);
      us[t] = kComplex ? hn::Reverse2(d, uv[t]) : uv[t];
    }
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      V av = hn::LoadU(d, cols[c] + i);
      HWY_UNROLL(16)
      for (size_t t = 0; t < kTerms; ++t) {
        av = hn::MulAdd(uv[t], s_re[t][c], av);
        if constexpr (kComplex) av = hn::MulAdd(us[t], s_im[t][c], av);
      }
      hn::StoreU(av, d, cols[c] + i);
    }
  }
  if (i < len) {
    const size_t count = len - i;
    V uv[kTerms], us[kTerms];
    HWY_UNROLL(16)
    for (size_t t = 0; t < kTerms; ++t) {
      uv[t] = hn::LoadN(d, u[t] + i, count);
      us[t] = kComplex ? hn::Reverse2(d, uv[t]) : uv[t];
    }
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      V av = hn::LoadN(d, cols[c] + i, count);
      HWY_UNROLL(16)
      for (size_t t = 0; t < kTerms; ++t) {
        av = hn::MulAdd(uv[t], s_re[t][c], av);
        if constexpr (kComplex) av = hn::MulAdd(us[t], s_im[t][c], av);
      }
      hn::StoreN(av, d, cols[c] + i, count);
    }
  }
}

// A[0:m, 0:n] += u0 * s0[0:n]^T (+ u1 * s1[0:n]^T if kTerms == 2). For
// kTerms == 1, u1 and s1 are ignored (but must be valid pointers).
template <size_t kTerms, typename T>
HWY_NOINLINE void RankUpdate(size_t m, size_t n, T* HWY_RESTRICT a, size_t lda,
                             const T* u0, const T* s0, const T* u1,
                             const T* s1) {
  if (m == 0) return;
  size_t j = 0;
  for (; j + kRankUpdateCols <= n; j += kRankUpdateCols) {
    RankUpdateCols<kRankUpdateCols, kTerms>(m, a + j * lda, lda, u0, s0 + j, u1,
                                            s1 + j);
  }
  for (; j < n; ++j) {
    RankUpdateCols<1, kTerms>(m, a + j * lda, lda, u0, s0 + j, u1, s1 + j);
  }
}

// As RankUpdate for the `lower` or upper triangle of the n x n matrix A
// (SYR, SYR2, HER, HER2). If hermitian, the imaginary parts of the diagonal
// are set to zero, as in the reference BLAS.
//
// Strips of kRankUpdateCols columns are split into a small diagonal block,
// updated element by element, and the rectangle below (lower) or above
// (upper) it, updated by RankUpdate.
template <size_t kTerms, typename T>
void TriangularRankUpdate(bool lower, bool hermitian, size_t n, T* a,
                          size_t lda, const T* u0, const T* s0, const T* u1,
                          const T* s1) {
  for (size_t j = 0; j < n; j += kRankUpdateCols) {
    const size_t cols = HWY_MIN(kRankUpdateCols, n - j);
    for (size_t jj = j; jj < j + cols; ++jj) {
      for (size_t ii = j; ii < j + cols; ++ii) {
        if (lower ? (ii < jj) : (ii > jj)) continue;
        T v = u0[ii] * s0[jj];
        if (kTerms == 2) v += u1[ii] * s1[jj];
        T& e = a[ii + jj * lda];
        if constexpr (IsComplex<T>::value) {
          if (hermitian && ii == jj) {
            e = T(e.real() + v.real(), 0);
            continue;
          }
        }
        e += v;
      }
    }
    const size_t row0 = lower ? j + cols : 0;
    const size_t m = lower ? n - row0 : j;
    RankUpdate<kTerms>(m, cols, a + row0 + j * lda, lda, u0 + row0, s0 + j,
                       u1 + row0, s1 + j);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_RANK_UPDATE_INL_H_
