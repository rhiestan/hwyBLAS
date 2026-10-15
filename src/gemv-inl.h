// GEMV kernels: y := alpha * op(A) * x + beta * y with contiguous x and y.
//
// GEMV reads each element of A exactly once, so it is bound by memory
// bandwidth. The kernels therefore aim to stream A with full-width vector
// loads and to touch x and y as rarely as possible:
//
//   - NoTrans: y += A[:, j:j+4] * (alpha * x[j:j+4]); four columns are
//     combined per pass over y ("axpy" form).
//   - Trans:   y[j] += alpha * dot(A[:, j], x); four columns share each load
//     of x ("dot" form).
//
// Complex numbers are interleaved (re, im). Products are formed with the same
// AddSub/Reverse2 identities as the GEMM micro-kernel (see gemm_kernel-inl.h).
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_GEMV_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_GEMV_INL_H_
#undef HWYBLAS_SRC_GEMV_INL_H_
#else
#define HWYBLAS_SRC_GEMV_INL_H_
#endif

#include <stddef.h>

#include <complex>

#include "common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Columns processed together by the GEMV kernels.
constexpr size_t kGemvCols = 4;

// y[0:n] := beta * y[0:n]; beta == 0 sets y to zero without reading it.
template <typename T>
HWY_INLINE void ScaleVector(size_t n, T beta, T* HWY_RESTRICT y) {
  if (IsOne(beta)) return;
  if (IsZero(beta)) {
    for (size_t i = 0; i < n; ++i) y[i] = T(0);
  } else {
    for (size_t i = 0; i < n; ++i) y[i] *= beta;
  }
}

// ---------------------------------------------------------------------------
// Real

// y[0:m] += A[0:m, 0:n] * t[0:n], A column-major.
template <typename T>
HWY_NOINLINE void GemvNReal(size_t m, size_t n, const T* HWY_RESTRICT a,
                            size_t lda, const T* HWY_RESTRICT t,
                            T* HWY_RESTRICT y) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);

  size_t j = 0;
  for (; j + kGemvCols <= n; j += kGemvCols) {
    const T* HWY_RESTRICT a0 = a + (j + 0) * lda;
    const T* HWY_RESTRICT a1 = a + (j + 1) * lda;
    const T* HWY_RESTRICT a2 = a + (j + 2) * lda;
    const T* HWY_RESTRICT a3 = a + (j + 3) * lda;
    const V t0 = hn::Set(d, t[j + 0]), t1 = hn::Set(d, t[j + 1]);
    const V t2 = hn::Set(d, t[j + 2]), t3 = hn::Set(d, t[j + 3]);
    size_t i = 0;
    for (; i + N <= m; i += N) {
      V acc = hn::LoadU(d, y + i);
      acc = hn::MulAdd(hn::LoadU(d, a0 + i), t0, acc);
      acc = hn::MulAdd(hn::LoadU(d, a1 + i), t1, acc);
      acc = hn::MulAdd(hn::LoadU(d, a2 + i), t2, acc);
      acc = hn::MulAdd(hn::LoadU(d, a3 + i), t3, acc);
      hn::StoreU(acc, d, y + i);
    }
    if (i < m) {
      const size_t count = m - i;
      V acc = hn::LoadN(d, y + i, count);
      acc = hn::MulAdd(hn::LoadN(d, a0 + i, count), t0, acc);
      acc = hn::MulAdd(hn::LoadN(d, a1 + i, count), t1, acc);
      acc = hn::MulAdd(hn::LoadN(d, a2 + i, count), t2, acc);
      acc = hn::MulAdd(hn::LoadN(d, a3 + i, count), t3, acc);
      hn::StoreN(acc, d, y + i, count);
    }
  }
  for (; j < n; ++j) {
    const T* HWY_RESTRICT aj = a + j * lda;
    const V tj = hn::Set(d, t[j]);
    size_t i = 0;
    for (; i + N <= m; i += N) {
      hn::StoreU(hn::MulAdd(hn::LoadU(d, aj + i), tj, hn::LoadU(d, y + i)), d,
                 y + i);
    }
    if (i < m) {
      const size_t count = m - i;
      hn::StoreN(hn::MulAdd(hn::LoadN(d, aj + i, count), tj,
                            hn::LoadN(d, y + i, count)),
                 d, y + i, count);
    }
  }
}

// y[j] += alpha * dot(A[0:m, j], x[0:m]) for j in [0, n).
template <typename T>
HWY_NOINLINE void GemvTReal(size_t m, size_t n, T alpha,
                            const T* HWY_RESTRICT a, size_t lda,
                            const T* HWY_RESTRICT x, T* HWY_RESTRICT y) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);

  size_t j = 0;
  for (; j + kGemvCols <= n; j += kGemvCols) {
    const T* HWY_RESTRICT a0 = a + (j + 0) * lda;
    const T* HWY_RESTRICT a1 = a + (j + 1) * lda;
    const T* HWY_RESTRICT a2 = a + (j + 2) * lda;
    const T* HWY_RESTRICT a3 = a + (j + 3) * lda;
    V s0 = hn::Zero(d), s1 = hn::Zero(d), s2 = hn::Zero(d), s3 = hn::Zero(d);
    size_t i = 0;
    for (; i + N <= m; i += N) {
      const V xv = hn::LoadU(d, x + i);
      s0 = hn::MulAdd(hn::LoadU(d, a0 + i), xv, s0);
      s1 = hn::MulAdd(hn::LoadU(d, a1 + i), xv, s1);
      s2 = hn::MulAdd(hn::LoadU(d, a2 + i), xv, s2);
      s3 = hn::MulAdd(hn::LoadU(d, a3 + i), xv, s3);
    }
    if (i < m) {
      // LoadN zeroes the lanes past `count`, so they do not contribute.
      const size_t count = m - i;
      const V xv = hn::LoadN(d, x + i, count);
      s0 = hn::MulAdd(hn::LoadN(d, a0 + i, count), xv, s0);
      s1 = hn::MulAdd(hn::LoadN(d, a1 + i, count), xv, s1);
      s2 = hn::MulAdd(hn::LoadN(d, a2 + i, count), xv, s2);
      s3 = hn::MulAdd(hn::LoadN(d, a3 + i, count), xv, s3);
    }
    y[j + 0] += alpha * hn::ReduceSum(d, s0);
    y[j + 1] += alpha * hn::ReduceSum(d, s1);
    y[j + 2] += alpha * hn::ReduceSum(d, s2);
    y[j + 3] += alpha * hn::ReduceSum(d, s3);
  }
  for (; j < n; ++j) {
    const T* HWY_RESTRICT aj = a + j * lda;
    V s = hn::Zero(d);
    size_t i = 0;
    for (; i + N <= m; i += N) {
      s = hn::MulAdd(hn::LoadU(d, aj + i), hn::LoadU(d, x + i), s);
    }
    if (i < m) {
      const size_t count = m - i;
      s = hn::MulAdd(hn::LoadN(d, aj + i, count), hn::LoadN(d, x + i, count),
                     s);
    }
    y[j] += alpha * hn::ReduceSum(d, s);
  }
}

// ---------------------------------------------------------------------------
// Complex. Pointers are to the underlying reals; lengths are in complex
// elements.

// y[0:m] += A[0:m, j:j+kCols] * t[j:j+kCols].
//
// Accumulates A * re(t) and A * im(t) separately and combines them once per
// vector of y: A*t = AddSub(A*re(t), Reverse2(A*im(t))).
template <size_t kCols, typename R>
HWY_INLINE void GemvNComplexCols(size_t m, size_t j, const R* HWY_RESTRICT a,
                                 size_t lda, const R* HWY_RESTRICT t,
                                 R* HWY_RESTRICT y) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);  // in reals
  const size_t m2 = 2 * m;

  const R* HWY_RESTRICT cols[kCols];
  V t_re[kCols], t_im[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    cols[c] = a + 2 * (j + c) * lda;
    t_re[c] = hn::Set(d, t[2 * (j + c)]);
    t_im[c] = hn::Set(d, t[2 * (j + c) + 1]);
  }

  size_t i = 0;
  for (; i + N <= m2; i += N) {
    V acc_re = hn::Zero(d), acc_im = hn::Zero(d);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadU(d, cols[c] + i);
      acc_re = hn::MulAdd(av, t_re[c], acc_re);
      acc_im = hn::MulAdd(av, t_im[c], acc_im);
    }
    const V prod = hn::AddSub(acc_re, hn::Reverse2(d, acc_im));
    hn::StoreU(hn::Add(hn::LoadU(d, y + i), prod), d, y + i);
  }
  if (i < m2) {
    const size_t count = m2 - i;
    V acc_re = hn::Zero(d), acc_im = hn::Zero(d);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadN(d, cols[c] + i, count);
      acc_re = hn::MulAdd(av, t_re[c], acc_re);
      acc_im = hn::MulAdd(av, t_im[c], acc_im);
    }
    const V prod = hn::AddSub(acc_re, hn::Reverse2(d, acc_im));
    hn::StoreN(hn::Add(hn::LoadN(d, y + i, count), prod), d, y + i, count);
  }
}

// y[0:m] += A[0:m, 0:n] * t[0:n].
template <typename R>
HWY_NOINLINE void GemvNComplex(size_t m, size_t n, const R* HWY_RESTRICT a,
                               size_t lda, const R* HWY_RESTRICT t,
                               R* HWY_RESTRICT y) {
  size_t j = 0;
  for (; j + kGemvCols <= n; j += kGemvCols) {
    GemvNComplexCols<kGemvCols>(m, j, a, lda, t, y);
  }
  for (; j < n; ++j) GemvNComplexCols<1>(m, j, a, lda, t, y);
}

// y[j+c] += alpha * dot(op(A[0:m, j+c]), x[0:m]) for c in [0, kCols), where
// op conjugates if kConj.
//
// With P = a * x and Q = a * Reverse2(x) (lane-wise products), the complex
// dot products follow from lane sums:
//   sum a * x       = (sum P_even - sum P_odd, sum Q)
//   sum conj(a) * x = (sum P, sum Q_even - sum Q_odd)
template <size_t kCols, bool kConj, typename R>
HWY_INLINE void GemvTComplexCols(size_t m, size_t j, std::complex<R> alpha,
                                 const R* HWY_RESTRICT a, size_t lda,
                                 const R* HWY_RESTRICT x, R* HWY_RESTRICT y) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);  // in reals
  const size_t m2 = 2 * m;

  const R* HWY_RESTRICT cols[kCols];
  V p[kCols], q[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    cols[c] = a + 2 * (j + c) * lda;
    p[c] = hn::Zero(d);
    q[c] = hn::Zero(d);
  }

  size_t i = 0;
  for (; i + N <= m2; i += N) {
    const V xv = hn::LoadU(d, x + i);
    const V xs = hn::Reverse2(d, xv);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadU(d, cols[c] + i);
      p[c] = hn::MulAdd(av, xv, p[c]);
      q[c] = hn::MulAdd(av, xs, q[c]);
    }
  }
  if (i < m2) {
    // LoadN zeroes the lanes past `count`, so they do not contribute.
    const size_t count = m2 - i;
    const V xv = hn::LoadN(d, x + i, count);
    const V xs = hn::Reverse2(d, xv);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadN(d, cols[c] + i, count);
      p[c] = hn::MulAdd(av, xv, p[c]);
      q[c] = hn::MulAdd(av, xs, q[c]);
    }
  }

  // +1 in even lanes, -1 in odd lanes.
  const V alternate = hn::OddEven(hn::Set(d, R(-1)), hn::Set(d, R(1)));
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    std::complex<R> dot;
    if (kConj) {
      dot = {hn::ReduceSum(d, p[c]),
             hn::ReduceSum(d, hn::Mul(q[c], alternate))};
    } else {
      dot = {hn::ReduceSum(d, hn::Mul(p[c], alternate)),
             hn::ReduceSum(d, q[c])};
    }
    const std::complex<R> r = alpha * dot;
    y[2 * (j + c)] += r.real();
    y[2 * (j + c) + 1] += r.imag();
  }
}

// y[0:n] += alpha * op(A[0:m, 0:n])^T * x[0:m].
template <bool kConj, typename R>
HWY_NOINLINE void GemvTComplex(size_t m, size_t n, std::complex<R> alpha,
                               const R* HWY_RESTRICT a, size_t lda,
                               const R* HWY_RESTRICT x, R* HWY_RESTRICT y) {
  size_t j = 0;
  for (; j + kGemvCols <= n; j += kGemvCols) {
    GemvTComplexCols<kGemvCols, kConj>(m, j, alpha, a, lda, x, y);
  }
  for (; j < n; ++j) GemvTComplexCols<1, kConj>(m, j, alpha, a, lda, x, y);
}

// ---------------------------------------------------------------------------
// Entry point: y := alpha * op(A) * x + beta * y with contiguous x and y.
// Callers handle argument checking and quick returns.

template <typename T>
void GemvContiguous(Op op, size_t m, size_t n, T alpha, const T* a, size_t lda,
                    const T* x, T beta, T* y) {
  const size_t len_y = (op == Op::kNoTrans) ? m : n;
  ScaleVector(len_y, beta, y);
  if (IsZero(alpha)) return;

  if constexpr (!IsComplex<T>::value) {
    if (op == Op::kNoTrans) {
      // t = alpha * x, so the inner loop is a pure multiply-add.
      T* t = static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
      for (size_t j = 0; j < n; ++j) t[j] = alpha * x[j];
      GemvNReal(m, n, a, lda, t, y);
    } else {
      GemvTReal(m, n, alpha, a, lda, x, y);
    }
  } else {
    using R = RealT<T>;
    const R* ar = reinterpret_cast<const R*>(a);
    const R* xr = reinterpret_cast<const R*>(x);
    R* yr = reinterpret_cast<R*>(y);
    if (op == Op::kNoTrans) {
      T* t = static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
      for (size_t j = 0; j < n; ++j) t[j] = alpha * x[j];
      GemvNComplex(m, n, ar, lda, reinterpret_cast<const R*>(t), yr);
    } else if (op == Op::kTrans) {
      GemvTComplex<false>(m, n, alpha, ar, lda, xr, yr);
    } else {
      GemvTComplex<true>(m, n, alpha, ar, lda, xr, yr);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_GEMV_INL_H_
