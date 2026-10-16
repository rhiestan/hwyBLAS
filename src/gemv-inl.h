// GEMV kernels: y := alpha * op(A) * x + beta * y with contiguous x and y.
//
// GEMV reads each element of A exactly once, so it is bound by memory
// bandwidth. The kernels therefore aim to stream A with full-width vector
// loads and to touch x and y as rarely as possible:
//
//   - NoTrans: y += A[:, j:j+4] * (alpha * x[j:j+4]); four columns are
//     combined per pass over y ("axpy" form, AxpyCols).
//   - Trans:   y[j] += alpha * dot(A[:, j], x); four columns share each load
//     of x ("dot" form, DotCols).
//
// AxpyCols and DotCols take the columns as an array of pointers rather than
// a leading dimension, so that the banded and packed routines (whose columns
// are not equally spaced) can use them too.
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
// Axpy form: y[0:m] += sum over c < kCols of cols[c][0:m] * t[c].

template <size_t kCols, typename T>
HWY_INLINE void AxpyColsReal(size_t m, const T* HWY_RESTRICT const* cols,
                             const T* HWY_RESTRICT t, T* HWY_RESTRICT y) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);

  const T* HWY_RESTRICT c_ptr[kCols];
  V vt[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    c_ptr[c] = cols[c];
    vt[c] = hn::Set(d, t[c]);
  }

  size_t i = 0;
  for (; i + N <= m; i += N) {
    V acc = hn::LoadU(d, y + i);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      acc = hn::MulAdd(hn::LoadU(d, c_ptr[c] + i), vt[c], acc);
    }
    hn::StoreU(acc, d, y + i);
  }
  if (i < m) {
    const size_t count = m - i;
    V acc = hn::LoadN(d, y + i, count);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      acc = hn::MulAdd(hn::LoadN(d, c_ptr[c] + i, count), vt[c], acc);
    }
    hn::StoreN(acc, d, y + i, count);
  }
}

// Complex: pointers to interleaved reals, m in complex elements.
// Accumulates A * re(t) and A * im(t) separately and combines them once per
// vector of y: A*t = AddSub(A*re(t), Reverse2(A*im(t))).
template <size_t kCols, typename R>
HWY_INLINE void AxpyColsComplex(size_t m, const R* HWY_RESTRICT const* cols,
                                const R* HWY_RESTRICT t, R* HWY_RESTRICT y) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);  // in reals
  const size_t m2 = 2 * m;

  const R* HWY_RESTRICT c_ptr[kCols];
  V t_re[kCols], t_im[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    c_ptr[c] = cols[c];
    t_re[c] = hn::Set(d, t[2 * c]);
    t_im[c] = hn::Set(d, t[2 * c + 1]);
  }

  size_t i = 0;
  for (; i + N <= m2; i += N) {
    V acc_re = hn::Zero(d), acc_im = hn::Zero(d);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadU(d, c_ptr[c] + i);
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
      const V av = hn::LoadN(d, c_ptr[c] + i, count);
      acc_re = hn::MulAdd(av, t_re[c], acc_re);
      acc_im = hn::MulAdd(av, t_im[c], acc_im);
    }
    const V prod = hn::AddSub(acc_re, hn::Reverse2(d, acc_im));
    hn::StoreN(hn::Add(hn::LoadN(d, y + i, count), prod), d, y + i, count);
  }
}

template <size_t kCols, typename T>
HWY_INLINE void AxpyCols(size_t m, const T* HWY_RESTRICT const* cols,
                         const T* HWY_RESTRICT t, T* HWY_RESTRICT y) {
  if (m == 0) return;
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    const R* rcols[kCols];
    for (size_t c = 0; c < kCols; ++c) {
      rcols[c] = reinterpret_cast<const R*>(cols[c]);
    }
    AxpyColsComplex<kCols>(m, rcols, reinterpret_cast<const R*>(t),
                           reinterpret_cast<R*>(y));
  } else {
    AxpyColsReal<kCols>(m, cols, t, y);
  }
}

// ---------------------------------------------------------------------------
// Dot form: y[c] += alpha * sum over i < m of op(cols[c][i]) * x[i] for
// c < kCols, where op conjugates if kConj.

template <size_t kCols, typename T>
HWY_INLINE void DotColsReal(size_t m, const T* HWY_RESTRICT const* cols,
                            T alpha, const T* HWY_RESTRICT x,
                            T* HWY_RESTRICT y) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);

  const T* HWY_RESTRICT c_ptr[kCols];
  V s[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    c_ptr[c] = cols[c];
    s[c] = hn::Zero(d);
  }

  size_t i = 0;
  for (; i + N <= m; i += N) {
    const V xv = hn::LoadU(d, x + i);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      s[c] = hn::MulAdd(hn::LoadU(d, c_ptr[c] + i), xv, s[c]);
    }
  }
  if (i < m) {
    // LoadN zeroes the lanes past `count`, so they do not contribute.
    const size_t count = m - i;
    const V xv = hn::LoadN(d, x + i, count);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      s[c] = hn::MulAdd(hn::LoadN(d, c_ptr[c] + i, count), xv, s[c]);
    }
  }
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) y[c] += alpha * hn::ReduceSum(d, s[c]);
}

// Complex. With P = a * x and Q = a * Reverse2(x) (lane-wise products), the
// complex dot products follow from lane sums:
//   sum a * x       = (sum P_even - sum P_odd, sum Q)
//   sum conj(a) * x = (sum P, sum Q_even - sum Q_odd)
template <size_t kCols, bool kConj, typename R>
HWY_INLINE void DotColsComplex(size_t m, const R* HWY_RESTRICT const* cols,
                               std::complex<R> alpha, const R* HWY_RESTRICT x,
                               R* HWY_RESTRICT y) {
  const hn::ScalableTag<R> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);  // in reals
  const size_t m2 = 2 * m;

  const R* HWY_RESTRICT c_ptr[kCols];
  V p[kCols], q[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    c_ptr[c] = cols[c];
    p[c] = hn::Zero(d);
    q[c] = hn::Zero(d);
  }

  size_t i = 0;
  for (; i + N <= m2; i += N) {
    const V xv = hn::LoadU(d, x + i);
    const V xs = hn::Reverse2(d, xv);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadU(d, c_ptr[c] + i);
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
      const V av = hn::LoadN(d, c_ptr[c] + i, count);
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
    y[2 * c] += r.real();
    y[2 * c + 1] += r.imag();
  }
}

template <size_t kCols, bool kConj, typename T>
HWY_INLINE void DotCols(size_t m, const T* HWY_RESTRICT const* cols, T alpha,
                        const T* HWY_RESTRICT x, T* HWY_RESTRICT y) {
  if (m == 0) return;
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    const R* rcols[kCols];
    for (size_t c = 0; c < kCols; ++c) {
      rcols[c] = reinterpret_cast<const R*>(cols[c]);
    }
    DotColsComplex<kCols, kConj>(m, rcols, alpha, reinterpret_cast<const R*>(x),
                                 reinterpret_cast<R*>(y));
  } else {
    DotColsReal<kCols>(m, cols, alpha, x, y);
  }
}

// ---------------------------------------------------------------------------
// Rectangles A[0:m, 0:n] with leading dimension lda.

// y[0:m] += A * t[0:n].
template <typename T>
HWY_NOINLINE void GemvN(size_t m, size_t n, const T* HWY_RESTRICT a, size_t lda,
                        const T* HWY_RESTRICT t, T* HWY_RESTRICT y) {
  size_t j = 0;
  for (; j + kGemvCols <= n; j += kGemvCols) {
    const T* cols[kGemvCols];
    for (size_t c = 0; c < kGemvCols; ++c) cols[c] = a + (j + c) * lda;
    AxpyCols<kGemvCols>(m, cols, t + j, y);
  }
  for (; j < n; ++j) {
    const T* col = a + j * lda;
    AxpyCols<1>(m, &col, t + j, y);
  }
}

// y[0:n] += alpha * op(A)^T * x[0:m], op conjugating if kConj.
template <bool kConj, typename T>
HWY_NOINLINE void GemvT(size_t m, size_t n, T alpha, const T* HWY_RESTRICT a,
                        size_t lda, const T* HWY_RESTRICT x,
                        T* HWY_RESTRICT y) {
  size_t j = 0;
  for (; j + kGemvCols <= n; j += kGemvCols) {
    const T* cols[kGemvCols];
    for (size_t c = 0; c < kGemvCols; ++c) cols[c] = a + (j + c) * lda;
    DotCols<kGemvCols, kConj>(m, cols, alpha, x, y + j);
  }
  for (; j < n; ++j) {
    const T* col = a + j * lda;
    DotCols<1, kConj>(m, &col, alpha, x, y + j);
  }
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

  if (op == Op::kNoTrans) {
    // t = alpha * x, so the inner loop is a pure multiply-add.
    T* t = static_cast<T*>(ScratchBuffer(kScratchTemp, n * sizeof(T)));
    for (size_t j = 0; j < n; ++j) t[j] = alpha * x[j];
    GemvN(m, n, a, lda, t, y);
  } else if (op == Op::kTrans) {
    GemvT<false>(m, n, alpha, a, lda, x, y);
  } else {
    GemvT<true>(m, n, alpha, a, lda, x, y);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_GEMV_INL_H_
