// SYMV / HEMV kernels: y := alpha * A * x + beta * y for a symmetric or
// Hermitian A of which only one triangle is stored; x and y contiguous.
//
// The matrix is processed in strips of a few columns. Each strip consists
// of a small diagonal block and a rectangle R of stored elements below it
// (lower) or above it (upper). Every element of R stands for two elements of
// A, so R contributes both
//
//   y[rows] += R * (alpha * x[cols])            ("axpy" form, like GEMV N)
//   y[cols] += alpha * op(R)^T * x[rows]        ("dot" form, like GEMV T)
//
// where op conjugates for Hermitian A. A fused kernel computes both in a
// single pass over R, so A is read only once.
//
// Per-target include guard (see Highway's "-inl.h" convention).
#if defined(HWYBLAS_SRC_SYMV_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HWYBLAS_SRC_SYMV_INL_H_
#undef HWYBLAS_SRC_SYMV_INL_H_
#else
#define HWYBLAS_SRC_SYMV_INL_H_
#endif

#include <stddef.h>

#include "common.h"
#include "hwy/highway.h"
#include "src/gemv-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Columns per strip. The strip kernels keep per-column broadcasts and
// accumulators in registers: 2 per column for real types, 4 for complex ones
// (which would spill with 4 columns on 16-register targets).
template <typename T>
constexpr size_t SymvCols() {
  return IsComplex<T>::value ? 4 : 4;
}

// Real: for the m x kCols rectangle R whose columns are cols[c]:
//   y[0:m] += R * t[0:kCols]   and   dots[c] = dot(R[:, c], x[0:m]).
template <size_t kCols, typename T>
HWY_INLINE void AxpyDotColsReal(size_t m, const T* HWY_RESTRICT const* cols,
                                const T* HWY_RESTRICT t,
                                const T* HWY_RESTRICT x, T* HWY_RESTRICT y,
                                T* HWY_RESTRICT dots) {
  const hn::ScalableTag<T> d;
  using V = hn::Vec<decltype(d)>;
  const size_t N = hn::Lanes(d);

  const T* HWY_RESTRICT c_ptr[kCols];
  V vt[kCols], s[kCols];
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    c_ptr[c] = cols[c];
    vt[c] = hn::Set(d, t[c]);
    s[c] = hn::Zero(d);
  }

  size_t i = 0;
  for (; i + N <= m; i += N) {
    const V xv = hn::LoadU(d, x + i);
    V yv = hn::LoadU(d, y + i);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadU(d, c_ptr[c] + i);
      yv = hn::MulAdd(av, vt[c], yv);
      s[c] = hn::MulAdd(av, xv, s[c]);
    }
    hn::StoreU(yv, d, y + i);
  }
  if (i < m) {
    // LoadN zeroes the lanes past `count`, so they do not contribute.
    const size_t count = m - i;
    const V xv = hn::LoadN(d, x + i, count);
    V yv = hn::LoadN(d, y + i, count);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadN(d, c_ptr[c] + i, count);
      yv = hn::MulAdd(av, vt[c], yv);
      s[c] = hn::MulAdd(av, xv, s[c]);
    }
    hn::StoreN(yv, d, y + i, count);
  }
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) dots[c] = hn::ReduceSum(d, s[c]);
}

// Complex (pointers to interleaved reals, m in complex elements):
//   y[0:m] += R * t[0:kCols]   and   dots[c] = dot(op(R[:, c]), x[0:m]),
// where op conjugates if kConj. See gemv-inl.h for the AddSub/Reverse2
// formulation of both parts.
template <size_t kCols, bool kConj, typename R>
HWY_INLINE void AxpyDotColsComplex(size_t m, const R* HWY_RESTRICT const* cols,
                                   const R* t, const R* HWY_RESTRICT x, R* y,
                                   std::complex<R>* HWY_RESTRICT dots) {
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

  // Note: no lambda here; capturing the accumulator arrays by reference
  // prevents compilers from keeping them in registers.
  //
  // With 4 columns, the loop has more live values than 16-register targets
  // hold. The broadcasts of t are therefore re-read from memory in every
  // iteration (cheap broadcast loads) rather than kept in registers: y is
  // deliberately not HWY_RESTRICT, so its stores might modify t and the
  // compiler cannot hoist them. Otherwise, register allocation may instead
  // keep an accumulator in memory, putting a store-to-load round trip into
  // its dependency chain (30% slower).
  size_t i = 0;
  for (; i + N <= m2; i += N) {
    const V xv = hn::LoadU(d, x + i);
    const V xs = hn::Reverse2(d, xv);
    V acc_re = hn::Zero(d), acc_im = hn::Zero(d);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadU(d, c_ptr[c] + i);
      acc_re = hn::MulAdd(av, hn::Set(d, t[2 * c]), acc_re);
      acc_im = hn::MulAdd(av, hn::Set(d, t[2 * c + 1]), acc_im);
      p[c] = hn::MulAdd(av, xv, p[c]);
      q[c] = hn::MulAdd(av, xs, q[c]);
    }
    const V prod = hn::AddSub(acc_re, hn::Reverse2(d, acc_im));
    hn::StoreU(hn::Add(hn::LoadU(d, y + i), prod), d, y + i);
  }
  if (i < m2) {
    // LoadN zeroes the lanes past `count`, so they do not contribute.
    const size_t count = m2 - i;
    const V xv = hn::LoadN(d, x + i, count);
    const V xs = hn::Reverse2(d, xv);
    V acc_re = hn::Zero(d), acc_im = hn::Zero(d);
    HWY_UNROLL(16)
    for (size_t c = 0; c < kCols; ++c) {
      const V av = hn::LoadN(d, c_ptr[c] + i, count);
      acc_re = hn::MulAdd(av, hn::Set(d, t[2 * c]), acc_re);
      acc_im = hn::MulAdd(av, hn::Set(d, t[2 * c + 1]), acc_im);
      p[c] = hn::MulAdd(av, xv, p[c]);
      q[c] = hn::MulAdd(av, xs, q[c]);
    }
    const V prod = hn::AddSub(acc_re, hn::Reverse2(d, acc_im));
    hn::StoreN(hn::Add(hn::LoadN(d, y + i, count), prod), d, y + i, count);
  }

  // +1 in even lanes, -1 in odd lanes.
  const V alternate = hn::OddEven(hn::Set(d, R(-1)), hn::Set(d, R(1)));
  HWY_UNROLL(16)
  for (size_t c = 0; c < kCols; ++c) {
    if (kConj) {
      dots[c] = {hn::ReduceSum(d, p[c]),
                 hn::ReduceSum(d, hn::Mul(q[c], alternate))};
    } else {
      dots[c] = {hn::ReduceSum(d, hn::Mul(p[c], alternate)),
                 hn::ReduceSum(d, q[c])};
    }
  }
}

// Typed entry point: y[0:m] += R * t[0:kCols] and dots[c] = sum over i of
// op(R(i, c)) * x[i], where op conjugates if kConj; the columns of R are
// cols[c].
template <size_t kCols, bool kConj, typename T>
HWY_INLINE void AxpyDotCols(size_t m, const T* HWY_RESTRICT const* cols,
                            const T* HWY_RESTRICT t, const T* HWY_RESTRICT x,
                            T* HWY_RESTRICT y, T* HWY_RESTRICT dots) {
  if constexpr (IsComplex<T>::value) {
    using R = RealT<T>;
    const R* rcols[kCols];
    for (size_t c = 0; c < kCols; ++c) {
      rcols[c] = reinterpret_cast<const R*>(cols[c]);
    }
    AxpyDotColsComplex<kCols, kConj>(m, rcols, reinterpret_cast<const R*>(t),
                                     reinterpret_cast<const R*>(x),
                                     reinterpret_cast<R*>(y), dots);
  } else {
    AxpyDotColsReal<kCols>(m, cols, t, x, y, dots);
  }
}

// Applies the rectangle of a strip: rows [row0, row0 + m) of columns
// [j, j + kCols).
template <size_t kCols, bool kHermitian, typename T>
HWY_INLINE void SymvStrip(size_t row0, size_t m, size_t j, T alpha,
                          const T* HWY_RESTRICT a, size_t lda,
                          const T* HWY_RESTRICT x, T* HWY_RESTRICT y) {
  if (m == 0) return;
  T t[kCols];
  T dots[kCols];
  for (size_t c = 0; c < kCols; ++c) t[c] = alpha * x[j + c];
  const T* cols[kCols];
  for (size_t c = 0; c < kCols; ++c) cols[c] = a + row0 + (j + c) * lda;
  AxpyDotCols<kCols, kHermitian>(m, cols, t, x + row0, y + row0, dots);
  for (size_t c = 0; c < kCols; ++c) y[j + c] += alpha * dots[c];
}

template <bool kHermitian, typename T>
void SymvContiguous(bool lower, size_t n, T alpha, const T* HWY_RESTRICT a,
                    size_t lda, const T* HWY_RESTRICT x, T beta,
                    T* HWY_RESTRICT y) {
  ScaleVector(n, beta, y);
  if (IsZero(alpha)) return;

  constexpr size_t kCols = SymvCols<T>();
  for (size_t j = 0; j < n; j += kCols) {
    const size_t cols = HWY_MIN(kCols, n - j);

    // Diagonal block [j, j + cols)^2, element by element from the stored
    // triangle (mirrored and conjugated as needed).
    for (size_t jj = j; jj < j + cols; ++jj) {
      for (size_t ii = j; ii < j + cols; ++ii) {
        const bool stored = lower ? (ii >= jj) : (ii <= jj);
        T v = stored ? a[ii + jj * lda] : a[jj + ii * lda];
        if (kHermitian) {
          if (ii == jj) {
            v = T(std::real(v));
          } else if (!stored) {
            v = Conj(v);
          }
        }
        y[ii] += alpha * v * x[jj];
      }
    }

    // The rectangle of stored elements below (lower) or above (upper).
    const size_t row0 = lower ? j + cols : 0;
    const size_t m = lower ? n - row0 : j;
    if (cols == kCols) {
      SymvStrip<kCols, kHermitian>(row0, m, j, alpha, a, lda, x, y);
    } else {
      for (size_t c = 0; c < cols; ++c) {
        SymvStrip<1, kHermitian>(row0, m, j + c, alpha, a, lda, x, y);
      }
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#endif  // HWYBLAS_SRC_SYMV_INL_H_
