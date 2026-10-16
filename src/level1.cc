// Level 1 routines: vector operations.
//
// Unit-stride vectors use the Highway kernels of level1-inl.h; other
// increments use plain loops (they are memory-bound and rarely used).
// Complex vectors are processed as arrays of 2n reals wherever the operation
// is the same for real and imaginary parts.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/level1.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "src/level1-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwyblas {
namespace HWY_NAMESPACE {

// Operations on all four types.
#define HWYBLAS_LEVEL1(SUFFIX, T)                                             \
  void Axpy##SUFFIX(size_t n, T alpha, const T* x, T* y) {                    \
    AxpyContiguous(n, alpha, x, y);                                           \
  }                                                                           \
  void Scal##SUFFIX(size_t n, T alpha, T* x) { ScalContiguous(n, alpha, x); } \
  size_t Iamax##SUFFIX(size_t n, const T* x) {                                \
    return ArgAbsExtremum<true>(n, x);                                        \
  }                                                                           \
  size_t Iamin##SUFFIX(size_t n, const T* x) {                                \
    return ArgAbsExtremum<false>(n, x);                                       \
  }                                                                           \
  void Axpby##SUFFIX(size_t n, T alpha, const T* x, T beta, T* y) {           \
    if (IsZero(beta)) {                                                       \
      AxpbyContiguous<false, false>(n, alpha, x, beta, y);                    \
    } else {                                                                  \
      AxpbyContiguous<false, true>(n, alpha, x, beta, y);                     \
    }                                                                         \
  }
HWYBLAS_LEVEL1(F32, float)
HWYBLAS_LEVEL1(F64, double)
HWYBLAS_LEVEL1(C32, c32)
HWYBLAS_LEVEL1(C64, c64)
#undef HWYBLAS_LEVEL1

// Operations on reals only; complex vectors pass their 2n reals.
#define HWYBLAS_LEVEL1_REAL(SUFFIX, R)                                      \
  R Asum##SUFFIX(size_t len, const R* x) { return AbsSum(len, x); }         \
  double SumSquares##SUFFIX(size_t len, const R* x) {                       \
    return SumSquares(len, x);                                              \
  }                                                                         \
  void Swap##SUFFIX(size_t len, R* x, R* y) { Swap(len, x, y); }            \
  void Rotate##SUFFIX(size_t len, R* x, R* y, R h11, R h12, R h21, R h22) { \
    Rotate(len, x, y, h11, h12, h21, h22);                                  \
  }                                                                         \
  R Dot##SUFFIX(size_t n, const R* x, const R* y) {                         \
    return DotReal<R>(n, x, y);                                             \
  }                                                                         \
  double DotWide##SUFFIX(size_t n, const R* x, const R* y) {                \
    return DotReal<double>(n, x, y);                                        \
  }                                                                         \
  std::complex<R> Dotu##SUFFIX(size_t n, const R* x, const R* y) {          \
    return DotComplex<false>(n, x, y);                                      \
  }                                                                         \
  std::complex<R> Dotc##SUFFIX(size_t n, const R* x, const R* y) {          \
    return DotComplex<true>(n, x, y);                                       \
  }
HWYBLAS_LEVEL1_REAL(F32, float)
HWYBLAS_LEVEL1_REAL(F64, double)
#undef HWYBLAS_LEVEL1_REAL

}  // namespace HWY_NAMESPACE
}  // namespace hwyblas
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include <string.h>

#include <cfloat>
#include <cmath>
#include <limits>
#include <utility>

#include "src/api.h"

namespace hwyblas {

HWYBLAS_EXPORT_BY_TYPE(Axpy);
HWYBLAS_EXPORT_BY_TYPE(Scal);
HWYBLAS_EXPORT_BY_TYPE(Iamax);
HWYBLAS_EXPORT_BY_TYPE(Iamin);
HWYBLAS_EXPORT_BY_TYPE(Axpby);
HWY_EXPORT(AsumF32);
HWY_EXPORT(AsumF64);
HWY_EXPORT(SumSquaresF32);
HWY_EXPORT(SumSquaresF64);
HWY_EXPORT(SwapF32);
HWY_EXPORT(SwapF64);
HWY_EXPORT(RotateF32);
HWY_EXPORT(RotateF64);
HWY_EXPORT(DotF32);
HWY_EXPORT(DotF64);
HWY_EXPORT(DotWideF32);
HWY_EXPORT(DotWideF64);
HWY_EXPORT(DotuF32);
HWY_EXPORT(DotuF64);
HWY_EXPORT(DotcF32);
HWY_EXPORT(DotcF64);

namespace {

// Element i of a strided vector (the start already adjusted for inc < 0).
template <typename T>
T& At(T* start, size_t i, blasint inc) {
  return start[static_cast<ptrdiff_t>(i) * inc];
}

// ---------------------------------------------------------------------------
// AXPY, SCAL, COPY, SWAP

// y := alpha * x + y.
template <typename T>
void Axpy(blasint n, T alpha, const T* x, blasint incx, T* y, blasint incy) {
  if (n <= 0 || IsZero(alpha)) return;
  if (incx == 1 && incy == 1) {
    return HWYBLAS_DISPATCH_BY_TYPE(T, Axpy, n, alpha, x, y);
  }
  const T* xs = VectorStart(x, n, incx);
  T* ys = VectorStart(y, n, incy);
  for (blasint i = 0; i < n; ++i) At(ys, i, incy) += alpha * At(xs, i, incx);
}

// y := alpha * x + beta * y (AXPBY, an OpenBLAS extension). As in OpenBLAS,
// x is not read if alpha == 0 and y is not read if beta == 0; there are no
// argument checks, and negative increments are allowed.
template <typename T>
void Axpby(blasint n, T alpha, const T* x, blasint incx, T beta, T* y,
           blasint incy) {
  if (n <= 0) return;
  T* ys = VectorStart(y, n, incy);
  if (IsZero(alpha)) {
    for (blasint i = 0; i < n; ++i) {
      T& yi = At(ys, i, incy);
      yi = IsZero(beta) ? T(0) : beta * yi;
    }
    return;
  }
  if (incx == 1 && incy == 1) {
    return HWYBLAS_DISPATCH_BY_TYPE(T, Axpby, n, alpha, x, beta, y);
  }
  const T* xs = VectorStart(x, n, incx);
  for (blasint i = 0; i < n; ++i) {
    T& yi = At(ys, i, incy);
    yi = IsZero(beta) ? alpha * At(xs, i, incx)
                      : alpha * At(xs, i, incx) + beta * yi;
  }
}

// x := alpha * x. Like the reference BLAS, alpha = 0 multiplies (so NaN
// stays NaN), and incx <= 0 is a no-op.
template <typename T>
void Scal(blasint n, T alpha, T* x, blasint incx) {
  if (n <= 0 || incx <= 0 || IsOne(alpha)) return;
  if (incx == 1) return HWYBLAS_DISPATCH_BY_TYPE(T, Scal, n, alpha, x);
  for (blasint i = 0; i < n; ++i) At(x, i, incx) *= alpha;
}

// x := alpha * x for a complex x and a real alpha (CSSCAL, ZDSCAL).
template <typename R>
void ScalComplexByReal(blasint n, R alpha, std::complex<R>* x, blasint incx) {
  if (n <= 0 || incx <= 0 || alpha == R(1)) return;
  if (incx == 1) {
    return HWYBLAS_DISPATCH_BY_TYPE(R, Scal, 2 * static_cast<size_t>(n), alpha,
                                    reinterpret_cast<R*>(x));
  }
  for (blasint i = 0; i < n; ++i) At(x, i, incx) *= alpha;
}

// y := x.
template <typename T>
void Copy(blasint n, const T* x, blasint incx, T* y, blasint incy) {
  if (n <= 0) return;
  if (incx == 1 && incy == 1) {
    memcpy(y, x, static_cast<size_t>(n) * sizeof(T));
    return;
  }
  const T* xs = VectorStart(x, n, incx);
  T* ys = VectorStart(y, n, incy);
  for (blasint i = 0; i < n; ++i) At(ys, i, incy) = At(xs, i, incx);
}

// x <-> y.
template <typename T>
void Swap(blasint n, T* x, blasint incx, T* y, blasint incy) {
  if (n <= 0) return;
  if (incx == 1 && incy == 1) {
    using R = RealT<T>;
    HWYBLAS_DISPATCH_BY_REAL(R, Swap, RealCount<T>(n), reinterpret_cast<R*>(x),
                             reinterpret_cast<R*>(y));
    return;
  }
  T* xs = VectorStart(x, n, incx);
  T* ys = VectorStart(y, n, incy);
  for (blasint i = 0; i < n; ++i) std::swap(At(xs, i, incx), At(ys, i, incy));
}

// ---------------------------------------------------------------------------
// DOT

// sum x_i * y_i for real vectors, accumulated in A (R or double).
template <typename A, typename R>
A Dot(blasint n, const R* x, blasint incx, const R* y, blasint incy) {
  if (n <= 0) return A(0);
  if (incx == 1 && incy == 1) {
    if constexpr (std::is_same_v<A, R>) {
      return HWYBLAS_DISPATCH_BY_REAL(R, Dot, n, x, y);
    } else {
      return HWYBLAS_DISPATCH_BY_REAL(R, DotWide, n, x, y);
    }
  }
  const R* xs = VectorStart(x, n, incx);
  const R* ys = VectorStart(y, n, incy);
  A sum = 0;
  for (blasint i = 0; i < n; ++i) {
    sum += static_cast<A>(At(xs, i, incx)) * static_cast<A>(At(ys, i, incy));
  }
  return sum;
}

// sum op(x_i) * y_i for complex vectors; op conjugates if conj.
template <typename R>
std::complex<R> DotComplex(bool conj, blasint n, const std::complex<R>* x,
                           blasint incx, const std::complex<R>* y,
                           blasint incy) {
  if (n <= 0) return 0;
  if (incx == 1 && incy == 1) {
    const R* xr = reinterpret_cast<const R*>(x);
    const R* yr = reinterpret_cast<const R*>(y);
    return conj ? HWYBLAS_DISPATCH_BY_REAL(R, Dotc, n, xr, yr)
                : HWYBLAS_DISPATCH_BY_REAL(R, Dotu, n, xr, yr);
  }
  const std::complex<R>* xs = VectorStart(x, n, incx);
  const std::complex<R>* ys = VectorStart(y, n, incy);
  std::complex<R> sum = 0;
  for (blasint i = 0; i < n; ++i) {
    const std::complex<R> xi = At(xs, i, incx);
    sum += (conj ? std::conj(xi) : xi) * At(ys, i, incy);
  }
  return sum;
}

// ---------------------------------------------------------------------------
// NRM2, ASUM, I?AMAX, I?AMIN

// floor(v / 2) and ceil(v / 2) for integers of any sign.
constexpr int FloorHalf(int v) { return v >= 0 ? v / 2 : -((1 - v) / 2); }
constexpr int CeilHalf(int v) { return -FloorHalf(-v); }

// Euclidean norm of n elements of kReals reals each (1 for real, 2 for
// complex), element i starting at x[i * stride], by the reference BLAS
// algorithm: Blue's method with separate accumulators for small, medium and
// large values, which neither overflows nor underflows harmfully.
template <typename R>
R BlueNorm(size_t n, size_t reals, const R* x, size_t stride) {
  using L = std::numeric_limits<R>;
  const int min_exp = L::min_exponent, max_exp = L::max_exponent;
  const int digits = L::digits;
  // Thresholds and scaling factors (powers of two) of the reference BLAS.
  const R tsml = std::ldexp(R(1), CeilHalf(min_exp - 1));
  const R tbig = std::ldexp(R(1), FloorHalf(max_exp - digits + 1));
  const R ssml = std::ldexp(R(1), -FloorHalf(min_exp - digits));
  const R sbig = std::ldexp(R(1), -CeilHalf(max_exp + digits - 1));

  R asml = 0, amed = 0, abig = 0;
  bool notbig = true;
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < reals; ++k) {
      const R ax = std::abs(x[i * stride + k]);
      if (ax > tbig) {
        abig += (ax * sbig) * (ax * sbig);
        notbig = false;
      } else if (ax < tsml) {
        if (notbig) asml += (ax * ssml) * (ax * ssml);
      } else {
        amed += ax * ax;
      }
    }
  }

  R scl, sumsq;
  if (abig > 0) {
    // Combine abig and amed if abig > 0.
    if (amed > 0 || std::isnan(amed)) abig += (amed * sbig) * sbig;
    scl = R(1) / sbig;
    sumsq = abig;
  } else if (asml > 0) {
    // Combine amed and asml if asml > 0.
    if (amed > 0 || std::isnan(amed)) {
      amed = std::sqrt(amed);
      asml = std::sqrt(asml) / ssml;
      const R ymin = asml > amed ? amed : asml;
      const R ymax = asml > amed ? asml : amed;
      scl = 1;
      sumsq = ymax * ymax * (R(1) + (ymin / ymax) * (ymin / ymax));
    } else {
      scl = R(1) / ssml;
      sumsq = asml;
    }
  } else {
    scl = 1;
    sumsq = amed;
  }
  return scl * std::sqrt(sumsq);
}

// Euclidean norm of n elements of T.
template <typename T>
RealT<T> Nrm2(blasint n, const T* x, blasint incx) {
  using R = RealT<T>;
  if (n < 1 || incx < 1) return 0;
  const R* xr = reinterpret_cast<const R*>(x);
  const size_t reals = RealCount<T>(1);
  const size_t stride = reals * static_cast<size_t>(incx);  // in reals

  // Fast path: plain sum of squares, accumulated in double.
  double sumsq = 0;
  if (incx == 1) {
    sumsq = HWYBLAS_DISPATCH_BY_REAL(R, SumSquares, RealCount<T>(n), xr);
  } else {
    for (blasint i = 0; i < n; ++i) {
      for (size_t k = 0; k < reals; ++k) {
        const double v = xr[static_cast<size_t>(i) * stride + k];
        sumsq += v * v;
      }
    }
  }
  // Float input summed in double can neither overflow nor underflow. For
  // double input the fast sum is accurate unless it overflowed or is so
  // small that squares of the elements may have underflowed; then fall back
  // to the safe algorithm.
  if constexpr (sizeof(R) == 4) {
    return static_cast<R>(std::sqrt(sumsq));
  } else {
    const double kSafeMin = std::ldexp(1.0, -500);
    if (sumsq >= kSafeMin && sumsq <= DBL_MAX) return std::sqrt(sumsq);
    return BlueNorm(static_cast<size_t>(n), reals, xr, stride);
  }
}

// sum |x_i| (|re| + |im| for complex).
template <typename T>
RealT<T> Asum(blasint n, const T* x, blasint incx) {
  using R = RealT<T>;
  if (n < 1 || incx < 1) return 0;
  const R* xr = reinterpret_cast<const R*>(x);
  if (incx == 1) return HWYBLAS_DISPATCH_BY_REAL(R, Asum, RealCount<T>(n), xr);
  R sum = 0;
  for (blasint i = 0; i < n; ++i) {
    for (size_t k = 0; k < RealCount<T>(1); ++k) {
      sum += std::abs(xr[RealCount<T>(static_cast<size_t>(i) * incx) + k]);
    }
  }
  return sum;
}

// |re| + |im| for complex, |x| for real values.
template <typename T>
RealT<T> Magnitude1(T x) {
  if constexpr (IsComplex<T>::value) {
    return std::abs(x.real()) + std::abs(x.imag());
  } else {
    return std::abs(x);
  }
}

// 1-based index of the first element of largest (kMax) or smallest
// magnitude; 0 if n < 1 or incx <= 0.
template <bool kMax, typename T>
blasint Iamax(blasint n, const T* x, blasint incx) {
  if (n < 1 || incx <= 0) return 0;
  if (n == 1) return 1;
  if (incx == 1) {
    const size_t index = kMax ? HWYBLAS_DISPATCH_BY_TYPE(T, Iamax, n, x)
                              : HWYBLAS_DISPATCH_BY_TYPE(T, Iamin, n, x);
    return static_cast<blasint>(index) + 1;
  }
  blasint best = 1;
  RealT<T> best_value = Magnitude1(x[0]);
  for (blasint i = 1; i < n; ++i) {
    const RealT<T> v = Magnitude1(At(x, i, incx));
    if (kMax ? (v > best_value) : (v < best_value)) {
      best = i + 1;
      best_value = v;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// ROT, ROTM

// Applies [x; y] := [h11 h12; h21 h22] [x; y] element-wise (for complex
// vectors with real coefficients, to real and imaginary parts alike).
template <typename T>
void Transform(blasint n, T* x, blasint incx, T* y, blasint incy, RealT<T> h11,
               RealT<T> h12, RealT<T> h21, RealT<T> h22) {
  using R = RealT<T>;
  if (n <= 0) return;
  if (incx == 1 && incy == 1) {
    HWYBLAS_DISPATCH_BY_REAL(R, Rotate, RealCount<T>(n),
                             reinterpret_cast<R*>(x), reinterpret_cast<R*>(y),
                             h11, h12, h21, h22);
    return;
  }
  T* xs = VectorStart(x, n, incx);
  T* ys = VectorStart(y, n, incy);
  for (blasint i = 0; i < n; ++i) {
    const T xi = At(xs, i, incx), yi = At(ys, i, incy);
    At(xs, i, incx) = h11 * xi + h12 * yi;
    At(ys, i, incy) = h21 * xi + h22 * yi;
  }
}

// Plane rotation: x := c x + s y, y := c y - s x.
template <typename T>
void Rot(blasint n, T* x, blasint incx, T* y, blasint incy, RealT<T> c,
         RealT<T> s) {
  Transform(n, x, incx, y, incy, c, s, -s, c);
}

// Modified Givens transformation H given by param = (flag, h11, h21, h12,
// h22); see the reference BLAS for the meaning of flag.
template <typename R>
void Rotm(blasint n, R* x, blasint incx, R* y, blasint incy, const R* param) {
  const R flag = param[0];
  if (n <= 0 || flag == R(-2)) return;
  if (flag < 0) {
    Transform(n, x, incx, y, incy, param[1], param[3], param[2], param[4]);
  } else if (flag == 0) {
    Transform(n, x, incx, y, incy, R(1), param[3], param[2], R(1));
  } else {
    Transform(n, x, incx, y, incy, param[1], R(1), R(-1), param[4]);
  }
}

// ---------------------------------------------------------------------------
// ROTG, ROTMG (scalar; ports of the reference BLAS 3.10+ algorithms)

// Real Givens rotation: [c s; -s c] [a; b] = [r; 0]. On return a = r and
// b = z, from which c and s can be reconstructed.
template <typename R>
void Rotg(R* a, R* b, R* c, R* s) {
  const R safmin = std::numeric_limits<R>::min();
  const R safmax = R(1) / safmin;
  const R anorm = std::abs(*a), bnorm = std::abs(*b);
  if (bnorm == 0) {
    *c = 1;
    *s = 0;
    *b = 0;
  } else if (anorm == 0) {
    *c = 0;
    *s = 1;
    *a = *b;
    *b = 1;
  } else {
    const R scl = HWY_MIN(safmax, HWY_MAX(safmin, HWY_MAX(anorm, bnorm)));
    const R sigma = std::copysign(R(1), anorm > bnorm ? *a : *b);
    const R as = *a / scl, bs = *b / scl;
    const R r = sigma * (scl * std::sqrt(as * as + bs * bs));
    *c = *a / r;
    *s = *b / r;
    R z;
    if (anorm > bnorm) {
      z = *s;
    } else if (*c != 0) {
      z = R(1) / *c;
    } else {
      z = 1;
    }
    *a = r;
    *b = z;
  }
}

template <typename R>
R AbsSq(std::complex<R> z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

// Complex Givens rotation: [c s; -conj(s) c] [a; b] = [r; 0] with real c.
// On return a = r.
template <typename R>
void RotgComplex(std::complex<R>* a, const std::complex<R>* b, R* c,
                 std::complex<R>* s) {
  using C = std::complex<R>;
  const R safmin = std::numeric_limits<R>::min();
  const R safmax = R(1) / safmin;
  const R rtmin = std::sqrt(safmin);
  const C f = *a, g = *b;
  C r;
  if (g == C(0)) {
    *c = 1;
    *s = 0;
    r = f;
  } else if (f == C(0)) {
    *c = 0;
    if (g.real() == 0) {
      const R d = std::abs(g.imag());
      *s = std::conj(g) / d;
      r = d;
    } else if (g.imag() == 0) {
      const R d = std::abs(g.real());
      *s = std::conj(g) / d;
      r = d;
    } else {
      const R g1 = HWY_MAX(std::abs(g.real()), std::abs(g.imag()));
      const R rtmax = std::sqrt(safmax / 2);
      if (g1 > rtmin && g1 < rtmax) {
        const R d = std::sqrt(AbsSq(g));
        *s = std::conj(g) / d;
        r = d;
      } else {
        const R u = HWY_MIN(safmax, HWY_MAX(safmin, g1));
        const C gs = g / u;
        const R d = std::sqrt(AbsSq(gs));
        *s = std::conj(gs) / d;
        r = d * u;
      }
    }
  } else {
    const R f1 = HWY_MAX(std::abs(f.real()), std::abs(f.imag()));
    const R g1 = HWY_MAX(std::abs(g.real()), std::abs(g.imag()));
    R rtmax = std::sqrt(safmax / 4);
    if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
      // Unscaled algorithm.
      const R f2 = AbsSq(f), g2 = AbsSq(g), h2 = f2 + g2;
      if (f2 >= h2 * safmin) {
        *c = std::sqrt(f2 / h2);
        r = f / *c;
        rtmax *= 2;
        if (f2 > rtmin && h2 < rtmax) {
          *s = std::conj(g) * (f / std::sqrt(f2 * h2));
        } else {
          *s = std::conj(g) * (r / h2);
        }
      } else {
        const R d = std::sqrt(f2 * h2);
        *c = f2 / d;
        r = (*c >= safmin) ? f / *c : f * (h2 / d);
        *s = std::conj(g) * (f / d);
      }
    } else {
      // Scaled algorithm.
      const R u = HWY_MIN(safmax, HWY_MAX(safmin, HWY_MAX(f1, g1)));
      const C gs = g / u;
      const R g2 = AbsSq(gs);
      R w, f2, h2;
      C fs;
      if (f1 / u < rtmin) {
        const R v = HWY_MIN(safmax, HWY_MAX(safmin, f1));
        w = v / u;
        fs = f / v;
        f2 = AbsSq(fs);
        h2 = f2 * w * w + g2;
      } else {
        w = 1;
        fs = f / u;
        f2 = AbsSq(fs);
        h2 = f2 + g2;
      }
      if (f2 >= h2 * safmin) {
        *c = std::sqrt(f2 / h2);
        r = fs / *c;
        rtmax *= 2;
        if (f2 > rtmin && h2 < rtmax) {
          *s = std::conj(gs) * (fs / std::sqrt(f2 * h2));
        } else {
          *s = std::conj(gs) * (r / h2);
        }
      } else {
        const R d = std::sqrt(f2 * h2);
        *c = f2 / d;
        r = (*c >= safmin) ? fs / *c : fs * (h2 / d);
        *s = std::conj(gs) * (fs / d);
      }
      *c *= w;
      r *= u;
    }
  }
  *a = r;
}

// Modified Givens rotation (reference BLAS ROTMG): computes H such that
// H [sqrt(d1) x1; sqrt(d2) y1] has a zero second component.
template <typename R>
void Rotmg(R* d1, R* d2, R* x1, R y1, R* param) {
  const R gam = 4096, gamsq = R(16777216), rgamsq = R(5.9604645e-8);
  R flag, h11 = 0, h12 = 0, h21 = 0, h22 = 0;
  if (*d1 < 0) {
    flag = -1;
    *d1 = *d2 = *x1 = 0;
  } else {
    const R p2 = *d2 * y1;
    if (p2 == 0) {
      param[0] = -2;
      return;
    }
    const R p1 = *d1 * *x1;
    const R q2 = p2 * y1;
    const R q1 = p1 * *x1;
    if (std::abs(q1) > std::abs(q2)) {
      h21 = -y1 / *x1;
      h12 = p2 / p1;
      const R u = R(1) - h12 * h21;
      // The reference leaves flag undefined if u <= 0 (impossible in exact
      // arithmetic); treat it as the zero-result case.
      flag = 0;
      if (u > 0) {
        *d1 /= u;
        *d2 /= u;
        *x1 *= u;
      } else {
        flag = -1;
        h11 = h12 = h21 = h22 = 0;
        *d1 = *d2 = *x1 = 0;
      }
    } else if (q2 < 0) {
      flag = -1;
      h11 = h12 = h21 = h22 = 0;
      *d1 = *d2 = *x1 = 0;
    } else {
      flag = 1;
      h11 = p1 / p2;
      h22 = *x1 / y1;
      const R u = R(1) + h11 * h22;
      const R temp = *d2 / u;
      *d2 = *d1 / u;
      *d1 = temp;
      *x1 = y1 * u;
    }
    // Rescale d1 and d2 into [rgamsq, gamsq].
    if (*d1 != 0) {
      while (*d1 <= rgamsq || *d1 >= gamsq) {
        if (flag == 0) {
          h11 = h22 = 1;
        } else {
          h21 = -1;
          h12 = 1;
        }
        flag = -1;
        if (*d1 <= rgamsq) {
          *d1 *= gam * gam;
          *x1 /= gam;
          h11 /= gam;
          h12 /= gam;
        } else {
          *d1 /= gam * gam;
          *x1 *= gam;
          h11 *= gam;
          h12 *= gam;
        }
      }
    }
    if (*d2 != 0) {
      while (std::abs(*d2) <= rgamsq || std::abs(*d2) >= gamsq) {
        if (flag == 0) {
          h11 = h22 = 1;
        } else {
          h21 = -1;
          h12 = 1;
        }
        flag = -1;
        if (std::abs(*d2) <= rgamsq) {
          *d2 *= gam * gam;
          h21 /= gam;
          h22 /= gam;
        } else {
          *d2 /= gam * gam;
          h21 *= gam;
          h22 *= gam;
        }
      }
    }
  }
  if (flag < 0) {
    param[1] = h11;
    param[2] = h21;
    param[3] = h12;
    param[4] = h22;
  } else if (flag == 0) {
    param[2] = h21;
    param[3] = h12;
  } else {
    param[1] = h11;
    param[4] = h22;
  }
  param[0] = flag;
}

}  // namespace
}  // namespace hwyblas

// ---------------------------------------------------------------------------
// Fortran-style entry points

using hwyblas::blasint;
using hwyblas::C32;
using hwyblas::C64;

namespace {

hwyblas_complex_float ToC(hwyblas::c32 z) { return {z.real(), z.imag()}; }
hwyblas_complex_double ToC(hwyblas::c64 z) { return {z.real(), z.imag()}; }

}  // namespace

extern "C" {

// AXPY

HWYBLAS_API void saxpy_(const blasint* n, const float* alpha, const float* x,
                        const blasint* incx, float* y, const blasint* incy) {
  hwyblas::Axpy(*n, *alpha, x, *incx, y, *incy);
}
HWYBLAS_API void daxpy_(const blasint* n, const double* alpha, const double* x,
                        const blasint* incx, double* y, const blasint* incy) {
  hwyblas::Axpy(*n, *alpha, x, *incx, y, *incy);
}
HWYBLAS_API void caxpy_(const blasint* n, const float* alpha, const float* x,
                        const blasint* incx, float* y, const blasint* incy) {
  hwyblas::Axpy(*n, *C32(alpha), C32(x), *incx, C32(y), *incy);
}
HWYBLAS_API void zaxpy_(const blasint* n, const double* alpha, const double* x,
                        const blasint* incx, double* y, const blasint* incy) {
  hwyblas::Axpy(*n, *C64(alpha), C64(x), *incx, C64(y), *incy);
}

// SCAL

HWYBLAS_API void sscal_(const blasint* n, const float* alpha, float* x,
                        const blasint* incx) {
  hwyblas::Scal(*n, *alpha, x, *incx);
}
HWYBLAS_API void dscal_(const blasint* n, const double* alpha, double* x,
                        const blasint* incx) {
  hwyblas::Scal(*n, *alpha, x, *incx);
}
HWYBLAS_API void cscal_(const blasint* n, const float* alpha, float* x,
                        const blasint* incx) {
  hwyblas::Scal(*n, *C32(alpha), C32(x), *incx);
}
HWYBLAS_API void zscal_(const blasint* n, const double* alpha, double* x,
                        const blasint* incx) {
  hwyblas::Scal(*n, *C64(alpha), C64(x), *incx);
}
HWYBLAS_API void csscal_(const blasint* n, const float* alpha, float* x,
                         const blasint* incx) {
  hwyblas::ScalComplexByReal(*n, *alpha, C32(x), *incx);
}
HWYBLAS_API void zdscal_(const blasint* n, const double* alpha, double* x,
                         const blasint* incx) {
  hwyblas::ScalComplexByReal(*n, *alpha, C64(x), *incx);
}

// COPY

HWYBLAS_API void scopy_(const blasint* n, const float* x, const blasint* incx,
                        float* y, const blasint* incy) {
  hwyblas::Copy(*n, x, *incx, y, *incy);
}
HWYBLAS_API void dcopy_(const blasint* n, const double* x, const blasint* incx,
                        double* y, const blasint* incy) {
  hwyblas::Copy(*n, x, *incx, y, *incy);
}
HWYBLAS_API void ccopy_(const blasint* n, const float* x, const blasint* incx,
                        float* y, const blasint* incy) {
  hwyblas::Copy(*n, C32(x), *incx, C32(y), *incy);
}
HWYBLAS_API void zcopy_(const blasint* n, const double* x, const blasint* incx,
                        double* y, const blasint* incy) {
  hwyblas::Copy(*n, C64(x), *incx, C64(y), *incy);
}

// SWAP

HWYBLAS_API void sswap_(const blasint* n, float* x, const blasint* incx,
                        float* y, const blasint* incy) {
  hwyblas::Swap(*n, x, *incx, y, *incy);
}
HWYBLAS_API void dswap_(const blasint* n, double* x, const blasint* incx,
                        double* y, const blasint* incy) {
  hwyblas::Swap(*n, x, *incx, y, *incy);
}
HWYBLAS_API void cswap_(const blasint* n, float* x, const blasint* incx,
                        float* y, const blasint* incy) {
  hwyblas::Swap(*n, C32(x), *incx, C32(y), *incy);
}
HWYBLAS_API void zswap_(const blasint* n, double* x, const blasint* incx,
                        double* y, const blasint* incy) {
  hwyblas::Swap(*n, C64(x), *incx, C64(y), *incy);
}

// DOT

HWYBLAS_API float sdot_(const blasint* n, const float* x, const blasint* incx,
                        const float* y, const blasint* incy) {
  return hwyblas::Dot<float>(*n, x, *incx, y, *incy);
}
HWYBLAS_API double ddot_(const blasint* n, const double* x, const blasint* incx,
                         const double* y, const blasint* incy) {
  return hwyblas::Dot<double>(*n, x, *incx, y, *incy);
}
HWYBLAS_API float sdsdot_(const blasint* n, const float* sb, const float* x,
                          const blasint* incx, const float* y,
                          const blasint* incy) {
  return static_cast<float>(static_cast<double>(*sb) +
                            hwyblas::Dot<double>(*n, x, *incx, y, *incy));
}
HWYBLAS_API double dsdot_(const blasint* n, const float* x, const blasint* incx,
                          const float* y, const blasint* incy) {
  return hwyblas::Dot<double>(*n, x, *incx, y, *incy);
}
HWYBLAS_API hwyblas_complex_float cdotu_(const blasint* n, const float* x,
                                         const blasint* incx, const float* y,
                                         const blasint* incy) {
  return ToC(hwyblas::DotComplex(false, *n, C32(x), *incx, C32(y), *incy));
}
HWYBLAS_API hwyblas_complex_float cdotc_(const blasint* n, const float* x,
                                         const blasint* incx, const float* y,
                                         const blasint* incy) {
  return ToC(hwyblas::DotComplex(true, *n, C32(x), *incx, C32(y), *incy));
}
HWYBLAS_API hwyblas_complex_double zdotu_(const blasint* n, const double* x,
                                          const blasint* incx, const double* y,
                                          const blasint* incy) {
  return ToC(hwyblas::DotComplex(false, *n, C64(x), *incx, C64(y), *incy));
}
HWYBLAS_API hwyblas_complex_double zdotc_(const blasint* n, const double* x,
                                          const blasint* incx, const double* y,
                                          const blasint* incy) {
  return ToC(hwyblas::DotComplex(true, *n, C64(x), *incx, C64(y), *incy));
}

// NRM2, ASUM

HWYBLAS_API float snrm2_(const blasint* n, const float* x,
                         const blasint* incx) {
  return hwyblas::Nrm2(*n, x, *incx);
}
HWYBLAS_API double dnrm2_(const blasint* n, const double* x,
                          const blasint* incx) {
  return hwyblas::Nrm2(*n, x, *incx);
}
HWYBLAS_API float scnrm2_(const blasint* n, const float* x,
                          const blasint* incx) {
  return hwyblas::Nrm2(*n, C32(x), *incx);
}
HWYBLAS_API double dznrm2_(const blasint* n, const double* x,
                           const blasint* incx) {
  return hwyblas::Nrm2(*n, C64(x), *incx);
}
HWYBLAS_API float sasum_(const blasint* n, const float* x,
                         const blasint* incx) {
  return hwyblas::Asum(*n, x, *incx);
}
HWYBLAS_API double dasum_(const blasint* n, const double* x,
                          const blasint* incx) {
  return hwyblas::Asum(*n, x, *incx);
}
HWYBLAS_API float scasum_(const blasint* n, const float* x,
                          const blasint* incx) {
  return hwyblas::Asum(*n, C32(x), *incx);
}
HWYBLAS_API double dzasum_(const blasint* n, const double* x,
                           const blasint* incx) {
  return hwyblas::Asum(*n, C64(x), *incx);
}

// I?AMAX

HWYBLAS_API blasint isamax_(const blasint* n, const float* x,
                            const blasint* incx) {
  return hwyblas::Iamax<true>(*n, x, *incx);
}
HWYBLAS_API blasint idamax_(const blasint* n, const double* x,
                            const blasint* incx) {
  return hwyblas::Iamax<true>(*n, x, *incx);
}
HWYBLAS_API blasint icamax_(const blasint* n, const float* x,
                            const blasint* incx) {
  return hwyblas::Iamax<true>(*n, C32(x), *incx);
}
HWYBLAS_API blasint izamax_(const blasint* n, const double* x,
                            const blasint* incx) {
  return hwyblas::Iamax<true>(*n, C64(x), *incx);
}

// ROT, ROTM

HWYBLAS_API void srot_(const blasint* n, float* x, const blasint* incx,
                       float* y, const blasint* incy, const float* c,
                       const float* s) {
  hwyblas::Rot(*n, x, *incx, y, *incy, *c, *s);
}
HWYBLAS_API void drot_(const blasint* n, double* x, const blasint* incx,
                       double* y, const blasint* incy, const double* c,
                       const double* s) {
  hwyblas::Rot(*n, x, *incx, y, *incy, *c, *s);
}
HWYBLAS_API void csrot_(const blasint* n, float* x, const blasint* incx,
                        float* y, const blasint* incy, const float* c,
                        const float* s) {
  hwyblas::Rot(*n, C32(x), *incx, C32(y), *incy, *c, *s);
}
HWYBLAS_API void zdrot_(const blasint* n, double* x, const blasint* incx,
                        double* y, const blasint* incy, const double* c,
                        const double* s) {
  hwyblas::Rot(*n, C64(x), *incx, C64(y), *incy, *c, *s);
}
HWYBLAS_API void srotm_(const blasint* n, float* x, const blasint* incx,
                        float* y, const blasint* incy, const float* param) {
  hwyblas::Rotm(*n, x, *incx, y, *incy, param);
}
HWYBLAS_API void drotm_(const blasint* n, double* x, const blasint* incx,
                        double* y, const blasint* incy, const double* param) {
  hwyblas::Rotm(*n, x, *incx, y, *incy, param);
}

// ROTG, ROTMG

HWYBLAS_API void srotg_(float* a, float* b, float* c, float* s) {
  hwyblas::Rotg(a, b, c, s);
}
HWYBLAS_API void drotg_(double* a, double* b, double* c, double* s) {
  hwyblas::Rotg(a, b, c, s);
}
HWYBLAS_API void crotg_(float* a, const float* b, float* c, float* s) {
  hwyblas::RotgComplex(C32(a), C32(b), c, C32(s));
}
HWYBLAS_API void zrotg_(double* a, const double* b, double* c, double* s) {
  hwyblas::RotgComplex(C64(a), C64(b), c, C64(s));
}
HWYBLAS_API void srotmg_(float* d1, float* d2, float* x1, const float* y1,
                         float* param) {
  hwyblas::Rotmg(d1, d2, x1, *y1, param);
}
HWYBLAS_API void drotmg_(double* d1, double* d2, double* x1, const double* y1,
                         double* param) {
  hwyblas::Rotmg(d1, d2, x1, *y1, param);
}

// CABS1

HWYBLAS_API float scabs1_(const float* z) {
  return std::abs(z[0]) + std::abs(z[1]);
}
HWYBLAS_API double dcabs1_(const double* z) {
  return std::abs(z[0]) + std::abs(z[1]);
}

// Extensions (as in OpenBLAS)

HWYBLAS_API void saxpby_(const blasint* n, const float* alpha, const float* x,
                         const blasint* incx, const float* beta, float* y,
                         const blasint* incy) {
  hwyblas::Axpby(*n, *alpha, x, *incx, *beta, y, *incy);
}
HWYBLAS_API void daxpby_(const blasint* n, const double* alpha, const double* x,
                         const blasint* incx, const double* beta, double* y,
                         const blasint* incy) {
  hwyblas::Axpby(*n, *alpha, x, *incx, *beta, y, *incy);
}
HWYBLAS_API void caxpby_(const blasint* n, const float* alpha, const float* x,
                         const blasint* incx, const float* beta, float* y,
                         const blasint* incy) {
  hwyblas::Axpby(*n, *C32(alpha), C32(x), *incx, *C32(beta), C32(y), *incy);
}
HWYBLAS_API void zaxpby_(const blasint* n, const double* alpha, const double* x,
                         const blasint* incx, const double* beta, double* y,
                         const blasint* incy) {
  hwyblas::Axpby(*n, *C64(alpha), C64(x), *incx, *C64(beta), C64(y), *incy);
}
HWYBLAS_API blasint isamin_(const blasint* n, const float* x,
                            const blasint* incx) {
  return hwyblas::Iamax<false>(*n, x, *incx);
}
HWYBLAS_API blasint idamin_(const blasint* n, const double* x,
                            const blasint* incx) {
  return hwyblas::Iamax<false>(*n, x, *incx);
}
HWYBLAS_API blasint icamin_(const blasint* n, const float* x,
                            const blasint* incx) {
  return hwyblas::Iamax<false>(*n, C32(x), *incx);
}
HWYBLAS_API blasint izamin_(const blasint* n, const double* x,
                            const blasint* incx) {
  return hwyblas::Iamax<false>(*n, C64(x), *incx);
}

}  // extern "C"
#endif  // HWY_ONCE
