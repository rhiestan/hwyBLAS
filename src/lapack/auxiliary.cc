// LAPACK's auxiliary routines for vectors and matrices: xLACGV, xLACPY,
// xLAPY2, xLAPY3, xLADIV, ILAxLC and ILAxLR.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLACGV: conjugates the complex vector X of length N (with increment INCX).
// Does nothing for real T.
template <typename T>
void Lacgv(blasint n, T* x, blasint incx) {
  if constexpr (IsComplex<T>::value) {
    const Vector<T> X(x);
    if (incx == 1) {
      for (blasint i = 1; i <= n; ++i) X(i) = std::conj(X(i));
    } else {
      blasint ioff = 1;
      if (incx < 0) ioff = 1 - (n - 1) * incx;
      for (blasint i = 1; i <= n; ++i) {
        X(ioff) = std::conj(X(ioff));
        ioff = ioff + incx;
      }
    }
  } else {
    (void)n, (void)x, (void)incx;
  }
}

// xLACPY: copies all (UPLO other than 'U' or 'L') or part of the M-by-N
// matrix A to B: the upper triangle or trapezoid (UPLO = 'U') or the lower
// one (UPLO = 'L').
template <typename T>
void Lacpy(char uplo, blasint m, blasint n, const T* a, blasint lda, T* b,
           blasint ldb) {
  const Matrix<const T> A(a, lda);
  const Matrix<T> B(b, ldb);
  if (Lsame(uplo, 'U')) {
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = 1; i <= std::min(j, m); ++i) B(i, j) = A(i, j);
    }
  } else if (Lsame(uplo, 'L')) {
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = j; i <= m; ++i) B(i, j) = A(i, j);
    }
  } else {
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = 1; i <= m; ++i) B(i, j) = A(i, j);
    }
  }
}

// xLAPY2: sqrt(x**2 + y**2), taking care not to cause unnecessary overflow
// and unnecessary underflow. Returns x or y if it is NaN.
template <typename R>
R Lapy2(R x, R y) {
  const R zero = R(0), one = R(1);
  R result = zero;
  const bool x_is_nan = std::isnan(x);
  const bool y_is_nan = std::isnan(y);
  if (x_is_nan) result = x;
  if (y_is_nan) result = y;
  const R hugeval = Lamch<R>('O');

  if (!(x_is_nan || y_is_nan)) {
    const R xabs = std::abs(x);
    const R yabs = std::abs(y);
    const R w = std::max(xabs, yabs);
    const R z = std::min(xabs, yabs);
    if (z == zero || w > hugeval) {
      result = w;
    } else {
      result = w * std::sqrt(one + (z / w) * (z / w));
    }
  }
  return result;
}

// xLAPY3: sqrt(x**2 + y**2 + z**2), taking care not to cause unnecessary
// overflow and unnecessary underflow.
template <typename R>
R Lapy3(R x, R y, R z) {
  const R zero = R(0);
  const R hugeval = Lamch<R>('O');
  const R xabs = std::abs(x);
  const R yabs = std::abs(y);
  const R zabs = std::abs(z);
  const R w = std::max(std::max(xabs, yabs), zabs);
  if (w == zero || w > hugeval) {
    // W can be zero for max(0,nan,0); adding all three entries together
    // will make sure NaN will not disappear.
    return xabs + yabs + zabs;
  }
  return w * std::sqrt((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) +
                       (zabs / w) * (zabs / w));
}

namespace {

// xLADIV2.
template <typename R>
R Ladiv2(R a, R b, R c, R d, R r, R t) {
  const R zero = R(0);
  if (r != zero) {
    const R br = b * r;
    if (br != zero) return (a + br) * t;
    return a * t + (b * t) * r;
  }
  return (a + d * (b / c)) * t;
}

// xLADIV1.
template <typename R>
void Ladiv1(R a, R b, R c, R d, R* p, R* q) {
  const R one = R(1);
  const R r = d / c;
  const R t = one / (c + d * r);
  *p = Ladiv2(a, b, c, d, r, t);
  a = -a;
  *q = Ladiv2(b, a, c, d, r, t);
}

}  // namespace

// xLADIV: performs complex division in real arithmetic,
//
//     p + i*q = (a + i*b) / (c + i*d),
//
// with the algorithm of M. Baudin and R. L. Smith, "A Robust Complex
// Division in Scilab" (2012), avoiding unnecessary overflow.
template <typename R>
void Ladiv(R a, R b, R c, R d, R* p, R* q) {
  const R bs = R(2), half = R(0.5), two = R(2);

  R aa = a;
  R bb = b;
  R cc = c;
  R dd = d;
  const R ab = std::max(std::abs(a), std::abs(b));
  const R cd = std::max(std::abs(c), std::abs(d));
  R s = R(1);

  const R ov = Lamch<R>('O');
  const R un = Lamch<R>('S');
  const R eps = Lamch<R>('E');
  const R be = bs / (eps * eps);

  if (ab >= half * ov) {
    aa = half * aa;
    bb = half * bb;
    s = two * s;
  }
  if (cd >= half * ov) {
    cc = half * cc;
    dd = half * dd;
    s = half * s;
  }
  if (ab <= un * bs / eps) {
    aa = aa * be;
    bb = bb * be;
    s = s / be;
  }
  if (cd <= un * bs / eps) {
    cc = cc * be;
    dd = dd * be;
    s = s * be;
  }
  if (std::abs(d) <= std::abs(c)) {
    Ladiv1(aa, bb, cc, dd, p, q);
  } else {
    Ladiv1(bb, aa, dd, cc, p, q);
    *q = -*q;
  }
  *p = *p * s;
  *q = *q * s;
}

// CLADIV / ZLADIV: x / y for complex x and y, without unnecessary overflow
// (see xLADIV).
template <typename T>
T Ladiv(T x, T y) {
  RealT<T> zr, zi;
  Ladiv(x.real(), x.imag(), y.real(), y.imag(), &zr, &zi);
  return T(zr, zi);
}

// ILAxLC: scans the M-by-N matrix A for its last non-zero column; returns
// 0 if A is zero (or N = 0).
template <typename T>
blasint Ilalc(blasint m, blasint n, const T* a, blasint lda) {
  const T zero = T(0);
  const Matrix<const T> A(a, lda);
  // Quick test for the common case where one corner is non-zero.
  if (n == 0) return n;
  if (A(1, n) != zero || A(m, n) != zero) return n;
  // Now scan each column from the end, returning with the first non-zero.
  blasint ilalc = n;
  for (; ilalc >= 1; --ilalc) {
    for (blasint i = 1; i <= m; ++i) {
      if (A(i, ilalc) != zero) return ilalc;
    }
  }
  return ilalc;
}

// ILAxLR: scans the M-by-N matrix A for its last non-zero row; returns 0 if
// A is zero (or M = 0).
template <typename T>
blasint Ilalr(blasint m, blasint n, const T* a, blasint lda) {
  const T zero = T(0);
  const Matrix<const T> A(a, lda);
  // Quick test for the common case where one corner is non-zero.
  if (m == 0) return m;
  if (A(m, 1) != zero || A(m, n) != zero) return m;
  // Scan up each column tracking the last zero row seen.
  blasint ilalr = 0;
  for (blasint j = 1; j <= n; ++j) {
    blasint i = m;
    while (i >= 1 && A(std::max<blasint>(i, 1), j) == zero) i = i - 1;
    ilalr = std::max(ilalr, i);
  }
  return ilalr;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                   \
  template void Lacgv<T>(blasint, T*, blasint);                         \
  template void Lacpy<T>(char, blasint, blasint, const T*, blasint, T*, \
                         blasint);                                      \
  template blasint Ilalc<T>(blasint, blasint, const T*, blasint);       \
  template blasint Ilalr<T>(blasint, blasint, const T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

template float Lapy2<float>(float, float);
template double Lapy2<double>(double, double);
template float Lapy3<float>(float, float, float);
template double Lapy3<double>(double, double, double);
template void Ladiv<float>(float, float, float, float, float*, float*);
template void Ladiv<double>(double, double, double, double, double*, double*);
template c32 Ladiv<c32>(c32, c32);
template c64 Ladiv<c64>(c64, c64);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

HWYBLAS_API void clacgv_(const hwyblas_int* n, float* x,
                         const hwyblas_int* incx) {
  hwyblas::lapack::Lacgv(*n, As<c32>(x), *incx);
}
HWYBLAS_API void zlacgv_(const hwyblas_int* n, double* x,
                         const hwyblas_int* incx) {
  hwyblas::lapack::Lacgv(*n, As<c64>(x), *incx);
}

#define HWYBLAS_LAPACK_AUXILIARY(prefix, T, R)                               \
  HWYBLAS_API void prefix##lacpy_(                                           \
      const char* uplo, const hwyblas_int* m, const hwyblas_int* n,          \
      const R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb) {    \
    hwyblas::lapack::Lacpy(*uplo, *m, *n, As<const T>(a), *lda, As<T>(b),    \
                           *ldb);                                            \
  }                                                                          \
  HWYBLAS_API hwyblas_int ila##prefix##lc_(const hwyblas_int* m,             \
                                           const hwyblas_int* n, const R* a, \
                                           const hwyblas_int* lda) {         \
    return hwyblas::lapack::Ilalc(*m, *n, As<const T>(a), *lda);             \
  }                                                                          \
  HWYBLAS_API hwyblas_int ila##prefix##lr_(const hwyblas_int* m,             \
                                           const hwyblas_int* n, const R* a, \
                                           const hwyblas_int* lda) {         \
    return hwyblas::lapack::Ilalr(*m, *n, As<const T>(a), *lda);             \
  }

HWYBLAS_LAPACK_AUXILIARY(s, float, float)
HWYBLAS_LAPACK_AUXILIARY(d, double, double)
HWYBLAS_LAPACK_AUXILIARY(c, c32, float)
HWYBLAS_LAPACK_AUXILIARY(z, c64, double)

#undef HWYBLAS_LAPACK_AUXILIARY

#define HWYBLAS_LAPACK_AUXILIARY_REAL(prefix, R)                      \
  HWYBLAS_API R prefix##lapy2_(const R* x, const R* y) {              \
    return hwyblas::lapack::Lapy2(*x, *y);                            \
  }                                                                   \
  HWYBLAS_API R prefix##lapy3_(const R* x, const R* y, const R* z) {  \
    return hwyblas::lapack::Lapy3(*x, *y, *z);                        \
  }                                                                   \
  HWYBLAS_API void prefix##ladiv_(const R* a, const R* b, const R* c, \
                                  const R* d, R* p, R* q) {           \
    hwyblas::lapack::Ladiv(*a, *b, *c, *d, p, q);                     \
  }

HWYBLAS_LAPACK_AUXILIARY_REAL(s, float)
HWYBLAS_LAPACK_AUXILIARY_REAL(d, double)

#undef HWYBLAS_LAPACK_AUXILIARY_REAL

}  // extern "C"
