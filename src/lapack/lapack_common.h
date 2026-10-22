// Helpers shared by the LAPACK routines: 1-based array views, machine
// constants, error reporting and block sizes.
//
// The LAPACK routines are ported from the Fortran reference implementation
// (LAPACK 3.12.1) line by line. To keep them comparable with the original,
// they use its names and its 1-based indexing: a Fortran array A(LDA, *) is
// accessed through a view `A` with A(i, j), and passed on as A.ptr(i, j)
// where Fortran passes A(I, J). Views are named like the Fortran arrays
// (upper case); scalars are lower case. Each routine is a template over the
// four precisions s, d, c, z (float, double, std::complex<float>,
// std::complex<double>), and returns INFO instead of setting an argument.
//
// Derived from LAPACK, Copyright (c) 1992-2023 The University of Tennessee
// and The University of Tennessee Research Foundation, The University of
// California Berkeley, and The University of Colorado Denver. Modified BSD
// license, see LICENSE-LAPACK.
#ifndef HWYBLAS_SRC_LAPACK_LAPACK_COMMON_H_
#define HWYBLAS_SRC_LAPACK_LAPACK_COMMON_H_

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "src/common.h"

namespace hwyblas {
namespace lapack {

// ---------------------------------------------------------------------------
// Views of Fortran arrays

// A(LDA, *): column-major matrix with 1-based indices.
template <typename T>
class Matrix {
 public:
  Matrix(T* data, blasint ld) : data_(data), ld_(ld) {}

  T& operator()(blasint i, blasint j) const { return *ptr(i, j); }
  // Address of A(i, j), e.g. to pass the submatrix starting there. Like the
  // Fortran original, this is also used for empty submatrices just past the
  // end of the array, which are never accessed.
  T* ptr(blasint i, blasint j) const {
    return data_ + (i - 1) + static_cast<ptrdiff_t>(j - 1) * ld_;
  }

 private:
  T* data_;
  blasint ld_;
};

// X(*): vector with 1-based indices.
template <typename T>
class Vector {
 public:
  explicit Vector(T* data) : data_(data) {}

  T& operator()(blasint i) const { return data_[i - 1]; }
  T* ptr(blasint i) const { return data_ + (i - 1); }

 private:
  T* data_;
};

// The Fortran-style interface passes complex arrays as pointers to (re, im)
// pairs of reals: As<c64>(p) is the double* p as a c64*. For real T it
// does nothing.
template <typename T, typename P>
T* As(P* p) {
  return reinterpret_cast<T*>(p);
}

// ---------------------------------------------------------------------------
// Precisions

// The precision letter of LAPACK's routine names: S, D, C or Z.
template <typename T>
constexpr char Prefix() {
  if constexpr (std::is_same_v<T, float>) {
    return 'S';
  } else if constexpr (std::is_same_v<T, double>) {
    return 'D';
  } else if constexpr (std::is_same_v<T, c32>) {
    return 'C';
  } else {
    static_assert(std::is_same_v<T, c64>, "Unsupported type");
    return 'Z';
  }
}

// The part of a routine name after the precision letter where the real and
// the complex routine differ, e.g. RealOrComplex<T>("ORGQR", "UNGQR").
template <typename T>
constexpr const char* RealOrComplex(const char* real, const char* complex) {
  return IsComplex<T>::value ? complex : real;
}

// The TRANS argument for (conjugate) transposition that the real and the
// complex routines accept: 'T' or 'C'.
template <typename T>
constexpr char ConjTrans() {
  return IsComplex<T>::value ? 'C' : 'T';
}

// LSAME(ca, cb): whether ca and cb are the same letter, ignoring case.
inline bool Lsame(char ca, char cb) { return ToUpper(ca) == ToUpper(cb); }

// CABS1(z) = |Re(z)| + |Im(z)|, the statement function of the complex
// routines (|z| for real z).
template <typename T>
RealT<T> Cabs1(T z) {
  return std::abs(std::real(z)) + std::abs(std::imag(z));
}

// CONJG(z) for complex z, z for real z (in branches of a merged template
// that only complex T reaches).
template <typename T>
T Conj(T z) {
  if constexpr (IsComplex<T>::value) {
    return std::conj(z);
  } else {
    return z;
  }
}

// CABS2(z) = |Re(z) / 2| + |Im(z) / 2|, the statement function of xLATRS
// and xLATBS.
template <typename T>
RealT<T> Cabs2(T z) {
  using R = RealT<T>;
  return std::abs(std::real(z) / R(2)) + std::abs(std::imag(z) / R(2));
}

// MAX(a, b, ...) and MIN(a, b, ...) as Fortran (flang) evaluates them:
// a > b ? a : b (a < b ? a : b for MIN), from the left. They differ from
// std::max and std::min only if an argument is NaN (and for zeros of
// different signs): MAX(NaN, x) is x, MAX(x, NaN) is NaN. LAPACK relies on
// this, e.g. xBDSQR's THRESH = MAX(TOL*SMINOA, ...) stays finite for a NaN
// SMINOA (with std::max, it would be NaN and xBDSQR would loop forever).
// The port uses them for every MAX and MIN, also of integers. (Not
// std::max({...}) in any case: MSVC's version never returns if an element is
// NaN.)
template <typename T, typename... Rest>
constexpr T Max(T a, T b, Rest... rest) {
  const T m = a > b ? a : b;
  if constexpr (sizeof...(rest) == 0) {
    return m;
  } else {
    return Max(m, rest...);
  }
}
template <typename T, typename... Rest>
constexpr T Min(T a, T b, Rest... rest) {
  const T m = a < b ? a : b;
  if constexpr (sizeof...(rest) == 0) {
    return m;
  } else {
    return Min(m, rest...);
  }
}

// xLAMC3(A, B) = A + B. In Fortran a function call, so that "optimizing
// compilers" cannot reassociate (x + y) + z: C++ evaluates x + y + z as
// (x + y) + z, and the build allows no reassociation.
template <typename R>
R Lamc3(R a, R b) {
  return a + b;
}

// NINT(x): x rounded to the nearest integer, halfway cases away from zero.
template <typename R>
blasint Nint(R x) {
  return static_cast<blasint>(std::lround(x));
}

// MAXLOC(X(1:n), 1) of a real vector as flang (and gfortran) evaluates it:
// the 1-based index of the first largest element, ignoring NaNs unless all
// elements are NaN (then 1); 0 if n < 1.
template <typename R>
blasint Maxloc(blasint n, const R* x) {
  if (n < 1) return 0;
  blasint loc = 1;
  for (blasint i = 2; i <= n; ++i) {
    const R prev = x[loc - 1];
    const R xi = x[i - 1];
    if (xi > prev || (std::isnan(prev) && !std::isnan(xi))) loc = i;
  }
  return loc;
}

// A routine name such as "DGETRF": the precision letter of T and `rest`.
class RoutineName {
 public:
  template <typename T>
  static RoutineName Of(const char* rest) {
    RoutineName name;
    name.str_[0] = Prefix<T>();
    size_t i = 0;
    for (; rest[i] != '\0' && i + 2 < sizeof(name.str_); ++i) {
      name.str_[i + 1] = rest[i];
    }
    name.str_[i + 1] = '\0';
    return name;
  }
  const char* c_str() const { return str_; }

 private:
  char str_[16] = {};
};

// ---------------------------------------------------------------------------
// Machine constants

// xLAMCH(cmach) for the real type R ('E', 'S', 'B', 'P', 'N', 'R', 'M',
// 'U', 'L', 'O'; case-insensitive; 0 for anything else). As in LAPACK, eps
// ('E') is the unit roundoff, half the machine epsilon.
template <typename R>
R Lamch(char cmach) {
  using Limits = std::numeric_limits<R>;
  const R one = R(1);
  const R rnd = one;
  const R eps = (one == rnd) ? Limits::epsilon() * R(0.5) : Limits::epsilon();
  switch (ToUpper(cmach)) {
    case 'E':
      return eps;
    case 'S': {
      R sfmin = Limits::min();
      const R small = one / Limits::max();
      if (small >= sfmin) {
        // Use SMALL plus a bit, to avoid the possibility of rounding
        // causing overflow when computing 1/sfmin.
        sfmin = small * (one + eps);
      }
      return sfmin;
    }
    case 'B':
      return R(Limits::radix);
    case 'P':
      return eps * R(Limits::radix);
    case 'N':
      return R(Limits::digits);
    case 'R':
      return rnd;
    case 'M':
      return R(Limits::min_exponent);
    case 'U':
      return Limits::min();
    case 'L':
      return R(Limits::max_exponent);
    case 'O':
      return Limits::max();
    default:
      return R(0);
  }
}

// The constants of LAPACK's LA_CONSTANTS module (la_constants.f90) for the
// real type R, used by the Fortran 90 routines (xLASSQ, xLARTG): Blue's
// scaling constants for sums of squares without overflow and underflow
// (tsml, tbig, ssml, sbig), and the safe minimum.
template <typename R>
struct LaConstants {
  using Limits = std::numeric_limits<R>;

  // floor(a / 2) and ceil(a / 2), also for negative a.
  static constexpr int FloorHalf(int a) {
    return a >= 0 ? a / 2 : -((1 - a) / 2);
  }
  static constexpr int CeilHalf(int a) { return -FloorHalf(-a); }

  // radix**ceiling((minexponent - 1) / 2): values below are squared with
  // scaling up (by ssml).
  static R tsml() {
    return std::ldexp(R(1), CeilHalf(Limits::min_exponent - 1));
  }
  // radix**floor((maxexponent - digits + 1) / 2): values above are squared
  // with scaling down (by sbig).
  static R tbig() {
    return std::ldexp(R(1),
                      FloorHalf(Limits::max_exponent - Limits::digits + 1));
  }
  // radix**(-floor((minexponent - digits) / 2)).
  static R ssml() {
    return std::ldexp(R(1), -FloorHalf(Limits::min_exponent - Limits::digits));
  }
  // radix**(-ceiling((maxexponent + digits - 1) / 2)).
  static R sbig() {
    return std::ldexp(R(1),
                      -CeilHalf(Limits::max_exponent + Limits::digits - 1));
  }
  // radix**max(minexponent - 1, 1 - maxexponent).
  static R safmin() {
    return std::ldexp(
        R(1), std::max(Limits::min_exponent - 1, 1 - Limits::max_exponent));
  }
};

// WORK(1) = LWORK: a workspace size returned in the first element of a
// workspace array of type T. In single precision (float and c32), the
// routines use SROUNDUP_LWORK(LWORK), which rounds up so that the float
// value is not smaller than LWORK (float has 24 mantissa bits). The ported
// routines use this for every workspace size, also where a single-precision
// routine of LAPACK 3.12.1 still returns REAL(LWORK), which can round down.
template <typename T>
T WorkSize(blasint lwork) {
  using R = RealT<T>;
  R size = static_cast<R>(lwork);
  if constexpr (std::is_same_v<R, float>) {
    if (static_cast<blasint>(size) < lwork) {
      size = size * (1.0f + std::numeric_limits<float>::epsilon());
    }
  }
  return T(size);
}

// ---------------------------------------------------------------------------
// Errors and block sizes

// CALL XERBLA(name, info): reports argument `info` (1-based) of the routine
// `name` as invalid (through hwyBLAS's error handler, see hwyblas.h).
inline void Xerbla(const char* name, blasint info) { ReportError(name, info); }
template <typename T>
void Xerbla(const char* rest, blasint info) {
  Xerbla(RoutineName::Of<T>(rest).c_str(), info);
}

// ILAENV(ispec, name, opts, n1, n2, n3, n4): `name` and `opts` are
// NUL-terminated. Calls the function installed with hwyblas_set_ilaenv, if
// any (e.g. LAPACK's test programs, which vary the block sizes).
blasint Ilaenv(blasint ispec, const char* name, const char* opts, blasint n1,
               blasint n2, blasint n3, blasint n4);
template <typename T>
blasint Ilaenv(blasint ispec, const char* rest, const char* opts, blasint n1,
               blasint n2, blasint n3, blasint n4) {
  return Ilaenv(ispec, RoutineName::Of<T>(rest).c_str(), opts, n1, n2, n3, n4);
}

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_LAPACK_COMMON_H_
