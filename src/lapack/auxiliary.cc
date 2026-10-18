// LAPACK's auxiliary routines for vectors and matrices: xLACGV, xLACPY,
// xLACP2, xLACRM, xLARCM, xLASET, xLASCL, xLAPY2, xLAPY3, xLADIV, ILAxLC
// and ILAxLR.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
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

// xLACP2: copies all (UPLO other than 'U' or 'L') or part of the real M-by-N
// matrix A to the complex matrix B, like xLACPY. Complex T only.
template <typename T>
void Lacp2(char uplo, blasint m, blasint n, const RealT<T>* a, blasint lda,
           T* b, blasint ldb) {
  const Matrix<const RealT<T>> A(a, lda);
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

// xLACRM: C := A * B for the complex M-by-N matrix A and the real N-by-N
// matrix B, as two real products of the real and imaginary parts of A.
// RWORK has 2*M*N elements. Complex T only.
template <typename T>
void Lacrm(blasint m, blasint n, const T* a, blasint lda, const RealT<T>* b,
           blasint ldb, T* c, blasint ldc, RealT<T>* rwork) {
  using R = RealT<T>;
  const R one = R(1);
  const R zero = R(0);
  const Matrix<const T> A(a, lda);
  const Matrix<T> C(c, ldc);
  const Vector<R> RWORK(rwork);
  if ((m == 0) || (n == 0)) return;
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      RWORK((j - 1) * m + i) = std::real(A(i, j));
    }
  }
  const blasint l = m * n + 1;
  Gemm<R>('N', 'N', m, n, n, one, rwork, m, b, ldb, zero, RWORK.ptr(l), m);
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) C(i, j) = RWORK(l + (j - 1) * m + i - 1);
  }
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      RWORK((j - 1) * m + i) = std::imag(A(i, j));
    }
  }
  Gemm<R>('N', 'N', m, n, n, one, rwork, m, b, ldb, zero, RWORK.ptr(l), m);
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      C(i, j) = T(std::real(C(i, j)), RWORK(l + (j - 1) * m + i - 1));
    }
  }
}

// xLARCM: C := A * B for the real M-by-M matrix A and the complex M-by-N
// matrix B, as two real products with the real and imaginary parts of B.
// RWORK has 2*M*N elements. Complex T only.
template <typename T>
void Larcm(blasint m, blasint n, const RealT<T>* a, blasint lda, const T* b,
           blasint ldb, T* c, blasint ldc, RealT<T>* rwork) {
  using R = RealT<T>;
  const R one = R(1);
  const R zero = R(0);
  const Matrix<const T> B(b, ldb);
  const Matrix<T> C(c, ldc);
  const Vector<R> RWORK(rwork);
  if ((m == 0) || (n == 0)) return;
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      RWORK((j - 1) * m + i) = std::real(B(i, j));
    }
  }
  const blasint l = m * n + 1;
  Gemm<R>('N', 'N', m, n, m, one, a, lda, rwork, m, zero, RWORK.ptr(l), m);
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) C(i, j) = RWORK(l + (j - 1) * m + i - 1);
  }
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      RWORK((j - 1) * m + i) = std::imag(B(i, j));
    }
  }
  Gemm<R>('N', 'N', m, n, m, one, a, lda, rwork, m, zero, RWORK.ptr(l), m);
  for (blasint j = 1; j <= n; ++j) {
    for (blasint i = 1; i <= m; ++i) {
      C(i, j) = T(std::real(C(i, j)), RWORK(l + (j - 1) * m + i - 1));
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

// xLASET: sets the off-diagonal elements of the M-by-N matrix A to ALPHA
// and its diagonal to BETA: of the strictly upper triangle (UPLO = 'U'),
// the strictly lower triangle (UPLO = 'L') or all of A (otherwise).
template <typename T>
void Laset(char uplo, blasint m, blasint n, T alpha, T beta, T* a,
           blasint lda) {
  const Matrix<T> A(a, lda);
  if (Lsame(uplo, 'U')) {
    // Set the strictly upper triangular or trapezoidal part of the array
    // to ALPHA.
    for (blasint j = 2; j <= n; ++j) {
      for (blasint i = 1; i <= std::min(j - 1, m); ++i) A(i, j) = alpha;
    }
  } else if (Lsame(uplo, 'L')) {
    // Set the strictly lower triangular or trapezoidal part of the array
    // to ALPHA.
    for (blasint j = 1; j <= std::min(m, n); ++j) {
      for (blasint i = j + 1; i <= m; ++i) A(i, j) = alpha;
    }
  } else {
    // Set the leading m-by-n submatrix to ALPHA.
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = 1; i <= m; ++i) A(i, j) = alpha;
    }
  }
  // Set the first min(M,N) diagonal elements to BETA.
  for (blasint i = 1; i <= std::min(m, n); ++i) A(i, i) = beta;
}

// xLASCL: multiplies the M-by-N matrix A by the real scalar CTO/CFROM,
// without over- or underflow if the final result CTO*A(I,J)/CFROM can be
// represented: in steps of factors that are safe (powers of the safe
// minimum, then the rest). TYPE selects the storage of A:
//   'G' full matrix, 'L' lower triangular, 'U' upper triangular,
//   'H' upper Hessenberg,
//   'B' lower half of a symmetric band matrix (lower bandwidth KL),
//   'Q' upper half of a symmetric band matrix (upper bandwidth KU),
//   'Z' band matrix with bandwidths KL, KU, in the storage of xGBTRF.
// Returns INFO (0, or -i for an illegal argument i).
template <typename T>
blasint Lascl(char type, blasint kl, blasint ku, RealT<T> cfrom, RealT<T> cto,
              blasint m, blasint n, T* a, blasint lda) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const Matrix<T> A(a, lda);

  // Test the input arguments.
  blasint info = 0;
  blasint itype;
  if (Lsame(type, 'G')) {
    itype = 0;
  } else if (Lsame(type, 'L')) {
    itype = 1;
  } else if (Lsame(type, 'U')) {
    itype = 2;
  } else if (Lsame(type, 'H')) {
    itype = 3;
  } else if (Lsame(type, 'B')) {
    itype = 4;
  } else if (Lsame(type, 'Q')) {
    itype = 5;
  } else if (Lsame(type, 'Z')) {
    itype = 6;
  } else {
    itype = -1;
  }

  if (itype == -1) {
    info = -1;
  } else if (cfrom == zero || std::isnan(cfrom)) {
    info = -4;
  } else if (std::isnan(cto)) {
    info = -5;
  } else if (m < 0) {
    info = -6;
  } else if (n < 0 || (itype == 4 && n != m) || (itype == 5 && n != m)) {
    info = -7;
  } else if (itype <= 3 && lda < std::max<blasint>(1, m)) {
    info = -9;
  } else if (itype >= 4) {
    if (kl < 0 || kl > std::max<blasint>(m - 1, 0)) {
      info = -2;
    } else if (ku < 0 || ku > std::max<blasint>(n - 1, 0) ||
               ((itype == 4 || itype == 5) && kl != ku)) {
      info = -3;
    } else if ((itype == 4 && lda < kl + 1) || (itype == 5 && lda < ku + 1) ||
               (itype == 6 && lda < 2 * kl + ku + 1)) {
      info = -9;
    }
  }

  if (info != 0) {
    Xerbla<T>("LASCL", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || m == 0) return info;

  // Get machine parameters.
  const R smlnum = Lamch<R>('S');
  const R bignum = one / smlnum;

  R cfromc = cfrom;
  R ctoc = cto;

  bool done = false;
  while (!done) {
    const R cfrom1 = cfromc * smlnum;
    R mul;
    if (cfrom1 == cfromc) {
      // CFROMC is an inf. Multiply by a correctly signed zero for finite
      // CTOC, or a NaN if CTOC is infinite.
      mul = ctoc / cfromc;
      done = true;
    } else {
      const R cto1 = ctoc / bignum;
      if (cto1 == ctoc) {
        // CTOC is either 0 or an inf. In both cases, CTOC itself serves as
        // the correct multiplication factor.
        mul = ctoc;
        done = true;
        cfromc = one;
      } else if (std::abs(cfrom1) > std::abs(ctoc) && ctoc != zero) {
        mul = smlnum;
        done = false;
        cfromc = cfrom1;
      } else if (std::abs(cto1) > std::abs(cfromc)) {
        mul = bignum;
        done = false;
        ctoc = cto1;
      } else {
        mul = ctoc / cfromc;
        done = true;
        if (mul == one) return info;
      }
    }

    if (itype == 0) {
      // Full matrix.
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = 1; i <= m; ++i) A(i, j) = A(i, j) * mul;
      }
    } else if (itype == 1) {
      // Lower triangular matrix.
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = j; i <= m; ++i) A(i, j) = A(i, j) * mul;
      }
    } else if (itype == 2) {
      // Upper triangular matrix.
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = 1; i <= std::min(j, m); ++i) A(i, j) = A(i, j) * mul;
      }
    } else if (itype == 3) {
      // Upper Hessenberg matrix.
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = 1; i <= std::min(j + 1, m); ++i) {
          A(i, j) = A(i, j) * mul;
        }
      }
    } else if (itype == 4) {
      // Lower half of a symmetric band matrix.
      const blasint k3 = kl + 1;
      const blasint k4 = n + 1;
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = 1; i <= std::min(k3, k4 - j); ++i) {
          A(i, j) = A(i, j) * mul;
        }
      }
    } else if (itype == 5) {
      // Upper half of a symmetric band matrix.
      const blasint k1 = ku + 2;
      const blasint k3 = ku + 1;
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = std::max<blasint>(k1 - j, 1); i <= k3; ++i) {
          A(i, j) = A(i, j) * mul;
        }
      }
    } else if (itype == 6) {
      // Band matrix.
      const blasint k1 = kl + ku + 2;
      const blasint k2 = kl + 1;
      const blasint k3 = 2 * kl + ku + 1;
      const blasint k4 = kl + ku + 1 + m;
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = std::max(k1 - j, k2); i <= std::min(k3, k4 - j); ++i) {
          A(i, j) = A(i, j) * mul;
        }
      }
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                   \
  template void Lacgv<T>(blasint, T*, blasint);                         \
  template void Lacpy<T>(char, blasint, blasint, const T*, blasint, T*, \
                         blasint);                                      \
  template blasint Ilalc<T>(blasint, blasint, const T*, blasint);       \
  template blasint Ilalr<T>(blasint, blasint, const T*, blasint);       \
  template void Laset<T>(char, blasint, blasint, T, T, T*, blasint);    \
  template blasint Lascl<T>(char, blasint, blasint, RealT<T>, RealT<T>, \
                            blasint, blasint, T*, blasint);
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

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template void Lacp2<T>(char, blasint, blasint, const RealT<T>*, blasint, T*, \
                         blasint);                                             \
  template void Lacrm<T>(blasint, blasint, const T*, blasint, const RealT<T>*, \
                         blasint, T*, blasint, RealT<T>*);                     \
  template void Larcm<T>(blasint, blasint, const RealT<T>*, blasint, const T*, \
                         blasint, T*, blasint, RealT<T>*);
HWYBLAS_LAPACK_INSTANTIATE(c32)
HWYBLAS_LAPACK_INSTANTIATE(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE

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

#define HWYBLAS_LAPACK_AUXILIARY(prefix, T, R)                                 \
  HWYBLAS_API void prefix##laset_(                                             \
      const char* uplo, const hwyblas_int* m, const hwyblas_int* n,            \
      const R* alpha, const R* beta, R* a, const hwyblas_int* lda) {           \
    hwyblas::lapack::Laset(*uplo, *m, *n, *As<const T>(alpha),                 \
                           *As<const T>(beta), As<T>(a), *lda);                \
  }                                                                            \
  HWYBLAS_API void prefix##lascl_(                                             \
      const char* type, const hwyblas_int* kl, const hwyblas_int* ku,          \
      const R* cfrom, const R* cto, const hwyblas_int* m,                      \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, hwyblas_int* info) { \
    *info = hwyblas::lapack::Lascl(*type, *kl, *ku, *cfrom, *cto, *m, *n,      \
                                   As<T>(a), *lda);                            \
  }                                                                            \
  HWYBLAS_API void prefix##lacpy_(                                             \
      const char* uplo, const hwyblas_int* m, const hwyblas_int* n,            \
      const R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb) {      \
    hwyblas::lapack::Lacpy(*uplo, *m, *n, As<const T>(a), *lda, As<T>(b),      \
                           *ldb);                                              \
  }                                                                            \
  HWYBLAS_API hwyblas_int ila##prefix##lc_(const hwyblas_int* m,               \
                                           const hwyblas_int* n, const R* a,   \
                                           const hwyblas_int* lda) {           \
    return hwyblas::lapack::Ilalc(*m, *n, As<const T>(a), *lda);               \
  }                                                                            \
  HWYBLAS_API hwyblas_int ila##prefix##lr_(const hwyblas_int* m,               \
                                           const hwyblas_int* n, const R* a,   \
                                           const hwyblas_int* lda) {           \
    return hwyblas::lapack::Ilalr(*m, *n, As<const T>(a), *lda);               \
  }

HWYBLAS_LAPACK_AUXILIARY(s, float, float)
HWYBLAS_LAPACK_AUXILIARY(d, double, double)
HWYBLAS_LAPACK_AUXILIARY(c, c32, float)
HWYBLAS_LAPACK_AUXILIARY(z, c64, double)

#undef HWYBLAS_LAPACK_AUXILIARY

#define HWYBLAS_LAPACK_AUXILIARY_COMPLEX(prefix, T, R)                        \
  HWYBLAS_API void prefix##lacp2_(                                            \
      const char* uplo, const hwyblas_int* m, const hwyblas_int* n,           \
      const R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb) {     \
    hwyblas::lapack::Lacp2(*uplo, *m, *n, a, *lda, As<T>(b), *ldb);           \
  }                                                                           \
  HWYBLAS_API void prefix##lacrm_(const hwyblas_int* m, const hwyblas_int* n, \
                                  const R* a, const hwyblas_int* lda,         \
                                  const R* b, const hwyblas_int* ldb, R* c,   \
                                  const hwyblas_int* ldc, R* rwork) {         \
    hwyblas::lapack::Lacrm(*m, *n, As<const T>(a), *lda, b, *ldb, As<T>(c),   \
                           *ldc, rwork);                                      \
  }                                                                           \
  HWYBLAS_API void prefix##larcm_(const hwyblas_int* m, const hwyblas_int* n, \
                                  const R* a, const hwyblas_int* lda,         \
                                  const R* b, const hwyblas_int* ldb, R* c,   \
                                  const hwyblas_int* ldc, R* rwork) {         \
    hwyblas::lapack::Larcm(*m, *n, a, *lda, As<const T>(b), *ldb, As<T>(c),   \
                           *ldc, rwork);                                      \
  }

HWYBLAS_LAPACK_AUXILIARY_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_AUXILIARY_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_AUXILIARY_COMPLEX

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
