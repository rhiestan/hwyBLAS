// Matrix norms: xLASSQ (scaled sum of squares), xLANGE (general), xLANSY
// (symmetric), xLANHE (Hermitian) and xLANTR (trapezoidal).
//
// NORM is 'M' (max |a(i,j)|, not a consistent matrix norm), '1' or 'O'
// (one norm, maximum column sum), 'I' (infinity norm, maximum row sum), or
// 'F' or 'E' (Frobenius norm). The norms propagate NaN.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLASSQ: returns the values scl and smsq such that
//
//     (scl**2) * smsq = x(1)**2 + ... + x(n)**2 + (scale**2) * sumsq,
//
// (with |x(i)|**2 for complex x, i.e. the sum of the squares of the real
// and imaginary parts), where scale and sumsq are the values on entry and
// scl and smsq the values on exit. The squares are accumulated in three
// sums of the small, medium and large values, scaled with Blue's constants
// to avoid unnecessary overflow and underflow (Anderson, "Algorithm 978:
// Safe scaling in the Level 1 BLAS", 2017). NaN in scale or sumsq is
// returned unchanged.
template <typename T>
void Lassq(blasint n, const T* x, blasint incx, RealT<T>* scale,
           RealT<T>* sumsq) {
  using R = RealT<T>;
  using C = LaConstants<R>;
  const R zero = R(0), one = R(1);
  const R tsml = C::tsml(), tbig = C::tbig(), ssml = C::ssml(),
          sbig = C::sbig();
  const Vector<const T> X(x);

  // Quick return if possible.
  if (std::isnan(*scale) || std::isnan(*sumsq)) return;
  if (*sumsq == zero) *scale = one;
  if (*scale == zero) {
    *scale = one;
    *sumsq = zero;
  }
  if (n <= 0) return;

  // Compute the sum of squares in 3 accumulators:
  //    abig -- sums of squares scaled down to avoid overflow
  //    asml -- sums of squares scaled up to avoid underflow
  //    amed -- sums of squares that do not require scaling
  // The thresholds and multipliers are
  //    tbig -- values bigger than this are scaled down by sbig
  //    tsml -- values smaller than this are scaled up by ssml
  bool notbig = true;
  R asml = zero, amed = zero, abig = zero;
  blasint ix = 1;
  if (incx < 0) ix = 1 - (n - 1) * incx;
  const auto accumulate = [&](R ax) {
    if (ax > tbig) {
      abig = abig + (ax * sbig) * (ax * sbig);
      notbig = false;
    } else if (ax < tsml) {
      if (notbig) asml = asml + (ax * ssml) * (ax * ssml);
    } else {
      amed = amed + ax * ax;
    }
  };
  for (blasint i = 1; i <= n; ++i) {
    if constexpr (IsComplex<T>::value) {
      accumulate(std::abs(X(ix).real()));
      accumulate(std::abs(X(ix).imag()));
    } else {
      accumulate(std::abs(X(ix)));
    }
    ix = ix + incx;
  }

  // Put the existing sum of squares into one of the accumulators.
  if (*sumsq > zero) {
    const R ax = *scale * std::sqrt(*sumsq);
    if (ax > tbig) {
      if (*scale > one) {
        *scale = *scale * sbig;
        abig = abig + *scale * (*scale * *sumsq);
      } else {
        // sumsq > tbig^2 => (sbig * (sbig * sumsq)) is representable.
        abig = abig + *scale * (*scale * (sbig * (sbig * *sumsq)));
      }
    } else if (ax < tsml) {
      if (notbig) {
        if (*scale < one) {
          *scale = *scale * ssml;
          asml = asml + *scale * (*scale * *sumsq);
        } else {
          // sumsq < tsml^2 => (ssml * (ssml * sumsq)) is representable.
          asml = asml + *scale * (*scale * (ssml * (ssml * *sumsq)));
        }
      }
    } else {
      amed = amed + *scale * (*scale * *sumsq);
    }
  }

  // Combine abig and amed or amed and asml if more than one accumulator
  // was used.
  if (abig > zero) {
    // Combine abig and amed if abig > 0.
    if (amed > zero || std::isnan(amed)) {
      abig = abig + (amed * sbig) * sbig;
    }
    *scale = one / sbig;
    *sumsq = abig;
  } else if (asml > zero) {
    // Combine amed and asml if asml > 0.
    if (amed > zero || std::isnan(amed)) {
      amed = std::sqrt(amed);
      asml = std::sqrt(asml) / ssml;
      R ymin, ymax;
      if (asml > amed) {
        ymin = amed;
        ymax = asml;
      } else {
        ymin = asml;
        ymax = amed;
      }
      *scale = one;
      *sumsq = ymax * ymax * (one + (ymin / ymax) * (ymin / ymax));
    } else {
      *scale = one / ssml;
      *sumsq = asml;
    }
  } else {
    // Otherwise all values are mid-range or zero.
    *scale = one;
    *sumsq = amed;
  }
}

// xLANGE: the norm NORM (see the top of the file) of the M-by-N matrix A.
// WORK (M elements) is referenced only for NORM = 'I'.
template <typename T>
RealT<T> Lange(char norm, blasint m, blasint n, const T* a, blasint lda,
               RealT<T>* work) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> A(a, lda);
  const Vector<R> WORK(work);

  R value = zero;
  if (Min(m, n) == 0) {
    value = zero;
  } else if (Lsame(norm, 'M')) {
    // Find max(abs(A(i,j))).
    value = zero;
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = 1; i <= m; ++i) {
        const R temp = std::abs(A(i, j));
        if (value < temp || std::isnan(temp)) value = temp;
      }
    }
  } else if (Lsame(norm, 'O') || norm == '1') {
    // Find norm1(A).
    value = zero;
    for (blasint j = 1; j <= n; ++j) {
      R sum = zero;
      for (blasint i = 1; i <= m; ++i) sum = sum + std::abs(A(i, j));
      if (value < sum || std::isnan(sum)) value = sum;
    }
  } else if (Lsame(norm, 'I')) {
    // Find normI(A).
    for (blasint i = 1; i <= m; ++i) WORK(i) = zero;
    for (blasint j = 1; j <= n; ++j) {
      for (blasint i = 1; i <= m; ++i) {
        WORK(i) = WORK(i) + std::abs(A(i, j));
      }
    }
    value = zero;
    for (blasint i = 1; i <= m; ++i) {
      const R temp = WORK(i);
      if (value < temp || std::isnan(temp)) value = temp;
    }
  } else if (Lsame(norm, 'F') || Lsame(norm, 'E')) {
    // Find normF(A).
    R scale = zero;
    R sum = one;
    for (blasint j = 1; j <= n; ++j) Lassq(m, A.ptr(1, j), 1, &scale, &sum);
    value = scale * std::sqrt(sum);
  }
  return value;
}

// xLANSY: the norm NORM of the N-by-N symmetric matrix A (complex
// symmetric for complex T), of which the triangle UPLO is referenced. WORK
// (N elements) is referenced only for NORM = 'I', '1' or 'O'.
template <typename T>
RealT<T> Lansy(char norm, char uplo, blasint n, const T* a, blasint lda,
               RealT<T>* work) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> A(a, lda);
  const Vector<R> WORK(work);

  R value = zero;
  if (n == 0) {
    value = zero;
  } else if (Lsame(norm, 'M')) {
    // Find max(abs(A(i,j))).
    value = zero;
    if (Lsame(uplo, 'U')) {
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = 1; i <= j; ++i) {
          const R sum = std::abs(A(i, j));
          if (value < sum || std::isnan(sum)) value = sum;
        }
      }
    } else {
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = j; i <= n; ++i) {
          const R sum = std::abs(A(i, j));
          if (value < sum || std::isnan(sum)) value = sum;
        }
      }
    }
  } else if (Lsame(norm, 'I') || Lsame(norm, 'O') || norm == '1') {
    // Find normI(A) ( = norm1(A), since A is symmetric).
    value = zero;
    if (Lsame(uplo, 'U')) {
      for (blasint j = 1; j <= n; ++j) {
        R sum = zero;
        for (blasint i = 1; i <= j - 1; ++i) {
          const R absa = std::abs(A(i, j));
          sum = sum + absa;
          WORK(i) = WORK(i) + absa;
        }
        WORK(j) = sum + std::abs(A(j, j));
      }
      for (blasint i = 1; i <= n; ++i) {
        const R sum = WORK(i);
        if (value < sum || std::isnan(sum)) value = sum;
      }
    } else {
      for (blasint i = 1; i <= n; ++i) WORK(i) = zero;
      for (blasint j = 1; j <= n; ++j) {
        R sum = WORK(j) + std::abs(A(j, j));
        for (blasint i = j + 1; i <= n; ++i) {
          const R absa = std::abs(A(i, j));
          sum = sum + absa;
          WORK(i) = WORK(i) + absa;
        }
        if (value < sum || std::isnan(sum)) value = sum;
      }
    }
  } else if (Lsame(norm, 'F') || Lsame(norm, 'E')) {
    // Find normF(A).
    R scale = zero;
    R sum = one;
    if (Lsame(uplo, 'U')) {
      for (blasint j = 2; j <= n; ++j) {
        Lassq(j - 1, A.ptr(1, j), 1, &scale, &sum);
      }
    } else {
      for (blasint j = 1; j <= n - 1; ++j) {
        Lassq(n - j, A.ptr(j + 1, j), 1, &scale, &sum);
      }
    }
    sum = 2 * sum;
    Lassq(n, a, lda + 1, &scale, &sum);
    value = scale * std::sqrt(sum);
  }
  return value;
}

// CLANHE / ZLANHE: the norm NORM of the N-by-N Hermitian matrix A, of which
// the triangle UPLO is referenced (and only the real parts of its
// diagonal). WORK (N elements) is referenced only for NORM = 'I', '1' or
// 'O'.
template <typename T>
RealT<T> Lanhe(char norm, char uplo, blasint n, const T* a, blasint lda,
               RealT<T>* work) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> A(a, lda);
  const Vector<R> WORK(work);

  R value = zero;
  if (n == 0) {
    value = zero;
  } else if (Lsame(norm, 'M')) {
    // Find max(abs(A(i,j))).
    value = zero;
    if (Lsame(uplo, 'U')) {
      for (blasint j = 1; j <= n; ++j) {
        for (blasint i = 1; i <= j - 1; ++i) {
          const R sum = std::abs(A(i, j));
          if (value < sum || std::isnan(sum)) value = sum;
        }
        const R sum = std::abs(std::real(A(j, j)));
        if (value < sum || std::isnan(sum)) value = sum;
      }
    } else {
      for (blasint j = 1; j <= n; ++j) {
        const R sum = std::abs(std::real(A(j, j)));
        if (value < sum || std::isnan(sum)) value = sum;
        for (blasint i = j + 1; i <= n; ++i) {
          const R s = std::abs(A(i, j));
          if (value < s || std::isnan(s)) value = s;
        }
      }
    }
  } else if (Lsame(norm, 'I') || Lsame(norm, 'O') || norm == '1') {
    // Find normI(A) ( = norm1(A), since A is hermitian).
    value = zero;
    if (Lsame(uplo, 'U')) {
      for (blasint j = 1; j <= n; ++j) {
        R sum = zero;
        for (blasint i = 1; i <= j - 1; ++i) {
          const R absa = std::abs(A(i, j));
          sum = sum + absa;
          WORK(i) = WORK(i) + absa;
        }
        WORK(j) = sum + std::abs(std::real(A(j, j)));
      }
      for (blasint i = 1; i <= n; ++i) {
        const R sum = WORK(i);
        if (value < sum || std::isnan(sum)) value = sum;
      }
    } else {
      for (blasint i = 1; i <= n; ++i) WORK(i) = zero;
      for (blasint j = 1; j <= n; ++j) {
        R sum = WORK(j) + std::abs(std::real(A(j, j)));
        for (blasint i = j + 1; i <= n; ++i) {
          const R absa = std::abs(A(i, j));
          sum = sum + absa;
          WORK(i) = WORK(i) + absa;
        }
        if (value < sum || std::isnan(sum)) value = sum;
      }
    }
  } else if (Lsame(norm, 'F') || Lsame(norm, 'E')) {
    // Find normF(A).
    R scale = zero;
    R sum = one;
    if (Lsame(uplo, 'U')) {
      for (blasint j = 2; j <= n; ++j) {
        Lassq(j - 1, A.ptr(1, j), 1, &scale, &sum);
      }
    } else {
      for (blasint j = 1; j <= n - 1; ++j) {
        Lassq(n - j, A.ptr(j + 1, j), 1, &scale, &sum);
      }
    }
    sum = 2 * sum;
    for (blasint i = 1; i <= n; ++i) {
      if (std::real(A(i, i)) != zero) {
        const R absa = std::abs(std::real(A(i, i)));
        if (scale < absa) {
          sum = one + sum * ((scale / absa) * (scale / absa));
          scale = absa;
        } else {
          sum = sum + (absa / scale) * (absa / scale);
        }
      }
    }
    value = scale * std::sqrt(sum);
  }
  return value;
}

// xLANTR: the norm NORM of the M-by-N trapezoidal (triangular if M = N)
// matrix A: its upper trapezoid (UPLO = 'U', A(i, j) = 0 for i > j) or lower
// trapezoid (UPLO = 'L', A(i, j) = 0 for i < j), with a unit diagonal for
// DIAG = 'U' (which is not referenced). WORK (M elements) is referenced only
// for NORM = 'I'.
template <typename T>
RealT<T> Lantr(char norm, char uplo, char diag, blasint m, blasint n,
               const T* a, blasint lda, RealT<T>* work) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Matrix<const T> A(a, lda);
  const Vector<R> WORK(work);

  R value = zero;
  if (Min(m, n) == 0) {
    value = zero;
  } else if (Lsame(norm, 'M')) {
    // Find max(abs(A(i,j))).
    if (Lsame(diag, 'U')) {
      value = one;
      if (Lsame(uplo, 'U')) {
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = 1; i <= Min(m, j - 1); ++i) {
            const R sum = std::abs(A(i, j));
            if (value < sum || std::isnan(sum)) value = sum;
          }
        }
      } else {
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = j + 1; i <= m; ++i) {
            const R sum = std::abs(A(i, j));
            if (value < sum || std::isnan(sum)) value = sum;
          }
        }
      }
    } else {
      value = zero;
      if (Lsame(uplo, 'U')) {
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = 1; i <= Min(m, j); ++i) {
            const R sum = std::abs(A(i, j));
            if (value < sum || std::isnan(sum)) value = sum;
          }
        }
      } else {
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = j; i <= m; ++i) {
            const R sum = std::abs(A(i, j));
            if (value < sum || std::isnan(sum)) value = sum;
          }
        }
      }
    }
  } else if (Lsame(norm, 'O') || norm == '1') {
    // Find norm1(A).
    value = zero;
    const bool udiag = Lsame(diag, 'U');
    if (Lsame(uplo, 'U')) {
      for (blasint j = 1; j <= n; ++j) {
        R sum;
        if (udiag && j <= m) {
          sum = one;
          for (blasint i = 1; i <= j - 1; ++i) sum = sum + std::abs(A(i, j));
        } else {
          sum = zero;
          for (blasint i = 1; i <= Min(m, j); ++i) {
            sum = sum + std::abs(A(i, j));
          }
        }
        if (value < sum || std::isnan(sum)) value = sum;
      }
    } else {
      for (blasint j = 1; j <= n; ++j) {
        R sum;
        if (udiag) {
          sum = one;
          for (blasint i = j + 1; i <= m; ++i) sum = sum + std::abs(A(i, j));
        } else {
          sum = zero;
          for (blasint i = j; i <= m; ++i) sum = sum + std::abs(A(i, j));
        }
        if (value < sum || std::isnan(sum)) value = sum;
      }
    }
  } else if (Lsame(norm, 'I')) {
    // Find normI(A).
    if (Lsame(uplo, 'U')) {
      if (Lsame(diag, 'U')) {
        for (blasint i = 1; i <= m; ++i) WORK(i) = one;
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = 1; i <= Min(m, j - 1); ++i) {
            WORK(i) = WORK(i) + std::abs(A(i, j));
          }
        }
      } else {
        for (blasint i = 1; i <= m; ++i) WORK(i) = zero;
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = 1; i <= Min(m, j); ++i) {
            WORK(i) = WORK(i) + std::abs(A(i, j));
          }
        }
      }
    } else {
      if (Lsame(diag, 'U')) {
        for (blasint i = 1; i <= Min(m, n); ++i) WORK(i) = one;
        for (blasint i = n + 1; i <= m; ++i) WORK(i) = zero;
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = j + 1; i <= m; ++i) {
            WORK(i) = WORK(i) + std::abs(A(i, j));
          }
        }
      } else {
        for (blasint i = 1; i <= m; ++i) WORK(i) = zero;
        for (blasint j = 1; j <= n; ++j) {
          for (blasint i = j; i <= m; ++i) {
            WORK(i) = WORK(i) + std::abs(A(i, j));
          }
        }
      }
    }
    value = zero;
    for (blasint i = 1; i <= m; ++i) {
      const R sum = WORK(i);
      if (value < sum || std::isnan(sum)) value = sum;
    }
  } else if (Lsame(norm, 'F') || Lsame(norm, 'E')) {
    // Find normF(A).
    R scale, sum;
    if (Lsame(uplo, 'U')) {
      if (Lsame(diag, 'U')) {
        scale = one;
        sum = R(Min(m, n));
        for (blasint j = 2; j <= n; ++j) {
          Lassq(Min(m, j - 1), A.ptr(1, j), 1, &scale, &sum);
        }
      } else {
        scale = zero;
        sum = one;
        for (blasint j = 1; j <= n; ++j) {
          Lassq(Min(m, j), A.ptr(1, j), 1, &scale, &sum);
        }
      }
    } else {
      if (Lsame(diag, 'U')) {
        scale = one;
        sum = R(Min(m, n));
        for (blasint j = 1; j <= n; ++j) {
          Lassq(m - j, A.ptr(Min(m, j + 1), j), 1, &scale, &sum);
        }
      } else {
        scale = zero;
        sum = one;
        for (blasint j = 1; j <= n; ++j) {
          Lassq(m - j + 1, A.ptr(j, j), 1, &scale, &sum);
        }
      }
    }
    value = scale * std::sqrt(sum);
  }
  return value;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template RealT<T> Lantr<T>(char, char, char, blasint, blasint, const T*,  \
                             blasint, RealT<T>*);                           \
  template void Lassq<T>(blasint, const T*, blasint, RealT<T>*, RealT<T>*); \
  template RealT<T> Lange<T>(char, blasint, blasint, const T*, blasint,     \
                             RealT<T>*);                                    \
  template RealT<T> Lansy<T>(char, char, blasint, const T*, blasint, RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

template float Lanhe<c32>(char, char, blasint, const c32*, blasint, float*);
template double Lanhe<c64>(char, char, blasint, const c64*, blasint, double*);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_NORMS(prefix, T, R)                                     \
  HWYBLAS_API void prefix##lassq_(const hwyblas_int* n, const R* x,            \
                                  const hwyblas_int* incx, R* scale,           \
                                  R* sumsq) {                                  \
    hwyblas::lapack::Lassq(*n, As<const T>(x), *incx, scale, sumsq);           \
  }                                                                            \
  HWYBLAS_API R prefix##lange_(const char* norm, const hwyblas_int* m,         \
                               const hwyblas_int* n, const R* a,               \
                               const hwyblas_int* lda, R* work) {              \
    return hwyblas::lapack::Lange(*norm, *m, *n, As<const T>(a), *lda, work);  \
  }                                                                            \
  HWYBLAS_API R prefix##lansy_(const char* norm, const char* uplo,             \
                               const hwyblas_int* n, const R* a,               \
                               const hwyblas_int* lda, R* work) {              \
    return hwyblas::lapack::Lansy(*norm, *uplo, *n, As<const T>(a), *lda,      \
                                  work);                                       \
  }                                                                            \
  HWYBLAS_API R prefix##lantr_(const char* norm, const char* uplo,             \
                               const char* diag, const hwyblas_int* m,         \
                               const hwyblas_int* n, const R* a,               \
                               const hwyblas_int* lda, R* work) {              \
    return hwyblas::lapack::Lantr(*norm, *uplo, *diag, *m, *n, As<const T>(a), \
                                  *lda, work);                                 \
  }

HWYBLAS_LAPACK_NORMS(s, float, float)
HWYBLAS_LAPACK_NORMS(d, double, double)
HWYBLAS_LAPACK_NORMS(c, c32, float)
HWYBLAS_LAPACK_NORMS(z, c64, double)

#undef HWYBLAS_LAPACK_NORMS

HWYBLAS_API float clanhe_(const char* norm, const char* uplo,
                          const hwyblas_int* n, const float* a,
                          const hwyblas_int* lda, float* work) {
  return hwyblas::lapack::Lanhe(*norm, *uplo, *n, As<const c32>(a), *lda, work);
}
HWYBLAS_API double zlanhe_(const char* norm, const char* uplo,
                           const hwyblas_int* n, const double* a,
                           const hwyblas_int* lda, double* work) {
  return hwyblas::lapack::Lanhe(*norm, *uplo, *n, As<const c64>(a), *lda, work);
}

}  // extern "C"
