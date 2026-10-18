// Eigenvalues (and eigenvectors) of real symmetric tridiagonal matrices:
// xSTEQR (implicit QL or QR method, with eigenvectors) and xSTERF
// (root-free variant, eigenvalues only), with their auxiliary routines
// xLAE2, xLAEV2 (2-by-2 symmetric eigenproblems), xLARTG (plane rotations,
// also for complex T), xLASR (applying sequences of plane rotations), xLANST
// (norms of tridiagonal matrices) and xLASRT (sorting).
//
// xSTEQR and xSTERF keep the control flow of the Fortran routines, which is
// built from GO TO statements: the labels have the Fortran statement
// numbers (L10 is statement 10).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>
#include <utility>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLAE2: computes the eigenvalues of the 2-by-2 symmetric matrix
//    [  A   B  ]
//    [  B   C  ]
// RT1 is the eigenvalue of larger absolute value, and RT2 the eigenvalue
// of smaller absolute value. RT1 is accurate to a few ulps barring
// over/underflow; RT2 may be inaccurate if there is massive cancellation
// in the determinant A*C-B*B.
template <typename R>
void Lae2(R a, R b, R c, R* rt1, R* rt2) {
  const R one = R(1), two = R(2), zero = R(0), half = R(0.5);

  // Compute the eigenvalues.
  const R sm = a + c;
  const R df = a - c;
  const R adf = std::abs(df);
  const R tb = b + b;
  const R ab = std::abs(tb);
  R acmx, acmn;
  if (std::abs(a) > std::abs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }
  R rt;
  if (adf > ab) {
    rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
  } else if (adf < ab) {
    rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
  } else {
    // Includes case AB=ADF=0.
    rt = ab * std::sqrt(two);
  }
  if (sm < zero) {
    *rt1 = half * (sm - rt);
    // Order of execution important. To get fully accurate smaller
    // eigenvalue, next line needs to be executed in higher precision.
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else if (sm > zero) {
    *rt1 = half * (sm + rt);
    // Order of execution important. To get fully accurate smaller
    // eigenvalue, next line needs to be executed in higher precision.
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else {
    // Includes case RT1 = RT2 = 0.
    *rt1 = half * rt;
    *rt2 = -half * rt;
  }
}

// xLAEV2: computes the eigendecomposition of the 2-by-2 symmetric matrix
//    [  A   B  ]
//    [  B   C  ]
// like xLAE2 (RT1, RT2), and (CS1, SN1), the unit right eigenvector for
// RT1:
//    [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
//    [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
template <typename R>
void Laev2(R a, R b, R c, R* rt1, R* rt2, R* cs1, R* sn1) {
  const R one = R(1), two = R(2), zero = R(0), half = R(0.5);

  // Compute the eigenvalues.
  const R sm = a + c;
  const R df = a - c;
  const R adf = std::abs(df);
  const R tb = b + b;
  const R ab = std::abs(tb);
  R acmx, acmn;
  if (std::abs(a) > std::abs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }
  R rt;
  if (adf > ab) {
    rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
  } else if (adf < ab) {
    rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
  } else {
    // Includes case AB=ADF=0.
    rt = ab * std::sqrt(two);
  }
  int sgn1;
  if (sm < zero) {
    *rt1 = half * (sm - rt);
    sgn1 = -1;
    // Order of execution important. To get fully accurate smaller
    // eigenvalue, next line needs to be executed in higher precision.
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else if (sm > zero) {
    *rt1 = half * (sm + rt);
    sgn1 = 1;
    // Order of execution important. To get fully accurate smaller
    // eigenvalue, next line needs to be executed in higher precision.
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else {
    // Includes case RT1 = RT2 = 0.
    *rt1 = half * rt;
    *rt2 = -half * rt;
    sgn1 = 1;
  }

  // Compute the eigenvector.
  R cs;
  int sgn2;
  if (df >= zero) {
    cs = df + rt;
    sgn2 = 1;
  } else {
    cs = df - rt;
    sgn2 = -1;
  }
  const R acs = std::abs(cs);
  if (acs > ab) {
    const R ct = -tb / cs;
    *sn1 = one / std::sqrt(one + ct * ct);
    *cs1 = ct * *sn1;
  } else {
    if (ab == zero) {
      *cs1 = one;
      *sn1 = zero;
    } else {
      const R tn = -cs / tb;
      *cs1 = one / std::sqrt(one + tn * tn);
      *sn1 = tn * *cs1;
    }
  }
  if (sgn1 == sgn2) {
    const R tn = *cs1;
    *cs1 = -*sn1;
    *sn1 = tn;
  }
}

// xLARTG (the Fortran 90 version of LAPACK 3.10 and later): generates a
// plane rotation so that
//    [  C  S  ]  .  [ F ]  =  [ R ]
//    [ -S  C  ]     [ G ]     [ 0 ]
// with C**2 + S**2 = 1, and the sign of R that of F (C >= 0), without
// unnecessary overflow and underflow (Anderson, "Algorithm 978", 2017).
template <typename R>
void Lartg(R f, R g, R* c, R* s, R* r) {
  const R zero = R(0), one = R(1);
  const R safmin = LaConstants<R>::safmin();
  const R safmax = one / safmin;
  const R rtmin = std::sqrt(safmin);
  const R rtmax = std::sqrt(safmax / 2);

  const R f1 = std::abs(f);
  const R g1 = std::abs(g);
  if (g == zero) {
    *c = one;
    *s = zero;
    *r = f;
  } else if (f == zero) {
    *c = zero;
    *s = std::copysign(one, g);
    *r = g1;
  } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
    const R d = std::sqrt(f * f + g * g);
    *c = f1 / d;
    *r = std::copysign(d, f);
    *s = g / *r;
  } else {
    const R u = std::min(safmax, std::max(std::max(safmin, f1), g1));
    const R fs = f / u;
    const R gs = g / u;
    const R d = std::sqrt(fs * fs + gs * gs);
    *c = std::abs(fs) / d;
    *r = std::copysign(d, f);
    *s = gs / *r;
    *r = *r * u;
  }
}

// xLARTG for complex F and G (CLARTG / ZLARTG, Fortran 90): generates a
// plane rotation with real C and complex S so that
//    [  C         S  ]  .  [ F ]  =  [ R ]
//    [ -conjg(S)  C  ]     [ G ]     [ 0 ]
// with C**2 + |S|**2 = 1 and C >= 0, without unnecessary overflow and
// underflow (Anderson, "Algorithm 978", 2017). The comments in the code are
// the reference's.
template <typename T>
std::enable_if_t<IsComplex<T>::value> Lartg(T f, T g, RealT<T>* c, T* s, T* r) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const T czero = T(0);
  const R safmin = LaConstants<R>::safmin();
  const R safmax = one / safmin;
  // ABSSQ(t) = real(t)**2 + aimag(t)**2.
  const auto abssq = [](T t) {
    return std::real(t) * std::real(t) + std::imag(t) * std::imag(t);
  };
  const R rtmin = std::sqrt(safmin);

  if (g == czero) {
    *c = one;
    *s = czero;
    *r = f;
  } else if (f == czero) {
    *c = zero;
    if (std::real(g) == zero) {
      *r = std::abs(std::imag(g));
      *s = std::conj(g) / *r;
    } else if (std::imag(g) == zero) {
      *r = std::abs(std::real(g));
      *s = std::conj(g) / *r;
    } else {
      const R g1 = std::max(std::abs(std::real(g)), std::abs(std::imag(g)));
      const R rtmax = std::sqrt(safmax / 2);
      if (g1 > rtmin && g1 < rtmax) {
        // Use unscaled algorithm.
        // The following two lines can be replaced by `d = abs( g )`.
        // This algorithm do not use the intrinsic complex abs.
        const R g2 = abssq(g);
        const R d = std::sqrt(g2);
        *s = std::conj(g) / d;
        *r = d;
      } else {
        // Use scaled algorithm.
        const R u = std::min(safmax, std::max(safmin, g1));
        const T gs = g / u;
        // The following two lines can be replaced by `d = abs( gs )`.
        // This algorithm do not use the intrinsic complex abs.
        const R g2 = abssq(gs);
        const R d = std::sqrt(g2);
        *s = std::conj(gs) / d;
        *r = d * u;
      }
    }
  } else {
    const R f1 = std::max(std::abs(std::real(f)), std::abs(std::imag(f)));
    const R g1 = std::max(std::abs(std::real(g)), std::abs(std::imag(g)));
    R rtmax = std::sqrt(safmax / 4);
    if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
      // Use unscaled algorithm.
      const R f2 = abssq(f);
      const R g2 = abssq(g);
      const R h2 = f2 + g2;
      // safmin <= f2 <= h2 <= safmax
      if (f2 >= h2 * safmin) {
        // safmin <= f2/h2 <= 1, and h2/f2 is finite
        *c = std::sqrt(f2 / h2);
        *r = f / *c;
        rtmax = rtmax * 2;
        if (f2 > rtmin && h2 < rtmax) {
          // safmin <= sqrt( f2*h2 ) <= safmax
          *s = std::conj(g) * (f / std::sqrt(f2 * h2));
        } else {
          *s = std::conj(g) * (*r / h2);
        }
      } else {
        // f2/h2 <= safmin may be subnormal, and h2/f2 may overflow.
        // Moreover,
        //  safmin <= f2*f2 * safmax < f2 * h2 < h2*h2 * safmin <= safmax,
        //  sqrt(safmin) <= sqrt(f2 * h2) <= sqrt(safmax).
        // Also,
        //  g2 >> f2, which means that h2 = g2.
        const R d = std::sqrt(f2 * h2);
        *c = f2 / d;
        if (*c >= safmin) {
          *r = f / *c;
        } else {
          // f2 / sqrt(f2 * h2) < safmin, then
          //  sqrt(safmin) <= f2 * sqrt(safmax) <= h2 / sqrt(f2 * h2) <=
          //  h2 * (safmin / f2) <= h2 <= safmax
          *r = f * (h2 / d);
        }
        *s = std::conj(g) * (f / d);
      }
    } else {
      // Use scaled algorithm.
      const R u = std::min({safmax, std::max({safmin, f1, g1})});
      const T gs = g / u;
      const R g2 = abssq(gs);
      R w, f2, h2;
      T fs;
      if (f1 / u < rtmin) {
        // f is not well-scaled when scaled by g1.
        // Use a different scaling for f.
        const R v = std::min(safmax, std::max(safmin, f1));
        w = v / u;
        fs = f / v;
        f2 = abssq(fs);
        h2 = f2 * (w * w) + g2;
      } else {
        // Otherwise use the same scaling for f and g.
        w = one;
        fs = f / u;
        f2 = abssq(fs);
        h2 = f2 + g2;
      }
      // safmin <= f2 <= h2 <= safmax
      if (f2 >= h2 * safmin) {
        // safmin <= f2/h2 <= 1, and h2/f2 is finite
        *c = std::sqrt(f2 / h2);
        *r = fs / *c;
        rtmax = rtmax * 2;
        if (f2 > rtmin && h2 < rtmax) {
          // safmin <= sqrt( f2*h2 ) <= safmax
          *s = std::conj(gs) * (fs / std::sqrt(f2 * h2));
        } else {
          *s = std::conj(gs) * (*r / h2);
        }
      } else {
        // f2/h2 <= safmin may be subnormal, and h2/f2 may overflow.
        // Moreover,
        //  safmin <= f2*f2 * safmax < f2 * h2 < h2*h2 * safmin <= safmax,
        //  sqrt(safmin) <= sqrt(f2 * h2) <= sqrt(safmax).
        // Also,
        //  g2 >> f2, which means that h2 = g2.
        const R d = std::sqrt(f2 * h2);
        *c = f2 / d;
        if (*c >= safmin) {
          *r = fs / *c;
        } else {
          // f2 / sqrt(f2 * h2) < safmin, then
          //  sqrt(safmin) <= f2 * sqrt(safmax) <= h2 / sqrt(f2 * h2) <=
          //  h2 * (safmin / f2) <= h2 <= safmax
          *r = fs * (h2 / d);
        }
        *s = std::conj(gs) * (fs / d);
      }
      // Rescale c and r.
      *c = *c * w;
      *r = *r * u;
    }
  }
}

// xLANST: the norm NORM ('M', '1' or 'O', 'I', 'F' or 'E'; see norms.cc)
// of the N-by-N symmetric tridiagonal matrix with diagonal D and
// off-diagonal E.
template <typename R>
R Lanst(char norm, blasint n, const R* d, const R* e) {
  const R one = R(1), zero = R(0);
  const Vector<const R> D(d);
  const Vector<const R> E(e);

  R anorm = zero;
  if (n <= 0) {
    anorm = zero;
  } else if (Lsame(norm, 'M')) {
    // Find max(abs(A(i,j))).
    anorm = std::abs(D(n));
    for (blasint i = 1; i <= n - 1; ++i) {
      R sum = std::abs(D(i));
      if (anorm < sum || std::isnan(sum)) anorm = sum;
      sum = std::abs(E(i));
      if (anorm < sum || std::isnan(sum)) anorm = sum;
    }
  } else if (Lsame(norm, 'O') || norm == '1' || Lsame(norm, 'I')) {
    // Find norm1(A).
    if (n == 1) {
      anorm = std::abs(D(1));
    } else {
      anorm = std::abs(D(1)) + std::abs(E(1));
      R sum = std::abs(E(n - 1)) + std::abs(D(n));
      if (anorm < sum || std::isnan(sum)) anorm = sum;
      for (blasint i = 2; i <= n - 1; ++i) {
        sum = std::abs(D(i)) + std::abs(E(i)) + std::abs(E(i - 1));
        if (anorm < sum || std::isnan(sum)) anorm = sum;
      }
    }
  } else if (Lsame(norm, 'F') || Lsame(norm, 'E')) {
    // Find normF(A).
    R scale = zero;
    R sum = one;
    if (n > 1) {
      Lassq(n - 1, e, 1, &scale, &sum);
      sum = 2 * sum;
    }
    Lassq(n, d, 1, &scale, &sum);
    anorm = scale * std::sqrt(sum);
  }
  return anorm;
}

// xLASRT: sorts the numbers in D in increasing (ID = 'I') or decreasing
// (ID = 'D') order, with Quick Sort, reverting to Insertion Sort on arrays
// of size <= 20. The dimension of STACK limits N to about 2**32.
template <typename R>
blasint Lasrt(char id, blasint n, R* d) {
  constexpr blasint select = 20;
  const Vector<R> D(d);
  blasint stack[2][32];  // STACK(2, 32), as stack[0..1][stkpnt - 1]

  // Test the input parameters.
  blasint info = 0;
  int dir = -1;
  if (Lsame(id, 'D')) {
    dir = 0;
  } else if (Lsame(id, 'I')) {
    dir = 1;
  }
  if (dir == -1) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  }
  if (info != 0) {
    Xerbla<R>("LASRT", -info);
    return info;
  }

  // Quick return if possible.
  if (n <= 1) return info;

  blasint stkpnt = 1;
  stack[0][0] = 1;
  stack[1][0] = n;
  const auto push = [&](blasint first, blasint last) {
    stkpnt = stkpnt + 1;
    stack[0][stkpnt - 1] = first;
    stack[1][stkpnt - 1] = last;
  };
  do {
    const blasint start = stack[0][stkpnt - 1];
    const blasint endd = stack[1][stkpnt - 1];
    stkpnt = stkpnt - 1;
    if (endd - start <= select && endd - start > 0) {
      // Do Insertion sort on D( START:ENDD ).
      if (dir == 0) {
        // Sort into decreasing order.
        for (blasint i = start + 1; i <= endd; ++i) {
          for (blasint j = i; j >= start + 1; --j) {
            if (D(j) > D(j - 1)) {
              std::swap(D(j), D(j - 1));
            } else {
              break;
            }
          }
        }
      } else {
        // Sort into increasing order.
        for (blasint i = start + 1; i <= endd; ++i) {
          for (blasint j = i; j >= start + 1; --j) {
            if (D(j) < D(j - 1)) {
              std::swap(D(j), D(j - 1));
            } else {
              break;
            }
          }
        }
      }
    } else if (endd - start > select) {
      // Partition D( START:ENDD ) and stack parts, largest one first.
      //
      // Choose partition entry as median of 3.
      const R d1 = D(start);
      const R d2 = D(endd);
      const R d3 = D((start + endd) / 2);
      R dmnmx;
      if (d1 < d2) {
        if (d3 < d1) {
          dmnmx = d1;
        } else if (d3 < d2) {
          dmnmx = d3;
        } else {
          dmnmx = d2;
        }
      } else {
        if (d3 < d2) {
          dmnmx = d2;
        } else if (d3 < d1) {
          dmnmx = d3;
        } else {
          dmnmx = d1;
        }
      }

      blasint i = start - 1;
      blasint j = endd + 1;
      if (dir == 0) {
        // Sort into decreasing order.
        for (;;) {
          do {
            j = j - 1;
          } while (D(j) < dmnmx);
          do {
            i = i + 1;
          } while (D(i) > dmnmx);
          if (i < j) {
            std::swap(D(i), D(j));
          } else {
            break;
          }
        }
      } else {
        // Sort into increasing order.
        for (;;) {
          do {
            j = j - 1;
          } while (D(j) > dmnmx);
          do {
            i = i + 1;
          } while (D(i) < dmnmx);
          if (i < j) {
            std::swap(D(i), D(j));
          } else {
            break;
          }
        }
      }
      if (j - start > endd - j - 1) {
        push(start, j);
        push(j + 1, endd);
      } else {
        push(j + 1, endd);
        push(start, j);
      }
    }
  } while (stkpnt > 0);
  return info;
}

// xLASR: applies a sequence of N-1 (SIDE = 'R') or M-1 (SIDE = 'L') plane
// rotations P(k) = [ c(k) s(k); -s(k) c(k) ] to the M-by-N matrix A, from
// the left (A := P*A) or the right (A := A*P**T), where P = P(z-1) . . .
// P(2) P(1) (DIRECT = 'F') or P(1) P(2) . . . P(z-1) (DIRECT = 'B'), and
// P(k) acts in the plane (k, k+1) (PIVOT = 'V', variable), (1, k+1)
// (PIVOT = 'T', top) or (k, z) (PIVOT = 'B', bottom), z = M or N. For
// complex T, the rotations are real.
//
// As in LAPACK, an invalid argument is reported with its position, but
// there is no INFO argument.
template <typename T>
void Lasr(char side, char pivot, char direct, blasint m, blasint n,
          const RealT<T>* c, const RealT<T>* s, T* a, blasint lda) {
  using R = RealT<T>;
  const R one = R(1), zero = R(0);
  const Vector<const R> C(c);
  const Vector<const R> S(s);
  const Matrix<T> A(a, lda);

  // Test the input parameters.
  blasint info = 0;
  if (!(Lsame(side, 'L') || Lsame(side, 'R'))) {
    info = 1;
  } else if (!(Lsame(pivot, 'V') || Lsame(pivot, 'T') || Lsame(pivot, 'B'))) {
    info = 2;
  } else if (!(Lsame(direct, 'F') || Lsame(direct, 'B'))) {
    info = 3;
  } else if (m < 0) {
    info = 4;
  } else if (n < 0) {
    info = 5;
  } else if (lda < std::max<blasint>(1, m)) {
    info = 9;
  }
  if (info != 0) {
    Xerbla<T>("LASR", info);
    return;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) return;

  // The rotation of rows (side 'L'; in columns i0:i1) or columns (side 'R')
  // p and q with (c, s): x(q) := c*x(q) - s*x(p), x(p) := s*x(q) + c*x(p).
  const auto rotate_rows = [&](blasint q, blasint p, R ctemp, R stemp,
                               blasint i0, blasint i1) {
    for (blasint i = i0; i <= i1; ++i) {
      const T temp = A(q, i);
      A(q, i) = ctemp * temp - stemp * A(p, i);
      A(p, i) = stemp * temp + ctemp * A(p, i);
    }
  };
  const auto rotate_columns = [&](blasint q, blasint p, R ctemp, R stemp) {
    for (blasint i = 1; i <= m; ++i) {
      const T temp = A(i, q);
      A(i, q) = ctemp * temp - stemp * A(i, p);
      A(i, p) = stemp * temp + ctemp * A(i, p);
    }
  };
  // Pivot 'B': x(j) := s*x(z) + c*x(j), x(z) := c*x(z) - s*x(j).
  const auto rotate_rows_bottom = [&](blasint j, R ctemp, R stemp, blasint i0,
                                      blasint i1) {
    for (blasint i = i0; i <= i1; ++i) {
      const T temp = A(j, i);
      A(j, i) = stemp * A(m, i) + ctemp * temp;
      A(m, i) = ctemp * A(m, i) - stemp * temp;
    }
  };
  const auto rotate_columns_bottom = [&](blasint j, R ctemp, R stemp) {
    for (blasint i = 1; i <= m; ++i) {
      const T temp = A(i, j);
      A(i, j) = stemp * A(i, n) + ctemp * temp;
      A(i, n) = ctemp * A(i, n) - stemp * temp;
    }
  };
  const auto nontrivial = [&](R ctemp, R stemp) {
    return ctemp != one || stemp != zero;
  };

  const bool forward = Lsame(direct, 'F');
  if (Lsame(side, 'L')) {
    // Form P * A.
    //
    // Deviation (performance): LAPACK applies each rotation to all columns
    // in turn, which accesses the rows of A with stride LDA. Here the
    // columns are processed in blocks of kBlock, and each block takes all
    // rotations in turn: every element undergoes the same operations in the
    // same order as in LAPACK, so that the result is identical, but the
    // accesses are contiguous (and the columns of a block are independent
    // dependency chains).
    constexpr blasint kBlock = 8;
    for (blasint i0 = 1; i0 <= n; i0 += kBlock) {
      const blasint i1 = std::min(n, i0 + kBlock - 1);
      if (Lsame(pivot, 'V')) {
        for (blasint k = 1; k <= m - 1; ++k) {
          const blasint j = forward ? k : m - k;
          if (nontrivial(C(j), S(j))) rotate_rows(j + 1, j, C(j), S(j), i0, i1);
        }
      } else if (Lsame(pivot, 'T')) {
        for (blasint k = 2; k <= m; ++k) {
          const blasint j = forward ? k : m + 2 - k;
          if (nontrivial(C(j - 1), S(j - 1))) {
            rotate_rows(j, 1, C(j - 1), S(j - 1), i0, i1);
          }
        }
      } else if (Lsame(pivot, 'B')) {
        for (blasint k = 1; k <= m - 1; ++k) {
          const blasint j = forward ? k : m - k;
          if (nontrivial(C(j), S(j))) rotate_rows_bottom(j, C(j), S(j), i0, i1);
        }
      }
    }
  } else if (Lsame(side, 'R')) {
    // Form A * P**T.
    if (Lsame(pivot, 'V')) {
      for (blasint k = 1; k <= n - 1; ++k) {
        const blasint j = forward ? k : n - k;
        if (nontrivial(C(j), S(j))) rotate_columns(j + 1, j, C(j), S(j));
      }
    } else if (Lsame(pivot, 'T')) {
      for (blasint k = 2; k <= n; ++k) {
        const blasint j = forward ? k : n + 2 - k;
        if (nontrivial(C(j - 1), S(j - 1))) {
          rotate_columns(j, 1, C(j - 1), S(j - 1));
        }
      }
    } else if (Lsame(pivot, 'B')) {
      for (blasint k = 1; k <= n - 1; ++k) {
        const blasint j = forward ? k : n - k;
        if (nontrivial(C(j), S(j))) rotate_columns_bottom(j, C(j), S(j));
      }
    }
  }
}

// xSTERF: computes all eigenvalues of the N-by-N symmetric tridiagonal
// matrix with diagonal D and off-diagonal E (destroyed), in increasing
// order in D, with the Pal-Walker-Kahan variant of the QL or QR algorithm.
// Returns INFO: 0, -i for an illegal argument i, or i > 0 if the algorithm
// failed to find all of the eigenvalues in 30*N iterations (i elements of
// E have not converged to zero).
template <typename R>
blasint Sterf(blasint n, R* d, R* e) {
  const R zero = R(0), one = R(1), two = R(2), three = R(3);
  constexpr blasint maxit = 30;
  const Vector<R> D(d);
  const Vector<R> E(e);

  blasint i, iscale, jtot, l, l1, lend, lendsv, lsv, m, nmaxit;
  R alpha, anorm, bb, c, eps, eps2, gamma, oldc, oldgam, p, r, rt1, rt2, rte, s,
      safmax, safmin, sigma, ssfmax, ssfmin;

  // Test the input parameters.
  blasint info = 0;

  // Quick return if possible.
  if (n < 0) {
    info = -1;
    Xerbla<R>("STERF", -info);
    return info;
  }
  if (n <= 1) return info;

  // Determine the unit roundoff for this environment.
  eps = Lamch<R>('E');
  eps2 = eps * eps;
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ssfmax = std::sqrt(safmax) / three;
  ssfmin = std::sqrt(safmin) / eps2;

  // Compute the eigenvalues of the tridiagonal matrix.
  nmaxit = n * maxit;
  sigma = zero;
  jtot = 0;

  // Determine where the matrix splits and choose QL or QR iteration for
  // each block, according to whether top or bottom diagonal element is
  // smaller.
  l1 = 1;

L10:
  if (l1 > n) goto L170;
  if (l1 > 1) E(l1 - 1) = zero;
  for (m = l1; m <= n - 1; ++m) {
    if (std::abs(E(m)) <=
        (std::sqrt(std::abs(D(m))) * std::sqrt(std::abs(D(m + 1)))) * eps) {
      E(m) = zero;
      goto L30;
    }
  }
  m = n;

L30:
  l = l1;
  lsv = l;
  lend = m;
  lendsv = lend;
  l1 = m + 1;
  if (lend == l) goto L10;

  // Scale submatrix in rows and columns L to LEND.
  anorm = Lanst('M', lend - l + 1, D.ptr(l), E.ptr(l));
  iscale = 0;
  if (anorm == zero) goto L10;
  if (anorm > ssfmax) {
    iscale = 1;
    Lascl('G', 0, 0, anorm, ssfmax, lend - l + 1, 1, D.ptr(l), n);
    Lascl('G', 0, 0, anorm, ssfmax, lend - l, 1, E.ptr(l), n);
  } else if (anorm < ssfmin) {
    iscale = 2;
    Lascl('G', 0, 0, anorm, ssfmin, lend - l + 1, 1, D.ptr(l), n);
    Lascl('G', 0, 0, anorm, ssfmin, lend - l, 1, E.ptr(l), n);
  }

  for (i = l; i <= lend - 1; ++i) E(i) = E(i) * E(i);

  // Choose between QL and QR iteration.
  if (std::abs(D(lend)) < std::abs(D(l))) {
    lend = lsv;
    l = lendsv;
  }

  if (lend >= l) {
    // QL Iteration.
    //
    // Look for small subdiagonal element.
  L50:
    if (l != lend) {
      for (m = l; m <= lend - 1; ++m) {
        if (std::abs(E(m)) <= eps2 * std::abs(D(m) * D(m + 1))) goto L70;
      }
    }
    m = lend;

  L70:
    if (m < lend) E(m) = zero;
    p = D(l);
    if (m == l) goto L90;

    // If remaining matrix is 2 by 2, use DLAE2 to compute its eigenvalues.
    if (m == l + 1) {
      rte = std::sqrt(E(l));
      Lae2(D(l), rte, D(l + 1), &rt1, &rt2);
      D(l) = rt1;
      D(l + 1) = rt2;
      E(l) = zero;
      l = l + 2;
      if (l <= lend) goto L50;
      goto L150;
    }

    if (jtot == nmaxit) goto L150;
    jtot = jtot + 1;

    // Form shift.
    rte = std::sqrt(E(l));
    sigma = (D(l + 1) - p) / (two * rte);
    r = Lapy2(sigma, one);
    sigma = p - (rte / (sigma + std::copysign(r, sigma)));

    c = one;
    s = zero;
    gamma = D(m) - sigma;
    p = gamma * gamma;

    // Inner loop.
    for (i = m - 1; i >= l; --i) {
      bb = E(i);
      r = p + bb;
      if (i != m - 1) E(i + 1) = s * r;
      oldc = c;
      c = p / r;
      s = bb / r;
      oldgam = gamma;
      alpha = D(i);
      gamma = c * (alpha - sigma) - s * oldgam;
      D(i + 1) = oldgam + (alpha - gamma);
      if (c != zero) {
        p = (gamma * gamma) / c;
      } else {
        p = oldc * bb;
      }
    }

    E(l) = s * p;
    D(l) = sigma + gamma;
    goto L50;

    // Eigenvalue found.
  L90:
    D(l) = p;

    l = l + 1;
    if (l <= lend) goto L50;
    goto L150;
  } else {
    // QR Iteration.
    //
    // Look for small superdiagonal element.
  L100:
    for (m = l; m >= lend + 1; --m) {
      if (std::abs(E(m - 1)) <= eps2 * std::abs(D(m) * D(m - 1))) goto L120;
    }
    m = lend;

  L120:
    if (m > lend) E(m - 1) = zero;
    p = D(l);
    if (m == l) goto L140;

    // If remaining matrix is 2 by 2, use DLAE2 to compute its eigenvalues.
    if (m == l - 1) {
      rte = std::sqrt(E(l - 1));
      Lae2(D(l), rte, D(l - 1), &rt1, &rt2);
      D(l) = rt1;
      D(l - 1) = rt2;
      E(l - 1) = zero;
      l = l - 2;
      if (l >= lend) goto L100;
      goto L150;
    }

    if (jtot == nmaxit) goto L150;
    jtot = jtot + 1;

    // Form shift.
    rte = std::sqrt(E(l - 1));
    sigma = (D(l - 1) - p) / (two * rte);
    r = Lapy2(sigma, one);
    sigma = p - (rte / (sigma + std::copysign(r, sigma)));

    c = one;
    s = zero;
    gamma = D(m) - sigma;
    p = gamma * gamma;

    // Inner loop.
    for (i = m; i <= l - 1; ++i) {
      bb = E(i);
      r = p + bb;
      if (i != m) E(i - 1) = s * r;
      oldc = c;
      c = p / r;
      s = bb / r;
      oldgam = gamma;
      alpha = D(i + 1);
      gamma = c * (alpha - sigma) - s * oldgam;
      D(i) = oldgam + (alpha - gamma);
      if (c != zero) {
        p = (gamma * gamma) / c;
      } else {
        p = oldc * bb;
      }
    }

    E(l - 1) = s * p;
    D(l) = sigma + gamma;
    goto L100;

    // Eigenvalue found.
  L140:
    D(l) = p;

    l = l - 1;
    if (l >= lend) goto L100;
    goto L150;
  }

  // Undo scaling if necessary.
L150:
  if (iscale == 1) {
    Lascl('G', 0, 0, ssfmax, anorm, lendsv - lsv + 1, 1, D.ptr(lsv), n);
  }
  if (iscale == 2) {
    Lascl('G', 0, 0, ssfmin, anorm, lendsv - lsv + 1, 1, D.ptr(lsv), n);
  }

  // Check for no convergence to an eigenvalue after a total of N*MAXIT
  // iterations.
  if (jtot < nmaxit) goto L10;
  for (i = 1; i <= n - 1; ++i) {
    if (E(i) != zero) info = info + 1;
  }
  return info;

  // Sort eigenvalues in increasing order.
L170:
  Lasrt('I', n, d);
  return info;
}

// xSTEQR: computes all eigenvalues and, optionally, eigenvectors of the
// N-by-N symmetric tridiagonal matrix with diagonal D and off-diagonal E
// (destroyed) with the implicit QL or QR method; the eigenvalues are
// returned in increasing order in D. The eigenvectors of a full or band
// symmetric (Hermitian) matrix can also be found if xSYTRD / xHETRD has
// been used to reduce it to tridiagonal form:
//   COMPZ = 'N': eigenvalues only;
//   COMPZ = 'V': Z (N-by-N, on entry the orthogonal / unitary matrix of
//                the reduction) is overwritten with the eigenvectors of
//                the original matrix;
//   COMPZ = 'I': Z is set to the eigenvectors of the tridiagonal matrix.
// WORK has max(1, 2*N-2) elements (not referenced for COMPZ = 'N').
// Returns INFO: 0, -i for an illegal argument i, or i > 0 if the algorithm
// failed to find all of the eigenvalues in 30*N iterations (i elements of
// E have not converged to zero; D and E contain the elements of a
// tridiagonal matrix orthogonally similar to the original one).
//
// The real and the complex routine differ in the norm with which a block is
// scaled before its iteration: max |a(i,j)| for real T, the one norm for
// complex T.
template <typename T>
blasint Steqr(char compz, blasint n, RealT<T>* d, RealT<T>* e, T* z,
              blasint ldz, RealT<T>* work) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1), two = R(2), three = R(3);
  constexpr blasint maxit = 30;
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Matrix<T> Z(z, ldz);
  const Vector<R> WORK(work);
  const char scaling_norm = IsComplex<T>::value ? 'I' : 'M';

  blasint i, icompz, ii, iscale, j, jtot, k, l, l1, lend, lendm1, lendp1,
      lendsv, lm1, lsv, m, mm, mm1, nm1, nmaxit;
  R anorm, b, c, eps, eps2, f, g, p, r, rt1, rt2, s, safmax, safmin, ssfmax,
      ssfmin, tst;

  // Test the input parameters.
  blasint info = 0;

  if (Lsame(compz, 'N')) {
    icompz = 0;
  } else if (Lsame(compz, 'V')) {
    icompz = 1;
  } else if (Lsame(compz, 'I')) {
    icompz = 2;
  } else {
    icompz = -1;
  }
  if (icompz < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldz < 1 || (icompz > 0 && ldz < std::max<blasint>(1, n))) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("STEQR", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  if (n == 1) {
    if (icompz == 2) Z(1, 1) = T(one);
    return info;
  }

  // Determine the unit roundoff and over/underflow thresholds.
  eps = Lamch<R>('E');
  eps2 = eps * eps;
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ssfmax = std::sqrt(safmax) / three;
  ssfmin = std::sqrt(safmin) / eps2;

  // Compute the eigenvalues and eigenvectors of the tridiagonal matrix.
  if (icompz == 2) Laset('F', n, n, T(zero), T(one), z, ldz);

  nmaxit = n * maxit;
  jtot = 0;

  // Determine where the matrix splits and choose QL or QR iteration for
  // each block, according to whether top or bottom diagonal element is
  // smaller.
  l1 = 1;
  nm1 = n - 1;

L10:
  if (l1 > n) goto L160;
  if (l1 > 1) E(l1 - 1) = zero;
  if (l1 <= nm1) {
    for (m = l1; m <= nm1; ++m) {
      tst = std::abs(E(m));
      if (tst == zero) goto L30;
      if (tst <=
          (std::sqrt(std::abs(D(m))) * std::sqrt(std::abs(D(m + 1)))) * eps) {
        E(m) = zero;
        goto L30;
      }
    }
  }
  m = n;

L30:
  l = l1;
  lsv = l;
  lend = m;
  lendsv = lend;
  l1 = m + 1;
  if (lend == l) goto L10;

  // Scale submatrix in rows and columns L to LEND.
  anorm = Lanst(scaling_norm, lend - l + 1, D.ptr(l), E.ptr(l));
  iscale = 0;
  if (anorm == zero) goto L10;
  if (anorm > ssfmax) {
    iscale = 1;
    Lascl('G', 0, 0, anorm, ssfmax, lend - l + 1, 1, D.ptr(l), n);
    Lascl('G', 0, 0, anorm, ssfmax, lend - l, 1, E.ptr(l), n);
  } else if (anorm < ssfmin) {
    iscale = 2;
    Lascl('G', 0, 0, anorm, ssfmin, lend - l + 1, 1, D.ptr(l), n);
    Lascl('G', 0, 0, anorm, ssfmin, lend - l, 1, E.ptr(l), n);
  }

  // Choose between QL and QR iteration.
  if (std::abs(D(lend)) < std::abs(D(l))) {
    lend = lsv;
    l = lendsv;
  }

  if (lend > l) {
    // QL Iteration.
    //
    // Look for small subdiagonal element.
  L40:
    if (l != lend) {
      lendm1 = lend - 1;
      for (m = l; m <= lendm1; ++m) {
        tst = std::abs(E(m)) * std::abs(E(m));
        if (tst <= (eps2 * std::abs(D(m))) * std::abs(D(m + 1)) + safmin) {
          goto L60;
        }
      }
    }

    m = lend;

  L60:
    if (m < lend) E(m) = zero;
    p = D(l);
    if (m == l) goto L80;

    // If remaining matrix is 2-by-2, use DLAE2 or SLAEV2 to compute its
    // eigensystem.
    if (m == l + 1) {
      if (icompz > 0) {
        Laev2(D(l), E(l), D(l + 1), &rt1, &rt2, &c, &s);
        WORK(l) = c;
        WORK(n - 1 + l) = s;
        Lasr('R', 'V', 'B', n, 2, WORK.ptr(l), WORK.ptr(n - 1 + l), Z.ptr(1, l),
             ldz);
      } else {
        Lae2(D(l), E(l), D(l + 1), &rt1, &rt2);
      }
      D(l) = rt1;
      D(l + 1) = rt2;
      E(l) = zero;
      l = l + 2;
      if (l <= lend) goto L40;
      goto L140;
    }

    if (jtot == nmaxit) goto L140;
    jtot = jtot + 1;

    // Form shift.
    g = (D(l + 1) - p) / (two * E(l));
    r = Lapy2(g, one);
    g = D(m) - p + (E(l) / (g + std::copysign(r, g)));

    s = one;
    c = one;
    p = zero;

    // Inner loop.
    mm1 = m - 1;
    for (i = mm1; i >= l; --i) {
      f = s * E(i);
      b = c * E(i);
      Lartg(g, f, &c, &s, &r);
      if (i != m - 1) E(i + 1) = r;
      g = D(i + 1) - p;
      r = (D(i) - g) * s + two * c * b;
      p = s * r;
      D(i + 1) = g + p;
      g = c * r - b;

      // If eigenvectors are desired, then save rotations.
      if (icompz > 0) {
        WORK(i) = c;
        WORK(n - 1 + i) = -s;
      }
    }

    // If eigenvectors are desired, then apply saved rotations.
    if (icompz > 0) {
      mm = m - l + 1;
      Lasr('R', 'V', 'B', n, mm, WORK.ptr(l), WORK.ptr(n - 1 + l), Z.ptr(1, l),
           ldz);
    }

    D(l) = D(l) - p;
    E(l) = g;
    goto L40;

    // Eigenvalue found.
  L80:
    D(l) = p;

    l = l + 1;
    if (l <= lend) goto L40;
    goto L140;
  } else {
    // QR Iteration.
    //
    // Look for small superdiagonal element.
  L90:
    if (l != lend) {
      lendp1 = lend + 1;
      for (m = l; m >= lendp1; --m) {
        tst = std::abs(E(m - 1)) * std::abs(E(m - 1));
        if (tst <= (eps2 * std::abs(D(m))) * std::abs(D(m - 1)) + safmin) {
          goto L110;
        }
      }
    }

    m = lend;

  L110:
    if (m > lend) E(m - 1) = zero;
    p = D(l);
    if (m == l) goto L130;

    // If remaining matrix is 2-by-2, use DLAE2 or SLAEV2 to compute its
    // eigensystem.
    if (m == l - 1) {
      if (icompz > 0) {
        Laev2(D(l - 1), E(l - 1), D(l), &rt1, &rt2, &c, &s);
        WORK(m) = c;
        WORK(n - 1 + m) = s;
        Lasr('R', 'V', 'F', n, 2, WORK.ptr(m), WORK.ptr(n - 1 + m),
             Z.ptr(1, l - 1), ldz);
      } else {
        Lae2(D(l - 1), E(l - 1), D(l), &rt1, &rt2);
      }
      D(l - 1) = rt1;
      D(l) = rt2;
      E(l - 1) = zero;
      l = l - 2;
      if (l >= lend) goto L90;
      goto L140;
    }

    if (jtot == nmaxit) goto L140;
    jtot = jtot + 1;

    // Form shift.
    g = (D(l - 1) - p) / (two * E(l - 1));
    r = Lapy2(g, one);
    g = D(m) - p + (E(l - 1) / (g + std::copysign(r, g)));

    s = one;
    c = one;
    p = zero;

    // Inner loop.
    lm1 = l - 1;
    for (i = m; i <= lm1; ++i) {
      f = s * E(i);
      b = c * E(i);
      Lartg(g, f, &c, &s, &r);
      if (i != m) E(i - 1) = r;
      g = D(i) - p;
      r = (D(i + 1) - g) * s + two * c * b;
      p = s * r;
      D(i) = g + p;
      g = c * r - b;

      // If eigenvectors are desired, then save rotations.
      if (icompz > 0) {
        WORK(i) = c;
        WORK(n - 1 + i) = s;
      }
    }

    // If eigenvectors are desired, then apply saved rotations.
    if (icompz > 0) {
      mm = l - m + 1;
      Lasr('R', 'V', 'F', n, mm, WORK.ptr(m), WORK.ptr(n - 1 + m), Z.ptr(1, m),
           ldz);
    }

    D(l) = D(l) - p;
    E(lm1) = g;
    goto L90;

    // Eigenvalue found.
  L130:
    D(l) = p;

    l = l - 1;
    if (l >= lend) goto L90;
    goto L140;
  }

  // Undo scaling if necessary.
L140:
  if (iscale == 1) {
    Lascl('G', 0, 0, ssfmax, anorm, lendsv - lsv + 1, 1, D.ptr(lsv), n);
    Lascl('G', 0, 0, ssfmax, anorm, lendsv - lsv, 1, E.ptr(lsv), n);
  } else if (iscale == 2) {
    Lascl('G', 0, 0, ssfmin, anorm, lendsv - lsv + 1, 1, D.ptr(lsv), n);
    Lascl('G', 0, 0, ssfmin, anorm, lendsv - lsv, 1, E.ptr(lsv), n);
  }

  // Check for no convergence to an eigenvalue after a total of N*MAXIT
  // iterations.
  if (jtot < nmaxit) goto L10;
  for (i = 1; i <= n - 1; ++i) {
    if (E(i) != zero) info = info + 1;
  }
  return info;

  // Order eigenvalues and eigenvectors.
L160:
  if (icompz == 0) {
    // Use Quick Sort.
    Lasrt('I', n, d);
  } else {
    // Use Selection Sort to minimize swaps of eigenvectors.
    for (ii = 2; ii <= n; ++ii) {
      i = ii - 1;
      k = i;
      p = D(i);
      for (j = ii; j <= n; ++j) {
        if (D(j) < p) {
          k = j;
          p = D(j);
        }
      }
      if (k != i) {
        D(k) = D(i);
        D(i) = p;
        Swap(n, Z.ptr(1, i), 1, Z.ptr(1, k), 1);
      }
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template void Lasr<T>(char, char, char, blasint, blasint, const RealT<T>*,  \
                        const RealT<T>*, T*, blasint);                        \
  template blasint Steqr<T>(char, blasint, RealT<T>*, RealT<T>*, T*, blasint, \
                            RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

#define HWYBLAS_LAPACK_INSTANTIATE(R)                     \
  template void Lae2<R>(R, R, R, R*, R*);                 \
  template void Laev2<R>(R, R, R, R*, R*, R*, R*);        \
  template void Lartg<R>(R, R, R*, R*, R*);               \
  template R Lanst<R>(char, blasint, const R*, const R*); \
  template blasint Lasrt<R>(char, blasint, R*);           \
  template blasint Sterf<R>(blasint, R*, R*);
HWYBLAS_LAPACK_INSTANTIATE(float)
HWYBLAS_LAPACK_INSTANTIATE(double)
#undef HWYBLAS_LAPACK_INSTANTIATE

template void Lartg<c32>(c32, c32, float*, c32*, c32*);
template void Lartg<c64>(c64, c64, double*, c64*, c64*);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_STEQR(prefix, T, R)                                     \
  HWYBLAS_API void prefix##lasr_(const char* side, const char* pivot,          \
                                 const char* direct, const hwyblas_int* m,     \
                                 const hwyblas_int* n, const R* c, const R* s, \
                                 R* a, const hwyblas_int* lda) {               \
    hwyblas::lapack::Lasr(*side, *pivot, *direct, *m, *n, c, s, As<T>(a),      \
                          *lda);                                               \
  }                                                                            \
  HWYBLAS_API void prefix##steqr_(const char* compz, const hwyblas_int* n,     \
                                  R* d, R* e, R* z, const hwyblas_int* ldz,    \
                                  R* work, hwyblas_int* info) {                \
    *info = hwyblas::lapack::Steqr(*compz, *n, d, e, As<T>(z), *ldz, work);    \
  }

HWYBLAS_LAPACK_STEQR(s, float, float)
HWYBLAS_LAPACK_STEQR(d, double, double)
HWYBLAS_LAPACK_STEQR(c, c32, float)
HWYBLAS_LAPACK_STEQR(z, c64, double)

#undef HWYBLAS_LAPACK_STEQR

#define HWYBLAS_LAPACK_STEQR_REAL(prefix, R)                                  \
  HWYBLAS_API void prefix##lae2_(const R* a, const R* b, const R* c, R* rt1,  \
                                 R* rt2) {                                    \
    hwyblas::lapack::Lae2(*a, *b, *c, rt1, rt2);                              \
  }                                                                           \
  HWYBLAS_API void prefix##laev2_(const R* a, const R* b, const R* c, R* rt1, \
                                  R* rt2, R* cs1, R* sn1) {                   \
    hwyblas::lapack::Laev2(*a, *b, *c, rt1, rt2, cs1, sn1);                   \
  }                                                                           \
  HWYBLAS_API void prefix##lartg_(const R* f, const R* g, R* c, R* s, R* r) { \
    hwyblas::lapack::Lartg(*f, *g, c, s, r);                                  \
  }                                                                           \
  HWYBLAS_API R prefix##lanst_(const char* norm, const hwyblas_int* n,        \
                               const R* d, const R* e) {                      \
    return hwyblas::lapack::Lanst(*norm, *n, d, e);                           \
  }                                                                           \
  HWYBLAS_API void prefix##lasrt_(const char* id, const hwyblas_int* n, R* d, \
                                  hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Lasrt(*id, *n, d);                               \
  }                                                                           \
  HWYBLAS_API void prefix##sterf_(const hwyblas_int* n, R* d, R* e,           \
                                  hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Sterf(*n, d, e);                                 \
  }

HWYBLAS_LAPACK_STEQR_REAL(s, float)
HWYBLAS_LAPACK_STEQR_REAL(d, double)

#undef HWYBLAS_LAPACK_STEQR_REAL

HWYBLAS_API void clartg_(const float* f, const float* g, float* c, float* s,
                         float* r) {
  hwyblas::lapack::Lartg(*As<const c32>(f), *As<const c32>(g), c, As<c32>(s),
                         As<c32>(r));
}
HWYBLAS_API void zlartg_(const double* f, const double* g, double* c, double* s,
                         double* r) {
  hwyblas::lapack::Lartg(*As<const c64>(f), *As<const c64>(g), c, As<c64>(s),
                         As<c64>(r));
}

}  // extern "C"
