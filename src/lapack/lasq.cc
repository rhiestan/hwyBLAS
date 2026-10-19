// The singular values of a real bidiagonal matrix, or the eigenvalues of
// the positive definite tridiagonal matrix of a qd array, to high relative
// accuracy with the dqds algorithm (differential quotient-difference with
// shifts) of Fernando and Parlett, as developed by Parlett and Marques:
// xLASQ1 (driver), xLASQ2 (the qd array), xLASQ3 (one dqds step with
// deflation), xLASQ4 (the shift), xLASQ5 (one dqds transform) and xLASQ6
// (one dqd transform without shift). Real R only.
//
// The qd array Z holds the 2*N-1 values q(1), e(1), q(2), e(2), ..., q(n)
// (xLASQ2), or, interleaved for the ping-pong iteration, 4*N values with
// PP = 0 (ping) or 1 (pong) selecting the current half.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>
#include <type_traits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// MIN(a, b) of xLASQ5, NaN if a or b is NaN: the IEEE variant of the dqds
// transform lets a NaN arise and relies on it reaching DMIN, which xLASQ3
// tests. (The Fortran standard leaves MIN with a NaN argument unspecified;
// Min, as flang's MIN, returns a NaN only if it is its second argument.)
template <typename R>
R MinNan(R a, R b) {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  return Min(a, b);
}

}  // namespace

// xLASQ6: one dqd transform (without shift) in ping-pong form of the qd
// array Z for the block I0:N0, with protection against underflow and
// overflow. Returns the minimum DMIN of the d's and the last three d's DN,
// DNM1 and DNM2, and the minimum DMIN1 (DMIN2) excluding DN (and DNM1).
template <typename R>
void Lasq6(blasint i0, blasint n0, R* z, blasint pp, R* dmin, R* dmin1,
           R* dmin2, R* dn, R* dnm1, R* dnm2) {
  const R zero = R(0);
  const Vector<R> Z(z);
  blasint j4, j4p2;
  R d, emin, safmin, temp;
  if ((n0 - i0 - 1) <= 0) return;
  safmin = Lamch<R>('S');
  j4 = 4 * i0 + pp - 3;
  emin = Z(j4 + 4);
  d = Z(j4);
  *dmin = d;
  if (pp == 0) {
    for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
      Z(j4 - 2) = d + Z(j4 - 1);
      if (Z(j4 - 2) == zero) {
        Z(j4) = zero;
        d = Z(j4 + 1);
        *dmin = d;
        emin = zero;
      } else if (safmin * Z(j4 + 1) < Z(j4 - 2) &&
                 safmin * Z(j4 - 2) < Z(j4 + 1)) {
        temp = Z(j4 + 1) / Z(j4 - 2);
        Z(j4) = Z(j4 - 1) * temp;
        d = d * temp;
      } else {
        Z(j4) = Z(j4 + 1) * (Z(j4 - 1) / Z(j4 - 2));
        d = Z(j4 + 1) * (d / Z(j4 - 2));
      }
      *dmin = Min(*dmin, d);
      emin = Min(emin, Z(j4));
    }
  } else {
    for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
      Z(j4 - 3) = d + Z(j4);
      if (Z(j4 - 3) == zero) {
        Z(j4 - 1) = zero;
        d = Z(j4 + 2);
        *dmin = d;
        emin = zero;
      } else if (safmin * Z(j4 + 2) < Z(j4 - 3) &&
                 safmin * Z(j4 - 3) < Z(j4 + 2)) {
        temp = Z(j4 + 2) / Z(j4 - 3);
        Z(j4 - 1) = Z(j4) * temp;
        d = d * temp;
      } else {
        Z(j4 - 1) = Z(j4 + 2) * (Z(j4) / Z(j4 - 3));
        d = Z(j4 + 2) * (d / Z(j4 - 3));
      }
      *dmin = Min(*dmin, d);
      emin = Min(emin, Z(j4 - 1));
    }
  }
  // Unroll last two steps.
  *dnm2 = d;
  *dmin2 = *dmin;
  j4 = 4 * (n0 - 2) - pp;
  j4p2 = j4 + 2 * pp - 1;
  Z(j4 - 2) = *dnm2 + Z(j4p2);
  if (Z(j4 - 2) == zero) {
    Z(j4) = zero;
    *dnm1 = Z(j4p2 + 2);
    *dmin = *dnm1;
    emin = zero;
  } else if (safmin * Z(j4p2 + 2) < Z(j4 - 2) &&
             safmin * Z(j4 - 2) < Z(j4p2 + 2)) {
    temp = Z(j4p2 + 2) / Z(j4 - 2);
    Z(j4) = Z(j4p2) * temp;
    *dnm1 = *dnm2 * temp;
  } else {
    Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
    *dnm1 = Z(j4p2 + 2) * (*dnm2 / Z(j4 - 2));
  }
  *dmin = Min(*dmin, *dnm1);
  *dmin1 = *dmin;
  j4 = j4 + 4;
  j4p2 = j4 + 2 * pp - 1;
  Z(j4 - 2) = *dnm1 + Z(j4p2);
  if (Z(j4 - 2) == zero) {
    Z(j4) = zero;
    *dn = Z(j4p2 + 2);
    *dmin = *dn;
    emin = zero;
  } else if (safmin * Z(j4p2 + 2) < Z(j4 - 2) &&
             safmin * Z(j4 - 2) < Z(j4p2 + 2)) {
    temp = Z(j4p2 + 2) / Z(j4 - 2);
    Z(j4) = Z(j4p2) * temp;
    *dn = *dnm1 * temp;
  } else {
    Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
    *dn = Z(j4p2 + 2) * (*dnm1 / Z(j4 - 2));
  }
  *dmin = Min(*dmin, *dn);
  Z(j4 + 2) = *dn;
  Z(4 * n0 - pp) = emin;
}

// xLASQ5: one dqds transform with shift TAU in ping-pong form of the qd
// array Z for the block I0:N0, with the same outputs as xLASQ6. IEEE
// selects the variant that relies on IEEE arithmetic (infinities and NaNs)
// instead of testing for negative d's. TAU is set to 0 if it is
// negligible compared to SIGMA, the accumulated shift; EPS is the machine
// precision.
template <typename R>
void Lasq5(blasint i0, blasint n0, R* z, blasint pp, R* tau, R sigma, R* dmin,
           R* dmin1, R* dmin2, R* dn, R* dnm1, R* dnm2, bool ieee, R eps) {
  const R zero = R(0);
  const R half = R(0.5);
  const Vector<R> Z(z);
  blasint j4, j4p2;
  R d, dthresh, emin, temp;
  if ((n0 - i0 - 1) <= 0) return;
  dthresh = eps * (sigma + *tau);
  if (*tau < dthresh * half) *tau = zero;
  if (*tau != zero) {
    j4 = 4 * i0 + pp - 3;
    emin = Z(j4 + 4);
    d = Z(j4) - *tau;
    *dmin = d;
    *dmin1 = -Z(j4);
    if (ieee) {
      // Code for IEEE arithmetic.
      if (pp == 0) {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 2) = d + Z(j4 - 1);
          temp = Z(j4 + 1) / Z(j4 - 2);
          d = d * temp - *tau;
          *dmin = MinNan(*dmin, d);
          Z(j4) = Z(j4 - 1) * temp;
          emin = MinNan(Z(j4), emin);
        }
      } else {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 3) = d + Z(j4);
          temp = Z(j4 + 2) / Z(j4 - 3);
          d = d * temp - *tau;
          *dmin = MinNan(*dmin, d);
          Z(j4 - 1) = Z(j4) * temp;
          emin = MinNan(Z(j4 - 1), emin);
        }
      }
      // Unroll last two steps.
      *dnm2 = d;
      *dmin2 = *dmin;
      j4 = 4 * (n0 - 2) - pp;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm2 + Z(j4p2);
      Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
      *dnm1 = Z(j4p2 + 2) * (*dnm2 / Z(j4 - 2)) - *tau;
      *dmin = MinNan(*dmin, *dnm1);
      *dmin1 = *dmin;
      j4 = j4 + 4;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm1 + Z(j4p2);
      Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
      *dn = Z(j4p2 + 2) * (*dnm1 / Z(j4 - 2)) - *tau;
      *dmin = MinNan(*dmin, *dn);
    } else {
      // Code for non IEEE arithmetic.
      if (pp == 0) {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 2) = d + Z(j4 - 1);
          if (d < zero) {
            return;
          } else {
            Z(j4) = Z(j4 + 1) * (Z(j4 - 1) / Z(j4 - 2));
            d = Z(j4 + 1) * (d / Z(j4 - 2)) - *tau;
          }
          *dmin = MinNan(*dmin, d);
          emin = MinNan(emin, Z(j4));
        }
      } else {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 3) = d + Z(j4);
          if (d < zero) {
            return;
          } else {
            Z(j4 - 1) = Z(j4 + 2) * (Z(j4) / Z(j4 - 3));
            d = Z(j4 + 2) * (d / Z(j4 - 3)) - *tau;
          }
          *dmin = MinNan(*dmin, d);
          emin = MinNan(emin, Z(j4 - 1));
        }
      }
      // Unroll last two steps.
      *dnm2 = d;
      *dmin2 = *dmin;
      j4 = 4 * (n0 - 2) - pp;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm2 + Z(j4p2);
      if (*dnm2 < zero) {
        return;
      } else {
        Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
        *dnm1 = Z(j4p2 + 2) * (*dnm2 / Z(j4 - 2)) - *tau;
      }
      *dmin = MinNan(*dmin, *dnm1);
      *dmin1 = *dmin;
      j4 = j4 + 4;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm1 + Z(j4p2);
      if (*dnm1 < zero) {
        return;
      } else {
        Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
        *dn = Z(j4p2 + 2) * (*dnm1 / Z(j4 - 2)) - *tau;
      }
      *dmin = MinNan(*dmin, *dn);
    }
  } else {
    // This is the version that sets d's to zero if they are small enough
    j4 = 4 * i0 + pp - 3;
    emin = Z(j4 + 4);
    d = Z(j4) - *tau;
    *dmin = d;
    *dmin1 = -Z(j4);
    if (ieee) {
      // Code for IEEE arithmetic.
      if (pp == 0) {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 2) = d + Z(j4 - 1);
          temp = Z(j4 + 1) / Z(j4 - 2);
          d = d * temp - *tau;
          if (d < dthresh) d = zero;
          *dmin = MinNan(*dmin, d);
          Z(j4) = Z(j4 - 1) * temp;
          emin = MinNan(Z(j4), emin);
        }
      } else {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 3) = d + Z(j4);
          temp = Z(j4 + 2) / Z(j4 - 3);
          d = d * temp - *tau;
          if (d < dthresh) d = zero;
          *dmin = MinNan(*dmin, d);
          Z(j4 - 1) = Z(j4) * temp;
          emin = MinNan(Z(j4 - 1), emin);
        }
      }
      // Unroll last two steps.
      *dnm2 = d;
      *dmin2 = *dmin;
      j4 = 4 * (n0 - 2) - pp;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm2 + Z(j4p2);
      Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
      *dnm1 = Z(j4p2 + 2) * (*dnm2 / Z(j4 - 2)) - *tau;
      *dmin = MinNan(*dmin, *dnm1);
      *dmin1 = *dmin;
      j4 = j4 + 4;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm1 + Z(j4p2);
      Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
      *dn = Z(j4p2 + 2) * (*dnm1 / Z(j4 - 2)) - *tau;
      *dmin = MinNan(*dmin, *dn);
    } else {
      // Code for non IEEE arithmetic.
      if (pp == 0) {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 2) = d + Z(j4 - 1);
          if (d < zero) {
            return;
          } else {
            Z(j4) = Z(j4 + 1) * (Z(j4 - 1) / Z(j4 - 2));
            d = Z(j4 + 1) * (d / Z(j4 - 2)) - *tau;
          }
          if (d < dthresh) d = zero;
          *dmin = MinNan(*dmin, d);
          emin = MinNan(emin, Z(j4));
        }
      } else {
        for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
          Z(j4 - 3) = d + Z(j4);
          if (d < zero) {
            return;
          } else {
            Z(j4 - 1) = Z(j4 + 2) * (Z(j4) / Z(j4 - 3));
            d = Z(j4 + 2) * (d / Z(j4 - 3)) - *tau;
          }
          if (d < dthresh) d = zero;
          *dmin = MinNan(*dmin, d);
          emin = MinNan(emin, Z(j4 - 1));
        }
      }
      // Unroll last two steps.
      *dnm2 = d;
      *dmin2 = *dmin;
      j4 = 4 * (n0 - 2) - pp;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm2 + Z(j4p2);
      if (*dnm2 < zero) {
        return;
      } else {
        Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
        *dnm1 = Z(j4p2 + 2) * (*dnm2 / Z(j4 - 2)) - *tau;
      }
      *dmin = MinNan(*dmin, *dnm1);
      *dmin1 = *dmin;
      j4 = j4 + 4;
      j4p2 = j4 + 2 * pp - 1;
      Z(j4 - 2) = *dnm1 + Z(j4p2);
      if (*dnm1 < zero) {
        return;
      } else {
        Z(j4) = Z(j4p2 + 2) * (Z(j4p2) / Z(j4 - 2));
        *dn = Z(j4p2 + 2) * (*dnm1 / Z(j4 - 2)) - *tau;
      }
      *dmin = MinNan(*dmin, *dn);
    }
  }
  Z(j4 + 2) = *dn;
  Z(4 * n0 - pp) = emin;
}

// xLASQ4: an approximation TAU to the smallest eigenvalue of the qd array
// Z for the block I0:N0, using the values of the previous transform (DMIN,
// DMIN1, DMIN2, DN, DN1, DN2) and N0IN, the value of N0 at the start of
// EIGTEST. TTYPE returns the type of the shift and G is the damping of
// type 6, kept between calls. As in LAPACK, some cases return without
// setting TAU.
template <typename R>
void Lasq4(blasint i0, blasint n0, const R* z, blasint pp, blasint n0in, R dmin,
           R dmin1, R dmin2, R dn, R dn1, R dn2, R* tau, blasint* ttype, R* g) {
  const R cnst1 = R(0.5630);
  const R cnst2 = R(1.010);
  const R cnst3 = R(1.050);
  const R qurtr = R(0.250);
  const R third = R(0.3330);
  const R half = R(0.50);
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R hundrd = R(100);
  const Vector<const R> Z(z);
  blasint i4, nn, np;
  R a2, b1, b2, gam, gap1, gap2;
  // S is set in every case that LAPACK's callers create (N0IN >= N0).
  R s = zero;
  if (dmin <= zero) {
    *tau = -dmin;
    *ttype = -1;
    return;
  }
  nn = 4 * n0 + pp;
  if (n0in == n0) {
    // No eigenvalues deflated.
    if (dmin == dn || dmin == dn1) {
      b1 = std::sqrt(Z(nn - 3)) * std::sqrt(Z(nn - 5));
      b2 = std::sqrt(Z(nn - 7)) * std::sqrt(Z(nn - 9));
      a2 = Z(nn - 7) + Z(nn - 5);
      // Cases 2 and 3.
      if (dmin == dn && dmin1 == dn1) {
        gap2 = dmin2 - a2 - dmin2 * qurtr;
        if (gap2 > zero && gap2 > b2) {
          gap1 = a2 - dn - (b2 / gap2) * b2;
        } else {
          gap1 = a2 - dn - (b1 + b2);
        }
        if (gap1 > zero && gap1 > b1) {
          s = Max(dn - (b1 / gap1) * b1, half * dmin);
          *ttype = -2;
        } else {
          s = zero;
          if (dn > b1) s = dn - b1;
          if (a2 > (b1 + b2)) s = Min(s, a2 - (b1 + b2));
          s = Max(s, third * dmin);
          *ttype = -3;
        }
      } else {
        // Case 4.
        *ttype = -4;
        s = qurtr * dmin;
        if (dmin == dn) {
          gam = dn;
          a2 = zero;
          if (Z(nn - 5) > Z(nn - 7)) return;
          b2 = Z(nn - 5) / Z(nn - 7);
          np = nn - 9;
        } else {
          np = nn - 2 * pp;
          gam = dn1;
          if (Z(np - 4) > Z(np - 2)) return;
          a2 = Z(np - 4) / Z(np - 2);
          if (Z(nn - 9) > Z(nn - 11)) return;
          b2 = Z(nn - 9) / Z(nn - 11);
          np = nn - 13;
        }
        // Approximate contribution to norm squared from I < NN-1.
        a2 = a2 + b2;
        for (i4 = np; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
          if (b2 == zero) goto L20;
          b1 = b2;
          if (Z(i4) > Z(i4 - 2)) return;
          b2 = b2 * (Z(i4) / Z(i4 - 2));
          a2 = a2 + b2;
          if (hundrd * Max(b2, b1) < a2 || cnst1 < a2) goto L20;
        }
      L20:
        a2 = cnst3 * a2;
        // Rayleigh quotient residual bound.
        if (a2 < cnst1) s = gam * (one - std::sqrt(a2)) / (one + a2);
      }
    } else if (dmin == dn2) {
      // Case 5.
      *ttype = -5;
      s = qurtr * dmin;
      // Compute contribution to norm squared from I > NN-2.
      np = nn - 2 * pp;
      b1 = Z(np - 2);
      b2 = Z(np - 6);
      gam = dn2;
      if (Z(np - 8) > b2 || Z(np - 4) > b1) return;
      a2 = (Z(np - 8) / b2) * (one + Z(np - 4) / b1);
      // Approximate contribution to norm squared from I < NN-2.
      if (n0 - i0 > 2) {
        b2 = Z(nn - 13) / Z(nn - 15);
        a2 = a2 + b2;
        for (i4 = nn - 17; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
          if (b2 == zero) goto L40;
          b1 = b2;
          if (Z(i4) > Z(i4 - 2)) return;
          b2 = b2 * (Z(i4) / Z(i4 - 2));
          a2 = a2 + b2;
          if (hundrd * Max(b2, b1) < a2 || cnst1 < a2) goto L40;
        }
      L40:
        a2 = cnst3 * a2;
      }
      if (a2 < cnst1) s = gam * (one - std::sqrt(a2)) / (one + a2);
    } else {
      // Case 6, no information to guide us.
      if (*ttype == -6) {
        *g = *g + third * (one - *g);
      } else if (*ttype == -18) {
        *g = qurtr * third;
      } else {
        *g = qurtr;
      }
      s = *g * dmin;
      *ttype = -6;
    }
  } else if (n0in == (n0 + 1)) {
    // One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
    if (dmin1 == dn1 && dmin2 == dn2) {
      // Cases 7 and 8.
      *ttype = -7;
      s = third * dmin1;
      if (Z(nn - 5) > Z(nn - 7)) return;
      b1 = Z(nn - 5) / Z(nn - 7);
      b2 = b1;
      if (b2 == zero) goto L60;
      for (i4 = 4 * n0 - 9 + pp; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
        a2 = b1;
        if (Z(i4) > Z(i4 - 2)) return;
        b1 = b1 * (Z(i4) / Z(i4 - 2));
        b2 = b2 + b1;
        if (hundrd * Max(b1, a2) < b2) goto L60;
      }
    L60:
      b2 = std::sqrt(cnst3 * b2);
      a2 = dmin1 / (one + b2 * b2);
      gap2 = half * dmin2 - a2;
      if (gap2 > zero && gap2 > b2 * a2) {
        s = Max(s, a2 * (one - cnst2 * a2 * (b2 / gap2) * b2));
      } else {
        s = Max(s, a2 * (one - cnst2 * b2));
        *ttype = -8;
      }
    } else {
      // Case 9.
      s = qurtr * dmin1;
      if (dmin1 == dn1) s = half * dmin1;
      *ttype = -9;
    }
  } else if (n0in == (n0 + 2)) {
    // Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
    // Cases 10 and 11.
    if (dmin2 == dn2 && two * Z(nn - 5) < Z(nn - 7)) {
      *ttype = -10;
      s = third * dmin2;
      if (Z(nn - 5) > Z(nn - 7)) return;
      b1 = Z(nn - 5) / Z(nn - 7);
      b2 = b1;
      if (b2 == zero) goto L80;
      for (i4 = 4 * n0 - 9 + pp; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
        if (Z(i4) > Z(i4 - 2)) return;
        b1 = b1 * (Z(i4) / Z(i4 - 2));
        b2 = b2 + b1;
        if (hundrd * b1 < b2) goto L80;
      }
    L80:
      b2 = std::sqrt(cnst3 * b2);
      a2 = dmin2 / (one + b2 * b2);
      gap2 = Z(nn - 7) + Z(nn - 9) -
             std::sqrt(Z(nn - 11)) * std::sqrt(Z(nn - 9)) - a2;
      if (gap2 > zero && gap2 > b2 * a2) {
        s = Max(s, a2 * (one - cnst2 * a2 * (b2 / gap2) * b2));
      } else {
        s = Max(s, a2 * (one - cnst2 * b2));
      }
    } else {
      s = qurtr * dmin2;
      *ttype = -11;
    }
  } else if (n0in > (n0 + 2)) {
    // Case 12, more than two eigenvalues deflated. No information.
    s = zero;
    *ttype = -12;
  }
  *tau = s;
}

// xLASQ3: checks for deflation, computes a shift (xLASQ4) and calls dqds
// (xLASQ5; xLASQ6 on failure) for the block I0:N0 of the qd array Z. In case
// of failure it changes shifts, and tries again until output is positive.
// On exit N0 is the new end of the block and PP the new ping-pong flag; the
// other in/out arguments carry the state of the iteration between calls
// (see xLASQ2).
template <typename R>
void Lasq3(blasint i0, blasint* n0, R* z, blasint* pp, R* dmin, R* sigma,
           R* desig, R* qmax, blasint* nfail, blasint* iter, blasint* ndiv,
           bool ieee, blasint* ttype, R* dmin1, R* dmin2, R* dn, R* dn1, R* dn2,
           R* g, R* tau) {
  const R cbias = R(1.50);
  const R zero = R(0);
  const R qurtr = R(0.250);
  const R half = R(0.5);
  const R one = R(1);
  const R two = R(2);
  const R hundrd = R(100);
  const Vector<R> Z(z);
  blasint ipn4, j4, n0in, nn;
  R eps, s, t, temp, tol, tol2;
  n0in = *n0;
  eps = Lamch<R>('P');
  tol = eps * hundrd;
  tol2 = tol * tol;
  // Check for deflation.
L10:
  if (*n0 < i0) return;
  if (*n0 == i0) goto L20;
  nn = 4 * *n0 + *pp;
  if (*n0 == (i0 + 1)) goto L40;
  // Check whether E(N0-1) is negligible, 1 eigenvalue.
  if (Z(nn - 5) > tol2 * (*sigma + Z(nn - 3)) &&
      Z(nn - 2 * *pp - 4) > tol2 * Z(nn - 7))
    goto L30;
L20:
  Z(4 * *n0 - 3) = Z(4 * *n0 + *pp - 3) + *sigma;
  *n0 = *n0 - 1;
  goto L10;
  // Check  whether E(N0-2) is negligible, 2 eigenvalues.
L30:
  if (Z(nn - 9) > tol2 * *sigma && Z(nn - 2 * *pp - 8) > tol2 * Z(nn - 11))
    goto L50;
L40:
  if (Z(nn - 3) > Z(nn - 7)) {
    s = Z(nn - 3);
    Z(nn - 3) = Z(nn - 7);
    Z(nn - 7) = s;
  }
  t = half * ((Z(nn - 7) - Z(nn - 3)) + Z(nn - 5));
  if (Z(nn - 5) > Z(nn - 3) * tol2 && t != zero) {
    s = Z(nn - 3) * (Z(nn - 5) / t);
    if (s <= t) {
      s = Z(nn - 3) * (Z(nn - 5) / (t * (one + std::sqrt(one + s / t))));
    } else {
      s = Z(nn - 3) * (Z(nn - 5) / (t + std::sqrt(t) * std::sqrt(t + s)));
    }
    t = Z(nn - 7) + (s + Z(nn - 5));
    Z(nn - 3) = Z(nn - 3) * (Z(nn - 7) / t);
    Z(nn - 7) = t;
  }
  Z(4 * *n0 - 7) = Z(nn - 7) + *sigma;
  Z(4 * *n0 - 3) = Z(nn - 3) + *sigma;
  *n0 = *n0 - 2;
  goto L10;
L50:
  if (*pp == 2) *pp = 0;
  // Reverse the qd-array, if warranted.
  if (*dmin <= zero || *n0 < n0in) {
    if (cbias * Z(4 * i0 + *pp - 3) < Z(4 * *n0 + *pp - 3)) {
      ipn4 = 4 * (i0 + *n0);
      for (j4 = 4 * i0; j4 <= 2 * (i0 + *n0 - 1); j4 += 4) {
        temp = Z(j4 - 3);
        Z(j4 - 3) = Z(ipn4 - j4 - 3);
        Z(ipn4 - j4 - 3) = temp;
        temp = Z(j4 - 2);
        Z(j4 - 2) = Z(ipn4 - j4 - 2);
        Z(ipn4 - j4 - 2) = temp;
        temp = Z(j4 - 1);
        Z(j4 - 1) = Z(ipn4 - j4 - 5);
        Z(ipn4 - j4 - 5) = temp;
        temp = Z(j4);
        Z(j4) = Z(ipn4 - j4 - 4);
        Z(ipn4 - j4 - 4) = temp;
      }
      if (*n0 - i0 <= 4) {
        Z(4 * *n0 + *pp - 1) = Z(4 * i0 + *pp - 1);
        Z(4 * *n0 - *pp) = Z(4 * i0 - *pp);
      }
      *dmin2 = Min(*dmin2, Z(4 * *n0 + *pp - 1));
      Z(4 * *n0 + *pp - 1) =
          Min(Z(4 * *n0 + *pp - 1), Z(4 * i0 + *pp - 1), Z(4 * i0 + *pp + 3));
      Z(4 * *n0 - *pp) =
          Min(Z(4 * *n0 - *pp), Z(4 * i0 - *pp), Z(4 * i0 - *pp + 4));
      *qmax = Max(*qmax, Z(4 * i0 + *pp - 3), Z(4 * i0 + *pp + 1));
      *dmin = -zero;
    }
  }
  // Choose a shift.
  Lasq4(i0, *n0, z, *pp, n0in, *dmin, *dmin1, *dmin2, *dn, *dn1, *dn2, tau,
        ttype, g);
  // Call dqds until DMIN > 0.
L70:
  Lasq5(i0, *n0, z, *pp, tau, *sigma, dmin, dmin1, dmin2, dn, dn1, dn2, ieee,
        eps);
  *ndiv = *ndiv + (*n0 - i0 + 2);
  *iter = *iter + 1;
  // Check status.
  if (*dmin >= zero && *dmin1 >= zero) {
    // Success.
    goto L90;
  } else if (*dmin < zero && *dmin1 > zero &&
             Z(4 * (*n0 - 1) - *pp) < tol * (*sigma + *dn1) &&
             std::abs(*dn) < tol * *sigma) {
    // Convergence hidden by negative DN.
    Z(4 * (*n0 - 1) - *pp + 2) = zero;
    *dmin = zero;
    goto L90;
  } else if (*dmin < zero) {
    // TAU too big. Select new TAU and try again.
    *nfail = *nfail + 1;
    if (*ttype < -22) {
      // Failed twice. Play it safe.
      *tau = zero;
    } else if (*dmin1 > zero) {
      // Late failure. Gives excellent shift.
      *tau = (*tau + *dmin) * (one - two * eps);
      *ttype = *ttype - 11;
    } else {
      // Early failure. Divide by 4.
      *tau = qurtr * *tau;
      *ttype = *ttype - 12;
    }
    goto L70;
  } else if (std::isnan(*dmin)) {
    // NaN.
    if (*tau == zero) {
      goto L80;
    } else {
      *tau = zero;
      goto L70;
    }
  } else {
    // Possible underflow. Play it safe.
    goto L80;
  }
  // Risk of underflow.
L80:
  Lasq6(i0, *n0, z, *pp, dmin, dmin1, dmin2, dn, dn1, dn2);
  *ndiv = *ndiv + (*n0 - i0 + 2);
  *iter = *iter + 1;
  *tau = zero;
L90:
  if (*tau < *sigma) {
    *desig = *desig + *tau;
    t = *sigma + *desig;
    *desig = *desig - (t - *sigma);
  } else {
    t = *sigma + *tau;
    *desig = *sigma - (t - *tau) + *desig;
  }
  *sigma = t;
}

// xLASQ2: computes all the eigenvalues of the symmetric positive definite
// tridiagonal matrix associated with the qd array Z (2*N-1 values q(1),
// e(1), ..., q(n), in a workspace of 4*N elements) to high relative
// accuracy, in the absence of denormalization, underflow and overflow.
//
// On exit, Z(1:N) holds the eigenvalues in decreasing order, Z(2N+1) the
// trace, Z(2N+2) the sum of the eigenvalues, Z(2N+3) the number of
// iterations, Z(2N+4) the number of divides per iteration squared and
// Z(2N+5) the percentage of shifts that failed. Returns INFO: 0; -1 for N <
// 0; -(200+i) if Z(i) is negative (XERBLA is called with 2 for those); 1 if
// a split was marked by a positive value in E; 2 if the current block of Z
// was not diagonalized after 100*N iterations (in the inner while loop),
// with Z holding a qd array that is equivalent to the input; 3 if the
// termination criterion of the outer while loop was not met (fewer than N
// blocks were found).
//
// As in LAPACK, the single-precision routine does not use the IEEE variant
// of the transform (xLASQ5) and computes Z(2N+5) with an integer division,
// HUNDRD*REAL(NFAIL/ITER); the double-precision routine asks ILAENV (ISPEC
// = 10) whether NaN arithmetic can be trusted.
template <typename R>
blasint Lasq2(blasint n, R* z) {
  const R cbias = R(1.50);
  const R zero = R(0);
  const R half = R(0.5);
  const R one = R(1);
  const R two = R(2);
  const R four = R(4);
  const R hundrd = R(100);
  const Vector<R> Z(z);
  blasint i0, i1, i4, ipn4, iter, iwhila, iwhilb, k, kmin, n0, n1, nbig, ndiv,
      nfail, pp, splt, ttype;
  bool ieee;
  R d, dee, deemin, desig, dmin, dmin1, dmin2, dn, dn1, dn2, e, emax, emin, eps,
      g, oldemn, qmax, qmin, s, safmin, sigma, t, tau, temp, tempe, tempq, tol,
      tol2, trace, zmax;
  blasint info = 0;
  eps = Lamch<R>('P');
  safmin = Lamch<R>('S');
  tol = eps * hundrd;
  tol2 = tol * tol;
  if (n < 0) {
    info = -1;
    Xerbla<R>("LASQ2", 1);
    return info;
  } else if (n == 0) {
    return info;
  } else if (n == 1) {
    // 1-by-1 case.
    if (Z(1) < zero) {
      info = -201;
      Xerbla<R>("LASQ2", 2);
    }
    return info;
  } else if (n == 2) {
    // 2-by-2 case.
    if (Z(1) < zero) {
      info = -201;
      Xerbla<R>("LASQ2", 2);
      return info;
    } else if (Z(2) < zero) {
      info = -202;
      Xerbla<R>("LASQ2", 2);
      return info;
    } else if (Z(3) < zero) {
      info = -203;
      Xerbla<R>("LASQ2", 2);
      return info;
    } else if (Z(3) > Z(1)) {
      d = Z(3);
      Z(3) = Z(1);
      Z(1) = d;
    }
    Z(5) = Z(1) + Z(2) + Z(3);
    if (Z(2) > Z(3) * tol2) {
      t = half * ((Z(1) - Z(3)) + Z(2));
      s = Z(3) * (Z(2) / t);
      if (s <= t) {
        s = Z(3) * (Z(2) / (t * (one + std::sqrt(one + s / t))));
      } else {
        s = Z(3) * (Z(2) / (t + std::sqrt(t) * std::sqrt(t + s)));
      }
      t = Z(1) + (s + Z(2));
      Z(3) = Z(3) * (Z(1) / t);
      Z(1) = t;
    }
    Z(2) = Z(3);
    Z(6) = Z(2) + Z(1);
    return info;
  }
  // Check for negative data and compute sums of q's and e's.
  Z(2 * n) = zero;
  emin = Z(2);
  qmax = zero;
  zmax = zero;
  d = zero;
  e = zero;
  for (k = 1; k <= 2 * (n - 1); k += 2) {
    if (Z(k) < zero) {
      info = -(200 + k);
      Xerbla<R>("LASQ2", 2);
      return info;
    } else if (Z(k + 1) < zero) {
      info = -(200 + k + 1);
      Xerbla<R>("LASQ2", 2);
      return info;
    }
    d = d + Z(k);
    e = e + Z(k + 1);
    qmax = Max(qmax, Z(k));
    emin = Min(emin, Z(k + 1));
    zmax = Max(qmax, zmax, Z(k + 1));
  }
  if (Z(2 * n - 1) < zero) {
    info = -(200 + 2 * n - 1);
    Xerbla<R>("LASQ2", 2);
    return info;
  }
  d = d + Z(2 * n - 1);
  qmax = Max(qmax, Z(2 * n - 1));
  zmax = Max(qmax, zmax);
  // Check for diagonality.
  if (e == zero) {
    for (k = 2; k <= n; ++k) {
      Z(k) = Z(2 * k - 1);
    }
    Lasrt('D', n, z);
    Z(2 * n - 1) = d;
    return info;
  }
  trace = d + e;
  // Check for zero data.
  if (trace == zero) {
    Z(2 * n - 1) = zero;
    return info;
  }
  // Check whether the machine is IEEE conformable.
  if constexpr (std::is_same_v<R, float>) {
    ieee = false;
  } else {
    ieee = (Ilaenv<R>(10, "LASQ2", "N", 1, 2, 3, 4) == 1);
  }
  // Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
  for (k = 2 * n; k >= 2; k -= 2) {
    Z(2 * k) = zero;
    Z(2 * k - 1) = Z(k);
    Z(2 * k - 2) = zero;
    Z(2 * k - 3) = Z(k - 1);
  }
  i0 = 1;
  n0 = n;
  // Reverse the qd-array, if warranted.
  if (cbias * Z(4 * i0 - 3) < Z(4 * n0 - 3)) {
    ipn4 = 4 * (i0 + n0);
    for (i4 = 4 * i0; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
      temp = Z(i4 - 3);
      Z(i4 - 3) = Z(ipn4 - i4 - 3);
      Z(ipn4 - i4 - 3) = temp;
      temp = Z(i4 - 1);
      Z(i4 - 1) = Z(ipn4 - i4 - 5);
      Z(ipn4 - i4 - 5) = temp;
    }
  }
  // Initial split checking via dqd and Li's test.
  pp = 0;
  for (k = 1; k <= 2; ++k) {
    d = Z(4 * n0 + pp - 3);
    for (i4 = 4 * (n0 - 1) + pp; i4 >= 4 * i0 + pp; i4 -= 4) {
      if (Z(i4 - 1) <= tol2 * d) {
        Z(i4 - 1) = -zero;
        d = Z(i4 - 3);
      } else {
        d = Z(i4 - 3) * (d / (d + Z(i4 - 1)));
      }
    }
    // dqd maps Z to ZZ plus Li's test.
    emin = Z(4 * i0 + pp + 1);
    d = Z(4 * i0 + pp - 3);
    for (i4 = 4 * i0 + pp; i4 <= 4 * (n0 - 1) + pp; i4 += 4) {
      Z(i4 - 2 * pp - 2) = d + Z(i4 - 1);
      if (Z(i4 - 1) <= tol2 * d) {
        Z(i4 - 1) = -zero;
        Z(i4 - 2 * pp - 2) = d;
        Z(i4 - 2 * pp) = zero;
        d = Z(i4 + 1);
      } else if (safmin * Z(i4 + 1) < Z(i4 - 2 * pp - 2) &&
                 safmin * Z(i4 - 2 * pp - 2) < Z(i4 + 1)) {
        temp = Z(i4 + 1) / Z(i4 - 2 * pp - 2);
        Z(i4 - 2 * pp) = Z(i4 - 1) * temp;
        d = d * temp;
      } else {
        Z(i4 - 2 * pp) = Z(i4 + 1) * (Z(i4 - 1) / Z(i4 - 2 * pp - 2));
        d = Z(i4 + 1) * (d / Z(i4 - 2 * pp - 2));
      }
      emin = Min(emin, Z(i4 - 2 * pp));
    }
    Z(4 * n0 - pp - 2) = d;
    // Now find qmax.
    qmax = Z(4 * i0 - pp - 2);
    for (i4 = 4 * i0 - pp + 2; i4 <= 4 * n0 - pp - 2; i4 += 4) {
      qmax = Max(qmax, Z(i4));
    }
    // Prepare for the next iteration on K.
    pp = 1 - pp;
  }
  // Initialise variables to pass to DLASQ3.
  ttype = 0;
  dmin1 = zero;
  dmin2 = zero;
  dn = zero;
  dn1 = zero;
  dn2 = zero;
  g = zero;
  tau = zero;
  iter = 2;
  nfail = 0;
  ndiv = 2 * (n0 - i0);
  for (iwhila = 1; iwhila <= n + 1; ++iwhila) {
    if (n0 < 1) goto L170;
    // While array unfinished do
    // E(N0) holds the value of SIGMA when submatrix in I0:N0
    // splits from the rest of the array, but is negated.
    desig = zero;
    if (n0 == n) {
      sigma = zero;
    } else {
      sigma = -Z(4 * n0 - 1);
    }
    if (sigma < zero) {
      info = 1;
      return info;
    }
    // Find last unreduced submatrix's top index I0, find QMAX and
    // EMIN. Find Gershgorin-type bound if Q's much greater than E's.
    emax = zero;
    if (n0 > i0) {
      emin = std::abs(Z(4 * n0 - 5));
    } else {
      emin = zero;
    }
    qmin = Z(4 * n0 - 3);
    qmax = qmin;
    for (i4 = 4 * n0; i4 >= 8; i4 -= 4) {
      if (Z(i4 - 5) <= zero) goto L100;
      if (qmin >= four * emax) {
        qmin = Min(qmin, Z(i4 - 3));
        emax = Max(emax, Z(i4 - 5));
      }
      qmax = Max(qmax, Z(i4 - 7) + Z(i4 - 5));
      emin = Min(emin, Z(i4 - 5));
    }
    i4 = 4;
  L100:
    i0 = i4 / 4;
    pp = 0;
    if (n0 - i0 > 1) {
      dee = Z(4 * i0 - 3);
      deemin = dee;
      kmin = i0;
      for (i4 = 4 * i0 + 1; i4 <= 4 * n0 - 3; i4 += 4) {
        dee = Z(i4) * (dee / (dee + Z(i4 - 2)));
        if (dee <= deemin) {
          deemin = dee;
          kmin = (i4 + 3) / 4;
        }
      }
      if ((kmin - i0) * 2 < n0 - kmin && deemin <= half * Z(4 * n0 - 3)) {
        ipn4 = 4 * (i0 + n0);
        pp = 2;
        for (i4 = 4 * i0; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
          temp = Z(i4 - 3);
          Z(i4 - 3) = Z(ipn4 - i4 - 3);
          Z(ipn4 - i4 - 3) = temp;
          temp = Z(i4 - 2);
          Z(i4 - 2) = Z(ipn4 - i4 - 2);
          Z(ipn4 - i4 - 2) = temp;
          temp = Z(i4 - 1);
          Z(i4 - 1) = Z(ipn4 - i4 - 5);
          Z(ipn4 - i4 - 5) = temp;
          temp = Z(i4);
          Z(i4) = Z(ipn4 - i4 - 4);
          Z(ipn4 - i4 - 4) = temp;
        }
      }
    }
    // Put -(initial shift) into DMIN.
    dmin = -Max(zero, qmin - two * std::sqrt(qmin) * std::sqrt(emax));
    // Now I0:N0 is unreduced.
    // PP = 0 for ping, PP = 1 for pong.
    // PP = 2 indicates that flipping was applied to the Z array and
    // and that the tests for deflation upon entry in DLASQ3
    // should not be performed.
    nbig = 100 * (n0 - i0 + 1);
    for (iwhilb = 1; iwhilb <= nbig; ++iwhilb) {
      if (i0 > n0) goto L150;
      // While submatrix unfinished take a good dqds step.
      Lasq3(i0, &n0, z, &pp, &dmin, &sigma, &desig, &qmax, &nfail, &iter, &ndiv,
            ieee, &ttype, &dmin1, &dmin2, &dn, &dn1, &dn2, &g, &tau);
      pp = 1 - pp;
      // When EMIN is very small check for splits.
      if (pp == 0 && n0 - i0 >= 3) {
        if (Z(4 * n0) <= tol2 * qmax || Z(4 * n0 - 1) <= tol2 * sigma) {
          splt = i0 - 1;
          qmax = Z(4 * i0 - 3);
          emin = Z(4 * i0 - 1);
          oldemn = Z(4 * i0);
          for (i4 = 4 * i0; i4 <= 4 * (n0 - 3); i4 += 4) {
            if (Z(i4) <= tol2 * Z(i4 - 3) || Z(i4 - 1) <= tol2 * sigma) {
              Z(i4 - 1) = -sigma;
              splt = i4 / 4;
              qmax = zero;
              emin = Z(i4 + 3);
              oldemn = Z(i4 + 4);
            } else {
              qmax = Max(qmax, Z(i4 + 1));
              emin = Min(emin, Z(i4 - 1));
              oldemn = Min(oldemn, Z(i4));
            }
          }
          Z(4 * n0 - 1) = emin;
          Z(4 * n0) = oldemn;
          i0 = splt + 1;
        }
      }
    }
    info = 2;
    // Maximum number of iterations exceeded, restore the shift
    // SIGMA and place the new d's and e's in a qd array.
    // This might need to be done for several blocks
    i1 = i0;
    n1 = n0;
  L145:
    tempq = Z(4 * i0 - 3);
    Z(4 * i0 - 3) = Z(4 * i0 - 3) + sigma;
    for (k = i0 + 1; k <= n0; ++k) {
      tempe = Z(4 * k - 5);
      Z(4 * k - 5) = Z(4 * k - 5) * (tempq / Z(4 * k - 7));
      tempq = Z(4 * k - 3);
      Z(4 * k - 3) = Z(4 * k - 3) + sigma + tempe - Z(4 * k - 5);
    }
    // Prepare to do this on the previous block if there is one
    if (i1 > 1) {
      n1 = i1 - 1;
      while ((i1 >= 2) && (Z(4 * i1 - 5) >= zero)) {
        i1 = i1 - 1;
      }
      sigma = -Z(4 * n1 - 1);
      goto L145;
    }
    for (k = 1; k <= n; ++k) {
      Z(2 * k - 1) = Z(4 * k - 3);
      // Only the block 1..N0 is unfinished.  The rest of the e's
      // must be essentially zero, although sometimes other data
      // has been stored in them.
      if (k < n0) {
        Z(2 * k) = Z(4 * k - 1);
      } else {
        Z(2 * k) = 0;
      }
    }
    return info;
    // end IWHILB
  L150:;
  }
  info = 3;
  return info;
  // end IWHILA
L170:
  // Move q's to the front.
  for (k = 2; k <= n; ++k) {
    Z(k) = Z(4 * k - 3);
  }
  // Sort and compute sum of eigenvalues.
  Lasrt('D', n, z);
  e = zero;
  for (k = n; k >= 1; --k) {
    e = e + Z(k);
  }
  // Store trace, sum(eigenvalues) and information on performance.
  Z(2 * n + 1) = trace;
  Z(2 * n + 2) = e;
  Z(2 * n + 3) = R(iter);
  Z(2 * n + 4) = R(ndiv) / R(n * n);
  if constexpr (std::is_same_v<R, float>) {
    // NOLINTNEXTLINE(bugprone-integer-division): as in SLASQ2
    Z(2 * n + 5) = hundrd * R(nfail / iter);
  } else {
    Z(2 * n + 5) = hundrd * nfail / R(iter);
  }
  return info;
}

// xLASQ1: computes the singular values of the real N-by-N bidiagonal matrix
// with diagonal D and off-diagonal E (N-1 elements; destroyed) to high
// relative accuracy, by squaring them into a qd array (WORK, 4*N elements)
// whose eigenvalues xLASQ2 computes. On exit, D holds the singular values
// in decreasing order. Returns INFO: 0; -1 for N < 0; or as xLASQ2: 1
// (internal error), 2 (no convergence; D and E then hold a bidiagonal
// matrix with the same singular values, which xBDSQR continues with) or 3
// (termination criterion of the outer loop not met).
template <typename R>
blasint Lasq1(blasint n, R* d, R* e, R* work) {
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<R> WORK(work);
  blasint i;
  R eps, safmin, scale, sigmn, sigmx;
  blasint info = 0;
  if (n < 0) {
    info = -1;
    Xerbla<R>("LASQ1", -info);
    return info;
  } else if (n == 0) {
    return info;
  } else if (n == 1) {
    D(1) = std::abs(D(1));
    return info;
  } else if (n == 2) {
    Las2(D(1), E(1), D(2), &sigmn, &sigmx);
    D(1) = sigmx;
    D(2) = sigmn;
    return info;
  }
  // Estimate the largest singular value.
  sigmx = zero;
  for (i = 1; i <= n - 1; ++i) {
    D(i) = std::abs(D(i));
    sigmx = Max(sigmx, std::abs(E(i)));
  }
  D(n) = std::abs(D(n));
  // Early return if SIGMX is zero (matrix is already diagonal).
  if (sigmx == zero) {
    Lasrt('D', n, d);
    return info;
  }
  for (i = 1; i <= n; ++i) {
    sigmx = Max(sigmx, D(i));
  }
  // Copy D and E into WORK (in the Z format) and scale (squaring the
  // input data makes scaling by a power of the radix pointless).
  eps = Lamch<R>('P');
  safmin = Lamch<R>('S');
  scale = std::sqrt(eps / safmin);
  Copy(n, d, 1, WORK.ptr(1), 2);
  Copy(n - 1, e, 1, WORK.ptr(2), 2);
  Lascl('G', 0, 0, sigmx, scale, 2 * n - 1, 1, work, 2 * n - 1);
  // Compute the q's and e's.
  for (i = 1; i <= 2 * n - 1; ++i) {
    WORK(i) = WORK(i) * WORK(i);
  }
  WORK(2 * n) = zero;
  info = Lasq2(n, work);
  if (info == 0) {
    for (i = 1; i <= n; ++i) {
      D(i) = std::sqrt(WORK(i));
    }
    Lascl('G', 0, 0, scale, sigmx, n, 1, d, n);
  } else if (info == 2) {
    // Maximum number of iterations exceeded.  Move data from WORK
    // into D and E so the calling subroutine can try to finish
    for (i = 1; i <= n; ++i) {
      D(i) = std::sqrt(WORK(2 * i - 1));
      E(i) = std::sqrt(WORK(2 * i));
    }
    Lascl('G', 0, 0, scale, sigmx, n, 1, d, n);
    Lascl('G', 0, 0, scale, sigmx, n, 1, e, n);
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(R)                                          \
  template void Lasq6<R>(blasint, blasint, R*, blasint, R*, R*, R*, R*, R*,    \
                         R*);                                                  \
  template void Lasq5<R>(blasint, blasint, R*, blasint, R*, R, R*, R*, R*, R*, \
                         R*, R*, bool, R);                                     \
  template void Lasq4<R>(blasint, blasint, const R*, blasint, blasint, R, R,   \
                         R, R, R, R, R*, blasint*, R*);                        \
  template void Lasq3<R>(blasint, blasint*, R*, blasint*, R*, R*, R*, R*,      \
                         blasint*, blasint*, blasint*, bool, blasint*, R*, R*, \
                         R*, R*, R*, R*, R*);                                  \
  template blasint Lasq2<R>(blasint, R*);                                      \
  template blasint Lasq1<R>(blasint, R*, R*, R*);
HWYBLAS_LAPACK_INSTANTIATE(float)
HWYBLAS_LAPACK_INSTANTIATE(double)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

extern "C" {

// The Fortran LOGICAL argument IEEE is a hwyblas_int (1 or 0), as in lsame_.
#define HWYBLAS_LAPACK_LASQ(prefix, R)                                         \
  HWYBLAS_API void prefix##lasq1_(const hwyblas_int* n, R* d, R* e, R* work,   \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Lasq1(*n, d, e, work);                            \
  }                                                                            \
  HWYBLAS_API void prefix##lasq2_(const hwyblas_int* n, R* z,                  \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Lasq2(*n, z);                                     \
  }                                                                            \
  HWYBLAS_API void prefix##lasq3_(                                             \
      const hwyblas_int* i0, hwyblas_int* n0, R* z, hwyblas_int* pp, R* dmin,  \
      R* sigma, R* desig, R* qmax, hwyblas_int* nfail, hwyblas_int* iter,      \
      hwyblas_int* ndiv, const hwyblas_int* ieee, hwyblas_int* ttype,          \
      R* dmin1, R* dmin2, R* dn, R* dn1, R* dn2, R* g, R* tau) {               \
    hwyblas::lapack::Lasq3(*i0, n0, z, pp, dmin, sigma, desig, qmax, nfail,    \
                           iter, ndiv, *ieee != 0, ttype, dmin1, dmin2, dn,    \
                           dn1, dn2, g, tau);                                  \
  }                                                                            \
  HWYBLAS_API void prefix##lasq4_(                                             \
      const hwyblas_int* i0, const hwyblas_int* n0, const R* z,                \
      const hwyblas_int* pp, const hwyblas_int* n0in, const R* dmin,           \
      const R* dmin1, const R* dmin2, const R* dn, const R* dn1, const R* dn2, \
      R* tau, hwyblas_int* ttype, R* g) {                                      \
    hwyblas::lapack::Lasq4(*i0, *n0, z, *pp, *n0in, *dmin, *dmin1, *dmin2,     \
                           *dn, *dn1, *dn2, tau, ttype, g);                    \
  }                                                                            \
  HWYBLAS_API void prefix##lasq5_(                                             \
      const hwyblas_int* i0, const hwyblas_int* n0, R* z,                      \
      const hwyblas_int* pp, R* tau, const R* sigma, R* dmin, R* dmin1,        \
      R* dmin2, R* dn, R* dnm1, R* dnm2, const hwyblas_int* ieee,              \
      const R* eps) {                                                          \
    hwyblas::lapack::Lasq5(*i0, *n0, z, *pp, tau, *sigma, dmin, dmin1, dmin2,  \
                           dn, dnm1, dnm2, *ieee != 0, *eps);                  \
  }                                                                            \
  HWYBLAS_API void prefix##lasq6_(const hwyblas_int* i0,                       \
                                  const hwyblas_int* n0, R* z,                 \
                                  const hwyblas_int* pp, R* dmin, R* dmin1,    \
                                  R* dmin2, R* dn, R* dnm1, R* dnm2) {         \
    hwyblas::lapack::Lasq6(*i0, *n0, z, *pp, dmin, dmin1, dmin2, dn, dnm1,     \
                           dnm2);                                              \
  }

HWYBLAS_LAPACK_LASQ(s, float)
HWYBLAS_LAPACK_LASQ(d, double)

#undef HWYBLAS_LAPACK_LASQ

}  // extern "C"
