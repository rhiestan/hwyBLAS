// The eigenvalues and eigenvectors of a symmetric tridiagonal matrix by
// divide and conquer: xSTEDC (driver), xLAED0 (the tree of subproblems:
// splits the matrix into subproblems of at most SMLSIZ+1 rows, solves them
// with xSTEQR and merges them bottom-up), xLAED1 (merge for COMPZ = 'I':
// deflation with xLAED2, the secular equation and the eigenvectors with
// xLAED3), xLAED7 (merge for COMPZ = 'V' and 'N', which keeps the merged
// eigenvectors in factored form: deflation with xLAED8, the secular
// equation with xLAED9, the rows of the factored eigenvectors needed for the
// next merge from xLAEDA), and xLAED4 / xLAED5, which find one root of the
// secular equation (xLAED6 in bdsdc.cc for the step of the middle-way
// method). The xLAED* routines are real (R); for complex T, xSTEDC,
// xLAED0, xLAED7 and xLAED8 have complex overloads (ZSTEDC, ZLAED0, ZLAED7,
// ZLAED8: the eigenvectors of the tridiagonal matrix are real and are
// multiplied into the complex Q).
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

// xLAED5: the I-th eigenvalue DLAM (I = 1 or 2) of the 2-by-2 rank-one
// modification diag(D) + RHO * Z * Z**T (D(1) < D(2), RHO > 0, norm(Z) =
// 1), and in DELTA its normalized eigenvector.
template <typename R>
void Laed5(blasint i, R* d, R* z, R* delta, R rho, R* dlam) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R four = R(4);
  const Vector<R> D(d);
  const Vector<R> DELTA(delta);
  const Vector<R> Z(z);
  R b, c, del, tau, temp, w;
  del = D(2) - D(1);
  if (i == 1) {
    w = one + two * rho * (Z(2) * Z(2) - Z(1) * Z(1)) / del;
    if (w > zero) {
      b = del + rho * (Z(1) * Z(1) + Z(2) * Z(2));
      c = rho * Z(1) * Z(1) * del;
      // B > ZERO, always
      tau = two * c / (b + std::sqrt(std::abs(b * b - four * c)));
      *dlam = D(1) + tau;
      DELTA(1) = -Z(1) / tau;
      DELTA(2) = Z(2) / (del - tau);
    } else {
      b = -del + rho * (Z(1) * Z(1) + Z(2) * Z(2));
      c = rho * Z(2) * Z(2) * del;
      if (b > zero) {
        tau = -two * c / (b + std::sqrt(b * b + four * c));
      } else {
        tau = (b - std::sqrt(b * b + four * c)) / two;
      }
      *dlam = D(2) + tau;
      DELTA(1) = -Z(1) / (del + tau);
      DELTA(2) = -Z(2) / tau;
    }
    temp = std::sqrt(DELTA(1) * DELTA(1) + DELTA(2) * DELTA(2));
    DELTA(1) = DELTA(1) / temp;
    DELTA(2) = DELTA(2) / temp;
  } else {
    // Now I=2
    b = -del + rho * (Z(1) * Z(1) + Z(2) * Z(2));
    c = rho * Z(2) * Z(2) * del;
    if (b > zero) {
      tau = (b + std::sqrt(b * b + four * c)) / two;
    } else {
      tau = two * c / (-b + std::sqrt(b * b + four * c));
    }
    *dlam = D(2) + tau;
    DELTA(1) = -Z(1) / (del + tau);
    DELTA(2) = -Z(2) / tau;
    temp = std::sqrt(DELTA(1) * DELTA(1) + DELTA(2) * DELTA(2));
    DELTA(1) = DELTA(1) / temp;
    DELTA(2) = DELTA(2) / temp;
  }
}

// xLAED4: the I-th eigenvalue DLAM of the rank-one modification
// diag(D) + RHO * Z * Z**T (D in ascending order, RHO > 0, norm(Z) = 1) as
// a root of the secular equation 1 + RHO * sum(Z(j)**2 / (D(j) - x)) = 0,
// and DELTA(j) = D(j) - DLAM for the eigenvector. Returns INFO = 1 if the
// iteration did not converge.
template <typename R>
blasint Laed4(blasint n, blasint i, R* d, R* z, R* delta, R rho, R* dlam) {
  const blasint maxit = 30;
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R three = R(3);
  const R four = R(4);
  const R eight = R(8);
  const R ten = R(10);
  const Vector<R> D(d);
  const Vector<R> DELTA(delta);
  const Vector<R> Z(z);
  R zz[3];
  const Vector<R> ZZ(zz);
  blasint ii, iim1, iip1, ip1, iter, j, niter;
  bool orgati, swtch, swtch3;
  R a, b, c, del, dltlb, dltub, dphi, dpsi, dw, eps, erretm, eta, midpt, phi,
      prew, psi, rhoinv, tau, temp, temp1, w;
  blasint info = 0;
  if (n == 1) {
    // Presumably, I=1 upon entry
    *dlam = D(1) + rho * Z(1) * Z(1);
    DELTA(1) = one;
    return info;
  }
  if (n == 2) {
    Laed5(i, d, z, delta, rho, dlam);
    return info;
  }
  // Compute machine epsilon
  eps = Lamch<R>('E');
  rhoinv = one / rho;
  // The case I = N
  if (i == n) {
    // Initialize some basic variables
    ii = n - 1;
    niter = 1;
    // Calculate initial guess
    midpt = rho / two;
    // If ||Z||_2 is not one, then TEMP should be set to
    // RHO * ||Z||_2^2 / TWO
    for (j = 1; j <= n; ++j) {
      DELTA(j) = (D(j) - D(i)) - midpt;
    }
    psi = zero;
    for (j = 1; j <= n - 2; ++j) {
      psi = psi + Z(j) * Z(j) / DELTA(j);
    }
    c = rhoinv + psi;
    w = c + Z(ii) * Z(ii) / DELTA(ii) + Z(n) * Z(n) / DELTA(n);
    if (w <= zero) {
      temp = Z(n - 1) * Z(n - 1) / (D(n) - D(n - 1) + rho) + Z(n) * Z(n) / rho;
      if (c <= temp) {
        tau = rho;
      } else {
        del = D(n) - D(n - 1);
        a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
        b = Z(n) * Z(n) * del;
        if (a < zero) {
          tau = two * b / (std::sqrt(a * a + four * b * c) - a);
        } else {
          tau = (a + std::sqrt(a * a + four * b * c)) / (two * c);
        }
      }
      // It can be proved that
      // D(N)+RHO/2 <= LAMBDA(N) < D(N)+TAU <= D(N)+RHO
      dltlb = midpt;
      dltub = rho;
    } else {
      del = D(n) - D(n - 1);
      a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
      b = Z(n) * Z(n) * del;
      if (a < zero) {
        tau = two * b / (std::sqrt(a * a + four * b * c) - a);
      } else {
        tau = (a + std::sqrt(a * a + four * b * c)) / (two * c);
      }
      // It can be proved that
      // D(N) < D(N)+TAU < LAMBDA(N) < D(N)+RHO/2
      dltlb = zero;
      dltub = midpt;
    }
    for (j = 1; j <= n; ++j) {
      DELTA(j) = (D(j) - D(i)) - tau;
    }
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= ii; ++j) {
      temp = Z(j) / DELTA(j);
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    temp = Z(n) / DELTA(n);
    phi = Z(n) * temp;
    dphi = temp * temp;
    erretm = eight * (-phi - psi) + erretm - phi + rhoinv +
             std::abs(tau) * (dpsi + dphi);
    w = rhoinv + phi + psi;
    // Test for convergence
    if (std::abs(w) <= eps * erretm) {
      *dlam = D(i) + tau;
      goto L250;
    }
    if (w <= zero) {
      dltlb = Max(dltlb, tau);
    } else {
      dltub = Min(dltub, tau);
    }
    // Calculate the new step
    niter = niter + 1;
    c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
    a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
    b = DELTA(n - 1) * DELTA(n) * w;
    if (c < zero) c = std::abs(c);
    if (c == zero) {
      // ETA = B/A
      // ETA = RHO - TAU
      // ETA = DLTUB - TAU
      // Update proposed by Li, Ren-Cang:
      eta = -w / (dpsi + dphi);
    } else if (a >= zero) {
      eta = (a + std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
    } else {
      eta = two * b / (a - std::sqrt(std::abs(a * a - four * b * c)));
    }
    // Note, eta should be positive if w is negative, and
    // eta should be negative otherwise. However,
    // if for some reason caused by roundoff, eta*w > 0,
    // we simply use one Newton step instead. This way
    // will guarantee eta*w < 0.
    if (w * eta > zero) eta = -w / (dpsi + dphi);
    temp = tau + eta;
    if (temp > dltub || temp < dltlb) {
      if (w < zero) {
        eta = (dltub - tau) / two;
      } else {
        eta = (dltlb - tau) / two;
      }
    }
    for (j = 1; j <= n; ++j) {
      DELTA(j) = DELTA(j) - eta;
    }
    tau = tau + eta;
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= ii; ++j) {
      temp = Z(j) / DELTA(j);
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    temp = Z(n) / DELTA(n);
    phi = Z(n) * temp;
    dphi = temp * temp;
    erretm = eight * (-phi - psi) + erretm - phi + rhoinv +
             std::abs(tau) * (dpsi + dphi);
    w = rhoinv + phi + psi;
    // Main loop to update the values of the array   DELTA
    iter = niter + 1;
    for (niter = iter; niter <= maxit; ++niter) {
      // Test for convergence
      if (std::abs(w) <= eps * erretm) {
        *dlam = D(i) + tau;
        goto L250;
      }
      if (w <= zero) {
        dltlb = Max(dltlb, tau);
      } else {
        dltub = Min(dltub, tau);
      }
      // Calculate the new step
      c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
      a = (DELTA(n - 1) + DELTA(n)) * w -
          DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
      b = DELTA(n - 1) * DELTA(n) * w;
      if (a >= zero) {
        eta = (a + std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
      } else {
        eta = two * b / (a - std::sqrt(std::abs(a * a - four * b * c)));
      }
      // Note, eta should be positive if w is negative, and
      // eta should be negative otherwise. However,
      // if for some reason caused by roundoff, eta*w > 0,
      // we simply use one Newton step instead. This way
      // will guarantee eta*w < 0.
      if (w * eta > zero) eta = -w / (dpsi + dphi);
      temp = tau + eta;
      if (temp > dltub || temp < dltlb) {
        if (w < zero) {
          eta = (dltub - tau) / two;
        } else {
          eta = (dltlb - tau) / two;
        }
      }
      for (j = 1; j <= n; ++j) {
        DELTA(j) = DELTA(j) - eta;
      }
      tau = tau + eta;
      // Evaluate PSI and the derivative DPSI
      dpsi = zero;
      psi = zero;
      erretm = zero;
      for (j = 1; j <= ii; ++j) {
        temp = Z(j) / DELTA(j);
        psi = psi + Z(j) * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
      }
      erretm = std::abs(erretm);
      // Evaluate PHI and the derivative DPHI
      temp = Z(n) / DELTA(n);
      phi = Z(n) * temp;
      dphi = temp * temp;
      erretm = eight * (-phi - psi) + erretm - phi + rhoinv +
               std::abs(tau) * (dpsi + dphi);
      w = rhoinv + phi + psi;
    }
    // Return with INFO = 1, NITER = MAXIT and not converged
    info = 1;
    *dlam = D(i) + tau;
    goto L250;
    // End for the case I = N
  } else {
    // The case for I < N
    niter = 1;
    ip1 = i + 1;
    // Calculate initial guess
    del = D(ip1) - D(i);
    midpt = del / two;
    for (j = 1; j <= n; ++j) {
      DELTA(j) = (D(j) - D(i)) - midpt;
    }
    psi = zero;
    for (j = 1; j <= i - 1; ++j) {
      psi = psi + Z(j) * Z(j) / DELTA(j);
    }
    phi = zero;
    for (j = n; j >= i + 2; --j) {
      phi = phi + Z(j) * Z(j) / DELTA(j);
    }
    c = rhoinv + psi + phi;
    w = c + Z(i) * Z(i) / DELTA(i) + Z(ip1) * Z(ip1) / DELTA(ip1);
    if (w > zero) {
      // d(i)< the ith eigenvalue < (d(i)+d(i+1))/2
      // We choose d(i) as origin.
      orgati = true;
      a = c * del + Z(i) * Z(i) + Z(ip1) * Z(ip1);
      b = Z(i) * Z(i) * del;
      if (a > zero) {
        tau = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
      } else {
        tau = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
      }
      dltlb = zero;
      dltub = midpt;
    } else {
      // (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
      // We choose d(i+1) as origin.
      orgati = false;
      a = c * del - Z(i) * Z(i) - Z(ip1) * Z(ip1);
      b = Z(ip1) * Z(ip1) * del;
      if (a < zero) {
        tau = two * b / (a - std::sqrt(std::abs(a * a + four * b * c)));
      } else {
        tau = -(a + std::sqrt(std::abs(a * a + four * b * c))) / (two * c);
      }
      dltlb = -midpt;
      dltub = zero;
    }
    if (orgati) {
      for (j = 1; j <= n; ++j) {
        DELTA(j) = (D(j) - D(i)) - tau;
      }
    } else {
      for (j = 1; j <= n; ++j) {
        DELTA(j) = (D(j) - D(ip1)) - tau;
      }
    }
    if (orgati) {
      ii = i;
    } else {
      ii = i + 1;
    }
    iim1 = ii - 1;
    iip1 = ii + 1;
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= iim1; ++j) {
      temp = Z(j) / DELTA(j);
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    dphi = zero;
    phi = zero;
    for (j = n; j >= iip1; --j) {
      temp = Z(j) / DELTA(j);
      phi = phi + Z(j) * temp;
      dphi = dphi + temp * temp;
      erretm = erretm + phi;
    }
    w = rhoinv + phi + psi;
    // W is the value of the secular function with
    // its ii-th element removed.
    swtch3 = false;
    if (orgati) {
      if (w < zero) swtch3 = true;
    } else {
      if (w > zero) swtch3 = true;
    }
    if (ii == 1 || ii == n) swtch3 = false;
    temp = Z(ii) / DELTA(ii);
    dw = dpsi + dphi + temp * temp;
    temp = Z(ii) * temp;
    w = w + temp;
    erretm = eight * (phi - psi) + erretm + two * rhoinv +
             three * std::abs(temp) + std::abs(tau) * dw;
    // Test for convergence
    if (std::abs(w) <= eps * erretm) {
      if (orgati) {
        *dlam = D(i) + tau;
      } else {
        *dlam = D(ip1) + tau;
      }
      goto L250;
    }
    if (w <= zero) {
      dltlb = Max(dltlb, tau);
    } else {
      dltub = Min(dltub, tau);
    }
    // Calculate the new step
    niter = niter + 1;
    if (!swtch3) {
      if (orgati) {
        c = w - DELTA(ip1) * dw -
            (D(i) - D(ip1)) * ((Z(i) / DELTA(i)) * (Z(i) / DELTA(i)));
      } else {
        c = w - DELTA(i) * dw -
            (D(ip1) - D(i)) * ((Z(ip1) / DELTA(ip1)) * (Z(ip1) / DELTA(ip1)));
      }
      a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
      b = DELTA(i) * DELTA(ip1) * w;
      if (c == zero) {
        if (a == zero) {
          if (orgati) {
            a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
          } else {
            a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
          }
        }
        eta = b / a;
      } else if (a <= zero) {
        eta = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
      } else {
        eta = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
      }
    } else {
      // Interpolation using THREE most relevant poles
      temp = rhoinv + psi + phi;
      if (orgati) {
        temp1 = Z(iim1) / DELTA(iim1);
        temp1 = temp1 * temp1;
        c = temp - DELTA(iip1) * (dpsi + dphi) - (D(iim1) - D(iip1)) * temp1;
        ZZ(1) = Z(iim1) * Z(iim1);
        ZZ(3) = DELTA(iip1) * DELTA(iip1) * ((dpsi - temp1) + dphi);
      } else {
        temp1 = Z(iip1) / DELTA(iip1);
        temp1 = temp1 * temp1;
        c = temp - DELTA(iim1) * (dpsi + dphi) - (D(iip1) - D(iim1)) * temp1;
        ZZ(1) = DELTA(iim1) * DELTA(iim1) * (dpsi + (dphi - temp1));
        ZZ(3) = Z(iip1) * Z(iip1);
      }
      ZZ(2) = Z(ii) * Z(ii);
      info = Laed6(niter, orgati, c, DELTA.ptr(iim1), zz, w, &eta);
      if (info != 0) goto L250;
    }
    // Note, eta should be positive if w is negative, and
    // eta should be negative otherwise. However,
    // if for some reason caused by roundoff, eta*w > 0,
    // we simply use one Newton step instead. This way
    // will guarantee eta*w < 0.
    if (w * eta >= zero) eta = -w / dw;
    temp = tau + eta;
    if (temp > dltub || temp < dltlb) {
      if (w < zero) {
        eta = (dltub - tau) / two;
      } else {
        eta = (dltlb - tau) / two;
      }
    }
    prew = w;
    for (j = 1; j <= n; ++j) {
      DELTA(j) = DELTA(j) - eta;
    }
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= iim1; ++j) {
      temp = Z(j) / DELTA(j);
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    dphi = zero;
    phi = zero;
    for (j = n; j >= iip1; --j) {
      temp = Z(j) / DELTA(j);
      phi = phi + Z(j) * temp;
      dphi = dphi + temp * temp;
      erretm = erretm + phi;
    }
    temp = Z(ii) / DELTA(ii);
    dw = dpsi + dphi + temp * temp;
    temp = Z(ii) * temp;
    w = rhoinv + phi + psi + temp;
    erretm = eight * (phi - psi) + erretm + two * rhoinv +
             three * std::abs(temp) + std::abs(tau + eta) * dw;
    swtch = false;
    if (orgati) {
      if (-w > std::abs(prew) / ten) swtch = true;
    } else {
      if (w > std::abs(prew) / ten) swtch = true;
    }
    tau = tau + eta;
    // Main loop to update the values of the array   DELTA
    iter = niter + 1;
    for (niter = iter; niter <= maxit; ++niter) {
      // Test for convergence
      if (std::abs(w) <= eps * erretm) {
        if (orgati) {
          *dlam = D(i) + tau;
        } else {
          *dlam = D(ip1) + tau;
        }
        goto L250;
      }
      if (w <= zero) {
        dltlb = Max(dltlb, tau);
      } else {
        dltub = Min(dltub, tau);
      }
      // Calculate the new step
      if (!swtch3) {
        if (!swtch) {
          if (orgati) {
            c = w - DELTA(ip1) * dw -
                (D(i) - D(ip1)) * ((Z(i) / DELTA(i)) * (Z(i) / DELTA(i)));
          } else {
            c = w - DELTA(i) * dw -
                (D(ip1) - D(i)) *
                    ((Z(ip1) / DELTA(ip1)) * (Z(ip1) / DELTA(ip1)));
          }
        } else {
          temp = Z(ii) / DELTA(ii);
          if (orgati) {
            dpsi = dpsi + temp * temp;
          } else {
            dphi = dphi + temp * temp;
          }
          c = w - DELTA(i) * dpsi - DELTA(ip1) * dphi;
        }
        a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
        b = DELTA(i) * DELTA(ip1) * w;
        if (c == zero) {
          if (a == zero) {
            if (!swtch) {
              if (orgati) {
                a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
              } else {
                a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
              }
            } else {
              a = DELTA(i) * DELTA(i) * dpsi + DELTA(ip1) * DELTA(ip1) * dphi;
            }
          }
          eta = b / a;
        } else if (a <= zero) {
          eta = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
        } else {
          eta = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
        }
      } else {
        // Interpolation using THREE most relevant poles
        temp = rhoinv + psi + phi;
        if (swtch) {
          c = temp - DELTA(iim1) * dpsi - DELTA(iip1) * dphi;
          ZZ(1) = DELTA(iim1) * DELTA(iim1) * dpsi;
          ZZ(3) = DELTA(iip1) * DELTA(iip1) * dphi;
        } else {
          if (orgati) {
            temp1 = Z(iim1) / DELTA(iim1);
            temp1 = temp1 * temp1;
            c = temp - DELTA(iip1) * (dpsi + dphi) -
                (D(iim1) - D(iip1)) * temp1;
            ZZ(1) = Z(iim1) * Z(iim1);
            ZZ(3) = DELTA(iip1) * DELTA(iip1) * ((dpsi - temp1) + dphi);
          } else {
            temp1 = Z(iip1) / DELTA(iip1);
            temp1 = temp1 * temp1;
            c = temp - DELTA(iim1) * (dpsi + dphi) -
                (D(iip1) - D(iim1)) * temp1;
            ZZ(1) = DELTA(iim1) * DELTA(iim1) * (dpsi + (dphi - temp1));
            ZZ(3) = Z(iip1) * Z(iip1);
          }
        }
        info = Laed6(niter, orgati, c, DELTA.ptr(iim1), zz, w, &eta);
        if (info != 0) goto L250;
      }
      // Note, eta should be positive if w is negative, and
      // eta should be negative otherwise. However,
      // if for some reason caused by roundoff, eta*w > 0,
      // we simply use one Newton step instead. This way
      // will guarantee eta*w < 0.
      if (w * eta >= zero) eta = -w / dw;
      temp = tau + eta;
      if (temp > dltub || temp < dltlb) {
        if (w < zero) {
          eta = (dltub - tau) / two;
        } else {
          eta = (dltlb - tau) / two;
        }
      }
      for (j = 1; j <= n; ++j) {
        DELTA(j) = DELTA(j) - eta;
      }
      tau = tau + eta;
      prew = w;
      // Evaluate PSI and the derivative DPSI
      dpsi = zero;
      psi = zero;
      erretm = zero;
      for (j = 1; j <= iim1; ++j) {
        temp = Z(j) / DELTA(j);
        psi = psi + Z(j) * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
      }
      erretm = std::abs(erretm);
      // Evaluate PHI and the derivative DPHI
      dphi = zero;
      phi = zero;
      for (j = n; j >= iip1; --j) {
        temp = Z(j) / DELTA(j);
        phi = phi + Z(j) * temp;
        dphi = dphi + temp * temp;
        erretm = erretm + phi;
      }
      temp = Z(ii) / DELTA(ii);
      dw = dpsi + dphi + temp * temp;
      temp = Z(ii) * temp;
      w = rhoinv + phi + psi + temp;
      erretm = eight * (phi - psi) + erretm + two * rhoinv +
               three * std::abs(temp) + std::abs(tau) * dw;
      if (w * prew > zero && std::abs(w) > std::abs(prew) / ten) swtch = !swtch;
    }
    // Return with INFO = 1, NITER = MAXIT and not converged
    info = 1;
    if (orgati) {
      *dlam = D(i) + tau;
    } else {
      *dlam = D(ip1) + tau;
    }
  }
L250:
  return info;
}

// xLAED3: solves the secular equation for the K non-deflated eigenvalues
// of the merge (xLAED4) and computes the eigenvectors of the rank-one
// modification and the merged eigenvectors Q = Q2 * those (COMPZ = 'I').
template <typename R>
blasint Laed3(blasint k, blasint n, blasint n1, R* d, R* q, blasint ldq, R rho,
              R* dlambda, R* q2, blasint* indx, blasint* ctot, R* w, R* s) {
  const R one = R(1);
  const R zero = R(0);
  const Vector<blasint> CTOT(ctot);
  const Vector<R> D(d);
  const Vector<R> DLAMBDA(dlambda);
  const Vector<blasint> INDX(indx);
  const Matrix<R> Q(q, ldq);
  const Vector<R> Q2(q2);
  const Vector<R> S(s);
  const Vector<R> W(w);
  blasint i, ii, iq2, j, n12, n2, n23;
  R temp;
  blasint info = 0;
  if (k < 0) {
    info = -1;
  } else if (n < k) {
    info = -2;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<R>("LAED3", -info);
    return info;
  }
  // Quick return if possible
  if (k == 0) return info;
  for (j = 1; j <= k; ++j) {
    info = Laed4(k, j, dlambda, w, Q.ptr(1, j), rho, D.ptr(j));
    // If the zero finder fails, the computation is terminated.
    if (info != 0) goto L120;
  }
  if (k == 1) goto L110;
  if (k == 2) {
    for (j = 1; j <= k; ++j) {
      W(1) = Q(1, j);
      W(2) = Q(2, j);
      ii = INDX(1);
      Q(1, j) = W(ii);
      ii = INDX(2);
      Q(2, j) = W(ii);
    }
    goto L110;
  }
  // Compute updated W.
  Copy(k, w, 1, s, 1);
  // Initialize W(I) = Q(I,I)
  Copy(k, q, ldq + 1, w, 1);
  for (j = 1; j <= k; ++j) {
    for (i = 1; i <= j - 1; ++i) {
      W(i) = W(i) * (Q(i, j) / (DLAMBDA(i) - DLAMBDA(j)));
    }
    for (i = j + 1; i <= k; ++i) {
      W(i) = W(i) * (Q(i, j) / (DLAMBDA(i) - DLAMBDA(j)));
    }
  }
  for (i = 1; i <= k; ++i) {
    W(i) = std::copysign(std::sqrt(-W(i)), S(i));
  }
  // Compute eigenvectors of the modified rank-1 modification.
  for (j = 1; j <= k; ++j) {
    for (i = 1; i <= k; ++i) {
      S(i) = W(i) / Q(i, j);
    }
    temp = Nrm2(k, s, 1);
    for (i = 1; i <= k; ++i) {
      ii = INDX(i);
      Q(i, j) = S(ii) / temp;
    }
  }
  // Compute the updated eigenvectors.
L110:
  n2 = n - n1;
  n12 = CTOT(1) + CTOT(2);
  n23 = CTOT(2) + CTOT(3);
  Lacpy('A', n23, k, Q.ptr(CTOT(1) + 1, 1), ldq, s, n23);
  iq2 = n1 * n12 + 1;
  if (n23 != 0) {
    Gemm('N', 'N', n2, k, n23, one, Q2.ptr(iq2), n2, s, n23, zero,
         Q.ptr(n1 + 1, 1), ldq);
  } else {
    Laset('A', n2, k, zero, zero, Q.ptr(n1 + 1, 1), ldq);
  }
  Lacpy('A', n12, k, q, ldq, s, n12);
  if (n12 != 0) {
    Gemm('N', 'N', n1, k, n12, one, q2, n1, s, n12, zero, q, ldq);
  } else {
    Laset('A', n1, k, zero, zero, Q.ptr(1, 1), ldq);
  }
L120:
  return info;
}

// xLAED2: the deflation step of xLAED1: merges the two sets of eigenvalues
// and deflates those with a small Z component or close to another one
// (rotating the eigenvectors). Returns the number K of non-deflated
// eigenvalues; RHO is modified to the value that xLAED3 needs.
template <typename R>
blasint Laed2(blasint* k, blasint n, blasint n1, R* d, R* q, blasint ldq,
              blasint* indxq, R* rho, R* z, R* dlambda, R* w, R* q2,
              blasint* indx, blasint* indxc, blasint* indxp, blasint* coltyp) {
  const R mone = -R(1);
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R eight = R(8);
  const Vector<blasint> COLTYP(coltyp);
  blasint ctot[4];
  const Vector<blasint> CTOT(ctot);
  const Vector<R> D(d);
  const Vector<R> DLAMBDA(dlambda);
  const Vector<blasint> INDX(indx);
  const Vector<blasint> INDXC(indxc);
  const Vector<blasint> INDXP(indxp);
  const Vector<blasint> INDXQ(indxq);
  blasint psm[4];
  const Vector<blasint> PSM(psm);
  const Matrix<R> Q(q, ldq);
  const Vector<R> Q2(q2);
  const Vector<R> W(w);
  const Vector<R> Z(z);
  blasint ct, i, imax, iq1, iq2, j, jmax, js, k2, n1p1, n2, nj;
  // Set by the first loop below: not all components of Z are deflated (see
  // the test of Z(IMAX) above).
  blasint pj = 0;
  R c, eps, s, t, tau, tol;
  blasint info = 0;
  if (n < 0) {
    info = -2;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -6;
  } else if (Min<blasint>(1, (n / 2)) > n1 || (n / 2) < n1) {
    info = -3;
  }
  if (info != 0) {
    Xerbla<R>("LAED2", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  n2 = n - n1;
  n1p1 = n1 + 1;
  if (*rho < zero) {
    Rscal(n2, mone, Z.ptr(n1p1), 1);
  }
  // Normalize z so that norm(z) = 1.  Since z is the concatenation of
  // two normalized vectors, norm2(z) = sqrt(2).
  t = one / std::sqrt(two);
  Rscal(n, t, z, 1);
  // RHO = ABS( norm(z)**2 * RHO )
  *rho = std::abs(two * *rho);
  // Sort the eigenvalues into increasing order
  for (i = n1p1; i <= n; ++i) {
    INDXQ(i) = INDXQ(i) + n1;
  }
  // re-integrate the deflated parts from the last pass
  for (i = 1; i <= n; ++i) {
    DLAMBDA(i) = D(INDXQ(i));
  }
  Lamrg(n1, n2, dlambda, 1, 1, indxc);
  for (i = 1; i <= n; ++i) {
    INDX(i) = INDXQ(INDXC(i));
  }
  // Calculate the allowable deflation tolerance
  imax = Iamax(n, z, 1);
  jmax = Iamax(n, d, 1);
  eps = Lamch<R>('E');
  tol = eight * eps * Max(std::abs(D(jmax)), std::abs(Z(imax)));
  // If the rank-1 modifier is small enough, no more needs to be done
  // except to reorganize Q so that its columns correspond with the
  // elements in D.
  if (*rho * std::abs(Z(imax)) <= tol) {
    *k = 0;
    iq2 = 1;
    for (j = 1; j <= n; ++j) {
      i = INDX(j);
      Copy(n, Q.ptr(1, i), 1, Q2.ptr(iq2), 1);
      DLAMBDA(j) = D(i);
      iq2 = iq2 + n;
    }
    Lacpy('A', n, n, q2, n, q, ldq);
    Copy(n, dlambda, 1, d, 1);
    goto L190;
  }
  // If there are multiple eigenvalues then the problem deflates.  Here
  // the number of equal eigenvalues are found.  As each equal
  // eigenvalue is found, an elementary reflector is computed to rotate
  // the corresponding eigensubspace so that the corresponding
  // components of Z are zero in this new basis.
  for (i = 1; i <= n1; ++i) {
    COLTYP(i) = 1;
  }
  for (i = n1p1; i <= n; ++i) {
    COLTYP(i) = 3;
  }
  *k = 0;
  k2 = n + 1;
  for (j = 1; j <= n; ++j) {
    nj = INDX(j);
    if (*rho * std::abs(Z(nj)) <= tol) {
      // Deflate due to small z component.
      k2 = k2 - 1;
      COLTYP(nj) = 4;
      INDXP(k2) = nj;
      if (j == n) goto L100;
    } else {
      pj = nj;
      goto L80;
    }
  }
L80:
  j = j + 1;
  // The Fortran reads NJ = INDX(J) before this test, also INDX(N+1) (which
  // is INDXC(1) in xLAED1's workspace); it is not used then.
  if (j > n) goto L100;
  nj = INDX(j);
  if (*rho * std::abs(Z(nj)) <= tol) {
    // Deflate due to small z component.
    k2 = k2 - 1;
    COLTYP(nj) = 4;
    INDXP(k2) = nj;
  } else {
    // Check if eigenvalues are close enough to allow deflation.
    s = Z(pj);
    c = Z(nj);
    // Find sqrt(a**2+b**2) without overflow or
    // destructive underflow.
    tau = Lapy2(c, s);
    t = D(nj) - D(pj);
    c = c / tau;
    s = -s / tau;
    if (std::abs(t * c * s) <= tol) {
      // Deflation is possible.
      Z(nj) = tau;
      Z(pj) = zero;
      if (COLTYP(nj) != COLTYP(pj)) COLTYP(nj) = 2;
      COLTYP(pj) = 4;
      Rot(n, Q.ptr(1, pj), 1, Q.ptr(1, nj), 1, c, s);
      t = D(pj) * (c * c) + D(nj) * (s * s);
      D(nj) = D(pj) * (s * s) + D(nj) * (c * c);
      D(pj) = t;
      k2 = k2 - 1;
      i = 1;
    L90:
      if (k2 + i <= n) {
        if (D(pj) < D(INDXP(k2 + i))) {
          INDXP(k2 + i - 1) = INDXP(k2 + i);
          INDXP(k2 + i) = pj;
          i = i + 1;
          goto L90;
        } else {
          INDXP(k2 + i - 1) = pj;
        }
      } else {
        INDXP(k2 + i - 1) = pj;
      }
      pj = nj;
    } else {
      *k = *k + 1;
      DLAMBDA(*k) = D(pj);
      W(*k) = Z(pj);
      INDXP(*k) = pj;
      pj = nj;
    }
  }
  goto L80;
L100:
  // Record the last eigenvalue.
  *k = *k + 1;
  DLAMBDA(*k) = D(pj);
  W(*k) = Z(pj);
  INDXP(*k) = pj;
  // Count up the total number of the various types of columns, then
  // form a permutation which positions the four column types into
  // four uniform groups (although one or more of these groups may be
  // empty).
  for (j = 1; j <= 4; ++j) {
    CTOT(j) = 0;
  }
  for (j = 1; j <= n; ++j) {
    ct = COLTYP(j);
    CTOT(ct) = CTOT(ct) + 1;
  }
  // PSM(*) = Position in SubMatrix (of types 1 through 4)
  PSM(1) = 1;
  PSM(2) = 1 + CTOT(1);
  PSM(3) = PSM(2) + CTOT(2);
  PSM(4) = PSM(3) + CTOT(3);
  *k = n - CTOT(4);
  // Fill out the INDXC array so that the permutation which it induces
  // will place all type-1 columns first, all type-2 columns next,
  // then all type-3's, and finally all type-4's.
  for (j = 1; j <= n; ++j) {
    js = INDXP(j);
    ct = COLTYP(js);
    INDX(PSM(ct)) = js;
    INDXC(PSM(ct)) = j;
    PSM(ct) = PSM(ct) + 1;
  }
  // Sort the eigenvalues and corresponding eigenvectors into DLAMBDA
  // and Q2 respectively.  The eigenvalues/vectors which were not
  // deflated go into the first K slots of DLAMBDA and Q2 respectively,
  // while those which were deflated go into the last N - K slots.
  i = 1;
  iq1 = 1;
  iq2 = 1 + (CTOT(1) + CTOT(2)) * n1;
  for (j = 1; j <= CTOT(1); ++j) {
    js = INDX(i);
    Copy(n1, Q.ptr(1, js), 1, Q2.ptr(iq1), 1);
    Z(i) = D(js);
    i = i + 1;
    iq1 = iq1 + n1;
  }
  for (j = 1; j <= CTOT(2); ++j) {
    js = INDX(i);
    Copy(n1, Q.ptr(1, js), 1, Q2.ptr(iq1), 1);
    Copy(n2, Q.ptr(n1 + 1, js), 1, Q2.ptr(iq2), 1);
    Z(i) = D(js);
    i = i + 1;
    iq1 = iq1 + n1;
    iq2 = iq2 + n2;
  }
  for (j = 1; j <= CTOT(3); ++j) {
    js = INDX(i);
    Copy(n2, Q.ptr(n1 + 1, js), 1, Q2.ptr(iq2), 1);
    Z(i) = D(js);
    i = i + 1;
    iq2 = iq2 + n2;
  }
  iq1 = iq2;
  for (j = 1; j <= CTOT(4); ++j) {
    js = INDX(i);
    Copy(n, Q.ptr(1, js), 1, Q2.ptr(iq2), 1);
    iq2 = iq2 + n;
    Z(i) = D(js);
    i = i + 1;
  }
  // The deflated eigenvalues and their corresponding vectors go back
  // into the last N - K slots of D and Q respectively.
  if (*k < n) {
    Lacpy('A', n, CTOT(4), Q2.ptr(iq1), n, Q.ptr(1, *k + 1), ldq);
    Copy(n - *k, Z.ptr(*k + 1), 1, D.ptr(*k + 1), 1);
  }
  // Copy CTOT into COLTYP for referencing in DLAED3.
  for (j = 1; j <= 4; ++j) {
    COLTYP(j) = CTOT(j);
  }
L190:
  return info;
}

// xLAED1: merges the eigensystems of the two halves (of orders CUTPNT and
// N - CUTPNT) of a symmetric tridiagonal matrix cut by the rank-one
// modification RHO (COMPZ = 'I'): on entry D and Q hold the eigenvalues and
// eigenvectors of the halves, INDXQ the permutations that sort each half;
// on exit those of the merged matrix. RHO is modified (by xLAED2).
template <typename R>
blasint Laed1(blasint n, R* d, R* q, blasint ldq, blasint* indxq, R* rho,
              blasint cutpnt, R* work, blasint* iwork) {
  const Vector<R> D(d);
  const Vector<blasint> INDXQ(indxq);
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> Q(q, ldq);
  const Vector<R> WORK(work);
  blasint coltyp, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, k, n1, n2,
      zpp1;
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -4;
  } else if (Min<blasint>(1, n / 2) > cutpnt || (n / 2) < cutpnt) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<R>("LAED1", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  // The following values are integer pointers which indicate
  // the portion of the workspace
  // used by a particular array in DLAED2 and DLAED3.
  iz = 1;
  idlmda = iz + n;
  iw = idlmda + n;
  iq2 = iw + n;
  indx = 1;
  indxc = indx + n;
  coltyp = indxc + n;
  indxp = coltyp + n;
  // Form the z-vector which consists of the last row of Q_1 and the
  // first row of Q_2.
  Copy(cutpnt, Q.ptr(cutpnt, 1), ldq, WORK.ptr(iz), 1);
  zpp1 = cutpnt + 1;
  Copy(n - cutpnt, Q.ptr(zpp1, zpp1), ldq, WORK.ptr(iz + cutpnt), 1);
  // Deflate eigenvalues.
  info = Laed2(&k, n, cutpnt, d, q, ldq, indxq, rho, WORK.ptr(iz),
               WORK.ptr(idlmda), WORK.ptr(iw), WORK.ptr(iq2), IWORK.ptr(indx),
               IWORK.ptr(indxc), IWORK.ptr(indxp), IWORK.ptr(coltyp));
  if (info != 0) goto L20;
  // Solve Secular Equation.
  if (k != 0) {
    is = (IWORK(coltyp) + IWORK(coltyp + 1)) * cutpnt +
         (IWORK(coltyp + 1) + IWORK(coltyp + 2)) * (n - cutpnt) + iq2;
    info =
        Laed3(k, n, cutpnt, d, q, ldq, *rho, WORK.ptr(idlmda), WORK.ptr(iq2),
              IWORK.ptr(indxc), IWORK.ptr(coltyp), WORK.ptr(iw), WORK.ptr(is));
    if (info != 0) goto L20;
    // Prepare the INDXQ sorting permutation.
    n1 = k;
    n2 = n - k;
    Lamrg(n1, n2, d, 1, -1, indxq);
  } else {
    for (i = 1; i <= n; ++i) {
      INDXQ(i) = i;
    }
  }
L20:
  return info;
}

// xLAEDA: the vector Z of the merge at level CURLVL, problem CURPBM, of
// xLAED7: the last row of the first and the first row of the second
// subproblem's eigenvectors, which are stored in factored form (the
// rotations, permutations and eigenvector blocks of the levels below).
template <typename R>
blasint Laeda(blasint n, blasint tlvls, blasint curlvl, blasint curpbm,
              blasint* prmptr, blasint* perm, blasint* givptr, blasint* givcol,
              R* givnum, R* q, blasint* qptr, R* z, R* ztemp) {
  const R zero = R(0);
  const R half = R(0.5);
  const R one = R(1);
  const Matrix<blasint> GIVCOL(givcol, 2);
  const Matrix<R> GIVNUM(givnum, 2);
  const Vector<blasint> GIVPTR(givptr);
  const Vector<blasint> PERM(perm);
  const Vector<blasint> PRMPTR(prmptr);
  const Vector<R> Q(q);
  const Vector<blasint> QPTR(qptr);
  const Vector<R> Z(z);
  const Vector<R> ZTEMP(ztemp);
  blasint bsiz1, bsiz2, curr, i, k, mid, psiz1, psiz2, ptr, zptr1;
  blasint info = 0;
  if (n < 0) {
    info = -1;
  }
  if (info != 0) {
    Xerbla<R>("LAEDA", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  // Determine location of first number in second half.
  mid = n / 2 + 1;
  // Gather last/first rows of appropriate eigenblocks into center of Z
  ptr = 1;
  // Determine location of lowest level subproblem in the full storage
  // scheme
  curr =
      ptr + curpbm * (blasint{1} << curlvl) + (blasint{1} << (curlvl - 1)) - 1;
  // Determine size of these matrices.  We add HALF to the value of
  // the SQRT in case the machine underestimates one of these square
  // roots.
  bsiz1 =
      static_cast<blasint>(half + std::sqrt(R(QPTR(curr + 1) - QPTR(curr))));
  bsiz2 = static_cast<blasint>(half +
                               std::sqrt(R(QPTR(curr + 2) - QPTR(curr + 1))));
  for (k = 1; k <= mid - bsiz1 - 1; ++k) {
    Z(k) = zero;
  }
  Copy(bsiz1, Q.ptr(QPTR(curr) + bsiz1 - 1), bsiz1, Z.ptr(mid - bsiz1), 1);
  Copy(bsiz2, Q.ptr(QPTR(curr + 1)), bsiz2, Z.ptr(mid), 1);
  for (k = mid + bsiz2; k <= n; ++k) {
    Z(k) = zero;
  }
  // Loop through remaining levels 1 -> CURLVL applying the Givens
  // rotations and permutation and then multiplying the center matrices
  // against the current Z.
  ptr = (blasint{1} << tlvls) + 1;
  for (k = 1; k <= curlvl - 1; ++k) {
    curr = ptr + curpbm * (blasint{1} << (curlvl - k)) +
           (blasint{1} << (curlvl - k - 1)) - 1;
    psiz1 = PRMPTR(curr + 1) - PRMPTR(curr);
    psiz2 = PRMPTR(curr + 2) - PRMPTR(curr + 1);
    zptr1 = mid - psiz1;
    // Apply Givens at CURR and CURR+1
    for (i = GIVPTR(curr); i <= GIVPTR(curr + 1) - 1; ++i) {
      Rot(1, Z.ptr(zptr1 + GIVCOL(1, i) - 1), 1,
          Z.ptr(zptr1 + GIVCOL(2, i) - 1), 1, GIVNUM(1, i), GIVNUM(2, i));
    }
    for (i = GIVPTR(curr + 1); i <= GIVPTR(curr + 2) - 1; ++i) {
      Rot(1, Z.ptr(mid - 1 + GIVCOL(1, i)), 1, Z.ptr(mid - 1 + GIVCOL(2, i)), 1,
          GIVNUM(1, i), GIVNUM(2, i));
    }
    psiz1 = PRMPTR(curr + 1) - PRMPTR(curr);
    psiz2 = PRMPTR(curr + 2) - PRMPTR(curr + 1);
    for (i = 0; i <= psiz1 - 1; ++i) {
      ZTEMP(i + 1) = Z(zptr1 + PERM(PRMPTR(curr) + i) - 1);
    }
    for (i = 0; i <= psiz2 - 1; ++i) {
      ZTEMP(psiz1 + i + 1) = Z(mid + PERM(PRMPTR(curr + 1) + i) - 1);
    }
    // Multiply Blocks at CURR and CURR+1
    // Determine size of these matrices.  We add HALF to the value of
    // the SQRT in case the machine underestimates one of these
    // square roots.
    bsiz1 =
        static_cast<blasint>(half + std::sqrt(R(QPTR(curr + 1) - QPTR(curr))));
    bsiz2 = static_cast<blasint>(half +
                                 std::sqrt(R(QPTR(curr + 2) - QPTR(curr + 1))));
    if (bsiz1 > 0) {
      Gemv('T', bsiz1, bsiz1, one, Q.ptr(QPTR(curr)), bsiz1, ZTEMP.ptr(1), 1,
           zero, Z.ptr(zptr1), 1);
    }
    Copy(psiz1 - bsiz1, ZTEMP.ptr(bsiz1 + 1), 1, Z.ptr(zptr1 + bsiz1), 1);
    if (bsiz2 > 0) {
      Gemv('T', bsiz2, bsiz2, one, Q.ptr(QPTR(curr + 1)), bsiz2,
           ZTEMP.ptr(psiz1 + 1), 1, zero, Z.ptr(mid), 1);
    }
    Copy(psiz2 - bsiz2, ZTEMP.ptr(psiz1 + bsiz2 + 1), 1, Z.ptr(mid + bsiz2), 1);
    ptr = ptr + (blasint{1} << (tlvls - k));
  }
  return info;
}

// xLAED9: the eigenvalues KSTART to KSTOP of the secular equation of the
// merge (xLAED4) and the eigenvectors of the rank-one modification, in S
// (xLAED7).
template <typename R>
blasint Laed9(blasint k, blasint kstart, blasint kstop, blasint n, R* d, R* q,
              blasint ldq, R rho, R* dlambda, R* w, R* s, blasint lds) {
  const Vector<R> D(d);
  const Vector<R> DLAMBDA(dlambda);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> S(s, lds);
  const Vector<R> W(w);
  blasint i, j;
  R temp;
  blasint info = 0;
  if (k < 0) {
    info = -1;
  } else if (kstart < 1 || kstart > Max<blasint>(1, k)) {
    info = -2;
  } else if (Max<blasint>(1, kstop) < kstart || kstop > Max<blasint>(1, k)) {
    info = -3;
  } else if (n < k) {
    info = -4;
  } else if (ldq < Max<blasint>(1, k)) {
    info = -7;
  } else if (lds < Max<blasint>(1, k)) {
    info = -12;
  }
  if (info != 0) {
    Xerbla<R>("LAED9", -info);
    return info;
  }
  // Quick return if possible
  if (k == 0) return info;
  for (j = kstart; j <= kstop; ++j) {
    info = Laed4(k, j, dlambda, w, Q.ptr(1, j), rho, D.ptr(j));
    // If the zero finder fails, the computation is terminated.
    if (info != 0) goto L120;
  }
  if (k == 1 || k == 2) {
    for (i = 1; i <= k; ++i) {
      for (j = 1; j <= k; ++j) {
        S(j, i) = Q(j, i);
      }
    }
    goto L120;
  }
  // Compute updated W.
  Copy(k, w, 1, s, 1);
  // Initialize W(I) = Q(I,I)
  Copy(k, q, ldq + 1, w, 1);
  for (j = 1; j <= k; ++j) {
    for (i = 1; i <= j - 1; ++i) {
      W(i) = W(i) * (Q(i, j) / (DLAMBDA(i) - DLAMBDA(j)));
    }
    for (i = j + 1; i <= k; ++i) {
      W(i) = W(i) * (Q(i, j) / (DLAMBDA(i) - DLAMBDA(j)));
    }
  }
  for (i = 1; i <= k; ++i) {
    W(i) = std::copysign(std::sqrt(-W(i)), S(i, 1));
  }
  // Compute eigenvectors of the modified rank-1 modification.
  for (j = 1; j <= k; ++j) {
    for (i = 1; i <= k; ++i) {
      Q(i, j) = W(i) / Q(i, j);
    }
    temp = Nrm2(k, Q.ptr(1, j), 1);
    for (i = 1; i <= k; ++i) {
      S(i, j) = Q(i, j) / temp;
    }
  }
L120:
  return info;
}

// xLAED8: the deflation step of xLAED7 (like xLAED2), which records the
// rotations (GIVPTR, GIVCOL, GIVNUM) and the permutation (PERM) for
// xLAEDA. Returns the number K of non-deflated eigenvalues; RHO is
// modified to the value that xLAED9 needs.
template <typename R>
blasint Laed8(blasint icompq, blasint* k, blasint n, blasint qsiz, R* d, R* q,
              blasint ldq, blasint* indxq, R* rho, blasint cutpnt, R* z,
              R* dlambda, R* q2, blasint ldq2, R* w, blasint* perm,
              blasint* givptr, blasint* givcol, R* givnum, blasint* indxp,
              blasint* indx) {
  const R mone = -R(1);
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R eight = R(8);
  const Vector<R> D(d);
  const Vector<R> DLAMBDA(dlambda);
  const Matrix<blasint> GIVCOL(givcol, 2);
  const Matrix<R> GIVNUM(givnum, 2);
  const Vector<blasint> INDX(indx);
  const Vector<blasint> INDXP(indxp);
  const Vector<blasint> INDXQ(indxq);
  const Vector<blasint> PERM(perm);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> Q2(q2, ldq2);
  const Vector<R> W(w);
  const Vector<R> Z(z);
  blasint i, imax, j, jmax, jp, k2, n1, n1p1, n2;
  // Set when the first undeflated eigenvalue is found, before any use.
  blasint jlam = 0;
  R c, eps, s, t, tau, tol;
  blasint info = 0;
  if (icompq < 0 || icompq > 1) {
    info = -1;
  } else if (n < 0) {
    info = -3;
  } else if (icompq == 1 && qsiz < n) {
    info = -4;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -7;
  } else if (cutpnt < Min<blasint>(1, n) || cutpnt > n) {
    info = -10;
  } else if (ldq2 < Max<blasint>(1, n)) {
    info = -14;
  }
  if (info != 0) {
    Xerbla<R>("LAED8", -info);
    return info;
  }
  // Need to initialize GIVPTR to O here in case of quick exit
  // to prevent an unspecified code behavior (usually sigfault)
  // when IWORK array on entry to *stedc is not zeroed
  // (or at least some IWORK entries which used in *laed7 for GIVPTR).
  *givptr = 0;
  // Quick return if possible
  if (n == 0) return info;
  n1 = cutpnt;
  n2 = n - n1;
  n1p1 = n1 + 1;
  if (*rho < zero) {
    Rscal(n2, mone, Z.ptr(n1p1), 1);
  }
  // Normalize z so that norm(z) = 1
  t = one / std::sqrt(two);
  for (j = 1; j <= n; ++j) {
    INDX(j) = j;
  }
  Rscal(n, t, z, 1);
  *rho = std::abs(two * *rho);
  // Sort the eigenvalues into increasing order
  for (i = cutpnt + 1; i <= n; ++i) {
    INDXQ(i) = INDXQ(i) + cutpnt;
  }
  for (i = 1; i <= n; ++i) {
    DLAMBDA(i) = D(INDXQ(i));
    W(i) = Z(INDXQ(i));
  }
  i = 1;
  j = cutpnt + 1;
  Lamrg(n1, n2, dlambda, 1, 1, indx);
  for (i = 1; i <= n; ++i) {
    D(i) = DLAMBDA(INDX(i));
    Z(i) = W(INDX(i));
  }
  // Calculate the allowable deflation tolerance
  imax = Iamax(n, z, 1);
  jmax = Iamax(n, d, 1);
  eps = Lamch<R>('E');
  tol = eight * eps * std::abs(D(jmax));
  // If the rank-1 modifier is small enough, no more needs to be done
  // except to reorganize Q so that its columns correspond with the
  // elements in D.
  if (*rho * std::abs(Z(imax)) <= tol) {
    *k = 0;
    if (icompq == 0) {
      for (j = 1; j <= n; ++j) {
        PERM(j) = INDXQ(INDX(j));
      }
    } else {
      for (j = 1; j <= n; ++j) {
        PERM(j) = INDXQ(INDX(j));
        Copy(qsiz, Q.ptr(1, PERM(j)), 1, Q2.ptr(1, j), 1);
      }
      Lacpy('A', qsiz, n, Q2.ptr(1, 1), ldq2, Q.ptr(1, 1), ldq);
    }
    return info;
  }
  // If there are multiple eigenvalues then the problem deflates.  Here
  // the number of equal eigenvalues are found.  As each equal
  // eigenvalue is found, an elementary reflector is computed to rotate
  // the corresponding eigensubspace so that the corresponding
  // components of Z are zero in this new basis.
  *k = 0;
  k2 = n + 1;
  for (j = 1; j <= n; ++j) {
    if (*rho * std::abs(Z(j)) <= tol) {
      // Deflate due to small z component.
      k2 = k2 - 1;
      INDXP(k2) = j;
      if (j == n) goto L110;
    } else {
      jlam = j;
      goto L80;
    }
  }
L80:
  j = j + 1;
  if (j > n) goto L100;
  if (*rho * std::abs(Z(j)) <= tol) {
    // Deflate due to small z component.
    k2 = k2 - 1;
    INDXP(k2) = j;
  } else {
    // Check if eigenvalues are close enough to allow deflation.
    s = Z(jlam);
    c = Z(j);
    // Find sqrt(a**2+b**2) without overflow or
    // destructive underflow.
    tau = Lapy2(c, s);
    t = D(j) - D(jlam);
    c = c / tau;
    s = -s / tau;
    if (std::abs(t * c * s) <= tol) {
      // Deflation is possible.
      Z(j) = tau;
      Z(jlam) = zero;
      // Record the appropriate Givens rotation
      *givptr = *givptr + 1;
      GIVCOL(1, *givptr) = INDXQ(INDX(jlam));
      GIVCOL(2, *givptr) = INDXQ(INDX(j));
      GIVNUM(1, *givptr) = c;
      GIVNUM(2, *givptr) = s;
      if (icompq == 1) {
        Rot(qsiz, Q.ptr(1, INDXQ(INDX(jlam))), 1, Q.ptr(1, INDXQ(INDX(j))), 1,
            c, s);
      }
      t = D(jlam) * c * c + D(j) * s * s;
      D(j) = D(jlam) * s * s + D(j) * c * c;
      D(jlam) = t;
      k2 = k2 - 1;
      i = 1;
    L90:
      if (k2 + i <= n) {
        if (D(jlam) < D(INDXP(k2 + i))) {
          INDXP(k2 + i - 1) = INDXP(k2 + i);
          INDXP(k2 + i) = jlam;
          i = i + 1;
          goto L90;
        } else {
          INDXP(k2 + i - 1) = jlam;
        }
      } else {
        INDXP(k2 + i - 1) = jlam;
      }
      jlam = j;
    } else {
      *k = *k + 1;
      W(*k) = Z(jlam);
      DLAMBDA(*k) = D(jlam);
      INDXP(*k) = jlam;
      jlam = j;
    }
  }
  goto L80;
L100:
  // Record the last eigenvalue.
  *k = *k + 1;
  W(*k) = Z(jlam);
  DLAMBDA(*k) = D(jlam);
  INDXP(*k) = jlam;
L110:
  // Sort the eigenvalues and corresponding eigenvectors into DLAMBDA
  // and Q2 respectively.  The eigenvalues/vectors which were not
  // deflated go into the first K slots of DLAMBDA and Q2 respectively,
  // while those which were deflated go into the last N - K slots.
  if (icompq == 0) {
    for (j = 1; j <= n; ++j) {
      jp = INDXP(j);
      DLAMBDA(j) = D(jp);
      PERM(j) = INDXQ(INDX(jp));
    }
  } else {
    for (j = 1; j <= n; ++j) {
      jp = INDXP(j);
      DLAMBDA(j) = D(jp);
      PERM(j) = INDXQ(INDX(jp));
      Copy(qsiz, Q.ptr(1, PERM(j)), 1, Q2.ptr(1, j), 1);
    }
  }
  // The deflated eigenvalues and their corresponding vectors go back
  // into the last N - K slots of D and Q respectively.
  if (*k < n) {
    if (icompq == 0) {
      Copy(n - *k, DLAMBDA.ptr(*k + 1), 1, D.ptr(*k + 1), 1);
    } else {
      Copy(n - *k, DLAMBDA.ptr(*k + 1), 1, D.ptr(*k + 1), 1);
      Lacpy('A', qsiz, n - *k, Q2.ptr(1, *k + 1), ldq2, Q.ptr(1, *k + 1), ldq);
    }
  }
  return info;
}

// ZLAED8 (complex Q): like xLAED8 with ICOMPQ = 1; Q and Q2 are complex.
template <typename T>
blasint Laed8(blasint* k, blasint n, blasint qsiz, T* q, blasint ldq,
              RealT<T>* d, RealT<T>* rho, blasint cutpnt, RealT<T>* z,
              RealT<T>* dlambda, T* q2, blasint ldq2, RealT<T>* w,
              blasint* indxp, blasint* indx, blasint* indxq, blasint* perm,
              blasint* givptr, blasint* givcol, RealT<T>* givnum) {
  using R = RealT<T>;

  const R mone = -R(1);
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R eight = R(8);
  const Vector<R> D(d);
  const Vector<R> DLAMBDA(dlambda);
  const Matrix<blasint> GIVCOL(givcol, 2);
  const Matrix<R> GIVNUM(givnum, 2);
  const Vector<blasint> INDX(indx);
  const Vector<blasint> INDXP(indxp);
  const Vector<blasint> INDXQ(indxq);
  const Vector<blasint> PERM(perm);
  const Matrix<T> Q(q, ldq);
  const Matrix<T> Q2(q2, ldq2);
  const Vector<R> W(w);
  const Vector<R> Z(z);
  blasint i, imax, j, jmax, jp, k2, n1, n1p1, n2;
  // Set when the first undeflated eigenvalue is found, before any use.
  blasint jlam = 0;
  R c, eps, s, t, tau, tol;
  blasint info = 0;
  if (n < 0) {
    info = -2;
  } else if (qsiz < n) {
    info = -3;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -5;
  } else if (cutpnt < Min<blasint>(1, n) || cutpnt > n) {
    info = -8;
  } else if (ldq2 < Max<blasint>(1, n)) {
    info = -12;
  }
  if (info != 0) {
    Xerbla<T>("LAED8", -info);
    return info;
  }
  // Need to initialize GIVPTR to O here in case of quick exit
  // to prevent an unspecified code behavior (usually sigfault)
  // when IWORK array on entry to *stedc is not zeroed
  // (or at least some IWORK entries which used in *laed7 for GIVPTR).
  *givptr = 0;
  // Quick return if possible
  if (n == 0) return info;
  n1 = cutpnt;
  n2 = n - n1;
  n1p1 = n1 + 1;
  if (*rho < zero) {
    Rscal<R>(n2, mone, Z.ptr(n1p1), 1);
  }
  // Normalize z so that norm(z) = 1
  t = one / std::sqrt(two);
  for (j = 1; j <= n; ++j) {
    INDX(j) = j;
  }
  Rscal<R>(n, t, z, 1);
  *rho = std::abs(two * *rho);
  // Sort the eigenvalues into increasing order
  for (i = cutpnt + 1; i <= n; ++i) {
    INDXQ(i) = INDXQ(i) + cutpnt;
  }
  for (i = 1; i <= n; ++i) {
    DLAMBDA(i) = D(INDXQ(i));
    W(i) = Z(INDXQ(i));
  }
  i = 1;
  j = cutpnt + 1;
  Lamrg<R>(n1, n2, dlambda, 1, 1, indx);
  for (i = 1; i <= n; ++i) {
    D(i) = DLAMBDA(INDX(i));
    Z(i) = W(INDX(i));
  }
  // Calculate the allowable deflation tolerance
  imax = Iamax(n, z, 1);
  jmax = Iamax(n, d, 1);
  eps = Lamch<R>('E');
  tol = eight * eps * std::abs(D(jmax));
  // If the rank-1 modifier is small enough, no more needs to be done
  // -- except to reorganize Q so that its columns correspond with the
  // elements in D.
  if (*rho * std::abs(Z(imax)) <= tol) {
    *k = 0;
    for (j = 1; j <= n; ++j) {
      PERM(j) = INDXQ(INDX(j));
      Copy(qsiz, Q.ptr(1, PERM(j)), 1, Q2.ptr(1, j), 1);
    }
    Lacpy('A', qsiz, n, Q2.ptr(1, 1), ldq2, Q.ptr(1, 1), ldq);
    return info;
  }
  // If there are multiple eigenvalues then the problem deflates.  Here
  // the number of equal eigenvalues are found.  As each equal
  // eigenvalue is found, an elementary reflector is computed to rotate
  // the corresponding eigensubspace so that the corresponding
  // components of Z are zero in this new basis.
  *k = 0;
  k2 = n + 1;
  for (j = 1; j <= n; ++j) {
    if (*rho * std::abs(Z(j)) <= tol) {
      // Deflate due to small z component.
      k2 = k2 - 1;
      INDXP(k2) = j;
      if (j == n) goto L100;
    } else {
      jlam = j;
      goto L70;
    }
  }
L70:
  j = j + 1;
  if (j > n) goto L90;
  if (*rho * std::abs(Z(j)) <= tol) {
    // Deflate due to small z component.
    k2 = k2 - 1;
    INDXP(k2) = j;
  } else {
    // Check if eigenvalues are close enough to allow deflation.
    s = Z(jlam);
    c = Z(j);
    // Find sqrt(a**2+b**2) without overflow or
    // destructive underflow.
    tau = Lapy2<R>(c, s);
    t = D(j) - D(jlam);
    c = c / tau;
    s = -s / tau;
    if (std::abs(t * c * s) <= tol) {
      // Deflation is possible.
      Z(j) = tau;
      Z(jlam) = zero;
      // Record the appropriate Givens rotation
      *givptr = *givptr + 1;
      GIVCOL(1, *givptr) = INDXQ(INDX(jlam));
      GIVCOL(2, *givptr) = INDXQ(INDX(j));
      GIVNUM(1, *givptr) = c;
      GIVNUM(2, *givptr) = s;
      Rot(qsiz, Q.ptr(1, INDXQ(INDX(jlam))), 1, Q.ptr(1, INDXQ(INDX(j))), 1, c,
          s);
      t = D(jlam) * c * c + D(j) * s * s;
      D(j) = D(jlam) * s * s + D(j) * c * c;
      D(jlam) = t;
      k2 = k2 - 1;
      i = 1;
    L80:
      if (k2 + i <= n) {
        if (D(jlam) < D(INDXP(k2 + i))) {
          INDXP(k2 + i - 1) = INDXP(k2 + i);
          INDXP(k2 + i) = jlam;
          i = i + 1;
          goto L80;
        } else {
          INDXP(k2 + i - 1) = jlam;
        }
      } else {
        INDXP(k2 + i - 1) = jlam;
      }
      jlam = j;
    } else {
      *k = *k + 1;
      W(*k) = Z(jlam);
      DLAMBDA(*k) = D(jlam);
      INDXP(*k) = jlam;
      jlam = j;
    }
  }
  goto L70;
L90:
  // Record the last eigenvalue.
  *k = *k + 1;
  W(*k) = Z(jlam);
  DLAMBDA(*k) = D(jlam);
  INDXP(*k) = jlam;
L100:
  // Sort the eigenvalues and corresponding eigenvectors into DLAMBDA
  // and Q2 respectively.  The eigenvalues/vectors which were not
  // deflated go into the first K slots of DLAMBDA and Q2 respectively,
  // while those which were deflated go into the last N - K slots.
  for (j = 1; j <= n; ++j) {
    jp = INDXP(j);
    DLAMBDA(j) = D(jp);
    PERM(j) = INDXQ(INDX(jp));
    Copy(qsiz, Q.ptr(1, PERM(j)), 1, Q2.ptr(1, j), 1);
  }
  // The deflated eigenvalues and their corresponding vectors go back
  // into the last N - K slots of D and Q respectively.
  if (*k < n) {
    Copy<R>(n - *k, DLAMBDA.ptr(*k + 1), 1, D.ptr(*k + 1), 1);
    Lacpy('A', qsiz, n - *k, Q2.ptr(1, *k + 1), ldq2, Q.ptr(1, *k + 1), ldq);
  }
  return info;
}

// xLAED7: merges the eigensystems of two subproblems of xLAED0 like xLAED1,
// for ICOMPQ = 0 (eigenvalues only) and 1 (eigenvectors of the full
// matrix, Q of order QSIZ), keeping the eigenvectors of the merge in
// factored form for later merges (QSTORE, QPTR, PRMPTR, PERM, GIVPTR,
// GIVCOL, GIVNUM). RHO is modified (by xLAED8).
template <typename R>
blasint Laed7(blasint icompq, blasint n, blasint qsiz, blasint tlvls,
              blasint curlvl, blasint curpbm, R* d, R* q, blasint ldq,
              blasint* indxq, R* rho, blasint cutpnt, R* qstore, blasint* qptr,
              blasint* prmptr, blasint* perm, blasint* givptr, blasint* givcol,
              R* givnum, R* work, blasint* iwork) {
  const R one = R(1);
  const R zero = R(0);
  const Vector<R> D(d);
  const Matrix<blasint> GIVCOL(givcol, 2);
  const Matrix<R> GIVNUM(givnum, 2);
  const Vector<blasint> GIVPTR(givptr);
  const Vector<blasint> INDXQ(indxq);
  const Vector<blasint> IWORK(iwork);
  const Vector<blasint> PERM(perm);
  const Vector<blasint> PRMPTR(prmptr);
  const Matrix<R> Q(q, ldq);
  const Vector<blasint> QPTR(qptr);
  const Vector<R> QSTORE(qstore);
  const Vector<R> WORK(work);
  blasint coltyp, curr, i, idlmda, indx, indxc, indxp, iq2, is, iw, iz, ldq2,
      n1, n2, ptr;
  blasint info = 0;
  if (icompq < 0 || icompq > 1) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (icompq == 1 && qsiz < n) {
    info = -3;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -9;
  } else if (Min<blasint>(1, n) > cutpnt || n < cutpnt) {
    info = -12;
  }
  if (info != 0) {
    Xerbla<R>("LAED7", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  // The following values are for bookkeeping purposes only.  They are
  // integer pointers which indicate the portion of the workspace
  // used by a particular array in DLAED8 and DLAED9.
  if (icompq == 1) {
    ldq2 = qsiz;
  } else {
    ldq2 = n;
  }
  iz = 1;
  idlmda = iz + n;
  iw = idlmda + n;
  iq2 = iw + n;
  is = iq2 + n * ldq2;
  indx = 1;
  indxc = indx + n;
  coltyp = indxc + n;
  indxp = coltyp + n;
  // Form the z-vector which consists of the last row of Q_1 and the
  // first row of Q_2.
  ptr = 1 + (blasint{1} << tlvls);
  for (i = 1; i <= curlvl - 1; ++i) {
    ptr = ptr + (blasint{1} << (tlvls - i));
  }
  curr = ptr + curpbm;
  Laeda(n, tlvls, curlvl, curpbm, prmptr, perm, givptr, givcol, givnum, qstore,
        qptr, WORK.ptr(iz), WORK.ptr(iz + n));
  // When solving the final problem, we no longer need the stored data,
  // so we will overwrite the data from this level onto the previously
  // used storage space.
  if (curlvl == tlvls) {
    QPTR(curr) = 1;
    PRMPTR(curr) = 1;
    GIVPTR(curr) = 1;
  }
  // Sort and Deflate eigenvalues.
  // Set by xLAED8 (its arguments are valid here).
  blasint k = 0;
  info = Laed8(icompq, &k, n, qsiz, d, q, ldq, indxq, rho, cutpnt, WORK.ptr(iz),
               WORK.ptr(idlmda), WORK.ptr(iq2), ldq2, WORK.ptr(iw),
               PERM.ptr(PRMPTR(curr)), GIVPTR.ptr(curr + 1),
               GIVCOL.ptr(1, GIVPTR(curr)), GIVNUM.ptr(1, GIVPTR(curr)),
               IWORK.ptr(indxp), IWORK.ptr(indx));
  PRMPTR(curr + 1) = PRMPTR(curr) + n;
  GIVPTR(curr + 1) = GIVPTR(curr + 1) + GIVPTR(curr);
  // Solve Secular Equation.
  if (k != 0) {
    info = Laed9(k, 1, k, n, d, WORK.ptr(is), k, *rho, WORK.ptr(idlmda),
                 WORK.ptr(iw), QSTORE.ptr(QPTR(curr)), k);
    if (info != 0) goto L30;
    if (icompq == 1) {
      Gemm('N', 'N', qsiz, k, k, one, WORK.ptr(iq2), ldq2,
           QSTORE.ptr(QPTR(curr)), k, zero, q, ldq);
    }
    QPTR(curr + 1) = QPTR(curr) + (k * k);
    // Prepare the INDXQ sorting permutation.
    n1 = k;
    n2 = n - k;
    Lamrg(n1, n2, d, 1, -1, indxq);
  } else {
    QPTR(curr + 1) = QPTR(curr);
    for (i = 1; i <= n; ++i) {
      INDXQ(i) = i;
    }
  }
L30:
  return info;
}

// ZLAED7 (complex Q): like xLAED7 with ICOMPQ = 1.
template <typename T>
blasint Laed7(blasint n, blasint cutpnt, blasint qsiz, blasint tlvls,
              blasint curlvl, blasint curpbm, RealT<T>* d, T* q, blasint ldq,
              RealT<T>* rho, blasint* indxq, RealT<T>* qstore, blasint* qptr,
              blasint* prmptr, blasint* perm, blasint* givptr, blasint* givcol,
              RealT<T>* givnum, T* work, RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;

  const Vector<R> D(d);
  const Matrix<blasint> GIVCOL(givcol, 2);
  const Matrix<R> GIVNUM(givnum, 2);
  const Vector<blasint> GIVPTR(givptr);
  const Vector<blasint> INDXQ(indxq);
  const Vector<blasint> IWORK(iwork);
  const Vector<blasint> PERM(perm);
  const Vector<blasint> PRMPTR(prmptr);
  const Matrix<T> Q(q, ldq);
  const Vector<blasint> QPTR(qptr);
  const Vector<R> QSTORE(qstore);
  const Vector<R> RWORK(rwork);
  const Vector<T> WORK(work);
  blasint coltyp, curr, i, idlmda, indx, indxc, indxp, iq, iw, iz, n1, n2, ptr;
  blasint info = 0;
  // IF( ICOMPQ.LT.0 .OR. ICOMPQ.GT.1 ) THEN
  // INFO = -1
  // ELSE IF( N.LT.0 ) THEN
  if (n < 0) {
    info = -1;
  } else if (Min<blasint>(1, n) > cutpnt || n < cutpnt) {
    info = -2;
  } else if (qsiz < n) {
    info = -3;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<T>("LAED7", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  // The following values are for bookkeeping purposes only.  They are
  // integer pointers which indicate the portion of the workspace
  // used by a particular array in DLAED2 and SLAED3.
  iz = 1;
  idlmda = iz + n;
  iw = idlmda + n;
  iq = iw + n;
  indx = 1;
  indxc = indx + n;
  coltyp = indxc + n;
  indxp = coltyp + n;
  // Form the z-vector which consists of the last row of Q_1 and the
  // first row of Q_2.
  ptr = 1 + (blasint{1} << tlvls);
  for (i = 1; i <= curlvl - 1; ++i) {
    ptr = ptr + (blasint{1} << (tlvls - i));
  }
  curr = ptr + curpbm;
  Laeda<R>(n, tlvls, curlvl, curpbm, prmptr, perm, givptr, givcol, givnum,
           qstore, qptr, RWORK.ptr(iz), RWORK.ptr(iz + n));
  // When solving the final problem, we no longer need the stored data,
  // so we will overwrite the data from this level onto the previously
  // used storage space.
  if (curlvl == tlvls) {
    QPTR(curr) = 1;
    PRMPTR(curr) = 1;
    GIVPTR(curr) = 1;
  }
  // Sort and Deflate eigenvalues.
  // Set by xLAED8 (its arguments are valid here).
  blasint k = 0;
  info = Laed8(&k, n, qsiz, q, ldq, d, rho, cutpnt, RWORK.ptr(iz),
               RWORK.ptr(idlmda), work, qsiz, RWORK.ptr(iw), IWORK.ptr(indxp),
               IWORK.ptr(indx), indxq, PERM.ptr(PRMPTR(curr)),
               GIVPTR.ptr(curr + 1), GIVCOL.ptr(1, GIVPTR(curr)),
               GIVNUM.ptr(1, GIVPTR(curr)));
  PRMPTR(curr + 1) = PRMPTR(curr) + n;
  GIVPTR(curr + 1) = GIVPTR(curr + 1) + GIVPTR(curr);
  // Solve Secular Equation.
  if (k != 0) {
    info = Laed9<R>(k, 1, k, n, d, RWORK.ptr(iq), k, *rho, RWORK.ptr(idlmda),
                    RWORK.ptr(iw), QSTORE.ptr(QPTR(curr)), k);
    Lacrm(qsiz, k, work, qsiz, QSTORE.ptr(QPTR(curr)), k, q, ldq,
          RWORK.ptr(iq));
    QPTR(curr + 1) = QPTR(curr) + (k * k);
    if (info != 0) {
      return info;
    }
    // Prepare the INDXQ sorting permutation.
    n1 = k;
    n2 = n - k;
    Lamrg<R>(n1, n2, d, 1, -1, indxq);
  } else {
    QPTR(curr + 1) = QPTR(curr);
    for (i = 1; i <= n; ++i) {
      INDXQ(i) = i;
    }
  }
  return info;
}

// xLAED0: all eigenvalues and (ICOMPQ = 1, 2) eigenvectors of the
// symmetric tridiagonal matrix (D, E) of order N by divide and conquer:
// ICOMPQ = 0 eigenvalues only, 1 also the eigenvectors of the full matrix
// (Q of order QSIZ holds the matrix that reduced it to tridiagonal form),
// 2 the eigenvectors of the tridiagonal matrix.
template <typename R>
blasint Laed0(blasint icompq, blasint qsiz, blasint n, R* d, R* e, R* q,
              blasint ldq, R* qstore, blasint ldqs, R* work, blasint* iwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> QSTORE(qstore, ldqs);
  const Vector<R> WORK(work);
  blasint curlvl, curprb, curr, i, igivnm, indxq, j, k, lgn, matsiz, msd2,
      smlsiz, smm1, spm1, spm2, submat, subpbs, tlvls;
  // Set and used only for ICOMPQ != 2.
  blasint igivcl = 0, igivpt = 0, iperm = 0, iprmpt = 0, iq = 0, iqptr = 0,
          iwrem = 0;
  R temp;
  blasint info = 0;
  if (icompq < 0 || icompq > 2) {
    info = -1;
  } else if ((icompq == 1) && (qsiz < Max<blasint>(0, n))) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldqs < Max<blasint>(1, n)) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<R>("LAED0", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  smlsiz = Ilaenv<R>(9, "LAED0", " ", 0, 0, 0, 0);
  // Determine the size and placement of the submatrices, and save in
  // the leading elements of IWORK.
  IWORK(1) = n;
  subpbs = 1;
  tlvls = 0;
L10:
  if (IWORK(subpbs) > smlsiz) {
    for (j = subpbs; j >= 1; --j) {
      IWORK(2 * j) = (IWORK(j) + 1) / 2;
      IWORK(2 * j - 1) = IWORK(j) / 2;
    }
    tlvls = tlvls + 1;
    subpbs = 2 * subpbs;
    goto L10;
  }
  for (j = 2; j <= subpbs; ++j) {
    IWORK(j) = IWORK(j) + IWORK(j - 1);
  }
  // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
  // using rank-1 modifications (cuts).
  spm1 = subpbs - 1;
  for (i = 1; i <= spm1; ++i) {
    submat = IWORK(i) + 1;
    smm1 = submat - 1;
    D(smm1) = D(smm1) - std::abs(E(smm1));
    D(submat) = D(submat) - std::abs(E(smm1));
  }
  indxq = 4 * n + 3;
  if (icompq != 2) {
    // Set up workspaces for eigenvalues only/accumulate new vectors
    // routine
    temp = std::log(R(n)) / std::log(two);
    lgn = static_cast<blasint>(temp);
    if ((blasint{1} << lgn) < n) lgn = lgn + 1;
    if ((blasint{1} << lgn) < n) lgn = lgn + 1;
    iprmpt = indxq + n + 1;
    iperm = iprmpt + n * lgn;
    iqptr = iperm + n * lgn;
    igivpt = iqptr + n + 2;
    igivcl = igivpt + n * lgn;
    igivnm = 1;
    iq = igivnm + 2 * n * lgn;
    iwrem = iq + (n * n) + 1;
    // Initialize pointers
    for (i = 0; i <= subpbs; ++i) {
      IWORK(iprmpt + i) = 1;
      IWORK(igivpt + i) = 1;
    }
    IWORK(iqptr) = 1;
  }
  // Solve each submatrix eigenproblem at the bottom of the divide and
  // conquer tree.
  curr = 0;
  for (i = 0; i <= spm1; ++i) {
    if (i == 0) {
      submat = 1;
      matsiz = IWORK(1);
    } else {
      submat = IWORK(i) + 1;
      matsiz = IWORK(i + 1) - IWORK(i);
    }
    if (icompq == 2) {
      info = Steqr('I', matsiz, D.ptr(submat), E.ptr(submat),
                   Q.ptr(submat, submat), ldq, work);
      if (info != 0) goto L130;
    } else {
      info = Steqr('I', matsiz, D.ptr(submat), E.ptr(submat),
                   WORK.ptr(iq - 1 + IWORK(iqptr + curr)), matsiz, work);
      if (info != 0) goto L130;
      if (icompq == 1) {
        Gemm('N', 'N', qsiz, matsiz, matsiz, one, Q.ptr(1, submat), ldq,
             WORK.ptr(iq - 1 + IWORK(iqptr + curr)), matsiz, zero,
             QSTORE.ptr(1, submat), ldqs);
      }
      IWORK(iqptr + curr + 1) = IWORK(iqptr + curr) + (matsiz * matsiz);
      curr = curr + 1;
    }
    k = 1;
    for (j = submat; j <= IWORK(i + 1); ++j) {
      IWORK(indxq + j) = k;
      k = k + 1;
    }
  }
  // Successively merge eigensystems of adjacent submatrices
  // into eigensystem for the corresponding larger matrix.
  // while ( SUBPBS > 1 )
  curlvl = 1;
L80:
  if (subpbs > 1) {
    spm2 = subpbs - 2;
    for (i = 0; i <= spm2; i += 2) {
      if (i == 0) {
        submat = 1;
        matsiz = IWORK(2);
        msd2 = IWORK(1);
        curprb = 0;
      } else {
        submat = IWORK(i) + 1;
        matsiz = IWORK(i + 2) - IWORK(i);
        msd2 = matsiz / 2;
        curprb = curprb + 1;
      }
      // Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
      // into an eigensystem of size MATSIZ.
      // DLAED1 is used only for the full eigensystem of a tridiagonal
      // matrix.
      // DLAED7 handles the cases in which eigenvalues only or eigenvalues
      // and eigenvectors of a full symmetric matrix (which was reduced to
      // tridiagonal form) are desired.
      if (icompq == 2) {
        info = Laed1(matsiz, D.ptr(submat), Q.ptr(submat, submat), ldq,
                     IWORK.ptr(indxq + submat), E.ptr(submat + msd2 - 1), msd2,
                     work, IWORK.ptr(subpbs + 1));
      } else {
        info = Laed7(icompq, matsiz, qsiz, tlvls, curlvl, curprb, D.ptr(submat),
                     QSTORE.ptr(1, submat), ldqs, IWORK.ptr(indxq + submat),
                     E.ptr(submat + msd2 - 1), msd2, WORK.ptr(iq),
                     IWORK.ptr(iqptr), IWORK.ptr(iprmpt), IWORK.ptr(iperm),
                     IWORK.ptr(igivpt), IWORK.ptr(igivcl), WORK.ptr(igivnm),
                     WORK.ptr(iwrem), IWORK.ptr(subpbs + 1));
      }
      if (info != 0) goto L130;
      IWORK(i / 2 + 1) = IWORK(i + 2);
    }
    subpbs = subpbs / 2;
    curlvl = curlvl + 1;
    goto L80;
  }
  // end while
  // Re-merge the eigenvalues/vectors which were deflated at the final
  // merge step.
  if (icompq == 1) {
    for (i = 1; i <= n; ++i) {
      j = IWORK(indxq + i);
      WORK(i) = D(j);
      Copy(qsiz, QSTORE.ptr(1, j), 1, Q.ptr(1, i), 1);
    }
    Copy(n, work, 1, d, 1);
  } else if (icompq == 2) {
    for (i = 1; i <= n; ++i) {
      j = IWORK(indxq + i);
      WORK(i) = D(j);
      Copy(n, Q.ptr(1, j), 1, WORK.ptr(n * i + 1), 1);
    }
    Copy(n, work, 1, d, 1);
    Lacpy('A', n, n, WORK.ptr(n + 1), n, q, ldq);
  } else {
    for (i = 1; i <= n; ++i) {
      j = IWORK(indxq + i);
      WORK(i) = D(j);
    }
    Copy(n, work, 1, d, 1);
  }
  goto L140;
L130:
  info = submat * (n + 1) + submat + matsiz - 1;
L140:
  return info;
}

// ZLAED0 (complex Q): like xLAED0 with ICOMPQ = 1.
template <typename T>
blasint Laed0(blasint qsiz, blasint n, RealT<T>* d, RealT<T>* e, T* q,
              blasint ldq, T* qstore, blasint ldqs, RealT<T>* rwork,
              blasint* iwork) {
  using R = RealT<T>;

  const R two = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Matrix<T> Q(q, ldq);
  const Matrix<T> QSTORE(qstore, ldqs);
  const Vector<R> RWORK(rwork);
  blasint curlvl, curprb, curr, i, igivcl, igivnm, igivpt, indxq, iperm, iprmpt,
      iq, iqptr, iwrem, j, k, lgn, ll, matsiz, msd2, smlsiz, smm1, spm1, spm2,
      submat, subpbs, tlvls;
  R temp;
  blasint info = 0;
  // IF( ICOMPQ .LT. 0 .OR. ICOMPQ .GT. 2 ) THEN
  // INFO = -1
  // ELSE IF( ( ICOMPQ .EQ. 1 ) .AND. ( QSIZ .LT. MAX( 0, N ) ) )
  // $        THEN
  if (qsiz < Max<blasint>(0, n)) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldq < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldqs < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("LAED0", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  smlsiz = Ilaenv<T>(9, "LAED0", " ", 0, 0, 0, 0);
  // Determine the size and placement of the submatrices, and save in
  // the leading elements of IWORK.
  IWORK(1) = n;
  subpbs = 1;
  tlvls = 0;
L10:
  if (IWORK(subpbs) > smlsiz) {
    for (j = subpbs; j >= 1; --j) {
      IWORK(2 * j) = (IWORK(j) + 1) / 2;
      IWORK(2 * j - 1) = IWORK(j) / 2;
    }
    tlvls = tlvls + 1;
    subpbs = 2 * subpbs;
    goto L10;
  }
  for (j = 2; j <= subpbs; ++j) {
    IWORK(j) = IWORK(j) + IWORK(j - 1);
  }
  // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
  // using rank-1 modifications (cuts).
  spm1 = subpbs - 1;
  for (i = 1; i <= spm1; ++i) {
    submat = IWORK(i) + 1;
    smm1 = submat - 1;
    D(smm1) = D(smm1) - std::abs(E(smm1));
    D(submat) = D(submat) - std::abs(E(smm1));
  }
  indxq = 4 * n + 3;
  // Set up workspaces for eigenvalues only/accumulate new vectors
  // routine
  temp = std::log(R(n)) / std::log(two);
  lgn = static_cast<blasint>(temp);
  if ((blasint{1} << lgn) < n) lgn = lgn + 1;
  if ((blasint{1} << lgn) < n) lgn = lgn + 1;
  iprmpt = indxq + n + 1;
  iperm = iprmpt + n * lgn;
  iqptr = iperm + n * lgn;
  igivpt = iqptr + n + 2;
  igivcl = igivpt + n * lgn;
  igivnm = 1;
  iq = igivnm + 2 * n * lgn;
  iwrem = iq + (n * n) + 1;
  // Initialize pointers
  for (i = 0; i <= subpbs; ++i) {
    IWORK(iprmpt + i) = 1;
    IWORK(igivpt + i) = 1;
  }
  IWORK(iqptr) = 1;
  // Solve each submatrix eigenproblem at the bottom of the divide and
  // conquer tree.
  curr = 0;
  for (i = 0; i <= spm1; ++i) {
    if (i == 0) {
      submat = 1;
      matsiz = IWORK(1);
    } else {
      submat = IWORK(i) + 1;
      matsiz = IWORK(i + 1) - IWORK(i);
    }
    ll = iq - 1 + IWORK(iqptr + curr);
    info = Steqr<R>('I', matsiz, D.ptr(submat), E.ptr(submat), RWORK.ptr(ll),
                    matsiz, rwork);
    Lacrm(qsiz, matsiz, Q.ptr(1, submat), ldq, RWORK.ptr(ll), matsiz,
          QSTORE.ptr(1, submat), ldqs, RWORK.ptr(iwrem));
    IWORK(iqptr + curr + 1) = IWORK(iqptr + curr) + (matsiz * matsiz);
    curr = curr + 1;
    if (info > 0) {
      info = submat * (n + 1) + submat + matsiz - 1;
      return info;
    }
    k = 1;
    for (j = submat; j <= IWORK(i + 1); ++j) {
      IWORK(indxq + j) = k;
      k = k + 1;
    }
  }
  // Successively merge eigensystems of adjacent submatrices
  // into eigensystem for the corresponding larger matrix.
  // while ( SUBPBS > 1 )
  curlvl = 1;
L80:
  if (subpbs > 1) {
    spm2 = subpbs - 2;
    for (i = 0; i <= spm2; i += 2) {
      if (i == 0) {
        submat = 1;
        matsiz = IWORK(2);
        msd2 = IWORK(1);
        curprb = 0;
      } else {
        submat = IWORK(i) + 1;
        matsiz = IWORK(i + 2) - IWORK(i);
        msd2 = matsiz / 2;
        curprb = curprb + 1;
      }
      // Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
      // into an eigensystem of size MATSIZ.  ZLAED7 handles the case
      // when the eigenvectors of a full or band Hermitian matrix (which
      // was reduced to tridiagonal form) are desired.
      // I am free to use Q as a valuable working space until Loop 150.
      info = Laed7(matsiz, msd2, qsiz, tlvls, curlvl, curprb, D.ptr(submat),
                   QSTORE.ptr(1, submat), ldqs, E.ptr(submat + msd2 - 1),
                   IWORK.ptr(indxq + submat), RWORK.ptr(iq), IWORK.ptr(iqptr),
                   IWORK.ptr(iprmpt), IWORK.ptr(iperm), IWORK.ptr(igivpt),
                   IWORK.ptr(igivcl), RWORK.ptr(igivnm), Q.ptr(1, submat),
                   RWORK.ptr(iwrem), IWORK.ptr(subpbs + 1));
      if (info > 0) {
        info = submat * (n + 1) + submat + matsiz - 1;
        return info;
      }
      IWORK(i / 2 + 1) = IWORK(i + 2);
    }
    subpbs = subpbs / 2;
    curlvl = curlvl + 1;
    goto L80;
  }
  // end while
  // Re-merge the eigenvalues/vectors which were deflated at the final
  // merge step.
  for (i = 1; i <= n; ++i) {
    j = IWORK(indxq + i);
    RWORK(i) = D(j);
    Copy(qsiz, QSTORE.ptr(1, j), 1, Q.ptr(1, i), 1);
  }
  Copy<R>(n, rwork, 1, d, 1);
  return info;
}

// xSTEDC: all eigenvalues and, optionally, eigenvectors of the symmetric
// tridiagonal matrix (D, E) by divide and conquer (xLAED0) for the
// unreduced blocks of more than SMLSIZ rows (from ILAENV), xSTEQR for the
// smaller ones, and xSTERF for COMPZ = 'N'. COMPZ = 'I': Z gets the
// eigenvectors of the tridiagonal matrix; 'V': Z (the matrix that reduced
// a full matrix to tridiagonal form) is multiplied by them.
template <typename R>
blasint Stedc(char compz, blasint n, R* d, R* e, R* z, blasint ldz, R* work,
              blasint lwork, blasint* iwork, blasint liwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint finish, i, icompz, ii, j, k, lgn, liwmin, lwmin, m, smlsiz, start,
      storez, strtrw;
  bool lquery;
  R eps, orgnrm, p, tiny;
  blasint info = 0;
  lquery = (lwork == -1 || liwork == -1);
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
  } else if ((ldz < 1) || (icompz > 0 && ldz < Max<blasint>(1, n))) {
    info = -6;
  }
  if (info == 0) {
    // Compute the workspace requirements
    smlsiz = Ilaenv<R>(9, "STEDC", " ", 0, 0, 0, 0);
    if (n <= 1 || icompz == 0) {
      liwmin = 1;
      lwmin = 1;
    } else if (n <= smlsiz) {
      liwmin = 1;
      lwmin = 2 * (n - 1);
    } else {
      lgn = static_cast<blasint>(std::log(R(n)) / std::log(two));
      if ((blasint{1} << lgn) < n) lgn = lgn + 1;
      if ((blasint{1} << lgn) < n) lgn = lgn + 1;
      if (icompz == 1) {
        lwmin = 1 + 3 * n + 2 * n * lgn + 4 * (n * n);
        liwmin = 6 + 6 * n + 5 * n * lgn;
      } else {  // ICOMPZ = 2
        lwmin = 1 + 4 * n + (n * n);
        liwmin = 3 + 5 * n;
      }
    }
    WORK(1) = WorkSize<R>(lwmin);
    IWORK(1) = liwmin;
    if (lwork < lwmin && !lquery) {
      info = -8;
    } else if (liwork < liwmin && !lquery) {
      info = -10;
    }
  }
  if (info != 0) {
    Xerbla<R>("STEDC", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  if (n == 1) {
    if (icompz != 0) Z(1, 1) = one;
    return info;
  }
  // If the following conditional clause is removed, then the routine
  // will use the Divide and Conquer routine to compute only the
  // eigenvalues, which requires (3N + 3N**2) real workspace and
  // (2 + 5N + 2N lg(N)) integer workspace.
  // Since on many architectures DSTERF is much faster than any other
  // algorithm for finding eigenvalues only, it is used here
  // as the default. If the conditional clause is removed, then
  // information on the size of workspace needs to be changed.
  // If COMPZ = 'N', use DSTERF to compute the eigenvalues.
  if (icompz == 0) {
    info = Sterf(n, d, e);
    goto L50;
  }
  // If N is smaller than the minimum divide size (SMLSIZ+1), then
  // solve the problem with another solver.
  if (n <= smlsiz) {
    info = Steqr(compz, n, d, e, z, ldz, work);
  } else {
    // If COMPZ = 'V', the Z matrix must be stored elsewhere for later
    // use.
    if (icompz == 1) {
      storez = 1 + n * n;
    } else {
      storez = 1;
    }
    if (icompz == 2) {
      Laset('F', n, n, zero, one, z, ldz);
    }
    // Scale.
    orgnrm = Lanst('M', n, d, e);
    if (orgnrm == zero) goto L50;
    eps = Lamch<R>('E');
    start = 1;
    // while ( START <= N )
  L10:
    if (start <= n) {
      // Let FINISH be the position of the next subdiagonal entry
      // such that E( FINISH ) <= TINY or FINISH = N if no such
      // subdiagonal exists.  The matrix identified by the elements
      // between START and FINISH constitutes an independent
      // sub-problem.
      finish = start;
    L20:
      if (finish < n) {
        tiny = eps * std::sqrt(std::abs(D(finish))) *
               std::sqrt(std::abs(D(finish + 1)));
        if (std::abs(E(finish)) > tiny) {
          finish = finish + 1;
          goto L20;
        }
      }
      // (Sub) Problem determined.  Compute its size and solve it.
      m = finish - start + 1;
      if (m == 1) {
        start = finish + 1;
        goto L10;
      }
      if (m > smlsiz) {
        // Scale.
        orgnrm = Lanst('M', m, D.ptr(start), E.ptr(start));
        info = Lascl('G', 0, 0, orgnrm, one, m, 1, D.ptr(start), m);
        info = Lascl('G', 0, 0, orgnrm, one, m - 1, 1, E.ptr(start), m - 1);
        if (icompz == 1) {
          strtrw = 1;
        } else {
          strtrw = start;
        }
        info = Laed0(icompz, n, m, D.ptr(start), E.ptr(start),
                     Z.ptr(strtrw, start), ldz, WORK.ptr(1), n,
                     WORK.ptr(storez), iwork);
        if (info != 0) {
          info = (info / (m + 1) + start - 1) * (n + 1) + info % (m + 1) +
                 start - 1;
          goto L50;
        }
        // Scale back.
        info = Lascl('G', 0, 0, one, orgnrm, m, 1, D.ptr(start), m);
      } else {
        if (icompz == 1) {
          // Since QR won't update a Z matrix which is larger than
          // the length of D, we must solve the sub-problem in a
          // workspace and then multiply back into Z.
          info = Steqr('I', m, D.ptr(start), E.ptr(start), work, m,
                       WORK.ptr(m * m + 1));
          Lacpy('A', n, m, Z.ptr(1, start), ldz, WORK.ptr(storez), n);
          Gemm('N', 'N', n, m, m, one, WORK.ptr(storez), n, work, m, zero,
               Z.ptr(1, start), ldz);
        } else if (icompz == 2) {
          info = Steqr('I', m, D.ptr(start), E.ptr(start), Z.ptr(start, start),
                       ldz, work);
        } else {
          info = Sterf(m, D.ptr(start), E.ptr(start));
        }
        if (info != 0) {
          info = start * (n + 1) + finish;
          goto L50;
        }
      }
      start = finish + 1;
      goto L10;
    }
    // endwhile
    if (icompz == 0) {
      // Use Quick Sort
      info = Lasrt('I', n, d);
    } else {
      // Use Selection Sort to minimize swaps of eigenvectors
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
  }
L50:
  WORK(1) = WorkSize<R>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

// ZSTEDC (complex Z): like xSTEDC; the eigenvectors of the tridiagonal
// matrix are computed in RWORK (by DSTEDC for COMPZ = 'I').
template <typename T>
blasint Stedc(char compz, blasint n, RealT<T>* d, RealT<T>* e, T* z,
              blasint ldz, T* work, blasint lwork, RealT<T>* rwork,
              blasint lrwork, blasint* iwork, blasint liwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> RWORK(rwork);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint finish, i, icompz, ii, j, k, lgn, liwmin, ll, lrwmin, lwmin, m,
      smlsiz, start;
  bool lquery;
  R eps, orgnrm, p, tiny;
  blasint info = 0;
  lquery = (lwork == -1 || lrwork == -1 || liwork == -1);
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
  } else if ((ldz < 1) || (icompz > 0 && ldz < Max<blasint>(1, n))) {
    info = -6;
  }
  if (info == 0) {
    // Compute the workspace requirements
    smlsiz = Ilaenv<T>(9, "STEDC", " ", 0, 0, 0, 0);
    if (n <= 1 || icompz == 0) {
      lwmin = 1;
      liwmin = 1;
      lrwmin = 1;
    } else if (n <= smlsiz) {
      lwmin = 1;
      liwmin = 1;
      lrwmin = 2 * (n - 1);
    } else if (icompz == 1) {
      lgn = static_cast<blasint>(std::log(R(n)) / std::log(two));
      if ((blasint{1} << lgn) < n) lgn = lgn + 1;
      if ((blasint{1} << lgn) < n) lgn = lgn + 1;
      lwmin = n * n;
      lrwmin = 1 + 3 * n + 2 * n * lgn + 4 * (n * n);
      liwmin = 6 + 6 * n + 5 * n * lgn;
    } else {  // ICOMPZ = 2
      lwmin = 1;
      lrwmin = 1 + 4 * n + 2 * (n * n);
      liwmin = 3 + 5 * n;
    }
    WORK(1) = WorkSize<T>(lwmin);
    RWORK(1) = WorkSize<R>(lrwmin);
    IWORK(1) = liwmin;
    if (lwork < lwmin && !lquery) {
      info = -8;
    } else if (lrwork < lrwmin && !lquery) {
      info = -10;
    } else if (liwork < liwmin && !lquery) {
      info = -12;
    }
  }
  if (info != 0) {
    Xerbla<T>("STEDC", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  if (n == 1) {
    if (icompz != 0) Z(1, 1) = one;
    return info;
  }
  // If the following conditional clause is removed, then the routine
  // will use the Divide and Conquer routine to compute only the
  // eigenvalues, which requires (3N + 3N**2) real workspace and
  // (2 + 5N + 2N lg(N)) integer workspace.
  // Since on many architectures DSTERF is much faster than any other
  // algorithm for finding eigenvalues only, it is used here
  // as the default. If the conditional clause is removed, then
  // information on the size of workspace needs to be changed.
  // If COMPZ = 'N', use DSTERF to compute the eigenvalues.
  if (icompz == 0) {
    info = Sterf<R>(n, d, e);
    goto L70;
  }
  // If N is smaller than the minimum divide size (SMLSIZ+1), then
  // solve the problem with another solver.
  if (n <= smlsiz) {
    info = Steqr(compz, n, d, e, z, ldz, rwork);
  } else {
    // If COMPZ = 'I', we simply call DSTEDC instead.
    if (icompz == 2) {
      Laset<R>('F', n, n, zero, one, rwork, n);
      ll = n * n + 1;
      info = Stedc<R>('I', n, d, e, rwork, n, RWORK.ptr(ll), lrwork - ll + 1,
                      iwork, liwork);
      for (j = 1; j <= n; ++j) {
        for (i = 1; i <= n; ++i) {
          Z(i, j) = RWORK((j - 1) * n + i);
        }
      }
      goto L70;
    }
    // From now on, only option left to be handled is COMPZ = 'V',
    // i.e. ICOMPZ = 1.
    // Scale.
    orgnrm = Lanst<R>('M', n, d, e);
    if (orgnrm == zero) goto L70;
    eps = Lamch<R>('E');
    start = 1;
    // while ( START <= N )
  L30:
    if (start <= n) {
      // Let FINISH be the position of the next subdiagonal entry
      // such that E( FINISH ) <= TINY or FINISH = N if no such
      // subdiagonal exists.  The matrix identified by the elements
      // between START and FINISH constitutes an independent
      // sub-problem.
      finish = start;
    L40:
      if (finish < n) {
        tiny = eps * std::sqrt(std::abs(D(finish))) *
               std::sqrt(std::abs(D(finish + 1)));
        if (std::abs(E(finish)) > tiny) {
          finish = finish + 1;
          goto L40;
        }
      }
      // (Sub) Problem determined.  Compute its size and solve it.
      m = finish - start + 1;
      if (m > smlsiz) {
        // Scale.
        orgnrm = Lanst<R>('M', m, D.ptr(start), E.ptr(start));
        info = Lascl<R>('G', 0, 0, orgnrm, one, m, 1, D.ptr(start), m);
        info = Lascl<R>('G', 0, 0, orgnrm, one, m - 1, 1, E.ptr(start), m - 1);
        info = Laed0(n, m, D.ptr(start), E.ptr(start), Z.ptr(1, start), ldz,
                     work, n, rwork, iwork);
        if (info > 0) {
          info = (info / (m + 1) + start - 1) * (n + 1) + info % (m + 1) +
                 start - 1;
          goto L70;
        }
        // Scale back.
        info = Lascl<R>('G', 0, 0, one, orgnrm, m, 1, D.ptr(start), m);
      } else {
        info = Steqr<R>('I', m, D.ptr(start), E.ptr(start), rwork, m,
                        RWORK.ptr(m * m + 1));
        Lacrm(n, m, Z.ptr(1, start), ldz, rwork, m, work, n,
              RWORK.ptr(m * m + 1));
        Lacpy('A', n, m, work, n, Z.ptr(1, start), ldz);
        if (info > 0) {
          info = start * (n + 1) + finish;
          goto L70;
        }
      }
      start = finish + 1;
      goto L30;
    }
    // endwhile
    // Use Selection Sort to minimize swaps of eigenvectors
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
L70:
  WORK(1) = WorkSize<T>(lwmin);
  RWORK(1) = WorkSize<R>(lrwmin);
  IWORK(1) = liwmin;
  return info;
}
#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template void Laed5<R>(blasint, R*, R*, R*, R, R*);                          \
  template blasint Laed4<R>(blasint, blasint, R*, R*, R*, R, R*);              \
  template blasint Laed3<R>(blasint, blasint, blasint, R*, R*, blasint, R, R*, \
                            R*, blasint*, blasint*, R*, R*);                   \
  template blasint Laed2<R>(blasint*, blasint, blasint, R*, R*, blasint,       \
                            blasint*, R*, R*, R*, R*, R*, blasint*, blasint*,  \
                            blasint*, blasint*);                               \
  template blasint Laed1<R>(blasint, R*, R*, blasint, blasint*, R*, blasint,   \
                            R*, blasint*);                                     \
  template blasint Laeda<R>(blasint, blasint, blasint, blasint, blasint*,      \
                            blasint*, blasint*, blasint*, R*, R*, blasint*,    \
                            R*, R*);                                           \
  template blasint Laed9<R>(blasint, blasint, blasint, blasint, R*, R*,        \
                            blasint, R, R*, R*, R*, blasint);                  \
  template blasint Laed8<R>(blasint, blasint*, blasint, blasint, R*, R*,       \
                            blasint, blasint*, R*, blasint, R*, R*, R*,        \
                            blasint, R*, blasint*, blasint*, blasint*, R*,     \
                            blasint*, blasint*);                               \
  template blasint Laed7<R>(blasint, blasint, blasint, blasint, blasint,       \
                            blasint, R*, R*, blasint, blasint*, R*, blasint,   \
                            R*, blasint*, blasint*, blasint*, blasint*,        \
                            blasint*, R*, R*, blasint*);                       \
  template blasint Laed0<R>(blasint, blasint, blasint, R*, R*, R*, blasint,    \
                            R*, blasint, R*, blasint*);                        \
  template blasint Stedc<R>(char, blasint, R*, R*, R*, blasint, R*, blasint,   \
                            blasint*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                 \
  template blasint Laed8<T>(                                                  \
      blasint*, blasint, blasint, T*, blasint, RealT<T>*, RealT<T>*, blasint, \
      RealT<T>*, RealT<T>*, T*, blasint, RealT<T>*, blasint*, blasint*,       \
      blasint*, blasint*, blasint*, blasint*, RealT<T>*);                     \
  template blasint Laed7<T>(                                                  \
      blasint, blasint, blasint, blasint, blasint, blasint, RealT<T>*, T*,    \
      blasint, RealT<T>*, blasint*, RealT<T>*, blasint*, blasint*, blasint*,  \
      blasint*, blasint*, RealT<T>*, T*, RealT<T>*, blasint*);                \
  template blasint Laed0<T>(blasint, blasint, RealT<T>*, RealT<T>*, T*,       \
                            blasint, T*, blasint, RealT<T>*, blasint*);       \
  template blasint Stedc<T>(char, blasint, RealT<T>*, RealT<T>*, T*, blasint, \
                            T*, blasint, RealT<T>*, blasint, blasint*,        \
                            blasint);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_STEDC_REAL(p, R)                                        \
  HWYBLAS_API void p##laed5_(const hwyblas_int* i, R* d, R* z, R* delta,       \
                             const R* rho, R* dlam) {                          \
    hwyblas::lapack::Laed5(*i, d, z, delta, *rho, dlam);                       \
  }                                                                            \
  HWYBLAS_API void p##laed4_(const hwyblas_int* n, const hwyblas_int* i, R* d, \
                             R* z, R* delta, const R* rho, R* dlam,            \
                             hwyblas_int* info) {                              \
    *info = hwyblas::lapack::Laed4(*n, *i, d, z, delta, *rho, dlam);           \
  }                                                                            \
  HWYBLAS_API void p##laed3_(                                                  \
      const hwyblas_int* k, const hwyblas_int* n, const hwyblas_int* n1, R* d, \
      R* q, const hwyblas_int* ldq, const R* rho, R* dlambda, R* q2,           \
      hwyblas_int* indx, hwyblas_int* ctot, R* w, R* s, hwyblas_int* info) {   \
    *info = hwyblas::lapack::Laed3(*k, *n, *n1, d, q, *ldq, *rho, dlambda, q2, \
                                   indx, ctot, w, s);                          \
  }                                                                            \
  HWYBLAS_API void p##laed2_(                                                  \
      hwyblas_int* k, const hwyblas_int* n, const hwyblas_int* n1, R* d, R* q, \
      const hwyblas_int* ldq, hwyblas_int* indxq, R* rho, R* z, R* dlambda,    \
      R* w, R* q2, hwyblas_int* indx, hwyblas_int* indxc, hwyblas_int* indxp,  \
      hwyblas_int* coltyp, hwyblas_int* info) {                                \
    *info =                                                                    \
        hwyblas::lapack::Laed2(k, *n, *n1, d, q, *ldq, indxq, rho, z, dlambda, \
                               w, q2, indx, indxc, indxp, coltyp);             \
  }                                                                            \
  HWYBLAS_API void p##laed1_(const hwyblas_int* n, R* d, R* q,                 \
                             const hwyblas_int* ldq, hwyblas_int* indxq,       \
                             R* rho, const hwyblas_int* cutpnt, R* work,       \
                             hwyblas_int* iwork, hwyblas_int* info) {          \
    *info = hwyblas::lapack::Laed1(*n, d, q, *ldq, indxq, rho, *cutpnt, work,  \
                                   iwork);                                     \
  }                                                                            \
  HWYBLAS_API void p##laeda_(                                                  \
      const hwyblas_int* n, const hwyblas_int* tlvls,                          \
      const hwyblas_int* curlvl, const hwyblas_int* curpbm,                    \
      hwyblas_int* prmptr, hwyblas_int* perm, hwyblas_int* givptr,             \
      hwyblas_int* givcol, R* givnum, R* q, hwyblas_int* qptr, R* z, R* ztemp, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Laeda(*n, *tlvls, *curlvl, *curpbm, prmptr, perm, \
                                   givptr, givcol, givnum, q, qptr, z, ztemp); \
  }                                                                            \
  HWYBLAS_API void p##laed9_(const hwyblas_int* k, const hwyblas_int* kstart,  \
                             const hwyblas_int* kstop, const hwyblas_int* n,   \
                             R* d, R* q, const hwyblas_int* ldq, const R* rho, \
                             R* dlambda, R* w, R* s, const hwyblas_int* lds,   \
                             hwyblas_int* info) {                              \
    *info = hwyblas::lapack::Laed9(*k, *kstart, *kstop, *n, d, q, *ldq, *rho,  \
                                   dlambda, w, s, *lds);                       \
  }                                                                            \
  HWYBLAS_API void p##laed8_(                                                  \
      const hwyblas_int* icompq, hwyblas_int* k, const hwyblas_int* n,         \
      const hwyblas_int* qsiz, R* d, R* q, const hwyblas_int* ldq,             \
      hwyblas_int* indxq, R* rho, const hwyblas_int* cutpnt, R* z, R* dlambda, \
      R* q2, const hwyblas_int* ldq2, R* w, hwyblas_int* perm,                 \
      hwyblas_int* givptr, hwyblas_int* givcol, R* givnum, hwyblas_int* indxp, \
      hwyblas_int* indx, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Laed8(*icompq, k, *n, *qsiz, d, q, *ldq, indxq,   \
                                   rho, *cutpnt, z, dlambda, q2, *ldq2, w,     \
                                   perm, givptr, givcol, givnum, indxp, indx); \
  }                                                                            \
  HWYBLAS_API void p##laed7_(                                                  \
      const hwyblas_int* icompq, const hwyblas_int* n,                         \
      const hwyblas_int* qsiz, const hwyblas_int* tlvls,                       \
      const hwyblas_int* curlvl, const hwyblas_int* curpbm, R* d, R* q,        \
      const hwyblas_int* ldq, hwyblas_int* indxq, R* rho,                      \
      const hwyblas_int* cutpnt, R* qstore, hwyblas_int* qptr,                 \
      hwyblas_int* prmptr, hwyblas_int* perm, hwyblas_int* givptr,             \
      hwyblas_int* givcol, R* givnum, R* work, hwyblas_int* iwork,             \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Laed7(*icompq, *n, *qsiz, *tlvls, *curlvl,        \
                                   *curpbm, d, q, *ldq, indxq, rho, *cutpnt,   \
                                   qstore, qptr, prmptr, perm, givptr, givcol, \
                                   givnum, work, iwork);                       \
  }                                                                            \
  HWYBLAS_API void p##laed0_(const hwyblas_int* icompq,                        \
                             const hwyblas_int* qsiz, const hwyblas_int* n,    \
                             R* d, R* e, R* q, const hwyblas_int* ldq,         \
                             R* qstore, const hwyblas_int* ldqs, R* work,      \
                             hwyblas_int* iwork, hwyblas_int* info) {          \
    *info = hwyblas::lapack::Laed0(*icompq, *qsiz, *n, d, e, q, *ldq, qstore,  \
                                   *ldqs, work, iwork);                        \
  }                                                                            \
  HWYBLAS_API void p##stedc_(const char* compz, const hwyblas_int* n, R* d,    \
                             R* e, R* z, const hwyblas_int* ldz, R* work,      \
                             const hwyblas_int* lwork, hwyblas_int* iwork,     \
                             const hwyblas_int* liwork, hwyblas_int* info) {   \
    *info = hwyblas::lapack::Stedc(*compz, *n, d, e, z, *ldz, work, *lwork,    \
                                   iwork, *liwork);                            \
  }

HWYBLAS_LAPACK_STEDC_REAL(s, float)
HWYBLAS_LAPACK_STEDC_REAL(d, double)

#undef HWYBLAS_LAPACK_STEDC_REAL

#define HWYBLAS_LAPACK_STEDC_COMPLEX(p, T, R)                                  \
  HWYBLAS_API void p##laed8_(                                                  \
      hwyblas_int* k, const hwyblas_int* n, const hwyblas_int* qsiz, R* q,     \
      const hwyblas_int* ldq, R* d, R* rho, const hwyblas_int* cutpnt, R* z,   \
      R* dlambda, R* q2, const hwyblas_int* ldq2, R* w, hwyblas_int* indxp,    \
      hwyblas_int* indx, hwyblas_int* indxq, hwyblas_int* perm,                \
      hwyblas_int* givptr, hwyblas_int* givcol, R* givnum,                     \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Laed8(                                            \
        k, *n, *qsiz, As<T>(q), *ldq, d, rho, *cutpnt, z, dlambda, As<T>(q2),  \
        *ldq2, w, indxp, indx, indxq, perm, givptr, givcol, givnum);           \
  }                                                                            \
  HWYBLAS_API void p##laed7_(                                                  \
      const hwyblas_int* n, const hwyblas_int* cutpnt,                         \
      const hwyblas_int* qsiz, const hwyblas_int* tlvls,                       \
      const hwyblas_int* curlvl, const hwyblas_int* curpbm, R* d, R* q,        \
      const hwyblas_int* ldq, R* rho, hwyblas_int* indxq, R* qstore,           \
      hwyblas_int* qptr, hwyblas_int* prmptr, hwyblas_int* perm,               \
      hwyblas_int* givptr, hwyblas_int* givcol, R* givnum, R* work, R* rwork,  \
      hwyblas_int* iwork, hwyblas_int* info) {                                 \
    *info = hwyblas::lapack::Laed7(*n, *cutpnt, *qsiz, *tlvls, *curlvl,        \
                                   *curpbm, d, As<T>(q), *ldq, rho, indxq,     \
                                   qstore, qptr, prmptr, perm, givptr, givcol, \
                                   givnum, As<T>(work), rwork, iwork);         \
  }                                                                            \
  HWYBLAS_API void p##laed0_(const hwyblas_int* qsiz, const hwyblas_int* n,    \
                             R* d, R* e, R* q, const hwyblas_int* ldq,         \
                             R* qstore, const hwyblas_int* ldqs, R* rwork,     \
                             hwyblas_int* iwork, hwyblas_int* info) {          \
    *info = hwyblas::lapack::Laed0(*qsiz, *n, d, e, As<T>(q), *ldq,            \
                                   As<T>(qstore), *ldqs, rwork, iwork);        \
  }                                                                            \
  HWYBLAS_API void p##stedc_(const char* compz, const hwyblas_int* n, R* d,    \
                             R* e, R* z, const hwyblas_int* ldz, R* work,      \
                             const hwyblas_int* lwork, R* rwork,               \
                             const hwyblas_int* lrwork, hwyblas_int* iwork,    \
                             const hwyblas_int* liwork, hwyblas_int* info) {   \
    *info =                                                                    \
        hwyblas::lapack::Stedc(*compz, *n, d, e, As<T>(z), *ldz, As<T>(work),  \
                               *lwork, rwork, *lrwork, iwork, *liwork);        \
  }

HWYBLAS_LAPACK_STEDC_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_STEDC_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_STEDC_COMPLEX

}  // extern "C"
