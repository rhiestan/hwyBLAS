// The singular value decomposition of a real bidiagonal matrix by divide
// and conquer: xBDSDC (driver), xLASD0 (singular vectors) and xLASDA
// (singular values and the factored form of the singular vectors), which
// split the matrix with the tree of xLASDT, solve the leaves with xLASDQ
// (xBDSQR on a possibly non-square bidiagonal matrix) and merge the
// subproblems bottom-up: xLASD1 / xLASD6 merge two of them (deflating with
// xLASD2 / xLASD7, then solving the secular equation and updating the
// vectors with xLASD3 / xLASD8), xLASD4 finds one root of the secular
// equation (xLASD5 for the 2-by-2 case, xLAED6 for the step of the
// middle-way method), and xLAMRG merges two sorted lists. Real R only.
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
// xLAMRG: the permutation INDEX that merges the two sorted lists
// A(1:N1) and A(N1+1:N1+N2) into one list in ascending order:
// A(INDEX(1)) <= ... <= A(INDEX(N1+N2)). DTRD1 (DTRD2) is 1 if the first
// (second) list is in ascending, -1 if it is in descending order.
template <typename R>
void Lamrg(blasint n1, blasint n2, const R* a, blasint dtrd1, blasint dtrd2,
           blasint* index) {
  const Vector<const R> A(a);
  const Vector<blasint> INDEX(index);
  blasint i, ind1, ind2, n1sv, n2sv;
  n1sv = n1;
  n2sv = n2;
  if (dtrd1 > 0) {
    ind1 = 1;
  } else {
    ind1 = n1;
  }
  if (dtrd2 > 0) {
    ind2 = 1 + n1;
  } else {
    ind2 = n1 + n2;
  }
  i = 1;
  // while ( (N1SV > 0) & (N2SV > 0) )
L10:
  if (n1sv > 0 && n2sv > 0) {
    if (A(ind1) <= A(ind2)) {
      INDEX(i) = ind1;
      i = i + 1;
      ind1 = ind1 + dtrd1;
      n1sv = n1sv - 1;
    } else {
      INDEX(i) = ind2;
      i = i + 1;
      ind2 = ind2 + dtrd2;
      n2sv = n2sv - 1;
    }
    goto L10;
  }
  // end while
  if (n1sv == 0) {
    for (n1sv = 1; n1sv <= n2sv; ++n1sv) {
      INDEX(i) = ind2;
      i = i + 1;
      ind2 = ind2 + dtrd2;
    }
  } else {
    // N2SV .EQ. 0
    for (n2sv = 1; n2sv <= n1sv; ++n2sv) {
      INDEX(i) = ind1;
      i = i + 1;
      ind1 = ind1 + dtrd1;
    }
  }
}

// xLASDT: the tree of subproblems for divide and conquer on a bidiagonal
// matrix of order N with subproblems of at most MSUB + 1 rows (in Fortran
// a routine with real arithmetic, so R is kept). Returns the number of
// levels LVL and of nodes ND; node i has the center row INODE(i) and the
// NDIML(i) and NDIMR(i) rows of its left and right child.
template <typename R>
void Lasdt(blasint n, blasint* lvl, blasint* nd, blasint* inode, blasint* ndiml,
           blasint* ndimr, blasint msub) {
  const R two = R(2);
  const Vector<blasint> INODE(inode);
  const Vector<blasint> NDIML(ndiml);
  const Vector<blasint> NDIMR(ndimr);
  blasint i, il, ir, llst, maxn, ncrnt, nlvl;
  R temp;
  maxn = Max<blasint>(1, n);
  temp = std::log(R(maxn) / R(msub + 1)) / std::log(two);
  *lvl = static_cast<blasint>(temp) + 1;
  i = n / 2;
  INODE(1) = i + 1;
  NDIML(1) = i;
  NDIMR(1) = n - i - 1;
  il = 0;
  ir = 1;
  llst = 1;
  for (nlvl = 1; nlvl <= *lvl - 1; ++nlvl) {
    // Constructing the tree at (NLVL+1)-st level. The number of
    // nodes created on this level is LLST * 2.
    for (i = 0; i <= llst - 1; ++i) {
      il = il + 2;
      ir = ir + 2;
      ncrnt = llst + i;
      NDIML(il) = NDIML(ncrnt) / 2;
      NDIMR(il) = NDIML(ncrnt) - NDIML(il) - 1;
      INODE(il) = INODE(ncrnt) - NDIMR(il) - 1;
      NDIML(ir) = NDIMR(ncrnt) / 2;
      NDIMR(ir) = NDIMR(ncrnt) - NDIML(ir) - 1;
      INODE(ir) = INODE(ncrnt) + NDIML(ir) + 1;
    }
    llst = llst * 2;
  }
  *nd = llst * 2 - 1;
}

// xLAED6: one Newton step (of the Gragg-Thornton-Warner cubic convergent
// scheme, iterated to convergence) for the root TAU of the secular
// equation F(x) = RHO + Z(1)/(D(1)-x) + Z(2)/(D(2)-x) + Z(3)/(D(3)-x) in
// the interval (D(2), D(3)) if ORGATI is true, else in (D(1), D(2)), with
// FINIT = F(0). KNITER is 2 for the first call of xLASD4's iteration.
// Returns INFO = 1 if it did not converge.
template <typename R>
blasint Laed6(blasint kniter, bool orgati, R rho, const R* d, const R* z,
              R finit, R* tau) {
  const blasint maxit = 40;
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R three = R(3);
  const R four = R(4);
  const R eight = R(8);
  const Vector<const R> D(d);
  R dscale[3];
  const Vector<R> DSCALE(dscale);
  const Vector<const R> Z(z);
  R zscale[3];
  const Vector<R> ZSCALE(zscale);
  blasint i, iter, niter;
  bool scale;
  R sclinv = one;
  R a, b, base, c, ddf, df, eps, erretm, eta, f, fc, lbd, sclfac, small1,
      small2, sminv1, sminv2, temp, temp1, temp2, temp3, temp4, ubd;
  blasint info = 0;
  if (orgati) {
    lbd = D(2);
    ubd = D(3);
  } else {
    lbd = D(1);
    ubd = D(2);
  }
  if (finit < zero) {
    lbd = zero;
  } else {
    ubd = zero;
  }
  niter = 1;
  *tau = zero;
  if (kniter == 2) {
    if (orgati) {
      temp = (D(3) - D(2)) / two;
      c = rho + Z(1) / ((D(1) - D(2)) - temp);
      a = c * (D(2) + D(3)) + Z(2) + Z(3);
      b = c * D(2) * D(3) + Z(2) * D(3) + Z(3) * D(2);
    } else {
      temp = (D(1) - D(2)) / two;
      c = rho + Z(3) / ((D(3) - D(2)) - temp);
      a = c * (D(1) + D(2)) + Z(1) + Z(2);
      b = c * D(1) * D(2) + Z(1) * D(2) + Z(2) * D(1);
    }
    temp = Max(std::abs(a), std::abs(b), std::abs(c));
    a = a / temp;
    b = b / temp;
    c = c / temp;
    if (c == zero) {
      *tau = b / a;
    } else if (a <= zero) {
      *tau = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
    } else {
      *tau = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
    }
    if (*tau < lbd || *tau > ubd) *tau = (lbd + ubd) / two;
    if (D(1) == *tau || D(2) == *tau || D(3) == *tau) {
      *tau = zero;
    } else {
      temp = finit + *tau * Z(1) / (D(1) * (D(1) - *tau)) +
             *tau * Z(2) / (D(2) * (D(2) - *tau)) +
             *tau * Z(3) / (D(3) * (D(3) - *tau));
      if (temp <= zero) {
        lbd = *tau;
      } else {
        ubd = *tau;
      }
      if (std::abs(finit) <= std::abs(temp)) *tau = zero;
    }
  }
  // get machine parameters for possible scaling to avoid overflow
  // modified by Sven: parameters SMALL1, SMINV1, SMALL2,
  // SMINV2, EPS are not SAVEd anymore between one call to the
  // others but recomputed at each call
  eps = Lamch<R>('E');
  base = Lamch<R>('B');
  // (A power of the base: std::pow returns it exactly, as a double for
  // float R.)
  small1 = static_cast<R>(
      std::pow(base, (static_cast<blasint>(std::log(Lamch<R>('S')) /
                                           std::log(base) / three))));
  sminv1 = one / small1;
  small2 = small1 * small1;
  sminv2 = sminv1 * sminv1;
  // Determine if scaling of inputs necessary to avoid overflow
  // when computing 1/TEMP**3
  if (orgati) {
    temp = Min(std::abs(D(2) - *tau), std::abs(D(3) - *tau));
  } else {
    temp = Min(std::abs(D(1) - *tau), std::abs(D(2) - *tau));
  }
  scale = false;
  if (temp <= small1) {
    scale = true;
    if (temp <= small2) {
      // Scale up by power of radix nearest 1/SAFMIN**(2/3)
      sclfac = sminv2;
      sclinv = small2;
    } else {
      // Scale up by power of radix nearest 1/SAFMIN**(1/3)
      sclfac = sminv1;
      sclinv = small1;
    }
    // Scaling up safe because D, Z, TAU scaled elsewhere to be O(1)
    for (i = 1; i <= 3; ++i) {
      DSCALE(i) = D(i) * sclfac;
      ZSCALE(i) = Z(i) * sclfac;
    }
    *tau = *tau * sclfac;
    lbd = lbd * sclfac;
    ubd = ubd * sclfac;
  } else {
    // Copy D and Z to DSCALE and ZSCALE
    for (i = 1; i <= 3; ++i) {
      DSCALE(i) = D(i);
      ZSCALE(i) = Z(i);
    }
  }
  fc = zero;
  df = zero;
  ddf = zero;
  for (i = 1; i <= 3; ++i) {
    temp = one / (DSCALE(i) - *tau);
    temp1 = ZSCALE(i) * temp;
    temp2 = temp1 * temp;
    temp3 = temp2 * temp;
    fc = fc + temp1 / DSCALE(i);
    df = df + temp2;
    ddf = ddf + temp3;
  }
  f = finit + *tau * fc;
  if (std::abs(f) <= zero) goto L60;
  if (f <= zero) {
    lbd = *tau;
  } else {
    ubd = *tau;
  }
  // Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
  // scheme
  // It is not hard to see that
  // 1) Iterations will go up monotonically
  // if FINIT < 0;
  // 2) Iterations will go down monotonically
  // if FINIT > 0.
  iter = niter + 1;
  for (niter = iter; niter <= maxit; ++niter) {
    if (orgati) {
      temp1 = DSCALE(2) - *tau;
      temp2 = DSCALE(3) - *tau;
    } else {
      temp1 = DSCALE(1) - *tau;
      temp2 = DSCALE(2) - *tau;
    }
    a = (temp1 + temp2) * f - temp1 * temp2 * df;
    b = temp1 * temp2 * f;
    c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
    temp = Max(std::abs(a), std::abs(b), std::abs(c));
    a = a / temp;
    b = b / temp;
    c = c / temp;
    if (c == zero) {
      eta = b / a;
    } else if (a <= zero) {
      eta = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
    } else {
      eta = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
    }
    if (f * eta >= zero) {
      eta = -f / df;
    }
    *tau = *tau + eta;
    if (*tau < lbd || *tau > ubd) *tau = (lbd + ubd) / two;
    fc = zero;
    erretm = zero;
    df = zero;
    ddf = zero;
    for (i = 1; i <= 3; ++i) {
      if ((DSCALE(i) - *tau) != zero) {
        temp = one / (DSCALE(i) - *tau);
        temp1 = ZSCALE(i) * temp;
        temp2 = temp1 * temp;
        temp3 = temp2 * temp;
        temp4 = temp1 / DSCALE(i);
        fc = fc + temp4;
        erretm = erretm + std::abs(temp4);
        df = df + temp2;
        ddf = ddf + temp3;
      } else {
        goto L60;
      }
    }
    f = finit + *tau * fc;
    erretm = eight * (std::abs(finit) + std::abs(*tau) * erretm) +
             std::abs(*tau) * df;
    if ((std::abs(f) <= four * eps * erretm) ||
        ((ubd - lbd) <= four * eps * std::abs(*tau)))
      goto L60;
    if (f <= zero) {
      lbd = *tau;
    } else {
      ubd = *tau;
    }
  }
  info = 1;
L60:
  // Undo scaling
  if (scale) *tau = *tau * sclinv;
  return info;
}

// xLASD5: the I-th (I = 1 or 2) square root DSIGMA of the eigenvalues of
// the 2-by-2 matrix diag(D)**2 + RHO * Z * Z**T, D(1) < D(2), with
// DELTA(j) = D(j) - DSIGMA and WORK(j) = D(j) + DSIGMA.
template <typename R>
void Lasd5(blasint i, const R* d, const R* z, R* delta, R rho, R* dsigma,
           R* work) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R three = R(3);
  const R four = R(4);
  const Vector<const R> D(d);
  const Vector<R> DELTA(delta);
  const Vector<R> WORK(work);
  const Vector<const R> Z(z);
  R b, c, del, delsq, tau, w;
  del = D(2) - D(1);
  delsq = del * (D(2) + D(1));
  if (i == 1) {
    w = one + four * rho *
                  (Z(2) * Z(2) / (D(1) + three * D(2)) -
                   Z(1) * Z(1) / (three * D(1) + D(2))) /
                  del;
    if (w > zero) {
      b = delsq + rho * (Z(1) * Z(1) + Z(2) * Z(2));
      c = rho * Z(1) * Z(1) * delsq;
      // B > ZERO, always
      // The following TAU is DSIGMA * DSIGMA - D( 1 ) * D( 1 )
      tau = two * c / (b + std::sqrt(std::abs(b * b - four * c)));
      // The following TAU is DSIGMA - D( 1 )
      tau = tau / (D(1) + std::sqrt(D(1) * D(1) + tau));
      *dsigma = D(1) + tau;
      DELTA(1) = -tau;
      DELTA(2) = del - tau;
      WORK(1) = two * D(1) + tau;
      WORK(2) = (D(1) + tau) + D(2);
      // DELTA( 1 ) = -Z( 1 ) / TAU
      // DELTA( 2 ) = Z( 2 ) / ( DEL-TAU )
    } else {
      b = -delsq + rho * (Z(1) * Z(1) + Z(2) * Z(2));
      c = rho * Z(2) * Z(2) * delsq;
      // The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 )
      if (b > zero) {
        tau = -two * c / (b + std::sqrt(b * b + four * c));
      } else {
        tau = (b - std::sqrt(b * b + four * c)) / two;
      }
      // The following TAU is DSIGMA - D( 2 )
      tau = tau / (D(2) + std::sqrt(std::abs(D(2) * D(2) + tau)));
      *dsigma = D(2) + tau;
      DELTA(1) = -(del + tau);
      DELTA(2) = -tau;
      WORK(1) = D(1) + tau + D(2);
      WORK(2) = two * D(2) + tau;
      // DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
      // DELTA( 2 ) = -Z( 2 ) / TAU
    }
    // TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
    // DELTA( 1 ) = DELTA( 1 ) / TEMP
    // DELTA( 2 ) = DELTA( 2 ) / TEMP
  } else {
    // Now I=2
    b = -delsq + rho * (Z(1) * Z(1) + Z(2) * Z(2));
    c = rho * Z(2) * Z(2) * delsq;
    // The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 )
    if (b > zero) {
      tau = (b + std::sqrt(b * b + four * c)) / two;
    } else {
      tau = two * c / (-b + std::sqrt(b * b + four * c));
    }
    // The following TAU is DSIGMA - D( 2 )
    tau = tau / (D(2) + std::sqrt(D(2) * D(2) + tau));
    *dsigma = D(2) + tau;
    DELTA(1) = -(del + tau);
    DELTA(2) = -tau;
    WORK(1) = D(1) + tau + D(2);
    WORK(2) = two * D(2) + tau;
    // DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
    // DELTA( 2 ) = -Z( 2 ) / TAU
    // TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
    // DELTA( 1 ) = DELTA( 1 ) / TEMP
    // DELTA( 2 ) = DELTA( 2 ) / TEMP
  }
}

// xLASD4: the I-th square root SIGMA of the eigenvalues of
// diag(D)**2 + RHO * Z * Z**T, with 0 <= D(1) < ... < D(N), RHO > 0 and
// norm(Z) = 1: the root of the secular equation, found by the middle-way
// method of Li. DELTA(j) = D(j) - SIGMA and WORK(j) = D(j) + SIGMA.
// Returns INFO = 1 if it did not converge.
template <typename R>
blasint Lasd4(blasint n, blasint i, const R* d, const R* z, R* delta, R rho,
              R* sigma, R* work) {
  const blasint maxit = 400;
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R three = R(3);
  const R four = R(4);
  const R eight = R(8);
  const R ten = R(10);
  const Vector<const R> D(d);
  R dd[3];
  const Vector<R> DD(dd);
  const Vector<R> DELTA(delta);
  const Vector<R> WORK(work);
  const Vector<const R> Z(z);
  R zz[3];
  const Vector<R> ZZ(zz);
  blasint ii, iim1, iip1, ip1, iter, j, niter;
  bool geomavg, orgati, swtch, swtch3;
  R a, b, c, delsq, delsq2, dphi, dpsi, dtiim, dtiip, dtipsq, dtisq, dtnsq,
      dtnsq1, dw, eps, erretm, eta, phi, prew, psi, rhoinv, sglb, sgub, sq2,
      tau, tau2, temp, temp1, temp2, w;
  blasint info = 0;
  if (n == 1) {
    // Presumably, I=1 upon entry
    *sigma = std::sqrt(D(1) * D(1) + rho * Z(1) * Z(1));
    DELTA(1) = one;
    WORK(1) = one;
    return info;
  }
  if (n == 2) {
    Lasd5(i, d, z, delta, rho, sigma, work);
    return info;
  }
  // Compute machine epsilon
  eps = Lamch<R>('E');
  rhoinv = one / rho;
  tau2 = zero;
  // The case I = N
  if (i == n) {
    // Initialize some basic variables
    ii = n - 1;
    niter = 1;
    // Calculate initial guess
    temp = rho / two;
    // If ||Z||_2 is not one, then TEMP should be set to
    // RHO * ||Z||_2^2 / TWO
    temp1 = temp / (D(n) + std::sqrt(D(n) * D(n) + temp));
    for (j = 1; j <= n; ++j) {
      WORK(j) = D(j) + D(n) + temp1;
      DELTA(j) = (D(j) - D(n)) - temp1;
    }
    psi = zero;
    for (j = 1; j <= n - 2; ++j) {
      psi = psi + Z(j) * Z(j) / (DELTA(j) * WORK(j));
    }
    c = rhoinv + psi;
    w = c + Z(ii) * Z(ii) / (DELTA(ii) * WORK(ii)) +
        Z(n) * Z(n) / (DELTA(n) * WORK(n));
    if (w <= zero) {
      temp1 = std::sqrt(D(n) * D(n) + rho);
      temp =
          Z(n - 1) * Z(n - 1) /
              ((D(n - 1) + temp1) * (D(n) - D(n - 1) + rho / (D(n) + temp1))) +
          Z(n) * Z(n) / rho;
      // The following TAU2 is to approximate
      // SIGMA_n^2 - D( N )*D( N )
      if (c <= temp) {
        tau = rho;
      } else {
        delsq = (D(n) - D(n - 1)) * (D(n) + D(n - 1));
        a = -c * delsq + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
        b = Z(n) * Z(n) * delsq;
        if (a < zero) {
          tau2 = two * b / (std::sqrt(a * a + four * b * c) - a);
        } else {
          tau2 = (a + std::sqrt(a * a + four * b * c)) / (two * c);
        }
        tau = tau2 / (D(n) + std::sqrt(D(n) * D(n) + tau2));
      }
      // It can be proved that
      // D(N)^2+RHO/2 <= SIGMA_n^2 < D(N)^2+TAU2 <= D(N)^2+RHO
    } else {
      delsq = (D(n) - D(n - 1)) * (D(n) + D(n - 1));
      a = -c * delsq + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
      b = Z(n) * Z(n) * delsq;
      // The following TAU2 is to approximate
      // SIGMA_n^2 - D( N )*D( N )
      if (a < zero) {
        tau2 = two * b / (std::sqrt(a * a + four * b * c) - a);
      } else {
        tau2 = (a + std::sqrt(a * a + four * b * c)) / (two * c);
      }
      tau = tau2 / (D(n) + std::sqrt(D(n) * D(n) + tau2));
      // It can be proved that
      // D(N)^2 < D(N)^2+TAU2 < SIGMA(N)^2 < D(N)^2+RHO/2
    }
    // The following TAU is to approximate SIGMA_n - D( N )
    // TAU = TAU2 / ( D( N )+SQRT( D( N )*D( N )+TAU2 ) )
    *sigma = D(n) + tau;
    for (j = 1; j <= n; ++j) {
      DELTA(j) = (D(j) - D(n)) - tau;
      WORK(j) = D(j) + D(n) + tau;
    }
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= ii; ++j) {
      temp = Z(j) / (DELTA(j) * WORK(j));
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    temp = Z(n) / (DELTA(n) * WORK(n));
    phi = Z(n) * temp;
    dphi = temp * temp;
    erretm = eight * (-phi - psi) + erretm - phi + rhoinv;
    // $          + ABS( TAU2 )*( DPSI+DPHI )
    w = rhoinv + phi + psi;
    // Test for convergence
    if (std::abs(w) <= eps * erretm) {
      goto L240;
    }
    // Calculate the new step
    niter = niter + 1;
    dtnsq1 = WORK(n - 1) * DELTA(n - 1);
    dtnsq = WORK(n) * DELTA(n);
    c = w - dtnsq1 * dpsi - dtnsq * dphi;
    a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
    b = dtnsq * dtnsq1 * w;
    if (c < zero) c = std::abs(c);
    if (c == zero) {
      eta = rho - *sigma * *sigma;
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
    temp = eta - dtnsq;
    if (temp > rho) eta = rho + dtnsq;
    eta = eta / (*sigma + std::sqrt(eta + *sigma * *sigma));
    tau = tau + eta;
    *sigma = *sigma + eta;
    for (j = 1; j <= n; ++j) {
      DELTA(j) = DELTA(j) - eta;
      WORK(j) = WORK(j) + eta;
    }
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= ii; ++j) {
      temp = Z(j) / (WORK(j) * DELTA(j));
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    tau2 = WORK(n) * DELTA(n);
    temp = Z(n) / tau2;
    phi = Z(n) * temp;
    dphi = temp * temp;
    erretm = eight * (-phi - psi) + erretm - phi + rhoinv;
    // $          + ABS( TAU2 )*( DPSI+DPHI )
    w = rhoinv + phi + psi;
    // Main loop to update the values of the array   DELTA
    iter = niter + 1;
    for (niter = iter; niter <= maxit; ++niter) {
      // Test for convergence
      if (std::abs(w) <= eps * erretm) {
        goto L240;
      }
      // Calculate the new step
      dtnsq1 = WORK(n - 1) * DELTA(n - 1);
      dtnsq = WORK(n) * DELTA(n);
      c = w - dtnsq1 * dpsi - dtnsq * dphi;
      a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
      b = dtnsq1 * dtnsq * w;
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
      temp = eta - dtnsq;
      if (temp <= zero) eta = eta / two;
      eta = eta / (*sigma + std::sqrt(eta + *sigma * *sigma));
      tau = tau + eta;
      *sigma = *sigma + eta;
      for (j = 1; j <= n; ++j) {
        DELTA(j) = DELTA(j) - eta;
        WORK(j) = WORK(j) + eta;
      }
      // Evaluate PSI and the derivative DPSI
      dpsi = zero;
      psi = zero;
      erretm = zero;
      for (j = 1; j <= ii; ++j) {
        temp = Z(j) / (WORK(j) * DELTA(j));
        psi = psi + Z(j) * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
      }
      erretm = std::abs(erretm);
      // Evaluate PHI and the derivative DPHI
      tau2 = WORK(n) * DELTA(n);
      temp = Z(n) / tau2;
      phi = Z(n) * temp;
      dphi = temp * temp;
      erretm = eight * (-phi - psi) + erretm - phi + rhoinv;
      // $             + ABS( TAU2 )*( DPSI+DPHI )
      w = rhoinv + phi + psi;
    }
    // Return with INFO = 1, NITER = MAXIT and not converged
    info = 1;
    goto L240;
    // End for the case I = N
  } else {
    // The case for I < N
    niter = 1;
    ip1 = i + 1;
    // Calculate initial guess
    delsq = (D(ip1) - D(i)) * (D(ip1) + D(i));
    delsq2 = delsq / two;
    sq2 = std::sqrt((D(i) * D(i) + D(ip1) * D(ip1)) / two);
    temp = delsq2 / (D(i) + sq2);
    for (j = 1; j <= n; ++j) {
      WORK(j) = D(j) + D(i) + temp;
      DELTA(j) = (D(j) - D(i)) - temp;
    }
    psi = zero;
    for (j = 1; j <= i - 1; ++j) {
      psi = psi + Z(j) * Z(j) / (WORK(j) * DELTA(j));
    }
    phi = zero;
    for (j = n; j >= i + 2; --j) {
      phi = phi + Z(j) * Z(j) / (WORK(j) * DELTA(j));
    }
    c = rhoinv + psi + phi;
    w = c + Z(i) * Z(i) / (WORK(i) * DELTA(i)) +
        Z(ip1) * Z(ip1) / (WORK(ip1) * DELTA(ip1));
    geomavg = false;
    if (w > zero) {
      // d(i)^2 < the ith *sigma^2 < (d(i)^2+d(i+1)^2)/2
      // We choose d(i) as origin.
      orgati = true;
      ii = i;
      sglb = zero;
      sgub = delsq2 / (D(i) + sq2);
      a = c * delsq + Z(i) * Z(i) + Z(ip1) * Z(ip1);
      b = Z(i) * Z(i) * delsq;
      if (a > zero) {
        tau2 = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
      } else {
        tau2 = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
      }
      // TAU2 now is an estimation of SIGMA^2 - D( I )^2. The
      // following, however, is the corresponding estimation of
      // SIGMA - D( I ).
      tau = tau2 / (D(i) + std::sqrt(D(i) * D(i) + tau2));
      temp = std::sqrt(eps);
      if ((D(i) <= temp * D(ip1)) && (std::abs(Z(i)) <= temp) &&
          (D(i) > zero)) {
        tau = Min(ten * D(i), sgub);
        geomavg = true;
      }
    } else {
      // (d(i)^2+d(i+1)^2)/2 <= the ith *sigma^2 < d(i+1)^2/2
      // We choose d(i+1) as origin.
      orgati = false;
      ii = ip1;
      sglb = -delsq2 / (D(ii) + sq2);
      sgub = zero;
      a = c * delsq - Z(i) * Z(i) - Z(ip1) * Z(ip1);
      b = Z(ip1) * Z(ip1) * delsq;
      if (a < zero) {
        tau2 = two * b / (a - std::sqrt(std::abs(a * a + four * b * c)));
      } else {
        tau2 = -(a + std::sqrt(std::abs(a * a + four * b * c))) / (two * c);
      }
      // TAU2 now is an estimation of SIGMA^2 - D( IP1 )^2. The
      // following, however, is the corresponding estimation of
      // SIGMA - D( IP1 ).
      tau = tau2 / (D(ip1) + std::sqrt(std::abs(D(ip1) * D(ip1) + tau2)));
    }
    *sigma = D(ii) + tau;
    for (j = 1; j <= n; ++j) {
      WORK(j) = D(j) + D(ii) + tau;
      DELTA(j) = (D(j) - D(ii)) - tau;
    }
    iim1 = ii - 1;
    iip1 = ii + 1;
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= iim1; ++j) {
      temp = Z(j) / (WORK(j) * DELTA(j));
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    dphi = zero;
    phi = zero;
    for (j = n; j >= iip1; --j) {
      temp = Z(j) / (WORK(j) * DELTA(j));
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
    temp = Z(ii) / (WORK(ii) * DELTA(ii));
    dw = dpsi + dphi + temp * temp;
    temp = Z(ii) * temp;
    w = w + temp;
    erretm =
        eight * (phi - psi) + erretm + two * rhoinv + three * std::abs(temp);
    // $          + ABS( TAU2 )*DW
    // Test for convergence
    if (std::abs(w) <= eps * erretm) {
      goto L240;
    }
    if (w <= zero) {
      sglb = Max(sglb, tau);
    } else {
      sgub = Min(sgub, tau);
    }
    // Calculate the new step
    niter = niter + 1;
    if (!swtch3) {
      dtipsq = WORK(ip1) * DELTA(ip1);
      dtisq = WORK(i) * DELTA(i);
      if (orgati) {
        c = w - dtipsq * dw + delsq * ((Z(i) / dtisq) * (Z(i) / dtisq));
      } else {
        c = w - dtisq * dw - delsq * ((Z(ip1) / dtipsq) * (Z(ip1) / dtipsq));
      }
      a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
      b = dtipsq * dtisq * w;
      if (c == zero) {
        if (a == zero) {
          if (orgati) {
            a = Z(i) * Z(i) + dtipsq * dtipsq * (dpsi + dphi);
          } else {
            a = Z(ip1) * Z(ip1) + dtisq * dtisq * (dpsi + dphi);
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
      dtiim = WORK(iim1) * DELTA(iim1);
      dtiip = WORK(iip1) * DELTA(iip1);
      temp = rhoinv + psi + phi;
      if (orgati) {
        temp1 = Z(iim1) / dtiim;
        temp1 = temp1 * temp1;
        c = (temp - dtiip * (dpsi + dphi)) -
            (D(iim1) - D(iip1)) * (D(iim1) + D(iip1)) * temp1;
        ZZ(1) = Z(iim1) * Z(iim1);
        if (dpsi < temp1) {
          ZZ(3) = dtiip * dtiip * dphi;
        } else {
          ZZ(3) = dtiip * dtiip * ((dpsi - temp1) + dphi);
        }
      } else {
        temp1 = Z(iip1) / dtiip;
        temp1 = temp1 * temp1;
        c = (temp - dtiim * (dpsi + dphi)) -
            (D(iip1) - D(iim1)) * (D(iim1) + D(iip1)) * temp1;
        if (dphi < temp1) {
          ZZ(1) = dtiim * dtiim * dpsi;
        } else {
          ZZ(1) = dtiim * dtiim * (dpsi + (dphi - temp1));
        }
        ZZ(3) = Z(iip1) * Z(iip1);
      }
      ZZ(2) = Z(ii) * Z(ii);
      DD(1) = dtiim;
      DD(2) = DELTA(ii) * WORK(ii);
      DD(3) = dtiip;
      info = Laed6(niter, orgati, c, dd, zz, w, &eta);
      if (info != 0) {
        // If INFO is not 0, i.e., DLAED6 failed, switch back
        // to 2 pole interpolation.
        swtch3 = false;
        info = 0;
        dtipsq = WORK(ip1) * DELTA(ip1);
        dtisq = WORK(i) * DELTA(i);
        if (orgati) {
          c = w - dtipsq * dw + delsq * ((Z(i) / dtisq) * (Z(i) / dtisq));
        } else {
          c = w - dtisq * dw - delsq * ((Z(ip1) / dtipsq) * (Z(ip1) / dtipsq));
        }
        a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
        b = dtipsq * dtisq * w;
        if (c == zero) {
          if (a == zero) {
            if (orgati) {
              a = Z(i) * Z(i) + dtipsq * dtipsq * (dpsi + dphi);
            } else {
              a = Z(ip1) * Z(ip1) + dtisq * dtisq * (dpsi + dphi);
            }
          }
          eta = b / a;
        } else if (a <= zero) {
          eta = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
        } else {
          eta = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
        }
      }
    }
    // Note, eta should be positive if w is negative, and
    // eta should be negative otherwise. However,
    // if for some reason caused by roundoff, eta*w > 0,
    // we simply use one Newton step instead. This way
    // will guarantee eta*w < 0.
    if (w * eta >= zero) eta = -w / dw;
    eta = eta / (*sigma + std::sqrt(*sigma * *sigma + eta));
    temp = tau + eta;
    if (temp > sgub || temp < sglb) {
      if (w < zero) {
        eta = (sgub - tau) / two;
      } else {
        eta = (sglb - tau) / two;
      }
      if (geomavg) {
        if (w < zero) {
          if (tau > zero) {
            eta = std::sqrt(sgub * tau) - tau;
          }
        } else {
          if (sglb > zero) {
            eta = std::sqrt(sglb * tau) - tau;
          }
        }
      }
    }
    prew = w;
    tau = tau + eta;
    *sigma = *sigma + eta;
    for (j = 1; j <= n; ++j) {
      WORK(j) = WORK(j) + eta;
      DELTA(j) = DELTA(j) - eta;
    }
    // Evaluate PSI and the derivative DPSI
    dpsi = zero;
    psi = zero;
    erretm = zero;
    for (j = 1; j <= iim1; ++j) {
      temp = Z(j) / (WORK(j) * DELTA(j));
      psi = psi + Z(j) * temp;
      dpsi = dpsi + temp * temp;
      erretm = erretm + psi;
    }
    erretm = std::abs(erretm);
    // Evaluate PHI and the derivative DPHI
    dphi = zero;
    phi = zero;
    for (j = n; j >= iip1; --j) {
      temp = Z(j) / (WORK(j) * DELTA(j));
      phi = phi + Z(j) * temp;
      dphi = dphi + temp * temp;
      erretm = erretm + phi;
    }
    tau2 = WORK(ii) * DELTA(ii);
    temp = Z(ii) / tau2;
    dw = dpsi + dphi + temp * temp;
    temp = Z(ii) * temp;
    w = rhoinv + phi + psi + temp;
    erretm =
        eight * (phi - psi) + erretm + two * rhoinv + three * std::abs(temp);
    // $          + ABS( TAU2 )*DW
    swtch = false;
    if (orgati) {
      if (-w > std::abs(prew) / ten) swtch = true;
    } else {
      if (w > std::abs(prew) / ten) swtch = true;
    }
    // Main loop to update the values of the array   DELTA and WORK
    iter = niter + 1;
    for (niter = iter; niter <= maxit; ++niter) {
      // Test for convergence
      if (std::abs(w) <= eps * erretm) {
        // $          .OR. (SGUB-SGLB).LE.EIGHT*ABS(SGUB+SGLB) ) THEN
        goto L240;
      }
      if (w <= zero) {
        sglb = Max(sglb, tau);
      } else {
        sgub = Min(sgub, tau);
      }
      // Calculate the new step
      if (!swtch3) {
        dtipsq = WORK(ip1) * DELTA(ip1);
        dtisq = WORK(i) * DELTA(i);
        if (!swtch) {
          if (orgati) {
            c = w - dtipsq * dw + delsq * ((Z(i) / dtisq) * (Z(i) / dtisq));
          } else {
            c = w - dtisq * dw -
                delsq * ((Z(ip1) / dtipsq) * (Z(ip1) / dtipsq));
          }
        } else {
          temp = Z(ii) / (WORK(ii) * DELTA(ii));
          if (orgati) {
            dpsi = dpsi + temp * temp;
          } else {
            dphi = dphi + temp * temp;
          }
          c = w - dtisq * dpsi - dtipsq * dphi;
        }
        a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
        b = dtipsq * dtisq * w;
        if (c == zero) {
          if (a == zero) {
            if (!swtch) {
              if (orgati) {
                a = Z(i) * Z(i) + dtipsq * dtipsq * (dpsi + dphi);
              } else {
                a = Z(ip1) * Z(ip1) + dtisq * dtisq * (dpsi + dphi);
              }
            } else {
              a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
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
        dtiim = WORK(iim1) * DELTA(iim1);
        dtiip = WORK(iip1) * DELTA(iip1);
        temp = rhoinv + psi + phi;
        if (swtch) {
          c = temp - dtiim * dpsi - dtiip * dphi;
          ZZ(1) = dtiim * dtiim * dpsi;
          ZZ(3) = dtiip * dtiip * dphi;
        } else {
          if (orgati) {
            temp1 = Z(iim1) / dtiim;
            temp1 = temp1 * temp1;
            temp2 = (D(iim1) - D(iip1)) * (D(iim1) + D(iip1)) * temp1;
            c = temp - dtiip * (dpsi + dphi) - temp2;
            ZZ(1) = Z(iim1) * Z(iim1);
            if (dpsi < temp1) {
              ZZ(3) = dtiip * dtiip * dphi;
            } else {
              ZZ(3) = dtiip * dtiip * ((dpsi - temp1) + dphi);
            }
          } else {
            temp1 = Z(iip1) / dtiip;
            temp1 = temp1 * temp1;
            temp2 = (D(iip1) - D(iim1)) * (D(iim1) + D(iip1)) * temp1;
            c = temp - dtiim * (dpsi + dphi) - temp2;
            if (dphi < temp1) {
              ZZ(1) = dtiim * dtiim * dpsi;
            } else {
              ZZ(1) = dtiim * dtiim * (dpsi + (dphi - temp1));
            }
            ZZ(3) = Z(iip1) * Z(iip1);
          }
        }
        DD(1) = dtiim;
        DD(2) = DELTA(ii) * WORK(ii);
        DD(3) = dtiip;
        info = Laed6(niter, orgati, c, dd, zz, w, &eta);
        if (info != 0) {
          // If INFO is not 0, i.e., DLAED6 failed, switch
          // back to two pole interpolation
          swtch3 = false;
          info = 0;
          dtipsq = WORK(ip1) * DELTA(ip1);
          dtisq = WORK(i) * DELTA(i);
          if (!swtch) {
            if (orgati) {
              c = w - dtipsq * dw + delsq * ((Z(i) / dtisq) * (Z(i) / dtisq));
            } else {
              c = w - dtisq * dw -
                  delsq * ((Z(ip1) / dtipsq) * (Z(ip1) / dtipsq));
            }
          } else {
            temp = Z(ii) / (WORK(ii) * DELTA(ii));
            if (orgati) {
              dpsi = dpsi + temp * temp;
            } else {
              dphi = dphi + temp * temp;
            }
            c = w - dtisq * dpsi - dtipsq * dphi;
          }
          a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
          b = dtipsq * dtisq * w;
          if (c == zero) {
            if (a == zero) {
              if (!swtch) {
                if (orgati) {
                  a = Z(i) * Z(i) + dtipsq * dtipsq * (dpsi + dphi);
                } else {
                  a = Z(ip1) * Z(ip1) + dtisq * dtisq * (dpsi + dphi);
                }
              } else {
                a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
              }
            }
            eta = b / a;
          } else if (a <= zero) {
            eta = (a - std::sqrt(std::abs(a * a - four * b * c))) / (two * c);
          } else {
            eta = two * b / (a + std::sqrt(std::abs(a * a - four * b * c)));
          }
        }
      }
      // Note, eta should be positive if w is negative, and
      // eta should be negative otherwise. However,
      // if for some reason caused by roundoff, eta*w > 0,
      // we simply use one Newton step instead. This way
      // will guarantee eta*w < 0.
      if (w * eta >= zero) eta = -w / dw;
      eta = eta / (*sigma + std::sqrt(*sigma * *sigma + eta));
      temp = tau + eta;
      if (temp > sgub || temp < sglb) {
        if (w < zero) {
          eta = (sgub - tau) / two;
        } else {
          eta = (sglb - tau) / two;
        }
        if (geomavg) {
          if (w < zero) {
            if (tau > zero) {
              eta = std::sqrt(sgub * tau) - tau;
            }
          } else {
            if (sglb > zero) {
              eta = std::sqrt(sglb * tau) - tau;
            }
          }
        }
      }
      prew = w;
      tau = tau + eta;
      *sigma = *sigma + eta;
      for (j = 1; j <= n; ++j) {
        WORK(j) = WORK(j) + eta;
        DELTA(j) = DELTA(j) - eta;
      }
      // Evaluate PSI and the derivative DPSI
      dpsi = zero;
      psi = zero;
      erretm = zero;
      for (j = 1; j <= iim1; ++j) {
        temp = Z(j) / (WORK(j) * DELTA(j));
        psi = psi + Z(j) * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
      }
      erretm = std::abs(erretm);
      // Evaluate PHI and the derivative DPHI
      dphi = zero;
      phi = zero;
      for (j = n; j >= iip1; --j) {
        temp = Z(j) / (WORK(j) * DELTA(j));
        phi = phi + Z(j) * temp;
        dphi = dphi + temp * temp;
        erretm = erretm + phi;
      }
      tau2 = WORK(ii) * DELTA(ii);
      temp = Z(ii) / tau2;
      dw = dpsi + dphi + temp * temp;
      temp = Z(ii) * temp;
      w = rhoinv + phi + psi + temp;
      erretm =
          eight * (phi - psi) + erretm + two * rhoinv + three * std::abs(temp);
      // $             + ABS( TAU2 )*DW
      if (w * prew > zero && std::abs(w) > std::abs(prew) / ten) swtch = !swtch;
    }
    // Return with INFO = 1, NITER = MAXIT and not converged
    info = 1;
  }
L240:
  return info;
}

// xLASD8: the K square roots D of the secular equation of the deflated
// merge of xLASD6 with the poles DSIGMA and the updating vector Z, the
// distances DIFL and DIFR (with the normalizing factors of the right
// singular vectors in DIFR(:, 2) if ICOMPQ = 1) to the poles, and the
// updated first and last components VF and VL of the right singular
// vectors. WORK has 3*K elements. Returns INFO = 1 if xLASD4 failed.
template <typename R>
blasint Lasd8(blasint icompq, blasint k, R* d, R* z, R* vf, R* vl, R* difl,
              R* difr, blasint lddifr, R* dsigma, R* work) {
  const R one = R(1);
  const Vector<R> D(d);
  const Vector<R> DIFL(difl);
  const Matrix<R> DIFR(difr, lddifr);
  const Vector<R> DSIGMA(dsigma);
  const Vector<R> VF(vf);
  const Vector<R> VL(vl);
  const Vector<R> WORK(work);
  const Vector<R> Z(z);
  blasint i, iwk1, iwk2, iwk2i, iwk3, iwk3i, j;
  R diflj, dj, dsigj, rho, temp;
  R difrj = R(0), dsigjp = R(0);
  blasint info = 0;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (k < 1) {
    info = -2;
  } else if (lddifr < k) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<R>("LASD8", -info);
    return info;
  }
  // Quick return if possible
  if (k == 1) {
    D(1) = std::abs(Z(1));
    DIFL(1) = D(1);
    if (icompq == 1) {
      DIFL(2) = one;
      DIFR(1, 2) = one;
    }
    return info;
  }
  // Book keeping.
  iwk1 = 1;
  iwk2 = iwk1 + k;
  iwk3 = iwk2 + k;
  iwk2i = iwk2 - 1;
  iwk3i = iwk3 - 1;
  // Normalize Z.
  rho = Nrm2(k, z, 1);
  info = Lascl('G', 0, 0, rho, one, k, 1, z, k);
  rho = rho * rho;
  // Initialize WORK(IWK3).
  Laset('A', k, 1, one, one, WORK.ptr(iwk3), k);
  // Compute the updated singular values, the arrays DIFL, DIFR,
  // and the updated Z.
  for (j = 1; j <= k; ++j) {
    info =
        Lasd4(k, j, dsigma, z, WORK.ptr(iwk1), rho, D.ptr(j), WORK.ptr(iwk2));
    // If the root finder fails, report the convergence failure.
    if (info != 0) {
      return info;
    }
    WORK(iwk3i + j) = WORK(iwk3i + j) * WORK(j) * WORK(iwk2i + j);
    DIFL(j) = -WORK(j);
    DIFR(j, 1) = -WORK(j + 1);
    for (i = 1; i <= j - 1; ++i) {
      WORK(iwk3i + i) = WORK(iwk3i + i) * WORK(i) * WORK(iwk2i + i) /
                        (DSIGMA(i) - DSIGMA(j)) / (DSIGMA(i) + DSIGMA(j));
    }
    for (i = j + 1; i <= k; ++i) {
      WORK(iwk3i + i) = WORK(iwk3i + i) * WORK(i) * WORK(iwk2i + i) /
                        (DSIGMA(i) - DSIGMA(j)) / (DSIGMA(i) + DSIGMA(j));
    }
  }
  // Compute updated Z.
  for (i = 1; i <= k; ++i) {
    Z(i) = std::copysign(std::sqrt(std::abs(WORK(iwk3i + i))), Z(i));
  }
  // Update VF and VL.
  for (j = 1; j <= k; ++j) {
    diflj = DIFL(j);
    dj = D(j);
    dsigj = -DSIGMA(j);
    if (j < k) {
      difrj = -DIFR(j, 1);
      dsigjp = -DSIGMA(j + 1);
    }
    WORK(j) = -Z(j) / diflj / (DSIGMA(j) + dj);
    // Use calls to the subroutine DLAMC3 to enforce the parentheses
    // (x+y)+z. The goal is to prevent optimizing compilers
    // from doing x+(y+z).
    for (i = 1; i <= j - 1; ++i) {
      WORK(i) = Z(i) / (Lamc3(DSIGMA(i), dsigj) - diflj) / (DSIGMA(i) + dj);
    }
    for (i = j + 1; i <= k; ++i) {
      WORK(i) = Z(i) / (Lamc3(DSIGMA(i), dsigjp) + difrj) / (DSIGMA(i) + dj);
    }
    temp = Nrm2(k, work, 1);
    WORK(iwk2i + j) = Dotc(k, work, 1, vf, 1) / temp;
    WORK(iwk3i + j) = Dotc(k, work, 1, vl, 1) / temp;
    if (icompq == 1) {
      DIFR(j, 2) = temp;
    }
  }
  Copy(k, WORK.ptr(iwk2), 1, vf, 1);
  Copy(k, WORK.ptr(iwk3), 1, vl, 1);
  return info;
}

// xLASD7: the deflation of the merge of xLASD6 (singular values only):
// sorts the singular values of the two subproblems, deflates small
// components of Z and close singular values (the Givens rotations go to
// GIVCOL / GIVNUM and the permutation to PERM if ICOMPQ = 1) and returns
// the order K of the deflated secular equation, its poles DSIGMA and the
// rotation (C, S) applied to the last row for SQRE = 1.
template <typename R>
blasint Lasd7(blasint icompq, blasint nl, blasint nr, blasint sqre, blasint* k,
              R* d, R* z, R* zw, R* vf, R* vfw, R* vl, R* vlw, R alpha, R beta,
              R* dsigma, blasint* idx, blasint* idxp, blasint* idxq,
              blasint* perm, blasint* givptr, blasint* givcol, blasint ldgcol,
              R* givnum, blasint ldgnum, R* c, R* s) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R eight = R(8);
  const Vector<R> D(d);
  const Vector<R> DSIGMA(dsigma);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldgnum);
  const Vector<blasint> IDX(idx);
  const Vector<blasint> IDXP(idxp);
  const Vector<blasint> IDXQ(idxq);
  const Vector<blasint> PERM(perm);
  const Vector<R> VF(vf);
  const Vector<R> VFW(vfw);
  const Vector<R> VL(vl);
  const Vector<R> VLW(vlw);
  const Vector<R> Z(z);
  const Vector<R> ZW(zw);
  blasint i, idxi, idxj, idxjp, j, jp, k2, m, n, nlp1, nlp2;
  // Always set before use: the first loop over J exits by GO TO.
  blasint jprev = 0;
  R eps, hlftol, tau, tol, z1;
  blasint info = 0;
  n = nl + nr + 1;
  m = n + sqre;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (nl < 1) {
    info = -2;
  } else if (nr < 1) {
    info = -3;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -4;
  } else if (ldgcol < n) {
    info = -22;
  } else if (ldgnum < n) {
    info = -24;
  }
  if (info != 0) {
    Xerbla<R>("LASD7", -info);
    return info;
  }
  nlp1 = nl + 1;
  nlp2 = nl + 2;
  if (icompq == 1) {
    *givptr = 0;
  }
  // Generate the first part of the vector Z and move the singular
  // values in the first part of D one position backward.
  z1 = alpha * VL(nlp1);
  VL(nlp1) = zero;
  tau = VF(nlp1);
  for (i = nl; i >= 1; --i) {
    Z(i + 1) = alpha * VL(i);
    VL(i) = zero;
    VF(i + 1) = VF(i);
    D(i + 1) = D(i);
    IDXQ(i + 1) = IDXQ(i) + 1;
  }
  VF(1) = tau;
  // Generate the second part of the vector Z.
  for (i = nlp2; i <= m; ++i) {
    Z(i) = beta * VF(i);
    VF(i) = zero;
  }
  // Sort the singular values into increasing order
  for (i = nlp2; i <= n; ++i) {
    IDXQ(i) = IDXQ(i) + nlp1;
  }
  // DSIGMA, IDXC, IDXC, and ZW are used as storage space.
  for (i = 2; i <= n; ++i) {
    DSIGMA(i) = D(IDXQ(i));
    ZW(i) = Z(IDXQ(i));
    VFW(i) = VF(IDXQ(i));
    VLW(i) = VL(IDXQ(i));
  }
  Lamrg(nl, nr, DSIGMA.ptr(2), 1, 1, IDX.ptr(2));
  for (i = 2; i <= n; ++i) {
    idxi = 1 + IDX(i);
    D(i) = DSIGMA(idxi);
    Z(i) = ZW(idxi);
    VF(i) = VFW(idxi);
    VL(i) = VLW(idxi);
  }
  // Calculate the allowable deflation tolerance
  eps = Lamch<R>('E');
  tol = Max(std::abs(alpha), std::abs(beta));
  tol = eight * eight * eps * Max(std::abs(D(n)), tol);
  // There are 2 kinds of deflation -- first a value in the z-vector
  // is small, second two (or more) singular values are very close
  // together (their difference is small).
  // If the value in the z-vector is small, we simply permute the
  // array so that the corresponding singular value is moved to the
  // end.
  // If two values in the D-vector are close, we perform a two-sided
  // rotation designed to make one of the corresponding z-vector
  // entries zero, and then permute the array so that the deflated
  // singular value is moved to the end.
  // If there are multiple singular values then the problem deflates.
  // Here the number of equal singular values are found.  As each equal
  // singular value is found, an elementary reflector is computed to
  // rotate the corresponding singular subspace so that the
  // corresponding components of Z are zero in this new basis.
  *k = 1;
  k2 = n + 1;
  for (j = 2; j <= n; ++j) {
    if (std::abs(Z(j)) <= tol) {
      // Deflate due to small z component.
      k2 = k2 - 1;
      IDXP(k2) = j;
      if (j == n) goto L100;
    } else {
      jprev = j;
      goto L70;
    }
  }
L70:
  j = jprev;
L80:
  j = j + 1;
  if (j > n) goto L90;
  if (std::abs(Z(j)) <= tol) {
    // Deflate due to small z component.
    k2 = k2 - 1;
    IDXP(k2) = j;
  } else {
    // Check if singular values are close enough to allow deflation.
    if (std::abs(D(j) - D(jprev)) <= tol) {
      // Deflation is possible.
      *s = Z(jprev);
      *c = Z(j);
      // Find sqrt(a**2+b**2) without overflow or
      // destructive underflow.
      tau = Lapy2(*c, *s);
      Z(j) = tau;
      Z(jprev) = zero;
      *c = *c / tau;
      *s = -*s / tau;
      // Record the appropriate Givens rotation
      if (icompq == 1) {
        *givptr = *givptr + 1;
        idxjp = IDXQ(IDX(jprev) + 1);
        idxj = IDXQ(IDX(j) + 1);
        if (idxjp <= nlp1) {
          idxjp = idxjp - 1;
        }
        if (idxj <= nlp1) {
          idxj = idxj - 1;
        }
        GIVCOL(*givptr, 2) = idxjp;
        GIVCOL(*givptr, 1) = idxj;
        GIVNUM(*givptr, 2) = *c;
        GIVNUM(*givptr, 1) = *s;
      }
      Rot(1, VF.ptr(jprev), 1, VF.ptr(j), 1, *c, *s);
      Rot(1, VL.ptr(jprev), 1, VL.ptr(j), 1, *c, *s);
      k2 = k2 - 1;
      IDXP(k2) = jprev;
      jprev = j;
    } else {
      *k = *k + 1;
      ZW(*k) = Z(jprev);
      DSIGMA(*k) = D(jprev);
      IDXP(*k) = jprev;
      jprev = j;
    }
  }
  goto L80;
L90:
  // Record the last singular value.
  *k = *k + 1;
  ZW(*k) = Z(jprev);
  DSIGMA(*k) = D(jprev);
  IDXP(*k) = jprev;
L100:
  // Sort the singular values into DSIGMA. The singular values which
  // were not deflated go into the first K slots of DSIGMA, except
  // that DSIGMA(1) is treated separately.
  for (j = 2; j <= n; ++j) {
    jp = IDXP(j);
    DSIGMA(j) = D(jp);
    VFW(j) = VF(jp);
    VLW(j) = VL(jp);
  }
  if (icompq == 1) {
    for (j = 2; j <= n; ++j) {
      jp = IDXP(j);
      PERM(j) = IDXQ(IDX(jp) + 1);
      if (PERM(j) <= nlp1) {
        PERM(j) = PERM(j) - 1;
      }
    }
  }
  // The deflated singular values go back into the last N - K slots of
  // D.
  Copy(n - *k, DSIGMA.ptr(*k + 1), 1, D.ptr(*k + 1), 1);
  // Determine DSIGMA(1), DSIGMA(2), Z(1), VF(1), VL(1), VF(M), and
  // VL(M).
  DSIGMA(1) = zero;
  hlftol = tol / two;
  if (std::abs(DSIGMA(2)) <= hlftol) DSIGMA(2) = hlftol;
  if (m > n) {
    Z(1) = Lapy2(z1, Z(m));
    if (Z(1) <= tol) {
      *c = one;
      *s = zero;
      Z(1) = tol;
    } else {
      *c = z1 / Z(1);
      *s = -Z(m) / Z(1);
    }
    Rot(1, VF.ptr(m), 1, VF.ptr(1), 1, *c, *s);
    Rot(1, VL.ptr(m), 1, VL.ptr(1), 1, *c, *s);
  } else {
    if (std::abs(z1) <= tol) {
      Z(1) = tol;
    } else {
      Z(1) = z1;
    }
  }
  // Restore Z, VF, and VL.
  Copy(*k - 1, ZW.ptr(2), 1, Z.ptr(2), 1);
  Copy(n - 1, VFW.ptr(2), 1, VF.ptr(2), 1);
  Copy(n - 1, VLW.ptr(2), 1, VL.ptr(2), 1);
  return info;
}

// xLASD6: merges two adjacent subproblems of xLASDA, the singular values
// D(1:NL) and D(NL+2:NL+NR+1) with the joining row ALPHA, BETA, into the
// singular values of the (N+SQRE)-by-N upper bidiagonal matrix, keeping
// only the first and last components VF and VL of the right singular
// vectors and the factored form (PERM, GIVCOL, GIVNUM, POLES, DIFL, DIFR,
// Z, K, C, S) of the singular vectors if ICOMPQ = 1. IDXQ is the
// permutation that sorts D in ascending order. WORK has 4*M and IWORK 3*N
// elements. Returns INFO = 1 if a singular value did not converge.
template <typename R>
blasint Lasd6(blasint icompq, blasint nl, blasint nr, blasint sqre, R* d, R* vf,
              R* vl, R* alpha, R* beta, blasint* idxq, blasint* perm,
              blasint* givptr, blasint* givcol, blasint ldgcol, R* givnum,
              blasint ldgnum, R* poles, R* difl, R* difr, R* z, blasint* k,
              R* c, R* s, R* work, blasint* iwork) {
  const R one = R(1);
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<R> DIFL(difl);
  const Vector<R> DIFR(difr);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldgnum);
  const Vector<blasint> IDXQ(idxq);
  const Vector<blasint> IWORK(iwork);
  const Vector<blasint> PERM(perm);
  const Matrix<R> POLES(poles, ldgnum);
  const Vector<R> VF(vf);
  const Vector<R> VL(vl);
  const Vector<R> WORK(work);
  const Vector<R> Z(z);
  blasint i, idx, idxc, idxp, isigma, ivfw, ivlw, iw, m, n, n1, n2;
  R orgnrm;
  blasint info = 0;
  n = nl + nr + 1;
  m = n + sqre;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (nl < 1) {
    info = -2;
  } else if (nr < 1) {
    info = -3;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -4;
  } else if (ldgcol < n) {
    info = -14;
  } else if (ldgnum < n) {
    info = -16;
  }
  if (info != 0) {
    Xerbla<R>("LASD6", -info);
    return info;
  }
  // The following values are for bookkeeping purposes only.  They are
  // integer pointers which indicate the portion of the workspace
  // used by a particular array in DLASD7 and DLASD8.
  isigma = 1;
  iw = isigma + n;
  ivfw = iw + m;
  ivlw = ivfw + m;
  idx = 1;
  idxc = idx + n;
  idxp = idxc + n;
  // Scale.
  orgnrm = Max(std::abs(*alpha), std::abs(*beta));
  D(nl + 1) = zero;
  for (i = 1; i <= n; ++i) {
    if (std::abs(D(i)) > orgnrm) {
      orgnrm = std::abs(D(i));
    }
  }
  info = Lascl('G', 0, 0, orgnrm, one, n, 1, d, n);
  *alpha = *alpha / orgnrm;
  *beta = *beta / orgnrm;
  // Sort and Deflate singular values.
  info = Lasd7(icompq, nl, nr, sqre, k, d, z, WORK.ptr(iw), vf, WORK.ptr(ivfw),
               vl, WORK.ptr(ivlw), *alpha, *beta, WORK.ptr(isigma),
               IWORK.ptr(idx), IWORK.ptr(idxp), idxq, perm, givptr, givcol,
               ldgcol, givnum, ldgnum, c, s);
  // Solve Secular Equation, compute DIFL, DIFR, and update VF, VL.
  info = Lasd8(icompq, *k, d, z, vf, vl, difl, difr, ldgnum, WORK.ptr(isigma),
               WORK.ptr(iw));
  // Report the possible convergence failure.
  if (info != 0) {
    return info;
  }
  // Save the poles if ICOMPQ = 1.
  if (icompq == 1) {
    Copy(*k, d, 1, POLES.ptr(1, 1), 1);
    Copy(*k, WORK.ptr(isigma), 1, POLES.ptr(1, 2), 1);
  }
  // Unscale.
  info = Lascl('G', 0, 0, one, orgnrm, n, 1, d, n);
  // Prepare the IDXQ sorting permutation.
  n1 = *k;
  n2 = n - *k;
  Lamrg(n1, n2, d, 1, -1, idxq);
  return info;
}

// xLASD3: the K singular values D of the deflated secular equation of
// xLASD1 (poles DSIGMA, updating vector Z) and the updated singular vector
// matrices U and VT of the merged problem, from the deflated vectors U2
// and VT2 grouped by structure (CTOT counts the columns of each of the four
// types, IDXC permutes them). Q is K-by-K workspace. Returns INFO = 1 if
// xLASD4 failed.
template <typename R>
blasint Lasd3(blasint nl, blasint nr, blasint sqre, blasint k, R* d, R* q,
              blasint ldq, R* dsigma, R* u, blasint ldu, R* u2, blasint ldu2,
              R* vt, blasint ldvt, R* vt2, blasint ldvt2, const blasint* idxc,
              const blasint* ctot, R* z) {
  const R one = R(1);
  const R zero = R(0);
  const R negone = -R(1);
  const Vector<const blasint> CTOT(ctot);
  const Vector<R> D(d);
  const Vector<R> DSIGMA(dsigma);
  const Vector<const blasint> IDXC(idxc);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> U(u, ldu);
  const Matrix<R> U2(u2, ldu2);
  const Matrix<R> VT(vt, ldvt);
  const Matrix<R> VT2(vt2, ldvt2);
  const Vector<R> Z(z);
  blasint ctemp, i, j, jc, ktemp, m, n, nlp1, nlp2, nrp1;
  R rho, temp;
  blasint info = 0;
  if (nl < 1) {
    info = -1;
  } else if (nr < 1) {
    info = -2;
  } else if ((sqre != 1) && (sqre != 0)) {
    info = -3;
  }
  n = nl + nr + 1;
  m = n + sqre;
  nlp1 = nl + 1;
  nlp2 = nl + 2;
  if ((k < 1) || (k > n)) {
    info = -4;
  } else if (ldq < k) {
    info = -7;
  } else if (ldu < n) {
    info = -10;
  } else if (ldu2 < n) {
    info = -12;
  } else if (ldvt < m) {
    info = -14;
  } else if (ldvt2 < m) {
    info = -16;
  }
  if (info != 0) {
    Xerbla<R>("LASD3", -info);
    return info;
  }
  // Quick return if possible
  if (k == 1) {
    D(1) = std::abs(Z(1));
    Copy(m, VT2.ptr(1, 1), ldvt2, VT.ptr(1, 1), ldvt);
    if (Z(1) > zero) {
      Copy(n, U2.ptr(1, 1), 1, U.ptr(1, 1), 1);
    } else {
      for (i = 1; i <= n; ++i) {
        U(i, 1) = -U2(i, 1);
      }
    }
    return info;
  }
  // Keep a copy of Z.
  Copy(k, z, 1, q, 1);
  // Normalize Z.
  rho = Nrm2(k, z, 1);
  info = Lascl('G', 0, 0, rho, one, k, 1, z, k);
  rho = rho * rho;
  // Find the new singular values.
  for (j = 1; j <= k; ++j) {
    info = Lasd4(k, j, dsigma, z, U.ptr(1, j), rho, D.ptr(j), VT.ptr(1, j));
    // If the zero finder fails, report the convergence failure.
    if (info != 0) {
      return info;
    }
  }
  // Compute updated Z.
  for (i = 1; i <= k; ++i) {
    Z(i) = U(i, k) * VT(i, k);
    for (j = 1; j <= i - 1; ++j) {
      Z(i) = Z(i) * (U(i, j) * VT(i, j) / (DSIGMA(i) - DSIGMA(j)) /
                     (DSIGMA(i) + DSIGMA(j)));
    }
    for (j = i; j <= k - 1; ++j) {
      Z(i) = Z(i) * (U(i, j) * VT(i, j) / (DSIGMA(i) - DSIGMA(j + 1)) /
                     (DSIGMA(i) + DSIGMA(j + 1)));
    }
    Z(i) = std::copysign(std::sqrt(std::abs(Z(i))), Q(i, 1));
  }
  // Compute left singular vectors of the modified diagonal matrix,
  // and store related information for the right singular vectors.
  for (i = 1; i <= k; ++i) {
    VT(1, i) = Z(1) / U(1, i) / VT(1, i);
    U(1, i) = negone;
    for (j = 2; j <= k; ++j) {
      VT(j, i) = Z(j) / U(j, i) / VT(j, i);
      U(j, i) = DSIGMA(j) * VT(j, i);
    }
    temp = Nrm2(k, U.ptr(1, i), 1);
    Q(1, i) = U(1, i) / temp;
    for (j = 2; j <= k; ++j) {
      jc = IDXC(j);
      Q(j, i) = U(jc, i) / temp;
    }
  }
  // Update the left singular vector matrix.
  if (k == 2) {
    Gemm('N', 'N', n, k, k, one, u2, ldu2, q, ldq, zero, u, ldu);
    goto L100;
  }
  if (CTOT(1) > 0) {
    Gemm('N', 'N', nl, k, CTOT(1), one, U2.ptr(1, 2), ldu2, Q.ptr(2, 1), ldq,
         zero, U.ptr(1, 1), ldu);
    if (CTOT(3) > 0) {
      ktemp = 2 + CTOT(1) + CTOT(2);
      Gemm('N', 'N', nl, k, CTOT(3), one, U2.ptr(1, ktemp), ldu2,
           Q.ptr(ktemp, 1), ldq, one, U.ptr(1, 1), ldu);
    }
  } else if (CTOT(3) > 0) {
    ktemp = 2 + CTOT(1) + CTOT(2);
    Gemm('N', 'N', nl, k, CTOT(3), one, U2.ptr(1, ktemp), ldu2, Q.ptr(ktemp, 1),
         ldq, zero, U.ptr(1, 1), ldu);
  } else {
    Lacpy('F', nl, k, u2, ldu2, u, ldu);
  }
  Copy(k, Q.ptr(1, 1), ldq, U.ptr(nlp1, 1), ldu);
  ktemp = 2 + CTOT(1);
  ctemp = CTOT(2) + CTOT(3);
  Gemm('N', 'N', nr, k, ctemp, one, U2.ptr(nlp2, ktemp), ldu2, Q.ptr(ktemp, 1),
       ldq, zero, U.ptr(nlp2, 1), ldu);
  // Generate the right singular vectors.
L100:
  for (i = 1; i <= k; ++i) {
    temp = Nrm2(k, VT.ptr(1, i), 1);
    Q(i, 1) = VT(1, i) / temp;
    for (j = 2; j <= k; ++j) {
      jc = IDXC(j);
      Q(i, j) = VT(jc, i) / temp;
    }
  }
  // Update the right singular vector matrix.
  if (k == 2) {
    Gemm('N', 'N', k, m, k, one, q, ldq, vt2, ldvt2, zero, vt, ldvt);
    return info;
  }
  ktemp = 1 + CTOT(1);
  Gemm('N', 'N', k, nlp1, ktemp, one, Q.ptr(1, 1), ldq, VT2.ptr(1, 1), ldvt2,
       zero, VT.ptr(1, 1), ldvt);
  ktemp = 2 + CTOT(1) + CTOT(2);
  if (ktemp <= ldvt2)
    Gemm('N', 'N', k, nlp1, CTOT(3), one, Q.ptr(1, ktemp), ldq,
         VT2.ptr(ktemp, 1), ldvt2, one, VT.ptr(1, 1), ldvt);
  ktemp = CTOT(1) + 1;
  nrp1 = nr + sqre;
  if (ktemp > 1) {
    for (i = 1; i <= k; ++i) {
      Q(i, ktemp) = Q(i, 1);
    }
    for (i = nlp2; i <= m; ++i) {
      VT2(ktemp, i) = VT2(1, i);
    }
  }
  ctemp = 1 + CTOT(2) + CTOT(3);
  Gemm('N', 'N', k, nrp1, ctemp, one, Q.ptr(1, ktemp), ldq,
       VT2.ptr(ktemp, nlp2), ldvt2, zero, VT.ptr(1, nlp2), ldvt);
  return info;
}

// xLASD2: the deflation of the merge of xLASD1: sorts the singular values
// of the two subproblems, deflates small components of Z and close
// singular values (rotating U and VT), and returns the order K of the
// deflated secular equation with its poles DSIGMA, and the vectors in U2
// and VT2 grouped into four types of columns (COLTYP(1:4) returns their
// counts).
template <typename R>
blasint Lasd2(blasint nl, blasint nr, blasint sqre, blasint* k, R* d, R* z,
              R alpha, R beta, R* u, blasint ldu, R* vt, blasint ldvt,
              R* dsigma, R* u2, blasint ldu2, R* vt2, blasint ldvt2,
              blasint* idxp, blasint* idx, blasint* idxc, blasint* idxq,
              blasint* coltyp) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R eight = R(8);
  const Vector<blasint> COLTYP(coltyp);
  blasint ctot[4];
  const Vector<blasint> CTOT(ctot);
  const Vector<R> D(d);
  const Vector<R> DSIGMA(dsigma);
  const Vector<blasint> IDX(idx);
  const Vector<blasint> IDXC(idxc);
  const Vector<blasint> IDXP(idxp);
  const Vector<blasint> IDXQ(idxq);
  blasint psm[4];
  const Vector<blasint> PSM(psm);
  const Matrix<R> U(u, ldu);
  const Matrix<R> U2(u2, ldu2);
  const Matrix<R> VT(vt, ldvt);
  const Matrix<R> VT2(vt2, ldvt2);
  const Vector<R> Z(z);
  blasint ct, i, idxi, idxj, idxjp, j, jp, k2, m, n, nlp1, nlp2;
  // Always set before use: the first loop over J exits by GO TO.
  blasint jprev = 0;
  R eps, hlftol, tau, tol, z1;
  R c = R(0), s = R(0);
  blasint info = 0;
  if (nl < 1) {
    info = -1;
  } else if (nr < 1) {
    info = -2;
  } else if ((sqre != 1) && (sqre != 0)) {
    info = -3;
  }
  n = nl + nr + 1;
  m = n + sqre;
  if (ldu < n) {
    info = -10;
  } else if (ldvt < m) {
    info = -12;
  } else if (ldu2 < n) {
    info = -15;
  } else if (ldvt2 < m) {
    info = -17;
  }
  if (info != 0) {
    Xerbla<R>("LASD2", -info);
    return info;
  }
  nlp1 = nl + 1;
  nlp2 = nl + 2;
  // Generate the first part of the vector Z; and move the singular
  // values in the first part of D one position backward.
  z1 = alpha * VT(nlp1, nlp1);
  Z(1) = z1;
  for (i = nl; i >= 1; --i) {
    Z(i + 1) = alpha * VT(i, nlp1);
    D(i + 1) = D(i);
    IDXQ(i + 1) = IDXQ(i) + 1;
  }
  // Generate the second part of the vector Z.
  for (i = nlp2; i <= m; ++i) {
    Z(i) = beta * VT(i, nlp2);
  }
  // Initialize some reference arrays.
  for (i = 2; i <= nlp1; ++i) {
    COLTYP(i) = 1;
  }
  for (i = nlp2; i <= n; ++i) {
    COLTYP(i) = 2;
  }
  // Sort the singular values into increasing order
  for (i = nlp2; i <= n; ++i) {
    IDXQ(i) = IDXQ(i) + nlp1;
  }
  // DSIGMA, IDXC, IDXC, and the first column of U2
  // are used as storage space.
  for (i = 2; i <= n; ++i) {
    DSIGMA(i) = D(IDXQ(i));
    U2(i, 1) = Z(IDXQ(i));
    IDXC(i) = COLTYP(IDXQ(i));
  }
  Lamrg(nl, nr, DSIGMA.ptr(2), 1, 1, IDX.ptr(2));
  for (i = 2; i <= n; ++i) {
    idxi = 1 + IDX(i);
    D(i) = DSIGMA(idxi);
    Z(i) = U2(idxi, 1);
    COLTYP(i) = IDXC(idxi);
  }
  // Calculate the allowable deflation tolerance
  eps = Lamch<R>('E');
  tol = Max(std::abs(alpha), std::abs(beta));
  tol = eight * eps * Max(std::abs(D(n)), tol);
  // There are 2 kinds of deflation -- first a value in the z-vector
  // is small, second two (or more) singular values are very close
  // together (their difference is small).
  // If the value in the z-vector is small, we simply permute the
  // array so that the corresponding singular value is moved to the
  // end.
  // If two values in the D-vector are close, we perform a two-sided
  // rotation designed to make one of the corresponding z-vector
  // entries zero, and then permute the array so that the deflated
  // singular value is moved to the end.
  // If there are multiple singular values then the problem deflates.
  // Here the number of equal singular values are found.  As each equal
  // singular value is found, an elementary reflector is computed to
  // rotate the corresponding singular subspace so that the
  // corresponding components of Z are zero in this new basis.
  *k = 1;
  k2 = n + 1;
  for (j = 2; j <= n; ++j) {
    if (std::abs(Z(j)) <= tol) {
      // Deflate due to small z component.
      k2 = k2 - 1;
      IDXP(k2) = j;
      COLTYP(j) = 4;
      if (j == n) goto L120;
    } else {
      jprev = j;
      goto L90;
    }
  }
L90:
  j = jprev;
L100:
  j = j + 1;
  if (j > n) goto L110;
  if (std::abs(Z(j)) <= tol) {
    // Deflate due to small z component.
    k2 = k2 - 1;
    IDXP(k2) = j;
    COLTYP(j) = 4;
  } else {
    // Check if singular values are close enough to allow deflation.
    if (std::abs(D(j) - D(jprev)) <= tol) {
      // Deflation is possible.
      s = Z(jprev);
      c = Z(j);
      // Find sqrt(a**2+b**2) without overflow or
      // destructive underflow.
      tau = Lapy2(c, s);
      c = c / tau;
      s = -s / tau;
      Z(j) = tau;
      Z(jprev) = zero;
      // Apply back the Givens rotation to the left and right
      // singular vector matrices.
      idxjp = IDXQ(IDX(jprev) + 1);
      idxj = IDXQ(IDX(j) + 1);
      if (idxjp <= nlp1) {
        idxjp = idxjp - 1;
      }
      if (idxj <= nlp1) {
        idxj = idxj - 1;
      }
      Rot(n, U.ptr(1, idxjp), 1, U.ptr(1, idxj), 1, c, s);
      Rot(m, VT.ptr(idxjp, 1), ldvt, VT.ptr(idxj, 1), ldvt, c, s);
      if (COLTYP(j) != COLTYP(jprev)) {
        COLTYP(j) = 3;
      }
      COLTYP(jprev) = 4;
      k2 = k2 - 1;
      IDXP(k2) = jprev;
      jprev = j;
    } else {
      *k = *k + 1;
      U2(*k, 1) = Z(jprev);
      DSIGMA(*k) = D(jprev);
      IDXP(*k) = jprev;
      jprev = j;
    }
  }
  goto L100;
L110:
  // Record the last singular value.
  *k = *k + 1;
  U2(*k, 1) = Z(jprev);
  DSIGMA(*k) = D(jprev);
  IDXP(*k) = jprev;
L120:
  // Count up the total number of the various types of columns, then
  // form a permutation which positions the four column types into
  // four groups of uniform structure (although one or more of these
  // groups may be empty).
  for (j = 1; j <= 4; ++j) {
    CTOT(j) = 0;
  }
  for (j = 2; j <= n; ++j) {
    ct = COLTYP(j);
    CTOT(ct) = CTOT(ct) + 1;
  }
  // PSM(*) = Position in SubMatrix (of types 1 through 4)
  PSM(1) = 2;
  PSM(2) = 2 + CTOT(1);
  PSM(3) = PSM(2) + CTOT(2);
  PSM(4) = PSM(3) + CTOT(3);
  // Fill out the IDXC array so that the permutation which it induces
  // will place all type-1 columns first, all type-2 columns next,
  // then all type-3's, and finally all type-4's, starting from the
  // second column. This applies similarly to the rows of VT.
  for (j = 2; j <= n; ++j) {
    jp = IDXP(j);
    ct = COLTYP(jp);
    IDXC(PSM(ct)) = j;
    PSM(ct) = PSM(ct) + 1;
  }
  // Sort the singular values and corresponding singular vectors into
  // DSIGMA, U2, and VT2 respectively.  The singular values/vectors
  // which were not deflated go into the first K slots of DSIGMA, U2,
  // and VT2 respectively, while those which were deflated go into the
  // last N - K slots, except that the first column/row will be treated
  // separately.
  for (j = 2; j <= n; ++j) {
    jp = IDXP(j);
    DSIGMA(j) = D(jp);
    idxj = IDXQ(IDX(IDXP(IDXC(j))) + 1);
    if (idxj <= nlp1) {
      idxj = idxj - 1;
    }
    Copy(n, U.ptr(1, idxj), 1, U2.ptr(1, j), 1);
    Copy(m, VT.ptr(idxj, 1), ldvt, VT2.ptr(j, 1), ldvt2);
  }
  // Determine DSIGMA(1), DSIGMA(2) and Z(1)
  DSIGMA(1) = zero;
  hlftol = tol / two;
  if (std::abs(DSIGMA(2)) <= hlftol) DSIGMA(2) = hlftol;
  if (m > n) {
    Z(1) = Lapy2(z1, Z(m));
    if (Z(1) <= tol) {
      c = one;
      s = zero;
      Z(1) = tol;
    } else {
      c = z1 / Z(1);
      s = Z(m) / Z(1);
    }
  } else {
    if (std::abs(z1) <= tol) {
      Z(1) = tol;
    } else {
      Z(1) = z1;
    }
  }
  // Move the rest of the updating row to Z.
  Copy(*k - 1, U2.ptr(2, 1), 1, Z.ptr(2), 1);
  // Determine the first column of U2, the first row of VT2 and the
  // last row of VT.
  Laset('A', n, 1, zero, zero, u2, ldu2);
  U2(nlp1, 1) = one;
  if (m > n) {
    for (i = 1; i <= nlp1; ++i) {
      VT(m, i) = -s * VT(nlp1, i);
      VT2(1, i) = c * VT(nlp1, i);
    }
    for (i = nlp2; i <= m; ++i) {
      VT2(1, i) = s * VT(m, i);
      VT(m, i) = c * VT(m, i);
    }
  } else {
    Copy(m, VT.ptr(nlp1, 1), ldvt, VT2.ptr(1, 1), ldvt2);
  }
  if (m > n) {
    Copy(m, VT.ptr(m, 1), ldvt, VT2.ptr(m, 1), ldvt2);
  }
  // The deflated singular values and their corresponding vectors go
  // into the back of D, U, and V respectively.
  if (n > *k) {
    Copy(n - *k, DSIGMA.ptr(*k + 1), 1, D.ptr(*k + 1), 1);
    Lacpy('A', n, n - *k, U2.ptr(1, *k + 1), ldu2, U.ptr(1, *k + 1), ldu);
    Lacpy('A', n - *k, m, VT2.ptr(*k + 1, 1), ldvt2, VT.ptr(*k + 1, 1), ldvt);
  }
  // Copy CTOT into COLTYP for referencing in DLASD3.
  for (j = 1; j <= 4; ++j) {
    COLTYP(j) = CTOT(j);
  }
  return info;
}

// xLASD1: merges two adjacent subproblems of xLASD0, the SVDs of the
// upper bidiagonal NL-by-(NL+1) and NR-by-(NR+SQRE) matrices with the
// joining row ALPHA, BETA, into the SVD of the (N+SQRE)-by-N matrix,
// N = NL+NR+1, updating D, U and VT. IDXQ is the permutation that sorts D
// in ascending order. IWORK has 4*N and WORK 3*M**2+2*M elements. Returns
// INFO = 1 if a singular value did not converge.
template <typename R>
blasint Lasd1(blasint nl, blasint nr, blasint sqre, R* d, R* alpha, R* beta,
              R* u, blasint ldu, R* vt, blasint ldvt, blasint* idxq,
              blasint* iwork, R* work) {
  const R one = R(1);
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<blasint> IDXQ(idxq);
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldvt);
  const Vector<R> WORK(work);
  blasint coltyp, i, idx, idxc, idxp, iq, isigma, iu2, ivt2, iz, k, ldq, ldu2,
      ldvt2, m, n, n1, n2;
  R orgnrm;
  blasint info = 0;
  if (nl < 1) {
    info = -1;
  } else if (nr < 1) {
    info = -2;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -3;
  }
  if (info != 0) {
    Xerbla<R>("LASD1", -info);
    return info;
  }
  n = nl + nr + 1;
  m = n + sqre;
  // The following values are for bookkeeping purposes only.  They are
  // integer pointers which indicate the portion of the workspace
  // used by a particular array in DLASD2 and DLASD3.
  ldu2 = n;
  ldvt2 = m;
  iz = 1;
  isigma = iz + m;
  iu2 = isigma + n;
  ivt2 = iu2 + ldu2 * n;
  iq = ivt2 + ldvt2 * m;
  idx = 1;
  idxc = idx + n;
  coltyp = idxc + n;
  idxp = coltyp + n;
  // Scale.
  orgnrm = Max(std::abs(*alpha), std::abs(*beta));
  D(nl + 1) = zero;
  for (i = 1; i <= n; ++i) {
    if (std::abs(D(i)) > orgnrm) {
      orgnrm = std::abs(D(i));
    }
  }
  info = Lascl('G', 0, 0, orgnrm, one, n, 1, d, n);
  *alpha = *alpha / orgnrm;
  *beta = *beta / orgnrm;
  // Deflate singular values.
  info = Lasd2(nl, nr, sqre, &k, d, WORK.ptr(iz), *alpha, *beta, u, ldu, vt,
               ldvt, WORK.ptr(isigma), WORK.ptr(iu2), ldu2, WORK.ptr(ivt2),
               ldvt2, IWORK.ptr(idxp), IWORK.ptr(idx), IWORK.ptr(idxc), idxq,
               IWORK.ptr(coltyp));
  // Solve Secular Equation and update singular vectors.
  ldq = k;
  info = Lasd3(nl, nr, sqre, k, d, WORK.ptr(iq), ldq, WORK.ptr(isigma), u, ldu,
               WORK.ptr(iu2), ldu2, vt, ldvt, WORK.ptr(ivt2), ldvt2,
               IWORK.ptr(idxc), IWORK.ptr(coltyp), WORK.ptr(iz));
  // Report the convergence failure.
  if (info != 0) {
    return info;
  }
  // Unscale.
  info = Lascl('G', 0, 0, one, orgnrm, n, 1, d, n);
  // Prepare the IDXQ sorting permutation.
  n1 = k;
  n2 = n - k;
  Lamrg(n1, n2, d, 1, -1, idxq);
  return info;
}

// xLASDQ: the SVD of the N-by-(N+SQRE) upper (UPLO = 'U') or the
// (N+SQRE)-by-N lower (UPLO = 'L') bidiagonal matrix, SQRE = 0 or 1, by
// rotating it to upper bidiagonal form and calling xBDSQR: VT (N-by-NCVT)
// is overwritten by P**T * VT, U (NRU-by-N) by U * Q and C (N-by-NCC) by
// Q**T * C, with the singular values in ascending order. WORK has 4*N
// elements. Returns INFO as xBDSQR.
template <typename R>
blasint Lasdq(char uplo, blasint sqre, blasint n, blasint ncvt, blasint nru,
              blasint ncc, R* d, R* e, R* vt, blasint ldvt, R* u, blasint ldu,
              R* c, blasint ldc, R* work) {
  const R zero = R(0);
  const Matrix<R> C(c, ldc);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldvt);
  const Vector<R> WORK(work);
  blasint i, isub, iuplo, j, np1, sqre1;
  bool rotate;
  R cs, r, smin, sn;
  blasint info = 0;
  iuplo = 0;
  if (Lsame(uplo, 'U')) iuplo = 1;
  if (Lsame(uplo, 'L')) iuplo = 2;
  if (iuplo == 0) {
    info = -1;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ncvt < 0) {
    info = -4;
  } else if (nru < 0) {
    info = -5;
  } else if (ncc < 0) {
    info = -6;
  } else if ((ncvt == 0 && ldvt < 1) ||
             (ncvt > 0 && ldvt < Max<blasint>(1, n))) {
    info = -10;
  } else if (ldu < Max<blasint>(1, nru)) {
    info = -12;
  } else if ((ncc == 0 && ldc < 1) || (ncc > 0 && ldc < Max<blasint>(1, n))) {
    info = -14;
  }
  if (info != 0) {
    Xerbla<R>("LASDQ", -info);
    return info;
  }
  if (n == 0) return info;
  // ROTATE is true if any singular vectors desired, false otherwise
  rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);
  np1 = n + 1;
  sqre1 = sqre;
  // If matrix non-square upper bidiagonal, rotate to be lower
  // bidiagonal.  The rotations are on the right.
  if ((iuplo == 1) && (sqre1 == 1)) {
    for (i = 1; i <= n - 1; ++i) {
      Lartg(D(i), E(i), &cs, &sn, &r);
      D(i) = r;
      E(i) = sn * D(i + 1);
      D(i + 1) = cs * D(i + 1);
      if (rotate) {
        WORK(i) = cs;
        WORK(n + i) = sn;
      }
    }
    Lartg(D(n), E(n), &cs, &sn, &r);
    D(n) = r;
    E(n) = zero;
    if (rotate) {
      WORK(n) = cs;
      WORK(n + n) = sn;
    }
    iuplo = 2;
    sqre1 = 0;
    // Update singular vectors if desired.
    if (ncvt > 0)
      Lasr('L', 'V', 'F', np1, ncvt, WORK.ptr(1), WORK.ptr(np1), vt, ldvt);
  }
  // If matrix lower bidiagonal, rotate to be upper bidiagonal
  // by applying Givens rotations on the left.
  if (iuplo == 2) {
    for (i = 1; i <= n - 1; ++i) {
      Lartg(D(i), E(i), &cs, &sn, &r);
      D(i) = r;
      E(i) = sn * D(i + 1);
      D(i + 1) = cs * D(i + 1);
      if (rotate) {
        WORK(i) = cs;
        WORK(n + i) = sn;
      }
    }
    // If matrix (N+1)-by-N lower bidiagonal, one additional
    // rotation is needed.
    if (sqre1 == 1) {
      Lartg(D(n), E(n), &cs, &sn, &r);
      D(n) = r;
      if (rotate) {
        WORK(n) = cs;
        WORK(n + n) = sn;
      }
    }
    // Update singular vectors if desired.
    if (nru > 0) {
      if (sqre1 == 0) {
        Lasr('R', 'V', 'F', nru, n, WORK.ptr(1), WORK.ptr(np1), u, ldu);
      } else {
        Lasr('R', 'V', 'F', nru, np1, WORK.ptr(1), WORK.ptr(np1), u, ldu);
      }
    }
    if (ncc > 0) {
      if (sqre1 == 0) {
        Lasr('L', 'V', 'F', n, ncc, WORK.ptr(1), WORK.ptr(np1), c, ldc);
      } else {
        Lasr('L', 'V', 'F', np1, ncc, WORK.ptr(1), WORK.ptr(np1), c, ldc);
      }
    }
  }
  // Call DBDSQR to compute the SVD of the reduced real
  // N-by-N upper bidiagonal matrix.
  info = Bdsqr('U', n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work);
  // Sort the singular values into ascending order (insertion sort on
  // singular values, but only one transposition per singular vector)
  for (i = 1; i <= n; ++i) {
    // Scan for smallest D(I).
    isub = i;
    smin = D(i);
    for (j = i + 1; j <= n; ++j) {
      if (D(j) < smin) {
        isub = j;
        smin = D(j);
      }
    }
    if (isub != i) {
      // Swap singular values and vectors.
      D(isub) = D(i);
      D(i) = smin;
      if (ncvt > 0) Swap(ncvt, VT.ptr(isub, 1), ldvt, VT.ptr(i, 1), ldvt);
      if (nru > 0) Swap(nru, U.ptr(1, isub), 1, U.ptr(1, i), 1);
      if (ncc > 0) Swap(ncc, C.ptr(isub, 1), ldc, C.ptr(i, 1), ldc);
    }
  }
  return info;
}

// xLASD0: the SVD of the N-by-(N+SQRE) upper bidiagonal matrix (D, E) by
// divide and conquer, with subproblems of at most SMLSIZ rows:
// B = U * diag(D) * VT**T. IWORK has 8*N and WORK 3*M**2+2*M elements,
// M = N+SQRE. Returns INFO = 1 if a singular value did not converge.
template <typename R>
blasint Lasd0(blasint n, blasint sqre, R* d, R* e, R* u, blasint ldu, R* vt,
              blasint ldvt, blasint smlsiz, blasint* iwork, R* work) {
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldvt);
  const Vector<R> WORK(work);
  blasint i, i1, ic, idxq, idxqc, im1, inode, itemp, iwk, j, lf, ll, lvl, m,
      ncc, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqrei;
  R alpha, beta;
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -2;
  }
  m = n + sqre;
  if (ldu < n) {
    info = -6;
  } else if (ldvt < m) {
    info = -8;
  } else if (smlsiz < 3) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<R>("LASD0", -info);
    return info;
  }
  // If the input matrix is too small, call DLASDQ to find the SVD.
  if (n <= smlsiz) {
    info = Lasdq('U', sqre, n, m, n, 0, d, e, vt, ldvt, u, ldu, u, ldu, work);
    return info;
  }
  // Set up the computation tree.
  inode = 1;
  ndiml = inode + n;
  ndimr = ndiml + n;
  idxq = ndimr + n;
  iwk = idxq + n;
  Lasdt<R>(n, &nlvl, &nd, IWORK.ptr(inode), IWORK.ptr(ndiml), IWORK.ptr(ndimr),
           smlsiz);
  // For the nodes on bottom level of the tree, solve
  // their subproblems by DLASDQ.
  ndb1 = (nd + 1) / 2;
  ncc = 0;
  for (i = ndb1; i <= nd; ++i) {
    // IC : center row of each node
    // NL : number of rows of left  subproblem
    // NR : number of rows of right subproblem
    // NLF: starting row of the left   subproblem
    // NRF: starting row of the right  subproblem
    i1 = i - 1;
    ic = IWORK(inode + i1);
    nl = IWORK(ndiml + i1);
    nlp1 = nl + 1;
    nr = IWORK(ndimr + i1);
    nrp1 = nr + 1;
    nlf = ic - nl;
    nrf = ic + 1;
    sqrei = 1;
    info = Lasdq('U', sqrei, nl, nlp1, nl, ncc, D.ptr(nlf), E.ptr(nlf),
                 VT.ptr(nlf, nlf), ldvt, U.ptr(nlf, nlf), ldu, U.ptr(nlf, nlf),
                 ldu, work);
    if (info != 0) {
      return info;
    }
    itemp = idxq + nlf - 2;
    for (j = 1; j <= nl; ++j) {
      IWORK(itemp + j) = j;
    }
    if (i == nd) {
      sqrei = sqre;
    } else {
      sqrei = 1;
    }
    nrp1 = nr + sqrei;
    info = Lasdq('U', sqrei, nr, nrp1, nr, ncc, D.ptr(nrf), E.ptr(nrf),
                 VT.ptr(nrf, nrf), ldvt, U.ptr(nrf, nrf), ldu, U.ptr(nrf, nrf),
                 ldu, work);
    if (info != 0) {
      return info;
    }
    itemp = idxq + ic;
    for (j = 1; j <= nr; ++j) {
      IWORK(itemp + j - 1) = j;
    }
  }
  // Now conquer each subproblem bottom-up.
  for (lvl = nlvl; lvl >= 1; --lvl) {
    // Find the first node LF and last node LL on the
    // current level LVL.
    if (lvl == 1) {
      lf = 1;
      ll = 1;
    } else {
      lf = (blasint{1} << (lvl - 1));
      ll = 2 * lf - 1;
    }
    for (i = lf; i <= ll; ++i) {
      im1 = i - 1;
      ic = IWORK(inode + im1);
      nl = IWORK(ndiml + im1);
      nr = IWORK(ndimr + im1);
      nlf = ic - nl;
      if ((sqre == 0) && (i == ll)) {
        sqrei = sqre;
      } else {
        sqrei = 1;
      }
      idxqc = idxq + nlf - 1;
      alpha = D(ic);
      beta = E(ic);
      info =
          Lasd1(nl, nr, sqrei, D.ptr(nlf), &alpha, &beta, U.ptr(nlf, nlf), ldu,
                VT.ptr(nlf, nlf), ldvt, IWORK.ptr(idxqc), IWORK.ptr(iwk), work);
      // Report the possible convergence failure.
      if (info != 0) {
        return info;
      }
    }
  }
  return info;
}

// xLASDA: the singular values of the N-by-(N+SQRE) upper bidiagonal matrix
// (D, E) by divide and conquer, with the singular vectors in factored form
// if ICOMPQ = 1 (the U and VT of the leaves, and per level K, DIFL, DIFR, Z,
// POLES, GIVPTR, GIVCOL, PERM, GIVNUM, C and S), as needed by xBDSDC's
// COMPQ = 'P' and xLALSA. Returns INFO = 1 if a singular value did not
// converge.
template <typename R>
blasint Lasda(blasint icompq, blasint smlsiz, blasint n, blasint sqre, R* d,
              R* e, R* u, blasint ldu, R* vt, blasint* k, R* difl, R* difr,
              R* z, R* poles, blasint* givptr, blasint* givcol, blasint ldgcol,
              blasint* perm, R* givnum, R* c, R* s, R* work, blasint* iwork) {
  const R zero = R(0);
  const R one = R(1);
  const Vector<R> C(c);
  const Vector<R> D(d);
  const Matrix<R> DIFL(difl, ldu);
  const Matrix<R> DIFR(difr, ldu);
  const Vector<R> E(e);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldu);
  const Vector<blasint> GIVPTR(givptr);
  const Vector<blasint> IWORK(iwork);
  const Vector<blasint> K(k);
  const Matrix<blasint> PERM(perm, ldgcol);
  const Matrix<R> POLES(poles, ldu);
  const Vector<R> S(s);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldu);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldu);
  blasint i, i1, ic, idxq, idxqi, im1, inode, itemp, iwk, j, lf, ll, lvl, lvl2,
      m, ncc, nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, nru,
      nwork1, nwork2, smlszp, sqrei, vf, vfi, vl, vli;
  R alpha, beta;
  blasint info = 0;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (smlsiz < 3) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -4;
  } else if (ldu < (n + sqre)) {
    info = -8;
  } else if (ldgcol < n) {
    info = -17;
  }
  if (info != 0) {
    Xerbla<R>("LASDA", -info);
    return info;
  }
  m = n + sqre;
  // If the input matrix is too small, call DLASDQ to find the SVD.
  if (n <= smlsiz) {
    if (icompq == 0) {
      info = Lasdq('U', sqre, n, 0, 0, 0, d, e, vt, ldu, u, ldu, u, ldu, work);
    } else {
      info = Lasdq('U', sqre, n, m, n, 0, d, e, vt, ldu, u, ldu, u, ldu, work);
    }
    return info;
  }
  // Book-keeping and  set up the computation tree.
  inode = 1;
  ndiml = inode + n;
  ndimr = ndiml + n;
  idxq = ndimr + n;
  iwk = idxq + n;
  ncc = 0;
  nru = 0;
  smlszp = smlsiz + 1;
  vf = 1;
  vl = vf + m;
  nwork1 = vl + m;
  nwork2 = nwork1 + smlszp * smlszp;
  Lasdt<R>(n, &nlvl, &nd, IWORK.ptr(inode), IWORK.ptr(ndiml), IWORK.ptr(ndimr),
           smlsiz);
  // for the nodes on bottom level of the tree, solve
  // their subproblems by DLASDQ.
  ndb1 = (nd + 1) / 2;
  for (i = ndb1; i <= nd; ++i) {
    // IC : center row of each node
    // NL : number of rows of left  subproblem
    // NR : number of rows of right subproblem
    // NLF: starting row of the left   subproblem
    // NRF: starting row of the right  subproblem
    i1 = i - 1;
    ic = IWORK(inode + i1);
    nl = IWORK(ndiml + i1);
    nlp1 = nl + 1;
    nr = IWORK(ndimr + i1);
    nlf = ic - nl;
    nrf = ic + 1;
    idxqi = idxq + nlf - 2;
    vfi = vf + nlf - 1;
    vli = vl + nlf - 1;
    sqrei = 1;
    if (icompq == 0) {
      Laset('A', nlp1, nlp1, zero, one, WORK.ptr(nwork1), smlszp);
      info = Lasdq('U', sqrei, nl, nlp1, nru, ncc, D.ptr(nlf), E.ptr(nlf),
                   WORK.ptr(nwork1), smlszp, WORK.ptr(nwork2), nl,
                   WORK.ptr(nwork2), nl, WORK.ptr(nwork2));
      itemp = nwork1 + nl * smlszp;
      Copy(nlp1, WORK.ptr(nwork1), 1, WORK.ptr(vfi), 1);
      Copy(nlp1, WORK.ptr(itemp), 1, WORK.ptr(vli), 1);
    } else {
      Laset('A', nl, nl, zero, one, U.ptr(nlf, 1), ldu);
      Laset('A', nlp1, nlp1, zero, one, VT.ptr(nlf, 1), ldu);
      info = Lasdq('U', sqrei, nl, nlp1, nl, ncc, D.ptr(nlf), E.ptr(nlf),
                   VT.ptr(nlf, 1), ldu, U.ptr(nlf, 1), ldu, U.ptr(nlf, 1), ldu,
                   WORK.ptr(nwork1));
      Copy(nlp1, VT.ptr(nlf, 1), 1, WORK.ptr(vfi), 1);
      Copy(nlp1, VT.ptr(nlf, nlp1), 1, WORK.ptr(vli), 1);
    }
    if (info != 0) {
      return info;
    }
    for (j = 1; j <= nl; ++j) {
      IWORK(idxqi + j) = j;
    }
    if ((i == nd) && (sqre == 0)) {
      sqrei = 0;
    } else {
      sqrei = 1;
    }
    idxqi = idxqi + nlp1;
    vfi = vfi + nlp1;
    vli = vli + nlp1;
    nrp1 = nr + sqrei;
    if (icompq == 0) {
      Laset('A', nrp1, nrp1, zero, one, WORK.ptr(nwork1), smlszp);
      info = Lasdq('U', sqrei, nr, nrp1, nru, ncc, D.ptr(nrf), E.ptr(nrf),
                   WORK.ptr(nwork1), smlszp, WORK.ptr(nwork2), nr,
                   WORK.ptr(nwork2), nr, WORK.ptr(nwork2));
      itemp = nwork1 + (nrp1 - 1) * smlszp;
      Copy(nrp1, WORK.ptr(nwork1), 1, WORK.ptr(vfi), 1);
      Copy(nrp1, WORK.ptr(itemp), 1, WORK.ptr(vli), 1);
    } else {
      Laset('A', nr, nr, zero, one, U.ptr(nrf, 1), ldu);
      Laset('A', nrp1, nrp1, zero, one, VT.ptr(nrf, 1), ldu);
      info = Lasdq('U', sqrei, nr, nrp1, nr, ncc, D.ptr(nrf), E.ptr(nrf),
                   VT.ptr(nrf, 1), ldu, U.ptr(nrf, 1), ldu, U.ptr(nrf, 1), ldu,
                   WORK.ptr(nwork1));
      Copy(nrp1, VT.ptr(nrf, 1), 1, WORK.ptr(vfi), 1);
      Copy(nrp1, VT.ptr(nrf, nrp1), 1, WORK.ptr(vli), 1);
    }
    if (info != 0) {
      return info;
    }
    for (j = 1; j <= nr; ++j) {
      IWORK(idxqi + j) = j;
    }
  }
  // Now conquer each subproblem bottom-up.
  j = (blasint{1} << nlvl);
  for (lvl = nlvl; lvl >= 1; --lvl) {
    lvl2 = lvl * 2 - 1;
    // Find the first node LF and last node LL on
    // the current level LVL.
    if (lvl == 1) {
      lf = 1;
      ll = 1;
    } else {
      lf = (blasint{1} << (lvl - 1));
      ll = 2 * lf - 1;
    }
    for (i = lf; i <= ll; ++i) {
      im1 = i - 1;
      ic = IWORK(inode + im1);
      nl = IWORK(ndiml + im1);
      nr = IWORK(ndimr + im1);
      nlf = ic - nl;
      nrf = ic + 1;
      if (i == ll) {
        sqrei = sqre;
      } else {
        sqrei = 1;
      }
      vfi = vf + nlf - 1;
      vli = vl + nlf - 1;
      idxqi = idxq + nlf - 1;
      alpha = D(ic);
      beta = E(ic);
      if (icompq == 0) {
        info = Lasd6(icompq, nl, nr, sqrei, D.ptr(nlf), WORK.ptr(vfi),
                     WORK.ptr(vli), &alpha, &beta, IWORK.ptr(idxqi), perm,
                     GIVPTR.ptr(1), givcol, ldgcol, givnum, ldu, poles, difl,
                     difr, z, K.ptr(1), C.ptr(1), S.ptr(1), WORK.ptr(nwork1),
                     IWORK.ptr(iwk));
      } else {
        j = j - 1;
        info = Lasd6(icompq, nl, nr, sqrei, D.ptr(nlf), WORK.ptr(vfi),
                     WORK.ptr(vli), &alpha, &beta, IWORK.ptr(idxqi),
                     PERM.ptr(nlf, lvl), GIVPTR.ptr(j), GIVCOL.ptr(nlf, lvl2),
                     ldgcol, GIVNUM.ptr(nlf, lvl2), ldu, POLES.ptr(nlf, lvl2),
                     DIFL.ptr(nlf, lvl), DIFR.ptr(nlf, lvl2), Z.ptr(nlf, lvl),
                     K.ptr(j), C.ptr(j), S.ptr(j), WORK.ptr(nwork1),
                     IWORK.ptr(iwk));
      }
      if (info != 0) {
        return info;
      }
    }
  }
  return info;
}

// xBDSDC: the SVD of the N-by-N upper (UPLO = 'U') or lower ('L')
// bidiagonal matrix B = U * S * VT by divide and conquer: the singular
// values only (COMPQ = 'N'), also the singular vectors in compact form in
// Q and IQ (COMPQ = 'P'), or U and VT (COMPQ = 'I'). The singular values
// are returned in D in decreasing order. WORK has 4*N (COMPQ = 'N'),
// 6*N (COMPQ = 'P') or 3*N**2+4*N elements, IWORK 8*N. Returns INFO: 0,
// -i for an illegal argument i, or 1 if a singular value did not converge.
//
// For COMPQ = 'P', the code uses up to N*(7 + 2*SMLSIZ + 8*MLVL) elements
// of Q and N*(3 + 3*MLVL) of IQ, MLVL = INT(LOG2(N/(SMLSIZ+1))) + 1, which
// is more than the reference documents (N*(11 + 2*SMLSIZ + 8*(MLVL-1)) and
// N*(3 + 3*(MLVL-1))).
// For N <= SMLSIZ, the reference stores U and VT in Q with leading dimension
// N one column apart, so that they overlap; this port keeps that (no
// routine of LAPACK uses COMPQ = 'P').
template <typename R>
blasint Bdsdc(char uplo, char compq, blasint n, R* d, R* e, R* u, blasint ldu,
              R* vt, blasint ldvt, R* q, blasint* iq, R* work, blasint* iwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IQ(iq);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> Q(q);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldvt);
  const Vector<R> WORK(work);
  blasint givptr, i, icompq, ii, iu, iuplo, j, k, kk, mlvl, nm1, nsize, perm,
      qstart, smlsiz, smlszp, sqre, start, wstart;
  // Set and used only for ICOMPQ = 2.
  blasint difl = 0, difr = 0, givcol = 0, givnum = 0, ic = 0, is = 0, ivt = 0,
          poles = 0, z = 0;
  R cs, eps, orgnrm, p, r, sn;
  blasint info = 0;
  iuplo = 0;
  if (Lsame(uplo, 'U')) iuplo = 1;
  if (Lsame(uplo, 'L')) iuplo = 2;
  if (Lsame(compq, 'N')) {
    icompq = 0;
  } else if (Lsame(compq, 'P')) {
    icompq = 1;
  } else if (Lsame(compq, 'I')) {
    icompq = 2;
  } else {
    icompq = -1;
  }
  if (iuplo == 0) {
    info = -1;
  } else if (icompq < 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if ((ldu < 1) || ((icompq == 2) && (ldu < n))) {
    info = -7;
  } else if ((ldvt < 1) || ((icompq == 2) && (ldvt < n))) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<R>("BDSDC", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  smlsiz = Ilaenv<R>(9, "BDSDC", " ", 0, 0, 0, 0);
  if (n == 1) {
    if (icompq == 1) {
      Q(1) = std::copysign(one, D(1));
      Q(1 + smlsiz * n) = one;
    } else if (icompq == 2) {
      U(1, 1) = std::copysign(one, D(1));
      VT(1, 1) = one;
    }
    D(1) = std::abs(D(1));
    return info;
  }
  nm1 = n - 1;
  // If matrix lower bidiagonal, rotate to be upper bidiagonal
  // by applying Givens rotations on the left
  wstart = 1;
  qstart = 3;
  if (icompq == 1) {
    Copy(n, d, 1, Q.ptr(1), 1);
    Copy(n - 1, e, 1, Q.ptr(n + 1), 1);
  }
  if (iuplo == 2) {
    qstart = 5;
    if (icompq == 2) wstart = 2 * n - 1;
    for (i = 1; i <= n - 1; ++i) {
      Lartg(D(i), E(i), &cs, &sn, &r);
      D(i) = r;
      E(i) = sn * D(i + 1);
      D(i + 1) = cs * D(i + 1);
      if (icompq == 1) {
        Q(i + 2 * n) = cs;
        Q(i + 3 * n) = sn;
      } else if (icompq == 2) {
        WORK(i) = cs;
        WORK(nm1 + i) = -sn;
      }
    }
  }
  // If ICOMPQ = 0, use DLASDQ to compute the singular values.
  if (icompq == 0) {
    // Ignore WSTART, instead using WORK( 1 ), since the two vectors
    // for CS and -SN above are added only if ICOMPQ == 2,
    // and adding them exceeds documented WORK size of 4*n.
    info =
        Lasdq('U', 0, n, 0, 0, 0, d, e, vt, ldvt, u, ldu, u, ldu, WORK.ptr(1));
    goto L40;
  }
  // If N is smaller than the minimum divide size SMLSIZ, then solve
  // the problem with another solver.
  if (n <= smlsiz) {
    if (icompq == 2) {
      Laset('A', n, n, zero, one, u, ldu);
      Laset('A', n, n, zero, one, vt, ldvt);
      info = Lasdq('U', 0, n, n, n, 0, d, e, vt, ldvt, u, ldu, u, ldu,
                   WORK.ptr(wstart));
    } else if (icompq == 1) {
      iu = 1;
      ivt = iu + n;
      Laset('A', n, n, zero, one, Q.ptr(iu + (qstart - 1) * n), n);
      Laset('A', n, n, zero, one, Q.ptr(ivt + (qstart - 1) * n), n);
      info = Lasdq('U', 0, n, n, n, 0, d, e, Q.ptr(ivt + (qstart - 1) * n), n,
                   Q.ptr(iu + (qstart - 1) * n), n,
                   Q.ptr(iu + (qstart - 1) * n), n, WORK.ptr(wstart));
    }
    goto L40;
  }
  if (icompq == 2) {
    Laset('A', n, n, zero, one, u, ldu);
    Laset('A', n, n, zero, one, vt, ldvt);
  }
  // Scale.
  orgnrm = Lanst('M', n, d, e);
  if (orgnrm == zero) return info;
  Lascl('G', 0, 0, orgnrm, one, n, 1, d, n);
  Lascl('G', 0, 0, orgnrm, one, nm1, 1, e, nm1);
  eps = (R(0.9e+0)) * Lamch<R>('E');
  mlvl =
      static_cast<blasint>(std::log(R(n) / R(smlsiz + 1)) / std::log(two)) + 1;
  smlszp = smlsiz + 1;
  if (icompq == 1) {
    iu = 1;
    ivt = 1 + smlsiz;
    difl = ivt + smlszp;
    difr = difl + mlvl;
    z = difr + mlvl * 2;
    ic = z + mlvl;
    is = ic + 1;
    poles = is + 1;
    givnum = poles + 2 * mlvl;
    k = 1;
    givptr = 2;
    perm = 3;
    givcol = perm + mlvl;
  }
  for (i = 1; i <= n; ++i) {
    if (std::abs(D(i)) < eps) {
      D(i) = std::copysign(eps, D(i));
    }
  }
  start = 1;
  sqre = 0;
  for (i = 1; i <= nm1; ++i) {
    if ((std::abs(E(i)) < eps) || (i == nm1)) {
      // Subproblem found. First determine its size and then
      // apply divide and conquer on it.
      if (i < nm1) {
        // A subproblem with E(I) small for I < NM1.
        nsize = i - start + 1;
      } else if (std::abs(E(i)) >= eps) {
        // A subproblem with E(NM1) not too small but I = NM1.
        nsize = n - start + 1;
      } else {
        // A subproblem with E(NM1) small. This implies an
        // 1-by-1 subproblem at D(N). Solve this 1-by-1 problem
        // first.
        nsize = i - start + 1;
        if (icompq == 2) {
          U(n, n) = std::copysign(one, D(n));
          VT(n, n) = one;
        } else if (icompq == 1) {
          Q(n + (qstart - 1) * n) = std::copysign(one, D(n));
          Q(n + (smlsiz + qstart - 1) * n) = one;
        }
        D(n) = std::abs(D(n));
      }
      if (icompq == 2) {
        info = Lasd0(nsize, sqre, D.ptr(start), E.ptr(start),
                     U.ptr(start, start), ldu, VT.ptr(start, start), ldvt,
                     smlsiz, iwork, WORK.ptr(wstart));
      } else {
        info = Lasda(
            icompq, smlsiz, nsize, sqre, D.ptr(start), E.ptr(start),
            Q.ptr(start + (iu + qstart - 2) * n), n,
            Q.ptr(start + (ivt + qstart - 2) * n), IQ.ptr(start + k * n),
            Q.ptr(start + (difl + qstart - 2) * n),
            Q.ptr(start + (difr + qstart - 2) * n),
            Q.ptr(start + (z + qstart - 2) * n),
            Q.ptr(start + (poles + qstart - 2) * n), IQ.ptr(start + givptr * n),
            IQ.ptr(start + givcol * n), n, IQ.ptr(start + perm * n),
            Q.ptr(start + (givnum + qstart - 2) * n),
            Q.ptr(start + (ic + qstart - 2) * n),
            Q.ptr(start + (is + qstart - 2) * n), WORK.ptr(wstart), iwork);
      }
      if (info != 0) {
        return info;
      }
      start = i + 1;
    }
  }
  // Unscale
  Lascl('G', 0, 0, one, orgnrm, n, 1, d, n);
L40:
  // Use Selection Sort to minimize swaps of singular vectors
  for (ii = 2; ii <= n; ++ii) {
    i = ii - 1;
    kk = i;
    p = D(i);
    for (j = ii; j <= n; ++j) {
      if (D(j) > p) {
        kk = j;
        p = D(j);
      }
    }
    if (kk != i) {
      D(kk) = D(i);
      D(i) = p;
      if (icompq == 1) {
        IQ(i) = kk;
      } else if (icompq == 2) {
        Swap(n, U.ptr(1, i), 1, U.ptr(1, kk), 1);
        Swap(n, VT.ptr(i, 1), ldvt, VT.ptr(kk, 1), ldvt);
      }
    } else if (icompq == 1) {
      IQ(i) = i;
    }
  }
  // If ICOMPQ = 1, use IQ(N,1) as the indicator for UPLO
  if (icompq == 1) {
    if (iuplo == 1) {
      IQ(n) = 1;
    } else {
      IQ(n) = 0;
    }
  }
  // If B is lower bidiagonal, update U by those Givens rotations
  // which rotated B to be upper bidiagonal
  if ((iuplo == 2) && (icompq == 2))
    Lasr('L', 'V', 'B', n, n, WORK.ptr(1), WORK.ptr(n), u, ldu);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(R)                                          \
  template void Lamrg<R>(blasint, blasint, const R*, blasint, blasint,         \
                         blasint*);                                            \
  template void Lasdt<R>(blasint, blasint*, blasint*, blasint*, blasint*,      \
                         blasint*, blasint);                                   \
  template blasint Laed6<R>(blasint, bool, R, const R*, const R*, R, R*);      \
  template void Lasd5<R>(blasint, const R*, const R*, R*, R, R*, R*);          \
  template blasint Lasd4<R>(blasint, blasint, const R*, const R*, R*, R, R*,   \
                            R*);                                               \
  template blasint Lasd8<R>(blasint, blasint, R*, R*, R*, R*, R*, R*, blasint, \
                            R*, R*);                                           \
  template blasint Lasd7<R>(blasint, blasint, blasint, blasint, blasint*, R*,  \
                            R*, R*, R*, R*, R*, R*, R, R, R*, blasint*,        \
                            blasint*, blasint*, blasint*, blasint*, blasint*,  \
                            blasint, R*, blasint, R*, R*);                     \
  template blasint Lasd6<R>(blasint, blasint, blasint, blasint, R*, R*, R*,    \
                            R*, R*, blasint*, blasint*, blasint*, blasint*,    \
                            blasint, R*, blasint, R*, R*, R*, R*, blasint*,    \
                            R*, R*, R*, blasint*);                             \
  template blasint Lasd3<R>(blasint, blasint, blasint, blasint, R*, R*,        \
                            blasint, R*, R*, blasint, R*, blasint, R*,         \
                            blasint, R*, blasint, const blasint*,              \
                            const blasint*, R*);                               \
  template blasint Lasd2<R>(blasint, blasint, blasint, blasint*, R*, R*, R, R, \
                            R*, blasint, R*, blasint, R*, R*, blasint, R*,     \
                            blasint, blasint*, blasint*, blasint*, blasint*,   \
                            blasint*);                                         \
  template blasint Lasd1<R>(blasint, blasint, blasint, R*, R*, R*, R*,         \
                            blasint, R*, blasint, blasint*, blasint*, R*);     \
  template blasint Lasdq<R>(char, blasint, blasint, blasint, blasint, blasint, \
                            R*, R*, R*, blasint, R*, blasint, R*, blasint,     \
                            R*);                                               \
  template blasint Lasd0<R>(blasint, blasint, R*, R*, R*, blasint, R*,         \
                            blasint, blasint, blasint*, R*);                   \
  template blasint Lasda<R>(blasint, blasint, blasint, blasint, R*, R*, R*,    \
                            blasint, R*, blasint*, R*, R*, R*, R*, blasint*,   \
                            blasint*, blasint, blasint*, R*, R*, R*, R*,       \
                            blasint*);                                         \
  template blasint Bdsdc<R>(char, char, blasint, R*, R*, R*, blasint, R*,      \
                            blasint, R*, blasint*, R*, blasint*);
HWYBLAS_LAPACK_INSTANTIATE(float)
HWYBLAS_LAPACK_INSTANTIATE(double)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

extern "C" {

// The Fortran LOGICAL argument ORGATI of xLAED6 is a hwyblas_int.
#define HWYBLAS_LAPACK_BDSDC(prefix, R)                                        \
  HWYBLAS_API void prefix##lamrg_(                                             \
      const hwyblas_int* n1, const hwyblas_int* n2, const R* a,                \
      const hwyblas_int* dtrd1, const hwyblas_int* dtrd2,                      \
      hwyblas_int* index) {                                                    \
    hwyblas::lapack::Lamrg(*n1, *n2, a, *dtrd1, *dtrd2, index);                \
  }                                                                            \
  HWYBLAS_API void prefix##lasdt_(const hwyblas_int* n, hwyblas_int* lvl,      \
                                  hwyblas_int* nd, hwyblas_int* inode,         \
                                  hwyblas_int* ndiml, hwyblas_int* ndimr,      \
                                  const hwyblas_int* msub) {                   \
    hwyblas::lapack::Lasdt<R>(*n, lvl, nd, inode, ndiml, ndimr, *msub);        \
  }                                                                            \
  HWYBLAS_API void prefix##laed6_(                                             \
      const hwyblas_int* kniter, const hwyblas_int* orgati, const R* rho,      \
      const R* d, const R* z, const R* finit, R* tau, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Laed6(*kniter, *orgati != 0, *rho, d, z, *finit,  \
                                   tau);                                       \
  }                                                                            \
  HWYBLAS_API void prefix##lasd5_(const hwyblas_int* i, const R* d,            \
                                  const R* z, R* delta, const R* rho,          \
                                  R* dsigma, R* work) {                        \
    hwyblas::lapack::Lasd5(*i, d, z, delta, *rho, dsigma, work);               \
  }                                                                            \
  HWYBLAS_API void prefix##lasd4_(                                             \
      const hwyblas_int* n, const hwyblas_int* i, const R* d, const R* z,      \
      R* delta, const R* rho, R* sigma, R* work, hwyblas_int* info) {          \
    *info = hwyblas::lapack::Lasd4(*n, *i, d, z, delta, *rho, sigma, work);    \
  }                                                                            \
  HWYBLAS_API void prefix##lasd8_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* k, R* d, R* z, R* vf,      \
      R* vl, R* difl, R* difr, const hwyblas_int* lddifr, R* dsigma, R* work,  \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lasd8(*icompq, *k, d, z, vf, vl, difl, difr,      \
                                   *lddifr, dsigma, work);                     \
  }                                                                            \
  HWYBLAS_API void prefix##lasd7_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr, \
      const hwyblas_int* sqre, hwyblas_int* k, R* d, R* z, R* zw, R* vf,       \
      R* vfw, R* vl, R* vlw, const R* alpha, const R* beta, R* dsigma,         \
      hwyblas_int* idx, hwyblas_int* idxp, hwyblas_int* idxq,                  \
      hwyblas_int* perm, hwyblas_int* givptr, hwyblas_int* givcol,             \
      const hwyblas_int* ldgcol, R* givnum, const hwyblas_int* ldgnum, R* c,   \
      R* s, hwyblas_int* info) {                                               \
    *info = hwyblas::lapack::Lasd7(*icompq, *nl, *nr, *sqre, k, d, z, zw, vf,  \
                                   vfw, vl, vlw, *alpha, *beta, dsigma, idx,   \
                                   idxp, idxq, perm, givptr, givcol, *ldgcol,  \
                                   givnum, *ldgnum, c, s);                     \
  }                                                                            \
  HWYBLAS_API void prefix##lasd6_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr, \
      const hwyblas_int* sqre, R* d, R* vf, R* vl, R* alpha, R* beta,          \
      hwyblas_int* idxq, hwyblas_int* perm, hwyblas_int* givptr,               \
      hwyblas_int* givcol, const hwyblas_int* ldgcol, R* givnum,               \
      const hwyblas_int* ldgnum, R* poles, R* difl, R* difr, R* z,             \
      hwyblas_int* k, R* c, R* s, R* work, hwyblas_int* iwork,                 \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lasd6(*icompq, *nl, *nr, *sqre, d, vf, vl, alpha, \
                                   beta, idxq, perm, givptr, givcol, *ldgcol,  \
                                   givnum, *ldgnum, poles, difl, difr, z, k,   \
                                   c, s, work, iwork);                         \
  }                                                                            \
  HWYBLAS_API void prefix##lasd3_(                                             \
      const hwyblas_int* nl, const hwyblas_int* nr, const hwyblas_int* sqre,   \
      const hwyblas_int* k, R* d, R* q, const hwyblas_int* ldq, R* dsigma,     \
      R* u, const hwyblas_int* ldu, R* u2, const hwyblas_int* ldu2, R* vt,     \
      const hwyblas_int* ldvt, R* vt2, const hwyblas_int* ldvt2,               \
      const hwyblas_int* idxc, const hwyblas_int* ctot, R* z,                  \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lasd3(*nl, *nr, *sqre, *k, d, q, *ldq, dsigma, u, \
                                   *ldu, u2, *ldu2, vt, *ldvt, vt2, *ldvt2,    \
                                   idxc, ctot, z);                             \
  }                                                                            \
  HWYBLAS_API void prefix##lasd2_(                                             \
      const hwyblas_int* nl, const hwyblas_int* nr, const hwyblas_int* sqre,   \
      hwyblas_int* k, R* d, R* z, const R* alpha, const R* beta, R* u,         \
      const hwyblas_int* ldu, R* vt, const hwyblas_int* ldvt, R* dsigma,       \
      R* u2, const hwyblas_int* ldu2, R* vt2, const hwyblas_int* ldvt2,        \
      hwyblas_int* idxp, hwyblas_int* idx, hwyblas_int* idxc,                  \
      hwyblas_int* idxq, hwyblas_int* coltyp, hwyblas_int* info) {             \
    *info = hwyblas::lapack::Lasd2(*nl, *nr, *sqre, k, d, z, *alpha, *beta, u, \
                                   *ldu, vt, *ldvt, dsigma, u2, *ldu2, vt2,    \
                                   *ldvt2, idxp, idx, idxc, idxq, coltyp);     \
  }                                                                            \
  HWYBLAS_API void prefix##lasd1_(                                             \
      const hwyblas_int* nl, const hwyblas_int* nr, const hwyblas_int* sqre,   \
      R* d, R* alpha, R* beta, R* u, const hwyblas_int* ldu, R* vt,            \
      const hwyblas_int* ldvt, hwyblas_int* idxq, hwyblas_int* iwork, R* work, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lasd1(*nl, *nr, *sqre, d, alpha, beta, u, *ldu,   \
                                   vt, *ldvt, idxq, iwork, work);              \
  }                                                                            \
  HWYBLAS_API void prefix##lasdq_(                                             \
      const char* uplo, const hwyblas_int* sqre, const hwyblas_int* n,         \
      const hwyblas_int* ncvt, const hwyblas_int* nru, const hwyblas_int* ncc, \
      R* d, R* e, R* vt, const hwyblas_int* ldvt, R* u,                        \
      const hwyblas_int* ldu, R* c, const hwyblas_int* ldc, R* work,           \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lasdq(*uplo, *sqre, *n, *ncvt, *nru, *ncc, d, e,  \
                                   vt, *ldvt, u, *ldu, c, *ldc, work);         \
  }                                                                            \
  HWYBLAS_API void prefix##lasd0_(                                             \
      const hwyblas_int* n, const hwyblas_int* sqre, R* d, R* e, R* u,         \
      const hwyblas_int* ldu, R* vt, const hwyblas_int* ldvt,                  \
      const hwyblas_int* smlsiz, hwyblas_int* iwork, R* work,                  \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lasd0(*n, *sqre, d, e, u, *ldu, vt, *ldvt,        \
                                   *smlsiz, iwork, work);                      \
  }                                                                            \
  HWYBLAS_API void prefix##lasda_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* smlsiz,                    \
      const hwyblas_int* n, const hwyblas_int* sqre, R* d, R* e, R* u,         \
      const hwyblas_int* ldu, R* vt, hwyblas_int* k, R* difl, R* difr, R* z,   \
      R* poles, hwyblas_int* givptr, hwyblas_int* givcol,                      \
      const hwyblas_int* ldgcol, hwyblas_int* perm, R* givnum, R* c, R* s,     \
      R* work, hwyblas_int* iwork, hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Lasda(                                            \
        *icompq, *smlsiz, *n, *sqre, d, e, u, *ldu, vt, k, difl, difr, z,      \
        poles, givptr, givcol, *ldgcol, perm, givnum, c, s, work, iwork);      \
  }                                                                            \
  HWYBLAS_API void prefix##bdsdc_(                                             \
      const char* uplo, const char* compq, const hwyblas_int* n, R* d, R* e,   \
      R* u, const hwyblas_int* ldu, R* vt, const hwyblas_int* ldvt, R* q,      \
      hwyblas_int* iq, R* work, hwyblas_int* iwork, hwyblas_int* info) {       \
    *info = hwyblas::lapack::Bdsdc(*uplo, *compq, *n, d, e, u, *ldu, vt,       \
                                   *ldvt, q, iq, work, iwork);                 \
  }

HWYBLAS_LAPACK_BDSDC(s, float)
HWYBLAS_LAPACK_BDSDC(d, double)

#undef HWYBLAS_LAPACK_BDSDC

}  // extern "C"
