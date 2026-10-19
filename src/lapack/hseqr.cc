// The eigenvalues and the Schur form of an upper Hessenberg matrix, by the
// small-bulge multishift QR algorithm with aggressive early deflation:
// xHSEQR (driver), xLAQR0 / xLAQR4 (QR sweeps), xLAQR2 / xLAQR3 (aggressive
// early deflation), xLAQR5 (one sweep of bulge chasing), xLAQR1 (the
// first column of the shift polynomial), xLAHQR (double-shift QR for small
// matrices) and xLANV2 (Schur factorization of a real 2-by-2 matrix).
//
// The real and the complex routines differ throughout (the real ones keep
// 2-by-2 blocks for complex conjugate eigenvalue pairs, return the
// eigenvalues as WR and WI and use pairs of shifts), so they are ported as
// separate functions with the same name; their argument lists differ.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <cmath>
#include <vector>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// The routines keep the reference's branches, some of which end in the same
// statements (e.g. the tests for the end of a 2-by-2 block in xLAQR2).
// NOLINTBEGIN(bugprone-branch-clone)

// xLANV2 (real only): the Schur factorization of the real 2-by-2
// nonsymmetric matrix in standardized form,
//   [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ]
//   [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ]
// where either CC = 0 (real eigenvalues AA and DD) or AA = DD and BB*CC < 0
// (complex conjugate eigenvalues AA +- sqrt(BB*CC)). A, B, C and D are
// overwritten with AA, BB, CC and DD; (RT1R, RT1I) and (RT2R, RT2I) are
// the eigenvalues.
template <typename R>
void Lanv2(R* a, R* b, R* c, R* d, R* rt1r, R* rt1i, R* rt2r, R* rt2i, R* cs,
           R* sn) {
  const R zero = R(0);
  const R half = R(0.5e+0);
  const R one = R(1);
  const R two = R(2);
  const R multpl = R(4);
  blasint count;
  R aa, bb, bcmax, bcmis, cc, cs1, dd, eps, p, sab, sac, safmin, safmn2, safmx2,
      scale, sigma, sn1, tau, temp, z;
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  safmn2 = static_cast<R>(
      std::pow(Lamch<R>('B'), static_cast<int>(std::log(safmin / eps) /
                                               std::log(Lamch<R>('B')) / two)));
  safmx2 = one / safmn2;
  if (*c == zero) {
    *cs = one;
    *sn = zero;
  } else if (*b == zero) {
    // Swap rows and columns
    *cs = zero;
    *sn = one;
    temp = *d;
    *d = *a;
    *a = temp;
    *b = -*c;
    *c = zero;
  } else if ((*a - *d) == zero &&
             std::copysign(one, *b) != std::copysign(one, *c)) {
    *cs = one;
    *sn = zero;
  } else {
    temp = *a - *d;
    p = half * temp;
    bcmax = Max(std::abs(*b), std::abs(*c));
    bcmis = Min(std::abs(*b), std::abs(*c)) * std::copysign(one, *b) *
            std::copysign(one, *c);
    scale = Max(std::abs(p), bcmax);
    z = (p / scale) * p + (bcmax / scale) * bcmis;
    // If Z is of the order of the machine accuracy, postpone the
    // decision on the nature of eigenvalues
    if (z >= multpl * eps) {
      // Real eigenvalues. Compute A and D.
      z = p + std::copysign(std::sqrt(scale) * std::sqrt(z), p);
      *a = *d + z;
      *d = *d - (bcmax / z) * bcmis;
      // Compute B and the rotation matrix
      tau = Lapy2(*c, z);
      *cs = z / tau;
      *sn = *c / tau;
      *b = *b - *c;
      *c = zero;
    } else {
      // Complex eigenvalues, or real (almost) equal eigenvalues.
      // Make diagonal elements equal.
      count = 0;
      sigma = *b + *c;
    L10:
      count = count + 1;
      scale = Max(std::abs(temp), std::abs(sigma));
      if (scale >= safmx2) {
        sigma = sigma * safmn2;
        temp = temp * safmn2;
        if (count <= 20) goto L10;
      }
      if (scale <= safmn2) {
        sigma = sigma * safmx2;
        temp = temp * safmx2;
        if (count <= 20) goto L10;
      }
      p = half * temp;
      tau = Lapy2(sigma, temp);
      *cs = std::sqrt(half * (one + std::abs(sigma) / tau));
      *sn = -(p / (tau * *cs)) * std::copysign(one, sigma);
      // Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
      // [ CC  DD ]   [ C  D ] [ SN  CS ]
      aa = *a * *cs + *b * *sn;
      bb = -*a * *sn + *b * *cs;
      cc = *c * *cs + *d * *sn;
      dd = -*c * *sn + *d * *cs;
      // Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
      // [ C  D ]   [-SN  CS ] [ CC  DD ]
      // Note: Some of the multiplications are wrapped in parentheses to
      // prevent compilers from using FMA instructions. See
      // https://github.com/Reference-LAPACK/lapack/issues/1031.
      *a = aa * *cs + cc * *sn;
      *b = (bb * *cs) + (dd * *sn);
      *c = -(aa * *sn) + (cc * *cs);
      *d = -bb * *sn + dd * *cs;
      temp = half * (*a + *d);
      *a = temp;
      *d = temp;
      if (*c != zero) {
        if (*b != zero) {
          if (std::copysign(one, *b) == std::copysign(one, *c)) {
            // Real eigenvalues: reduce to upper triangular form
            sab = std::sqrt(std::abs(*b));
            sac = std::sqrt(std::abs(*c));
            p = std::copysign(sab * sac, *c);
            tau = one / std::sqrt(std::abs(*b + *c));
            *a = temp + p;
            *d = temp - p;
            *b = *b - *c;
            *c = zero;
            cs1 = sab * tau;
            sn1 = sac * tau;
            temp = *cs * cs1 - *sn * sn1;
            *sn = *cs * sn1 + *sn * cs1;
            *cs = temp;
          }
        } else {
          *b = -*c;
          *c = zero;
          temp = *cs;
          *cs = -*sn;
          *sn = temp;
        }
      }
    }
  }
  // Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I).
  *rt1r = *a;
  *rt2r = *d;
  if (*c == zero) {
    *rt1i = zero;
    *rt2i = zero;
  } else {
    *rt1i = std::sqrt(std::abs(*b)) * std::sqrt(std::abs(*c));
    *rt2i = -*rt1i;
  }
}

// xLAQR1 (real T): a multiple of the first column of
// (H - (SR1 + i*SI1)*I) * (H - (SR2 + i*SI2)*I) for the N-by-N (N = 2 or 3)
// Hessenberg matrix H, in V; SR1 + i*SI1 and SR2 + i*SI2 are both real or
// complex conjugates.
template <typename R>
void Laqr1(blasint n, const R* h, blasint ldh, R sr1, R si1, R sr2, R si2,
           R* v) {
  const R zero = R(0);
  const Matrix<const R> H(h, ldh);
  const Vector<R> V(v);
  R h21s, h31s, s;
  if (n != 2 && n != 3) {
    return;
  }
  if (n == 2) {
    s = std::abs(H(1, 1) - sr2) + std::abs(si2) + std::abs(H(2, 1));
    if (s == zero) {
      V(1) = zero;
      V(2) = zero;
    } else {
      h21s = H(2, 1) / s;
      V(1) = h21s * H(1, 2) + (H(1, 1) - sr1) * ((H(1, 1) - sr2) / s) -
             si1 * (si2 / s);
      V(2) = h21s * (H(1, 1) + H(2, 2) - sr1 - sr2);
    }
  } else {
    s = std::abs(H(1, 1) - sr2) + std::abs(si2) + std::abs(H(2, 1)) +
        std::abs(H(3, 1));
    if (s == zero) {
      V(1) = zero;
      V(2) = zero;
      V(3) = zero;
    } else {
      h21s = H(2, 1) / s;
      h31s = H(3, 1) / s;
      V(1) = (H(1, 1) - sr1) * ((H(1, 1) - sr2) / s) - si1 * (si2 / s) +
             H(1, 2) * h21s + H(1, 3) * h31s;
      V(2) = h21s * (H(1, 1) + H(2, 2) - sr1 - sr2) + H(2, 3) * h31s;
      V(3) = h31s * (H(1, 1) + H(3, 3) - sr1 - sr2) + h21s * H(3, 2);
    }
  }
}

// xLAQR1 (complex T): a multiple of the first column of
// (H - S1*I) * (H - S2*I) for the N-by-N (N = 2 or 3) Hessenberg matrix H,
// in V.
template <typename T>
void Laqr1(blasint n, const T* h, blasint ldh, T s1, T s2, T* v) {
  using R = RealT<T>;

  const T zero = T(0);
  const R rzero = R(0);
  const Matrix<const T> H(h, ldh);
  const Vector<T> V(v);
  R s;
  T h21s, h31s;
  if (n != 2 && n != 3) {
    return;
  }
  if (n == 2) {
    s = Cabs1(H(1, 1) - s2) + Cabs1(H(2, 1));
    if (s == rzero) {
      V(1) = zero;
      V(2) = zero;
    } else {
      h21s = H(2, 1) / s;
      V(1) = h21s * H(1, 2) + (H(1, 1) - s1) * ((H(1, 1) - s2) / s);
      V(2) = h21s * (H(1, 1) + H(2, 2) - s1 - s2);
    }
  } else {
    s = Cabs1(H(1, 1) - s2) + Cabs1(H(2, 1)) + Cabs1(H(3, 1));
    if (s == zero) {
      V(1) = zero;
      V(2) = zero;
      V(3) = zero;
    } else {
      h21s = H(2, 1) / s;
      h31s = H(3, 1) / s;
      V(1) = (H(1, 1) - s1) * ((H(1, 1) - s2) / s) + H(1, 2) * h21s +
             H(1, 3) * h31s;
      V(2) = h21s * (H(1, 1) + H(2, 2) - s1 - s2) + H(2, 3) * h31s;
      V(3) = h31s * (H(1, 1) + H(3, 3) - s1 - s2) + h21s * H(3, 2);
    }
  }
}

// xLAHQR (real T): the eigenvalues WR + i*WI of the upper Hessenberg matrix
// H in rows and columns ILO:IHI, by the double-shift QR algorithm, and the
// Schur form T in H (WANTT) and the Schur vectors applied to rows
// ILOZ:IHIZ of Z (WANTZ).
//
// Returns INFO: 0, or i > 0 if the algorithm failed to compute all the
// eigenvalues in 30*max(10, IHI-ILO+1) iterations; eigenvalues i+1:IHI
// are then in WR and WI, and rows and columns ILO:i of H hold the
// unconverged part.
template <typename R>
blasint Lahqr(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, blasint iloz, blasint ihiz, R* z,
              blasint ldz) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R dat1 = R(3) / R(4);
  const R dat2 = -R(0.4375);
  const blasint kexsh = 10;
  const Matrix<R> H(h, ldh);
  R v[3];
  const Vector<R> V(v);
  const Vector<R> WI(wi);
  const Vector<R> WR(wr);
  const Matrix<R> Z(z, ldz);
  blasint i, i1, i2, itmax, its, j, k, kdefl, l, m, nh, nr, nz;
  R aa, ab, ba, bb, cs, det, h11, h12, h21, h21s, h22, rt1i, rt1r, rt2i, rt2r,
      rtdisc, s, safmax, safmin, smlnum, sn, sum, t1, t2, t3, tr, tst, ulp, v2,
      v3;
  blasint info = 0;
  // Quick return if possible
  if (n == 0) return info;
  if (ilo == ihi) {
    WR(ilo) = H(ilo, ilo);
    WI(ilo) = zero;
    return info;
  }
  // ==== clear out the trash ====
  for (j = ilo; j <= ihi - 3; ++j) {
    H(j + 2, j) = zero;
    H(j + 3, j) = zero;
  }
  if (ilo <= ihi - 2) H(ihi, ihi - 2) = zero;
  nh = ihi - ilo + 1;
  nz = ihiz - iloz + 1;
  // Set machine-dependent constants for the stopping criterion.
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(nh) / ulp);
  // I1 and I2 are the indices of the first row and last column of H
  // to which transformations must be applied. If eigenvalues only are
  // being computed, I1 and I2 are set inside the main loop.
  if (wantt) {
    i1 = 1;
    i2 = n;
  }
  // ITMAX is the total number of QR iterations allowed.
  itmax = 30 * Max<blasint>(10, nh);
  // KDEFL counts the number of iterations since a deflation
  kdefl = 0;
  // The main loop begins here. I is the loop index and decreases from
  // IHI to ILO in steps of 1 or 2. Each iteration of the loop works
  // with the active submatrix in rows and columns L to I.
  // Eigenvalues I+1 to IHI have already converged. Either L = ILO or
  // H(L,L-1) is negligible so that the matrix splits.
  i = ihi;
L20:
  l = ilo;
  if (i < ilo) goto L160;
  // Perform QR iterations on rows and columns ILO to I until a
  // submatrix of order 1 or 2 splits off at the bottom because a
  // subdiagonal element has become negligible.
  for (its = 0; its <= itmax; ++its) {
    // Look for a single small subdiagonal element.
    for (k = i; k >= l + 1; --k) {
      if (std::abs(H(k, k - 1)) <= smlnum) goto L40;
      tst = std::abs(H(k - 1, k - 1)) + std::abs(H(k, k));
      if (tst == zero) {
        if (k - 2 >= ilo) tst = tst + std::abs(H(k - 1, k - 2));
        if (k + 1 <= ihi) tst = tst + std::abs(H(k + 1, k));
      }
      // ==== The following is a conservative small subdiagonal
      // .    deflation  criterion due to Ahues & Tisseur (LAWN 122,
      // .    1997). It has better mathematical foundation and
      // .    improves accuracy in some cases.  ====
      if (std::abs(H(k, k - 1)) <= ulp * tst) {
        ab = Max(std::abs(H(k, k - 1)), std::abs(H(k - 1, k)));
        ba = Min(std::abs(H(k, k - 1)), std::abs(H(k - 1, k)));
        aa = Max(std::abs(H(k, k)), std::abs(H(k - 1, k - 1) - H(k, k)));
        bb = Min(std::abs(H(k, k)), std::abs(H(k - 1, k - 1) - H(k, k)));
        s = aa + ab;
        if (ba * (ab / s) <= Max(smlnum, ulp * (bb * (aa / s)))) goto L40;
      }
    }
  L40:
    l = k;
    if (l > ilo) {
      // H(L,L-1) is negligible
      H(l, l - 1) = zero;
    }
    // Exit from loop if a submatrix of order 1 or 2 has split off.
    if (l >= i - 1) goto L150;
    kdefl = kdefl + 1;
    // Now the active submatrix is in rows and columns L to I. If
    // eigenvalues only are being computed, only the active submatrix
    // need be transformed.
    if (!wantt) {
      i1 = l;
      i2 = i;
    }
    if (kdefl % (2 * kexsh) == 0) {
      // Exceptional shift.
      s = std::abs(H(i, i - 1)) + std::abs(H(i - 1, i - 2));
      h11 = dat1 * s + H(i, i);
      h12 = dat2 * s;
      h21 = s;
      h22 = h11;
    } else if (kdefl % kexsh == 0) {
      // Exceptional shift.
      s = std::abs(H(l + 1, l)) + std::abs(H(l + 2, l + 1));
      h11 = dat1 * s + H(l, l);
      h12 = dat2 * s;
      h21 = s;
      h22 = h11;
    } else {
      // Prepare to use Francis' double shift
      // (i.e. 2nd degree generalized Rayleigh quotient)
      h11 = H(i - 1, i - 1);
      h21 = H(i, i - 1);
      h12 = H(i - 1, i);
      h22 = H(i, i);
    }
    s = std::abs(h11) + std::abs(h12) + std::abs(h21) + std::abs(h22);
    if (s == zero) {
      rt1r = zero;
      rt1i = zero;
      rt2r = zero;
      rt2i = zero;
    } else {
      h11 = h11 / s;
      h21 = h21 / s;
      h12 = h12 / s;
      h22 = h22 / s;
      tr = (h11 + h22) / two;
      det = (h11 - tr) * (h22 - tr) - h12 * h21;
      rtdisc = std::sqrt(std::abs(det));
      if (det >= zero) {
        // ==== complex conjugate shifts ====
        rt1r = tr * s;
        rt2r = rt1r;
        rt1i = rtdisc * s;
        rt2i = -rt1i;
      } else {
        // ==== real shifts (use only one of them)  ====
        rt1r = tr + rtdisc;
        rt2r = tr - rtdisc;
        if (std::abs(rt1r - h22) <= std::abs(rt2r - h22)) {
          rt1r = rt1r * s;
          rt2r = rt1r;
        } else {
          rt2r = rt2r * s;
          rt1r = rt2r;
        }
        rt1i = zero;
        rt2i = zero;
      }
    }
    // Look for two consecutive small subdiagonal elements.
    for (m = i - 2; m >= l; --m) {
      // Determine the effect of starting the double-shift QR
      // iteration at row M, and see if this would make H(M,M-1)
      // negligible.  (The following uses scaling to avoid
      // overflows and most underflows.)
      h21s = H(m + 1, m);
      s = std::abs(H(m, m) - rt2r) + std::abs(rt2i) + std::abs(h21s);
      h21s = H(m + 1, m) / s;
      V(1) = h21s * H(m, m + 1) + (H(m, m) - rt1r) * ((H(m, m) - rt2r) / s) -
             rt1i * (rt2i / s);
      V(2) = h21s * (H(m, m) + H(m + 1, m + 1) - rt1r - rt2r);
      V(3) = h21s * H(m + 2, m + 1);
      s = std::abs(V(1)) + std::abs(V(2)) + std::abs(V(3));
      V(1) = V(1) / s;
      V(2) = V(2) / s;
      V(3) = V(3) / s;
      if (m == l) goto L60;
      if (std::abs(H(m, m - 1)) * (std::abs(V(2)) + std::abs(V(3))) <=
          ulp * std::abs(V(1)) *
              (std::abs(H(m - 1, m - 1)) + std::abs(H(m, m)) +
               std::abs(H(m + 1, m + 1))))
        goto L60;
    }
  L60:
    // Double-shift QR step
    for (k = m; k <= i - 1; ++k) {
      // The first iteration of this loop determines a reflection G
      // from the vector V and applies it from left and right to H,
      // thus creating a nonzero bulge below the subdiagonal.
      // Each subsequent iteration determines a reflection G to
      // restore the Hessenberg form in the (K-1)th column, and thus
      // chases the bulge one step toward the bottom of the active
      // submatrix. NR is the order of G.
      nr = Min<blasint>(3, i - k + 1);
      if (k > m) Copy(nr, H.ptr(k, k - 1), 1, v, 1);
      Larfg(nr, V.ptr(1), V.ptr(2), 1, &t1);
      if (k > m) {
        H(k, k - 1) = V(1);
        H(k + 1, k - 1) = zero;
        if (k < i - 1) H(k + 2, k - 1) = zero;
      } else if (m > l) {
        // ==== Use the following instead of
        // .    H( K, K-1 ) = -H( K, K-1 ) to
        // .    avoid a bug when v(2) and v(3)
        // .    underflow. ====
        H(k, k - 1) = H(k, k - 1) * (one - t1);
      }
      v2 = V(2);
      t2 = t1 * v2;
      if (nr == 3) {
        v3 = V(3);
        t3 = t1 * v3;
        // Apply G from the left to transform the rows of the matrix
        // in columns K to I2.
        for (j = k; j <= i2; ++j) {
          sum = H(k, j) + v2 * H(k + 1, j) + v3 * H(k + 2, j);
          H(k, j) = H(k, j) - sum * t1;
          H(k + 1, j) = H(k + 1, j) - sum * t2;
          H(k + 2, j) = H(k + 2, j) - sum * t3;
        }
        // Apply G from the right to transform the columns of the
        // matrix in rows I1 to min(K+3,I).
        for (j = i1; j <= Min<blasint>(k + 3, i); ++j) {
          sum = H(j, k) + v2 * H(j, k + 1) + v3 * H(j, k + 2);
          H(j, k) = H(j, k) - sum * t1;
          H(j, k + 1) = H(j, k + 1) - sum * t2;
          H(j, k + 2) = H(j, k + 2) - sum * t3;
        }
        if (wantz) {
          // Accumulate transformations in the matrix Z
          for (j = iloz; j <= ihiz; ++j) {
            sum = Z(j, k) + v2 * Z(j, k + 1) + v3 * Z(j, k + 2);
            Z(j, k) = Z(j, k) - sum * t1;
            Z(j, k + 1) = Z(j, k + 1) - sum * t2;
            Z(j, k + 2) = Z(j, k + 2) - sum * t3;
          }
        }
      } else if (nr == 2) {
        // Apply G from the left to transform the rows of the matrix
        // in columns K to I2.
        for (j = k; j <= i2; ++j) {
          sum = H(k, j) + v2 * H(k + 1, j);
          H(k, j) = H(k, j) - sum * t1;
          H(k + 1, j) = H(k + 1, j) - sum * t2;
        }
        // Apply G from the right to transform the columns of the
        // matrix in rows I1 to min(K+3,I).
        for (j = i1; j <= i; ++j) {
          sum = H(j, k) + v2 * H(j, k + 1);
          H(j, k) = H(j, k) - sum * t1;
          H(j, k + 1) = H(j, k + 1) - sum * t2;
        }
        if (wantz) {
          // Accumulate transformations in the matrix Z
          for (j = iloz; j <= ihiz; ++j) {
            sum = Z(j, k) + v2 * Z(j, k + 1);
            Z(j, k) = Z(j, k) - sum * t1;
            Z(j, k + 1) = Z(j, k + 1) - sum * t2;
          }
        }
      }
    }
  }
  // Failure to converge in remaining number of iterations
  info = i;
  return info;
L150:
  if (l == i) {
    // H(I,I-1) is negligible: one eigenvalue has converged.
    WR(i) = H(i, i);
    WI(i) = zero;
  } else if (l == i - 1) {
    // H(I-1,I-2) is negligible: a pair of eigenvalues have converged.
    // Transform the 2-by-2 submatrix to standard Schur form,
    // and compute and store the eigenvalues.
    Lanv2(H.ptr(i - 1, i - 1), H.ptr(i - 1, i), H.ptr(i, i - 1), H.ptr(i, i),
          WR.ptr(i - 1), WI.ptr(i - 1), WR.ptr(i), WI.ptr(i), &cs, &sn);
    if (wantt) {
      // Apply the transformation to the rest of H.
      if (i2 > i)
        Rot(i2 - i, H.ptr(i - 1, i + 1), ldh, H.ptr(i, i + 1), ldh, cs, sn);
      Rot(i - i1 - 1, H.ptr(i1, i - 1), 1, H.ptr(i1, i), 1, cs, sn);
    }
    if (wantz) {
      // Apply the transformation to Z.
      Rot(nz, Z.ptr(iloz, i - 1), 1, Z.ptr(iloz, i), 1, cs, sn);
    }
  }
  // reset deflation counter
  kdefl = 0;
  // return to start of the main loop with new value of I.
  i = l - 1;
  goto L20;
L160:
  return info;
}

// xLAHQR (complex T): as the real routine, with the eigenvalues in W.
template <typename T>
blasint Lahqr(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, blasint iloz, blasint ihiz, T* z,
              blasint ldz) {
  using R = RealT<T>;

  const T zero = T(0);
  const T one = T(1);
  const R rzero = R(0);
  const R rone = R(1);
  const R half = R(0.5);
  const R dat1 = R(3) / R(4);
  const blasint kexsh = 10;
  const Matrix<T> H(h, ldh);
  T v[2];
  const Vector<T> V(v);
  const Vector<T> W(w);
  const Matrix<T> Z(z, ldz);
  blasint i, i1, i2, itmax, its, j, jhi, jlo, k, kdefl, l, m, nh, nz;
  R aa, ab, ba, bb, h10, h21, rtemp, s, safmax, safmin, smlnum, sx, t2, tst,
      ulp;
  T h11, h11s, h22, sc, sum, t, t1, temp, u, v2, x, y;
  blasint info = 0;
  // Quick return if possible
  if (n == 0) return info;
  if (ilo == ihi) {
    W(ilo) = H(ilo, ilo);
    return info;
  }
  // ==== clear out the trash ====
  for (j = ilo; j <= ihi - 3; ++j) {
    H(j + 2, j) = zero;
    H(j + 3, j) = zero;
  }
  if (ilo <= ihi - 2) H(ihi, ihi - 2) = zero;
  // ==== ensure that subdiagonal entries are real ====
  if (wantt) {
    jlo = 1;
    jhi = n;
  } else {
    jlo = ilo;
    jhi = ihi;
  }
  for (i = ilo + 1; i <= ihi; ++i) {
    if (std::imag(H(i, i - 1)) != rzero) {
      // ==== The following redundant normalization
      // .    avoids problems with both gradual and
      // .    sudden underflow in ABS(H(I,I-1)) ====
      sc = H(i, i - 1) / Cabs1(H(i, i - 1));
      sc = Conj(sc) / std::abs(sc);
      H(i, i - 1) = std::abs(H(i, i - 1));
      Scal(jhi - i + 1, sc, H.ptr(i, i), ldh);
      Scal(Min<blasint>(jhi, i + 1) - jlo + 1, Conj(sc), H.ptr(jlo, i), 1);
      if (wantz) Scal(ihiz - iloz + 1, Conj(sc), Z.ptr(iloz, i), 1);
    }
  }
  nh = ihi - ilo + 1;
  nz = ihiz - iloz + 1;
  // Set machine-dependent constants for the stopping criterion.
  safmin = Lamch<R>('S');
  safmax = rone / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(nh) / ulp);
  // I1 and I2 are the indices of the first row and last column of H
  // to which transformations must be applied. If eigenvalues only are
  // being computed, I1 and I2 are set inside the main loop.
  if (wantt) {
    i1 = 1;
    i2 = n;
  }
  // ITMAX is the total number of QR iterations allowed.
  itmax = 30 * Max<blasint>(10, nh);
  // KDEFL counts the number of iterations since a deflation
  kdefl = 0;
  // The main loop begins here. I is the loop index and decreases from
  // IHI to ILO in steps of 1. Each iteration of the loop works
  // with the active submatrix in rows and columns L to I.
  // Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
  // H(L,L-1) is negligible so that the matrix splits.
  i = ihi;
L30:
  if (i < ilo) goto L150;
  // Perform QR iterations on rows and columns ILO to I until a
  // submatrix of order 1 splits off at the bottom because a
  // subdiagonal element has become negligible.
  l = ilo;
  for (its = 0; its <= itmax; ++its) {
    // Look for a single small subdiagonal element.
    for (k = i; k >= l + 1; --k) {
      if (Cabs1(H(k, k - 1)) <= smlnum) goto L50;
      tst = Cabs1(H(k - 1, k - 1)) + Cabs1(H(k, k));
      if (tst == zero) {
        if (k - 2 >= ilo) tst = tst + std::abs(std::real(H(k - 1, k - 2)));
        if (k + 1 <= ihi) tst = tst + std::abs(std::real(H(k + 1, k)));
      }
      // ==== The following is a conservative small subdiagonal
      // .    deflation criterion due to Ahues & Tisseur (LAWN 122,
      // .    1997). It has better mathematical foundation and
      // .    improves accuracy in some examples.  ====
      if (std::abs(std::real(H(k, k - 1))) <= ulp * tst) {
        ab = Max(Cabs1(H(k, k - 1)), Cabs1(H(k - 1, k)));
        ba = Min(Cabs1(H(k, k - 1)), Cabs1(H(k - 1, k)));
        aa = Max(Cabs1(H(k, k)), Cabs1(H(k - 1, k - 1) - H(k, k)));
        bb = Min(Cabs1(H(k, k)), Cabs1(H(k - 1, k - 1) - H(k, k)));
        s = aa + ab;
        if (ba * (ab / s) <= Max(smlnum, ulp * (bb * (aa / s)))) goto L50;
      }
    }
  L50:
    l = k;
    if (l > ilo) {
      // H(L,L-1) is negligible
      H(l, l - 1) = zero;
    }
    // Exit from loop if a submatrix of order 1 has split off.
    if (l >= i) goto L140;
    kdefl = kdefl + 1;
    // Now the active submatrix is in rows and columns L to I. If
    // eigenvalues only are being computed, only the active submatrix
    // need be transformed.
    if (!wantt) {
      i1 = l;
      i2 = i;
    }
    if (kdefl % (2 * kexsh) == 0) {
      // Exceptional shift.
      s = dat1 * std::abs(std::real(H(i, i - 1)));
      t = s + H(i, i);
    } else if (kdefl % kexsh == 0) {
      // Exceptional shift.
      s = dat1 * std::abs(std::real(H(l + 1, l)));
      t = s + H(l, l);
    } else {
      // Wilkinson's shift.
      t = H(i, i);
      u = std::sqrt(H(i - 1, i)) * std::sqrt(H(i, i - 1));
      s = Cabs1(u);
      if (s != rzero) {
        x = half * (H(i - 1, i - 1) - t);
        sx = Cabs1(x);
        s = Max(s, Cabs1(x));
        y = s * std::sqrt((x / s) * (x / s) + (u / s) * (u / s));
        if (sx > rzero) {
          if (std::real(x / sx) * std::real(y) +
                  std::imag(x / sx) * std::imag(y) <
              rzero)
            y = -y;
        }
        t = t - u * Ladiv(u, (x + y));
      }
    }
    // Look for two consecutive small subdiagonal elements.
    for (m = i - 1; m >= l + 1; --m) {
      // Determine the effect of starting the single-shift QR
      // iteration at row M, and see if this would make H(M,M-1)
      // negligible.
      h11 = H(m, m);
      h22 = H(m + 1, m + 1);
      h11s = h11 - t;
      h21 = std::real(H(m + 1, m));
      s = Cabs1(h11s) + std::abs(h21);
      h11s = h11s / s;
      h21 = h21 / s;
      V(1) = h11s;
      V(2) = h21;
      h10 = std::real(H(m, m - 1));
      if (std::abs(h10) * std::abs(h21) <=
          ulp * (Cabs1(h11s) * (Cabs1(h11) + Cabs1(h22))))
        goto L70;
    }
    h11 = H(l, l);
    h22 = H(l + 1, l + 1);
    h11s = h11 - t;
    h21 = std::real(H(l + 1, l));
    s = Cabs1(h11s) + std::abs(h21);
    h11s = h11s / s;
    h21 = h21 / s;
    V(1) = h11s;
    V(2) = h21;
  L70:
    // Single-shift QR step
    for (k = m; k <= i - 1; ++k) {
      // The first iteration of this loop determines a reflection G
      // from the vector V and applies it from left and right to H,
      // thus creating a nonzero bulge below the subdiagonal.
      // Each subsequent iteration determines a reflection G to
      // restore the Hessenberg form in the (K-1)th column, and thus
      // chases the bulge one step toward the bottom of the active
      // submatrix.
      // V(2) is always real before the call to ZLARFG, and hence
      // after the call T2 ( = T1*V(2) ) is also real.
      if (k > m) Copy(2, H.ptr(k, k - 1), 1, v, 1);
      Larfg(2, V.ptr(1), V.ptr(2), 1, &t1);
      if (k > m) {
        H(k, k - 1) = V(1);
        H(k + 1, k - 1) = zero;
      }
      v2 = V(2);
      t2 = std::real(t1 * v2);
      // Apply G from the left to transform the rows of the matrix
      // in columns K to I2.
      for (j = k; j <= i2; ++j) {
        sum = Conj(t1) * H(k, j) + t2 * H(k + 1, j);
        H(k, j) = H(k, j) - sum;
        H(k + 1, j) = H(k + 1, j) - sum * v2;
      }
      // Apply G from the right to transform the columns of the
      // matrix in rows I1 to min(K+2,I).
      for (j = i1; j <= Min<blasint>(k + 2, i); ++j) {
        sum = t1 * H(j, k) + t2 * H(j, k + 1);
        H(j, k) = H(j, k) - sum;
        H(j, k + 1) = H(j, k + 1) - sum * Conj(v2);
      }
      if (wantz) {
        // Accumulate transformations in the matrix Z
        for (j = iloz; j <= ihiz; ++j) {
          sum = t1 * Z(j, k) + t2 * Z(j, k + 1);
          Z(j, k) = Z(j, k) - sum;
          Z(j, k + 1) = Z(j, k + 1) - sum * Conj(v2);
        }
      }
      if (k == m && m > l) {
        // If the QR step was started at row M > L because two
        // consecutive small subdiagonals were found, then extra
        // scaling must be performed to ensure that H(M,M-1) remains
        // real.
        temp = one - t1;
        temp = temp / std::abs(temp);
        H(m + 1, m) = H(m + 1, m) * Conj(temp);
        if (m + 2 <= i) H(m + 2, m + 1) = H(m + 2, m + 1) * temp;
        for (j = m; j <= i; ++j) {
          if (j != m + 1) {
            if (i2 > j) Scal(i2 - j, temp, H.ptr(j, j + 1), ldh);
            Scal(j - i1, Conj(temp), H.ptr(i1, j), 1);
            if (wantz) {
              Scal(nz, Conj(temp), Z.ptr(iloz, j), 1);
            }
          }
        }
      }
    }
    // Ensure that H(I,I-1) is real.
    temp = H(i, i - 1);
    if (std::imag(temp) != rzero) {
      rtemp = std::abs(temp);
      H(i, i - 1) = rtemp;
      temp = temp / rtemp;
      if (i2 > i) Scal(i2 - i, Conj(temp), H.ptr(i, i + 1), ldh);
      Scal(i - i1, temp, H.ptr(i1, i), 1);
      if (wantz) {
        Scal(nz, temp, Z.ptr(iloz, i), 1);
      }
    }
  }
  // Failure to converge in remaining number of iterations
  info = i;
  return info;
L140:
  // H(I,I-1) is negligible: one eigenvalue has converged.
  W(i) = H(i, i);
  // reset deflation counter
  kdefl = 0;
  // return to start of the main loop with new value of I.
  i = l - 1;
  goto L30;
L150:
  return info;
}

// xLAQR5 (real T): one small-bulge multishift QR sweep with the NSHFTS
// shifts SR + i*SI on rows and columns KTOP:KBOT of H.
template <typename R>
void Laqr5(bool wantt, bool wantz, blasint kacc22, blasint n, blasint ktop,
           blasint kbot, blasint nshfts, R* sr, R* si, R* h, blasint ldh,
           blasint iloz, blasint ihiz, R* z, blasint ldz, R* v, blasint ldv,
           R* u, blasint ldu, blasint nv, R* wv, blasint ldwv, blasint nh,
           R* wh, blasint ldwh) {
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> H(h, ldh);
  const Vector<R> SI(si);
  const Vector<R> SR(sr);
  const Matrix<R> U(u, ldu);
  const Matrix<R> V(v, ldv);
  R vt[3];
  const Vector<R> VT(vt);
  const Matrix<R> WH(wh, ldwh);
  const Matrix<R> WV(wv, ldwv);
  const Matrix<R> Z(z, ldz);
  blasint i, i2, i4, incol, j, jbot, jcol, jlen, jrow, jtop, k, k1, kdu, kms,
      krcol, m, m22, mbot, mtop, nbmps, ndcol, ns, nu;
  bool accum, bmp22;
  R alpha, beta, h11, h12, h21, h22, refsum, safmax, safmin, scl, smlnum, swap,
      t1, t2, t3, tst1, tst2, ulp;
  if (nshfts < 2) return;
  // ==== If the active block is empty or 1-by-1, then there
  // .    is nothing to do. ====
  if (ktop >= kbot) return;
  // ==== Shuffle shifts into pairs of real shifts and pairs
  // .    of complex conjugate shifts assuming complex
  // .    conjugate shifts are already adjacent to one
  // .    another. ====
  for (i = 1; i <= nshfts - 2; i += 2) {
    if (SI(i) != -SI(i + 1)) {
      swap = SR(i);
      SR(i) = SR(i + 1);
      SR(i + 1) = SR(i + 2);
      SR(i + 2) = swap;
      swap = SI(i);
      SI(i) = SI(i + 1);
      SI(i + 1) = SI(i + 2);
      SI(i + 2) = swap;
    }
  }
  // ==== NSHFTS is supposed to be even, but if it is odd,
  // .    then simply reduce it by one.  The shuffle above
  // .    ensures that the dropped shift is real and that
  // .    the remaining shifts are paired. ====
  ns = nshfts - nshfts % 2;
  // ==== Machine constants for deflation ====
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(n) / ulp);
  // ==== Use accumulated reflections to update far-from-diagonal
  // .    entries ? ====
  accum = (kacc22 == 1) || (kacc22 == 2);
  // ==== clear trash ====
  if (ktop + 2 <= kbot) H(ktop + 2, ktop) = zero;
  // ==== NBMPS = number of 2-shift bulges in the chain ====
  nbmps = ns / 2;
  // ==== KDU = width of slab ====
  kdu = 4 * nbmps;
  // ==== Create and chase chains of NBMPS bulges ====
  for (incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps) {
    // JTOP = Index from which updates from the right start.
    if (accum) {
      jtop = Max<blasint>(ktop, incol);
    } else if (wantt) {
      jtop = 1;
    } else {
      jtop = ktop;
    }
    ndcol = incol + kdu;
    if (accum) Laset('A', kdu, kdu, zero, one, u, ldu);
    // ==== Near-the-diagonal bulge chase.  The following loop
    // .    performs the near-the-diagonal part of a small bulge
    // .    multi-shift QR sweep.  Each 4*NBMPS column diagonal
    // .    chunk extends from column INCOL to column NDCOL
    // .    (including both column INCOL and column NDCOL). The
    // .    following loop chases a 2*NBMPS+1 column long chain of
    // .    NBMPS bulges 2*NBMPS columns to the right.  (INCOL
    // .    may be less than KTOP and and NDCOL may be greater than
    // .    KBOT indicating phantom columns from which to chase
    // .    bulges before they are actually introduced or to which
    // .    to chase bulges beyond column KBOT.)  ====
    for (krcol = incol; krcol <= Min<blasint>(incol + 2 * nbmps - 1, kbot - 2);
         ++krcol) {
      // ==== Bulges number MTOP to MBOT are active double implicit
      // .    shift bulges.  There may or may not also be small
      // .    2-by-2 bulge, if there is room.  The inactive bulges
      // .    (if any) must wait until the active bulges have moved
      // .    down the diagonal to make room.  The phantom matrix
      // .    paradigm described above helps keep track.  ====
      mtop = Max<blasint>(1, (ktop - krcol) / 2 + 1);
      mbot = Min<blasint>(nbmps, (kbot - krcol - 1) / 2);
      m22 = mbot + 1;
      bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1)) == (kbot - 2);
      // ==== Generate reflections to chase the chain right
      // .    one column.  (The minimum value of K is KTOP-1.) ====
      if (bmp22) {
        // ==== Special case: 2-by-2 reflection at bottom treated
        // .    separately ====
        k = krcol + 2 * (m22 - 1);
        if (k == ktop - 1) {
          Laqr1(2, H.ptr(k + 1, k + 1), ldh, SR(2 * m22 - 1), SI(2 * m22 - 1),
                SR(2 * m22), SI(2 * m22), V.ptr(1, m22));
          beta = V(1, m22);
          Larfg(2, &beta, V.ptr(2, m22), 1, V.ptr(1, m22));
        } else {
          beta = H(k + 1, k);
          V(2, m22) = H(k + 2, k);
          Larfg(2, &beta, V.ptr(2, m22), 1, V.ptr(1, m22));
          H(k + 1, k) = beta;
          H(k + 2, k) = zero;
        }
        // ==== Perform update from right within
        // .    computational window. ====
        t1 = V(1, m22);
        t2 = t1 * V(2, m22);
        for (j = jtop; j <= Min<blasint>(kbot, k + 3); ++j) {
          refsum = H(j, k + 1) + V(2, m22) * H(j, k + 2);
          H(j, k + 1) = H(j, k + 1) - refsum * t1;
          H(j, k + 2) = H(j, k + 2) - refsum * t2;
        }
        // ==== Perform update from left within
        // .    computational window. ====
        if (accum) {
          jbot = Min<blasint>(ndcol, kbot);
        } else if (wantt) {
          jbot = n;
        } else {
          jbot = kbot;
        }
        t1 = V(1, m22);
        t2 = t1 * V(2, m22);
        for (j = k + 1; j <= jbot; ++j) {
          refsum = H(k + 1, j) + V(2, m22) * H(k + 2, j);
          H(k + 1, j) = H(k + 1, j) - refsum * t1;
          H(k + 2, j) = H(k + 2, j) - refsum * t2;
        }
        // ==== The following convergence test requires that
        // .    the tradition small-compared-to-nearby-diagonals
        // .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
        // .    criteria both be satisfied.  The latter improves
        // .    accuracy in some examples. Falling back on an
        // .    alternate convergence criterion when TST1 or TST2
        // .    is zero (as done here) is traditional but probably
        // .    unnecessary. ====
        if (k >= ktop) {
          if (H(k + 1, k) != zero) {
            tst1 = std::abs(H(k, k)) + std::abs(H(k + 1, k + 1));
            if (tst1 == zero) {
              if (k >= ktop + 1) tst1 = tst1 + std::abs(H(k, k - 1));
              if (k >= ktop + 2) tst1 = tst1 + std::abs(H(k, k - 2));
              if (k >= ktop + 3) tst1 = tst1 + std::abs(H(k, k - 3));
              if (k <= kbot - 2) tst1 = tst1 + std::abs(H(k + 2, k + 1));
              if (k <= kbot - 3) tst1 = tst1 + std::abs(H(k + 3, k + 1));
              if (k <= kbot - 4) tst1 = tst1 + std::abs(H(k + 4, k + 1));
            }
            if (std::abs(H(k + 1, k)) <= Max(smlnum, ulp * tst1)) {
              h12 = Max(std::abs(H(k + 1, k)), std::abs(H(k, k + 1)));
              h21 = Min(std::abs(H(k + 1, k)), std::abs(H(k, k + 1)));
              h11 = Max(std::abs(H(k + 1, k + 1)),
                        std::abs(H(k, k) - H(k + 1, k + 1)));
              h22 = Min(std::abs(H(k + 1, k + 1)),
                        std::abs(H(k, k) - H(k + 1, k + 1)));
              scl = h11 + h12;
              tst2 = h22 * (h11 / scl);
              if (tst2 == zero ||
                  h21 * (h12 / scl) <= Max(smlnum, ulp * tst2)) {
                H(k + 1, k) = zero;
              }
            }
          }
        }
        // ==== Accumulate orthogonal transformations. ====
        if (accum) {
          kms = k - incol;
          t1 = V(1, m22);
          t2 = t1 * V(2, m22);
          for (j = Max<blasint>(1, ktop - incol); j <= kdu; ++j) {
            refsum = U(j, kms + 1) + V(2, m22) * U(j, kms + 2);
            U(j, kms + 1) = U(j, kms + 1) - refsum * t1;
            U(j, kms + 2) = U(j, kms + 2) - refsum * t2;
          }
        } else if (wantz) {
          t1 = V(1, m22);
          t2 = t1 * V(2, m22);
          for (j = iloz; j <= ihiz; ++j) {
            refsum = Z(j, k + 1) + V(2, m22) * Z(j, k + 2);
            Z(j, k + 1) = Z(j, k + 1) - refsum * t1;
            Z(j, k + 2) = Z(j, k + 2) - refsum * t2;
          }
        }
      }
      // ==== Normal case: Chain of 3-by-3 reflections ====
      for (m = mbot; m >= mtop; --m) {
        k = krcol + 2 * (m - 1);
        if (k == ktop - 1) {
          Laqr1(3, H.ptr(ktop, ktop), ldh, SR(2 * m - 1), SI(2 * m - 1),
                SR(2 * m), SI(2 * m), V.ptr(1, m));
          alpha = V(1, m);
          Larfg(3, &alpha, V.ptr(2, m), 1, V.ptr(1, m));
        } else {
          // ==== Perform delayed transformation of row below
          // .    Mth bulge. Exploit fact that first two elements
          // .    of row are actually zero. ====
          t1 = V(1, m);
          t2 = t1 * V(2, m);
          t3 = t1 * V(3, m);
          refsum = V(3, m) * H(k + 3, k + 2);
          H(k + 3, k) = -refsum * t1;
          H(k + 3, k + 1) = -refsum * t2;
          H(k + 3, k + 2) = H(k + 3, k + 2) - refsum * t3;
          // ==== Calculate reflection to move
          // .    Mth bulge one step. ====
          beta = H(k + 1, k);
          V(2, m) = H(k + 2, k);
          V(3, m) = H(k + 3, k);
          Larfg(3, &beta, V.ptr(2, m), 1, V.ptr(1, m));
          // ==== A Bulge may collapse because of vigilant
          // .    deflation or destructive underflow.  In the
          // .    underflow case, try the two-small-subdiagonals
          // .    trick to try to reinflate the bulge.  ====
          if (H(k + 3, k) != zero || H(k + 3, k + 1) != zero ||
              H(k + 3, k + 2) == zero) {
            // ==== Typical case: not collapsed (yet). ====
            H(k + 1, k) = beta;
            H(k + 2, k) = zero;
            H(k + 3, k) = zero;
          } else {
            // ==== Atypical case: collapsed.  Attempt to
            // .    reintroduce ignoring H(K+1,K) and H(K+2,K).
            // .    If the fill resulting from the new
            // .    reflector is too large, then abandon it.
            // .    Otherwise, use the new one. ====
            Laqr1(3, H.ptr(k + 1, k + 1), ldh, SR(2 * m - 1), SI(2 * m - 1),
                  SR(2 * m), SI(2 * m), vt);
            alpha = VT(1);
            Larfg(3, &alpha, VT.ptr(2), 1, VT.ptr(1));
            t1 = VT(1);
            t2 = t1 * VT(2);
            t3 = t1 * VT(3);
            refsum = H(k + 1, k) + VT(2) * H(k + 2, k);
            if (std::abs(H(k + 2, k) - refsum * t2) + std::abs(refsum * t3) >
                ulp * (std::abs(H(k, k)) + std::abs(H(k + 1, k + 1)) +
                       std::abs(H(k + 2, k + 2)))) {
              // ==== Starting a new bulge here would
              // .    create non-negligible fill.  Use
              // .    the old one with trepidation. ====
              H(k + 1, k) = beta;
              H(k + 2, k) = zero;
              H(k + 3, k) = zero;
            } else {
              // ==== Starting a new bulge here would
              // .    create only negligible fill.
              // .    Replace the old reflector with
              // .    the new one. ====
              H(k + 1, k) = H(k + 1, k) - refsum * t1;
              H(k + 2, k) = zero;
              H(k + 3, k) = zero;
              V(1, m) = VT(1);
              V(2, m) = VT(2);
              V(3, m) = VT(3);
            }
          }
        }
        // ====  Apply reflection from the right and
        // .     the first column of update from the left.
        // .     These updates are required for the vigilant
        // .     deflation check. We still delay most of the
        // .     updates from the left for efficiency. ====
        t1 = V(1, m);
        t2 = t1 * V(2, m);
        t3 = t1 * V(3, m);
        for (j = jtop; j <= Min<blasint>(kbot, k + 3); ++j) {
          refsum = H(j, k + 1) + V(2, m) * H(j, k + 2) + V(3, m) * H(j, k + 3);
          H(j, k + 1) = H(j, k + 1) - refsum * t1;
          H(j, k + 2) = H(j, k + 2) - refsum * t2;
          H(j, k + 3) = H(j, k + 3) - refsum * t3;
        }
        // ==== Perform update from left for subsequent
        // .    column. ====
        refsum = H(k + 1, k + 1) + V(2, m) * H(k + 2, k + 1) +
                 V(3, m) * H(k + 3, k + 1);
        H(k + 1, k + 1) = H(k + 1, k + 1) - refsum * t1;
        H(k + 2, k + 1) = H(k + 2, k + 1) - refsum * t2;
        H(k + 3, k + 1) = H(k + 3, k + 1) - refsum * t3;
        // ==== The following convergence test requires that
        // .    the tradition small-compared-to-nearby-diagonals
        // .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
        // .    criteria both be satisfied.  The latter improves
        // .    accuracy in some examples. Falling back on an
        // .    alternate convergence criterion when TST1 or TST2
        // .    is zero (as done here) is traditional but probably
        // .    unnecessary. ====
        if (k < ktop) continue;
        if (H(k + 1, k) != zero) {
          tst1 = std::abs(H(k, k)) + std::abs(H(k + 1, k + 1));
          if (tst1 == zero) {
            if (k >= ktop + 1) tst1 = tst1 + std::abs(H(k, k - 1));
            if (k >= ktop + 2) tst1 = tst1 + std::abs(H(k, k - 2));
            if (k >= ktop + 3) tst1 = tst1 + std::abs(H(k, k - 3));
            if (k <= kbot - 2) tst1 = tst1 + std::abs(H(k + 2, k + 1));
            if (k <= kbot - 3) tst1 = tst1 + std::abs(H(k + 3, k + 1));
            if (k <= kbot - 4) tst1 = tst1 + std::abs(H(k + 4, k + 1));
          }
          if (std::abs(H(k + 1, k)) <= Max(smlnum, ulp * tst1)) {
            h12 = Max(std::abs(H(k + 1, k)), std::abs(H(k, k + 1)));
            h21 = Min(std::abs(H(k + 1, k)), std::abs(H(k, k + 1)));
            h11 = Max(std::abs(H(k + 1, k + 1)),
                      std::abs(H(k, k) - H(k + 1, k + 1)));
            h22 = Min(std::abs(H(k + 1, k + 1)),
                      std::abs(H(k, k) - H(k + 1, k + 1)));
            scl = h11 + h12;
            tst2 = h22 * (h11 / scl);
            if (tst2 == zero || h21 * (h12 / scl) <= Max(smlnum, ulp * tst2)) {
              H(k + 1, k) = zero;
            }
          }
        }
      }
      // ==== Multiply H by reflections from the left ====
      if (accum) {
        jbot = Min<blasint>(ndcol, kbot);
      } else if (wantt) {
        jbot = n;
      } else {
        jbot = kbot;
      }
      for (m = mbot; m >= mtop; --m) {
        k = krcol + 2 * (m - 1);
        t1 = V(1, m);
        t2 = t1 * V(2, m);
        t3 = t1 * V(3, m);
        for (j = Max<blasint>(ktop, krcol + 2 * m); j <= jbot; ++j) {
          refsum = H(k + 1, j) + V(2, m) * H(k + 2, j) + V(3, m) * H(k + 3, j);
          H(k + 1, j) = H(k + 1, j) - refsum * t1;
          H(k + 2, j) = H(k + 2, j) - refsum * t2;
          H(k + 3, j) = H(k + 3, j) - refsum * t3;
        }
      }
      // ==== Accumulate orthogonal transformations. ====
      if (accum) {
        // ==== Accumulate U. (If needed, update Z later
        // .    with an efficient matrix-matrix
        // .    multiply.) ====
        for (m = mbot; m >= mtop; --m) {
          k = krcol + 2 * (m - 1);
          kms = k - incol;
          i2 = Max<blasint>(1, ktop - incol);
          i2 = Max<blasint>(i2, kms - (krcol - incol) + 1);
          i4 = Min<blasint>(kdu, krcol + 2 * (mbot - 1) - incol + 5);
          t1 = V(1, m);
          t2 = t1 * V(2, m);
          t3 = t1 * V(3, m);
          for (j = i2; j <= i4; ++j) {
            refsum = U(j, kms + 1) + V(2, m) * U(j, kms + 2) +
                     V(3, m) * U(j, kms + 3);
            U(j, kms + 1) = U(j, kms + 1) - refsum * t1;
            U(j, kms + 2) = U(j, kms + 2) - refsum * t2;
            U(j, kms + 3) = U(j, kms + 3) - refsum * t3;
          }
        }
      } else if (wantz) {
        // ==== U is not accumulated, so update Z
        // .    now by multiplying by reflections
        // .    from the right. ====
        for (m = mbot; m >= mtop; --m) {
          k = krcol + 2 * (m - 1);
          t1 = V(1, m);
          t2 = t1 * V(2, m);
          t3 = t1 * V(3, m);
          for (j = iloz; j <= ihiz; ++j) {
            refsum =
                Z(j, k + 1) + V(2, m) * Z(j, k + 2) + V(3, m) * Z(j, k + 3);
            Z(j, k + 1) = Z(j, k + 1) - refsum * t1;
            Z(j, k + 2) = Z(j, k + 2) - refsum * t2;
            Z(j, k + 3) = Z(j, k + 3) - refsum * t3;
          }
        }
      }
      // ==== End of near-the-diagonal bulge chase. ====
    }
    // ==== Use U (if accumulated) to update far-from-diagonal
    // .    entries in H.  If required, use U to update Z as
    // .    well. ====
    if (accum) {
      if (wantt) {
        jtop = 1;
        jbot = n;
      } else {
        jtop = ktop;
        jbot = kbot;
      }
      k1 = Max<blasint>(1, ktop - incol);
      nu = (kdu - Max<blasint>(0, ndcol - kbot)) - k1 + 1;
      // ==== Horizontal Multiply ====
      for (jcol = Min<blasint>(ndcol, kbot) + 1; jcol <= jbot; jcol += nh) {
        jlen = Min<blasint>(nh, jbot - jcol + 1);
        Gemm('C', 'N', nu, jlen, nu, one, U.ptr(k1, k1), ldu,
             H.ptr(incol + k1, jcol), ldh, zero, wh, ldwh);
        Lacpy('A', nu, jlen, wh, ldwh, H.ptr(incol + k1, jcol), ldh);
      }
      // ==== Vertical multiply ====
      for (jrow = jtop; jrow <= Max<blasint>(ktop, incol) - 1; jrow += nv) {
        jlen = Min<blasint>(nv, Max<blasint>(ktop, incol) - jrow);
        Gemm('N', 'N', jlen, nu, nu, one, H.ptr(jrow, incol + k1), ldh,
             U.ptr(k1, k1), ldu, zero, wv, ldwv);
        Lacpy('A', jlen, nu, wv, ldwv, H.ptr(jrow, incol + k1), ldh);
      }
      // ==== Z multiply (also vertical) ====
      if (wantz) {
        for (jrow = iloz; jrow <= ihiz; jrow += nv) {
          jlen = Min<blasint>(nv, ihiz - jrow + 1);
          Gemm('N', 'N', jlen, nu, nu, one, Z.ptr(jrow, incol + k1), ldz,
               U.ptr(k1, k1), ldu, zero, wv, ldwv);
          Lacpy('A', jlen, nu, wv, ldwv, Z.ptr(jrow, incol + k1), ldz);
        }
      }
    }
  }
  // ==== End of DLAQR5 ====
}

// xLAQR5 (complex T): as the real routine, with the shifts S.
template <typename T>
void Laqr5(bool wantt, bool wantz, blasint kacc22, blasint n, blasint ktop,
           blasint kbot, blasint nshfts, T* s, T* h, blasint ldh, blasint iloz,
           blasint ihiz, T* z, blasint ldz, T* v, blasint ldv, T* u,
           blasint ldu, blasint nv, T* wv, blasint ldwv, blasint nh, T* wh,
           blasint ldwh) {
  using R = RealT<T>;

  const T zero = T(0);
  const T one = T(1);
  const R rzero = R(0);
  const R rone = R(1);
  const Matrix<T> H(h, ldh);
  const Vector<T> S(s);
  const Matrix<T> U(u, ldu);
  const Matrix<T> V(v, ldv);
  T vt[3];
  const Vector<T> VT(vt);
  const Matrix<T> WH(wh, ldwh);
  const Matrix<T> WV(wv, ldwv);
  const Matrix<T> Z(z, ldz);
  blasint i2, i4, incol, j, jbot, jcol, jlen, jrow, jtop, k, k1, kdu, kms,
      krcol, m, m22, mbot, mtop, nbmps, ndcol, ns, nu;
  bool accum, bmp22;
  R h11, h12, h21, h22, safmax, safmin, scl, smlnum, tst1, tst2, ulp;
  T alpha, beta, refsum, t1, t2, t3;
  if (nshfts < 2) return;
  // ==== If the active block is empty or 1-by-1, then there
  // .    is nothing to do. ====
  if (ktop >= kbot) return;
  // ==== NSHFTS is supposed to be even, but if it is odd,
  // .    then simply reduce it by one.  ====
  ns = nshfts - nshfts % 2;
  // ==== Machine constants for deflation ====
  safmin = Lamch<R>('S');
  safmax = rone / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(n) / ulp);
  // ==== Use accumulated reflections to update far-from-diagonal
  // .    entries ? ====
  accum = (kacc22 == 1) || (kacc22 == 2);
  // ==== clear trash ====
  if (ktop + 2 <= kbot) H(ktop + 2, ktop) = zero;
  // ==== NBMPS = number of 2-shift bulges in the chain ====
  nbmps = ns / 2;
  // ==== KDU = width of slab ====
  kdu = 4 * nbmps;
  // ==== Create and chase chains of NBMPS bulges ====
  for (incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps) {
    // JTOP = Index from which updates from the right start.
    if (accum) {
      jtop = Max<blasint>(ktop, incol);
    } else if (wantt) {
      jtop = 1;
    } else {
      jtop = ktop;
    }
    ndcol = incol + kdu;
    if (accum) Laset('A', kdu, kdu, zero, one, u, ldu);
    // ==== Near-the-diagonal bulge chase.  The following loop
    // .    performs the near-the-diagonal part of a small bulge
    // .    multi-shift QR sweep.  Each 4*NBMPS column diagonal
    // .    chunk extends from column INCOL to column NDCOL
    // .    (including both column INCOL and column NDCOL). The
    // .    following loop chases a 2*NBMPS+1 column long chain of
    // .    NBMPS bulges 2*NBMPS columns to the right.  (INCOL
    // .    may be less than KTOP and and NDCOL may be greater than
    // .    KBOT indicating phantom columns from which to chase
    // .    bulges before they are actually introduced or to which
    // .    to chase bulges beyond column KBOT.)  ====
    for (krcol = incol; krcol <= Min<blasint>(incol + 2 * nbmps - 1, kbot - 2);
         ++krcol) {
      // ==== Bulges number MTOP to MBOT are active double implicit
      // .    shift bulges.  There may or may not also be small
      // .    2-by-2 bulge, if there is room.  The inactive bulges
      // .    (if any) must wait until the active bulges have moved
      // .    down the diagonal to make room.  The phantom matrix
      // .    paradigm described above helps keep track.  ====
      mtop = Max<blasint>(1, (ktop - krcol) / 2 + 1);
      mbot = Min<blasint>(nbmps, (kbot - krcol - 1) / 2);
      m22 = mbot + 1;
      bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1)) == (kbot - 2);
      // ==== Generate reflections to chase the chain right
      // .    one column.  (The minimum value of K is KTOP-1.) ====
      if (bmp22) {
        // ==== Special case: 2-by-2 reflection at bottom treated
        // .    separately ====
        k = krcol + 2 * (m22 - 1);
        if (k == ktop - 1) {
          Laqr1(2, H.ptr(k + 1, k + 1), ldh, S(2 * m22 - 1), S(2 * m22),
                V.ptr(1, m22));
          beta = V(1, m22);
          Larfg(2, &beta, V.ptr(2, m22), 1, V.ptr(1, m22));
        } else {
          beta = H(k + 1, k);
          V(2, m22) = H(k + 2, k);
          Larfg(2, &beta, V.ptr(2, m22), 1, V.ptr(1, m22));
          H(k + 1, k) = beta;
          H(k + 2, k) = zero;
        }
        // ==== Perform update from right within
        // .    computational window. ====
        t1 = V(1, m22);
        t2 = t1 * Conj(V(2, m22));
        for (j = jtop; j <= Min<blasint>(kbot, k + 3); ++j) {
          refsum = H(j, k + 1) + V(2, m22) * H(j, k + 2);
          H(j, k + 1) = H(j, k + 1) - refsum * t1;
          H(j, k + 2) = H(j, k + 2) - refsum * t2;
        }
        // ==== Perform update from left within
        // .    computational window. ====
        if (accum) {
          jbot = Min<blasint>(ndcol, kbot);
        } else if (wantt) {
          jbot = n;
        } else {
          jbot = kbot;
        }
        t1 = Conj(V(1, m22));
        t2 = t1 * V(2, m22);
        for (j = k + 1; j <= jbot; ++j) {
          refsum = H(k + 1, j) + Conj(V(2, m22)) * H(k + 2, j);
          H(k + 1, j) = H(k + 1, j) - refsum * t1;
          H(k + 2, j) = H(k + 2, j) - refsum * t2;
        }
        // ==== The following convergence test requires that
        // .    the tradition small-compared-to-nearby-diagonals
        // .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
        // .    criteria both be satisfied.  The latter improves
        // .    accuracy in some examples. Falling back on an
        // .    alternate convergence criterion when TST1 or TST2
        // .    is zero (as done here) is traditional but probably
        // .    unnecessary. ====
        if (k >= ktop) {
          if (H(k + 1, k) != zero) {
            tst1 = Cabs1(H(k, k)) + Cabs1(H(k + 1, k + 1));
            if (tst1 == rzero) {
              if (k >= ktop + 1) tst1 = tst1 + Cabs1(H(k, k - 1));
              if (k >= ktop + 2) tst1 = tst1 + Cabs1(H(k, k - 2));
              if (k >= ktop + 3) tst1 = tst1 + Cabs1(H(k, k - 3));
              if (k <= kbot - 2) tst1 = tst1 + Cabs1(H(k + 2, k + 1));
              if (k <= kbot - 3) tst1 = tst1 + Cabs1(H(k + 3, k + 1));
              if (k <= kbot - 4) tst1 = tst1 + Cabs1(H(k + 4, k + 1));
            }
            if (Cabs1(H(k + 1, k)) <= Max(smlnum, ulp * tst1)) {
              h12 = Max(Cabs1(H(k + 1, k)), Cabs1(H(k, k + 1)));
              h21 = Min(Cabs1(H(k + 1, k)), Cabs1(H(k, k + 1)));
              h11 =
                  Max(Cabs1(H(k + 1, k + 1)), Cabs1(H(k, k) - H(k + 1, k + 1)));
              h22 =
                  Min(Cabs1(H(k + 1, k + 1)), Cabs1(H(k, k) - H(k + 1, k + 1)));
              scl = h11 + h12;
              tst2 = h22 * (h11 / scl);
              if (tst2 == rzero || h21 * (h12 / scl) <= Max(smlnum, ulp * tst2))
                H(k + 1, k) = zero;
            }
          }
        }
        // ==== Accumulate orthogonal transformations. ====
        if (accum) {
          kms = k - incol;
          for (j = Max<blasint>(1, ktop - incol); j <= kdu; ++j) {
            refsum = V(1, m22) * (U(j, kms + 1) + V(2, m22) * U(j, kms + 2));
            U(j, kms + 1) = U(j, kms + 1) - refsum;
            U(j, kms + 2) = U(j, kms + 2) - refsum * Conj(V(2, m22));
          }
        } else if (wantz) {
          for (j = iloz; j <= ihiz; ++j) {
            refsum = V(1, m22) * (Z(j, k + 1) + V(2, m22) * Z(j, k + 2));
            Z(j, k + 1) = Z(j, k + 1) - refsum;
            Z(j, k + 2) = Z(j, k + 2) - refsum * Conj(V(2, m22));
          }
        }
      }
      // ==== Normal case: Chain of 3-by-3 reflections ====
      for (m = mbot; m >= mtop; --m) {
        k = krcol + 2 * (m - 1);
        if (k == ktop - 1) {
          Laqr1(3, H.ptr(ktop, ktop), ldh, S(2 * m - 1), S(2 * m), V.ptr(1, m));
          alpha = V(1, m);
          Larfg(3, &alpha, V.ptr(2, m), 1, V.ptr(1, m));
        } else {
          // ==== Perform delayed transformation of row below
          // .    Mth bulge. Exploit fact that first two elements
          // .    of row are actually zero. ====
          t1 = V(1, m);
          t2 = t1 * Conj(V(2, m));
          t3 = t1 * Conj(V(3, m));
          refsum = V(3, m) * H(k + 3, k + 2);
          H(k + 3, k) = -refsum * t1;
          H(k + 3, k + 1) = -refsum * t2;
          H(k + 3, k + 2) = H(k + 3, k + 2) - refsum * t3;
          // ==== Calculate reflection to move
          // .    Mth bulge one step. ====
          beta = H(k + 1, k);
          V(2, m) = H(k + 2, k);
          V(3, m) = H(k + 3, k);
          Larfg(3, &beta, V.ptr(2, m), 1, V.ptr(1, m));
          // ==== A Bulge may collapse because of vigilant
          // .    deflation or destructive underflow.  In the
          // .    underflow case, try the two-small-subdiagonals
          // .    trick to try to reinflate the bulge.  ====
          if (H(k + 3, k) != zero || H(k + 3, k + 1) != zero ||
              H(k + 3, k + 2) == zero) {
            // ==== Typical case: not collapsed (yet). ====
            H(k + 1, k) = beta;
            H(k + 2, k) = zero;
            H(k + 3, k) = zero;
          } else {
            // ==== Atypical case: collapsed.  Attempt to
            // .    reintroduce ignoring H(K+1,K) and H(K+2,K).
            // .    If the fill resulting from the new
            // .    reflector is too large, then abandon it.
            // .    Otherwise, use the new one. ====
            Laqr1(3, H.ptr(k + 1, k + 1), ldh, S(2 * m - 1), S(2 * m), vt);
            alpha = VT(1);
            Larfg(3, &alpha, VT.ptr(2), 1, VT.ptr(1));
            t1 = Conj(VT(1));
            t2 = t1 * VT(2);
            t3 = t1 * VT(3);
            refsum = H(k + 1, k) + Conj(VT(2)) * H(k + 2, k);
            if (Cabs1(H(k + 2, k) - refsum * t2) + Cabs1(refsum * t3) >
                ulp * (Cabs1(H(k, k)) + Cabs1(H(k + 1, k + 1)) +
                       Cabs1(H(k + 2, k + 2)))) {
              // ==== Starting a new bulge here would
              // .    create non-negligible fill.  Use
              // .    the old one with trepidation. ====
              H(k + 1, k) = beta;
              H(k + 2, k) = zero;
              H(k + 3, k) = zero;
            } else {
              // ==== Starting a new bulge here would
              // .    create only negligible fill.
              // .    Replace the old reflector with
              // .    the new one. ====
              H(k + 1, k) = H(k + 1, k) - refsum * t1;
              H(k + 2, k) = zero;
              H(k + 3, k) = zero;
              V(1, m) = VT(1);
              V(2, m) = VT(2);
              V(3, m) = VT(3);
            }
          }
        }
        // ====  Apply reflection from the right and
        // .     the first column of update from the left.
        // .     These updates are required for the vigilant
        // .     deflation check. We still delay most of the
        // .     updates from the left for efficiency. ====
        t1 = V(1, m);
        t2 = t1 * Conj(V(2, m));
        t3 = t1 * Conj(V(3, m));
        for (j = jtop; j <= Min<blasint>(kbot, k + 3); ++j) {
          refsum = H(j, k + 1) + V(2, m) * H(j, k + 2) + V(3, m) * H(j, k + 3);
          H(j, k + 1) = H(j, k + 1) - refsum * t1;
          H(j, k + 2) = H(j, k + 2) - refsum * t2;
          H(j, k + 3) = H(j, k + 3) - refsum * t3;
        }
        // ==== Perform update from left for subsequent
        // .    column. ====
        t1 = Conj(V(1, m));
        t2 = t1 * V(2, m);
        t3 = t1 * V(3, m);
        refsum = H(k + 1, k + 1) + Conj(V(2, m)) * H(k + 2, k + 1) +
                 Conj(V(3, m)) * H(k + 3, k + 1);
        H(k + 1, k + 1) = H(k + 1, k + 1) - refsum * t1;
        H(k + 2, k + 1) = H(k + 2, k + 1) - refsum * t2;
        H(k + 3, k + 1) = H(k + 3, k + 1) - refsum * t3;
        // ==== The following convergence test requires that
        // .    the tradition small-compared-to-nearby-diagonals
        // .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
        // .    criteria both be satisfied.  The latter improves
        // .    accuracy in some examples. Falling back on an
        // .    alternate convergence criterion when TST1 or TST2
        // .    is zero (as done here) is traditional but probably
        // .    unnecessary. ====
        if (k < ktop) continue;
        if (H(k + 1, k) != zero) {
          tst1 = Cabs1(H(k, k)) + Cabs1(H(k + 1, k + 1));
          if (tst1 == rzero) {
            if (k >= ktop + 1) tst1 = tst1 + Cabs1(H(k, k - 1));
            if (k >= ktop + 2) tst1 = tst1 + Cabs1(H(k, k - 2));
            if (k >= ktop + 3) tst1 = tst1 + Cabs1(H(k, k - 3));
            if (k <= kbot - 2) tst1 = tst1 + Cabs1(H(k + 2, k + 1));
            if (k <= kbot - 3) tst1 = tst1 + Cabs1(H(k + 3, k + 1));
            if (k <= kbot - 4) tst1 = tst1 + Cabs1(H(k + 4, k + 1));
          }
          if (Cabs1(H(k + 1, k)) <= Max(smlnum, ulp * tst1)) {
            h12 = Max(Cabs1(H(k + 1, k)), Cabs1(H(k, k + 1)));
            h21 = Min(Cabs1(H(k + 1, k)), Cabs1(H(k, k + 1)));
            h11 = Max(Cabs1(H(k + 1, k + 1)), Cabs1(H(k, k) - H(k + 1, k + 1)));
            h22 = Min(Cabs1(H(k + 1, k + 1)), Cabs1(H(k, k) - H(k + 1, k + 1)));
            scl = h11 + h12;
            tst2 = h22 * (h11 / scl);
            if (tst2 == rzero || h21 * (h12 / scl) <= Max(smlnum, ulp * tst2))
              H(k + 1, k) = zero;
          }
        }
      }
      // ==== Multiply H by reflections from the left ====
      if (accum) {
        jbot = Min<blasint>(ndcol, kbot);
      } else if (wantt) {
        jbot = n;
      } else {
        jbot = kbot;
      }
      for (m = mbot; m >= mtop; --m) {
        k = krcol + 2 * (m - 1);
        t1 = Conj(V(1, m));
        t2 = t1 * V(2, m);
        t3 = t1 * V(3, m);
        for (j = Max<blasint>(ktop, krcol + 2 * m); j <= jbot; ++j) {
          refsum = H(k + 1, j) + Conj(V(2, m)) * H(k + 2, j) +
                   Conj(V(3, m)) * H(k + 3, j);
          H(k + 1, j) = H(k + 1, j) - refsum * t1;
          H(k + 2, j) = H(k + 2, j) - refsum * t2;
          H(k + 3, j) = H(k + 3, j) - refsum * t3;
        }
      }
      // ==== Accumulate orthogonal transformations. ====
      if (accum) {
        // ==== Accumulate U. (If needed, update Z later
        // .    with an efficient matrix-matrix
        // .    multiply.) ====
        for (m = mbot; m >= mtop; --m) {
          k = krcol + 2 * (m - 1);
          kms = k - incol;
          i2 = Max<blasint>(1, ktop - incol);
          i2 = Max<blasint>(i2, kms - (krcol - incol) + 1);
          i4 = Min<blasint>(kdu, krcol + 2 * (mbot - 1) - incol + 5);
          t1 = V(1, m);
          t2 = t1 * Conj(V(2, m));
          t3 = t1 * Conj(V(3, m));
          for (j = i2; j <= i4; ++j) {
            refsum = U(j, kms + 1) + V(2, m) * U(j, kms + 2) +
                     V(3, m) * U(j, kms + 3);
            U(j, kms + 1) = U(j, kms + 1) - refsum * t1;
            U(j, kms + 2) = U(j, kms + 2) - refsum * t2;
            U(j, kms + 3) = U(j, kms + 3) - refsum * t3;
          }
        }
      } else if (wantz) {
        // ==== U is not accumulated, so update Z
        // .    now by multiplying by reflections
        // .    from the right. ====
        for (m = mbot; m >= mtop; --m) {
          k = krcol + 2 * (m - 1);
          t1 = V(1, m);
          t2 = t1 * Conj(V(2, m));
          t3 = t1 * Conj(V(3, m));
          for (j = iloz; j <= ihiz; ++j) {
            refsum =
                Z(j, k + 1) + V(2, m) * Z(j, k + 2) + V(3, m) * Z(j, k + 3);
            Z(j, k + 1) = Z(j, k + 1) - refsum * t1;
            Z(j, k + 2) = Z(j, k + 2) - refsum * t2;
            Z(j, k + 3) = Z(j, k + 3) - refsum * t3;
          }
        }
      }
      // ==== End of near-the-diagonal bulge chase. ====
    }
    // ==== Use U (if accumulated) to update far-from-diagonal
    // .    entries in H.  If required, use U to update Z as
    // .    well. ====
    if (accum) {
      if (wantt) {
        jtop = 1;
        jbot = n;
      } else {
        jtop = ktop;
        jbot = kbot;
      }
      k1 = Max<blasint>(1, ktop - incol);
      nu = (kdu - Max<blasint>(0, ndcol - kbot)) - k1 + 1;
      // ==== Horizontal Multiply ====
      for (jcol = Min<blasint>(ndcol, kbot) + 1; jcol <= jbot; jcol += nh) {
        jlen = Min<blasint>(nh, jbot - jcol + 1);
        Gemm('C', 'N', nu, jlen, nu, one, U.ptr(k1, k1), ldu,
             H.ptr(incol + k1, jcol), ldh, zero, wh, ldwh);
        Lacpy('A', nu, jlen, wh, ldwh, H.ptr(incol + k1, jcol), ldh);
      }
      // ==== Vertical multiply ====
      for (jrow = jtop; jrow <= Max<blasint>(ktop, incol) - 1; jrow += nv) {
        jlen = Min<blasint>(nv, Max<blasint>(ktop, incol) - jrow);
        Gemm('N', 'N', jlen, nu, nu, one, H.ptr(jrow, incol + k1), ldh,
             U.ptr(k1, k1), ldu, zero, wv, ldwv);
        Lacpy('A', jlen, nu, wv, ldwv, H.ptr(jrow, incol + k1), ldh);
      }
      // ==== Z multiply (also vertical) ====
      if (wantz) {
        for (jrow = iloz; jrow <= ihiz; jrow += nv) {
          jlen = Min<blasint>(nv, ihiz - jrow + 1);
          Gemm('N', 'N', jlen, nu, nu, one, Z.ptr(jrow, incol + k1), ldz,
               U.ptr(k1, k1), ldu, zero, wv, ldwv);
          Lacpy('A', jlen, nu, wv, ldwv, Z.ptr(jrow, incol + k1), ldz);
        }
      }
    }
  }
  // ==== End of ZLAQR5 ====
}

// xLAQR2 (real T): aggressive early deflation on the trailing NW-by-NW
// window of the active block KTOP:KBOT of H (as xLAQR3, but xLAHQR for the
// window's Schur form). NS returns the number of unconverged eigenvalues
// (the shifts for the next sweep, in SR + i*SI), ND the number of
// deflated ones.
template <typename R>
void Laqr2(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, R* h, blasint ldh, blasint iloz, blasint ihiz, R* z,
           blasint ldz, blasint* ns, blasint* nd, R* sr, R* si, R* v,
           blasint ldv, blasint nh, R* t, blasint ldt, blasint nv, R* wv,
           blasint ldwv, R* work, blasint lwork) {
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> H(h, ldh);
  const Vector<R> SI(si);
  const Vector<R> SR(sr);
  const Matrix<R> Tm(t, ldt);
  const Matrix<R> V(v, ldv);
  const Vector<R> WORK(work);
  const Matrix<R> WV(wv, ldwv);
  const Matrix<R> Z(z, ldz);
  blasint i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, kln, krow, kwtop,
      ltop, lwk1, lwk2, lwkopt;
  bool bulge, sorted;
  R aa, bb, beta, cc, cs, dd, evi, evk, foo, s, safmax, safmin, smlnum, sn, tau,
      ulp;
  jw = Min<blasint>(nw, kbot - ktop + 1);
  if (jw <= 2) {
    lwkopt = 1;
  } else {
    // ==== Workspace query call to DGEHRD ====
    info = Gehrd(jw, 1, jw - 1, t, ldt, work, work, -1);
    lwk1 = static_cast<blasint>(WORK(1));
    // ==== Workspace query call to DORMHR ====
    info = Unmhr('R', 'N', jw, jw, 1, jw - 1, t, ldt, work, v, ldv, work, -1);
    lwk2 = static_cast<blasint>(WORK(1));
    // ==== Optimal workspace ====
    lwkopt = jw + Max<blasint>(lwk1, lwk2);
  }
  // ==== Quick return in case of workspace query. ====
  if (lwork == -1) {
    WORK(1) = WorkSize<R>(lwkopt);
    return;
  }
  // ==== Nothing to do ...
  *ns = 0;
  *nd = 0;
  WORK(1) = one;
  if (ktop > kbot) return;
  if (nw < 1) return;
  // ==== Machine constants ====
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(n) / ulp);
  // ==== Setup deflation window ====
  jw = Min<blasint>(nw, kbot - ktop + 1);
  kwtop = kbot - jw + 1;
  if (kwtop == ktop) {
    s = zero;
  } else {
    s = H(kwtop, kwtop - 1);
  }
  if (kbot == kwtop) {
    // ==== 1-by-1 deflation window: not much to do ====
    SR(kwtop) = H(kwtop, kwtop);
    SI(kwtop) = zero;
    *ns = 1;
    *nd = 0;
    if (std::abs(s) <= Max(smlnum, ulp * std::abs(H(kwtop, kwtop)))) {
      *ns = 0;
      *nd = 1;
      if (kwtop > ktop) H(kwtop, kwtop - 1) = zero;
    }
    WORK(1) = one;
    return;
  }
  // ==== Convert to spike-triangular form.  (In case of a
  // .    rare QR failure, this routine continues to do
  // .    aggressive early deflation using that part of
  // .    the deflation window that converged using INFQR
  // .    here and there to keep track.) ====
  Lacpy('U', jw, jw, H.ptr(kwtop, kwtop), ldh, t, ldt);
  Copy(jw - 1, H.ptr(kwtop + 1, kwtop), ldh + 1, Tm.ptr(2, 1), ldt + 1);
  Laset('A', jw, jw, zero, one, v, ldv);
  infqr = Lahqr(true, true, jw, 1, jw, t, ldt, SR.ptr(kwtop), SI.ptr(kwtop), 1,
                jw, v, ldv);
  // ==== DTREXC needs a clean margin near the diagonal ====
  for (j = 1; j <= jw - 3; ++j) {
    Tm(j + 2, j) = zero;
    Tm(j + 3, j) = zero;
  }
  if (jw > 2) Tm(jw, jw - 2) = zero;
  // ==== Deflation detection loop ====
  *ns = jw;
  ilst = infqr + 1;
L20:
  if (ilst <= *ns) {
    if (*ns == 1) {
      bulge = false;
    } else {
      bulge = Tm(*ns, *ns - 1) != zero;
    }
    // ==== Small spike tip test for deflation ====
    if (!bulge) {
      // ==== Real eigenvalue ====
      foo = std::abs(Tm(*ns, *ns));
      if (foo == zero) foo = std::abs(s);
      if (std::abs(s * V(1, *ns)) <= Max(smlnum, ulp * foo)) {
        // ==== Deflatable ====
        *ns = *ns - 1;
      } else {
        // ==== Undeflatable.   Move it up out of the way.
        // .    (DTREXC can not fail in this case.) ====
        ifst = *ns;
        info = Trexc('V', jw, t, ldt, v, ldv, &ifst, &ilst, work);
        ilst = ilst + 1;
      }
    } else {
      // ==== Complex conjugate pair ====
      foo = std::abs(Tm(*ns, *ns)) + std::sqrt(std::abs(Tm(*ns, *ns - 1))) *
                                         std::sqrt(std::abs(Tm(*ns - 1, *ns)));
      if (foo == zero) foo = std::abs(s);
      if (Max(std::abs(s * V(1, *ns)), std::abs(s * V(1, *ns - 1))) <=
          Max(smlnum, ulp * foo)) {
        // ==== Deflatable ====
        *ns = *ns - 2;
      } else {
        // ==== Undeflatable. Move them up out of the way.
        // .    Fortunately, DTREXC does the right thing with
        // .    ILST in case of a rare exchange failure. ====
        ifst = *ns;
        info = Trexc('V', jw, t, ldt, v, ldv, &ifst, &ilst, work);
        ilst = ilst + 2;
      }
    }
    // ==== End deflation detection loop ====
    goto L20;
  }
  // ==== Return to Hessenberg form ====
  if (*ns == 0) s = zero;
  if (*ns < jw) {
    // ==== sorting diagonal blocks of T improves accuracy for
    // .    graded matrices.  Bubble sort deals well with
    // .    exchange failures. ====
    sorted = false;
    i = *ns + 1;
  L30:
    if (sorted) goto L50;
    sorted = true;
    kend = i - 1;
    i = infqr + 1;
    if (i == *ns) {
      k = i + 1;
    } else if (Tm(i + 1, i) == zero) {
      k = i + 1;
    } else {
      k = i + 2;
    }
  L40:
    if (k <= kend) {
      if (k == i + 1) {
        evi = std::abs(Tm(i, i));
      } else {
        evi = std::abs(Tm(i, i)) + std::sqrt(std::abs(Tm(i + 1, i))) *
                                       std::sqrt(std::abs(Tm(i, i + 1)));
      }
      if (k == kend) {
        evk = std::abs(Tm(k, k));
      } else if (Tm(k + 1, k) == zero) {
        evk = std::abs(Tm(k, k));
      } else {
        evk = std::abs(Tm(k, k)) + std::sqrt(std::abs(Tm(k + 1, k))) *
                                       std::sqrt(std::abs(Tm(k, k + 1)));
      }
      if (evi >= evk) {
        i = k;
      } else {
        sorted = false;
        ifst = i;
        ilst = k;
        info = Trexc('V', jw, t, ldt, v, ldv, &ifst, &ilst, work);
        if (info == 0) {
          i = ilst;
        } else {
          i = k;
        }
      }
      if (i == kend) {
        k = i + 1;
      } else if (Tm(i + 1, i) == zero) {
        k = i + 1;
      } else {
        k = i + 2;
      }
      goto L40;
    }
    goto L30;
  L50:;
  }
  // ==== Restore shift/eigenvalue array from T ====
  i = jw;
L60:
  if (i >= infqr + 1) {
    if (i == infqr + 1) {
      SR(kwtop + i - 1) = Tm(i, i);
      SI(kwtop + i - 1) = zero;
      i = i - 1;
    } else if (Tm(i, i - 1) == zero) {
      SR(kwtop + i - 1) = Tm(i, i);
      SI(kwtop + i - 1) = zero;
      i = i - 1;
    } else {
      aa = Tm(i - 1, i - 1);
      cc = Tm(i, i - 1);
      bb = Tm(i - 1, i);
      dd = Tm(i, i);
      Lanv2(&aa, &bb, &cc, &dd, SR.ptr(kwtop + i - 2), SI.ptr(kwtop + i - 2),
            SR.ptr(kwtop + i - 1), SI.ptr(kwtop + i - 1), &cs, &sn);
      i = i - 2;
    }
    goto L60;
  }
  if (*ns < jw || s == zero) {
    if (*ns > 1 && s != zero) {
      // ==== Reflect spike back into lower triangle ====
      Copy(*ns, v, ldv, work, 1);
      beta = WORK(1);
      Larfg(*ns, &beta, WORK.ptr(2), 1, &tau);
      Laset('L', jw - 2, jw - 2, zero, zero, Tm.ptr(3, 1), ldt);
      Larf1f('L', *ns, jw, work, 1, tau, t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', *ns, *ns, work, 1, tau, t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', jw, *ns, work, 1, tau, v, ldv, WORK.ptr(jw + 1));
      info = Gehrd(jw, 1, *ns, t, ldt, work, WORK.ptr(jw + 1), lwork - jw);
    }
    // ==== Copy updated reduced window into place ====
    if (kwtop > 1) H(kwtop, kwtop - 1) = s * V(1, 1);
    Lacpy('U', jw, jw, t, ldt, H.ptr(kwtop, kwtop), ldh);
    Copy(jw - 1, Tm.ptr(2, 1), ldt + 1, H.ptr(kwtop + 1, kwtop), ldh + 1);
    // ==== Accumulate orthogonal matrix in order update
    // .    H and Z, if requested.  ====
    if (*ns > 1 && s != zero)
      info = Unmhr('R', 'N', jw, *ns, 1, *ns, t, ldt, work, v, ldv,
                   WORK.ptr(jw + 1), lwork - jw);
    // ==== Update vertical slab in H ====
    if (wantt) {
      ltop = 1;
    } else {
      ltop = ktop;
    }
    for (krow = ltop; krow <= kwtop - 1; krow += nv) {
      kln = Min<blasint>(nv, kwtop - krow);
      Gemm('N', 'N', kln, jw, jw, one, H.ptr(krow, kwtop), ldh, v, ldv, zero,
           wv, ldwv);
      Lacpy('A', kln, jw, wv, ldwv, H.ptr(krow, kwtop), ldh);
    }
    // ==== Update horizontal slab in H ====
    if (wantt) {
      for (kcol = kbot + 1; kcol <= n; kcol += nh) {
        kln = Min<blasint>(nh, n - kcol + 1);
        Gemm('C', 'N', jw, kln, jw, one, v, ldv, H.ptr(kwtop, kcol), ldh, zero,
             t, ldt);
        Lacpy('A', jw, kln, t, ldt, H.ptr(kwtop, kcol), ldh);
      }
    }
    // ==== Update vertical slab in Z ====
    if (wantz) {
      for (krow = iloz; krow <= ihiz; krow += nv) {
        kln = Min<blasint>(nv, ihiz - krow + 1);
        Gemm('N', 'N', kln, jw, jw, one, Z.ptr(krow, kwtop), ldz, v, ldv, zero,
             wv, ldwv);
        Lacpy('A', kln, jw, wv, ldwv, Z.ptr(krow, kwtop), ldz);
      }
    }
  }
  // ==== Return the number of deflations ... ====
  *nd = jw - *ns;
  // ==== ... and the number of shifts. (Subtracting
  // .    INFQR from the spike length takes care
  // .    of the case of a rare QR failure while
  // .    calculating eigenvalues of the deflation
  // .    window.)  ====
  *ns = *ns - infqr;
  // ==== Return optimal workspace. ====
  WORK(1) = WorkSize<R>(lwkopt);
  // ==== End of DLAQR2 ====
}

// xLAQR2 (complex T): as the real routine, with the shifts in SH.
template <typename T>
void Laqr2(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, T* h, blasint ldh, blasint iloz, blasint ihiz, T* z,
           blasint ldz, blasint* ns, blasint* nd, T* sh, T* v, blasint ldv,
           blasint nh, T* t, blasint ldt, blasint nv, T* wv, blasint ldwv,
           T* work, blasint lwork) {
  using R = RealT<T>;

  const T zero = T(0);
  const T one = T(1);
  const R rzero = R(0);
  const R rone = R(1);
  const Matrix<T> H(h, ldh);
  const Vector<T> SH(sh);
  const Matrix<T> Tm(t, ldt);
  const Matrix<T> V(v, ldv);
  const Vector<T> WORK(work);
  const Matrix<T> WV(wv, ldwv);
  const Matrix<T> Z(z, ldz);
  blasint i, ifst, ilst, infqr, j, jw, kcol, kln, knt, krow, kwtop, ltop, lwk1,
      lwk2, lwkopt;
  R foo, safmax, safmin, smlnum, ulp;
  T s, tau;
  jw = Min<blasint>(nw, kbot - ktop + 1);
  if (jw <= 2) {
    lwkopt = 1;
  } else {
    // ==== Workspace query call to ZGEHRD ====
    Gehrd(jw, 1, jw - 1, t, ldt, work, work, -1);
    lwk1 = static_cast<blasint>(std::real(WORK(1)));
    // ==== Workspace query call to ZUNMHR ====
    Unmhr('R', 'N', jw, jw, 1, jw - 1, t, ldt, work, v, ldv, work, -1);
    lwk2 = static_cast<blasint>(std::real(WORK(1)));
    // ==== Optimal workspace ====
    lwkopt = jw + Max<blasint>(lwk1, lwk2);
  }
  // ==== Quick return in case of workspace query. ====
  if (lwork == -1) {
    WORK(1) = WorkSize<T>(lwkopt);
    return;
  }
  // ==== Nothing to do ...
  *ns = 0;
  *nd = 0;
  WORK(1) = one;
  if (ktop > kbot) return;
  if (nw < 1) return;
  // ==== Machine constants ====
  safmin = Lamch<R>('S');
  safmax = rone / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(n) / ulp);
  // ==== Setup deflation window ====
  jw = Min<blasint>(nw, kbot - ktop + 1);
  kwtop = kbot - jw + 1;
  if (kwtop == ktop) {
    s = zero;
  } else {
    s = H(kwtop, kwtop - 1);
  }
  if (kbot == kwtop) {
    // ==== 1-by-1 deflation window: not much to do ====
    SH(kwtop) = H(kwtop, kwtop);
    *ns = 1;
    *nd = 0;
    if (Cabs1(s) <= Max(smlnum, ulp * Cabs1(H(kwtop, kwtop)))) {
      *ns = 0;
      *nd = 1;
      if (kwtop > ktop) H(kwtop, kwtop - 1) = zero;
    }
    WORK(1) = one;
    return;
  }
  // ==== Convert to spike-triangular form.  (In case of a
  // .    rare QR failure, this routine continues to do
  // .    aggressive early deflation using that part of
  // .    the deflation window that converged using INFQR
  // .    here and there to keep track.) ====
  Lacpy('U', jw, jw, H.ptr(kwtop, kwtop), ldh, t, ldt);
  Copy(jw - 1, H.ptr(kwtop + 1, kwtop), ldh + 1, Tm.ptr(2, 1), ldt + 1);
  Laset('A', jw, jw, zero, one, v, ldv);
  infqr = Lahqr(true, true, jw, 1, jw, t, ldt, SH.ptr(kwtop), 1, jw, v, ldv);
  // ==== Deflation detection loop ====
  *ns = jw;
  ilst = infqr + 1;
  for (knt = infqr + 1; knt <= jw; ++knt) {
    // ==== Small spike tip deflation test ====
    foo = Cabs1(Tm(*ns, *ns));
    if (foo == rzero) foo = Cabs1(s);
    if (Cabs1(s) * Cabs1(V(1, *ns)) <= Max(smlnum, ulp * foo)) {
      // ==== One more converged eigenvalue ====
      *ns = *ns - 1;
    } else {
      // ==== One undeflatable eigenvalue.  Move it up out of the
      // .    way.   (ZTREXC can not fail in this case.) ====
      ifst = *ns;
      Trexc('V', jw, t, ldt, v, ldv, ifst, ilst);
      ilst = ilst + 1;
    }
  }
  // ==== Return to Hessenberg form ====
  if (*ns == 0) s = zero;
  if (*ns < jw) {
    // ==== sorting the diagonal of T improves accuracy for
    // .    graded matrices.  ====
    for (i = infqr + 1; i <= *ns; ++i) {
      ifst = i;
      for (j = i + 1; j <= *ns; ++j) {
        if (Cabs1(Tm(j, j)) > Cabs1(Tm(ifst, ifst))) ifst = j;
      }
      ilst = i;
      if (ifst != ilst) Trexc('V', jw, t, ldt, v, ldv, ifst, ilst);
    }
  }
  // ==== Restore shift/eigenvalue array from T ====
  for (i = infqr + 1; i <= jw; ++i) {
    SH(kwtop + i - 1) = Tm(i, i);
  }
  if (*ns < jw || s == zero) {
    if (*ns > 1 && s != zero) {
      // ==== Reflect spike back into lower triangle ====
      Copy(*ns, v, ldv, work, 1);
      for (i = 1; i <= *ns; ++i) {
        WORK(i) = Conj(WORK(i));
      }
      Larfg(*ns, WORK.ptr(1), WORK.ptr(2), 1, &tau);
      Laset('L', jw - 2, jw - 2, zero, zero, Tm.ptr(3, 1), ldt);
      Larf1f('L', *ns, jw, work, 1, Conj(tau), t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', *ns, *ns, work, 1, tau, t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', jw, *ns, work, 1, tau, v, ldv, WORK.ptr(jw + 1));
      Gehrd(jw, 1, *ns, t, ldt, work, WORK.ptr(jw + 1), lwork - jw);
    }
    // ==== Copy updated reduced window into place ====
    if (kwtop > 1) H(kwtop, kwtop - 1) = s * Conj(V(1, 1));
    Lacpy('U', jw, jw, t, ldt, H.ptr(kwtop, kwtop), ldh);
    Copy(jw - 1, Tm.ptr(2, 1), ldt + 1, H.ptr(kwtop + 1, kwtop), ldh + 1);
    // ==== Accumulate orthogonal matrix in order update
    // .    H and Z, if requested.  ====
    if (*ns > 1 && s != zero)
      Unmhr('R', 'N', jw, *ns, 1, *ns, t, ldt, work, v, ldv, WORK.ptr(jw + 1),
            lwork - jw);
    // ==== Update vertical slab in H ====
    if (wantt) {
      ltop = 1;
    } else {
      ltop = ktop;
    }
    for (krow = ltop; krow <= kwtop - 1; krow += nv) {
      kln = Min<blasint>(nv, kwtop - krow);
      Gemm('N', 'N', kln, jw, jw, one, H.ptr(krow, kwtop), ldh, v, ldv, zero,
           wv, ldwv);
      Lacpy('A', kln, jw, wv, ldwv, H.ptr(krow, kwtop), ldh);
    }
    // ==== Update horizontal slab in H ====
    if (wantt) {
      for (kcol = kbot + 1; kcol <= n; kcol += nh) {
        kln = Min<blasint>(nh, n - kcol + 1);
        Gemm('C', 'N', jw, kln, jw, one, v, ldv, H.ptr(kwtop, kcol), ldh, zero,
             t, ldt);
        Lacpy('A', jw, kln, t, ldt, H.ptr(kwtop, kcol), ldh);
      }
    }
    // ==== Update vertical slab in Z ====
    if (wantz) {
      for (krow = iloz; krow <= ihiz; krow += nv) {
        kln = Min<blasint>(nv, ihiz - krow + 1);
        Gemm('N', 'N', kln, jw, jw, one, Z.ptr(krow, kwtop), ldz, v, ldv, zero,
             wv, ldwv);
        Lacpy('A', kln, jw, wv, ldwv, Z.ptr(krow, kwtop), ldz);
      }
    }
  }
  // ==== Return the number of deflations ... ====
  *nd = jw - *ns;
  // ==== ... and the number of shifts. (Subtracting
  // .    INFQR from the spike length takes care
  // .    of the case of a rare QR failure while
  // .    calculating eigenvalues of the deflation
  // .    window.)  ====
  *ns = *ns - infqr;
  // ==== Return optimal workspace. ====
  WORK(1) = WorkSize<T>(lwkopt);
  // ==== End of ZLAQR2 ====
}

// xLAQR3 (real T): aggressive early deflation as xLAQR2, with xLAQR4 for
// the Schur form of large windows.
template <typename R>
void Laqr3(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, R* h, blasint ldh, blasint iloz, blasint ihiz, R* z,
           blasint ldz, blasint* ns, blasint* nd, R* sr, R* si, R* v,
           blasint ldv, blasint nh, R* t, blasint ldt, blasint nv, R* wv,
           blasint ldwv, R* work, blasint lwork) {
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> H(h, ldh);
  const Vector<R> SI(si);
  const Vector<R> SR(sr);
  const Matrix<R> Tm(t, ldt);
  const Matrix<R> V(v, ldv);
  const Vector<R> WORK(work);
  const Matrix<R> WV(wv, ldwv);
  const Matrix<R> Z(z, ldz);
  blasint i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, kln, krow, kwtop,
      ltop, lwk1, lwk2, lwk3, lwkopt, nmin;
  bool bulge, sorted;
  R aa, bb, beta, cc, cs, dd, evi, evk, foo, s, safmax, safmin, smlnum, sn, tau,
      ulp;
  jw = Min<blasint>(nw, kbot - ktop + 1);
  if (jw <= 2) {
    lwkopt = 1;
  } else {
    // ==== Workspace query call to DGEHRD ====
    info = Gehrd(jw, 1, jw - 1, t, ldt, work, work, -1);
    lwk1 = static_cast<blasint>(WORK(1));
    // ==== Workspace query call to DORMHR ====
    info = Unmhr('R', 'N', jw, jw, 1, jw - 1, t, ldt, work, v, ldv, work, -1);
    lwk2 = static_cast<blasint>(WORK(1));
    // ==== Workspace query call to DLAQR4 ====
    infqr =
        Laqr4(true, true, jw, 1, jw, t, ldt, sr, si, 1, jw, v, ldv, work, -1);
    lwk3 = static_cast<blasint>(WORK(1));
    // ==== Optimal workspace ====
    lwkopt = Max<blasint>(jw + Max<blasint>(lwk1, lwk2), lwk3);
  }
  // ==== Quick return in case of workspace query. ====
  if (lwork == -1) {
    WORK(1) = WorkSize<R>(lwkopt);
    return;
  }
  // ==== Nothing to do ...
  *ns = 0;
  *nd = 0;
  WORK(1) = one;
  if (ktop > kbot) return;
  if (nw < 1) return;
  // ==== Machine constants ====
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(n) / ulp);
  // ==== Setup deflation window ====
  jw = Min<blasint>(nw, kbot - ktop + 1);
  kwtop = kbot - jw + 1;
  if (kwtop == ktop) {
    s = zero;
  } else {
    s = H(kwtop, kwtop - 1);
  }
  if (kbot == kwtop) {
    // ==== 1-by-1 deflation window: not much to do ====
    SR(kwtop) = H(kwtop, kwtop);
    SI(kwtop) = zero;
    *ns = 1;
    *nd = 0;
    if (std::abs(s) <= Max(smlnum, ulp * std::abs(H(kwtop, kwtop)))) {
      *ns = 0;
      *nd = 1;
      if (kwtop > ktop) H(kwtop, kwtop - 1) = zero;
    }
    WORK(1) = one;
    return;
  }
  // ==== Convert to spike-triangular form.  (In case of a
  // .    rare QR failure, this routine continues to do
  // .    aggressive early deflation using that part of
  // .    the deflation window that converged using INFQR
  // .    here and there to keep track.) ====
  Lacpy('U', jw, jw, H.ptr(kwtop, kwtop), ldh, t, ldt);
  Copy(jw - 1, H.ptr(kwtop + 1, kwtop), ldh + 1, Tm.ptr(2, 1), ldt + 1);
  Laset('A', jw, jw, zero, one, v, ldv);
  nmin = Ilaenv<R>(12, "LAQR3", "SV", jw, 1, jw, lwork);
  if (jw > nmin) {
    infqr = Laqr4(true, true, jw, 1, jw, t, ldt, SR.ptr(kwtop), SI.ptr(kwtop),
                  1, jw, v, ldv, work, lwork);
  } else {
    infqr = Lahqr(true, true, jw, 1, jw, t, ldt, SR.ptr(kwtop), SI.ptr(kwtop),
                  1, jw, v, ldv);
  }
  // ==== DTREXC needs a clean margin near the diagonal ====
  for (j = 1; j <= jw - 3; ++j) {
    Tm(j + 2, j) = zero;
    Tm(j + 3, j) = zero;
  }
  if (jw > 2) Tm(jw, jw - 2) = zero;
  // ==== Deflation detection loop ====
  *ns = jw;
  ilst = infqr + 1;
L20:
  if (ilst <= *ns) {
    if (*ns == 1) {
      bulge = false;
    } else {
      bulge = Tm(*ns, *ns - 1) != zero;
    }
    // ==== Small spike tip test for deflation ====
    if (!bulge) {
      // ==== Real eigenvalue ====
      foo = std::abs(Tm(*ns, *ns));
      if (foo == zero) foo = std::abs(s);
      if (std::abs(s * V(1, *ns)) <= Max(smlnum, ulp * foo)) {
        // ==== Deflatable ====
        *ns = *ns - 1;
      } else {
        // ==== Undeflatable.   Move it up out of the way.
        // .    (DTREXC can not fail in this case.) ====
        ifst = *ns;
        info = Trexc('V', jw, t, ldt, v, ldv, &ifst, &ilst, work);
        ilst = ilst + 1;
      }
    } else {
      // ==== Complex conjugate pair ====
      foo = std::abs(Tm(*ns, *ns)) + std::sqrt(std::abs(Tm(*ns, *ns - 1))) *
                                         std::sqrt(std::abs(Tm(*ns - 1, *ns)));
      if (foo == zero) foo = std::abs(s);
      if (Max(std::abs(s * V(1, *ns)), std::abs(s * V(1, *ns - 1))) <=
          Max(smlnum, ulp * foo)) {
        // ==== Deflatable ====
        *ns = *ns - 2;
      } else {
        // ==== Undeflatable. Move them up out of the way.
        // .    Fortunately, DTREXC does the right thing with
        // .    ILST in case of a rare exchange failure. ====
        ifst = *ns;
        info = Trexc('V', jw, t, ldt, v, ldv, &ifst, &ilst, work);
        ilst = ilst + 2;
      }
    }
    // ==== End deflation detection loop ====
    goto L20;
  }
  // ==== Return to Hessenberg form ====
  if (*ns == 0) s = zero;
  if (*ns < jw) {
    // ==== sorting diagonal blocks of T improves accuracy for
    // .    graded matrices.  Bubble sort deals well with
    // .    exchange failures. ====
    sorted = false;
    i = *ns + 1;
  L30:
    if (sorted) goto L50;
    sorted = true;
    kend = i - 1;
    i = infqr + 1;
    if (i == *ns) {
      k = i + 1;
    } else if (Tm(i + 1, i) == zero) {
      k = i + 1;
    } else {
      k = i + 2;
    }
  L40:
    if (k <= kend) {
      if (k == i + 1) {
        evi = std::abs(Tm(i, i));
      } else {
        evi = std::abs(Tm(i, i)) + std::sqrt(std::abs(Tm(i + 1, i))) *
                                       std::sqrt(std::abs(Tm(i, i + 1)));
      }
      if (k == kend) {
        evk = std::abs(Tm(k, k));
      } else if (Tm(k + 1, k) == zero) {
        evk = std::abs(Tm(k, k));
      } else {
        evk = std::abs(Tm(k, k)) + std::sqrt(std::abs(Tm(k + 1, k))) *
                                       std::sqrt(std::abs(Tm(k, k + 1)));
      }
      if (evi >= evk) {
        i = k;
      } else {
        sorted = false;
        ifst = i;
        ilst = k;
        info = Trexc('V', jw, t, ldt, v, ldv, &ifst, &ilst, work);
        if (info == 0) {
          i = ilst;
        } else {
          i = k;
        }
      }
      if (i == kend) {
        k = i + 1;
      } else if (Tm(i + 1, i) == zero) {
        k = i + 1;
      } else {
        k = i + 2;
      }
      goto L40;
    }
    goto L30;
  L50:;
  }
  // ==== Restore shift/eigenvalue array from T ====
  i = jw;
L60:
  if (i >= infqr + 1) {
    if (i == infqr + 1) {
      SR(kwtop + i - 1) = Tm(i, i);
      SI(kwtop + i - 1) = zero;
      i = i - 1;
    } else if (Tm(i, i - 1) == zero) {
      SR(kwtop + i - 1) = Tm(i, i);
      SI(kwtop + i - 1) = zero;
      i = i - 1;
    } else {
      aa = Tm(i - 1, i - 1);
      cc = Tm(i, i - 1);
      bb = Tm(i - 1, i);
      dd = Tm(i, i);
      Lanv2(&aa, &bb, &cc, &dd, SR.ptr(kwtop + i - 2), SI.ptr(kwtop + i - 2),
            SR.ptr(kwtop + i - 1), SI.ptr(kwtop + i - 1), &cs, &sn);
      i = i - 2;
    }
    goto L60;
  }
  if (*ns < jw || s == zero) {
    if (*ns > 1 && s != zero) {
      // ==== Reflect spike back into lower triangle ====
      Copy(*ns, v, ldv, work, 1);
      beta = WORK(1);
      Larfg(*ns, &beta, WORK.ptr(2), 1, &tau);
      Laset('L', jw - 2, jw - 2, zero, zero, Tm.ptr(3, 1), ldt);
      Larf1f('L', *ns, jw, work, 1, tau, t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', *ns, *ns, work, 1, tau, t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', jw, *ns, work, 1, tau, v, ldv, WORK.ptr(jw + 1));
      info = Gehrd(jw, 1, *ns, t, ldt, work, WORK.ptr(jw + 1), lwork - jw);
    }
    // ==== Copy updated reduced window into place ====
    if (kwtop > 1) H(kwtop, kwtop - 1) = s * V(1, 1);
    Lacpy('U', jw, jw, t, ldt, H.ptr(kwtop, kwtop), ldh);
    Copy(jw - 1, Tm.ptr(2, 1), ldt + 1, H.ptr(kwtop + 1, kwtop), ldh + 1);
    // ==== Accumulate orthogonal matrix in order update
    // .    H and Z, if requested.  ====
    if (*ns > 1 && s != zero)
      info = Unmhr('R', 'N', jw, *ns, 1, *ns, t, ldt, work, v, ldv,
                   WORK.ptr(jw + 1), lwork - jw);
    // ==== Update vertical slab in H ====
    if (wantt) {
      ltop = 1;
    } else {
      ltop = ktop;
    }
    for (krow = ltop; krow <= kwtop - 1; krow += nv) {
      kln = Min<blasint>(nv, kwtop - krow);
      Gemm('N', 'N', kln, jw, jw, one, H.ptr(krow, kwtop), ldh, v, ldv, zero,
           wv, ldwv);
      Lacpy('A', kln, jw, wv, ldwv, H.ptr(krow, kwtop), ldh);
    }
    // ==== Update horizontal slab in H ====
    if (wantt) {
      for (kcol = kbot + 1; kcol <= n; kcol += nh) {
        kln = Min<blasint>(nh, n - kcol + 1);
        Gemm('C', 'N', jw, kln, jw, one, v, ldv, H.ptr(kwtop, kcol), ldh, zero,
             t, ldt);
        Lacpy('A', jw, kln, t, ldt, H.ptr(kwtop, kcol), ldh);
      }
    }
    // ==== Update vertical slab in Z ====
    if (wantz) {
      for (krow = iloz; krow <= ihiz; krow += nv) {
        kln = Min<blasint>(nv, ihiz - krow + 1);
        Gemm('N', 'N', kln, jw, jw, one, Z.ptr(krow, kwtop), ldz, v, ldv, zero,
             wv, ldwv);
        Lacpy('A', kln, jw, wv, ldwv, Z.ptr(krow, kwtop), ldz);
      }
    }
  }
  // ==== Return the number of deflations ... ====
  *nd = jw - *ns;
  // ==== ... and the number of shifts. (Subtracting
  // .    INFQR from the spike length takes care
  // .    of the case of a rare QR failure while
  // .    calculating eigenvalues of the deflation
  // .    window.)  ====
  *ns = *ns - infqr;
  // ==== Return optimal workspace. ====
  WORK(1) = WorkSize<R>(lwkopt);
  // ==== End of DLAQR3 ====
}

// xLAQR3 (complex T): as the real routine, with the shifts in SH.
template <typename T>
void Laqr3(bool wantt, bool wantz, blasint n, blasint ktop, blasint kbot,
           blasint nw, T* h, blasint ldh, blasint iloz, blasint ihiz, T* z,
           blasint ldz, blasint* ns, blasint* nd, T* sh, T* v, blasint ldv,
           blasint nh, T* t, blasint ldt, blasint nv, T* wv, blasint ldwv,
           T* work, blasint lwork) {
  using R = RealT<T>;

  const T zero = T(0);
  const T one = T(1);
  const R rzero = R(0);
  const R rone = R(1);
  const Matrix<T> H(h, ldh);
  const Vector<T> SH(sh);
  const Matrix<T> Tm(t, ldt);
  const Matrix<T> V(v, ldv);
  const Vector<T> WORK(work);
  const Matrix<T> WV(wv, ldwv);
  const Matrix<T> Z(z, ldz);
  blasint i, ifst, ilst, infqr, j, jw, kcol, kln, knt, krow, kwtop, ltop, lwk1,
      lwk2, lwk3, lwkopt, nmin;
  R foo, safmax, safmin, smlnum, ulp;
  T s, tau;
  jw = Min<blasint>(nw, kbot - ktop + 1);
  if (jw <= 2) {
    lwkopt = 1;
  } else {
    // ==== Workspace query call to ZGEHRD ====
    Gehrd(jw, 1, jw - 1, t, ldt, work, work, -1);
    lwk1 = static_cast<blasint>(std::real(WORK(1)));
    // ==== Workspace query call to ZUNMHR ====
    Unmhr('R', 'N', jw, jw, 1, jw - 1, t, ldt, work, v, ldv, work, -1);
    lwk2 = static_cast<blasint>(std::real(WORK(1)));
    // ==== Workspace query call to ZLAQR4 ====
    infqr = Laqr4(true, true, jw, 1, jw, t, ldt, sh, 1, jw, v, ldv, work, -1);
    lwk3 = static_cast<blasint>(std::real(WORK(1)));
    // ==== Optimal workspace ====
    lwkopt = Max<blasint>(jw + Max<blasint>(lwk1, lwk2), lwk3);
  }
  // ==== Quick return in case of workspace query. ====
  if (lwork == -1) {
    WORK(1) = WorkSize<T>(lwkopt);
    return;
  }
  // ==== Nothing to do ...
  *ns = 0;
  *nd = 0;
  WORK(1) = one;
  if (ktop > kbot) return;
  if (nw < 1) return;
  // ==== Machine constants ====
  safmin = Lamch<R>('S');
  safmax = rone / safmin;
  ulp = Lamch<R>('P');
  smlnum = safmin * (R(n) / ulp);
  // ==== Setup deflation window ====
  jw = Min<blasint>(nw, kbot - ktop + 1);
  kwtop = kbot - jw + 1;
  if (kwtop == ktop) {
    s = zero;
  } else {
    s = H(kwtop, kwtop - 1);
  }
  if (kbot == kwtop) {
    // ==== 1-by-1 deflation window: not much to do ====
    SH(kwtop) = H(kwtop, kwtop);
    *ns = 1;
    *nd = 0;
    if (Cabs1(s) <= Max(smlnum, ulp * Cabs1(H(kwtop, kwtop)))) {
      *ns = 0;
      *nd = 1;
      if (kwtop > ktop) H(kwtop, kwtop - 1) = zero;
    }
    WORK(1) = one;
    return;
  }
  // ==== Convert to spike-triangular form.  (In case of a
  // .    rare QR failure, this routine continues to do
  // .    aggressive early deflation using that part of
  // .    the deflation window that converged using INFQR
  // .    here and there to keep track.) ====
  Lacpy('U', jw, jw, H.ptr(kwtop, kwtop), ldh, t, ldt);
  Copy(jw - 1, H.ptr(kwtop + 1, kwtop), ldh + 1, Tm.ptr(2, 1), ldt + 1);
  Laset('A', jw, jw, zero, one, v, ldv);
  nmin = Ilaenv<T>(12, "LAQR3", "SV", jw, 1, jw, lwork);
  if (jw > nmin) {
    infqr = Laqr4(true, true, jw, 1, jw, t, ldt, SH.ptr(kwtop), 1, jw, v, ldv,
                  work, lwork);
  } else {
    infqr = Lahqr(true, true, jw, 1, jw, t, ldt, SH.ptr(kwtop), 1, jw, v, ldv);
  }
  // ==== Deflation detection loop ====
  *ns = jw;
  ilst = infqr + 1;
  for (knt = infqr + 1; knt <= jw; ++knt) {
    // ==== Small spike tip deflation test ====
    foo = Cabs1(Tm(*ns, *ns));
    if (foo == rzero) foo = Cabs1(s);
    if (Cabs1(s) * Cabs1(V(1, *ns)) <= Max(smlnum, ulp * foo)) {
      // ==== One more converged eigenvalue ====
      *ns = *ns - 1;
    } else {
      // ==== One undeflatable eigenvalue.  Move it up out of the
      // .    way.   (ZTREXC can not fail in this case.) ====
      ifst = *ns;
      Trexc('V', jw, t, ldt, v, ldv, ifst, ilst);
      ilst = ilst + 1;
    }
  }
  // ==== Return to Hessenberg form ====
  if (*ns == 0) s = zero;
  if (*ns < jw) {
    // ==== sorting the diagonal of T improves accuracy for
    // .    graded matrices.  ====
    for (i = infqr + 1; i <= *ns; ++i) {
      ifst = i;
      for (j = i + 1; j <= *ns; ++j) {
        if (Cabs1(Tm(j, j)) > Cabs1(Tm(ifst, ifst))) ifst = j;
      }
      ilst = i;
      if (ifst != ilst) Trexc('V', jw, t, ldt, v, ldv, ifst, ilst);
    }
  }
  // ==== Restore shift/eigenvalue array from T ====
  for (i = infqr + 1; i <= jw; ++i) {
    SH(kwtop + i - 1) = Tm(i, i);
  }
  if (*ns < jw || s == zero) {
    if (*ns > 1 && s != zero) {
      // ==== Reflect spike back into lower triangle ====
      Copy(*ns, v, ldv, work, 1);
      for (i = 1; i <= *ns; ++i) {
        WORK(i) = Conj(WORK(i));
      }
      Larfg(*ns, WORK.ptr(1), WORK.ptr(2), 1, &tau);
      Laset('L', jw - 2, jw - 2, zero, zero, Tm.ptr(3, 1), ldt);
      Larf1f('L', *ns, jw, work, 1, Conj(tau), t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', *ns, *ns, work, 1, tau, t, ldt, WORK.ptr(jw + 1));
      Larf1f('R', jw, *ns, work, 1, tau, v, ldv, WORK.ptr(jw + 1));
      Gehrd(jw, 1, *ns, t, ldt, work, WORK.ptr(jw + 1), lwork - jw);
    }
    // ==== Copy updated reduced window into place ====
    if (kwtop > 1) H(kwtop, kwtop - 1) = s * Conj(V(1, 1));
    Lacpy('U', jw, jw, t, ldt, H.ptr(kwtop, kwtop), ldh);
    Copy(jw - 1, Tm.ptr(2, 1), ldt + 1, H.ptr(kwtop + 1, kwtop), ldh + 1);
    // ==== Accumulate orthogonal matrix in order update
    // .    H and Z, if requested.  ====
    if (*ns > 1 && s != zero)
      Unmhr('R', 'N', jw, *ns, 1, *ns, t, ldt, work, v, ldv, WORK.ptr(jw + 1),
            lwork - jw);
    // ==== Update vertical slab in H ====
    if (wantt) {
      ltop = 1;
    } else {
      ltop = ktop;
    }
    for (krow = ltop; krow <= kwtop - 1; krow += nv) {
      kln = Min<blasint>(nv, kwtop - krow);
      Gemm('N', 'N', kln, jw, jw, one, H.ptr(krow, kwtop), ldh, v, ldv, zero,
           wv, ldwv);
      Lacpy('A', kln, jw, wv, ldwv, H.ptr(krow, kwtop), ldh);
    }
    // ==== Update horizontal slab in H ====
    if (wantt) {
      for (kcol = kbot + 1; kcol <= n; kcol += nh) {
        kln = Min<blasint>(nh, n - kcol + 1);
        Gemm('C', 'N', jw, kln, jw, one, v, ldv, H.ptr(kwtop, kcol), ldh, zero,
             t, ldt);
        Lacpy('A', jw, kln, t, ldt, H.ptr(kwtop, kcol), ldh);
      }
    }
    // ==== Update vertical slab in Z ====
    if (wantz) {
      for (krow = iloz; krow <= ihiz; krow += nv) {
        kln = Min<blasint>(nv, ihiz - krow + 1);
        Gemm('N', 'N', kln, jw, jw, one, Z.ptr(krow, kwtop), ldz, v, ldv, zero,
             wv, ldwv);
        Lacpy('A', kln, jw, wv, ldwv, Z.ptr(krow, kwtop), ldz);
      }
    }
  }
  // ==== Return the number of deflations ... ====
  *nd = jw - *ns;
  // ==== ... and the number of shifts. (Subtracting
  // .    INFQR from the spike length takes care
  // .    of the case of a rare QR failure while
  // .    calculating eigenvalues of the deflation
  // .    window.)  ====
  *ns = *ns - infqr;
  // ==== Return optimal workspace. ====
  WORK(1) = WorkSize<T>(lwkopt);
  // ==== End of ZLAQR3 ====
}

// xLAQR4 (real T): the multishift QR sweeps of xLAQR0, with xLAQR2 for
// aggressive early deflation (it is called by xLAQR3 for the deflation
// window, which avoids a recursion).
template <typename R>
blasint Laqr4(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, blasint iloz, blasint ihiz, R* z,
              blasint ldz, R* work, blasint lwork) {
  const blasint ntiny = 15;
  const blasint kexnw = 5;
  const blasint kexsh = 6;
  const R wilk1 = R(0.75);
  const R wilk2 = -R(0.4375);
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> H(h, ldh);
  char jbcmpz[3] = {' ', ' ', '\0'};
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  const Matrix<R> Z(z, ldz);
  R zdum[1];
  blasint i, inf, it, itmax, k, kacc22, kbot, kdu, ks, kt, ktop, ku, kv, kwh,
      kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh, nho, nibble, nmin, ns, nsmax,
      nsr, nve, nw, nwmax, nwr, nwupbd;
  bool sorted;
  R aa, bb, cc, cs, dd, sn, ss, swap;
  blasint info = 0;
  // ==== Quick return for N = 0: nothing to do. ====
  if (n == 0) {
    WORK(1) = one;
    return info;
  }
  if (n <= ntiny) {
    // ==== Tiny matrices must use DLAHQR. ====
    lwkopt = 1;
    if (lwork != -1)
      info =
          Lahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz);
  } else {
    // ==== Use small bulge multi-shift QR with aggressive early
    // .    deflation on larger-than-tiny matrices. ====
    // ==== Hope for the best. ====
    info = 0;
    // ==== Set up job flags for ILAENV. ====
    if (wantt) {
      jbcmpz[0] = 'S';
    } else {
      jbcmpz[0] = 'E';
    }
    if (wantz) {
      jbcmpz[1] = 'V';
    } else {
      jbcmpz[1] = 'N';
    }
    // ==== NWR = recommended deflation window size.  At this
    // .    point,  N .GT. NTINY = 15, so there is enough
    // .    subdiagonal workspace for NWR.GE.2 as required.
    // .    (In fact, there is enough subdiagonal space for
    // .    NWR.GE.4.) ====
    nwr = Ilaenv<R>(13, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nwr = Max<blasint>(2, nwr);
    nwr = Min<blasint>(ihi - ilo + 1, (n - 1) / 3, nwr);
    // ==== NSR = recommended number of simultaneous shifts.
    // .    At this point N .GT. NTINY = 15, so there is at
    // .    enough subdiagonal workspace for NSR to be even
    // .    and greater than or equal to two as required. ====
    nsr = Ilaenv<R>(15, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nsr = Min<blasint>(nsr, (n - 3) / 6, ihi - ilo);
    nsr = Max<blasint>(2, nsr - nsr % 2);
    // ==== Estimate optimal workspace ====
    // ==== Workspace query call to DLAQR2 ====
    Laqr2(wantt, wantz, n, ilo, ihi, nwr + 1, h, ldh, iloz, ihiz, z, ldz, &ls,
          &ld, wr, wi, h, ldh, n, h, ldh, n, h, ldh, work, -1);
    // ==== Optimal workspace = MAX(DLAQR5, DLAQR2) ====
    lwkopt = Max(3 * nsr / 2, static_cast<blasint>(WORK(1)));
    // ==== Quick return in case of workspace query. ====
    if (lwork == -1) {
      WORK(1) = WorkSize<R>(lwkopt);
      return info;
    }
    // ==== DLAHQR/DLAQR0 crossover point ====
    nmin = Ilaenv<R>(12, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nmin = Max<blasint>(ntiny, nmin);
    // ==== Nibble crossover point ====
    nibble = Ilaenv<R>(14, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nibble = Max<blasint>(0, nibble);
    // ==== Accumulate reflections during ttswp?  Use block
    // .    2-by-2 structure during matrix-matrix multiply? ====
    kacc22 = Ilaenv<R>(16, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    kacc22 = Max<blasint>(0, kacc22);
    kacc22 = Min<blasint>(2, kacc22);
    // ==== NWMAX = the largest possible deflation window for
    // .    which there is sufficient workspace. ====
    nwmax = Min<blasint>((n - 1) / 3, lwork / 2);
    nw = nwmax;
    // ==== NSMAX = the Largest number of simultaneous shifts
    // .    for which there is sufficient workspace. ====
    nsmax = Min<blasint>((n - 3) / 6, 2 * lwork / 3);
    nsmax = nsmax - nsmax % 2;
    // ==== NDFL: an iteration count restarted at deflation. ====
    ndfl = 1;
    // ==== ITMAX = iteration limit ====
    itmax = Max<blasint>(30, 2 * kexsh) * Max<blasint>(10, (ihi - ilo + 1));
    // ==== Last row and column in the active block ====
    kbot = ihi;
    // ==== Main Loop ====
    for (it = 1; it <= itmax; ++it) {
      // ==== Done when KBOT falls below ILO ====
      if (kbot < ilo) goto L90;
      // ==== Locate active block ====
      for (k = kbot; k >= ilo + 1; --k) {
        if (H(k, k - 1) == zero) goto L20;
      }
      k = ilo;
    L20:
      ktop = k;
      // ==== Select deflation window size:
      // .    Typical Case:
      // .      If possible and advisable, nibble the entire
      // .      active block.  If not, use size MIN(NWR,NWMAX)
      // .      or MIN(NWR+1,NWMAX) depending upon which has
      // .      the smaller corresponding subdiagonal entry
      // .      (a heuristic).
      // .
      // .    Exceptional Case:
      // .      If there have been no deflations in KEXNW or
      // .      more iterations, then vary the deflation window
      // .      size.   At first, because, larger windows are,
      // .      in general, more powerful than smaller ones,
      // .      rapidly increase the window to the maximum possible.
      // .      Then, gradually reduce the window size. ====
      nh = kbot - ktop + 1;
      nwupbd = Min<blasint>(nh, nwmax);
      if (ndfl < kexnw) {
        nw = Min<blasint>(nwupbd, nwr);
      } else {
        nw = Min<blasint>(nwupbd, 2 * nw);
      }
      if (nw < nwmax) {
        if (nw >= nh - 1) {
          nw = nh;
        } else {
          kwtop = kbot - nw + 1;
          if (std::abs(H(kwtop, kwtop - 1)) > std::abs(H(kwtop - 1, kwtop - 2)))
            nw = nw + 1;
        }
      }
      if (ndfl < kexnw) {
        ndec = -1;
      } else if (ndec >= 0 || nw >= nwupbd) {
        ndec = ndec + 1;
        if (nw - ndec < 2) ndec = 0;
        nw = nw - ndec;
      }
      // ==== Aggressive early deflation:
      // .    split workspace under the subdiagonal into
      // .      - an nw-by-nw work array V in the lower
      // .        left-hand-corner,
      // .      - an NW-by-at-least-NW-but-more-is-better
      // .        (NW-by-NHO) horizontal work array along
      // .        the bottom edge,
      // .      - an at-least-NW-but-more-is-better (NHV-by-NW)
      // .        vertical work array along the left-hand-edge.
      // .        ====
      kv = n - nw + 1;
      kt = nw + 1;
      nho = (n - nw - 1) - kt + 1;
      kwv = nw + 2;
      nve = (n - nw) - kwv + 1;
      // ==== Aggressive early deflation ====
      Laqr2(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, &ls,
            &ld, wr, wi, H.ptr(kv, 1), ldh, nho, H.ptr(kv, kt), ldh, nve,
            H.ptr(kwv, 1), ldh, work, lwork);
      // ==== Adjust KBOT accounting for new deflations. ====
      kbot = kbot - ld;
      // ==== KS points to the shifts. ====
      ks = kbot - ls + 1;
      // ==== Skip an expensive QR sweep if there is a (partly
      // .    heuristic) reason to expect that many eigenvalues
      // .    will deflate without it.  Here, the QR sweep is
      // .    skipped if many eigenvalues have just been deflated
      // .    or if the remaining active block is small.
      if ((ld == 0) || ((100 * ld <= nw * nibble) &&
                        (kbot - ktop + 1 > Min<blasint>(nmin, nwmax)))) {
        // ==== NS = nominal number of simultaneous shifts.
        // .    This may be lowered (slightly) if DLAQR2
        // .    did not provide that many shifts. ====
        ns = Min<blasint>(nsmax, nsr, Max<blasint>(2, kbot - ktop));
        ns = ns - ns % 2;
        // ==== If there have been no deflations
        // .    in a multiple of KEXSH iterations,
        // .    then try exceptional shifts.
        // .    Otherwise use shifts provided by
        // .    DLAQR2 above or from the eigenvalues
        // .    of a trailing principal submatrix. ====
        if (ndfl % kexsh == 0) {
          ks = kbot - ns + 1;
          for (i = kbot; i >= Max<blasint>(ks + 1, ktop + 2); i -= 2) {
            ss = std::abs(H(i, i - 1)) + std::abs(H(i - 1, i - 2));
            aa = wilk1 * ss + H(i, i);
            bb = ss;
            cc = wilk2 * ss;
            dd = aa;
            Lanv2(&aa, &bb, &cc, &dd, WR.ptr(i - 1), WI.ptr(i - 1), WR.ptr(i),
                  WI.ptr(i), &cs, &sn);
          }
          if (ks == ktop) {
            WR(ks + 1) = H(ks + 1, ks + 1);
            WI(ks + 1) = zero;
            WR(ks) = WR(ks + 1);
            WI(ks) = WI(ks + 1);
          }
        } else {
          // ==== Got NS/2 or fewer shifts? Use DLAHQR
          // .    on a trailing principal submatrix to
          // .    get more. (Since NS.LE.NSMAX.LE.(N-3)/6,
          // .    there is enough space below the subdiagonal
          // .    to fit an NS-by-NS scratch array.) ====
          if (kbot - ks + 1 <= ns / 2) {
            ks = kbot - ns + 1;
            kt = n - ns + 1;
            Lacpy('A', ns, ns, H.ptr(ks, ks), ldh, H.ptr(kt, 1), ldh);
            inf = Lahqr(false, false, ns, 1, ns, H.ptr(kt, 1), ldh, WR.ptr(ks),
                        WI.ptr(ks), 1, 1, zdum, 1);
            ks = ks + inf;
            // ==== In case of a rare QR failure use
            // .    eigenvalues of the trailing 2-by-2
            // .    principal submatrix.  ====
            if (ks >= kbot) {
              aa = H(kbot - 1, kbot - 1);
              cc = H(kbot, kbot - 1);
              bb = H(kbot - 1, kbot);
              dd = H(kbot, kbot);
              Lanv2(&aa, &bb, &cc, &dd, WR.ptr(kbot - 1), WI.ptr(kbot - 1),
                    WR.ptr(kbot), WI.ptr(kbot), &cs, &sn);
              ks = kbot - 1;
            }
          }
          if (kbot - ks + 1 > ns) {
            // ==== Sort the shifts (Helps a little)
            // .    Bubble sort keeps complex conjugate
            // .    pairs together. ====
            sorted = false;
            for (k = kbot; k >= ks + 1; --k) {
              if (sorted) goto L60;
              sorted = true;
              for (i = ks; i <= k - 1; ++i) {
                if (std::abs(WR(i)) + std::abs(WI(i)) <
                    std::abs(WR(i + 1)) + std::abs(WI(i + 1))) {
                  sorted = false;
                  swap = WR(i);
                  WR(i) = WR(i + 1);
                  WR(i + 1) = swap;
                  swap = WI(i);
                  WI(i) = WI(i + 1);
                  WI(i + 1) = swap;
                }
              }
            }
          L60:;
          }
          // ==== Shuffle shifts into pairs of real shifts
          // .    and pairs of complex conjugate shifts
          // .    assuming complex conjugate shifts are
          // .    already adjacent to one another. (Yes,
          // .    they are.)  ====
          for (i = kbot; i >= ks + 2; i -= 2) {
            if (WI(i) != -WI(i - 1)) {
              swap = WR(i);
              WR(i) = WR(i - 1);
              WR(i - 1) = WR(i - 2);
              WR(i - 2) = swap;
              swap = WI(i);
              WI(i) = WI(i - 1);
              WI(i - 1) = WI(i - 2);
              WI(i - 2) = swap;
            }
          }
        }
        // ==== If there are only two shifts and both are
        // .    real, then use only one.  ====
        if (kbot - ks + 1 == 2) {
          if (WI(kbot) == zero) {
            if (std::abs(WR(kbot) - H(kbot, kbot)) <
                std::abs(WR(kbot - 1) - H(kbot, kbot))) {
              WR(kbot - 1) = WR(kbot);
            } else {
              WR(kbot) = WR(kbot - 1);
            }
          }
        }
        // ==== Use up to NS of the the smallest magnitude
        // .    shifts.  If there aren't NS shifts available,
        // .    then use them all, possibly dropping one to
        // .    make the number of shifts even. ====
        ns = Min<blasint>(ns, kbot - ks + 1);
        ns = ns - ns % 2;
        ks = kbot - ns + 1;
        // ==== Small-bulge multi-shift QR sweep:
        // .    split workspace under the subdiagonal into
        // .    - a KDU-by-KDU work array U in the lower
        // .      left-hand-corner,
        // .    - a KDU-by-at-least-KDU-but-more-is-better
        // .      (KDU-by-NHo) horizontal work array WH along
        // .      the bottom edge,
        // .    - and an at-least-KDU-but-more-is-better-by-KDU
        // .      (NVE-by-KDU) vertical work WV arrow along
        // .      the left-hand-edge. ====
        kdu = 2 * ns;
        ku = n - kdu + 1;
        kwh = kdu + 1;
        nho = (n - kdu + 1 - 4) - (kdu + 1) + 1;
        kwv = kdu + 4;
        nve = n - kdu - kwv + 1;
        // ==== Small-bulge multi-shift QR sweep ====
        Laqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, WR.ptr(ks), WI.ptr(ks),
              h, ldh, iloz, ihiz, z, ldz, work, 3, H.ptr(ku, 1), ldh, nve,
              H.ptr(kwv, 1), ldh, nho, H.ptr(ku, kwh), ldh);
      }
      // ==== Note progress (or the lack of it). ====
      if (ld > 0) {
        ndfl = 1;
      } else {
        ndfl = ndfl + 1;
      }
      // ==== End of main loop ====
    }
    // ==== Iteration limit exceeded.  Set INFO to show where
    // .    the problem occurred and exit. ====
    info = kbot;
  L90:;
  }
  // ==== Return the optimal value of LWORK. ====
  WORK(1) = WorkSize<R>(lwkopt);
  return info;
}

// xLAQR4 (complex T): as the real routine, with the eigenvalues in W.
template <typename T>
blasint Laqr4(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, blasint iloz, blasint ihiz, T* z, blasint ldz,
              T* work, blasint lwork) {
  using R = RealT<T>;

  const blasint ntiny = 15;
  const blasint kexnw = 5;
  const blasint kexsh = 6;
  const R wilk1 = R(0.75);
  const T zero = T(0);
  const T one = T(1);
  const R two = R(2);
  const Matrix<T> H(h, ldh);
  char jbcmpz[3] = {' ', ' ', '\0'};
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  T zdum[1];
  blasint i, inf, it, itmax, k, kacc22, kbot, kdu, ks, kt, ktop, ku, kv, kwh,
      kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh, nho, nibble, nmin, ns, nsmax,
      nsr, nve, nw, nwmax, nwr, nwupbd;
  bool sorted;
  R s;
  T aa, bb, cc, dd, det, rtdisc, swap, tr2;
  blasint info = 0;
  // ==== Quick return for N = 0: nothing to do. ====
  if (n == 0) {
    WORK(1) = one;
    return info;
  }
  if (n <= ntiny) {
    // ==== Tiny matrices must use ZLAHQR. ====
    lwkopt = 1;
    if (lwork != -1)
      info = Lahqr(wantt, wantz, n, ilo, ihi, h, ldh, w, iloz, ihiz, z, ldz);
  } else {
    // ==== Use small bulge multi-shift QR with aggressive early
    // .    deflation on larger-than-tiny matrices. ====
    // ==== Hope for the best. ====
    info = 0;
    // ==== Set up job flags for ILAENV. ====
    if (wantt) {
      jbcmpz[0] = 'S';
    } else {
      jbcmpz[0] = 'E';
    }
    if (wantz) {
      jbcmpz[1] = 'V';
    } else {
      jbcmpz[1] = 'N';
    }
    // ==== NWR = recommended deflation window size.  At this
    // .    point,  N .GT. NTINY = 15, so there is enough
    // .    subdiagonal workspace for NWR.GE.2 as required.
    // .    (In fact, there is enough subdiagonal space for
    // .    NWR.GE.4.) ====
    nwr = Ilaenv<T>(13, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nwr = Max<blasint>(2, nwr);
    nwr = Min<blasint>(ihi - ilo + 1, (n - 1) / 3, nwr);
    // ==== NSR = recommended number of simultaneous shifts.
    // .    At this point N .GT. NTINY = 15, so there is at
    // .    enough subdiagonal workspace for NSR to be even
    // .    and greater than or equal to two as required. ====
    nsr = Ilaenv<T>(15, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nsr = Min<blasint>(nsr, (n - 3) / 6, ihi - ilo);
    nsr = Max<blasint>(2, nsr - nsr % 2);
    // ==== Estimate optimal workspace ====
    // ==== Workspace query call to ZLAQR2 ====
    Laqr2(wantt, wantz, n, ilo, ihi, nwr + 1, h, ldh, iloz, ihiz, z, ldz, &ls,
          &ld, w, h, ldh, n, h, ldh, n, h, ldh, work, -1);
    // ==== Optimal workspace = MAX(ZLAQR5, ZLAQR2) ====
    lwkopt = Max(3 * nsr / 2, static_cast<blasint>(std::real(WORK(1))));
    // ==== Quick return in case of workspace query. ====
    if (lwork == -1) {
      WORK(1) = WorkSize<T>(lwkopt);
      return info;
    }
    // ==== ZLAHQR/ZLAQR0 crossover point ====
    nmin = Ilaenv<T>(12, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nmin = Max<blasint>(ntiny, nmin);
    // ==== Nibble crossover point ====
    nibble = Ilaenv<T>(14, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    nibble = Max<blasint>(0, nibble);
    // ==== Accumulate reflections during ttswp?  Use block
    // .    2-by-2 structure during matrix-matrix multiply? ====
    kacc22 = Ilaenv<T>(16, "LAQR4", jbcmpz, n, ilo, ihi, lwork);
    kacc22 = Max<blasint>(0, kacc22);
    kacc22 = Min<blasint>(2, kacc22);
    // ==== NWMAX = the largest possible deflation window for
    // .    which there is sufficient workspace. ====
    nwmax = Min<blasint>((n - 1) / 3, lwork / 2);
    nw = nwmax;
    // ==== NSMAX = the Largest number of simultaneous shifts
    // .    for which there is sufficient workspace. ====
    nsmax = Min<blasint>((n - 3) / 6, 2 * lwork / 3);
    nsmax = nsmax - nsmax % 2;
    // ==== NDFL: an iteration count restarted at deflation. ====
    ndfl = 1;
    // ==== ITMAX = iteration limit ====
    itmax = Max<blasint>(30, 2 * kexsh) * Max<blasint>(10, (ihi - ilo + 1));
    // ==== Last row and column in the active block ====
    kbot = ihi;
    // ==== Main Loop ====
    for (it = 1; it <= itmax; ++it) {
      // ==== Done when KBOT falls below ILO ====
      if (kbot < ilo) goto L80;
      // ==== Locate active block ====
      for (k = kbot; k >= ilo + 1; --k) {
        if (H(k, k - 1) == zero) goto L20;
      }
      k = ilo;
    L20:
      ktop = k;
      // ==== Select deflation window size:
      // .    Typical Case:
      // .      If possible and advisable, nibble the entire
      // .      active block.  If not, use size MIN(NWR,NWMAX)
      // .      or MIN(NWR+1,NWMAX) depending upon which has
      // .      the smaller corresponding subdiagonal entry
      // .      (a heuristic).
      // .
      // .    Exceptional Case:
      // .      If there have been no deflations in KEXNW or
      // .      more iterations, then vary the deflation window
      // .      size.   At first, because, larger windows are,
      // .      in general, more powerful than smaller ones,
      // .      rapidly increase the window to the maximum possible.
      // .      Then, gradually reduce the window size. ====
      nh = kbot - ktop + 1;
      nwupbd = Min<blasint>(nh, nwmax);
      if (ndfl < kexnw) {
        nw = Min<blasint>(nwupbd, nwr);
      } else {
        nw = Min<blasint>(nwupbd, 2 * nw);
      }
      if (nw < nwmax) {
        if (nw >= nh - 1) {
          nw = nh;
        } else {
          kwtop = kbot - nw + 1;
          if (Cabs1(H(kwtop, kwtop - 1)) > Cabs1(H(kwtop - 1, kwtop - 2)))
            nw = nw + 1;
        }
      }
      if (ndfl < kexnw) {
        ndec = -1;
      } else if (ndec >= 0 || nw >= nwupbd) {
        ndec = ndec + 1;
        if (nw - ndec < 2) ndec = 0;
        nw = nw - ndec;
      }
      // ==== Aggressive early deflation:
      // .    split workspace under the subdiagonal into
      // .      - an nw-by-nw work array V in the lower
      // .        left-hand-corner,
      // .      - an NW-by-at-least-NW-but-more-is-better
      // .        (NW-by-NHO) horizontal work array along
      // .        the bottom edge,
      // .      - an at-least-NW-but-more-is-better (NHV-by-NW)
      // .        vertical work array along the left-hand-edge.
      // .        ====
      kv = n - nw + 1;
      kt = nw + 1;
      nho = (n - nw - 1) - kt + 1;
      kwv = nw + 2;
      nve = (n - nw) - kwv + 1;
      // ==== Aggressive early deflation ====
      Laqr2(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, &ls,
            &ld, w, H.ptr(kv, 1), ldh, nho, H.ptr(kv, kt), ldh, nve,
            H.ptr(kwv, 1), ldh, work, lwork);
      // ==== Adjust KBOT accounting for new deflations. ====
      kbot = kbot - ld;
      // ==== KS points to the shifts. ====
      ks = kbot - ls + 1;
      // ==== Skip an expensive QR sweep if there is a (partly
      // .    heuristic) reason to expect that many eigenvalues
      // .    will deflate without it.  Here, the QR sweep is
      // .    skipped if many eigenvalues have just been deflated
      // .    or if the remaining active block is small.
      if ((ld == 0) || ((100 * ld <= nw * nibble) &&
                        (kbot - ktop + 1 > Min<blasint>(nmin, nwmax)))) {
        // ==== NS = nominal number of simultaneous shifts.
        // .    This may be lowered (slightly) if ZLAQR2
        // .    did not provide that many shifts. ====
        ns = Min<blasint>(nsmax, nsr, Max<blasint>(2, kbot - ktop));
        ns = ns - ns % 2;
        // ==== If there have been no deflations
        // .    in a multiple of KEXSH iterations,
        // .    then try exceptional shifts.
        // .    Otherwise use shifts provided by
        // .    ZLAQR2 above or from the eigenvalues
        // .    of a trailing principal submatrix. ====
        if (ndfl % kexsh == 0) {
          ks = kbot - ns + 1;
          for (i = kbot; i >= ks + 1; i -= 2) {
            W(i) = H(i, i) + wilk1 * Cabs1(H(i, i - 1));
            W(i - 1) = W(i);
          }
        } else {
          // ==== Got NS/2 or fewer shifts? Use ZLAHQR
          // .    on a trailing principal submatrix to
          // .    get more. (Since NS.LE.NSMAX.LE.(N-3)/6,
          // .    there is enough space below the subdiagonal
          // .    to fit an NS-by-NS scratch array.) ====
          if (kbot - ks + 1 <= ns / 2) {
            ks = kbot - ns + 1;
            kt = n - ns + 1;
            Lacpy('A', ns, ns, H.ptr(ks, ks), ldh, H.ptr(kt, 1), ldh);
            inf = Lahqr(false, false, ns, 1, ns, H.ptr(kt, 1), ldh, W.ptr(ks),
                        1, 1, zdum, 1);
            ks = ks + inf;
            // ==== In case of a rare QR failure use
            // .    eigenvalues of the trailing 2-by-2
            // .    principal submatrix.  Scale to avoid
            // .    overflows, underflows and subnormals.
            // .    (The scale factor S can not be zero,
            // .    because H(KBOT,KBOT-1) is nonzero.) ====
            if (ks >= kbot) {
              s = Cabs1(H(kbot - 1, kbot - 1)) + Cabs1(H(kbot, kbot - 1)) +
                  Cabs1(H(kbot - 1, kbot)) + Cabs1(H(kbot, kbot));
              aa = H(kbot - 1, kbot - 1) / s;
              cc = H(kbot, kbot - 1) / s;
              bb = H(kbot - 1, kbot) / s;
              dd = H(kbot, kbot) / s;
              tr2 = (aa + dd) / two;
              det = (aa - tr2) * (dd - tr2) - bb * cc;
              rtdisc = std::sqrt(-det);
              W(kbot - 1) = (tr2 + rtdisc) * s;
              W(kbot) = (tr2 - rtdisc) * s;
              ks = kbot - 1;
            }
          }
          if (kbot - ks + 1 > ns) {
            // ==== Sort the shifts (Helps a little) ====
            sorted = false;
            for (k = kbot; k >= ks + 1; --k) {
              if (sorted) goto L60;
              sorted = true;
              for (i = ks; i <= k - 1; ++i) {
                if (Cabs1(W(i)) < Cabs1(W(i + 1))) {
                  sorted = false;
                  swap = W(i);
                  W(i) = W(i + 1);
                  W(i + 1) = swap;
                }
              }
            }
          L60:;
          }
        }
        // ==== If there are only two shifts, then use
        // .    only one.  ====
        if (kbot - ks + 1 == 2) {
          if (Cabs1(W(kbot) - H(kbot, kbot)) <
              Cabs1(W(kbot - 1) - H(kbot, kbot))) {
            W(kbot - 1) = W(kbot);
          } else {
            W(kbot) = W(kbot - 1);
          }
        }
        // ==== Use up to NS of the the smallest magnitude
        // .    shifts.  If there aren't NS shifts available,
        // .    then use them all, possibly dropping one to
        // .    make the number of shifts even. ====
        ns = Min<blasint>(ns, kbot - ks + 1);
        ns = ns - ns % 2;
        ks = kbot - ns + 1;
        // ==== Small-bulge multi-shift QR sweep:
        // .    split workspace under the subdiagonal into
        // .    - a KDU-by-KDU work array U in the lower
        // .      left-hand-corner,
        // .    - a KDU-by-at-least-KDU-but-more-is-better
        // .      (KDU-by-NHo) horizontal work array WH along
        // .      the bottom edge,
        // .    - and an at-least-KDU-but-more-is-better-by-KDU
        // .      (NVE-by-KDU) vertical work WV arrow along
        // .      the left-hand-edge. ====
        kdu = 2 * ns;
        ku = n - kdu + 1;
        kwh = kdu + 1;
        nho = (n - kdu + 1 - 4) - (kdu + 1) + 1;
        kwv = kdu + 4;
        nve = n - kdu - kwv + 1;
        // ==== Small-bulge multi-shift QR sweep ====
        Laqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, W.ptr(ks), h, ldh, iloz,
              ihiz, z, ldz, work, 3, H.ptr(ku, 1), ldh, nve, H.ptr(kwv, 1), ldh,
              nho, H.ptr(ku, kwh), ldh);
      }
      // ==== Note progress (or the lack of it). ====
      if (ld > 0) {
        ndfl = 1;
      } else {
        ndfl = ndfl + 1;
      }
      // ==== End of main loop ====
    }
    // ==== Iteration limit exceeded.  Set INFO to show where
    // .    the problem occurred and exit. ====
    info = kbot;
  L80:;
  }
  // ==== Return the optimal value of LWORK. ====
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xLAQR0 (real T): the eigenvalues WR + i*WI of the upper Hessenberg matrix
// H in rows and columns ILO:IHI, and optionally the Schur form T in H
// (WANTT) and the Schur vectors applied to rows ILOZ:IHIZ of Z (WANTZ), by
// the small-bulge multishift QR algorithm with aggressive early deflation.
// WORK has LWORK >= max(1, N) elements (LWORK = -1: workspace query).
//
// Returns INFO: 0, or i > 0 if the algorithm failed to compute all the
// eigenvalues; eigenvalues i+1:IHI are then in WR and WI.
template <typename R>
blasint Laqr0(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, blasint iloz, blasint ihiz, R* z,
              blasint ldz, R* work, blasint lwork) {
  const blasint ntiny = 15;
  const blasint kexnw = 5;
  const blasint kexsh = 6;
  const R wilk1 = R(0.75);
  const R wilk2 = -R(0.4375);
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> H(h, ldh);
  char jbcmpz[3] = {' ', ' ', '\0'};
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  const Matrix<R> Z(z, ldz);
  R zdum[1];
  blasint i, inf, it, itmax, k, kacc22, kbot, kdu, ks, kt, ktop, ku, kv, kwh,
      kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh, nho, nibble, nmin, ns, nsmax,
      nsr, nve, nw, nwmax, nwr, nwupbd;
  bool sorted;
  R aa, bb, cc, cs, dd, sn, ss, swap;
  blasint info = 0;
  // ==== Quick return for N = 0: nothing to do. ====
  if (n == 0) {
    WORK(1) = one;
    return info;
  }
  if (n <= ntiny) {
    // ==== Tiny matrices must use DLAHQR. ====
    lwkopt = 1;
    if (lwork != -1)
      info =
          Lahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz);
  } else {
    // ==== Use small bulge multi-shift QR with aggressive early
    // .    deflation on larger-than-tiny matrices. ====
    // ==== Hope for the best. ====
    info = 0;
    // ==== Set up job flags for ILAENV. ====
    if (wantt) {
      jbcmpz[0] = 'S';
    } else {
      jbcmpz[0] = 'E';
    }
    if (wantz) {
      jbcmpz[1] = 'V';
    } else {
      jbcmpz[1] = 'N';
    }
    // ==== NWR = recommended deflation window size.  At this
    // .    point,  N .GT. NTINY = 15, so there is enough
    // .    subdiagonal workspace for NWR.GE.2 as required.
    // .    (In fact, there is enough subdiagonal space for
    // .    NWR.GE.4.) ====
    nwr = Ilaenv<R>(13, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nwr = Max<blasint>(2, nwr);
    nwr = Min<blasint>(ihi - ilo + 1, (n - 1) / 3, nwr);
    // ==== NSR = recommended number of simultaneous shifts.
    // .    At this point N .GT. NTINY = 15, so there is at
    // .    enough subdiagonal workspace for NSR to be even
    // .    and greater than or equal to two as required. ====
    nsr = Ilaenv<R>(15, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nsr = Min<blasint>(nsr, (n - 3) / 6, ihi - ilo);
    nsr = Max<blasint>(2, nsr - nsr % 2);
    // ==== Estimate optimal workspace ====
    // ==== Workspace query call to DLAQR3 ====
    Laqr3(wantt, wantz, n, ilo, ihi, nwr + 1, h, ldh, iloz, ihiz, z, ldz, &ls,
          &ld, wr, wi, h, ldh, n, h, ldh, n, h, ldh, work, -1);
    // ==== Optimal workspace = MAX(DLAQR5, DLAQR3) ====
    lwkopt = Max(3 * nsr / 2, static_cast<blasint>(WORK(1)));
    // ==== Quick return in case of workspace query. ====
    if (lwork == -1) {
      WORK(1) = WorkSize<R>(lwkopt);
      return info;
    }
    // ==== DLAHQR/DLAQR0 crossover point ====
    nmin = Ilaenv<R>(12, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nmin = Max<blasint>(ntiny, nmin);
    // ==== Nibble crossover point ====
    nibble = Ilaenv<R>(14, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nibble = Max<blasint>(0, nibble);
    // ==== Accumulate reflections during ttswp?  Use block
    // .    2-by-2 structure during matrix-matrix multiply? ====
    kacc22 = Ilaenv<R>(16, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    kacc22 = Max<blasint>(0, kacc22);
    kacc22 = Min<blasint>(2, kacc22);
    // ==== NWMAX = the largest possible deflation window for
    // .    which there is sufficient workspace. ====
    nwmax = Min<blasint>((n - 1) / 3, lwork / 2);
    nw = nwmax;
    // ==== NSMAX = the Largest number of simultaneous shifts
    // .    for which there is sufficient workspace. ====
    nsmax = Min<blasint>((n - 3) / 6, 2 * lwork / 3);
    nsmax = nsmax - nsmax % 2;
    // ==== NDFL: an iteration count restarted at deflation. ====
    ndfl = 1;
    // ==== ITMAX = iteration limit ====
    itmax = Max<blasint>(30, 2 * kexsh) * Max<blasint>(10, (ihi - ilo + 1));
    // ==== Last row and column in the active block ====
    kbot = ihi;
    // ==== Main Loop ====
    for (it = 1; it <= itmax; ++it) {
      // ==== Done when KBOT falls below ILO ====
      if (kbot < ilo) goto L90;
      // ==== Locate active block ====
      for (k = kbot; k >= ilo + 1; --k) {
        if (H(k, k - 1) == zero) goto L20;
      }
      k = ilo;
    L20:
      ktop = k;
      // ==== Select deflation window size:
      // .    Typical Case:
      // .      If possible and advisable, nibble the entire
      // .      active block.  If not, use size MIN(NWR,NWMAX)
      // .      or MIN(NWR+1,NWMAX) depending upon which has
      // .      the smaller corresponding subdiagonal entry
      // .      (a heuristic).
      // .
      // .    Exceptional Case:
      // .      If there have been no deflations in KEXNW or
      // .      more iterations, then vary the deflation window
      // .      size.   At first, because, larger windows are,
      // .      in general, more powerful than smaller ones,
      // .      rapidly increase the window to the maximum possible.
      // .      Then, gradually reduce the window size. ====
      nh = kbot - ktop + 1;
      nwupbd = Min<blasint>(nh, nwmax);
      if (ndfl < kexnw) {
        nw = Min<blasint>(nwupbd, nwr);
      } else {
        nw = Min<blasint>(nwupbd, 2 * nw);
      }
      if (nw < nwmax) {
        if (nw >= nh - 1) {
          nw = nh;
        } else {
          kwtop = kbot - nw + 1;
          if (std::abs(H(kwtop, kwtop - 1)) > std::abs(H(kwtop - 1, kwtop - 2)))
            nw = nw + 1;
        }
      }
      if (ndfl < kexnw) {
        ndec = -1;
      } else if (ndec >= 0 || nw >= nwupbd) {
        ndec = ndec + 1;
        if (nw - ndec < 2) ndec = 0;
        nw = nw - ndec;
      }
      // ==== Aggressive early deflation:
      // .    split workspace under the subdiagonal into
      // .      - an nw-by-nw work array V in the lower
      // .        left-hand-corner,
      // .      - an NW-by-at-least-NW-but-more-is-better
      // .        (NW-by-NHO) horizontal work array along
      // .        the bottom edge,
      // .      - an at-least-NW-but-more-is-better (NHV-by-NW)
      // .        vertical work array along the left-hand-edge.
      // .        ====
      kv = n - nw + 1;
      kt = nw + 1;
      nho = (n - nw - 1) - kt + 1;
      kwv = nw + 2;
      nve = (n - nw) - kwv + 1;
      // ==== Aggressive early deflation ====
      Laqr3(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, &ls,
            &ld, wr, wi, H.ptr(kv, 1), ldh, nho, H.ptr(kv, kt), ldh, nve,
            H.ptr(kwv, 1), ldh, work, lwork);
      // ==== Adjust KBOT accounting for new deflations. ====
      kbot = kbot - ld;
      // ==== KS points to the shifts. ====
      ks = kbot - ls + 1;
      // ==== Skip an expensive QR sweep if there is a (partly
      // .    heuristic) reason to expect that many eigenvalues
      // .    will deflate without it.  Here, the QR sweep is
      // .    skipped if many eigenvalues have just been deflated
      // .    or if the remaining active block is small.
      if ((ld == 0) || ((100 * ld <= nw * nibble) &&
                        (kbot - ktop + 1 > Min<blasint>(nmin, nwmax)))) {
        // ==== NS = nominal number of simultaneous shifts.
        // .    This may be lowered (slightly) if DLAQR3
        // .    did not provide that many shifts. ====
        ns = Min<blasint>(nsmax, nsr, Max<blasint>(2, kbot - ktop));
        ns = ns - ns % 2;
        // ==== If there have been no deflations
        // .    in a multiple of KEXSH iterations,
        // .    then try exceptional shifts.
        // .    Otherwise use shifts provided by
        // .    DLAQR3 above or from the eigenvalues
        // .    of a trailing principal submatrix. ====
        if (ndfl % kexsh == 0) {
          ks = kbot - ns + 1;
          for (i = kbot; i >= Max<blasint>(ks + 1, ktop + 2); i -= 2) {
            ss = std::abs(H(i, i - 1)) + std::abs(H(i - 1, i - 2));
            aa = wilk1 * ss + H(i, i);
            bb = ss;
            cc = wilk2 * ss;
            dd = aa;
            Lanv2(&aa, &bb, &cc, &dd, WR.ptr(i - 1), WI.ptr(i - 1), WR.ptr(i),
                  WI.ptr(i), &cs, &sn);
          }
          if (ks == ktop) {
            WR(ks + 1) = H(ks + 1, ks + 1);
            WI(ks + 1) = zero;
            WR(ks) = WR(ks + 1);
            WI(ks) = WI(ks + 1);
          }
        } else {
          // ==== Got NS/2 or fewer shifts? Use DLAQR4 or
          // .    DLAHQR on a trailing principal submatrix to
          // .    get more. (Since NS.LE.NSMAX.LE.(N-3)/6,
          // .    there is enough space below the subdiagonal
          // .    to fit an NS-by-NS scratch array.) ====
          if (kbot - ks + 1 <= ns / 2) {
            ks = kbot - ns + 1;
            kt = n - ns + 1;
            Lacpy('A', ns, ns, H.ptr(ks, ks), ldh, H.ptr(kt, 1), ldh);
            if (ns > nmin) {
              inf = Laqr4(false, false, ns, 1, ns, H.ptr(kt, 1), ldh,
                          WR.ptr(ks), WI.ptr(ks), 1, 1, zdum, 1, work, lwork);
            } else {
              inf = Lahqr(false, false, ns, 1, ns, H.ptr(kt, 1), ldh,
                          WR.ptr(ks), WI.ptr(ks), 1, 1, zdum, 1);
            }
            ks = ks + inf;
            // ==== In case of a rare QR failure use
            // .    eigenvalues of the trailing 2-by-2
            // .    principal submatrix.  ====
            if (ks >= kbot) {
              aa = H(kbot - 1, kbot - 1);
              cc = H(kbot, kbot - 1);
              bb = H(kbot - 1, kbot);
              dd = H(kbot, kbot);
              Lanv2(&aa, &bb, &cc, &dd, WR.ptr(kbot - 1), WI.ptr(kbot - 1),
                    WR.ptr(kbot), WI.ptr(kbot), &cs, &sn);
              ks = kbot - 1;
            }
          }
          if (kbot - ks + 1 > ns) {
            // ==== Sort the shifts (Helps a little)
            // .    Bubble sort keeps complex conjugate
            // .    pairs together. ====
            sorted = false;
            for (k = kbot; k >= ks + 1; --k) {
              if (sorted) goto L60;
              sorted = true;
              for (i = ks; i <= k - 1; ++i) {
                if (std::abs(WR(i)) + std::abs(WI(i)) <
                    std::abs(WR(i + 1)) + std::abs(WI(i + 1))) {
                  sorted = false;
                  swap = WR(i);
                  WR(i) = WR(i + 1);
                  WR(i + 1) = swap;
                  swap = WI(i);
                  WI(i) = WI(i + 1);
                  WI(i + 1) = swap;
                }
              }
            }
          L60:;
          }
          // ==== Shuffle shifts into pairs of real shifts
          // .    and pairs of complex conjugate shifts
          // .    assuming complex conjugate shifts are
          // .    already adjacent to one another. (Yes,
          // .    they are.)  ====
          for (i = kbot; i >= ks + 2; i -= 2) {
            if (WI(i) != -WI(i - 1)) {
              swap = WR(i);
              WR(i) = WR(i - 1);
              WR(i - 1) = WR(i - 2);
              WR(i - 2) = swap;
              swap = WI(i);
              WI(i) = WI(i - 1);
              WI(i - 1) = WI(i - 2);
              WI(i - 2) = swap;
            }
          }
        }
        // ==== If there are only two shifts and both are
        // .    real, then use only one.  ====
        if (kbot - ks + 1 == 2) {
          if (WI(kbot) == zero) {
            if (std::abs(WR(kbot) - H(kbot, kbot)) <
                std::abs(WR(kbot - 1) - H(kbot, kbot))) {
              WR(kbot - 1) = WR(kbot);
            } else {
              WR(kbot) = WR(kbot - 1);
            }
          }
        }
        // ==== Use up to NS of the the smallest magnitude
        // .    shifts.  If there aren't NS shifts available,
        // .    then use them all, possibly dropping one to
        // .    make the number of shifts even. ====
        ns = Min<blasint>(ns, kbot - ks + 1);
        ns = ns - ns % 2;
        ks = kbot - ns + 1;
        // ==== Small-bulge multi-shift QR sweep:
        // .    split workspace under the subdiagonal into
        // .    - a KDU-by-KDU work array U in the lower
        // .      left-hand-corner,
        // .    - a KDU-by-at-least-KDU-but-more-is-better
        // .      (KDU-by-NHo) horizontal work array WH along
        // .      the bottom edge,
        // .    - and an at-least-KDU-but-more-is-better-by-KDU
        // .      (NVE-by-KDU) vertical work WV arrow along
        // .      the left-hand-edge. ====
        kdu = 2 * ns;
        ku = n - kdu + 1;
        kwh = kdu + 1;
        nho = (n - kdu + 1 - 4) - (kdu + 1) + 1;
        kwv = kdu + 4;
        nve = n - kdu - kwv + 1;
        // ==== Small-bulge multi-shift QR sweep ====
        Laqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, WR.ptr(ks), WI.ptr(ks),
              h, ldh, iloz, ihiz, z, ldz, work, 3, H.ptr(ku, 1), ldh, nve,
              H.ptr(kwv, 1), ldh, nho, H.ptr(ku, kwh), ldh);
      }
      // ==== Note progress (or the lack of it). ====
      if (ld > 0) {
        ndfl = 1;
      } else {
        ndfl = ndfl + 1;
      }
      // ==== End of main loop ====
    }
    // ==== Iteration limit exceeded.  Set INFO to show where
    // .    the problem occurred and exit. ====
    info = kbot;
  L90:;
  }
  // ==== Return the optimal value of LWORK. ====
  WORK(1) = WorkSize<R>(lwkopt);
  return info;
}

// xLAQR0 (complex T): as the real routine, with the eigenvalues in W.
template <typename T>
blasint Laqr0(bool wantt, bool wantz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, blasint iloz, blasint ihiz, T* z, blasint ldz,
              T* work, blasint lwork) {
  using R = RealT<T>;

  const blasint ntiny = 15;
  const blasint kexnw = 5;
  const blasint kexsh = 6;
  const R wilk1 = R(0.75);
  const T zero = T(0);
  const T one = T(1);
  const R two = R(2);
  const Matrix<T> H(h, ldh);
  char jbcmpz[3] = {' ', ' ', '\0'};
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  T zdum[1];
  blasint i, inf, it, itmax, k, kacc22, kbot, kdu, ks, kt, ktop, ku, kv, kwh,
      kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh, nho, nibble, nmin, ns, nsmax,
      nsr, nve, nw, nwmax, nwr, nwupbd;
  bool sorted;
  R s;
  T aa, bb, cc, dd, det, rtdisc, swap, tr2;
  blasint info = 0;
  // ==== Quick return for N = 0: nothing to do. ====
  if (n == 0) {
    WORK(1) = one;
    return info;
  }
  if (n <= ntiny) {
    // ==== Tiny matrices must use ZLAHQR. ====
    lwkopt = 1;
    if (lwork != -1)
      info = Lahqr(wantt, wantz, n, ilo, ihi, h, ldh, w, iloz, ihiz, z, ldz);
  } else {
    // ==== Use small bulge multi-shift QR with aggressive early
    // .    deflation on larger-than-tiny matrices. ====
    // ==== Hope for the best. ====
    info = 0;
    // ==== Set up job flags for ILAENV. ====
    if (wantt) {
      jbcmpz[0] = 'S';
    } else {
      jbcmpz[0] = 'E';
    }
    if (wantz) {
      jbcmpz[1] = 'V';
    } else {
      jbcmpz[1] = 'N';
    }
    // ==== NWR = recommended deflation window size.  At this
    // .    point,  N .GT. NTINY = 15, so there is enough
    // .    subdiagonal workspace for NWR.GE.2 as required.
    // .    (In fact, there is enough subdiagonal space for
    // .    NWR.GE.4.) ====
    nwr = Ilaenv<T>(13, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nwr = Max<blasint>(2, nwr);
    nwr = Min<blasint>(ihi - ilo + 1, (n - 1) / 3, nwr);
    // ==== NSR = recommended number of simultaneous shifts.
    // .    At this point N .GT. NTINY = 15, so there is at
    // .    enough subdiagonal workspace for NSR to be even
    // .    and greater than or equal to two as required. ====
    nsr = Ilaenv<T>(15, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nsr = Min<blasint>(nsr, (n - 3) / 6, ihi - ilo);
    nsr = Max<blasint>(2, nsr - nsr % 2);
    // ==== Estimate optimal workspace ====
    // ==== Workspace query call to ZLAQR3 ====
    Laqr3(wantt, wantz, n, ilo, ihi, nwr + 1, h, ldh, iloz, ihiz, z, ldz, &ls,
          &ld, w, h, ldh, n, h, ldh, n, h, ldh, work, -1);
    // ==== Optimal workspace = MAX(ZLAQR5, ZLAQR3) ====
    lwkopt = Max(3 * nsr / 2, static_cast<blasint>(std::real(WORK(1))));
    // ==== Quick return in case of workspace query. ====
    if (lwork == -1) {
      WORK(1) = WorkSize<T>(lwkopt);
      return info;
    }
    // ==== ZLAHQR/ZLAQR0 crossover point ====
    nmin = Ilaenv<T>(12, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nmin = Max<blasint>(ntiny, nmin);
    // ==== Nibble crossover point ====
    nibble = Ilaenv<T>(14, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    nibble = Max<blasint>(0, nibble);
    // ==== Accumulate reflections during ttswp?  Use block
    // .    2-by-2 structure during matrix-matrix multiply? ====
    kacc22 = Ilaenv<T>(16, "LAQR0", jbcmpz, n, ilo, ihi, lwork);
    kacc22 = Max<blasint>(0, kacc22);
    kacc22 = Min<blasint>(2, kacc22);
    // ==== NWMAX = the largest possible deflation window for
    // .    which there is sufficient workspace. ====
    nwmax = Min<blasint>((n - 1) / 3, lwork / 2);
    nw = nwmax;
    // ==== NSMAX = the Largest number of simultaneous shifts
    // .    for which there is sufficient workspace. ====
    nsmax = Min<blasint>((n - 3) / 6, 2 * lwork / 3);
    nsmax = nsmax - nsmax % 2;
    // ==== NDFL: an iteration count restarted at deflation. ====
    ndfl = 1;
    // ==== ITMAX = iteration limit ====
    itmax = Max<blasint>(30, 2 * kexsh) * Max<blasint>(10, (ihi - ilo + 1));
    // ==== Last row and column in the active block ====
    kbot = ihi;
    // ==== Main Loop ====
    for (it = 1; it <= itmax; ++it) {
      // ==== Done when KBOT falls below ILO ====
      if (kbot < ilo) goto L80;
      // ==== Locate active block ====
      for (k = kbot; k >= ilo + 1; --k) {
        if (H(k, k - 1) == zero) goto L20;
      }
      k = ilo;
    L20:
      ktop = k;
      // ==== Select deflation window size:
      // .    Typical Case:
      // .      If possible and advisable, nibble the entire
      // .      active block.  If not, use size MIN(NWR,NWMAX)
      // .      or MIN(NWR+1,NWMAX) depending upon which has
      // .      the smaller corresponding subdiagonal entry
      // .      (a heuristic).
      // .
      // .    Exceptional Case:
      // .      If there have been no deflations in KEXNW or
      // .      more iterations, then vary the deflation window
      // .      size.   At first, because, larger windows are,
      // .      in general, more powerful than smaller ones,
      // .      rapidly increase the window to the maximum possible.
      // .      Then, gradually reduce the window size. ====
      nh = kbot - ktop + 1;
      nwupbd = Min<blasint>(nh, nwmax);
      if (ndfl < kexnw) {
        nw = Min<blasint>(nwupbd, nwr);
      } else {
        nw = Min<blasint>(nwupbd, 2 * nw);
      }
      if (nw < nwmax) {
        if (nw >= nh - 1) {
          nw = nh;
        } else {
          kwtop = kbot - nw + 1;
          if (Cabs1(H(kwtop, kwtop - 1)) > Cabs1(H(kwtop - 1, kwtop - 2)))
            nw = nw + 1;
        }
      }
      if (ndfl < kexnw) {
        ndec = -1;
      } else if (ndec >= 0 || nw >= nwupbd) {
        ndec = ndec + 1;
        if (nw - ndec < 2) ndec = 0;
        nw = nw - ndec;
      }
      // ==== Aggressive early deflation:
      // .    split workspace under the subdiagonal into
      // .      - an nw-by-nw work array V in the lower
      // .        left-hand-corner,
      // .      - an NW-by-at-least-NW-but-more-is-better
      // .        (NW-by-NHO) horizontal work array along
      // .        the bottom edge,
      // .      - an at-least-NW-but-more-is-better (NHV-by-NW)
      // .        vertical work array along the left-hand-edge.
      // .        ====
      kv = n - nw + 1;
      kt = nw + 1;
      nho = (n - nw - 1) - kt + 1;
      kwv = nw + 2;
      nve = (n - nw) - kwv + 1;
      // ==== Aggressive early deflation ====
      Laqr3(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, &ls,
            &ld, w, H.ptr(kv, 1), ldh, nho, H.ptr(kv, kt), ldh, nve,
            H.ptr(kwv, 1), ldh, work, lwork);
      // ==== Adjust KBOT accounting for new deflations. ====
      kbot = kbot - ld;
      // ==== KS points to the shifts. ====
      ks = kbot - ls + 1;
      // ==== Skip an expensive QR sweep if there is a (partly
      // .    heuristic) reason to expect that many eigenvalues
      // .    will deflate without it.  Here, the QR sweep is
      // .    skipped if many eigenvalues have just been deflated
      // .    or if the remaining active block is small.
      if ((ld == 0) || ((100 * ld <= nw * nibble) &&
                        (kbot - ktop + 1 > Min<blasint>(nmin, nwmax)))) {
        // ==== NS = nominal number of simultaneous shifts.
        // .    This may be lowered (slightly) if ZLAQR3
        // .    did not provide that many shifts. ====
        ns = Min<blasint>(nsmax, nsr, Max<blasint>(2, kbot - ktop));
        ns = ns - ns % 2;
        // ==== If there have been no deflations
        // .    in a multiple of KEXSH iterations,
        // .    then try exceptional shifts.
        // .    Otherwise use shifts provided by
        // .    ZLAQR3 above or from the eigenvalues
        // .    of a trailing principal submatrix. ====
        if (ndfl % kexsh == 0) {
          ks = kbot - ns + 1;
          for (i = kbot; i >= ks + 1; i -= 2) {
            W(i) = H(i, i) + wilk1 * Cabs1(H(i, i - 1));
            W(i - 1) = W(i);
          }
        } else {
          // ==== Got NS/2 or fewer shifts? Use ZLAQR4 or
          // .    ZLAHQR on a trailing principal submatrix to
          // .    get more. (Since NS.LE.NSMAX.LE.(N-3)/6,
          // .    there is enough space below the subdiagonal
          // .    to fit an NS-by-NS scratch array.) ====
          if (kbot - ks + 1 <= ns / 2) {
            ks = kbot - ns + 1;
            kt = n - ns + 1;
            Lacpy('A', ns, ns, H.ptr(ks, ks), ldh, H.ptr(kt, 1), ldh);
            if (ns > nmin) {
              inf = Laqr4(false, false, ns, 1, ns, H.ptr(kt, 1), ldh, W.ptr(ks),
                          1, 1, zdum, 1, work, lwork);
            } else {
              inf = Lahqr(false, false, ns, 1, ns, H.ptr(kt, 1), ldh, W.ptr(ks),
                          1, 1, zdum, 1);
            }
            ks = ks + inf;
            // ==== In case of a rare QR failure use
            // .    eigenvalues of the trailing 2-by-2
            // .    principal submatrix.  Scale to avoid
            // .    overflows, underflows and subnormals.
            // .    (The scale factor S can not be zero,
            // .    because H(KBOT,KBOT-1) is nonzero.) ====
            if (ks >= kbot) {
              s = Cabs1(H(kbot - 1, kbot - 1)) + Cabs1(H(kbot, kbot - 1)) +
                  Cabs1(H(kbot - 1, kbot)) + Cabs1(H(kbot, kbot));
              aa = H(kbot - 1, kbot - 1) / s;
              cc = H(kbot, kbot - 1) / s;
              bb = H(kbot - 1, kbot) / s;
              dd = H(kbot, kbot) / s;
              tr2 = (aa + dd) / two;
              det = (aa - tr2) * (dd - tr2) - bb * cc;
              rtdisc = std::sqrt(-det);
              W(kbot - 1) = (tr2 + rtdisc) * s;
              W(kbot) = (tr2 - rtdisc) * s;
              ks = kbot - 1;
            }
          }
          if (kbot - ks + 1 > ns) {
            // ==== Sort the shifts (Helps a little) ====
            sorted = false;
            for (k = kbot; k >= ks + 1; --k) {
              if (sorted) goto L60;
              sorted = true;
              for (i = ks; i <= k - 1; ++i) {
                if (Cabs1(W(i)) < Cabs1(W(i + 1))) {
                  sorted = false;
                  swap = W(i);
                  W(i) = W(i + 1);
                  W(i + 1) = swap;
                }
              }
            }
          L60:;
          }
        }
        // ==== If there are only two shifts, then use
        // .    only one.  ====
        if (kbot - ks + 1 == 2) {
          if (Cabs1(W(kbot) - H(kbot, kbot)) <
              Cabs1(W(kbot - 1) - H(kbot, kbot))) {
            W(kbot - 1) = W(kbot);
          } else {
            W(kbot) = W(kbot - 1);
          }
        }
        // ==== Use up to NS of the the smallest magnitude
        // .    shifts.  If there aren't NS shifts available,
        // .    then use them all, possibly dropping one to
        // .    make the number of shifts even. ====
        ns = Min<blasint>(ns, kbot - ks + 1);
        ns = ns - ns % 2;
        ks = kbot - ns + 1;
        // ==== Small-bulge multi-shift QR sweep:
        // .    split workspace under the subdiagonal into
        // .    - a KDU-by-KDU work array U in the lower
        // .      left-hand-corner,
        // .    - a KDU-by-at-least-KDU-but-more-is-better
        // .      (KDU-by-NHo) horizontal work array WH along
        // .      the bottom edge,
        // .    - and an at-least-KDU-but-more-is-better-by-KDU
        // .      (NVE-by-KDU) vertical work WV arrow along
        // .      the left-hand-edge. ====
        kdu = 2 * ns;
        ku = n - kdu + 1;
        kwh = kdu + 1;
        nho = (n - kdu + 1 - 4) - (kdu + 1) + 1;
        kwv = kdu + 4;
        nve = n - kdu - kwv + 1;
        // ==== Small-bulge multi-shift QR sweep ====
        Laqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, W.ptr(ks), h, ldh, iloz,
              ihiz, z, ldz, work, 3, H.ptr(ku, 1), ldh, nve, H.ptr(kwv, 1), ldh,
              nho, H.ptr(ku, kwh), ldh);
      }
      // ==== Note progress (or the lack of it). ====
      if (ld > 0) {
        ndfl = 1;
      } else {
        ndfl = ndfl + 1;
      }
      // ==== End of main loop ====
    }
    // ==== Iteration limit exceeded.  Set INFO to show where
    // .    the problem occurred and exit. ====
    info = kbot;
  L80:;
  }
  // ==== Return the optimal value of LWORK. ====
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xHSEQR (real T): the eigenvalues WR + i*WI of the upper Hessenberg matrix
// H (in rows and columns ILO:IHI; xGEBAL's ILO and IHI, or 1 and N), and
// for JOB = 'S' the Schur form T in H (JOB = 'E': eigenvalues only).
// COMPZ = 'N': no Schur vectors; 'I': Z = the Schur vectors Q of H; 'V':
// Z := Z * Q. WORK has LWORK >= max(1, N) elements (LWORK = -1: workspace
// query).
//
// Returns INFO: 0, -i for an illegal argument i, or i > 0 if xLAQR0 /
// xLAHQR failed to compute all the eigenvalues; eigenvalues i+1:IHI are
// then in WR and WI.
template <typename R>
blasint Hseqr(char job, char compz, blasint n, blasint ilo, blasint ihi, R* h,
              blasint ldh, R* wr, R* wi, R* z, blasint ldz, R* work,
              blasint lwork) {
  const blasint ntiny = 15;
  const blasint nl = 49;
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> H(h, ldh);
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  const Matrix<R> Z(z, ldz);
  blasint i, kbot, nmin;
  bool initz, lquery, wantt, wantz;
  wantt = Lsame(job, 'S');
  initz = Lsame(compz, 'I');
  wantz = initz || Lsame(compz, 'V');
  WORK(1) = R(Max<blasint>(1, n));
  lquery = lwork == -1;
  blasint info = 0;
  if (!Lsame(job, 'E') && !wantt) {
    info = -1;
  } else if (!Lsame(compz, 'N') && !wantz) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1 || ilo > Max<blasint>(1, n)) {
    info = -4;
  } else if (ihi < Min<blasint>(ilo, n) || ihi > n) {
    info = -5;
  } else if (ldh < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldz < 1 || (wantz && ldz < Max<blasint>(1, n))) {
    info = -11;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -13;
  }
  if (info != 0) {
    // ==== Quick return in case of invalid argument. ====
    Xerbla<R>("HSEQR", -info);
    return info;
  } else if (n == 0) {
    // ==== Quick return in case N = 0; nothing to do. ====
    return info;
  } else if (lquery) {
    // ==== Quick return in case of a workspace query ====
    info = Laqr0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz,
                 work, lwork);
    // ==== Ensure reported workspace size is backward-compatible with
    // .    previous LAPACK versions. ====
    WORK(1) = Max(R(Max<blasint>(1, n)), WORK(1));
    return info;
  } else {
    // ==== copy eigenvalues isolated by DGEBAL ====
    for (i = 1; i <= ilo - 1; ++i) {
      WR(i) = H(i, i);
      WI(i) = zero;
    }
    for (i = ihi + 1; i <= n; ++i) {
      WR(i) = H(i, i);
      WI(i) = zero;
    }
    // ==== Initialize Z, if requested ====
    if (initz) Laset('A', n, n, zero, one, z, ldz);
    // ==== Quick return if possible ====
    if (ilo == ihi) {
      WR(ilo) = H(ilo, ilo);
      WI(ilo) = zero;
      return info;
    }
    // ==== DLAHQR/DLAQR0 crossover point ====
    const char opts[3] = {job, compz, '\0'};
    nmin = Ilaenv<R>(12, "HSEQR", opts, n, ilo, ihi, lwork);
    nmin = Max<blasint>(ntiny, nmin);
    // ==== DLAQR0 for big matrices; DLAHQR for small ones ====
    if (n > nmin) {
      info = Laqr0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz,
                   work, lwork);
    } else {
      // ==== Small matrix ====
      info = Lahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz);
      if (info > 0) {
        // ==== A rare DLAHQR failure!  DLAQR0 sometimes succeeds
        // .    when DLAHQR fails. ====
        kbot = info;
        if (n >= nl) {
          // ==== Larger matrices have enough subdiagonal scratch
          // .    space to call DLAQR0 directly. ====
          info = Laqr0(wantt, wantz, n, ilo, kbot, h, ldh, wr, wi, ilo, ihi, z,
                       ldz, work, lwork);
        } else {
          // ==== Tiny matrices don't have enough subdiagonal
          // .    scratch space to benefit from DLAQR0.  Hence,
          // .    tiny matrices must be copied into a larger
          // .    array before calling DLAQR0. ====
          // HL(NL, NL) and WORKL(NL), local arrays of the reference, are
          // only needed here.
          std::vector<R> hl(nl * nl), workl(nl);
          const Matrix<R> HL(hl.data(), nl);
          Lacpy('A', n, n, h, ldh, hl.data(), nl);
          HL(n + 1, n) = zero;
          Laset('A', nl, nl - n, zero, zero, HL.ptr(1, n + 1), nl);
          info = Laqr0(wantt, wantz, nl, ilo, kbot, hl.data(), nl, wr, wi, ilo,
                       ihi, z, ldz, workl.data(), nl);
          if (wantt || info != 0) Lacpy('A', n, n, hl.data(), nl, h, ldh);
        }
      }
    }
    // ==== Clear out the trash, if necessary. ====
    if ((wantt || info != 0) && n > 2)
      Laset('L', n - 2, n - 2, zero, zero, H.ptr(3, 1), ldh);
    // ==== Ensure reported workspace size is backward-compatible with
    // .    previous LAPACK versions. ====
    WORK(1) = Max(R(Max<blasint>(1, n)), WORK(1));
  }
  return info;
}

// xHSEQR (complex T): as the real routine, with the eigenvalues in W.
template <typename T>
blasint Hseqr(char job, char compz, blasint n, blasint ilo, blasint ihi, T* h,
              blasint ldh, T* w, T* z, blasint ldz, T* work, blasint lwork) {
  using R = RealT<T>;

  const blasint ntiny = 15;
  const blasint nl = 49;
  const T zero = T(0);
  const T one = T(1);
  const R rzero = R(0);
  const Matrix<T> H(h, ldh);
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint kbot, nmin;
  bool initz, lquery, wantt, wantz;
  wantt = Lsame(job, 'S');
  initz = Lsame(compz, 'I');
  wantz = initz || Lsame(compz, 'V');
  WORK(1) = T(R(Max<blasint>(1, n)), rzero);
  lquery = lwork == -1;
  blasint info = 0;
  if (!Lsame(job, 'E') && !wantt) {
    info = -1;
  } else if (!Lsame(compz, 'N') && !wantz) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1 || ilo > Max<blasint>(1, n)) {
    info = -4;
  } else if (ihi < Min<blasint>(ilo, n) || ihi > n) {
    info = -5;
  } else if (ldh < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldz < 1 || (wantz && ldz < Max<blasint>(1, n))) {
    info = -10;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -12;
  }
  if (info != 0) {
    // ==== Quick return in case of invalid argument. ====
    Xerbla<T>("HSEQR", -info);
    return info;
  } else if (n == 0) {
    // ==== Quick return in case N = 0; nothing to do. ====
    return info;
  } else if (lquery) {
    // ==== Quick return in case of a workspace query ====
    info = Laqr0(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz, work,
                 lwork);
    // ==== Ensure reported workspace size is backward-compatible with
    // .    previous LAPACK versions. ====
    WORK(1) = T(Max(std::real(WORK(1)), R(Max<blasint>(1, n))), rzero);
    return info;
  } else {
    // ==== copy eigenvalues isolated by ZGEBAL ====
    if (ilo > 1) Copy(ilo - 1, h, ldh + 1, w, 1);
    if (ihi < n)
      Copy(n - ihi, H.ptr(ihi + 1, ihi + 1), ldh + 1, W.ptr(ihi + 1), 1);
    // ==== Initialize Z, if requested ====
    if (initz) Laset('A', n, n, zero, one, z, ldz);
    // ==== Quick return if possible ====
    if (ilo == ihi) {
      W(ilo) = H(ilo, ilo);
      return info;
    }
    // ==== ZLAHQR/ZLAQR0 crossover point ====
    const char opts[3] = {job, compz, '\0'};
    nmin = Ilaenv<T>(12, "HSEQR", opts, n, ilo, ihi, lwork);
    nmin = Max<blasint>(ntiny, nmin);
    // ==== ZLAQR0 for big matrices; ZLAHQR for small ones ====
    if (n > nmin) {
      info = Laqr0(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz, work,
                   lwork);
    } else {
      // ==== Small matrix ====
      info = Lahqr(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz);
      if (info > 0) {
        // ==== A rare ZLAHQR failure!  ZLAQR0 sometimes succeeds
        // .    when ZLAHQR fails. ====
        kbot = info;
        if (n >= nl) {
          // ==== Larger matrices have enough subdiagonal scratch
          // .    space to call ZLAQR0 directly. ====
          info = Laqr0(wantt, wantz, n, ilo, kbot, h, ldh, w, ilo, ihi, z, ldz,
                       work, lwork);
        } else {
          // ==== Tiny matrices don't have enough subdiagonal
          // .    scratch space to benefit from ZLAQR0.  Hence,
          // .    tiny matrices must be copied into a larger
          // .    array before calling ZLAQR0. ====
          // HL(NL, NL) and WORKL(NL), local arrays of the reference, are
          // only needed here.
          std::vector<T> hl(nl * nl), workl(nl);
          const Matrix<T> HL(hl.data(), nl);
          Lacpy('A', n, n, h, ldh, hl.data(), nl);
          HL(n + 1, n) = zero;
          Laset('A', nl, nl - n, zero, zero, HL.ptr(1, n + 1), nl);
          info = Laqr0(wantt, wantz, nl, ilo, kbot, hl.data(), nl, w, ilo, ihi,
                       z, ldz, workl.data(), nl);
          if (wantt || info != 0) Lacpy('A', n, n, hl.data(), nl, h, ldh);
        }
      }
    }
    // ==== Clear out the trash, if necessary. ====
    if ((wantt || info != 0) && n > 2)
      Laset('L', n - 2, n - 2, zero, zero, H.ptr(3, 1), ldh);
    // ==== Ensure reported workspace size is backward-compatible with
    // .    previous LAPACK versions. ====
    WORK(1) = T(Max(R(Max<blasint>(1, n)), std::real(WORK(1))), rzero);
  }
  return info;
}

// NOLINTEND(bugprone-branch-clone)

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                    \
  template void Lanv2<R>(R*, R*, R*, R*, R*, R*, R*, R*, R*, R*);             \
  template void Laqr1<R>(blasint, const R*, blasint, R, R, R, R, R*);         \
  template blasint Lahqr<R>(bool, bool, blasint, blasint, blasint, R*,        \
                            blasint, R*, R*, blasint, blasint, R*, blasint);  \
  template void Laqr5<R>(bool, bool, blasint, blasint, blasint, blasint,      \
                         blasint, R*, R*, R*, blasint, blasint, blasint, R*,  \
                         blasint, R*, blasint, R*, blasint, blasint, R*,      \
                         blasint, blasint, R*, blasint);                      \
  template void Laqr2<R>(bool, bool, blasint, blasint, blasint, blasint, R*,  \
                         blasint, blasint, blasint, R*, blasint, blasint*,    \
                         blasint*, R*, R*, R*, blasint, blasint, R*, blasint, \
                         blasint, R*, blasint, R*, blasint);                  \
  template void Laqr3<R>(bool, bool, blasint, blasint, blasint, blasint, R*,  \
                         blasint, blasint, blasint, R*, blasint, blasint*,    \
                         blasint*, R*, R*, R*, blasint, blasint, R*, blasint, \
                         blasint, R*, blasint, R*, blasint);                  \
  template blasint Laqr4<R>(bool, bool, blasint, blasint, blasint, R*,        \
                            blasint, R*, R*, blasint, blasint, R*, blasint,   \
                            R*, blasint);                                     \
  template blasint Laqr0<R>(bool, bool, blasint, blasint, blasint, R*,        \
                            blasint, R*, R*, blasint, blasint, R*, blasint,   \
                            R*, blasint);                                     \
  template blasint Hseqr<R>(char, char, blasint, blasint, blasint, R*,        \
                            blasint, R*, R*, R*, blasint, R*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                \
  template void Laqr1<T>(blasint, const T*, blasint, T, T, T*);              \
  template blasint Lahqr<T>(bool, bool, blasint, blasint, blasint, T*,       \
                            blasint, T*, blasint, blasint, T*, blasint);     \
  template void Laqr5<T>(bool, bool, blasint, blasint, blasint, blasint,     \
                         blasint, T*, T*, blasint, blasint, blasint, T*,     \
                         blasint, T*, blasint, T*, blasint, blasint, T*,     \
                         blasint, blasint, T*, blasint);                     \
  template void Laqr2<T>(bool, bool, blasint, blasint, blasint, blasint, T*, \
                         blasint, blasint, blasint, T*, blasint, blasint*,   \
                         blasint*, T*, T*, blasint, blasint, T*, blasint,    \
                         blasint, T*, blasint, T*, blasint);                 \
  template void Laqr3<T>(bool, bool, blasint, blasint, blasint, blasint, T*, \
                         blasint, blasint, blasint, T*, blasint, blasint*,   \
                         blasint*, T*, T*, blasint, blasint, T*, blasint,    \
                         blasint, T*, blasint, T*, blasint);                 \
  template blasint Laqr4<T>(bool, bool, blasint, blasint, blasint, T*,       \
                            blasint, T*, blasint, blasint, T*, blasint, T*,  \
                            blasint);                                        \
  template blasint Laqr0<T>(bool, bool, blasint, blasint, blasint, T*,       \
                            blasint, T*, blasint, blasint, T*, blasint, T*,  \
                            blasint);                                        \
  template blasint Hseqr<T>(char, char, blasint, blasint, blasint, T*,       \
                            blasint, T*, T*, blasint, T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// WANTT and WANTZ are Fortran LOGICALs (nonzero is true).
#define HWYBLAS_LAPACK_HSEQR_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##lanv2_(R* a, R* b, R* c, R* d, R* rt1r, R* rt1i,    \
                                  R* rt2r, R* rt2i, R* cs, R* sn) {            \
    hwyblas::lapack::Lanv2(a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn);        \
  }                                                                            \
  HWYBLAS_API void prefix##laqr1_(                                             \
      const hwyblas_int* n, const R* h, const hwyblas_int* ldh, const R* sr1,  \
      const R* si1, const R* sr2, const R* si2, R* v) {                        \
    hwyblas::lapack::Laqr1(*n, h, *ldh, *sr1, *si1, *sr2, *si2, v);            \
  }                                                                            \
  HWYBLAS_API void prefix##lahqr_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* wr, R* wi, const hwyblas_int* iloz,     \
      const hwyblas_int* ihiz, R* z, const hwyblas_int* ldz,                   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lahqr(*wantt != 0, *wantz != 0, *n, *ilo, *ihi,   \
                                   h, *ldh, wr, wi, *iloz, *ihiz, z, *ldz);    \
  }                                                                            \
  HWYBLAS_API void prefix##laqr5_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* kacc22, const hwyblas_int* n,                         \
      const hwyblas_int* ktop, const hwyblas_int* kbot,                        \
      const hwyblas_int* nshfts, R* sr, R* si, R* h, const hwyblas_int* ldh,   \
      const hwyblas_int* iloz, const hwyblas_int* ihiz, R* z,                  \
      const hwyblas_int* ldz, R* v, const hwyblas_int* ldv, R* u,              \
      const hwyblas_int* ldu, const hwyblas_int* nv, R* wv,                    \
      const hwyblas_int* ldwv, const hwyblas_int* nh, R* wh,                   \
      const hwyblas_int* ldwh) {                                               \
    hwyblas::lapack::Laqr5(*wantt != 0, *wantz != 0, *kacc22, *n, *ktop,       \
                           *kbot, *nshfts, sr, si, h, *ldh, *iloz, *ihiz, z,   \
                           *ldz, v, *ldv, u, *ldu, *nv, wv, *ldwv, *nh, wh,    \
                           *ldwh);                                             \
  }                                                                            \
  HWYBLAS_API void prefix##laqr2_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ktop, const hwyblas_int* kbot,  \
      const hwyblas_int* nw, R* h, const hwyblas_int* ldh,                     \
      const hwyblas_int* iloz, const hwyblas_int* ihiz, R* z,                  \
      const hwyblas_int* ldz, hwyblas_int* ns, hwyblas_int* nd, R* sr, R* si,  \
      R* v, const hwyblas_int* ldv, const hwyblas_int* nh, R* t,               \
      const hwyblas_int* ldt, const hwyblas_int* nv, R* wv,                    \
      const hwyblas_int* ldwv, R* work, const hwyblas_int* lwork) {            \
    hwyblas::lapack::Laqr2(*wantt != 0, *wantz != 0, *n, *ktop, *kbot, *nw, h, \
                           *ldh, *iloz, *ihiz, z, *ldz, ns, nd, sr, si, v,     \
                           *ldv, *nh, t, *ldt, *nv, wv, *ldwv, work, *lwork);  \
  }                                                                            \
  HWYBLAS_API void prefix##laqr3_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ktop, const hwyblas_int* kbot,  \
      const hwyblas_int* nw, R* h, const hwyblas_int* ldh,                     \
      const hwyblas_int* iloz, const hwyblas_int* ihiz, R* z,                  \
      const hwyblas_int* ldz, hwyblas_int* ns, hwyblas_int* nd, R* sr, R* si,  \
      R* v, const hwyblas_int* ldv, const hwyblas_int* nh, R* t,               \
      const hwyblas_int* ldt, const hwyblas_int* nv, R* wv,                    \
      const hwyblas_int* ldwv, R* work, const hwyblas_int* lwork) {            \
    hwyblas::lapack::Laqr3(*wantt != 0, *wantz != 0, *n, *ktop, *kbot, *nw, h, \
                           *ldh, *iloz, *ihiz, z, *ldz, ns, nd, sr, si, v,     \
                           *ldv, *nh, t, *ldt, *nv, wv, *ldwv, work, *lwork);  \
  }                                                                            \
  HWYBLAS_API void prefix##laqr4_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* wr, R* wi, const hwyblas_int* iloz,     \
      const hwyblas_int* ihiz, R* z, const hwyblas_int* ldz, R* work,          \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Laqr4(*wantt != 0, *wantz != 0, *n, *ilo, *ihi,   \
                                   h, *ldh, wr, wi, *iloz, *ihiz, z, *ldz,     \
                                   work, *lwork);                              \
  }                                                                            \
  HWYBLAS_API void prefix##laqr0_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* wr, R* wi, const hwyblas_int* iloz,     \
      const hwyblas_int* ihiz, R* z, const hwyblas_int* ldz, R* work,          \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Laqr0(*wantt != 0, *wantz != 0, *n, *ilo, *ihi,   \
                                   h, *ldh, wr, wi, *iloz, *ihiz, z, *ldz,     \
                                   work, *lwork);                              \
  }                                                                            \
  HWYBLAS_API void prefix##hseqr_(                                             \
      const char* job, const char* compz, const hwyblas_int* n,                \
      const hwyblas_int* ilo, const hwyblas_int* ihi, R* h,                    \
      const hwyblas_int* ldh, R* wr, R* wi, R* z, const hwyblas_int* ldz,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Hseqr(*job, *compz, *n, *ilo, *ihi, h, *ldh, wr,  \
                                   wi, z, *ldz, work, *lwork);                 \
  }

HWYBLAS_LAPACK_HSEQR_REAL(s, float)
HWYBLAS_LAPACK_HSEQR_REAL(d, double)

#undef HWYBLAS_LAPACK_HSEQR_REAL

#define HWYBLAS_LAPACK_HSEQR_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##laqr1_(const hwyblas_int* n, const R* h,            \
                                  const hwyblas_int* ldh, const R* s1,         \
                                  const R* s2, R* v) {                         \
    hwyblas::lapack::Laqr1(*n, As<const T>(h), *ldh, *As<const T>(s1),         \
                           *As<const T>(s2), As<T>(v));                        \
  }                                                                            \
  HWYBLAS_API void prefix##lahqr_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* w, const hwyblas_int* iloz,             \
      const hwyblas_int* ihiz, R* z, const hwyblas_int* ldz,                   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lahqr(*wantt != 0, *wantz != 0, *n, *ilo, *ihi,   \
                                   As<T>(h), *ldh, As<T>(w), *iloz, *ihiz,     \
                                   As<T>(z), *ldz);                            \
  }                                                                            \
  HWYBLAS_API void prefix##laqr5_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* kacc22, const hwyblas_int* n,                         \
      const hwyblas_int* ktop, const hwyblas_int* kbot,                        \
      const hwyblas_int* nshfts, R* s, R* h, const hwyblas_int* ldh,           \
      const hwyblas_int* iloz, const hwyblas_int* ihiz, R* z,                  \
      const hwyblas_int* ldz, R* v, const hwyblas_int* ldv, R* u,              \
      const hwyblas_int* ldu, const hwyblas_int* nv, R* wv,                    \
      const hwyblas_int* ldwv, const hwyblas_int* nh, R* wh,                   \
      const hwyblas_int* ldwh) {                                               \
    hwyblas::lapack::Laqr5(                                                    \
        *wantt != 0, *wantz != 0, *kacc22, *n, *ktop, *kbot, *nshfts,          \
        As<T>(s), As<T>(h), *ldh, *iloz, *ihiz, As<T>(z), *ldz, As<T>(v),      \
        *ldv, As<T>(u), *ldu, *nv, As<T>(wv), *ldwv, *nh, As<T>(wh), *ldwh);   \
  }                                                                            \
  HWYBLAS_API void prefix##laqr2_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ktop, const hwyblas_int* kbot,  \
      const hwyblas_int* nw, R* h, const hwyblas_int* ldh,                     \
      const hwyblas_int* iloz, const hwyblas_int* ihiz, R* z,                  \
      const hwyblas_int* ldz, hwyblas_int* ns, hwyblas_int* nd, R* sh, R* v,   \
      const hwyblas_int* ldv, const hwyblas_int* nh, R* t,                     \
      const hwyblas_int* ldt, const hwyblas_int* nv, R* wv,                    \
      const hwyblas_int* ldwv, R* work, const hwyblas_int* lwork) {            \
    hwyblas::lapack::Laqr2(*wantt != 0, *wantz != 0, *n, *ktop, *kbot, *nw,    \
                           As<T>(h), *ldh, *iloz, *ihiz, As<T>(z), *ldz, ns,   \
                           nd, As<T>(sh), As<T>(v), *ldv, *nh, As<T>(t), *ldt, \
                           *nv, As<T>(wv), *ldwv, As<T>(work), *lwork);        \
  }                                                                            \
  HWYBLAS_API void prefix##laqr3_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ktop, const hwyblas_int* kbot,  \
      const hwyblas_int* nw, R* h, const hwyblas_int* ldh,                     \
      const hwyblas_int* iloz, const hwyblas_int* ihiz, R* z,                  \
      const hwyblas_int* ldz, hwyblas_int* ns, hwyblas_int* nd, R* sh, R* v,   \
      const hwyblas_int* ldv, const hwyblas_int* nh, R* t,                     \
      const hwyblas_int* ldt, const hwyblas_int* nv, R* wv,                    \
      const hwyblas_int* ldwv, R* work, const hwyblas_int* lwork) {            \
    hwyblas::lapack::Laqr3(*wantt != 0, *wantz != 0, *n, *ktop, *kbot, *nw,    \
                           As<T>(h), *ldh, *iloz, *ihiz, As<T>(z), *ldz, ns,   \
                           nd, As<T>(sh), As<T>(v), *ldv, *nh, As<T>(t), *ldt, \
                           *nv, As<T>(wv), *ldwv, As<T>(work), *lwork);        \
  }                                                                            \
  HWYBLAS_API void prefix##laqr4_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* w, const hwyblas_int* iloz,             \
      const hwyblas_int* ihiz, R* z, const hwyblas_int* ldz, R* work,          \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Laqr4(*wantt != 0, *wantz != 0, *n, *ilo, *ihi,   \
                                   As<T>(h), *ldh, As<T>(w), *iloz, *ihiz,     \
                                   As<T>(z), *ldz, As<T>(work), *lwork);       \
  }                                                                            \
  HWYBLAS_API void prefix##laqr0_(                                             \
      const hwyblas_int* wantt, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* w, const hwyblas_int* iloz,             \
      const hwyblas_int* ihiz, R* z, const hwyblas_int* ldz, R* work,          \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Laqr0(*wantt != 0, *wantz != 0, *n, *ilo, *ihi,   \
                                   As<T>(h), *ldh, As<T>(w), *iloz, *ihiz,     \
                                   As<T>(z), *ldz, As<T>(work), *lwork);       \
  }                                                                            \
  HWYBLAS_API void prefix##hseqr_(                                             \
      const char* job, const char* compz, const hwyblas_int* n,                \
      const hwyblas_int* ilo, const hwyblas_int* ihi, R* h,                    \
      const hwyblas_int* ldh, R* w, R* z, const hwyblas_int* ldz, R* work,     \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info =                                                                    \
        hwyblas::lapack::Hseqr(*job, *compz, *n, *ilo, *ihi, As<T>(h), *ldh,   \
                               As<T>(w), As<T>(z), *ldz, As<T>(work), *lwork); \
  }

HWYBLAS_LAPACK_HSEQR_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_HSEQR_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_HSEQR_COMPLEX

}  // extern "C"
