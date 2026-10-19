// The singular value decomposition of a real bidiagonal matrix by the
// implicit zero-shift QR algorithm: xBDSQR, with the 2-by-2 SVDs xLAS2
// (singular values) and xLASV2 (singular values and vectors).
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

// xLAS2: the singular values of the 2-by-2 upper triangular matrix
//   [ F G ]
//   [ 0 H ],
// SSMIN the smaller and SSMAX the larger. Barring over/underflow, they are
// accurate to a few ulps (a relative error of at most 4 * eps for the
// computation itself).
template <typename R>
void Las2(R f, R g, R h, R* ssmin, R* ssmax) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  R as, at, au, c, fa, fhmn, fhmx, ga, ha;
  fa = std::abs(f);
  ga = std::abs(g);
  ha = std::abs(h);
  fhmn = Min(fa, ha);
  fhmx = Max(fa, ha);
  if (fhmn == zero) {
    *ssmin = zero;
    if (fhmx == zero) {
      *ssmax = ga;
    } else {
      *ssmax =
          Max(fhmx, ga) * std::sqrt(one + (Min(fhmx, ga) / Max(fhmx, ga)) *
                                              (Min(fhmx, ga) / Max(fhmx, ga)));
    }
  } else {
    if (ga < fhmx) {
      as = one + fhmn / fhmx;
      at = (fhmx - fhmn) / fhmx;
      au = (ga / fhmx) * (ga / fhmx);
      c = two / (std::sqrt(as * as + au) + std::sqrt(at * at + au));
      *ssmin = fhmn * c;
      *ssmax = fhmx / c;
    } else {
      au = fhmx / ga;
      if (au == zero) {
        // Avoid possible harmful underflow if exponent range
        // asymmetric (true SSMIN may not underflow even if
        // AU underflows)
        *ssmin = (fhmn * fhmx) / ga;
        *ssmax = ga;
      } else {
        as = one + fhmn / fhmx;
        at = (fhmx - fhmn) / fhmx;
        c = one / (std::sqrt(one + (as * au) * (as * au)) +
                   std::sqrt(one + (at * au) * (at * au)));
        *ssmin = (fhmn * c) * au;
        *ssmin = *ssmin + *ssmin;
        *ssmax = ga / (c + c);
      }
    }
  }
}

// xLASV2: the singular value decomposition of the 2-by-2 upper triangular
// matrix [ F G; 0 H ]:
//   [ CSL SNL ] [ F G ] [ CSR -SNR ]   [ SSMAX   0   ]
//   [-SNL CSL ] [ 0 H ] [ SNR  CSR ] = [   0   SSMIN ],
// with abs(SSMAX) the larger singular value, and (CSL, SNL) and (CSR, SNR)
// the left and right singular vectors for abs(SSMAX).
template <typename R>
void Lasv2(R f, R g, R h, R* ssmin, R* ssmax, R* snr, R* csr, R* snl, R* csl) {
  const R zero = R(0);
  const R half = R(0.5);
  const R one = R(1);
  const R two = R(2);
  const R four = R(4);
  blasint pmax;
  bool gasmal, swap;
  R a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m, mm, r, s, slt, srt, t, temp,
      tsign, tt;
  ft = f;
  fa = std::abs(ft);
  ht = h;
  ha = std::abs(h);
  // PMAX points to the maximum absolute element of matrix
  // PMAX = 1 if F largest in absolute values
  // PMAX = 2 if G largest in absolute values
  // PMAX = 3 if H largest in absolute values
  pmax = 1;
  swap = (ha > fa);
  if (swap) {
    pmax = 3;
    temp = ft;
    ft = ht;
    ht = temp;
    temp = fa;
    fa = ha;
    ha = temp;
    // Now FA .ge. HA
  }
  gt = g;
  ga = std::abs(gt);
  if (ga == zero) {
    // Diagonal matrix
    *ssmin = ha;
    *ssmax = fa;
    clt = one;
    crt = one;
    slt = zero;
    srt = zero;
  } else {
    gasmal = true;
    if (ga > fa) {
      pmax = 2;
      if ((fa / ga) < Lamch<R>('E')) {
        // Case of very large GA
        gasmal = false;
        *ssmax = ga;
        if (ha > one) {
          *ssmin = fa / (ga / ha);
        } else {
          *ssmin = (fa / ga) * ha;
        }
        clt = one;
        slt = ht / gt;
        srt = one;
        crt = ft / gt;
      }
    }
    if (gasmal) {
      // Normal case
      d = fa - ha;
      if (d == fa) {
        // Copes with infinite F or H
        l = one;
      } else {
        l = d / fa;
      }
      // Note that 0 .le. L .le. 1
      m = gt / ft;
      // Note that abs(M) .le. 1/macheps
      t = two - l;
      // Note that T .ge. 1
      mm = m * m;
      tt = t * t;
      s = std::sqrt(tt + mm);
      // Note that 1 .le. S .le. 1 + 1/macheps
      if (l == zero) {
        r = std::abs(m);
      } else {
        r = std::sqrt(l * l + mm);
      }
      // Note that 0 .le. R .le. 1 + 1/macheps
      a = half * (s + r);
      // Note that 1 .le. A .le. 1 + abs(M)
      *ssmin = ha / a;
      *ssmax = fa * a;
      if (mm == zero) {
        // Note that M is very tiny
        if (l == zero) {
          t = std::copysign(two, ft) * std::copysign(one, gt);
        } else {
          t = gt / std::copysign(d, ft) + m / t;
        }
      } else {
        t = (m / (s + t) + m / (r + l)) * (one + a);
      }
      l = std::sqrt(t * t + four);
      crt = two / l;
      srt = t / l;
      clt = (crt + srt * m) / a;
      slt = (ht / ft) * srt / a;
    }
  }
  if (swap) {
    *csl = srt;
    *snl = crt;
    *csr = slt;
    *snr = clt;
  } else {
    *csl = clt;
    *snl = slt;
    *csr = crt;
    *snr = srt;
  }
  // Correct signs of SSMAX and SSMIN
  if (pmax == 1)
    tsign = std::copysign(one, *csr) * std::copysign(one, *csl) *
            std::copysign(one, f);
  if (pmax == 2)
    tsign = std::copysign(one, *snr) * std::copysign(one, *csl) *
            std::copysign(one, g);
  if (pmax == 3)
    tsign = std::copysign(one, *snr) * std::copysign(one, *snl) *
            std::copysign(one, h);
  *ssmax = std::copysign(*ssmax, tsign);
  *ssmin = std::copysign(*ssmin,
                         tsign * std::copysign(one, f) * std::copysign(one, h));
}

// xBDSQR: computes the singular values and, optionally, the right and/or
// left singular vectors from the singular value decomposition B = Q * S *
// P**H of the N-by-N real upper ('U') or lower ('L') bidiagonal matrix B
// with diagonal D and off-diagonal E, with the implicit zero-shift QR
// algorithm of Demmel and Kahan; without singular vectors (NCVT = NRU =
// NCC = 0) with the dqds algorithm (xLASQ1). On exit, D holds the singular
// values in decreasing order, and
//   VT (N-by-NCVT) is overwritten by P**H * VT,
//   U (NRU-by-N) by U * Q,
//   C (N-by-NCC) by Q**H * C.
// For complex T, VT, U and C are complex and the rotations real. WORK has
// 4*(N-1) elements (4*N without vectors). Returns INFO: 0, -i for an
// illegal argument i, or i > 0 if i elements of E did not converge to zero
// (B is then orthogonally equivalent to the bidiagonal matrix D, E).
template <typename T>
blasint Bdsqr(char uplo, blasint n, blasint ncvt, blasint nru, blasint ncc,
              RealT<T>* d, RealT<T>* e, T* vt, blasint ldvt, T* u, blasint ldu,
              T* c, blasint ldc, RealT<T>* work) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R negone = -R(1);
  const R hndrth = R(0.01);
  const R ten = R(10);
  const R hndrd = R(100);
  const R meigth = -R(0.125);
  const blasint maxitr = 6;
  const Matrix<T> C(c, ldc);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<R> WORK(work);
  const Matrix<T> U(u, ldu);
  const Matrix<T> VT(vt, ldvt);
  blasint i, idir, isub, iter, iterdivn, j, ll, lll, m, maxitdivn, nm1, nm12,
      nm13, oldll, oldm;
  bool lower, rotate;
  R abse, abss, cosl, cosr, cs, eps, f, g, h, mu, oldcs, oldsn, r, shift, sigmn,
      sigmx, sinl, sinr, sll, smax, smin, sminoa, sn, thresh, tol, tolmul, unfl;
  blasint info = 0;
  lower = Lsame(uplo, 'L');
  if (!Lsame(uplo, 'U') && !lower) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ncvt < 0) {
    info = -3;
  } else if (nru < 0) {
    info = -4;
  } else if (ncc < 0) {
    info = -5;
  } else if ((ncvt == 0 && ldvt < 1) ||
             (ncvt > 0 && ldvt < Max<blasint>(1, n))) {
    info = -9;
  } else if (ldu < Max<blasint>(1, nru)) {
    info = -11;
  } else if ((ncc == 0 && ldc < 1) || (ncc > 0 && ldc < Max<blasint>(1, n))) {
    info = -13;
  }
  if (info != 0) {
    Xerbla<T>("BDSQR", -info);
    return info;
  }
  if (n == 0) return info;
  if (n == 1) goto L160;
  // ROTATE is true if any singular vectors desired, false otherwise
  rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);
  // If no singular vectors desired, use qd algorithm
  if (!rotate) {
    info = Lasq1(n, d, e, work);
    // If INFO equals 2, dqds didn't finish, try to finish
    if (info != 2) return info;
    info = 0;
  }
  nm1 = n - 1;
  nm12 = nm1 + nm1;
  nm13 = nm12 + nm1;
  idir = 0;
  // Get machine constants
  eps = Lamch<R>('E');
  unfl = Lamch<R>('S');
  // If matrix lower bidiagonal, rotate to be upper bidiagonal
  // by applying Givens rotations on the left
  if (lower) {
    for (i = 1; i <= n - 1; ++i) {
      Lartg(D(i), E(i), &cs, &sn, &r);
      D(i) = r;
      E(i) = sn * D(i + 1);
      D(i + 1) = cs * D(i + 1);
      WORK(i) = cs;
      WORK(nm1 + i) = sn;
    }
    // Update singular vectors if desired
    if (nru > 0) Lasr('R', 'V', 'F', nru, n, WORK.ptr(1), WORK.ptr(n), u, ldu);
    if (ncc > 0) Lasr('L', 'V', 'F', n, ncc, WORK.ptr(1), WORK.ptr(n), c, ldc);
  }
  // Compute singular values to relative accuracy TOL
  // (By setting TOL to be negative, algorithm will compute
  // singular values to absolute accuracy ABS(TOL)*norm(input matrix))
  tolmul = Max(ten, Min(hndrd, std::pow(eps, meigth)));
  tol = tolmul * eps;
  // Compute approximate maximum, minimum singular values
  smax = zero;
  for (i = 1; i <= n; ++i) {
    smax = Max(smax, std::abs(D(i)));
  }
  for (i = 1; i <= n - 1; ++i) {
    smax = Max(smax, std::abs(E(i)));
  }
  smin = zero;
  if (tol >= zero) {
    // Relative accuracy desired
    sminoa = std::abs(D(1));
    if (sminoa == zero) goto L50;
    mu = sminoa;
    for (i = 2; i <= n; ++i) {
      mu = std::abs(D(i)) * (mu / (mu + std::abs(E(i - 1))));
      sminoa = Min(sminoa, mu);
      if (sminoa == zero) goto L50;
    }
  L50:
    sminoa = sminoa / std::sqrt(R(n));
    thresh = Max(tol * sminoa, maxitr * (n * (n * unfl)));
  } else {
    // Absolute accuracy desired
    thresh = Max(std::abs(tol) * smax, maxitr * (n * (n * unfl)));
  }
  // Prepare for main iteration loop for the singular values
  // (MAXIT is the maximum number of passes through the inner
  // loop permitted before nonconvergence signalled.)
  maxitdivn = maxitr * n;
  iterdivn = 0;
  iter = -1;
  oldll = -1;
  oldm = -1;
  // M points to last element of unconverged part of matrix
  m = n;
  // Begin main iteration loop
L60:
  // Check for convergence or exceeding iteration count
  if (m <= 1) goto L160;
  if (iter >= n) {
    iter = iter - n;
    iterdivn = iterdivn + 1;
    if (iterdivn >= maxitdivn) goto L200;
  }
  // Find diagonal block of matrix to work on
  if (tol < zero && std::abs(D(m)) <= thresh) D(m) = zero;
  smax = std::abs(D(m));
  for (lll = 1; lll <= m - 1; ++lll) {
    ll = m - lll;
    abss = std::abs(D(ll));
    abse = std::abs(E(ll));
    if (tol < zero && abss <= thresh) D(ll) = zero;
    if (abse <= thresh) goto L80;
    smax = Max(smax, abss, abse);
  }
  ll = 0;
  goto L90;
L80:
  E(ll) = zero;
  // Matrix splits since E(LL) = 0
  if (ll == m - 1) {
    // Convergence of bottom singular value, return to top of loop
    m = m - 1;
    goto L60;
  }
L90:
  ll = ll + 1;
  // E(LL) through E(M-1) are nonzero, E(LL-1) is zero
  if (ll == m - 1) {
    // 2 by 2 block, handle separately
    Lasv2(D(m - 1), E(m - 1), D(m), &sigmn, &sigmx, &sinr, &cosr, &sinl, &cosl);
    D(m - 1) = sigmx;
    E(m - 1) = zero;
    D(m) = sigmn;
    // Compute singular vectors, if desired
    if (ncvt > 0)
      Rot(ncvt, VT.ptr(m - 1, 1), ldvt, VT.ptr(m, 1), ldvt, cosr, sinr);
    if (nru > 0) Rot(nru, U.ptr(1, m - 1), 1, U.ptr(1, m), 1, cosl, sinl);
    if (ncc > 0) Rot(ncc, C.ptr(m - 1, 1), ldc, C.ptr(m, 1), ldc, cosl, sinl);
    m = m - 2;
    goto L60;
  }
  // If working on new submatrix, choose shift direction
  // (from larger end diagonal element towards smaller)
  if (ll > oldm || m < oldll) {
    if (std::abs(D(ll)) >= std::abs(D(m))) {
      // Chase bulge from top (big end) to bottom (small end)
      idir = 1;
    } else {
      // Chase bulge from bottom (big end) to top (small end)
      idir = 2;
    }
  }
  // Apply convergence tests
  if (idir == 1) {
    // Run convergence test in forward direction
    // First apply standard test to bottom of matrix
    if (std::abs(E(m - 1)) <= std::abs(tol) * std::abs(D(m)) ||
        (tol < zero && std::abs(E(m - 1)) <= thresh)) {
      E(m - 1) = zero;
      goto L60;
    }
    if (tol >= zero) {
      // If relative accuracy desired,
      // apply convergence criterion forward
      mu = std::abs(D(ll));
      smin = mu;
      for (lll = ll; lll <= m - 1; ++lll) {
        if (std::abs(E(lll)) <= tol * mu) {
          E(lll) = zero;
          goto L60;
        }
        mu = std::abs(D(lll + 1)) * (mu / (mu + std::abs(E(lll))));
        smin = Min(smin, mu);
      }
    }
  } else {
    // Run convergence test in backward direction
    // First apply standard test to top of matrix
    if (std::abs(E(ll)) <= std::abs(tol) * std::abs(D(ll)) ||
        (tol < zero && std::abs(E(ll)) <= thresh)) {
      E(ll) = zero;
      goto L60;
    }
    if (tol >= zero) {
      // If relative accuracy desired,
      // apply convergence criterion backward
      mu = std::abs(D(m));
      smin = mu;
      for (lll = m - 1; lll >= ll; --lll) {
        if (std::abs(E(lll)) <= tol * mu) {
          E(lll) = zero;
          goto L60;
        }
        mu = std::abs(D(lll)) * (mu / (mu + std::abs(E(lll))));
        smin = Min(smin, mu);
      }
    }
  }
  oldll = ll;
  oldm = m;
  // Compute shift.  First, test if shifting would ruin relative
  // accuracy, and if so set the shift to zero.
  if (tol >= zero && n * tol * (smin / smax) <= Max(eps, hndrth * tol)) {
    // Use a zero shift to avoid loss of relative accuracy
    shift = zero;
  } else {
    // Compute the shift from 2-by-2 block at end of matrix
    if (idir == 1) {
      sll = std::abs(D(ll));
      Las2(D(m - 1), E(m - 1), D(m), &shift, &r);
    } else {
      sll = std::abs(D(m));
      Las2(D(ll), E(ll), D(ll + 1), &shift, &r);
    }
    // Test if shift negligible, and if so set to zero
    if (sll > zero) {
      if ((shift / sll) * (shift / sll) < eps) shift = zero;
    }
  }
  // Increment iteration count
  iter = iter + m - ll;
  // If SHIFT = 0, do simplified QR iteration
  if (shift == zero) {
    if (idir == 1) {
      // Chase bulge from top to bottom
      // Save cosines and sines for later singular vector updates
      cs = one;
      oldcs = one;
      for (i = ll; i <= m - 1; ++i) {
        Lartg(D(i) * cs, E(i), &cs, &sn, &r);
        if (i > ll) E(i - 1) = oldsn * r;
        Lartg(oldcs * r, D(i + 1) * sn, &oldcs, &oldsn, D.ptr(i));
        WORK(i - ll + 1) = cs;
        WORK(i - ll + 1 + nm1) = sn;
        WORK(i - ll + 1 + nm12) = oldcs;
        WORK(i - ll + 1 + nm13) = oldsn;
      }
      h = D(m) * cs;
      D(m) = h * oldcs;
      E(m - 1) = h * oldsn;
      // Update singular vectors
      if (ncvt > 0)
        Lasr('L', 'V', 'F', m - ll + 1, ncvt, WORK.ptr(1), WORK.ptr(n),
             VT.ptr(ll, 1), ldvt);
      if (nru > 0)
        Lasr('R', 'V', 'F', nru, m - ll + 1, WORK.ptr(nm12 + 1),
             WORK.ptr(nm13 + 1), U.ptr(1, ll), ldu);
      if (ncc > 0)
        Lasr('L', 'V', 'F', m - ll + 1, ncc, WORK.ptr(nm12 + 1),
             WORK.ptr(nm13 + 1), C.ptr(ll, 1), ldc);
      // Test convergence
      if (std::abs(E(m - 1)) <= thresh) E(m - 1) = zero;
    } else {
      // Chase bulge from bottom to top
      // Save cosines and sines for later singular vector updates
      cs = one;
      oldcs = one;
      for (i = m; i >= ll + 1; --i) {
        Lartg(D(i) * cs, E(i - 1), &cs, &sn, &r);
        if (i < m) E(i) = oldsn * r;
        Lartg(oldcs * r, D(i - 1) * sn, &oldcs, &oldsn, D.ptr(i));
        WORK(i - ll) = cs;
        WORK(i - ll + nm1) = -sn;
        WORK(i - ll + nm12) = oldcs;
        WORK(i - ll + nm13) = -oldsn;
      }
      h = D(ll) * cs;
      D(ll) = h * oldcs;
      E(ll) = h * oldsn;
      // Update singular vectors
      if (ncvt > 0)
        Lasr('L', 'V', 'B', m - ll + 1, ncvt, WORK.ptr(nm12 + 1),
             WORK.ptr(nm13 + 1), VT.ptr(ll, 1), ldvt);
      if (nru > 0)
        Lasr('R', 'V', 'B', nru, m - ll + 1, WORK.ptr(1), WORK.ptr(n),
             U.ptr(1, ll), ldu);
      if (ncc > 0)
        Lasr('L', 'V', 'B', m - ll + 1, ncc, WORK.ptr(1), WORK.ptr(n),
             C.ptr(ll, 1), ldc);
      // Test convergence
      if (std::abs(E(ll)) <= thresh) E(ll) = zero;
    }
  } else {
    // Use nonzero shift
    if (idir == 1) {
      // Chase bulge from top to bottom
      // Save cosines and sines for later singular vector updates
      f = (std::abs(D(ll)) - shift) *
          (std::copysign(one, D(ll)) + shift / D(ll));
      g = E(ll);
      for (i = ll; i <= m - 1; ++i) {
        Lartg(f, g, &cosr, &sinr, &r);
        if (i > ll) E(i - 1) = r;
        f = cosr * D(i) + sinr * E(i);
        E(i) = cosr * E(i) - sinr * D(i);
        g = sinr * D(i + 1);
        D(i + 1) = cosr * D(i + 1);
        Lartg(f, g, &cosl, &sinl, &r);
        D(i) = r;
        f = cosl * E(i) + sinl * D(i + 1);
        D(i + 1) = cosl * D(i + 1) - sinl * E(i);
        if (i < m - 1) {
          g = sinl * E(i + 1);
          E(i + 1) = cosl * E(i + 1);
        }
        WORK(i - ll + 1) = cosr;
        WORK(i - ll + 1 + nm1) = sinr;
        WORK(i - ll + 1 + nm12) = cosl;
        WORK(i - ll + 1 + nm13) = sinl;
      }
      E(m - 1) = f;
      // Update singular vectors
      if (ncvt > 0)
        Lasr('L', 'V', 'F', m - ll + 1, ncvt, WORK.ptr(1), WORK.ptr(n),
             VT.ptr(ll, 1), ldvt);
      if (nru > 0)
        Lasr('R', 'V', 'F', nru, m - ll + 1, WORK.ptr(nm12 + 1),
             WORK.ptr(nm13 + 1), U.ptr(1, ll), ldu);
      if (ncc > 0)
        Lasr('L', 'V', 'F', m - ll + 1, ncc, WORK.ptr(nm12 + 1),
             WORK.ptr(nm13 + 1), C.ptr(ll, 1), ldc);
      // Test convergence
      if (std::abs(E(m - 1)) <= thresh) E(m - 1) = zero;
    } else {
      // Chase bulge from bottom to top
      // Save cosines and sines for later singular vector updates
      f = (std::abs(D(m)) - shift) * (std::copysign(one, D(m)) + shift / D(m));
      g = E(m - 1);
      for (i = m; i >= ll + 1; --i) {
        Lartg(f, g, &cosr, &sinr, &r);
        if (i < m) E(i) = r;
        f = cosr * D(i) + sinr * E(i - 1);
        E(i - 1) = cosr * E(i - 1) - sinr * D(i);
        g = sinr * D(i - 1);
        D(i - 1) = cosr * D(i - 1);
        Lartg(f, g, &cosl, &sinl, &r);
        D(i) = r;
        f = cosl * E(i - 1) + sinl * D(i - 1);
        D(i - 1) = cosl * D(i - 1) - sinl * E(i - 1);
        if (i > ll + 1) {
          g = sinl * E(i - 2);
          E(i - 2) = cosl * E(i - 2);
        }
        WORK(i - ll) = cosr;
        WORK(i - ll + nm1) = -sinr;
        WORK(i - ll + nm12) = cosl;
        WORK(i - ll + nm13) = -sinl;
      }
      E(ll) = f;
      // Test convergence
      if (std::abs(E(ll)) <= thresh) E(ll) = zero;
      // Update singular vectors if desired
      if (ncvt > 0)
        Lasr('L', 'V', 'B', m - ll + 1, ncvt, WORK.ptr(nm12 + 1),
             WORK.ptr(nm13 + 1), VT.ptr(ll, 1), ldvt);
      if (nru > 0)
        Lasr('R', 'V', 'B', nru, m - ll + 1, WORK.ptr(1), WORK.ptr(n),
             U.ptr(1, ll), ldu);
      if (ncc > 0)
        Lasr('L', 'V', 'B', m - ll + 1, ncc, WORK.ptr(1), WORK.ptr(n),
             C.ptr(ll, 1), ldc);
    }
  }
  // QR iteration finished, go back and check convergence
  goto L60;
  // All singular values converged, so make them positive
L160:
  for (i = 1; i <= n; ++i) {
    if (D(i) == zero) {
      // Avoid -ZERO
      D(i) = zero;
    }
    if (D(i) < zero) {
      D(i) = -D(i);
      // Change sign of singular vectors, if desired
      if (ncvt > 0) Rscal(ncvt, negone, VT.ptr(i, 1), ldvt);
    }
  }
  // Sort the singular values into decreasing order (insertion sort on
  // singular values, but only one transposition per singular vector)
  for (i = 1; i <= n - 1; ++i) {
    // Scan for smallest D(I)
    isub = 1;
    smin = D(1);
    for (j = 2; j <= n + 1 - i; ++j) {
      if (D(j) <= smin) {
        isub = j;
        smin = D(j);
      }
    }
    if (isub != n + 1 - i) {
      // Swap singular values and vectors
      D(isub) = D(n + 1 - i);
      D(n + 1 - i) = smin;
      if (ncvt > 0)
        Swap(ncvt, VT.ptr(isub, 1), ldvt, VT.ptr(n + 1 - i, 1), ldvt);
      if (nru > 0) Swap(nru, U.ptr(1, isub), 1, U.ptr(1, n + 1 - i), 1);
      if (ncc > 0) Swap(ncc, C.ptr(isub, 1), ldc, C.ptr(n + 1 - i, 1), ldc);
    }
  }
  goto L220;
  // Maximum number of iterations exceeded, failure to converge
L200:
  info = 0;
  for (i = 1; i <= n - 1; ++i) {
    if (E(i) != zero) info = info + 1;
  }
L220:
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template blasint Bdsqr<T>(char, blasint, blasint, blasint, blasint,       \
                            RealT<T>*, RealT<T>*, T*, blasint, T*, blasint, \
                            T*, blasint, RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

template void Las2<float>(float, float, float, float*, float*);
template void Las2<double>(double, double, double, double*, double*);
template void Lasv2<float>(float, float, float, float*, float*, float*, float*,
                           float*, float*);
template void Lasv2<double>(double, double, double, double*, double*, double*,
                            double*, double*, double*);

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_LAS2(prefix, R)                                         \
  HWYBLAS_API void prefix##las2_(const R* f, const R* g, const R* h, R* ssmin, \
                                 R* ssmax) {                                   \
    hwyblas::lapack::Las2(*f, *g, *h, ssmin, ssmax);                           \
  }                                                                            \
  HWYBLAS_API void prefix##lasv2_(const R* f, const R* g, const R* h,          \
                                  R* ssmin, R* ssmax, R* snr, R* csr, R* snl,  \
                                  R* csl) {                                    \
    hwyblas::lapack::Lasv2(*f, *g, *h, ssmin, ssmax, snr, csr, snl, csl);      \
  }

HWYBLAS_LAPACK_LAS2(s, float)
HWYBLAS_LAPACK_LAS2(d, double)

#undef HWYBLAS_LAPACK_LAS2

// xBDSQR; the last argument is WORK for real T and RWORK for complex T.
#define HWYBLAS_LAPACK_BDSQR(prefix, T, R)                                    \
  HWYBLAS_API void prefix##bdsqr_(                                            \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* ncvt,        \
      const hwyblas_int* nru, const hwyblas_int* ncc, R* d, R* e, R* vt,      \
      const hwyblas_int* ldvt, R* u, const hwyblas_int* ldu, R* c,            \
      const hwyblas_int* ldc, R* work, hwyblas_int* info) {                   \
    *info =                                                                   \
        hwyblas::lapack::Bdsqr(*uplo, *n, *ncvt, *nru, *ncc, d, e, As<T>(vt), \
                               *ldvt, As<T>(u), *ldu, As<T>(c), *ldc, work);  \
  }

HWYBLAS_LAPACK_BDSQR(s, float, float)
HWYBLAS_LAPACK_BDSQR(d, double, double)
HWYBLAS_LAPACK_BDSQR(c, c32, float)
HWYBLAS_LAPACK_BDSQR(z, c64, double)

#undef HWYBLAS_LAPACK_BDSQR

}  // extern "C"
