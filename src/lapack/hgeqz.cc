// The QZ iteration for the generalized eigenvalue problem of a pair (H, T)
// of an upper Hessenberg and an upper triangular matrix: xHGEQZ (the
// single- / double-shift QZ method of Moler and Stewart), with xLAG2 (the
// eigenvalues of a 2-by-2 pair, scaled to avoid over- and underflow).
//
// The real and the complex xHGEQZ differ in their eigenvalues (ALPHAR and
// ALPHAI for ALPHA), their shifts (the real routine uses double shifts
// for complex conjugate pairs and returns a generalized Schur form with
// 2-by-2 blocks) and workspaces, so they are two overloads. The matrix T
// is named Tm (T is the template parameter).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLAG2: the eigenvalues of the 2-by-2 pair (A, B), B upper triangular,
// with scaling to avoid over- and underflow: the eigenvalues are WR1 /
// SCALE1 and WR2 / SCALE2 (real, WI = 0) or (WR1 +- i*WI) / SCALE1 (a
// complex pair, WR1 = WR2, SCALE1 = SCALE2). B's diagonal elements are
// perturbed up to sqrt(SAFMIN) * norm(B) to make it nonsingular. SAFMIN is
// about the underflow threshold (xHGEQZ passes 100 times it).
template <typename R>
void Lag2(const R* a, blasint lda, const R* b, blasint ldb, R safmin, R* scale1,
          R* scale2, R* wr1, R* wr2, R* wi) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = one / two;
  const R fuzzy1 = one + R(1.0e-5);
  const Matrix<const R> A(a, lda);
  const Matrix<const R> B(b, ldb);
  R a11, a12, a21, a22, abi22, anorm, as11, as12, as22, ascale, b11, b12, b22,
      binv11, binv22, bmin, bnorm, bscale, bsize, c1, c2, c3, c4, c5, diff,
      discr, pp, qq, r, rtmax, rtmin, s1, s2, safmax, shift, ss, sum, wabs,
      wbig, wdet, wscale, wsize, wsmall;
  rtmin = std::sqrt(safmin);
  rtmax = one / rtmin;
  safmax = one / safmin;
  // Scale A
  anorm = Max(std::abs(A(1, 1)) + std::abs(A(2, 1)),
              std::abs(A(1, 2)) + std::abs(A(2, 2)), safmin);
  ascale = one / anorm;
  a11 = ascale * A(1, 1);
  a21 = ascale * A(2, 1);
  a12 = ascale * A(1, 2);
  a22 = ascale * A(2, 2);
  // Perturb B if necessary to insure non-singularity
  b11 = B(1, 1);
  b12 = B(1, 2);
  b22 = B(2, 2);
  bmin = rtmin * Max(std::abs(b11), std::abs(b12), std::abs(b22), rtmin);
  if (std::abs(b11) < bmin) b11 = std::copysign(bmin, b11);
  if (std::abs(b22) < bmin) b22 = std::copysign(bmin, b22);
  // Scale B
  bnorm = Max(std::abs(b11), std::abs(b12) + std::abs(b22), safmin);
  bsize = Max(std::abs(b11), std::abs(b22));
  bscale = one / bsize;
  b11 = b11 * bscale;
  b12 = b12 * bscale;
  b22 = b22 * bscale;
  // Compute larger eigenvalue by method described by C. van Loan
  // ( AS is A shifted by -SHIFT*B )
  binv11 = one / b11;
  binv22 = one / b22;
  s1 = a11 * binv11;
  s2 = a22 * binv22;
  if (std::abs(s1) <= std::abs(s2)) {
    as12 = a12 - s1 * b12;
    as22 = a22 - s1 * b22;
    ss = a21 * (binv11 * binv22);
    abi22 = as22 * binv22 - ss * b12;
    pp = half * abi22;
    shift = s1;
  } else {
    as12 = a12 - s2 * b12;
    as11 = a11 - s2 * b11;
    ss = a21 * (binv11 * binv22);
    abi22 = -ss * b12;
    pp = half * (as11 * binv11 + abi22);
    shift = s2;
  }
  qq = ss * as12;
  if (std::abs(pp * rtmin) >= one) {
    discr = ((rtmin * pp) * (rtmin * pp)) + qq * safmin;
    r = std::sqrt(std::abs(discr)) * rtmax;
  } else {
    if ((pp * pp) + std::abs(qq) <= safmin) {
      discr = ((rtmax * pp) * (rtmax * pp)) + qq * safmax;
      r = std::sqrt(std::abs(discr)) * rtmin;
    } else {
      discr = (pp * pp) + qq;
      r = std::sqrt(std::abs(discr));
    }
  }
  // Note: the test of R in the following IF is to cover the case when
  // DISCR is small and negative and is flushed to zero during
  // the calculation of R.  On machines which have a consistent
  // flush-to-zero threshold and handle numbers above that
  // threshold correctly, it would not be necessary.
  if (discr >= zero || r == zero) {
    sum = pp + std::copysign(r, pp);
    diff = pp - std::copysign(r, pp);
    wbig = shift + sum;
    // Compute smaller eigenvalue
    wsmall = shift + diff;
    if (half * std::abs(wbig) > Max(std::abs(wsmall), safmin)) {
      wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
      wsmall = wdet / wbig;
    }
    // Choose (real) eigenvalue closest to 2,2 element of A*B**(-1)
    // for WR1.
    if (pp > abi22) {
      *wr1 = Min(wbig, wsmall);
      *wr2 = Max(wbig, wsmall);
    } else {
      *wr1 = Max(wbig, wsmall);
      *wr2 = Min(wbig, wsmall);
    }
    *wi = zero;
  } else {
    // Complex eigenvalues
    *wr1 = shift + pp;
    *wr2 = *wr1;
    *wi = r;
  }
  // Further scaling to avoid underflow and overflow in computing
  // SCALE1 and overflow in computing w*B.
  // This scale factor (WSCALE) is bounded from above using C1 and C2,
  // and from below using C3 and C4.
  // C1 implements the condition  s A  must never overflow.
  // C2 implements the condition  w B  must never overflow.
  // C3, with C2,
  // implement the condition that s A - w B must never overflow.
  // C4 implements the condition  s    should not underflow.
  // C5 implements the condition  max(s,|w|) should be at least 2.
  c1 = bsize * (safmin * Max(one, ascale));
  c2 = safmin * Max(one, bnorm);
  c3 = bsize * safmin;
  if (ascale <= one && bsize <= one) {
    c4 = Min(one, (ascale / safmin) * bsize);
  } else {
    c4 = one;
  }
  if (ascale <= one || bsize <= one) {
    c5 = Min(one, ascale * bsize);
  } else {
    c5 = one;
  }
  // Scale first eigenvalue
  wabs = std::abs(*wr1) + std::abs(*wi);
  wsize =
      Max(safmin, c1, fuzzy1 * (wabs * c2 + c3), Min(c4, half * Max(wabs, c5)));
  if (wsize != one) {
    wscale = one / wsize;
    if (wsize > one) {
      *scale1 = (Max(ascale, bsize) * wscale) * Min(ascale, bsize);
    } else {
      *scale1 = (Min(ascale, bsize) * wscale) * Max(ascale, bsize);
    }
    *wr1 = *wr1 * wscale;
    if (*wi != zero) {
      *wi = *wi * wscale;
      *wr2 = *wr1;
      *scale2 = *scale1;
    }
  } else {
    *scale1 = ascale * bsize;
    *scale2 = *scale1;
  }
  // Scale second eigenvalue (if real)
  if (*wi == zero) {
    wsize = Max(safmin, c1, fuzzy1 * (std::abs(*wr2) * c2 + c3),
                Min(c4, half * Max(std::abs(*wr2), c5)));
    if (wsize != one) {
      wscale = one / wsize;
      if (wsize > one) {
        *scale2 = (Max(ascale, bsize) * wscale) * Min(ascale, bsize);
      } else {
        *scale2 = (Min(ascale, bsize) * wscale) * Max(ascale, bsize);
      }
      *wr2 = *wr2 * wscale;
    } else {
      *scale2 = ascale * bsize;
    }
  }
}

// xHGEQZ (real T): computes the eigenvalues (ALPHAR(j) + i*ALPHAI(j)) /
// BETA(j) of the pair (H, T), H upper Hessenberg and T upper triangular
// in rows and columns ILO:IHI (upper triangular outside it, as xGGHRD
// leaves them). JOB = 'E': eigenvalues only; 'S': also the generalized
// Schur form, H = Q*S*Z**T and T = Q*P*Z**T with S quasi-upper triangular
// (1-by-1 and 2-by-2 blocks, the latter standardized so that P has a
// diagonal 2-by-2 block with positive entries) in H and P upper triangular
// with a nonnegative diagonal in T. COMPQ / COMPZ = 'N': Q / Z not
// computed; 'I': initialized to the identity; 'V': Q1 / Z1 on entry are
// multiplied, Q1 * Q and Z1 * Z (as for the Q and Z of xGGHRD). Complex
// conjugate eigenvalue pairs are consecutive, the one with positive
// imaginary part first. WORK has LWORK >= max(1, N) elements (LWORK = -1:
// workspace query).
//
// Returns INFO: 0, -i for an illegal argument i, i = 1..N if the QZ
// iteration did not converge ((H, T) is not in Schur form, but
// ALPHAR(i+1:N), ALPHAI(i+1:N), BETA(i+1:N) are correct), i = N+1..2*N if
// the shift calculation failed (the same, for i-N).
template <typename R>
blasint Hgeqz(char job, char compq, char compz, blasint n, blasint ilo,
              blasint ihi, R* h, blasint ldh, R* t, blasint ldt, R* alphar,
              R* alphai, R* beta, R* q, blasint ldq, R* z, blasint ldz, R* work,
              blasint lwork) {
  using T = R;

  const R half = R(0.5e+0);
  const R zero = R(0);
  const R one = R(1);
  const R safety = R(100);
  const Vector<R> ALPHAI(alphai);
  const Vector<R> ALPHAR(alphar);
  const Vector<R> BETA(beta);
  const Matrix<R> H(h, ldh);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> Tm(t, ldt);
  R v[3];
  const Vector<R> V(v);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint icompq, icompz, ifirst, ifrstm, iiter, ilast, ilastm, in, ischur,
      istart, j, jc, jch, jiter, jr, maxit;
  bool ilazr2, ilazro, ilpivt, lquery;
  // Set below for every valid JOB, COMPQ and COMPZ.
  bool ilq = false, ilschr = false, ilz = false;
  R a11, a12, a1i, a1r, a21, a22, a2i, a2r, ad11, ad11l, ad12, ad12l, ad21,
      ad21l, ad22, ad22l, ad32l, an, anorm, ascale, atol, b11, b1a, b1i, b1r,
      b22, b2a, b2i, b2r, bn, bnorm, bscale, btol, c, c11i, c11r, c12, c21,
      c22i, c22r, cl, cq, cr, cz, eshift, s, s1, s1inv, s2, safmax, safmin,
      scale, sl, sqi, sqr, sr, szi, szr, t1, t2, t3, tau, temp, temp2, tempi,
      tempr, u1, u12, u12l, u2, ulp, vs, w11, w12, w21, w22, wabs, wi, wr, wr2;
  // Set by xLAG2 where they are read; initialized for clang-tidy.
  s1 = s2 = zero;
  if (Lsame(job, 'E')) {
    ilschr = false;
    ischur = 1;
  } else if (Lsame(job, 'S')) {
    ilschr = true;
    ischur = 2;
  } else {
    ischur = 0;
  }
  if (Lsame(compq, 'N')) {
    ilq = false;
    icompq = 1;
  } else if (Lsame(compq, 'V')) {
    ilq = true;
    icompq = 2;
  } else if (Lsame(compq, 'I')) {
    ilq = true;
    icompq = 3;
  } else {
    icompq = 0;
  }
  if (Lsame(compz, 'N')) {
    ilz = false;
    icompz = 1;
  } else if (Lsame(compz, 'V')) {
    ilz = true;
    icompz = 2;
  } else if (Lsame(compz, 'I')) {
    ilz = true;
    icompz = 3;
  } else {
    icompz = 0;
  }
  // Check Argument Values
  blasint info = 0;
  WORK(1) = R(Max<blasint>(1, n));
  lquery = (lwork == -1);
  if (ischur == 0) {
    info = -1;
  } else if (icompq == 0) {
    info = -2;
  } else if (icompz == 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (ilo < 1) {
    info = -5;
  } else if (ihi > n || ihi < ilo - 1) {
    info = -6;
  } else if (ldh < n) {
    info = -8;
  } else if (ldt < n) {
    info = -10;
  } else if (ldq < 1 || (ilq && ldq < n)) {
    info = -15;
  } else if (ldz < 1 || (ilz && ldz < n)) {
    info = -17;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -19;
  }
  if (info != 0) {
    Xerbla<T>("HGEQZ", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n <= 0) {
    WORK(1) = R(1);
    return info;
  }
  // Initialize Q and Z
  if (icompq == 3) Laset('F', n, n, zero, one, q, ldq);
  if (icompz == 3) Laset('F', n, n, zero, one, z, ldz);
  // Machine Constants
  in = ihi + 1 - ilo;
  safmin = Lamch<R>('S');
  safmax = one / safmin;
  ulp = Lamch<R>('E') * Lamch<R>('B');
  anorm = Lanhs('F', in, H.ptr(ilo, ilo), ldh, work);
  bnorm = Lanhs('F', in, Tm.ptr(ilo, ilo), ldt, work);
  atol = Max(safmin, ulp * anorm);
  btol = Max(safmin, ulp * bnorm);
  ascale = one / Max(safmin, anorm);
  bscale = one / Max(safmin, bnorm);
  // Set Eigenvalues IHI+1:N
  for (j = ihi + 1; j <= n; ++j) {
    if (Tm(j, j) < zero) {
      if (ilschr) {
        for (jr = 1; jr <= j; ++jr) {
          H(jr, j) = -H(jr, j);
          Tm(jr, j) = -Tm(jr, j);
        }
      } else {
        H(j, j) = -H(j, j);
        Tm(j, j) = -Tm(j, j);
      }
      if (ilz) {
        for (jr = 1; jr <= n; ++jr) {
          Z(jr, j) = -Z(jr, j);
        }
      }
    }
    ALPHAR(j) = H(j, j);
    ALPHAI(j) = zero;
    BETA(j) = Tm(j, j);
  }
  // If IHI < ILO, skip QZ steps
  if (ihi < ilo) goto L380;
  // MAIN QZ ITERATION LOOP
  // Initialize dynamic indices
  // Eigenvalues ILAST+1:N have been found.
  // Column operations modify rows IFRSTM:whatever.
  // Row operations modify columns whatever:ILASTM.
  // If only eigenvalues are being computed, then
  // IFRSTM is the row of the last splitting row above row ILAST;
  // this is always at least ILO.
  // IITER counts iterations since the last eigenvalue was found,
  // to tell when to use an extraordinary shift.
  // MAXIT is the maximum number of QZ sweeps allowed.
  ilast = ihi;
  if (ilschr) {
    ifrstm = 1;
    ilastm = n;
  } else {
    ifrstm = ilo;
    ilastm = ihi;
  }
  iiter = 0;
  eshift = zero;
  maxit = 30 * (ihi - ilo + 1);
  for (jiter = 1; jiter <= maxit; ++jiter) {
    // Split the matrix if possible.
    // Two tests:
    // 1: H(j,j-1)=0  or  j=ILO
    // 2: Tm(j,j)=0
    if (ilast == ilo) {
      // Special case: j=ILAST
      goto L80;
    } else {
      if (std::abs(H(ilast, ilast - 1)) <=
          Max(safmin, ulp * (std::abs(H(ilast, ilast)) +
                             std::abs(H(ilast - 1, ilast - 1))))) {
        H(ilast, ilast - 1) = zero;
        goto L80;
      }
    }
    if (std::abs(Tm(ilast, ilast)) <= btol) {
      Tm(ilast, ilast) = zero;
      goto L70;
    }
    // General case: j<ILAST
    for (j = ilast - 1; j >= ilo; --j) {
      // Test 1: for H(j,j-1)=0 or j=ILO
      if (j == ilo) {
        ilazro = true;
      } else {
        if (std::abs(H(j, j - 1)) <=
            Max(safmin,
                ulp * (std::abs(H(j, j)) + std::abs(H(j - 1, j - 1))))) {
          H(j, j - 1) = zero;
          ilazro = true;
        } else {
          ilazro = false;
        }
      }
      // Test 2: for Tm(j,j)=0
      if (std::abs(Tm(j, j)) < btol) {
        Tm(j, j) = zero;
        // Test 1a: Check for 2 consecutive small subdiagonals in A
        ilazr2 = false;
        if (!ilazro) {
          temp = std::abs(H(j, j - 1));
          temp2 = std::abs(H(j, j));
          tempr = Max(temp, temp2);
          if (tempr < one && tempr != zero) {
            temp = temp / tempr;
            temp2 = temp2 / tempr;
          }
          if (temp * (ascale * std::abs(H(j + 1, j))) <=
              temp2 * (ascale * atol))
            ilazr2 = true;
        }
        // If both tests pass (1 & 2), i.e., the leading diagonal
        // element of B in the block is zero, split a 1x1 block off
        // at the top. (I.e., at the J-th row/column) The leading
        // diagonal element of the remainder can also be zero, so
        // this may have to be done repeatedly.
        if (ilazro || ilazr2) {
          for (jch = j; jch <= ilast - 1; ++jch) {
            temp = H(jch, jch);
            Lartg(temp, H(jch + 1, jch), &c, &s, H.ptr(jch, jch));
            H(jch + 1, jch) = zero;
            Rot(ilastm - jch, H.ptr(jch, jch + 1), ldh, H.ptr(jch + 1, jch + 1),
                ldh, c, s);
            Rot(ilastm - jch, Tm.ptr(jch, jch + 1), ldt,
                Tm.ptr(jch + 1, jch + 1), ldt, c, s);
            if (ilq) Rot(n, Q.ptr(1, jch), 1, Q.ptr(1, jch + 1), 1, c, s);
            if (ilazr2) H(jch, jch - 1) = H(jch, jch - 1) * c;
            ilazr2 = false;
            if (std::abs(Tm(jch + 1, jch + 1)) >= btol) {
              if (jch + 1 >= ilast) {
                goto L80;
              } else {
                ifirst = jch + 1;
                goto L110;
              }
            }
            Tm(jch + 1, jch + 1) = zero;
          }
          goto L70;
        } else {
          // Only test 2 passed -- chase the zero to Tm(ILAST,ILAST)
          // Then process as in the case Tm(ILAST,ILAST)=0
          for (jch = j; jch <= ilast - 1; ++jch) {
            temp = Tm(jch, jch + 1);
            Lartg(temp, Tm(jch + 1, jch + 1), &c, &s, Tm.ptr(jch, jch + 1));
            Tm(jch + 1, jch + 1) = zero;
            if (jch < ilastm - 1)
              Rot(ilastm - jch - 1, Tm.ptr(jch, jch + 2), ldt,
                  Tm.ptr(jch + 1, jch + 2), ldt, c, s);
            Rot(ilastm - jch + 2, H.ptr(jch, jch - 1), ldh,
                H.ptr(jch + 1, jch - 1), ldh, c, s);
            if (ilq) Rot(n, Q.ptr(1, jch), 1, Q.ptr(1, jch + 1), 1, c, s);
            temp = H(jch + 1, jch);
            Lartg(temp, H(jch + 1, jch - 1), &c, &s, H.ptr(jch + 1, jch));
            H(jch + 1, jch - 1) = zero;
            Rot(jch + 1 - ifrstm, H.ptr(ifrstm, jch), 1, H.ptr(ifrstm, jch - 1),
                1, c, s);
            Rot(jch - ifrstm, Tm.ptr(ifrstm, jch), 1, Tm.ptr(ifrstm, jch - 1),
                1, c, s);
            if (ilz) Rot(n, Z.ptr(1, jch), 1, Z.ptr(1, jch - 1), 1, c, s);
          }
          goto L70;
        }
      } else if (ilazro) {
        // Only test 1 passed -- work on J:ILAST
        ifirst = j;
        goto L110;
      }
      // Neither test passed -- try next J
    }
    // (Drop-through is "impossible")
    info = n + 1;
    goto L420;
    // Tm(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
    // 1x1 block.
  L70:
    temp = H(ilast, ilast);
    Lartg(temp, H(ilast, ilast - 1), &c, &s, H.ptr(ilast, ilast));
    H(ilast, ilast - 1) = zero;
    Rot(ilast - ifrstm, H.ptr(ifrstm, ilast), 1, H.ptr(ifrstm, ilast - 1), 1, c,
        s);
    Rot(ilast - ifrstm, Tm.ptr(ifrstm, ilast), 1, Tm.ptr(ifrstm, ilast - 1), 1,
        c, s);
    if (ilz) Rot(n, Z.ptr(1, ilast), 1, Z.ptr(1, ilast - 1), 1, c, s);
    // H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHAR, ALPHAI,
    // and BETA
  L80:
    if (Tm(ilast, ilast) < zero) {
      if (ilschr) {
        for (j = ifrstm; j <= ilast; ++j) {
          H(j, ilast) = -H(j, ilast);
          Tm(j, ilast) = -Tm(j, ilast);
        }
      } else {
        H(ilast, ilast) = -H(ilast, ilast);
        Tm(ilast, ilast) = -Tm(ilast, ilast);
      }
      if (ilz) {
        for (j = 1; j <= n; ++j) {
          Z(j, ilast) = -Z(j, ilast);
        }
      }
    }
    ALPHAR(ilast) = H(ilast, ilast);
    ALPHAI(ilast) = zero;
    BETA(ilast) = Tm(ilast, ilast);
    // Go to next block -- exit if finished.
    ilast = ilast - 1;
    if (ilast < ilo) goto L380;
    // Reset counters
    iiter = 0;
    eshift = zero;
    if (!ilschr) {
      ilastm = ilast;
      if (ifrstm > ilast) ifrstm = ilo;
    }
    goto L350;
    // QZ step
    // This iteration only involves rows/columns IFIRST:ILAST. We
    // assume IFIRST < ILAST, and that the diagonal of B is non-zero.
  L110:
    iiter = iiter + 1;
    if (!ilschr) {
      ifrstm = ifirst;
    }
    // Compute single shifts.
    // At this point, IFIRST < ILAST, and the diagonal elements of
    // Tm(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in
    // magnitude)
    if ((iiter / 10) * 10 == iiter) {
      // Exceptional shift.  Chosen for no particularly good reason.
      // (Single shift only.)
      if ((R(maxit) * safmin) * std::abs(H(ilast, ilast - 1)) <
          std::abs(Tm(ilast - 1, ilast - 1))) {
        eshift = H(ilast, ilast - 1) / Tm(ilast - 1, ilast - 1);
      } else {
        eshift = eshift + one / (safmin * R(maxit));
      }
      s1 = one;
      wr = eshift;
    } else {
      // Shifts based on the generalized eigenvalues of the
      // bottom-right 2x2 block of A and B. The first eigenvalue
      // returned by DLAG2 is the Wilkinson shift (AEP p.512),
      Lag2<R>(H.ptr(ilast - 1, ilast - 1), ldh, Tm.ptr(ilast - 1, ilast - 1),
              ldt, safmin * safety, &s1, &s2, &wr, &wr2, &wi);
      if (std::abs((wr / s1) * Tm(ilast, ilast) - H(ilast, ilast)) >
          std::abs((wr2 / s2) * Tm(ilast, ilast) - H(ilast, ilast))) {
        temp = wr;
        wr = wr2;
        wr2 = temp;
        temp = s1;
        s1 = s2;
        s2 = temp;
      }
      temp = Max(s1, safmin * Max(one, std::abs(wr), std::abs(wi)));
      if (wi != zero) goto L200;
    }
    // Fiddle with shift to avoid overflow
    temp = Min(ascale, one) * (half * safmax);
    if (s1 > temp) {
      scale = temp / s1;
    } else {
      scale = one;
    }
    temp = Min(bscale, one) * (half * safmax);
    if (std::abs(wr) > temp) scale = Min(scale, temp / std::abs(wr));
    s1 = scale * s1;
    wr = scale * wr;
    // Now check for two consecutive small subdiagonals.
    for (j = ilast - 1; j >= ifirst + 1; --j) {
      istart = j;
      temp = std::abs(s1 * H(j, j - 1));
      temp2 = std::abs(s1 * H(j, j) - wr * Tm(j, j));
      tempr = Max(temp, temp2);
      if (tempr < one && tempr != zero) {
        temp = temp / tempr;
        temp2 = temp2 / tempr;
      }
      if (std::abs((ascale * H(j + 1, j)) * temp) <= (ascale * atol) * temp2)
        goto L130;
    }
    istart = ifirst;
  L130:
    // Do an implicit single-shift QZ sweep.
    // Initial Q
    temp = s1 * H(istart, istart) - wr * Tm(istart, istart);
    temp2 = s1 * H(istart + 1, istart);
    Lartg(temp, temp2, &c, &s, &tempr);
    // Sweep
    for (j = istart; j <= ilast - 1; ++j) {
      if (j > istart) {
        temp = H(j, j - 1);
        Lartg(temp, H(j + 1, j - 1), &c, &s, H.ptr(j, j - 1));
        H(j + 1, j - 1) = zero;
      }
      for (jc = j; jc <= ilastm; ++jc) {
        temp = c * H(j, jc) + s * H(j + 1, jc);
        H(j + 1, jc) = -s * H(j, jc) + c * H(j + 1, jc);
        H(j, jc) = temp;
        temp2 = c * Tm(j, jc) + s * Tm(j + 1, jc);
        Tm(j + 1, jc) = -s * Tm(j, jc) + c * Tm(j + 1, jc);
        Tm(j, jc) = temp2;
      }
      if (ilq) {
        for (jr = 1; jr <= n; ++jr) {
          temp = c * Q(jr, j) + s * Q(jr, j + 1);
          Q(jr, j + 1) = -s * Q(jr, j) + c * Q(jr, j + 1);
          Q(jr, j) = temp;
        }
      }
      temp = Tm(j + 1, j + 1);
      Lartg(temp, Tm(j + 1, j), &c, &s, Tm.ptr(j + 1, j + 1));
      Tm(j + 1, j) = zero;
      for (jr = ifrstm; jr <= Min<blasint>(j + 2, ilast); ++jr) {
        temp = c * H(jr, j + 1) + s * H(jr, j);
        H(jr, j) = -s * H(jr, j + 1) + c * H(jr, j);
        H(jr, j + 1) = temp;
      }
      for (jr = ifrstm; jr <= j; ++jr) {
        temp = c * Tm(jr, j + 1) + s * Tm(jr, j);
        Tm(jr, j) = -s * Tm(jr, j + 1) + c * Tm(jr, j);
        Tm(jr, j + 1) = temp;
      }
      if (ilz) {
        for (jr = 1; jr <= n; ++jr) {
          temp = c * Z(jr, j + 1) + s * Z(jr, j);
          Z(jr, j) = -s * Z(jr, j + 1) + c * Z(jr, j);
          Z(jr, j + 1) = temp;
        }
      }
    }
    goto L350;
    // Use Francis double-shift
    // Note: the Francis double-shift should work with real shifts,
    // but only if the block is at least 3x3.
    // This code may break if this point is reached with
    // a 2x2 block with real eigenvalues.
  L200:
    if (ifirst + 1 == ilast) {
      // Special case -- 2x2 block with complex eigenvectors
      // Step 1: Standardize, that is, rotate so that
      // ( B11  0  )
      // B = (         )  with B11 non-negative.
      // (  0  B22 )
      Lasv2(Tm(ilast - 1, ilast - 1), Tm(ilast - 1, ilast), Tm(ilast, ilast),
            &b22, &b11, &sr, &cr, &sl, &cl);
      if (b11 < zero) {
        cr = -cr;
        sr = -sr;
        b11 = -b11;
        b22 = -b22;
      }
      Rot(ilastm + 1 - ifirst, H.ptr(ilast - 1, ilast - 1), ldh,
          H.ptr(ilast, ilast - 1), ldh, cl, sl);
      Rot(ilast + 1 - ifrstm, H.ptr(ifrstm, ilast - 1), 1, H.ptr(ifrstm, ilast),
          1, cr, sr);
      if (ilast < ilastm)
        Rot(ilastm - ilast, Tm.ptr(ilast - 1, ilast + 1), ldt,
            Tm.ptr(ilast, ilast + 1), ldt, cl, sl);
      if (ifrstm < ilast - 1)
        Rot(ifirst - ifrstm, Tm.ptr(ifrstm, ilast - 1), 1,
            Tm.ptr(ifrstm, ilast), 1, cr, sr);
      if (ilq) Rot(n, Q.ptr(1, ilast - 1), 1, Q.ptr(1, ilast), 1, cl, sl);
      if (ilz) Rot(n, Z.ptr(1, ilast - 1), 1, Z.ptr(1, ilast), 1, cr, sr);
      Tm(ilast - 1, ilast - 1) = b11;
      Tm(ilast - 1, ilast) = zero;
      Tm(ilast, ilast - 1) = zero;
      Tm(ilast, ilast) = b22;
      // If B22 is negative, negate column ILAST
      if (b22 < zero) {
        for (j = ifrstm; j <= ilast; ++j) {
          H(j, ilast) = -H(j, ilast);
          Tm(j, ilast) = -Tm(j, ilast);
        }
        if (ilz) {
          for (j = 1; j <= n; ++j) {
            Z(j, ilast) = -Z(j, ilast);
          }
        }
        b22 = -b22;
      }
      // Step 2: Compute ALPHAR, ALPHAI, and BETA (see refs.)
      // Recompute shift
      Lag2<R>(H.ptr(ilast - 1, ilast - 1), ldh, Tm.ptr(ilast - 1, ilast - 1),
              ldt, safmin * safety, &s1, &temp, &wr, &temp2, &wi);
      // If standardization has perturbed the shift onto real line,
      // do another (real single-shift) QR step.
      if (wi == zero) goto L350;
      s1inv = one / s1;
      // Do EISPACK (QZVAL) computation of alpha and beta
      a11 = H(ilast - 1, ilast - 1);
      a21 = H(ilast, ilast - 1);
      a12 = H(ilast - 1, ilast);
      a22 = H(ilast, ilast);
      // Compute complex Givens rotation on right
      // (Assume some element of C = (sA - wB) > unfl )
      // __
      // (sA - wB) ( CZ   -SZ )
      // ( SZ    CZ )
      c11r = s1 * a11 - wr * b11;
      c11i = -wi * b11;
      c12 = s1 * a12;
      c21 = s1 * a21;
      c22r = s1 * a22 - wr * b22;
      c22i = -wi * b22;
      if (std::abs(c11r) + std::abs(c11i) + std::abs(c12) >
          std::abs(c21) + std::abs(c22r) + std::abs(c22i)) {
        t1 = Lapy3(c12, c11r, c11i);
        cz = c12 / t1;
        szr = -c11r / t1;
        szi = -c11i / t1;
      } else {
        cz = Lapy2(c22r, c22i);
        if (cz <= safmin) {
          cz = zero;
          szr = one;
          szi = zero;
        } else {
          tempr = c22r / cz;
          tempi = c22i / cz;
          t1 = Lapy2(cz, c21);
          cz = cz / t1;
          szr = -c21 * tempr / t1;
          szi = c21 * tempi / t1;
        }
      }
      // Compute Givens rotation on left
      // (  CQ   SQ )
      // (  __      )  A or B
      // ( -SQ   CQ )
      an = std::abs(a11) + std::abs(a12) + std::abs(a21) + std::abs(a22);
      bn = std::abs(b11) + std::abs(b22);
      wabs = std::abs(wr) + std::abs(wi);
      if (s1 * an > wabs * bn) {
        cq = cz * b11;
        sqr = szr * b22;
        sqi = -szi * b22;
      } else {
        a1r = cz * a11 + szr * a12;
        a1i = szi * a12;
        a2r = cz * a21 + szr * a22;
        a2i = szi * a22;
        cq = Lapy2(a1r, a1i);
        if (cq <= safmin) {
          cq = zero;
          sqr = one;
          sqi = zero;
        } else {
          tempr = a1r / cq;
          tempi = a1i / cq;
          sqr = tempr * a2r + tempi * a2i;
          sqi = tempi * a2r - tempr * a2i;
        }
      }
      t1 = Lapy3(cq, sqr, sqi);
      cq = cq / t1;
      sqr = sqr / t1;
      sqi = sqi / t1;
      // Compute diagonal elements of QBZ
      tempr = sqr * szr - sqi * szi;
      tempi = sqr * szi + sqi * szr;
      b1r = cq * cz * b11 + tempr * b22;
      b1i = tempi * b22;
      b1a = Lapy2(b1r, b1i);
      b2r = cq * cz * b22 + tempr * b11;
      b2i = -tempi * b11;
      b2a = Lapy2(b2r, b2i);
      // Normalize so beta > 0, and Im( alpha1 ) > 0
      BETA(ilast - 1) = b1a;
      BETA(ilast) = b2a;
      ALPHAR(ilast - 1) = (wr * b1a) * s1inv;
      ALPHAI(ilast - 1) = (wi * b1a) * s1inv;
      ALPHAR(ilast) = (wr * b2a) * s1inv;
      ALPHAI(ilast) = -(wi * b2a) * s1inv;
      // Step 3: Go to next block -- exit if finished.
      ilast = ifirst - 1;
      if (ilast < ilo) goto L380;
      // Reset counters
      iiter = 0;
      eshift = zero;
      if (!ilschr) {
        ilastm = ilast;
        if (ifrstm > ilast) ifrstm = ilo;
      }
      goto L350;
    } else {
      // Usual case: 3x3 or larger block, using Francis implicit
      // double-shift
      // 2
      // Eigenvalue equation is  w  - c w + d = 0,
      // -1 2        -1
      // so compute 1st column of  (A B  )  - c A B   + d
      // using the formula in QZIT (from EISPACK)
      // We assume that the block is at least 3x3
      ad11 = (ascale * H(ilast - 1, ilast - 1)) /
             (bscale * Tm(ilast - 1, ilast - 1));
      ad21 =
          (ascale * H(ilast, ilast - 1)) / (bscale * Tm(ilast - 1, ilast - 1));
      ad12 = (ascale * H(ilast - 1, ilast)) / (bscale * Tm(ilast, ilast));
      ad22 = (ascale * H(ilast, ilast)) / (bscale * Tm(ilast, ilast));
      u12 = Tm(ilast - 1, ilast) / Tm(ilast, ilast);
      ad11l = (ascale * H(ifirst, ifirst)) / (bscale * Tm(ifirst, ifirst));
      ad21l = (ascale * H(ifirst + 1, ifirst)) / (bscale * Tm(ifirst, ifirst));
      ad12l = (ascale * H(ifirst, ifirst + 1)) /
              (bscale * Tm(ifirst + 1, ifirst + 1));
      ad22l = (ascale * H(ifirst + 1, ifirst + 1)) /
              (bscale * Tm(ifirst + 1, ifirst + 1));
      ad32l = (ascale * H(ifirst + 2, ifirst + 1)) /
              (bscale * Tm(ifirst + 1, ifirst + 1));
      u12l = Tm(ifirst, ifirst + 1) / Tm(ifirst + 1, ifirst + 1);
      V(1) = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21 +
             ad21 * u12 * ad11l + (ad12l - ad11l * u12l) * ad21l;
      V(2) = ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) - (ad22 - ad11l) +
              ad21 * u12) *
             ad21l;
      V(3) = ad32l * ad21l;
      istart = ifirst;
      Larfg(3, V.ptr(1), V.ptr(2), 1, &tau);
      V(1) = one;
      // Sweep
      for (j = istart; j <= ilast - 2; ++j) {
        // All but last elements: use 3x3 Householder transforms.
        // Zero (j-1)st column of A
        if (j > istart) {
          V(1) = H(j, j - 1);
          V(2) = H(j + 1, j - 1);
          V(3) = H(j + 2, j - 1);
          Larfg(3, H.ptr(j, j - 1), V.ptr(2), 1, &tau);
          V(1) = one;
          H(j + 1, j - 1) = zero;
          H(j + 2, j - 1) = zero;
        }
        t2 = tau * V(2);
        t3 = tau * V(3);
        for (jc = j; jc <= ilastm; ++jc) {
          temp = H(j, jc) + V(2) * H(j + 1, jc) + V(3) * H(j + 2, jc);
          H(j, jc) = H(j, jc) - temp * tau;
          H(j + 1, jc) = H(j + 1, jc) - temp * t2;
          H(j + 2, jc) = H(j + 2, jc) - temp * t3;
          temp2 = Tm(j, jc) + V(2) * Tm(j + 1, jc) + V(3) * Tm(j + 2, jc);
          Tm(j, jc) = Tm(j, jc) - temp2 * tau;
          Tm(j + 1, jc) = Tm(j + 1, jc) - temp2 * t2;
          Tm(j + 2, jc) = Tm(j + 2, jc) - temp2 * t3;
        }
        if (ilq) {
          for (jr = 1; jr <= n; ++jr) {
            temp = Q(jr, j) + V(2) * Q(jr, j + 1) + V(3) * Q(jr, j + 2);
            Q(jr, j) = Q(jr, j) - temp * tau;
            Q(jr, j + 1) = Q(jr, j + 1) - temp * t2;
            Q(jr, j + 2) = Q(jr, j + 2) - temp * t3;
          }
        }
        // Zero j-th column of B (see DLAGBC for details)
        // Swap rows to pivot
        ilpivt = false;
        temp = Max(std::abs(Tm(j + 1, j + 1)), std::abs(Tm(j + 1, j + 2)));
        temp2 = Max(std::abs(Tm(j + 2, j + 1)), std::abs(Tm(j + 2, j + 2)));
        if (Max(temp, temp2) < safmin) {
          scale = zero;
          u1 = one;
          u2 = zero;
          goto L250;
        } else if (temp >= temp2) {
          w11 = Tm(j + 1, j + 1);
          w21 = Tm(j + 2, j + 1);
          w12 = Tm(j + 1, j + 2);
          w22 = Tm(j + 2, j + 2);
          u1 = Tm(j + 1, j);
          u2 = Tm(j + 2, j);
        } else {
          w21 = Tm(j + 1, j + 1);
          w11 = Tm(j + 2, j + 1);
          w22 = Tm(j + 1, j + 2);
          w12 = Tm(j + 2, j + 2);
          u2 = Tm(j + 1, j);
          u1 = Tm(j + 2, j);
        }
        // Swap columns if nec.
        if (std::abs(w12) > std::abs(w11)) {
          ilpivt = true;
          temp = w12;
          temp2 = w22;
          w12 = w11;
          w22 = w21;
          w11 = temp;
          w21 = temp2;
        }
        // LU-factor
        temp = w21 / w11;
        u2 = u2 - temp * u1;
        w22 = w22 - temp * w12;
        w21 = zero;
        // Compute SCALE
        scale = one;
        if (std::abs(w22) < safmin) {
          scale = zero;
          u2 = one;
          u1 = -w12 / w11;
          goto L250;
        }
        if (std::abs(w22) < std::abs(u2)) scale = std::abs(w22 / u2);
        if (std::abs(w11) < std::abs(u1))
          scale = Min(scale, std::abs(w11 / u1));
        // Solve
        u2 = (scale * u2) / w22;
        u1 = (scale * u1 - w12 * u2) / w11;
      L250:
        if (ilpivt) {
          temp = u2;
          u2 = u1;
          u1 = temp;
        }
        // Compute Householder Vector
        t1 = std::sqrt((scale * scale) + (u1 * u1) + (u2 * u2));
        tau = one + scale / t1;
        vs = -one / (scale + t1);
        V(1) = one;
        V(2) = vs * u1;
        V(3) = vs * u2;
        // Apply transformations from the right.
        t2 = tau * V(2);
        t3 = tau * V(3);
        for (jr = ifrstm; jr <= Min<blasint>(j + 3, ilast); ++jr) {
          temp = H(jr, j) + V(2) * H(jr, j + 1) + V(3) * H(jr, j + 2);
          H(jr, j) = H(jr, j) - temp * tau;
          H(jr, j + 1) = H(jr, j + 1) - temp * t2;
          H(jr, j + 2) = H(jr, j + 2) - temp * t3;
        }
        for (jr = ifrstm; jr <= j + 2; ++jr) {
          temp = Tm(jr, j) + V(2) * Tm(jr, j + 1) + V(3) * Tm(jr, j + 2);
          Tm(jr, j) = Tm(jr, j) - temp * tau;
          Tm(jr, j + 1) = Tm(jr, j + 1) - temp * t2;
          Tm(jr, j + 2) = Tm(jr, j + 2) - temp * t3;
        }
        if (ilz) {
          for (jr = 1; jr <= n; ++jr) {
            temp = Z(jr, j) + V(2) * Z(jr, j + 1) + V(3) * Z(jr, j + 2);
            Z(jr, j) = Z(jr, j) - temp * tau;
            Z(jr, j + 1) = Z(jr, j + 1) - temp * t2;
            Z(jr, j + 2) = Z(jr, j + 2) - temp * t3;
          }
        }
        Tm(j + 1, j) = zero;
        Tm(j + 2, j) = zero;
      }
      // Last elements: Use Givens rotations
      // Rotations from the left
      j = ilast - 1;
      temp = H(j, j - 1);
      Lartg(temp, H(j + 1, j - 1), &c, &s, H.ptr(j, j - 1));
      H(j + 1, j - 1) = zero;
      for (jc = j; jc <= ilastm; ++jc) {
        temp = c * H(j, jc) + s * H(j + 1, jc);
        H(j + 1, jc) = -s * H(j, jc) + c * H(j + 1, jc);
        H(j, jc) = temp;
        temp2 = c * Tm(j, jc) + s * Tm(j + 1, jc);
        Tm(j + 1, jc) = -s * Tm(j, jc) + c * Tm(j + 1, jc);
        Tm(j, jc) = temp2;
      }
      if (ilq) {
        for (jr = 1; jr <= n; ++jr) {
          temp = c * Q(jr, j) + s * Q(jr, j + 1);
          Q(jr, j + 1) = -s * Q(jr, j) + c * Q(jr, j + 1);
          Q(jr, j) = temp;
        }
      }
      // Rotations from the right.
      temp = Tm(j + 1, j + 1);
      Lartg(temp, Tm(j + 1, j), &c, &s, Tm.ptr(j + 1, j + 1));
      Tm(j + 1, j) = zero;
      for (jr = ifrstm; jr <= ilast; ++jr) {
        temp = c * H(jr, j + 1) + s * H(jr, j);
        H(jr, j) = -s * H(jr, j + 1) + c * H(jr, j);
        H(jr, j + 1) = temp;
      }
      for (jr = ifrstm; jr <= ilast - 1; ++jr) {
        temp = c * Tm(jr, j + 1) + s * Tm(jr, j);
        Tm(jr, j) = -s * Tm(jr, j + 1) + c * Tm(jr, j);
        Tm(jr, j + 1) = temp;
      }
      if (ilz) {
        for (jr = 1; jr <= n; ++jr) {
          temp = c * Z(jr, j + 1) + s * Z(jr, j);
          Z(jr, j) = -s * Z(jr, j + 1) + c * Z(jr, j);
          Z(jr, j + 1) = temp;
        }
      }
      // End of Double-Shift code
    }
    goto L350;
    // End of iteration loop
  L350:;
  }
  // Drop-through = non-convergence
  info = ilast;
  goto L420;
  // Successful completion of all QZ steps
L380:
  // Set Eigenvalues 1:ILO-1
  for (j = 1; j <= ilo - 1; ++j) {
    if (Tm(j, j) < zero) {
      if (ilschr) {
        for (jr = 1; jr <= j; ++jr) {
          H(jr, j) = -H(jr, j);
          Tm(jr, j) = -Tm(jr, j);
        }
      } else {
        H(j, j) = -H(j, j);
        Tm(j, j) = -Tm(j, j);
      }
      if (ilz) {
        for (jr = 1; jr <= n; ++jr) {
          Z(jr, j) = -Z(jr, j);
        }
      }
    }
    ALPHAR(j) = H(j, j);
    ALPHAI(j) = zero;
    BETA(j) = Tm(j, j);
  }
  // Normal Termination
  info = 0;
  // Exit (other than argument error) -- return optimal workspace size
L420:
  WORK(1) = WorkSize<T>(n);
  return info;
}

// xHGEQZ (complex T): as the real routine, for ALPHA(j) / BETA(j); the
// generalized Schur form is upper triangular (H = Q*S*Z**H, T = Q*P*Z**H,
// P with a real nonnegative diagonal). WORK has LWORK >= max(1, N)
// elements, RWORK N.
template <typename T>
blasint Hgeqz(char job, char compq, char compz, blasint n, blasint ilo,
              blasint ihi, T* h, blasint ldh, T* t, blasint ldt, T* alpha,
              T* beta, T* q, blasint ldq, T* z, blasint ldz, T* work,
              blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;

  const T czero = T(0);
  const T cone = T(1);
  const R zero = R(0);
  const R one = R(1);
  const R half = R(0.5e+0);
  const Vector<T> ALPHA(alpha);
  const Vector<T> BETA(beta);
  const Matrix<T> H(h, ldh);
  const Matrix<T> Q(q, ldq);
  const Vector<R> RWORK(rwork);
  const Matrix<T> Tm(t, ldt);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint icompq, icompz, ifirst, ifrstm, iiter, ilast, ilastm, in, ischur,
      istart, j, jc, jch, jiter, jr, maxit;
  bool ilazr2, ilazro, ilq, ilschr, ilz, lquery;
  R absb, anorm, ascale, atol, bnorm, bscale, btol, c, safmin, temp, temp2,
      tempr, ulp;
  T abi12, abi22, ad11, ad12, ad21, ad22, ctemp, ctemp2, ctemp3, eshift, s,
      shift, signbc, u12, x, y;
  if (Lsame(job, 'E')) {
    ilschr = false;
    ischur = 1;
  } else if (Lsame(job, 'S')) {
    ilschr = true;
    ischur = 2;
  } else {
    ilschr = true;
    ischur = 0;
  }
  if (Lsame(compq, 'N')) {
    ilq = false;
    icompq = 1;
  } else if (Lsame(compq, 'V')) {
    ilq = true;
    icompq = 2;
  } else if (Lsame(compq, 'I')) {
    ilq = true;
    icompq = 3;
  } else {
    ilq = true;
    icompq = 0;
  }
  if (Lsame(compz, 'N')) {
    ilz = false;
    icompz = 1;
  } else if (Lsame(compz, 'V')) {
    ilz = true;
    icompz = 2;
  } else if (Lsame(compz, 'I')) {
    ilz = true;
    icompz = 3;
  } else {
    ilz = true;
    icompz = 0;
  }
  // Check Argument Values
  blasint info = 0;
  WORK(1) = WorkSize<T>(Max<blasint>(1, n));
  lquery = (lwork == -1);
  if (ischur == 0) {
    info = -1;
  } else if (icompq == 0) {
    info = -2;
  } else if (icompz == 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (ilo < 1) {
    info = -5;
  } else if (ihi > n || ihi < ilo - 1) {
    info = -6;
  } else if (ldh < n) {
    info = -8;
  } else if (ldt < n) {
    info = -10;
  } else if (ldq < 1 || (ilq && ldq < n)) {
    info = -14;
  } else if (ldz < 1 || (ilz && ldz < n)) {
    info = -16;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -18;
  }
  if (info != 0) {
    Xerbla<T>("HGEQZ", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  // WORK( 1 ) = CMPLX( 1 )
  if (n <= 0) {
    WORK(1) = WorkSize<T>(1);
    return info;
  }
  // Initialize Q and Z
  if (icompq == 3) Laset('F', n, n, czero, cone, q, ldq);
  if (icompz == 3) Laset('F', n, n, czero, cone, z, ldz);
  // Machine Constants
  in = ihi + 1 - ilo;
  safmin = Lamch<R>('S');
  ulp = Lamch<R>('E') * Lamch<R>('B');
  anorm = Lanhs('F', in, H.ptr(ilo, ilo), ldh, rwork);
  bnorm = Lanhs('F', in, Tm.ptr(ilo, ilo), ldt, rwork);
  atol = Max(safmin, ulp * anorm);
  btol = Max(safmin, ulp * bnorm);
  ascale = one / Max(safmin, anorm);
  bscale = one / Max(safmin, bnorm);
  // Set Eigenvalues IHI+1:N
  for (j = ihi + 1; j <= n; ++j) {
    absb = std::abs(Tm(j, j));
    if (absb > safmin) {
      signbc = Conj(Tm(j, j) / absb);
      Tm(j, j) = absb;
      if (ilschr) {
        Scal(j - 1, signbc, Tm.ptr(1, j), 1);
        Scal(j, signbc, H.ptr(1, j), 1);
      } else {
        Scal(1, signbc, H.ptr(j, j), 1);
      }
      if (ilz) Scal(n, signbc, Z.ptr(1, j), 1);
    } else {
      Tm(j, j) = czero;
    }
    ALPHA(j) = H(j, j);
    BETA(j) = Tm(j, j);
  }
  // If IHI < ILO, skip QZ steps
  if (ihi < ilo) goto L190;
  // MAIN QZ ITERATION LOOP
  // Initialize dynamic indices
  // Eigenvalues ILAST+1:N have been found.
  // Column operations modify rows IFRSTM:whatever
  // Row operations modify columns whatever:ILASTM
  // If only eigenvalues are being computed, then
  // IFRSTM is the row of the last splitting row above row ILAST;
  // this is always at least ILO.
  // IITER counts iterations since the last eigenvalue was found,
  // to tell when to use an extraordinary shift.
  // MAXIT is the maximum number of QZ sweeps allowed.
  ilast = ihi;
  if (ilschr) {
    ifrstm = 1;
    ilastm = n;
  } else {
    ifrstm = ilo;
    ilastm = ihi;
  }
  iiter = 0;
  eshift = czero;
  maxit = 30 * (ihi - ilo + 1);
  for (jiter = 1; jiter <= maxit; ++jiter) {
    // Check for too many iterations.
    if (jiter > maxit) goto L180;
    // Split the matrix if possible.
    // Two tests:
    // 1: H(j,j-1)=0  or  j=ILO
    // 2: Tm(j,j)=0
    // Special case: j=ILAST
    if (ilast == ilo) {
      goto L60;
    } else {
      if (Cabs1(H(ilast, ilast - 1)) <=
          Max(safmin, ulp * (Cabs1(H(ilast, ilast)) +
                             Cabs1(H(ilast - 1, ilast - 1))))) {
        H(ilast, ilast - 1) = czero;
        goto L60;
      }
    }
    if (std::abs(Tm(ilast, ilast)) <= btol) {
      Tm(ilast, ilast) = czero;
      goto L50;
    }
    // General case: j<ILAST
    for (j = ilast - 1; j >= ilo; --j) {
      // Test 1: for H(j,j-1)=0 or j=ILO
      if (j == ilo) {
        ilazro = true;
      } else {
        if (Cabs1(H(j, j - 1)) <=
            Max(safmin, ulp * (Cabs1(H(j, j)) + Cabs1(H(j - 1, j - 1))))) {
          H(j, j - 1) = czero;
          ilazro = true;
        } else {
          ilazro = false;
        }
      }
      // Test 2: for Tm(j,j)=0
      if (std::abs(Tm(j, j)) < btol) {
        Tm(j, j) = czero;
        // Test 1a: Check for 2 consecutive small subdiagonals in A
        ilazr2 = false;
        if (!ilazro) {
          if (Cabs1(H(j, j - 1)) * (ascale * Cabs1(H(j + 1, j))) <=
              Cabs1(H(j, j)) * (ascale * atol))
            ilazr2 = true;
        }
        // If both tests pass (1 & 2), i.e., the leading diagonal
        // element of B in the block is zero, split a 1x1 block off
        // at the top. (I.e., at the J-th row/column) The leading
        // diagonal element of the remainder can also be zero, so
        // this may have to be done repeatedly.
        if (ilazro || ilazr2) {
          for (jch = j; jch <= ilast - 1; ++jch) {
            ctemp = H(jch, jch);
            Lartg(ctemp, H(jch + 1, jch), &c, &s, H.ptr(jch, jch));
            H(jch + 1, jch) = czero;
            Rot(ilastm - jch, H.ptr(jch, jch + 1), ldh, H.ptr(jch + 1, jch + 1),
                ldh, c, s);
            Rot(ilastm - jch, Tm.ptr(jch, jch + 1), ldt,
                Tm.ptr(jch + 1, jch + 1), ldt, c, s);
            if (ilq) Rot(n, Q.ptr(1, jch), 1, Q.ptr(1, jch + 1), 1, c, Conj(s));
            if (ilazr2) H(jch, jch - 1) = H(jch, jch - 1) * c;
            ilazr2 = false;
            if (Cabs1(Tm(jch + 1, jch + 1)) >= btol) {
              if (jch + 1 >= ilast) {
                goto L60;
              } else {
                ifirst = jch + 1;
                goto L70;
              }
            }
            Tm(jch + 1, jch + 1) = czero;
          }
          goto L50;
        } else {
          // Only test 2 passed -- chase the zero to Tm(ILAST,ILAST)
          // Then process as in the case Tm(ILAST,ILAST)=0
          for (jch = j; jch <= ilast - 1; ++jch) {
            ctemp = Tm(jch, jch + 1);
            Lartg(ctemp, Tm(jch + 1, jch + 1), &c, &s, Tm.ptr(jch, jch + 1));
            Tm(jch + 1, jch + 1) = czero;
            if (jch < ilastm - 1)
              Rot(ilastm - jch - 1, Tm.ptr(jch, jch + 2), ldt,
                  Tm.ptr(jch + 1, jch + 2), ldt, c, s);
            Rot(ilastm - jch + 2, H.ptr(jch, jch - 1), ldh,
                H.ptr(jch + 1, jch - 1), ldh, c, s);
            if (ilq) Rot(n, Q.ptr(1, jch), 1, Q.ptr(1, jch + 1), 1, c, Conj(s));
            ctemp = H(jch + 1, jch);
            Lartg(ctemp, H(jch + 1, jch - 1), &c, &s, H.ptr(jch + 1, jch));
            H(jch + 1, jch - 1) = czero;
            Rot(jch + 1 - ifrstm, H.ptr(ifrstm, jch), 1, H.ptr(ifrstm, jch - 1),
                1, c, s);
            Rot(jch - ifrstm, Tm.ptr(ifrstm, jch), 1, Tm.ptr(ifrstm, jch - 1),
                1, c, s);
            if (ilz) Rot(n, Z.ptr(1, jch), 1, Z.ptr(1, jch - 1), 1, c, s);
          }
          goto L50;
        }
      } else if (ilazro) {
        // Only test 1 passed -- work on J:ILAST
        ifirst = j;
        goto L70;
      }
      // Neither test passed -- try next J
    }
    // (Drop-through is "impossible")
    info = 2 * n + 1;
    goto L210;
    // Tm(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
    // 1x1 block.
  L50:
    ctemp = H(ilast, ilast);
    Lartg(ctemp, H(ilast, ilast - 1), &c, &s, H.ptr(ilast, ilast));
    H(ilast, ilast - 1) = czero;
    Rot(ilast - ifrstm, H.ptr(ifrstm, ilast), 1, H.ptr(ifrstm, ilast - 1), 1, c,
        s);
    Rot(ilast - ifrstm, Tm.ptr(ifrstm, ilast), 1, Tm.ptr(ifrstm, ilast - 1), 1,
        c, s);
    if (ilz) Rot(n, Z.ptr(1, ilast), 1, Z.ptr(1, ilast - 1), 1, c, s);
    // H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHA and BETA
  L60:
    absb = std::abs(Tm(ilast, ilast));
    if (absb > safmin) {
      signbc = Conj(Tm(ilast, ilast) / absb);
      Tm(ilast, ilast) = absb;
      if (ilschr) {
        Scal(ilast - ifrstm, signbc, Tm.ptr(ifrstm, ilast), 1);
        Scal(ilast + 1 - ifrstm, signbc, H.ptr(ifrstm, ilast), 1);
      } else {
        Scal(1, signbc, H.ptr(ilast, ilast), 1);
      }
      if (ilz) Scal(n, signbc, Z.ptr(1, ilast), 1);
    } else {
      Tm(ilast, ilast) = czero;
    }
    ALPHA(ilast) = H(ilast, ilast);
    BETA(ilast) = Tm(ilast, ilast);
    // Go to next block -- exit if finished.
    ilast = ilast - 1;
    if (ilast < ilo) goto L190;
    // Reset counters
    iiter = 0;
    eshift = czero;
    if (!ilschr) {
      ilastm = ilast;
      if (ifrstm > ilast) ifrstm = ilo;
    }
    goto L160;
    // QZ step
    // This iteration only involves rows/columns IFIRST:ILAST.  We
    // assume IFIRST < ILAST, and that the diagonal of B is non-zero.
  L70:
    iiter = iiter + 1;
    if (!ilschr) {
      ifrstm = ifirst;
    }
    // Compute the Shift.
    // At this point, IFIRST < ILAST, and the diagonal elements of
    // Tm(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in
    // magnitude)
    if ((iiter / 10) * 10 != iiter) {
      // The Wilkinson shift (AEP p.512), i.e., the eigenvalue of
      // the bottom-right 2x2 block of A inv(B) which is nearest to
      // the bottom-right element.
      // We factor B as U*D, where U has unit diagonals, and
      // compute (A*inv(D))*inv(U).
      u12 = (bscale * Tm(ilast - 1, ilast)) / (bscale * Tm(ilast, ilast));
      ad11 = (ascale * H(ilast - 1, ilast - 1)) /
             (bscale * Tm(ilast - 1, ilast - 1));
      ad21 =
          (ascale * H(ilast, ilast - 1)) / (bscale * Tm(ilast - 1, ilast - 1));
      ad12 = (ascale * H(ilast - 1, ilast)) / (bscale * Tm(ilast, ilast));
      ad22 = (ascale * H(ilast, ilast)) / (bscale * Tm(ilast, ilast));
      abi22 = ad22 - u12 * ad21;
      abi12 = ad12 - u12 * ad11;
      shift = abi22;
      ctemp = std::sqrt(abi12) * std::sqrt(ad21);
      temp = Cabs1(ctemp);
      if (ctemp != zero) {
        x = half * (ad11 - shift);
        temp2 = Cabs1(x);
        temp = Max(temp, Cabs1(x));
        y = temp * std::sqrt(((x / temp) * (x / temp)) +
                             ((ctemp / temp) * (ctemp / temp)));
        if (temp2 > zero) {
          if (std::real(x / temp2) * std::real(y) +
                  std::imag(x / temp2) * std::imag(y) <
              zero)
            y = -y;
        }
        shift = shift - ctemp * Ladiv(ctemp, (x + y));
      }
    } else {
      // Exceptional shift.  Chosen for no particularly good reason.
      if ((iiter / 20) * 20 == iiter &&
          bscale * Cabs1(Tm(ilast, ilast)) > safmin) {
        eshift =
            eshift + (ascale * H(ilast, ilast)) / (bscale * Tm(ilast, ilast));
      } else {
        eshift = eshift + (ascale * H(ilast, ilast - 1)) /
                              (bscale * Tm(ilast - 1, ilast - 1));
      }
      shift = eshift;
    }
    // Now check for two consecutive small subdiagonals.
    for (j = ilast - 1; j >= ifirst + 1; --j) {
      istart = j;
      ctemp = ascale * H(j, j) - shift * (bscale * Tm(j, j));
      temp = Cabs1(ctemp);
      temp2 = ascale * Cabs1(H(j + 1, j));
      tempr = Max(temp, temp2);
      if (tempr < one && tempr != zero) {
        temp = temp / tempr;
        temp2 = temp2 / tempr;
      }
      if (Cabs1(H(j, j - 1)) * temp2 <= temp * atol) goto L90;
    }
    istart = ifirst;
    ctemp = ascale * H(ifirst, ifirst) - shift * (bscale * Tm(ifirst, ifirst));
  L90:
    // Do an implicit-shift QZ sweep.
    // Initial Q
    ctemp2 = ascale * H(istart + 1, istart);
    Lartg(ctemp, ctemp2, &c, &s, &ctemp3);
    // Sweep
    for (j = istart; j <= ilast - 1; ++j) {
      if (j > istart) {
        ctemp = H(j, j - 1);
        Lartg(ctemp, H(j + 1, j - 1), &c, &s, H.ptr(j, j - 1));
        H(j + 1, j - 1) = czero;
      }
      for (jc = j; jc <= ilastm; ++jc) {
        ctemp = c * H(j, jc) + s * H(j + 1, jc);
        H(j + 1, jc) = -Conj(s) * H(j, jc) + c * H(j + 1, jc);
        H(j, jc) = ctemp;
        ctemp2 = c * Tm(j, jc) + s * Tm(j + 1, jc);
        Tm(j + 1, jc) = -Conj(s) * Tm(j, jc) + c * Tm(j + 1, jc);
        Tm(j, jc) = ctemp2;
      }
      if (ilq) {
        for (jr = 1; jr <= n; ++jr) {
          ctemp = c * Q(jr, j) + Conj(s) * Q(jr, j + 1);
          Q(jr, j + 1) = -s * Q(jr, j) + c * Q(jr, j + 1);
          Q(jr, j) = ctemp;
        }
      }
      ctemp = Tm(j + 1, j + 1);
      Lartg(ctemp, Tm(j + 1, j), &c, &s, Tm.ptr(j + 1, j + 1));
      Tm(j + 1, j) = czero;
      for (jr = ifrstm; jr <= Min<blasint>(j + 2, ilast); ++jr) {
        ctemp = c * H(jr, j + 1) + s * H(jr, j);
        H(jr, j) = -Conj(s) * H(jr, j + 1) + c * H(jr, j);
        H(jr, j + 1) = ctemp;
      }
      for (jr = ifrstm; jr <= j; ++jr) {
        ctemp = c * Tm(jr, j + 1) + s * Tm(jr, j);
        Tm(jr, j) = -Conj(s) * Tm(jr, j + 1) + c * Tm(jr, j);
        Tm(jr, j + 1) = ctemp;
      }
      if (ilz) {
        for (jr = 1; jr <= n; ++jr) {
          ctemp = c * Z(jr, j + 1) + s * Z(jr, j);
          Z(jr, j) = -Conj(s) * Z(jr, j + 1) + c * Z(jr, j);
          Z(jr, j + 1) = ctemp;
        }
      }
    }
  L160:;
  }
  // Drop-through = non-convergence
L180:
  info = ilast;
  goto L210;
  // Successful completion of all QZ steps
L190:
  // Set Eigenvalues 1:ILO-1
  for (j = 1; j <= ilo - 1; ++j) {
    absb = std::abs(Tm(j, j));
    if (absb > safmin) {
      signbc = Conj(Tm(j, j) / absb);
      Tm(j, j) = absb;
      if (ilschr) {
        Scal(j - 1, signbc, Tm.ptr(1, j), 1);
        Scal(j, signbc, H.ptr(1, j), 1);
      } else {
        Scal(1, signbc, H.ptr(j, j), 1);
      }
      if (ilz) Scal(n, signbc, Z.ptr(1, j), 1);
    } else {
      Tm(j, j) = czero;
    }
    ALPHA(j) = H(j, j);
    BETA(j) = Tm(j, j);
  }
  // Normal Termination
  info = 0;
  // Exit (other than argument error) -- return optimal workspace size
L210:
  WORK(1) = WorkSize<T>(n);
  return info;
}

template void Lag2<float>(const float*, blasint, const float*, blasint, float,
                          float*, float*, float*, float*, float*);
template void Lag2<double>(const double*, blasint, const double*, blasint,
                           double, double*, double*, double*, double*, double*);

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template blasint Hgeqz<R>(char, char, char, blasint, blasint, blasint, R*,   \
                            blasint, R*, blasint, R*, R*, R*, R*, blasint, R*, \
                            blasint, R*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                \
  template blasint Hgeqz<T>(char, char, char, blasint, blasint, blasint, T*, \
                            blasint, T*, blasint, T*, T*, T*, blasint, T*,   \
                            blasint, T*, blasint, RealT<T>*);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_HGEQZ_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##lag2_(                                              \
      const R* a, const hwyblas_int* lda, const R* b, const hwyblas_int* ldb,  \
      const R* safmin, R* scale1, R* scale2, R* wr1, R* wr2, R* wi) {          \
    hwyblas::lapack::Lag2(a, *lda, b, *ldb, *safmin, scale1, scale2, wr1, wr2, \
                          wi);                                                 \
  }                                                                            \
  HWYBLAS_API void prefix##hgeqz_(                                             \
      const char* job, const char* compq, const char* compz,                   \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* h, const hwyblas_int* ldh, R* t, const hwyblas_int* ldt, R* alphar,   \
      R* alphai, R* beta, R* q, const hwyblas_int* ldq, R* z,                  \
      const hwyblas_int* ldz, R* work, const hwyblas_int* lwork,               \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Hgeqz(*job, *compq, *compz, *n, *ilo, *ihi, h,    \
                                   *ldh, t, *ldt, alphar, alphai, beta, q,     \
                                   *ldq, z, *ldz, work, *lwork);               \
  }

HWYBLAS_LAPACK_HGEQZ_REAL(s, float)
HWYBLAS_LAPACK_HGEQZ_REAL(d, double)

#undef HWYBLAS_LAPACK_HGEQZ_REAL

#define HWYBLAS_LAPACK_HGEQZ_COMPLEX(prefix, T, R)                            \
  HWYBLAS_API void prefix##hgeqz_(                                            \
      const char* job, const char* compq, const char* compz,                  \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,   \
      R* h, const hwyblas_int* ldh, R* t, const hwyblas_int* ldt, R* alpha,   \
      R* beta, R* q, const hwyblas_int* ldq, R* z, const hwyblas_int* ldz,    \
      R* work, const hwyblas_int* lwork, R* rwork, hwyblas_int* info) {       \
    *info = hwyblas::lapack::Hgeqz(                                           \
        *job, *compq, *compz, *n, *ilo, *ihi, As<T>(h), *ldh, As<T>(t), *ldt, \
        As<T>(alpha), As<T>(beta), As<T>(q), *ldq, As<T>(z), *ldz,            \
        As<T>(work), *lwork, rwork);                                          \
  }

HWYBLAS_LAPACK_HGEQZ_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_HGEQZ_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_HGEQZ_COMPLEX

}  // extern "C"
