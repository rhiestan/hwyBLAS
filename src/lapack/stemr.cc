// Eigenvalues and eigenvectors of a symmetric tridiagonal matrix by the
// algorithm of Multiple Relatively Robust Representations (MRRR): xSTEMR
// (driver), xLARRE (splits the matrix, finds a root representation L D L**T
// of each block and its eigenvalues by dqds or bisection), xLARRV (builds
// the representation tree and computes the eigenvectors) and their
// helpers: xLARRA (splitting), xLARRB, xLARRD, xLARRJ, xLARRK (bisection),
// xLARRC (Sturm counts), xLARRF (the representation of a child cluster),
// xLARRR (whether the matrix defines its eigenvalues to high relative
// accuracy), xLAR1V (an eigenvector by a twisted factorization) and
// xLANEG (the Sturm count of L D L**T - SIGMA*I). The helpers are real
// (R); xSTEMR, xLARRV and xLAR1V serve all four types (for complex T, the
// real eigenvectors are computed into the complex Z).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <type_traits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLANEG: the number of negative pivots (the Sturm count) of
// L D L**T - SIGMA*I (D: N, LLD(i) = L(i)**2 * D(i): N-1) by the twisted
// factorization with twist index R; pivots smaller than PIVMIN are not
// replaced, a NaN from Inf/Inf is (by 1) in a slower rerun of the block of
// 128 rows where it occurred. PIVMIN is not used (as in LAPACK).
template <typename R>
blasint Laneg(blasint n, R* d, R* lld, R sigma, R /*pivmin*/, blasint r) {
  const R zero = R(0);
  const R one = R(1);
  const blasint blklen = 128;
  const Vector<R> D(d);
  const Vector<R> LLD(lld);
  blasint bj, j, neg1, neg2, negcnt;
  bool sawnan;
  R bsav, dminus, dplus, gamma, p, t, tmp;
  negcnt = 0;
  // I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
  t = -sigma;
  for (bj = 1; bj <= r - 1; bj += blklen) {
    neg1 = 0;
    bsav = t;
    for (j = bj; j <= Min<blasint>(bj + blklen - 1, r - 1); ++j) {
      dplus = D(j) + t;
      if (dplus < zero) neg1 = neg1 + 1;
      tmp = t / dplus;
      t = tmp * LLD(j) - sigma;
    }
    sawnan = std::isnan(t);
    // Run a slower version of the above loop if a NaN is detected.
    // A NaN should occur only with a zero pivot after an infinite
    // pivot.  In that case, substituting 1 for T/DPLUS is the
    // correct limit.
    if (sawnan) {
      neg1 = 0;
      t = bsav;
      for (j = bj; j <= Min<blasint>(bj + blklen - 1, r - 1); ++j) {
        dplus = D(j) + t;
        if (dplus < zero) neg1 = neg1 + 1;
        tmp = t / dplus;
        if (std::isnan(tmp)) tmp = one;
        t = tmp * LLD(j) - sigma;
      }
    }
    negcnt = negcnt + neg1;
  }
  // II) lower part: L D L^T - SIGMA I = U- D- U-^T
  p = D(n) - sigma;
  for (bj = n - 1; bj >= r; bj -= blklen) {
    neg2 = 0;
    bsav = p;
    for (j = bj; j >= Max<blasint>(bj - blklen + 1, r); --j) {
      dminus = LLD(j) + p;
      if (dminus < zero) neg2 = neg2 + 1;
      tmp = p / dminus;
      p = tmp * D(j) - sigma;
    }
    sawnan = std::isnan(p);
    // As above, run a slower version that substitutes 1 for Inf/Inf.
    if (sawnan) {
      neg2 = 0;
      p = bsav;
      for (j = bj; j >= Max<blasint>(bj - blklen + 1, r); --j) {
        dminus = LLD(j) + p;
        if (dminus < zero) neg2 = neg2 + 1;
        tmp = p / dminus;
        if (std::isnan(tmp)) tmp = one;
        p = tmp * D(j) - sigma;
      }
    }
    negcnt = negcnt + neg2;
  }
  // III) Twist index
  // T was shifted by SIGMA initially.
  gamma = (t + sigma) + p;
  if (gamma < zero) negcnt = negcnt + 1;
  return negcnt;
}

// xLAR1V: the eigenvector Z (rows B1..BN) of L D L**T for the eigenvalue
// approximation LAMBDA by the twisted factorization N_r Delta_r N_r**T of
// L D L**T - LAMBDA*I (LD(i) = L(i)*D(i), LLD(i) = L(i)**2 * D(i)), at the
// twist index R where |gamma(r)| (MINGMA) is minimal (R = 0 on entry:
// chosen in B1..BN; otherwise kept); entries are set to zero below
// GAPTOL, ISUPPZ gives the support. Returns NEGCNT (the Sturm count, if
// WANTNC), ZTZ = Z**T Z, NRMINV = 1/sqrt(ZTZ), RESID = |MINGMA|*NRMINV and
// the Rayleigh quotient correction RQCORR = MINGMA/ZTZ. WORK has 4*N
// elements. For complex T, Z is real (stored in the complex Z).
template <typename T>
void Lar1v(blasint n, blasint b1, blasint bn, RealT<T> lambda, RealT<T>* d,
           RealT<T>* l, RealT<T>* ld, RealT<T>* lld, RealT<T> pivmin,
           RealT<T> gaptol, T* z, bool wantnc, blasint* negcnt, RealT<T>* ztz,
           RealT<T>* mingma, blasint* r, blasint* isuppz, RealT<T>* nrminv,
           RealT<T>* resid, RealT<T>* rqcorr, RealT<T>* work) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T cone = T(1);
  const Vector<R> D(d);
  const Vector<blasint> ISUPPZ(isuppz);
  const Vector<R> L(l);
  const Vector<R> LD(ld);
  const Vector<R> LLD(lld);
  const Vector<R> WORK(work);
  const Vector<T> Z(z);
  blasint i, indlpl, indp, inds, indumn, neg1, neg2, r1, r2;
  bool sawnan1, sawnan2;
  R dminus, dplus, eps, s, tmp;
  eps = Lamch<R>('P');
  if (*r == 0) {
    r1 = b1;
    r2 = bn;
  } else {
    r1 = *r;
    r2 = *r;
  }
  // Storage for LPLUS
  indlpl = 0;
  // Storage for UMINUS
  indumn = n;
  inds = 2 * n + 1;
  indp = 3 * n + 1;
  if (b1 == 1) {
    WORK(inds) = zero;
  } else {
    WORK(inds + b1 - 1) = LLD(b1 - 1);
  }
  // Compute the stationary transform (using the differential form)
  // until the index R2.
  sawnan1 = false;
  neg1 = 0;
  s = WORK(inds + b1 - 1) - lambda;
  for (i = b1; i <= r1 - 1; ++i) {
    dplus = D(i) + s;
    WORK(indlpl + i) = LD(i) / dplus;
    if (dplus < zero) neg1 = neg1 + 1;
    WORK(inds + i) = s * WORK(indlpl + i) * L(i);
    s = WORK(inds + i) - lambda;
  }
  sawnan1 = std::isnan(s);
  if (sawnan1) goto L60;
  for (i = r1; i <= r2 - 1; ++i) {
    dplus = D(i) + s;
    WORK(indlpl + i) = LD(i) / dplus;
    WORK(inds + i) = s * WORK(indlpl + i) * L(i);
    s = WORK(inds + i) - lambda;
  }
  sawnan1 = std::isnan(s);
L60:
  if (sawnan1) {
    // Runs a slower version of the above loop if a NaN is detected
    neg1 = 0;
    s = WORK(inds + b1 - 1) - lambda;
    for (i = b1; i <= r1 - 1; ++i) {
      dplus = D(i) + s;
      if (std::abs(dplus) < pivmin) dplus = -pivmin;
      WORK(indlpl + i) = LD(i) / dplus;
      if (dplus < zero) neg1 = neg1 + 1;
      WORK(inds + i) = s * WORK(indlpl + i) * L(i);
      if (WORK(indlpl + i) == zero) WORK(inds + i) = LLD(i);
      s = WORK(inds + i) - lambda;
    }
    for (i = r1; i <= r2 - 1; ++i) {
      dplus = D(i) + s;
      if (std::abs(dplus) < pivmin) dplus = -pivmin;
      WORK(indlpl + i) = LD(i) / dplus;
      WORK(inds + i) = s * WORK(indlpl + i) * L(i);
      if (WORK(indlpl + i) == zero) WORK(inds + i) = LLD(i);
      s = WORK(inds + i) - lambda;
    }
  }
  // Compute the progressive transform (using the differential form)
  // until the index R1
  sawnan2 = false;
  neg2 = 0;
  WORK(indp + bn - 1) = D(bn) - lambda;
  for (i = bn - 1; i >= r1; --i) {
    dminus = LLD(i) + WORK(indp + i);
    tmp = D(i) / dminus;
    if (dminus < zero) neg2 = neg2 + 1;
    WORK(indumn + i) = L(i) * tmp;
    WORK(indp + i - 1) = WORK(indp + i) * tmp - lambda;
  }
  tmp = WORK(indp + r1 - 1);
  sawnan2 = std::isnan(tmp);
  if (sawnan2) {
    // Runs a slower version of the above loop if a NaN is detected
    neg2 = 0;
    for (i = bn - 1; i >= r1; --i) {
      dminus = LLD(i) + WORK(indp + i);
      if (std::abs(dminus) < pivmin) dminus = -pivmin;
      tmp = D(i) / dminus;
      if (dminus < zero) neg2 = neg2 + 1;
      WORK(indumn + i) = L(i) * tmp;
      WORK(indp + i - 1) = WORK(indp + i) * tmp - lambda;
      if (tmp == zero) WORK(indp + i - 1) = D(i) - lambda;
    }
  }
  // Find the index (from R1 to R2) of the largest (in magnitude)
  // diagonal element of the inverse
  *mingma = WORK(inds + r1 - 1) + WORK(indp + r1 - 1);
  if (*mingma < zero) neg1 = neg1 + 1;
  if (wantnc) {
    *negcnt = neg1 + neg2;
  } else {
    *negcnt = -1;
  }
  if (std::abs(*mingma) == zero) *mingma = eps * WORK(inds + r1 - 1);
  *r = r1;
  for (i = r1; i <= r2 - 1; ++i) {
    tmp = WORK(inds + i) + WORK(indp + i);
    if (tmp == zero) tmp = eps * WORK(inds + i);
    if (std::abs(tmp) <= std::abs(*mingma)) {
      *mingma = tmp;
      *r = i + 1;
    }
  }
  // Compute the FP vector: solve N^T v = e_r
  ISUPPZ(1) = b1;
  ISUPPZ(2) = bn;
  Z(*r) = cone;
  *ztz = one;
  // Compute the FP vector upwards from R
  if (!sawnan1 && !sawnan2) {
    for (i = *r - 1; i >= b1; --i) {
      Z(i) = -(WORK(indlpl + i) * Z(i + 1));
      if ((std::abs(Z(i)) + std::abs(Z(i + 1))) * std::abs(LD(i)) < gaptol) {
        Z(i) = zero;
        ISUPPZ(1) = i + 1;
        goto L220;
      }
      *ztz = *ztz + std::real(Z(i) * Z(i));
    }
  L220:;
  } else {
    // Run slower loop if NaN occurred.
    for (i = *r - 1; i >= b1; --i) {
      if (Z(i + 1) == zero) {
        Z(i) = -(LD(i + 1) / LD(i)) * Z(i + 2);
      } else {
        Z(i) = -(WORK(indlpl + i) * Z(i + 1));
      }
      if ((std::abs(Z(i)) + std::abs(Z(i + 1))) * std::abs(LD(i)) < gaptol) {
        Z(i) = zero;
        ISUPPZ(1) = i + 1;
        goto L240;
      }
      *ztz = *ztz + std::real(Z(i) * Z(i));
    }
  L240:;
  }
  // Compute the FP vector downwards from R in blocks of size BLKSIZ
  if (!sawnan1 && !sawnan2) {
    for (i = *r; i <= bn - 1; ++i) {
      Z(i + 1) = -(WORK(indumn + i) * Z(i));
      if ((std::abs(Z(i)) + std::abs(Z(i + 1))) * std::abs(LD(i)) < gaptol) {
        Z(i + 1) = zero;
        ISUPPZ(2) = i;
        goto L260;
      }
      *ztz = *ztz + std::real(Z(i + 1) * Z(i + 1));
    }
  L260:;
  } else {
    // Run slower loop if NaN occurred.
    for (i = *r; i <= bn - 1; ++i) {
      if (Z(i) == zero) {
        Z(i + 1) = -(LD(i - 1) / LD(i)) * Z(i - 1);
      } else {
        Z(i + 1) = -(WORK(indumn + i) * Z(i));
      }
      if ((std::abs(Z(i)) + std::abs(Z(i + 1))) * std::abs(LD(i)) < gaptol) {
        Z(i + 1) = zero;
        ISUPPZ(2) = i;
        goto L280;
      }
      *ztz = *ztz + std::real(Z(i + 1) * Z(i + 1));
    }
  L280:;
  }
  // Compute quantities for convergence test
  tmp = one / *ztz;
  *nrminv = std::sqrt(tmp);
  *resid = std::abs(*mingma) * *nrminv;
  *rqcorr = *mingma * tmp;
}

// xLARRA: the splitting points of the symmetric tridiagonal matrix with
// diagonal D and off-diagonal E (squares E2): E(i) (and E2(i)) is set to
// zero if |E(i)| <= |SPLTOL| * TNRM (SPLTOL < 0) or <= SPLTOL *
// sqrt(|D(i)|) * sqrt(|D(i+1)|) (relative accuracy). Returns the NSPLIT
// blocks ending at ISPLIT and INFO = 0.
template <typename R>
blasint Larra(blasint n, R* d, R* e, R* e2, R spltol, R tnrm, blasint* nsplit,
              blasint* isplit) {
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<R> E2(e2);
  const Vector<blasint> ISPLIT(isplit);
  blasint i;
  R eabs, tmp1;
  blasint info = 0;
  *nsplit = 1;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  // Compute splitting points
  if (spltol < zero) {
    // Criterion based on absolute off-diagonal value
    tmp1 = std::abs(spltol) * tnrm;
    for (i = 1; i <= n - 1; ++i) {
      eabs = std::abs(E(i));
      if (eabs <= tmp1) {
        E(i) = zero;
        E2(i) = zero;
        ISPLIT(*nsplit) = i;
        *nsplit = *nsplit + 1;
      }
    }
  } else {
    // Criterion that guarantees relative accuracy
    for (i = 1; i <= n - 1; ++i) {
      eabs = std::abs(E(i));
      if (eabs <=
          spltol * std::sqrt(std::abs(D(i))) * std::sqrt(std::abs(D(i + 1)))) {
        E(i) = zero;
        E2(i) = zero;
        ISPLIT(*nsplit) = i;
        *nsplit = *nsplit + 1;
      }
    }
  }
  ISPLIT(*nsplit) = n;
  return info;
}

// xLARRB: refines by bisection the eigenvalue approximations W(i-OFFSET) +-
// WERR(i-OFFSET), i = IFIRST..ILAST, of L D L**T (D, LLD) until the
// interval is smaller than max(RTOL1 * gap, RTOL2 * max|end|) (gap: from
// WGAP), with the Sturm counts of xLANEG (twist index TWIST; outside 1..N:
// N). Updates W, WERR and WGAP. WORK has 2*N, IWORK 2*N elements. Returns
// INFO = 0.
template <typename R>
blasint Larrb(blasint n, R* d, R* lld, blasint ifirst, blasint ilast, R rtol1,
              R rtol2, blasint offset, R* w, R* wgap, R* werr, R* work,
              blasint* iwork, R pivmin, R spdiam, blasint twist) {
  const R zero = R(0);
  const R two = R(2);
  const R half = R(0.5);
  const Vector<R> D(d);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> LLD(lld);
  const Vector<R> W(w);
  const Vector<R> WERR(werr);
  const Vector<R> WGAP(wgap);
  const Vector<R> WORK(work);
  blasint i, i1, ii, ip, iter, k, maxitr, negcnt, next, nint, olnint, prev, r;
  R back, cvrgd, gap, left, lgap, mid, mnwdth, rgap, right, tmp, width;
  blasint info = 0;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  maxitr = static_cast<blasint>((std::log(spdiam + pivmin) - std::log(pivmin)) /
                                std::log(two)) +
           2;
  mnwdth = two * pivmin;
  r = twist;
  if ((r < 1) || (r > n)) r = n;
  // Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
  // The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
  // Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 )
  // for an unconverged interval is set to the index of the next unconverged
  // interval, and is -1 or 0 for a converged interval. Thus a linked
  // list of unconverged intervals is set up.
  i1 = ifirst;
  // The number of unconverged intervals
  nint = 0;
  // The last unconverged interval found
  prev = 0;
  rgap = WGAP(i1 - offset);
  for (i = i1; i <= ilast; ++i) {
    k = 2 * i;
    ii = i - offset;
    left = W(ii) - WERR(ii);
    right = W(ii) + WERR(ii);
    lgap = rgap;
    rgap = WGAP(ii);
    gap = Min(lgap, rgap);
    // Make sure that [LEFT,RIGHT] contains the desired eigenvalue
    // Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT
    // Do while( NEGCNT(LEFT).GT.I-1 )
    back = WERR(ii);
  L20:
    negcnt = Laneg(n, d, lld, left, pivmin, r);
    if (negcnt > i - 1) {
      left = left - back;
      back = two * back;
      goto L20;
    }
    // Do while( NEGCNT(RIGHT).LT.I )
    // Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT
    back = WERR(ii);
  L50:
    negcnt = Laneg(n, d, lld, right, pivmin, r);
    if (negcnt < i) {
      right = right + back;
      back = two * back;
      goto L50;
    }
    width = half * std::abs(left - right);
    tmp = Max(std::abs(left), std::abs(right));
    cvrgd = Max(rtol1 * gap, rtol2 * tmp);
    if (width <= cvrgd || width <= mnwdth) {
      // This interval has already converged and does not need refinement.
      // (Note that the gaps might change through refining the
      // eigenvalues, however, they can only get bigger.)
      // Remove it from the list.
      IWORK(k - 1) = -1;
      // Make sure that I1 always points to the first unconverged interval
      if ((i == i1) && (i < ilast)) i1 = i + 1;
      if ((prev >= i1) && (i <= ilast)) IWORK(2 * prev - 1) = i + 1;
    } else {
      // unconverged interval found
      prev = i;
      nint = nint + 1;
      IWORK(k - 1) = i + 1;
      IWORK(k) = negcnt;
    }
    WORK(k - 1) = left;
    WORK(k) = right;
  }
  // Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
  // and while (ITER.LT.MAXITR)
  iter = 0;
L80:
  prev = i1 - 1;
  i = i1;
  olnint = nint;
  for (ip = 1; ip <= olnint; ++ip) {
    k = 2 * i;
    ii = i - offset;
    rgap = WGAP(ii);
    lgap = rgap;
    if (ii > 1) lgap = WGAP(ii - 1);
    gap = Min(lgap, rgap);
    next = IWORK(k - 1);
    left = WORK(k - 1);
    right = WORK(k);
    mid = half * (left + right);
    // semiwidth of interval
    width = right - mid;
    tmp = Max(std::abs(left), std::abs(right));
    cvrgd = Max(rtol1 * gap, rtol2 * tmp);
    if ((width <= cvrgd) || (width <= mnwdth) || (iter == maxitr)) {
      // reduce number of unconverged intervals
      nint = nint - 1;
      // Mark interval as converged.
      IWORK(k - 1) = 0;
      if (i1 == i) {
        i1 = next;
      } else {
        // Prev holds the last unconverged interval previously examined
        if (prev >= i1) IWORK(2 * prev - 1) = next;
      }
      i = next;
      goto L100;
    }
    prev = i;
    // Perform one bisection step
    negcnt = Laneg(n, d, lld, mid, pivmin, r);
    if (negcnt <= i - 1) {
      WORK(k - 1) = mid;
    } else {
      WORK(k) = mid;
    }
    i = next;
  L100:;
  }
  iter = iter + 1;
  // do another loop if there are still unconverged intervals
  // However, in the last iteration, all intervals are accepted
  // since this is the best we can do.
  if ((nint > 0) && (iter <= maxitr)) goto L80;
  // At this point, all the intervals have converged
  for (i = ifirst; i <= ilast; ++i) {
    k = 2 * i;
    ii = i - offset;
    // All intervals marked by '0' have been refined.
    if (IWORK(k - 1) == 0) {
      W(ii) = half * (WORK(k - 1) + WORK(k));
      WERR(ii) = WORK(k) - W(ii);
    }
  }
  for (i = ifirst + 1; i <= ilast; ++i) {
    k = 2 * i;
    ii = i - offset;
    WGAP(ii - 1) = Max(zero, W(ii) - WERR(ii) - W(ii - 1) - WERR(ii - 1));
  }
  return info;
}

// xLARRC: the number of eigenvalues in (VL, VU] of the symmetric
// tridiagonal matrix T (JOBT = 'T': diagonal D, off-diagonal E) or of
// L D L**T (JOBT = 'L': D, L in E): EIGCNT = RCNT - LCNT, the Sturm counts
// at VU and VL. PIVMIN is not used (as in LAPACK). Returns INFO = 0.
template <typename R>
blasint Larrc(char jobt, blasint n, R vl, R vu, R* d, R* e, R /*pivmin*/,
              blasint* eigcnt, blasint* lcnt, blasint* rcnt) {
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<R> E(e);
  blasint i;
  bool matt;
  R lpivot, rpivot, sl, su, tmp, tmp2;
  blasint info = 0;
  *lcnt = 0;
  *rcnt = 0;
  *eigcnt = 0;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  matt = Lsame(jobt, 'T');
  if (matt) {
    // Sturm sequence count on T
    lpivot = D(1) - vl;
    rpivot = D(1) - vu;
    if (lpivot <= zero) {
      *lcnt = *lcnt + 1;
    }
    if (rpivot <= zero) {
      *rcnt = *rcnt + 1;
    }
    for (i = 1; i <= n - 1; ++i) {
      tmp = (E(i) * E(i));
      lpivot = (D(i + 1) - vl) - tmp / lpivot;
      rpivot = (D(i + 1) - vu) - tmp / rpivot;
      if (lpivot <= zero) {
        *lcnt = *lcnt + 1;
      }
      if (rpivot <= zero) {
        *rcnt = *rcnt + 1;
      }
    }
  } else {
    // Sturm sequence count on L D L^T
    sl = -vl;
    su = -vu;
    for (i = 1; i <= n - 1; ++i) {
      lpivot = D(i) + sl;
      rpivot = D(i) + su;
      if (lpivot <= zero) {
        *lcnt = *lcnt + 1;
      }
      if (rpivot <= zero) {
        *rcnt = *rcnt + 1;
      }
      tmp = E(i) * D(i) * E(i);
      tmp2 = tmp / lpivot;
      if (tmp2 == zero) {
        sl = tmp - vl;
      } else {
        sl = sl * tmp2 - vl;
      }
      tmp2 = tmp / rpivot;
      if (tmp2 == zero) {
        su = tmp - vu;
      } else {
        su = su * tmp2 - vu;
      }
    }
    lpivot = D(n) + sl;
    rpivot = D(n) + su;
    if (lpivot <= zero) {
      *lcnt = *lcnt + 1;
    }
    if (rpivot <= zero) {
      *rcnt = *rcnt + 1;
    }
  }
  *eigcnt = *rcnt - *lcnt;
  return info;
}

// xLARRD: the eigenvalues of the symmetric tridiagonal matrix (D, E, E2)
// split into NSPLIT blocks (ISPLIT) by bisection (xLAEBZ), to a relative
// accuracy RELTOL, with the Gershgorin intervals GERS: all (RANGE = 'A'),
// in (VL, VU] or the IL-th through IU-th, ordered by block (ORDER = 'B')
// or overall ('E'). Returns M eigenvalues W with error bounds WERR, block
// numbers IBLOCK and indices INDEXW within their blocks, and (WL, WU]
// containing them. WORK has 4*N, IWORK 3*N elements. Returns INFO: 0, -i
// for an illegal argument i, 1 to 3 (not converged, too few eigenvalues
// found, both) or 4 (bad arithmetic), or the INFO of xLAEBZ.
template <typename R>
blasint Larrd(char range, char order, blasint n, R vl, R vu, blasint il,
              blasint iu, R* gers, R reltol, R* d, R* e, R* e2, R pivmin,
              blasint nsplit, blasint* isplit, blasint* m, R* w, R* werr, R* wl,
              R* wu, blasint* iblock, blasint* indexw, R* work,
              blasint* iwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = one / two;
  const R fudge = two;
  const blasint allrng = 1;
  const blasint valrng = 2;
  const blasint indrng = 3;
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<R> E2(e2);
  const Vector<R> GERS(gers);
  const Vector<blasint> IBLOCK(iblock);
  blasint idumma[1];
  const Vector<blasint> INDEXW(indexw);
  const Vector<blasint> ISPLIT(isplit);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WERR(werr);
  const Vector<R> WORK(work);
  blasint i, ib, ibegin, idiscl, idiscu, ie, iend, iinfo, im, in, ioff, iout,
      irange, itmax, itmp1, itmp2, iw, iwoff, j, jblk, jdisc, je, jee, nb, nwl,
      nwu;
  bool ncnvrg, toofew;
  R atoli, eps, gl, gu, rtoli, tmp1, tmp2, tnorm, uflow, wkill;
  // Set and used only for RANGE = 'I'.
  R wlu = R(0), wul = R(0);
  blasint info = 0;
  *m = 0;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  // Decode RANGE
  if (Lsame(range, 'A')) {
    irange = allrng;
  } else if (Lsame(range, 'V')) {
    irange = valrng;
  } else if (Lsame(range, 'I')) {
    irange = indrng;
  } else {
    irange = 0;
  }
  // Check for Errors
  if (irange <= 0) {
    info = -1;
  } else if (!(Lsame(order, 'B') || Lsame(order, 'E'))) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (irange == valrng) {
    if (vl >= vu) info = -5;
  } else if (irange == indrng && (il < 1 || il > Max<blasint>(1, n))) {
    info = -6;
  } else if (irange == indrng && (iu < Min<blasint>(n, il) || iu > n)) {
    info = -7;
  }
  if (info != 0) {
    return info;
  }
  // Initialize error flags
  ncnvrg = false;
  toofew = false;
  // Simplification:
  if (irange == indrng && il == 1 && iu == n) irange = 1;
  // Get machine constants
  eps = Lamch<R>('P');
  uflow = Lamch<R>('U');
  // Special Case when N=1
  // Treat case of 1x1 matrix for quick return
  if (n == 1) {
    if ((irange == allrng) ||
        ((irange == valrng) && (D(1) > vl) && (D(1) <= vu)) ||
        ((irange == indrng) && (il == 1) && (iu == 1))) {
      *m = 1;
      W(1) = D(1);
      // The computation error of the eigenvalue is zero
      WERR(1) = zero;
      IBLOCK(1) = 1;
      INDEXW(1) = 1;
    }
    return info;
  }
  // NB is the minimum vector length for vector bisection, or 0
  // if only scalar is to be done.
  nb = Ilaenv<R>(1, "STEBZ", " ", n, -1, -1, -1);
  if (nb <= 1) nb = 0;
  // Find global spectral radius
  gl = D(1);
  gu = D(1);
  for (i = 1; i <= n; ++i) {
    gl = Min(gl, GERS(2 * i - 1));
    gu = Max(gu, GERS(2 * i));
  }
  // Compute global Gerschgorin bounds and spectral diameter
  tnorm = Max(std::abs(gl), std::abs(gu));
  gl = gl - fudge * tnorm * eps * n - fudge * two * pivmin;
  gu = gu + fudge * tnorm * eps * n + fudge * two * pivmin;
  // [JAN/28/2009] remove the line below since SPDIAM variable not use
  // SPDIAM = GU - GL
  // Input arguments for DLAEBZ:
  // The relative tolerance.  An interval (a,b] lies within
  // "relative tolerance" if  b-a < RELTOL*max(|a|,|b|),
  rtoli = reltol;
  // Set the absolute tolerance for interval convergence to zero to force
  // interval convergence based on relative size of the interval.
  // This is dangerous because intervals might not converge when RELTOL is
  // small. But at least a very small number should be selected so that for
  // strongly graded matrices, the code can get relatively accurate
  // eigenvalues.
  atoli = fudge * two * uflow + fudge * two * pivmin;
  if (irange == indrng) {
    // RANGE='I': Compute an interval containing eigenvalues
    // IL through IU. The initial interval [GL,GU] from the global
    // Gerschgorin bounds GL and GU is refined by DLAEBZ.
    itmax = static_cast<blasint>((std::log(tnorm + pivmin) - std::log(pivmin)) /
                                 std::log(two)) +
            2;
    WORK(n + 1) = gl;
    WORK(n + 2) = gl;
    WORK(n + 3) = gu;
    WORK(n + 4) = gu;
    WORK(n + 5) = gl;
    WORK(n + 6) = gu;
    IWORK(1) = -1;
    IWORK(2) = -1;
    IWORK(3) = n + 1;
    IWORK(4) = n + 1;
    IWORK(5) = il - 1;
    IWORK(6) = iu;
    iinfo = Laebz(3, itmax, n, 2, 2, nb, atoli, rtoli, pivmin, d, e, e2,
                  IWORK.ptr(5), WORK.ptr(n + 1), WORK.ptr(n + 5), &iout, iwork,
                  w, iblock);
    if (iinfo != 0) {
      info = iinfo;
      return info;
    }
    // On exit, output intervals may not be ordered by ascending negcount
    if (IWORK(6) == iu) {
      *wl = WORK(n + 1);
      wlu = WORK(n + 3);
      nwl = IWORK(1);
      *wu = WORK(n + 4);
      wul = WORK(n + 2);
      nwu = IWORK(4);
    } else {
      *wl = WORK(n + 2);
      wlu = WORK(n + 4);
      nwl = IWORK(2);
      *wu = WORK(n + 3);
      wul = WORK(n + 1);
      nwu = IWORK(3);
    }
    // On exit, the interval [WL, WLU] contains a value with negcount NWL,
    // and [WUL, WU] contains a value with negcount NWU.
    if (nwl < 0 || nwl >= n || nwu < 1 || nwu > n) {
      info = 4;
      return info;
    }
  } else if (irange == valrng) {
    *wl = vl;
    *wu = vu;
  } else if (irange == allrng) {
    *wl = gl;
    *wu = gu;
  }
  // Find Eigenvalues -- Loop Over blocks and recompute NWL and NWU.
  // NWL accumulates the number of eigenvalues .le. WL,
  // NWU accumulates the number of eigenvalues .le. WU
  *m = 0;
  iend = 0;
  info = 0;
  nwl = 0;
  nwu = 0;
  for (jblk = 1; jblk <= nsplit; ++jblk) {
    ioff = iend;
    ibegin = ioff + 1;
    iend = ISPLIT(jblk);
    in = iend - ioff;
    if (in == 1) {
      // 1x1 block
      if (*wl >= D(ibegin) - pivmin) nwl = nwl + 1;
      if (*wu >= D(ibegin) - pivmin) nwu = nwu + 1;
      if (irange == allrng ||
          (*wl < D(ibegin) - pivmin && *wu >= D(ibegin) - pivmin)) {
        *m = *m + 1;
        W(*m) = D(ibegin);
        WERR(*m) = zero;
        // The gap for a single block doesn't matter for the later
        // algorithm and is assigned an arbitrary large value
        IBLOCK(*m) = jblk;
        INDEXW(*m) = 1;
      }
      // Disabled 2x2 case because of a failure on the following matrix
      // RANGE = 'I', IL = IU = 4
      // Original Tridiagonal, d = [
      // -0.150102010615740E+00
      // -0.849897989384260E+00
      // -0.128208148052635E-15
      // 0.128257718286320E-15
      // ];
      // e = [
      // -0.357171383266986E+00
      // -0.180411241501588E-15
      // -0.175152352710251E-15
      // ];
      // ELSE IF( IN.EQ.2 ) THEN
      // *           2x2 block
      // DISC = SQRT( (HALF*(D(IBEGIN)-D(IEND)))**2 + E(IBEGIN)**2 )
      // TMP1 = HALF*(D(IBEGIN)+D(IEND))
      // L1 = TMP1 - DISC
      // IF( WL.GE. L1-PIVMIN )
      // $         NWL = NWL + 1
      // IF( WU.GE. L1-PIVMIN )
      // $         NWU = NWU + 1
      // IF( IRANGE.EQ.ALLRNG .OR. ( WL.LT.L1-PIVMIN .AND. WU.GE.
      // $          L1-PIVMIN ) ) THEN
      // M = M + 1
      // W( M ) = L1
      // *              The uncertainty of eigenvalues of a 2x2 matrix is very
      // small WERR( M ) = EPS * ABS( W( M ) ) * TWO IBLOCK( M ) = JBLK INDEXW(
      // M ) = 1 ENDIF L2 = TMP1 + DISC IF( WL.GE. L2-PIVMIN ) $         NWL =
      // NWL + 1 IF( WU.GE. L2-PIVMIN ) $         NWU = NWU + 1 IF(
      // IRANGE.EQ.ALLRNG .OR. ( WL.LT.L2-PIVMIN .AND. WU.GE. $ L2-PIVMIN ) )
      // THEN M = M + 1 W( M ) = L2
      // *              The uncertainty of eigenvalues of a 2x2 matrix is very
      // small WERR( M ) = EPS * ABS( W( M ) ) * TWO IBLOCK( M ) = JBLK INDEXW(
      // M ) = 2 ENDIF
    } else {
      // General Case - block of size IN >= 2
      // Compute local Gerschgorin interval and use it as the initial
      // interval for DLAEBZ
      gu = D(ibegin);
      gl = D(ibegin);
      tmp1 = zero;
      for (j = ibegin; j <= iend; ++j) {
        gl = Min(gl, GERS(2 * j - 1));
        gu = Max(gu, GERS(2 * j));
      }
      // [JAN/28/2009]
      // change SPDIAM by TNORM in lines 2 and 3 thereafter
      // line 1: remove computation of SPDIAM (not useful anymore)
      // SPDIAM = GU - GL
      // GL = GL - FUDGE*SPDIAM*EPS*IN - FUDGE*PIVMIN
      // GU = GU + FUDGE*SPDIAM*EPS*IN + FUDGE*PIVMIN
      gl = gl - fudge * tnorm * eps * in - fudge * pivmin;
      gu = gu + fudge * tnorm * eps * in + fudge * pivmin;
      if (irange > 1) {
        if (gu < *wl) {
          // the local block contains none of the wanted eigenvalues
          nwl = nwl + in;
          nwu = nwu + in;
          goto L70;
        }
        // refine search interval if possible, only range (WL,WU] matters
        gl = Max(gl, *wl);
        gu = Min(gu, *wu);
        if (gl >= gu) goto L70;
      }
      // Find negcount of initial interval boundaries GL and GU
      WORK(n + 1) = gl;
      WORK(n + in + 1) = gu;
      iinfo = Laebz(1, 0, in, in, 1, nb, atoli, rtoli, pivmin, D.ptr(ibegin),
                    E.ptr(ibegin), E2.ptr(ibegin), idumma, WORK.ptr(n + 1),
                    WORK.ptr(n + 2 * in + 1), &im, iwork, W.ptr(*m + 1),
                    IBLOCK.ptr(*m + 1));
      if (iinfo != 0) {
        info = iinfo;
        return info;
      }
      nwl = nwl + IWORK(1);
      nwu = nwu + IWORK(in + 1);
      iwoff = *m - IWORK(1);
      // Compute Eigenvalues
      itmax =
          static_cast<blasint>((std::log(gu - gl + pivmin) - std::log(pivmin)) /
                               std::log(two)) +
          2;
      iinfo = Laebz(2, itmax, in, in, 1, nb, atoli, rtoli, pivmin,
                    D.ptr(ibegin), E.ptr(ibegin), E2.ptr(ibegin), idumma,
                    WORK.ptr(n + 1), WORK.ptr(n + 2 * in + 1), &iout, iwork,
                    W.ptr(*m + 1), IBLOCK.ptr(*m + 1));
      if (iinfo != 0) {
        info = iinfo;
        return info;
      }
      // Copy eigenvalues into W and IBLOCK
      // Use -JBLK for block number for unconverged eigenvalues.
      // Loop over the number of output intervals from DLAEBZ
      for (j = 1; j <= iout; ++j) {
        // eigenvalue approximation is middle point of interval
        tmp1 = half * (WORK(j + n) + WORK(j + in + n));
        // semi length of error interval
        tmp2 = half * std::abs(WORK(j + n) - WORK(j + in + n));
        if (j > iout - iinfo) {
          // Flag non-convergence.
          ncnvrg = true;
          ib = -jblk;
        } else {
          ib = jblk;
        }
        for (je = IWORK(j) + 1 + iwoff; je <= IWORK(j + in) + iwoff; ++je) {
          W(je) = tmp1;
          WERR(je) = tmp2;
          INDEXW(je) = je - iwoff;
          IBLOCK(je) = ib;
        }
      }
      *m = *m + im;
    }
  L70:;
  }
  // If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
  // If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
  if (irange == indrng) {
    idiscl = il - 1 - nwl;
    idiscu = nwu - iu;
    if (idiscl > 0) {
      im = 0;
      for (je = 1; je <= *m; ++je) {
        // Remove some of the smallest eigenvalues from the left so that
        // at the end IDISCL =0. Move all eigenvalues up to the left.
        if (W(je) <= wlu && idiscl > 0) {
          idiscl = idiscl - 1;
        } else {
          im = im + 1;
          W(im) = W(je);
          WERR(im) = WERR(je);
          INDEXW(im) = INDEXW(je);
          IBLOCK(im) = IBLOCK(je);
        }
      }
      *m = im;
    }
    if (idiscu > 0) {
      // Remove some of the largest eigenvalues from the right so that
      // at the end IDISCU =0. Move all eigenvalues up to the left.
      im = *m + 1;
      for (je = *m; je >= 1; --je) {
        if (W(je) >= wul && idiscu > 0) {
          idiscu = idiscu - 1;
        } else {
          im = im - 1;
          W(im) = W(je);
          WERR(im) = WERR(je);
          INDEXW(im) = INDEXW(je);
          IBLOCK(im) = IBLOCK(je);
        }
      }
      jee = 0;
      for (je = im; je <= *m; ++je) {
        jee = jee + 1;
        W(jee) = W(je);
        WERR(jee) = WERR(je);
        INDEXW(jee) = INDEXW(je);
        IBLOCK(jee) = IBLOCK(je);
      }
      *m = *m - im + 1;
    }
    if (idiscl > 0 || idiscu > 0) {
      // Code to deal with effects of bad arithmetic. (If N(w) is
      // monotone non-decreasing, this should never happen.)
      // Some low eigenvalues to be discarded are not in (WL,WLU],
      // or high eigenvalues to be discarded are not in (WUL,WU]
      // so just kill off the smallest IDISCL/largest IDISCU
      // eigenvalues, by marking the corresponding IBLOCK = 0
      if (idiscl > 0) {
        wkill = *wu;
        for (jdisc = 1; jdisc <= idiscl; ++jdisc) {
          iw = 0;
          for (je = 1; je <= *m; ++je) {
            if (IBLOCK(je) != 0 && (W(je) < wkill || iw == 0)) {
              iw = je;
              wkill = W(je);
            }
          }
          IBLOCK(iw) = 0;
        }
      }
      if (idiscu > 0) {
        wkill = *wl;
        for (jdisc = 1; jdisc <= idiscu; ++jdisc) {
          iw = 0;
          for (je = 1; je <= *m; ++je) {
            if (IBLOCK(je) != 0 && (W(je) >= wkill || iw == 0)) {
              iw = je;
              wkill = W(je);
            }
          }
          IBLOCK(iw) = 0;
        }
      }
      // Now erase all eigenvalues with IBLOCK set to zero
      im = 0;
      for (je = 1; je <= *m; ++je) {
        if (IBLOCK(je) != 0) {
          im = im + 1;
          W(im) = W(je);
          WERR(im) = WERR(je);
          INDEXW(im) = INDEXW(je);
          IBLOCK(im) = IBLOCK(je);
        }
      }
      *m = im;
    }
    if (idiscl < 0 || idiscu < 0) {
      toofew = true;
    }
  }
  if ((irange == allrng && *m != n) ||
      (irange == indrng && *m != iu - il + 1)) {
    toofew = true;
  }
  // If ORDER='B', do nothing the eigenvalues are already sorted by
  // block.
  // If ORDER='E', sort the eigenvalues from smallest to largest
  if (Lsame(order, 'E') && nsplit > 1) {
    for (je = 1; je <= *m - 1; ++je) {
      ie = 0;
      tmp1 = W(je);
      for (j = je + 1; j <= *m; ++j) {
        if (W(j) < tmp1) {
          ie = j;
          tmp1 = W(j);
        }
      }
      if (ie != 0) {
        tmp2 = WERR(ie);
        itmp1 = IBLOCK(ie);
        itmp2 = INDEXW(ie);
        W(ie) = W(je);
        WERR(ie) = WERR(je);
        IBLOCK(ie) = IBLOCK(je);
        INDEXW(ie) = INDEXW(je);
        W(je) = tmp1;
        WERR(je) = tmp2;
        IBLOCK(je) = itmp1;
        INDEXW(je) = itmp2;
      }
    }
  }
  info = 0;
  if (ncnvrg) info = info + 1;
  if (toofew) info = info + 2;
  return info;
}

// xLARRE: splits the symmetric tridiagonal matrix (D, E; E2 = E**2) into
// NSPLIT blocks (ISPLIT; xLARRA with SPLTOL), and for each block finds a
// shift SIGMA (stored in E(end of block)) such that L D L**T = T - SIGMA*I
// is a relatively robust representation (stored in D and E), and the
// approximations W +- WERR (gaps WGAP) of its wanted eigenvalues (RANGE
// 'A', 'V': in (VL, VU], 'I': IL..IU; VL and VU are set to an interval
// containing them) by dqds (xLASQ2) or bisection (xLARRD, xLARRB, with
// RTOL1, RTOL2). Returns M, IBLOCK, INDEXW, the Gershgorin intervals GERS
// and PIVMIN. WORK has 6*N, IWORK 5*N elements. Returns INFO: 0, or < 0 /
// 2 if a subroutine failed (as documented in LAPACK). As in LAPACK, the
// single precision routine perturbs the representation by 4*EPS (double:
// 8*EPS) and bisects to RTL = 100*EPS (sqrt(EPS)) and BSRTOL =
// sqrt(EPS)*0.5E-3 (sqrt(EPS)).
template <typename R>
blasint Larre(char range, blasint n, R* vl, R* vu, blasint il, blasint iu, R* d,
              R* e, R* e2, R rtol1, R rtol2, R spltol, blasint* nsplit,
              blasint* isplit, blasint* m, R* w, R* werr, R* wgap,
              blasint* iblock, blasint* indexw, R* gers, R* pivmin, R* work,
              blasint* iwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R four = R(4);
  const R hndrd = R(100);
  // SLARRE perturbs by 4*EPS, DLARRE by 8*EPS.
  const R pert = std::is_same_v<R, float> ? R(4) : R(8);
  const R half = one / two;
  const R fourth = one / four;
  const R fac = half;
  const R maxgrowth = R(64);
  const R fudge = R(2);
  const blasint maxtry = 6;
  const blasint allrng = 1;
  const blasint indrng = 2;
  const blasint valrng = 3;
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<R> E2(e2);
  const Vector<R> GERS(gers);
  const Vector<blasint> IBLOCK(iblock);
  const Vector<blasint> INDEXW(indexw);
  blasint iseed[4];
  const Vector<blasint> ISEED(iseed);
  const Vector<blasint> ISPLIT(isplit);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WERR(werr);
  const Vector<R> WGAP(wgap);
  const Vector<R> WORK(work);
  blasint cnt, cnt1, cnt2, i, ibegin, idum, iend, iinfo, in, irange, j, jblk,
      mm, wbegin, wend;
  // Set for each block: when its eigenvalues are counted, or for DQDS.
  blasint indl = 0, indu = 0, mb = 0;
  bool forceb, norep, usedqd;
  R avgap, bsrtol, clwdth, dmax, dpivot, eabs, emax, eold, eps, gl, gu, isleft,
      isrght, rtl, rtol, s1, s2, safmin, sgndef, sigma, spdiam, tau, tmp, tmp1;
  blasint info = 0;
  *nsplit = 0;
  *m = 0;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  // RANGE is valid (checked by the caller); CNT, CNT1 and CNT2 are set
  // by xLARRC where they are read (MB > 1).
  irange = 0;
  cnt = cnt1 = cnt2 = 0;
  // Decode RANGE
  if (Lsame(range, 'A')) {
    irange = allrng;
  } else if (Lsame(range, 'V')) {
    irange = valrng;
  } else if (Lsame(range, 'I')) {
    irange = indrng;
  }
  // Get machine constants
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  // Set parameters
  if constexpr (std::is_same_v<R, float>) {
    rtl = hndrd * eps;
    bsrtol = std::sqrt(eps) * R(0.5e-3);
  } else {
    rtl = std::sqrt(eps);
    bsrtol = std::sqrt(eps);
  }
  // Treat case of 1x1 matrix for quick return
  if (n == 1) {
    if ((irange == allrng) ||
        ((irange == valrng) && (D(1) > *vl) && (D(1) <= *vu)) ||
        ((irange == indrng) && (il == 1) && (iu == 1))) {
      *m = 1;
      W(1) = D(1);
      // The computation error of the eigenvalue is zero
      WERR(1) = zero;
      WGAP(1) = zero;
      IBLOCK(1) = 1;
      INDEXW(1) = 1;
      GERS(1) = D(1);
      GERS(2) = D(1);
    }
    // store the shift for the initial RRR, which is zero in this case
    E(1) = zero;
    return info;
  }
  // General case: tridiagonal matrix of order > 1
  // Init WERR, WGAP. Compute Gerschgorin intervals and spectral diameter.
  // Compute maximum off-diagonal entry and *pivmin.
  gl = D(1);
  gu = D(1);
  eold = zero;
  emax = zero;
  E(n) = zero;
  for (i = 1; i <= n; ++i) {
    WERR(i) = zero;
    WGAP(i) = zero;
    eabs = std::abs(E(i));
    if (eabs >= emax) {
      emax = eabs;
    }
    tmp1 = eabs + eold;
    GERS(2 * i - 1) = D(i) - tmp1;
    gl = Min(gl, GERS(2 * i - 1));
    GERS(2 * i) = D(i) + tmp1;
    gu = Max(gu, GERS(2 * i));
    eold = eabs;
  }
  // The minimum pivot allowed in the Sturm sequence for T
  *pivmin = safmin * Max(one, (emax * emax));
  // Compute spectral diameter. The Gerschgorin bounds give an
  // estimate that is wrong by at most a factor of SQRT(2)
  spdiam = gu - gl;
  // Compute splitting points
  iinfo = Larra(n, d, e, e2, spltol, spdiam, nsplit, isplit);
  // Can force use of bisection instead of faster DQDS.
  // Option left in the code for future multisection work.
  forceb = false;
  // Initialize USEDQD, DQDS should be used for ALLRNG unless someone
  // explicitly wants bisection.
  usedqd = ((irange == allrng) && (!forceb));
  if ((irange == allrng) && (!forceb)) {
    // Set interval [VL,VU] that contains all eigenvalues
    *vl = gl;
    *vu = gu;
  } else {
    // We call DLARRD to find crude approximations to the eigenvalues
    // in the desired range. In case IRANGE = INDRNG, we also obtain the
    // interval (VL,VU] that contains all the wanted eigenvalues.
    // An interval [LEFT,RIGHT] has converged if
    // RIGHT-LEFT.LT.RTOL*MAX(ABS(LEFT),ABS(RIGHT))
    // DLARRD needs a WORK of size 4*N, IWORK of size 3*N
    iinfo = Larrd(range, 'B', n, *vl, *vu, il, iu, gers, bsrtol, d, e, e2,
                  *pivmin, *nsplit, isplit, &mm, w, werr, vl, vu, iblock,
                  indexw, work, iwork);
    if (iinfo != 0) {
      info = -1;
      return info;
    }
    // Make sure that the entries M+1 to N in W, WERR, IBLOCK, INDEXW are 0
    for (i = mm + 1; i <= n; ++i) {
      W(i) = zero;
      WERR(i) = zero;
      IBLOCK(i) = 0;
      INDEXW(i) = 0;
    }
  }
  // **
  // Loop over unreduced blocks
  ibegin = 1;
  wbegin = 1;
  for (jblk = 1; jblk <= *nsplit; ++jblk) {
    iend = ISPLIT(jblk);
    in = iend - ibegin + 1;
    // 1 X 1 block
    if (in == 1) {
      if ((irange == allrng) ||
          ((irange == valrng) && (D(ibegin) > *vl) && (D(ibegin) <= *vu)) ||
          ((irange == indrng) && (IBLOCK(wbegin) == jblk))) {
        *m = *m + 1;
        W(*m) = D(ibegin);
        WERR(*m) = zero;
        // The gap for a single block doesn't matter for the later
        // algorithm and is assigned an arbitrary large value
        WGAP(*m) = zero;
        IBLOCK(*m) = jblk;
        INDEXW(*m) = 1;
        wbegin = wbegin + 1;
      }
      // E( IEND ) holds the shift for the initial RRR
      E(iend) = zero;
      ibegin = iend + 1;
      goto L170;
    }
    // Blocks of size larger than 1x1
    // E( IEND ) will hold the shift for the initial RRR, for now set it =0
    E(iend) = zero;
    // Find local outer bounds GL,GU for the block
    gl = D(ibegin);
    gu = D(ibegin);
    for (i = ibegin; i <= iend; ++i) {
      gl = Min(GERS(2 * i - 1), gl);
      gu = Max(GERS(2 * i), gu);
    }
    spdiam = gu - gl;
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): as LAPACK
    if (!((irange == allrng) && (!forceb))) {
      // Count the number of eigenvalues in the current block.
      mb = 0;
      for (i = wbegin; i <= mm; ++i) {
        if (IBLOCK(i) == jblk) {
          mb = mb + 1;
        } else {
          goto L21;
        }
      }
    L21:
      if (mb == 0) {
        // No eigenvalue in the current block lies in the desired range
        // E( IEND ) holds the shift for the initial RRR
        E(iend) = zero;
        ibegin = iend + 1;
        goto L170;
      } else {
        // Decide whether dqds or bisection is more efficient
        usedqd = ((mb > fac * in) && (!forceb));
        wend = wbegin + mb - 1;
        // Calculate gaps for the current block
        // In later stages, when representations for individual
        // eigenvalues are different, we use SIGMA = E( IEND ).
        sigma = zero;
        for (i = wbegin; i <= wend - 1; ++i) {
          WGAP(i) = Max(zero, W(i + 1) - WERR(i + 1) - (W(i) + WERR(i)));
        }
        WGAP(wend) = Max(zero, *vu - sigma - (W(wend) + WERR(wend)));
        // Find local index of the first and last desired evalue.
        indl = INDEXW(wbegin);
        indu = INDEXW(wend);
      }
    }
    if (((irange == allrng) && (!forceb)) || usedqd) {
      // Case of DQDS
      // Find approximations to the extremal eigenvalues of the block
      iinfo = Larrk(in, 1, gl, gu, D.ptr(ibegin), E2.ptr(ibegin), *pivmin, rtl,
                    &tmp, &tmp1);
      if (iinfo != 0) {
        info = -1;
        return info;
      }
      isleft = Max(gl, tmp - tmp1 - hndrd * eps * std::abs(tmp - tmp1));
      iinfo = Larrk(in, in, gl, gu, D.ptr(ibegin), E2.ptr(ibegin), *pivmin, rtl,
                    &tmp, &tmp1);
      if (iinfo != 0) {
        info = -1;
        return info;
      }
      isrght = Min(gu, tmp + tmp1 + hndrd * eps * std::abs(tmp + tmp1));
      // Improve the estimate of the spectral diameter
      spdiam = isrght - isleft;
    } else {
      // Case of bisection
      // Find approximations to the wanted extremal eigenvalues
      isleft = Max(gl, W(wbegin) - WERR(wbegin) -
                           hndrd * eps * std::abs(W(wbegin) - WERR(wbegin)));
      isrght = Min(gu, W(wend) + WERR(wend) +
                           hndrd * eps * std::abs(W(wend) + WERR(wend)));
    }
    // Decide whether the base representation for the current block
    // L_JBLK D_JBLK L_JBLK^T = T_JBLK - sigma_JBLK I
    // should be on the left or the right end of the current block.
    // The strategy is to shift to the end which is "more populated"
    // Furthermore, decide whether to use DQDS for the computation of
    // the eigenvalue approximations at the end of DLARRE or bisection.
    // dqds is chosen if all eigenvalues are desired or the number of
    // eigenvalues to be computed is large compared to the blocksize.
    if ((irange == allrng) && (!forceb)) {
      // If all the eigenvalues have to be computed, we use dqd
      usedqd = true;
      // INDL is the local index of the first eigenvalue to compute
      indl = 1;
      indu = in;
      // MB =  number of eigenvalues to compute
      mb = in;
      wend = wbegin + mb - 1;
      // Define 1/4 and 3/4 points of the spectrum
      s1 = isleft + fourth * spdiam;
      s2 = isrght - fourth * spdiam;
    } else {
      // DLARRD has computed IBLOCK and INDEXW for each eigenvalue
      // approximation.
      // choose sigma
      if (usedqd) {
        s1 = isleft + fourth * spdiam;
        s2 = isrght - fourth * spdiam;
      } else {
        tmp = Min(isrght, *vu) - Max(isleft, *vl);
        s1 = Max(isleft, *vl) + fourth * tmp;
        s2 = Min(isrght, *vu) - fourth * tmp;
      }
    }
    // Compute the negcount at the 1/4 and 3/4 points
    if (mb > 1) {
      iinfo = Larrc('T', in, s1, s2, D.ptr(ibegin), E.ptr(ibegin), *pivmin,
                    &cnt, &cnt1, &cnt2);
    }
    if (mb == 1) {
      sigma = gl;
      sgndef = one;
    } else if (cnt1 - indl >= indu - cnt2) {
      if ((irange == allrng) && (!forceb)) {
        sigma = Max(isleft, gl);
      } else if (usedqd) {
        // use Gerschgorin bound as shift to get pos def matrix
        // for dqds
        sigma = isleft;
      } else {
        // use approximation of the first desired eigenvalue of the
        // block as shift
        sigma = Max(isleft, *vl);
      }
      sgndef = one;
    } else {
      if ((irange == allrng) && (!forceb)) {
        sigma = Min(isrght, gu);
      } else if (usedqd) {
        // use Gerschgorin bound as shift to get neg def matrix
        // for dqds
        sigma = isrght;
      } else {
        // use approximation of the first desired eigenvalue of the
        // block as shift
        sigma = Min(isrght, *vu);
      }
      sgndef = -one;
    }
    // An initial SIGMA has been chosen that will be used for computing
    // T - SIGMA I = L D L^T
    // Define the increment TAU of the shift in case the initial shift
    // needs to be refined to obtain a factorization with not too much
    // element growth.
    if (usedqd) {
      // The initial SIGMA was to the outer end of the spectrum
      // the matrix is definite and we need not retreat.
      tau = spdiam * eps * n + two * *pivmin;
      tau = Max(tau, two * eps * std::abs(sigma));
    } else {
      if (mb > 1) {
        clwdth = W(wend) + WERR(wend) - W(wbegin) - WERR(wbegin);
        avgap = std::abs(clwdth / R(wend - wbegin));
        if (sgndef == one) {
          tau = half * Max(WGAP(wbegin), avgap);
          tau = Max(tau, WERR(wbegin));
        } else {
          tau = half * Max(WGAP(wend - 1), avgap);
          tau = Max(tau, WERR(wend));
        }
      } else {
        tau = WERR(wbegin);
      }
    }
    for (idum = 1; idum <= maxtry; ++idum) {
      // Compute L D L^T factorization of tridiagonal matrix T - sigma I.
      // Store D in WORK(1:IN), L in WORK(IN+1:2*IN), and reciprocals of
      // pivots in WORK(2*IN+1:3*IN)
      dpivot = D(ibegin) - sigma;
      WORK(1) = dpivot;
      dmax = std::abs(WORK(1));
      j = ibegin;
      for (i = 1; i <= in - 1; ++i) {
        WORK(2 * in + i) = one / WORK(i);
        tmp = E(j) * WORK(2 * in + i);
        WORK(in + i) = tmp;
        dpivot = (D(j + 1) - sigma) - tmp * E(j);
        WORK(i + 1) = dpivot;
        dmax = Max(dmax, std::abs(dpivot));
        j = j + 1;
      }
      // check for element growth
      if (dmax > maxgrowth * spdiam) {
        norep = true;  // NOLINT(readability-simplify-boolean-expr): as LAPACK
      } else {
        norep = false;
      }
      if (usedqd && !norep) {
        // Ensure the definiteness of the representation
        // All entries of D (of L D L^T) must have the same sign
        for (i = 1; i <= in; ++i) {
          tmp = sgndef * WORK(i);
          if (tmp < zero) norep = true;
        }
      }
      if (norep) {
        // Note that in the case of IRANGE=ALLRNG, we use the Gerschgorin
        // shift which makes the matrix definite. So we should end up
        // here really only in the case of IRANGE = VALRNG or INDRNG.
        if (idum == maxtry - 1) {
          if (sgndef == one) {
            // The fudged Gerschgorin shift should succeed
            sigma = gl - fudge * spdiam * eps * n - fudge * two * *pivmin;
          } else {
            sigma = gu + fudge * spdiam * eps * n + fudge * two * *pivmin;
          }
        } else {
          sigma = sigma - sgndef * tau;
          tau = two * tau;
        }
      } else {
        // an initial RRR is found
        goto L83;
      }
    }
    // if the program reaches this point, no base representation could be
    // found in MAXTRY iterations.
    info = 2;
    return info;
  L83:
    // At this point, we have found an initial base representation
    // T - SIGMA I = L D L^T with not too much element growth.
    // Store the shift.
    E(iend) = sigma;
    // Store D and L.
    Copy(in, work, 1, D.ptr(ibegin), 1);
    Copy(in - 1, WORK.ptr(in + 1), 1, E.ptr(ibegin), 1);
    if (mb > 1) {
      // Perturb each entry of the base representation by a small
      // (but random) relative amount to overcome difficulties with
      // glued matrices.
      for (i = 1; i <= 4; ++i) {
        ISEED(i) = 1;
      }
      Larnv(2, iseed, 2 * in - 1, WORK.ptr(1));
      for (i = 1; i <= in - 1; ++i) {
        D(ibegin + i - 1) = D(ibegin + i - 1) * (one + eps * pert * WORK(i));
        E(ibegin + i - 1) =
            E(ibegin + i - 1) * (one + eps * pert * WORK(in + i));
      }
      D(iend) = D(iend) * (one + eps * four * WORK(in));
    }
    // Don't update the Gerschgorin intervals because keeping track
    // of the updates would be too much work in DLARRV.
    // We update W instead and use it to locate the proper Gerschgorin
    // intervals.
    // Compute the required eigenvalues of L D L' by bisection or dqds
    if (!usedqd) {
      // If DLARRD has been used, shift the eigenvalue approximations
      // according to their representation. This is necessary for
      // a uniform DLARRV since dqds computes eigenvalues of the
      // shifted representation. In DLARRV, W will always hold the
      // UNshifted eigenvalue approximation.
      for (j = wbegin; j <= wend; ++j) {
        W(j) = W(j) - sigma;
        WERR(j) = WERR(j) + std::abs(W(j)) * eps;
      }
      // call DLARRB to reduce eigenvalue error of the approximations
      // from DLARRD
      for (i = ibegin; i <= iend - 1; ++i) {
        WORK(i) = D(i) * (E(i) * E(i));
      }
      // use bisection to find EV from INDL to INDU
      iinfo =
          Larrb(in, D.ptr(ibegin), WORK.ptr(ibegin), indl, indu, rtol1, rtol2,
                indl - 1, W.ptr(wbegin), WGAP.ptr(wbegin), WERR.ptr(wbegin),
                WORK.ptr(2 * n + 1), iwork, *pivmin, spdiam, in);
      if (iinfo != 0) {
        info = -4;
        return info;
      }
      // DLARRB computes all gaps correctly except for the last one
      // Record distance to VU/GU
      WGAP(wend) = Max(zero, (*vu - sigma) - (W(wend) + WERR(wend)));
      for (i = indl; i <= indu; ++i) {
        *m = *m + 1;
        IBLOCK(*m) = jblk;
        INDEXW(*m) = i;
      }
    } else {
      // Call dqds to get all eigs (and then possibly delete unwanted
      // eigenvalues).
      // Note that dqds finds the eigenvalues of the L D L^T representation
      // of T to high relative accuracy. High relative accuracy
      // might be lost when the shift of the RRR is subtracted to obtain
      // the eigenvalues of T. However, T is not guaranteed to define its
      // eigenvalues to high relative accuracy anyway.
      // Set RTOL to the order of the tolerance used in DLASQ2
      // This is an ESTIMATED error, the worst case bound is 4*N*EPS
      // which is usually too large and requires unnecessary work to be
      // done by bisection when computing the eigenvectors
      rtol = std::log(R(in)) * four * eps;
      j = ibegin;
      for (i = 1; i <= in - 1; ++i) {
        WORK(2 * i - 1) = std::abs(D(j));
        WORK(2 * i) = E(j) * E(j) * WORK(2 * i - 1);
        j = j + 1;
      }
      WORK(2 * in - 1) = std::abs(D(iend));
      WORK(2 * in) = zero;
      iinfo = Lasq2(in, work);
      if (iinfo != 0) {
        // If IINFO = -5 then an index is part of a tight cluster
        // and should be changed. The index is in IWORK(1) and the
        // gap is in WORK(N+1)
        info = -5;
        return info;
      } else {
        // Test that all eigenvalues are positive as expected
        for (i = 1; i <= in; ++i) {
          if (WORK(i) < zero) {
            info = -6;
            return info;
          }
        }
      }
      if (sgndef > zero) {
        for (i = indl; i <= indu; ++i) {
          *m = *m + 1;
          W(*m) = WORK(in - i + 1);
          IBLOCK(*m) = jblk;
          INDEXW(*m) = i;
        }
      } else {
        for (i = indl; i <= indu; ++i) {
          *m = *m + 1;
          W(*m) = -WORK(i);
          IBLOCK(*m) = jblk;
          INDEXW(*m) = i;
        }
      }
      for (i = *m - mb + 1; i <= *m; ++i) {
        // the value of RTOL below should be the tolerance in DLASQ2
        WERR(i) = rtol * std::abs(W(i));
      }
      for (i = *m - mb + 1; i <= *m - 1; ++i) {
        // compute the right gap between the intervals
        WGAP(i) = Max(zero, W(i + 1) - WERR(i + 1) - (W(i) + WERR(i)));
      }
      WGAP(*m) = Max(zero, (*vu - sigma) - (W(*m) + WERR(*m)));
    }
    // proceed with next block
    ibegin = iend + 1;
    wbegin = wend + 1;
  L170:;
  }
  return info;
}

// xLARRF: a new relatively robust representation L(+) D(+) L(+)**T =
// L D L**T - SIGMA*I (DPLUS, LPLUS) for the cluster CLSTRT..CLEND of
// eigenvalue approximations W +- WERR (gaps WGAP, CLGAPL, CLGAPR), with a
// shift SIGMA just outside the cluster and little element growth. WORK
// has 2*N elements. Returns INFO: 0, or 1 if no representation was
// acceptable. The first shifts are 2*EPS (single precision) or 4*EPS
// (double) relative outside the cluster, as in LAPACK.
template <typename R>
blasint Larrf(blasint n, R* d, R* l, R* ld, blasint clstrt, blasint clend, R* w,
              R* wgap, R* werr, R spdiam, R clgapl, R clgapr, R pivmin,
              R* sigma, R* dplus, R* lplus, R* work) {
  const R one = R(1);
  const R two = R(2);
  const R four = R(4);
  const R quart = R(0.25);
  const R maxgrowth1 = R(8);
  const R maxgrowth2 = R(8);
  const blasint ktrymax = 1;
  const blasint sleft = 1;
  const blasint sright = 2;
  const Vector<R> D(d);
  const Vector<R> DPLUS(dplus);
  const Vector<R> L(l);
  const Vector<R> LD(ld);
  const Vector<R> LPLUS(lplus);
  const Vector<R> W(w);
  const Vector<R> WERR(werr);
  const Vector<R> WGAP(wgap);
  const Vector<R> WORK(work);
  blasint i, indx, ktry, shift;
  bool dorrr1, forcer, nofail, sawnan1, sawnan2, tryrrr1;
  R avgap, bestshift, clwdth, eps, fact, fail, fail2, growthbound, ldelta,
      ldmax, lsigma, max1, max2, mingap, oldp, prod, rdelta, rdmax, rrr1, rrr2,
      rsigma, s, smlgrowth, tmp, znm2;
  blasint info = 0;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  fact = R(blasint{1} << ktrymax);
  eps = Lamch<R>('P');
  shift = 0;
  forcer = false;
  // Note that we cannot guarantee that for any of the shifts tried,
  // the factorization has a small or even moderate element growth.
  // There could be Ritz values at both ends of the cluster and despite
  // backing off, there are examples where all factorizations tried
  // (in IEEE mode, allowing zero pivots & infinities) have INFINITE
  // element growth.
  // For this reason, we should use PIVMIN in this subroutine so that at
  // least the L D L^T factorization exists. It can be checked afterwards
  // whether the element growth caused bad residuals/orthogonality.
  // Decide whether the code should accept the best among all
  // representations despite large element growth or signal INFO=1
  // Setting NOFAIL to .FALSE. for quick fix for bug 113
  nofail = false;
  // Compute the average gap length of the cluster
  clwdth = std::abs(W(clend) - W(clstrt)) + WERR(clend) + WERR(clstrt);
  avgap = clwdth / R(clend - clstrt);
  mingap = Min(clgapl, clgapr);
  // Initial values for shifts to both ends of cluster
  lsigma = Min(W(clstrt), W(clend)) - WERR(clstrt);
  rsigma = Max(W(clstrt), W(clend)) + WERR(clend);
  // Use a small fudge to make sure that we really shift to the outside
  if constexpr (std::is_same_v<R, float>) {
    lsigma = lsigma - std::abs(lsigma) * two * eps;
    rsigma = rsigma + std::abs(rsigma) * two * eps;
  } else {
    lsigma = lsigma - std::abs(lsigma) * four * eps;
    rsigma = rsigma + std::abs(rsigma) * four * eps;
  }
  // Compute upper bounds for how much to back off the initial shifts
  ldmax = quart * mingap + two * pivmin;
  rdmax = quart * mingap + two * pivmin;
  ldelta = Max(avgap, WGAP(clstrt)) / fact;
  rdelta = Max(avgap, WGAP(clend - 1)) / fact;
  // Initialize the record of the best representation found
  s = Lamch<R>('S');
  smlgrowth = one / s;
  fail = R(n - 1) * mingap / (spdiam * eps);
  fail2 = R(n - 1) * mingap / (spdiam * std::sqrt(eps));
  bestshift = lsigma;
  // while (KTRY <= KTRYMAX)
  ktry = 0;
  growthbound = maxgrowth1 * spdiam;
L5:
  sawnan1 = false;
  sawnan2 = false;
  // Ensure that we do not back off too much of the initial shifts
  ldelta = Min(ldmax, ldelta);
  rdelta = Min(rdmax, rdelta);
  // Compute the element growth when shifting to both ends of the cluster
  // accept the shift if there is no element growth at one of the two ends
  // Left end
  s = -lsigma;
  DPLUS(1) = D(1) + s;
  if (std::abs(DPLUS(1)) < pivmin) {
    DPLUS(1) = -pivmin;
    // Need to set SAWNAN1 because refined RRR test should not be used
    // in this case
    sawnan1 = true;
  }
  max1 = std::abs(DPLUS(1));
  for (i = 1; i <= n - 1; ++i) {
    LPLUS(i) = LD(i) / DPLUS(i);
    s = s * LPLUS(i) * L(i) - lsigma;
    DPLUS(i + 1) = D(i + 1) + s;
    if (std::abs(DPLUS(i + 1)) < pivmin) {
      DPLUS(i + 1) = -pivmin;
      // Need to set SAWNAN1 because refined RRR test should not be used
      // in this case
      sawnan1 = true;
    }
    max1 = Max(max1, std::abs(DPLUS(i + 1)));
  }
  sawnan1 = sawnan1 || std::isnan(max1);
  if (forcer || (max1 <= growthbound && !sawnan1)) {
    *sigma = lsigma;
    shift = sleft;
    goto L100;
  }
  // Right end
  s = -rsigma;
  WORK(1) = D(1) + s;
  if (std::abs(WORK(1)) < pivmin) {
    WORK(1) = -pivmin;
    // Need to set SAWNAN2 because refined RRR test should not be used
    // in this case
    sawnan2 = true;
  }
  max2 = std::abs(WORK(1));
  for (i = 1; i <= n - 1; ++i) {
    WORK(n + i) = LD(i) / WORK(i);
    s = s * WORK(n + i) * L(i) - rsigma;
    WORK(i + 1) = D(i + 1) + s;
    if (std::abs(WORK(i + 1)) < pivmin) {
      WORK(i + 1) = -pivmin;
      // Need to set SAWNAN2 because refined RRR test should not be used
      // in this case
      sawnan2 = true;
    }
    max2 = Max(max2, std::abs(WORK(i + 1)));
  }
  sawnan2 = sawnan2 || std::isnan(max2);
  if (forcer || (max2 <= growthbound && !sawnan2)) {
    *sigma = rsigma;
    shift = sright;
    goto L100;
  }
  // If we are at this point, both shifts led to too much element growth
  // Record the better of the two shifts (provided it didn't lead to NaN)
  if (sawnan1 && sawnan2) {
    // both MAX1 and MAX2 are NaN
    goto L50;
  } else {
    if (!sawnan1) {
      indx = 1;
      if (max1 <= smlgrowth) {
        smlgrowth = max1;
        bestshift = lsigma;
      }
    }
    if (!sawnan2) {
      if (sawnan1 || max2 <= max1) indx = 2;
      if (max2 <= smlgrowth) {
        smlgrowth = max2;
        bestshift = rsigma;
      }
    }
  }
  // If we are here, both the left and the right shift led to
  // element growth. If the element growth is moderate, then
  // we may still accept the representation, if it passes a
  // refined test for RRR. This test supposes that no NaN occurred.
  // Moreover, we use the refined RRR test only for isolated clusters.
  if ((clwdth < mingap / R(128)) && (Min(max1, max2) < fail2) && (!sawnan1) &&
      (!sawnan2)) {
    dorrr1 = true;  // NOLINT(readability-simplify-boolean-expr): as LAPACK
  } else {
    dorrr1 = false;
  }
  tryrrr1 = true;
  if (tryrrr1 && dorrr1) {
    if (indx == 1) {
      tmp = std::abs(DPLUS(n));
      znm2 = one;
      prod = one;
      oldp = one;
      for (i = n - 1; i >= 1; --i) {
        if (prod <= eps) {
          prod = ((DPLUS(i + 1) * WORK(n + i + 1)) / (DPLUS(i) * WORK(n + i))) *
                 oldp;
        } else {
          prod = prod * std::abs(WORK(n + i));
        }
        oldp = prod;
        znm2 = znm2 + (prod * prod);
        tmp = Max(tmp, std::abs(DPLUS(i) * prod));
      }
      rrr1 = tmp / (spdiam * std::sqrt(znm2));
      if (rrr1 <= maxgrowth2) {
        *sigma = lsigma;
        shift = sleft;
        goto L100;
      }
    } else if (indx == 2) {
      tmp = std::abs(WORK(n));
      znm2 = one;
      prod = one;
      oldp = one;
      for (i = n - 1; i >= 1; --i) {
        if (prod <= eps) {
          prod = ((WORK(i + 1) * LPLUS(i + 1)) / (WORK(i) * LPLUS(i))) * oldp;
        } else {
          prod = prod * std::abs(LPLUS(i));
        }
        oldp = prod;
        znm2 = znm2 + (prod * prod);
        tmp = Max(tmp, std::abs(WORK(i) * prod));
      }
      rrr2 = tmp / (spdiam * std::sqrt(znm2));
      if (rrr2 <= maxgrowth2) {
        *sigma = rsigma;
        shift = sright;
        goto L100;
      }
    }
  }
L50:
  if (ktry < ktrymax) {
    // If we are here, both shifts failed also the RRR test.
    // Back off to the outside
    lsigma = Max(lsigma - ldelta, lsigma - ldmax);
    rsigma = Min(rsigma + rdelta, rsigma + rdmax);
    ldelta = two * ldelta;
    rdelta = two * rdelta;
    ktry = ktry + 1;
    goto L5;
  } else {
    // None of the representations investigated satisfied our
    // criteria. Take the best one we found.
    if ((smlgrowth < fail) || nofail) {
      lsigma = bestshift;
      rsigma = bestshift;
      forcer = true;
      goto L5;
    } else {
      info = 1;
      return info;
    }
  }
L100:
  if (shift == sleft) {
  } else if (shift == sright) {
    // store new L and D back into DPLUS, LPLUS
    Copy(n, work, 1, dplus, 1);
    Copy(n - 1, WORK.ptr(n + 1), 1, lplus, 1);
  }
  return info;
}

// xLARRJ: refines by bisection the eigenvalue approximations W(i-OFFSET)
// +- WERR(i-OFFSET), i = IFIRST..ILAST, of the symmetric tridiagonal
// matrix (D, E2) to the relative accuracy RTOL. WORK has 2*N, IWORK 2*N
// elements. Returns INFO = 0.
template <typename R>
blasint Larrj(blasint n, R* d, R* e2, blasint ifirst, blasint ilast, R rtol,
              blasint offset, R* w, R* werr, R* work, blasint* iwork, R pivmin,
              R spdiam) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = R(0.5);
  const Vector<R> D(d);
  const Vector<R> E2(e2);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WERR(werr);
  const Vector<R> WORK(work);
  blasint cnt, i, i1, i2, ii, iter, j, k, maxitr, next, nint, olnint, p, prev,
      savi1;
  R dplus, fac, left, mid, right, s, tmp, width;
  blasint info = 0;
  // Quick return if possible
  if (n <= 0) {
    return info;
  }
  maxitr = static_cast<blasint>((std::log(spdiam + pivmin) - std::log(pivmin)) /
                                std::log(two)) +
           2;
  // Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
  // The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
  // Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 )
  // for an unconverged interval is set to the index of the next unconverged
  // interval, and is -1 or 0 for a converged interval. Thus a linked
  // list of unconverged intervals is set up.
  i1 = ifirst;
  i2 = ilast;
  // The number of unconverged intervals
  nint = 0;
  // The last unconverged interval found
  prev = 0;
  for (i = i1; i <= i2; ++i) {
    k = 2 * i;
    ii = i - offset;
    left = W(ii) - WERR(ii);
    mid = W(ii);
    right = W(ii) + WERR(ii);
    width = right - mid;
    tmp = Max(std::abs(left), std::abs(right));
    // The following test prevents the test of converged intervals
    if (width < rtol * tmp) {
      // This interval has already converged and does not need refinement.
      // (Note that the gaps might change through refining the
      // eigenvalues, however, they can only get bigger.)
      // Remove it from the list.
      IWORK(k - 1) = -1;
      // Make sure that I1 always points to the first unconverged interval
      if ((i == i1) && (i < i2)) i1 = i + 1;
      if ((prev >= i1) && (i <= i2)) IWORK(2 * prev - 1) = i + 1;
    } else {
      // unconverged interval found
      prev = i;
      // Make sure that [LEFT,RIGHT] contains the desired eigenvalue
      // Do while( CNT(LEFT).GT.I-1 )
      fac = one;
    L20:
      cnt = 0;
      s = left;
      dplus = D(1) - s;
      if (dplus < zero) cnt = cnt + 1;
      for (j = 2; j <= n; ++j) {
        dplus = D(j) - s - E2(j - 1) / dplus;
        if (dplus < zero) cnt = cnt + 1;
      }
      if (cnt > i - 1) {
        left = left - WERR(ii) * fac;
        fac = two * fac;
        goto L20;
      }
      // Do while( CNT(RIGHT).LT.I )
      fac = one;
    L50:
      cnt = 0;
      s = right;
      dplus = D(1) - s;
      if (dplus < zero) cnt = cnt + 1;
      for (j = 2; j <= n; ++j) {
        dplus = D(j) - s - E2(j - 1) / dplus;
        if (dplus < zero) cnt = cnt + 1;
      }
      if (cnt < i) {
        right = right + WERR(ii) * fac;
        fac = two * fac;
        goto L50;
      }
      nint = nint + 1;
      IWORK(k - 1) = i + 1;
      IWORK(k) = cnt;
    }
    WORK(k - 1) = left;
    WORK(k) = right;
  }
  savi1 = i1;
  // Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
  // and while (ITER.LT.MAXITR)
  iter = 0;
L80:
  prev = i1 - 1;
  i = i1;
  olnint = nint;
  for (p = 1; p <= olnint; ++p) {
    k = 2 * i;
    ii = i - offset;
    next = IWORK(k - 1);
    left = WORK(k - 1);
    right = WORK(k);
    mid = half * (left + right);
    // semiwidth of interval
    width = right - mid;
    tmp = Max(std::abs(left), std::abs(right));
    if ((width < rtol * tmp) || (iter == maxitr)) {
      // reduce number of unconverged intervals
      nint = nint - 1;
      // Mark interval as converged.
      IWORK(k - 1) = 0;
      if (i1 == i) {
        i1 = next;
      } else {
        // Prev holds the last unconverged interval previously examined
        if (prev >= i1) IWORK(2 * prev - 1) = next;
      }
      i = next;
      goto L100;
    }
    prev = i;
    // Perform one bisection step
    cnt = 0;
    s = mid;
    dplus = D(1) - s;
    if (dplus < zero) cnt = cnt + 1;
    for (j = 2; j <= n; ++j) {
      dplus = D(j) - s - E2(j - 1) / dplus;
      if (dplus < zero) cnt = cnt + 1;
    }
    if (cnt <= i - 1) {
      WORK(k - 1) = mid;
    } else {
      WORK(k) = mid;
    }
    i = next;
  L100:;
  }
  iter = iter + 1;
  // do another loop if there are still unconverged intervals
  // However, in the last iteration, all intervals are accepted
  // since this is the best we can do.
  if ((nint > 0) && (iter <= maxitr)) goto L80;
  // At this point, all the intervals have converged
  for (i = savi1; i <= ilast; ++i) {
    k = 2 * i;
    ii = i - offset;
    // All intervals marked by '0' have been refined.
    if (IWORK(k - 1) == 0) {
      W(ii) = half * (WORK(k - 1) + WORK(k));
      WERR(ii) = WORK(k) - W(ii);
    }
  }
  return info;
}

// xLARRK: the IW-th eigenvalue W +- WERR of the symmetric tridiagonal
// matrix (D, E2) by bisection of the Gershgorin interval [GL, GU] to the
// relative accuracy RELTOL. Returns INFO: 0, or -1 if the iteration did
// not converge.
template <typename R>
blasint Larrk(blasint n, blasint iw, R gl, R gu, R* d, R* e2, R pivmin,
              R reltol, R* w, R* werr) {
  blasint info;

  const R half = R(0.5);
  const R two = R(2);
  const R fudge = two;
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<R> E2(e2);
  blasint i, it, itmax, negcnt;
  R atoli, eps, left, mid, right, rtoli, tmp1, tmp2, tnorm;
  if (n <= 0) {
    blasint info = 0;
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  tnorm = Max(std::abs(gl), std::abs(gu));
  rtoli = reltol;
  atoli = fudge * two * pivmin;
  itmax = static_cast<blasint>((std::log(tnorm + pivmin) - std::log(pivmin)) /
                               std::log(two)) +
          2;
  info = -1;
  left = gl - fudge * tnorm * eps * n - fudge * two * pivmin;
  right = gu + fudge * tnorm * eps * n + fudge * two * pivmin;
  it = 0;
L10:
  // Check if interval converged or maximum number of iterations reached
  tmp1 = std::abs(right - left);
  tmp2 = Max(std::abs(right), std::abs(left));
  if (tmp1 < Max(atoli, pivmin, rtoli * tmp2)) {
    info = 0;
    goto L30;
  }
  if (it > itmax) goto L30;
  // Count number of negative pivots for mid-point
  it = it + 1;
  mid = half * (left + right);
  negcnt = 0;
  tmp1 = D(1) - mid;
  if (std::abs(tmp1) < pivmin) tmp1 = -pivmin;
  if (tmp1 <= zero) negcnt = negcnt + 1;
  for (i = 2; i <= n; ++i) {
    tmp1 = D(i) - E2(i - 1) / tmp1 - mid;
    if (std::abs(tmp1) < pivmin) tmp1 = -pivmin;
    if (tmp1 <= zero) negcnt = negcnt + 1;
  }
  if (negcnt >= iw) {
    right = mid;
  } else {
    left = mid;
  }
  goto L10;
L30:
  // Converged or maximum number of iterations reached
  *w = half * (left + right);
  *werr = half * std::abs(right - left);
  return info;
}

// xLARRR: whether the symmetric tridiagonal matrix (D, E) warrants the
// relatively accurate computation of its eigenvalues (it is scaled
// diagonally dominant). Returns INFO: 0 if so, 1 otherwise.
template <typename R>
blasint Larrr(blasint n, R* d, R* e) {
  blasint info;

  const R zero = R(0);
  const R relcond = R(0.999);
  const Vector<R> D(d);
  const Vector<R> E(e);
  blasint i;
  bool yesrel;
  R eps, offdig, offdig2, rmin, safmin, smlnum, tmp, tmp2;
  if (n <= 0) {
    blasint info = 0;
    return info;
  }
  // As a default, do NOT go for relative-accuracy preserving computations.
  info = 1;
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  smlnum = safmin / eps;
  rmin = std::sqrt(smlnum);
  // Tests for relative accuracy
  // Test for scaled diagonal dominance
  // Scale the diagonal entries to one and check whether the sum of the
  // off-diagonals is less than one
  // The sdd relative error bounds have a 1/(1- 2*x) factor in them,
  // x = max(OFFDIG + OFFDIG2), so when x is close to 1/2, no relative
  // accuracy is promised.  In the notation of the code fragment below,
  // 1/(1 - (OFFDIG + OFFDIG2)) is the condition number.
  // We don't think it is worth going into "sdd mode" unless the relative
  // condition number is reasonable, not 1/macheps.
  // The threshold should be compatible with other thresholds used in the
  // code. We set  OFFDIG + OFFDIG2 <= .999 =: RELCOND, it corresponds
  // to losing at most 3 decimal digits: 1 / (1 - (OFFDIG + OFFDIG2)) <= 1000
  // instead of the current OFFDIG + OFFDIG2 < 1
  yesrel = true;
  offdig = zero;
  tmp = std::sqrt(std::abs(D(1)));
  if (tmp < rmin) yesrel = false;
  if (!yesrel) goto L11;
  for (i = 2; i <= n; ++i) {
    tmp2 = std::sqrt(std::abs(D(i)));
    if (tmp2 < rmin) yesrel = false;
    if (!yesrel) goto L11;
    offdig2 = std::abs(E(i - 1)) / (tmp * tmp2);
    if (offdig + offdig2 >= relcond) yesrel = false;
    if (!yesrel) goto L11;
    tmp = tmp2;
    offdig = offdig2;
  }
L11:
  if (yesrel) {
    info = 0;
    return info;
  }
  // *** MORE TO BE IMPLEMENTED ***
  // Test if the lower bidiagonal matrix L from T = L D L^T
  // (zero shift facto) is well conditioned
  // Test if the upper bidiagonal matrix U from T = U D U^T
  // (zero shift facto) is well conditioned.
  // In this case, the matrix needs to be flipped and, at the end
  // of the eigenvector computation, the flip needs to be applied
  // to the computed eigenvectors (and the support)
  return info;
}

// xLARRV: the eigenvectors (columns DOL..DOU of the M wanted, in Z) of the
// symmetric tridiagonal matrix whose blocks (ISPLIT) are given by their
// root representations L D L**T (D, L; shift in L(end of block)) from
// xLARRE, with the eigenvalue approximations W +- WERR (gaps WGAP, block
// numbers IBLOCK, indices INDEXW, Gershgorin intervals GERS): builds the
// representation tree (child clusters where the relative gap is below
// MINRGP, xLARRF) and computes each singleton's eigenvector by xLAR1V with
// Rayleigh quotient or bisection (xLARRB) correction. W returns the
// refined eigenvalues, ISUPPZ the supports. WORK has 12*N, IWORK 7*N
// elements. RTOL1 and RTOL2 are set to 4*EPS if DOL > 1 or DOU < M (a
// pointer: the Fortran overwrites the caller's variables). For complex T
// the eigenvectors are real; ZLARRV keeps the child representations in
// WORK before it copies them into Z, so its WORK layout differs. Returns
// INFO: 0, or < 0 / 5 if a subroutine failed (as documented in LAPACK).
// VU is not used (as in LAPACK).
template <typename T>
blasint Larrv(blasint n, RealT<T> vl, RealT<T> /*vu*/, RealT<T>* d, RealT<T>* l,
              RealT<T> pivmin, blasint* isplit, blasint m, blasint dol,
              blasint dou, RealT<T> minrgp, RealT<T>* rtol1, RealT<T>* rtol2,
              RealT<T>* w, RealT<T>* werr, RealT<T>* wgap, blasint* iblock,
              blasint* indexw, RealT<T>* gers, T* z, blasint ldz,
              blasint* isuppz, RealT<T>* work, blasint* iwork) {
  using R = RealT<T>;

  const blasint maxitr = 10;
  const T czero = T(0);
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R three = R(3);
  const R four = R(4);
  const R half = R(0.5);
  const Vector<R> D(d);
  const Vector<R> GERS(gers);
  const Vector<blasint> IBLOCK(iblock);
  const Vector<blasint> INDEXW(indexw);
  const Vector<blasint> ISPLIT(isplit);
  const Vector<blasint> ISUPPZ(isuppz);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> L(l);
  const Vector<R> W(w);
  const Vector<R> WERR(werr);
  const Vector<R> WGAP(wgap);
  const Vector<R> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint done, i, ibegin, idone, iend, ii, iindc1, iindc2, iindr, iindwk,
      iinfo, im, in, indeig, indin1, indin2, indld, indlld, indwrk, isupmn,
      isupmx, iter, itmp1, j, jblk, k, miniwsize, minwsize, nclus, ndepth,
      negcnt, newcls, newfst, newftt, newlst, newsiz, offset, oldcls, oldfst,
      oldien, oldlst, oldncl, p, parity, q, wbegin, wend, windex, windmn,
      windpl, zfrom, zto, zusedl, zusedu, zusedw;
  bool eskip, needbs, stp2ii, tryrqc, usedbs, usedrq;
  R bstres, bstw, eps, fudge, gap, gaptol, gl, gu, lambda, left, lgap, mingma,
      nrminv, resid, rgap, right, rqcorr, rqtol, savgap, sgndef, sigma, spdiam,
      ssigma, tau, tmp, tol, ztz;
  blasint info = 0;
  // Quick return if possible
  if ((n <= 0) || (m <= 0)) {
    return info;
  }
  // The first N entries of WORK are reserved for the eigenvalues
  indld = n + 1;
  indlld = 2 * n + 1;
  if constexpr (IsComplex<T>::value) {
    indin1 = 3 * n + 1;
    indin2 = 4 * n + 1;
    indwrk = 5 * n + 1;
  } else {
    indwrk = 3 * n + 1;
  }
  minwsize = 12 * n;
  for (i = 1; i <= minwsize; ++i) {
    WORK(i) = zero;
  }
  // IWORK(IINDR+1:IINDR+N) hold the twist indices R for the
  // factorization used to compute the FP vector
  iindr = 0;
  // IWORK(IINDC1+1:IINC2+N) are used to store the clusters of the current
  // layer and the one above.
  iindc1 = n;
  iindc2 = 2 * n;
  iindwk = 3 * n + 1;
  miniwsize = 7 * n;
  for (i = 1; i <= miniwsize; ++i) {
    IWORK(i) = 0;
  }
  zusedl = 1;
  if (dol > 1) {
    // Set lower bound for use of Z
    zusedl = dol - 1;
  }
  zusedu = m;
  if (dou < m) {
    // Set lower bound for use of Z
    zusedu = dou + 1;
  }
  // The width of the part of Z that is used
  zusedw = zusedu - zusedl + 1;
  Laset('F', n, zusedw, czero, czero, Z.ptr(1, zusedl), ldz);
  eps = Lamch<R>('P');
  rqtol = two * eps;
  // Set expert flags for standard code.
  tryrqc = true;
  if ((dol == 1) && (dou == m)) {
  } else {
    // Only selected eigenpairs are computed. Since the other evalues
    // are not refined by RQ iteration, bisection has to compute to full
    // accuracy.
    *rtol1 = four * eps;
    *rtol2 = four * eps;
  }
  // The entries WBEGIN:WEND in W, WERR, WGAP correspond to the
  // desired eigenvalues. The support of the nonzero eigenvector
  // entries is contained in the interval IBEGIN:IEND.
  // Remark that if k eigenpairs are desired, then the eigenvectors
  // are stored in k contiguous columns of Z.
  // DONE is the number of eigenvectors already computed
  done = 0;
  ibegin = 1;
  wbegin = 1;
  for (jblk = 1; jblk <= IBLOCK(m); ++jblk) {
    iend = ISPLIT(jblk);
    sigma = L(iend);
    // Find the eigenvectors of the submatrix indexed IBEGIN
    // through IEND.
    wend = wbegin - 1;
  L15:
    if (wend < m) {
      if (IBLOCK(wend + 1) == jblk) {
        wend = wend + 1;
        goto L15;
      }
    }
    if (wend < wbegin) {
      ibegin = iend + 1;
      goto L170;
    } else if ((wend < dol) || (wbegin > dou)) {
      ibegin = iend + 1;
      wbegin = wend + 1;
      goto L170;
    }
    // Find local spectral diameter of the block
    gl = GERS(2 * ibegin - 1);
    gu = GERS(2 * ibegin);
    for (i = ibegin + 1; i <= iend; ++i) {
      gl = Min(GERS(2 * i - 1), gl);
      gu = Max(GERS(2 * i), gu);
    }
    spdiam = gu - gl;
    // OLDIEN is the last index of the previous block
    oldien = ibegin - 1;
    // Calculate the size of the current block
    in = iend - ibegin + 1;
    // The number of eigenvalues in the current block
    im = wend - wbegin + 1;
    // This is for a 1x1 block
    if (ibegin == iend) {
      done = done + 1;
      Z(ibegin, wbegin) = T(one);
      ISUPPZ(2 * wbegin - 1) = ibegin;
      ISUPPZ(2 * wbegin) = ibegin;
      W(wbegin) = W(wbegin) + sigma;
      WORK(wbegin) = W(wbegin);
      ibegin = iend + 1;
      wbegin = wbegin + 1;
      goto L170;
    }
    // The desired (shifted) eigenvalues are stored in W(WBEGIN:WEND)
    // Note that these can be approximations, in this case, the corresp.
    // entries of WERR give the size of the uncertainty interval.
    // The eigenvalue approximations will be refined when necessary as
    // high relative accuracy is required for the computation of the
    // corresponding eigenvectors.
    Copy<R>(im, W.ptr(wbegin), 1, WORK.ptr(wbegin), 1);
    // We store in W the eigenvalue approximations w.r.t. the original
    // matrix T.
    for (i = 1; i <= im; ++i) {
      W(wbegin + i - 1) = W(wbegin + i - 1) + sigma;
    }
    // NDEPTH is the current depth of the representation tree
    ndepth = 0;
    // PARITY is either 1 or 0
    parity = 1;
    // NCLUS is the number of clusters for the next level of the
    // representation tree, we start with NCLUS = 1 for the root
    nclus = 1;
    IWORK(iindc1 + 1) = 1;
    IWORK(iindc1 + 2) = im;
    // IDONE is the number of eigenvectors already computed in the current
    // block
    idone = 0;
    // loop while( IDONE.LT.IM )
    // generate the representation tree for the current block and
    // compute the eigenvectors
  L40:
    if (idone < im) {
      // This is a crude protection against infinitely deep trees
      if (ndepth > m) {
        info = -2;
        return info;
      }
      // breadth first processing of the current level of the representation
      // tree: OLDNCL = number of clusters on current level
      oldncl = nclus;
      // reset NCLUS to count the number of child clusters
      nclus = 0;
      parity = 1 - parity;
      if (parity == 0) {
        oldcls = iindc1;
        newcls = iindc2;
      } else {
        oldcls = iindc2;
        newcls = iindc1;
      }
      // Process the clusters on the current level
      for (i = 1; i <= oldncl; ++i) {
        j = oldcls + 2 * i;
        // OLDFST, OLDLST = first, last index of current cluster.
        // cluster indices start with 1 and are relative
        // to WBEGIN when accessing W, WGAP, WERR, Z
        oldfst = IWORK(j - 1);
        oldlst = IWORK(j);
        if (ndepth > 0) {
          // Retrieve relatively robust representation (RRR) of cluster
          // that has been computed at the previous level
          // The RRR is stored in Z and overwritten once the eigenvectors
          // have been computed or when the cluster is refined
          if ((dol == 1) && (dou == m)) {
            // Get representation from location of the leftmost evalue
            // of the cluster
            j = wbegin + oldfst - 1;
          } else {
            if (wbegin + oldfst - 1 < dol) {
              // Get representation from the left end of Z array
              j = dol - 1;
            } else if (wbegin + oldfst - 1 > dou) {
              // Get representation from the right end of Z array
              j = dou;
            } else {
              j = wbegin + oldfst - 1;
            }
          }
          if constexpr (IsComplex<T>::value) {
            for (k = 1; k <= in - 1; ++k) {
              D(ibegin + k - 1) = std::real(Z(ibegin + k - 1, j));
              L(ibegin + k - 1) = std::real(Z(ibegin + k - 1, j + 1));
            }
            D(iend) = std::real(Z(iend, j));
            sigma = std::real(Z(iend, j + 1));
          } else {
            Copy(in, Z.ptr(ibegin, j), 1, D.ptr(ibegin), 1);
            Copy(in - 1, Z.ptr(ibegin, j + 1), 1, L.ptr(ibegin), 1);
            sigma = Z(iend, j + 1);
          }
          // Set the corresponding entries in Z to zero
          Laset('F', in, 2, czero, czero, Z.ptr(ibegin, j), ldz);
        }
        // Compute DL and DLL of current RRR
        for (j = ibegin; j <= iend - 1; ++j) {
          tmp = D(j) * L(j);
          WORK(indld - 1 + j) = tmp;
          WORK(indlld - 1 + j) = tmp * L(j);
        }
        if (ndepth > 0) {
          // P and Q are index of the first and last eigenvalue to compute
          // within the current block
          p = INDEXW(wbegin - 1 + oldfst);
          q = INDEXW(wbegin - 1 + oldlst);
          // Offset for the arrays WORK, WGAP and WERR, i.e., the P-OFFSET
          // through the Q-OFFSET elements of these arrays are to be used.
          // OFFSET = P-OLDFST
          offset = INDEXW(wbegin) - 1;
          // perform limited bisection (if necessary) to get approximate
          // eigenvalues to the precision needed.
          iinfo = Larrb(in, D.ptr(ibegin), WORK.ptr(indlld + ibegin - 1), p, q,
                        *rtol1, *rtol2, offset, WORK.ptr(wbegin),
                        WGAP.ptr(wbegin), WERR.ptr(wbegin), WORK.ptr(indwrk),
                        IWORK.ptr(iindwk), pivmin, spdiam, in);
          if (iinfo != 0) {
            info = -1;
            return info;
          }
          // We also recompute the extremal gaps. W holds all eigenvalues
          // of the unshifted matrix and must be used for computation
          // of WGAP, the entries of WORK might stem from RRRs with
          // different shifts. The gaps from WBEGIN-1+OLDFST to
          // WBEGIN-1+OLDLST are correctly computed in DLARRB.
          // However, we only allow the gaps to become greater since
          // this is what should happen when we decrease WERR
          if (oldfst > 1) {
            WGAP(wbegin + oldfst - 2) =
                Max(WGAP(wbegin + oldfst - 2),
                    W(wbegin + oldfst - 1) - WERR(wbegin + oldfst - 1) -
                        W(wbegin + oldfst - 2) - WERR(wbegin + oldfst - 2));
          }
          if (wbegin + oldlst - 1 < wend) {
            WGAP(wbegin + oldlst - 1) =
                Max(WGAP(wbegin + oldlst - 1),
                    W(wbegin + oldlst) - WERR(wbegin + oldlst) -
                        W(wbegin + oldlst - 1) - WERR(wbegin + oldlst - 1));
          }
          // Each time the eigenvalues in WORK get refined, we store
          // the newly found approximation with all shifts applied in W
          for (j = oldfst; j <= oldlst; ++j) {
            W(wbegin + j - 1) = WORK(wbegin + j - 1) + sigma;
          }
        }
        // Process the current node.
        newfst = oldfst;
        for (j = oldfst; j <= oldlst; ++j) {
          // NOLINTNEXTLINE(bugprone-branch-clone): as LAPACK
          if (j == oldlst) {
            // we are at the right end of the cluster, this is also the
            // boundary of the child cluster
            newlst = j;
          } else if (WGAP(wbegin + j - 1) >=
                     minrgp * std::abs(WORK(wbegin + j - 1))) {
            // the right relative gap is big enough, the child cluster
            // (NEWFST,..,NEWLST) is well separated from the following
            newlst = j;
          } else {
            // inside a child cluster, the relative gap is not
            // big enough.
            goto L140;
          }
          // Compute size of child cluster found
          newsiz = newlst - newfst + 1;
          // NEWFTT is the place in Z where the new RRR or the computed
          // eigenvector is to be stored
          if ((dol == 1) && (dou == m)) {
            // Store representation at location of the leftmost evalue
            // of the cluster
            newftt = wbegin + newfst - 1;
          } else {
            if (wbegin + newfst - 1 < dol) {
              // Store representation at the left end of Z array
              newftt = dol - 1;
            } else if (wbegin + newfst - 1 > dou) {
              // Store representation at the right end of Z array
              newftt = dou;
            } else {
              newftt = wbegin + newfst - 1;
            }
          }
          if (newsiz > 1) {
            // Current child is not a singleton but a cluster.
            // Compute and store new representation of child.
            // Compute left and right cluster gap.
            // LGAP and RGAP are not computed from WORK because
            // the eigenvalue approximations may stem from RRRs
            // different shifts. However, W hold all eigenvalues
            // of the unshifted matrix. Still, the entries in WGAP
            // have to be computed from WORK since the entries
            // in W might be of the same order so that gaps are not
            // exhibited correctly for very close eigenvalues.
            if (newfst == 1) {
              lgap = Max(zero, W(wbegin) - WERR(wbegin) - vl);
            } else {
              lgap = WGAP(wbegin + newfst - 2);
            }
            rgap = WGAP(wbegin + newlst - 1);
            // Compute left- and rightmost eigenvalue of child
            // to high precision in order to shift as close
            // as possible and obtain as large relative gaps
            // as possible
            for (k = 1; k <= 2; ++k) {
              if (k == 1) {
                p = INDEXW(wbegin - 1 + newfst);
              } else {
                p = INDEXW(wbegin - 1 + newlst);
              }
              offset = INDEXW(wbegin) - 1;
              iinfo =
                  Larrb(in, D.ptr(ibegin), WORK.ptr(indlld + ibegin - 1), p, p,
                        rqtol, rqtol, offset, WORK.ptr(wbegin),
                        WGAP.ptr(wbegin), WERR.ptr(wbegin), WORK.ptr(indwrk),
                        IWORK.ptr(iindwk), pivmin, spdiam, in);
            }
            if ((wbegin + newlst - 1 < dol) || (wbegin + newfst - 1 > dou)) {
              // if the cluster contains no desired eigenvalues
              // skip the computation of that branch of the rep. tree
              // We could skip before the refinement of the extremal
              // eigenvalues of the child, but then the representation
              // tree could be different from the one when nothing is
              // skipped. For this reason we skip at this place.
              idone = idone + newlst - newfst + 1;
              goto L139;
            }
            // Compute RRR of child cluster.
            // Note that the new RRR is stored in Z
            // DLARRF needs LWORK = 2*N
            if constexpr (IsComplex<T>::value) {
              iinfo =
                  Larrf(in, D.ptr(ibegin), L.ptr(ibegin),
                        WORK.ptr(indld + ibegin - 1), newfst, newlst,
                        WORK.ptr(wbegin), WGAP.ptr(wbegin), WERR.ptr(wbegin),
                        spdiam, lgap, rgap, pivmin, &tau, WORK.ptr(indin1),
                        WORK.ptr(indin2), WORK.ptr(indwrk));
              // In the complex case, DLARRF cannot write
              // the new RRR directly into Z and needs an intermediate
              // workspace
              for (k = 1; k <= in - 1; ++k) {
                Z(ibegin + k - 1, newftt) = T(WORK(indin1 + k - 1), zero);
                Z(ibegin + k - 1, newftt + 1) = T(WORK(indin2 + k - 1), zero);
              }
              Z(iend, newftt) = T(WORK(indin1 + in - 1), zero);
            } else {
              iinfo =
                  Larrf(in, D.ptr(ibegin), L.ptr(ibegin),
                        WORK.ptr(indld + ibegin - 1), newfst, newlst,
                        WORK.ptr(wbegin), WGAP.ptr(wbegin), WERR.ptr(wbegin),
                        spdiam, lgap, rgap, pivmin, &tau, Z.ptr(ibegin, newftt),
                        Z.ptr(ibegin, newftt + 1), WORK.ptr(indwrk));
            }
            if (iinfo == 0) {
              // a new RRR for the cluster was found by DLARRF
              // update shift and store it
              ssigma = sigma + tau;
              Z(iend, newftt + 1) = T(ssigma);
              // WORK() are the midpoints and WERR() the semi-width
              // Note that the entries in W are unchanged.
              for (k = newfst; k <= newlst; ++k) {
                fudge = three * eps * std::abs(WORK(wbegin + k - 1));
                WORK(wbegin + k - 1) = WORK(wbegin + k - 1) - tau;
                fudge = fudge + four * eps * std::abs(WORK(wbegin + k - 1));
                // Fudge errors
                WERR(wbegin + k - 1) = WERR(wbegin + k - 1) + fudge;
                // Gaps are not fudged. Provided that WERR is small
                // when eigenvalues are close, a zero gap indicates
                // that a new representation is needed for resolving
                // the cluster. A fudge could lead to a wrong decision
                // of judging eigenvalues 'separated' which in
                // reality are not. This could have a negative impact
                // on the orthogonality of the computed eigenvectors.
              }
              nclus = nclus + 1;
              k = newcls + 2 * nclus;
              IWORK(k - 1) = newfst;
              IWORK(k) = newlst;
            } else {
              info = -2;
              return info;
            }
          } else {
            // Compute eigenvector of singleton
            iter = 0;
            tol = four * std::log(R(in)) * eps;
            k = newfst;
            windex = wbegin + k - 1;
            windmn = Max<blasint>(windex - 1, 1);
            windpl = Min<blasint>(windex + 1, m);
            lambda = WORK(windex);
            done = done + 1;
            // Check if eigenvector computation is to be skipped
            if ((windex < dol) || (windex > dou)) {
              eskip = true;
              goto L125;
            } else {
              eskip = false;
            }
            left = WORK(windex) - WERR(windex);
            right = WORK(windex) + WERR(windex);
            indeig = INDEXW(windex);
            // Note that since we compute the eigenpairs for a child,
            // all eigenvalue approximations are w.r.t the same shift.
            // In this case, the entries in WORK should be used for
            // computing the gaps since they exhibit even very small
            // differences in the eigenvalues, as opposed to the
            // entries in W which might "look" the same.
            if (k == 1) {
              // In the case RANGE='I' and with not much initial
              // accuracy in LAMBDA and VL, the formula
              // LGAP = MAX( ZERO, (SIGMA - VL) + LAMBDA )
              // can lead to an overestimation of the left gap and
              // thus to inadequately early RQI 'convergence'.
              // Prevent this by forcing a small left gap.
              lgap = eps * Max(std::abs(left), std::abs(right));
            } else {
              lgap = WGAP(windmn);
            }
            if (k == im) {
              // In the case RANGE='I' and with not much initial
              // accuracy in LAMBDA and VU, the formula
              // can lead to an overestimation of the right gap and
              // thus to inadequately early RQI 'convergence'.
              // Prevent this by forcing a small right gap.
              rgap = eps * Max(std::abs(left), std::abs(right));
            } else {
              rgap = WGAP(windex);
            }
            gap = Min(lgap, rgap);
            if ((k == 1) || (k == im)) {
              // The eigenvector support can become wrong
              // because significant entries could be cut off due to a
              // large GAPTOL parameter in LAR1V. Prevent this.
              gaptol = zero;
            } else {
              gaptol = gap * eps;
            }
            isupmn = in;
            isupmx = 1;
            // Update WGAP so that it holds the minimum gap
            // to the left or the right. This is crucial in the
            // case where bisection is used to ensure that the
            // eigenvalue is refined up to the required precision.
            // The correct value is restored afterwards.
            savgap = WGAP(windex);
            WGAP(windex) = gap;
            // We want to use the Rayleigh Quotient Correction
            // as often as possible since it converges quadratically
            // when we are close enough to the desired eigenvalue.
            // However, the Rayleigh Quotient can have the wrong sign
            // and lead us away from the desired eigenvalue. In this
            // case, the best we can do is to use bisection.
            usedbs = false;
            usedrq = false;
            // Bisection is initially turned off unless it is forced
            needbs = !tryrqc;
          L120:
            // Check if bisection should be used to refine eigenvalue
            if (needbs) {
              // Take the bisection as new iterate
              usedbs = true;
              itmp1 = IWORK(iindr + windex);
              offset = INDEXW(wbegin) - 1;
              iinfo = Larrb(in, D.ptr(ibegin), WORK.ptr(indlld + ibegin - 1),
                            indeig, indeig, zero, two * eps, offset,
                            WORK.ptr(wbegin), WGAP.ptr(wbegin),
                            WERR.ptr(wbegin), WORK.ptr(indwrk),
                            IWORK.ptr(iindwk), pivmin, spdiam, itmp1);
              if (iinfo != 0) {
                info = -3;
                return info;
              }
              lambda = WORK(windex);
              // Reset twist index from inaccurate LAMBDA to
              // force computation of true MINGMA
              IWORK(iindr + windex) = 0;
            }
            // Given LAMBDA, compute the eigenvector.
            Lar1v(in, 1, in, lambda, D.ptr(ibegin), L.ptr(ibegin),
                  WORK.ptr(indld + ibegin - 1), WORK.ptr(indlld + ibegin - 1),
                  pivmin, gaptol, Z.ptr(ibegin, windex), !usedbs, &negcnt, &ztz,
                  &mingma, IWORK.ptr(iindr + windex),
                  ISUPPZ.ptr(2 * windex - 1), &nrminv, &resid, &rqcorr,
                  WORK.ptr(indwrk));
            // NOLINTNEXTLINE(bugprone-branch-clone): as LAPACK
            if (iter == 0) {
              bstres = resid;
              bstw = lambda;
            } else if (resid < bstres) {
              bstres = resid;
              bstw = lambda;
            }
            isupmn = Min<blasint>(isupmn, ISUPPZ(2 * windex - 1));
            isupmx = Max<blasint>(isupmx, ISUPPZ(2 * windex));
            iter = iter + 1;
            // sin alpha <= |resid|/gap
            // Note that both the residual and the gap are
            // proportional to the matrix, so ||T|| doesn't play
            // a role in the quotient
            // Convergence test for Rayleigh-Quotient iteration
            // (omitted when Bisection has been used)
            if (resid > tol * gap &&
                std::abs(rqcorr) > rqtol * std::abs(lambda) && !usedbs) {
              // We need to check that the RQCORR update doesn't
              // move the eigenvalue away from the desired one and
              // towards a neighbor. -> protection with bisection
              if (indeig <= negcnt) {
                // The wanted eigenvalue lies to the left
                sgndef = -one;
              } else {
                // The wanted eigenvalue lies to the right
                sgndef = one;
              }
              // We only use the RQCORR if it improves the
              // the iterate reasonably.
              if ((rqcorr * sgndef >= zero) && (lambda + rqcorr <= right) &&
                  (lambda + rqcorr >= left)) {
                usedrq = true;
                // Store new midpoint of bisection interval in WORK
                if (sgndef == one) {
                  // The current LAMBDA is on the left of the true
                  // eigenvalue
                  left = lambda;
                  // We prefer to assume that the error estimate
                  // is correct. We could make the interval not
                  // as a bracket but to be modified if the RQCORR
                  // chooses to. In this case, the RIGHT side should
                  // be modified as follows:
                  // RIGHT = MAX(RIGHT, LAMBDA + RQCORR)
                } else {
                  // The current LAMBDA is on the right of the true
                  // eigenvalue
                  right = lambda;
                  // See comment about assuming the error estimate is
                  // correct above.
                  // LEFT = MIN(LEFT, LAMBDA + RQCORR)
                }
                WORK(windex) = half * (right + left);
                // Take RQCORR since it has the correct sign and
                // improves the iterate reasonably
                lambda = lambda + rqcorr;
                // Update width of error interval
                WERR(windex) = half * (right - left);
              } else {
                needbs = true;
              }
              if (right - left < rqtol * std::abs(lambda)) {
                // The eigenvalue is computed to bisection accuracy
                // compute eigenvector and stop
                usedbs = true;
                goto L120;
              } else if (iter < maxitr) {
                goto L120;
              } else if (iter == maxitr) {
                needbs = true;
                goto L120;
              } else {
                info = 5;
                return info;
              }
            } else {
              stp2ii = false;
              if (usedrq && usedbs && bstres <= resid) {
                lambda = bstw;
                stp2ii = true;
              }
              if (stp2ii) {
                // improve error angle by second step
                Lar1v(in, 1, in, lambda, D.ptr(ibegin), L.ptr(ibegin),
                      WORK.ptr(indld + ibegin - 1),
                      WORK.ptr(indlld + ibegin - 1), pivmin, gaptol,
                      Z.ptr(ibegin, windex), !usedbs, &negcnt, &ztz, &mingma,
                      IWORK.ptr(iindr + windex), ISUPPZ.ptr(2 * windex - 1),
                      &nrminv, &resid, &rqcorr, WORK.ptr(indwrk));
              }
              WORK(windex) = lambda;
            }
            // Compute FP-vector support w.r.t. whole matrix
            ISUPPZ(2 * windex - 1) = ISUPPZ(2 * windex - 1) + oldien;
            ISUPPZ(2 * windex) = ISUPPZ(2 * windex) + oldien;
            zfrom = ISUPPZ(2 * windex - 1);
            zto = ISUPPZ(2 * windex);
            isupmn = isupmn + oldien;
            isupmx = isupmx + oldien;
            // Ensure vector is ok if support in the RQI has changed
            if (isupmn < zfrom) {
              for (ii = isupmn; ii <= zfrom - 1; ++ii) {
                Z(ii, windex) = T(zero);
              }
            }
            if (isupmx > zto) {
              for (ii = zto + 1; ii <= isupmx; ++ii) {
                Z(ii, windex) = T(zero);
              }
            }
            Rscal(zto - zfrom + 1, nrminv, Z.ptr(zfrom, windex), 1);
          L125:
            // Update W
            W(windex) = lambda + sigma;
            // Recompute the gaps on the left and right
            // But only allow them to become larger and not
            // smaller (which can only happen through "bad"
            // cancellation and doesn't reflect the theory
            // where the initial gaps are underestimated due
            // to WERR being too crude.)
            if (!eskip) {
              if (k > 1) {
                WGAP(windmn) = Max(WGAP(windmn), W(windex) - WERR(windex) -
                                                     W(windmn) - WERR(windmn));
              }
              if (windex < wend) {
                WGAP(windex) = Max(savgap, W(windpl) - WERR(windpl) -
                                               W(windex) - WERR(windex));
              }
            }
            idone = idone + 1;
          }
          // here ends the code for the current child
        L139:
          // Proceed to any remaining child nodes
          newfst = j + 1;
        L140:;
        }
      }
      ndepth = ndepth + 1;
      goto L40;
    }
    ibegin = iend + 1;
    wbegin = wend + 1;
  L170:;
  }
  return info;
}

// xSTEMR: selected eigenvalues and, optionally (JOBZ = 'V'), eigenvectors
// of the symmetric tridiagonal matrix with diagonal D and off-diagonal E
// (N; both destroyed) by MRRR: all (RANGE = 'A'), those in (VL, VU] ('V')
// or the IL-th through IU-th ('I'). Returns M eigenvalues W in ascending
// order, the eigenvectors in Z (NZC columns available; NZC = -1 is a query
// for the number needed, returned in Z(1, 1)) and their supports ISUPPZ.
// TRYRAC asks for high relative accuracy if the matrix defines its
// eigenvalues so (xLARRR); it is set to false otherwise. LWORK >= 18*N
// (12*N for JOBZ = 'N'), LIWORK >= 10*N (8*N); -1 is a workspace query.
// Returns INFO: 0, -i for an illegal argument i, or > 0 (1 or 2: internal
// error; 1x: xLARRE failed; 2x: xLARRV failed). For complex T, Z is
// complex and the eigenvectors real. As in LAPACK, single precision uses
// the minimal relative gap MINRGP = 3E-3 and RTOL1 = max(sqrt(EPS)*5E-2,
// 4*EPS) (double: 1E-3 and sqrt(EPS)). For N = 2 and E(1) = 0, ISUPPZ is
// (2, 2) for both eigenvectors (as LAPACK, also for the eigenvector
// (1, 0)).
template <typename T>
blasint Stemr(char jobz, char range, blasint n, RealT<T>* d, RealT<T>* e,
              RealT<T> vl, RealT<T> vu, blasint il, blasint iu, blasint* m,
              RealT<T>* w, T* z, blasint ldz, blasint nzc, blasint* isuppz,
              bool* tryrac, RealT<T>* work, blasint lwork, blasint* iwork,
              blasint liwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R four = R(4);
  // SSTEMR and CSTEMR: 3.0E-3, DSTEMR and ZSTEMR: 1.0D-3.
  const R minrgp = std::is_same_v<R, float> ? R(3.0e-3) : R(1.0e-3);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> ISUPPZ(isuppz);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint i, ibegin, iend, ifirst, iil, iindbl, iindw, iindwk, iinfo, iinspl,
      iiu, ilast, in, indd, inde2, inderr, indgp, indgrs, indwrk, itmp, itmp2,
      j, jblk, jj, liwmin, lwmin, nsplit, nzcmin, offset, wbegin, wend;
  bool alleig, indeig, laeswap, lquery, valeig, wantz, zquery;
  R bignum, cs, eps, pivmin, rmax, rmin, rtol1, rtol2, safmin, scale, smlnum,
      sn, thresh, tmp, tnrm, wl, wu;
  // Set by xLAE2 or xLAEV2 for N = 2 (a query returns before).
  R r1 = R(0), r2 = R(0);
  wantz = Lsame(jobz, 'V');
  alleig = Lsame(range, 'A');
  valeig = Lsame(range, 'V');
  indeig = Lsame(range, 'I');
  lquery = ((lwork == -1) || (liwork == -1));
  zquery = (nzc == -1);
  laeswap = false;
  // DSTEMR needs WORK of size 6*N, IWORK of size 3*N.
  // In addition, DLARRE needs WORK of size 6*N, IWORK of size 5*N.
  // Furthermore, ZLARRV needs WORK of size 12*N, IWORK of size 7*N.
  if (wantz) {
    lwmin = 18 * n;
    liwmin = 10 * n;
  } else {
    // need less workspace if only the eigenvalues are wanted
    lwmin = 12 * n;
    liwmin = 8 * n;
  }
  wl = zero;
  wu = zero;
  iil = 0;
  iiu = 0;
  nsplit = 0;
  if (valeig) {
    // We do not reference VL, VU in the cases RANGE = 'I','A'
    // The interval (WL, WU] contains all the wanted eigenvalues.
    // It is either given by the user or computed in DLARRE.
    wl = vl;
    wu = vu;
  } else if (indeig) {
    // We do not reference IL, IU in the cases RANGE = 'V','A'
    iil = il;
    iiu = iu;
  }
  blasint info = 0;
  if (!(wantz || Lsame(jobz, 'N'))) {
    info = -1;
  } else if (!(alleig || valeig || indeig)) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (valeig && n > 0 && wu <= wl) {
    info = -7;
  } else if (indeig && (iil < 1 || iil > n)) {
    info = -8;
  } else if (indeig && (iiu < iil || iiu > n)) {
    info = -9;
  } else if (ldz < 1 || (wantz && ldz < n)) {
    info = -13;
  } else if (lwork < lwmin && !lquery) {
    info = -17;
  } else if (liwork < liwmin && !lquery) {
    info = -19;
  }
  // Get machine constants.
  safmin = Lamch<R>('S');
  eps = Lamch<R>('P');
  smlnum = safmin / eps;
  bignum = one / smlnum;
  rmin = std::sqrt(smlnum);
  rmax = Min(std::sqrt(bignum), one / std::sqrt(std::sqrt(safmin)));
  if (info == 0) {
    WORK(1) = WorkSize<R>(lwmin);
    IWORK(1) = liwmin;
    if (wantz && alleig) {
      nzcmin = n;
    } else if (wantz && valeig) {
      info = Larrc('T', n, vl, vu, d, e, safmin, &nzcmin, &itmp, &itmp2);
    } else if (wantz && indeig) {
      nzcmin = iiu - iil + 1;
    } else {
      // WANTZ .EQ. FALSE.
      nzcmin = 0;
    }
    if (zquery && info == 0) {
      Z(1, 1) = T(R(nzcmin));
    } else if (nzc < nzcmin && !zquery) {
      info = -14;
    }
  }
  if (info != 0) {
    Xerbla<T>("STEMR", -info);
    return info;
  } else if (lquery || zquery) {
    return info;
  }
  // Handle N = 0, 1, and 2 cases immediately
  *m = 0;
  if (n == 0) return info;
  if (n == 1) {
    if (alleig || indeig) {
      *m = 1;
      W(1) = D(1);
    } else {
      if (wl < D(1) && wu >= D(1)) {
        *m = 1;
        W(1) = D(1);
      }
    }
    if (wantz && (!zquery)) {
      Z(1, 1) = T(one);
      ISUPPZ(1) = 1;
      ISUPPZ(2) = 1;
    }
    return info;
  }
  if (n == 2) {
    if (!wantz) {
      Lae2(D(1), E(1), D(2), &r1, &r2);
    } else if (wantz && (!zquery)) {
      Laev2(D(1), E(1), D(2), &r1, &r2, &cs, &sn);
    }
    // D/S/LAE2 and D/S/LAEV2 outputs satisfy |R1| >= |R2|. However,
    // the following code requires R1 >= R2. Hence, we correct
    // the order of R1, R2, CS, SN if R1 < R2 before further processing.
    if (r1 < r2) {
      E(2) = r1;
      r1 = r2;
      r2 = E(2);
      laeswap = true;
    }
    if (alleig || (valeig && (r2 > wl) && (r2 <= wu)) ||
        (indeig && (iil == 1))) {
      *m = *m + 1;
      W(*m) = r2;
      if (wantz && (!zquery)) {
        if (laeswap) {
          Z(1, *m) = T(cs);
          Z(2, *m) = T(sn);
        } else {
          Z(1, *m) = T(-sn);
          Z(2, *m) = T(cs);
        }
        // Note: At most one of SN and CS can be zero.
        if (sn != zero) {
          if (cs != zero) {
            ISUPPZ(2 * *m - 1) = 1;
            ISUPPZ(2 * *m) = 2;
          } else {
            ISUPPZ(2 * *m - 1) = 1;
            ISUPPZ(2 * *m) = 1;
          }
        } else {
          ISUPPZ(2 * *m - 1) = 2;
          ISUPPZ(2 * *m) = 2;
        }
      }
    }
    if (alleig || (valeig && (r1 > wl) && (r1 <= wu)) ||
        (indeig && (iiu == 2))) {
      *m = *m + 1;
      W(*m) = r1;
      if (wantz && (!zquery)) {
        if (laeswap) {
          Z(1, *m) = T(-sn);
          Z(2, *m) = T(cs);
        } else {
          Z(1, *m) = T(cs);
          Z(2, *m) = T(sn);
        }
        // Note: At most one of SN and CS can be zero.
        if (sn != zero) {
          if (cs != zero) {
            ISUPPZ(2 * *m - 1) = 1;
            ISUPPZ(2 * *m) = 2;
          } else {
            ISUPPZ(2 * *m - 1) = 1;
            ISUPPZ(2 * *m) = 1;
          }
        } else {
          ISUPPZ(2 * *m - 1) = 2;
          ISUPPZ(2 * *m) = 2;
        }
      }
    }
  } else {
    // Continue with general N
    indgrs = 1;
    inderr = 2 * n + 1;
    indgp = 3 * n + 1;
    indd = 4 * n + 1;
    inde2 = 5 * n + 1;
    indwrk = 6 * n + 1;
    iinspl = 1;
    iindbl = n + 1;
    iindw = 2 * n + 1;
    iindwk = 3 * n + 1;
    // Scale matrix to allowable range, if necessary.
    // The allowable range is related to the PIVMIN parameter; see the
    // comments in DLARRD.  The preference for scaling small values
    // up is heuristic; we expect users' matrices not to be close to the
    // RMAX threshold.
    scale = one;
    tnrm = Lanst('M', n, d, e);
    if (tnrm > zero && tnrm < rmin) {
      scale = rmin / tnrm;
    } else if (tnrm > rmax) {
      scale = rmax / tnrm;
    }
    if (scale != one) {
      Scal(n, scale, d, 1);
      Scal(n - 1, scale, e, 1);
      tnrm = tnrm * scale;
      if (valeig) {
        // If eigenvalues in interval have to be found,
        // scale (WL, WU] accordingly
        wl = wl * scale;
        wu = wu * scale;
      }
    }
    // Compute the desired eigenvalues of the tridiagonal after splitting
    // into smaller subblocks if the corresponding off-diagonal elements
    // are small
    // THRESH is the splitting parameter for DLARRE
    // A negative THRESH forces the old splitting criterion based on the
    // size of the off-diagonal. A positive THRESH switches to splitting
    // which preserves relative accuracy.
    if (*tryrac) {
      // Test whether the matrix warrants the more expensive relative approach.
      iinfo = Larrr(n, d, e);
    } else {
      // The user does not care about relative accurately eigenvalues
      iinfo = -1;
    }
    // Set the splitting criterion
    if (iinfo == 0) {
      thresh = eps;
    } else {
      thresh = -eps;
      // relative accuracy is desired but T does not guarantee it
      *tryrac = false;
    }
    if (*tryrac) {
      // Copy original diagonal, needed to guarantee relative accuracy
      Copy(n, d, 1, WORK.ptr(indd), 1);
    }
    // Store the squares of the offdiagonal values of T
    for (j = 1; j <= n - 1; ++j) {
      WORK(inde2 + j - 1) = (E(j) * E(j));
    }
    // Set the tolerance parameters for bisection
    if (!wantz) {
      // DLARRE computes the eigenvalues to full precision.
      rtol1 = four * eps;
      rtol2 = four * eps;
    } else {
      // DLARRE computes the eigenvalues to less than full precision.
      // ZLARRV will refine the eigenvalue approximations, and we only
      // need less accurate initial bisection in DLARRE.
      // Note: these settings do only affect the subset case and DLARRE
      if constexpr (std::is_same_v<R, float>) {
        rtol1 = Max(std::sqrt(eps) * R(5.0e-2), four * eps);
      } else {
        rtol1 = std::sqrt(eps);
      }
      rtol2 = Max(std::sqrt(eps) * R(5.0e-3), four * eps);
    }
    iinfo =
        Larre(range, n, &wl, &wu, iil, iiu, d, e, WORK.ptr(inde2), rtol1, rtol2,
              thresh, &nsplit, IWORK.ptr(iinspl), m, w, WORK.ptr(inderr),
              WORK.ptr(indgp), IWORK.ptr(iindbl), IWORK.ptr(iindw),
              WORK.ptr(indgrs), &pivmin, WORK.ptr(indwrk), IWORK.ptr(iindwk));
    if (iinfo != 0) {
      info = 10 + std::abs(iinfo);
      return info;
    }
    // Note that if RANGE .NE. 'V', DLARRE computes bounds on the desired
    // part of the spectrum. All desired eigenvalues are contained in
    // (WL,WU]
    if (wantz) {
      // Compute the desired eigenvectors corresponding to the computed
      // eigenvalues
      iinfo =
          Larrv(n, wl, wu, d, e, pivmin, IWORK.ptr(iinspl), *m, 1, *m, minrgp,
                &rtol1, &rtol2, w, WORK.ptr(inderr), WORK.ptr(indgp),
                IWORK.ptr(iindbl), IWORK.ptr(iindw), WORK.ptr(indgrs), z, ldz,
                isuppz, WORK.ptr(indwrk), IWORK.ptr(iindwk));
      if (iinfo != 0) {
        info = 20 + std::abs(iinfo);
        return info;
      }
    } else {
      // DLARRE computes eigenvalues of the (shifted) root representation
      // ZLARRV returns the eigenvalues of the unshifted matrix.
      // However, if the eigenvectors are not desired by the user, we need
      // to apply the corresponding shifts from DLARRE to obtain the
      // eigenvalues of the original matrix.
      for (j = 1; j <= *m; ++j) {
        itmp = IWORK(iindbl + j - 1);
        W(j) = W(j) + E(IWORK(iinspl + itmp - 1));
      }
    }
    if (*tryrac) {
      // Refine computed eigenvalues so that they are relatively accurate
      // with respect to the original matrix T.
      ibegin = 1;
      wbegin = 1;
      for (jblk = 1; jblk <= IWORK(iindbl + *m - 1); ++jblk) {
        iend = IWORK(iinspl + jblk - 1);
        in = iend - ibegin + 1;
        wend = wbegin - 1;
        // check if any eigenvalues have to be refined in this block
      L36:
        if (wend < *m) {
          if (IWORK(iindbl + wend) == jblk) {
            wend = wend + 1;
            goto L36;
          }
        }
        if (wend < wbegin) {
          ibegin = iend + 1;
          goto L39;
        }
        offset = IWORK(iindw + wbegin - 1) - 1;
        ifirst = IWORK(iindw + wbegin - 1);
        ilast = IWORK(iindw + wend - 1);
        rtol2 = four * eps;
        iinfo = Larrj(in, WORK.ptr(indd + ibegin - 1),
                      WORK.ptr(inde2 + ibegin - 1), ifirst, ilast, rtol2,
                      offset, W.ptr(wbegin), WORK.ptr(inderr + wbegin - 1),
                      WORK.ptr(indwrk), IWORK.ptr(iindwk), pivmin, tnrm);
        ibegin = iend + 1;
        wbegin = wend + 1;
      L39:;
      }
    }
    // If matrix was scaled, then rescale eigenvalues appropriately.
    if (scale != one) {
      Scal(*m, one / scale, w, 1);
    }
  }
  // If eigenvalues are not in increasing order, then sort them,
  // possibly along with eigenvectors.
  if (nsplit > 1 || n == 2) {
    if (!wantz) {
      iinfo = Lasrt('I', *m, w);
      if (iinfo != 0) {
        info = 3;
        return info;
      }
    } else {
      for (j = 1; j <= *m - 1; ++j) {
        i = 0;
        tmp = W(j);
        for (jj = j + 1; jj <= *m; ++jj) {
          if (W(jj) < tmp) {
            i = jj;
            tmp = W(jj);
          }
        }
        if (i != 0) {
          W(i) = W(j);
          W(j) = tmp;
          if (wantz) {
            Swap(n, Z.ptr(1, i), 1, Z.ptr(1, j), 1);
            itmp = ISUPPZ(2 * i - 1);
            ISUPPZ(2 * i - 1) = ISUPPZ(2 * j - 1);
            ISUPPZ(2 * j - 1) = itmp;
            itmp = ISUPPZ(2 * i);
            ISUPPZ(2 * i) = ISUPPZ(2 * j);
            ISUPPZ(2 * j) = itmp;
          }
        }
      }
    }
  }
  WORK(1) = WorkSize<R>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template blasint Laneg<R>(blasint, R*, R*, R, R, blasint);                   \
  template blasint Larra<R>(blasint, R*, R*, R*, R, R, blasint*, blasint*);    \
  template blasint Larrb<R>(blasint, R*, R*, blasint, blasint, R, R, blasint,  \
                            R*, R*, R*, R*, blasint*, R, R, blasint);          \
  template blasint Larrc<R>(char, blasint, R, R, R*, R*, R, blasint*,          \
                            blasint*, blasint*);                               \
  template blasint Larrd<R>(char, char, blasint, R, R, blasint, blasint, R*,   \
                            R, R*, R*, R*, R, blasint, blasint*, blasint*, R*, \
                            R*, R*, R*, blasint*, blasint*, R*, blasint*);     \
  template blasint Larre<R>(char, blasint, R*, R*, blasint, blasint, R*, R*,   \
                            R*, R, R, R, blasint*, blasint*, blasint*, R*, R*, \
                            R*, blasint*, blasint*, R*, R*, R*, blasint*);     \
  template blasint Larrf<R>(blasint, R*, R*, R*, blasint, blasint, R*, R*, R*, \
                            R, R, R, R, R*, R*, R*, R*);                       \
  template blasint Larrj<R>(blasint, R*, R*, blasint, blasint, R, blasint, R*, \
                            R*, R*, blasint*, R, R);                           \
  template blasint Larrk<R>(blasint, blasint, R, R, R*, R*, R, R, R*, R*);     \
  template blasint Larrr<R>(blasint, R*, R*);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template void Lar1v<T>(                                                      \
      blasint, blasint, blasint, RealT<T>, RealT<T>*, RealT<T>*, RealT<T>*,    \
      RealT<T>*, RealT<T>, RealT<T>, T*, bool, blasint*, RealT<T>*, RealT<T>*, \
      blasint*, blasint*, RealT<T>*, RealT<T>*, RealT<T>*, RealT<T>*);         \
  template blasint Larrv<T>(                                                   \
      blasint, RealT<T>, RealT<T>, RealT<T>*, RealT<T>*, RealT<T>, blasint*,   \
      blasint, blasint, blasint, RealT<T>, RealT<T>*, RealT<T>*, RealT<T>*,    \
      RealT<T>*, RealT<T>*, blasint*, blasint*, RealT<T>*, T*, blasint,        \
      blasint*, RealT<T>*, blasint*);                                          \
  template blasint Stemr<T>(char, char, blasint, RealT<T>*, RealT<T>*,         \
                            RealT<T>, RealT<T>, blasint, blasint, blasint*,    \
                            RealT<T>*, T*, blasint, blasint, blasint*, bool*,  \
                            RealT<T>*, blasint, blasint*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_STEMR_REAL(p, R)                                        \
  HWYBLAS_API hwyblas_int p##laneg_(const hwyblas_int* n, R* d, R* lld,        \
                                    const R* sigma, const R* pivmin,           \
                                    const hwyblas_int* r) {                    \
    return hwyblas::lapack::Laneg(*n, d, lld, *sigma, *pivmin, *r);            \
  }                                                                            \
  HWYBLAS_API void p##larra_(                                                  \
      const hwyblas_int* n, R* d, R* e, R* e2, const R* spltol, const R* tnrm, \
      hwyblas_int* nsplit, hwyblas_int* isplit, hwyblas_int* info) {           \
    *info =                                                                    \
        hwyblas::lapack::Larra(*n, d, e, e2, *spltol, *tnrm, nsplit, isplit);  \
  }                                                                            \
  HWYBLAS_API void p##larrb_(                                                  \
      const hwyblas_int* n, R* d, R* lld, const hwyblas_int* ifirst,           \
      const hwyblas_int* ilast, const R* rtol1, const R* rtol2,                \
      const hwyblas_int* offset, R* w, R* wgap, R* werr, R* work,              \
      hwyblas_int* iwork, const R* pivmin, const R* spdiam,                    \
      const hwyblas_int* twist, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Larrb(*n, d, lld, *ifirst, *ilast, *rtol1,        \
                                   *rtol2, *offset, w, wgap, werr, work,       \
                                   iwork, *pivmin, *spdiam, *twist);           \
  }                                                                            \
  HWYBLAS_API void p##larrc_(                                                  \
      const char* jobt, const hwyblas_int* n, const R* vl, const R* vu, R* d,  \
      R* e, const R* pivmin, hwyblas_int* eigcnt, hwyblas_int* lcnt,           \
      hwyblas_int* rcnt, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Larrc(*jobt, *n, *vl, *vu, d, e, *pivmin, eigcnt, \
                                   lcnt, rcnt);                                \
  }                                                                            \
  HWYBLAS_API void p##larrd_(                                                  \
      const char* range, const char* order, const hwyblas_int* n, const R* vl, \
      const R* vu, const hwyblas_int* il, const hwyblas_int* iu, R* gers,      \
      const R* reltol, R* d, R* e, R* e2, const R* pivmin,                     \
      const hwyblas_int* nsplit, hwyblas_int* isplit, hwyblas_int* m, R* w,    \
      R* werr, R* wl, R* wu, hwyblas_int* iblock, hwyblas_int* indexw,         \
      R* work, hwyblas_int* iwork, hwyblas_int* info) {                        \
    *info =                                                                    \
        hwyblas::lapack::Larrd(*range, *order, *n, *vl, *vu, *il, *iu, gers,   \
                               *reltol, d, e, e2, *pivmin, *nsplit, isplit, m, \
                               w, werr, wl, wu, iblock, indexw, work, iwork);  \
  }                                                                            \
  HWYBLAS_API void p##larre_(                                                  \
      const char* range, const hwyblas_int* n, R* vl, R* vu,                   \
      const hwyblas_int* il, const hwyblas_int* iu, R* d, R* e, R* e2,         \
      const R* rtol1, const R* rtol2, const R* spltol, hwyblas_int* nsplit,    \
      hwyblas_int* isplit, hwyblas_int* m, R* w, R* werr, R* wgap,             \
      hwyblas_int* iblock, hwyblas_int* indexw, R* gers, R* pivmin, R* work,   \
      hwyblas_int* iwork, hwyblas_int* info) {                                 \
    *info = hwyblas::lapack::Larre(*range, *n, vl, vu, *il, *iu, d, e, e2,     \
                                   *rtol1, *rtol2, *spltol, nsplit, isplit, m, \
                                   w, werr, wgap, iblock, indexw, gers,        \
                                   pivmin, work, iwork);                       \
  }                                                                            \
  HWYBLAS_API void p##larrf_(                                                  \
      const hwyblas_int* n, R* d, R* l, R* ld, const hwyblas_int* clstrt,      \
      const hwyblas_int* clend, R* w, R* wgap, R* werr, const R* spdiam,       \
      const R* clgapl, const R* clgapr, const R* pivmin, R* sigma, R* dplus,   \
      R* lplus, R* work, hwyblas_int* info) {                                  \
    *info = hwyblas::lapack::Larrf(*n, d, l, ld, *clstrt, *clend, w, wgap,     \
                                   werr, *spdiam, *clgapl, *clgapr, *pivmin,   \
                                   sigma, dplus, lplus, work);                 \
  }                                                                            \
  HWYBLAS_API void p##larrj_(                                                  \
      const hwyblas_int* n, R* d, R* e2, const hwyblas_int* ifirst,            \
      const hwyblas_int* ilast, const R* rtol, const hwyblas_int* offset,      \
      R* w, R* werr, R* work, hwyblas_int* iwork, const R* pivmin,             \
      const R* spdiam, hwyblas_int* info) {                                    \
    *info = hwyblas::lapack::Larrj(*n, d, e2, *ifirst, *ilast, *rtol, *offset, \
                                   w, werr, work, iwork, *pivmin, *spdiam);    \
  }                                                                            \
  HWYBLAS_API void p##larrk_(const hwyblas_int* n, const hwyblas_int* iw,      \
                             const R* gl, const R* gu, R* d, R* e2,            \
                             const R* pivmin, const R* reltol, R* w, R* werr,  \
                             hwyblas_int* info) {                              \
    *info = hwyblas::lapack::Larrk(*n, *iw, *gl, *gu, d, e2, *pivmin, *reltol, \
                                   w, werr);                                   \
  }                                                                            \
  HWYBLAS_API void p##larrr_(const hwyblas_int* n, R* d, R* e,                 \
                             hwyblas_int* info) {                              \
    *info = hwyblas::lapack::Larrr(*n, d, e);                                  \
  }

HWYBLAS_LAPACK_STEMR_REAL(s, float)
HWYBLAS_LAPACK_STEMR_REAL(d, double)

#undef HWYBLAS_LAPACK_STEMR_REAL

#define HWYBLAS_LAPACK_STEMR(p, T, R)                                          \
  HWYBLAS_API void p##lar1v_(                                                  \
      const hwyblas_int* n, const hwyblas_int* b1, const hwyblas_int* bn,      \
      const R* lambda, R* d, R* l, R* ld, R* lld, const R* pivmin,             \
      const R* gaptol, R* z, const hwyblas_int* wantnc, hwyblas_int* negcnt,   \
      R* ztz, R* mingma, hwyblas_int* r, hwyblas_int* isuppz, R* nrminv,       \
      R* resid, R* rqcorr, R* work) {                                          \
    hwyblas::lapack::Lar1v(*n, *b1, *bn, *lambda, d, l, ld, lld, *pivmin,      \
                           *gaptol, As<T>(z), *wantnc != 0, negcnt, ztz,       \
                           mingma, r, isuppz, nrminv, resid, rqcorr, work);    \
  }                                                                            \
  HWYBLAS_API void p##larrv_(                                                  \
      const hwyblas_int* n, const R* vl, const R* vu, R* d, R* l,              \
      const R* pivmin, hwyblas_int* isplit, const hwyblas_int* m,              \
      const hwyblas_int* dol, const hwyblas_int* dou, const R* minrgp,         \
      R* rtol1, R* rtol2, R* w, R* werr, R* wgap, hwyblas_int* iblock,         \
      hwyblas_int* indexw, R* gers, R* z, const hwyblas_int* ldz,              \
      hwyblas_int* isuppz, R* work, hwyblas_int* iwork, hwyblas_int* info) {   \
    *info = hwyblas::lapack::Larrv(*n, *vl, *vu, d, l, *pivmin, isplit, *m,    \
                                   *dol, *dou, *minrgp, rtol1, rtol2, w, werr, \
                                   wgap, iblock, indexw, gers, As<T>(z), *ldz, \
                                   isuppz, work, iwork);                       \
  }                                                                            \
  HWYBLAS_API void p##stemr_(                                                  \
      const char* jobz, const char* range, const hwyblas_int* n, R* d, R* e,   \
      const R* vl, const R* vu, const hwyblas_int* il, const hwyblas_int* iu,  \
      hwyblas_int* m, R* w, R* z, const hwyblas_int* ldz,                      \
      const hwyblas_int* nzc, hwyblas_int* isuppz, hwyblas_int* tryrac,        \
      R* work, const hwyblas_int* lwork, hwyblas_int* iwork,                   \
      const hwyblas_int* liwork, hwyblas_int* info) {                          \
    bool rac = *tryrac != 0;                                                   \
    *info = hwyblas::lapack::Stemr(*jobz, *range, *n, d, e, *vl, *vu, *il,     \
                                   *iu, m, w, As<T>(z), *ldz, *nzc, isuppz,    \
                                   &rac, work, *lwork, iwork, *liwork);        \
    *tryrac = rac ? 1 : 0;                                                     \
  }

HWYBLAS_LAPACK_STEMR(s, float, float)
HWYBLAS_LAPACK_STEMR(d, double, double)
HWYBLAS_LAPACK_STEMR(c, c32, float)
HWYBLAS_LAPACK_STEMR(z, c64, double)

#undef HWYBLAS_LAPACK_STEMR

}  // extern "C"
