// Eigenvalues of a symmetric tridiagonal matrix by bisection and
// eigenvectors by inverse iteration: xSTEBZ (bisection with Sturm counts,
// done by xLAEBZ), xSTEIN (inverse iteration, with xLAGTF / xLAGTS for the
// shifted tridiagonal systems), and the random number generators xLARUV /
// xLARNV that xSTEIN and xLARRE use for their starting vectors. xSTEBZ,
// xLAEBZ, xLAGTF, xLAGTS and xLARUV are real (R); xSTEIN and xLARNV serve
// all four types (for complex T, xSTEIN computes the real eigenvectors of
// the real tridiagonal matrix into the complex Z).
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

// xLAEBZ: the bisection kernel of xSTEBZ and xLARRD for the symmetric
// tridiagonal matrix with diagonal D, off-diagonal E and squares E2 (N):
// IJOB = 1 counts the eigenvalues below the ends of the MINP intervals
// [AB(j, 1), AB(j, 2)] into NAB (MOUT = their number); IJOB = 2 bisects
// the intervals until they are shorter than max(ABSTOL, PIVMIN,
// RELTOL*max|end|), splitting those with eigenvalues in both halves (at
// most MMAX intervals; MOUT = their number); IJOB = 3 bisects them to the
// point where the count is NVAL(j). NBMIN > 0 is the number of intervals
// from which the counts of all intervals are computed in one vectorizable
// loop (WORK, IWORK), instead of one interval at a time. AB and NAB have
// leading dimension MMAX. Returns INFO: 0, -1 for an illegal IJOB, MMAX+1
// if more than MMAX intervals were needed, or the number of intervals that
// did not converge in NITMAX iterations.
template <typename R>
blasint Laebz(blasint ijob, blasint nitmax, blasint n, blasint mmax,
              blasint minp, blasint nbmin, R abstol, R reltol, R pivmin, R* d,
              R* e, R* e2, blasint* nval, R* ab, R* c, blasint* mout,
              blasint* nab, R* work, blasint* iwork) {
  const R zero = R(0);
  const R two = R(2);
  const R half = R(1) / two;
  const Matrix<R> AB(ab, mmax);
  const Vector<R> C(c);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<R> E2(e2);
  const Vector<blasint> IWORK(iwork);
  const Matrix<blasint> NAB(nab, mmax);
  const Vector<blasint> NVAL(nval);
  const Vector<R> WORK(work);
  blasint itmp1, itmp2, j, ji, jit, jp, kf, kfnew, kl, klnew;
  R tmp1, tmp2;
  blasint info = 0;
  if (ijob < 1 || ijob > 3) {
    info = -1;
    return info;
  }
  // Initialize NAB
  if (ijob == 1) {
    // Compute the number of eigenvalues in the initial intervals.
    *mout = 0;
    for (ji = 1; ji <= minp; ++ji) {
      for (jp = 1; jp <= 2; ++jp) {
        tmp1 = D(1) - AB(ji, jp);
        if (std::abs(tmp1) < pivmin) tmp1 = -pivmin;
        NAB(ji, jp) = 0;
        if (tmp1 <= zero) NAB(ji, jp) = 1;
        for (j = 2; j <= n; ++j) {
          tmp1 = D(j) - E2(j - 1) / tmp1 - AB(ji, jp);
          if (std::abs(tmp1) < pivmin) tmp1 = -pivmin;
          if (tmp1 <= zero) NAB(ji, jp) = NAB(ji, jp) + 1;
        }
      }
      *mout = *mout + NAB(ji, 2) - NAB(ji, 1);
    }
    return info;
  }
  // Initialize for loop
  // KF and KL have the following meaning:
  // Intervals 1,...,KF-1 have converged.
  // Intervals KF,...,KL  still need to be refined.
  kf = 1;
  kl = minp;
  // If IJOB=2, initialize C.
  // If IJOB=3, use the user-supplied starting point.
  if (ijob == 2) {
    for (ji = 1; ji <= minp; ++ji) {
      C(ji) = half * (AB(ji, 1) + AB(ji, 2));
    }
  }
  // Iteration loop
  for (jit = 1; jit <= nitmax; ++jit) {
    // Loop over intervals
    if (kl - kf + 1 >= nbmin && nbmin > 0) {
      // Begin of Parallel Version of the loop
      for (ji = kf; ji <= kl; ++ji) {
        // Compute N(c), the number of eigenvalues less than c
        WORK(ji) = D(1) - C(ji);
        IWORK(ji) = 0;
        if (WORK(ji) <= pivmin) {
          IWORK(ji) = 1;
          WORK(ji) = Min(WORK(ji), -pivmin);
        }
        for (j = 2; j <= n; ++j) {
          WORK(ji) = D(j) - E2(j - 1) / WORK(ji) - C(ji);
          if (WORK(ji) <= pivmin) {
            IWORK(ji) = IWORK(ji) + 1;
            WORK(ji) = Min(WORK(ji), -pivmin);
          }
        }
      }
      if (ijob <= 2) {
        // IJOB=2: Choose all intervals containing eigenvalues.
        klnew = kl;
        for (ji = kf; ji <= kl; ++ji) {
          // Insure that N(w) is monotone
          IWORK(ji) =
              Min<blasint>(NAB(ji, 2), Max<blasint>(NAB(ji, 1), IWORK(ji)));
          // Update the Queue -- add intervals if both halves
          // contain eigenvalues.
          if (IWORK(ji) == NAB(ji, 2)) {
            // No eigenvalue in the upper interval:
            // just use the lower interval.
            AB(ji, 2) = C(ji);
          } else if (IWORK(ji) == NAB(ji, 1)) {
            // No eigenvalue in the lower interval:
            // just use the upper interval.
            AB(ji, 1) = C(ji);
          } else {
            klnew = klnew + 1;
            if (klnew <= mmax) {
              // Eigenvalue in both intervals -- add upper to
              // queue.
              AB(klnew, 2) = AB(ji, 2);
              NAB(klnew, 2) = NAB(ji, 2);
              AB(klnew, 1) = C(ji);
              NAB(klnew, 1) = IWORK(ji);
              AB(ji, 2) = C(ji);
              NAB(ji, 2) = IWORK(ji);
            } else {
              info = mmax + 1;
            }
          }
        }
        if (info != 0) return info;
        kl = klnew;
      } else {
        // IJOB=3: Binary search.  Keep only the interval containing
        // w   s.t. N(w) = NVAL
        for (ji = kf; ji <= kl; ++ji) {
          if (IWORK(ji) <= NVAL(ji)) {
            AB(ji, 1) = C(ji);
            NAB(ji, 1) = IWORK(ji);
          }
          if (IWORK(ji) >= NVAL(ji)) {
            AB(ji, 2) = C(ji);
            NAB(ji, 2) = IWORK(ji);
          }
        }
      }
    } else {
      // End of Parallel Version of the loop
      // Begin of Serial Version of the loop
      klnew = kl;
      for (ji = kf; ji <= kl; ++ji) {
        // Compute N(w), the number of eigenvalues less than w
        tmp1 = C(ji);
        tmp2 = D(1) - tmp1;
        itmp1 = 0;
        if (tmp2 <= pivmin) {
          itmp1 = 1;
          tmp2 = Min(tmp2, -pivmin);
        }
        for (j = 2; j <= n; ++j) {
          tmp2 = D(j) - E2(j - 1) / tmp2 - tmp1;
          if (tmp2 <= pivmin) {
            itmp1 = itmp1 + 1;
            tmp2 = Min(tmp2, -pivmin);
          }
        }
        if (ijob <= 2) {
          // IJOB=2: Choose all intervals containing eigenvalues.
          // Insure that N(w) is monotone
          itmp1 = Min<blasint>(NAB(ji, 2), Max<blasint>(NAB(ji, 1), itmp1));
          // Update the Queue -- add intervals if both halves
          // contain eigenvalues.
          if (itmp1 == NAB(ji, 2)) {
            // No eigenvalue in the upper interval:
            // just use the lower interval.
            AB(ji, 2) = tmp1;
          } else if (itmp1 == NAB(ji, 1)) {
            // No eigenvalue in the lower interval:
            // just use the upper interval.
            AB(ji, 1) = tmp1;
          } else if (klnew < mmax) {
            // Eigenvalue in both intervals -- add upper to queue.
            klnew = klnew + 1;
            AB(klnew, 2) = AB(ji, 2);
            NAB(klnew, 2) = NAB(ji, 2);
            AB(klnew, 1) = tmp1;
            NAB(klnew, 1) = itmp1;
            AB(ji, 2) = tmp1;
            NAB(ji, 2) = itmp1;
          } else {
            info = mmax + 1;
            return info;
          }
        } else {
          // IJOB=3: Binary search.  Keep only the interval
          // containing  w  s.t. N(w) = NVAL
          if (itmp1 <= NVAL(ji)) {
            AB(ji, 1) = tmp1;
            NAB(ji, 1) = itmp1;
          }
          if (itmp1 >= NVAL(ji)) {
            AB(ji, 2) = tmp1;
            NAB(ji, 2) = itmp1;
          }
        }
      }
      kl = klnew;
    }
    // Check for convergence
    kfnew = kf;
    for (ji = kf; ji <= kl; ++ji) {
      tmp1 = std::abs(AB(ji, 2) - AB(ji, 1));
      tmp2 = Max(std::abs(AB(ji, 2)), std::abs(AB(ji, 1)));
      if (tmp1 < Max(abstol, pivmin, reltol * tmp2) ||
          NAB(ji, 1) >= NAB(ji, 2)) {
        // Converged -- Swap with position KFNEW,
        // then increment KFNEW
        if (ji > kfnew) {
          tmp1 = AB(ji, 1);
          tmp2 = AB(ji, 2);
          itmp1 = NAB(ji, 1);
          itmp2 = NAB(ji, 2);
          AB(ji, 1) = AB(kfnew, 1);
          AB(ji, 2) = AB(kfnew, 2);
          NAB(ji, 1) = NAB(kfnew, 1);
          NAB(ji, 2) = NAB(kfnew, 2);
          AB(kfnew, 1) = tmp1;
          AB(kfnew, 2) = tmp2;
          NAB(kfnew, 1) = itmp1;
          NAB(kfnew, 2) = itmp2;
          if (ijob == 3) {
            itmp1 = NVAL(ji);
            NVAL(ji) = NVAL(kfnew);
            NVAL(kfnew) = itmp1;
          }
        }
        kfnew = kfnew + 1;
      }
    }
    kf = kfnew;
    // Choose Midpoints
    for (ji = kf; ji <= kl; ++ji) {
      C(ji) = half * (AB(ji, 1) + AB(ji, 2));
    }
    // If no more intervals to refine, quit.
    if (kf > kl) goto L140;
  }
  // Converged
L140:
  info = Max<blasint>(kl + 1 - kf, 0);
  *mout = kl;
  return info;
}

// xSTEBZ: the eigenvalues of the symmetric tridiagonal matrix with diagonal
// D and off-diagonal E (N) by bisection: all (RANGE = 'A'), those in (VL,
// VU] ('V') or the IL-th through IU-th ('I'), to an absolute accuracy
// ABSTOL (<= 0: EPS * the norm of each block), ordered by block (ORDER =
// 'B') or overall (ORDER = 'E'). The matrix is split where E(i) is
// negligible: NSPLIT blocks ending at ISPLIT(1..NSPLIT). Returns M
// eigenvalues in W with their block numbers IBLOCK (negative for those
// that did not converge). WORK has 4*N, IWORK 3*N elements. Returns INFO:
// 0, -i for an illegal argument i, 1 (some eigenvalues did not converge), 2
// (not all of IL..IU were found), 3 (both), or 4 (RANGE = 'I' and the
// Gershgorin interval was too small; bad arithmetic).
template <typename R>
blasint Stebz(char range, char order, blasint n, R vl, R vu, blasint il,
              blasint iu, R abstol, R* d, R* e, blasint* m, blasint* nsplit,
              R* w, blasint* iblock, blasint* isplit, R* work, blasint* iwork) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = R(1) / two;
  const R fudge = R(2.1);
  const R relfac = R(2);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IBLOCK(iblock);
  blasint idumma[1];
  const Vector<blasint> ISPLIT(isplit);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WORK(work);
  blasint ib, ibegin, idiscl, idiscu, ie, iend, iinfo, im, in, ioff, iorder,
      iout, irange, itmax, itmp1, iw, iwoff, j, jb, jdisc, je, nb, nwl, nwu;
  bool ncnvrg, toofew;
  R atoli, bnorm, gl, gu, pivmin, rtoli, safemn, tmp1, tmp2, tnorm, ulp, wkill,
      wl, wlu, wu, wul;
  blasint info = 0;
  // Decode RANGE
  if (Lsame(range, 'A')) {
    irange = 1;
  } else if (Lsame(range, 'V')) {
    irange = 2;
  } else if (Lsame(range, 'I')) {
    irange = 3;
  } else {
    irange = 0;
  }
  // Decode ORDER
  if (Lsame(order, 'B')) {
    iorder = 2;
  } else if (Lsame(order, 'E')) {
    iorder = 1;
  } else {
    iorder = 0;
  }
  // Check for Errors
  if (irange <= 0) {
    info = -1;
  } else if (iorder <= 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (irange == 2) {
    if (vl >= vu) info = -5;
  } else if (irange == 3 && (il < 1 || il > Max<blasint>(1, n))) {
    info = -6;
  } else if (irange == 3 && (iu < Min<blasint>(n, il) || iu > n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<R>("STEBZ", -info);
    return info;
  }
  // Initialize error flags
  info = 0;
  ncnvrg = false;
  toofew = false;
  // Quick return if possible
  *m = 0;
  if (n == 0) return info;
  // Simplifications:
  if (irange == 3 && il == 1 && iu == n) irange = 1;
  // Get machine constants
  // NB is the minimum vector length for vector bisection, or 0
  // if only scalar is to be done.
  safemn = Lamch<R>('S');
  ulp = Lamch<R>('P');
  rtoli = ulp * relfac;
  nb = Ilaenv<R>(1, "STEBZ", " ", n, -1, -1, -1);
  if (nb <= 1) nb = 0;
  // Special Case when N=1
  if (n == 1) {
    *nsplit = 1;
    ISPLIT(1) = 1;
    if (irange == 2 && (vl >= D(1) || vu < D(1))) {
      *m = 0;
    } else {
      W(1) = D(1);
      IBLOCK(1) = 1;
      *m = 1;
    }
    return info;
  }
  // Compute Splitting Points
  *nsplit = 1;
  WORK(n) = zero;
  pivmin = one;
  for (j = 2; j <= n; ++j) {
    tmp1 = (E(j - 1) * E(j - 1));
    if (std::abs(D(j) * D(j - 1)) * (ulp * ulp) + safemn > tmp1) {
      ISPLIT(*nsplit) = j - 1;
      *nsplit = *nsplit + 1;
      WORK(j - 1) = zero;
    } else {
      WORK(j - 1) = tmp1;
      pivmin = Max(pivmin, tmp1);
    }
  }
  ISPLIT(*nsplit) = n;
  pivmin = pivmin * safemn;
  // Compute Interval and ATOLI
  if (irange == 3) {
    // RANGE='I': Compute the interval containing eigenvalues
    // IL through IU.
    // Compute Gershgorin interval for entire (split) matrix
    // and use it as the initial interval
    gu = D(1);
    gl = D(1);
    tmp1 = zero;
    for (j = 1; j <= n - 1; ++j) {
      tmp2 = std::sqrt(WORK(j));
      gu = Max(gu, D(j) + tmp1 + tmp2);
      gl = Min(gl, D(j) - tmp1 - tmp2);
      tmp1 = tmp2;
    }
    gu = Max(gu, D(n) + tmp1);
    gl = Min(gl, D(n) - tmp1);
    tnorm = Max(std::abs(gl), std::abs(gu));
    gl = gl - fudge * tnorm * ulp * n - fudge * two * pivmin;
    gu = gu + fudge * tnorm * ulp * n + fudge * pivmin;
    // Compute Iteration parameters
    itmax = static_cast<blasint>((std::log(tnorm + pivmin) - std::log(pivmin)) /
                                 std::log(two)) +
            2;
    if (abstol <= zero) {
      atoli = ulp * tnorm;
    } else {
      atoli = abstol;
    }
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
    Laebz(3, itmax, n, 2, 2, nb, atoli, rtoli, pivmin, d, e, work, IWORK.ptr(5),
          WORK.ptr(n + 1), WORK.ptr(n + 5), &iout, iwork, w, iblock);
    if (IWORK(6) == iu) {
      wl = WORK(n + 1);
      wlu = WORK(n + 3);
      nwl = IWORK(1);
      wu = WORK(n + 4);
      wul = WORK(n + 2);
      nwu = IWORK(4);
    } else {
      wl = WORK(n + 2);
      wlu = WORK(n + 4);
      nwl = IWORK(2);
      wu = WORK(n + 3);
      wul = WORK(n + 1);
      nwu = IWORK(3);
    }
    if (nwl < 0 || nwl >= n || nwu < 1 || nwu > n) {
      info = 4;
      return info;
    }
  } else {
    // RANGE='A' or 'V' -- Set ATOLI
    tnorm = Max(std::abs(D(1)) + std::abs(E(1)),
                std::abs(D(n)) + std::abs(E(n - 1)));
    for (j = 2; j <= n - 1; ++j) {
      tnorm = Max(tnorm, std::abs(D(j)) + std::abs(E(j - 1)) + std::abs(E(j)));
    }
    if (abstol <= zero) {
      atoli = ulp * tnorm;
    } else {
      atoli = abstol;
    }
    if (irange == 2) {
      wl = vl;
      wu = vu;
    } else {
      wl = zero;
      wu = zero;
    }
  }
  // Find Eigenvalues -- Loop Over Blocks and recompute NWL and NWU.
  // NWL accumulates the number of eigenvalues .le. WL,
  // NWU accumulates the number of eigenvalues .le. WU
  *m = 0;
  iend = 0;
  info = 0;
  nwl = 0;
  nwu = 0;
  for (jb = 1; jb <= *nsplit; ++jb) {
    ioff = iend;
    ibegin = ioff + 1;
    iend = ISPLIT(jb);
    in = iend - ioff;
    if (in == 1) {
      // Special Case -- IN=1
      if (irange == 1 || wl >= D(ibegin) - pivmin) nwl = nwl + 1;
      if (irange == 1 || wu >= D(ibegin) - pivmin) nwu = nwu + 1;
      if (irange == 1 ||
          (wl < D(ibegin) - pivmin && wu >= D(ibegin) - pivmin)) {
        *m = *m + 1;
        W(*m) = D(ibegin);
        IBLOCK(*m) = jb;
      }
    } else {
      // General Case -- IN > 1
      // Compute Gershgorin Interval
      // and use it as the initial interval
      gu = D(ibegin);
      gl = D(ibegin);
      tmp1 = zero;
      for (j = ibegin; j <= iend - 1; ++j) {
        tmp2 = std::abs(E(j));
        gu = Max(gu, D(j) + tmp1 + tmp2);
        gl = Min(gl, D(j) - tmp1 - tmp2);
        tmp1 = tmp2;
      }
      gu = Max(gu, D(iend) + tmp1);
      gl = Min(gl, D(iend) - tmp1);
      bnorm = Max(std::abs(gl), std::abs(gu));
      gl = gl - fudge * bnorm * ulp * in - fudge * pivmin;
      gu = gu + fudge * bnorm * ulp * in + fudge * pivmin;
      // Compute ATOLI for the current submatrix
      if (abstol <= zero) {
        atoli = ulp * Max(std::abs(gl), std::abs(gu));
      } else {
        atoli = abstol;
      }
      if (irange > 1) {
        if (gu < wl) {
          nwl = nwl + in;
          nwu = nwu + in;
          goto L70;
        }
        gl = Max(gl, wl);
        gu = Min(gu, wu);
        if (gl >= gu) goto L70;
      }
      // Set Up Initial Interval
      WORK(n + 1) = gl;
      WORK(n + in + 1) = gu;
      iinfo = Laebz(1, 0, in, in, 1, nb, atoli, rtoli, pivmin, D.ptr(ibegin),
                    E.ptr(ibegin), WORK.ptr(ibegin), idumma, WORK.ptr(n + 1),
                    WORK.ptr(n + 2 * in + 1), &im, iwork, W.ptr(*m + 1),
                    IBLOCK.ptr(*m + 1));
      nwl = nwl + IWORK(1);
      nwu = nwu + IWORK(in + 1);
      iwoff = *m - IWORK(1);
      // Compute Eigenvalues
      itmax =
          static_cast<blasint>((std::log(gu - gl + pivmin) - std::log(pivmin)) /
                               std::log(two)) +
          2;
      iinfo = Laebz(2, itmax, in, in, 1, nb, atoli, rtoli, pivmin,
                    D.ptr(ibegin), E.ptr(ibegin), WORK.ptr(ibegin), idumma,
                    WORK.ptr(n + 1), WORK.ptr(n + 2 * in + 1), &iout, iwork,
                    W.ptr(*m + 1), IBLOCK.ptr(*m + 1));
      // Copy Eigenvalues Into W and IBLOCK
      // Use -JB for block number for unconverged eigenvalues.
      for (j = 1; j <= iout; ++j) {
        tmp1 = half * (WORK(j + n) + WORK(j + in + n));
        // Flag non-convergence.
        if (j > iout - iinfo) {
          ncnvrg = true;
          ib = -jb;
        } else {
          ib = jb;
        }
        for (je = IWORK(j) + 1 + iwoff; je <= IWORK(j + in) + iwoff; ++je) {
          W(je) = tmp1;
          IBLOCK(je) = ib;
        }
      }
      *m = *m + im;
    }
  L70:;
  }
  // If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
  // If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
  if (irange == 3) {
    im = 0;
    idiscl = il - 1 - nwl;
    idiscu = nwu - iu;
    if (idiscl > 0 || idiscu > 0) {
      for (je = 1; je <= *m; ++je) {
        if (W(je) <= wlu && idiscl > 0) {
          idiscl = idiscl - 1;
        } else if (W(je) >= wul && idiscu > 0) {
          idiscu = idiscu - 1;
        } else {
          im = im + 1;
          W(im) = W(je);
          IBLOCK(im) = IBLOCK(je);
        }
      }
      *m = im;
    }
    if (idiscl > 0 || idiscu > 0) {
      // Code to deal with effects of bad arithmetic:
      // Some low eigenvalues to be discarded are not in (WL,WLU],
      // or high eigenvalues to be discarded are not in (WUL,WU]
      // so just kill off the smallest IDISCL/largest IDISCU
      // eigenvalues, by simply finding the smallest/largest
      // eigenvalue(s).
      // (If N(w) is monotone non-decreasing, this should never
      // happen.)
      if (idiscl > 0) {
        wkill = wu;
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
        wkill = wl;
        for (jdisc = 1; jdisc <= idiscu; ++jdisc) {
          iw = 0;
          for (je = 1; je <= *m; ++je) {
            if (IBLOCK(je) != 0 && (W(je) > wkill || iw == 0)) {
              iw = je;
              wkill = W(je);
            }
          }
          IBLOCK(iw) = 0;
        }
      }
      im = 0;
      for (je = 1; je <= *m; ++je) {
        if (IBLOCK(je) != 0) {
          im = im + 1;
          W(im) = W(je);
          IBLOCK(im) = IBLOCK(je);
        }
      }
      *m = im;
    }
    if (idiscl < 0 || idiscu < 0) {
      toofew = true;
    }
  }
  // If ORDER='B', do nothing -- the eigenvalues are already sorted
  // by block.
  // If ORDER='E', sort the eigenvalues from smallest to largest
  if (iorder == 1 && *nsplit > 1) {
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
        itmp1 = IBLOCK(ie);
        W(ie) = W(je);
        IBLOCK(ie) = IBLOCK(je);
        W(je) = tmp1;
        IBLOCK(je) = itmp1;
      }
    }
  }
  info = 0;
  if (ncnvrg) info = info + 1;
  if (toofew) info = info + 2;
  return info;
}

// xLAGTF: factors the tridiagonal matrix T - LAMBDA*I (T with diagonal A,
// superdiagonal B, subdiagonal C; N) as P * L * U with partial pivoting:
// on exit A holds the diagonal of U, B and D its first and second
// superdiagonals, C the multipliers of L and IN the pivots; IN(N) is the
// smallest k with a pivot below TOL (relative; at least EPS), 0 if none.
// Returns INFO: 0 or -1 for N < 0. Real R only.
template <typename R>
blasint Lagtf(blasint n, R* a, R lambda, R* b, R* c, R tol, R* d, blasint* in) {
  const R zero = R(0);
  const Vector<R> A(a);
  const Vector<R> B(b);
  const Vector<R> C(c);
  const Vector<R> D(d);
  const Vector<blasint> IN(in);
  blasint k;
  R eps, mult, piv1, piv2, scale1, scale2, temp, tl;
  blasint info = 0;
  if (n < 0) {
    info = -1;
    Xerbla<R>("LAGTF", -info);
    return info;
  }
  if (n == 0) return info;
  A(1) = A(1) - lambda;
  IN(n) = 0;
  if (n == 1) {
    if (A(1) == zero) IN(1) = 1;
    return info;
  }
  eps = Lamch<R>('E');
  tl = Max(tol, eps);
  scale1 = std::abs(A(1)) + std::abs(B(1));
  for (k = 1; k <= n - 1; ++k) {
    A(k + 1) = A(k + 1) - lambda;
    scale2 = std::abs(C(k)) + std::abs(A(k + 1));
    if (k < (n - 1)) scale2 = scale2 + std::abs(B(k + 1));
    if (A(k) == zero) {
      piv1 = zero;
    } else {
      piv1 = std::abs(A(k)) / scale1;
    }
    if (C(k) == zero) {
      IN(k) = 0;
      piv2 = zero;
      scale1 = scale2;
      if (k < (n - 1)) D(k) = zero;
    } else {
      piv2 = std::abs(C(k)) / scale2;
      if (piv2 <= piv1) {
        IN(k) = 0;
        scale1 = scale2;
        C(k) = C(k) / A(k);
        A(k + 1) = A(k + 1) - C(k) * B(k);
        if (k < (n - 1)) D(k) = zero;
      } else {
        IN(k) = 1;
        mult = A(k) / C(k);
        A(k) = C(k);
        temp = A(k + 1);
        A(k + 1) = B(k) - mult * temp;
        if (k < (n - 1)) {
          D(k) = B(k + 1);
          B(k + 1) = -mult * D(k);
        }
        B(k) = temp;
        C(k) = mult;
      }
    }
    if ((Max(piv1, piv2) <= tl) && (IN(n) == 0)) IN(n) = k;
  }
  if ((std::abs(A(n)) <= scale1 * tl) && (IN(n) == 0)) IN(n) = n;
  return info;
}

// xLAGTS: solves (T - LAMBDA*I) x = Y (JOB = 1 or -1) or its transpose
// (JOB = 2 or -2) with the factorization of xLAGTF; Y is overwritten by x.
// For JOB < 0, small pivots are perturbed by multiples of TOL (TOL <= 0 on
// entry: EPS * the largest element of U, returned in TOL) so that no
// overflow occurs; for JOB > 0, INFO = k > 0 says that the k-th element of
// x would overflow. Returns INFO: 0, -i for an illegal argument i, or k.
template <typename R>
blasint Lagts(blasint job, blasint n, R* a, R* b, R* c, R* d, blasint* in, R* y,
              R* tol) {
  const R one = R(1);
  const R zero = R(0);
  const Vector<R> A(a);
  const Vector<R> B(b);
  const Vector<R> C(c);
  const Vector<R> D(d);
  const Vector<blasint> IN(in);
  const Vector<R> Y(y);
  blasint k;
  R absak, ak, bignum, eps, pert, sfmin, temp;
  blasint info = 0;
  if ((std::abs(job) > 2) || (job == 0)) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  }
  if (info != 0) {
    Xerbla<R>("LAGTS", -info);
    return info;
  }
  if (n == 0) return info;
  eps = Lamch<R>('E');
  sfmin = Lamch<R>('S');
  bignum = one / sfmin;
  if (job < 0) {
    if (*tol <= zero) {
      *tol = std::abs(A(1));
      if (n > 1) *tol = Max(*tol, std::abs(A(2)), std::abs(B(1)));
      for (k = 3; k <= n; ++k) {
        *tol =
            Max(*tol, std::abs(A(k)), std::abs(B(k - 1)), std::abs(D(k - 2)));
      }
      *tol = *tol * eps;
      if (*tol == zero) *tol = eps;
    }
  }
  if (std::abs(job) == 1) {
    for (k = 2; k <= n; ++k) {
      if (IN(k - 1) == 0) {
        Y(k) = Y(k) - C(k - 1) * Y(k - 1);
      } else {
        temp = Y(k - 1);
        Y(k - 1) = Y(k);
        Y(k) = temp - C(k - 1) * Y(k);
      }
    }
    if (job == 1) {
      for (k = n; k >= 1; --k) {
        if (k <= n - 2) {
          temp = Y(k) - B(k) * Y(k + 1) - D(k) * Y(k + 2);
        } else if (k == n - 1) {
          temp = Y(k) - B(k) * Y(k + 1);
        } else {
          temp = Y(k);
        }
        ak = A(k);
        absak = std::abs(ak);
        if (absak < one) {
          if (absak < sfmin) {
            if (absak == zero || std::abs(temp) * sfmin > absak) {
              info = k;
              return info;
            } else {
              temp = temp * bignum;
              ak = ak * bignum;
            }
          } else if (std::abs(temp) > absak * bignum) {
            info = k;
            return info;
          }
        }
        Y(k) = temp / ak;
      }
    } else {
      for (k = n; k >= 1; --k) {
        if (k <= n - 2) {
          temp = Y(k) - B(k) * Y(k + 1) - D(k) * Y(k + 2);
        } else if (k == n - 1) {
          temp = Y(k) - B(k) * Y(k + 1);
        } else {
          temp = Y(k);
        }
        ak = A(k);
        pert = std::copysign(*tol, ak);
      L40:
        absak = std::abs(ak);
        if (absak < one) {
          if (absak < sfmin) {
            if (absak == zero || std::abs(temp) * sfmin > absak) {
              ak = ak + pert;
              pert = 2 * pert;
              goto L40;
            } else {
              temp = temp * bignum;
              ak = ak * bignum;
            }
          } else if (std::abs(temp) > absak * bignum) {
            ak = ak + pert;
            pert = 2 * pert;
            goto L40;
          }
        }
        Y(k) = temp / ak;
      }
    }
  } else {
    // Come to here if  JOB = 2 or -2
    if (job == 2) {
      for (k = 1; k <= n; ++k) {
        if (k >= 3) {
          temp = Y(k) - B(k - 1) * Y(k - 1) - D(k - 2) * Y(k - 2);
        } else if (k == 2) {
          temp = Y(k) - B(k - 1) * Y(k - 1);
        } else {
          temp = Y(k);
        }
        ak = A(k);
        absak = std::abs(ak);
        if (absak < one) {
          if (absak < sfmin) {
            if (absak == zero || std::abs(temp) * sfmin > absak) {
              info = k;
              return info;
            } else {
              temp = temp * bignum;
              ak = ak * bignum;
            }
          } else if (std::abs(temp) > absak * bignum) {
            info = k;
            return info;
          }
        }
        Y(k) = temp / ak;
      }
    } else {
      for (k = 1; k <= n; ++k) {
        if (k >= 3) {
          temp = Y(k) - B(k - 1) * Y(k - 1) - D(k - 2) * Y(k - 2);
        } else if (k == 2) {
          temp = Y(k) - B(k - 1) * Y(k - 1);
        } else {
          temp = Y(k);
        }
        ak = A(k);
        pert = std::copysign(*tol, ak);
      L70:
        absak = std::abs(ak);
        if (absak < one) {
          if (absak < sfmin) {
            if (absak == zero || std::abs(temp) * sfmin > absak) {
              ak = ak + pert;
              pert = 2 * pert;
              goto L70;
            } else {
              temp = temp * bignum;
              ak = ak * bignum;
            }
          } else if (std::abs(temp) > absak * bignum) {
            ak = ak + pert;
            pert = 2 * pert;
            goto L70;
          }
        }
        Y(k) = temp / ak;
      }
    }
    for (k = n; k >= 2; --k) {
      if (IN(k - 1) == 0) {
        Y(k - 1) = Y(k - 1) - C(k - 1) * Y(k);
      } else {
        temp = Y(k - 1);
        Y(k - 1) = Y(k);
        Y(k) = temp - C(k - 1) * Y(k);
      }
    }
  }
  return info;
}

// xLARUV: N (at most 128) uniform random numbers in (0, 1) into X, from
// the seed ISEED (4 integers in [0, 4095], ISEED(4) odd), which is
// updated: a multiplicative congruential generator modulo 2**48, with the
// multipliers MM(i, 1..4) = the i-th power of the multiplier in base 4096.
template <typename R>
void Laruv(blasint* iseed, blasint n, R* x) {
  const R one = R(1);
  const blasint lv = 128, ipw2 = 4096;
  const R r = one / ipw2;
  const Vector<blasint> ISEED(iseed);
  const Vector<R> X(x);
  static const blasint mm[128][4] = {
      {494, 322, 2508, 2549},   {2637, 789, 3754, 1145},
      {255, 1440, 1766, 2253},  {2008, 752, 3572, 305},
      {1253, 2859, 2893, 3301}, {3344, 123, 307, 1065},
      {4084, 1848, 1297, 3133}, {1739, 643, 3966, 2913},
      {3143, 2405, 758, 3285},  {3468, 2638, 2598, 1241},
      {688, 2344, 3406, 1197},  {1657, 46, 2922, 3729},
      {1238, 3814, 1038, 2501}, {3166, 913, 2934, 1673},
      {1292, 3649, 2091, 541},  {3422, 339, 2451, 2753},
      {1270, 3808, 1580, 949},  {2016, 822, 1958, 2361},
      {154, 2832, 2055, 1165},  {2862, 3078, 1507, 4081},
      {697, 3633, 1078, 2725},  {1706, 2970, 3273, 3305},
      {491, 637, 17, 3069},     {931, 2249, 854, 3617},
      {1444, 2081, 2916, 3733}, {444, 4019, 3971, 409},
      {3577, 1478, 2889, 2157}, {3944, 242, 3831, 1361},
      {2184, 481, 2621, 3973},  {1661, 2075, 1541, 1865},
      {3482, 4058, 893, 2525},  {657, 622, 736, 1409},
      {3023, 3376, 3992, 3445}, {3618, 812, 787, 3577},
      {1267, 234, 2125, 77},    {1828, 641, 2364, 3761},
      {164, 4005, 2460, 2149},  {3798, 1122, 257, 1449},
      {3087, 3135, 1574, 3005}, {2400, 2640, 3912, 225},
      {2870, 2302, 1216, 85},   {3876, 40, 3248, 3673},
      {1905, 1832, 3401, 3117}, {1593, 2247, 2124, 3089},
      {1797, 2034, 2762, 1349}, {1234, 2637, 149, 2057},
      {3460, 1287, 2245, 413},  {328, 1691, 166, 65},
      {2861, 496, 466, 1845},   {1950, 1597, 4018, 697},
      {617, 2394, 1399, 3085},  {2070, 2584, 190, 3441},
      {3331, 1843, 2879, 1573}, {769, 336, 153, 3689},
      {1558, 1472, 2320, 2941}, {2412, 2407, 18, 929},
      {2800, 433, 712, 533},    {189, 2096, 2159, 2841},
      {287, 1761, 2318, 4077},  {2045, 2810, 2091, 721},
      {1227, 566, 3443, 2821},  {2838, 442, 1510, 2249},
      {209, 41, 449, 2397},     {2770, 1238, 1956, 2817},
      {3654, 1086, 2201, 245},  {3993, 603, 3137, 1913},
      {192, 840, 3399, 1997},   {2253, 3168, 1321, 3121},
      {3491, 1499, 2271, 997},  {2889, 1084, 3667, 1833},
      {2857, 3438, 2703, 2877}, {2094, 2408, 629, 1633},
      {1818, 1589, 2365, 981},  {688, 2391, 2431, 2009},
      {1407, 288, 1113, 941},   {634, 26, 3922, 2449},
      {3231, 512, 2554, 197},   {815, 1456, 184, 2441},
      {3524, 171, 2099, 285},   {1914, 1677, 3228, 1473},
      {516, 2657, 4012, 2741},  {164, 2270, 1921, 3129},
      {303, 2587, 3452, 909},   {2144, 2961, 3901, 2801},
      {3480, 1970, 572, 421},   {119, 1817, 3309, 4073},
      {3357, 676, 3171, 2813},  {837, 1410, 817, 2337},
      {2826, 3723, 3039, 1429}, {2332, 2803, 1696, 1177},
      {2089, 3185, 1256, 1901}, {3780, 184, 3715, 81},
      {1700, 663, 2077, 1669},  {3712, 499, 3019, 2633},
      {150, 3784, 1497, 2269},  {2000, 1631, 1101, 129},
      {3375, 1925, 717, 1141},  {1621, 3912, 51, 249},
      {3090, 1398, 981, 3917},  {3765, 1349, 1978, 2481},
      {1149, 1441, 1813, 3941}, {3146, 2224, 3881, 2217},
      {33, 2411, 76, 2749},     {3082, 1907, 3846, 3041},
      {2741, 3192, 3694, 1877}, {359, 2786, 1682, 345},
      {3316, 382, 124, 2861},   {1749, 37, 1660, 1809},
      {185, 759, 3997, 3141},   {2784, 2948, 479, 2825},
      {2202, 1862, 1141, 157},  {2199, 3802, 886, 2881},
      {1364, 2423, 3514, 3637}, {1244, 2051, 1301, 1465},
      {2020, 2295, 3604, 2829}, {3160, 1332, 1888, 2161},
      {2785, 1832, 1836, 3365}, {2772, 2405, 1990, 361},
      {1217, 3638, 2058, 2685}, {1822, 3661, 692, 3745},
      {1245, 327, 1194, 2325},  {2252, 3660, 20, 3609},
      {3904, 716, 3285, 3821},  {2774, 1842, 2046, 3537},
      {997, 3987, 2107, 517},   {2573, 1368, 3508, 3017},
      {1148, 1848, 3525, 2141}, {545, 2366, 3801, 1537}};
  auto MM = [](blasint i, blasint j) { return mm[i - 1][j - 1]; };
  blasint i, i1, i2, i3, i4, it1 = 0, it2 = 0, it3 = 0, it4 = 0;

  // Quick return for N < 1
  if (n < 1) return;

  i1 = ISEED(1);
  i2 = ISEED(2);
  i3 = ISEED(3);
  i4 = ISEED(4);

  for (i = 1; i <= Min<blasint>(n, lv); ++i) {
  L20:
    // Multiply the seed by i-th power of the multiplier modulo 2**48
    it4 = i4 * MM(i, 4);
    it3 = it4 / ipw2;
    it4 = it4 - ipw2 * it3;
    it3 = it3 + i3 * MM(i, 4) + i4 * MM(i, 3);
    it2 = it3 / ipw2;
    it3 = it3 - ipw2 * it2;
    it2 = it2 + i2 * MM(i, 4) + i3 * MM(i, 3) + i4 * MM(i, 2);
    it1 = it2 / ipw2;
    it2 = it2 - ipw2 * it1;
    it1 = it1 + i1 * MM(i, 4) + i2 * MM(i, 3) + i3 * MM(i, 2) + i4 * MM(i, 1);
    it1 = it1 % ipw2;

    // Convert 48-bit integer to a real number in the interval (0,1)
    X(i) = r * (R(it1) + r * (R(it2) + r * (R(it3) + r * R(it4))));

    if (X(i) == one) {
      // If a real number has n bits of precision, and the first
      // n bits of the 48-bit integer above happen to be all 1 (which
      // will occur about once every 2**n calls), then X( I ) will
      // be rounded to exactly 1.0.
      // Since X( I ) is not supposed to return exactly 0.0 or 1.0,
      // the statistically correct thing to do in this situation is
      // simply to iterate again.
      // N.B. the case X( I ) = 0.0 should not be possible.
      i1 = i1 + 2;
      i2 = i2 + 2;
      i3 = i3 + 2;
      i4 = i4 + 2;
      goto L20;
    }
  }

  // Return final value of seed
  ISEED(1) = it1;
  ISEED(2) = it2;
  ISEED(3) = it3;
  ISEED(4) = it4;
}

namespace {

// DLARNV (real T).
template <typename R>
void LarnvReal(blasint idist, blasint* iseed, blasint n, R* x) {
  const R one = R(1);
  const R two = R(2);
  const blasint lv = 128;
  const R twopi = R(6.28318530717958647692528676655900576839e+0);
  const Vector<blasint> ISEED(iseed);
  R u[lv] = {};
  const Vector<R> U(u);
  const Vector<R> X(x);
  blasint i, il, il2, iv;
  for (iv = 1; iv <= n; iv += lv / 2) {
    il = Min<blasint>(lv / 2, n - iv + 1);
    if (idist == 3) {
      il2 = 2 * il;
    } else {
      il2 = il;
    }
    // Call DLARUV to generate IL2 numbers from a uniform (0,1)
    // distribution (IL2 <= LV)
    Laruv(iseed, il2, u);
    if (idist == 1) {
      // Copy generated numbers
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = U(i);
      }
    } else if (idist == 2) {
      // Convert generated numbers to uniform (-1,1) distribution
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = two * U(i) - one;
      }
    } else if (idist == 3) {
      // Convert generated numbers to normal (0,1) distribution
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = std::sqrt(-two * std::log(U(2 * i - 1))) *
                        std::cos(twopi * U(2 * i));
      }
    }
  }
}

// ZLARNV (complex T).
template <typename T>
void LarnvComplex(blasint idist, blasint* iseed, blasint n, T* x) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const blasint lv = 128;
  const R twopi = R(6.28318530717958647692528676655900576839e+0);
  const Vector<blasint> ISEED(iseed);
  R u[lv] = {};
  const Vector<R> U(u);
  const Vector<T> X(x);
  blasint i, il, iv;
  for (iv = 1; iv <= n; iv += lv / 2) {
    il = Min<blasint>(lv / 2, n - iv + 1);
    // Call DLARUV to generate 2*IL real numbers from a uniform (0,1)
    // distribution (2*IL <= LV)
    Laruv<R>(iseed, 2 * il, u);
    if (idist == 1) {
      // Copy generated numbers
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = T(U(2 * i - 1), U(2 * i));
      }
    } else if (idist == 2) {
      // Convert generated numbers to uniform (-1,1) distribution
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = T(two * U(2 * i - 1) - one, two * U(2 * i) - one);
      }
    } else if (idist == 3) {
      // Convert generated numbers to normal (0,1) distribution
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = std::sqrt(-two * std::log(U(2 * i - 1))) *
                        std::exp(T(zero, twopi * U(2 * i)));
      }
    } else if (idist == 4) {
      // Convert generated numbers to complex numbers uniformly
      // distributed on the unit disk
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) =
            std::sqrt(U(2 * i - 1)) * std::exp(T(zero, twopi * U(2 * i)));
      }
    } else if (idist == 5) {
      // Convert generated numbers to complex numbers uniformly
      // distributed on the unit circle
      for (i = 1; i <= il; ++i) {
        X(iv + i - 1) = std::exp(T(zero, twopi * U(2 * i)));
      }
    }
  }
}

}  // namespace

// xLARNV: N random numbers into X from the seed ISEED (as in xLARUV,
// updated), with the distribution IDIST: 1 uniform (0, 1), 2 uniform (-1,
// 1), 3 normal (0, 1); for complex T, real and imaginary parts so
// distributed, or 4 uniform on the unit disk, 5 uniform on the unit
// circle.
template <typename T>
void Larnv(blasint idist, blasint* iseed, blasint n, T* x) {
  if constexpr (IsComplex<T>::value) {
    LarnvComplex(idist, iseed, n, x);
  } else {
    LarnvReal(idist, iseed, n, x);
  }
}

// xSTEIN: the eigenvectors of the symmetric tridiagonal matrix with
// diagonal D and off-diagonal E (N) for the M eigenvalues W, given by
// block (IBLOCK, ISPLIT; as from xSTEBZ with ORDER = 'B'), by inverse
// iteration from random starting vectors, reorthogonalized (modified
// Gram-Schmidt) against those of close eigenvalues of the same block. On
// exit the columns of Z hold the eigenvectors; IFAIL the indices of those
// that did not converge in MAXITS iterations. WORK has 5*N, IWORK N
// elements. Returns INFO: 0, -i for an illegal argument i, or the number
// of eigenvectors that did not converge. For complex T the eigenvectors
// are real; the real routine reorthogonalizes with xDOT and xAXPY, the
// complex one with the same loops written out (as LAPACK).
template <typename T>
blasint Stein(blasint n, RealT<T>* d, RealT<T>* e, blasint m, RealT<T>* w,
              blasint* iblock, blasint* isplit, T* z, blasint ldz,
              RealT<T>* work, blasint* iwork, blasint* ifail) {
  using R = RealT<T>;

  const T czero = T(0);
  const T cone = T(1);
  const R zero = R(0);
  const R one = R(1);
  const R ten = R(10);
  const R odm3 = R(1.0e-3);
  const R odm1 = R(1.0e-1);
  const blasint maxits = 5;
  const blasint extra = 2;
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IBLOCK(iblock);
  const Vector<blasint> IFAIL(ifail);
  blasint iseed[4];
  const Vector<blasint> ISEED(iseed);
  const Vector<blasint> ISPLIT(isplit);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> W(w);
  const Vector<R> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint b1, blksiz, bn, gpind, i, indrv1, indrv2, indrv3, indrv4, indrv5, its,
      j, j1, jblk, jmax, jr, nblk, nrmchk;
  R dtpcrt, eps, eps1, nrm, onenrm, ortol, pertol, scl, sep, tol, xj, xjm, ztr;
  blasint info = 0;
  for (i = 1; i <= m; ++i) {
    IFAIL(i) = 0;
  }
  if (n < 0) {
    info = -1;
  } else if (m < 0 || m > n) {
    info = -4;
  } else if (ldz < Max<blasint>(1, n)) {
    info = -9;
  } else {
    for (j = 2; j <= m; ++j) {
      if (IBLOCK(j) < IBLOCK(j - 1)) {
        info = -6;
        goto L30;
      }
      if (IBLOCK(j) == IBLOCK(j - 1) && W(j) < W(j - 1)) {
        info = -5;
        goto L30;
      }
    }
  L30:;
  }
  if (info != 0) {
    Xerbla<T>("STEIN", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0 || m == 0) {
    return info;
  } else if (n == 1) {
    Z(1, 1) = cone;
    return info;
  }
  // Get machine constants.
  eps = Lamch<R>('P');
  // Initialize seed for random number generator DLARNV.
  for (i = 1; i <= 4; ++i) {
    ISEED(i) = 1;
  }
  // Initialize pointers.
  indrv1 = 0;
  indrv2 = indrv1 + n;
  indrv3 = indrv2 + n;
  indrv4 = indrv3 + n;
  indrv5 = indrv4 + n;
  // Compute eigenvectors of matrix blocks.
  j1 = 1;
  for (nblk = 1; nblk <= IBLOCK(m); ++nblk) {
    // Find starting and ending indices of block nblk.
    if (nblk == 1) {
      b1 = 1;
    } else {
      b1 = ISPLIT(nblk - 1) + 1;
    }
    bn = ISPLIT(nblk);
    blksiz = bn - b1 + 1;
    if (blksiz == 1) goto L60;
    gpind = j1;
    // Compute reorthogonalization criterion and stopping criterion.
    onenrm = std::abs(D(b1)) + std::abs(E(b1));
    onenrm = Max(onenrm, std::abs(D(bn)) + std::abs(E(bn - 1)));
    for (i = b1 + 1; i <= bn - 1; ++i) {
      onenrm =
          Max(onenrm, std::abs(D(i)) + std::abs(E(i - 1)) + std::abs(E(i)));
    }
    ortol = odm3 * onenrm;
    dtpcrt = std::sqrt(odm1 / blksiz);
    // Loop through eigenvalues of block nblk.
  L60:
    jblk = 0;
    for (j = j1; j <= m; ++j) {
      if (IBLOCK(j) != nblk) {
        j1 = j;
        goto L180;
      }
      jblk = jblk + 1;
      xj = W(j);
      // Skip all the work if the block size is one.
      if (blksiz == 1) {
        WORK(indrv1 + 1) = one;
        goto L140;
      }
      // If eigenvalues j and j-1 are too close, add a relatively
      // small perturbation.
      if (jblk > 1) {
        eps1 = std::abs(eps * xj);
        pertol = ten * eps1;
        sep = xj - xjm;
        if (sep < pertol) xj = xjm + pertol;
      }
      its = 0;
      nrmchk = 0;
      // Get random starting vector.
      Larnv(2, iseed, blksiz, WORK.ptr(indrv1 + 1));
      // Copy the matrix T so it won't be destroyed in factorization.
      Copy(blksiz, D.ptr(b1), 1, WORK.ptr(indrv4 + 1), 1);
      Copy(blksiz - 1, E.ptr(b1), 1, WORK.ptr(indrv2 + 2), 1);
      Copy(blksiz - 1, E.ptr(b1), 1, WORK.ptr(indrv3 + 1), 1);
      // Compute LU factors with partial pivoting  ( PT = LU )
      tol = zero;
      Lagtf(blksiz, WORK.ptr(indrv4 + 1), xj, WORK.ptr(indrv2 + 2),
            WORK.ptr(indrv3 + 1), tol, WORK.ptr(indrv5 + 1), iwork);
      // Update iteration count.
    L70:
      its = its + 1;
      if (its > maxits) goto L120;
      // Normalize and scale the righthand side vector Pb.
      jmax = Iamax(blksiz, WORK.ptr(indrv1 + 1), 1);
      scl = blksiz * onenrm * Max(eps, std::abs(WORK(indrv4 + blksiz))) /
            std::abs(WORK(indrv1 + jmax));
      Scal(blksiz, scl, WORK.ptr(indrv1 + 1), 1);
      // Solve the system LU = Pb.
      Lagts(-1, blksiz, WORK.ptr(indrv4 + 1), WORK.ptr(indrv2 + 2),
            WORK.ptr(indrv3 + 1), WORK.ptr(indrv5 + 1), iwork,
            WORK.ptr(indrv1 + 1), &tol);
      // Reorthogonalize by modified Gram-Schmidt if eigenvalues are
      // close enough.
      if (jblk == 1) goto L110;
      if (std::abs(xj - xjm) > ortol) gpind = j;
      if (gpind != j) {
        for (i = gpind; i <= j - 1; ++i) {
          if constexpr (IsComplex<T>::value) {
            ztr = zero;
            for (jr = 1; jr <= blksiz; ++jr) {
              ztr = ztr + WORK(indrv1 + jr) * std::real(Z(b1 - 1 + jr, i));
            }
            for (jr = 1; jr <= blksiz; ++jr) {
              WORK(indrv1 + jr) =
                  WORK(indrv1 + jr) - ztr * std::real(Z(b1 - 1 + jr, i));
            }
          } else {
            ztr = -Dotc(blksiz, WORK.ptr(indrv1 + 1), 1, Z.ptr(b1, i), 1);
            Axpy(blksiz, ztr, Z.ptr(b1, i), 1, WORK.ptr(indrv1 + 1), 1);
          }
        }
      }
      // Check the infinity norm of the iterate.
    L110:
      jmax = Iamax(blksiz, WORK.ptr(indrv1 + 1), 1);
      nrm = std::abs(WORK(indrv1 + jmax));
      // Continue for additional iterations after norm reaches
      // stopping criterion.
      if (nrm < dtpcrt) goto L70;
      nrmchk = nrmchk + 1;
      if (nrmchk < extra + 1) goto L70;
      goto L130;
      // If stopping criterion was not satisfied, update info and
      // store eigenvector number in array ifail.
    L120:
      info = info + 1;
      IFAIL(info) = j;
      // Accept iterate as jth eigenvector.
    L130:
      scl = one / Nrm2(blksiz, WORK.ptr(indrv1 + 1), 1);
      jmax = Iamax(blksiz, WORK.ptr(indrv1 + 1), 1);
      if (WORK(indrv1 + jmax) < zero) scl = -scl;
      Scal(blksiz, scl, WORK.ptr(indrv1 + 1), 1);
    L140:
      for (i = 1; i <= n; ++i) {
        Z(i, j) = czero;
      }
      for (i = 1; i <= blksiz; ++i) {
        Z(b1 + i - 1, j) = T(WORK(indrv1 + i));
      }
      // Save the shift to check eigenvalue spacing at next
      // iteration.
      xjm = xj;
    }
  L180:;
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                  \
  template blasint Laebz<R>(blasint, blasint, blasint, blasint, blasint,    \
                            blasint, R, R, R, R*, R*, R*, blasint*, R*, R*, \
                            blasint*, blasint*, R*, blasint*);              \
  template blasint Stebz<R>(char, char, blasint, R, R, blasint, blasint, R, \
                            R*, R*, blasint*, blasint*, R*, blasint*,       \
                            blasint*, R*, blasint*);                        \
  template blasint Lagtf<R>(blasint, R*, R, R*, R*, R, R*, blasint*);       \
  template blasint Lagts<R>(blasint, blasint, R*, R*, R*, R*, blasint*, R*, \
                            R*);                                            \
  template void Laruv<R>(blasint*, blasint, R*);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template void Larnv<T>(blasint, blasint*, blasint, T*);                      \
  template blasint Stein<T>(blasint, RealT<T>*, RealT<T>*, blasint, RealT<T>*, \
                            blasint*, blasint*, T*, blasint, RealT<T>*,        \
                            blasint*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_STEBZ_REAL(p, R)                                        \
  HWYBLAS_API void p##laebz_(                                                  \
      const hwyblas_int* ijob, const hwyblas_int* nitmax,                      \
      const hwyblas_int* n, const hwyblas_int* mmax, const hwyblas_int* minp,  \
      const hwyblas_int* nbmin, const R* abstol, const R* reltol,              \
      const R* pivmin, R* d, R* e, R* e2, hwyblas_int* nval, R* ab, R* c,      \
      hwyblas_int* mout, hwyblas_int* nab, R* work, hwyblas_int* iwork,        \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Laebz(*ijob, *nitmax, *n, *mmax, *minp, *nbmin,   \
                                   *abstol, *reltol, *pivmin, d, e, e2, nval,  \
                                   ab, c, mout, nab, work, iwork);             \
  }                                                                            \
  HWYBLAS_API void p##stebz_(                                                  \
      const char* range, const char* order, const hwyblas_int* n, const R* vl, \
      const R* vu, const hwyblas_int* il, const hwyblas_int* iu,               \
      const R* abstol, R* d, R* e, hwyblas_int* m, hwyblas_int* nsplit, R* w,  \
      hwyblas_int* iblock, hwyblas_int* isplit, R* work, hwyblas_int* iwork,   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Stebz(*range, *order, *n, *vl, *vu, *il, *iu,     \
                                   *abstol, d, e, m, nsplit, w, iblock,        \
                                   isplit, work, iwork);                       \
  }                                                                            \
  HWYBLAS_API void p##lagtf_(const hwyblas_int* n, R* a, const R* lambda,      \
                             R* b, R* c, const R* tol, R* d, hwyblas_int* in,  \
                             hwyblas_int* info) {                              \
    *info = hwyblas::lapack::Lagtf(*n, a, *lambda, b, c, *tol, d, in);         \
  }                                                                            \
  HWYBLAS_API void p##lagts_(const hwyblas_int* job, const hwyblas_int* n,     \
                             R* a, R* b, R* c, R* d, hwyblas_int* in, R* y,    \
                             R* tol, hwyblas_int* info) {                      \
    *info = hwyblas::lapack::Lagts(*job, *n, a, b, c, d, in, y, tol);          \
  }                                                                            \
  HWYBLAS_API void p##laruv_(hwyblas_int* iseed, const hwyblas_int* n, R* x) { \
    hwyblas::lapack::Laruv(iseed, *n, x);                                      \
  }                                                                            \
  HWYBLAS_API void p##larnv_(const hwyblas_int* idist, hwyblas_int* iseed,     \
                             const hwyblas_int* n, R* x) {                     \
    hwyblas::lapack::Larnv(*idist, iseed, *n, x);                              \
  }                                                                            \
  HWYBLAS_API void p##stein_(                                                  \
      const hwyblas_int* n, R* d, R* e, const hwyblas_int* m, R* w,            \
      hwyblas_int* iblock, hwyblas_int* isplit, R* z, const hwyblas_int* ldz,  \
      R* work, hwyblas_int* iwork, hwyblas_int* ifail, hwyblas_int* info) {    \
    *info = hwyblas::lapack::Stein(*n, d, e, *m, w, iblock, isplit, z, *ldz,   \
                                   work, iwork, ifail);                        \
  }

HWYBLAS_LAPACK_STEBZ_REAL(s, float)
HWYBLAS_LAPACK_STEBZ_REAL(d, double)

#undef HWYBLAS_LAPACK_STEBZ_REAL

#define HWYBLAS_LAPACK_STEIN_COMPLEX(p, T, R)                                 \
  HWYBLAS_API void p##larnv_(const hwyblas_int* idist, hwyblas_int* iseed,    \
                             const hwyblas_int* n, R* x) {                    \
    hwyblas::lapack::Larnv(*idist, iseed, *n, As<T>(x));                      \
  }                                                                           \
  HWYBLAS_API void p##stein_(                                                 \
      const hwyblas_int* n, R* d, R* e, const hwyblas_int* m, R* w,           \
      hwyblas_int* iblock, hwyblas_int* isplit, R* z, const hwyblas_int* ldz, \
      R* work, hwyblas_int* iwork, hwyblas_int* ifail, hwyblas_int* info) {   \
    *info = hwyblas::lapack::Stein(*n, d, e, *m, w, iblock, isplit, As<T>(z), \
                                   *ldz, work, iwork, ifail);                 \
  }

HWYBLAS_LAPACK_STEIN_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_STEIN_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_STEIN_COMPLEX

}  // extern "C"
