// The eigenvectors of a pair (S, P) in generalized Schur form: xTGEVC.
//
// The real and the complex routine differ in the eigenvalues (2-by-2 blocks
// of S for complex conjugate pairs in the real routine), the solver of the
// small systems and the workspaces, so they are two overloads.
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

// xTGEVC (real T): computes some or all of the right and / or left
// eigenvectors of the pair (S, P), S quasi-upper triangular and P upper
// triangular (as returned by xHGEQZ with JOB = 'S'): S*x = w*P*x and
// y**H*S = w*y**H*P. SIDE = 'R', 'L' or 'B' (both). HOWMNY = 'A': all
// eigenvectors; 'B': all, back-transformed by the matrices in VL / VR on
// entry (Q and Z of xHGEQZ: the eigenvectors of the original pair); 'S':
// those selected by SELECT (a complex pair if either of SELECT(j),
// SELECT(j+1) is true; SELECT is not changed). A complex eigenvector takes
// two columns (real and imaginary part). Each eigenvector is scaled so
// that its largest component has |re| + |im| = 1. MM is the number of
// columns of VL / VR, M returns the number used. WORK has 6*N elements.
//
// Returns INFO: 0, -i for an illegal argument i (-5 / -7: S / P has an
// illegal 2-by-2 block), or the index of the eigenvalue whose 2-by-2
// block's eigenvalues are real (xLAG2): no eigenvectors returned then.
// (The variable COMPL is named compl_: compl is a C++ keyword.)
template <typename R>
blasint Tgevc(char side, char howmny, const blasint* select, blasint n,
              const R* s, blasint lds, const R* p, blasint ldp, R* vl,
              blasint ldvl, R* vr, blasint ldvr, blasint mm, blasint* m,
              R* work) {
  using T = R;

  const R zero = R(0);
  const R one = R(1);
  const R safety = R(100);
  // BDIAG(2) is not used for NA = 1 (initialized for clang-tidy).
  R bdiag[2] = {};
  const Vector<R> BDIAG(bdiag);
  const Matrix<const R> P(p, ldp);
  const Matrix<const R> S(s, lds);
  const Vector<const blasint> SELECT(select);
  R sum[4];
  const Matrix<R> SUM(sum, 2);
  R sump[4];
  const Matrix<R> SUMP(sump, 2);
  R sums[4];
  const Matrix<R> SUMS(sums, 2);
  const Matrix<R> VL(vl, ldvl);
  const Matrix<R> VR(vr, ldvr);
  const Vector<R> WORK(work);
  blasint i, ibeg, ieig, iend, ihwmny, im, iside, j, ja, jc, je, jr, jw, na, nw;
  bool compl_, compr, il2by2, ilabad, ilall, ilback, ilbbad, ilcomp, ilcplx,
      lsa, lsb;
  R acoef, acoefa, anorm, ascale, bcoefa, bcoefi, bcoefr, big, bignum, bnorm,
      bscale, cim2a, cim2b, cimaga, cimagb, cre2a, cre2b, creala, crealb, dmin,
      safmin, salfar, sbeta, scale, small, temp, temp2, temp2i, temp2r, ulp,
      xmax, xscale;
  if (Lsame(howmny, 'A')) {
    ihwmny = 1;
    ilall = true;
    ilback = false;
  } else if (Lsame(howmny, 'S')) {
    ihwmny = 2;
    ilall = false;
    ilback = false;
  } else if (Lsame(howmny, 'B')) {
    ihwmny = 3;
    ilall = true;
    ilback = true;
  } else {
    ihwmny = -1;
    ilall = true;
  }
  if (Lsame(side, 'R')) {
    iside = 1;
    compl_ = false;
    compr = true;
  } else if (Lsame(side, 'L')) {
    iside = 2;
    compl_ = true;
    compr = false;
  } else if (Lsame(side, 'B')) {
    iside = 3;
    compl_ = true;
    compr = true;
  } else {
    iside = -1;
  }
  blasint info = 0;
  if (iside < 0) {
    info = -1;
  } else if (ihwmny < 0) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (lds < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldp < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("TGEVC", -info);
    return info;
  }
  // Count the number of eigenvectors to be computed
  if (!ilall) {
    im = 0;
    ilcplx = false;
    for (j = 1; j <= n; ++j) {
      if (ilcplx) {
        ilcplx = false;
        goto L10;
      }
      if (j < n) {
        if (S(j + 1, j) != zero) ilcplx = true;
      }
      if (ilcplx) {
        if (SELECT(j) || SELECT(j + 1)) im = im + 2;
      } else {
        if (SELECT(j)) im = im + 1;
      }
    L10:;
    }
  } else {
    im = n;
  }
  // Check 2-by-2 diagonal blocks of A, B
  ilabad = false;
  ilbbad = false;
  for (j = 1; j <= n - 1; ++j) {
    if (S(j + 1, j) != zero) {
      if (P(j, j) == zero || P(j + 1, j + 1) == zero || P(j, j + 1) != zero)
        ilbbad = true;
      if (j < n - 1) {
        if (S(j + 2, j + 1) != zero) ilabad = true;
      }
    }
  }
  if (ilabad) {
    info = -5;
  } else if (ilbbad) {
    info = -7;
  } else if ((compl_ && ldvl < n) || ldvl < 1) {
    info = -10;
  } else if ((compr && ldvr < n) || ldvr < 1) {
    info = -12;
  } else if (mm < im) {
    info = -13;
  }
  if (info != 0) {
    Xerbla<T>("TGEVC", -info);
    return info;
  }
  // Quick return if possible
  *m = im;
  if (n == 0) return info;
  // Machine Constants
  safmin = Lamch<R>('S');
  big = one / safmin;
  ulp = Lamch<R>('E') * Lamch<R>('B');
  small = safmin * n / ulp;
  big = one / small;
  bignum = one / (safmin * n);
  // Compute the 1-norm of each column of the strictly upper triangular
  // part (i.e., excluding all elements belonging to the diagonal
  // blocks) of A and B to check for possible overflow in the
  // triangular solver.
  anorm = std::abs(S(1, 1));
  if (n > 1) anorm = anorm + std::abs(S(2, 1));
  bnorm = std::abs(P(1, 1));
  WORK(1) = zero;
  WORK(n + 1) = zero;
  for (j = 2; j <= n; ++j) {
    temp = zero;
    temp2 = zero;
    if (S(j, j - 1) == zero) {
      iend = j - 1;
    } else {
      iend = j - 2;
    }
    for (i = 1; i <= iend; ++i) {
      temp = temp + std::abs(S(i, j));
      temp2 = temp2 + std::abs(P(i, j));
    }
    WORK(j) = temp;
    WORK(n + j) = temp2;
    for (i = iend + 1; i <= Min<blasint>(j + 1, n); ++i) {
      temp = temp + std::abs(S(i, j));
      temp2 = temp2 + std::abs(P(i, j));
    }
    anorm = Max(anorm, temp);
    bnorm = Max(bnorm, temp2);
  }
  ascale = one / Max(anorm, safmin);
  bscale = one / Max(bnorm, safmin);
  // Left eigenvectors
  if (compl_) {
    ieig = 0;
    // Main loop over eigenvalues
    ilcplx = false;
    for (je = 1; je <= n; ++je) {
      // Skip this iteration if (a) HOWMNY='S' and SELECT=.FALSE., or
      // (b) this would be the second of a complex pair.
      // Check for complex eigenvalue, so as to be sure of which
      // entry(-ies) of SELECT to look at.
      if (ilcplx) {
        ilcplx = false;
        goto L220;
      }
      nw = 1;
      if (je < n) {
        if (S(je + 1, je) != zero) {
          ilcplx = true;
          nw = 2;
        }
      }
      if (ilall) {
        ilcomp = true;
      } else if (ilcplx) {
        ilcomp = SELECT(je) || SELECT(je + 1);
      } else {
        ilcomp = SELECT(je);
      }
      if (!ilcomp) goto L220;
      // Decide if (a) singular pencil, (b) real eigenvalue, or
      // (c) complex eigenvalue.
      if (!ilcplx) {
        if (std::abs(S(je, je)) <= safmin && std::abs(P(je, je)) <= safmin) {
          // Singular matrix pencil -- return unit eigenvector
          ieig = ieig + 1;
          for (jr = 1; jr <= n; ++jr) {
            VL(jr, ieig) = zero;
          }
          VL(ieig, ieig) = one;
          goto L220;
        }
      }
      // Clear vector
      for (jr = 1; jr <= nw * n; ++jr) {
        WORK(2 * n + jr) = zero;
      }
      // T
      // Compute coefficients in  ( a A - b B )  y = 0
      // a  is  ACOEF
      // b  is  BCOEFR + i*BCOEFI
      if (!ilcplx) {
        // Real eigenvalue
        temp = one / Max(std::abs(S(je, je)) * ascale,
                         std::abs(P(je, je)) * bscale, safmin);
        salfar = (temp * S(je, je)) * ascale;
        sbeta = (temp * P(je, je)) * bscale;
        acoef = sbeta * ascale;
        bcoefr = salfar * bscale;
        bcoefi = zero;
        // Scale to avoid underflow
        scale = one;
        lsa = std::abs(sbeta) >= safmin && std::abs(acoef) < small;
        lsb = std::abs(salfar) >= safmin && std::abs(bcoefr) < small;
        if (lsa) scale = (small / std::abs(sbeta)) * Min(anorm, big);
        if (lsb)
          scale = Max(scale, (small / std::abs(salfar)) * Min(bnorm, big));
        if (lsa || lsb) {
          scale =
              Min(scale,
                  one / (safmin * Max(one, std::abs(acoef), std::abs(bcoefr))));
          if (lsa) {
            acoef = ascale * (scale * sbeta);
          } else {
            acoef = scale * acoef;
          }
          if (lsb) {
            bcoefr = bscale * (scale * salfar);
          } else {
            bcoefr = scale * bcoefr;
          }
        }
        acoefa = std::abs(acoef);
        bcoefa = std::abs(bcoefr);
        // First component is 1
        WORK(2 * n + je) = one;
        xmax = one;
      } else {
        // Complex eigenvalue
        Lag2(S.ptr(je, je), lds, P.ptr(je, je), ldp, safmin * safety, &acoef,
             &temp, &bcoefr, &temp2, &bcoefi);
        bcoefi = -bcoefi;
        if (bcoefi == zero) {
          info = je;
          return info;
        }
        // Scale to avoid over/underflow
        acoefa = std::abs(acoef);
        bcoefa = std::abs(bcoefr) + std::abs(bcoefi);
        scale = one;
        if (acoefa * ulp < safmin && acoefa >= safmin)
          scale = (safmin / ulp) / acoefa;
        if (bcoefa * ulp < safmin && bcoefa >= safmin)
          scale = Max(scale, (safmin / ulp) / bcoefa);
        if (safmin * acoefa > ascale) scale = ascale / (safmin * acoefa);
        if (safmin * bcoefa > bscale)
          scale = Min(scale, bscale / (safmin * bcoefa));
        if (scale != one) {
          acoef = scale * acoef;
          acoefa = std::abs(acoef);
          bcoefr = scale * bcoefr;
          bcoefi = scale * bcoefi;
          bcoefa = std::abs(bcoefr) + std::abs(bcoefi);
        }
        // Compute first two components of eigenvector
        temp = acoef * S(je + 1, je);
        temp2r = acoef * S(je, je) - bcoefr * P(je, je);
        temp2i = -bcoefi * P(je, je);
        if (std::abs(temp) > std::abs(temp2r) + std::abs(temp2i)) {
          WORK(2 * n + je) = one;
          WORK(3 * n + je) = zero;
          WORK(2 * n + je + 1) = -temp2r / temp;
          WORK(3 * n + je + 1) = -temp2i / temp;
        } else {
          WORK(2 * n + je + 1) = one;
          WORK(3 * n + je + 1) = zero;
          temp = acoef * S(je, je + 1);
          WORK(2 * n + je) =
              (bcoefr * P(je + 1, je + 1) - acoef * S(je + 1, je + 1)) / temp;
          WORK(3 * n + je) = bcoefi * P(je + 1, je + 1) / temp;
        }
        xmax = Max(
            std::abs(WORK(2 * n + je)) + std::abs(WORK(3 * n + je)),
            std::abs(WORK(2 * n + je + 1)) + std::abs(WORK(3 * n + je + 1)));
      }
      dmin = Max(ulp * acoefa * anorm, ulp * bcoefa * bnorm, safmin);
      // T
      // Triangular solve of  (a A - b B)  y = 0
      // T
      // (rowwise in  (a A - b B) , or columnwise in (a A - b B) )
      il2by2 = false;
      for (j = je + nw; j <= n; ++j) {
        if (il2by2) {
          il2by2 = false;
          goto L160;
        }
        na = 1;
        BDIAG(1) = P(j, j);
        if (j < n) {
          if (S(j + 1, j) != zero) {
            il2by2 = true;
            BDIAG(2) = P(j + 1, j + 1);
            na = 2;
          }
        }
        // Check whether scaling is necessary for dot products
        xscale = one / Max(one, xmax);
        temp =
            Max(WORK(j), WORK(n + j), acoefa * WORK(j) + bcoefa * WORK(n + j));
        if (il2by2)
          temp = Max(temp, WORK(j + 1), WORK(n + j + 1),
                     acoefa * WORK(j + 1) + bcoefa * WORK(n + j + 1));
        if (temp > bignum * xscale) {
          for (jw = 0; jw <= nw - 1; ++jw) {
            for (jr = je; jr <= j - 1; ++jr) {
              WORK((jw + 2) * n + jr) = xscale * WORK((jw + 2) * n + jr);
            }
          }
          xmax = xmax * xscale;
        }
        // Compute dot products
        // j-1
        // SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
        // k=je
        // To reduce the op count, this is done as
        // _        j-1                  _        j-1
        // a*conjg( sum  S(k,j)*x(k) ) - b*conjg( sum  P(k,j)*x(k) )
        // k=je                          k=je
        // which may cause underflow problems if A or B are close
        // to underflow.  (E.g., less than SMALL.)
        for (jw = 1; jw <= nw; ++jw) {
          for (ja = 1; ja <= na; ++ja) {
            SUMS(ja, jw) = zero;
            SUMP(ja, jw) = zero;
            for (jr = je; jr <= j - 1; ++jr) {
              SUMS(ja, jw) =
                  SUMS(ja, jw) + S(jr, j + ja - 1) * WORK((jw + 1) * n + jr);
              SUMP(ja, jw) =
                  SUMP(ja, jw) + P(jr, j + ja - 1) * WORK((jw + 1) * n + jr);
            }
          }
        }
        for (ja = 1; ja <= na; ++ja) {
          if (ilcplx) {
            SUM(ja, 1) = -acoef * SUMS(ja, 1) + bcoefr * SUMP(ja, 1) -
                         bcoefi * SUMP(ja, 2);
            SUM(ja, 2) = -acoef * SUMS(ja, 2) + bcoefr * SUMP(ja, 2) +
                         bcoefi * SUMP(ja, 1);
          } else {
            SUM(ja, 1) = -acoef * SUMS(ja, 1) + bcoefr * SUMP(ja, 1);
          }
        }
        // T
        // Solve  ( a A - b B )  y = SUM(,)
        // with scaling and perturbation of the denominator
        // IINFO of xLALN2 (perturbed system) is not used, as in LAPACK.
        Laln2(true, na, nw, dmin, acoef, S.ptr(j, j), lds, BDIAG(1), BDIAG(2),
              sum, 2, bcoefr, bcoefi, WORK.ptr(2 * n + j), n, &scale, &temp);
        if (scale < one) {
          for (jw = 0; jw <= nw - 1; ++jw) {
            for (jr = je; jr <= j - 1; ++jr) {
              WORK((jw + 2) * n + jr) = scale * WORK((jw + 2) * n + jr);
            }
          }
          xmax = scale * xmax;
        }
        xmax = Max(xmax, temp);
      L160:;
      }
      // Copy eigenvector to VL, back transforming if
      // HOWMNY='B'.
      ieig = ieig + 1;
      if (ilback) {
        for (jw = 0; jw <= nw - 1; ++jw) {
          Gemv('N', n, n + 1 - je, one, VL.ptr(1, je), ldvl,
               WORK.ptr((jw + 2) * n + je), 1, zero, WORK.ptr((jw + 4) * n + 1),
               1);
        }
        Lacpy(' ', n, nw, WORK.ptr(4 * n + 1), n, VL.ptr(1, je), ldvl);
        ibeg = 1;
      } else {
        Lacpy(' ', n, nw, WORK.ptr(2 * n + 1), n, VL.ptr(1, ieig), ldvl);
        ibeg = je;
      }
      // Scale eigenvector
      xmax = zero;
      if (ilcplx) {
        for (j = ibeg; j <= n; ++j) {
          xmax = Max(xmax, std::abs(VL(j, ieig)) + std::abs(VL(j, ieig + 1)));
        }
      } else {
        for (j = ibeg; j <= n; ++j) {
          xmax = Max(xmax, std::abs(VL(j, ieig)));
        }
      }
      if (xmax > safmin) {
        xscale = one / xmax;
        for (jw = 0; jw <= nw - 1; ++jw) {
          for (jr = ibeg; jr <= n; ++jr) {
            VL(jr, ieig + jw) = xscale * VL(jr, ieig + jw);
          }
        }
      }
      ieig = ieig + nw - 1;
    L220:;
    }
  }
  // Right eigenvectors
  if (compr) {
    ieig = im + 1;
    // Main loop over eigenvalues
    ilcplx = false;
    for (je = n; je >= 1; --je) {
      // Skip this iteration if (a) HOWMNY='S' and SELECT=.FALSE., or
      // (b) this would be the second of a complex pair.
      // Check for complex eigenvalue, so as to be sure of which
      // entry(-ies) of SELECT to look at -- if complex, SELECT(JE)
      // or SELECT(JE-1).
      // If this is a complex pair, the 2-by-2 diagonal block
      // corresponding to the eigenvalue is in rows/columns JE-1:JE
      if (ilcplx) {
        ilcplx = false;
        goto L500;
      }
      nw = 1;
      if (je > 1) {
        if (S(je, je - 1) != zero) {
          ilcplx = true;
          nw = 2;
        }
      }
      if (ilall) {
        ilcomp = true;
      } else if (ilcplx) {
        ilcomp = SELECT(je) || SELECT(je - 1);
      } else {
        ilcomp = SELECT(je);
      }
      if (!ilcomp) goto L500;
      // Decide if (a) singular pencil, (b) real eigenvalue, or
      // (c) complex eigenvalue.
      if (!ilcplx) {
        if (std::abs(S(je, je)) <= safmin && std::abs(P(je, je)) <= safmin) {
          // Singular matrix pencil -- unit eigenvector
          ieig = ieig - 1;
          for (jr = 1; jr <= n; ++jr) {
            VR(jr, ieig) = zero;
          }
          VR(ieig, ieig) = one;
          goto L500;
        }
      }
      // Clear vector
      for (jw = 0; jw <= nw - 1; ++jw) {
        for (jr = 1; jr <= n; ++jr) {
          WORK((jw + 2) * n + jr) = zero;
        }
      }
      // Compute coefficients in  ( a A - b B ) x = 0
      // a  is  ACOEF
      // b  is  BCOEFR + i*BCOEFI
      if (!ilcplx) {
        // Real eigenvalue
        temp = one / Max(std::abs(S(je, je)) * ascale,
                         std::abs(P(je, je)) * bscale, safmin);
        salfar = (temp * S(je, je)) * ascale;
        sbeta = (temp * P(je, je)) * bscale;
        acoef = sbeta * ascale;
        bcoefr = salfar * bscale;
        bcoefi = zero;
        // Scale to avoid underflow
        scale = one;
        lsa = std::abs(sbeta) >= safmin && std::abs(acoef) < small;
        lsb = std::abs(salfar) >= safmin && std::abs(bcoefr) < small;
        if (lsa) scale = (small / std::abs(sbeta)) * Min(anorm, big);
        if (lsb)
          scale = Max(scale, (small / std::abs(salfar)) * Min(bnorm, big));
        if (lsa || lsb) {
          scale =
              Min(scale,
                  one / (safmin * Max(one, std::abs(acoef), std::abs(bcoefr))));
          if (lsa) {
            acoef = ascale * (scale * sbeta);
          } else {
            acoef = scale * acoef;
          }
          if (lsb) {
            bcoefr = bscale * (scale * salfar);
          } else {
            bcoefr = scale * bcoefr;
          }
        }
        acoefa = std::abs(acoef);
        bcoefa = std::abs(bcoefr);
        // First component is 1
        WORK(2 * n + je) = one;
        xmax = one;
        // Compute contribution from column JE of A and B to sum
        // (See "Further Details", above.)
        for (jr = 1; jr <= je - 1; ++jr) {
          WORK(2 * n + jr) = bcoefr * P(jr, je) - acoef * S(jr, je);
        }
      } else {
        // Complex eigenvalue
        Lag2(S.ptr(je - 1, je - 1), lds, P.ptr(je - 1, je - 1), ldp,
             safmin * safety, &acoef, &temp, &bcoefr, &temp2, &bcoefi);
        if (bcoefi == zero) {
          info = je - 1;
          return info;
        }
        // Scale to avoid over/underflow
        acoefa = std::abs(acoef);
        bcoefa = std::abs(bcoefr) + std::abs(bcoefi);
        scale = one;
        if (acoefa * ulp < safmin && acoefa >= safmin)
          scale = (safmin / ulp) / acoefa;
        if (bcoefa * ulp < safmin && bcoefa >= safmin)
          scale = Max(scale, (safmin / ulp) / bcoefa);
        if (safmin * acoefa > ascale) scale = ascale / (safmin * acoefa);
        if (safmin * bcoefa > bscale)
          scale = Min(scale, bscale / (safmin * bcoefa));
        if (scale != one) {
          acoef = scale * acoef;
          acoefa = std::abs(acoef);
          bcoefr = scale * bcoefr;
          bcoefi = scale * bcoefi;
          bcoefa = std::abs(bcoefr) + std::abs(bcoefi);
        }
        // Compute first two components of eigenvector
        // and contribution to sums
        temp = acoef * S(je, je - 1);
        temp2r = acoef * S(je, je) - bcoefr * P(je, je);
        temp2i = -bcoefi * P(je, je);
        if (std::abs(temp) >= std::abs(temp2r) + std::abs(temp2i)) {
          WORK(2 * n + je) = one;
          WORK(3 * n + je) = zero;
          WORK(2 * n + je - 1) = -temp2r / temp;
          WORK(3 * n + je - 1) = -temp2i / temp;
        } else {
          WORK(2 * n + je - 1) = one;
          WORK(3 * n + je - 1) = zero;
          temp = acoef * S(je - 1, je);
          WORK(2 * n + je) =
              (bcoefr * P(je - 1, je - 1) - acoef * S(je - 1, je - 1)) / temp;
          WORK(3 * n + je) = bcoefi * P(je - 1, je - 1) / temp;
        }
        xmax = Max(
            std::abs(WORK(2 * n + je)) + std::abs(WORK(3 * n + je)),
            std::abs(WORK(2 * n + je - 1)) + std::abs(WORK(3 * n + je - 1)));
        // Compute contribution from columns JE and JE-1
        // of A and B to the sums.
        creala = acoef * WORK(2 * n + je - 1);
        cimaga = acoef * WORK(3 * n + je - 1);
        crealb = bcoefr * WORK(2 * n + je - 1) - bcoefi * WORK(3 * n + je - 1);
        cimagb = bcoefi * WORK(2 * n + je - 1) + bcoefr * WORK(3 * n + je - 1);
        cre2a = acoef * WORK(2 * n + je);
        cim2a = acoef * WORK(3 * n + je);
        cre2b = bcoefr * WORK(2 * n + je) - bcoefi * WORK(3 * n + je);
        cim2b = bcoefi * WORK(2 * n + je) + bcoefr * WORK(3 * n + je);
        for (jr = 1; jr <= je - 2; ++jr) {
          WORK(2 * n + jr) = -creala * S(jr, je - 1) + crealb * P(jr, je - 1) -
                             cre2a * S(jr, je) + cre2b * P(jr, je);
          WORK(3 * n + jr) = -cimaga * S(jr, je - 1) + cimagb * P(jr, je - 1) -
                             cim2a * S(jr, je) + cim2b * P(jr, je);
        }
      }
      dmin = Max(ulp * acoefa * anorm, ulp * bcoefa * bnorm, safmin);
      // Columnwise triangular solve of  (a A - b B)  x = 0
      il2by2 = false;
      for (j = je - nw; j >= 1; --j) {
        // If a 2-by-2 block, is in position j-1:j, wait until
        // next iteration to process it (when it will be j:j+1)
        if (!il2by2 && j > 1) {
          if (S(j, j - 1) != zero) {
            il2by2 = true;
            goto L370;
          }
        }
        BDIAG(1) = P(j, j);
        if (il2by2) {
          na = 2;
          BDIAG(2) = P(j + 1, j + 1);
        } else {
          na = 1;
        }
        // Compute x(j) (and x(j+1), if 2-by-2 block)
        Laln2(false, na, nw, dmin, acoef, S.ptr(j, j), lds, BDIAG(1), BDIAG(2),
              WORK.ptr(2 * n + j), n, bcoefr, bcoefi, sum, 2, &scale, &temp);
        if (scale < one) {
          for (jw = 0; jw <= nw - 1; ++jw) {
            for (jr = 1; jr <= je; ++jr) {
              WORK((jw + 2) * n + jr) = scale * WORK((jw + 2) * n + jr);
            }
          }
        }
        xmax = Max(scale * xmax, temp);
        for (jw = 1; jw <= nw; ++jw) {
          for (ja = 1; ja <= na; ++ja) {
            WORK((jw + 1) * n + j + ja - 1) = SUM(ja, jw);
          }
        }
        // w = w + x(j)*(a S(*,j) - b P(*,j) ) with scaling
        if (j > 1) {
          // Check whether scaling is necessary for sum.
          xscale = one / Max(one, xmax);
          temp = acoefa * WORK(j) + bcoefa * WORK(n + j);
          if (il2by2)
            temp = Max(temp, acoefa * WORK(j + 1) + bcoefa * WORK(n + j + 1));
          temp = Max(temp, acoefa, bcoefa);
          if (temp > bignum * xscale) {
            for (jw = 0; jw <= nw - 1; ++jw) {
              for (jr = 1; jr <= je; ++jr) {
                WORK((jw + 2) * n + jr) = xscale * WORK((jw + 2) * n + jr);
              }
            }
            xmax = xmax * xscale;
          }
          // Compute the contributions of the off-diagonals of
          // column j (and j+1, if 2-by-2 block) of A and B to the
          // sums.
          for (ja = 1; ja <= na; ++ja) {
            if (ilcplx) {
              creala = acoef * WORK(2 * n + j + ja - 1);
              cimaga = acoef * WORK(3 * n + j + ja - 1);
              crealb = bcoefr * WORK(2 * n + j + ja - 1) -
                       bcoefi * WORK(3 * n + j + ja - 1);
              cimagb = bcoefi * WORK(2 * n + j + ja - 1) +
                       bcoefr * WORK(3 * n + j + ja - 1);
              for (jr = 1; jr <= j - 1; ++jr) {
                WORK(2 * n + jr) = WORK(2 * n + jr) -
                                   creala * S(jr, j + ja - 1) +
                                   crealb * P(jr, j + ja - 1);
                WORK(3 * n + jr) = WORK(3 * n + jr) -
                                   cimaga * S(jr, j + ja - 1) +
                                   cimagb * P(jr, j + ja - 1);
              }
            } else {
              creala = acoef * WORK(2 * n + j + ja - 1);
              crealb = bcoefr * WORK(2 * n + j + ja - 1);
              for (jr = 1; jr <= j - 1; ++jr) {
                WORK(2 * n + jr) = WORK(2 * n + jr) -
                                   creala * S(jr, j + ja - 1) +
                                   crealb * P(jr, j + ja - 1);
              }
            }
          }
        }
        il2by2 = false;
      L370:;
      }
      // Copy eigenvector to VR, back transforming if
      // HOWMNY='B'.
      ieig = ieig - nw;
      if (ilback) {
        for (jw = 0; jw <= nw - 1; ++jw) {
          for (jr = 1; jr <= n; ++jr) {
            WORK((jw + 4) * n + jr) = WORK((jw + 2) * n + 1) * VR(jr, 1);
          }
          // A series of compiler directives to defeat
          // vectorization for the next loop
          for (jc = 2; jc <= je; ++jc) {
            for (jr = 1; jr <= n; ++jr) {
              WORK((jw + 4) * n + jr) = WORK((jw + 4) * n + jr) +
                                        WORK((jw + 2) * n + jc) * VR(jr, jc);
            }
          }
        }
        for (jw = 0; jw <= nw - 1; ++jw) {
          for (jr = 1; jr <= n; ++jr) {
            VR(jr, ieig + jw) = WORK((jw + 4) * n + jr);
          }
        }
        iend = n;
      } else {
        for (jw = 0; jw <= nw - 1; ++jw) {
          for (jr = 1; jr <= n; ++jr) {
            VR(jr, ieig + jw) = WORK((jw + 2) * n + jr);
          }
        }
        iend = je;
      }
      // Scale eigenvector
      xmax = zero;
      if (ilcplx) {
        for (j = 1; j <= iend; ++j) {
          xmax = Max(xmax, std::abs(VR(j, ieig)) + std::abs(VR(j, ieig + 1)));
        }
      } else {
        for (j = 1; j <= iend; ++j) {
          xmax = Max(xmax, std::abs(VR(j, ieig)));
        }
      }
      if (xmax > safmin) {
        xscale = one / xmax;
        for (jw = 0; jw <= nw - 1; ++jw) {
          for (jr = 1; jr <= iend; ++jr) {
            VR(jr, ieig + jw) = xscale * VR(jr, ieig + jw);
          }
        }
      }
    L500:;
    }
  }
  return info;
}

// xTGEVC (complex T): as the real routine for the upper triangular S and P
// (P with a real diagonal, as xHGEQZ returns it). WORK has 2*N elements,
// RWORK 2*N.
template <typename T>
blasint Tgevc(char side, char howmny, const blasint* select, blasint n,
              const T* s, blasint lds, const T* p, blasint ldp, T* vl,
              blasint ldvl, T* vr, blasint ldvr, blasint mm, blasint* m,
              T* work, RealT<T>* rwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<const T> P(p, ldp);
  const Vector<R> RWORK(rwork);
  const Matrix<const T> S(s, lds);
  const Vector<const blasint> SELECT(select);
  const Matrix<T> VL(vl, ldvl);
  const Matrix<T> VR(vr, ldvr);
  const Vector<T> WORK(work);
  blasint i, ibeg, ieig, iend, ihwmny, im, iside, isrc, j, je, jr;
  bool compl_, compr, ilall, ilback, ilbbad, ilcomp, lsa, lsb;
  R acoefa, acoeff, anorm, ascale, bcoefa, big, bignum, bnorm, bscale, dmin,
      safmin, sbeta, scale, small, temp, ulp, xmax;
  T bcoeff, ca, cb, d, salpha, sum, suma, sumb;
  if (Lsame(howmny, 'A')) {
    ihwmny = 1;
    ilall = true;
    ilback = false;
  } else if (Lsame(howmny, 'S')) {
    ihwmny = 2;
    ilall = false;
    ilback = false;
  } else if (Lsame(howmny, 'B')) {
    ihwmny = 3;
    ilall = true;
    ilback = true;
  } else {
    ihwmny = -1;
  }
  if (Lsame(side, 'R')) {
    iside = 1;
    compl_ = false;
    compr = true;
  } else if (Lsame(side, 'L')) {
    iside = 2;
    compl_ = true;
    compr = false;
  } else if (Lsame(side, 'B')) {
    iside = 3;
    compl_ = true;
    compr = true;
  } else {
    iside = -1;
  }
  blasint info = 0;
  if (iside < 0) {
    info = -1;
  } else if (ihwmny < 0) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (lds < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldp < Max<blasint>(1, n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("TGEVC", -info);
    return info;
  }
  // Count the number of eigenvectors
  if (!ilall) {
    im = 0;
    for (j = 1; j <= n; ++j) {
      if (SELECT(j)) im = im + 1;
    }
  } else {
    im = n;
  }
  // Check diagonal of B
  ilbbad = false;
  for (j = 1; j <= n; ++j) {
    if (std::imag(P(j, j)) != zero) ilbbad = true;
  }
  if (ilbbad) {
    info = -7;
  } else if ((compl_ && ldvl < n) || ldvl < 1) {
    info = -10;
  } else if ((compr && ldvr < n) || ldvr < 1) {
    info = -12;
  } else if (mm < im) {
    info = -13;
  }
  if (info != 0) {
    Xerbla<T>("TGEVC", -info);
    return info;
  }
  // Quick return if possible
  *m = im;
  if (n == 0) return info;
  // Machine Constants
  safmin = Lamch<R>('S');
  big = one / safmin;
  ulp = Lamch<R>('E') * Lamch<R>('B');
  small = safmin * n / ulp;
  big = one / small;
  bignum = one / (safmin * n);
  // Compute the 1-norm of each column of the strictly upper triangular
  // part of A and B to check for possible overflow in the triangular
  // solver.
  anorm = Cabs1(S(1, 1));
  bnorm = Cabs1(P(1, 1));
  RWORK(1) = zero;
  RWORK(n + 1) = zero;
  for (j = 2; j <= n; ++j) {
    RWORK(j) = zero;
    RWORK(n + j) = zero;
    for (i = 1; i <= j - 1; ++i) {
      RWORK(j) = RWORK(j) + Cabs1(S(i, j));
      RWORK(n + j) = RWORK(n + j) + Cabs1(P(i, j));
    }
    anorm = Max(anorm, RWORK(j) + Cabs1(S(j, j)));
    bnorm = Max(bnorm, RWORK(n + j) + Cabs1(P(j, j)));
  }
  ascale = one / Max(anorm, safmin);
  bscale = one / Max(bnorm, safmin);
  // Left eigenvectors
  if (compl_) {
    ieig = 0;
    // Main loop over eigenvalues
    for (je = 1; je <= n; ++je) {
      if (ilall) {
        ilcomp = true;
      } else {
        ilcomp = SELECT(je);
      }
      if (ilcomp) {
        ieig = ieig + 1;
        if (Cabs1(S(je, je)) <= safmin &&
            std::abs(std::real(P(je, je))) <= safmin) {
          // Singular matrix pencil -- return unit eigenvector
          for (jr = 1; jr <= n; ++jr) {
            VL(jr, ieig) = czero;
          }
          VL(ieig, ieig) = cone;
          goto L140;
        }
        // Non-singular eigenvalue:
        // Compute coefficients  a  and  b  in
        // H
        // y  ( a A - b B ) = 0
        temp = one / Max(Cabs1(S(je, je)) * ascale,
                         std::abs(std::real(P(je, je))) * bscale, safmin);
        salpha = (temp * S(je, je)) * ascale;
        sbeta = (temp * std::real(P(je, je))) * bscale;
        acoeff = sbeta * ascale;
        bcoeff = salpha * bscale;
        // Scale to avoid underflow
        lsa = std::abs(sbeta) >= safmin && std::abs(acoeff) < small;
        lsb = Cabs1(salpha) >= safmin && Cabs1(bcoeff) < small;
        scale = one;
        if (lsa) scale = (small / std::abs(sbeta)) * Min(anorm, big);
        if (lsb) scale = Max(scale, (small / Cabs1(salpha)) * Min(bnorm, big));
        if (lsa || lsb) {
          scale = Min(scale, one / (safmin *
                                    Max(one, std::abs(acoeff), Cabs1(bcoeff))));
          if (lsa) {
            acoeff = ascale * (scale * sbeta);
          } else {
            acoeff = scale * acoeff;
          }
          if (lsb) {
            bcoeff = bscale * (scale * salpha);
          } else {
            bcoeff = scale * bcoeff;
          }
        }
        acoefa = std::abs(acoeff);
        bcoefa = Cabs1(bcoeff);
        xmax = one;
        for (jr = 1; jr <= n; ++jr) {
          WORK(jr) = czero;
        }
        WORK(je) = cone;
        dmin = Max(ulp * acoefa * anorm, ulp * bcoefa * bnorm, safmin);
        // H
        // Triangular solve of  (a A - b B)  y = 0
        // H
        // (rowwise in  (a A - b B) , or columnwise in a A - b B)
        for (j = je + 1; j <= n; ++j) {
          // Compute
          // j-1
          // SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
          // k=je
          // (Scale if necessary)
          temp = one / xmax;
          if (acoefa * RWORK(j) + bcoefa * RWORK(n + j) > bignum * temp) {
            for (jr = je; jr <= j - 1; ++jr) {
              WORK(jr) = temp * WORK(jr);
            }
            xmax = one;
          }
          suma = czero;
          sumb = czero;
          for (jr = je; jr <= j - 1; ++jr) {
            suma = suma + Conj(S(jr, j)) * WORK(jr);
            sumb = sumb + Conj(P(jr, j)) * WORK(jr);
          }
          sum = acoeff * suma - Conj(bcoeff) * sumb;
          // Form x(j) = - SUM / conjg( a*S(j,j) - b*P(j,j) )
          // with scaling and perturbation of the denominator
          d = Conj(acoeff * S(j, j) - bcoeff * P(j, j));
          if (Cabs1(d) <= dmin) d = T(dmin);
          if (Cabs1(d) < one) {
            if (Cabs1(sum) >= bignum * Cabs1(d)) {
              temp = one / Cabs1(sum);
              for (jr = je; jr <= j - 1; ++jr) {
                WORK(jr) = temp * WORK(jr);
              }
              xmax = temp * xmax;
              sum = temp * sum;
            }
          }
          WORK(j) = Ladiv(-sum, d);
          xmax = Max(xmax, Cabs1(WORK(j)));
        }
        // Back transform eigenvector if HOWMNY='B'.
        if (ilback) {
          Gemv('N', n, n + 1 - je, cone, VL.ptr(1, je), ldvl, WORK.ptr(je), 1,
               czero, WORK.ptr(n + 1), 1);
          isrc = 2;
          ibeg = 1;
        } else {
          isrc = 1;
          ibeg = je;
        }
        // Copy and scale eigenvector into column of VL
        xmax = zero;
        for (jr = ibeg; jr <= n; ++jr) {
          xmax = Max(xmax, Cabs1(WORK((isrc - 1) * n + jr)));
        }
        if (xmax > safmin) {
          temp = one / xmax;
          for (jr = ibeg; jr <= n; ++jr) {
            VL(jr, ieig) = temp * WORK((isrc - 1) * n + jr);
          }
        } else {
          ibeg = n + 1;
        }
        for (jr = 1; jr <= ibeg - 1; ++jr) {
          VL(jr, ieig) = czero;
        }
      }
    L140:;
    }
  }
  // Right eigenvectors
  if (compr) {
    ieig = im + 1;
    // Main loop over eigenvalues
    for (je = n; je >= 1; --je) {
      if (ilall) {
        ilcomp = true;
      } else {
        ilcomp = SELECT(je);
      }
      if (ilcomp) {
        ieig = ieig - 1;
        if (Cabs1(S(je, je)) <= safmin &&
            std::abs(std::real(P(je, je))) <= safmin) {
          // Singular matrix pencil -- return unit eigenvector
          for (jr = 1; jr <= n; ++jr) {
            VR(jr, ieig) = czero;
          }
          VR(ieig, ieig) = cone;
          goto L250;
        }
        // Non-singular eigenvalue:
        // Compute coefficients  a  and  b  in
        // ( a A - b B ) x  = 0
        temp = one / Max(Cabs1(S(je, je)) * ascale,
                         std::abs(std::real(P(je, je))) * bscale, safmin);
        salpha = (temp * S(je, je)) * ascale;
        sbeta = (temp * std::real(P(je, je))) * bscale;
        acoeff = sbeta * ascale;
        bcoeff = salpha * bscale;
        // Scale to avoid underflow
        lsa = std::abs(sbeta) >= safmin && std::abs(acoeff) < small;
        lsb = Cabs1(salpha) >= safmin && Cabs1(bcoeff) < small;
        scale = one;
        if (lsa) scale = (small / std::abs(sbeta)) * Min(anorm, big);
        if (lsb) scale = Max(scale, (small / Cabs1(salpha)) * Min(bnorm, big));
        if (lsa || lsb) {
          scale = Min(scale, one / (safmin *
                                    Max(one, std::abs(acoeff), Cabs1(bcoeff))));
          if (lsa) {
            acoeff = ascale * (scale * sbeta);
          } else {
            acoeff = scale * acoeff;
          }
          if (lsb) {
            bcoeff = bscale * (scale * salpha);
          } else {
            bcoeff = scale * bcoeff;
          }
        }
        acoefa = std::abs(acoeff);
        bcoefa = Cabs1(bcoeff);
        xmax = one;
        for (jr = 1; jr <= n; ++jr) {
          WORK(jr) = czero;
        }
        WORK(je) = cone;
        dmin = Max(ulp * acoefa * anorm, ulp * bcoefa * bnorm, safmin);
        // Triangular solve of  (a A - b B) x = 0  (columnwise)
        // WORK(1:j-1) contains sums w,
        // WORK(j+1:JE) contains x
        for (jr = 1; jr <= je - 1; ++jr) {
          WORK(jr) = acoeff * S(jr, je) - bcoeff * P(jr, je);
        }
        WORK(je) = cone;
        for (j = je - 1; j >= 1; --j) {
          // Form x(j) := - w(j) / d
          // with scaling and perturbation of the denominator
          d = acoeff * S(j, j) - bcoeff * P(j, j);
          if (Cabs1(d) <= dmin) d = T(dmin);
          if (Cabs1(d) < one) {
            if (Cabs1(WORK(j)) >= bignum * Cabs1(d)) {
              temp = one / Cabs1(WORK(j));
              for (jr = 1; jr <= je; ++jr) {
                WORK(jr) = temp * WORK(jr);
              }
            }
          }
          WORK(j) = Ladiv(-WORK(j), d);
          if (j > 1) {
            // w = w + x(j)*(a S(*,j) - b P(*,j) ) with scaling
            if (Cabs1(WORK(j)) > one) {
              temp = one / Cabs1(WORK(j));
              if (acoefa * RWORK(j) + bcoefa * RWORK(n + j) >= bignum * temp) {
                for (jr = 1; jr <= je; ++jr) {
                  WORK(jr) = temp * WORK(jr);
                }
              }
            }
            ca = acoeff * WORK(j);
            cb = bcoeff * WORK(j);
            for (jr = 1; jr <= j - 1; ++jr) {
              WORK(jr) = WORK(jr) + ca * S(jr, j) - cb * P(jr, j);
            }
          }
        }
        // Back transform eigenvector if HOWMNY='B'.
        if (ilback) {
          Gemv('N', n, je, cone, vr, ldvr, work, 1, czero, WORK.ptr(n + 1), 1);
          isrc = 2;
          iend = n;
        } else {
          isrc = 1;
          iend = je;
        }
        // Copy and scale eigenvector into column of VR
        xmax = zero;
        for (jr = 1; jr <= iend; ++jr) {
          xmax = Max(xmax, Cabs1(WORK((isrc - 1) * n + jr)));
        }
        if (xmax > safmin) {
          temp = one / xmax;
          for (jr = 1; jr <= iend; ++jr) {
            VR(jr, ieig) = temp * WORK((isrc - 1) * n + jr);
          }
        } else {
          iend = 0;
        }
        for (jr = iend + 1; jr <= n; ++jr) {
          VR(jr, ieig) = czero;
        }
      }
    L250:;
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                 \
  template blasint Tgevc<R>(char, char, const blasint*, blasint, const R*, \
                            blasint, const R*, blasint, R*, blasint, R*,   \
                            blasint, blasint, blasint*, R*);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                              \
  template blasint Tgevc<T>(char, char, const blasint*, blasint, const T*, \
                            blasint, const T*, blasint, T*, blasint, T*,   \
                            blasint, blasint, blasint*, T*, RealT<T>*);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// SELECT is a LOGICAL array (nonzero is true).
#define HWYBLAS_LAPACK_TGEVC_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##tgevc_(                                             \
      const char* side, const char* howmny, const hwyblas_int* select,         \
      const hwyblas_int* n, const R* s, const hwyblas_int* lds, const R* p,    \
      const hwyblas_int* ldp, R* vl, const hwyblas_int* ldvl, R* vr,           \
      const hwyblas_int* ldvr, const hwyblas_int* mm, hwyblas_int* m, R* work, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Tgevc(*side, *howmny, select, *n, s, *lds, p,     \
                                   *ldp, vl, *ldvl, vr, *ldvr, *mm, m, work);  \
  }

HWYBLAS_LAPACK_TGEVC_REAL(s, float)
HWYBLAS_LAPACK_TGEVC_REAL(d, double)

#undef HWYBLAS_LAPACK_TGEVC_REAL

#define HWYBLAS_LAPACK_TGEVC_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##tgevc_(                                             \
      const char* side, const char* howmny, const hwyblas_int* select,         \
      const hwyblas_int* n, const R* s, const hwyblas_int* lds, const R* p,    \
      const hwyblas_int* ldp, R* vl, const hwyblas_int* ldvl, R* vr,           \
      const hwyblas_int* ldvr, const hwyblas_int* mm, hwyblas_int* m, R* work, \
      R* rwork, hwyblas_int* info) {                                           \
    *info = hwyblas::lapack::Tgevc(                                            \
        *side, *howmny, select, *n, As<const T>(s), *lds, As<const T>(p),      \
        *ldp, As<T>(vl), *ldvl, As<T>(vr), *ldvr, *mm, m, As<T>(work), rwork); \
  }

HWYBLAS_LAPACK_TGEVC_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_TGEVC_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_TGEVC_COMPLEX

}  // extern "C"
