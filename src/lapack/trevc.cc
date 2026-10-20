// Eigenvectors of a matrix in Schur form: xTREVC3 (blocked back-transform
// with the Level 3 BLAS).
//
// The real and the complex routine differ throughout (2-by-2 blocks of the
// real Schur form, solved with xLALN2, against xLATRS for the triangular
// complex one) and in their arguments (RWORK), so they are ported as two
// overloads.
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

// xTREVC3 (real T): computes right (SIDE = 'R') and/or left ('L', 'B':
// both) eigenvectors of the N-by-N upper quasi-triangular matrix T in Schur
// canonical form (xHSEQR). HOWMNY = 'A': all of them, in VR / VL; 'B': all
// of them, back-transformed with the matrices Q in VR / VL on entry (Schur
// vectors of A = Q*T*Q**T give eigenvectors of A); 'S': the ones that
// SELECT chooses (a complex pair if either of SELECT(j), SELECT(j+1) is
// true; SELECT is then normalized to SELECT(j) = true, SELECT(j+1) =
// false). A complex eigenvector takes two columns (real and imaginary
// part). Each vector is scaled so that its largest component has
// |Re| + |Im| = 1. MM is the number of columns of VL / VR, M (output) the
// number used. WORK has LWORK >= max(1, 3*N) elements; with LWORK >= N +
// 2*N*NBMIN (NBMIN = 8) and HOWMNY = 'B', the back-transform is blocked
// (LWORK = -1: workspace query, the optimal size N + 2*N*NB in WORK(1)).
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename R>
blasint Trevc3(char side, char howmny, blasint* select, blasint n, const R* t,
               blasint ldt, R* vl, blasint ldvl, R* vr, blasint ldvr,
               blasint mm, blasint* m, R* work, blasint lwork) {
  using T = R;
  const R zero = R(0);
  const R one = R(1);
  const blasint nbmin = 8;
  constexpr blasint nbmax = 128;
  blasint iscomplex[nbmax];
  const Vector<blasint> ISCOMPLEX(iscomplex);
  const Vector<blasint> SELECT(select);
  const Matrix<const R> Tm(t, ldt);
  const Matrix<R> VL(vl, ldvl);
  const Matrix<R> VR(vr, ldvr);
  const Vector<R> WORK(work);
  R x[4];
  const Matrix<R> X(x, 2);
  blasint i, ii, ip, is, iv, j, j1, j2, jnxt, k, ki, ki2, maxwrk, nb;
  bool allv, bothv, leftv, lquery, over, pair, rightv, somev;
  R beta, bignum, emax, rec, scale, smin, smlnum, ulp, unfl, vcrit, vmax, wi,
      wr, xnorm;
  // Set for every vector before it is used (ISCOMPLEX(K) = -1 reuses the
  // value of the first vector of the pair); initialized for the compiler.
  R remax = zero;
  // Decode and test the input parameters
  bothv = Lsame(side, 'B');
  rightv = Lsame(side, 'R') || bothv;
  leftv = Lsame(side, 'L') || bothv;
  allv = Lsame(howmny, 'A');
  over = Lsame(howmny, 'B');
  somev = Lsame(howmny, 'S');
  blasint info = 0;
  const char opts[3] = {side, howmny, '\0'};
  nb = Ilaenv<T>(1, "TREVC", opts, n, -1, -1, -1);
  maxwrk = Max<blasint>(1, n + 2 * n * nb);
  WORK(1) = WorkSize<T>(maxwrk);
  lquery = (lwork == -1);
  if (!rightv && !leftv) {
    info = -1;
  } else if (!allv && !over && !somev) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldvl < 1 || (leftv && ldvl < n)) {
    info = -8;
  } else if (ldvr < 1 || (rightv && ldvr < n)) {
    info = -10;
  } else if (lwork < Max<blasint>(1, 3 * n) && !lquery) {
    info = -14;
  } else {
    // Set M to the number of columns required to store the selected
    // eigenvectors, standardize the array SELECT if necessary, and
    // test MM.
    if (somev) {
      *m = 0;
      pair = false;
      for (j = 1; j <= n; ++j) {
        if (pair) {
          pair = false;
          SELECT(j) = 0;
        } else {
          if (j < n) {
            if (Tm(j + 1, j) == zero) {
              if (SELECT(j)) *m = *m + 1;
            } else {
              pair = true;
              if (SELECT(j) || SELECT(j + 1)) {
                SELECT(j) = 1;
                *m = *m + 2;
              }
            }
          } else {
            if (SELECT(n)) *m = *m + 1;
          }
        }
      }
    } else {
      *m = n;
    }
    if (mm < *m) {
      info = -11;
    }
  }
  if (info != 0) {
    Xerbla<T>("TREVC3", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible.
  if (n == 0) return info;
  // Use blocked version of back-transformation if sufficient workspace.
  // Zero-out the workspace to avoid potential NaN propagation.
  if (over && lwork >= n + 2 * n * nbmin) {
    nb = (lwork - n) / (2 * n);
    nb = Min<blasint>(nb, nbmax);
    Laset('F', n, 1 + 2 * nb, zero, zero, work, n);
  } else {
    nb = 1;
  }
  // Set the constants to control overflow.
  // (OVFL = 1/UNFL is computed by the Fortran routine, but not used.)
  unfl = Lamch<R>('S');
  ulp = Lamch<R>('P');
  smlnum = unfl * (n / ulp);
  bignum = (one - ulp) / smlnum;
  // Compute 1-norm of each column of strictly upper triangular
  // part of T to control overflow in triangular solver.
  WORK(1) = zero;
  for (j = 2; j <= n; ++j) {
    WORK(j) = zero;
    for (i = 1; i <= j - 1; ++i) {
      WORK(j) = WORK(j) + std::abs(Tm(i, j));
    }
  }
  // Index IP is used to specify the real or complex eigenvalue:
  // IP = 0, real eigenvalue,
  //      1, first  of conjugate complex pair: (wr,wi)
  //     -1, second of conjugate complex pair: (wr,wi)
  // ISCOMPLEX array stores IP for each column in current block.
  if (rightv) {
    // ============================================================
    // Compute right eigenvectors.
    //
    // IV is index of column in current block.
    // For complex right vector, uses IV-1 for real part and IV for complex
    // part. Non-blocked version always uses IV=2;
    // blocked     version starts with IV=NB, goes down to 1 or 2.
    // (Note the "0-th" column is used for 1-norms computed above.)
    iv = 2;
    if (nb > 2) {
      iv = nb;
    }
    ip = 0;
    is = *m;
    for (ki = n; ki >= 1; --ki) {
      // Two branches set IP = 0, as in the Fortran.
      // NOLINTBEGIN(bugprone-branch-clone)
      if (ip == -1) {
        // previous iteration (ki+1) was second of conjugate pair,
        // so this ki is first of conjugate pair; skip to end of loop
        ip = 1;
        continue;
      } else if (ki == 1) {
        // last column, so this ki must be real eigenvalue
        ip = 0;
      } else if (Tm(ki, ki - 1) == zero) {
        // zero on sub-diagonal, so this ki is real eigenvalue
        ip = 0;
      } else {
        // non-zero on sub-diagonal, so this ki is second of conjugate pair
        ip = -1;
      }
      // NOLINTEND(bugprone-branch-clone)
      if (somev) {
        if (ip == 0) {
          if (!SELECT(ki)) continue;
        } else {
          if (!SELECT(ki - 1)) continue;
        }
      }
      // Compute the KI-th eigenvalue (WR,WI).
      wr = Tm(ki, ki);
      wi = zero;
      if (ip != 0)
        wi = std::sqrt(std::abs(Tm(ki, ki - 1))) *
             std::sqrt(std::abs(Tm(ki - 1, ki)));
      smin = Max(ulp * (std::abs(wr) + std::abs(wi)), smlnum);
      if (ip == 0) {
        // --------------------------------------------------------
        // Real right eigenvector
        WORK(ki + iv * n) = one;
        // Form right-hand side.
        for (k = 1; k <= ki - 1; ++k) {
          WORK(k + iv * n) = -Tm(k, ki);
        }
        // Solve upper quasi-triangular system:
        // [ T(1:KI-1,1:KI-1) - WR ]*X = SCALE*WORK.
        jnxt = ki - 1;
        for (j = ki - 1; j >= 1; --j) {
          if (j > jnxt) continue;
          j1 = j;
          j2 = j;
          jnxt = j - 1;
          if (j > 1) {
            if (Tm(j, j - 1) != zero) {
              j1 = j - 1;
              jnxt = j - 2;
            }
          }
          if (j1 == j2) {
            // 1-by-1 diagonal block
            Laln2(false, 1, 1, smin, one, Tm.ptr(j, j), ldt, one, one,
                  WORK.ptr(j + iv * n), n, wr, zero, x, 2, &scale, &xnorm);
            // Scale X(1,1) to avoid overflow when updating
            // the right-hand side.
            if (xnorm > one) {
              if (WORK(j) > bignum / xnorm) {
                X(1, 1) = X(1, 1) / xnorm;
                scale = scale / xnorm;
              }
            }
            // Scale if necessary
            if (scale != one) Rscal(ki, scale, WORK.ptr(1 + iv * n), 1);
            WORK(j + iv * n) = X(1, 1);
            // Update right-hand side
            Axpy(j - 1, -X(1, 1), Tm.ptr(1, j), 1, WORK.ptr(1 + iv * n), 1);
          } else {
            // 2-by-2 diagonal block
            Laln2(false, 2, 1, smin, one, Tm.ptr(j - 1, j - 1), ldt, one, one,
                  WORK.ptr(j - 1 + iv * n), n, wr, zero, x, 2, &scale, &xnorm);
            // Scale X(1,1) and X(2,1) to avoid overflow when
            // updating the right-hand side.
            if (xnorm > one) {
              beta = Max(WORK(j - 1), WORK(j));
              if (beta > bignum / xnorm) {
                X(1, 1) = X(1, 1) / xnorm;
                X(2, 1) = X(2, 1) / xnorm;
                scale = scale / xnorm;
              }
            }
            // Scale if necessary
            if (scale != one) Rscal(ki, scale, WORK.ptr(1 + iv * n), 1);
            WORK(j - 1 + iv * n) = X(1, 1);
            WORK(j + iv * n) = X(2, 1);
            // Update right-hand side
            Axpy(j - 2, -X(1, 1), Tm.ptr(1, j - 1), 1, WORK.ptr(1 + iv * n), 1);
            Axpy(j - 2, -X(2, 1), Tm.ptr(1, j), 1, WORK.ptr(1 + iv * n), 1);
          }
        }
        // Copy the vector x or Q*x to VR and normalize.
        if (!over) {
          // ------------------------------
          // no back-transform: copy x to VR and normalize.
          Copy(ki, WORK.ptr(1 + iv * n), 1, VR.ptr(1, is), 1);
          ii = Iamax(ki, VR.ptr(1, is), 1);
          remax = one / std::abs(VR(ii, is));
          Rscal(ki, remax, VR.ptr(1, is), 1);
          for (k = ki + 1; k <= n; ++k) {
            VR(k, is) = zero;
          }
        } else if (nb == 1) {
          // ------------------------------
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki > 1)
            Gemv('N', n, ki - 1, one, vr, ldvr, WORK.ptr(1 + iv * n), 1,
                 WORK(ki + iv * n), VR.ptr(1, ki), 1);
          ii = Iamax(n, VR.ptr(1, ki), 1);
          remax = one / std::abs(VR(ii, ki));
          Rscal(n, remax, VR.ptr(1, ki), 1);
        } else {
          // ------------------------------
          // version 2: back-transform block of vectors with GEMM
          // zero out below vector
          for (k = ki + 1; k <= n; ++k) {
            WORK(k + iv * n) = zero;
          }
          ISCOMPLEX(iv) = ip;
          // back-transform and normalization is done below
        }
      } else {
        // --------------------------------------------------------
        // Complex right eigenvector.
        //
        // Initial solve
        // [ ( T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I*WI) ]*X = 0.
        // [ ( T(KI,  KI-1) T(KI,  KI) )               ]
        if (std::abs(Tm(ki - 1, ki)) >= std::abs(Tm(ki, ki - 1))) {
          WORK(ki - 1 + (iv - 1) * n) = one;
          WORK(ki + iv * n) = wi / Tm(ki - 1, ki);
        } else {
          WORK(ki - 1 + (iv - 1) * n) = -wi / Tm(ki, ki - 1);
          WORK(ki + iv * n) = one;
        }
        WORK(ki + (iv - 1) * n) = zero;
        WORK(ki - 1 + iv * n) = zero;
        // Form right-hand side.
        for (k = 1; k <= ki - 2; ++k) {
          WORK(k + (iv - 1) * n) = -WORK(ki - 1 + (iv - 1) * n) * Tm(k, ki - 1);
          WORK(k + iv * n) = -WORK(ki + iv * n) * Tm(k, ki);
        }
        // Solve upper quasi-triangular system:
        // [ T(1:KI-2,1:KI-2) - (WR+i*WI) ]*X = SCALE*(WORK+i*WORK2)
        jnxt = ki - 2;
        for (j = ki - 2; j >= 1; --j) {
          if (j > jnxt) continue;
          j1 = j;
          j2 = j;
          jnxt = j - 1;
          if (j > 1) {
            if (Tm(j, j - 1) != zero) {
              j1 = j - 1;
              jnxt = j - 2;
            }
          }
          if (j1 == j2) {
            // 1-by-1 diagonal block
            Laln2(false, 1, 2, smin, one, Tm.ptr(j, j), ldt, one, one,
                  WORK.ptr(j + (iv - 1) * n), n, wr, wi, x, 2, &scale, &xnorm);
            // Scale X(1,1) and X(1,2) to avoid overflow when
            // updating the right-hand side.
            if (xnorm > one) {
              if (WORK(j) > bignum / xnorm) {
                X(1, 1) = X(1, 1) / xnorm;
                X(1, 2) = X(1, 2) / xnorm;
                scale = scale / xnorm;
              }
            }
            // Scale if necessary
            if (scale != one) {
              Rscal(ki, scale, WORK.ptr(1 + (iv - 1) * n), 1);
              Rscal(ki, scale, WORK.ptr(1 + iv * n), 1);
            }
            WORK(j + (iv - 1) * n) = X(1, 1);
            WORK(j + iv * n) = X(1, 2);
            // Update the right-hand side
            Axpy(j - 1, -X(1, 1), Tm.ptr(1, j), 1, WORK.ptr(1 + (iv - 1) * n),
                 1);
            Axpy(j - 1, -X(1, 2), Tm.ptr(1, j), 1, WORK.ptr(1 + iv * n), 1);
          } else {
            // 2-by-2 diagonal block
            Laln2(false, 2, 2, smin, one, Tm.ptr(j - 1, j - 1), ldt, one, one,
                  WORK.ptr(j - 1 + (iv - 1) * n), n, wr, wi, x, 2, &scale,
                  &xnorm);
            // Scale X to avoid overflow when updating
            // the right-hand side.
            if (xnorm > one) {
              beta = Max(WORK(j - 1), WORK(j));
              if (beta > bignum / xnorm) {
                rec = one / xnorm;
                X(1, 1) = X(1, 1) * rec;
                X(1, 2) = X(1, 2) * rec;
                X(2, 1) = X(2, 1) * rec;
                X(2, 2) = X(2, 2) * rec;
                scale = scale * rec;
              }
            }
            // Scale if necessary
            if (scale != one) {
              Rscal(ki, scale, WORK.ptr(1 + (iv - 1) * n), 1);
              Rscal(ki, scale, WORK.ptr(1 + iv * n), 1);
            }
            WORK(j - 1 + (iv - 1) * n) = X(1, 1);
            WORK(j + (iv - 1) * n) = X(2, 1);
            WORK(j - 1 + iv * n) = X(1, 2);
            WORK(j + iv * n) = X(2, 2);
            // Update the right-hand side
            Axpy(j - 2, -X(1, 1), Tm.ptr(1, j - 1), 1,
                 WORK.ptr(1 + (iv - 1) * n), 1);
            Axpy(j - 2, -X(2, 1), Tm.ptr(1, j), 1, WORK.ptr(1 + (iv - 1) * n),
                 1);
            Axpy(j - 2, -X(1, 2), Tm.ptr(1, j - 1), 1, WORK.ptr(1 + iv * n), 1);
            Axpy(j - 2, -X(2, 2), Tm.ptr(1, j), 1, WORK.ptr(1 + iv * n), 1);
          }
        }
        // Copy the vector x or Q*x to VR and normalize.
        if (!over) {
          // ------------------------------
          // no back-transform: copy x to VR and normalize.
          Copy(ki, WORK.ptr(1 + (iv - 1) * n), 1, VR.ptr(1, is - 1), 1);
          Copy(ki, WORK.ptr(1 + iv * n), 1, VR.ptr(1, is), 1);
          emax = zero;
          for (k = 1; k <= ki; ++k) {
            emax = Max(emax, std::abs(VR(k, is - 1)) + std::abs(VR(k, is)));
          }
          remax = one / emax;
          Rscal(ki, remax, VR.ptr(1, is - 1), 1);
          Rscal(ki, remax, VR.ptr(1, is), 1);
          for (k = ki + 1; k <= n; ++k) {
            VR(k, is - 1) = zero;
            VR(k, is) = zero;
          }
        } else if (nb == 1) {
          // ------------------------------
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki > 2) {
            Gemv('N', n, ki - 2, one, vr, ldvr, WORK.ptr(1 + (iv - 1) * n), 1,
                 WORK(ki - 1 + (iv - 1) * n), VR.ptr(1, ki - 1), 1);
            Gemv('N', n, ki - 2, one, vr, ldvr, WORK.ptr(1 + iv * n), 1,
                 WORK(ki + iv * n), VR.ptr(1, ki), 1);
          } else {
            Rscal(n, WORK(ki - 1 + (iv - 1) * n), VR.ptr(1, ki - 1), 1);
            Rscal(n, WORK(ki + iv * n), VR.ptr(1, ki), 1);
          }
          emax = zero;
          for (k = 1; k <= n; ++k) {
            emax = Max(emax, std::abs(VR(k, ki - 1)) + std::abs(VR(k, ki)));
          }
          remax = one / emax;
          Rscal(n, remax, VR.ptr(1, ki - 1), 1);
          Rscal(n, remax, VR.ptr(1, ki), 1);
        } else {
          // ------------------------------
          // version 2: back-transform block of vectors with GEMM
          // zero out below vector
          for (k = ki + 1; k <= n; ++k) {
            WORK(k + (iv - 1) * n) = zero;
            WORK(k + iv * n) = zero;
          }
          ISCOMPLEX(iv - 1) = -ip;
          ISCOMPLEX(iv) = ip;
          iv = iv - 1;
          // back-transform and normalization is done below
        }
      }
      if (nb > 1) {
        // --------------------------------------------------------
        // Blocked version of back-transform
        // For complex case, KI2 includes both vectors (KI-1 and KI)
        if (ip == 0) {
          ki2 = ki;
        } else {
          ki2 = ki - 1;
        }
        // Columns IV:NB of work are valid vectors.
        // When the number of vectors stored reaches NB-1 or NB,
        // or if this was last vector, do the GEMM
        if ((iv <= 2) || (ki2 == 1)) {
          Gemm('N', 'N', n, nb - iv + 1, ki2 + nb - iv, one, vr, ldvr,
               WORK.ptr(1 + iv * n), n, zero, WORK.ptr(1 + (nb + iv) * n), n);
          // normalize vectors
          for (k = iv; k <= nb; ++k) {
            if (ISCOMPLEX(k) == 0) {
              // real eigenvector
              ii = Iamax(n, WORK.ptr(1 + (nb + k) * n), 1);
              remax = one / std::abs(WORK(ii + (nb + k) * n));
            } else if (ISCOMPLEX(k) == 1) {
              // first eigenvector of conjugate pair
              emax = zero;
              for (ii = 1; ii <= n; ++ii) {
                emax = Max(emax, std::abs(WORK(ii + (nb + k) * n)) +
                                     std::abs(WORK(ii + (nb + k + 1) * n)));
              }
              remax = one / emax;
              // else if ISCOMPLEX(K).EQ.-1
              //    second eigenvector of conjugate pair
              //    reuse same REMAX as previous K
            }
            Rscal(n, remax, WORK.ptr(1 + (nb + k) * n), 1);
          }
          Lacpy('F', n, nb - iv + 1, WORK.ptr(1 + (nb + iv) * n), n,
                VR.ptr(1, ki2), ldvr);
          iv = nb;
        } else {
          iv = iv - 1;
        }
      }  // blocked back-transform
      is = is - 1;
      if (ip != 0) is = is - 1;
    }
  }
  if (leftv) {
    // ============================================================
    // Compute left eigenvectors.
    //
    // IV is index of column in current block.
    // For complex left vector, uses IV for real part and IV+1 for complex
    // part. Non-blocked version always uses IV=1;
    // blocked     version starts with IV=1, goes up to NB-1 or NB.
    // (Note the "0-th" column is used for 1-norms computed above.)
    iv = 1;
    ip = 0;
    is = 1;
    for (ki = 1; ki <= n; ++ki) {
      // Two branches set IP = 0, as in the Fortran.
      // NOLINTBEGIN(bugprone-branch-clone)
      if (ip == 1) {
        // previous iteration (ki-1) was first of conjugate pair,
        // so this ki is second of conjugate pair; skip to end of loop
        ip = -1;
        continue;
      } else if (ki == n) {
        // last column, so this ki must be real eigenvalue
        ip = 0;
      } else if (Tm(ki + 1, ki) == zero) {
        // zero on sub-diagonal, so this ki is real eigenvalue
        ip = 0;
      } else {
        // non-zero on sub-diagonal, so this ki is first of conjugate pair
        ip = 1;
      }
      // NOLINTEND(bugprone-branch-clone)
      if (somev) {
        if (!SELECT(ki)) continue;
      }
      // Compute the KI-th eigenvalue (WR,WI).
      wr = Tm(ki, ki);
      wi = zero;
      if (ip != 0)
        wi = std::sqrt(std::abs(Tm(ki, ki + 1))) *
             std::sqrt(std::abs(Tm(ki + 1, ki)));
      smin = Max(ulp * (std::abs(wr) + std::abs(wi)), smlnum);
      if (ip == 0) {
        // --------------------------------------------------------
        // Real left eigenvector
        WORK(ki + iv * n) = one;
        // Form right-hand side.
        for (k = ki + 1; k <= n; ++k) {
          WORK(k + iv * n) = -Tm(ki, k);
        }
        // Solve transposed quasi-triangular system:
        // [ T(KI+1:N,KI+1:N) - WR ]**T * X = SCALE*WORK
        vmax = one;
        vcrit = bignum;
        jnxt = ki + 1;
        for (j = ki + 1; j <= n; ++j) {
          if (j < jnxt) continue;
          j1 = j;
          j2 = j;
          jnxt = j + 1;
          if (j < n) {
            if (Tm(j + 1, j) != zero) {
              j2 = j + 1;
              jnxt = j + 2;
            }
          }
          if (j1 == j2) {
            // 1-by-1 diagonal block
            //
            // Scale if necessary to avoid overflow when forming
            // the right-hand side.
            if (WORK(j) > vcrit) {
              rec = one / vmax;
              Rscal(n - ki + 1, rec, WORK.ptr(ki + iv * n), 1);
              vmax = one;
              vcrit = bignum;
            }
            WORK(j + iv * n) =
                WORK(j + iv * n) - Dotc(j - ki - 1, Tm.ptr(ki + 1, j), 1,
                                        WORK.ptr(ki + 1 + iv * n), 1);
            // Solve [ T(J,J) - WR ]**T * X = WORK
            Laln2(false, 1, 1, smin, one, Tm.ptr(j, j), ldt, one, one,
                  WORK.ptr(j + iv * n), n, wr, zero, x, 2, &scale, &xnorm);
            // Scale if necessary
            if (scale != one)
              Rscal(n - ki + 1, scale, WORK.ptr(ki + iv * n), 1);
            WORK(j + iv * n) = X(1, 1);
            vmax = Max(std::abs(WORK(j + iv * n)), vmax);
            vcrit = bignum / vmax;
          } else {
            // 2-by-2 diagonal block
            //
            // Scale if necessary to avoid overflow when forming
            // the right-hand side.
            beta = Max(WORK(j), WORK(j + 1));
            if (beta > vcrit) {
              rec = one / vmax;
              Rscal(n - ki + 1, rec, WORK.ptr(ki + iv * n), 1);
              vmax = one;
              vcrit = bignum;
            }
            WORK(j + iv * n) =
                WORK(j + iv * n) - Dotc(j - ki - 1, Tm.ptr(ki + 1, j), 1,
                                        WORK.ptr(ki + 1 + iv * n), 1);
            WORK(j + 1 + iv * n) =
                WORK(j + 1 + iv * n) - Dotc(j - ki - 1, Tm.ptr(ki + 1, j + 1),
                                            1, WORK.ptr(ki + 1 + iv * n), 1);
            // Solve
            // [ T(J,J)-WR   T(J,J+1)      ]**T * X = SCALE*( WORK1 )
            // [ T(J+1,J)    T(J+1,J+1)-WR ]                ( WORK2 )
            Laln2(true, 2, 1, smin, one, Tm.ptr(j, j), ldt, one, one,
                  WORK.ptr(j + iv * n), n, wr, zero, x, 2, &scale, &xnorm);
            // Scale if necessary
            if (scale != one)
              Rscal(n - ki + 1, scale, WORK.ptr(ki + iv * n), 1);
            WORK(j + iv * n) = X(1, 1);
            WORK(j + 1 + iv * n) = X(2, 1);
            vmax = Max(std::abs(WORK(j + iv * n)),
                       std::abs(WORK(j + 1 + iv * n)), vmax);
            vcrit = bignum / vmax;
          }
        }
        // Copy the vector x or Q*x to VL and normalize.
        if (!over) {
          // ------------------------------
          // no back-transform: copy x to VL and normalize.
          Copy(n - ki + 1, WORK.ptr(ki + iv * n), 1, VL.ptr(ki, is), 1);
          ii = Iamax(n - ki + 1, VL.ptr(ki, is), 1) + ki - 1;
          remax = one / std::abs(VL(ii, is));
          Rscal(n - ki + 1, remax, VL.ptr(ki, is), 1);
          for (k = 1; k <= ki - 1; ++k) {
            VL(k, is) = zero;
          }
        } else if (nb == 1) {
          // ------------------------------
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki < n)
            Gemv('N', n, n - ki, one, VL.ptr(1, ki + 1), ldvl,
                 WORK.ptr(ki + 1 + iv * n), 1, WORK(ki + iv * n), VL.ptr(1, ki),
                 1);
          ii = Iamax(n, VL.ptr(1, ki), 1);
          remax = one / std::abs(VL(ii, ki));
          Rscal(n, remax, VL.ptr(1, ki), 1);
        } else {
          // ------------------------------
          // version 2: back-transform block of vectors with GEMM
          // zero out above vector
          // could go from KI-NV+1 to KI-1
          for (k = 1; k <= ki - 1; ++k) {
            WORK(k + iv * n) = zero;
          }
          ISCOMPLEX(iv) = ip;
          // back-transform and normalization is done below
        }
      } else {
        // --------------------------------------------------------
        // Complex left eigenvector.
        //
        // Initial solve:
        // [ ( T(KI,KI)    T(KI,KI+1)  )**T - (WR - I* WI) ]*X = 0.
        // [ ( T(KI+1,KI) T(KI+1,KI+1) )                   ]
        if (std::abs(Tm(ki, ki + 1)) >= std::abs(Tm(ki + 1, ki))) {
          WORK(ki + iv * n) = wi / Tm(ki, ki + 1);
          WORK(ki + 1 + (iv + 1) * n) = one;
        } else {
          WORK(ki + iv * n) = one;
          WORK(ki + 1 + (iv + 1) * n) = -wi / Tm(ki + 1, ki);
        }
        WORK(ki + 1 + iv * n) = zero;
        WORK(ki + (iv + 1) * n) = zero;
        // Form right-hand side.
        for (k = ki + 2; k <= n; ++k) {
          WORK(k + iv * n) = -WORK(ki + iv * n) * Tm(ki, k);
          WORK(k + (iv + 1) * n) = -WORK(ki + 1 + (iv + 1) * n) * Tm(ki + 1, k);
        }
        // Solve transposed quasi-triangular system:
        // [ T(KI+2:N,KI+2:N)**T - (WR-i*WI) ]*X = WORK1+i*WORK2
        vmax = one;
        vcrit = bignum;
        jnxt = ki + 2;
        for (j = ki + 2; j <= n; ++j) {
          if (j < jnxt) continue;
          j1 = j;
          j2 = j;
          jnxt = j + 1;
          if (j < n) {
            if (Tm(j + 1, j) != zero) {
              j2 = j + 1;
              jnxt = j + 2;
            }
          }
          if (j1 == j2) {
            // 1-by-1 diagonal block
            //
            // Scale if necessary to avoid overflow when
            // forming the right-hand side elements.
            if (WORK(j) > vcrit) {
              rec = one / vmax;
              Rscal(n - ki + 1, rec, WORK.ptr(ki + iv * n), 1);
              Rscal(n - ki + 1, rec, WORK.ptr(ki + (iv + 1) * n), 1);
              vmax = one;
              vcrit = bignum;
            }
            WORK(j + iv * n) =
                WORK(j + iv * n) - Dotc(j - ki - 2, Tm.ptr(ki + 2, j), 1,
                                        WORK.ptr(ki + 2 + iv * n), 1);
            WORK(j + (iv + 1) * n) = WORK(j + (iv + 1) * n) -
                                     Dotc(j - ki - 2, Tm.ptr(ki + 2, j), 1,
                                          WORK.ptr(ki + 2 + (iv + 1) * n), 1);
            // Solve [ T(J,J)-(WR-i*WI) ]*(X11+i*X12)= WK+I*WK2
            Laln2(false, 1, 2, smin, one, Tm.ptr(j, j), ldt, one, one,
                  WORK.ptr(j + iv * n), n, wr, -wi, x, 2, &scale, &xnorm);
            // Scale if necessary
            if (scale != one) {
              Rscal(n - ki + 1, scale, WORK.ptr(ki + iv * n), 1);
              Rscal(n - ki + 1, scale, WORK.ptr(ki + (iv + 1) * n), 1);
            }
            WORK(j + iv * n) = X(1, 1);
            WORK(j + (iv + 1) * n) = X(1, 2);
            vmax = Max(std::abs(WORK(j + iv * n)),
                       std::abs(WORK(j + (iv + 1) * n)), vmax);
            vcrit = bignum / vmax;
          } else {
            // 2-by-2 diagonal block
            //
            // Scale if necessary to avoid overflow when forming
            // the right-hand side elements.
            beta = Max(WORK(j), WORK(j + 1));
            if (beta > vcrit) {
              rec = one / vmax;
              Rscal(n - ki + 1, rec, WORK.ptr(ki + iv * n), 1);
              Rscal(n - ki + 1, rec, WORK.ptr(ki + (iv + 1) * n), 1);
              vmax = one;
              vcrit = bignum;
            }
            WORK(j + iv * n) =
                WORK(j + iv * n) - Dotc(j - ki - 2, Tm.ptr(ki + 2, j), 1,
                                        WORK.ptr(ki + 2 + iv * n), 1);
            WORK(j + (iv + 1) * n) = WORK(j + (iv + 1) * n) -
                                     Dotc(j - ki - 2, Tm.ptr(ki + 2, j), 1,
                                          WORK.ptr(ki + 2 + (iv + 1) * n), 1);
            WORK(j + 1 + iv * n) =
                WORK(j + 1 + iv * n) - Dotc(j - ki - 2, Tm.ptr(ki + 2, j + 1),
                                            1, WORK.ptr(ki + 2 + iv * n), 1);
            WORK(j + 1 + (iv + 1) * n) =
                WORK(j + 1 + (iv + 1) * n) -
                Dotc(j - ki - 2, Tm.ptr(ki + 2, j + 1), 1,
                     WORK.ptr(ki + 2 + (iv + 1) * n), 1);
            // Solve 2-by-2 complex linear equation
            // [ (T(j,j)   T(j,j+1)  )**T - (wr-i*wi)*I ]*X = SCALE*B
            // [ (T(j+1,j) T(j+1,j+1))                  ]
            Laln2(true, 2, 2, smin, one, Tm.ptr(j, j), ldt, one, one,
                  WORK.ptr(j + iv * n), n, wr, -wi, x, 2, &scale, &xnorm);
            // Scale if necessary
            if (scale != one) {
              Rscal(n - ki + 1, scale, WORK.ptr(ki + iv * n), 1);
              Rscal(n - ki + 1, scale, WORK.ptr(ki + (iv + 1) * n), 1);
            }
            WORK(j + iv * n) = X(1, 1);
            WORK(j + (iv + 1) * n) = X(1, 2);
            WORK(j + 1 + iv * n) = X(2, 1);
            WORK(j + 1 + (iv + 1) * n) = X(2, 2);
            vmax = Max(std::abs(X(1, 1)), std::abs(X(1, 2)), std::abs(X(2, 1)),
                       std::abs(X(2, 2)), vmax);
            vcrit = bignum / vmax;
          }
        }
        // Copy the vector x or Q*x to VL and normalize.
        if (!over) {
          // ------------------------------
          // no back-transform: copy x to VL and normalize.
          Copy(n - ki + 1, WORK.ptr(ki + iv * n), 1, VL.ptr(ki, is), 1);
          Copy(n - ki + 1, WORK.ptr(ki + (iv + 1) * n), 1, VL.ptr(ki, is + 1),
               1);
          emax = zero;
          for (k = ki; k <= n; ++k) {
            emax = Max(emax, std::abs(VL(k, is)) + std::abs(VL(k, is + 1)));
          }
          remax = one / emax;
          Rscal(n - ki + 1, remax, VL.ptr(ki, is), 1);
          Rscal(n - ki + 1, remax, VL.ptr(ki, is + 1), 1);
          for (k = 1; k <= ki - 1; ++k) {
            VL(k, is) = zero;
            VL(k, is + 1) = zero;
          }
        } else if (nb == 1) {
          // ------------------------------
          // version 1: back-transform each vector with GEMV, Q*x.
          if (ki < n - 1) {
            Gemv('N', n, n - ki - 1, one, VL.ptr(1, ki + 2), ldvl,
                 WORK.ptr(ki + 2 + iv * n), 1, WORK(ki + iv * n), VL.ptr(1, ki),
                 1);
            Gemv('N', n, n - ki - 1, one, VL.ptr(1, ki + 2), ldvl,
                 WORK.ptr(ki + 2 + (iv + 1) * n), 1,
                 WORK(ki + 1 + (iv + 1) * n), VL.ptr(1, ki + 1), 1);
          } else {
            Rscal(n, WORK(ki + iv * n), VL.ptr(1, ki), 1);
            Rscal(n, WORK(ki + 1 + (iv + 1) * n), VL.ptr(1, ki + 1), 1);
          }
          emax = zero;
          for (k = 1; k <= n; ++k) {
            emax = Max(emax, std::abs(VL(k, ki)) + std::abs(VL(k, ki + 1)));
          }
          remax = one / emax;
          Rscal(n, remax, VL.ptr(1, ki), 1);
          Rscal(n, remax, VL.ptr(1, ki + 1), 1);
        } else {
          // ------------------------------
          // version 2: back-transform block of vectors with GEMM
          // zero out above vector
          // could go from KI-NV+1 to KI-1
          for (k = 1; k <= ki - 1; ++k) {
            WORK(k + iv * n) = zero;
            WORK(k + (iv + 1) * n) = zero;
          }
          ISCOMPLEX(iv) = ip;
          ISCOMPLEX(iv + 1) = -ip;
          iv = iv + 1;
          // back-transform and normalization is done below
        }
      }
      if (nb > 1) {
        // --------------------------------------------------------
        // Blocked version of back-transform
        // For complex case, KI2 includes both vectors (KI and KI+1)
        if (ip == 0) {
          ki2 = ki;
        } else {
          ki2 = ki + 1;
        }
        // Columns 1:IV of work are valid vectors.
        // When the number of vectors stored reaches NB-1 or NB,
        // or if this was last vector, do the GEMM
        if ((iv >= nb - 1) || (ki2 == n)) {
          Gemm('N', 'N', n, iv, n - ki2 + iv, one, VL.ptr(1, ki2 - iv + 1),
               ldvl, WORK.ptr(ki2 - iv + 1 + 1 * n), n, zero,
               WORK.ptr(1 + (nb + 1) * n), n);
          // normalize vectors
          for (k = 1; k <= iv; ++k) {
            if (ISCOMPLEX(k) == 0) {
              // real eigenvector
              ii = Iamax(n, WORK.ptr(1 + (nb + k) * n), 1);
              remax = one / std::abs(WORK(ii + (nb + k) * n));
            } else if (ISCOMPLEX(k) == 1) {
              // first eigenvector of conjugate pair
              emax = zero;
              for (ii = 1; ii <= n; ++ii) {
                emax = Max(emax, std::abs(WORK(ii + (nb + k) * n)) +
                                     std::abs(WORK(ii + (nb + k + 1) * n)));
              }
              remax = one / emax;
              // else if ISCOMPLEX(K).EQ.-1
              //    second eigenvector of conjugate pair
              //    reuse same REMAX as previous K
            }
            Rscal(n, remax, WORK.ptr(1 + (nb + k) * n), 1);
          }
          Lacpy('F', n, iv, WORK.ptr(1 + (nb + 1) * n), n,
                VL.ptr(1, ki2 - iv + 1), ldvl);
          iv = 1;
        } else {
          iv = iv + 1;
        }
      }  // blocked back-transform
      is = is + 1;
      if (ip != 0) is = is + 1;
    }
  }
  return info;
}

// xTREVC3 (complex T): as the real routine, for the upper triangular T (in
// which the diagonal is changed and restored; SELECT is input only, one
// column per eigenvector). WORK has LWORK >= max(1, 2*N) elements, RWORK
// LRWORK >= max(1, N) (LWORK = -1 or LRWORK = -1: workspace query).
template <typename T>
blasint Trevc3(char side, char howmny, const blasint* select, blasint n, T* t,
               blasint ldt, T* vl, blasint ldvl, T* vr, blasint ldvr,
               blasint mm, blasint* m, T* work, blasint lwork, RealT<T>* rwork,
               blasint lrwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const blasint nbmin = 8;
  const blasint nbmax = 128;
  const Vector<R> RWORK(rwork);
  const Vector<const blasint> SELECT(select);
  const Matrix<T> Tm(t, ldt);
  const Matrix<T> VL(vl, ldvl);
  const Matrix<T> VR(vr, ldvr);
  const Vector<T> WORK(work);
  blasint i, ii, is, iv, j, k, ki, maxwrk, nb;
  bool allv, bothv, leftv, lquery, over, rightv, somev;
  R remax, scale, smin, smlnum, ulp, unfl;
  // Decode and test the input parameters
  bothv = Lsame(side, 'B');
  rightv = Lsame(side, 'R') || bothv;
  leftv = Lsame(side, 'L') || bothv;
  allv = Lsame(howmny, 'A');
  over = Lsame(howmny, 'B');
  somev = Lsame(howmny, 'S');
  // Set M to the number of columns required to store the selected
  // eigenvectors.
  if (somev) {
    *m = 0;
    for (j = 1; j <= n; ++j) {
      if (SELECT(j)) *m = *m + 1;
    }
  } else {
    *m = n;
  }
  blasint info = 0;
  const char opts[3] = {side, howmny, '\0'};
  nb = Ilaenv<T>(1, "TREVC", opts, n, -1, -1, -1);
  maxwrk = Max<blasint>(1, n + 2 * n * nb);
  WORK(1) = WorkSize<T>(maxwrk);
  RWORK(1) = static_cast<R>(Max<blasint>(1, n));
  lquery = (lwork == -1 || lrwork == -1);
  if (!rightv && !leftv) {
    info = -1;
  } else if (!allv && !over && !somev) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldvl < 1 || (leftv && ldvl < n)) {
    info = -8;
  } else if (ldvr < 1 || (rightv && ldvr < n)) {
    info = -10;
  } else if (mm < *m) {
    info = -11;
  } else if (lwork < Max<blasint>(1, 2 * n) && !lquery) {
    info = -14;
  } else if (lrwork < Max<blasint>(1, n) && !lquery) {
    info = -16;
  }
  if (info != 0) {
    Xerbla<T>("TREVC3", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible.
  if (n == 0) return info;
  // Use blocked version of back-transformation if sufficient workspace.
  // Zero-out the workspace to avoid potential NaN propagation.
  if (over && lwork >= n + 2 * n * nbmin) {
    nb = (lwork - n) / (2 * n);
    nb = Min<blasint>(nb, nbmax);
    Laset('F', n, 1 + 2 * nb, czero, czero, work, n);
  } else {
    nb = 1;
  }
  // Set the constants to control overflow.
  // (OVFL = 1/UNFL is computed by the Fortran routine, but not used.)
  unfl = Lamch<R>('S');
  ulp = Lamch<R>('P');
  smlnum = unfl * (n / ulp);
  // Store the diagonal elements of T in working array WORK.
  for (i = 1; i <= n; ++i) {
    WORK(i) = Tm(i, i);
  }
  // Compute 1-norm of each column of strictly upper triangular
  // part of T to control overflow in triangular solver.
  RWORK(1) = zero;
  for (j = 2; j <= n; ++j) {
    RWORK(j) = Asum(j - 1, Tm.ptr(1, j), 1);
  }
  if (rightv) {
    // ============================================================
    // Compute right eigenvectors.
    //
    // IV is index of column in current block.
    // Non-blocked version always uses IV=NB=1;
    // blocked     version starts with IV=NB, goes down to 1.
    // (Note the "0-th" column is used to store the original diagonal.)
    iv = nb;
    is = *m;
    for (ki = n; ki >= 1; --ki) {
      if (somev) {
        if (!SELECT(ki)) continue;
      }
      smin = Max(ulp * (Cabs1(Tm(ki, ki))), smlnum);
      // --------------------------------------------------------
      // Complex right eigenvector
      WORK(ki + iv * n) = cone;
      // Form right-hand side.
      for (k = 1; k <= ki - 1; ++k) {
        WORK(k + iv * n) = -Tm(k, ki);
      }
      // Solve upper triangular system:
      // [ T(1:KI-1,1:KI-1) - T(KI,KI) ]*X = SCALE*WORK.
      for (k = 1; k <= ki - 1; ++k) {
        Tm(k, k) = Tm(k, k) - Tm(ki, ki);
        if (Cabs1(Tm(k, k)) < smin) Tm(k, k) = smin;
      }
      if (ki > 1) {
        info = Latrs('U', 'N', 'N', 'Y', ki - 1, t, ldt, WORK.ptr(1 + iv * n),
                     &scale, rwork);
        WORK(ki + iv * n) = scale;
      }
      // Copy the vector x or Q*x to VR and normalize.
      if (!over) {
        // ------------------------------
        // no back-transform: copy x to VR and normalize.
        Copy(ki, WORK.ptr(1 + iv * n), 1, VR.ptr(1, is), 1);
        ii = Iamax(ki, VR.ptr(1, is), 1);
        remax = one / Cabs1(VR(ii, is));
        Rscal(ki, remax, VR.ptr(1, is), 1);
        for (k = ki + 1; k <= n; ++k) {
          VR(k, is) = czero;
        }
      } else if (nb == 1) {
        // ------------------------------
        // version 1: back-transform each vector with GEMV, Q*x.
        if (ki > 1)
          Gemv('N', n, ki - 1, cone, vr, ldvr, WORK.ptr(1 + iv * n), 1,
               T(scale), VR.ptr(1, ki), 1);
        ii = Iamax(n, VR.ptr(1, ki), 1);
        remax = one / Cabs1(VR(ii, ki));
        Rscal(n, remax, VR.ptr(1, ki), 1);
      } else {
        // ------------------------------
        // version 2: back-transform block of vectors with GEMM
        // zero out below vector
        for (k = ki + 1; k <= n; ++k) {
          WORK(k + iv * n) = czero;
        }
        // Columns IV:NB of work are valid vectors.
        // When the number of vectors stored reaches NB,
        // or if this was last vector, do the GEMM
        if ((iv == 1) || (ki == 1)) {
          Gemm('N', 'N', n, nb - iv + 1, ki + nb - iv, cone, vr, ldvr,
               WORK.ptr(1 + iv * n), n, czero, WORK.ptr(1 + (nb + iv) * n), n);
          // normalize vectors
          for (k = iv; k <= nb; ++k) {
            ii = Iamax(n, WORK.ptr(1 + (nb + k) * n), 1);
            remax = one / Cabs1(WORK(ii + (nb + k) * n));
            Rscal(n, remax, WORK.ptr(1 + (nb + k) * n), 1);
          }
          Lacpy('F', n, nb - iv + 1, WORK.ptr(1 + (nb + iv) * n), n,
                VR.ptr(1, ki), ldvr);
          iv = nb;
        } else {
          iv = iv - 1;
        }
      }  // blocked back-transform
      // Restore the original diagonal elements of T.
      for (k = 1; k <= ki - 1; ++k) {
        Tm(k, k) = WORK(k);
      }
      is = is - 1;
    }
  }
  if (leftv) {
    // ============================================================
    // Compute left eigenvectors.
    //
    // IV is index of column in current block.
    // Non-blocked version always uses IV=1;
    // blocked     version starts with IV=1, goes up to NB.
    // (Note the "0-th" column is used to store the original diagonal.)
    iv = 1;
    is = 1;
    for (ki = 1; ki <= n; ++ki) {
      if (somev) {
        if (!SELECT(ki)) continue;
      }
      smin = Max(ulp * (Cabs1(Tm(ki, ki))), smlnum);
      // --------------------------------------------------------
      // Complex left eigenvector
      WORK(ki + iv * n) = cone;
      // Form right-hand side.
      for (k = ki + 1; k <= n; ++k) {
        WORK(k + iv * n) = -Conj(Tm(ki, k));
      }
      // Solve conjugate-transposed triangular system:
      // [ T(KI+1:N,KI+1:N) - T(KI,KI) ]**H * X = SCALE*WORK.
      for (k = ki + 1; k <= n; ++k) {
        Tm(k, k) = Tm(k, k) - Tm(ki, ki);
        if (Cabs1(Tm(k, k)) < smin) Tm(k, k) = smin;
      }
      if (ki < n) {
        info = Latrs('U', 'C', 'N', 'Y', n - ki, Tm.ptr(ki + 1, ki + 1), ldt,
                     WORK.ptr(ki + 1 + iv * n), &scale, rwork);
        WORK(ki + iv * n) = scale;
      }
      // Copy the vector x or Q*x to VL and normalize.
      if (!over) {
        // ------------------------------
        // no back-transform: copy x to VL and normalize.
        Copy(n - ki + 1, WORK.ptr(ki + iv * n), 1, VL.ptr(ki, is), 1);
        ii = Iamax(n - ki + 1, VL.ptr(ki, is), 1) + ki - 1;
        remax = one / Cabs1(VL(ii, is));
        Rscal(n - ki + 1, remax, VL.ptr(ki, is), 1);
        for (k = 1; k <= ki - 1; ++k) {
          VL(k, is) = czero;
        }
      } else if (nb == 1) {
        // ------------------------------
        // version 1: back-transform each vector with GEMV, Q*x.
        if (ki < n)
          Gemv('N', n, n - ki, cone, VL.ptr(1, ki + 1), ldvl,
               WORK.ptr(ki + 1 + iv * n), 1, T(scale), VL.ptr(1, ki), 1);
        ii = Iamax(n, VL.ptr(1, ki), 1);
        remax = one / Cabs1(VL(ii, ki));
        Rscal(n, remax, VL.ptr(1, ki), 1);
      } else {
        // ------------------------------
        // version 2: back-transform block of vectors with GEMM
        // zero out above vector
        // could go from KI-NV+1 to KI-1
        for (k = 1; k <= ki - 1; ++k) {
          WORK(k + iv * n) = czero;
        }
        // Columns 1:IV of work are valid vectors.
        // When the number of vectors stored reaches NB,
        // or if this was last vector, do the GEMM
        if ((iv == nb) || (ki == n)) {
          Gemm('N', 'N', n, iv, n - ki + iv, cone, VL.ptr(1, ki - iv + 1), ldvl,
               WORK.ptr(ki - iv + 1 + 1 * n), n, czero,
               WORK.ptr(1 + (nb + 1) * n), n);
          // normalize vectors
          for (k = 1; k <= iv; ++k) {
            ii = Iamax(n, WORK.ptr(1 + (nb + k) * n), 1);
            remax = one / Cabs1(WORK(ii + (nb + k) * n));
            Rscal(n, remax, WORK.ptr(1 + (nb + k) * n), 1);
          }
          Lacpy('F', n, iv, WORK.ptr(1 + (nb + 1) * n), n,
                VL.ptr(1, ki - iv + 1), ldvl);
          iv = 1;
        } else {
          iv = iv + 1;
        }
      }  // blocked back-transform
      // Restore the original diagonal elements of T.
      for (k = ki + 1; k <= n; ++k) {
        Tm(k, k) = WORK(k);
      }
      is = is + 1;
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template blasint Trevc3<R>(char, char, blasint*, blasint, const R*, blasint, \
                             R*, blasint, R*, blasint, blasint, blasint*, R*,  \
                             blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                  \
  template blasint Trevc3<T>(char, char, const blasint*, blasint, T*, blasint, \
                             T*, blasint, T*, blasint, blasint, blasint*, T*,  \
                             blasint, RealT<T>*, blasint);
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
#define HWYBLAS_LAPACK_TREVC_REAL(prefix, R)                                 \
  HWYBLAS_API void prefix##trevc3_(                                          \
      const char* side, const char* howmny, hwyblas_int* select,             \
      const hwyblas_int* n, const R* t, const hwyblas_int* ldt, R* vl,       \
      const hwyblas_int* ldvl, R* vr, const hwyblas_int* ldvr,               \
      const hwyblas_int* mm, hwyblas_int* m, R* work,                        \
      const hwyblas_int* lwork, hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Trevc3(*side, *howmny, select, *n, t, *ldt, vl, \
                                    *ldvl, vr, *ldvr, *mm, m, work, *lwork); \
  }

HWYBLAS_LAPACK_TREVC_REAL(s, float)
HWYBLAS_LAPACK_TREVC_REAL(d, double)

#undef HWYBLAS_LAPACK_TREVC_REAL

#define HWYBLAS_LAPACK_TREVC_COMPLEX(prefix, T, R)                      \
  HWYBLAS_API void prefix##trevc3_(                                     \
      const char* side, const char* howmny, const hwyblas_int* select,  \
      const hwyblas_int* n, R* t, const hwyblas_int* ldt, R* vl,        \
      const hwyblas_int* ldvl, R* vr, const hwyblas_int* ldvr,          \
      const hwyblas_int* mm, hwyblas_int* m, R* work,                   \
      const hwyblas_int* lwork, R* rwork, const hwyblas_int* lrwork,    \
      hwyblas_int* info) {                                              \
    *info = hwyblas::lapack::Trevc3(                                    \
        *side, *howmny, select, *n, As<T>(t), *ldt, As<T>(vl), *ldvl,   \
        As<T>(vr), *ldvr, *mm, m, As<T>(work), *lwork, rwork, *lrwork); \
  }

HWYBLAS_LAPACK_TREVC_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_TREVC_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_TREVC_COMPLEX

}  // extern "C"
