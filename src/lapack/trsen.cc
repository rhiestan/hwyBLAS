// Reordering of the Schur factorization and the condition numbers of a
// cluster of eigenvalues: xTRSEN (reorders a selected cluster to the top
// and estimates its condition numbers), xTREXC (moves one diagonal block),
// xLAEXC (swaps two adjacent blocks of a real Schur form), xTRSYL (the
// Sylvester equation of two quasi-triangular matrices), xLASY2 (a small
// Sylvester equation), xLALN2 (a small scaled system with a perturbed
// diagonal), xLACN2 (the 1-norm estimator by reverse communication) and
// xLARFX (an elementary reflector applied with unrolled code for orders up
// to 10).
//
// The real and the complex routines differ throughout (2-by-2 blocks of
// the real Schur form), so they are ported as separate functions with the
// same name where their arguments differ, and as TrsylReal / TrsylComplex
// behind Trsyl otherwise. xLARFX is one template for both (the complex
// routine conjugates, which changes nothing for real T).
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

// xLARFX: applies the elementary reflector H = I - tau * v * v**H to the
// M-by-N matrix C from the left (SIDE = 'L', H * C; V has M elements) or
// the right (SIDE = 'R', C * H; V has N elements), with unrolled code if
// the order of H is at most 10 and xLARF otherwise. WORK has N (SIDE =
// 'L') or M (SIDE = 'R') elements and is not used for orders <= 10.
template <typename T>
void Larfx(char side, blasint m, blasint n, const T* v, T tau, T* c,
           blasint ldc, T* work) {
  const T zero = T(0);
  const T one = T(1);
  const Matrix<T> C(c, ldc);
  const Vector<const T> V(v);
  const Vector<T> WORK(work);
  blasint j;
  T sum, t1, t10, t2, t3, t4, t5, t6, t7, t8, t9, v1, v10, v2, v3, v4, v5, v6,
      v7, v8, v9;
  if (tau == zero) return;
  if (Lsame(side, 'L')) {
    // Form  H * C, where H has order m.
    switch (m) {
      case 1:
        goto L10;
      case 2:
        goto L30;
      case 3:
        goto L50;
      case 4:
        goto L70;
      case 5:
        goto L90;
      case 6:
        goto L110;
      case 7:
        goto L130;
      case 8:
        goto L150;
      case 9:
        goto L170;
      case 10:
        goto L190;
      default:  // Out of range: no jump.
        break;
    }
    // Code for general M
    Larf(side, m, n, v, 1, tau, c, ldc, work);
    goto L410;
  L10:
    // Special code for 1 x 1 Householder
    t1 = one - tau * V(1) * Conj(V(1));
    for (j = 1; j <= n; ++j) {
      C(1, j) = t1 * C(1, j);
    }
    goto L410;
  L30:
    // Special code for 2 x 2 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
    }
    goto L410;
  L50:
    // Special code for 3 x 3 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
    }
    goto L410;
  L70:
    // Special code for 4 x 4 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
    }
    goto L410;
  L90:
    // Special code for 5 x 5 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    v5 = Conj(V(5));
    t5 = tau * Conj(v5);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j) +
            v5 * C(5, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
      C(5, j) = C(5, j) - sum * t5;
    }
    goto L410;
  L110:
    // Special code for 6 x 6 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    v5 = Conj(V(5));
    t5 = tau * Conj(v5);
    v6 = Conj(V(6));
    t6 = tau * Conj(v6);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j) +
            v5 * C(5, j) + v6 * C(6, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
      C(5, j) = C(5, j) - sum * t5;
      C(6, j) = C(6, j) - sum * t6;
    }
    goto L410;
  L130:
    // Special code for 7 x 7 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    v5 = Conj(V(5));
    t5 = tau * Conj(v5);
    v6 = Conj(V(6));
    t6 = tau * Conj(v6);
    v7 = Conj(V(7));
    t7 = tau * Conj(v7);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j) +
            v5 * C(5, j) + v6 * C(6, j) + v7 * C(7, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
      C(5, j) = C(5, j) - sum * t5;
      C(6, j) = C(6, j) - sum * t6;
      C(7, j) = C(7, j) - sum * t7;
    }
    goto L410;
  L150:
    // Special code for 8 x 8 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    v5 = Conj(V(5));
    t5 = tau * Conj(v5);
    v6 = Conj(V(6));
    t6 = tau * Conj(v6);
    v7 = Conj(V(7));
    t7 = tau * Conj(v7);
    v8 = Conj(V(8));
    t8 = tau * Conj(v8);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j) +
            v5 * C(5, j) + v6 * C(6, j) + v7 * C(7, j) + v8 * C(8, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
      C(5, j) = C(5, j) - sum * t5;
      C(6, j) = C(6, j) - sum * t6;
      C(7, j) = C(7, j) - sum * t7;
      C(8, j) = C(8, j) - sum * t8;
    }
    goto L410;
  L170:
    // Special code for 9 x 9 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    v5 = Conj(V(5));
    t5 = tau * Conj(v5);
    v6 = Conj(V(6));
    t6 = tau * Conj(v6);
    v7 = Conj(V(7));
    t7 = tau * Conj(v7);
    v8 = Conj(V(8));
    t8 = tau * Conj(v8);
    v9 = Conj(V(9));
    t9 = tau * Conj(v9);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j) +
            v5 * C(5, j) + v6 * C(6, j) + v7 * C(7, j) + v8 * C(8, j) +
            v9 * C(9, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
      C(5, j) = C(5, j) - sum * t5;
      C(6, j) = C(6, j) - sum * t6;
      C(7, j) = C(7, j) - sum * t7;
      C(8, j) = C(8, j) - sum * t8;
      C(9, j) = C(9, j) - sum * t9;
    }
    goto L410;
  L190:
    // Special code for 10 x 10 Householder
    v1 = Conj(V(1));
    t1 = tau * Conj(v1);
    v2 = Conj(V(2));
    t2 = tau * Conj(v2);
    v3 = Conj(V(3));
    t3 = tau * Conj(v3);
    v4 = Conj(V(4));
    t4 = tau * Conj(v4);
    v5 = Conj(V(5));
    t5 = tau * Conj(v5);
    v6 = Conj(V(6));
    t6 = tau * Conj(v6);
    v7 = Conj(V(7));
    t7 = tau * Conj(v7);
    v8 = Conj(V(8));
    t8 = tau * Conj(v8);
    v9 = Conj(V(9));
    t9 = tau * Conj(v9);
    v10 = Conj(V(10));
    t10 = tau * Conj(v10);
    for (j = 1; j <= n; ++j) {
      sum = v1 * C(1, j) + v2 * C(2, j) + v3 * C(3, j) + v4 * C(4, j) +
            v5 * C(5, j) + v6 * C(6, j) + v7 * C(7, j) + v8 * C(8, j) +
            v9 * C(9, j) + v10 * C(10, j);
      C(1, j) = C(1, j) - sum * t1;
      C(2, j) = C(2, j) - sum * t2;
      C(3, j) = C(3, j) - sum * t3;
      C(4, j) = C(4, j) - sum * t4;
      C(5, j) = C(5, j) - sum * t5;
      C(6, j) = C(6, j) - sum * t6;
      C(7, j) = C(7, j) - sum * t7;
      C(8, j) = C(8, j) - sum * t8;
      C(9, j) = C(9, j) - sum * t9;
      C(10, j) = C(10, j) - sum * t10;
    }
    goto L410;
  } else {
    // Form  C * H, where H has order n.
    switch (n) {
      case 1:
        goto L210;
      case 2:
        goto L230;
      case 3:
        goto L250;
      case 4:
        goto L270;
      case 5:
        goto L290;
      case 6:
        goto L310;
      case 7:
        goto L330;
      case 8:
        goto L350;
      case 9:
        goto L370;
      case 10:
        goto L390;
      default:  // Out of range: no jump.
        break;
    }
    // Code for general N
    Larf(side, m, n, v, 1, tau, c, ldc, work);
    goto L410;
  L210:
    // Special code for 1 x 1 Householder
    t1 = one - tau * V(1) * Conj(V(1));
    for (j = 1; j <= m; ++j) {
      C(j, 1) = t1 * C(j, 1);
    }
    goto L410;
  L230:
    // Special code for 2 x 2 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
    }
    goto L410;
  L250:
    // Special code for 3 x 3 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
    }
    goto L410;
  L270:
    // Special code for 4 x 4 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
    }
    goto L410;
  L290:
    // Special code for 5 x 5 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    v5 = V(5);
    t5 = tau * Conj(v5);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4) +
            v5 * C(j, 5);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
      C(j, 5) = C(j, 5) - sum * t5;
    }
    goto L410;
  L310:
    // Special code for 6 x 6 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    v5 = V(5);
    t5 = tau * Conj(v5);
    v6 = V(6);
    t6 = tau * Conj(v6);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4) +
            v5 * C(j, 5) + v6 * C(j, 6);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
      C(j, 5) = C(j, 5) - sum * t5;
      C(j, 6) = C(j, 6) - sum * t6;
    }
    goto L410;
  L330:
    // Special code for 7 x 7 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    v5 = V(5);
    t5 = tau * Conj(v5);
    v6 = V(6);
    t6 = tau * Conj(v6);
    v7 = V(7);
    t7 = tau * Conj(v7);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4) +
            v5 * C(j, 5) + v6 * C(j, 6) + v7 * C(j, 7);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
      C(j, 5) = C(j, 5) - sum * t5;
      C(j, 6) = C(j, 6) - sum * t6;
      C(j, 7) = C(j, 7) - sum * t7;
    }
    goto L410;
  L350:
    // Special code for 8 x 8 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    v5 = V(5);
    t5 = tau * Conj(v5);
    v6 = V(6);
    t6 = tau * Conj(v6);
    v7 = V(7);
    t7 = tau * Conj(v7);
    v8 = V(8);
    t8 = tau * Conj(v8);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4) +
            v5 * C(j, 5) + v6 * C(j, 6) + v7 * C(j, 7) + v8 * C(j, 8);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
      C(j, 5) = C(j, 5) - sum * t5;
      C(j, 6) = C(j, 6) - sum * t6;
      C(j, 7) = C(j, 7) - sum * t7;
      C(j, 8) = C(j, 8) - sum * t8;
    }
    goto L410;
  L370:
    // Special code for 9 x 9 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    v5 = V(5);
    t5 = tau * Conj(v5);
    v6 = V(6);
    t6 = tau * Conj(v6);
    v7 = V(7);
    t7 = tau * Conj(v7);
    v8 = V(8);
    t8 = tau * Conj(v8);
    v9 = V(9);
    t9 = tau * Conj(v9);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4) +
            v5 * C(j, 5) + v6 * C(j, 6) + v7 * C(j, 7) + v8 * C(j, 8) +
            v9 * C(j, 9);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
      C(j, 5) = C(j, 5) - sum * t5;
      C(j, 6) = C(j, 6) - sum * t6;
      C(j, 7) = C(j, 7) - sum * t7;
      C(j, 8) = C(j, 8) - sum * t8;
      C(j, 9) = C(j, 9) - sum * t9;
    }
    goto L410;
  L390:
    // Special code for 10 x 10 Householder
    v1 = V(1);
    t1 = tau * Conj(v1);
    v2 = V(2);
    t2 = tau * Conj(v2);
    v3 = V(3);
    t3 = tau * Conj(v3);
    v4 = V(4);
    t4 = tau * Conj(v4);
    v5 = V(5);
    t5 = tau * Conj(v5);
    v6 = V(6);
    t6 = tau * Conj(v6);
    v7 = V(7);
    t7 = tau * Conj(v7);
    v8 = V(8);
    t8 = tau * Conj(v8);
    v9 = V(9);
    t9 = tau * Conj(v9);
    v10 = V(10);
    t10 = tau * Conj(v10);
    for (j = 1; j <= m; ++j) {
      sum = v1 * C(j, 1) + v2 * C(j, 2) + v3 * C(j, 3) + v4 * C(j, 4) +
            v5 * C(j, 5) + v6 * C(j, 6) + v7 * C(j, 7) + v8 * C(j, 8) +
            v9 * C(j, 9) + v10 * C(j, 10);
      C(j, 1) = C(j, 1) - sum * t1;
      C(j, 2) = C(j, 2) - sum * t2;
      C(j, 3) = C(j, 3) - sum * t3;
      C(j, 4) = C(j, 4) - sum * t4;
      C(j, 5) = C(j, 5) - sum * t5;
      C(j, 6) = C(j, 6) - sum * t6;
      C(j, 7) = C(j, 7) - sum * t7;
      C(j, 8) = C(j, 8) - sum * t8;
      C(j, 9) = C(j, 9) - sum * t9;
      C(j, 10) = C(j, 10) - sum * t10;
    }
    goto L410;
  }
L410:;
}

namespace {

// xTRSYL for real T.
template <typename T>
blasint TrsylReal(char trana, char tranb, blasint isgn, blasint m, blasint n,
                  const T* a, blasint lda, const T* b, blasint ldb, T* c,
                  blasint ldc, T* scale) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<const R> A(a, lda);
  const Matrix<const R> B(b, ldb);
  const Matrix<R> C(c, ldc);
  R dum[1];
  R vec[4];
  const Matrix<R> VEC(vec, 2);
  R x[4];
  const Matrix<R> X(x, 2);
  blasint ierr, j, k, k1, k2, knext, l, l1, l2, lnext;
  bool notrna, notrnb;
  R a11, bignum, da11, db, eps, scaloc, sgn, smin, smlnum, suml, sumr, xnorm;
  notrna = Lsame(trana, 'N');
  notrnb = Lsame(tranb, 'N');
  blasint info = 0;
  if (!notrna && !Lsame(trana, 'T') && !Lsame(trana, 'C')) {
    info = -1;
  } else if (!notrnb && !Lsame(tranb, 'T') && !Lsame(tranb, 'C')) {
    info = -2;
  } else if (isgn != 1 && isgn != -1) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, m)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>("TRSYL", -info);
    return info;
  }
  // Quick return if possible
  *scale = one;
  if (m == 0 || n == 0) return info;
  // Set constants to control overflow
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S');
  bignum = one / smlnum;
  smlnum = smlnum * R(m * n) / eps;
  bignum = one / smlnum;
  smin = Max(smlnum, eps * Lange('M', m, m, a, lda, dum),
             eps * Lange('M', n, n, b, ldb, dum));
  sgn = R(isgn);
  if (notrna && notrnb) {
    // Solve    A*X + ISGN*X*B = *scale*C.
    // The (K,L)th block of X is determined starting from
    // bottom-left corner column by column by
    // A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
    // Where
    // M                         L-1
    // R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)].
    // I=K+1                       J=1
    // Start column loop (index = L)
    // L1 (L2) : column index of the first (first) row of X(K,L).
    lnext = 1;
    for (l = 1; l <= n; ++l) {
      if (l < lnext) goto L60;
      if (l == n) {
        l1 = l;
        l2 = l;
      } else {
        if (B(l + 1, l) != zero) {
          l1 = l;
          l2 = l + 1;
          lnext = l + 2;
        } else {
          l1 = l;
          l2 = l;
          lnext = l + 1;
        }
      }
      // Start row loop (index = K)
      // K1 (K2): row index of the first (last) row of X(K,L).
      knext = m;
      for (k = m; k >= 1; --k) {
        if (k > knext) goto L50;
        if (k == 1) {
          k1 = k;
          k2 = k;
        } else {
          if (A(k, k - 1) != zero) {
            k1 = k - 1;
            k2 = k;
            knext = k - 2;
          } else {
            k1 = k;
            k2 = k;
            knext = k - 1;
          }
        }
        if (l1 == l2 && k1 == k2) {
          suml = Dotc(m - k1, A.ptr(k1, Min<blasint>(k1 + 1, m)), lda,
                      C.ptr(Min<blasint>(k1 + 1, m), l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          scaloc = one;
          a11 = A(k1, k1) + sgn * B(l1, l1);
          da11 = std::abs(a11);
          if (da11 <= smin) {
            a11 = smin;
            da11 = smin;
            info = 1;
          }
          db = std::abs(VEC(1, 1));
          if (da11 < one && db > one) {
            if (db > bignum * da11) scaloc = one / db;
          }
          X(1, 1) = (VEC(1, 1) * scaloc) / a11;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
        } else if (l1 == l2 && k1 != k2) {
          suml = Dotc(m - k2, A.ptr(k1, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k2, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k2, 1), ldc, B.ptr(1, l1), 1);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          ierr = Laln2(false, 2, 1, smin, one, A.ptr(k1, k1), lda, one, one,
                       vec, 2, -sgn * B(l1, l1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k2, l1) = X(2, 1);
        } else if (l1 != l2 && k1 == k2) {
          suml = Dotc(m - k1, A.ptr(k1, Min<blasint>(k1 + 1, m)), lda,
                      C.ptr(Min<blasint>(k1 + 1, m), l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = sgn * (C(k1, l1) - (suml + sgn * sumr));
          suml = Dotc(m - k1, A.ptr(k1, Min<blasint>(k1 + 1, m)), lda,
                      C.ptr(Min<blasint>(k1 + 1, m), l2), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l2), 1);
          VEC(2, 1) = sgn * (C(k1, l2) - (suml + sgn * sumr));
          ierr = Laln2(true, 2, 1, smin, one, B.ptr(l1, l1), ldb, one, one, vec,
                       2, -sgn * A(k1, k1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(2, 1);
        } else if (l1 != l2 && k1 != k2) {
          suml = Dotc(m - k2, A.ptr(k1, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k1, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l2), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l2), 1);
          VEC(1, 2) = C(k1, l2) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k2, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k2, 1), ldc, B.ptr(1, l1), 1);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k2, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l2), 1);
          sumr = Dotc(l1 - 1, C.ptr(k2, 1), ldc, B.ptr(1, l2), 1);
          VEC(2, 2) = C(k2, l2) - (suml + sgn * sumr);
          ierr = Lasy2(false, false, isgn, 2, 2, A.ptr(k1, k1), lda,
                       B.ptr(l1, l1), ldb, vec, 2, &scaloc, x, 2, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(1, 2);
          C(k2, l1) = X(2, 1);
          C(k2, l2) = X(2, 2);
        }
      L50:;
      }
    L60:;
    }
  } else if (!notrna && notrnb) {
    // Solve    A**T *X + ISGN*X*B = *scale*C.
    // The (K,L)th block of X is determined starting from
    // upper-left corner column by column by
    // A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
    // Where
    // K-1        T                    L-1
    // R(K,L) = SUM [A(I,K)**T*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]
    // I=1                          J=1
    // Start column loop (index = L)
    // L1 (L2): column index of the first (last) row of X(K,L)
    lnext = 1;
    for (l = 1; l <= n; ++l) {
      if (l < lnext) goto L120;
      if (l == n) {
        l1 = l;
        l2 = l;
      } else {
        if (B(l + 1, l) != zero) {
          l1 = l;
          l2 = l + 1;
          lnext = l + 2;
        } else {
          l1 = l;
          l2 = l;
          lnext = l + 1;
        }
      }
      // Start row loop (index = K)
      // K1 (K2): row index of the first (last) row of X(K,L)
      knext = 1;
      for (k = 1; k <= m; ++k) {
        if (k < knext) goto L110;
        if (k == m) {
          k1 = k;
          k2 = k;
        } else {
          if (A(k + 1, k) != zero) {
            k1 = k;
            k2 = k + 1;
            knext = k + 2;
          } else {
            k1 = k;
            k2 = k;
            knext = k + 1;
          }
        }
        if (l1 == l2 && k1 == k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          scaloc = one;
          a11 = A(k1, k1) + sgn * B(l1, l1);
          da11 = std::abs(a11);
          if (da11 <= smin) {
            a11 = smin;
            da11 = smin;
            info = 1;
          }
          db = std::abs(VEC(1, 1));
          if (da11 < one && db > one) {
            if (db > bignum * da11) scaloc = one / db;
          }
          X(1, 1) = (VEC(1, 1) * scaloc) / a11;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
        } else if (l1 == l2 && k1 != k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k2), 1, C.ptr(1, l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k2, 1), ldc, B.ptr(1, l1), 1);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          ierr = Laln2(true, 2, 1, smin, one, A.ptr(k1, k1), lda, one, one, vec,
                       2, -sgn * B(l1, l1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k2, l1) = X(2, 1);
        } else if (l1 != l2 && k1 == k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = sgn * (C(k1, l1) - (suml + sgn * sumr));
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l2), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l2), 1);
          VEC(2, 1) = sgn * (C(k1, l2) - (suml + sgn * sumr));
          ierr = Laln2(true, 2, 1, smin, one, B.ptr(l1, l1), ldb, one, one, vec,
                       2, -sgn * A(k1, k1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(2, 1);
        } else if (l1 != l2 && k1 != k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l1), 1);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l2), 1);
          sumr = Dotc(l1 - 1, C.ptr(k1, 1), ldc, B.ptr(1, l2), 1);
          VEC(1, 2) = C(k1, l2) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k2), 1, C.ptr(1, l1), 1);
          sumr = Dotc(l1 - 1, C.ptr(k2, 1), ldc, B.ptr(1, l1), 1);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k2), 1, C.ptr(1, l2), 1);
          sumr = Dotc(l1 - 1, C.ptr(k2, 1), ldc, B.ptr(1, l2), 1);
          VEC(2, 2) = C(k2, l2) - (suml + sgn * sumr);
          ierr = Lasy2(true, false, isgn, 2, 2, A.ptr(k1, k1), lda,
                       B.ptr(l1, l1), ldb, vec, 2, &scaloc, x, 2, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(1, 2);
          C(k2, l1) = X(2, 1);
          C(k2, l2) = X(2, 2);
        }
      L110:;
      }
    L120:;
    }
  } else if (!notrna && !notrnb) {
    // Solve    A**T*X + ISGN*X*B**T = *scale*C.
    // The (K,L)th block of X is determined starting from
    // top-right corner column by column by
    // A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
    // Where
    // K-1                            N
    // R(K,L) = SUM [A(I,K)**T*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
    // I=1                          J=L+1
    // Start column loop (index = L)
    // L1 (L2): column index of the first (last) row of X(K,L)
    lnext = n;
    for (l = n; l >= 1; --l) {
      if (l > lnext) goto L180;
      if (l == 1) {
        l1 = l;
        l2 = l;
      } else {
        if (B(l, l - 1) != zero) {
          l1 = l - 1;
          l2 = l;
          lnext = l - 2;
        } else {
          l1 = l;
          l2 = l;
          lnext = l - 1;
        }
      }
      // Start row loop (index = K)
      // K1 (K2): row index of the first (last) row of X(K,L)
      knext = 1;
      for (k = 1; k <= m; ++k) {
        if (k < knext) goto L170;
        if (k == m) {
          k1 = k;
          k2 = k;
        } else {
          if (A(k + 1, k) != zero) {
            k1 = k;
            k2 = k + 1;
            knext = k + 2;
          } else {
            k1 = k;
            k2 = k;
            knext = k + 1;
          }
        }
        if (l1 == l2 && k1 == k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(n - l1, C.ptr(k1, Min<blasint>(l1 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l1 + 1, n)), ldb);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          scaloc = one;
          a11 = A(k1, k1) + sgn * B(l1, l1);
          da11 = std::abs(a11);
          if (da11 <= smin) {
            a11 = smin;
            da11 = smin;
            info = 1;
          }
          db = std::abs(VEC(1, 1));
          if (da11 < one && db > one) {
            if (db > bignum * da11) scaloc = one / db;
          }
          X(1, 1) = (VEC(1, 1) * scaloc) / a11;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
        } else if (l1 == l2 && k1 != k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k2), 1, C.ptr(1, l1), 1);
          sumr = Dotc(n - l2, C.ptr(k2, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          ierr = Laln2(true, 2, 1, smin, one, A.ptr(k1, k1), lda, one, one, vec,
                       2, -sgn * B(l1, l1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k2, l1) = X(2, 1);
        } else if (l1 != l2 && k1 == k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 1) = sgn * (C(k1, l1) - (suml + sgn * sumr));
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l2), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l2, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 1) = sgn * (C(k1, l2) - (suml + sgn * sumr));
          ierr = Laln2(false, 2, 1, smin, one, B.ptr(l1, l1), ldb, one, one,
                       vec, 2, -sgn * A(k1, k1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(2, 1);
        } else if (l1 != l2 && k1 != k2) {
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l1), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k1), 1, C.ptr(1, l2), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l2, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 2) = C(k1, l2) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k2), 1, C.ptr(1, l1), 1);
          sumr = Dotc(n - l2, C.ptr(k2, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          suml = Dotc(k1 - 1, A.ptr(1, k2), 1, C.ptr(1, l2), 1);
          sumr = Dotc(n - l2, C.ptr(k2, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l2, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 2) = C(k2, l2) - (suml + sgn * sumr);
          ierr = Lasy2(true, true, isgn, 2, 2, A.ptr(k1, k1), lda,
                       B.ptr(l1, l1), ldb, vec, 2, &scaloc, x, 2, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(1, 2);
          C(k2, l1) = X(2, 1);
          C(k2, l2) = X(2, 2);
        }
      L170:;
      }
    L180:;
    }
  } else if (notrna && !notrnb) {
    // Solve    A*X + ISGN*X*B**T = *scale*C.
    // The (K,L)th block of X is determined starting from
    // bottom-right corner column by column by
    // A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
    // Where
    // M                          N
    // R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
    // I=K+1                      J=L+1
    // Start column loop (index = L)
    // L1 (L2): column index of the first (last) row of X(K,L)
    lnext = n;
    for (l = n; l >= 1; --l) {
      if (l > lnext) goto L240;
      if (l == 1) {
        l1 = l;
        l2 = l;
      } else {
        if (B(l, l - 1) != zero) {
          l1 = l - 1;
          l2 = l;
          lnext = l - 2;
        } else {
          l1 = l;
          l2 = l;
          lnext = l - 1;
        }
      }
      // Start row loop (index = K)
      // K1 (K2): row index of the first (last) row of X(K,L)
      knext = m;
      for (k = m; k >= 1; --k) {
        if (k > knext) goto L230;
        if (k == 1) {
          k1 = k;
          k2 = k;
        } else {
          if (A(k, k - 1) != zero) {
            k1 = k - 1;
            k2 = k;
            knext = k - 2;
          } else {
            k1 = k;
            k2 = k;
            knext = k - 1;
          }
        }
        if (l1 == l2 && k1 == k2) {
          suml = Dotc(m - k1, A.ptr(k1, Min<blasint>(k1 + 1, m)), lda,
                      C.ptr(Min<blasint>(k1 + 1, m), l1), 1);
          sumr = Dotc(n - l1, C.ptr(k1, Min<blasint>(l1 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l1 + 1, n)), ldb);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          scaloc = one;
          a11 = A(k1, k1) + sgn * B(l1, l1);
          da11 = std::abs(a11);
          if (da11 <= smin) {
            a11 = smin;
            da11 = smin;
            info = 1;
          }
          db = std::abs(VEC(1, 1));
          if (da11 < one && db > one) {
            if (db > bignum * da11) scaloc = one / db;
          }
          X(1, 1) = (VEC(1, 1) * scaloc) / a11;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
        } else if (l1 == l2 && k1 != k2) {
          suml = Dotc(m - k2, A.ptr(k1, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k2, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(n - l2, C.ptr(k2, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          ierr = Laln2(false, 2, 1, smin, one, A.ptr(k1, k1), lda, one, one,
                       vec, 2, -sgn * B(l1, l1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k2, l1) = X(2, 1);
        } else if (l1 != l2 && k1 == k2) {
          suml = Dotc(m - k1, A.ptr(k1, Min<blasint>(k1 + 1, m)), lda,
                      C.ptr(Min<blasint>(k1 + 1, m), l1), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 1) = sgn * (C(k1, l1) - (suml + sgn * sumr));
          suml = Dotc(m - k1, A.ptr(k1, Min<blasint>(k1 + 1, m)), lda,
                      C.ptr(Min<blasint>(k1 + 1, m), l2), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l2, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 1) = sgn * (C(k1, l2) - (suml + sgn * sumr));
          ierr = Laln2(false, 2, 1, smin, one, B.ptr(l1, l1), ldb, one, one,
                       vec, 2, -sgn * A(k1, k1), zero, x, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(2, 1);
        } else if (l1 != l2 && k1 != k2) {
          suml = Dotc(m - k2, A.ptr(k1, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 1) = C(k1, l1) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k1, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l2), 1);
          sumr = Dotc(n - l2, C.ptr(k1, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l2, Min<blasint>(l2 + 1, n)), ldb);
          VEC(1, 2) = C(k1, l2) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k2, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l1), 1);
          sumr = Dotc(n - l2, C.ptr(k2, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l1, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 1) = C(k2, l1) - (suml + sgn * sumr);
          suml = Dotc(m - k2, A.ptr(k2, Min<blasint>(k2 + 1, m)), lda,
                      C.ptr(Min<blasint>(k2 + 1, m), l2), 1);
          sumr = Dotc(n - l2, C.ptr(k2, Min<blasint>(l2 + 1, n)), ldc,
                      B.ptr(l2, Min<blasint>(l2 + 1, n)), ldb);
          VEC(2, 2) = C(k2, l2) - (suml + sgn * sumr);
          ierr = Lasy2(false, true, isgn, 2, 2, A.ptr(k1, k1), lda,
                       B.ptr(l1, l1), ldb, vec, 2, &scaloc, x, 2, &xnorm);
          if (ierr != 0) info = 1;
          if (scaloc != one) {
            for (j = 1; j <= n; ++j) {
              Rscal(m, scaloc, C.ptr(1, j), 1);
            }
            *scale = *scale * scaloc;
          }
          C(k1, l1) = X(1, 1);
          C(k1, l2) = X(1, 2);
          C(k2, l1) = X(2, 1);
          C(k2, l2) = X(2, 2);
        }
      L230:;
      }
    L240:;
    }
  }
  return info;
}

// xTRSYL for complex T.
template <typename T>
blasint TrsylComplex(char trana, char tranb, blasint isgn, blasint m, blasint n,
                     const T* a, blasint lda, const T* b, blasint ldb, T* c,
                     blasint ldc, RealT<T>* scale) {
  using R = RealT<T>;

  const R one = R(1);
  const Matrix<const T> A(a, lda);
  const Matrix<const T> B(b, ldb);
  const Matrix<T> C(c, ldc);
  R dum[1];
  blasint j, k, l;
  bool notrna, notrnb;
  R bignum, da11, db, eps, scaloc, sgn, smin, smlnum;
  T a11, suml, sumr, vec, x11;
  notrna = Lsame(trana, 'N');
  notrnb = Lsame(tranb, 'N');
  blasint info = 0;
  if (!notrna && !Lsame(trana, 'C')) {
    info = -1;
  } else if (!notrnb && !Lsame(tranb, 'C')) {
    info = -2;
  } else if (isgn != 1 && isgn != -1) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, m)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>("TRSYL", -info);
    return info;
  }
  // Quick return if possible
  *scale = one;
  if (m == 0 || n == 0) return info;
  // Set constants to control overflow
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S');
  bignum = one / smlnum;
  smlnum = smlnum * R(m * n) / eps;
  bignum = one / smlnum;
  smin = Max(smlnum, eps * Lange('M', m, m, a, lda, dum),
             eps * Lange('M', n, n, b, ldb, dum));
  sgn = R(isgn);
  if (notrna && notrnb) {
    // Solve    A*X + ISGN*X*B = *scale*C.
    // The (K,L)th block of X is determined starting from
    // bottom-left corner column by column by
    // A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
    // Where
    // M                        L-1
    // R(K,L) = SUM [A(K,I)*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)].
    // I=K+1                      J=1
    for (l = 1; l <= n; ++l) {
      for (k = m; k >= 1; --k) {
        suml = Dotu(m - k, A.ptr(k, Min<blasint>(k + 1, m)), lda,
                    C.ptr(Min<blasint>(k + 1, m), l), 1);
        sumr = Dotu(l - 1, C.ptr(k, 1), ldc, B.ptr(1, l), 1);
        vec = C(k, l) - (suml + sgn * sumr);
        scaloc = one;
        a11 = A(k, k) + sgn * B(l, l);
        da11 = std::abs(std::real(a11)) + std::abs(std::imag(a11));
        if (da11 <= smin) {
          a11 = smin;
          da11 = smin;
          info = 1;
        }
        db = std::abs(std::real(vec)) + std::abs(std::imag(vec));
        if (da11 < one && db > one) {
          if (db > bignum * da11) scaloc = one / db;
        }
        x11 = Ladiv(vec * T(scaloc), a11);
        if (scaloc != one) {
          for (j = 1; j <= n; ++j) {
            Rscal(m, scaloc, C.ptr(1, j), 1);
          }
          *scale = *scale * scaloc;
        }
        C(k, l) = x11;
      }
    }
  } else if (!notrna && notrnb) {
    // Solve    A**H *X + ISGN*X*B = *scale*C.
    // The (K,L)th block of X is determined starting from
    // upper-left corner column by column by
    // A**H(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
    // Where
    // K-1                           L-1
    // R(K,L) = SUM [A**H(I,K)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)]
    // I=1                           J=1
    for (l = 1; l <= n; ++l) {
      for (k = 1; k <= m; ++k) {
        suml = Dotc(k - 1, A.ptr(1, k), 1, C.ptr(1, l), 1);
        sumr = Dotu(l - 1, C.ptr(k, 1), ldc, B.ptr(1, l), 1);
        vec = C(k, l) - (suml + sgn * sumr);
        scaloc = one;
        a11 = Conj(A(k, k)) + sgn * B(l, l);
        da11 = std::abs(std::real(a11)) + std::abs(std::imag(a11));
        if (da11 <= smin) {
          a11 = smin;
          da11 = smin;
          info = 1;
        }
        db = std::abs(std::real(vec)) + std::abs(std::imag(vec));
        if (da11 < one && db > one) {
          if (db > bignum * da11) scaloc = one / db;
        }
        x11 = Ladiv(vec * T(scaloc), a11);
        if (scaloc != one) {
          for (j = 1; j <= n; ++j) {
            Rscal(m, scaloc, C.ptr(1, j), 1);
          }
          *scale = *scale * scaloc;
        }
        C(k, l) = x11;
      }
    }
  } else if (!notrna && !notrnb) {
    // Solve    A**H*X + ISGN*X*B**H = C.
    // The (K,L)th block of X is determined starting from
    // upper-right corner column by column by
    // A**H(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L)
    // Where
    // K-1
    // R(K,L) = SUM [A**H(I,K)*X(I,L)] +
    // I=1
    // N
    // ISGN*SUM [X(K,J)*B**H(L,J)].
    // J=L+1
    for (l = n; l >= 1; --l) {
      for (k = 1; k <= m; ++k) {
        suml = Dotc(k - 1, A.ptr(1, k), 1, C.ptr(1, l), 1);
        sumr = Dotc(n - l, C.ptr(k, Min<blasint>(l + 1, n)), ldc,
                    B.ptr(l, Min<blasint>(l + 1, n)), ldb);
        vec = C(k, l) - (suml + sgn * Conj(sumr));
        scaloc = one;
        a11 = Conj(A(k, k) + sgn * B(l, l));
        da11 = std::abs(std::real(a11)) + std::abs(std::imag(a11));
        if (da11 <= smin) {
          a11 = smin;
          da11 = smin;
          info = 1;
        }
        db = std::abs(std::real(vec)) + std::abs(std::imag(vec));
        if (da11 < one && db > one) {
          if (db > bignum * da11) scaloc = one / db;
        }
        x11 = Ladiv(vec * T(scaloc), a11);
        if (scaloc != one) {
          for (j = 1; j <= n; ++j) {
            Rscal(m, scaloc, C.ptr(1, j), 1);
          }
          *scale = *scale * scaloc;
        }
        C(k, l) = x11;
      }
    }
  } else if (notrna && !notrnb) {
    // Solve    A*X + ISGN*X*B**H = C.
    // The (K,L)th block of X is determined starting from
    // bottom-left corner column by column by
    // A(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L)
    // Where
    // M                          N
    // R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B**H(L,J)]
    // I=K+1                      J=L+1
    for (l = n; l >= 1; --l) {
      for (k = m; k >= 1; --k) {
        suml = Dotu(m - k, A.ptr(k, Min<blasint>(k + 1, m)), lda,
                    C.ptr(Min<blasint>(k + 1, m), l), 1);
        sumr = Dotc(n - l, C.ptr(k, Min<blasint>(l + 1, n)), ldc,
                    B.ptr(l, Min<blasint>(l + 1, n)), ldb);
        vec = C(k, l) - (suml + sgn * Conj(sumr));
        scaloc = one;
        a11 = A(k, k) + sgn * Conj(B(l, l));
        da11 = std::abs(std::real(a11)) + std::abs(std::imag(a11));
        if (da11 <= smin) {
          a11 = smin;
          da11 = smin;
          info = 1;
        }
        db = std::abs(std::real(vec)) + std::abs(std::imag(vec));
        if (da11 < one && db > one) {
          if (db > bignum * da11) scaloc = one / db;
        }
        x11 = Ladiv(vec * T(scaloc), a11);
        if (scaloc != one) {
          for (j = 1; j <= n; ++j) {
            Rscal(m, scaloc, C.ptr(1, j), 1);
          }
          *scale = *scale * scaloc;
        }
        C(k, l) = x11;
      }
    }
  }
  return info;
}

}  // namespace

// xLASY2 (real only): solves for the N1-by-N2 matrix X (N1, N2 = 1 or 2)
// in op(TL) * X + ISGN * X * op(TR) = SCALE * B, op(T) = T or T**T
// (LTRANL, LTRANR), ISGN = 1 or -1, with SCALE <= 1 chosen to avoid
// overflow; XNORM is the infinity norm of X.
//
// Returns INFO: 0, or 1 if TL and TR have too close eigenvalues (then
// they are perturbed).
//
// Deviation (a fixed reference bug): the 4-by-4 elimination chooses the
// pivot with ABS(T16(IP, JP)) >= XMAX, which is false for NaN, so when
// all remaining elements are NaN (a NaN in TL or TR spreads there by
// step 3) the reference swaps with the IPSV and JPSV of the previous
// step. (At I = 1, T16 has zero elements, so a pivot is always found
// there, but compilers warn that IPSV may be unset.) The port starts the
// search with IPSV = JPSV = I, i.e. no swap; X is NaN either way.
template <typename R>
blasint Lasy2(bool ltranl, bool ltranr, blasint isgn, blasint n1, blasint n2,
              const R* tl, blasint ldtl, const R* tr, blasint ldtr, const R* b,
              blasint ldb, R* scale, R* x, blasint ldx, R* xnorm) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const R half = R(0.5e+0);
  const R eight = R(8);
  const Matrix<const R> B(b, ldb);
  R btmp[4];
  const Vector<R> BTMP(btmp);
  blasint jpiv[4];
  const Vector<blasint> JPIV(jpiv);
  R t16[16];
  const Matrix<R> T16(t16, 4);
  const Matrix<const R> TL(tl, ldtl);
  R tmp[4];
  const Vector<R> TMP(tmp);
  const Matrix<const R> TR(tr, ldtr);
  const Matrix<R> X(x, ldx);
  R x2[2];
  const Vector<R> X2(x2);
  blasint i, ip, ipiv, ipsv, j, jp, jpsv, k;
  bool bswap, xswap;
  R bet, eps, gam, l21, sgn, smin, smlnum, tau1, temp, u11, u12, u22, xmax;
  static const blasint locu12[4] = {3, 4, 1, 2};
  static const blasint locl21[4] = {2, 1, 4, 3};
  static const blasint locu22[4] = {4, 3, 2, 1};
  const Vector<const blasint> LOCU12(locu12);
  const Vector<const blasint> LOCL21(locl21);
  const Vector<const blasint> LOCU22(locu22);
  static const bool xswpiv[4] = {false, false, true, true};
  const Vector<const bool> XSWPIV(xswpiv);
  static const bool bswpiv[4] = {false, true, false, true};
  const Vector<const bool> BSWPIV(bswpiv);
  // Do not check the input parameters for errors
  blasint info = 0;
  // Quick return if possible
  if (n1 == 0 || n2 == 0) return info;
  // Set constants to control overflow
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  sgn = R(isgn);
  k = n1 + n1 + n2 - 2;
  switch (k) {
    case 1:
      goto L10;
    case 2:
      goto L20;
    case 3:
      goto L30;
    case 4:
      goto L50;
    default:  // Out of range: no jump.
      break;
  }
  // 1 by 1: TL11*X + SGN*X*TR11 = B11
L10:
  tau1 = TL(1, 1) + sgn * TR(1, 1);
  bet = std::abs(tau1);
  if (bet <= smlnum) {
    tau1 = smlnum;
    bet = smlnum;
    info = 1;
  }
  *scale = one;
  gam = std::abs(B(1, 1));
  if (smlnum * gam > bet) *scale = one / gam;
  X(1, 1) = (B(1, 1) * *scale) / tau1;
  *xnorm = std::abs(X(1, 1));
  return info;
  // 1 by 2:
  // TL11*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12]  = [B11 B12]
  // [TR21 TR22]
L20:
  smin =
      Max(eps * Max(std::abs(TL(1, 1)), std::abs(TR(1, 1)), std::abs(TR(1, 2)),
                    std::abs(TR(2, 1)), std::abs(TR(2, 2))),
          smlnum);
  TMP(1) = TL(1, 1) + sgn * TR(1, 1);
  TMP(4) = TL(1, 1) + sgn * TR(2, 2);
  if (ltranr) {
    TMP(2) = sgn * TR(2, 1);
    TMP(3) = sgn * TR(1, 2);
  } else {
    TMP(2) = sgn * TR(1, 2);
    TMP(3) = sgn * TR(2, 1);
  }
  BTMP(1) = B(1, 1);
  BTMP(2) = B(1, 2);
  goto L40;
  // 2 by 1:
  // op[TL11 TL12]*[X11] + ISGN* [X11]*TR11  = [B11]
  // [TL21 TL22] [X21]         [X21]         [B21]
L30:
  smin =
      Max(eps * Max(std::abs(TR(1, 1)), std::abs(TL(1, 1)), std::abs(TL(1, 2)),
                    std::abs(TL(2, 1)), std::abs(TL(2, 2))),
          smlnum);
  TMP(1) = TL(1, 1) + sgn * TR(1, 1);
  TMP(4) = TL(2, 2) + sgn * TR(1, 1);
  if (ltranl) {
    TMP(2) = TL(1, 2);
    TMP(3) = TL(2, 1);
  } else {
    TMP(2) = TL(2, 1);
    TMP(3) = TL(1, 2);
  }
  BTMP(1) = B(1, 1);
  BTMP(2) = B(2, 1);
L40:
  // Solve 2 by 2 system using complete pivoting.
  // Set pivots less than SMIN to SMIN.
  ipiv = Iamax(4, tmp, 1);
  u11 = TMP(ipiv);
  if (std::abs(u11) <= smin) {
    info = 1;
    u11 = smin;
  }
  u12 = TMP(LOCU12(ipiv));
  l21 = TMP(LOCL21(ipiv)) / u11;
  u22 = TMP(LOCU22(ipiv)) - u12 * l21;
  xswap = XSWPIV(ipiv);
  bswap = BSWPIV(ipiv);
  if (std::abs(u22) <= smin) {
    info = 1;
    u22 = smin;
  }
  if (bswap) {
    temp = BTMP(2);
    BTMP(2) = BTMP(1) - l21 * temp;
    BTMP(1) = temp;
  } else {
    BTMP(2) = BTMP(2) - l21 * BTMP(1);
  }
  *scale = one;
  if ((two * smlnum) * std::abs(BTMP(2)) > std::abs(u22) ||
      (two * smlnum) * std::abs(BTMP(1)) > std::abs(u11)) {
    *scale = half / Max(std::abs(BTMP(1)), std::abs(BTMP(2)));
    BTMP(1) = BTMP(1) * *scale;
    BTMP(2) = BTMP(2) * *scale;
  }
  X2(2) = BTMP(2) / u22;
  X2(1) = BTMP(1) / u11 - (u12 / u11) * X2(2);
  if (xswap) {
    temp = X2(2);
    X2(2) = X2(1);
    X2(1) = temp;
  }
  X(1, 1) = X2(1);
  if (n1 == 1) {
    X(1, 2) = X2(2);
    *xnorm = std::abs(X(1, 1)) + std::abs(X(1, 2));
  } else {
    X(2, 1) = X2(2);
    *xnorm = Max(std::abs(X(1, 1)), std::abs(X(2, 1)));
  }
  return info;
  // 2 by 2:
  // op[TL11 TL12]*[X11 X12] +ISGN* [X11 X12]*op[TR11 TR12] = [B11 B12]
  // [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22]
  // Solve equivalent 4 by 4 system using complete pivoting.
  // Set pivots less than SMIN to SMIN.
L50:
  smin = Max(std::abs(TR(1, 1)), std::abs(TR(1, 2)), std::abs(TR(2, 1)),
             std::abs(TR(2, 2)));
  smin = Max(smin, std::abs(TL(1, 1)), std::abs(TL(1, 2)), std::abs(TL(2, 1)),
             std::abs(TL(2, 2)));
  smin = Max(eps * smin, smlnum);
  BTMP(1) = zero;
  Copy(16, btmp, 0, t16, 1);
  T16(1, 1) = TL(1, 1) + sgn * TR(1, 1);
  T16(2, 2) = TL(2, 2) + sgn * TR(1, 1);
  T16(3, 3) = TL(1, 1) + sgn * TR(2, 2);
  T16(4, 4) = TL(2, 2) + sgn * TR(2, 2);
  if (ltranl) {
    T16(1, 2) = TL(2, 1);
    T16(2, 1) = TL(1, 2);
    T16(3, 4) = TL(2, 1);
    T16(4, 3) = TL(1, 2);
  } else {
    T16(1, 2) = TL(1, 2);
    T16(2, 1) = TL(2, 1);
    T16(3, 4) = TL(1, 2);
    T16(4, 3) = TL(2, 1);
  }
  if (ltranr) {
    T16(1, 3) = sgn * TR(1, 2);
    T16(2, 4) = sgn * TR(1, 2);
    T16(3, 1) = sgn * TR(2, 1);
    T16(4, 2) = sgn * TR(2, 1);
  } else {
    T16(1, 3) = sgn * TR(2, 1);
    T16(2, 4) = sgn * TR(2, 1);
    T16(3, 1) = sgn * TR(1, 2);
    T16(4, 2) = sgn * TR(1, 2);
  }
  BTMP(1) = B(1, 1);
  BTMP(2) = B(2, 1);
  BTMP(3) = B(1, 2);
  BTMP(4) = B(2, 2);
  // Perform elimination
  for (i = 1; i <= 3; ++i) {
    xmax = zero;
    // Not in the reference (see above): no swap if all elements are NaN.
    ipsv = i;
    jpsv = i;
    for (ip = i; ip <= 4; ++ip) {
      for (jp = i; jp <= 4; ++jp) {
        if (std::abs(T16(ip, jp)) >= xmax) {
          xmax = std::abs(T16(ip, jp));
          ipsv = ip;
          jpsv = jp;
        }
      }
    }
    if (ipsv != i) {
      Swap(4, T16.ptr(ipsv, 1), 4, T16.ptr(i, 1), 4);
      temp = BTMP(i);
      BTMP(i) = BTMP(ipsv);
      BTMP(ipsv) = temp;
    }
    if (jpsv != i) Swap(4, T16.ptr(1, jpsv), 1, T16.ptr(1, i), 1);
    JPIV(i) = jpsv;
    if (std::abs(T16(i, i)) < smin) {
      info = 1;
      T16(i, i) = smin;
    }
    for (j = i + 1; j <= 4; ++j) {
      T16(j, i) = T16(j, i) / T16(i, i);
      BTMP(j) = BTMP(j) - T16(j, i) * BTMP(i);
      for (k = i + 1; k <= 4; ++k) {
        T16(j, k) = T16(j, k) - T16(j, i) * T16(i, k);
      }
    }
  }
  if (std::abs(T16(4, 4)) < smin) {
    info = 1;
    T16(4, 4) = smin;
  }
  *scale = one;
  if ((eight * smlnum) * std::abs(BTMP(1)) > std::abs(T16(1, 1)) ||
      (eight * smlnum) * std::abs(BTMP(2)) > std::abs(T16(2, 2)) ||
      (eight * smlnum) * std::abs(BTMP(3)) > std::abs(T16(3, 3)) ||
      (eight * smlnum) * std::abs(BTMP(4)) > std::abs(T16(4, 4))) {
    *scale = (one / eight) / Max(std::abs(BTMP(1)), std::abs(BTMP(2)),
                                 std::abs(BTMP(3)), std::abs(BTMP(4)));
    BTMP(1) = BTMP(1) * *scale;
    BTMP(2) = BTMP(2) * *scale;
    BTMP(3) = BTMP(3) * *scale;
    BTMP(4) = BTMP(4) * *scale;
  }
  for (i = 1; i <= 4; ++i) {
    k = 5 - i;
    temp = one / T16(k, k);
    TMP(k) = BTMP(k) * temp;
    for (j = k + 1; j <= 4; ++j) {
      TMP(k) = TMP(k) - (temp * T16(k, j)) * TMP(j);
    }
  }
  for (i = 1; i <= 3; ++i) {
    if (JPIV(4 - i) != 4 - i) {
      temp = TMP(4 - i);
      TMP(4 - i) = TMP(JPIV(4 - i));
      TMP(JPIV(4 - i)) = temp;
    }
  }
  X(1, 1) = TMP(1);
  X(2, 1) = TMP(2);
  X(1, 2) = TMP(3);
  X(2, 2) = TMP(4);
  *xnorm = Max(std::abs(TMP(1)) + std::abs(TMP(3)),
               std::abs(TMP(2)) + std::abs(TMP(4)));
  return info;
}

// xLALN2 (real only): solves (ca A - w D) X = s B or (ca A**T - w D) X = s B
// (LTRANS) for the NA-by-NA (NA = 1 or 2) matrix A, D = diag(D1, D2), the
// real (NW = 1) or complex (NW = 2, w = WR + i*WI, X and B have a real and
// an imaginary column) scalar w, with the scale factor s = SCALE <= 1
// chosen to avoid overflow and perturbations of size SMIN if the system is
// nearly singular; XNORM is the infinity norm of X.
//
// Returns INFO: 0, or 1 if the system was perturbed.
template <typename R>
blasint Laln2(bool ltrans, blasint na, blasint nw, R smin, R ca, const R* a,
              blasint lda, R d1, R d2, const R* b, blasint ldb, R wr, R wi,
              R* x, blasint ldx, R* scale, R* xnorm) {
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Matrix<const R> A(a, lda);
  const Matrix<const R> B(b, ldb);
  // CI(2,2) and CIV(4), and CR(2,2) and CRV(4), share their storage
  // (EQUIVALENCE).
  R ci[4];
  const Matrix<R> CI(ci, 2);
  const Vector<R> CIV(ci);
  R cr[4];
  const Matrix<R> CR(cr, 2);
  const Vector<R> CRV(cr);
  const Matrix<R> X(x, ldx);
  blasint icmax, j;
  R bbnd, bi1, bi2, bignum, bnorm, br1, br2, ci21, ci22, cmax, cnorm, cr21,
      cr22, csi, csr, li21, lr21, smini, smlnum, temp, u22abs, ui11, ui11r,
      ui12, ui12s, ui22, ur11, ur11r, ur12, ur12s, ur22, xi1, xi2, xr1, xr2;
  static const bool zswap[4] = {false, false, true, true};
  const Vector<const bool> ZSWAP(zswap);
  static const bool rswap[4] = {false, true, false, true};
  const Vector<const bool> RSWAP(rswap);
  static const blasint ipivot[16] = {1, 2, 3, 4, 2, 1, 4, 3,
                                     3, 4, 1, 2, 4, 3, 2, 1};
  const Matrix<const blasint> IPIVOT(ipivot, 4);
  // Compute BIGNUM
  smlnum = two * Lamch<R>('S');
  bignum = one / smlnum;
  smini = Max(smin, smlnum);
  // Don't check for input errors
  blasint info = 0;
  // Standard Initializations
  *scale = one;
  if (na == 1) {
    // 1 x 1  (i.e., scalar) system   C X = B
    if (nw == 1) {
      // Real 1x1 system.
      // C = ca A - w D
      csr = ca * A(1, 1) - wr * d1;
      cnorm = std::abs(csr);
      // If | C | < SMINI, use C = SMINI
      if (cnorm < smini) {
        csr = smini;
        cnorm = smini;
        info = 1;
      }
      // Check scaling for  X = B / C
      bnorm = std::abs(B(1, 1));
      if (cnorm < one && bnorm > one) {
        if (bnorm > bignum * cnorm) *scale = one / bnorm;
      }
      // Compute X
      X(1, 1) = (B(1, 1) * *scale) / csr;
      *xnorm = std::abs(X(1, 1));
    } else {
      // Complex 1x1 system (w is complex)
      // C = ca A - w D
      csr = ca * A(1, 1) - wr * d1;
      csi = -wi * d1;
      cnorm = std::abs(csr) + std::abs(csi);
      // If | C | < SMINI, use C = SMINI
      if (cnorm < smini) {
        csr = smini;
        csi = zero;
        cnorm = smini;
        info = 1;
      }
      // Check scaling for  X = B / C
      bnorm = std::abs(B(1, 1)) + std::abs(B(1, 2));
      if (cnorm < one && bnorm > one) {
        if (bnorm > bignum * cnorm) *scale = one / bnorm;
      }
      // Compute X
      Ladiv(*scale * B(1, 1), *scale * B(1, 2), csr, csi, X.ptr(1, 1),
            X.ptr(1, 2));
      *xnorm = std::abs(X(1, 1)) + std::abs(X(1, 2));
    }
  } else {
    // 2x2 System
    // Compute the real part of  C = ca A - w D  (or  ca A**T - w D )
    CR(1, 1) = ca * A(1, 1) - wr * d1;
    CR(2, 2) = ca * A(2, 2) - wr * d2;
    if (ltrans) {
      CR(1, 2) = ca * A(2, 1);
      CR(2, 1) = ca * A(1, 2);
    } else {
      CR(2, 1) = ca * A(2, 1);
      CR(1, 2) = ca * A(1, 2);
    }
    if (nw == 1) {
      // Real 2x2 system  (w is real)
      // Find the largest element in C
      cmax = zero;
      icmax = 0;
      for (j = 1; j <= 4; ++j) {
        if (std::abs(CRV(j)) > cmax) {
          cmax = std::abs(CRV(j));
          icmax = j;
        }
      }
      // If norm(C) < SMINI, use SMINI*identity.
      if (cmax < smini) {
        bnorm = Max(std::abs(B(1, 1)), std::abs(B(2, 1)));
        if (smini < one && bnorm > one) {
          if (bnorm > bignum * smini) *scale = one / bnorm;
        }
        temp = *scale / smini;
        X(1, 1) = temp * B(1, 1);
        X(2, 1) = temp * B(2, 1);
        *xnorm = temp * bnorm;
        info = 1;
        return info;
      }
      // Gaussian elimination with complete pivoting.
      ur11 = CRV(icmax);
      cr21 = CRV(IPIVOT(2, icmax));
      ur12 = CRV(IPIVOT(3, icmax));
      cr22 = CRV(IPIVOT(4, icmax));
      ur11r = one / ur11;
      lr21 = ur11r * cr21;
      ur22 = cr22 - ur12 * lr21;
      // If smaller pivot < SMINI, use SMINI
      if (std::abs(ur22) < smini) {
        ur22 = smini;
        info = 1;
      }
      if (RSWAP(icmax)) {
        br1 = B(2, 1);
        br2 = B(1, 1);
      } else {
        br1 = B(1, 1);
        br2 = B(2, 1);
      }
      br2 = br2 - lr21 * br1;
      bbnd = Max(std::abs(br1 * (ur22 * ur11r)), std::abs(br2));
      if (bbnd > one && std::abs(ur22) < one) {
        if (bbnd >= bignum * std::abs(ur22)) *scale = one / bbnd;
      }
      xr2 = (br2 * *scale) / ur22;
      xr1 = (*scale * br1) * ur11r - xr2 * (ur11r * ur12);
      if (ZSWAP(icmax)) {
        X(1, 1) = xr2;
        X(2, 1) = xr1;
      } else {
        X(1, 1) = xr1;
        X(2, 1) = xr2;
      }
      *xnorm = Max(std::abs(xr1), std::abs(xr2));
      // Further scaling if  norm(A) norm(X) > overflow
      if (*xnorm > one && cmax > one) {
        if (*xnorm > bignum / cmax) {
          temp = cmax / bignum;
          X(1, 1) = temp * X(1, 1);
          X(2, 1) = temp * X(2, 1);
          *xnorm = temp * *xnorm;
          *scale = temp * *scale;
        }
      }
    } else {
      // Complex 2x2 system  (w is complex)
      // Find the largest element in C
      CI(1, 1) = -wi * d1;
      CI(2, 1) = zero;
      CI(1, 2) = zero;
      CI(2, 2) = -wi * d2;
      cmax = zero;
      icmax = 0;
      for (j = 1; j <= 4; ++j) {
        if (std::abs(CRV(j)) + std::abs(CIV(j)) > cmax) {
          cmax = std::abs(CRV(j)) + std::abs(CIV(j));
          icmax = j;
        }
      }
      // If norm(C) < SMINI, use SMINI*identity.
      if (cmax < smini) {
        bnorm = Max(std::abs(B(1, 1)) + std::abs(B(1, 2)),
                    std::abs(B(2, 1)) + std::abs(B(2, 2)));
        if (smini < one && bnorm > one) {
          if (bnorm > bignum * smini) *scale = one / bnorm;
        }
        temp = *scale / smini;
        X(1, 1) = temp * B(1, 1);
        X(2, 1) = temp * B(2, 1);
        X(1, 2) = temp * B(1, 2);
        X(2, 2) = temp * B(2, 2);
        *xnorm = temp * bnorm;
        info = 1;
        return info;
      }
      // Gaussian elimination with complete pivoting.
      ur11 = CRV(icmax);
      ui11 = CIV(icmax);
      cr21 = CRV(IPIVOT(2, icmax));
      ci21 = CIV(IPIVOT(2, icmax));
      ur12 = CRV(IPIVOT(3, icmax));
      ui12 = CIV(IPIVOT(3, icmax));
      cr22 = CRV(IPIVOT(4, icmax));
      ci22 = CIV(IPIVOT(4, icmax));
      if (icmax == 1 || icmax == 4) {
        // Code when off-diagonals of pivoted C are real
        if (std::abs(ur11) > std::abs(ui11)) {
          temp = ui11 / ur11;
          ur11r = one / (ur11 * (one + temp * temp));
          ui11r = -temp * ur11r;
        } else {
          temp = ur11 / ui11;
          ui11r = -one / (ui11 * (one + temp * temp));
          ur11r = -temp * ui11r;
        }
        lr21 = cr21 * ur11r;
        li21 = cr21 * ui11r;
        ur12s = ur12 * ur11r;
        ui12s = ur12 * ui11r;
        ur22 = cr22 - ur12 * lr21;
        ui22 = ci22 - ur12 * li21;
      } else {
        // Code when diagonals of pivoted C are real
        ur11r = one / ur11;
        ui11r = zero;
        lr21 = cr21 * ur11r;
        li21 = ci21 * ur11r;
        ur12s = ur12 * ur11r;
        ui12s = ui12 * ur11r;
        ur22 = cr22 - ur12 * lr21 + ui12 * li21;
        ui22 = -ur12 * li21 - ui12 * lr21;
      }
      u22abs = std::abs(ur22) + std::abs(ui22);
      // If smaller pivot < SMINI, use SMINI
      if (u22abs < smini) {
        ur22 = smini;
        ui22 = zero;
        info = 1;
      }
      if (RSWAP(icmax)) {
        br2 = B(1, 1);
        br1 = B(2, 1);
        bi2 = B(1, 2);
        bi1 = B(2, 2);
      } else {
        br1 = B(1, 1);
        br2 = B(2, 1);
        bi1 = B(1, 2);
        bi2 = B(2, 2);
      }
      br2 = br2 - lr21 * br1 + li21 * bi1;
      bi2 = bi2 - li21 * br1 - lr21 * bi1;
      bbnd = Max((std::abs(br1) + std::abs(bi1)) *
                     (u22abs * (std::abs(ur11r) + std::abs(ui11r))),
                 std::abs(br2) + std::abs(bi2));
      if (bbnd > one && u22abs < one) {
        if (bbnd >= bignum * u22abs) {
          *scale = one / bbnd;
          br1 = *scale * br1;
          bi1 = *scale * bi1;
          br2 = *scale * br2;
          bi2 = *scale * bi2;
        }
      }
      Ladiv(br2, bi2, ur22, ui22, &xr2, &xi2);
      xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
      xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
      if (ZSWAP(icmax)) {
        X(1, 1) = xr2;
        X(2, 1) = xr1;
        X(1, 2) = xi2;
        X(2, 2) = xi1;
      } else {
        X(1, 1) = xr1;
        X(2, 1) = xr2;
        X(1, 2) = xi1;
        X(2, 2) = xi2;
      }
      *xnorm =
          Max(std::abs(xr1) + std::abs(xi1), std::abs(xr2) + std::abs(xi2));
      // Further scaling if  norm(A) norm(X) > overflow
      if (*xnorm > one && cmax > one) {
        if (*xnorm > bignum / cmax) {
          temp = cmax / bignum;
          X(1, 1) = temp * X(1, 1);
          X(2, 1) = temp * X(2, 1);
          X(1, 2) = temp * X(1, 2);
          X(2, 2) = temp * X(2, 2);
          *xnorm = temp * *xnorm;
          *scale = temp * *scale;
        }
      }
    }
  }
  return info;
}

// xLAEXC (real only): swaps the adjacent diagonal blocks T11 (N1-by-N1)
// and T22 (N2-by-N2), N1, N2 = 1 or 2, starting at row and column J1 of
// the upper quasi-triangular matrix T in Schur canonical form by an
// orthogonal similarity transformation, and applies it to Q (WANTQ).
// WORK has N elements.
//
// Returns INFO: 0, or 1 if the swap was rejected (the result would have
// been too far from Schur form; T and Q are unchanged).
template <typename R>
blasint Laexc(bool wantq, blasint n, R* t, blasint ldt, R* q, blasint ldq,
              blasint j1, blasint n1, blasint n2, R* work) {
  const R zero = R(0);
  const R one = R(1);
  const R ten = R(10);
  const blasint ldd = 4;
  const blasint ldx = 2;
  R d[ldd * 4];
  const Matrix<R> D(d, ldd);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> Tm(t, ldt);
  R u[3];
  const Vector<R> U(u);
  R u1[3];
  const Vector<R> U1(u1);
  R u2[3];
  const Vector<R> U2(u2);
  const Vector<R> WORK(work);
  // Initialized (outputs of Lasy2, which the static analyzer cannot see).
  R x[ldx * 2] = {};
  const Matrix<R> X(x, ldx);
  blasint j2, j3, j4, k, nd;
  R cs, dnorm, eps, scale = R(0), smlnum, sn, t11, t22, t33, tau, tau1, tau2,
                    temp, thresh, wi1, wi2, wr1, wr2, xnorm;
  blasint info = 0;
  // Quick return if possible
  if (n == 0 || n1 == 0 || n2 == 0) return info;
  if (j1 + n1 > n) return info;
  j2 = j1 + 1;
  j3 = j1 + 2;
  j4 = j1 + 3;
  if (n1 == 1 && n2 == 1) {
    // Swap two 1-by-1 blocks.
    t11 = Tm(j1, j1);
    t22 = Tm(j2, j2);
    // Determine the transformation to perform the interchange.
    Lartg(Tm(j1, j2), t22 - t11, &cs, &sn, &temp);
    // Apply transformation to the matrix T.
    if (j3 <= n)
      Rot(n - j1 - 1, Tm.ptr(j1, j3), ldt, Tm.ptr(j2, j3), ldt, cs, sn);
    Rot(j1 - 1, Tm.ptr(1, j1), 1, Tm.ptr(1, j2), 1, cs, sn);
    Tm(j1, j1) = t22;
    Tm(j2, j2) = t11;
    if (wantq) {
      // Accumulate transformation in the matrix Q.
      Rot(n, Q.ptr(1, j1), 1, Q.ptr(1, j2), 1, cs, sn);
    }
  } else {
    // Swapping involves at least one 2-by-2 block.
    // Copy the diagonal block of order N1+N2 to the local array D
    // and compute its norm.
    nd = n1 + n2;
    Lacpy('F', nd, nd, Tm.ptr(j1, j1), ldt, d, ldd);
    dnorm = Lange('M', nd, nd, d, ldd, work);
    // Compute machine-dependent threshold for test for accepting
    // swap.
    eps = Lamch<R>('P');
    smlnum = Lamch<R>('S') / eps;
    thresh = Max(ten * eps * dnorm, smlnum);
    // Solve T11*X - X*T22 = scale*T12 for X.
    Lasy2(false, false, -1, n1, n2, d, ldd, D.ptr(n1 + 1, n1 + 1), ldd,
          D.ptr(1, n1 + 1), ldd, &scale, x, ldx, &xnorm);
    // Swap the adjacent diagonal blocks.
    k = n1 + n1 + n2 - 3;
    switch (k) {
      case 1:
        goto L10;
      case 2:
        goto L20;
      case 3:
        goto L30;
      default:  // Out of range: no jump.
        break;
    }
  L10:
    // N1 = 1, N2 = 2: generate elementary reflector H so that:
    // ( scale, X11, X12 ) H = ( 0, 0, * )
    U(1) = scale;
    U(2) = X(1, 1);
    U(3) = X(1, 2);
    Larfg(3, U.ptr(3), u, 1, &tau);
    U(3) = one;
    t11 = Tm(j1, j1);
    // Perform swap provisionally on diagonal block in D.
    Larfx('L', 3, 3, u, tau, d, ldd, work);
    Larfx('R', 3, 3, u, tau, d, ldd, work);
    // Test whether to reject swap.
    if (Max(std::abs(D(3, 1)), std::abs(D(3, 2)), std::abs(D(3, 3) - t11)) >
        thresh)
      goto L50;
    // Accept swap: apply transformation to the entire matrix T.
    Larfx('L', 3, n - j1 + 1, u, tau, Tm.ptr(j1, j1), ldt, work);
    Larfx('R', j2, 3, u, tau, Tm.ptr(1, j1), ldt, work);
    Tm(j3, j1) = zero;
    Tm(j3, j2) = zero;
    Tm(j3, j3) = t11;
    if (wantq) {
      // Accumulate transformation in the matrix Q.
      Larfx('R', n, 3, u, tau, Q.ptr(1, j1), ldq, work);
    }
    goto L40;
  L20:
    // N1 = 2, N2 = 1: generate elementary reflector H so that:
    // H (  -X11 ) = ( * )
    // (  -X21 ) = ( 0 )
    // ( scale ) = ( 0 )
    U(1) = -X(1, 1);
    U(2) = -X(2, 1);
    U(3) = scale;
    Larfg(3, U.ptr(1), U.ptr(2), 1, &tau);
    U(1) = one;
    t33 = Tm(j3, j3);
    // Perform swap provisionally on diagonal block in D.
    Larfx('L', 3, 3, u, tau, d, ldd, work);
    Larfx('R', 3, 3, u, tau, d, ldd, work);
    // Test whether to reject swap.
    if (Max(std::abs(D(2, 1)), std::abs(D(3, 1)), std::abs(D(1, 1) - t33)) >
        thresh)
      goto L50;
    // Accept swap: apply transformation to the entire matrix T.
    Larfx('R', j3, 3, u, tau, Tm.ptr(1, j1), ldt, work);
    Larfx('L', 3, n - j1, u, tau, Tm.ptr(j1, j2), ldt, work);
    Tm(j1, j1) = t33;
    Tm(j2, j1) = zero;
    Tm(j3, j1) = zero;
    if (wantq) {
      // Accumulate transformation in the matrix Q.
      Larfx('R', n, 3, u, tau, Q.ptr(1, j1), ldq, work);
    }
    goto L40;
  L30:
    // N1 = 2, N2 = 2: generate elementary reflectors H(1) and H(2) so
    // that:
    // H(2) H(1) (  -X11  -X12 ) = (  *  * )
    // (  -X21  -X22 )   (  0  * )
    // ( scale    0  )   (  0  0 )
    // (    0  scale )   (  0  0 )
    U1(1) = -X(1, 1);
    U1(2) = -X(2, 1);
    U1(3) = scale;
    Larfg(3, U1.ptr(1), U1.ptr(2), 1, &tau1);
    U1(1) = one;
    temp = -tau1 * (X(1, 2) + U1(2) * X(2, 2));
    U2(1) = -temp * U1(2) - X(2, 2);
    U2(2) = -temp * U1(3);
    U2(3) = scale;
    Larfg(3, U2.ptr(1), U2.ptr(2), 1, &tau2);
    U2(1) = one;
    // Perform swap provisionally on diagonal block in D.
    Larfx('L', 3, 4, u1, tau1, d, ldd, work);
    Larfx('R', 4, 3, u1, tau1, d, ldd, work);
    Larfx('L', 3, 4, u2, tau2, D.ptr(2, 1), ldd, work);
    Larfx('R', 4, 3, u2, tau2, D.ptr(1, 2), ldd, work);
    // Test whether to reject swap.
    if (Max(std::abs(D(3, 1)), std::abs(D(3, 2)), std::abs(D(4, 1)),
            std::abs(D(4, 2))) > thresh)
      goto L50;
    // Accept swap: apply transformation to the entire matrix T.
    Larfx('L', 3, n - j1 + 1, u1, tau1, Tm.ptr(j1, j1), ldt, work);
    Larfx('R', j4, 3, u1, tau1, Tm.ptr(1, j1), ldt, work);
    Larfx('L', 3, n - j1 + 1, u2, tau2, Tm.ptr(j2, j1), ldt, work);
    Larfx('R', j4, 3, u2, tau2, Tm.ptr(1, j2), ldt, work);
    Tm(j3, j1) = zero;
    Tm(j3, j2) = zero;
    Tm(j4, j1) = zero;
    Tm(j4, j2) = zero;
    if (wantq) {
      // Accumulate transformation in the matrix Q.
      Larfx('R', n, 3, u1, tau1, Q.ptr(1, j1), ldq, work);
      Larfx('R', n, 3, u2, tau2, Q.ptr(1, j2), ldq, work);
    }
  L40:
    if (n2 == 2) {
      // Standardize new 2-by-2 block T11
      Lanv2(Tm.ptr(j1, j1), Tm.ptr(j1, j2), Tm.ptr(j2, j1), Tm.ptr(j2, j2),
            &wr1, &wi1, &wr2, &wi2, &cs, &sn);
      Rot(n - j1 - 1, Tm.ptr(j1, j1 + 2), ldt, Tm.ptr(j2, j1 + 2), ldt, cs, sn);
      Rot(j1 - 1, Tm.ptr(1, j1), 1, Tm.ptr(1, j2), 1, cs, sn);
      if (wantq) Rot(n, Q.ptr(1, j1), 1, Q.ptr(1, j2), 1, cs, sn);
    }
    if (n1 == 2) {
      // Standardize new 2-by-2 block T22
      j3 = j1 + n2;
      j4 = j3 + 1;
      Lanv2(Tm.ptr(j3, j3), Tm.ptr(j3, j4), Tm.ptr(j4, j3), Tm.ptr(j4, j4),
            &wr1, &wi1, &wr2, &wi2, &cs, &sn);
      if (j3 + 2 <= n)
        Rot(n - j3 - 1, Tm.ptr(j3, j3 + 2), ldt, Tm.ptr(j4, j3 + 2), ldt, cs,
            sn);
      Rot(j3 - 1, Tm.ptr(1, j3), 1, Tm.ptr(1, j4), 1, cs, sn);
      if (wantq) Rot(n, Q.ptr(1, j3), 1, Q.ptr(1, j4), 1, cs, sn);
    }
  }
  return info;
  // Exit with INFO = 1 if swap was rejected.
L50:
  info = 1;
  return info;
}

// xLACN2 (real T): estimates the 1-norm of a square matrix A by reverse
// communication: on the first call KASE = 0; while KASE != 0 on return,
// the caller overwrites X with A * X (KASE = 1) or A**T * X (KASE = 2)
// and calls again. EST is the estimate, V (N elements) holds A * V =
// W with EST = norm(V) / norm(W), ISGN (N elements) and ISAVE (3) keep the
// state between calls.
template <typename R>
void Lacn2(blasint n, R* v, R* x, blasint* isgn, R* est, blasint* kase,
           blasint* isave) {
  const blasint itmax = 5;
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Vector<blasint> ISAVE(isave);
  const Vector<blasint> ISGN(isgn);
  const Vector<R> V(v);
  const Vector<R> X(x);
  blasint i, jlast;
  R altsgn, estold, temp, xs;
  if (*kase == 0) {
    for (i = 1; i <= n; ++i) {
      X(i) = one / R(n);
    }
    *kase = 1;
    ISAVE(1) = 1;
    return;
  }
  switch (ISAVE(1)) {
    case 1:
      goto L20;
    case 2:
      goto L40;
    case 3:
      goto L70;
    case 4:
      goto L110;
    case 5:
      goto L140;
    default:  // Out of range: no jump.
      break;
  }
  // FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X.
L20:
  if (n == 1) {
    V(1) = X(1);
    *est = std::abs(V(1));
    goto L150;
  }
  *est = Asum(n, x, 1);
  for (i = 1; i <= n; ++i) {
    if (X(i) >= zero) {
      X(i) = one;
    } else {
      X(i) = -one;
    }
    ISGN(i) = Nint(X(i));
  }
  *kase = 2;
  ISAVE(1) = 2;
  return;
  // FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
L40:
  ISAVE(2) = Iamax(n, x, 1);
  ISAVE(3) = 2;
  // MAIN LOOP - ITERATIONS 2,3,...,ITMAX.
L50:
  for (i = 1; i <= n; ++i) {
    X(i) = zero;
  }
  X(ISAVE(2)) = one;
  *kase = 1;
  ISAVE(1) = 3;
  return;
  // X HAS BEEN OVERWRITTEN BY A*X.
L70:
  Copy(n, x, 1, v, 1);
  estold = *est;
  *est = Asum(n, v, 1);
  for (i = 1; i <= n; ++i) {
    if (X(i) >= zero) {
      xs = one;
    } else {
      xs = -one;
    }
    if (Nint(xs) != ISGN(i)) goto L90;
  }
  // REPEATED SIGN VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED.
  goto L120;
L90:
  // TEST FOR CYCLING.
  if (*est <= estold) goto L120;
  for (i = 1; i <= n; ++i) {
    if (X(i) >= zero) {
      X(i) = one;
    } else {
      X(i) = -one;
    }
    ISGN(i) = Nint(X(i));
  }
  *kase = 2;
  ISAVE(1) = 4;
  return;
  // X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.
L110:
  jlast = ISAVE(2);
  ISAVE(2) = Iamax(n, x, 1);
  if ((X(jlast) != std::abs(X(ISAVE(2)))) && (ISAVE(3) < itmax)) {
    ISAVE(3) = ISAVE(3) + 1;
    goto L50;
  }
  // ITERATION COMPLETE.  FINAL STAGE.
L120:
  altsgn = one;
  for (i = 1; i <= n; ++i) {
    X(i) = altsgn * (one + R(i - 1) / R(n - 1));
    altsgn = -altsgn;
  }
  *kase = 1;
  ISAVE(1) = 5;
  return;
  // X HAS BEEN OVERWRITTEN BY A*X.
L140:
  temp = two * (Asum(n, x, 1) / R(3 * n));
  if (temp > *est) {
    Copy(n, x, 1, v, 1);
    *est = temp;
  }
L150:
  *kase = 0;
}

// xLACN2 (complex T): as the real routine (with A**H for KASE = 2).
template <typename T>
void Lacn2(blasint n, T* v, T* x, RealT<T>* est, blasint* kase,
           blasint* isave) {
  using R = RealT<T>;

  const blasint itmax = 5;
  const R one = R(1);
  const R two = R(2);
  const T czero = T(0);
  const T cone = T(1);
  const Vector<blasint> ISAVE(isave);
  const Vector<T> V(v);
  const Vector<T> X(x);
  blasint i, jlast;
  R absxi, altsgn, estold, safmin, temp;
  safmin = Lamch<R>('S');
  if (*kase == 0) {
    for (i = 1; i <= n; ++i) {
      X(i) = T(one / R(n));
    }
    *kase = 1;
    ISAVE(1) = 1;
    return;
  }
  switch (ISAVE(1)) {
    case 1:
      goto L20;
    case 2:
      goto L40;
    case 3:
      goto L70;
    case 4:
      goto L90;
    case 5:
      goto L120;
    default:  // Out of range: no jump.
      break;
  }
  // FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X.
L20:
  if (n == 1) {
    V(1) = X(1);
    *est = std::abs(V(1));
    goto L130;
  }
  *est = Sum1(n, x, 1);
  for (i = 1; i <= n; ++i) {
    absxi = std::abs(X(i));
    if (absxi > safmin) {
      X(i) = T(std::real(X(i)) / absxi, std::imag(X(i)) / absxi);
    } else {
      X(i) = cone;
    }
  }
  *kase = 2;
  ISAVE(1) = 2;
  return;
  // FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY CTRANS(A)*X.
L40:
  ISAVE(2) = Imax1(n, x, 1);
  ISAVE(3) = 2;
  // MAIN LOOP - ITERATIONS 2,3,...,ITMAX.
L50:
  for (i = 1; i <= n; ++i) {
    X(i) = czero;
  }
  X(ISAVE(2)) = cone;
  *kase = 1;
  ISAVE(1) = 3;
  return;
  // X HAS BEEN OVERWRITTEN BY A*X.
L70:
  Copy(n, x, 1, v, 1);
  estold = *est;
  *est = Sum1(n, v, 1);
  // TEST FOR CYCLING.
  if (*est <= estold) goto L100;
  for (i = 1; i <= n; ++i) {
    absxi = std::abs(X(i));
    if (absxi > safmin) {
      X(i) = T(std::real(X(i)) / absxi, std::imag(X(i)) / absxi);
    } else {
      X(i) = cone;
    }
  }
  *kase = 2;
  ISAVE(1) = 4;
  return;
  // X HAS BEEN OVERWRITTEN BY CTRANS(A)*X.
L90:
  jlast = ISAVE(2);
  ISAVE(2) = Imax1(n, x, 1);
  if ((std::abs(X(jlast)) != std::abs(X(ISAVE(2)))) && (ISAVE(3) < itmax)) {
    ISAVE(3) = ISAVE(3) + 1;
    goto L50;
  }
  // ITERATION COMPLETE.  FINAL STAGE.
L100:
  altsgn = one;
  for (i = 1; i <= n; ++i) {
    X(i) = T(altsgn * (one + R(i - 1) / R(n - 1)));
    altsgn = -altsgn;
  }
  *kase = 1;
  ISAVE(1) = 5;
  return;
  // X HAS BEEN OVERWRITTEN BY A*X.
L120:
  temp = two * (Sum1(n, x, 1) / R(3 * n));
  if (temp > *est) {
    Copy(n, x, 1, v, 1);
    *est = temp;
  }
L130:
  *kase = 0;
}

// xTRSYL: solves for the M-by-N matrix X in op(A) * X + ISGN * X * op(B) =
// SCALE * C for the upper (quasi-)triangular A and B in Schur canonical
// form, op(A) = A or A**H (TRANA = 'N', or 'T' (real T) / 'C'), op(B)
// likewise (TRANB), ISGN = 1 or -1, with SCALE <= 1 chosen to avoid
// overflow. X overwrites C.
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if A and -ISGN*B
// have common or close eigenvalues (they were perturbed).
template <typename T>
blasint Trsyl(char trana, char tranb, blasint isgn, blasint m, blasint n,
              const T* a, blasint lda, const T* b, blasint ldb, T* c,
              blasint ldc, RealT<T>* scale) {
  if constexpr (IsComplex<T>::value) {
    return TrsylComplex(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc,
                        scale);
  } else {
    return TrsylReal(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
  }
}

// xTREXC (real T): reorders the real Schur factorization A = Q*T*Q**T so
// that the diagonal block of T starting at row IFST moves to row ILST, by
// an orthogonal similarity transformation, and updates Q (COMPQ = 'V';
// 'N': no Q). IFST and ILST are adjusted to the first rows of 2-by-2
// blocks. WORK has N elements.
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if two adjacent
// blocks were too close to swap (ILST is then where the block stopped).
template <typename R>
blasint Trexc(char compq, blasint n, R* t, blasint ldt, R* q, blasint ldq,
              blasint* ifst, blasint* ilst, R* work) {
  using T = R;

  const R zero = R(0);
  const Matrix<R> Q(q, ldq);
  const Matrix<R> Tm(t, ldt);
  const Vector<R> WORK(work);
  blasint here, nbf, nbl, nbnext;
  bool wantq;
  blasint info = 0;
  wantq = Lsame(compq, 'V');
  if (!wantq && !Lsame(compq, 'N')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -4;
  } else if (ldq < 1 || (wantq && ldq < Max<blasint>(1, n))) {
    info = -6;
  } else if ((*ifst < 1 || *ifst > n) && (n > 0)) {
    info = -7;
  } else if ((*ilst < 1 || *ilst > n) && (n > 0)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("TREXC", -info);
    return info;
  }
  // Quick return if possible
  if (n <= 1) return info;
  // Determine the first row of specified block
  // and find out it is 1 by 1 or 2 by 2.
  if (*ifst > 1) {
    if (Tm(*ifst, *ifst - 1) != zero) *ifst = *ifst - 1;
  }
  nbf = 1;
  if (*ifst < n) {
    if (Tm(*ifst + 1, *ifst) != zero) nbf = 2;
  }
  // Determine the first row of the final block
  // and find out it is 1 by 1 or 2 by 2.
  if (*ilst > 1) {
    if (Tm(*ilst, *ilst - 1) != zero) *ilst = *ilst - 1;
  }
  nbl = 1;
  if (*ilst < n) {
    if (Tm(*ilst + 1, *ilst) != zero) nbl = 2;
  }
  if (*ifst == *ilst) return info;
  if (*ifst < *ilst) {
    // Update ILST
    if (nbf == 2 && nbl == 1) *ilst = *ilst - 1;
    if (nbf == 1 && nbl == 2) *ilst = *ilst + 1;
    here = *ifst;
  L10:
    // Swap block with next one below
    if (nbf == 1 || nbf == 2) {
      // Current block either 1 by 1 or 2 by 2
      nbnext = 1;
      if (here + nbf + 1 <= n) {
        if (Tm(here + nbf + 1, here + nbf) != zero) nbnext = 2;
      }
      info = Laexc(wantq, n, t, ldt, q, ldq, here, nbf, nbnext, work);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      here = here + nbnext;
      // Test if 2 by 2 block breaks into two 1 by 1 blocks
      if (nbf == 2) {
        if (Tm(here + 1, here) == zero) nbf = 3;
      }
    } else {
      // Current block consists of two 1 by 1 blocks each of which
      // must be swapped individually
      nbnext = 1;
      if (here + 3 <= n) {
        if (Tm(here + 3, here + 2) != zero) nbnext = 2;
      }
      info = Laexc(wantq, n, t, ldt, q, ldq, here + 1, 1, nbnext, work);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      if (nbnext == 1) {
        // Swap two 1 by 1 blocks, no problems possible
        info = Laexc(wantq, n, t, ldt, q, ldq, here, 1, nbnext, work);
        here = here + 1;
      } else {
        // Recompute NBNEXT in case 2 by 2 split
        if (Tm(here + 2, here + 1) == zero) nbnext = 1;
        if (nbnext == 2) {
          // 2 by 2 Block did not split
          info = Laexc(wantq, n, t, ldt, q, ldq, here, 1, nbnext, work);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here + 2;
        } else {
          // 2 by 2 Block did split
          info = Laexc(wantq, n, t, ldt, q, ldq, here, 1, 1, work);
          info = Laexc(wantq, n, t, ldt, q, ldq, here + 1, 1, 1, work);
          here = here + 2;
        }
      }
    }
    if (here < *ilst) goto L10;
  } else {
    here = *ifst;
  L20:
    // Swap block with next one above
    if (nbf == 1 || nbf == 2) {
      // Current block either 1 by 1 or 2 by 2
      nbnext = 1;
      if (here >= 3) {
        if (Tm(here - 1, here - 2) != zero) nbnext = 2;
      }
      info = Laexc(wantq, n, t, ldt, q, ldq, here - nbnext, nbnext, nbf, work);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      here = here - nbnext;
      // Test if 2 by 2 block breaks into two 1 by 1 blocks
      if (nbf == 2) {
        if (Tm(here + 1, here) == zero) nbf = 3;
      }
    } else {
      // Current block consists of two 1 by 1 blocks each of which
      // must be swapped individually
      nbnext = 1;
      if (here >= 3) {
        if (Tm(here - 1, here - 2) != zero) nbnext = 2;
      }
      info = Laexc(wantq, n, t, ldt, q, ldq, here - nbnext, nbnext, 1, work);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      if (nbnext == 1) {
        // Swap two 1 by 1 blocks, no problems possible
        info = Laexc(wantq, n, t, ldt, q, ldq, here, nbnext, 1, work);
        here = here - 1;
      } else {
        // Recompute NBNEXT in case 2 by 2 split
        if (Tm(here, here - 1) == zero) nbnext = 1;
        if (nbnext == 2) {
          // 2 by 2 Block did not split
          info = Laexc(wantq, n, t, ldt, q, ldq, here - 1, 2, 1, work);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here - 2;
        } else {
          // 2 by 2 Block did split
          info = Laexc(wantq, n, t, ldt, q, ldq, here, 1, 1, work);
          info = Laexc(wantq, n, t, ldt, q, ldq, here - 1, 1, 1, work);
          here = here - 2;
        }
      }
    }
    if (here > *ilst) goto L20;
  }
  *ilst = here;
  return info;
}

// xTREXC (complex T): as the real routine, without blocks.
template <typename T>
blasint Trexc(char compq, blasint n, T* t, blasint ldt, T* q, blasint ldq,
              blasint ifst, blasint ilst) {
  using R = RealT<T>;

  const Matrix<T> Q(q, ldq);
  const Matrix<T> Tm(t, ldt);
  blasint k, m1, m2, m3;
  bool wantq;
  R cs;
  T sn, t11, t22, temp;
  blasint info = 0;
  wantq = Lsame(compq, 'V');
  if (!Lsame(compq, 'N') && !wantq) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -4;
  } else if (ldq < 1 || (wantq && ldq < Max<blasint>(1, n))) {
    info = -6;
  } else if ((ifst < 1 || ifst > n) && (n > 0)) {
    info = -7;
  } else if ((ilst < 1 || ilst > n) && (n > 0)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("TREXC", -info);
    return info;
  }
  // Quick return if possible
  if (n <= 1 || ifst == ilst) return info;
  if (ifst < ilst) {
    // Move the IFST-th diagonal element forward down the diagonal.
    m1 = 0;
    m2 = -1;
    m3 = 1;
  } else {
    // Move the IFST-th diagonal element backward up the diagonal.
    m1 = -1;
    m2 = 0;
    m3 = -1;
  }
  // (M3 = 1 or -1: the loop runs up or down.)
  for (k = ifst + m1; m3 > 0 ? k <= ilst + m2 : k >= ilst + m2; k += m3) {
    // Interchange the k-th and (k+1)-th diagonal elements.
    t11 = Tm(k, k);
    t22 = Tm(k + 1, k + 1);
    // Determine the transformation to perform the interchange.
    Lartg(Tm(k, k + 1), t22 - t11, &cs, &sn, &temp);
    // Apply transformation to the matrix T.
    if (k + 2 <= n)
      Rot(n - k - 1, Tm.ptr(k, k + 2), ldt, Tm.ptr(k + 1, k + 2), ldt, cs, sn);
    Rot(k - 1, Tm.ptr(1, k), 1, Tm.ptr(1, k + 1), 1, cs, Conj(sn));
    Tm(k, k) = t22;
    Tm(k + 1, k + 1) = t11;
    if (wantq) {
      // Accumulate transformation in the matrix Q.
      Rot(n, Q.ptr(1, k), 1, Q.ptr(1, k + 1), 1, cs, Conj(sn));
    }
  }
  return info;
}

// xTRSEN (real T): reorders the real Schur factorization A = Q*T*Q**T so
// that the eigenvalues selected by SELECT (a LOGICAL array; both of a
// complex conjugate pair are moved if either is selected) are in the
// leading M-by-M block T11, updates Q (COMPQ = 'V'; 'N': no Q), and
// returns the reordered eigenvalues WR + i*WI. JOB = 'N': nothing more;
// 'E': S = the reciprocal condition number of the cluster; 'V': SEP = the
// reciprocal condition number of the invariant subspace; 'B': both. WORK
// and IWORK have LWORK and LIWORK elements (-1: workspace query), at least
// max(1, N) and 1 (JOB = 'N'), max(1, 2*M*(N-M)) and 1 ('E'), and
// max(1, 2*M*(N-M)) and max(1, M*(N-M)) ('V', 'B').
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if the reordering
// failed (two blocks were too close to swap).
template <typename R>
blasint Trsen(char job, char compq, const blasint* select, blasint n, R* t,
              blasint ldt, R* q, blasint ldq, R* wr, R* wi, blasint* m, R* s,
              R* sep, R* work, blasint lwork, blasint* iwork, blasint liwork) {
  using T = R;

  const R zero = R(0);
  const R one = R(1);
  blasint isave[3];
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> Q(q, ldq);
  const Vector<const blasint> SELECT(select);
  const Matrix<R> Tm(t, ldt);
  const Vector<R> WI(wi);
  const Vector<R> WORK(work);
  const Vector<R> WR(wr);
  blasint ierr, k, kase, kk, ks, n1, n2, nn;
  // Set below for every valid JOB.
  blasint liwmin = 0, lwmin = 0;
  bool lquery, pair, swap, wantbh, wantq, wants, wantsp;
  // SCALE is initialized: Trsyl sets it after the first Lacn2 call, which
  // the static analyzer cannot see.
  R est, rnorm, scale = R(0);
  wantbh = Lsame(job, 'B');
  wants = Lsame(job, 'E') || wantbh;
  wantsp = Lsame(job, 'V') || wantbh;
  wantq = Lsame(compq, 'V');
  blasint info = 0;
  lquery = (lwork == -1);
  if (!Lsame(job, 'N') && !wants && !wantsp) {
    info = -1;
  } else if (!Lsame(compq, 'N') && !wantq) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -8;
  } else {
    // Set M to the dimension of the specified invariant subspace,
    // and test LWORK and LIWORK.
    *m = 0;
    pair = false;
    for (k = 1; k <= n; ++k) {
      if (pair) {
        pair = false;
      } else {
        if (k < n) {
          if (Tm(k + 1, k) == zero) {
            if (SELECT(k)) *m = *m + 1;
          } else {
            pair = true;
            if (SELECT(k) || SELECT(k + 1)) *m = *m + 2;
          }
        } else {
          if (SELECT(n)) *m = *m + 1;
        }
      }
    }
    n1 = *m;
    n2 = n - *m;
    nn = n1 * n2;
    if (wantsp) {
      lwmin = Max<blasint>(1, 2 * nn);
      liwmin = Max<blasint>(1, nn);
    } else if (Lsame(job, 'N')) {
      lwmin = Max<blasint>(1, n);
      liwmin = 1;
    } else if (Lsame(job, 'E')) {
      lwmin = Max<blasint>(1, nn);
      liwmin = 1;
    }
    if (lwork < lwmin && !lquery) {
      info = -15;
    } else if (liwork < liwmin && !lquery) {
      info = -17;
    }
  }
  if (info == 0) {
    WORK(1) = WorkSize<T>(lwmin);
    IWORK(1) = liwmin;
  }
  if (info != 0) {
    Xerbla<T>("TRSEN", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible.
  if (*m == n || *m == 0) {
    if (wants) *s = one;
    if (wantsp) *sep = Lange('1', n, n, t, ldt, work);
    goto L40;
  }
  // Collect the selected blocks at the top-left corner of T.
  ks = 0;
  pair = false;
  for (k = 1; k <= n; ++k) {
    if (pair) {
      pair = false;
    } else {
      swap = SELECT(k);
      if (k < n) {
        if (Tm(k + 1, k) != zero) {
          pair = true;
          swap = swap || SELECT(k + 1);
        }
      }
      if (swap) {
        ks = ks + 1;
        // Swap the K-th block to position KS.
        ierr = 0;
        kk = k;
        if (k != ks) ierr = Trexc(compq, n, t, ldt, q, ldq, &kk, &ks, work);
        if (ierr == 1 || ierr == 2) {
          // Blocks too close to swap: exit.
          info = 1;
          if (wants) *s = zero;
          if (wantsp) *sep = zero;
          goto L40;
        }
        if (pair) ks = ks + 1;
      }
    }
  }
  if (wants) {
    // Solve Sylvester equation for R:
    // T11*R - R*T22 = scale*T12
    Lacpy('F', n1, n2, Tm.ptr(1, n1 + 1), ldt, work, n1);
    ierr = Trsyl('N', 'N', -1, n1, n2, t, ldt, Tm.ptr(n1 + 1, n1 + 1), ldt,
                 work, n1, &scale);
    // Estimate the reciprocal of the condition number of the cluster
    // of eigenvalues.
    rnorm = Lange('F', n1, n2, work, n1, work);
    if (rnorm == zero) {
      *s = one;
    } else {
      *s =
          scale / (std::sqrt(scale * scale / rnorm + rnorm) * std::sqrt(rnorm));
    }
  }
  if (wantsp) {
    // Estimate *sep(T11,T22).
    est = zero;
    kase = 0;
  L30:
    Lacn2(nn, WORK.ptr(nn + 1), work, iwork, &est, &kase, isave);
    if (kase != 0) {
      if (kase == 1) {
        // Solve  T11*R - R*T22 = scale*X.
        ierr = Trsyl('N', 'N', -1, n1, n2, t, ldt, Tm.ptr(n1 + 1, n1 + 1), ldt,
                     work, n1, &scale);
      } else {
        // Solve T11**T*R - R*T22**T = scale*X.
        ierr = Trsyl('T', 'T', -1, n1, n2, t, ldt, Tm.ptr(n1 + 1, n1 + 1), ldt,
                     work, n1, &scale);
      }
      goto L30;
    }
    *sep = scale / est;
  }
L40:
  // Store the output eigenvalues in WR and WI.
  for (k = 1; k <= n; ++k) {
    WR(k) = Tm(k, k);
    WI(k) = zero;
  }
  for (k = 1; k <= n - 1; ++k) {
    if (Tm(k + 1, k) != zero) {
      WI(k) =
          std::sqrt(std::abs(Tm(k, k + 1))) * std::sqrt(std::abs(Tm(k + 1, k)));
      WI(k + 1) = -WI(k);
    }
  }
  WORK(1) = WorkSize<T>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

// xTRSEN (complex T): as the real routine, without blocks, with the
// eigenvalues in W and without IWORK.
template <typename T>
blasint Trsen(char job, char compq, const blasint* select, blasint n, T* t,
              blasint ldt, T* q, blasint ldq, T* w, blasint* m, RealT<T>* s,
              RealT<T>* sep, T* work, blasint lwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  blasint isave[3];
  const Matrix<T> Q(q, ldq);
  R rwork[1];
  const Vector<const blasint> SELECT(select);
  const Matrix<T> Tm(t, ldt);
  const Vector<T> W(w);
  const Vector<T> WORK(work);
  blasint k, kase, ks, n1, n2, nn;
  // Set below for every valid JOB.
  blasint lwmin = 0;
  bool lquery, wantbh, wantq, wants, wantsp;
  // SCALE is initialized: Trsyl sets it after the first Lacn2 call, which
  // the static analyzer cannot see.
  R est, rnorm, scale = R(0);
  wantbh = Lsame(job, 'B');
  wants = Lsame(job, 'E') || wantbh;
  wantsp = Lsame(job, 'V') || wantbh;
  wantq = Lsame(compq, 'V');
  // Set M to the number of selected eigenvalues.
  *m = 0;
  for (k = 1; k <= n; ++k) {
    if (SELECT(k)) *m = *m + 1;
  }
  n1 = *m;
  n2 = n - *m;
  nn = n1 * n2;
  blasint info = 0;
  lquery = (lwork == -1);
  if (wantsp) {
    lwmin = Max<blasint>(1, 2 * nn);
  } else if (Lsame(job, 'N')) {
    lwmin = 1;
  } else if (Lsame(job, 'E')) {
    lwmin = Max<blasint>(1, nn);
  }
  if (!Lsame(job, 'N') && !wants && !wantsp) {
    info = -1;
  } else if (!Lsame(compq, 'N') && !wantq) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -8;
  } else if (lwork < lwmin && !lquery) {
    info = -14;
  }
  if (info == 0) {
    WORK(1) = WorkSize<T>(lwmin);
  }
  if (info != 0) {
    Xerbla<T>("TRSEN", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (*m == n || *m == 0) {
    if (wants) *s = one;
    if (wantsp) *sep = Lange('1', n, n, t, ldt, rwork);
    goto L40;
  }
  // Collect the selected eigenvalues at the top left corner of T.
  ks = 0;
  for (k = 1; k <= n; ++k) {
    if (SELECT(k)) {
      ks = ks + 1;
      // Swap the K-th eigenvalue to position KS.
      if (k != ks) Trexc(compq, n, t, ldt, q, ldq, k, ks);
    }
  }
  if (wants) {
    // Solve the Sylvester equation for R:
    // T11*R - R*T22 = scale*T12
    Lacpy('F', n1, n2, Tm.ptr(1, n1 + 1), ldt, work, n1);
    Trsyl('N', 'N', -1, n1, n2, t, ldt, Tm.ptr(n1 + 1, n1 + 1), ldt, work, n1,
          &scale);
    // Estimate the reciprocal of the condition number of the cluster
    // of eigenvalues.
    rnorm = Lange('F', n1, n2, work, n1, rwork);
    if (rnorm == zero) {
      *s = one;
    } else {
      *s =
          scale / (std::sqrt(scale * scale / rnorm + rnorm) * std::sqrt(rnorm));
    }
  }
  if (wantsp) {
    // Estimate *sep(T11,T22).
    est = zero;
    kase = 0;
  L30:
    Lacn2(nn, WORK.ptr(nn + 1), work, &est, &kase, isave);
    if (kase != 0) {
      if (kase == 1) {
        // Solve T11*R - R*T22 = scale*X.
        Trsyl('N', 'N', -1, n1, n2, t, ldt, Tm.ptr(n1 + 1, n1 + 1), ldt, work,
              n1, &scale);
      } else {
        // Solve T11**H*R - R*T22**H = scale*X.
        Trsyl('C', 'C', -1, n1, n2, t, ldt, Tm.ptr(n1 + 1, n1 + 1), ldt, work,
              n1, &scale);
      }
      goto L30;
    }
    *sep = scale / est;
  }
L40:
  // Copy reordered eigenvalues to W.
  for (k = 1; k <= n; ++k) {
    W(k) = Tm(k, k);
  }
  WORK(1) = WorkSize<T>(lwmin);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                        \
  template void Larfx<T>(char, blasint, blasint, const T*, T, T*, blasint,   \
                         T*);                                                \
  template blasint Trsyl<T>(char, char, blasint, blasint, blasint, const T*, \
                            blasint, const T*, blasint, T*, blasint,         \
                            RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template blasint Lasy2<R>(bool, bool, blasint, blasint, blasint, const R*,   \
                            blasint, const R*, blasint, const R*, blasint, R*, \
                            R*, blasint, R*);                                  \
  template blasint Laln2<R>(bool, blasint, blasint, R, R, const R*, blasint,   \
                            R, R, const R*, blasint, R, R, R*, blasint, R*,    \
                            R*);                                               \
  template blasint Laexc<R>(bool, blasint, R*, blasint, R*, blasint, blasint,  \
                            blasint, blasint, R*);                             \
  template void Lacn2<R>(blasint, R*, R*, blasint*, R*, blasint*, blasint*);   \
  template blasint Trexc<R>(char, blasint, R*, blasint, R*, blasint, blasint*, \
                            blasint*, R*);                                     \
  template blasint Trsen<R>(char, char, const blasint*, blasint, R*, blasint,  \
                            R*, blasint, R*, R*, blasint*, R*, R*, R*,         \
                            blasint, blasint*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                 \
  template void Lacn2<T>(blasint, T*, T*, RealT<T>*, blasint*, blasint*);     \
  template blasint Trexc<T>(char, blasint, T*, blasint, T*, blasint, blasint, \
                            blasint);                                         \
  template blasint Trsen<T>(char, char, const blasint*, blasint, T*, blasint, \
                            T*, blasint, T*, blasint*, RealT<T>*, RealT<T>*,  \
                            T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_TRSEN(prefix, T, R)                                     \
  HWYBLAS_API void prefix##larfx_(                                             \
      const char* side, const hwyblas_int* m, const hwyblas_int* n,            \
      const R* v, const R* tau, R* c, const hwyblas_int* ldc, R* work) {       \
    hwyblas::lapack::Larfx(*side, *m, *n, As<const T>(v), *As<const T>(tau),   \
                           As<T>(c), *ldc, As<T>(work));                       \
  }                                                                            \
  HWYBLAS_API void prefix##trsyl_(                                             \
      const char* trana, const char* tranb, const hwyblas_int* isgn,           \
      const hwyblas_int* m, const hwyblas_int* n, const R* a,                  \
      const hwyblas_int* lda, const R* b, const hwyblas_int* ldb, R* c,        \
      const hwyblas_int* ldc, R* scale, hwyblas_int* info) {                   \
    *info = hwyblas::lapack::Trsyl(*trana, *tranb, *isgn, *m, *n,              \
                                   As<const T>(a), *lda, As<const T>(b), *ldb, \
                                   As<T>(c), *ldc, scale);                     \
  }

HWYBLAS_LAPACK_TRSEN(s, float, float)
HWYBLAS_LAPACK_TRSEN(d, double, double)
HWYBLAS_LAPACK_TRSEN(c, c32, float)
HWYBLAS_LAPACK_TRSEN(z, c64, double)

#undef HWYBLAS_LAPACK_TRSEN

// LTRANL, LTRANR, LTRANS and WANTQ are Fortran LOGICALs, SELECT a LOGICAL
// array (nonzero is true).
#define HWYBLAS_LAPACK_TRSEN_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##lasy2_(                                             \
      const hwyblas_int* ltranl, const hwyblas_int* ltranr,                    \
      const hwyblas_int* isgn, const hwyblas_int* n1, const hwyblas_int* n2,   \
      const R* tl, const hwyblas_int* ldtl, const R* tr,                       \
      const hwyblas_int* ldtr, const R* b, const hwyblas_int* ldb, R* scale,   \
      R* x, const hwyblas_int* ldx, R* xnorm, hwyblas_int* info) {             \
    *info = hwyblas::lapack::Lasy2(*ltranl != 0, *ltranr != 0, *isgn, *n1,     \
                                   *n2, tl, *ldtl, tr, *ldtr, b, *ldb, scale,  \
                                   x, *ldx, xnorm);                            \
  }                                                                            \
  HWYBLAS_API void prefix##laln2_(                                             \
      const hwyblas_int* ltrans, const hwyblas_int* na, const hwyblas_int* nw, \
      const R* smin, const R* ca, const R* a, const hwyblas_int* lda,          \
      const R* d1, const R* d2, const R* b, const hwyblas_int* ldb,            \
      const R* wr, const R* wi, R* x, const hwyblas_int* ldx, R* scale,        \
      R* xnorm, hwyblas_int* info) {                                           \
    *info = hwyblas::lapack::Laln2(*ltrans != 0, *na, *nw, *smin, *ca, a,      \
                                   *lda, *d1, *d2, b, *ldb, *wr, *wi, x, *ldx, \
                                   scale, xnorm);                              \
  }                                                                            \
  HWYBLAS_API void prefix##laexc_(                                             \
      const hwyblas_int* wantq, const hwyblas_int* n, R* t,                    \
      const hwyblas_int* ldt, R* q, const hwyblas_int* ldq,                    \
      const hwyblas_int* j1, const hwyblas_int* n1, const hwyblas_int* n2,     \
      R* work, hwyblas_int* info) {                                            \
    *info = hwyblas::lapack::Laexc(*wantq != 0, *n, t, *ldt, q, *ldq, *j1,     \
                                   *n1, *n2, work);                            \
  }                                                                            \
  HWYBLAS_API void prefix##lacn2_(const hwyblas_int* n, R* v, R* x,            \
                                  hwyblas_int* isgn, R* est,                   \
                                  hwyblas_int* kase, hwyblas_int* isave) {     \
    hwyblas::lapack::Lacn2(*n, v, x, isgn, est, kase, isave);                  \
  }                                                                            \
  HWYBLAS_API void prefix##trexc_(                                             \
      const char* compq, const hwyblas_int* n, R* t, const hwyblas_int* ldt,   \
      R* q, const hwyblas_int* ldq, hwyblas_int* ifst, hwyblas_int* ilst,      \
      R* work, hwyblas_int* info) {                                            \
    *info = hwyblas::lapack::Trexc(*compq, *n, t, *ldt, q, *ldq, ifst, ilst,   \
                                   work);                                      \
  }                                                                            \
  HWYBLAS_API void prefix##trsen_(                                             \
      const char* job, const char* compq, const hwyblas_int* select,           \
      const hwyblas_int* n, R* t, const hwyblas_int* ldt, R* q,                \
      const hwyblas_int* ldq, R* wr, R* wi, hwyblas_int* m, R* s, R* sep,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* iwork,                   \
      const hwyblas_int* liwork, hwyblas_int* info) {                          \
    *info =                                                                    \
        hwyblas::lapack::Trsen(*job, *compq, select, *n, t, *ldt, q, *ldq, wr, \
                               wi, m, s, sep, work, *lwork, iwork, *liwork);   \
  }

HWYBLAS_LAPACK_TRSEN_REAL(s, float)
HWYBLAS_LAPACK_TRSEN_REAL(d, double)

#undef HWYBLAS_LAPACK_TRSEN_REAL

#define HWYBLAS_LAPACK_TRSEN_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##lacn2_(const hwyblas_int* n, R* v, R* x, R* est,    \
                                  hwyblas_int* kase, hwyblas_int* isave) {     \
    hwyblas::lapack::Lacn2(*n, As<T>(v), As<T>(x), est, kase, isave);          \
  }                                                                            \
  HWYBLAS_API void prefix##trexc_(                                             \
      const char* compq, const hwyblas_int* n, R* t, const hwyblas_int* ldt,   \
      R* q, const hwyblas_int* ldq, const hwyblas_int* ifst,                   \
      const hwyblas_int* ilst, hwyblas_int* info) {                            \
    *info = hwyblas::lapack::Trexc(*compq, *n, As<T>(t), *ldt, As<T>(q), *ldq, \
                                   *ifst, *ilst);                              \
  }                                                                            \
  HWYBLAS_API void prefix##trsen_(                                             \
      const char* job, const char* compq, const hwyblas_int* select,           \
      const hwyblas_int* n, R* t, const hwyblas_int* ldt, R* q,                \
      const hwyblas_int* ldq, R* w, hwyblas_int* m, R* s, R* sep, R* work,     \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Trsen(*job, *compq, select, *n, As<T>(t), *ldt,   \
                                   As<T>(q), *ldq, As<T>(w), m, s, sep,        \
                                   As<T>(work), *lwork);                       \
  }

HWYBLAS_LAPACK_TRSEN_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_TRSEN_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_TRSEN_COMPLEX

}  // extern "C"
