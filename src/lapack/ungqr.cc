// The orthogonal (unitary) matrix Q of a QR, LQ or RQ factorization:
// generating it (xORGQR / xUNGQR, xORGLQ / xUNGLQ, and their unblocked
// versions xORG2R / xUNG2R, xORGL2 / xUNGL2) and multiplying with it
// (xORMQR / xUNMQR, xORMLQ / xUNMLQ, xORMRQ / xUNMRQ, unblocked xORM2R /
// xUNM2R, xORML2 / xUNML2, xORMR2 / xUNMR2).
//
// Each template is the real routine for real T and the complex one for
// complex T; they differ only in conjugations and in TRANS = 'T' (real) or
// 'C' (complex) for Q**T / Q**H.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>
#include <type_traits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// xORMQR / xUNMQR, xORMLQ / xUNMLQ and xORMRQ / xUNMRQ: the maximum block
// size, and the
// leading dimension and size of the triangular factor at the end of WORK.
constexpr blasint kNbMax = 64;
constexpr blasint kLdt = kNbMax + 1;
constexpr blasint kTsize = kLdt * kNbMax;

}  // namespace

// xORG2R / xUNG2R: generates an M-by-N matrix Q with orthonormal columns,
// defined as the first N columns of a product of K elementary reflectors of
// order M,
//
//     Q = H(1) H(2) . . . H(k)
//
// as returned by xGEQRF: on entry, column i of A (1 <= i <= k) holds the
// vector of H(i), and TAU(i) its scalar; on exit, A is Q. M >= N >= K >= 0.
// WORK has N elements.
template <typename T>
blasint Ung2r(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work) {
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);

  // Test the input arguments.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0 || n > m) {
    info = -2;
  } else if (k < 0 || k > n) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORG2R", "UNG2R"), -info);
    return info;
  }

  // Quick return if possible.
  if (n <= 0) return info;

  // Initialise columns k+1:n to columns of the unit matrix.
  for (blasint j = k + 1; j <= n; ++j) {
    for (blasint l = 1; l <= m; ++l) A(l, j) = zero;
    A(j, j) = one;
  }

  for (blasint i = k; i >= 1; --i) {
    // Apply H(i) to A(i:m,i:n) from the left.
    if (i < n) {
      Larf1f('L', m - i + 1, n - i, A.ptr(i, i), 1, TAU(i), A.ptr(i, i + 1),
             lda, work);
    }
    if (i < m) Scal(m - i, -TAU(i), A.ptr(i + 1, i), 1);
    A(i, i) = one - TAU(i);

    // Set A(1:i-1,i) to zero.
    for (blasint l = 1; l <= i - 1; ++l) A(l, i) = zero;
  }
  return info;
}

// xORGQR / xUNGQR: generates Q like xORG2R / xUNG2R (see there), with the
// blocked algorithm: from the last block of NB reflectors (from ILAENV) to
// the first, each block reflector is applied with xLARFB to the columns to
// its right; the last NX columns use the unblocked code. LWORK >= max(1, N);
// the optimal LWORK is N*NB (returned in WORK(1); LWORK = -1 is a workspace
// query).
template <typename T>
blasint Ungqr(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork) {
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORGQR", "UNGQR");

  // Test the input arguments.
  blasint info = 0;
  blasint nb = Ilaenv<T>(1, name, " ", m, n, k, -1);
  const blasint lwkopt = Max<blasint>(1, n) * nb;
  WORK(1) = WorkSize<T>(lwkopt);
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0 || n > m) {
    info = -2;
  } else if (k < 0 || k > n) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n <= 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nbmin = 2;
  blasint nx = 0;
  blasint iws = n;
  blasint ldwork = 0;
  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Max<blasint>(0, Ilaenv<T>(3, name, " ", m, n, k, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = n;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and determine
        // the minimum value of NB.
        nb = lwork / ldwork;
        nbmin = Max<blasint>(2, Ilaenv<T>(2, name, " ", m, n, k, -1));
      }
    }
  }

  blasint ki = 0;
  blasint kk;
  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code after the last block. The first kk columns are
    // handled by the block method, and the rest by the unblocked method.
    ki = ((k - nx - 1) / nb) * nb;
    kk = Min(k, ki + nb);

    // Set A(1:kk,kk+1:n) to zero.
    for (blasint j = kk + 1; j <= n; ++j) {
      for (blasint i = 1; i <= kk; ++i) A(i, j) = zero;
    }
  } else {
    kk = 0;
  }

  // Use unblocked code for the last or only block.
  if (kk < n) {
    Ung2r(m - kk, n - kk, k - kk, A.ptr(kk + 1, kk + 1), lda, TAU.ptr(kk + 1),
          work);
  }

  if (kk > 0) {
    // Use blocked code.
    for (blasint i = ki + 1; i >= 1; i -= nb) {
      const blasint ib = Min(nb, k - i + 1);
      if (i + ib <= n) {
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1).
        Larft('F', 'C', m - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i), work,
              ldwork);

        // Apply H to A(i:m,i+ib:n) from the left.
        Larfb('L', 'N', 'F', 'C', m - i + 1, n - i - ib + 1, ib, A.ptr(i, i),
              lda, work, ldwork, A.ptr(i, i + ib), lda, WORK.ptr(ib + 1),
              ldwork);
      }

      // Apply H to rows i:m of the current block.
      Ung2r(m - i + 1, ib, ib, A.ptr(i, i), lda, TAU.ptr(i), work);

      // Set rows 1:i-1 of the current block to zero.
      for (blasint j = i; j <= i + ib - 1; ++j) {
        for (blasint l = 1; l <= i - 1; ++l) A(l, j) = zero;
      }
    }
  }

  WORK(1) = WorkSize<T>(iws);
  return info;
}

// xORGL2 / xUNGL2: generates an M-by-N matrix Q with orthonormal rows,
// defined as the first M rows of a product of K elementary reflectors of
// order N,
//
//     Q = H(k)**H . . . H(2)**H H(1)**H
//
// as returned by xGELQF: on entry, row i of A (1 <= i <= k) holds the
// (conjugated) vector of H(i), and TAU(i) its scalar; on exit, A is Q.
// N >= M >= K >= 0. WORK has M elements.
template <typename T>
blasint Ungl2(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work) {
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);

  // Test the input arguments.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < m) {
    info = -2;
  } else if (k < 0 || k > m) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORGL2", "UNGL2"), -info);
    return info;
  }

  // Quick return if possible.
  if (m <= 0) return info;

  if (k < m) {
    // Initialise rows k+1:m to rows of the unit matrix.
    for (blasint j = 1; j <= n; ++j) {
      for (blasint l = k + 1; l <= m; ++l) A(l, j) = zero;
      if (j > k && j <= m) A(j, j) = one;
    }
  }

  for (blasint i = k; i >= 1; --i) {
    // Apply H(i)**H to A(i:m,i:n) from the right.
    if (i < n) {
      Lacgv(n - i, A.ptr(i, i + 1), lda);
      if (i < m) {
        Larf1f('R', m - i, n - i + 1, A.ptr(i, i), lda, Conj(TAU(i)),
               A.ptr(i + 1, i), lda, work);
      }
      Scal(n - i, -TAU(i), A.ptr(i, i + 1), lda);
      Lacgv(n - i, A.ptr(i, i + 1), lda);
    }
    A(i, i) = one - Conj(TAU(i));

    // Set A(i,1:i-1) to zero.
    for (blasint l = 1; l <= i - 1; ++l) A(i, l) = zero;
  }
  return info;
}

// xORGLQ / xUNGLQ: generates Q like xORGL2 / xUNGL2 (see there), with the
// blocked algorithm (see xORGQR / xUNGQR). LWORK >= max(1, M); the optimal
// LWORK is M*NB.
template <typename T>
blasint Unglq(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork) {
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORGLQ", "UNGLQ");

  // Test the input arguments.
  blasint info = 0;
  blasint nb = Ilaenv<T>(1, name, " ", m, n, k, -1);
  const blasint lwkopt = Max<blasint>(1, m) * nb;
  WORK(1) = WorkSize<T>(lwkopt);
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < m) {
    info = -2;
  } else if (k < 0 || k > m) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (lwork < Max<blasint>(1, m) && !lquery) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (m <= 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nbmin = 2;
  blasint nx = 0;
  blasint iws = m;
  blasint ldwork = 0;
  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Max<blasint>(0, Ilaenv<T>(3, name, " ", m, n, k, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = m;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and determine
        // the minimum value of NB.
        nb = lwork / ldwork;
        nbmin = Max<blasint>(2, Ilaenv<T>(2, name, " ", m, n, k, -1));
      }
    }
  }

  blasint ki = 0;
  blasint kk;
  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code after the last block. The first kk rows are handled
    // by the block method, and the rest by the unblocked method.
    ki = ((k - nx - 1) / nb) * nb;
    kk = Min(k, ki + nb);

    // Set A(kk+1:m,1:kk) to zero.
    for (blasint j = 1; j <= kk; ++j) {
      for (blasint i = kk + 1; i <= m; ++i) A(i, j) = zero;
    }
  } else {
    kk = 0;
  }

  // Use unblocked code for the last or only block.
  if (kk < m) {
    Ungl2(m - kk, n - kk, k - kk, A.ptr(kk + 1, kk + 1), lda, TAU.ptr(kk + 1),
          work);
  }

  if (kk > 0) {
    // Use blocked code.
    for (blasint i = ki + 1; i >= 1; i -= nb) {
      const blasint ib = Min(nb, k - i + 1);
      if (i + ib <= m) {
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1).
        Larft('F', 'R', n - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i), work,
              ldwork);

        // Apply H**H to A(i+ib:m,i:n) from the right.
        Larfb('R', 'C', 'F', 'R', m - i - ib + 1, n - i + 1, ib, A.ptr(i, i),
              lda, work, ldwork, A.ptr(i + ib, i), lda, WORK.ptr(ib + 1),
              ldwork);
      }

      // Apply H**H to columns i:n of the current block.
      Ungl2(ib, n - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i), work);

      // Set columns 1:i-1 of the current block to zero.
      for (blasint j = 1; j <= i - 1; ++j) {
        for (blasint l = i; l <= i + ib - 1; ++l) A(l, j) = zero;
      }
    }
  }

  WORK(1) = WorkSize<T>(iws);
  return info;
}

// xORM2R / xUNM2R: overwrites the M-by-N matrix C with
//
//     Q * C (SIDE = 'L', TRANS = 'N'),   Q**H * C (SIDE = 'L', TRANS = 'C'),
//     C * Q (SIDE = 'R', TRANS = 'N'),   C * Q**H (SIDE = 'R', TRANS = 'C'),
//
// (TRANS = 'T' and Q**T for real T), where Q = H(1) H(2) . . . H(k) is the
// product of K elementary reflectors as returned by xGEQRF, of order M
// (SIDE = 'L') or N (SIDE = 'R'): A(LDA, K) and TAU(K). WORK has N
// (SIDE = 'L') or M (SIDE = 'R') elements. Unblocked.
template <typename T>
blasint Unm2r(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc,
              T* work) {
  const Matrix<const T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Matrix<T> C(c, ldc);

  // Test the input arguments.
  blasint info = 0;
  const bool left = Lsame(side, 'L');
  const bool notran = Lsame(trans, 'N');

  // NQ is the order of Q.
  const blasint nq = left ? m : n;
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Max<blasint>(1, nq)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORM2R", "UNM2R"), -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0 || k == 0) return info;

  blasint i1, i2, i3;
  if ((left && !notran) || (!left && notran)) {
    i1 = 1;
    i2 = k;
    i3 = 1;
  } else {
    i1 = k;
    i2 = 1;
    i3 = -1;
  }

  blasint mi = 0, ni = 0, ic = 0, jc = 0;
  if (left) {
    ni = n;
    jc = 1;
  } else {
    mi = m;
    ic = 1;
  }

  // DO I = I1, I2, I3.
  for (blasint i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
    if (left) {
      // H(i) or H(i)**H is applied to C(i:m,1:n).
      mi = m - i + 1;
      ic = i;
    } else {
      // H(i) or H(i)**H is applied to C(1:m,i:n).
      ni = n - i + 1;
      jc = i;
    }

    // Apply H(i) or H(i)**H.
    const T taui = notran ? TAU(i) : Conj(TAU(i));
    Larf1f(side, mi, ni, A.ptr(i, i), 1, taui, C.ptr(ic, jc), ldc, work);
  }
  return info;
}

// xORMQR / xUNMQR: overwrites C with Q * C, Q**H * C, C * Q or C * Q**H
// like xORM2R / xUNM2R (see there), with the blocked algorithm: blocks of
// NB reflectors (from ILAENV, at most 64) are applied with xLARFT and
// xLARFB. LWORK >= max(1, N) (SIDE = 'L') or max(1, M) (SIDE = 'R'); the
// optimal LWORK is N*NB + 65*64 (or M*NB + 65*64).
template <typename T>
blasint Unmqr(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  const Matrix<const T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Matrix<T> C(c, ldc);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORMQR", "UNMQR");

  // Test the input arguments.
  blasint info = 0;
  const bool left = Lsame(side, 'L');
  const bool notran = Lsame(trans, 'N');
  const bool lquery = (lwork == -1);

  // NQ is the order of Q and NW is the minimum dimension of WORK.
  const blasint nq = left ? m : n;
  const blasint nw = left ? Max<blasint>(1, n) : Max<blasint>(1, m);
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Max<blasint>(1, nq)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  } else if (lwork < nw && !lquery) {
    info = -12;
  }

  blasint nb = 0, lwkopt = 0;
  if (info == 0) {
    // Compute the workspace requirements.
    const char opts[3] = {side, trans, '\0'};
    nb = Min(kNbMax, Ilaenv<T>(1, name, opts, m, n, k, -1));
    lwkopt = nw * nb + kTsize;
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (m == 0 || n == 0 || k == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nbmin = 2;
  const blasint ldwork = nw;
  if (nb > 1 && nb < k) {
    if (lwork < lwkopt) {
      nb = (lwork - kTsize) / ldwork;
      const char opts[3] = {side, trans, '\0'};
      nbmin = Max<blasint>(2, Ilaenv<T>(2, name, opts, m, n, k, -1));
    }
  }

  if (nb < nbmin || nb >= k) {
    // Use unblocked code.
    Unm2r(side, trans, m, n, k, a, lda, tau, c, ldc, work);
  } else {
    // Use blocked code.
    const blasint iwt = 1 + nw * nb;
    blasint i1, i2, i3;
    if ((left && !notran) || (!left && notran)) {
      i1 = 1;
      i2 = k;
      i3 = nb;
    } else {
      i1 = ((k - 1) / nb) * nb + 1;
      i2 = 1;
      i3 = -nb;
    }

    blasint mi = 0, ni = 0, ic = 0, jc = 0;
    if (left) {
      ni = n;
      jc = 1;
    } else {
      mi = m;
      ic = 1;
    }

    // DO I = I1, I2, I3.
    for (blasint i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
      const blasint ib = Min(nb, k - i + 1);

      // Form the triangular factor of the block reflector
      // H = H(i) H(i+1) . . . H(i+ib-1).
      Larft('F', 'C', nq - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i),
            WORK.ptr(iwt), kLdt);
      if (left) {
        // H or H**H is applied to C(i:m,1:n).
        mi = m - i + 1;
        ic = i;
      } else {
        // H or H**H is applied to C(1:m,i:n).
        ni = n - i + 1;
        jc = i;
      }

      // Apply H or H**H.
      Larfb(side, trans, 'F', 'C', mi, ni, ib, A.ptr(i, i), lda, WORK.ptr(iwt),
            kLdt, C.ptr(ic, jc), ldc, work, ldwork);
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORML2 / xUNML2: overwrites the M-by-N matrix C with Q * C, Q**H * C,
// C * Q or C * Q**H (see xORM2R / xUNM2R), where
// Q = H(k)**H . . . H(2)**H H(1)**H is the product of K elementary
// reflectors as returned by xGELQF: A(LDA, M) (SIDE = 'L') or A(LDA, N)
// (SIDE = 'R'), with LDA >= max(1, K), and TAU(K). Unblocked.
template <typename T>
blasint Unml2(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work) {
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Matrix<T> C(c, ldc);

  // Test the input arguments.
  blasint info = 0;
  const bool left = Lsame(side, 'L');
  const bool notran = Lsame(trans, 'N');

  // NQ is the order of Q.
  const blasint nq = left ? m : n;
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Max<blasint>(1, k)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORML2", "UNML2"), -info);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0 || k == 0) return info;

  blasint i1, i2, i3;
  if ((left && notran) || (!left && !notran)) {
    i1 = 1;
    i2 = k;
    i3 = 1;
  } else {
    i1 = k;
    i2 = 1;
    i3 = -1;
  }

  blasint mi = 0, ni = 0, ic = 0, jc = 0;
  if (left) {
    ni = n;
    jc = 1;
  } else {
    mi = m;
    ic = 1;
  }

  // DO I = I1, I2, I3.
  for (blasint i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
    if (left) {
      // H(i) or H(i)**H is applied to C(i:m,1:n).
      mi = m - i + 1;
      ic = i;
    } else {
      // H(i) or H(i)**H is applied to C(1:m,i:n).
      ni = n - i + 1;
      jc = i;
    }

    // Apply H(i) or H(i)**H.
    const T taui = notran ? Conj(TAU(i)) : TAU(i);
    if (i < nq) Lacgv(nq - i, A.ptr(i, i + 1), lda);
    Larf1f(side, mi, ni, A.ptr(i, i), lda, taui, C.ptr(ic, jc), ldc, work);
    if (i < nq) Lacgv(nq - i, A.ptr(i, i + 1), lda);
  }
  return info;
}

// xORMLQ / xUNMLQ: overwrites C like xORML2 / xUNML2 (see there), with the
// blocked algorithm (see xORMQR / xUNMQR).
template <typename T>
blasint Unmlq(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Matrix<T> C(c, ldc);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORMLQ", "UNMLQ");

  // Test the input arguments.
  blasint info = 0;
  const bool left = Lsame(side, 'L');
  const bool notran = Lsame(trans, 'N');
  const bool lquery = (lwork == -1);

  // NQ is the order of Q and NW is the minimum dimension of WORK.
  const blasint nq = left ? m : n;
  const blasint nw = left ? Max<blasint>(1, n) : Max<blasint>(1, m);
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Max<blasint>(1, k)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  } else if (lwork < nw && !lquery) {
    info = -12;
  }

  blasint nb = 0, lwkopt = 0;
  if (info == 0) {
    // Compute the workspace requirements.
    const char opts[3] = {side, trans, '\0'};
    nb = Min(kNbMax, Ilaenv<T>(1, name, opts, m, n, k, -1));
    lwkopt = nw * nb + kTsize;
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (m == 0 || n == 0 || k == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nbmin = 2;
  const blasint ldwork = nw;
  if (nb > 1 && nb < k) {
    if (lwork < lwkopt) {
      nb = (lwork - kTsize) / ldwork;
      const char opts[3] = {side, trans, '\0'};
      nbmin = Max<blasint>(2, Ilaenv<T>(2, name, opts, m, n, k, -1));
    }
  }

  if (nb < nbmin || nb >= k) {
    // Use unblocked code.
    Unml2(side, trans, m, n, k, a, lda, tau, c, ldc, work);
  } else {
    // Use blocked code.
    const blasint iwt = 1 + nw * nb;
    blasint i1, i2, i3;
    if ((left && notran) || (!left && !notran)) {
      i1 = 1;
      i2 = k;
      i3 = nb;
    } else {
      i1 = ((k - 1) / nb) * nb + 1;
      i2 = 1;
      i3 = -nb;
    }

    blasint mi = 0, ni = 0, ic = 0, jc = 0;
    if (left) {
      ni = n;
      jc = 1;
    } else {
      mi = m;
      ic = 1;
    }

    const char transt = notran ? ConjTrans<T>() : 'N';

    // DO I = I1, I2, I3.
    for (blasint i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
      const blasint ib = Min(nb, k - i + 1);

      // Form the triangular factor of the block reflector
      // H = H(i) H(i+1) . . . H(i+ib-1).
      Larft('F', 'R', nq - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i),
            WORK.ptr(iwt), kLdt);
      if (left) {
        // H or H**H is applied to C(i:m,1:n).
        mi = m - i + 1;
        ic = i;
      } else {
        // H or H**H is applied to C(1:m,i:n).
        ni = n - i + 1;
        jc = i;
      }

      // Apply H or H**H.
      Larfb(side, transt, 'F', 'R', mi, ni, ib, A.ptr(i, i), lda, WORK.ptr(iwt),
            kLdt, C.ptr(ic, jc), ldc, work, ldwork);
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORMR2 / xUNMR2: overwrites the M-by-N matrix C with Q * C, Q**H * C,
// C * Q or C * Q**H (see xORM2R / xUNM2R), where
// Q = H(1)**H H(2)**H . . . H(k)**H is the product of K elementary
// reflectors as returned by xGERQF: A(LDA, M) (SIDE = 'L') or A(LDA, N)
// (SIDE = 'R'), with LDA >= max(1, K), and TAU(K). Unblocked. DORMR2 (but
// not SORMR2) applies H(i) with xLARF after setting the unit element of
// its vector in A, the other three with xLARF1L, which takes it as 1; the
// two can round differently, so the double routine keeps xLARF.
template <typename T>
blasint Unmr2(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work) {
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  blasint i, i1, i2, i3, nq;
  // MI and NI are set in the loop where they are not set here.
  blasint mi = 0, ni = 0;
  bool left, notran;
  T taui;

  // Test the input arguments
  blasint info = 0;
  left = Lsame(side, 'L');
  notran = Lsame(trans, 'N');

  // NQ is the order of Q
  if (left) {
    nq = m;
  } else {
    nq = n;
  }
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Max<blasint>(1, k)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORMR2", "UNMR2"), -info);
    return info;
  }

  // Quick return if possible
  if (m == 0 || n == 0 || k == 0) return info;

  if ((left && !notran) || (!left && notran)) {
    i1 = 1;
    i2 = k;
    i3 = 1;
  } else {
    i1 = k;
    i2 = 1;
    i3 = -1;
  }

  if (left) {
    ni = n;
  } else {
    mi = m;
  }

  for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
    if (left) {
      // H(i) or H(i)**H is applied to C(1:m-k+i,1:n)
      mi = m - k + i;
    } else {
      // H(i) or H(i)**H is applied to C(1:m,1:n-k+i)
      ni = n - k + i;
    }

    // Apply H(i) or H(i)**H
    if (notran) {
      taui = Conj(TAU(i));
    } else {
      taui = TAU(i);
    }
    if constexpr (std::is_same_v<T, double>) {
      const T aii = A(i, nq - k + i);
      A(i, nq - k + i) = T(1);
      Larf(side, mi, ni, A.ptr(i, 1), lda, taui, c, ldc, work);
      A(i, nq - k + i) = aii;
    } else {
      Lacgv(nq - k + i - 1, A.ptr(i, 1), lda);
      Larf1l(side, mi, ni, A.ptr(i, 1), lda, taui, c, ldc, work);
      Lacgv(nq - k + i - 1, A.ptr(i, 1), lda);
    }
  }
  return info;
}

// xORMRQ / xUNMRQ: overwrites C like xORMR2 / xUNMR2 (see there), with the
// blocked algorithm (see xORMQR / xUNMQR).
template <typename T>
blasint Unmrq(char side, char trans, blasint m, blasint n, blasint k, T* a,
              blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  const Vector<const T> TAU(tau);
  const Matrix<T> A(a, lda);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORMRQ", "UNMRQ");
  const char opts[3] = {side, trans, '\0'};
  blasint i, i1, i2, i3, ib, iwt, ldwork, lwkopt, nbmin, nq, nw;
  // NB is set where M > 0 and N > 0 (the only case that reads it); MI and
  // NI in the loop where they are not set here.
  blasint nb = 0, mi = 0, ni = 0;
  bool left, lquery, notran;
  char transt;

  // Test the input arguments
  blasint info = 0;
  left = Lsame(side, 'L');
  notran = Lsame(trans, 'N');
  lquery = (lwork == -1);

  // NQ is the order of Q and NW is the minimum dimension of WORK
  if (left) {
    nq = m;
    nw = Max<blasint>(1, n);
  } else {
    nq = n;
    nw = Max<blasint>(1, m);
  }
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Max<blasint>(1, k)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  } else if (lwork < nw && !lquery) {
    info = -12;
  }

  lwkopt = 1;
  if (info == 0) {
    // Compute the workspace requirements
    if (m == 0 || n == 0) {
      lwkopt = 1;
    } else {
      nb = Min(kNbMax, Ilaenv<T>(1, name, opts, m, n, k, -1));
      lwkopt = nw * nb + kTsize;
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible
  if (m == 0 || n == 0) return info;

  nbmin = 2;
  ldwork = nw;
  if (nb > 1 && nb < k) {
    if (lwork < lwkopt) {
      nb = (lwork - kTsize) / ldwork;
      nbmin = Max<blasint>(2, Ilaenv<T>(2, name, opts, m, n, k, -1));
    }
  }

  if (nb < nbmin || nb >= k) {
    // Use unblocked code
    Unmr2(side, trans, m, n, k, a, lda, tau, c, ldc, work);
  } else {
    // Use blocked code
    iwt = 1 + nw * nb;
    if ((left && !notran) || (!left && notran)) {
      i1 = 1;
      i2 = k;
      i3 = nb;
    } else {
      i1 = ((k - 1) / nb) * nb + 1;
      i2 = 1;
      i3 = -nb;
    }

    if (left) {
      ni = n;
    } else {
      mi = m;
    }

    if (notran) {
      transt = ConjTrans<T>();
    } else {
      transt = 'N';
    }

    for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
      ib = Min(nb, k - i + 1);

      // Form the triangular factor of the block reflector
      // H = H(i+ib-1) . . . H(i+1) H(i)
      Larft('B', 'R', nq - k + i + ib - 1, ib, A.ptr(i, 1), lda, TAU.ptr(i),
            WORK.ptr(iwt), kLdt);
      if (left) {
        // H or H**H is applied to C(1:m-k+i+ib-1,1:n)
        mi = m - k + i + ib - 1;
      } else {
        // H or H**H is applied to C(1:m,1:n-k+i+ib-1)
        ni = n - k + i + ib - 1;
      }

      // Apply H or H**H
      Larfb(side, transt, 'B', 'R', mi, ni, ib, A.ptr(i, 1), lda, WORK.ptr(iwt),
            kLdt, c, ldc, work, ldwork);
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Ung2r<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*);                                              \
  template blasint Ungqr<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*, blasint);                                     \
  template blasint Ungl2<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*);                                              \
  template blasint Unglq<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*, blasint);                                     \
  template blasint Unm2r<T>(char, char, blasint, blasint, blasint, const T*,  \
                            blasint, const T*, T*, blasint, T*);              \
  template blasint Unmqr<T>(char, char, blasint, blasint, blasint, const T*,  \
                            blasint, const T*, T*, blasint, T*, blasint);     \
  template blasint Unml2<T>(char, char, blasint, blasint, blasint, T*,        \
                            blasint, const T*, T*, blasint, T*);              \
  template blasint Unmlq<T>(char, char, blasint, blasint, blasint, T*,        \
                            blasint, const T*, T*, blasint, T*, blasint);     \
  template blasint Unmr2<T>(char, char, blasint, blasint, blasint, T*,        \
                            blasint, const T*, T*, blasint, T*);              \
  template blasint Unmrq<T>(char, char, blasint, blasint, blasint, T*,        \
                            blasint, const T*, T*, blasint, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

namespace {

// A of xUNML2, xUNMLQ, xUNMR2 and xUNMRQ is const in the interface (as in the
// reference's lapack.h), but the complex routines conjugate rows of it while
// they run; they restore them on exit.
template <typename T, typename R>
T* Mutable(const R* a) {
  return const_cast<T*>(As<const T>(a));
}

}  // namespace

extern "C" {

// xORGQR etc. for real T, xUNGQR etc. for complex T: `gen` and `mul` are
// the middle of the names ("org" / "ung", "orm" / "unm").
#define HWYBLAS_LAPACK_UNGQR(prefix, gen, mul, T, R)                           \
  HWYBLAS_API void prefix##gen##2r_(                                           \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* work, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Ung2r(*m, *n, *k, As<T>(a), *lda,                 \
                                   As<const T>(tau), As<T>(work));             \
  }                                                                            \
  HWYBLAS_API void prefix##gen##qr_(                                           \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* work, const hwyblas_int* lwork, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Ungqr(*m, *n, *k, As<T>(a), *lda,                 \
                                   As<const T>(tau), As<T>(work), *lwork);     \
  }                                                                            \
  HWYBLAS_API void prefix##gen##l2_(                                           \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* work, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Ungl2(*m, *n, *k, As<T>(a), *lda,                 \
                                   As<const T>(tau), As<T>(work));             \
  }                                                                            \
  HWYBLAS_API void prefix##gen##lq_(                                           \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* work, const hwyblas_int* lwork, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Unglq(*m, *n, *k, As<T>(a), *lda,                 \
                                   As<const T>(tau), As<T>(work), *lwork);     \
  }                                                                            \
  HWYBLAS_API void prefix##mul##2r_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, hwyblas_int* info) {                                            \
    *info = hwyblas::lapack::Unm2r(*side, *trans, *m, *n, *k, As<const T>(a),  \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work));                               \
  }                                                                            \
  HWYBLAS_API void prefix##mul##qr_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Unmqr(*side, *trans, *m, *n, *k, As<const T>(a),  \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work), *lwork);                       \
  }                                                                            \
  HWYBLAS_API void prefix##mul##l2_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, hwyblas_int* info) {                                            \
    *info =                                                                    \
        hwyblas::lapack::Unml2(*side, *trans, *m, *n, *k, Mutable<T>(a), *lda, \
                               As<const T>(tau), As<T>(c), *ldc, As<T>(work)); \
  }                                                                            \
  HWYBLAS_API void prefix##mul##lq_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Unmlq(*side, *trans, *m, *n, *k, Mutable<T>(a),   \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work), *lwork);                       \
  }                                                                            \
  HWYBLAS_API void prefix##mul##r2_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, hwyblas_int* info) {                                            \
    *info =                                                                    \
        hwyblas::lapack::Unmr2(*side, *trans, *m, *n, *k, Mutable<T>(a), *lda, \
                               As<const T>(tau), As<T>(c), *ldc, As<T>(work)); \
  }                                                                            \
  HWYBLAS_API void prefix##mul##rq_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Unmrq(*side, *trans, *m, *n, *k, Mutable<T>(a),   \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work), *lwork);                       \
  }

HWYBLAS_LAPACK_UNGQR(s, org, orm, float, float)
HWYBLAS_LAPACK_UNGQR(d, org, orm, double, double)
HWYBLAS_LAPACK_UNGQR(c, ung, unm, c32, float)
HWYBLAS_LAPACK_UNGQR(z, ung, unm, c64, double)

#undef HWYBLAS_LAPACK_UNGQR

}  // extern "C"
