// The orthogonal (unitary) matrix Q of a tridiagonal reduction: Q itself
// (xORGTR / xUNGTR) and its product with a matrix (xORMTR / xUNMTR), and
// the routines for the matrix Q of a QL factorization that they use (xORGQL
// / xUNGQL blocked and xORG2L / xUNG2L unblocked; xORMQL / xUNMQL blocked
// and xORM2L / xUNM2L unblocked).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xORG2L / xUNG2L: generates an M-by-N matrix Q with orthonormal columns,
// defined as the last N columns of a product of K elementary reflectors of
// order M,
//
//     Q = H(k) . . . H(2) H(1)
//
// as returned by xGEQLF: on entry, column n-k+i of A holds the vector of
// H(i), with its last element at A(m-k+i, n-k+i), and TAU(i) its scalar.
// M >= N >= K >= 0. WORK has N elements.
template <typename T>
blasint Ung2l(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
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
    Xerbla<T>(RealOrComplex<T>("ORG2L", "UNG2L"), -info);
    return info;
  }

  // Quick return if possible.
  if (n <= 0) return info;

  // Initialise columns 1:n-k to columns of the unit matrix.
  for (blasint j = 1; j <= n - k; ++j) {
    for (blasint l = 1; l <= m; ++l) A(l, j) = zero;
    A(m - n + j, j) = one;
  }

  for (blasint i = 1; i <= k; ++i) {
    const blasint ii = n - k + i;

    // Apply H(i) to A(1:m-k+i,1:n-k+i) from the left. (The real routine
    // does not set the unit element, which xLARF1L does not reference.)
    A(m - n + ii, ii) = one;
    Larf1l('L', m - n + ii, ii - 1, A.ptr(1, ii), 1, TAU(i), a, lda, work);
    Scal(m - n + ii - 1, -TAU(i), A.ptr(1, ii), 1);
    A(m - n + ii, ii) = one - TAU(i);

    // Set A(m-k+i+1:m,n-k+i) to zero.
    for (blasint l = m - n + ii + 1; l <= m; ++l) A(l, ii) = zero;
  }
  return info;
}

// xORGQL / xUNGQL: generates Q like xORG2L / xUNG2L (see there), with the
// blocked algorithm: from the first block of NB reflectors (from ILAENV) to
// the last, each block reflector (backward, columnwise) is applied with
// xLARFB to the columns to its left; the first NX reflectors use the
// unblocked code. LWORK >= max(1, N); the optimal LWORK is N*NB.
template <typename T>
blasint Ungql(blasint m, blasint n, blasint k, T* a, blasint lda, const T* tau,
              T* work, blasint lwork) {
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORGQL", "UNGQL");

  // Test the input arguments.
  blasint info = 0;
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0 || n > m) {
    info = -2;
  } else if (k < 0 || k > n) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  }

  blasint nb = 0;
  if (info == 0) {
    blasint lwkopt;
    if (n == 0) {
      lwkopt = 1;
    } else {
      nb = Ilaenv<T>(1, name, " ", m, n, k, -1);
      lwkopt = n * nb;
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (lwork < Max<blasint>(1, n) && !lquery) info = -8;
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n <= 0) return info;

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

  blasint kk;
  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code after the first block. The last kk columns are
    // handled by the block method.
    kk = Min(k, ((k - nx + nb - 1) / nb) * nb);

    // Set A(m-kk+1:m,1:n-kk) to zero.
    for (blasint j = 1; j <= n - kk; ++j) {
      for (blasint i = m - kk + 1; i <= m; ++i) A(i, j) = zero;
    }
  } else {
    kk = 0;
  }

  // Use unblocked code for the first or only block.
  Ung2l(m - kk, n - kk, k - kk, a, lda, tau, work);

  if (kk > 0) {
    // Use blocked code.
    for (blasint i = k - kk + 1; i <= k; i += nb) {
      const blasint ib = Min(nb, k - i + 1);
      if (n - k + i > 1) {
        // Form the triangular factor of the block reflector
        // H = H(i+ib-1) . . . H(i+1) H(i).
        Larft('B', 'C', m - k + i + ib - 1, ib, A.ptr(1, n - k + i), lda,
              TAU.ptr(i), work, ldwork);

        // Apply H to A(1:m-k+i+ib-1,1:n-k+i-1) from the left.
        Larfb('L', 'N', 'B', 'C', m - k + i + ib - 1, n - k + i - 1, ib,
              A.ptr(1, n - k + i), lda, work, ldwork, a, lda, WORK.ptr(ib + 1),
              ldwork);
      }

      // Apply H to rows 1:m-k+i+ib-1 of current block.
      Ung2l(m - k + i + ib - 1, ib, ib, A.ptr(1, n - k + i), lda, TAU.ptr(i),
            work);

      // Set rows m-k+i+ib:m of current block to zero.
      for (blasint j = n - k + i; j <= n - k + i + ib - 1; ++j) {
        for (blasint l = m - k + i + ib; l <= m; ++l) A(l, j) = zero;
      }
    }
  }

  WORK(1) = WorkSize<T>(iws);
  return info;
}

// xORGTR / xUNGTR: generates the N-by-N orthogonal (unitary) matrix Q of
// the reduction to tridiagonal form by xSYTRD / xHETRD with the same UPLO
// (see tridiagonal.cc): from the reflectors in A and TAU (N-1 elements),
// with xORGQL / xUNGQL (UPLO = 'U') or xORGQR / xUNGQR (UPLO = 'L').
// LWORK >= max(1, N-1); the optimal LWORK is (N-1)*NB.
template <typename T>
blasint Ungtr(char uplo, blasint n, T* a, blasint lda, const T* tau, T* work,
              blasint lwork) {
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const bool lquery = (lwork == -1);
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  } else if (lwork < Max<blasint>(1, n - 1) && !lquery) {
    info = -7;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    const blasint nb = upper ? Ilaenv<T>(1, RealOrComplex<T>("ORGQL", "UNGQL"),
                                         " ", n - 1, n - 1, n - 1, -1)
                             : Ilaenv<T>(1, RealOrComplex<T>("ORGQR", "UNGQR"),
                                         " ", n - 1, n - 1, n - 1, -1);
    lwkopt = Max<blasint>(1, n - 1) * nb;
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORGTR", "UNGTR"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) {
    WORK(1) = T(1);
    return info;
  }

  if (upper) {
    // Q was determined by a call to xSYTRD / xHETRD with UPLO = 'U'.
    //
    // Shift the vectors which define the elementary reflectors one column
    // to the left, and set the last row and column of Q to those of the
    // unit matrix.
    for (blasint j = 1; j <= n - 1; ++j) {
      for (blasint i = 1; i <= j - 1; ++i) A(i, j) = A(i, j + 1);
      A(n, j) = zero;
    }
    for (blasint i = 1; i <= n - 1; ++i) A(i, n) = zero;
    A(n, n) = one;

    // Generate Q(1:n-1,1:n-1).
    Ungql(n - 1, n - 1, n - 1, a, lda, tau, work, lwork);
  } else {
    // Q was determined by a call to xSYTRD / xHETRD with UPLO = 'L'.
    //
    // Shift the vectors which define the elementary reflectors one column
    // to the right, and set the first row and column of Q to those of the
    // unit matrix.
    for (blasint j = n; j >= 2; --j) {
      A(1, j) = zero;
      for (blasint i = j + 1; i <= n; ++i) A(i, j) = A(i, j - 1);
    }
    A(1, 1) = one;
    for (blasint i = 2; i <= n; ++i) A(i, 1) = zero;
    if (n > 1) {
      // Generate Q(2:n,2:n).
      Ungqr(n - 1, n - 1, n - 1, A.ptr(2, 2), lda, tau, work, lwork);
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORM2L / xUNM2L: overwrites the M-by-N matrix C with
//
//     Q * C (SIDE = 'L', TRANS = 'N'),   Q**H * C (SIDE = 'L', TRANS = 'C'),
//     C * Q (SIDE = 'R', TRANS = 'N'),   C * Q**H (SIDE = 'R', TRANS = 'C'),
//
// (TRANS = 'T' and Q**T for real T), where Q = H(k) . . . H(2) H(1) is the
// product of K elementary reflectors as returned by xGEQLF, of order M
// (SIDE = 'L') or N (SIDE = 'R'): A(LDA, K) and TAU(K). WORK has N
// (SIDE = 'L') or M (SIDE = 'R') elements. Unblocked.
template <typename T>
blasint Unm2l(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc,
              T* work) {
  const Matrix<const T> A(a, lda);
  const Vector<const T> TAU(tau);

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
    Xerbla<T>(RealOrComplex<T>("ORM2L", "UNM2L"), -info);
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

  blasint mi = 0, ni = 0;
  if (left) {
    ni = n;
  } else {
    mi = m;
  }

  // DO I = I1, I2, I3.
  for (blasint i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
    if (left) {
      // H(i) or H(i)**H is applied to C(1:m-k+i,1:n).
      mi = m - k + i;
    } else {
      // H(i) or H(i)**H is applied to C(1:m,1:n-k+i).
      ni = n - k + i;
    }

    // Apply H(i) or H(i)**H (the real routine passes TAU(I) for both).
    const T taui = notran ? TAU(i) : Conj(TAU(i));
    Larf1l(side, mi, ni, A.ptr(1, i), 1, taui, c, ldc, work);
  }
  return info;
}

// xORMQL / xUNMQL: overwrites C with Q * C, Q**H * C, C * Q or C * Q**H
// like xORM2L / xUNM2L (see there), with the blocked algorithm: blocks of
// NB reflectors (from ILAENV, at most NBMAX = 64) are applied with xLARFT
// (backward, columnwise) and xLARFB. LWORK >= max(1, N) (SIDE = 'L') or
// max(1, M) (SIDE = 'R'); the optimal LWORK is that times NB plus TSIZE =
// (NBMAX + 1) * NBMAX for the triangular factor.
template <typename T>
blasint Unmql(char side, char trans, blasint m, blasint n, blasint k,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  const blasint nbmax = 64;
  const blasint ldt = nbmax + 1;
  const blasint tsize = ldt * nbmax;
  const Matrix<const T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("ORMQL", "UNMQL");

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

  const char opts[3] = {side, trans, '\0'};
  blasint nb = 0, lwkopt = 0;
  if (info == 0) {
    // Compute the workspace requirements.
    if (m == 0 || n == 0) {
      lwkopt = 1;
    } else {
      nb = Min(nbmax, Ilaenv<T>(1, name, opts, m, n, k, -1));
      lwkopt = nw * nb + tsize;
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (m == 0 || n == 0) return info;

  blasint nbmin = 2;
  const blasint ldwork = nw;
  if (nb > 1 && nb < k) {
    if (lwork < lwkopt) {
      nb = (lwork - tsize) / ldwork;
      nbmin = Max<blasint>(2, Ilaenv<T>(2, name, opts, m, n, k, -1));
    }
  }

  if (nb < nbmin || nb >= k) {
    // Use unblocked code.
    Unm2l(side, trans, m, n, k, a, lda, tau, c, ldc, work);
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

    blasint mi = 0, ni = 0;
    if (left) {
      ni = n;
    } else {
      mi = m;
    }

    // DO I = I1, I2, I3.
    for (blasint i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
      const blasint ib = Min(nb, k - i + 1);

      // Form the triangular factor of the block reflector
      // H = H(i+ib-1) . . . H(i+1) H(i).
      Larft('B', 'C', nq - k + i + ib - 1, ib, A.ptr(1, i), lda, TAU.ptr(i),
            WORK.ptr(iwt), ldt);
      if (left) {
        // H or H**H is applied to C(1:m-k+i+ib-1,1:n).
        mi = m - k + i + ib - 1;
      } else {
        // H or H**H is applied to C(1:m,1:n-k+i+ib-1).
        ni = n - k + i + ib - 1;
      }

      // Apply H or H**H.
      Larfb(side, trans, 'B', 'C', mi, ni, ib, A.ptr(1, i), lda, WORK.ptr(iwt),
            ldt, c, ldc, work, ldwork);
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORMTR / xUNMTR: overwrites the M-by-N matrix C with Q * C, Q**H * C,
// C * Q or C * Q**H (TRANS = 'T' and Q**T for real T), where Q is the
// orthogonal (unitary) matrix of order NQ = M (SIDE = 'L') or N (SIDE =
// 'R') of the reduction to tridiagonal form by xSYTRD / xHETRD with the
// same UPLO: the product of NQ-1 elementary reflectors in A and TAU,
// applied with xORMQL / xUNMQL (UPLO = 'U') or xORMQR / xUNMQR (UPLO =
// 'L'). LWORK >= max(1, N) (SIDE = 'L') or max(1, M) (SIDE = 'R'); the
// optimal LWORK is that times NB.
template <typename T>
blasint Unmtr(char side, char uplo, char trans, blasint m, blasint n,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  const Matrix<const T> A(a, lda);
  const Matrix<T> C(c, ldc);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const bool left = Lsame(side, 'L');
  const bool upper = Lsame(uplo, 'U');
  const bool lquery = (lwork == -1);

  // NQ is the order of Q and NW is the minimum dimension of WORK.
  const blasint nq = left ? m : n;
  const blasint nw = left ? Max<blasint>(1, n) : Max<blasint>(1, m);
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!upper && !Lsame(uplo, 'L')) {
    info = -2;
  } else if (!Lsame(trans, 'N') && !Lsame(trans, ConjTrans<T>())) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, nq)) {
    info = -7;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -10;
  } else if (lwork < nw && !lquery) {
    info = -12;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    const char opts[3] = {side, trans, '\0'};
    const char* name = upper ? RealOrComplex<T>("ORMQL", "UNMQL")
                             : RealOrComplex<T>("ORMQR", "UNMQR");
    blasint nb;
    if (left) {
      nb = Ilaenv<T>(1, name, opts, m - 1, n, m - 1, -1);
    } else {
      nb = Ilaenv<T>(1, name, opts, m, n - 1, n - 1, -1);
    }
    lwkopt = nw * nb;
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORMTR", "UNMTR"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (m == 0 || n == 0 || nq == 1) {
    WORK(1) = T(1);
    return info;
  }

  blasint mi, ni;
  if (left) {
    mi = m - 1;
    ni = n;
  } else {
    mi = m;
    ni = n - 1;
  }

  if (upper) {
    // Q was determined by a call to xSYTRD / xHETRD with UPLO = 'U'.
    Unmql(side, trans, mi, ni, nq - 1, A.ptr(1, 2), lda, tau, c, ldc, work,
          lwork);
  } else {
    // Q was determined by a call to xSYTRD / xHETRD with UPLO = 'L'.
    blasint i1, i2;
    if (left) {
      i1 = 2;
      i2 = 1;
    } else {
      i1 = 1;
      i2 = 2;
    }
    Unmqr(side, trans, mi, ni, nq - 1, A.ptr(2, 1), lda, tau, C.ptr(i1, i2),
          ldc, work, lwork);
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Ung2l<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*);                                              \
  template blasint Ungql<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*, blasint);                                     \
  template blasint Ungtr<T>(char, blasint, T*, blasint, const T*, T*,         \
                            blasint);                                         \
  template blasint Unm2l<T>(char, char, blasint, blasint, blasint, const T*,  \
                            blasint, const T*, T*, blasint, T*);              \
  template blasint Unmql<T>(char, char, blasint, blasint, blasint, const T*,  \
                            blasint, const T*, T*, blasint, T*, blasint);     \
  template blasint Unmtr<T>(char, char, char, blasint, blasint, const T*,     \
                            blasint, const T*, T*, blasint, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// `g` is "org" (real) or "ung" (complex), `m` "orm" or "unm".
#define HWYBLAS_LAPACK_UNGTR(prefix, g, m, T, R)                               \
  HWYBLAS_API void prefix##g##2l_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* work, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Ung2l(*m, *n, *k, As<T>(a), *lda,                 \
                                   As<const T>(tau), As<T>(work));             \
  }                                                                            \
  HWYBLAS_API void prefix##g##ql_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* work, const hwyblas_int* lwork, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Ungql(*m, *n, *k, As<T>(a), *lda,                 \
                                   As<const T>(tau), As<T>(work), *lwork);     \
  }                                                                            \
  HWYBLAS_API void prefix##g##tr_(                                             \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,    \
      const R* tau, R* work, const hwyblas_int* lwork, hwyblas_int* info) {    \
    *info = hwyblas::lapack::Ungtr(*uplo, *n, As<T>(a), *lda,                  \
                                   As<const T>(tau), As<T>(work), *lwork);     \
  }                                                                            \
  HWYBLAS_API void prefix##m##2l_(                                             \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, hwyblas_int* info) {                                            \
    *info = hwyblas::lapack::Unm2l(*side, *trans, *m, *n, *k, As<const T>(a),  \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work));                               \
  }                                                                            \
  HWYBLAS_API void prefix##m##ql_(                                             \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Unmql(*side, *trans, *m, *n, *k, As<const T>(a),  \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work), *lwork);                       \
  }                                                                            \
  HWYBLAS_API void prefix##m##tr_(                                             \
      const char* side, const char* uplo, const char* trans,                   \
      const hwyblas_int* m, const hwyblas_int* n, const R* a,                  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Unmtr(*side, *uplo, *trans, *m, *n,               \
                                   As<const T>(a), *lda, As<const T>(tau),     \
                                   As<T>(c), *ldc, As<T>(work), *lwork);       \
  }

HWYBLAS_LAPACK_UNGTR(s, org, orm, float, float)
HWYBLAS_LAPACK_UNGTR(d, org, orm, double, double)
HWYBLAS_LAPACK_UNGTR(c, ung, unm, c32, float)
HWYBLAS_LAPACK_UNGTR(z, ung, unm, c64, double)

#undef HWYBLAS_LAPACK_UNGTR

}  // extern "C"
