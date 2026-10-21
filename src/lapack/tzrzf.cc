// The RZ factorization of an upper trapezoidal matrix: xTZRZF (blocked) and
// xLATRZ (unblocked), the elementary reflectors of the form of xLARZ, their
// block form (xLARZT, xLARZB), and multiplying with Z (xORMRZ / xUNMRZ,
// unblocked xORMR3 / xUNMR3).
//
// Each template is the real routine for real T and the complex one for
// complex T; they differ only in conjugations and in TRANS = 'T' (real) or
// 'C' (complex) for Z**T / Z**H.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// xORMRZ / xUNMRZ: the maximum block size, and the leading dimension and
// size of the triangular factor at the end of WORK.
constexpr blasint kNbMax = 64;
constexpr blasint kLdt = kNbMax + 1;
constexpr blasint kTsize = kLdt * kNbMax;

}  // namespace

// xLARZ: applies the elementary reflector H = I - tau * v * v**H to the
// M-by-N matrix C from the left (SIDE = 'L') or the right ('R'), where v
// is 1 in its first element, zero in the next M-L (N-L) and V(1:L) in the
// last L. For the complex routine, H**H is applied by passing conj(tau).
// WORK has N (SIDE = 'L') or M (SIDE = 'R') elements.
template <typename T>
void Larz(char side, blasint m, blasint n, blasint l, const T* v, blasint incv,
          T tau, T* c, blasint ldc, T* work) {
  const T one = T(1);
  const T zero = T(0);
  const Matrix<T> C(c, ldc);

  if (Lsame(side, 'L')) {
    // Form  H * C
    if (tau != zero) {
      // w( 1:n ) = conjg( C( 1, 1:n ) )
      Copy(n, c, ldc, work, 1);
      Lacgv(n, work, 1);

      // w( 1:n ) = conjg( w( 1:n ) + C( m-l+1:m, 1:n )**H * v( 1:l ) )
      Gemv(ConjTrans<T>(), l, n, one, C.ptr(m - l + 1, 1), ldc, v, incv, one,
           work, 1);
      Lacgv(n, work, 1);

      // C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n )
      Axpy(n, -tau, work, 1, c, ldc);

      // C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
      //                     tau * v( 1:l ) * w( 1:n )**H
      Geru(l, n, -tau, v, incv, work, 1, C.ptr(m - l + 1, 1), ldc);
    }
  } else {
    // Form  C * H
    if (tau != zero) {
      // w( 1:m ) = C( 1:m, 1 )
      Copy(m, c, 1, work, 1);

      // w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l )
      Gemv('N', m, l, one, C.ptr(1, n - l + 1), ldc, v, incv, one, work, 1);

      // C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m )
      Axpy(m, -tau, work, 1, c, 1);

      // C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
      //                     tau * w( 1:m ) * v( 1:l )**H
      Gerc(m, l, -tau, work, 1, v, incv, C.ptr(1, n - l + 1), ldc);
    }
  }
}

// xLARZB: applies the block reflector H = I - V**H * T * V (DIRECT = 'B',
// STOREV = 'R', the only supported options) or H**H (TRANS = 'C' / 'T')
// to the M-by-N matrix C from the left or the right. V is K-by-L (its
// identity part is not stored), T the K-by-K lower triangular factor of
// xLARZT. WORK is LDWORK-by-K, LDWORK >= max(1, N) (SIDE = 'L') or max(1,
// M) (SIDE = 'R'). The complex routine conjugates T and V while it runs and
// restores them.
template <typename T>
void Larzb(char side, char trans, char direct, char storev, blasint m,
           blasint n, blasint k, blasint l, T* v, blasint ldv, T* t,
           blasint ldt, T* c, blasint ldc, T* work, blasint ldwork) {
  const T one = T(1);
  const Matrix<T> C(c, ldc);
  const Matrix<T> Tm(t, ldt);
  const Matrix<T> V(v, ldv);
  const Matrix<T> WORK(work, ldwork);
  blasint i, info, j;
  char transt;

  // Quick return if possible
  if (m <= 0 || n <= 0) return;

  // Check for currently supported options
  info = 0;
  if (!Lsame(direct, 'B')) {
    info = -3;
  } else if (!Lsame(storev, 'R')) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("LARZB", -info);
    return;
  }

  if (Lsame(trans, 'N')) {
    transt = ConjTrans<T>();
  } else {
    transt = 'N';
  }

  if (Lsame(side, 'L')) {
    // Form  H * C  or  H**H * C

    // W( 1:n, 1:k ) = C( 1:k, 1:n )**H
    for (j = 1; j <= k; ++j) {
      Copy(n, C.ptr(j, 1), ldc, WORK.ptr(1, j), 1);
    }

    // W( 1:n, 1:k ) = W( 1:n, 1:k ) + ...
    //                 C( m-l+1:m, 1:n )**H * V( 1:k, 1:l )**T
    if (l > 0) {
      Gemm('T', ConjTrans<T>(), n, k, l, one, C.ptr(m - l + 1, 1), ldc, v, ldv,
           one, work, ldwork);
    }

    // W( 1:n, 1:k ) = W( 1:n, 1:k ) * T**T  or  W( 1:m, 1:k ) * T
    Trmm('R', 'L', transt, 'N', n, k, one, t, ldt, work, ldwork);

    // C( 1:k, 1:n ) = C( 1:k, 1:n ) - W( 1:n, 1:k )**H
    for (j = 1; j <= n; ++j) {
      for (i = 1; i <= k; ++i) {
        C(i, j) = C(i, j) - WORK(j, i);
      }
    }

    // C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ...
    //                     V( 1:k, 1:l )**H * W( 1:n, 1:k )**H
    if (l > 0) {
      Gemm('T', 'T', l, n, k, -one, v, ldv, work, ldwork, one,
           C.ptr(m - l + 1, 1), ldc);
    }
  } else if (Lsame(side, 'R')) {
    // Form  C * H  or  C * H**H

    // W( 1:m, 1:k ) = C( 1:m, 1:k )
    for (j = 1; j <= k; ++j) {
      Copy(m, C.ptr(1, j), 1, WORK.ptr(1, j), 1);
    }

    // W( 1:m, 1:k ) = W( 1:m, 1:k ) + ...
    //                 C( 1:m, n-l+1:n ) * V( 1:k, 1:l )**H
    if (l > 0) {
      Gemm('N', 'T', m, k, l, one, C.ptr(1, n - l + 1), ldc, v, ldv, one, work,
           ldwork);
    }

    // W( 1:m, 1:k ) = W( 1:m, 1:k ) * conjg( T )  or
    //                 W( 1:m, 1:k ) * T**H
    for (j = 1; j <= k; ++j) {
      Lacgv(k - j + 1, Tm.ptr(j, j), 1);
    }
    Trmm('R', 'L', trans, 'N', m, k, one, t, ldt, work, ldwork);
    for (j = 1; j <= k; ++j) {
      Lacgv(k - j + 1, Tm.ptr(j, j), 1);
    }

    // C( 1:m, 1:k ) = C( 1:m, 1:k ) - W( 1:m, 1:k )
    for (j = 1; j <= k; ++j) {
      for (i = 1; i <= m; ++i) {
        C(i, j) = C(i, j) - WORK(i, j);
      }
    }

    // C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ...
    //                     W( 1:m, 1:k ) * conjg( V( 1:k, 1:l ) )
    for (j = 1; j <= l; ++j) {
      Lacgv(k, V.ptr(1, j), 1);
    }
    if (l > 0) {
      Gemm('N', 'N', m, l, k, -one, work, ldwork, v, ldv, one,
           C.ptr(1, n - l + 1), ldc);
    }
    for (j = 1; j <= l; ++j) {
      Lacgv(k, V.ptr(1, j), 1);
    }
  }
}

// xLARZT: forms the K-by-K lower triangular factor T of the block
// reflector H = H(1) H(2) . . . H(k) of order N, H = I - V**H * T * V
// (DIRECT = 'B', STOREV = 'R', the only supported options; then H =
// H(k) . . . H(2) H(1)), where the K-by-N matrix V holds the vectors of
// xLARZ by rows. The complex routine conjugates rows of V while it runs
// and restores them.
template <typename T>
void Larzt(char direct, char storev, blasint n, blasint k, T* v, blasint ldv,
           const T* tau, T* t, blasint ldt) {
  const T zero = T(0);
  const Matrix<T> Tm(t, ldt);
  const Vector<const T> TAU(tau);
  const Matrix<T> V(v, ldv);
  blasint i, info, j;

  // Check for currently supported options
  info = 0;
  if (!Lsame(direct, 'B')) {
    info = -1;
  } else if (!Lsame(storev, 'R')) {
    info = -2;
  }
  if (info != 0) {
    Xerbla<T>("LARZT", -info);
    return;
  }

  for (i = k; i >= 1; --i) {
    if (TAU(i) == zero) {
      // H(i)  =  I
      for (j = i; j <= k; ++j) {
        Tm(j, i) = zero;
      }
    } else {
      // general case
      if (i < k) {
        // T(i+1:k,i) = - tau(i) * V(i+1:k,1:n) * V(i,1:n)**H
        Lacgv(n, V.ptr(i, 1), ldv);
        Gemv('N', k - i, n, -TAU(i), V.ptr(i + 1, 1), ldv, V.ptr(i, 1), ldv,
             zero, Tm.ptr(i + 1, i), 1);
        Lacgv(n, V.ptr(i, 1), ldv);

        // T(i+1:k,i) = T(i+1:k,i+1:k) * T(i+1:k,i)
        Trmv('L', 'N', 'N', k - i, Tm.ptr(i + 1, i + 1), ldt, Tm.ptr(i + 1, i),
             1);
      }
      Tm(i, i) = TAU(i);
    }
  }
}

// xLATRZ: factors the M-by-(M+L) upper trapezoidal matrix [ A1 A2 ] =
// [ A(1:M,1:M) A(1:M,N-L+1:N) ] as ( R 0 ) * Z with the unblocked
// algorithm, where Z is a product of M elementary reflectors of the form
// of xLARZ. On exit, R is in the leading M-by-M upper triangle of A, the
// vectors in A(1:M,N-L+1:N), their scalars in TAU. WORK has M elements.
template <typename T>
void Latrz(blasint m, blasint n, blasint l, T* a, blasint lda, T* tau,
           T* work) {
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  blasint i;
  T alpha;

  // Quick return if possible
  if (m == 0) {
    return;
  } else if (m == n) {
    for (i = 1; i <= n; ++i) {
      TAU(i) = zero;
    }
    return;
  }

  for (i = m; i >= 1; --i) {
    // Generate elementary reflector H(i) to annihilate
    // [ A(i,i) A(i,n-l+1:n) ]
    Lacgv(l, A.ptr(i, n - l + 1), lda);
    alpha = Conj(A(i, i));
    Larfg(l + 1, &alpha, A.ptr(i, n - l + 1), lda, TAU.ptr(i));
    TAU(i) = Conj(TAU(i));

    // Apply H(i) to A(1:i-1,i:n) from the right
    Larz('R', i - 1, n - i + 1, l, A.ptr(i, n - l + 1), lda, Conj(TAU(i)),
         A.ptr(1, i), lda, work);
    A(i, i) = Conj(alpha);
  }
}

// xTZRZF: reduces the M-by-N (M <= N) upper trapezoidal matrix A to upper
// triangular form, A = ( R 0 ) * Z (see xLATRZ for the result), with the
// blocked algorithm: panels of NB rows from the bottom are factored with
// xLATRZ and applied to the rows above with xLARZT and xLARZB. LWORK >=
// max(1, M) (1 if M = 0 or M = N); the optimal LWORK is M*NB. LWORK = -1
// is a workspace query.
template <typename T>
blasint Tzrzf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork) {
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);
  blasint i, ib, iws, ki, kk, ldwork, lwkmin, lwkopt, m1, mu, nbmin, nx;
  // Set where M > 0 and M < N (the only case that reads it).
  blasint nb = 0;
  bool lquery;

  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < m) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }

  lwkopt = 1;
  if (info == 0) {
    if (m == 0 || m == n) {
      lwkopt = 1;
      lwkmin = 1;
    } else {
      // Determine the block size.
      nb = Ilaenv<T>(1, "GERQF", " ", m, n, -1, -1);
      lwkopt = m * nb;
      lwkmin = Max<blasint>(1, m);
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (lwork < lwkmin && !lquery) info = -7;
  }

  if (info != 0) {
    Xerbla<T>("TZRZF", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible
  if (m == 0) {
    return info;
  } else if (m == n) {
    for (i = 1; i <= n; ++i) {
      TAU(i) = zero;
    }
    return info;
  }

  nbmin = 2;
  nx = 1;
  iws = m;
  ldwork = 0;
  if (nb > 1 && nb < m) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Max<blasint>(0, Ilaenv<T>(3, "GERQF", " ", m, n, -1, -1));
    if (nx < m) {
      // Determine if workspace is large enough for blocked code.
      ldwork = m;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB:  reduce NB and
        // determine the minimum value of NB.
        nb = lwork / ldwork;
        nbmin = Max<blasint>(2, Ilaenv<T>(2, "GERQF", " ", m, n, -1, -1));
      }
    }
  }

  if (nb >= nbmin && nb < m && nx < m) {
    // Use blocked code initially.
    // The last kk rows are handled by the block method.
    m1 = Min(m + 1, n);
    ki = ((m - nx - 1) / nb) * nb;
    kk = Min(m, ki + nb);

    for (i = m - kk + ki + 1; i >= m - kk + 1; i -= nb) {
      ib = Min(m - i + 1, nb);

      // Compute the TZ factorization of the current block
      // A(i:i+ib-1,i:n)
      Latrz(ib, n - i + 1, n - m, A.ptr(i, i), lda, TAU.ptr(i), work);
      if (i > 1) {
        // Form the triangular factor of the block reflector
        // H = H(i+ib-1) . . . H(i+1) H(i)
        Larzt('B', 'R', n - m, ib, A.ptr(i, m1), lda, TAU.ptr(i), work, ldwork);

        // Apply H to A(1:i-1,i:n) from the right
        Larzb('R', 'N', 'B', 'R', i - 1, n - i + 1, ib, n - m, A.ptr(i, m1),
              lda, work, ldwork, A.ptr(1, i), lda, WORK.ptr(ib + 1), ldwork);
      }
    }
    mu = i + nb - 1;
  } else {
    mu = m;
  }

  // Use unblocked code to factor the last or only block
  if (mu > 0) Latrz(mu, n, n - m, a, lda, tau, work);

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORMR3 / xUNMR3: overwrites the M-by-N matrix C with Z * C, Z**H * C,
// C * Z or C * Z**H (TRANS = 'N' or 'T' / 'C'), where
// Z = H(1) H(2) . . . H(k) is the product of K elementary reflectors as
// returned by xTZRZF, whose vectors (L elements each) are in A(1:K,
// M-L+1:M) (SIDE = 'L') or A(1:K, N-L+1:N) (SIDE = 'R'). Unblocked; WORK
// has N (SIDE = 'L') or M (SIDE = 'R') elements.
template <typename T>
blasint Unmr3(char side, char trans, blasint m, blasint n, blasint k, blasint l,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc,
              T* work) {
  const Matrix<const T> A(a, lda);
  const Matrix<T> C(c, ldc);
  const Vector<const T> TAU(tau);
  blasint i, i1, i2, i3, ja, nq;
  // Set in the loop where they are not set here.
  blasint ic = 0, jc = 0, mi = 0, ni = 0;
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
  } else if (l < 0 || (left && (l > m)) || (!left && (l > n))) {
    info = -6;
  } else if (lda < Max<blasint>(1, k)) {
    info = -8;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORMR3", "UNMR3"), -info);
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
    ja = m - l + 1;
    jc = 1;
  } else {
    mi = m;
    ja = n - l + 1;
    ic = 1;
  }

  for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
    if (left) {
      // H(i) or H(i)**H is applied to C(i:m,1:n)
      mi = m - i + 1;
      ic = i;
    } else {
      // H(i) or H(i)**H is applied to C(1:m,i:n)
      ni = n - i + 1;
      jc = i;
    }

    // Apply H(i) or H(i)**H
    if (notran) {
      taui = TAU(i);
    } else {
      taui = Conj(TAU(i));
    }
    Larz(side, mi, ni, l, A.ptr(i, ja), lda, taui, C.ptr(ic, jc), ldc, work);
  }
  return info;
}

// xORMRZ / xUNMRZ: overwrites C like xORMR3 / xUNMR3 (see there), with the
// blocked algorithm: NB reflectors (ILAENV's block size of xORMRQ /
// xUNMRQ, at most 64) are applied with xLARZT and xLARZB. LWORK >= max(1,
// N) (SIDE = 'L') or max(1, M) (SIDE = 'R'); the optimal LWORK is NW*NB +
// 65*64. LWORK = -1 is a workspace query. The complex routine asks ILAENV
// for NB a second time after the quick return, the real one does not.
template <typename T>
blasint Unmrz(char side, char trans, blasint m, blasint n, blasint k, blasint l,
              const T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  const Matrix<const T> A(a, lda);
  const Matrix<T> C(c, ldc);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);
  const char* nbname = RealOrComplex<T>("ORMRQ", "UNMRQ");
  const char opts[3] = {side, trans, '\0'};
  blasint i, i1, i2, i3, ib, iwt, ja, ldwork, lwkopt, nbmin, nq, nw;
  // NB is set where M > 0 and N > 0 (the only case that reads it); the
  // others in the loop where they are not set here.
  blasint nb = 0, ic = 0, jc = 0, mi = 0, ni = 0;
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
  } else if (l < 0 || (left && (l > m)) || (!left && (l > n))) {
    info = -6;
  } else if (lda < Max<blasint>(1, k)) {
    info = -8;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -11;
  } else if (lwork < nw && !lquery) {
    // As LAPACK (ZUNMRZ compares with MAX(1, NW), which is the same).
    info = -13;
  }

  lwkopt = 1;
  if (info == 0) {
    // Compute the workspace requirements
    if (m == 0 || n == 0) {
      lwkopt = 1;
    } else {
      nb = Min(kNbMax, Ilaenv<T>(1, nbname, opts, m, n, k, -1));
      lwkopt = nw * nb + kTsize;
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORMRZ", "UNMRZ"), -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible
  if (m == 0 || n == 0) return info;

  // Determine the block size.  NB may be at most NBMAX, where NBMAX
  // is used to define the local array T.
  if constexpr (IsComplex<T>::value) {
    nb = Min(kNbMax, Ilaenv<T>(1, nbname, opts, m, n, k, -1));
  }

  nbmin = 2;
  ldwork = nw;
  if (nb > 1 && nb < k) {
    if (lwork < lwkopt) {
      nb = (lwork - kTsize) / ldwork;
      nbmin = Max<blasint>(2, Ilaenv<T>(2, nbname, opts, m, n, k, -1));
    }
  }

  if (nb < nbmin || nb >= k) {
    // Use unblocked code
    Unmr3(side, trans, m, n, k, l, a, lda, tau, c, ldc, work);
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
      jc = 1;
      ja = m - l + 1;
    } else {
      mi = m;
      ic = 1;
      ja = n - l + 1;
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
      Larzt('B', 'R', l, ib, const_cast<T*>(A.ptr(i, ja)), lda, TAU.ptr(i),
            WORK.ptr(iwt), kLdt);

      if (left) {
        // H or H**H is applied to C(i:m,1:n)
        mi = m - i + 1;
        ic = i;
      } else {
        // H or H**H is applied to C(1:m,i:n)
        ni = n - i + 1;
        jc = i;
      }

      // Apply H or H**H
      Larzb(side, transt, 'B', 'R', mi, ni, ib, l, const_cast<T*>(A.ptr(i, ja)),
            lda, WORK.ptr(iwt), kLdt, C.ptr(ic, jc), ldc, work, ldwork);
    }
  }

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template void Larz<T>(char, blasint, blasint, blasint, const T*, blasint, T, \
                        T*, blasint, T*);                                      \
  template void Larzb<T>(char, char, char, char, blasint, blasint, blasint,    \
                         blasint, T*, blasint, T*, blasint, T*, blasint, T*,   \
                         blasint);                                             \
  template void Larzt<T>(char, char, blasint, blasint, T*, blasint, const T*,  \
                         T*, blasint);                                         \
  template void Latrz<T>(blasint, blasint, blasint, T*, blasint, T*, T*);      \
  template blasint Tzrzf<T>(blasint, blasint, T*, blasint, T*, T*, blasint);   \
  template blasint Unmr3<T>(char, char, blasint, blasint, blasint, blasint,    \
                            const T*, blasint, const T*, T*, blasint, T*);     \
  template blasint Unmrz<T>(char, char, blasint, blasint, blasint, blasint,    \
                            const T*, blasint, const T*, T*, blasint, T*,      \
                            blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// The real routines (ORMR3, ORMRZ) for real T, the complex ones (UNMR3,
// UNMRZ) for complex T: `mul` is the middle of their names.
#define HWYBLAS_LAPACK_TZRZF(prefix, mul, T, R)                                \
  HWYBLAS_API void prefix##larz_(                                              \
      const char* side, const hwyblas_int* m, const hwyblas_int* n,            \
      const hwyblas_int* l, const R* v, const hwyblas_int* incv, const R* tau, \
      R* c, const hwyblas_int* ldc, R* work) {                                 \
    hwyblas::lapack::Larz(*side, *m, *n, *l, As<const T>(v), *incv,            \
                          *As<const T>(tau), As<T>(c), *ldc, As<T>(work));     \
  }                                                                            \
  HWYBLAS_API void prefix##larzb_(                                             \
      const char* side, const char* trans, const char* direct,                 \
      const char* storev, const hwyblas_int* m, const hwyblas_int* n,          \
      const hwyblas_int* k, const hwyblas_int* l, R* v,                        \
      const hwyblas_int* ldv, R* t, const hwyblas_int* ldt, R* c,              \
      const hwyblas_int* ldc, R* work, const hwyblas_int* ldwork) {            \
    hwyblas::lapack::Larzb(*side, *trans, *direct, *storev, *m, *n, *k, *l,    \
                           As<T>(v), *ldv, As<T>(t), *ldt, As<T>(c), *ldc,     \
                           As<T>(work), *ldwork);                              \
  }                                                                            \
  HWYBLAS_API void prefix##larzt_(const char* direct, const char* storev,      \
                                  const hwyblas_int* n, const hwyblas_int* k,  \
                                  R* v, const hwyblas_int* ldv, const R* tau,  \
                                  R* t, const hwyblas_int* ldt) {              \
    hwyblas::lapack::Larzt(*direct, *storev, *n, *k, As<T>(v), *ldv,           \
                           As<const T>(tau), As<T>(t), *ldt);                  \
  }                                                                            \
  HWYBLAS_API void prefix##latrz_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  const hwyblas_int* l, R* a,                  \
                                  const hwyblas_int* lda, R* tau, R* work) {   \
    hwyblas::lapack::Latrz(*m, *n, *l, As<T>(a), *lda, As<T>(tau),             \
                           As<T>(work));                                       \
  }                                                                            \
  HWYBLAS_API void prefix##tzrzf_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  R* a, const hwyblas_int* lda, R* tau,        \
                                  R* work, const hwyblas_int* lwork,           \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Tzrzf(*m, *n, As<T>(a), *lda, As<T>(tau),         \
                                   As<T>(work), *lwork);                       \
  }                                                                            \
  HWYBLAS_API void prefix##mul##r3_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const hwyblas_int* l,        \
      const R* a, const hwyblas_int* lda, const R* tau, R* c,                  \
      const hwyblas_int* ldc, R* work, hwyblas_int* info) {                    \
    *info = hwyblas::lapack::Unmr3(*side, *trans, *m, *n, *k, *l,              \
                                   As<const T>(a), *lda, As<const T>(tau),     \
                                   As<T>(c), *ldc, As<T>(work));               \
  }                                                                            \
  HWYBLAS_API void prefix##mul##rz_(                                           \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const hwyblas_int* l,        \
      const R* a, const hwyblas_int* lda, const R* tau, R* c,                  \
      const hwyblas_int* ldc, R* work, const hwyblas_int* lwork,               \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Unmrz(*side, *trans, *m, *n, *k, *l,              \
                                   As<const T>(a), *lda, As<const T>(tau),     \
                                   As<T>(c), *ldc, As<T>(work), *lwork);       \
  }

HWYBLAS_LAPACK_TZRZF(s, orm, float, float)
HWYBLAS_LAPACK_TZRZF(d, orm, double, double)
HWYBLAS_LAPACK_TZRZF(c, unm, c32, float)
HWYBLAS_LAPACK_TZRZF(z, unm, c64, double)

#undef HWYBLAS_LAPACK_TZRZF

}  // extern "C"
