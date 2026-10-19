// Reduction of a general matrix to bidiagonal form by orthogonal (unitary)
// transformations, Q**H * A * P = B: xGEBRD (blocked), xGEBD2 (unblocked)
// and xLABRD (one block), and the matrices Q and P**H: xORGBR / xUNGBR
// (generate) and xORMBR / xUNMBR (apply). Each template is the real routine
// for real T and the complex routine for complex T; they differ in
// conjugations, which the templates do for both (they change nothing for
// real T).
//
// For M >= N, B is upper bidiagonal; Q = H(1) H(2) . . . H(n) and P = G(1)
// G(2) . . . G(n-1) are products of elementary reflectors (see
// householder.cc), H(i) = I - tauq * v * v**H and G(i) = I - taup * u *
// u**H, with
//   v(1:i-1) = 0, v(i) = 1, v(i+1:m) in A(i+1:m,i);
//   u(1:i) = 0, u(i+1) = 1, u(i+2:n) in A(i,i+2:n).
// For M < N, B is lower bidiagonal; Q = H(1) . . . H(m-1), P = G(1) . . .
// G(m), with
//   v(1:i) = 0, v(i+1) = 1, v(i+2:m) in A(i+2:m,i);
//   u(1:i-1) = 0, u(i) = 1, u(i+1:n) in A(i,i+1:n).
// The diagonal D and the off-diagonal E of B are real, and overwrite the
// corresponding elements of A.
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

// xGEBD2: reduces the M-by-N matrix A to bidiagonal form B with the
// unblocked algorithm (see the top of the file). D has min(M,N) elements, E
// min(M,N)-1, TAUQ and TAUP min(M,N); WORK has max(M,N).
template <typename T>
blasint Gebd2(blasint m, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tauq, T* taup, T* work) {
  using R = RealT<T>;
  const T zero = T(0);
  const Matrix<T> A(a, lda);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<T> TAUQ(tauq);
  const Vector<T> TAUP(taup);

  // Test the input parameters.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }
  if (info < 0) {
    Xerbla<T>("GEBD2", -info);
    return info;
  }

  if (m >= n) {
    // Reduce to upper bidiagonal form.
    for (blasint i = 1; i <= n; ++i) {
      // Generate elementary reflector H(i) to annihilate A(i+1:m,i).
      T alpha = A(i, i);
      Larfg(m - i + 1, &alpha, A.ptr(Min(i + 1, m), i), 1, TAUQ.ptr(i));
      D(i) = std::real(alpha);

      // Apply H(i)**H to A(i:m,i+1:n) from the left.
      if (i < n) {
        Larf1f('L', m - i + 1, n - i, A.ptr(i, i), 1, Conj(TAUQ(i)),
               A.ptr(i, i + 1), lda, work);
      }
      A(i, i) = D(i);

      if (i < n) {
        // Generate elementary reflector G(i) to annihilate A(i,i+2:n).
        Lacgv(n - i, A.ptr(i, i + 1), lda);
        alpha = A(i, i + 1);
        Larfg(n - i, &alpha, A.ptr(i, Min(i + 2, n)), lda, TAUP.ptr(i));
        E(i) = std::real(alpha);

        // Apply G(i) to A(i+1:m,i+1:n) from the right.
        Larf1f('R', m - i, n - i, A.ptr(i, i + 1), lda, TAUP(i),
               A.ptr(i + 1, i + 1), lda, work);
        Lacgv(n - i, A.ptr(i, i + 1), lda);
        A(i, i + 1) = E(i);
      } else {
        TAUP(i) = zero;
      }
    }
  } else {
    // Reduce to lower bidiagonal form.
    for (blasint i = 1; i <= m; ++i) {
      // Generate G(i) to annihilate A(i,i+1:n).
      Lacgv(n - i + 1, A.ptr(i, i), lda);
      T alpha = A(i, i);
      Larfg(n - i + 1, &alpha, A.ptr(i, Min(i + 1, n)), lda, TAUP.ptr(i));
      D(i) = std::real(alpha);

      // Apply G(i) to A(i+1:m,i:n) from the right.
      if (i < m) {
        Larf1f('R', m - i, n - i + 1, A.ptr(i, i), lda, TAUP(i),
               A.ptr(i + 1, i), lda, work);
      }
      Lacgv(n - i + 1, A.ptr(i, i), lda);
      A(i, i) = D(i);

      if (i < m) {
        // Generate H(i) to annihilate A(i+2:m,i).
        alpha = A(i + 1, i);
        Larfg(m - i, &alpha, A.ptr(Min(i + 2, m), i), 1, TAUQ.ptr(i));
        E(i) = std::real(alpha);

        // Apply H(i)**H to A(i+1:m,i+1:n) from the left.
        Larf1f('L', m - i, n - i, A.ptr(i + 1, i), 1, Conj(TAUQ(i)),
               A.ptr(i + 1, i + 1), lda, work);
        A(i + 1, i) = E(i);
      } else {
        TAUQ(i) = zero;
      }
    }
  }
  return info;
}

// xLABRD: reduces the first NB rows and columns of the M-by-N matrix A to
// upper (M >= N) or lower (M < N) bidiagonal form, and returns the M-by-NB
// matrix X and the N-by-NB matrix Y which are needed to apply the
// transformation to the unreduced part of A: A := A - V * Y**H - X * U**H,
// with V and U the reflectors' vectors (see the top of the file). The
// diagonal and the off-diagonal elements of the reduced part are not
// stored back into A, which holds the reflectors' leading ones there
// instead (except for A(1:nb,1:nb) of the last reflector's diagonal: as in
// LAPACK, the real routine stores beta there and the complex routine leaves
// the element unchanged when NB = N for M >= N, or NB = M for M < N).
template <typename T>
void Labrd(blasint m, blasint n, blasint nb, T* a, blasint lda, RealT<T>* d,
           RealT<T>* e, T* tauq, T* taup, T* x, blasint ldx, T* y,
           blasint ldy) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const T zero = T(0), one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<T> TAUQ(tauq);
  const Vector<T> TAUP(taup);
  const Matrix<T> X(x, ldx);
  const Matrix<T> Y(y, ldy);

  // Quick return if possible.
  if (m <= 0 || n <= 0) return;

  if (m >= n) {
    // Reduce to upper bidiagonal form.
    for (blasint i = 1; i <= nb; ++i) {
      // Update A(i:m,i).
      Lacgv(i - 1, Y.ptr(i, 1), ldy);
      Gemv('N', m - i + 1, i - 1, -one, A.ptr(i, 1), lda, Y.ptr(i, 1), ldy, one,
           A.ptr(i, i), 1);
      Lacgv(i - 1, Y.ptr(i, 1), ldy);
      Gemv('N', m - i + 1, i - 1, -one, X.ptr(i, 1), ldx, A.ptr(1, i), 1, one,
           A.ptr(i, i), 1);

      // Generate reflection Q(i) to annihilate A(i+1:m,i).
      T alpha = A(i, i);
      Larfg(m - i + 1, &alpha, A.ptr(Min(i + 1, m), i), 1, TAUQ.ptr(i));
      D(i) = std::real(alpha);
      if constexpr (!kComplex) A(i, i) = alpha;
      if (i < n) {
        A(i, i) = one;

        // Compute Y(i+1:n,i).
        Gemv('C', m - i + 1, n - i, one, A.ptr(i, i + 1), lda, A.ptr(i, i), 1,
             zero, Y.ptr(i + 1, i), 1);
        Gemv('C', m - i + 1, i - 1, one, A.ptr(i, 1), lda, A.ptr(i, i), 1, zero,
             Y.ptr(1, i), 1);
        Gemv('N', n - i, i - 1, -one, Y.ptr(i + 1, 1), ldy, Y.ptr(1, i), 1, one,
             Y.ptr(i + 1, i), 1);
        Gemv('C', m - i + 1, i - 1, one, X.ptr(i, 1), ldx, A.ptr(i, i), 1, zero,
             Y.ptr(1, i), 1);
        Gemv('C', i - 1, n - i, -one, A.ptr(1, i + 1), lda, Y.ptr(1, i), 1, one,
             Y.ptr(i + 1, i), 1);
        Scal(n - i, TAUQ(i), Y.ptr(i + 1, i), 1);

        // Update A(i,i+1:n).
        Lacgv(n - i, A.ptr(i, i + 1), lda);
        Lacgv(i, A.ptr(i, 1), lda);
        Gemv('N', n - i, i, -one, Y.ptr(i + 1, 1), ldy, A.ptr(i, 1), lda, one,
             A.ptr(i, i + 1), lda);
        Lacgv(i, A.ptr(i, 1), lda);
        Lacgv(i - 1, X.ptr(i, 1), ldx);
        Gemv('C', i - 1, n - i, -one, A.ptr(1, i + 1), lda, X.ptr(i, 1), ldx,
             one, A.ptr(i, i + 1), lda);
        Lacgv(i - 1, X.ptr(i, 1), ldx);

        // Generate reflection P(i) to annihilate A(i,i+2:n).
        alpha = A(i, i + 1);
        Larfg(n - i, &alpha, A.ptr(i, Min(i + 2, n)), lda, TAUP.ptr(i));
        E(i) = std::real(alpha);
        A(i, i + 1) = one;

        // Compute X(i+1:m,i).
        Gemv('N', m - i, n - i, one, A.ptr(i + 1, i + 1), lda, A.ptr(i, i + 1),
             lda, zero, X.ptr(i + 1, i), 1);
        Gemv('C', n - i, i, one, Y.ptr(i + 1, 1), ldy, A.ptr(i, i + 1), lda,
             zero, X.ptr(1, i), 1);
        Gemv('N', m - i, i, -one, A.ptr(i + 1, 1), lda, X.ptr(1, i), 1, one,
             X.ptr(i + 1, i), 1);
        Gemv('N', i - 1, n - i, one, A.ptr(1, i + 1), lda, A.ptr(i, i + 1), lda,
             zero, X.ptr(1, i), 1);
        Gemv('N', m - i, i - 1, -one, X.ptr(i + 1, 1), ldx, X.ptr(1, i), 1, one,
             X.ptr(i + 1, i), 1);
        Scal(m - i, TAUP(i), X.ptr(i + 1, i), 1);
        Lacgv(n - i, A.ptr(i, i + 1), lda);
      }
    }
  } else {
    // Reduce to lower bidiagonal form.
    for (blasint i = 1; i <= nb; ++i) {
      // Update A(i,i:n).
      Lacgv(n - i + 1, A.ptr(i, i), lda);
      Lacgv(i - 1, A.ptr(i, 1), lda);
      Gemv('N', n - i + 1, i - 1, -one, Y.ptr(i, 1), ldy, A.ptr(i, 1), lda, one,
           A.ptr(i, i), lda);
      Lacgv(i - 1, A.ptr(i, 1), lda);
      Lacgv(i - 1, X.ptr(i, 1), ldx);
      Gemv('C', i - 1, n - i + 1, -one, A.ptr(1, i), lda, X.ptr(i, 1), ldx, one,
           A.ptr(i, i), lda);
      Lacgv(i - 1, X.ptr(i, 1), ldx);

      // Generate reflection P(i) to annihilate A(i,i+1:n).
      T alpha = A(i, i);
      Larfg(n - i + 1, &alpha, A.ptr(i, Min(i + 1, n)), lda, TAUP.ptr(i));
      D(i) = std::real(alpha);
      if constexpr (!kComplex) A(i, i) = alpha;
      if (i < m) {
        A(i, i) = one;

        // Compute X(i+1:m,i).
        Gemv('N', m - i, n - i + 1, one, A.ptr(i + 1, i), lda, A.ptr(i, i), lda,
             zero, X.ptr(i + 1, i), 1);
        Gemv('C', n - i + 1, i - 1, one, Y.ptr(i, 1), ldy, A.ptr(i, i), lda,
             zero, X.ptr(1, i), 1);
        Gemv('N', m - i, i - 1, -one, A.ptr(i + 1, 1), lda, X.ptr(1, i), 1, one,
             X.ptr(i + 1, i), 1);
        Gemv('N', i - 1, n - i + 1, one, A.ptr(1, i), lda, A.ptr(i, i), lda,
             zero, X.ptr(1, i), 1);
        Gemv('N', m - i, i - 1, -one, X.ptr(i + 1, 1), ldx, X.ptr(1, i), 1, one,
             X.ptr(i + 1, i), 1);
        Scal(m - i, TAUP(i), X.ptr(i + 1, i), 1);
        Lacgv(n - i + 1, A.ptr(i, i), lda);

        // Update A(i+1:m,i).
        Lacgv(i - 1, Y.ptr(i, 1), ldy);
        Gemv('N', m - i, i - 1, -one, A.ptr(i + 1, 1), lda, Y.ptr(i, 1), ldy,
             one, A.ptr(i + 1, i), 1);
        Lacgv(i - 1, Y.ptr(i, 1), ldy);
        Gemv('N', m - i, i, -one, X.ptr(i + 1, 1), ldx, A.ptr(1, i), 1, one,
             A.ptr(i + 1, i), 1);

        // Generate reflection Q(i) to annihilate A(i+2:m,i).
        alpha = A(i + 1, i);
        Larfg(m - i, &alpha, A.ptr(Min(i + 2, m), i), 1, TAUQ.ptr(i));
        E(i) = std::real(alpha);
        A(i + 1, i) = one;

        // Compute Y(i+1:n,i).
        Gemv('C', m - i, n - i, one, A.ptr(i + 1, i + 1), lda, A.ptr(i + 1, i),
             1, zero, Y.ptr(i + 1, i), 1);
        Gemv('C', m - i, i - 1, one, A.ptr(i + 1, 1), lda, A.ptr(i + 1, i), 1,
             zero, Y.ptr(1, i), 1);
        Gemv('N', n - i, i - 1, -one, Y.ptr(i + 1, 1), ldy, Y.ptr(1, i), 1, one,
             Y.ptr(i + 1, i), 1);
        Gemv('C', m - i, i, one, X.ptr(i + 1, 1), ldx, A.ptr(i + 1, i), 1, zero,
             Y.ptr(1, i), 1);
        Gemv('C', i, n - i, -one, A.ptr(1, i + 1), lda, Y.ptr(1, i), 1, one,
             Y.ptr(i + 1, i), 1);
        Scal(n - i, TAUQ(i), Y.ptr(i + 1, i), 1);
      } else {
        Lacgv(n - i + 1, A.ptr(i, i), lda);
      }
    }
  }
}

// xGEBRD: reduces the M-by-N matrix A to bidiagonal form B (see the top of
// the file) with the blocked algorithm: panels of NB rows and columns (from
// ILAENV) are reduced with xLABRD and applied to the rest of the matrix
// with xGEMM, down to NX columns (at least NB), which xGEBD2 reduces.
// LWORK >= max(1, M, N); the optimal LWORK is (M+N)*NB; LWORK = -1 is a
// workspace query.
template <typename T>
blasint Gebrd(blasint m, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tauq, T* taup, T* work, blasint lwork) {
  using R = RealT<T>;
  const T one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<T> TAUQ(tauq);
  const Vector<T> TAUP(taup);
  const Vector<T> WORK(work);

  blasint info = 0;
  const blasint minmn = Min(m, n);
  blasint lwkmin, lwkopt, nb = 0;
  if (minmn == 0) {
    lwkmin = 1;
    lwkopt = 1;
  } else {
    lwkmin = Max(m, n);
    nb = Max<blasint>(1, Ilaenv<T>(1, "GEBRD", " ", m, n, -1, -1));
    lwkopt = (m + n) * nb;
  }
  WORK(1) = WorkSize<T>(lwkopt);

  // Test the input parameters.
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  } else if (lwork < lwkmin && !lquery) {
    info = -10;
  }
  if (info < 0) {
    Xerbla<T>("GEBRD", -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (minmn == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint ws = Max(m, n);
  const blasint ldwrkx = m;
  const blasint ldwrky = n;

  blasint nx;
  if (nb > 1 && nb < minmn) {
    // Set the crossover point NX.
    nx = Max(nb, Ilaenv<T>(3, "GEBRD", " ", m, n, -1, -1));

    // Determine when to switch from blocked to unblocked code.
    if (nx < minmn) {
      ws = lwkopt;
      if (lwork < ws) {
        // Not enough work space for the optimal NB, consider using a
        // smaller block size.
        const blasint nbmin = Ilaenv<T>(2, "GEBRD", " ", m, n, -1, -1);
        if (lwork >= (m + n) * nbmin) {
          nb = lwork / (m + n);
        } else {
          nb = 1;
          nx = minmn;
        }
      }
    }
  } else {
    nx = minmn;
  }

  blasint i = 1;
  for (; i <= minmn - nx; i += nb) {
    // Reduce rows and columns i:i+nb-1 to bidiagonal form and return the
    // matrices X and Y which are needed to update the unreduced part of
    // the matrix.
    Labrd(m - i + 1, n - i + 1, nb, A.ptr(i, i), lda, D.ptr(i), E.ptr(i),
          TAUQ.ptr(i), TAUP.ptr(i), work, ldwrkx, WORK.ptr(ldwrkx * nb + 1),
          ldwrky);

    // Update the trailing submatrix A(i+nb:m,i+nb:n), using an update of
    // the form A := A - V*Y**H - X*U**H.
    Gemm('N', 'C', m - i - nb + 1, n - i - nb + 1, nb, -one, A.ptr(i + nb, i),
         lda, WORK.ptr(ldwrkx * nb + nb + 1), ldwrky, one,
         A.ptr(i + nb, i + nb), lda);
    Gemm('N', 'N', m - i - nb + 1, n - i - nb + 1, nb, -one, WORK.ptr(nb + 1),
         ldwrkx, A.ptr(i, i + nb), lda, one, A.ptr(i + nb, i + nb), lda);

    // Copy diagonal and off-diagonal elements of B back into A.
    if (m >= n) {
      for (blasint j = i; j <= i + nb - 1; ++j) {
        A(j, j) = D(j);
        A(j, j + 1) = E(j);
      }
    } else {
      for (blasint j = i; j <= i + nb - 1; ++j) {
        A(j, j) = D(j);
        A(j + 1, j) = E(j);
      }
    }
  }

  // Use unblocked code to reduce the remainder of the matrix.
  Gebd2(m - i + 1, n - i + 1, A.ptr(i, i), lda, D.ptr(i), E.ptr(i), TAUQ.ptr(i),
        TAUP.ptr(i), work);
  WORK(1) = WorkSize<T>(ws);
  return info;
}

// xORGBR / xUNGBR: generates one of the matrices Q or P**H determined by
// xGEBRD when reducing a matrix A to bidiagonal form, A = Q * B * P**H. Q
// and P**H are defined as products of elementary reflectors H(i) or G(i)
// respectively.
//
// VECT = 'Q': A is assumed to have been an M-by-K matrix, and Q is of order
// M: if M >= K, Q = H(1) H(2) . . . H(k) and the first N columns of Q are
// generated (M >= N >= K); if M < K, Q = H(1) H(2) . . . H(m-1) and Q is
// generated as an M-by-M matrix.
//
// VECT = 'P': A is assumed to have been a K-by-N matrix, and P**H is of
// order N: if K < N, P**H = G(k) . . . G(2) G(1) and its first M rows are
// generated (N >= M >= K); if K >= N, P**H = G(n-1) . . . G(2) G(1) and
// P**H is generated as an N-by-N matrix.
//
// LWORK >= max(1, min(M, N)); LWORK = -1 is a workspace query.
template <typename T>
blasint Ungbr(char vect, blasint m, blasint n, blasint k, T* a, blasint lda,
              const T* tau, T* work, blasint lwork) {
  const T zero = T(0), one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const bool wantq = Lsame(vect, 'Q');
  const blasint mn = Min(m, n);
  const bool lquery = (lwork == -1);
  if (!wantq && !Lsame(vect, 'P')) {
    info = -1;
  } else if (m < 0) {
    info = -2;
  } else if (n < 0 || (wantq && (n > m || n < Min(m, k))) ||
             (!wantq && (m > n || m < Min(n, k)))) {
    info = -3;
  } else if (k < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, m)) {
    info = -6;
  } else if (lwork < Max<blasint>(1, mn) && !lquery) {
    info = -9;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    WORK(1) = T(1);
    if (wantq) {
      if (m >= k) {
        Ungqr(m, n, k, a, lda, tau, work, -1);
      } else if (m > 1) {
        Ungqr(m - 1, m - 1, m - 1, a, lda, tau, work, -1);
      }
    } else {
      if (k < n) {
        Unglq(m, n, k, a, lda, tau, work, -1);
      } else if (n > 1) {
        Unglq(n - 1, n - 1, n - 1, a, lda, tau, work, -1);
      }
    }
    lwkopt = static_cast<blasint>(std::real(WORK(1)));
    lwkopt = Max(lwkopt, mn);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORGBR", "UNGBR"), -info);
    return info;
  }
  if (lquery) {
    WORK(1) = WorkSize<T>(lwkopt);
    return info;
  }

  // Quick return if possible.
  if (m == 0 || n == 0) {
    WORK(1) = T(1);
    return info;
  }

  if (wantq) {
    // Form Q, determined by a call to xGEBRD to reduce an M-by-K matrix.
    if (m >= k) {
      // If M >= K, assume M >= N >= K.
      Ungqr(m, n, k, a, lda, tau, work, lwork);
    } else {
      // If M < K, assume M = N.
      //
      // Shift the vectors which define the elementary reflectors one
      // column to the right, and set the first row and column of Q to
      // those of the unit matrix.
      for (blasint j = m; j >= 2; --j) {
        A(1, j) = zero;
        for (blasint i = j + 1; i <= m; ++i) A(i, j) = A(i, j - 1);
      }
      A(1, 1) = one;
      for (blasint i = 2; i <= m; ++i) A(i, 1) = zero;
      if (m > 1) {
        // Form Q(2:m,2:m).
        Ungqr(m - 1, m - 1, m - 1, A.ptr(2, 2), lda, tau, work, lwork);
      }
    }
  } else {
    // Form P**H, determined by a call to xGEBRD to reduce a K-by-N matrix.
    if (k < n) {
      // If K < N, assume K <= M <= N.
      Unglq(m, n, k, a, lda, tau, work, lwork);
    } else {
      // If K >= N, assume M = N.
      //
      // Shift the vectors which define the elementary reflectors one row
      // downward, and set the first row and column of P**H to those of the
      // unit matrix.
      A(1, 1) = one;
      for (blasint i = 2; i <= n; ++i) A(i, 1) = zero;
      for (blasint j = 2; j <= n; ++j) {
        for (blasint i = j - 1; i >= 2; --i) A(i, j) = A(i - 1, j);
        A(1, j) = zero;
      }
      if (n > 1) {
        // Form P**H(2:n,2:n).
        Unglq(n - 1, n - 1, n - 1, A.ptr(2, 2), lda, tau, work, lwork);
      }
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORMBR / xUNMBR: overwrites the M-by-N matrix C with
//   VECT = 'Q': Q * C, Q**H * C, C * Q or C * Q**H;
//   VECT = 'P': P * C, P**H * C, C * P or C * P**H
// (SIDE = 'L' or 'R', TRANS = 'N' or 'C'; 'T' for real T), with Q and P**H
// from xGEBRD of an NQ-by-K (VECT = 'Q') or K-by-NQ (VECT = 'P') matrix,
// where NQ = M (SIDE = 'L') or N (SIDE = 'R'): if NQ >= K, Q = H(1) H(2)
// . . . H(k); if NQ < K, Q = H(1) H(2) . . . H(nq-1); if K < NQ, P = G(1)
// G(2) . . . G(k); if K >= NQ, P = G(1) G(2) . . . G(nq-1).
//
// LWORK >= max(1, N) (SIDE = 'L') or max(1, M) (SIDE = 'R'); LWORK = -1 is
// a workspace query. A is changed (conjugated) for complex T and VECT =
// 'P', but restored on exit (see xUNMLQ).
//
// The real routine returns the optimal workspace size from a query also
// when M or N is 0 (N*NB or M*NB, with NB = ILAENV(1, ...) of the empty
// matrix), the complex routine 1; as in LAPACK.
template <typename T>
blasint Unmbr(char vect, char side, char trans, blasint m, blasint n, blasint k,
              T* a, blasint lda, const T* tau, T* c, blasint ldc, T* work,
              blasint lwork) {
  constexpr bool kComplex = IsComplex<T>::value;
  const Matrix<T> A(a, lda);
  const Matrix<T> C(c, ldc);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const bool applyq = Lsame(vect, 'Q');
  const bool left = Lsame(side, 'L');
  const bool notran = Lsame(trans, 'N');
  const bool lquery = (lwork == -1);

  // NQ is the order of Q or P and NW is the minimum dimension of WORK.
  blasint nq, nw;
  if (left) {
    nq = m;
    nw = Max<blasint>(1, n);
  } else {
    nq = n;
    nw = Max<blasint>(1, m);
  }
  if (!applyq && !Lsame(vect, 'P')) {
    info = -1;
  } else if (!left && !Lsame(side, 'R')) {
    info = -2;
  } else if (!notran && !Lsame(trans, ConjTrans<T>())) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (k < 0) {
    info = -6;
  } else if ((applyq && lda < Max<blasint>(1, nq)) ||
             (!applyq && lda < Max<blasint>(1, Min(nq, k)))) {
    info = -8;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -11;
  } else if (lwork < nw && !lquery) {
    info = -13;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    if ((m > 0 && n > 0) || !kComplex) {
      const char opts[3] = {side, trans, '\0'};
      const char* name = applyq ? RealOrComplex<T>("ORMQR", "UNMQR")
                                : RealOrComplex<T>("ORMLQ", "UNMLQ");
      blasint nb;
      if (left) {
        nb = Ilaenv<T>(1, name, opts, m - 1, n, m - 1, -1);
      } else {
        nb = Ilaenv<T>(1, name, opts, m, n - 1, n - 1, -1);
      }
      lwkopt = nw * nb;
    } else {
      lwkopt = 1;
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORMBR", "UNMBR"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if constexpr (!kComplex) WORK(1) = T(1);
  if (m == 0 || n == 0) return info;

  blasint mi, ni, i1, i2;
  if (left) {
    mi = m - 1;
    ni = n;
    i1 = 2;
    i2 = 1;
  } else {
    mi = m;
    ni = n - 1;
    i1 = 1;
    i2 = 2;
  }
  if (applyq) {
    // Apply Q.
    if (nq >= k) {
      // Q was determined by a call to xGEBRD with nq >= k.
      Unmqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
    } else if (nq > 1) {
      // Q was determined by a call to xGEBRD with nq < k.
      Unmqr(side, trans, mi, ni, nq - 1, A.ptr(2, 1), lda, tau, C.ptr(i1, i2),
            ldc, work, lwork);
    }
  } else {
    // Apply P.
    const char transt = notran ? ConjTrans<T>() : 'N';
    if (nq > k) {
      // P was determined by a call to xGEBRD with nq > k.
      Unmlq(side, transt, m, n, k, a, lda, tau, c, ldc, work, lwork);
    } else if (nq > 1) {
      // P was determined by a call to xGEBRD with nq <= k.
      Unmlq(side, transt, mi, ni, nq - 1, A.ptr(1, 2), lda, tau, C.ptr(i1, i2),
            ldc, work, lwork);
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                        \
  template blasint Gebd2<T>(blasint, blasint, T*, blasint, RealT<T>*,        \
                            RealT<T>*, T*, T*, T*);                          \
  template void Labrd<T>(blasint, blasint, blasint, T*, blasint, RealT<T>*,  \
                         RealT<T>*, T*, T*, T*, blasint, T*, blasint);       \
  template blasint Gebrd<T>(blasint, blasint, T*, blasint, RealT<T>*,        \
                            RealT<T>*, T*, T*, T*, blasint);                 \
  template blasint Ungbr<T>(char, blasint, blasint, blasint, T*, blasint,    \
                            const T*, T*, blasint);                          \
  template blasint Unmbr<T>(char, char, char, blasint, blasint, blasint, T*, \
                            blasint, const T*, T*, blasint, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// xORGBR / xORMBR for real T, xUNGBR / xUNMBR for complex T: `og` is "org"
// or "ung", `om` "orm" or "unm".
#define HWYBLAS_LAPACK_BIDIAGONAL(prefix, og, om, T, R)                        \
  HWYBLAS_API void prefix##gebd2_(const hwyblas_int* m, const hwyblas_int* n,  \
                                  R* a, const hwyblas_int* lda, R* d, R* e,    \
                                  R* tauq, R* taup, R* work,                   \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Gebd2(*m, *n, As<T>(a), *lda, d, e, As<T>(tauq),  \
                                   As<T>(taup), As<T>(work));                  \
  }                                                                            \
  HWYBLAS_API void prefix##labrd_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nb, R* a, \
      const hwyblas_int* lda, R* d, R* e, R* tauq, R* taup, R* x,              \
      const hwyblas_int* ldx, R* y, const hwyblas_int* ldy) {                  \
    hwyblas::lapack::Labrd(*m, *n, *nb, As<T>(a), *lda, d, e, As<T>(tauq),     \
                           As<T>(taup), As<T>(x), *ldx, As<T>(y), *ldy);       \
  }                                                                            \
  HWYBLAS_API void prefix##gebrd_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, R* a,                        \
      const hwyblas_int* lda, R* d, R* e, R* tauq, R* taup, R* work,           \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Gebrd(*m, *n, As<T>(a), *lda, d, e, As<T>(tauq),  \
                                   As<T>(taup), As<T>(work), *lwork);          \
  }                                                                            \
  HWYBLAS_API void prefix##og##br_(                                            \
      const char* vect, const hwyblas_int* m, const hwyblas_int* n,            \
      const hwyblas_int* k, R* a, const hwyblas_int* lda, const R* tau,        \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Ungbr(*vect, *m, *n, *k, As<T>(a), *lda,          \
                                   As<const T>(tau), As<T>(work), *lwork);     \
  }                                                                            \
  HWYBLAS_API void prefix##om##br_(                                            \
      const char* vect, const char* side, const char* trans,                   \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* k, R* a,  \
      const hwyblas_int* lda, const R* tau, R* c, const hwyblas_int* ldc,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Unmbr(*vect, *side, *trans, *m, *n, *k, As<T>(a), \
                                   *lda, As<const T>(tau), As<T>(c), *ldc,     \
                                   As<T>(work), *lwork);                       \
  }

HWYBLAS_LAPACK_BIDIAGONAL(s, org, orm, float, float)
HWYBLAS_LAPACK_BIDIAGONAL(d, org, orm, double, double)
HWYBLAS_LAPACK_BIDIAGONAL(c, ung, unm, c32, float)
HWYBLAS_LAPACK_BIDIAGONAL(z, ung, unm, c64, double)

#undef HWYBLAS_LAPACK_BIDIAGONAL

}  // extern "C"
