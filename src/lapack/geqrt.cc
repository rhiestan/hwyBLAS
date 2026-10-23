// QR factorization in the compact WY representation: xGEQRT (blocked),
// xGEQRT2 (unblocked) and xGEQRT3 (recursive), and xGEMQRT (multiplies by
// its Q).
//
// Unlike xGEQRF, these return the triangular factors of the block
// reflectors: for each block of NB columns, Q's block H = I - V * T * V**H,
// with the reflector vectors V below the diagonal of A (unit diagonal not
// stored) and the NB-by-NB upper triangular T stored side by side in the
// NB-by-N array T.
//
// Each template is the real routine for real T and the complex one for
// complex T; they differ only in conjugations and in TRANS = 'T' (real) or
// 'C' (complex) for Q**T / Q**H.
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

// xGEQRT2: computes a QR factorization of the M-by-N matrix A, M >= N,
// A = Q * R, with the unblocked algorithm, in the compact WY representation
// Q = I - V * T * V**H.
//
// On exit, the elements on and above the diagonal of A contain the N-by-N
// upper triangular matrix R; the elements below the diagonal are the
// columns of V (the i-th column is the vector of the reflector H(i); its
// unit diagonal element is not stored); T (LDT >= max(1, N)) is the N-by-N
// upper triangular factor (its strictly lower triangle is not referenced,
// except that column 1 is used as workspace and set to zero below the
// diagonal).
template <typename T>
blasint Geqrt2(blasint m, blasint n, T* a, blasint lda, T* t, blasint ldt) {
  const T one = T(1), zero = T(0);
  const Matrix<T> A(a, lda);
  const Matrix<T> Tm(t, ldt);
  const char trans = ConjTrans<T>();

  // Test the input arguments
  blasint info = 0;
  if (n < 0) {
    info = -2;
  } else if (m < n) {
    info = -1;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("GEQRT2", -info);
    return info;
  }

  const blasint k = Min(m, n);

  for (blasint i = 1; i <= k; ++i) {
    // Generate elem. refl. H(i) to annihilate A(i+1:m,i), tau(I) -> T(I,1)
    Larfg(m - i + 1, A.ptr(i, i), A.ptr(Min(i + 1, m), i), 1, Tm.ptr(i, 1));
    if (i < n) {
      // Apply H(i) to A(I:M,I+1:N) from the left
      const T aii = A(i, i);
      A(i, i) = one;

      // W(1:N-I) := A(I:M,I+1:N)^H * A(I:M,I) [W = T(:,N)]
      Gemv(trans, m - i + 1, n - i, one, A.ptr(i, i + 1), lda, A.ptr(i, i), 1,
           zero, Tm.ptr(1, n), 1);

      // A(I:M,I+1:N) = A(I:m,I+1:N) + alpha*A(I:M,I)*W(1:N-1)^H
      const T alpha = -Conj(Tm(i, 1));
      Gerc(m - i + 1, n - i, alpha, A.ptr(i, i), 1, Tm.ptr(1, n), 1,
           A.ptr(i, i + 1), lda);
      A(i, i) = aii;
    }
  }

  for (blasint i = 2; i <= n; ++i) {
    const T aii = A(i, i);
    A(i, i) = one;

    // T(1:I-1,I) := alpha * A(I:M,1:I-1)**H * A(I:M,I)
    const T alpha = -Tm(i, 1);
    Gemv(trans, m - i + 1, i - 1, alpha, A.ptr(i, 1), lda, A.ptr(i, i), 1, zero,
         Tm.ptr(1, i), 1);
    A(i, i) = aii;

    // T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
    Trmv('U', 'N', 'N', i - 1, t, ldt, Tm.ptr(1, i), 1);

    // T(I,I) = tau(I)
    Tm(i, i) = Tm(i, 1);
    Tm(i, 1) = zero;
  }
  return info;
}

// xGEQRT3: computes a QR factorization of the M-by-N matrix A, M >= N,
// like xGEQRT2 (see there for the result), with the recursive algorithm of
// Elmroth and Gustavson: it factors the left and right halves of the
// columns recursively and joins their triangular factors.
//
// Deviation (bug fix): the reference has no quick return for N = 0, for
// which it calls itself with the same arguments until the stack overflows
// (xGEQRT never calls it with N = 0, but it is callable directly); here it
// returns.
template <typename T>
blasint Geqrt3(blasint m, blasint n, T* a, blasint lda, T* t, blasint ldt) {
  const T one = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> Tm(t, ldt);
  const char trans = ConjTrans<T>();

  blasint info = 0;
  if (n < 0) {
    info = -2;
  } else if (m < n) {
    info = -1;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("GEQRT3", -info);
    return info;
  }

  // Quick return (see above).
  if (n == 0) return info;

  if (n == 1) {
    // Compute Householder transform when N=1
    Larfg(m, A.ptr(1, 1), A.ptr(Min<blasint>(2, m), 1), 1, Tm.ptr(1, 1));
  } else {
    // Otherwise, split A into blocks...
    const blasint n1 = n / 2;
    const blasint n2 = n - n1;
    const blasint j1 = Min(n1 + 1, n);
    const blasint i1 = Min(n + 1, m);

    // Compute A(1:M,1:N1) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H
    Geqrt3(m, n1, a, lda, t, ldt);

    // Compute A(1:M,J1:N) = Q1^H A(1:M,J1:N) [workspace: T(1:N1,J1:N)]
    for (blasint j = 1; j <= n2; ++j) {
      for (blasint i = 1; i <= n1; ++i) Tm(i, j + n1) = A(i, j + n1);
    }
    Trmm('L', 'L', trans, 'U', n1, n2, one, a, lda, Tm.ptr(1, j1), ldt);

    Gemm(trans, 'N', n1, n2, m - n1, one, A.ptr(j1, 1), lda, A.ptr(j1, j1), lda,
         one, Tm.ptr(1, j1), ldt);

    Trmm('L', 'U', trans, 'N', n1, n2, one, t, ldt, Tm.ptr(1, j1), ldt);

    Gemm('N', 'N', m - n1, n2, n1, -one, A.ptr(j1, 1), lda, Tm.ptr(1, j1), ldt,
         one, A.ptr(j1, j1), lda);

    Trmm('L', 'L', 'N', 'U', n1, n2, one, a, lda, Tm.ptr(1, j1), ldt);

    for (blasint j = 1; j <= n2; ++j) {
      for (blasint i = 1; i <= n1; ++i) {
        A(i, j + n1) = A(i, j + n1) - Tm(i, j + n1);
      }
    }

    // Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H
    Geqrt3(m - n1, n2, A.ptr(j1, j1), lda, Tm.ptr(j1, j1), ldt);

    // Compute T3 = T(1:N1,J1:N) = -T1 Y1^H Y2 T2
    for (blasint i = 1; i <= n1; ++i) {
      for (blasint j = 1; j <= n2; ++j) Tm(i, j + n1) = Conj(A(j + n1, i));
    }

    Trmm('R', 'L', 'N', 'U', n1, n2, one, A.ptr(j1, j1), lda, Tm.ptr(1, j1),
         ldt);

    Gemm(trans, 'N', n1, n2, m - n, one, A.ptr(i1, 1), lda, A.ptr(i1, j1), lda,
         one, Tm.ptr(1, j1), ldt);

    Trmm('L', 'U', 'N', 'N', n1, n2, -one, t, ldt, Tm.ptr(1, j1), ldt);

    Trmm('R', 'U', 'N', 'N', n1, n2, one, Tm.ptr(j1, j1), ldt, Tm.ptr(1, j1),
         ldt);

    // Y = (Y1,Y2); R = [ R1  A(1:N1,J1:N) ];  T = [T1 T3]
    //                  [  0        R2     ]       [ 0 T2]
  }
  return info;
}

// xGEQRT: computes a blocked QR factorization of the M-by-N matrix A,
// A = Q * R, in the compact WY representation: the blocks of NB columns
// (1 <= NB <= min(M,N) if min(M,N) > 0) are factored by xGEQRT3, which
// gives the block reflectors H(j) = I - V(j) * T(j) * V(j)**H, with
// Q = H(1) H(2) . . . H(b), b = ceil(min(M,N)/NB).
//
// On exit, the elements on and above the diagonal of A contain the
// min(M,N)-by-N upper trapezoidal matrix R; below the diagonal are the
// reflector vectors (unit diagonal elements not stored), and T (LDT >= NB)
// holds the NB-by-NB upper triangular factors T(j) side by side, the last
// one of order min(M,N) - (b-1)*NB. WORK has NB*N elements.
template <typename T>
blasint Geqrt(blasint m, blasint n, blasint nb, T* a, blasint lda, T* t,
              blasint ldt, T* work) {
  const Matrix<T> A(a, lda);
  const Matrix<T> Tm(t, ldt);
  // (The reference's parameter USE_RECURSIVE_QR = .TRUE. selects xGEQRT3
  // over xGEQRT2.)

  // Test the input arguments
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nb < 1 || (nb > Min(m, n) && Min(m, n) > 0)) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldt < nb) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("GEQRT", -info);
    return info;
  }

  // Quick return if possible
  const blasint k = Min(m, n);
  if (k == 0) return info;

  // Blocked loop of length K
  for (blasint i = 1; i <= k; i += nb) {
    const blasint ib = Min(k - i + 1, nb);

    // Compute the QR factorization of the current block A(I:M,I:I+IB-1)
    Geqrt3(m - i + 1, ib, A.ptr(i, i), lda, Tm.ptr(1, i), ldt);
    if (i + ib <= n) {
      // Update by applying H**H to A(I:M,I+IB:N) from the left
      Larfb('L', ConjTrans<T>(), 'F', 'C', m - i + 1, n - i - ib + 1, ib,
            A.ptr(i, i), lda, Tm.ptr(1, i), ldt, A.ptr(i, i + ib), lda, work,
            n - i - ib + 1);
    }
  }
  return info;
}

// xGEMQRT: overwrites the M-by-N matrix C with
//
//     Q * C (SIDE = 'L', TRANS = 'N'),   Q**H * C (SIDE = 'L', TRANS = 'C'),
//     C * Q (SIDE = 'R', TRANS = 'N'),   C * Q**H (SIDE = 'R', TRANS = 'C'),
//
// (TRANS = 'T' and Q**T for real T), where Q = H(1) H(2) . . . H(k) is the
// product of K elementary reflectors of order M (SIDE = 'L') or N
// (SIDE = 'R') returned by xGEQRT with block size NB: V (LDV >= max(1, M)
// or max(1, N)) holds their vectors, T (LDT >= NB) the triangular factors.
// WORK has N*NB (SIDE = 'L') or M*NB (SIDE = 'R') elements.
template <typename T>
blasint Gemqrt(char side, char trans, blasint m, blasint n, blasint k,
               blasint nb, const T* v, blasint ldv, const T* t, blasint ldt,
               T* c, blasint ldc, T* work) {
  const Matrix<const T> V(v, ldv);
  const Matrix<const T> Tm(t, ldt);
  const Matrix<T> C(c, ldc);
  const char transc = ConjTrans<T>();

  // Test the input arguments
  blasint info = 0;
  const bool left = Lsame(side, 'L');
  const bool right = Lsame(side, 'R');
  const bool tran = Lsame(trans, transc);
  const bool notran = Lsame(trans, 'N');

  blasint ldwork = 0, q = 0;
  if (left) {
    ldwork = Max<blasint>(1, n);
    q = m;
  } else if (right) {
    ldwork = Max<blasint>(1, m);
    q = n;
  }
  if (!left && !right) {
    info = -1;
  } else if (!tran && !notran) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > q) {
    info = -5;
  } else if (nb < 1 || (nb > k && k > 0)) {
    info = -6;
  } else if (ldv < Max<blasint>(1, q)) {
    info = -8;
  } else if (ldt < nb) {
    info = -10;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -12;
  }

  if (info != 0) {
    Xerbla<T>("GEMQRT", -info);
    return info;
  }

  // Quick return if possible
  if (m == 0 || n == 0 || k == 0) return info;

  if (left && tran) {
    for (blasint i = 1; i <= k; i += nb) {
      const blasint ib = Min(nb, k - i + 1);
      Larfb('L', transc, 'F', 'C', m - i + 1, n, ib, V.ptr(i, i), ldv,
            Tm.ptr(1, i), ldt, C.ptr(i, 1), ldc, work, ldwork);
    }
  } else if (right && notran) {
    for (blasint i = 1; i <= k; i += nb) {
      const blasint ib = Min(nb, k - i + 1);
      Larfb('R', 'N', 'F', 'C', m, n - i + 1, ib, V.ptr(i, i), ldv,
            Tm.ptr(1, i), ldt, C.ptr(1, i), ldc, work, ldwork);
    }
  } else if (left && notran) {
    const blasint kf = ((k - 1) / nb) * nb + 1;
    for (blasint i = kf; i >= 1; i -= nb) {
      const blasint ib = Min(nb, k - i + 1);
      Larfb('L', 'N', 'F', 'C', m - i + 1, n, ib, V.ptr(i, i), ldv,
            Tm.ptr(1, i), ldt, C.ptr(i, 1), ldc, work, ldwork);
    }
  } else if (right && tran) {
    const blasint kf = ((k - 1) / nb) * nb + 1;
    for (blasint i = kf; i >= 1; i -= nb) {
      const blasint ib = Min(nb, k - i + 1);
      Larfb('R', transc, 'F', 'C', m, n - i + 1, ib, V.ptr(i, i), ldv,
            Tm.ptr(1, i), ldt, C.ptr(1, i), ldc, work, ldwork);
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                        \
  template blasint Geqrt2<T>(blasint, blasint, T*, blasint, T*, blasint);    \
  template blasint Geqrt3<T>(blasint, blasint, T*, blasint, T*, blasint);    \
  template blasint Geqrt<T>(blasint, blasint, blasint, T*, blasint, T*,      \
                            blasint, T*);                                    \
  template blasint Gemqrt<T>(char, char, blasint, blasint, blasint, blasint, \
                             const T*, blasint, const T*, blasint, T*,       \
                             blasint, T*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GEQRT(prefix, T, R)                                     \
  HWYBLAS_API void prefix##geqrt2_(const hwyblas_int* m, const hwyblas_int* n, \
                                   R* a, const hwyblas_int* lda, R* t,         \
                                   const hwyblas_int* ldt,                     \
                                   hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Geqrt2(*m, *n, As<T>(a), *lda, As<T>(t), *ldt);   \
  }                                                                            \
  HWYBLAS_API void prefix##geqrt3_(const hwyblas_int* m, const hwyblas_int* n, \
                                   R* a, const hwyblas_int* lda, R* t,         \
                                   const hwyblas_int* ldt,                     \
                                   hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Geqrt3(*m, *n, As<T>(a), *lda, As<T>(t), *ldt);   \
  }                                                                            \
  HWYBLAS_API void prefix##geqrt_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nb, R* a, \
      const hwyblas_int* lda, R* t, const hwyblas_int* ldt, R* work,           \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Geqrt(*m, *n, *nb, As<T>(a), *lda, As<T>(t),      \
                                   *ldt, As<T>(work));                         \
  }                                                                            \
  HWYBLAS_API void prefix##gemqrt_(                                            \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* k, const hwyblas_int* nb,       \
      const R* v, const hwyblas_int* ldv, const R* t, const hwyblas_int* ldt,  \
      R* c, const hwyblas_int* ldc, R* work, hwyblas_int* info) {              \
    *info = hwyblas::lapack::Gemqrt(*side, *trans, *m, *n, *k, *nb,            \
                                    As<const T>(v), *ldv, As<const T>(t),      \
                                    *ldt, As<T>(c), *ldc, As<T>(work));        \
  }

HWYBLAS_LAPACK_GEQRT(s, float, float)
HWYBLAS_LAPACK_GEQRT(d, double, double)
HWYBLAS_LAPACK_GEQRT(c, c32, float)
HWYBLAS_LAPACK_GEQRT(z, c64, double)

#undef HWYBLAS_LAPACK_GEQRT

}  // extern "C"
