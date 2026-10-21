// QR, LQ and RQ factorizations: xGEQRF, xGELQF and xGERQF (blocked),
// xGEQR2, xGELQ2 and xGERQ2 (unblocked).
//
// The factorizations store Q as a product of elementary reflectors (see
// householder.cc): the vectors v below the diagonal of A (QR, unit first
// element not stored) or to the right of it (LQ), and their scalars in TAU.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <algorithm>

#include "hwyblas/lapack.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xGEQR2: computes a QR factorization of the M-by-N matrix A, A = Q * R,
// with the unblocked algorithm.
//
// On exit, the elements on and above the diagonal of A contain the
// min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular if
// M >= N); the elements below the diagonal, with the array TAU, represent
// the unitary matrix Q as a product of k = min(M,N) elementary reflectors
//
//     Q = H(1) H(2) . . . H(k),   H(i) = I - tau * v * v**H,
//
// where tau is TAU(i) and v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored in
// A(i+1:m,i). WORK has N elements.
template <typename T>
blasint Geqr2(blasint m, blasint n, T* a, blasint lda, T* tau, T* work) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);

  // Test the input arguments.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("GEQR2", -info);
    return info;
  }

  const blasint k = Min(m, n);
  for (blasint i = 1; i <= k; ++i) {
    // Generate elementary reflector H(i) to annihilate A(i+1:m,i).
    Larfg(m - i + 1, A.ptr(i, i), A.ptr(Min(i + 1, m), i), 1, TAU.ptr(i));
    if (i < n) {
      // Apply H(i)**H to A(i:m,i+1:n) from the left.
      Larf1f('L', m - i + 1, n - i, A.ptr(i, i), 1, Conj(TAU(i)),
             A.ptr(i, i + 1), lda, work);
    }
  }
  return info;
}

// xGEQRF: computes a QR factorization of the M-by-N matrix A, A = Q * R,
// like xGEQR2 (see there for the result).
//
// This is the blocked version: it factors panels of NB columns (from
// ILAENV) with xGEQR2, and applies their block reflectors (xLARFT, xLARFB)
// to the rest of the matrix, down to the last NX columns, which xGEQR2
// factors. LWORK >= max(1, N) (or 1 if min(M,N) = 0); the optimal LWORK is
// N*NB. LWORK = -1 is a workspace query: WORK(1) is set to the optimal
// LWORK, and nothing else is done. On exit, WORK(1) is the workspace size
// that the routine used (blocked) or needs (unblocked).
template <typename T>
blasint Geqrf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);

  // Test the input arguments.
  const blasint k = Min(m, n);
  blasint info = 0;
  blasint nb = Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1);
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  } else if (!lquery) {
    if (lwork <= 0 || (m > 0 && lwork < Max<blasint>(1, n))) info = -7;
  }
  if (info != 0) {
    Xerbla<T>("GEQRF", -info);
    return info;
  }
  if (lquery) {
    const blasint lwkopt = (k == 0) ? 1 : n * nb;
    WORK(1) = WorkSize<T>(lwkopt);
    return info;
  }

  // Quick return if possible.
  if (k == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nbmin = 2;
  blasint nx = 0;
  blasint iws = n;
  blasint ldwork = 0;
  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Max<blasint>(0, Ilaenv<T>(3, "GEQRF", " ", m, n, -1, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = n;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and determine
        // the minimum value of NB.
        nb = lwork / ldwork;
        nbmin = Max<blasint>(2, Ilaenv<T>(2, "GEQRF", " ", m, n, -1, -1));
      }
    }
  }

  blasint i = 1;
  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code initially.
    for (; i <= k - nx; i += nb) {
      const blasint ib = Min(k - i + 1, nb);

      // Compute the QR factorization of the current block A(i:m,i:i+ib-1).
      Geqr2(m - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i), work);
      if (i + ib <= n) {
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1).
        Larft('F', 'C', m - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i), work,
              ldwork);

        // Apply H**H to A(i:m,i+ib:n) from the left.
        Larfb('L', 'C', 'F', 'C', m - i + 1, n - i - ib + 1, ib, A.ptr(i, i),
              lda, work, ldwork, A.ptr(i, i + ib), lda, WORK.ptr(ib + 1),
              ldwork);
      }
    }
  }

  // Use unblocked code to factor the last or only block.
  if (i <= k) Geqr2(m - i + 1, n - i + 1, A.ptr(i, i), lda, TAU.ptr(i), work);

  WORK(1) = WorkSize<T>(iws);
  return info;
}

// xGELQ2: computes an LQ factorization of the M-by-N matrix A, A = L * Q,
// with the unblocked algorithm.
//
// On exit, the elements on and below the diagonal of A contain the
// M-by-min(M,N) lower trapezoidal matrix L (L is lower triangular if
// M <= N); the elements above the diagonal, with the array TAU, represent
// the unitary matrix Q as a product of k = min(M,N) elementary reflectors
//
//     Q = H(k)**H . . . H(2)**H H(1)**H,   H(i) = I - tau * v * v**H,
//
// where tau is TAU(i) and v(1:i-1) = 0, v(i) = 1, and conj(v(i+1:n)) is
// stored in A(i,i+1:n). WORK has M elements.
template <typename T>
blasint Gelq2(blasint m, blasint n, T* a, blasint lda, T* tau, T* work) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);

  // Test the input arguments.
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("GELQ2", -info);
    return info;
  }

  const blasint k = Min(m, n);
  for (blasint i = 1; i <= k; ++i) {
    // Generate elementary reflector H(i) to annihilate A(i,i+1:n).
    Lacgv(n - i + 1, A.ptr(i, i), lda);
    Larfg(n - i + 1, A.ptr(i, i), A.ptr(i, Min(i + 1, n)), lda, TAU.ptr(i));
    if (i < m) {
      // Apply H(i) to A(i+1:m,i:n) from the right.
      Larf1f('R', m - i, n - i + 1, A.ptr(i, i), lda, TAU(i), A.ptr(i + 1, i),
             lda, work);
    }
    Lacgv(n - i + 1, A.ptr(i, i), lda);
  }
  return info;
}

// xGELQF: computes an LQ factorization of the M-by-N matrix A, A = L * Q,
// like xGELQ2 (see there for the result).
//
// This is the blocked version (see xGEQRF; here the panels are NB rows).
// LWORK >= max(1, M) (or 1 if min(M,N) = 0); the optimal LWORK is M*NB.
template <typename T>
blasint Gelqf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const blasint k = Min(m, n);
  blasint nb = Ilaenv<T>(1, "GELQF", " ", m, n, -1, -1);
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  } else if (!lquery) {
    if (lwork <= 0 || (n > 0 && lwork < Max<blasint>(1, m))) info = -7;
  }
  if (info != 0) {
    Xerbla<T>("GELQF", -info);
    return info;
  }
  if (lquery) {
    const blasint lwkopt = (k == 0) ? 1 : m * nb;
    WORK(1) = WorkSize<T>(lwkopt);
    return info;
  }

  // Quick return if possible.
  if (k == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nbmin = 2;
  blasint nx = 0;
  blasint iws = m;
  blasint ldwork = 0;
  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Max<blasint>(0, Ilaenv<T>(3, "GELQF", " ", m, n, -1, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = m;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and determine
        // the minimum value of NB.
        nb = lwork / ldwork;
        nbmin = Max<blasint>(2, Ilaenv<T>(2, "GELQF", " ", m, n, -1, -1));
      }
    }
  }

  blasint i = 1;
  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code initially.
    for (; i <= k - nx; i += nb) {
      const blasint ib = Min(k - i + 1, nb);

      // Compute the LQ factorization of the current block A(i:i+ib-1,i:n).
      Gelq2(ib, n - i + 1, A.ptr(i, i), lda, TAU.ptr(i), work);
      if (i + ib <= m) {
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1).
        Larft('F', 'R', n - i + 1, ib, A.ptr(i, i), lda, TAU.ptr(i), work,
              ldwork);

        // Apply H to A(i+ib:m,i:n) from the right.
        Larfb('R', 'N', 'F', 'R', m - i - ib + 1, n - i + 1, ib, A.ptr(i, i),
              lda, work, ldwork, A.ptr(i + ib, i), lda, WORK.ptr(ib + 1),
              ldwork);
      }
    }
  }

  // Use unblocked code to factor the last or only block.
  if (i <= k) Gelq2(m - i + 1, n - i + 1, A.ptr(i, i), lda, TAU.ptr(i), work);

  WORK(1) = WorkSize<T>(iws);
  return info;
}

// xGERQ2: computes an RQ factorization of the M-by-N matrix A, A = R * Q,
// with the unblocked algorithm.
//
// On exit, if M <= N, the upper triangle of the subarray A(1:m,n-m+1:n)
// contains the M-by-M upper triangular matrix R; if M >= N, the elements on
// and above the (m-n)-th subdiagonal contain the M-by-N upper trapezoidal
// matrix R; the remaining elements, with the array TAU, represent the
// unitary matrix Q as a product of k = min(M,N) elementary reflectors
//
//     Q = H(1)**H H(2)**H . . . H(k)**H,   H(i) = I - tau * v * v**H,
//
// where tau is TAU(i) and v(n-k+i+1:n) = 0, v(n-k+i) = 1, and
// conj(v(1:n-k+i-1)) is stored in A(m-k+i,1:n-k+i-1). WORK has M elements.
template <typename T>
blasint Gerq2(blasint m, blasint n, T* a, blasint lda, T* tau, T* work) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  blasint i, k;

  // Test the input arguments
  blasint info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("GERQ2", -info);
    return info;
  }

  k = Min(m, n);
  for (i = k; i >= 1; --i) {
    // Generate elementary reflector H(i) to annihilate
    // A(m-k+i,1:n-k+i-1)
    Lacgv(n - k + i, A.ptr(m - k + i, 1), lda);
    Larfg(n - k + i, A.ptr(m - k + i, n - k + i), A.ptr(m - k + i, 1), lda,
          TAU.ptr(i));

    // Apply H(i) to A(1:m-k+i-1,1:n-k+i) from the right
    Larf1l('R', m - k + i - 1, n - k + i, A.ptr(m - k + i, 1), lda, TAU(i), a,
           lda, work);
    Lacgv(n - k + i - 1, A.ptr(m - k + i, 1), lda);
  }
  return info;
}

// xGERQF: computes an RQ factorization of the M-by-N matrix A, A = R * Q,
// like xGERQ2 (see there for the result).
//
// This is the blocked version: it factors panels of NB rows from the bottom
// with xGERQ2 and applies their block reflectors (xLARFT, xLARFB) to the
// rows above, down to the first NX rows, which xGERQ2 factors. LWORK >=
// max(1, M) (or 1 if min(M,N) = 0; any LWORK >= 1 if N = 0); the optimal
// LWORK is M*NB. LWORK = -1 is a workspace query.
template <typename T>
blasint Gerqf(blasint m, blasint n, T* a, blasint lda, T* tau, T* work,
              blasint lwork) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);
  blasint i, ib, iws, k, ki, kk, ldwork, lwkopt, mu, nbmin, nu, nx;
  // Set where K > 0 (the only case that reads it).
  blasint nb = 0;
  bool lquery;

  // Test the input arguments
  blasint info = 0;
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, m)) {
    info = -4;
  }

  k = Min(m, n);
  if (info == 0) {
    if (k == 0) {
      lwkopt = 1;
    } else {
      nb = Ilaenv<T>(1, "GERQF", " ", m, n, -1, -1);
      lwkopt = m * nb;
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (!lquery) {
      if (lwork <= 0 || (n > 0 && lwork < Max<blasint>(1, m))) info = -7;
    }
  }

  if (info != 0) {
    Xerbla<T>("GERQF", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible
  if (k == 0) return info;

  nbmin = 2;
  nx = 1;
  iws = m;
  ldwork = 0;
  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Max<blasint>(0, Ilaenv<T>(3, "GERQF", " ", m, n, -1, -1));
    if (nx < k) {
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

  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code initially.
    // The last kk rows are handled by the block method.
    ki = ((k - nx - 1) / nb) * nb;
    kk = Min(k, ki + nb);

    for (i = k - kk + ki + 1; i >= k - kk + 1; i -= nb) {
      ib = Min(k - i + 1, nb);

      // Compute the RQ factorization of the current block
      // A(m-k+i:m-k+i+ib-1,1:n-k+i+ib-1)
      Gerq2(ib, n - k + i + ib - 1, A.ptr(m - k + i, 1), lda, TAU.ptr(i), work);
      if (m - k + i > 1) {
        // Form the triangular factor of the block reflector
        // H = H(i+ib-1) . . . H(i+1) H(i)
        Larft('B', 'R', n - k + i + ib - 1, ib, A.ptr(m - k + i, 1), lda,
              TAU.ptr(i), work, ldwork);

        // Apply H to A(1:m-k+i-1,1:n-k+i+ib-1) from the right
        Larfb('R', 'N', 'B', 'R', m - k + i - 1, n - k + i + ib - 1, ib,
              A.ptr(m - k + i, 1), lda, work, ldwork, a, lda, WORK.ptr(ib + 1),
              ldwork);
      }
    }
    mu = m - k + i + nb - 1;
    nu = n - k + i + nb - 1;
  } else {
    mu = m;
    nu = n;
  }

  // Use unblocked code to factor the last or only block
  if (mu > 0 && nu > 0) Gerq2(mu, nu, a, lda, tau, work);

  WORK(1) = WorkSize<T>(iws);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                        \
  template blasint Geqr2<T>(blasint, blasint, T*, blasint, T*, T*);          \
  template blasint Geqrf<T>(blasint, blasint, T*, blasint, T*, T*, blasint); \
  template blasint Gelq2<T>(blasint, blasint, T*, blasint, T*, T*);          \
  template blasint Gelqf<T>(blasint, blasint, T*, blasint, T*, T*, blasint); \
  template blasint Gerq2<T>(blasint, blasint, T*, blasint, T*, T*);          \
  template blasint Gerqf<T>(blasint, blasint, T*, blasint, T*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GEQRF(prefix, T, R)                                    \
  HWYBLAS_API void prefix##geqr2_(const hwyblas_int* m, const hwyblas_int* n, \
                                  R* a, const hwyblas_int* lda, R* tau,       \
                                  R* work, hwyblas_int* info) {               \
    *info = hwyblas::lapack::Geqr2(*m, *n, As<T>(a), *lda, As<T>(tau),        \
                                   As<T>(work));                              \
  }                                                                           \
  HWYBLAS_API void prefix##geqrf_(const hwyblas_int* m, const hwyblas_int* n, \
                                  R* a, const hwyblas_int* lda, R* tau,       \
                                  R* work, const hwyblas_int* lwork,          \
                                  hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Geqrf(*m, *n, As<T>(a), *lda, As<T>(tau),        \
                                   As<T>(work), *lwork);                      \
  }                                                                           \
  HWYBLAS_API void prefix##gelq2_(const hwyblas_int* m, const hwyblas_int* n, \
                                  R* a, const hwyblas_int* lda, R* tau,       \
                                  R* work, hwyblas_int* info) {               \
    *info = hwyblas::lapack::Gelq2(*m, *n, As<T>(a), *lda, As<T>(tau),        \
                                   As<T>(work));                              \
  }                                                                           \
  HWYBLAS_API void prefix##gelqf_(const hwyblas_int* m, const hwyblas_int* n, \
                                  R* a, const hwyblas_int* lda, R* tau,       \
                                  R* work, const hwyblas_int* lwork,          \
                                  hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Gelqf(*m, *n, As<T>(a), *lda, As<T>(tau),        \
                                   As<T>(work), *lwork);                      \
  }                                                                           \
  HWYBLAS_API void prefix##gerq2_(const hwyblas_int* m, const hwyblas_int* n, \
                                  R* a, const hwyblas_int* lda, R* tau,       \
                                  R* work, hwyblas_int* info) {               \
    *info = hwyblas::lapack::Gerq2(*m, *n, As<T>(a), *lda, As<T>(tau),        \
                                   As<T>(work));                              \
  }                                                                           \
  HWYBLAS_API void prefix##gerqf_(const hwyblas_int* m, const hwyblas_int* n, \
                                  R* a, const hwyblas_int* lda, R* tau,       \
                                  R* work, const hwyblas_int* lwork,          \
                                  hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Gerqf(*m, *n, As<T>(a), *lda, As<T>(tau),        \
                                   As<T>(work), *lwork);                      \
  }

HWYBLAS_LAPACK_GEQRF(s, float, float)
HWYBLAS_LAPACK_GEQRF(d, double, double)
HWYBLAS_LAPACK_GEQRF(c, c32, float)
HWYBLAS_LAPACK_GEQRF(z, c64, double)

#undef HWYBLAS_LAPACK_GEQRF

}  // extern "C"
