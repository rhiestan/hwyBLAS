// The orthogonal (unitary) matrix Q of a tridiagonal reduction (xORGTR /
// xUNGTR), and the matrix Q of a QL factorization that it uses (xORGQL /
// xUNGQL blocked, xORG2L / xUNG2L unblocked).
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
  } else if (lda < std::max<blasint>(1, m)) {
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
  } else if (lda < std::max<blasint>(1, m)) {
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

    if (lwork < std::max<blasint>(1, n) && !lquery) info = -8;
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
    nx = std::max<blasint>(0, Ilaenv<T>(3, name, " ", m, n, k, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = n;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and determine
        // the minimum value of NB.
        nb = lwork / ldwork;
        nbmin = std::max<blasint>(2, Ilaenv<T>(2, name, " ", m, n, k, -1));
      }
    }
  }

  blasint kk;
  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code after the first block. The last kk columns are
    // handled by the block method.
    kk = std::min(k, ((k - nx + nb - 1) / nb) * nb);

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
      const blasint ib = std::min(nb, k - i + 1);
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
  } else if (lda < std::max<blasint>(1, n)) {
    info = -4;
  } else if (lwork < std::max<blasint>(1, n - 1) && !lquery) {
    info = -7;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    const blasint nb = upper ? Ilaenv<T>(1, RealOrComplex<T>("ORGQL", "UNGQL"),
                                         " ", n - 1, n - 1, n - 1, -1)
                             : Ilaenv<T>(1, RealOrComplex<T>("ORGQR", "UNGQR"),
                                         " ", n - 1, n - 1, n - 1, -1);
    lwkopt = std::max<blasint>(1, n - 1) * nb;
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

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Ung2l<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*);                                              \
  template blasint Ungql<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*, blasint);                                     \
  template blasint Ungtr<T>(char, blasint, T*, blasint, const T*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// `g` is "org" (real) or "ung" (complex).
#define HWYBLAS_LAPACK_UNGTR(prefix, g, T, R)                                  \
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
  }

HWYBLAS_LAPACK_UNGTR(s, org, float, float)
HWYBLAS_LAPACK_UNGTR(d, org, double, double)
HWYBLAS_LAPACK_UNGTR(c, ung, c32, float)
HWYBLAS_LAPACK_UNGTR(z, ung, c64, double)

#undef HWYBLAS_LAPACK_UNGTR

}  // extern "C"
