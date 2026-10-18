// Reduction of a symmetric (Hermitian) matrix to real symmetric tridiagonal
// form by an orthogonal (unitary) similarity transformation, Q**H * A * Q =
// T: xSYTRD / xHETRD (blocked), xSYTD2 / xHETD2 (unblocked) and xLATRD (one
// block). Each template is the real routine for real T and the complex
// routine for complex T; they differ in conjugations and in taking the
// real parts of the diagonal, which the templates do for both (it changes
// nothing for real T).
//
// Q is a product of N-1 elementary reflectors (see householder.cc), stored
// in A and TAU:
//   UPLO = 'U': Q = H(n-1) . . . H(2) H(1), with v(i+1:n) = 0, v(i) = 1
//               and v(1:i-1) in A(1:i-1,i+1);
//   UPLO = 'L': Q = H(1) H(2) . . . H(n-1), with v(1:i) = 0, v(i+1) = 1
//               and v(i+2:n) in A(i+2:n,i).
// The diagonal D and the off-diagonal E of T are returned, and overwrite
// the corresponding elements of A.
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

// xSYTD2 / xHETD2: reduces the N-by-N symmetric (Hermitian) matrix A, of
// which the triangle UPLO is referenced, to tridiagonal form T with the
// unblocked algorithm (see the top of the file). D has N elements, E and
// TAU N-1.
template <typename T>
blasint Hetd2(char uplo, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tau) {
  using R = RealT<T>;
  const T one = T(1), zero = T(0), half = T(0.5);
  const Matrix<T> A(a, lda);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<T> TAU(tau);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < std::max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYTD2", "HETD2"), -info);
    return info;
  }

  // Quick return if possible.
  if (n <= 0) return info;

  if (upper) {
    // Reduce the upper triangle of A.
    A(n, n) = std::real(A(n, n));
    for (blasint i = n - 1; i >= 1; --i) {
      // Generate elementary reflector H(i) = I - tau * v * v**H to
      // annihilate A(1:i-1,i+1).
      T alpha = A(i, i + 1);
      T taui;
      Larfg(i, &alpha, A.ptr(1, i + 1), 1, &taui);
      E(i) = std::real(alpha);

      if (taui != zero) {
        // Apply H(i) from both sides to A(1:i,1:i).
        A(i, i + 1) = one;

        // Compute x := tau * A * v, storing x in TAU(1:i).
        Hemv(uplo, i, taui, a, lda, A.ptr(1, i + 1), 1, zero, tau, 1);

        // Compute w := x - 1/2 * tau * (x**H * v) * v.
        alpha = -half * taui * Dotc(i, tau, 1, A.ptr(1, i + 1), 1);
        Axpy(i, alpha, A.ptr(1, i + 1), 1, tau, 1);

        // Apply the transformation as a rank-2 update:
        // A := A - v * w**H - w * v**H.
        Her2(uplo, i, -one, A.ptr(1, i + 1), 1, tau, 1, a, lda);
      } else {
        A(i, i) = std::real(A(i, i));
      }
      A(i, i + 1) = E(i);
      D(i + 1) = std::real(A(i + 1, i + 1));
      TAU(i) = taui;
    }
    D(1) = std::real(A(1, 1));
  } else {
    // Reduce the lower triangle of A.
    A(1, 1) = std::real(A(1, 1));
    for (blasint i = 1; i <= n - 1; ++i) {
      // Generate elementary reflector H(i) = I - tau * v * v**H to
      // annihilate A(i+2:n,i).
      T alpha = A(i + 1, i);
      T taui;
      Larfg(n - i, &alpha, A.ptr(std::min(i + 2, n), i), 1, &taui);
      E(i) = std::real(alpha);

      if (taui != zero) {
        // Apply H(i) from both sides to A(i+1:n,i+1:n).
        A(i + 1, i) = one;

        // Compute x := tau * A * v, storing x in TAU(i:n-1).
        Hemv(uplo, n - i, taui, A.ptr(i + 1, i + 1), lda, A.ptr(i + 1, i), 1,
             zero, TAU.ptr(i), 1);

        // Compute w := x - 1/2 * tau * (x**H * v) * v.
        alpha = -half * taui * Dotc(n - i, TAU.ptr(i), 1, A.ptr(i + 1, i), 1);
        Axpy(n - i, alpha, A.ptr(i + 1, i), 1, TAU.ptr(i), 1);

        // Apply the transformation as a rank-2 update:
        // A := A - v * w**H - w * v**H.
        Her2(uplo, n - i, -one, A.ptr(i + 1, i), 1, TAU.ptr(i), 1,
             A.ptr(i + 1, i + 1), lda);
      } else {
        A(i + 1, i + 1) = std::real(A(i + 1, i + 1));
      }
      A(i + 1, i) = E(i);
      D(i) = std::real(A(i, i));
      TAU(i) = taui;
    }
    D(n) = std::real(A(n, n));
  }
  return info;
}

// xLATRD: reduces NB rows and columns of the N-by-N symmetric (Hermitian)
// matrix A to tridiagonal form by an orthogonal (unitary) similarity
// transformation, and returns the matrices V and W which are needed to
// apply the transformation to the unreduced part of A: A := A - V * W**H -
// W * V**H.
//
// UPLO = 'U': the last NB columns are reduced (E(n-nb:n-1), TAU(n-nb:n-1));
// UPLO = 'L': the first NB columns (E(1:nb), TAU(1:nb)). W is N-by-NB.
template <typename T>
void Latrd(char uplo, blasint n, blasint nb, T* a, blasint lda, RealT<T>* e,
           T* tau, T* w, blasint ldw) {
  const T one = T(1), zero = T(0), half = T(0.5);
  const Matrix<T> A(a, lda);
  const Vector<RealT<T>> E(e);
  const Vector<T> TAU(tau);
  const Matrix<T> W(w, ldw);

  // Quick return if possible.
  if (n <= 0) return;

  if (Lsame(uplo, 'U')) {
    // Reduce last NB columns of upper triangle.
    for (blasint i = n; i >= n - nb + 1; --i) {
      const blasint iw = i - n + nb;
      if (i < n) {
        // Update A(1:i,i).
        A(i, i) = std::real(A(i, i));
        Lacgv(n - i, W.ptr(i, iw + 1), ldw);
        Gemv('N', i, n - i, -one, A.ptr(1, i + 1), lda, W.ptr(i, iw + 1), ldw,
             one, A.ptr(1, i), 1);
        Lacgv(n - i, W.ptr(i, iw + 1), ldw);
        Lacgv(n - i, A.ptr(i, i + 1), lda);
        Gemv('N', i, n - i, -one, W.ptr(1, iw + 1), ldw, A.ptr(i, i + 1), lda,
             one, A.ptr(1, i), 1);
        Lacgv(n - i, A.ptr(i, i + 1), lda);
        A(i, i) = std::real(A(i, i));
      }
      if (i > 1) {
        // Generate elementary reflector H(i-1) to annihilate A(1:i-2,i).
        T alpha = A(i - 1, i);
        Larfg(i - 1, &alpha, A.ptr(1, i), 1, TAU.ptr(i - 1));
        E(i - 1) = std::real(alpha);
        A(i - 1, i) = one;

        // Compute W(1:i-1,i).
        Hemv('U', i - 1, one, a, lda, A.ptr(1, i), 1, zero, W.ptr(1, iw), 1);
        if (i < n) {
          Gemv('C', i - 1, n - i, one, W.ptr(1, iw + 1), ldw, A.ptr(1, i), 1,
               zero, W.ptr(i + 1, iw), 1);
          Gemv('N', i - 1, n - i, -one, A.ptr(1, i + 1), lda, W.ptr(i + 1, iw),
               1, one, W.ptr(1, iw), 1);
          Gemv('C', i - 1, n - i, one, A.ptr(1, i + 1), lda, A.ptr(1, i), 1,
               zero, W.ptr(i + 1, iw), 1);
          Gemv('N', i - 1, n - i, -one, W.ptr(1, iw + 1), ldw, W.ptr(i + 1, iw),
               1, one, W.ptr(1, iw), 1);
        }
        Scal(i - 1, TAU(i - 1), W.ptr(1, iw), 1);
        alpha =
            -half * TAU(i - 1) * Dotc(i - 1, W.ptr(1, iw), 1, A.ptr(1, i), 1);
        Axpy(i - 1, alpha, A.ptr(1, i), 1, W.ptr(1, iw), 1);
      }
    }
  } else {
    // Reduce first NB columns of lower triangle.
    for (blasint i = 1; i <= nb; ++i) {
      // Update A(i:n,i).
      A(i, i) = std::real(A(i, i));
      Lacgv(i - 1, W.ptr(i, 1), ldw);
      Gemv('N', n - i + 1, i - 1, -one, A.ptr(i, 1), lda, W.ptr(i, 1), ldw, one,
           A.ptr(i, i), 1);
      Lacgv(i - 1, W.ptr(i, 1), ldw);
      Lacgv(i - 1, A.ptr(i, 1), lda);
      Gemv('N', n - i + 1, i - 1, -one, W.ptr(i, 1), ldw, A.ptr(i, 1), lda, one,
           A.ptr(i, i), 1);
      Lacgv(i - 1, A.ptr(i, 1), lda);
      A(i, i) = std::real(A(i, i));
      if (i < n) {
        // Generate elementary reflector H(i) to annihilate A(i+2:n,i).
        T alpha = A(i + 1, i);
        Larfg(n - i, &alpha, A.ptr(std::min(i + 2, n), i), 1, TAU.ptr(i));
        E(i) = std::real(alpha);
        A(i + 1, i) = one;

        // Compute W(i+1:n,i).
        Hemv('L', n - i, one, A.ptr(i + 1, i + 1), lda, A.ptr(i + 1, i), 1,
             zero, W.ptr(i + 1, i), 1);
        Gemv('C', n - i, i - 1, one, W.ptr(i + 1, 1), ldw, A.ptr(i + 1, i), 1,
             zero, W.ptr(1, i), 1);
        Gemv('N', n - i, i - 1, -one, A.ptr(i + 1, 1), lda, W.ptr(1, i), 1, one,
             W.ptr(i + 1, i), 1);
        Gemv('C', n - i, i - 1, one, A.ptr(i + 1, 1), lda, A.ptr(i + 1, i), 1,
             zero, W.ptr(1, i), 1);
        Gemv('N', n - i, i - 1, -one, W.ptr(i + 1, 1), ldw, W.ptr(1, i), 1, one,
             W.ptr(i + 1, i), 1);
        Scal(n - i, TAU(i), W.ptr(i + 1, i), 1);
        alpha = -half * TAU(i) *
                Dotc(n - i, W.ptr(i + 1, i), 1, A.ptr(i + 1, i), 1);
        Axpy(n - i, alpha, A.ptr(i + 1, i), 1, W.ptr(i + 1, i), 1);
      }
    }
  }
}

// xSYTRD / xHETRD: reduces the N-by-N symmetric (Hermitian) matrix A to
// tridiagonal form T (see the top of the file) with the blocked algorithm:
// panels of NB columns (from ILAENV) are reduced with xLATRD and applied to
// the rest of the matrix with xSYR2K / xHER2K, down to NX columns (at least
// NB), which xSYTD2 / xHETD2 reduces. LWORK >= 1; the optimal LWORK is
// N*NB; LWORK = -1 is a workspace query.
template <typename T>
blasint Hetrd(char uplo, blasint n, T* a, blasint lda, RealT<T>* d, RealT<T>* e,
              T* tau, T* work, blasint lwork) {
  using R = RealT<T>;
  const R one = R(1);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);
  const char* name = RealOrComplex<T>("SYTRD", "HETRD");
  const char opts[2] = {uplo, '\0'};

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool lquery = (lwork == -1);
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < std::max<blasint>(1, n)) {
    info = -4;
  } else if (lwork < 1 && !lquery) {
    info = -9;
  }

  blasint nb = 0, lwkopt = 0;
  if (info == 0) {
    // Determine the block size.
    nb = Ilaenv<T>(1, name, opts, n, -1, -1, -1);
    lwkopt = std::max<blasint>(1, n * nb);
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint nx = n;
  blasint ldwork = 0;
  if (nb > 1 && nb < n) {
    // Determine when to cross over from blocked to unblocked code (last
    // block is always handled by unblocked code).
    nx = std::max(nb, Ilaenv<T>(3, name, opts, n, -1, -1, -1));
    if (nx < n) {
      // Determine if workspace is large enough for blocked code.
      ldwork = n;
      const blasint iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: determine the minimum
        // value of NB, and reduce NB or force use of unblocked code by
        // setting NX = N.
        nb = std::max<blasint>(lwork / ldwork, 1);
        const blasint nbmin = Ilaenv<T>(2, name, opts, n, -1, -1, -1);
        if (nb < nbmin) nx = n;
      }
    } else {
      nx = n;
    }
  } else {
    nb = 1;
  }

  if (upper) {
    // Reduce the upper triangle of A. Columns 1:kk are handled by the
    // unblocked method.
    const blasint kk = n - ((n - nx + nb - 1) / nb) * nb;
    for (blasint i = n - nb + 1; i >= kk + 1; i -= nb) {
      // Reduce columns i:i+nb-1 to tridiagonal form and form the matrix W
      // which is needed to update the unreduced part of the matrix.
      Latrd(uplo, i + nb - 1, nb, a, lda, e, tau, work, ldwork);

      // Update the unreduced submatrix A(1:i-1,1:i-1), using an update of
      // the form: A := A - V*W**H - W*V**H.
      Her2k(uplo, 'N', i - 1, nb, -cone, A.ptr(1, i), lda, work, ldwork, one, a,
            lda);

      // Copy superdiagonal elements back into A, and diagonal elements
      // into D.
      for (blasint j = i; j <= i + nb - 1; ++j) {
        A(j - 1, j) = E(j - 1);
        D(j) = std::real(A(j, j));
      }
    }

    // Use unblocked code to reduce the last or only block.
    Hetd2(uplo, kk, a, lda, d, e, tau);
  } else {
    // Reduce the lower triangle of A.
    blasint i = 1;
    for (; i <= n - nx; i += nb) {
      // Reduce columns i:i+nb-1 to tridiagonal form and form the matrix W
      // which is needed to update the unreduced part of the matrix.
      Latrd(uplo, n - i + 1, nb, A.ptr(i, i), lda, E.ptr(i), TAU.ptr(i), work,
            ldwork);

      // Update the unreduced submatrix A(i+nb:n,i+nb:n), using an update
      // of the form: A := A - V*W**H - W*V**H.
      Her2k(uplo, 'N', n - i - nb + 1, nb, -cone, A.ptr(i + nb, i), lda,
            WORK.ptr(nb + 1), ldwork, one, A.ptr(i + nb, i + nb), lda);

      // Copy subdiagonal elements back into A, and diagonal elements into
      // D.
      for (blasint j = i; j <= i + nb - 1; ++j) {
        A(j + 1, j) = E(j);
        D(j) = std::real(A(j, j));
      }
    }

    // Use unblocked code to reduce the last or only block.
    Hetd2(uplo, n - i + 1, A.ptr(i, i), lda, D.ptr(i), E.ptr(i), TAU.ptr(i));
  }

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Hetd2<T>(char, blasint, T*, blasint, RealT<T>*, RealT<T>*, \
                            T*);                                              \
  template void Latrd<T>(char, blasint, blasint, T*, blasint, RealT<T>*, T*,  \
                         T*, blasint);                                        \
  template blasint Hetrd<T>(char, blasint, T*, blasint, RealT<T>*, RealT<T>*, \
                            T*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// xSYTRD etc. for real T, xHETRD etc. for complex T: `sy` is "sy" or "he".
#define HWYBLAS_LAPACK_TRIDIAGONAL(prefix, sy, T, R)                           \
  HWYBLAS_API void prefix##sy##td2_(const char* uplo, const hwyblas_int* n,    \
                                    R* a, const hwyblas_int* lda, R* d, R* e,  \
                                    R* tau, hwyblas_int* info) {               \
    *info =                                                                    \
        hwyblas::lapack::Hetd2(*uplo, *n, As<T>(a), *lda, d, e, As<T>(tau));   \
  }                                                                            \
  HWYBLAS_API void prefix##sy##trd_(const char* uplo, const hwyblas_int* n,    \
                                    R* a, const hwyblas_int* lda, R* d, R* e,  \
                                    R* tau, R* work, const hwyblas_int* lwork, \
                                    hwyblas_int* info) {                       \
    *info = hwyblas::lapack::Hetrd(*uplo, *n, As<T>(a), *lda, d, e,            \
                                   As<T>(tau), As<T>(work), *lwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##latrd_(                                             \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nb, R* a,     \
      const hwyblas_int* lda, R* e, R* tau, R* w, const hwyblas_int* ldw) {    \
    hwyblas::lapack::Latrd(*uplo, *n, *nb, As<T>(a), *lda, e, As<T>(tau),      \
                           As<T>(w), *ldw);                                    \
  }

HWYBLAS_LAPACK_TRIDIAGONAL(s, sy, float, float)
HWYBLAS_LAPACK_TRIDIAGONAL(d, sy, double, double)
HWYBLAS_LAPACK_TRIDIAGONAL(c, he, c32, float)
HWYBLAS_LAPACK_TRIDIAGONAL(z, he, c64, double)

#undef HWYBLAS_LAPACK_TRIDIAGONAL

}  // extern "C"
