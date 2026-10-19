// Balancing of a general matrix, xGEBAL, and its back transformation,
// xGEBAK; reduction of a general matrix to upper Hessenberg form by an
// orthogonal (unitary) similarity transformation, Q**H * A * Q = H: xGEHRD
// (blocked), xGEHD2 (unblocked) and xLAHR2 (one block); and the matrix Q:
// xORGHR / xUNGHR (generate) and xORMHR / xUNMHR (apply). Each template is
// the real routine for real T and the complex routine for complex T; they
// differ in conjugations, which the templates do for both (they change
// nothing for real T).
//
// Q = H(ilo) H(ilo+1) . . . H(ihi-1) is a product of elementary reflectors
// (see householder.cc), H(i) = I - tau * v * v**H with v(1:i) = 0,
// v(i+1) = 1 and v(i+2:ihi) in A(i+2:ihi,i), v(ihi+1:n) = 0.
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

// xGEBAL: balances the N-by-N matrix A. JOB = 'N': nothing; 'P': permutes
// A to isolate eigenvalues, so that A(i,j) = 0 for i > j and j < ILO or
// i > IHI; 'S': scales rows and columns ILO:IHI to make their norms
// closer; 'B': both. SCALE(j) is the index of the row and column
// interchanged with j for j < ILO and j > IHI, and the scale factor for
// ILO <= j <= IHI.
//
// Returns INFO: 0, or -i for an illegal argument i (-3 if A has a NaN in
// the balanced part).
template <typename T>
blasint Gebal(char job, blasint n, T* a, blasint lda, blasint* ilo,
              blasint* ihi, RealT<T>* scale) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const R sclfac = R(2);
  const R factor = R(0.95);
  const Matrix<T> A(a, lda);
  const Vector<R> SCALE(scale);

  // Test the input parameters.
  blasint info = 0;
  if (!Lsame(job, 'N') && !Lsame(job, 'P') && !Lsame(job, 'S') &&
      !Lsame(job, 'B')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("GEBAL", -info);
    return info;
  }

  // Quick returns.
  if (n == 0) {
    *ilo = 1;
    *ihi = 0;
    return info;
  }
  if (Lsame(job, 'N')) {
    for (blasint i = 1; i <= n; ++i) SCALE(i) = one;
    *ilo = 1;
    *ihi = n;
    return info;
  }

  // Permutation to isolate eigenvalues if possible.
  blasint k = 1;
  blasint l = n;
  if (!Lsame(job, 'S')) {
    // Row and column exchange.
    bool noconv = true;
    while (noconv) {
      // Search for rows isolating an eigenvalue and push them down.
      noconv = false;
      for (blasint i = l; i >= 1; --i) {
        bool canswap = true;
        for (blasint j = 1; j <= l; ++j) {
          // (ZGEBAL tests the real and the imaginary part for zero.)
          if (i != j && A(i, j) != T(0)) {
            canswap = false;
            break;
          }
        }
        if (canswap) {
          SCALE(l) = R(i);
          if (i != l) {
            Swap(l, A.ptr(1, i), 1, A.ptr(1, l), 1);
            Swap(n - k + 1, A.ptr(i, k), lda, A.ptr(l, k), lda);
          }
          noconv = true;
          if (l == 1) {
            *ilo = 1;
            *ihi = 1;
            return info;
          }
          l = l - 1;
        }
      }
    }

    noconv = true;
    while (noconv) {
      // Search for columns isolating an eigenvalue and push them left.
      noconv = false;
      for (blasint j = k; j <= l; ++j) {
        bool canswap = true;
        for (blasint i = k; i <= l; ++i) {
          if (i != j && A(i, j) != T(0)) {
            canswap = false;
            break;
          }
        }
        if (canswap) {
          SCALE(k) = R(j);
          if (j != k) {
            Swap(l, A.ptr(1, j), 1, A.ptr(1, k), 1);
            Swap(n - k + 1, A.ptr(j, k), lda, A.ptr(k, k), lda);
          }
          noconv = true;
          k = k + 1;
        }
      }
    }
  }

  // Initialize SCALE for non-permuted submatrix.
  for (blasint i = k; i <= l; ++i) SCALE(i) = one;

  // If we only had to permute, we are done.
  if (Lsame(job, 'P')) {
    *ilo = k;
    *ihi = l;
    return info;
  }

  // Balance the submatrix in rows K to L.
  // Iterative loop for norm reduction.
  const R sfmin1 = Lamch<R>('S') / Lamch<R>('P');
  const R sfmax1 = one / sfmin1;
  const R sfmin2 = sfmin1 * sclfac;
  const R sfmax2 = one / sfmin2;

  bool noconv = true;
  while (noconv) {
    noconv = false;
    for (blasint i = k; i <= l; ++i) {
      R c = Nrm2(l - k + 1, A.ptr(k, i), 1);
      R r = Nrm2(l - k + 1, A.ptr(i, k), lda);
      const blasint ica = Iamax(l, A.ptr(1, i), 1);
      R ca = std::abs(A(ica, i));
      const blasint ira = Iamax(n - k + 1, A.ptr(i, k), lda);
      R ra = std::abs(A(i, ira + k - 1));

      // Guard against zero C or R due to underflow.
      if (c == zero || r == zero) continue;

      // Exit if NaN to avoid infinite loop.
      if (std::isnan(c + ca + r + ra)) {
        info = -3;
        Xerbla<T>("GEBAL", -info);
        return info;
      }

      R g = r / sclfac;
      R f = one;
      const R s = c + r;

      while (c < g && Max(f, c, ca) < sfmax2 && Min(r, g, ra) > sfmin2) {
        f = f * sclfac;
        c = c * sclfac;
        ca = ca * sclfac;
        r = r / sclfac;
        g = g / sclfac;
        ra = ra / sclfac;
      }

      g = c / sclfac;

      while (g >= r && Max(r, ra) < sfmax2 && Min(f, c, g, ca) > sfmin2) {
        f = f / sclfac;
        c = c / sclfac;
        g = g / sclfac;
        ca = ca / sclfac;
        r = r * sclfac;
        ra = ra * sclfac;
      }

      // Now balance.
      if ((c + r) >= factor * s) continue;
      if (f < one && SCALE(i) < one) {
        if (f * SCALE(i) <= sfmin1) continue;
      }
      if (f > one && SCALE(i) > one) {
        if (SCALE(i) >= sfmax1 / f) continue;
      }
      g = one / f;
      SCALE(i) = SCALE(i) * f;
      noconv = true;

      Rscal(n - k + 1, g, A.ptr(i, k), lda);
      Rscal(l, f, A.ptr(1, i), 1);
    }
  }

  *ilo = k;
  *ihi = l;
  return info;
}

// xGEBAK: forms the right (SIDE = 'R') or left (SIDE = 'L') eigenvectors
// of A from those of the matrix balanced by xGEBAL (JOB, ILO, IHI and
// SCALE as returned by xGEBAL): back-transforms the M columns of the
// N-by-M matrix V.
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename T>
blasint Gebak(char job, char side, blasint n, blasint ilo, blasint ihi,
              const RealT<T>* scale, blasint m, T* v, blasint ldv) {
  using R = RealT<T>;
  const R one = R(1);
  const Vector<const R> SCALE(scale);
  const Matrix<T> V(v, ldv);

  // Decode and test the input parameters.
  const bool rightv = Lsame(side, 'R');
  const bool leftv = Lsame(side, 'L');

  blasint info = 0;
  if (!Lsame(job, 'N') && !Lsame(job, 'P') && !Lsame(job, 'S') &&
      !Lsame(job, 'B')) {
    info = -1;
  } else if (!rightv && !leftv) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1 || ilo > Max<blasint>(1, n)) {
    info = -4;
  } else if (ihi < Min(ilo, n) || ihi > n) {
    info = -5;
  } else if (m < 0) {
    info = -7;
  } else if (ldv < Max<blasint>(1, n)) {
    info = -9;
  }
  if (info != 0) {
    Xerbla<T>("GEBAK", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;
  if (m == 0) return info;
  if (Lsame(job, 'N')) return info;

  if (ilo != ihi) {
    // Backward balance.
    if (Lsame(job, 'S') || Lsame(job, 'B')) {
      if (rightv) {
        for (blasint i = ilo; i <= ihi; ++i) {
          const R s = SCALE(i);
          Rscal(m, s, V.ptr(i, 1), ldv);
        }
      }
      if (leftv) {
        for (blasint i = ilo; i <= ihi; ++i) {
          const R s = one / SCALE(i);
          Rscal(m, s, V.ptr(i, 1), ldv);
        }
      }
    }
  }

  // Backward permutation:
  // for I = ILO-1 step -1 until 1, IHI+1 step 1 until N do --
  if (Lsame(job, 'P') || Lsame(job, 'B')) {
    if (rightv) {
      for (blasint ii = 1; ii <= n; ++ii) {
        blasint i = ii;
        if (i >= ilo && i <= ihi) continue;
        if (i < ilo) i = ilo - ii;
        const auto k = static_cast<blasint>(SCALE(i));
        if (k == i) continue;
        Swap(m, V.ptr(i, 1), ldv, V.ptr(k, 1), ldv);
      }
    }
    if (leftv) {
      for (blasint ii = 1; ii <= n; ++ii) {
        blasint i = ii;
        if (i >= ilo && i <= ihi) continue;
        if (i < ilo) i = ilo - ii;
        const auto k = static_cast<blasint>(SCALE(i));
        if (k == i) continue;
        Swap(m, V.ptr(i, 1), ldv, V.ptr(k, 1), ldv);
      }
    }
  }
  return info;
}

// xGEHD2: reduces rows and columns ILO:IHI of the N-by-N matrix A to upper
// Hessenberg form with the unblocked algorithm (see the top of the file);
// A is assumed upper triangular in rows and columns 1:ILO-1 and IHI+1:N
// (as after xGEBAL). TAU has N-1 elements, WORK N.
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename T>
blasint Gehd2(blasint n, blasint ilo, blasint ihi, T* a, blasint lda, T* tau,
              T* work) {
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);

  // Test the input parameters.
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (ilo < 1 || ilo > Max<blasint>(1, n)) {
    info = -2;
  } else if (ihi < Min(ilo, n) || ihi > n) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  }
  if (info != 0) {
    Xerbla<T>("GEHD2", -info);
    return info;
  }

  for (blasint i = ilo; i <= ihi - 1; ++i) {
    // Compute elementary reflector H(i) to annihilate A(i+2:ihi,i).
    Larfg(ihi - i, A.ptr(i + 1, i), A.ptr(Min<blasint>(i + 2, n), i), 1,
          TAU.ptr(i));

    // Apply H(i) to A(1:ihi,i+1:ihi) from the right.
    Larf1f('R', ihi, ihi - i, A.ptr(i + 1, i), 1, TAU(i), A.ptr(1, i + 1), lda,
           work);

    // Apply H(i)**H to A(i+1:ihi,i+1:n) from the left.
    Larf1f('L', ihi - i, n - i, A.ptr(i + 1, i), 1, Conj(TAU(i)),
           A.ptr(i + 1, i + 1), lda, work);
  }
  return info;
}

// xLAHR2: reduces the first NB columns of the N-by-(N-K+1) general matrix A
// so that elements below the K-th subdiagonal are zero, by a unitary
// similarity transformation Q**H * A * Q = I - V * T * V**H (Q = H(1) H(2)
// . . . H(nb), with v(1:i+k-1) = 0, v(i+k) = 1 and v(i+k+1:n) in
// A(i+k+1:n,i)), and returns the NB-by-NB upper triangular T (LDT >= NB)
// and the N-by-NB matrix Y = A * V * T (LDY >= N) needed to apply the
// transformation to the unreduced part of A. TAU has NB elements.
template <typename T>
void Lahr2(blasint n, blasint k, blasint nb, T* a, blasint lda, T* tau, T* t,
           blasint ldt, T* y, blasint ldy) {
  const T zero = T(0), one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  const Matrix<T> Tm(t, ldt);  // T(LDT, NB) in Fortran.
  const Matrix<T> Y(y, ldy);

  // Quick return if possible.
  if (n <= 1) return;

  T ei = zero;  // Set in the first iteration, used from the second on.
  for (blasint i = 1; i <= nb; ++i) {
    if (i > 1) {
      // Update A(K+1:N,I).

      // Update I-th column of A - Y * V**H.
      Lacgv(i - 1, A.ptr(k + i - 1, 1), lda);
      Gemv('N', n - k, i - 1, -one, Y.ptr(k + 1, 1), ldy, A.ptr(k + i - 1, 1),
           lda, one, A.ptr(k + 1, i), 1);
      Lacgv(i - 1, A.ptr(k + i - 1, 1), lda);

      // Apply I - V * T**H * V**H to this column (call it b) from the
      // left, using the last column of T as workspace.
      //
      // Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
      //          ( V2 )             ( b2 )
      //
      // where V1 is unit lower triangular.
      //
      // w := V1**H * b1.
      Copy(i - 1, A.ptr(k + 1, i), 1, Tm.ptr(1, nb), 1);
      Trmv('L', 'C', 'U', i - 1, A.ptr(k + 1, 1), lda, Tm.ptr(1, nb), 1);

      // w := w + V2**H * b2.
      Gemv('C', n - k - i + 1, i - 1, one, A.ptr(k + i, 1), lda,
           A.ptr(k + i, i), 1, one, Tm.ptr(1, nb), 1);

      // w := T**H * w.
      Trmv('U', 'C', 'N', i - 1, t, ldt, Tm.ptr(1, nb), 1);

      // b2 := b2 - V2*w.
      Gemv('N', n - k - i + 1, i - 1, -one, A.ptr(k + i, 1), lda, Tm.ptr(1, nb),
           1, one, A.ptr(k + i, i), 1);

      // b1 := b1 - V1*w.
      Trmv('L', 'N', 'U', i - 1, A.ptr(k + 1, 1), lda, Tm.ptr(1, nb), 1);
      Axpy(i - 1, -one, Tm.ptr(1, nb), 1, A.ptr(k + 1, i), 1);

      A(k + i - 1, i - 1) = ei;
    }

    // Generate the elementary reflector H(I) to annihilate
    // A(K+I+1:N,I).
    Larfg(n - k - i + 1, A.ptr(k + i, i), A.ptr(Min<blasint>(k + i + 1, n), i),
          1, TAU.ptr(i));
    ei = A(k + i, i);
    A(k + i, i) = one;

    // Compute Y(K+1:N,I).
    Gemv('N', n - k, n - k - i + 1, one, A.ptr(k + 1, i + 1), lda,
         A.ptr(k + i, i), 1, zero, Y.ptr(k + 1, i), 1);
    Gemv('C', n - k - i + 1, i - 1, one, A.ptr(k + i, 1), lda, A.ptr(k + i, i),
         1, zero, Tm.ptr(1, i), 1);
    Gemv('N', n - k, i - 1, -one, Y.ptr(k + 1, 1), ldy, Tm.ptr(1, i), 1, one,
         Y.ptr(k + 1, i), 1);
    Scal(n - k, TAU(i), Y.ptr(k + 1, i), 1);

    // Compute T(1:I,I).
    Scal(i - 1, -TAU(i), Tm.ptr(1, i), 1);
    Trmv('U', 'N', 'N', i - 1, t, ldt, Tm.ptr(1, i), 1);
    Tm(i, i) = TAU(i);
  }
  A(k + nb, nb) = ei;

  // Compute Y(1:K,1:NB).
  Lacpy('A', k, nb, A.ptr(1, 2), lda, y, ldy);
  Trmm('R', 'L', 'N', 'U', k, nb, one, A.ptr(k + 1, 1), lda, y, ldy);
  if (n > k + nb) {
    Gemm('N', 'N', k, nb, n - k - nb, one, A.ptr(1, 2 + nb), lda,
         A.ptr(k + 1 + nb, 1), lda, one, y, ldy);
  }
  Trmm('R', 'U', 'N', 'N', k, nb, one, t, ldt, y, ldy);
}

// xGEHRD: reduces rows and columns ILO:IHI of the N-by-N matrix A to upper
// Hessenberg form with the blocked algorithm (see xGEHD2). TAU has N-1
// elements; WORK has LWORK >= max(1, N) elements, N*NB + 65*64 for the
// optimal block size NB (LWORK = -1: workspace query).
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename T>
blasint Gehrd(blasint n, blasint ilo, blasint ihi, T* a, blasint lda, T* tau,
              T* work, blasint lwork) {
  constexpr blasint nbmax = 64;
  constexpr blasint ldt = nbmax + 1;
  constexpr blasint tsize = ldt * nbmax;
  const T zero = T(0), one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);

  // Test the input parameters.
  blasint info = 0;
  const bool lquery = (lwork == -1);
  if (n < 0) {
    info = -1;
  } else if (ilo < 1 || ilo > Max<blasint>(1, n)) {
    info = -2;
  } else if (ihi < Min(ilo, n) || ihi > n) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (lwork < Max<blasint>(1, n) && !lquery) {
    info = -8;
  }

  const blasint nh = ihi - ilo + 1;
  blasint lwkopt = 1;
  if (info == 0) {
    // Compute the workspace requirements.
    if (nh <= 1) {
      lwkopt = 1;
    } else {
      const blasint nb =
          Min(nbmax, Ilaenv<T>(1, "GEHRD", " ", n, ilo, ihi, -1));
      lwkopt = n * nb + tsize;
    }
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>("GEHRD", -info);
    return info;
  }
  if (lquery) return info;

  // Set elements 1:ILO-1 and IHI:N-1 of TAU to zero.
  for (blasint i = 1; i <= ilo - 1; ++i) TAU(i) = zero;
  for (blasint i = Max<blasint>(1, ihi); i <= n - 1; ++i) TAU(i) = zero;

  // Quick return if possible.
  if (nh <= 1) {
    WORK(1) = T(1);
    return info;
  }

  // Determine the block size.
  blasint nb = Min(nbmax, Ilaenv<T>(1, "GEHRD", " ", n, ilo, ihi, -1));
  blasint nbmin = 2;
  blasint nx = 0;  // Set (and used) only if nb > 1 and nb < nh.
  if (nb > 1 && nb < nh) {
    // Determine when to cross over from blocked to unblocked code (last
    // block is always handled by unblocked code).
    nx = Max(nb, Ilaenv<T>(3, "GEHRD", " ", n, ilo, ihi, -1));
    if (nx < nh) {
      // Determine if workspace is large enough for blocked code.
      if (lwork < lwkopt) {
        // Not enough workspace to use optimal NB: determine the minimum
        // value of NB, and reduce NB or force use of unblocked code.
        nbmin = Max<blasint>(2, Ilaenv<T>(2, "GEHRD", " ", n, ilo, ihi, -1));
        if (lwork >= (n * nbmin + tsize)) {
          nb = (lwork - tsize) / n;
        } else {
          nb = 1;
        }
      }
    }
  }
  const blasint ldwork = n;

  blasint i;
  if (nb < nbmin || nb >= nh) {
    // Use unblocked code below.
    i = ilo;
  } else {
    // Use blocked code.
    const blasint iwt = 1 + n * nb;
    for (i = ilo; i <= ihi - 1 - nx; i += nb) {
      const blasint ib = Min(nb, ihi - i);

      // Reduce columns i:i+ib-1 to Hessenberg form, returning the matrices
      // V and T of the block reflector H = I - V*T*V**H which performs the
      // reduction, and also the matrix Y = A*V*T.
      Lahr2(ihi, i, ib, A.ptr(1, i), lda, TAU.ptr(i), WORK.ptr(iwt), ldt, work,
            ldwork);

      // Apply the block reflector H to A(1:ihi,i+ib:ihi) from the right,
      // computing A := A - Y * V**H. V(i+ib,ib-1) must be set to 1.
      const T ei = A(i + ib, i + ib - 1);
      A(i + ib, i + ib - 1) = one;
      Gemm('N', 'C', ihi, ihi - i - ib + 1, ib, -one, work, ldwork,
           A.ptr(i + ib, i), lda, one, A.ptr(1, i + ib), lda);
      A(i + ib, i + ib - 1) = ei;

      // Apply the block reflector H to A(1:i,i+1:i+ib-1) from the right.
      Trmm('R', 'L', 'C', 'U', i, ib - 1, one, A.ptr(i + 1, i), lda, work,
           ldwork);
      for (blasint j = 0; j <= ib - 2; ++j) {
        Axpy(i, -one, WORK.ptr(ldwork * j + 1), 1, A.ptr(1, i + j + 1), 1);
      }

      // Apply the block reflector H to A(i+1:ihi,i+ib:n) from the left.
      Larfb('L', 'C', 'F', 'C', ihi - i, n - i - ib + 1, ib, A.ptr(i + 1, i),
            lda, WORK.ptr(iwt), ldt, A.ptr(i + 1, i + ib), lda, work, ldwork);
    }
  }

  // Use unblocked code to reduce the rest of the matrix.
  Gehd2(n, i, ihi, a, lda, tau, work);

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORGHR / xUNGHR: generates the N-by-N matrix Q of xGEHRD (ILO, IHI, the
// reflectors in A and TAU as returned by it) in A. WORK has LWORK >=
// max(1, IHI-ILO) elements, (IHI-ILO)*NB for the optimal block size NB
// (LWORK = -1: workspace query).
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename T>
blasint Unghr(blasint n, blasint ilo, blasint ihi, T* a, blasint lda,
              const T* tau, T* work, blasint lwork) {
  const T zero = T(0), one = T(1);
  const Matrix<T> A(a, lda);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const blasint nh = ihi - ilo;
  const bool lquery = (lwork == -1);
  if (n < 0) {
    info = -1;
  } else if (ilo < 1 || ilo > Max<blasint>(1, n)) {
    info = -2;
  } else if (ihi < Min(ilo, n) || ihi > n) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (lwork < Max<blasint>(1, nh) && !lquery) {
    info = -8;
  }

  blasint lwkopt = 1;
  if (info == 0) {
    const blasint nb =
        Ilaenv<T>(1, RealOrComplex<T>("ORGQR", "UNGQR"), " ", nh, nh, nh, -1);
    lwkopt = Max<blasint>(1, nh) * nb;
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORGHR", "UNGHR"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) {
    WORK(1) = T(1);
    return info;
  }

  // Shift the vectors which define the elementary reflectors one column to
  // the right, and set the first ilo and the last n-ihi rows and columns
  // to those of the unit matrix.
  for (blasint j = ihi; j >= ilo + 1; --j) {
    for (blasint i = 1; i <= j - 1; ++i) A(i, j) = zero;
    for (blasint i = j + 1; i <= ihi; ++i) A(i, j) = A(i, j - 1);
    for (blasint i = ihi + 1; i <= n; ++i) A(i, j) = zero;
  }
  for (blasint j = 1; j <= ilo; ++j) {
    for (blasint i = 1; i <= n; ++i) A(i, j) = zero;
    A(j, j) = one;
  }
  for (blasint j = ihi + 1; j <= n; ++j) {
    for (blasint i = 1; i <= n; ++i) A(i, j) = zero;
    A(j, j) = one;
  }

  if (nh > 0) {
    // Generate Q(ilo+1:ihi,ilo+1:ihi).
    Ungqr(nh, nh, nh, A.ptr(ilo + 1, ilo + 1), lda, TAU.ptr(ilo), work, lwork);
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xORMHR / xUNMHR: overwrites the M-by-N matrix C with Q * C, Q**H * C
// (SIDE = 'L'), C * Q or C * Q**H (SIDE = 'R'), for the matrix Q of
// xGEHRD (ILO, IHI, the reflectors in A and TAU as returned by it). TRANS
// = 'T' (real T) or 'C' (complex T) selects Q**H. WORK has LWORK >=
// max(1, N) (SIDE = 'L') or max(1, M) (SIDE = 'R') elements, times NB for
// the optimal block size NB (LWORK = -1: workspace query).
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename T>
blasint Unmhr(char side, char trans, blasint m, blasint n, blasint ilo,
              blasint ihi, const T* a, blasint lda, const T* tau, T* c,
              blasint ldc, T* work, blasint lwork) {
  const Matrix<const T> A(a, lda);
  const Matrix<T> C(c, ldc);
  const Vector<const T> TAU(tau);
  const Vector<T> WORK(work);

  // Test the input arguments.
  blasint info = 0;
  const blasint nh = ihi - ilo;
  const bool left = Lsame(side, 'L');
  const bool lquery = (lwork == -1);

  // NQ is the order of Q and NW is the minimum dimension of WORK.
  blasint nq, nw;
  if (left) {
    nq = m;
    nw = Max<blasint>(1, n);
  } else {
    nq = n;
    nw = Max<blasint>(1, m);
  }
  if (!left && !Lsame(side, 'R')) {
    info = -1;
  } else if (!Lsame(trans, 'N') && !Lsame(trans, ConjTrans<T>())) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (ilo < 1 || ilo > Max<blasint>(1, nq)) {
    info = -5;
  } else if (ihi < Min(ilo, nq) || ihi > nq) {
    info = -6;
  } else if (lda < Max<blasint>(1, nq)) {
    info = -8;
  } else if (ldc < Max<blasint>(1, m)) {
    info = -11;
  } else if (lwork < nw && !lquery) {
    info = -13;
  }

  blasint lwkopt = 1;
  if (info == 0) {
    const char opts[3] = {side, trans, '\0'};
    const char* name = RealOrComplex<T>("ORMQR", "UNMQR");
    blasint nb;
    if (left) {
      nb = Ilaenv<T>(1, name, opts, nh, n, nh, -1);
    } else {
      nb = Ilaenv<T>(1, name, opts, m, nh, nh, -1);
    }
    lwkopt = nw * nb;
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("ORMHR", "UNMHR"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (m == 0 || n == 0 || nh == 0) {
    WORK(1) = T(1);
    return info;
  }

  blasint mi, ni, i1, i2;
  if (left) {
    mi = nh;
    ni = n;
    i1 = ilo + 1;
    i2 = 1;
  } else {
    mi = m;
    ni = nh;
    i1 = 1;
    i2 = ilo + 1;
  }

  Unmqr(side, trans, mi, ni, nh, A.ptr(ilo + 1, ilo), lda, TAU.ptr(ilo),
        C.ptr(i1, i2), ldc, work, lwork);

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Gebal<T>(char, blasint, T*, blasint, blasint*, blasint*,   \
                            RealT<T>*);                                       \
  template blasint Gebak<T>(char, char, blasint, blasint, blasint,            \
                            const RealT<T>*, blasint, T*, blasint);           \
  template blasint Gehd2<T>(blasint, blasint, blasint, T*, blasint, T*, T*);  \
  template void Lahr2<T>(blasint, blasint, blasint, T*, blasint, T*, T*,      \
                         blasint, T*, blasint);                               \
  template blasint Gehrd<T>(blasint, blasint, blasint, T*, blasint, T*, T*,   \
                            blasint);                                         \
  template blasint Unghr<T>(blasint, blasint, blasint, T*, blasint, const T*, \
                            T*, blasint);                                     \
  template blasint Unmhr<T>(char, char, blasint, blasint, blasint, blasint,   \
                            const T*, blasint, const T*, T*, blasint, T*,     \
                            blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// xORGHR / xORMHR for real T, xUNGHR / xUNMHR for complex T: `og` is "org"
// or "ung", `om` "orm" or "unm".
#define HWYBLAS_LAPACK_HESSENBERG(prefix, og, om, T, R)                        \
  HWYBLAS_API void prefix##gebal_(                                             \
      const char* job, const hwyblas_int* n, R* a, const hwyblas_int* lda,     \
      hwyblas_int* ilo, hwyblas_int* ihi, R* scale, hwyblas_int* info) {       \
    *info = hwyblas::lapack::Gebal(*job, *n, As<T>(a), *lda, ilo, ihi, scale); \
  }                                                                            \
  HWYBLAS_API void prefix##gebak_(                                             \
      const char* job, const char* side, const hwyblas_int* n,                 \
      const hwyblas_int* ilo, const hwyblas_int* ihi, const R* scale,          \
      const hwyblas_int* m, R* v, const hwyblas_int* ldv, hwyblas_int* info) { \
    *info = hwyblas::lapack::Gebak(*job, *side, *n, *ilo, *ihi, scale, *m,     \
                                   As<T>(v), *ldv);                            \
  }                                                                            \
  HWYBLAS_API void prefix##gehd2_(                                             \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* a, const hwyblas_int* lda, R* tau, R* work, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Gehd2(*n, *ilo, *ihi, As<T>(a), *lda, As<T>(tau), \
                                   As<T>(work));                               \
  }                                                                            \
  HWYBLAS_API void prefix##lahr2_(                                             \
      const hwyblas_int* n, const hwyblas_int* k, const hwyblas_int* nb, R* a, \
      const hwyblas_int* lda, R* tau, R* t, const hwyblas_int* ldt, R* y,      \
      const hwyblas_int* ldy) {                                                \
    hwyblas::lapack::Lahr2(*n, *k, *nb, As<T>(a), *lda, As<T>(tau), As<T>(t),  \
                           *ldt, As<T>(y), *ldy);                              \
  }                                                                            \
  HWYBLAS_API void prefix##gehrd_(                                             \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* a, const hwyblas_int* lda, R* tau, R* work, const hwyblas_int* lwork, \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Gehrd(*n, *ilo, *ihi, As<T>(a), *lda, As<T>(tau), \
                                   As<T>(work), *lwork);                       \
  }                                                                            \
  HWYBLAS_API void prefix##og##hr_(                                            \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      R* a, const hwyblas_int* lda, const R* tau, R* work,                     \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Unghr(*n, *ilo, *ihi, As<T>(a), *lda,             \
                                   As<const T>(tau), As<T>(work), *lwork);     \
  }                                                                            \
  HWYBLAS_API void prefix##om##hr_(                                            \
      const char* side, const char* trans, const hwyblas_int* m,               \
      const hwyblas_int* n, const hwyblas_int* ilo, const hwyblas_int* ihi,    \
      const R* a, const hwyblas_int* lda, const R* tau, R* c,                  \
      const hwyblas_int* ldc, R* work, const hwyblas_int* lwork,               \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Unmhr(*side, *trans, *m, *n, *ilo, *ihi,          \
                                   As<const T>(a), *lda, As<const T>(tau),     \
                                   As<T>(c), *ldc, As<T>(work), *lwork);       \
  }

HWYBLAS_LAPACK_HESSENBERG(s, org, orm, float, float)
HWYBLAS_LAPACK_HESSENBERG(d, org, orm, double, double)
HWYBLAS_LAPACK_HESSENBERG(c, ung, unm, c32, float)
HWYBLAS_LAPACK_HESSENBERG(z, ung, unm, c64, double)

#undef HWYBLAS_LAPACK_HESSENBERG

}  // extern "C"
