// QR factorization with column pivoting: xGEQP3 (blocked), xLAQPS (one
// block of columns) and xLAQP2 (unblocked).
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

// xLAQP2: computes a QR factorization with column pivoting of the block
// A(OFFSET+1:M,1:N), with the unblocked algorithm. The block A(1:OFFSET,1:N)
// is accordingly pivoted, but not factorized.
//
// JPVT(i) is the original column of the i-th column of A*P (updated in
// place). TAU(1:min(M-OFFSET,N)) are the scalars of the reflectors (see
// xGEQRF). VN1 and VN2 hold the partial and the exact column norms (updated
// for the remaining columns); WORK has N elements.
//
// The norms of the remaining columns are downdated after each step, and
// recomputed where cancellation makes the downdate inaccurate (LAPACK
// Working Note 176).
template <typename T>
void Laqp2(blasint m, blasint n, blasint offset, T* a, blasint lda,
           blasint* jpvt, T* tau, RealT<T>* vn1, RealT<T>* vn2, T* work) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> JPVT(jpvt);
  const Vector<T> TAU(tau);
  const Vector<R> VN1(vn1);
  const Vector<R> VN2(vn2);

  const blasint mn = std::min(m - offset, n);
  const R tol3z = std::sqrt(Lamch<R>('E'));

  // Compute factorization.
  for (blasint i = 1; i <= mn; ++i) {
    const blasint offpi = offset + i;

    // Determine ith pivot column and swap if necessary.
    const blasint pvt = (i - 1) + Iamax(n - i + 1, VN1.ptr(i), 1);

    if (pvt != i) {
      Swap(m, A.ptr(1, pvt), 1, A.ptr(1, i), 1);
      const blasint itemp = JPVT(pvt);
      JPVT(pvt) = JPVT(i);
      JPVT(i) = itemp;
      VN1(pvt) = VN1(i);
      VN2(pvt) = VN2(i);
    }

    // Generate elementary reflector H(i).
    if (offpi < m) {
      Larfg(m - offpi + 1, A.ptr(offpi, i), A.ptr(offpi + 1, i), 1, TAU.ptr(i));
    } else {
      Larfg(1, A.ptr(m, i), A.ptr(m, i), 1, TAU.ptr(i));
    }

    if (i < n) {
      // Apply H(i)**H to A(offset+i:m,i+1:n) from the left.
      Larf1f('L', m - offpi + 1, n - i, A.ptr(offpi, i), 1, Conj(TAU(i)),
             A.ptr(offpi, i + 1), lda, work);
    }

    // Update partial column norms.
    for (blasint j = i + 1; j <= n; ++j) {
      if (VN1(j) != zero) {
        // NOTE: The following 4 lines follow from the analysis in
        // Lapack Working Note 176.
        const R ratio = std::abs(A(offpi, j)) / VN1(j);
        R temp = one - ratio * ratio;
        temp = std::max(temp, zero);
        const R vratio = VN1(j) / VN2(j);
        const R temp2 = temp * (vratio * vratio);
        if (temp2 <= tol3z) {
          if (offpi < m) {
            VN1(j) = Nrm2(m - offpi, A.ptr(offpi + 1, j), 1);
            VN2(j) = VN1(j);
          } else {
            VN1(j) = zero;
            VN2(j) = zero;
          }
        } else {
          VN1(j) = VN1(j) * std::sqrt(temp);
        }
      }
    }
  }
}

// xLAQPS: computes a step of QR factorization with column pivoting of the
// block A(OFFSET+1:M,1:N) by using Level 3 BLAS: it tries to factorize NB
// columns from A(OFFSET+1:M,1:N), and returns the number KB of columns
// actually factorized (fewer than NB when the norm downdates of the
// remaining columns become inaccurate). The block A(1:OFFSET,1:N) is
// accordingly pivoted, but not factorized.
//
// JPVT, TAU, VN1 and VN2 are as in xLAQP2. AUXV has NB elements; F
// (LDF-by-NB, LDF >= N) holds F**H, where A(RK:M, KB+1:N) is updated with
// the rank-KB product A(RK:M, 1:KB) * F(KB+1:N, 1:KB)**H.
//
// The real routine updates the pivot row with xGEMV and the complex one
// with an xGEMM of one row; the template does the same for each.
template <typename T>
blasint Laqps(blasint m, blasint n, blasint offset, blasint nb, T* a,
              blasint lda, blasint* jpvt, T* tau, RealT<T>* vn1, RealT<T>* vn2,
              T* auxv, T* f, blasint ldf) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1);
  const T czero = T(0), cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> JPVT(jpvt);
  const Vector<T> TAU(tau);
  const Vector<R> VN1(vn1);
  const Vector<R> VN2(vn2);
  const Matrix<T> F(f, ldf);

  const blasint lastrk = std::min(m, n + offset);
  blasint lsticc = 0;
  blasint k = 0;
  const R tol3z = std::sqrt(Lamch<R>('E'));
  blasint rk = 0;

  // Beginning of while loop.
  while (k < nb && lsticc == 0) {
    k = k + 1;
    rk = offset + k;

    // Determine ith pivot column and swap if necessary.
    const blasint pvt = (k - 1) + Iamax(n - k + 1, VN1.ptr(k), 1);
    if (pvt != k) {
      Swap(m, A.ptr(1, pvt), 1, A.ptr(1, k), 1);
      Swap(k - 1, F.ptr(pvt, 1), ldf, F.ptr(k, 1), ldf);
      const blasint itemp = JPVT(pvt);
      JPVT(pvt) = JPVT(k);
      JPVT(k) = itemp;
      VN1(pvt) = VN1(k);
      VN2(pvt) = VN2(k);
    }

    // Apply previous Householder reflectors to column K:
    // A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**H.
    if (k > 1) {
      for (blasint j = 1; j <= k - 1; ++j) F(k, j) = Conj(F(k, j));
      Gemv('N', m - rk + 1, k - 1, -cone, A.ptr(rk, 1), lda, F.ptr(k, 1), ldf,
           cone, A.ptr(rk, k), 1);
      for (blasint j = 1; j <= k - 1; ++j) F(k, j) = Conj(F(k, j));
    }

    // Generate elementary reflector H(k).
    if (rk < m) {
      Larfg(m - rk + 1, A.ptr(rk, k), A.ptr(rk + 1, k), 1, TAU.ptr(k));
    } else {
      Larfg(1, A.ptr(rk, k), A.ptr(rk, k), 1, TAU.ptr(k));
    }

    const T akk = A(rk, k);
    A(rk, k) = cone;

    // Compute Kth column of F:
    // F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**H*A(RK:M,K).
    if (k < n) {
      Gemv('C', m - rk + 1, n - k, TAU(k), A.ptr(rk, k + 1), lda, A.ptr(rk, k),
           1, czero, F.ptr(k + 1, k), 1);
    }

    // Padding F(1:K,K) with zeros.
    for (blasint j = 1; j <= k; ++j) F(j, k) = czero;

    // Incremental updating of F:
    // F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)**H
    //             *A(RK:M,K).
    if (k > 1) {
      Gemv('C', m - rk + 1, k - 1, -TAU(k), A.ptr(rk, 1), lda, A.ptr(rk, k), 1,
           czero, auxv, 1);
      Gemv('N', n, k - 1, cone, F.ptr(1, 1), ldf, auxv, 1, cone, F.ptr(1, k),
           1);
    }

    // Update the current row of A:
    // A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)**H.
    if (k < n) {
      if constexpr (IsComplex<T>::value) {
        Gemm('N', 'C', 1, n - k, k, -cone, A.ptr(rk, 1), lda, F.ptr(k + 1, 1),
             ldf, cone, A.ptr(rk, k + 1), lda);
      } else {
        Gemv('N', n - k, k, -one, F.ptr(k + 1, 1), ldf, A.ptr(rk, 1), lda, one,
             A.ptr(rk, k + 1), lda);
      }
    }

    // Update partial column norms.
    if (rk < lastrk) {
      for (blasint j = k + 1; j <= n; ++j) {
        if (VN1(j) != zero) {
          // NOTE: The following 4 lines follow from the analysis in
          // Lapack Working Note 176.
          R temp = std::abs(A(rk, j)) / VN1(j);
          temp = std::max(zero, (one + temp) * (one - temp));
          const R vratio = VN1(j) / VN2(j);
          const R temp2 = temp * (vratio * vratio);
          if (temp2 <= tol3z) {
            // VN2(J) links the columns whose norms must be recomputed.
            VN2(j) = R(lsticc);
            lsticc = j;
          } else {
            VN1(j) = VN1(j) * std::sqrt(temp);
          }
        }
      }
    }

    A(rk, k) = akk;
  }
  const blasint kb = k;
  rk = offset + kb;

  // Apply the block reflector to the rest of the matrix:
  // A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
  //                     A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)**H.
  if (kb < std::min(n, m - offset)) {
    Gemm('N', 'C', m - rk, n - kb, kb, -cone, A.ptr(rk + 1, 1), lda,
         F.ptr(kb + 1, 1), ldf, cone, A.ptr(rk + 1, kb + 1), lda);
  }

  // Recomputation of difficult columns.
  while (lsticc > 0) {
    const auto itemp = static_cast<blasint>(std::lround(VN2(lsticc)));
    VN1(lsticc) = Nrm2(m - rk, A.ptr(rk + 1, lsticc), 1);

    // NOTE: The computation of VN1( LSTICC ) relies on the fact that
    // SNRM2 does not fail on vectors with norm below the value of
    // SQRT(DLAMCH('S')).
    VN2(lsticc) = VN1(lsticc);
    lsticc = itemp;
  }
  return kb;
}

// xGEQP3: computes a QR factorization with column pivoting of the M-by-N
// matrix A: A*P = Q*R, using Level 3 BLAS.
//
// On entry, JPVT(j) != 0 moves column j of A to the front of A*P (a
// leading column), JPVT(j) = 0 makes it a free column. On exit, JPVT(j) =
// k if the j-th column of A*P was the k-th column of A. The upper triangle
// of A holds R and the rest, with TAU, the reflectors of Q (see xGEQRF).
//
// LWORK >= 3*N+1 (real) or N+1 (complex); the optimal LWORK is
// 2*N+(N+1)*NB (real) or (N+1)*NB (complex); LWORK = -1 is a workspace
// query. RWORK (2*N elements, complex T only) holds the column norms, which
// the real routine keeps in WORK. On exit, WORK(1) is the optimal LWORK, or
// (real T) the workspace size that the routine used.
template <typename T>
blasint Geqp3(blasint m, blasint n, T* a, blasint lda, blasint* jpvt, T* tau,
              T* work, blasint lwork, RealT<T>* rwork) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  constexpr blasint inb = 1, inbmin = 2, ixover = 3;
  const Matrix<T> A(a, lda);
  const Vector<blasint> JPVT(jpvt);
  const Vector<T> TAU(tau);
  const Vector<T> WORK(work);

  // Test input arguments.
  blasint info = 0;
  const bool lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < std::max<blasint>(1, m)) {
    info = -4;
  }

  const blasint minmn = std::min(m, n);
  blasint iws = 1;
  blasint lwkopt = 1;
  blasint nb = 0;
  if (info == 0) {
    if (minmn == 0) {
      iws = 1;
      lwkopt = 1;
    } else {
      iws = kComplex ? n + 1 : 3 * n + 1;
      nb = Ilaenv<T>(inb, "GEQRF", " ", m, n, -1, -1);
      lwkopt = kComplex ? (n + 1) * nb : 2 * n + (n + 1) * nb;
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (lwork < iws && !lquery) info = -8;
  }

  if (info != 0) {
    Xerbla<T>("GEQP3", -info);
    return info;
  }
  if (lquery) return info;

  // Move initial columns up front.
  blasint nfxd = 1;
  for (blasint j = 1; j <= n; ++j) {
    if (JPVT(j) != 0) {
      if (j != nfxd) {
        Swap(m, A.ptr(1, j), 1, A.ptr(1, nfxd), 1);
        JPVT(j) = JPVT(nfxd);
        JPVT(nfxd) = j;
      } else {
        JPVT(j) = j;
      }
      nfxd = nfxd + 1;
    } else {
      JPVT(j) = j;
    }
  }
  nfxd = nfxd - 1;

  // Factorize fixed columns
  // =======================
  //
  // Compute the QR factorization of fixed columns and update remaining
  // columns.
  if (nfxd > 0) {
    const blasint na = std::min(m, nfxd);
    Geqrf(m, na, a, lda, tau, work, lwork);
    iws = std::max(iws, static_cast<blasint>(std::real(WORK(1))));
    if (na < n) {
      Unmqr('L', ConjTrans<T>(), m, n - na, na, a, lda, tau, A.ptr(1, na + 1),
            lda, work, lwork);
      iws = std::max(iws, static_cast<blasint>(std::real(WORK(1))));
    }
  }

  // Factorize free columns
  // ======================
  if (nfxd < minmn) {
    const blasint sm = m - nfxd;
    const blasint sn = n - nfxd;
    const blasint sminmn = minmn - nfxd;

    // Determine the block size.
    nb = Ilaenv<T>(inb, "GEQRF", " ", sm, sn, -1, -1);
    blasint nbmin = 2;
    blasint nx = 0;

    if (nb > 1 && nb < sminmn) {
      // Determine when to cross over from blocked to unblocked code.
      nx =
          std::max<blasint>(0, Ilaenv<T>(ixover, "GEQRF", " ", sm, sn, -1, -1));

      if (nx < sminmn) {
        // Determine if workspace is large enough for blocked code.
        //
        // Deviation: the real routine (LAPACK 3.12.1) uses 2*SN here,
        // although it keeps the norms of all N columns in WORK(1:2*N) and
        // xLAQPS's workspace after them; with fixed columns (NFXD > 0) and
        // less than the optimal LWORK, xLAQPS then writes up to 2*NFXD
        // elements past the end of WORK. The template uses 2*N.
        const blasint minws = kComplex ? (sn + 1) * nb : 2 * n + (sn + 1) * nb;
        iws = std::max(iws, minws);
        if (lwork < minws) {
          // Not enough workspace to use optimal NB: Reduce NB and
          // determine the minimum value of NB.
          nb = kComplex ? lwork / (sn + 1) : (lwork - 2 * n) / (sn + 1);
          nbmin = std::max<blasint>(
              2, Ilaenv<T>(inbmin, "GEQRF", " ", sm, sn, -1, -1));
        }
      }
    }

    // The column norms: in RWORK (complex T) or at the start of WORK (real
    // T), followed by the workspace of xLAQPS and xLAQP2.
    R* norms;
    T* rest;
    if constexpr (kComplex) {
      norms = rwork;
      rest = work;
    } else {
      norms = work;
      rest = WORK.ptr(2 * n + 1);
    }
    const Vector<R> NORMS(norms);

    // Initialize partial column norms. The first N elements of work store
    // the exact column norms.
    for (blasint j = nfxd + 1; j <= n; ++j) {
      NORMS(j) = Nrm2(sm, A.ptr(nfxd + 1, j), 1);
      NORMS(n + j) = NORMS(j);
    }

    blasint j = nfxd + 1;
    if (nb >= nbmin && nb < sminmn && nx < sminmn) {
      // Use blocked code initially.
      const blasint topbmn = minmn - nx;
      while (j <= topbmn) {
        const blasint jb = std::min(nb, topbmn - j + 1);

        // Factorize JB columns among columns J:N.
        const blasint fjb = Laqps(m, n - j + 1, j - 1, jb, A.ptr(1, j), lda,
                                  JPVT.ptr(j), TAU.ptr(j), NORMS.ptr(j),
                                  NORMS.ptr(n + j), rest, rest + jb, n - j + 1);

        j = j + fjb;
      }
    }

    // Use unblocked code to factor the last or only block.
    if (j <= minmn) {
      Laqp2(m, n - j + 1, j - 1, A.ptr(1, j), lda, JPVT.ptr(j), TAU.ptr(j),
            NORMS.ptr(j), NORMS.ptr(n + j), rest);
    }
  }

  WORK(1) = WorkSize<T>(kComplex ? lwkopt : iws);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template void Laqp2<T>(blasint, blasint, blasint, T*, blasint, blasint*, T*, \
                         RealT<T>*, RealT<T>*, T*);                            \
  template blasint Laqps<T>(blasint, blasint, blasint, blasint, T*, blasint,   \
                            blasint*, T*, RealT<T>*, RealT<T>*, T*, T*,        \
                            blasint);                                          \
  template blasint Geqp3<T>(blasint, blasint, T*, blasint, blasint*, T*, T*,   \
                            blasint, RealT<T>*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GEQP3_COMMON(prefix, T, R)                             \
  HWYBLAS_API void prefix##laqp2_(const hwyblas_int* m, const hwyblas_int* n, \
                                  const hwyblas_int* offset, R* a,            \
                                  const hwyblas_int* lda, hwyblas_int* jpvt,  \
                                  R* tau, R* vn1, R* vn2, R* work) {          \
    hwyblas::lapack::Laqp2(*m, *n, *offset, As<T>(a), *lda, jpvt, As<T>(tau), \
                           vn1, vn2, As<T>(work));                            \
  }                                                                           \
  HWYBLAS_API void prefix##laqps_(                                            \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* offset,  \
      const hwyblas_int* nb, hwyblas_int* kb, R* a, const hwyblas_int* lda,   \
      hwyblas_int* jpvt, R* tau, R* vn1, R* vn2, R* auxv, R* f,               \
      const hwyblas_int* ldf) {                                               \
    *kb = hwyblas::lapack::Laqps(*m, *n, *offset, *nb, As<T>(a), *lda, jpvt,  \
                                 As<T>(tau), vn1, vn2, As<T>(auxv), As<T>(f), \
                                 *ldf);                                       \
  }

HWYBLAS_LAPACK_GEQP3_COMMON(s, float, float)
HWYBLAS_LAPACK_GEQP3_COMMON(d, double, double)
HWYBLAS_LAPACK_GEQP3_COMMON(c, c32, float)
HWYBLAS_LAPACK_GEQP3_COMMON(z, c64, double)

#undef HWYBLAS_LAPACK_GEQP3_COMMON

HWYBLAS_API void sgeqp3_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, float* tau,
                         float* work, const hwyblas_int* lwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Geqp3(*m, *n, a, *lda, jpvt, tau, work, *lwork,
                                 static_cast<float*>(nullptr));
}
HWYBLAS_API void dgeqp3_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, double* tau,
                         double* work, const hwyblas_int* lwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Geqp3(*m, *n, a, *lda, jpvt, tau, work, *lwork,
                                 static_cast<double*>(nullptr));
}
HWYBLAS_API void cgeqp3_(const hwyblas_int* m, const hwyblas_int* n, float* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, float* tau,
                         float* work, const hwyblas_int* lwork, float* rwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Geqp3(*m, *n, As<c32>(a), *lda, jpvt, As<c32>(tau),
                                 As<c32>(work), *lwork, rwork);
}
HWYBLAS_API void zgeqp3_(const hwyblas_int* m, const hwyblas_int* n, double* a,
                         const hwyblas_int* lda, hwyblas_int* jpvt, double* tau,
                         double* work, const hwyblas_int* lwork, double* rwork,
                         hwyblas_int* info) {
  *info = hwyblas::lapack::Geqp3(*m, *n, As<c64>(a), *lda, jpvt, As<c64>(tau),
                                 As<c64>(work), *lwork, rwork);
}

}  // extern "C"
