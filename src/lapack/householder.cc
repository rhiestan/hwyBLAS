// Elementary reflectors (Householder transformations): xLARFG (generate),
// xLARF, xLARF1F and xLARF1L (apply one), xLARFT (form the triangular
// factor of a block reflector) and xLARFB (apply a block reflector).
//
// A reflector is H = I - tau * v * v**H, with a scalar tau and a vector v;
// for complex T, H is not Hermitian in general, and H**H (with conj(tau)) is
// applied where needed.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLARFG: generates an elementary reflector H of order N, such that
//
//     H**H * ( alpha ) = ( beta ),   H**H * H = I,
//            (   x   )   (   0  )
//
// where alpha and beta are scalars, with beta real, and x is an (N-1)-element
// vector. H is represented in the form H = I - tau * ( 1 ) * ( 1 v**H ),
//                                                    ( v )
// where tau is a scalar and v is an (N-1)-element vector; for complex T, tau
// is complex and H is not Hermitian.
//
// If the elements of x are all zero (and, for complex T, alpha is real),
// then tau = 0 and H is taken to be the unit matrix. Otherwise
// 1 <= real(tau) <= 2 and abs(tau - 1) <= 1.
//
// On exit, ALPHA is overwritten with beta, and X with v.
//
// The real and the complex routines differ in the order N below which H is
// the unit matrix (N <= 1 for real T, N <= 0 for complex T, where H also
// makes a scalar alpha real).
template <typename T>
void Larfg(blasint n, T* alpha, T* x, blasint incx, T* tau) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const R one = R(1), zero = R(0);

  if (n <= (kComplex ? 0 : 1)) {
    *tau = T(zero);
    return;
  }

  R xnorm = Nrm2(n - 1, x, incx);
  R alphr = std::real(*alpha);
  R alphi = kComplex ? std::imag(*alpha) : zero;

  // The real routine compares only xnorm (alphi is zero).
  if (xnorm == zero && alphi == zero) {
    // H = I.
    *tau = T(zero);
  } else {
    // General case.
    const auto norm = [&] {
      if constexpr (IsComplex<T>::value) {
        return Lapy3(alphr, alphi, xnorm);
      } else {
        return Lapy2(alphr, xnorm);
      }
    };
    R beta = -std::copysign(norm(), alphr);
    const R safmin = Lamch<R>('S') / Lamch<R>('E');
    const R rsafmn = one / safmin;

    blasint knt = 0;
    if (std::abs(beta) < safmin) {
      // XNORM, BETA may be inaccurate; scale X and recompute them.
      do {
        knt = knt + 1;
        Rscal(n - 1, rsafmn, x, incx);
        beta = beta * rsafmn;
        alphi = alphi * rsafmn;
        alphr = alphr * rsafmn;
      } while (std::abs(beta) < safmin && knt < 20);

      // New BETA is at most 1, at least SAFMIN.
      xnorm = Nrm2(n - 1, x, incx);
      if constexpr (kComplex) *alpha = T(alphr, alphi);
      beta = -std::copysign(norm(), alphr);
    }
    if constexpr (kComplex) {
      *tau = T((beta - alphr) / beta, -alphi / beta);
      *alpha = Ladiv(T(one), T(alphr, alphi) - beta);
      Scal(n - 1, *alpha, x, incx);
    } else {
      *tau = (beta - alphr) / beta;
      Scal(n - 1, one / (alphr - beta), x, incx);
    }

    // If ALPHA is subnormal, it may lose relative accuracy.
    for (blasint j = 1; j <= knt; ++j) beta = beta * safmin;
    *alpha = T(beta);
  }
}

// xLARF: applies an elementary reflector H = I - tau * v * v**H to the
// M-by-N matrix C, from the left (SIDE = 'L': H * C) or from the right
// (SIDE = 'R': C * H). To apply H**H, supply conj(tau) instead. If tau = 0,
// H is the unit matrix. WORK has N (SIDE = 'L') or M (SIDE = 'R') elements.
//
// Trailing zeros of v, and the rows (columns) of C that they would multiply,
// are skipped.
template <typename T>
void Larf(char side, blasint m, blasint n, const T* v, blasint incv, T tau,
          T* c, blasint ldc, T* work) {
  const T one = T(1), zero = T(0);
  const Vector<const T> V(v);

  const bool applyleft = Lsame(side, 'L');
  blasint lastv = 0;
  blasint lastc = 0;
  if (tau != zero) {
    // Set up variables for scanning V. LASTV begins pointing to the end of
    // V.
    lastv = applyleft ? m : n;
    blasint i = (incv > 0) ? 1 + (lastv - 1) * incv : 1;
    // Look for the last non-zero row in V.
    while (lastv > 0 && V(i) == zero) {
      lastv = lastv - 1;
      i = i - incv;
    }
    if (applyleft) {
      // Scan for the last non-zero column in C(1:lastv,:).
      lastc = Ilalc(lastv, n, c, ldc);
    } else {
      // Scan for the last non-zero row in C(:,1:lastv).
      lastc = Ilalr(m, lastv, c, ldc);
    }
  }
  // Note that lastc == 0 renders the BLAS operations null; no special case
  // is needed at this level.
  if (applyleft) {
    // Form H * C.
    if (lastv > 0) {
      // w(1:lastc,1) := C(1:lastv,1:lastc)**H * v(1:lastv,1)
      Gemv('C', lastv, lastc, one, c, ldc, v, incv, zero, work, 1);
      // C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**H
      Gerc(lastv, lastc, -tau, v, incv, work, 1, c, ldc);
    }
  } else {
    // Form C * H.
    if (lastv > 0) {
      // w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
      Gemv('N', lastc, lastv, one, c, ldc, v, incv, zero, work, 1);
      // C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**H
      Gerc(lastc, lastv, -tau, work, 1, v, incv, c, ldc);
    }
  }
}

// xLARF1F: like xLARF, for a vector v whose first element is 1 and not
// referenced (as in the reflectors that xGEQRF stores below the diagonal).
//
// The real routine applies the first element's contribution with xAXPY and
// the complex one with loops; the template uses the loops for both, which
// does the same arithmetic.
template <typename T>
void Larf1f(char side, blasint m, blasint n, const T* v, blasint incv, T tau,
            T* c, blasint ldc, T* work) {
  const T one = T(1), zero = T(0);
  const Vector<const T> V(v);
  const Matrix<T> C(c, ldc);
  const Vector<T> WORK(work);

  const bool applyleft = Lsame(side, 'L');
  blasint lastv = 1;
  blasint lastc = 0;
  if (tau != zero) {
    // Set up variables for scanning V. LASTV begins pointing to the end of
    // V.
    lastv = applyleft ? m : n;
    blasint i = (incv > 0) ? 1 + (lastv - 1) * incv : 1;
    // Look for the last non-zero row in V. Since we are assuming that
    // V(1) = 1, and it is not stored, so we shouldn't access it.
    while (lastv > 1 && V(i) == zero) {
      lastv = lastv - 1;
      i = i - incv;
    }
    if (applyleft) {
      // Scan for the last non-zero column in C(1:lastv,:).
      lastc = Ilalc(lastv, n, c, ldc);
    } else {
      // Scan for the last non-zero row in C(:,1:lastv).
      lastc = Ilalr(m, lastv, c, ldc);
    }
  }
  if (lastc == 0) return;
  if (applyleft) {
    // Check if lastv = 1. This means v = 1, so we just need to compute
    // C := H * C = (1 - tau) * C.
    if (lastv == 1) {
      Scal(lastc, one - tau, c, ldc);
    } else {
      // w(1:lastc,1) := C(2:lastv,1:lastc)**H * v(2:lastv,1)
      Gemv('C', lastv - 1, lastc, one, C.ptr(1 + 1, 1), ldc, V.ptr(1 + incv),
           incv, zero, work, 1);
      // w(1:lastc,1) += C(1,1:lastc)**H * v(1,1) = C(1,1:lastc)**H
      for (blasint i = 1; i <= lastc; ++i) WORK(i) = WORK(i) + Conj(C(1, i));
      // C(1,1:lastc) := C(...) - tau * v(1,1) * w(1:lastc,1)**H
      //               = C(...) - tau * w(1:lastc,1)**H
      for (blasint i = 1; i <= lastc; ++i) {
        C(1, i) = C(1, i) - tau * Conj(WORK(i));
      }
      // C(2:lastv,1:lastc) := C(...) - tau * v(2:lastv,1) * w(1:lastc,1)**H
      Gerc(lastv - 1, lastc, -tau, V.ptr(1 + incv), incv, work, 1,
           C.ptr(1 + 1, 1), ldc);
    }
  } else {
    // Check if n = 1. This means v = 1, so we just need to compute
    // C := C * H = C * (1 - tau).
    if (lastv == 1) {
      Scal(lastc, one - tau, c, 1);
    } else {
      // w(1:lastc,1) := C(1:lastc,2:lastv) * v(2:lastv,1)
      Gemv('N', lastc, lastv - 1, one, C.ptr(1, 1 + 1), ldc, V.ptr(1 + incv),
           incv, zero, work, 1);
      // w(1:lastc,1) += C(1:lastc,1) * v(1,1) = C(1:lastc,1)
      Axpy(lastc, one, c, 1, work, 1);
      // C(1:lastc,1) := C(...) - tau * w(1:lastc,1) * v(1,1)**H
      //               = C(...) - tau * w(1:lastc,1)
      Axpy(lastc, -tau, work, 1, c, 1);
      // C(1:lastc,2:lastv) := C(...) - tau * w(1:lastc,1) * v(2:lastv)**H
      Gerc(lastc, lastv - 1, -tau, work, 1, V.ptr(1 + incv), incv,
           C.ptr(1, 1 + 1), ldc);
    }
  }
}

// xLARF1L: like xLARF, for a vector v whose last element is 1 and not
// referenced (as in the reflectors that xGEQLF stores above the diagonal).
// Leading zeros of v are skipped.
//
// Deviation: if v has only zeros before its last element (or N = 1), the
// real routine applied from the right scales column 1 of C instead of
// column N (as of LAPACK 3.12.1); the template does what the complex
// routine does, which is correct.
template <typename T>
void Larf1l(char side, blasint m, blasint n, const T* v, blasint incv, T tau,
            T* c, blasint ldc, T* work) {
  const T one = T(1), zero = T(0);
  const Vector<const T> V(v);
  const Matrix<T> C(c, ldc);
  const Vector<T> WORK(work);

  const bool applyleft = Lsame(side, 'L');
  blasint firstv = 1;
  blasint lastv = 0;
  blasint lastc = 0;
  blasint i = 1;
  if (tau != zero) {
    // Set up variables for scanning V. LASTV begins pointing to the end of
    // V up to V(1).
    lastv = applyleft ? m : n;
    // Look for the first non-zero row in V.
    while (lastv > firstv && V(i) == zero) {
      firstv = firstv + 1;
      i = i + incv;
    }
    if (applyleft) {
      // Scan for the last non-zero column in C(1:lastv,:).
      lastc = Ilalc(lastv, n, c, ldc);
    } else {
      // Scan for the last non-zero row in C(:,1:lastv).
      lastc = Ilalr(m, lastv, c, ldc);
    }
  }
  if (lastc == 0) return;
  if (applyleft) {
    if (lastv == firstv) {
      // v = e(lastv): C(lastv,:) := (1 - tau) * C(lastv,:).
      Scal(lastc, one - tau, C.ptr(lastv, 1), ldc);
    } else {
      // w(1:lastc,1) := C(firstv:lastv-1,1:lastc)**H * v(firstv:lastv-1,1)
      Gemv('C', lastv - firstv, lastc, one, C.ptr(firstv, 1), ldc, V.ptr(i),
           incv, zero, work, 1);
      // w(1:lastc,1) += C(lastv,1:lastc)**H * v(lastv,1)
      for (blasint j = 1; j <= lastc; ++j) {
        WORK(j) = WORK(j) + Conj(C(lastv, j));
      }
      // C(lastv,1:lastc) := C(...) - tau * v(lastv,1) * w(1:lastc,1)**H
      for (blasint j = 1; j <= lastc; ++j) {
        C(lastv, j) = C(lastv, j) - tau * Conj(WORK(j));
      }
      // C(firstv:lastv-1,1:lastc) := C(...) - tau * v(...) * w(1:lastc,1)**H
      Gerc(lastv - firstv, lastc, -tau, V.ptr(i), incv, work, 1,
           C.ptr(firstv, 1), ldc);
    }
  } else {
    if (lastv == firstv) {
      // v = e(lastv): C(:,lastv) := (1 - tau) * C(:,lastv).
      Scal(lastc, one - tau, C.ptr(1, lastv), 1);
    } else {
      // w(1:lastc,1) := C(1:lastc,firstv:lastv-1) * v(firstv:lastv-1,1)
      Gemv('N', lastc, lastv - firstv, one, C.ptr(1, firstv), ldc, V.ptr(i),
           incv, zero, work, 1);
      // w(1:lastc,1) += C(1:lastc,lastv) * v(lastv,1) = C(1:lastc,lastv)
      Axpy(lastc, one, C.ptr(1, lastv), 1, work, 1);
      // C(1:lastc,lastv) := C(...) - tau * w(1:lastc,1) * v(lastv,1)**H
      //                   = C(...) - tau * w(1:lastc,1)
      Axpy(lastc, -tau, work, 1, C.ptr(1, lastv), 1);
      // C(1:lastc,firstv:lastv-1) := C(...) - tau * w(1:lastc,1) * v(...)**H
      Gerc(lastc, lastv - firstv, -tau, work, 1, V.ptr(i), incv,
           C.ptr(1, firstv), ldc);
    }
  }
}

// xLARFT: forms the triangular factor T of a block reflector H of order N,
// which is defined as a product of K elementary reflectors:
//
//   DIRECT = 'F': H = H(1) H(2) . . . H(k) and T is upper triangular;
//   DIRECT = 'B': H = H(k) . . . H(2) H(1) and T is lower triangular.
//
// The vector which defines the elementary reflector H(i) is stored in the
// i-th column of the array V (STOREV = 'C'), and H = I - V * T * V**H; or in
// the i-th row of V (STOREV = 'R'), and H = I - V**H * T * V. The unit
// elements of the vectors (V(i, i) for DIRECT = 'F') are not referenced,
// nor are the elements on the other side of them (the triangle of V that
// is zero).
//
// This is the recursive version of LAPACK 3.12.1: it forms the factors of
// the first and the last half of the reflectors and combines them with
// xTRMM and xGEMM. (T is the factor array, not the type; the template
// parameter is still T, as everywhere else.)
template <typename T>
void Larft(char direct, char storev, blasint n, blasint k, const T* v,
           blasint ldv, const T* tau, T* t, blasint ldt) {
  const T one = T(1), neg_one = T(-1);
  const Matrix<const T> V(v, ldv);
  const Vector<const T> TAU(tau);
  const Matrix<T> Tm(t, ldt);  // T(LDT, *) in Fortran.

  // Quick return if possible.
  if (n == 0 || k == 0) return;
  // Base case.
  if (n == 1 || k == 1) {
    Tm(1, 1) = TAU(1);
    return;
  }

  // Beginning of executable statements.
  const blasint l = k / 2;
  // Determine what kind of Q we need to compute. We assume that if the user
  // doesn't provide 'F' for DIRECT, then they meant to provide 'B', and if
  // they don't provide 'C' for STOREV, then they meant to provide 'R'.
  const bool dirf = Lsame(direct, 'F');
  const bool colv = Lsame(storev, 'C');
  // QR happens when we have forward direction in column storage.
  const bool qr = dirf && colv;
  // LQ happens when we have forward direction in row storage.
  const bool lq = dirf && !colv;
  // QL happens when we have backward direction in column storage.
  const bool ql = !dirf && colv;
  // The last case is RQ. Due to how we structured this, if the above 3 are
  // false, then RQ must be true, so we never store this. RQ happens when we
  // have backward direction in row storage.
  if (qr) {
    // Break V apart into 6 components:
    //
    // V = |---------------|
    //     |V_{1,1} 0      |
    //     |V_{2,1} V_{2,2}|
    //     |V_{3,1} V_{3,2}|
    //     |---------------|
    //
    // V_{1,1}\in\C^{l,l}      unit lower triangular
    // V_{2,1}\in\C^{k-l,l}    rectangular
    // V_{3,1}\in\C^{n-k,l}    rectangular
    //
    // V_{2,2}\in\C^{k-l,k-l}  unit lower triangular
    // V_{3,2}\in\C^{n-k,k-l}  rectangular
    //
    // We will construct the T matrix
    // T = |---------------|
    //     |T_{1,1} T_{1,2}|
    //     |0       T_{2,2}|
    //     |---------------|
    //
    // T is the triangular factor obtained from block reflectors. To
    // motivate the structure, assume we have already computed T_{1,1} and
    // T_{2,2}. Then collect the associated reflectors in V_1 and V_2.
    //
    // T_{1,1}\in\C^{l, l}     upper triangular
    // T_{2,2}\in\C^{k-l, k-l} upper triangular
    // T_{1,2}\in\C^{l, k-l}   rectangular
    //
    // Where l = floor(k/2).
    //
    // Then, consider the product:
    //
    // (I - V_1*T_{1,1}*V_1')*(I - V_2*T_{2,2}*V_2')
    // = I - V_1*T_{1,1}*V_1' - V_2*T_{2,2}*V_2'
    //     + V_1*T_{1,1}*V_1'*V_2*T_{2,2}*V_2'
    //
    // Define T_{1,2} = -T_{1,1}*V_1'*V_2*T_{2,2}.
    //
    // Then, we can define the matrix V as
    // V = |-------|
    //     |V_1 V_2|
    //     |-------|
    //
    // So, our product is equivalent to the matrix product
    // I - V*T*V'
    // This means, we can compute T_{1,1} and T_{2,2}, then use this
    // information to compute T_{1,2}.

    // Compute T_{1,1} recursively.
    Larft(direct, storev, n, l, v, ldv, tau, t, ldt);
    // Compute T_{2,2} recursively.
    Larft(direct, storev, n - l, k - l, V.ptr(l + 1, l + 1), ldv,
          TAU.ptr(l + 1), Tm.ptr(l + 1, l + 1), ldt);

    // Compute T_{1,2}.
    // T_{1,2} = V_{2,1}'
    for (blasint j = 1; j <= l; ++j) {
      for (blasint i = 1; i <= k - l; ++i) {
        Tm(j, l + i) = Conj(V(l + i, j));
      }
    }
    // T_{1,2} = T_{1,2}*V_{2,2}
    Trmm('R', 'L', 'N', 'U', l, k - l, one, V.ptr(l + 1, l + 1), ldv,
         Tm.ptr(1, l + 1), ldt);
    // T_{1,2} = V_{3,1}'*V_{3,2} + T_{1,2}
    // Note: We assume K <= N, and GEMM will do nothing if N=K.
    Gemm('C', 'N', l, k - l, n - k, one, V.ptr(k + 1, 1), ldv,
         V.ptr(k + 1, l + 1), ldv, one, Tm.ptr(1, l + 1), ldt);
    // At this point, we have that T_{1,2} = V_1'*V_2. All that is left is
    // to pre and post multiply by -T_{1,1} and T_{2,2} respectively.
    // T_{1,2} = -T_{1,1}*T_{1,2}
    Trmm('L', 'U', 'N', 'N', l, k - l, neg_one, t, ldt, Tm.ptr(1, l + 1), ldt);
    // T_{1,2} = T_{1,2}*T_{2,2}
    Trmm('R', 'U', 'N', 'N', l, k - l, one, Tm.ptr(l + 1, l + 1), ldt,
         Tm.ptr(1, l + 1), ldt);
  } else if (lq) {
    // Break V apart into 6 components:
    //
    // V = |----------------------|
    //     |V_{1,1} V_{1,2} V{1,3}|
    //     |0       V_{2,2} V{2,3}|
    //     |----------------------|
    //
    // V_{1,1}\in\C^{l,l}      unit upper triangular
    // V_{1,2}\in\C^{l,k-l}    rectangular
    // V_{1,3}\in\C^{l,n-k}    rectangular
    //
    // V_{2,2}\in\C^{k-l,k-l}  unit upper triangular
    // V_{2,3}\in\C^{k-l,n-k}  rectangular
    //
    // Where l = floor(k/2).
    //
    // We will construct the T matrix
    // T = |---------------|
    //     |T_{1,1} T_{1,2}|
    //     |0       T_{2,2}|
    //     |---------------|
    //
    // T is the triangular factor obtained from block reflectors. To
    // motivate the structure, assume we have already computed T_{1,1} and
    // T_{2,2}. Then collect the associated reflectors in V_1 and V_2.
    //
    // T_{1,1}\in\C^{l, l}     upper triangular
    // T_{2,2}\in\C^{k-l, k-l} upper triangular
    // T_{1,2}\in\C^{l, k-l}   rectangular
    //
    // Then, consider the product:
    //
    // (I - V_1'*T_{1,1}*V_1)*(I - V_2'*T_{2,2}*V_2)
    // = I - V_1'*T_{1,1}*V_1 - V_2'*T_{2,2}*V_2
    //     + V_1'*T_{1,1}*V_1*V_2'*T_{2,2}*V_2
    //
    // Define T_{1,2} = -T_{1,1}*V_1*V_2'*T_{2,2}.
    //
    // Then, we can define the matrix V as
    // V = |---|
    //     |V_1|
    //     |V_2|
    //     |---|
    //
    // So, our product is equivalent to the matrix product
    // I - V'*T*V
    // This means, we can compute T_{1,1} and T_{2,2}, then use this
    // information to compute T_{1,2}.

    // Compute T_{1,1} recursively.
    Larft(direct, storev, n, l, v, ldv, tau, t, ldt);
    // Compute T_{2,2} recursively.
    Larft(direct, storev, n - l, k - l, V.ptr(l + 1, l + 1), ldv,
          TAU.ptr(l + 1), Tm.ptr(l + 1, l + 1), ldt);

    // Compute T_{1,2}.
    // T_{1,2} = V_{1,2}
    Lacpy('A', l, k - l, V.ptr(1, l + 1), ldv, Tm.ptr(1, l + 1), ldt);
    // T_{1,2} = T_{1,2}*V_{2,2}'
    Trmm('R', 'U', 'C', 'U', l, k - l, one, V.ptr(l + 1, l + 1), ldv,
         Tm.ptr(1, l + 1), ldt);
    // T_{1,2} = V_{1,3}*V_{2,3}' + T_{1,2}
    // Note: We assume K <= N, and GEMM will do nothing if N=K.
    Gemm('N', 'C', l, k - l, n - k, one, V.ptr(1, k + 1), ldv,
         V.ptr(l + 1, k + 1), ldv, one, Tm.ptr(1, l + 1), ldt);
    // At this point, we have that T_{1,2} = V_1*V_2'. All that is left is
    // to pre and post multiply by -T_{1,1} and T_{2,2} respectively.
    // T_{1,2} = -T_{1,1}*T_{1,2}
    Trmm('L', 'U', 'N', 'N', l, k - l, neg_one, t, ldt, Tm.ptr(1, l + 1), ldt);
    // T_{1,2} = T_{1,2}*T_{2,2}
    Trmm('R', 'U', 'N', 'N', l, k - l, one, Tm.ptr(l + 1, l + 1), ldt,
         Tm.ptr(1, l + 1), ldt);
  } else if (ql) {
    // Break V apart into 6 components:
    //
    // V = |---------------|
    //     |V_{1,1} V_{1,2}|
    //     |V_{2,1} V_{2,2}|
    //     |0       V_{3,2}|
    //     |---------------|
    //
    // V_{1,1}\in\C^{n-k,k-l}  rectangular
    // V_{2,1}\in\C^{k-l,k-l}  unit upper triangular
    //
    // V_{1,2}\in\C^{n-k,l}    rectangular
    // V_{2,2}\in\C^{k-l,l}    rectangular
    // V_{3,2}\in\C^{l,l}      unit upper triangular
    //
    // We will construct the T matrix
    // T = |---------------|
    //     |T_{1,1} 0      |
    //     |T_{2,1} T_{2,2}|
    //     |---------------|
    //
    // T is the triangular factor obtained from block reflectors. To
    // motivate the structure, assume we have already computed T_{1,1} and
    // T_{2,2}. Then collect the associated reflectors in V_1 and V_2.
    //
    // T_{1,1}\in\C^{k-l, k-l} non-unit lower triangular
    // T_{2,2}\in\C^{l, l}     non-unit lower triangular
    // T_{2,1}\in\C^{k-l, l}   rectangular
    //
    // Where l = floor(k/2).
    //
    // Then, consider the product:
    //
    // (I - V_2*T_{2,2}*V_2')*(I - V_1*T_{1,1}*V_1')
    // = I - V_2*T_{2,2}*V_2' - V_1*T_{1,1}*V_1'
    //     + V_2*T_{2,2}*V_2'*V_1*T_{1,1}*V_1'
    //
    // Define T_{2,1} = -T_{2,2}*V_2'*V_1*T_{1,1}.
    //
    // Then, we can define the matrix V as
    // V = |-------|
    //     |V_1 V_2|
    //     |-------|
    //
    // So, our product is equivalent to the matrix product
    // I - V*T*V'
    // This means, we can compute T_{1,1} and T_{2,2}, then use this
    // information to compute T_{2,1}.

    // Compute T_{1,1} recursively.
    Larft(direct, storev, n - l, k - l, v, ldv, tau, t, ldt);
    // Compute T_{2,2} recursively.
    Larft(direct, storev, n, l, V.ptr(1, k - l + 1), ldv, TAU.ptr(k - l + 1),
          Tm.ptr(k - l + 1, k - l + 1), ldt);

    // Compute T_{2,1}.
    // T_{2,1} = V_{2,2}'
    for (blasint j = 1; j <= k - l; ++j) {
      for (blasint i = 1; i <= l; ++i) {
        Tm(k - l + i, j) = Conj(V(n - k + j, k - l + i));
      }
    }
    // T_{2,1} = T_{2,1}*V_{2,1}
    Trmm('R', 'U', 'N', 'U', l, k - l, one, V.ptr(n - k + 1, 1), ldv,
         Tm.ptr(k - l + 1, 1), ldt);
    // T_{2,1} = V_{2,2}'*V_{2,1} + T_{2,1}
    // Note: We assume K <= N, and GEMM will do nothing if N=K.
    Gemm('C', 'N', l, k - l, n - k, one, V.ptr(1, k - l + 1), ldv, v, ldv, one,
         Tm.ptr(k - l + 1, 1), ldt);
    // At this point, we have that T_{2,1} = V_2'*V_1. All that is left is
    // to pre and post multiply by -T_{2,2} and T_{1,1} respectively.
    // T_{2,1} = -T_{2,2}*T_{2,1}
    Trmm('L', 'L', 'N', 'N', l, k - l, neg_one, Tm.ptr(k - l + 1, k - l + 1),
         ldt, Tm.ptr(k - l + 1, 1), ldt);
    // T_{2,1} = T_{2,1}*T_{1,1}
    Trmm('R', 'L', 'N', 'N', l, k - l, one, t, ldt, Tm.ptr(k - l + 1, 1), ldt);
  } else {
    // Else means RQ case.
    //
    // Break V apart into 6 components:
    //
    // V = |-----------------------|
    //     |V_{1,1} V_{1,2} 0      |
    //     |V_{2,1} V_{2,2} V_{2,3}|
    //     |-----------------------|
    //
    // V_{1,1}\in\C^{k-l,n-k}  rectangular
    // V_{1,2}\in\C^{k-l,k-l}  unit lower triangular
    //
    // V_{2,1}\in\C^{l,n-k}    rectangular
    // V_{2,2}\in\C^{l,k-l}    rectangular
    // V_{2,3}\in\C^{l,l}      unit lower triangular
    //
    // We will construct the T matrix
    // T = |---------------|
    //     |T_{1,1} 0      |
    //     |T_{2,1} T_{2,2}|
    //     |---------------|
    //
    // T is the triangular factor obtained from block reflectors. To
    // motivate the structure, assume we have already computed T_{1,1} and
    // T_{2,2}. Then collect the associated reflectors in V_1 and V_2.
    //
    // T_{1,1}\in\C^{k-l, k-l} non-unit lower triangular
    // T_{2,2}\in\C^{l, l}     non-unit lower triangular
    // T_{2,1}\in\C^{k-l, l}   rectangular
    //
    // Where l = floor(k/2).
    //
    // Then, consider the product:
    //
    // (I - V_2'*T_{2,2}*V_2)*(I - V_1'*T_{1,1}*V_1)
    // = I - V_2'*T_{2,2}*V_2 - V_1'*T_{1,1}*V_1
    //     + V_2'*T_{2,2}*V_2*V_1'*T_{1,1}*V_1
    //
    // Define T_{2,1} = -T_{2,2}*V_2*V_1'*T_{1,1}.
    //
    // Then, we can define the matrix V as
    // V = |---|
    //     |V_1|
    //     |V_2|
    //     |---|
    //
    // So, our product is equivalent to the matrix product
    // I - V'*T*V
    // This means, we can compute T_{1,1} and T_{2,2}, then use this
    // information to compute T_{2,1}.

    // Compute T_{1,1} recursively.
    Larft(direct, storev, n - l, k - l, v, ldv, tau, t, ldt);
    // Compute T_{2,2} recursively.
    Larft(direct, storev, n, l, V.ptr(k - l + 1, 1), ldv, TAU.ptr(k - l + 1),
          Tm.ptr(k - l + 1, k - l + 1), ldt);

    // Compute T_{2,1}.
    // T_{2,1} = V_{2,2}
    Lacpy('A', l, k - l, V.ptr(k - l + 1, n - k + 1), ldv, Tm.ptr(k - l + 1, 1),
          ldt);
    // T_{2,1} = T_{2,1}*V_{1,2}'
    Trmm('R', 'L', 'C', 'U', l, k - l, one, V.ptr(1, n - k + 1), ldv,
         Tm.ptr(k - l + 1, 1), ldt);
    // T_{2,1} = V_{2,1}*V_{1,1}' + T_{2,1}
    // Note: We assume K <= N, and GEMM will do nothing if N=K.
    Gemm('N', 'C', l, k - l, n - k, one, V.ptr(k - l + 1, 1), ldv, v, ldv, one,
         Tm.ptr(k - l + 1, 1), ldt);
    // At this point, we have that T_{2,1} = V_2*V_1'. All that is left is
    // to pre and post multiply by -T_{2,2} and T_{1,1} respectively.
    // T_{2,1} = -T_{2,2}*T_{2,1}
    Trmm('L', 'L', 'N', 'N', l, k - l, neg_one, Tm.ptr(k - l + 1, k - l + 1),
         ldt, Tm.ptr(k - l + 1, 1), ldt);
    // T_{2,1} = T_{2,1}*T_{1,1}
    Trmm('R', 'L', 'N', 'N', l, k - l, one, t, ldt, Tm.ptr(k - l + 1, 1), ldt);
  }
}

// xLARFB: applies a block reflector H or its conjugate transpose H**H to
// the M-by-N matrix C, from the left (SIDE = 'L') or from the right
// (SIDE = 'R'): TRANS = 'N' applies H, TRANS = 'C' (or 'T' for real T)
// applies H**H. H = I - V * T * V**H (STOREV = 'C') or I - V**H * T * V
// (STOREV = 'R') is the product of K reflectors in the order DIRECT (see
// xLARFT, which forms T). WORK is LDWORK-by-K, LDWORK >= N (SIDE = 'L') or
// >= M (SIDE = 'R').
template <typename T>
void Larfb(char side, char trans, char direct, char storev, blasint m,
           blasint n, blasint k, const T* v, blasint ldv, const T* t,
           blasint ldt, T* c, blasint ldc, T* work, blasint ldwork) {
  const T one = T(1);
  const Matrix<const T> V(v, ldv);
  const Matrix<T> C(c, ldc);
  const Matrix<T> WORK(work, ldwork);

  // Quick return if possible.
  if (m <= 0 || n <= 0) return;

  const char transt = Lsame(trans, 'N') ? 'C' : 'N';

  if (Lsame(storev, 'C')) {
    if (Lsame(direct, 'F')) {
      // Let V = ( V1 )    (first K rows)
      //         ( V2 )
      // where V1 is unit lower triangular.
      if (Lsame(side, 'L')) {
        // Form H * C or H**H * C where C = ( C1 )
        //                                  ( C2 )
        //
        // W := C**H * V = (C1**H * V1 + C2**H * V2) (stored in WORK)
        //
        // W := C1**H
        for (blasint j = 1; j <= k; ++j) {
          Copy(n, C.ptr(j, 1), ldc, WORK.ptr(1, j), 1);
          Lacgv(n, WORK.ptr(1, j), 1);
        }
        // W := W * V1
        Trmm('R', 'L', 'N', 'U', n, k, one, v, ldv, work, ldwork);
        if (m > k) {
          // W := W + C2**H * V2
          Gemm('C', 'N', n, k, m - k, one, C.ptr(k + 1, 1), ldc,
               V.ptr(k + 1, 1), ldv, one, work, ldwork);
        }
        // W := W * T**H or W * T
        Trmm('R', 'U', transt, 'N', n, k, one, t, ldt, work, ldwork);
        // C := C - V * W**H
        if (m > k) {
          // C2 := C2 - V2 * W**H
          Gemm('N', 'C', m - k, n, k, -one, V.ptr(k + 1, 1), ldv, work, ldwork,
               one, C.ptr(k + 1, 1), ldc);
        }
        // W := W * V1**H
        Trmm('R', 'L', 'C', 'U', n, k, one, v, ldv, work, ldwork);
        // C1 := C1 - W**H
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= n; ++i) {
            C(j, i) = C(j, i) - Conj(WORK(i, j));
          }
        }
      } else if (Lsame(side, 'R')) {
        // Form C * H or C * H**H where C = ( C1 C2 )
        //
        // W := C * V = (C1*V1 + C2*V2) (stored in WORK)
        //
        // W := C1
        for (blasint j = 1; j <= k; ++j) {
          Copy(m, C.ptr(1, j), 1, WORK.ptr(1, j), 1);
        }
        // W := W * V1
        Trmm('R', 'L', 'N', 'U', m, k, one, v, ldv, work, ldwork);
        if (n > k) {
          // W := W + C2 * V2
          Gemm('N', 'N', m, k, n - k, one, C.ptr(1, k + 1), ldc,
               V.ptr(k + 1, 1), ldv, one, work, ldwork);
        }
        // W := W * T or W * T**H
        Trmm('R', 'U', trans, 'N', m, k, one, t, ldt, work, ldwork);
        // C := C - W * V**H
        if (n > k) {
          // C2 := C2 - W * V2**H
          Gemm('N', 'C', m, n - k, k, -one, work, ldwork, V.ptr(k + 1, 1), ldv,
               one, C.ptr(1, k + 1), ldc);
        }
        // W := W * V1**H
        Trmm('R', 'L', 'C', 'U', m, k, one, v, ldv, work, ldwork);
        // C1 := C1 - W
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= m; ++i) C(i, j) = C(i, j) - WORK(i, j);
        }
      }
    } else {
      // Let V = ( V1 )
      //         ( V2 )    (last K rows)
      // where V2 is unit upper triangular.
      if (Lsame(side, 'L')) {
        // Form H * C or H**H * C where C = ( C1 )
        //                                  ( C2 )
        //
        // W := C**H * V = (C1**H * V1 + C2**H * V2) (stored in WORK)
        //
        // W := C2**H
        for (blasint j = 1; j <= k; ++j) {
          Copy(n, C.ptr(m - k + j, 1), ldc, WORK.ptr(1, j), 1);
          Lacgv(n, WORK.ptr(1, j), 1);
        }
        // W := W * V2
        Trmm('R', 'U', 'N', 'U', n, k, one, V.ptr(m - k + 1, 1), ldv, work,
             ldwork);
        if (m > k) {
          // W := W + C1**H * V1
          Gemm('C', 'N', n, k, m - k, one, c, ldc, v, ldv, one, work, ldwork);
        }
        // W := W * T**H or W * T
        Trmm('R', 'L', transt, 'N', n, k, one, t, ldt, work, ldwork);
        // C := C - V * W**H
        if (m > k) {
          // C1 := C1 - V1 * W**H
          Gemm('N', 'C', m - k, n, k, -one, v, ldv, work, ldwork, one, c, ldc);
        }
        // W := W * V2**H
        Trmm('R', 'U', 'C', 'U', n, k, one, V.ptr(m - k + 1, 1), ldv, work,
             ldwork);
        // C2 := C2 - W**H
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= n; ++i) {
            C(m - k + j, i) = C(m - k + j, i) - Conj(WORK(i, j));
          }
        }
      } else if (Lsame(side, 'R')) {
        // Form C * H or C * H**H where C = ( C1 C2 )
        //
        // W := C * V = (C1*V1 + C2*V2) (stored in WORK)
        //
        // W := C2
        for (blasint j = 1; j <= k; ++j) {
          Copy(m, C.ptr(1, n - k + j), 1, WORK.ptr(1, j), 1);
        }
        // W := W * V2
        Trmm('R', 'U', 'N', 'U', m, k, one, V.ptr(n - k + 1, 1), ldv, work,
             ldwork);
        if (n > k) {
          // W := W + C1 * V1
          Gemm('N', 'N', m, k, n - k, one, c, ldc, v, ldv, one, work, ldwork);
        }
        // W := W * T or W * T**H
        Trmm('R', 'L', trans, 'N', m, k, one, t, ldt, work, ldwork);
        // C := C - W * V**H
        if (n > k) {
          // C1 := C1 - W * V1**H
          Gemm('N', 'C', m, n - k, k, -one, work, ldwork, v, ldv, one, c, ldc);
        }
        // W := W * V2**H
        Trmm('R', 'U', 'C', 'U', m, k, one, V.ptr(n - k + 1, 1), ldv, work,
             ldwork);
        // C2 := C2 - W
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= m; ++i) {
            C(i, n - k + j) = C(i, n - k + j) - WORK(i, j);
          }
        }
      }
    }
  } else if (Lsame(storev, 'R')) {
    if (Lsame(direct, 'F')) {
      // Let V = ( V1  V2 )    (V1: first K columns)
      // where V1 is unit upper triangular.
      if (Lsame(side, 'L')) {
        // Form H * C or H**H * C where C = ( C1 )
        //                                  ( C2 )
        //
        // W := C**H * V**H = (C1**H * V1**H + C2**H * V2**H) (stored in
        // WORK)
        //
        // W := C1**H
        for (blasint j = 1; j <= k; ++j) {
          Copy(n, C.ptr(j, 1), ldc, WORK.ptr(1, j), 1);
          Lacgv(n, WORK.ptr(1, j), 1);
        }
        // W := W * V1**H
        Trmm('R', 'U', 'C', 'U', n, k, one, v, ldv, work, ldwork);
        if (m > k) {
          // W := W + C2**H * V2**H
          Gemm('C', 'C', n, k, m - k, one, C.ptr(k + 1, 1), ldc,
               V.ptr(1, k + 1), ldv, one, work, ldwork);
        }
        // W := W * T**H or W * T
        Trmm('R', 'U', transt, 'N', n, k, one, t, ldt, work, ldwork);
        // C := C - V**H * W**H
        if (m > k) {
          // C2 := C2 - V2**H * W**H
          Gemm('C', 'C', m - k, n, k, -one, V.ptr(1, k + 1), ldv, work, ldwork,
               one, C.ptr(k + 1, 1), ldc);
        }
        // W := W * V1
        Trmm('R', 'U', 'N', 'U', n, k, one, v, ldv, work, ldwork);
        // C1 := C1 - W**H
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= n; ++i) {
            C(j, i) = C(j, i) - Conj(WORK(i, j));
          }
        }
      } else if (Lsame(side, 'R')) {
        // Form C * H or C * H**H where C = ( C1 C2 )
        //
        // W := C * V**H = (C1*V1**H + C2*V2**H) (stored in WORK)
        //
        // W := C1
        for (blasint j = 1; j <= k; ++j) {
          Copy(m, C.ptr(1, j), 1, WORK.ptr(1, j), 1);
        }
        // W := W * V1**H
        Trmm('R', 'U', 'C', 'U', m, k, one, v, ldv, work, ldwork);
        if (n > k) {
          // W := W + C2 * V2**H
          Gemm('N', 'C', m, k, n - k, one, C.ptr(1, k + 1), ldc,
               V.ptr(1, k + 1), ldv, one, work, ldwork);
        }
        // W := W * T or W * T**H
        Trmm('R', 'U', trans, 'N', m, k, one, t, ldt, work, ldwork);
        // C := C - W * V
        if (n > k) {
          // C2 := C2 - W * V2
          Gemm('N', 'N', m, n - k, k, -one, work, ldwork, V.ptr(1, k + 1), ldv,
               one, C.ptr(1, k + 1), ldc);
        }
        // W := W * V1
        Trmm('R', 'U', 'N', 'U', m, k, one, v, ldv, work, ldwork);
        // C1 := C1 - W
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= m; ++i) C(i, j) = C(i, j) - WORK(i, j);
        }
      }
    } else {
      // Let V = ( V1  V2 )    (V2: last K columns)
      // where V2 is unit lower triangular.
      if (Lsame(side, 'L')) {
        // Form H * C or H**H * C where C = ( C1 )
        //                                  ( C2 )
        //
        // W := C**H * V**H = (C1**H * V1**H + C2**H * V2**H) (stored in
        // WORK)
        //
        // W := C2**H
        for (blasint j = 1; j <= k; ++j) {
          Copy(n, C.ptr(m - k + j, 1), ldc, WORK.ptr(1, j), 1);
          Lacgv(n, WORK.ptr(1, j), 1);
        }
        // W := W * V2**H
        Trmm('R', 'L', 'C', 'U', n, k, one, V.ptr(1, m - k + 1), ldv, work,
             ldwork);
        if (m > k) {
          // W := W + C1**H * V1**H
          Gemm('C', 'C', n, k, m - k, one, c, ldc, v, ldv, one, work, ldwork);
        }
        // W := W * T**H or W * T
        Trmm('R', 'L', transt, 'N', n, k, one, t, ldt, work, ldwork);
        // C := C - V**H * W**H
        if (m > k) {
          // C1 := C1 - V1**H * W**H
          Gemm('C', 'C', m - k, n, k, -one, v, ldv, work, ldwork, one, c, ldc);
        }
        // W := W * V2
        Trmm('R', 'L', 'N', 'U', n, k, one, V.ptr(1, m - k + 1), ldv, work,
             ldwork);
        // C2 := C2 - W**H
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= n; ++i) {
            C(m - k + j, i) = C(m - k + j, i) - Conj(WORK(i, j));
          }
        }
      } else if (Lsame(side, 'R')) {
        // Form C * H or C * H**H where C = ( C1 C2 )
        //
        // W := C * V**H = (C1*V1**H + C2*V2**H) (stored in WORK)
        //
        // W := C2
        for (blasint j = 1; j <= k; ++j) {
          Copy(m, C.ptr(1, n - k + j), 1, WORK.ptr(1, j), 1);
        }
        // W := W * V2**H
        Trmm('R', 'L', 'C', 'U', m, k, one, V.ptr(1, n - k + 1), ldv, work,
             ldwork);
        if (n > k) {
          // W := W + C1 * V1**H
          Gemm('N', 'C', m, k, n - k, one, c, ldc, v, ldv, one, work, ldwork);
        }
        // W := W * T or W * T**H
        Trmm('R', 'L', trans, 'N', m, k, one, t, ldt, work, ldwork);
        // C := C - W * V
        if (n > k) {
          // C1 := C1 - W * V1
          Gemm('N', 'N', m, n - k, k, -one, work, ldwork, v, ldv, one, c, ldc);
        }
        // W := W * V2
        Trmm('R', 'L', 'N', 'U', m, k, one, V.ptr(1, n - k + 1), ldv, work,
             ldwork);
        // C1 := C1 - W
        for (blasint j = 1; j <= k; ++j) {
          for (blasint i = 1; i <= m; ++i) {
            C(i, n - k + j) = C(i, n - k + j) - WORK(i, j);
          }
        }
      }
    }
  }
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template void Larfg<T>(blasint, T*, T*, blasint, T*);                     \
  template void Larf<T>(char, blasint, blasint, const T*, blasint, T, T*,   \
                        blasint, T*);                                       \
  template void Larf1f<T>(char, blasint, blasint, const T*, blasint, T, T*, \
                          blasint, T*);                                     \
  template void Larf1l<T>(char, blasint, blasint, const T*, blasint, T, T*, \
                          blasint, T*);                                     \
  template void Larft<T>(char, char, blasint, blasint, const T*, blasint,   \
                         const T*, T*, blasint);                            \
  template void Larfb<T>(char, char, char, char, blasint, blasint, blasint, \
                         const T*, blasint, const T*, blasint, T*, blasint, \
                         T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_HOUSEHOLDER(prefix, T, R)                              \
  HWYBLAS_API void prefix##larfg_(const hwyblas_int* n, R* alpha, R* x,       \
                                  const hwyblas_int* incx, R* tau) {          \
    hwyblas::lapack::Larfg(*n, As<T>(alpha), As<T>(x), *incx, As<T>(tau));    \
  }                                                                           \
  HWYBLAS_API void prefix##larf_(const char* side, const hwyblas_int* m,      \
                                 const hwyblas_int* n, const R* v,            \
                                 const hwyblas_int* incv, const R* tau, R* c, \
                                 const hwyblas_int* ldc, R* work) {           \
    hwyblas::lapack::Larf(*side, *m, *n, As<const T>(v), *incv,               \
                          *As<const T>(tau), As<T>(c), *ldc, As<T>(work));    \
  }                                                                           \
  HWYBLAS_API void prefix##larf1f_(const char* side, const hwyblas_int* m,    \
                                   const hwyblas_int* n, const R* v,          \
                                   const hwyblas_int* incv, const R* tau,     \
                                   R* c, const hwyblas_int* ldc, R* work) {   \
    hwyblas::lapack::Larf1f(*side, *m, *n, As<const T>(v), *incv,             \
                            *As<const T>(tau), As<T>(c), *ldc, As<T>(work));  \
  }                                                                           \
  HWYBLAS_API void prefix##larf1l_(const char* side, const hwyblas_int* m,    \
                                   const hwyblas_int* n, const R* v,          \
                                   const hwyblas_int* incv, const R* tau,     \
                                   R* c, const hwyblas_int* ldc, R* work) {   \
    hwyblas::lapack::Larf1l(*side, *m, *n, As<const T>(v), *incv,             \
                            *As<const T>(tau), As<T>(c), *ldc, As<T>(work));  \
  }                                                                           \
  HWYBLAS_API void prefix##larft_(                                            \
      const char* direct, const char* storev, const hwyblas_int* n,           \
      const hwyblas_int* k, const R* v, const hwyblas_int* ldv, const R* tau, \
      R* t, const hwyblas_int* ldt) {                                         \
    hwyblas::lapack::Larft(*direct, *storev, *n, *k, As<const T>(v), *ldv,    \
                           As<const T>(tau), As<T>(t), *ldt);                 \
  }                                                                           \
  HWYBLAS_API void prefix##larfb_(                                            \
      const char* side, const char* trans, const char* direct,                \
      const char* storev, const hwyblas_int* m, const hwyblas_int* n,         \
      const hwyblas_int* k, const R* v, const hwyblas_int* ldv, const R* t,   \
      const hwyblas_int* ldt, R* c, const hwyblas_int* ldc, R* work,          \
      const hwyblas_int* ldwork) {                                            \
    hwyblas::lapack::Larfb(*side, *trans, *direct, *storev, *m, *n, *k,       \
                           As<const T>(v), *ldv, As<const T>(t), *ldt,        \
                           As<T>(c), *ldc, As<T>(work), *ldwork);             \
  }

HWYBLAS_LAPACK_HOUSEHOLDER(s, float, float)
HWYBLAS_LAPACK_HOUSEHOLDER(d, double, double)
HWYBLAS_LAPACK_HOUSEHOLDER(c, c32, float)
HWYBLAS_LAPACK_HOUSEHOLDER(z, c64, double)

#undef HWYBLAS_LAPACK_HOUSEHOLDER

}  // extern "C"
