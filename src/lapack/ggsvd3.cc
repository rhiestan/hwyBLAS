// The generalized singular value decomposition (GSVD) of a pair (A, B):
// the driver xGGSVD3, the preprocessing xGGSVP3 (reduces (A, B) to upper
// triangular form with rank-revealing QR factorizations), Jacobi-Kogbetliantz
// iteration on the triangular pair, xTGSJA, with xLAGS2 (2-by-2 pairs),
// xLAPLL (the smallest singular value of two vectors) and the column
// permutation xLAPMT.
//
// Each template is the real routine for real T and the complex routine for
// complex T; they differ in conjugations and in TRANS = 'T' (real) or 'C'
// (complex). xLAGS2 is two bodies behind one template (Lags2Real /
// Lags2Complex), where the complex routine first makes its 2-by-2 matrix
// real. xGGSVP3 and xGGSVD3 take RWORK for complex T only; xGGSVD3's view
// RWORK points into WORK for real T, where the real routine sorts ALPHA.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <limits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xLAPLL: the smallest singular value SSMIN of the N-by-2 matrix ( X Y ),
// from a QR factorization (X and Y are overwritten). The real routine
// passes the 2-by-2 triangular factor to xLAS2 with its signs, the complex
// routine their absolute values (xLAS2 uses only their absolute values).
template <typename T>
void Lapll(blasint n, T* x, blasint incx, T* y, blasint incy, RealT<T>* ssmin) {
  using R = RealT<T>;

  const R zero = R(0);
  const T cone = T(1);
  const Vector<T> X(x);
  const Vector<T> Y(y);
  R ssmax;
  T a11, a12, a22, c, tau;
  if (n <= 1) {
    *ssmin = zero;
    return;
  }
  // Compute the QR factorization of the N-by-2 matrix ( X Y )
  Larfg(n, X.ptr(1), X.ptr(1 + incx), incx, &tau);
  a11 = X(1);
  X(1) = cone;
  c = -Conj(tau) * Dotc(n, x, incx, y, incy);
  Axpy(n, c, x, incx, y, incy);
  Larfg(n - 1, Y.ptr(1 + incy), Y.ptr(1 + 2 * incy), incy, &tau);
  a12 = Y(1);
  a22 = Y(1 + incy);
  // Compute the SVD of 2-by-2 Upper triangular matrix.
  if constexpr (IsComplex<T>::value) {
    Las2<R>(std::abs(a11), std::abs(a12), std::abs(a22), ssmin, &ssmax);
  } else {
    Las2<R>(a11, a12, a22, ssmin, &ssmax);
  }
}

// xLAPMT: permutes the columns of the M-by-N matrix X by the permutation
// K(1:N): forward (FORWRD): X(:,K(j)) is moved to X(:,j); backward:
// X(:,j) is moved to X(:,K(j)). K is used as workspace and restored.
template <typename T>
void Lapmt(bool forwrd, blasint m, blasint n, T* x, blasint ldx, blasint* k) {
  const Vector<blasint> K(k);
  const Matrix<T> X(x, ldx);
  blasint i, ii, in, j;
  T temp;
  if (n <= 1) return;
  for (i = 1; i <= n; ++i) {
    K(i) = -K(i);
  }
  if (forwrd) {
    // Forward permutation
    for (i = 1; i <= n; ++i) {
      if (K(i) > 0) goto L40;
      j = i;
      K(j) = -K(j);
      in = K(j);
    L20:
      if (K(in) > 0) goto L40;
      for (ii = 1; ii <= m; ++ii) {
        temp = X(ii, j);
        X(ii, j) = X(ii, in);
        X(ii, in) = temp;
      }
      K(in) = -K(in);
      j = in;
      in = K(in);
      goto L20;
    L40:;
    }
  } else {
    // Backward permutation
    for (i = 1; i <= n; ++i) {
      if (K(i) > 0) goto L80;
      K(i) = -K(i);
      j = K(i);
    L60:
      if (j == i) goto L80;
      for (ii = 1; ii <= m; ++ii) {
        temp = X(ii, i);
        X(ii, i) = X(ii, j);
        X(ii, j) = temp;
      }
      K(j) = -K(j);
      j = K(j);
      goto L60;
    L80:;
    }
  }
}

// xLAGS2 (real T): computes 2-by-2 orthogonal matrices U, V and Q such
// that, if UPPER, U**T*A*Q and V**T*B*Q are lower triangular with
// zero (1,2) elements (A = ( A1 A2 ; 0 A3 ), B likewise, upper
// triangular), or (not UPPER, lower triangular A and B) upper triangular
// with zero (2,1) elements. U = ( CSU SNU ; -SNU CSU ), V and Q likewise.
// NOLINTBEGIN(bugprone-branch-clone): as LAPACK
template <typename T>
void Lags2Real(bool upper, T a1, T a2, T a3, T b1, T b2, T b3, T* csu, T* snu,
               T* csv, T* snv, T* csq, T* snq) {
  using R = T;

  const R zero = R(0);
  R a, aua11, aua12, aua21, aua22, avb11, avb12, avb21, avb22, b, c, csl, csr,
      d, r, s1, s2, snl, snr, ua11, ua11r, ua12, ua21, ua22, ua22r, vb11, vb11r,
      vb12, vb21, vb22, vb22r;
  if (upper) {
    // Input matrices A and B are upper triangular matrices
    // Form matrix C = A*adj(B) = ( a b )
    // ( 0 d )
    a = a1 * b3;
    d = a3 * b1;
    b = a2 * b1 - a1 * b2;
    // The SVD of real 2-by-2 triangular C
    // ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
    // ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
    Lasv2(a, b, d, &s1, &s2, &snr, &csr, &snl, &csl);
    if (std::abs(csl) >= std::abs(snl) || std::abs(csr) >= std::abs(snr)) {
      // Compute the (1,1) and (1,2) elements of U**T *A and V**T *B,
      // and (1,2) element of |U|**T *|A| and |V|**T *|B|.
      ua11r = csl * a1;
      ua12 = csl * a2 + snl * a3;
      vb11r = csr * b1;
      vb12 = csr * b2 + snr * b3;
      aua12 = std::abs(csl) * std::abs(a2) + std::abs(snl) * std::abs(a3);
      avb12 = std::abs(csr) * std::abs(b2) + std::abs(snr) * std::abs(b3);
      // zero (1,2) elements of U**T *A and V**T *B
      if ((std::abs(ua11r) + std::abs(ua12)) != zero) {
        if (aua12 / (std::abs(ua11r) + std::abs(ua12)) <=
            avb12 / (std::abs(vb11r) + std::abs(vb12))) {
          Lartg(-ua11r, ua12, csq, snq, &r);
        } else {
          Lartg(-vb11r, vb12, csq, snq, &r);
        }
      } else {
        Lartg(-vb11r, vb12, csq, snq, &r);
      }
      *csu = csl;
      *snu = -snl;
      *csv = csr;
      *snv = -snr;
    } else {
      // Compute the (2,1) and (2,2) elements of U**T *A and V**T *B,
      // and (2,2) element of |U|**T *|A| and |V|**T *|B|.
      ua21 = -snl * a1;
      ua22 = -snl * a2 + csl * a3;
      vb21 = -snr * b1;
      vb22 = -snr * b2 + csr * b3;
      aua22 = std::abs(snl) * std::abs(a2) + std::abs(csl) * std::abs(a3);
      avb22 = std::abs(snr) * std::abs(b2) + std::abs(csr) * std::abs(b3);
      // zero (2,2) elements of U**T*A and V**T*B, and then swap.
      if ((std::abs(ua21) + std::abs(ua22)) != zero) {
        if (aua22 / (std::abs(ua21) + std::abs(ua22)) <=
            avb22 / (std::abs(vb21) + std::abs(vb22))) {
          Lartg(-ua21, ua22, csq, snq, &r);
        } else {
          Lartg(-vb21, vb22, csq, snq, &r);
        }
      } else {
        Lartg(-vb21, vb22, csq, snq, &r);
      }
      *csu = snl;
      *snu = csl;
      *csv = snr;
      *snv = csr;
    }
  } else {
    // Input matrices A and B are lower triangular matrices
    // Form matrix C = A*adj(B) = ( a 0 )
    // ( c d )
    a = a1 * b3;
    d = a3 * b1;
    c = a2 * b3 - a3 * b2;
    // The SVD of real 2-by-2 triangular C
    // ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
    // ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
    Lasv2(a, c, d, &s1, &s2, &snr, &csr, &snl, &csl);
    if (std::abs(csr) >= std::abs(snr) || std::abs(csl) >= std::abs(snl)) {
      // Compute the (2,1) and (2,2) elements of U**T *A and V**T *B,
      // and (2,1) element of |U|**T *|A| and |V|**T *|B|.
      ua21 = -snr * a1 + csr * a2;
      ua22r = csr * a3;
      vb21 = -snl * b1 + csl * b2;
      vb22r = csl * b3;
      aua21 = std::abs(snr) * std::abs(a1) + std::abs(csr) * std::abs(a2);
      avb21 = std::abs(snl) * std::abs(b1) + std::abs(csl) * std::abs(b2);
      // zero (2,1) elements of U**T *A and V**T *B.
      if ((std::abs(ua21) + std::abs(ua22r)) != zero) {
        if (aua21 / (std::abs(ua21) + std::abs(ua22r)) <=
            avb21 / (std::abs(vb21) + std::abs(vb22r))) {
          Lartg(ua22r, ua21, csq, snq, &r);
        } else {
          Lartg(vb22r, vb21, csq, snq, &r);
        }
      } else {
        Lartg(vb22r, vb21, csq, snq, &r);
      }
      *csu = csr;
      *snu = -snr;
      *csv = csl;
      *snv = -snl;
    } else {
      // Compute the (1,1) and (1,2) elements of U**T *A and V**T *B,
      // and (1,1) element of |U|**T *|A| and |V|**T *|B|.
      ua11 = csr * a1 + snr * a2;
      ua12 = snr * a3;
      vb11 = csl * b1 + snl * b2;
      vb12 = snl * b3;
      aua11 = std::abs(csr) * std::abs(a1) + std::abs(snr) * std::abs(a2);
      avb11 = std::abs(csl) * std::abs(b1) + std::abs(snl) * std::abs(b2);
      // zero (1,1) elements of U**T*A and V**T*B, and then swap.
      if ((std::abs(ua11) + std::abs(ua12)) != zero) {
        if (aua11 / (std::abs(ua11) + std::abs(ua12)) <=
            avb11 / (std::abs(vb11) + std::abs(vb12))) {
          Lartg(ua12, ua11, csq, snq, &r);
        } else {
          Lartg(vb12, vb11, csq, snq, &r);
        }
      } else {
        Lartg(vb12, vb11, csq, snq, &r);
      }
      *csu = snr;
      *snu = csr;
      *csv = snl;
      *snv = csl;
    }
  }
}

// xLAGS2 (complex T): as the real routine for the unitary U, V, Q of the
// complex A and B with real diagonals.
template <typename T>
void Lags2Complex(bool upper, RealT<T> a1, T a2, RealT<T> a3, RealT<T> b1, T b2,
                  RealT<T> b3, RealT<T>* csu, T* snu, RealT<T>* csv, T* snv,
                  RealT<T>* csq, T* snq) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  R a, aua11, aua12, aua21, aua22, avb11, avb12, avb21, avb22, csl, csr, d, fb,
      fc, s1, s2, snl, snr, ua11r, ua22r, vb11r, vb22r;
  T b, c, d1, r, ua11, ua12, ua21, ua22, vb11, vb12, vb21, vb22;
  if (upper) {
    // Input matrices A and B are upper triangular matrices
    // Form matrix C = A*adj(B) = ( a b )
    // ( 0 d )
    a = a1 * b3;
    d = a3 * b1;
    b = a2 * b1 - a1 * b2;
    fb = std::abs(b);
    // Transform complex 2-by-2 matrix C to real matrix by unitary
    // diagonal matrix diag(1,D1).
    d1 = one;
    if (fb != zero) d1 = b / fb;
    // The SVD of real 2 by 2 triangular C
    // ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
    // ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
    Lasv2<R>(a, fb, d, &s1, &s2, &snr, &csr, &snl, &csl);
    if (std::abs(csl) >= std::abs(snl) || std::abs(csr) >= std::abs(snr)) {
      // Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
      // and (1,2) element of |U|**H *|A| and |V|**H *|B|.
      ua11r = csl * a1;
      ua12 = csl * a2 + d1 * snl * a3;
      vb11r = csr * b1;
      vb12 = csr * b2 + d1 * snr * b3;
      aua12 = std::abs(csl) * Cabs1(a2) + std::abs(snl) * std::abs(a3);
      avb12 = std::abs(csr) * Cabs1(b2) + std::abs(snr) * std::abs(b3);
      // zero (1,2) elements of U**H *A and V**H *B
      if ((std::abs(ua11r) + Cabs1(ua12)) == zero) {
        Lartg(-T(vb11r), Conj(vb12), csq, snq, &r);
      } else if ((std::abs(vb11r) + Cabs1(vb12)) == zero) {
        Lartg(-T(ua11r), Conj(ua12), csq, snq, &r);
      } else if (aua12 / (std::abs(ua11r) + Cabs1(ua12)) <=
                 avb12 / (std::abs(vb11r) + Cabs1(vb12))) {
        Lartg(-T(ua11r), Conj(ua12), csq, snq, &r);
      } else {
        Lartg(-T(vb11r), Conj(vb12), csq, snq, &r);
      }
      *csu = csl;
      *snu = -d1 * snl;
      *csv = csr;
      *snv = -d1 * snr;
    } else {
      // Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
      // and (2,2) element of |U|**H *|A| and |V|**H *|B|.
      ua21 = -Conj(d1) * snl * a1;
      ua22 = -Conj(d1) * snl * a2 + csl * a3;
      vb21 = -Conj(d1) * snr * b1;
      vb22 = -Conj(d1) * snr * b2 + csr * b3;
      aua22 = std::abs(snl) * Cabs1(a2) + std::abs(csl) * std::abs(a3);
      avb22 = std::abs(snr) * Cabs1(b2) + std::abs(csr) * std::abs(b3);
      // zero (2,2) elements of U**H *A and V**H *B, and then swap.
      if ((Cabs1(ua21) + Cabs1(ua22)) == zero) {
        Lartg(-Conj(vb21), Conj(vb22), csq, snq, &r);
      } else if ((Cabs1(vb21) + std::abs(vb22)) == zero) {
        Lartg(-Conj(ua21), Conj(ua22), csq, snq, &r);
      } else if (aua22 / (Cabs1(ua21) + Cabs1(ua22)) <=
                 avb22 / (Cabs1(vb21) + Cabs1(vb22))) {
        Lartg(-Conj(ua21), Conj(ua22), csq, snq, &r);
      } else {
        Lartg(-Conj(vb21), Conj(vb22), csq, snq, &r);
      }
      *csu = snl;
      *snu = d1 * csl;
      *csv = snr;
      *snv = d1 * csr;
    }
  } else {
    // Input matrices A and B are lower triangular matrices
    // Form matrix C = A*adj(B) = ( a 0 )
    // ( c d )
    a = a1 * b3;
    d = a3 * b1;
    c = a2 * b3 - a3 * b2;
    fc = std::abs(c);
    // Transform complex 2-by-2 matrix C to real matrix by unitary
    // diagonal matrix diag(d1,1).
    d1 = one;
    if (fc != zero) d1 = c / fc;
    // The SVD of real 2 by 2 triangular C
    // ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
    // ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
    Lasv2<R>(a, fc, d, &s1, &s2, &snr, &csr, &snl, &csl);
    if (std::abs(csr) >= std::abs(snr) || std::abs(csl) >= std::abs(snl)) {
      // Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
      // and (2,1) element of |U|**H *|A| and |V|**H *|B|.
      ua21 = -d1 * snr * a1 + csr * a2;
      ua22r = csr * a3;
      vb21 = -d1 * snl * b1 + csl * b2;
      vb22r = csl * b3;
      aua21 = std::abs(snr) * std::abs(a1) + std::abs(csr) * Cabs1(a2);
      avb21 = std::abs(snl) * std::abs(b1) + std::abs(csl) * Cabs1(b2);
      // zero (2,1) elements of U**H *A and V**H *B.
      if ((Cabs1(ua21) + std::abs(ua22r)) == zero) {
        Lartg(T(vb22r), vb21, csq, snq, &r);
      } else if ((Cabs1(vb21) + std::abs(vb22r)) == zero) {
        Lartg(T(ua22r), ua21, csq, snq, &r);
      } else if (aua21 / (Cabs1(ua21) + std::abs(ua22r)) <=
                 avb21 / (Cabs1(vb21) + std::abs(vb22r))) {
        Lartg(T(ua22r), ua21, csq, snq, &r);
      } else {
        Lartg(T(vb22r), vb21, csq, snq, &r);
      }
      *csu = csr;
      *snu = -Conj(d1) * snr;
      *csv = csl;
      *snv = -Conj(d1) * snl;
    } else {
      // Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
      // and (1,1) element of |U|**H *|A| and |V|**H *|B|.
      ua11 = csr * a1 + Conj(d1) * snr * a2;
      ua12 = Conj(d1) * snr * a3;
      vb11 = csl * b1 + Conj(d1) * snl * b2;
      vb12 = Conj(d1) * snl * b3;
      aua11 = std::abs(csr) * std::abs(a1) + std::abs(snr) * Cabs1(a2);
      avb11 = std::abs(csl) * std::abs(b1) + std::abs(snl) * Cabs1(b2);
      // zero (1,1) elements of U**H *A and V**H *B, and then swap.
      if ((Cabs1(ua11) + Cabs1(ua12)) == zero) {
        Lartg(vb12, vb11, csq, snq, &r);
      } else if ((Cabs1(vb11) + Cabs1(vb12)) == zero) {
        Lartg(ua12, ua11, csq, snq, &r);
      } else if (aua11 / (Cabs1(ua11) + Cabs1(ua12)) <=
                 avb11 / (Cabs1(vb11) + Cabs1(vb12))) {
        Lartg(ua12, ua11, csq, snq, &r);
      } else {
        Lartg(vb12, vb11, csq, snq, &r);
      }
      *csu = snr;
      *snu = Conj(d1) * csr;
      *csv = snl;
      *snv = Conj(d1) * csl;
    }
  }
}

// NOLINTEND(bugprone-branch-clone)

template <typename T>
void Lags2(bool upper, RealT<T> a1, T a2, RealT<T> a3, RealT<T> b1, T b2,
           RealT<T> b3, RealT<T>* csu, T* snu, RealT<T>* csv, T* snv,
           RealT<T>* csq, T* snq) {
  if constexpr (IsComplex<T>::value) {
    Lags2Complex(upper, a1, a2, a3, b1, b2, b3, csu, snu, csv, snv, csq, snq);
  } else {
    Lags2Real(upper, a1, a2, a3, b1, b2, b3, csu, snu, csv, snv, csq, snq);
  }
}

// xTGSJA: computes the GSVD of the M-by-N matrix A and the P-by-N matrix B
// in the upper triangular form of xGGSVP3 (with K and L from it) by
// Jacobi-Kogbetliantz iteration: U**H*A*Q = D1*( 0 R ), V**H*B*Q = D2*( 0
// R ), with C = diag(ALPHA(K+1:K+L)), S = diag(BETA(K+1:K+L)) in D1 and D2,
// C**2 + S**2 = I, and R upper triangular in A (and B, if M-K < L). JOBU,
// JOBV, JOBQ = 'U' / 'V' / 'Q': update U, V, Q (on entry the matrices of
// xGGSVP3); 'I': initialize them to the identity; 'N': not computed. TOLA,
// TOLB are the convergence thresholds (those of xGGSVP3). WORK has 2*N
// elements; NCYCLE returns the number of cycles.
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if the iteration did
// not converge in MAXIT = 40 cycles. The real routine does not make the
// diagonals of A and B real after each rotation (if constexpr).
template <typename T>
blasint Tgsja(char jobu, char jobv, char jobq, blasint m, blasint p, blasint n,
              blasint k, blasint l, T* a, blasint lda, T* b, blasint ldb,
              RealT<T> tola, RealT<T> tolb, RealT<T>* alpha, RealT<T>* beta,
              T* u, blasint ldu, T* v, blasint ldv, T* q, blasint ldq, T* work,
              blasint* ncycle) {
  using R = RealT<T>;

  const blasint maxit = 40;
  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const T cone = T(1);
  const R hugenum = std::numeric_limits<R>::max();
  const Matrix<T> A(a, lda);
  const Vector<R> ALPHA(alpha);
  const Matrix<T> B(b, ldb);
  const Vector<R> BETA(beta);
  const Matrix<T> Q(q, ldq);
  const Matrix<T> U(u, ldu);
  const Matrix<T> V(v, ldv);
  const Vector<T> WORK(work);
  blasint i, j, kcycle;
  bool initq, initu, initv, upper, wantq, wantu, wantv;
  R a1, a3, b1, b3, csq, csu, csv, error, gamma, rwk, ssmin;
  T a2, b2, snq, snu, snv;
  initu = Lsame(jobu, 'I');
  wantu = initu || Lsame(jobu, 'U');
  initv = Lsame(jobv, 'I');
  wantv = initv || Lsame(jobv, 'V');
  initq = Lsame(jobq, 'I');
  wantq = initq || Lsame(jobq, 'Q');
  blasint info = 0;
  if (!(initu || wantu || Lsame(jobu, 'N'))) {
    info = -1;
  } else if (!(initv || wantv || Lsame(jobv, 'N'))) {
    info = -2;
  } else if (!(initq || wantq || Lsame(jobq, 'N'))) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (p < 0) {
    info = -5;
  } else if (n < 0) {
    info = -6;
  } else if (lda < Max<blasint>(1, m)) {
    info = -10;
  } else if (ldb < Max<blasint>(1, p)) {
    info = -12;
  } else if (ldu < 1 || (wantu && ldu < m)) {
    info = -18;
  } else if (ldv < 1 || (wantv && ldv < p)) {
    info = -20;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -22;
  }
  if (info != 0) {
    Xerbla<T>("TGSJA", -info);
    return info;
  }
  // Initialize U, V and Q, if necessary
  if (initu) Laset('F', m, m, czero, cone, u, ldu);
  if (initv) Laset('F', p, p, czero, cone, v, ldv);
  if (initq) Laset('F', n, n, czero, cone, q, ldq);
  // Loop until convergence
  upper = false;
  for (kcycle = 1; kcycle <= maxit; ++kcycle) {
    upper = !upper;
    for (i = 1; i <= l - 1; ++i) {
      for (j = i + 1; j <= l; ++j) {
        a1 = zero;
        a2 = czero;
        a3 = zero;
        if (k + i <= m) a1 = std::real(A(k + i, n - l + i));
        if (k + j <= m) a3 = std::real(A(k + j, n - l + j));
        b1 = std::real(B(i, n - l + i));
        b3 = std::real(B(j, n - l + j));
        if (upper) {
          if (k + i <= m) a2 = A(k + i, n - l + j);
          b2 = B(i, n - l + j);
        } else {
          if (k + j <= m) a2 = A(k + j, n - l + i);
          b2 = B(j, n - l + i);
        }
        Lags2(upper, a1, a2, a3, b1, b2, b3, &csu, &snu, &csv, &snv, &csq,
              &snq);
        // Update (K+I)-th and (K+J)-th rows of matrix A: U**H *A
        if (k + j <= m)
          Rot(l, A.ptr(k + j, n - l + 1), lda, A.ptr(k + i, n - l + 1), lda,
              csu, Conj(snu));
        // Update I-th and J-th rows of matrix B: V**H *B
        Rot(l, B.ptr(j, n - l + 1), ldb, B.ptr(i, n - l + 1), ldb, csv,
            Conj(snv));
        // Update (N-L+I)-th and (N-L+J)-th columns of matrices
        // A and B: A*Q and B*Q
        Rot(Min<blasint>(k + l, m), A.ptr(1, n - l + j), 1, A.ptr(1, n - l + i),
            1, csq, snq);
        Rot(l, B.ptr(1, n - l + j), 1, B.ptr(1, n - l + i), 1, csq, snq);
        if (upper) {
          if (k + i <= m) A(k + i, n - l + j) = czero;
          B(i, n - l + j) = czero;
        } else {
          if (k + j <= m) A(k + j, n - l + i) = czero;
          B(j, n - l + i) = czero;
        }
        // Ensure that the diagonal elements of A and B are real.
        if constexpr (IsComplex<T>::value) {
          if (k + i <= m) A(k + i, n - l + i) = std::real(A(k + i, n - l + i));
          if (k + j <= m) A(k + j, n - l + j) = std::real(A(k + j, n - l + j));
          B(i, n - l + i) = std::real(B(i, n - l + i));
          B(j, n - l + j) = std::real(B(j, n - l + j));
        }
        // Update unitary matrices U, V, Q, if desired.
        if (wantu && k + j <= m)
          Rot(m, U.ptr(1, k + j), 1, U.ptr(1, k + i), 1, csu, snu);
        if (wantv) Rot(p, V.ptr(1, j), 1, V.ptr(1, i), 1, csv, snv);
        if (wantq)
          Rot(n, Q.ptr(1, n - l + j), 1, Q.ptr(1, n - l + i), 1, csq, snq);
      }
    }
    if (!upper) {
      // The matrices A13 and B13 were lower triangular at the start
      // of the cycle, and are now upper triangular.
      // Convergence test: test the parallelism of the corresponding
      // rows of A and B.
      error = zero;
      for (i = 1; i <= Min<blasint>(l, m - k); ++i) {
        Copy(l - i + 1, A.ptr(k + i, n - l + i), lda, work, 1);
        Copy(l - i + 1, B.ptr(i, n - l + i), ldb, WORK.ptr(l + 1), 1);
        Lapll(l - i + 1, work, 1, WORK.ptr(l + 1), 1, &ssmin);
        error = Max(error, ssmin);
      }
      if (std::abs(error) <= Min(tola, tolb)) goto L50;
    }
    // End of cycle loop
  }
  // The algorithm has not converged after MAXIT cycles.
  info = 1;
  goto L100;
L50:
  // If ERROR <= MIN(TOLA,TOLB), then the algorithm has converged.
  // Compute the generalized singular value pairs (ALPHA, BETA), and
  // set the triangular matrix R to array A.
  for (i = 1; i <= k; ++i) {
    ALPHA(i) = one;
    BETA(i) = zero;
  }
  for (i = 1; i <= Min<blasint>(l, m - k); ++i) {
    a1 = std::real(A(k + i, n - l + i));
    b1 = std::real(B(i, n - l + i));
    gamma = b1 / a1;
    if ((gamma <= hugenum) && (gamma >= -hugenum)) {
      if (gamma < zero) {
        Rscal(l - i + 1, -one, B.ptr(i, n - l + i), ldb);
        if (wantv) Rscal(p, -one, V.ptr(1, i), 1);
      }
      Lartg(std::abs(gamma), one, BETA.ptr(k + i), ALPHA.ptr(k + i), &rwk);
      if (ALPHA(k + i) >= BETA(k + i)) {
        Rscal(l - i + 1, one / ALPHA(k + i), A.ptr(k + i, n - l + i), lda);
      } else {
        Rscal(l - i + 1, one / BETA(k + i), B.ptr(i, n - l + i), ldb);
        Copy(l - i + 1, B.ptr(i, n - l + i), ldb, A.ptr(k + i, n - l + i), lda);
      }
    } else {
      ALPHA(k + i) = zero;
      BETA(k + i) = one;
      Copy(l - i + 1, B.ptr(i, n - l + i), ldb, A.ptr(k + i, n - l + i), lda);
    }
  }
  // Post-assignment
  for (i = m + 1; i <= k + l; ++i) {
    ALPHA(i) = zero;
    BETA(i) = one;
  }
  if (k + l < n) {
    for (i = k + l + 1; i <= n; ++i) {
      ALPHA(i) = zero;
      BETA(i) = zero;
    }
  }
L100:
  *ncycle = kcycle;
  return info;
}

// xGGSVP3: the preprocessing for the GSVD of the M-by-N matrix A and the
// P-by-N matrix B: orthogonal (unitary) U, V, Q such that U**H*A*Q = ( 0
// A12 A13 ; 0 0 A23 ; 0 0 0 ) and V**H*B*Q = ( 0 0 B13 ; 0 0 0 ) with K
// + L columns, A12 (K-by-K) and B13 (L-by-L) upper triangular and
// nonsingular, by QR factorizations with column pivoting (xGEQP3) with
// the rank thresholds TOLA and TOLB. IWORK has N elements, TAU N, RWORK
// 2*N (complex T only), WORK LWORK >= max(1, N, M, P) (LWORK = -1:
// workspace query).
template <typename T>
blasint Ggsvp3(char jobu, char jobv, char jobq, blasint m, blasint p, blasint n,
               T* a, blasint lda, T* b, blasint ldb, RealT<T> tola,
               RealT<T> tolb, blasint* k, blasint* l, T* u, blasint ldu, T* v,
               blasint ldv, T* q, blasint ldq, blasint* iwork, RealT<T>* rwork,
               T* tau, T* work, blasint lwork) {
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<blasint> IWORK(iwork);
  const Matrix<T> Q(q, ldq);
  const Vector<T> TAU(tau);
  const Matrix<T> U(u, ldu);
  const Matrix<T> V(v, ldv);
  const Vector<T> WORK(work);
  blasint i, j, lwkopt;
  bool forwrd, lquery, wantq, wantu, wantv;
  wantu = Lsame(jobu, 'U');
  wantv = Lsame(jobv, 'V');
  wantq = Lsame(jobq, 'Q');
  forwrd = true;
  lquery = (lwork == -1);
  lwkopt = 1;
  // Test the input arguments
  blasint info = 0;
  if (!(wantu || Lsame(jobu, 'N'))) {
    info = -1;
  } else if (!(wantv || Lsame(jobv, 'N'))) {
    info = -2;
  } else if (!(wantq || Lsame(jobq, 'N'))) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (p < 0) {
    info = -5;
  } else if (n < 0) {
    info = -6;
  } else if (lda < Max<blasint>(1, m)) {
    info = -8;
  } else if (ldb < Max<blasint>(1, p)) {
    info = -10;
  } else if (ldu < 1 || (wantu && ldu < m)) {
    info = -16;
  } else if (ldv < 1 || (wantv && ldv < p)) {
    info = -18;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -20;
  } else if (lwork < 1 && !lquery) {
    info = -24;
  }
  // Compute workspace
  if (info == 0) {
    info = Geqp3(p, n, b, ldb, iwork, tau, work, -1, rwork);
    lwkopt = static_cast<blasint>(std::real(WORK(1)));
    if (wantv) {
      lwkopt = Max<blasint>(lwkopt, p);
    }
    lwkopt = Max<blasint>(lwkopt, Min<blasint>(n, p));
    lwkopt = Max<blasint>(lwkopt, m);
    if (wantq) {
      lwkopt = Max<blasint>(lwkopt, n);
    }
    info = Geqp3(m, n, a, lda, iwork, tau, work, -1, rwork);
    lwkopt = Max(lwkopt, static_cast<blasint>(std::real(WORK(1))));
    lwkopt = Max<blasint>(1, lwkopt);
    WORK(1) = WorkSize<T>(lwkopt);
  }
  if (info != 0) {
    Xerbla<T>("GGSVP3", -info);
    return info;
  }
  if (lquery) {
    return info;
  }
  // QR with column pivoting of B: B*P = V*( S11 S12 )
  // (  0   0  )
  for (i = 1; i <= n; ++i) {
    IWORK(i) = 0;
  }
  info = Geqp3(p, n, b, ldb, iwork, tau, work, lwork, rwork);
  // Update A := A*P
  Lapmt(forwrd, m, n, a, lda, iwork);
  // Determine the effective rank of matrix B.
  *l = 0;
  for (i = 1; i <= Min<blasint>(p, n); ++i) {
    if (std::abs(B(i, i)) > tolb) *l = *l + 1;
  }
  if (wantv) {
    // Copy the details of V, and form V.
    Laset('F', p, p, czero, czero, v, ldv);
    if (p > 1) Lacpy('L', p - 1, n, B.ptr(2, 1), ldb, V.ptr(2, 1), ldv);
    info = Ung2r(p, p, Min<blasint>(p, n), v, ldv, tau, work);
  }
  // Clean up B
  for (j = 1; j <= *l - 1; ++j) {
    for (i = j + 1; i <= *l; ++i) {
      B(i, j) = czero;
    }
  }
  if (p > *l) Laset('F', p - *l, n, czero, czero, B.ptr(*l + 1, 1), ldb);
  if (wantq) {
    // Set Q = I and Update Q := Q*P
    Laset('F', n, n, czero, cone, q, ldq);
    Lapmt(forwrd, n, n, q, ldq, iwork);
  }
  if (p >= *l && n != *l) {
    // RQ factorization of ( S11 S12 ) = ( 0 S12 )*Z
    info = Gerq2(*l, n, b, ldb, tau, work);
    // Update A := A*Z**H
    info = Unmr2('R', ConjTrans<T>(), m, n, *l, b, ldb, tau, a, lda, work);
    if (wantq) {
      // Update Q := Q*Z**H
      info = Unmr2('R', ConjTrans<T>(), n, n, *l, b, ldb, tau, q, ldq, work);
    }
    // Clean up B
    Laset('F', *l, n - *l, czero, czero, b, ldb);
    for (j = n - *l + 1; j <= n; ++j) {
      for (i = j - n + *l + 1; i <= *l; ++i) {
        B(i, j) = czero;
      }
    }
  }
  // Let              N-L     L
  // A = ( A11    A12 ) M,
  // then the following does the complete QR decomposition of A11:
  // A11 = U*(  0  T12 )*P1**H
  // (  0   0  )
  for (i = 1; i <= n - *l; ++i) {
    IWORK(i) = 0;
  }
  info = Geqp3(m, n - *l, a, lda, iwork, tau, work, lwork, rwork);
  // Determine the effective rank of A11
  *k = 0;
  for (i = 1; i <= Min<blasint>(m, n - *l); ++i) {
    if (std::abs(A(i, i)) > tola) *k = *k + 1;
  }
  // Update A12 := U**H*A12, where A12 = A( 1:M, N-L+1:N )
  info = Unm2r('L', ConjTrans<T>(), m, *l, Min<blasint>(m, n - *l), a, lda, tau,
               A.ptr(1, n - *l + 1), lda, work);
  if (wantu) {
    // Copy the details of U, and form U
    Laset('F', m, m, czero, czero, u, ldu);
    if (m > 1) Lacpy('L', m - 1, n - *l, A.ptr(2, 1), lda, U.ptr(2, 1), ldu);
    info = Ung2r(m, m, Min<blasint>(m, n - *l), u, ldu, tau, work);
  }
  if (wantq) {
    // Update Q( 1:N, 1:N-L )  = Q( 1:N, 1:N-L )*P1
    Lapmt(forwrd, n, n - *l, q, ldq, iwork);
  }
  // Clean up A: set the strictly lower triangular part of
  // A(1:K, 1:K) = 0, and A( K+1:M, 1:N-L ) = 0.
  for (j = 1; j <= *k - 1; ++j) {
    for (i = j + 1; i <= *k; ++i) {
      A(i, j) = czero;
    }
  }
  if (m > *k) Laset('F', m - *k, n - *l, czero, czero, A.ptr(*k + 1, 1), lda);
  if (n - *l > *k) {
    // RQ factorization of ( T11 T12 ) = ( 0 T12 )*Z1
    info = Gerq2(*k, n - *l, a, lda, tau, work);
    if (wantq) {
      // Update Q( 1:N,1:N-L ) = Q( 1:N,1:N-L )*Z1**H
      info =
          Unmr2('R', ConjTrans<T>(), n, n - *l, *k, a, lda, tau, q, ldq, work);
    }
    // Clean up A
    Laset('F', *k, n - *l - *k, czero, czero, a, lda);
    for (j = n - *l - *k + 1; j <= n - *l; ++j) {
      for (i = j - n + *l + *k + 1; i <= *k; ++i) {
        A(i, j) = czero;
      }
    }
  }
  if (m > *k) {
    // QR factorization of A( K+1:M,N-L+1:N )
    info = Geqr2(m - *k, *l, A.ptr(*k + 1, n - *l + 1), lda, tau, work);
    if (wantu) {
      // Update U(:,K+1:M) := U(:,K+1:M)*U1
      info = Unm2r('R', 'N', m, m - *k, Min<blasint>(m - *k, *l),
                   A.ptr(*k + 1, n - *l + 1), lda, tau, U.ptr(1, *k + 1), ldu,
                   work);
    }
    // Clean up
    for (j = n - *l + 1; j <= n; ++j) {
      for (i = j - n + *k + *l + 1; i <= m; ++i) {
        A(i, j) = czero;
      }
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xGGSVD3: the generalized singular value decomposition of the M-by-N
// matrix A and the P-by-N matrix B: U**H*A*Q = D1*( 0 R ), V**H*B*Q =
// D2*( 0 R ) with orthogonal (unitary) U, V, Q, R upper triangular
// nonsingular of order K+L = the effective rank of ( A ; B ), and the
// generalized singular values ALPHA / BETA (the diagonals of D1 and D2;
// see LAPACK's xGGSVD3 for their layout; IWORK returns the sorting
// permutation). JOBU / JOBV / JOBQ = 'U' / 'V' / 'Q' compute U / V / Q,
// 'N' not. WORK has LWORK elements (LWORK = -1: workspace query), RWORK
// 2*N (complex T only), IWORK N.
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if the Jacobi
// iteration did not converge.
template <typename T>
blasint Ggsvd3(char jobu, char jobv, char jobq, blasint m, blasint n, blasint p,
               blasint* k, blasint* l, T* a, blasint lda, T* b, blasint ldb,
               RealT<T>* alpha, RealT<T>* beta, T* u, blasint ldu, T* v,
               blasint ldv, T* q, blasint ldq, T* work, blasint lwork,
               RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;

  const Matrix<T> A(a, lda);
  const Vector<R> ALPHA(alpha);
  const Matrix<T> B(b, ldb);
  const Vector<R> BETA(beta);
  const Vector<blasint> IWORK(iwork);
  const Matrix<T> Q(q, ldq);
  // The real routine sorts ALPHA in WORK.
  R* const rw = IsComplex<T>::value ? rwork : reinterpret_cast<R*>(work);
  const Vector<R> RWORK(rw);
  const Matrix<T> U(u, ldu);
  const Matrix<T> V(v, ldv);
  const Vector<T> WORK(work);
  blasint i, ibnd, isub, j, lwkopt, ncycle;
  // Set wherever they are read (ISUB with SMAX).
  isub = 0;
  bool lquery, wantq, wantu, wantv;
  R anorm, bnorm, smax, temp, ulp, unfl;
  // Not used by the workspace query of xGGSVP3 (undefined there in LAPACK).
  R tola = R(0), tolb = R(0);
  wantu = Lsame(jobu, 'U');
  wantv = Lsame(jobv, 'V');
  wantq = Lsame(jobq, 'Q');
  lquery = (lwork == -1);
  lwkopt = 1;
  // Test the input arguments
  blasint info = 0;
  if (!(wantu || Lsame(jobu, 'N'))) {
    info = -1;
  } else if (!(wantv || Lsame(jobv, 'N'))) {
    info = -2;
  } else if (!(wantq || Lsame(jobq, 'N'))) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (p < 0) {
    info = -6;
  } else if (lda < Max<blasint>(1, m)) {
    info = -10;
  } else if (ldb < Max<blasint>(1, p)) {
    info = -12;
  } else if (ldu < 1 || (wantu && ldu < m)) {
    info = -16;
  } else if (ldv < 1 || (wantv && ldv < p)) {
    info = -18;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -20;
  } else if (lwork < 1 && !lquery) {
    info = -24;
  }
  // Compute workspace
  if (info == 0) {
    info = Ggsvp3(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, tola, tolb, k, l,
                  u, ldu, v, ldv, q, ldq, iwork, rwork, work, work, -1);
    lwkopt = n + static_cast<blasint>(std::real(WORK(1)));
    lwkopt = Max<blasint>(2 * n, lwkopt);
    lwkopt = Max<blasint>(1, lwkopt);
    WORK(1) = WorkSize<T>(lwkopt);
  }
  if (info != 0) {
    Xerbla<T>("GGSVD3", -info);
    return info;
  }
  if (lquery) {
    return info;
  }
  // Compute the Frobenius norm of matrices A and B
  anorm = Lange('1', m, n, a, lda, rw);
  bnorm = Lange('1', p, n, b, ldb, rw);
  // Get machine precision and set up threshold for determining
  // the effective numerical rank of the matrices A and B.
  ulp = Lamch<R>('P');
  unfl = Lamch<R>('S');
  tola = R(Max<blasint>(m, n)) * Max(anorm, unfl) * ulp;
  tolb = R(Max<blasint>(p, n)) * Max(bnorm, unfl) * ulp;
  info = Ggsvp3(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, tola, tolb, k, l, u,
                ldu, v, ldv, q, ldq, iwork, rwork, work, WORK.ptr(n + 1),
                lwork - n);
  // Compute the GSVD of two upper "triangular" matrices
  info = Tgsja(jobu, jobv, jobq, m, p, n, *k, *l, a, lda, b, ldb, tola, tolb,
               alpha, beta, u, ldu, v, ldv, q, ldq, work, &ncycle);
  // Sort the singular values and store the pivot indices in IWORK
  // Copy ALPHA to RWORK, then sort ALPHA in RWORK
  Copy<R>(n, alpha, 1, rw, 1);
  ibnd = Min<blasint>(*l, m - *k);
  for (i = 1; i <= ibnd; ++i) {
    // Scan for largest ALPHA(K+I)
    isub = i;
    smax = RWORK(*k + i);
    for (j = i + 1; j <= ibnd; ++j) {
      temp = RWORK(*k + j);
      if (temp > smax) {
        isub = j;
        smax = temp;
      }
    }
    if (isub != i) {
      RWORK(*k + isub) = RWORK(*k + i);
      RWORK(*k + i) = smax;
      IWORK(*k + i) = *k + isub;
    } else {
      IWORK(*k + i) = *k + i;
    }
  }
  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template void Lapll<T>(blasint, T*, blasint, T*, blasint, RealT<T>*);        \
  template void Lapmt<T>(bool, blasint, blasint, T*, blasint, blasint*);       \
  template void Lags2<T>(bool, RealT<T>, T, RealT<T>, RealT<T>, T, RealT<T>,   \
                         RealT<T>*, T*, RealT<T>*, T*, RealT<T>*, T*);         \
  template blasint Tgsja<T>(char, char, char, blasint, blasint, blasint,       \
                            blasint, blasint, T*, blasint, T*, blasint,        \
                            RealT<T>, RealT<T>, RealT<T>*, RealT<T>*, T*,      \
                            blasint, T*, blasint, T*, blasint, T*, blasint*);  \
  template blasint Ggsvp3<T>(char, char, char, blasint, blasint, blasint, T*,  \
                             blasint, T*, blasint, RealT<T>, RealT<T>,         \
                             blasint*, blasint*, T*, blasint, T*, blasint, T*, \
                             blasint, blasint*, RealT<T>*, T*, T*, blasint);   \
  template blasint Ggsvd3<T>(char, char, char, blasint, blasint, blasint,      \
                             blasint*, blasint*, T*, blasint, T*, blasint,     \
                             RealT<T>*, RealT<T>*, T*, blasint, T*, blasint,   \
                             T*, blasint, T*, blasint, RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// FORWRD and UPPER are LOGICALs (nonzero is true).
#define HWYBLAS_LAPACK_GGSVD3(prefix, T, R)                                  \
  HWYBLAS_API void prefix##lapll_(const hwyblas_int* n, R* x,                \
                                  const hwyblas_int* incx, R* y,             \
                                  const hwyblas_int* incy, R* ssmin) {       \
    hwyblas::lapack::Lapll(*n, As<T>(x), *incx, As<T>(y), *incy, ssmin);     \
  }                                                                          \
  HWYBLAS_API void prefix##lapmt_(                                           \
      const hwyblas_int* forwrd, const hwyblas_int* m, const hwyblas_int* n, \
      R* x, const hwyblas_int* ldx, hwyblas_int* k) {                        \
    hwyblas::lapack::Lapmt(*forwrd != 0, *m, *n, As<T>(x), *ldx, k);         \
  }                                                                          \
  HWYBLAS_API void prefix##tgsja_(                                           \
      const char* jobu, const char* jobv, const char* jobq,                  \
      const hwyblas_int* m, const hwyblas_int* p, const hwyblas_int* n,      \
      const hwyblas_int* k, const hwyblas_int* l, R* a,                      \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, const R* tola,   \
      const R* tolb, R* alpha, R* beta, R* u, const hwyblas_int* ldu, R* v,  \
      const hwyblas_int* ldv, R* q, const hwyblas_int* ldq, R* work,         \
      hwyblas_int* ncycle, hwyblas_int* info) {                              \
    *info = hwyblas::lapack::Tgsja(                                          \
        *jobu, *jobv, *jobq, *m, *p, *n, *k, *l, As<T>(a), *lda, As<T>(b),   \
        *ldb, *tola, *tolb, alpha, beta, As<T>(u), *ldu, As<T>(v), *ldv,     \
        As<T>(q), *ldq, As<T>(work), ncycle);                                \
  }

HWYBLAS_LAPACK_GGSVD3(s, float, float)
HWYBLAS_LAPACK_GGSVD3(d, double, double)
HWYBLAS_LAPACK_GGSVD3(c, c32, float)
HWYBLAS_LAPACK_GGSVD3(z, c64, double)

#undef HWYBLAS_LAPACK_GGSVD3

#define HWYBLAS_LAPACK_GGSVD3_REAL(prefix, R)                                  \
  HWYBLAS_API void prefix##lags2_(const hwyblas_int* upper, const R* a1,       \
                                  const R* a2, const R* a3, const R* b1,       \
                                  const R* b2, const R* b3, R* csu, R* snu,    \
                                  R* csv, R* snv, R* csq, R* snq) {            \
    hwyblas::lapack::Lags2<R>(*upper != 0, *a1, *a2, *a3, *b1, *b2, *b3, csu,  \
                              snu, csv, snv, csq, snq);                        \
  }                                                                            \
  HWYBLAS_API void prefix##ggsvp3_(                                            \
      const char* jobu, const char* jobv, const char* jobq,                    \
      const hwyblas_int* m, const hwyblas_int* p, const hwyblas_int* n, R* a,  \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, const R* tola,     \
      const R* tolb, hwyblas_int* k, hwyblas_int* l, R* u,                     \
      const hwyblas_int* ldu, R* v, const hwyblas_int* ldv, R* q,              \
      const hwyblas_int* ldq, hwyblas_int* iwork, R* tau, R* work,             \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Ggsvp3(                                           \
        *jobu, *jobv, *jobq, *m, *p, *n, a, *lda, b, *ldb, *tola, *tolb, k, l, \
        u, *ldu, v, *ldv, q, *ldq, iwork, static_cast<R*>(nullptr), tau, work, \
        *lwork);                                                               \
  }                                                                            \
  HWYBLAS_API void prefix##ggsvd3_(                                            \
      const char* jobu, const char* jobv, const char* jobq,                    \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* p,        \
      hwyblas_int* k, hwyblas_int* l, R* a, const hwyblas_int* lda, R* b,      \
      const hwyblas_int* ldb, R* alpha, R* beta, R* u, const hwyblas_int* ldu, \
      R* v, const hwyblas_int* ldv, R* q, const hwyblas_int* ldq, R* work,     \
      const hwyblas_int* lwork, hwyblas_int* iwork, hwyblas_int* info) {       \
    *info = hwyblas::lapack::Ggsvd3(*jobu, *jobv, *jobq, *m, *n, *p, k, l, a,  \
                                    *lda, b, *ldb, alpha, beta, u, *ldu, v,    \
                                    *ldv, q, *ldq, work, *lwork,               \
                                    static_cast<R*>(nullptr), iwork);          \
  }

HWYBLAS_LAPACK_GGSVD3_REAL(s, float)
HWYBLAS_LAPACK_GGSVD3_REAL(d, double)

#undef HWYBLAS_LAPACK_GGSVD3_REAL

#define HWYBLAS_LAPACK_GGSVD3_COMPLEX(prefix, T, R)                            \
  HWYBLAS_API void prefix##lags2_(const hwyblas_int* upper, const R* a1,       \
                                  const R* a2, const R* a3, const R* b1,       \
                                  const R* b2, const R* b3, R* csu, R* snu,    \
                                  R* csv, R* snv, R* csq, R* snq) {            \
    hwyblas::lapack::Lags2<T>(*upper != 0, *a1, *As<const T>(a2), *a3, *b1,    \
                              *As<const T>(b2), *b3, csu, As<T>(snu), csv,     \
                              As<T>(snv), csq, As<T>(snq));                    \
  }                                                                            \
  HWYBLAS_API void prefix##ggsvp3_(                                            \
      const char* jobu, const char* jobv, const char* jobq,                    \
      const hwyblas_int* m, const hwyblas_int* p, const hwyblas_int* n, R* a,  \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, const R* tola,     \
      const R* tolb, hwyblas_int* k, hwyblas_int* l, R* u,                     \
      const hwyblas_int* ldu, R* v, const hwyblas_int* ldv, R* q,              \
      const hwyblas_int* ldq, hwyblas_int* iwork, R* rwork, R* tau, R* work,   \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info = hwyblas::lapack::Ggsvp3(                                           \
        *jobu, *jobv, *jobq, *m, *p, *n, As<T>(a), *lda, As<T>(b), *ldb,       \
        *tola, *tolb, k, l, As<T>(u), *ldu, As<T>(v), *ldv, As<T>(q), *ldq,    \
        iwork, rwork, As<T>(tau), As<T>(work), *lwork);                        \
  }                                                                            \
  HWYBLAS_API void prefix##ggsvd3_(                                            \
      const char* jobu, const char* jobv, const char* jobq,                    \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* p,        \
      hwyblas_int* k, hwyblas_int* l, R* a, const hwyblas_int* lda, R* b,      \
      const hwyblas_int* ldb, R* alpha, R* beta, R* u, const hwyblas_int* ldu, \
      R* v, const hwyblas_int* ldv, R* q, const hwyblas_int* ldq, R* work,     \
      const hwyblas_int* lwork, R* rwork, hwyblas_int* iwork,                  \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Ggsvd3(                                           \
        *jobu, *jobv, *jobq, *m, *n, *p, k, l, As<T>(a), *lda, As<T>(b), *ldb, \
        alpha, beta, As<T>(u), *ldu, As<T>(v), *ldv, As<T>(q), *ldq,           \
        As<T>(work), *lwork, rwork, iwork);                                    \
  }

HWYBLAS_LAPACK_GGSVD3_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GGSVD3_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GGSVD3_COMPLEX

}  // extern "C"
