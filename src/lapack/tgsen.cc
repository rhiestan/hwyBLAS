// Reordering of the generalized Schur form of a pair (A, B) and the
// condition numbers of the reordered cluster: xTGSEN, with xTGEXC (moves a
// diagonal block), xTGEX2 (swaps two adjacent blocks, with xLAGV2 for the
// real 2-by-2 blocks), the generalized Sylvester equation solvers xTGSYL
// (blocked) and xTGSY2 (unblocked), the LU factorization with complete
// pivoting of a small matrix xGETC2 and its solver xGESC2, and the
// contribution to the reciprocal Dif-estimate, xLATDF.
//
// xGETC2 and xGESC2 are one template each, the real routine for real T and
// the complex routine for complex T. The real and the complex routines of
// the others differ in their arguments (ALPHAR and ALPHAI for ALPHA, the
// 2-by-2 blocks of the real Schur form, workspaces) and are two overloads
// (xTGSEN, xTGEXC, xTGEX2, xTGSY2), or two bodies behind one template
// where their arguments are the same (TgsylReal / TgsylComplex,
// LatdfReal / LatdfComplex).
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

// xGETC2: the LU factorization with complete pivoting of the N-by-N matrix
// A, A = P * L * U * Q (P, Q permutations from the row and column
// interchanges IPIV, JPIV). Pivots smaller than SMIN = max(eps * max|A|,
// SMLNUM) are set to SMIN (a slightly perturbed system is solved then).
//
// Returns INFO: 0, or k > 0 if U(k,k) was perturbed.
template <typename T>
blasint Getc2(blasint n, T* a, blasint lda, blasint* ipiv, blasint* jpiv) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);
  const Vector<blasint> JPIV(jpiv);
  blasint i, ip, j, jp;
  // Set wherever they are read (A has a largest element); undefined in
  // LAPACK if A(i:n,i:n) holds only NaNs.
  blasint ipv = 0, jpv = 0;
  R eps, smlnum, xmax;
  // Set in the first iteration.
  R smin = R(0);
  blasint info = 0;
  // Quick return if possible
  if (n == 0) return info;
  // Set constants to control overflow
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  // Handle the case N=1 by itself
  if (n == 1) {
    IPIV(1) = 1;
    JPIV(1) = 1;
    if (std::abs(A(1, 1)) < smlnum) {
      info = 1;
      A(1, 1) = T(smlnum);
    }
    return info;
  }
  // Factorize A using complete pivoting.
  // Set pivots less than SMIN to SMIN
  for (i = 1; i <= n - 1; ++i) {
    // Find max element in matrix A
    xmax = zero;
    for (jp = i; jp <= n; ++jp) {
      for (ip = i; ip <= n; ++ip) {
        if (std::abs(A(ip, jp)) >= xmax) {
          xmax = std::abs(A(ip, jp));
          ipv = ip;
          jpv = jp;
        }
      }
    }
    if (i == 1) smin = Max(eps * xmax, smlnum);
    // Swap rows
    if (ipv != i) Swap(n, A.ptr(ipv, 1), lda, A.ptr(i, 1), lda);
    IPIV(i) = ipv;
    // Swap columns
    if (jpv != i) Swap(n, A.ptr(1, jpv), 1, A.ptr(1, i), 1);
    JPIV(i) = jpv;
    // Check for singularity
    if (std::abs(A(i, i)) < smin) {
      info = i;
      A(i, i) = T(smin);
    }
    for (j = i + 1; j <= n; ++j) {
      A(j, i) = A(j, i) / A(i, i);
    }
    Geru(n - i, n - i, -T(one), A.ptr(i + 1, i), 1, A.ptr(i, i + 1), lda,
         A.ptr(i + 1, i + 1), lda);
  }
  if (std::abs(A(n, n)) < smin) {
    info = n;
    A(n, n) = T(smin);
  }
  // Set last pivots to N
  IPIV(n) = n;
  JPIV(n) = n;
  return info;
}

// xGESC2: solves A * X = SCALE * RHS with the LU factorization with
// complete pivoting of xGETC2 (A, IPIV, JPIV). SCALE <= 1 is chosen to
// prevent overflow in the solution, which overwrites RHS.
template <typename T>
void Gesc2(blasint n, const T* a, blasint lda, T* rhs, const blasint* ipiv,
           const blasint* jpiv, RealT<T>* scale) {
  using R = RealT<T>;

  const R one = R(1);
  const R two = R(2);
  const Matrix<const T> A(a, lda);
  const Vector<T> RHS(rhs);
  blasint i, j;
  R bignum, eps, smlnum;
  T temp;
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  bignum = one / smlnum;
  // Apply permutations IPIV to RHS
  Laswp(1, rhs, lda, 1, n - 1, ipiv, 1);
  // Solve for L part
  for (i = 1; i <= n - 1; ++i) {
    for (j = i + 1; j <= n; ++j) {
      RHS(j) = RHS(j) - A(j, i) * RHS(i);
    }
  }
  // Solve for U part
  *scale = one;
  // Check for scaling
  i = Iamax(n, rhs, 1);
  if (two * smlnum * std::abs(RHS(i)) > std::abs(A(n, n))) {
    temp = T(one / two) / std::abs(RHS(i));
    Scal(n, temp, RHS.ptr(1), 1);
    *scale = *scale * std::real(temp);
  }
  for (i = n; i >= 1; --i) {
    temp = T(one) / A(i, i);
    RHS(i) = RHS(i) * temp;
    for (j = i + 1; j <= n; ++j) {
      RHS(i) = RHS(i) - RHS(j) * (A(i, j) * temp);
    }
  }
  // Apply permutations JPIV to the solution (RHS)
  Laswp(1, rhs, lda, 1, n - 1, jpiv, -1);
}

// xLATDF (real T): computes the contribution of the solution of Z * x = b
// (Z of order N <= 8 factored by xGETC2, IPIV, JPIV) to the reciprocal
// Dif-estimate by solving with a right-hand side RHS of +-1 entries chosen
// to make the solution large (IJOB = 2: an approximation from xGECON's
// estimate of the null vector; otherwise a look-ahead strategy), and
// updates the sum of squares RDSCAL**2 * RDSUM of the solutions found so
// far (xLASSQ). RHS returns the solution.
template <typename R>
void LatdfReal(blasint ijob, blasint n, const R* z, blasint ldz, R* rhs,
               R* rdsum, R* rdscal, const blasint* ipiv, const blasint* jpiv) {
  const blasint maxdim = 8;
  const R zero = R(0);
  const R one = R(1);
  blasint iwork[maxdim];
  const Vector<R> RHS(rhs);
  R work[4 * maxdim];
  const Vector<R> WORK(work);
  R xm[maxdim];
  R xp[maxdim];
  const Vector<R> XP(xp);
  const Matrix<const R> Z(z, ldz);
  blasint i, j, k;
  R bm, bp, pmone, sminu, splus, temp;
  if (ijob != 2) {
    // Apply permutations IPIV to RHS
    Laswp(1, rhs, ldz, 1, n - 1, ipiv, 1);
    // Solve for L-part choosing RHS either to +1 or -1.
    pmone = -one;
    for (j = 1; j <= n - 1; ++j) {
      bp = RHS(j) + one;
      bm = RHS(j) - one;
      splus = one;
      // Look-ahead for L-part RHS(1:N-1) = + or -1, SPLUS and
      // SMIN computed more efficiently than in BSOLVE [1].
      splus = splus + Dotu(n - j, Z.ptr(j + 1, j), 1, Z.ptr(j + 1, j), 1);
      sminu = Dotu(n - j, Z.ptr(j + 1, j), 1, RHS.ptr(j + 1), 1);
      splus = splus * RHS(j);
      if (splus > sminu) {
        RHS(j) = bp;
      } else if (sminu > splus) {
        RHS(j) = bm;
      } else {
        // In this case the updating sums are equal and we can
        // choose RHS(J) +1 or -1. The first time this happens
        // we choose -1, thereafter +1. This is a simple way to
        // get good estimates of matrices like Byers well-known
        // example (see [1]). (Not done in BSOLVE.)
        RHS(j) = RHS(j) + pmone;
        pmone = one;
      }
      // Compute the remaining r.h.s.
      temp = -RHS(j);
      Axpy(n - j, temp, Z.ptr(j + 1, j), 1, RHS.ptr(j + 1), 1);
    }
    // Solve for U-part, look-ahead for RHS(N) = +-1. This is not done
    // in BSOLVE and will hopefully give us a better estimate because
    // any ill-conditioning of the original matrix is transferred to U
    // and not to L. U(N, N) is an approximation to sigma_min(LU).
    Copy(n - 1, rhs, 1, xp, 1);
    XP(n) = RHS(n) + one;
    RHS(n) = RHS(n) - one;
    splus = zero;
    sminu = zero;
    for (i = n; i >= 1; --i) {
      temp = one / Z(i, i);
      XP(i) = XP(i) * temp;
      RHS(i) = RHS(i) * temp;
      for (k = i + 1; k <= n; ++k) {
        XP(i) = XP(i) - XP(k) * (Z(i, k) * temp);
        RHS(i) = RHS(i) - RHS(k) * (Z(i, k) * temp);
      }
      splus = splus + std::abs(XP(i));
      sminu = sminu + std::abs(RHS(i));
    }
    if (splus > sminu) Copy(n, xp, 1, rhs, 1);
    // Apply the permutations JPIV to the computed solution (RHS)
    Laswp(1, rhs, ldz, 1, n - 1, jpiv, -1);
    // Compute the sum of squares
    Lassq(n, rhs, 1, rdscal, rdsum);
  } else {
    // IJOB = 2, Compute approximate nullvector XM of Z
    // (INFO of xGECON is not used.)
    Gecon<R>('I', n, z, ldz, one, &temp, work, nullptr, iwork);
    Copy(n, WORK.ptr(n + 1), 1, xm, 1);
    // Compute RHS
    Laswp(1, xm, ldz, 1, n - 1, ipiv, -1);
    temp = one / std::sqrt(Dotu(n, xm, 1, xm, 1));
    Rscal(n, temp, xm, 1);
    Copy(n, xm, 1, xp, 1);
    Axpy(n, one, rhs, 1, xp, 1);
    Axpy(n, -one, xm, 1, rhs, 1);
    Gesc2(n, z, ldz, rhs, ipiv, jpiv, &temp);
    Gesc2(n, z, ldz, xp, ipiv, jpiv, &temp);
    if (Asum(n, xp, 1) > Asum(n, rhs, 1)) Copy(n, xp, 1, rhs, 1);
    // Compute the sum of squares
    Lassq(n, rhs, 1, rdscal, rdsum);
  }
}

// xLATDF (complex T): as the real routine, for N <= 2 (and without the
// look-ahead at the last component). RWORK has 2*MAXDIM elements here: the
// reference ZLATDF's RWORK has MAXDIM = 2, but ZGECON, to which it is
// passed for IJOB = 2, writes 2*N of them.
template <typename T>
void LatdfComplex(blasint ijob, blasint n, const T* z, blasint ldz, T* rhs,
                  RealT<T>* rdsum, RealT<T>* rdscal, const blasint* ipiv,
                  const blasint* jpiv) {
  using R = RealT<T>;

  const blasint maxdim = 2;
  const R zero = R(0);
  const R one = R(1);
  const T cone = T(1);
  const Vector<T> RHS(rhs);
  R rwork[2 * maxdim];
  T work[4 * maxdim];
  const Vector<T> WORK(work);
  T xm[maxdim];
  T xp[maxdim];
  const Matrix<const T> Z(z, ldz);
  blasint i, j, k;
  R rtemp, scale, sminu, splus;
  T bm, bp, pmone, temp;
  if (ijob != 2) {
    // Apply permutations IPIV to RHS
    Laswp(1, rhs, ldz, 1, n - 1, ipiv, 1);
    // Solve for L-part choosing RHS either to +1 or -1.
    pmone = -cone;
    for (j = 1; j <= n - 1; ++j) {
      bp = RHS(j) + cone;
      bm = RHS(j) - cone;
      splus = one;
      // Look-ahead for L- part RHS(1:N-1) = +-1
      // SPLUS and SMIN computed more efficiently than in BSOLVE[1].
      splus = splus +
              std::real(Dotc(n - j, Z.ptr(j + 1, j), 1, Z.ptr(j + 1, j), 1));
      sminu = std::real(Dotc(n - j, Z.ptr(j + 1, j), 1, RHS.ptr(j + 1), 1));
      splus = splus * std::real(RHS(j));
      if (splus > sminu) {
        RHS(j) = bp;
      } else if (sminu > splus) {
        RHS(j) = bm;
      } else {
        // In this case the updating sums are equal and we can
        // choose RHS(J) +1 or -1. The first time this happens we
        // choose -1, thereafter +1. This is a simple way to get
        // good estimates of matrices like Byers well-known example
        // (see [1]). (Not done in BSOLVE.)
        RHS(j) = RHS(j) + pmone;
        pmone = cone;
      }
      // Compute the remaining r.h.s.
      temp = -RHS(j);
      Axpy(n - j, temp, Z.ptr(j + 1, j), 1, RHS.ptr(j + 1), 1);
    }
    // Solve for U- part, lockahead for RHS(N) = +-1. This is not done
    // In BSOLVE and will hopefully give us a better estimate because
    // any ill-conditioning of the original matrix is transferred to U
    // and not to L. U(N, N) is an approximation to sigma_min(LU).
    Copy(n - 1, rhs, 1, work, 1);
    WORK(n) = RHS(n) + cone;
    RHS(n) = RHS(n) - cone;
    splus = zero;
    sminu = zero;
    for (i = n; i >= 1; --i) {
      temp = cone / Z(i, i);
      WORK(i) = WORK(i) * temp;
      RHS(i) = RHS(i) * temp;
      for (k = i + 1; k <= n; ++k) {
        WORK(i) = WORK(i) - WORK(k) * (Z(i, k) * temp);
        RHS(i) = RHS(i) - RHS(k) * (Z(i, k) * temp);
      }
      splus = splus + std::abs(WORK(i));
      sminu = sminu + std::abs(RHS(i));
    }
    if (splus > sminu) Copy(n, work, 1, rhs, 1);
    // Apply the permutations JPIV to the computed solution (RHS)
    Laswp(1, rhs, ldz, 1, n - 1, jpiv, -1);
    // Compute the sum of squares
    Lassq(n, rhs, 1, rdscal, rdsum);
    return;
  }
  // ENTRY IJOB = 2
  // Compute approximate nullvector XM of Z
  // (INFO of xGECON is not used.)
  Gecon('I', n, z, ldz, one, &rtemp, work, rwork, nullptr);
  Copy(n, WORK.ptr(n + 1), 1, xm, 1);
  // Compute RHS
  Laswp(1, xm, ldz, 1, n - 1, ipiv, -1);
  temp = cone / std::sqrt(Dotc(n, xm, 1, xm, 1));
  Scal(n, temp, xm, 1);
  Copy(n, xm, 1, xp, 1);
  Axpy(n, cone, rhs, 1, xp, 1);
  Axpy(n, -cone, xm, 1, rhs, 1);
  Gesc2(n, z, ldz, rhs, ipiv, jpiv, &scale);
  Gesc2(n, z, ldz, xp, ipiv, jpiv, &scale);
  if (Asum(n, xp, 1) > Asum(n, rhs, 1)) Copy(n, xp, 1, rhs, 1);
  // Compute the sum of squares
  Lassq(n, rhs, 1, rdscal, rdsum);
}

template <typename T>
void Latdf(blasint ijob, blasint n, const T* z, blasint ldz, T* rhs,
           RealT<T>* rdsum, RealT<T>* rdscal, const blasint* ipiv,
           const blasint* jpiv) {
  if constexpr (IsComplex<T>::value) {
    LatdfComplex(ijob, n, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
  } else {
    LatdfReal(ijob, n, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
  }
}

// xTGSY2 (real T): solves the generalized Sylvester equation with the
// quasi-upper triangular A and D (M-by-M) and the upper triangular B and E
// (N-by-N, B quasi-upper triangular) of generalized Schur forms,
//
//     A * R - L * B = SCALE * C,
//     D * R - L * E = SCALE * F           (TRANS = 'N')
//
// or its transpose A**T * R + D**T * L = SCALE * C, R * B**T + L * E**T =
// SCALE * (-F) (TRANS = 'T'), by solving 1-by-1 and 2-by-2 subsystems (of
// order up to 8, with xGETC2 / xGESC2). R and L overwrite C and F; SCALE
// <= 1 avoids overflow. IJOB = 1, 2 (TRANS = 'N' only): also the
// contribution of the subsystems to the Dif-estimate (RDSUM, RDSCAL, see
// xLATDF; xGESC2 is then not used). IWORK has M+N+2 elements; PQ returns
// the number of subsystems solved.
//
// Returns INFO: 0, -i for an illegal argument i, or > 0 if a subsystem was
// perturbed to make it nonsingular (a solution of a slightly perturbed
// system is returned).
template <typename R>
blasint Tgsy2(char trans, blasint ijob, blasint m, blasint n, const R* a,
              blasint lda, const R* b, blasint ldb, R* c, blasint ldc,
              const R* d, blasint ldd, const R* e, blasint lde, R* f,
              blasint ldf, R* scale, R* rdsum, R* rdscal, blasint* iwork,
              blasint* pq) {
  using T = R;

  const blasint ldz = 8;
  const R zero = R(0);
  const R one = R(1);
  const Matrix<const R> A(a, lda);
  const Matrix<const R> B(b, ldb);
  const Matrix<R> C(c, ldc);
  const Matrix<const R> D(d, ldd);
  const Matrix<const R> E(e, lde);
  const Matrix<R> F(f, ldf);
  blasint ipiv[ldz];
  const Vector<blasint> IWORK(iwork);
  blasint jpiv[ldz];
  R rhs[ldz];
  const Vector<R> RHS(rhs);
  R z[ldz * ldz];
  const Matrix<R> Z(z, ldz);
  blasint i, ie, ierr, ii, is, isp1, j, je, jj, js, jsp1, k, mb, nb, p, q, zdim;
  bool notran;
  R alpha, scaloc;
  blasint info = 0;
  ierr = 0;
  notran = Lsame(trans, 'N');
  if (!notran && !Lsame(trans, 'T')) {
    info = -1;
  } else if (notran) {
    if ((ijob < 0) || (ijob > 2)) {
      info = -2;
    }
  }
  if (info == 0) {
    if (m <= 0) {
      info = -3;
    } else if (n <= 0) {
      info = -4;
    } else if (lda < Max<blasint>(1, m)) {
      info = -6;
    } else if (ldb < Max<blasint>(1, n)) {
      info = -8;
    } else if (ldc < Max<blasint>(1, m)) {
      info = -10;
    } else if (ldd < Max<blasint>(1, m)) {
      info = -12;
    } else if (lde < Max<blasint>(1, n)) {
      info = -14;
    } else if (ldf < Max<blasint>(1, m)) {
      info = -16;
    }
  }
  if (info != 0) {
    Xerbla<T>("TGSY2", -info);
    return info;
  }
  // Determine block structure of A
  *pq = 0;
  p = 0;
  i = 1;
L10:
  if (i > m) goto L20;
  p = p + 1;
  IWORK(p) = i;
  if (i == m) goto L20;
  if (A(i + 1, i) != zero) {
    i = i + 2;
  } else {
    i = i + 1;
  }
  goto L10;
L20:
  IWORK(p + 1) = m + 1;
  // Determine block structure of B
  q = p + 1;
  j = 1;
L30:
  if (j > n) goto L40;
  q = q + 1;
  IWORK(q) = j;
  if (j == n) goto L40;
  if (B(j + 1, j) != zero) {
    j = j + 2;
  } else {
    j = j + 1;
  }
  goto L30;
L40:
  IWORK(q + 1) = n + 1;
  *pq = p * (q - p - 1);
  if (notran) {
    // Solve (I, J) - subsystem
    // A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
    // D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
    // for I = P, P - 1, ..., 1; J = 1, 2, ..., Q
    *scale = one;
    scaloc = one;
    for (j = p + 2; j <= q; ++j) {
      js = IWORK(j);
      jsp1 = js + 1;
      je = IWORK(j + 1) - 1;
      nb = je - js + 1;
      for (i = p; i >= 1; --i) {
        is = IWORK(i);
        isp1 = is + 1;
        ie = IWORK(i + 1) - 1;
        mb = ie - is + 1;
        zdim = mb * nb * 2;
        if ((mb == 1) && (nb == 1)) {
          // Build a 2-by-2 system Z * x = RHS
          Z(1, 1) = A(is, is);
          Z(2, 1) = D(is, is);
          Z(1, 2) = -B(js, js);
          Z(2, 2) = -E(js, js);
          // Set up right hand side(s)
          RHS(1) = C(is, js);
          RHS(2) = F(is, js);
          // Solve Z * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          if (ijob == 0) {
            Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
            if (scaloc != one) {
              for (k = 1; k <= n; ++k) {
                Rscal(m, scaloc, C.ptr(1, k), 1);
                Rscal(m, scaloc, F.ptr(1, k), 1);
              }
              *scale = *scale * scaloc;
            }
          } else {
            Latdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
          }
          // Unpack solution vector(s)
          C(is, js) = RHS(1);
          F(is, js) = RHS(2);
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (i > 1) {
            alpha = -RHS(1);
            Axpy(is - 1, alpha, A.ptr(1, is), 1, C.ptr(1, js), 1);
            Axpy(is - 1, alpha, D.ptr(1, is), 1, F.ptr(1, js), 1);
          }
          if (j < q) {
            Axpy(n - je, RHS(2), B.ptr(js, je + 1), ldb, C.ptr(is, je + 1),
                 ldc);
            Axpy(n - je, RHS(2), E.ptr(js, je + 1), lde, F.ptr(is, je + 1),
                 ldf);
          }
        } else if ((mb == 1) && (nb == 2)) {
          // Build a 4-by-4 system Z * x = RHS
          Z(1, 1) = A(is, is);
          Z(2, 1) = zero;
          Z(3, 1) = D(is, is);
          Z(4, 1) = zero;
          Z(1, 2) = zero;
          Z(2, 2) = A(is, is);
          Z(3, 2) = zero;
          Z(4, 2) = D(is, is);
          Z(1, 3) = -B(js, js);
          Z(2, 3) = -B(js, jsp1);
          Z(3, 3) = -E(js, js);
          Z(4, 3) = -E(js, jsp1);
          Z(1, 4) = -B(jsp1, js);
          Z(2, 4) = -B(jsp1, jsp1);
          Z(3, 4) = zero;
          Z(4, 4) = -E(jsp1, jsp1);
          // Set up right hand side(s)
          RHS(1) = C(is, js);
          RHS(2) = C(is, jsp1);
          RHS(3) = F(is, js);
          RHS(4) = F(is, jsp1);
          // Solve Z * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          if (ijob == 0) {
            Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
            if (scaloc != one) {
              for (k = 1; k <= n; ++k) {
                Rscal(m, scaloc, C.ptr(1, k), 1);
                Rscal(m, scaloc, F.ptr(1, k), 1);
              }
              *scale = *scale * scaloc;
            }
          } else {
            Latdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
          }
          // Unpack solution vector(s)
          C(is, js) = RHS(1);
          C(is, jsp1) = RHS(2);
          F(is, js) = RHS(3);
          F(is, jsp1) = RHS(4);
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (i > 1) {
            Geru(is - 1, nb, -one, A.ptr(1, is), 1, RHS.ptr(1), 1, C.ptr(1, js),
                 ldc);
            Geru(is - 1, nb, -one, D.ptr(1, is), 1, RHS.ptr(1), 1, F.ptr(1, js),
                 ldf);
          }
          if (j < q) {
            Axpy(n - je, RHS(3), B.ptr(js, je + 1), ldb, C.ptr(is, je + 1),
                 ldc);
            Axpy(n - je, RHS(3), E.ptr(js, je + 1), lde, F.ptr(is, je + 1),
                 ldf);
            Axpy(n - je, RHS(4), B.ptr(jsp1, je + 1), ldb, C.ptr(is, je + 1),
                 ldc);
            Axpy(n - je, RHS(4), E.ptr(jsp1, je + 1), lde, F.ptr(is, je + 1),
                 ldf);
          }
        } else if ((mb == 2) && (nb == 1)) {
          // Build a 4-by-4 system Z * x = RHS
          Z(1, 1) = A(is, is);
          Z(2, 1) = A(isp1, is);
          Z(3, 1) = D(is, is);
          Z(4, 1) = zero;
          Z(1, 2) = A(is, isp1);
          Z(2, 2) = A(isp1, isp1);
          Z(3, 2) = D(is, isp1);
          Z(4, 2) = D(isp1, isp1);
          Z(1, 3) = -B(js, js);
          Z(2, 3) = zero;
          Z(3, 3) = -E(js, js);
          Z(4, 3) = zero;
          Z(1, 4) = zero;
          Z(2, 4) = -B(js, js);
          Z(3, 4) = zero;
          Z(4, 4) = -E(js, js);
          // Set up right hand side(s)
          RHS(1) = C(is, js);
          RHS(2) = C(isp1, js);
          RHS(3) = F(is, js);
          RHS(4) = F(isp1, js);
          // Solve Z * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          if (ijob == 0) {
            Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
            if (scaloc != one) {
              for (k = 1; k <= n; ++k) {
                Rscal(m, scaloc, C.ptr(1, k), 1);
                Rscal(m, scaloc, F.ptr(1, k), 1);
              }
              *scale = *scale * scaloc;
            }
          } else {
            Latdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
          }
          // Unpack solution vector(s)
          C(is, js) = RHS(1);
          C(isp1, js) = RHS(2);
          F(is, js) = RHS(3);
          F(isp1, js) = RHS(4);
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (i > 1) {
            Gemv('N', is - 1, mb, -one, A.ptr(1, is), lda, RHS.ptr(1), 1, one,
                 C.ptr(1, js), 1);
            Gemv('N', is - 1, mb, -one, D.ptr(1, is), ldd, RHS.ptr(1), 1, one,
                 F.ptr(1, js), 1);
          }
          if (j < q) {
            Geru(mb, n - je, one, RHS.ptr(3), 1, B.ptr(js, je + 1), ldb,
                 C.ptr(is, je + 1), ldc);
            Geru(mb, n - je, one, RHS.ptr(3), 1, E.ptr(js, je + 1), lde,
                 F.ptr(is, je + 1), ldf);
          }
        } else if ((mb == 2) && (nb == 2)) {
          // Build an 8-by-8 system Z * x = RHS
          Laset('F', ldz, ldz, zero, zero, z, ldz);
          Z(1, 1) = A(is, is);
          Z(2, 1) = A(isp1, is);
          Z(5, 1) = D(is, is);
          Z(1, 2) = A(is, isp1);
          Z(2, 2) = A(isp1, isp1);
          Z(5, 2) = D(is, isp1);
          Z(6, 2) = D(isp1, isp1);
          Z(3, 3) = A(is, is);
          Z(4, 3) = A(isp1, is);
          Z(7, 3) = D(is, is);
          Z(3, 4) = A(is, isp1);
          Z(4, 4) = A(isp1, isp1);
          Z(7, 4) = D(is, isp1);
          Z(8, 4) = D(isp1, isp1);
          Z(1, 5) = -B(js, js);
          Z(3, 5) = -B(js, jsp1);
          Z(5, 5) = -E(js, js);
          Z(7, 5) = -E(js, jsp1);
          Z(2, 6) = -B(js, js);
          Z(4, 6) = -B(js, jsp1);
          Z(6, 6) = -E(js, js);
          Z(8, 6) = -E(js, jsp1);
          Z(1, 7) = -B(jsp1, js);
          Z(3, 7) = -B(jsp1, jsp1);
          Z(7, 7) = -E(jsp1, jsp1);
          Z(2, 8) = -B(jsp1, js);
          Z(4, 8) = -B(jsp1, jsp1);
          Z(8, 8) = -E(jsp1, jsp1);
          // Set up right hand side(s)
          k = 1;
          ii = mb * nb + 1;
          for (jj = 0; jj <= nb - 1; ++jj) {
            Copy(mb, C.ptr(is, js + jj), 1, RHS.ptr(k), 1);
            Copy(mb, F.ptr(is, js + jj), 1, RHS.ptr(ii), 1);
            k = k + mb;
            ii = ii + mb;
          }
          // Solve Z * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          if (ijob == 0) {
            Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
            if (scaloc != one) {
              for (k = 1; k <= n; ++k) {
                Rscal(m, scaloc, C.ptr(1, k), 1);
                Rscal(m, scaloc, F.ptr(1, k), 1);
              }
              *scale = *scale * scaloc;
            }
          } else {
            Latdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
          }
          // Unpack solution vector(s)
          k = 1;
          ii = mb * nb + 1;
          for (jj = 0; jj <= nb - 1; ++jj) {
            Copy(mb, RHS.ptr(k), 1, C.ptr(is, js + jj), 1);
            Copy(mb, RHS.ptr(ii), 1, F.ptr(is, js + jj), 1);
            k = k + mb;
            ii = ii + mb;
          }
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (i > 1) {
            Gemm('N', 'N', is - 1, nb, mb, -one, A.ptr(1, is), lda, RHS.ptr(1),
                 mb, one, C.ptr(1, js), ldc);
            Gemm('N', 'N', is - 1, nb, mb, -one, D.ptr(1, is), ldd, RHS.ptr(1),
                 mb, one, F.ptr(1, js), ldf);
          }
          if (j < q) {
            k = mb * nb + 1;
            Gemm('N', 'N', mb, n - je, nb, one, RHS.ptr(k), mb,
                 B.ptr(js, je + 1), ldb, one, C.ptr(is, je + 1), ldc);
            Gemm('N', 'N', mb, n - je, nb, one, RHS.ptr(k), mb,
                 E.ptr(js, je + 1), lde, one, F.ptr(is, je + 1), ldf);
          }
        }
      }
    }
  } else {
    // Solve (I, J) - subsystem
    // A(I, I)**T * R(I, J) + D(I, I)**T * L(J, J)  =  C(I, J)
    // R(I, I)  * B(J, J) + L(I, J)  * E(J, J)  = -F(I, J)
    // for I = 1, 2, ..., P, J = Q, Q - 1, ..., 1
    *scale = one;
    scaloc = one;
    for (i = 1; i <= p; ++i) {
      is = IWORK(i);
      isp1 = is + 1;
      ie = IWORK(i + 1) - 1;
      mb = ie - is + 1;
      for (j = q; j >= p + 2; --j) {
        js = IWORK(j);
        jsp1 = js + 1;
        je = IWORK(j + 1) - 1;
        nb = je - js + 1;
        zdim = mb * nb * 2;
        if ((mb == 1) && (nb == 1)) {
          // Build a 2-by-2 system Z**T * x = RHS
          Z(1, 1) = A(is, is);
          Z(2, 1) = -B(js, js);
          Z(1, 2) = D(is, is);
          Z(2, 2) = -E(js, js);
          // Set up right hand side(s)
          RHS(1) = C(is, js);
          RHS(2) = F(is, js);
          // Solve Z**T * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
          if (scaloc != one) {
            for (k = 1; k <= n; ++k) {
              Rscal(m, scaloc, C.ptr(1, k), 1);
              Rscal(m, scaloc, F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
          // Unpack solution vector(s)
          C(is, js) = RHS(1);
          F(is, js) = RHS(2);
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (j > p + 2) {
            alpha = RHS(1);
            Axpy(js - 1, alpha, B.ptr(1, js), 1, F.ptr(is, 1), ldf);
            alpha = RHS(2);
            Axpy(js - 1, alpha, E.ptr(1, js), 1, F.ptr(is, 1), ldf);
          }
          if (i < p) {
            alpha = -RHS(1);
            Axpy(m - ie, alpha, A.ptr(is, ie + 1), lda, C.ptr(ie + 1, js), 1);
            alpha = -RHS(2);
            Axpy(m - ie, alpha, D.ptr(is, ie + 1), ldd, C.ptr(ie + 1, js), 1);
          }
        } else if ((mb == 1) && (nb == 2)) {
          // Build a 4-by-4 system Z**T * x = RHS
          Z(1, 1) = A(is, is);
          Z(2, 1) = zero;
          Z(3, 1) = -B(js, js);
          Z(4, 1) = -B(jsp1, js);
          Z(1, 2) = zero;
          Z(2, 2) = A(is, is);
          Z(3, 2) = -B(js, jsp1);
          Z(4, 2) = -B(jsp1, jsp1);
          Z(1, 3) = D(is, is);
          Z(2, 3) = zero;
          Z(3, 3) = -E(js, js);
          Z(4, 3) = zero;
          Z(1, 4) = zero;
          Z(2, 4) = D(is, is);
          Z(3, 4) = -E(js, jsp1);
          Z(4, 4) = -E(jsp1, jsp1);
          // Set up right hand side(s)
          RHS(1) = C(is, js);
          RHS(2) = C(is, jsp1);
          RHS(3) = F(is, js);
          RHS(4) = F(is, jsp1);
          // Solve Z**T * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
          if (scaloc != one) {
            for (k = 1; k <= n; ++k) {
              Rscal(m, scaloc, C.ptr(1, k), 1);
              Rscal(m, scaloc, F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
          // Unpack solution vector(s)
          C(is, js) = RHS(1);
          C(is, jsp1) = RHS(2);
          F(is, js) = RHS(3);
          F(is, jsp1) = RHS(4);
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (j > p + 2) {
            Axpy(js - 1, RHS(1), B.ptr(1, js), 1, F.ptr(is, 1), ldf);
            Axpy(js - 1, RHS(2), B.ptr(1, jsp1), 1, F.ptr(is, 1), ldf);
            Axpy(js - 1, RHS(3), E.ptr(1, js), 1, F.ptr(is, 1), ldf);
            Axpy(js - 1, RHS(4), E.ptr(1, jsp1), 1, F.ptr(is, 1), ldf);
          }
          if (i < p) {
            Geru(m - ie, nb, -one, A.ptr(is, ie + 1), lda, RHS.ptr(1), 1,
                 C.ptr(ie + 1, js), ldc);
            Geru(m - ie, nb, -one, D.ptr(is, ie + 1), ldd, RHS.ptr(3), 1,
                 C.ptr(ie + 1, js), ldc);
          }
        } else if ((mb == 2) && (nb == 1)) {
          // Build a 4-by-4 system Z**T * x = RHS
          Z(1, 1) = A(is, is);
          Z(2, 1) = A(is, isp1);
          Z(3, 1) = -B(js, js);
          Z(4, 1) = zero;
          Z(1, 2) = A(isp1, is);
          Z(2, 2) = A(isp1, isp1);
          Z(3, 2) = zero;
          Z(4, 2) = -B(js, js);
          Z(1, 3) = D(is, is);
          Z(2, 3) = D(is, isp1);
          Z(3, 3) = -E(js, js);
          Z(4, 3) = zero;
          Z(1, 4) = zero;
          Z(2, 4) = D(isp1, isp1);
          Z(3, 4) = zero;
          Z(4, 4) = -E(js, js);
          // Set up right hand side(s)
          RHS(1) = C(is, js);
          RHS(2) = C(isp1, js);
          RHS(3) = F(is, js);
          RHS(4) = F(isp1, js);
          // Solve Z**T * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
          if (scaloc != one) {
            for (k = 1; k <= n; ++k) {
              Rscal(m, scaloc, C.ptr(1, k), 1);
              Rscal(m, scaloc, F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
          // Unpack solution vector(s)
          C(is, js) = RHS(1);
          C(isp1, js) = RHS(2);
          F(is, js) = RHS(3);
          F(isp1, js) = RHS(4);
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (j > p + 2) {
            Geru(mb, js - 1, one, RHS.ptr(1), 1, B.ptr(1, js), 1, F.ptr(is, 1),
                 ldf);
            Geru(mb, js - 1, one, RHS.ptr(3), 1, E.ptr(1, js), 1, F.ptr(is, 1),
                 ldf);
          }
          if (i < p) {
            Gemv('T', mb, m - ie, -one, A.ptr(is, ie + 1), lda, RHS.ptr(1), 1,
                 one, C.ptr(ie + 1, js), 1);
            Gemv('T', mb, m - ie, -one, D.ptr(is, ie + 1), ldd, RHS.ptr(3), 1,
                 one, C.ptr(ie + 1, js), 1);
          }
        } else if ((mb == 2) && (nb == 2)) {
          // Build an 8-by-8 system Z**T * x = RHS
          Laset('F', ldz, ldz, zero, zero, z, ldz);
          Z(1, 1) = A(is, is);
          Z(2, 1) = A(is, isp1);
          Z(5, 1) = -B(js, js);
          Z(7, 1) = -B(jsp1, js);
          Z(1, 2) = A(isp1, is);
          Z(2, 2) = A(isp1, isp1);
          Z(6, 2) = -B(js, js);
          Z(8, 2) = -B(jsp1, js);
          Z(3, 3) = A(is, is);
          Z(4, 3) = A(is, isp1);
          Z(5, 3) = -B(js, jsp1);
          Z(7, 3) = -B(jsp1, jsp1);
          Z(3, 4) = A(isp1, is);
          Z(4, 4) = A(isp1, isp1);
          Z(6, 4) = -B(js, jsp1);
          Z(8, 4) = -B(jsp1, jsp1);
          Z(1, 5) = D(is, is);
          Z(2, 5) = D(is, isp1);
          Z(5, 5) = -E(js, js);
          Z(2, 6) = D(isp1, isp1);
          Z(6, 6) = -E(js, js);
          Z(3, 7) = D(is, is);
          Z(4, 7) = D(is, isp1);
          Z(5, 7) = -E(js, jsp1);
          Z(7, 7) = -E(jsp1, jsp1);
          Z(4, 8) = D(isp1, isp1);
          Z(6, 8) = -E(js, jsp1);
          Z(8, 8) = -E(jsp1, jsp1);
          // Set up right hand side(s)
          k = 1;
          ii = mb * nb + 1;
          for (jj = 0; jj <= nb - 1; ++jj) {
            Copy(mb, C.ptr(is, js + jj), 1, RHS.ptr(k), 1);
            Copy(mb, F.ptr(is, js + jj), 1, RHS.ptr(ii), 1);
            k = k + mb;
            ii = ii + mb;
          }
          // Solve Z**T * x = RHS
          ierr = Getc2(zdim, z, ldz, ipiv, jpiv);
          if (ierr > 0) info = ierr;
          Gesc2(zdim, z, ldz, rhs, ipiv, jpiv, &scaloc);
          if (scaloc != one) {
            for (k = 1; k <= n; ++k) {
              Rscal(m, scaloc, C.ptr(1, k), 1);
              Rscal(m, scaloc, F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
          // Unpack solution vector(s)
          k = 1;
          ii = mb * nb + 1;
          for (jj = 0; jj <= nb - 1; ++jj) {
            Copy(mb, RHS.ptr(k), 1, C.ptr(is, js + jj), 1);
            Copy(mb, RHS.ptr(ii), 1, F.ptr(is, js + jj), 1);
            k = k + mb;
            ii = ii + mb;
          }
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (j > p + 2) {
            Gemm('N', 'T', mb, js - 1, nb, one, C.ptr(is, js), ldc,
                 B.ptr(1, js), ldb, one, F.ptr(is, 1), ldf);
            Gemm('N', 'T', mb, js - 1, nb, one, F.ptr(is, js), ldf,
                 E.ptr(1, js), lde, one, F.ptr(is, 1), ldf);
          }
          if (i < p) {
            Gemm('T', 'N', m - ie, nb, mb, -one, A.ptr(is, ie + 1), lda,
                 C.ptr(is, js), ldc, one, C.ptr(ie + 1, js), ldc);
            Gemm('T', 'N', m - ie, nb, mb, -one, D.ptr(is, ie + 1), ldd,
                 F.ptr(is, js), ldf, one, C.ptr(ie + 1, js), ldc);
          }
        }
      }
    }
  }
  return info;
}

// xTGSY2 (complex T): as the real routine with upper triangular A, B, D, E
// (1-by-1 subsystems, of order 2) and TRANS = 'C' for the conjugate
// transposed equation.
template <typename T>
blasint Tgsy2(char trans, blasint ijob, blasint m, blasint n, const T* a,
              blasint lda, const T* b, blasint ldb, T* c, blasint ldc,
              const T* d, blasint ldd, const T* e, blasint lde, T* f,
              blasint ldf, RealT<T>* scale, RealT<T>* rdsum, RealT<T>* rdscal) {
  using R = RealT<T>;

  const R one = R(1);
  const blasint ldz = 2;
  const Matrix<const T> A(a, lda);
  const Matrix<const T> B(b, ldb);
  const Matrix<T> C(c, ldc);
  const Matrix<const T> D(d, ldd);
  const Matrix<const T> E(e, lde);
  const Matrix<T> F(f, ldf);
  blasint ipiv[ldz];
  blasint jpiv[ldz];
  T rhs[ldz];
  const Vector<T> RHS(rhs);
  T z[ldz * ldz];
  const Matrix<T> Z(z, ldz);
  blasint i, ierr, j, k;
  bool notran;
  R scaloc;
  T alpha;
  blasint info = 0;
  ierr = 0;
  notran = Lsame(trans, 'N');
  if (!notran && !Lsame(trans, 'C')) {
    info = -1;
  } else if (notran) {
    if ((ijob < 0) || (ijob > 2)) {
      info = -2;
    }
  }
  if (info == 0) {
    if (m <= 0) {
      info = -3;
    } else if (n <= 0) {
      info = -4;
    } else if (lda < Max<blasint>(1, m)) {
      info = -6;
    } else if (ldb < Max<blasint>(1, n)) {
      info = -8;
    } else if (ldc < Max<blasint>(1, m)) {
      info = -10;
    } else if (ldd < Max<blasint>(1, m)) {
      info = -12;
    } else if (lde < Max<blasint>(1, n)) {
      info = -14;
    } else if (ldf < Max<blasint>(1, m)) {
      info = -16;
    }
  }
  if (info != 0) {
    Xerbla<T>("TGSY2", -info);
    return info;
  }
  if (notran) {
    // Solve (I, J) - system
    // A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
    // D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
    // for I = M, M - 1, ..., 1; J = 1, 2, ..., N
    *scale = one;
    scaloc = one;
    for (j = 1; j <= n; ++j) {
      for (i = m; i >= 1; --i) {
        // Build 2 by 2 system
        Z(1, 1) = A(i, i);
        Z(2, 1) = D(i, i);
        Z(1, 2) = -B(j, j);
        Z(2, 2) = -E(j, j);
        // Set up right hand side(s)
        RHS(1) = C(i, j);
        RHS(2) = F(i, j);
        // Solve Z * x = RHS
        ierr = Getc2(ldz, z, ldz, ipiv, jpiv);
        if (ierr > 0) info = ierr;
        if (ijob == 0) {
          Gesc2(ldz, z, ldz, rhs, ipiv, jpiv, &scaloc);
          if (scaloc != one) {
            for (k = 1; k <= n; ++k) {
              Scal(m, T(scaloc), C.ptr(1, k), 1);
              Scal(m, T(scaloc), F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
        } else {
          Latdf(ijob, ldz, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv);
        }
        // Unpack solution vector(s)
        C(i, j) = RHS(1);
        F(i, j) = RHS(2);
        // Substitute R(I, J) and L(I, J) into remaining equation.
        if (i > 1) {
          alpha = -RHS(1);
          Axpy(i - 1, alpha, A.ptr(1, i), 1, C.ptr(1, j), 1);
          Axpy(i - 1, alpha, D.ptr(1, i), 1, F.ptr(1, j), 1);
        }
        if (j < n) {
          Axpy(n - j, RHS(2), B.ptr(j, j + 1), ldb, C.ptr(i, j + 1), ldc);
          Axpy(n - j, RHS(2), E.ptr(j, j + 1), lde, F.ptr(i, j + 1), ldf);
        }
      }
    }
  } else {
    // Solve transposed (I, J) - system:
    // A(I, I)**H * R(I, J) + D(I, I)**H * L(J, J) = C(I, J)
    // R(I, I) * B(J, J) + L(I, J) * E(J, J)   = -F(I, J)
    // for I = 1, 2, ..., M, J = N, N - 1, ..., 1
    *scale = one;
    scaloc = one;
    for (i = 1; i <= m; ++i) {
      for (j = n; j >= 1; --j) {
        // Build 2 by 2 system Z**H
        Z(1, 1) = Conj(A(i, i));
        Z(2, 1) = -Conj(B(j, j));
        Z(1, 2) = Conj(D(i, i));
        Z(2, 2) = -Conj(E(j, j));
        // Set up right hand side(s)
        RHS(1) = C(i, j);
        RHS(2) = F(i, j);
        // Solve Z**H * x = RHS
        ierr = Getc2(ldz, z, ldz, ipiv, jpiv);
        if (ierr > 0) info = ierr;
        Gesc2(ldz, z, ldz, rhs, ipiv, jpiv, &scaloc);
        if (scaloc != one) {
          for (k = 1; k <= n; ++k) {
            Scal(m, T(scaloc), C.ptr(1, k), 1);
            Scal(m, T(scaloc), F.ptr(1, k), 1);
          }
          *scale = *scale * scaloc;
        }
        // Unpack solution vector(s)
        C(i, j) = RHS(1);
        F(i, j) = RHS(2);
        // Substitute R(I, J) and L(I, J) into remaining equation.
        for (k = 1; k <= j - 1; ++k) {
          F(i, k) = F(i, k) + RHS(1) * Conj(B(k, j)) + RHS(2) * Conj(E(k, j));
        }
        for (k = i + 1; k <= m; ++k) {
          C(k, j) = C(k, j) - Conj(A(i, k)) * RHS(1) - Conj(D(i, k)) * RHS(2);
        }
      }
    }
  }
  return info;
}

// xTGSYL (real T): solves the generalized Sylvester equation of xTGSY2
// (TRANS = 'N' or 'T') with a blocked algorithm (block sizes from ILAENV,
// the blocks solved by xTGSY2, the updates with xGEMM). IJOB = 0: solve
// only; 1, 2: also the Dif-estimate (DIF, the Frobenius-norm-based
// estimate of 1 / Dif[(A,D), (B,E)]; 1: look-ahead strategy, 2: xGECON's
// estimate in xLATDF), solving the equation twice; 3, 4: only the
// Dif-estimate (as 1, 2), C and F are not referenced. WORK has LWORK >=
// max(1, 2*M*N) elements for IJOB = 1, 2 (TRANS = 'N'), 1 otherwise
// (LWORK = -1: workspace query); IWORK has M+N+6 elements.
//
// Returns INFO: 0, -i for an illegal argument i, or > 0 if (A, D) and (B,
// E) have common or close eigenvalues (perturbed values were used).
template <typename R>
blasint TgsylReal(char trans, blasint ijob, blasint m, blasint n, const R* a,
                  blasint lda, const R* b, blasint ldb, R* c, blasint ldc,
                  const R* d, blasint ldd, const R* e, blasint lde, R* f,
                  blasint ldf, R* scale, R* dif, R* work, blasint lwork,
                  blasint* iwork) {
  using T = R;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<const R> A(a, lda);
  const Matrix<const R> B(b, ldb);
  const Matrix<R> C(c, ldc);
  const Matrix<const R> D(d, ldd);
  const Matrix<const R> E(e, lde);
  const Matrix<R> F(f, ldf);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> WORK(work);
  blasint i, ie, ifunc, iround, is, isolve, j, je, js, k, linfo, lwmin, mb, nb,
      p, ppqq, pq, q;
  bool lquery, notran;
  R dscale, dsum, scale2, scaloc;
  blasint info = 0;
  notran = Lsame(trans, 'N');
  lquery = (lwork == -1);
  if (!notran && !Lsame(trans, 'T')) {
    info = -1;
  } else if (notran) {
    if ((ijob < 0) || (ijob > 4)) {
      info = -2;
    }
  }
  if (info == 0) {
    if (m <= 0) {
      info = -3;
    } else if (n <= 0) {
      info = -4;
    } else if (lda < Max<blasint>(1, m)) {
      info = -6;
    } else if (ldb < Max<blasint>(1, n)) {
      info = -8;
    } else if (ldc < Max<blasint>(1, m)) {
      info = -10;
    } else if (ldd < Max<blasint>(1, m)) {
      info = -12;
    } else if (lde < Max<blasint>(1, n)) {
      info = -14;
    } else if (ldf < Max<blasint>(1, m)) {
      info = -16;
    }
  }
  if (info == 0) {
    if (notran) {
      if (ijob == 1 || ijob == 2) {
        lwmin = Max<blasint>(1, 2 * m * n);
      } else {
        lwmin = 1;
      }
    } else {
      lwmin = 1;
    }
    WORK(1) = WorkSize<T>(lwmin);
    if (lwork < lwmin && !lquery) {
      info = -20;
    }
  }
  if (info != 0) {
    Xerbla<T>("TGSYL", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (m == 0 || n == 0) {
    *scale = 1;
    if (notran) {
      if (ijob != 0) {
        *dif = 0;
      }
    }
    return info;
  }
  // Determine optimal block sizes MB and NB
  const char opts[2] = {trans, '\0'};
  mb = Ilaenv<T>(2, "TGSYL", opts, m, n, -1, -1);
  nb = Ilaenv<T>(5, "TGSYL", opts, m, n, -1, -1);
  isolve = 1;
  ifunc = 0;
  if (notran) {
    if (ijob >= 3) {
      ifunc = ijob - 2;
      Laset('F', m, n, zero, zero, c, ldc);
      Laset('F', m, n, zero, zero, f, ldf);
    } else if (ijob >= 1) {
      isolve = 2;
    }
  }
  if ((mb <= 1 && nb <= 1) || (mb >= m && nb >= n)) {
    for (iround = 1; iround <= isolve; ++iround) {
      // Use unblocked Level 2 solver
      dscale = zero;
      dsum = one;
      pq = 0;
      info = Tgsy2(trans, ifunc, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde,
                   f, ldf, scale, &dsum, &dscale, iwork, &pq);
      if (dscale != zero) {
        if (ijob == 1 || ijob == 3) {
          *dif = std::sqrt(R(2 * m * n)) / (dscale * std::sqrt(dsum));
        } else {
          *dif = std::sqrt(R(pq)) / (dscale * std::sqrt(dsum));
        }
      }
      if (isolve == 2 && iround == 1) {
        // NOLINTNEXTLINE(bugprone-redundant-branch-condition): as LAPACK
        if (notran) {
          ifunc = ijob;
        }
        scale2 = *scale;
        Lacpy('F', m, n, c, ldc, work, m);
        Lacpy('F', m, n, f, ldf, WORK.ptr(m * n + 1), m);
        Laset('F', m, n, zero, zero, c, ldc);
        Laset('F', m, n, zero, zero, f, ldf);
      } else if (isolve == 2 && iround == 2) {
        Lacpy('F', m, n, work, m, c, ldc);
        Lacpy('F', m, n, WORK.ptr(m * n + 1), m, f, ldf);
        *scale = scale2;
      }
    }
    return info;
  }
  // Determine block structure of A
  p = 0;
  i = 1;
L40:
  if (i > m) goto L50;
  p = p + 1;
  IWORK(p) = i;
  i = i + mb;
  if (i >= m) goto L50;
  if (A(i, i - 1) != zero) i = i + 1;
  goto L40;
L50:
  IWORK(p + 1) = m + 1;
  if (IWORK(p) == IWORK(p + 1)) p = p - 1;
  // Determine block structure of B
  q = p + 1;
  j = 1;
L60:
  if (j > n) goto L70;
  q = q + 1;
  IWORK(q) = j;
  j = j + nb;
  if (j >= n) goto L70;
  if (B(j, j - 1) != zero) j = j + 1;
  goto L60;
L70:
  IWORK(q + 1) = n + 1;
  if (IWORK(q) == IWORK(q + 1)) q = q - 1;
  if (notran) {
    for (iround = 1; iround <= isolve; ++iround) {
      // Solve (I, J)-subsystem
      // A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
      // D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
      // for I = P, P - 1,..., 1; J = 1, 2,..., Q
      dscale = zero;
      dsum = one;
      pq = 0;
      *scale = one;
      for (j = p + 2; j <= q; ++j) {
        js = IWORK(j);
        je = IWORK(j + 1) - 1;
        nb = je - js + 1;
        for (i = p; i >= 1; --i) {
          is = IWORK(i);
          ie = IWORK(i + 1) - 1;
          mb = ie - is + 1;
          ppqq = 0;
          linfo = Tgsy2(trans, ifunc, mb, nb, A.ptr(is, is), lda, B.ptr(js, js),
                        ldb, C.ptr(is, js), ldc, D.ptr(is, is), ldd,
                        E.ptr(js, js), lde, F.ptr(is, js), ldf, &scaloc, &dsum,
                        &dscale, IWORK.ptr(q + 2), &ppqq);
          if (linfo > 0) info = linfo;
          pq = pq + ppqq;
          if (scaloc != one) {
            for (k = 1; k <= js - 1; ++k) {
              Rscal(m, scaloc, C.ptr(1, k), 1);
              Rscal(m, scaloc, F.ptr(1, k), 1);
            }
            for (k = js; k <= je; ++k) {
              Rscal(is - 1, scaloc, C.ptr(1, k), 1);
              Rscal(is - 1, scaloc, F.ptr(1, k), 1);
            }
            for (k = js; k <= je; ++k) {
              Rscal(m - ie, scaloc, C.ptr(ie + 1, k), 1);
              Rscal(m - ie, scaloc, F.ptr(ie + 1, k), 1);
            }
            for (k = je + 1; k <= n; ++k) {
              Rscal(m, scaloc, C.ptr(1, k), 1);
              Rscal(m, scaloc, F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
          // Substitute R(I, J) and L(I, J) into remaining
          // equation.
          if (i > 1) {
            Gemm('N', 'N', is - 1, nb, mb, -one, A.ptr(1, is), lda,
                 C.ptr(is, js), ldc, one, C.ptr(1, js), ldc);
            Gemm('N', 'N', is - 1, nb, mb, -one, D.ptr(1, is), ldd,
                 C.ptr(is, js), ldc, one, F.ptr(1, js), ldf);
          }
          if (j < q) {
            Gemm('N', 'N', mb, n - je, nb, one, F.ptr(is, js), ldf,
                 B.ptr(js, je + 1), ldb, one, C.ptr(is, je + 1), ldc);
            Gemm('N', 'N', mb, n - je, nb, one, F.ptr(is, js), ldf,
                 E.ptr(js, je + 1), lde, one, F.ptr(is, je + 1), ldf);
          }
        }
      }
      if (dscale != zero) {
        if (ijob == 1 || ijob == 3) {
          *dif = std::sqrt(R(2 * m * n)) / (dscale * std::sqrt(dsum));
        } else {
          *dif = std::sqrt(R(pq)) / (dscale * std::sqrt(dsum));
        }
      }
      if (isolve == 2 && iround == 1) {
        // NOLINTNEXTLINE(bugprone-redundant-branch-condition): as LAPACK
        if (notran) {
          ifunc = ijob;
        }
        scale2 = *scale;
        Lacpy('F', m, n, c, ldc, work, m);
        Lacpy('F', m, n, f, ldf, WORK.ptr(m * n + 1), m);
        Laset('F', m, n, zero, zero, c, ldc);
        Laset('F', m, n, zero, zero, f, ldf);
      } else if (isolve == 2 && iround == 2) {
        Lacpy('F', m, n, work, m, c, ldc);
        Lacpy('F', m, n, WORK.ptr(m * n + 1), m, f, ldf);
        *scale = scale2;
      }
    }
  } else {
    // Solve transposed (I, J)-subsystem
    // A(I, I)**T * R(I, J)  + D(I, I)**T * L(I, J)  =  C(I, J)
    // R(I, J)  * B(J, J)**T + L(I, J)  * E(J, J)**T = -F(I, J)
    // for I = 1,2,..., P; J = Q, Q-1,..., 1
    *scale = one;
    for (i = 1; i <= p; ++i) {
      is = IWORK(i);
      ie = IWORK(i + 1) - 1;
      mb = ie - is + 1;
      for (j = q; j >= p + 2; --j) {
        js = IWORK(j);
        je = IWORK(j + 1) - 1;
        nb = je - js + 1;
        linfo = Tgsy2(trans, ifunc, mb, nb, A.ptr(is, is), lda, B.ptr(js, js),
                      ldb, C.ptr(is, js), ldc, D.ptr(is, is), ldd,
                      E.ptr(js, js), lde, F.ptr(is, js), ldf, &scaloc, &dsum,
                      &dscale, IWORK.ptr(q + 2), &ppqq);
        if (linfo > 0) info = linfo;
        if (scaloc != one) {
          for (k = 1; k <= js - 1; ++k) {
            Rscal(m, scaloc, C.ptr(1, k), 1);
            Rscal(m, scaloc, F.ptr(1, k), 1);
          }
          for (k = js; k <= je; ++k) {
            Rscal(is - 1, scaloc, C.ptr(1, k), 1);
            Rscal(is - 1, scaloc, F.ptr(1, k), 1);
          }
          for (k = js; k <= je; ++k) {
            Rscal(m - ie, scaloc, C.ptr(ie + 1, k), 1);
            Rscal(m - ie, scaloc, F.ptr(ie + 1, k), 1);
          }
          for (k = je + 1; k <= n; ++k) {
            Rscal(m, scaloc, C.ptr(1, k), 1);
            Rscal(m, scaloc, F.ptr(1, k), 1);
          }
          *scale = *scale * scaloc;
        }
        // Substitute R(I, J) and L(I, J) into remaining equation.
        if (j > p + 2) {
          Gemm('N', 'T', mb, js - 1, nb, one, C.ptr(is, js), ldc, B.ptr(1, js),
               ldb, one, F.ptr(is, 1), ldf);
          Gemm('N', 'T', mb, js - 1, nb, one, F.ptr(is, js), ldf, E.ptr(1, js),
               lde, one, F.ptr(is, 1), ldf);
        }
        if (i < p) {
          Gemm('T', 'N', m - ie, nb, mb, -one, A.ptr(is, ie + 1), lda,
               C.ptr(is, js), ldc, one, C.ptr(ie + 1, js), ldc);
          Gemm('T', 'N', m - ie, nb, mb, -one, D.ptr(is, ie + 1), ldd,
               F.ptr(is, js), ldf, one, C.ptr(ie + 1, js), ldc);
        }
      }
    }
  }
  WORK(1) = WorkSize<T>(lwmin);
  return info;
}

// xTGSYL (complex T): as the real routine with TRANS = 'N' or 'C' and
// upper triangular A, B, D, E.
template <typename T>
blasint TgsylComplex(char trans, blasint ijob, blasint m, blasint n, const T* a,
                     blasint lda, const T* b, blasint ldb, T* c, blasint ldc,
                     const T* d, blasint ldd, const T* e, blasint lde, T* f,
                     blasint ldf, RealT<T>* scale, RealT<T>* dif, T* work,
                     blasint lwork, blasint* iwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const T czero = T(0);
  const Matrix<const T> A(a, lda);
  const Matrix<const T> B(b, ldb);
  const Matrix<T> C(c, ldc);
  const Matrix<const T> D(d, ldd);
  const Matrix<const T> E(e, lde);
  const Matrix<T> F(f, ldf);
  const Vector<blasint> IWORK(iwork);
  const Vector<T> WORK(work);
  blasint i, ie, ifunc, iround, is, isolve, j, je, js, k, linfo, lwmin, mb, nb,
      p, pq, q;
  bool lquery, notran;
  R dscale, dsum, scale2, scaloc;
  blasint info = 0;
  notran = Lsame(trans, 'N');
  lquery = (lwork == -1);
  if (!notran && !Lsame(trans, 'C')) {
    info = -1;
  } else if (notran) {
    if ((ijob < 0) || (ijob > 4)) {
      info = -2;
    }
  }
  if (info == 0) {
    if (m <= 0) {
      info = -3;
    } else if (n <= 0) {
      info = -4;
    } else if (lda < Max<blasint>(1, m)) {
      info = -6;
    } else if (ldb < Max<blasint>(1, n)) {
      info = -8;
    } else if (ldc < Max<blasint>(1, m)) {
      info = -10;
    } else if (ldd < Max<blasint>(1, m)) {
      info = -12;
    } else if (lde < Max<blasint>(1, n)) {
      info = -14;
    } else if (ldf < Max<blasint>(1, m)) {
      info = -16;
    }
  }
  if (info == 0) {
    if (notran) {
      if (ijob == 1 || ijob == 2) {
        lwmin = Max<blasint>(1, 2 * m * n);
      } else {
        lwmin = 1;
      }
    } else {
      lwmin = 1;
    }
    WORK(1) = WorkSize<T>(lwmin);
    if (lwork < lwmin && !lquery) {
      info = -20;
    }
  }
  if (info != 0) {
    Xerbla<T>("TGSYL", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (m == 0 || n == 0) {
    *scale = 1;
    if (notran) {
      if (ijob != 0) {
        *dif = 0;
      }
    }
    return info;
  }
  // Determine  optimal block sizes MB and NB
  const char opts[2] = {trans, '\0'};
  mb = Ilaenv<T>(2, "TGSYL", opts, m, n, -1, -1);
  nb = Ilaenv<T>(5, "TGSYL", opts, m, n, -1, -1);
  isolve = 1;
  ifunc = 0;
  if (notran) {
    if (ijob >= 3) {
      ifunc = ijob - 2;
      Laset('F', m, n, czero, czero, c, ldc);
      Laset('F', m, n, czero, czero, f, ldf);
      // NOLINTNEXTLINE(bugprone-redundant-branch-condition): as LAPACK
    } else if (ijob >= 1 && notran) {
      isolve = 2;
    }
  }
  if ((mb <= 1 && nb <= 1) || (mb >= m && nb >= n)) {
    // Use unblocked Level 2 solver
    for (iround = 1; iround <= isolve; ++iround) {
      *scale = one;
      dscale = zero;
      dsum = one;
      pq = m * n;
      info = Tgsy2(trans, ifunc, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde,
                   f, ldf, scale, &dsum, &dscale);
      if (dscale != zero) {
        if (ijob == 1 || ijob == 3) {
          *dif = std::sqrt(R(2 * m * n)) / (dscale * std::sqrt(dsum));
        } else {
          *dif = std::sqrt(R(pq)) / (dscale * std::sqrt(dsum));
        }
      }
      if (isolve == 2 && iround == 1) {
        // NOLINTNEXTLINE(bugprone-redundant-branch-condition): as LAPACK
        if (notran) {
          ifunc = ijob;
        }
        scale2 = *scale;
        Lacpy('F', m, n, c, ldc, work, m);
        Lacpy('F', m, n, f, ldf, WORK.ptr(m * n + 1), m);
        Laset('F', m, n, czero, czero, c, ldc);
        Laset('F', m, n, czero, czero, f, ldf);
      } else if (isolve == 2 && iround == 2) {
        Lacpy('F', m, n, work, m, c, ldc);
        Lacpy('F', m, n, WORK.ptr(m * n + 1), m, f, ldf);
        *scale = scale2;
      }
    }
    return info;
  }
  // Determine block structure of A
  p = 0;
  i = 1;
L40:
  if (i > m) goto L50;
  p = p + 1;
  IWORK(p) = i;
  i = i + mb;
  if (i >= m) goto L50;
  goto L40;
L50:
  IWORK(p + 1) = m + 1;
  if (IWORK(p) == IWORK(p + 1)) p = p - 1;
  // Determine block structure of B
  q = p + 1;
  j = 1;
L60:
  if (j > n) goto L70;
  q = q + 1;
  IWORK(q) = j;
  j = j + nb;
  if (j >= n) goto L70;
  goto L60;
L70:
  IWORK(q + 1) = n + 1;
  if (IWORK(q) == IWORK(q + 1)) q = q - 1;
  if (notran) {
    for (iround = 1; iround <= isolve; ++iround) {
      // Solve (I, J) - subsystem
      // A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
      // D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
      // for I = P, P - 1, ..., 1; J = 1, 2, ..., Q
      pq = 0;
      *scale = one;
      dscale = zero;
      dsum = one;
      for (j = p + 2; j <= q; ++j) {
        js = IWORK(j);
        je = IWORK(j + 1) - 1;
        nb = je - js + 1;
        for (i = p; i >= 1; --i) {
          is = IWORK(i);
          ie = IWORK(i + 1) - 1;
          mb = ie - is + 1;
          linfo =
              Tgsy2(trans, ifunc, mb, nb, A.ptr(is, is), lda, B.ptr(js, js),
                    ldb, C.ptr(is, js), ldc, D.ptr(is, is), ldd, E.ptr(js, js),
                    lde, F.ptr(is, js), ldf, &scaloc, &dsum, &dscale);
          if (linfo > 0) info = linfo;
          pq = pq + mb * nb;
          if (scaloc != one) {
            for (k = 1; k <= js - 1; ++k) {
              Scal(m, T(scaloc), C.ptr(1, k), 1);
              Scal(m, T(scaloc), F.ptr(1, k), 1);
            }
            for (k = js; k <= je; ++k) {
              Scal(is - 1, T(scaloc), C.ptr(1, k), 1);
              Scal(is - 1, T(scaloc), F.ptr(1, k), 1);
            }
            for (k = js; k <= je; ++k) {
              Scal(m - ie, T(scaloc), C.ptr(ie + 1, k), 1);
              Scal(m - ie, T(scaloc), F.ptr(ie + 1, k), 1);
            }
            for (k = je + 1; k <= n; ++k) {
              Scal(m, T(scaloc), C.ptr(1, k), 1);
              Scal(m, T(scaloc), F.ptr(1, k), 1);
            }
            *scale = *scale * scaloc;
          }
          // Substitute R(I,J) and L(I,J) into remaining equation.
          if (i > 1) {
            Gemm('N', 'N', is - 1, nb, mb, T(-one), A.ptr(1, is), lda,
                 C.ptr(is, js), ldc, T(one), C.ptr(1, js), ldc);
            Gemm('N', 'N', is - 1, nb, mb, T(-one), D.ptr(1, is), ldd,
                 C.ptr(is, js), ldc, T(one), F.ptr(1, js), ldf);
          }
          if (j < q) {
            Gemm('N', 'N', mb, n - je, nb, T(one), F.ptr(is, js), ldf,
                 B.ptr(js, je + 1), ldb, T(one), C.ptr(is, je + 1), ldc);
            Gemm('N', 'N', mb, n - je, nb, T(one), F.ptr(is, js), ldf,
                 E.ptr(js, je + 1), lde, T(one), F.ptr(is, je + 1), ldf);
          }
        }
      }
      if (dscale != zero) {
        if (ijob == 1 || ijob == 3) {
          *dif = std::sqrt(R(2 * m * n)) / (dscale * std::sqrt(dsum));
        } else {
          *dif = std::sqrt(R(pq)) / (dscale * std::sqrt(dsum));
        }
      }
      if (isolve == 2 && iround == 1) {
        // NOLINTNEXTLINE(bugprone-redundant-branch-condition): as LAPACK
        if (notran) {
          ifunc = ijob;
        }
        scale2 = *scale;
        Lacpy('F', m, n, c, ldc, work, m);
        Lacpy('F', m, n, f, ldf, WORK.ptr(m * n + 1), m);
        Laset('F', m, n, czero, czero, c, ldc);
        Laset('F', m, n, czero, czero, f, ldf);
      } else if (isolve == 2 && iround == 2) {
        Lacpy('F', m, n, work, m, c, ldc);
        Lacpy('F', m, n, WORK.ptr(m * n + 1), m, f, ldf);
        *scale = scale2;
      }
    }
  } else {
    // Solve transposed (I, J)-subsystem
    // A(I, I)**H * R(I, J) + D(I, I)**H * L(I, J) = C(I, J)
    // R(I, J) * B(J, J)  + L(I, J) * E(J, J) = -F(I, J)
    // for I = 1,2,..., P; J = Q, Q-1,..., 1
    *scale = one;
    for (i = 1; i <= p; ++i) {
      is = IWORK(i);
      ie = IWORK(i + 1) - 1;
      mb = ie - is + 1;
      for (j = q; j >= p + 2; --j) {
        js = IWORK(j);
        je = IWORK(j + 1) - 1;
        nb = je - js + 1;
        linfo =
            Tgsy2(trans, ifunc, mb, nb, A.ptr(is, is), lda, B.ptr(js, js), ldb,
                  C.ptr(is, js), ldc, D.ptr(is, is), ldd, E.ptr(js, js), lde,
                  F.ptr(is, js), ldf, &scaloc, &dsum, &dscale);
        if (linfo > 0) info = linfo;
        if (scaloc != one) {
          for (k = 1; k <= js - 1; ++k) {
            Scal(m, T(scaloc), C.ptr(1, k), 1);
            Scal(m, T(scaloc), F.ptr(1, k), 1);
          }
          for (k = js; k <= je; ++k) {
            Scal(is - 1, T(scaloc), C.ptr(1, k), 1);
            Scal(is - 1, T(scaloc), F.ptr(1, k), 1);
          }
          for (k = js; k <= je; ++k) {
            Scal(m - ie, T(scaloc), C.ptr(ie + 1, k), 1);
            Scal(m - ie, T(scaloc), F.ptr(ie + 1, k), 1);
          }
          for (k = je + 1; k <= n; ++k) {
            Scal(m, T(scaloc), C.ptr(1, k), 1);
            Scal(m, T(scaloc), F.ptr(1, k), 1);
          }
          *scale = *scale * scaloc;
        }
        // Substitute R(I,J) and L(I,J) into remaining equation.
        if (j > p + 2) {
          Gemm('N', 'C', mb, js - 1, nb, T(one), C.ptr(is, js), ldc,
               B.ptr(1, js), ldb, T(one), F.ptr(is, 1), ldf);
          Gemm('N', 'C', mb, js - 1, nb, T(one), F.ptr(is, js), ldf,
               E.ptr(1, js), lde, T(one), F.ptr(is, 1), ldf);
        }
        if (i < p) {
          Gemm('C', 'N', m - ie, nb, mb, T(-one), A.ptr(is, ie + 1), lda,
               C.ptr(is, js), ldc, T(one), C.ptr(ie + 1, js), ldc);
          Gemm('C', 'N', m - ie, nb, mb, T(-one), D.ptr(is, ie + 1), ldd,
               F.ptr(is, js), ldf, T(one), C.ptr(ie + 1, js), ldc);
        }
      }
    }
  }
  WORK(1) = WorkSize<T>(lwmin);
  return info;
}

template <typename T>
blasint Tgsyl(char trans, blasint ijob, blasint m, blasint n, const T* a,
              blasint lda, const T* b, blasint ldb, T* c, blasint ldc,
              const T* d, blasint ldd, const T* e, blasint lde, T* f,
              blasint ldf, RealT<T>* scale, RealT<T>* dif, T* work,
              blasint lwork, blasint* iwork) {
  if constexpr (IsComplex<T>::value) {
    return TgsylComplex(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e,
                        lde, f, ldf, scale, dif, work, lwork, iwork);
  } else {
    return TgsylReal(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde,
                     f, ldf, scale, dif, work, lwork, iwork);
  }
}

// xLAGV2: the generalized Schur factorization of the real 2-by-2 pair (A,
// B), B upper triangular: rotations (CSL, SNL) from the left and (CSR,
// SNR) from the right make (A, B) upper triangular for real eigenvalues;
// for a complex pair, A stays a 2-by-2 block and B becomes diagonal (LAPACK
// documents positive entries, which the reference does not ensure: their
// signs are those of the rotated B, as in the port). The eigenvalues are
// (ALPHAR(k) + i*ALPHAI(k)) / BETA(k).
template <typename R>
void Lagv2(R* a, blasint lda, R* b, blasint ldb, R* alphar, R* alphai, R* beta,
           R* csl, R* snl, R* csr, R* snr) {
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Vector<R> ALPHAI(alphai);
  const Vector<R> ALPHAR(alphar);
  const Matrix<R> B(b, ldb);
  const Vector<R> BETA(beta);
  R anorm, ascale, bnorm, bscale, h1, h2, h3, qq, r, rr, safmin, scale1, scale2,
      t, ulp, wi, wr1, wr2;
  // Set by xLAG2 where they are read; initialized for clang-tidy.
  scale1 = scale2 = wi = wr1 = wr2 = zero;
  safmin = Lamch<R>('S');
  ulp = Lamch<R>('P');
  // Scale A
  anorm = Max(std::abs(A(1, 1)) + std::abs(A(2, 1)),
              std::abs(A(1, 2)) + std::abs(A(2, 2)), safmin);
  ascale = one / anorm;
  A(1, 1) = ascale * A(1, 1);
  A(1, 2) = ascale * A(1, 2);
  A(2, 1) = ascale * A(2, 1);
  A(2, 2) = ascale * A(2, 2);
  // Scale B
  bnorm = Max(std::abs(B(1, 1)), std::abs(B(1, 2)) + std::abs(B(2, 2)), safmin);
  bscale = one / bnorm;
  B(1, 1) = bscale * B(1, 1);
  B(1, 2) = bscale * B(1, 2);
  B(2, 2) = bscale * B(2, 2);
  // Check if A can be deflated
  if (std::abs(A(2, 1)) <= ulp) {
    *csl = one;
    *snl = zero;
    *csr = one;
    *snr = zero;
    A(2, 1) = zero;
    B(2, 1) = zero;
    wi = zero;
    // Check if B is singular
  } else if (std::abs(B(1, 1)) <= ulp) {
    Lartg(A(1, 1), A(2, 1), csl, snl, &r);
    *csr = one;
    *snr = zero;
    Rot(2, A.ptr(1, 1), lda, A.ptr(2, 1), lda, *csl, *snl);
    Rot(2, B.ptr(1, 1), ldb, B.ptr(2, 1), ldb, *csl, *snl);
    A(2, 1) = zero;
    B(1, 1) = zero;
    B(2, 1) = zero;
    wi = zero;
  } else if (std::abs(B(2, 2)) <= ulp) {
    Lartg(A(2, 2), A(2, 1), csr, snr, &t);
    *snr = -*snr;
    Rot(2, A.ptr(1, 1), 1, A.ptr(1, 2), 1, *csr, *snr);
    Rot(2, B.ptr(1, 1), 1, B.ptr(1, 2), 1, *csr, *snr);
    *csl = one;
    *snl = zero;
    A(2, 1) = zero;
    B(2, 1) = zero;
    B(2, 2) = zero;
    wi = zero;
  } else {
    // B is nonsingular, first compute the eigenvalues of (A,B)
    Lag2<R>(a, lda, b, ldb, safmin, &scale1, &scale2, &wr1, &wr2, &wi);
    if (wi == zero) {
      // two real eigenvalues, compute s*A-w*B
      h1 = scale1 * A(1, 1) - wr1 * B(1, 1);
      h2 = scale1 * A(1, 2) - wr1 * B(1, 2);
      h3 = scale1 * A(2, 2) - wr1 * B(2, 2);
      rr = Lapy2(h1, h2);
      qq = Lapy2(scale1 * A(2, 1), h3);
      if (rr > qq) {
        // find right rotation matrix to zero 1,1 element of
        // (sA - wB)
        Lartg(h2, h1, csr, snr, &t);
      } else {
        // find right rotation matrix to zero 2,1 element of
        // (sA - wB)
        Lartg(h3, scale1 * A(2, 1), csr, snr, &t);
      }
      *snr = -*snr;
      Rot(2, A.ptr(1, 1), 1, A.ptr(1, 2), 1, *csr, *snr);
      Rot(2, B.ptr(1, 1), 1, B.ptr(1, 2), 1, *csr, *snr);
      // compute inf norms of A and B
      h1 = Max(std::abs(A(1, 1)) + std::abs(A(1, 2)),
               std::abs(A(2, 1)) + std::abs(A(2, 2)));
      h2 = Max(std::abs(B(1, 1)) + std::abs(B(1, 2)),
               std::abs(B(2, 1)) + std::abs(B(2, 2)));
      if ((scale1 * h1) >= std::abs(wr1) * h2) {
        // find left rotation matrix Q to zero out B(2,1)
        Lartg(B(1, 1), B(2, 1), csl, snl, &r);
      } else {
        // find left rotation matrix Q to zero out A(2,1)
        Lartg(A(1, 1), A(2, 1), csl, snl, &r);
      }
      Rot(2, A.ptr(1, 1), lda, A.ptr(2, 1), lda, *csl, *snl);
      Rot(2, B.ptr(1, 1), ldb, B.ptr(2, 1), ldb, *csl, *snl);
      A(2, 1) = zero;
      B(2, 1) = zero;
    } else {
      // a pair of complex conjugate eigenvalues
      // first compute the SVD of the matrix B
      Lasv2(B(1, 1), B(1, 2), B(2, 2), &r, &t, snr, csr, snl, csl);
      // Form (A,B) := Q(A,B)Z**T where Q is left rotation matrix and
      // Z is right rotation matrix computed from DLASV2
      Rot(2, A.ptr(1, 1), lda, A.ptr(2, 1), lda, *csl, *snl);
      Rot(2, B.ptr(1, 1), ldb, B.ptr(2, 1), ldb, *csl, *snl);
      Rot(2, A.ptr(1, 1), 1, A.ptr(1, 2), 1, *csr, *snr);
      Rot(2, B.ptr(1, 1), 1, B.ptr(1, 2), 1, *csr, *snr);
      B(2, 1) = zero;
      B(1, 2) = zero;
    }
  }
  // Unscaling
  A(1, 1) = anorm * A(1, 1);
  A(2, 1) = anorm * A(2, 1);
  A(1, 2) = anorm * A(1, 2);
  A(2, 2) = anorm * A(2, 2);
  B(1, 1) = bnorm * B(1, 1);
  B(2, 1) = bnorm * B(2, 1);
  B(1, 2) = bnorm * B(1, 2);
  B(2, 2) = bnorm * B(2, 2);
  if (wi == zero) {
    ALPHAR(1) = A(1, 1);
    ALPHAR(2) = A(2, 2);
    ALPHAI(1) = zero;
    ALPHAI(2) = zero;
    BETA(1) = B(1, 1);
    BETA(2) = B(2, 2);
  } else {
    ALPHAR(1) = anorm * wr1 / scale1 / bnorm;
    ALPHAI(1) = anorm * wi / scale1 / bnorm;
    ALPHAR(2) = ALPHAR(1);
    ALPHAI(2) = -ALPHAI(1);
    BETA(1) = one;
    BETA(2) = one;
  }
}

// xTGEX2 (real T): swaps the adjacent diagonal blocks (A11, B11) and (A22,
// B22) of orders N1 and N2 (1 or 2) starting at row J1 of the pair (A, B)
// in generalized real Schur form by an orthogonal equivalence
// transformation, which is applied to Q and Z if WANTQ / WANTZ. The swap
// is accepted only if it is backward stable (tests with the Frobenius
// norms of A and B). WORK has LWORK >= max(1, N*(N2+N1), 2*(N2+N1)**2)
// elements.
//
// Returns INFO: 0, 1 if the swap was rejected (A, B, Q, Z unchanged), or
// -16 if LWORK is too small (WORK(1) returns the minimum).
template <typename R>
blasint Tgex2(bool wantq, bool wantz, blasint n, R* a, blasint lda, R* b,
              blasint ldb, R* q, blasint ldq, R* z, blasint ldz, blasint j1,
              blasint n1, blasint n2, R* work, blasint lwork) {
  const R zero = R(0);
  const R one = R(1);
  const R twenty = R(20);
  const blasint ldst = 4;
  const bool wands = true;
  const Matrix<R> A(a, lda);
  R ai[2];
  R ar[2];
  const Matrix<R> B(b, ldb);
  R be[2];
  R ir[ldst * ldst];
  const Matrix<R> IR(ir, ldst);
  R ircop[ldst * ldst];
  blasint iwork[ldst + 2];
  R li[ldst * ldst];
  const Matrix<R> LI(li, ldst);
  R licop[ldst * ldst];
  const Matrix<R> Q(q, ldq);
  R s[ldst * ldst];
  const Matrix<R> S(s, ldst);
  R scpy[ldst * ldst];
  const Matrix<R> SCPY(scpy, ldst);
  R t[ldst * ldst];
  const Matrix<R> T(t, ldst);
  R taul[ldst];
  R taur[ldst];
  R tcpy[ldst * ldst];
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint i, idum, linfo, m;
  bool strong, weak;
  R bqra21, brqa21, ddum, dnorma, dnormb, dscale, dsum, eps, f, g, sa, sb,
      scale, smlnum, thresha, threshb;
  blasint info = 0;
  // Quick return if possible
  if (n <= 1 || n1 <= 0 || n2 <= 0) return info;
  if (n1 > n || (j1 + n1) > n) return info;
  m = n1 + n2;
  // (STGEX2 has MAX(N*M, M*M*2), the same: M >= 2.)
  if (lwork < Max<blasint>(1, n * m, m * m * 2)) {
    info = -16;
    WORK(1) = R(Max<blasint>(1, n * m, m * m * 2));
    return info;
  }
  weak = false;
  strong = false;
  // Make a local copy of selected block
  Laset('F', ldst, ldst, zero, zero, li, ldst);
  Laset('F', ldst, ldst, zero, zero, ir, ldst);
  Lacpy('F', m, m, A.ptr(j1, j1), lda, s, ldst);
  Lacpy('F', m, m, B.ptr(j1, j1), ldb, t, ldst);
  // Compute threshold for testing acceptance of swapping.
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  dscale = zero;
  dsum = one;
  Lacpy('F', m, m, s, ldst, work, m);
  Lassq(m * m, work, 1, &dscale, &dsum);
  dnorma = dscale * std::sqrt(dsum);
  dscale = zero;
  dsum = one;
  Lacpy('F', m, m, t, ldst, work, m);
  Lassq(m * m, work, 1, &dscale, &dsum);
  dnormb = dscale * std::sqrt(dsum);
  // THRES has been changed from
  // THRESH = MAX( TEN*EPS*SA, SMLNUM )
  // to
  // THRESH = MAX( TWENTY*EPS*SA, SMLNUM )
  // on 04/01/10.
  // "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by
  // Jim Demmel and Guillaume Revy. See forum post 1783.
  thresha = Max(twenty * eps * dnorma, smlnum);
  threshb = Max(twenty * eps * dnormb, smlnum);
  if (m == 2) {
    // CASE 1: Swap 1-by-1 and 1-by-1 blocks.
    // Compute orthogonal QL and RQ that swap 1-by-1 and 1-by-1 blocks
    // using Givens rotations and perform the swap tentatively.
    f = S(2, 2) * T(1, 1) - T(2, 2) * S(1, 1);
    g = S(2, 2) * T(1, 2) - T(2, 2) * S(1, 2);
    sa = std::abs(S(2, 2)) * std::abs(T(1, 1));
    sb = std::abs(S(1, 1)) * std::abs(T(2, 2));
    Lartg(f, g, IR.ptr(1, 2), IR.ptr(1, 1), &ddum);
    IR(2, 1) = -IR(1, 2);
    IR(2, 2) = IR(1, 1);
    Rot(2, S.ptr(1, 1), 1, S.ptr(1, 2), 1, IR(1, 1), IR(2, 1));
    Rot(2, T.ptr(1, 1), 1, T.ptr(1, 2), 1, IR(1, 1), IR(2, 1));
    if (sa >= sb) {
      Lartg(S(1, 1), S(2, 1), LI.ptr(1, 1), LI.ptr(2, 1), &ddum);
    } else {
      Lartg(T(1, 1), T(2, 1), LI.ptr(1, 1), LI.ptr(2, 1), &ddum);
    }
    Rot(2, S.ptr(1, 1), ldst, S.ptr(2, 1), ldst, LI(1, 1), LI(2, 1));
    Rot(2, T.ptr(1, 1), ldst, T.ptr(2, 1), ldst, LI(1, 1), LI(2, 1));
    LI(2, 2) = LI(1, 1);
    LI(1, 2) = -LI(2, 1);
    // Weak stability test: |S21| <= O(EPS F-norm((A)))
    // and  |T21| <= O(EPS F-norm((B)))
    weak = std::abs(S(2, 1)) <= thresha && std::abs(T(2, 1)) <= threshb;
    if (!weak) goto L70;
    if (wands) {
      // Strong stability test:
      // F-norm((A-QL**H*S*QR)) <= O(EPS*F-norm((A)))
      // and
      // F-norm((B-QL**H*T*QR)) <= O(EPS*F-norm((B)))
      Lacpy('F', m, m, A.ptr(j1, j1), lda, WORK.ptr(m * m + 1), m);
      Gemm('N', 'N', m, m, m, one, li, ldst, s, ldst, zero, work, m);
      Gemm('N', 'T', m, m, m, -one, work, m, ir, ldst, one, WORK.ptr(m * m + 1),
           m);
      dscale = zero;
      dsum = one;
      Lassq(m * m, WORK.ptr(m * m + 1), 1, &dscale, &dsum);
      sa = dscale * std::sqrt(dsum);
      Lacpy('F', m, m, B.ptr(j1, j1), ldb, WORK.ptr(m * m + 1), m);
      Gemm('N', 'N', m, m, m, one, li, ldst, t, ldst, zero, work, m);
      Gemm('N', 'T', m, m, m, -one, work, m, ir, ldst, one, WORK.ptr(m * m + 1),
           m);
      dscale = zero;
      dsum = one;
      Lassq(m * m, WORK.ptr(m * m + 1), 1, &dscale, &dsum);
      sb = dscale * std::sqrt(dsum);
      strong = sa <= thresha && sb <= threshb;
      if (!strong) goto L70;
    }
    // Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
    // (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).
    Rot(j1 + 1, A.ptr(1, j1), 1, A.ptr(1, j1 + 1), 1, IR(1, 1), IR(2, 1));
    Rot(j1 + 1, B.ptr(1, j1), 1, B.ptr(1, j1 + 1), 1, IR(1, 1), IR(2, 1));
    Rot(n - j1 + 1, A.ptr(j1, j1), lda, A.ptr(j1 + 1, j1), lda, LI(1, 1),
        LI(2, 1));
    Rot(n - j1 + 1, B.ptr(j1, j1), ldb, B.ptr(j1 + 1, j1), ldb, LI(1, 1),
        LI(2, 1));
    // Set  N1-by-N2 (2,1) - blocks to ZERO.
    A(j1 + 1, j1) = zero;
    B(j1 + 1, j1) = zero;
    // Accumulate transformations into Q and Z if requested.
    if (wantz) Rot(n, Z.ptr(1, j1), 1, Z.ptr(1, j1 + 1), 1, IR(1, 1), IR(2, 1));
    if (wantq) Rot(n, Q.ptr(1, j1), 1, Q.ptr(1, j1 + 1), 1, LI(1, 1), LI(2, 1));
    // Exit with INFO = 0 if swap was successfully performed.
    return info;
  } else {
    // CASE 2: Swap 1-by-1 and 2-by-2 blocks, or 2-by-2
    // and 2-by-2 blocks.
    // Solve the generalized Sylvester equation
    // S11 * R - L * S22 = SCALE * S12
    // T11 * R - L * T22 = SCALE * T12
    // for R and L. Solutions in LI and IR.
    Lacpy('F', n1, n2, T.ptr(1, n1 + 1), ldst, li, ldst);
    Lacpy('F', n1, n2, S.ptr(1, n1 + 1), ldst, IR.ptr(n2 + 1, n1 + 1), ldst);
    linfo =
        Tgsy2<R>('N', 0, n1, n2, s, ldst, S.ptr(n1 + 1, n1 + 1), ldst,
                 IR.ptr(n2 + 1, n1 + 1), ldst, t, ldst, T.ptr(n1 + 1, n1 + 1),
                 ldst, li, ldst, &scale, &dsum, &dscale, iwork, &idum);
    if (linfo != 0) goto L70;
    // Compute orthogonal matrix QL:
    // QL**T * LI = [ TL ]
    // [ 0  ]
    // where
    // LI =  [      -L              ]
    // [ SCALE * identity(N2) ]
    for (i = 1; i <= n2; ++i) {
      Rscal(n1, -one, LI.ptr(1, i), 1);
      LI(n1 + i, i) = scale;
    }
    linfo = Geqr2(m, n2, li, ldst, taul, work);
    if (linfo != 0) goto L70;
    linfo = Ung2r(m, m, n2, li, ldst, taul, work);
    if (linfo != 0) goto L70;
    // Compute orthogonal matrix RQ:
    // IR * RQ**T =   [ 0  TR],
    // where IR = [ SCALE * identity(N1), R ]
    for (i = 1; i <= n1; ++i) {
      // IR(N2+I, I), written directly: clang-tidy does not know that N1,
      // N2 <= 2.
      // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
      ir[(n2 + i - 1) + (i - 1) * ldst] = scale;
    }
    linfo = Gerq2(n1, m, IR.ptr(n2 + 1, 1), ldst, taur, work);
    if (linfo != 0) goto L70;
    linfo = Ungr2(m, m, n1, ir, ldst, taur, work);
    if (linfo != 0) goto L70;
    // Perform the swapping tentatively:
    Gemm('T', 'N', m, m, m, one, li, ldst, s, ldst, zero, work, m);
    Gemm('N', 'T', m, m, m, one, work, m, ir, ldst, zero, s, ldst);
    Gemm('T', 'N', m, m, m, one, li, ldst, t, ldst, zero, work, m);
    Gemm('N', 'T', m, m, m, one, work, m, ir, ldst, zero, t, ldst);
    Lacpy('F', m, m, s, ldst, scpy, ldst);
    Lacpy('F', m, m, t, ldst, tcpy, ldst);
    Lacpy('F', m, m, ir, ldst, ircop, ldst);
    Lacpy('F', m, m, li, ldst, licop, ldst);
    // Triangularize the B-part by an RQ factorization.
    // Apply transformation (from left) to A-part, giving S.
    linfo = Gerq2(m, m, t, ldst, taur, work);
    if (linfo != 0) goto L70;
    linfo = Unmr2('R', 'T', m, m, m, t, ldst, taur, s, ldst, work);
    if (linfo != 0) goto L70;
    linfo = Unmr2('L', 'N', m, m, m, t, ldst, taur, ir, ldst, work);
    if (linfo != 0) goto L70;
    // Compute F-norm(S21) in BRQA21. (T21 is 0.)
    dscale = zero;
    dsum = one;
    for (i = 1; i <= n2; ++i) {
      Lassq(n1, S.ptr(n2 + 1, i), 1, &dscale, &dsum);
    }
    brqa21 = dscale * std::sqrt(dsum);
    // Triangularize the B-part by a QR factorization.
    // Apply transformation (from right) to A-part, giving S.
    linfo = Geqr2(m, m, tcpy, ldst, taul, work);
    if (linfo != 0) goto L70;
    info = Unm2r('L', 'T', m, m, m, tcpy, ldst, taul, scpy, ldst, work);
    info = Unm2r('R', 'N', m, m, m, tcpy, ldst, taul, licop, ldst, work);
    if (linfo != 0) goto L70;
    // Compute F-norm(S21) in BQRA21. (T21 is 0.)
    dscale = zero;
    dsum = one;
    for (i = 1; i <= n2; ++i) {
      Lassq(n1, SCPY.ptr(n2 + 1, i), 1, &dscale, &dsum);
    }
    bqra21 = dscale * std::sqrt(dsum);
    // Decide which method to use.
    // Weak stability test:
    // F-norm(S21) <= O(EPS * F-norm((S)))
    if (bqra21 <= brqa21 && bqra21 <= thresha) {
      Lacpy('F', m, m, scpy, ldst, s, ldst);
      Lacpy('F', m, m, tcpy, ldst, t, ldst);
      Lacpy('F', m, m, ircop, ldst, ir, ldst);
      Lacpy('F', m, m, licop, ldst, li, ldst);
    } else if (brqa21 >= thresha) {
      goto L70;
    }
    // Set lower triangle of B-part to zero
    Laset('L', m - 1, m - 1, zero, zero, T.ptr(2, 1), ldst);
    if (wands) {
      // Strong stability test:
      // F-norm((A-QL**H*S*QR)) <= O(EPS*F-norm((A)))
      // and
      // F-norm((B-QL**H*T*QR)) <= O(EPS*F-norm((B)))
      Lacpy('F', m, m, A.ptr(j1, j1), lda, WORK.ptr(m * m + 1), m);
      Gemm('N', 'N', m, m, m, one, li, ldst, s, ldst, zero, work, m);
      Gemm('N', 'N', m, m, m, -one, work, m, ir, ldst, one, WORK.ptr(m * m + 1),
           m);
      dscale = zero;
      dsum = one;
      Lassq(m * m, WORK.ptr(m * m + 1), 1, &dscale, &dsum);
      sa = dscale * std::sqrt(dsum);
      Lacpy('F', m, m, B.ptr(j1, j1), ldb, WORK.ptr(m * m + 1), m);
      Gemm('N', 'N', m, m, m, one, li, ldst, t, ldst, zero, work, m);
      Gemm('N', 'N', m, m, m, -one, work, m, ir, ldst, one, WORK.ptr(m * m + 1),
           m);
      dscale = zero;
      dsum = one;
      Lassq(m * m, WORK.ptr(m * m + 1), 1, &dscale, &dsum);
      sb = dscale * std::sqrt(dsum);
      strong = sa <= thresha && sb <= threshb;
      if (!strong) goto L70;
    }
    // If the swap is accepted ("weakly" and "strongly"), apply the
    // transformations and set N1-by-N2 (2,1)-block to zero.
    Laset('F', n1, n2, zero, zero, S.ptr(n2 + 1, 1), ldst);
    // copy back M-by-M diagonal block starting at index J1 of (A, B)
    Lacpy('F', m, m, s, ldst, A.ptr(j1, j1), lda);
    Lacpy('F', m, m, t, ldst, B.ptr(j1, j1), ldb);
    Laset('F', ldst, ldst, zero, zero, t, ldst);
    // Standardize existing 2-by-2 blocks.
    Laset('F', m, m, zero, zero, work, m);
    WORK(1) = one;
    T(1, 1) = one;
    idum = lwork - m * m - 2;
    if (n2 > 1) {
      Lagv2<R>(A.ptr(j1, j1), lda, B.ptr(j1, j1), ldb, ar, ai, be, WORK.ptr(1),
               WORK.ptr(2), T.ptr(1, 1), T.ptr(2, 1));
      WORK(m + 1) = -WORK(2);
      WORK(m + 2) = WORK(1);
      T(n2, n2) = T(1, 1);
      T(1, 2) = -T(2, 1);
    }
    WORK(m * m) = one;
    T(m, m) = one;
    if (n1 > 1) {
      Lagv2<R>(A.ptr(j1 + n2, j1 + n2), lda, B.ptr(j1 + n2, j1 + n2), ldb, taur,
               taul, WORK.ptr(m * m + 1), WORK.ptr(n2 * m + n2 + 1),
               WORK.ptr(n2 * m + n2 + 2), T.ptr(n2 + 1, n2 + 1),
               T.ptr(m, m - 1));
      WORK(m * m) = WORK(n2 * m + n2 + 1);
      WORK(m * m - 1) = -WORK(n2 * m + n2 + 2);
      T(m, m) = T(n2 + 1, n2 + 1);
      T(m - 1, m) = -T(m, m - 1);
    }
    Gemm('T', 'N', n2, n1, n2, one, work, m, A.ptr(j1, j1 + n2), lda, zero,
         WORK.ptr(m * m + 1), n2);
    Lacpy('F', n2, n1, WORK.ptr(m * m + 1), n2, A.ptr(j1, j1 + n2), lda);
    Gemm('T', 'N', n2, n1, n2, one, work, m, B.ptr(j1, j1 + n2), ldb, zero,
         WORK.ptr(m * m + 1), n2);
    Lacpy('F', n2, n1, WORK.ptr(m * m + 1), n2, B.ptr(j1, j1 + n2), ldb);
    Gemm('N', 'N', m, m, m, one, li, ldst, work, m, zero, WORK.ptr(m * m + 1),
         m);
    Lacpy('F', m, m, WORK.ptr(m * m + 1), m, li, ldst);
    Gemm('N', 'N', n2, n1, n1, one, A.ptr(j1, j1 + n2), lda,
         T.ptr(n2 + 1, n2 + 1), ldst, zero, work, n2);
    Lacpy('F', n2, n1, work, n2, A.ptr(j1, j1 + n2), lda);
    Gemm('N', 'N', n2, n1, n1, one, B.ptr(j1, j1 + n2), ldb,
         T.ptr(n2 + 1, n2 + 1), ldst, zero, work, n2);
    Lacpy('F', n2, n1, work, n2, B.ptr(j1, j1 + n2), ldb);
    Gemm('T', 'N', m, m, m, one, ir, ldst, t, ldst, zero, work, m);
    Lacpy('F', m, m, work, m, ir, ldst);
    // Accumulate transformations into Q and Z if requested.
    if (wantq) {
      Gemm('N', 'N', n, m, m, one, Q.ptr(1, j1), ldq, li, ldst, zero, work, n);
      Lacpy('F', n, m, work, n, Q.ptr(1, j1), ldq);
    }
    if (wantz) {
      Gemm('N', 'N', n, m, m, one, Z.ptr(1, j1), ldz, ir, ldst, zero, work, n);
      Lacpy('F', n, m, work, n, Z.ptr(1, j1), ldz);
    }
    // Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and
    // (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).
    i = j1 + m;
    if (i <= n) {
      Gemm('T', 'N', m, n - i + 1, m, one, li, ldst, A.ptr(j1, i), lda, zero,
           work, m);
      Lacpy('F', m, n - i + 1, work, m, A.ptr(j1, i), lda);
      Gemm('T', 'N', m, n - i + 1, m, one, li, ldst, B.ptr(j1, i), ldb, zero,
           work, m);
      Lacpy('F', m, n - i + 1, work, m, B.ptr(j1, i), ldb);
    }
    i = j1 - 1;
    if (i > 0) {
      Gemm('N', 'N', i, m, m, one, A.ptr(1, j1), lda, ir, ldst, zero, work, i);
      Lacpy('F', i, m, work, i, A.ptr(1, j1), lda);
      Gemm('N', 'N', i, m, m, one, B.ptr(1, j1), ldb, ir, ldst, zero, work, i);
      Lacpy('F', i, m, work, i, B.ptr(1, j1), ldb);
    }
    // Exit with INFO = 0 if swap was successfully performed.
    return info;
  }
  // Exit with INFO = 1 if swap was rejected.
L70:
  info = 1;
  return info;
}

// xTGEX2 (complex T): swaps the adjacent 1-by-1 blocks at J1 and J1+1 of
// the pair (A, B) in generalized Schur form (as the real routine; no
// workspace).
template <typename T>
blasint Tgex2(bool wantq, bool wantz, blasint n, T* a, blasint lda, T* b,
              blasint ldb, T* q, blasint ldq, T* z, blasint ldz, blasint j1) {
  using R = RealT<T>;

  const T czero = T(0);
  const T cone = T(1);
  const R twenty = R(20);
  const blasint ldst = 2;
  const bool wands = true;
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Matrix<T> Q(q, ldq);
  T s[ldst * ldst];
  const Matrix<T> S(s, ldst);
  T t[ldst * ldst];
  const Matrix<T> Tm(t, ldst);
  T work[8];
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint i, m;
  bool strong, weak;
  R cq, cz, eps, sa, sb, scale, smlnum, sum, thresha, threshb;
  T cdum, f, g, sq, sz;
  blasint info = 0;
  // Quick return if possible
  if (n <= 1) return info;
  m = ldst;
  weak = false;
  strong = false;
  // Make a local copy of selected block in (A, B)
  Lacpy('F', m, m, A.ptr(j1, j1), lda, s, ldst);
  Lacpy('F', m, m, B.ptr(j1, j1), ldb, t, ldst);
  // Compute the threshold for testing the acceptance of swapping.
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  scale = std::real(czero);
  sum = std::real(cone);
  Lacpy('F', m, m, s, ldst, work, m);
  Lacpy('F', m, m, t, ldst, WORK.ptr(m * m + 1), m);
  Lassq(m * m, work, 1, &scale, &sum);
  sa = scale * std::sqrt(sum);
  scale = std::real(czero);
  sum = std::real(cone);
  Lassq(m * m, WORK.ptr(m * m + 1), 1, &scale, &sum);
  sb = scale * std::sqrt(sum);
  // THRES has been changed from
  // THRESH = MAX( TEN*EPS*SA, SMLNUM )
  // to
  // THRESH = MAX( TWENTY*EPS*SA, SMLNUM )
  // on 04/01/10.
  // "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by
  // Jim Demmel and Guillaume Revy. See forum post 1783.
  thresha = Max(twenty * eps * sa, smlnum);
  threshb = Max(twenty * eps * sb, smlnum);
  // Compute unitary QL and RQ that swap 1-by-1 and 1-by-1 blocks
  // using Givens rotations and perform the swap tentatively.
  f = S(2, 2) * Tm(1, 1) - Tm(2, 2) * S(1, 1);
  g = S(2, 2) * Tm(1, 2) - Tm(2, 2) * S(1, 2);
  sa = std::abs(S(2, 2)) * std::abs(Tm(1, 1));
  sb = std::abs(S(1, 1)) * std::abs(Tm(2, 2));
  Lartg(g, f, &cz, &sz, &cdum);
  sz = -sz;
  Rot(2, S.ptr(1, 1), 1, S.ptr(1, 2), 1, cz, Conj(sz));
  Rot(2, Tm.ptr(1, 1), 1, Tm.ptr(1, 2), 1, cz, Conj(sz));
  if (sa >= sb) {
    Lartg(S(1, 1), S(2, 1), &cq, &sq, &cdum);
  } else {
    Lartg(Tm(1, 1), Tm(2, 1), &cq, &sq, &cdum);
  }
  Rot(2, S.ptr(1, 1), ldst, S.ptr(2, 1), ldst, cq, sq);
  Rot(2, Tm.ptr(1, 1), ldst, Tm.ptr(2, 1), ldst, cq, sq);
  // Weak stability test: |S21| <= O(EPS F-norm((A)))
  // and  |T21| <= O(EPS F-norm((B)))
  weak = std::abs(S(2, 1)) <= thresha && std::abs(Tm(2, 1)) <= threshb;
  if (!weak) goto L20;
  if (wands) {
    // Strong stability test:
    // F-norm((A-QL**H*S*QR)) <= O(EPS*F-norm((A)))
    // and
    // F-norm((B-QL**H*T*QR)) <= O(EPS*F-norm((B)))
    Lacpy('F', m, m, s, ldst, work, m);
    Lacpy('F', m, m, t, ldst, WORK.ptr(m * m + 1), m);
    Rot(2, work, 1, WORK.ptr(3), 1, cz, -Conj(sz));
    Rot(2, WORK.ptr(5), 1, WORK.ptr(7), 1, cz, -Conj(sz));
    Rot(2, work, 2, WORK.ptr(2), 2, cq, -sq);
    Rot(2, WORK.ptr(5), 2, WORK.ptr(6), 2, cq, -sq);
    for (i = 1; i <= 2; ++i) {
      WORK(i) = WORK(i) - A(j1 + i - 1, j1);
      WORK(i + 2) = WORK(i + 2) - A(j1 + i - 1, j1 + 1);
      WORK(i + 4) = WORK(i + 4) - B(j1 + i - 1, j1);
      WORK(i + 6) = WORK(i + 6) - B(j1 + i - 1, j1 + 1);
    }
    scale = std::real(czero);
    sum = std::real(cone);
    Lassq(m * m, work, 1, &scale, &sum);
    sa = scale * std::sqrt(sum);
    scale = std::real(czero);
    sum = std::real(cone);
    Lassq(m * m, WORK.ptr(m * m + 1), 1, &scale, &sum);
    sb = scale * std::sqrt(sum);
    strong = sa <= thresha && sb <= threshb;
    if (!strong) goto L20;
  }
  // If the swap is accepted ("weakly" and "strongly"), apply the
  // equivalence transformations to the original matrix pair (A,B)
  Rot(j1 + 1, A.ptr(1, j1), 1, A.ptr(1, j1 + 1), 1, cz, Conj(sz));
  Rot(j1 + 1, B.ptr(1, j1), 1, B.ptr(1, j1 + 1), 1, cz, Conj(sz));
  Rot(n - j1 + 1, A.ptr(j1, j1), lda, A.ptr(j1 + 1, j1), lda, cq, sq);
  Rot(n - j1 + 1, B.ptr(j1, j1), ldb, B.ptr(j1 + 1, j1), ldb, cq, sq);
  // Set  N1 by N2 (2,1) blocks to 0
  A(j1 + 1, j1) = czero;
  B(j1 + 1, j1) = czero;
  // Accumulate transformations into Q and Z if requested.
  if (wantz) Rot(n, Z.ptr(1, j1), 1, Z.ptr(1, j1 + 1), 1, cz, Conj(sz));
  if (wantq) Rot(n, Q.ptr(1, j1), 1, Q.ptr(1, j1 + 1), 1, cq, Conj(sq));
  // Exit with INFO = 0 if swap was successfully performed.
  return info;
  // Exit with INFO = 1 if swap was rejected.
L20:
  info = 1;
  return info;
}

// xTGEXC (real T): moves the diagonal block of the pair (A, B) in
// generalized real Schur form starting at row IFST to row ILST by a
// sequence of swaps (xTGEX2), updating Q and Z if WANTQ / WANTZ. IFST and
// ILST are adjusted to the first rows of 2-by-2 blocks; ILST returns the
// final position. WORK has LWORK >= 1 (N <= 1) or 4*N+16 elements (LWORK
// = -1: workspace query).
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if a swap was
// rejected (ILST returns the position reached).
template <typename R>
blasint Tgexc(bool wantq, bool wantz, blasint n, R* a, blasint lda, R* b,
              blasint ldb, R* q, blasint ldq, R* z, blasint ldz, blasint* ifst,
              blasint* ilst, R* work, blasint lwork) {
  using T = R;

  const R zero = R(0);
  const Matrix<R> A(a, lda);
  const Matrix<R> B(b, ldb);
  const Matrix<R> Q(q, ldq);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint here, lwmin, nbf, nbl, nbnext;
  bool lquery;
  blasint info = 0;
  lquery = (lwork == -1);
  if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldq < 1 || (wantq && ldq < Max<blasint>(1, n))) {
    info = -9;
  } else if (ldz < 1 || (wantz && ldz < Max<blasint>(1, n))) {
    info = -11;
  } else if (*ifst < 1 || *ifst > n) {
    info = -12;
  } else if (*ilst < 1 || *ilst > n) {
    info = -13;
  }
  if (info == 0) {
    if (n <= 1) {
      lwmin = 1;
    } else {
      lwmin = 4 * n + 16;
    }
    WORK(1) = WorkSize<T>(lwmin);
    if (lwork < lwmin && !lquery) {
      info = -15;
    }
  }
  if (info != 0) {
    Xerbla<T>("TGEXC", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible
  if (n <= 1) return info;
  // Determine the first row of the specified block and find out
  // if it is 1-by-1 or 2-by-2.
  if (*ifst > 1) {
    if (A(*ifst, *ifst - 1) != zero) *ifst = *ifst - 1;
  }
  nbf = 1;
  if (*ifst < n) {
    if (A(*ifst + 1, *ifst) != zero) nbf = 2;
  }
  // Determine the first row of the final block
  // and find out if it is 1-by-1 or 2-by-2.
  if (*ilst > 1) {
    if (A(*ilst, *ilst - 1) != zero) *ilst = *ilst - 1;
  }
  nbl = 1;
  if (*ilst < n) {
    if (A(*ilst + 1, *ilst) != zero) nbl = 2;
  }
  if (*ifst == *ilst) return info;
  if (*ifst < *ilst) {
    // Update ILST.
    if (nbf == 2 && nbl == 1) *ilst = *ilst - 1;
    if (nbf == 1 && nbl == 2) *ilst = *ilst + 1;
    here = *ifst;
  L10:
    // Swap with next one below.
    if (nbf == 1 || nbf == 2) {
      // Current block either 1-by-1 or 2-by-2.
      nbnext = 1;
      if (here + nbf + 1 <= n) {
        if (A(here + nbf + 1, here + nbf) != zero) nbnext = 2;
      }
      info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, nbf,
                   nbnext, work, lwork);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      here = here + nbnext;
      // Test if 2-by-2 block breaks into two 1-by-1 blocks.
      if (nbf == 2) {
        if (A(here + 1, here) == zero) nbf = 3;
      }
    } else {
      // Current block consists of two 1-by-1 blocks, each of which
      // must be swapped individually.
      nbnext = 1;
      if (here + 3 <= n) {
        if (A(here + 3, here + 2) != zero) nbnext = 2;
      }
      info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here + 1, 1,
                   nbnext, work, lwork);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      if (nbnext == 1) {
        // Swap two 1-by-1 blocks.
        info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1,
                     1, work, lwork);
        if (info != 0) {
          *ilst = here;
          return info;
        }
        here = here + 1;
      } else {
        // Recompute NBNEXT in case of 2-by-2 split.
        if (A(here + 2, here + 1) == zero) nbnext = 1;
        if (nbnext == 2) {
          // 2-by-2 block did not split.
          info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1,
                       nbnext, work, lwork);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here + 2;
        } else {
          // 2-by-2 block did split.
          info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1,
                       1, work, lwork);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here + 1;
          info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1,
                       1, work, lwork);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here + 1;
        }
      }
    }
    if (here < *ilst) goto L10;
  } else {
    here = *ifst;
  L20:
    // Swap with next one below.
    if (nbf == 1 || nbf == 2) {
      // Current block either 1-by-1 or 2-by-2.
      nbnext = 1;
      if (here >= 3) {
        if (A(here - 1, here - 2) != zero) nbnext = 2;
      }
      info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz,
                   here - nbnext, nbnext, nbf, work, lwork);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      here = here - nbnext;
      // Test if 2-by-2 block breaks into two 1-by-1 blocks.
      if (nbf == 2) {
        if (A(here + 1, here) == zero) nbf = 3;
      }
    } else {
      // Current block consists of two 1-by-1 blocks, each of which
      // must be swapped individually.
      nbnext = 1;
      if (here >= 3) {
        if (A(here - 1, here - 2) != zero) nbnext = 2;
      }
      info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz,
                   here - nbnext, nbnext, 1, work, lwork);
      if (info != 0) {
        *ilst = here;
        return info;
      }
      if (nbnext == 1) {
        // Swap two 1-by-1 blocks.
        info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here,
                     nbnext, 1, work, lwork);
        if (info != 0) {
          *ilst = here;
          return info;
        }
        here = here - 1;
      } else {
        // Recompute NBNEXT in case of 2-by-2 split.
        if (A(here, here - 1) == zero) nbnext = 1;
        if (nbnext == 2) {
          // 2-by-2 block did not split.
          info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz,
                       here - 1, 2, 1, work, lwork);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here - 2;
        } else {
          // 2-by-2 block did split.
          info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1,
                       1, work, lwork);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here - 1;
          info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1,
                       1, work, lwork);
          if (info != 0) {
            *ilst = here;
            return info;
          }
          here = here - 1;
        }
      }
    }
    if (here > *ilst) goto L20;
  }
  *ilst = here;
  WORK(1) = WorkSize<T>(lwmin);
  return info;
}

// xTGEXC (complex T): as the real routine, for 1-by-1 blocks (IFST is not
// changed; no workspace). As in the reference, ILST returns the final
// position minus one after a move down (IFST < ILST): the routine
// decrements its counter after the last swap.
template <typename T>
blasint Tgexc(bool wantq, bool wantz, blasint n, T* a, blasint lda, T* b,
              blasint ldb, T* q, blasint ldq, T* z, blasint ldz, blasint ifst,
              blasint* ilst) {
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Matrix<T> Q(q, ldq);
  const Matrix<T> Z(z, ldz);
  blasint here;
  blasint info = 0;
  if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldq < 1 || (wantq && ldq < Max<blasint>(1, n))) {
    info = -9;
  } else if (ldz < 1 || (wantz && ldz < Max<blasint>(1, n))) {
    info = -11;
  } else if (ifst < 1 || ifst > n) {
    info = -12;
  } else if (*ilst < 1 || *ilst > n) {
    info = -13;
  }
  if (info != 0) {
    Xerbla<T>("TGEXC", -info);
    return info;
  }
  // Quick return if possible
  if (n <= 1) return info;
  if (ifst == *ilst) return info;
  if (ifst < *ilst) {
    here = ifst;
  L10:
    // Swap with next one below
    info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here);
    if (info != 0) {
      *ilst = here;
      return info;
    }
    here = here + 1;
    if (here < *ilst) goto L10;
    here = here - 1;
  } else {
    here = ifst - 1;
  L20:
    // Swap with next one above
    info = Tgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here);
    if (info != 0) {
      *ilst = here;
      return info;
    }
    here = here - 1;
    if (here >= *ilst) goto L20;
    here = here + 1;
  }
  *ilst = here;
  return info;
}

// xTGSEN (real T): reorders the generalized real Schur form (A, B) = (Q *
// S * Z**T, Q * T * Z**T) so that the eigenvalues selected by SELECT (a
// complex pair if either of SELECT(j), SELECT(j+1) is true) are in the
// leading M-by-M block, updates Q and Z if WANTQ / WANTZ, and returns the
// eigenvalues of the reordered pair (ALPHAR + i*ALPHAI) / BETA. IJOB = 0:
// reorder only; 1: also PL and PR, reciprocal norms of the projections
// onto the left and right deflating subspaces; 2: DIF(1:2), estimates of
// Difu and Difl (Frobenius norm based); 3: DIF (1-norm based, more
// expensive); 4: as 1 and 2; 5: as 1 and 3. WORK has LWORK >= max(1,
// 4*N+16) elements, also >= 2*M*(N-M)+1 for IJOB = 1, 2, 4 and >= 4*M*(N-M)
// for IJOB = 3, 5; IWORK has LIWORK >= max(1, N+6) elements, max(1,
// 2*M*(N-M), N+6) for IJOB = 3, 5 (LWORK = -1 or LIWORK = -1: workspace
// query).
//
// Returns INFO: 0, -i for an illegal argument i, or 1 if the reordering
// failed (a swap was rejected: the pair is ill-conditioned and partly
// reordered).
//
// Deviation: for IJOB = 1, 2, 4 the minimum LWORK is 2*M*(N-M)+1 where the
// reference has 2*M*(N-M) (also as the result of a workspace query). The
// routine passes LWORK - 2*M*(N-M) to xTGSYL, which needs at least 1, so
// that with the reference's minimum xTGSYL reports an illegal LWORK and
// PL, PR or DIF are not computed. (In the complex routine, whose minimum
// has no 4*N+16 term, this happens whenever the minimum is used.)
template <typename R>
blasint Tgsen(blasint ijob, bool wantq, bool wantz, const blasint* select,
              blasint n, R* a, blasint lda, R* b, blasint ldb, R* alphar,
              R* alphai, R* beta, R* q, blasint ldq, R* z, blasint ldz,
              blasint* m, R* pl, R* pr, R* dif, R* work, blasint lwork,
              blasint* iwork, blasint liwork) {
  using T = R;

  const blasint idifjb = 3;
  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> A(a, lda);
  const Vector<R> ALPHAI(alphai);
  const Vector<R> ALPHAR(alphar);
  const Matrix<R> B(b, ldb);
  const Vector<R> BETA(beta);
  const Vector<R> DIF(dif);
  blasint isave[3];
  const Vector<blasint> IWORK(iwork);
  const Matrix<R> Q(q, ldq);
  const Vector<const blasint> SELECT(select);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldz);
  blasint i, ierr, ijb, k, kase, kk, ks, liwmin, lwmin, mn2, n1, n2;
  bool lquery, pair, swap, wantd, wantd1, wantd2, wantp;
  R dsum, eps, rdscal, smlnum;
  // Set by xTGSYL where it is read; initialized for clang-tidy.
  R dscale = zero;
  blasint info = 0;
  lquery = (lwork == -1 || liwork == -1);
  if (ijob < 0 || ijob > 5) {
    info = -1;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -14;
  } else if (ldz < 1 || (wantz && ldz < n)) {
    info = -16;
  }
  if (info != 0) {
    Xerbla<T>("TGSEN", -info);
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  ierr = 0;
  wantp = ijob == 1 || ijob >= 4;
  wantd1 = ijob == 2 || ijob == 4;
  wantd2 = ijob == 3 || ijob == 5;
  wantd = wantd1 || wantd2;
  // Set M to the dimension of the specified pair of deflating
  // subspaces.
  *m = 0;
  pair = false;
  if (!lquery || ijob != 0) {
    for (k = 1; k <= n; ++k) {
      if (pair) {
        pair = false;
      } else {
        if (k < n) {
          if (A(k + 1, k) == zero) {
            if (SELECT(k)) *m = *m + 1;
          } else {
            pair = true;
            if (SELECT(k) || SELECT(k + 1)) *m = *m + 2;
          }
        } else {
          if (SELECT(n)) *m = *m + 1;
        }
      }
    }
  }
  if (ijob == 1 || ijob == 2 || ijob == 4) {
    // Deviation (see above): + 1.
    lwmin = Max<blasint>(1, 4 * n + 16, 2 * *m * (n - *m) + 1);
    liwmin = Max<blasint>(1, n + 6);
  } else if (ijob == 3 || ijob == 5) {
    lwmin = Max<blasint>(1, 4 * n + 16, 4 * *m * (n - *m));
    liwmin = Max<blasint>(1, 2 * *m * (n - *m), n + 6);
  } else {
    lwmin = Max<blasint>(1, 4 * n + 16);
    liwmin = 1;
  }
  WORK(1) = WorkSize<T>(lwmin);
  IWORK(1) = liwmin;
  if (lwork < lwmin && !lquery) {
    info = -22;
  } else if (liwork < liwmin && !lquery) {
    info = -24;
  }
  if (info != 0) {
    Xerbla<T>("TGSEN", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible.
  if (*m == n || *m == 0) {
    if (wantp) {
      *pl = one;
      *pr = one;
    }
    if (wantd) {
      dscale = zero;
      dsum = one;
      for (i = 1; i <= n; ++i) {
        Lassq(n, A.ptr(1, i), 1, &dscale, &dsum);
        Lassq(n, B.ptr(1, i), 1, &dscale, &dsum);
      }
      DIF(1) = dscale * std::sqrt(dsum);
      DIF(2) = DIF(1);
    }
    goto L60;
  }
  // Collect the selected blocks at the top-left corner of (A, B).
  ks = 0;
  pair = false;
  for (k = 1; k <= n; ++k) {
    if (pair) {
      pair = false;
    } else {
      swap = SELECT(k);
      if (k < n) {
        if (A(k + 1, k) != zero) {
          pair = true;
          swap = swap || SELECT(k + 1);
        }
      }
      if (swap) {
        ks = ks + 1;
        // Swap the K-th block to position KS.
        // Perform the reordering of diagonal blocks in (A, B)
        // by orthogonal transformation matrices and update
        // Q and Z accordingly (if requested):
        kk = k;
        if (k != ks)
          ierr = Tgexc<R>(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &kk,
                          &ks, work, lwork);
        if (ierr > 0) {
          // Swap is rejected: exit.
          info = 1;
          if (wantp) {
            *pl = zero;
            *pr = zero;
          }
          if (wantd) {
            DIF(1) = zero;
            DIF(2) = zero;
          }
          goto L60;
        }
        if (pair) ks = ks + 1;
      }
    }
  }
  if (wantp) {
    // Solve generalized Sylvester equation for R and L
    // and compute PL and PR.
    n1 = *m;
    n2 = n - *m;
    i = n1 + 1;
    ijb = 0;
    Lacpy('F', n1, n2, A.ptr(1, i), lda, work, n1);
    Lacpy('F', n1, n2, B.ptr(1, i), ldb, WORK.ptr(n1 * n2 + 1), n1);
    ierr = Tgsyl<R>('N', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1, b,
                    ldb, B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1, &dscale,
                    DIF.ptr(1), WORK.ptr(n1 * n2 * 2 + 1), lwork - 2 * n1 * n2,
                    iwork);
    // Estimate the reciprocal of norms of "projections" onto left
    // and right eigenspaces.
    rdscal = zero;
    dsum = one;
    Lassq(n1 * n2, work, 1, &rdscal, &dsum);
    *pl = rdscal * std::sqrt(dsum);
    if (*pl == zero) {
      *pl = one;
    } else {
      *pl = dscale / (std::sqrt(dscale * dscale / *pl + *pl) * std::sqrt(*pl));
    }
    rdscal = zero;
    dsum = one;
    Lassq(n1 * n2, WORK.ptr(n1 * n2 + 1), 1, &rdscal, &dsum);
    *pr = rdscal * std::sqrt(dsum);
    if (*pr == zero) {
      *pr = one;
    } else {
      *pr = dscale / (std::sqrt(dscale * dscale / *pr + *pr) * std::sqrt(*pr));
    }
  }
  if (wantd) {
    // Compute estimates of Difu and Difl.
    if (wantd1) {
      n1 = *m;
      n2 = n - *m;
      i = n1 + 1;
      ijb = idifjb;
      // Frobenius norm-based Difu-estimate.
      ierr = Tgsyl<R>('N', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1, b,
                      ldb, B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1, &dscale,
                      DIF.ptr(1), WORK.ptr(2 * n1 * n2 + 1),
                      lwork - 2 * n1 * n2, iwork);
      // Frobenius norm-based Difl-estimate.
      ierr = Tgsyl<R>('N', ijb, n2, n1, A.ptr(i, i), lda, a, lda, work, n2,
                      B.ptr(i, i), ldb, b, ldb, WORK.ptr(n1 * n2 + 1), n2,
                      &dscale, DIF.ptr(2), WORK.ptr(2 * n1 * n2 + 1),
                      lwork - 2 * n1 * n2, iwork);
    } else {
      // Compute 1-norm-based estimates of Difu and Difl using
      // reversed communication with DLACN2. In each step a
      // generalized Sylvester equation or a transposed variant
      // is solved.
      kase = 0;
      n1 = *m;
      n2 = n - *m;
      i = n1 + 1;
      ijb = 0;
      mn2 = 2 * n1 * n2;
      // 1-norm-based estimate of Difu.
    L40:
      Lacn2(mn2, WORK.ptr(mn2 + 1), work, iwork, DIF.ptr(1), &kase, isave);
      if (kase != 0) {
        if (kase == 1) {
          // Solve generalized Sylvester equation.
          ierr = Tgsyl<R>('N', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1,
                          b, ldb, B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1,
                          &dscale, DIF.ptr(1), WORK.ptr(2 * n1 * n2 + 1),
                          lwork - 2 * n1 * n2, iwork);
        } else {
          // Solve the transposed variant.
          ierr = Tgsyl<R>('T', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1,
                          b, ldb, B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1,
                          &dscale, DIF.ptr(1), WORK.ptr(2 * n1 * n2 + 1),
                          lwork - 2 * n1 * n2, iwork);
        }
        goto L40;
      }
      DIF(1) = dscale / DIF(1);
      // 1-norm-based estimate of Difl.
    L50:
      Lacn2(mn2, WORK.ptr(mn2 + 1), work, iwork, DIF.ptr(2), &kase, isave);
      if (kase != 0) {
        if (kase == 1) {
          // Solve generalized Sylvester equation.
          ierr = Tgsyl<R>('N', ijb, n2, n1, A.ptr(i, i), lda, a, lda, work, n2,
                          B.ptr(i, i), ldb, b, ldb, WORK.ptr(n1 * n2 + 1), n2,
                          &dscale, DIF.ptr(2), WORK.ptr(2 * n1 * n2 + 1),
                          lwork - 2 * n1 * n2, iwork);
        } else {
          // Solve the transposed variant.
          ierr = Tgsyl<R>('T', ijb, n2, n1, A.ptr(i, i), lda, a, lda, work, n2,
                          B.ptr(i, i), ldb, b, ldb, WORK.ptr(n1 * n2 + 1), n2,
                          &dscale, DIF.ptr(2), WORK.ptr(2 * n1 * n2 + 1),
                          lwork - 2 * n1 * n2, iwork);
        }
        goto L50;
      }
      DIF(2) = dscale / DIF(2);
    }
  }
L60:
  // Compute generalized eigenvalues of reordered pair (A, B) and
  // normalize the generalized Schur form.
  pair = false;
  for (k = 1; k <= n; ++k) {
    if (pair) {
      pair = false;
    } else {
      if (k < n) {
        if (A(k + 1, k) != zero) {
          pair = true;
        }
      }
      if (pair) {
        // Compute the eigenvalue(s) at position K.
        WORK(1) = A(k, k);
        WORK(2) = A(k + 1, k);
        WORK(3) = A(k, k + 1);
        WORK(4) = A(k + 1, k + 1);
        WORK(5) = B(k, k);
        WORK(6) = B(k + 1, k);
        WORK(7) = B(k, k + 1);
        WORK(8) = B(k + 1, k + 1);
        Lag2<R>(work, 2, WORK.ptr(5), 2, smlnum * eps, BETA.ptr(k),
                BETA.ptr(k + 1), ALPHAR.ptr(k), ALPHAR.ptr(k + 1),
                ALPHAI.ptr(k));
        ALPHAI(k + 1) = -ALPHAI(k);
      } else {
        if (std::copysign(one, B(k, k)) < zero) {
          // If B(K,K) is negative, make it positive
          for (i = 1; i <= n; ++i) {
            A(k, i) = -A(k, i);
            B(k, i) = -B(k, i);
            if (wantq) Q(i, k) = -Q(i, k);
          }
        }
        ALPHAR(k) = A(k, k);
        ALPHAI(k) = zero;
        BETA(k) = B(k, k);
      }
    }
  }
  WORK(1) = WorkSize<T>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

// xTGSEN (complex T): as the real routine for the upper triangular
// generalized Schur form and the eigenvalues ALPHA / BETA; WORK has LWORK
// >= max(1, 2*M*(N-M)+1) elements for IJOB = 1, 2, 4, max(1, 4*M*(N-M))
// for IJOB = 3, 5, 1 otherwise; IWORK has LIWORK >= max(1, N+2) elements,
// max(1, 2*M*(N-M), N+2) for IJOB = 3, 5.
//
// Deviations: the minimum LWORK for IJOB = 1, 2, 4 (see the real routine),
// and: the reference C/ZTGSEN pass B11 and B22 to the transposed
// solve of the 1-norm-based estimate of Difl (IJOB = 3, 5) in the wrong
// order (D = B11, E = B22, where the other three solves of that estimate
// and DTGSEN have D = B22, E = B11), so that the estimator does not apply
// the transpose of the operator it estimates; the port passes them as
// DTGSEN does.
template <typename T>
blasint Tgsen(blasint ijob, bool wantq, bool wantz, const blasint* select,
              blasint n, T* a, blasint lda, T* b, blasint ldb, T* alpha,
              T* beta, T* q, blasint ldq, T* z, blasint ldz, blasint* m,
              RealT<T>* pl, RealT<T>* pr, RealT<T>* dif, T* work, blasint lwork,
              blasint* iwork, blasint liwork) {
  using R = RealT<T>;

  const blasint idifjb = 3;
  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> A(a, lda);
  const Vector<T> ALPHA(alpha);
  const Matrix<T> B(b, ldb);
  const Vector<T> BETA(beta);
  const Vector<R> DIF(dif);
  blasint isave[3];
  const Vector<blasint> IWORK(iwork);
  const Matrix<T> Q(q, ldq);
  const Vector<const blasint> SELECT(select);
  const Vector<T> WORK(work);
  const Matrix<T> Z(z, ldz);
  blasint i, ierr, ijb, k, kase, ks, liwmin, lwmin, mn2, n1, n2;
  bool lquery, swap, wantd, wantd1, wantd2, wantp;
  R dsum, rdscal, safmin;
  // Set by xTGSYL where it is read; initialized for clang-tidy.
  R dscale = zero;
  T temp1, temp2;
  blasint info = 0;
  lquery = (lwork == -1 || liwork == -1);
  if (ijob < 0 || ijob > 5) {
    info = -1;
  } else if (n < 0) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if (ldq < 1 || (wantq && ldq < n)) {
    info = -13;
  } else if (ldz < 1 || (wantz && ldz < n)) {
    info = -15;
  }
  if (info != 0) {
    Xerbla<T>("TGSEN", -info);
    return info;
  }
  ierr = 0;
  wantp = ijob == 1 || ijob >= 4;
  wantd1 = ijob == 2 || ijob == 4;
  wantd2 = ijob == 3 || ijob == 5;
  wantd = wantd1 || wantd2;
  // Set M to the dimension of the specified pair of deflating
  // subspaces.
  *m = 0;
  if (!lquery || ijob != 0) {
    for (k = 1; k <= n; ++k) {
      ALPHA(k) = A(k, k);
      BETA(k) = B(k, k);
      if (k < n) {
        if (SELECT(k)) *m = *m + 1;
      } else {
        if (SELECT(n)) *m = *m + 1;
      }
    }
  }
  if (ijob == 1 || ijob == 2 || ijob == 4) {
    // Deviation (see above): + 1.
    lwmin = Max<blasint>(1, 2 * *m * (n - *m) + 1);
    liwmin = Max<blasint>(1, n + 2);
  } else if (ijob == 3 || ijob == 5) {
    lwmin = Max<blasint>(1, 4 * *m * (n - *m));
    liwmin = Max<blasint>(1, 2 * *m * (n - *m), n + 2);
  } else {
    lwmin = 1;
    liwmin = 1;
  }
  WORK(1) = WorkSize<T>(lwmin);
  IWORK(1) = liwmin;
  if (lwork < lwmin && !lquery) {
    info = -21;
  } else if (liwork < liwmin && !lquery) {
    info = -23;
  }
  if (info != 0) {
    Xerbla<T>("TGSEN", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible.
  if (*m == n || *m == 0) {
    if (wantp) {
      *pl = one;
      *pr = one;
    }
    if (wantd) {
      dscale = zero;
      dsum = one;
      for (i = 1; i <= n; ++i) {
        Lassq(n, A.ptr(1, i), 1, &dscale, &dsum);
        Lassq(n, B.ptr(1, i), 1, &dscale, &dsum);
      }
      DIF(1) = dscale * std::sqrt(dsum);
      DIF(2) = DIF(1);
    }
    goto L70;
  }
  // Get machine constant
  safmin = Lamch<R>('S');
  // Collect the selected blocks at the top-left corner of (A, B).
  ks = 0;
  for (k = 1; k <= n; ++k) {
    swap = SELECT(k);
    if (swap) {
      ks = ks + 1;
      // Swap the K-th block to position KS. Compute unitary Q
      // and Z that will swap adjacent diagonal blocks in (A, B).
      if (k != ks)
        ierr = Tgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, k, &ks);
      if (ierr > 0) {
        // Swap is rejected: exit.
        info = 1;
        if (wantp) {
          *pl = zero;
          *pr = zero;
        }
        if (wantd) {
          DIF(1) = zero;
          DIF(2) = zero;
        }
        goto L70;
      }
    }
  }
  if (wantp) {
    // Solve generalized Sylvester equation for R and L:
    // A11 * R - L * A22 = A12
    // B11 * R - L * B22 = B12
    n1 = *m;
    n2 = n - *m;
    i = n1 + 1;
    Lacpy('F', n1, n2, A.ptr(1, i), lda, work, n1);
    Lacpy('F', n1, n2, B.ptr(1, i), ldb, WORK.ptr(n1 * n2 + 1), n1);
    ijb = 0;
    ierr =
        Tgsyl('N', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1, b, ldb,
              B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1, &dscale, DIF.ptr(1),
              WORK.ptr(n1 * n2 * 2 + 1), lwork - 2 * n1 * n2, iwork);
    // Estimate the reciprocal of norms of "projections" onto
    // left and right eigenspaces
    rdscal = zero;
    dsum = one;
    Lassq(n1 * n2, work, 1, &rdscal, &dsum);
    *pl = rdscal * std::sqrt(dsum);
    if (*pl == zero) {
      *pl = one;
    } else {
      *pl = dscale / (std::sqrt(dscale * dscale / *pl + *pl) * std::sqrt(*pl));
    }
    rdscal = zero;
    dsum = one;
    Lassq(n1 * n2, WORK.ptr(n1 * n2 + 1), 1, &rdscal, &dsum);
    *pr = rdscal * std::sqrt(dsum);
    if (*pr == zero) {
      *pr = one;
    } else {
      *pr = dscale / (std::sqrt(dscale * dscale / *pr + *pr) * std::sqrt(*pr));
    }
  }
  if (wantd) {
    // Compute estimates Difu and Difl.
    if (wantd1) {
      n1 = *m;
      n2 = n - *m;
      i = n1 + 1;
      ijb = idifjb;
      // Frobenius norm-based Difu estimate.
      ierr = Tgsyl('N', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1, b, ldb,
                   B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1, &dscale,
                   DIF.ptr(1), WORK.ptr(n1 * n2 * 2 + 1), lwork - 2 * n1 * n2,
                   iwork);
      // Frobenius norm-based Difl estimate.
      ierr = Tgsyl('N', ijb, n2, n1, A.ptr(i, i), lda, a, lda, work, n2,
                   B.ptr(i, i), ldb, b, ldb, WORK.ptr(n1 * n2 + 1), n2, &dscale,
                   DIF.ptr(2), WORK.ptr(n1 * n2 * 2 + 1), lwork - 2 * n1 * n2,
                   iwork);
    } else {
      // Compute 1-norm-based estimates of Difu and Difl using
      // reversed communication with ZLACN2. In each step a
      // generalized Sylvester equation or a transposed variant
      // is solved.
      kase = 0;
      n1 = *m;
      n2 = n - *m;
      i = n1 + 1;
      ijb = 0;
      mn2 = 2 * n1 * n2;
      // 1-norm-based estimate of Difu.
    L40:
      Lacn2(mn2, WORK.ptr(mn2 + 1), work, DIF.ptr(1), &kase, isave);
      if (kase != 0) {
        if (kase == 1) {
          // Solve generalized Sylvester equation
          ierr = Tgsyl('N', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1, b,
                       ldb, B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1,
                       &dscale, DIF.ptr(1), WORK.ptr(n1 * n2 * 2 + 1),
                       lwork - 2 * n1 * n2, iwork);
        } else {
          // Solve the transposed variant.
          ierr = Tgsyl('C', ijb, n1, n2, a, lda, A.ptr(i, i), lda, work, n1, b,
                       ldb, B.ptr(i, i), ldb, WORK.ptr(n1 * n2 + 1), n1,
                       &dscale, DIF.ptr(1), WORK.ptr(n1 * n2 * 2 + 1),
                       lwork - 2 * n1 * n2, iwork);
        }
        goto L40;
      }
      DIF(1) = dscale / DIF(1);
      // 1-norm-based estimate of Difl.
    L50:
      Lacn2(mn2, WORK.ptr(mn2 + 1), work, DIF.ptr(2), &kase, isave);
      if (kase != 0) {
        if (kase == 1) {
          // Solve generalized Sylvester equation
          ierr = Tgsyl('N', ijb, n2, n1, A.ptr(i, i), lda, a, lda, work, n2,
                       B.ptr(i, i), ldb, b, ldb, WORK.ptr(n1 * n2 + 1), n2,
                       &dscale, DIF.ptr(2), WORK.ptr(n1 * n2 * 2 + 1),
                       lwork - 2 * n1 * n2, iwork);
        } else {
          // Solve the transposed variant.
          ierr = Tgsyl('C', ijb, n2, n1, A.ptr(i, i), lda, a, lda, work, n2,
                       B.ptr(i, i), ldb, b, ldb, WORK.ptr(n1 * n2 + 1), n2,
                       &dscale, DIF.ptr(2), WORK.ptr(n1 * n2 * 2 + 1),
                       lwork - 2 * n1 * n2, iwork);
        }
        goto L50;
      }
      DIF(2) = dscale / DIF(2);
    }
  }
  // If B(K,K) is complex, make it real and positive (normalization
  // of the generalized Schur form) and Store the generalized
  // eigenvalues of reordered pair (A, B)
  for (k = 1; k <= n; ++k) {
    dscale = std::abs(B(k, k));
    if (dscale > safmin) {
      temp1 = Conj(B(k, k) / dscale);
      temp2 = B(k, k) / dscale;
      B(k, k) = dscale;
      Scal(n - k, temp1, B.ptr(k, k + 1), ldb);
      Scal(n - k + 1, temp1, A.ptr(k, k), lda);
      if (wantq) Scal(n, temp2, Q.ptr(1, k), 1);
    } else {
      B(k, k) = T(zero);
    }
    ALPHA(k) = A(k, k);
    BETA(k) = B(k, k);
  }
L70:
  WORK(1) = WorkSize<T>(lwmin);
  IWORK(1) = liwmin;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template blasint Getc2<T>(blasint, T*, blasint, blasint*, blasint*);         \
  template void Gesc2<T>(blasint, const T*, blasint, T*, const blasint*,       \
                         const blasint*, RealT<T>*);                           \
  template void Latdf<T>(blasint, blasint, const T*, blasint, T*, RealT<T>*,   \
                         RealT<T>*, const blasint*, const blasint*);           \
  template blasint Tgsyl<T>(char, blasint, blasint, blasint, const T*,         \
                            blasint, const T*, blasint, T*, blasint, const T*, \
                            blasint, const T*, blasint, T*, blasint,           \
                            RealT<T>*, RealT<T>*, T*, blasint, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template blasint Tgsy2<R>(char, blasint, blasint, blasint, const R*,         \
                            blasint, const R*, blasint, R*, blasint, const R*, \
                            blasint, const R*, blasint, R*, blasint, R*, R*,   \
                            R*, blasint*, blasint*);                           \
  template void Lagv2<R>(R*, blasint, R*, blasint, R*, R*, R*, R*, R*, R*,     \
                         R*);                                                  \
  template blasint Tgex2<R>(bool, bool, blasint, R*, blasint, R*, blasint, R*, \
                            blasint, R*, blasint, blasint, blasint, blasint,   \
                            R*, blasint);                                      \
  template blasint Tgexc<R>(bool, bool, blasint, R*, blasint, R*, blasint, R*, \
                            blasint, R*, blasint, blasint*, blasint*, R*,      \
                            blasint);                                          \
  template blasint Tgsen<R>(blasint, bool, bool, const blasint*, blasint, R*,  \
                            blasint, R*, blasint, R*, R*, R*, R*, blasint, R*, \
                            blasint, blasint*, R*, R*, R*, R*, blasint,        \
                            blasint*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                  \
  template blasint Tgsy2<T>(char, blasint, blasint, blasint, const T*,         \
                            blasint, const T*, blasint, T*, blasint, const T*, \
                            blasint, const T*, blasint, T*, blasint,           \
                            RealT<T>*, RealT<T>*, RealT<T>*);                  \
  template blasint Tgex2<T>(bool, bool, blasint, T*, blasint, T*, blasint, T*, \
                            blasint, T*, blasint, blasint);                    \
  template blasint Tgexc<T>(bool, bool, blasint, T*, blasint, T*, blasint, T*, \
                            blasint, T*, blasint, blasint, blasint*);          \
  template blasint Tgsen<T>(blasint, bool, bool, const blasint*, blasint, T*,  \
                            blasint, T*, blasint, T*, T*, T*, blasint, T*,     \
                            blasint, blasint*, RealT<T>*, RealT<T>*,           \
                            RealT<T>*, T*, blasint, blasint*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// WANTQ and WANTZ are LOGICALs, SELECT a LOGICAL array (nonzero is true).
#define HWYBLAS_LAPACK_TGSEN(prefix, T, R)                                     \
  HWYBLAS_API void prefix##getc2_(const hwyblas_int* n, R* a,                  \
                                  const hwyblas_int* lda, hwyblas_int* ipiv,   \
                                  hwyblas_int* jpiv, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Getc2(*n, As<T>(a), *lda, ipiv, jpiv);            \
  }                                                                            \
  HWYBLAS_API void prefix##gesc2_(                                             \
      const hwyblas_int* n, const R* a, const hwyblas_int* lda, R* rhs,        \
      const hwyblas_int* ipiv, const hwyblas_int* jpiv, R* scale) {            \
    hwyblas::lapack::Gesc2(*n, As<const T>(a), *lda, As<T>(rhs), ipiv, jpiv,   \
                           scale);                                             \
  }                                                                            \
  HWYBLAS_API void prefix##latdf_(                                             \
      const hwyblas_int* ijob, const hwyblas_int* n, const R* z,               \
      const hwyblas_int* ldz, R* rhs, R* rdsum, R* rdscal,                     \
      const hwyblas_int* ipiv, const hwyblas_int* jpiv) {                      \
    hwyblas::lapack::Latdf(*ijob, *n, As<const T>(z), *ldz, As<T>(rhs), rdsum, \
                           rdscal, ipiv, jpiv);                                \
  }                                                                            \
  HWYBLAS_API void prefix##tgsyl_(                                             \
      const char* trans, const hwyblas_int* ijob, const hwyblas_int* m,        \
      const hwyblas_int* n, const R* a, const hwyblas_int* lda, const R* b,    \
      const hwyblas_int* ldb, R* c, const hwyblas_int* ldc, const R* d,        \
      const hwyblas_int* ldd, const R* e, const hwyblas_int* lde, R* f,        \
      const hwyblas_int* ldf, R* scale, R* dif, R* work,                       \
      const hwyblas_int* lwork, hwyblas_int* iwork, hwyblas_int* info) {       \
    *info = hwyblas::lapack::Tgsyl(                                            \
        *trans, *ijob, *m, *n, As<const T>(a), *lda, As<const T>(b), *ldb,     \
        As<T>(c), *ldc, As<const T>(d), *ldd, As<const T>(e), *lde, As<T>(f),  \
        *ldf, scale, dif, As<T>(work), *lwork, iwork);                         \
  }

HWYBLAS_LAPACK_TGSEN(s, float, float)
HWYBLAS_LAPACK_TGSEN(d, double, double)
HWYBLAS_LAPACK_TGSEN(c, c32, float)
HWYBLAS_LAPACK_TGSEN(z, c64, double)

#undef HWYBLAS_LAPACK_TGSEN

#define HWYBLAS_LAPACK_TGSEN_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##tgsy2_(                                             \
      const char* trans, const hwyblas_int* ijob, const hwyblas_int* m,        \
      const hwyblas_int* n, const R* a, const hwyblas_int* lda, const R* b,    \
      const hwyblas_int* ldb, R* c, const hwyblas_int* ldc, const R* d,        \
      const hwyblas_int* ldd, const R* e, const hwyblas_int* lde, R* f,        \
      const hwyblas_int* ldf, R* scale, R* rdsum, R* rdscal,                   \
      hwyblas_int* iwork, hwyblas_int* pq, hwyblas_int* info) {                \
    *info = hwyblas::lapack::Tgsy2(*trans, *ijob, *m, *n, a, *lda, b, *ldb, c, \
                                   *ldc, d, *ldd, e, *lde, f, *ldf, scale,     \
                                   rdsum, rdscal, iwork, pq);                  \
  }                                                                            \
  HWYBLAS_API void prefix##lagv2_(                                             \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* alphar,   \
      R* alphai, R* beta, R* csl, R* snl, R* csr, R* snr) {                    \
    hwyblas::lapack::Lagv2(a, *lda, b, *ldb, alphar, alphai, beta, csl, snl,   \
                           csr, snr);                                          \
  }                                                                            \
  HWYBLAS_API void prefix##tgex2_(                                             \
      const hwyblas_int* wantq, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* q, const hwyblas_int* ldq, R* z,              \
      const hwyblas_int* ldz, const hwyblas_int* j1, const hwyblas_int* n1,    \
      const hwyblas_int* n2, R* work, const hwyblas_int* lwork,                \
      hwyblas_int* info) {                                                     \
    *info =                                                                    \
        hwyblas::lapack::Tgex2(*wantq != 0, *wantz != 0, *n, a, *lda, b, *ldb, \
                               q, *ldq, z, *ldz, *j1, *n1, *n2, work, *lwork); \
  }                                                                            \
  HWYBLAS_API void prefix##tgexc_(                                             \
      const hwyblas_int* wantq, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* q, const hwyblas_int* ldq, R* z,              \
      const hwyblas_int* ldz, hwyblas_int* ifst, hwyblas_int* ilst, R* work,   \
      const hwyblas_int* lwork, hwyblas_int* info) {                           \
    *info =                                                                    \
        hwyblas::lapack::Tgexc(*wantq != 0, *wantz != 0, *n, a, *lda, b, *ldb, \
                               q, *ldq, z, *ldz, ifst, ilst, work, *lwork);    \
  }                                                                            \
  HWYBLAS_API void prefix##tgsen_(                                             \
      const hwyblas_int* ijob, const hwyblas_int* wantq,                       \
      const hwyblas_int* wantz, const hwyblas_int* select,                     \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* alphar, R* alphai, R* beta, R* q,             \
      const hwyblas_int* ldq, R* z, const hwyblas_int* ldz, hwyblas_int* m,    \
      R* pl, R* pr, R* dif, R* work, const hwyblas_int* lwork,                 \
      hwyblas_int* iwork, const hwyblas_int* liwork, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Tgsen(*ijob, *wantq != 0, *wantz != 0, select,    \
                                   *n, a, *lda, b, *ldb, alphar, alphai, beta, \
                                   q, *ldq, z, *ldz, m, pl, pr, dif, work,     \
                                   *lwork, iwork, *liwork);                    \
  }

HWYBLAS_LAPACK_TGSEN_REAL(s, float)
HWYBLAS_LAPACK_TGSEN_REAL(d, double)

#undef HWYBLAS_LAPACK_TGSEN_REAL

#define HWYBLAS_LAPACK_TGSEN_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##tgsy2_(                                             \
      const char* trans, const hwyblas_int* ijob, const hwyblas_int* m,        \
      const hwyblas_int* n, const R* a, const hwyblas_int* lda, const R* b,    \
      const hwyblas_int* ldb, R* c, const hwyblas_int* ldc, const R* d,        \
      const hwyblas_int* ldd, const R* e, const hwyblas_int* lde, R* f,        \
      const hwyblas_int* ldf, R* scale, R* rdsum, R* rdscal,                   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Tgsy2(*trans, *ijob, *m, *n, As<const T>(a),      \
                                   *lda, As<const T>(b), *ldb, As<T>(c), *ldc, \
                                   As<const T>(d), *ldd, As<const T>(e), *lde, \
                                   As<T>(f), *ldf, scale, rdsum, rdscal);      \
  }                                                                            \
  HWYBLAS_API void prefix##tgex2_(                                             \
      const hwyblas_int* wantq, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* q, const hwyblas_int* ldq, R* z,              \
      const hwyblas_int* ldz, const hwyblas_int* j1, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Tgex2(*wantq != 0, *wantz != 0, *n, As<T>(a),     \
                                   *lda, As<T>(b), *ldb, As<T>(q), *ldq,       \
                                   As<T>(z), *ldz, *j1);                       \
  }                                                                            \
  HWYBLAS_API void prefix##tgexc_(                                             \
      const hwyblas_int* wantq, const hwyblas_int* wantz,                      \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* q, const hwyblas_int* ldq, R* z,              \
      const hwyblas_int* ldz, const hwyblas_int* ifst, hwyblas_int* ilst,      \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Tgexc(*wantq != 0, *wantz != 0, *n, As<T>(a),     \
                                   *lda, As<T>(b), *ldb, As<T>(q), *ldq,       \
                                   As<T>(z), *ldz, *ifst, ilst);               \
  }                                                                            \
  HWYBLAS_API void prefix##tgsen_(                                             \
      const hwyblas_int* ijob, const hwyblas_int* wantq,                       \
      const hwyblas_int* wantz, const hwyblas_int* select,                     \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* alpha, R* beta, R* q, const hwyblas_int* ldq, \
      R* z, const hwyblas_int* ldz, hwyblas_int* m, R* pl, R* pr, R* dif,      \
      R* work, const hwyblas_int* lwork, hwyblas_int* iwork,                   \
      const hwyblas_int* liwork, hwyblas_int* info) {                          \
    *info = hwyblas::lapack::Tgsen(                                            \
        *ijob, *wantq != 0, *wantz != 0, select, *n, As<T>(a), *lda, As<T>(b), \
        *ldb, As<T>(alpha), As<T>(beta), As<T>(q), *ldq, As<T>(z), *ldz, m,    \
        pl, pr, dif, As<T>(work), *lwork, iwork, *liwork);                     \
  }

HWYBLAS_LAPACK_TGSEN_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_TGSEN_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_TGSEN_COMPLEX

}  // extern "C"
