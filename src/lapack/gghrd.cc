// Balancing of a pair of general matrices (A, B), xGGBAL, and its back
// transformation, xGGBAK; reduction of (A, B) to generalized upper
// Hessenberg form by orthogonal (unitary) transformations, Q**H * A * Z = H
// upper Hessenberg and Q**H * B * Z = T upper triangular: xGGHRD. Each
// template is the real routine for real T and the complex routine for
// complex T; they differ in conjugations, which the templates do for both
// (they change nothing for real T).
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

// xGGBAL: balances the pair of N-by-N matrices (A, B). JOB = 'N': nothing;
// 'P': permutes (A, B) to isolate eigenvalues, so that A(i,j) = B(i,j) = 0
// for i > j and (j < ILO or i > IHI); 'S': scales rows ILO:IHI and columns
// ILO:IHI with powers of 10, to make the magnitudes of the elements of A
// and B in that block as close to 1 as possible; 'B': both. LSCALE(j) and
// RSCALE(j) are the left and right permutation indices (j outside
// ILO:IHI) or scale factors (j in ILO:IHI). WORK has 6*N elements for JOB
// = 'S' or 'B', at least 1 otherwise.
//
// The real routine's test of a zero element (tested after being copied to
// TA / TB) is the complex routine's; the results are the same. 10**IR is
// computed as Fortran's real**integer (Powi).
template <typename T>
blasint Ggbal(char job, blasint n, T* a, blasint lda, T* b, blasint ldb,
              blasint* ilo, blasint* ihi, RealT<T>* lscale, RealT<T>* rscale,
              RealT<T>* work) {
  using R = RealT<T>;
  const R zero = R(0);
  const R half = R(0.5e+0);
  const R one = R(1);
  const R three = R(3);
  const R sclfac = R(10);
  const T czero = T(0);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<R> LSCALE(lscale);
  const Vector<R> RSCALE(rscale);
  const Vector<R> WORK(work);
  blasint i, icab, iflow, ip1, ir, irab, it, j, jc, jp1, k, kount, l, lcab, lm1,
      lrab, lsfmax, lsfmin, m, nr, nrp2;
  R alpha, basl, beta, cab, cmax, coef, coef2, coef5, cor, ew, ewc, gamma, rab,
      sfmax, sfmin, sum, t, ta, tb, tc;
  // Set in the first iteration, read from the second on.
  R pgamma = zero;
  // Set wherever they are read (the GO TOs reach their uses only after
  // their assignments); initialized for the compiler.
  ip1 = jp1 = lm1 = m = iflow = 0;
  // Test the input parameters
  blasint info = 0;
  if (!Lsame(job, 'N') && !Lsame(job, 'P') && !Lsame(job, 'S') &&
      !Lsame(job, 'B')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("GGBAL", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) {
    *ilo = 1;
    *ihi = n;
    return info;
  }
  if (n == 1) {
    *ilo = 1;
    *ihi = n;
    LSCALE(1) = one;
    RSCALE(1) = one;
    return info;
  }
  if (Lsame(job, 'N')) {
    *ilo = 1;
    *ihi = n;
    for (i = 1; i <= n; ++i) {
      LSCALE(i) = one;
      RSCALE(i) = one;
    }
    return info;
  }
  k = 1;
  l = n;
  if (Lsame(job, 'S')) goto L190;
  goto L30;
  // Permute the matrices A and B to isolate the eigenvalues.
  // Find row with one nonzero in columns 1 through L
L20:
  l = lm1;
  if (l != 1) goto L30;
  RSCALE(1) = one;
  LSCALE(1) = one;
  goto L190;
L30:
  lm1 = l - 1;
  for (i = l; i >= 1; --i) {
    for (j = 1; j <= lm1; ++j) {
      jp1 = j + 1;
      if (A(i, j) != czero || B(i, j) != czero) goto L50;
    }
    j = l;
    goto L70;
  L50:
    for (j = jp1; j <= l; ++j) {
      if (A(i, j) != czero || B(i, j) != czero) goto L80;
    }
    j = jp1 - 1;
  L70:
    m = l;
    iflow = 1;
    goto L160;
  L80:;
  }
  goto L100;
  // Find column with one nonzero in rows K through N
L90:
  k = k + 1;
L100:
  for (j = k; j <= l; ++j) {
    for (i = k; i <= lm1; ++i) {
      ip1 = i + 1;
      if (A(i, j) != czero || B(i, j) != czero) goto L120;
    }
    i = l;
    goto L140;
  L120:
    for (i = ip1; i <= l; ++i) {
      if (A(i, j) != czero || B(i, j) != czero) goto L150;
    }
    i = ip1 - 1;
  L140:
    m = k;
    iflow = 2;
    goto L160;
  L150:;
  }
  goto L190;
  // Permute rows M and I
L160:
  LSCALE(m) = R(i);
  if (i == m) goto L170;
  Swap(n - k + 1, A.ptr(i, k), lda, A.ptr(m, k), lda);
  Swap(n - k + 1, B.ptr(i, k), ldb, B.ptr(m, k), ldb);
  // Permute columns M and J
L170:
  RSCALE(m) = R(j);
  if (j == m) goto L180;
  Swap(l, A.ptr(1, j), 1, A.ptr(1, m), 1);
  Swap(l, B.ptr(1, j), 1, B.ptr(1, m), 1);
L180:
  switch (iflow) {
    case 1:
      goto L20;
    case 2:
      goto L90;
    default:  // Out of range: no jump.
      break;
  }
L190:
  *ilo = k;
  *ihi = l;
  if (Lsame(job, 'P')) {
    for (i = *ilo; i <= *ihi; ++i) {
      LSCALE(i) = one;
      RSCALE(i) = one;
    }
    return info;
  }
  if (*ilo == *ihi) return info;
  // Balance the submatrix in rows ILO to IHI.
  nr = *ihi - *ilo + 1;
  for (i = *ilo; i <= *ihi; ++i) {
    RSCALE(i) = zero;
    LSCALE(i) = zero;
    WORK(i) = zero;
    WORK(i + n) = zero;
    WORK(i + 2 * n) = zero;
    WORK(i + 3 * n) = zero;
    WORK(i + 4 * n) = zero;
    WORK(i + 5 * n) = zero;
  }
  // Compute right side vector in resulting linear equations
  basl = std::log10(sclfac);
  for (i = *ilo; i <= *ihi; ++i) {
    for (j = *ilo; j <= *ihi; ++j) {
      if (A(i, j) == czero) {
        ta = zero;
        goto L210;
      }
      ta = std::log10(Cabs1(A(i, j))) / basl;
    L210:
      if (B(i, j) == czero) {
        tb = zero;
        goto L220;
      }
      tb = std::log10(Cabs1(B(i, j))) / basl;
    L220:
      WORK(i + 4 * n) = WORK(i + 4 * n) - ta - tb;
      WORK(j + 5 * n) = WORK(j + 5 * n) - ta - tb;
    }
  }
  coef = one / R(2 * nr);
  coef2 = coef * coef;
  coef5 = half * coef2;
  nrp2 = nr + 2;
  beta = zero;
  it = 1;
  // Start generalized conjugate gradient iteration
L250:
  gamma = Dotu<R>(nr, WORK.ptr(*ilo + 4 * n), 1, WORK.ptr(*ilo + 4 * n), 1) +
          Dotu<R>(nr, WORK.ptr(*ilo + 5 * n), 1, WORK.ptr(*ilo + 5 * n), 1);
  ew = zero;
  ewc = zero;
  for (i = *ilo; i <= *ihi; ++i) {
    ew = ew + WORK(i + 4 * n);
    ewc = ewc + WORK(i + 5 * n);
  }
  gamma = coef * gamma - coef2 * ((ew * ew) + (ewc * ewc)) -
          coef5 * ((ew - ewc) * (ew - ewc));
  if (gamma == zero) goto L350;
  if (it != 1) beta = gamma / pgamma;
  t = coef5 * (ewc - three * ew);
  tc = coef5 * (ew - three * ewc);
  Scal<R>(nr, beta, WORK.ptr(*ilo), 1);
  Scal<R>(nr, beta, WORK.ptr(*ilo + n), 1);
  Axpy<R>(nr, coef, WORK.ptr(*ilo + 4 * n), 1, WORK.ptr(*ilo + n), 1);
  Axpy<R>(nr, coef, WORK.ptr(*ilo + 5 * n), 1, WORK.ptr(*ilo), 1);
  for (i = *ilo; i <= *ihi; ++i) {
    WORK(i) = WORK(i) + tc;
    WORK(i + n) = WORK(i + n) + t;
  }
  // Apply matrix to vector
  for (i = *ilo; i <= *ihi; ++i) {
    kount = 0;
    sum = zero;
    for (j = *ilo; j <= *ihi; ++j) {
      if (A(i, j) == czero) goto L280;
      kount = kount + 1;
      sum = sum + WORK(j);
    L280:
      if (B(i, j) == czero) continue;
      kount = kount + 1;
      sum = sum + WORK(j);
    }
    WORK(i + 2 * n) = R(kount) * WORK(i + n) + sum;
  }
  for (j = *ilo; j <= *ihi; ++j) {
    kount = 0;
    sum = zero;
    for (i = *ilo; i <= *ihi; ++i) {
      if (A(i, j) == czero) goto L310;
      kount = kount + 1;
      sum = sum + WORK(i + n);
    L310:
      if (B(i, j) == czero) continue;
      kount = kount + 1;
      sum = sum + WORK(i + n);
    }
    WORK(j + 3 * n) = R(kount) * WORK(j) + sum;
  }
  sum = Dotu<R>(nr, WORK.ptr(*ilo + n), 1, WORK.ptr(*ilo + 2 * n), 1) +
        Dotu<R>(nr, WORK.ptr(*ilo), 1, WORK.ptr(*ilo + 3 * n), 1);
  alpha = gamma / sum;
  // Determine correction to current iteration
  cmax = zero;
  for (i = *ilo; i <= *ihi; ++i) {
    cor = alpha * WORK(i + n);
    if (std::abs(cor) > cmax) cmax = std::abs(cor);
    LSCALE(i) = LSCALE(i) + cor;
    cor = alpha * WORK(i);
    if (std::abs(cor) > cmax) cmax = std::abs(cor);
    RSCALE(i) = RSCALE(i) + cor;
  }
  if (cmax < half) goto L350;
  Axpy<R>(nr, -alpha, WORK.ptr(*ilo + 2 * n), 1, WORK.ptr(*ilo + 4 * n), 1);
  Axpy<R>(nr, -alpha, WORK.ptr(*ilo + 3 * n), 1, WORK.ptr(*ilo + 5 * n), 1);
  pgamma = gamma;
  it = it + 1;
  if (it <= nrp2) goto L250;
  // End generalized conjugate gradient iteration
L350:
  sfmin = Lamch<R>('S');
  sfmax = one / sfmin;
  lsfmin = static_cast<blasint>(std::log10(sfmin) / basl + one);
  lsfmax = static_cast<blasint>(std::log10(sfmax) / basl);
  for (i = *ilo; i <= *ihi; ++i) {
    irab = Iamax(n - *ilo + 1, A.ptr(i, *ilo), lda);
    rab = std::abs(A(i, irab + *ilo - 1));
    irab = Iamax(n - *ilo + 1, B.ptr(i, *ilo), ldb);
    rab = Max(rab, std::abs(B(i, irab + *ilo - 1)));
    lrab = static_cast<blasint>(std::log10(rab + sfmin) / basl + one);
    ir = static_cast<blasint>(LSCALE(i) + std::copysign(half, LSCALE(i)));
    ir = Min<blasint>(Max<blasint>(ir, lsfmin), lsfmax, lsfmax - lrab);
    LSCALE(i) = Powi(sclfac, ir);
    icab = Iamax(*ihi, A.ptr(1, i), 1);
    cab = std::abs(A(icab, i));
    icab = Iamax(*ihi, B.ptr(1, i), 1);
    cab = Max(cab, std::abs(B(icab, i)));
    lcab = static_cast<blasint>(std::log10(cab + sfmin) / basl + one);
    jc = static_cast<blasint>(RSCALE(i) + std::copysign(half, RSCALE(i)));
    jc = Min<blasint>(Max<blasint>(jc, lsfmin), lsfmax, lsfmax - lcab);
    RSCALE(i) = Powi(sclfac, jc);
  }
  // Row scaling of matrices A and B
  for (i = *ilo; i <= *ihi; ++i) {
    Rscal(n - *ilo + 1, LSCALE(i), A.ptr(i, *ilo), lda);
    Rscal(n - *ilo + 1, LSCALE(i), B.ptr(i, *ilo), ldb);
  }
  // Column scaling of matrices A and B
  for (j = *ilo; j <= *ihi; ++j) {
    Rscal(*ihi, RSCALE(j), A.ptr(1, j), 1);
    Rscal(*ihi, RSCALE(j), B.ptr(1, j), 1);
  }
  return info;
}

// xGGBAK: back-transforms the M eigenvectors in the N-by-M matrix V of the
// pair balanced by xGGBAL (JOB, ILO, IHI, LSCALE, RSCALE as returned by
// it) into those of the original pair: right eigenvectors (SIDE = 'R',
// with RSCALE) or left ones (SIDE = 'L', with LSCALE).
template <typename T>
blasint Ggbak(char job, char side, blasint n, blasint ilo, blasint ihi,
              const RealT<T>* lscale, const RealT<T>* rscale, blasint m, T* v,
              blasint ldv) {
  using R = RealT<T>;
  const Vector<const R> LSCALE(lscale);
  const Vector<const R> RSCALE(rscale);
  const Matrix<T> V(v, ldv);
  blasint i, k;
  bool leftv, rightv;
  // Test the input parameters
  rightv = Lsame(side, 'R');
  leftv = Lsame(side, 'L');
  blasint info = 0;
  // NOLINTBEGIN(bugprone-branch-clone): as LAPACK
  if (!Lsame(job, 'N') && !Lsame(job, 'P') && !Lsame(job, 'S') &&
      !Lsame(job, 'B')) {
    info = -1;
  } else if (!rightv && !leftv) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1) {
    info = -4;
  } else if (n == 0 && ihi == 0 && ilo != 1) {
    info = -4;
  } else if (n > 0 && (ihi < ilo || ihi > Max<blasint>(1, n))) {
    info = -5;
  } else if (n == 0 && ilo == 1 && ihi != 0) {
    info = -5;
  } else if (m < 0) {
    info = -8;
  } else if (ldv < Max<blasint>(1, n)) {
    info = -10;
  }
  // NOLINTEND(bugprone-branch-clone)
  if (info != 0) {
    Xerbla<T>("GGBAK", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  if (m == 0) return info;
  if (Lsame(job, 'N')) return info;
  if (ilo == ihi) goto L30;
  // Backward balance
  if (Lsame(job, 'S') || Lsame(job, 'B')) {
    // Backward transformation on right eigenvectors
    if (rightv) {
      for (i = ilo; i <= ihi; ++i) {
        Rscal(m, RSCALE(i), V.ptr(i, 1), ldv);
      }
    }
    // Backward transformation on left eigenvectors
    if (leftv) {
      for (i = ilo; i <= ihi; ++i) {
        Rscal(m, LSCALE(i), V.ptr(i, 1), ldv);
      }
    }
  }
  // Backward permutation
L30:
  if (Lsame(job, 'P') || Lsame(job, 'B')) {
    // Backward permutation on right eigenvectors
    if (rightv) {
      if (ilo == 1) goto L50;
      for (i = ilo - 1; i >= 1; --i) {
        k = static_cast<blasint>(RSCALE(i));
        if (k == i) continue;
        Swap(m, V.ptr(i, 1), ldv, V.ptr(k, 1), ldv);
      }
    L50:
      if (ihi == n) goto L70;
      for (i = ihi + 1; i <= n; ++i) {
        k = static_cast<blasint>(RSCALE(i));
        if (k == i) continue;
        Swap(m, V.ptr(i, 1), ldv, V.ptr(k, 1), ldv);
      }
    }
    // Backward permutation on left eigenvectors
  L70:
    if (leftv) {
      if (ilo == 1) goto L90;
      for (i = ilo - 1; i >= 1; --i) {
        k = static_cast<blasint>(LSCALE(i));
        if (k == i) continue;
        Swap(m, V.ptr(i, 1), ldv, V.ptr(k, 1), ldv);
      }
    L90:
      if (ihi == n) return info;
      for (i = ihi + 1; i <= n; ++i) {
        k = static_cast<blasint>(LSCALE(i));
        if (k == i) continue;
        Swap(m, V.ptr(i, 1), ldv, V.ptr(k, 1), ldv);
      }
    }
  }
  return info;
}

// xGGHRD: reduces the pair (A, B), B upper triangular, to generalized upper
// Hessenberg form by orthogonal (unitary) transformations: Q**H * A * Z = H
// upper Hessenberg in A, Q**H * B * Z = T upper triangular in B. A is
// already upper triangular in rows and columns 1:ILO-1 and IHI+1:N (as
// xGGBAL leaves it). COMPQ / COMPZ = 'N': Q / Z not computed; 'I': Q / Z
// are initialized to the identity and returned; 'V': Q / Z on entry (Q1,
// Z1) are multiplied by them, Q1 * Q and Z1 * Z. Plane rotations, not
// blocked.
template <typename T>
blasint Gghrd(char compq, char compz, blasint n, blasint ilo, blasint ihi, T* a,
              blasint lda, T* b, blasint ldb, T* q, blasint ldq, T* z,
              blasint ldz) {
  using R = RealT<T>;
  const T cone = T(1);
  const T czero = T(0);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Matrix<T> Q(q, ldq);
  const Matrix<T> Z(z, ldz);
  blasint icompq, icompz, jcol, jrow;
  bool ilq = false, ilz = false;
  R c;
  T ctemp, s;
  // Decode COMPQ
  if (Lsame(compq, 'N')) {
    ilq = false;
    icompq = 1;
  } else if (Lsame(compq, 'V')) {
    ilq = true;
    icompq = 2;
  } else if (Lsame(compq, 'I')) {
    ilq = true;
    icompq = 3;
  } else {
    icompq = 0;
  }
  // Decode COMPZ
  if (Lsame(compz, 'N')) {
    ilz = false;
    icompz = 1;
  } else if (Lsame(compz, 'V')) {
    ilz = true;
    icompz = 2;
  } else if (Lsame(compz, 'I')) {
    ilz = true;
    icompz = 3;
  } else {
    icompz = 0;
  }
  // Test the input parameters.
  blasint info = 0;
  if (icompq <= 0) {
    info = -1;
  } else if (icompz <= 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1) {
    info = -4;
  } else if (ihi > n || ihi < ilo - 1) {
    info = -5;
  } else if (lda < Max<blasint>(1, n)) {
    info = -7;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -9;
  } else if ((ilq && ldq < n) || ldq < 1) {
    info = -11;
  } else if ((ilz && ldz < n) || ldz < 1) {
    info = -13;
  }
  if (info != 0) {
    Xerbla<T>("GGHRD", -info);
    return info;
  }
  // Initialize Q and Z if desired.
  if (icompq == 3) Laset('F', n, n, czero, cone, q, ldq);
  if (icompz == 3) Laset('F', n, n, czero, cone, z, ldz);
  // Quick return if possible
  if (n <= 1) return info;
  // Zero out lower triangle of B
  for (jcol = 1; jcol <= n - 1; ++jcol) {
    for (jrow = jcol + 1; jrow <= n; ++jrow) {
      B(jrow, jcol) = czero;
    }
  }
  // Reduce A and B
  for (jcol = ilo; jcol <= ihi - 2; ++jcol) {
    for (jrow = ihi; jrow >= jcol + 2; --jrow) {
      // Step 1: rotate rows JROW-1, JROW to kill A(JROW,JCOL)
      ctemp = A(jrow - 1, jcol);
      Lartg(ctemp, A(jrow, jcol), &c, &s, A.ptr(jrow - 1, jcol));
      A(jrow, jcol) = czero;
      Rot(n - jcol, A.ptr(jrow - 1, jcol + 1), lda, A.ptr(jrow, jcol + 1), lda,
          c, s);
      Rot(n + 2 - jrow, B.ptr(jrow - 1, jrow - 1), ldb, B.ptr(jrow, jrow - 1),
          ldb, c, s);
      if (ilq) Rot(n, Q.ptr(1, jrow - 1), 1, Q.ptr(1, jrow), 1, c, Conj(s));
      // Step 2: rotate columns JROW, JROW-1 to kill B(JROW,JROW-1)
      ctemp = B(jrow, jrow);
      Lartg(ctemp, B(jrow, jrow - 1), &c, &s, B.ptr(jrow, jrow));
      B(jrow, jrow - 1) = czero;
      Rot(ihi, A.ptr(1, jrow), 1, A.ptr(1, jrow - 1), 1, c, s);
      Rot(jrow - 1, B.ptr(1, jrow), 1, B.ptr(1, jrow - 1), 1, c, s);
      if (ilz) Rot(n, Z.ptr(1, jrow), 1, Z.ptr(1, jrow - 1), 1, c, s);
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template blasint Ggbal<T>(char, blasint, T*, blasint, T*, blasint, blasint*, \
                            blasint*, RealT<T>*, RealT<T>*, RealT<T>*);        \
  template blasint Ggbak<T>(char, char, blasint, blasint, blasint,             \
                            const RealT<T>*, const RealT<T>*, blasint, T*,     \
                            blasint);                                          \
  template blasint Gghrd<T>(char, char, blasint, blasint, blasint, T*,         \
                            blasint, T*, blasint, T*, blasint, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GGHRD(prefix, T, R)                                   \
  HWYBLAS_API void prefix##ggbal_(                                           \
      const char* job, const hwyblas_int* n, R* a, const hwyblas_int* lda,   \
      R* b, const hwyblas_int* ldb, hwyblas_int* ilo, hwyblas_int* ihi,      \
      R* lscale, R* rscale, R* work, hwyblas_int* info) {                    \
    *info = hwyblas::lapack::Ggbal(*job, *n, As<T>(a), *lda, As<T>(b), *ldb, \
                                   ilo, ihi, lscale, rscale, work);          \
  }                                                                          \
  HWYBLAS_API void prefix##ggbak_(                                           \
      const char* job, const char* side, const hwyblas_int* n,               \
      const hwyblas_int* ilo, const hwyblas_int* ihi, const R* lscale,       \
      const R* rscale, const hwyblas_int* m, R* v, const hwyblas_int* ldv,   \
      hwyblas_int* info) {                                                   \
    *info = hwyblas::lapack::Ggbak(*job, *side, *n, *ilo, *ihi, lscale,      \
                                   rscale, *m, As<T>(v), *ldv);              \
  }                                                                          \
  HWYBLAS_API void prefix##gghrd_(                                           \
      const char* compq, const char* compz, const hwyblas_int* n,            \
      const hwyblas_int* ilo, const hwyblas_int* ihi, R* a,                  \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* q,            \
      const hwyblas_int* ldq, R* z, const hwyblas_int* ldz,                  \
      hwyblas_int* info) {                                                   \
    *info = hwyblas::lapack::Gghrd(*compq, *compz, *n, *ilo, *ihi, As<T>(a), \
                                   *lda, As<T>(b), *ldb, As<T>(q), *ldq,     \
                                   As<T>(z), *ldz);                          \
  }

HWYBLAS_LAPACK_GGHRD(s, float, float)
HWYBLAS_LAPACK_GGHRD(d, double, double)
HWYBLAS_LAPACK_GGHRD(c, c32, float)
HWYBLAS_LAPACK_GGHRD(z, c64, double)

#undef HWYBLAS_LAPACK_GGHRD

}  // extern "C"
