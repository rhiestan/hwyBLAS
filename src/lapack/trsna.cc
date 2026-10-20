// Condition numbers of the eigenvalues and eigenvectors of a matrix in Schur
// form: xTRSNA, with xLAQTR (a real quasi-triangular system in real or
// complex arithmetic, with scaling against overflow).
//
// The real and the complex xTRSNA differ throughout (2-by-2 blocks of the
// real Schur form, the real one solves with xLAQTR and the complex one with
// xLATRS) and in their workspaces (IWORK against RWORK), so they are
// ported as two overloads.
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

// xLAQTR (real only): solves op(T)*p = scale*c (LREAL = true) or
// op(T + i*B)*(p+iq) = scale*(c+id) (LREAL = false) in real arithmetic,
// where T is the N-by-N upper quasi-triangular matrix in Schur canonical
// form, op(T) = T or T**T (LTRAN), and B = diag(W, B(2), ..., B(N)) (B has
// N elements, B(1) is not referenced; W is used for the first diagonal
// element). X holds c (or c and d, 2*N elements) on entry and the solution
// on exit; SCALE <= 1 is chosen to avoid overflow. WORK has N elements.
//
// Returns INFO: 0, or 1 if a diagonal block was perturbed to make it
// nonsingular (by a 1-by-1 block of T), 2 if by a 2-by-2 block (xLALN2).
template <typename R>
blasint Laqtr(bool ltran, bool lreal, blasint n, const R* t, blasint ldt,
              const R* b, R w, R* scale, R* x, R* work) {
  const R zero = R(0);
  const R one = R(1);
  const Vector<const R> B(b);
  R d[4];
  const Matrix<R> D(d, 2);
  const Matrix<const R> Tm(t, ldt);
  R v[4];
  const Matrix<R> V(v, 2);
  const Vector<R> WORK(work);
  const Vector<R> X(x);
  blasint i, ierr, j, j1, j2, jnext, k, n1, n2;
  bool notran;
  R bignum, eps, rec, scaloc, si, smin, sminw, smlnum, sr, tjj, tmp, xj, xmax,
      xnorm, z;
  // Do not test the input parameters for errors
  notran = !ltran;
  blasint info = 0;
  // Quick return if possible
  if (n == 0) return info;
  // Set constants to control overflow
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  bignum = one / smlnum;
  xnorm = Lange('M', n, n, t, ldt, d);
  if (!lreal) xnorm = Max(xnorm, std::abs(w), Lange('M', n, 1, b, n, d));
  smin = Max(smlnum, eps * xnorm);
  // Compute 1-norm of each column of strictly upper triangular
  // part of T to control overflow in triangular solver.
  WORK(1) = zero;
  for (j = 2; j <= n; ++j) {
    WORK(j) = Asum(j - 1, Tm.ptr(1, j), 1);
  }
  if (!lreal) {
    for (i = 2; i <= n; ++i) {
      WORK(i) = WORK(i) + std::abs(B(i));
    }
  }
  n2 = 2 * n;
  n1 = n;
  if (!lreal) n1 = n2;
  k = Iamax(n1, x, 1);
  xmax = std::abs(X(k));
  *scale = one;
  if (xmax > bignum) {
    *scale = bignum / xmax;
    Rscal(n1, *scale, x, 1);
    xmax = bignum;
  }
  if (lreal) {
    if (notran) {
      // Solve T*p = scale*c
      jnext = n;
      for (j = n; j >= 1; --j) {
        if (j > jnext) continue;
        j1 = j;
        j2 = j;
        jnext = j - 1;
        if (j > 1) {
          if (Tm(j, j - 1) != zero) {
            j1 = j - 1;
            jnext = j - 2;
          }
        }
        if (j1 == j2) {
          // Meet 1 by 1 diagonal block
          //
          // Scale to avoid overflow when computing
          //     x(j) = b(j)/T(j,j)
          xj = std::abs(X(j1));
          tjj = std::abs(Tm(j1, j1));
          tmp = Tm(j1, j1);
          if (tjj < smin) {
            tmp = smin;
            tjj = smin;
            info = 1;
          }
          if (xj == zero) continue;
          if (tjj < one) {
            if (xj > bignum * tjj) {
              rec = one / xj;
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          X(j1) = X(j1) / tmp;
          xj = std::abs(X(j1));
          // Scale x if necessary to avoid overflow when adding a
          // multiple of column j1 of T.
          if (xj > one) {
            rec = one / xj;
            if (WORK(j1) > (bignum - xmax) * rec) {
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
            }
          }
          if (j1 > 1) {
            Axpy(j1 - 1, -X(j1), Tm.ptr(1, j1), 1, x, 1);
            k = Iamax(j1 - 1, x, 1);
            xmax = std::abs(X(k));
          }
        } else {
          // Meet 2 by 2 diagonal block
          //
          // Call 2 by 2 linear system solve, to take
          // care of possible overflow by scaling factor.
          D(1, 1) = X(j1);
          D(2, 1) = X(j2);
          ierr = Laln2(false, 2, 1, smin, one, Tm.ptr(j1, j1), ldt, one, one, d,
                       2, zero, zero, v, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 2;
          if (scaloc != one) {
            Rscal(n, scaloc, x, 1);
            *scale = *scale * scaloc;
          }
          X(j1) = V(1, 1);
          X(j2) = V(2, 1);
          // Scale V(1,1) (= X(J1)) and/or V(2,1) (=X(J2))
          // to avoid overflow in updating right-hand side.
          xj = Max(std::abs(V(1, 1)), std::abs(V(2, 1)));
          if (xj > one) {
            rec = one / xj;
            if (Max(WORK(j1), WORK(j2)) > (bignum - xmax) * rec) {
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
            }
          }
          // Update right-hand side
          if (j1 > 1) {
            Axpy(j1 - 1, -X(j1), Tm.ptr(1, j1), 1, x, 1);
            Axpy(j1 - 1, -X(j2), Tm.ptr(1, j2), 1, x, 1);
            k = Iamax(j1 - 1, x, 1);
            xmax = std::abs(X(k));
          }
        }
      }
    } else {
      // Solve T**T*p = scale*c
      jnext = 1;
      for (j = 1; j <= n; ++j) {
        if (j < jnext) continue;
        j1 = j;
        j2 = j;
        jnext = j + 1;
        if (j < n) {
          if (Tm(j + 1, j) != zero) {
            j2 = j + 1;
            jnext = j + 2;
          }
        }
        if (j1 == j2) {
          // 1 by 1 diagonal block
          //
          // Scale if necessary to avoid overflow in forming the
          // right-hand side element by inner product.
          xj = std::abs(X(j1));
          if (xmax > one) {
            rec = one / xmax;
            if (WORK(j1) > (bignum - xj) * rec) {
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          X(j1) = X(j1) - Dotc(j1 - 1, Tm.ptr(1, j1), 1, x, 1);
          xj = std::abs(X(j1));
          tjj = std::abs(Tm(j1, j1));
          tmp = Tm(j1, j1);
          if (tjj < smin) {
            tmp = smin;
            tjj = smin;
            info = 1;
          }
          if (tjj < one) {
            if (xj > bignum * tjj) {
              rec = one / xj;
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          X(j1) = X(j1) / tmp;
          xmax = Max(xmax, std::abs(X(j1)));
        } else {
          // 2 by 2 diagonal block
          //
          // Scale if necessary to avoid overflow in forming the
          // right-hand side elements by inner product.
          xj = Max(std::abs(X(j1)), std::abs(X(j2)));
          if (xmax > one) {
            rec = one / xmax;
            if (Max(WORK(j2), WORK(j1)) > (bignum - xj) * rec) {
              Rscal(n, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          D(1, 1) = X(j1) - Dotc(j1 - 1, Tm.ptr(1, j1), 1, x, 1);
          D(2, 1) = X(j2) - Dotc(j1 - 1, Tm.ptr(1, j2), 1, x, 1);
          ierr = Laln2(true, 2, 1, smin, one, Tm.ptr(j1, j1), ldt, one, one, d,
                       2, zero, zero, v, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 2;
          if (scaloc != one) {
            Rscal(n, scaloc, x, 1);
            *scale = *scale * scaloc;
          }
          X(j1) = V(1, 1);
          X(j2) = V(2, 1);
          xmax = Max(std::abs(X(j1)), std::abs(X(j2)), xmax);
        }
      }
    }
  } else {
    sminw = Max(eps * std::abs(w), smin);
    if (notran) {
      // Solve (T + iB)*(p+iq) = c+id
      jnext = n;
      for (j = n; j >= 1; --j) {
        if (j > jnext) continue;
        j1 = j;
        j2 = j;
        jnext = j - 1;
        if (j > 1) {
          if (Tm(j, j - 1) != zero) {
            j1 = j - 1;
            jnext = j - 2;
          }
        }
        if (j1 == j2) {
          // 1 by 1 diagonal block
          //
          // Scale if necessary to avoid overflow in division
          z = w;
          if (j1 == 1) z = B(1);
          xj = std::abs(X(j1)) + std::abs(X(n + j1));
          tjj = std::abs(Tm(j1, j1)) + std::abs(z);
          tmp = Tm(j1, j1);
          if (tjj < sminw) {
            tmp = sminw;
            tjj = sminw;
            info = 1;
          }
          if (xj == zero) continue;
          if (tjj < one) {
            if (xj > bignum * tjj) {
              rec = one / xj;
              Rscal(n2, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          Ladiv(X(j1), X(n + j1), tmp, z, &sr, &si);
          X(j1) = sr;
          X(n + j1) = si;
          xj = std::abs(X(j1)) + std::abs(X(n + j1));
          // Scale x if necessary to avoid overflow when adding a
          // multiple of column j1 of T.
          if (xj > one) {
            rec = one / xj;
            if (WORK(j1) > (bignum - xmax) * rec) {
              Rscal(n2, rec, x, 1);
              *scale = *scale * rec;
            }
          }
          if (j1 > 1) {
            Axpy(j1 - 1, -X(j1), Tm.ptr(1, j1), 1, x, 1);
            Axpy(j1 - 1, -X(n + j1), Tm.ptr(1, j1), 1, X.ptr(n + 1), 1);
            X(1) = X(1) + B(j1) * X(n + j1);
            X(n + 1) = X(n + 1) - B(j1) * X(j1);
            xmax = zero;
            for (k = 1; k <= j1 - 1; ++k) {
              xmax = Max(xmax, std::abs(X(k)) + std::abs(X(k + n)));
            }
          }
        } else {
          // Meet 2 by 2 diagonal block
          D(1, 1) = X(j1);
          D(2, 1) = X(j2);
          D(1, 2) = X(n + j1);
          D(2, 2) = X(n + j2);
          ierr = Laln2(false, 2, 2, sminw, one, Tm.ptr(j1, j1), ldt, one, one,
                       d, 2, zero, -w, v, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 2;
          if (scaloc != one) {
            Rscal(2 * n, scaloc, x, 1);
            *scale = scaloc * *scale;
          }
          X(j1) = V(1, 1);
          X(j2) = V(2, 1);
          X(n + j1) = V(1, 2);
          X(n + j2) = V(2, 2);
          // Scale X(J1), .... to avoid overflow in
          // updating right hand side.
          xj = Max(std::abs(V(1, 1)) + std::abs(V(1, 2)),
                   std::abs(V(2, 1)) + std::abs(V(2, 2)));
          if (xj > one) {
            rec = one / xj;
            if (Max(WORK(j1), WORK(j2)) > (bignum - xmax) * rec) {
              Rscal(n2, rec, x, 1);
              *scale = *scale * rec;
            }
          }
          // Update the right-hand side.
          if (j1 > 1) {
            Axpy(j1 - 1, -X(j1), Tm.ptr(1, j1), 1, x, 1);
            Axpy(j1 - 1, -X(j2), Tm.ptr(1, j2), 1, x, 1);
            Axpy(j1 - 1, -X(n + j1), Tm.ptr(1, j1), 1, X.ptr(n + 1), 1);
            Axpy(j1 - 1, -X(n + j2), Tm.ptr(1, j2), 1, X.ptr(n + 1), 1);
            X(1) = X(1) + B(j1) * X(n + j1) + B(j2) * X(n + j2);
            X(n + 1) = X(n + 1) - B(j1) * X(j1) - B(j2) * X(j2);
            xmax = zero;
            for (k = 1; k <= j1 - 1; ++k) {
              xmax = Max(std::abs(X(k)) + std::abs(X(k + n)), xmax);
            }
          }
        }
      }
    } else {
      // Solve (T + iB)**T*(p+iq) = c+id
      jnext = 1;
      for (j = 1; j <= n; ++j) {
        if (j < jnext) continue;
        j1 = j;
        j2 = j;
        jnext = j + 1;
        if (j < n) {
          if (Tm(j + 1, j) != zero) {
            j2 = j + 1;
            jnext = j + 2;
          }
        }
        if (j1 == j2) {
          // 1 by 1 diagonal block
          //
          // Scale if necessary to avoid overflow in forming the
          // right-hand side element by inner product.
          xj = std::abs(X(j1)) + std::abs(X(j1 + n));
          if (xmax > one) {
            rec = one / xmax;
            if (WORK(j1) > (bignum - xj) * rec) {
              Rscal(n2, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          X(j1) = X(j1) - Dotc(j1 - 1, Tm.ptr(1, j1), 1, x, 1);
          X(n + j1) =
              X(n + j1) - Dotc(j1 - 1, Tm.ptr(1, j1), 1, X.ptr(n + 1), 1);
          if (j1 > 1) {
            X(j1) = X(j1) - B(j1) * X(n + 1);
            X(n + j1) = X(n + j1) + B(j1) * X(1);
          }
          xj = std::abs(X(j1)) + std::abs(X(j1 + n));
          z = w;
          if (j1 == 1) z = B(1);
          // Scale if necessary to avoid overflow in
          // complex division
          tjj = std::abs(Tm(j1, j1)) + std::abs(z);
          tmp = Tm(j1, j1);
          if (tjj < sminw) {
            tmp = sminw;
            tjj = sminw;
            info = 1;
          }
          if (tjj < one) {
            if (xj > bignum * tjj) {
              rec = one / xj;
              Rscal(n2, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          Ladiv(X(j1), X(n + j1), tmp, -z, &sr, &si);
          X(j1) = sr;
          X(j1 + n) = si;
          xmax = Max(std::abs(X(j1)) + std::abs(X(j1 + n)), xmax);
        } else {
          // 2 by 2 diagonal block
          //
          // Scale if necessary to avoid overflow in forming the
          // right-hand side element by inner product.
          xj = Max(std::abs(X(j1)) + std::abs(X(n + j1)),
                   std::abs(X(j2)) + std::abs(X(n + j2)));
          if (xmax > one) {
            rec = one / xmax;
            if (Max(WORK(j1), WORK(j2)) > (bignum - xj) / xmax) {
              Rscal(n2, rec, x, 1);
              *scale = *scale * rec;
              xmax = xmax * rec;
            }
          }
          D(1, 1) = X(j1) - Dotc(j1 - 1, Tm.ptr(1, j1), 1, x, 1);
          D(2, 1) = X(j2) - Dotc(j1 - 1, Tm.ptr(1, j2), 1, x, 1);
          D(1, 2) = X(n + j1) - Dotc(j1 - 1, Tm.ptr(1, j1), 1, X.ptr(n + 1), 1);
          D(2, 2) = X(n + j2) - Dotc(j1 - 1, Tm.ptr(1, j2), 1, X.ptr(n + 1), 1);
          D(1, 1) = D(1, 1) - B(j1) * X(n + 1);
          D(2, 1) = D(2, 1) - B(j2) * X(n + 1);
          D(1, 2) = D(1, 2) + B(j1) * X(1);
          D(2, 2) = D(2, 2) + B(j2) * X(1);
          ierr = Laln2(true, 2, 2, sminw, one, Tm.ptr(j1, j1), ldt, one, one, d,
                       2, zero, w, v, 2, &scaloc, &xnorm);
          if (ierr != 0) info = 2;
          if (scaloc != one) {
            Rscal(n2, scaloc, x, 1);
            *scale = scaloc * *scale;
          }
          X(j1) = V(1, 1);
          X(j2) = V(2, 1);
          X(n + j1) = V(1, 2);
          X(n + j2) = V(2, 2);
          xmax = Max(std::abs(X(j1)) + std::abs(X(n + j1)),
                     std::abs(X(j2)) + std::abs(X(n + j2)), xmax);
        }
      }
    }
  }
  return info;
}

// xTRSNA (real T): estimates reciprocal condition numbers of the
// eigenvalues (JOB = 'E', in S) and/or the right eigenvectors (JOB = 'V',
// in SEP; 'B': both) of the N-by-N upper quasi-triangular matrix T in
// Schur canonical form, for all eigenpairs (HOWMNY = 'A') or those that
// SELECT chooses (HOWMNY = 'S'; for a complex pair, SELECT(j) or
// SELECT(j+1)). For JOB = 'E' or 'B', VL and VR hold the left and right
// eigenvectors of those eigenvalues (as returned by xTREVC3, a complex one
// in two columns). MM is the number of elements of S and SEP, M (output)
// the number used. WORK is LDWORK-by-(N+6) and IWORK has 2*(N-1)
// elements; both are used for JOB = 'V' or 'B' only.
//
// Returns INFO: 0, or -i for an illegal argument i.
template <typename R>
blasint Trsna(char job, char howmny, const blasint* select, blasint n,
              const R* t, blasint ldt, const R* vl, blasint ldvl, const R* vr,
              blasint ldvr, R* s, R* sep, blasint mm, blasint* m, R* work,
              blasint ldwork, blasint* iwork) {
  using T = R;
  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  R dummy[1];
  blasint isave[3];
  const Vector<R> S(s);
  const Vector<const blasint> SELECT(select);
  const Vector<R> SEP(sep);
  const Matrix<const R> Tm(t, ldt);
  const Matrix<const R> VL(vl, ldvl);
  const Matrix<const R> VR(vr, ldvr);
  const Matrix<R> WORK(work, ldwork);
  blasint i, ierr, ifst, ilst, j, k, kase, ks, n2, nn;
  bool pair, somcon, wantbh, wants, wantsp;
  R bignum, cond, cs, delta, eps, est, lnrm, mu, prod, prod1, prod2, rnrm,
      smlnum, sn;
  // Set by xLAQTR before it is read (the first call of xLACN2 returns
  // KASE = 1); initialized for the compiler.
  R scale = one;
  // W of xLAQTR for LREAL = true: not referenced (the Fortran passes the
  // uninitialized DUMM).
  const R dumm = zero;
  // Decode and test the input parameters
  wantbh = Lsame(job, 'B');
  wants = Lsame(job, 'E') || wantbh;
  wantsp = Lsame(job, 'V') || wantbh;
  somcon = Lsame(howmny, 'S');
  blasint info = 0;
  if (!wants && !wantsp) {
    info = -1;
  } else if (!Lsame(howmny, 'A') && !somcon) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldvl < 1 || (wants && ldvl < n)) {
    info = -8;
  } else if (ldvr < 1 || (wants && ldvr < n)) {
    info = -10;
  } else {
    // Set M to the number of eigenpairs for which condition numbers
    // are required, and test MM.
    if (somcon) {
      *m = 0;
      pair = false;
      for (k = 1; k <= n; ++k) {
        if (pair) {
          pair = false;
        } else {
          if (k < n) {
            if (Tm(k + 1, k) == zero) {
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
    } else {
      *m = n;
    }
    if (mm < *m) {
      info = -13;
    } else if (ldwork < 1 || (wantsp && ldwork < n)) {
      info = -16;
    }
  }
  if (info != 0) {
    Xerbla<T>("TRSNA", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  if (n == 1) {
    if (somcon) {
      if (!SELECT(1)) return info;
    }
    if (wants) S(1) = one;
    if (wantsp) SEP(1) = std::abs(Tm(1, 1));
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  bignum = one / smlnum;
  ks = 0;
  pair = false;
  for (k = 1; k <= n; ++k) {
    // Determine whether T(k,k) begins a 1-by-1 or 2-by-2 block.
    if (pair) {
      pair = false;
      continue;
    } else {
      if (k < n) pair = Tm(k + 1, k) != zero;
    }
    // Determine whether condition numbers are required for the k-th
    // eigenpair.
    if (somcon) {
      if (pair) {
        if (!SELECT(k) && !SELECT(k + 1)) continue;
      } else {
        if (!SELECT(k)) continue;
      }
    }
    ks = ks + 1;
    if (wants) {
      // Compute the reciprocal condition number of the k-th
      // eigenvalue.
      if (!pair) {
        // Real eigenvalue.
        prod = Dotc(n, VR.ptr(1, ks), 1, VL.ptr(1, ks), 1);
        rnrm = Nrm2(n, VR.ptr(1, ks), 1);
        lnrm = Nrm2(n, VL.ptr(1, ks), 1);
        S(ks) = std::abs(prod) / (rnrm * lnrm);
      } else {
        // Complex eigenvalue.
        prod1 = Dotc(n, VR.ptr(1, ks), 1, VL.ptr(1, ks), 1);
        prod1 = prod1 + Dotc(n, VR.ptr(1, ks + 1), 1, VL.ptr(1, ks + 1), 1);
        prod2 = Dotc(n, VL.ptr(1, ks), 1, VR.ptr(1, ks + 1), 1);
        prod2 = prod2 - Dotc(n, VL.ptr(1, ks + 1), 1, VR.ptr(1, ks), 1);
        rnrm = Lapy2(Nrm2(n, VR.ptr(1, ks), 1), Nrm2(n, VR.ptr(1, ks + 1), 1));
        lnrm = Lapy2(Nrm2(n, VL.ptr(1, ks), 1), Nrm2(n, VL.ptr(1, ks + 1), 1));
        cond = Lapy2(prod1, prod2) / (rnrm * lnrm);
        S(ks) = cond;
        S(ks + 1) = cond;
      }
    }
    if (wantsp) {
      // Estimate the reciprocal condition number of the k-th
      // eigenvector.
      //
      // Copy the matrix T to the array WORK and swap the diagonal
      // block beginning at T(k,k) to the (1,1) position.
      Lacpy('F', n, n, t, ldt, work, ldwork);
      ifst = k;
      ilst = 1;
      ierr = Trexc('N', n, work, ldwork, dummy, 1, &ifst, &ilst,
                   WORK.ptr(1, n + 1));
      if (ierr == 1 || ierr == 2) {
        // Could not swap because blocks not well separated
        scale = one;
        est = bignum;
      } else {
        // Reordering successful
        if (WORK(2, 1) == zero) {
          // Form C = T22 - lambda*I in WORK(2:N,2:N).
          for (i = 2; i <= n; ++i) {
            WORK(i, i) = WORK(i, i) - WORK(1, 1);
          }
          n2 = 1;
          nn = n - 1;
        } else {
          // Triangularize the 2 by 2 block by unitary
          // transformation U = [  cs   i*ss ]
          //                    [ i*ss   cs  ].
          // such that the (1,1) position of WORK is complex
          // eigenvalue lambda with positive imaginary part. (2,2)
          // position of WORK is the complex eigenvalue lambda
          // with negative imaginary  part.
          mu =
              std::sqrt(std::abs(WORK(1, 2))) * std::sqrt(std::abs(WORK(2, 1)));
          delta = Lapy2(mu, WORK(2, 1));
          cs = mu / delta;
          sn = -WORK(2, 1) / delta;
          // Form
          //
          // C**T = WORK(2:N,2:N) + i*[rwork(1) ..... rwork(n-1) ]
          //                          [   mu                     ]
          //                          [         ..               ]
          //                          [             ..           ]
          //                          [                  mu      ]
          // where C**T is transpose of matrix C,
          // and RWORK is stored starting in the N+1-st column of
          // WORK.
          for (j = 3; j <= n; ++j) {
            WORK(2, j) = cs * WORK(2, j);
            WORK(j, j) = WORK(j, j) - WORK(1, 1);
          }
          WORK(2, 2) = zero;
          WORK(1, n + 1) = two * mu;
          for (i = 2; i <= n - 1; ++i) {
            WORK(i, n + 1) = sn * WORK(1, i + 1);
          }
          n2 = 2;
          nn = 2 * (n - 1);
        }
        // Estimate norm(inv(C**T))
        est = zero;
        kase = 0;
      L50:
        Lacn2(nn, WORK.ptr(1, n + 2), WORK.ptr(1, n + 4), iwork, &est, &kase,
              isave);
        if (kase != 0) {
          // (The INFO of xLAQTR, a perturbed diagonal block, is not used.)
          if (kase == 1) {
            if (n2 == 1) {
              // Real eigenvalue: solve C**T*x = scale*c.
              Laqtr(true, true, n - 1, WORK.ptr(2, 2), ldwork, dummy, dumm,
                    &scale, WORK.ptr(1, n + 4), WORK.ptr(1, n + 6));
            } else {
              // Complex eigenvalue: solve
              // C**T*(p+iq) = scale*(c+id) in real arithmetic.
              Laqtr(true, false, n - 1, WORK.ptr(2, 2), ldwork,
                    WORK.ptr(1, n + 1), mu, &scale, WORK.ptr(1, n + 4),
                    WORK.ptr(1, n + 6));
            }
          } else {
            if (n2 == 1) {
              // Real eigenvalue: solve C*x = scale*c.
              Laqtr(false, true, n - 1, WORK.ptr(2, 2), ldwork, dummy, dumm,
                    &scale, WORK.ptr(1, n + 4), WORK.ptr(1, n + 6));
            } else {
              // Complex eigenvalue: solve
              // C*(p+iq) = scale*(c+id) in real arithmetic.
              Laqtr(false, false, n - 1, WORK.ptr(2, 2), ldwork,
                    WORK.ptr(1, n + 1), mu, &scale, WORK.ptr(1, n + 4),
                    WORK.ptr(1, n + 6));
            }
          }
          goto L50;
        }
      }
      SEP(ks) = scale / Max(est, smlnum);
      if (pair) SEP(ks + 1) = SEP(ks);
    }
    if (pair) ks = ks + 1;
  }
  return info;
}

// xTRSNA (complex T): as the real routine, for the upper triangular T (one
// column of VL and VR per eigenvector). WORK is LDWORK-by-(N+1) and RWORK
// has N elements (used for JOB = 'V' or 'B' only).
template <typename T>
blasint Trsna(char job, char howmny, const blasint* select, blasint n,
              const T* t, blasint ldt, const T* vl, blasint ldvl, const T* vr,
              blasint ldvr, RealT<T>* s, RealT<T>* sep, blasint mm, blasint* m,
              T* work, blasint ldwork, RealT<T>* rwork) {
  using R = RealT<T>;
  const R zero = R(0);
  const R one = R(1);
  T dummy[1];
  blasint isave[3];
  const Vector<R> S(s);
  const Vector<const blasint> SELECT(select);
  const Vector<R> SEP(sep);
  const Matrix<const T> Tm(t, ldt);
  const Matrix<const T> VL(vl, ldvl);
  const Matrix<const T> VR(vr, ldvr);
  const Matrix<T> WORK(work, ldwork);
  blasint i, ix, j, k, kase, ks;
  bool somcon, wantbh, wants, wantsp;
  R bignum, eps, est, lnrm, rnrm, scale, smlnum, xnorm;
  T prod;
  char normin;
  // Decode and test the input parameters
  wantbh = Lsame(job, 'B');
  wants = Lsame(job, 'E') || wantbh;
  wantsp = Lsame(job, 'V') || wantbh;
  somcon = Lsame(howmny, 'S');
  // Set M to the number of eigenpairs for which condition numbers are
  // to be computed.
  if (somcon) {
    *m = 0;
    for (j = 1; j <= n; ++j) {
      if (SELECT(j)) *m = *m + 1;
    }
  } else {
    *m = n;
  }
  blasint info = 0;
  if (!wants && !wantsp) {
    info = -1;
  } else if (!Lsame(howmny, 'A') && !somcon) {
    info = -2;
  } else if (n < 0) {
    info = -4;
  } else if (ldt < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldvl < 1 || (wants && ldvl < n)) {
    info = -8;
  } else if (ldvr < 1 || (wants && ldvr < n)) {
    info = -10;
  } else if (mm < *m) {
    info = -13;
  } else if (ldwork < 1 || (wantsp && ldwork < n)) {
    info = -16;
  }
  if (info != 0) {
    Xerbla<T>("TRSNA", -info);
    return info;
  }
  // Quick return if possible
  if (n == 0) return info;
  if (n == 1) {
    if (somcon) {
      if (!SELECT(1)) return info;
    }
    if (wants) S(1) = one;
    if (wantsp) SEP(1) = std::abs(Tm(1, 1));
    return info;
  }
  // Get machine constants
  eps = Lamch<R>('P');
  smlnum = Lamch<R>('S') / eps;
  bignum = one / smlnum;
  ks = 1;
  for (k = 1; k <= n; ++k) {
    if (somcon) {
      if (!SELECT(k)) continue;
    }
    if (wants) {
      // Compute the reciprocal condition number of the k-th
      // eigenvalue.
      prod = Dotc(n, VR.ptr(1, ks), 1, VL.ptr(1, ks), 1);
      rnrm = Nrm2(n, VR.ptr(1, ks), 1);
      lnrm = Nrm2(n, VL.ptr(1, ks), 1);
      S(ks) = std::abs(prod) / (rnrm * lnrm);
    }
    if (wantsp) {
      // Estimate the reciprocal condition number of the k-th
      // eigenvector.
      //
      // Copy the matrix T to the array WORK and swap the k-th
      // diagonal element to the (1,1) position.
      Lacpy('F', n, n, t, ldt, work, ldwork);
      // (The INFO of xTREXC, always 0 here, is not used.)
      Trexc('N', n, work, ldwork, dummy, 1, k, 1);
      // Form  C = T22 - lambda*I in WORK(2:N,2:N).
      for (i = 2; i <= n; ++i) {
        WORK(i, i) = WORK(i, i) - WORK(1, 1);
      }
      // Estimate a lower bound for the 1-norm of inv(C**H). The 1st
      // and (N+1)th columns of WORK are used to store work vectors.
      SEP(ks) = zero;
      est = zero;
      kase = 0;
      normin = 'N';
    L30:
      Lacn2(n - 1, WORK.ptr(1, n + 1), work, &est, &kase, isave);
      if (kase != 0) {
        if (kase == 1) {
          // Solve C**H*x = scale*b
          Latrs('U', 'C', 'N', normin, n - 1, WORK.ptr(2, 2), ldwork, work,
                &scale, rwork);
        } else {
          // Solve C*x = scale*b
          Latrs('U', 'N', 'N', normin, n - 1, WORK.ptr(2, 2), ldwork, work,
                &scale, rwork);
        }
        normin = 'Y';
        if (scale != one) {
          // Multiply by 1/SCALE if doing so will not cause
          // overflow.
          ix = Iamax(n - 1, work, 1);
          xnorm = Cabs1(WORK(ix, 1));
          if (scale < xnorm * smlnum || scale == zero) goto L40;
          Rscl(n, scale, work, 1);
        }
        goto L30;
      }
      SEP(ks) = one / Max(est, smlnum);
    }
  L40:
    ks = ks + 1;
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(R)                                     \
  template blasint Laqtr<R>(bool, bool, blasint, const R*, blasint, const R*,  \
                            R, R*, R*, R*);                                    \
  template blasint Trsna<R>(char, char, const blasint*, blasint, const R*,     \
                            blasint, const R*, blasint, const R*, blasint, R*, \
                            R*, blasint, blasint*, R*, blasint, blasint*);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                              \
  template blasint Trsna<T>(char, char, const blasint*, blasint, const T*, \
                            blasint, const T*, blasint, const T*, blasint, \
                            RealT<T>*, RealT<T>*, blasint, blasint*, T*,   \
                            blasint, RealT<T>*);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// LTRAN and LREAL are Fortran LOGICALs, SELECT a LOGICAL array (nonzero is
// true).
#define HWYBLAS_LAPACK_TRSNA_REAL(prefix, R)                                 \
  HWYBLAS_API void prefix##laqtr_(                                           \
      const hwyblas_int* ltran, const hwyblas_int* lreal,                    \
      const hwyblas_int* n, const R* t, const hwyblas_int* ldt, const R* b,  \
      const R* w, R* scale, R* x, R* work, hwyblas_int* info) {              \
    *info = hwyblas::lapack::Laqtr(*ltran != 0, *lreal != 0, *n, t, *ldt, b, \
                                   *w, scale, x, work);                      \
  }                                                                          \
  HWYBLAS_API void prefix##trsna_(                                           \
      const char* job, const char* howmny, const hwyblas_int* select,        \
      const hwyblas_int* n, const R* t, const hwyblas_int* ldt, const R* vl, \
      const hwyblas_int* ldvl, const R* vr, const hwyblas_int* ldvr, R* s,   \
      R* sep, const hwyblas_int* mm, hwyblas_int* m, R* work,                \
      const hwyblas_int* ldwork, hwyblas_int* iwork, hwyblas_int* info) {    \
    *info = hwyblas::lapack::Trsna(*job, *howmny, select, *n, t, *ldt, vl,   \
                                   *ldvl, vr, *ldvr, s, sep, *mm, m, work,   \
                                   *ldwork, iwork);                          \
  }

HWYBLAS_LAPACK_TRSNA_REAL(s, float)
HWYBLAS_LAPACK_TRSNA_REAL(d, double)

#undef HWYBLAS_LAPACK_TRSNA_REAL

#define HWYBLAS_LAPACK_TRSNA_COMPLEX(prefix, T, R)                            \
  HWYBLAS_API void prefix##trsna_(                                            \
      const char* job, const char* howmny, const hwyblas_int* select,         \
      const hwyblas_int* n, const R* t, const hwyblas_int* ldt, const R* vl,  \
      const hwyblas_int* ldvl, const R* vr, const hwyblas_int* ldvr, R* s,    \
      R* sep, const hwyblas_int* mm, hwyblas_int* m, R* work,                 \
      const hwyblas_int* ldwork, R* rwork, hwyblas_int* info) {               \
    *info = hwyblas::lapack::Trsna(*job, *howmny, select, *n, As<const T>(t), \
                                   *ldt, As<const T>(vl), *ldvl,              \
                                   As<const T>(vr), *ldvr, s, sep, *mm, m,    \
                                   As<T>(work), *ldwork, rwork);              \
  }

HWYBLAS_LAPACK_TRSNA_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_TRSNA_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_TRSNA_COMPLEX

}  // extern "C"
