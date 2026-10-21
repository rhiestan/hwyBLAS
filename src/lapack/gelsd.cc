// Linear least squares by the singular value decomposition with divide and
// conquer: xGELSD, which reduces A to bidiagonal form and solves the
// bidiagonal problem with xLALSD (divide and conquer of xLASDA, applied to
// the right-hand sides by xLALSA and xLALS0).
//
// The complex xLALSD, xLALSA and xLALS0 apply the real singular vectors to
// the real and imaginary parts of B separately (in RWORK); they are
// separate bodies behind one template, as xGELSD (whose workspaces
// differ).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>
#include <complex>
#include <type_traits>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// DLALS0 (real T).
template <typename T>
blasint Lals0Real(blasint icompq, blasint nl, blasint nr, blasint sqre,
                  blasint nrhs, T* b, blasint ldb, T* bx, blasint ldbx,
                  blasint* perm, blasint givptr, blasint* givcol,
                  blasint ldgcol, T* givnum, blasint ldgnum, T* poles, T* difl,
                  T* difr, T* z, blasint k, T c, T s, T* work) {
  using R = T;

  const R one = R(1);
  const R zero = R(0);
  const R negone = -R(1);
  const Matrix<R> B(b, ldb);
  const Matrix<R> BX(bx, ldbx);
  const Vector<R> DIFL(difl);
  const Matrix<R> DIFR(difr, ldgnum);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldgnum);
  const Vector<blasint> PERM(perm);
  const Matrix<R> POLES(poles, ldgnum);
  const Vector<R> WORK(work);
  const Vector<R> Z(z);
  blasint i, j, m, n, nlp1;
  R diflj, difrj, dj, dsigj, dsigjp, temp;
  blasint info = 0;
  n = nl + nr + 1;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (nl < 1) {
    info = -2;
  } else if (nr < 1) {
    info = -3;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -4;
  } else if (nrhs < 1) {
    info = -5;
  } else if (ldb < n) {
    info = -7;
  } else if (ldbx < n) {
    info = -9;
  } else if (givptr < 0) {
    info = -11;
  } else if (ldgcol < n) {
    info = -13;
  } else if (ldgnum < n) {
    info = -15;
  } else if (k < 1) {
    info = -20;
  }
  if (info != 0) {
    Xerbla<T>("LALS0", -info);
    return info;
  }
  m = n + sqre;
  nlp1 = nl + 1;
  if (icompq == 0) {
    // Apply back orthogonal transformations from the left.
    // Step (1L): apply back the Givens rotations performed.
    for (i = 1; i <= givptr; ++i) {
      Rot(nrhs, B.ptr(GIVCOL(i, 2), 1), ldb, B.ptr(GIVCOL(i, 1), 1), ldb,
          GIVNUM(i, 2), GIVNUM(i, 1));
    }
    // Step (2L): permute rows of B.
    Copy(nrhs, B.ptr(nlp1, 1), ldb, BX.ptr(1, 1), ldbx);
    for (i = 2; i <= n; ++i) {
      Copy(nrhs, B.ptr(PERM(i), 1), ldb, BX.ptr(i, 1), ldbx);
    }
    // Step (3L): apply the inverse of the left singular vector
    // matrix to BX.
    if (k == 1) {
      Copy(nrhs, bx, ldbx, b, ldb);
      if (Z(1) < zero) {
        Rscal(nrhs, negone, b, ldb);
      }
    } else {
      for (j = 1; j <= k; ++j) {
        diflj = DIFL(j);
        dj = POLES(j, 1);
        dsigj = -POLES(j, 2);
        if (j < k) {
          difrj = -DIFR(j, 1);
          dsigjp = -POLES(j + 1, 2);
        }
        if ((Z(j) == zero) || (POLES(j, 2) == zero)) {
          WORK(j) = zero;
        } else {
          WORK(j) = -POLES(j, 2) * Z(j) / diflj / (POLES(j, 2) + dj);
        }
        for (i = 1; i <= j - 1; ++i) {
          if ((Z(i) == zero) || (POLES(i, 2) == zero)) {
            WORK(i) = zero;
          } else {
            // Use calls to the subroutine DLAMC3 to enforce the
            // parentheses (x+y)+z. The goal is to prevent
            // optimizing compilers from doing x+(y+z).
            WORK(i) = POLES(i, 2) * Z(i) / (Lamc3(POLES(i, 2), dsigj) - diflj) /
                      (POLES(i, 2) + dj);
          }
        }
        for (i = j + 1; i <= k; ++i) {
          if ((Z(i) == zero) || (POLES(i, 2) == zero)) {
            WORK(i) = zero;
          } else {
            WORK(i) = POLES(i, 2) * Z(i) /
                      (Lamc3(POLES(i, 2), dsigjp) + difrj) / (POLES(i, 2) + dj);
          }
        }
        WORK(1) = negone;
        temp = Nrm2(k, work, 1);
        Gemv('T', k, nrhs, one, bx, ldbx, work, 1, zero, B.ptr(j, 1), ldb);
        info = Lascl('G', 0, 0, temp, one, 1, nrhs, B.ptr(j, 1), ldb);
      }
    }
    // Move the deflated rows of BX to B also.
    if (k < Max<blasint>(m, n))
      Lacpy('A', n - k, nrhs, BX.ptr(k + 1, 1), ldbx, B.ptr(k + 1, 1), ldb);
  } else {
    // Apply back the right orthogonal transformations.
    // Step (1R): apply back the new right singular vector matrix
    // to B.
    if (k == 1) {
      Copy(nrhs, b, ldb, bx, ldbx);
    } else {
      for (j = 1; j <= k; ++j) {
        dsigj = POLES(j, 2);
        if (Z(j) == zero) {
          WORK(j) = zero;
        } else {
          WORK(j) = -Z(j) / DIFL(j) / (dsigj + POLES(j, 1)) / DIFR(j, 2);
        }
        for (i = 1; i <= j - 1; ++i) {
          if (Z(j) == zero) {
            WORK(i) = zero;
          } else {
            // Use calls to the subroutine DLAMC3 to enforce the
            // parentheses (x+y)+z. The goal is to prevent
            // optimizing compilers from doing x+(y+z).
            WORK(i) = Z(j) / (Lamc3(dsigj, -POLES(i + 1, 2)) - DIFR(i, 1)) /
                      (dsigj + POLES(i, 1)) / DIFR(i, 2);
          }
        }
        for (i = j + 1; i <= k; ++i) {
          if (Z(j) == zero) {
            WORK(i) = zero;
          } else {
            WORK(i) = Z(j) / (Lamc3(dsigj, -POLES(i, 2)) - DIFL(i)) /
                      (dsigj + POLES(i, 1)) / DIFR(i, 2);
          }
        }
        Gemv('T', k, nrhs, one, b, ldb, work, 1, zero, BX.ptr(j, 1), ldbx);
      }
    }
    // Step (2R): if SQRE = 1, apply back the rotation that is
    // related to the right null space of the subproblem.
    if (sqre == 1) {
      Copy(nrhs, B.ptr(m, 1), ldb, BX.ptr(m, 1), ldbx);
      Rot(nrhs, BX.ptr(1, 1), ldbx, BX.ptr(m, 1), ldbx, c, s);
    }
    if (k < Max<blasint>(m, n))
      Lacpy('A', n - k, nrhs, B.ptr(k + 1, 1), ldb, BX.ptr(k + 1, 1), ldbx);
    // Step (3R): permute rows of B.
    Copy(nrhs, BX.ptr(1, 1), ldbx, B.ptr(nlp1, 1), ldb);
    if (sqre == 1) {
      Copy(nrhs, BX.ptr(m, 1), ldbx, B.ptr(m, 1), ldb);
    }
    for (i = 2; i <= n; ++i) {
      Copy(nrhs, BX.ptr(i, 1), ldbx, B.ptr(PERM(i), 1), ldb);
    }
    // Step (4R): apply back the Givens rotations performed.
    for (i = givptr; i >= 1; --i) {
      Rot(nrhs, B.ptr(GIVCOL(i, 2), 1), ldb, B.ptr(GIVCOL(i, 1), 1), ldb,
          GIVNUM(i, 2), -GIVNUM(i, 1));
    }
  }
  return info;
}

// ZLALS0 (complex T).
template <typename T>
blasint Lals0Complex(blasint icompq, blasint nl, blasint nr, blasint sqre,
                     blasint nrhs, T* b, blasint ldb, T* bx, blasint ldbx,
                     blasint* perm, blasint givptr, blasint* givcol,
                     blasint ldgcol, RealT<T>* givnum, blasint ldgnum,
                     RealT<T>* poles, RealT<T>* difl, RealT<T>* difr,
                     RealT<T>* z, blasint k, RealT<T> c, RealT<T> s,
                     RealT<T>* rwork) {
  using R = RealT<T>;

  const R one = R(1);
  const R zero = R(0);
  const R negone = -R(1);
  const Matrix<T> B(b, ldb);
  const Matrix<T> BX(bx, ldbx);
  const Vector<R> DIFL(difl);
  const Matrix<R> DIFR(difr, ldgnum);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldgnum);
  const Vector<blasint> PERM(perm);
  const Matrix<R> POLES(poles, ldgnum);
  const Vector<R> RWORK(rwork);
  const Vector<R> Z(z);
  blasint i, j, jcol, jrow, m, n, nlp1;
  R diflj, difrj, dj, dsigj, dsigjp, temp;
  blasint info = 0;
  n = nl + nr + 1;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (nl < 1) {
    info = -2;
  } else if (nr < 1) {
    info = -3;
  } else if ((sqre < 0) || (sqre > 1)) {
    info = -4;
  } else if (nrhs < 1) {
    info = -5;
  } else if (ldb < n) {
    info = -7;
  } else if (ldbx < n) {
    info = -9;
  } else if (givptr < 0) {
    info = -11;
  } else if (ldgcol < n) {
    info = -13;
  } else if (ldgnum < n) {
    info = -15;
  } else if (k < 1) {
    info = -20;
  }
  if (info != 0) {
    Xerbla<T>("LALS0", -info);
    return info;
  }
  m = n + sqre;
  nlp1 = nl + 1;
  if (icompq == 0) {
    // Apply back orthogonal transformations from the left.
    // Step (1L): apply back the Givens rotations performed.
    for (i = 1; i <= givptr; ++i) {
      Rot(nrhs, B.ptr(GIVCOL(i, 2), 1), ldb, B.ptr(GIVCOL(i, 1), 1), ldb,
          GIVNUM(i, 2), GIVNUM(i, 1));
    }
    // Step (2L): permute rows of B.
    Copy(nrhs, B.ptr(nlp1, 1), ldb, BX.ptr(1, 1), ldbx);
    for (i = 2; i <= n; ++i) {
      Copy(nrhs, B.ptr(PERM(i), 1), ldb, BX.ptr(i, 1), ldbx);
    }
    // Step (3L): apply the inverse of the left singular vector
    // matrix to BX.
    if (k == 1) {
      Copy(nrhs, bx, ldbx, b, ldb);
      if (Z(1) < zero) {
        Rscal(nrhs, negone, b, ldb);
      }
    } else {
      for (j = 1; j <= k; ++j) {
        diflj = DIFL(j);
        dj = POLES(j, 1);
        dsigj = -POLES(j, 2);
        if (j < k) {
          difrj = -DIFR(j, 1);
          dsigjp = -POLES(j + 1, 2);
        }
        if ((Z(j) == zero) || (POLES(j, 2) == zero)) {
          RWORK(j) = zero;
        } else {
          RWORK(j) = -POLES(j, 2) * Z(j) / diflj / (POLES(j, 2) + dj);
        }
        for (i = 1; i <= j - 1; ++i) {
          if ((Z(i) == zero) || (POLES(i, 2) == zero)) {
            RWORK(i) = zero;
          } else {
            // Use calls to the subroutine DLAMC3 to enforce the
            // parentheses (x+y)+z. The goal is to prevent
            // optimizing compilers from doing x+(y+z).
            RWORK(i) = POLES(i, 2) * Z(i) /
                       (Lamc3<R>(POLES(i, 2), dsigj) - diflj) /
                       (POLES(i, 2) + dj);
          }
        }
        for (i = j + 1; i <= k; ++i) {
          if ((Z(i) == zero) || (POLES(i, 2) == zero)) {
            RWORK(i) = zero;
          } else {
            RWORK(i) = POLES(i, 2) * Z(i) /
                       (Lamc3<R>(POLES(i, 2), dsigjp) + difrj) /
                       (POLES(i, 2) + dj);
          }
        }
        RWORK(1) = negone;
        temp = Nrm2<R>(k, rwork, 1);
        // Since B and BX are complex, the following call to DGEMV
        // is performed in two steps (real and imaginary parts).
        // CALL DGEMV( 'T', K, NRHS, ONE, BX, LDBX, WORK, 1, ZERO,
        // $                     B( J, 1 ), LDB )
        i = k + nrhs * 2;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = 1; jrow <= k; ++jrow) {
            i = i + 1;
            RWORK(i) = std::real(BX(jrow, jcol));
          }
        }
        Gemv<R>('T', k, nrhs, one, RWORK.ptr(1 + k + nrhs * 2), k, RWORK.ptr(1),
                1, zero, RWORK.ptr(1 + k), 1);
        i = k + nrhs * 2;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = 1; jrow <= k; ++jrow) {
            i = i + 1;
            RWORK(i) = std::imag(BX(jrow, jcol));
          }
        }
        Gemv<R>('T', k, nrhs, one, RWORK.ptr(1 + k + nrhs * 2), k, RWORK.ptr(1),
                1, zero, RWORK.ptr(1 + k + nrhs), 1);
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          B(j, jcol) = T(RWORK(jcol + k), RWORK(jcol + k + nrhs));
        }
        info = Lascl('G', 0, 0, temp, one, 1, nrhs, B.ptr(j, 1), ldb);
      }
    }
    // Move the deflated rows of BX to B also.
    if (k < Max<blasint>(m, n))
      Lacpy('A', n - k, nrhs, BX.ptr(k + 1, 1), ldbx, B.ptr(k + 1, 1), ldb);
  } else {
    // Apply back the right orthogonal transformations.
    // Step (1R): apply back the new right singular vector matrix
    // to B.
    if (k == 1) {
      Copy(nrhs, b, ldb, bx, ldbx);
    } else {
      for (j = 1; j <= k; ++j) {
        dsigj = POLES(j, 2);
        if (Z(j) == zero) {
          RWORK(j) = zero;
        } else {
          RWORK(j) = -Z(j) / DIFL(j) / (dsigj + POLES(j, 1)) / DIFR(j, 2);
        }
        for (i = 1; i <= j - 1; ++i) {
          if (Z(j) == zero) {
            RWORK(i) = zero;
          } else {
            // Use calls to the subroutine DLAMC3 to enforce the
            // parentheses (x+y)+z. The goal is to prevent
            // optimizing compilers from doing x+(y+z).
            RWORK(i) = Z(j) / (Lamc3<R>(dsigj, -POLES(i + 1, 2)) - DIFR(i, 1)) /
                       (dsigj + POLES(i, 1)) / DIFR(i, 2);
          }
        }
        for (i = j + 1; i <= k; ++i) {
          if (Z(j) == zero) {
            RWORK(i) = zero;
          } else {
            RWORK(i) = Z(j) / (Lamc3<R>(dsigj, -POLES(i, 2)) - DIFL(i)) /
                       (dsigj + POLES(i, 1)) / DIFR(i, 2);
          }
        }
        // Since B and BX are complex, the following call to DGEMV
        // is performed in two steps (real and imaginary parts).
        // CALL DGEMV( 'T', K, NRHS, ONE, B, LDB, WORK, 1, ZERO,
        // $                     BX( J, 1 ), LDBX )
        i = k + nrhs * 2;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = 1; jrow <= k; ++jrow) {
            i = i + 1;
            RWORK(i) = std::real(B(jrow, jcol));
          }
        }
        Gemv<R>('T', k, nrhs, one, RWORK.ptr(1 + k + nrhs * 2), k, RWORK.ptr(1),
                1, zero, RWORK.ptr(1 + k), 1);
        i = k + nrhs * 2;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = 1; jrow <= k; ++jrow) {
            i = i + 1;
            RWORK(i) = std::imag(B(jrow, jcol));
          }
        }
        Gemv<R>('T', k, nrhs, one, RWORK.ptr(1 + k + nrhs * 2), k, RWORK.ptr(1),
                1, zero, RWORK.ptr(1 + k + nrhs), 1);
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          BX(j, jcol) = T(RWORK(jcol + k), RWORK(jcol + k + nrhs));
        }
      }
    }
    // Step (2R): if SQRE = 1, apply back the rotation that is
    // related to the right null space of the subproblem.
    if (sqre == 1) {
      Copy(nrhs, B.ptr(m, 1), ldb, BX.ptr(m, 1), ldbx);
      Rot(nrhs, BX.ptr(1, 1), ldbx, BX.ptr(m, 1), ldbx, c, s);
    }
    if (k < Max<blasint>(m, n))
      Lacpy('A', n - k, nrhs, B.ptr(k + 1, 1), ldb, BX.ptr(k + 1, 1), ldbx);
    // Step (3R): permute rows of B.
    Copy(nrhs, BX.ptr(1, 1), ldbx, B.ptr(nlp1, 1), ldb);
    if (sqre == 1) {
      Copy(nrhs, BX.ptr(m, 1), ldbx, B.ptr(m, 1), ldb);
    }
    for (i = 2; i <= n; ++i) {
      Copy(nrhs, BX.ptr(i, 1), ldbx, B.ptr(PERM(i), 1), ldb);
    }
    // Step (4R): apply back the Givens rotations performed.
    for (i = givptr; i >= 1; --i) {
      Rot(nrhs, B.ptr(GIVCOL(i, 2), 1), ldb, B.ptr(GIVCOL(i, 1), 1), ldb,
          GIVNUM(i, 2), -GIVNUM(i, 1));
    }
  }
  return info;
}

// DLALSA (real T).
template <typename T>
blasint LalsaReal(blasint icompq, blasint smlsiz, blasint n, blasint nrhs, T* b,
                  blasint ldb, T* bx, blasint ldbx, T* u, blasint ldu, T* vt,
                  blasint* k, T* difl, T* difr, T* z, T* poles, blasint* givptr,
                  blasint* givcol, blasint ldgcol, blasint* perm, T* givnum,
                  T* c, T* s, T* work, blasint* iwork) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<R> B(b, ldb);
  const Matrix<R> BX(bx, ldbx);
  const Vector<R> C(c);
  const Matrix<R> DIFL(difl, ldu);
  const Matrix<R> DIFR(difr, ldu);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldu);
  const Vector<blasint> GIVPTR(givptr);
  const Vector<blasint> IWORK(iwork);
  const Vector<blasint> K(k);
  const Matrix<blasint> PERM(perm, ldgcol);
  const Matrix<R> POLES(poles, ldu);
  const Vector<R> S(s);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldu);
  const Vector<R> WORK(work);
  const Matrix<R> Z(z, ldu);
  blasint i, i1, ic, im1, inode, j, lf, ll, lvl, lvl2, nd, ndb1, ndiml, ndimr,
      nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqre;
  blasint info = 0;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (smlsiz < 3) {
    info = -2;
  } else if (n < smlsiz) {
    info = -3;
  } else if (nrhs < 1) {
    info = -4;
  } else if (ldb < n) {
    info = -6;
  } else if (ldbx < n) {
    info = -8;
  } else if (ldu < n) {
    info = -10;
  } else if (ldgcol < n) {
    info = -19;
  }
  if (info != 0) {
    Xerbla<T>("LALSA", -info);
    return info;
  }
  // Book-keeping and  setting up the computation tree.
  inode = 1;
  ndiml = inode + n;
  ndimr = ndiml + n;
  Lasdt<R>(n, &nlvl, &nd, IWORK.ptr(inode), IWORK.ptr(ndiml), IWORK.ptr(ndimr),
           smlsiz);
  // The following code applies back the left singular vector factors.
  // For applying back the right singular vector factors, go to 50.
  if (icompq == 1) {
    goto L50;
  }
  // The nodes on the bottom level of the tree were solved
  // by DLASDQ. The corresponding left and right singular vector
  // matrices are in explicit form. First apply back the left
  // singular vector matrices.
  ndb1 = (nd + 1) / 2;
  for (i = ndb1; i <= nd; ++i) {
    // IC : center row of each node
    // NL : number of rows of left  subproblem
    // NR : number of rows of right subproblem
    // NLF: starting row of the left   subproblem
    // NRF: starting row of the right  subproblem
    i1 = i - 1;
    ic = IWORK(inode + i1);
    nl = IWORK(ndiml + i1);
    nr = IWORK(ndimr + i1);
    nlf = ic - nl;
    nrf = ic + 1;
    Gemm('T', 'N', nl, nrhs, nl, one, U.ptr(nlf, 1), ldu, B.ptr(nlf, 1), ldb,
         zero, BX.ptr(nlf, 1), ldbx);
    Gemm('T', 'N', nr, nrhs, nr, one, U.ptr(nrf, 1), ldu, B.ptr(nrf, 1), ldb,
         zero, BX.ptr(nrf, 1), ldbx);
  }
  // Next copy the rows of B that correspond to unchanged rows
  // in the bidiagonal matrix to BX.
  for (i = 1; i <= nd; ++i) {
    ic = IWORK(inode + i - 1);
    Copy(nrhs, B.ptr(ic, 1), ldb, BX.ptr(ic, 1), ldbx);
  }
  // Finally go through the left singular vector matrices of all
  // the other subproblems bottom-up on the tree.
  j = blasint{1} << nlvl;
  sqre = 0;
  for (lvl = nlvl; lvl >= 1; --lvl) {
    lvl2 = 2 * lvl - 1;
    // find the first node LF and last node LL on
    // the current level LVL
    if (lvl == 1) {
      lf = 1;
      ll = 1;
    } else {
      lf = blasint{1} << (lvl - 1);
      ll = 2 * lf - 1;
    }
    for (i = lf; i <= ll; ++i) {
      im1 = i - 1;
      ic = IWORK(inode + im1);
      nl = IWORK(ndiml + im1);
      nr = IWORK(ndimr + im1);
      nlf = ic - nl;
      nrf = ic + 1;
      j = j - 1;
      info = Lals0(
          icompq, nl, nr, sqre, nrhs, BX.ptr(nlf, 1), ldbx, B.ptr(nlf, 1), ldb,
          PERM.ptr(nlf, lvl), GIVPTR(j), GIVCOL.ptr(nlf, lvl2), ldgcol,
          GIVNUM.ptr(nlf, lvl2), ldu, POLES.ptr(nlf, lvl2), DIFL.ptr(nlf, lvl),
          DIFR.ptr(nlf, lvl2), Z.ptr(nlf, lvl), K(j), C(j), S(j), work);
    }
  }
  goto L90;
  // ICOMPQ = 1: applying back the right singular vector factors.
L50:
  // First now go through the right singular vector matrices of all
  // the tree nodes top-down.
  j = 0;
  for (lvl = 1; lvl <= nlvl; ++lvl) {
    lvl2 = 2 * lvl - 1;
    // Find the first node LF and last node LL on
    // the current level LVL.
    if (lvl == 1) {
      lf = 1;
      ll = 1;
    } else {
      lf = blasint{1} << (lvl - 1);
      ll = 2 * lf - 1;
    }
    for (i = ll; i >= lf; --i) {
      im1 = i - 1;
      ic = IWORK(inode + im1);
      nl = IWORK(ndiml + im1);
      nr = IWORK(ndimr + im1);
      nlf = ic - nl;
      nrf = ic + 1;
      if (i == ll) {
        sqre = 0;
      } else {
        sqre = 1;
      }
      j = j + 1;
      info = Lals0(
          icompq, nl, nr, sqre, nrhs, B.ptr(nlf, 1), ldb, BX.ptr(nlf, 1), ldbx,
          PERM.ptr(nlf, lvl), GIVPTR(j), GIVCOL.ptr(nlf, lvl2), ldgcol,
          GIVNUM.ptr(nlf, lvl2), ldu, POLES.ptr(nlf, lvl2), DIFL.ptr(nlf, lvl),
          DIFR.ptr(nlf, lvl2), Z.ptr(nlf, lvl), K(j), C(j), S(j), work);
    }
  }
  // The nodes on the bottom level of the tree were solved
  // by DLASDQ. The corresponding right singular vector
  // matrices are in explicit form. Apply them back.
  ndb1 = (nd + 1) / 2;
  for (i = ndb1; i <= nd; ++i) {
    i1 = i - 1;
    ic = IWORK(inode + i1);
    nl = IWORK(ndiml + i1);
    nr = IWORK(ndimr + i1);
    nlp1 = nl + 1;
    if (i == nd) {
      nrp1 = nr;
    } else {
      nrp1 = nr + 1;
    }
    nlf = ic - nl;
    nrf = ic + 1;
    Gemm('T', 'N', nlp1, nrhs, nlp1, one, VT.ptr(nlf, 1), ldu, B.ptr(nlf, 1),
         ldb, zero, BX.ptr(nlf, 1), ldbx);
    Gemm('T', 'N', nrp1, nrhs, nrp1, one, VT.ptr(nrf, 1), ldu, B.ptr(nrf, 1),
         ldb, zero, BX.ptr(nrf, 1), ldbx);
  }
L90:
  return info;
}

// ZLALSA (complex T).
template <typename T>
blasint LalsaComplex(blasint icompq, blasint smlsiz, blasint n, blasint nrhs,
                     T* b, blasint ldb, T* bx, blasint ldbx, RealT<T>* u,
                     blasint ldu, RealT<T>* vt, blasint* k, RealT<T>* difl,
                     RealT<T>* difr, RealT<T>* z, RealT<T>* poles,
                     blasint* givptr, blasint* givcol, blasint ldgcol,
                     blasint* perm, RealT<T>* givnum, RealT<T>* c, RealT<T>* s,
                     RealT<T>* rwork, blasint* iwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const Matrix<T> B(b, ldb);
  const Matrix<T> BX(bx, ldbx);
  const Vector<R> C(c);
  const Matrix<R> DIFL(difl, ldu);
  const Matrix<R> DIFR(difr, ldu);
  const Matrix<blasint> GIVCOL(givcol, ldgcol);
  const Matrix<R> GIVNUM(givnum, ldu);
  const Vector<blasint> GIVPTR(givptr);
  const Vector<blasint> IWORK(iwork);
  const Vector<blasint> K(k);
  const Matrix<blasint> PERM(perm, ldgcol);
  const Matrix<R> POLES(poles, ldu);
  const Vector<R> RWORK(rwork);
  const Vector<R> S(s);
  const Matrix<R> U(u, ldu);
  const Matrix<R> VT(vt, ldu);
  const Matrix<R> Z(z, ldu);
  blasint i, i1, ic, im1, inode, j, jcol, jimag, jreal, jrow, lf, ll, lvl, lvl2,
      nd, ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqre;
  blasint info = 0;
  if ((icompq < 0) || (icompq > 1)) {
    info = -1;
  } else if (smlsiz < 3) {
    info = -2;
  } else if (n < smlsiz) {
    info = -3;
  } else if (nrhs < 1) {
    info = -4;
  } else if (ldb < n) {
    info = -6;
  } else if (ldbx < n) {
    info = -8;
  } else if (ldu < n) {
    info = -10;
  } else if (ldgcol < n) {
    info = -19;
  }
  if (info != 0) {
    Xerbla<T>("LALSA", -info);
    return info;
  }
  // Book-keeping and  setting up the computation tree.
  inode = 1;
  ndiml = inode + n;
  ndimr = ndiml + n;
  Lasdt<R>(n, &nlvl, &nd, IWORK.ptr(inode), IWORK.ptr(ndiml), IWORK.ptr(ndimr),
           smlsiz);
  // The following code applies back the left singular vector factors.
  // For applying back the right singular vector factors, go to 170.
  if (icompq == 1) {
    goto L170;
  }
  // The nodes on the bottom level of the tree were solved
  // by DLASDQ. The corresponding left and right singular vector
  // matrices are in explicit form. First apply back the left
  // singular vector matrices.
  ndb1 = (nd + 1) / 2;
  for (i = ndb1; i <= nd; ++i) {
    // IC : center row of each node
    // NL : number of rows of left  subproblem
    // NR : number of rows of right subproblem
    // NLF: starting row of the left   subproblem
    // NRF: starting row of the right  subproblem
    i1 = i - 1;
    ic = IWORK(inode + i1);
    nl = IWORK(ndiml + i1);
    nr = IWORK(ndimr + i1);
    nlf = ic - nl;
    nrf = ic + 1;
    // Since B and BX are complex, the following call to DGEMM
    // is performed in two steps (real and imaginary parts).
    // CALL DGEMM( 'T', 'N', NL, NRHS, NL, ONE, U( NLF, 1 ), LDU,
    // $               B( NLF, 1 ), LDB, ZERO, BX( NLF, 1 ), LDBX )
    j = nl * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nlf; jrow <= nlf + nl - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::real(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nl, nrhs, nl, one, U.ptr(nlf, 1), ldu,
            RWORK.ptr(1 + nl * nrhs * 2), nl, zero, RWORK.ptr(1), nl);
    j = nl * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nlf; jrow <= nlf + nl - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::imag(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nl, nrhs, nl, one, U.ptr(nlf, 1), ldu,
            RWORK.ptr(1 + nl * nrhs * 2), nl, zero, RWORK.ptr(1 + nl * nrhs),
            nl);
    jreal = 0;
    jimag = nl * nrhs;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nlf; jrow <= nlf + nl - 1; ++jrow) {
        jreal = jreal + 1;
        jimag = jimag + 1;
        BX(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
      }
    }
    // Since B and BX are complex, the following call to DGEMM
    // is performed in two steps (real and imaginary parts).
    // CALL DGEMM( 'T', 'N', NR, NRHS, NR, ONE, U( NRF, 1 ), LDU,
    // $               B( NRF, 1 ), LDB, ZERO, BX( NRF, 1 ), LDBX )
    j = nr * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nrf; jrow <= nrf + nr - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::real(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nr, nrhs, nr, one, U.ptr(nrf, 1), ldu,
            RWORK.ptr(1 + nr * nrhs * 2), nr, zero, RWORK.ptr(1), nr);
    j = nr * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nrf; jrow <= nrf + nr - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::imag(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nr, nrhs, nr, one, U.ptr(nrf, 1), ldu,
            RWORK.ptr(1 + nr * nrhs * 2), nr, zero, RWORK.ptr(1 + nr * nrhs),
            nr);
    jreal = 0;
    jimag = nr * nrhs;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nrf; jrow <= nrf + nr - 1; ++jrow) {
        jreal = jreal + 1;
        jimag = jimag + 1;
        BX(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
      }
    }
  }
  // Next copy the rows of B that correspond to unchanged rows
  // in the bidiagonal matrix to BX.
  for (i = 1; i <= nd; ++i) {
    ic = IWORK(inode + i - 1);
    Copy(nrhs, B.ptr(ic, 1), ldb, BX.ptr(ic, 1), ldbx);
  }
  // Finally go through the left singular vector matrices of all
  // the other subproblems bottom-up on the tree.
  j = blasint{1} << nlvl;
  sqre = 0;
  for (lvl = nlvl; lvl >= 1; --lvl) {
    lvl2 = 2 * lvl - 1;
    // find the first node LF and last node LL on
    // the current level LVL
    if (lvl == 1) {
      lf = 1;
      ll = 1;
    } else {
      lf = blasint{1} << (lvl - 1);
      ll = 2 * lf - 1;
    }
    for (i = lf; i <= ll; ++i) {
      im1 = i - 1;
      ic = IWORK(inode + im1);
      nl = IWORK(ndiml + im1);
      nr = IWORK(ndimr + im1);
      nlf = ic - nl;
      nrf = ic + 1;
      j = j - 1;
      info = Lals0(
          icompq, nl, nr, sqre, nrhs, BX.ptr(nlf, 1), ldbx, B.ptr(nlf, 1), ldb,
          PERM.ptr(nlf, lvl), GIVPTR(j), GIVCOL.ptr(nlf, lvl2), ldgcol,
          GIVNUM.ptr(nlf, lvl2), ldu, POLES.ptr(nlf, lvl2), DIFL.ptr(nlf, lvl),
          DIFR.ptr(nlf, lvl2), Z.ptr(nlf, lvl), K(j), C(j), S(j), rwork);
    }
  }
  goto L330;
  // ICOMPQ = 1: applying back the right singular vector factors.
L170:
  // First now go through the right singular vector matrices of all
  // the tree nodes top-down.
  j = 0;
  for (lvl = 1; lvl <= nlvl; ++lvl) {
    lvl2 = 2 * lvl - 1;
    // Find the first node LF and last node LL on
    // the current level LVL.
    if (lvl == 1) {
      lf = 1;
      ll = 1;
    } else {
      lf = blasint{1} << (lvl - 1);
      ll = 2 * lf - 1;
    }
    for (i = ll; i >= lf; --i) {
      im1 = i - 1;
      ic = IWORK(inode + im1);
      nl = IWORK(ndiml + im1);
      nr = IWORK(ndimr + im1);
      nlf = ic - nl;
      nrf = ic + 1;
      if (i == ll) {
        sqre = 0;
      } else {
        sqre = 1;
      }
      j = j + 1;
      info = Lals0(
          icompq, nl, nr, sqre, nrhs, B.ptr(nlf, 1), ldb, BX.ptr(nlf, 1), ldbx,
          PERM.ptr(nlf, lvl), GIVPTR(j), GIVCOL.ptr(nlf, lvl2), ldgcol,
          GIVNUM.ptr(nlf, lvl2), ldu, POLES.ptr(nlf, lvl2), DIFL.ptr(nlf, lvl),
          DIFR.ptr(nlf, lvl2), Z.ptr(nlf, lvl), K(j), C(j), S(j), rwork);
    }
  }
  // The nodes on the bottom level of the tree were solved
  // by DLASDQ. The corresponding right singular vector
  // matrices are in explicit form. Apply them back.
  ndb1 = (nd + 1) / 2;
  for (i = ndb1; i <= nd; ++i) {
    i1 = i - 1;
    ic = IWORK(inode + i1);
    nl = IWORK(ndiml + i1);
    nr = IWORK(ndimr + i1);
    nlp1 = nl + 1;
    if (i == nd) {
      nrp1 = nr;
    } else {
      nrp1 = nr + 1;
    }
    nlf = ic - nl;
    nrf = ic + 1;
    // Since B and BX are complex, the following call to DGEMM is
    // performed in two steps (real and imaginary parts).
    // CALL DGEMM( 'T', 'N', NLP1, NRHS, NLP1, ONE, VT( NLF, 1 ), LDU,
    // $               B( NLF, 1 ), LDB, ZERO, BX( NLF, 1 ), LDBX )
    j = nlp1 * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nlf; jrow <= nlf + nlp1 - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::real(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nlp1, nrhs, nlp1, one, VT.ptr(nlf, 1), ldu,
            RWORK.ptr(1 + nlp1 * nrhs * 2), nlp1, zero, RWORK.ptr(1), nlp1);
    j = nlp1 * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nlf; jrow <= nlf + nlp1 - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::imag(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nlp1, nrhs, nlp1, one, VT.ptr(nlf, 1), ldu,
            RWORK.ptr(1 + nlp1 * nrhs * 2), nlp1, zero,
            RWORK.ptr(1 + nlp1 * nrhs), nlp1);
    jreal = 0;
    jimag = nlp1 * nrhs;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nlf; jrow <= nlf + nlp1 - 1; ++jrow) {
        jreal = jreal + 1;
        jimag = jimag + 1;
        BX(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
      }
    }
    // Since B and BX are complex, the following call to DGEMM is
    // performed in two steps (real and imaginary parts).
    // CALL DGEMM( 'T', 'N', NRP1, NRHS, NRP1, ONE, VT( NRF, 1 ), LDU,
    // $               B( NRF, 1 ), LDB, ZERO, BX( NRF, 1 ), LDBX )
    j = nrp1 * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nrf; jrow <= nrf + nrp1 - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::real(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nrp1, nrhs, nrp1, one, VT.ptr(nrf, 1), ldu,
            RWORK.ptr(1 + nrp1 * nrhs * 2), nrp1, zero, RWORK.ptr(1), nrp1);
    j = nrp1 * nrhs * 2;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nrf; jrow <= nrf + nrp1 - 1; ++jrow) {
        j = j + 1;
        RWORK(j) = std::imag(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', nrp1, nrhs, nrp1, one, VT.ptr(nrf, 1), ldu,
            RWORK.ptr(1 + nrp1 * nrhs * 2), nrp1, zero,
            RWORK.ptr(1 + nrp1 * nrhs), nrp1);
    jreal = 0;
    jimag = nrp1 * nrhs;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = nrf; jrow <= nrf + nrp1 - 1; ++jrow) {
        jreal = jreal + 1;
        jimag = jimag + 1;
        BX(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
      }
    }
  }
L330:
  return info;
}

// DLALSD (real T).
template <typename T>
blasint LalsdReal(char uplo, blasint smlsiz, blasint n, blasint nrhs, T* d,
                  T* e, T* b, blasint ldb, T rcond, blasint* rank, T* work,
                  blasint* iwork) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Matrix<R> B(b, ldb);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> WORK(work);
  blasint bx, bxst, c, difl, difr, givcol, givnum, givptr, i, icmpq1, icmpq2,
      iwk, j, k, nlvl, nm1, nsize, nsub, nwork, perm, poles, s, sizei, smlszp,
      sqre, st, st1, u, vt, z;
  R cs, eps, orgnrm, r, rcnd, sn, tol;
  blasint info = 0;
  if (n < 0) {
    info = -3;
  } else if (nrhs < 1) {
    info = -4;
  } else if ((ldb < 1) || (ldb < n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<R>("LALSD", -info);
    return info;
  }
  eps = Lamch<R>('E');
  // Set up the tolerance.
  if ((rcond <= zero) || (rcond >= one)) {
    rcnd = eps;
  } else {
    rcnd = rcond;
  }
  *rank = 0;
  // Quick return if possible.
  if (n == 0) {
    return info;
  } else if (n == 1) {
    if (D(1) == zero) {
      Laset('A', 1, nrhs, zero, zero, b, ldb);
    } else {
      *rank = 1;
      info = Lascl('G', 0, 0, D(1), one, 1, nrhs, b, ldb);
      D(1) = std::abs(D(1));
    }
    return info;
  }
  // Rotate the matrix if it is lower bidiagonal.
  if (uplo == 'L') {
    for (i = 1; i <= n - 1; ++i) {
      Lartg(D(i), E(i), &cs, &sn, &r);
      D(i) = r;
      E(i) = sn * D(i + 1);
      D(i + 1) = cs * D(i + 1);
      if (nrhs == 1) {
        Rot(1, B.ptr(i, 1), 1, B.ptr(i + 1, 1), 1, cs, sn);
      } else {
        WORK(i * 2 - 1) = cs;
        WORK(i * 2) = sn;
      }
    }
    if (nrhs > 1) {
      for (i = 1; i <= nrhs; ++i) {
        for (j = 1; j <= n - 1; ++j) {
          cs = WORK(j * 2 - 1);
          sn = WORK(j * 2);
          Rot(1, B.ptr(j, i), 1, B.ptr(j + 1, i), 1, cs, sn);
        }
      }
    }
  }
  // Scale.
  nm1 = n - 1;
  orgnrm = Lanst('M', n, d, e);
  if (orgnrm == zero) {
    Laset('A', n, nrhs, zero, zero, b, ldb);
    return info;
  }
  info = Lascl('G', 0, 0, orgnrm, one, n, 1, d, n);
  info = Lascl('G', 0, 0, orgnrm, one, nm1, 1, e, nm1);
  // If N is smaller than the minimum divide size SMLSIZ, then solve
  // the problem with another solver.
  if (n <= smlsiz) {
    nwork = 1 + n * n;
    Laset('A', n, n, zero, one, work, n);
    info = Lasdq('U', 0, n, n, 0, nrhs, d, e, work, n, work, n, b, ldb,
                 WORK.ptr(nwork));
    if (info != 0) {
      return info;
    }
    tol = rcnd * std::abs(D(Iamax(n, d, 1)));
    for (i = 1; i <= n; ++i) {
      if (D(i) <= tol) {
        Laset('A', 1, nrhs, zero, zero, B.ptr(i, 1), ldb);
      } else {
        info = Lascl('G', 0, 0, D(i), one, 1, nrhs, B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      }
    }
    Gemm('T', 'N', n, nrhs, n, one, work, n, b, ldb, zero, WORK.ptr(nwork), n);
    Lacpy('A', n, nrhs, WORK.ptr(nwork), n, b, ldb);
    // Unscale.
    info = Lascl('G', 0, 0, one, orgnrm, n, 1, d, n);
    info = Lasrt('D', n, d);
    info = Lascl('G', 0, 0, orgnrm, one, n, nrhs, b, ldb);
    return info;
  }
  // Book-keeping and setting up some constants.
  nlvl =
      static_cast<blasint>(std::log(R(n) / R(smlsiz + 1)) / std::log(two)) + 1;
  smlszp = smlsiz + 1;
  u = 1;
  vt = 1 + smlsiz * n;
  difl = vt + smlszp * n;
  difr = difl + nlvl * n;
  z = difr + nlvl * n * 2;
  c = z + nlvl * n;
  s = c + n;
  poles = s + n;
  givnum = poles + 2 * nlvl * n;
  bx = givnum + 2 * nlvl * n;
  nwork = bx + n * nrhs;
  sizei = 1 + n;
  k = sizei + n;
  givptr = k + n;
  perm = givptr + n;
  givcol = perm + nlvl * n;
  iwk = givcol + nlvl * n * 2;
  st = 1;
  sqre = 0;
  icmpq1 = 1;
  icmpq2 = 0;
  nsub = 0;
  for (i = 1; i <= n; ++i) {
    if (std::abs(D(i)) < eps) {
      D(i) = std::copysign(eps, D(i));
    }
  }
  for (i = 1; i <= nm1; ++i) {
    if ((std::abs(E(i)) < eps) || (i == nm1)) {
      nsub = nsub + 1;
      IWORK(nsub) = st;
      // Subproblem found. First determine its size and then
      // apply divide and conquer on it.
      if (i < nm1) {
        // A subproblem with E(I) small for I < NM1.
        nsize = i - st + 1;
        IWORK(sizei + nsub - 1) = nsize;
      } else if (std::abs(E(i)) >= eps) {
        // A subproblem with E(NM1) not too small but I = NM1.
        nsize = n - st + 1;
        IWORK(sizei + nsub - 1) = nsize;
      } else {
        // A subproblem with E(NM1) small. This implies an
        // 1-by-1 subproblem at D(N), which is not solved
        // explicitly.
        nsize = i - st + 1;
        IWORK(sizei + nsub - 1) = nsize;
        nsub = nsub + 1;
        IWORK(nsub) = n;
        IWORK(sizei + nsub - 1) = 1;
        Copy(nrhs, B.ptr(n, 1), ldb, WORK.ptr(bx + nm1), n);
      }
      st1 = st - 1;
      if (nsize == 1) {
        // This is a 1-by-1 subproblem and is not solved
        // explicitly.
        Copy(nrhs, B.ptr(st, 1), ldb, WORK.ptr(bx + st1), n);
      } else if (nsize <= smlsiz) {
        // This is a small subproblem and is solved by DLASDQ.
        Laset('A', nsize, nsize, zero, one, WORK.ptr(vt + st1), n);
        info = Lasdq('U', 0, nsize, nsize, 0, nrhs, D.ptr(st), E.ptr(st),
                     WORK.ptr(vt + st1), n, WORK.ptr(nwork), n, B.ptr(st, 1),
                     ldb, WORK.ptr(nwork));
        if (info != 0) {
          return info;
        }
        Lacpy('A', nsize, nrhs, B.ptr(st, 1), ldb, WORK.ptr(bx + st1), n);
      } else {
        // A large problem. Solve it using divide and conquer.
        info =
            Lasda(icmpq1, smlsiz, nsize, sqre, D.ptr(st), E.ptr(st),
                  WORK.ptr(u + st1), n, WORK.ptr(vt + st1), IWORK.ptr(k + st1),
                  WORK.ptr(difl + st1), WORK.ptr(difr + st1), WORK.ptr(z + st1),
                  WORK.ptr(poles + st1), IWORK.ptr(givptr + st1),
                  IWORK.ptr(givcol + st1), n, IWORK.ptr(perm + st1),
                  WORK.ptr(givnum + st1), WORK.ptr(c + st1), WORK.ptr(s + st1),
                  WORK.ptr(nwork), IWORK.ptr(iwk));
        if (info != 0) {
          return info;
        }
        bxst = bx + st1;
        info = Lalsa(
            icmpq2, smlsiz, nsize, nrhs, B.ptr(st, 1), ldb, WORK.ptr(bxst), n,
            WORK.ptr(u + st1), n, WORK.ptr(vt + st1), IWORK.ptr(k + st1),
            WORK.ptr(difl + st1), WORK.ptr(difr + st1), WORK.ptr(z + st1),
            WORK.ptr(poles + st1), IWORK.ptr(givptr + st1),
            IWORK.ptr(givcol + st1), n, IWORK.ptr(perm + st1),
            WORK.ptr(givnum + st1), WORK.ptr(c + st1), WORK.ptr(s + st1),
            WORK.ptr(nwork), IWORK.ptr(iwk));
        if (info != 0) {
          return info;
        }
      }
      st = i + 1;
    }
  }
  // Apply the singular values and treat the tiny ones as zero.
  tol = rcnd * std::abs(D(Iamax(n, d, 1)));
  for (i = 1; i <= n; ++i) {
    // Some of the elements in D can be negative because 1-by-1
    // subproblems were not solved explicitly.
    if (std::abs(D(i)) <= tol) {
      Laset('A', 1, nrhs, zero, zero, WORK.ptr(bx + i - 1), n);
    } else {
      *rank = *rank + 1;
      info = Lascl('G', 0, 0, D(i), one, 1, nrhs, WORK.ptr(bx + i - 1), n);
    }
    D(i) = std::abs(D(i));
  }
  // Now apply back the right singular vectors.
  icmpq2 = 1;
  for (i = 1; i <= nsub; ++i) {
    st = IWORK(i);
    st1 = st - 1;
    nsize = IWORK(sizei + i - 1);
    bxst = bx + st1;
    if (nsize == 1) {
      Copy(nrhs, WORK.ptr(bxst), n, B.ptr(st, 1), ldb);
    } else if (nsize <= smlsiz) {
      Gemm('T', 'N', nsize, nrhs, nsize, one, WORK.ptr(vt + st1), n,
           WORK.ptr(bxst), n, zero, B.ptr(st, 1), ldb);
    } else {
      info = Lalsa(icmpq2, smlsiz, nsize, nrhs, WORK.ptr(bxst), n, B.ptr(st, 1),
                   ldb, WORK.ptr(u + st1), n, WORK.ptr(vt + st1),
                   IWORK.ptr(k + st1), WORK.ptr(difl + st1),
                   WORK.ptr(difr + st1), WORK.ptr(z + st1),
                   WORK.ptr(poles + st1), IWORK.ptr(givptr + st1),
                   IWORK.ptr(givcol + st1), n, IWORK.ptr(perm + st1),
                   WORK.ptr(givnum + st1), WORK.ptr(c + st1), WORK.ptr(s + st1),
                   WORK.ptr(nwork), IWORK.ptr(iwk));
      if (info != 0) {
        return info;
      }
    }
  }
  // Unscale and sort the singular values.
  info = Lascl('G', 0, 0, one, orgnrm, n, 1, d, n);
  info = Lasrt('D', n, d);
  info = Lascl('G', 0, 0, orgnrm, one, n, nrhs, b, ldb);
  return info;
}

// ZLALSD (complex T).
template <typename T>
blasint LalsdComplex(char uplo, blasint smlsiz, blasint n, blasint nrhs,
                     RealT<T>* d, RealT<T>* e, T* b, blasint ldb,
                     RealT<T> rcond, blasint* rank, T* work, RealT<T>* rwork,
                     blasint* iwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const T czero = T(0);
  const Matrix<T> B(b, ldb);
  const Vector<R> D(d);
  const Vector<R> E(e);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> RWORK(rwork);
  const Vector<T> WORK(work);
  blasint bx, bxst, c, difl, difr, givcol, givnum, givptr, i, icmpq1, icmpq2,
      irwb, irwib, irwrb, irwu, irwvt, irwwrk, iwk, j, jcol, jimag, jreal, jrow,
      k, nlvl, nm1, nrwork, nsize, nsub, perm, poles, s, sizei, smlszp, sqre,
      st, st1, u, vt, z;
  R cs, eps, orgnrm, r, rcnd, sn, tol;
  blasint info = 0;
  if (n < 0) {
    info = -3;
  } else if (nrhs < 1) {
    info = -4;
  } else if ((ldb < 1) || (ldb < n)) {
    info = -8;
  }
  if (info != 0) {
    Xerbla<T>("LALSD", -info);
    return info;
  }
  eps = Lamch<R>('E');
  // Set up the tolerance.
  if ((rcond <= zero) || (rcond >= one)) {
    rcnd = eps;
  } else {
    rcnd = rcond;
  }
  *rank = 0;
  // Quick return if possible.
  if (n == 0) {
    return info;
  } else if (n == 1) {
    if (D(1) == zero) {
      Laset('A', 1, nrhs, czero, czero, b, ldb);
    } else {
      *rank = 1;
      info = Lascl('G', 0, 0, D(1), one, 1, nrhs, b, ldb);
      D(1) = std::abs(D(1));
    }
    return info;
  }
  // Rotate the matrix if it is lower bidiagonal.
  if (uplo == 'L') {
    for (i = 1; i <= n - 1; ++i) {
      Lartg<R>(D(i), E(i), &cs, &sn, &r);
      D(i) = r;
      E(i) = sn * D(i + 1);
      D(i + 1) = cs * D(i + 1);
      if (nrhs == 1) {
        Rot(1, B.ptr(i, 1), 1, B.ptr(i + 1, 1), 1, cs, sn);
      } else {
        RWORK(i * 2 - 1) = cs;
        RWORK(i * 2) = sn;
      }
    }
    if (nrhs > 1) {
      for (i = 1; i <= nrhs; ++i) {
        for (j = 1; j <= n - 1; ++j) {
          cs = RWORK(j * 2 - 1);
          sn = RWORK(j * 2);
          Rot(1, B.ptr(j, i), 1, B.ptr(j + 1, i), 1, cs, sn);
        }
      }
    }
  }
  // Scale.
  nm1 = n - 1;
  orgnrm = Lanst<R>('M', n, d, e);
  if (orgnrm == zero) {
    Laset('A', n, nrhs, czero, czero, b, ldb);
    return info;
  }
  info = Lascl<R>('G', 0, 0, orgnrm, one, n, 1, d, n);
  info = Lascl<R>('G', 0, 0, orgnrm, one, nm1, 1, e, nm1);
  // If N is smaller than the minimum divide size SMLSIZ, then solve
  // the problem with another solver.
  if (n <= smlsiz) {
    irwu = 1;
    irwvt = irwu + n * n;
    irwwrk = irwvt + n * n;
    irwrb = irwwrk;
    irwib = irwrb + n * nrhs;
    irwb = irwib + n * nrhs;
    Laset<R>('A', n, n, zero, one, RWORK.ptr(irwu), n);
    Laset<R>('A', n, n, zero, one, RWORK.ptr(irwvt), n);
    info =
        Lasdq<R>('U', 0, n, n, n, 0, d, e, RWORK.ptr(irwvt), n, RWORK.ptr(irwu),
                 n, RWORK.ptr(irwwrk), 1, RWORK.ptr(irwwrk));
    if (info != 0) {
      return info;
    }
    // In the real version, B is passed to DLASDQ and multiplied
    // internally by Q**H. Here B is complex and that product is
    // computed below in two steps (real and imaginary parts).
    j = irwb - 1;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = 1; jrow <= n; ++jrow) {
        j = j + 1;
        RWORK(j) = std::real(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', n, nrhs, n, one, RWORK.ptr(irwu), n, RWORK.ptr(irwb), n,
            zero, RWORK.ptr(irwrb), n);
    j = irwb - 1;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = 1; jrow <= n; ++jrow) {
        j = j + 1;
        RWORK(j) = std::imag(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', n, nrhs, n, one, RWORK.ptr(irwu), n, RWORK.ptr(irwb), n,
            zero, RWORK.ptr(irwib), n);
    jreal = irwrb - 1;
    jimag = irwib - 1;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = 1; jrow <= n; ++jrow) {
        jreal = jreal + 1;
        jimag = jimag + 1;
        B(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
      }
    }
    tol = rcnd * std::abs(D(Iamax(n, d, 1)));
    for (i = 1; i <= n; ++i) {
      if (D(i) <= tol) {
        Laset('A', 1, nrhs, czero, czero, B.ptr(i, 1), ldb);
      } else {
        info = Lascl('G', 0, 0, D(i), one, 1, nrhs, B.ptr(i, 1), ldb);
        *rank = *rank + 1;
      }
    }
    // Since B is complex, the following call to DGEMM is performed
    // in two steps (real and imaginary parts). That is for V * B
    // (in the real version of the code V**H is stored in WORK).
    // CALL DGEMM( 'T', 'N', N, NRHS, N, ONE, WORK, N, B, LDB, ZERO,
    // $               WORK( NWORK ), N )
    j = irwb - 1;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = 1; jrow <= n; ++jrow) {
        j = j + 1;
        RWORK(j) = std::real(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', n, nrhs, n, one, RWORK.ptr(irwvt), n, RWORK.ptr(irwb), n,
            zero, RWORK.ptr(irwrb), n);
    j = irwb - 1;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = 1; jrow <= n; ++jrow) {
        j = j + 1;
        RWORK(j) = std::imag(B(jrow, jcol));
      }
    }
    Gemm<R>('T', 'N', n, nrhs, n, one, RWORK.ptr(irwvt), n, RWORK.ptr(irwb), n,
            zero, RWORK.ptr(irwib), n);
    jreal = irwrb - 1;
    jimag = irwib - 1;
    for (jcol = 1; jcol <= nrhs; ++jcol) {
      for (jrow = 1; jrow <= n; ++jrow) {
        jreal = jreal + 1;
        jimag = jimag + 1;
        B(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
      }
    }
    // Unscale.
    info = Lascl<R>('G', 0, 0, one, orgnrm, n, 1, d, n);
    info = Lasrt<R>('D', n, d);
    info = Lascl('G', 0, 0, orgnrm, one, n, nrhs, b, ldb);
    return info;
  }
  // Book-keeping and setting up some constants.
  nlvl =
      static_cast<blasint>(std::log(R(n) / R(smlsiz + 1)) / std::log(two)) + 1;
  smlszp = smlsiz + 1;
  u = 1;
  vt = 1 + smlsiz * n;
  difl = vt + smlszp * n;
  difr = difl + nlvl * n;
  z = difr + nlvl * n * 2;
  c = z + nlvl * n;
  s = c + n;
  poles = s + n;
  givnum = poles + 2 * nlvl * n;
  nrwork = givnum + 2 * nlvl * n;
  bx = 1;
  irwrb = nrwork;
  irwib = irwrb + smlsiz * nrhs;
  irwb = irwib + smlsiz * nrhs;
  sizei = 1 + n;
  k = sizei + n;
  givptr = k + n;
  perm = givptr + n;
  givcol = perm + nlvl * n;
  iwk = givcol + nlvl * n * 2;
  st = 1;
  sqre = 0;
  icmpq1 = 1;
  icmpq2 = 0;
  nsub = 0;
  for (i = 1; i <= n; ++i) {
    if (std::abs(D(i)) < eps) {
      D(i) = std::copysign(eps, D(i));
    }
  }
  for (i = 1; i <= nm1; ++i) {
    if ((std::abs(E(i)) < eps) || (i == nm1)) {
      nsub = nsub + 1;
      IWORK(nsub) = st;
      // Subproblem found. First determine its size and then
      // apply divide and conquer on it.
      if (i < nm1) {
        // A subproblem with E(I) small for I < NM1.
        nsize = i - st + 1;
        IWORK(sizei + nsub - 1) = nsize;
      } else if (std::abs(E(i)) >= eps) {
        // A subproblem with E(NM1) not too small but I = NM1.
        nsize = n - st + 1;
        IWORK(sizei + nsub - 1) = nsize;
      } else {
        // A subproblem with E(NM1) small. This implies an
        // 1-by-1 subproblem at D(N), which is not solved
        // explicitly.
        nsize = i - st + 1;
        IWORK(sizei + nsub - 1) = nsize;
        nsub = nsub + 1;
        IWORK(nsub) = n;
        IWORK(sizei + nsub - 1) = 1;
        Copy(nrhs, B.ptr(n, 1), ldb, WORK.ptr(bx + nm1), n);
      }
      st1 = st - 1;
      if (nsize == 1) {
        // This is a 1-by-1 subproblem and is not solved
        // explicitly.
        Copy(nrhs, B.ptr(st, 1), ldb, WORK.ptr(bx + st1), n);
      } else if (nsize <= smlsiz) {
        // This is a small subproblem and is solved by DLASDQ.
        Laset<R>('A', nsize, nsize, zero, one, RWORK.ptr(vt + st1), n);
        Laset<R>('A', nsize, nsize, zero, one, RWORK.ptr(u + st1), n);
        info = Lasdq<R>('U', 0, nsize, nsize, nsize, 0, D.ptr(st), E.ptr(st),
                        RWORK.ptr(vt + st1), n, RWORK.ptr(u + st1), n,
                        RWORK.ptr(nrwork), 1, RWORK.ptr(nrwork));
        if (info != 0) {
          return info;
        }
        // In the real version, B is passed to DLASDQ and multiplied
        // internally by Q**H. Here B is complex and that product is
        // computed below in two steps (real and imaginary parts).
        j = irwb - 1;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = st; jrow <= st + nsize - 1; ++jrow) {
            j = j + 1;
            RWORK(j) = std::real(B(jrow, jcol));
          }
        }
        Gemm<R>('T', 'N', nsize, nrhs, nsize, one, RWORK.ptr(u + st1), n,
                RWORK.ptr(irwb), nsize, zero, RWORK.ptr(irwrb), nsize);
        j = irwb - 1;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = st; jrow <= st + nsize - 1; ++jrow) {
            j = j + 1;
            RWORK(j) = std::imag(B(jrow, jcol));
          }
        }
        Gemm<R>('T', 'N', nsize, nrhs, nsize, one, RWORK.ptr(u + st1), n,
                RWORK.ptr(irwb), nsize, zero, RWORK.ptr(irwib), nsize);
        jreal = irwrb - 1;
        jimag = irwib - 1;
        for (jcol = 1; jcol <= nrhs; ++jcol) {
          for (jrow = st; jrow <= st + nsize - 1; ++jrow) {
            jreal = jreal + 1;
            jimag = jimag + 1;
            B(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
          }
        }
        Lacpy('A', nsize, nrhs, B.ptr(st, 1), ldb, WORK.ptr(bx + st1), n);
      } else {
        // A large problem. Solve it using divide and conquer.
        info = Lasda<R>(icmpq1, smlsiz, nsize, sqre, D.ptr(st), E.ptr(st),
                        RWORK.ptr(u + st1), n, RWORK.ptr(vt + st1),
                        IWORK.ptr(k + st1), RWORK.ptr(difl + st1),
                        RWORK.ptr(difr + st1), RWORK.ptr(z + st1),
                        RWORK.ptr(poles + st1), IWORK.ptr(givptr + st1),
                        IWORK.ptr(givcol + st1), n, IWORK.ptr(perm + st1),
                        RWORK.ptr(givnum + st1), RWORK.ptr(c + st1),
                        RWORK.ptr(s + st1), RWORK.ptr(nrwork), IWORK.ptr(iwk));
        if (info != 0) {
          return info;
        }
        bxst = bx + st1;
        info = Lalsa(
            icmpq2, smlsiz, nsize, nrhs, B.ptr(st, 1), ldb, WORK.ptr(bxst), n,
            RWORK.ptr(u + st1), n, RWORK.ptr(vt + st1), IWORK.ptr(k + st1),
            RWORK.ptr(difl + st1), RWORK.ptr(difr + st1), RWORK.ptr(z + st1),
            RWORK.ptr(poles + st1), IWORK.ptr(givptr + st1),
            IWORK.ptr(givcol + st1), n, IWORK.ptr(perm + st1),
            RWORK.ptr(givnum + st1), RWORK.ptr(c + st1), RWORK.ptr(s + st1),
            RWORK.ptr(nrwork), IWORK.ptr(iwk));
        if (info != 0) {
          return info;
        }
      }
      st = i + 1;
    }
  }
  // Apply the singular values and treat the tiny ones as zero.
  tol = rcnd * std::abs(D(Iamax(n, d, 1)));
  for (i = 1; i <= n; ++i) {
    // Some of the elements in D can be negative because 1-by-1
    // subproblems were not solved explicitly.
    if (std::abs(D(i)) <= tol) {
      Laset('A', 1, nrhs, czero, czero, WORK.ptr(bx + i - 1), n);
    } else {
      *rank = *rank + 1;
      info = Lascl('G', 0, 0, D(i), one, 1, nrhs, WORK.ptr(bx + i - 1), n);
    }
    D(i) = std::abs(D(i));
  }
  // Now apply back the right singular vectors.
  icmpq2 = 1;
  for (i = 1; i <= nsub; ++i) {
    st = IWORK(i);
    st1 = st - 1;
    nsize = IWORK(sizei + i - 1);
    bxst = bx + st1;
    if (nsize == 1) {
      Copy(nrhs, WORK.ptr(bxst), n, B.ptr(st, 1), ldb);
    } else if (nsize <= smlsiz) {
      // Since B and BX are complex, the following call to DGEMM
      // is performed in two steps (real and imaginary parts).
      // CALL DGEMM( 'T', 'N', NSIZE, NRHS, NSIZE, ONE,
      // $                  RWORK( VT+ST1 ), N, RWORK( BXST ), N, ZERO,
      // $                  B( ST, 1 ), LDB )
      j = bxst - n - 1;
      jreal = irwb - 1;
      for (jcol = 1; jcol <= nrhs; ++jcol) {
        j = j + n;
        for (jrow = 1; jrow <= nsize; ++jrow) {
          jreal = jreal + 1;
          RWORK(jreal) = std::real(WORK(j + jrow));
        }
      }
      Gemm<R>('T', 'N', nsize, nrhs, nsize, one, RWORK.ptr(vt + st1), n,
              RWORK.ptr(irwb), nsize, zero, RWORK.ptr(irwrb), nsize);
      j = bxst - n - 1;
      jimag = irwb - 1;
      for (jcol = 1; jcol <= nrhs; ++jcol) {
        j = j + n;
        for (jrow = 1; jrow <= nsize; ++jrow) {
          jimag = jimag + 1;
          RWORK(jimag) = std::imag(WORK(j + jrow));
        }
      }
      Gemm<R>('T', 'N', nsize, nrhs, nsize, one, RWORK.ptr(vt + st1), n,
              RWORK.ptr(irwb), nsize, zero, RWORK.ptr(irwib), nsize);
      jreal = irwrb - 1;
      jimag = irwib - 1;
      for (jcol = 1; jcol <= nrhs; ++jcol) {
        for (jrow = st; jrow <= st + nsize - 1; ++jrow) {
          jreal = jreal + 1;
          jimag = jimag + 1;
          B(jrow, jcol) = T(RWORK(jreal), RWORK(jimag));
        }
      }
    } else {
      info = Lalsa(icmpq2, smlsiz, nsize, nrhs, WORK.ptr(bxst), n, B.ptr(st, 1),
                   ldb, RWORK.ptr(u + st1), n, RWORK.ptr(vt + st1),
                   IWORK.ptr(k + st1), RWORK.ptr(difl + st1),
                   RWORK.ptr(difr + st1), RWORK.ptr(z + st1),
                   RWORK.ptr(poles + st1), IWORK.ptr(givptr + st1),
                   IWORK.ptr(givcol + st1), n, IWORK.ptr(perm + st1),
                   RWORK.ptr(givnum + st1), RWORK.ptr(c + st1),
                   RWORK.ptr(s + st1), RWORK.ptr(nrwork), IWORK.ptr(iwk));
      if (info != 0) {
        return info;
      }
    }
  }
  // Unscale and sort the singular values.
  info = Lascl<R>('G', 0, 0, one, orgnrm, n, 1, d, n);
  info = Lasrt<R>('D', n, d);
  info = Lascl('G', 0, 0, orgnrm, one, n, nrhs, b, ldb);
  return info;
}

// DGELSD (real T).
template <typename T>
blasint GelsdReal(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
                  blasint ldb, T* s, T rcond, blasint* rank, T* work,
                  blasint lwork, blasint* iwork) {
  using R = T;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const Matrix<R> A(a, lda);
  const Matrix<R> B(b, ldb);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> S(s);
  const Vector<R> WORK(work);
  blasint iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, maxmn,
      maxwrk, minmn, minwrk, mm, mnthr, nlvl, nwork, smlsiz, wlalsd;
  bool lquery;
  R anrm, bignum, bnrm, eps, sfmin, smlnum;
  blasint info = 0;
  minmn = Min<blasint>(m, n);
  maxmn = Max<blasint>(m, n);
  // MNTHR and SMLSIZ are set where they are read (by SGELSD only if
  // min(M,N) > 0).
  mnthr = 0;
  smlsiz = 0;
  if constexpr (std::is_same_v<T, double>) {
    mnthr = Ilaenv<T>(6, "GELSD", " ", m, n, nrhs, -1);
  }
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, maxmn)) {
    info = -7;
  }
  if constexpr (std::is_same_v<T, double>) {
    smlsiz = Ilaenv<T>(9, "GELSD", " ", 0, 0, 0, 0);

    // Compute workspace.
    // (Note: Comments in the code beginning "Workspace:" describe the
    // minimal amount of workspace needed at that point in the code,
    // as well as the preferred amount for good performance.
    // NB refers to the optimal block size for the immediately
    // following subroutine, as returned by ILAENV.)
    minwrk = 1;
    liwork = 1;
    minmn = Max<blasint>(1, minmn);
    nlvl = Max(static_cast<blasint>(std::log(R(minmn) / R(smlsiz + 1)) /
                                    std::log(two)) +
                   1,
               0);

    if (info == 0) {
      maxwrk = 1;
      liwork = 3 * minmn * nlvl + 11 * minmn;
      mm = m;
      if (m >= n && m >= mnthr) {
        // Path 1a - overdetermined, with many more rows than columns.
        mm = n;
        maxwrk = Max(maxwrk, n + n * Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1));
        maxwrk =
            Max(maxwrk, n + nrhs * Ilaenv<T>(1, "ORMQR", "LT", m, nrhs, n, -1));
      }
      if (m >= n) {
        // Path 1 - overdetermined or exactly determined.
        maxwrk = Max(maxwrk, 3 * n + (mm + n) * Ilaenv<T>(1, "GEBRD", " ", mm,
                                                          n, -1, -1));
        maxwrk = Max(maxwrk, 3 * n + nrhs * Ilaenv<T>(1, "ORMBR", "QLT", mm,
                                                      nrhs, n, -1));
        maxwrk = Max(maxwrk, 3 * n + (n - 1) * Ilaenv<T>(1, "ORMBR", "PLN", n,
                                                         nrhs, n, -1));
        wlalsd = 9 * n + 2 * n * smlsiz + 8 * n * nlvl + n * nrhs +
                 ((smlsiz + 1) * (smlsiz + 1));
        maxwrk = Max(maxwrk, 3 * n + wlalsd);
        minwrk = Max(3 * n + mm, 3 * n + nrhs, 3 * n + wlalsd);
      }
      if (n > m) {
        wlalsd = 9 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs +
                 ((smlsiz + 1) * (smlsiz + 1));
        if (n >= mnthr) {
          // Path 2a - underdetermined, with many more columns
          // than rows.
          maxwrk = m + m * Ilaenv<T>(1, "GELQF", " ", m, n, -1, -1);
          maxwrk =
              Max(maxwrk, m * m + 4 * m +
                              2 * m * Ilaenv<T>(1, "GEBRD", " ", m, m, -1, -1));
          maxwrk = Max(maxwrk,
                       m * m + 4 * m +
                           nrhs * Ilaenv<T>(1, "ORMBR", "QLT", m, nrhs, m, -1));
          maxwrk = Max(maxwrk, m * m + 4 * m +
                                   (m - 1) * Ilaenv<T>(1, "ORMBR", "PLN", m,
                                                       nrhs, m, -1));
          if (nrhs > 1) {
            maxwrk = Max(maxwrk, m * m + m + m * nrhs);
          } else {
            maxwrk = Max(maxwrk, m * m + 2 * m);
          }
          maxwrk = Max(maxwrk,
                       m + nrhs * Ilaenv<T>(1, "ORMLQ", "LT", n, nrhs, m, -1));
          maxwrk = Max(maxwrk, m * m + 4 * m + wlalsd);
          // XXX: Ensure the Path 2a case below is triggered.  The workspace
          // calculation should use queries for all routines eventually.
          maxwrk = Max(maxwrk, 4 * m + m * m +
                                   Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m));
        } else {
          // Path 2 - remaining underdetermined cases.
          maxwrk = 3 * m + (n + m) * Ilaenv<T>(1, "GEBRD", " ", m, n, -1, -1);
          maxwrk = Max(maxwrk, 3 * m + nrhs * Ilaenv<T>(1, "ORMBR", "QLT", m,
                                                        nrhs, n, -1));
          maxwrk = Max(
              maxwrk, 3 * m + m * Ilaenv<T>(1, "ORMBR", "PLN", n, nrhs, m, -1));
          maxwrk = Max(maxwrk, 3 * m + wlalsd);
        }
        minwrk = Max(3 * m + nrhs, 3 * m + m, 3 * m + wlalsd);
      }
      minwrk = Min(minwrk, maxwrk);
      WORK(1) = WorkSize<T>(maxwrk);
      IWORK(1) = liwork;

      if (lwork < minwrk && !lquery) info = -12;
    }
  } else {
    // SGELSD: as above, but only for min(M,N) > 0.
    if (info == 0) {
      minwrk = 1;
      maxwrk = 1;
      liwork = 1;
      if (minmn > 0) {
        smlsiz = Ilaenv<T>(9, "GELSD", " ", 0, 0, 0, 0);
        mnthr = Ilaenv<T>(6, "GELSD", " ", m, n, nrhs, -1);
        nlvl = Max(static_cast<blasint>(std::log(R(minmn) / R(smlsiz + 1)) /
                                        std::log(two)) +
                       1,
                   0);
        liwork = 3 * minmn * nlvl + 11 * minmn;
        mm = m;
        if (m >= n && m >= mnthr) {
          // Path 1a - overdetermined, with many more rows than
          //           columns.
          mm = n;
          maxwrk =
              Max(maxwrk, n + n * Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1));
          maxwrk = Max(maxwrk,
                       n + nrhs * Ilaenv<T>(1, "ORMQR", "LT", m, nrhs, n, -1));
        }
        if (m >= n) {
          // Path 1 - overdetermined or exactly determined.
          maxwrk = Max(maxwrk, 3 * n + (mm + n) * Ilaenv<T>(1, "GEBRD", " ", mm,
                                                            n, -1, -1));
          maxwrk = Max(maxwrk, 3 * n + nrhs * Ilaenv<T>(1, "ORMBR", "QLT", mm,
                                                        nrhs, n, -1));
          maxwrk = Max(maxwrk, 3 * n + (n - 1) * Ilaenv<T>(1, "ORMBR", "PLN", n,
                                                           nrhs, n, -1));
          wlalsd = 9 * n + 2 * n * smlsiz + 8 * n * nlvl + n * nrhs +
                   ((smlsiz + 1) * (smlsiz + 1));
          maxwrk = Max(maxwrk, 3 * n + wlalsd);
          minwrk = Max(3 * n + mm, 3 * n + nrhs, 3 * n + wlalsd);
        }
        if (n > m) {
          wlalsd = 9 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs +
                   ((smlsiz + 1) * (smlsiz + 1));
          if (n >= mnthr) {
            // Path 2a - underdetermined, with many more columns
            //           than rows.
            maxwrk = m + m * Ilaenv<T>(1, "GELQF", " ", m, n, -1, -1);
            maxwrk = Max(maxwrk,
                         m * m + 4 * m +
                             2 * m * Ilaenv<T>(1, "GEBRD", " ", m, m, -1, -1));
            maxwrk = Max(maxwrk, m * m + 4 * m +
                                     nrhs * Ilaenv<T>(1, "ORMBR", "QLT", m,
                                                      nrhs, m, -1));
            maxwrk = Max(maxwrk, m * m + 4 * m +
                                     (m - 1) * Ilaenv<T>(1, "ORMBR", "PLN", m,
                                                         nrhs, m, -1));
            if (nrhs > 1) {
              maxwrk = Max(maxwrk, m * m + m + m * nrhs);
            } else {
              maxwrk = Max(maxwrk, m * m + 2 * m);
            }
            maxwrk = Max(
                maxwrk, m + nrhs * Ilaenv<T>(1, "ORMLQ", "LT", n, nrhs, m, -1));
            maxwrk = Max(maxwrk, m * m + 4 * m + wlalsd);
            // XXX: Ensure the Path 2a case below is triggered.  The workspace
            // calculation should use queries for all routines eventually.
            maxwrk =
                Max(maxwrk, 4 * m + m * m +
                                Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m));
          } else {
            // Path 2 - remaining underdetermined cases.
            maxwrk = 3 * m + (n + m) * Ilaenv<T>(1, "GEBRD", " ", m, n, -1, -1);
            maxwrk = Max(maxwrk, 3 * m + nrhs * Ilaenv<T>(1, "ORMBR", "QLT", m,
                                                          nrhs, n, -1));
            maxwrk = Max(maxwrk, 3 * m + m * Ilaenv<T>(1, "ORMBR", "PLN", n,
                                                       nrhs, m, -1));
            maxwrk = Max(maxwrk, 3 * m + wlalsd);
          }
          minwrk = Max(3 * m + nrhs, 3 * m + m, 3 * m + wlalsd);
        }
      }
      minwrk = Min(minwrk, maxwrk);
      WORK(1) = WorkSize<T>(maxwrk);
      IWORK(1) = liwork;

      if (lwork < minwrk && !lquery) info = -12;
    }
  }

  if (info != 0) {
    Xerbla<T>("GELSD", -info);
    return info;
  } else if (lquery) {
    goto L10;
  }
  // Quick return if possible.
  if (m == 0 || n == 0) {
    *rank = 0;
    return info;
  }
  // Get machine parameters.
  eps = Lamch<R>('P');
  sfmin = Lamch<R>('S');
  smlnum = sfmin / eps;
  bignum = one / smlnum;
  // Scale A if max entry outside range [SMLNUM,BIGNUM].
  anrm = Lange('M', m, n, a, lda, work);
  iascl = 0;
  if (anrm > zero && anrm < smlnum) {
    // Scale matrix norm up to SMLNUM.
    info = Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
    iascl = 1;
  } else if (anrm > bignum) {
    // Scale matrix norm down to BIGNUM.
    info = Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
    iascl = 2;
  } else if (anrm == zero) {
    // Matrix all zero. Return zero solution.
    Laset('F', Max<blasint>(m, n), nrhs, zero, zero, b, ldb);
    Laset('F', minmn, 1, zero, zero, s, 1);
    *rank = 0;
    goto L10;
  }
  // Scale B if max entry outside range [SMLNUM,BIGNUM].
  bnrm = Lange('M', m, nrhs, b, ldb, work);
  ibscl = 0;
  if (bnrm > zero && bnrm < smlnum) {
    // Scale matrix norm up to SMLNUM.
    info = Lascl('G', 0, 0, bnrm, smlnum, m, nrhs, b, ldb);
    ibscl = 1;
  } else if (bnrm > bignum) {
    // Scale matrix norm down to BIGNUM.
    info = Lascl('G', 0, 0, bnrm, bignum, m, nrhs, b, ldb);
    ibscl = 2;
  }
  // If M < N make sure certain entries of B are zero.
  if (m < n) Laset('F', n - m, nrhs, zero, zero, B.ptr(m + 1, 1), ldb);
  // Overdetermined case.
  if (m >= n) {
    // Path 1 - overdetermined or exactly determined.
    mm = m;
    if (m >= mnthr) {
      // Path 1a - overdetermined, with many more rows than columns.
      mm = n;
      itau = 1;
      nwork = itau + n;
      // Compute A=Q*R.
      // (Workspace: need 2*N, prefer N+N*NB)
      info = Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
                   lwork - nwork + 1);
      // Multiply B by transpose(Q).
      // (Workspace: need N+NRHS, prefer N+NRHS*NB)
      info = Unmqr('L', 'T', m, nrhs, n, a, lda, WORK.ptr(itau), b, ldb,
                   WORK.ptr(nwork), lwork - nwork + 1);
      // Zero out below R.
      if (n > 1) {
        Laset('L', n - 1, n - 1, zero, zero, A.ptr(2, 1), lda);
      }
    }
    ie = 1;
    itauq = ie + n;
    itaup = itauq + n;
    nwork = itaup + n;
    // Bidiagonalize R in A.
    // (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB)
    info = Gebrd(mm, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of R.
    // (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB)
    info = Unmbr('Q', 'L', 'T', mm, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
    // Solve the bidiagonal least squares problem.
    info = Lalsd('U', smlsiz, n, nrhs, s, WORK.ptr(ie), b, ldb, rcond, rank,
                 WORK.ptr(nwork), static_cast<R*>(nullptr), iwork);
    if (info != 0) {
      goto L10;
    }
    // Multiply B by right bidiagonalizing vectors of R.
    info = Unmbr('P', 'L', 'N', n, nrhs, n, a, lda, WORK.ptr(itaup), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
  } else if (n >= mnthr &&
             lwork >= 4 * m + m * m +
                          Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m, wlalsd)) {
    // Path 2a - underdetermined, with many more columns than rows
    // and sufficient workspace for an efficient algorithm.
    ldwork = m;
    if (lwork >= Max<blasint>(4 * m + m * lda +
                                  Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m),
                              m * lda + m + m * nrhs, 4 * m + m * lda + wlalsd))
      ldwork = lda;
    itau = 1;
    nwork = m + 1;
    // Compute A=L*Q.
    // (Workspace: need 2*M, prefer M+M*NB)
    info =
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
    il = nwork;
    // Copy L to WORK(IL), zeroing out above its diagonal.
    Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwork);
    Laset('U', m - 1, m - 1, zero, zero, WORK.ptr(il + ldwork), ldwork);
    ie = il + ldwork * m;
    itauq = ie + m;
    itaup = itauq + m;
    nwork = itaup + m;
    // Bidiagonalize L in WORK(IL).
    // (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB)
    info = Gebrd(m, m, WORK.ptr(il), ldwork, s, WORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of L.
    // (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
    info = Unmbr('Q', 'L', 'T', m, nrhs, m, WORK.ptr(il), ldwork,
                 WORK.ptr(itauq), b, ldb, WORK.ptr(nwork), lwork - nwork + 1);
    // Solve the bidiagonal least squares problem.
    info = Lalsd('U', smlsiz, m, nrhs, s, WORK.ptr(ie), b, ldb, rcond, rank,
                 WORK.ptr(nwork), static_cast<R*>(nullptr), iwork);
    if (info != 0) {
      goto L10;
    }
    // Multiply B by right bidiagonalizing vectors of L.
    info = Unmbr('P', 'L', 'N', m, nrhs, m, WORK.ptr(il), ldwork,
                 WORK.ptr(itaup), b, ldb, WORK.ptr(nwork), lwork - nwork + 1);
    // Zero out below first M rows of B.
    Laset('F', n - m, nrhs, zero, zero, B.ptr(m + 1, 1), ldb);
    nwork = itau + m;
    // Multiply transpose(Q) by B.
    // (Workspace: need M+NRHS, prefer M+NRHS*NB)
    info = Unmlq('L', 'T', n, nrhs, m, a, lda, WORK.ptr(itau), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
  } else {
    // Path 2 - remaining underdetermined cases.
    ie = 1;
    itauq = ie + m;
    itaup = itauq + m;
    nwork = itaup + m;
    // Bidiagonalize A.
    // (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
    info = Gebrd(m, n, a, lda, s, WORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors.
    // (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB)
    info = Unmbr('Q', 'L', 'T', m, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
    // Solve the bidiagonal least squares problem.
    info = Lalsd('L', smlsiz, m, nrhs, s, WORK.ptr(ie), b, ldb, rcond, rank,
                 WORK.ptr(nwork), static_cast<R*>(nullptr), iwork);
    if (info != 0) {
      goto L10;
    }
    // Multiply B by right bidiagonalizing vectors of A.
    info = Unmbr('P', 'L', 'N', n, nrhs, m, a, lda, WORK.ptr(itaup), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
  }
  // Undo scaling.
  if (iascl == 1) {
    info = Lascl('G', 0, 0, anrm, smlnum, n, nrhs, b, ldb);
    info = Lascl('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
  } else if (iascl == 2) {
    info = Lascl('G', 0, 0, anrm, bignum, n, nrhs, b, ldb);
    info = Lascl('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
  }
  if (ibscl == 1) {
    info = Lascl('G', 0, 0, smlnum, bnrm, n, nrhs, b, ldb);
  } else if (ibscl == 2) {
    info = Lascl('G', 0, 0, bignum, bnrm, n, nrhs, b, ldb);
  }
L10:
  WORK(1) = WorkSize<T>(maxwrk);
  IWORK(1) = liwork;
  return info;
}

// ZGELSD (complex T).
template <typename T>
blasint GelsdComplex(blasint m, blasint n, blasint nrhs, T* a, blasint lda,
                     T* b, blasint ldb, RealT<T>* s, RealT<T> rcond,
                     blasint* rank, T* work, blasint lwork, RealT<T>* rwork,
                     blasint* iwork) {
  using R = RealT<T>;

  const R zero = R(0);
  const R one = R(1);
  const R two = R(2);
  const T czero = T(0);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<blasint> IWORK(iwork);
  const Vector<R> RWORK(rwork);
  const Vector<R> S(s);
  const Vector<T> WORK(work);
  blasint iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, lrwork,
      maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nrwork, nwork, smlsiz;
  bool lquery;
  R anrm, bignum, bnrm, eps, sfmin, smlnum;
  blasint info = 0;
  minmn = Min<blasint>(m, n);
  maxmn = Max<blasint>(m, n);
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, maxmn)) {
    info = -7;
  }
  // Compute workspace.
  // (Note: Comments in the code beginning "Workspace:" describe the
  // minimal amount of workspace needed at that point in the code,
  // as well as the preferred amount for good performance.
  // NB refers to the optimal block size for the immediately
  // following subroutine, as returned by ILAENV.)
  if (info == 0) {
    minwrk = 1;
    maxwrk = 1;
    liwork = 1;
    lrwork = 1;
    if (minmn > 0) {
      smlsiz = Ilaenv<T>(9, "GELSD", " ", 0, 0, 0, 0);
      mnthr = Ilaenv<T>(6, "GELSD", " ", m, n, nrhs, -1);
      nlvl = Max(static_cast<blasint>(std::log(R(minmn) / R(smlsiz + 1)) /
                                      std::log(two)) +
                     1,
                 0);
      liwork = 3 * minmn * nlvl + 11 * minmn;
      mm = m;
      if (m >= n && m >= mnthr) {
        // Path 1a - overdetermined, with many more rows than
        // columns.
        mm = n;
        maxwrk =
            Max<blasint>(maxwrk, n * Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1));
        maxwrk = Max<blasint>(
            maxwrk, nrhs * Ilaenv<T>(1, "UNMQR", "LC", m, nrhs, n, -1));
      }
      if (m >= n) {
        // Path 1 - overdetermined or exactly determined.
        lrwork = 10 * n + 2 * n * smlsiz + 8 * n * nlvl + 3 * smlsiz * nrhs +
                 Max<blasint>(((smlsiz + 1) * (smlsiz + 1)),
                              n * (1 + nrhs) + 2 * nrhs);
        maxwrk = Max<blasint>(
            maxwrk,
            2 * n + (mm + n) * Ilaenv<T>(1, "GEBRD", " ", mm, n, -1, -1));
        maxwrk = Max<blasint>(
            maxwrk,
            2 * n + nrhs * Ilaenv<T>(1, "UNMBR", "QLC", mm, nrhs, n, -1));
        maxwrk = Max<blasint>(
            maxwrk,
            2 * n + (n - 1) * Ilaenv<T>(1, "UNMBR", "PLN", n, nrhs, n, -1));
        maxwrk = Max<blasint>(maxwrk, 2 * n + n * nrhs);
        minwrk = Max<blasint>(2 * n + mm, 2 * n + n * nrhs);
      }
      if (n > m) {
        lrwork = 10 * m + 2 * m * smlsiz + 8 * m * nlvl + 3 * smlsiz * nrhs +
                 Max<blasint>(((smlsiz + 1) * (smlsiz + 1)),
                              n * (1 + nrhs) + 2 * nrhs);
        if (n >= mnthr) {
          // Path 2a - underdetermined, with many more columns
          // than rows.
          maxwrk = m + m * Ilaenv<T>(1, "GELQF", " ", m, n, -1, -1);
          maxwrk = Max<blasint>(
              maxwrk,
              m * m + 4 * m + 2 * m * Ilaenv<T>(1, "GEBRD", " ", m, m, -1, -1));
          maxwrk = Max<blasint>(
              maxwrk, m * m + 4 * m +
                          nrhs * Ilaenv<T>(1, "UNMBR", "QLC", m, nrhs, m, -1));
          maxwrk = Max<blasint>(
              maxwrk,
              m * m + 4 * m +
                  (m - 1) * Ilaenv<T>(1, "UNMLQ", "LC", n, nrhs, m, -1));
          if (nrhs > 1) {
            maxwrk = Max<blasint>(maxwrk, m * m + m + m * nrhs);
          } else {
            maxwrk = Max<blasint>(maxwrk, m * m + 2 * m);
          }
          maxwrk = Max<blasint>(maxwrk, m * m + 4 * m + m * nrhs);
          // XXX: Ensure the Path 2a case below is triggered.  The workspace
          // calculation should use queries for all routines eventually.
          maxwrk = Max<blasint>(
              maxwrk,
              4 * m + m * m + Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m));
        } else {
          // Path 2 - underdetermined.
          maxwrk = 2 * m + (n + m) * Ilaenv<T>(1, "GEBRD", " ", m, n, -1, -1);
          maxwrk = Max<blasint>(
              maxwrk,
              2 * m + nrhs * Ilaenv<T>(1, "UNMBR", "QLC", m, nrhs, m, -1));
          maxwrk = Max<blasint>(
              maxwrk, 2 * m + m * Ilaenv<T>(1, "UNMBR", "PLN", n, nrhs, m, -1));
          maxwrk = Max<blasint>(maxwrk, 2 * m + m * nrhs);
        }
        minwrk = Max<blasint>(2 * m + n, 2 * m + m * nrhs);
      }
    }
    minwrk = Min<blasint>(minwrk, maxwrk);
    WORK(1) = WorkSize<T>(maxwrk);
    IWORK(1) = liwork;
    RWORK(1) = static_cast<R>(lrwork);
    if (lwork < minwrk && !lquery) {
      info = -12;
    }
  }
  if (info != 0) {
    Xerbla<T>("GELSD", -info);
    return info;
  } else if (lquery) {
    return info;
  }
  // Quick return if possible.
  if (m == 0 || n == 0) {
    *rank = 0;
    return info;
  }
  // Get machine parameters.
  eps = Lamch<R>('P');
  sfmin = Lamch<R>('S');
  smlnum = sfmin / eps;
  bignum = one / smlnum;
  // Scale A if max entry outside range [SMLNUM,BIGNUM].
  anrm = Lange('M', m, n, a, lda, rwork);
  iascl = 0;
  if (anrm > zero && anrm < smlnum) {
    // Scale matrix norm up to SMLNUM
    info = Lascl('G', 0, 0, anrm, smlnum, m, n, a, lda);
    iascl = 1;
  } else if (anrm > bignum) {
    // Scale matrix norm down to BIGNUM.
    info = Lascl('G', 0, 0, anrm, bignum, m, n, a, lda);
    iascl = 2;
  } else if (anrm == zero) {
    // Matrix all zero. Return zero solution.
    Laset('F', Max<blasint>(m, n), nrhs, czero, czero, b, ldb);
    Laset<R>('F', minmn, 1, zero, zero, s, 1);
    *rank = 0;
    goto L10;
  }
  // Scale B if max entry outside range [SMLNUM,BIGNUM].
  bnrm = Lange('M', m, nrhs, b, ldb, rwork);
  ibscl = 0;
  if (bnrm > zero && bnrm < smlnum) {
    // Scale matrix norm up to SMLNUM.
    info = Lascl('G', 0, 0, bnrm, smlnum, m, nrhs, b, ldb);
    ibscl = 1;
  } else if (bnrm > bignum) {
    // Scale matrix norm down to BIGNUM.
    info = Lascl('G', 0, 0, bnrm, bignum, m, nrhs, b, ldb);
    ibscl = 2;
  }
  // If M < N make sure B(M+1:N,:) = 0
  if (m < n) Laset('F', n - m, nrhs, czero, czero, B.ptr(m + 1, 1), ldb);
  // Overdetermined case.
  if (m >= n) {
    // Path 1 - overdetermined or exactly determined.
    mm = m;
    if (m >= mnthr) {
      // Path 1a - overdetermined, with many more rows than columns
      mm = n;
      itau = 1;
      nwork = itau + n;
      // Compute A=Q*R.
      // (RWorkspace: need N)
      // (CWorkspace: need N, prefer N*NB)
      info = Geqrf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork),
                   lwork - nwork + 1);
      // Multiply B by transpose(Q).
      // (RWorkspace: need N)
      // (CWorkspace: need NRHS, prefer NRHS*NB)
      info = Unmqr('L', 'C', m, nrhs, n, a, lda, WORK.ptr(itau), b, ldb,
                   WORK.ptr(nwork), lwork - nwork + 1);
      // Zero out below R.
      if (n > 1) {
        Laset('L', n - 1, n - 1, czero, czero, A.ptr(2, 1), lda);
      }
    }
    itauq = 1;
    itaup = itauq + n;
    nwork = itaup + n;
    ie = 1;
    nrwork = ie + n;
    // Bidiagonalize R in A.
    // (RWorkspace: need N)
    // (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
    info = Gebrd(mm, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of R.
    // (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
    info = Unmbr('Q', 'L', 'C', mm, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
    // Solve the bidiagonal least squares problem.
    info = Lalsd('U', smlsiz, n, nrhs, s, RWORK.ptr(ie), b, ldb, rcond, rank,
                 WORK.ptr(nwork), RWORK.ptr(nrwork), iwork);
    if (info != 0) {
      goto L10;
    }
    // Multiply B by right bidiagonalizing vectors of R.
    info = Unmbr('P', 'L', 'N', n, nrhs, n, a, lda, WORK.ptr(itaup), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
  } else if (n >= mnthr &&
             lwork >=
                 4 * m + m * m + Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m)) {
    // Path 2a - underdetermined, with many more columns than rows
    // and sufficient workspace for an efficient algorithm.
    ldwork = m;
    if (lwork >= Max<blasint>(4 * m + m * lda +
                                  Max<blasint>(m, 2 * m - 4, nrhs, n - 3 * m),
                              m * lda + m + m * nrhs))
      ldwork = lda;
    itau = 1;
    nwork = m + 1;
    // Compute A=L*Q.
    // (CWorkspace: need 2*M, prefer M+M*NB)
    info =
        Gelqf(m, n, a, lda, WORK.ptr(itau), WORK.ptr(nwork), lwork - nwork + 1);
    il = nwork;
    // Copy L to WORK(IL), zeroing out above its diagonal.
    Lacpy('L', m, m, a, lda, WORK.ptr(il), ldwork);
    Laset('U', m - 1, m - 1, czero, czero, WORK.ptr(il + ldwork), ldwork);
    itauq = il + ldwork * m;
    itaup = itauq + m;
    nwork = itaup + m;
    ie = 1;
    nrwork = ie + m;
    // Bidiagonalize L in WORK(IL).
    // (RWorkspace: need M)
    // (CWorkspace: need M*M+4*M, prefer M*M+4*M+2*M*NB)
    info = Gebrd(m, m, WORK.ptr(il), ldwork, s, RWORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors of L.
    // (CWorkspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
    info = Unmbr('Q', 'L', 'C', m, nrhs, m, WORK.ptr(il), ldwork,
                 WORK.ptr(itauq), b, ldb, WORK.ptr(nwork), lwork - nwork + 1);
    // Solve the bidiagonal least squares problem.
    info = Lalsd('U', smlsiz, m, nrhs, s, RWORK.ptr(ie), b, ldb, rcond, rank,
                 WORK.ptr(nwork), RWORK.ptr(nrwork), iwork);
    if (info != 0) {
      goto L10;
    }
    // Multiply B by right bidiagonalizing vectors of L.
    info = Unmbr('P', 'L', 'N', m, nrhs, m, WORK.ptr(il), ldwork,
                 WORK.ptr(itaup), b, ldb, WORK.ptr(nwork), lwork - nwork + 1);
    // Zero out below first M rows of B.
    Laset('F', n - m, nrhs, czero, czero, B.ptr(m + 1, 1), ldb);
    nwork = itau + m;
    // Multiply transpose(Q) by B.
    // (CWorkspace: need NRHS, prefer NRHS*NB)
    info = Unmlq('L', 'C', n, nrhs, m, a, lda, WORK.ptr(itau), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
  } else {
    // Path 2 - remaining underdetermined cases.
    itauq = 1;
    itaup = itauq + m;
    nwork = itaup + m;
    ie = 1;
    nrwork = ie + m;
    // Bidiagonalize A.
    // (RWorkspace: need M)
    // (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
    info = Gebrd(m, n, a, lda, s, RWORK.ptr(ie), WORK.ptr(itauq),
                 WORK.ptr(itaup), WORK.ptr(nwork), lwork - nwork + 1);
    // Multiply B by transpose of left bidiagonalizing vectors.
    // (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
    info = Unmbr('Q', 'L', 'C', m, nrhs, n, a, lda, WORK.ptr(itauq), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
    // Solve the bidiagonal least squares problem.
    info = Lalsd('L', smlsiz, m, nrhs, s, RWORK.ptr(ie), b, ldb, rcond, rank,
                 WORK.ptr(nwork), RWORK.ptr(nrwork), iwork);
    if (info != 0) {
      goto L10;
    }
    // Multiply B by right bidiagonalizing vectors of A.
    info = Unmbr('P', 'L', 'N', n, nrhs, m, a, lda, WORK.ptr(itaup), b, ldb,
                 WORK.ptr(nwork), lwork - nwork + 1);
  }
  // Undo scaling.
  if (iascl == 1) {
    info = Lascl('G', 0, 0, anrm, smlnum, n, nrhs, b, ldb);
    info = Lascl<R>('G', 0, 0, smlnum, anrm, minmn, 1, s, minmn);
  } else if (iascl == 2) {
    info = Lascl('G', 0, 0, anrm, bignum, n, nrhs, b, ldb);
    info = Lascl<R>('G', 0, 0, bignum, anrm, minmn, 1, s, minmn);
  }
  if (ibscl == 1) {
    info = Lascl('G', 0, 0, smlnum, bnrm, n, nrhs, b, ldb);
  } else if (ibscl == 2) {
    info = Lascl('G', 0, 0, bignum, bnrm, n, nrhs, b, ldb);
  }
L10:
  WORK(1) = WorkSize<T>(maxwrk);
  IWORK(1) = liwork;
  RWORK(1) = static_cast<R>(lrwork);
  return info;
}

}  // namespace

// xLALS0: applies back the multiplying factors of the merge step of
// xLASD6 (the divide and conquer of xLASDA) to the NRHS right-hand sides in
// B (ICOMPQ = 0: left singular vector matrix, into BX) or in BX (ICOMPQ =
// 1: right singular vector matrix, into B). WORK (real for complex T) has
// K (real T) or K*(1+2*NRHS) elements (complex T).
template <typename T>
blasint Lals0(blasint icompq, blasint nl, blasint nr, blasint sqre,
              blasint nrhs, T* b, blasint ldb, T* bx, blasint ldbx,
              blasint* perm, blasint givptr, blasint* givcol, blasint ldgcol,
              RealT<T>* givnum, blasint ldgnum, RealT<T>* poles, RealT<T>* difl,
              RealT<T>* difr, RealT<T>* z, blasint k, RealT<T> c, RealT<T> s,
              RealT<T>* work) {
  if constexpr (IsComplex<T>::value) {
    return Lals0Complex(icompq, nl, nr, sqre, nrhs, b, ldb, bx, ldbx, perm,
                        givptr, givcol, ldgcol, givnum, ldgnum, poles, difl,
                        difr, z, k, c, s, work);
  } else {
    return Lals0Real(icompq, nl, nr, sqre, nrhs, b, ldb, bx, ldbx, perm, givptr,
                     givcol, ldgcol, givnum, ldgnum, poles, difl, difr, z, k, c,
                     s, work);
  }
}

// xLALSA: applies the singular vectors of the divide and conquer of xLASDA
// (computed in compact form) to the NRHS right-hand sides in B, from the
// left (ICOMPQ = 0, into BX) or the right (ICOMPQ = 1, into B... as LAPACK:
// the inverse transformation). WORK (real for complex T) and IWORK as
// documented in LAPACK.
template <typename T>
blasint Lalsa(blasint icompq, blasint smlsiz, blasint n, blasint nrhs, T* b,
              blasint ldb, T* bx, blasint ldbx, RealT<T>* u, blasint ldu,
              RealT<T>* vt, blasint* k, RealT<T>* difl, RealT<T>* difr,
              RealT<T>* z, RealT<T>* poles, blasint* givptr, blasint* givcol,
              blasint ldgcol, blasint* perm, RealT<T>* givnum, RealT<T>* c,
              RealT<T>* s, RealT<T>* work, blasint* iwork) {
  if constexpr (IsComplex<T>::value) {
    return LalsaComplex(icompq, smlsiz, n, nrhs, b, ldb, bx, ldbx, u, ldu, vt,
                        k, difl, difr, z, poles, givptr, givcol, ldgcol, perm,
                        givnum, c, s, work, iwork);
  } else {
    return LalsaReal(icompq, smlsiz, n, nrhs, b, ldb, bx, ldbx, u, ldu, vt, k,
                     difl, difr, z, poles, givptr, givcol, ldgcol, perm, givnum,
                     c, s, work, iwork);
  }
}

// xLALSD: solves the least squares problem min || B - A*X || for the N-by-N
// upper (UPLO = 'U') or lower ('L') bidiagonal matrix A with diagonal D and
// off-diagonal E, by its singular value decomposition with divide and
// conquer; singular values <= RCOND * the largest count as zero (RCOND <= 0
// or >= 1: machine epsilon), RANK is the number of the others. On exit, D
// holds the singular values in decreasing order, E is destroyed, B the
// solutions. WORK (real T) or RWORK (complex T; then WORK has N*NRHS
// elements) and IWORK as documented in LAPACK; RWORK is not referenced for
// real T. Returns INFO: 0, -i for an illegal argument i, or i > 0 if a
// singular value did not converge.
template <typename T>
blasint Lalsd(char uplo, blasint smlsiz, blasint n, blasint nrhs, RealT<T>* d,
              RealT<T>* e, T* b, blasint ldb, RealT<T> rcond, blasint* rank,
              T* work, RealT<T>* rwork, blasint* iwork) {
  if constexpr (IsComplex<T>::value) {
    return LalsdComplex(uplo, smlsiz, n, nrhs, d, e, b, ldb, rcond, rank, work,
                        rwork, iwork);
  } else {
    return LalsdReal(uplo, smlsiz, n, nrhs, d, e, b, ldb, rcond, rank, work,
                     iwork);
  }
}

// xGELSD: computes the minimum norm solution of the least squares problem
// min || B - A*X || for the M-by-N matrix A of any rank and the NRHS
// right-hand sides in B (LDB >= max(1, M, N)), by the singular value
// decomposition of A with divide and conquer: A is reduced to bidiagonal
// form (after a QR or LQ factorization if it has many more rows or columns
// than the other), the bidiagonal problem is solved with xLALSD, and the
// solution transformed back. Singular values S(i) <= RCOND*S(1) count as
// zero (RCOND < 0: machine precision); RANK is the effective rank. On exit,
// S holds the singular values in decreasing order, B the solutions, A is
// destroyed. LWORK, LRWORK (complex T) and LIWORK as documented in LAPACK;
// LWORK = -1 is a workspace query, which returns the optimal LWORK in
// WORK(1), the minimal LIWORK in IWORK(1) and the minimal LRWORK in
// RWORK(1). Returns INFO: 0, -i for an illegal argument i, or i > 0 if a
// singular value did not converge. The single precision real routine
// sizes the workspace only for min(M,N) > 0 (DGELSD also for empty
// matrices, with min(M,N) = 1).
template <typename T>
blasint Gelsd(blasint m, blasint n, blasint nrhs, T* a, blasint lda, T* b,
              blasint ldb, RealT<T>* s, RealT<T> rcond, blasint* rank, T* work,
              blasint lwork, RealT<T>* rwork, blasint* iwork) {
  if constexpr (IsComplex<T>::value) {
    return GelsdComplex(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork,
                        rwork, iwork);
  } else {
    return GelsdReal(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork,
                     iwork);
  }
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template blasint Lals0<T>(blasint, blasint, blasint, blasint, blasint, T*,   \
                            blasint, T*, blasint, blasint*, blasint, blasint*, \
                            blasint, RealT<T>*, blasint, RealT<T>*, RealT<T>*, \
                            RealT<T>*, RealT<T>*, blasint, RealT<T>, RealT<T>, \
                            RealT<T>*);                                        \
  template blasint Lalsa<T>(                                                   \
      blasint, blasint, blasint, blasint, T*, blasint, T*, blasint, RealT<T>*, \
      blasint, RealT<T>*, blasint*, RealT<T>*, RealT<T>*, RealT<T>*,           \
      RealT<T>*, blasint*, blasint*, blasint, blasint*, RealT<T>*, RealT<T>*,  \
      RealT<T>*, RealT<T>*, blasint*);                                         \
  template blasint Lalsd<T>(char, blasint, blasint, blasint, RealT<T>*,        \
                            RealT<T>*, T*, blasint, RealT<T>, blasint*, T*,    \
                            RealT<T>*, blasint*);                              \
  template blasint Gelsd<T>(blasint, blasint, blasint, T*, blasint, T*,        \
                            blasint, RealT<T>*, RealT<T>, blasint*, T*,        \
                            blasint, RealT<T>*, blasint*);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GELSD_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##lals0_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr, \
      const hwyblas_int* sqre, const hwyblas_int* nrhs, R* b,                  \
      const hwyblas_int* ldb, R* bx, const hwyblas_int* ldbx,                  \
      hwyblas_int* perm, const hwyblas_int* givptr, hwyblas_int* givcol,       \
      const hwyblas_int* ldgcol, R* givnum, const hwyblas_int* ldgnum,         \
      R* poles, R* difl, R* difr, R* z, const hwyblas_int* k, const R* c,      \
      const R* s, R* work, hwyblas_int* info) {                                \
    *info = hwyblas::lapack::Lals0(*icompq, *nl, *nr, *sqre, *nrhs, b, *ldb,   \
                                   bx, *ldbx, perm, *givptr, givcol, *ldgcol,  \
                                   givnum, *ldgnum, poles, difl, difr, z, *k,  \
                                   *c, *s, work);                              \
  }                                                                            \
  HWYBLAS_API void prefix##lalsa_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* smlsiz,                    \
      const hwyblas_int* n, const hwyblas_int* nrhs, R* b,                     \
      const hwyblas_int* ldb, R* bx, const hwyblas_int* ldbx, R* u,            \
      const hwyblas_int* ldu, R* vt, hwyblas_int* k, R* difl, R* difr, R* z,   \
      R* poles, hwyblas_int* givptr, hwyblas_int* givcol,                      \
      const hwyblas_int* ldgcol, hwyblas_int* perm, R* givnum, R* c, R* s,     \
      R* work, hwyblas_int* iwork, hwyblas_int* info) {                        \
    *info = hwyblas::lapack::Lalsa(*icompq, *smlsiz, *n, *nrhs, b, *ldb, bx,   \
                                   *ldbx, u, *ldu, vt, k, difl, difr, z,       \
                                   poles, givptr, givcol, *ldgcol, perm,       \
                                   givnum, c, s, work, iwork);                 \
  }                                                                            \
  HWYBLAS_API void prefix##lalsd_(                                             \
      const char* uplo, const hwyblas_int* smlsiz, const hwyblas_int* n,       \
      const hwyblas_int* nrhs, R* d, R* e, R* b, const hwyblas_int* ldb,       \
      const R* rcond, hwyblas_int* rank, R* work, hwyblas_int* iwork,          \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Lalsd(*uplo, *smlsiz, *n, *nrhs, d, e, b, *ldb,   \
                                   *rcond, rank, work,                         \
                                   static_cast<R*>(nullptr), iwork);           \
  }                                                                            \
  HWYBLAS_API void prefix##gelsd_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nrhs,     \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* s,        \
      const R* rcond, hwyblas_int* rank, R* work, const hwyblas_int* lwork,    \
      hwyblas_int* iwork, hwyblas_int* info) {                                 \
    *info = hwyblas::lapack::Gelsd(*m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond, \
                                   rank, work, *lwork,                         \
                                   static_cast<R*>(nullptr), iwork);           \
  }

HWYBLAS_LAPACK_GELSD_REAL(s, float)
HWYBLAS_LAPACK_GELSD_REAL(d, double)

#undef HWYBLAS_LAPACK_GELSD_REAL

#define HWYBLAS_LAPACK_GELSD_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##lals0_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* nl, const hwyblas_int* nr, \
      const hwyblas_int* sqre, const hwyblas_int* nrhs, R* b,                  \
      const hwyblas_int* ldb, R* bx, const hwyblas_int* ldbx,                  \
      hwyblas_int* perm, const hwyblas_int* givptr, hwyblas_int* givcol,       \
      const hwyblas_int* ldgcol, R* givnum, const hwyblas_int* ldgnum,         \
      R* poles, R* difl, R* difr, R* z, const hwyblas_int* k, const R* c,      \
      const R* s, R* rwork, hwyblas_int* info) {                               \
    *info = hwyblas::lapack::Lals0(*icompq, *nl, *nr, *sqre, *nrhs, As<T>(b),  \
                                   *ldb, As<T>(bx), *ldbx, perm, *givptr,      \
                                   givcol, *ldgcol, givnum, *ldgnum, poles,    \
                                   difl, difr, z, *k, *c, *s, rwork);          \
  }                                                                            \
  HWYBLAS_API void prefix##lalsa_(                                             \
      const hwyblas_int* icompq, const hwyblas_int* smlsiz,                    \
      const hwyblas_int* n, const hwyblas_int* nrhs, R* b,                     \
      const hwyblas_int* ldb, R* bx, const hwyblas_int* ldbx, R* u,            \
      const hwyblas_int* ldu, R* vt, hwyblas_int* k, R* difl, R* difr, R* z,   \
      R* poles, hwyblas_int* givptr, hwyblas_int* givcol,                      \
      const hwyblas_int* ldgcol, hwyblas_int* perm, R* givnum, R* c, R* s,     \
      R* rwork, hwyblas_int* iwork, hwyblas_int* info) {                       \
    *info = hwyblas::lapack::Lalsa(*icompq, *smlsiz, *n, *nrhs, As<T>(b),      \
                                   *ldb, As<T>(bx), *ldbx, u, *ldu, vt, k,     \
                                   difl, difr, z, poles, givptr, givcol,       \
                                   *ldgcol, perm, givnum, c, s, rwork, iwork); \
  }                                                                            \
  HWYBLAS_API void prefix##lalsd_(                                             \
      const char* uplo, const hwyblas_int* smlsiz, const hwyblas_int* n,       \
      const hwyblas_int* nrhs, R* d, R* e, R* b, const hwyblas_int* ldb,       \
      const R* rcond, hwyblas_int* rank, R* work, R* rwork,                    \
      hwyblas_int* iwork, hwyblas_int* info) {                                 \
    *info =                                                                    \
        hwyblas::lapack::Lalsd(*uplo, *smlsiz, *n, *nrhs, d, e, As<T>(b),      \
                               *ldb, *rcond, rank, As<T>(work), rwork, iwork); \
  }                                                                            \
  HWYBLAS_API void prefix##gelsd_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* nrhs,     \
      R* a, const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* s,        \
      const R* rcond, hwyblas_int* rank, R* work, const hwyblas_int* lwork,    \
      R* rwork, hwyblas_int* iwork, hwyblas_int* info) {                       \
    *info = hwyblas::lapack::Gelsd(*m, *n, *nrhs, As<T>(a), *lda, As<T>(b),    \
                                   *ldb, s, *rcond, rank, As<T>(work), *lwork, \
                                   rwork, iwork);                              \
  }

HWYBLAS_LAPACK_GELSD_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_GELSD_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_GELSD_COMPLEX

}  // extern "C"
