// Generalized linear least squares problems: the linear equality-constrained
// problem (xGGLSE) and the general Gauss-Markov linear model (xGGGLM), with
// the generalized QR and RQ factorizations (xGGQRF, xGGRQF).
//
// Each template is the real routine for real T and the complex one for
// complex T; they differ only in conjugations and in TRANS = 'T' (real) or
// 'C' (complex) for Q**T / Q**H.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <complex>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {
namespace {

// INT(WORK(i)): the workspace size a routine returned in WORK(i).
template <typename T>
blasint Lwork(const T& x) {
  return static_cast<blasint>(std::real(x));
}

}  // namespace

// xGGQRF: computes the generalized QR factorization of the N-by-M matrix A
// and the N-by-P matrix B: A = Q*R, B = Q*T*Z, with Q and Z orthogonal
// (unitary), R upper trapezoidal and T upper trapezoidal (the RQ form of
// xGERQF). On exit, A and TAUA hold R and Q as returned by xGEQRF, B and
// TAUB hold T and Z as returned by xGERQF. LWORK >= max(1, N, M, P); the
// optimal LWORK is max(N, M, P) * NB. LWORK = -1 is a workspace query.
template <typename T>
blasint Ggqrf(blasint n, blasint m, blasint p, T* a, blasint lda, T* taua, T* b,
              blasint ldb, T* taub, T* work, blasint lwork) {
  const Vector<T> WORK(work);
  blasint lopt, lwkopt, nb, nb1, nb2, nb3;
  bool lquery;

  // Test the input parameters
  blasint info = 0;
  nb1 = Ilaenv<T>(1, "GEQRF", " ", n, m, -1, -1);
  nb2 = Ilaenv<T>(1, "GERQF", " ", n, p, -1, -1);
  nb3 = Ilaenv<T>(1, RealOrComplex<T>("ORMQR", "UNMQR"), " ", n, m, p, -1);
  nb = Max(nb1, nb2, nb3);
  lwkopt = Max<blasint>(1, Max(n, m, p) * nb);
  WORK(1) = WorkSize<T>(lwkopt);
  lquery = (lwork == -1);
  if (n < 0) {
    info = -1;
  } else if (m < 0) {
    info = -2;
  } else if (p < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  } else if (lwork < Max<blasint>(1, n, m, p) && !lquery) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>("GGQRF", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // QR factorization of N-by-M matrix A: A = Q*R
  info = Geqrf(n, m, a, lda, taua, work, lwork);
  lopt = Lwork(WORK(1));

  // Update B := Q**H*B.
  info = Unmqr('L', ConjTrans<T>(), n, p, Min(n, m), a, lda, taua, b, ldb, work,
               lwork);
  lopt = Max(lopt, Lwork(WORK(1)));

  // RQ factorization of N-by-P matrix B: B = T*Z.
  info = Gerqf(n, p, b, ldb, taub, work, lwork);
  WORK(1) = WorkSize<T>(Max(lopt, Lwork(WORK(1))));

  return info;
}

// xGGRQF: computes the generalized RQ factorization of the M-by-N matrix A
// and the P-by-N matrix B: A = R*Q, B = Z*T*Q, with Q and Z orthogonal
// (unitary), R upper trapezoidal (the RQ form of xGERQF) and T upper
// trapezoidal. On exit, A and TAUA hold R and Q as returned by xGERQF, B
// and TAUB hold T and Z as returned by xGEQRF. LWORK >= max(1, N, M, P);
// the optimal LWORK is max(N, M, P) * NB. LWORK = -1 is a workspace query.
template <typename T>
blasint Ggrqf(blasint m, blasint p, blasint n, T* a, blasint lda, T* taua, T* b,
              blasint ldb, T* taub, T* work, blasint lwork) {
  const Matrix<T> A(a, lda);
  const Vector<T> WORK(work);
  blasint lopt, lwkopt, nb, nb1, nb2, nb3;
  bool lquery;

  // Test the input parameters
  blasint info = 0;
  nb1 = Ilaenv<T>(1, "GERQF", " ", m, n, -1, -1);
  nb2 = Ilaenv<T>(1, "GEQRF", " ", p, n, -1, -1);
  nb3 = Ilaenv<T>(1, RealOrComplex<T>("ORMRQ", "UNMRQ"), " ", m, n, p, -1);
  nb = Max(nb1, nb2, nb3);
  lwkopt = Max<blasint>(1, Max(n, m, p) * nb);
  WORK(1) = WorkSize<T>(lwkopt);
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (p < 0) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, p)) {
    info = -8;
  } else if (lwork < Max<blasint>(1, m, p, n) && !lquery) {
    info = -11;
  }
  if (info != 0) {
    Xerbla<T>("GGRQF", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // RQ factorization of M-by-N matrix A: A = R*Q
  info = Gerqf(m, n, a, lda, taua, work, lwork);
  lopt = Lwork(WORK(1));

  // Update B := B*Q**H
  info = Unmrq('R', ConjTrans<T>(), p, n, Min(m, n),
               A.ptr(Max<blasint>(1, m - n + 1), 1), lda, taua, b, ldb, work,
               lwork);
  lopt = Max(lopt, Lwork(WORK(1)));

  // QR factorization of P-by-N matrix B: B = Z*T
  info = Geqrf(p, n, b, ldb, taub, work, lwork);
  WORK(1) = WorkSize<T>(Max(lopt, Lwork(WORK(1))));

  return info;
}

// xGGLSE: solves the linear equality-constrained least squares problem
//
//     minimize || c - A*x ||_2   subject to   B*x = d
//
// for the M-by-N matrix A and the P-by-N matrix B (P <= N <= M+P), with
// rank(B) = P and rank( (A; B) ) = N, by the generalized RQ factorization
// of (B, A) (xGGRQF). On exit, X holds the solution, C(N-P+1:M) the
// residual sum of squares in its norm, D is destroyed, and A and B hold
// the factors. LWORK >= max(1, M+N+P); the optimal LWORK is P +
// min(M,N) + max(M,N)*NB. LWORK = -1 is a workspace query. Returns INFO:
// 0, -i for an illegal argument i, 1 if T12 (B's factor) is singular, or
// 2 if R11 (A's) is singular.
template <typename T>
blasint Gglse(blasint m, blasint n, blasint p, T* a, blasint lda, T* b,
              blasint ldb, T* c, T* d, T* x, T* work, blasint lwork) {
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const Vector<T> C(c);
  const Vector<T> D(d);
  const Vector<T> WORK(work);
  const Vector<T> X(x);
  blasint lopt, lwkmin, lwkopt, mn, nb, nb1, nb2, nb3, nb4, nr;
  bool lquery;

  // Test the input parameters
  blasint info = 0;
  mn = Min(m, n);
  lquery = (lwork == -1);
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (p < 0 || p > n || p < n - m) {
    info = -3;
  } else if (lda < Max<blasint>(1, m)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, p)) {
    info = -7;
  }

  // Calculate workspace
  if (info == 0) {
    if (n == 0) {
      lwkmin = 1;
      lwkopt = 1;
    } else {
      nb1 = Ilaenv<T>(1, "GEQRF", " ", m, n, -1, -1);
      nb2 = Ilaenv<T>(1, "GERQF", " ", m, n, -1, -1);
      nb3 = Ilaenv<T>(1, RealOrComplex<T>("ORMQR", "UNMQR"), " ", m, n, p, -1);
      nb4 = Ilaenv<T>(1, RealOrComplex<T>("ORMRQ", "UNMRQ"), " ", m, n, p, -1);
      nb = Max(nb1, nb2, nb3, nb4);
      lwkmin = m + n + p;
      lwkopt = p + mn + Max(m, n) * nb;
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (lwork < lwkmin && !lquery) info = -12;
  }

  if (info != 0) {
    Xerbla<T>("GGLSE", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible
  if (n == 0) return info;

  // Compute the GRQ factorization of matrices B and A:
  //
  //        B*Q**H = (  0  T12 ) P   Z**H*A*Q**H = ( R11 R12 ) N-P
  //                  N-P  P                     (  0  R22 ) M+P-N
  //                                               N-P  P
  //
  // where T12 and R11 are upper triangular, and Q and Z are
  // unitary.
  info = Ggrqf(p, m, n, b, ldb, work, a, lda, WORK.ptr(p + 1),
               WORK.ptr(p + mn + 1), lwork - p - mn);
  lopt = Lwork(WORK(p + mn + 1));

  // Update c = Z**H *c = ( c1 ) N-P
  //                      ( c2 ) M+P-N
  info = Unmqr('L', ConjTrans<T>(), m, 1, mn, a, lda, WORK.ptr(p + 1), c,
               Max<blasint>(1, m), WORK.ptr(p + mn + 1), lwork - p - mn);
  lopt = Max(lopt, Lwork(WORK(p + mn + 1)));

  // Solve T12*x2 = d for x2
  if (p > 0) {
    info = Trtrs('U', 'N', 'N', p, 1, B.ptr(1, n - p + 1), ldb, d, p);

    if (info > 0) {
      info = 1;
      return info;
    }

    // Put the solution in X
    Copy(p, d, 1, X.ptr(n - p + 1), 1);

    // Update c1
    Gemv('N', n - p, p, -cone, A.ptr(1, n - p + 1), lda, d, 1, cone, c, 1);
  }

  // Solve R11*x1 = c1 for x1
  if (n > p) {
    info = Trtrs('U', 'N', 'N', n - p, 1, a, lda, c, n - p);

    if (info > 0) {
      info = 2;
      return info;
    }

    // Put the solutions in X
    Copy(n - p, c, 1, x, 1);
  }

  // Compute the residual vector:
  if (m < n) {
    nr = m + p - n;
    if (nr > 0) {
      Gemv('N', nr, n - m, -cone, A.ptr(n - p + 1, m + 1), lda, D.ptr(nr + 1),
           1, cone, C.ptr(n - p + 1), 1);
    }
  } else {
    nr = p;
  }
  if (nr > 0) {
    Trmv('U', 'N', 'N', nr, A.ptr(n - p + 1, n - p + 1), lda, d, 1);
    Axpy(nr, -cone, d, 1, C.ptr(n - p + 1), 1);
  }

  // Backward transformation x = Q**H*x
  info = Unmrq('L', ConjTrans<T>(), n, 1, p, b, ldb, WORK.ptr(1), x, n,
               WORK.ptr(p + mn + 1), lwork - p - mn);
  WORK(1) = WorkSize<T>(p + mn + Max(lopt, Lwork(WORK(p + mn + 1))));

  return info;
}

// xGGGLM: solves the general Gauss-Markov linear model problem
//
//     minimize || y ||_2   subject to   d = A*x + B*y
//
// for the N-by-M matrix A and the N-by-P matrix B (M <= N <= M+P), with
// rank(A) = M and rank( A B ) = N, by the generalized QR factorization of
// (A, B) (xGGQRF). On exit, X and Y hold the solution, D is destroyed, and
// A and B hold the factors. LWORK >= max(1, N+M+P); the optimal LWORK is M
// + min(N,P) + max(N,P)*NB. LWORK = -1 is a workspace query. Returns INFO:
// 0, -i for an illegal argument i, 1 if T22 (B's factor) is singular, or
// 2 if R11 (A's) is singular.
template <typename T>
blasint Ggglm(blasint n, blasint m, blasint p, T* a, blasint lda, T* b,
              blasint ldb, T* d, T* x, T* y, T* work, blasint lwork) {
  const T czero = T(0);
  const T cone = T(1);
  const Matrix<T> B(b, ldb);
  const Vector<T> D(d);
  const Vector<T> WORK(work);
  const Vector<T> X(x);
  const Vector<T> Y(y);
  blasint i, lopt, lwkmin, lwkopt, nb, nb1, nb2, nb3, nb4, np;
  bool lquery;

  // Test the input parameters
  blasint info = 0;
  np = Min(n, p);
  lquery = (lwork == -1);
  if (n < 0) {
    info = -1;
  } else if (m < 0 || m > n) {
    info = -2;
  } else if (p < 0 || p < n - m) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }

  // Calculate workspace
  if (info == 0) {
    if (n == 0) {
      lwkmin = 1;
      lwkopt = 1;
    } else {
      nb1 = Ilaenv<T>(1, "GEQRF", " ", n, m, -1, -1);
      nb2 = Ilaenv<T>(1, "GERQF", " ", n, m, -1, -1);
      nb3 = Ilaenv<T>(1, RealOrComplex<T>("ORMQR", "UNMQR"), " ", n, m, p, -1);
      nb4 = Ilaenv<T>(1, RealOrComplex<T>("ORMRQ", "UNMRQ"), " ", n, m, p, -1);
      nb = Max(nb1, nb2, nb3, nb4);
      lwkmin = m + n + p;
      lwkopt = m + np + Max(n, p) * nb;
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (lwork < lwkmin && !lquery) info = -12;
  }

  if (info != 0) {
    Xerbla<T>("GGGLM", -info);
    return info;
  } else if (lquery) {
    return info;
  }

  // Quick return if possible
  if (n == 0) {
    for (i = 1; i <= m; ++i) {
      X(i) = czero;
    }
    for (i = 1; i <= p; ++i) {
      Y(i) = czero;
    }
    return info;
  }

  // Compute the GQR factorization of matrices A and B:
  //
  //       Q**H*A = ( R11 ) M,    Q**H*B*Z**H = ( T11   T12 ) M
  //                (  0  ) N-M                 (  0    T22 ) N-M
  //                   M                         M+P-N  N-M
  //
  // where R11 and T22 are upper triangular, and Q and Z are
  // unitary.
  info = Ggqrf(n, m, p, a, lda, work, b, ldb, WORK.ptr(m + 1),
               WORK.ptr(m + np + 1), lwork - m - np);
  lopt = Lwork(WORK(m + np + 1));

  // Update left-hand-side vector d = Q**H*d = ( d1 ) M
  //                                           ( d2 ) N-M
  info = Unmqr('L', ConjTrans<T>(), n, 1, m, a, lda, work, d,
               Max<blasint>(1, n), WORK.ptr(m + np + 1), lwork - m - np);
  lopt = Max(lopt, Lwork(WORK(m + np + 1)));

  // Solve T22*y2 = d2 for y2
  if (n > m) {
    info = Trtrs('U', 'N', 'N', n - m, 1, B.ptr(m + 1, m + p - n + 1), ldb,
                 D.ptr(m + 1), n - m);

    if (info > 0) {
      info = 1;
      return info;
    }

    Copy(n - m, D.ptr(m + 1), 1, Y.ptr(m + p - n + 1), 1);
  }

  // Set y1 = 0
  for (i = 1; i <= m + p - n; ++i) {
    Y(i) = czero;
  }

  // Update d1 = d1 - T12*y2
  Gemv('N', m, n - m, -cone, B.ptr(1, m + p - n + 1), ldb, Y.ptr(m + p - n + 1),
       1, cone, d, 1);

  // Solve triangular system: R11*x = d1
  if (m > 0) {
    info = Trtrs('U', 'N', 'N', m, 1, a, lda, d, m);

    if (info > 0) {
      info = 2;
      return info;
    }

    // Copy D to X
    Copy(m, d, 1, x, 1);
  }

  // Backward transformation y = Z**H *y
  info = Unmrq('L', ConjTrans<T>(), p, 1, np,
               B.ptr(Max<blasint>(1, n - p + 1), 1), ldb, WORK.ptr(m + 1), y,
               Max<blasint>(1, p), WORK.ptr(m + np + 1), lwork - m - np);
  WORK(1) = WorkSize<T>(m + np + Max(lopt, Lwork(WORK(m + np + 1))));

  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                       \
  template blasint Ggqrf<T>(blasint, blasint, blasint, T*, blasint, T*, T*, \
                            blasint, T*, T*, blasint);                      \
  template blasint Ggrqf<T>(blasint, blasint, blasint, T*, blasint, T*, T*, \
                            blasint, T*, T*, blasint);                      \
  template blasint Gglse<T>(blasint, blasint, blasint, T*, blasint, T*,     \
                            blasint, T*, T*, T*, T*, blasint);              \
  template blasint Ggglm<T>(blasint, blasint, blasint, T*, blasint, T*,     \
                            blasint, T*, T*, T*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GGLSE(prefix, T, R)                                     \
  HWYBLAS_API void prefix##ggqrf_(                                             \
      const hwyblas_int* n, const hwyblas_int* m, const hwyblas_int* p, R* a,  \
      const hwyblas_int* lda, R* taua, R* b, const hwyblas_int* ldb, R* taub,  \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Ggqrf(*n, *m, *p, As<T>(a), *lda, As<T>(taua),    \
                                   As<T>(b), *ldb, As<T>(taub), As<T>(work),   \
                                   *lwork);                                    \
  }                                                                            \
  HWYBLAS_API void prefix##ggrqf_(                                             \
      const hwyblas_int* m, const hwyblas_int* p, const hwyblas_int* n, R* a,  \
      const hwyblas_int* lda, R* taua, R* b, const hwyblas_int* ldb, R* taub,  \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Ggrqf(*m, *p, *n, As<T>(a), *lda, As<T>(taua),    \
                                   As<T>(b), *ldb, As<T>(taub), As<T>(work),   \
                                   *lwork);                                    \
  }                                                                            \
  HWYBLAS_API void prefix##gglse_(                                             \
      const hwyblas_int* m, const hwyblas_int* n, const hwyblas_int* p, R* a,  \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* c, R* d, R* x,  \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Gglse(*m, *n, *p, As<T>(a), *lda, As<T>(b), *ldb, \
                                   As<T>(c), As<T>(d), As<T>(x), As<T>(work),  \
                                   *lwork);                                    \
  }                                                                            \
  HWYBLAS_API void prefix##ggglm_(                                             \
      const hwyblas_int* n, const hwyblas_int* m, const hwyblas_int* p, R* a,  \
      const hwyblas_int* lda, R* b, const hwyblas_int* ldb, R* d, R* x, R* y,  \
      R* work, const hwyblas_int* lwork, hwyblas_int* info) {                  \
    *info = hwyblas::lapack::Ggglm(*n, *m, *p, As<T>(a), *lda, As<T>(b), *ldb, \
                                   As<T>(d), As<T>(x), As<T>(y), As<T>(work),  \
                                   *lwork);                                    \
  }

HWYBLAS_LAPACK_GGLSE(s, float, float)
HWYBLAS_LAPACK_GGLSE(d, double, double)
HWYBLAS_LAPACK_GGLSE(c, c32, float)
HWYBLAS_LAPACK_GGLSE(z, c64, double)

#undef HWYBLAS_LAPACK_GGLSE

}  // extern "C"
