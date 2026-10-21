// LAPACKE functions of the least squares solvers: xGELS, xGELSS, xGELSD,
// xGELSY, xGGLSE, xGGGLM, and the factorizations they use: xGERQF,
// xTZRZF, xGGQRF, xGGRQF, xORMRQ / xUNMRQ and xORMRZ / xUNMRZ (see
// lapacke.cc and lapacke_util.h).
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#include <algorithm>
#include <string>

#include "hwyblas/lapacke.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"
#include "src/lapack/lapacke_util.h"

namespace hwyblas {
namespace lapacke {
namespace {

// ---------------------------------------------------------------------------
// Least squares

// The NaN checks of A (m x n) and B (max(m,n) x nrhs) and, for the rank
// revealing drivers, RCOND.
template <typename T>
lapack_int LsNancheck(int matrix_layout, lapack_int m, lapack_int n,
                      lapack_int nrhs, const T* a, lapack_int lda, const T* b,
                      lapack_int ldb, lapack_int a_info) {
  if (GeNancheck(matrix_layout, m, n, a, lda)) return a_info;
  if (GeNancheck(matrix_layout, std::max(m, n), nrhs, b, ldb)) {
    return a_info - 2;
  }
  return 0;
}

// xGELS. In row-major layout, A (m x n) and B (max(m,n) x nrhs) are copied
// in and back.
template <typename T>
lapack_int GelsWork(int matrix_layout, char trans, lapack_int m, lapack_int n,
                    lapack_int nrhs, T* a, lapack_int lda, T* b, lapack_int ldb,
                    T* work, lapack_int lwork) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gels_work"), lwork == -1,
      {{{a, m, n, lda, -7}, {b, std::max(m, n), nrhs, ldb, -9}}},
      [&](auto& p, auto& ld) {
        return lapack::Gels(trans, m, n, nrhs, p[0], ld[0], p[1], ld[1], work,
                            lwork);
      });
}

template <typename T>
lapack_int Gels(int matrix_layout, char trans, lapack_int m, lapack_int n,
                lapack_int nrhs, T* a, lapack_int lda, T* b, lapack_int ldb) {
  const std::string name = Name<T>("gels");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int info =
        LsNancheck(matrix_layout, m, n, nrhs, a, lda, b, ldb, -6);
    if (info != 0) return info;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GelsWork(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb, work,
                    lwork);
  });
}

// xGELSS; rwork for complex T only.
template <typename T>
lapack_int GelssWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int nrhs, T* a, lapack_int lda, T* b,
                     lapack_int ldb, RealT<T>* s, RealT<T> rcond,
                     lapack_int* rank, T* work, lapack_int lwork,
                     RealT<T>* rwork) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gelss_work"), lwork == -1,
      {{{a, m, n, lda, -6}, {b, std::max(m, n), nrhs, ldb, -8}}},
      [&](auto& p, auto& ld) {
        return lapack::Gelss(m, n, nrhs, p[0], ld[0], p[1], ld[1], s, rcond,
                             rank, work, lwork, rwork);
      });
}

template <typename T>
lapack_int Gelss(int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
                 T* a, lapack_int lda, T* b, lapack_int ldb, RealT<T>* s,
                 RealT<T> rcond, lapack_int* rank) {
  using R = RealT<T>;
  const std::string name = Name<T>("gelss");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int info =
        LsNancheck(matrix_layout, m, n, nrhs, a, lda, b, ldb, -5);
    if (info != 0) return info;
    if (VectorNancheck(1, &rcond, 1)) return -10;
  }
  // RWORK (complex T) has 5*min(M,N) elements.
  const auto call = [&](R* rwork) {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return GelssWork(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond,
                       rank, work, lwork, rwork);
    });
  };
  if constexpr (IsComplex<T>::value) {
    return WithWorkArray<R>(name, std::max<lapack_int>(1, 5 * std::min(m, n)),
                            call);
  } else {
    return call(static_cast<R*>(nullptr));
  }
}

// xGELSD; rwork for complex T only.
template <typename T>
lapack_int GelsdWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int nrhs, T* a, lapack_int lda, T* b,
                     lapack_int ldb, RealT<T>* s, RealT<T> rcond,
                     lapack_int* rank, T* work, lapack_int lwork,
                     RealT<T>* rwork, lapack_int* iwork) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gelsd_work"), lwork == -1,
      {{{a, m, n, lda, -6}, {b, std::max(m, n), nrhs, ldb, -8}}},
      [&](auto& p, auto& ld) {
        return lapack::Gelsd(m, n, nrhs, p[0], ld[0], p[1], ld[1], s, rcond,
                             rank, work, lwork, rwork, iwork);
      });
}

template <typename T>
lapack_int Gelsd(int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
                 T* a, lapack_int lda, T* b, lapack_int ldb, RealT<T>* s,
                 RealT<T> rcond, lapack_int* rank) {
  const std::string name = Name<T>("gelsd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int info =
        LsNancheck(matrix_layout, m, n, nrhs, a, lda, b, ldb, -5);
    if (info != 0) return info;
    if (VectorNancheck(1, &rcond, 1)) return -10;
  }
  // The query gives LWORK, LRWORK (complex T) and LIWORK.
  return WithWorkspaces<T>(name, [&](T* work, lapack_int lwork, RealT<T>* rwork,
                                     lapack_int /*lrwork*/, lapack_int* iwork,
                                     lapack_int /*liwork*/) {
    return GelsdWork(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank,
                     work, lwork, rwork, iwork);
  });
}

// xGELSY; rwork for complex T only.
template <typename T>
lapack_int GelsyWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int nrhs, T* a, lapack_int lda, T* b,
                     lapack_int ldb, lapack_int* jpvt, RealT<T> rcond,
                     lapack_int* rank, T* work, lapack_int lwork,
                     RealT<T>* rwork) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gelsy_work"), lwork == -1,
      {{{a, m, n, lda, -6}, {b, std::max(m, n), nrhs, ldb, -8}}},
      [&](auto& p, auto& ld) {
        return lapack::Gelsy(m, n, nrhs, p[0], ld[0], p[1], ld[1], jpvt, rcond,
                             rank, work, lwork, rwork);
      });
}

template <typename T>
lapack_int Gelsy(int matrix_layout, lapack_int m, lapack_int n, lapack_int nrhs,
                 T* a, lapack_int lda, T* b, lapack_int ldb, lapack_int* jpvt,
                 RealT<T> rcond, lapack_int* rank) {
  using R = RealT<T>;
  const std::string name = Name<T>("gelsy");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int info =
        LsNancheck(matrix_layout, m, n, nrhs, a, lda, b, ldb, -5);
    if (info != 0) return info;
    if (VectorNancheck(1, &rcond, 1)) return -10;
  }
  // RWORK (complex T) has 2*N elements.
  const auto call = [&](R* rwork) {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return GelsyWork(matrix_layout, m, n, nrhs, a, lda, b, ldb, jpvt, rcond,
                       rank, work, lwork, rwork);
    });
  };
  if constexpr (IsComplex<T>::value) {
    return WithWorkArray<R>(name, std::max<lapack_int>(1, 2 * n), call);
  } else {
    return call(static_cast<R*>(nullptr));
  }
}

// ---------------------------------------------------------------------------
// Generalized least squares

// xGGLSE: A m x n, B p x n.
template <typename T>
lapack_int GglseWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int p, T* a, lapack_int lda, T* b, lapack_int ldb,
                     T* c, T* d, T* x, T* work, lapack_int lwork) {
  return CallWork<T, 2>(matrix_layout, Name<T>("gglse_work"), lwork == -1,
                        {{{a, m, n, lda, -6}, {b, p, n, ldb, -8}}},
                        [&](auto& pp, auto& ld) {
                          return lapack::Gglse(m, n, p, pp[0], ld[0], pp[1],
                                               ld[1], c, d, x, work, lwork);
                        });
}

template <typename T>
lapack_int Gglse(int matrix_layout, lapack_int m, lapack_int n, lapack_int p,
                 T* a, lapack_int lda, T* b, lapack_int ldb, T* c, T* d, T* x) {
  const std::string name = Name<T>("gglse");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, p, n, b, ldb)) return -7;
    if (VectorNancheck(m, c, 1)) return -9;
    if (VectorNancheck(p, d, 1)) return -10;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GglseWork(matrix_layout, m, n, p, a, lda, b, ldb, c, d, x, work,
                     lwork);
  });
}

// xGGGLM: A n x m, B n x p.
template <typename T>
lapack_int GgglmWork(int matrix_layout, lapack_int n, lapack_int m,
                     lapack_int p, T* a, lapack_int lda, T* b, lapack_int ldb,
                     T* d, T* x, T* y, T* work, lapack_int lwork) {
  return CallWork<T, 2>(matrix_layout, Name<T>("ggglm_work"), lwork == -1,
                        {{{a, n, m, lda, -6}, {b, n, p, ldb, -8}}},
                        [&](auto& pp, auto& ld) {
                          return lapack::Ggglm(n, m, p, pp[0], ld[0], pp[1],
                                               ld[1], d, x, y, work, lwork);
                        });
}

template <typename T>
lapack_int Ggglm(int matrix_layout, lapack_int n, lapack_int m, lapack_int p,
                 T* a, lapack_int lda, T* b, lapack_int ldb, T* d, T* x, T* y) {
  const std::string name = Name<T>("ggglm");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, m, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, p, b, ldb)) return -7;
    if (VectorNancheck(n, d, 1)) return -9;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GgglmWork(matrix_layout, n, m, p, a, lda, b, ldb, d, x, y, work,
                     lwork);
  });
}

// xGGQRF (rq = false; A n x m, B n x p, with the arguments (N, M, P)) and
// xGGRQF (rq = true; A m x n, B p x n, with the arguments (M, P, N)).
template <typename T>
lapack_int GgqrfWork(int matrix_layout, lapack_int d1, lapack_int d2,
                     lapack_int d3, T* a, lapack_int lda, T* taua, T* b,
                     lapack_int ldb, T* taub, T* work, lapack_int lwork,
                     bool rq) {
  const MatrixArg<T> a_arg =
      rq ? MatrixArg<T>{a, d1, d3, lda, -6} : MatrixArg<T>{a, d1, d2, lda, -6};
  const MatrixArg<T> b_arg =
      rq ? MatrixArg<T>{b, d2, d3, ldb, -9} : MatrixArg<T>{b, d1, d3, ldb, -9};
  return CallWork<T, 2>(
      matrix_layout, Name<T>(rq ? "ggrqf_work" : "ggqrf_work"), lwork == -1,
      {{a_arg, b_arg}}, [&](auto& p, auto& ld) {
        return rq ? lapack::Ggrqf(d1, d2, d3, p[0], ld[0], taua, p[1], ld[1],
                                  taub, work, lwork)
                  : lapack::Ggqrf(d1, d2, d3, p[0], ld[0], taua, p[1], ld[1],
                                  taub, work, lwork);
      });
}

template <typename T>
lapack_int Ggqrf(int matrix_layout, lapack_int d1, lapack_int d2, lapack_int d3,
                 T* a, lapack_int lda, T* taua, T* b, lapack_int ldb, T* taub,
                 bool rq) {
  const std::string name = Name<T>(rq ? "ggrqf" : "ggqrf");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (rq ? GeNancheck(matrix_layout, d1, d3, a, lda)
           : GeNancheck(matrix_layout, d1, d2, a, lda)) {
      return -5;
    }
    if (rq ? GeNancheck(matrix_layout, d2, d3, b, ldb)
           : GeNancheck(matrix_layout, d1, d3, b, ldb)) {
      return -8;
    }
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GgqrfWork(matrix_layout, d1, d2, d3, a, lda, taua, b, ldb, taub,
                     work, lwork, rq);
  });
}

// ---------------------------------------------------------------------------
// RQ and RZ factorizations

// xGERQF (rz = false) and xTZRZF (rz = true).
template <typename T>
lapack_int GerqfWork(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, T* tau, T* work, lapack_int lwork,
                     bool rz) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(rz ? "tzrzf_work" : "gerqf_work"), lwork == -1,
      {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
        return rz ? lapack::Tzrzf(m, n, p[0], ld[0], tau, work, lwork)
                  : lapack::Gerqf(m, n, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Gerqf(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, T* tau, bool rz) {
  const std::string name = Name<T>(rz ? "tzrzf" : "gerqf");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GerqfWork(matrix_layout, m, n, a, lda, tau, work, lwork, rz);
  });
}

// xORMRQ / xUNMRQ (rz = false) and xORMRZ / xUNMRZ (rz = true, with L).
// A is K x M (SIDE = 'L') or K x N (SIDE = 'R'). Deviation: the reference
// LAPACKE (3.12.1) takes A as K x M also for SIDE = 'R', in the NaN check
// and in row-major layout (where it checks LDA against M and copies only M
// columns, of which LAPACK reads N); here A is K x N for SIDE = 'R'.
template <typename T>
lapack_int UnmrqWork(int matrix_layout, char side, char trans, lapack_int m,
                     lapack_int n, lapack_int k, lapack_int l, const T* a,
                     lapack_int lda, const T* tau, T* c, lapack_int ldc,
                     T* work, lapack_int lwork, bool rz) {
  const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
  const lapack_int shift = rz ? 1 : 0;
  return CallWork<T, 2>(
      matrix_layout,
      Name<T>(rz ? RealOrComplex<T>("ormrz_work", "unmrz_work")
                 : RealOrComplex<T>("ormrq_work", "unmrq_work")),
      lwork == -1,
      {{Input(a, k, r, lda, -8 - shift), {c, m, n, ldc, -11 - shift}}},
      [&](auto& p, auto& ld) {
        // The complex routines conjugate rows of A while they run, and
        // restore them: the routine changes A, but not the caller's A.
        return rz ? lapack::Unmrz(side, trans, m, n, k, l, p[0], ld[0], tau,
                                  p[1], ld[1], work, lwork)
                  : lapack::Unmrq(side, trans, m, n, k, p[0], ld[0], tau, p[1],
                                  ld[1], work, lwork);
      });
}

template <typename T>
lapack_int Unmrq(int matrix_layout, char side, char trans, lapack_int m,
                 lapack_int n, lapack_int k, lapack_int l, const T* a,
                 lapack_int lda, const T* tau, T* c, lapack_int ldc, bool rz) {
  const std::string name = Name<T>(rz ? RealOrComplex<T>("ormrz", "unmrz")
                                      : RealOrComplex<T>("ormrq", "unmrq"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  const lapack_int shift = rz ? 1 : 0;
  if (NanCheck()) {
    const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
    if (GeNancheck(matrix_layout, k, r, a, lda)) return -7 - shift;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -10 - shift;
    if (VectorNancheck(k, tau, 1)) return -9 - shift;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnmrqWork(matrix_layout, side, trans, m, n, k, l, a, lda, tau, c,
                     ldc, work, lwork, rz);
  });
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose arguments are the same in all four precisions.
#define HWYBLAS_LAPACKE_GELS(p, T, R)                                          \
  HWYBLAS_API lapack_int LAPACKE_##p##gels(                                    \
      int layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs,     \
      T* a, lapack_int lda, T* b, lapack_int ldb) {                            \
    return lapacke::Gels(layout, trans, m, n, nrhs, a, lda, b, ldb);           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gels_work(                               \
      int layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs,     \
      T* a, lapack_int lda, T* b, lapack_int ldb, T* work, lapack_int lwork) { \
    return lapacke::GelsWork(layout, trans, m, n, nrhs, a, lda, b, ldb, work,  \
                             lwork);                                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelss(                                   \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, T* a,           \
      lapack_int lda, T* b, lapack_int ldb, R* s, R rcond, lapack_int* rank) { \
    return lapacke::Gelss(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelsd(                                   \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, T* a,           \
      lapack_int lda, T* b, lapack_int ldb, R* s, R rcond, lapack_int* rank) { \
    return lapacke::Gelsd(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelsy(                                   \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, T* a,           \
      lapack_int lda, T* b, lapack_int ldb, lapack_int* jpvt, R rcond,         \
      lapack_int* rank) {                                                      \
    return lapacke::Gelsy(layout, m, n, nrhs, a, lda, b, ldb, jpvt, rcond,     \
                          rank);                                               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gglse(                                   \
      int layout, lapack_int m, lapack_int n, lapack_int p_, T* a,             \
      lapack_int lda, T* b, lapack_int ldb, T* c, T* d, T* x) {                \
    return lapacke::Gglse(layout, m, n, p_, a, lda, b, ldb, c, d, x);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gglse_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int p_, T* a,             \
      lapack_int lda, T* b, lapack_int ldb, T* c, T* d, T* x, T* work,         \
      lapack_int lwork) {                                                      \
    return lapacke::GglseWork(layout, m, n, p_, a, lda, b, ldb, c, d, x, work, \
                              lwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ggglm(                                   \
      int layout, lapack_int n, lapack_int m, lapack_int p_, T* a,             \
      lapack_int lda, T* b, lapack_int ldb, T* d, T* x, T* y) {                \
    return lapacke::Ggglm(layout, n, m, p_, a, lda, b, ldb, d, x, y);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ggglm_work(                              \
      int layout, lapack_int n, lapack_int m, lapack_int p_, T* a,             \
      lapack_int lda, T* b, lapack_int ldb, T* d, T* x, T* y, T* work,         \
      lapack_int lwork) {                                                      \
    return lapacke::GgglmWork(layout, n, m, p_, a, lda, b, ldb, d, x, y, work, \
                              lwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ggqrf(                                   \
      int layout, lapack_int n, lapack_int m, lapack_int p_, T* a,             \
      lapack_int lda, T* taua, T* b, lapack_int ldb, T* taub) {                \
    return lapacke::Ggqrf(layout, n, m, p_, a, lda, taua, b, ldb, taub,        \
                          false);                                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ggqrf_work(                              \
      int layout, lapack_int n, lapack_int m, lapack_int p_, T* a,             \
      lapack_int lda, T* taua, T* b, lapack_int ldb, T* taub, T* work,         \
      lapack_int lwork) {                                                      \
    return lapacke::GgqrfWork(layout, n, m, p_, a, lda, taua, b, ldb, taub,    \
                              work, lwork, false);                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ggrqf(                                   \
      int layout, lapack_int m, lapack_int p_, lapack_int n, T* a,             \
      lapack_int lda, T* taua, T* b, lapack_int ldb, T* taub) {                \
    return lapacke::Ggqrf(layout, m, p_, n, a, lda, taua, b, ldb, taub, true); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ggrqf_work(                              \
      int layout, lapack_int m, lapack_int p_, lapack_int n, T* a,             \
      lapack_int lda, T* taua, T* b, lapack_int ldb, T* taub, T* work,         \
      lapack_int lwork) {                                                      \
    return lapacke::GgqrfWork(layout, m, p_, n, a, lda, taua, b, ldb, taub,    \
                              work, lwork, true);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gerqf(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) {  \
    return lapacke::Gerqf(layout, m, n, a, lda, tau, false);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gerqf_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,    \
      T* work, lapack_int lwork) {                                             \
    return lapacke::GerqfWork(layout, m, n, a, lda, tau, work, lwork, false);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##tzrzf(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) {  \
    return lapacke::Gerqf(layout, m, n, a, lda, tau, true);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##tzrzf_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,    \
      T* work, lapack_int lwork) {                                             \
    return lapacke::GerqfWork(layout, m, n, a, lda, tau, work, lwork, true);   \
  }

HWYBLAS_LAPACKE_GELS(s, float, float)
HWYBLAS_LAPACKE_GELS(d, double, double)
HWYBLAS_LAPACKE_GELS(c, c32, float)
HWYBLAS_LAPACKE_GELS(z, c64, double)

#undef HWYBLAS_LAPACKE_GELS

// The functions with a real and a complex version: `mul` is the middle of
// the names ("orm" / "unm").
#define HWYBLAS_LAPACKE_UNMRQ(p, mul, T)                                       \
  HWYBLAS_API lapack_int LAPACKE_##p##mul##rq(                                 \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc) {                                                        \
    return lapacke::Unmrq(layout, side, trans, m, n, k, 0, a, lda, tau, c,     \
                          ldc, false);                                         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul##rq_work(                            \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc, T* work, lapack_int lwork) {                             \
    return lapacke::UnmrqWork(layout, side, trans, m, n, k, 0, a, lda, tau, c, \
                              ldc, work, lwork, false);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul##rz(                                 \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, lapack_int l, const T* a, lapack_int lda, const T* tau,    \
      T* c, lapack_int ldc) {                                                  \
    return lapacke::Unmrq(layout, side, trans, m, n, k, l, a, lda, tau, c,     \
                          ldc, true);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul##rz_work(                            \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, lapack_int l, const T* a, lapack_int lda, const T* tau,    \
      T* c, lapack_int ldc, T* work, lapack_int lwork) {                       \
    return lapacke::UnmrqWork(layout, side, trans, m, n, k, l, a, lda, tau, c, \
                              ldc, work, lwork, true);                         \
  }

HWYBLAS_LAPACKE_UNMRQ(s, orm, float)
HWYBLAS_LAPACKE_UNMRQ(d, orm, double)
HWYBLAS_LAPACKE_UNMRQ(c, unm, c32)
HWYBLAS_LAPACKE_UNMRQ(z, unm, c64)

#undef HWYBLAS_LAPACKE_UNMRQ

// The functions whose work arrays differ between real and complex.
#define HWYBLAS_LAPACKE_GELS_REAL(p, R)                                        \
  HWYBLAS_API lapack_int LAPACKE_##p##gelss_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, R* a,           \
      lapack_int lda, R* b, lapack_int ldb, R* s, R rcond, lapack_int* rank,   \
      R* work, lapack_int lwork) {                                             \
    return lapacke::GelssWork(layout, m, n, nrhs, a, lda, b, ldb, s, rcond,    \
                              rank, work, lwork, static_cast<R*>(nullptr));    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelsd_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, R* a,           \
      lapack_int lda, R* b, lapack_int ldb, R* s, R rcond, lapack_int* rank,   \
      R* work, lapack_int lwork, lapack_int* iwork) {                          \
    return lapacke::GelsdWork(layout, m, n, nrhs, a, lda, b, ldb, s, rcond,    \
                              rank, work, lwork, static_cast<R*>(nullptr),     \
                              iwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelsy_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, R* a,           \
      lapack_int lda, R* b, lapack_int ldb, lapack_int* jpvt, R rcond,         \
      lapack_int* rank, R* work, lapack_int lwork) {                           \
    return lapacke::GelsyWork(layout, m, n, nrhs, a, lda, b, ldb, jpvt, rcond, \
                              rank, work, lwork, static_cast<R*>(nullptr));    \
  }

HWYBLAS_LAPACKE_GELS_REAL(s, float)
HWYBLAS_LAPACKE_GELS_REAL(d, double)

#undef HWYBLAS_LAPACKE_GELS_REAL

#define HWYBLAS_LAPACKE_GELS_COMPLEX(p, T, R)                                  \
  HWYBLAS_API lapack_int LAPACKE_##p##gelss_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, T* a,           \
      lapack_int lda, T* b, lapack_int ldb, R* s, R rcond, lapack_int* rank,   \
      T* work, lapack_int lwork, R* rwork) {                                   \
    return lapacke::GelssWork(layout, m, n, nrhs, a, lda, b, ldb, s, rcond,    \
                              rank, work, lwork, rwork);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelsd_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, T* a,           \
      lapack_int lda, T* b, lapack_int ldb, R* s, R rcond, lapack_int* rank,   \
      T* work, lapack_int lwork, R* rwork, lapack_int* iwork) {                \
    return lapacke::GelsdWork(layout, m, n, nrhs, a, lda, b, ldb, s, rcond,    \
                              rank, work, lwork, rwork, iwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelsy_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int nrhs, T* a,           \
      lapack_int lda, T* b, lapack_int ldb, lapack_int* jpvt, R rcond,         \
      lapack_int* rank, T* work, lapack_int lwork, R* rwork) {                 \
    return lapacke::GelsyWork(layout, m, n, nrhs, a, lda, b, ldb, jpvt, rcond, \
                              rank, work, lwork, rwork);                       \
  }

HWYBLAS_LAPACKE_GELS_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_GELS_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_GELS_COMPLEX

}  // extern "C"
