// LAPACKE: the C interface to the LAPACK routines (see hwyblas/lapacke.h).
// Ported from the reference LAPACKE (LAPACK 3.12.1), with the same checks,
// INFO values and error messages. The reference repeats the same code for
// each routine and precision; here the utilities (NaN checks, transposes)
// are templates over the four precisions, and helpers do what all
// functions do (lapacke_util.h). The functions call the ported routines
// directly (which report invalid arguments, like the reference's Fortran
// routines). This file has the global functions (LAPACKE_xerbla, the NaN
// check switch) and the functions of the first ported routines; others are
// in lapacke_<family>.cc.
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#include "hwyblas/lapacke.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <string>

#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"
#include "src/lapack/lapacke_util.h"

namespace hwyblas {
namespace lapacke {
namespace {

// -1: not yet determined (see LAPACKE_get_nancheck).
std::atomic<int> g_nancheck{-1};

// ---------------------------------------------------------------------------
// LU factorization

template <typename T>
lapack_int GetrfWork(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, lapack_int* ipiv) {
  return CallWork<T, 1>(matrix_layout, Name<T>("getrf_work"), false,
                        {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
                          return lapack::Getrf(m, n, p[0], ld[0], ipiv);
                        });
}

template <typename T>
lapack_int Getrf(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, lapack_int* ipiv) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("getrf"), -1);
    return -1;
  }
  // Optionally check input matrices for NaNs.
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return GetrfWork(matrix_layout, m, n, a, lda, ipiv);
}

template <typename T>
lapack_int Getrf2Work(int matrix_layout, lapack_int m, lapack_int n, T* a,
                      lapack_int lda, lapack_int* ipiv) {
  return CallWork<T, 1>(matrix_layout, Name<T>("getrf2_work"), false,
                        {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
                          return lapack::Getrf2(m, n, p[0], ld[0], ipiv);
                        });
}

template <typename T>
lapack_int Getrf2(int matrix_layout, lapack_int m, lapack_int n, T* a,
                  lapack_int lda, lapack_int* ipiv) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("getrf2"), -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return Getrf2Work(matrix_layout, m, n, a, lda, ipiv);
}

// The row-major copy of A has as many rows as LASWP may touch: K2 and the
// pivot indices.
template <typename T>
lapack_int LaswpWork(int matrix_layout, lapack_int n, T* a, lapack_int lda,
                     lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                     lapack_int incx) {
  lapack_int rows = std::max<lapack_int>(1, k2);
  if (matrix_layout == LAPACK_ROW_MAJOR) {
    for (lapack_int i = k1; i <= k2; ++i) {
      rows = std::max(rows, ipiv[k1 + (i - k1) * std::abs(incx) - 1]);
    }
  }
  return CallWork<T, 1>(matrix_layout, Name<T>("laswp_work"), false,
                        {{{a, rows, n, lda, -4}}}, [&](auto& p, auto& ld) {
                          lapack::Laswp(n, p[0], ld[0], k1, k2, ipiv, incx);
                          return lapack_int{0};
                        });
}

// As in the reference, there is no NaN check: the number of rows of A is
// not known.
template <typename T>
lapack_int Laswp(int matrix_layout, lapack_int n, T* a, lapack_int lda,
                 lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                 lapack_int incx) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("laswp"), -1);
    return -1;
  }
  return LaswpWork(matrix_layout, n, a, lda, k1, k2, ipiv, incx);
}

// ---------------------------------------------------------------------------
// Cholesky factorization

template <typename T>
lapack_int PotrfWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda, bool recursive) {
  MatrixArg<T> arg{a, n, n, lda, -5};
  arg.uplo = lapack::Lsame(uplo, 'U') ? 'U' : 'L';
  return CallWork<T, 1>(
      matrix_layout, Name<T>(recursive ? "potrf2_work" : "potrf_work"), false,
      {{arg}}, [&](auto& p, auto& ld) {
        return recursive ? lapack::Potrf2(uplo, n, p[0], ld[0])
                         : lapack::Potrf(uplo, n, p[0], ld[0]);
      });
}

template <typename T>
lapack_int Potrf(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda, bool recursive) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>(recursive ? "potrf2" : "potrf"), -1);
    return -1;
  }
  if (NanCheck() && PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
  return PotrfWork(matrix_layout, uplo, n, a, lda, recursive);
}

// ---------------------------------------------------------------------------
// QR and LQ factorizations

// xGEQRF (lq = false) and xGELQF (lq = true).
template <typename T>
lapack_int GeqrfWork(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, T* tau, T* work, lapack_int lwork,
                     bool lq) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(lq ? "gelqf_work" : "geqrf_work"), lwork == -1,
      {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
        return lq ? lapack::Gelqf(m, n, p[0], ld[0], tau, work, lwork)
                  : lapack::Geqrf(m, n, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Geqrf(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, T* tau, bool lq) {
  const std::string name = Name<T>(lq ? "gelqf" : "geqrf");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GeqrfWork(matrix_layout, m, n, a, lda, tau, work, lwork, lq);
  });
}

// xGEQR2 (lq = false) and xGELQ2 (lq = true).
template <typename T>
lapack_int Geqr2Work(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, T* tau, T* work, bool lq) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(lq ? "gelq2_work" : "geqr2_work"), false,
      {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
        return lq ? lapack::Gelq2(m, n, p[0], ld[0], tau, work)
                  : lapack::Geqr2(m, n, p[0], ld[0], tau, work);
      });
}

template <typename T>
lapack_int Geqr2(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, T* tau, bool lq) {
  const std::string name = Name<T>(lq ? "gelq2" : "geqr2");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return WithWorkArray<T>(
      name, std::max<lapack_int>(1, lq ? m : n), [&](T* work) {
        return Geqr2Work(matrix_layout, m, n, a, lda, tau, work, lq);
      });
}

// ---------------------------------------------------------------------------
// Orthogonal / unitary matrices of QR and LQ factorizations

// xORGQR / xUNGQR (lq = false) and xORGLQ / xUNGLQ (lq = true).
template <typename T>
const char* UngqrName(bool lq, bool work) {
  if (lq) {
    return work ? RealOrComplex<T>("orglq_work", "unglq_work")
                : RealOrComplex<T>("orglq", "unglq");
  }
  return work ? RealOrComplex<T>("orgqr_work", "ungqr_work")
              : RealOrComplex<T>("orgqr", "ungqr");
}

template <typename T>
lapack_int UngqrWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int k, T* a, lapack_int lda, const T* tau, T* work,
                     lapack_int lwork, bool lq) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(UngqrName<T>(lq, true)), lwork == -1,
      {{{a, m, n, lda, -6}}}, [&](auto& p, auto& ld) {
        return lq ? lapack::Unglq(m, n, k, p[0], ld[0], tau, work, lwork)
                  : lapack::Ungqr(m, n, k, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Ungqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                 T* a, lapack_int lda, const T* tau, bool lq) {
  const std::string name = Name<T>(UngqrName<T>(lq, false));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -5;
    if (VectorNancheck(k, tau, 1)) return -7;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UngqrWork(matrix_layout, m, n, k, a, lda, tau, work, lwork, lq);
  });
}

// xORMQR / xUNMQR (lq = false) and xORMLQ / xUNMLQ (lq = true). A is r x k
// (QR) or k x r (LQ), with r = m for side = 'L' and r = n for side = 'R'.
//
// Deviation: for complex T, the reference LAPACKE (3.12.1) checks and
// copies the k x m part of A for LQ also for side = 'R'; the template uses
// k x r, as the reference does for real T.
template <typename T>
const char* UnmqrName(bool lq, bool work) {
  if (lq) {
    return work ? RealOrComplex<T>("ormlq_work", "unmlq_work")
                : RealOrComplex<T>("ormlq", "unmlq");
  }
  return work ? RealOrComplex<T>("ormqr_work", "unmqr_work")
              : RealOrComplex<T>("ormqr", "unmqr");
}

template <typename T>
lapack_int UnmqrWork(int matrix_layout, char side, char trans, lapack_int m,
                     lapack_int n, lapack_int k, const T* a, lapack_int lda,
                     const T* tau, T* c, lapack_int ldc, T* work,
                     lapack_int lwork, bool lq) {
  const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
  return CallWork<T, 2>(
      matrix_layout, Name<T>(UnmqrName<T>(lq, true)), lwork == -1,
      {{lq ? Input(a, k, r, lda, -8) : Input(a, r, k, lda, -8),
        {c, m, n, ldc, -11}}},
      [&](auto& p, auto& ld) {
        // The complex LQ routines conjugate rows of A while they run, and
        // restore them: the routine changes A, but not the caller's A.
        return lq ? lapack::Unmlq(side, trans, m, n, k, p[0], ld[0], tau, p[1],
                                  ld[1], work, lwork)
                  : lapack::Unmqr(side, trans, m, n, k, p[0], ld[0], tau, p[1],
                                  ld[1], work, lwork);
      });
}

template <typename T>
lapack_int Unmqr(int matrix_layout, char side, char trans, lapack_int m,
                 lapack_int n, lapack_int k, const T* a, lapack_int lda,
                 const T* tau, T* c, lapack_int ldc, bool lq) {
  const std::string name = Name<T>(UnmqrName<T>(lq, false));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
    if (lq ? GeNancheck(matrix_layout, k, r, a, lda)
           : GeNancheck(matrix_layout, r, k, a, lda)) {
      return -7;
    }
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -10;
    if (VectorNancheck(k, tau, 1)) return -9;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnmqrWork(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc,
                     work, lwork, lq);
  });
}

// ---------------------------------------------------------------------------
// Auxiliary routines

// Deviation: in row-major layout, the reference LAPACKE (3.12.1) copies B
// back without copying it in first, so that the elements of B outside the
// triangle uplo become undefined; here they are kept.
template <typename T>
lapack_int LacpyWork(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                     const T* a, lapack_int lda, T* b, lapack_int ldb) {
  return CallWork<T, 2>(matrix_layout, Name<T>("lacpy_work"), false,
                        {{Input(a, m, n, lda, -6), {b, m, n, ldb, -8}}},
                        [&](auto& p, auto& ld) {
                          lapack::Lacpy(uplo, m, n, p[0], ld[0], p[1], ld[1]);
                          return lapack_int{0};
                        });
}

template <typename T>
lapack_int Lacpy(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                 const T* a, lapack_int lda, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("lacpy"), -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -5;
  return LacpyWork(matrix_layout, uplo, m, n, a, lda, b, ldb);
}

template <typename T>
lapack_int Larfg(lapack_int n, T* alpha, T* x, lapack_int incx, T* tau) {
  if (NanCheck()) {
    if (VectorNancheck(1, alpha, 1)) return -2;
    if (VectorNancheck(n - 1, x, incx)) return -3;
  }
  lapack::Larfg(n, alpha, x, incx, tau);
  return 0;
}

// The dimensions of V of xLARFT: n x k for STOREV = 'C', k x n for 'R'
// (1 x 1 otherwise).
inline std::array<lapack_int, 2> LarftVDims(char storev, lapack_int n,
                                            lapack_int k) {
  if (lapack::Lsame(storev, 'C')) return {n, k};
  if (lapack::Lsame(storev, 'R')) return {k, n};
  return {1, 1};
}

template <typename T>
lapack_int LarftWork(int matrix_layout, char direct, char storev, lapack_int n,
                     lapack_int k, const T* v, lapack_int ldv, const T* tau,
                     T* t, lapack_int ldt) {
  const auto [nrows_v, ncols_v] = LarftVDims(storev, n, k);
  MatrixArg<T> t_arg{t, k, k, ldt, -10};
  t_arg.in = false;
  return CallWork<T, 2>(
      matrix_layout, Name<T>("larft_work"), false,
      {{t_arg, Input(v, nrows_v, ncols_v, ldv, -7)}}, [&](auto& p, auto& ld) {
        lapack::Larft(direct, storev, n, k, p[1], ld[1], tau, p[0], ld[0]);
        return lapack_int{0};
      });
}

template <typename T>
lapack_int Larft(int matrix_layout, char direct, char storev, lapack_int n,
                 lapack_int k, const T* v, lapack_int ldv, const T* tau, T* t,
                 lapack_int ldt) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("larft"), -1);
    return -1;
  }
  if (NanCheck()) {
    const auto [nrows_v, ncols_v] = LarftVDims(storev, n, k);
    if (VectorNancheck(k, tau, 1)) return -8;
    if (GeNancheck(matrix_layout, nrows_v, ncols_v, v, ldv)) return -6;
  }
  return LarftWork(matrix_layout, direct, storev, n, k, v, ldv, tau, t, ldt);
}

// The dimensions of V of xLARFB, and the triangle of its unit triangular
// part.
struct LarfbV {
  lapack_int rows;
  lapack_int cols;
  char uplo;
  bool k_too_large;  // K exceeds the order of the triangle: INFO = -8
};

inline LarfbV LarfbVShape(char side, char direct, char storev, lapack_int m,
                          lapack_int n, lapack_int k) {
  const bool left = lapack::Lsame(side, 'L');
  const bool col = lapack::Lsame(storev, 'C');
  const bool forward = lapack::Lsame(direct, 'F');
  LarfbV v{};
  v.rows = (col && left) ? m : ((col && !left) ? n : (!col ? k : 1));
  v.cols = (!col && left) ? m : ((!col && !left) ? n : (col ? k : 1));
  v.uplo = ((forward && col) || !(forward || col)) ? 'L' : 'U';
  v.k_too_large = (col && k > v.rows) || (!col && k > v.cols);
  return v;
}

// V is copied in full (the reference copies only its trapezoidal part,
// LAPACKE_xtz_trans; xLARFB reads no other element).
template <typename T>
lapack_int LarfbWork(int matrix_layout, char side, char trans, char direct,
                     char storev, lapack_int m, lapack_int n, lapack_int k,
                     const T* v, lapack_int ldv, const T* t, lapack_int ldt,
                     T* c, lapack_int ldc, T* work, lapack_int ldwork) {
  const std::string name = Name<T>("larfb_work");
  const LarfbV shape = LarfbVShape(side, direct, storev, m, n, k);
  // In row-major layout, the reference checks K after the leading
  // dimensions (which CallWork checks).
  if (matrix_layout == LAPACK_ROW_MAJOR && ldc >= n && ldt >= k &&
      ldv >= shape.cols && shape.k_too_large) {
    Xerbla(name, -8);
    return -8;
  }
  return CallWork<T, 3>(matrix_layout, name, false,
                        {{{c, m, n, ldc, -14},
                          Input(t, k, k, ldt, -12),
                          Input(v, shape.rows, shape.cols, ldv, -10)}},
                        [&](auto& p, auto& ld) {
                          lapack::Larfb(side, trans, direct, storev, m, n, k,
                                        p[2], ld[2], p[1], ld[1], p[0], ld[0],
                                        work, ldwork);
                          return lapack_int{0};
                        });
}

template <typename T>
lapack_int Larfb(int matrix_layout, char side, char trans, char direct,
                 char storev, lapack_int m, lapack_int n, lapack_int k,
                 const T* v, lapack_int ldv, const T* t, lapack_int ldt, T* c,
                 lapack_int ldc) {
  const std::string name = Name<T>("larfb");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const LarfbV shape = LarfbVShape(side, direct, storev, m, n, k);
    if (shape.k_too_large) {
      Xerbla(name, -8);
      return -8;
    }
    if (TzNancheck(matrix_layout, direct, shape.uplo, 'U', shape.rows,
                   shape.cols, v, ldv)) {
      return -9;
    }
    if (GeNancheck(matrix_layout, k, k, t, ldt)) return -11;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -13;
  }
  lapack_int ldwork = 1;
  if (lapack::Lsame(side, 'L')) {
    ldwork = n;
  } else if (lapack::Lsame(side, 'R')) {
    ldwork = m;
  }
  return WithWorkArray<T>(
      name, ldwork * std::max<lapack_int>(1, k), [&](T* work) {
        return LarfbWork(matrix_layout, side, trans, direct, storev, m, n, k, v,
                         ldv, t, ldt, c, ldc, work, ldwork);
      });
}

template <typename T>
lapack_int Lacgv(lapack_int n, T* x, lapack_int incx) {
  if (NanCheck() && VectorNancheck(n, x, incx)) return -2;
  lapack::Lacgv(n, x, incx);
  return 0;
}

// NaN arguments give minus their position (as a floating-point value).
template <typename R>
R Lapy2(R x, R y) {
  if (NanCheck()) {
    if (std::isnan(x)) return R(-1);
    if (std::isnan(y)) return R(-2);
  }
  return lapack::Lapy2(x, y);
}

template <typename R>
R Lapy3(R x, R y, R z) {
  if (NanCheck()) {
    if (std::isnan(x)) return R(-1);
    if (std::isnan(y)) return R(-2);
    if (std::isnan(z)) return R(-3);
  }
  return lapack::Lapy3(x, y, z);
}

// ---------------------------------------------------------------------------
// QR factorization with column pivoting

// rwork (2 * n elements) for complex T only.
template <typename T>
lapack_int Geqp3Work(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, lapack_int* jpvt, T* tau, T* work,
                     lapack_int lwork, RealT<T>* rwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("geqp3_work"), lwork == -1,
                        {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
                          return lapack::Geqp3(m, n, p[0], ld[0], jpvt, tau,
                                               work, lwork, rwork);
                        });
}

template <typename T>
lapack_int Geqp3(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, lapack_int* jpvt, T* tau) {
  using R = RealT<T>;
  const std::string name = Name<T>("geqp3");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  // Allocate memory for working array(s).
  R* rwork = nullptr;
  if constexpr (IsComplex<T>::value) {
    rwork = static_cast<R*>(malloc(
        sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, 2 * n))));
    if (rwork == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return LAPACK_WORK_MEMORY_ERROR;
    }
  }
  const lapack_int info =
      WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
        return Geqp3Work(matrix_layout, m, n, a, lda, jpvt, tau, work, lwork,
                         rwork);
      });
  free(rwork);
  return info;
}

// ---------------------------------------------------------------------------
// Norms

// xLANGE: in row-major layout, the norm of the transpose (the same array
// in column-major layout) with the one and infinity norms exchanged; no
// copy.
template <typename T>
RealT<T> LangeWork(int matrix_layout, char norm, lapack_int m, lapack_int n,
                   const T* a, lapack_int lda, RealT<T>* work) {
  using R = RealT<T>;
  if (matrix_layout == LAPACK_COL_MAJOR) {
    return lapack::Lange(norm, m, n, a, lda, work);
  }
  const std::string name = Name<T>("lange_work");
  if (matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(name, -1);
    return R(-1);
  }
  // Check leading dimension(s).
  if (lda < n) {
    Xerbla(name, -6);
    return R(-6);
  }
  char norm_lapack = norm;
  if (lapack::Lsame(norm, '1') || lapack::Lsame(norm, 'O')) {
    norm_lapack = 'I';
  } else if (lapack::Lsame(norm, 'I')) {
    norm_lapack = '1';
  }
  // Allocate memory for work array(s).
  R* work_lapack = nullptr;
  if (lapack::Lsame(norm_lapack, 'I')) {
    work_lapack = static_cast<R*>(
        malloc(sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, n))));
    if (work_lapack == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return R(0);
    }
  }
  const R res = lapack::Lange(norm_lapack, n, m, a, lda, work_lapack);
  free(work_lapack);
  return res;
}

template <typename T>
RealT<T> Lange(int matrix_layout, char norm, lapack_int m, lapack_int n,
               const T* a, lapack_int lda) {
  using R = RealT<T>;
  const std::string name = Name<T>("lange");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return R(-1);
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return R(-5);
  // Allocate memory for working array(s).
  R* work = nullptr;
  if (lapack::Lsame(norm, 'I')) {
    work = static_cast<R*>(
        malloc(sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, m))));
    if (work == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return R(0);
    }
  }
  const R res = LangeWork(matrix_layout, norm, m, n, a, lda, work);
  free(work);
  return res;
}

// xLANSY (hermitian = false) and xLANHE (hermitian = true): in row-major
// layout, the triangle uplo is copied into a column-major temporary.
template <typename T>
RealT<T> LansyWork(int matrix_layout, char norm, char uplo, lapack_int n,
                   const T* a, lapack_int lda, RealT<T>* work, bool hermitian) {
  using R = RealT<T>;
  const auto call = [&](const T* x, lapack_int ldx) {
    if constexpr (IsComplex<T>::value) {
      if (hermitian) return lapack::Lanhe(norm, uplo, n, x, ldx, work);
    }
    return lapack::Lansy(norm, uplo, n, x, ldx, work);
  };
  if (matrix_layout == LAPACK_COL_MAJOR) return call(a, lda);
  const std::string name = Name<T>(hermitian ? "lanhe_work" : "lansy_work");
  if (matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(name, -1);
    return R(-1);
  }
  // Check leading dimension(s).
  if (lda < n) {
    Xerbla(name, -6);
    return R(-6);
  }
  // Allocate memory for temporary array(s).
  const lapack_int lda_t = std::max<lapack_int>(1, n);
  T* a_t = static_cast<T*>(malloc(sizeof(T) * static_cast<size_t>(lda_t) *
                                  std::max<lapack_int>(1, n)));
  if (a_t == nullptr) {
    Xerbla(name, LAPACK_TRANSPOSE_MEMORY_ERROR);
    return R(LAPACK_TRANSPOSE_MEMORY_ERROR);
  }
  // Transpose input matrices.
  TrTrans(matrix_layout, uplo, 'N', n, a, lda, a_t, lda_t);
  const R res = call(a_t, lda_t);
  free(a_t);
  return res;
}

template <typename T>
RealT<T> Lansy(int matrix_layout, char norm, char uplo, lapack_int n,
               const T* a, lapack_int lda, bool hermitian) {
  using R = RealT<T>;
  const std::string name = Name<T>(hermitian ? "lanhe" : "lansy");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return R(-1);
  }
  if (NanCheck() && TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) {
    return R(-5);
  }
  // Allocate memory for working array(s).
  R* work = nullptr;
  const bool needs_work = lapack::Lsame(norm, 'I') ||
                          lapack::Lsame(norm, '1') || lapack::Lsame(norm, 'O');
  if (needs_work) {
    work = static_cast<R*>(
        malloc(sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, n))));
    if (work == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return R(0);
    }
  }
  const R res =
      LansyWork(matrix_layout, norm, uplo, n, a, lda, work, hermitian);
  free(work);
  return res;
}

// ---------------------------------------------------------------------------
// Scaling and setting

// The rows of A in its storage for xLASCL's type: band storage for 'B',
// 'Q' and 'Z'.
lapack_int LasclRows(char type, lapack_int kl, lapack_int ku, lapack_int m) {
  return lapack::Lsame(type, 'B')   ? kl + 1
         : lapack::Lsame(type, 'Q') ? ku + 1
         : lapack::Lsame(type, 'Z') ? 2 * kl + ku + 1
                                    : m;
}

// Whether the elements of A that xLASCL scales contain a NaN.
//
// The reference LAPACKE (3.12.1) checks these elements with its band
// matrix checks (and, like it, only for upper-case type letters); this
// checks the same elements directly.
template <typename T>
bool LasclNancheck(int matrix_layout, char type, lapack_int kl, lapack_int ku,
                   lapack_int m, lapack_int n, const T* a, lapack_int lda) {
  // Element (i, j) (1-based) of the storage array.
  const auto at = [&](lapack_int i, lapack_int j) {
    return matrix_layout == LAPACK_COL_MAJOR
               ? a[(i - 1) + static_cast<size_t>(j - 1) * lda]
               : a[static_cast<size_t>(i - 1) * lda + (j - 1)];
  };
  // Rows first(j) .. last(j) of each column j.
  const auto check = [&](auto first, auto last) {
    for (lapack_int j = 1; j <= n; ++j) {
      for (lapack_int i = first(j); i <= last(j); ++i) {
        if (IsNan(at(i, j))) return true;
      }
    }
    return false;
  };
  switch (type) {
    case 'G':
      return GeNancheck(matrix_layout, m, n, a, lda);
    case 'L':
      return check([](lapack_int j) { return j; },
                   [&](lapack_int) { return m; });
    case 'U':
      return check([](lapack_int) { return lapack_int{1}; },
                   [&](lapack_int j) { return std::min(j, m); });
    case 'H':
      return check([](lapack_int) { return lapack_int{1}; },
                   [&](lapack_int j) { return std::min(j + 1, m); });
    case 'B':
      return check([](lapack_int) { return lapack_int{1}; },
                   [&](lapack_int j) { return std::min(kl + 1, n + 1 - j); });
    case 'Q':
      return check(
          [&](lapack_int j) { return std::max<lapack_int>(ku + 2 - j, 1); },
          [&](lapack_int) { return ku + 1; });
    case 'Z':
      return check(
          [&](lapack_int j) { return std::max(kl + ku + 2 - j, kl + 1); },
          [&](lapack_int j) {
            return std::min(2 * kl + ku + 1, kl + ku + 1 + m - j);
          });
    default:
      return false;
  }
}

template <typename T>
lapack_int LasclWork(int matrix_layout, char type, lapack_int kl, lapack_int ku,
                     RealT<T> cfrom, RealT<T> cto, lapack_int m, lapack_int n,
                     T* a, lapack_int lda) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>("lascl_work"), false,
      {{{a, LasclRows(type, kl, ku, m), n, lda, -9}}}, [&](auto& p, auto& ld) {
        return lapack::Lascl(type, kl, ku, cfrom, cto, m, n, p[0], ld[0]);
      });
}

template <typename T>
lapack_int Lascl(int matrix_layout, char type, lapack_int kl, lapack_int ku,
                 RealT<T> cfrom, RealT<T> cto, lapack_int m, lapack_int n, T* a,
                 lapack_int lda) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("lascl"), -1);
    return -1;
  }
  if (NanCheck() && LasclNancheck(matrix_layout, type, kl, ku, m, n, a, lda)) {
    return -9;
  }
  return LasclWork(matrix_layout, type, kl, ku, cfrom, cto, m, n, a, lda);
}

template <typename T>
lapack_int LasetWork(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                     T alpha, T beta, T* a, lapack_int lda) {
  return CallWork<T, 1>(matrix_layout, Name<T>("laset_work"), false,
                        {{{a, m, n, lda, -8}}}, [&](auto& p, auto& ld) {
                          lapack::Laset(uplo, m, n, alpha, beta, p[0], ld[0]);
                          return lapack_int{0};
                        });
}

template <typename T>
lapack_int Laset(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                 T alpha, T beta, T* a, lapack_int lda) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("laset"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (IsNan(alpha)) return -5;
    if (IsNan(beta)) return -6;
  }
  return LasetWork(matrix_layout, uplo, m, n, alpha, beta, a, lda);
}

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) eigenproblems

// xSYEV / xHEEV; rwork (max(1, 3*n-2) elements) for complex T only. In
// row-major layout, the triangle uplo of A is copied in, and back, or all
// of A (the eigenvectors) for jobz = 'V'.
template <typename T>
lapack_int HeevWork(int matrix_layout, char jobz, char uplo, lapack_int n, T* a,
                    lapack_int lda, RealT<T>* w, T* work, lapack_int lwork,
                    RealT<T>* rwork) {
  MatrixArg<T> arg{a, n, n, lda, -6};
  arg.uplo = lapack::Lsame(uplo, 'U') ? 'U' : 'L';
  arg.out_general = lapack::Lsame(jobz, 'V');
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("syev_work", "heev_work")),
      lwork == -1, {{arg}}, [&](auto& p, auto& ld) {
        return lapack::Heev(jobz, uplo, n, p[0], ld[0], w, work, lwork, rwork);
      });
}

template <typename T>
lapack_int Heev(int matrix_layout, char jobz, char uplo, lapack_int n, T* a,
                lapack_int lda, RealT<T>* w) {
  using R = RealT<T>;
  const std::string name = Name<T>(RealOrComplex<T>("syev", "heev"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) {
    return -5;
  }
  // Allocate memory for working array(s).
  R* rwork = nullptr;
  if constexpr (IsComplex<T>::value) {
    rwork = static_cast<R*>(malloc(
        sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, 3 * n - 2))));
    if (rwork == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return LAPACK_WORK_MEMORY_ERROR;
    }
  }
  const lapack_int info =
      WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
        return HeevWork(matrix_layout, jobz, uplo, n, a, lda, w, work, lwork,
                        rwork);
      });
  free(rwork);
  return info;
}

// xSYTRD / xHETRD.
template <typename T>
lapack_int HetrdWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda, RealT<T>* d, RealT<T>* e, T* tau, T* work,
                     lapack_int lwork) {
  MatrixArg<T> arg{a, n, n, lda, -5};
  arg.uplo = lapack::Lsame(uplo, 'U') ? 'U' : 'L';
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("sytrd_work", "hetrd_work")),
      lwork == -1, {{arg}}, [&](auto& p, auto& ld) {
        return lapack::Hetrd(uplo, n, p[0], ld[0], d, e, tau, work, lwork);
      });
}

template <typename T>
lapack_int Hetrd(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda, RealT<T>* d, RealT<T>* e, T* tau) {
  const std::string name = Name<T>(RealOrComplex<T>("sytrd", "hetrd"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) {
    return -4;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return HetrdWork(matrix_layout, uplo, n, a, lda, d, e, tau, work, lwork);
  });
}

// xORGTR / xUNGTR.
template <typename T>
lapack_int UngtrWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda, const T* tau, T* work, lapack_int lwork) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("orgtr_work", "ungtr_work")),
      lwork == -1, {{{a, n, n, lda, -5}}}, [&](auto& p, auto& ld) {
        return lapack::Ungtr(uplo, n, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Ungtr(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda, const T* tau) {
  const std::string name = Name<T>(RealOrComplex<T>("orgtr", "ungtr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) return -4;
    if (VectorNancheck(n - 1, tau, 1)) return -6;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UngtrWork(matrix_layout, uplo, n, a, lda, tau, work, lwork);
  });
}

// xORGQL / xUNGQL.
template <typename T>
lapack_int UngqlWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int k, T* a, lapack_int lda, const T* tau, T* work,
                     lapack_int lwork) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("orgql_work", "ungql_work")),
      lwork == -1, {{{a, m, n, lda, -6}}}, [&](auto& p, auto& ld) {
        return lapack::Ungql(m, n, k, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Ungql(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                 T* a, lapack_int lda, const T* tau) {
  const std::string name = Name<T>(RealOrComplex<T>("orgql", "ungql"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -5;
    if (VectorNancheck(k, tau, 1)) return -7;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UngqlWork(matrix_layout, m, n, k, a, lda, tau, work, lwork);
  });
}

// xSTEQR: in row-major layout, Z is copied in for compz = 'V' and back for
// 'V' and 'I'; for 'N', it gets no temporary.
template <typename T>
lapack_int SteqrWork(int matrix_layout, char compz, lapack_int n, RealT<T>* d,
                     RealT<T>* e, T* z, lapack_int ldz, RealT<T>* work) {
  MatrixArg<T> arg{z, n, n, ldz, -7};
  arg.in = lapack::Lsame(compz, 'V');
  arg.used = arg.in || lapack::Lsame(compz, 'I');
  arg.out = arg.used;
  return CallWork<T, 1>(matrix_layout, Name<T>("steqr_work"), false, {{arg}},
                        [&](auto& p, auto& ld) {
                          return lapack::Steqr(compz, n, d, e, p[0], ld[0],
                                               work);
                        });
}

template <typename T>
lapack_int Steqr(int matrix_layout, char compz, lapack_int n, RealT<T>* d,
                 RealT<T>* e, T* z, lapack_int ldz) {
  using R = RealT<T>;
  const std::string name = Name<T>("steqr");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -4;
    if (VectorNancheck(n - 1, e, 1)) return -5;
    if (lapack::Lsame(compz, 'V') && GeNancheck(matrix_layout, n, n, z, ldz)) {
      return -6;
    }
  }
  // Allocate memory for working array(s).
  const lapack_int lwork =
      lapack::Lsame(compz, 'N') ? 1 : std::max<lapack_int>(1, 2 * n - 2);
  R* work = static_cast<R*>(malloc(sizeof(R) * static_cast<size_t>(lwork)));
  if (work == nullptr) {
    Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
    return LAPACK_WORK_MEMORY_ERROR;
  }
  const lapack_int info =
      SteqrWork(matrix_layout, compz, n, d, e, z, ldz, work);
  free(work);
  return info;
}

// xSTERF and xLASRT (real only; no matrix_layout).
template <typename R>
lapack_int Sterf(lapack_int n, R* d, R* e) {
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -2;
    if (VectorNancheck(n - 1, e, 1)) return -3;
  }
  return lapack::Sterf(n, d, e);
}

template <typename R>
lapack_int Lasrt(char id, lapack_int n, R* d) {
  if (NanCheck() && VectorNancheck(n, d, 1)) return -3;
  return lapack::Lasrt(id, n, d);
}

// ---------------------------------------------------------------------------
// Singular value decomposition

// xGESVD; rwork (5*min(m,n) elements) for complex T only. In row-major
// layout, U is copied back for jobu = 'A' or 'S' (m x m or m x min(m,n))
// and VT for jobvt = 'A' or 'S' (n x n or min(m,n) x n); otherwise they get
// no temporary.
template <typename T>
lapack_int GesvdWork(int matrix_layout, char jobu, char jobvt, lapack_int m,
                     lapack_int n, T* a, lapack_int lda, RealT<T>* s, T* u,
                     lapack_int ldu, T* vt, lapack_int ldvt, T* work,
                     lapack_int lwork, RealT<T>* rwork) {
  const lapack_int minmn = std::min(m, n);
  const bool u_all = lapack::Lsame(jobu, 'A');
  const bool u_some = lapack::Lsame(jobu, 'S');
  const bool vt_all = lapack::Lsame(jobvt, 'A');
  const bool vt_some = lapack::Lsame(jobvt, 'S');
  const lapack_int nrows_u = (u_all || u_some) ? m : 1;
  const lapack_int ncols_u = u_all ? m : (u_some ? minmn : 1);
  const lapack_int nrows_vt = vt_all ? n : (vt_some ? minmn : 1);
  const lapack_int ncols_vt = (vt_all || vt_some) ? n : 1;
  MatrixArg<T> u_arg{u, nrows_u, ncols_u, ldu, -10};
  u_arg.in = false;
  u_arg.used = u_all || u_some;
  MatrixArg<T> vt_arg{vt, nrows_vt, ncols_vt, ldvt, -12};
  vt_arg.in = false;
  vt_arg.used = vt_all || vt_some;
  return CallWork<T, 3>(
      matrix_layout, Name<T>("gesvd_work"), lwork == -1,
      {{{a, m, n, lda, -7}, u_arg, vt_arg}}, [&](auto& p, auto& ld) {
        return lapack::Gesvd(jobu, jobvt, m, n, p[0], ld[0], s, p[1], ld[1],
                             p[2], ld[2], work, lwork, rwork);
      });
}

// xGESVD: superb (min(m,n)-1 elements) returns the unconverged
// superdiagonal elements of the bidiagonal matrix (info > 0), from WORK(2)
// (real T) or RWORK(1) (complex T) of the routine.
template <typename T>
lapack_int Gesvd(int matrix_layout, char jobu, char jobvt, lapack_int m,
                 lapack_int n, T* a, lapack_int lda, RealT<T>* s, T* u,
                 lapack_int ldu, T* vt, lapack_int ldvt, RealT<T>* superb) {
  using R = RealT<T>;
  const std::string name = Name<T>("gesvd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -6;
  const lapack_int minmn = std::min(m, n);
  // Allocate memory for working array(s).
  R* rwork = nullptr;
  if constexpr (IsComplex<T>::value) {
    rwork = static_cast<R*>(malloc(
        sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, 5 * minmn))));
    if (rwork == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return LAPACK_WORK_MEMORY_ERROR;
    }
  }
  // Query optimal working array(s) size.
  T work_query = T(0);
  lapack_int info = GesvdWork(matrix_layout, jobu, jobvt, m, n, a, lda, s, u,
                              ldu, vt, ldvt, &work_query, -1, rwork);
  if (info == 0) {
    const lapack_int lwork = static_cast<lapack_int>(std::real(work_query));
    // Allocate memory for work arrays.
    T* work = static_cast<T*>(malloc(sizeof(T) * static_cast<size_t>(lwork)));
    if (work == nullptr) {
      info = LAPACK_WORK_MEMORY_ERROR;
      Xerbla(name, info);
    } else {
      // Call middle-level interface.
      info = GesvdWork(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                       ldvt, work, lwork, rwork);
      // Backup significant data from working array(s).
      for (lapack_int i = 0; i < minmn - 1; ++i) {
        // RWORK has 5 * min(m, n) elements, and LWORK >= 5 * min(m, n) (the
        // minimum of the real routine).
        if constexpr (IsComplex<T>::value) {
          // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
          superb[i] = rwork[i];
        } else {
          // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
          superb[i] = work[i + 1];
        }
      }
      free(work);
    }
  }
  free(rwork);
  return info;
}

// xGEBRD.
template <typename T>
lapack_int GebrdWork(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, RealT<T>* d, RealT<T>* e, T* tauq, T* taup,
                     T* work, lapack_int lwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gebrd_work"), lwork == -1,
                        {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
                          return lapack::Gebrd(m, n, p[0], ld[0], d, e, tauq,
                                               taup, work, lwork);
                        });
}

template <typename T>
lapack_int Gebrd(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, RealT<T>* d, RealT<T>* e, T* tauq, T* taup) {
  const std::string name = Name<T>("gebrd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GebrdWork(matrix_layout, m, n, a, lda, d, e, tauq, taup, work,
                     lwork);
  });
}

// xORGBR / xUNGBR.
template <typename T>
lapack_int UngbrWork(int matrix_layout, char vect, lapack_int m, lapack_int n,
                     lapack_int k, T* a, lapack_int lda, const T* tau, T* work,
                     lapack_int lwork) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("orgbr_work", "ungbr_work")),
      lwork == -1, {{{a, m, n, lda, -7}}}, [&](auto& p, auto& ld) {
        return lapack::Ungbr(vect, m, n, k, p[0], ld[0], tau, work, lwork);
      });
}

// As in the reference, the NaN check of tau covers min(m, k) elements, also
// for vect = 'P' (where xORGBR / xUNGBR uses min(n, k)).
template <typename T>
lapack_int Ungbr(int matrix_layout, char vect, lapack_int m, lapack_int n,
                 lapack_int k, T* a, lapack_int lda, const T* tau) {
  const std::string name = Name<T>(RealOrComplex<T>("orgbr", "ungbr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -6;
    if (VectorNancheck(std::min(m, k), tau, 1)) return -8;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UngbrWork(matrix_layout, vect, m, n, k, a, lda, tau, work, lwork);
  });
}

// xORMBR / xUNMBR. A is nq x min(nq,k) (vect = 'Q') or min(nq,k) x nq
// ('P'), with nq = m for side = 'L' and nq = n for side = 'R'.
//
// Deviation: for complex T, the reference LAPACKE (3.12.1) checks and
// copies the r x min(nq,k) part of A (r = nq for 'Q', min(nq,k) for 'P'),
// which for vect = 'P' and nq > k misses the columns min(nq,k)+1:nq of the
// reflectors; the template uses the dimensions above, as the reference does
// for real T.
template <typename T>
lapack_int UnmbrWork(int matrix_layout, char vect, char side, char trans,
                     lapack_int m, lapack_int n, lapack_int k, const T* a,
                     lapack_int lda, const T* tau, T* c, lapack_int ldc,
                     T* work, lapack_int lwork) {
  const lapack_int nq = lapack::Lsame(side, 'L') ? m : n;
  const bool applyq = lapack::Lsame(vect, 'Q');
  const lapack_int ar = applyq ? nq : std::min(nq, k);
  const lapack_int ac = applyq ? std::min(nq, k) : nq;
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("ormbr_work", "unmbr_work")),
      lwork == -1, {{Input(a, ar, ac, lda, -9), {c, m, n, ldc, -12}}},
      [&](auto& p, auto& ld) {
        // The complex routine conjugates rows of A for vect = 'P' while it
        // runs, and restores them: the routine changes A, but not the
        // caller's A.
        return lapack::Unmbr(vect, side, trans, m, n, k, p[0], ld[0], tau, p[1],
                             ld[1], work, lwork);
      });
}

template <typename T>
lapack_int Unmbr(int matrix_layout, char vect, char side, char trans,
                 lapack_int m, lapack_int n, lapack_int k, const T* a,
                 lapack_int lda, const T* tau, T* c, lapack_int ldc) {
  const std::string name = Name<T>(RealOrComplex<T>("ormbr", "unmbr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int nq = lapack::Lsame(side, 'L') ? m : n;
    const bool applyq = lapack::Lsame(vect, 'Q');
    const lapack_int ar = applyq ? nq : std::min(nq, k);
    const lapack_int ac = applyq ? std::min(nq, k) : nq;
    if (GeNancheck(matrix_layout, ar, ac, a, lda)) return -8;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -11;
    if (VectorNancheck(std::min(nq, k), tau, 1)) return -10;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnmbrWork(matrix_layout, vect, side, trans, m, n, k, a, lda, tau, c,
                     ldc, work, lwork);
  });
}

// xBDSQR: work (rwork for complex T) has 4*n elements. In row-major layout,
// VT, U and C get temporaries if ncvt, nru and ncc are not 0.
template <typename T>
lapack_int BdsqrWork(int matrix_layout, char uplo, lapack_int n,
                     lapack_int ncvt, lapack_int nru, lapack_int ncc,
                     RealT<T>* d, RealT<T>* e, T* vt, lapack_int ldvt, T* u,
                     lapack_int ldu, T* c, lapack_int ldc, RealT<T>* work) {
  // The leading dimensions are checked in the order C, U, VT.
  MatrixArg<T> c_arg{c, n, ncc, ldc, -14};
  c_arg.used = ncc != 0;
  MatrixArg<T> u_arg{u, nru, n, ldu, -12};
  u_arg.used = nru != 0;
  MatrixArg<T> vt_arg{vt, n, ncvt, ldvt, -10};
  vt_arg.used = ncvt != 0;
  return CallWork<T, 3>(matrix_layout, Name<T>("bdsqr_work"), false,
                        {{c_arg, u_arg, vt_arg}}, [&](auto& p, auto& ld) {
                          return lapack::Bdsqr(uplo, n, ncvt, nru, ncc, d, e,
                                               p[2], ld[2], p[1], ld[1], p[0],
                                               ld[0], work);
                        });
}

template <typename T>
lapack_int Bdsqr(int matrix_layout, char uplo, lapack_int n, lapack_int ncvt,
                 lapack_int nru, lapack_int ncc, RealT<T>* d, RealT<T>* e,
                 T* vt, lapack_int ldvt, T* u, lapack_int ldu, T* c,
                 lapack_int ldc) {
  using R = RealT<T>;
  const std::string name = Name<T>("bdsqr");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (ncc != 0 && GeNancheck(matrix_layout, n, ncc, c, ldc)) return -13;
    if (VectorNancheck(n, d, 1)) return -7;
    if (VectorNancheck(n - 1, e, 1)) return -8;
    if (nru != 0 && GeNancheck(matrix_layout, nru, n, u, ldu)) return -11;
    if (ncvt != 0 && GeNancheck(matrix_layout, n, ncvt, vt, ldvt)) return -9;
  }
  return WithWorkArray<R>(name, std::max<lapack_int>(1, 4 * n), [&](R* work) {
    return BdsqrWork(matrix_layout, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u,
                     ldu, c, ldc, work);
  });
}

// ---------------------------------------------------------------------------
// Singular value decomposition by divide and conquer

// xGESDD. As in the reference, the row-major layout requires ldvt >= n
// whether or not VT is referenced.
template <typename T>
lapack_int GesddWork(int matrix_layout, char jobz, lapack_int m, lapack_int n,
                     T* a, lapack_int lda, RealT<T>* s, T* u, lapack_int ldu,
                     T* vt, lapack_int ldvt, T* work, lapack_int lwork,
                     RealT<T>* rwork, lapack_int* iwork) {
  const lapack_int minmn = std::min(m, n);
  const bool all = lapack::Lsame(jobz, 'A');
  const bool some = lapack::Lsame(jobz, 'S');
  const bool over = lapack::Lsame(jobz, 'O');
  const lapack_int nrows_u = (all || some || (over && m < n)) ? m : 1;
  const lapack_int ncols_u = (all || (over && m < n)) ? m : (some ? minmn : 1);
  const lapack_int nrows_vt =
      (all || (over && m >= n)) ? n : (some ? minmn : 1);
  MatrixArg<T> u_arg{u, nrows_u, ncols_u, ldu, -9};
  u_arg.in = false;
  u_arg.used = all || some || (over && m < n);
  MatrixArg<T> vt_arg{vt, nrows_vt, n, ldvt, -11};
  vt_arg.in = false;
  vt_arg.used = all || some || (over && m >= n);
  return CallWork<T, 3>(
      matrix_layout, Name<T>("gesdd_work"), lwork == -1,
      {{{a, m, n, lda, -6}, u_arg, vt_arg}}, [&](auto& p, auto& ld) {
        return lapack::Gesdd(jobz, m, n, p[0], ld[0], s, p[1], ld[1], p[2],
                             ld[2], work, lwork, rwork, iwork);
      });
}

template <typename T>
lapack_int Gesdd(int matrix_layout, char jobz, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, RealT<T>* s, T* u, lapack_int ldu, T* vt,
                 lapack_int ldvt) {
  using R = RealT<T>;
  const std::string name = Name<T>("gesdd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -5;
  const lapack_int minmn = std::min(m, n);
  // Additional scalars initializations for work arrays (the reference's
  // sizes for complex T, which exceed the documented minimum of RWORK).
  R* rwork = nullptr;
  if constexpr (IsComplex<T>::value) {
    const size_t lrwork =
        lapack::Lsame(jobz, 'N')
            ? static_cast<size_t>(std::max<lapack_int>(1, 7 * minmn))
            : static_cast<size_t>(std::max<lapack_int>(
                  1, minmn * std::max(5 * minmn + 7,
                                      2 * std::max(m, n) + 2 * minmn + 1)));
    // Allocate memory for working array(s).
    rwork = static_cast<R*>(malloc(sizeof(R) * lrwork));
    if (rwork == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return LAPACK_WORK_MEMORY_ERROR;
    }
  }
  lapack_int info = WithWorkArray<lapack_int>(
      name, std::max<lapack_int>(1, 8 * minmn), [&](lapack_int* iwork) {
        // Query optimal working array(s) size.
        T work_query = T(0);
        lapack_int status =
            GesddWork(matrix_layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt,
                      &work_query, -1, rwork, iwork);
        if (status != 0) return status;
        const lapack_int lwork = static_cast<lapack_int>(std::real(work_query));
        return WithWorkArray<T>(name, lwork, [&](T* work) {
          return GesddWork(matrix_layout, jobz, m, n, a, lda, s, u, ldu, vt,
                           ldvt, work, lwork, rwork, iwork);
        });
      });
  free(rwork);
  return info;
}

// xBDSDC (real only). U and VT are output only; Q and IQ are not
// transposed.
template <typename R>
lapack_int BdsdcWork(int matrix_layout, char uplo, char compq, lapack_int n,
                     R* d, R* e, R* u, lapack_int ldu, R* vt, lapack_int ldvt,
                     R* q, lapack_int* iq, R* work, lapack_int* iwork) {
  const bool vectors = lapack::Lsame(compq, 'I');
  MatrixArg<R> u_arg{u, n, n, ldu, -8};
  u_arg.in = false;
  u_arg.used = vectors;
  MatrixArg<R> vt_arg{vt, n, n, ldvt, -10};
  vt_arg.in = false;
  vt_arg.used = vectors;
  return CallWork<R, 2>(matrix_layout, Name<R>("bdsdc_work"), false,
                        {{u_arg, vt_arg}}, [&](auto& p, auto& ld) {
                          return lapack::Bdsdc(uplo, compq, n, d, e, p[0],
                                               ld[0], p[1], ld[1], q, iq, work,
                                               iwork);
                        });
}

template <typename R>
lapack_int Bdsdc(int matrix_layout, char uplo, char compq, lapack_int n, R* d,
                 R* e, R* u, lapack_int ldu, R* vt, lapack_int ldvt, R* q,
                 lapack_int* iq) {
  const std::string name = Name<R>("bdsdc");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -5;
    if (VectorNancheck(n - 1, e, 1)) return -6;
  }
  // Additional scalars initializations for work arrays.
  const size_t n1 = static_cast<size_t>(std::max<lapack_int>(1, n));
  size_t lwork = 1;  // any value
  if (lapack::Lsame(compq, 'I')) {
    lwork = 3 * n1 * n1 + 4 * n1;
  } else if (lapack::Lsame(compq, 'P')) {
    lwork = static_cast<size_t>(std::max<lapack_int>(1, 6 * n));
  } else if (lapack::Lsame(compq, 'N')) {
    lwork = static_cast<size_t>(std::max<lapack_int>(1, 4 * n));
  }
  return WithWorkArray<lapack_int>(
      name, std::max<lapack_int>(1, 8 * n), [&](lapack_int* iwork) {
        // Allocate memory for working array(s).
        R* work = static_cast<R*>(malloc(sizeof(R) * lwork));
        if (work == nullptr) {
          Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
          return LAPACK_WORK_MEMORY_ERROR;
        }
        // Call middle-level interface.
        const lapack_int info = BdsdcWork(matrix_layout, uplo, compq, n, d, e,
                                          u, ldu, vt, ldvt, q, iq, work, iwork);
        free(work);
        return info;
      });
}

// Complex matrices with real ones (complex T only): xLACP2, xLACRM and
// xLARCM, whose matrices have different types. The row-major layout copies
// them into column-major temporaries as CallWork does.

// Allocates N column-major temporaries of the given sizes, or returns
// false (after the error message) if one fails.
template <size_t N>
bool AllocateTemporaries(const std::string& name,
                         const std::array<size_t, N>& bytes,
                         std::array<void*, N>& out) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = malloc(bytes[i]);
    if (out[i] == nullptr) {
      for (size_t j = 0; j < i; ++j) free(out[j]);
      Xerbla(name, LAPACK_TRANSPOSE_MEMORY_ERROR);
      return false;
    }
  }
  return true;
}

// xLACP2. As for LAPACKE_xlacpy, the row-major layout also copies B in, so
// that the elements of B outside the triangle (UPLO = 'U' or 'L') are kept
// (the reference returns them undefined).
template <typename T>
lapack_int Lacp2Work(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                     const RealT<T>* a, lapack_int lda, T* b, lapack_int ldb) {
  using R = RealT<T>;
  const std::string name = Name<T>("lacp2_work");
  if (matrix_layout == LAPACK_COL_MAJOR) {
    lapack::Lacp2(uplo, m, n, a, lda, b, ldb);
    return 0;
  }
  if (matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(name, -1);
    return -1;
  }
  // Check leading dimension(s).
  if (lda < n) {
    Xerbla(name, -6);
    return -6;
  }
  if (ldb < n) {
    Xerbla(name, -8);
    return -8;
  }
  const lapack_int ld_t = std::max<lapack_int>(1, m);
  const size_t size = static_cast<size_t>(ld_t) *
                      static_cast<size_t>(std::max<lapack_int>(1, n));
  std::array<void*, 2> t{};
  if (!AllocateTemporaries<2>(name, {sizeof(R) * size, sizeof(T) * size}, t)) {
    return LAPACK_TRANSPOSE_MEMORY_ERROR;
  }
  R* a_t = static_cast<R*>(t[0]);
  T* b_t = static_cast<T*>(t[1]);
  GeTrans(matrix_layout, m, n, a, lda, a_t, ld_t);
  GeTrans(matrix_layout, m, n, b, ldb, b_t, ld_t);
  lapack::Lacp2(uplo, m, n, a_t, ld_t, b_t, ld_t);
  GeTrans(LAPACK_COL_MAJOR, m, n, b_t, ld_t, b, ldb);
  free(b_t);
  free(a_t);
  return 0;
}

template <typename T>
lapack_int Lacp2(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                 const RealT<T>* a, lapack_int lda, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("lacp2"), -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -5;
  return Lacp2Work(matrix_layout, uplo, m, n, a, lda, b, ldb);
}

// xLACRM (A complex m x n, B real n x n) and xLARCM (A real m x m, B
// complex m x n): C (m x n) := A * B.
template <typename T, bool kRealA>
lapack_int RealComplexProductWork(int matrix_layout, lapack_int m, lapack_int n,
                                  const void* a, lapack_int lda, const void* b,
                                  lapack_int ldb, T* c, lapack_int ldc,
                                  RealT<T>* rwork) {
  using R = RealT<T>;
  using TA = std::conditional_t<kRealA, R, T>;
  using TB = std::conditional_t<kRealA, T, R>;
  const std::string name = Name<T>(kRealA ? "larcm_work" : "lacrm_work");
  const TA* ap = static_cast<const TA*>(a);
  const TB* bp = static_cast<const TB*>(b);
  const auto call = [&](const TA* aa, lapack_int la, const TB* bb,
                        lapack_int lb, T* cc, lapack_int lc) {
    if constexpr (kRealA) {
      lapack::Larcm(m, n, aa, la, bb, lb, cc, lc, rwork);
    } else {
      lapack::Lacrm(m, n, aa, la, bb, lb, cc, lc, rwork);
    }
  };
  if (matrix_layout == LAPACK_COL_MAJOR) {
    call(ap, lda, bp, ldb, c, ldc);
    return 0;
  }
  if (matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(name, -1);
    return -1;
  }
  // A is m x m (xLARCM) or m x n (xLACRM), B is m x n or n x n.
  const lapack_int a_cols = kRealA ? m : n;
  const lapack_int b_rows = kRealA ? m : n;
  // Check leading dimension(s).
  if (lda < a_cols) {
    Xerbla(name, -5);
    return -5;
  }
  if (ldb < n) {
    Xerbla(name, -7);
    return -7;
  }
  if (ldc < n) {
    Xerbla(name, -9);
    return -9;
  }
  const lapack_int lda_t = std::max<lapack_int>(1, m);
  const lapack_int ldb_t = std::max<lapack_int>(1, b_rows);
  const lapack_int ldc_t = std::max<lapack_int>(1, m);
  const auto cols = [](lapack_int k) {
    return static_cast<size_t>(std::max<lapack_int>(1, k));
  };
  std::array<void*, 3> t{};
  if (!AllocateTemporaries<3>(name,
                              {sizeof(TA) * cols(lda_t) * cols(a_cols),
                               sizeof(TB) * cols(ldb_t) * cols(n),
                               sizeof(T) * cols(ldc_t) * cols(n)},
                              t)) {
    return LAPACK_TRANSPOSE_MEMORY_ERROR;
  }
  TA* a_t = static_cast<TA*>(t[0]);
  TB* b_t = static_cast<TB*>(t[1]);
  T* c_t = static_cast<T*>(t[2]);
  // Transpose input matrices.
  GeTrans(matrix_layout, m, a_cols, ap, lda, a_t, lda_t);
  GeTrans(matrix_layout, b_rows, n, bp, ldb, b_t, ldb_t);
  call(a_t, lda_t, b_t, ldb_t, c_t, ldc_t);
  // Transpose output matrices.
  GeTrans(LAPACK_COL_MAJOR, m, n, c_t, ldc_t, c, ldc);
  free(c_t);
  free(b_t);
  free(a_t);
  return 0;
}

template <typename T, bool kRealA>
lapack_int RealComplexProduct(int matrix_layout, lapack_int m, lapack_int n,
                              const void* a, lapack_int lda, const void* b,
                              lapack_int ldb, T* c, lapack_int ldc) {
  using R = RealT<T>;
  const std::string name = Name<T>(kRealA ? "larcm" : "lacrm");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if constexpr (kRealA) {
      if (GeNancheck(matrix_layout, m, m, static_cast<const R*>(a), lda)) {
        return -4;
      }
      if (GeNancheck(matrix_layout, m, n, static_cast<const T*>(b), ldb)) {
        return -6;
      }
    } else {
      if (GeNancheck(matrix_layout, m, n, static_cast<const T*>(a), lda)) {
        return -4;
      }
      if (GeNancheck(matrix_layout, n, n, static_cast<const R*>(b), ldb)) {
        return -6;
      }
    }
  }
  return WithWorkArray<R>(
      name, std::max<lapack_int>(1, 2 * m * n), [&](R* rwork) {
        return RealComplexProductWork<T, kRealA>(matrix_layout, m, n, a, lda, b,
                                                 ldb, c, ldc, rwork);
      });
}

// ---------------------------------------------------------------------------
// Nonsymmetric eigenproblem

// xGEBAL. In row-major layout, A is copied only if JOB permutes or scales.
template <typename T>
lapack_int GebalWork(int matrix_layout, char job, lapack_int n, T* a,
                     lapack_int lda, lapack_int* ilo, lapack_int* ihi,
                     RealT<T>* scale) {
  MatrixArg<T> arg{a, n, n, lda, -5};
  arg.used = lapack::Lsame(job, 'B') || lapack::Lsame(job, 'P') ||
             lapack::Lsame(job, 'S');
  arg.in = arg.out = arg.used;
  return CallWork<T, 1>(matrix_layout, Name<T>("gebal_work"), false, {{arg}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gebal(job, n, p[0], ld[0], ilo, ihi,
                                               scale);
                        });
}

template <typename T>
lapack_int Gebal(int matrix_layout, char job, lapack_int n, T* a,
                 lapack_int lda, lapack_int* ilo, lapack_int* ihi,
                 RealT<T>* scale) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gebal"), -1);
    return -1;
  }
  if (NanCheck() && (lapack::Lsame(job, 'B') || lapack::Lsame(job, 'P') ||
                     lapack::Lsame(job, 'S'))) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -4;
  }
  return GebalWork(matrix_layout, job, n, a, lda, ilo, ihi, scale);
}

// xGEBAK: V is n x m.
template <typename T>
lapack_int GebakWork(int matrix_layout, char job, char side, lapack_int n,
                     lapack_int ilo, lapack_int ihi, const RealT<T>* scale,
                     lapack_int m, T* v, lapack_int ldv) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gebak_work"), false,
                        {{{v, n, m, ldv, -10}}}, [&](auto& p, auto& ld) {
                          return lapack::Gebak(job, side, n, ilo, ihi, scale, m,
                                               p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Gebak(int matrix_layout, char job, char side, lapack_int n,
                 lapack_int ilo, lapack_int ihi, const RealT<T>* scale,
                 lapack_int m, T* v, lapack_int ldv) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gebak"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, scale, 1)) return -7;
    if (GeNancheck(matrix_layout, n, m, v, ldv)) return -9;
  }
  return GebakWork(matrix_layout, job, side, n, ilo, ihi, scale, m, v, ldv);
}

template <typename T>
lapack_int GehrdWork(int matrix_layout, lapack_int n, lapack_int ilo,
                     lapack_int ihi, T* a, lapack_int lda, T* tau, T* work,
                     lapack_int lwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gehrd_work"), lwork == -1,
                        {{{a, n, n, lda, -6}}}, [&](auto& p, auto& ld) {
                          return lapack::Gehrd(n, ilo, ihi, p[0], ld[0], tau,
                                               work, lwork);
                        });
}

template <typename T>
lapack_int Gehrd(int matrix_layout, lapack_int n, lapack_int ilo,
                 lapack_int ihi, T* a, lapack_int lda, T* tau) {
  const std::string name = Name<T>("gehrd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -5;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GehrdWork(matrix_layout, n, ilo, ihi, a, lda, tau, work, lwork);
  });
}

// xORGHR / xUNGHR.
template <typename T>
lapack_int UnghrWork(int matrix_layout, lapack_int n, lapack_int ilo,
                     lapack_int ihi, T* a, lapack_int lda, const T* tau,
                     T* work, lapack_int lwork) {
  return CallWork<T, 1>(
      matrix_layout,
      Name<T>(lapack::RealOrComplex<T>("orghr_work", "unghr_work")),
      lwork == -1, {{{a, n, n, lda, -6}}}, [&](auto& p, auto& ld) {
        return lapack::Unghr(n, ilo, ihi, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Unghr(int matrix_layout, lapack_int n, lapack_int ilo,
                 lapack_int ihi, T* a, lapack_int lda, const T* tau) {
  const std::string name = Name<T>(lapack::RealOrComplex<T>("orghr", "unghr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -5;
    if (VectorNancheck(n - 1, tau, 1)) return -7;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnghrWork(matrix_layout, n, ilo, ihi, a, lda, tau, work, lwork);
  });
}

// xORMHR / xUNMHR: A is r x r with r = m (SIDE = 'L') or n.
template <typename T>
lapack_int UnmhrWork(int matrix_layout, char side, char trans, lapack_int m,
                     lapack_int n, lapack_int ilo, lapack_int ihi, const T* a,
                     lapack_int lda, const T* tau, T* c, lapack_int ldc,
                     T* work, lapack_int lwork) {
  const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
  return CallWork<T, 2>(
      matrix_layout,
      Name<T>(lapack::RealOrComplex<T>("ormhr_work", "unmhr_work")),
      lwork == -1, {{Input(a, r, r, lda, -9), {c, m, n, ldc, -12}}},
      [&](auto& p, auto& ld) {
        return lapack::Unmhr(side, trans, m, n, ilo, ihi, p[0], ld[0], tau,
                             p[1], ld[1], work, lwork);
      });
}

template <typename T>
lapack_int Unmhr(int matrix_layout, char side, char trans, lapack_int m,
                 lapack_int n, lapack_int ilo, lapack_int ihi, const T* a,
                 lapack_int lda, const T* tau, T* c, lapack_int ldc) {
  const std::string name = Name<T>(lapack::RealOrComplex<T>("ormhr", "unmhr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
    if (GeNancheck(matrix_layout, r, r, a, lda)) return -8;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -11;
    if (VectorNancheck(r - 1, tau, 1)) return -10;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnmhrWork(matrix_layout, side, trans, m, n, ilo, ihi, a, lda, tau, c,
                     ldc, work, lwork);
  });
}

// xHSEQR: WR and WI for real T (wi is unused for complex T, where wr is
// W). Z is copied in for COMPZ = 'V' and out for 'I' and 'V'; LDZ is
// checked in row-major layout also for COMPZ = 'N', as in the reference.
template <typename T>
lapack_int HseqrWork(int matrix_layout, char job, char compz, lapack_int n,
                     lapack_int ilo, lapack_int ihi, T* h, lapack_int ldh,
                     T* wr, T* wi, T* z, lapack_int ldz, T* work,
                     lapack_int lwork) {
  constexpr bool kComplex = IsComplex<T>::value;
  MatrixArg<T> z_arg{z, n, n, ldz, kComplex ? -11 : -12};
  z_arg.in = lapack::Lsame(compz, 'V');
  z_arg.used = z_arg.in || lapack::Lsame(compz, 'I');
  z_arg.out = z_arg.used;
  return CallWork<T, 2>(
      matrix_layout, Name<T>("hseqr_work"), lwork == -1,
      {{{h, n, n, ldh, -8}, z_arg}}, [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)wi;
          return lapack::Hseqr(job, compz, n, ilo, ihi, p[0], ld[0], wr, p[1],
                               ld[1], work, lwork);
        } else {
          return lapack::Hseqr(job, compz, n, ilo, ihi, p[0], ld[0], wr, wi,
                               p[1], ld[1], work, lwork);
        }
      });
}

template <typename T>
lapack_int Hseqr(int matrix_layout, char job, char compz, lapack_int n,
                 lapack_int ilo, lapack_int ihi, T* h, lapack_int ldh, T* wr,
                 T* wi, T* z, lapack_int ldz) {
  constexpr bool kComplex = IsComplex<T>::value;
  const std::string name = Name<T>("hseqr");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, h, ldh)) return -7;
    if (lapack::Lsame(compz, 'I') || lapack::Lsame(compz, 'V')) {
      if (GeNancheck(matrix_layout, n, n, z, ldz)) return kComplex ? -10 : -11;
    }
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return HseqrWork(matrix_layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, z,
                     ldz, work, lwork);
  });
}

// xTREXC. Q is used (and its LDQ checked in row-major layout) only for
// COMPQ = 'V'. The real routine has the pointers IFST and ILST and a work
// array of N elements; the complex one takes their values (and no work).
template <typename T>
lapack_int TrexcWork(int matrix_layout, char compq, lapack_int n, T* t,
                     lapack_int ldt, T* q, lapack_int ldq, lapack_int* ifst,
                     lapack_int* ilst, T* work) {
  MatrixArg<T> q_arg{q, n, n, ldq, -7};
  q_arg.used = lapack::Lsame(compq, 'V');
  q_arg.in = q_arg.out = q_arg.used;
  if (!q_arg.used) q_arg.cols = 0;
  return CallWork<T, 2>(matrix_layout, Name<T>("trexc_work"), false,
                        {{{t, n, n, ldt, -5}, q_arg}}, [&](auto& p, auto& ld) {
                          if constexpr (IsComplex<T>::value) {
                            (void)work;
                            return lapack::Trexc(compq, n, p[0], ld[0], p[1],
                                                 ld[1], *ifst, *ilst);
                          } else {
                            return lapack::Trexc(compq, n, p[0], ld[0], p[1],
                                                 ld[1], ifst, ilst, work);
                          }
                        });
}

template <typename T>
lapack_int Trexc(int matrix_layout, char compq, lapack_int n, T* t,
                 lapack_int ldt, T* q, lapack_int ldq, lapack_int* ifst,
                 lapack_int* ilst) {
  const std::string name = Name<T>("trexc");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (lapack::Lsame(compq, 'V') && GeNancheck(matrix_layout, n, n, q, ldq)) {
      return -6;
    }
    if (GeNancheck(matrix_layout, n, n, t, ldt)) return -4;
  }
  if constexpr (IsComplex<T>::value) {
    return TrexcWork(matrix_layout, compq, n, t, ldt, q, ldq, ifst, ilst,
                     static_cast<T*>(nullptr));
  } else {
    return WithWorkArray<T>(name, std::max<lapack_int>(1, n), [&](T* work) {
      return TrexcWork(matrix_layout, compq, n, t, ldt, q, ldq, ifst, ilst,
                       work);
    });
  }
}

// xTRSEN: WR and WI for real T (wi is unused for complex T, where wr is
// W); IWORK for real T only. LDQ is checked in row-major layout also for
// COMPQ = 'N'. As in the reference, the real routine's workspace query in
// row-major layout gets the transposed T (DTRSEN reads it to count the
// selected eigenvalues), the complex one's the caller's.
template <typename T>
lapack_int TrsenWork(int matrix_layout, char job, char compq,
                     const lapack_logical* select, lapack_int n, T* t,
                     lapack_int ldt, T* q, lapack_int ldq, T* wr, T* wi,
                     lapack_int* m, RealT<T>* s, RealT<T>* sep, T* work,
                     lapack_int lwork, lapack_int* iwork, lapack_int liwork) {
  constexpr bool kComplex = IsComplex<T>::value;
  const bool query = lwork == -1 || (!kComplex && liwork == -1);
  MatrixArg<T> t_arg{t, n, n, ldt, -7};
  MatrixArg<T> q_arg{q, n, n, ldq, -9};
  q_arg.used = lapack::Lsame(compq, 'V');
  q_arg.in = q_arg.out = q_arg.used;
  if constexpr (!kComplex) {
    // The query copies T in, but nothing back, and does not use Q.
    if (query) {
      t_arg.out = false;
      q_arg.used = q_arg.in = q_arg.out = false;
    }
  }
  return CallWork<T, 2>(
      matrix_layout, Name<T>("trsen_work"), kComplex && query, {{t_arg, q_arg}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)wi;
          (void)iwork;
          (void)liwork;
          return lapack::Trsen(job, compq, select, n, p[0], ld[0], p[1], ld[1],
                               wr, m, s, sep, work, lwork);
        } else {
          // (A query in row-major layout gets the caller's Q.)
          T* qp = (query && matrix_layout == LAPACK_ROW_MAJOR) ? q : p[1];
          return lapack::Trsen(job, compq, select, n, p[0], ld[0], qp, ld[1],
                               wr, wi, m, s, sep, work, lwork, iwork, liwork);
        }
      });
}

template <typename T>
lapack_int Trsen(int matrix_layout, char job, char compq,
                 const lapack_logical* select, lapack_int n, T* t,
                 lapack_int ldt, T* q, lapack_int ldq, T* wr, T* wi,
                 lapack_int* m, RealT<T>* s, RealT<T>* sep) {
  const std::string name = Name<T>("trsen");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (lapack::Lsame(compq, 'V') && GeNancheck(matrix_layout, n, n, q, ldq)) {
      return -8;
    }
    if (GeNancheck(matrix_layout, n, n, t, ldt)) return -6;
  }
  if constexpr (IsComplex<T>::value) {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return TrsenWork(matrix_layout, job, compq, select, n, t, ldt, q, ldq, wr,
                       wi, m, s, sep, work, lwork,
                       static_cast<lapack_int*>(nullptr), lapack_int{0});
    });
  } else {
    // Query optimal working array(s) size.
    T work_query = T(0);
    lapack_int iwork_query = 0;
    lapack_int info =
        TrsenWork(matrix_layout, job, compq, select, n, t, ldt, q, ldq, wr, wi,
                  m, s, sep, &work_query, -1, &iwork_query, -1);
    if (info != 0) return info;
    const lapack_int liwork = iwork_query;
    const lapack_int lwork = static_cast<lapack_int>(work_query);
    // Deviation: the reference allocates IWORK only for JOB = 'B' or 'V'
    // and passes a null pointer otherwise, but DTRSEN always sets
    // IWORK(1), so the reference crashes for JOB = 'N' and 'E'.
    return WithWorkArray<lapack_int>(
        name, std::max<lapack_int>(1, liwork), [&](lapack_int* iwork) {
          return WithWorkArray<T>(name, lwork, [&](T* work) {
            return TrsenWork(matrix_layout, job, compq, select, n, t, ldt, q,
                             ldq, wr, wi, m, s, sep, work, lwork, iwork,
                             liwork);
          });
        });
  }
}

template <typename T>
lapack_int TrsylWork(int matrix_layout, char trana, char tranb, lapack_int isgn,
                     lapack_int m, lapack_int n, const T* a, lapack_int lda,
                     const T* b, lapack_int ldb, T* c, lapack_int ldc,
                     RealT<T>* scale) {
  return CallWork<T, 3>(matrix_layout, Name<T>("trsyl_work"), false,
                        {{Input(a, m, m, lda, -8),
                          Input(b, n, n, ldb, -10),
                          {c, m, n, ldc, -12}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Trsyl(trana, tranb, isgn, m, n, p[0],
                                               ld[0], p[1], ld[1], p[2], ld[2],
                                               scale);
                        });
}

template <typename T>
lapack_int Trsyl(int matrix_layout, char trana, char tranb, lapack_int isgn,
                 lapack_int m, lapack_int n, const T* a, lapack_int lda,
                 const T* b, lapack_int ldb, T* c, lapack_int ldc,
                 RealT<T>* scale) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("trsyl"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, m, a, lda)) return -7;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -9;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -11;
  }
  return TrsylWork(matrix_layout, trana, tranb, isgn, m, n, a, lda, b, ldb, c,
                   ldc, scale);
}

// xLARFX: V and TAU are not transposed.
template <typename T>
lapack_int LarfxWork(int matrix_layout, char side, lapack_int m, lapack_int n,
                     const T* v, T tau, T* c, lapack_int ldc, T* work) {
  return CallWork<T, 1>(matrix_layout, Name<T>("larfx_work"), false,
                        {{{c, m, n, ldc, -8}}}, [&](auto& p, auto& ld) {
                          lapack::Larfx(side, m, n, v, tau, p[0], ld[0], work);
                          return lapack_int{0};
                        });
}

template <typename T>
lapack_int Larfx(int matrix_layout, char side, lapack_int m, lapack_int n,
                 const T* v, T tau, T* c, lapack_int ldc, T* work) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("larfx"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -7;
    if (VectorNancheck(1, &tau, 1)) return -6;
    const lapack_int lv = lapack::Lsame(side, 'L') ? m : n;
    if (VectorNancheck(lv, v, 1)) return -5;
  }
  return LarfxWork(matrix_layout, side, m, n, v, tau, c, ldc, work);
}

// xLACN2 (no matrix_layout). ISGN is used for real T only. As in the
// reference, a NaN in EST gives -5 also for complex T (where EST is the
// fourth argument).
template <typename T>
lapack_int Lacn2(lapack_int n, T* v, T* x, lapack_int* isgn, RealT<T>* est,
                 lapack_int* kase, lapack_int* isave) {
  if (NanCheck()) {
    if (VectorNancheck(1, est, 1)) return -5;
    if (VectorNancheck(n, x, 1)) return -3;
  }
  if constexpr (IsComplex<T>::value) {
    (void)isgn;
    lapack::Lacn2(n, v, x, est, kase, isave);
  } else {
    lapack::Lacn2(n, v, x, isgn, est, kase, isave);
  }
  return 0;
}

// xGEES: WR and WI for real T (wi is unused for complex T, where wr is W);
// RWORK for complex T only. VS is output only (JOBVS = 'V'); LDVS is
// checked in row-major layout also for JOBVS = 'N'.
template <typename T, typename Select>
lapack_int GeesWork(int matrix_layout, char jobvs, char sort, Select select,
                    lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* wr,
                    T* wi, T* vs, lapack_int ldvs, T* work, lapack_int lwork,
                    RealT<T>* rwork, lapack_logical* bwork) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  MatrixArg<T> vs_arg{vs, n, n, ldvs, kComplex ? -11 : -12};
  vs_arg.in = false;
  vs_arg.used = vs_arg.out = lapack::Lsame(jobvs, 'V');
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gees_work"), lwork == -1,
      {{{a, n, n, lda, -7}, vs_arg}}, [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)wi;
          // LAPACK_C_SELECT1 / LAPACK_Z_SELECT1 take a pointer to the
          // complex eigenvalue, the routine one to its two parts.
          const auto sel =
              reinterpret_cast<lapack::GeesSelectComplex<R>>(select);
          return lapack::Gees(jobvs, sort, sel, n, p[0], ld[0], sdim, wr, p[1],
                              ld[1], work, lwork, rwork, bwork);
        } else {
          (void)rwork;
          return lapack::Gees(jobvs, sort, select, n, p[0], ld[0], sdim, wr, wi,
                              p[1], ld[1], work, lwork, bwork);
        }
      });
}

template <typename T, typename Select>
lapack_int Gees(int matrix_layout, char jobvs, char sort, Select select,
                lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* wr,
                T* wi, T* vs, lapack_int ldvs) {
  using R = RealT<T>;
  const std::string name = Name<T>("gees");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -6;
  // Allocate memory for working array(s): BWORK for SORT = 'S', RWORK for
  // complex T.
  lapack_logical* bwork = nullptr;
  if (lapack::Lsame(sort, 'S')) {
    bwork = static_cast<lapack_logical*>(
        malloc(sizeof(lapack_logical) *
               static_cast<size_t>(std::max<lapack_int>(1, n))));
    if (bwork == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return LAPACK_WORK_MEMORY_ERROR;
    }
  }
  R* rwork = nullptr;
  if constexpr (IsComplex<T>::value) {
    rwork = static_cast<R*>(
        malloc(sizeof(R) * static_cast<size_t>(std::max<lapack_int>(1, n))));
    if (rwork == nullptr) {
      free(bwork);
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return LAPACK_WORK_MEMORY_ERROR;
    }
  }
  const lapack_int info =
      WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
        return GeesWork(matrix_layout, jobvs, sort, select, n, a, lda, sdim, wr,
                        wi, vs, ldvs, work, lwork, rwork, bwork);
      });
  free(rwork);
  free(bwork);
  return info;
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

HWYBLAS_API void LAPACKE_xerbla(const char* name, lapack_int info) {
  if (info == LAPACK_WORK_MEMORY_ERROR) {
    printf("Not enough memory to allocate work array in %s\n", name);
  } else if (info == LAPACK_TRANSPOSE_MEMORY_ERROR) {
    printf("Not enough memory to transpose matrix in %s\n", name);
  } else if (info < 0) {
    printf("Wrong parameter %d in %s\n", -static_cast<int>(info), name);
  }
}

HWYBLAS_API void LAPACKE_set_nancheck(int flag) {
  lapacke::g_nancheck.store(flag ? 1 : 0);
}

HWYBLAS_API int LAPACKE_get_nancheck(void) {
  const int flag = lapacke::g_nancheck.load();
  if (flag != -1) return flag;
  // Check the environment variable, once and only once. By default, NaN
  // checking is enabled.
#ifdef _MSC_VER
#pragma warning(suppress : 4996)  // getenv is safe here
#endif
  const char* env = getenv("LAPACKE_NANCHECK");
  const int value = (env == nullptr || atoi(env) != 0) ? 1 : 0;
  int expected = -1;
  lapacke::g_nancheck.compare_exchange_strong(expected, value);
  return lapacke::g_nancheck.load();
}

// The functions of the routines of all four precisions; `gen` and `mul` are
// orgqr / ungqr etc. (without the precision letter).
#define HWYBLAS_LAPACKE(p, T, gen_qr, gen_lq, mul_qr, mul_lq)                  \
  HWYBLAS_API lapack_int LAPACKE_##p##getrf(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda,            \
      lapack_int* ipiv) {                                                      \
    return lapacke::Getrf(layout, m, n, a, lda, ipiv);                         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##getrf_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda,            \
      lapack_int* ipiv) {                                                      \
    return lapacke::GetrfWork(layout, m, n, a, lda, ipiv);                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##getrf2(                                  \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda,            \
      lapack_int* ipiv) {                                                      \
    return lapacke::Getrf2(layout, m, n, a, lda, ipiv);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##getrf2_work(                             \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda,            \
      lapack_int* ipiv) {                                                      \
    return lapacke::Getrf2Work(layout, m, n, a, lda, ipiv);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##laswp(                                   \
      int layout, lapack_int n, T* a, lapack_int lda, lapack_int k1,           \
      lapack_int k2, const lapack_int* ipiv, lapack_int incx) {                \
    return lapacke::Laswp(layout, n, a, lda, k1, k2, ipiv, incx);              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##laswp_work(                              \
      int layout, lapack_int n, T* a, lapack_int lda, lapack_int k1,           \
      lapack_int k2, const lapack_int* ipiv, lapack_int incx) {                \
    return lapacke::LaswpWork(layout, n, a, lda, k1, k2, ipiv, incx);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potrf(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::Potrf(layout, uplo, n, a, lda, false);                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potrf_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::PotrfWork(layout, uplo, n, a, lda, false);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potrf2(                                  \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::Potrf(layout, uplo, n, a, lda, true);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potrf2_work(                             \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::PotrfWork(layout, uplo, n, a, lda, true);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrf(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) {  \
    return lapacke::Geqrf(layout, m, n, a, lda, tau, false);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrf_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,    \
      T* work, lapack_int lwork) {                                             \
    return lapacke::GeqrfWork(layout, m, n, a, lda, tau, work, lwork, false);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelqf(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) {  \
    return lapacke::Geqrf(layout, m, n, a, lda, tau, true);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelqf_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,    \
      T* work, lapack_int lwork) {                                             \
    return lapacke::GeqrfWork(layout, m, n, a, lda, tau, work, lwork, true);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geqr2(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) {  \
    return lapacke::Geqr2(layout, m, n, a, lda, tau, false);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geqr2_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,    \
      T* work) {                                                               \
    return lapacke::Geqr2Work(layout, m, n, a, lda, tau, work, false);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelq2(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) {  \
    return lapacke::Geqr2(layout, m, n, a, lda, tau, true);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gelq2_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,    \
      T* work) {                                                               \
    return lapacke::Geqr2Work(layout, m, n, a, lda, tau, work, true);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gen_qr(int layout, lapack_int m,         \
                                             lapack_int n, lapack_int k, T* a, \
                                             lapack_int lda, const T* tau) {   \
    return lapacke::Ungqr(layout, m, n, k, a, lda, tau, false);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gen_qr##_work(                           \
      int layout, lapack_int m, lapack_int n, lapack_int k, T* a,              \
      lapack_int lda, const T* tau, T* work, lapack_int lwork) {               \
    return lapacke::UngqrWork(layout, m, n, k, a, lda, tau, work, lwork,       \
                              false);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gen_lq(int layout, lapack_int m,         \
                                             lapack_int n, lapack_int k, T* a, \
                                             lapack_int lda, const T* tau) {   \
    return lapacke::Ungqr(layout, m, n, k, a, lda, tau, true);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gen_lq##_work(                           \
      int layout, lapack_int m, lapack_int n, lapack_int k, T* a,              \
      lapack_int lda, const T* tau, T* work, lapack_int lwork) {               \
    return lapacke::UngqrWork(layout, m, n, k, a, lda, tau, work, lwork,       \
                              true);                                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul_qr(                                  \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc) {                                                        \
    return lapacke::Unmqr(layout, side, trans, m, n, k, a, lda, tau, c, ldc,   \
                          false);                                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul_qr##_work(                           \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc, T* work, lapack_int lwork) {                             \
    return lapacke::UnmqrWork(layout, side, trans, m, n, k, a, lda, tau, c,    \
                              ldc, work, lwork, false);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul_lq(                                  \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc) {                                                        \
    return lapacke::Unmqr(layout, side, trans, m, n, k, a, lda, tau, c, ldc,   \
                          true);                                               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mul_lq##_work(                           \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc, T* work, lapack_int lwork) {                             \
    return lapacke::UnmqrWork(layout, side, trans, m, n, k, a, lda, tau, c,    \
                              ldc, work, lwork, true);                         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacpy(                                   \
      int layout, char uplo, lapack_int m, lapack_int n, const T* a,           \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::Lacpy(layout, uplo, m, n, a, lda, b, ldb);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacpy_work(                              \
      int layout, char uplo, lapack_int m, lapack_int n, const T* a,           \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::LacpyWork(layout, uplo, m, n, a, lda, b, ldb);             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larfg(lapack_int n, T* alpha, T* x,      \
                                            lapack_int incx, T* tau) {         \
    return lapacke::Larfg(n, alpha, x, incx, tau);                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larfg_work(lapack_int n, T* alpha, T* x, \
                                                 lapack_int incx, T* tau) {    \
    hwyblas::lapack::Larfg(n, alpha, x, incx, tau);                            \
    return 0;                                                                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larft(                                   \
      int layout, char direct, char storev, lapack_int n, lapack_int k,        \
      const T* v, lapack_int ldv, const T* tau, T* t, lapack_int ldt) {        \
    return lapacke::Larft(layout, direct, storev, n, k, v, ldv, tau, t, ldt);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larft_work(                              \
      int layout, char direct, char storev, lapack_int n, lapack_int k,        \
      const T* v, lapack_int ldv, const T* tau, T* t, lapack_int ldt) {        \
    return lapacke::LarftWork(layout, direct, storev, n, k, v, ldv, tau, t,    \
                              ldt);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larfb(                                   \
      int layout, char side, char trans, char direct, char storev,             \
      lapack_int m, lapack_int n, lapack_int k, const T* v, lapack_int ldv,    \
      const T* t, lapack_int ldt, T* c, lapack_int ldc) {                      \
    return lapacke::Larfb(layout, side, trans, direct, storev, m, n, k, v,     \
                          ldv, t, ldt, c, ldc);                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larfb_work(                              \
      int layout, char side, char trans, char direct, char storev,             \
      lapack_int m, lapack_int n, lapack_int k, const T* v, lapack_int ldv,    \
      const T* t, lapack_int ldt, T* c, lapack_int ldc, T* work,               \
      lapack_int ldwork) {                                                     \
    return lapacke::LarfbWork(layout, side, trans, direct, storev, m, n, k, v, \
                              ldv, t, ldt, c, ldc, work, ldwork);              \
  }

HWYBLAS_LAPACKE(s, float, orgqr, orglq, ormqr, ormlq)
HWYBLAS_LAPACKE(d, double, orgqr, orglq, ormqr, ormlq)
HWYBLAS_LAPACKE(c, c32, ungqr, unglq, unmqr, unmlq)
HWYBLAS_LAPACKE(z, c64, ungqr, unglq, unmqr, unmlq)

#undef HWYBLAS_LAPACKE

#define HWYBLAS_LAPACKE_COMPLEX(p, T)                                \
  HWYBLAS_API lapack_int LAPACKE_##p##lacgv(lapack_int n, T* x,      \
                                            lapack_int incx) {       \
    return lapacke::Lacgv(n, x, incx);                               \
  }                                                                  \
  HWYBLAS_API lapack_int LAPACKE_##p##lacgv_work(lapack_int n, T* x, \
                                                 lapack_int incx) {  \
    hwyblas::lapack::Lacgv(n, x, incx);                              \
    return 0;                                                        \
  }

HWYBLAS_LAPACKE_COMPLEX(c, c32)
HWYBLAS_LAPACKE_COMPLEX(z, c64)

#undef HWYBLAS_LAPACKE_COMPLEX

#define HWYBLAS_LAPACKE_REAL(p, R)                                            \
  HWYBLAS_API R LAPACKE_##p##lapy2(R x, R y) { return lapacke::Lapy2(x, y); } \
  HWYBLAS_API R LAPACKE_##p##lapy2_work(R x, R y) {                           \
    return hwyblas::lapack::Lapy2(x, y);                                      \
  }                                                                           \
  HWYBLAS_API R LAPACKE_##p##lapy3(R x, R y, R z) {                           \
    return lapacke::Lapy3(x, y, z);                                           \
  }                                                                           \
  HWYBLAS_API R LAPACKE_##p##lapy3_work(R x, R y, R z) {                      \
    return hwyblas::lapack::Lapy3(x, y, z);                                   \
  }

HWYBLAS_LAPACKE_REAL(s, float)
HWYBLAS_LAPACKE_REAL(d, double)

#undef HWYBLAS_LAPACKE_REAL

// The functions of the routines added with the pivoted QR factorization;
// R is T's real type.
#define HWYBLAS_LAPACKE_2(p, T, R)                                             \
  HWYBLAS_API R LAPACKE_##p##lange(int layout, char norm, lapack_int m,        \
                                   lapack_int n, const T* a, lapack_int lda) { \
    return lapacke::Lange(layout, norm, m, n, a, lda);                         \
  }                                                                            \
  HWYBLAS_API R LAPACKE_##p##lange_work(int layout, char norm, lapack_int m,   \
                                        lapack_int n, const T* a,              \
                                        lapack_int lda, R* work) {             \
    return lapacke::LangeWork(layout, norm, m, n, a, lda, work);               \
  }                                                                            \
  HWYBLAS_API R LAPACKE_##p##lansy(int layout, char norm, char uplo,           \
                                   lapack_int n, const T* a, lapack_int lda) { \
    return lapacke::Lansy(layout, norm, uplo, n, a, lda, false);               \
  }                                                                            \
  HWYBLAS_API R LAPACKE_##p##lansy_work(int layout, char norm, char uplo,      \
                                        lapack_int n, const T* a,              \
                                        lapack_int lda, R* work) {             \
    return lapacke::LansyWork(layout, norm, uplo, n, a, lda, work, false);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lascl(                                   \
      int layout, char type, lapack_int kl, lapack_int ku, R cfrom, R cto,     \
      lapack_int m, lapack_int n, T* a, lapack_int lda) {                      \
    return lapacke::Lascl(layout, type, kl, ku, cfrom, cto, m, n, a, lda);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lascl_work(                              \
      int layout, char type, lapack_int kl, lapack_int ku, R cfrom, R cto,     \
      lapack_int m, lapack_int n, T* a, lapack_int lda) {                      \
    return lapacke::LasclWork(layout, type, kl, ku, cfrom, cto, m, n, a, lda); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##laset(                                   \
      int layout, char uplo, lapack_int m, lapack_int n, T alpha, T beta,      \
      T* a, lapack_int lda) {                                                  \
    return lapacke::Laset(layout, uplo, m, n, alpha, beta, a, lda);            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##laset_work(                              \
      int layout, char uplo, lapack_int m, lapack_int n, T alpha, T beta,      \
      T* a, lapack_int lda) {                                                  \
    return lapacke::LasetWork(layout, uplo, m, n, alpha, beta, a, lda);        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geqp3(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda,            \
      lapack_int* jpvt, T* tau) {                                              \
    return lapacke::Geqp3(layout, m, n, a, lda, jpvt, tau);                    \
  }

HWYBLAS_LAPACKE_2(s, float, float)
HWYBLAS_LAPACKE_2(d, double, double)
HWYBLAS_LAPACKE_2(c, c32, float)
HWYBLAS_LAPACKE_2(z, c64, double)

#undef HWYBLAS_LAPACKE_2

HWYBLAS_API lapack_int LAPACKE_sgeqp3_work(int layout, lapack_int m,
                                           lapack_int n, float* a,
                                           lapack_int lda, lapack_int* jpvt,
                                           float* tau, float* work,
                                           lapack_int lwork) {
  return lapacke::Geqp3Work(layout, m, n, a, lda, jpvt, tau, work, lwork,
                            static_cast<float*>(nullptr));
}
HWYBLAS_API lapack_int LAPACKE_dgeqp3_work(int layout, lapack_int m,
                                           lapack_int n, double* a,
                                           lapack_int lda, lapack_int* jpvt,
                                           double* tau, double* work,
                                           lapack_int lwork) {
  return lapacke::Geqp3Work(layout, m, n, a, lda, jpvt, tau, work, lwork,
                            static_cast<double*>(nullptr));
}
HWYBLAS_API lapack_int LAPACKE_cgeqp3_work(int layout, lapack_int m,
                                           lapack_int n, c32* a, lapack_int lda,
                                           lapack_int* jpvt, c32* tau,
                                           c32* work, lapack_int lwork,
                                           float* rwork) {
  return lapacke::Geqp3Work(layout, m, n, a, lda, jpvt, tau, work, lwork,
                            rwork);
}
HWYBLAS_API lapack_int LAPACKE_zgeqp3_work(int layout, lapack_int m,
                                           lapack_int n, c64* a, lapack_int lda,
                                           lapack_int* jpvt, c64* tau,
                                           c64* work, lapack_int lwork,
                                           double* rwork) {
  return lapacke::Geqp3Work(layout, m, n, a, lda, jpvt, tau, work, lwork,
                            rwork);
}

HWYBLAS_API float LAPACKE_clanhe(int layout, char norm, char uplo, lapack_int n,
                                 const c32* a, lapack_int lda) {
  return lapacke::Lansy(layout, norm, uplo, n, a, lda, true);
}
HWYBLAS_API double LAPACKE_zlanhe(int layout, char norm, char uplo,
                                  lapack_int n, const c64* a, lapack_int lda) {
  return lapacke::Lansy(layout, norm, uplo, n, a, lda, true);
}
HWYBLAS_API float LAPACKE_clanhe_work(int layout, char norm, char uplo,
                                      lapack_int n, const c32* a,
                                      lapack_int lda, float* work) {
  return lapacke::LansyWork(layout, norm, uplo, n, a, lda, work, true);
}
HWYBLAS_API double LAPACKE_zlanhe_work(int layout, char norm, char uplo,
                                       lapack_int n, const c64* a,
                                       lapack_int lda, double* work) {
  return lapacke::LansyWork(layout, norm, uplo, n, a, lda, work, true);
}

// The eigenvalue routines: `sy`/`og` are "sy"/"org" (real) or "he"/"ung"
// (complex).
#define HWYBLAS_LAPACKE_3(p, T, R, sy, og)                                     \
  HWYBLAS_API lapack_int LAPACKE_##p##sy##ev(int layout, char jobz, char uplo, \
                                             lapack_int n, T* a,               \
                                             lapack_int lda, R* w) {           \
    return lapacke::Heev(layout, jobz, uplo, n, a, lda, w);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sy##trd(                                 \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda, R* d, R* e,   \
      T* tau) {                                                                \
    return lapacke::Hetrd(layout, uplo, n, a, lda, d, e, tau);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sy##trd_work(                            \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda, R* d, R* e,   \
      T* tau, T* work, lapack_int lwork) {                                     \
    return lapacke::HetrdWork(layout, uplo, n, a, lda, d, e, tau, work,        \
                              lwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##tr(int layout, char uplo,            \
                                             lapack_int n, T* a,               \
                                             lapack_int lda, const T* tau) {   \
    return lapacke::Ungtr(layout, uplo, n, a, lda, tau);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##tr_work(                             \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda, const T* tau, \
      T* work, lapack_int lwork) {                                             \
    return lapacke::UngtrWork(layout, uplo, n, a, lda, tau, work, lwork);      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##ql(int layout, lapack_int m,         \
                                             lapack_int n, lapack_int k, T* a, \
                                             lapack_int lda, const T* tau) {   \
    return lapacke::Ungql(layout, m, n, k, a, lda, tau);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##ql_work(                             \
      int layout, lapack_int m, lapack_int n, lapack_int k, T* a,              \
      lapack_int lda, const T* tau, T* work, lapack_int lwork) {               \
    return lapacke::UngqlWork(layout, m, n, k, a, lda, tau, work, lwork);      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##steqr(int layout, char compz,            \
                                            lapack_int n, R* d, R* e, T* z,    \
                                            lapack_int ldz) {                  \
    return lapacke::Steqr(layout, compz, n, d, e, z, ldz);                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##steqr_work(                              \
      int layout, char compz, lapack_int n, R* d, R* e, T* z, lapack_int ldz,  \
      R* work) {                                                               \
    return lapacke::SteqrWork(layout, compz, n, d, e, z, ldz, work);           \
  }

HWYBLAS_LAPACKE_3(s, float, float, sy, org)
HWYBLAS_LAPACKE_3(d, double, double, sy, org)
HWYBLAS_LAPACKE_3(c, c32, float, he, ung)
HWYBLAS_LAPACKE_3(z, c64, double, he, ung)

#undef HWYBLAS_LAPACKE_3

#define HWYBLAS_LAPACKE_3_REAL(p, R)                                         \
  HWYBLAS_API lapack_int LAPACKE_##p##syev_work(                             \
      int layout, char jobz, char uplo, lapack_int n, R* a, lapack_int lda,  \
      R* w, R* work, lapack_int lwork) {                                     \
    return lapacke::HeevWork(layout, jobz, uplo, n, a, lda, w, work, lwork,  \
                             static_cast<R*>(nullptr));                      \
  }                                                                          \
  HWYBLAS_API lapack_int LAPACKE_##p##sterf(lapack_int n, R* d, R* e) {      \
    return lapacke::Sterf(n, d, e);                                          \
  }                                                                          \
  HWYBLAS_API lapack_int LAPACKE_##p##sterf_work(lapack_int n, R* d, R* e) { \
    return hwyblas::lapack::Sterf(n, d, e);                                  \
  }                                                                          \
  HWYBLAS_API lapack_int LAPACKE_##p##lasrt(char id, lapack_int n, R* d) {   \
    return lapacke::Lasrt(id, n, d);                                         \
  }                                                                          \
  HWYBLAS_API lapack_int LAPACKE_##p##lasrt_work(char id, lapack_int n,      \
                                                 R* d) {                     \
    return hwyblas::lapack::Lasrt(id, n, d);                                 \
  }

HWYBLAS_LAPACKE_3_REAL(s, float)
HWYBLAS_LAPACKE_3_REAL(d, double)

#undef HWYBLAS_LAPACKE_3_REAL

#define HWYBLAS_LAPACKE_3_COMPLEX(p, T, R)                                  \
  HWYBLAS_API lapack_int LAPACKE_##p##heev_work(                            \
      int layout, char jobz, char uplo, lapack_int n, T* a, lapack_int lda, \
      R* w, T* work, lapack_int lwork, R* rwork) {                          \
    return lapacke::HeevWork(layout, jobz, uplo, n, a, lda, w, work, lwork, \
                             rwork);                                        \
  }

HWYBLAS_LAPACKE_3_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_3_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_3_COMPLEX

// The singular value decomposition: `og`/`om` are "org"/"orm" (real) or
// "ung"/"unm" (complex).
#define HWYBLAS_LAPACKE_4(p, T, R, og, om)                                     \
  HWYBLAS_API lapack_int LAPACKE_##p##gesvd(                                   \
      int layout, char jobu, char jobvt, lapack_int m, lapack_int n, T* a,     \
      lapack_int lda, R* s, T* u, lapack_int ldu, T* vt, lapack_int ldvt,      \
      R* superb) {                                                             \
    return lapacke::Gesvd(layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,    \
                          ldvt, superb);                                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gebrd(                                   \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, R* d,      \
      R* e, T* tauq, T* taup) {                                                \
    return lapacke::Gebrd(layout, m, n, a, lda, d, e, tauq, taup);             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gebrd_work(                              \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, R* d,      \
      R* e, T* tauq, T* taup, T* work, lapack_int lwork) {                     \
    return lapacke::GebrdWork(layout, m, n, a, lda, d, e, tauq, taup, work,    \
                              lwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##br(                                  \
      int layout, char vect, lapack_int m, lapack_int n, lapack_int k, T* a,   \
      lapack_int lda, const T* tau) {                                          \
    return lapacke::Ungbr(layout, vect, m, n, k, a, lda, tau);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##br_work(                             \
      int layout, char vect, lapack_int m, lapack_int n, lapack_int k, T* a,   \
      lapack_int lda, const T* tau, T* work, lapack_int lwork) {               \
    return lapacke::UngbrWork(layout, vect, m, n, k, a, lda, tau, work,        \
                              lwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##om##br(                                  \
      int layout, char vect, char side, char trans, lapack_int m,              \
      lapack_int n, lapack_int k, const T* a, lapack_int lda, const T* tau,    \
      T* c, lapack_int ldc) {                                                  \
    return lapacke::Unmbr(layout, vect, side, trans, m, n, k, a, lda, tau, c,  \
                          ldc);                                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##om##br_work(                             \
      int layout, char vect, char side, char trans, lapack_int m,              \
      lapack_int n, lapack_int k, const T* a, lapack_int lda, const T* tau,    \
      T* c, lapack_int ldc, T* work, lapack_int lwork) {                       \
    return lapacke::UnmbrWork(layout, vect, side, trans, m, n, k, a, lda, tau, \
                              c, ldc, work, lwork);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##bdsqr(                                   \
      int layout, char uplo, lapack_int n, lapack_int ncvt, lapack_int nru,    \
      lapack_int ncc, R* d, R* e, T* vt, lapack_int ldvt, T* u,                \
      lapack_int ldu, T* c, lapack_int ldc) {                                  \
    return lapacke::Bdsqr(layout, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u,  \
                          ldu, c, ldc);                                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##bdsqr_work(                              \
      int layout, char uplo, lapack_int n, lapack_int ncvt, lapack_int nru,    \
      lapack_int ncc, R* d, R* e, T* vt, lapack_int ldvt, T* u,                \
      lapack_int ldu, T* c, lapack_int ldc, R* work) {                         \
    return lapacke::BdsqrWork(layout, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, \
                              u, ldu, c, ldc, work);                           \
  }

HWYBLAS_LAPACKE_4(s, float, float, org, orm)
HWYBLAS_LAPACKE_4(d, double, double, org, orm)
HWYBLAS_LAPACKE_4(c, c32, float, ung, unm)
HWYBLAS_LAPACKE_4(z, c64, double, ung, unm)

#undef HWYBLAS_LAPACKE_4

#define HWYBLAS_LAPACKE_4_REAL(p, R)                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##gesvd_work(                              \
      int layout, char jobu, char jobvt, lapack_int m, lapack_int n, R* a,     \
      lapack_int lda, R* s, R* u, lapack_int ldu, R* vt, lapack_int ldvt,      \
      R* work, lapack_int lwork) {                                             \
    return lapacke::GesvdWork(layout, jobu, jobvt, m, n, a, lda, s, u, ldu,    \
                              vt, ldvt, work, lwork,                           \
                              static_cast<R*>(nullptr));                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesdd(                                   \
      int layout, char jobz, lapack_int m, lapack_int n, R* a, lapack_int lda, \
      R* s, R* u, lapack_int ldu, R* vt, lapack_int ldvt) {                    \
    return lapacke::Gesdd(layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesdd_work(                              \
      int layout, char jobz, lapack_int m, lapack_int n, R* a, lapack_int lda, \
      R* s, R* u, lapack_int ldu, R* vt, lapack_int ldvt, R* work,             \
      lapack_int lwork, lapack_int* iwork) {                                   \
    return lapacke::GesddWork(layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, \
                              work, lwork, static_cast<R*>(nullptr), iwork);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##bdsdc(                                   \
      int layout, char uplo, char compq, lapack_int n, R* d, R* e, R* u,       \
      lapack_int ldu, R* vt, lapack_int ldvt, R* q, lapack_int* iq) {          \
    return lapacke::Bdsdc(layout, uplo, compq, n, d, e, u, ldu, vt, ldvt, q,   \
                          iq);                                                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##bdsdc_work(                              \
      int layout, char uplo, char compq, lapack_int n, R* d, R* e, R* u,       \
      lapack_int ldu, R* vt, lapack_int ldvt, R* q, lapack_int* iq, R* work,   \
      lapack_int* iwork) {                                                     \
    return lapacke::BdsdcWork(layout, uplo, compq, n, d, e, u, ldu, vt, ldvt,  \
                              q, iq, work, iwork);                             \
  }

HWYBLAS_LAPACKE_4_REAL(s, float)
HWYBLAS_LAPACKE_4_REAL(d, double)

#undef HWYBLAS_LAPACKE_4_REAL

#define HWYBLAS_LAPACKE_4_COMPLEX(p, T, R)                                     \
  HWYBLAS_API lapack_int LAPACKE_##p##gesvd_work(                              \
      int layout, char jobu, char jobvt, lapack_int m, lapack_int n, T* a,     \
      lapack_int lda, R* s, T* u, lapack_int ldu, T* vt, lapack_int ldvt,      \
      T* work, lapack_int lwork, R* rwork) {                                   \
    return lapacke::GesvdWork(layout, jobu, jobvt, m, n, a, lda, s, u, ldu,    \
                              vt, ldvt, work, lwork, rwork);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesdd(                                   \
      int layout, char jobz, lapack_int m, lapack_int n, T* a, lapack_int lda, \
      R* s, T* u, lapack_int ldu, T* vt, lapack_int ldvt) {                    \
    return lapacke::Gesdd(layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesdd_work(                              \
      int layout, char jobz, lapack_int m, lapack_int n, T* a, lapack_int lda, \
      R* s, T* u, lapack_int ldu, T* vt, lapack_int ldvt, T* work,             \
      lapack_int lwork, R* rwork, lapack_int* iwork) {                         \
    return lapacke::GesddWork(layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, \
                              work, lwork, rwork, iwork);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacp2(                                   \
      int layout, char uplo, lapack_int m, lapack_int n, const R* a,           \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::Lacp2(layout, uplo, m, n, a, lda, b, ldb);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacp2_work(                              \
      int layout, char uplo, lapack_int m, lapack_int n, const R* a,           \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::Lacp2Work(layout, uplo, m, n, a, lda, b, ldb);             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacrm(                                   \
      int layout, lapack_int m, lapack_int n, const T* a, lapack_int lda,      \
      const R* b, lapack_int ldb, T* c, lapack_int ldc) {                      \
    return lapacke::RealComplexProduct<T, false>(layout, m, n, a, lda, b, ldb, \
                                                 c, ldc);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacrm_work(                              \
      int layout, lapack_int m, lapack_int n, const T* a, lapack_int lda,      \
      const R* b, lapack_int ldb, T* c, lapack_int ldc, R* rwork) {            \
    return lapacke::RealComplexProductWork<T, false>(layout, m, n, a, lda, b,  \
                                                     ldb, c, ldc, rwork);      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larcm(                                   \
      int layout, lapack_int m, lapack_int n, const R* a, lapack_int lda,      \
      const T* b, lapack_int ldb, T* c, lapack_int ldc) {                      \
    return lapacke::RealComplexProduct<T, true>(layout, m, n, a, lda, b, ldb,  \
                                                c, ldc);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larcm_work(                              \
      int layout, lapack_int m, lapack_int n, const R* a, lapack_int lda,      \
      const T* b, lapack_int ldb, T* c, lapack_int ldc, R* rwork) {            \
    return lapacke::RealComplexProductWork<T, true>(layout, m, n, a, lda, b,   \
                                                    ldb, c, ldc, rwork);       \
  }

HWYBLAS_LAPACKE_4_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_4_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_4_COMPLEX

// Nonsymmetric eigenproblem: the functions with the same arguments in all
// precisions; `og` and `om` are orghr / unghr and ormhr / unmhr (without
// the precision letter).
#define HWYBLAS_LAPACKE_5(p, T, R, og, om)                                     \
  HWYBLAS_API lapack_int LAPACKE_##p##gebal(                                   \
      int layout, char job, lapack_int n, T* a, lapack_int lda,                \
      lapack_int* ilo, lapack_int* ihi, R* scale) {                            \
    return lapacke::Gebal(layout, job, n, a, lda, ilo, ihi, scale);            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gebal_work(                              \
      int layout, char job, lapack_int n, T* a, lapack_int lda,                \
      lapack_int* ilo, lapack_int* ihi, R* scale) {                            \
    return lapacke::GebalWork(layout, job, n, a, lda, ilo, ihi, scale);        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gebak(                                   \
      int layout, char job, char side, lapack_int n, lapack_int ilo,           \
      lapack_int ihi, const R* scale, lapack_int m, T* v, lapack_int ldv) {    \
    return lapacke::Gebak(layout, job, side, n, ilo, ihi, scale, m, v, ldv);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gebak_work(                              \
      int layout, char job, char side, lapack_int n, lapack_int ilo,           \
      lapack_int ihi, const R* scale, lapack_int m, T* v, lapack_int ldv) {    \
    return lapacke::GebakWork(layout, job, side, n, ilo, ihi, scale, m, v,     \
                              ldv);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gehrd(int layout, lapack_int n,          \
                                            lapack_int ilo, lapack_int ihi,    \
                                            T* a, lapack_int lda, T* tau) {    \
    return lapacke::Gehrd(layout, n, ilo, ihi, a, lda, tau);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gehrd_work(                              \
      int layout, lapack_int n, lapack_int ilo, lapack_int ihi, T* a,          \
      lapack_int lda, T* tau, T* work, lapack_int lwork) {                     \
    return lapacke::GehrdWork(layout, n, ilo, ihi, a, lda, tau, work, lwork);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##hr(                                  \
      int layout, lapack_int n, lapack_int ilo, lapack_int ihi, T* a,          \
      lapack_int lda, const T* tau) {                                          \
    return lapacke::Unghr(layout, n, ilo, ihi, a, lda, tau);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##og##hr_work(                             \
      int layout, lapack_int n, lapack_int ilo, lapack_int ihi, T* a,          \
      lapack_int lda, const T* tau, T* work, lapack_int lwork) {               \
    return lapacke::UnghrWork(layout, n, ilo, ihi, a, lda, tau, work, lwork);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##om##hr(                                  \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int ilo, lapack_int ihi, const T* a, lapack_int lda,              \
      const T* tau, T* c, lapack_int ldc) {                                    \
    return lapacke::Unmhr(layout, side, trans, m, n, ilo, ihi, a, lda, tau, c, \
                          ldc);                                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##om##hr_work(                             \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int ilo, lapack_int ihi, const T* a, lapack_int lda,              \
      const T* tau, T* c, lapack_int ldc, T* work, lapack_int lwork) {         \
    return lapacke::UnmhrWork(layout, side, trans, m, n, ilo, ihi, a, lda,     \
                              tau, c, ldc, work, lwork);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsyl(                                   \
      int layout, char trana, char tranb, lapack_int isgn, lapack_int m,       \
      lapack_int n, const T* a, lapack_int lda, const T* b, lapack_int ldb,    \
      T* c, lapack_int ldc, R* scale) {                                        \
    return lapacke::Trsyl(layout, trana, tranb, isgn, m, n, a, lda, b, ldb, c, \
                          ldc, scale);                                         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsyl_work(                              \
      int layout, char trana, char tranb, lapack_int isgn, lapack_int m,       \
      lapack_int n, const T* a, lapack_int lda, const T* b, lapack_int ldb,    \
      T* c, lapack_int ldc, R* scale) {                                        \
    return lapacke::TrsylWork(layout, trana, tranb, isgn, m, n, a, lda, b,     \
                              ldb, c, ldc, scale);                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larfx(                                   \
      int layout, char side, lapack_int m, lapack_int n, const T* v, T tau,    \
      T* c, lapack_int ldc, T* work) {                                         \
    return lapacke::Larfx(layout, side, m, n, v, tau, c, ldc, work);           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larfx_work(                              \
      int layout, char side, lapack_int m, lapack_int n, const T* v, T tau,    \
      T* c, lapack_int ldc, T* work) {                                         \
    return lapacke::LarfxWork(layout, side, m, n, v, tau, c, ldc, work);       \
  }

HWYBLAS_LAPACKE_5(s, float, float, org, orm)
HWYBLAS_LAPACKE_5(d, double, double, org, orm)
HWYBLAS_LAPACKE_5(c, c32, float, ung, unm)
HWYBLAS_LAPACKE_5(z, c64, double, ung, unm)

#undef HWYBLAS_LAPACKE_5

#define HWYBLAS_LAPACKE_5_REAL(p, R, P)                                        \
  HWYBLAS_API lapack_int LAPACKE_##p##hseqr(                                   \
      int layout, char job, char compz, lapack_int n, lapack_int ilo,          \
      lapack_int ihi, R* h, lapack_int ldh, R* wr, R* wi, R* z,                \
      lapack_int ldz) {                                                        \
    return lapacke::Hseqr(layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, z,  \
                          ldz);                                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hseqr_work(                              \
      int layout, char job, char compz, lapack_int n, lapack_int ilo,          \
      lapack_int ihi, R* h, lapack_int ldh, R* wr, R* wi, R* z,                \
      lapack_int ldz, R* work, lapack_int lwork) {                             \
    return lapacke::HseqrWork(layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, \
                              z, ldz, work, lwork);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trexc(                                   \
      int layout, char compq, lapack_int n, R* t, lapack_int ldt, R* q,        \
      lapack_int ldq, lapack_int* ifst, lapack_int* ilst) {                    \
    return lapacke::Trexc(layout, compq, n, t, ldt, q, ldq, ifst, ilst);       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trexc_work(                              \
      int layout, char compq, lapack_int n, R* t, lapack_int ldt, R* q,        \
      lapack_int ldq, lapack_int* ifst, lapack_int* ilst, R* work) {           \
    return lapacke::TrexcWork(layout, compq, n, t, ldt, q, ldq, ifst, ilst,    \
                              work);                                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsen(                                   \
      int layout, char job, char compq, const lapack_logical* select,          \
      lapack_int n, R* t, lapack_int ldt, R* q, lapack_int ldq, R* wr, R* wi,  \
      lapack_int* m, R* s, R* sep) {                                           \
    return lapacke::Trsen(layout, job, compq, select, n, t, ldt, q, ldq, wr,   \
                          wi, m, s, sep);                                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsen_work(                              \
      int layout, char job, char compq, const lapack_logical* select,          \
      lapack_int n, R* t, lapack_int ldt, R* q, lapack_int ldq, R* wr, R* wi,  \
      lapack_int* m, R* s, R* sep, R* work, lapack_int lwork,                  \
      lapack_int* iwork, lapack_int liwork) {                                  \
    return lapacke::TrsenWork(layout, job, compq, select, n, t, ldt, q, ldq,   \
                              wr, wi, m, s, sep, work, lwork, iwork, liwork);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacn2(                                   \
      lapack_int n, R* v, R* x, lapack_int* isgn, R* est, lapack_int* kase,    \
      lapack_int* isave) {                                                     \
    return lapacke::Lacn2(n, v, x, isgn, est, kase, isave);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacn2_work(                              \
      lapack_int n, R* v, R* x, lapack_int* isgn, R* est, lapack_int* kase,    \
      lapack_int* isave) {                                                     \
    return lapacke::Lacn2(n, v, x, isgn, est, kase, isave);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gees(                                    \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT2 select,          \
      lapack_int n, R* a, lapack_int lda, lapack_int* sdim, R* wr, R* wi,      \
      R* vs, lapack_int ldvs) {                                                \
    return lapacke::Gees(layout, jobvs, sort, select, n, a, lda, sdim, wr, wi, \
                         vs, ldvs);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gees_work(                               \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT2 select,          \
      lapack_int n, R* a, lapack_int lda, lapack_int* sdim, R* wr, R* wi,      \
      R* vs, lapack_int ldvs, R* work, lapack_int lwork,                       \
      lapack_logical* bwork) {                                                 \
    return lapacke::GeesWork(layout, jobvs, sort, select, n, a, lda, sdim, wr, \
                             wi, vs, ldvs, work, lwork,                        \
                             static_cast<R*>(nullptr), bwork);                 \
  }

HWYBLAS_LAPACKE_5_REAL(s, float, S)
HWYBLAS_LAPACKE_5_REAL(d, double, D)

#undef HWYBLAS_LAPACKE_5_REAL

#define HWYBLAS_LAPACKE_5_COMPLEX(p, T, R, P)                                  \
  HWYBLAS_API lapack_int LAPACKE_##p##hseqr(                                   \
      int layout, char job, char compz, lapack_int n, lapack_int ilo,          \
      lapack_int ihi, T* h, lapack_int ldh, T* w, T* z, lapack_int ldz) {      \
    return lapacke::Hseqr(layout, job, compz, n, ilo, ihi, h, ldh, w,          \
                          static_cast<T*>(nullptr), z, ldz);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hseqr_work(                              \
      int layout, char job, char compz, lapack_int n, lapack_int ilo,          \
      lapack_int ihi, T* h, lapack_int ldh, T* w, T* z, lapack_int ldz,        \
      T* work, lapack_int lwork) {                                             \
    return lapacke::HseqrWork(layout, job, compz, n, ilo, ihi, h, ldh, w,      \
                              static_cast<T*>(nullptr), z, ldz, work, lwork);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trexc(                                   \
      int layout, char compq, lapack_int n, T* t, lapack_int ldt, T* q,        \
      lapack_int ldq, lapack_int ifst, lapack_int ilst) {                      \
    return lapacke::Trexc(layout, compq, n, t, ldt, q, ldq, &ifst, &ilst);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trexc_work(                              \
      int layout, char compq, lapack_int n, T* t, lapack_int ldt, T* q,        \
      lapack_int ldq, lapack_int ifst, lapack_int ilst) {                      \
    return lapacke::TrexcWork(layout, compq, n, t, ldt, q, ldq, &ifst, &ilst,  \
                              static_cast<T*>(nullptr));                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsen(                                   \
      int layout, char job, char compq, const lapack_logical* select,          \
      lapack_int n, T* t, lapack_int ldt, T* q, lapack_int ldq, T* w,          \
      lapack_int* m, R* s, R* sep) {                                           \
    return lapacke::Trsen(layout, job, compq, select, n, t, ldt, q, ldq, w,    \
                          static_cast<T*>(nullptr), m, s, sep);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsen_work(                              \
      int layout, char job, char compq, const lapack_logical* select,          \
      lapack_int n, T* t, lapack_int ldt, T* q, lapack_int ldq, T* w,          \
      lapack_int* m, R* s, R* sep, T* work, lapack_int lwork) {                \
    return lapacke::TrsenWork(layout, job, compq, select, n, t, ldt, q, ldq,   \
                              w, static_cast<T*>(nullptr), m, s, sep, work,    \
                              lwork, static_cast<lapack_int*>(nullptr), 0);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacn2(                                   \
      lapack_int n, T* v, T* x, R* est, lapack_int* kase, lapack_int* isave) { \
    return lapacke::Lacn2(n, v, x, static_cast<lapack_int*>(nullptr), est,     \
                          kase, isave);                                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lacn2_work(                              \
      lapack_int n, T* v, T* x, R* est, lapack_int* kase, lapack_int* isave) { \
    return lapacke::Lacn2(n, v, x, static_cast<lapack_int*>(nullptr), est,     \
                          kase, isave);                                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gees(                                    \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT1 select,          \
      lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* w, T* vs,       \
      lapack_int ldvs) {                                                       \
    return lapacke::Gees(layout, jobvs, sort, select, n, a, lda, sdim, w,      \
                         static_cast<T*>(nullptr), vs, ldvs);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gees_work(                               \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT1 select,          \
      lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* w, T* vs,       \
      lapack_int ldvs, T* work, lapack_int lwork, R* rwork,                    \
      lapack_logical* bwork) {                                                 \
    return lapacke::GeesWork(layout, jobvs, sort, select, n, a, lda, sdim, w,  \
                             static_cast<T*>(nullptr), vs, ldvs, work, lwork,  \
                             rwork, bwork);                                    \
  }

HWYBLAS_LAPACKE_5_COMPLEX(c, c32, float, C)
HWYBLAS_LAPACKE_5_COMPLEX(z, c64, double, Z)

#undef HWYBLAS_LAPACKE_5_COMPLEX

}  // extern "C"
