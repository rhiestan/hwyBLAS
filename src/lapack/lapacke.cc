// LAPACKE: the C interface to the LAPACK routines (see hwyblas/lapacke.h).
// Ported from the reference LAPACKE (LAPACK 3.12.1), with the same checks,
// INFO values and error messages. The reference repeats the same code for
// each routine and precision; here the utilities (NaN checks, transposes)
// are templates over the four precisions, and two helpers do what all
// functions do:
//  - CallWork: the LAPACKE_xyyzzz_work functions. In row-major layout, it
//    checks the leading dimensions, copies the matrices into column-major
//    temporaries (transposing them) and back, and adjusts INFO.
//  - WithWorkspace: the LAPACKE_xyyzzz functions of routines with a
//    workspace (LWORK): a workspace query, then the allocation.
// The functions call the ported routines directly (which report invalid
// arguments, like the reference's Fortran routines).
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

namespace hwyblas {
namespace lapacke {
namespace {

using lapack::RealOrComplex;
using lapack::RoutineName;

// -1: not yet determined (see LAPACKE_get_nancheck).
std::atomic<int> g_nancheck{-1};

// ---------------------------------------------------------------------------
// Utilities (LAPACKE's utils/)

bool ValidLayout(int matrix_layout) {
  return matrix_layout == LAPACK_COL_MAJOR || matrix_layout == LAPACK_ROW_MAJOR;
}

template <typename T>
bool IsNan(const T& x) {
  if constexpr (IsComplex<T>::value) {
    return std::isnan(x.real()) || std::isnan(x.imag());
  } else {
    return std::isnan(x);
  }
}

// LAPACKE_x_nancheck: whether the vector x (n elements with increment incx;
// only x[0] for incx = 0) contains a NaN.
template <typename T>
bool VectorNancheck(lapack_int n, const T* x, lapack_int incx) {
  if (incx == 0) return IsNan(x[0]);
  const lapack_int inc = incx > 0 ? incx : -incx;
  for (lapack_int i = 0; i < n * inc; i += inc) {
    if (IsNan(x[i])) return true;
  }
  return false;
}

// LAPACKE_xge_nancheck: whether the m x n matrix A (in the given layout)
// contains a NaN. Like the reference, it looks at most at lda rows (columns
// for row-major A).
template <typename T>
bool GeNancheck(int matrix_layout, lapack_int m, lapack_int n, const T* a,
                lapack_int lda) {
  if (a == nullptr) return false;
  if (matrix_layout == LAPACK_COL_MAJOR) {
    for (lapack_int j = 0; j < n; ++j) {
      for (lapack_int i = 0; i < std::min(m, lda); ++i) {
        if (IsNan(a[i + static_cast<size_t>(j) * lda])) return true;
      }
    }
  } else if (matrix_layout == LAPACK_ROW_MAJOR) {
    for (lapack_int i = 0; i < m; ++i) {
      for (lapack_int j = 0; j < std::min(n, lda); ++j) {
        if (IsNan(a[static_cast<size_t>(i) * lda + j])) return true;
      }
    }
  }
  return false;
}

// LAPACKE_xtr_nancheck: whether the triangle uplo ('U' or 'L') of the
// n x n matrix A contains a NaN; without the diagonal for diag = 'U'.
// Returns false for invalid arguments.
template <typename T>
bool TrNancheck(int matrix_layout, char uplo, char diag, lapack_int n,
                const T* a, lapack_int lda) {
  if (a == nullptr) return false;
  const bool colmaj = (matrix_layout == LAPACK_COL_MAJOR);
  const bool lower = lapack::Lsame(uplo, 'L');
  const bool unit = lapack::Lsame(diag, 'U');
  if ((!colmaj && matrix_layout != LAPACK_ROW_MAJOR) ||
      (!lower && !lapack::Lsame(uplo, 'U')) ||
      (!unit && !lapack::Lsame(diag, 'N'))) {
    // Just exit if any of input parameters are wrong.
    return false;
  }
  // If unit, then don't touch diagonal, start from 1st column or row.
  const lapack_int st = unit ? 1 : 0;
  // Column-major upper and row-major lower are the same, and so are
  // column-major lower and row-major upper.
  if (colmaj != lower) {
    for (lapack_int j = st; j < n; ++j) {
      for (lapack_int i = 0; i < std::min(j + 1 - st, lda); ++i) {
        if (IsNan(a[i + static_cast<size_t>(j) * lda])) return true;
      }
    }
  } else {
    for (lapack_int j = 0; j < n - st; ++j) {
      for (lapack_int i = j + st; i < std::min(n, lda); ++i) {
        if (IsNan(a[i + static_cast<size_t>(j) * lda])) return true;
      }
    }
  }
  return false;
}

// LAPACKE_xpo_nancheck: the triangle uplo of the symmetric (Hermitian)
// matrix A.
template <typename T>
bool PoNancheck(int matrix_layout, char uplo, lapack_int n, const T* a,
                lapack_int lda) {
  return TrNancheck(matrix_layout, uplo, 'N', n, a, lda);
}

// LAPACKE_xge_trans: copies the transpose of the m x n matrix `in` (in the
// given layout) to `out`, which then holds it in the other layout. In case
// of incorrect m, n, ldin or ldout it does nothing (or copies less).
template <typename T>
void GeTrans(int matrix_layout, lapack_int m, lapack_int n, const T* in,
             lapack_int ldin, T* out, lapack_int ldout) {
  if (in == nullptr || out == nullptr) return;
  lapack_int x, y;
  if (matrix_layout == LAPACK_COL_MAJOR) {
    x = n;
    y = m;
  } else if (matrix_layout == LAPACK_ROW_MAJOR) {
    x = m;
    y = n;
  } else {
    return;  // unknown input layout
  }
  for (lapack_int i = 0; i < std::min(y, ldin); ++i) {
    for (lapack_int j = 0; j < std::min(x, ldout); ++j) {
      out[static_cast<size_t>(i) * ldout + j] =
          in[static_cast<size_t>(j) * ldin + i];
    }
  }
}

// LAPACKE_xtr_trans: like GeTrans, for the triangle uplo of the n x n
// matrix `in` (without the diagonal for diag = 'U'). Does nothing for
// invalid arguments.
template <typename T>
void TrTrans(int matrix_layout, char uplo, char diag, lapack_int n, const T* in,
             lapack_int ldin, T* out, lapack_int ldout) {
  if (in == nullptr || out == nullptr) return;
  const bool colmaj = (matrix_layout == LAPACK_COL_MAJOR);
  const bool lower = lapack::Lsame(uplo, 'L');
  const bool unit = lapack::Lsame(diag, 'U');
  if ((!colmaj && matrix_layout != LAPACK_ROW_MAJOR) ||
      (!lower && !lapack::Lsame(uplo, 'U')) ||
      (!unit && !lapack::Lsame(diag, 'N'))) {
    // Just exit if any of input parameters are wrong.
    return;
  }
  // If unit, then don't touch diagonal, start from 1st column or row.
  const lapack_int st = unit ? 1 : 0;
  // Column-major upper and row-major lower are the same, and so are
  // column-major lower and row-major upper.
  if (colmaj != lower) {
    for (lapack_int j = st; j < std::min(n, ldout); ++j) {
      for (lapack_int i = 0; i < std::min(j + 1 - st, ldin); ++i) {
        out[j + static_cast<size_t>(i) * ldout] =
            in[i + static_cast<size_t>(j) * ldin];
      }
    }
  } else {
    for (lapack_int j = 0; j < std::min(n - st, ldout); ++j) {
      for (lapack_int i = j + st; i < std::min(n, ldin); ++i) {
        out[j + static_cast<size_t>(i) * ldout] =
            in[i + static_cast<size_t>(j) * ldin];
      }
    }
  }
}

// The function names of the error messages, e.g. Name<double>("getrf") =
// "LAPACKE_dgetrf".
template <typename T>
std::string Name(const char* rest) {
  std::string name = RoutineName::Of<T>(rest).c_str();
  for (char& c : name) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return "LAPACKE_" + name;
}

void Xerbla(const std::string& name, lapack_int info) {
  LAPACKE_xerbla(name.c_str(), info);
}

// ---------------------------------------------------------------------------
// The common parts of the functions

// A matrix argument of a LAPACKE_xyyzzz_work function. In row-major layout,
// the LAPACK routine gets a column-major copy (rows x cols, as the routine
// sees the matrix).
template <typename T>
struct MatrixArg {
  T* data;
  lapack_int rows;
  lapack_int cols;
  lapack_int ld;
  // INFO (minus the position of ld) if ld < cols in row-major layout.
  lapack_int ld_info;
  // Whether the routine reads the matrix (it is copied in) and writes it
  // (it is copied back). A matrix that is only read may be const; its
  // const is cast away only to keep one type for all arguments.
  bool in = true;
  bool out = true;
  // 'U' or 'L': only this triangle of the n x n matrix is copied
  // (LAPACKE_xtr_trans), as for xPOTRF; otherwise all of it.
  char uplo = 'G';
};

// A matrix that the routine only reads.
template <typename T>
MatrixArg<T> Input(const T* a, lapack_int rows, lapack_int cols, lapack_int ld,
                   lapack_int ld_info) {
  return {const_cast<T*>(a), rows, cols, ld, ld_info, true, false};
}

template <typename T>
void Transpose(int matrix_layout, const MatrixArg<T>& arg, const T* in,
               lapack_int ldin, T* out, lapack_int ldout) {
  if (arg.uplo == 'U' || arg.uplo == 'L') {
    TrTrans(matrix_layout, arg.uplo, 'N', arg.rows, in, ldin, out, ldout);
  } else {
    GeTrans(matrix_layout, arg.rows, arg.cols, in, ldin, out, ldout);
  }
}

// The body of a LAPACKE_xyyzzz_work function `name`: calls
// lapack(a, ld), which calls the LAPACK routine with the matrices a[i] and
// their leading dimensions ld[i] and returns INFO, and adjusts INFO (minus
// one for an invalid argument: matrix_layout is the first one).
//
// In row-major layout, it first checks the leading dimensions, then copies
// the matrices into column-major temporaries, and back after the call. A
// workspace query (`query`) gets the caller's matrices (with the leading
// dimensions of the temporaries), as in the reference.
template <typename T, size_t N, typename Lapack>
lapack_int CallWork(int matrix_layout, const std::string& name, bool query,
                    const std::array<MatrixArg<T>, N>& args,
                    const Lapack& lapack) {
  std::array<T*, N> a{};
  std::array<lapack_int, N> ld{};
  if (matrix_layout == LAPACK_COL_MAJOR) {
    // Call LAPACK function and adjust info.
    for (size_t i = 0; i < N; ++i) {
      a[i] = args[i].data;
      ld[i] = args[i].ld;
    }
    const lapack_int info = lapack(a, ld);
    return info < 0 ? info - 1 : info;
  }
  if (matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(name, -1);
    return -1;
  }

  // Check leading dimension(s).
  for (const MatrixArg<T>& arg : args) {
    if (arg.ld < arg.cols) {
      Xerbla(name, arg.ld_info);
      return arg.ld_info;
    }
  }
  for (size_t i = 0; i < N; ++i) {
    ld[i] = std::max<lapack_int>(1, args[i].rows);
  }
  // Query optimal working array(s) size if requested.
  if (query) {
    for (size_t i = 0; i < N; ++i) a[i] = args[i].data;
    const lapack_int info = lapack(a, ld);
    return info < 0 ? info - 1 : info;
  }

  // Allocate memory for temporary array(s).
  const auto free_all = [&a] {
    for (T* p : a) free(p);
  };
  for (size_t i = 0; i < N; ++i) {
    a[i] = static_cast<T*>(malloc(sizeof(T) * static_cast<size_t>(ld[i]) *
                                  std::max<lapack_int>(1, args[i].cols)));
    if (a[i] == nullptr) {
      free_all();
      Xerbla(name, LAPACK_TRANSPOSE_MEMORY_ERROR);
      return LAPACK_TRANSPOSE_MEMORY_ERROR;
    }
  }
  // Transpose input matrices.
  for (size_t i = 0; i < N; ++i) {
    if (args[i].in) {
      Transpose(LAPACK_ROW_MAJOR, args[i], args[i].data, args[i].ld, a[i],
                ld[i]);
    }
  }
  // Call LAPACK function and adjust info.
  lapack_int info = lapack(a, ld);
  if (info < 0) info = info - 1;
  // Transpose output matrices.
  for (size_t i = 0; i < N; ++i) {
    if (args[i].out) {
      Transpose(LAPACK_COL_MAJOR, args[i], a[i], ld[i], args[i].data,
                args[i].ld);
    }
  }
  // Release memory and exit.
  free_all();
  return info;
}

// The rest of a LAPACKE_xyyzzz function `name` after its NaN checks, for a
// routine with a workspace: work(work, lwork) calls LAPACKE_xyyzzz_work.
// Queries the optimal workspace size, allocates it and calls work.
template <typename T, typename Work>
lapack_int WithWorkspace(const std::string& name, const Work& work) {
  // Query optimal working array(s) size.
  T work_query = T(0);
  lapack_int info = work(&work_query, -1);
  if (info != 0) return info;
  const lapack_int lwork = static_cast<lapack_int>(std::real(work_query));
  // Allocate memory for work arrays.
  T* buffer = static_cast<T*>(malloc(sizeof(T) * static_cast<size_t>(lwork)));
  if (buffer == nullptr) {
    Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
    return LAPACK_WORK_MEMORY_ERROR;
  }
  // Call middle-level interface.
  info = work(buffer, lwork);
  // Release memory and exit.
  free(buffer);
  return info;
}

// The rest of a LAPACKE_xyyzzz function `name` for a routine with a work
// array of `size` elements (e.g. xGEQR2): allocates it and calls
// work(work).
template <typename T, typename Work>
lapack_int WithWorkArray(const std::string& name, lapack_int size,
                         const Work& work) {
  // Allocate memory for working array(s).
  T* buffer = static_cast<T*>(malloc(sizeof(T) * static_cast<size_t>(size)));
  if (buffer == nullptr) {
    Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
    return LAPACK_WORK_MEMORY_ERROR;
  }
  // Call middle-level interface.
  const lapack_int info = work(buffer);
  // Release memory and exit.
  free(buffer);
  return info;
}

// Whether the NaN checks are on (LAPACKE_get_nancheck).
bool NanCheck() { return LAPACKE_get_nancheck() != 0; }

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

}  // extern "C"
