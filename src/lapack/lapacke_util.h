// The shared parts of the LAPACKE functions (lapacke*.cc): the reference
// LAPACKE's utilities (NaN checks, transposes) as templates over the four
// precisions, and the helpers that do what all functions do:
//  - CallWork: the body of the LAPACKE_xyyzzz_work functions. In row-major
//    layout, it checks the leading dimensions, copies the matrices into
//    column-major temporaries (transposing them) and back, and adjusts INFO.
//  - WithWorkspace and WithWorkArray: the LAPACKE_xyyzzz functions of
//    routines with a workspace: a workspace query (for LWORK), then the
//    allocation.
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#ifndef HWYBLAS_SRC_LAPACK_LAPACKE_UTIL_H_
#define HWYBLAS_SRC_LAPACK_LAPACKE_UTIL_H_

#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>

#include "hwyblas/lapacke.h"
#include "src/lapack/lapack_common.h"

namespace hwyblas {
namespace lapacke {

using lapack::RealOrComplex;
using lapack::RoutineName;

// ---------------------------------------------------------------------------
// Utilities (LAPACKE's utils/)

inline bool ValidLayout(int matrix_layout) {
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

// LAPACKE_xtz_nancheck: whether the m x n trapezoidal matrix A contains a
// NaN: its triangle uplo (without the diagonal for diag = 'U') at the front
// (direct = 'F') or back ('B'), and the rectangular rest. Returns false for
// invalid arguments.
template <typename T>
bool TzNancheck(int matrix_layout, char direct, char uplo, char diag,
                lapack_int m, lapack_int n, const T* a, lapack_int lda) {
  if (a == nullptr) return false;
  const bool colmaj = (matrix_layout == LAPACK_COL_MAJOR);
  const bool front = lapack::Lsame(direct, 'F');
  const bool lower = lapack::Lsame(uplo, 'L');
  const bool unit = lapack::Lsame(diag, 'U');
  if ((!colmaj && matrix_layout != LAPACK_ROW_MAJOR) ||
      (!front && !lapack::Lsame(direct, 'B')) ||
      (!lower && !lapack::Lsame(uplo, 'U')) ||
      (!unit && !lapack::Lsame(diag, 'N'))) {
    // Just exit if any of input parameters are wrong.
    return false;
  }
  // Initial offsets and sizes of triangular and rectangular parts.
  const lapack_int tri_n = std::min(m, n);
  const lapack_int rect_m = m > n ? m - n : m;
  const lapack_int rect_n = n > m ? n - m : n;
  // Offset of k rows (along_ld false: k elements, else k leading dimensions).
  const auto offset = [lda](lapack_int k, bool along_ld) {
    return static_cast<ptrdiff_t>(k) * (along_ld ? lda : 1);
  };
  ptrdiff_t tri_offset = 0;
  ptrdiff_t rect_offset = -1;
  // Fix offsets depending on the shape of the matrix.
  if (front) {
    if (lower && m > n) {
      rect_offset = offset(tri_n, !colmaj);
    } else if (!lower && n > m) {
      rect_offset = offset(tri_n, colmaj);
    }
  } else {
    if (m > n) {
      tri_offset = offset(rect_m, !colmaj);
      if (!lower) rect_offset = 0;
    } else if (n > m) {
      tri_offset = offset(rect_n, colmaj);
      if (lower) rect_offset = 0;
    }
  }
  // Check rectangular part.
  if (rect_offset >= 0 &&
      GeNancheck(matrix_layout, rect_m, rect_n, a + rect_offset, lda)) {
    return true;
  }
  // Check triangular part.
  return TrNancheck(matrix_layout, uplo, diag, tri_n, a + tri_offset, lda);
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

inline void Xerbla(const std::string& name, lapack_int info) {
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
  // Whether all of the matrix is copied back, also with uplo 'U' or 'L'
  // (e.g. the eigenvectors of xSYEV that overwrite A).
  bool out_general = false;
  // false: the routine does not reference the matrix (e.g. Z of xSTEQR for
  // COMPZ = 'N'), which then gets no temporary (a null pointer).
  bool used = true;
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
  const bool back = matrix_layout == LAPACK_COL_MAJOR;
  if ((arg.uplo == 'U' || arg.uplo == 'L') && !(back && arg.out_general)) {
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
    if (!args[i].used) continue;
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
    if (args[i].used && args[i].in) {
      Transpose(LAPACK_ROW_MAJOR, args[i], args[i].data, args[i].ld, a[i],
                ld[i]);
    }
  }
  // Call LAPACK function and adjust info.
  lapack_int info = lapack(a, ld);
  if (info < 0) info = info - 1;
  // Transpose output matrices.
  for (size_t i = 0; i < N; ++i) {
    if (args[i].used && args[i].out) {
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

// Allocates the work arrays of a routine whose real version takes WORK
// (real_size elements) and IWORK (int_size), and its complex version WORK
// (complex_size) and RWORK (rwork_size), and calls f(work, rwork, iwork)
// with null pointers for the arrays that T's routine does not take.
template <typename T, typename F>
lapack_int WithWork(const std::string& name, lapack_int real_size,
                    lapack_int int_size, lapack_int complex_size,
                    lapack_int rwork_size, const F& f) {
  using R = RealT<T>;
  if constexpr (IsComplex<T>::value) {
    return WithWorkArray<R>(
        name, std::max<lapack_int>(1, rwork_size), [&](R* rwork) {
          return WithWorkArray<T>(
              name, std::max<lapack_int>(1, complex_size), [&](T* work) {
                return f(work, rwork, static_cast<lapack_int*>(nullptr));
              });
        });
  } else {
    return WithWorkArray<lapack_int>(
        name, std::max<lapack_int>(1, int_size), [&](lapack_int* iwork) {
          return WithWorkArray<T>(
              name, std::max<lapack_int>(1, real_size), [&](T* work) {
                return f(work, static_cast<R*>(nullptr), iwork);
              });
        });
  }
}

// The rest of a LAPACKE_xyyzzz function `name` for a routine with WORK,
// IWORK and, for complex T, RWORK, whose sizes it returns in a workspace
// query: work(work, lwork, rwork, lrwork, iwork, liwork) calls
// LAPACKE_xyyzzz_work. Queries the sizes, allocates the arrays and calls
// work (for real T with a null RWORK).
template <typename T, typename Work>
lapack_int WithWorkspaces(const std::string& name, const Work& work) {
  using R = RealT<T>;
  // Query optimal working array(s) size.
  T work_query = T(0);
  R rwork_query = R(0);
  lapack_int iwork_query = 0;
  lapack_int info =
      work(&work_query, -1, &rwork_query, -1, &iwork_query, lapack_int{-1});
  if (info != 0) return info;
  const lapack_int liwork = iwork_query;
  const lapack_int lrwork = static_cast<lapack_int>(rwork_query);
  const lapack_int lwork = static_cast<lapack_int>(std::real(work_query));
  // Allocate memory for work arrays and call middle-level interface.
  return WithWorkArray<lapack_int>(name, liwork, [&](lapack_int* iwork) {
    if constexpr (IsComplex<T>::value) {
      return WithWorkArray<R>(name, lrwork, [&](R* rwork) {
        return WithWorkArray<T>(name, lwork, [&](T* w) {
          return work(w, lwork, rwork, lrwork, iwork, liwork);
        });
      });
    } else {
      return WithWorkArray<T>(name, lwork, [&](T* w) {
        return work(w, lwork, static_cast<R*>(nullptr), lapack_int{0}, iwork,
                    liwork);
      });
    }
  });
}

// A triangular matrix argument: only the triangle uplo is transposed.
template <typename T>
MatrixArg<T> Triangle(MatrixArg<T> arg, char uplo) {
  arg.uplo = lapack::Lsame(uplo, 'U') ? 'U' : 'L';
  return arg;
}

// Whether the NaN checks are on (LAPACKE_get_nancheck).
inline bool NanCheck() { return LAPACKE_get_nancheck() != 0; }

}  // namespace lapacke
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_LAPACKE_UTIL_H_
