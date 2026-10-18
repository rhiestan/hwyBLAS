// LAPACKE: the C interface to the LAPACK routines (see hwyblas/lapacke.h).
// Ported from the reference LAPACKE (LAPACK 3.12.1), with the same checks,
// INFO values and error messages; the utilities (NaN checks, transposes) are
// templates over the four precisions.
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#include "hwyblas/lapacke.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>

#include "hwyblas/lapack.h"
#include "src/lapack/lapack_common.h"

namespace hwyblas {
namespace lapacke {
namespace {

using lapack::RoutineName;

// -1: not yet determined (see LAPACKE_get_nancheck).
std::atomic<int> g_nancheck{-1};

template <typename T>
bool IsNan(const T& x) {
  if constexpr (IsComplex<T>::value) {
    return std::isnan(x.real()) || std::isnan(x.imag());
  } else {
    return std::isnan(x);
  }
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

// "LAPACKE_dgetrf" etc.
template <typename T>
std::string Name(const char* rest) {
  return std::string("LAPACKE_") + RoutineName::Of<T>(rest).c_str();
}

void Xerbla(const std::string& name, lapack_int info) {
  LAPACKE_xerbla(name.c_str(), info);
}

// The Fortran-style routines take float/double pointers for complex
// matrices.
template <typename T>
auto* Raw(T* p) {
  return reinterpret_cast<RealT<T>*>(p);
}

// ---------------------------------------------------------------------------
// xGETRF

template <typename T>
void CallGetrf(lapack_int* m, lapack_int* n, T* a, lapack_int* lda,
               lapack_int* ipiv, lapack_int* info) {
  if constexpr (std::is_same_v<T, float>) {
    sgetrf_(m, n, a, lda, ipiv, info);
  } else if constexpr (std::is_same_v<T, double>) {
    dgetrf_(m, n, a, lda, ipiv, info);
  } else if constexpr (std::is_same_v<T, c32>) {
    cgetrf_(m, n, Raw(a), lda, ipiv, info);
  } else {
    zgetrf_(m, n, Raw(a), lda, ipiv, info);
  }
}

template <typename T>
lapack_int GetrfWork(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, lapack_int* ipiv) {
  lapack_int info = 0;
  if (matrix_layout == LAPACK_COL_MAJOR) {
    // Call LAPACK function and adjust info.
    CallGetrf(&m, &n, a, &lda, ipiv, &info);
    if (info < 0) info = info - 1;
  } else if (matrix_layout == LAPACK_ROW_MAJOR) {
    lapack_int lda_t = std::max<lapack_int>(1, m);
    // Check leading dimension(s).
    if (lda < n) {
      info = -5;
      Xerbla(Name<T>("GETRF_WORK"), info);
      return info;
    }
    // Allocate memory for temporary array(s).
    T* a_t =
        static_cast<T*>(malloc(sizeof(T) * lda_t * std::max<lapack_int>(1, n)));
    if (a_t == nullptr) {
      info = LAPACK_TRANSPOSE_MEMORY_ERROR;
      Xerbla(Name<T>("GETRF_WORK"), info);
      return info;
    }
    // Transpose input matrices.
    GeTrans(matrix_layout, m, n, a, lda, a_t, lda_t);
    // Call LAPACK function and adjust info.
    CallGetrf(&m, &n, a_t, &lda_t, ipiv, &info);
    if (info < 0) info = info - 1;
    // Transpose output matrices.
    GeTrans(LAPACK_COL_MAJOR, m, n, a_t, lda_t, a, lda);
    // Release memory and exit.
    free(a_t);
  } else {
    info = -1;
    Xerbla(Name<T>("GETRF_WORK"), info);
  }
  return info;
}

template <typename T>
lapack_int Getrf(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, lapack_int* ipiv) {
  if (matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(Name<T>("GETRF"), -1);
    return -1;
  }
  if (LAPACKE_get_nancheck()) {
    // Optionally check input matrices for NaNs.
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  }
  return GetrfWork(matrix_layout, m, n, a, lda, ipiv);
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;

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
  hwyblas::lapacke::g_nancheck.store(flag ? 1 : 0);
}

HWYBLAS_API int LAPACKE_get_nancheck(void) {
  const int flag = hwyblas::lapacke::g_nancheck.load();
  if (flag != -1) return flag;
  // Check the environment variable, once and only once. By default, NaN
  // checking is enabled.
#ifdef _MSC_VER
#pragma warning(suppress : 4996)  // getenv is safe here
#endif
  const char* env = getenv("LAPACKE_NANCHECK");
  const int value = (env == nullptr || atoi(env) != 0) ? 1 : 0;
  int expected = -1;
  hwyblas::lapacke::g_nancheck.compare_exchange_strong(expected, value);
  return hwyblas::lapacke::g_nancheck.load();
}

#define HWYBLAS_LAPACKE_GETRF(prefix, T)                                   \
  HWYBLAS_API lapack_int LAPACKE_##prefix##getrf(                          \
      int matrix_layout, lapack_int m, lapack_int n, T* a, lapack_int lda, \
      lapack_int* ipiv) {                                                  \
    return hwyblas::lapacke::Getrf(matrix_layout, m, n, a, lda, ipiv);     \
  }                                                                        \
  HWYBLAS_API lapack_int LAPACKE_##prefix##getrf_work(                     \
      int matrix_layout, lapack_int m, lapack_int n, T* a, lapack_int lda, \
      lapack_int* ipiv) {                                                  \
    return hwyblas::lapacke::GetrfWork(matrix_layout, m, n, a, lda, ipiv); \
  }

HWYBLAS_LAPACKE_GETRF(s, float)
HWYBLAS_LAPACKE_GETRF(d, double)
HWYBLAS_LAPACKE_GETRF(c, c32)
HWYBLAS_LAPACKE_GETRF(z, c64)

#undef HWYBLAS_LAPACKE_GETRF

}  // extern "C"
