// LAPACKE functions of the linear systems routines: xGETRS, xGESV, xGETRI,
// xGECON, xGEEQU, xGERFS, xGESVX, xTRTRI, xTRTRS, xTRCON, xTRRFS and
// xLANTR (see lapacke.cc and lapacke_util.h).
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#include <stdlib.h>

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
// LU factorization: solution and inverse

template <typename T>
lapack_int GetrsWork(int matrix_layout, char trans, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda,
                     const lapack_int* ipiv, T* b, lapack_int ldb) {
  return CallWork<T, 2>(matrix_layout, Name<T>("getrs_work"), false,
                        {{Input(a, n, n, lda, -6), {b, n, nrhs, ldb, -9}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Getrs(trans, n, nrhs, p[0], ld[0],
                                               ipiv, p[1], ld[1]);
                        });
}

template <typename T>
lapack_int Getrs(int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
                 const T* a, lapack_int lda, const lapack_int* ipiv, T* b,
                 lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("getrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -8;
  }
  return GetrsWork(matrix_layout, trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <typename T>
lapack_int GesvWork(int matrix_layout, lapack_int n, lapack_int nrhs, T* a,
                    lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gesv_work"), false,
      {{{a, n, n, lda, -5}, {b, n, nrhs, ldb, -8}}}, [&](auto& p, auto& ld) {
        return lapack::Gesv(n, nrhs, p[0], ld[0], ipiv, p[1], ld[1]);
      });
}

template <typename T>
lapack_int Gesv(int matrix_layout, lapack_int n, lapack_int nrhs, T* a,
                lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gesv"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -4;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -7;
  }
  return GesvWork(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb);
}

template <typename T>
lapack_int GetriWork(int matrix_layout, lapack_int n, T* a, lapack_int lda,
                     const lapack_int* ipiv, T* work, lapack_int lwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("getri_work"), lwork == -1,
                        {{{a, n, n, lda, -4}}}, [&](auto& p, auto& ld) {
                          return lapack::Getri(n, p[0], ld[0], ipiv, work,
                                               lwork);
                        });
}

template <typename T>
lapack_int Getri(int matrix_layout, lapack_int n, T* a, lapack_int lda,
                 const lapack_int* ipiv) {
  const std::string name = Name<T>("getri");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -3;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GetriWork(matrix_layout, n, a, lda, ipiv, work, lwork);
  });
}

// ---------------------------------------------------------------------------
// Condition number, equilibration, refinement, expert driver

template <typename T>
lapack_int GeconWork(int matrix_layout, char norm, lapack_int n, const T* a,
                     lapack_int lda, RealT<T> anorm, RealT<T>* rcond, T* work,
                     RealT<T>* rwork, lapack_int* iwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gecon_work"), false,
                        {{Input(a, n, n, lda, -5)}}, [&](auto& p, auto& ld) {
                          return lapack::Gecon(norm, n, p[0], ld[0], anorm,
                                               rcond, work, rwork, iwork);
                        });
}

template <typename T>
lapack_int Gecon(int matrix_layout, char norm, lapack_int n, const T* a,
                 lapack_int lda, RealT<T> anorm, RealT<T>* rcond) {
  const std::string name = Name<T>("gecon");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -4;
    if (VectorNancheck(1, &anorm, 1)) return -6;
  }
  return WithWork<T>(name, 4 * n, n, 2 * n, 2 * n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return GeconWork(matrix_layout, norm, n, a, lda, anorm,
                                        rcond, work, rwork, iwork);
                     });
}

template <typename T>
lapack_int GeequWork(int matrix_layout, lapack_int m, lapack_int n, const T* a,
                     lapack_int lda, RealT<T>* r, RealT<T>* c, RealT<T>* rowcnd,
                     RealT<T>* colcnd, RealT<T>* amax) {
  return CallWork<T, 1>(matrix_layout, Name<T>("geequ_work"), false,
                        {{Input(a, m, n, lda, -5)}}, [&](auto& p, auto& ld) {
                          return lapack::Geequ(m, n, p[0], ld[0], r, c, rowcnd,
                                               colcnd, amax);
                        });
}

template <typename T>
lapack_int Geequ(int matrix_layout, lapack_int m, lapack_int n, const T* a,
                 lapack_int lda, RealT<T>* r, RealT<T>* c, RealT<T>* rowcnd,
                 RealT<T>* colcnd, RealT<T>* amax) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("geequ"), -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return GeequWork(matrix_layout, m, n, a, lda, r, c, rowcnd, colcnd, amax);
}

template <typename T>
lapack_int GerfsWork(int matrix_layout, char trans, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda, const T* af,
                     lapack_int ldaf, const lapack_int* ipiv, const T* b,
                     lapack_int ldb, T* x, lapack_int ldx, RealT<T>* ferr,
                     RealT<T>* berr, T* work, RealT<T>* rwork,
                     lapack_int* iwork) {
  return CallWork<T, 4>(matrix_layout, Name<T>("gerfs_work"), false,
                        {{Input(a, n, n, lda, -6),
                          Input(af, n, n, ldaf, -8),
                          Input(b, n, nrhs, ldb, -11),
                          {x, n, nrhs, ldx, -13}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gerfs(trans, n, nrhs, p[0], ld[0],
                                               p[1], ld[1], ipiv, p[2], ld[2],
                                               p[3], ld[3], ferr, berr, work,
                                               rwork, iwork);
                        });
}

template <typename T>
lapack_int Gerfs(int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
                 const T* a, lapack_int lda, const T* af, lapack_int ldaf,
                 const lapack_int* ipiv, const T* b, lapack_int ldb, T* x,
                 lapack_int ldx, RealT<T>* ferr, RealT<T>* berr) {
  const std::string name = Name<T>("gerfs");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, n, af, ldaf)) return -7;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -10;
    if (GeNancheck(matrix_layout, n, nrhs, x, ldx)) return -12;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return GerfsWork(matrix_layout, trans, n, nrhs, a, lda,
                                        af, ldaf, ipiv, b, ldb, x, ldx, ferr,
                                        berr, work, rwork, iwork);
                     });
}

// xGESVX. In row-major layout, the reference copies the equilibrated A back
// for FACT = 'E', AF back for FACT = 'E' or 'N', and B back only for
// FACT = 'F' (if EQUED is not 'N'), although B is also scaled for
// FACT = 'E'. Deviation (bug fix): B is copied back for FACT = 'E' too, as
// in column-major layout. (The copies back of matrices that the routine did
// not change, e.g. A for EQUED = 'N', leave them as they were.)
template <typename T>
lapack_int GesvxWork(int matrix_layout, char fact, char trans, lapack_int n,
                     lapack_int nrhs, T* a, lapack_int lda, T* af,
                     lapack_int ldaf, lapack_int* ipiv, char* equed,
                     RealT<T>* r, RealT<T>* c, T* b, lapack_int ldb, T* x,
                     lapack_int ldx, RealT<T>* rcond, RealT<T>* ferr,
                     RealT<T>* berr, T* work, RealT<T>* rwork,
                     lapack_int* iwork) {
  const bool given = lapack::Lsame(fact, 'F');
  const bool equilibrate = lapack::Lsame(fact, 'E');
  MatrixArg<T> a_arg{a, n, n, lda, -7};
  a_arg.out = equilibrate;
  MatrixArg<T> af_arg{af, n, n, ldaf, -9};
  af_arg.in = given;
  af_arg.out = !given;
  MatrixArg<T> b_arg{b, n, nrhs, ldb, -15};
  b_arg.out = given || equilibrate;
  MatrixArg<T> x_arg{x, n, nrhs, ldx, -17};
  x_arg.in = false;
  return CallWork<T, 4>(
      matrix_layout, Name<T>("gesvx_work"), false,
      {{a_arg, af_arg, b_arg, x_arg}}, [&](auto& p, auto& ld) {
        return lapack::Gesvx(fact, trans, n, nrhs, p[0], ld[0], p[1], ld[1],
                             ipiv, equed, r, c, p[2], ld[2], p[3], ld[3], rcond,
                             ferr, berr, work, rwork, iwork);
      });
}

template <typename T>
lapack_int Gesvx(int matrix_layout, char fact, char trans, lapack_int n,
                 lapack_int nrhs, T* a, lapack_int lda, T* af, lapack_int ldaf,
                 lapack_int* ipiv, char* equed, RealT<T>* r, RealT<T>* c, T* b,
                 lapack_int ldb, T* x, lapack_int ldx, RealT<T>* rcond,
                 RealT<T>* ferr, RealT<T>* berr, RealT<T>* rpivot) {
  using R = RealT<T>;
  const std::string name = Name<T>("gesvx");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const bool given = lapack::Lsame(fact, 'F');
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -6;
    if (given && GeNancheck(matrix_layout, n, n, af, ldaf)) return -8;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -14;
    if (given && (lapack::Lsame(*equed, 'B') || lapack::Lsame(*equed, 'C')) &&
        VectorNancheck(n, c, 1)) {
      return -13;
    }
    if (given && (lapack::Lsame(*equed, 'B') || lapack::Lsame(*equed, 'R')) &&
        VectorNancheck(n, r, 1)) {
      return -12;
    }
  }
  return WithWork<T>(
      name, 4 * n, n, 2 * n, 2 * n, [&](T* work, R* rwork, lapack_int* iwork) {
        const lapack_int info = GesvxWork(
            matrix_layout, fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed,
            r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, iwork);
        // Backup significant data from working array(s).
        if constexpr (IsComplex<T>::value) {
          *rpivot = rwork[0];
        } else {
          *rpivot = work[0];
        }
        return info;
      });
}

// ---------------------------------------------------------------------------
// Triangular matrices

template <typename T>
lapack_int TrtriWork(int matrix_layout, char uplo, char diag, lapack_int n,
                     T* a, lapack_int lda) {
  return CallWork<T, 1>(matrix_layout, Name<T>("trtri_work"), false,
                        {{Triangle<T>({a, n, n, lda, -6}, uplo)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Trtri(uplo, diag, n, p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Trtri(int matrix_layout, char uplo, char diag, lapack_int n, T* a,
                 lapack_int lda) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("trtri"), -1);
    return -1;
  }
  if (NanCheck() && TrNancheck(matrix_layout, uplo, diag, n, a, lda)) {
    return -5;
  }
  return TrtriWork(matrix_layout, uplo, diag, n, a, lda);
}

template <typename T>
lapack_int TrtrsWork(int matrix_layout, char uplo, char trans, char diag,
                     lapack_int n, lapack_int nrhs, const T* a, lapack_int lda,
                     T* b, lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("trtrs_work"), false,
      {{Triangle(Input(a, n, n, lda, -8), uplo), {b, n, nrhs, ldb, -10}}},
      [&](auto& p, auto& ld) {
        return lapack::Trtrs(uplo, trans, diag, n, nrhs, p[0], ld[0], p[1],
                             ld[1]);
      });
}

template <typename T>
lapack_int Trtrs(int matrix_layout, char uplo, char trans, char diag,
                 lapack_int n, lapack_int nrhs, const T* a, lapack_int lda,
                 T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("trtrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, diag, n, a, lda)) return -7;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -9;
  }
  return TrtrsWork(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

template <typename T>
lapack_int TrconWork(int matrix_layout, char norm, char uplo, char diag,
                     lapack_int n, const T* a, lapack_int lda, RealT<T>* rcond,
                     T* work, RealT<T>* rwork, lapack_int* iwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("trcon_work"), false,
                        {{Triangle(Input(a, n, n, lda, -7), uplo)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Trcon(norm, uplo, diag, n, p[0], ld[0],
                                               rcond, work, rwork, iwork);
                        });
}

template <typename T>
lapack_int Trcon(int matrix_layout, char norm, char uplo, char diag,
                 lapack_int n, const T* a, lapack_int lda, RealT<T>* rcond) {
  const std::string name = Name<T>("trcon");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && TrNancheck(matrix_layout, uplo, diag, n, a, lda)) {
    return -6;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return TrconWork(matrix_layout, norm, uplo, diag, n, a,
                                        lda, rcond, work, rwork, iwork);
                     });
}

template <typename T>
lapack_int TrrfsWork(int matrix_layout, char uplo, char trans, char diag,
                     lapack_int n, lapack_int nrhs, const T* a, lapack_int lda,
                     const T* b, lapack_int ldb, const T* x, lapack_int ldx,
                     RealT<T>* ferr, RealT<T>* berr, T* work, RealT<T>* rwork,
                     lapack_int* iwork) {
  return CallWork<T, 3>(
      matrix_layout, Name<T>("trrfs_work"), false,
      {{Triangle(Input(a, n, n, lda, -8), uplo), Input(b, n, nrhs, ldb, -10),
        Input(x, n, nrhs, ldx, -12)}},
      [&](auto& p, auto& ld) {
        return lapack::Trrfs(uplo, trans, diag, n, nrhs, p[0], ld[0], p[1],
                             ld[1], p[2], ld[2], ferr, berr, work, rwork,
                             iwork);
      });
}

template <typename T>
lapack_int Trrfs(int matrix_layout, char uplo, char trans, char diag,
                 lapack_int n, lapack_int nrhs, const T* a, lapack_int lda,
                 const T* b, lapack_int ldb, const T* x, lapack_int ldx,
                 RealT<T>* ferr, RealT<T>* berr) {
  const std::string name = Name<T>("trrfs");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, diag, n, a, lda)) return -7;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -9;
    if (GeNancheck(matrix_layout, n, nrhs, x, ldx)) return -11;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return TrrfsWork(matrix_layout, uplo, trans, diag, n,
                                        nrhs, a, lda, b, ldb, x, ldx, ferr,
                                        berr, work, rwork, iwork);
                     });
}

// xLANTR: in row-major layout, the norm of the transpose (the 1-norm and
// the infinity norm, and the triangles, swapped) of the matrix as it is.
template <typename T>
RealT<T> LantrWork(int matrix_layout, char norm, char uplo, char diag,
                   lapack_int m, lapack_int n, const T* a, lapack_int lda,
                   RealT<T>* work) {
  using R = RealT<T>;
  if (matrix_layout == LAPACK_COL_MAJOR) {
    return lapack::Lantr(norm, uplo, diag, m, n, a, lda, work);
  }
  const std::string name = Name<T>("lantr_work");
  if (matrix_layout != LAPACK_ROW_MAJOR) {
    Xerbla(name, -1);
    return R(-1);
  }
  // Check leading dimension(s).
  if (lda < n) {
    Xerbla(name, -8);
    return R(-8);
  }
  char norm_lapack = norm;
  if (lapack::Lsame(norm, '1') || lapack::Lsame(norm, 'O')) {
    norm_lapack = 'I';
  } else if (lapack::Lsame(norm, 'I')) {
    norm_lapack = '1';
  }
  const char uplo_lapack = lapack::Lsame(uplo, 'U') ? 'L' : 'U';
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
  const R res =
      lapack::Lantr(norm_lapack, uplo_lapack, diag, n, m, a, lda, work_lapack);
  free(work_lapack);
  return res;
}

template <typename T>
RealT<T> Lantr(int matrix_layout, char norm, char uplo, char diag, lapack_int m,
               lapack_int n, const T* a, lapack_int lda) {
  using R = RealT<T>;
  const std::string name = Name<T>("lantr");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return R(-1);
  }
  if (NanCheck() && TzNancheck(matrix_layout, 'F', uplo, diag, m, n, a, lda)) {
    return R(-7);
  }
  // Allocate memory for working array(s).
  R* work = nullptr;
  if (lapack::Lsame(norm, 'I')) {
    work = static_cast<R*>(malloc(
        sizeof(R) * static_cast<size_t>(std::max<lapack_int>({1, m, n}))));
    if (work == nullptr) {
      Xerbla(name, LAPACK_WORK_MEMORY_ERROR);
      return R(0);
    }
  }
  const R res = LantrWork(matrix_layout, norm, uplo, diag, m, n, a, lda, work);
  free(work);
  return res;
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose arguments are the same in all four precisions.
#define HWYBLAS_LAPACKE_LINEAR(p, T, R)                                        \
  HWYBLAS_API lapack_int LAPACKE_##p##getrs(                                   \
      int layout, char trans, lapack_int n, lapack_int nrhs, const T* a,       \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::Getrs(layout, trans, n, nrhs, a, lda, ipiv, b, ldb);       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##getrs_work(                              \
      int layout, char trans, lapack_int n, lapack_int nrhs, const T* a,       \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::GetrsWork(layout, trans, n, nrhs, a, lda, ipiv, b, ldb);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesv(                                    \
      int layout, lapack_int n, lapack_int nrhs, T* a, lapack_int lda,         \
      lapack_int* ipiv, T* b, lapack_int ldb) {                                \
    return lapacke::Gesv(layout, n, nrhs, a, lda, ipiv, b, ldb);               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesv_work(                               \
      int layout, lapack_int n, lapack_int nrhs, T* a, lapack_int lda,         \
      lapack_int* ipiv, T* b, lapack_int ldb) {                                \
    return lapacke::GesvWork(layout, n, nrhs, a, lda, ipiv, b, ldb);           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##getri(int layout, lapack_int n, T* a,    \
                                            lapack_int lda,                    \
                                            const lapack_int* ipiv) {          \
    return lapacke::Getri(layout, n, a, lda, ipiv);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##getri_work(                              \
      int layout, lapack_int n, T* a, lapack_int lda, const lapack_int* ipiv,  \
      T* work, lapack_int lwork) {                                             \
    return lapacke::GetriWork(layout, n, a, lda, ipiv, work, lwork);           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gecon(                                   \
      int layout, char norm, lapack_int n, const T* a, lapack_int lda,         \
      R anorm, R* rcond) {                                                     \
    return lapacke::Gecon(layout, norm, n, a, lda, anorm, rcond);              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geequ(                                   \
      int layout, lapack_int m, lapack_int n, const T* a, lapack_int lda,      \
      R* r, R* c, R* rowcnd, R* colcnd, R* amax) {                             \
    return lapacke::Geequ(layout, m, n, a, lda, r, c, rowcnd, colcnd, amax);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geequ_work(                              \
      int layout, lapack_int m, lapack_int n, const T* a, lapack_int lda,      \
      R* r, R* c, R* rowcnd, R* colcnd, R* amax) {                             \
    return lapacke::GeequWork(layout, m, n, a, lda, r, c, rowcnd, colcnd,      \
                              amax);                                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gerfs(                                   \
      int layout, char trans, lapack_int n, lapack_int nrhs, const T* a,       \
      lapack_int lda, const T* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr) {    \
    return lapacke::Gerfs(layout, trans, n, nrhs, a, lda, af, ldaf, ipiv, b,   \
                          ldb, x, ldx, ferr, berr);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesvx(                                   \
      int layout, char fact, char trans, lapack_int n, lapack_int nrhs, T* a,  \
      lapack_int lda, T* af, lapack_int ldaf, lapack_int* ipiv, char* equed,   \
      R* r, R* c, T* b, lapack_int ldb, T* x, lapack_int ldx, R* rcond,        \
      R* ferr, R* berr, R* rpivot) {                                           \
    return lapacke::Gesvx(layout, fact, trans, n, nrhs, a, lda, af, ldaf,      \
                          ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr,      \
                          berr, rpivot);                                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trtri(                                   \
      int layout, char uplo, char diag, lapack_int n, T* a, lapack_int lda) {  \
    return lapacke::Trtri(layout, uplo, diag, n, a, lda);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trtri_work(                              \
      int layout, char uplo, char diag, lapack_int n, T* a, lapack_int lda) {  \
    return lapacke::TrtriWork(layout, uplo, diag, n, a, lda);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trtrs(                                   \
      int layout, char uplo, char trans, char diag, lapack_int n,              \
      lapack_int nrhs, const T* a, lapack_int lda, T* b, lapack_int ldb) {     \
    return lapacke::Trtrs(layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trtrs_work(                              \
      int layout, char uplo, char trans, char diag, lapack_int n,              \
      lapack_int nrhs, const T* a, lapack_int lda, T* b, lapack_int ldb) {     \
    return lapacke::TrtrsWork(layout, uplo, trans, diag, n, nrhs, a, lda, b,   \
                              ldb);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trcon(                                   \
      int layout, char norm, char uplo, char diag, lapack_int n, const T* a,   \
      lapack_int lda, R* rcond) {                                              \
    return lapacke::Trcon(layout, norm, uplo, diag, n, a, lda, rcond);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trrfs(                                   \
      int layout, char uplo, char trans, char diag, lapack_int n,              \
      lapack_int nrhs, const T* a, lapack_int lda, const T* b, lapack_int ldb, \
      const T* x, lapack_int ldx, R* ferr, R* berr) {                          \
    return lapacke::Trrfs(layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb,  \
                          x, ldx, ferr, berr);                                 \
  }                                                                            \
  HWYBLAS_API R LAPACKE_##p##lantr(int layout, char norm, char uplo,           \
                                   char diag, lapack_int m, lapack_int n,      \
                                   const T* a, lapack_int lda) {               \
    return lapacke::Lantr(layout, norm, uplo, diag, m, n, a, lda);             \
  }                                                                            \
  HWYBLAS_API R LAPACKE_##p##lantr_work(int layout, char norm, char uplo,      \
                                        char diag, lapack_int m, lapack_int n, \
                                        const T* a, lapack_int lda, R* work) { \
    return lapacke::LantrWork(layout, norm, uplo, diag, m, n, a, lda, work);   \
  }

HWYBLAS_LAPACKE_LINEAR(s, float, float)
HWYBLAS_LAPACKE_LINEAR(d, double, double)
HWYBLAS_LAPACKE_LINEAR(c, c32, float)
HWYBLAS_LAPACKE_LINEAR(z, c64, double)

#undef HWYBLAS_LAPACKE_LINEAR

// The _work functions that take IWORK (real) or RWORK (complex).
#define HWYBLAS_LAPACKE_LINEAR_REAL(p, R)                                      \
  HWYBLAS_API lapack_int LAPACKE_##p##gecon_work(                              \
      int layout, char norm, lapack_int n, const R* a, lapack_int lda,         \
      R anorm, R* rcond, R* work, lapack_int* iwork) {                         \
    return lapacke::GeconWork(layout, norm, n, a, lda, anorm, rcond, work,     \
                              static_cast<R*>(nullptr), iwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gerfs_work(                              \
      int layout, char trans, lapack_int n, lapack_int nrhs, const R* a,       \
      lapack_int lda, const R* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const R* b, lapack_int ldb, R* x, lapack_int ldx, R* ferr, R* berr,      \
      R* work, lapack_int* iwork) {                                            \
    return lapacke::GerfsWork(layout, trans, n, nrhs, a, lda, af, ldaf, ipiv,  \
                              b, ldb, x, ldx, ferr, berr, work,                \
                              static_cast<R*>(nullptr), iwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesvx_work(                              \
      int layout, char fact, char trans, lapack_int n, lapack_int nrhs, R* a,  \
      lapack_int lda, R* af, lapack_int ldaf, lapack_int* ipiv, char* equed,   \
      R* r, R* c, R* b, lapack_int ldb, R* x, lapack_int ldx, R* rcond,        \
      R* ferr, R* berr, R* work, lapack_int* iwork) {                          \
    return lapacke::GesvxWork(layout, fact, trans, n, nrhs, a, lda, af, ldaf,  \
                              ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr,  \
                              berr, work, static_cast<R*>(nullptr), iwork);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trcon_work(                              \
      int layout, char norm, char uplo, char diag, lapack_int n, const R* a,   \
      lapack_int lda, R* rcond, R* work, lapack_int* iwork) {                  \
    return lapacke::TrconWork(layout, norm, uplo, diag, n, a, lda, rcond,      \
                              work, static_cast<R*>(nullptr), iwork);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trrfs_work(                              \
      int layout, char uplo, char trans, char diag, lapack_int n,              \
      lapack_int nrhs, const R* a, lapack_int lda, const R* b, lapack_int ldb, \
      const R* x, lapack_int ldx, R* ferr, R* berr, R* work,                   \
      lapack_int* iwork) {                                                     \
    return lapacke::TrrfsWork(layout, uplo, trans, diag, n, nrhs, a, lda, b,   \
                              ldb, x, ldx, ferr, berr, work,                   \
                              static_cast<R*>(nullptr), iwork);                \
  }

#define HWYBLAS_LAPACKE_LINEAR_COMPLEX(p, T, R)                                \
  HWYBLAS_API lapack_int LAPACKE_##p##gecon_work(                              \
      int layout, char norm, lapack_int n, const T* a, lapack_int lda,         \
      R anorm, R* rcond, T* work, R* rwork) {                                  \
    return lapacke::GeconWork(layout, norm, n, a, lda, anorm, rcond, work,     \
                              rwork, static_cast<lapack_int*>(nullptr));       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gerfs_work(                              \
      int layout, char trans, lapack_int n, lapack_int nrhs, const T* a,       \
      lapack_int lda, const T* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr,      \
      T* work, R* rwork) {                                                     \
    return lapacke::GerfsWork(layout, trans, n, nrhs, a, lda, af, ldaf, ipiv,  \
                              b, ldb, x, ldx, ferr, berr, work, rwork,         \
                              static_cast<lapack_int*>(nullptr));              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gesvx_work(                              \
      int layout, char fact, char trans, lapack_int n, lapack_int nrhs, T* a,  \
      lapack_int lda, T* af, lapack_int ldaf, lapack_int* ipiv, char* equed,   \
      R* r, R* c, T* b, lapack_int ldb, T* x, lapack_int ldx, R* rcond,        \
      R* ferr, R* berr, T* work, R* rwork) {                                   \
    return lapacke::GesvxWork(layout, fact, trans, n, nrhs, a, lda, af, ldaf,  \
                              ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr,  \
                              berr, work, rwork,                               \
                              static_cast<lapack_int*>(nullptr));              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trcon_work(                              \
      int layout, char norm, char uplo, char diag, lapack_int n, const T* a,   \
      lapack_int lda, R* rcond, T* work, R* rwork) {                           \
    return lapacke::TrconWork(layout, norm, uplo, diag, n, a, lda, rcond,      \
                              work, rwork, static_cast<lapack_int*>(nullptr)); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trrfs_work(                              \
      int layout, char uplo, char trans, char diag, lapack_int n,              \
      lapack_int nrhs, const T* a, lapack_int lda, const T* b, lapack_int ldb, \
      const T* x, lapack_int ldx, R* ferr, R* berr, T* work, R* rwork) {       \
    return lapacke::TrrfsWork(layout, uplo, trans, diag, n, nrhs, a, lda, b,   \
                              ldb, x, ldx, ferr, berr, work, rwork,            \
                              static_cast<lapack_int*>(nullptr));              \
  }

HWYBLAS_LAPACKE_LINEAR_REAL(s, float)
HWYBLAS_LAPACKE_LINEAR_REAL(d, double)
HWYBLAS_LAPACKE_LINEAR_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_LINEAR_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_LINEAR_REAL
#undef HWYBLAS_LAPACKE_LINEAR_COMPLEX

}  // extern "C"
