// LAPACKE functions of the MRRR, bisection and inverse iteration
// symmetric (Hermitian) eigensolvers: xSYEVR / xHEEVR, xSYEVX / xHEEVX,
// xSTEVR, xSTEVX, xSTEMR, xSTEBZ, xSTEIN and xLARNV (see lapacke.cc and
// lapacke_util.h).
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

// The number of columns of Z in row-major layout (as the reference): N for
// RANGE = 'A' or 'V', IU-IL+1 for 'I'.
inline lapack_int ColumnsOfZ(char range, lapack_int n, lapack_int il,
                             lapack_int iu) {
  return (lapack::Lsame(range, 'A') || lapack::Lsame(range, 'V')) ? n
         : lapack::Lsame(range, 'I')                              ? iu - il + 1
                                                                  : 1;
}

// ---------------------------------------------------------------------------
// Dense matrices

// xSYEVR / xHEEVR; rwork and lrwork for complex T only. In row-major
// layout, the triangle uplo of A is copied in and back, and Z (n x
// ncols_z; 1 column for jobz = 'N') out for jobz = 'V'.
template <typename T>
lapack_int HeevrWork(int matrix_layout, char jobz, char range, char uplo,
                     lapack_int n, T* a, lapack_int lda, RealT<T> vl,
                     RealT<T> vu, lapack_int il, lapack_int iu, RealT<T> abstol,
                     lapack_int* m, RealT<T>* w, T* z, lapack_int ldz,
                     lapack_int* isuppz, T* work, lapack_int lwork,
                     RealT<T>* rwork, lapack_int lrwork, lapack_int* iwork,
                     lapack_int liwork) {
  const bool wantz = lapack::Lsame(jobz, 'V');
  MatrixArg<T> z_arg{z, n, wantz ? ColumnsOfZ(range, n, il, iu) : 1, ldz, -16};
  z_arg.in = false;
  z_arg.out = wantz;
  z_arg.used = wantz;
  const bool query =
      liwork == -1 || lwork == -1 || (IsComplex<T>::value && lrwork == -1);
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("syevr_work", "heevr_work")),
      query, {{Triangle<T>({a, n, n, lda, -7}, uplo), z_arg}},
      [&](auto& p, auto& ld) {
        return lapack::Heevr(jobz, range, uplo, n, p[0], ld[0], vl, vu, il, iu,
                             abstol, m, w, p[1], ld[1], isuppz, work, lwork,
                             rwork, lrwork, iwork, liwork);
      });
}

template <typename T>
lapack_int Heevr(int matrix_layout, char jobz, char range, char uplo,
                 lapack_int n, T* a, lapack_int lda, RealT<T> vl, RealT<T> vu,
                 lapack_int il, lapack_int iu, RealT<T> abstol, lapack_int* m,
                 RealT<T>* w, T* z, lapack_int ldz, lapack_int* isuppz) {
  const std::string name = Name<T>(RealOrComplex<T>("syevr", "heevr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) return -6;
    if (VectorNancheck(1, &abstol, 1)) return -12;
    if (lapack::Lsame(range, 'V')) {
      if (VectorNancheck(1, &vl, 1)) return -8;
      if (VectorNancheck(1, &vu, 1)) return -9;
    }
  }
  return WithWorkspaces<T>(
      name, [&](T* work, lapack_int lwork, RealT<T>* rwork, lapack_int lrwork,
                lapack_int* iwork, lapack_int liwork) {
        return HeevrWork(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu,
                         il, iu, abstol, m, w, z, ldz, isuppz, work, lwork,
                         rwork, lrwork, iwork, liwork);
      });
}

// xSYEVX / xHEEVX; rwork for complex T only. In row-major layout, the
// triangle uplo of A is copied in and back, and Z (n x ncols_z) out for
// jobz = 'V'.
template <typename T>
lapack_int HeevxWork(int matrix_layout, char jobz, char range, char uplo,
                     lapack_int n, T* a, lapack_int lda, RealT<T> vl,
                     RealT<T> vu, lapack_int il, lapack_int iu, RealT<T> abstol,
                     lapack_int* m, RealT<T>* w, T* z, lapack_int ldz, T* work,
                     lapack_int lwork, RealT<T>* rwork, lapack_int* iwork,
                     lapack_int* ifail) {
  const bool wantz = lapack::Lsame(jobz, 'V');
  MatrixArg<T> z_arg{z, n, wantz ? ColumnsOfZ(range, n, il, iu) : 1, ldz, -16};
  z_arg.in = false;
  z_arg.out = wantz;
  z_arg.used = wantz;
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("syevx_work", "heevx_work")),
      lwork == -1, {{Triangle<T>({a, n, n, lda, -7}, uplo), z_arg}},
      [&](auto& p, auto& ld) {
        return lapack::Heevx(jobz, range, uplo, n, p[0], ld[0], vl, vu, il, iu,
                             abstol, m, w, p[1], ld[1], work, lwork, rwork,
                             iwork, ifail);
      });
}

template <typename T>
lapack_int Heevx(int matrix_layout, char jobz, char range, char uplo,
                 lapack_int n, T* a, lapack_int lda, RealT<T> vl, RealT<T> vu,
                 lapack_int il, lapack_int iu, RealT<T> abstol, lapack_int* m,
                 RealT<T>* w, T* z, lapack_int ldz, lapack_int* ifail) {
  using R = RealT<T>;
  const std::string name = Name<T>(RealOrComplex<T>("syevx", "heevx"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) return -6;
    if (VectorNancheck(1, &abstol, 1)) return -12;
    if (lapack::Lsame(range, 'V')) {
      if (VectorNancheck(1, &vl, 1)) return -8;
      if (VectorNancheck(1, &vu, 1)) return -9;
    }
  }
  // IWORK (5*N) and, for complex T, RWORK (7*N) have fixed sizes; WORK is
  // queried.
  const auto call = [&](R* rwork) {
    return WithWorkArray<lapack_int>(
        name, std::max<lapack_int>(1, 5 * n), [&](lapack_int* iwork) {
          return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
            return HeevxWork(matrix_layout, jobz, range, uplo, n, a, lda, vl,
                             vu, il, iu, abstol, m, w, z, ldz, work, lwork,
                             rwork, iwork, ifail);
          });
        });
  };
  if constexpr (IsComplex<T>::value) {
    return WithWorkArray<R>(name, std::max<lapack_int>(1, 7 * n), call);
  } else {
    return call(static_cast<R*>(nullptr));
  }
}

// ---------------------------------------------------------------------------
// Tridiagonal matrices

// xSTEVR (real R only). Z (n x ncols_z) is copied out for jobz = 'V'; its
// leading dimension is checked against ncols_z also for jobz = 'N' (as the
// reference).
template <typename R>
lapack_int StevrWork(int matrix_layout, char jobz, char range, lapack_int n,
                     R* d, R* e, R vl, R vu, lapack_int il, lapack_int iu,
                     R abstol, lapack_int* m, R* w, R* z, lapack_int ldz,
                     lapack_int* isuppz, R* work, lapack_int lwork,
                     lapack_int* iwork, lapack_int liwork) {
  const bool wantz = lapack::Lsame(jobz, 'V');
  MatrixArg<R> arg{z, n, ColumnsOfZ(range, n, il, iu), ldz, -15};
  arg.in = false;
  arg.out = wantz;
  arg.used = wantz;
  return CallWork<R, 1>(
      matrix_layout, Name<R>("stevr_work"), liwork == -1 || lwork == -1,
      {{arg}}, [&](auto& p, auto& ld) {
        return lapack::Stevr(jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w,
                             p[0], ld[0], isuppz, work, lwork, iwork, liwork);
      });
}

template <typename R>
lapack_int Stevr(int matrix_layout, char jobz, char range, lapack_int n, R* d,
                 R* e, R vl, R vu, lapack_int il, lapack_int iu, R abstol,
                 lapack_int* m, R* w, R* z, lapack_int ldz,
                 lapack_int* isuppz) {
  const std::string name = Name<R>("stevr");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(1, &abstol, 1)) return -11;
    if (VectorNancheck(n, d, 1)) return -5;
    if (VectorNancheck(n - 1, e, 1)) return -6;
    if (lapack::Lsame(range, 'V')) {
      if (VectorNancheck(1, &vl, 1)) return -7;
      if (VectorNancheck(1, &vu, 1)) return -8;
    }
  }
  return WithWorkspaces<R>(name, [&](R* work, lapack_int lwork, R* /*rwork*/,
                                     lapack_int /*lrwork*/, lapack_int* iwork,
                                     lapack_int liwork) {
    return StevrWork(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu,
                     abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork);
  });
}

// xSTEVX (real R only). Z (n x ncols_z) is copied out for jobz = 'V'; its
// leading dimension is checked against ncols_z also for jobz = 'N' (as the
// reference).
template <typename R>
lapack_int StevxWork(int matrix_layout, char jobz, char range, lapack_int n,
                     R* d, R* e, R vl, R vu, lapack_int il, lapack_int iu,
                     R abstol, lapack_int* m, R* w, R* z, lapack_int ldz,
                     R* work, lapack_int* iwork, lapack_int* ifail) {
  const bool wantz = lapack::Lsame(jobz, 'V');
  MatrixArg<R> arg{z, n, ColumnsOfZ(range, n, il, iu), ldz, -15};
  arg.in = false;
  arg.out = wantz;
  arg.used = wantz;
  return CallWork<R, 1>(matrix_layout, Name<R>("stevx_work"), false, {{arg}},
                        [&](auto& p, auto& ld) {
                          return lapack::Stevx(jobz, range, n, d, e, vl, vu, il,
                                               iu, abstol, m, w, p[0], ld[0],
                                               work, iwork, ifail);
                        });
}

template <typename R>
lapack_int Stevx(int matrix_layout, char jobz, char range, lapack_int n, R* d,
                 R* e, R vl, R vu, lapack_int il, lapack_int iu, R abstol,
                 lapack_int* m, R* w, R* z, lapack_int ldz, lapack_int* ifail) {
  const std::string name = Name<R>("stevx");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(1, &abstol, 1)) return -11;
    if (VectorNancheck(n, d, 1)) return -5;
    if (VectorNancheck(n - 1, e, 1)) return -6;
    if (lapack::Lsame(range, 'V')) {
      if (VectorNancheck(1, &vl, 1)) return -7;
      if (VectorNancheck(1, &vu, 1)) return -8;
    }
  }
  return WithWorkArray<lapack_int>(
      name, std::max<lapack_int>(1, 5 * n), [&](lapack_int* iwork) {
        return WithWorkArray<R>(
            name, std::max<lapack_int>(1, 5 * n), [&](R* work) {
              return StevxWork(matrix_layout, jobz, range, n, d, e, vl, vu, il,
                               iu, abstol, m, w, z, ldz, work, iwork, ifail);
            });
      });
}

// xSTEMR. Z (n x n) is copied out for jobz = 'V'; in row-major layout ldz
// must be at least 1, and n for jobz = 'V' (as the reference).
template <typename T>
lapack_int StemrWork(int matrix_layout, char jobz, char range, lapack_int n,
                     RealT<T>* d, RealT<T>* e, RealT<T> vl, RealT<T> vu,
                     lapack_int il, lapack_int iu, lapack_int* m, RealT<T>* w,
                     T* z, lapack_int ldz, lapack_int nzc, lapack_int* isuppz,
                     lapack_logical* tryrac, RealT<T>* work, lapack_int lwork,
                     lapack_int* iwork, lapack_int liwork) {
  const bool wantz = lapack::Lsame(jobz, 'V');
  MatrixArg<T> arg{z, n, wantz ? std::max<lapack_int>(1, n) : 1, ldz, -14};
  arg.in = false;
  arg.out = wantz;
  arg.used = wantz;
  return CallWork<T, 1>(
      matrix_layout, Name<T>("stemr_work"), liwork == -1 || lwork == -1,
      {{arg}}, [&](auto& p, auto& ld) {
        bool rac = *tryrac != 0;
        const lapack_int info =
            lapack::Stemr(jobz, range, n, d, e, vl, vu, il, iu, m, w, p[0],
                          ld[0], nzc, isuppz, &rac, work, lwork, iwork, liwork);
        *tryrac = rac ? 1 : 0;
        return info;
      });
}

template <typename T>
lapack_int Stemr(int matrix_layout, char jobz, char range, lapack_int n,
                 RealT<T>* d, RealT<T>* e, RealT<T> vl, RealT<T> vu,
                 lapack_int il, lapack_int iu, lapack_int* m, RealT<T>* w, T* z,
                 lapack_int ldz, lapack_int nzc, lapack_int* isuppz,
                 lapack_logical* tryrac) {
  using R = RealT<T>;
  const std::string name = Name<T>("stemr");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -5;
    if (VectorNancheck(n - 1, e, 1)) return -6;
    if (VectorNancheck(1, &vl, 1)) return -7;
    if (VectorNancheck(1, &vu, 1)) return -8;
  }
  // The real workspaces of xSTEMR (also for complex T).
  return WithWorkspaces<R>(name, [&](R* work, lapack_int lwork, R* /*rwork*/,
                                     lapack_int /*lrwork*/, lapack_int* iwork,
                                     lapack_int liwork) {
    return StemrWork(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w,
                     z, ldz, nzc, isuppz, tryrac, work, lwork, iwork, liwork);
  });
}

// xSTEBZ (real R only; no layout, INFO is not adjusted).
template <typename R>
lapack_int StebzWork(char range, char order, lapack_int n, R vl, R vu,
                     lapack_int il, lapack_int iu, R abstol, const R* d,
                     const R* e, lapack_int* m, lapack_int* nsplit, R* w,
                     lapack_int* iblock, lapack_int* isplit, R* work,
                     lapack_int* iwork) {
  return lapack::Stebz(range, order, n, vl, vu, il, iu, abstol,
                       const_cast<R*>(d), const_cast<R*>(e), m, nsplit, w,
                       iblock, isplit, work, iwork);
}

template <typename R>
lapack_int Stebz(char range, char order, lapack_int n, R vl, R vu,
                 lapack_int il, lapack_int iu, R abstol, const R* d, const R* e,
                 lapack_int* m, lapack_int* nsplit, R* w, lapack_int* iblock,
                 lapack_int* isplit) {
  const std::string name = Name<R>("stebz");
  if (NanCheck()) {
    if (VectorNancheck(1, &abstol, 1)) return -8;
    if (VectorNancheck(n, d, 1)) return -9;
    if (VectorNancheck(n - 1, e, 1)) return -10;
    if (lapack::Lsame(range, 'V')) {
      if (VectorNancheck(1, &vl, 1)) return -4;
      if (VectorNancheck(1, &vu, 1)) return -5;
    }
  }
  return WithWorkArray<lapack_int>(
      name, std::max<lapack_int>(1, 3 * n), [&](lapack_int* iwork) {
        return WithWorkArray<R>(
            name, std::max<lapack_int>(1, 4 * n), [&](R* work) {
              return StebzWork(range, order, n, vl, vu, il, iu, abstol, d, e, m,
                               nsplit, w, iblock, isplit, work, iwork);
            });
      });
}

// xSTEIN. Z (n x m) is copied out.
template <typename T>
lapack_int SteinWork(int matrix_layout, lapack_int n, const RealT<T>* d,
                     const RealT<T>* e, lapack_int m, const RealT<T>* w,
                     const lapack_int* iblock, const lapack_int* isplit, T* z,
                     lapack_int ldz, RealT<T>* work, lapack_int* iwork,
                     lapack_int* ifailv) {
  using R = RealT<T>;
  MatrixArg<T> arg{z, n, m, ldz, -10};
  arg.in = false;
  return CallWork<T, 1>(
      matrix_layout, Name<T>("stein_work"), false, {{arg}},
      [&](auto& p, auto& ld) {
        return lapack::Stein(n, const_cast<R*>(d), const_cast<R*>(e), m,
                             const_cast<R*>(w), const_cast<lapack_int*>(iblock),
                             const_cast<lapack_int*>(isplit), p[0], ld[0], work,
                             iwork, ifailv);
      });
}

template <typename T>
lapack_int Stein(int matrix_layout, lapack_int n, const RealT<T>* d,
                 const RealT<T>* e, lapack_int m, const RealT<T>* w,
                 const lapack_int* iblock, const lapack_int* isplit, T* z,
                 lapack_int ldz, lapack_int* ifailv) {
  using R = RealT<T>;
  const std::string name = Name<T>("stein");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -3;
    if (VectorNancheck(n - 1, e, 1)) return -4;
    if (VectorNancheck(n, w, 1)) return -6;
  }
  return WithWorkArray<lapack_int>(
      name, std::max<lapack_int>(1, n), [&](lapack_int* iwork) {
        return WithWorkArray<R>(
            name, std::max<lapack_int>(1, 5 * n), [&](R* work) {
              return SteinWork(matrix_layout, n, d, e, m, w, iblock, isplit, z,
                               ldz, work, iwork, ifailv);
            });
      });
}

// xLARNV (no layout and no NaN check).
template <typename T>
lapack_int Larnv(lapack_int idist, lapack_int* iseed, lapack_int n, T* x) {
  lapack::Larnv(idist, iseed, n, x);
  return 0;
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose arguments are the same in all four precisions.
#define HWYBLAS_LAPACKE_SYEVR(p, T, R)                                         \
  HWYBLAS_API lapack_int LAPACKE_##p##stemr(                                   \
      int layout, char jobz, char range, lapack_int n, R* d, R* e, R vl, R vu, \
      lapack_int il, lapack_int iu, lapack_int* m, R* w, T* z, lapack_int ldz, \
      lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac) {            \
    return lapacke::Stemr(layout, jobz, range, n, d, e, vl, vu, il, iu, m, w,  \
                          z, ldz, nzc, isuppz, tryrac);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stemr_work(                              \
      int layout, char jobz, char range, lapack_int n, R* d, R* e, R vl, R vu, \
      lapack_int il, lapack_int iu, lapack_int* m, R* w, T* z, lapack_int ldz, \
      lapack_int nzc, lapack_int* isuppz, lapack_logical* tryrac, R* work,     \
      lapack_int lwork, lapack_int* iwork, lapack_int liwork) {                \
    return lapacke::StemrWork(layout, jobz, range, n, d, e, vl, vu, il, iu, m, \
                              w, z, ldz, nzc, isuppz, tryrac, work, lwork,     \
                              iwork, liwork);                                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stein(                                   \
      int layout, lapack_int n, const R* d, const R* e, lapack_int m,          \
      const R* w, const lapack_int* iblock, const lapack_int* isplit, T* z,    \
      lapack_int ldz, lapack_int* ifailv) {                                    \
    return lapacke::Stein(layout, n, d, e, m, w, iblock, isplit, z, ldz,       \
                          ifailv);                                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stein_work(                              \
      int layout, lapack_int n, const R* d, const R* e, lapack_int m,          \
      const R* w, const lapack_int* iblock, const lapack_int* isplit, T* z,    \
      lapack_int ldz, R* work, lapack_int* iwork, lapack_int* ifailv) {        \
    return lapacke::SteinWork(layout, n, d, e, m, w, iblock, isplit, z, ldz,   \
                              work, iwork, ifailv);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larnv(                                   \
      lapack_int idist, lapack_int* iseed, lapack_int n, T* x) {               \
    return lapacke::Larnv(idist, iseed, n, x);                                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##larnv_work(                              \
      lapack_int idist, lapack_int* iseed, lapack_int n, T* x) {               \
    return lapacke::Larnv(idist, iseed, n, x);                                 \
  }

HWYBLAS_LAPACKE_SYEVR(s, float, float)
HWYBLAS_LAPACKE_SYEVR(d, double, double)
HWYBLAS_LAPACKE_SYEVR(c, c32, float)
HWYBLAS_LAPACKE_SYEVR(z, c64, double)

#undef HWYBLAS_LAPACKE_SYEVR

#define HWYBLAS_LAPACKE_SYEVR_REAL(p, R)                                       \
  HWYBLAS_API lapack_int LAPACKE_##p##syevr(                                   \
      int layout, char jobz, char range, char uplo, lapack_int n, R* a,        \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,      \
      lapack_int* m, R* w, R* z, lapack_int ldz, lapack_int* isuppz) {         \
    return lapacke::Heevr(layout, jobz, range, uplo, n, a, lda, vl, vu, il,    \
                          iu, abstol, m, w, z, ldz, isuppz);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syevr_work(                              \
      int layout, char jobz, char range, char uplo, lapack_int n, R* a,        \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,      \
      lapack_int* m, R* w, R* z, lapack_int ldz, lapack_int* isuppz, R* work,  \
      lapack_int lwork, lapack_int* iwork, lapack_int liwork) {                \
    return lapacke::HeevrWork(                                                 \
        layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, \
        ldz, isuppz, work, lwork, static_cast<R*>(nullptr), 0, iwork, liwork); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stevr(                                   \
      int layout, char jobz, char range, lapack_int n, R* d, R* e, R vl, R vu, \
      lapack_int il, lapack_int iu, R abstol, lapack_int* m, R* w, R* z,       \
      lapack_int ldz, lapack_int* isuppz) {                                    \
    return lapacke::Stevr(layout, jobz, range, n, d, e, vl, vu, il, iu,        \
                          abstol, m, w, z, ldz, isuppz);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stevr_work(                              \
      int layout, char jobz, char range, lapack_int n, R* d, R* e, R vl, R vu, \
      lapack_int il, lapack_int iu, R abstol, lapack_int* m, R* w, R* z,       \
      lapack_int ldz, lapack_int* isuppz, R* work, lapack_int lwork,           \
      lapack_int* iwork, lapack_int liwork) {                                  \
    return lapacke::StevrWork(layout, jobz, range, n, d, e, vl, vu, il, iu,    \
                              abstol, m, w, z, ldz, isuppz, work, lwork,       \
                              iwork, liwork);                                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syevx(                                   \
      int layout, char jobz, char range, char uplo, lapack_int n, R* a,        \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,      \
      lapack_int* m, R* w, R* z, lapack_int ldz, lapack_int* ifail) {          \
    return lapacke::Heevx(layout, jobz, range, uplo, n, a, lda, vl, vu, il,    \
                          iu, abstol, m, w, z, ldz, ifail);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syevx_work(                              \
      int layout, char jobz, char range, char uplo, lapack_int n, R* a,        \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,      \
      lapack_int* m, R* w, R* z, lapack_int ldz, R* work, lapack_int lwork,    \
      lapack_int* iwork, lapack_int* ifail) {                                  \
    return lapacke::HeevxWork(layout, jobz, range, uplo, n, a, lda, vl, vu,    \
                              il, iu, abstol, m, w, z, ldz, work, lwork,       \
                              static_cast<R*>(nullptr), iwork, ifail);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stevx(                                   \
      int layout, char jobz, char range, lapack_int n, R* d, R* e, R vl, R vu, \
      lapack_int il, lapack_int iu, R abstol, lapack_int* m, R* w, R* z,       \
      lapack_int ldz, lapack_int* ifail) {                                     \
    return lapacke::Stevx(layout, jobz, range, n, d, e, vl, vu, il, iu,        \
                          abstol, m, w, z, ldz, ifail);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stevx_work(                              \
      int layout, char jobz, char range, lapack_int n, R* d, R* e, R vl, R vu, \
      lapack_int il, lapack_int iu, R abstol, lapack_int* m, R* w, R* z,       \
      lapack_int ldz, R* work, lapack_int* iwork, lapack_int* ifail) {         \
    return lapacke::StevxWork(layout, jobz, range, n, d, e, vl, vu, il, iu,    \
                              abstol, m, w, z, ldz, work, iwork, ifail);       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stebz(                                   \
      char range, char order, lapack_int n, R vl, R vu, lapack_int il,         \
      lapack_int iu, R abstol, const R* d, const R* e, lapack_int* m,          \
      lapack_int* nsplit, R* w, lapack_int* iblock, lapack_int* isplit) {      \
    return lapacke::Stebz(range, order, n, vl, vu, il, iu, abstol, d, e, m,    \
                          nsplit, w, iblock, isplit);                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stebz_work(                              \
      char range, char order, lapack_int n, R vl, R vu, lapack_int il,         \
      lapack_int iu, R abstol, const R* d, const R* e, lapack_int* m,          \
      lapack_int* nsplit, R* w, lapack_int* iblock, lapack_int* isplit,        \
      R* work, lapack_int* iwork) {                                            \
    return lapacke::StebzWork(range, order, n, vl, vu, il, iu, abstol, d, e,   \
                              m, nsplit, w, iblock, isplit, work, iwork);      \
  }

HWYBLAS_LAPACKE_SYEVR_REAL(s, float)
HWYBLAS_LAPACKE_SYEVR_REAL(d, double)

#undef HWYBLAS_LAPACKE_SYEVR_REAL

#define HWYBLAS_LAPACKE_HEEVR(p, T, R)                                        \
  HWYBLAS_API lapack_int LAPACKE_##p##heevr(                                  \
      int layout, char jobz, char range, char uplo, lapack_int n, T* a,       \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,     \
      lapack_int* m, R* w, T* z, lapack_int ldz, lapack_int* isuppz) {        \
    return lapacke::Heevr(layout, jobz, range, uplo, n, a, lda, vl, vu, il,   \
                          iu, abstol, m, w, z, ldz, isuppz);                  \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##heevr_work(                             \
      int layout, char jobz, char range, char uplo, lapack_int n, T* a,       \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,     \
      lapack_int* m, R* w, T* z, lapack_int ldz, lapack_int* isuppz, T* work, \
      lapack_int lwork, R* rwork, lapack_int lrwork, lapack_int* iwork,       \
      lapack_int liwork) {                                                    \
    return lapacke::HeevrWork(layout, jobz, range, uplo, n, a, lda, vl, vu,   \
                              il, iu, abstol, m, w, z, ldz, isuppz, work,     \
                              lwork, rwork, lrwork, iwork, liwork);           \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##heevx(                                  \
      int layout, char jobz, char range, char uplo, lapack_int n, T* a,       \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,     \
      lapack_int* m, R* w, T* z, lapack_int ldz, lapack_int* ifail) {         \
    return lapacke::Heevx(layout, jobz, range, uplo, n, a, lda, vl, vu, il,   \
                          iu, abstol, m, w, z, ldz, ifail);                   \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##heevx_work(                             \
      int layout, char jobz, char range, char uplo, lapack_int n, T* a,       \
      lapack_int lda, R vl, R vu, lapack_int il, lapack_int iu, R abstol,     \
      lapack_int* m, R* w, T* z, lapack_int ldz, T* work, lapack_int lwork,   \
      R* rwork, lapack_int* iwork, lapack_int* ifail) {                       \
    return lapacke::HeevxWork(layout, jobz, range, uplo, n, a, lda, vl, vu,   \
                              il, iu, abstol, m, w, z, ldz, work, lwork,      \
                              rwork, iwork, ifail);                           \
  }

HWYBLAS_LAPACKE_HEEVR(c, c32, float)
HWYBLAS_LAPACKE_HEEVR(z, c64, double)

#undef HWYBLAS_LAPACKE_HEEVR

}  // extern "C"
