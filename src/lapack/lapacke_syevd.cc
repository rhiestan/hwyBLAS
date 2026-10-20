// LAPACKE functions of the divide and conquer symmetric (Hermitian)
// eigensolvers and the generalized symmetric-definite eigenproblem: xSYEVD /
// xHEEVD, xSTEVD, xSTEDC, xORMTR / xUNMTR, xORMQL / xUNMQL, xSYGST /
// xHEGST, xSYGV / xHEGV and xSYGVD / xHEGVD (see lapacke.cc and
// lapacke_util.h).
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#include <stdlib.h>

#include <string>

#include "hwyblas/lapacke.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"
#include "src/lapack/lapacke_util.h"

namespace hwyblas {
namespace lapacke {
namespace {

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

// ---------------------------------------------------------------------------
// Standard problems

// xSYEVD / xHEEVD; rwork and lrwork for complex T only. In row-major
// layout, the triangle uplo of A is copied in, and back, or all of A (the
// eigenvectors) for jobz = 'V'.
template <typename T>
lapack_int HeevdWork(int matrix_layout, char jobz, char uplo, lapack_int n,
                     T* a, lapack_int lda, RealT<T>* w, T* work,
                     lapack_int lwork, RealT<T>* rwork, lapack_int lrwork,
                     lapack_int* iwork, lapack_int liwork) {
  MatrixArg<T> arg = Triangle<T>({a, n, n, lda, -6}, uplo);
  arg.out_general = lapack::Lsame(jobz, 'V');
  const bool query =
      liwork == -1 || lwork == -1 || (IsComplex<T>::value && lrwork == -1);
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("syevd_work", "heevd_work")),
      query, {{arg}}, [&](auto& p, auto& ld) {
        return lapack::Heevd(jobz, uplo, n, p[0], ld[0], w, work, lwork, rwork,
                             lrwork, iwork, liwork);
      });
}

template <typename T>
lapack_int Heevd(int matrix_layout, char jobz, char uplo, lapack_int n, T* a,
                 lapack_int lda, RealT<T>* w) {
  const std::string name = Name<T>(RealOrComplex<T>("syevd", "heevd"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) {
    return -5;
  }
  return WithWorkspaces<T>(
      name, [&](T* work, lapack_int lwork, RealT<T>* rwork, lapack_int lrwork,
                lapack_int* iwork, lapack_int liwork) {
        return HeevdWork(matrix_layout, jobz, uplo, n, a, lda, w, work, lwork,
                         rwork, lrwork, iwork, liwork);
      });
}

// xSTEDC; rwork and lrwork for complex T only. Z is copied in for compz =
// 'V' and out for 'V' and 'I' (not referenced for 'N').
template <typename T>
lapack_int StedcWork(int matrix_layout, char compz, lapack_int n, RealT<T>* d,
                     RealT<T>* e, T* z, lapack_int ldz, T* work,
                     lapack_int lwork, RealT<T>* rwork, lapack_int lrwork,
                     lapack_int* iwork, lapack_int liwork) {
  MatrixArg<T> arg{z, n, n, ldz, -7};
  arg.in = lapack::Lsame(compz, 'V');
  arg.out = arg.in || lapack::Lsame(compz, 'I');
  arg.used = arg.out;
  const bool query =
      liwork == -1 || lwork == -1 || (IsComplex<T>::value && lrwork == -1);
  return CallWork<T, 1>(matrix_layout, Name<T>("stedc_work"), query, {{arg}},
                        [&](auto& p, auto& ld) {
                          if constexpr (IsComplex<T>::value) {
                            return lapack::Stedc(compz, n, d, e, p[0], ld[0],
                                                 work, lwork, rwork, lrwork,
                                                 iwork, liwork);
                          } else {
                            return lapack::Stedc(compz, n, d, e, p[0], ld[0],
                                                 work, lwork, iwork, liwork);
                          }
                        });
}

template <typename T>
lapack_int Stedc(int matrix_layout, char compz, lapack_int n, RealT<T>* d,
                 RealT<T>* e, T* z, lapack_int ldz) {
  const std::string name = Name<T>("stedc");
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
  return WithWorkspaces<T>(
      name, [&](T* work, lapack_int lwork, RealT<T>* rwork, lapack_int lrwork,
                lapack_int* iwork, lapack_int liwork) {
        return StedcWork(matrix_layout, compz, n, d, e, z, ldz, work, lwork,
                         rwork, lrwork, iwork, liwork);
      });
}

// xSTEVD (real R only). Z is copied out for jobz = 'V'.
template <typename R>
lapack_int StevdWork(int matrix_layout, char jobz, lapack_int n, R* d, R* e,
                     R* z, lapack_int ldz, R* work, lapack_int lwork,
                     lapack_int* iwork, lapack_int liwork) {
  MatrixArg<R> arg{z, n, n, ldz, -7};
  arg.in = false;
  arg.out = lapack::Lsame(jobz, 'V');
  arg.used = arg.out;
  return CallWork<R, 1>(matrix_layout, Name<R>("stevd_work"),
                        liwork == -1 || lwork == -1, {{arg}},
                        [&](auto& p, auto& ld) {
                          return lapack::Stevd(jobz, n, d, e, p[0], ld[0], work,
                                               lwork, iwork, liwork);
                        });
}

template <typename R>
lapack_int Stevd(int matrix_layout, char jobz, lapack_int n, R* d, R* e, R* z,
                 lapack_int ldz) {
  const std::string name = Name<R>("stevd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -4;
    if (VectorNancheck(n - 1, e, 1)) return -5;
  }
  return WithWorkspaces<R>(
      name, [&](R* work, lapack_int lwork, R* /*rwork*/, lapack_int /*lrwork*/,
                lapack_int* iwork, lapack_int liwork) {
        return StevdWork(matrix_layout, jobz, n, d, e, z, ldz, work, lwork,
                         iwork, liwork);
      });
}

// ---------------------------------------------------------------------------
// The Q of the tridiagonal reduction and of a QL factorization

// xORMTR / xUNMTR: A (r x r, r = m for side = 'L', n for 'R') is copied
// in, C in and out.
template <typename T>
lapack_int UnmtrWork(int matrix_layout, char side, char uplo, char trans,
                     lapack_int m, lapack_int n, const T* a, lapack_int lda,
                     const T* tau, T* c, lapack_int ldc, T* work,
                     lapack_int lwork) {
  const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("ormtr_work", "unmtr_work")),
      lwork == -1, {{Input(a, r, r, lda, -8), {c, m, n, ldc, -11}}},
      [&](auto& p, auto& ld) {
        return lapack::Unmtr(side, uplo, trans, m, n, p[0], ld[0], tau, p[1],
                             ld[1], work, lwork);
      });
}

// The NaN check of A: the triangle uplo for real T, all of it for complex
// T (as the reference).
template <typename T>
lapack_int Unmtr(int matrix_layout, char side, char uplo, char trans,
                 lapack_int m, lapack_int n, const T* a, lapack_int lda,
                 const T* tau, T* c, lapack_int ldc) {
  const std::string name = Name<T>(RealOrComplex<T>("ormtr", "unmtr"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
    if constexpr (IsComplex<T>::value) {
      if (GeNancheck(matrix_layout, r, r, a, lda)) return -7;
    } else {
      if (TrNancheck(matrix_layout, uplo, 'N', r, a, lda)) return -7;
    }
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -10;
    if (VectorNancheck(r - 1, tau, 1)) return -9;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnmtrWork(matrix_layout, side, uplo, trans, m, n, a, lda, tau, c,
                     ldc, work, lwork);
  });
}

// xORMQL / xUNMQL: A (r x k) is copied in, C in and out.
template <typename T>
lapack_int UnmqlWork(int matrix_layout, char side, char trans, lapack_int m,
                     lapack_int n, lapack_int k, const T* a, lapack_int lda,
                     const T* tau, T* c, lapack_int ldc, T* work,
                     lapack_int lwork) {
  const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("ormql_work", "unmql_work")),
      lwork == -1, {{Input(a, r, k, lda, -8), {c, m, n, ldc, -11}}},
      [&](auto& p, auto& ld) {
        return lapack::Unmql(side, trans, m, n, k, p[0], ld[0], tau, p[1],
                             ld[1], work, lwork);
      });
}

template <typename T>
lapack_int Unmql(int matrix_layout, char side, char trans, lapack_int m,
                 lapack_int n, lapack_int k, const T* a, lapack_int lda,
                 const T* tau, T* c, lapack_int ldc) {
  const std::string name = Name<T>(RealOrComplex<T>("ormql", "unmql"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const lapack_int r = lapack::Lsame(side, 'L') ? m : n;
    if (GeNancheck(matrix_layout, r, k, a, lda)) return -7;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -10;
    if (VectorNancheck(k, tau, 1)) return -9;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UnmqlWork(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc,
                     work, lwork);
  });
}

// ---------------------------------------------------------------------------
// Generalized problems

// xSYGST / xHEGST: the triangle uplo of A is copied in and out, all of B
// in. (B is const, as in the reference; the complex routine conjugates
// parts of it in between and restores them.)
template <typename T>
lapack_int HegstWork(int matrix_layout, lapack_int itype, char uplo,
                     lapack_int n, T* a, lapack_int lda, const T* b,
                     lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("sygst_work", "hegst_work")),
      false, {{Triangle<T>({a, n, n, lda, -6}, uplo), Input(b, n, n, ldb, -8)}},
      [&](auto& p, auto& ld) {
        return lapack::Hegst(itype, uplo, n, p[0], ld[0], p[1], ld[1]);
      });
}

template <typename T>
lapack_int Hegst(int matrix_layout, lapack_int itype, char uplo, lapack_int n,
                 T* a, lapack_int lda, const T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>(RealOrComplex<T>("sygst", "hegst")), -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) return -5;
    if (TrNancheck(matrix_layout, uplo, 'N', n, b, ldb)) return -7;
  }
  return HegstWork(matrix_layout, itype, uplo, n, a, lda, b, ldb);
}

// xSYGV / xHEGV; rwork (max(1, 3*n-2) elements) for complex T only. All of
// A and B are copied in and out.
template <typename T>
lapack_int HegvWork(int matrix_layout, lapack_int itype, char jobz, char uplo,
                    lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                    RealT<T>* w, T* work, lapack_int lwork, RealT<T>* rwork) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("sygv_work", "hegv_work")),
      lwork == -1, {{{a, n, n, lda, -7}, {b, n, n, ldb, -9}}},
      [&](auto& p, auto& ld) {
        return lapack::Hegv(itype, jobz, uplo, n, p[0], ld[0], p[1], ld[1], w,
                            work, lwork, rwork);
      });
}

template <typename T>
lapack_int Hegv(int matrix_layout, lapack_int itype, char jobz, char uplo,
                lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                RealT<T>* w) {
  using R = RealT<T>;
  const std::string name = Name<T>(RealOrComplex<T>("sygv", "hegv"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) return -6;
    if (TrNancheck(matrix_layout, uplo, 'N', n, b, ldb)) return -8;
  }
  const auto call = [&](R* rwork) {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return HegvWork(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w,
                      work, lwork, rwork);
    });
  };
  if constexpr (IsComplex<T>::value) {
    return WithWorkArray<R>(name, std::max<lapack_int>(1, 3 * n - 2), call);
  } else {
    return call(nullptr);
  }
}

// xSYGVD / xHEGVD; rwork and lrwork for complex T only. All of A and B are
// copied in and out.
template <typename T>
lapack_int HegvdWork(int matrix_layout, lapack_int itype, char jobz, char uplo,
                     lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                     RealT<T>* w, T* work, lapack_int lwork, RealT<T>* rwork,
                     lapack_int lrwork, lapack_int* iwork, lapack_int liwork) {
  const bool query =
      liwork == -1 || lwork == -1 || (IsComplex<T>::value && lrwork == -1);
  return CallWork<T, 2>(
      matrix_layout, Name<T>(RealOrComplex<T>("sygvd_work", "hegvd_work")),
      query, {{{a, n, n, lda, -7}, {b, n, n, ldb, -9}}},
      [&](auto& p, auto& ld) {
        return lapack::Hegvd(itype, jobz, uplo, n, p[0], ld[0], p[1], ld[1], w,
                             work, lwork, rwork, lrwork, iwork, liwork);
      });
}

template <typename T>
lapack_int Hegvd(int matrix_layout, lapack_int itype, char jobz, char uplo,
                 lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                 RealT<T>* w) {
  const std::string name = Name<T>(RealOrComplex<T>("sygvd", "hegvd"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (TrNancheck(matrix_layout, uplo, 'N', n, a, lda)) return -6;
    if (TrNancheck(matrix_layout, uplo, 'N', n, b, ldb)) return -8;
  }
  return WithWorkspaces<T>(
      name, [&](T* work, lapack_int lwork, RealT<T>* rwork, lapack_int lrwork,
                lapack_int* iwork, lapack_int liwork) {
        return HegvdWork(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w,
                         work, lwork, rwork, lrwork, iwork, liwork);
      });
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose arguments are the same in all four precisions: `mtr`,
// `mql` and `gst` are "ormtr", "ormql" and "sygst" (real) or "unmtr", "unmql"
// and "hegst" (complex).
#define HWYBLAS_LAPACKE_SYEVD(p, T, R, mtr, mql, gst)                          \
  HWYBLAS_API lapack_int LAPACKE_##p##stedc(int layout, char compz,            \
                                            lapack_int n, R* d, R* e, T* z,    \
                                            lapack_int ldz) {                  \
    return lapacke::Stedc(layout, compz, n, d, e, z, ldz);                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mtr(                                     \
      int layout, char side, char uplo, char trans, lapack_int m,              \
      lapack_int n, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc) {                                                        \
    return lapacke::Unmtr(layout, side, uplo, trans, m, n, a, lda, tau, c,     \
                          ldc);                                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mtr##_work(                              \
      int layout, char side, char uplo, char trans, lapack_int m,              \
      lapack_int n, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc, T* work, lapack_int lwork) {                             \
    return lapacke::UnmtrWork(layout, side, uplo, trans, m, n, a, lda, tau, c, \
                              ldc, work, lwork);                               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mql(                                     \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc) {                                                        \
    return lapacke::Unmql(layout, side, trans, m, n, k, a, lda, tau, c, ldc);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##mql##_work(                              \
      int layout, char side, char trans, lapack_int m, lapack_int n,           \
      lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,            \
      lapack_int ldc, T* work, lapack_int lwork) {                             \
    return lapacke::UnmqlWork(layout, side, trans, m, n, k, a, lda, tau, c,    \
                              ldc, work, lwork);                               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gst(                                     \
      int layout, lapack_int itype, char uplo, lapack_int n, T* a,             \
      lapack_int lda, const T* b, lapack_int ldb) {                            \
    return lapacke::Hegst(layout, itype, uplo, n, a, lda, b, ldb);             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gst##_work(                              \
      int layout, lapack_int itype, char uplo, lapack_int n, T* a,             \
      lapack_int lda, const T* b, lapack_int ldb) {                            \
    return lapacke::HegstWork(layout, itype, uplo, n, a, lda, b, ldb);         \
  }

HWYBLAS_LAPACKE_SYEVD(s, float, float, ormtr, ormql, sygst)
HWYBLAS_LAPACKE_SYEVD(d, double, double, ormtr, ormql, sygst)
HWYBLAS_LAPACKE_SYEVD(c, c32, float, unmtr, unmql, hegst)
HWYBLAS_LAPACKE_SYEVD(z, c64, double, unmtr, unmql, hegst)

#undef HWYBLAS_LAPACKE_SYEVD

#define HWYBLAS_LAPACKE_SYEVD_REAL(p, R)                                       \
  HWYBLAS_API lapack_int LAPACKE_##p##syevd(int layout, char jobz, char uplo,  \
                                            lapack_int n, R* a,                \
                                            lapack_int lda, R* w) {            \
    return lapacke::Heevd(layout, jobz, uplo, n, a, lda, w);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syevd_work(                              \
      int layout, char jobz, char uplo, lapack_int n, R* a, lapack_int lda,    \
      R* w, R* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork) { \
    return lapacke::HeevdWork(layout, jobz, uplo, n, a, lda, w, work, lwork,   \
                              static_cast<R*>(nullptr), 0, iwork, liwork);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stedc_work(                              \
      int layout, char compz, lapack_int n, R* d, R* e, R* z, lapack_int ldz,  \
      R* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork) {       \
    return lapacke::StedcWork(layout, compz, n, d, e, z, ldz, work, lwork,     \
                              static_cast<R*>(nullptr), 0, iwork, liwork);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stevd(                                   \
      int layout, char jobz, lapack_int n, R* d, R* e, R* z, lapack_int ldz) { \
    return lapacke::Stevd(layout, jobz, n, d, e, z, ldz);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stevd_work(                              \
      int layout, char jobz, lapack_int n, R* d, R* e, R* z, lapack_int ldz,   \
      R* work, lapack_int lwork, lapack_int* iwork, lapack_int liwork) {       \
    return lapacke::StevdWork(layout, jobz, n, d, e, z, ldz, work, lwork,      \
                              iwork, liwork);                                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sygv(                                    \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, R* a,  \
      lapack_int lda, R* b, lapack_int ldb, R* w) {                            \
    return lapacke::Hegv(layout, itype, jobz, uplo, n, a, lda, b, ldb, w);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sygv_work(                               \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, R* a,  \
      lapack_int lda, R* b, lapack_int ldb, R* w, R* work, lapack_int lwork) { \
    return lapacke::HegvWork(layout, itype, jobz, uplo, n, a, lda, b, ldb, w,  \
                             work, lwork, static_cast<R*>(nullptr));           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sygvd(                                   \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, R* a,  \
      lapack_int lda, R* b, lapack_int ldb, R* w) {                            \
    return lapacke::Hegvd(layout, itype, jobz, uplo, n, a, lda, b, ldb, w);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sygvd_work(                              \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, R* a,  \
      lapack_int lda, R* b, lapack_int ldb, R* w, R* work, lapack_int lwork,   \
      lapack_int* iwork, lapack_int liwork) {                                  \
    return lapacke::HegvdWork(layout, itype, jobz, uplo, n, a, lda, b, ldb, w, \
                              work, lwork, static_cast<R*>(nullptr), 0, iwork, \
                              liwork);                                         \
  }

HWYBLAS_LAPACKE_SYEVD_REAL(s, float)
HWYBLAS_LAPACKE_SYEVD_REAL(d, double)

#undef HWYBLAS_LAPACKE_SYEVD_REAL

#define HWYBLAS_LAPACKE_SYEVD_COMPLEX(p, T, R)                                 \
  HWYBLAS_API lapack_int LAPACKE_##p##heevd(int layout, char jobz, char uplo,  \
                                            lapack_int n, T* a,                \
                                            lapack_int lda, R* w) {            \
    return lapacke::Heevd(layout, jobz, uplo, n, a, lda, w);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##heevd_work(                              \
      int layout, char jobz, char uplo, lapack_int n, T* a, lapack_int lda,    \
      R* w, T* work, lapack_int lwork, R* rwork, lapack_int lrwork,            \
      lapack_int* iwork, lapack_int liwork) {                                  \
    return lapacke::HeevdWork(layout, jobz, uplo, n, a, lda, w, work, lwork,   \
                              rwork, lrwork, iwork, liwork);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##stedc_work(                              \
      int layout, char compz, lapack_int n, R* d, R* e, T* z, lapack_int ldz,  \
      T* work, lapack_int lwork, R* rwork, lapack_int lrwork,                  \
      lapack_int* iwork, lapack_int liwork) {                                  \
    return lapacke::StedcWork(layout, compz, n, d, e, z, ldz, work, lwork,     \
                              rwork, lrwork, iwork, liwork);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hegv(                                    \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, T* a,  \
      lapack_int lda, T* b, lapack_int ldb, R* w) {                            \
    return lapacke::Hegv(layout, itype, jobz, uplo, n, a, lda, b, ldb, w);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hegv_work(                               \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, T* a,  \
      lapack_int lda, T* b, lapack_int ldb, R* w, T* work, lapack_int lwork,   \
      R* rwork) {                                                              \
    return lapacke::HegvWork(layout, itype, jobz, uplo, n, a, lda, b, ldb, w,  \
                             work, lwork, rwork);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hegvd(                                   \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, T* a,  \
      lapack_int lda, T* b, lapack_int ldb, R* w) {                            \
    return lapacke::Hegvd(layout, itype, jobz, uplo, n, a, lda, b, ldb, w);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hegvd_work(                              \
      int layout, lapack_int itype, char jobz, char uplo, lapack_int n, T* a,  \
      lapack_int lda, T* b, lapack_int ldb, R* w, T* work, lapack_int lwork,   \
      R* rwork, lapack_int lrwork, lapack_int* iwork, lapack_int liwork) {     \
    return lapacke::HegvdWork(layout, itype, jobz, uplo, n, a, lda, b, ldb, w, \
                              work, lwork, rwork, lrwork, iwork, liwork);      \
  }

HWYBLAS_LAPACKE_SYEVD_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_SYEVD_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_SYEVD_COMPLEX

}  // extern "C"
