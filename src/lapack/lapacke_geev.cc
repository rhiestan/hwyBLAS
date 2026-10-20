// LAPACKE functions of the nonsymmetric eigenproblem drivers xGEEV, xGEEVX
// and xGEESX and of xTRSNA (see lapacke.cc and lapacke_util.h). The
// reference LAPACKE has no function for xTREVC3 and xLAQTR (LAPACKE_xtrevc
// calls xTREVC, which is not ported).
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

// The lambdas test IsComplex<T>::value, not a local constexpr kComplex:
// MSVC does not treat a captured local constexpr as a constant there.

// The n x n eigenvector matrix V of xGEEV and xGEEVX, which the routine
// only writes, if wanted. LDV is checked in row-major layout as in the
// reference: ldv >= 1, and ldv >= n if the vectors are wanted.
template <typename T>
MatrixArg<T> Eigenvectors(T* v, lapack_int n, lapack_int ldv, bool wanted,
                          lapack_int ld_info) {
  MatrixArg<T> arg{v, n, wanted ? n : 1, ldv, ld_info};
  arg.in = false;
  arg.used = arg.out = wanted;
  return arg;
}

// Calls f with a work array of `size` elements of A if `needed`, else with
// a null pointer.
template <typename A, typename F>
lapack_int MaybeWithWorkArray(const std::string& name, bool needed,
                              lapack_int size, const F& f) {
  if (!needed) return f(static_cast<A*>(nullptr));
  return WithWorkArray<A>(name, size, f);
}

// ---------------------------------------------------------------------------
// xGEEV and xGEEVX

// WR and WI for real T (wi is unused for complex T, where wr is W); RWORK
// for complex T only.
template <typename T>
lapack_int GeevWork(int matrix_layout, char jobvl, char jobvr, lapack_int n,
                    T* a, lapack_int lda, T* wr, T* wi, T* vl, lapack_int ldvl,
                    T* vr, lapack_int ldvr, T* work, lapack_int lwork,
                    RealT<T>* rwork) {
  constexpr bool kComplex = IsComplex<T>::value;
  return CallWork<T, 3>(
      matrix_layout, Name<T>("geev_work"), lwork == -1,
      {{{a, n, n, lda, -6},
        Eigenvectors(vl, n, ldvl, lapack::Lsame(jobvl, 'V'),
                     kComplex ? -9 : -10),
        Eigenvectors(vr, n, ldvr, lapack::Lsame(jobvr, 'V'),
                     kComplex ? -11 : -12)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)wi;
          return lapack::Geev(jobvl, jobvr, n, p[0], ld[0], wr, p[1], ld[1],
                              p[2], ld[2], work, lwork, rwork);
        } else {
          (void)rwork;
          return lapack::Geev(jobvl, jobvr, n, p[0], ld[0], wr, wi, p[1], ld[1],
                              p[2], ld[2], work, lwork);
        }
      });
}

template <typename T>
lapack_int Geev(int matrix_layout, char jobvl, char jobvr, lapack_int n, T* a,
                lapack_int lda, T* wr, T* wi, T* vl, lapack_int ldvl, T* vr,
                lapack_int ldvr) {
  using R = RealT<T>;
  const std::string name = Name<T>("geev");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -5;
  // Allocate memory for working array(s): RWORK for complex T.
  return MaybeWithWorkArray<R>(
      name, IsComplex<T>::value, std::max<lapack_int>(1, 2 * n), [&](R* rwork) {
        return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
          return GeevWork(matrix_layout, jobvl, jobvr, n, a, lda, wr, wi, vl,
                          ldvl, vr, ldvr, work, lwork, rwork);
        });
      });
}

// WR and WI for real T (wi is unused for complex T, where wr is W); IWORK
// for real T, RWORK for complex T.
template <typename T>
lapack_int GeevxWork(int matrix_layout, char balanc, char jobvl, char jobvr,
                     char sense, lapack_int n, T* a, lapack_int lda, T* wr,
                     T* wi, T* vl, lapack_int ldvl, T* vr, lapack_int ldvr,
                     lapack_int* ilo, lapack_int* ihi, RealT<T>* scale,
                     RealT<T>* abnrm, RealT<T>* rconde, RealT<T>* rcondv,
                     T* work, lapack_int lwork, lapack_int* iwork,
                     RealT<T>* rwork) {
  constexpr bool kComplex = IsComplex<T>::value;
  return CallWork<T, 3>(
      matrix_layout, Name<T>("geevx_work"), lwork == -1,
      {{{a, n, n, lda, -8},
        Eigenvectors(vl, n, ldvl, lapack::Lsame(jobvl, 'V'),
                     kComplex ? -11 : -12),
        Eigenvectors(vr, n, ldvr, lapack::Lsame(jobvr, 'V'),
                     kComplex ? -13 : -14)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)wi;
          (void)iwork;
          return lapack::Geevx(balanc, jobvl, jobvr, sense, n, p[0], ld[0], wr,
                               p[1], ld[1], p[2], ld[2], ilo, ihi, scale, abnrm,
                               rconde, rcondv, work, lwork, rwork);
        } else {
          (void)rwork;
          return lapack::Geevx(balanc, jobvl, jobvr, sense, n, p[0], ld[0], wr,
                               wi, p[1], ld[1], p[2], ld[2], ilo, ihi, scale,
                               abnrm, rconde, rcondv, work, lwork, iwork);
        }
      });
}

template <typename T>
lapack_int Geevx(int matrix_layout, char balanc, char jobvl, char jobvr,
                 char sense, lapack_int n, T* a, lapack_int lda, T* wr, T* wi,
                 T* vl, lapack_int ldvl, T* vr, lapack_int ldvr,
                 lapack_int* ilo, lapack_int* ihi, RealT<T>* scale,
                 RealT<T>* abnrm, RealT<T>* rconde, RealT<T>* rcondv) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const std::string name = Name<T>("geevx");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -7;
  // Allocate memory for working array(s): IWORK for real T and SENSE = 'B'
  // or 'V', RWORK for complex T.
  const bool want_iwork =
      !kComplex && (lapack::Lsame(sense, 'B') || lapack::Lsame(sense, 'V'));
  return MaybeWithWorkArray<lapack_int>(
      name, want_iwork, std::max<lapack_int>(1, 2 * n - 2),
      [&](lapack_int* iwork) {
        return MaybeWithWorkArray<R>(
            name, kComplex, std::max<lapack_int>(1, 2 * n), [&](R* rwork) {
              return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
                return GeevxWork(matrix_layout, balanc, jobvl, jobvr, sense, n,
                                 a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi,
                                 scale, abnrm, rconde, rcondv, work, lwork,
                                 iwork, rwork);
              });
            });
      });
}

// ---------------------------------------------------------------------------
// xGEESX

// WR and WI for real T (wi is unused for complex T, where wr is W); IWORK
// for real T, RWORK for complex T. VS is output only (JOBVS = 'V'); LDVS is
// checked in row-major layout also for JOBVS = 'N', as in the reference.
template <typename T, typename Select>
lapack_int GeesxWork(int matrix_layout, char jobvs, char sort, Select select,
                     char sense, lapack_int n, T* a, lapack_int lda,
                     lapack_int* sdim, T* wr, T* wi, T* vs, lapack_int ldvs,
                     RealT<T>* rconde, RealT<T>* rcondv, T* work,
                     lapack_int lwork, lapack_int* iwork, lapack_int liwork,
                     RealT<T>* rwork, lapack_logical* bwork) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  MatrixArg<T> vs_arg{vs, n, n, ldvs, kComplex ? -12 : -13};
  vs_arg.in = false;
  vs_arg.used = vs_arg.out = lapack::Lsame(jobvs, 'V');
  const bool query = lwork == -1 || (!kComplex && liwork == -1);
  return CallWork<T, 2>(
      matrix_layout, Name<T>("geesx_work"), query,
      {{{a, n, n, lda, -8}, vs_arg}}, [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)wi;
          (void)iwork;
          (void)liwork;
          // LAPACK_C_SELECT1 / LAPACK_Z_SELECT1 take a pointer to the
          // complex eigenvalue, the routine one to its two parts.
          const auto sel =
              reinterpret_cast<lapack::GeesSelectComplex<R>>(select);
          return lapack::Geesx(jobvs, sort, sel, sense, n, p[0], ld[0], sdim,
                               wr, p[1], ld[1], rconde, rcondv, work, lwork,
                               rwork, bwork);
        } else {
          (void)rwork;
          return lapack::Geesx(jobvs, sort, select, sense, n, p[0], ld[0], sdim,
                               wr, wi, p[1], ld[1], rconde, rcondv, work, lwork,
                               iwork, liwork, bwork);
        }
      });
}

template <typename T, typename Select>
lapack_int Geesx(int matrix_layout, char jobvs, char sort, Select select,
                 char sense, lapack_int n, T* a, lapack_int lda,
                 lapack_int* sdim, T* wr, T* wi, T* vs, lapack_int ldvs,
                 RealT<T>* rconde, RealT<T>* rcondv) {
  using R = RealT<T>;
  const std::string name = Name<T>("geesx");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -7;
  // Allocate memory for working array(s): BWORK for SORT = 'S', RWORK for
  // complex T.
  return MaybeWithWorkArray<lapack_logical>(
      name, lapack::Lsame(sort, 'S'), std::max<lapack_int>(1, n),
      [&](lapack_logical* bwork) {
        if constexpr (IsComplex<T>::value) {
          return WithWorkArray<R>(
              name, std::max<lapack_int>(1, n), [&](R* rwork) {
                return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
                  return GeesxWork(matrix_layout, jobvs, sort, select, sense, n,
                                   a, lda, sdim, wr, wi, vs, ldvs, rconde,
                                   rcondv, work, lwork,
                                   static_cast<lapack_int*>(nullptr),
                                   lapack_int{0}, rwork, bwork);
                });
              });
        } else {
          // Query optimal working array(s) size.
          T work_query = T(0);
          lapack_int iwork_query = 0;
          const lapack_int info =
              GeesxWork(matrix_layout, jobvs, sort, select, sense, n, a, lda,
                        sdim, wr, wi, vs, ldvs, rconde, rcondv, &work_query, -1,
                        &iwork_query, -1, static_cast<R*>(nullptr), bwork);
          if (info != 0) return info;
          const lapack_int lwork = static_cast<lapack_int>(work_query);
          // Deviation: the reference passes LIWORK = IWORK(1) of the query,
          // which DGEESX sets to N*N/4 for SENSE = 'V' or 'B', 0 for N = 1,
          // and then rejects (LIWORK < 1). The function uses at least 1.
          const lapack_int liwork = std::max<lapack_int>(1, iwork_query);
          const bool want_iwork =
              lapack::Lsame(sense, 'B') || lapack::Lsame(sense, 'V');
          return WithWorkArray<lapack_int>(
              name, want_iwork ? liwork : 1, [&](lapack_int* iwork) {
                return WithWorkArray<T>(name, lwork, [&](T* work) {
                  return GeesxWork(matrix_layout, jobvs, sort, select, sense, n,
                                   a, lda, sdim, wr, wi, vs, ldvs, rconde,
                                   rcondv, work, lwork, iwork, liwork,
                                   static_cast<R*>(nullptr), bwork);
                });
              });
        }
      });
}

// ---------------------------------------------------------------------------
// xTRSNA

// IWORK for real T, RWORK for complex T. VL and VR (n x mm) are only read,
// and only for JOB = 'B' or 'E'; LDVL and LDVR are checked in row-major
// layout also for JOB = 'V', as in the reference.
template <typename T>
lapack_int TrsnaWork(int matrix_layout, char job, char howmny,
                     const lapack_logical* select, lapack_int n, const T* t,
                     lapack_int ldt, const T* vl, lapack_int ldvl, const T* vr,
                     lapack_int ldvr, RealT<T>* s, RealT<T>* sep, lapack_int mm,
                     lapack_int* m, T* work, lapack_int ldwork,
                     lapack_int* iwork, RealT<T>* rwork) {
  const bool vectors = lapack::Lsame(job, 'B') || lapack::Lsame(job, 'E');
  MatrixArg<T> vl_arg = Input(vl, n, mm, ldvl, -9);
  MatrixArg<T> vr_arg = Input(vr, n, mm, ldvr, -11);
  vl_arg.used = vr_arg.used = vectors;
  return CallWork<T, 3>(
      matrix_layout, Name<T>("trsna_work"), false,
      {{Input(t, n, n, ldt, -7), vl_arg, vr_arg}}, [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)iwork;
          return lapack::Trsna(job, howmny, select, n, p[0], ld[0], p[1], ld[1],
                               p[2], ld[2], s, sep, mm, m, work, ldwork, rwork);
        } else {
          (void)rwork;
          return lapack::Trsna(job, howmny, select, n, p[0], ld[0], p[1], ld[1],
                               p[2], ld[2], s, sep, mm, m, work, ldwork, iwork);
        }
      });
}

template <typename T>
lapack_int Trsna(int matrix_layout, char job, char howmny,
                 const lapack_logical* select, lapack_int n, const T* t,
                 lapack_int ldt, const T* vl, lapack_int ldvl, const T* vr,
                 lapack_int ldvr, RealT<T>* s, RealT<T>* sep, lapack_int mm,
                 lapack_int* m) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const std::string name = Name<T>("trsna");
  const lapack_int ldwork =
      lapack::Lsame(job, 'E') ? 1 : std::max<lapack_int>(1, n);
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  const bool vectors = lapack::Lsame(job, 'B') || lapack::Lsame(job, 'E');
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, t, ldt)) return -6;
    if (vectors && GeNancheck(matrix_layout, n, mm, vl, ldvl)) return -8;
    if (vectors && GeNancheck(matrix_layout, n, mm, vr, ldvr)) return -10;
  }
  // Allocate memory for working array(s) for JOB = 'B' or 'V': IWORK
  // (real T) or RWORK (complex T), and WORK.
  const bool sep_wanted = lapack::Lsame(job, 'B') || lapack::Lsame(job, 'V');
  return MaybeWithWorkArray<lapack_int>(
      name, sep_wanted && !kComplex, std::max<lapack_int>(1, 2 * (n - 1)),
      [&](lapack_int* iwork) {
        return MaybeWithWorkArray<R>(
            name, sep_wanted && kComplex, std::max<lapack_int>(1, n),
            [&](R* rwork) {
              return MaybeWithWorkArray<T>(
                  name, sep_wanted, ldwork * std::max<lapack_int>(1, n + 6),
                  [&](T* work) {
                    return TrsnaWork(matrix_layout, job, howmny, select, n, t,
                                     ldt, vl, ldvl, vr, ldvr, s, sep, mm, m,
                                     work, ldwork, iwork, rwork);
                  });
            });
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
#define HWYBLAS_LAPACKE_GEEV(p, T, R)                                         \
  HWYBLAS_API lapack_int LAPACKE_##p##trsna(                                  \
      int layout, char job, char howmny, const lapack_logical* select,        \
      lapack_int n, const T* t, lapack_int ldt, const T* vl, lapack_int ldvl, \
      const T* vr, lapack_int ldvr, R* s, R* sep, lapack_int mm,              \
      lapack_int* m) {                                                        \
    return lapacke::Trsna(layout, job, howmny, select, n, t, ldt, vl, ldvl,   \
                          vr, ldvr, s, sep, mm, m);                           \
  }

HWYBLAS_LAPACKE_GEEV(s, float, float)
HWYBLAS_LAPACKE_GEEV(d, double, double)
HWYBLAS_LAPACKE_GEEV(c, c32, float)
HWYBLAS_LAPACKE_GEEV(z, c64, double)

#undef HWYBLAS_LAPACKE_GEEV

#define HWYBLAS_LAPACKE_GEEV_REAL(p, P, R)                                     \
  HWYBLAS_API lapack_int LAPACKE_##p##geev(                                    \
      int layout, char jobvl, char jobvr, lapack_int n, R* a, lapack_int lda,  \
      R* wr, R* wi, R* vl, lapack_int ldvl, R* vr, lapack_int ldvr) {          \
    return lapacke::Geev(layout, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl,    \
                         vr, ldvr);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geev_work(                               \
      int layout, char jobvl, char jobvr, lapack_int n, R* a, lapack_int lda,  \
      R* wr, R* wi, R* vl, lapack_int ldvl, R* vr, lapack_int ldvr, R* work,   \
      lapack_int lwork) {                                                      \
    return lapacke::GeevWork(layout, jobvl, jobvr, n, a, lda, wr, wi, vl,      \
                             ldvl, vr, ldvr, work, lwork,                      \
                             static_cast<R*>(nullptr));                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geevx(                                   \
      int layout, char balanc, char jobvl, char jobvr, char sense,             \
      lapack_int n, R* a, lapack_int lda, R* wr, R* wi, R* vl,                 \
      lapack_int ldvl, R* vr, lapack_int ldvr, lapack_int* ilo,                \
      lapack_int* ihi, R* scale, R* abnrm, R* rconde, R* rcondv) {             \
    return lapacke::Geevx(layout, balanc, jobvl, jobvr, sense, n, a, lda, wr,  \
                          wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm,      \
                          rconde, rcondv);                                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geevx_work(                              \
      int layout, char balanc, char jobvl, char jobvr, char sense,             \
      lapack_int n, R* a, lapack_int lda, R* wr, R* wi, R* vl,                 \
      lapack_int ldvl, R* vr, lapack_int ldvr, lapack_int* ilo,                \
      lapack_int* ihi, R* scale, R* abnrm, R* rconde, R* rcondv, R* work,      \
      lapack_int lwork, lapack_int* iwork) {                                   \
    return lapacke::GeevxWork(layout, balanc, jobvl, jobvr, sense, n, a, lda,  \
                              wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale,     \
                              abnrm, rconde, rcondv, work, lwork, iwork,       \
                              static_cast<R*>(nullptr));                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geesx(                                   \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT2 select,          \
      char sense, lapack_int n, R* a, lapack_int lda, lapack_int* sdim, R* wr, \
      R* wi, R* vs, lapack_int ldvs, R* rconde, R* rcondv) {                   \
    return lapacke::Geesx(layout, jobvs, sort, select, sense, n, a, lda, sdim, \
                          wr, wi, vs, ldvs, rconde, rcondv);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geesx_work(                              \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT2 select,          \
      char sense, lapack_int n, R* a, lapack_int lda, lapack_int* sdim, R* wr, \
      R* wi, R* vs, lapack_int ldvs, R* rconde, R* rcondv, R* work,            \
      lapack_int lwork, lapack_int* iwork, lapack_int liwork,                  \
      lapack_logical* bwork) {                                                 \
    return lapacke::GeesxWork(layout, jobvs, sort, select, sense, n, a, lda,   \
                              sdim, wr, wi, vs, ldvs, rconde, rcondv, work,    \
                              lwork, iwork, liwork, static_cast<R*>(nullptr),  \
                              bwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsna_work(                              \
      int layout, char job, char howmny, const lapack_logical* select,         \
      lapack_int n, const R* t, lapack_int ldt, const R* vl, lapack_int ldvl,  \
      const R* vr, lapack_int ldvr, R* s, R* sep, lapack_int mm,               \
      lapack_int* m, R* work, lapack_int ldwork, lapack_int* iwork) {          \
    return lapacke::TrsnaWork(layout, job, howmny, select, n, t, ldt, vl,      \
                              ldvl, vr, ldvr, s, sep, mm, m, work, ldwork,     \
                              iwork, static_cast<R*>(nullptr));                \
  }

HWYBLAS_LAPACKE_GEEV_REAL(s, S, float)
HWYBLAS_LAPACKE_GEEV_REAL(d, D, double)

#undef HWYBLAS_LAPACKE_GEEV_REAL

#define HWYBLAS_LAPACKE_GEEV_COMPLEX(p, P, T, R)                               \
  HWYBLAS_API lapack_int LAPACKE_##p##geev(                                    \
      int layout, char jobvl, char jobvr, lapack_int n, T* a, lapack_int lda,  \
      T* w, T* vl, lapack_int ldvl, T* vr, lapack_int ldvr) {                  \
    return lapacke::Geev(layout, jobvl, jobvr, n, a, lda, w,                   \
                         static_cast<T*>(nullptr), vl, ldvl, vr, ldvr);        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geev_work(                               \
      int layout, char jobvl, char jobvr, lapack_int n, T* a, lapack_int lda,  \
      T* w, T* vl, lapack_int ldvl, T* vr, lapack_int ldvr, T* work,           \
      lapack_int lwork, R* rwork) {                                            \
    return lapacke::GeevWork(layout, jobvl, jobvr, n, a, lda, w,               \
                             static_cast<T*>(nullptr), vl, ldvl, vr, ldvr,     \
                             work, lwork, rwork);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geevx(                                   \
      int layout, char balanc, char jobvl, char jobvr, char sense,             \
      lapack_int n, T* a, lapack_int lda, T* w, T* vl, lapack_int ldvl, T* vr, \
      lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, R* scale, R* abnrm,   \
      R* rconde, R* rcondv) {                                                  \
    return lapacke::Geevx(layout, balanc, jobvl, jobvr, sense, n, a, lda, w,   \
                          static_cast<T*>(nullptr), vl, ldvl, vr, ldvr, ilo,   \
                          ihi, scale, abnrm, rconde, rcondv);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geevx_work(                              \
      int layout, char balanc, char jobvl, char jobvr, char sense,             \
      lapack_int n, T* a, lapack_int lda, T* w, T* vl, lapack_int ldvl, T* vr, \
      lapack_int ldvr, lapack_int* ilo, lapack_int* ihi, R* scale, R* abnrm,   \
      R* rconde, R* rcondv, T* work, lapack_int lwork, R* rwork) {             \
    return lapacke::GeevxWork(layout, balanc, jobvl, jobvr, sense, n, a, lda,  \
                              w, static_cast<T*>(nullptr), vl, ldvl, vr, ldvr, \
                              ilo, ihi, scale, abnrm, rconde, rcondv, work,    \
                              lwork, static_cast<lapack_int*>(nullptr),        \
                              rwork);                                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geesx(                                   \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT1 select,          \
      char sense, lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* w,  \
      T* vs, lapack_int ldvs, R* rconde, R* rcondv) {                          \
    return lapacke::Geesx(layout, jobvs, sort, select, sense, n, a, lda, sdim, \
                          w, static_cast<T*>(nullptr), vs, ldvs, rconde,       \
                          rcondv);                                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##geesx_work(                              \
      int layout, char jobvs, char sort, LAPACK_##P##_SELECT1 select,          \
      char sense, lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* w,  \
      T* vs, lapack_int ldvs, R* rconde, R* rcondv, T* work, lapack_int lwork, \
      R* rwork, lapack_logical* bwork) {                                       \
    return lapacke::GeesxWork(                                                 \
        layout, jobvs, sort, select, sense, n, a, lda, sdim, w,                \
        static_cast<T*>(nullptr), vs, ldvs, rconde, rcondv, work, lwork,       \
        static_cast<lapack_int*>(nullptr), 0, rwork, bwork);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##trsna_work(                              \
      int layout, char job, char howmny, const lapack_logical* select,         \
      lapack_int n, const T* t, lapack_int ldt, const T* vl, lapack_int ldvl,  \
      const T* vr, lapack_int ldvr, R* s, R* sep, lapack_int mm,               \
      lapack_int* m, T* work, lapack_int ldwork, R* rwork) {                   \
    return lapacke::TrsnaWork(layout, job, howmny, select, n, t, ldt, vl,      \
                              ldvl, vr, ldvr, s, sep, mm, m, work, ldwork,     \
                              static_cast<lapack_int*>(nullptr), rwork);       \
  }

HWYBLAS_LAPACKE_GEEV_COMPLEX(c, C, c32, float)
HWYBLAS_LAPACKE_GEEV_COMPLEX(z, Z, c64, double)

#undef HWYBLAS_LAPACKE_GEEV_COMPLEX

}  // extern "C"
