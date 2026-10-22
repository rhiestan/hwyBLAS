// LAPACKE functions of the generalized nonsymmetric eigenproblem (xGGBAL,
// xGGBAK, xGGHRD, xHGEQZ, xTGEVC, xTGEXC, xTGSEN, xTGSYL, xGGES, xGGEV) and
// of the generalized singular value decomposition (xGGSVD3, xGGSVP3,
// xTGSJA, xLAPMT); see lapacke.cc and lapacke_util.h.
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

using lapack::Lsame;

// Q or Z of xGGHRD and xHGEQZ: read for COMP = 'V', written for 'I' or 'V'.
template <typename T>
MatrixArg<T> Transformation(T* q, lapack_int n, lapack_int ldq, char comp,
                            lapack_int ld_info) {
  return Optional(q, n, n, ldq, ld_info, Lsame(comp, 'V'),
                  Lsame(comp, 'I') || Lsame(comp, 'V'));
}

// ---------------------------------------------------------------------------
// xGGBAL, xGGBAK

template <typename T>
lapack_int GgbalWork(int matrix_layout, char job, lapack_int n, T* a,
                     lapack_int lda, T* b, lapack_int ldb, lapack_int* ilo,
                     lapack_int* ihi, RealT<T>* lscale, RealT<T>* rscale,
                     RealT<T>* work) {
  // A and B are not referenced for JOB = 'N'.
  const bool used = Lsame(job, 'P') || Lsame(job, 'S') || Lsame(job, 'B');
  return CallWork<T, 2>(matrix_layout, Name<T>("ggbal_work"), false,
                        {{Optional(a, n, n, lda, -5, used, used),
                          Optional(b, n, n, ldb, -7, used, used)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Ggbal(job, n, p[0], ld[0], p[1], ld[1],
                                               ilo, ihi, lscale, rscale, work);
                        });
}

template <typename T>
lapack_int Ggbal(int matrix_layout, char job, lapack_int n, T* a,
                 lapack_int lda, T* b, lapack_int ldb, lapack_int* ilo,
                 lapack_int* ihi, RealT<T>* lscale, RealT<T>* rscale) {
  using R = RealT<T>;
  const std::string name = Name<T>("ggbal");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const bool used = Lsame(job, 'P') || Lsame(job, 'S') || Lsame(job, 'B');
    if (used && GeNancheck(matrix_layout, n, n, a, lda)) return -4;
    if (used && GeNancheck(matrix_layout, n, n, b, ldb)) return -6;
  }
  const lapack_int lwork =
      Lsame(job, 'S') || Lsame(job, 'B') ? std::max<lapack_int>(1, 6 * n) : 1;
  return WithWorkArray<R>(name, lwork, [&](R* work) {
    return GgbalWork(matrix_layout, job, n, a, lda, b, ldb, ilo, ihi, lscale,
                     rscale, work);
  });
}

template <typename T>
lapack_int GgbakWork(int matrix_layout, char job, char side, lapack_int n,
                     lapack_int ilo, lapack_int ihi, const RealT<T>* lscale,
                     const RealT<T>* rscale, lapack_int m, T* v,
                     lapack_int ldv) {
  return CallWork<T, 1>(matrix_layout, Name<T>("ggbak_work"), false,
                        {{{v, n, m, ldv, -11}}}, [&](auto& p, auto& ld) {
                          return lapack::Ggbak(job, side, n, ilo, ihi, lscale,
                                               rscale, m, p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Ggbak(int matrix_layout, char job, char side, lapack_int n,
                 lapack_int ilo, lapack_int ihi, const RealT<T>* lscale,
                 const RealT<T>* rscale, lapack_int m, T* v, lapack_int ldv) {
  const std::string name = Name<T>("ggbak");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (VectorNancheck(n, lscale, 1)) return -7;
    if (VectorNancheck(n, rscale, 1)) return -8;
    if (GeNancheck(matrix_layout, n, m, v, ldv)) return -10;
  }
  return GgbakWork(matrix_layout, job, side, n, ilo, ihi, lscale, rscale, m, v,
                   ldv);
}

// ---------------------------------------------------------------------------
// xGGHRD, xHGEQZ

template <typename T>
lapack_int GghrdWork(int matrix_layout, char compq, char compz, lapack_int n,
                     lapack_int ilo, lapack_int ihi, T* a, lapack_int lda, T* b,
                     lapack_int ldb, T* q, lapack_int ldq, T* z,
                     lapack_int ldz) {
  return CallWork<T, 4>(matrix_layout, Name<T>("gghrd_work"), false,
                        {{{a, n, n, lda, -8},
                          {b, n, n, ldb, -10},
                          Transformation(q, n, ldq, compq, -12),
                          Transformation(z, n, ldz, compz, -14)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gghrd(compq, compz, n, ilo, ihi, p[0],
                                               ld[0], p[1], ld[1], p[2], ld[2],
                                               p[3], ld[3]);
                        });
}

template <typename T>
lapack_int Gghrd(int matrix_layout, char compq, char compz, lapack_int n,
                 lapack_int ilo, lapack_int ihi, T* a, lapack_int lda, T* b,
                 lapack_int ldb, T* q, lapack_int ldq, T* z, lapack_int ldz) {
  const std::string name = Name<T>("gghrd");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -7;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -9;
    if ((Lsame(compq, 'I') || Lsame(compq, 'V')) &&
        GeNancheck(matrix_layout, n, n, q, ldq)) {
      return -11;
    }
    if ((Lsame(compz, 'I') || Lsame(compz, 'V')) &&
        GeNancheck(matrix_layout, n, n, z, ldz)) {
      return -13;
    }
  }
  return GghrdWork(matrix_layout, compq, compz, n, ilo, ihi, a, lda, b, ldb, q,
                   ldq, z, ldz);
}

// ALPHAR and ALPHAI for real T (alphai is unused for complex T, where
// alphar is ALPHA); RWORK for complex T only. H, Q, T, Z are checked in
// that order, as in the reference.
template <typename T>
lapack_int HgeqzWork(int matrix_layout, char job, char compq, char compz,
                     lapack_int n, lapack_int ilo, lapack_int ihi, T* h,
                     lapack_int ldh, T* t, lapack_int ldt, T* alphar, T* alphai,
                     T* beta, T* q, lapack_int ldq, T* z, lapack_int ldz,
                     T* work, lapack_int lwork, RealT<T>* rwork) {
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  return CallWork<T, 4>(
      matrix_layout, Name<T>("hgeqz_work"), lwork == -1,
      {{{h, n, n, ldh, -9},
        Transformation(q, n, ldq, compq, -16 + c),
        {t, n, n, ldt, -11},
        Transformation(z, n, ldz, compz, -18 + c)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)alphai;
          return lapack::Hgeqz(job, compq, compz, n, ilo, ihi, p[0], ld[0],
                               p[2], ld[2], alphar, beta, p[1], ld[1], p[3],
                               ld[3], work, lwork, rwork);
        } else {
          (void)rwork;
          return lapack::Hgeqz(job, compq, compz, n, ilo, ihi, p[0], ld[0],
                               p[2], ld[2], alphar, alphai, beta, p[1], ld[1],
                               p[3], ld[3], work, lwork);
        }
      });
}

template <typename T>
lapack_int Hgeqz(int matrix_layout, char job, char compq, char compz,
                 lapack_int n, lapack_int ilo, lapack_int ihi, T* h,
                 lapack_int ldh, T* t, lapack_int ldt, T* alphar, T* alphai,
                 T* beta, T* q, lapack_int ldq, T* z, lapack_int ldz) {
  using R = RealT<T>;
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  const std::string name = Name<T>("hgeqz");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, h, ldh)) return -8;
    if ((Lsame(compq, 'I') || Lsame(compq, 'V')) &&
        GeNancheck(matrix_layout, n, n, q, ldq)) {
      return -15 + c;
    }
    if (GeNancheck(matrix_layout, n, n, t, ldt)) return -10;
    if ((Lsame(compz, 'I') || Lsame(compz, 'V')) &&
        GeNancheck(matrix_layout, n, n, z, ldz)) {
      return -17 + c;
    }
  }
  // Allocate memory for working array(s): RWORK for complex T.
  return MaybeWithWorkArray<R>(
      name, IsComplex<T>::value, std::max<lapack_int>(1, n), [&](R* rwork) {
        return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
          return HgeqzWork(matrix_layout, job, compq, compz, n, ilo, ihi, h,
                           ldh, t, ldt, alphar, alphai, beta, q, ldq, z, ldz,
                           work, lwork, rwork);
        });
      });
}

// ---------------------------------------------------------------------------
// xTGEVC, xTGEXC, xTGSEN, xTGSYL

// WORK: 6*N (real T) or 2*N elements (complex T); RWORK (2*N) for complex
// T only. VL and VR (n x mm) are read for HOWMNY = 'B' and written, if
// wanted (SIDE). P, S, VL, VR are checked in that order, as in the
// reference.
template <typename T>
lapack_int TgevcWork(int matrix_layout, char side, char howmny,
                     const lapack_logical* select, lapack_int n, const T* s,
                     lapack_int lds, const T* p, lapack_int ldp, T* vl,
                     lapack_int ldvl, T* vr, lapack_int ldvr, lapack_int mm,
                     lapack_int* m, T* work, RealT<T>* rwork) {
  const bool left = Lsame(side, 'B') || Lsame(side, 'L');
  const bool right = Lsame(side, 'B') || Lsame(side, 'R');
  const bool back = Lsame(howmny, 'B');
  return CallWork<T, 4>(
      matrix_layout, Name<T>("tgevc_work"), false,
      {{Input(p, n, n, ldp, -9), Input(s, n, n, lds, -7),
        Optional(vl, n, mm, ldvl, -11, left && back, left),
        Optional(vr, n, mm, ldvr, -13, right && back, right)}},
      [&](auto& a, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          return lapack::Tgevc(side, howmny, select, n, a[1], ld[1], a[0],
                               ld[0], a[2], ld[2], a[3], ld[3], mm, m, work,
                               rwork);
        } else {
          (void)rwork;
          return lapack::Tgevc(side, howmny, select, n, a[1], ld[1], a[0],
                               ld[0], a[2], ld[2], a[3], ld[3], mm, m, work);
        }
      });
}

template <typename T>
lapack_int Tgevc(int matrix_layout, char side, char howmny,
                 const lapack_logical* select, lapack_int n, const T* s,
                 lapack_int lds, const T* p, lapack_int ldp, T* vl,
                 lapack_int ldvl, T* vr, lapack_int ldvr, lapack_int mm,
                 lapack_int* m) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const std::string name = Name<T>("tgevc");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, p, ldp)) return -8;
    if (GeNancheck(matrix_layout, n, n, s, lds)) return -6;
    if ((Lsame(side, 'B') || Lsame(side, 'L')) &&
        GeNancheck(matrix_layout, n, mm, vl, ldvl)) {
      return -10;
    }
    if ((Lsame(side, 'B') || Lsame(side, 'R')) &&
        GeNancheck(matrix_layout, n, mm, vr, ldvr)) {
      return -12;
    }
  }
  // Allocate memory for working array(s).
  return MaybeWithWorkArray<R>(
      name, kComplex, std::max<lapack_int>(1, 2 * n), [&](R* rwork) {
        return WithWorkArray<T>(
            name, std::max<lapack_int>(1, (kComplex ? 2 : 6) * n),
            [&](T* work) {
              return TgevcWork(matrix_layout, side, howmny, select, n, s, lds,
                               p, ldp, vl, ldvl, vr, ldvr, mm, m, work, rwork);
            });
      });
}

// Real T: IFST and ILST are in / out, WORK and LWORK are used. Complex T:
// IFST and ILST are passed by value (the reference LAPACKE_?tgexc does not
// return ILST), no workspace.
template <typename T>
lapack_int TgexcWork(int matrix_layout, lapack_logical wantq,
                     lapack_logical wantz, lapack_int n, T* a, lapack_int lda,
                     T* b, lapack_int ldb, T* q, lapack_int ldq, T* z,
                     lapack_int ldz, lapack_int* ifst, lapack_int* ilst,
                     T* work, lapack_int lwork) {
  return CallWork<T, 4>(
      matrix_layout, Name<T>("tgexc_work"), !IsComplex<T>::value && lwork == -1,
      {{{a, n, n, lda, -6},
        {b, n, n, ldb, -8},
        Optional(q, n, n, ldq, -10, wantq != 0, wantq != 0),
        Optional(z, n, n, ldz, -12, wantz != 0, wantz != 0)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)work;
          (void)lwork;
          return lapack::Tgexc(wantq != 0, wantz != 0, n, p[0], ld[0], p[1],
                               ld[1], p[2], ld[2], p[3], ld[3], *ifst, ilst);
        } else {
          return lapack::Tgexc(wantq != 0, wantz != 0, n, p[0], ld[0], p[1],
                               ld[1], p[2], ld[2], p[3], ld[3], ifst, ilst,
                               work, lwork);
        }
      });
}

template <typename T>
lapack_int Tgexc(int matrix_layout, lapack_logical wantq, lapack_logical wantz,
                 lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb, T* q,
                 lapack_int ldq, T* z, lapack_int ldz, lapack_int* ifst,
                 lapack_int* ilst) {
  const std::string name = Name<T>("tgexc");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -7;
    if (wantq && GeNancheck(matrix_layout, n, n, q, ldq)) return -9;
    if (wantz && GeNancheck(matrix_layout, n, n, z, ldz)) return -11;
  }
  if constexpr (IsComplex<T>::value) {
    return TgexcWork(matrix_layout, wantq, wantz, n, a, lda, b, ldb, q, ldq, z,
                     ldz, ifst, ilst, static_cast<T*>(nullptr), 0);
  } else {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return TgexcWork(matrix_layout, wantq, wantz, n, a, lda, b, ldb, q, ldq,
                       z, ldz, ifst, ilst, work, lwork);
    });
  }
}

// ALPHAR and ALPHAI for real T (alphai is unused for complex T, where
// alphar is ALPHA).
template <typename T>
lapack_int TgsenWork(int matrix_layout, lapack_int ijob, lapack_logical wantq,
                     lapack_logical wantz, const lapack_logical* select,
                     lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                     T* alphar, T* alphai, T* beta, T* q, lapack_int ldq, T* z,
                     lapack_int ldz, lapack_int* m, RealT<T>* pl, RealT<T>* pr,
                     RealT<T>* dif, T* work, lapack_int lwork,
                     lapack_int* iwork, lapack_int liwork) {
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  return CallWork<T, 4>(
      matrix_layout, Name<T>("tgsen_work"), liwork == -1 || lwork == -1,
      {{{a, n, n, lda, -8},
        {b, n, n, ldb, -10},
        Optional(q, n, n, ldq, -15 + c, wantq != 0, wantq != 0),
        Optional(z, n, n, ldz, -17 + c, wantz != 0, wantz != 0)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)alphai;
          return lapack::Tgsen(ijob, wantq != 0, wantz != 0, select, n, p[0],
                               ld[0], p[1], ld[1], alphar, beta, p[2], ld[2],
                               p[3], ld[3], m, pl, pr, dif, work, lwork, iwork,
                               liwork);
        } else {
          return lapack::Tgsen(ijob, wantq != 0, wantz != 0, select, n, p[0],
                               ld[0], p[1], ld[1], alphar, alphai, beta, p[2],
                               ld[2], p[3], ld[3], m, pl, pr, dif, work, lwork,
                               iwork, liwork);
        }
      });
}

template <typename T>
lapack_int Tgsen(int matrix_layout, lapack_int ijob, lapack_logical wantq,
                 lapack_logical wantz, const lapack_logical* select,
                 lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb,
                 T* alphar, T* alphai, T* beta, T* q, lapack_int ldq, T* z,
                 lapack_int ldz, lapack_int* m, RealT<T>* pl, RealT<T>* pr,
                 RealT<T>* dif) {
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  const std::string name = Name<T>("tgsen");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -7;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -9;
    if (wantq && GeNancheck(matrix_layout, n, n, q, ldq)) return -14 + c;
    if (wantz && GeNancheck(matrix_layout, n, n, z, ldz)) return -16 + c;
  }
  // Query optimal working array(s) size.
  T work_query = T(0);
  lapack_int iwork_query = 0;
  lapack_int info =
      TgsenWork(matrix_layout, ijob, wantq, wantz, select, n, a, lda, b, ldb,
                alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif,
                &work_query, -1, &iwork_query, -1);
  if (info != 0) return info;
  const lapack_int liwork = iwork_query;
  const lapack_int lwork = static_cast<lapack_int>(std::real(work_query));
  // Allocate memory for work arrays and call middle-level interface.
  return WithWorkArray<lapack_int>(name, liwork, [&](lapack_int* iwork) {
    return WithWorkArray<T>(name, lwork, [&](T* work) {
      return TgsenWork(matrix_layout, ijob, wantq, wantz, select, n, a, lda, b,
                       ldb, alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr,
                       dif, work, lwork, iwork, liwork);
    });
  });
}

template <typename T>
lapack_int TgsylWork(int matrix_layout, char trans, lapack_int ijob,
                     lapack_int m, lapack_int n, const T* a, lapack_int lda,
                     const T* b, lapack_int ldb, T* c, lapack_int ldc,
                     const T* d, lapack_int ldd, const T* e, lapack_int lde,
                     T* f, lapack_int ldf, RealT<T>* scale, RealT<T>* dif,
                     T* work, lapack_int lwork, lapack_int* iwork) {
  return CallWork<T, 6>(matrix_layout, Name<T>("tgsyl_work"), lwork == -1,
                        {{Input(a, m, m, lda, -7),
                          Input(b, n, n, ldb, -9),
                          {c, m, n, ldc, -11},
                          Input(d, m, m, ldd, -13),
                          Input(e, n, n, lde, -15),
                          {f, m, n, ldf, -17}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Tgsyl(trans, ijob, m, n, p[0], ld[0],
                                               p[1], ld[1], p[2], ld[2], p[3],
                                               ld[3], p[4], ld[4], p[5], ld[5],
                                               scale, dif, work, lwork, iwork);
                        });
}

template <typename T>
lapack_int Tgsyl(int matrix_layout, char trans, lapack_int ijob, lapack_int m,
                 lapack_int n, const T* a, lapack_int lda, const T* b,
                 lapack_int ldb, T* c, lapack_int ldc, const T* d,
                 lapack_int ldd, const T* e, lapack_int lde, T* f,
                 lapack_int ldf, RealT<T>* scale, RealT<T>* dif) {
  const std::string name = Name<T>("tgsyl");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, m, a, lda)) return -6;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -8;
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -10;
    if (GeNancheck(matrix_layout, m, m, d, ldd)) return -12;
    if (GeNancheck(matrix_layout, n, n, e, lde)) return -14;
    if (GeNancheck(matrix_layout, m, n, f, ldf)) return -16;
  }
  // Allocate memory for working array(s): IWORK, then the queried WORK.
  const lapack_int liwork =
      std::max<lapack_int>(1, m + n + (IsComplex<T>::value ? 2 : 6));
  return WithWorkArray<lapack_int>(name, liwork, [&](lapack_int* iwork) {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return TgsylWork(matrix_layout, trans, ijob, m, n, a, lda, b, ldb, c, ldc,
                       d, ldd, e, lde, f, ldf, scale, dif, work, lwork, iwork);
    });
  });
}

// ---------------------------------------------------------------------------
// xGGES, xGGEV

// ALPHAR and ALPHAI for real T (alphai is unused for complex T, where
// alphar is ALPHA); RWORK for complex T only. VSL and VSR are output only
// (JOBVSL / JOBVSR = 'V'); LDVSL and LDVSR are checked in row-major layout
// also for 'N', as in the reference.
template <typename T, typename Select>
lapack_int GgesWork(int matrix_layout, char jobvsl, char jobvsr, char sort,
                    Select selctg, lapack_int n, T* a, lapack_int lda, T* b,
                    lapack_int ldb, lapack_int* sdim, T* alphar, T* alphai,
                    T* beta, T* vsl, lapack_int ldvsl, T* vsr, lapack_int ldvsr,
                    T* work, lapack_int lwork, RealT<T>* rwork,
                    lapack_logical* bwork) {
  using R = RealT<T>;
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  return CallWork<T, 4>(
      matrix_layout, Name<T>("gges_work"), lwork == -1,
      {{{a, n, n, lda, -8},
        {b, n, n, ldb, -10},
        Eigenvectors(vsl, n, ldvsl, Lsame(jobvsl, 'V'), -16 + c),
        Eigenvectors(vsr, n, ldvsr, Lsame(jobvsr, 'V'), -18 + c)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)alphai;
          // LAPACK_C_SELECT2 / LAPACK_Z_SELECT2 take pointers to the complex
          // ALPHA and BETA, the routine pointers to their two parts.
          const auto sel =
              reinterpret_cast<lapack::GgesSelectComplex<R>>(selctg);
          return lapack::Gges(jobvsl, jobvsr, sort, sel, n, p[0], ld[0], p[1],
                              ld[1], sdim, alphar, beta, p[2], ld[2], p[3],
                              ld[3], work, lwork, rwork, bwork);
        } else {
          (void)rwork;
          return lapack::Gges(jobvsl, jobvsr, sort, selctg, n, p[0], ld[0],
                              p[1], ld[1], sdim, alphar, alphai, beta, p[2],
                              ld[2], p[3], ld[3], work, lwork, bwork);
        }
      });
}

template <typename T, typename Select>
lapack_int Gges(int matrix_layout, char jobvsl, char jobvsr, char sort,
                Select selctg, lapack_int n, T* a, lapack_int lda, T* b,
                lapack_int ldb, lapack_int* sdim, T* alphar, T* alphai, T* beta,
                T* vsl, lapack_int ldvsl, T* vsr, lapack_int ldvsr) {
  using R = RealT<T>;
  const std::string name = Name<T>("gges");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -7;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -9;
  }
  // Allocate memory for working array(s): BWORK for SORT = 'S', RWORK for
  // complex T.
  return MaybeWithWorkArray<lapack_logical>(
      name, Lsame(sort, 'S'), std::max<lapack_int>(1, n),
      [&](lapack_logical* bwork) {
        return MaybeWithWorkArray<R>(
            name, IsComplex<T>::value, std::max<lapack_int>(1, 8 * n),
            [&](R* rwork) {
              return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
                return GgesWork(matrix_layout, jobvsl, jobvsr, sort, selctg, n,
                                a, lda, b, ldb, sdim, alphar, alphai, beta, vsl,
                                ldvsl, vsr, ldvsr, work, lwork, rwork, bwork);
              });
            });
      });
}

// ALPHAR and ALPHAI for real T (alphai is unused for complex T, where
// alphar is ALPHA); RWORK for complex T only.
template <typename T>
lapack_int GgevWork(int matrix_layout, char jobvl, char jobvr, lapack_int n,
                    T* a, lapack_int lda, T* b, lapack_int ldb, T* alphar,
                    T* alphai, T* beta, T* vl, lapack_int ldvl, T* vr,
                    lapack_int ldvr, T* work, lapack_int lwork,
                    RealT<T>* rwork) {
  const lapack_int c = IsComplex<T>::value ? 1 : 0;
  return CallWork<T, 4>(
      matrix_layout, Name<T>("ggev_work"), lwork == -1,
      {{{a, n, n, lda, -6},
        {b, n, n, ldb, -8},
        Eigenvectors(vl, n, ldvl, Lsame(jobvl, 'V'), -13 + c),
        Eigenvectors(vr, n, ldvr, Lsame(jobvr, 'V'), -15 + c)}},
      [&](auto& p, auto& ld) {
        if constexpr (IsComplex<T>::value) {
          (void)alphai;
          return lapack::Ggev(jobvl, jobvr, n, p[0], ld[0], p[1], ld[1], alphar,
                              beta, p[2], ld[2], p[3], ld[3], work, lwork,
                              rwork);
        } else {
          (void)rwork;
          return lapack::Ggev(jobvl, jobvr, n, p[0], ld[0], p[1], ld[1], alphar,
                              alphai, beta, p[2], ld[2], p[3], ld[3], work,
                              lwork);
        }
      });
}

template <typename T>
lapack_int Ggev(int matrix_layout, char jobvl, char jobvr, lapack_int n, T* a,
                lapack_int lda, T* b, lapack_int ldb, T* alphar, T* alphai,
                T* beta, T* vl, lapack_int ldvl, T* vr, lapack_int ldvr) {
  using R = RealT<T>;
  const std::string name = Name<T>("ggev");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, n, b, ldb)) return -7;
  }
  // Allocate memory for working array(s): RWORK for complex T.
  return MaybeWithWorkArray<R>(
      name, IsComplex<T>::value, std::max<lapack_int>(1, 8 * n), [&](R* rwork) {
        return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
          return GgevWork(matrix_layout, jobvl, jobvr, n, a, lda, b, ldb,
                          alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork,
                          rwork);
        });
      });
}

// ---------------------------------------------------------------------------
// xGGSVD3, xGGSVP3, xTGSJA, xLAPMT

// U, V and Q are output only (JOBU = 'U', JOBV = 'V', JOBQ = 'Q'); their
// leading dimensions are checked in row-major layout in any case, as in the
// reference. RWORK for complex T only.
template <typename T>
lapack_int Ggsvd3Work(int matrix_layout, char jobu, char jobv, char jobq,
                      lapack_int m, lapack_int n, lapack_int p, lapack_int* k,
                      lapack_int* l, T* a, lapack_int lda, T* b, lapack_int ldb,
                      RealT<T>* alpha, RealT<T>* beta, T* u, lapack_int ldu,
                      T* v, lapack_int ldv, T* q, lapack_int ldq, T* work,
                      lapack_int lwork, RealT<T>* rwork, lapack_int* iwork) {
  const bool wantu = Lsame(jobu, 'U');
  const bool wantv = Lsame(jobv, 'V');
  const bool wantq = Lsame(jobq, 'Q');
  return CallWork<T, 5>(matrix_layout, Name<T>("ggsvd3_work"), lwork == -1,
                        {{{a, m, n, lda, -11},
                          {b, p, n, ldb, -13},
                          Optional(q, n, n, ldq, -21, false, wantq),
                          Optional(u, m, m, ldu, -17, false, wantu),
                          Optional(v, p, p, ldv, -19, false, wantv)}},
                        [&](auto& x, auto& ld) {
                          return lapack::Ggsvd3(
                              jobu, jobv, jobq, m, n, p, k, l, x[0], ld[0],
                              x[1], ld[1], alpha, beta, x[3], ld[3], x[4],
                              ld[4], x[2], ld[2], work, lwork, rwork, iwork);
                        });
}

template <typename T>
lapack_int Ggsvd3(int matrix_layout, char jobu, char jobv, char jobq,
                  lapack_int m, lapack_int n, lapack_int p, lapack_int* k,
                  lapack_int* l, T* a, lapack_int lda, T* b, lapack_int ldb,
                  RealT<T>* alpha, RealT<T>* beta, T* u, lapack_int ldu, T* v,
                  lapack_int ldv, T* q, lapack_int ldq, lapack_int* iwork) {
  using R = RealT<T>;
  const std::string name = Name<T>("ggsvd3");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -10;
    if (GeNancheck(matrix_layout, p, n, b, ldb)) return -12;
  }
  // Allocate memory for working array(s): RWORK for complex T.
  return MaybeWithWorkArray<R>(
      name, IsComplex<T>::value, std::max<lapack_int>(1, 2 * n), [&](R* rwork) {
        return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
          return Ggsvd3Work(matrix_layout, jobu, jobv, jobq, m, n, p, k, l, a,
                            lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq,
                            work, lwork, rwork, iwork);
        });
      });
}

// RWORK for complex T only.
template <typename T>
lapack_int Ggsvp3Work(int matrix_layout, char jobu, char jobv, char jobq,
                      lapack_int m, lapack_int p, lapack_int n, T* a,
                      lapack_int lda, T* b, lapack_int ldb, RealT<T> tola,
                      RealT<T> tolb, lapack_int* k, lapack_int* l, T* u,
                      lapack_int ldu, T* v, lapack_int ldv, T* q,
                      lapack_int ldq, lapack_int* iwork, RealT<T>* rwork,
                      T* tau, T* work, lapack_int lwork) {
  const bool wantu = Lsame(jobu, 'U');
  const bool wantv = Lsame(jobv, 'V');
  const bool wantq = Lsame(jobq, 'Q');
  return CallWork<T, 5>(matrix_layout, Name<T>("ggsvp3_work"), lwork == -1,
                        {{{a, m, n, lda, -9},
                          {b, p, n, ldb, -11},
                          Optional(q, n, n, ldq, -21, false, wantq),
                          Optional(u, m, m, ldu, -17, false, wantu),
                          Optional(v, p, p, ldv, -19, false, wantv)}},
                        [&](auto& x, auto& ld) {
                          return lapack::Ggsvp3(
                              jobu, jobv, jobq, m, p, n, x[0], ld[0], x[1],
                              ld[1], tola, tolb, k, l, x[3], ld[3], x[4], ld[4],
                              x[2], ld[2], iwork, rwork, tau, work, lwork);
                        });
}

template <typename T>
lapack_int Ggsvp3(int matrix_layout, char jobu, char jobv, char jobq,
                  lapack_int m, lapack_int p, lapack_int n, T* a,
                  lapack_int lda, T* b, lapack_int ldb, RealT<T> tola,
                  RealT<T> tolb, lapack_int* k, lapack_int* l, T* u,
                  lapack_int ldu, T* v, lapack_int ldv, T* q, lapack_int ldq) {
  using R = RealT<T>;
  const std::string name = Name<T>("ggsvp3");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -8;
    if (GeNancheck(matrix_layout, p, n, b, ldb)) return -10;
    if (VectorNancheck(1, &tola, 1)) return -12;
    if (VectorNancheck(1, &tolb, 1)) return -13;
  }
  // Allocate memory for working array(s): IWORK, RWORK (complex T), TAU,
  // then the queried WORK.
  const lapack_int size = std::max<lapack_int>(1, n);
  return WithWorkArray<lapack_int>(name, size, [&](lapack_int* iwork) {
    return MaybeWithWorkArray<R>(
        name, IsComplex<T>::value, std::max<lapack_int>(1, 2 * n),
        [&](R* rwork) {
          return WithWorkArray<T>(name, size, [&](T* tau) {
            return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
              return Ggsvp3Work(matrix_layout, jobu, jobv, jobq, m, p, n, a,
                                lda, b, ldb, tola, tolb, k, l, u, ldu, v, ldv,
                                q, ldq, iwork, rwork, tau, work, lwork);
            });
          });
        });
  });
}

// U, V and Q are read for JOBU = 'U', JOBV = 'V', JOBQ = 'Q' and written
// also for 'I'.
template <typename T>
lapack_int TgsjaWork(int matrix_layout, char jobu, char jobv, char jobq,
                     lapack_int m, lapack_int p, lapack_int n, lapack_int k,
                     lapack_int l, T* a, lapack_int lda, T* b, lapack_int ldb,
                     RealT<T> tola, RealT<T> tolb, RealT<T>* alpha,
                     RealT<T>* beta, T* u, lapack_int ldu, T* v, lapack_int ldv,
                     T* q, lapack_int ldq, T* work, lapack_int* ncycle) {
  return CallWork<T, 5>(matrix_layout, Name<T>("tgsja_work"), false,
                        {{{a, m, n, lda, -11},
                          {b, p, n, ldb, -13},
                          Optional(q, n, n, ldq, -23, Lsame(jobq, 'Q'),
                                   Lsame(jobq, 'I') || Lsame(jobq, 'Q')),
                          Optional(u, m, m, ldu, -19, Lsame(jobu, 'U'),
                                   Lsame(jobu, 'I') || Lsame(jobu, 'U')),
                          Optional(v, p, p, ldv, -21, Lsame(jobv, 'V'),
                                   Lsame(jobv, 'I') || Lsame(jobv, 'V'))}},
                        [&](auto& x, auto& ld) {
                          return lapack::Tgsja(
                              jobu, jobv, jobq, m, p, n, k, l, x[0], ld[0],
                              x[1], ld[1], tola, tolb, alpha, beta, x[3], ld[3],
                              x[4], ld[4], x[2], ld[2], work, ncycle);
                        });
}

template <typename T>
lapack_int Tgsja(int matrix_layout, char jobu, char jobv, char jobq,
                 lapack_int m, lapack_int p, lapack_int n, lapack_int k,
                 lapack_int l, T* a, lapack_int lda, T* b, lapack_int ldb,
                 RealT<T> tola, RealT<T> tolb, RealT<T>* alpha, RealT<T>* beta,
                 T* u, lapack_int ldu, T* v, lapack_int ldv, T* q,
                 lapack_int ldq, lapack_int* ncycle) {
  const std::string name = Name<T>("tgsja");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -10;
    if (GeNancheck(matrix_layout, p, n, b, ldb)) return -12;
    if ((Lsame(jobq, 'I') || Lsame(jobq, 'Q')) &&
        GeNancheck(matrix_layout, n, n, q, ldq)) {
      return -22;
    }
    if (VectorNancheck(1, &tola, 1)) return -14;
    if (VectorNancheck(1, &tolb, 1)) return -15;
    if ((Lsame(jobu, 'I') || Lsame(jobu, 'U')) &&
        GeNancheck(matrix_layout, m, m, u, ldu)) {
      return -18;
    }
    if ((Lsame(jobv, 'I') || Lsame(jobv, 'V')) &&
        GeNancheck(matrix_layout, p, p, v, ldv)) {
      return -20;
    }
  }
  return WithWorkArray<T>(name, std::max<lapack_int>(1, 2 * n), [&](T* work) {
    return TgsjaWork(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, a, lda, b,
                     ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q, ldq, work,
                     ncycle);
  });
}

template <typename T>
lapack_int LapmtWork(int matrix_layout, lapack_logical forwrd, lapack_int m,
                     lapack_int n, T* x, lapack_int ldx, lapack_int* k) {
  return CallWork<T, 1>(matrix_layout, Name<T>("lapmt_work"), false,
                        {{{x, m, n, ldx, -6}}}, [&](auto& p, auto& ld) {
                          lapack::Lapmt(forwrd != 0, m, n, p[0], ld[0], k);
                          return lapack_int{0};
                        });
}

template <typename T>
lapack_int Lapmt(int matrix_layout, lapack_logical forwrd, lapack_int m,
                 lapack_int n, T* x, lapack_int ldx, lapack_int* k) {
  const std::string name = Name<T>("lapmt");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, x, ldx)) return -5;
  return LapmtWork(matrix_layout, forwrd, m, n, x, ldx, k);
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose arguments are the same in all four precisions.
#define HWYBLAS_LAPACKE_GGEV(prefix, T, R)                                     \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggbal(                              \
      int layout, char job, lapack_int n, T* a, lapack_int lda, T* b,          \
      lapack_int ldb, lapack_int* ilo, lapack_int* ihi, R* lscale,             \
      R* rscale) {                                                             \
    return lapacke::Ggbal(layout, job, n, a, lda, b, ldb, ilo, ihi, lscale,    \
                          rscale);                                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggbal_work(                         \
      int layout, char job, lapack_int n, T* a, lapack_int lda, T* b,          \
      lapack_int ldb, lapack_int* ilo, lapack_int* ihi, R* lscale, R* rscale,  \
      R* work) {                                                               \
    return lapacke::GgbalWork(layout, job, n, a, lda, b, ldb, ilo, ihi,        \
                              lscale, rscale, work);                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggbak(                              \
      int layout, char job, char side, lapack_int n, lapack_int ilo,           \
      lapack_int ihi, const R* lscale, const R* rscale, lapack_int m, T* v,    \
      lapack_int ldv) {                                                        \
    return lapacke::Ggbak(layout, job, side, n, ilo, ihi, lscale, rscale, m,   \
                          v, ldv);                                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggbak_work(                         \
      int layout, char job, char side, lapack_int n, lapack_int ilo,           \
      lapack_int ihi, const R* lscale, const R* rscale, lapack_int m, T* v,    \
      lapack_int ldv) {                                                        \
    return lapacke::GgbakWork(layout, job, side, n, ilo, ihi, lscale, rscale,  \
                              m, v, ldv);                                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##gghrd(                              \
      int layout, char compq, char compz, lapack_int n, lapack_int ilo,        \
      lapack_int ihi, T* a, lapack_int lda, T* b, lapack_int ldb, T* q,        \
      lapack_int ldq, T* z, lapack_int ldz) {                                  \
    return lapacke::Gghrd(layout, compq, compz, n, ilo, ihi, a, lda, b, ldb,   \
                          q, ldq, z, ldz);                                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##gghrd_work(                         \
      int layout, char compq, char compz, lapack_int n, lapack_int ilo,        \
      lapack_int ihi, T* a, lapack_int lda, T* b, lapack_int ldb, T* q,        \
      lapack_int ldq, T* z, lapack_int ldz) {                                  \
    return lapacke::GghrdWork(layout, compq, compz, n, ilo, ihi, a, lda, b,    \
                              ldb, q, ldq, z, ldz);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsyl(                              \
      int layout, char trans, lapack_int ijob, lapack_int m, lapack_int n,     \
      const T* a, lapack_int lda, const T* b, lapack_int ldb, T* c,            \
      lapack_int ldc, const T* d, lapack_int ldd, const T* e, lapack_int lde,  \
      T* f, lapack_int ldf, R* scale, R* dif) {                                \
    return lapacke::Tgsyl(layout, trans, ijob, m, n, a, lda, b, ldb, c, ldc,   \
                          d, ldd, e, lde, f, ldf, scale, dif);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsyl_work(                         \
      int layout, char trans, lapack_int ijob, lapack_int m, lapack_int n,     \
      const T* a, lapack_int lda, const T* b, lapack_int ldb, T* c,            \
      lapack_int ldc, const T* d, lapack_int ldd, const T* e, lapack_int lde,  \
      T* f, lapack_int ldf, R* scale, R* dif, T* work, lapack_int lwork,       \
      lapack_int* iwork) {                                                     \
    return lapacke::TgsylWork(layout, trans, ijob, m, n, a, lda, b, ldb, c,    \
                              ldc, d, ldd, e, lde, f, ldf, scale, dif, work,   \
                              lwork, iwork);                                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggsvd3(                             \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int n, \
      lapack_int p, lapack_int* k, lapack_int* l, T* a, lapack_int lda, T* b,  \
      lapack_int ldb, R* alpha, R* beta, T* u, lapack_int ldu, T* v,           \
      lapack_int ldv, T* q, lapack_int ldq, lapack_int* iwork) {               \
    return lapacke::Ggsvd3(layout, jobu, jobv, jobq, m, n, p, k, l, a, lda, b, \
                           ldb, alpha, beta, u, ldu, v, ldv, q, ldq, iwork);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggsvp3(                             \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int p, \
      lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb, R tola,        \
      R tolb, lapack_int* k, lapack_int* l, T* u, lapack_int ldu, T* v,        \
      lapack_int ldv, T* q, lapack_int ldq) {                                  \
    return lapacke::Ggsvp3(layout, jobu, jobv, jobq, m, p, n, a, lda, b, ldb,  \
                           tola, tolb, k, l, u, ldu, v, ldv, q, ldq);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsja(                              \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int p, \
      lapack_int n, lapack_int k, lapack_int l, T* a, lapack_int lda, T* b,    \
      lapack_int ldb, R tola, R tolb, R* alpha, R* beta, T* u, lapack_int ldu, \
      T* v, lapack_int ldv, T* q, lapack_int ldq, lapack_int* ncycle) {        \
    return lapacke::Tgsja(layout, jobu, jobv, jobq, m, p, n, k, l, a, lda, b,  \
                          ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q,     \
                          ldq, ncycle);                                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsja_work(                         \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int p, \
      lapack_int n, lapack_int k, lapack_int l, T* a, lapack_int lda, T* b,    \
      lapack_int ldb, R tola, R tolb, R* alpha, R* beta, T* u, lapack_int ldu, \
      T* v, lapack_int ldv, T* q, lapack_int ldq, T* work,                     \
      lapack_int* ncycle) {                                                    \
    return lapacke::TgsjaWork(layout, jobu, jobv, jobq, m, p, n, k, l, a, lda, \
                              b, ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, \
                              q, ldq, work, ncycle);                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##lapmt(                              \
      int layout, lapack_logical forwrd, lapack_int m, lapack_int n, T* x,     \
      lapack_int ldx, lapack_int* k) {                                         \
    return lapacke::Lapmt(layout, forwrd, m, n, x, ldx, k);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##lapmt_work(                         \
      int layout, lapack_logical forwrd, lapack_int m, lapack_int n, T* x,     \
      lapack_int ldx, lapack_int* k) {                                         \
    return lapacke::LapmtWork(layout, forwrd, m, n, x, ldx, k);                \
  }

HWYBLAS_LAPACKE_GGEV(s, float, float)
HWYBLAS_LAPACKE_GGEV(d, double, double)
HWYBLAS_LAPACKE_GGEV(c, c32, float)
HWYBLAS_LAPACKE_GGEV(z, c64, double)

#undef HWYBLAS_LAPACKE_GGEV

#define HWYBLAS_LAPACKE_GGEV_REAL(prefix, P, R)                                \
  HWYBLAS_API lapack_int LAPACKE_##prefix##hgeqz(                              \
      int layout, char job, char compq, char compz, lapack_int n,              \
      lapack_int ilo, lapack_int ihi, R* h, lapack_int ldh, R* t,              \
      lapack_int ldt, R* alphar, R* alphai, R* beta, R* q, lapack_int ldq,     \
      R* z, lapack_int ldz) {                                                  \
    return lapacke::Hgeqz(layout, job, compq, compz, n, ilo, ihi, h, ldh, t,   \
                          ldt, alphar, alphai, beta, q, ldq, z, ldz);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##hgeqz_work(                         \
      int layout, char job, char compq, char compz, lapack_int n,              \
      lapack_int ilo, lapack_int ihi, R* h, lapack_int ldh, R* t,              \
      lapack_int ldt, R* alphar, R* alphai, R* beta, R* q, lapack_int ldq,     \
      R* z, lapack_int ldz, R* work, lapack_int lwork) {                       \
    return lapacke::HgeqzWork(layout, job, compq, compz, n, ilo, ihi, h, ldh,  \
                              t, ldt, alphar, alphai, beta, q, ldq, z, ldz,    \
                              work, lwork, static_cast<R*>(nullptr));          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgevc(                              \
      int layout, char side, char howmny, const lapack_logical* select,        \
      lapack_int n, const R* s, lapack_int lds, const R* p, lapack_int ldp,    \
      R* vl, lapack_int ldvl, R* vr, lapack_int ldvr, lapack_int mm,           \
      lapack_int* m) {                                                         \
    return lapacke::Tgevc(layout, side, howmny, select, n, s, lds, p, ldp, vl, \
                          ldvl, vr, ldvr, mm, m);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgevc_work(                         \
      int layout, char side, char howmny, const lapack_logical* select,        \
      lapack_int n, const R* s, lapack_int lds, const R* p, lapack_int ldp,    \
      R* vl, lapack_int ldvl, R* vr, lapack_int ldvr, lapack_int mm,           \
      lapack_int* m, R* work) {                                                \
    return lapacke::TgevcWork(layout, side, howmny, select, n, s, lds, p, ldp, \
                              vl, ldvl, vr, ldvr, mm, m, work,                 \
                              static_cast<R*>(nullptr));                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgexc(                              \
      int layout, lapack_logical wantq, lapack_logical wantz, lapack_int n,    \
      R* a, lapack_int lda, R* b, lapack_int ldb, R* q, lapack_int ldq, R* z,  \
      lapack_int ldz, lapack_int* ifst, lapack_int* ilst) {                    \
    return lapacke::Tgexc(layout, wantq, wantz, n, a, lda, b, ldb, q, ldq, z,  \
                          ldz, ifst, ilst);                                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgexc_work(                         \
      int layout, lapack_logical wantq, lapack_logical wantz, lapack_int n,    \
      R* a, lapack_int lda, R* b, lapack_int ldb, R* q, lapack_int ldq, R* z,  \
      lapack_int ldz, lapack_int* ifst, lapack_int* ilst, R* work,             \
      lapack_int lwork) {                                                      \
    return lapacke::TgexcWork(layout, wantq, wantz, n, a, lda, b, ldb, q, ldq, \
                              z, ldz, ifst, ilst, work, lwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsen(                              \
      int layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, \
      const lapack_logical* select, lapack_int n, R* a, lapack_int lda, R* b,  \
      lapack_int ldb, R* alphar, R* alphai, R* beta, R* q, lapack_int ldq,     \
      R* z, lapack_int ldz, lapack_int* m, R* pl, R* pr, R* dif) {             \
    return lapacke::Tgsen(layout, ijob, wantq, wantz, select, n, a, lda, b,    \
                          ldb, alphar, alphai, beta, q, ldq, z, ldz, m, pl,    \
                          pr, dif);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsen_work(                         \
      int layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, \
      const lapack_logical* select, lapack_int n, R* a, lapack_int lda, R* b,  \
      lapack_int ldb, R* alphar, R* alphai, R* beta, R* q, lapack_int ldq,     \
      R* z, lapack_int ldz, lapack_int* m, R* pl, R* pr, R* dif, R* work,      \
      lapack_int lwork, lapack_int* iwork, lapack_int liwork) {                \
    return lapacke::TgsenWork(layout, ijob, wantq, wantz, select, n, a, lda,   \
                              b, ldb, alphar, alphai, beta, q, ldq, z, ldz, m, \
                              pl, pr, dif, work, lwork, iwork, liwork);        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##gges(                               \
      int layout, char jobvsl, char jobvsr, char sort,                         \
      LAPACK_##P##_SELECT3 selctg, lapack_int n, R* a, lapack_int lda, R* b,   \
      lapack_int ldb, lapack_int* sdim, R* alphar, R* alphai, R* beta, R* vsl, \
      lapack_int ldvsl, R* vsr, lapack_int ldvsr) {                            \
    return lapacke::Gges(layout, jobvsl, jobvsr, sort, selctg, n, a, lda, b,   \
                         ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr,     \
                         ldvsr);                                               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##gges_work(                          \
      int layout, char jobvsl, char jobvsr, char sort,                         \
      LAPACK_##P##_SELECT3 selctg, lapack_int n, R* a, lapack_int lda, R* b,   \
      lapack_int ldb, lapack_int* sdim, R* alphar, R* alphai, R* beta, R* vsl, \
      lapack_int ldvsl, R* vsr, lapack_int ldvsr, R* work, lapack_int lwork,   \
      lapack_logical* bwork) {                                                 \
    return lapacke::GgesWork(layout, jobvsl, jobvsr, sort, selctg, n, a, lda,  \
                             b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl,   \
                             vsr, ldvsr, work, lwork,                          \
                             static_cast<R*>(nullptr), bwork);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggev(                               \
      int layout, char jobvl, char jobvr, lapack_int n, R* a, lapack_int lda,  \
      R* b, lapack_int ldb, R* alphar, R* alphai, R* beta, R* vl,              \
      lapack_int ldvl, R* vr, lapack_int ldvr) {                               \
    return lapacke::Ggev(layout, jobvl, jobvr, n, a, lda, b, ldb, alphar,      \
                         alphai, beta, vl, ldvl, vr, ldvr);                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggev_work(                          \
      int layout, char jobvl, char jobvr, lapack_int n, R* a, lapack_int lda,  \
      R* b, lapack_int ldb, R* alphar, R* alphai, R* beta, R* vl,              \
      lapack_int ldvl, R* vr, lapack_int ldvr, R* work, lapack_int lwork) {    \
    return lapacke::GgevWork(layout, jobvl, jobvr, n, a, lda, b, ldb, alphar,  \
                             alphai, beta, vl, ldvl, vr, ldvr, work, lwork,    \
                             static_cast<R*>(nullptr));                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggsvd3_work(                        \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int n, \
      lapack_int p, lapack_int* k, lapack_int* l, R* a, lapack_int lda, R* b,  \
      lapack_int ldb, R* alpha, R* beta, R* u, lapack_int ldu, R* v,           \
      lapack_int ldv, R* q, lapack_int ldq, R* work, lapack_int lwork,         \
      lapack_int* iwork) {                                                     \
    return lapacke::Ggsvd3Work(                                                \
        layout, jobu, jobv, jobq, m, n, p, k, l, a, lda, b, ldb, alpha, beta,  \
        u, ldu, v, ldv, q, ldq, work, lwork, static_cast<R*>(nullptr), iwork); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggsvp3_work(                        \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int p, \
      lapack_int n, R* a, lapack_int lda, R* b, lapack_int ldb, R tola,        \
      R tolb, lapack_int* k, lapack_int* l, R* u, lapack_int ldu, R* v,        \
      lapack_int ldv, R* q, lapack_int ldq, lapack_int* iwork, R* tau,         \
      R* work, lapack_int lwork) {                                             \
    return lapacke::Ggsvp3Work(layout, jobu, jobv, jobq, m, p, n, a, lda, b,   \
                               ldb, tola, tolb, k, l, u, ldu, v, ldv, q, ldq,  \
                               iwork, static_cast<R*>(nullptr), tau, work,     \
                               lwork);                                         \
  }

HWYBLAS_LAPACKE_GGEV_REAL(s, S, float)
HWYBLAS_LAPACKE_GGEV_REAL(d, D, double)

#undef HWYBLAS_LAPACKE_GGEV_REAL

#define HWYBLAS_LAPACKE_GGEV_COMPLEX(prefix, P, T, R)                          \
  HWYBLAS_API lapack_int LAPACKE_##prefix##hgeqz(                              \
      int layout, char job, char compq, char compz, lapack_int n,              \
      lapack_int ilo, lapack_int ihi, T* h, lapack_int ldh, T* t,              \
      lapack_int ldt, T* alpha, T* beta, T* q, lapack_int ldq, T* z,           \
      lapack_int ldz) {                                                        \
    return lapacke::Hgeqz(layout, job, compq, compz, n, ilo, ihi, h, ldh, t,   \
                          ldt, alpha, static_cast<T*>(nullptr), beta, q, ldq,  \
                          z, ldz);                                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##hgeqz_work(                         \
      int layout, char job, char compq, char compz, lapack_int n,              \
      lapack_int ilo, lapack_int ihi, T* h, lapack_int ldh, T* t,              \
      lapack_int ldt, T* alpha, T* beta, T* q, lapack_int ldq, T* z,           \
      lapack_int ldz, T* work, lapack_int lwork, R* rwork) {                   \
    return lapacke::HgeqzWork(layout, job, compq, compz, n, ilo, ihi, h, ldh,  \
                              t, ldt, alpha, static_cast<T*>(nullptr), beta,   \
                              q, ldq, z, ldz, work, lwork, rwork);             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgevc(                              \
      int layout, char side, char howmny, const lapack_logical* select,        \
      lapack_int n, const T* s, lapack_int lds, const T* p, lapack_int ldp,    \
      T* vl, lapack_int ldvl, T* vr, lapack_int ldvr, lapack_int mm,           \
      lapack_int* m) {                                                         \
    return lapacke::Tgevc(layout, side, howmny, select, n, s, lds, p, ldp, vl, \
                          ldvl, vr, ldvr, mm, m);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgevc_work(                         \
      int layout, char side, char howmny, const lapack_logical* select,        \
      lapack_int n, const T* s, lapack_int lds, const T* p, lapack_int ldp,    \
      T* vl, lapack_int ldvl, T* vr, lapack_int ldvr, lapack_int mm,           \
      lapack_int* m, T* work, R* rwork) {                                      \
    return lapacke::TgevcWork(layout, side, howmny, select, n, s, lds, p, ldp, \
                              vl, ldvl, vr, ldvr, mm, m, work, rwork);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgexc(                              \
      int layout, lapack_logical wantq, lapack_logical wantz, lapack_int n,    \
      T* a, lapack_int lda, T* b, lapack_int ldb, T* q, lapack_int ldq, T* z,  \
      lapack_int ldz, lapack_int ifst, lapack_int ilst) {                      \
    return lapacke::Tgexc(layout, wantq, wantz, n, a, lda, b, ldb, q, ldq, z,  \
                          ldz, &ifst, &ilst);                                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgexc_work(                         \
      int layout, lapack_logical wantq, lapack_logical wantz, lapack_int n,    \
      T* a, lapack_int lda, T* b, lapack_int ldb, T* q, lapack_int ldq, T* z,  \
      lapack_int ldz, lapack_int ifst, lapack_int ilst) {                      \
    return lapacke::TgexcWork(layout, wantq, wantz, n, a, lda, b, ldb, q, ldq, \
                              z, ldz, &ifst, &ilst, static_cast<T*>(nullptr),  \
                              0);                                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsen(                              \
      int layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, \
      const lapack_logical* select, lapack_int n, T* a, lapack_int lda, T* b,  \
      lapack_int ldb, T* alpha, T* beta, T* q, lapack_int ldq, T* z,           \
      lapack_int ldz, lapack_int* m, R* pl, R* pr, R* dif) {                   \
    return lapacke::Tgsen(layout, ijob, wantq, wantz, select, n, a, lda, b,    \
                          ldb, alpha, static_cast<T*>(nullptr), beta, q, ldq,  \
                          z, ldz, m, pl, pr, dif);                             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##tgsen_work(                         \
      int layout, lapack_int ijob, lapack_logical wantq, lapack_logical wantz, \
      const lapack_logical* select, lapack_int n, T* a, lapack_int lda, T* b,  \
      lapack_int ldb, T* alpha, T* beta, T* q, lapack_int ldq, T* z,           \
      lapack_int ldz, lapack_int* m, R* pl, R* pr, R* dif, T* work,            \
      lapack_int lwork, lapack_int* iwork, lapack_int liwork) {                \
    return lapacke::TgsenWork(layout, ijob, wantq, wantz, select, n, a, lda,   \
                              b, ldb, alpha, static_cast<T*>(nullptr), beta,   \
                              q, ldq, z, ldz, m, pl, pr, dif, work, lwork,     \
                              iwork, liwork);                                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##gges(                               \
      int layout, char jobvsl, char jobvsr, char sort,                         \
      LAPACK_##P##_SELECT2 selctg, lapack_int n, T* a, lapack_int lda, T* b,   \
      lapack_int ldb, lapack_int* sdim, T* alpha, T* beta, T* vsl,             \
      lapack_int ldvsl, T* vsr, lapack_int ldvsr) {                            \
    return lapacke::Gges(layout, jobvsl, jobvsr, sort, selctg, n, a, lda, b,   \
                         ldb, sdim, alpha, static_cast<T*>(nullptr), beta,     \
                         vsl, ldvsl, vsr, ldvsr);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##gges_work(                          \
      int layout, char jobvsl, char jobvsr, char sort,                         \
      LAPACK_##P##_SELECT2 selctg, lapack_int n, T* a, lapack_int lda, T* b,   \
      lapack_int ldb, lapack_int* sdim, T* alpha, T* beta, T* vsl,             \
      lapack_int ldvsl, T* vsr, lapack_int ldvsr, T* work, lapack_int lwork,   \
      R* rwork, lapack_logical* bwork) {                                       \
    return lapacke::GgesWork(layout, jobvsl, jobvsr, sort, selctg, n, a, lda,  \
                             b, ldb, sdim, alpha, static_cast<T*>(nullptr),    \
                             beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, \
                             bwork);                                           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggev(                               \
      int layout, char jobvl, char jobvr, lapack_int n, T* a, lapack_int lda,  \
      T* b, lapack_int ldb, T* alpha, T* beta, T* vl, lapack_int ldvl, T* vr,  \
      lapack_int ldvr) {                                                       \
    return lapacke::Ggev(layout, jobvl, jobvr, n, a, lda, b, ldb, alpha,       \
                         static_cast<T*>(nullptr), beta, vl, ldvl, vr, ldvr);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggev_work(                          \
      int layout, char jobvl, char jobvr, lapack_int n, T* a, lapack_int lda,  \
      T* b, lapack_int ldb, T* alpha, T* beta, T* vl, lapack_int ldvl, T* vr,  \
      lapack_int ldvr, T* work, lapack_int lwork, R* rwork) {                  \
    return lapacke::GgevWork(layout, jobvl, jobvr, n, a, lda, b, ldb, alpha,   \
                             static_cast<T*>(nullptr), beta, vl, ldvl, vr,     \
                             ldvr, work, lwork, rwork);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggsvd3_work(                        \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int n, \
      lapack_int p, lapack_int* k, lapack_int* l, T* a, lapack_int lda, T* b,  \
      lapack_int ldb, R* alpha, R* beta, T* u, lapack_int ldu, T* v,           \
      lapack_int ldv, T* q, lapack_int ldq, T* work, lapack_int lwork,         \
      R* rwork, lapack_int* iwork) {                                           \
    return lapacke::Ggsvd3Work(layout, jobu, jobv, jobq, m, n, p, k, l, a,     \
                               lda, b, ldb, alpha, beta, u, ldu, v, ldv, q,    \
                               ldq, work, lwork, rwork, iwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##prefix##ggsvp3_work(                        \
      int layout, char jobu, char jobv, char jobq, lapack_int m, lapack_int p, \
      lapack_int n, T* a, lapack_int lda, T* b, lapack_int ldb, R tola,        \
      R tolb, lapack_int* k, lapack_int* l, T* u, lapack_int ldu, T* v,        \
      lapack_int ldv, T* q, lapack_int ldq, lapack_int* iwork, R* rwork,       \
      T* tau, T* work, lapack_int lwork) {                                     \
    return lapacke::Ggsvp3Work(layout, jobu, jobv, jobq, m, p, n, a, lda, b,   \
                               ldb, tola, tolb, k, l, u, ldu, v, ldv, q, ldq,  \
                               iwork, rwork, tau, work, lwork);                \
  }

HWYBLAS_LAPACKE_GGEV_COMPLEX(c, C, c32, float)
HWYBLAS_LAPACKE_GGEV_COMPLEX(z, Z, c64, double)

#undef HWYBLAS_LAPACKE_GGEV_COMPLEX

}  // extern "C"
