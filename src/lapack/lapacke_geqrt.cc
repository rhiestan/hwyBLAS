// LAPACKE functions of the QL factorization xGEQLF, of xORGRQ / xUNGRQ,
// and of the compact WY QR factorization xGEQRT, xGEQRT2, xGEQRT3 and
// xGEMQRT (see lapacke.cc and lapacke_util.h).
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
// QL factorization and the Q of an RQ factorization

template <typename T>
lapack_int GeqlfWork(int matrix_layout, lapack_int m, lapack_int n, T* a,
                     lapack_int lda, T* tau, T* work, lapack_int lwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("geqlf_work"), lwork == -1,
                        {{{a, m, n, lda, -5}}}, [&](auto& p, auto& ld) {
                          return lapack::Geqlf(m, n, p[0], ld[0], tau, work,
                                               lwork);
                        });
}

template <typename T>
lapack_int Geqlf(int matrix_layout, lapack_int m, lapack_int n, T* a,
                 lapack_int lda, T* tau) {
  const std::string name = Name<T>("geqlf");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return GeqlfWork(matrix_layout, m, n, a, lda, tau, work, lwork);
  });
}

// xORGRQ / xUNGRQ.
template <typename T>
lapack_int UngrqWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int k, T* a, lapack_int lda, const T* tau, T* work,
                     lapack_int lwork) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(RealOrComplex<T>("orgrq_work", "ungrq_work")),
      lwork == -1, {{{a, m, n, lda, -6}}}, [&](auto& p, auto& ld) {
        return lapack::Ungrq(m, n, k, p[0], ld[0], tau, work, lwork);
      });
}

template <typename T>
lapack_int Ungrq(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                 T* a, lapack_int lda, const T* tau) {
  const std::string name = Name<T>(RealOrComplex<T>("orgrq", "ungrq"));
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, m, n, a, lda)) return -5;
    if (VectorNancheck(k, tau, 1)) return -7;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return UngrqWork(matrix_layout, m, n, k, a, lda, tau, work, lwork);
  });
}

// ---------------------------------------------------------------------------
// QR factorization in the compact WY representation

// xGEQRT: in row-major layout, T is NB x min(M,N) (LDT >= min(M,N)); it is
// only written.
template <typename T>
lapack_int GeqrtWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int nb, T* a, lapack_int lda, T* t, lapack_int ldt,
                     T* work) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("geqrt_work"), false,
      {{{a, m, n, lda, -6}, {t, nb, std::min(m, n), ldt, -8, false}}},
      [&](auto& p, auto& ld) {
        return lapack::Geqrt(m, n, nb, p[0], ld[0], p[1], ld[1], work);
      });
}

template <typename T>
lapack_int Geqrt(int matrix_layout, lapack_int m, lapack_int n, lapack_int nb,
                 T* a, lapack_int lda, T* t, lapack_int ldt) {
  const std::string name = Name<T>("geqrt");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -5;
  return WithWorkArray<T>(
      name, std::max<lapack_int>(1, nb) * std::max<lapack_int>(1, n),
      [&](T* work) {
        return GeqrtWork(matrix_layout, m, n, nb, a, lda, t, ldt, work);
      });
}

// xGEQRT2 (recursive = false) and xGEQRT3 (recursive = true): T is N x N,
// only written.
template <typename T>
lapack_int Geqrt2Work(int matrix_layout, lapack_int m, lapack_int n, T* a,
                      lapack_int lda, T* t, lapack_int ldt, bool recursive) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>(recursive ? "geqrt3_work" : "geqrt2_work"), false,
      {{{a, m, n, lda, -5}, {t, n, n, ldt, -7, false}}},
      [&](auto& p, auto& ld) {
        return recursive ? lapack::Geqrt3(m, n, p[0], ld[0], p[1], ld[1])
                         : lapack::Geqrt2(m, n, p[0], ld[0], p[1], ld[1]);
      });
}

template <typename T>
lapack_int Geqrt2(int matrix_layout, lapack_int m, lapack_int n, T* a,
                  lapack_int lda, T* t, lapack_int ldt, bool recursive) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>(recursive ? "geqrt3" : "geqrt2"), -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, m, n, a, lda)) return -4;
  return Geqrt2Work(matrix_layout, m, n, a, lda, t, ldt, recursive);
}

// xGEMQRT: V is Q x K (Q = M for SIDE = 'L', N for SIDE = 'R'), T is NB x K,
// as xGEQRT returns them.
//
// Deviations: the reference LAPACKE (3.12.1) allocates a work array of
// M*NB elements, where LAPACK needs N*NB for SIDE = 'L' (overflow for
// N > M); here it has max(1, N)*NB (SIDE = 'L') or max(1, M)*NB elements.
// In row-major layout, the reference copies the LDV x K array V and the
// LDT x NB array T into temporaries with these leading dimensions, which
// reads beyond V for LDV > Q, gives xGEMQRT an invalid LDV for LDV < Q, and
// gives it only NB of the K columns of T (correct only for K <= NB); here
// V (LDV >= K) and T (LDT >= K, else INFO = -11) are copied as the Q x K
// and NB x K matrices.
template <typename T>
lapack_int GemqrtWork(int matrix_layout, char side, char trans, lapack_int m,
                      lapack_int n, lapack_int k, lapack_int nb, const T* v,
                      lapack_int ldv, const T* t, lapack_int ldt, T* c,
                      lapack_int ldc, T* work) {
  const lapack_int q = lapack::Lsame(side, 'L') ? m : n;
  return CallWork<T, 3>(matrix_layout, Name<T>("gemqrt_work"), false,
                        {{{c, m, n, ldc, -13},
                          Input(t, nb, k, ldt, -11),
                          Input(v, q, k, ldv, -9)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gemqrt(side, trans, m, n, k, nb, p[2],
                                                ld[2], p[1], ld[1], p[0], ld[0],
                                                work);
                        });
}

template <typename T>
lapack_int Gemqrt(int matrix_layout, char side, char trans, lapack_int m,
                  lapack_int n, lapack_int k, lapack_int nb, const T* v,
                  lapack_int ldv, const T* t, lapack_int ldt, T* c,
                  lapack_int ldc) {
  const std::string name = Name<T>("gemqrt");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  const bool left = lapack::Lsame(side, 'L');
  if (NanCheck()) {
    const lapack_int nrows_v = left ? m : (lapack::Lsame(side, 'R') ? n : 0);
    if (GeNancheck(matrix_layout, m, n, c, ldc)) return -12;
    if (GeNancheck(matrix_layout, nb, k, t, ldt)) return -10;
    if (GeNancheck(matrix_layout, nrows_v, k, v, ldv)) return -8;
  }
  return WithWorkArray<T>(
      name, std::max<lapack_int>(1, left ? n : m) * std::max<lapack_int>(1, nb),
      [&](T* work) {
        return GemqrtWork(matrix_layout, side, trans, m, n, k, nb, v, ldv, t,
                          ldt, c, ldc, work);
      });
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose names are the same in all four precisions.
#define HWYBLAS_LAPACKE_GEQRT(p, T)                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqlf(                                  \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau) { \
    return lapacke::Geqlf(layout, m, n, a, lda, tau);                         \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqlf_work(                             \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau,   \
      T* work, lapack_int lwork) {                                            \
    return lapacke::GeqlfWork(layout, m, n, a, lda, tau, work, lwork);        \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrt(                                  \
      int layout, lapack_int m, lapack_int n, lapack_int nb, T* a,            \
      lapack_int lda, T* t, lapack_int ldt) {                                 \
    return lapacke::Geqrt(layout, m, n, nb, a, lda, t, ldt);                  \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrt_work(                             \
      int layout, lapack_int m, lapack_int n, lapack_int nb, T* a,            \
      lapack_int lda, T* t, lapack_int ldt, T* work) {                        \
    return lapacke::GeqrtWork(layout, m, n, nb, a, lda, t, ldt, work);        \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrt2(                                 \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* t,     \
      lapack_int ldt) {                                                       \
    return lapacke::Geqrt2(layout, m, n, a, lda, t, ldt, false);              \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrt2_work(                            \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* t,     \
      lapack_int ldt) {                                                       \
    return lapacke::Geqrt2Work(layout, m, n, a, lda, t, ldt, false);          \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrt3(                                 \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* t,     \
      lapack_int ldt) {                                                       \
    return lapacke::Geqrt2(layout, m, n, a, lda, t, ldt, true);               \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##geqrt3_work(                            \
      int layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* t,     \
      lapack_int ldt) {                                                       \
    return lapacke::Geqrt2Work(layout, m, n, a, lda, t, ldt, true);           \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##gemqrt(                                 \
      int layout, char side, char trans, lapack_int m, lapack_int n,          \
      lapack_int k, lapack_int nb, const T* v, lapack_int ldv, const T* t,    \
      lapack_int ldt, T* c, lapack_int ldc) {                                 \
    return lapacke::Gemqrt(layout, side, trans, m, n, k, nb, v, ldv, t, ldt,  \
                           c, ldc);                                           \
  }                                                                           \
  HWYBLAS_API lapack_int LAPACKE_##p##gemqrt_work(                            \
      int layout, char side, char trans, lapack_int m, lapack_int n,          \
      lapack_int k, lapack_int nb, const T* v, lapack_int ldv, const T* t,    \
      lapack_int ldt, T* c, lapack_int ldc, T* work) {                        \
    return lapacke::GemqrtWork(layout, side, trans, m, n, k, nb, v, ldv, t,   \
                               ldt, c, ldc, work);                            \
  }

HWYBLAS_LAPACKE_GEQRT(s, float)
HWYBLAS_LAPACKE_GEQRT(d, double)
HWYBLAS_LAPACKE_GEQRT(c, c32)
HWYBLAS_LAPACKE_GEQRT(z, c64)

#undef HWYBLAS_LAPACKE_GEQRT

// The functions with a real and a complex version: `gen` is the middle of
// the names ("org" / "ung").
#define HWYBLAS_LAPACKE_UNGRQ(p, gen, T)                                  \
  HWYBLAS_API lapack_int LAPACKE_##p##gen##rq(                            \
      int layout, lapack_int m, lapack_int n, lapack_int k, T* a,         \
      lapack_int lda, const T* tau) {                                     \
    return lapacke::Ungrq(layout, m, n, k, a, lda, tau);                  \
  }                                                                       \
  HWYBLAS_API lapack_int LAPACKE_##p##gen##rq_work(                       \
      int layout, lapack_int m, lapack_int n, lapack_int k, T* a,         \
      lapack_int lda, const T* tau, T* work, lapack_int lwork) {          \
    return lapacke::UngrqWork(layout, m, n, k, a, lda, tau, work, lwork); \
  }

HWYBLAS_LAPACKE_UNGRQ(s, org, float)
HWYBLAS_LAPACKE_UNGRQ(d, org, double)
HWYBLAS_LAPACKE_UNGRQ(c, ung, c32)
HWYBLAS_LAPACKE_UNGRQ(z, ung, c64)

#undef HWYBLAS_LAPACKE_UNGRQ

}  // extern "C"
