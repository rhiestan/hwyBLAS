// LAPACKE functions of the band and tridiagonal routines: xGBTRF, xGBTRS,
// xGBSV, xGBCON, xPBTRF, xPBTRS, xPBSV, xGTTRF, xGTTRS, xGTSV, xPTTRF,
// xPTTRS and xPTSV (see lapacke.cc and lapacke_util.h). In row-major
// layout, a band matrix is the transpose of LAPACK's band storage: band
// row i (0-based) of column j at ab[i*ldab + j], with ldab >= n.
//
// Derived from LAPACKE, Copyright (c) 2010, Intel Corp. (BSD license), and
// LAPACK, see LICENSE-LAPACK.
#include <string>

#include "hwyblas/lapacke.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"
#include "src/lapack/lapacke_util.h"

namespace hwyblas {
namespace lapacke {
namespace {

// ---------------------------------------------------------------------------
// General band matrices

template <typename T>
lapack_int GbtrfWork(int matrix_layout, lapack_int m, lapack_int n,
                     lapack_int kl, lapack_int ku, T* ab, lapack_int ldab,
                     lapack_int* ipiv) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gbtrf_work"), false,
                        {{Band(ab, m, n, kl, kl + ku, ldab, -7)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gbtrf(m, n, kl, ku, p[0], ld[0], ipiv);
                        });
}

template <typename T>
lapack_int Gbtrf(int matrix_layout, lapack_int m, lapack_int n, lapack_int kl,
                 lapack_int ku, T* ab, lapack_int ldab, lapack_int* ipiv) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gbtrf"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GbNancheck(matrix_layout, m, n, kl, kl + ku, ab, ldab)) return -6;
  }
  return GbtrfWork(matrix_layout, m, n, kl, ku, ab, ldab, ipiv);
}

template <typename T>
lapack_int GbtrsWork(int matrix_layout, char trans, lapack_int n,
                     lapack_int kl, lapack_int ku, lapack_int nrhs,
                     const T* ab, lapack_int ldab, const lapack_int* ipiv, T* b,
                     lapack_int ldb) {
  MatrixArg<T> band = Band(const_cast<T*>(ab), n, n, kl, kl + ku, ldab, -8);
  band.out = false;
  return CallWork<T, 2>(matrix_layout, Name<T>("gbtrs_work"), false,
                        {{band, {b, n, nrhs, ldb, -11}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gbtrs(trans, n, kl, ku, nrhs, p[0],
                                               ld[0], ipiv, p[1], ld[1]);
                        });
}

template <typename T>
lapack_int Gbtrs(int matrix_layout, char trans, lapack_int n, lapack_int kl,
                 lapack_int ku, lapack_int nrhs, const T* ab, lapack_int ldab,
                 const lapack_int* ipiv, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gbtrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GbNancheck(matrix_layout, n, n, kl, kl + ku, ab, ldab)) return -7;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -10;
  }
  return GbtrsWork(matrix_layout, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b,
                   ldb);
}

template <typename T>
lapack_int GbsvWork(int matrix_layout, lapack_int n, lapack_int kl,
                    lapack_int ku, lapack_int nrhs, T* ab, lapack_int ldab,
                    lapack_int* ipiv, T* b, lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("gbsv_work"), false,
      {{Band(ab, n, n, kl, kl + ku, ldab, -7), {b, n, nrhs, ldb, -10}}},
      [&](auto& p, auto& ld) {
        return lapack::Gbsv(n, kl, ku, nrhs, p[0], ld[0], ipiv, p[1], ld[1]);
      });
}

// As the reference, the NaN check also looks at the first KL rows of AB,
// which xGBSV only uses as workspace.
template <typename T>
lapack_int Gbsv(int matrix_layout, lapack_int n, lapack_int kl, lapack_int ku,
                lapack_int nrhs, T* ab, lapack_int ldab, lapack_int* ipiv, T* b,
                lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gbsv"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GbNancheck(matrix_layout, n, n, kl, kl + ku, ab, ldab)) return -6;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -9;
  }
  return GbsvWork(matrix_layout, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

template <typename T>
lapack_int GbconWork(int matrix_layout, char norm, lapack_int n, lapack_int kl,
                     lapack_int ku, const T* ab, lapack_int ldab,
                     const lapack_int* ipiv, RealT<T> anorm, RealT<T>* rcond,
                     T* work, RealT<T>* rwork, lapack_int* iwork) {
  MatrixArg<T> band = Band(const_cast<T*>(ab), n, n, kl, kl + ku, ldab, -7);
  band.out = false;
  return CallWork<T, 1>(matrix_layout, Name<T>("gbcon_work"), false, {{band}},
                        [&](auto& p, auto& ld) {
                          return lapack::Gbcon(norm, n, kl, ku, p[0], ld[0],
                                               ipiv, anorm, rcond, work, rwork,
                                               iwork);
                        });
}

template <typename T>
lapack_int Gbcon(int matrix_layout, char norm, lapack_int n, lapack_int kl,
                 lapack_int ku, const T* ab, lapack_int ldab,
                 const lapack_int* ipiv, RealT<T> anorm, RealT<T>* rcond) {
  const std::string name = Name<T>("gbcon");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (GbNancheck(matrix_layout, n, n, kl, kl + ku, ab, ldab)) return -6;
    if (VectorNancheck(1, &anorm, 1)) return -9;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, 2 * n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return GbconWork(matrix_layout, norm, n, kl, ku, ab,
                                        ldab, ipiv, anorm, rcond, work, rwork,
                                        iwork);
                     });
}

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) positive definite band matrices

template <typename T>
lapack_int PbtrfWork(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                     T* ab, lapack_int ldab) {
  return CallWork<T, 1>(matrix_layout, Name<T>("pbtrf_work"), false,
                        {{SymmetricBand(ab, uplo, n, kd, ldab, -6)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Pbtrf(uplo, n, kd, p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Pbtrf(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                 T* ab, lapack_int ldab) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("pbtrf"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (PbNancheck(matrix_layout, uplo, n, kd, ab, ldab)) return -5;
  }
  return PbtrfWork(matrix_layout, uplo, n, kd, ab, ldab);
}

template <typename T>
lapack_int PbtrsWork(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                     lapack_int nrhs, const T* ab, lapack_int ldab, T* b,
                     lapack_int ldb) {
  MatrixArg<T> band =
      SymmetricBand(const_cast<T*>(ab), uplo, n, kd, ldab, -7);
  band.out = false;
  return CallWork<T, 2>(matrix_layout, Name<T>("pbtrs_work"), false,
                        {{band, {b, n, nrhs, ldb, -9}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Pbtrs(uplo, n, kd, nrhs, p[0], ld[0],
                                               p[1], ld[1]);
                        });
}

template <typename T>
lapack_int Pbtrs(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                 lapack_int nrhs, const T* ab, lapack_int ldab, T* b,
                 lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("pbtrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (PbNancheck(matrix_layout, uplo, n, kd, ab, ldab)) return -6;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -8;
  }
  return PbtrsWork(matrix_layout, uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

template <typename T>
lapack_int PbsvWork(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                    lapack_int nrhs, T* ab, lapack_int ldab, T* b,
                    lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("pbsv_work"), false,
      {{SymmetricBand(ab, uplo, n, kd, ldab, -7), {b, n, nrhs, ldb, -9}}},
      [&](auto& p, auto& ld) {
        return lapack::Pbsv(uplo, n, kd, nrhs, p[0], ld[0], p[1], ld[1]);
      });
}

template <typename T>
lapack_int Pbsv(int matrix_layout, char uplo, lapack_int n, lapack_int kd,
                lapack_int nrhs, T* ab, lapack_int ldab, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("pbsv"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (PbNancheck(matrix_layout, uplo, n, kd, ab, ldab)) return -6;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -8;
  }
  return PbsvWork(matrix_layout, uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

// ---------------------------------------------------------------------------
// Tridiagonal matrices. xGTTRF and xPTTRF have no matrix_layout.

template <typename T>
lapack_int GttrfWork(lapack_int n, T* dl, T* d, T* du, T* du2,
                     lapack_int* ipiv) {
  return lapack::Gttrf(n, dl, d, du, du2, ipiv);
}

template <typename T>
lapack_int Gttrf(lapack_int n, T* dl, T* d, T* du, T* du2, lapack_int* ipiv) {
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -3;
    if (VectorNancheck(n - 1, dl, 1)) return -2;
    if (VectorNancheck(n - 1, du, 1)) return -4;
  }
  return GttrfWork(n, dl, d, du, du2, ipiv);
}

template <typename T>
lapack_int GttrsWork(int matrix_layout, char trans, lapack_int n,
                     lapack_int nrhs, const T* dl, const T* d, const T* du,
                     const T* du2, const lapack_int* ipiv, T* b,
                     lapack_int ldb) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gttrs_work"), false,
                        {{{b, n, nrhs, ldb, -11}}}, [&](auto& p, auto& ld) {
                          return lapack::Gttrs(trans, n, nrhs, dl, d, du, du2,
                                               ipiv, p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Gttrs(int matrix_layout, char trans, lapack_int n, lapack_int nrhs,
                 const T* dl, const T* d, const T* du, const T* du2,
                 const lapack_int* ipiv, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gttrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -10;
    if (VectorNancheck(n, d, 1)) return -6;
    if (VectorNancheck(n - 1, dl, 1)) return -5;
    if (VectorNancheck(n - 1, du, 1)) return -7;
    if (VectorNancheck(n - 2, du2, 1)) return -8;
  }
  return GttrsWork(matrix_layout, trans, n, nrhs, dl, d, du, du2, ipiv, b,
                   ldb);
}

template <typename T>
lapack_int GtsvWork(int matrix_layout, lapack_int n, lapack_int nrhs, T* dl,
                    T* d, T* du, T* b, lapack_int ldb) {
  return CallWork<T, 1>(matrix_layout, Name<T>("gtsv_work"), false,
                        {{{b, n, nrhs, ldb, -8}}}, [&](auto& p, auto& ld) {
                          return lapack::Gtsv(n, nrhs, dl, d, du, p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Gtsv(int matrix_layout, lapack_int n, lapack_int nrhs, T* dl, T* d,
                T* du, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("gtsv"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -7;
    if (VectorNancheck(n, d, 1)) return -5;
    if (VectorNancheck(n - 1, dl, 1)) return -4;
    if (VectorNancheck(n - 1, du, 1)) return -6;
  }
  return GtsvWork(matrix_layout, n, nrhs, dl, d, du, b, ldb);
}

template <typename T>
lapack_int PttrfWork(lapack_int n, RealT<T>* d, T* e) {
  return lapack::Pttrf(n, d, e);
}

template <typename T>
lapack_int Pttrf(lapack_int n, RealT<T>* d, T* e) {
  if (NanCheck()) {
    if (VectorNancheck(n, d, 1)) return -2;
    if (VectorNancheck(n - 1, e, 1)) return -3;
  }
  return PttrfWork(n, d, e);
}

// xPTTRS: uplo is used for complex T only (the real routine has none, and
// the positions of the arguments after it are one less).
template <typename T>
lapack_int PttrsWork(int matrix_layout, char uplo, lapack_int n,
                     lapack_int nrhs, const RealT<T>* d, const T* e, T* b,
                     lapack_int ldb) {
  constexpr lapack_int kShift = IsComplex<T>::value ? 1 : 0;
  return CallWork<T, 1>(matrix_layout, Name<T>("pttrs_work"), false,
                        {{{b, n, nrhs, ldb, -7 - kShift}}},
                        [&](auto& p, auto& ld) {
                          if constexpr (IsComplex<T>::value) {
                            return lapack::Pttrs(uplo, n, nrhs, d, e, p[0],
                                                 ld[0]);
                          } else {
                            (void)uplo;
                            return lapack::Pttrs(n, nrhs, d, e, p[0], ld[0]);
                          }
                        });
}

template <typename T>
lapack_int Pttrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                 const RealT<T>* d, const T* e, T* b, lapack_int ldb) {
  constexpr lapack_int kShift = IsComplex<T>::value ? 1 : 0;
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("pttrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -6 - kShift;
    if (VectorNancheck(n, d, 1)) return -4 - kShift;
    if (VectorNancheck(n - 1, e, 1)) return -5 - kShift;
  }
  return PttrsWork(matrix_layout, uplo, n, nrhs, d, e, b, ldb);
}

template <typename T>
lapack_int PtsvWork(int matrix_layout, lapack_int n, lapack_int nrhs,
                    RealT<T>* d, T* e, T* b, lapack_int ldb) {
  return CallWork<T, 1>(matrix_layout, Name<T>("ptsv_work"), false,
                        {{{b, n, nrhs, ldb, -7}}}, [&](auto& p, auto& ld) {
                          return lapack::Ptsv(n, nrhs, d, e, p[0], ld[0]);
                        });
}

template <typename T>
lapack_int Ptsv(int matrix_layout, lapack_int n, lapack_int nrhs, RealT<T>* d,
                T* e, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("ptsv"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -6;
    if (VectorNancheck(n, d, 1)) return -4;
    if (VectorNancheck(n - 1, e, 1)) return -5;
  }
  return PtsvWork(matrix_layout, n, nrhs, d, e, b, ldb);
}

}  // namespace
}  // namespace lapacke
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
namespace lapacke = hwyblas::lapacke;

extern "C" {

// The functions whose arguments are the same in all four precisions.
#define HWYBLAS_LAPACKE_BAND(p, T, R)                                          \
  HWYBLAS_API lapack_int LAPACKE_##p##gbtrf(                                   \
      int layout, lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,    \
      T* ab, lapack_int ldab, lapack_int* ipiv) {                              \
    return lapacke::Gbtrf(layout, m, n, kl, ku, ab, ldab, ipiv);               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gbtrf_work(                              \
      int layout, lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,    \
      T* ab, lapack_int ldab, lapack_int* ipiv) {                              \
    return lapacke::GbtrfWork(layout, m, n, kl, ku, ab, ldab, ipiv);           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gbtrs(                                   \
      int layout, char trans, lapack_int n, lapack_int kl, lapack_int ku,      \
      lapack_int nrhs, const T* ab, lapack_int ldab, const lapack_int* ipiv,   \
      T* b, lapack_int ldb) {                                                  \
    return lapacke::Gbtrs(layout, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b,   \
                          ldb);                                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gbtrs_work(                              \
      int layout, char trans, lapack_int n, lapack_int kl, lapack_int ku,      \
      lapack_int nrhs, const T* ab, lapack_int ldab, const lapack_int* ipiv,   \
      T* b, lapack_int ldb) {                                                  \
    return lapacke::GbtrsWork(layout, trans, n, kl, ku, nrhs, ab, ldab, ipiv,  \
                              b, ldb);                                         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gbsv(                                    \
      int layout, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, \
      T* ab, lapack_int ldab, lapack_int* ipiv, T* b, lapack_int ldb) {        \
    return lapacke::Gbsv(layout, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gbsv_work(                               \
      int layout, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, \
      T* ab, lapack_int ldab, lapack_int* ipiv, T* b, lapack_int ldb) {        \
    return lapacke::GbsvWork(layout, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gbcon(                                   \
      int layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,       \
      const T* ab, lapack_int ldab, const lapack_int* ipiv, R anorm,           \
      R* rcond) {                                                              \
    return lapacke::Gbcon(layout, norm, n, kl, ku, ab, ldab, ipiv, anorm,      \
                          rcond);                                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pbtrf(int layout, char uplo,             \
                                            lapack_int n, lapack_int kd,       \
                                            T* ab, lapack_int ldab) {          \
    return lapacke::Pbtrf(layout, uplo, n, kd, ab, ldab);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pbtrf_work(int layout, char uplo,        \
                                                 lapack_int n, lapack_int kd,  \
                                                 T* ab, lapack_int ldab) {     \
    return lapacke::PbtrfWork(layout, uplo, n, kd, ab, ldab);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pbtrs(                                   \
      int layout, char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,     \
      const T* ab, lapack_int ldab, T* b, lapack_int ldb) {                    \
    return lapacke::Pbtrs(layout, uplo, n, kd, nrhs, ab, ldab, b, ldb);        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pbtrs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,     \
      const T* ab, lapack_int ldab, T* b, lapack_int ldb) {                    \
    return lapacke::PbtrsWork(layout, uplo, n, kd, nrhs, ab, ldab, b, ldb);    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pbsv(                                    \
      int layout, char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,     \
      T* ab, lapack_int ldab, T* b, lapack_int ldb) {                          \
    return lapacke::Pbsv(layout, uplo, n, kd, nrhs, ab, ldab, b, ldb);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pbsv_work(                               \
      int layout, char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,     \
      T* ab, lapack_int ldab, T* b, lapack_int ldb) {                          \
    return lapacke::PbsvWork(layout, uplo, n, kd, nrhs, ab, ldab, b, ldb);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gttrf(lapack_int n, T* dl, T* d, T* du,  \
                                            T* du2, lapack_int* ipiv) {        \
    return lapacke::Gttrf(n, dl, d, du, du2, ipiv);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gttrf_work(                              \
      lapack_int n, T* dl, T* d, T* du, T* du2, lapack_int* ipiv) {            \
    return lapacke::GttrfWork(n, dl, d, du, du2, ipiv);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gttrs(                                   \
      int layout, char trans, lapack_int n, lapack_int nrhs, const T* dl,      \
      const T* d, const T* du, const T* du2, const lapack_int* ipiv, T* b,     \
      lapack_int ldb) {                                                        \
    return lapacke::Gttrs(layout, trans, n, nrhs, dl, d, du, du2, ipiv, b,     \
                          ldb);                                                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gttrs_work(                              \
      int layout, char trans, lapack_int n, lapack_int nrhs, const T* dl,      \
      const T* d, const T* du, const T* du2, const lapack_int* ipiv, T* b,     \
      lapack_int ldb) {                                                        \
    return lapacke::GttrsWork(layout, trans, n, nrhs, dl, d, du, du2, ipiv, b, \
                              ldb);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gtsv(int layout, lapack_int n,           \
                                           lapack_int nrhs, T* dl, T* d,       \
                                           T* du, T* b, lapack_int ldb) {      \
    return lapacke::Gtsv(layout, n, nrhs, dl, d, du, b, ldb);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##gtsv_work(int layout, lapack_int n,      \
                                                lapack_int nrhs, T* dl, T* d,  \
                                                T* du, T* b, lapack_int ldb) { \
    return lapacke::GtsvWork(layout, n, nrhs, dl, d, du, b, ldb);              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pttrf(lapack_int n, R* d, T* e) {        \
    return lapacke::Pttrf(n, d, e);                                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pttrf_work(lapack_int n, R* d, T* e) {   \
    return lapacke::PttrfWork(n, d, e);                                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ptsv(int layout, lapack_int n,           \
                                           lapack_int nrhs, R* d, T* e, T* b,  \
                                           lapack_int ldb) {                   \
    return lapacke::Ptsv(layout, n, nrhs, d, e, b, ldb);                       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##ptsv_work(int layout, lapack_int n,      \
                                                lapack_int nrhs, R* d, T* e,   \
                                                T* b, lapack_int ldb) {        \
    return lapacke::PtsvWork(layout, n, nrhs, d, e, b, ldb);                   \
  }

HWYBLAS_LAPACKE_BAND(s, float, float)
HWYBLAS_LAPACKE_BAND(d, double, double)
HWYBLAS_LAPACKE_BAND(c, c32, float)
HWYBLAS_LAPACKE_BAND(z, c64, double)

#undef HWYBLAS_LAPACKE_BAND

// The functions that take IWORK (real) or RWORK (complex), and xPTTRS,
// which takes UPLO for complex T only.
#define HWYBLAS_LAPACKE_BAND_REAL(p, R)                                        \
  HWYBLAS_API lapack_int LAPACKE_##p##gbcon_work(                              \
      int layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,       \
      const R* ab, lapack_int ldab, const lapack_int* ipiv, R anorm, R* rcond, \
      R* work, lapack_int* iwork) {                                            \
    return lapacke::GbconWork(layout, norm, n, kl, ku, ab, ldab, ipiv, anorm,  \
                              rcond, work, static_cast<R*>(nullptr), iwork);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pttrs(int layout, lapack_int n,          \
                                            lapack_int nrhs, const R* d,       \
                                            const R* e, R* b,                  \
                                            lapack_int ldb) {                  \
    return lapacke::Pttrs(layout, 'L', n, nrhs, d, e, b, ldb);                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pttrs_work(int layout, lapack_int n,     \
                                                 lapack_int nrhs, const R* d,  \
                                                 const R* e, R* b,             \
                                                 lapack_int ldb) {             \
    return lapacke::PttrsWork(layout, 'L', n, nrhs, d, e, b, ldb);             \
  }

#define HWYBLAS_LAPACKE_BAND_COMPLEX(p, T, R)                                  \
  HWYBLAS_API lapack_int LAPACKE_##p##gbcon_work(                              \
      int layout, char norm, lapack_int n, lapack_int kl, lapack_int ku,       \
      const T* ab, lapack_int ldab, const lapack_int* ipiv, R anorm, R* rcond, \
      T* work, R* rwork) {                                                     \
    return lapacke::GbconWork(layout, norm, n, kl, ku, ab, ldab, ipiv, anorm,  \
                              rcond, work, rwork,                              \
                              static_cast<lapack_int*>(nullptr));              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pttrs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const R* d,        \
      const T* e, T* b, lapack_int ldb) {                                      \
    return lapacke::Pttrs(layout, uplo, n, nrhs, d, e, b, ldb);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pttrs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const R* d,        \
      const T* e, T* b, lapack_int ldb) {                                      \
    return lapacke::PttrsWork(layout, uplo, n, nrhs, d, e, b, ldb);            \
  }

HWYBLAS_LAPACKE_BAND_REAL(s, float)
HWYBLAS_LAPACKE_BAND_REAL(d, double)
HWYBLAS_LAPACKE_BAND_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_BAND_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_BAND_REAL
#undef HWYBLAS_LAPACKE_BAND_COMPLEX

}  // extern "C"
