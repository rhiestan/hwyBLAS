// LAPACKE functions of the symmetric (Hermitian) linear systems routines:
// positive definite (xPOTRS, xPOSV, xLAUUM, xPOTRI, xPOCON, xPOEQU, xPORFS,
// xPOSVX, xPSTRF) and indefinite (xSYTRF, xSYTRS, xSYTRS2, xSYCONV, xSYTRI,
// xSYCON, xSYRFS, xSYSV, xSYSVX and their xHE* counterparts) (see
// lapacke.cc and lapacke_util.h).
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
// Positive definite matrices

template <typename T>
lapack_int PotrsWork(int matrix_layout, char uplo, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda, T* b,
                     lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("potrs_work"), false,
      {{Triangle(Input(a, n, n, lda, -6), uplo), {b, n, nrhs, ldb, -8}}},
      [&](auto& p, auto& ld) {
        return lapack::Potrs(uplo, n, nrhs, p[0], ld[0], p[1], ld[1]);
      });
}

template <typename T>
lapack_int Potrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                 const T* a, lapack_int lda, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("potrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -7;
  }
  return PotrsWork(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
}

template <typename T>
lapack_int PosvWork(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                    T* a, lapack_int lda, T* b, lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, Name<T>("posv_work"), false,
      {{Triangle<T>({a, n, n, lda, -6}, uplo), {b, n, nrhs, ldb, -8}}},
      [&](auto& p, auto& ld) {
        return lapack::Posv(uplo, n, nrhs, p[0], ld[0], p[1], ld[1]);
      });
}

template <typename T>
lapack_int Posv(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                T* a, lapack_int lda, T* b, lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("posv"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -7;
  }
  return PosvWork(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
}

// The LAPACKE functions of the routines that only take the triangle uplo of
// A (in and out): xLAUUM and xPOTRI.
template <typename T, typename Routine>
lapack_int TriangleWork(int matrix_layout, const char* name, char uplo,
                        lapack_int n, T* a, lapack_int lda,
                        const Routine& routine) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>(name), false,
      {{Triangle<T>({a, n, n, lda, -5}, uplo)}},
      [&](auto& p, auto& ld) { return routine(uplo, n, p[0], ld[0]); });
}

template <typename T, typename Routine>
lapack_int TriangleFunction(int matrix_layout, const char* name, char uplo,
                            lapack_int n, T* a, lapack_int lda,
                            const Routine& routine) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>(name), -1);
    return -1;
  }
  if (NanCheck() && PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
  return routine(matrix_layout, uplo, n, a, lda);
}

template <typename T>
lapack_int LauumWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda) {
  return TriangleWork(matrix_layout, "lauum_work", uplo, n, a, lda,
                      &lapack::Lauum<T>);
}

template <typename T>
lapack_int Lauum(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda) {
  return TriangleFunction(matrix_layout, "lauum", uplo, n, a, lda,
                          &LauumWork<T>);
}

template <typename T>
lapack_int PotriWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda) {
  return TriangleWork(matrix_layout, "potri_work", uplo, n, a, lda,
                      &lapack::Potri<T>);
}

template <typename T>
lapack_int Potri(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda) {
  return TriangleFunction(matrix_layout, "potri", uplo, n, a, lda,
                          &PotriWork<T>);
}

template <typename T>
lapack_int PoconWork(int matrix_layout, char uplo, lapack_int n, const T* a,
                     lapack_int lda, RealT<T> anorm, RealT<T>* rcond, T* work,
                     RealT<T>* rwork, lapack_int* iwork) {
  return CallWork<T, 1>(matrix_layout, Name<T>("pocon_work"), false,
                        {{Triangle(Input(a, n, n, lda, -5), uplo)}},
                        [&](auto& p, auto& ld) {
                          return lapack::Pocon(uplo, n, p[0], ld[0], anorm,
                                               rcond, work, rwork, iwork);
                        });
}

template <typename T>
lapack_int Pocon(int matrix_layout, char uplo, lapack_int n, const T* a,
                 lapack_int lda, RealT<T> anorm, RealT<T>* rcond) {
  const std::string name = Name<T>("pocon");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
    if (VectorNancheck(1, &anorm, 1)) return -6;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return PoconWork(matrix_layout, uplo, n, a, lda, anorm,
                                        rcond, work, rwork, iwork);
                     });
}

template <typename T>
lapack_int PoequWork(int matrix_layout, lapack_int n, const T* a,
                     lapack_int lda, RealT<T>* s, RealT<T>* scond,
                     RealT<T>* amax) {
  return CallWork<T, 1>(matrix_layout, Name<T>("poequ_work"), false,
                        {{Input(a, n, n, lda, -4)}}, [&](auto& p, auto& ld) {
                          return lapack::Poequ(n, p[0], ld[0], s, scond, amax);
                        });
}

template <typename T>
lapack_int Poequ(int matrix_layout, lapack_int n, const T* a, lapack_int lda,
                 RealT<T>* s, RealT<T>* scond, RealT<T>* amax) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("poequ"), -1);
    return -1;
  }
  if (NanCheck() && GeNancheck(matrix_layout, n, n, a, lda)) return -3;
  return PoequWork(matrix_layout, n, a, lda, s, scond, amax);
}

template <typename T>
lapack_int PorfsWork(int matrix_layout, char uplo, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda, const T* af,
                     lapack_int ldaf, const T* b, lapack_int ldb, T* x,
                     lapack_int ldx, RealT<T>* ferr, RealT<T>* berr, T* work,
                     RealT<T>* rwork, lapack_int* iwork) {
  return CallWork<T, 4>(matrix_layout, Name<T>("porfs_work"), false,
                        {{Triangle(Input(a, n, n, lda, -6), uplo),
                          Triangle(Input(af, n, n, ldaf, -8), uplo),
                          Input(b, n, nrhs, ldb, -10),
                          {x, n, nrhs, ldx, -12}}},
                        [&](auto& p, auto& ld) {
                          return lapack::Porfs(uplo, n, nrhs, p[0], ld[0], p[1],
                                               ld[1], p[2], ld[2], p[3], ld[3],
                                               ferr, berr, work, rwork, iwork);
                        });
}

template <typename T>
lapack_int Porfs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                 const T* a, lapack_int lda, const T* af, lapack_int ldaf,
                 const T* b, lapack_int ldb, T* x, lapack_int ldx,
                 RealT<T>* ferr, RealT<T>* berr) {
  const std::string name = Name<T>("porfs");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (PoNancheck(matrix_layout, uplo, n, af, ldaf)) return -7;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -9;
    if (GeNancheck(matrix_layout, n, nrhs, x, ldx)) return -11;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return PorfsWork(matrix_layout, uplo, n, nrhs, a, lda,
                                        af, ldaf, b, ldb, x, ldx, ferr, berr,
                                        work, rwork, iwork);
                     });
}

// xPOSVX. In row-major layout, the reference copies A back for FACT = 'E'
// (if EQUED = 'Y'), AF for FACT = 'E' or 'N', and B and X always. (Copying
// back the A that the routine did not change leaves it as it was.)
template <typename T>
lapack_int PosvxWork(int matrix_layout, char fact, char uplo, lapack_int n,
                     lapack_int nrhs, T* a, lapack_int lda, T* af,
                     lapack_int ldaf, char* equed, RealT<T>* s, T* b,
                     lapack_int ldb, T* x, lapack_int ldx, RealT<T>* rcond,
                     RealT<T>* ferr, RealT<T>* berr, T* work, RealT<T>* rwork,
                     lapack_int* iwork) {
  const bool given = lapack::Lsame(fact, 'F');
  MatrixArg<T> a_arg = Triangle<T>({a, n, n, lda, -7}, uplo);
  a_arg.out = lapack::Lsame(fact, 'E');
  MatrixArg<T> af_arg = Triangle<T>({af, n, n, ldaf, -9}, uplo);
  af_arg.in = given;
  af_arg.out = !given;
  MatrixArg<T> x_arg{x, n, nrhs, ldx, -15};
  x_arg.in = false;
  return CallWork<T, 4>(
      matrix_layout, Name<T>("posvx_work"), false,
      {{a_arg, af_arg, {b, n, nrhs, ldb, -13}, x_arg}}, [&](auto& p, auto& ld) {
        return lapack::Posvx(fact, uplo, n, nrhs, p[0], ld[0], p[1], ld[1],
                             equed, s, p[2], ld[2], p[3], ld[3], rcond, ferr,
                             berr, work, rwork, iwork);
      });
}

template <typename T>
lapack_int Posvx(int matrix_layout, char fact, char uplo, lapack_int n,
                 lapack_int nrhs, T* a, lapack_int lda, T* af, lapack_int ldaf,
                 char* equed, RealT<T>* s, T* b, lapack_int ldb, T* x,
                 lapack_int ldx, RealT<T>* rcond, RealT<T>* ferr,
                 RealT<T>* berr) {
  const std::string name = Name<T>("posvx");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    const bool given = lapack::Lsame(fact, 'F');
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -6;
    if (given && PoNancheck(matrix_layout, uplo, n, af, ldaf)) return -8;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -12;
    if (given && lapack::Lsame(*equed, 'Y') && VectorNancheck(n, s, 1)) {
      return -11;
    }
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return PosvxWork(matrix_layout, fact, uplo, n, nrhs, a,
                                        lda, af, ldaf, equed, s, b, ldb, x, ldx,
                                        rcond, ferr, berr, work, rwork, iwork);
                     });
}

template <typename T>
lapack_int PstrfWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda, lapack_int* piv, lapack_int* rank,
                     RealT<T> tol, RealT<T>* work) {
  return CallWork<T, 1>(
      matrix_layout, Name<T>("pstrf_work"), false,
      {{Triangle<T>({a, n, n, lda, -5}, uplo)}}, [&](auto& p, auto& ld) {
        return lapack::Pstrf(uplo, n, p[0], ld[0], piv, rank, tol, work);
      });
}

template <typename T>
lapack_int Pstrf(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda, lapack_int* piv, lapack_int* rank,
                 RealT<T> tol) {
  using R = RealT<T>;
  const std::string name = Name<T>("pstrf");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
    if (VectorNancheck(1, &tol, 1)) return -8;
  }
  return WithWorkArray<R>(name, std::max<lapack_int>(1, 2 * n), [&](R* work) {
    return PstrfWork(matrix_layout, uplo, n, a, lda, piv, rank, tol, work);
  });
}

// ---------------------------------------------------------------------------
// Indefinite matrices. The functions of xSY* and xHE* routines are the same
// but for the routine; kHerm selects xHE*.

template <bool kHerm, typename T>
std::string SyName(const char* sy, const char* he) {
  return Name<T>(kHerm ? he : sy);
}

template <bool kHerm, typename T>
lapack_int SytrfWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda, lapack_int* ipiv, T* work,
                     lapack_int lwork) {
  return CallWork<T, 1>(
      matrix_layout, SyName<kHerm, T>("sytrf_work", "hetrf_work"), lwork == -1,
      {{Triangle<T>({a, n, n, lda, -5}, uplo)}}, [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hetrf(uplo, n, p[0], ld[0], ipiv, work, lwork);
        } else {
          return lapack::Sytrf(uplo, n, p[0], ld[0], ipiv, work, lwork);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Sytrf(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda, lapack_int* ipiv) {
  const std::string name = SyName<kHerm, T>("sytrf", "hetrf");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return SytrfWork<kHerm>(matrix_layout, uplo, n, a, lda, ipiv, work, lwork);
  });
}

template <bool kHerm, typename T>
lapack_int SytrsWork(int matrix_layout, char uplo, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda,
                     const lapack_int* ipiv, T* b, lapack_int ldb) {
  return CallWork<T, 2>(
      matrix_layout, SyName<kHerm, T>("sytrs_work", "hetrs_work"), false,
      {{Triangle(Input(a, n, n, lda, -6), uplo), {b, n, nrhs, ldb, -9}}},
      [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hetrs(uplo, n, nrhs, p[0], ld[0], ipiv, p[1], ld[1]);
        } else {
          return lapack::Sytrs(uplo, n, nrhs, p[0], ld[0], ipiv, p[1], ld[1]);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Sytrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                 const T* a, lapack_int lda, const lapack_int* ipiv, T* b,
                 lapack_int ldb) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(SyName<kHerm, T>("sytrs", "hetrs"), -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -8;
  }
  return SytrsWork<kHerm>(matrix_layout, uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

// xSYTRS2 / xHETRS2 change A, but restore it; in row-major layout, the
// transposed A is not copied back (as in the reference).
template <bool kHerm, typename T>
lapack_int Sytrs2Work(int matrix_layout, char uplo, lapack_int n,
                      lapack_int nrhs, const T* a, lapack_int lda,
                      const lapack_int* ipiv, T* b, lapack_int ldb, T* work) {
  return CallWork<T, 2>(
      matrix_layout, SyName<kHerm, T>("sytrs2_work", "hetrs2_work"), false,
      {{Triangle(Input(a, n, n, lda, -6), uplo), {b, n, nrhs, ldb, -9}}},
      [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hetrs2(uplo, n, nrhs, p[0], ld[0], ipiv, p[1], ld[1],
                                work);
        } else {
          return lapack::Sytrs2(uplo, n, nrhs, p[0], ld[0], ipiv, p[1], ld[1],
                                work);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Sytrs2(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                  const T* a, lapack_int lda, const lapack_int* ipiv, T* b,
                  lapack_int ldb) {
  const std::string name = SyName<kHerm, T>("sytrs2", "hetrs2");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -8;
  }
  return WithWorkArray<T>(name, std::max<lapack_int>(1, n), [&](T* work) {
    return Sytrs2Work<kHerm>(matrix_layout, uplo, n, nrhs, a, lda, ipiv, b, ldb,
                             work);
  });
}

// xSYCONV. Deviation (bug fix): in row-major layout, the n x n matrix A is
// transposed. The reference transposes lda rows of it, reading and writing
// past the n rows of the array if lda > n.
template <typename T>
lapack_int SyconvWork(int matrix_layout, char uplo, char way, lapack_int n,
                      T* a, lapack_int lda, const lapack_int* ipiv, T* e) {
  return CallWork<T, 1>(matrix_layout, Name<T>("syconv_work"), false,
                        {{{a, n, n, lda, -6}}}, [&](auto& p, auto& ld) {
                          return lapack::Syconv(uplo, way, n, p[0], ld[0], ipiv,
                                                e);
                        });
}

template <typename T>
lapack_int Syconv(int matrix_layout, char uplo, char way, lapack_int n, T* a,
                  lapack_int lda, const lapack_int* ipiv, T* e) {
  if (!ValidLayout(matrix_layout)) {
    Xerbla(Name<T>("syconv"), -1);
    return -1;
  }
  if (NanCheck() && PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
  return SyconvWork(matrix_layout, uplo, way, n, a, lda, ipiv, e);
}

template <bool kHerm, typename T>
lapack_int SytriWork(int matrix_layout, char uplo, lapack_int n, T* a,
                     lapack_int lda, const lapack_int* ipiv, T* work) {
  return CallWork<T, 1>(
      matrix_layout, SyName<kHerm, T>("sytri_work", "hetri_work"), false,
      {{Triangle<T>({a, n, n, lda, -5}, uplo)}}, [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hetri(uplo, n, p[0], ld[0], ipiv, work);
        } else {
          return lapack::Sytri(uplo, n, p[0], ld[0], ipiv, work);
        }
      });
}

// The reference allocates 2*N elements of WORK for xSYTRI (N suffice), N
// for xHETRI.
template <bool kHerm, typename T>
lapack_int Sytri(int matrix_layout, char uplo, lapack_int n, T* a,
                 lapack_int lda, const lapack_int* ipiv) {
  const std::string name = SyName<kHerm, T>("sytri", "hetri");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck() && PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
  const lapack_int size = kHerm ? n : 2 * n;
  return WithWorkArray<T>(name, std::max<lapack_int>(1, size), [&](T* work) {
    return SytriWork<kHerm>(matrix_layout, uplo, n, a, lda, ipiv, work);
  });
}

template <bool kHerm, typename T>
lapack_int SyconWork(int matrix_layout, char uplo, lapack_int n, const T* a,
                     lapack_int lda, const lapack_int* ipiv, RealT<T> anorm,
                     RealT<T>* rcond, T* work, lapack_int* iwork) {
  return CallWork<T, 1>(
      matrix_layout, SyName<kHerm, T>("sycon_work", "hecon_work"), false,
      {{Triangle(Input(a, n, n, lda, -5), uplo)}}, [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hecon(uplo, n, p[0], ld[0], ipiv, anorm, rcond, work);
        } else {
          return lapack::Sycon(uplo, n, p[0], ld[0], ipiv, anorm, rcond, work,
                               iwork);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Sycon(int matrix_layout, char uplo, lapack_int n, const T* a,
                 lapack_int lda, const lapack_int* ipiv, RealT<T> anorm,
                 RealT<T>* rcond) {
  const std::string name = SyName<kHerm, T>("sycon", "hecon");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -4;
    if (VectorNancheck(1, &anorm, 1)) return -7;
  }
  // WORK has 2*N elements, also for real T; IWORK N for real T.
  return WithWork<T>(name, 2 * n, n, 2 * n, 0,
                     [&](T* work, RealT<T>*, lapack_int* iwork) {
                       return SyconWork<kHerm>(matrix_layout, uplo, n, a, lda,
                                               ipiv, anorm, rcond, work, iwork);
                     });
}

template <bool kHerm, typename T>
lapack_int SyrfsWork(int matrix_layout, char uplo, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda, const T* af,
                     lapack_int ldaf, const lapack_int* ipiv, const T* b,
                     lapack_int ldb, T* x, lapack_int ldx, RealT<T>* ferr,
                     RealT<T>* berr, T* work, RealT<T>* rwork,
                     lapack_int* iwork) {
  return CallWork<T, 4>(
      matrix_layout, SyName<kHerm, T>("syrfs_work", "herfs_work"), false,
      {{Triangle(Input(a, n, n, lda, -6), uplo),
        Triangle(Input(af, n, n, ldaf, -8), uplo),
        Input(b, n, nrhs, ldb, -11),
        {x, n, nrhs, ldx, -13}}},
      [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Herfs(uplo, n, nrhs, p[0], ld[0], p[1], ld[1], ipiv,
                               p[2], ld[2], p[3], ld[3], ferr, berr, work,
                               rwork);
        } else {
          return lapack::Syrfs(uplo, n, nrhs, p[0], ld[0], p[1], ld[1], ipiv,
                               p[2], ld[2], p[3], ld[3], ferr, berr, work,
                               rwork, iwork);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Syrfs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                 const T* a, lapack_int lda, const T* af, lapack_int ldaf,
                 const lapack_int* ipiv, const T* b, lapack_int ldb, T* x,
                 lapack_int ldx, RealT<T>* ferr, RealT<T>* berr) {
  const std::string name = SyName<kHerm, T>("syrfs", "herfs");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (PoNancheck(matrix_layout, uplo, n, af, ldaf)) return -7;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -10;
    if (GeNancheck(matrix_layout, n, nrhs, x, ldx)) return -12;
  }
  return WithWork<T>(name, 3 * n, n, 2 * n, n,
                     [&](T* work, RealT<T>* rwork, lapack_int* iwork) {
                       return SyrfsWork<kHerm>(
                           matrix_layout, uplo, n, nrhs, a, lda, af, ldaf, ipiv,
                           b, ldb, x, ldx, ferr, berr, work, rwork, iwork);
                     });
}

template <bool kHerm, typename T>
lapack_int SysvWork(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                    T* a, lapack_int lda, lapack_int* ipiv, T* b,
                    lapack_int ldb, T* work, lapack_int lwork) {
  return CallWork<T, 2>(
      matrix_layout, SyName<kHerm, T>("sysv_work", "hesv_work"), lwork == -1,
      {{Triangle<T>({a, n, n, lda, -6}, uplo), {b, n, nrhs, ldb, -9}}},
      [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hesv(uplo, n, nrhs, p[0], ld[0], ipiv, p[1], ld[1],
                              work, lwork);
        } else {
          return lapack::Sysv(uplo, n, nrhs, p[0], ld[0], ipiv, p[1], ld[1],
                              work, lwork);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Sysv(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                T* a, lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb) {
  const std::string name = SyName<kHerm, T>("sysv", "hesv");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -5;
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -8;
  }
  return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
    return SysvWork<kHerm>(matrix_layout, uplo, n, nrhs, a, lda, ipiv, b, ldb,
                           work, lwork);
  });
}

// xSYSVX / xHESVX. In row-major layout, AF is copied back for FACT = 'N'.
template <bool kHerm, typename T>
lapack_int SysvxWork(int matrix_layout, char fact, char uplo, lapack_int n,
                     lapack_int nrhs, const T* a, lapack_int lda, T* af,
                     lapack_int ldaf, lapack_int* ipiv, const T* b,
                     lapack_int ldb, T* x, lapack_int ldx, RealT<T>* rcond,
                     RealT<T>* ferr, RealT<T>* berr, T* work, lapack_int lwork,
                     RealT<T>* rwork, lapack_int* iwork) {
  const bool given = lapack::Lsame(fact, 'F');
  MatrixArg<T> af_arg = Triangle<T>({af, n, n, ldaf, -9}, uplo);
  af_arg.in = given;
  af_arg.out = !given;
  MatrixArg<T> x_arg{x, n, nrhs, ldx, -14};
  x_arg.in = false;
  return CallWork<T, 4>(
      matrix_layout, SyName<kHerm, T>("sysvx_work", "hesvx_work"), lwork == -1,
      {{Triangle(Input(a, n, n, lda, -7), uplo), af_arg,
        Input(b, n, nrhs, ldb, -12), x_arg}},
      [&](auto& p, auto& ld) {
        if constexpr (kHerm) {
          return lapack::Hesvx(fact, uplo, n, nrhs, p[0], ld[0], p[1], ld[1],
                               ipiv, p[2], ld[2], p[3], ld[3], rcond, ferr,
                               berr, work, lwork, rwork);
        } else {
          return lapack::Sysvx(fact, uplo, n, nrhs, p[0], ld[0], p[1], ld[1],
                               ipiv, p[2], ld[2], p[3], ld[3], rcond, ferr,
                               berr, work, lwork, rwork, iwork);
        }
      });
}

template <bool kHerm, typename T>
lapack_int Sysvx(int matrix_layout, char fact, char uplo, lapack_int n,
                 lapack_int nrhs, const T* a, lapack_int lda, T* af,
                 lapack_int ldaf, lapack_int* ipiv, const T* b, lapack_int ldb,
                 T* x, lapack_int ldx, RealT<T>* rcond, RealT<T>* ferr,
                 RealT<T>* berr) {
  using R = RealT<T>;
  const std::string name = SyName<kHerm, T>("sysvx", "hesvx");
  if (!ValidLayout(matrix_layout)) {
    Xerbla(name, -1);
    return -1;
  }
  if (NanCheck()) {
    if (PoNancheck(matrix_layout, uplo, n, a, lda)) return -6;
    if (lapack::Lsame(fact, 'F') &&
        PoNancheck(matrix_layout, uplo, n, af, ldaf)) {
      return -8;
    }
    if (GeNancheck(matrix_layout, n, nrhs, b, ldb)) return -11;
  }
  // IWORK (real T) or RWORK (complex T), then WORK of the queried size.
  return WithWork<T>(name, 0, n, 0, n, [&](T*, R* rwork, lapack_int* iwork) {
    return WithWorkspace<T>(name, [&](T* work, lapack_int lwork) {
      return SysvxWork<kHerm>(matrix_layout, fact, uplo, n, nrhs, a, lda, af,
                              ldaf, ipiv, b, ldb, x, ldx, rcond, ferr, berr,
                              work, lwork, rwork, iwork);
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
#define HWYBLAS_LAPACKE_SYMMETRIC(p, T, R)                                     \
  HWYBLAS_API lapack_int LAPACKE_##p##potrs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::Potrs(layout, uplo, n, nrhs, a, lda, b, ldb);              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potrs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::PotrsWork(layout, uplo, n, nrhs, a, lda, b, ldb);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##posv(                                    \
      int layout, char uplo, lapack_int n, lapack_int nrhs, T* a,              \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::Posv(layout, uplo, n, nrhs, a, lda, b, ldb);               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##posv_work(                               \
      int layout, char uplo, lapack_int n, lapack_int nrhs, T* a,              \
      lapack_int lda, T* b, lapack_int ldb) {                                  \
    return lapacke::PosvWork(layout, uplo, n, nrhs, a, lda, b, ldb);           \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lauum(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::Lauum(layout, uplo, n, a, lda);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##lauum_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::LauumWork(layout, uplo, n, a, lda);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potri(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::Potri(layout, uplo, n, a, lda);                            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##potri_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda) {             \
    return lapacke::PotriWork(layout, uplo, n, a, lda);                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pocon(                                   \
      int layout, char uplo, lapack_int n, const T* a, lapack_int lda,         \
      R anorm, R* rcond) {                                                     \
    return lapacke::Pocon(layout, uplo, n, a, lda, anorm, rcond);              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##poequ(int layout, lapack_int n,          \
                                            const T* a, lapack_int lda, R* s,  \
                                            R* scond, R* amax) {               \
    return lapacke::Poequ(layout, n, a, lda, s, scond, amax);                  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##poequ_work(int layout, lapack_int n,     \
                                                 const T* a, lapack_int lda,   \
                                                 R* s, R* scond, R* amax) {    \
    return lapacke::PoequWork(layout, n, a, lda, s, scond, amax);              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##porfs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const T* af, lapack_int ldaf, const T* b,                \
      lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr) {                \
    return lapacke::Porfs(layout, uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x,  \
                          ldx, ferr, berr);                                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##posvx(                                   \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs, T* a,   \
      lapack_int lda, T* af, lapack_int ldaf, char* equed, R* s, T* b,         \
      lapack_int ldb, T* x, lapack_int ldx, R* rcond, R* ferr, R* berr) {      \
    return lapacke::Posvx(layout, fact, uplo, n, nrhs, a, lda, af, ldaf,       \
                          equed, s, b, ldb, x, ldx, rcond, ferr, berr);        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pstrf(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      lapack_int* piv, lapack_int* rank, R tol) {                              \
    return lapacke::Pstrf(layout, uplo, n, a, lda, piv, rank, tol);            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##pstrf_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      lapack_int* piv, lapack_int* rank, R tol, R* work) {                     \
    return lapacke::PstrfWork(layout, uplo, n, a, lda, piv, rank, tol, work);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytrf(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      lapack_int* ipiv) {                                                      \
    return lapacke::Sytrf<false>(layout, uplo, n, a, lda, ipiv);               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytrf_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      lapack_int* ipiv, T* work, lapack_int lwork) {                           \
    return lapacke::SytrfWork<false>(layout, uplo, n, a, lda, ipiv, work,      \
                                     lwork);                                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytrs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::Sytrs<false>(layout, uplo, n, nrhs, a, lda, ipiv, b, ldb); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytrs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::SytrsWork<false>(layout, uplo, n, nrhs, a, lda, ipiv, b,   \
                                     ldb);                                     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytrs2(                                  \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::Sytrs2<false>(layout, uplo, n, nrhs, a, lda, ipiv, b,      \
                                  ldb);                                        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytrs2_work(                             \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb, T* work) { \
    return lapacke::Sytrs2Work<false>(layout, uplo, n, nrhs, a, lda, ipiv, b,  \
                                      ldb, work);                              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syconv(                                  \
      int layout, char uplo, char way, lapack_int n, T* a, lapack_int lda,     \
      const lapack_int* ipiv, T* e) {                                          \
    return lapacke::Syconv(layout, uplo, way, n, a, lda, ipiv, e);             \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syconv_work(                             \
      int layout, char uplo, char way, lapack_int n, T* a, lapack_int lda,     \
      const lapack_int* ipiv, T* e) {                                          \
    return lapacke::SyconvWork(layout, uplo, way, n, a, lda, ipiv, e);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytri(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      const lapack_int* ipiv) {                                                \
    return lapacke::Sytri<false>(layout, uplo, n, a, lda, ipiv);               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sytri_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      const lapack_int* ipiv, T* work) {                                       \
    return lapacke::SytriWork<false>(layout, uplo, n, a, lda, ipiv, work);     \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sycon(                                   \
      int layout, char uplo, lapack_int n, const T* a, lapack_int lda,         \
      const lapack_int* ipiv, R anorm, R* rcond) {                             \
    return lapacke::Sycon<false>(layout, uplo, n, a, lda, ipiv, anorm, rcond); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syrfs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const T* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr) {    \
    return lapacke::Syrfs<false>(layout, uplo, n, nrhs, a, lda, af, ldaf,      \
                                 ipiv, b, ldb, x, ldx, ferr, berr);            \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sysv(                                    \
      int layout, char uplo, lapack_int n, lapack_int nrhs, T* a,              \
      lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb) {                \
    return lapacke::Sysv<false>(layout, uplo, n, nrhs, a, lda, ipiv, b, ldb);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sysv_work(                               \
      int layout, char uplo, lapack_int n, lapack_int nrhs, T* a,              \
      lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb, T* work,         \
      lapack_int lwork) {                                                      \
    return lapacke::SysvWork<false>(layout, uplo, n, nrhs, a, lda, ipiv, b,    \
                                    ldb, work, lwork);                         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sysvx(                                   \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs,         \
      const T* a, lapack_int lda, T* af, lapack_int ldaf, lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* rcond, R* ferr,     \
      R* berr) {                                                               \
    return lapacke::Sysvx<false>(layout, fact, uplo, n, nrhs, a, lda, af,      \
                                 ldaf, ipiv, b, ldb, x, ldx, rcond, ferr,      \
                                 berr);                                        \
  }

HWYBLAS_LAPACKE_SYMMETRIC(s, float, float)
HWYBLAS_LAPACKE_SYMMETRIC(d, double, double)
HWYBLAS_LAPACKE_SYMMETRIC(c, c32, float)
HWYBLAS_LAPACKE_SYMMETRIC(z, c64, double)

#undef HWYBLAS_LAPACKE_SYMMETRIC

// The _work functions that take IWORK (real) or RWORK (complex).
#define HWYBLAS_LAPACKE_SYMMETRIC_REAL(p, R)                                   \
  HWYBLAS_API lapack_int LAPACKE_##p##pocon_work(                              \
      int layout, char uplo, lapack_int n, const R* a, lapack_int lda,         \
      R anorm, R* rcond, R* work, lapack_int* iwork) {                         \
    return lapacke::PoconWork(layout, uplo, n, a, lda, anorm, rcond, work,     \
                              static_cast<R*>(nullptr), iwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##porfs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const R* a,        \
      lapack_int lda, const R* af, lapack_int ldaf, const R* b,                \
      lapack_int ldb, R* x, lapack_int ldx, R* ferr, R* berr, R* work,         \
      lapack_int* iwork) {                                                     \
    return lapacke::PorfsWork(layout, uplo, n, nrhs, a, lda, af, ldaf, b, ldb, \
                              x, ldx, ferr, berr, work,                        \
                              static_cast<R*>(nullptr), iwork);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##posvx_work(                              \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs, R* a,   \
      lapack_int lda, R* af, lapack_int ldaf, char* equed, R* s, R* b,         \
      lapack_int ldb, R* x, lapack_int ldx, R* rcond, R* ferr, R* berr,        \
      R* work, lapack_int* iwork) {                                            \
    return lapacke::PosvxWork(layout, fact, uplo, n, nrhs, a, lda, af, ldaf,   \
                              equed, s, b, ldb, x, ldx, rcond, ferr, berr,     \
                              work, static_cast<R*>(nullptr), iwork);          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sycon_work(                              \
      int layout, char uplo, lapack_int n, const R* a, lapack_int lda,         \
      const lapack_int* ipiv, R anorm, R* rcond, R* work, lapack_int* iwork) { \
    return lapacke::SyconWork<false>(layout, uplo, n, a, lda, ipiv, anorm,     \
                                     rcond, work, iwork);                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syrfs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const R* a,        \
      lapack_int lda, const R* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const R* b, lapack_int ldb, R* x, lapack_int ldx, R* ferr, R* berr,      \
      R* work, lapack_int* iwork) {                                            \
    return lapacke::SyrfsWork<false>(layout, uplo, n, nrhs, a, lda, af, ldaf,  \
                                     ipiv, b, ldb, x, ldx, ferr, berr, work,   \
                                     static_cast<R*>(nullptr), iwork);         \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sysvx_work(                              \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs,         \
      const R* a, lapack_int lda, R* af, lapack_int ldaf, lapack_int* ipiv,    \
      const R* b, lapack_int ldb, R* x, lapack_int ldx, R* rcond, R* ferr,     \
      R* berr, R* work, lapack_int lwork, lapack_int* iwork) {                 \
    return lapacke::SysvxWork<false>(                                          \
        layout, fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx,   \
        rcond, ferr, berr, work, lwork, static_cast<R*>(nullptr), iwork);      \
  }

#define HWYBLAS_LAPACKE_SYMMETRIC_COMPLEX(p, T, R)                             \
  HWYBLAS_API lapack_int LAPACKE_##p##pocon_work(                              \
      int layout, char uplo, lapack_int n, const T* a, lapack_int lda,         \
      R anorm, R* rcond, T* work, R* rwork) {                                  \
    return lapacke::PoconWork(layout, uplo, n, a, lda, anorm, rcond, work,     \
                              rwork, static_cast<lapack_int*>(nullptr));       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##porfs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const T* af, lapack_int ldaf, const T* b,                \
      lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr, T* work,         \
      R* rwork) {                                                              \
    return lapacke::PorfsWork(layout, uplo, n, nrhs, a, lda, af, ldaf, b, ldb, \
                              x, ldx, ferr, berr, work, rwork,                 \
                              static_cast<lapack_int*>(nullptr));              \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##posvx_work(                              \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs, T* a,   \
      lapack_int lda, T* af, lapack_int ldaf, char* equed, R* s, T* b,         \
      lapack_int ldb, T* x, lapack_int ldx, R* rcond, R* ferr, R* berr,        \
      T* work, R* rwork) {                                                     \
    return lapacke::PosvxWork(layout, fact, uplo, n, nrhs, a, lda, af, ldaf,   \
                              equed, s, b, ldb, x, ldx, rcond, ferr, berr,     \
                              work, rwork, static_cast<lapack_int*>(nullptr)); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sycon_work(                              \
      int layout, char uplo, lapack_int n, const T* a, lapack_int lda,         \
      const lapack_int* ipiv, R anorm, R* rcond, T* work) {                    \
    return lapacke::SyconWork<false>(layout, uplo, n, a, lda, ipiv, anorm,     \
                                     rcond, work,                              \
                                     static_cast<lapack_int*>(nullptr));       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##syrfs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const T* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr,      \
      T* work, R* rwork) {                                                     \
    return lapacke::SyrfsWork<false>(                                          \
        layout, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr,   \
        berr, work, rwork, static_cast<lapack_int*>(nullptr));                 \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##sysvx_work(                              \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs,         \
      const T* a, lapack_int lda, T* af, lapack_int ldaf, lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* rcond, R* ferr,     \
      R* berr, T* work, lapack_int lwork, R* rwork) {                          \
    return lapacke::SysvxWork<false>(layout, fact, uplo, n, nrhs, a, lda, af,  \
                                     ldaf, ipiv, b, ldb, x, ldx, rcond, ferr,  \
                                     berr, work, lwork, rwork,                 \
                                     static_cast<lapack_int*>(nullptr));       \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetrf(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      lapack_int* ipiv) {                                                      \
    return lapacke::Sytrf<true>(layout, uplo, n, a, lda, ipiv);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetrf_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      lapack_int* ipiv, T* work, lapack_int lwork) {                           \
    return lapacke::SytrfWork<true>(layout, uplo, n, a, lda, ipiv, work,       \
                                    lwork);                                    \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetrs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::Sytrs<true>(layout, uplo, n, nrhs, a, lda, ipiv, b, ldb);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetrs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::SytrsWork<true>(layout, uplo, n, nrhs, a, lda, ipiv, b,    \
                                    ldb);                                      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetrs2(                                  \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb) {          \
    return lapacke::Sytrs2<true>(layout, uplo, n, nrhs, a, lda, ipiv, b, ldb); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetrs2_work(                             \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb, T* work) { \
    return lapacke::Sytrs2Work<true>(layout, uplo, n, nrhs, a, lda, ipiv, b,   \
                                     ldb, work);                               \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetri(                                   \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      const lapack_int* ipiv) {                                                \
    return lapacke::Sytri<true>(layout, uplo, n, a, lda, ipiv);                \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hetri_work(                              \
      int layout, char uplo, lapack_int n, T* a, lapack_int lda,               \
      const lapack_int* ipiv, T* work) {                                       \
    return lapacke::SytriWork<true>(layout, uplo, n, a, lda, ipiv, work);      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hecon(                                   \
      int layout, char uplo, lapack_int n, const T* a, lapack_int lda,         \
      const lapack_int* ipiv, R anorm, R* rcond) {                             \
    return lapacke::Sycon<true>(layout, uplo, n, a, lda, ipiv, anorm, rcond);  \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hecon_work(                              \
      int layout, char uplo, lapack_int n, const T* a, lapack_int lda,         \
      const lapack_int* ipiv, R anorm, R* rcond, T* work) {                    \
    return lapacke::SyconWork<true>(layout, uplo, n, a, lda, ipiv, anorm,      \
                                    rcond, work,                               \
                                    static_cast<lapack_int*>(nullptr));        \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##herfs(                                   \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const T* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr) {    \
    return lapacke::Syrfs<true>(layout, uplo, n, nrhs, a, lda, af, ldaf, ipiv, \
                                b, ldb, x, ldx, ferr, berr);                   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##herfs_work(                              \
      int layout, char uplo, lapack_int n, lapack_int nrhs, const T* a,        \
      lapack_int lda, const T* af, lapack_int ldaf, const lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* ferr, R* berr,      \
      T* work, R* rwork) {                                                     \
    return lapacke::SyrfsWork<true>(layout, uplo, n, nrhs, a, lda, af, ldaf,   \
                                    ipiv, b, ldb, x, ldx, ferr, berr, work,    \
                                    rwork, static_cast<lapack_int*>(nullptr)); \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hesv(                                    \
      int layout, char uplo, lapack_int n, lapack_int nrhs, T* a,              \
      lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb) {                \
    return lapacke::Sysv<true>(layout, uplo, n, nrhs, a, lda, ipiv, b, ldb);   \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hesv_work(                               \
      int layout, char uplo, lapack_int n, lapack_int nrhs, T* a,              \
      lapack_int lda, lapack_int* ipiv, T* b, lapack_int ldb, T* work,         \
      lapack_int lwork) {                                                      \
    return lapacke::SysvWork<true>(layout, uplo, n, nrhs, a, lda, ipiv, b,     \
                                   ldb, work, lwork);                          \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hesvx(                                   \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs,         \
      const T* a, lapack_int lda, T* af, lapack_int ldaf, lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* rcond, R* ferr,     \
      R* berr) {                                                               \
    return lapacke::Sysvx<true>(layout, fact, uplo, n, nrhs, a, lda, af, ldaf, \
                                ipiv, b, ldb, x, ldx, rcond, ferr, berr);      \
  }                                                                            \
  HWYBLAS_API lapack_int LAPACKE_##p##hesvx_work(                              \
      int layout, char fact, char uplo, lapack_int n, lapack_int nrhs,         \
      const T* a, lapack_int lda, T* af, lapack_int ldaf, lapack_int* ipiv,    \
      const T* b, lapack_int ldb, T* x, lapack_int ldx, R* rcond, R* ferr,     \
      R* berr, T* work, lapack_int lwork, R* rwork) {                          \
    return lapacke::SysvxWork<true>(layout, fact, uplo, n, nrhs, a, lda, af,   \
                                    ldaf, ipiv, b, ldb, x, ldx, rcond, ferr,   \
                                    berr, work, lwork, rwork,                  \
                                    static_cast<lapack_int*>(nullptr));        \
  }

HWYBLAS_LAPACKE_SYMMETRIC_REAL(s, float)
HWYBLAS_LAPACKE_SYMMETRIC_REAL(d, double)
HWYBLAS_LAPACKE_SYMMETRIC_COMPLEX(c, c32, float)
HWYBLAS_LAPACKE_SYMMETRIC_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACKE_SYMMETRIC_REAL
#undef HWYBLAS_LAPACKE_SYMMETRIC_COMPLEX

}  // extern "C"
