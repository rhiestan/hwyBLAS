// The generalized symmetric-definite (Hermitian-definite) eigenproblem:
// the reduction to standard form (xSYGST / xHEGST blocked, xSYGS2 /
// xHEGS2 unblocked) and the drivers xSYGV / xHEGV (with xSYEV / xHEEV) and
// xSYGVD / xHEGVD (with xSYEVD / xHEEVD).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xSYGS2 / xHEGS2: reduces the generalized problem with the symmetric
// (Hermitian) N-by-N matrix A (triangle UPLO) and the Cholesky factor of B
// from xPOTRF (same UPLO) to standard form: ITYPE = 1: A := inv(U**H) * A
// * inv(U) or inv(L) * A * inv(L**H); ITYPE = 2 or 3: A := U * A * U**H or
// L**H * A * L. Unblocked. B is restored on exit (the complex routine
// conjugates parts of it in between).
template <typename T>
blasint Hegs2(blasint itype, char uplo, blasint n, T* a, blasint lda, T* b,
              blasint ldb) {
  using R = RealT<T>;
  const R one = R(1);
  const R half = R(0.5);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (itype < 1 || itype > 3) {
    info = -1;
  } else if (!upper && !Lsame(uplo, 'L')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYGS2", "HEGS2"), -info);
    return info;
  }

  if (itype == 1) {
    if (upper) {
      // Compute inv(U**H)*A*inv(U).
      for (blasint k = 1; k <= n; ++k) {
        // Update the upper triangle of A(k:n,k:n).
        R akk = std::real(A(k, k));
        const R bkk = std::real(B(k, k));
        akk = akk / (bkk * bkk);
        A(k, k) = akk;
        if (k < n) {
          Rscal(n - k, one / bkk, A.ptr(k, k + 1), lda);
          const T ct = -half * akk;
          Lacgv(n - k, A.ptr(k, k + 1), lda);
          Lacgv(n - k, B.ptr(k, k + 1), ldb);
          Axpy(n - k, ct, B.ptr(k, k + 1), ldb, A.ptr(k, k + 1), lda);
          Her2(uplo, n - k, -cone, A.ptr(k, k + 1), lda, B.ptr(k, k + 1), ldb,
               A.ptr(k + 1, k + 1), lda);
          Axpy(n - k, ct, B.ptr(k, k + 1), ldb, A.ptr(k, k + 1), lda);
          Lacgv(n - k, B.ptr(k, k + 1), ldb);
          Trsv(uplo, ConjTrans<T>(), 'N', n - k, B.ptr(k + 1, k + 1), ldb,
               A.ptr(k, k + 1), lda);
          Lacgv(n - k, A.ptr(k, k + 1), lda);
        }
      }
    } else {
      // Compute inv(L)*A*inv(L**H).
      for (blasint k = 1; k <= n; ++k) {
        // Update the lower triangle of A(k:n,k:n).
        R akk = std::real(A(k, k));
        const R bkk = std::real(B(k, k));
        akk = akk / (bkk * bkk);
        A(k, k) = akk;
        if (k < n) {
          Rscal(n - k, one / bkk, A.ptr(k + 1, k), 1);
          const T ct = -half * akk;
          Axpy(n - k, ct, B.ptr(k + 1, k), 1, A.ptr(k + 1, k), 1);
          Her2(uplo, n - k, -cone, A.ptr(k + 1, k), 1, B.ptr(k + 1, k), 1,
               A.ptr(k + 1, k + 1), lda);
          Axpy(n - k, ct, B.ptr(k + 1, k), 1, A.ptr(k + 1, k), 1);
          Trsv(uplo, 'N', 'N', n - k, B.ptr(k + 1, k + 1), ldb, A.ptr(k + 1, k),
               1);
        }
      }
    }
  } else {
    if (upper) {
      // Compute U*A*U**H.
      for (blasint k = 1; k <= n; ++k) {
        // Update the upper triangle of A(1:k,1:k).
        const R akk = std::real(A(k, k));
        const R bkk = std::real(B(k, k));
        Trmv(uplo, 'N', 'N', k - 1, b, ldb, A.ptr(1, k), 1);
        const T ct = half * akk;
        Axpy(k - 1, ct, B.ptr(1, k), 1, A.ptr(1, k), 1);
        Her2(uplo, k - 1, cone, A.ptr(1, k), 1, B.ptr(1, k), 1, a, lda);
        Axpy(k - 1, ct, B.ptr(1, k), 1, A.ptr(1, k), 1);
        Rscal(k - 1, bkk, A.ptr(1, k), 1);
        A(k, k) = akk * (bkk * bkk);
      }
    } else {
      // Compute L**H *A*L.
      for (blasint k = 1; k <= n; ++k) {
        // Update the lower triangle of A(1:k,1:k).
        const R akk = std::real(A(k, k));
        const R bkk = std::real(B(k, k));
        Lacgv(k - 1, A.ptr(k, 1), lda);
        Trmv(uplo, ConjTrans<T>(), 'N', k - 1, b, ldb, A.ptr(k, 1), lda);
        const T ct = half * akk;
        Lacgv(k - 1, B.ptr(k, 1), ldb);
        Axpy(k - 1, ct, B.ptr(k, 1), ldb, A.ptr(k, 1), lda);
        Her2(uplo, k - 1, cone, A.ptr(k, 1), lda, B.ptr(k, 1), ldb, a, lda);
        Axpy(k - 1, ct, B.ptr(k, 1), ldb, A.ptr(k, 1), lda);
        Lacgv(k - 1, B.ptr(k, 1), ldb);
        Rscal(k - 1, bkk, A.ptr(k, 1), lda);
        Lacgv(k - 1, A.ptr(k, 1), lda);
        A(k, k) = akk * (bkk * bkk);
      }
    }
  }
  return info;
}

// xSYGST / xHEGST: the reduction of xSYGS2 / xHEGS2 (see there), blocked:
// blocks of NB columns (from ILAENV) are reduced with xSYGS2 / xHEGS2 and
// the rest of the matrix updated with Level 3 BLAS.
template <typename T>
blasint Hegst(blasint itype, char uplo, blasint n, T* a, blasint lda, T* b,
              blasint ldb) {
  using R = RealT<T>;
  const R one = R(1);
  const T cone = T(1);
  const T half = T(R(0.5));
  const Matrix<T> A(a, lda);
  const Matrix<T> B(b, ldb);
  const char* name = RealOrComplex<T>("SYGST", "HEGST");

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (itype < 1 || itype > 3) {
    info = -1;
  } else if (!upper && !Lsame(uplo, 'L')) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Max<blasint>(1, n)) {
    info = -5;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Determine the block size for this environment.
  const char opts[2] = {uplo, '\0'};
  const blasint nb = Ilaenv<T>(1, name, opts, n, -1, -1, -1);

  if (nb <= 1 || nb >= n) {
    // Use unblocked code.
    info = Hegs2(itype, uplo, n, a, lda, b, ldb);
  } else {
    // Use blocked code.
    if (itype == 1) {
      if (upper) {
        // Compute inv(U**H)*A*inv(U).
        for (blasint k = 1; k <= n; k += nb) {
          const blasint kb = Min(n - k + 1, nb);

          // Update the upper triangle of A(k:n,k:n).
          info = Hegs2(itype, uplo, kb, A.ptr(k, k), lda, B.ptr(k, k), ldb);
          if (k + kb <= n) {
            Trsm('L', uplo, ConjTrans<T>(), 'N', kb, n - k - kb + 1, cone,
                 B.ptr(k, k), ldb, A.ptr(k, k + kb), lda);
            Hemm('L', uplo, kb, n - k - kb + 1, -half, A.ptr(k, k), lda,
                 B.ptr(k, k + kb), ldb, cone, A.ptr(k, k + kb), lda);
            Her2k(uplo, ConjTrans<T>(), n - k - kb + 1, kb, -cone,
                  A.ptr(k, k + kb), lda, B.ptr(k, k + kb), ldb, one,
                  A.ptr(k + kb, k + kb), lda);
            Hemm('L', uplo, kb, n - k - kb + 1, -half, A.ptr(k, k), lda,
                 B.ptr(k, k + kb), ldb, cone, A.ptr(k, k + kb), lda);
            Trsm('R', uplo, 'N', 'N', kb, n - k - kb + 1, cone,
                 B.ptr(k + kb, k + kb), ldb, A.ptr(k, k + kb), lda);
          }
        }
      } else {
        // Compute inv(L)*A*inv(L**H).
        for (blasint k = 1; k <= n; k += nb) {
          const blasint kb = Min(n - k + 1, nb);

          // Update the lower triangle of A(k:n,k:n).
          info = Hegs2(itype, uplo, kb, A.ptr(k, k), lda, B.ptr(k, k), ldb);
          if (k + kb <= n) {
            Trsm('R', uplo, ConjTrans<T>(), 'N', n - k - kb + 1, kb, cone,
                 B.ptr(k, k), ldb, A.ptr(k + kb, k), lda);
            Hemm('R', uplo, n - k - kb + 1, kb, -half, A.ptr(k, k), lda,
                 B.ptr(k + kb, k), ldb, cone, A.ptr(k + kb, k), lda);
            Her2k(uplo, 'N', n - k - kb + 1, kb, -cone, A.ptr(k + kb, k), lda,
                  B.ptr(k + kb, k), ldb, one, A.ptr(k + kb, k + kb), lda);
            Hemm('R', uplo, n - k - kb + 1, kb, -half, A.ptr(k, k), lda,
                 B.ptr(k + kb, k), ldb, cone, A.ptr(k + kb, k), lda);
            Trsm('L', uplo, 'N', 'N', n - k - kb + 1, kb, cone,
                 B.ptr(k + kb, k + kb), ldb, A.ptr(k + kb, k), lda);
          }
        }
      }
    } else {
      if (upper) {
        // Compute U*A*U**H.
        for (blasint k = 1; k <= n; k += nb) {
          const blasint kb = Min(n - k + 1, nb);

          // Update the upper triangle of A(1:k+kb-1,1:k+kb-1).
          Trmm('L', uplo, 'N', 'N', k - 1, kb, cone, b, ldb, A.ptr(1, k), lda);
          Hemm('R', uplo, k - 1, kb, half, A.ptr(k, k), lda, B.ptr(1, k), ldb,
               cone, A.ptr(1, k), lda);
          Her2k(uplo, 'N', k - 1, kb, cone, A.ptr(1, k), lda, B.ptr(1, k), ldb,
                one, a, lda);
          Hemm('R', uplo, k - 1, kb, half, A.ptr(k, k), lda, B.ptr(1, k), ldb,
               cone, A.ptr(1, k), lda);
          Trmm('R', uplo, ConjTrans<T>(), 'N', k - 1, kb, cone, B.ptr(k, k),
               ldb, A.ptr(1, k), lda);
          info = Hegs2(itype, uplo, kb, A.ptr(k, k), lda, B.ptr(k, k), ldb);
        }
      } else {
        // Compute L**H*A*L.
        for (blasint k = 1; k <= n; k += nb) {
          const blasint kb = Min(n - k + 1, nb);

          // Update the lower triangle of A(1:k+kb-1,1:k+kb-1).
          Trmm('R', uplo, 'N', 'N', kb, k - 1, cone, b, ldb, A.ptr(k, 1), lda);
          Hemm('L', uplo, kb, k - 1, half, A.ptr(k, k), lda, B.ptr(k, 1), ldb,
               cone, A.ptr(k, 1), lda);
          Her2k(uplo, ConjTrans<T>(), k - 1, kb, cone, A.ptr(k, 1), lda,
                B.ptr(k, 1), ldb, one, a, lda);
          Hemm('L', uplo, kb, k - 1, half, A.ptr(k, k), lda, B.ptr(k, 1), ldb,
               cone, A.ptr(k, 1), lda);
          Trmm('L', uplo, ConjTrans<T>(), 'N', kb, k - 1, cone, B.ptr(k, k),
               ldb, A.ptr(k, 1), lda);
          info = Hegs2(itype, uplo, kb, A.ptr(k, k), lda, B.ptr(k, k), ldb);
        }
      }
    }
  }
  return info;
}

namespace {

// The back transformation of the eigenvectors (the first NEIG columns of
// A) of the standard problem to those of the generalized one (xSYGV /
// xHEGV, xSYGVD / xHEGVD).
template <typename T>
void BackTransform(blasint itype, char uplo, blasint n, blasint neig,
                   const T* b, blasint ldb, T* a, blasint lda) {
  const T one = T(1);
  const bool upper = Lsame(uplo, 'U');
  if (itype == 1 || itype == 2) {
    // For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
    // backtransform eigenvectors: x = inv(L)**H *y or inv(U)*y.
    const char trans = upper ? 'N' : ConjTrans<T>();
    Trsm('L', uplo, trans, 'N', n, neig, one, b, ldb, a, lda);
  } else if (itype == 3) {
    // For B*A*x=(lambda)*x;
    // backtransform eigenvectors: x = L*y or U**H *y.
    const char trans = upper ? ConjTrans<T>() : 'N';
    Trmm('L', uplo, trans, 'N', n, neig, one, b, ldb, a, lda);
  }
}

}  // namespace

// xSYGV / xHEGV: computes all eigenvalues and, optionally (JOBZ = 'V'),
// eigenvectors of the generalized problem A*x = lambda*B*x (ITYPE = 1),
// A*B*x = lambda*x (2) or B*A*x = lambda*x (3) with symmetric (Hermitian)
// A and positive definite B (the triangles UPLO): B is factored with
// xPOTRF (B returns the factor), the problem reduced to standard form
// (xSYGST / xHEGST) and solved with xSYEV / xHEEV, and the eigenvectors
// transformed back. On exit, W holds the eigenvalues in ascending order
// and A the eigenvectors (normalized as Z**H*B*Z = I for ITYPE = 1, 2,
// Z**H*inv(B)*Z = I for ITYPE = 3), or A is destroyed (JOBZ = 'N').
// Returns INFO: 0, -i for an illegal argument i, i in 1..N if xSYEV /
// xHEEV failed to converge, or N+i if the leading minor of order i of B is
// not positive.
//
// Workspaces as xSYEV / xHEEV: real T: LWORK >= max(1, 3*N-1), optimal
// max(that, (NB+2)*N); complex T: LWORK >= max(1, 2*N-1), optimal
// max(1, (NB+1)*N) and RWORK (max(1, 3*N-2) elements). LWORK = -1 is a
// workspace query.
template <typename T>
blasint Hegv(blasint itype, char jobz, char uplo, blasint n, T* a, blasint lda,
             T* b, blasint ldb, RealT<T>* w, T* work, blasint lwork,
             RealT<T>* rwork) {
  constexpr bool kComplex = IsComplex<T>::value;
  const Vector<T> WORK(work);

  // Test the input parameters.
  const bool wantz = Lsame(jobz, 'V');
  const bool upper = Lsame(uplo, 'U');
  const bool lquery = (lwork == -1);

  blasint info = 0;
  if (itype < 1 || itype > 3) {
    info = -1;
  } else if (!(wantz || Lsame(jobz, 'N'))) {
    info = -2;
  } else if (!(upper || Lsame(uplo, 'L'))) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }

  blasint lwkopt = 0;
  if (info == 0) {
    const char opts[2] = {uplo, '\0'};
    const blasint nb =
        Ilaenv<T>(1, RealOrComplex<T>("SYTRD", "HETRD"), opts, n, -1, -1, -1);
    blasint lwkmin;
    if constexpr (kComplex) {
      lwkmin = Max<blasint>(1, 2 * n - 1);
      lwkopt = Max<blasint>(1, (nb + 1) * n);
    } else {
      lwkmin = Max<blasint>(1, 3 * n - 1);
      lwkopt = Max(lwkmin, (nb + 2) * n);
    }
    WORK(1) = WorkSize<T>(lwkopt);

    if (lwork < lwkmin && !lquery) info = -11;
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYGV", "HEGV"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) return info;

  // Form a Cholesky factorization of B.
  info = Potrf(uplo, n, b, ldb);
  if (info != 0) {
    info = n + info;
    return info;
  }

  // Transform problem to standard eigenvalue problem and solve.
  Hegst(itype, uplo, n, a, lda, b, ldb);
  info = Heev(jobz, uplo, n, a, lda, w, work, lwork, rwork);

  if (wantz) {
    // Backtransform eigenvectors to the original problem.
    blasint neig = n;
    if (info > 0) neig = info - 1;
    BackTransform(itype, uplo, n, neig, b, ldb, a, lda);
  }

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xSYGVD / xHEGVD: as xSYGV / xHEGV (see there), with the divide and
// conquer xSYEVD / xHEEVD (whose workspaces they take) for the standard
// problem; the eigenvectors are transformed back only if it succeeded.
// The optimal workspace sizes returned are the larger of the minimal ones
// and those xSYEVD / xHEEVD returned.
template <typename T>
blasint Hegvd(blasint itype, char jobz, char uplo, blasint n, T* a, blasint lda,
              T* b, blasint ldb, RealT<T>* w, T* work, blasint lwork,
              RealT<T>* rwork, blasint lrwork, blasint* iwork, blasint liwork) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const Vector<T> WORK(work);
  const Vector<R> RWORK(rwork);
  const Vector<blasint> IWORK(iwork);

  // Test the input parameters.
  const bool wantz = Lsame(jobz, 'V');
  const bool upper = Lsame(uplo, 'U');
  bool lquery = (lwork == -1 || liwork == -1);
  if constexpr (kComplex) lquery = lquery || lrwork == -1;

  blasint info = 0;
  blasint lwmin, lrwmin = 1, liwmin;
  if (n <= 1) {
    lwmin = 1;
    lrwmin = 1;
    liwmin = 1;
  } else if (wantz) {
    if constexpr (kComplex) {
      lwmin = 2 * n + n * n;
      lrwmin = 1 + 5 * n + 2 * n * n;
    } else {
      lwmin = 1 + 6 * n + 2 * (n * n);
    }
    liwmin = 3 + 5 * n;
  } else {
    if constexpr (kComplex) {
      lwmin = n + 1;
      lrwmin = n;
    } else {
      lwmin = 2 * n + 1;
    }
    liwmin = 1;
  }
  blasint lopt = lwmin;
  blasint lropt = lrwmin;
  blasint liopt = liwmin;
  if (itype < 1 || itype > 3) {
    info = -1;
  } else if (!(wantz || Lsame(jobz, 'N'))) {
    info = -2;
  } else if (!(upper || Lsame(uplo, 'L'))) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (lda < Max<blasint>(1, n)) {
    info = -6;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -8;
  }

  if (info == 0) {
    WORK(1) = WorkSize<T>(lopt);
    if constexpr (kComplex) RWORK(1) = WorkSize<R>(lropt);
    IWORK(1) = liopt;

    if (lwork < lwmin && !lquery) {
      info = -11;
    } else if (kComplex && lrwork < lrwmin && !lquery) {
      info = -13;
    } else if (liwork < liwmin && !lquery) {
      info = kComplex ? -15 : -13;
    }
  }

  if (info != 0) {
    Xerbla<T>(RealOrComplex<T>("SYGVD", "HEGVD"), -info);
    return info;
  }
  if (lquery) return info;

  // Quick return if possible.
  if (n == 0) return info;

  // Form a Cholesky factorization of B.
  info = Potrf(uplo, n, b, ldb);
  if (info != 0) {
    info = n + info;
    return info;
  }

  // Transform problem to standard eigenvalue problem and solve.
  Hegst(itype, uplo, n, a, lda, b, ldb);
  info = Heevd(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork,
               liwork);
  lopt = static_cast<blasint>(Max(R(lopt), std::real(WORK(1))));
  if constexpr (kComplex) {
    lropt = static_cast<blasint>(Max(R(lropt), RWORK(1)));
  }
  liopt = static_cast<blasint>(Max(R(liopt), R(IWORK(1))));

  if (wantz && info == 0) {
    // Backtransform eigenvectors to the original problem.
    BackTransform(itype, uplo, n, n, b, ldb, a, lda);
  }

  WORK(1) = WorkSize<T>(lopt);
  if constexpr (kComplex) RWORK(1) = WorkSize<R>(lropt);
  IWORK(1) = liopt;
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                          \
  template blasint Hegs2<T>(blasint, char, blasint, T*, blasint, T*, blasint); \
  template blasint Hegst<T>(blasint, char, blasint, T*, blasint, T*, blasint); \
  template blasint Hegv<T>(blasint, char, char, blasint, T*, blasint, T*,      \
                           blasint, RealT<T>*, T*, blasint, RealT<T>*);        \
  template blasint Hegvd<T>(blasint, char, char, blasint, T*, blasint, T*,     \
                            blasint, RealT<T>*, T*, blasint, RealT<T>*,        \
                            blasint, blasint*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

// ?sygst / ?hegst take B as const (as the reference lapack.h); the complex
// routine conjugates parts of it in between and restores them.
#define HWYBLAS_LAPACK_HEGST(p, gst, gs2, T, R)                           \
  HWYBLAS_API void p##gs2##_(const hwyblas_int* itype, const char* uplo,  \
                             const hwyblas_int* n, R* a,                  \
                             const hwyblas_int* lda, R* b,                \
                             const hwyblas_int* ldb, hwyblas_int* info) { \
    *info = hwyblas::lapack::Hegs2(*itype, *uplo, *n, As<T>(a), *lda,     \
                                   As<T>(b), *ldb);                       \
  }                                                                       \
  HWYBLAS_API void p##gst##_(const hwyblas_int* itype, const char* uplo,  \
                             const hwyblas_int* n, R* a,                  \
                             const hwyblas_int* lda, const R* b,          \
                             const hwyblas_int* ldb, hwyblas_int* info) { \
    *info = hwyblas::lapack::Hegst(*itype, *uplo, *n, As<T>(a), *lda,     \
                                   As<T>(const_cast<R*>(b)), *ldb);       \
  }

HWYBLAS_LAPACK_HEGST(s, sygst, sygs2, float, float)
HWYBLAS_LAPACK_HEGST(d, sygst, sygs2, double, double)
HWYBLAS_LAPACK_HEGST(c, hegst, hegs2, c32, float)
HWYBLAS_LAPACK_HEGST(z, hegst, hegs2, c64, double)

#undef HWYBLAS_LAPACK_HEGST

#define HWYBLAS_LAPACK_SYGV(p, R)                                              \
  HWYBLAS_API void p##sygv_(const hwyblas_int* itype, const char* jobz,        \
                            const char* uplo, const hwyblas_int* n, R* a,      \
                            const hwyblas_int* lda, R* b,                      \
                            const hwyblas_int* ldb, R* w, R* work,             \
                            const hwyblas_int* lwork, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Hegv(*itype, *jobz, *uplo, *n, a, *lda, b, *ldb,  \
                                  w, work, *lwork, static_cast<R*>(nullptr));  \
  }                                                                            \
  HWYBLAS_API void p##sygvd_(                                                  \
      const hwyblas_int* itype, const char* jobz, const char* uplo,            \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,                \
      const hwyblas_int* ldb, R* w, R* work, const hwyblas_int* lwork,         \
      hwyblas_int* iwork, const hwyblas_int* liwork, hwyblas_int* info) {      \
    *info = hwyblas::lapack::Hegvd(*itype, *jobz, *uplo, *n, a, *lda, b, *ldb, \
                                   w, work, *lwork, static_cast<R*>(nullptr),  \
                                   0, iwork, *liwork);                         \
  }

HWYBLAS_LAPACK_SYGV(s, float)
HWYBLAS_LAPACK_SYGV(d, double)

#undef HWYBLAS_LAPACK_SYGV

#define HWYBLAS_LAPACK_HEGV(p, T, R)                                          \
  HWYBLAS_API void p##hegv_(                                                  \
      const hwyblas_int* itype, const char* jobz, const char* uplo,           \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,               \
      const hwyblas_int* ldb, R* w, R* work, const hwyblas_int* lwork,        \
      R* rwork, hwyblas_int* info) {                                          \
    *info =                                                                   \
        hwyblas::lapack::Hegv(*itype, *jobz, *uplo, *n, As<T>(a), *lda,       \
                              As<T>(b), *ldb, w, As<T>(work), *lwork, rwork); \
  }                                                                           \
  HWYBLAS_API void p##hegvd_(                                                 \
      const hwyblas_int* itype, const char* jobz, const char* uplo,           \
      const hwyblas_int* n, R* a, const hwyblas_int* lda, R* b,               \
      const hwyblas_int* ldb, R* w, R* work, const hwyblas_int* lwork,        \
      R* rwork, const hwyblas_int* lrwork, hwyblas_int* iwork,                \
      const hwyblas_int* liwork, hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Hegvd(*itype, *jobz, *uplo, *n, As<T>(a), *lda,  \
                                   As<T>(b), *ldb, w, As<T>(work), *lwork,    \
                                   rwork, *lrwork, iwork, *liwork);           \
  }

HWYBLAS_LAPACK_HEGV(c, c32, float)
HWYBLAS_LAPACK_HEGV(z, c64, double)

#undef HWYBLAS_LAPACK_HEGV

}  // extern "C"
