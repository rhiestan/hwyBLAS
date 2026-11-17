// Workspace queries (LWORK = -1) with the arguments that the reference LAPACK
// does not reference there passed as null pointers. Callers leave these unset:
// NumPy's lstsq hands xGELSD an uninitialized RCOND in its query, which made
// the C entry point crash when it read it (intermittently, with the garbage
// on the stack). An argument is read only where the Fortran routine reads it.
//
// Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.

#include <complex>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class WorkspaceQueryTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(WorkspaceQueryTest, ScalarTypes);

using I = hwyblas_int;

// The optimal workspace of a query is at least 1.
template <typename T>
void ExpectQuery(I info, const std::vector<T>& work) {
  EXPECT_EQ(info, 0);
  EXPECT_GE(std::real(work[0]), 1);
}

TYPED_TEST(WorkspaceQueryTest, LeastSquaresWithoutRcond) {
  using T = TypeParam;
  using R = RealT<T>;
  const I m = 6, n = 4, nrhs = 2, lda = 6, ldb = 6, lwork = -1;
  std::vector<T> a(lda * n), b(ldb * nrhs), work(1);
  std::vector<R> s(n), rwork(1);
  std::vector<I> iwork(1), jpvt(n);
  I rank = 0, info = -99;
  const R* rcond = nullptr;
  // xGELSD
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, gelsd, &m, &n, &nrhs, P(a.data()), &lda,
                         P(b.data()), &ldb, s.data(), rcond, &rank,
                         P(work.data()), &lwork, rwork.data(), iwork.data(),
                         &info);
  } else {
    HWYBLAS_CALL_REAL(T, gelsd, &m, &n, &nrhs, P(a.data()), &lda, P(b.data()),
                      &ldb, s.data(), rcond, &rank, P(work.data()), &lwork,
                      iwork.data(), &info);
  }
  ExpectQuery(info, work);
  // xGELSS
  info = -99;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, gelss, &m, &n, &nrhs, P(a.data()), &lda,
                         P(b.data()), &ldb, s.data(), rcond, &rank,
                         P(work.data()), &lwork, rwork.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, gelss, &m, &n, &nrhs, P(a.data()), &lda, P(b.data()),
                      &ldb, s.data(), rcond, &rank, P(work.data()), &lwork,
                      &info);
  }
  ExpectQuery(info, work);
  // xGELSY
  info = -99;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, gelsy, &m, &n, &nrhs, P(a.data()), &lda,
                         P(b.data()), &ldb, jpvt.data(), rcond, &rank,
                         P(work.data()), &lwork, rwork.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, gelsy, &m, &n, &nrhs, P(a.data()), &lda, P(b.data()),
                      &ldb, jpvt.data(), rcond, &rank, P(work.data()), &lwork,
                      &info);
  }
  ExpectQuery(info, work);
}

// RANGE = 'A': VL, VU, IL and IU are not referenced, nor is ABSTOL in a query.
TYPED_TEST(WorkspaceQueryTest, EigenvaluesWithoutRangeArguments) {
  using T = TypeParam;
  using R = RealT<T>;
  const I n = 4, lda = 4, ldz = 4, lwork = -1, liwork = -1, lrwork = -1;
  std::vector<T> a(lda * n), z(ldz * n), work(1);
  std::vector<R> w(n), rwork(1);
  std::vector<I> iwork(1), isuppz(2 * n), ifail(n);
  I m = 0, info = -99;
  const R* none_r = nullptr;
  const I* none_i = nullptr;
  // xSYEVR / xHEEVR
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heevr, "V", "A", "L", &n, P(a.data()), &lda, none_r,
                         none_r, none_i, none_i, none_r, &m, w.data(),
                         P(z.data()), &ldz, isuppz.data(), P(work.data()),
                         &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
                         &info);
  } else {
    HWYBLAS_CALL_REAL(T, syevr, "V", "A", "L", &n, P(a.data()), &lda, none_r,
                      none_r, none_i, none_i, none_r, &m, w.data(), P(z.data()),
                      &ldz, isuppz.data(), P(work.data()), &lwork, iwork.data(),
                      &liwork, &info);
  }
  ExpectQuery(info, work);
  // xSYEVX / xHEEVX
  info = -99;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, heevx, "V", "A", "L", &n, P(a.data()), &lda, none_r,
                         none_r, none_i, none_i, none_r, &m, w.data(),
                         P(z.data()), &ldz, P(work.data()), &lwork,
                         rwork.data(), iwork.data(), ifail.data(), &info);
  } else {
    HWYBLAS_CALL_REAL(T, syevx, "V", "A", "L", &n, P(a.data()), &lda, none_r,
                      none_r, none_i, none_i, none_r, &m, w.data(), P(z.data()),
                      &ldz, P(work.data()), &lwork, iwork.data(), ifail.data(),
                      &info);
  }
  ExpectQuery(info, work);
}

// The query still validates the arguments the range selects: a null pointer
// is only passed where the reference does not look.
TYPED_TEST(WorkspaceQueryTest, QueryChecksTheSelectedRange) {
  using T = TypeParam;
  using R = RealT<T>;
  const I n = 4, lda = 4, ldz = 4, lwork = -1, liwork = -1, lrwork = -1;
  std::vector<T> a(lda * n), z(ldz * n), work(1);
  std::vector<R> w(n), rwork(1);
  std::vector<I> iwork(1), isuppz(2 * n);
  I m = 0, info = -99;
  const R* none_r = nullptr;
  const R vl = 1, vu = 0;  // VU <= VL
  const I il = 3, iu = 2;  // IU < IL
  const I* none_i = nullptr;
  ErrorCapture errors;
  auto syevr = [&](const char* range, const R* lo, const R* hi, const I* first,
                   const I* last) {
    info = -99;
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevr, "V", range, "L", &n, P(a.data()), &lda, lo,
                           hi, first, last, none_r, &m, w.data(), P(z.data()),
                           &ldz, isuppz.data(), P(work.data()), &lwork,
                           rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    } else {
      HWYBLAS_CALL_REAL(T, syevr, "V", range, "L", &n, P(a.data()), &lda, lo,
                        hi, first, last, none_r, &m, w.data(), P(z.data()),
                        &ldz, isuppz.data(), P(work.data()), &lwork,
                        iwork.data(), &liwork, &info);
    }
    return info;
  };
  EXPECT_EQ(syevr("V", &vl, &vu, none_i, none_i), -8);
  EXPECT_EQ(syevr("I", none_r, none_r, &il, &iu), -10);
}

// A computation with RANGE = 'A' and null VL, VU, IL, IU gives the eigenvalues.
TYPED_TEST(WorkspaceQueryTest, EigenvaluesOfADiagonalMatrixWithoutRange) {
  using T = TypeParam;
  using R = RealT<T>;
  const I n = 4, lda = 4, ldz = 4;
  std::vector<T> a(lda * n, T(0)), z(ldz * n), work(1);
  for (I i = 0; i < n; ++i) a[i + i * lda] = T(R(i + 1));
  std::vector<R> w(n), rwork(1);
  std::vector<I> iwork(1), isuppz(2 * n);
  I m = 0, info = -99, lwork = -1, liwork = -1, lrwork = -1;
  const R abstol = 0;
  const R* none_r = nullptr;
  const I* none_i = nullptr;
  auto call = [&] {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, heevr, "V", "A", "L", &n, P(a.data()), &lda,
                           none_r, none_r, none_i, none_i, &abstol, &m,
                           w.data(), P(z.data()), &ldz, isuppz.data(),
                           P(work.data()), &lwork, rwork.data(), &lrwork,
                           iwork.data(), &liwork, &info);
    } else {
      HWYBLAS_CALL_REAL(T, syevr, "V", "A", "L", &n, P(a.data()), &lda, none_r,
                        none_r, none_i, none_i, &abstol, &m, w.data(),
                        P(z.data()), &ldz, isuppz.data(), P(work.data()),
                        &lwork, iwork.data(), &liwork, &info);
    }
  };
  call();  // the query
  ASSERT_EQ(info, 0);
  work.resize(static_cast<size_t>(std::real(work[0])));
  lwork = static_cast<I>(work.size());
  rwork.resize(IsComplex<T>::value ? static_cast<size_t>(rwork[0]) : 1);
  lrwork = static_cast<I>(rwork.size());
  iwork.resize(static_cast<size_t>(iwork[0]));
  liwork = static_cast<I>(iwork.size());
  call();
  ASSERT_EQ(info, 0);
  ASSERT_EQ(m, n);
  for (I i = 0; i < n; ++i) EXPECT_NEAR(w[i], R(i + 1), 10 * Eps<T>());
}

TYPED_TEST(WorkspaceQueryTest, TridiagonalEigenvaluesWithoutRangeArguments) {
  using T = TypeParam;
  using R = RealT<T>;
  const I n = 4, ldz = 4, nzc = 4, lwork = -1, liwork = -1;
  std::vector<R> d(n), e(n), w(n), work(1);
  std::vector<T> z(ldz * n);
  std::vector<I> iwork(1), isuppz(2 * n);
  I m = 0, info = -99;
  const R* none_r = nullptr;
  const I* none_i = nullptr;
  I* no_tryrac = nullptr;
  // xSTEMR
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, stemr, "V", "A", &n, d.data(), e.data(), none_r,
                         none_r, none_i, none_i, &m, w.data(), P(z.data()),
                         &ldz, &nzc, isuppz.data(), no_tryrac, work.data(),
                         &lwork, iwork.data(), &liwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, stemr, "V", "A", &n, d.data(), e.data(), none_r,
                      none_r, none_i, none_i, &m, w.data(), P(z.data()), &ldz,
                      &nzc, isuppz.data(), no_tryrac, work.data(), &lwork,
                      iwork.data(), &liwork, &info);
  }
  EXPECT_EQ(info, 0);
  EXPECT_GE(work[0], 1);
  if constexpr (!IsComplex<T>::value) {
    // xSTEVR
    info = -99;
    const R* abstol = nullptr;
    HWYBLAS_CALL_REAL(T, stevr, "V", "A", &n, d.data(), e.data(), none_r,
                      none_r, none_i, none_i, abstol, &m, w.data(), P(z.data()),
                      &ldz, isuppz.data(), work.data(), &lwork, iwork.data(),
                      &liwork, &info);
    EXPECT_EQ(info, 0);
    EXPECT_GE(work[0], 1);
  }
}

TYPED_TEST(WorkspaceQueryTest, GeneralizedSvdAndSylvesterWithoutTolerances) {
  using T = TypeParam;
  using R = RealT<T>;
  const I m = 4, p = 3, n = 4, lda = 4, ldb = 3, ldu = 4, ldv = 3, ldq = 4;
  const I lwork = -1;
  std::vector<T> a(lda * n), b(ldb * n), u(ldu * m), v(ldv * p), q(ldq * n),
      tau(n), work(1);
  std::vector<R> rwork(2 * n);
  std::vector<I> iwork(n);
  I k = 0, lrows = 0, info = -99;
  const R* no_tol = nullptr;
  // xGGSVP3
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, ggsvp3, "U", "V", "Q", &m, &p, &n, P(a.data()),
                         &lda, P(b.data()), &ldb, no_tol, no_tol, &k, &lrows,
                         P(u.data()), &ldu, P(v.data()), &ldv, P(q.data()),
                         &ldq, iwork.data(), rwork.data(), P(tau.data()),
                         P(work.data()), &lwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, ggsvp3, "U", "V", "Q", &m, &p, &n, P(a.data()), &lda,
                      P(b.data()), &ldb, no_tol, no_tol, &k, &lrows,
                      P(u.data()), &ldu, P(v.data()), &ldv, P(q.data()), &ldq,
                      iwork.data(), P(tau.data()), P(work.data()), &lwork,
                      &info);
  }
  ExpectQuery(info, work);
  // xTGSYL with TRANS = 'T' (complex: 'C'): IJOB is not referenced.
  const I mm = 3, nn = 3, ld = 3;
  std::vector<T> c(ld * nn), f(ld * nn), s(ld * mm), t(ld * mm), x(ld * nn),
      y(ld * nn);
  R scale = 0, dif = 0;
  const I* no_ijob = nullptr;
  info = -99;
  HWYBLAS_CALL(T, tgsyl, IsComplex<T>::value ? "C" : "T", no_ijob, &mm, &nn,
               P(s.data()), &ld, P(t.data()), &ld, P(c.data()), &ld,
               P(x.data()), &ld, P(y.data()), &ld, P(f.data()), &ld, &scale,
               &dif, P(work.data()), &lwork, iwork.data(), &info);
  ExpectQuery(info, work);
}

}  // namespace
}  // namespace hwyblas_test
