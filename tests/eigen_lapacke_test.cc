// Checks that Eigen, compiled with EIGEN_USE_LAPACKE (and EIGEN_USE_BLAS),
// routes its decompositions through hwyBLAS's LAPACKE functions, in both
// storage orders, and gets correct results.
//
// Eigen declares the LAPACKE functions in its own copy of lapacke.h: the
// test links only because hwyBLAS provides every LAPACKE function Eigen may
// call. That the hwyBLAS routines run is checked by counting their ILAENV
// queries through hwyblas_set_ilaenv (Eigen's own code makes none).
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include <complex>
#include <cstring>
#include <set>
#include <string>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "test_util.h"

// Not hwyblas/lapack.h: its Fortran prototypes would conflict with those of
// Eigen's lapacke.h.
extern "C" {
typedef int (*IlaenvFunction)(const int* ispec, const char* name,
                              const char* opts, const int* n1, const int* n2,
                              const int* n3, const int* n4, size_t name_len,
                              size_t opts_len);
#if defined(_WIN32)
__declspec(dllimport)
#endif
IlaenvFunction hwyblas_set_ilaenv(IlaenvFunction function);
#if defined(_WIN32)
__declspec(dllimport)
#endif
int ilaenv_(const int* ispec, const char* name, const char* opts, const int* n1,
            const int* n2, const int* n3, const int* n4, size_t name_len,
            size_t opts_len);
}

namespace hwyblas_test {
namespace {

// The routine names of the ILAENV queries since the last Clear().
std::set<std::string>& Queried() {
  static std::set<std::string> names;
  return names;
}

int CountingIlaenv(const int* ispec, const char* name, const char* opts,
                   const int* n1, const int* n2, const int* n3, const int* n4,
                   size_t name_len, size_t opts_len) {
  Queried().insert(std::string(name, strnlen(name, name_len)));
  return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
}

// Installs CountingIlaenv for the lifetime of the object.
class ScopedIlaenvCount {
 public:
  ScopedIlaenvCount() : previous_(hwyblas_set_ilaenv(CountingIlaenv)) {
    Queried().clear();
  }
  ~ScopedIlaenvCount() { hwyblas_set_ilaenv(previous_); }
  ScopedIlaenvCount(const ScopedIlaenvCount&) = delete;
  ScopedIlaenvCount& operator=(const ScopedIlaenvCount&) = delete;

  // Whether a hwyBLAS routine queried ILAENV for `name` (e.g. "DGETRF").
  static bool Used(const std::string& name) {
    return Queried().count(name) != 0;
  }

 private:
  IlaenvFunction previous_;
};

template <typename T>
char Prefix() {
  if (std::is_same_v<T, float>) return 'S';
  if (std::is_same_v<T, double>) return 'D';
  if (std::is_same_v<T, std::complex<float>>) return 'C';
  return 'Z';
}

// The name of the routine that ILAENV is queried for, e.g. Name<double>
// ("GETRF") == "DGETRF".
template <typename T>
std::string Name(const char* rest) {
  return Prefix<T>() + std::string(rest);
}

template <typename MatA, typename MatB>
double RelError(const MatA& actual, const MatB& expected) {
  return static_cast<double>((actual - expected).norm()) /
         static_cast<double>(expected.norm());
}

// Scalar type and storage order.
template <typename T, int kOrder>
struct Param {
  using Scalar = T;
  static constexpr int kStorage = kOrder;
};

template <typename P>
class EigenLapackeTest : public ::testing::Test {};

using Params = ::testing::Types<
    Param<float, Eigen::ColMajor>, Param<double, Eigen::ColMajor>,
    Param<std::complex<float>, Eigen::ColMajor>,
    Param<std::complex<double>, Eigen::ColMajor>, Param<float, Eigen::RowMajor>,
    Param<double, Eigen::RowMajor>, Param<std::complex<float>, Eigen::RowMajor>,
    Param<std::complex<double>, Eigen::RowMajor>>;
TYPED_TEST_SUITE(EigenLapackeTest, Params);

// Large enough for the blocked LAPACK code paths.
constexpr Eigen::Index kN = 150;

TYPED_TEST(EigenLapackeTest, Cholesky) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const Mat m = Mat::Random(kN, kN);
  const Mat spd = m * m.adjoint() + Mat::Identity(kN, kN) * T(kN);
  const double tol = 100 * kN * Eps<T>();
  ScopedIlaenvCount count;
  const Eigen::LLT<Mat, Eigen::Lower> lower(spd);
  ASSERT_EQ(lower.info(), Eigen::Success);
  EXPECT_LE(RelError(lower.reconstructedMatrix(), spd), tol);
  const Eigen::LLT<Mat, Eigen::Upper> upper(spd);
  ASSERT_EQ(upper.info(), Eigen::Success);
  EXPECT_LE(RelError(upper.reconstructedMatrix(), spd), tol);
  EXPECT_TRUE(count.Used(Name<T>("POTRF")));
  // Not positive definite: LAPACK's INFO > 0 becomes NumericalIssue.
  Mat indefinite = spd;
  indefinite(kN / 2, kN / 2) = T(-1);
  EXPECT_EQ(Eigen::LLT<Mat>(indefinite).info(), Eigen::NumericalIssue);
}

TYPED_TEST(EigenLapackeTest, PartialPivLu) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const Mat a = Mat::Random(kN, kN);
  const double tol = 100 * kN * Eps<T>();
  ScopedIlaenvCount count;
  const Eigen::PartialPivLU<Mat> lu(a);
  EXPECT_LE(RelError(lu.reconstructedMatrix(), a), tol);
  const Mat rhs = Mat::Random(kN, 3);
  EXPECT_LE(RelError(Mat(a * lu.solve(rhs)), rhs), 100 * tol);
  EXPECT_TRUE(count.Used(Name<T>("GETRF")));
}

TYPED_TEST(EigenLapackeTest, HouseholderQr) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const double tol = 100 * kN * Eps<T>();
  for (const auto& [m, n] :
       {std::pair<Eigen::Index, Eigen::Index>{kN, 90}, {90, kN}}) {
    const Mat a = Mat::Random(m, n);
    ScopedIlaenvCount count;
    const Eigen::HouseholderQR<Mat> qr(a);
    const Mat q = qr.householderQ();
    const Mat r = qr.matrixQR().template triangularView<Eigen::Upper>();
    EXPECT_LE(RelError(Mat(q * r), a), tol);
    EXPECT_LE(RelError(Mat(q.adjoint() * q), Mat(Mat::Identity(m, m))), tol);
    EXPECT_TRUE(count.Used(Name<T>("GEQRF")));
  }
}

TYPED_TEST(EigenLapackeTest, ColPivHouseholderQr) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const double tol = 100 * kN * Eps<T>();
  // Rank 60.
  const Mat a = Mat::Random(kN, 60) * Mat::Random(60, 100);
  ScopedIlaenvCount count;
  const Eigen::ColPivHouseholderQR<Mat> qr(a);
  const Mat q = qr.householderQ();
  const Mat r = qr.matrixR().template triangularView<Eigen::Upper>();
  EXPECT_LE(RelError(Mat(q * r * qr.colsPermutation().transpose()), a), tol);
  EXPECT_EQ(qr.rank(), 60);
  // xGEQP3 queries the block size of xGEQRF.
  EXPECT_TRUE(count.Used(Name<T>("GEQRF")));
}

template <typename Mat, typename Svd>
void CheckSvd(const Mat& a, const Svd& svd, bool full, double tol) {
  using Real = typename Eigen::NumTraits<typename Mat::Scalar>::Real;
  const Eigen::Index k = std::min(a.rows(), a.cols());
  const Eigen::Matrix<Real, Eigen::Dynamic, 1>& s = svd.singularValues();
  ASSERT_EQ(s.size(), k);
  for (Eigen::Index i = 1; i < k; ++i) EXPECT_GE(s(i - 1), s(i));
  const Mat& u = svd.matrixU();
  const Mat& v = svd.matrixV();
  EXPECT_EQ(u.cols(), full ? a.rows() : k);
  EXPECT_EQ(v.cols(), full ? a.cols() : k);
  const Mat reconstructed =
      u.leftCols(k) * s.template cast<typename Mat::Scalar>().asDiagonal() *
      v.leftCols(k).adjoint();
  EXPECT_LE(RelError(reconstructed, a), tol);
  EXPECT_LE(
      RelError(Mat(u.adjoint() * u), Mat(Mat::Identity(u.cols(), u.cols()))),
      tol);
  EXPECT_LE(
      RelError(Mat(v.adjoint() * v), Mat(Mat::Identity(v.cols(), v.cols()))),
      tol);
}

TYPED_TEST(EigenLapackeTest, JacobiSvd) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const double tol = 100 * kN * Eps<T>();
  for (const auto& [m, n] :
       {std::pair<Eigen::Index, Eigen::Index>{kN, 70}, {70, kN}}) {
    const Mat a = Mat::Random(m, n);
    ScopedIlaenvCount count;
    const Eigen::JacobiSVD<Mat, Eigen::ComputeThinU | Eigen::ComputeThinV> thin(
        a);
    CheckSvd(a, thin, false, tol);
    const Eigen::JacobiSVD<Mat, Eigen::ComputeFullU | Eigen::ComputeFullV> full(
        a);
    CheckSvd(a, full, true, tol);
    EXPECT_TRUE(count.Used(Name<T>("GESVD")));
  }
}

TYPED_TEST(EigenLapackeTest, BdcSvd) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const double tol = 100 * kN * Eps<T>();
  for (const auto& [m, n] :
       {std::pair<Eigen::Index, Eigen::Index>{kN, 70}, {70, kN}}) {
    const Mat a = Mat::Random(m, n);
    ScopedIlaenvCount count;
    const Eigen::BDCSVD<Mat, Eigen::ComputeThinU | Eigen::ComputeThinV> thin(a);
    CheckSvd(a, thin, false, tol);
    const Eigen::BDCSVD<Mat, Eigen::ComputeFullU | Eigen::ComputeFullV> full(a);
    CheckSvd(a, full, true, tol);
    // xGESDD queries ILAENV only through xGEBRD and the like.
    EXPECT_TRUE(count.Used(Name<T>("GEBRD")));
  }
}

TYPED_TEST(EigenLapackeTest, SelfAdjointEigenSolver) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const double tol = 100 * kN * Eps<T>();
  const Mat m = Mat::Random(kN, kN);
  const Mat h = m + m.adjoint();
  ScopedIlaenvCount count;
  const Eigen::SelfAdjointEigenSolver<Mat> eig(h);
  ASSERT_EQ(eig.info(), Eigen::Success);
  const Mat& v = eig.eigenvectors();
  const auto& w = eig.eigenvalues();
  for (Eigen::Index i = 1; i < kN; ++i) EXPECT_GE(w(i), w(i - 1));
  EXPECT_LE(
      RelError(Mat(v * w.template cast<T>().asDiagonal() * v.adjoint()), h),
      tol);
  EXPECT_TRUE(
      count.Used(Name<T>(Eigen::NumTraits<T>::IsComplex ? "HETRD" : "SYTRD")));
  // Eigenvalues only.
  const Eigen::SelfAdjointEigenSolver<Mat> values(h, Eigen::EigenvaluesOnly);
  ASSERT_EQ(values.info(), Eigen::Success);
  EXPECT_LE(static_cast<double>((values.eigenvalues() - w).norm()),
            tol * static_cast<double>(w.norm()));
}

TYPED_TEST(EigenLapackeTest, Schur) {
  using T = typename TypeParam::Scalar;
  using Mat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, TypeParam::kStorage>;
  const double tol = 100 * kN * Eps<T>();
  const Mat a = Mat::Random(kN, kN);
  ScopedIlaenvCount count;
  Mat u, t;
  if constexpr (Eigen::NumTraits<T>::IsComplex) {
    const Eigen::ComplexSchur<Mat> schur(a);
    ASSERT_EQ(schur.info(), Eigen::Success);
    u = schur.matrixU();
    t = schur.matrixT();
    // Upper triangular.
    EXPECT_EQ(Mat(t.template triangularView<Eigen::StrictlyLower>()).norm(), 0);
  } else {
    const Eigen::RealSchur<Mat> schur(a);
    ASSERT_EQ(schur.info(), Eigen::Success);
    u = schur.matrixU();
    t = schur.matrixT();
    // Quasi upper triangular: no two consecutive subdiagonal entries.
    for (Eigen::Index i = 0; i + 2 < kN; ++i) {
      EXPECT_TRUE(t(i + 1, i) == T(0) || t(i + 2, i + 1) == T(0));
    }
    for (Eigen::Index j = 0; j < kN; ++j) {
      for (Eigen::Index i = j + 2; i < kN; ++i) EXPECT_EQ(t(i, j), T(0));
    }
  }
  EXPECT_LE(RelError(Mat(u * t * u.adjoint()), a), tol);
  EXPECT_LE(RelError(Mat(u.adjoint() * u), Mat(Mat::Identity(kN, kN))), tol);
  EXPECT_TRUE(count.Used(Name<T>("GEHRD")));
}

}  // namespace
}  // namespace hwyblas_test
