// Checks that Eigen, compiled with EIGEN_USE_BLAS, routes its matrix products
// through hwyBLAS and gets correct results.
//
// Only operations backed by routines that hwyBLAS implements may be used
// here: with EIGEN_USE_BLAS, e.g. a triangular solve references dtrsm_.
#define EIGEN_USE_BLAS
#include <complex>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

// Reference product by explicit loops (never dispatched to BLAS).
template <typename MatA, typename MatB>
auto NaiveProduct(const MatA& a, const MatB& b) {
  using Scalar = typename MatA::Scalar;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> c(a.rows(), b.cols());
  for (Eigen::Index j = 0; j < b.cols(); ++j) {
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
      Scalar sum(0);
      for (Eigen::Index p = 0; p < a.cols(); ++p) sum += a(i, p) * b(p, j);
      c(i, j) = sum;
    }
  }
  return c;
}

template <typename T>
class EigenTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(EigenTest, ScalarTypes);

TYPED_TEST(EigenTest, MatrixProducts) {
  using T = TypeParam;
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // Products with a large dimension and k >= 20 go through BLAS gemm.
  const Eigen::Index m = 67, n = 45, k = 83;
  const Mat a = Mat::Random(m, k);
  const Mat b = Mat::Random(k, n);
  const RowMat b_row = b;
  const Mat at = a.transpose();
  const Mat bt = b.transpose();

  const double tol = 50 * k * Eps<T>();
  const auto expected = NaiveProduct(a, b);
  const double scale = expected.norm();

  Mat c = a * b;
  EXPECT_LE((c - expected).norm(), tol * scale);
  c = a * b_row;  // row-major operand: transb = 'T'
  EXPECT_LE((c - expected).norm(), tol * scale);
  c = at.transpose() * bt.transpose();
  EXPECT_LE((c - expected).norm(), tol * scale);
  c = at.adjoint() * b;  // transa = 'C' for complex
  EXPECT_LE((c - NaiveProduct(Mat(at.adjoint()), b)).norm(), tol * scale);

  // Accumulation and scaling: C += alpha * A * B.
  Mat acc = Mat::Random(m, n);
  const Mat acc0 = acc;
  acc.noalias() += T(2) * a * b;
  EXPECT_LE((acc - (acc0 + T(2) * expected)).norm(), tol * scale);
}

TYPED_TEST(EigenTest, MatrixVectorProducts) {
  using T = TypeParam;
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMat =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  const Eigen::Index m = 71, n = 53;
  const Mat a = Mat::Random(m, n);
  const RowMat a_row = a;
  const Vec x = Vec::Random(n);
  const Vec xt = Vec::Random(m);
  const double tol = 50 * (m + n) * Eps<T>();

  const Mat expected = NaiveProduct(a, Mat(x));
  Vec y = a * x;  // gemv 'N'
  EXPECT_LE((y - expected).norm(), tol * expected.norm());
  y = a_row * x;  // row-major: gemv 'T'
  EXPECT_LE((y - expected).norm(), tol * expected.norm());

  const Mat expected_t = NaiveProduct(Mat(a.adjoint()), Mat(xt));
  y = a.adjoint() * xt;  // gemv 'C' for complex
  EXPECT_LE((y - expected_t).norm(), tol * expected_t.norm());

  // A dynamic matrix-matrix product with one column becomes a gemv at runtime.
  const Mat c = a * Mat(x);
  EXPECT_LE((c - expected).norm(), tol * expected.norm());
}

}  // namespace
}  // namespace hwyblas_test
