// Checks that Eigen, compiled with EIGEN_USE_BLAS, routes its matrix
// products, triangular and self-adjoint operations and decompositions through
// hwyBLAS and gets correct results.
//
// Including Eigen/Dense also checks that hwyBLAS provides every BLAS routine
// Eigen may call: a missing one would be a link error.
#define EIGEN_USE_BLAS
#include <complex>

#include "Eigen/Dense"
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

// Relative error of `actual` against `expected` (Frobenius norm).
template <typename MatA, typename MatB>
double RelError(const MatA& actual, const MatB& expected) {
  return static_cast<double>((actual - expected).norm()) /
         static_cast<double>(expected.norm());
}

TYPED_TEST(EigenTest, TriangularAndSelfAdjoint) {
  using T = TypeParam;
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  const Eigen::Index n = 90, k = 33;
  const double tol = 100 * n * Eps<T>();
  Mat a = Mat::Random(n, n);
  // A real diagonal (so that self-adjoint views are valid Hermitian
  // matrices), large enough for well conditioned triangular solves.
  a.diagonal() = a.diagonal().real().template cast<T>();
  a.diagonal().array() += T(n);
  const Mat b = Mat::Random(n, k);
  const Vec x = Vec::Random(n);

  // TRMM / TRMV
  const Mat lower = a.template triangularView<Eigen::Lower>();
  EXPECT_LE(RelError(Mat(a.template triangularView<Eigen::Lower>() * b),
                     NaiveProduct(lower, b)),
            tol);
  EXPECT_LE(RelError(Vec(a.template triangularView<Eigen::Lower>() * x),
                     NaiveProduct(lower, Mat(x))),
            tol);
  const Mat upper_t = Mat(a.template triangularView<Eigen::Upper>()).adjoint();
  EXPECT_LE(
      RelError(Mat(a.template triangularView<Eigen::Upper>().adjoint() * b),
               NaiveProduct(upper_t, b)),
      tol);

  // TRSM: solve and check the residual.
  const Mat sol = a.template triangularView<Eigen::Upper>().solve(b);
  const Mat upper = a.template triangularView<Eigen::Upper>();
  EXPECT_LE(RelError(NaiveProduct(upper, sol), b), tol);
  Mat sol_right = b.transpose();
  a.template triangularView<Eigen::Lower>()
      .template solveInPlace<Eigen::OnTheRight>(sol_right);
  EXPECT_LE(RelError(NaiveProduct(sol_right, lower), Mat(b.transpose())), tol);

  // SYMM / HEMM and SYMV / HEMV (self-adjoint: symmetric or Hermitian).
  const Mat sa = a.template selfadjointView<Eigen::Lower>();
  EXPECT_LE(RelError(Mat(a.template selfadjointView<Eigen::Lower>() * b),
                     NaiveProduct(sa, b)),
            tol);
  EXPECT_LE(RelError(Vec(a.template selfadjointView<Eigen::Lower>() * x),
                     NaiveProduct(sa, Mat(x))),
            tol);

  // SYRK (rank update of a self-adjoint matrix).
  Mat c = Mat::Zero(n, n);
  c.template selfadjointView<Eigen::Lower>().rankUpdate(b);
  const Mat expected_c = NaiveProduct(b, Mat(b.adjoint()));
  EXPECT_LE(RelError(Mat(c.template triangularView<Eigen::Lower>()),
                     Mat(expected_c.template triangularView<Eigen::Lower>())),
            tol);
}

TYPED_TEST(EigenTest, Decompositions) {
  using T = TypeParam;
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  // Large enough for Eigen's blocked algorithms (which call TRSM and GEMM).
  const Eigen::Index n = 200, k = 5;
  const double tol = 1000 * n * Eps<T>();
  const Mat m = Mat::Random(n, n);
  const Mat spd = m * m.adjoint() + Mat::Identity(n, n) * T(n);
  const Mat rhs = Mat::Random(n, k);

  const Mat x_llt = spd.llt().solve(rhs);
  EXPECT_LE(RelError(NaiveProduct(spd, x_llt), rhs), tol);
  const Mat x_ldlt = spd.ldlt().solve(rhs);
  EXPECT_LE(RelError(NaiveProduct(spd, x_ldlt), rhs), tol);
  const Mat x_lu = m.partialPivLu().solve(rhs);
  EXPECT_LE(RelError(NaiveProduct(m, x_lu), rhs), tol);
}

}  // namespace
}  // namespace hwyblas_test
