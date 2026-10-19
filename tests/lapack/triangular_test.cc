// Tests of the triangular matrix routines: xTRTI2, xTRTRI, xTRTRS, xLATRS,
// xTRCON, xTRRFS, xLANTR and their LAPACKE functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class TriangularTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(TriangularTest, ScalarTypes);

// Whether element (i, j) is in the strict triangle uplo.
bool InStrictTriangle(char uplo, size_t i, size_t j) {
  return uplo == 'U' ? i < j : i > j;
}

// A random n x n triangular matrix (leading dimension lda): off-diagonal
// elements in [-offdiag, offdiag], diagonal elements of absolute value in
// [1, 2]. The elements outside the triangle are 7 (and the diagonal too,
// for diag = 'U'), which the routines must neither read nor change.
template <typename T>
std::vector<T> TriangularMatrix(Random<T>& rng, char uplo, char diag, size_t n,
                                size_t lda, double offdiag = 1.0) {
  using R = RealT<T>;
  std::vector<T> a(lda * std::max<size_t>(n, 1), T(7));
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      if (InStrictTriangle(uplo, i, j)) {
        At(a, i, j, lda) = rng.Next() * static_cast<R>(offdiag);
      } else if (i == j && diag == 'N') {
        const T x = rng.Next();
        const double ax = Abs(x);
        At(a, i, j, lda) = ax == 0 ? T(1) : x * static_cast<R>((1 + ax) / ax);
      }
    }
  }
  return a;
}

// The triangular matrix as stored in a (unit diagonal for diag = 'U').
template <typename T>
Dense<T> DenseTriangle(const std::vector<T>& a, char uplo, char diag, size_t n,
                       size_t lda) {
  Dense<T> d(n, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      if (i == j) {
        d(i, i) =
            diag == 'U' ? RefT<T>(1) : static_cast<RefT<T>>(At(a, i, i, lda));
      } else if (InStrictTriangle(uplo, i, j)) {
        d(i, j) = static_cast<RefT<T>>(At(a, i, j, lda));
      }
    }
  }
  return d;
}

// op(d) for op = 'N', 'T' or 'C'.
template <typename T>
Dense<T> Op(char op, const Dense<T>& d) {
  if (op == 'N') return d;
  Dense<T> t(d.cols, d.rows);
  for (size_t j = 0; j < d.cols; ++j) {
    for (size_t i = 0; i < d.rows; ++i) {
      t(j, i) = op == 'C' ? Conj(d(i, j)) : d(i, j);
    }
  }
  return t;
}

// The infinity norm (maximum row sum) of a dense matrix.
template <typename T>
double NormInf(const Dense<T>& a) {
  double norm = 0;
  for (size_t i = 0; i < a.rows; ++i) {
    double sum = 0;
    for (size_t j = 0; j < a.cols; ++j) sum += std::abs(a(i, j));
    norm = std::max(norm, sum);
  }
  return norm;
}

// The inverse of the triangular matrix d (by substitution, in the
// reference precision).
template <typename T>
Dense<T> InverseTriangle(const Dense<T>& d, char uplo) {
  const size_t n = d.rows;
  Dense<T> inv(n, n);
  for (size_t c = 0; c < n; ++c) {
    // Solve d * x = e_c.
    if (uplo == 'U') {
      for (size_t i = n; i-- > 0;) {
        RefT<T> sum = i == c ? RefT<T>(1) : RefT<T>(0);
        for (size_t k = i + 1; k < n; ++k) sum -= d(i, k) * inv(k, c);
        inv(i, c) = sum / d(i, i);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        RefT<T> sum = i == c ? RefT<T>(1) : RefT<T>(0);
        for (size_t k = 0; k < i; ++k) sum -= d(i, k) * inv(k, c);
        inv(i, c) = sum / d(i, i);
      }
    }
  }
  return inv;
}

// Checks that the elements of a outside the triangle uplo (and the diagonal
// for diag = 'U') and the rows lda > n are those of a0.
template <typename T>
void ExpectOutsideUnchanged(const std::vector<T>& a, const std::vector<T>& a0,
                            char uplo, char diag, size_t n, size_t lda) {
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < lda; ++i) {
      const bool inside =
          i < n && (InStrictTriangle(uplo, i, j) || (i == j && diag == 'N'));
      if (!inside) {
        ASSERT_EQ(At(a, i, j, lda), At(a0, i, j, lda)) << i << ", " << j;
      }
    }
  }
}

template <typename T>
hwyblas_int Trtri(bool unblocked, char uplo, char diag, hwyblas_int n,
                  std::vector<T>& a, hwyblas_int lda) {
  hwyblas_int info = -99;
  if (unblocked) {
    HWYBLAS_CALL(T, trti2, &uplo, &diag, &n, P(a.data()), &lda, &info);
  } else {
    HWYBLAS_CALL(T, trtri, &uplo, &diag, &n, P(a.data()), &lda, &info);
  }
  return info;
}

template <typename T>
hwyblas_int Trtrs(char uplo, char trans, char diag, hwyblas_int n,
                  hwyblas_int nrhs, const std::vector<T>& a, hwyblas_int lda,
                  std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, trtrs, &uplo, &trans, &diag, &n, &nrhs, P(a.data()), &lda,
               P(b.data()), &ldb, &info);
  return info;
}

template <typename T>
hwyblas_int Latrs(char uplo, char trans, char diag, char normin, hwyblas_int n,
                  const std::vector<T>& a, hwyblas_int lda, std::vector<T>& x,
                  RealT<T>* scale, std::vector<RealT<T>>& cnorm) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, latrs, &uplo, &trans, &diag, &normin, &n, P(a.data()), &lda,
               P(x.data()), scale, cnorm.data(), &info);
  return info;
}

// xTRCON with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Trcon(char norm, char uplo, char diag, hwyblas_int n,
                  const std::vector<T>& a, hwyblas_int lda, RealT<T>* rcond) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    HWYBLAS_CALL_COMPLEX(T, trcon, &norm, &uplo, &diag, &n, P(a.data()), &lda,
                         rcond, P(work.data()), rwork.data(), &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, trcon, &norm, &uplo, &diag, &n, a.data(), &lda, rcond,
                      work.data(), iwork.data(), &info);
  }
  return info;
}

// xTRRFS with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Trrfs(char uplo, char trans, char diag, hwyblas_int n,
                  hwyblas_int nrhs, const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& b, hwyblas_int ldb,
                  const std::vector<T>& x, hwyblas_int ldx,
                  std::vector<RealT<T>>& ferr, std::vector<RealT<T>>& berr) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    HWYBLAS_CALL_COMPLEX(T, trrfs, &uplo, &trans, &diag, &n, &nrhs, P(a.data()),
                         &lda, P(b.data()), &ldb, P(x.data()), &ldx,
                         ferr.data(), berr.data(), P(work.data()), rwork.data(),
                         &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, trrfs, &uplo, &trans, &diag, &n, &nrhs, a.data(), &lda,
                      b.data(), &ldb, x.data(), &ldx, ferr.data(), berr.data(),
                      work.data(), iwork.data(), &info);
  }
  return info;
}

template <typename T>
RealT<T> Lantr(char norm, char uplo, char diag, hwyblas_int m, hwyblas_int n,
               const std::vector<T>& a, hwyblas_int lda) {
  std::vector<RealT<T>> work(std::max<hwyblas_int>(1, m));
  RealT<T> value = 0;
  if constexpr (std::is_same_v<T, float>) {
    value = slantr_(&norm, &uplo, &diag, &m, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, double>) {
    value = dlantr_(&norm, &uplo, &diag, &m, &n, a.data(), &lda, work.data());
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    value =
        clantr_(&norm, &uplo, &diag, &m, &n, P(a.data()), &lda, work.data());
  } else {
    value =
        zlantr_(&norm, &uplo, &diag, &m, &n, P(a.data()), &lda, work.data());
  }
  return value;
}

std::vector<char> Diags() { return {'N', 'U'}; }

// A power of 2 that scales a right hand side so that the denominators of
// the backward errors of xGERFS and xTRRFS fall below SAFE2 (where SAFE1 is
// added).
template <typename T>
double NearUnderflow() {
  return std::is_same_v<RealT<T>, float> ? std::ldexp(1.0, -110)
                                         : std::ldexp(1.0, -982);
}

// The componentwise backward error of xGERFS and xTRRFS for the residual r
// and the denominators denom = (abs(op(A))*abs(X) + abs(B))(i), with
// SAFE1 added where denom(i) <= SAFE2.
template <typename T>
double BackwardError(const std::vector<double>& r,
                     const std::vector<double>& denom) {
  using R = RealT<T>;
  const double nz = static_cast<double>(r.size() + 1);
  const double safe1 = nz * std::numeric_limits<R>::min();
  const double safe2 = safe1 / LapackEps<T>();
  double s = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    s = std::max(s, denom[i] > safe2 ? r[i] / denom[i]
                                     : (r[i] + safe1) / (denom[i] + safe1));
  }
  return s;
}

// |re| + |im| (|x| for real x), the absolute value of CABS1.
template <typename X>
double Cabs1Ref(X x) {
  return std::abs(static_cast<double>(std::real(x))) +
         std::abs(static_cast<double>(std::imag(x)));
}

// ---------------------------------------------------------------------------
// Inverse

// The test ratio of LAPACK's xTRT01: ||A * inv(A) - I||_1 /
// (n ||A||_1 ||inv(A)||_1 eps), for the blocked and the unblocked routine
// and several block sizes.
TYPED_TEST(TriangularTest, Inverse) {
  using T = TypeParam;
  Random<T> rng(11);
  for (const bool unblocked : {false, true}) {
    for (const hwyblas_int nb : BlockSizes()) {
      if (unblocked && nb != 0) continue;
      const ScopedBlockSize block_size(nb);
      for (const char uplo : {'U', 'L'}) {
        for (const char diag : Diags()) {
          for (const hwyblas_int n : {0, 1, 2, 5, 16, 33, 64}) {
            const hwyblas_int lda = n + 2;
            SCOPED_TRACE(::testing::Message()
                         << (unblocked ? "trti2" : "trtri") << " " << uplo
                         << diag << " n=" << n << " nb=" << nb);
            const std::vector<T> a =
                TriangularMatrix(rng, uplo, diag, n, lda, 0.5);
            std::vector<T> inv = a;
            ASSERT_EQ(Trtri(unblocked, uplo, diag, n, inv, lda), 0);
            ExpectOutsideUnchanged(inv, a, uplo, diag, n, lda);
            if (n == 0) continue;
            const Dense<T> da = DenseTriangle(a, uplo, diag, n, lda);
            const Dense<T> di = DenseTriangle(inv, uplo, diag, n, lda);
            const double resid = Norm1(Sub(Mul(da, di), Identity<T>(n)));
            EXPECT_LT(resid / n / Norm1(da) / Norm1(di) / LapackEps<T>(),
                      kLapackThreshold);
          }
        }
      }
    }
  }
}

// A zero diagonal element makes xTRTRI return its index (the first one),
// with A unchanged; with a unit diagonal it is not referenced.
TYPED_TEST(TriangularTest, SingularInverse) {
  using T = TypeParam;
  Random<T> rng(13);
  for (const hwyblas_int nb : {0, 3}) {
    const ScopedBlockSize block_size(nb);
    for (const char uplo : {'U', 'L'}) {
      const hwyblas_int n = 9, lda = 9;
      std::vector<T> a = TriangularMatrix(rng, uplo, 'N', n, lda);
      At(a, 4, 4, lda) = T(0);
      At(a, 6, 6, lda) = T(0);
      std::vector<T> inv = a;
      EXPECT_EQ(Trtri(false, uplo, 'N', n, inv, lda), 5);
      EXPECT_EQ(inv, a);
      EXPECT_EQ(Trtri(false, uplo, 'U', n, inv, lda), 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Solution of triangular systems

// The test ratio of LAPACK's xTRT02: ||op(A) * X - B||_1 /
// (n ||op(A)||_1 ||X||_1 eps).
TYPED_TEST(TriangularTest, Solve) {
  using T = TypeParam;
  Random<T> rng(17);
  for (const char uplo : {'U', 'L'}) {
    for (const char trans : Ops<T>()) {
      for (const char diag : Diags()) {
        for (const hwyblas_int n : {0, 1, 3, 17, 40}) {
          for (const hwyblas_int nrhs : {0, 1, 4}) {
            SCOPED_TRACE(::testing::Message() << uplo << trans << diag << " n="
                                              << n << " nrhs=" << nrhs);
            const hwyblas_int lda = n + 1, ldb = n + 3;
            const std::vector<T> a = TriangularMatrix(rng, uplo, diag, n, lda);
            const std::vector<T> b =
                rng.Vector(static_cast<size_t>(ldb) * std::max(nrhs, 1));
            std::vector<T> x = b;
            ASSERT_EQ(Trtrs(uplo, trans, diag, n, nrhs, a, lda, x, ldb), 0);
            if (n == 0 || nrhs == 0) {
              EXPECT_EQ(x, b);
              continue;
            }
            const Dense<T> opa =
                Op(trans, DenseTriangle(a, uplo, diag, n, lda));
            const Dense<T> dx = ToDense(x, n, nrhs, ldb);
            const double resid =
                Norm1(Sub(Mul(opa, dx), ToDense(b, n, nrhs, ldb)));
            EXPECT_LT(resid / n / Norm1(opa) / Norm1(dx) / LapackEps<T>(),
                      kLapackThreshold);
          }
        }
      }
    }
  }
  // A zero diagonal element: INFO is its index, B is unchanged.
  const hwyblas_int n = 6;
  std::vector<T> a = TriangularMatrix(rng, 'L', 'N', n, n);
  At(a, 3, 3, n) = T(0);
  std::vector<T> b = rng.Vector(n), b0 = b;
  EXPECT_EQ(Trtrs('L', 'N', 'N', n, 1, a, n, b, n), 4);
  EXPECT_EQ(b, b0);
}

// Checks the solution x and scale of xLATRS for the right hand side b:
// op(A) * x = scale * b, with the test ratio of LAPACK's xTRT03 (computed
// on op(A) / max|a| and x / max|x|, which may be near overflow).
template <typename T>
void CheckLatrs(char uplo, char trans, char diag, const std::vector<T>& a,
                size_t n, size_t lda, const std::vector<T>& b,
                const std::vector<T>& x, RealT<T> scale) {
  const Dense<T> opa = Op(trans, DenseTriangle(a, uplo, diag, n, lda));
  double amax = 0, xmax = 0;
  for (const auto& e : opa.v) amax = std::max(amax, std::abs(e));
  for (const T& e : x) xmax = std::max(xmax, Abs(e));
  ASSERT_TRUE(std::isfinite(xmax));
  ASSERT_GT(xmax, 0);
  // SCALE <= 1, except where xLATRS scaled A by TSCAL < 1 (column norms
  // near overflow): then it returns SCALE / TSCAL.
  ASSERT_GE(scale, 0);
  ASSERT_TRUE(std::isfinite(scale));
  Dense<T> as(n, n);
  for (size_t i = 0; i < opa.v.size(); ++i) as.v[i] = opa.v[i] / amax;
  const double bscale = static_cast<double>(scale) / xmax / amax;
  double resid = 0;
  for (size_t i = 0; i < n; ++i) {
    RefT<T> sum = -bscale * static_cast<RefT<T>>(b[i]);
    for (size_t k = 0; k < n; ++k) {
      sum += as(i, k) * (static_cast<RefT<T>>(x[k]) / xmax);
    }
    resid = std::max(resid, std::abs(sum));
  }
  EXPECT_LT(resid / NormInf(as) / static_cast<double>(n) / LapackEps<T>(),
            kLapackThreshold)
      << "scale " << scale;
}

// xLATRS on well-scaled problems (solved by xTRSV, scale = 1) and on
// problems whose solution would overflow: large off-diagonal elements
// (growth), tiny diagonal elements, column norms that overflow (case 2 of
// the column scaling). The column norms given (NORMIN = 'Y') give the same
// results as computed ones.
TYPED_TEST(TriangularTest, Latrs) {
  using T = TypeParam;
  using R = RealT<T>;
  const R smlnum = std::numeric_limits<R>::min() /
                   std::numeric_limits<R>::epsilon();  // xLAMCH('S')/('P')
  const R bignum = 1 / smlnum;
  Random<T> rng(19);
  // kHugeColumn: one column (with its diagonal element) near overflow, so
  // that its norm overflows (case 2, with the in-line dot products), but
  // the solution is of order 1, where its residual shows errors in any
  // component.
  enum Kind { kRandom, kGrowth, kTiny, kHuge, kHugeColumn };
  for (const Kind kind : {kRandom, kGrowth, kTiny, kHuge, kHugeColumn}) {
    for (const char uplo : {'U', 'L'}) {
      for (const char trans : Ops<T>()) {
        for (const char diag : Diags()) {
          if ((kind == kTiny || kind == kHugeColumn) && diag == 'U') continue;
          const hwyblas_int n = kind == kHuge || kind == kHugeColumn ? 6 : 30;
          const hwyblas_int lda = n + 2;
          SCOPED_TRACE(::testing::Message()
                       << "kind " << kind << " " << uplo << trans << diag);
          std::vector<T> a = TriangularMatrix(rng, uplo, diag, n, lda, 0.5);
          std::vector<T> b = rng.Vector(n);
          // Complex (if T is) to exercise the conjugation of the in-line
          // dot products for A**H.
          const T huge = (IsComplex<T>::value ? MakeScalar<T>(0.6, 0.8) : T(1)) *
                         (std::numeric_limits<R>::max() / 4);
          if (kind == kHugeColumn) {
            const hwyblas_int jc = uplo == 'U' ? n - 1 : 0;
            for (hwyblas_int i = 0; i < n; ++i) {
              if (InStrictTriangle(uplo, i, jc) || i == jc) {
                At(a, i, jc, lda) = huge;
              }
            }
          } else if (kind != kRandom) {
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = 0; i < n; ++i) {
                if (!InStrictTriangle(uplo, i, j)) continue;
                if (kind == kGrowth) {
                  // -g everywhere: the solution grows like (1 + g)**n.
                  At(a, i, j, lda) =
                      T(-std::pow(bignum, R(2) / static_cast<R>(n)));
                } else if (kind == kTiny) {
                  At(a, i, j, lda) = T(0);
                } else {
                  At(a, i, j, lda) = huge;
                }
              }
              if (kind == kTiny) At(a, j, j, lda) = T(smlnum / 2);
              if (kind == kGrowth && diag == 'N') At(a, j, j, lda) = T(1);
            }
            for (T& e : b) e = T(1);
          }
          std::vector<T> x = b;
          std::vector<R> cnorm(n);
          R scale = -1;
          ASSERT_EQ(Latrs(uplo, trans, diag, 'N', n, a, lda, x, &scale, cnorm),
                    0);
          CheckLatrs(uplo, trans, diag, a, n, lda, b, x, scale);
          if (kind == kRandom) {
            EXPECT_EQ(scale, 1);
          } else if (kind != kHuge && kind != kHugeColumn) {
            EXPECT_LT(scale, 1);
          }
          // Given column norms. (Not for kHuge, whose column norms are
          // scaled by TSCAL and back, and so returned with rounding errors.)
          if (kind == kHuge || kind == kHugeColumn) continue;
          std::vector<T> x2 = b;
          R scale2 = -1;
          const std::vector<R> cnorm0 = cnorm;
          ASSERT_EQ(
              Latrs(uplo, trans, diag, 'Y', n, a, lda, x2, &scale2, cnorm), 0);
          EXPECT_EQ(scale2, scale);
          EXPECT_EQ(x2, x);
          EXPECT_EQ(cnorm, cnorm0);
        }
      }
    }
  }
}

// A zero diagonal element: scale = 0 and x is a solution of op(A) * x = 0
// with x(j) = 1 for the zero A(j, j) that was reached.
TYPED_TEST(TriangularTest, LatrsSingular) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(23);
  for (const char uplo : {'U', 'L'}) {
    for (const char trans : Ops<T>()) {
      const hwyblas_int n = 8, lda = 8;
      std::vector<T> a = TriangularMatrix(rng, uplo, 'N', n, lda, 0.5);
      At(a, 3, 3, lda) = T(0);
      std::vector<T> x = rng.Vector(n);
      std::vector<R> cnorm(n);
      R scale = -1;
      ASSERT_EQ(Latrs(uplo, trans, 'N', 'N', n, a, lda, x, &scale, cnorm), 0);
      EXPECT_EQ(scale, 0);
      EXPECT_EQ(x[3], T(1));
      const std::vector<T> zero(n, T(0));
      CheckLatrs(uplo, trans, 'N', a, n, lda, zero, x, R(0));
    }
  }
}

// ---------------------------------------------------------------------------
// Condition number and error bounds

// The estimate of xTRCON against the exact reciprocal condition number:
// the ratio of LAPACK's xGET06 (the larger of the two over the smaller).
TYPED_TEST(TriangularTest, ConditionNumber) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(29);
  for (const char norm : {'1', 'O', 'I'}) {
    for (const char uplo : {'U', 'L'}) {
      for (const char diag : Diags()) {
        for (const hwyblas_int n : {1, 2, 7, 25, 40}) {
          SCOPED_TRACE(::testing::Message()
                       << norm << uplo << diag << " n=" << n);
          const hwyblas_int lda = n + 1;
          const std::vector<T> a =
              TriangularMatrix(rng, uplo, diag, n, lda, 0.3);
          R rcond = -1;
          ASSERT_EQ(Trcon(norm, uplo, diag, n, a, lda, &rcond), 0);
          const Dense<T> d = DenseTriangle(a, uplo, diag, n, lda);
          const Dense<T> inv = InverseTriangle(d, uplo);
          const double rcondc = norm == 'I' ? 1 / (NormInf(d) * NormInf(inv))
                                            : 1 / (Norm1(d) * Norm1(inv));
          ASSERT_GT(rcond, 0);
          EXPECT_LT(std::max(rcond / rcondc, rcondc / rcond), kLapackThreshold)
              << rcond << " " << rcondc;
        }
      }
    }
  }
  // n = 0: RCOND = 1. A = 0: RCOND = 0.
  std::vector<T> a(4, T(0));
  R rcond = -1;
  EXPECT_EQ(Trcon('1', 'U', 'N', 0, a, 1, &rcond), 0);
  EXPECT_EQ(rcond, 1);
  EXPECT_EQ(Trcon('1', 'U', 'N', 2, a, 2, &rcond), 0);
  EXPECT_EQ(rcond, 0);
}

// The backward error (exactly as in xTRRFS: |re| + |im| for complex T, with
// SAFE1 added near underflow) of solutions perturbed by a known relative
// error, which the forward error bound must bound.
TYPED_TEST(TriangularTest, ErrorBounds) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(31);
  const double delta = std::is_same_v<R, float> ? 1e-3 : 1e-9;
  for (const char uplo : {'U', 'L'}) {
    for (const char trans : Ops<T>()) {
      for (const char diag : Diags()) {
        for (const hwyblas_int n : {1, 5, 30}) {
          for (const double bscale : {1.0, NearUnderflow<T>()}) {
            SCOPED_TRACE(::testing::Message() << uplo << trans << diag
                                              << " n=" << n << " " << bscale);
            const hwyblas_int nrhs = 3, lda = n + 1, ldb = n + 2, ldx = n + 3;
            const std::vector<T> a =
                TriangularMatrix(rng, uplo, diag, n, lda, 0.5);
            const Dense<T> opa =
                Op(trans, DenseTriangle(a, uplo, diag, n, lda));
            // The exact solution xt of op(A) * xt = B (in the reference
            // precision), and X = xt * (1 + delta * random).
            std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
            for (T& e : b) e *= static_cast<RealT<T>>(bscale);
            const Dense<T> inv =
                InverseTriangle(DenseTriangle(a, uplo, diag, n, lda), uplo);
            const Dense<T> xt = Mul(Op(trans, inv), ToDense(b, n, nrhs, ldb));
            std::vector<T> x(static_cast<size_t>(ldx) * nrhs);
            for (hwyblas_int j = 0; j < nrhs; ++j) {
              for (hwyblas_int i = 0; i < n; ++i) {
                const double p = 1 + delta * std::real(rng.Next());
                At(x, i, j, ldx) = static_cast<T>(xt(i, j) * p);
              }
            }
            std::vector<R> ferr(nrhs, -1), berr(nrhs, -1);
            ASSERT_EQ(Trrfs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx,
                            ferr, berr),
                      0);
            for (hwyblas_int j = 0; j < nrhs; ++j) {
              // The backward error, as xTRRFS computes it.
              double err = 0, xnorm = 0;
              std::vector<double> resid(n), denom(n);
              for (hwyblas_int i = 0; i < n; ++i) {
                RefT<T> r = -static_cast<RefT<T>>(At(b, i, j, ldb));
                denom[i] = Cabs1Ref(At(b, i, j, ldb));
                for (hwyblas_int k = 0; k < n; ++k) {
                  r += opa(i, k) * static_cast<RefT<T>>(At(x, k, j, ldx));
                  denom[i] += Cabs1Ref(opa(i, k)) * Cabs1Ref(At(x, k, j, ldx));
                }
                resid[i] = Cabs1Ref(r);
                err = std::max(err,
                               Cabs1Ref(static_cast<RefT<T>>(At(x, i, j, ldx)) -
                                        xt(i, j)));
                xnorm = std::max(xnorm, Cabs1Ref(At(x, i, j, ldx)));
              }
              const double berr_c = BackwardError<T>(resid, denom);
              EXPECT_NEAR(berr[j], berr_c,
                          0.01 * berr_c + 4 * n * LapackEps<T>());
              // The error bound holds (up to the rounding of the reference
              // solution) and is not absurd.
              EXPECT_GE(ferr[j], 0.3 * err / xnorm);
              EXPECT_LT(ferr[j], 1);
              if (bscale != 1) {
                // W >= SAFE1 in the bound.
                const double safe1 = (n + 1) * std::numeric_limits<R>::min();
                EXPECT_GE(ferr[j], 0.3 * safe1 / (NormInf(opa) * xnorm));
              }
            }
          }
        }
      }
    }
  }
  // n = 0 or nrhs = 0: FERR = BERR = 0.
  std::vector<T> a(1, T(1)), b(1), x(1);
  std::vector<R> ferr(2, -1), berr(2, -1);
  EXPECT_EQ(Trrfs('U', 'N', 'N', 0, 2, a, 1, b, 1, x, 1, ferr, berr), 0);
  EXPECT_EQ(ferr, (std::vector<R>{0, 0}));
  EXPECT_EQ(berr, (std::vector<R>{0, 0}));
}

// ---------------------------------------------------------------------------
// Norm

// xLANTR against the norms of the dense trapezoidal matrix, for all norms,
// shapes and diagonals; NaN in the trapezoid propagates, NaN outside it
// (and on a unit diagonal) is not read.
TYPED_TEST(TriangularTest, Lantr) {
  using T = TypeParam;
  Random<T> rng(37);
  struct Dims {
    hwyblas_int m, n;
  };
  for (const Dims s : {Dims{0, 3}, Dims{3, 0}, Dims{1, 1}, Dims{5, 5},
                       Dims{7, 4}, Dims{4, 7}}) {
    for (const char uplo : {'U', 'L'}) {
      for (const char diag : Diags()) {
        const hwyblas_int lda = s.m + 1;
        std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * s.n);
        // The dense trapezoid.
        Dense<T> d(s.m, s.n);
        for (hwyblas_int j = 0; j < s.n; ++j) {
          for (hwyblas_int i = 0; i < s.m; ++i) {
            if (i == j) {
              d(i, j) = diag == 'U' ? RefT<T>(1)
                                    : static_cast<RefT<T>>(At(a, i, j, lda));
            } else if (InStrictTriangle(uplo, i, j)) {
              d(i, j) = static_cast<RefT<T>>(At(a, i, j, lda));
            }
          }
        }
        double maxabs = 0, frob = 0;
        for (const auto& e : d.v) {
          maxabs = std::max(maxabs, std::abs(e));
          frob += std::norm(e);
        }
        frob = std::sqrt(frob);
        const double tol = 4 * (s.m + s.n + 1) * Eps<T>();
        SCOPED_TRACE(::testing::Message()
                     << uplo << diag << " " << s.m << "x" << s.n);
        EXPECT_NEAR(Lantr('M', uplo, diag, s.m, s.n, a, lda), maxabs,
                    tol * maxabs);
        EXPECT_NEAR(Lantr('1', uplo, diag, s.m, s.n, a, lda), Norm1(d),
                    tol * Norm1(d));
        EXPECT_NEAR(Lantr('O', uplo, diag, s.m, s.n, a, lda), Norm1(d),
                    tol * Norm1(d));
        EXPECT_NEAR(Lantr('I', uplo, diag, s.m, s.n, a, lda), NormInf(d),
                    tol * NormInf(d));
        EXPECT_NEAR(Lantr('F', uplo, diag, s.m, s.n, a, lda), frob, tol * frob);
        EXPECT_NEAR(Lantr('E', uplo, diag, s.m, s.n, a, lda), frob, tol * frob);
        if (s.m < 2 || s.n < 2) continue;
        // NaN outside the trapezoid, and on the unit diagonal.
        const hwyblas_int out_i = uplo == 'U' ? 1 : 0;
        const hwyblas_int out_j = uplo == 'U' ? 0 : 1;
        At(a, out_i, out_j, lda) = NaN<T>();
        if (diag == 'U') At(a, 0, 0, lda) = NaN<T>();
        for (const char norm : {'M', '1', 'I', 'F'}) {
          EXPECT_FALSE(std::isnan(Lantr(norm, uplo, diag, s.m, s.n, a, lda)))
              << norm;
        }
        // NaN in the trapezoid.
        At(a, out_j, out_i, lda) = NaN<T>();
        for (const char norm : {'M', '1', 'I', 'F'}) {
          EXPECT_TRUE(std::isnan(Lantr(norm, uplo, diag, s.m, s.n, a, lda)))
              << norm;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Errors

TYPED_TEST(TriangularTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16), x(16);
  std::vector<R> cnorm(4), ferr(4), berr(4);
  R scale = 0, rcond = 0;
  ErrorCapture errors;
  EXPECT_EQ(Trtri(true, 'X', 'N', 4, a, 4), -1);
  EXPECT_EQ(Trtri(true, 'U', 'X', 4, a, 4), -2);
  EXPECT_EQ(Trtri(true, 'U', 'N', -1, a, 4), -3);
  EXPECT_EQ(Trtri(true, 'U', 'N', 4, a, 3), -5);
  EXPECT_EQ(Trtri(false, 'X', 'N', 4, a, 4), -1);
  EXPECT_EQ(Trtri(false, 'U', 'X', 4, a, 4), -2);
  EXPECT_EQ(Trtri(false, 'U', 'N', -1, a, 4), -3);
  EXPECT_EQ(Trtri(false, 'U', 'N', 4, a, 3), -5);
  const std::string ti2 = Name<T>("TRTI2"), tri = Name<T>("TRTRI");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{ti2 + ":1", ti2 + ":2", ti2 + ":3",
                                      ti2 + ":5", tri + ":1", tri + ":2",
                                      tri + ":3", tri + ":5"}));
  EXPECT_EQ(Trtrs('X', 'N', 'N', 4, 1, a, 4, b, 4), -1);
  EXPECT_EQ(Trtrs('U', 'X', 'N', 4, 1, a, 4, b, 4), -2);
  EXPECT_EQ(Trtrs('U', 'N', 'X', 4, 1, a, 4, b, 4), -3);
  EXPECT_EQ(Trtrs('U', 'N', 'N', -1, 1, a, 4, b, 4), -4);
  EXPECT_EQ(Trtrs('U', 'N', 'N', 4, -1, a, 4, b, 4), -5);
  EXPECT_EQ(Trtrs('U', 'N', 'N', 4, 1, a, 3, b, 4), -7);
  EXPECT_EQ(Trtrs('U', 'N', 'N', 4, 1, a, 4, b, 3), -9);
  const std::string trs = Name<T>("TRTRS");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               trs + ":1", trs + ":2", trs + ":3", trs + ":4",
                               trs + ":5", trs + ":7", trs + ":9"}));
  EXPECT_EQ(Latrs('X', 'N', 'N', 'N', 4, a, 4, x, &scale, cnorm), -1);
  EXPECT_EQ(Latrs('U', 'X', 'N', 'N', 4, a, 4, x, &scale, cnorm), -2);
  EXPECT_EQ(Latrs('U', 'N', 'X', 'N', 4, a, 4, x, &scale, cnorm), -3);
  EXPECT_EQ(Latrs('U', 'N', 'N', 'X', 4, a, 4, x, &scale, cnorm), -4);
  EXPECT_EQ(Latrs('U', 'N', 'N', 'N', -1, a, 4, x, &scale, cnorm), -5);
  EXPECT_EQ(Latrs('U', 'N', 'N', 'N', 4, a, 3, x, &scale, cnorm), -7);
  const std::string lat = Name<T>("LATRS");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{lat + ":1", lat + ":2", lat + ":3",
                                      lat + ":4", lat + ":5", lat + ":7"}));
  EXPECT_EQ(Trcon('X', 'U', 'N', 4, a, 4, &rcond), -1);
  EXPECT_EQ(Trcon('1', 'X', 'N', 4, a, 4, &rcond), -2);
  EXPECT_EQ(Trcon('1', 'U', 'X', 4, a, 4, &rcond), -3);
  EXPECT_EQ(Trcon('1', 'U', 'N', -1, a, 4, &rcond), -4);
  EXPECT_EQ(Trcon('1', 'U', 'N', 4, a, 3, &rcond), -6);
  const std::string con = Name<T>("TRCON");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{con + ":1", con + ":2", con + ":3",
                                      con + ":4", con + ":6"}));
  EXPECT_EQ(Trrfs('X', 'N', 'N', 4, 1, a, 4, b, 4, x, 4, ferr, berr), -1);
  EXPECT_EQ(Trrfs('U', 'X', 'N', 4, 1, a, 4, b, 4, x, 4, ferr, berr), -2);
  EXPECT_EQ(Trrfs('U', 'N', 'X', 4, 1, a, 4, b, 4, x, 4, ferr, berr), -3);
  EXPECT_EQ(Trrfs('U', 'N', 'N', -1, 1, a, 4, b, 4, x, 4, ferr, berr), -4);
  EXPECT_EQ(Trrfs('U', 'N', 'N', 4, -1, a, 4, b, 4, x, 4, ferr, berr), -5);
  EXPECT_EQ(Trrfs('U', 'N', 'N', 4, 1, a, 3, b, 4, x, 4, ferr, berr), -7);
  EXPECT_EQ(Trrfs('U', 'N', 'N', 4, 1, a, 4, b, 3, x, 4, ferr, berr), -9);
  EXPECT_EQ(Trrfs('U', 'N', 'N', 4, 1, a, 4, b, 4, x, 3, ferr, berr), -11);
  const std::string rfs = Name<T>("TRRFS");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{rfs + ":1", rfs + ":2", rfs + ":3",
                                      rfs + ":4", rfs + ":5", rfs + ":7",
                                      rfs + ":9", rfs + ":11"}));
}

// ---------------------------------------------------------------------------
// LAPACKE

#define LAPACKE_CALL(T, NAME, ...)                                 \
  [&]() {                                                          \
    if constexpr (std::is_same_v<T, float>) {                      \
      return LAPACKE_s##NAME(__VA_ARGS__);                         \
    } else if constexpr (std::is_same_v<T, double>) {              \
      return LAPACKE_d##NAME(__VA_ARGS__);                         \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      return LAPACKE_c##NAME(__VA_ARGS__);                         \
    } else {                                                       \
      return LAPACKE_z##NAME(__VA_ARGS__);                         \
    }                                                              \
  }()

// The m x n column-major matrix col (leading dimension ldc) in row-major
// layout (leading dimension ldr).
template <typename T>
std::vector<T> RowMajor(const std::vector<T>& col, size_t m, size_t n,
                        size_t ldc, size_t ldr) {
  std::vector<T> row(ldr * std::max<size_t>(m, 1), T(5));
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) row[i * ldr + j] = At(col, i, j, ldc);
  }
  return row;
}

// The row-major functions give the results of the column-major ones (the
// same arithmetic on transposed copies; xLANTR computes the norm of the
// transpose directly). Elements outside the triangle are not changed.
TYPED_TEST(TriangularTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(41);
  const lapack_int n = 9, nrhs = 3, lda = 11, ldb = 12;
  for (const char uplo : {'U', 'L'}) {
    for (const char diag : Diags()) {
      SCOPED_TRACE(::testing::Message() << uplo << diag);
      const std::vector<T> a = TriangularMatrix(rng, uplo, diag, n, lda, 0.5);
      const std::vector<T> ar = RowMajor(a, n, n, lda, lda);
      // xTRTRI.
      std::vector<T> ic = a, ir = ar;
      ASSERT_EQ(LAPACKE_CALL(T, trtri, LAPACK_COL_MAJOR, uplo, diag, n,
                             ic.data(), lda),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, trtri, LAPACK_ROW_MAJOR, uplo, diag, n,
                             ir.data(), lda),
                0);
      for (lapack_int i = 0; i < n; ++i) {
        for (lapack_int j = 0; j < n; ++j) {
          ASSERT_EQ(ir[i * lda + j], At(ic, i, j, lda)) << i << ", " << j;
        }
      }
      // xTRTRS, xTRCON, xTRRFS.
      const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
      for (const char trans : Ops<T>()) {
        std::vector<T> xc = b, xr = RowMajor(b, n, nrhs, ldb, ldb);
        const std::vector<T> b_row = xr;
        ASSERT_EQ(LAPACKE_CALL(T, trtrs, LAPACK_COL_MAJOR, uplo, trans, diag, n,
                               nrhs, a.data(), lda, xc.data(), ldb),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, trtrs, LAPACK_ROW_MAJOR, uplo, trans, diag, n,
                               nrhs, ar.data(), lda, xr.data(), ldb),
                  0);
        EXPECT_EQ(xr, RowMajor(xc, n, nrhs, ldb, ldb));
        std::vector<R> ferr_c(nrhs), berr_c(nrhs), ferr_r(nrhs), berr_r(nrhs);
        ASSERT_EQ(LAPACKE_CALL(T, trrfs, LAPACK_COL_MAJOR, uplo, trans, diag, n,
                               nrhs, a.data(), lda, b.data(), ldb, xc.data(),
                               ldb, ferr_c.data(), berr_c.data()),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, trrfs, LAPACK_ROW_MAJOR, uplo, trans, diag, n,
                               nrhs, ar.data(), lda, b_row.data(), ldb,
                               xr.data(), ldb, ferr_r.data(), berr_r.data()),
                  0);
        EXPECT_EQ(ferr_r, ferr_c);
        EXPECT_EQ(berr_r, berr_c);
      }
      for (const char norm : {'1', 'I'}) {
        R rc = -1, rr = -2;
        ASSERT_EQ(LAPACKE_CALL(T, trcon, LAPACK_COL_MAJOR, norm, uplo, diag, n,
                               a.data(), lda, &rc),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, trcon, LAPACK_ROW_MAJOR, norm, uplo, diag, n,
                               ar.data(), lda, &rr),
                  0);
        EXPECT_EQ(rr, rc);
      }
      // xLANTR of a trapezoid.
      const lapack_int m = 7;
      for (const char norm : {'M', '1', 'I', 'F'}) {
        const R nc = LAPACKE_CALL(T, lantr, LAPACK_COL_MAJOR, norm, uplo, diag,
                                  m, n, a.data(), lda);
        const R nr =
            LAPACKE_CALL(T, lantr, LAPACK_ROW_MAJOR, norm, uplo, diag, m, n,
                         RowMajor(a, m, n, lda, lda).data(), lda);
        EXPECT_NEAR(nr, nc, 4 * n * Eps<T>() * nc) << norm;
      }
    }
  }
}

TYPED_TEST(TriangularTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16, T(1));
  R rcond = 0;
  std::vector<R> ferr(4), berr(4);
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LAPACKE_CALL(T, trtri, 0, 'U', 'N', 4, a.data(), 4), -1);
  EXPECT_EQ(LAPACKE_CALL(T, trtri, LAPACK_ROW_MAJOR, 'U', 'N', 4, a.data(), 3),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, trtrs, LAPACK_ROW_MAJOR, 'U', 'N', 'N', 4, 2,
                         a.data(), 4, b.data(), 1),
            -10);
  EXPECT_EQ(LAPACKE_CALL(T, trcon, LAPACK_ROW_MAJOR, '1', 'U', 'N', 4, a.data(),
                         3, &rcond),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, lantr, LAPACK_ROW_MAJOR, '1', 'U', 'N', 4, 4,
                         a.data(), 3),
            R(-8));
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("trtri") +
                "\nWrong parameter 6 in " + LapackeName<T>("trtri_work") +
                "\nWrong parameter 10 in " + LapackeName<T>("trtrs_work") +
                "\nWrong parameter 7 in " + LapackeName<T>("trcon_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("lantr_work") +
                "\n");
  // NaN checks: in the triangle only (not on a unit diagonal).
  a[1] = NaN<T>();  // (1, 0) column-major: lower triangle
  EXPECT_EQ(LAPACKE_CALL(T, trtri, LAPACK_COL_MAJOR, 'L', 'N', 4, a.data(), 4),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, trtri, LAPACK_COL_MAJOR, 'U', 'N', 4, a.data(), 4),
            0);
  EXPECT_EQ(LAPACKE_CALL(T, trcon, LAPACK_COL_MAJOR, '1', 'L', 'N', 4, a.data(),
                         4, &rcond),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, lantr, LAPACK_COL_MAJOR, 'M', 'U', 'N', 4, 4,
                         a.data(), 4),
            R(1));
  EXPECT_EQ(LAPACKE_CALL(T, lantr, LAPACK_COL_MAJOR, 'M', 'L', 'N', 4, 4,
                         a.data(), 4),
            R(-7));
  a[1] = T(0);
  a[0] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, trtrs, LAPACK_COL_MAJOR, 'U', 'N', 'U', 4, 1,
                         a.data(), 4, b.data(), 4),
            0);
  b[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, trtrs, LAPACK_COL_MAJOR, 'U', 'N', 'U', 4, 1,
                         a.data(), 4, b.data(), 4),
            -9);
  EXPECT_EQ(
      LAPACKE_CALL(T, trrfs, LAPACK_COL_MAJOR, 'U', 'N', 'U', 4, 1, a.data(), 4,
                   b.data(), 4, b.data(), 4, ferr.data(), berr.data()),
      -9);
}

}  // namespace
}  // namespace hwyblas_test
