// Tests of the QR factorization with column pivoting: xGEQP3 (with xLAQPS
// and xLAQP2) and its LAPACKE functions.
#include <algorithm>
#include <cmath>
#include <complex>
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
class Geqp3Test : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Geqp3Test, ScalarTypes);

// xGEQP3 with a workspace of `lwork` elements (-1: the optimal size from a
// workspace query).
template <typename T>
hwyblas_int Geqp3(hwyblas_int m, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda, std::vector<hwyblas_int>& jpvt,
                  std::vector<T>& tau, hwyblas_int lwork) {
  std::vector<RealT<T>> rwork(std::max<hwyblas_int>(1, 2 * n), NAN);
  hwyblas_int info = -99;
  if (lwork == -1) {
    std::vector<T> query(1);
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                           P(tau.data()), P(query.data()), &lwork, rwork.data(),
                           &info);
    } else {
      HWYBLAS_CALL_REAL(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                        P(tau.data()), P(query.data()), &lwork, &info);
    }
    if (info != 0) return info;
    lwork = static_cast<hwyblas_int>(std::real(query[0]));
  }
  std::vector<T> work(std::max<hwyblas_int>(1, lwork), NaN<T>());
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                         P(tau.data()), P(work.data()), &lwork, rwork.data(),
                         &info);
  } else {
    HWYBLAS_CALL_REAL(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                      P(tau.data()), P(work.data()), &lwork, &info);
  }
  return info;
}

// The minimum LWORK: 3 * n + 1 (real) or n + 1 (complex).
template <typename T>
hwyblas_int MinimumLwork(hwyblas_int n) {
  return IsComplex<T>::value ? n + 1 : 3 * n + 1;
}

// Q (m x m) of the reflectors that xGEQP3 left in af and tau.
template <typename T>
Dense<T> QOf(hwyblas_int m, hwyblas_int n, const std::vector<T>& af,
             hwyblas_int lda, const std::vector<T>& tau) {
  const hwyblas_int k = std::min(m, n);
  std::vector<T> q(static_cast<size_t>(m) * m, T(0));
  for (hwyblas_int j = 0; j < k; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) At(q, i, j, m) = At(af, i, j, lda);
  }
  hwyblas_int info = -99, lwork = std::max<hwyblas_int>(1, m) * 64;
  std::vector<T> work(lwork);
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, ungqr, &m, &m, &k, P(q.data()), &m, P(tau.data()),
                         P(work.data()), &lwork, &info);
  } else {
    HWYBLAS_CALL_REAL(T, orgqr, &m, &m, &k, P(q.data()), &m, P(tau.data()),
                      P(work.data()), &lwork, &info);
  }
  EXPECT_EQ(info, 0);
  return ToDense(q, m, m, m);
}

// Checks the factorization A * P = Q * R of `a` (m x n) in af, tau and
// jpvt: the residual, the orthogonality of Q, that jpvt is a permutation
// with the fixed columns (fixed[j] for column j) first in their order, and
// the pivoting: |R(k, k)| is at least the norm of R(k:, j) for each free
// column k and j > k (up to the accuracy of the downdated norms).
template <typename T>
void CheckFactorization(hwyblas_int m, hwyblas_int n, const std::vector<T>& a,
                        const std::vector<T>& af, hwyblas_int lda,
                        const std::vector<hwyblas_int>& jpvt,
                        const std::vector<T>& tau,
                        const std::vector<bool>& fixed) {
  // jpvt: a permutation, fixed columns first.
  std::vector<bool> seen(n, false);
  hwyblas_int nfxd = 0;
  for (bool f : fixed) nfxd += f ? 1 : 0;
  hwyblas_int last_fixed = 0;
  for (hwyblas_int j = 0; j < n; ++j) {
    const hwyblas_int col = jpvt[j];
    ASSERT_GE(col, 1);
    ASSERT_LE(col, n);
    ASSERT_FALSE(seen[col - 1]) << "column " << col << " twice";
    seen[col - 1] = true;
    ASSERT_EQ(fixed[col - 1], j < nfxd) << "jpvt(" << j + 1 << ") = " << col;
    if (j < nfxd) {
      ASSERT_GT(col, last_fixed);
      last_fixed = col;
    }
  }

  const hwyblas_int k = std::min(m, n);
  Dense<T> r(m, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i <= std::min(j, m - 1); ++i) {
      r(i, j) = static_cast<RefT<T>>(At(af, i, j, lda));
    }
  }
  Dense<T> ap(m, n);  // A * P
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      ap(i, j) = static_cast<RefT<T>>(At(a, i, jpvt[j] - 1, lda));
    }
  }
  const Dense<T> q = QOf(m, n, af, lda, tau);
  EXPECT_LT(ResidualRatio(ap, Mul(q, r), m), kLapackThreshold);
  EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);

  // Pivoting of the free columns (with slack for the norm downdates, which
  // are recomputed when they lose more than half of the digits, and for
  // columns at the rounding error level of rank-deficient matrices).
  const double slack = 1 - 64 * std::sqrt(LapackEps<T>());
  const double noise = k > 0 ? 100.0 * n * Eps<T>() * std::abs(r(0, 0)) : 0;
  for (hwyblas_int kk = nfxd; kk < k; ++kk) {
    const double rkk = std::abs(r(kk, kk));
    for (hwyblas_int j = kk + 1; j < n; ++j) {
      double norm2 = 0;
      for (hwyblas_int i = kk; i <= std::min(j, m - 1); ++i) {
        norm2 += std::norm(r(i, j));
      }
      ASSERT_GE(rkk, slack * std::sqrt(norm2) - noise)
          << "R(" << kk << ", " << kk << ") vs column " << j;
    }
  }
}

struct Shape {
  hwyblas_int m, n;
};
const Shape kShapes[] = {{1, 1},   {1, 5},   {5, 1},   {3, 3},
                         {4, 9},   {9, 4},   {20, 20}, {33, 17},
                         {17, 33}, {64, 70}, {90, 45}};

struct Blocking {
  hwyblas_int nb, nx;
};
const Blocking kBlockings[] = {{0, 0}, {1, 0}, {2, 0}, {3, 4}, {8, 0}, {8, 6}};

TYPED_TEST(Geqp3Test, Factorization) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(43);
    for (const Blocking& b : kBlockings) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const Shape& s : kShapes) {
        // Fixed columns: none, the first, some in the middle and the last.
        for (const int pattern : {0, 1, 2}) {
          // LWORK: optimal, minimum, minimum plus room for blocks of 3.
          for (const int lwork_kind : {0, 1, 2}) {
            const hwyblas_int lda = s.m + 1;
            SCOPED_TRACE(::testing::Message()
                         << "m=" << s.m << " n=" << s.n << " nb=" << b.nb
                         << " nx=" << b.nx << " fixed=" << pattern
                         << " lwork=" << lwork_kind);
            const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * s.n);
            std::vector<bool> fixed(s.n, false);
            if (pattern == 1) fixed[0] = true;
            if (pattern == 2) {
              for (hwyblas_int j = s.n / 3; j < s.n; j += 3) fixed[j] = true;
              fixed[s.n - 1] = true;
            }
            std::vector<hwyblas_int> jpvt(s.n);
            for (hwyblas_int j = 0; j < s.n; ++j) jpvt[j] = fixed[j] ? 7 : 0;
            const hwyblas_int minimum = MinimumLwork<T>(s.n);
            const hwyblas_int lwork = lwork_kind == 0 ? -1
                                      : lwork_kind == 1
                                          ? minimum
                                          : minimum + 3 * (s.n + 1);
            std::vector<T> af = a;
            std::vector<T> tau(std::max<hwyblas_int>(1, std::min(s.m, s.n)));
            ASSERT_EQ(Geqp3(s.m, s.n, af, lda, jpvt, tau, lwork), 0);
            CheckFactorization(s.m, s.n, a, af, lda, jpvt, tau, fixed);
            for (hwyblas_int j = 0; j < s.n; ++j) {
              ASSERT_EQ(At(af, s.m, j, lda), At(a, s.m, j, lda));
            }
          }
        }
      }
    }
  });
}

// A matrix of rank r: the trailing block R(r+1:, r+1:) is negligible, and
// the norm downdates of the nearly dependent columns are recomputed.
TYPED_TEST(Geqp3Test, RankDeficient) {
  using T = TypeParam;
  Random<T> rng(47);
  for (const Blocking& b : kBlockings) {
    const ScopedBlockSize block_size(b.nb, b.nx);
    for (const hwyblas_int r : {1, 3, 10}) {
      const hwyblas_int m = 30, n = 25;
      SCOPED_TRACE(::testing::Message()
                   << "rank=" << r << " nb=" << b.nb << " nx=" << b.nx);
      const Dense<T> x = ToDense(rng.Vector(m * r), m, r, m);
      const Dense<T> y = ToDense(rng.Vector(r * n), r, n, r);
      const Dense<T> xy = Mul(x, y);
      std::vector<T> a(m * n);
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < m; ++i) {
          // Columns of very different sizes.
          At(a, i, j, m) = static_cast<T>(xy(i, j) * std::pow(2.0, j % 7));
        }
      }
      std::vector<T> af = a;
      std::vector<hwyblas_int> jpvt(n, 0);
      std::vector<T> tau(n);
      ASSERT_EQ(Geqp3(m, n, af, m, jpvt, tau, -1), 0);
      CheckFactorization(m, n, a, af, m, jpvt, tau,
                         std::vector<bool>(n, false));
      double tail = 0, head = 0;
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i <= std::min(j, m - 1); ++i) {
          (i >= r ? tail : head) =
              std::max(i >= r ? tail : head, Abs(At(af, i, j, m)));
        }
      }
      EXPECT_LT(tail, 100 * n * Eps<T>() * head);
    }
  }
}

TYPED_TEST(Geqp3Test, WorkspaceQuery) {
  using T = TypeParam;
  const ScopedBlockSize block_size(8);
  hwyblas_int m = 20, n = 12, lda = 20, lwork = -1, info = -99;
  std::vector<T> a(lda * n), tau(n), work(1);
  std::vector<hwyblas_int> jpvt(n);
  std::vector<RealT<T>> rwork(2 * n);
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                         P(tau.data()), P(work.data()), &lwork, rwork.data(),
                         &info);
    EXPECT_EQ(std::real(work[0]), (n + 1) * 8);
  } else {
    HWYBLAS_CALL_REAL(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                      P(tau.data()), P(work.data()), &lwork, &info);
    EXPECT_EQ(work[0], 2 * n + (n + 1) * 8);
  }
  EXPECT_EQ(info, 0);
  // Empty: 1.
  n = 0;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                         P(tau.data()), P(work.data()), &lwork, rwork.data(),
                         &info);
  } else {
    HWYBLAS_CALL_REAL(T, geqp3, &m, &n, P(a.data()), &lda, jpvt.data(),
                      P(tau.data()), P(work.data()), &lwork, &info);
  }
  EXPECT_EQ(std::real(work[0]), 1);
}

TYPED_TEST(Geqp3Test, InvalidArguments) {
  using T = TypeParam;
  std::vector<T> a(64, T(1)), tau(8);
  std::vector<hwyblas_int> jpvt(8, 0);
  const std::string name = Name<T>("GEQP3");
  ErrorCapture errors;
  EXPECT_EQ(Geqp3<T>(-1, 4, a, 4, jpvt, tau, 64), -1);
  EXPECT_EQ(Geqp3<T>(4, -1, a, 4, jpvt, tau, 64), -2);
  EXPECT_EQ(Geqp3<T>(4, 4, a, 3, jpvt, tau, 64), -4);
  EXPECT_EQ(Geqp3<T>(4, 4, a, 4, jpvt, tau, MinimumLwork<T>(4) - 1), -8);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{name + ":1", name + ":2", name + ":4",
                                      name + ":8"}));
}

template <typename T>
lapack_int LapackeGeqp3(int layout, lapack_int m, lapack_int n, T* a,
                        lapack_int lda, lapack_int* jpvt, T* tau) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_sgeqp3(layout, m, n, a, lda, jpvt, tau);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dgeqp3(layout, m, n, a, lda, jpvt, tau);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return LAPACKE_cgeqp3(layout, m, n, a, lda, jpvt, tau);
  } else {
    return LAPACKE_zgeqp3(layout, m, n, a, lda, jpvt, tau);
  }
}

TYPED_TEST(Geqp3Test, Lapacke) {
  using T = TypeParam;
  Random<T> rng(53);
  const lapack_int m = 19, n = 13, ld = 21;
  const std::vector<T> a = rng.Vector(ld * ld);
  std::vector<T> col = a, row(a.size()), tau_col(n), tau_row(n);
  for (lapack_int j = 0; j < n; ++j) {
    for (lapack_int i = 0; i < m; ++i) row[i * ld + j] = At(a, i, j, ld);
  }
  std::vector<lapack_int> jpvt_col(n, 0), jpvt_row(n, 0);
  jpvt_col[4] = jpvt_row[4] = 1;
  ASSERT_EQ(LapackeGeqp3(LAPACK_COL_MAJOR, m, n, col.data(), ld,
                         jpvt_col.data(), tau_col.data()),
            0);
  ASSERT_EQ(LapackeGeqp3(LAPACK_ROW_MAJOR, m, n, row.data(), ld,
                         jpvt_row.data(), tau_row.data()),
            0);
  EXPECT_EQ(jpvt_col, jpvt_row);
  EXPECT_EQ(jpvt_col[0], 5);
  EXPECT_EQ(tau_col, tau_row);
  for (lapack_int j = 0; j < n; ++j) {
    for (lapack_int i = 0; i < m; ++i) {
      ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << i << ", " << j;
    }
  }

  ::testing::internal::CaptureStdout();
  EXPECT_EQ(
      LapackeGeqp3(5, m, n, col.data(), ld, jpvt_col.data(), tau_col.data()),
      -1);
  EXPECT_EQ(LapackeGeqp3(LAPACK_ROW_MAJOR, m, n, row.data(), n - 1,
                         jpvt_row.data(), tau_row.data()),
            -5);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("geqp3") +
                "\nWrong parameter 5 in " + LapackeName<T>("geqp3_work") +
                "\n");
  col[3] = NaN<T>();
  EXPECT_EQ(LapackeGeqp3(LAPACK_COL_MAJOR, m, n, col.data(), ld,
                         jpvt_col.data(), tau_col.data()),
            -4);
}

}  // namespace
}  // namespace hwyblas_test
