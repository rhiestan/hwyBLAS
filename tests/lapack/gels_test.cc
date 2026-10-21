// Tests of the least squares solvers: xGELS, xGELSS, xGELSD (xLALSD,
// xLALSA, xLALS0), xGELSY (xLAIC1), xGGLSE, xGGGLM, the factorizations they
// use (xGERQF / xGERQ2, xORMRQ / xUNMRQ, xORMR2 / xUNMR2, xTZRZF / xLATRZ,
// xORMRZ / xUNMRZ / xORMR3 / xUNMR3 and xLARZ*, xGGQRF, xGGRQF), and their
// LAPACKE functions.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/eigen_test_util.h"
#include "lapack/lapack_test_util.h"
#include "lapack/linear_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class GelsTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GelsTest, ScalarTypes);

// ---------------------------------------------------------------------------
// Test problems

// An m x r matrix with orthonormal columns (r <= m): Gram-Schmidt, twice, of
// a random matrix, in the reference precision.
template <typename T>
Dense<T> OrthonormalColumns(size_t m, size_t r, Random<T>& rng) {
  Dense<T> q(m, r);
  for (auto& x : q.v) x = static_cast<RefT<T>>(rng.Next());
  for (size_t j = 0; j < r; ++j) {
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t k = 0; k < j; ++k) {
        RefT<T> dot = 0;
        for (size_t i = 0; i < m; ++i) dot += Conj(q(i, k)) * q(i, j);
        for (size_t i = 0; i < m; ++i) q(i, j) -= dot * q(i, k);
      }
    }
    double norm = 0;
    for (size_t i = 0; i < m; ++i) norm += std::norm(q(i, j));
    norm = std::sqrt(norm);
    for (size_t i = 0; i < m; ++i) q(i, j) /= norm;
  }
  return q;
}

// The m x n matrix A = U * diag(sv) * V**H of rank r = sv.size() (U m x r,
// V n x r orthonormal) in a (leading dimension lda), and its pseudo-inverse
// V * diag(1/sv) * U**H.
template <typename T>
struct Problem {
  std::vector<T> a;
  Dense<T> pinv{0, 0};
  std::vector<double> sv;  // in decreasing order
};

template <typename T>
Problem<T> MakeProblem(size_t m, size_t n, size_t lda,
                       const std::vector<double>& sv, Random<T>& rng) {
  const size_t r = sv.size();
  const Dense<T> u = OrthonormalColumns(m, r, rng);
  const Dense<T> v = OrthonormalColumns(n, r, rng);
  Problem<T> p;
  p.sv = sv;
  p.a.assign(lda * std::max<size_t>(n, 1), NaN<T>());
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      RefT<T> x = 0;
      for (size_t k = 0; k < r; ++k) x += u(i, k) * sv[k] * Conj(v(j, k));
      At(p.a, i, j, lda) = static_cast<T>(x);
    }
  }
  p.pinv = Dense<T>(n, m);
  for (size_t j = 0; j < m; ++j) {
    for (size_t i = 0; i < n; ++i) {
      RefT<T> x = 0;
      for (size_t k = 0; k < r; ++k) x += v(i, k) / sv[k] * Conj(u(j, k));
      p.pinv(i, j) = x;
    }
  }
  return p;
}

// r singular values from scale down to scale/cond (geometrically).
inline std::vector<double> SingularValues(size_t r, double scale, double cond) {
  std::vector<double> sv(r);
  for (size_t k = 0; k < r; ++k) {
    sv[k] = r == 1 ? scale
                   : scale * std::pow(cond, -static_cast<double>(k) /
                                                static_cast<double>(r - 1));
  }
  return sv;
}

// ||x - xref||_1 / ||xref||_1 (or ||x||_1 if xref = 0) for the rows x nrhs
// solution in b (leading dimension ldb).
template <typename T>
double SolutionError(const std::vector<T>& b, size_t rows, size_t nrhs,
                     size_t ldb, const Dense<T>& xref) {
  const Dense<T> x = ToDense(b, rows, nrhs, ldb);
  const double diff = Norm1(Sub(x, xref));
  const double norm = Norm1(xref);
  return norm > 0 ? diff / norm : diff;
}

// The bound of the relative error of a least squares solution: n * cond**2
// * eps times a safety factor (the error grows with cond**2 for problems
// with a large residual).
template <typename T>
double LsTolerance(size_t m, size_t n, double cond) {
  return 30.0 * static_cast<double>(std::max<size_t>({m, n, 1})) * cond * cond *
         Eps<T>();
}

// op(A) for TRANS = 'N' or 'T' / 'C' as a dense matrix.
template <typename T>
Dense<T> OpDense(char trans, const std::vector<T>& a, size_t m, size_t n,
                 size_t lda) {
  const Dense<T> d = ToDense(a, m, n, lda);
  return trans == 'N' ? d : Adjoint(d);
}

// ---------------------------------------------------------------------------
// xGELS

// xGELS with the minimal workspace (minimal) or that of a workspace query,
// which WORK(1) gives again on exit.
template <typename T>
hwyblas_int Gels(char trans, hwyblas_int m, hwyblas_int n, hwyblas_int nrhs,
                 std::vector<T>& a, hwyblas_int lda, std::vector<T>& b,
                 hwyblas_int ldb, bool minimal) {
  const hwyblas_int mn = std::min(m, n);
  hwyblas_int lwork, info = -99;
  if (minimal) {
    lwork = std::max<hwyblas_int>(1, mn + std::max(mn, nrhs));
  } else {
    T wq = T(0);
    const hwyblas_int m1 = -1;
    HWYBLAS_CALL(T, gels, &trans, &m, &n, &nrhs, P(a.data()), &lda, P(b.data()),
                 &ldb, P(&wq), &m1, &info);
    if (info != 0) return info;
    lwork = QueryResult(wq);
  }
  std::vector<T> work(lwork, NaN<T>());
  HWYBLAS_CALL(T, gels, &trans, &m, &n, &nrhs, P(a.data()), &lda, P(b.data()),
               &ldb, P(work.data()), &lwork, &info);
  if (!minimal && info == 0) EXPECT_EQ(QueryResult(work[0]), lwork);
  return info;
}

// xGELS: the least squares solution of overdetermined systems (with the
// residual in the rows below it) and the minimum norm solution of
// underdetermined ones, with A and A**H, blocked and unblocked, for
// matrices that the driver scales and for A = 0 (zero solution).
TYPED_TEST(GelsTest, Gels) {
  using T = TypeParam;
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  Random<T> rng(401);
  // Below and above the scaling thresholds SMLNUM = SAFMIN / EPS and
  // 1 / SMLNUM (of the largest element).
  const double smlnum =
      static_cast<double>(std::numeric_limits<RealT<T>>::min()) /
      static_cast<double>(std::numeric_limits<RealT<T>>::epsilon());
  const hwyblas_int shapes[][2] = {{1, 1},  {5, 3},   {3, 5},   {20, 7},
                                   {7, 20}, {33, 33}, {65, 30}, {30, 65}};
  for (const hwyblas_int nb : {0, 3}) {
    const ScopedBlockSize block_size(nb, nb == 0 ? 0 : 2);
    for (const auto& shape : shapes) {
      const hwyblas_int m = shape[0], n = shape[1];
      const size_t mn = static_cast<size_t>(std::min(m, n));
      for (const double scale : {1.0, smlnum / 2, 16 / smlnum}) {
        const double cond = 10;
        const hwyblas_int lda = m + 1;
        const Problem<T> prob =
            MakeProblem<T>(m, n, lda, SingularValues(mn, scale, cond), rng);
        for (const char trans : {'N', ctrans}) {
          // op(A) is rows x cols.
          const size_t rows = trans == 'N' ? m : n;
          const size_t cols = trans == 'N' ? n : m;
          const Dense<T> pinv = trans == 'N' ? prob.pinv : Adjoint(prob.pinv);
          for (const hwyblas_int nrhs : {1, 3}) {
            const bool minimal = (m + nrhs) % 2 == 0;
            SCOPED_TRACE(::testing::Message()
                         << "nb=" << nb << " m=" << m << " n=" << n
                         << " scale=" << scale << " trans=" << trans
                         << " nrhs=" << nrhs << " minimal=" << minimal);
            const hwyblas_int ldb = std::max(m, n) + 2;
            std::vector<T> b(ldb * nrhs, NaN<T>());
            for (hwyblas_int j = 0; j < nrhs; ++j) {
              for (size_t i = 0; i < rows; ++i) {
                At(b, i, j, ldb) = rng.Next() * static_cast<RealT<T>>(scale);
              }
            }
            const Dense<T> b0 = ToDense(b, rows, nrhs, ldb);
            const Dense<T> xref = Mul(pinv, b0);
            std::vector<T> a = prob.a;
            ASSERT_EQ(Gels(trans, m, n, nrhs, a, lda, b, ldb, minimal), 0);
            EXPECT_LT(SolutionError(b, cols, nrhs, ldb, xref),
                      LsTolerance<T>(m, n, cond));
            if (rows > cols) {
              // The residual: rows cols+1 .. rows of B hold Q**H * r, as
              // LAPACK in the units of the scaled B (only the solution is
              // scaled back): B was multiplied by SMLNUM / BNRM or BIGNUM /
              // BNRM if its largest element BNRM was outside [SMLNUM,
              // BIGNUM].
              double bnrm = 0;
              for (const auto& x : b0.v) bnrm = std::max(bnrm, std::abs(x));
              const double bscale = bnrm < smlnum       ? smlnum / bnrm
                                    : bnrm > 1 / smlnum ? 1 / smlnum / bnrm
                                                        : 1.0;
              const Dense<T> r =
                  Sub(Mul(OpDense(trans, prob.a, m, n, lda), xref), b0);
              for (hwyblas_int j = 0; j < nrhs; ++j) {
                // (Divided by scale, which can be near the overflow or
                // underflow threshold.)
                double expected = 0, got = 0;
                for (size_t i = 0; i < rows; ++i) {
                  expected += std::norm(r(i, j) / scale);
                }
                for (size_t i = cols; i < rows; ++i) {
                  got +=
                      std::norm(static_cast<RefT<T>>(At(b, i, j, ldb)) / scale);
                }
                EXPECT_NEAR(std::sqrt(got), bscale * std::sqrt(expected),
                            LsTolerance<T>(m, n, cond) * (1 + bscale))
                    << j;
              }
            }
          }
        }
      }
    }
  }
  // A = 0: the zero solution (in max(M,N) rows), also without the
  // factorization; M = 0, N = 0 or NRHS = 0: B is set to zero.
  for (const auto& shape :
       {std::pair<hwyblas_int, hwyblas_int>{4, 3}, {3, 4}, {0, 3}, {3, 0}}) {
    const hwyblas_int m = shape.first, n = shape.second, nrhs = 2;
    const hwyblas_int lda = std::max<hwyblas_int>(1, m),
                      ldb = std::max<hwyblas_int>({1, m, n});
    std::vector<T> a(lda * std::max<hwyblas_int>(1, n), T(0));
    std::vector<T> b(ldb * nrhs, T(1));
    ASSERT_EQ(Gels('N', m, n, nrhs, a, lda, b, ldb, false), 0);
    for (hwyblas_int j = 0; j < nrhs; ++j) {
      for (hwyblas_int i = 0; i < std::max(m, n); ++i) {
        EXPECT_EQ(At(b, i, j, ldb), T(0)) << m << " " << n << " " << i;
      }
    }
  }
  // A not of full rank: INFO = the index of the zero diagonal element of R.
  {
    const hwyblas_int m = 4, n = 3, nrhs = 1, lda = 4, ldb = 4;
    std::vector<T> a = rng.Vector(lda * n);
    for (hwyblas_int i = 0; i < m; ++i) At(a, i, 1, lda) = T(0);
    std::vector<T> b = rng.Vector(ldb);
    EXPECT_EQ(Gels('N', m, n, nrhs, a, lda, b, ldb, false), 2);
  }
}

// ---------------------------------------------------------------------------
// xGELSS, xGELSD, xGELSY

// The rank revealing drivers: which = 0 xGELSS, 1 xGELSD, 2 xGELSY, with the
// minimal workspaces (minimal) or those of a workspace query. s gets the
// singular values (xGELSS, xGELSD), jpvt the pivots (xGELSY).
template <typename T>
hwyblas_int RankRevealing(int which, hwyblas_int m, hwyblas_int n,
                          hwyblas_int nrhs, std::vector<T>& a, hwyblas_int lda,
                          std::vector<T>& b, hwyblas_int ldb,
                          std::vector<RealT<T>>& s,
                          std::vector<hwyblas_int>& jpvt, RealT<T> rcond,
                          hwyblas_int* rank, bool minimal) {
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const hwyblas_int mn = std::min(m, n), mx = std::max(m, n);
  s.assign(std::max<hwyblas_int>(1, mn), NAN);
  hwyblas_int info = -99;
  const hwyblas_int m1 = -1;
  // Workspace queries.
  T wq = T(0);
  R rq = 0;
  hwyblas_int iq = 0;
  std::vector<R> rwork0(1);
  std::vector<hwyblas_int> iwork0(1);
  if (which == 0) {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gelss, &m, &n, &nrhs, P(a.data()), &lda,
                           P(b.data()), &ldb, s.data(), &rcond, rank, P(&wq),
                           &m1, rwork0.data(), &info);
    } else {
      HWYBLAS_CALL_REAL(T, gelss, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                        s.data(), &rcond, rank, &wq, &m1, &info);
    }
  } else if (which == 1) {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gelsd, &m, &n, &nrhs, P(a.data()), &lda,
                           P(b.data()), &ldb, s.data(), &rcond, rank, P(&wq),
                           &m1, &rq, &iq, &info);
    } else {
      HWYBLAS_CALL_REAL(T, gelsd, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                        s.data(), &rcond, rank, &wq, &m1, &iq, &info);
    }
  } else {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gelsy, &m, &n, &nrhs, P(a.data()), &lda,
                           P(b.data()), &ldb, jpvt.data(), &rcond, rank, P(&wq),
                           &m1, rwork0.data(), &info);
    } else {
      HWYBLAS_CALL_REAL(T, gelsy, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                        jpvt.data(), &rcond, rank, &wq, &m1, &info);
    }
  }
  if (info != 0) return info;
  hwyblas_int lwork = QueryResult(wq);
  if (minimal) {
    // The documented minima.
    if (which == 0) {
      lwork = kComplex ? 2 * mn + std::max({m, n, nrhs})
                       : 3 * mn + std::max({2 * mn, mx, nrhs});
    } else if (which == 2) {
      // The real routine checks only MN + max(2*MN, N+1, MN+NRHS), but
      // xGEQP3 needs the documented max(MN+3*N+1, 2*MN+NRHS).
      lwork = kComplex ? mn + std::max({2 * mn, n + 1, mn + nrhs})
                       : std::max(mn + 3 * n + 1, 2 * mn + nrhs);
    }
    lwork = std::max<hwyblas_int>(1, lwork);
  }
  const hwyblas_int lrwork =
      which == 1 ? std::max<hwyblas_int>(1, QueryResult(rq))
                 : std::max<hwyblas_int>(1, which == 0 ? 5 * mn : 2 * n);
  const hwyblas_int liwork = std::max<hwyblas_int>(1, iq);
  std::vector<T> work(lwork, NaN<T>());
  std::vector<R> rwork(lrwork, NAN);
  std::vector<hwyblas_int> iwork(liwork, -77);
  if (which == 0) {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gelss, &m, &n, &nrhs, P(a.data()), &lda,
                           P(b.data()), &ldb, s.data(), &rcond, rank,
                           P(work.data()), &lwork, rwork.data(), &info);
    } else {
      HWYBLAS_CALL_REAL(T, gelss, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                        s.data(), &rcond, rank, work.data(), &lwork, &info);
    }
  } else if (which == 1) {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gelsd, &m, &n, &nrhs, P(a.data()), &lda,
                           P(b.data()), &ldb, s.data(), &rcond, rank,
                           P(work.data()), &lwork, rwork.data(), iwork.data(),
                           &info);
    } else {
      HWYBLAS_CALL_REAL(T, gelsd, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                        s.data(), &rcond, rank, work.data(), &lwork,
                        iwork.data(), &info);
    }
  } else {
    if constexpr (IsComplex<T>::value) {
      HWYBLAS_CALL_COMPLEX(T, gelsy, &m, &n, &nrhs, P(a.data()), &lda,
                           P(b.data()), &ldb, jpvt.data(), &rcond, rank,
                           P(work.data()), &lwork, rwork.data(), &info);
    } else {
      HWYBLAS_CALL_REAL(T, gelsy, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                        jpvt.data(), &rcond, rank, work.data(), &lwork, &info);
    }
  }
  if (!minimal && info == 0) EXPECT_EQ(QueryResult(work[0]), lwork);
  return info;
}

const char* const kRankRevealing[] = {"gelss", "gelsd", "gelsy"};

// xGELSS, xGELSD and xGELSY: the minimum norm least squares solution
// (A**+ * B) and the effective rank of full-rank and rank-deficient
// matrices, of the scaled problems, of A = 0, with one and several
// right-hand sides and with the minimal workspaces (which take the chunked
// and the smaller paths of xGELSS); the singular values (xGELSS, xGELSD),
// the right singular vectors in A (xGELSS), and the columns that xGELSY
// keeps in front. xGELSD also with a small subproblem size, so that its
// divide and conquer (xLALSA, xLALS0) runs on small matrices.
TYPED_TEST(GelsTest, RankRevealing) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(409);
  // Below and above the scaling thresholds SMLNUM = SAFMIN / EPS and
  // 1 / SMLNUM (of the largest element).
  const double smlnum = static_cast<double>(std::numeric_limits<R>::min()) /
                        static_cast<double>(std::numeric_limits<R>::epsilon());
  const R rcond = std::sqrt(std::numeric_limits<R>::epsilon());
  const hwyblas_int shapes[][2] = {{1, 1},  {6, 4},   {4, 6},   {20, 8},
                                   {8, 20}, {40, 40}, {70, 30}, {30, 70}};
  for (const hwyblas_int smlsiz : {0, 4}) {
    const ScopedSubproblemSize subproblem(smlsiz);
    for (const auto& shape : shapes) {
      const hwyblas_int m = shape[0], n = shape[1];
      const size_t mn = static_cast<size_t>(std::min(m, n));
      for (const size_t rank : {mn, (mn + 1) / 2, size_t{1}}) {
        for (const double scale : {1.0, smlnum / 2, 16 / smlnum}) {
          if (scale != 1.0 && rank != mn) continue;
          const double cond = 10;
          const hwyblas_int lda = m + 1;
          const Problem<T> prob =
              MakeProblem<T>(m, n, lda, SingularValues(rank, scale, cond), rng);
          for (const hwyblas_int nrhs : {1, 3}) {
            const hwyblas_int ldb = std::max(m, n) + 1;
            std::vector<T> b0(ldb * nrhs, NaN<T>());
            for (hwyblas_int j = 0; j < nrhs; ++j) {
              for (hwyblas_int i = 0; i < m; ++i) {
                At(b0, i, j, ldb) = rng.Next() * static_cast<R>(scale);
              }
            }
            const Dense<T> xref = Mul(prob.pinv, ToDense(b0, m, nrhs, ldb));
            for (int which = 0; which < 3; ++which) {
              if (smlsiz != 0 && which != 1) continue;
              for (const bool minimal : {false, true}) {
                if (which == 1 && minimal) continue;
                SCOPED_TRACE(::testing::Message()
                             << kRankRevealing[which] << " smlsiz=" << smlsiz
                             << " m=" << m << " n=" << n << " rank=" << rank
                             << " scale=" << scale << " nrhs=" << nrhs
                             << " minimal=" << minimal);
                std::vector<T> a = prob.a, b = b0;
                std::vector<R> s;
                // xGELSY: every third column is kept in front.
                std::vector<hwyblas_int> jpvt(std::max<hwyblas_int>(1, n), 0);
                hwyblas_int nfix = 0;
                for (hwyblas_int j = 0; j < n; ++j) {
                  if (j % 3 == 2) {
                    jpvt[j] = 1;
                    ++nfix;
                  }
                }
                hwyblas_int r = -1;
                ASSERT_EQ(RankRevealing(which, m, n, nrhs, a, lda, b, ldb, s,
                                        jpvt, rcond, &r, minimal),
                          0);
                EXPECT_EQ(r, static_cast<hwyblas_int>(rank));
                EXPECT_LT(SolutionError(b, n, nrhs, ldb, xref),
                          LsTolerance<T>(m, n, cond));
                if (which < 2) {
                  for (size_t k = 0; k < mn; ++k) {
                    const double expected = k < rank ? prob.sv[k] : 0;
                    EXPECT_NEAR(s[k], expected, 30 * mn * Eps<T>() * prob.sv[0])
                        << k;
                  }
                }
                if (which == 0 && m >= n) {
                  // The first min(M,N) rows of A: the right singular
                  // vectors (for M < N not on the path with an LQ
                  // factorization, which leaves that in A).
                  const Dense<T> vt = Adjoint(ToDense(a, mn, n, lda));
                  EXPECT_LT(OrthogonalityRatio(vt), kLapackThreshold);
                }
                if (which == 2) {
                  std::vector<bool> fixed(n, false);
                  for (hwyblas_int j = 0; j < nfix; ++j) {
                    ASSERT_GE(jpvt[j], 1);
                    ASSERT_LE(jpvt[j], n);
                    EXPECT_EQ((jpvt[j] - 1) % 3, 2) << j;
                    fixed[jpvt[j] - 1] = true;
                  }
                  std::vector<bool> seen(n, false);
                  for (hwyblas_int j = 0; j < n; ++j) {
                    ASSERT_GE(jpvt[j], 1);
                    ASSERT_LE(jpvt[j], n);
                    EXPECT_FALSE(seen[jpvt[j] - 1]) << j;
                    seen[jpvt[j] - 1] = true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // xGELSD with RCOND >= 1: xLALSD then uses EPS (full rank).
  {
    const hwyblas_int m = 30, n = 20, nrhs = 1, lda = 30, ldb = 30;
    const Problem<T> prob =
        MakeProblem<T>(m, n, lda, SingularValues(n, 1.0, 10), rng);
    std::vector<T> a = prob.a, b = rng.Vector(ldb * nrhs);
    std::vector<R> s;
    std::vector<hwyblas_int> jpvt(n, 0);
    hwyblas_int r = -1;
    ASSERT_EQ(
        RankRevealing(1, m, n, nrhs, a, lda, b, ldb, s, jpvt, R(2), &r, false),
        0);
    EXPECT_EQ(r, n);
  }
  // A = 0 and the empty problems: rank 0 and a zero solution.
  for (int which = 0; which < 3; ++which) {
    for (const auto& shape :
         {std::pair<hwyblas_int, hwyblas_int>{4, 3}, {3, 4}, {0, 3}, {3, 0}}) {
      const hwyblas_int m = shape.first, n = shape.second, nrhs = 2;
      SCOPED_TRACE(::testing::Message()
                   << kRankRevealing[which] << " m=" << m << " n=" << n);
      const hwyblas_int lda = std::max<hwyblas_int>(1, m),
                        ldb = std::max<hwyblas_int>({1, m, n});
      std::vector<T> a(lda * std::max<hwyblas_int>(1, n), T(0));
      std::vector<T> b(ldb * nrhs, T(1));
      std::vector<R> s;
      std::vector<hwyblas_int> jpvt(std::max<hwyblas_int>(1, n), 0);
      hwyblas_int r = -1;
      ASSERT_EQ(RankRevealing(which, m, n, nrhs, a, lda, b, ldb, s, jpvt, rcond,
                              &r, false),
                0);
      EXPECT_EQ(r, 0);
      if (m > 0 && n > 0) {
        for (hwyblas_int j = 0; j < nrhs; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) {
            EXPECT_EQ(At(b, i, j, ldb), T(0)) << i;
          }
        }
      }
    }
  }
}

// xLALSD directly: an upper and a lower bidiagonal matrix (the latter is
// rotated to upper form, B with it, separately for one right-hand side),
// solved as least squares problems by divide and conquer (N > SMLSIZ) and
// by xLASDQ, against the solution of the dense problem.
TYPED_TEST(GelsTest, Lalsd) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<R> rrng(419);
  Random<T> rng(421);
  const hwyblas_int smlsiz = 4;
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int n : {1, 3, 10, 30}) {
      for (const hwyblas_int nrhs : {1, 2}) {
        SCOPED_TRACE(::testing::Message()
                     << "uplo=" << uplo << " n=" << n << " nrhs=" << nrhs);
        std::vector<R> d(n), e(std::max<hwyblas_int>(1, n - 1));
        for (auto& x : d) x = R(1) + rrng.Next() / 2;
        for (auto& x : e) x = rrng.Next() / 2;
        // The dense bidiagonal matrix and its inverse.
        Dense<T> bd(n, n);
        for (hwyblas_int i = 0; i < n; ++i) {
          bd(i, i) = d[i];
          if (i + 1 < n) {
            if (uplo == 'U') {
              bd(i, i + 1) = e[i];
            } else {
              bd(i + 1, i) = e[i];
            }
          }
        }
        const Dense<T> inv = Inverse(bd);
        const hwyblas_int ldb = n + 1;
        std::vector<T> b(ldb * nrhs, NaN<T>());
        for (hwyblas_int j = 0; j < nrhs; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) At(b, i, j, ldb) = rng.Next();
        }
        const Dense<T> xref = Mul(inv, ToDense(b, n, nrhs, ldb));
        const hwyblas_int nlvl = 8;
        const hwyblas_int lwork = 9 * n + 2 * n * smlsiz + 8 * n * nlvl +
                                  n * nrhs + (smlsiz + 1) * (smlsiz + 1);
        std::vector<T> work(std::max<hwyblas_int>(lwork, n * nrhs), NaN<T>());
        std::vector<R> rwork(lwork + 2 * n * nrhs + 64, NAN);
        std::vector<hwyblas_int> iwork(3 * n * nlvl + 11 * n + 8, -77);
        const R rcond = -1;
        hwyblas_int rank = -1, info = -99;
        if constexpr (IsComplex<T>::value) {
          HWYBLAS_CALL_COMPLEX(T, lalsd, &uplo, &smlsiz, &n, &nrhs, d.data(),
                               e.data(), P(b.data()), &ldb, &rcond, &rank,
                               P(work.data()), rwork.data(), iwork.data(),
                               &info);
        } else {
          HWYBLAS_CALL_REAL(T, lalsd, &uplo, &smlsiz, &n, &nrhs, d.data(),
                            e.data(), b.data(), &ldb, &rcond, &rank,
                            work.data(), iwork.data(), &info);
        }
        ASSERT_EQ(info, 0);
        EXPECT_EQ(rank, n);
        EXPECT_LT(SolutionError(b, n, nrhs, ldb, xref),
                  LsTolerance<T>(n, n, 4));
        for (hwyblas_int i = 1; i < n; ++i) EXPECT_GE(d[i - 1], d[i]) << i;
      }
    }
  }
}

// xLAIC1: one step of incremental condition estimation is exact for J = 1:
// SESTPR is the largest (JOB = 1) or smallest (JOB = 2) singular value of
// ( SEST 0; ALPHA GAMMA ) with ALPHA = conj(X(1)) * W(1), and S**2 + C**2
// = 1 (in absolute values), also in the special cases (SEST = 0, GAMMA or
// ALPHA negligible).
TYPED_TEST(GelsTest, Laic1) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(431);
  const R eps = std::numeric_limits<R>::epsilon();
  std::vector<std::pair<T, T>> cases;  // (w, gamma)
  cases.reserve(24);
  for (int k = 0; k < 20; ++k) cases.push_back({rng.Next(), rng.Next()});
  cases.push_back({T(0), T(R(0.5))});
  cases.push_back({T(R(0.5)), T(0)});
  cases.push_back({T(R(0.5)) * eps * eps, T(R(0.7))});
  cases.push_back({T(R(0.7)), T(R(0.5)) * eps * eps});
  for (const R sest : {R(1), R(0.25), R(0), eps * eps}) {
    for (const auto& [w, gamma] : cases) {
      for (const hwyblas_int job : {1, 2}) {
        SCOPED_TRACE(::testing::Message()
                     << "sest=" << sest << " w=" << w << " gamma=" << gamma
                     << " job=" << job);
        const T x = T(1);
        R sestpr = -1;
        T s = NaN<T>(), c = NaN<T>();
        const hwyblas_int j = 1;
        HWYBLAS_CALL(T, laic1, &job, &j, P(&x), &sest, P(&w), P(&gamma),
                     &sestpr, P(&s), P(&c));
        // The singular values of M = ( sest 0; alpha gamma ).
        const std::complex<double> alpha = static_cast<RefT<T>>(w);
        const std::complex<double> g = static_cast<RefT<T>>(gamma);
        const double ss = static_cast<double>(sest);
        const double fro = ss * ss + std::norm(alpha) + std::norm(g);
        const double det = std::abs(ss * g);
        const double disc = std::sqrt(std::max(0.0, fro * fro - 4 * det * det));
        const double smax = std::sqrt((fro + disc) / 2);
        const double smin = smax > 0 ? det / smax : 0;
        const double expected = job == 1 ? smax : smin;
        EXPECT_NEAR(sestpr, expected, 8 * Eps<T>() * (smax + 1e-30));
        EXPECT_NEAR(std::norm(static_cast<RefT<T>>(s)) +
                        std::norm(static_cast<RefT<T>>(c)),
                    1.0, 8 * Eps<T>());
        // The approximate singular vector y = ( s; c ) of L' = ( SEST 0;
        // ALPHA**H GAMMA**H ): || L' * y || = SESTPR.
        const std::complex<double> sv = static_cast<RefT<T>>(s);
        const std::complex<double> cv = static_cast<RefT<T>>(c);
        const double attained =
            std::sqrt(std::norm(ss * sv) +
                      std::norm(std::conj(alpha) * sv + std::conj(g) * cv));
        EXPECT_NEAR(attained, static_cast<double>(sestpr),
                    16 * Eps<T>() * (smax + 1e-30));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// RQ and RZ factorizations

// The explicit unitary matrix of an RQ (rz = false) or RZ (rz = true)
// factorization: its xUNMRQ / xUNMRZ (xORM*) applied to the identity.
template <typename T>
Dense<T> ExplicitQ(bool rz, hwyblas_int k, hwyblas_int nq, std::vector<T> a,
                   hwyblas_int lda, const std::vector<T>& tau) {
  std::vector<T> c(nq * nq, T(0));
  for (hwyblas_int i = 0; i < nq; ++i) At(c, i, i, nq) = T(1);
  const char side = 'L', trans = 'N';
  const hwyblas_int l = nq - k;
  hwyblas_int lwork = 64 * std::max<hwyblas_int>(1, nq) + 65 * 64, info = -99;
  std::vector<T> work(lwork);
  if (rz) {
    CALL_RC(T, ormrz, unmrz, &side, &trans, &nq, &nq, &k, &l, P(a.data()), &lda,
            P(tau.data()), P(c.data()), &nq, P(work.data()), &lwork, &info);
  } else {
    CALL_RC(T, ormrq, unmrq, &side, &trans, &nq, &nq, &k, P(a.data()), &lda,
            P(tau.data()), P(c.data()), &nq, P(work.data()), &lwork, &info);
  }
  EXPECT_EQ(info, 0);
  return ToDense(c, nq, nq, nq);
}

// xGERQF and xGERQ2: A = R * Q with orthonormal Q (applied by xORMRQ /
// xUNMRQ), blocked and unblocked; and xORMRQ / xUNMRQ, xORMR2 / xUNMR2
// for all sides and transpositions, against the explicit Q.
TYPED_TEST(GelsTest, Rq) {
  using T = TypeParam;
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  Random<T> rng(433);
  const hwyblas_int shapes[][2] = {{1, 1},   {3, 5},   {5, 3},  {12, 12},
                                   {20, 45}, {45, 20}, {70, 70}};
  for (const hwyblas_int nb : {0, 1, 3}) {
    const ScopedBlockSize block_size(nb, nb == 0 ? 0 : 2);
    for (const auto& shape : shapes) {
      const hwyblas_int m = shape[0], n = shape[1];
      const hwyblas_int k = std::min(m, n), lda = m + 2;
      for (const bool unblocked : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "nb=" << nb << " m=" << m << " n="
                                          << n << " unblocked=" << unblocked);
        const std::vector<T> a0 = rng.Vector(lda * n);
        std::vector<T> a = a0, tau(std::max<hwyblas_int>(1, k), NaN<T>());
        hwyblas_int info = -99;
        if (unblocked) {
          std::vector<T> work(std::max<hwyblas_int>(1, m), NaN<T>());
          HWYBLAS_CALL(T, gerq2, &m, &n, P(a.data()), &lda, P(tau.data()),
                       P(work.data()), &info);
        } else {
          hwyblas_int lwork = std::max<hwyblas_int>(1, m) * 4;
          std::vector<T> work(lwork, NaN<T>());
          HWYBLAS_CALL(T, gerqf, &m, &n, P(a.data()), &lda, P(tau.data()),
                       P(work.data()), &lwork, &info);
        }
        ASSERT_EQ(info, 0);
        // R: the elements on and above the (n-m)-th superdiagonal.
        Dense<T> r(m, n);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < m; ++i) {
            if (j - i >= n - m) r(i, j) = At(a, i, j, lda);
          }
        }
        // Q (n x n) from the last k rows of A.
        std::vector<T> v(k * std::max<hwyblas_int>(1, n));
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < k; ++i) {
            At(v, i, j, k) = At(a, m - k + i, j, lda);
          }
        }
        const Dense<T> q = ExplicitQ(false, k, n, v, k, tau);
        EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);
        EXPECT_LT(ResidualRatio(ToDense(a0, m, n, lda), Mul(r, q), n),
                  kLapackThreshold);
        if (unblocked || m > 20) continue;
        // xORMRQ / xUNMRQ and xORMR2 / xUNMR2 for all options.
        for (const char side : {'L', 'R'}) {
          for (const char trans : {'N', ctrans}) {
            for (const bool r2 : {false, true}) {
              const hwyblas_int cm = side == 'L' ? n : 4,
                                cn = side == 'L' ? 4 : n;
              std::vector<T> c = rng.Vector(cm * cn);
              const Dense<T> c0 = ToDense(c, cm, cn, cm);
              std::vector<T> vv = v;
              hwyblas_int inf = -99;
              if (r2) {
                std::vector<T> work(std::max(cm, cn));
                CALL_RC(T, ormr2, unmr2, &side, &trans, &cm, &cn, &k,
                        P(vv.data()), &k, P(tau.data()), P(c.data()), &cm,
                        P(work.data()), &inf);
              } else {
                hwyblas_int lwork = std::max(cm, cn);
                std::vector<T> work(lwork);
                CALL_RC(T, ormrq, unmrq, &side, &trans, &cm, &cn, &k,
                        P(vv.data()), &k, P(tau.data()), P(c.data()), &cm,
                        P(work.data()), &lwork, &inf);
              }
              ASSERT_EQ(inf, 0);
              EXPECT_EQ(vv, v);  // restored by the complex routines
              const Dense<T> op = trans == 'N' ? q : Adjoint(q);
              const Dense<T> expected = side == 'L' ? Mul(op, c0) : Mul(c0, op);
              EXPECT_LT(ResidualRatio(expected, ToDense(c, cm, cn, cm), n),
                        kLapackThreshold)
                  << side << trans << r2;
            }
          }
        }
      }
    }
  }
}

// xTZRZF and xLATRZ (unblocked, nb = 1): A = ( R 0 ) * Z for upper
// trapezoidal A (M <= N), with orthonormal Z (applied by xORMRZ /
// xUNMRZ); xORMRZ / xUNMRZ and xORMR3 / xUNMR3 for all sides and
// transpositions against the explicit Z.
TYPED_TEST(GelsTest, Rz) {
  using T = TypeParam;
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  Random<T> rng(439);
  const hwyblas_int shapes[][2] = {{0, 3},   {1, 1},   {3, 3},  {3, 7},
                                   {12, 30}, {40, 41}, {70, 90}};
  for (const hwyblas_int nb : {0, 1, 3}) {
    const ScopedBlockSize block_size(nb, nb == 0 ? 0 : 2);
    for (const auto& shape : shapes) {
      const hwyblas_int m = shape[0], n = shape[1];
      const hwyblas_int lda = std::max<hwyblas_int>(1, m) + 1;
      SCOPED_TRACE(::testing::Message()
                   << "nb=" << nb << " m=" << m << " n=" << n);
      std::vector<T> a0 = rng.Vector(lda * n);
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = j + 1; i < m; ++i) At(a0, i, j, lda) = T(0);
      }
      std::vector<T> a = a0, tau(std::max<hwyblas_int>(1, m), NaN<T>());
      // A workspace for the blocked code (LWORK = M would make it
      // unblocked).
      hwyblas_int lwork = 64 * std::max<hwyblas_int>(1, m), info = -99;
      std::vector<T> work(lwork, NaN<T>());
      HWYBLAS_CALL(T, tzrzf, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &lwork, &info);
      ASSERT_EQ(info, 0);
      if (m == 0) continue;
      Dense<T> r(m, n);
      for (hwyblas_int j = 0; j < m; ++j) {
        for (hwyblas_int i = 0; i <= j; ++i) r(i, j) = At(a, i, j, lda);
      }
      const Dense<T> z = ExplicitQ(true, m, n, a, lda, tau);
      EXPECT_LT(OrthogonalityRatio(z), kLapackThreshold);
      EXPECT_LT(ResidualRatio(ToDense(a0, m, n, lda), Mul(r, z), n),
                kLapackThreshold);
      if (m > 12 || m == n) continue;
      for (const char side : {'L', 'R'}) {
        for (const char trans : {'N', ctrans}) {
          for (const bool r3 : {false, true}) {
            const hwyblas_int cm = side == 'L' ? n : 3,
                              cn = side == 'L' ? 3 : n;
            const hwyblas_int l = n - m;
            std::vector<T> c = rng.Vector(cm * cn);
            const Dense<T> c0 = ToDense(c, cm, cn, cm);
            std::vector<T> aa = a;
            hwyblas_int inf = -99;
            if (r3) {
              std::vector<T> w(std::max(cm, cn));
              CALL_RC(T, ormr3, unmr3, &side, &trans, &cm, &cn, &m, &l,
                      P(aa.data()), &lda, P(tau.data()), P(c.data()), &cm,
                      P(w.data()), &inf);
            } else {
              hwyblas_int lw = std::max(cm, cn);
              std::vector<T> w(lw);
              CALL_RC(T, ormrz, unmrz, &side, &trans, &cm, &cn, &m, &l,
                      P(aa.data()), &lda, P(tau.data()), P(c.data()), &cm,
                      P(w.data()), &lw, &inf);
            }
            ASSERT_EQ(inf, 0);
            EXPECT_EQ(aa, a);  // restored by the complex routines
            const Dense<T> op = trans == 'N' ? z : Adjoint(z);
            const Dense<T> expected = side == 'L' ? Mul(op, c0) : Mul(c0, op);
            EXPECT_LT(ResidualRatio(expected, ToDense(c, cm, cn, cm), n),
                      kLapackThreshold)
                << side << trans << r3;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Generalized least squares

// xGGQRF and xGGRQF: A = Q * R, B = Q * T * Z (GQR) and A = R * Q, B = Z *
// T * Q (GRQ), checked by applying the factors back with xUNMQR / xUNMRQ.
TYPED_TEST(GelsTest, GeneralizedFactorizations) {
  using T = TypeParam;
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  Random<T> rng(443);
  for (const hwyblas_int nb : {0, 3}) {
    const ScopedBlockSize block_size(nb, nb == 0 ? 0 : 2);
    for (const auto& d : {std::array<hwyblas_int, 3>{5, 3, 7},
                          {6, 6, 2},
                          {3, 8, 5},
                          {20, 9, 25}}) {
      // GQR: A is n x m, B n x p with (n, m, p) = d.
      {
        const hwyblas_int n = d[0], m = d[1], p = d[2];
        SCOPED_TRACE(::testing::Message() << "ggqrf nb=" << nb << " n=" << n
                                          << " m=" << m << " p=" << p);
        const hwyblas_int lda = n + 1, ldb = n + 2;
        const std::vector<T> a0 = rng.Vector(lda * m), b0 = rng.Vector(ldb * p);
        std::vector<T> a = a0, b = b0;
        std::vector<T> taua(std::max<hwyblas_int>(1, std::min(n, m)));
        std::vector<T> taub(std::max<hwyblas_int>(1, std::min(n, p)));
        hwyblas_int lwork = 64 * (n + m + p), info = -99;
        std::vector<T> work(lwork);
        HWYBLAS_CALL(T, ggqrf, &n, &m, &p, P(a.data()), &lda, P(taua.data()),
                     P(b.data()), &ldb, P(taub.data()), P(work.data()), &lwork,
                     &info);
        ASSERT_EQ(info, 0);
        // Q**H * A = R (upper trapezoidal).
        std::vector<T> qa = a0;
        const hwyblas_int ka = std::min(n, m);
        const char side = 'L';
        CALL_RC(T, ormqr, unmqr, &side, &ctrans, &n, &m, &ka, P(a.data()), &lda,
                P(taua.data()), P(qa.data()), &lda, P(work.data()), &lwork,
                &info);
        ASSERT_EQ(info, 0);
        Dense<T> rr(n, m);
        for (hwyblas_int j = 0; j < m; ++j) {
          for (hwyblas_int i = 0; i <= std::min(j, n - 1); ++i) {
            rr(i, j) = At(a, i, j, lda);
          }
        }
        EXPECT_LT(ResidualRatio(ToDense(qa, n, m, lda), rr, n),
                  kLapackThreshold);
        // Q**H * B * Z**H = T (upper trapezoidal, RQ form).
        std::vector<T> qb = b0;
        CALL_RC(T, ormqr, unmqr, &side, &ctrans, &n, &p, &ka, P(a.data()), &lda,
                P(taua.data()), P(qb.data()), &ldb, P(work.data()), &lwork,
                &info);
        ASSERT_EQ(info, 0);
        const hwyblas_int kb = std::min(n, p);
        const char right = 'R';
        CALL_RC(T, ormrq, unmrq, &right, &ctrans, &n, &p, &kb, P(&b[n - kb]),
                &ldb, P(taub.data()), P(qb.data()), &ldb, P(work.data()),
                &lwork, &info);
        ASSERT_EQ(info, 0);
        Dense<T> tt(n, p);
        for (hwyblas_int j = 0; j < p; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) {
            if (j - i >= p - n) tt(i, j) = At(b, i, j, ldb);
          }
        }
        EXPECT_LT(ResidualRatio(ToDense(qb, n, p, ldb), tt, n),
                  kLapackThreshold);
      }
      // GRQ: A is m x n, B p x n with (m, p, n) = d.
      {
        const hwyblas_int m = d[0], p = d[1], n = d[2];
        SCOPED_TRACE(::testing::Message() << "ggrqf nb=" << nb << " m=" << m
                                          << " p=" << p << " n=" << n);
        const hwyblas_int lda = m + 1, ldb = p + 2;
        const std::vector<T> a0 = rng.Vector(lda * n), b0 = rng.Vector(ldb * n);
        std::vector<T> a = a0, b = b0;
        std::vector<T> taua(std::max<hwyblas_int>(1, std::min(m, n)));
        std::vector<T> taub(std::max<hwyblas_int>(1, std::min(p, n)));
        hwyblas_int lwork = 64 * (n + m + p), info = -99;
        std::vector<T> work(lwork);
        HWYBLAS_CALL(T, ggrqf, &m, &p, &n, P(a.data()), &lda, P(taua.data()),
                     P(b.data()), &ldb, P(taub.data()), P(work.data()), &lwork,
                     &info);
        ASSERT_EQ(info, 0);
        // A * Q**H = R (RQ form).
        std::vector<T> aq = a0;
        const hwyblas_int ka = std::min(m, n);
        const char right = 'R', left = 'L';
        CALL_RC(T, ormrq, unmrq, &right, &ctrans, &m, &n, &ka, P(&a[m - ka]),
                &lda, P(taua.data()), P(aq.data()), &lda, P(work.data()),
                &lwork, &info);
        ASSERT_EQ(info, 0);
        Dense<T> rr(m, n);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < m; ++i) {
            if (j - i >= n - m) rr(i, j) = At(a, i, j, lda);
          }
        }
        EXPECT_LT(ResidualRatio(ToDense(aq, m, n, lda), rr, n),
                  kLapackThreshold);
        // Z**H * B * Q**H = T (upper trapezoidal).
        std::vector<T> zb = b0;
        CALL_RC(T, ormrq, unmrq, &right, &ctrans, &p, &n, &ka, P(&a[m - ka]),
                &lda, P(taua.data()), P(zb.data()), &ldb, P(work.data()),
                &lwork, &info);
        ASSERT_EQ(info, 0);
        const hwyblas_int kb = std::min(p, n);
        CALL_RC(T, ormqr, unmqr, &left, &ctrans, &p, &n, &kb, P(b.data()), &ldb,
                P(taub.data()), P(zb.data()), &ldb, P(work.data()), &lwork,
                &info);
        ASSERT_EQ(info, 0);
        Dense<T> tt(p, n);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i <= std::min(j, p - 1); ++i) {
            tt(i, j) = At(b, i, j, ldb);
          }
        }
        EXPECT_LT(ResidualRatio(ToDense(zb, p, n, ldb), tt, n),
                  kLapackThreshold);
      }
    }
  }
}

// The solution of the m x m system in the reference precision.
template <typename T>
Dense<T> Solve(const Dense<T>& a, const Dense<T>& b) {
  return Mul(Inverse(a), b);
}

// xGGLSE: x minimizes || c - A*x || subject to B*x = d, for B = S *
// Q1**H (S p x p, Q = [Q1 Q2] n x n orthonormal), whose null space is
// range(Q2): x = Q1 * S**-1 * d + Q2 * z, with z the least squares
// solution of (A*Q2) z = c - A*Q1*S**-1*d. Also p = 0 (an unconstrained
// problem), p = n (x determined by the constraint), M < N, and the
// singular cases (INFO = 1: B is rank deficient, INFO = 2: ( A; B ) is).
TYPED_TEST(GelsTest, Gglse) {
  using T = TypeParam;
  Random<T> rng(449);
  for (const auto& d : {std::array<hwyblas_int, 3>{6, 4, 2},
                        {8, 5, 0},
                        {4, 4, 4},
                        {3, 6, 4},
                        {30, 20, 7},
                        {40, 40, 15}}) {
    const hwyblas_int m = d[0], n = d[1], p = d[2];
    for (const bool minimal : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n << " p=" << p
                                        << " minimal=" << minimal);
      const Dense<T> q = OrthonormalColumns<T>(n, n, rng);
      Dense<T> q1(n, p), q2(n, n - p);
      for (hwyblas_int i = 0; i < n; ++i) {
        for (hwyblas_int j = 0; j < p; ++j) q1(i, j) = q(i, j);
        for (hwyblas_int j = p; j < n; ++j) q2(i, j - p) = q(i, j);
      }
      Dense<T> sm(p, p);
      for (auto& x : sm.v) x = static_cast<RefT<T>>(rng.Next());
      for (hwyblas_int i = 0; i < p; ++i) sm(i, i) += 2.0;
      const Dense<T> bd = Mul(sm, Adjoint(q1));
      const hwyblas_int lda = m + 1, ldb = std::max<hwyblas_int>(1, p) + 1;
      std::vector<T> a = rng.Vector(lda * n), b(ldb * n, NaN<T>());
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < p; ++i) {
          At(b, i, j, ldb) = static_cast<T>(bd(i, j));
        }
      }
      std::vector<T> c = rng.Vector(m),
                     dd = rng.Vector(std::max<hwyblas_int>(1, p));
      // The reference solution (with B as rounded).
      const Dense<T> ad = ToDense(a, m, n, lda);
      const Dense<T> bdr = ToDense(b, p, n, ldb);
      // x1 = Q1 * S**-1 * d solves B*x = d for B = S*Q1**H; with the
      // rounded B, use x1 = B**+ d = B**H (B B**H)**-1 d instead.
      Dense<T> dv(p, 1), cv(m, 1);
      for (hwyblas_int i = 0; i < p; ++i)
        dv(i, 0) = static_cast<RefT<T>>(dd[i]);
      for (hwyblas_int i = 0; i < m; ++i) cv(i, 0) = static_cast<RefT<T>>(c[i]);
      Dense<T> x1(n, 1);
      if (p > 0) {
        x1 = Mul(Adjoint(bdr), Solve(Mul(bdr, Adjoint(bdr)), dv));
      }
      // The null space of the rounded B: Q2 projected.
      Dense<T> n2 = q2;
      if (p > 0) {
        const Dense<T> proj =
            Mul(Adjoint(bdr), Solve(Mul(bdr, Adjoint(bdr)), Mul(bdr, q2)));
        n2 = Sub(q2, proj);
      }
      Dense<T> xref = x1;
      if (n > p) {
        const Dense<T> an = Mul(ad, n2);
        const Dense<T> rhs = Sub(cv, Mul(ad, x1));
        const Dense<T> z = Solve(Mul(Adjoint(an), an), Mul(Adjoint(an), rhs));
        const Dense<T> nz = Mul(n2, z);
        for (hwyblas_int i = 0; i < n; ++i) xref(i, 0) += nz(i, 0);
      }
      std::vector<T> x(std::max<hwyblas_int>(1, n), NaN<T>());
      hwyblas_int lwork, info = -99;
      if (minimal) {
        lwork = std::max<hwyblas_int>(1, m + n + p);
      } else {
        T wq = T(0);
        const hwyblas_int m1 = -1;
        HWYBLAS_CALL(T, gglse, &m, &n, &p, P(a.data()), &lda, P(b.data()), &ldb,
                     P(c.data()), P(dd.data()), P(x.data()), P(&wq), &m1,
                     &info);
        ASSERT_EQ(info, 0);
        lwork = QueryResult(wq);
      }
      std::vector<T> work(lwork, NaN<T>());
      HWYBLAS_CALL(T, gglse, &m, &n, &p, P(a.data()), &lda, P(b.data()), &ldb,
                   P(c.data()), P(dd.data()), P(x.data()), P(work.data()),
                   &lwork, &info);
      ASSERT_EQ(info, 0);
      EXPECT_LT(SolutionError(x, n, 1, n, xref), LsTolerance<T>(m, n, 10));
      // The residual sum of squares: || C(N-P+1:M) ||**2 = || c - A*x ||**2.
      const Dense<T> res = Sub(cv, Mul(ad, xref));
      double expected = 0, got = 0;
      for (hwyblas_int i = 0; i < m; ++i) expected += std::norm(res(i, 0));
      for (hwyblas_int i = n - p; i < m; ++i) {
        got += std::norm(static_cast<RefT<T>>(c[i]));
      }
      EXPECT_NEAR(std::sqrt(got), std::sqrt(expected),
                  LsTolerance<T>(m, n, 10) * (1 + std::sqrt(expected)));
    }
  }
  // INFO = 1: B is rank deficient; INFO = 2: ( A; B ) is.
  for (const hwyblas_int expected : {1, 2}) {
    const hwyblas_int m = 4, n = 3, p = 2, lda = 4, ldb = 2;
    std::vector<T> a = rng.Vector(lda * n), b = rng.Vector(ldb * n);
    if (expected == 1) {
      // A zero row of B: exactly zero in T12.
      for (hwyblas_int j = 0; j < n; ++j) At(b, 1, j, ldb) = T(0);
    } else {
      // A*x = 0 and B*x = 0 for x = e1.
      for (hwyblas_int i = 0; i < m; ++i) At(a, i, 0, lda) = T(0);
      for (hwyblas_int i = 0; i < p; ++i) At(b, i, 0, ldb) = T(0);
    }
    std::vector<T> c = rng.Vector(m), dd = rng.Vector(p), x(n);
    hwyblas_int lwork = 64, info = -99;
    std::vector<T> work(lwork);
    HWYBLAS_CALL(T, gglse, &m, &n, &p, P(a.data()), &lda, P(b.data()), &ldb,
                 P(c.data()), P(dd.data()), P(x.data()), P(work.data()), &lwork,
                 &info);
    EXPECT_EQ(info, expected);
  }
}

// xGGGLM: y minimizes || y || subject to d = A*x + B*y, for A = Q1 * R
// (Q = [Q1 Q2] n x n orthonormal): y = B**H * Q2 * w with (Q2**H B B**H
// Q2) w = Q2**H d, and x the solution of R*x = Q1**H (d - B*y). Also n =
// m, m = 0, and the singular cases.
TYPED_TEST(GelsTest, Ggglm) {
  using T = TypeParam;
  Random<T> rng(457);
  for (const auto& d : {std::array<hwyblas_int, 3>{5, 3, 4},
                        {4, 4, 2},
                        {4, 0, 6},
                        {6, 2, 4},
                        {30, 18, 20},
                        {40, 25, 40}}) {
    const hwyblas_int n = d[0], m = d[1], p = d[2];
    for (const bool minimal : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " m=" << m << " p=" << p
                                        << " minimal=" << minimal);
      const hwyblas_int lda = n + 1, ldb = n + 2;
      std::vector<T> a = rng.Vector(lda * std::max<hwyblas_int>(1, m));
      std::vector<T> b = rng.Vector(ldb * p), dd = rng.Vector(n);
      const Dense<T> ad = ToDense(a, n, m, lda), bd = ToDense(b, n, p, ldb);
      Dense<T> dv(n, 1);
      for (hwyblas_int i = 0; i < n; ++i)
        dv(i, 0) = static_cast<RefT<T>>(dd[i]);
      // The reference: minimize ||y|| subject to A x + B y = d, from the
      // normal equations of the KKT system: y = B**H u, where u solves
      // (P B B**H P) u = P d in range(P), P the projector on null(A**H).
      Dense<T> proj = Identity<T>(n);
      if (m > 0) {
        proj = Sub(proj, Mul(ad, Solve(Mul(Adjoint(ad), ad), Adjoint(ad))));
      }
      const Dense<T> pb = Mul(proj, bd);
      // u = (P B)^+**H ... : y = (P B)^+ (P d) (minimum norm solution of
      // P B y = P d), since A x = d - B y then has a solution.
      const Dense<T> pbh = Adjoint(pb);
      // (P B)^+ = (P B)**H (P B (P B)**H)^+; P B (P B)**H has rank n - m:
      // add A A**H to make it invertible without changing the solution.
      Dense<T> g = Mul(pb, pbh);
      if (m > 0) {
        const Dense<T> aah = Mul(ad, Adjoint(ad));
        for (size_t i = 0; i < g.v.size(); ++i) g.v[i] += aah.v[i];
      }
      const Dense<T> yref = Mul(pbh, Solve(g, Mul(proj, dv)));
      Dense<T> xref(m, 1);
      if (m > 0) {
        xref = Solve(Mul(Adjoint(ad), ad),
                     Mul(Adjoint(ad), Sub(dv, Mul(bd, yref))));
      }
      std::vector<T> x(std::max<hwyblas_int>(1, m), NaN<T>()),
          y(std::max<hwyblas_int>(1, p), NaN<T>());
      hwyblas_int lwork, info = -99;
      if (minimal) {
        lwork = std::max<hwyblas_int>(1, n + m + p);
      } else {
        T wq = T(0);
        const hwyblas_int m1 = -1;
        HWYBLAS_CALL(T, ggglm, &n, &m, &p, P(a.data()), &lda, P(b.data()), &ldb,
                     P(dd.data()), P(x.data()), P(y.data()), P(&wq), &m1,
                     &info);
        ASSERT_EQ(info, 0);
        lwork = QueryResult(wq);
      }
      std::vector<T> work(lwork, NaN<T>());
      HWYBLAS_CALL(T, ggglm, &n, &m, &p, P(a.data()), &lda, P(b.data()), &ldb,
                   P(dd.data()), P(x.data()), P(y.data()), P(work.data()),
                   &lwork, &info);
      ASSERT_EQ(info, 0);
      if (n == m) {
        // y = 0 (the reference only approximately).
        for (hwyblas_int i = 0; i < p; ++i) EXPECT_EQ(y[i], T(0)) << i;
      } else {
        EXPECT_LT(SolutionError(y, p, 1, p, yref), LsTolerance<T>(n, p, 10));
      }
      if (m > 0) {
        EXPECT_LT(SolutionError(x, m, 1, m, xref), LsTolerance<T>(n, p, 10));
      }
    }
  }
  // N = 0: X and Y are set to zero. INFO = 1: T22 (of B) singular; INFO =
  // 2: R11 (of A) singular.
  {
    const hwyblas_int n = 0, m = 0, p = 2, lda = 1, ldb = 1;
    std::vector<T> a(1), b(2), dd(1), x(1, T(5)), y(2, T(5));
    hwyblas_int lwork = 4, info = -99;
    std::vector<T> work(lwork);
    HWYBLAS_CALL(T, ggglm, &n, &m, &p, P(a.data()), &lda, P(b.data()), &ldb,
                 P(dd.data()), P(x.data()), P(y.data()), P(work.data()), &lwork,
                 &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(y[0], T(0));
    EXPECT_EQ(y[1], T(0));
  }
  for (const hwyblas_int expected : {1, 2}) {
    const hwyblas_int n = 4, m = 2, p = 3, lda = 4, ldb = 4;
    std::vector<T> a = rng.Vector(lda * m), b = rng.Vector(ldb * p);
    // B = 0 makes T22 zero, a zero column of A R11 singular (exactly).
    if (expected == 1) {
      for (T& v : b) v = T(0);
    } else {
      for (hwyblas_int i = 0; i < n; ++i) At(a, i, 1, lda) = T(0);
    }
    std::vector<T> dd = rng.Vector(n), x(m), y(p);
    hwyblas_int lwork = 64, info = -99;
    std::vector<T> work(lwork);
    HWYBLAS_CALL(T, ggglm, &n, &m, &p, P(a.data()), &lda, P(b.data()), &ldb,
                 P(dd.data()), P(x.data()), P(y.data()), P(work.data()), &lwork,
                 &info);
    EXPECT_EQ(info, expected);
  }
}

// ---------------------------------------------------------------------------
// Workspace queries and errors

// The optimal workspaces of a query with NB = 30 (ILAENV's block size for
// every routine), and the argument checks of all the routines.
TYPED_TEST(GelsTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  const char ctrans = kComplex ? 'C' : 'T';
  const hwyblas_int nb = 30, m1 = -1;
  std::vector<T> a(4096), b(4096), c(4096), d(64), x(64), y(64), tau(64);
  std::vector<R> s(64), rwork(4096);
  std::vector<hwyblas_int> iwork(4096), jpvt(64);
  hwyblas_int info = -99, rank = -1;
  const R rcond = -1;
  {
    const ScopedBlockSize block_size(nb);
    T wq = T(0);
    // xGELS: max(1, MN + max(MN, NRHS) * NB).
    for (const auto& dims :
         {std::array<hwyblas_int, 3>{40, 20, 3}, {20, 40, 50}, {0, 5, 2}}) {
      const hwyblas_int m = dims[0], n = dims[1], nrhs = dims[2];
      const hwyblas_int lda = std::max<hwyblas_int>(1, m),
                        ldb = std::max<hwyblas_int>({1, m, n});
      HWYBLAS_CALL(T, gels, &ctrans, &m, &n, &nrhs, P(a.data()), &lda,
                   P(b.data()), &ldb, P(&wq), &m1, &info);
      EXPECT_EQ(info, 0);
      const hwyblas_int mn = std::min(m, n);
      EXPECT_EQ(QueryResult(wq),
                std::max<hwyblas_int>(1, mn + std::max(mn, nrhs) * nb));
    }
    // xGGLSE: P + min(M,N) + max(M,N)*NB; xGGGLM: M + min(N,P) +
    // max(N,P)*NB; xGGQRF / xGGRQF: max(1, max(N,M,P)*NB).
    {
      const hwyblas_int mm = 9, n = 7, p = 3, lda = 9, ldb = 9;
      HWYBLAS_CALL(T, gglse, &mm, &n, &p, P(a.data()), &lda, P(b.data()), &ldb,
                   P(c.data()), P(d.data()), P(x.data()), P(&wq), &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), p + std::min(mm, n) + std::max(mm, n) * nb);
      const hwyblas_int gn = 7, gm = 4, gp = 5;
      HWYBLAS_CALL(T, ggglm, &gn, &gm, &gp, P(a.data()), &lda, P(b.data()),
                   &ldb, P(d.data()), P(x.data()), P(y.data()), P(&wq), &m1,
                   &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), gm + std::min(gn, gp) + std::max(gn, gp) * nb);
      HWYBLAS_CALL(T, ggqrf, &gn, &gm, &gp, P(a.data()), &lda, P(tau.data()),
                   P(b.data()), &ldb, P(tau.data()), P(&wq), &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), std::max({gn, gm, gp}) * nb);
      HWYBLAS_CALL(T, ggrqf, &gm, &gp, &gn, P(a.data()), &lda, P(tau.data()),
                   P(b.data()), &ldb, P(tau.data()), P(&wq), &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), std::max({gn, gm, gp}) * nb);
    }
    // xGERQF: M*NB (1 if min(M,N) = 0); xTZRZF: M*NB (1 if M = 0 or M =
    // N); xORMRQ / xORMRZ: NW*NB + 65*64 (1 if M = 0 or N = 0).
    for (const auto& dims :
         {std::array<hwyblas_int, 2>{8, 12}, {0, 3}, {5, 5}}) {
      const hwyblas_int m = dims[0], n = dims[1];
      const hwyblas_int lda = std::max<hwyblas_int>(1, m);
      HWYBLAS_CALL(T, gerqf, &m, &n, P(a.data()), &lda, P(tau.data()), P(&wq),
                   &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), std::min(m, n) == 0 ? 1 : m * nb);
      HWYBLAS_CALL(T, tzrzf, &m, &n, P(a.data()), &lda, P(tau.data()), P(&wq),
                   &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), (m == 0 || m == n) ? 1 : m * nb);
      const char side = 'R';
      const hwyblas_int cm = 6, k = m, l = n - m;
      CALL_RC(T, ormrq, unmrq, &side, &ctrans, &cm, &n, &k, P(a.data()), &lda,
              P(tau.data()), P(c.data()), &cm, P(&wq), &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), n == 0 ? 1 : cm * nb + 65 * 64);
      CALL_RC(T, ormrz, unmrz, &side, &ctrans, &cm, &n, &k, &l, P(a.data()),
              &lda, P(tau.data()), P(c.data()), &cm, P(&wq), &m1, &info);
      EXPECT_EQ(info, 0);
      EXPECT_EQ(QueryResult(wq), n == 0 ? 1 : cm * nb + 65 * 64);
    }
    // xGELSD: also the minimal LIWORK in IWORK(1) and, for complex T, the
    // minimal LRWORK in RWORK(1) (positive).
    {
      const hwyblas_int m = 50, n = 30, nrhs = 2, lda = 50, ldb = 50;
      iwork[0] = -77;
      rwork[0] = -1;
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, gelsd, &m, &n, &nrhs, P(a.data()), &lda,
                             P(b.data()), &ldb, s.data(), &rcond, &rank, P(&wq),
                             &m1, rwork.data(), iwork.data(), &info);
        EXPECT_GT(rwork[0], R(0));
      } else {
        HWYBLAS_CALL_REAL(T, gelsd, &m, &n, &nrhs, a.data(), &lda, b.data(),
                          &ldb, s.data(), &rcond, &rank, &wq, &m1, iwork.data(),
                          &info);
      }
      EXPECT_EQ(info, 0);
      EXPECT_GT(QueryResult(wq), 0);
      // 3*MINMN*NLVL + 11*MINMN with SMLSIZ = 25: NLVL = 1.
      EXPECT_EQ(iwork[0], 3 * n * 1 + 11 * n);
    }
  }

  // Invalid arguments.
  ErrorCapture errors;
  const hwyblas_int k4 = 4, big = 999;
  std::vector<std::string> expected;
  const auto expect = [&](const char* routine,
                          std::initializer_list<int> positions) {
    for (const int pos : positions) {
      expected.push_back(Name<T>(routine) + ":" + std::to_string(pos));
    }
  };
  const auto gels = [&](char trans, hwyblas_int m, hwyblas_int n,
                        hwyblas_int nrhs, hwyblas_int lda, hwyblas_int ldb,
                        hwyblas_int lwork) {
    hwyblas_int inf = -99;
    HWYBLAS_CALL(T, gels, &trans, &m, &n, &nrhs, P(a.data()), &lda, P(b.data()),
                 &ldb, P(c.data()), &lwork, &inf);
    return inf;
  };
  EXPECT_EQ(gels('X', 4, 3, 1, 4, 4, big), -1);
  EXPECT_EQ(gels(kComplex ? 'T' : 'C', 4, 3, 1, 4, 4, big), -1);
  EXPECT_EQ(gels('N', -1, 3, 1, 4, 4, big), -2);
  EXPECT_EQ(gels('N', 4, -1, 1, 4, 4, big), -3);
  EXPECT_EQ(gels('N', 4, 3, -1, 4, 4, big), -4);
  EXPECT_EQ(gels('N', 4, 3, 1, 3, 4, big), -6);
  EXPECT_EQ(gels('N', 3, 4, 1, 3, 3, big), -8);
  EXPECT_EQ(gels('N', 4, 3, 2, 4, 4, 3 + 3 - 1), -10);
  expect("GELS", {1, 1, 2, 3, 4, 6, 8, 10});

  // xGELSS (which = 0), xGELSD (1) and xGELSY (2): the same positions.
  const auto ls = [&](int which, hwyblas_int m, hwyblas_int n, hwyblas_int nrhs,
                      hwyblas_int lda, hwyblas_int ldb, hwyblas_int lwork) {
    hwyblas_int inf = -99, r = -1;
    if (which == 0) {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, gelss, &m, &n, &nrhs, P(a.data()), &lda,
                             P(b.data()), &ldb, s.data(), &rcond, &r,
                             P(c.data()), &lwork, rwork.data(), &inf);
      } else {
        HWYBLAS_CALL_REAL(T, gelss, &m, &n, &nrhs, a.data(), &lda, b.data(),
                          &ldb, s.data(), &rcond, &r, c.data(), &lwork, &inf);
      }
    } else if (which == 1) {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, gelsd, &m, &n, &nrhs, P(a.data()), &lda,
                             P(b.data()), &ldb, s.data(), &rcond, &r,
                             P(c.data()), &lwork, rwork.data(), iwork.data(),
                             &inf);
      } else {
        HWYBLAS_CALL_REAL(T, gelsd, &m, &n, &nrhs, a.data(), &lda, b.data(),
                          &ldb, s.data(), &rcond, &r, c.data(), &lwork,
                          iwork.data(), &inf);
      }
    } else {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, gelsy, &m, &n, &nrhs, P(a.data()), &lda,
                             P(b.data()), &ldb, jpvt.data(), &rcond, &r,
                             P(c.data()), &lwork, rwork.data(), &inf);
      } else {
        HWYBLAS_CALL_REAL(T, gelsy, &m, &n, &nrhs, a.data(), &lda, b.data(),
                          &ldb, jpvt.data(), &rcond, &r, c.data(), &lwork,
                          &inf);
      }
    }
    return inf;
  };
  for (int which = 0; which < 3; ++which) {
    EXPECT_EQ(ls(which, -1, 3, 1, 4, 4, big), -1);
    EXPECT_EQ(ls(which, 4, -1, 1, 4, 4, big), -2);
    EXPECT_EQ(ls(which, 4, 3, -1, 4, 4, big), -3);
    EXPECT_EQ(ls(which, 4, 3, 1, 3, 4, big), -5);
    EXPECT_EQ(ls(which, 3, 4, 1, 3, 3, big), -7);
    EXPECT_EQ(ls(which, 4, 3, 1, 4, 4, 2), -12);
    expect(which == 0   ? "GELSS"
           : which == 1 ? "GELSD"
                        : "GELSY",
           {1, 2, 3, 5, 7, 12});
  }

  const auto gglse = [&](hwyblas_int m, hwyblas_int n, hwyblas_int p,
                         hwyblas_int lda, hwyblas_int ldb, hwyblas_int lwork) {
    hwyblas_int inf = -99;
    HWYBLAS_CALL(T, gglse, &m, &n, &p, P(a.data()), &lda, P(b.data()), &ldb,
                 P(c.data()), P(d.data()), P(x.data()), P(c.data() + 64),
                 &lwork, &inf);
    return inf;
  };
  EXPECT_EQ(gglse(-1, 3, 2, 4, 2, big), -1);
  EXPECT_EQ(gglse(4, -1, 2, 4, 2, big), -2);
  EXPECT_EQ(gglse(4, 3, 4, 4, 4, big), -3);  // P > N
  EXPECT_EQ(gglse(1, 4, 2, 4, 2, big), -3);  // P < N - M
  EXPECT_EQ(gglse(4, 3, 2, 3, 2, big), -5);
  EXPECT_EQ(gglse(4, 3, 2, 4, 1, big), -7);
  EXPECT_EQ(gglse(4, 3, 2, 4, 2, 4 + 3 + 2 - 1), -12);
  expect("GGLSE", {1, 2, 3, 3, 5, 7, 12});

  const auto ggglm = [&](hwyblas_int n, hwyblas_int m, hwyblas_int p,
                         hwyblas_int lda, hwyblas_int ldb, hwyblas_int lwork) {
    hwyblas_int inf = -99;
    HWYBLAS_CALL(T, ggglm, &n, &m, &p, P(a.data()), &lda, P(b.data()), &ldb,
                 P(d.data()), P(x.data()), P(y.data()), P(c.data()), &lwork,
                 &inf);
    return inf;
  };
  EXPECT_EQ(ggglm(-1, 2, 3, 4, 4, big), -1);
  EXPECT_EQ(ggglm(4, 5, 3, 4, 4, big), -2);  // M > N
  EXPECT_EQ(ggglm(4, 2, 1, 4, 4, big), -3);  // P < N - M
  EXPECT_EQ(ggglm(4, 2, 3, 3, 4, big), -5);
  EXPECT_EQ(ggglm(4, 2, 3, 4, 3, big), -7);
  EXPECT_EQ(ggglm(4, 2, 3, 4, 4, 4 + 2 + 3 - 1), -12);
  expect("GGGLM", {1, 2, 3, 5, 7, 12});

  for (const bool rq : {false, true}) {
    const auto gg = [&](hwyblas_int d1, hwyblas_int d2, hwyblas_int d3,
                        hwyblas_int lda, hwyblas_int ldb, hwyblas_int lwork) {
      hwyblas_int inf = -99;
      if (rq) {
        HWYBLAS_CALL(T, ggrqf, &d1, &d2, &d3, P(a.data()), &lda, P(tau.data()),
                     P(b.data()), &ldb, P(tau.data()), P(c.data()), &lwork,
                     &inf);
      } else {
        HWYBLAS_CALL(T, ggqrf, &d1, &d2, &d3, P(a.data()), &lda, P(tau.data()),
                     P(b.data()), &ldb, P(tau.data()), P(c.data()), &lwork,
                     &inf);
      }
      return inf;
    };
    EXPECT_EQ(gg(-1, 2, 3, 4, 4, big), -1);
    EXPECT_EQ(gg(4, -1, 3, 4, 4, big), -2);
    EXPECT_EQ(gg(4, 2, -1, 4, 4, big), -3);
    // GQR: LDA, LDB >= N (the first); GRQ: LDA >= M (the first), LDB >= P.
    EXPECT_EQ(gg(4, 2, 3, 3, 4, big), -5);
    EXPECT_EQ(gg(4, 2, 3, 4, rq ? 1 : 3, big), -8);
    EXPECT_EQ(gg(4, 2, 3, 4, 4, 3), -11);
    expect(rq ? "GGRQF" : "GGQRF", {1, 2, 3, 5, 8, 11});
  }

  const auto gerqf = [&](bool unblocked, hwyblas_int m, hwyblas_int n,
                         hwyblas_int lda, hwyblas_int lwork) {
    hwyblas_int inf = -99;
    if (unblocked) {
      HWYBLAS_CALL(T, gerq2, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(c.data()), &inf);
    } else {
      HWYBLAS_CALL(T, gerqf, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(c.data()), &lwork, &inf);
    }
    return inf;
  };
  EXPECT_EQ(gerqf(false, -1, 3, 4, big), -1);
  EXPECT_EQ(gerqf(false, 4, -1, 4, big), -2);
  EXPECT_EQ(gerqf(false, 4, 3, 3, big), -4);
  EXPECT_EQ(gerqf(false, 4, 3, 4, 3), -7);
  EXPECT_EQ(gerqf(false, 4, 0, 4, 0), -7);  // LWORK <= 0
  EXPECT_EQ(gerqf(false, 4, 0, 4, 1), 0);   // N = 0: any LWORK >= 1
  EXPECT_EQ(gerqf(true, -1, 3, 4, 0), -1);
  EXPECT_EQ(gerqf(true, 4, -1, 4, 0), -2);
  EXPECT_EQ(gerqf(true, 4, 3, 3, 0), -4);
  expect("GERQF", {1, 2, 4, 7, 7});
  expect("GERQ2", {1, 2, 4});

  const auto tzrzf = [&](hwyblas_int m, hwyblas_int n, hwyblas_int lda,
                         hwyblas_int lwork) {
    hwyblas_int inf = -99;
    HWYBLAS_CALL(T, tzrzf, &m, &n, P(a.data()), &lda, P(tau.data()),
                 P(c.data()), &lwork, &inf);
    return inf;
  };
  EXPECT_EQ(tzrzf(-1, 3, 4, big), -1);
  EXPECT_EQ(tzrzf(4, 3, 4, big), -2);
  EXPECT_EQ(tzrzf(3, 4, 2, big), -4);
  EXPECT_EQ(tzrzf(3, 4, 3, 2), -7);
  expect("TZRZF", {1, 2, 4, 7});

  // xORMRQ / xUNMRQ (blocked) and xORMR2 / xUNMR2; xORMRZ / xUNMRZ and
  // xORMR3 / xUNMR3 (with L).
  for (int variant = 0; variant < 4; ++variant) {
    const bool rz = variant >= 2, blocked = variant % 2 == 0;
    const hwyblas_int shift = rz ? 1 : 0;
    const auto mul = [&](char side, char trans, hwyblas_int m, hwyblas_int n,
                         hwyblas_int k, hwyblas_int l, hwyblas_int lda,
                         hwyblas_int ldc, hwyblas_int lwork) {
      hwyblas_int inf = -99;
      if (rz && blocked) {
        CALL_RC(T, ormrz, unmrz, &side, &trans, &m, &n, &k, &l, P(a.data()),
                &lda, P(tau.data()), P(c.data()), &ldc, P(b.data()), &lwork,
                &inf);
      } else if (rz) {
        CALL_RC(T, ormr3, unmr3, &side, &trans, &m, &n, &k, &l, P(a.data()),
                &lda, P(tau.data()), P(c.data()), &ldc, P(b.data()), &inf);
      } else if (blocked) {
        CALL_RC(T, ormrq, unmrq, &side, &trans, &m, &n, &k, P(a.data()), &lda,
                P(tau.data()), P(c.data()), &ldc, P(b.data()), &lwork, &inf);
      } else {
        CALL_RC(T, ormr2, unmr2, &side, &trans, &m, &n, &k, P(a.data()), &lda,
                P(tau.data()), P(c.data()), &ldc, P(b.data()), &inf);
      }
      return inf;
    };
    EXPECT_EQ(mul('X', 'N', 4, 3, 2, 1, 2, 4, big), -1);
    EXPECT_EQ(mul('L', kComplex ? 'T' : 'C', 4, 3, 2, 1, 2, 4, big), -2);
    EXPECT_EQ(mul('L', 'N', -1, 3, 2, 1, 2, 4, big), -3);
    EXPECT_EQ(mul('L', 'N', 4, -1, 2, 1, 2, 4, big), -4);
    EXPECT_EQ(mul('L', 'N', 4, 3, 5, 1, 5, 4, big), -5);
    if (rz) {
      EXPECT_EQ(mul('L', 'N', 4, 3, 2, 5, 2, 4, big), -6);
      EXPECT_EQ(mul('R', 'N', 4, 3, 2, -1, 2, 4, big), -6);
    }
    EXPECT_EQ(mul('L', 'N', 4, 3, 2, 1, 1, 4, big), -7 - shift);
    EXPECT_EQ(mul('L', 'N', 4, 3, 2, 1, 2, 3, big), -10 - shift);
    if (blocked) EXPECT_EQ(mul('L', 'N', 4, 3, 2, 1, 2, 4, 2), -12 - shift);
    const char* name = rz ? (blocked ? (kComplex ? "UNMRZ" : "ORMRZ")
                                     : (kComplex ? "UNMR3" : "ORMR3"))
                          : (blocked ? (kComplex ? "UNMRQ" : "ORMRQ")
                                     : (kComplex ? "UNMR2" : "ORMR2"));
    expect(name, {1, 2, 3, 4, 5});
    if (rz) expect(name, {6, 6});
    expect(name, {7 + static_cast<int>(shift), 10 + static_cast<int>(shift)});
    if (blocked) expect(name, {12 + static_cast<int>(shift)});
  }

  // xLARZB and xLARZT support DIRECT = 'B' and STOREV = 'R' only.
  {
    const hwyblas_int one = 1, two = 2;
    const char side = 'L', notrans = 'N', back = 'B', rowwise = 'R', fwd = 'F',
               colwise = 'C';
    HWYBLAS_CALL(T, larzb, &side, &notrans, &fwd, &rowwise, &two, &two, &one,
                 &one, P(a.data()), &one, P(b.data()), &one, P(c.data()), &two,
                 P(d.data()), &two);
    HWYBLAS_CALL(T, larzb, &side, &notrans, &back, &colwise, &two, &two, &one,
                 &one, P(a.data()), &one, P(b.data()), &one, P(c.data()), &two,
                 P(d.data()), &two);
    HWYBLAS_CALL(T, larzt, &fwd, &rowwise, &two, &one, P(a.data()), &one,
                 P(tau.data()), P(b.data()), &one);
    HWYBLAS_CALL(T, larzt, &back, &colwise, &two, &one, P(a.data()), &one,
                 P(tau.data()), P(b.data()), &one);
    expect("LARZB", {3, 4});
    expect("LARZT", {1, 2});
  }

  // xLALSD, xLALSA and xLALS0.
  {
    const char upper = 'U';
    const hwyblas_int smlsiz = 4;
    const auto lalsd = [&](hwyblas_int n, hwyblas_int nrhs, hwyblas_int ldb) {
      hwyblas_int inf = -99, r = -1;
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, lalsd, &upper, &smlsiz, &n, &nrhs, s.data(),
                             rwork.data(), P(b.data()), &ldb, &rcond, &r,
                             P(c.data()), rwork.data() + 64, iwork.data(),
                             &inf);
      } else {
        HWYBLAS_CALL_REAL(T, lalsd, &upper, &smlsiz, &n, &nrhs, s.data(),
                          rwork.data(), b.data(), &ldb, &rcond, &r, c.data(),
                          iwork.data(), &inf);
      }
      return inf;
    };
    EXPECT_EQ(lalsd(-1, 1, 1), -3);
    EXPECT_EQ(lalsd(4, 0, 4), -4);
    EXPECT_EQ(lalsd(4, 1, 3), -8);
    expect("LALSD", {3, 4, 8});
    const auto lalsa = [&](hwyblas_int icompq, hwyblas_int sml, hwyblas_int n,
                           hwyblas_int nrhs, hwyblas_int ldb, hwyblas_int ldbx,
                           hwyblas_int ldu, hwyblas_int ldgcol) {
      hwyblas_int inf = -99;
      R* rw = rwork.data();
      hwyblas_int* iw = iwork.data();
      HWYBLAS_CALL(T, lalsa, &icompq, &sml, &n, &nrhs, P(b.data()), &ldb,
                   P(c.data()), &ldbx, rw, &ldu, rw, iw, rw, rw, rw, rw, iw, iw,
                   &ldgcol, iw, rw, rw, rw, rw + 2048, iw + 2048, &inf);
      return inf;
    };
    EXPECT_EQ(lalsa(2, 4, 8, 1, 8, 8, 8, 8), -1);
    EXPECT_EQ(lalsa(0, 2, 8, 1, 8, 8, 8, 8), -2);
    EXPECT_EQ(lalsa(0, 4, 3, 1, 8, 8, 8, 8), -3);
    EXPECT_EQ(lalsa(0, 4, 8, 0, 8, 8, 8, 8), -4);
    EXPECT_EQ(lalsa(0, 4, 8, 1, 7, 8, 8, 8), -6);
    EXPECT_EQ(lalsa(0, 4, 8, 1, 8, 7, 8, 8), -8);
    EXPECT_EQ(lalsa(0, 4, 8, 1, 8, 8, 7, 8), -10);
    EXPECT_EQ(lalsa(0, 4, 8, 1, 8, 8, 8, 7), -19);
    expect("LALSA", {1, 2, 3, 4, 6, 8, 10, 19});
    const auto lals0 = [&](hwyblas_int icompq, hwyblas_int nl, hwyblas_int nr,
                           hwyblas_int sqre, hwyblas_int nrhs, hwyblas_int ldb,
                           hwyblas_int ldbx, hwyblas_int givptr,
                           hwyblas_int ldgcol, hwyblas_int ldgnum,
                           hwyblas_int k) {
      hwyblas_int inf = -99;
      R* rw = rwork.data();
      hwyblas_int* iw = iwork.data();
      const R cs = 1, sn = 0;
      HWYBLAS_CALL(T, lals0, &icompq, &nl, &nr, &sqre, &nrhs, P(b.data()), &ldb,
                   P(c.data()), &ldbx, iw, &givptr, iw, &ldgcol, rw, &ldgnum,
                   rw, rw, rw, rw, &k, &cs, &sn, rw + 2048, &inf);
      return inf;
    };
    // N = NL + NR + 1 = 5.
    EXPECT_EQ(lals0(2, 2, 2, 0, 1, 5, 5, 0, 5, 5, 1), -1);
    EXPECT_EQ(lals0(0, 0, 2, 0, 1, 5, 5, 0, 5, 5, 1), -2);
    EXPECT_EQ(lals0(0, 2, 0, 0, 1, 5, 5, 0, 5, 5, 1), -3);
    EXPECT_EQ(lals0(0, 2, 2, 2, 1, 5, 5, 0, 5, 5, 1), -4);
    EXPECT_EQ(lals0(0, 2, 2, 0, 0, 5, 5, 0, 5, 5, 1), -5);
    EXPECT_EQ(lals0(0, 2, 2, 0, 1, 4, 5, 0, 5, 5, 1), -7);
    EXPECT_EQ(lals0(0, 2, 2, 0, 1, 5, 4, 0, 5, 5, 1), -9);
    EXPECT_EQ(lals0(0, 2, 2, 0, 1, 5, 5, -1, 5, 5, 1), -11);
    EXPECT_EQ(lals0(0, 2, 2, 0, 1, 5, 5, 0, 4, 5, 1), -13);
    EXPECT_EQ(lals0(0, 2, 2, 0, 1, 5, 5, 0, 5, 4, 1), -15);
    EXPECT_EQ(lals0(0, 2, 2, 0, 1, 5, 5, 0, 5, 5, 0), -20);
    expect("LALS0", {1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 20});
  }
  (void)k4;
  EXPECT_EQ(errors.Take(), expected);
}

// ---------------------------------------------------------------------------
// LAPACKE

// Row-major results are the column-major ones of the transposed matrices,
// bitwise, for all the functions (also xORMRQ / xORMRZ with SIDE = 'R' and
// N > M, where the reference LAPACKE copies A as K x M).
TYPED_TEST(GelsTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  const char ctrans = IsComplex<T>::value ? 'C' : 'T';
  Random<T> rng(461);
  // Copies of an m x n column-major matrix in both layouts (ld = 9).
  const lapack_int ld = 9;
  const auto both = [&](lapack_int m, lapack_int n) {
    std::vector<T> col(ld * ld, NaN<T>());
    for (lapack_int j = 0; j < n; ++j) {
      for (lapack_int i = 0; i < m; ++i) At(col, i, j, ld) = rng.Next();
    }
    return std::make_pair(col, Transposed(col, ld));
  };
  // The m x n results agree.
  const auto same = [&](const std::vector<T>& col, const std::vector<T>& row,
                        lapack_int m, lapack_int n, const char* what) {
    for (lapack_int i = 0; i < m; ++i) {
      for (lapack_int j = 0; j < n; ++j) {
        ASSERT_EQ(row[i * ld + j], At(col, i, j, ld))
            << what << " " << i << ", " << j;
      }
    }
  };
  for (const auto& dims : {std::pair<lapack_int, lapack_int>{7, 4}, {4, 7}}) {
    const lapack_int m = dims.first, n = dims.second, nrhs = 2;
    const lapack_int mx = std::max(m, n);
    SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n);
    std::vector<T> ac, ar;
    std::tie(ac, ar) = both(m, n);
    std::vector<T> bc, br;
    std::tie(bc, br) = both(mx, nrhs);
    // xGELS.
    for (const char trans : {'N', ctrans}) {
      std::vector<T> a1 = ac, a2 = ar, b1 = bc, b2 = br;
      ASSERT_EQ(LAPACKE_CALL(T, gels, LAPACK_COL_MAJOR, trans, m, n, nrhs,
                             a1.data(), ld, b1.data(), ld),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, gels, LAPACK_ROW_MAJOR, trans, m, n, nrhs,
                             a2.data(), ld, b2.data(), ld),
                0);
      same(a1, a2, m, n, "gels a");
      same(b1, b2, mx, nrhs, "gels b");
    }
    // xGELSS, xGELSD, xGELSY.
    const R rcond = -1;
    {
      std::vector<T> a1 = ac, a2 = ar, b1 = bc, b2 = br;
      std::vector<R> s1(ld), s2(ld);
      lapack_int r1 = -1, r2 = -1;
      ASSERT_EQ(LAPACKE_CALL(T, gelss, LAPACK_COL_MAJOR, m, n, nrhs, a1.data(),
                             ld, b1.data(), ld, s1.data(), rcond, &r1),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, gelss, LAPACK_ROW_MAJOR, m, n, nrhs, a2.data(),
                             ld, b2.data(), ld, s2.data(), rcond, &r2),
                0);
      EXPECT_EQ(r1, r2);
      EXPECT_EQ(s1, s2);
      same(a1, a2, m, n, "gelss a");
      same(b1, b2, mx, nrhs, "gelss b");
    }
    {
      std::vector<T> a1 = ac, a2 = ar, b1 = bc, b2 = br;
      std::vector<R> s1(ld), s2(ld);
      lapack_int r1 = -1, r2 = -1;
      ASSERT_EQ(LAPACKE_CALL(T, gelsd, LAPACK_COL_MAJOR, m, n, nrhs, a1.data(),
                             ld, b1.data(), ld, s1.data(), rcond, &r1),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, gelsd, LAPACK_ROW_MAJOR, m, n, nrhs, a2.data(),
                             ld, b2.data(), ld, s2.data(), rcond, &r2),
                0);
      EXPECT_EQ(r1, r2);
      EXPECT_EQ(s1, s2);
      same(b1, b2, mx, nrhs, "gelsd b");
    }
    {
      std::vector<T> a1 = ac, a2 = ar, b1 = bc, b2 = br;
      std::vector<lapack_int> j1(n, 0), j2(n, 0);
      j1[1] = j2[1] = 1;
      lapack_int r1 = -1, r2 = -1;
      ASSERT_EQ(LAPACKE_CALL(T, gelsy, LAPACK_COL_MAJOR, m, n, nrhs, a1.data(),
                             ld, b1.data(), ld, j1.data(), rcond, &r1),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, gelsy, LAPACK_ROW_MAJOR, m, n, nrhs, a2.data(),
                             ld, b2.data(), ld, j2.data(), rcond, &r2),
                0);
      EXPECT_EQ(r1, r2);
      EXPECT_EQ(j1, j2);
      same(a1, a2, m, n, "gelsy a");
      same(b1, b2, mx, nrhs, "gelsy b");
    }
    // xGERQF, xGGQRF / xGGRQF (A m x n, B m x 3 / 3 x n).
    {
      std::vector<T> a1 = ac, a2 = ar, t1(ld), t2(ld);
      ASSERT_EQ(LAPACKE_CALL(T, gerqf, LAPACK_COL_MAJOR, m, n, a1.data(), ld,
                             t1.data()),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, gerqf, LAPACK_ROW_MAJOR, m, n, a2.data(), ld,
                             t2.data()),
                0);
      EXPECT_EQ(t1, t2);
      same(a1, a2, m, n, "gerqf");
    }
    {
      std::vector<T> qc, qr;
      std::tie(qc, qr) = both(m, 3);
      std::vector<T> a1 = ac, a2 = ar, b1 = qc, b2 = qr, ta1(ld), ta2(ld),
                     tb1(ld), tb2(ld);
      // ggqrf( N = m, M = n, P = 3 ): A m x n, B m x 3.
      ASSERT_EQ(LAPACKE_CALL(T, ggqrf, LAPACK_COL_MAJOR, m, n, 3, a1.data(), ld,
                             ta1.data(), b1.data(), ld, tb1.data()),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, ggqrf, LAPACK_ROW_MAJOR, m, n, 3, a2.data(), ld,
                             ta2.data(), b2.data(), ld, tb2.data()),
                0);
      EXPECT_EQ(ta1, ta2);
      EXPECT_EQ(tb1, tb2);
      same(a1, a2, m, n, "ggqrf a");
      same(b1, b2, m, 3, "ggqrf b");
    }
    {
      std::vector<T> qc, qr;
      std::tie(qc, qr) = both(3, n);
      std::vector<T> a1 = ac, a2 = ar, b1 = qc, b2 = qr, ta1(ld), ta2(ld),
                     tb1(ld), tb2(ld);
      // ggrqf( M = m, P = 3, N = n ): A m x n, B 3 x n.
      ASSERT_EQ(LAPACKE_CALL(T, ggrqf, LAPACK_COL_MAJOR, m, 3, n, a1.data(), ld,
                             ta1.data(), b1.data(), ld, tb1.data()),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, ggrqf, LAPACK_ROW_MAJOR, m, 3, n, a2.data(), ld,
                             ta2.data(), b2.data(), ld, tb2.data()),
                0);
      EXPECT_EQ(ta1, ta2);
      EXPECT_EQ(tb1, tb2);
      same(a1, a2, m, n, "ggrqf a");
      same(b1, b2, 3, n, "ggrqf b");
    }
    // xORMRQ / xUNMRQ and xORMRZ / xUNMRZ: K = 3 reflectors (from xGERQF
    // and xTZRZF of a 3 x nq matrix) applied to C (m x n) from both sides.
    for (const char side : {'L', 'R'}) {
      const lapack_int nq = side == 'L' ? m : n, k = 3;
      std::vector<T> vc, vr;
      std::tie(vc, vr) = both(k, nq);
      std::vector<T> tq(ld), tz(ld), vz = vc;
      ASSERT_EQ(LAPACKE_CALL(T, gerqf, LAPACK_COL_MAJOR, k, nq, vc.data(), ld,
                             tq.data()),
                0);
      for (lapack_int j = 0; j < nq; ++j) {
        for (lapack_int i = j + 1; i < k; ++i) At(vz, i, j, ld) = T(0);
      }
      ASSERT_EQ(LAPACKE_CALL(T, tzrzf, LAPACK_COL_MAJOR, k, nq, vz.data(), ld,
                             tz.data()),
                0);
      const std::vector<T> vqr = Transposed(vc, ld), vzr = Transposed(vz, ld);
      for (const char trans : {'N', ctrans}) {
        std::vector<T> cc, cr;
        std::tie(cc, cr) = both(m, n);
        std::vector<T> c1 = cc, c2 = cr;
        ASSERT_EQ(
            LAPACKE_CALL_RC(T, ormrq, unmrq, LAPACK_COL_MAJOR, side, trans, m,
                            n, k, vc.data(), ld, tq.data(), c1.data(), ld),
            0);
        ASSERT_EQ(
            LAPACKE_CALL_RC(T, ormrq, unmrq, LAPACK_ROW_MAJOR, side, trans, m,
                            n, k, vqr.data(), ld, tq.data(), c2.data(), ld),
            0);
        same(c1, c2, m, n, "ormrq");
        c1 = cc;
        c2 = cr;
        ASSERT_EQ(LAPACKE_CALL_RC(T, ormrz, unmrz, LAPACK_COL_MAJOR, side,
                                  trans, m, n, k, nq - k, vz.data(), ld,
                                  tz.data(), c1.data(), ld),
                  0);
        ASSERT_EQ(LAPACKE_CALL_RC(T, ormrz, unmrz, LAPACK_ROW_MAJOR, side,
                                  trans, m, n, k, nq - k, vzr.data(), ld,
                                  tz.data(), c2.data(), ld),
                  0);
        same(c1, c2, m, n, "ormrz");
      }
    }
  }
  // xGGLSE (A 6 x 4, B 2 x 4) and xGGGLM (A 5 x 3, B 5 x 4).
  {
    std::vector<T> ac, ar;
    std::tie(ac, ar) = both(6, 4);
    std::vector<T> bc, br;
    std::tie(bc, br) = both(2, 4);
    const std::vector<T> c0 = rng.Vector(6), d0 = rng.Vector(2);
    std::vector<T> c1 = c0, c2 = c0, d1 = d0, d2 = d0, x1(4), x2(4);
    ASSERT_EQ(LAPACKE_CALL(T, gglse, LAPACK_COL_MAJOR, 6, 4, 2, ac.data(), ld,
                           bc.data(), ld, c1.data(), d1.data(), x1.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gglse, LAPACK_ROW_MAJOR, 6, 4, 2, ar.data(), ld,
                           br.data(), ld, c2.data(), d2.data(), x2.data()),
              0);
    EXPECT_EQ(x1, x2);
    EXPECT_EQ(c1, c2);
    same(ac, ar, 6, 4, "gglse a");
  }
  {
    std::vector<T> ac, ar;
    std::tie(ac, ar) = both(5, 3);
    std::vector<T> bc, br;
    std::tie(bc, br) = both(5, 4);
    const std::vector<T> d0 = rng.Vector(5);
    std::vector<T> d1 = d0, d2 = d0, x1(3), x2(3), y1(4), y2(4);
    ASSERT_EQ(LAPACKE_CALL(T, ggglm, LAPACK_COL_MAJOR, 5, 3, 4, ac.data(), ld,
                           bc.data(), ld, d1.data(), x1.data(), y1.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, ggglm, LAPACK_ROW_MAJOR, 5, 3, 4, ar.data(), ld,
                           br.data(), ld, d2.data(), x2.data(), y2.data()),
              0);
    EXPECT_EQ(x1, x2);
    EXPECT_EQ(y1, y2);
    same(bc, br, 5, 4, "ggglm b");
  }
}

// Invalid layouts and leading dimensions, and the NaN checks.
TYPED_TEST(GelsTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  constexpr bool kComplex = IsComplex<T>::value;
  std::vector<T> a(64, T(0)), b(64, T(0)), c(64, T(0)), tau(8, T(0));
  for (lapack_int i = 0; i < 4; ++i) a[i * 5] = T(1);
  std::vector<R> s(8);
  std::vector<lapack_int> jpvt(8, 0);
  lapack_int rank = -1;
  const R rcond = -1, nan = NAN;
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LAPACKE_CALL(T, gels, 0, 'N', 4, 4, 1, a.data(), 4, b.data(), 4),
            -1);
  EXPECT_EQ(LAPACKE_CALL(T, gels, LAPACK_ROW_MAJOR, 'N', 4, 3, 1, a.data(), 2,
                         b.data(), 1),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, gelss, LAPACK_ROW_MAJOR, 4, 3, 2, a.data(), 3,
                         b.data(), 1, s.data(), rcond, &rank),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, gelsd, LAPACK_ROW_MAJOR, 4, 3, 1, a.data(), 2,
                         b.data(), 1, s.data(), rcond, &rank),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, gelsy, LAPACK_ROW_MAJOR, 4, 3, 1, a.data(), 2,
                         b.data(), 1, jpvt.data(), rcond, &rank),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, gglse, LAPACK_ROW_MAJOR, 4, 3, 2, a.data(), 3,
                         b.data(), 2, c.data(), c.data() + 8, c.data() + 16),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, ggglm, LAPACK_ROW_MAJOR, 4, 2, 3, a.data(), 1,
                         b.data(), 3, c.data(), c.data() + 8, c.data() + 16),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, ggqrf, LAPACK_ROW_MAJOR, 4, 2, 3, a.data(), 2,
                         tau.data(), b.data(), 2, tau.data()),
            -9);
  EXPECT_EQ(LAPACKE_CALL(T, ggrqf, LAPACK_ROW_MAJOR, 4, 2, 3, a.data(), 2,
                         tau.data(), b.data(), 3, tau.data()),
            -6);
  EXPECT_EQ(
      LAPACKE_CALL(T, gerqf, LAPACK_ROW_MAJOR, 4, 3, a.data(), 2, tau.data()),
      -5);
  EXPECT_EQ(
      LAPACKE_CALL(T, tzrzf, LAPACK_ROW_MAJOR, 2, 3, a.data(), 2, tau.data()),
      -5);
  // A is K x N for SIDE = 'R' (N = 5 > M = 4): LDA = 4 is too small.
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormrq, unmrq, LAPACK_ROW_MAJOR, 'R', 'N', 4, 5,
                            2, a.data(), 4, tau.data(), c.data(), 5),
            -8);
  EXPECT_EQ(LAPACKE_CALL_RC(T, ormrz, unmrz, LAPACK_ROW_MAJOR, 'L', 'N', 4, 3,
                            2, 1, a.data(), 4, tau.data(), c.data(), 2),
            -12);
  const std::string rq = kComplex ? "unmrq_work" : "ormrq_work";
  const std::string rz = kComplex ? "unmrz_work" : "ormrz_work";
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("gels") +
                "\nWrong parameter 7 in " + LapackeName<T>("gels_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("gelss_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("gelsd_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("gelsy_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("gglse_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("ggglm_work") +
                "\nWrong parameter 9 in " + LapackeName<T>("ggqrf_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("ggrqf_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("gerqf_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("tzrzf_work") +
                "\nWrong parameter 8 in " + LapackeName<T>(rq.c_str()) +
                "\nWrong parameter 12 in " + LapackeName<T>(rz.c_str()) + "\n");

  // NaN checks: A, B and RCOND of the least squares drivers.
  const auto check = [&](int which, lapack_int expected_a,
                         lapack_int expected_b) {
    const auto call = [&]() {
      switch (which) {
        case 0:
          return LAPACKE_CALL(T, gels, LAPACK_COL_MAJOR, 'N', 4, 4, 1, a.data(),
                              4, b.data(), 4);
        case 1:
          return LAPACKE_CALL(T, gelss, LAPACK_COL_MAJOR, 4, 4, 1, a.data(), 4,
                              b.data(), 4, s.data(), rcond, &rank);
        case 2:
          return LAPACKE_CALL(T, gelsd, LAPACK_COL_MAJOR, 4, 4, 1, a.data(), 4,
                              b.data(), 4, s.data(), rcond, &rank);
        default:
          return LAPACKE_CALL(T, gelsy, LAPACK_COL_MAJOR, 4, 4, 1, a.data(), 4,
                              b.data(), 4, jpvt.data(), rcond, &rank);
      }
    };
    a[5] = NaN<T>();
    EXPECT_EQ(call(), expected_a) << which;
    a[5] = T(1);
    b[3] = NaN<T>();
    EXPECT_EQ(call(), expected_b) << which;
    b[3] = T(0);
    EXPECT_EQ(call(), 0) << which;
  };
  check(0, -6, -8);
  for (int which = 1; which < 4; ++which) check(which, -5, -7);
  EXPECT_EQ(LAPACKE_CALL(T, gelss, LAPACK_COL_MAJOR, 4, 4, 1, a.data(), 4,
                         b.data(), 4, s.data(), nan, &rank),
            -10);
  EXPECT_EQ(LAPACKE_CALL(T, gelsy, LAPACK_COL_MAJOR, 4, 4, 1, a.data(), 4,
                         b.data(), 4, jpvt.data(), nan, &rank),
            -10);
  // xGGLSE: A -5, B -7, C -9, D -10; xGGGLM: A -5, B -7, D -9.
  {
    std::vector<T> aa = a, bb(16, T(1)), cc(4, T(1)), dd(4, T(1)), xx(4);
    for (lapack_int i = 0; i < 2; ++i) bb[i * 2 + i] = T(2);
    const auto gglse = [&]() {
      return LAPACKE_CALL(T, gglse, LAPACK_COL_MAJOR, 4, 3, 2, aa.data(), 4,
                          bb.data(), 2, cc.data(), dd.data(), xx.data());
    };
    cc[3] = NaN<T>();
    EXPECT_EQ(gglse(), -9);
    cc[3] = T(1);
    dd[1] = NaN<T>();
    EXPECT_EQ(gglse(), -10);
    dd[1] = T(1);
    bb[5] = NaN<T>();
    EXPECT_EQ(gglse(), -7);
    bb[6] = NaN<T>();  // outside B (2 x 3): row 0 of column 3
    bb[5] = T(1);
    EXPECT_EQ(gglse(), 0);
  }
  // xORMRQ / xUNMRQ with SIDE = 'R': A is K x N (column 4 checked for N =
  // 5 > M = 4; the reference checks K x M).
  {
    std::vector<T> aa(2 * 5, T(0)), cc(4 * 5, T(1)), tt(2, T(0));
    aa[2 * 4] = NaN<T>();
    EXPECT_EQ(LAPACKE_CALL_RC(T, ormrq, unmrq, LAPACK_COL_MAJOR, 'R', 'N', 4, 5,
                              2, aa.data(), 2, tt.data(), cc.data(), 4),
              -7);
    EXPECT_EQ(LAPACKE_CALL_RC(T, ormrz, unmrz, LAPACK_COL_MAJOR, 'R', 'N', 4, 5,
                              2, 3, aa.data(), 2, tt.data(), cc.data(), 4),
              -8);
  }
}

}  // namespace
}  // namespace hwyblas_test
