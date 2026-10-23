// Tests of the QL factorization (xGEQLF / xGEQL2), the Q of the RQ
// factorization (xORGRQ / xUNGRQ), the QR factorization in the compact WY
// representation (xGEQRT, xGEQRT2, xGEQRT3) and its multiplication
// (xGEMQRT), and their LAPACKE functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/eigen_test_util.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class GeqrtTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GeqrtTest, ScalarTypes);

// Calls LAPACKE_x##NAME of T's precision.
#define LAPACKE_CALL(T, NAME, ...) LAPACKE_CALL_RC(T, NAME, NAME, __VA_ARGS__)

template <typename T>
char ConjTrans() {
  return IsComplex<T>::value ? 'C' : 'T';
}

template <typename T>
bool IsNan(const T& x) {
  return std::isnan(std::real(x)) || std::isnan(std::imag(x));
}

struct Shape {
  hwyblas_int m, n;
};
const Shape kShapes[] = {{0, 3},   {3, 0},   {1, 1},   {1, 5},   {5, 1},
                         {2, 2},   {3, 7},   {7, 3},   {10, 10}, {33, 17},
                         {17, 33}, {40, 40}, {45, 50}, {60, 25}};

// The block sizes nb and crossover points nx of the tests of the blocked
// routines: ILAENV's, and small ones with blocked code down to the end
// (nx = 0) or not (nx = 5).
struct Blocking {
  hwyblas_int nb, nx;
};
const Blocking kBlockings[] = {{0, 0}, {1, 0}, {2, 0}, {3, 5}, {8, 0}, {8, 5}};

// LWORK of the blocked calls: the optimal one (from a workspace query) or
// the minimum.
enum class Lwork { kOptimal, kMinimum };

// The elementary reflector I - tau * v * v**H (order n) in dense form.
template <typename T>
Dense<T> Reflector(const Dense<T>& v, RefT<T> tau) {
  Dense<T> h = Identity<T>(v.rows);
  for (size_t j = 0; j < v.rows; ++j) {
    for (size_t i = 0; i < v.rows; ++i)
      h(i, j) -= tau * v(i, 0) * Conj(v(j, 0));
  }
  return h;
}

// The m x m matrix Q = H(k) . . . H(2) H(1) of a QL factorization of the
// m x n matrix A (k = min(m,n)): reflector i (1-based) has v(m-k+i) = 1,
// v(m-k+i+1:m) = 0 and v(1:m-k+i-1) in A(1:m-k+i-1,n-k+i).
template <typename T>
Dense<T> QlQ(hwyblas_int m, hwyblas_int n, const std::vector<T>& af,
             hwyblas_int lda, const std::vector<T>& tau) {
  const hwyblas_int k = std::min(m, n);
  Dense<T> q = Identity<T>(m);
  for (hwyblas_int i = 0; i < k; ++i) {
    Dense<T> v(m, 1);
    const hwyblas_int row = m - k + i, col = n - k + i;
    v(row, 0) = 1;
    for (hwyblas_int r = 0; r < row; ++r) {
      v(r, 0) = static_cast<RefT<T>>(At(af, r, col, lda));
    }
    // Q = H(k) ... H(1): H(i) is applied after H(1) ... H(i-1).
    q = Mul(Reflector(v, static_cast<RefT<T>>(tau[i])), q);
  }
  return q;
}

// The n x n matrix Q = H(1)**H H(2)**H . . . H(k)**H of the k reflectors of
// an RQ factorization in the last k of the m rows of af (n columns), with
// the scalars tau[0 ... k-1]: reflector i (1-based) has v(n-k+i) = 1,
// v(n-k+i+1:n) = 0 and conj(v(1:n-k+i-1)) in A(m-k+i,1:n-k+i-1).
template <typename T>
Dense<T> RqQ(hwyblas_int m, hwyblas_int n, hwyblas_int k,
             const std::vector<T>& af, hwyblas_int lda, const T* tau) {
  Dense<T> q = Identity<T>(n);
  for (hwyblas_int i = 0; i < k; ++i) {
    Dense<T> v(n, 1);
    const hwyblas_int row = m - k + i, col = n - k + i;
    v(col, 0) = 1;
    for (hwyblas_int c = 0; c < col; ++c) {
      v(c, 0) = Conj(static_cast<RefT<T>>(At(af, row, c, lda)));
    }
    q = Mul(q, Reflector(v, Conj(static_cast<RefT<T>>(tau[i]))));
  }
  return q;
}

// The q x q matrix Q = H(1) H(2) . . . H(b) of k reflectors in the compact
// WY representation of xGEQRT with block size nb: H(j) = I - V * T * V**H
// for the block of columns i0 ... i0+ib-1 of v (unit diagonal, zeros above,
// the elements below the diagonal of v) and the ib x ib upper triangle of
// the columns i0 ... i0+ib-1 of t.
template <typename T>
Dense<T> WyQ(hwyblas_int q, hwyblas_int k, hwyblas_int nb,
             const std::vector<T>& v, hwyblas_int ldv, const std::vector<T>& t,
             hwyblas_int ldt) {
  Dense<T> result = Identity<T>(q);
  for (hwyblas_int i0 = 0; i0 < k; i0 += nb) {
    const hwyblas_int ib = std::min(nb, k - i0);
    Dense<T> vb(q, ib), tb(ib, ib);
    for (hwyblas_int c = 0; c < ib; ++c) {
      vb(i0 + c, c) = 1;
      for (hwyblas_int r = i0 + c + 1; r < q; ++r) {
        vb(r, c) = static_cast<RefT<T>>(At(v, r, i0 + c, ldv));
      }
      for (hwyblas_int r = 0; r <= c; ++r) {
        tb(r, c) = static_cast<RefT<T>>(At(t, r, i0 + c, ldt));
      }
    }
    result = Mul(result, Sub(Identity<T>(q), Mul(Mul(vb, tb), Adjoint(vb))));
  }
  return result;
}

// The m x n matrix of the elements (i, j) of af with keep(i, j), zeros
// elsewhere.
template <typename T, typename Keep>
Dense<T> Part(hwyblas_int m, hwyblas_int n, const std::vector<T>& af,
              hwyblas_int lda, const Keep& keep) {
  Dense<T> r(m, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      if (keep(i, j)) r(i, j) = static_cast<RefT<T>>(At(af, i, j, lda));
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// QL factorization

// xGEQLF (blocked) or xGEQL2.
template <typename T>
hwyblas_int Geqlf(bool blocked, Lwork lwork_kind, hwyblas_int m, hwyblas_int n,
                  std::vector<T>& a, hwyblas_int lda, std::vector<T>& tau) {
  hwyblas_int info = -99;
  if (!blocked) {
    std::vector<T> work(std::max<hwyblas_int>(1, n), NaN<T>());
    HWYBLAS_CALL(T, geql2, &m, &n, P(a.data()), &lda, P(tau.data()),
                 P(work.data()), &info);
    return info;
  }
  hwyblas_int lwork = -1;
  T query = T(0);
  HWYBLAS_CALL(T, geqlf, &m, &n, P(a.data()), &lda, P(tau.data()), P(&query),
               &lwork, &info);
  if (info != 0) return info;
  const hwyblas_int minimum = std::max<hwyblas_int>(1, n);
  lwork = lwork_kind == Lwork::kOptimal ? std::max(QueryResult(query), minimum)
                                        : minimum;
  std::vector<T> work(lwork, NaN<T>());
  HWYBLAS_CALL(T, geqlf, &m, &n, P(a.data()), &lda, P(tau.data()),
               P(work.data()), &lwork, &info);
  return info;
}

// A = Q * L with unitary Q, for xGEQL2 and xGEQLF (several block sizes and
// both workspace sizes); the rows of the array below A are not changed.
TYPED_TEST(GeqrtTest, QlFactorization) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(51);
    for (const bool blocked : {false, true}) {
      for (const Blocking& b : kBlockings) {
        if (!blocked && b.nb != 0) continue;
        const ScopedBlockSize block_size(b.nb, b.nx);
        for (const Lwork lwork : {Lwork::kOptimal, Lwork::kMinimum}) {
          if (!blocked && lwork != Lwork::kOptimal) continue;
          for (const Shape& s : kShapes) {
            const hwyblas_int m = s.m, n = s.n, lda = m + 2;
            SCOPED_TRACE(::testing::Message()
                         << (blocked ? "blocked" : "unblocked") << " m=" << m
                         << " n=" << n << " nb=" << b.nb << " nx=" << b.nx
                         << " lwork=" << static_cast<int>(lwork));
            const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) *
                                                std::max<hwyblas_int>(1, n));
            std::vector<T> af = a;
            std::vector<T> tau(std::max<hwyblas_int>(1, std::min(m, n)));
            ASSERT_EQ(Geqlf(blocked, lwork, m, n, af, lda, tau), 0);
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = m; i < lda; ++i) {
                ASSERT_EQ(At(af, i, j, lda), At(a, i, j, lda));
              }
            }
            if (m == 0 || n == 0) continue;
            // L: on and below the (n-m)-th superdiagonal.
            const Dense<T> l = Part(
                m, n, af, lda,
                [&](hwyblas_int i, hwyblas_int j) { return j - i <= n - m; });
            const Dense<T> q = QlQ(m, n, af, lda, tau);
            EXPECT_LT(ResidualRatio(ToDense(a, m, n, lda), Mul(q, l), m),
                      kLapackThreshold);
            EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// The Q of the RQ factorization

// xORGRQ / xUNGRQ (blocked) or xORGR2 / xUNGR2.
template <typename T>
hwyblas_int Ungrq(bool blocked, Lwork lwork_kind, hwyblas_int m, hwyblas_int n,
                  hwyblas_int k, std::vector<T>& a, hwyblas_int lda,
                  const T* tau) {
  hwyblas_int info = -99;
  if (!blocked) {
    std::vector<T> work(std::max<hwyblas_int>(1, m), NaN<T>());
    CALL_RC(T, orgr2, ungr2, &m, &n, &k, P(a.data()), &lda, P(tau),
            P(work.data()), &info);
    return info;
  }
  hwyblas_int lwork = -1;
  T query = T(0);
  CALL_RC(T, orgrq, ungrq, &m, &n, &k, P(a.data()), &lda, P(tau), P(&query),
          &lwork, &info);
  if (info != 0) return info;
  const hwyblas_int minimum = std::max<hwyblas_int>(1, m);
  lwork = lwork_kind == Lwork::kOptimal ? std::max(QueryResult(query), minimum)
                                        : minimum;
  std::vector<T> work(lwork, NaN<T>());
  CALL_RC(T, orgrq, ungrq, &m, &n, &k, P(a.data()), &lda, P(tau),
          P(work.data()), &lwork, &info);
  return info;
}

// The n x n matrix Q of an RQ factorization (xGERQF) generated from all
// min(m,n) reflectors gives A = R * Q; with m <= n, the m x n matrix of
// the last k reflectors (k = 0, 1, m/2, m) is the last m rows of their
// product, which has orthonormal rows.
TYPED_TEST(GeqrtTest, RqGenerateQ) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(53);
    for (const bool blocked : {false, true}) {
      for (const Blocking& b : kBlockings) {
        if (!blocked && b.nb != 0) continue;
        for (const Lwork lwork : {Lwork::kOptimal, Lwork::kMinimum}) {
          if (!blocked && lwork != Lwork::kOptimal) continue;
          for (const Shape& s : kShapes) {
            const hwyblas_int m = s.m, n = s.n, lda = std::max(m, n) + 2;
            const hwyblas_int k = std::min(m, n);
            SCOPED_TRACE(::testing::Message()
                         << (blocked ? "blocked" : "unblocked") << " m=" << m
                         << " n=" << n << " nb=" << b.nb << " nx=" << b.nx
                         << " lwork=" << static_cast<int>(lwork));
            const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) *
                                                std::max<hwyblas_int>(1, n));
            std::vector<T> af = a;
            std::vector<T> tau(std::max<hwyblas_int>(1, k));
            {
              hwyblas_int info = -99, lw = std::max<hwyblas_int>(1, m) * 64;
              std::vector<T> work(lw);
              HWYBLAS_CALL(T, gerqf, &m, &n, P(af.data()), &lda, P(tau.data()),
                           P(work.data()), &lw, &info);
              ASSERT_EQ(info, 0);
            }
            const ScopedBlockSize block_size(b.nb, b.nx);

            // The full n x n Q: the reflectors in the last k rows.
            const hwyblas_int ldq = n + 1;
            std::vector<T> q(static_cast<size_t>(ldq) * n, NaN<T>());
            for (hwyblas_int r = 0; r < k; ++r) {
              for (hwyblas_int j = 0; j < n; ++j) {
                At(q, n - k + r, j, ldq) = At(af, m - k + r, j, lda);
              }
            }
            ASSERT_EQ(Ungrq(blocked, lwork, n, n, k, q, ldq, tau.data()), 0);
            if (n > 0) {
              const Dense<T> qd = ToDense(q, n, n, ldq);
              const Dense<T> qref = RqQ(m, n, k, af, lda, tau.data());
              EXPECT_LT(ResidualRatio(qref, qd, n), kLapackThreshold);
              EXPECT_LT(OrthogonalityRatio(qd), kLapackThreshold);
              if (m > 0) {
                // R: on and above the (m-n)-th subdiagonal.
                const Dense<T> r =
                    Part(m, n, af, lda, [&](hwyblas_int i, hwyblas_int j) {
                      return j - i >= n - m;
                    });
                EXPECT_LT(ResidualRatio(ToDense(a, m, n, lda), Mul(r, qd), n),
                          kLapackThreshold);
              }
            }

            // The m x n matrices of the last kp reflectors.
            if (m > n) continue;
            for (const hwyblas_int kp :
                 {hwyblas_int{0}, std::min<hwyblas_int>(1, m), m / 2, m}) {
              SCOPED_TRACE(::testing::Message() << "k=" << kp);
              std::vector<T> q1 = af;
              ASSERT_EQ(Ungrq(blocked, lwork, m, n, kp, q1, lda,
                              tau.data() + (k - kp)),
                        0);
              const Dense<T> full =
                  RqQ(m, n, kp, af, lda, tau.data() + (k - kp));
              Dense<T> last(m, n);
              for (hwyblas_int j = 0; j < n; ++j) {
                for (hwyblas_int i = 0; i < m; ++i) {
                  last(i, j) = full(n - m + i, j);
                }
              }
              const Dense<T> q1d = ToDense(q1, m, n, lda);
              EXPECT_LT(ResidualRatio(last, q1d, n), kLapackThreshold);
              EXPECT_LT(OrthogonalityRatio(Adjoint(q1d)), kLapackThreshold);
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// QR factorization in the compact WY representation

// xGEQRT for all block sizes: A = Q * R with Q of the reflectors and
// triangular factors; T is written only in the upper triangles of the
// blocks (NaN stays elsewhere).
TYPED_TEST(GeqrtTest, GeqrtFactorization) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(55);
    for (const Shape& s : kShapes) {
      const hwyblas_int m = s.m, n = s.n, k = std::min(m, n);
      for (const hwyblas_int nb : {hwyblas_int{1}, hwyblas_int{2},
                                   hwyblas_int{3}, hwyblas_int{8}, k}) {
        if (nb < 1 || (k > 0 && nb > k)) continue;
        SCOPED_TRACE(::testing::Message()
                     << "m=" << m << " n=" << n << " nb=" << nb);
        const hwyblas_int lda = m + 2, ldt = nb + 1;
        const std::vector<T> a =
            rng.Vector(static_cast<size_t>(lda) * std::max<hwyblas_int>(1, n));
        std::vector<T> af = a;
        std::vector<T> t(static_cast<size_t>(ldt) * std::max<hwyblas_int>(1, k),
                         NaN<T>());
        std::vector<T> work(
            static_cast<size_t>(nb) * std::max<hwyblas_int>(1, n), NaN<T>());
        hwyblas_int info = -99;
        HWYBLAS_CALL(T, geqrt, &m, &n, &nb, P(af.data()), &lda, P(t.data()),
                     &ldt, P(work.data()), &info);
        ASSERT_EQ(info, 0);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = m; i < lda; ++i) {
            ASSERT_EQ(At(af, i, j, lda), At(a, i, j, lda));
          }
        }
        for (hwyblas_int j = 0; j < k; ++j) {
          const hwyblas_int c = j % nb;  // column within its block
          for (hwyblas_int i = 0; i < ldt; ++i) {
            EXPECT_EQ(IsNan(At(t, i, j, ldt)), i > c) << i << ", " << j;
          }
        }
        if (k == 0) continue;
        const Dense<T> q = WyQ(m, k, nb, af, lda, t, ldt);
        const Dense<T> r = Part(
            m, n, af, lda, [](hwyblas_int i, hwyblas_int j) { return i <= j; });
        EXPECT_LT(ResidualRatio(ToDense(a, m, n, lda), Mul(q, r), m),
                  kLapackThreshold);
        EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);
      }
    }
  });
}

// xGEQRT2 and xGEQRT3 (M >= N): A = Q * R with Q = I - V * T * V**H.
// xGEQRT3 does not write the strictly lower triangle of T; xGEQRT2 zeros
// its first column there (where it keeps the scalars of the reflectors).
TYPED_TEST(GeqrtTest, Geqrt2AndGeqrt3) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(57);
    for (const bool recursive : {false, true}) {
      for (const Shape& s : kShapes) {
        const hwyblas_int m = s.m, n = s.n;
        if (m < n) continue;
        SCOPED_TRACE(::testing::Message() << (recursive ? "GEQRT3" : "GEQRT2")
                                          << " m=" << m << " n=" << n);
        const hwyblas_int lda = m + 2, ldt = n + 1;
        const std::vector<T> a =
            rng.Vector(static_cast<size_t>(lda) * std::max<hwyblas_int>(1, n));
        std::vector<T> af = a;
        std::vector<T> t(static_cast<size_t>(ldt) * std::max<hwyblas_int>(1, n),
                         NaN<T>());
        hwyblas_int info = -99;
        if (recursive) {
          HWYBLAS_CALL(T, geqrt3, &m, &n, P(af.data()), &lda, P(t.data()), &ldt,
                       &info);
        } else {
          HWYBLAS_CALL(T, geqrt2, &m, &n, P(af.data()), &lda, P(t.data()), &ldt,
                       &info);
        }
        ASSERT_EQ(info, 0);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < ldt; ++i) {
            if (i <= j) {
              EXPECT_FALSE(IsNan(At(t, i, j, ldt))) << i << ", " << j;
            } else if (!recursive && j == 0 && i < n) {
              EXPECT_EQ(At(t, i, j, ldt), T(0)) << i << ", " << j;
            } else {
              EXPECT_TRUE(IsNan(At(t, i, j, ldt))) << i << ", " << j;
            }
          }
        }
        if (n == 0) continue;
        const Dense<T> q = WyQ(m, n, n, af, lda, t, ldt);
        const Dense<T> r = Part(
            m, n, af, lda, [](hwyblas_int i, hwyblas_int j) { return i <= j; });
        EXPECT_LT(ResidualRatio(ToDense(a, m, n, lda), Mul(q, r), m),
                  kLapackThreshold);
        EXPECT_LT(OrthogonalityRatio(q), kLapackThreshold);
      }
    }
  });
}

// xGEMQRT: Q * C, Q**H * C, C * Q and C * Q**H against the explicit Q of
// xGEQRT's reflectors (k <= q of them, in blocks of nb), for C of both
// orientations; the work array has the documented size, and the elements
// on and above the diagonal of V are not read.
TYPED_TEST(GeqrtTest, Gemqrt) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(59);
    const char trans_c = ConjTrans<T>();
    struct Case {
      hwyblas_int q, k, other;  // order of Q, reflectors, other dimension of C
    };
    const Case cases[] = {{1, 1, 3},  {5, 0, 4},    {5, 3, 1},   {6, 6, 9},
                          {17, 9, 4}, {20, 13, 31}, {33, 33, 12}};
    for (const Case& cs : cases) {
      for (const hwyblas_int nb :
           {hwyblas_int{1}, hwyblas_int{2}, hwyblas_int{4}, cs.k}) {
        if (nb < 1 || (cs.k > 0 && nb > cs.k)) continue;
        // The reflectors: xGEQRT of a q x k matrix.
        const hwyblas_int ldv = cs.q + 1, ldt = nb + 2;
        std::vector<T> v = rng.Vector(static_cast<size_t>(ldv) *
                                      std::max<hwyblas_int>(1, cs.k));
        std::vector<T> t(
            static_cast<size_t>(ldt) * std::max<hwyblas_int>(1, cs.k),
            NaN<T>());
        {
          std::vector<T> work(static_cast<size_t>(nb) *
                              std::max<hwyblas_int>(1, cs.k));
          hwyblas_int info = -99;
          HWYBLAS_CALL(T, geqrt, &cs.q, &cs.k, &nb, P(v.data()), &ldv,
                       P(t.data()), &ldt, P(work.data()), &info);
          ASSERT_EQ(info, 0);
        }
        for (hwyblas_int j = 0; j < cs.k; ++j) {
          for (hwyblas_int i = 0; i <= j; ++i) At(v, i, j, ldv) = NaN<T>();
        }
        const Dense<T> q = WyQ(cs.q, cs.k, nb, v, ldv, t, ldt);
        for (const char side : {'L', 'R'}) {
          for (const char trans : {'N', trans_c}) {
            const hwyblas_int m = side == 'L' ? cs.q : cs.other;
            const hwyblas_int n = side == 'L' ? cs.other : cs.q;
            SCOPED_TRACE(::testing::Message()
                         << "side=" << side << " trans=" << trans
                         << " q=" << cs.q << " k=" << cs.k << " nb=" << nb
                         << " other=" << cs.other);
            const hwyblas_int ldc = m + 3;
            const std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) *
                                                std::max<hwyblas_int>(1, n));
            std::vector<T> cf = c;
            std::vector<T> work(static_cast<size_t>(std::max<hwyblas_int>(
                                    1, side == 'L' ? n : m)) *
                                    nb,
                                NaN<T>());
            hwyblas_int info = -99;
            HWYBLAS_CALL(T, gemqrt, &side, &trans, &m, &n, &cs.k, &nb,
                         P(v.data()), &ldv, P(t.data()), &ldt, P(cf.data()),
                         &ldc, P(work.data()), &info);
            ASSERT_EQ(info, 0);
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = m; i < ldc; ++i) {
                ASSERT_EQ(At(cf, i, j, ldc), At(c, i, j, ldc));
              }
            }
            const Dense<T> qop = trans == 'N' ? q : Adjoint(q);
            const Dense<T> cd = ToDense(c, m, n, ldc);
            const Dense<T> expected = side == 'L' ? Mul(qop, cd) : Mul(cd, qop);
            EXPECT_LT(ResidualRatio(expected, ToDense(cf, m, n, ldc), cs.q),
                      kLapackThreshold);
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Arguments

// Workspace queries return the documented optimal sizes.
TYPED_TEST(GeqrtTest, WorkspaceQueries) {
  using T = TypeParam;
  const ScopedBlockSize block_size(8);
  const hwyblas_int minus1 = -1, zero = 0, lda = 30;
  std::vector<T> a(lda * 30), tau(30);
  T work = T(0);
  hwyblas_int info = -99;
  {
    const hwyblas_int m = 30, n = 20;
    HWYBLAS_CALL(T, geqlf, &m, &n, P(a.data()), &lda, P(tau.data()), P(&work),
                 &minus1, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(work), n * 8);
    HWYBLAS_CALL(T, geqlf, &zero, &n, P(a.data()), &lda, P(tau.data()),
                 P(&work), &minus1, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(work), 1);
  }
  {
    const hwyblas_int m = 20, n = 30, k = 10;
    CALL_RC(T, orgrq, ungrq, &m, &n, &k, P(a.data()), &lda, P(tau.data()),
            P(&work), &minus1, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(work), m * 8);
    CALL_RC(T, orgrq, ungrq, &zero, &n, &zero, P(a.data()), &lda, P(tau.data()),
            P(&work), &minus1, &info);
    EXPECT_EQ(info, 0);
    EXPECT_EQ(QueryResult(work), 1);
  }
}

// Each invalid argument in turn (the first one that LAPACK checks).
TYPED_TEST(GeqrtTest, InvalidArguments) {
  using T = TypeParam;
  std::vector<T> a(64, T(1)), tau(8, T(1)), t(64, T(1)), c(64, T(1)),
      work(64, T(1));
  ErrorCapture errors;
  const auto expect = [&errors](hwyblas_int info, hwyblas_int expected,
                                const std::string& name) {
    EXPECT_EQ(info, -expected) << name;
    EXPECT_EQ(errors.Take(),
              std::vector<std::string>{name + ":" + std::to_string(expected)});
  };

  // m, n, lda (and lwork) of xGEQL2 and xGEQLF.
  const auto geqlf = [&](bool blocked, hwyblas_int m, hwyblas_int n,
                         hwyblas_int lda, hwyblas_int lwork) {
    hwyblas_int info = -99;
    if (blocked) {
      HWYBLAS_CALL(T, geqlf, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &lwork, &info);
    } else {
      HWYBLAS_CALL(T, geql2, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &info);
    }
    return info;
  };
  for (const bool blocked : {false, true}) {
    const std::string name = Name<T>(blocked ? "GEQLF" : "GEQL2");
    expect(geqlf(blocked, -1, 4, 4, 64), 1, name);
    expect(geqlf(blocked, 4, -1, 4, 64), 2, name);
    expect(geqlf(blocked, 4, 4, 3, 64), 4, name);
  }
  expect(geqlf(true, 4, 5, 4, 4), 7, Name<T>("GEQLF"));  // lwork < n
  expect(geqlf(true, 4, 4, 4, 0), 7, Name<T>("GEQLF"));
  EXPECT_EQ(geqlf(true, 0, 4, 4, 1), 0);  // m = 0: any lwork >= 1
  EXPECT_EQ(errors.Take(), std::vector<std::string>{});

  // xORGRQ / xUNGRQ.
  const std::string ungrq = Name<T>(IsComplex<T>::value ? "UNGRQ" : "ORGRQ");
  const auto generate = [&](hwyblas_int m, hwyblas_int n, hwyblas_int k,
                            hwyblas_int lda, hwyblas_int lwork) {
    hwyblas_int info = -99;
    CALL_RC(T, orgrq, ungrq, &m, &n, &k, P(a.data()), &lda, P(tau.data()),
            P(work.data()), &lwork, &info);
    return info;
  };
  expect(generate(-1, 4, 0, 4, 64), 1, ungrq);
  expect(generate(4, 3, 2, 4, 64), 2, ungrq);
  expect(generate(4, 6, -1, 4, 64), 3, ungrq);
  expect(generate(4, 6, 5, 4, 64), 3, ungrq);
  expect(generate(4, 6, 2, 3, 64), 5, ungrq);
  expect(generate(4, 6, 2, 4, 3), 8, ungrq);

  // xGEQRT.
  const auto geqrt = [&](hwyblas_int m, hwyblas_int n, hwyblas_int nb,
                         hwyblas_int lda, hwyblas_int ldt) {
    hwyblas_int info = -99;
    HWYBLAS_CALL(T, geqrt, &m, &n, &nb, P(a.data()), &lda, P(t.data()), &ldt,
                 P(work.data()), &info);
    return info;
  };
  const std::string geqrt_name = Name<T>("GEQRT");
  expect(geqrt(-1, 4, 1, 4, 4), 1, geqrt_name);
  expect(geqrt(4, -1, 1, 4, 4), 2, geqrt_name);
  expect(geqrt(4, 3, 0, 4, 4), 3, geqrt_name);
  expect(geqrt(4, 3, 4, 4, 4), 3, geqrt_name);  // nb > min(m, n)
  expect(geqrt(4, 3, 2, 3, 4), 5, geqrt_name);
  expect(geqrt(4, 3, 2, 4, 1), 7, geqrt_name);
  EXPECT_EQ(geqrt(0, 3, 5, 1, 5), 0);  // min(m, n) = 0: any nb >= 1
  EXPECT_EQ(errors.Take(), std::vector<std::string>{});

  // xGEQRT2 and xGEQRT3.
  for (const bool recursive : {false, true}) {
    const auto geqrt2 = [&](hwyblas_int m, hwyblas_int n, hwyblas_int lda,
                            hwyblas_int ldt) {
      hwyblas_int info = -99;
      if (recursive) {
        HWYBLAS_CALL(T, geqrt3, &m, &n, P(a.data()), &lda, P(t.data()), &ldt,
                     &info);
      } else {
        HWYBLAS_CALL(T, geqrt2, &m, &n, P(a.data()), &lda, P(t.data()), &ldt,
                     &info);
      }
      return info;
    };
    const std::string name = Name<T>(recursive ? "GEQRT3" : "GEQRT2");
    expect(geqrt2(4, -1, 4, 4), 2, name);
    expect(geqrt2(3, 4, 4, 4), 1, name);  // m < n
    expect(geqrt2(4, 3, 3, 3), 4, name);
    expect(geqrt2(4, 3, 4, 2), 6, name);
  }

  // xGEMQRT.
  const auto gemqrt = [&](char side, char trans, hwyblas_int m, hwyblas_int n,
                          hwyblas_int k, hwyblas_int nb, hwyblas_int ldv,
                          hwyblas_int ldt, hwyblas_int ldc) {
    hwyblas_int info = -99;
    HWYBLAS_CALL(T, gemqrt, &side, &trans, &m, &n, &k, &nb, P(a.data()), &ldv,
                 P(t.data()), &ldt, P(c.data()), &ldc, P(work.data()), &info);
    return info;
  };
  const std::string gemqrt_name = Name<T>("GEMQRT");
  // TRANS = 'T' is invalid for complex T, 'C' for real T.
  const char wrong_trans = IsComplex<T>::value ? 'T' : 'C';
  const char tr = ConjTrans<T>();
  expect(gemqrt('X', 'N', 4, 3, 2, 2, 4, 2, 4), 1, gemqrt_name);
  expect(gemqrt('L', wrong_trans, 4, 3, 2, 2, 4, 2, 4), 2, gemqrt_name);
  expect(gemqrt('L', tr, -1, 3, 2, 2, 4, 2, 4), 3, gemqrt_name);
  expect(gemqrt('R', 'N', 4, -1, 2, 2, 4, 2, 4), 4, gemqrt_name);
  expect(gemqrt('L', 'N', 4, 3, -1, 2, 4, 2, 4), 5, gemqrt_name);
  expect(gemqrt('R', 'N', 4, 3, 4, 2, 4, 2, 4), 5, gemqrt_name);  // k > n
  expect(gemqrt('L', 'N', 4, 3, 2, 0, 4, 2, 4), 6, gemqrt_name);
  expect(gemqrt('L', 'N', 4, 3, 2, 3, 4, 3, 4), 6, gemqrt_name);  // nb > k
  expect(gemqrt('L', 'N', 4, 3, 2, 2, 3, 2, 4), 8, gemqrt_name);
  expect(gemqrt('R', tr, 4, 5, 2, 2, 4, 2, 4), 8, gemqrt_name);  // ldv < n
  expect(gemqrt('L', 'N', 4, 3, 2, 2, 4, 1, 4), 10, gemqrt_name);
  expect(gemqrt('L', 'N', 4, 3, 2, 2, 4, 2, 3), 12, gemqrt_name);
  EXPECT_EQ(gemqrt('L', 'N', 4, 3, 0, 4, 4, 4, 4), 0);  // k = 0: any nb >= 1
  EXPECT_EQ(errors.Take(), std::vector<std::string>{});
}

// ---------------------------------------------------------------------------
// LAPACKE

// Row-major LAPACKE calls give the same results as column-major calls.
// xGEMQRT runs with K > NB (several blocks of T) and, for SIDE = 'L',
// N > M (a work array of N*NB elements).
TYPED_TEST(GeqrtTest, LapackeLayouts) {
  using T = TypeParam;
  Random<T> rng(61);
  const lapack_int ld = 30;
  const auto expect_same = [&](const std::vector<T>& col,
                               const std::vector<T>& row, lapack_int rows,
                               lapack_int cols) {
    for (lapack_int j = 0; j < cols; ++j) {
      for (lapack_int i = 0; i < rows; ++i) {
        ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << i << ", " << j;
      }
    }
  };
  const std::vector<T> a = rng.Vector(static_cast<size_t>(ld) * ld);

  // xGEQLF, then xORGRQ / xUNGRQ of xGERQF's reflectors.
  {
    const lapack_int m = 23, n = 15;
    std::vector<T> col = a, row = Transposed(a, ld), tau_col(n), tau_row(n);
    ASSERT_EQ(LAPACKE_CALL(T, geqlf, LAPACK_COL_MAJOR, m, n, col.data(), ld,
                           tau_col.data()),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, geqlf, LAPACK_ROW_MAJOR, m, n, row.data(), ld,
                           tau_row.data()),
              0);
    expect_same(col, row, m, n);
    EXPECT_EQ(tau_col, tau_row);
  }
  {
    const lapack_int m = 15, n = 23, k = 10;
    std::vector<T> col = a, tau(m);
    ASSERT_EQ(LAPACKE_CALL(T, gerqf, LAPACK_COL_MAJOR, m, n, col.data(), ld,
                           tau.data()),
              0);
    std::vector<T> row = Transposed(col, ld);
    ASSERT_EQ(LAPACKE_CALL_RC(T, orgrq, ungrq, LAPACK_COL_MAJOR, m, n, k,
                              col.data(), ld, tau.data() + (m - k)),
              0);
    ASSERT_EQ(LAPACKE_CALL_RC(T, orgrq, ungrq, LAPACK_ROW_MAJOR, m, n, k,
                              row.data(), ld, tau.data() + (m - k)),
              0);
    expect_same(col, row, m, n);
  }

  // xGEQRT2 and xGEQRT3 (T is n x n).
  {
    const lapack_int m = 23, n = 15;
    for (const bool recursive : {false, true}) {
      SCOPED_TRACE(recursive ? "geqrt3" : "geqrt2");
      std::vector<T> col = a, row = Transposed(a, ld);
      std::vector<T> t_col(ld * ld, T(0)), t_row(ld * ld, T(0));
      if (recursive) {
        ASSERT_EQ(LAPACKE_CALL(T, geqrt3, LAPACK_COL_MAJOR, m, n, col.data(),
                               ld, t_col.data(), ld),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, geqrt3, LAPACK_ROW_MAJOR, m, n, row.data(),
                               ld, t_row.data(), ld),
                  0);
      } else {
        ASSERT_EQ(LAPACKE_CALL(T, geqrt2, LAPACK_COL_MAJOR, m, n, col.data(),
                               ld, t_col.data(), ld),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, geqrt2, LAPACK_ROW_MAJOR, m, n, row.data(),
                               ld, t_row.data(), ld),
                  0);
      }
      expect_same(col, row, m, n);
      // The upper triangle of T (the rest is not specified).
      for (lapack_int j = 0; j < n; ++j) {
        for (lapack_int i = 0; i <= j; ++i) {
          ASSERT_EQ(t_row[i * ld + j], At(t_col, i, j, ld)) << i << ", " << j;
        }
      }
    }
  }

  // xGEQRT (T is nb x min(m,n)), then xGEMQRT with its reflectors.
  const lapack_int nb = 4;
  for (const char side : {'L', 'R'}) {
    for (const char trans : {'N', ConjTrans<T>()}) {
      SCOPED_TRACE(::testing::Message()
                   << "side=" << side << " trans=" << trans);
      const lapack_int q = 23, k = 15;
      std::vector<T> v_col = a, v_row = Transposed(a, ld);
      std::vector<T> t_col(ld * ld, T(0)), t_row(ld * ld, T(0));
      ASSERT_EQ(LAPACKE_CALL(T, geqrt, LAPACK_COL_MAJOR, q, k, nb, v_col.data(),
                             ld, t_col.data(), ld),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, geqrt, LAPACK_ROW_MAJOR, q, k, nb, v_row.data(),
                             ld, t_row.data(), ld),
                0);
      expect_same(v_col, v_row, q, k);
      for (lapack_int j = 0; j < k; ++j) {
        for (lapack_int i = 0; i <= j % nb; ++i) {
          ASSERT_EQ(t_row[i * ld + j], At(t_col, i, j, ld)) << i << ", " << j;
        }
        // Below the blocks, the row-major T has the uninitialized elements
        // of LAPACKE's temporary (not read by xGEMQRT, but by its NaN
        // check).
        for (lapack_int i = j % nb + 1; i < nb; ++i) t_row[i * ld + j] = T(0);
      }
      const lapack_int m = side == 'L' ? q : 27, n = side == 'L' ? 27 : q;
      const std::vector<T> c = rng.Vector(static_cast<size_t>(ld) * ld);
      std::vector<T> c_col = c, c_row = Transposed(c, ld);
      ASSERT_EQ(
          LAPACKE_CALL(T, gemqrt, LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
                       v_col.data(), ld, t_col.data(), ld, c_col.data(), ld),
          0);
      ASSERT_EQ(
          LAPACKE_CALL(T, gemqrt, LAPACK_ROW_MAJOR, side, trans, m, n, k, nb,
                       v_row.data(), ld, t_row.data(), ld, c_row.data(), ld),
          0);
      expect_same(c_col, c_row, m, n);
    }
  }
}

// LAPACKE's argument checks: an invalid layout, leading dimensions of
// row-major matrices, and NaN in the input matrices.
TYPED_TEST(GeqrtTest, LapackeErrors) {
  using T = TypeParam;
  std::vector<T> a(64, T(1)), tau(8, T(0.5)), t(64, T(1)), c(64, T(1)),
      work(64);
  ::testing::internal::CaptureStdout();
  const lapack_int row = LAPACK_ROW_MAJOR, col = LAPACK_COL_MAJOR;
  const char tr = ConjTrans<T>();
  EXPECT_EQ(LAPACKE_CALL(T, geqlf, 0, 4, 4, a.data(), 4, tau.data()), -1);
  EXPECT_EQ(LAPACKE_CALL(T, geqlf_work, row, 4, 5, a.data(), 4, tau.data(),
                         work.data(), 64),
            -5);
  EXPECT_EQ(LAPACKE_CALL_RC(T, orgrq_work, ungrq_work, row, 4, 5, 2, a.data(),
                            4, tau.data(), work.data(), 64),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, geqrt_work, row, 5, 4, 2, a.data(), 3, t.data(), 4,
                         work.data()),
            -6);
  // T is nb x min(m, n) in row-major layout: ldt >= min(m, n).
  EXPECT_EQ(LAPACKE_CALL(T, geqrt_work, row, 5, 4, 2, a.data(), 4, t.data(), 3,
                         work.data()),
            -8);
  // A wide matrix: ldt = min(m, n) < n is enough.
  {
    std::vector<T> wide(64, T(1)), t_wide(64);
    for (size_t i = 0; i < wide.size(); ++i)
      wide[i] = MakeScalar<T>(static_cast<double>(i % 7) - 3, 0);
    EXPECT_EQ(LAPACKE_CALL(T, geqrt_work, row, 4, 5, 2, wide.data(), 5,
                           t_wide.data(), 4, work.data()),
              0);
  }
  EXPECT_EQ(LAPACKE_CALL(T, geqrt2_work, row, 5, 4, a.data(), 3, t.data(), 4),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, geqrt3_work, row, 5, 4, a.data(), 4, t.data(), 3),
            -7);
  // xGEMQRT: C (m x n), T (nb x k) and V (q x k) in row-major layout.
  EXPECT_EQ(LAPACKE_CALL(T, gemqrt_work, row, 'L', tr, 5, 4, 3, 2, a.data(), 3,
                         t.data(), 3, c.data(), 3, work.data()),
            -13);
  EXPECT_EQ(LAPACKE_CALL(T, gemqrt_work, row, 'L', tr, 5, 4, 3, 2, a.data(), 3,
                         t.data(), 2, c.data(), 4, work.data()),
            -11);
  EXPECT_EQ(LAPACKE_CALL(T, gemqrt_work, row, 'R', 'N', 5, 4, 3, 2, a.data(), 2,
                         t.data(), 3, c.data(), 4, work.data()),
            -9);
  const std::string out = ::testing::internal::GetCapturedStdout();
  const std::string ungrq = IsComplex<T>::value ? "ungrq_work" : "orgrq_work";
  EXPECT_EQ(out, "Wrong parameter 1 in " + LapackeName<T>("geqlf") +
                     "\nWrong parameter 5 in " + LapackeName<T>("geqlf_work") +
                     "\nWrong parameter 6 in " + LapackeName<T>(ungrq.c_str()) +
                     "\nWrong parameter 6 in " + LapackeName<T>("geqrt_work") +
                     "\nWrong parameter 8 in " + LapackeName<T>("geqrt_work") +
                     "\nWrong parameter 5 in " + LapackeName<T>("geqrt2_work") +
                     "\nWrong parameter 7 in " + LapackeName<T>("geqrt3_work") +
                     "\nWrong parameter 13 in " +
                     LapackeName<T>("gemqrt_work") +
                     "\nWrong parameter 11 in " +
                     LapackeName<T>("gemqrt_work") + "\nWrong parameter 9 in " +
                     LapackeName<T>("gemqrt_work") + "\n");

  // NaN checks of the high-level functions.
  std::vector<T> a_nan = a, tau_nan = tau, t_nan = t, c_nan = c;
  a_nan[5] = NaN<T>();
  tau_nan[1] = NaN<T>();
  t_nan[4] = NaN<T>();  // T(0, 2) of the 2 x 3 T (ldt 2)
  c_nan[9] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, geqlf, col, 4, 4, a_nan.data(), 4, tau.data()), -4);
  EXPECT_EQ(LAPACKE_CALL_RC(T, orgrq, ungrq, col, 4, 4, 2, a_nan.data(), 4,
                            tau.data()),
            -5);
  EXPECT_EQ(LAPACKE_CALL_RC(T, orgrq, ungrq, col, 4, 4, 2, a.data(), 4,
                            tau_nan.data()),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, geqrt, col, 4, 4, 2, a_nan.data(), 4, t.data(), 2),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, geqrt2, col, 4, 4, a_nan.data(), 4, t.data(), 4),
            -4);
  EXPECT_EQ(LAPACKE_CALL(T, geqrt3, col, 4, 4, a_nan.data(), 4, t.data(), 4),
            -4);
  EXPECT_EQ(LAPACKE_CALL(T, gemqrt, col, 'L', 'N', 4, 4, 3, 2, a.data(), 4,
                         t.data(), 2, c_nan.data(), 4),
            -12);
  EXPECT_EQ(LAPACKE_CALL(T, gemqrt, col, 'L', 'N', 4, 4, 3, 2, a.data(), 4,
                         t_nan.data(), 2, c.data(), 4),
            -10);
  EXPECT_EQ(LAPACKE_CALL(T, gemqrt, col, 'R', 'N', 4, 4, 3, 2, a_nan.data(), 4,
                         t.data(), 2, c.data(), 4),
            -8);
}

}  // namespace
}  // namespace hwyblas_test
