// Tests of the QR and LQ factorizations and their Q: xGEQRF, xGEQR2, xGELQF,
// xGELQ2, xORGQR / xUNGQR, xORG2R / xUNG2R, xORGLQ / xUNGLQ, xORGL2 /
// xUNGL2, xORMQR / xUNMQR, xORM2R / xUNM2R, xORMLQ / xUNMLQ, xORML2 /
// xUNML2, and their LAPACKE functions.
#include <algorithm>
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
class QrTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(QrTest, ScalarTypes);

template <typename T>
char ConjTrans() {
  return IsComplex<T>::value ? 'C' : 'T';
}

// Calls the real routine `rname` (s, d) or the complex routine `cname`
// (c, z) of T's precision.
#define CALL_REAL_OR_COMPLEX(T, rname, cname, ...)                 \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      s##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, double>) {              \
      d##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      c##cname##_(__VA_ARGS__);                                    \
    } else {                                                       \
      z##cname##_(__VA_ARGS__);                                    \
    }                                                              \
  } while (0)

// The factorization: QR (A = Q * R) or LQ (A = L * Q).
enum class Kind { kQr, kLq };

const char* KindName(Kind kind) { return kind == Kind::kQr ? "QR" : "LQ"; }

// LWORK of the calls: the optimal one (from a workspace query), the
// minimum, or three times the minimum (with a reduced block size).
enum class Lwork { kOptimal, kMinimum, kReduced };

template <typename T>
hwyblas_int QueryLwork(const std::vector<T>& work_query) {
  return static_cast<hwyblas_int>(std::real(work_query[0]));
}

// xGEQRF / xGELQF (blocked) or xGEQR2 / xGELQ2.
template <typename T>
hwyblas_int Factor(Kind kind, bool blocked, Lwork lwork_kind, hwyblas_int m,
                   hwyblas_int n, std::vector<T>& a, hwyblas_int lda,
                   std::vector<T>& tau) {
  hwyblas_int info = -99;
  if (!blocked) {
    std::vector<T> work(std::max<hwyblas_int>({1, m, n}), NaN<T>());
    if (kind == Kind::kQr) {
      HWYBLAS_CALL(T, geqr2, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &info);
    } else {
      HWYBLAS_CALL(T, gelq2, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &info);
    }
    return info;
  }
  const hwyblas_int minimum =
      std::max<hwyblas_int>(1, kind == Kind::kQr ? n : m);
  hwyblas_int lwork = -1;
  std::vector<T> query(1);
  if (kind == Kind::kQr) {
    HWYBLAS_CALL(T, geqrf, &m, &n, P(a.data()), &lda, P(tau.data()),
                 P(query.data()), &lwork, &info);
  } else {
    HWYBLAS_CALL(T, gelqf, &m, &n, P(a.data()), &lda, P(tau.data()),
                 P(query.data()), &lwork, &info);
  }
  if (info != 0) return info;
  lwork = lwork_kind == Lwork::kOptimal   ? std::max(QueryLwork(query), minimum)
          : lwork_kind == Lwork::kMinimum ? minimum
                                          : 3 * minimum;
  std::vector<T> work(lwork, NaN<T>());
  if (kind == Kind::kQr) {
    HWYBLAS_CALL(T, geqrf, &m, &n, P(a.data()), &lda, P(tau.data()),
                 P(work.data()), &lwork, &info);
  } else {
    HWYBLAS_CALL(T, gelqf, &m, &n, P(a.data()), &lda, P(tau.data()),
                 P(work.data()), &lwork, &info);
  }
  return info;
}

// xORGQR / xUNGQR, xORGLQ / xUNGLQ (blocked) or the unblocked routines:
// overwrites q (rows x cols) with the matrix Q of k reflectors.
template <typename T>
hwyblas_int GenerateQ(Kind kind, bool blocked, Lwork lwork_kind,
                      hwyblas_int rows, hwyblas_int cols, hwyblas_int k,
                      std::vector<T>& q, hwyblas_int ldq,
                      const std::vector<T>& tau) {
  hwyblas_int info = -99;
  if (!blocked) {
    std::vector<T> work(std::max<hwyblas_int>({1, rows, cols}), NaN<T>());
    if (kind == Kind::kQr) {
      CALL_REAL_OR_COMPLEX(T, org2r, ung2r, &rows, &cols, &k, P(q.data()), &ldq,
                           P(tau.data()), P(work.data()), &info);
    } else {
      CALL_REAL_OR_COMPLEX(T, orgl2, ungl2, &rows, &cols, &k, P(q.data()), &ldq,
                           P(tau.data()), P(work.data()), &info);
    }
    return info;
  }
  const hwyblas_int minimum =
      std::max<hwyblas_int>(1, kind == Kind::kQr ? cols : rows);
  hwyblas_int lwork = -1;
  std::vector<T> query(1);
  if (kind == Kind::kQr) {
    CALL_REAL_OR_COMPLEX(T, orgqr, ungqr, &rows, &cols, &k, P(q.data()), &ldq,
                         P(tau.data()), P(query.data()), &lwork, &info);
  } else {
    CALL_REAL_OR_COMPLEX(T, orglq, unglq, &rows, &cols, &k, P(q.data()), &ldq,
                         P(tau.data()), P(query.data()), &lwork, &info);
  }
  if (info != 0) return info;
  lwork = lwork_kind == Lwork::kOptimal   ? std::max(QueryLwork(query), minimum)
          : lwork_kind == Lwork::kMinimum ? minimum
                                          : 3 * minimum;
  std::vector<T> work(lwork, NaN<T>());
  if (kind == Kind::kQr) {
    CALL_REAL_OR_COMPLEX(T, orgqr, ungqr, &rows, &cols, &k, P(q.data()), &ldq,
                         P(tau.data()), P(work.data()), &lwork, &info);
  } else {
    CALL_REAL_OR_COMPLEX(T, orglq, unglq, &rows, &cols, &k, P(q.data()), &ldq,
                         P(tau.data()), P(work.data()), &lwork, &info);
  }
  return info;
}

// xORMQR / xUNMQR, xORMLQ / xUNMLQ (blocked) or the unblocked routines.
template <typename T>
hwyblas_int MultiplyQ(Kind kind, bool blocked, Lwork lwork_kind, char side,
                      char trans, hwyblas_int m, hwyblas_int n, hwyblas_int k,
                      const std::vector<T>& a, hwyblas_int lda,
                      const std::vector<T>& tau, std::vector<T>& c,
                      hwyblas_int ldc) {
  hwyblas_int info = -99;
  if (!blocked) {
    std::vector<T> work(std::max<hwyblas_int>({1, m, n}), NaN<T>());
    if (kind == Kind::kQr) {
      CALL_REAL_OR_COMPLEX(T, orm2r, unm2r, &side, &trans, &m, &n, &k,
                           P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                           P(work.data()), &info);
    } else {
      CALL_REAL_OR_COMPLEX(T, orml2, unml2, &side, &trans, &m, &n, &k,
                           P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                           P(work.data()), &info);
    }
    return info;
  }
  const hwyblas_int minimum = std::max<hwyblas_int>(1, side == 'L' ? n : m);
  hwyblas_int lwork = -1;
  std::vector<T> query(1);
  if (kind == Kind::kQr) {
    CALL_REAL_OR_COMPLEX(T, ormqr, unmqr, &side, &trans, &m, &n, &k,
                         P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                         P(query.data()), &lwork, &info);
  } else {
    CALL_REAL_OR_COMPLEX(T, ormlq, unmlq, &side, &trans, &m, &n, &k,
                         P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                         P(query.data()), &lwork, &info);
  }
  if (info != 0) return info;
  // The reduced workspace leaves room for the triangular factor (65 x 64)
  // and blocks of 3 reflectors.
  lwork = lwork_kind == Lwork::kOptimal   ? QueryLwork(query)
          : lwork_kind == Lwork::kMinimum ? minimum
                                          : 65 * 64 + 3 * minimum;
  std::vector<T> work(lwork, NaN<T>());
  if (kind == Kind::kQr) {
    CALL_REAL_OR_COMPLEX(T, ormqr, unmqr, &side, &trans, &m, &n, &k,
                         P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                         P(work.data()), &lwork, &info);
  } else {
    CALL_REAL_OR_COMPLEX(T, ormlq, unmlq, &side, &trans, &m, &n, &k,
                         P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                         P(work.data()), &lwork, &info);
  }
  return info;
}

// The full orthogonal / unitary Q of a factorization (m x m for QR, n x n
// for LQ) from the reflectors in af and tau.
template <typename T>
Dense<T> FullQ(Kind kind, bool blocked, Lwork lwork, hwyblas_int m,
               hwyblas_int n, const std::vector<T>& af, hwyblas_int lda,
               const std::vector<T>& tau) {
  const hwyblas_int k = std::min(m, n);
  const hwyblas_int order = (kind == Kind::kQr) ? m : n;
  const hwyblas_int ldq = order + 1;
  std::vector<T> q(static_cast<size_t>(ldq) * order, NaN<T>());
  // The reflectors: below the diagonal (QR) or right of it (LQ).
  for (hwyblas_int j = 0; j < (kind == Kind::kQr ? k : order); ++j) {
    for (hwyblas_int i = 0; i < (kind == Kind::kQr ? order : k); ++i) {
      At(q, i, j, ldq) = At(af, i, j, lda);
    }
  }
  EXPECT_EQ(GenerateQ(kind, blocked, lwork, order, order, k, q, ldq, tau), 0);
  return ToDense(q, order, order, ldq);
}

// R (upper trapezoidal) or L (lower trapezoidal) of the factorization.
template <typename T>
Dense<T> Triangle(Kind kind, hwyblas_int m, hwyblas_int n,
                  const std::vector<T>& af, hwyblas_int lda) {
  Dense<T> r(m, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < m; ++i) {
      if (kind == Kind::kQr ? i <= j : i >= j) {
        r(i, j) = static_cast<RefT<T>>(At(af, i, j, lda));
      }
    }
  }
  return r;
}

struct Shape {
  hwyblas_int m, n;
};
const Shape kShapes[] = {{1, 1},   {1, 5},   {5, 1},   {2, 2},
                         {3, 7},   {7, 3},   {10, 10}, {33, 17},
                         {17, 33}, {64, 64}, {65, 70}, {100, 40}};

// The block sizes nb and crossover points nx of the tests: ILAENV's, and
// small ones with blocked code down to the end (nx = 0) or not (nx = 5).
struct Blocking {
  hwyblas_int nb, nx;
};
const Blocking kBlockings[] = {{0, 0}, {1, 0}, {2, 0}, {3, 5}, {8, 0}, {8, 5}};

// A = Q * R (or L * Q) with orthogonal / unitary Q, for all routines
// generating Q, and Q1 * R1 for the first min(m, n) columns (rows) of Q.
TYPED_TEST(QrTest, Factorization) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(31);
    for (const Kind kind : {Kind::kQr, Kind::kLq}) {
      for (const bool blocked : {false, true}) {
        for (const Blocking& b : kBlockings) {
          if (!blocked && b.nb != 0) continue;
          const ScopedBlockSize block_size(b.nb, b.nx);
          for (const Lwork lwork :
               {Lwork::kOptimal, Lwork::kMinimum, Lwork::kReduced}) {
            if (!blocked && lwork != Lwork::kOptimal) continue;
            for (const Shape& s : kShapes) {
              const hwyblas_int lda = s.m + 2;
              SCOPED_TRACE(::testing::Message()
                           << KindName(kind) << (blocked ? " blocked" : "")
                           << " m=" << s.m << " n=" << s.n << " nb=" << b.nb
                           << " nx=" << b.nx
                           << " lwork=" << static_cast<int>(lwork));
              const std::vector<T> a =
                  rng.Vector(static_cast<size_t>(lda) * s.n);
              std::vector<T> af = a;
              std::vector<T> tau(std::max<hwyblas_int>(1, std::min(s.m, s.n)));
              ASSERT_EQ(Factor(kind, blocked, lwork, s.m, s.n, af, lda, tau),
                        0);
              for (hwyblas_int j = 0; j < s.n; ++j) {
                for (hwyblas_int i = s.m; i < lda; ++i) {
                  ASSERT_EQ(At(af, i, j, lda), At(a, i, j, lda));
                }
              }

              const Dense<T> q =
                  FullQ(kind, blocked, lwork, s.m, s.n, af, lda, tau);
              const Dense<T> r = Triangle(kind, s.m, s.n, af, lda);
              const Dense<T> ad = ToDense(a, s.m, s.n, lda);
              const Dense<T> prod = (kind == Kind::kQr) ? Mul(q, r) : Mul(r, q);
              EXPECT_LT(ResidualRatio(ad, prod, kind == Kind::kQr ? s.m : s.n),
                        kLapackThreshold);
              EXPECT_LT(OrthogonalityRatio(kind == Kind::kQr ? q : Adjoint(q)),
                        kLapackThreshold);

              // The first k columns (QR) or rows (LQ) of Q only.
              const hwyblas_int k = std::min(s.m, s.n);
              const hwyblas_int rows = (kind == Kind::kQr) ? s.m : k;
              const hwyblas_int cols = (kind == Kind::kQr) ? k : s.n;
              std::vector<T> q1 = af;
              ASSERT_EQ(
                  GenerateQ(kind, blocked, lwork, rows, cols, k, q1, lda, tau),
                  0);
              const Dense<T> q1d = ToDense(q1, rows, cols, lda);
              EXPECT_LT(
                  OrthogonalityRatio(kind == Kind::kQr ? q1d : Adjoint(q1d)),
                  kLapackThreshold);
              Dense<T> r1(kind == Kind::kQr ? k : s.m,
                          kind == Kind::kQr ? s.n : k);
              for (size_t j = 0; j < r1.cols; ++j) {
                for (size_t i = 0; i < r1.rows; ++i) r1(i, j) = r(i, j);
              }
              EXPECT_LT(ResidualRatio(
                            ad, kind == Kind::kQr ? Mul(q1d, r1) : Mul(r1, q1d),
                            kind == Kind::kQr ? s.m : s.n),
                        kLapackThreshold);
            }
          }
        }
      }
    }
  });
}

// Q * C, Q**H * C, C * Q and C * Q**H against the explicit Q, for Q of k
// reflectors of order nq (k <= nq).
TYPED_TEST(QrTest, MultiplyQ) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(37);
    for (const Kind kind : {Kind::kQr, Kind::kLq}) {
      for (const bool blocked : {false, true}) {
        for (const Blocking& b : kBlockings) {
          if (!blocked && b.nb != 0) continue;
          const ScopedBlockSize block_size(b.nb, b.nx);
          for (const Lwork lwork :
               {Lwork::kOptimal, Lwork::kMinimum, Lwork::kReduced}) {
            if (!blocked && lwork != Lwork::kOptimal) continue;
            for (const char side : {'L', 'R'}) {
              for (const char trans : {'N', ConjTrans<T>()}) {
                for (const hwyblas_int nq : {1, 6, 30}) {
                  for (const hwyblas_int k : {1, 5, 30}) {
                    if (k > nq) continue;
                    for (const hwyblas_int other : {1, 13}) {
                      const hwyblas_int m = (side == 'L') ? nq : other;
                      const hwyblas_int n = (side == 'L') ? other : nq;
                      SCOPED_TRACE(::testing::Message()
                                   << KindName(kind)
                                   << (blocked ? " blocked" : "")
                                   << " side=" << side << " trans=" << trans
                                   << " m=" << m << " n=" << n << " k=" << k
                                   << " nb=" << b.nb << " nx=" << b.nx
                                   << " lwork=" << static_cast<int>(lwork));
                      // Reflectors from factoring a random nq x k (QR) or
                      // k x nq (LQ) matrix.
                      const hwyblas_int am = (kind == Kind::kQr) ? nq : k;
                      const hwyblas_int an = (kind == Kind::kQr) ? k : nq;
                      const hwyblas_int lda = am + 1;
                      std::vector<T> af =
                          rng.Vector(static_cast<size_t>(lda) * an);
                      std::vector<T> tau(k);
                      ASSERT_EQ(Factor(kind, true, Lwork::kOptimal, am, an, af,
                                       lda, tau),
                                0);
                      const Dense<T> q = FullQ(kind, true, Lwork::kOptimal, am,
                                               an, af, lda, tau);
                      const Dense<T> qop = (trans == 'N') ? q : Adjoint(q);

                      const hwyblas_int ldc = m + 3;
                      std::vector<T> c =
                          rng.Vector(static_cast<size_t>(ldc) * n);
                      const std::vector<T> c0 = c;
                      const std::vector<T> af0 = af;
                      ASSERT_EQ(MultiplyQ(kind, blocked, lwork, side, trans, m,
                                          n, k, af, lda, tau, c, ldc),
                                0);
                      const Dense<T> c0d = ToDense(c0, m, n, ldc);
                      const Dense<T> expected =
                          (side == 'L') ? Mul(qop, c0d) : Mul(c0d, qop);
                      EXPECT_LT(
                          ResidualRatio(expected, ToDense(c, m, n, ldc), nq),
                          kLapackThreshold);
                      // A is restored (the complex LQ routines conjugate
                      // its rows while they run), padding is not touched.
                      ASSERT_EQ(af, af0);
                      for (hwyblas_int j = 0; j < n; ++j) {
                        for (hwyblas_int i = m; i < ldc; ++i) {
                          ASSERT_EQ(At(c, i, j, ldc), At(c0, i, j, ldc));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

// Workspace queries return the documented optimal sizes.
TYPED_TEST(QrTest, WorkspaceQueries) {
  using T = TypeParam;
  const ScopedBlockSize block_size(8);
  const hwyblas_int m = 30, n = 20, k = 10, lda = 30, minus1 = -1;
  std::vector<T> a(lda * 30), tau(30), c(lda * 30), work(1);
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, geqrf, &m, &n, P(a.data()), &lda, P(tau.data()),
               P(work.data()), &minus1, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(QueryLwork(work), n * 8);
  HWYBLAS_CALL(T, gelqf, &m, &n, P(a.data()), &lda, P(tau.data()),
               P(work.data()), &minus1, &info);
  EXPECT_EQ(QueryLwork(work), m * 8);
  CALL_REAL_OR_COMPLEX(T, orgqr, ungqr, &m, &n, &k, P(a.data()), &lda,
                       P(tau.data()), P(work.data()), &minus1, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(QueryLwork(work), n * 8);
  CALL_REAL_OR_COMPLEX(T, orglq, unglq, &n, &m, &k, P(a.data()), &lda,
                       P(tau.data()), P(work.data()), &minus1, &info);
  EXPECT_EQ(QueryLwork(work), n * 8);
  const char side = 'R', trans = 'N';
  CALL_REAL_OR_COMPLEX(T, ormqr, unmqr, &side, &trans, &m, &n, &k, P(a.data()),
                       &lda, P(tau.data()), P(c.data()), &lda, P(work.data()),
                       &minus1, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(QueryLwork(work), m * 8 + 65 * 64);
  CALL_REAL_OR_COMPLEX(T, ormlq, unmlq, &side, &trans, &m, &n, &k, P(a.data()),
                       &lda, P(tau.data()), P(c.data()), &lda, P(work.data()),
                       &minus1, &info);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(QueryLwork(work), m * 8 + 65 * 64);
  // Empty matrices: 1.
  const hwyblas_int zero = 0;
  HWYBLAS_CALL(T, geqrf, &zero, &n, P(a.data()), &lda, P(tau.data()),
               P(work.data()), &minus1, &info);
  EXPECT_EQ(QueryLwork(work), 1);
}

// Each invalid argument in turn (the first one that LAPACK checks).
TYPED_TEST(QrTest, InvalidArguments) {
  using T = TypeParam;
  std::vector<T> a(64, T(1)), tau(8, T(1)), c(64, T(1)), work(64, T(1));
  ErrorCapture errors;
  const auto expect = [&errors](hwyblas_int info, hwyblas_int expected,
                                const std::string& name) {
    EXPECT_EQ(info, -expected) << name;
    EXPECT_EQ(errors.Take(),
              std::vector<std::string>{name + ":" + std::to_string(expected)});
  };
  // m, n, lda, lwork of the factorizations.
  const auto factor = [&](Kind kind, bool blocked, hwyblas_int m, hwyblas_int n,
                          hwyblas_int lda, hwyblas_int lwork) {
    hwyblas_int info = -99;
    if (kind == Kind::kQr && blocked) {
      HWYBLAS_CALL(T, geqrf, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &lwork, &info);
    } else if (kind == Kind::kQr) {
      HWYBLAS_CALL(T, geqr2, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &info);
    } else if (blocked) {
      HWYBLAS_CALL(T, gelqf, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &lwork, &info);
    } else {
      HWYBLAS_CALL(T, gelq2, &m, &n, P(a.data()), &lda, P(tau.data()),
                   P(work.data()), &info);
    }
    return info;
  };
  for (const Kind kind : {Kind::kQr, Kind::kLq}) {
    for (const bool blocked : {false, true}) {
      const std::string name =
          Name<T>(kind == Kind::kQr ? (blocked ? "GEQRF" : "GEQR2")
                                    : (blocked ? "GELQF" : "GELQ2"));
      expect(factor(kind, blocked, -1, 4, 4, 64), 1, name);
      expect(factor(kind, blocked, 4, -1, 4, 64), 2, name);
      expect(factor(kind, blocked, 4, 4, 3, 64), 4, name);
      if (blocked) {
        // lwork >= n (QR) or m (LQ), the longer dimension here.
        const bool qr = kind == Kind::kQr;
        expect(factor(kind, blocked, qr ? 4 : 5, qr ? 5 : 4, 5, 4), 7, name);
        expect(factor(kind, blocked, 4, 4, 4, 0), 7, name);
      }
    }
  }

  const std::string r_or_c = IsComplex<T>::value ? "UN" : "OR";
  const auto generate = [&](Kind kind, bool blocked, hwyblas_int m,
                            hwyblas_int n, hwyblas_int k, hwyblas_int lda,
                            hwyblas_int lwork) {
    hwyblas_int info = -99;
    if (kind == Kind::kQr && blocked) {
      CALL_REAL_OR_COMPLEX(T, orgqr, ungqr, &m, &n, &k, P(a.data()), &lda,
                           P(tau.data()), P(work.data()), &lwork, &info);
    } else if (kind == Kind::kQr) {
      CALL_REAL_OR_COMPLEX(T, org2r, ung2r, &m, &n, &k, P(a.data()), &lda,
                           P(tau.data()), P(work.data()), &info);
    } else if (blocked) {
      CALL_REAL_OR_COMPLEX(T, orglq, unglq, &m, &n, &k, P(a.data()), &lda,
                           P(tau.data()), P(work.data()), &lwork, &info);
    } else {
      CALL_REAL_OR_COMPLEX(T, orgl2, ungl2, &m, &n, &k, P(a.data()), &lda,
                           P(tau.data()), P(work.data()), &info);
    }
    return info;
  };
  for (const Kind kind : {Kind::kQr, Kind::kLq}) {
    for (const bool blocked : {false, true}) {
      const std::string name =
          Name<T>((r_or_c + (kind == Kind::kQr ? (blocked ? "GQR" : "G2R")
                                               : (blocked ? "GLQ" : "GL2")))
                      .c_str());
      const bool qr = kind == Kind::kQr;
      expect(generate(kind, blocked, -1, 0, 0, 4, 64), 1, name);
      // QR: n > m; LQ: n < m.
      expect(generate(kind, blocked, qr ? 3 : 4, qr ? 4 : 3, 0, 4, 64), 2,
             name);
      expect(generate(kind, blocked, 4, 4, -1, 4, 64), 3, name);
      expect(generate(kind, blocked, qr ? 4 : 3, qr ? 3 : 4, 4, 4, 64), 3,
             name);
      expect(generate(kind, blocked, 4, 4, 4, 3, 64), 5, name);
      if (blocked) expect(generate(kind, blocked, 4, 4, 4, 4, 3), 8, name);
    }
  }

  const auto multiply = [&](Kind kind, bool blocked, char side, char trans,
                            hwyblas_int m, hwyblas_int n, hwyblas_int k,
                            hwyblas_int lda, hwyblas_int ldc,
                            hwyblas_int lwork) {
    hwyblas_int info = -99;
    if (kind == Kind::kQr && blocked) {
      CALL_REAL_OR_COMPLEX(T, ormqr, unmqr, &side, &trans, &m, &n, &k,
                           P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                           P(work.data()), &lwork, &info);
    } else if (kind == Kind::kQr) {
      CALL_REAL_OR_COMPLEX(T, orm2r, unm2r, &side, &trans, &m, &n, &k,
                           P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                           P(work.data()), &info);
    } else if (blocked) {
      CALL_REAL_OR_COMPLEX(T, ormlq, unmlq, &side, &trans, &m, &n, &k,
                           P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                           P(work.data()), &lwork, &info);
    } else {
      CALL_REAL_OR_COMPLEX(T, orml2, unml2, &side, &trans, &m, &n, &k,
                           P(a.data()), &lda, P(tau.data()), P(c.data()), &ldc,
                           P(work.data()), &info);
    }
    return info;
  };
  const char wrong_trans = IsComplex<T>::value ? 'T' : 'C';
  for (const Kind kind : {Kind::kQr, Kind::kLq}) {
    for (const bool blocked : {false, true}) {
      const std::string suffix = r_or_c + "M" +
                                 (kind == Kind::kQr ? (blocked ? "QR" : "2R")
                                                    : (blocked ? "LQ" : "L2"));
      const std::string name = Name<T>(suffix.c_str());
      const bool qr = kind == Kind::kQr;
      expect(multiply(kind, blocked, 'X', 'N', 4, 4, 2, 4, 4, 64), 1, name);
      expect(multiply(kind, blocked, 'L', wrong_trans, 4, 4, 2, 4, 4, 64), 2,
             name);
      expect(multiply(kind, blocked, 'L', 'N', -1, 4, 2, 4, 4, 64), 3, name);
      expect(multiply(kind, blocked, 'L', 'N', 4, -1, 2, 4, 4, 64), 4, name);
      expect(multiply(kind, blocked, 'L', 'N', 4, 4, 5, 4, 4, 64), 5, name);
      expect(multiply(kind, blocked, 'R', 'N', 4, 3, 4, 4, 4, 64), 5, name);
      // lda >= nq (QR) or >= k (LQ).
      expect(multiply(kind, blocked, 'L', 'N', 4, 4, 2, qr ? 3 : 1, 4, 64), 7,
             name);
      expect(multiply(kind, blocked, 'L', 'N', 4, 4, 2, 4, 3, 64), 10, name);
      if (blocked) {
        expect(multiply(kind, blocked, 'L', 'N', 4, 4, 2, 4, 4, 3), 12, name);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

// Row-major LAPACKE calls give the same results as column-major calls.
TYPED_TEST(QrTest, LapackeLayouts) {
  using T = TypeParam;
  Random<T> rng(41);
  const lapack_int m = 23, n = 15, ld = 30;
  // Row-major copy of the column-major m x n matrix a (leading dimension
  // ld), and back.
  const auto to_row = [&](const std::vector<T>& a, lapack_int rows,
                          lapack_int cols) {
    std::vector<T> r(a.size(), NaN<T>());
    for (lapack_int j = 0; j < cols; ++j) {
      for (lapack_int i = 0; i < rows; ++i) r[i * ld + j] = At(a, i, j, ld);
    }
    return r;
  };
  const auto expect_same = [&](const std::vector<T>& col,
                               const std::vector<T>& row, lapack_int rows,
                               lapack_int cols) {
    for (lapack_int j = 0; j < cols; ++j) {
      for (lapack_int i = 0; i < rows; ++i) {
        ASSERT_EQ(row[i * ld + j], At(col, i, j, ld)) << i << ", " << j;
      }
    }
  };

  for (const bool lq : {false, true}) {
    SCOPED_TRACE(lq ? "LQ" : "QR");
    const std::vector<T> a = rng.Vector(static_cast<size_t>(ld) * ld);
    std::vector<T> col = a, row = to_row(a, m, n);
    std::vector<T> tau_col(n), tau_row(n);
    // Factorization (both high-level functions).
    const auto factor = [&](int layout, std::vector<T>& x,
                            std::vector<T>& tau) {
      if constexpr (std::is_same_v<T, float>) {
        return lq ? LAPACKE_sgelqf(layout, m, n, x.data(), ld, tau.data())
                  : LAPACKE_sgeqrf(layout, m, n, x.data(), ld, tau.data());
      } else if constexpr (std::is_same_v<T, double>) {
        return lq ? LAPACKE_dgelqf(layout, m, n, x.data(), ld, tau.data())
                  : LAPACKE_dgeqrf(layout, m, n, x.data(), ld, tau.data());
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return lq ? LAPACKE_cgelqf(layout, m, n, x.data(), ld, tau.data())
                  : LAPACKE_cgeqrf(layout, m, n, x.data(), ld, tau.data());
      } else {
        return lq ? LAPACKE_zgelqf(layout, m, n, x.data(), ld, tau.data())
                  : LAPACKE_zgeqrf(layout, m, n, x.data(), ld, tau.data());
      }
    };
    ASSERT_EQ(factor(LAPACK_COL_MAJOR, col, tau_col), 0);
    ASSERT_EQ(factor(LAPACK_ROW_MAJOR, row, tau_row), 0);
    expect_same(col, row, m, n);
    EXPECT_EQ(tau_col, tau_row);

    // Q * C from the left with the reflectors (multiply), then Q itself
    // (generate).
    const lapack_int k = std::min(m, n);
    const lapack_int nq = lq ? n : m;  // order of Q
    const std::vector<T> c = rng.Vector(static_cast<size_t>(ld) * ld);
    std::vector<T> c_col = c, c_row = to_row(c, nq, 7);
    const char trans = ConjTrans<T>();
    const auto multiply = [&](int layout, const std::vector<T>& x,
                              std::vector<T>& y) {
      if constexpr (std::is_same_v<T, float>) {
        return lq ? LAPACKE_sormlq(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld)
                  : LAPACKE_sormqr(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld);
      } else if constexpr (std::is_same_v<T, double>) {
        return lq ? LAPACKE_dormlq(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld)
                  : LAPACKE_dormqr(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return lq ? LAPACKE_cunmlq(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld)
                  : LAPACKE_cunmqr(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld);
      } else {
        return lq ? LAPACKE_zunmlq(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld)
                  : LAPACKE_zunmqr(layout, 'L', trans, nq, 7, k, x.data(), ld,
                                   tau_col.data(), y.data(), ld);
      }
    };
    ASSERT_EQ(multiply(LAPACK_COL_MAJOR, col, c_col), 0);
    ASSERT_EQ(multiply(LAPACK_ROW_MAJOR, row, c_row), 0);
    expect_same(c_col, c_row, nq, 7);

    // Q: the first k columns (QR, m x k) or rows (LQ, k x n).
    const lapack_int qm = lq ? k : m, qn = lq ? n : k;
    const auto generate = [&](int layout, std::vector<T>& x) {
      if constexpr (std::is_same_v<T, float>) {
        return lq ? LAPACKE_sorglq(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data())
                  : LAPACKE_sorgqr(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data());
      } else if constexpr (std::is_same_v<T, double>) {
        return lq ? LAPACKE_dorglq(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data())
                  : LAPACKE_dorgqr(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data());
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return lq ? LAPACKE_cunglq(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data())
                  : LAPACKE_cungqr(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data());
      } else {
        return lq ? LAPACKE_zunglq(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data())
                  : LAPACKE_zungqr(layout, qm, qn, k, x.data(), ld,
                                   tau_col.data());
      }
    };
    ASSERT_EQ(generate(LAPACK_COL_MAJOR, col), 0);
    ASSERT_EQ(generate(LAPACK_ROW_MAJOR, row), 0);
    expect_same(col, row, qm, qn);
  }
}

// LAPACKE's argument checks: an invalid layout, leading dimensions of
// row-major matrices, and NaN in the input matrices.
TYPED_TEST(QrTest, LapackeErrors) {
  using T = TypeParam;
  std::vector<T> a(64, T(1)), tau(8, T(0.5)), c(64, T(1)), work(64);
  ::testing::internal::CaptureStdout();
  lapack_int r_geqrf, r_gelqf_work, r_geqr2_work, r_gen, r_mul;
  if constexpr (std::is_same_v<T, double>) {
    r_geqrf = LAPACKE_dgeqrf(0, 4, 4, a.data(), 4, tau.data());
    r_gelqf_work = LAPACKE_dgelqf_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data(), 64);
    r_geqr2_work = LAPACKE_dgeqr2_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data());
    r_gen = LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, 6, 5, 4, a.data(), 4,
                                tau.data(), work.data(), 64);
    r_mul = LAPACKE_dormqr_work(LAPACK_ROW_MAJOR, 'L', 'N', 4, 6, 3, a.data(),
                                4, tau.data(), c.data(), 5, work.data(), 64);
  } else if constexpr (std::is_same_v<T, float>) {
    r_geqrf = LAPACKE_sgeqrf(0, 4, 4, a.data(), 4, tau.data());
    r_gelqf_work = LAPACKE_sgelqf_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data(), 64);
    r_geqr2_work = LAPACKE_sgeqr2_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data());
    r_gen = LAPACKE_sorgqr_work(LAPACK_ROW_MAJOR, 6, 5, 4, a.data(), 4,
                                tau.data(), work.data(), 64);
    r_mul = LAPACKE_sormqr_work(LAPACK_ROW_MAJOR, 'L', 'N', 4, 6, 3, a.data(),
                                4, tau.data(), c.data(), 5, work.data(), 64);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    r_geqrf = LAPACKE_cgeqrf(0, 4, 4, a.data(), 4, tau.data());
    r_gelqf_work = LAPACKE_cgelqf_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data(), 64);
    r_geqr2_work = LAPACKE_cgeqr2_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data());
    r_gen = LAPACKE_cungqr_work(LAPACK_ROW_MAJOR, 6, 5, 4, a.data(), 4,
                                tau.data(), work.data(), 64);
    r_mul = LAPACKE_cunmqr_work(LAPACK_ROW_MAJOR, 'L', 'N', 4, 6, 3, a.data(),
                                4, tau.data(), c.data(), 5, work.data(), 64);
  } else {
    r_geqrf = LAPACKE_zgeqrf(0, 4, 4, a.data(), 4, tau.data());
    r_gelqf_work = LAPACKE_zgelqf_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data(), 64);
    r_geqr2_work = LAPACKE_zgeqr2_work(LAPACK_ROW_MAJOR, 4, 5, a.data(), 4,
                                       tau.data(), work.data());
    r_gen = LAPACKE_zungqr_work(LAPACK_ROW_MAJOR, 6, 5, 4, a.data(), 4,
                                tau.data(), work.data(), 64);
    r_mul = LAPACKE_zunmqr_work(LAPACK_ROW_MAJOR, 'L', 'N', 4, 6, 3, a.data(),
                                4, tau.data(), c.data(), 5, work.data(), 64);
  }
  const std::string out = ::testing::internal::GetCapturedStdout();
  EXPECT_EQ(r_geqrf, -1);
  EXPECT_EQ(r_gelqf_work, -5);
  EXPECT_EQ(r_geqr2_work, -5);
  EXPECT_EQ(r_gen, -6);
  EXPECT_EQ(r_mul, -11);
  const std::string gen = IsComplex<T>::value ? "ungqr_work" : "orgqr_work";
  const std::string mul = IsComplex<T>::value ? "unmqr_work" : "ormqr_work";
  EXPECT_EQ(out, "Wrong parameter 1 in " + LapackeName<T>("geqrf") +
                     "\nWrong parameter 5 in " + LapackeName<T>("gelqf_work") +
                     "\nWrong parameter 5 in " + LapackeName<T>("geqr2_work") +
                     "\nWrong parameter 6 in " + LapackeName<T>(gen.c_str()) +
                     "\nWrong parameter 11 in " + LapackeName<T>(mul.c_str()) +
                     "\n");

  // NaN checks of the high-level functions.
  a[5] = NaN<T>();
  lapack_int r_nan_a = 0, r_nan_c = 0, r_nan_tau = 0;
  std::vector<T> a_ok(64, T(1));
  std::vector<T> c_nan = c, tau_nan = tau;
  c_nan[9] = NaN<T>();
  tau_nan[1] = NaN<T>();
  if constexpr (std::is_same_v<T, double>) {
    r_nan_a = LAPACKE_dgelqf(LAPACK_COL_MAJOR, 4, 4, a.data(), 4, tau.data());
    r_nan_c = LAPACKE_dormlq(LAPACK_COL_MAJOR, 'R', 'N', 4, 4, 2, a_ok.data(),
                             4, tau.data(), c_nan.data(), 4);
    r_nan_tau = LAPACKE_dorgqr(LAPACK_COL_MAJOR, 4, 4, 2, a_ok.data(), 4,
                               tau_nan.data());
  } else if constexpr (std::is_same_v<T, float>) {
    r_nan_a = LAPACKE_sgelqf(LAPACK_COL_MAJOR, 4, 4, a.data(), 4, tau.data());
    r_nan_c = LAPACKE_sormlq(LAPACK_COL_MAJOR, 'R', 'N', 4, 4, 2, a_ok.data(),
                             4, tau.data(), c_nan.data(), 4);
    r_nan_tau = LAPACKE_sorgqr(LAPACK_COL_MAJOR, 4, 4, 2, a_ok.data(), 4,
                               tau_nan.data());
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    r_nan_a = LAPACKE_cgelqf(LAPACK_COL_MAJOR, 4, 4, a.data(), 4, tau.data());
    r_nan_c = LAPACKE_cunmlq(LAPACK_COL_MAJOR, 'R', 'N', 4, 4, 2, a_ok.data(),
                             4, tau.data(), c_nan.data(), 4);
    r_nan_tau = LAPACKE_cungqr(LAPACK_COL_MAJOR, 4, 4, 2, a_ok.data(), 4,
                               tau_nan.data());
  } else {
    r_nan_a = LAPACKE_zgelqf(LAPACK_COL_MAJOR, 4, 4, a.data(), 4, tau.data());
    r_nan_c = LAPACKE_zunmlq(LAPACK_COL_MAJOR, 'R', 'N', 4, 4, 2, a_ok.data(),
                             4, tau.data(), c_nan.data(), 4);
    r_nan_tau = LAPACKE_zungqr(LAPACK_COL_MAJOR, 4, 4, 2, a_ok.data(), 4,
                               tau_nan.data());
  }
  EXPECT_EQ(r_nan_a, -4);
  EXPECT_EQ(r_nan_c, -10);
  EXPECT_EQ(r_nan_tau, -7);
}

}  // namespace
}  // namespace hwyblas_test
