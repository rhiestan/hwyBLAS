// Tests for the level 3 routines besides GEMM: SYMM, HEMM, SYRK, HERK,
// SYR2K, HER2K, GEMMTR, TRMM and TRSM, against dense reference products
// computed in double precision.
//
// Structured input matrices have NaN wherever BLAS must not read (see
// StructuredMatrix), so any such read shows up as a mismatch.
#include <stddef.h>

#include <complex>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class Level3Test : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Level3Test, ScalarTypes);

// Sizes covering partial tiles, panels crossing the diagonal and several
// cache blocks.
const int kSizes[] = {1, 2, 5, 13, 37, 100, 300};

// ---------------------------------------------------------------------------
// SYMM / HEMM

template <typename T>
void CheckSymm(Shape shape, char side, char uplo, int m, int n, T alpha, T beta,
               uint64_t seed) {
  const int ka = (side == 'L') ? m : n;  // A is ka x ka
  const int lda = ka + 2, ldb = m + 1, ldc = m + 3;
  Random<T> rng(seed);
  const std::vector<T> a = StructuredMatrix(shape, uplo, ka, lda, rng);
  const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * n);
  std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) * n);
  if (beta == T(0)) c.assign(c.size(), NaN<T>());
  const std::vector<T> c0 = c;

  const Dense<T> da = DenseStructured(shape, uplo, 'N', a, ka, lda);
  const Dense<T> db = DenseOp('N', b, m, n, ldb);
  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  if (side == 'L') {
    ReferenceProduct(da, db, alpha, beta, c0, ldc, &expected, &bound);
  } else {
    ReferenceProduct(db, da, alpha, beta, c0, ldc, &expected, &bound);
  }

  if (shape == Shape::kHermitian) {
    HWYBLAS_CALL_COMPLEX(T, hemm, &side, &uplo, &m, &n, P(&alpha), P(a.data()),
                         &lda, P(b.data()), &ldb, P(&beta), P(c.data()), &ldc);
  } else {
    HWYBLAS_CALL(T, symm, &side, &uplo, &m, &n, P(&alpha), P(a.data()), &lda,
                 P(b.data()), &ldb, P(&beta), P(c.data()), &ldc);
  }
  const double tol = 4.0 * (ka + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(c, expected, bound, tol, m, n, ldc))
      << "side=" << side << " uplo=" << uplo << " m=" << m << " n=" << n;
}

TYPED_TEST(Level3Test, Symm) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char side : {'L', 'R'}) {
      for (char uplo : {'L', 'U'}) {
        for (int m : kSizes) {
          for (int n : {1, 7, 40}) {
            CheckSymm<T>(Shape::kSymmetric, side, uplo, m, n,
                         MakeScalar<T>(1.5, -0.5), MakeScalar<T>(0.5, 0.25),
                         seed++);
            CheckSymm<T>(Shape::kSymmetric, side, uplo, n, m,
                         MakeScalar<T>(-1.0, 0.0), MakeScalar<T>(0.0, 0.0),
                         seed++);
          }
        }
      }
    }
  });
}

TYPED_TEST(Level3Test, Hemm) {
  using T = TypeParam;
  if constexpr (IsComplex<T>::value) {
    ForEachTarget([] {
      uint64_t seed = 1;
      for (char side : {'L', 'R'}) {
        for (char uplo : {'L', 'U'}) {
          for (int m : kSizes) {
            CheckSymm<T>(Shape::kHermitian, side, uplo, m, 23,
                         MakeScalar<T>(1.5, -0.5), MakeScalar<T>(0.5, 0.25),
                         seed++);
            CheckSymm<T>(Shape::kHermitian, side, uplo, 9, m,
                         MakeScalar<T>(0.0, 1.0), MakeScalar<T>(0.0, 0.0),
                         seed++);
          }
        }
      }
    });
  }
}

// ---------------------------------------------------------------------------
// SYRK / HERK

template <typename T>
void CheckSyrk(bool hermitian, char uplo, char trans, int n, int k,
               double alpha_re, double beta_re, uint64_t seed) {
  const bool no_trans = (trans == 'N');
  const int rows_a = no_trans ? n : k, cols_a = no_trans ? k : n;
  const int lda = rows_a + 1, ldc = n + 2;
  // Complex SYRK takes complex alpha/beta; HERK takes real ones.
  const T alpha = MakeScalar<T>(alpha_re, hermitian ? 0.0 : 0.5);
  const T beta = MakeScalar<T>(beta_re, (hermitian || beta_re == 0) ? 0 : -1);

  Random<T> rng(seed);
  const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * cols_a);
  std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) * n);
  // The other triangle must stay untouched; for HERK the imaginary parts of
  // the diagonal of C must not be read.
  const T sentinel = MakeScalar<T>(7.0, 7.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      T& cij = c[i + static_cast<size_t>(j) * ldc];
      if (uplo == 'L' ? (i < j) : (i > j)) cij = sentinel;
      if (i == j && hermitian) {
        cij = MakeScalar<T>(std::real(cij), NaN<double>());
      }
      if (beta == T(0) && (uplo == 'L' ? (i >= j) : (i <= j))) cij = NaN<T>();
    }
  }
  std::vector<T> c0 = c;
  if (hermitian) {  // the reference sees the real diagonal
    for (int j = 0; j < n; ++j) {
      T& cjj = c0[j + static_cast<size_t>(j) * ldc];
      if (!(beta == T(0))) cjj = T(std::real(cjj));
    }
  }

  // op(A) is n x k; the product is op(A) * op(A)' with ' = ^T or ^H.
  const Dense<T> da = DenseOp(no_trans ? 'N' : trans, a, rows_a, cols_a, lda);
  Dense<T> dat(k, n);
  for (int j = 0; j < n; ++j) {
    for (int p = 0; p < k; ++p) {
      dat(p, j) = hermitian ? Conj(da(j, p)) : da(j, p);
    }
  }
  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  ReferenceProduct(da, dat, alpha, beta, c0, ldc, &expected, &bound);

  if (hermitian) {
    using R = RealT<T>;
    const R ra = static_cast<R>(alpha_re), rb = static_cast<R>(beta_re);
    HWYBLAS_CALL_COMPLEX(T, herk, &uplo, &trans, &n, &k, &ra, P(a.data()), &lda,
                         &rb, P(c.data()), &ldc);
  } else {
    HWYBLAS_CALL(T, syrk, &uplo, &trans, &n, &k, P(&alpha), P(a.data()), &lda,
                 P(&beta), P(c.data()), &ldc);
  }

  const double tol = 4.0 * (k + 2) * Eps<T>();
  size_t failures = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const size_t idx = i + static_cast<size_t>(j) * ldc;
      if (uplo == 'L' ? (i < j) : (i > j)) {
        if (!(c[idx] == sentinel)) ++failures;
        continue;
      }
      const RefT<T> got = static_cast<RefT<T>>(c[idx]);
      const double err = std::abs(got - expected[idx]);
      if (!(err <= tol * bound[idx])) ++failures;
      if (hermitian && i == j && std::imag(c[idx]) != 0) ++failures;
    }
  }
  EXPECT_EQ(failures, 0u) << "herk=" << hermitian << " uplo=" << uplo
                          << " trans=" << trans << " n=" << n << " k=" << k
                          << " alpha=" << alpha_re << " beta=" << beta_re;
}

TYPED_TEST(Level3Test, Syrk) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char uplo : {'L', 'U'}) {
      for (char trans : {'N', 'T'}) {
        for (int n : kSizes) {
          for (int k : {1, 9, 300}) {
            CheckSyrk<T>(false, uplo, trans, n, k, 1.5, 0.5, seed++);
          }
          CheckSyrk<T>(false, uplo, trans, n, 20, -1.0, 0.0, seed++);
          CheckSyrk<T>(false, uplo, trans, n, 20, 0.0, 2.0, seed++);
          CheckSyrk<T>(false, uplo, trans, n, 0, 1.0, 0.5, seed++);
        }
      }
    }
  });
}

TYPED_TEST(Level3Test, Herk) {
  using T = TypeParam;
  if constexpr (IsComplex<T>::value) {
    ForEachTarget([] {
      uint64_t seed = 1;
      for (char uplo : {'L', 'U'}) {
        for (char trans : {'N', 'C'}) {
          for (int n : kSizes) {
            CheckSyrk<T>(true, uplo, trans, n, 41, 1.5, 0.5, seed++);
            CheckSyrk<T>(true, uplo, trans, n, 300, -1.0, 0.0, seed++);
            CheckSyrk<T>(true, uplo, trans, n, 7, 0.0, 2.0, seed++);
          }
        }
      }
    });
  }
}

// ---------------------------------------------------------------------------
// SYR2K / HER2K / GEMMTR: products that update one triangle of C.

enum class TriangleRoutine { kSyr2k, kHer2k, kGemmtr };

// Dense op(X)' (n x k) for op(X) given as a dense n x k matrix, where ' is
// ^H if conj, else ^T.
template <typename T>
Dense<T> Transposed(const Dense<T>& x, bool conj) {
  Dense<T> t(x.cols, x.rows);
  for (size_t j = 0; j < x.rows; ++j) {
    for (size_t p = 0; p < x.cols; ++p) {
      t(p, j) = conj ? Conj(x(j, p)) : x(j, p);
    }
  }
  return t;
}

// For SYR2K/HER2K, trans_a is TRANS and trans_b is ignored.
template <typename T>
void CheckTriangle(TriangleRoutine routine, char uplo, char trans_a,
                   char trans_b, int n, int k, T alpha, T beta, uint64_t seed) {
  const bool her2k = (routine == TriangleRoutine::kHer2k);
  if (routine != TriangleRoutine::kGemmtr) trans_b = trans_a;
  if (her2k) beta = T(std::real(beta));
  // op(A) is n x k. For GEMMTR op(B) is k x n; for SYR2K/HER2K op(B) has the
  // shape of op(A).
  const bool ta = (trans_a != 'N'), tb = (trans_b != 'N');
  const bool b_like_a = (routine != TriangleRoutine::kGemmtr);
  const int rows_a = ta ? k : n, cols_a = ta ? n : k;
  const int rows_b = b_like_a ? rows_a : (tb ? n : k);
  const int cols_b = b_like_a ? cols_a : (tb ? k : n);
  const int lda = rows_a + 1, ldb = rows_b + 3, ldc = n + 2;

  Random<T> rng(seed);
  const std::vector<T> a = rng.Vector(static_cast<size_t>(lda) * cols_a);
  const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * cols_b);
  std::vector<T> c = rng.Vector(static_cast<size_t>(ldc) * n);
  const auto in_triangle = [&](int i, int j) {
    return uplo == 'L' ? (i >= j) : (i <= j);
  };
  // The other triangle must stay untouched; for HER2K the imaginary parts of
  // the diagonal of C must not be read; for beta = 0, C must not be read.
  const T sentinel = MakeScalar<T>(7.0, 7.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      T& cij = c[i + static_cast<size_t>(j) * ldc];
      if (!in_triangle(i, j)) cij = sentinel;
      if (i == j && her2k) cij = MakeScalar<T>(std::real(cij), NaN<double>());
      if (beta == T(0) && in_triangle(i, j)) cij = NaN<T>();
    }
  }
  std::vector<T> c0 = c;
  if (her2k && !(beta == T(0))) {
    for (int j = 0; j < n; ++j) {
      T& cjj = c0[j + static_cast<size_t>(j) * ldc];
      cjj = T(std::real(cjj));
    }
  }

  std::vector<RefT<T>> expected, expected2;
  std::vector<double> bound, bound2;
  const Dense<T> da = DenseOp(trans_a, a, rows_a, cols_a, lda);
  if (routine == TriangleRoutine::kGemmtr) {
    const Dense<T> db = DenseOp(trans_b, b, rows_b, cols_b, ldb);
    ReferenceProduct(da, db, alpha, beta, c0, ldc, &expected, &bound);
  } else {
    // op(B) has the shape of op(A); ' is ^H for HER2K, ^T for SYR2K.
    const Dense<T> db = DenseOp(trans_a, b, rows_b, cols_b, ldb);
    const T alpha2 = her2k ? Conj(alpha) : alpha;
    ReferenceProduct(da, Transposed(db, her2k), alpha, beta, c0, ldc, &expected,
                     &bound);
    ReferenceProduct(db, Transposed(da, her2k), alpha2, T(0), c0, ldc,
                     &expected2, &bound2);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] += expected2[i];
      bound[i] += bound2[i];
    }
  }

  switch (routine) {
    case TriangleRoutine::kSyr2k:
      HWYBLAS_CALL(T, syr2k, &uplo, &trans_a, &n, &k, P(&alpha), P(a.data()),
                   &lda, P(b.data()), &ldb, P(&beta), P(c.data()), &ldc);
      break;
    case TriangleRoutine::kHer2k: {
      const RealT<T> real_beta = std::real(beta);
      HWYBLAS_CALL_COMPLEX(T, her2k, &uplo, &trans_a, &n, &k, P(&alpha),
                           P(a.data()), &lda, P(b.data()), &ldb, &real_beta,
                           P(c.data()), &ldc);
      break;
    }
    case TriangleRoutine::kGemmtr:
      HWYBLAS_CALL(T, gemmtr, &uplo, &trans_a, &trans_b, &n, &k, P(&alpha),
                   P(a.data()), &lda, P(b.data()), &ldb, P(&beta), P(c.data()),
                   &ldc);
      break;
  }

  const double tol = 4.0 * (2 * k + 2) * Eps<T>();
  size_t failures = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const size_t idx = i + static_cast<size_t>(j) * ldc;
      if (!in_triangle(i, j)) {
        if (!(c[idx] == sentinel)) ++failures;
        continue;
      }
      const RefT<T> got = static_cast<RefT<T>>(c[idx]);
      const double err = std::abs(got - expected[idx]);
      if (!(err <= tol * bound[idx])) ++failures;
      if (her2k && i == j && std::imag(c[idx]) != 0) ++failures;
    }
  }
  EXPECT_EQ(failures, 0u) << "routine=" << static_cast<int>(routine)
                          << " uplo=" << uplo << " trans=" << trans_a << trans_b
                          << " n=" << n << " k=" << k << " alpha=" << alpha
                          << " beta=" << beta;
}

TYPED_TEST(Level3Test, Syr2kHer2k) {
  using T = TypeParam;
  // SYR2K for all types; HER2K (trans N or C) for complex types.
  std::vector<std::pair<TriangleRoutine, char>> cases = {
      {TriangleRoutine::kSyr2k, 'N'}, {TriangleRoutine::kSyr2k, 'T'}};
  if constexpr (IsComplex<T>::value) {
    cases.push_back({TriangleRoutine::kHer2k, 'N'});
    cases.push_back({TriangleRoutine::kHer2k, 'C'});
  } else {
    cases.push_back({TriangleRoutine::kSyr2k, 'C'});  // same as T
  }
  ForEachTarget([&] {
    uint64_t seed = 1;
    for (const auto& [routine, trans] : cases) {
      for (char uplo : {'L', 'U'}) {
        for (int n : kSizes) {
          for (int k : {1, 9, 300}) {
            CheckTriangle<T>(routine, uplo, trans, 'N', n, k,
                             MakeScalar<T>(1.5, 0.5), MakeScalar<T>(0.5, -1),
                             seed++);
          }
          CheckTriangle<T>(routine, uplo, trans, 'N', n, 20,
                           MakeScalar<T>(-1.0, 0.25), T(0), seed++);
          CheckTriangle<T>(routine, uplo, trans, 'N', n, 20, T(0),
                           MakeScalar<T>(2.0, 0.0), seed++);
          CheckTriangle<T>(routine, uplo, trans, 'N', n, 0, T(1),
                           MakeScalar<T>(0.5, 0.0), seed++);
        }
      }
    }
  });
}

TYPED_TEST(Level3Test, Gemmtr) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (char uplo : {'L', 'U'}) {
      for (char ta : Ops<T>()) {
        for (char tb : Ops<T>()) {
          for (int n : kSizes) {
            CheckTriangle<T>(TriangleRoutine::kGemmtr, uplo, ta, tb, n, 37,
                             MakeScalar<T>(1.5, 0.5), MakeScalar<T>(0.5, -1),
                             seed++);
          }
          CheckTriangle<T>(TriangleRoutine::kGemmtr, uplo, ta, tb, 100, 300,
                           MakeScalar<T>(-1.0, 0.0), T(0), seed++);
          CheckTriangle<T>(TriangleRoutine::kGemmtr, uplo, ta, tb, 13, 0, T(1),
                           MakeScalar<T>(2.0, 1.0), seed++);
        }
      }
    }
  });
}

TYPED_TEST(Level3Test, GemmtIsGemmtr) {
  using T = TypeParam;
  const int n = 5, k = 3;
  Random<T> rng(7);
  const std::vector<T> a = rng.Vector(n * k), b = rng.Vector(k * n);
  std::vector<T> c1 = rng.Vector(n * n), c2 = c1;
  const T alpha = MakeScalar<T>(1.5, 0.5), beta = MakeScalar<T>(0.5, 0.0);
  const char uplo = 'U', tn = 'N';
  HWYBLAS_CALL(T, gemmtr, &uplo, &tn, &tn, &n, &k, P(&alpha), P(a.data()), &n,
               P(b.data()), &k, P(&beta), P(c1.data()), &n);
  HWYBLAS_CALL(T, gemmt, &uplo, &tn, &tn, &n, &k, P(&alpha), P(a.data()), &n,
               P(b.data()), &k, P(&beta), P(c2.data()), &n);
  EXPECT_EQ(c1, c2);
}

// ---------------------------------------------------------------------------
// TRMM / TRSM

struct TriCase {
  char side, uplo, trans, diag;
};

// All 16 (or 24 for complex) combinations of side, uplo, trans and diag.
template <typename T>
std::vector<TriCase> TriCases() {
  std::vector<TriCase> cases;
  for (char side : {'L', 'R'}) {
    for (char uplo : {'L', 'U'}) {
      for (char trans : Ops<T>()) {
        for (char diag : {'N', 'U'}) cases.push_back({side, uplo, trans, diag});
      }
    }
  }
  return cases;
}

template <typename T>
void CheckTrmm(const TriCase& tc, int m, int n, T alpha, uint64_t seed) {
  const int ka = (tc.side == 'L') ? m : n;
  const int lda = ka + 1, ldb = m + 2;
  const Shape shape =
      (tc.diag == 'U') ? Shape::kUnitTriangular : Shape::kTriangular;
  Random<T> rng(seed);
  const std::vector<T> a = StructuredMatrix(shape, tc.uplo, ka, lda, rng);
  std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * n);
  const std::vector<T> b0 = b;

  const Dense<T> da = DenseStructured(shape, tc.uplo, tc.trans, a, ka, lda);
  const Dense<T> db = DenseOp('N', b0, m, n, ldb);
  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  if (tc.side == 'L') {
    ReferenceProduct(da, db, alpha, T(0), b0, ldb, &expected, &bound);
  } else {
    ReferenceProduct(db, da, alpha, T(0), b0, ldb, &expected, &bound);
  }

  HWYBLAS_CALL(T, trmm, &tc.side, &tc.uplo, &tc.trans, &tc.diag, &m, &n,
               P(&alpha), P(a.data()), &lda, P(b.data()), &ldb);
  const double tol = 4.0 * (ka + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(b, expected, bound, tol, m, n, ldb))
      << "side=" << tc.side << " uplo=" << tc.uplo << " trans=" << tc.trans
      << " diag=" << tc.diag << " m=" << m << " n=" << n;
}

TYPED_TEST(Level3Test, Trmm) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (const TriCase& tc : TriCases<T>()) {
      for (int m : kSizes) {
        CheckTrmm<T>(tc, m, 19, MakeScalar<T>(1.5, -0.5), seed++);
        CheckTrmm<T>(tc, 11, m, MakeScalar<T>(-1.0, 0.0), seed++);
      }
    }
  });
}

// Checks the solution X of a TRSM through the residual op(A) X - alpha B,
// which is small relative to |op(A)| |X| for a backward stable solver. ldb
// defaults to m + 2.
template <typename T>
void CheckTrsm(const TriCase& tc, int m, int n, T alpha, uint64_t seed,
               int ldb = 0) {
  const int ka = (tc.side == 'L') ? m : n;
  const int lda = ka + 1;
  if (ldb == 0) ldb = m + 2;
  const Shape shape =
      (tc.diag == 'U') ? Shape::kUnitTriangular : Shape::kTriangular;
  Random<T> rng(seed);
  // Scaled off-diagonal part: random triangular matrices are otherwise
  // exponentially ill-conditioned.
  const std::vector<T> a =
      StructuredMatrix(shape, tc.uplo, ka, lda, rng, 1.0 / ka);
  std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * n);
  const std::vector<T> b0 = b;

  HWYBLAS_CALL(T, trsm, &tc.side, &tc.uplo, &tc.trans, &tc.diag, &m, &n,
               P(&alpha), P(a.data()), &lda, P(b.data()), &ldb);

  // Residual check: op(A) X (left) or X op(A) (right) against alpha * B0.
  const Dense<T> da = DenseStructured(shape, tc.uplo, tc.trans, a, ka, lda);
  const Dense<T> dx = DenseOp('N', b, m, n, ldb);
  std::vector<RefT<T>> product;
  std::vector<double> bound;
  if (tc.side == 'L') {
    ReferenceProduct(da, dx, T(1), T(0), b0, ldb, &product, &bound);
  } else {
    ReferenceProduct(dx, da, T(1), T(0), b0, ldb, &product, &bound);
  }
  std::vector<T> rhs(b0.size());
  for (size_t i = 0; i < b0.size(); ++i) {
    rhs[i] = alpha * b0[i];
    bound[i] += Abs(rhs[i]);
  }
  const double tol = 8.0 * (ka + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(rhs, product, bound, tol, m, n, ldb))
      << "side=" << tc.side << " uplo=" << tc.uplo << " trans=" << tc.trans
      << " diag=" << tc.diag << " m=" << m << " n=" << n;
}

TYPED_TEST(Level3Test, Trsm) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (const TriCase& tc : TriCases<T>()) {
      for (int m : {1, 2, 5, 31, 32, 33, 100, 300}) {
        CheckTrsm<T>(tc, m, 17, MakeScalar<T>(1.5, -0.5), seed++);
        CheckTrsm<T>(tc, 13, m, MakeScalar<T>(1.0, 0.0), seed++);
      }
      // Leading dimension a multiple of 1 KiB: the right side then solves on
      // copies of blocks of rows.
      const int unit = static_cast<int>(1024 / sizeof(T));
      CheckTrsm<T>(tc, 300, 45, MakeScalar<T>(0.5, 1.0), seed++,
                   (300 + unit - 1) / unit * unit);
    }
  });
}

TYPED_TEST(Level3Test, TrmmTrsmAlphaZero) {
  using T = TypeParam;
  // alpha == 0 sets B to zero without reading A (here all NaN).
  const int m = 3, n = 2, ld = 3;
  std::vector<T> a(9, NaN<T>());
  std::vector<T> b(6, T(1));
  const T zero(0);
  const char side = 'L', uplo = 'L', trans = 'N', diag = 'N';
  HWYBLAS_CALL(T, trmm, &side, &uplo, &trans, &diag, &m, &n, P(&zero),
               P(a.data()), &ld, P(b.data()), &ld);
  for (const T& x : b) EXPECT_EQ(x, T(0));
  b.assign(6, T(1));
  HWYBLAS_CALL(T, trsm, &side, &uplo, &trans, &diag, &m, &n, P(&zero),
               P(a.data()), &ld, P(b.data()), &ld);
  for (const T& x : b) EXPECT_EQ(x, T(0));
}

}  // namespace
}  // namespace hwyblas_test
