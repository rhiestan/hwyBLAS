// Tests of the symmetric (xSY*) and Hermitian (xHE*) indefinite linear
// systems routines: the Bunch-Kaufman factorization xSYTRF / xHETRF (with
// xLASYF / xLAHEF and xSYTF2 / xHETF2), the solvers xSYTRS, xSYTRS2,
// xHETRS, xHETRS2 and xSYCONV, the inverse xSYTRI / xHETRI, xSYCON /
// xHECON, xSYRFS / xHERFS, the drivers xSYSV / xHESV and xSYSVX / xHESVX,
// LAPACK's complex symmetric CSYMV / ZSYMV and CSYR / ZSYR, and their
// LAPACKE functions.
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
#include "lapack/linear_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class SysvTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SysvTest, ScalarTypes);

// The kinds of matrices of T: symmetric (herm = false) and, for complex T,
// Hermitian (herm = true).
template <typename T>
std::vector<bool> Kinds() {
  if constexpr (IsComplex<T>::value) {
    return {false, true};
  } else {
    return {false};
  }
}

// Calls the xSY* routine `sy` or, if herm, the xHE* routine `he`.
#define SY_OR_HE(herm, T, sy, he, ...)          \
  do {                                          \
    if (herm) {                                 \
      HWYBLAS_CALL_COMPLEX(T, he, __VA_ARGS__); \
    } else {                                    \
      HWYBLAS_CALL(T, sy, __VA_ARGS__);         \
    }                                           \
  } while (0)

// The routine name of the kind, e.g. SyName<double>(false, "TRF") =
// "DSYTRF".
template <typename T>
std::string SyName(bool herm, const char* rest) {
  return Name<T>((std::string(herm ? "HE" : "SY") + rest).c_str());
}

// The symmetric or Hermitian matrix of the triangle uplo of a.
template <typename T>
Dense<T> FullOf(bool herm, char uplo, const std::vector<T>& a, size_t n,
                size_t lda) {
  return herm ? HermitianOf(uplo, a, n, lda) : SymmetricOf(uplo, a, n, lda);
}

// xSYTRF / xHETRF, or xSYTF2 / xHETF2 if unblocked.
template <typename T>
hwyblas_int Sytrf(bool herm, bool unblocked, char uplo, hwyblas_int n,
                  std::vector<T>& a, hwyblas_int lda,
                  std::vector<hwyblas_int>& ipiv, std::vector<T>& work,
                  hwyblas_int lwork) {
  hwyblas_int info = -99;
  ipiv.resize(std::max<hwyblas_int>(1, n));
  if (unblocked) {
    SY_OR_HE(herm, T, sytf2, hetf2, &uplo, &n, P(a.data()), &lda, ipiv.data(),
             &info);
  } else {
    SY_OR_HE(herm, T, sytrf, hetrf, &uplo, &n, P(a.data()), &lda, ipiv.data(),
             P(work.data()), &lwork, &info);
  }
  return info;
}

// xSYTRF / xHETRF with the optimal workspace.
template <typename T>
hwyblas_int Factor(bool herm, char uplo, hwyblas_int n, std::vector<T>& a,
                   hwyblas_int lda, std::vector<hwyblas_int>& ipiv) {
  std::vector<T> work(1);
  EXPECT_EQ(Sytrf(herm, false, uplo, n, a, lda, ipiv, work, -1), 0);
  const auto lwork = static_cast<hwyblas_int>(std::real(work[0]));
  work.resize(std::max<hwyblas_int>(1, lwork));
  return Sytrf(herm, false, uplo, n, a, lda, ipiv, work, lwork);
}

// xSYTRS / xHETRS, or xSYTRS2 / xHETRS2 if level3.
template <typename T>
hwyblas_int Sytrs(bool herm, bool level3, char uplo, hwyblas_int n,
                  hwyblas_int nrhs, std::vector<T>& a, hwyblas_int lda,
                  const std::vector<hwyblas_int>& ipiv, std::vector<T>& b,
                  hwyblas_int ldb) {
  hwyblas_int info = -99;
  if (level3) {
    std::vector<T> work(std::max<hwyblas_int>(1, n));
    SY_OR_HE(herm, T, sytrs2, hetrs2, &uplo, &n, &nrhs, P(a.data()), &lda,
             ipiv.data(), P(b.data()), &ldb, P(work.data()), &info);
  } else {
    SY_OR_HE(herm, T, sytrs, hetrs, &uplo, &n, &nrhs, P(a.data()), &lda,
             ipiv.data(), P(b.data()), &ldb, &info);
  }
  return info;
}

template <typename T>
hwyblas_int Syconv(char uplo, char way, hwyblas_int n, std::vector<T>& a,
                   hwyblas_int lda, const std::vector<hwyblas_int>& ipiv,
                   std::vector<T>& e) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, syconv, &uplo, &way, &n, P(a.data()), &lda, ipiv.data(),
               P(e.data()), &info);
  return info;
}

template <typename T>
hwyblas_int Sytri(bool herm, char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda, const std::vector<hwyblas_int>& ipiv) {
  hwyblas_int info = -99;
  std::vector<T> work(std::max<hwyblas_int>(1, n));
  SY_OR_HE(herm, T, sytri, hetri, &uplo, &n, P(a.data()), &lda, ipiv.data(),
           P(work.data()), &info);
  return info;
}

// xSYCON / xHECON with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Sycon(bool herm, char uplo, hwyblas_int n, const std::vector<T>& a,
                  hwyblas_int lda, const std::vector<hwyblas_int>& ipiv,
                  RealT<T> anorm, RealT<T>* rcond) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  std::vector<T> work(2 * n1);
  if constexpr (IsComplex<T>::value) {
    SY_OR_HE(herm, T, sycon, hecon, &uplo, &n, P(a.data()), &lda, ipiv.data(),
             &anorm, rcond, P(work.data()), &info);
  } else {
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, sycon, &uplo, &n, a.data(), &lda, ipiv.data(), &anorm,
                      rcond, work.data(), iwork.data(), &info);
  }
  return info;
}

// xSYRFS / xHERFS with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Syrfs(bool herm, char uplo, hwyblas_int n, hwyblas_int nrhs,
                  const std::vector<T>& a, hwyblas_int lda,
                  const std::vector<T>& af, hwyblas_int ldaf,
                  const std::vector<hwyblas_int>& ipiv, const std::vector<T>& b,
                  hwyblas_int ldb, std::vector<T>& x, hwyblas_int ldx,
                  std::vector<RealT<T>>& ferr, std::vector<RealT<T>>& berr) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    SY_OR_HE(herm, T, syrfs, herfs, &uplo, &n, &nrhs, P(a.data()), &lda,
             P(af.data()), &ldaf, ipiv.data(), P(b.data()), &ldb, P(x.data()),
             &ldx, ferr.data(), berr.data(), P(work.data()), rwork.data(),
             &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, syrfs, &uplo, &n, &nrhs, a.data(), &lda, af.data(),
                      &ldaf, ipiv.data(), b.data(), &ldb, x.data(), &ldx,
                      ferr.data(), berr.data(), work.data(), iwork.data(),
                      &info);
  }
  return info;
}

template <typename T>
hwyblas_int Sysv(bool herm, char uplo, hwyblas_int n, hwyblas_int nrhs,
                 std::vector<T>& a, hwyblas_int lda,
                 std::vector<hwyblas_int>& ipiv, std::vector<T>& b,
                 hwyblas_int ldb, std::vector<T>& work, hwyblas_int lwork) {
  hwyblas_int info = -99;
  ipiv.resize(std::max<hwyblas_int>(1, n));
  SY_OR_HE(herm, T, sysv, hesv, &uplo, &n, &nrhs, P(a.data()), &lda,
           ipiv.data(), P(b.data()), &ldb, P(work.data()), &lwork, &info);
  return info;
}

// The arguments and results of xSYSVX / xHESVX.
template <typename T>
struct SysvxCall {
  using R = RealT<T>;
  bool herm = false;
  char fact = 'N', uplo = 'U';
  hwyblas_int n = 0, nrhs = 0, lda = 1, ldaf = 1, ldb = 1, ldx = 1;
  hwyblas_int lwork = 0;  // 0: the minimal size
  std::vector<T> a, af, b, x, work;
  std::vector<hwyblas_int> ipiv;
  std::vector<R> ferr, berr;
  R rcond = -1;

  // The minimal LWORK.
  hwyblas_int MinWork() const {
    return std::max<hwyblas_int>(1, (IsComplex<T>::value ? 2 : 3) * n);
  }

  // Calls xSYSVX / xHESVX; returns INFO.
  hwyblas_int Run() {
    const size_t n1 = std::max<hwyblas_int>(1, n);
    af.resize(static_cast<size_t>(ldaf) * n1);
    x.resize(static_cast<size_t>(ldx) * std::max<hwyblas_int>(1, nrhs));
    ipiv.resize(n1);
    ferr.assign(std::max<hwyblas_int>(1, nrhs), -1);
    berr.assign(std::max<hwyblas_int>(1, nrhs), -1);
    if (lwork == 0) lwork = MinWork();
    work.resize(std::max<hwyblas_int>(1, lwork));
    hwyblas_int info = -99;
    if constexpr (IsComplex<T>::value) {
      std::vector<R> rwork(n1);
      SY_OR_HE(herm, T, sysvx, hesvx, &fact, &uplo, &n, &nrhs, P(a.data()),
               &lda, P(af.data()), &ldaf, ipiv.data(), P(b.data()), &ldb,
               P(x.data()), &ldx, &rcond, ferr.data(), berr.data(),
               P(work.data()), &lwork, rwork.data(), &info);
    } else {
      std::vector<hwyblas_int> iwork(n1);
      HWYBLAS_CALL_REAL(T, sysvx, &fact, &uplo, &n, &nrhs, a.data(), &lda,
                        af.data(), &ldaf, ipiv.data(), b.data(), &ldb, x.data(),
                        &ldx, &rcond, ferr.data(), berr.data(), work.data(),
                        &lwork, iwork.data(), &info);
    }
    return info;
  }
};

// The test matrices.
enum class Matrix {
  // Random, with the diagonal shifted by +-n/2 alternately: indefinite and
  // well-conditioned; mostly 1-by-1 pivots, with interchanges.
  kShifted,
  // [0 B; B**H 0] (B**T for symmetric matrices) with a diagonally dominant
  // B (and a 1 at the end for odd n): mostly 2-by-2 pivots.
  kSaddle,
};

// A random symmetric (Hermitian) n x n matrix in the triangle uplo of a
// (leading dimension lda), with Sentinel() elsewhere.
template <typename T>
std::vector<T> Indefinite(bool herm, Matrix kind, char uplo, hwyblas_int n,
                          hwyblas_int lda, Random<T>& rng) {
  Dense<T> full(n, n);
  if (kind == Matrix::kShifted) {
    for (hwyblas_int j = 0; j < n; ++j) {
      for (hwyblas_int i = 0; i <= j; ++i) {
        RefT<T> v = static_cast<RefT<T>>(rng.Next());
        if (i == j) {
          if (herm) v = std::real(v);
          v += (j % 2 == 0 ? 0.5 : -0.5) * n;
        }
        full(i, j) = v;
        full(j, i) = herm ? Conj(v) : v;
      }
    }
  } else {
    const hwyblas_int p = n / 2;
    for (hwyblas_int j = 0; j < p; ++j) {
      for (hwyblas_int i = 0; i < p; ++i) {
        RefT<T> v = static_cast<RefT<T>>(rng.Next());
        if (i == j) v += 0.5 * p + 1;
        full(i, p + j) = v;
        full(p + j, i) = herm ? Conj(v) : v;
      }
    }
    if (n % 2 == 1) full(n - 1, n - 1) = 1;
  }
  std::vector<T> a(static_cast<size_t>(lda) * std::max<hwyblas_int>(n, 1),
                   Sentinel<T>());
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      if (InTriangle(uplo, i, j)) At(a, i, j, lda) = static_cast<T>(full(i, j));
    }
  }
  return a;
}

// The test ratio of LAPACK's xSYT02: ||A * X - B||_1 / (n ||A||_1 ||X||_1
// eps).
template <typename T>
double SolveRatio(const Dense<T>& da, const std::vector<T>& x, hwyblas_int ldx,
                  const std::vector<T>& b, hwyblas_int ldb, hwyblas_int n,
                  hwyblas_int nrhs) {
  const Dense<T> dx = ToDense(x, n, nrhs, ldx);
  const double resid = Norm1(Sub(Mul(da, dx), ToDense(b, n, nrhs, ldb)));
  return resid / std::max<hwyblas_int>(1, n) / Norm1(da) / Norm1(dx) /
         LapackEps<T>();
}

// Checks that the elements of a outside the triangle uplo (and below row n)
// are those of a0.
template <typename T>
void ExpectOutsideUnchanged(char uplo, const std::vector<T>& a,
                            const std::vector<T>& a0, hwyblas_int n,
                            hwyblas_int lda) {
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < lda; ++i) {
      if (i < n && InTriangle(uplo, i, j)) continue;
      ASSERT_EQ(At(a, i, j, lda), At(a0, i, j, lda)) << i << ", " << j;
    }
  }
}

// ---------------------------------------------------------------------------
// Factorization and solution

// xSYTRF / xHETRF with block sizes that select the panel factorization
// (xLASYF / xLAHEF) and the unblocked code, and xSYTF2 / xHETF2, on both
// kinds of test matrices; the solutions of xSYTRS and xSYTRS2 (xHETRS,
// xHETRS2), which restores A. IPIV describes 1-by-1 and 2-by-2 blocks; the
// diagonal of D is real for Hermitian A; nothing outside the triangle
// changes.
TYPED_TEST(SysvTest, FactorSolve) {
  using T = TypeParam;
  Random<T> rng(7);
  for (const bool herm : Kinds<T>()) {
    for (const hwyblas_int nb : BlockSizes()) {
      const ScopedBlockSize block_size(nb);
      for (const bool unblocked : {false, true}) {
        if (unblocked && nb != 0) continue;
        for (const Matrix kind : {Matrix::kShifted, Matrix::kSaddle}) {
          for (const char uplo : {'U', 'L'}) {
            for (const hwyblas_int n : {0, 1, 2, 5, 16, 33, 70}) {
              SCOPED_TRACE(::testing::Message()
                           << (herm ? "he" : "sy")
                           << (unblocked ? "tf2" : "trf") << " " << uplo
                           << " n=" << n << " nb=" << nb
                           << " kind=" << static_cast<int>(kind));
              const hwyblas_int lda = n + 2, ldb = n + 1, nrhs = 3;
              const std::vector<T> a =
                  Indefinite<T>(herm, kind, uplo, n, lda, rng);
              std::vector<T> f = a;
              std::vector<hwyblas_int> ipiv;
              hwyblas_int info;
              if (unblocked) {
                std::vector<T> work(1);
                info = Sytrf(herm, true, uplo, n, f, lda, ipiv, work, 1);
              } else {
                info = Factor(herm, uplo, n, f, lda, ipiv);
              }
              ASSERT_EQ(info, 0);
              ExpectOutsideUnchanged(uplo, f, a, n, lda);
              // IPIV: 2-by-2 blocks are pairs of equal negative entries.
              bool two_by_two = false;
              for (hwyblas_int k = 0; k < n; ++k) {
                const hwyblas_int p = ipiv[k];
                ASSERT_NE(p, 0);
                ASSERT_LE(std::abs(p), n);
                if (p < 0) two_by_two = true;
                if (herm && p > 0) {
                  ASSERT_EQ(std::imag(At(f, k, k, lda)), 0) << k;
                }
              }
              if (kind == Matrix::kSaddle && n >= 2) {
                EXPECT_TRUE(two_by_two);
              }
              if (n == 0) continue;
              const Dense<T> da = FullOf(herm, uplo, a, n, lda);
              const std::vector<T> b =
                  rng.Vector(static_cast<size_t>(ldb) * nrhs);
              for (const bool level3 : {false, true}) {
                SCOPED_TRACE(level3 ? "trs2" : "trs");
                std::vector<T> x = b, f2 = f;
                ASSERT_EQ(
                    Sytrs(herm, level3, uplo, n, nrhs, f2, lda, ipiv, x, ldb),
                    0);
                EXPECT_EQ(f2, f);  // xSYTRS2 restores A
                EXPECT_LT(SolveRatio(da, x, ldb, b, ldb, n, nrhs),
                          kLapackThreshold);
              }
              // The inverse as the solution of A * X = I.
              std::vector<T> id(static_cast<size_t>(n) * n, T(0));
              for (hwyblas_int i = 0; i < n; ++i) At(id, i, i, n) = T(1);
              std::vector<T> x = id;
              ASSERT_EQ(Sytrs(herm, true, uplo, n, n, f, lda, ipiv, x, n), 0);
              EXPECT_LT(SolveRatio(da, x, n, id, n, n, n), kLapackThreshold);
            }
          }
        }
      }
    }
  }
}

// The pairs in IPIV of 2-by-2 blocks: for UPLO = 'U' IPIV(k-1) = IPIV(k) <
// 0, for 'L' IPIV(k) = IPIV(k+1) < 0, scanning as the solvers do.
TYPED_TEST(SysvTest, PivotStructure) {
  using T = TypeParam;
  Random<T> rng(9);
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      const hwyblas_int n = 21, lda = n;
      std::vector<T> f =
          Indefinite<T>(herm, Matrix::kSaddle, uplo, n, lda, rng);
      std::vector<hwyblas_int> ipiv;
      ASSERT_EQ(Factor(herm, uplo, n, f, lda, ipiv), 0);
      hwyblas_int blocks = 0;
      if (uplo == 'U') {
        for (hwyblas_int k = n - 1; k >= 0; --k) {
          if (ipiv[k] < 0) {
            ASSERT_GE(k, 1);
            EXPECT_EQ(ipiv[k - 1], ipiv[k]);
            --k;
            ++blocks;
          }
        }
      } else {
        for (hwyblas_int k = 0; k < n; ++k) {
          if (ipiv[k] < 0) {
            ASSERT_LT(k, n - 1);
            EXPECT_EQ(ipiv[k + 1], ipiv[k]);
            ++k;
            ++blocks;
          }
        }
      }
      EXPECT_GE(blocks, 5);
    }
  }
}

// xSYTRF / xHETRF with a workspace below the optimal size factor with
// smaller panels (NB = LWORK / N, here 10) or, below the minimal block size,
// unblocked code, and give a factorization that solves the system; the
// workspace query returns N * NB (NB from ILAENV).
TYPED_TEST(SysvTest, FactorWorkspaceSizes) {
  using T = TypeParam;
  Random<T> rng(11);
  // n > 64 (ILAENV's NB), so that the optimal size is N * NB.
  const hwyblas_int n = 100, lda = n;
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      const std::vector<T> a =
          Indefinite<T>(herm, Matrix::kShifted, uplo, n, lda, rng);
      const Dense<T> da = FullOf(herm, uplo, a, n, lda);
      std::vector<T> q(1);
      std::vector<hwyblas_int> ipiv;
      std::vector<T> f = a;
      ASSERT_EQ(Sytrf(herm, false, uplo, n, f, lda, ipiv, q, -1), 0);
      EXPECT_EQ(f, a);                               // only a query
      EXPECT_EQ(std::real(q[0]), RealT<T>(64 * n));  // ILAENV's NB = 64
      for (const hwyblas_int lwork : {1, n - 1, 3 * n, 10 * n}) {
        SCOPED_TRACE(::testing::Message()
                     << (herm ? "he " : "sy ") << uplo << " lwork=" << lwork);
        std::vector<T> work(lwork);
        f = a;
        ASSERT_EQ(Sytrf(herm, false, uplo, n, f, lda, ipiv, work, lwork), 0);
        EXPECT_EQ(std::real(work[0]), RealT<T>(64 * n));
        const std::vector<T> b = rng.Vector(n);
        std::vector<T> x = b;
        ASSERT_EQ(Sytrs(herm, false, uplo, n, 1, f, lda, ipiv, x, n), 0);
        EXPECT_LT(SolveRatio(da, x, n, b, n, n, 1), kLapackThreshold);
      }
    }
  }
}

// xSYSV / xHESV give the factorization of xSYTRF / xHETRF and the solution
// of xSYTRS2 / xHETRS2 (LWORK >= N) or xSYTRS / xHETRS (LWORK < N). The
// workspace query: xSYSV that of xSYTRF, xHESV N * NB (1 for N = 0).
TYPED_TEST(SysvTest, Driver) {
  using T = TypeParam;
  Random<T> rng(13);
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {0, 1, 7, 40}) {
        const hwyblas_int lda = n + 1, ldb = n + 2, nrhs = 2;
        const std::vector<T> a =
            Indefinite<T>(herm, Matrix::kShifted, uplo, n, lda, rng);
        const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
        std::vector<T> q(1);
        std::vector<hwyblas_int> ipiv;
        std::vector<T> f = a, x = b;
        ASSERT_EQ(Sysv(herm, uplo, n, nrhs, f, lda, ipiv, x, ldb, q, -1), 0);
        EXPECT_EQ(std::real(q[0]), RealT<T>(n == 0 ? 1 : 64 * n));
        const auto lopt = static_cast<hwyblas_int>(std::real(q[0]));
        for (const hwyblas_int lwork : {lopt, n - 1}) {
          if (lwork < 1) continue;
          SCOPED_TRACE(::testing::Message()
                       << (herm ? "he " : "sy ") << uplo << " n=" << n
                       << " lwork=" << lwork);
          std::vector<T> work(lwork);
          f = a;
          x = b;
          ASSERT_EQ(
              Sysv(herm, uplo, n, nrhs, f, lda, ipiv, x, ldb, work, lwork), 0);
          EXPECT_EQ(std::real(work[0]), RealT<T>(n == 0 ? 1 : 64 * n));
          // The same with the separate routines.
          std::vector<T> g = a, y = b, w2(lwork);
          std::vector<hwyblas_int> ipiv2;
          ASSERT_EQ(Sytrf(herm, false, uplo, n, g, lda, ipiv2, w2, lwork), 0);
          ASSERT_EQ(
              Sytrs(herm, lwork >= n, uplo, n, nrhs, g, lda, ipiv2, y, ldb), 0);
          EXPECT_EQ(f, g);
          EXPECT_EQ(ipiv, ipiv2);
          EXPECT_EQ(x, y);
          if (n > 0) {
            EXPECT_LT(SolveRatio(FullOf(herm, uplo, a, n, lda), x, ldb, b, ldb,
                                 n, nrhs),
                      kLapackThreshold);
          }
        }
      }
    }
  }
}

// A zero column (and row) k: INFO = k for xSYTRF / xHETRF (blocked and
// unblocked) and xSYSV / xHESV, which then does not solve; xSYTRI / xHETRI
// return INFO = k too.
TYPED_TEST(SysvTest, Singular) {
  using T = TypeParam;
  Random<T> rng(17);
  const hwyblas_int n = 30, k = 12;
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int nb : {0, 1, 3}) {
        SCOPED_TRACE(::testing::Message()
                     << (herm ? "he " : "sy ") << uplo << " nb=" << nb);
        const ScopedBlockSize block_size(nb);
        std::vector<T> a =
            Indefinite<T>(herm, Matrix::kShifted, uplo, n, n, rng);
        for (hwyblas_int i = 0; i < n; ++i) {
          At(a, i, k - 1, n) = T(0);
          At(a, k - 1, i, n) = T(0);
        }
        std::vector<T> f = a;
        std::vector<hwyblas_int> ipiv;
        // The zero pivot is found when column k is reached: first or last.
        const hwyblas_int info = Factor(herm, uplo, n, f, n, ipiv);
        EXPECT_EQ(info, k);
        std::vector<T> inv = f;
        EXPECT_EQ(Sytri(herm, uplo, n, inv, n, ipiv), k);
        const std::vector<T> b = rng.Vector(n);
        std::vector<T> x = b, g = a, work(64 * n);
        EXPECT_EQ(Sysv(herm, uplo, n, 1, g, n, ipiv, x, n, work, 64 * n), k);
        EXPECT_EQ(x, b);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Inverse, conversion, condition number

// The test ratio of LAPACK's xSYT03: ||I - A * inv(A)||_1 /
// (n ||A||_1 ||inv(A)||_1 eps) for xSYTRI / xHETRI; nothing outside the
// triangle changes.
TYPED_TEST(SysvTest, Inverse) {
  using T = TypeParam;
  Random<T> rng(19);
  for (const bool herm : Kinds<T>()) {
    for (const Matrix kind : {Matrix::kShifted, Matrix::kSaddle}) {
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {0, 1, 2, 9, 33}) {
          SCOPED_TRACE(::testing::Message()
                       << (herm ? "he " : "sy ") << uplo << " n=" << n
                       << " kind=" << static_cast<int>(kind));
          const hwyblas_int lda = n + 2;
          const std::vector<T> a = Indefinite<T>(herm, kind, uplo, n, lda, rng);
          std::vector<T> inv = a;
          std::vector<hwyblas_int> ipiv;
          ASSERT_EQ(Factor(herm, uplo, n, inv, lda, ipiv), 0);
          ASSERT_EQ(Sytri(herm, uplo, n, inv, lda, ipiv), 0);
          ExpectOutsideUnchanged(uplo, inv, a, n, lda);
          if (n == 0) continue;
          const Dense<T> da = FullOf(herm, uplo, a, n, lda);
          const Dense<T> di = FullOf(herm, uplo, inv, n, lda);
          const double resid = Norm1(Sub(Identity<T>(n), Mul(da, di)));
          EXPECT_LT(resid / n / Norm1(da) / Norm1(di) / LapackEps<T>(),
                    kLapackThreshold);
          if (herm) {
            for (hwyblas_int i = 0; i < n; ++i) {
              EXPECT_EQ(std::imag(At(inv, i, i, lda)), 0);
            }
          }
        }
      }
    }
  }
}

// xSYCONV moves the off-diagonal elements of the 2-by-2 blocks of D to E
// and applies the interchanges; reverting restores A exactly.
TYPED_TEST(SysvTest, Convert) {
  using T = TypeParam;
  Random<T> rng(23);
  for (const char uplo : {'U', 'L'}) {
    const hwyblas_int n = 15, lda = n + 1;
    std::vector<T> f = Indefinite<T>(false, Matrix::kSaddle, uplo, n, lda, rng);
    std::vector<hwyblas_int> ipiv;
    ASSERT_EQ(Factor(false, uplo, n, f, lda, ipiv), 0);
    std::vector<T> c = f, e(n, T(9));
    ASSERT_EQ(Syconv(uplo, 'C', n, c, lda, ipiv, e), 0);
    // E has the off-diagonal elements of the 2-by-2 blocks of D (at the
    // second (U) or first (L) index of the block), which are zero in the
    // converted A; the diagonal is unchanged.
    hwyblas_int nonzero = 0;
    for (hwyblas_int k = 0; k < n; ++k) {
      EXPECT_EQ(At(c, k, k, lda), At(f, k, k, lda));
      if (e[k] != T(0)) {
        ++nonzero;
        const hwyblas_int i = uplo == 'U' ? k - 1 : k;
        ASSERT_GE(i, 0);
        EXPECT_EQ(e[k],
                  uplo == 'U' ? At(f, i, i + 1, lda) : At(f, i + 1, i, lda));
        EXPECT_EQ(uplo == 'U' ? At(c, i, i + 1, lda) : At(c, i + 1, i, lda),
                  T(0));
      }
    }
    EXPECT_GT(nonzero, 0);
    ExpectOutsideUnchanged(uplo, c, f, n, lda);
    ASSERT_EQ(Syconv(uplo, 'R', n, c, lda, ipiv, e), 0);
    EXPECT_EQ(c, f);
  }
}

// The estimate of xSYCON / xHECON against the exact reciprocal condition
// number (1-norm), within the ratio of LAPACK's xGET06.
TYPED_TEST(SysvTest, ConditionNumber) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(29);
  for (const bool herm : Kinds<T>()) {
    for (const Matrix kind : {Matrix::kShifted, Matrix::kSaddle}) {
      for (const char uplo : {'U', 'L'}) {
        for (const hwyblas_int n : {1, 2, 7, 25, 50}) {
          SCOPED_TRACE(::testing::Message()
                       << (herm ? "he " : "sy ") << uplo << " n=" << n
                       << " kind=" << static_cast<int>(kind));
          const hwyblas_int lda = n + 1;
          const std::vector<T> a = Indefinite<T>(herm, kind, uplo, n, lda, rng);
          const Dense<T> da = FullOf(herm, uplo, a, n, lda);
          std::vector<T> f = a;
          std::vector<hwyblas_int> ipiv;
          ASSERT_EQ(Factor(herm, uplo, n, f, lda, ipiv), 0);
          R rcond = -1;
          ASSERT_EQ(Sycon(herm, uplo, n, f, lda, ipiv,
                          static_cast<R>(Norm1(da)), &rcond),
                    0);
          const double rcondc = 1 / (Norm1(da) * Norm1(Inverse(da)));
          ASSERT_GT(rcond, 0);
          EXPECT_LT(std::max(rcond / rcondc, rcondc / rcond), kLapackThreshold)
              << rcond << " " << rcondc;
        }
      }
    }
  }
}

// n = 0 gives RCOND = 1; ANORM = 0 and a zero 1-by-1 block of D give
// RCOND = 0 (INFO = 0).
TYPED_TEST(SysvTest, ConditionNumberSpecialCases) {
  using T = TypeParam;
  using R = RealT<T>;
  for (const bool herm : Kinds<T>()) {
    std::vector<T> a(9, T(0));
    At(a, 0, 0, 3) = T(1);
    At(a, 2, 2, 3) = T(2);
    const std::vector<hwyblas_int> ipiv = {1, 2, 3};
    R rcond = -1;
    EXPECT_EQ(Sycon(herm, 'U', 0, a, 1, ipiv, R(1), &rcond), 0);
    EXPECT_EQ(rcond, 1);
    rcond = -1;
    EXPECT_EQ(Sycon(herm, 'U', 3, a, 3, ipiv, R(0), &rcond), 0);
    EXPECT_EQ(rcond, 0);
    for (const char uplo : {'U', 'L'}) {
      rcond = -1;
      EXPECT_EQ(Sycon(herm, uplo, 3, a, 3, ipiv, R(2), &rcond), 0);
      EXPECT_EQ(rcond, 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Iterative refinement and expert driver

// Solutions perturbed by a known relative error: xSYRFS / xHERFS refine them
// and bound the remaining error; near underflow the backward error is that
// of the formula with SAFE1.
TYPED_TEST(SysvTest, Refinement) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(31);
  const double delta = std::is_same_v<R, float> ? 1e-2 : 1e-6;
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {1, 6, 40}) {
        for (const double bscale : {1.0, NearUnderflow<T>()}) {
          SCOPED_TRACE(::testing::Message() << (herm ? "he " : "sy ") << uplo
                                            << " n=" << n << " " << bscale);
          const hwyblas_int nrhs = 2, lda = n + 1, ldb = n + 2, ldx = n + 3;
          const std::vector<T> a =
              Indefinite<T>(herm, Matrix::kShifted, uplo, n, lda, rng);
          std::vector<T> af = a;
          std::vector<hwyblas_int> ipiv;
          ASSERT_EQ(Factor(herm, uplo, n, af, lda, ipiv), 0);
          std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
          for (T& e : b) e *= static_cast<R>(bscale);
          const Dense<T> da = FullOf(herm, uplo, a, n, lda);
          const Dense<T> xt = Mul(Inverse(da), ToDense(b, n, nrhs, ldb));
          std::vector<T> x(static_cast<size_t>(ldx) * nrhs);
          for (hwyblas_int j = 0; j < nrhs; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              At(x, i, j, ldx) = static_cast<T>(
                  xt(i, j) * (1 + delta * std::real(rng.Next())));
            }
          }
          std::vector<R> ferr(nrhs, -1), berr(nrhs, -1);
          ASSERT_EQ(Syrfs(herm, uplo, n, nrhs, a, lda, af, lda, ipiv, b, ldb, x,
                          ldx, ferr, berr),
                    0);
          for (hwyblas_int j = 0; j < nrhs; ++j) {
            double err = 0, xnorm = 0;
            for (hwyblas_int i = 0; i < n; ++i) {
              err = std::max(
                  err,
                  Cabs1Ref(static_cast<RefT<T>>(At(x, i, j, ldx)) - xt(i, j)));
              xnorm = std::max(xnorm, Cabs1Ref(At(x, i, j, ldx)));
            }
            EXPECT_LT(err / xnorm, 100 * n * LapackEps<T>());
            EXPECT_GE(ferr[j], 0.3 * err / xnorm);
            if (bscale == 1) {
              EXPECT_LT(berr[j], 4 * (n + 1) * LapackEps<T>());
              EXPECT_LT(ferr[j], 1e3 * n * LapackEps<T>());
            } else {
              // The diagonal as the routines see it: CABS1(A(k, k)) for
              // xSYRFS, abs(Re(A(k, k))) for xHERFS (the same here).
              std::vector<double> resid(n), denom(n);
              for (hwyblas_int i = 0; i < n; ++i) {
                RefT<T> r = -static_cast<RefT<T>>(At(b, i, j, ldb));
                denom[i] = Cabs1Ref(At(b, i, j, ldb));
                for (hwyblas_int k = 0; k < n; ++k) {
                  r += da(i, k) * static_cast<RefT<T>>(At(x, k, j, ldx));
                  denom[i] += Cabs1Ref(da(i, k)) * Cabs1Ref(At(x, k, j, ldx));
                }
                resid[i] = Cabs1Ref(r);
              }
              const double berr_c = BackwardError<T>(resid, denom);
              EXPECT_NEAR(berr[j], berr_c, 0.01 * berr_c);
            }
          }
        }
      }
    }
  }
}

// xSYSVX / xHESVX with FACT = 'N' and then 'F' with that factorization
// (the same solution); RCOND is that of xSYCON / xHECON; the workspace
// query (LWORK = -1) returns max(minimal size, N * NB) for FACT = 'N', the
// minimal size for 'F'.
TYPED_TEST(SysvTest, ExpertDriver) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(37);
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int n : {0, 1, 5, 30}) {
        SCOPED_TRACE(::testing::Message()
                     << (herm ? "he " : "sy ") << uplo << " n=" << n);
        SysvxCall<T> c;
        c.herm = herm;
        c.uplo = uplo;
        c.n = n;
        c.nrhs = 2;
        c.lda = n + 1;
        c.ldaf = n + 2;
        c.ldb = n + 3;
        c.ldx = n + 4;
        c.a = Indefinite<T>(herm, Matrix::kSaddle, uplo, n, c.lda, rng);
        c.b = rng.Vector(static_cast<size_t>(c.ldb) * c.nrhs);
        const std::vector<T> a0 = c.a, b0 = c.b;
        SysvxCall<T> q = c;
        q.lwork = -1;
        ASSERT_EQ(q.Run(), 0);
        EXPECT_EQ(std::real(q.work[0]),
                  RealT<T>(std::max<hwyblas_int>(c.MinWork(), 64 * n)));
        q.fact = 'F';
        ASSERT_EQ(q.Run(), 0);
        EXPECT_EQ(std::real(q.work[0]), RealT<T>(c.MinWork()));
        ASSERT_EQ(c.Run(), 0);
        EXPECT_EQ(c.a, a0);
        EXPECT_EQ(c.b, b0);
        if (n == 0) {
          EXPECT_EQ(c.rcond, 1);
          continue;
        }
        const Dense<T> da = FullOf(herm, uplo, a0, n, c.lda);
        EXPECT_LT(SolveRatio(da, c.x, c.ldx, b0, c.ldb, n, c.nrhs),
                  kLapackThreshold);
        R rcond = -1;
        ASSERT_EQ(Sycon(herm, uplo, n, c.af, c.ldaf, c.ipiv,
                        static_cast<R>(NormInf(da)), &rcond),
                  0);
        EXPECT_NEAR(c.rcond, rcond, 1e-3 * rcond);
        for (hwyblas_int j = 0; j < c.nrhs; ++j) {
          EXPECT_GE(c.ferr[j], 0);
          EXPECT_LT(c.berr[j], 4 * (n + 1) * LapackEps<T>());
        }
        SysvxCall<T> f = c;
        f.fact = 'F';
        f.x.assign(f.x.size(), T(0));
        ASSERT_EQ(f.Run(), 0);
        EXPECT_EQ(f.x, c.x);
        EXPECT_EQ(f.rcond, c.rcond);
        EXPECT_EQ(f.af, c.af);
        // The optimal workspace gives the same results.
        SysvxCall<T> o = c;
        o.lwork = std::max<hwyblas_int>(c.MinWork(), 64 * n);
        ASSERT_EQ(o.Run(), 0);
        EXPECT_EQ(o.x, c.x);
      }
    }
  }
}

// A zero 1-by-1 block of D: INFO = k, RCOND = 0, no solution. A matrix
// singular to working precision: INFO = n + 1 with a solution.
TYPED_TEST(SysvTest, ExpertDriverSingular) {
  using T = TypeParam;
  Random<T> rng(41);
  const hwyblas_int n = 6;
  for (const bool herm : Kinds<T>()) {
    SysvxCall<T> c;
    c.herm = herm;
    c.n = n;
    c.nrhs = 1;
    c.lda = c.ldaf = c.ldb = c.ldx = n;
    c.a.assign(n * n, T(0));
    for (hwyblas_int i = 0; i < n; ++i) At(c.a, i, i, n) = T(RealT<T>(i % 2 ? 2 : -2));
    At(c.a, 3, 3, n) = T(0);
    c.b = rng.Vector(n);
    EXPECT_EQ(c.Run(), 4);
    EXPECT_EQ(c.rcond, 0);
    EXPECT_EQ(c.ferr[0], -1);  // not computed

    SysvxCall<T> d = c;
    d.uplo = 'L';
    At(d.a, 3, 3, n) = T(static_cast<RealT<T>>(Eps<T>() * Eps<T>()));
    EXPECT_EQ(d.Run(), n + 1);
    EXPECT_LT(d.rcond, LapackEps<T>());
    EXPECT_GE(d.ferr[0], 0);
  }
}

// ---------------------------------------------------------------------------
// Complex symmetric matrix-vector product and rank-1 update

// CSYMV / ZSYMV: y := alpha * A * x + beta * y and CSYR / ZSYR: A := alpha *
// x * x**T + A for complex symmetric A, against the reference, with
// increments of both signs; the quick returns.
TYPED_TEST(SysvTest, ComplexSymmetricBlas) {
  using T = TypeParam;
  if constexpr (IsComplex<T>::value) {
    Random<T> rng(43);
    const hwyblas_int n = 9, lda = n + 2;
    for (const char uplo : {'U', 'L'}) {
      for (const hwyblas_int incx : {1, 2, -1, -3}) {
        for (const hwyblas_int incy : {1, -2}) {
          for (const T beta : {T(0), T(1), MakeScalar<T>(0.5, -1)}) {
            SCOPED_TRACE(::testing::Message()
                         << uplo << " incx=" << incx << " incy=" << incy << " "
                         << beta);
            const std::vector<T> a =
                Indefinite<T>(false, Matrix::kShifted, uplo, n, lda, rng);
            const Dense<T> da = SymmetricOf(uplo, a, n, lda);
            const size_t ax = std::abs(incx), ay = std::abs(incy);
            const std::vector<T> x = rng.Vector(1 + (n - 1) * ax);
            std::vector<T> y = rng.Vector(1 + (n - 1) * ay);
            const T alpha = MakeScalar<T>(-0.75, 0.25);
            // Element i of a vector with increment inc.
            const auto elem = [&](hwyblas_int i, hwyblas_int inc) {
              return static_cast<size_t>(inc > 0 ? i * inc
                                                 : (n - 1 - i) * -inc);
            };
            // The elements between those of y stay as they are.
            std::vector<RefT<T>> want(y.begin(), y.end());
            std::vector<double> bound(y.size(), 0);
            for (hwyblas_int i = 0; i < n; ++i) {
              RefT<T> s = 0;
              double sb = 0;
              for (hwyblas_int k = 0; k < n; ++k) {
                const RefT<T> xk = static_cast<RefT<T>>(x[elem(k, incx)]);
                s += da(i, k) * xk;
                sb += std::abs(da(i, k)) * std::abs(xk);
              }
              const RefT<T> yi = static_cast<RefT<T>>(y[elem(i, incy)]);
              const RefT<T> al = static_cast<RefT<T>>(alpha);
              const RefT<T> be = static_cast<RefT<T>>(beta);
              want[elem(i, incy)] =
                  al * s + (beta == T(0) ? RefT<T>(0) : be * yi);
              bound[elem(i, incy)] =
                  std::abs(al) * sb + std::abs(be) * std::abs(yi) + 1e-300;
            }
            HWYBLAS_CALL_COMPLEX(T, symv, &uplo, &n, P(&alpha), P(a.data()),
                                 &lda, P(x.data()), &incx, P(&beta),
                                 P(y.data()), &incy);
            EXPECT_TRUE(NearWithBound(y, want, bound, 4 * n * Eps<T>(),
                                      y.size(), 1, y.size()));
            // CSYR / ZSYR with the same x.
            std::vector<T> u = a;
            HWYBLAS_CALL_COMPLEX(T, syr, &uplo, &n, P(&alpha), P(x.data()),
                                 &incx, P(u.data()), &lda);
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = 0; i < n; ++i) {
                if (!InTriangle(uplo, i, j)) continue;
                const RefT<T> w =
                    da(i, j) + static_cast<RefT<T>>(alpha) *
                                   static_cast<RefT<T>>(x[elem(i, incx)]) *
                                   static_cast<RefT<T>>(x[elem(j, incx)]);
                EXPECT_LT(std::abs(static_cast<RefT<T>>(At(u, i, j, lda)) - w),
                          8 * Eps<T>() * (std::abs(da(i, j)) + 1))
                    << i << ", " << j;
              }
            }
            ExpectOutsideUnchanged(uplo, u, a, n, lda);
          }
        }
      }
    }
    // Quick returns: alpha = 0 and beta = 1 leave y alone (even with NaN in
    // A), alpha = 0 leaves A alone; alpha = 0 with beta = 0 zeroes y.
    std::vector<T> a(9, NaN<T>()), x(3, T(1)), y(3, T(2));
    const T zero = T(0), one = T(1);
    const hwyblas_int three = 3, inc = 1;
    const char uplo = 'U';
    HWYBLAS_CALL_COMPLEX(T, symv, &uplo, &three, P(&zero), P(a.data()), &three,
                         P(x.data()), &inc, P(&one), P(y.data()), &inc);
    EXPECT_EQ(y, std::vector<T>(3, T(2)));
    HWYBLAS_CALL_COMPLEX(T, symv, &uplo, &three, P(&zero), P(a.data()), &three,
                         P(x.data()), &inc, P(&zero), P(y.data()), &inc);
    EXPECT_EQ(y, std::vector<T>(3, T(0)));
    std::vector<T> b(9, T(3));
    HWYBLAS_CALL_COMPLEX(T, syr, &uplo, &three, P(&zero), P(x.data()), &inc,
                         P(b.data()), &three);
    EXPECT_EQ(b, std::vector<T>(9, T(3)));
  }
}

// ---------------------------------------------------------------------------
// Errors

TYPED_TEST(SysvTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16, T(1)), work(16);
  std::vector<hwyblas_int> ipiv = {1, 2, 3, 4};
  std::vector<R> ferr(4), berr(4);
  R rcond;
  ErrorCapture errors;
  for (const bool herm : Kinds<T>()) {
    for (const bool unblocked : {false, true}) {
      const std::string trf = SyName<T>(herm, unblocked ? "TF2" : "TRF");
      std::vector<hwyblas_int> p;
      EXPECT_EQ(Sytrf(herm, unblocked, 'X', 4, a, 4, p, work, 16), -1);
      EXPECT_EQ(Sytrf(herm, unblocked, 'U', -1, a, 4, p, work, 16), -2);
      EXPECT_EQ(Sytrf(herm, unblocked, 'U', 4, a, 3, p, work, 16), -4);
      std::vector<std::string> want = {trf + ":1", trf + ":2", trf + ":4"};
      if (!unblocked) {
        EXPECT_EQ(Sytrf(herm, false, 'U', 4, a, 4, p, work, 0), -7);
        want.push_back(trf + ":7");
      }
      EXPECT_EQ(errors.Take(), want);
    }
    for (const bool level3 : {false, true}) {
      const std::string trs = SyName<T>(herm, level3 ? "TRS2" : "TRS");
      EXPECT_EQ(Sytrs(herm, level3, 'X', 4, 1, a, 4, ipiv, b, 4), -1);
      EXPECT_EQ(Sytrs(herm, level3, 'U', -1, 1, a, 4, ipiv, b, 4), -2);
      EXPECT_EQ(Sytrs(herm, level3, 'U', 4, -1, a, 4, ipiv, b, 4), -3);
      EXPECT_EQ(Sytrs(herm, level3, 'U', 4, 1, a, 3, ipiv, b, 4), -5);
      EXPECT_EQ(Sytrs(herm, level3, 'U', 4, 1, a, 4, ipiv, b, 3), -8);
      EXPECT_EQ(errors.Take(),
                (std::vector<std::string>{trs + ":1", trs + ":2", trs + ":3",
                                          trs + ":5", trs + ":8"}));
    }
    const std::string tri = SyName<T>(herm, "TRI");
    EXPECT_EQ(Sytri(herm, 'X', 4, a, 4, ipiv), -1);
    EXPECT_EQ(Sytri(herm, 'U', -1, a, 4, ipiv), -2);
    EXPECT_EQ(Sytri(herm, 'U', 4, a, 3, ipiv), -4);
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{tri + ":1", tri + ":2", tri + ":4"}));
    const std::string con = SyName<T>(herm, "CON");
    EXPECT_EQ(Sycon(herm, 'X', 4, a, 4, ipiv, R(1), &rcond), -1);
    EXPECT_EQ(Sycon(herm, 'U', -1, a, 4, ipiv, R(1), &rcond), -2);
    EXPECT_EQ(Sycon(herm, 'U', 4, a, 3, ipiv, R(1), &rcond), -4);
    EXPECT_EQ(Sycon(herm, 'U', 4, a, 4, ipiv, R(-1), &rcond), -6);
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{con + ":1", con + ":2", con + ":4",
                                        con + ":6"}));
    const std::string rfs = SyName<T>(herm, "RFS");
    EXPECT_EQ(Syrfs(herm, 'X', 4, 1, a, 4, a, 4, ipiv, b, 4, b, 4, ferr, berr),
              -1);
    EXPECT_EQ(Syrfs(herm, 'U', -1, 1, a, 4, a, 4, ipiv, b, 4, b, 4, ferr, berr),
              -2);
    EXPECT_EQ(Syrfs(herm, 'U', 4, -1, a, 4, a, 4, ipiv, b, 4, b, 4, ferr, berr),
              -3);
    EXPECT_EQ(Syrfs(herm, 'U', 4, 1, a, 3, a, 4, ipiv, b, 4, b, 4, ferr, berr),
              -5);
    EXPECT_EQ(Syrfs(herm, 'U', 4, 1, a, 4, a, 3, ipiv, b, 4, b, 4, ferr, berr),
              -7);
    EXPECT_EQ(Syrfs(herm, 'U', 4, 1, a, 4, a, 4, ipiv, b, 3, b, 4, ferr, berr),
              -10);
    EXPECT_EQ(Syrfs(herm, 'U', 4, 1, a, 4, a, 4, ipiv, b, 4, b, 3, ferr, berr),
              -12);
    EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                                 rfs + ":1", rfs + ":2", rfs + ":3", rfs + ":5",
                                 rfs + ":7", rfs + ":10", rfs + ":12"}));
    const std::string sv = SyName<T>(herm, "SV");
    std::vector<hwyblas_int> p;
    EXPECT_EQ(Sysv(herm, 'X', 4, 1, a, 4, p, b, 4, work, 16), -1);
    EXPECT_EQ(Sysv(herm, 'U', -1, 1, a, 4, p, b, 4, work, 16), -2);
    EXPECT_EQ(Sysv(herm, 'U', 4, -1, a, 4, p, b, 4, work, 16), -3);
    EXPECT_EQ(Sysv(herm, 'U', 4, 1, a, 3, p, b, 4, work, 16), -5);
    EXPECT_EQ(Sysv(herm, 'U', 4, 1, a, 4, p, b, 3, work, 16), -8);
    EXPECT_EQ(Sysv(herm, 'U', 4, 1, a, 4, p, b, 4, work, 0), -10);
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{sv + ":1", sv + ":2", sv + ":3",
                                        sv + ":5", sv + ":8", sv + ":10"}));
    const auto sysvx = [&](char fact, char uplo, hwyblas_int n,
                           hwyblas_int nrhs, hwyblas_int lda, hwyblas_int ldaf,
                           hwyblas_int ldb, hwyblas_int ldx,
                           hwyblas_int lwork) {
      SysvxCall<T> call;
      call.herm = herm;
      call.fact = fact;
      call.uplo = uplo;
      call.n = n;
      call.nrhs = nrhs;
      call.lda = lda;
      call.ldaf = ldaf;
      call.ldb = ldb;
      call.ldx = ldx;
      call.lwork = lwork;
      call.a.assign(16, T(1));
      call.b.assign(16, T(1));
      call.af.assign(16, T(1));
      return call.Run();
    };
    EXPECT_EQ(sysvx('X', 'U', 4, 1, 4, 4, 4, 4, 0), -1);
    EXPECT_EQ(sysvx('N', 'X', 4, 1, 4, 4, 4, 4, 0), -2);
    EXPECT_EQ(sysvx('N', 'U', -1, 1, 4, 4, 4, 4, 0), -3);
    EXPECT_EQ(sysvx('N', 'U', 4, -1, 4, 4, 4, 4, 0), -4);
    EXPECT_EQ(sysvx('N', 'U', 4, 1, 3, 4, 4, 4, 0), -6);
    EXPECT_EQ(sysvx('N', 'U', 4, 1, 4, 3, 4, 4, 0), -8);
    EXPECT_EQ(sysvx('N', 'U', 4, 1, 4, 4, 3, 4, 0), -11);
    EXPECT_EQ(sysvx('N', 'U', 4, 1, 4, 4, 4, 3, 0), -13);
    const hwyblas_int too_small = (IsComplex<T>::value ? 2 : 3) * 4 - 1;
    EXPECT_EQ(sysvx('N', 'U', 4, 1, 4, 4, 4, 4, too_small), -18);
    const std::string svx = SyName<T>(herm, "SVX");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{
                  svx + ":1", svx + ":2", svx + ":3", svx + ":4", svx + ":6",
                  svx + ":8", svx + ":11", svx + ":13", svx + ":18"}));
  }
  std::vector<T> e(4);
  EXPECT_EQ(Syconv('X', 'C', 4, a, 4, ipiv, e), -1);
  EXPECT_EQ(Syconv('U', 'X', 4, a, 4, ipiv, e), -2);
  EXPECT_EQ(Syconv('U', 'C', -1, a, 4, ipiv, e), -3);
  EXPECT_EQ(Syconv('U', 'R', 4, a, 3, ipiv, e), -5);
  const std::string conv = Name<T>("SYCONV");
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{conv + ":1", conv + ":2", conv + ":3",
                                      conv + ":5"}));
  if constexpr (IsComplex<T>::value) {
    const T one = T(1);
    const char u = 'U', x = 'X';
    const hwyblas_int four = 4, minus = -1, three = 3, zero = 0, inc = 1;
    HWYBLAS_CALL_COMPLEX(T, symv, &x, &four, P(&one), P(a.data()), &four,
                         P(b.data()), &inc, P(&one), P(e.data()), &inc);
    HWYBLAS_CALL_COMPLEX(T, symv, &u, &minus, P(&one), P(a.data()), &four,
                         P(b.data()), &inc, P(&one), P(e.data()), &inc);
    HWYBLAS_CALL_COMPLEX(T, symv, &u, &four, P(&one), P(a.data()), &three,
                         P(b.data()), &inc, P(&one), P(e.data()), &inc);
    HWYBLAS_CALL_COMPLEX(T, symv, &u, &four, P(&one), P(a.data()), &four,
                         P(b.data()), &zero, P(&one), P(e.data()), &inc);
    HWYBLAS_CALL_COMPLEX(T, symv, &u, &four, P(&one), P(a.data()), &four,
                         P(b.data()), &inc, P(&one), P(e.data()), &zero);
    const std::string symv = Name<T>("SYMV");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{symv + ":1", symv + ":2", symv + ":5",
                                        symv + ":7", symv + ":10"}));
    HWYBLAS_CALL_COMPLEX(T, syr, &x, &four, P(&one), P(b.data()), &inc,
                         P(a.data()), &four);
    HWYBLAS_CALL_COMPLEX(T, syr, &u, &minus, P(&one), P(b.data()), &inc,
                         P(a.data()), &four);
    HWYBLAS_CALL_COMPLEX(T, syr, &u, &four, P(&one), P(b.data()), &zero,
                         P(a.data()), &four);
    HWYBLAS_CALL_COMPLEX(T, syr, &u, &four, P(&one), P(b.data()), &inc,
                         P(a.data()), &three);
    const std::string syr = Name<T>("SYR");
    EXPECT_EQ(errors.Take(),
              (std::vector<std::string>{syr + ":1", syr + ":2", syr + ":5",
                                        syr + ":7"}));
  }
}

// ---------------------------------------------------------------------------
// LAPACKE

// The LAPACKE function `name` of the kind (LAPACKE_?sy* or LAPACKE_?he*).
#define LAPACKE_SY_OR_HE(herm, T, sy, he, ...)           \
  [&]() -> lapack_int {                                  \
    if constexpr (IsComplex<T>::value) {                 \
      if (herm) return LAPACKE_CALL(T, he, __VA_ARGS__); \
    }                                                    \
    return LAPACKE_CALL(T, sy, __VA_ARGS__);             \
  }()

// The row-major functions give the results of the column-major ones (the
// same arithmetic on transposed copies of the triangle uplo).
TYPED_TEST(SysvTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(47);
  const lapack_int n = 9, nrhs = 3, lda = 11, ldb = 10;
  const auto same = [&](const std::vector<T>& row, const std::vector<T>& col,
                        lapack_int rows, lapack_int cols, lapack_int ldr,
                        lapack_int ldc) {
    for (lapack_int i = 0; i < rows; ++i) {
      for (lapack_int j = 0; j < cols; ++j) {
        ASSERT_EQ(row[i * ldr + j], At(col, i, j, ldc)) << i << ", " << j;
      }
    }
  };
  for (const bool herm : Kinds<T>()) {
    for (const char uplo : {'U', 'L'}) {
      SCOPED_TRACE(::testing::Message() << (herm ? "he " : "sy ") << uplo);
      const std::vector<T> a =
          Indefinite<T>(herm, Matrix::kSaddle, uplo, n, lda, rng);
      const std::vector<T> ar = RowMajor(a, n, n, lda, lda);
      const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * n);
      const std::vector<T> br = RowMajor(b, n, nrhs, ldb, ldb);
      // xSYSV, xSYTRF, xSYTRS, xSYTRS2.
      std::vector<T> fc = a, fr = ar, xc = b, xr = br;
      std::vector<lapack_int> pc(n), pr(n);
      ASSERT_EQ(
          LAPACKE_SY_OR_HE(herm, T, sysv, hesv, LAPACK_COL_MAJOR, uplo, n, nrhs,
                           fc.data(), lda, pc.data(), xc.data(), ldb),
          0);
      ASSERT_EQ(
          LAPACKE_SY_OR_HE(herm, T, sysv, hesv, LAPACK_ROW_MAJOR, uplo, n, nrhs,
                           fr.data(), lda, pr.data(), xr.data(), ldb),
          0);
      EXPECT_EQ(pr, pc);
      same(fr, fc, n, n, lda, lda);
      same(xr, xc, n, nrhs, ldb, ldb);
      std::vector<T> gc = a, gr = ar;
      ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytrf, hetrf, LAPACK_COL_MAJOR, uplo,
                                 n, gc.data(), lda, pc.data()),
                0);
      ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytrf, hetrf, LAPACK_ROW_MAJOR, uplo,
                                 n, gr.data(), lda, pr.data()),
                0);
      same(gr, gc, n, n, lda, lda);
      for (const bool level3 : {false, true}) {
        std::vector<T> yc = b, yr = br;
        if (level3) {
          ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytrs2, hetrs2, LAPACK_COL_MAJOR,
                                     uplo, n, nrhs, gc.data(), lda, pc.data(),
                                     yc.data(), ldb),
                    0);
          ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytrs2, hetrs2, LAPACK_ROW_MAJOR,
                                     uplo, n, nrhs, gr.data(), lda, pr.data(),
                                     yr.data(), ldb),
                    0);
        } else {
          ASSERT_EQ(
              LAPACKE_SY_OR_HE(herm, T, sytrs, hetrs, LAPACK_COL_MAJOR, uplo, n,
                               nrhs, gc.data(), lda, pc.data(), yc.data(), ldb),
              0);
          ASSERT_EQ(
              LAPACKE_SY_OR_HE(herm, T, sytrs, hetrs, LAPACK_ROW_MAJOR, uplo, n,
                               nrhs, gr.data(), lda, pr.data(), yr.data(), ldb),
              0);
        }
        same(yr, yc, n, nrhs, ldb, ldb);
        same(gr, gc, n, n, lda, lda);  // restored
        // xSYRFS.
        std::vector<R> f1(nrhs), e1(nrhs), f2(nrhs), e2(nrhs);
        ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, syrfs, herfs, LAPACK_COL_MAJOR,
                                   uplo, n, nrhs, a.data(), lda, gc.data(), lda,
                                   pc.data(), b.data(), ldb, yc.data(), ldb,
                                   f1.data(), e1.data()),
                  0);
        ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, syrfs, herfs, LAPACK_ROW_MAJOR,
                                   uplo, n, nrhs, ar.data(), lda, gr.data(),
                                   lda, pr.data(), br.data(), ldb, yr.data(),
                                   ldb, f2.data(), e2.data()),
                  0);
        same(yr, yc, n, nrhs, ldb, ldb);
        EXPECT_EQ(f2, f1);
        EXPECT_EQ(e2, e1);
      }
      // xSYCON.
      R rc = -1, rr = -2;
      ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sycon, hecon, LAPACK_COL_MAJOR, uplo,
                                 n, gc.data(), lda, pc.data(), R(3), &rc),
                0);
      ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sycon, hecon, LAPACK_ROW_MAJOR, uplo,
                                 n, gr.data(), lda, pr.data(), R(3), &rr),
                0);
      EXPECT_EQ(rr, rc);
      // xSYCONV (symmetric).
      if (!herm) {
        std::vector<T> cc = gc, cr = gr, ec(n), er(n);
        for (const char way : {'C', 'R'}) {
          ASSERT_EQ(LAPACKE_CALL(T, syconv, LAPACK_COL_MAJOR, uplo, way, n,
                                 cc.data(), lda, pc.data(), ec.data()),
                    0);
          ASSERT_EQ(LAPACKE_CALL(T, syconv, LAPACK_ROW_MAJOR, uplo, way, n,
                                 cr.data(), lda, pr.data(), er.data()),
                    0);
          same(cr, cc, n, n, lda, lda);
          EXPECT_EQ(er, ec);
        }
      }
      // xSYTRI.
      ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytri, hetri, LAPACK_COL_MAJOR, uplo,
                                 n, gc.data(), lda, pc.data()),
                0);
      ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytri, hetri, LAPACK_ROW_MAJOR, uplo,
                                 n, gr.data(), lda, pr.data()),
                0);
      same(gr, gc, n, n, lda, lda);
      // xSYSVX: AF (FACT = 'N') and X are those of the column-major call.
      for (const char fact : {'N', 'F'}) {
        SCOPED_TRACE(fact);
        std::vector<T> afc(lda * n), afr(lda * n);
        std::vector<lapack_int> p1(n), p2(n);
        if (fact == 'F') {
          afc = a;
          afr = ar;
          ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytrf, hetrf, LAPACK_COL_MAJOR,
                                     uplo, n, afc.data(), lda, p1.data()),
                    0);
          ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sytrf, hetrf, LAPACK_ROW_MAJOR,
                                     uplo, n, afr.data(), lda, p2.data()),
                    0);
        }
        std::vector<T> x1(ldb * nrhs), x2(ldb * n);
        std::vector<R> g1(nrhs), h1(nrhs), g2(nrhs), h2(nrhs);
        R rc1, rc2;
        ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sysvx, hesvx, LAPACK_COL_MAJOR,
                                   fact, uplo, n, nrhs, a.data(), lda,
                                   afc.data(), lda, p1.data(), b.data(), ldb,
                                   x1.data(), ldb, &rc1, g1.data(), h1.data()),
                  0);
        ASSERT_EQ(LAPACKE_SY_OR_HE(herm, T, sysvx, hesvx, LAPACK_ROW_MAJOR,
                                   fact, uplo, n, nrhs, ar.data(), lda,
                                   afr.data(), lda, p2.data(), br.data(), ldb,
                                   x2.data(), ldb, &rc2, g2.data(), h2.data()),
                  0);
        EXPECT_EQ(p2, p1);
        for (lapack_int i = 0; i < n; ++i) {
          for (lapack_int j = 0; j < n; ++j) {
            if (InTriangle(uplo, i, j)) {
              ASSERT_EQ(afr[i * lda + j], At(afc, i, j, lda)) << i << ", " << j;
            }
          }
        }
        same(x2, x1, n, nrhs, ldb, ldb);
        EXPECT_EQ(rc2, rc1);
        EXPECT_EQ(g2, g1);
        EXPECT_EQ(h2, h1);
      }
    }
  }
}

TYPED_TEST(SysvTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), b(16, T(1)), x(16), e(4);
  std::vector<lapack_int> ipiv = {1, 2, 3, 4};
  std::vector<R> ferr(4), berr(4);
  R rcond = 0;
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LAPACKE_CALL(T, sysv, 0, 'U', 4, 1, a.data(), 4, ipiv.data(),
                         b.data(), 4),
            -1);
  EXPECT_EQ(LAPACKE_CALL(T, sysv, LAPACK_ROW_MAJOR, 'U', 4, 2, a.data(), 4,
                         ipiv.data(), b.data(), 1),
            -9);
  EXPECT_EQ(LAPACKE_CALL(T, sytrf, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3,
                         ipiv.data()),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, sytrs, LAPACK_ROW_MAJOR, 'U', 4, 1, a.data(), 3,
                         ipiv.data(), b.data(), 1),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, sytrs2, LAPACK_ROW_MAJOR, 'U', 4, 2, a.data(), 4,
                         ipiv.data(), b.data(), 1),
            -9);
  EXPECT_EQ(LAPACKE_CALL(T, syconv, LAPACK_ROW_MAJOR, 'U', 'C', 4, a.data(), 3,
                         ipiv.data(), e.data()),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, sytri, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3,
                         ipiv.data()),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, sycon, LAPACK_ROW_MAJOR, 'U', 4, a.data(), 3,
                         ipiv.data(), R(1), &rcond),
            -5);
  EXPECT_EQ(LAPACKE_CALL(T, syrfs, LAPACK_ROW_MAJOR, 'U', 4, 2, a.data(), 4,
                         a.data(), 4, ipiv.data(), b.data(), 4, x.data(), 1,
                         ferr.data(), berr.data()),
            -13);
  EXPECT_EQ(LAPACKE_CALL(T, sysvx, LAPACK_ROW_MAJOR, 'N', 'U', 4, 2, a.data(),
                         4, a.data(), 4, ipiv.data(), b.data(), 1, x.data(), 4,
                         &rcond, ferr.data(), berr.data()),
            -12);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("sysv") +
                "\nWrong parameter 9 in " + LapackeName<T>("sysv_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("sytrf_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("sytrs_work") +
                "\nWrong parameter 9 in " + LapackeName<T>("sytrs2_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("syconv_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("sytri_work") +
                "\nWrong parameter 5 in " + LapackeName<T>("sycon_work") +
                "\nWrong parameter 13 in " + LapackeName<T>("syrfs_work") +
                "\nWrong parameter 12 in " + LapackeName<T>("sysvx_work") +
                "\n");
  if constexpr (IsComplex<T>::value) {
    ::testing::internal::CaptureStdout();
    EXPECT_EQ(LAPACKE_CALL(T, hesv, LAPACK_ROW_MAJOR, 'U', 4, 1, a.data(), 3,
                           ipiv.data(), b.data(), 1),
              -6);
    EXPECT_EQ(LAPACKE_CALL(T, hesvx, LAPACK_ROW_MAJOR, 'N', 'U', 4, 2, a.data(),
                           4, a.data(), 4, ipiv.data(), b.data(), 4, x.data(),
                           1, &rcond, ferr.data(), berr.data()),
              -14);
    EXPECT_EQ(::testing::internal::GetCapturedStdout(),
              "Wrong parameter 6 in " + LapackeName<T>("hesv_work") +
                  "\nWrong parameter 14 in " + LapackeName<T>("hesvx_work") +
                  "\n");
  }
  // NaN checks: only the triangle uplo of A (and AF) is checked.
  a[1] = NaN<T>();  // A(2, 1): in the lower triangle
  EXPECT_EQ(LAPACKE_CALL(T, sytrf, LAPACK_COL_MAJOR, 'L', 4, a.data(), 4,
                         ipiv.data()),
            -4);
  b[0] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, sytrs, LAPACK_COL_MAJOR, 'U', 4, 1, a.data(), 4,
                         ipiv.data(), b.data(), 4),
            -8);
  EXPECT_EQ(
      LAPACKE_CALL(T, sycon, LAPACK_COL_MAJOR, 'U', 4, a.data(), 4, ipiv.data(),
                   std::numeric_limits<R>::quiet_NaN(), &rcond),
      -7);
  EXPECT_EQ(LAPACKE_CALL(T, sysvx, LAPACK_COL_MAJOR, 'N', 'U', 4, 1, a.data(),
                         4, a.data(), 4, ipiv.data(), b.data(), 4, x.data(), 4,
                         &rcond, ferr.data(), berr.data()),
            -11);
  EXPECT_EQ(LAPACKE_CALL(T, sysvx, LAPACK_COL_MAJOR, 'F', 'L', 4, 1, b.data(),
                         4, a.data(), 4, ipiv.data(), b.data(), 4, x.data(), 4,
                         &rcond, ferr.data(), berr.data()),
            -6);
  std::vector<T> au = b;  // b[0] = A(1, 1) is NaN
  au[0] = T(1);
  EXPECT_EQ(LAPACKE_CALL(T, sysvx, LAPACK_COL_MAJOR, 'F', 'L', 4, 1, au.data(),
                         4, a.data(), 4, ipiv.data(), au.data(), 4, x.data(), 4,
                         &rcond, ferr.data(), berr.data()),
            -8);
}

// In row-major layout, LAPACKE_?syconv transposes the n x n matrix A, and
// nothing beyond its n rows (the reference transposes lda rows).
TYPED_TEST(SysvTest, LapackeSyconvRowMajor) {
  using T = TypeParam;
  Random<T> rng(53);
  const lapack_int n = 6, lda = 9;
  for (const char uplo : {'U', 'L'}) {
    std::vector<T> f = Indefinite<T>(false, Matrix::kSaddle, uplo, n, n, rng);
    std::vector<lapack_int> ipiv(n);
    ASSERT_EQ(Factor(false, uplo, n, f, n, ipiv), 0);
    // Exactly n rows of lda elements.
    std::vector<T> r(static_cast<size_t>(n) * lda, T(5));
    for (lapack_int i = 0; i < n; ++i) {
      for (lapack_int j = 0; j < n; ++j) r[i * lda + j] = At(f, i, j, n);
    }
    std::vector<T> e1(n), e2(n), c = f;
    ASSERT_EQ(LAPACKE_CALL(T, syconv, LAPACK_ROW_MAJOR, uplo, 'C', n, r.data(),
                           lda, ipiv.data(), e1.data()),
              0);
    ASSERT_EQ(Syconv(uplo, 'C', n, c, n, ipiv, e2), 0);
    EXPECT_EQ(e1, e2);
    for (lapack_int i = 0; i < n; ++i) {
      for (lapack_int j = 0; j < lda; ++j) {
        ASSERT_EQ(r[i * lda + j], j < n ? At(c, i, j, n) : T(5))
            << i << ", " << j;
      }
    }
  }
}

}  // namespace
}  // namespace hwyblas_test
