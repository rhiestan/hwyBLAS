// Tests of the band and tridiagonal routines: xGBTF2, xGBTRF, xGBTRS,
// xGBSV, xGBCON, xLATBS, xPOTF2, xPBTF2, xPBTRF, xPBTRS, xPBSV, xGTTRF,
// xGTTS2, xGTTRS, xGTSV, xPTTRF, xPTTS2, xPTTRS, xPTSV and their LAPACKE
// functions.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <tuple>
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
class BandTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(BandTest, ScalarTypes);

// ---------------------------------------------------------------------------
// Calls

template <typename T>
hwyblas_int Gbtrf(bool unblocked, hwyblas_int m, hwyblas_int n, hwyblas_int kl,
                  hwyblas_int ku, std::vector<T>& ab, hwyblas_int ldab,
                  std::vector<hwyblas_int>& ipiv) {
  hwyblas_int info = -99;
  if (unblocked) {
    HWYBLAS_CALL(T, gbtf2, &m, &n, &kl, &ku, P(ab.data()), &ldab, ipiv.data(),
                 &info);
  } else {
    HWYBLAS_CALL(T, gbtrf, &m, &n, &kl, &ku, P(ab.data()), &ldab, ipiv.data(),
                 &info);
  }
  return info;
}

template <typename T>
hwyblas_int Gbtrs(char trans, hwyblas_int n, hwyblas_int kl, hwyblas_int ku,
                  hwyblas_int nrhs, const std::vector<T>& ab, hwyblas_int ldab,
                  const std::vector<hwyblas_int>& ipiv, std::vector<T>& b,
                  hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gbtrs, &trans, &n, &kl, &ku, &nrhs, P(ab.data()), &ldab,
               ipiv.data(), P(b.data()), &ldb, &info);
  return info;
}

template <typename T>
hwyblas_int Gbsv(hwyblas_int n, hwyblas_int kl, hwyblas_int ku,
                 hwyblas_int nrhs, std::vector<T>& ab, hwyblas_int ldab,
                 std::vector<hwyblas_int>& ipiv, std::vector<T>& b,
                 hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gbsv, &n, &kl, &ku, &nrhs, P(ab.data()), &ldab, ipiv.data(),
               P(b.data()), &ldb, &info);
  return info;
}

// xGBCON with workspaces of the minimal sizes.
template <typename T>
hwyblas_int Gbcon(char norm, hwyblas_int n, hwyblas_int kl, hwyblas_int ku,
                  const std::vector<T>& ab, hwyblas_int ldab,
                  const std::vector<hwyblas_int>& ipiv, RealT<T> anorm,
                  RealT<T>* rcond) {
  hwyblas_int info = -99;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  if constexpr (IsComplex<T>::value) {
    std::vector<T> work(2 * n1);
    std::vector<RealT<T>> rwork(n1);
    HWYBLAS_CALL_COMPLEX(T, gbcon, &norm, &n, &kl, &ku, P(ab.data()), &ldab,
                         ipiv.data(), &anorm, rcond, P(work.data()),
                         rwork.data(), &info);
  } else {
    std::vector<T> work(3 * n1);
    std::vector<hwyblas_int> iwork(n1);
    HWYBLAS_CALL_REAL(T, gbcon, &norm, &n, &kl, &ku, ab.data(), &ldab,
                      ipiv.data(), &anorm, rcond, work.data(), iwork.data(),
                      &info);
  }
  return info;
}

template <typename T>
hwyblas_int Latbs(char uplo, char trans, char diag, char normin, hwyblas_int n,
                  hwyblas_int kd, const std::vector<T>& ab, hwyblas_int ldab,
                  std::vector<T>& x, RealT<T>* scale,
                  std::vector<RealT<T>>& cnorm) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, latbs, &uplo, &trans, &diag, &normin, &n, &kd, P(ab.data()),
               &ldab, P(x.data()), scale, cnorm.data(), &info);
  return info;
}

template <typename T>
hwyblas_int Potf2(char uplo, hwyblas_int n, std::vector<T>& a,
                  hwyblas_int lda) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, potf2, &uplo, &n, P(a.data()), &lda, &info);
  return info;
}

template <typename T>
hwyblas_int Pbtrf(bool unblocked, char uplo, hwyblas_int n, hwyblas_int kd,
                  std::vector<T>& ab, hwyblas_int ldab) {
  hwyblas_int info = -99;
  if (unblocked) {
    HWYBLAS_CALL(T, pbtf2, &uplo, &n, &kd, P(ab.data()), &ldab, &info);
  } else {
    HWYBLAS_CALL(T, pbtrf, &uplo, &n, &kd, P(ab.data()), &ldab, &info);
  }
  return info;
}

template <typename T>
hwyblas_int Pbtrs(char uplo, hwyblas_int n, hwyblas_int kd, hwyblas_int nrhs,
                  const std::vector<T>& ab, hwyblas_int ldab,
                  std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, pbtrs, &uplo, &n, &kd, &nrhs, P(ab.data()), &ldab,
               P(b.data()), &ldb, &info);
  return info;
}

template <typename T>
hwyblas_int Pbsv(char uplo, hwyblas_int n, hwyblas_int kd, hwyblas_int nrhs,
                 std::vector<T>& ab, hwyblas_int ldab, std::vector<T>& b,
                 hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, pbsv, &uplo, &n, &kd, &nrhs, P(ab.data()), &ldab,
               P(b.data()), &ldb, &info);
  return info;
}

// A tridiagonal matrix (DL, D, DU) and its factorization (DU2, IPIV).
template <typename T>
struct Tridiagonal {
  std::vector<T> dl, d, du, du2;
  std::vector<hwyblas_int> ipiv;
};

template <typename T>
hwyblas_int Gttrf(hwyblas_int n, Tridiagonal<T>& t) {
  t.du2.assign(std::max<hwyblas_int>(1, n - 2), NaN<T>());
  t.ipiv.assign(std::max<hwyblas_int>(1, n), -1);
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gttrf, &n, P(t.dl.data()), P(t.d.data()), P(t.du.data()),
               P(t.du2.data()), t.ipiv.data(), &info);
  return info;
}

template <typename T>
hwyblas_int Gttrs(char trans, hwyblas_int n, hwyblas_int nrhs,
                  const Tridiagonal<T>& t, std::vector<T>& b,
                  hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gttrs, &trans, &n, &nrhs, P(t.dl.data()), P(t.d.data()),
               P(t.du.data()), P(t.du2.data()), t.ipiv.data(), P(b.data()),
               &ldb, &info);
  return info;
}

template <typename T>
void Gtts2(hwyblas_int itrans, hwyblas_int n, hwyblas_int nrhs,
           const Tridiagonal<T>& t, std::vector<T>& b, hwyblas_int ldb) {
  HWYBLAS_CALL(T, gtts2, &itrans, &n, &nrhs, P(t.dl.data()), P(t.d.data()),
               P(t.du.data()), P(t.du2.data()), t.ipiv.data(), P(b.data()),
               &ldb);
}

template <typename T>
hwyblas_int Gtsv(hwyblas_int n, hwyblas_int nrhs, Tridiagonal<T>& t,
                 std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, gtsv, &n, &nrhs, P(t.dl.data()), P(t.d.data()),
               P(t.du.data()), P(b.data()), &ldb, &info);
  return info;
}

template <typename T>
hwyblas_int Pttrf(hwyblas_int n, std::vector<RealT<T>>& d, std::vector<T>& e) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, pttrf, &n, d.data(), P(e.data()), &info);
  return info;
}

// xPTTRS (uplo is used for complex T only).
template <typename T>
hwyblas_int Pttrs(char uplo, hwyblas_int n, hwyblas_int nrhs,
                  const std::vector<RealT<T>>& d, const std::vector<T>& e,
                  std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, pttrs, &uplo, &n, &nrhs, d.data(), P(e.data()),
                         P(b.data()), &ldb, &info);
  } else {
    (void)uplo;
    HWYBLAS_CALL_REAL(T, pttrs, &n, &nrhs, d.data(), e.data(), b.data(), &ldb,
                      &info);
  }
  return info;
}

// xPTTS2 (iuplo is used for complex T only).
template <typename T>
void Ptts2(hwyblas_int iuplo, hwyblas_int n, hwyblas_int nrhs,
           const std::vector<RealT<T>>& d, const std::vector<T>& e,
           std::vector<T>& b, hwyblas_int ldb) {
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, ptts2, &iuplo, &n, &nrhs, d.data(), P(e.data()),
                         P(b.data()), &ldb);
  } else {
    (void)iuplo;
    HWYBLAS_CALL_REAL(T, ptts2, &n, &nrhs, d.data(), e.data(), b.data(), &ldb);
  }
}

template <typename T>
hwyblas_int Ptsv(hwyblas_int n, hwyblas_int nrhs, std::vector<RealT<T>>& d,
                 std::vector<T>& e, std::vector<T>& b, hwyblas_int ldb) {
  hwyblas_int info = -99;
  HWYBLAS_CALL(T, ptsv, &n, &nrhs, d.data(), P(e.data()), P(b.data()), &ldb,
               &info);
  return info;
}

// Whether a and b are equal, with NaN equal to NaN (the elements outside
// the band).
template <typename T>
bool SameValues(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const bool nan_a = std::isnan(std::real(a[i])) || std::isnan(std::imag(a[i]));
    const bool nan_b = std::isnan(std::real(b[i])) || std::isnan(std::imag(b[i]));
    if (nan_a != nan_b || (!nan_a && a[i] != b[i])) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Band matrices

// A random m x n band matrix with kl subdiagonals and ku superdiagonals in
// band storage with kv superdiagonals (kv = ku, or kl + ku for xGBTRF),
// leading dimension ldab: A(i, j) (0-based) in ab[kv+i-j + j*ldab]. The
// elements of AB outside the band (the corners, the rows for fill-in and
// the rows beyond kl+kv+1) are NaN: the routines must not read them, or
// must overwrite them first.
template <typename T>
std::vector<T> RandomBand(Random<T>& rng, hwyblas_int m, hwyblas_int n,
                          hwyblas_int kl, hwyblas_int ku, hwyblas_int kv,
                          hwyblas_int ldab) {
  std::vector<T> ab(static_cast<size_t>(ldab) * std::max<hwyblas_int>(1, n),
                    NaN<T>());
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = std::max<hwyblas_int>(0, j - ku);
         i < std::min(m, j + kl + 1); ++i) {
      At(ab, kv + i - j, j, ldab) = rng.Next();
    }
  }
  return ab;
}

// The dense m x n matrix of the band matrix ab (as RandomBand stores it).
template <typename T>
Dense<T> DenseOfBand(const std::vector<T>& ab, hwyblas_int m, hwyblas_int n,
                     hwyblas_int kl, hwyblas_int ku, hwyblas_int kv,
                     hwyblas_int ldab) {
  Dense<T> d(m, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = std::max<hwyblas_int>(0, j - ku);
         i < std::min(m, j + kl + 1); ++i) {
      d(i, j) = static_cast<RefT<T>>(At(ab, kv + i - j, j, ldab));
    }
  }
  return d;
}

// The product P(1) * L(1) * ... * P(k) * L(k) * U of the factorization of
// xGBTRF (U in the first kl+ku+1 rows of ab, the multipliers below): the
// reconstruction of LAPACK's xGBT01.
template <typename T>
Dense<T> GbProduct(const std::vector<T>& ab, hwyblas_int m, hwyblas_int n,
                   hwyblas_int kl, hwyblas_int ku, hwyblas_int ldab,
                   const std::vector<hwyblas_int>& ipiv) {
  const hwyblas_int kv = kl + ku;
  Dense<T> x(m, n);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = std::max<hwyblas_int>(0, j - kv);
         i <= std::min(j, m - 1); ++i) {
      x(i, j) = static_cast<RefT<T>>(At(ab, kv + i - j, j, ldab));
    }
  }
  for (hwyblas_int j = std::min(m, n) - 1; j >= 0; --j) {
    const hwyblas_int km = std::min(kl, m - 1 - j);
    for (hwyblas_int c = 0; c < n; ++c) {
      for (hwyblas_int i = j + 1; i <= j + km; ++i) {
        x(i, c) += static_cast<RefT<T>>(At(ab, kv + i - j, j, ldab)) * x(j, c);
      }
    }
    const hwyblas_int p = ipiv[j] - 1;
    if (p != j) {
      for (hwyblas_int c = 0; c < n; ++c) std::swap(x(j, c), x(p, c));
    }
  }
  return x;
}

// The test ratio of LAPACK's xGET02 for op(A) * X = B.
template <typename T>
double SolveRatio(const Dense<T>& opa, const std::vector<T>& x,
                  hwyblas_int ldx, const std::vector<T>& b, hwyblas_int ldb,
                  hwyblas_int n, hwyblas_int nrhs) {
  const Dense<T> dx = ToDense(x, n, nrhs, ldx);
  const double resid = Norm1(Sub(Mul(opa, dx), ToDense(b, n, nrhs, ldb)));
  return resid / std::max<hwyblas_int>(1, n) / std::max(Norm1(opa), 1e-300) /
         std::max(Norm1(dx), 1e-300) / LapackEps<T>();
}

// (m, n, kl, ku) of the general band matrices.
const std::vector<std::tuple<hwyblas_int, hwyblas_int, hwyblas_int,
                             hwyblas_int>>&
GbShapes() {
  static const auto* shapes = new std::vector<
      std::tuple<hwyblas_int, hwyblas_int, hwyblas_int, hwyblas_int>>{
      {0, 0, 1, 1},   {0, 3, 1, 1},   {3, 0, 1, 1},    {1, 1, 0, 0},
      {1, 1, 2, 3},   {4, 4, 0, 0},   {5, 5, 1, 0},    {5, 5, 0, 1},
      {6, 6, 1, 1},   {8, 8, 3, 2},   {12, 12, 2, 5},  {17, 17, 5, 3},
      {33, 33, 8, 8}, {40, 40, 9, 2}, {70, 70, 12, 20}, {6, 9, 2, 3},
      {9, 6, 3, 2},   {30, 20, 4, 5}, {20, 30, 6, 1},  {25, 25, 30, 30}};
  return *shapes;
}

TYPED_TEST(BandTest, GeneralFactorization) {
  using T = TypeParam;
  Random<T> rng(3);
  for (const hwyblas_int nb : {0, 1, 2, 3, 8}) {
    const ScopedBlockSize block_size(nb);
    for (const auto& [m, n, kl, ku] : GbShapes()) {
      for (const bool unblocked : {false, true}) {
        if (unblocked && nb > 0) continue;
        SCOPED_TRACE(::testing::Message()
                     << "m=" << m << " n=" << n << " kl=" << kl
                     << " ku=" << ku << " nb=" << nb << " unblocked="
                     << unblocked);
        const hwyblas_int ldab = 2 * kl + ku + 2;
        const std::vector<T> a = RandomBand(rng, m, n, kl, ku, kl + ku, ldab);
        std::vector<T> lu = a;
        std::vector<hwyblas_int> ipiv(std::max<hwyblas_int>(1, std::min(m, n)),
                                      -1);
        ASSERT_EQ(Gbtrf(unblocked, m, n, kl, ku, lu, ldab, ipiv), 0);
        if (m == 0 || n == 0) {
          for (size_t i = 0; i < a.size(); ++i) {
            ASSERT_TRUE(std::isnan(std::real(lu[i])));
          }
          continue;
        }
        for (hwyblas_int j = 0; j < std::min(m, n); ++j) {
          EXPECT_GE(ipiv[j], j + 1);
          EXPECT_LE(ipiv[j], std::min(m, j + kl + 1));
        }
        // The row beyond the band storage stays untouched.
        for (hwyblas_int j = 0; j < n; ++j) {
          ASSERT_TRUE(std::isnan(std::real(At(lu, ldab - 1, j, ldab))));
        }
        const Dense<T> da = DenseOfBand(a, m, n, kl, ku, kl + ku, ldab);
        const Dense<T> prod = GbProduct(lu, m, n, kl, ku, ldab, ipiv);
        EXPECT_LT(ResidualRatio(da, prod, std::max(m, n)), kLapackThreshold);
      }
    }
  }
  // An exactly zero column k: INFO = k, the factorization is completed.
  for (const bool unblocked : {false, true}) {
    const ScopedBlockSize block_size(unblocked ? 0 : 2);
    const hwyblas_int n = 10, kl = 3, ku = 2, ldab = 2 * kl + ku + 1;
    std::vector<T> ab = RandomBand(rng, n, n, kl, ku, kl + ku, ldab);
    for (hwyblas_int i = 0; i < n; ++i) {
      if (i >= 4 - ku && i <= 4 + kl) At(ab, kl + ku + i - 4, 4, ldab) = T(0);
      if (i >= 7 - ku && i <= 7 + kl) At(ab, kl + ku + i - 7, 7, ldab) = T(0);
    }
    const std::vector<T> a = ab;
    std::vector<hwyblas_int> ipiv(n);
    EXPECT_EQ(Gbtrf(unblocked, n, n, kl, ku, ab, ldab, ipiv), 5);
    const Dense<T> prod = GbProduct(ab, n, n, kl, ku, ldab, ipiv);
    EXPECT_LT(ResidualRatio(DenseOfBand(a, n, n, kl, ku, kl + ku, ldab), prod,
                            n),
              kLapackThreshold);
  }
}

// The blocked code with ILAENV's own block size (32 for KU > 64).
TYPED_TEST(BandTest, GeneralFactorizationDefaultBlockSize) {
  using T = TypeParam;
  Random<T> rng(5);
  const hwyblas_int m = 160, n = 150, kl = 40, ku = 70, ldab = 2 * kl + ku + 1;
  const std::vector<T> a = RandomBand(rng, m, n, kl, ku, kl + ku, ldab);
  std::vector<T> lu = a, lu2 = a;
  std::vector<hwyblas_int> ipiv(n), ipiv2(n);
  ASSERT_EQ(Gbtrf(false, m, n, kl, ku, lu, ldab, ipiv), 0);
  ASSERT_EQ(Gbtrf(true, m, n, kl, ku, lu2, ldab, ipiv2), 0);
  const Dense<T> da = DenseOfBand(a, m, n, kl, ku, kl + ku, ldab);
  EXPECT_LT(ResidualRatio(da, GbProduct(lu, m, n, kl, ku, ldab, ipiv), m),
            kLapackThreshold);
  // The blocked factorization is not the unblocked one (other rounding).
  EXPECT_NE(lu, lu2);
}

TYPED_TEST(BandTest, GeneralSolve) {
  using T = TypeParam;
  Random<T> rng(7);
  for (const hwyblas_int nb : {0, 2}) {
    const ScopedBlockSize block_size(nb);
    for (const auto& [m, n, kl, ku] : GbShapes()) {
      if (m != n) continue;
      for (const hwyblas_int nrhs : {0, 1, 4}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " kl=" << kl
                                          << " ku=" << ku << " nrhs=" << nrhs);
        const hwyblas_int ldab = 2 * kl + ku + 1, ldb = n + 2;
        const std::vector<T> a = RandomBand(rng, n, n, kl, ku, kl + ku, ldab);
        const std::vector<T> b =
            rng.Vector(static_cast<size_t>(ldb) * std::max<hwyblas_int>(1, nrhs));
        // xGBSV.
        std::vector<T> lu = a, x = b;
        std::vector<hwyblas_int> ipiv(std::max<hwyblas_int>(1, n));
        ASSERT_EQ(Gbsv(n, kl, ku, nrhs, lu, ldab, ipiv, x, ldb), 0);
        if (n == 0 || nrhs == 0) {
          EXPECT_EQ(x, b);
          continue;
        }
        const Dense<T> da = DenseOfBand(a, n, n, kl, ku, kl + ku, ldab);
        EXPECT_LT(SolveRatio(da, x, ldb, b, ldb, n, nrhs), kLapackThreshold);
        // xGBTRS with the factorization, for each op (and 'C' for real T).
        for (const char trans : {'N', 'T', 'C'}) {
          std::vector<T> xt = b;
          ASSERT_EQ(Gbtrs(trans, n, kl, ku, nrhs, lu, ldab, ipiv, xt, ldb), 0);
          const char op = IsComplex<T>::value ? trans : (trans == 'N' ? 'N' : 'T');
          EXPECT_LT(SolveRatio(Op(op, da), xt, ldb, b, ldb, n, nrhs),
                    kLapackThreshold)
              << trans;
          if (trans == 'N') {
            EXPECT_EQ(xt, x);
          }
        }
      }
    }
  }
  // An exactly singular matrix: INFO = k, B unchanged.
  const hwyblas_int n = 6, kl = 1, ku = 1, ldab = 4;
  std::vector<T> ab = RandomBand(rng, n, n, kl, ku, kl + ku, ldab);
  At(ab, 1, 2, ldab) = At(ab, 2, 2, ldab) = At(ab, 3, 2, ldab) = T(0);
  std::vector<T> b = rng.Vector(n), b0 = b;
  std::vector<hwyblas_int> ipiv(n);
  EXPECT_EQ(Gbsv(n, kl, ku, 1, ab, ldab, ipiv, b, n), 3);
  EXPECT_EQ(b, b0);
}

// The infinity norm or 1-norm of a dense matrix.
template <typename T>
double NormOf(char norm, const Dense<T>& a) {
  return norm == 'I' ? NormInf(a) : Norm1(a);
}

// The estimate of xGBCON against the exact reciprocal condition number (the
// ratio of LAPACK's xGET06), also for an ill-conditioned matrix whose
// solves with xLATBS scale.
TYPED_TEST(BandTest, GeneralConditionNumber) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(13);
  for (const char norm : {'1', 'O', 'I'}) {
    for (const auto& [m, n, kl, ku] : GbShapes()) {
      if (m != n || n == 0) continue;
      for (const bool graded : {false, true}) {
        SCOPED_TRACE(::testing::Message() << norm << " n=" << n << " kl=" << kl
                                          << " ku=" << ku << " " << graded);
        const hwyblas_int ldab = 2 * kl + ku + 1;
        std::vector<T> a = RandomBand(rng, n, n, kl, ku, kl + ku, ldab);
        if (graded) {
          // Columns scaled by 2**(-4j): the reciprocal condition number is
          // far below eps for larger n.
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < ldab; ++i) {
              At(a, i, j, ldab) *= static_cast<R>(std::ldexp(1.0, -4 * j));
            }
          }
          if (n * 4 > std::numeric_limits<R>::max_exponent - 10) continue;
        }
        const Dense<T> da = DenseOfBand(a, n, n, kl, ku, kl + ku, ldab);
        std::vector<T> lu = a;
        std::vector<hwyblas_int> ipiv(n);
        ASSERT_EQ(Gbtrf(false, n, n, kl, ku, lu, ldab, ipiv), 0);
        const R anorm = static_cast<R>(NormOf(norm, da));
        R rcond = -1;
        ASSERT_EQ(Gbcon(norm, n, kl, ku, lu, ldab, ipiv, anorm, &rcond), 0);
        const double rcondc = 1 / (NormOf(norm, da) * NormOf(norm, Inverse(da)));
        ASSERT_GT(rcond, 0);
        EXPECT_LT(std::max(rcond / rcondc, rcondc / rcond), kLapackThreshold)
            << rcond << " " << rcondc;
      }
    }
  }
  // The 1-norm and the infinity norm differ: I plus c = 10 in the rest of
  // the first row (KL = 0, KU = N-1) has ||inv(A)||_1 = 1 + c and
  // ||inv(A)||_inf = 1 + (n-1) c.
  {
    const hwyblas_int n = 100, kl = 0, ku = n - 1, ldab = ku + 1;
    std::vector<T> a(static_cast<size_t>(ldab) * n, T(0));
    for (hwyblas_int j = 0; j < n; ++j) {
      At(a, ku, j, ldab) = T(1);
      if (j > 0) At(a, ku - j, j, ldab) = T(10);
    }
    const Dense<T> da = DenseOfBand(a, n, n, kl, ku, ku, ldab);
    std::vector<hwyblas_int> ipiv(n);
    std::vector<T> lu = a;
    ASSERT_EQ(Gbtrf(false, n, n, kl, ku, lu, ldab, ipiv), 0);
    for (const char norm : {'1', 'I'}) {
      R rcond = -1;
      ASSERT_EQ(Gbcon(norm, n, kl, ku, lu, ldab, ipiv,
                      static_cast<R>(NormOf(norm, da)), &rcond),
                0);
      const double rcondc = 1 / (NormOf(norm, da) * NormOf(norm, Inverse(da)));
      EXPECT_LT(std::max(rcond / rcondc, rcondc / rcond), kLapackThreshold)
          << norm << " " << rcond << " " << rcondc;
    }
  }
  // N = 0 (RCOND = 1), ANORM = 0 (RCOND = 0) and a singular factor (RCOND =
  // 0).
  const hwyblas_int n = 4, kl = 1, ku = 1, ldab = 4;
  std::vector<T> lu = RandomBand(rng, n, n, kl, ku, kl + ku, ldab);
  std::vector<hwyblas_int> ipiv(n);
  ASSERT_EQ(Gbtrf(false, n, n, kl, ku, lu, ldab, ipiv), 0);
  R rcond = -1;
  EXPECT_EQ(Gbcon('1', 0, kl, ku, lu, ldab, ipiv, R(1), &rcond), 0);
  EXPECT_EQ(rcond, 1);
  EXPECT_EQ(Gbcon('1', n, kl, ku, lu, ldab, ipiv, R(0), &rcond), 0);
  EXPECT_EQ(rcond, 0);
  rcond = -1;
  At(lu, kl + ku, 2, ldab) = T(0);
  EXPECT_EQ(Gbcon('I', n, kl, ku, lu, ldab, ipiv, R(1), &rcond), 0);
  EXPECT_EQ(rcond, 0);
}

// ---------------------------------------------------------------------------
// Triangular band solves (xLATBS)

// The triangular band matrix with kd super- (uplo = 'U') or subdiagonals
// ('L') in band storage, as a dense matrix (unit diagonal for diag = 'U').
template <typename T>
Dense<T> DenseOfTriangularBand(char uplo, char diag, const std::vector<T>& ab,
                               hwyblas_int n, hwyblas_int kd,
                               hwyblas_int ldab) {
  const bool upper = uplo == 'U';
  Dense<T> d =
      DenseOfBand(ab, n, n, upper ? 0 : kd, upper ? kd : 0, upper ? kd : 0,
                  ldab);
  if (diag == 'U') {
    for (hwyblas_int i = 0; i < n; ++i) d(i, i) = 1;
  }
  return d;
}

// The test ratio of LAPACK's xTBT03 for op(A) * x = scale * b: ||op(A) * x
// - scale * b|| / (||op(A)|| * ||x|| * eps), with x and scale * b divided
// by max|x(i)| first so that nothing overflows.
template <typename T>
double LatbsRatio(const Dense<T>& opa, const std::vector<T>& x,
                  RealT<T> scale, const std::vector<T>& b, hwyblas_int n) {
  double xmax = 0;
  for (hwyblas_int i = 0; i < n; ++i) {
    xmax = std::max(xmax, std::abs(static_cast<RefT<T>>(x[i])));
  }
  if (xmax == 0) xmax = 1;
  Dense<T> dx(n, 1), sb(n, 1);
  for (hwyblas_int i = 0; i < n; ++i) {
    dx(i, 0) = static_cast<RefT<T>>(x[i]) / xmax;
    sb(i, 0) = static_cast<RefT<T>>(b[i]) * (static_cast<double>(scale) / xmax);
  }
  const double resid = Norm1(Sub(Mul(opa, dx), sb));
  return resid / std::max<hwyblas_int>(1, n) / std::max(Norm1(opa), 1e-300) /
         LapackEps<T>();
}

TYPED_TEST(BandTest, TriangularBandSolve) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(17);
  const double smlnum = static_cast<double>(std::numeric_limits<R>::min()) /
                        static_cast<double>(std::numeric_limits<R>::epsilon());
  // Matrix kinds: 0 random with a dominant diagonal (xTBSV), 1 a tiny
  // diagonal (growth), 2 huge off-diagonal elements (CNORM > BIGNUM), 3 a
  // zero diagonal element (singular: SCALE = 0), 4 graded diagonal with a
  // large right-hand side.
  for (const int kind : {0, 1, 2, 3, 4}) {
    for (const char uplo : {'U', 'L'}) {
      for (const char trans : {'N', 'T', 'C'}) {
        for (const char diag : {'N', 'U'}) {
          for (const hwyblas_int n : {0, 1, 2, 5, 13, 40}) {
            for (const hwyblas_int kd : {0, 1, 2, 5}) {
              if (kind == 3 && diag == 'U') continue;
              SCOPED_TRACE(::testing::Message()
                           << "kind=" << kind << " " << uplo << trans << diag
                           << " n=" << n << " kd=" << kd);
              const hwyblas_int ldab = kd + 2;
              const bool upper = uplo == 'U';
              std::vector<T> ab = RandomBand(rng, n, n, upper ? 0 : kd,
                                             upper ? kd : 0, upper ? kd : 0,
                                             ldab);
              const hwyblas_int dr = upper ? kd : 0;  // diagonal row
              for (hwyblas_int j = 0; j < n; ++j) {
                T& djj = At(ab, dr, j, ldab);
                if (kind == 0) djj += T(static_cast<R>(kd + 2));
                if (kind == 1) djj *= static_cast<R>(smlnum * 64);
                if (kind == 2) {
                  for (hwyblas_int i = 0; i < kd + 1; ++i) {
                    if (i != dr && !std::isnan(std::real(At(ab, i, j, ldab)))) {
                      At(ab, i, j, ldab) *= static_cast<R>(
                          std::numeric_limits<R>::max() / (16 * (kd + 1)));
                    }
                  }
                }
                if (kind == 4) {
                  djj = T(static_cast<R>(std::ldexp(1.0, -2 * (j % 30))));
                }
              }
              if (kind == 3 && n > 0) At(ab, dr, n / 2, ldab) = T(0);
              std::vector<T> b = rng.Vector(std::max<hwyblas_int>(1, n));
              if (kind == 4) {
                for (T& v : b) v *= static_cast<R>(1 / (smlnum * 1e3));
              }
              for (const char normin : {'N', 'Y'}) {
                std::vector<R> cnorm(std::max<hwyblas_int>(1, n));
                if (normin == 'Y') {
                  std::vector<T> x0 = b;
                  R s0 = -1;
                  ASSERT_EQ(Latbs(uplo, trans, diag, 'N', n, kd, ab, ldab, x0,
                                  &s0, cnorm),
                            0);
                }
                std::vector<T> x = b;
                R scale = -1;
                ASSERT_EQ(Latbs(uplo, trans, diag, normin, n, kd, ab, ldab, x,
                                &scale, cnorm),
                          0);
                if (n == 0) {
                  EXPECT_EQ(scale, 1);
                  continue;
                }
                EXPECT_GE(scale, 0);
                EXPECT_LE(scale, 1);
                if (kind == 3) {
                  EXPECT_EQ(scale, 0);
                }
                // The column norms of the off-diagonal parts (kind 2: as
                // TSCAL scaled them, which stays within 1 ulp).
                for (hwyblas_int j = 0; j < n; ++j) {
                  double s = 0;
                  for (hwyblas_int i = 0; i < kd + 1; ++i) {
                    const hwyblas_int row = upper ? j - kd + i : j + i;
                    if (i == dr || row < 0 || row >= n) continue;
                    s += Cabs1Ref(At(ab, i, j, ldab));
                  }
                  EXPECT_NEAR(cnorm[j], s, 4 * n * Eps<T>() * s) << j;
                }
                const char op =
                    IsComplex<T>::value ? trans : (trans == 'N' ? 'N' : 'T');
                const Dense<T> opa =
                    Op(op, DenseOfTriangularBand(uplo, diag, ab, n, kd, ldab));
                EXPECT_LT(LatbsRatio(opa, x, scale, b, n), kLapackThreshold)
                    << normin << " scale=" << scale;
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) positive definite band matrices

// A random Hermitian positive definite band matrix of order n with kd
// super- (uplo = 'U') or subdiagonals ('L') in band storage, NaN outside the
// band (the corners and the rows beyond kd+1).
template <typename T>
std::vector<T> PositiveDefiniteBand(Random<T>& rng, char uplo, hwyblas_int n,
                                    hwyblas_int kd, hwyblas_int ldab) {
  const bool upper = uplo == 'U';
  std::vector<T> ab = RandomBand(rng, n, n, upper ? 0 : kd, upper ? kd : 0,
                                 upper ? kd : 0, ldab);
  const hwyblas_int dr = upper ? kd : 0;
  for (hwyblas_int j = 0; j < n; ++j) {
    At(ab, dr, j, ldab) =
        T(static_cast<RealT<T>>(2 * kd + 1) + std::real(At(ab, dr, j, ldab)));
  }
  return ab;
}

// The Hermitian matrix of the band storage ab (triangle uplo).
template <typename T>
Dense<T> HermitianOfBand(char uplo, const std::vector<T>& ab, hwyblas_int n,
                         hwyblas_int kd, hwyblas_int ldab) {
  Dense<T> d = DenseOfTriangularBand(uplo, 'N', ab, n, kd, ldab);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      if (uplo == 'U' ? i > j : i < j) d(i, j) = Conj(d(j, i));
    }
  }
  return d;
}

// U**H * U or L * L**H of the Cholesky factor in the band storage ab.
template <typename T>
Dense<T> CholeskyProduct(char uplo, const std::vector<T>& ab, hwyblas_int n,
                         hwyblas_int kd, hwyblas_int ldab) {
  const Dense<T> f = DenseOfTriangularBand(uplo, 'N', ab, n, kd, ldab);
  return uplo == 'U' ? Mul(Adjoint(f), f) : Mul(f, Adjoint(f));
}

TYPED_TEST(BandTest, PositiveDefiniteBand) {
  using T = TypeParam;
  Random<T> rng(19);
  for (const hwyblas_int nb : {0, 1, 2, 3, 8}) {
    const ScopedBlockSize block_size(nb);
    for (const char uplo : {'U', 'L'}) {
      for (const bool unblocked : {false, true}) {
        if (unblocked && nb > 0) continue;
        for (const auto& [n, kd] :
             std::vector<std::pair<hwyblas_int, hwyblas_int>>{
                 {0, 1}, {1, 0}, {1, 3}, {4, 0}, {5, 1}, {9, 3}, {16, 4},
                 {17, 8}, {40, 7}, {50, 16}, {30, 40}}) {
          SCOPED_TRACE(::testing::Message() << uplo << " n=" << n << " kd="
                                            << kd << " nb=" << nb
                                            << " unblocked=" << unblocked);
          const hwyblas_int ldab = kd + 2;
          const std::vector<T> a = PositiveDefiniteBand(rng, uplo, n, kd, ldab);
          std::vector<T> f = a;
          ASSERT_EQ(Pbtrf(unblocked, uplo, n, kd, f, ldab), 0);
          if (n == 0) continue;
          const Dense<T> da = HermitianOfBand(uplo, a, n, kd, ldab);
          EXPECT_LT(ResidualRatio(da, CholeskyProduct(uplo, f, n, kd, ldab), n),
                    kLapackThreshold);
          for (hwyblas_int j = 0; j < n; ++j) {
            ASSERT_TRUE(std::isnan(std::real(At(f, ldab - 1, j, ldab))));
            EXPECT_EQ(std::imag(At(f, uplo == 'U' ? kd : 0, j, ldab)), 0);
          }
          // xPBTRS and xPBSV.
          for (const hwyblas_int nrhs : {0, 1, 3}) {
            const hwyblas_int ldb = n + 1;
            const std::vector<T> b = rng.Vector(
                static_cast<size_t>(ldb) * std::max<hwyblas_int>(1, nrhs));
            std::vector<T> x = b, x2 = b, f2 = a;
            ASSERT_EQ(Pbtrs(uplo, n, kd, nrhs, f, ldab, x, ldb), 0);
            ASSERT_EQ(Pbsv(uplo, n, kd, nrhs, f2, ldab, x2, ldb), 0);
            if (nrhs == 0) {
              EXPECT_EQ(x, b);
              continue;
            }
            EXPECT_LT(SolveRatio(da, x, ldb, b, ldb, n, nrhs),
                      kLapackThreshold);
            if (!unblocked) {
              EXPECT_TRUE(SameValues(f2, f));
              EXPECT_EQ(x2, x);
            }
          }
        }
      }
    }
  }
  // Not positive definite at k: INFO = k (the diagonal element is replaced
  // by the real value that failed), also inside a block.
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int nb : {0, 3}) {
      const ScopedBlockSize block_size(nb);
      for (const bool unblocked : {false, true}) {
        const hwyblas_int n = 12, kd = 4, ldab = kd + 1;
        std::vector<T> ab = PositiveDefiniteBand(rng, uplo, n, kd, ldab);
        At(ab, uplo == 'U' ? kd : 0, 7, ldab) = T(-50);
        std::vector<T> b = rng.Vector(n), b0 = b;
        EXPECT_EQ(Pbtrf(unblocked, uplo, n, kd, ab, ldab), 8)
            << uplo << nb << unblocked;
        std::vector<T> ab2 = PositiveDefiniteBand(rng, uplo, n, kd, ldab);
        At(ab2, uplo == 'U' ? kd : 0, 2, ldab) = T(0);
        EXPECT_EQ(Pbsv(uplo, n, kd, 1, ab2, ldab, b, n), 3);
        EXPECT_EQ(b, b0);
      }
    }
  }
}

// xPOTF2 against the residual of the factorization and xPOTRF.
TYPED_TEST(BandTest, UnblockedCholesky) {
  using T = TypeParam;
  Random<T> rng(23);
  for (const char uplo : {'U', 'L'}) {
    for (const hwyblas_int n : {0, 1, 2, 7, 30}) {
      SCOPED_TRACE(::testing::Message() << uplo << " n=" << n);
      const hwyblas_int lda = n + 3;
      const std::vector<T> a = PositiveDefinite<T>(uplo, n, lda, rng);
      std::vector<T> f = a;
      ASSERT_EQ(Potf2(uplo, n, f, lda), 0);
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < n; ++i) {
          if (!InTriangle(uplo, i, j)) {
            ASSERT_EQ(At(f, i, j, lda), Sentinel<T>());
          }
        }
      }
      if (n == 0) continue;
      Dense<T> fd(n, n);
      for (hwyblas_int j = 0; j < n; ++j) {
        for (hwyblas_int i = 0; i < n; ++i) {
          if (InTriangle(uplo, i, j)) fd(i, j) = At(f, i, j, lda);
        }
      }
      const Dense<T> prod =
          uplo == 'U' ? Mul(Adjoint(fd), fd) : Mul(fd, Adjoint(fd));
      EXPECT_LT(ResidualRatio(HermitianOf(uplo, a, n, lda), prod, n),
                kLapackThreshold);
    }
  }
  // Not positive definite (or NaN) at k: INFO = k, A(k, k) the value.
  for (const char uplo : {'U', 'L'}) {
    const hwyblas_int n = 6;
    std::vector<T> a = PositiveDefinite<T>(uplo, n, n, rng);
    At(a, 3, 3, n) = T(-1000);
    EXPECT_EQ(Potf2(uplo, n, a, n), 4);
    EXPECT_LT(std::real(At(a, 3, 3, n)), 0);
    std::vector<T> b = PositiveDefinite<T>(uplo, n, n, rng);
    At(b, 2, 2, n) = NaN<T>();
    EXPECT_EQ(Potf2(uplo, n, b, n), 3);
  }
}

// ---------------------------------------------------------------------------
// Tridiagonal matrices

template <typename T>
Tridiagonal<T> RandomTridiagonal(Random<T>& rng, hwyblas_int n) {
  Tridiagonal<T> t;
  const size_t n1 = std::max<hwyblas_int>(1, n);
  t.dl = rng.Vector(n1);
  t.d = rng.Vector(n1);
  t.du = rng.Vector(n1);
  return t;
}

template <typename T>
Dense<T> DenseOfTridiagonal(const Tridiagonal<T>& t, hwyblas_int n) {
  Dense<T> d(n, n);
  for (hwyblas_int i = 0; i < n; ++i) {
    d(i, i) = static_cast<RefT<T>>(t.d[i]);
    if (i + 1 < n) {
      d(i + 1, i) = static_cast<RefT<T>>(t.dl[i]);
      d(i, i + 1) = static_cast<RefT<T>>(t.du[i]);
    }
  }
  return d;
}

// L * U of xGTTRF (LAPACK's xGTT01): U with the diagonals D, DU and DU2,
// L = P(1) * L(1) * ... with the multipliers DL and the interchanges IPIV.
template <typename T>
Dense<T> GtProduct(const Tridiagonal<T>& f, hwyblas_int n) {
  Dense<T> x(n, n);
  for (hwyblas_int i = 0; i < n; ++i) {
    x(i, i) = static_cast<RefT<T>>(f.d[i]);
    if (i + 1 < n) x(i, i + 1) = static_cast<RefT<T>>(f.du[i]);
    if (i + 2 < n) x(i, i + 2) = static_cast<RefT<T>>(f.du2[i]);
  }
  for (hwyblas_int j = n - 2; j >= 0; --j) {
    for (hwyblas_int c = 0; c < n; ++c) {
      x(j + 1, c) += static_cast<RefT<T>>(f.dl[j]) * x(j, c);
    }
    if (f.ipiv[j] != j + 1) {
      EXPECT_EQ(f.ipiv[j], j + 2);
      for (hwyblas_int c = 0; c < n; ++c) std::swap(x(j, c), x(j + 1, c));
    }
  }
  return x;
}

TYPED_TEST(BandTest, Tridiagonal) {
  using T = TypeParam;
  Random<T> rng(29);
  for (const hwyblas_int n : {0, 1, 2, 3, 4, 7, 20, 64}) {
    for (const bool small_diagonal : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " " << small_diagonal);
      Tridiagonal<T> t = RandomTridiagonal(rng, n);
      if (small_diagonal) {
        // Mostly row interchanges; one zero subdiagonal element.
        for (T& v : t.d) v *= RealT<T>(0.01);
        if (n > 3) t.dl[1] = T(0);
      }
      const Dense<T> da = DenseOfTridiagonal(t, n);
      Tridiagonal<T> f = t;
      ASSERT_EQ(Gttrf(n, f), 0);
      if (n > 0) {
        EXPECT_LT(ResidualRatio(da, GtProduct(f, n), n), kLapackThreshold);
      }
      for (const hwyblas_int nb : {0, 2, 8}) {
        const ScopedBlockSize block_size(nb);
        for (const hwyblas_int nrhs : {0, 1, 2, 5}) {
          const hwyblas_int ldb = n + 1;
          const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) *
                                              std::max<hwyblas_int>(1, nrhs));
          for (const char trans : {'N', 'T', 'C', 'n', 't', 'c'}) {
            SCOPED_TRACE(::testing::Message()
                         << "nb=" << nb << " nrhs=" << nrhs << " " << trans);
            std::vector<T> x = b;
            ASSERT_EQ(Gttrs(trans, n, nrhs, f, x, ldb), 0);
            if (n == 0 || nrhs == 0) {
              EXPECT_EQ(x, b);
              continue;
            }
            const char upper = static_cast<char>(std::toupper(trans));
            const char op =
                IsComplex<T>::value ? upper : (upper == 'N' ? 'N' : 'T');
            EXPECT_LT(SolveRatio(Op(op, da), x, ldb, b, ldb, n, nrhs),
                      kLapackThreshold);
            // xGTTS2 directly.
            std::vector<T> x2 = b;
            const hwyblas_int itrans = op == 'N' ? 0 : (op == 'T' ? 1 : 2);
            Gtts2(itrans, n, nrhs, f, x2, ldb);
            EXPECT_EQ(x2, x);
          }
          // xGTSV.
          Tridiagonal<T> g = t;
          std::vector<T> x = b;
          ASSERT_EQ(Gtsv(n, nrhs, g, x, ldb), 0);
          if (n > 0 && nrhs > 0) {
            EXPECT_LT(SolveRatio(da, x, ldb, b, ldb, n, nrhs),
                      kLapackThreshold);
          }
        }
      }
    }
  }
  // Exactly singular: INFO = k for U(k, k) = 0. xGTTRF completes the
  // factorization; xGTSV stops (B is partly eliminated).
  for (const hwyblas_int k : {1, 3, 6}) {
    const hwyblas_int n = 6;
    Tridiagonal<T> t = RandomTridiagonal(rng, n);
    // Column k-1 zero in rows k-1 and k (and row k-2 above): with D(k-1) =
    // DL(k-1) = 0, the elimination of column k-1 finds a zero pivot.
    t.d[k - 1] = T(0);
    if (k < n) t.dl[k - 1] = T(0);
    if (k > 1) t.du[k - 2] = T(0);
    Tridiagonal<T> f = t;
    EXPECT_EQ(Gttrf(n, f), k);
    std::vector<T> b = rng.Vector(n);
    EXPECT_EQ(Gtsv(n, 1, t, b, n), k);
  }
}

template <typename T>
Dense<T> DenseOfPositiveTridiagonal(const std::vector<RealT<T>>& d,
                                    const std::vector<T>& e, hwyblas_int n) {
  Dense<T> a(n, n);
  for (hwyblas_int i = 0; i < n; ++i) {
    a(i, i) = d[i];
    if (i + 1 < n) {
      a(i + 1, i) = static_cast<RefT<T>>(e[i]);
      a(i, i + 1) = Conj(static_cast<RefT<T>>(e[i]));
    }
  }
  return a;
}

TYPED_TEST(BandTest, PositiveDefiniteTridiagonal) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(31);
  Random<R> rrng(37);
  for (const hwyblas_int n : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30}) {
    SCOPED_TRACE(::testing::Message() << "n=" << n);
    const size_t n1 = std::max<hwyblas_int>(1, n);
    std::vector<R> d = rrng.Vector(n1);
    for (R& v : d) v += R(4);
    const std::vector<T> e = rng.Vector(n1);
    const Dense<T> da = DenseOfPositiveTridiagonal(d, e, n);
    std::vector<R> df = d;
    std::vector<T> ef = e;
    ASSERT_EQ(Pttrf(n, df, ef), 0);
    // L * D * L**H.
    if (n > 0) {
      Dense<T> l = Identity<T>(n), dd(n, n);
      for (hwyblas_int i = 0; i < n; ++i) {
        dd(i, i) = df[i];
        if (i + 1 < n) l(i + 1, i) = static_cast<RefT<T>>(ef[i]);
      }
      EXPECT_LT(ResidualRatio(da, Mul(Mul(l, dd), Adjoint(l)), n),
                kLapackThreshold);
    }
    for (const hwyblas_int nb : {0, 2}) {
      const ScopedBlockSize block_size(nb);
      for (const hwyblas_int nrhs : {0, 1, 2, 3, 5}) {
        const hwyblas_int ldb = n + 2;
        const std::vector<T> b =
            rng.Vector(static_cast<size_t>(ldb) * std::max<hwyblas_int>(1, nrhs));
        for (const char uplo : {'U', 'L', 'u', 'l'}) {
          if (!IsComplex<T>::value && uplo != 'L') continue;
          SCOPED_TRACE(::testing::Message()
                       << "nb=" << nb << " nrhs=" << nrhs << " " << uplo);
          // For UPLO = 'U', E holds the superdiagonal of U = L**H: the
          // conjugate of the subdiagonal of L.
          std::vector<T> eu = ef;
          if (std::toupper(uplo) == 'U') {
            for (T& v : eu) v = Conj(v);
          }
          std::vector<T> x = b;
          ASSERT_EQ(Pttrs(uplo, n, nrhs, df, eu, x, ldb), 0);
          if (n == 0 || nrhs == 0) {
            EXPECT_EQ(x, b);
            continue;
          }
          EXPECT_LT(SolveRatio(da, x, ldb, b, ldb, n, nrhs), kLapackThreshold);
          std::vector<T> x2 = b;
          Ptts2(std::toupper(uplo) == 'U' ? 1 : 0, n, nrhs, df, eu, x2, ldb);
          EXPECT_EQ(x2, x);
        }
        // xPTSV.
        std::vector<R> d2 = d;
        std::vector<T> e2 = e, x = b;
        ASSERT_EQ(Ptsv(n, nrhs, d2, e2, x, ldb), 0);
        EXPECT_EQ(d2, df);
        EXPECT_EQ(e2, ef);
        if (n > 0 && nrhs > 0) {
          EXPECT_LT(SolveRatio(da, x, ldb, b, ldb, n, nrhs), kLapackThreshold);
        }
      }
    }
  }
  // Not positive definite at k (at each position of the unrolled loop, and
  // at N): INFO = k.
  for (const hwyblas_int n : {7, 8, 9}) {
    for (hwyblas_int k = 1; k <= n; ++k) {
      std::vector<R> d(n, R(4));
      std::vector<T> e = rng.Vector(n);
      d[k - 1] = R(-1);
      std::vector<R> d2 = d;
      std::vector<T> e2 = e, b = rng.Vector(n), b0 = b;
      EXPECT_EQ(Pttrf(n, d, e), k) << n << " " << k;
      EXPECT_EQ(Ptsv(n, 1, d2, e2, b, n), k);
      EXPECT_EQ(b, b0);
    }
  }
}

// ---------------------------------------------------------------------------
// Errors

TYPED_TEST(BandTest, Errors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(64, T(1)), b(16, T(1));
  std::vector<hwyblas_int> ipiv(8, 1);
  std::vector<R> cnorm(8), d(8, R(2));
  R rcond, scale;
  ErrorCapture errors;
  const auto check = [&](const char* name, const std::vector<int>& args) {
    std::vector<std::string> expected;
    expected.reserve(args.size());
    for (const int arg : args) {
      expected.push_back(Name<T>(name) + ":" + std::to_string(arg));
    }
    EXPECT_EQ(errors.Take(), expected) << name;
  };
  for (const bool unblocked : {false, true}) {
    EXPECT_EQ(Gbtrf(unblocked, -1, 4, 1, 1, a, 4, ipiv), -1);
    EXPECT_EQ(Gbtrf(unblocked, 4, -1, 1, 1, a, 4, ipiv), -2);
    EXPECT_EQ(Gbtrf(unblocked, 4, 4, -1, 1, a, 4, ipiv), -3);
    EXPECT_EQ(Gbtrf(unblocked, 4, 4, 1, -1, a, 4, ipiv), -4);
    EXPECT_EQ(Gbtrf(unblocked, 4, 4, 1, 1, a, 3, ipiv), -6);
    check(unblocked ? "GBTF2" : "GBTRF", {1, 2, 3, 4, 6});
  }
  EXPECT_EQ(Gbtrs('X', 4, 1, 1, 1, a, 4, ipiv, b, 4), -1);
  EXPECT_EQ(Gbtrs('N', -1, 1, 1, 1, a, 4, ipiv, b, 4), -2);
  EXPECT_EQ(Gbtrs('N', 4, -1, 1, 1, a, 4, ipiv, b, 4), -3);
  EXPECT_EQ(Gbtrs('N', 4, 1, -1, 1, a, 4, ipiv, b, 4), -4);
  EXPECT_EQ(Gbtrs('N', 4, 1, 1, -1, a, 4, ipiv, b, 4), -5);
  EXPECT_EQ(Gbtrs('N', 4, 1, 1, 1, a, 3, ipiv, b, 4), -7);
  EXPECT_EQ(Gbtrs('N', 4, 1, 1, 1, a, 4, ipiv, b, 3), -10);
  check("GBTRS", {1, 2, 3, 4, 5, 7, 10});
  EXPECT_EQ(Gbsv(-1, 1, 1, 1, a, 4, ipiv, b, 4), -1);
  EXPECT_EQ(Gbsv(4, -1, 1, 1, a, 4, ipiv, b, 4), -2);
  EXPECT_EQ(Gbsv(4, 1, -1, 1, a, 4, ipiv, b, 4), -3);
  EXPECT_EQ(Gbsv(4, 1, 1, -1, a, 4, ipiv, b, 4), -4);
  EXPECT_EQ(Gbsv(4, 1, 1, 1, a, 3, ipiv, b, 4), -6);
  EXPECT_EQ(Gbsv(4, 1, 1, 1, a, 4, ipiv, b, 3), -9);
  check("GBSV", {1, 2, 3, 4, 6, 9});
  EXPECT_EQ(Gbcon('X', 4, 1, 1, a, 4, ipiv, R(1), &rcond), -1);
  EXPECT_EQ(Gbcon('1', -1, 1, 1, a, 4, ipiv, R(1), &rcond), -2);
  EXPECT_EQ(Gbcon('1', 4, -1, 1, a, 4, ipiv, R(1), &rcond), -3);
  EXPECT_EQ(Gbcon('1', 4, 1, -1, a, 4, ipiv, R(1), &rcond), -4);
  EXPECT_EQ(Gbcon('1', 4, 1, 1, a, 3, ipiv, R(1), &rcond), -6);
  EXPECT_EQ(Gbcon('1', 4, 1, 1, a, 4, ipiv, R(-1), &rcond), -8);
  check("GBCON", {1, 2, 3, 4, 6, 8});
  std::vector<T> x(8);
  EXPECT_EQ(Latbs('X', 'N', 'N', 'N', 4, 1, a, 2, x, &scale, cnorm), -1);
  EXPECT_EQ(Latbs('U', 'X', 'N', 'N', 4, 1, a, 2, x, &scale, cnorm), -2);
  EXPECT_EQ(Latbs('U', 'N', 'X', 'N', 4, 1, a, 2, x, &scale, cnorm), -3);
  EXPECT_EQ(Latbs('U', 'N', 'N', 'X', 4, 1, a, 2, x, &scale, cnorm), -4);
  EXPECT_EQ(Latbs('U', 'N', 'N', 'N', -1, 1, a, 2, x, &scale, cnorm), -5);
  EXPECT_EQ(Latbs('U', 'N', 'N', 'N', 4, -1, a, 2, x, &scale, cnorm), -6);
  EXPECT_EQ(Latbs('U', 'N', 'N', 'N', 4, 1, a, 1, x, &scale, cnorm), -8);
  check("LATBS", {1, 2, 3, 4, 5, 6, 8});
  EXPECT_EQ(Potf2('X', 4, a, 4), -1);
  EXPECT_EQ(Potf2('U', -1, a, 4), -2);
  EXPECT_EQ(Potf2('U', 4, a, 3), -4);
  check("POTF2", {1, 2, 4});
  for (const bool unblocked : {false, true}) {
    EXPECT_EQ(Pbtrf(unblocked, 'X', 4, 1, a, 2), -1);
    EXPECT_EQ(Pbtrf(unblocked, 'U', -1, 1, a, 2), -2);
    EXPECT_EQ(Pbtrf(unblocked, 'U', 4, -1, a, 2), -3);
    EXPECT_EQ(Pbtrf(unblocked, 'U', 4, 1, a, 1), -5);
    check(unblocked ? "PBTF2" : "PBTRF", {1, 2, 3, 5});
  }
  EXPECT_EQ(Pbtrs('X', 4, 1, 1, a, 2, b, 4), -1);
  EXPECT_EQ(Pbtrs('U', -1, 1, 1, a, 2, b, 4), -2);
  EXPECT_EQ(Pbtrs('U', 4, -1, 1, a, 2, b, 4), -3);
  EXPECT_EQ(Pbtrs('U', 4, 1, -1, a, 2, b, 4), -4);
  EXPECT_EQ(Pbtrs('U', 4, 1, 1, a, 1, b, 4), -6);
  EXPECT_EQ(Pbtrs('U', 4, 1, 1, a, 2, b, 3), -8);
  check("PBTRS", {1, 2, 3, 4, 6, 8});
  EXPECT_EQ(Pbsv('X', 4, 1, 1, a, 2, b, 4), -1);
  EXPECT_EQ(Pbsv('U', -1, 1, 1, a, 2, b, 4), -2);
  EXPECT_EQ(Pbsv('U', 4, -1, 1, a, 2, b, 4), -3);
  EXPECT_EQ(Pbsv('U', 4, 1, -1, a, 2, b, 4), -4);
  EXPECT_EQ(Pbsv('U', 4, 1, 1, a, 1, b, 4), -6);
  EXPECT_EQ(Pbsv('U', 4, 1, 1, a, 2, b, 3), -8);
  check("PBSV", {1, 2, 3, 4, 6, 8});
  Random<T> rng1(1);
  Tridiagonal<T> t = RandomTridiagonal(rng1, 4);
  EXPECT_EQ(Gttrf(-1, t), -1);
  check("GTTRF", {1});
  ASSERT_EQ(Gttrf(4, t), 0);
  EXPECT_EQ(Gttrs('X', 4, 1, t, b, 4), -1);
  EXPECT_EQ(Gttrs('N', -1, 1, t, b, 4), -2);
  EXPECT_EQ(Gttrs('N', 4, -1, t, b, 4), -3);
  EXPECT_EQ(Gttrs('N', 4, 1, t, b, 3), -10);
  check("GTTRS", {1, 2, 3, 10});
  EXPECT_EQ(Gtsv(-1, 1, t, b, 4), -1);
  EXPECT_EQ(Gtsv(4, -1, t, b, 4), -2);
  EXPECT_EQ(Gtsv(4, 1, t, b, 3), -7);
  check("GTSV", {1, 2, 7});
  std::vector<T> e(8, T(0));
  EXPECT_EQ(Pttrf(-1, d, e), -1);
  check("PTTRF", {1});
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(Pttrs('X', 4, 1, d, e, b, 4), -1);
    EXPECT_EQ(Pttrs('U', -1, 1, d, e, b, 4), -2);
    EXPECT_EQ(Pttrs('U', 4, -1, d, e, b, 4), -3);
    EXPECT_EQ(Pttrs('U', 4, 1, d, e, b, 3), -7);
    check("PTTRS", {1, 2, 3, 7});
  } else {
    EXPECT_EQ(Pttrs('U', -1, 1, d, e, b, 4), -1);
    EXPECT_EQ(Pttrs('U', 4, -1, d, e, b, 4), -2);
    EXPECT_EQ(Pttrs('U', 4, 1, d, e, b, 3), -6);
    check("PTTRS", {1, 2, 6});
  }
  EXPECT_EQ(Ptsv(-1, 1, d, e, b, 4), -1);
  EXPECT_EQ(Ptsv(4, -1, d, e, b, 4), -2);
  EXPECT_EQ(Ptsv(4, 1, d, e, b, 3), -6);
  check("PTSV", {1, 2, 6});
}

// ---------------------------------------------------------------------------
// LAPACKE

// The row-major functions give the results of the column-major ones; in
// row-major layout, a band matrix is the transpose of the band storage.
TYPED_TEST(BandTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(41);
  const lapack_int n = 9, kl = 2, ku = 3, nrhs = 3, ldab = 2 * kl + ku + 2,
                   ldr = n + 2, ldb = n + 1, ldbr = nrhs + 1;
  const auto same = [&](const std::vector<T>& row, const std::vector<T>& col,
                        lapack_int rows, lapack_int cols, lapack_int ldrow,
                        lapack_int ldcol) {
    for (lapack_int i = 0; i < rows; ++i) {
      for (lapack_int j = 0; j < cols; ++j) {
        const T c = At(col, i, j, ldcol), r = row[i * ldrow + j];
        // NaN outside the band in both.
        if (std::isnan(std::real(c))) {
          ASSERT_TRUE(std::isnan(std::real(r))) << i << ", " << j;
        } else {
          ASSERT_EQ(r, c) << i << ", " << j;
        }
      }
    }
  };
  // xGBSV, xGBTRF, xGBTRS, xGBCON. (The NaN checks are off: AB has NaN
  // outside the band.)
  LAPACKE_set_nancheck(0);
  std::vector<T> ab = RandomBand(rng, n, n, kl, ku, kl + ku, ldab);
  const std::vector<T> b = rng.Vector(static_cast<size_t>(ldb) * nrhs);
  std::vector<T> fc = ab, fr = RowMajor(ab, ldab, n, ldab, ldr);
  std::vector<T> xc = b, xr = RowMajor(b, n, nrhs, ldb, ldbr);
  std::vector<lapack_int> pc(n), pr(n);
  ASSERT_EQ(LAPACKE_CALL(T, gbsv, LAPACK_COL_MAJOR, n, kl, ku, nrhs, fc.data(),
                         ldab, pc.data(), xc.data(), ldb),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, gbsv, LAPACK_ROW_MAJOR, n, kl, ku, nrhs, fr.data(),
                         ldr, pr.data(), xr.data(), ldbr),
            0);
  EXPECT_EQ(pr, pc);
  same(fr, fc, 2 * kl + ku + 1, n, ldr, ldab);
  same(xr, xc, n, nrhs, ldbr, ldb);
  std::vector<T> gc = ab, gr = RowMajor(ab, ldab, n, ldab, ldr);
  ASSERT_EQ(LAPACKE_CALL(T, gbtrf, LAPACK_COL_MAJOR, n, n, kl, ku, gc.data(),
                         ldab, pc.data()),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, gbtrf, LAPACK_ROW_MAJOR, n, n, kl, ku, gr.data(),
                         ldr, pr.data()),
            0);
  EXPECT_EQ(pr, pc);
  same(gr, gc, 2 * kl + ku + 1, n, ldr, ldab);
  EXPECT_TRUE(SameValues(gr, fr));
  for (const char trans : Ops<T>()) {
    std::vector<T> yc = b, yr = RowMajor(b, n, nrhs, ldb, ldbr);
    ASSERT_EQ(LAPACKE_CALL(T, gbtrs, LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
                           gc.data(), ldab, pc.data(), yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gbtrs, LAPACK_ROW_MAJOR, trans, n, kl, ku, nrhs,
                           gr.data(), ldr, pr.data(), yr.data(), ldbr),
              0);
    same(yr, yc, n, nrhs, ldbr, ldb);
  }
  for (const char norm : {'1', 'I'}) {
    R rc = -1, rr = -2;
    ASSERT_EQ(LAPACKE_CALL(T, gbcon, LAPACK_COL_MAJOR, norm, n, kl, ku,
                           gc.data(), ldab, pc.data(), R(3), &rc),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gbcon, LAPACK_ROW_MAJOR, norm, n, kl, ku,
                           gr.data(), ldr, pr.data(), R(3), &rr),
              0);
    EXPECT_GT(rc, 0);
    EXPECT_EQ(rr, rc);
  }
  // xPBSV, xPBTRF, xPBTRS.
  for (const char uplo : {'U', 'L'}) {
    const lapack_int kd = 3, ldpb = kd + 2;
    const std::vector<T> pb = PositiveDefiniteBand(rng, uplo, n, kd, ldpb);
    std::vector<T> qc = pb, qr = RowMajor(pb, ldpb, n, ldpb, ldr);
    ASSERT_EQ(LAPACKE_CALL(T, pbtrf, LAPACK_COL_MAJOR, uplo, n, kd, qc.data(),
                           ldpb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, pbtrf, LAPACK_ROW_MAJOR, uplo, n, kd, qr.data(),
                           ldr),
              0);
    same(qr, qc, kd + 1, n, ldr, ldpb);
    std::vector<T> yc = b, yr = RowMajor(b, n, nrhs, ldb, ldbr);
    ASSERT_EQ(LAPACKE_CALL(T, pbtrs, LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
                           qc.data(), ldpb, yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, pbtrs, LAPACK_ROW_MAJOR, uplo, n, kd, nrhs,
                           qr.data(), ldr, yr.data(), ldbr),
              0);
    same(yr, yc, n, nrhs, ldbr, ldb);
    std::vector<T> sc = pb, sr = RowMajor(pb, ldpb, n, ldpb, ldr);
    std::vector<T> zc = b, zr = RowMajor(b, n, nrhs, ldb, ldbr);
    ASSERT_EQ(LAPACKE_CALL(T, pbsv, LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
                           sc.data(), ldpb, zc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, pbsv, LAPACK_ROW_MAJOR, uplo, n, kd, nrhs,
                           sr.data(), ldr, zr.data(), ldbr),
              0);
    same(sr, sc, kd + 1, n, ldr, ldpb);
    same(zr, zc, n, nrhs, ldbr, ldb);
    same(zr, yc, n, nrhs, ldbr, ldb);
  }
  LAPACKE_set_nancheck(1);
  // Tridiagonal: xGTTRF (no layout), xGTTRS, xGTSV, xPTTRF, xPTTRS, xPTSV.
  Tridiagonal<T> t = RandomTridiagonal(rng, n);
  std::vector<T> du2(n);
  std::vector<lapack_int> ip(n);
  Tridiagonal<T> f = t;
  ASSERT_EQ(LAPACKE_CALL(T, gttrf, n, f.dl.data(), f.d.data(), f.du.data(),
                         du2.data(), ip.data()),
            0);
  Tridiagonal<T> f2 = t;
  ASSERT_EQ(Gttrf(n, f2), 0);
  EXPECT_EQ(f.d, f2.d);
  EXPECT_EQ(ip, f2.ipiv);
  for (const char trans : Ops<T>()) {
    std::vector<T> yc = b, yr = RowMajor(b, n, nrhs, ldb, ldbr);
    ASSERT_EQ(LAPACKE_CALL(T, gttrs, LAPACK_COL_MAJOR, trans, n, nrhs,
                           f.dl.data(), f.d.data(), f.du.data(), du2.data(),
                           ip.data(), yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gttrs, LAPACK_ROW_MAJOR, trans, n, nrhs,
                           f.dl.data(), f.d.data(), f.du.data(), du2.data(),
                           ip.data(), yr.data(), ldbr),
              0);
    same(yr, yc, n, nrhs, ldbr, ldb);
  }
  {
    Tridiagonal<T> gc = t, gr = t;
    std::vector<T> yc = b, yr = RowMajor(b, n, nrhs, ldb, ldbr);
    ASSERT_EQ(LAPACKE_CALL(T, gtsv, LAPACK_COL_MAJOR, n, nrhs, gc.dl.data(),
                           gc.d.data(), gc.du.data(), yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, gtsv, LAPACK_ROW_MAJOR, n, nrhs, gr.dl.data(),
                           gr.d.data(), gr.du.data(), yr.data(), ldbr),
              0);
    same(yr, yc, n, nrhs, ldbr, ldb);
  }
  std::vector<R> d(n);
  for (R& v : d) v = R(3);
  std::vector<T> e = rng.Vector(n);
  std::vector<R> dc = d, dr = d;
  std::vector<T> ec = e, er = e;
  ASSERT_EQ(LAPACKE_CALL(T, pttrf, n, dc.data(), ec.data()), 0);
  std::vector<T> yc = b, yr = RowMajor(b, n, nrhs, ldb, ldbr);
  if constexpr (IsComplex<T>::value) {
    ASSERT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_COL_MAJOR, 'L', n, nrhs, dc.data(),
                           ec.data(), yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_ROW_MAJOR, 'L', n, nrhs, dc.data(),
                           ec.data(), yr.data(), ldbr),
              0);
  } else {
    ASSERT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_COL_MAJOR, n, nrhs, dc.data(),
                           ec.data(), yc.data(), ldb),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_ROW_MAJOR, n, nrhs, dc.data(),
                           ec.data(), yr.data(), ldbr),
              0);
  }
  same(yr, yc, n, nrhs, ldbr, ldb);
  std::vector<T> zr = RowMajor(b, n, nrhs, ldb, ldbr);
  ASSERT_EQ(LAPACKE_CALL(T, ptsv, LAPACK_ROW_MAJOR, n, nrhs, dr.data(),
                         er.data(), zr.data(), ldbr),
            0);
  EXPECT_EQ(dr, dc);
  EXPECT_EQ(er, ec);
  same(zr, yc, n, nrhs, ldbr, ldb);
}

TYPED_TEST(BandTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> ab(64, T(1)), b(16, T(1)), dl(4, T(1)), du(4, T(1)),
      du2(4, T(1)), dt(4, T(4)), e(4, T(0));
  std::vector<lapack_int> ipiv = {1, 2, 3, 4};
  std::vector<R> d(4, R(4));
  R rcond = 0;
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LAPACKE_CALL(T, gbtrf, 0, 4, 4, 1, 1, ab.data(), 4, ipiv.data()),
            -1);
  EXPECT_EQ(LAPACKE_CALL(T, gbtrf, LAPACK_ROW_MAJOR, 4, 4, 1, 1, ab.data(), 3,
                         ipiv.data()),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, gbtrs, LAPACK_ROW_MAJOR, 'N', 4, 1, 1, 1,
                         ab.data(), 3, ipiv.data(), b.data(), 1),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, gbtrs, LAPACK_ROW_MAJOR, 'N', 4, 1, 1, 2,
                         ab.data(), 4, ipiv.data(), b.data(), 1),
            -11);
  EXPECT_EQ(LAPACKE_CALL(T, gbsv, LAPACK_ROW_MAJOR, 4, 1, 1, 1, ab.data(), 3,
                         ipiv.data(), b.data(), 1),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, gbsv, LAPACK_ROW_MAJOR, 4, 1, 1, 2, ab.data(), 4,
                         ipiv.data(), b.data(), 1),
            -10);
  EXPECT_EQ(LAPACKE_CALL(T, gbcon, LAPACK_ROW_MAJOR, '1', 4, 1, 1, ab.data(),
                         3, ipiv.data(), R(1), &rcond),
            -7);
  EXPECT_EQ(
      LAPACKE_CALL(T, pbtrf, LAPACK_ROW_MAJOR, 'U', 4, 1, ab.data(), 3), -6);
  EXPECT_EQ(LAPACKE_CALL(T, pbtrs, LAPACK_ROW_MAJOR, 'U', 4, 1, 1, ab.data(),
                         3, b.data(), 1),
            -7);
  EXPECT_EQ(LAPACKE_CALL(T, pbtrs, LAPACK_ROW_MAJOR, 'U', 4, 1, 2, ab.data(),
                         4, b.data(), 1),
            -9);
  EXPECT_EQ(LAPACKE_CALL(T, pbsv, LAPACK_ROW_MAJOR, 'U', 4, 1, 2, ab.data(), 4,
                         b.data(), 1),
            -9);
  EXPECT_EQ(LAPACKE_CALL(T, gttrs, LAPACK_ROW_MAJOR, 'N', 4, 2, dl.data(),
                         dt.data(), du.data(), du2.data(), ipiv.data(),
                         b.data(), 1),
            -11);
  EXPECT_EQ(LAPACKE_CALL(T, gtsv, LAPACK_ROW_MAJOR, 4, 2, dl.data(), dt.data(),
                         du.data(), b.data(), 1),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, ptsv, LAPACK_ROW_MAJOR, 4, 2, d.data(), e.data(),
                         b.data(), 1),
            -7);
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_ROW_MAJOR, 'L', 4, 2, d.data(),
                           e.data(), b.data(), 1),
              -8);
  } else {
    EXPECT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_ROW_MAJOR, 4, 2, d.data(),
                           e.data(), b.data(), 1),
              -7);
  }
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + LapackeName<T>("gbtrf") +
                "\nWrong parameter 7 in " + LapackeName<T>("gbtrf_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("gbtrs_work") +
                "\nWrong parameter 11 in " + LapackeName<T>("gbtrs_work") +
                "\nWrong parameter 7 in " + LapackeName<T>("gbsv_work") +
                "\nWrong parameter 10 in " + LapackeName<T>("gbsv_work") +
                "\nWrong parameter 7 in " + LapackeName<T>("gbcon_work") +
                "\nWrong parameter 6 in " + LapackeName<T>("pbtrf_work") +
                "\nWrong parameter 7 in " + LapackeName<T>("pbtrs_work") +
                "\nWrong parameter 9 in " + LapackeName<T>("pbtrs_work") +
                "\nWrong parameter 9 in " + LapackeName<T>("pbsv_work") +
                "\nWrong parameter 11 in " + LapackeName<T>("gttrs_work") +
                "\nWrong parameter 8 in " + LapackeName<T>("gtsv_work") +
                "\nWrong parameter 7 in " + LapackeName<T>("ptsv_work") +
                "\nWrong parameter " + (IsComplex<T>::value ? "8" : "7") +
                " in " + LapackeName<T>("pttrs_work") + "\n");
  // NaN checks: the band of AB (also the first KL rows for xGBSV, as the
  // reference), B, the diagonals.
  std::vector<T> band(4 * 4, T(1));
  At(band, 0, 3, 4) = NaN<T>();  // fill-in row of xGBTRF / xGBSV (kl = 1)
  EXPECT_EQ(LAPACKE_CALL(T, gbsv, LAPACK_COL_MAJOR, 4, 1, 1, 1, band.data(), 4,
                         ipiv.data(), b.data(), 4),
            -6);
  EXPECT_EQ(LAPACKE_CALL(T, gbtrf, LAPACK_COL_MAJOR, 4, 4, 1, 1, band.data(),
                         4, ipiv.data()),
            -6);
  At(band, 0, 0, 4) = NaN<T>();  // outside the band: not checked
  At(band, 0, 3, 4) = T(1);
  EXPECT_EQ(LAPACKE_CALL(T, gbcon, LAPACK_COL_MAJOR, '1', 4, 1, 1, band.data(),
                         4, ipiv.data(), R(1), &rcond),
            0);
  EXPECT_EQ(LAPACKE_CALL(T, gbcon, LAPACK_COL_MAJOR, '1', 4, 1, 1, band.data(),
                         4, ipiv.data(), std::numeric_limits<R>::quiet_NaN(),
                         &rcond),
            -9);
  EXPECT_NE(
      LAPACKE_CALL(T, pbtrf, LAPACK_COL_MAJOR, 'U', 4, 1, band.data(), 2), -5);
  std::vector<T> pb(8, T(4));
  pb[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, pbtrf, LAPACK_COL_MAJOR, 'L', 4, 1, pb.data(), 2),
            -5);
  b[1] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, pbsv, LAPACK_COL_MAJOR, 'L', 4, 1, 1, pb.data(), 2,
                         b.data(), 4),
            -6);
  pb[2] = T(4);
  EXPECT_EQ(LAPACKE_CALL(T, pbtrs, LAPACK_COL_MAJOR, 'L', 4, 1, 1, pb.data(),
                         2, b.data(), 4),
            -8);
  EXPECT_EQ(LAPACKE_CALL(T, gtsv, LAPACK_COL_MAJOR, 4, 1, dl.data(), dt.data(),
                         du.data(), b.data(), 4),
            -7);
  b[1] = T(1);
  du2[1] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, gttrs, LAPACK_COL_MAJOR, 'N', 4, 1, dl.data(),
                         dt.data(), du.data(), du2.data(), ipiv.data(),
                         b.data(), 4),
            -8);
  du[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, gttrf, 4, dl.data(), dt.data(), du.data(),
                         du2.data(), ipiv.data()),
            -4);
  e[2] = NaN<T>();
  EXPECT_EQ(LAPACKE_CALL(T, pttrf, 4, d.data(), e.data()), -3);
  EXPECT_EQ(LAPACKE_CALL(T, ptsv, LAPACK_COL_MAJOR, 4, 1, d.data(), e.data(),
                         b.data(), 4),
            -5);
  d[0] = std::numeric_limits<R>::quiet_NaN();
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_COL_MAJOR, 'L', 4, 1, d.data(),
                           e.data(), b.data(), 4),
              -5);
  } else {
    EXPECT_EQ(LAPACKE_CALL(T, pttrs, LAPACK_COL_MAJOR, 4, 1, d.data(),
                           e.data(), b.data(), 4),
              -4);
  }
}

}  // namespace
}  // namespace hwyblas_test
