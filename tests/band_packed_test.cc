// Tests for the banded and packed Level 2 routines against dense references
// computed in double precision. The test matrices are created densely (with
// StructuredMatrix, zero outside the band) and converted to band or packed
// storage; storage that BLAS must not read holds NaN.
//
// The sizes cover both code paths of the square band and packed routines:
// column by column (fewer than 128 off-diagonals) and in strips.
#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <complex>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class BandPackedTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(BandPackedTest, ScalarTypes);

// Band storage (leading dimension ldab >= kl + ku + 1) of the rows x n matrix
// a (leading dimension lda): element (i, j) goes to (ku + i - j) + j * ldab.
// Elements outside the band are dropped; unused storage is NaN.
template <typename T>
std::vector<T> ToBand(const std::vector<T>& a, int rows, int n, int lda, int kl,
                      int ku, int ldab) {
  std::vector<T> ab(static_cast<size_t>(ldab) * n, NaN<T>());
  for (int j = 0; j < n; ++j) {
    for (int i = std::max(0, j - ku); i < std::min(rows, j + kl + 1); ++i) {
      ab[(ku + i - j) + static_cast<size_t>(j) * ldab] =
          a[i + static_cast<size_t>(j) * lda];
    }
  }
  return ab;
}

// Packed storage of the uplo triangle of the n x n matrix a.
template <typename T>
std::vector<T> ToPacked(const std::vector<T>& a, int n, int lda, char uplo) {
  std::vector<T> ap;
  for (int j = 0; j < n; ++j) {
    const int begin = (uplo == 'L') ? j : 0, end = (uplo == 'L') ? n : j + 1;
    for (int i = begin; i < end; ++i) {
      ap.push_back(a[i + static_cast<size_t>(j) * lda]);
    }
  }
  return ap;
}

// Sets the elements of a (n columns, leading dimension lda) farther than kl
// below or ku above the diagonal to zero.
template <typename T>
void ClearOutsideBand(std::vector<T>& a, int rows, int n, int lda, int kl,
                      int ku) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (i - j > kl || j - i > ku) a[i + static_cast<size_t>(j) * lda] = T(0);
    }
  }
}

// Storage formats of the structured routines.
enum class Format { kBand, kPacked };

const char* Name(Format f) { return f == Format::kBand ? "band" : "packed"; }

// ---------------------------------------------------------------------------
// GBMV

template <typename T>
void CheckGbmv(char trans, int m, int n, int kl, int ku, int incx, int incy,
               T alpha, T beta, uint64_t seed) {
  const int ldab = kl + ku + 2;
  Random<T> rng(seed);
  std::vector<T> a = rng.Vector(static_cast<size_t>(std::max(m, 1)) * n);
  ClearOutsideBand(a, m, n, std::max(m, 1), kl, ku);
  const std::vector<T> ab = ToBand(a, m, n, std::max(m, 1), kl, ku, ldab);

  const bool no_trans = (trans == 'N');
  const int len_x = no_trans ? n : m, len_y = no_trans ? m : n;
  StridedVector<T> x(len_x, incx, rng), y(len_y, incy, rng);
  if (beta == T(0)) {
    for (int i = 0; i < len_y; ++i) y[i] = NaN<T>();
  }
  std::vector<T> y0(len_y);
  for (int i = 0; i < len_y; ++i) y0[i] = y[i];

  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  // As in the reference BLAS, y is left unchanged (not scaled by beta) if
  // m == 0 or n == 0.
  const bool empty = (m == 0 || n == 0);
  ReferenceProduct(DenseOp(trans, a, m, n, std::max(m, 1)), x.ToDense(),
                   empty ? T(0) : alpha, empty ? T(1) : beta, y0, len_y,
                   &expected, &bound);
  HWYBLAS_CALL(T, gbmv, &trans, &m, &n, &kl, &ku, P(&alpha), P(ab.data()),
               &ldab, P(x.data.data()), &incx, P(&beta), P(y.data.data()),
               &incy);
  const double tol = 4.0 * (std::max(m, n) + 2) * Eps<T>();
  EXPECT_TRUE(
      NearWithBound(y.Elements(), expected, bound, tol, len_y, 1, len_y))
      << "trans=" << trans << " m=" << m << " n=" << n << " kl=" << kl
      << " ku=" << ku << " incx=" << incx << " incy=" << incy;
}

TYPED_TEST(BandPackedTest, Gbmv) {
  using T = TypeParam;
  const int shapes[][2] = {{0, 5}, {5, 0},   {1, 1},   {7, 3},
                           {3, 7}, {40, 25}, {25, 40}, {100, 100}};
  ForEachTarget([&] {
    uint64_t seed = 1;
    for (char trans : Ops<T>()) {
      for (const auto& s : shapes) {
        for (int kl : {0, 1, 5, 120}) {
          for (int ku : {0, 2, 7}) {
            CheckGbmv<T>(trans, s[0], s[1], kl, ku, 1, 1,
                         MakeScalar<T>(1.5, -0.5), MakeScalar<T>(0.5, 0.25),
                         seed++);
          }
        }
      }
      CheckGbmv<T>(trans, 30, 20, 3, 4, -2, 3, MakeScalar<T>(-1.0, 0.5), T(0),
                   seed++);
      CheckGbmv<T>(trans, 20, 30, 2, 1, 1, 1, T(0), MakeScalar<T>(2.0, 0.0),
                   seed++);
    }
  });
}

// ---------------------------------------------------------------------------
// SBMV, HBMV, SPMV, HPMV

template <typename T>
void CheckSymmetricMv(Format format, char uplo, int n, int k, int incx,
                      int incy, T alpha, T beta, uint64_t seed) {
  constexpr bool kHermitian = IsComplex<T>::value;
  const Shape shape = kHermitian ? Shape::kHermitian : Shape::kSymmetric;
  const int lda = std::max(n, 1), ldab = k + 3;
  Random<T> rng(seed);
  std::vector<T> a = StructuredMatrix(shape, uplo, n, lda, rng);
  if (format == Format::kBand) ClearOutsideBand(a, n, n, lda, k, k);
  const std::vector<T> stored =
      (format == Format::kBand)
          ? ToBand(a, n, n, lda, uplo == 'L' ? k : 0, uplo == 'L' ? 0 : k, ldab)
          : ToPacked(a, n, lda, uplo);

  StridedVector<T> x(n, incx, rng), y(n, incy, rng);
  if (beta == T(0)) {
    for (int i = 0; i < n; ++i) y[i] = NaN<T>();
  }
  std::vector<T> y0(n);
  for (int i = 0; i < n; ++i) y0[i] = y[i];
  std::vector<RefT<T>> expected;
  std::vector<double> bound;
  ReferenceProduct(DenseStructured(shape, uplo, 'N', a, n, lda), x.ToDense(),
                   alpha, beta, y0, n, &expected, &bound);

  if (format == Format::kBand) {
    if constexpr (kHermitian) {
      HWYBLAS_CALL_COMPLEX(T, hbmv, &uplo, &n, &k, P(&alpha), P(stored.data()),
                           &ldab, P(x.data.data()), &incx, P(&beta),
                           P(y.data.data()), &incy);
    } else {
      HWYBLAS_CALL_REAL(T, sbmv, &uplo, &n, &k, P(&alpha), P(stored.data()),
                        &ldab, P(x.data.data()), &incx, P(&beta),
                        P(y.data.data()), &incy);
    }
  } else {
    if constexpr (kHermitian) {
      HWYBLAS_CALL_COMPLEX(T, hpmv, &uplo, &n, P(&alpha), P(stored.data()),
                           P(x.data.data()), &incx, P(&beta), P(y.data.data()),
                           &incy);
    } else {
      HWYBLAS_CALL_REAL(T, spmv, &uplo, &n, P(&alpha), P(stored.data()),
                        P(x.data.data()), &incx, P(&beta), P(y.data.data()),
                        &incy);
    }
  }
  const double tol = 4.0 * (n + 2) * Eps<T>();
  EXPECT_TRUE(NearWithBound(y.Elements(), expected, bound, tol, n, 1, n))
      << Name(format) << " uplo=" << uplo << " n=" << n << " k=" << k
      << " incx=" << incx << " incy=" << incy;
}

TYPED_TEST(BandPackedTest, SymmetricMv) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (Format format : {Format::kBand, Format::kPacked}) {
      // k only matters for band storage.
      const std::vector<int> ks = (format == Format::kBand)
                                      ? std::vector<int>{0, 1, 3, 8, 200}
                                      : std::vector<int>{0};
      for (char uplo : {'L', 'U'}) {
        for (int n : {0, 1, 2, 3, 5, 8, 17, 64, 130}) {
          for (int k : ks) {
            CheckSymmetricMv<T>(format, uplo, n, k, 1, 1,
                                MakeScalar<T>(1.5, -0.5),
                                MakeScalar<T>(0.5, 0.25), seed++);
          }
        }
        // Band strips with ragged rows (128 <= k < n).
        CheckSymmetricMv<T>(format, uplo, 300, 150, 1, 1,
                            MakeScalar<T>(0.5, 1.0), MakeScalar<T>(1.0, 0.0),
                            seed++);
        CheckSymmetricMv<T>(format, uplo, 50, 4, 2, -3,
                            MakeScalar<T>(-1.0, 0.5), T(0), seed++);
        CheckSymmetricMv<T>(format, uplo, 20, 3, -1, 1, T(0),
                            MakeScalar<T>(2.0, 0.0), seed++);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// TBMV, TBSV, TPMV, TPSV

template <typename T>
void CheckTriangular(Format format, bool solve, char uplo, char trans,
                     char diag, int n, int k, int incx, uint64_t seed) {
  const Shape shape =
      (diag == 'U') ? Shape::kUnitTriangular : Shape::kTriangular;
  const int lda = std::max(n, 1), ldab = k + 2;
  Random<T> rng(seed);
  // Scaled off-diagonal part keeps the solves well conditioned.
  std::vector<T> a = StructuredMatrix(shape, uplo, n, lda, rng,
                                      solve ? 1.0 / std::max(n, 1) : 1.0);
  if (format == Format::kBand) ClearOutsideBand(a, n, n, lda, k, k);
  const std::vector<T> stored =
      (format == Format::kBand)
          ? ToBand(a, n, n, lda, uplo == 'L' ? k : 0, uplo == 'L' ? 0 : k, ldab)
          : ToPacked(a, n, lda, uplo);

  StridedVector<T> x(n, incx, rng);
  std::vector<T> b(n);
  for (int i = 0; i < n; ++i) b[i] = x[i];

  const T* s = stored.data();
  T* xd = x.data.data();
  if (format == Format::kBand) {
    if (solve) {
      HWYBLAS_CALL(T, tbsv, &uplo, &trans, &diag, &n, &k, P(s), &ldab, P(xd),
                   &incx);
    } else {
      HWYBLAS_CALL(T, tbmv, &uplo, &trans, &diag, &n, &k, P(s), &ldab, P(xd),
                   &incx);
    }
  } else {
    if (solve) {
      HWYBLAS_CALL(T, tpsv, &uplo, &trans, &diag, &n, P(s), P(xd), &incx);
    } else {
      HWYBLAS_CALL(T, tpmv, &uplo, &trans, &diag, &n, P(s), P(xd), &incx);
    }
  }

  const Dense<T> op_a = DenseStructured(shape, uplo, trans, a, n, lda);
  std::vector<RefT<T>> product;
  std::vector<double> bound;
  const std::vector<T> result = x.Elements();
  if (solve) {
    // Residual check: op(A) * x against b.
    ReferenceProduct(op_a, x.ToDense(), T(1), T(0), std::vector<T>(n), n,
                     &product, &bound);
    for (int i = 0; i < n; ++i) bound[i] += Abs(b[i]);
    EXPECT_TRUE(
        NearWithBound(b, product, bound, 8.0 * (n + 2) * Eps<T>(), n, 1, n))
        << Name(format) << " tbsv/tpsv uplo=" << uplo << " trans=" << trans
        << " diag=" << diag << " n=" << n << " k=" << k << " incx=" << incx;
  } else {
    Dense<T> db(n, 1);
    for (int i = 0; i < n; ++i) db(i, 0) = static_cast<RefT<T>>(b[i]);
    ReferenceProduct(op_a, db, T(1), T(0), std::vector<T>(n), n, &product,
                     &bound);
    EXPECT_TRUE(NearWithBound(result, product, bound, 4.0 * (n + 2) * Eps<T>(),
                              n, 1, n))
        << Name(format) << " tbmv/tpmv uplo=" << uplo << " trans=" << trans
        << " diag=" << diag << " n=" << n << " k=" << k << " incx=" << incx;
  }
}

TYPED_TEST(BandPackedTest, Triangular) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (Format format : {Format::kBand, Format::kPacked}) {
      const std::vector<int> ks = (format == Format::kBand)
                                      ? std::vector<int>{0, 1, 4, 200}
                                      : std::vector<int>{0};
      for (bool solve : {false, true}) {
        for (char uplo : {'L', 'U'}) {
          for (char trans : Ops<T>()) {
            for (char diag : {'N', 'U'}) {
              for (int n : {0, 1, 2, 5, 17, 70, 150}) {
                for (int k : ks) {
                  CheckTriangular<T>(format, solve, uplo, trans, diag, n, k, 1,
                                     seed++);
                }
              }
              CheckTriangular<T>(format, solve, uplo, trans, diag, 300, 150, 1,
                                 seed++);
              CheckTriangular<T>(format, solve, uplo, trans, diag, 33, 3, -2,
                                 seed++);
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// SPR, HPR, SPR2, HPR2

template <typename T>
void CheckPackedUpdate(int terms, char uplo, int n, int incx, int incy, T alpha,
                       uint64_t seed) {
  constexpr bool kHermitian = IsComplex<T>::value;
  const int lda = std::max(n, 1);
  Random<T> rng(seed);
  // For Hermitian matrices the imaginary parts of the diagonal are NaN: they
  // must not be read.
  const std::vector<T> a = StructuredMatrix(
      kHermitian ? Shape::kHermitian : Shape::kSymmetric, uplo, n, lda, rng);
  std::vector<T> ap = ToPacked(a, n, lda, uplo);
  const std::vector<T> ap0 = ap;
  StridedVector<T> x(n, incx, rng), y(n, incy, rng);
  if (terms == 1) alpha = T(std::real(alpha));  // SPR and HPR: real alpha

  if (terms == 1) {
    const RealT<T> real_alpha = std::real(alpha);
    if constexpr (kHermitian) {
      HWYBLAS_CALL_COMPLEX(T, hpr, &uplo, &n, &real_alpha, P(x.data.data()),
                           &incx, P(ap.data()));
    } else {
      HWYBLAS_CALL_REAL(T, spr, &uplo, &n, &real_alpha, P(x.data.data()), &incx,
                        P(ap.data()));
    }
  } else {
    if constexpr (kHermitian) {
      HWYBLAS_CALL_COMPLEX(T, hpr2, &uplo, &n, P(&alpha), P(x.data.data()),
                           &incx, P(y.data.data()), &incy, P(ap.data()));
    } else {
      HWYBLAS_CALL_REAL(T, spr2, &uplo, &n, P(&alpha), P(x.data.data()), &incx,
                        P(y.data.data()), &incy, P(ap.data()));
    }
  }

  const bool any = !(alpha == T(0));
  const RefT<T> al = static_cast<RefT<T>>(alpha);
  size_t failures = 0, idx = 0;
  for (int j = 0; j < n; ++j) {
    const int begin = (uplo == 'L') ? j : 0, end = (uplo == 'L') ? n : j + 1;
    for (int i = begin; i < end; ++i, ++idx) {
      if (!any) {
        if (memcmp(&ap[idx], &ap0[idx], sizeof(T)) != 0) ++failures;
        continue;
      }
      const RefT<T> xi = x[i], xj = x[j], yi = y[i], yj = y[j];
      RefT<T> update = al * xi * Conj(xj);
      double scale = Abs(alpha) * std::abs(xi) * std::abs(xj);
      if (terms == 2) {
        update = al * xi * Conj(yj) + Conj(al) * yi * Conj(xj);
        scale = Abs(alpha) *
                (std::abs(xi) * std::abs(yj) + std::abs(yi) * std::abs(xj));
      }
      RefT<T> a_ij = static_cast<RefT<T>>(ap0[idx]);
      if (kHermitian && i == j) {
        a_ij = std::real(a_ij);
        update = std::real(update);
        if (std::imag(ap[idx]) != 0) ++failures;
      }
      const double err =
          std::abs(static_cast<RefT<T>>(ap[idx]) - (a_ij + update));
      if (!(err <= 4 * Eps<T>() * (std::abs(a_ij) + scale))) ++failures;
    }
  }
  EXPECT_EQ(failures, 0u) << "terms=" << terms << " uplo=" << uplo << " n=" << n
                          << " incx=" << incx << " incy=" << incy;
  x.Elements();
  y.Elements();
}

TYPED_TEST(BandPackedTest, PackedUpdate) {
  using T = TypeParam;
  ForEachTarget([] {
    uint64_t seed = 1;
    for (int terms : {1, 2}) {
      for (char uplo : {'L', 'U'}) {
        for (int n : {0, 1, 2, 3, 5, 8, 17, 64, 101}) {
          CheckPackedUpdate<T>(terms, uplo, n, 1, 1, MakeScalar<T>(1.5, -0.5),
                               seed++);
        }
        CheckPackedUpdate<T>(terms, uplo, 40, 2, -1, MakeScalar<T>(-0.5, 2.0),
                             seed++);
        CheckPackedUpdate<T>(terms, uplo, 10, 1, 1, T(0), seed++);
      }
    }
  });
}

}  // namespace
}  // namespace hwyblas_test
