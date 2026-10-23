// Shared helpers of the singular value decomposition tests (gesvd_test.cc,
// gesdd_test.cc): bidiagonal and diagonal matrices, test bidiagonal
// matrices, the comparison of singular vectors, and LAPACKE layouts.
#ifndef HWYBLAS_TESTS_LAPACK_SVD_TEST_UTIL_H_
#define HWYBLAS_TESTS_LAPACK_SVD_TEST_UTIL_H_

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {

// Calls the real routine `rname` or the complex routine `cname`.
#define CALL_RC(T, rname, cname, ...)                              \
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

// The k x k bidiagonal matrix (d, e), upper or lower.
template <typename T>
Dense<T> Bidiagonal(const std::vector<RealT<T>>& d,
                    const std::vector<RealT<T>>& e, bool upper) {
  const size_t k = d.size();
  Dense<T> b(k, k);
  for (size_t i = 0; i < k; ++i) {
    b(i, i) = d[i];
    if (i + 1 < k) {
      if (upper) {
        b(i, i + 1) = e[i];
      } else {
        b(i + 1, i) = e[i];
      }
    }
  }
  return b;
}

// diag(s) as a k x k matrix.
template <typename T>
Dense<T> Diagonal(const std::vector<RealT<T>>& s, size_t k) {
  Dense<T> d(k, k);
  for (size_t i = 0; i < k; ++i) d(i, i) = s[i];
  return d;
}

// The first `cols` columns of a.
template <typename T>
Dense<T> Columns(const Dense<T>& a, size_t cols) {
  Dense<T> c(a.rows, cols);
  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < a.rows; ++i) c(i, j) = a(i, j);
  }
  return c;
}

// The first `rows` rows of a.
template <typename T>
Dense<T> Rows(const Dense<T>& a, size_t rows) {
  Dense<T> r(rows, a.cols);
  for (size_t j = 0; j < a.cols; ++j) {
    for (size_t i = 0; i < rows; ++i) r(i, j) = a(i, j);
  }
  return r;
}

// Singular values: non-negative and in decreasing order.
template <typename R>
void ExpectDecreasing(const std::vector<R>& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    ASSERT_GE(s[i], R(0)) << i;
    if (i > 0) {
      ASSERT_GE(s[i - 1], s[i]) << i;
    }
  }
}

struct Shape2 {
  hwyblas_int m, n;
};

// A bidiagonal matrix of one of several kinds: random, with zeros on the
// off-diagonal (it splits) or on the diagonal, graded (singular values
// over many orders of magnitude), with equal elements, or scaled near the
// overflow threshold or towards the underflow threshold.
template <typename R>
void TestBidiagonal(int kind, hwyblas_int n, Random<R>& rng, std::vector<R>& d,
                    std::vector<R>& e) {
  d = rng.Vector(n);
  e = rng.Vector(std::max<hwyblas_int>(0, n - 1));
  switch (kind) {
    case 1:  // splits
      for (hwyblas_int i = 0; i + 1 < n; i += 3) e[i] = 0;
      break;
    case 2:  // zeros on the diagonal
      for (hwyblas_int i = 0; i < n; i += 4) d[i] = 0;
      break;
    case 3:  // graded
      for (hwyblas_int i = 0; i < n; ++i) {
        const R scale = static_cast<R>(std::pow(10.0, -(i % 12)));
        d[i] *= scale;
        if (i + 1 < n) e[i] *= scale;
      }
      break;
    case 4:  // equal elements
      for (auto& x : d) x = 1;
      for (auto& x : e) x = 1;
      break;
    case 5:    // near overflow
    case 6: {  // near underflow
      // (Not closer to the underflow threshold: xBDSQR treats elements
      // below 6 * n**2 * safmin as zero, as it does not scale.)
      const R eps = std::numeric_limits<R>::epsilon();
      const R scale = kind == 5 ? std::numeric_limits<R>::max() / 8
                                : std::numeric_limits<R>::min() / (eps * eps);
      for (auto& x : d) x *= scale;
      for (auto& x : e) x *= scale;
      break;
    }
    default:
      break;
  }
}

// max_j ||x(:,j) - c_j y(:,j)||_1 / (rows * eps * g_j) over the columns of
// x and y, with |c_j| = 1 chosen so that the largest elements agree
// (LAPACK's xUNT03 ratio: singular vectors are determined up to such
// factors), and g_j = max(1, s_1 / gap_j) for the singular values s: the
// error bound of the j-th vector is proportional to 1 / gap_j, its distance
// to the other singular values (including 0 for x with more rows than
// columns).
template <typename T>
double VectorsRatio(const Dense<T>& x, const Dense<T>& y,
                    const std::vector<RealT<T>>& s) {
  const size_t k = x.cols;
  double worst = 0;
  for (size_t j = 0; j < k; ++j) {
    double gap = x.rows > k ? static_cast<double>(s[j]) : HUGE_VAL;
    if (j > 0) gap = std::min(gap, static_cast<double>(s[j - 1] - s[j]));
    if (j + 1 < k) gap = std::min(gap, static_cast<double>(s[j] - s[j + 1]));
    const double g = gap > 0 ? std::max(1.0, s[0] / gap) : 1.0;
    size_t imax = 0;
    for (size_t i = 0; i < x.rows; ++i) {
      if (std::abs(y(i, j)) > std::abs(y(imax, j))) imax = i;
    }
    RefT<T> c = 1;
    if (std::abs(y(imax, j)) > 0) {
      c = x(imax, j) / y(imax, j);
      c /= std::abs(c);
    }
    double diff = 0;
    for (size_t i = 0; i < x.rows; ++i) diff += std::abs(x(i, j) - c * y(i, j));
    worst = std::max(worst, diff / g);
  }
  return worst / std::max<double>(1, static_cast<double>(x.rows)) /
         LapackEps<T>();
}

#define LAPACKE_CALL(T, rname, cname, ...)                         \
  [&] {                                                            \
    if constexpr (std::is_same_v<T, float>) {                      \
      return LAPACKE_s##rname(__VA_ARGS__);                        \
    } else if constexpr (std::is_same_v<T, double>) {              \
      return LAPACKE_d##rname(__VA_ARGS__);                        \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      return LAPACKE_c##cname(__VA_ARGS__);                        \
    } else {                                                       \
      return LAPACKE_z##cname(__VA_ARGS__);                        \
    }                                                              \
  }()

// The m x n column-major matrix col (leading dimension ldc) in row-major
// layout with leading dimension ldr.
template <typename T>
std::vector<T> ToRowMajor(const std::vector<T>& col, lapack_int m, lapack_int n,
                          lapack_int ldc, lapack_int ldr) {
  std::vector<T> row(std::max<lapack_int>(1, m * ldr), NaN<T>());
  for (lapack_int i = 0; i < m; ++i) {
    for (lapack_int j = 0; j < n; ++j) row[i * ldr + j] = At(col, i, j, ldc);
  }
  return row;
}

template <typename T>
void ExpectSameRowMajor(const std::vector<T>& row, const std::vector<T>& col,
                        lapack_int m, lapack_int n, lapack_int ldr,
                        lapack_int ldc) {
  for (lapack_int i = 0; i < m; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      ASSERT_EQ(row[i * ldr + j], At(col, i, j, ldc)) << i << ", " << j;
    }
  }
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_LAPACK_SVD_TEST_UTIL_H_
