// Shared helpers for hwyBLAS tests: random matrices, reference routines and
// error-bound based comparisons.
#ifndef HWYBLAS_TESTS_TEST_UTIL_H_
#define HWYBLAS_TESTS_TEST_UTIL_H_

#include <stddef.h>

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"

namespace hwyblas_test {

template <typename T>
struct IsComplex : std::false_type {};
template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
struct RealOf {
  using type = T;
};
template <typename T>
struct RealOf<std::complex<T>> {
  using type = T;
};
template <typename T>
using RealT = typename RealOf<T>::type;

// Higher precision type used by reference computations.
template <typename T>
using RefT =
    std::conditional_t<IsComplex<T>::value, std::complex<double>, double>;

template <typename T>
T Conj(T x) {
  if constexpr (IsComplex<T>::value) {
    return std::conj(x);
  } else {
    return x;
  }
}

template <typename T>
double Abs(T x) {
  return static_cast<double>(std::abs(x));
}

// Uniform random values in [-1, 1] (both parts for complex).
template <typename T>
class Random {
 public:
  explicit Random(uint64_t seed = 12345) : rng_(seed) {}

  T Next() {
    if constexpr (IsComplex<T>::value) {
      const auto re = static_cast<RealT<T>>(dist_(rng_));
      const auto im = static_cast<RealT<T>>(dist_(rng_));
      return T(re, im);
    } else {
      return static_cast<T>(dist_(rng_));
    }
  }

  // Fills a vector of `size` random elements.
  std::vector<T> Vector(size_t size) {
    std::vector<T> v(size);
    for (T& x : v) x = Next();
    return v;
  }

 private:
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> dist_{-1.0, 1.0};
};

// Column-major matrix element accessor.
template <typename T>
T& At(std::vector<T>& v, size_t i, size_t j, size_t ld) {
  return v[i + j * ld];
}
template <typename T>
const T& At(const std::vector<T>& v, size_t i, size_t j, size_t ld) {
  return v[i + j * ld];
}

// Returns element (i, j) of op(X) where X is column-major with leading
// dimension ld and op is 'N', 'T' or 'C'.
template <typename T>
T OpAt(char op, const std::vector<T>& x, size_t i, size_t j, size_t ld) {
  switch (op) {
    case 'N':
      return At(x, i, j, ld);
    case 'T':
      return At(x, j, i, ld);
    default:
      return Conj(At(x, j, i, ld));
  }
}

// Unit roundoff of T.
template <typename T>
double Eps() {
  return static_cast<double>(std::numeric_limits<RealT<T>>::epsilon());
}

// Checks |actual - expected| <= tol * bound element-wise, where bound is a
// per-element error scale (e.g. the sum of absolute values of all terms that
// contributed). Reports at most a few mismatches.
template <typename T>
::testing::AssertionResult NearWithBound(const std::vector<T>& actual,
                                         const std::vector<RefT<T>>& expected,
                                         const std::vector<double>& bound,
                                         double tol, size_t m, size_t n,
                                         size_t ld) {
  size_t failures = 0;
  std::string msg;
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      const size_t idx = i + j * ld;
      const RefT<T> got = static_cast<RefT<T>>(actual[idx]);
      const double err = std::abs(got - expected[idx]);
      // `!(a <= b)` also catches NaN.
      if (!(err <= tol * bound[idx])) {
        if (++failures <= 5) {
          msg += "\n  (" + std::to_string(i) + ", " + std::to_string(j) +
                 "): err " + std::to_string(err) + " > " +
                 std::to_string(tol * bound[idx]);
        }
      }
    }
  }
  if (failures == 0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << failures << " mismatching elements" << msg;
}

// Runs `test` once for every SIMD target that the CPU supports and that the
// library might dispatch to, so all compiled code paths get exercised.
template <class Func>
void ForEachTarget(const Func& test) {
  const int64_t targets = hwyblas_available_targets();
  for (int64_t bits = targets; bits != 0; bits &= bits - 1) {
    const int64_t target = bits & -bits;  // lowest set bit
    hwyblas_force_target(target);
    SCOPED_TRACE(hwyblas_target_name());
    test();
  }
  hwyblas_force_target(0);  // restore automatic selection
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_TEST_UTIL_H_
