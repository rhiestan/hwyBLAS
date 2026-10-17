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

// ---------------------------------------------------------------------------
// Calling BLAS routines generically

// Complex arguments are passed as pointers to interleaved reals.
inline float* P(float* p) { return p; }
inline const float* P(const float* p) { return p; }
inline double* P(double* p) { return p; }
inline const double* P(const double* p) { return p; }
inline float* P(std::complex<float>* p) { return reinterpret_cast<float*>(p); }
inline const float* P(const std::complex<float>* p) {
  return reinterpret_cast<const float*>(p);
}
inline double* P(std::complex<double>* p) {
  return reinterpret_cast<double*>(p);
}
inline const double* P(const std::complex<double>* p) {
  return reinterpret_cast<const double*>(p);
}

// Calls s/d/c/z##NAME##_(args...) according to T.
#define HWYBLAS_CALL(T, NAME, ...)                                 \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      s##NAME##_(__VA_ARGS__);                                     \
    } else if constexpr (std::is_same_v<T, double>) {              \
      d##NAME##_(__VA_ARGS__);                                     \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      c##NAME##_(__VA_ARGS__);                                     \
    } else {                                                       \
      z##NAME##_(__VA_ARGS__);                                     \
    }                                                              \
  } while (0)

// result = i##T-prefix##NAME##_(args...), e.g. isamax_ for float.
#define HWYBLAS_CALL_RESULT(T, result, NAME, ...)                  \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      result = is##NAME##_(__VA_ARGS__);                           \
    } else if constexpr (std::is_same_v<T, double>) {              \
      result = id##NAME##_(__VA_ARGS__);                           \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      result = ic##NAME##_(__VA_ARGS__);                           \
    } else {                                                       \
      result = iz##NAME##_(__VA_ARGS__);                           \
    }                                                              \
  } while (0)

// As HWYBLAS_CALL for routines that only exist for complex types (c/z).
#define HWYBLAS_CALL_COMPLEX(T, NAME, ...)                          \
  do {                                                              \
    if constexpr (std::is_same_v<T, std::complex<float>>) {         \
      c##NAME##_(__VA_ARGS__);                                      \
    } else if constexpr (std::is_same_v<T, std::complex<double>>) { \
      z##NAME##_(__VA_ARGS__);                                      \
    }                                                               \
  } while (0)

// As HWYBLAS_CALL for routines that only exist for real types (s/d).
#define HWYBLAS_CALL_REAL(T, NAME, ...)               \
  do {                                                \
    if constexpr (std::is_same_v<T, float>) {         \
      s##NAME##_(__VA_ARGS__);                        \
    } else if constexpr (std::is_same_v<T, double>) { \
      d##NAME##_(__VA_ARGS__);                        \
    }                                                 \
  } while (0)

template <typename T>
T MakeScalar(double re, double im) {
  if constexpr (IsComplex<T>::value) {
    return T(static_cast<RealT<T>>(re), static_cast<RealT<T>>(im));
  } else {
    return static_cast<T>(re);
  }
}

template <typename T>
T NaN() {
  return T(std::numeric_limits<RealT<T>>::quiet_NaN());
}

// Transposition options valid for T: N, T (and C for complex).
template <typename T>
std::vector<char> Ops() {
  if constexpr (IsComplex<T>::value) {
    return {'N', 'T', 'C'};
  } else {
    return {'N', 'T'};
  }
}

// ---------------------------------------------------------------------------
// Dense reference matrices (column-major, in double precision)

template <typename T>
struct Dense {
  size_t rows = 0, cols = 0;
  std::vector<RefT<T>> v;

  Dense(size_t r, size_t c) : rows(r), cols(c), v(r * c) {}
  RefT<T>& operator()(size_t i, size_t j) { return v[i + j * rows]; }
  const RefT<T>& operator()(size_t i, size_t j) const {
    return v[i + j * rows];
  }
};

// Dense copy of op(X) for a general stored matrix X (rows x cols, leading
// dimension ld) and op in {N, T, C}.
template <typename T>
Dense<T> DenseOp(char op, const std::vector<T>& x, size_t rows, size_t cols,
                 size_t ld) {
  const bool trans = (op != 'N');
  Dense<T> d(trans ? cols : rows, trans ? rows : cols);
  for (size_t j = 0; j < d.cols; ++j) {
    for (size_t i = 0; i < d.rows; ++i) {
      d(i, j) = static_cast<RefT<T>>(OpAt(op, x, i, j, ld));
    }
  }
  return d;
}

enum class Shape { kSymmetric, kHermitian, kTriangular, kUnitTriangular };

// Random n x n matrix (leading dimension ld) of the given shape with only
// the `uplo` triangle valid. Everything BLAS must not read is NaN: the other
// triangle, the diagonal of unit triangular matrices and the imaginary parts
// of the diagonal of Hermitian matrices. `offdiag_scale` scales the strictly
// triangular part (to keep triangular solves well conditioned).
template <typename T>
std::vector<T> StructuredMatrix(Shape shape, char uplo, size_t n, size_t ld,
                                Random<T>& rng, double offdiag_scale = 1.0) {
  std::vector<T> a(ld * n, NaN<T>());
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      const bool stored = (uplo == 'L') ? (i > j) : (i < j);
      if (stored) {
        a[i + j * ld] = rng.Next() * static_cast<RealT<T>>(offdiag_scale);
      }
    }
    T& diag = a[j + j * ld];
    switch (shape) {
      case Shape::kUnitTriangular:
        break;  // stays NaN
      case Shape::kTriangular: {
        // |diag| around 1.5 keeps triangular solves well conditioned.
        const double sign = std::real(rng.Next()) >= 0 ? 1.0 : -1.0;
        diag = MakeScalar<T>(1.5 * sign, 0.0) +
               rng.Next() * static_cast<RealT<T>>(0.5);
        break;
      }
      case Shape::kHermitian:
        diag = MakeScalar<T>(std::real(rng.Next()),
                             std::numeric_limits<double>::quiet_NaN());
        break;
      default:
        diag = rng.Next();
    }
  }
  return a;
}

// Dense op(A) for a structured matrix created by StructuredMatrix.
template <typename T>
Dense<T> DenseStructured(Shape shape, char uplo, char op,
                         const std::vector<T>& a, size_t n, size_t ld) {
  Dense<T> full(n, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      const bool stored = (uplo == 'L') ? (i > j) : (i < j);
      RefT<T> v = 0;
      if (i == j) {
        if (shape == Shape::kUnitTriangular) {
          v = 1;
        } else if (shape == Shape::kHermitian) {
          v = std::real(static_cast<RefT<T>>(a[i + i * ld]));
        } else {
          v = static_cast<RefT<T>>(a[i + i * ld]);
        }
      } else if (stored) {
        v = static_cast<RefT<T>>(a[i + j * ld]);
      } else if (shape == Shape::kSymmetric) {
        v = static_cast<RefT<T>>(a[j + i * ld]);
      } else if (shape == Shape::kHermitian) {
        v = Conj(static_cast<RefT<T>>(a[j + i * ld]));
      }
      full(i, j) = v;
    }
  }
  if (op == 'N') return full;
  Dense<T> t(n, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      t(i, j) = (op == 'C') ? Conj(full(j, i)) : full(j, i);
    }
  }
  return t;
}

// Reference C := alpha * A * B + beta * C0 for dense A (m x k) and B (k x n).
// Fills `expected` and the per-element error scale `bound` (sum of absolute
// values of all contributing terms), both with leading dimension ldc.
template <typename T>
void ReferenceProduct(const Dense<T>& a, const Dense<T>& b, T alpha, T beta,
                      const std::vector<T>& c0, size_t ldc,
                      std::vector<RefT<T>>* expected,
                      std::vector<double>* bound) {
  using R = RefT<T>;
  expected->assign(c0.size(), R(0));
  bound->assign(c0.size(), 0.0);
  for (size_t j = 0; j < b.cols; ++j) {
    for (size_t i = 0; i < a.rows; ++i) {
      R sum = 0;
      double abs_sum = 0;
      for (size_t p = 0; p < a.cols; ++p) {
        sum += a(i, p) * b(p, j);
        abs_sum += std::abs(a(i, p)) * std::abs(b(p, j));
      }
      const size_t idx = i + j * ldc;
      R result = static_cast<R>(alpha) * sum;
      double scale = Abs(alpha) * abs_sum;
      if (beta != T(0)) {
        result += static_cast<R>(beta) * static_cast<R>(c0[idx]);
        scale += Abs(beta) * Abs(c0[idx]);
      }
      (*expected)[idx] = result;
      (*bound)[idx] = scale + std::numeric_limits<RealT<T>>::min();
    }
  }
}

// Strided vector with n elements and increment inc (inc < 0: stored
// backwards). Gaps between elements hold a sentinel to detect stray writes.
template <typename T>
struct StridedVector {
  size_t n;
  int inc;
  std::vector<T> data;

  StridedVector(size_t n_, int inc_, Random<T>& rng)
      : n(n_),
        inc(inc_),
        data(n_ == 0 ? 0 : 1 + (n_ - 1) * std::abs(inc_), Sentinel()) {
    for (size_t i = 0; i < n; ++i) (*this)[i] = rng.Next();
  }
  static T Sentinel() { return MakeScalar<T>(7.0, 7.0); }
  T& operator()(size_t i) { return (*this)[i]; }
  T& operator[](size_t i) {
    return data[inc > 0 ? i * inc : (n - 1 - i) * static_cast<size_t>(-inc)];
  }
  // Dense n x 1 copy in double precision.
  Dense<T> ToDense() {
    Dense<T> d(n, 1);
    for (size_t i = 0; i < n; ++i) d(i, 0) = static_cast<RefT<T>>((*this)[i]);
    return d;
  }
  // Checks that only the elements were modified, then returns them.
  std::vector<T> Elements() {
    std::vector<bool> is_element(data.size(), false);
    for (size_t i = 0; i < n; ++i) is_element[&(*this)[i] - data.data()] = true;
    for (size_t i = 0; i < data.size(); ++i) {
      if (!is_element[i]) EXPECT_EQ(data[i], Sentinel()) << "gap " << i;
    }
    std::vector<T> e(n);
    for (size_t i = 0; i < n; ++i) e[i] = (*this)[i];
    return e;
  }
};

// Runs `test` once for every SIMD target that the CPU supports and that the
// library might dispatch to, so all compiled code paths get exercised. (Only
// for the target named by HWYBLAS_TARGET, if that is set.)
template <class Func>
void ForEachTarget(const Func& test) {
  const int64_t targets = hwyblas_available_targets();
  for (int64_t bits = targets; bits != 0; bits &= bits - 1) {
    const int64_t target = bits & -bits;  // lowest set bit
    hwyblas_force_target(target);
    SCOPED_TRACE(hwyblas_target_name());
    test();
  }
  hwyblas_force_target(0);  // restore the default selection
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_TEST_UTIL_H_
