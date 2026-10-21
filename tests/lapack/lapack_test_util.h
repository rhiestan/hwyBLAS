// Shared helpers of the LAPACK tests: block sizes, error capture and the
// test ratios of LAPACK's own test programs.
#ifndef HWYBLAS_TESTS_LAPACK_LAPACK_TEST_UTIL_H_
#define HWYBLAS_TESTS_LAPACK_LAPACK_TEST_UTIL_H_

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "test_util.h"

namespace hwyblas_test {

// LAPACK's test programs accept a test ratio below this threshold (the
// value of their input files).
constexpr double kLapackThreshold = 30.0;

// Forces the block size (ILAENV ispec = 1) of all LAPACK routines while it
// exists, and the crossover point nx (ispec = 3) below which the blocked
// routines use unblocked code, so that small matrices exercise the blocked
// code paths. nb = 0 keeps ILAENV's choices.
class ScopedBlockSize {
 public:
  explicit ScopedBlockSize(hwyblas_int nb, hwyblas_int nx = 0) {
    nb_ = nb;
    nx_ = nx;
    previous_ = hwyblas_set_ilaenv(nb > 0 ? &Ilaenv : nullptr);
  }
  ~ScopedBlockSize() {
    hwyblas_set_ilaenv(previous_);
    nb_ = 0;
    nx_ = 0;
  }
  ScopedBlockSize(const ScopedBlockSize&) = delete;
  ScopedBlockSize& operator=(const ScopedBlockSize&) = delete;

 private:
  static hwyblas_int Ilaenv(const hwyblas_int* ispec, const char* name,
                            const char* opts, const hwyblas_int* n1,
                            const hwyblas_int* n2, const hwyblas_int* n3,
                            const hwyblas_int* n4, size_t name_len,
                            size_t opts_len) {
    if (*ispec == 1) return nb_;
    if (*ispec == 3) return nx_;
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }

  static inline hwyblas_int nb_ = 0;
  static inline hwyblas_int nx_ = 0;
  hwyblas_ilaenv_function previous_ = nullptr;
};

// Forces the maximum size SMLSIZ of the subproblems of divide and conquer
// (ILAENV ispec = 9) while it exists, so that small matrices are split into
// several levels.
class ScopedSubproblemSize {
 public:
  explicit ScopedSubproblemSize(hwyblas_int smlsiz) {
    smlsiz_ = smlsiz;
    previous_ = hwyblas_set_ilaenv(smlsiz > 0 ? &Ilaenv : nullptr);
  }
  ~ScopedSubproblemSize() {
    hwyblas_set_ilaenv(previous_);
    smlsiz_ = 0;
  }
  ScopedSubproblemSize(const ScopedSubproblemSize&) = delete;
  ScopedSubproblemSize& operator=(const ScopedSubproblemSize&) = delete;

 private:
  static hwyblas_int Ilaenv(const hwyblas_int* ispec, const char* name,
                            const char* opts, const hwyblas_int* n1,
                            const hwyblas_int* n2, const hwyblas_int* n3,
                            const hwyblas_int* n4, size_t name_len,
                            size_t opts_len) {
    if (*ispec == 9) return smlsiz_;
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }

  static inline hwyblas_int smlsiz_ = 0;
  hwyblas_ilaenv_function previous_ = nullptr;
};

// Block sizes to test: ILAENV's own (0), unblocked (1), and small ones that
// split small matrices into several blocks.
inline const std::vector<hwyblas_int>& BlockSizes() {
  static const std::vector<hwyblas_int> sizes = {0, 1, 2, 3, 8};
  return sizes;
}

// Records the errors reported while it exists ("DGETRF:4" for argument 4 of
// DGETRF), instead of printing them.
class ErrorCapture {
 public:
  ErrorCapture() : previous_(hwyblas_set_error_handler(&Record)) {
    active_ = this;
  }
  ~ErrorCapture() {
    hwyblas_set_error_handler(previous_);
    active_ = nullptr;
  }
  ErrorCapture(const ErrorCapture&) = delete;
  ErrorCapture& operator=(const ErrorCapture&) = delete;

  // The errors reported so far, then forgets them.
  std::vector<std::string> Take() {
    std::vector<std::string> e;
    e.swap(errors_);
    return e;
  }

 private:
  static void Record(const char* name, hwyblas_int position) {
    active_->errors_.push_back(std::string(name) + ":" +
                               std::to_string(position));
  }

  static inline ErrorCapture* active_ = nullptr;
  hwyblas_error_handler previous_;
  std::vector<std::string> errors_;
};

// The routine name of T's precision, e.g. Name<double>("GETRF") = "DGETRF".
template <typename T>
std::string Name(const char* rest) {
  const char prefix = std::is_same_v<T, float>                 ? 'S'
                      : std::is_same_v<T, double>              ? 'D'
                      : std::is_same_v<T, std::complex<float>> ? 'C'
                                                               : 'Z';
  return std::string(1, prefix) + rest;
}

// The name of a LAPACKE function in its error messages, e.g.
// LapackeName<double>("getrf_work") = "LAPACKE_dgetrf_work".
template <typename T>
std::string LapackeName(const char* rest) {
  std::string name = Name<T>(rest);
  name[0] = static_cast<char>(name[0] - 'A' + 'a');
  return "LAPACKE_" + name;
}

// The 1-norm (maximum column sum) of a dense matrix; NaN if an element is
// NaN (std::max would drop it, and a NaN result would pass the ratio
// checks).
template <typename T>
double Norm1(const Dense<T>& a) {
  double norm = 0;
  for (size_t j = 0; j < a.cols; ++j) {
    double sum = 0;
    for (size_t i = 0; i < a.rows; ++i) sum += std::abs(a(i, j));
    if (std::isnan(sum)) return sum;
    norm = std::max(norm, sum);
  }
  return norm;
}

// The 1-norm of the m x n matrix x (leading dimension ld).
template <typename T>
double Norm1(const std::vector<T>& x, size_t m, size_t n, size_t ld) {
  double norm = 0;
  for (size_t j = 0; j < n; ++j) {
    double sum = 0;
    for (size_t i = 0; i < m; ++i) sum += Abs(At(x, i, j, ld));
    if (std::isnan(sum)) return sum;
    norm = std::max(norm, sum);
  }
  return norm;
}

// LAPACK's eps (xLAMCH('Epsilon')): the unit roundoff, half the machine
// epsilon.
template <typename T>
double LapackEps() {
  return Eps<T>() / 2;
}

// The m x n matrix a (leading dimension lda) in higher precision.
template <typename T>
Dense<T> ToDense(const std::vector<T>& a, size_t m, size_t n, size_t lda) {
  Dense<T> d(m, n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i)
      d(i, j) = static_cast<RefT<T>>(At(a, i, j, lda));
  }
  return d;
}

template <typename T>
Dense<T> Identity(size_t n) {
  Dense<T> d(n, n);
  for (size_t i = 0; i < n; ++i) d(i, i) = 1;
  return d;
}

// a * b.
template <typename T>
Dense<T> Mul(const Dense<T>& a, const Dense<T>& b) {
  Dense<T> c(a.rows, b.cols);
  for (size_t j = 0; j < b.cols; ++j) {
    for (size_t l = 0; l < a.cols; ++l) {
      const RefT<T> blj = b(l, j);
      for (size_t i = 0; i < a.rows; ++i) c(i, j) += a(i, l) * blj;
    }
  }
  return c;
}

// a**H (a**T for real T).
template <typename T>
Dense<T> Adjoint(const Dense<T>& a) {
  Dense<T> t(a.cols, a.rows);
  for (size_t j = 0; j < a.cols; ++j) {
    for (size_t i = 0; i < a.rows; ++i) t(j, i) = Conj(a(i, j));
  }
  return t;
}

// a - b.
template <typename T>
Dense<T> Sub(const Dense<T>& a, const Dense<T>& b) {
  Dense<T> d(a.rows, a.cols);
  for (size_t i = 0; i < d.v.size(); ++i) d.v[i] = a.v[i] - b.v[i];
  return d;
}

// ||I - Q**H * Q||_1 / (n * eps) for the m x n matrix q: the orthogonality
// ratio of LAPACK's test programs (e.g. xQRT01).
template <typename T>
double OrthogonalityRatio(const Dense<T>& q) {
  const double resid = Norm1(Sub<T>(Identity<T>(q.cols), Mul(Adjoint(q), q)));
  return resid / std::max<double>(1, static_cast<double>(q.rows)) /
         LapackEps<T>();
}

// ||a - b||_1 / (scale * ||a||_1 * eps), or the residual ratio when a = 0.
template <typename T>
double ResidualRatio(const Dense<T>& a, const Dense<T>& b, double scale) {
  const double anorm = Norm1(a);
  const double resid = Norm1(Sub(a, b));
  const double eps = LapackEps<T>();
  if (anorm <= 0) return resid / eps;
  return resid / std::max(1.0, scale) / anorm / eps;
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_LAPACK_LAPACK_TEST_UTIL_H_
