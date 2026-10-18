// Shared helpers of the LAPACK tests: block sizes, error capture and the
// test ratios of LAPACK's own test programs.
#ifndef HWYBLAS_TESTS_LAPACK_LAPACK_TEST_UTIL_H_
#define HWYBLAS_TESTS_LAPACK_LAPACK_TEST_UTIL_H_

#include <stddef.h>

#include <algorithm>
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
// exists, so that small matrices exercise the blocked code paths. nb = 0
// keeps ILAENV's choice.
class ScopedBlockSize {
 public:
  explicit ScopedBlockSize(hwyblas_int nb) {
    nb_ = nb;
    previous_ = hwyblas_set_ilaenv(nb > 0 ? &Ilaenv : nullptr);
  }
  ~ScopedBlockSize() {
    hwyblas_set_ilaenv(previous_);
    nb_ = 0;
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
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }

  static inline hwyblas_int nb_ = 0;
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

// The 1-norm (maximum column sum) of a dense matrix.
template <typename T>
double Norm1(const Dense<T>& a) {
  double norm = 0;
  for (size_t j = 0; j < a.cols; ++j) {
    double sum = 0;
    for (size_t i = 0; i < a.rows; ++i) sum += std::abs(a(i, j));
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
    norm = std::max(norm, sum);
  }
  return norm;
}

}  // namespace hwyblas_test

#endif  // HWYBLAS_TESTS_LAPACK_LAPACK_TEST_UTIL_H_
