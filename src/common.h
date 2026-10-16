// Target-independent helpers shared by all hwyBLAS routines.
#ifndef HWYBLAS_SRC_COMMON_H_
#define HWYBLAS_SRC_COMMON_H_

#include <stddef.h>

#include <complex>
#include <type_traits>

#include "hwyblas/hwyblas.h"

namespace hwyblas {

using blasint = hwyblas_int;
using c32 = std::complex<float>;
using c64 = std::complex<double>;

// ---------------------------------------------------------------------------
// Scalar type traits

template <typename T>
struct IsComplex : std::false_type {};
template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

// Underlying real type: float for float and std::complex<float>, etc.
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

// Number of reals in n elements of T (2n for complex T).
template <typename T>
constexpr size_t RealCount(size_t n) {
  return IsComplex<T>::value ? 2 * n : n;
}

template <typename T>
inline T Conj(T x) {
  if constexpr (IsComplex<T>::value) {
    return std::conj(x);
  } else {
    return x;
  }
}

// Exact comparisons against 0 and 1 are intended here: BLAS semantics treat
// alpha == 0 and beta == 0/1 as special cases (e.g. beta == 0 means C is not
// read, so NaNs in C do not propagate).
template <typename T>
inline bool IsZero(T x) {
  return x == T(0);
}
template <typename T>
inline bool IsOne(T x) {
  return x == T(1);
}

// ---------------------------------------------------------------------------
// Argument decoding (Fortran CHARACTER*1 arguments)

inline char ToUpper(char c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }

// op(X) for GEMM-style arguments.
enum class Op { kNoTrans, kTrans, kConjTrans };

// Returns false if `c` is not one of N, T, C (case-insensitive).
inline bool DecodeOp(char c, Op* op) {
  switch (ToUpper(c)) {
    case 'N':
      *op = Op::kNoTrans;
      return true;
    case 'T':
      *op = Op::kTrans;
      return true;
    case 'C':
      *op = Op::kConjTrans;
      return true;
    default:
      return false;
  }
}

// Reports an invalid argument via xerbla_ (`info` is the 1-based position).
void ReportError(const char* routine, blasint info);

// ---------------------------------------------------------------------------
// Scratch memory

// Returns a 64-byte aligned buffer of at least `bytes` bytes. Each `slot`
// is an independent, thread-local buffer that is reused across calls and only
// grows. The pointer remains valid until the next call with the same slot on
// the same thread.
void* ScratchBuffer(size_t slot, size_t bytes);

// Scratch slots. Routines never call each other while holding a buffer, so
// different routines may share slot numbers.
enum : size_t {
  kScratchPackA = 0,  // GEMM: packed block of A
  kScratchPackB = 1,  // GEMM: packed block of B
  kScratchX = 0,      // level 2: contiguous copy of x
  kScratchY = 1,      // level 2: contiguous copy of y
  kScratchTemp = 2,   // level 2: kernel temporaries
  kNumScratchSlots = 3
};

// TRSM recursion stops at triangles of at most this many rows, which are
// solved by a substitution kernel.
constexpr size_t kTrsmBase = 32;

// Returns a pointer to element 0 of a BLAS vector with `n` elements and
// increment `inc` (element i is at start + i * inc). For negative increments
// BLAS stores the vector backwards, starting at x[(1 - n) * inc].
template <typename T>
T* VectorStart(T* x, size_t n, blasint inc) {
  return inc >= 0 ? x : x - static_cast<ptrdiff_t>(n - 1) * inc;
}

}  // namespace hwyblas

#endif  // HWYBLAS_SRC_COMMON_H_
