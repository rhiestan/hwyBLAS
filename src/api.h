// Helpers for the Fortran-style API layer: argument decoding, complex pointer
// casts and dispatch to per-target implementations.
//
// Include only in HWY_ONCE sections (or files not using foreach_target).
#ifndef HWYBLAS_SRC_API_H_
#define HWYBLAS_SRC_API_H_

#include <complex>
#include <type_traits>

#include "common.h"
#include "gemm.h"
#include "hwy/highway.h"

namespace hwyblas {

// ---------------------------------------------------------------------------
// Complex arguments arrive as pointers to interleaved (re, im) reals.

inline const c32* C32(const float* p) {
  return reinterpret_cast<const c32*>(p);
}
inline c32* C32(float* p) { return reinterpret_cast<c32*>(p); }
inline const c64* C64(const double* p) {
  return reinterpret_cast<const c64*>(p);
}
inline c64* C64(double* p) { return reinterpret_cast<c64*>(p); }

// ---------------------------------------------------------------------------
// Per-target functions are defined for each precision with suffixes
// F32, F64, C32 and C64; these macros export and call them.

#define HWYBLAS_EXPORT_BY_TYPE(NAME) \
  HWY_EXPORT(NAME##F32);             \
  HWY_EXPORT(NAME##F64);             \
  HWY_EXPORT(NAME##C32);             \
  HWY_EXPORT(NAME##C64)

// Calls the per-target function NAME##<suffix of T>(args...).
#define HWYBLAS_DISPATCH_BY_TYPE(T, NAME, ...)                      \
  do {                                                              \
    if constexpr (std::is_same_v<T, float>) {                       \
      HWY_DYNAMIC_DISPATCH(NAME##F32)(__VA_ARGS__);                 \
    } else if constexpr (std::is_same_v<T, double>) {               \
      HWY_DYNAMIC_DISPATCH(NAME##F64)(__VA_ARGS__);                 \
    } else if constexpr (std::is_same_v<T, ::hwyblas::c32>) {       \
      HWY_DYNAMIC_DISPATCH(NAME##C32)(__VA_ARGS__);                 \
    } else {                                                        \
      static_assert(std::is_same_v<T, ::hwyblas::c64>, "Bad type"); \
      HWY_DYNAMIC_DISPATCH(NAME##C64)(__VA_ARGS__);                 \
    }                                                               \
  } while (0)

// ---------------------------------------------------------------------------
// Decoding of CHARACTER*1 arguments. Each returns false for invalid values.

inline bool DecodeUplo(char c, bool* lower) {
  switch (ToUpper(c)) {
    case 'L':
      *lower = true;
      return true;
    case 'U':
      *lower = false;
      return true;
    default:
      return false;
  }
}

inline bool DecodeSide(char c, bool* left) {
  switch (ToUpper(c)) {
    case 'L':
      *left = true;
      return true;
    case 'R':
      *left = false;
      return true;
    default:
      return false;
  }
}

inline bool DecodeDiag(char c, bool* unit) {
  switch (ToUpper(c)) {
    case 'U':
      *unit = true;
      return true;
    case 'N':
      *unit = false;
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Operands

// op(X) for a general column-major matrix X with leading dimension ld.
template <typename T>
Operand<T> GeneralOperand(Op op, const T* x, blasint ld) {
  Operand<T> o;
  o.data = x;
  // op(X)(i, p) is X(i, p) = x[i + p * ld], or X(p, i) = x[p + i * ld].
  o.rs = (op == Op::kNoTrans) ? 1 : ld;
  o.cs = (op == Op::kNoTrans) ? ld : 1;
  o.conj = (op == Op::kConjTrans);
  return o;
}

// op(X) for a structured (symmetric, Hermitian or triangular) column-major
// matrix whose `lower` or upper triangle is stored. Transposing op(X) swaps
// which triangle is stored.
template <typename T>
Operand<T> StructuredOperand(Structure structure, bool lower, Op op, const T* x,
                             blasint ld, bool unit_diag = false) {
  Operand<T> o = GeneralOperand(op, x, ld);
  o.structure = structure;
  o.lower = (op == Op::kNoTrans) ? lower : !lower;
  o.unit_diag = unit_diag;
  return o;
}

}  // namespace hwyblas

#endif  // HWYBLAS_SRC_API_H_
