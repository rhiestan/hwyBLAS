// Target-independent helpers for level 2 routines: the kernels work on
// contiguous vectors, so strided BLAS vectors are gathered into scratch
// buffers and scattered back.
#ifndef HWYBLAS_SRC_LEVEL2_H_
#define HWYBLAS_SRC_LEVEL2_H_

#include <stddef.h>

#include "common.h"

namespace hwyblas {

// Returns a contiguous view of the input vector x (n elements, increment inc):
// x itself if inc == 1, otherwise a copy in scratch slot `slot`.
template <typename T>
const T* ContiguousInput(const T* x, size_t n, blasint inc, size_t slot) {
  if (inc == 1) return x;
  const T* start = VectorStart(x, n, inc);
  T* buf = static_cast<T*>(ScratchBuffer(slot, n * sizeof(T)));
  for (size_t i = 0; i < n; ++i)
    buf[i] = start[static_cast<ptrdiff_t>(i) * inc];
  return buf;
}

// Returns a contiguous buffer for the output vector y (n elements, increment
// inc): y itself if inc == 1, otherwise scratch slot `slot`, initialized from
// y only if `read`. Call FinishOutput afterwards.
template <typename T>
T* BeginOutput(T* y, size_t n, blasint inc, size_t slot, bool read) {
  if (inc == 1) return y;
  T* buf = static_cast<T*>(ScratchBuffer(slot, n * sizeof(T)));
  if (read) {
    const T* start = VectorStart(y, n, inc);
    for (size_t i = 0; i < n; ++i) {
      buf[i] = start[static_cast<ptrdiff_t>(i) * inc];
    }
  }
  return buf;
}

// Copies the buffer returned by BeginOutput back into y (if it was a copy).
template <typename T>
void FinishOutput(const T* buf, T* y, size_t n, blasint inc) {
  if (inc == 1) return;
  T* start = VectorStart(y, n, inc);
  for (size_t i = 0; i < n; ++i)
    start[static_cast<ptrdiff_t>(i) * inc] = buf[i];
}

}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LEVEL2_H_
