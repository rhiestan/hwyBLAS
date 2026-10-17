// Included in all of Eigen's tests when they are compiled with clang-cl: a
// few of them (threads_*) use the POSIX rand_r, which Eigen only replaces
// for MSVC itself.
#ifndef HWYBLAS_TESTS_EIGEN_CLANG_CL_COMPAT_H_
#define HWYBLAS_TESTS_EIGEN_CLANG_CL_COMPAT_H_

#include <stdlib.h>

// rand() of the Microsoft C runtime is already thread-safe.
inline int rand_r(unsigned int* seed) {
  (void)seed;
  return rand();
}

#endif  // HWYBLAS_TESTS_EIGEN_CLANG_CL_COMPAT_H_
