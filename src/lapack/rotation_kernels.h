// Kernel for the sequences of plane rotations of xLASR (rotation_kernels.cc).
#ifndef HWYBLAS_SRC_LAPACK_ROTATION_KERNELS_H_
#define HWYBLAS_SRC_LAPACK_ROTATION_KERNELS_H_

#include "src/common.h"

namespace hwyblas {
namespace lapack {

// xLASR with PIVOT = 'V' (the only pivot of xBDSQR, xSTEQR and xBDSDC):
// applies the rotations P(k) = [ c(k) s(k); -s(k) c(k) ] in the planes
// (k, k+1) to the M-by-N matrix A, from the left (left: A := P*A) or the
// right (A := A*P**T), with P = P(z-1) . . . P(1) (forward) or
// P(1) . . . P(z-1), z = M (left) or N. Each element of A undergoes the
// same operations in the same order as in LAPACK (identity rotations are
// skipped), so the result is identical. m, n >= 1, lda >= m.
template <typename T>
void LasrVariable(bool left, bool forward, blasint m, blasint n,
                  const RealT<T>* c, const RealT<T>* s, T* a, blasint lda);

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_ROTATION_KERNELS_H_
