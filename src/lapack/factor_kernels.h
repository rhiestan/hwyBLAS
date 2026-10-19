// Kernels for the leaves of the recursive LU and Cholesky factorizations
// (factor_kernels.cc).
//
// xGETRF2 and xPOTRF2 recurse down to single columns; hwyBLAS stops the
// recursion at blocks with at most kGetrfLeaf rows or columns (of order at
// most kPotrfLeaf) and factors them with these kernels, the unblocked
// right-looking algorithms of xGETF2 and xPOTF2. This is a documented
// deviation: the results are those of the same algorithm (with the same pivots,
// chosen as by xIAMAX), but they are rounded differently.
#ifndef HWYBLAS_SRC_LAPACK_FACTOR_KERNELS_H_
#define HWYBLAS_SRC_LAPACK_FACTOR_KERNELS_H_

#include <stddef.h>

#include "src/common.h"

namespace hwyblas {

// The largest leaf the kernels accept.
constexpr size_t kFactorLeafMax = 64;

namespace lapack {

// Leaf sizes of xGETRF2 (the smaller dimension) and xPOTRF2 (the order),
// the fastest on Zen 3 (larger leaves lose to the recursion's GEMM and
// HERK updates).
constexpr blasint kGetrfLeaf = 16;
constexpr blasint kPotrfLeaf = 32;
static_assert(kPotrfLeaf <= static_cast<blasint>(kFactorLeafMax));

// LU factorization with partial pivoting of the m x n matrix A, with the
// results and INFO of xGETRF2 (IPIV relative to A, the first zero pivot as
// INFO > 0). m, n >= 1.
template <typename T>
blasint GetrfLeaf(blasint m, blasint n, T* a, blasint lda, blasint* ipiv);

// Cholesky factorization of the triangle (upper or lower) of the n x n
// matrix A, with the results and INFO of xPOTRF2. 1 <= n <= kFactorLeafMax.
template <typename T>
blasint PotrfLeaf(bool upper, blasint n, T* a, blasint lda);

}  // namespace lapack
}  // namespace hwyblas

#endif  // HWYBLAS_SRC_LAPACK_FACTOR_KERNELS_H_
