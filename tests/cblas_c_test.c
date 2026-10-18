/* Checks that the public headers compile as C and that the CBLAS functions
 * can be called from C: a 2 x 3 row-major matrix times a vector. With
 * HWYBLAS_TEST_LAPACK, also the LAPACK and LAPACKE functions: LU
 * factorizations that interchange the two rows of [0 1; 2 3]. */
#include <stdio.h>
#include <string.h>

#include "hwyblas/cblas.h"
#if defined(HWYBLAS_TEST_LAPACK)
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"

static int TestLapack(void) {
  double a[4] = {0, 1, 2, 3}; /* row-major */
  double b[4] = {0, 2, 1, 3}; /* the same matrix, column-major */
  double z[8] = {0, 0, 1, 0, 2, 0, 3, 0};
  lapack_complex_double zc[4];
  lapack_int ipiv[2], info;
  const lapack_int n = 2;
  memcpy(zc, z, sizeof(z)); /* row-major, complex */
  dgetrf_(&n, &n, b, &n, ipiv, &info);
  if (info != 0 || ipiv[0] != 2 || b[0] != 2 || b[1] != 0) return 1;
  if (LAPACKE_dgetrf(LAPACK_ROW_MAJOR, 2, 2, a, 2, ipiv) != 0 || ipiv[0] != 2 ||
      a[0] != 2 || a[2] != 0) {
    return 1;
  }
  if (LAPACKE_zgetrf(LAPACK_ROW_MAJOR, 2, 2, zc, 2, ipiv) != 0) return 1;
  memcpy(z, zc, sizeof(z));
  return ipiv[0] != 2 || z[0] != 2 || z[4] != 0;
}
#else
static int TestLapack(void) { return 0; }
#endif

int main(void) {
  const double a[6] = {1, 2, 3, 4, 5, 6}; /* row-major [1 2 3; 4 5 6] */
  const double x[3] = {1, 0, -1};
  double y[2] = {0, 0};
  hwyblas_complex_float dot;
  const float cx[2] = {1, 2}, cy[2] = {3, 4};

  cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 3, 1.0, a, 3, x, 1, 0.0, y, 1);
  dot = cblas_cdotu(1, cx, 1, cy, 1); /* (1 + 2i)(3 + 4i) = -5 + 10i */
  if (y[0] != -2 || y[1] != -2 || dot.real != -5 || dot.imag != 10 ||
      cblas_idamax(3, x, 1) != 0 || TestLapack() != 0) {
    fprintf(stderr, "cblas_c_test failed\n");
    return 1;
  }
  return 0;
}
