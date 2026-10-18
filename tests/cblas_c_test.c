/* Checks that the public headers compile as C and that the CBLAS functions
 * can be called from C: a 2 x 3 row-major matrix times a vector. With
 * HWYBLAS_TEST_LAPACK, also the LAPACK and LAPACKE functions: LU
 * factorizations that interchange the two rows of [0 1; 2 3], and small
 * Cholesky and QR factorizations. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hwyblas/cblas.h"
#if defined(HWYBLAS_TEST_LAPACK)
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"

/* Cholesky and QR factorizations with exactly representable results. */
static int TestLapackFactorizations(void) {
  double s[4] = {4, 2, 2, 5}; /* = L * L**T, L = [2 0; 1 2] */
  double c[2] = {3, 4};       /* 2 x 1 */
  double tau[1];
  lapack_complex_double h[4];
  const double hz[8] = {4, 0, 0, 2, 0, -2, 5, 0}; /* [4 2i; -2i 5] */
  double l[8];
  memcpy(h, hz, sizeof(hz));
  if (LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', 2, s, 2) != 0 || s[0] != 2 ||
      s[1] != 1 || s[3] != 2) {
    return 1;
  }
  /* H = L * L**H with L = [2 0; -i 2]. */
  if (LAPACKE_zpotrf(LAPACK_ROW_MAJOR, 'L', 2, h, 2) != 0) return 1;
  memcpy(l, h, sizeof(l));
  if (l[0] != 2 || l[4] != 0 || l[5] != -1 || l[6] != 2) return 1;
  /* [3; 4] = Q * R with R = -5 and Q(:, 1) = [-0.6; -0.8]. */
  if (LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2, 1, c, 2, tau) != 0 || c[0] != -5) {
    return 1;
  }
  if (LAPACKE_dorgqr(LAPACK_COL_MAJOR, 2, 1, 1, c, 2, tau) != 0) return 1;
  return fabs(c[0] + 0.6) > 1e-15 || fabs(c[1] + 0.8) > 1e-15;
}

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
  if (ipiv[0] != 2 || z[0] != 2 || z[4] != 0) return 1;
  return TestLapackFactorizations();
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
