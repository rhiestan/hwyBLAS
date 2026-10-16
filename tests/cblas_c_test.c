/* Checks that the public headers compile as C and that the CBLAS functions
 * can be called from C: a 2 x 3 row-major matrix times a vector. */
#include <stdio.h>

#include "hwyblas/cblas.h"

int main(void) {
  const double a[6] = {1, 2, 3, 4, 5, 6}; /* row-major [1 2 3; 4 5 6] */
  const double x[3] = {1, 0, -1};
  double y[2] = {0, 0};
  hwyblas_complex_float dot;
  const float cx[2] = {1, 2}, cy[2] = {3, 4};

  cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 3, 1.0, a, 3, x, 1, 0.0, y, 1);
  dot = cblas_cdotu(1, cx, 1, cy, 1); /* (1 + 2i)(3 + 4i) = -5 + 10i */
  if (y[0] != -2 || y[1] != -2 || dot.real != -5 || dot.imag != 10 ||
      cblas_idamax(3, x, 1) != 0) {
    fprintf(stderr, "cblas_c_test failed\n");
    return 1;
  }
  return 0;
}
