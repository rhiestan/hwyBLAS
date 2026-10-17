/*
 * Error-exit hook for the Netlib Fortran BLAS testers (the versions in
 * OpenBLAS's test/ directory).
 *
 * The Level 2 and 3 testers define their own XERBLA, which checks the
 * routine name and argument position of each error exit, and call
 * OPENBLAS_TEST_SET_XERBLA before testing them. A DLL's xerbla_ cannot be
 * replaced, so this installs a handler with hwyblas_set_error_handler that
 * calls the tester's XERBLA instead.
 *
 * It assumes the usual Fortran conventions of gfortran and flang: lowercase
 * names with a trailing underscore, and the lengths of character arguments
 * passed as size_t after all other arguments.
 *
 * The testers are not part of the build. To run one by hand, e.g. with
 * gfortran (or flang, as in Visual Studio 2026):
 *   gfortran OpenBLAS/test/dblat2.f fortran_xerbla_hook.c -Iinclude \
 *            -Lbuild/lib -lhwyblas -o dblat2
 *   ./dblat2 < OpenBLAS/test/dblat2.dat     # report: DBLAT2.SUMM
 * The Level 1 testers need no hook and read no input. OpenBLAS's input
 * files for the complex Level 3 testers switch off the error-exit tests;
 * set "T TO TEST ERROR EXITS" to T to run them too.
 *
 * Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.
 */
#include <string.h>

#include "hwyblas/hwyblas.h"

static void CallTesterXerbla(const char* name, hwyblas_int position) {
  char srname[6]; /* CHARACTER*6, blank-padded */
  const size_t len = strlen(name);
  memset(srname, ' ', sizeof(srname));
  memcpy(srname, name, len < sizeof(srname) ? len : sizeof(srname));
  xerbla_(srname, &position, sizeof(srname));
}

void openblas_test_set_xerbla_(void) {
  hwyblas_set_error_handler(CallTesterXerbla);
}
