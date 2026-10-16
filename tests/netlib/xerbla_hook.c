/*
 * Replacement for c_xerbla.c of the Netlib CBLAS testers.
 *
 * The error-exit tests (c_?2chke.c, c_?3chke.c) set the expected routine
 * name and argument position in cblas_rout and cblas_info, call a routine
 * with that argument invalid, and expect the library to report it. The
 * original c_xerbla.c replaces cblas_xerbla, which a DLL does not allow;
 * here the check runs in a handler installed with hwyblas_set_error_handler.
 *
 * hwyBLAS reports the position in the CBLAS argument list, for row-major
 * calls too, which is what cblas_info holds. (The reference CBLAS reports
 * row-major errors at the positions of the transposed Fortran call, which
 * the original c_xerbla.c maps back.)
 *
 * Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cblas_test.h"
#include "common.h"

extern int cblas_ok, cblas_lerr, cblas_info;
extern char* cblas_rout;

static void CheckError(const char* name, hwyblas_int position) {
  if (cblas_rout != NULL && strcmp(cblas_rout, name) != 0) {
    printf(
        "***** XERBLA WAS CALLED WITH SRNAME = <%s> INSTEAD OF <%s> *******\n",
        name, cblas_rout);
    cblas_ok = FALSE;
  }
  if (position != cblas_info) {
    printf(
        "***** XERBLA WAS CALLED WITH INFO = %d INSTEAD OF %d in %s *******\n",
        (int)position, cblas_info, name);
    cblas_lerr = PASSED;
    cblas_ok = FALSE;
  } else {
    cblas_lerr = FAILED;  // the error was detected (chkxer checks this)
  }
}

void cblas_test_set_xerbla(void) { hwyblas_set_error_handler(CheckError); }

void cblas_test_fail(void) { exit(EXIT_FAILURE); }
