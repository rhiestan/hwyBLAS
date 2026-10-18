/*
 * Connects LAPACK's own test programs (TESTING/LIN and TESTING/EIG of the
 * reference LAPACK) to hwyBLAS, to run them against the ported routines.
 *
 * The test programs define their own XERBLA, which checks the error exits,
 * and their own ILAENV, which sets the block sizes that they vary. A DLL's
 * xerbla_ and ilaenv_ cannot be replaced at link time, so before main()
 * this installs hwyBLAS hooks that call the program's versions.
 *
 * Usage: link a test program with this file, hwyblas and then the reference
 * LAPACK library, in that order, so that the ported routines replace the
 * reference ones (also where unported reference routines call them):
 *   ... xlintstd objects ... lapack_tester_hooks.obj libtmglib.lib
 *       hwyblas.lib liblapack.lib
 * The tests are not part of the build; see tests/lapack/run_lapack_testers.sh.
 *
 * Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.
 */
#include <string.h>

#include "hwyblas/lapack.h"

/* xerbla_ and ilaenv_ (declared in hwyblas.h and lapack.h) resolve to the
 * test program's definitions, which take precedence over the library's. */

static void CallTesterXerbla(const char* name, hwyblas_int position) {
  xerbla_(name, &position, strlen(name));
}

static void InstallHooks(void) {
  hwyblas_set_error_handler(CallTesterXerbla);
  hwyblas_set_ilaenv(ilaenv_);
}

#if defined(_MSC_VER)
/* The C runtime calls the functions in .CRT$XCU before main(). Not static,
 * so that the compiler keeps the unreferenced pointer. */
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) void (*const hwyblas_tester_hooks)(void) =
    InstallHooks;
#else
__attribute__((constructor)) static void InstallHooksAtStartup(void) {
  InstallHooks();
}
#endif
