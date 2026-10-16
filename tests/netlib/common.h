/*
 * Stand-in for OpenBLAS's internal common.h, which the C versions of the
 * Netlib CBLAS testers in OpenBLAS's ctest/ directory include.
 *
 * Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.
 */
#ifndef HWYBLAS_TESTS_NETLIB_COMMON_H_
#define HWYBLAS_TESTS_NETLIB_COMMON_H_

#include "hwyblas/cblas.h"

typedef hwyblas_int blasint;
#define OPENBLAS_CONST const

#endif /* HWYBLAS_TESTS_NETLIB_COMMON_H_ */
