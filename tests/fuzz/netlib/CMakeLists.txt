# The Netlib reference BLAS and CBLAS of LAPACK as one shared library, a
# reference for the fuzzer (blas_fuzz --netlib), built by hand:
#
#   cmake -S tests/fuzz/netlib -B <build dir> -DLAPACK_SOURCE_DIR=<lapack>
#   cmake --build <build dir>
#   blas_fuzz --netlib Netlib=<build dir>/libnetlibblas.so
#
# Needs a Fortran compiler. Built without floating-point contraction, as
# hwyBLAS is.
cmake_minimum_required(VERSION 3.20)
project(netlibblas C Fortran)

set(LAPACK_SOURCE_DIR "" CACHE PATH "LAPACK source tree (3.12.1)")
if(NOT EXISTS "${LAPACK_SOURCE_DIR}/BLAS/SRC/dgemm.f")
  message(FATAL_ERROR "Set LAPACK_SOURCE_DIR to a LAPACK source tree")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FortranCInterface)
FortranCInterface_HEADER("${CMAKE_BINARY_DIR}/include/cblas_mangling.h"
                         MACRO_NAMESPACE "F77_" SYMBOL_NAMESPACE "F77_")
file(GLOB blas "${LAPACK_SOURCE_DIR}/BLAS/SRC/*.f"
               "${LAPACK_SOURCE_DIR}/BLAS/SRC/*.f90")
file(GLOB cblas "${LAPACK_SOURCE_DIR}/CBLAS/src/*.c"
                "${LAPACK_SOURCE_DIR}/CBLAS/src/*.f")
# CBLAS brings its own xerbla_.
list(REMOVE_ITEM blas "${LAPACK_SOURCE_DIR}/BLAS/SRC/xerbla.f")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(netlibblas SHARED ${blas} ${cblas})
target_include_directories(netlibblas PRIVATE
  "${LAPACK_SOURCE_DIR}/CBLAS/include" "${CMAKE_BINARY_DIR}/include")
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU|Flang")
  target_compile_options(netlibblas PRIVATE
    "$<$<COMPILE_LANGUAGE:Fortran>:-ffp-contract=off>")
endif()
