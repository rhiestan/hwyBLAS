# hwyBLAS: a BLAS implementation built on Google Highway.
cmake_minimum_required(VERSION 3.20)
project(hwyblas VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HWYBLAS_BUILD_TESTS "Build the unit tests" ${PROJECT_IS_TOP_LEVEL})
option(HWYBLAS_BUILD_BENCH "Build the benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_SHARED_LIBS "Build hwyblas as a shared library" ON)

# Put all executables and DLLs in one directory so that tests and benchmarks
# find hwyblas.dll on Windows.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# ---------------------------------------------------------------------------
# Highway. Preference order:
#   1. HWYBLAS_HWY_SOURCE_DIR (defaults to a sibling checkout ../highway)
#   2. an installed package (find_package)
#   3. download via FetchContent
# Highway is always linked statically into hwyblas.
set(HWYBLAS_HWY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../highway"
    CACHE PATH "Path to a Highway source checkout")

if(NOT TARGET hwy::hwy)
  if(EXISTS "${HWYBLAS_HWY_SOURCE_DIR}/CMakeLists.txt")
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
    add_subdirectory("${HWYBLAS_HWY_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/highway"
                     EXCLUDE_FROM_ALL)
  else()
    find_package(hwy QUIET)
    if(NOT hwy_FOUND)
      include(FetchContent)
      set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
      set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
      FetchContent_Declare(highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG 1.3.0)
      FetchContent_MakeAvailable(highway)
    endif()
  endif()
endif()
if(TARGET hwy AND NOT TARGET hwy::hwy)
  add_library(hwy::hwy ALIAS hwy)
endif()

# ---------------------------------------------------------------------------
# Compiler flags shared by all hwyBLAS targets.
add_library(hwyblas_options INTERFACE)
if(MSVC)
  # /fp:precise keeps FMA contraction under our control (Highway emits FMA
  # explicitly); /Zc:__cplusplus for correct feature detection.
  target_compile_options(hwyblas_options INTERFACE /W3 /Zc:__cplusplus /fp:precise)
  target_compile_definitions(hwyblas_options INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(hwyblas_options INTERFACE -Wall -Wextra -ffp-contract=off)
endif()

add_subdirectory(src)

if(HWYBLAS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(HWYBLAS_BUILD_BENCH)
  add_subdirectory(bench)
endif()
