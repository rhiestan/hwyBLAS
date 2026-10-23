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
option(HWYBLAS_INSTALL "Generate the install target" ${PROJECT_IS_TOP_LEVEL})
option(HWYBLAS_LAPACK "Include the LAPACK routines (and LAPACKE)" ON)
option(HWYBLAS_CLANG_TIDY "Run clang-tidy (checks in .clang-tidy) while compiling"
       OFF)

include(GNUInstallDirs)

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
        GIT_TAG 1.4.0)
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
else()
  target_compile_options(hwyblas_options INTERFACE -Wall -Wextra -ffp-contract=off)
endif()
if(WIN32)
  target_compile_definitions(hwyblas_options INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# clang-tidy runs on hwyBLAS's own C++ sources, not on Highway (added above)
# or GoogleTest. It is slow: Highway compiles each source once per target.
if(HWYBLAS_CLANG_TIDY)
  find_program(HWYBLAS_CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
  set(CMAKE_CXX_CLANG_TIDY "${HWYBLAS_CLANG_TIDY_EXE}")
endif()

add_subdirectory(src)

if(HWYBLAS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(HWYBLAS_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# Installation: the library, its headers, a CMake package (find_package(
# hwyblas), target hwyblas::hwyblas) and a pkg-config file (hwyblas.pc).
if(HWYBLAS_INSTALL)
  include(CMakePackageConfigHelpers)
  set(HWYBLAS_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/hwyblas")

  install(TARGETS hwyblas EXPORT hwyblasTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(DIRECTORY include/hwyblas
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/hwyblas")
  if(HWYBLAS_LAPACK)
    install(FILES LICENSE-LAPACK
            DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/hwyblas")
  endif()
  # Highway built here is part of the installed library, so its license is
  # installed too (an installed Highway package comes with its own).
  get_target_property(_hwy_imported hwy::hwy IMPORTED)
  if(NOT _hwy_imported)
    get_target_property(_hwy_source_dir hwy::hwy SOURCE_DIR)
    install(FILES "${_hwy_source_dir}/LICENSE" RENAME LICENSE-highway
            DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/hwyblas")
  endif()

  # A shared hwyblas contains Highway. A static one needs it at link time:
  # either the installed Highway package it was built against, or the
  # Highway built here, installed as hwyblas_hwy so that it cannot clash
  # with another Highway installation.
  set(HWYBLAS_FIND_HWY_PACKAGE FALSE)
  set(HWYBLAS_PC_LIBS_PRIVATE "")
  if(NOT BUILD_SHARED_LIBS)
    if(_hwy_imported)
      set(HWYBLAS_FIND_HWY_PACKAGE TRUE)
      set(HWYBLAS_PC_LIBS_PRIVATE "-lhwy")
    else()
      get_target_property(_hwy_target hwy::hwy ALIASED_TARGET)
      if(NOT _hwy_target)
        set(_hwy_target hwy)
      endif()
      set_target_properties(${_hwy_target} PROPERTIES
        OUTPUT_NAME hwyblas_hwy EXPORT_NAME hwy)
      install(TARGETS ${_hwy_target} EXPORT hwyblasTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
      set(HWYBLAS_PC_LIBS_PRIVATE "-lhwyblas_hwy")
    endif()
    if(NOT MSVC)
      string(APPEND HWYBLAS_PC_LIBS_PRIVATE " -lstdc++ -lm")
    endif()
  endif()

  install(EXPORT hwyblasTargets NAMESPACE hwyblas::
          DESTINATION "${HWYBLAS_CMAKE_DIR}")
  configure_package_config_file(cmake/hwyblasConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/hwyblasConfig.cmake"
    INSTALL_DESTINATION "${HWYBLAS_CMAKE_DIR}")
  # 0.x: minor versions may be incompatible.
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hwyblasConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hwyblasConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/hwyblasConfigVersion.cmake"
          DESTINATION "${HWYBLAS_CMAKE_DIR}")
  configure_file(cmake/hwyblas.pc.in "${CMAKE_CURRENT_BINARY_DIR}/hwyblas.pc"
                 @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hwyblas.pc"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif()
