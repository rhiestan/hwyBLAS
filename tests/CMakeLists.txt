include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# Always link GoogleTest statically, independent of BUILD_SHARED_LIBS.
set(_hwyblas_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(googletest)
set(BUILD_SHARED_LIBS ${_hwyblas_shared})
include(GoogleTest)

# ---------------------------------------------------------------------------
# Per-target test runs. Most unit tests already loop over all targets that
# the CPU supports (ForEachTarget), but the CBLAS, Eigen and Netlib tests only
# use the best one. So each test program also runs once per target in
# HWYBLAS_TEST_TARGETS, with the environment variable HWYBLAS_TARGET set (the
# ForEachTarget loops then only use that target). The tests are named
# <target>.<program>[.<shard>] and labeled target_<target>, e.g.
# ctest -L target_SSE4. Targets that the CPU cannot run are skipped, unless
# Intel SDE (the Software Development Emulator) is given in HWYBLAS_SDE: the
# AVX-512 targets then run in SDE, emulating HWYBLAS_SDE_CPU.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|x86|i[3-6]86)$")
  set(_targets "SSE2;SSE4;AVX2;AVX3;AVX3_ZEN4")
else()
  set(_targets "")
endif()
set(HWYBLAS_TEST_TARGETS "${_targets}" CACHE STRING
    "SIMD targets to run each test program on (HWYBLAS_TARGET names)")
set(HWYBLAS_SDE "" CACHE FILEPATH
    "Intel SDE executable, to run the AVX-512 tests on other CPUs")
set(HWYBLAS_SDE_CPU "-spr" CACHE STRING
    "SDE option selecting the emulated CPU (Sapphire Rapids: all of AVX-512)")
set(HWYBLAS_SDE_SHARDS 16 CACHE STRING
    "Number of parallel parts of hwyblas_tests in SDE")

# Sets `out` to the command prefix that runs a program for `target`.
function(hwyblas_target_launcher target out)
  if(HWYBLAS_SDE AND target MATCHES "^AVX3")
    set(${out} "${HWYBLAS_SDE}" ${HWYBLAS_SDE_CPU} -- PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction()

# Adds the test `name` running COMMAND for `target`, optionally split into
# SHARDS GoogleTest shards. ENVIRONMENT adds environment variables.
function(hwyblas_add_target_test target name)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "SHARDS" "COMMAND;ENVIRONMENT")
  if(NOT arg_SHARDS)
    set(arg_SHARDS 1)
  endif()
  math(EXPR _last "${arg_SHARDS} - 1")
  foreach(shard RANGE ${_last})
    set(_name ${target}.${name})
    set(_env HWYBLAS_TARGET=${target} ${arg_ENVIRONMENT})
    if(arg_SHARDS GREATER 1)
      string(APPEND _name .${shard})
      list(APPEND _env GTEST_TOTAL_SHARDS=${arg_SHARDS}
                       GTEST_SHARD_INDEX=${shard})
    endif()
    add_test(NAME ${_name} COMMAND ${arg_COMMAND})
    # The library reports an unavailable target; the GoogleTest programs
    # then exit with 77.
    set_tests_properties(${_name} PROPERTIES
      ENVIRONMENT "${_env}"
      SKIP_RETURN_CODE 77
      SKIP_REGULAR_EXPRESSION "HWYBLAS_TARGET=[^ ]* is not available"
      LABELS "target;target_${target}")
  endforeach()
endfunction()

# Runs the GoogleTest program `exe` on every target in HWYBLAS_TEST_TARGETS.
function(hwyblas_add_gtest_target_tests exe shards)
  foreach(target IN LISTS HWYBLAS_TEST_TARGETS)
    hwyblas_target_launcher(${target} _launcher)
    set(_shards ${shards})
    if(_launcher AND shards GREATER 1)
      set(_shards ${HWYBLAS_SDE_SHARDS})
    endif()
    hwyblas_add_target_test(${target} ${exe} SHARDS ${_shards}
      COMMAND ${_launcher} $<TARGET_FILE:${exe}> --gtest_brief=1)
  endforeach()
endfunction()

add_executable(hwyblas_tests
  gemm_test.cc
  gemv_test.cc
  level1_test.cc
  band_packed_test.cc
  cblas_test.cc
  extensions_test.cc
  level2_test.cc
  level3_test.cc
  utest_cases_test.cc
  test_main.cc
)
target_link_libraries(hwyblas_tests PRIVATE hwyblas::hwyblas GTest::gtest hwyblas_options)

gtest_discover_tests(hwyblas_tests DISCOVERY_MODE PRE_TEST)
hwyblas_add_gtest_target_tests(hwyblas_tests 4)

# The public headers must also work from C.
add_executable(hwyblas_cblas_c_test cblas_c_test.c)
target_link_libraries(hwyblas_cblas_c_test PRIVATE hwyblas::hwyblas)
add_test(NAME CblasFromC COMMAND hwyblas_cblas_c_test)

# Eigen integration test (EIGEN_USE_BLAS), if an Eigen checkout is available.
set(HWYBLAS_EIGEN_DIR "${PROJECT_SOURCE_DIR}/../eigen"
    CACHE PATH "Path to an Eigen source checkout (for the Eigen test)")
if(EXISTS "${HWYBLAS_EIGEN_DIR}/Eigen/Core")
  add_executable(hwyblas_eigen_test eigen_test.cc test_main.cc)
  target_include_directories(hwyblas_eigen_test PRIVATE ${HWYBLAS_EIGEN_DIR})
  target_link_libraries(hwyblas_eigen_test PRIVATE hwyblas::hwyblas GTest::gtest hwyblas_options)
  gtest_discover_tests(hwyblas_eigen_test DISCOVERY_MODE PRE_TEST)
  hwyblas_add_gtest_target_tests(hwyblas_eigen_test 1)
else()
  message(STATUS "Eigen not found at ${HWYBLAS_EIGEN_DIR}: skipping Eigen test")
endif()

# Netlib CBLAS testers (C versions, from OpenBLAS's ctest/ directory), if an
# OpenBLAS source checkout is available. They test all CBLAS routines in
# column- and row-major storage, and the error exits.
set(HWYBLAS_NETLIB_CBLAS_TESTS_DIR "${PROJECT_SOURCE_DIR}/../OpenBLAS/ctest"
    CACHE PATH "Path to OpenBLAS's ctest/ directory (Netlib CBLAS testers)")
if(EXISTS "${HWYBLAS_NETLIB_CBLAS_TESTS_DIR}/c_dblat3c.c")
  set(_dir "${HWYBLAS_NETLIB_CBLAS_TESTS_DIR}")
  foreach(t s d c z)
    foreach(level 1 2 3)
      set(_exe hwyblas_netlib_${t}cblat${level})
      set(_sources "${_dir}/c_${t}blat${level}c.c" "${_dir}/c_${t}blas${level}.c")
      if(NOT level EQUAL 1)
        list(APPEND _sources "${_dir}/c_${t}${level}chke.c" "${_dir}/auxiliary.c"
             "${_dir}/constant.c" netlib/xerbla_hook.c)
      endif()
      add_executable(${_exe} ${_sources})
      # netlib/ provides the common.h and cblas.h the testers include.
      target_include_directories(${_exe} PRIVATE netlib "${_dir}")
      # ADD_: Fortran-style names with a trailing underscore.
      target_compile_definitions(${_exe} PRIVATE ADD_ CBLAS _CRT_SECURE_NO_WARNINGS)
      if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        # Machine-translated Fortran: silence its warnings.
        target_compile_options(${_exe} PRIVATE -w)
      elseif(MSVC)
        target_compile_options(${_exe} PRIVATE /w)
      endif()
      target_link_libraries(${_exe} PRIVATE hwyblas::hwyblas)
      set(_run -DEXE=$<TARGET_FILE:${_exe}>)
      if(NOT level EQUAL 1)
        list(APPEND _run "-DINPUT=${_dir}/${t}in${level}")
      endif()
      set(_script "${CMAKE_CURRENT_SOURCE_DIR}/netlib/run_tester.cmake")
      add_test(NAME NetlibCblas_${t}blat${level}
               COMMAND ${CMAKE_COMMAND} ${_run} -P "${_script}")
      foreach(target IN LISTS HWYBLAS_TEST_TARGETS)
        hwyblas_target_launcher(${target} _launcher)
        # The launcher is passed as a "|"-separated list.
        string(REPLACE ";" "|" _launcher "${_launcher}")
        hwyblas_add_target_test(${target} NetlibCblas_${t}blat${level}
          COMMAND ${CMAKE_COMMAND} ${_run} "-DLAUNCHER=${_launcher}"
                  -P "${_script}")
      endforeach()
    endforeach()
  endforeach()
else()
  message(STATUS "Netlib CBLAS testers not found at "
                 "${HWYBLAS_NETLIB_CBLAS_TESTS_DIR}: skipping them")
endif()
