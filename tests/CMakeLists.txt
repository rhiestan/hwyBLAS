include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# Always link GoogleTest statically, independent of BUILD_SHARED_LIBS, and
# do not run clang-tidy on it.
set(_hwyblas_shared ${BUILD_SHARED_LIBS})
set(_hwyblas_tidy "${CMAKE_CXX_CLANG_TIDY}")
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_CLANG_TIDY "")
FetchContent_MakeAvailable(googletest)
set(BUILD_SHARED_LIBS ${_hwyblas_shared})
set(CMAKE_CXX_CLANG_TIDY "${_hwyblas_tidy}")
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
  semantics_test.cc
  test_main.cc
)
target_link_libraries(hwyblas_tests PRIVATE hwyblas::hwyblas GTest::gtest hwyblas_options)
target_include_directories(hwyblas_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(HWYBLAS_LAPACK)
  target_sources(hwyblas_tests PRIVATE
    lapack/auxiliary_test.cc
    lapack/gees_test.cc
    lapack/geqp3_test.cc
    lapack/gesdd_test.cc
    lapack/gesvd_test.cc
    lapack/getrf_test.cc
    lapack/householder_test.cc
    lapack/linear_test.cc
    lapack/norms_test.cc
    lapack/potrf_test.cc
    lapack/qr_test.cc
    lapack/syev_test.cc
    lapack/triangular_test.cc
  )
endif()

gtest_discover_tests(hwyblas_tests DISCOVERY_MODE PRE_TEST)
hwyblas_add_gtest_target_tests(hwyblas_tests 4)

# The public headers must also work from C.
add_executable(hwyblas_cblas_c_test cblas_c_test.c)
target_link_libraries(hwyblas_cblas_c_test PRIVATE hwyblas::hwyblas)
if(HWYBLAS_LAPACK)
  target_compile_definitions(hwyblas_cblas_c_test PRIVATE HWYBLAS_TEST_LAPACK)
endif()
add_test(NAME CblasFromC COMMAND hwyblas_cblas_c_test)

# Eigen integration test (EIGEN_USE_BLAS), if an Eigen checkout is available.
set(HWYBLAS_EIGEN_DIR "${PROJECT_SOURCE_DIR}/../eigen"
    CACHE PATH "Path to an Eigen source checkout (for the Eigen test)")
if(EXISTS "${HWYBLAS_EIGEN_DIR}/Eigen/Core")
  add_executable(hwyblas_eigen_test eigen_test.cc test_main.cc)
  # SYSTEM: no warnings (or clang-tidy diagnostics) from Eigen's headers.
  target_include_directories(hwyblas_eigen_test SYSTEM PRIVATE ${HWYBLAS_EIGEN_DIR})
  target_link_libraries(hwyblas_eigen_test PRIVATE hwyblas::hwyblas GTest::gtest hwyblas_options)
  gtest_discover_tests(hwyblas_eigen_test DISCOVERY_MODE PRE_TEST)
  hwyblas_add_gtest_target_tests(hwyblas_eigen_test 1)
  if(HWYBLAS_LAPACK)
    # The same with EIGEN_USE_LAPACKE: Eigen's decompositions call hwyBLAS's
    # LAPACKE functions.
    add_executable(hwyblas_eigen_lapacke_test eigen_lapacke_test.cc test_main.cc)
    target_include_directories(hwyblas_eigen_lapacke_test SYSTEM PRIVATE ${HWYBLAS_EIGEN_DIR})
    target_link_libraries(hwyblas_eigen_lapacke_test PRIVATE hwyblas::hwyblas GTest::gtest hwyblas_options)
    if(MSVC)
      target_compile_options(hwyblas_eigen_lapacke_test PRIVATE /bigobj)
    endif()
    gtest_discover_tests(hwyblas_eigen_lapacke_test DISCOVERY_MODE PRE_TEST)
    hwyblas_add_gtest_target_tests(hwyblas_eigen_lapacke_test 1)
  endif()

  # Eigen's own test suite, compiled with EIGEN_USE_BLAS against hwyBLAS
  # (about 900 programs: building them takes a while), and with
  # EIGEN_USE_LAPACKE if hwyBLAS has LAPACK and HWYBLAS_EIGEN_TESTSUITE_LAPACKE
  # is on. It runs as the test EigenTestSuite, and once per target that the
  # CPU supports (<target>.EigenTestSuite; not in SDE).
  option(HWYBLAS_EIGEN_TESTSUITE
         "Build and run Eigen's test suite against hwyBLAS" OFF)
  option(HWYBLAS_EIGEN_TESTSUITE_LAPACKE
         "Compile Eigen's test suite with EIGEN_USE_LAPACKE (needs HWYBLAS_LAPACK)" ON)
  # Tests that fail with any BLAS because of Eigen's EIGEN_USE_BLAS code:
  # nomalloc and product_notemporary (its BLAS wrappers allocate
  # temporaries), product_extra (passes negative strides, which BLAS
  # defines differently, so every BLAS reads out of bounds). Also
  # threads_non_blocking_thread_pool, which does not use BLAS and hangs
  # in about one of 20 runs.
  set(HWYBLAS_EIGEN_TESTSUITE_EXCLUDE
      "^((nomalloc|product_notemporary|product_extra)_|threads_non_blocking_thread_pool$)"
      CACHE STRING "Eigen tests to exclude (regular expression)")
  set(HWYBLAS_EIGEN_TESTSUITE_SEED 3 CACHE STRING
      "Random seed of Eigen's tests (EIGEN_SEED)")
  cmake_host_system_information(RESULT _cores QUERY NUMBER_OF_LOGICAL_CORES)
  set(HWYBLAS_EIGEN_TESTSUITE_JOBS ${_cores} CACHE STRING
      "Parallel jobs for Eigen's test suite")
  if(HWYBLAS_EIGEN_TESTSUITE AND NOT BUILD_SHARED_LIBS)
    message(WARNING "HWYBLAS_EIGEN_TESTSUITE needs a shared hwyblas")
  elseif(HWYBLAS_EIGEN_TESTSUITE)
    include(ExternalProject)
    set(_eigen_build "${CMAKE_CURRENT_BINARY_DIR}/eigen_testsuite")
    set(_eigen_args
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_BUILD_TYPE=Release
      -DEIGEN_TEST_EXTERNAL_BLAS=ON
      "-DBLAS_LIBRARIES=$<TARGET_LINKER_FILE:hwyblas>"
      -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF
      -DEIGEN_BUILD_DEMOS=OFF -DEIGEN_BUILD_DOC=OFF)
    set(_eigen_flags "")
    if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      list(APPEND _eigen_flags "/FI${CMAKE_CURRENT_SOURCE_DIR}/eigen/clang_cl_compat.h")
    endif()
    set(_eigen_exclude "${HWYBLAS_EIGEN_TESTSUITE_EXCLUDE}")
    set(_eigen_keep_going "")
    if(HWYBLAS_LAPACK AND HWYBLAS_EIGEN_TESTSUITE_LAPACKE)
      # Eigen declares the LAPACKE functions itself; hwyblas provides them.
      # Eigen links only its official tests with BLAS_LIBRARIES; the others
      # (not run) call LAPACKE too.
      list(APPEND _eigen_flags "-DEIGEN_USE_LAPACKE")
      list(APPEND _eigen_args
           "-DEIGEN_TEST_CUSTOM_LINKER_FLAGS=$<TARGET_LINKER_FILE:hwyblas>")
      # Tests that fail with any LAPACKE because of Eigen's EIGEN_USE_LAPACKE
      # code:
      # - jacobisvd_41 and _42 (svd_inf_nan) do not compile: Eigen's
      #   JacobiSVD_LAPACKE.h calls the non-const outerStride() of its input,
      #   an expression there. The build keeps going past them (a program
      #   that fails to build for another reason still fails its test,
      #   which cannot run).
      # - bdcsvd_37 to _40 corrupt the heap: JacobiSVD_LAPACKE.h passes the
      #   outer stride of its input as LDA of its own compact copy, and
      #   BDCSVD gives JacobiSVD blocks of a larger matrix.
      # - jacobisvd_54 (svd_preallocate) forbids allocations, which the
      #   LAPACKE wrappers make.
      # - schur_real_2, eigensolver_generic_2 and eigensolver_complex_2
      #   expect NoConvergence after setMaxIterations(1), which the LAPACKE
      #   path ignores.
      string(APPEND _eigen_exclude
             "|^(jacobisvd_(4[12]|54)|bdcsvd_(3[7-9]|40)|schur_real_2"
             "|eigensolver_(generic|complex)_2)$")
      if(CMAKE_GENERATOR MATCHES "Ninja")
        set(_eigen_keep_going "-k 0")
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(_eigen_keep_going "-k")
      endif()
    endif()
    if(_eigen_flags)
      list(JOIN _eigen_flags " " _eigen_flags)
      list(APPEND _eigen_args "-DEIGEN_TEST_CUSTOM_CXX_FLAGS=${_eigen_flags}")
    endif()
    ExternalProject_Add(eigen_testsuite
      SOURCE_DIR "${HWYBLAS_EIGEN_DIR}"
      BINARY_DIR "${_eigen_build}"
      CMAKE_ARGS ${_eigen_args}
      BUILD_COMMAND ${CMAKE_COMMAND} -DEIGEN_BUILD=<BINARY_DIR>
                    "-DKEEP_GOING=${_eigen_keep_going}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/eigen/build_testsuite.cmake"
      INSTALL_COMMAND ""
      DEPENDS hwyblas)
    set(_run ${CMAKE_COMMAND} -DEIGEN_BUILD=${_eigen_build}
        -DHWYBLAS_DIR=$<TARGET_FILE_DIR:hwyblas> -DCTEST=${CMAKE_CTEST_COMMAND}
        -DJOBS=${HWYBLAS_EIGEN_TESTSUITE_JOBS}
        -DSEED=${HWYBLAS_EIGEN_TESTSUITE_SEED}
        "-DEXCLUDE=${_eigen_exclude}")
    set(_script "${CMAKE_CURRENT_SOURCE_DIR}/eigen/run_testsuite.cmake")
    add_test(NAME EigenTestSuite COMMAND ${_run} -P "${_script}")
    # It runs many tests in parallel itself.
    set_tests_properties(EigenTestSuite PROPERTIES RUN_SERIAL ON)
    foreach(target IN LISTS HWYBLAS_TEST_TARGETS)
      hwyblas_target_launcher(${target} _launcher)
      if(_launcher)
        continue()  # too slow in SDE
      endif()
      hwyblas_add_target_test(${target} EigenTestSuite
        COMMAND ${_run} -DCHECK=$<TARGET_FILE:hwyblas_eigen_test>
                -P "${_script}")
      set_tests_properties(${target}.EigenTestSuite PROPERTIES RUN_SERIAL ON)
    endforeach()
  endif()
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
      if(UNIX)
        target_link_libraries(${_exe} PRIVATE m)  # sqrt etc.
      endif()
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

# Differential fuzzer: compares hwyBLAS with a reference BLAS (OpenBLAS by
# default) on random arguments. Run it by hand for longer soak tests, e.g.
#   blas_fuzz --time=600 --seed=7 OpenBLAS=path/to/libopenblas.dll
add_executable(blas_fuzz fuzz/blas_fuzz.cc)
target_include_directories(blas_fuzz PRIVATE "${PROJECT_SOURCE_DIR}/bench")
target_link_libraries(blas_fuzz PRIVATE hwyblas_options ${CMAKE_DL_LIBS})
# It loads hwyblas at runtime (as the benchmark does), from this path.
add_dependencies(blas_fuzz hwyblas)
target_compile_definitions(blas_fuzz PRIVATE
  "HWYBLAS_LIBRARY=\"$<TARGET_FILE:hwyblas>\"")

if(WIN32)
  set(_openblas "${PROJECT_SOURCE_DIR}/../OpenBLAS-0.3.34-x64/bin/libopenblas.dll")
else()
  set(_openblas "")
endif()
set(HWYBLAS_FUZZ_REFERENCE "${_openblas}" CACHE FILEPATH
    "Reference BLAS library for the fuzz tests (e.g. OpenBLAS)")
set(HWYBLAS_FUZZ_ITERATIONS 3000 CACHE STRING
    "Cases per fuzz test run")
set(HWYBLAS_SDE_FUZZ_ITERATIONS 1000 CACHE STRING
    "Cases per fuzz test run in SDE")
if(EXISTS "${HWYBLAS_FUZZ_REFERENCE}")
  set(_reference "Reference=${HWYBLAS_FUZZ_REFERENCE}")
  add_test(NAME Fuzz COMMAND blas_fuzz --iterations=${HWYBLAS_FUZZ_ITERATIONS}
                                       ${_reference})
  foreach(target IN LISTS HWYBLAS_TEST_TARGETS)
    hwyblas_target_launcher(${target} _launcher)
    set(_iterations ${HWYBLAS_FUZZ_ITERATIONS})
    if(_launcher)
      set(_iterations ${HWYBLAS_SDE_FUZZ_ITERATIONS})
    endif()
    hwyblas_add_target_test(${target} Fuzz
      COMMAND ${_launcher} $<TARGET_FILE:blas_fuzz> --iterations=${_iterations}
              ${_reference})
  endforeach()
else()
  message(STATUS "No reference BLAS for the fuzz tests "
                 "(HWYBLAS_FUZZ_REFERENCE): skipping them")
endif()
