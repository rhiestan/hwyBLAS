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
)
target_link_libraries(hwyblas_tests PRIVATE hwyblas::hwyblas GTest::gtest_main hwyblas_options)

gtest_discover_tests(hwyblas_tests DISCOVERY_MODE PRE_TEST)

# The public headers must also work from C.
add_executable(hwyblas_cblas_c_test cblas_c_test.c)
target_link_libraries(hwyblas_cblas_c_test PRIVATE hwyblas::hwyblas)
add_test(NAME CblasFromC COMMAND hwyblas_cblas_c_test)

# Eigen integration test (EIGEN_USE_BLAS), if an Eigen checkout is available.
set(HWYBLAS_EIGEN_DIR "${PROJECT_SOURCE_DIR}/../eigen"
    CACHE PATH "Path to an Eigen source checkout (for the Eigen test)")
if(EXISTS "${HWYBLAS_EIGEN_DIR}/Eigen/Core")
  add_executable(hwyblas_eigen_test eigen_test.cc)
  target_include_directories(hwyblas_eigen_test PRIVATE ${HWYBLAS_EIGEN_DIR})
  target_link_libraries(hwyblas_eigen_test PRIVATE hwyblas::hwyblas GTest::gtest_main hwyblas_options)
  gtest_discover_tests(hwyblas_eigen_test DISCOVERY_MODE PRE_TEST)
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
      add_test(NAME NetlibCblas_${t}blat${level}
               COMMAND ${CMAKE_COMMAND} ${_run}
                       -P "${CMAKE_CURRENT_SOURCE_DIR}/netlib/run_tester.cmake")
    endforeach()
  endforeach()
else()
  message(STATUS "Netlib CBLAS testers not found at "
                 "${HWYBLAS_NETLIB_CBLAS_TESTS_DIR}: skipping them")
endif()
