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
)
target_link_libraries(hwyblas_tests PRIVATE hwyblas::hwyblas GTest::gtest_main hwyblas_options)

gtest_discover_tests(hwyblas_tests DISCOVERY_MODE PRE_TEST)

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
