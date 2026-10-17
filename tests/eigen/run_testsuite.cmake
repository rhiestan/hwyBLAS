# Runs Eigen's test suite, built by the HWYBLAS_EIGEN_TESTSUITE option:
#   cmake -DEIGEN_BUILD=<dir> -DHWYBLAS_DIR=<dir of the library> -DCTEST=<ctest>
#         -DJOBS=<n> -DSEED=<n> -DEXCLUDE=<regex> [-DCHECK=<program>] -P ...
#
# Only Eigen's "Official" tests run (not those of unsupported/), with a fixed
# seed so that results are reproducible. CHECK is a GoogleTest program of
# hwyBLAS: if it exits with 77, HWYBLAS_TARGET is not available on this CPU
# and the run is skipped.

if(DEFINED CHECK)
  execute_process(COMMAND "${CHECK}" --gtest_list_tests OUTPUT_QUIET
                  RESULT_VARIABLE rc)
  if(rc EQUAL 77)
    # Matched by the test's SKIP_REGULAR_EXPRESSION.
    message("HWYBLAS_TARGET=$ENV{HWYBLAS_TARGET} is not available")
    return()
  endif()
endif()

if(WIN32)
  # The tests find hwyblas.dll through PATH.
  file(TO_NATIVE_PATH "${HWYBLAS_DIR}" dir)
  set(ENV{PATH} "${dir};$ENV{PATH}")
endif()
set(ENV{EIGEN_SEED} "${SEED}")
execute_process(
  COMMAND "${CTEST}" --test-dir "${EIGEN_BUILD}" -L Official -E "${EXCLUDE}"
          -j ${JOBS} --timeout 900 --output-on-failure
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "Eigen's test suite failed")
endif()
