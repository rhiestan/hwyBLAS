# Builds Eigen's test programs (the HWYBLAS_EIGEN_TESTSUITE option):
#   cmake -DEIGEN_BUILD=<dir> [-DKEEP_GOING=<build tool flags>] -P ...
#
# With KEEP_GOING (e.g. "-k 0" for Ninja), programs that fail to compile do
# not stop the build: the build only warns, and their tests fail when the
# suite runs (unless they are excluded).

separate_arguments(keep_going NATIVE_COMMAND "${KEEP_GOING}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${EIGEN_BUILD}" --target buildtests
          -- ${keep_going}
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  if(KEEP_GOING)
    message(WARNING "Some of Eigen's test programs failed to build")
  else()
    message(FATAL_ERROR "Building Eigen's test programs failed")
  endif()
endif()
