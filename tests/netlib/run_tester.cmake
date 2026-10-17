# Runs a Netlib CBLAS tester:
#   cmake -DEXE=<tester> [-DINPUT=<file>] [-DLAUNCHER=<cmd>|<arg>|...] -P ...
#
# LAUNCHER is an optional command prefix, e.g. an emulator.
#
# The Level 2 and 3 testers read their parameters from stdin. A tester fails
# with a nonzero exit code or, for some failures, only with a message in its
# report; both fail the test. The report is printed on failure.

string(REPLACE "|" ";" launcher "${LAUNCHER}")
if(DEFINED INPUT)
  execute_process(COMMAND ${launcher} "${EXE}" INPUT_FILE "${INPUT}"
                  OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
else()
  execute_process(COMMAND ${launcher} "${EXE}"
                  OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
endif()

set(report "${out}${err}")
# An unavailable HWYBLAS_TARGET: print the library's message, so that ctest
# reports the test as skipped (SKIP_REGULAR_EXPRESSION).
if(report MATCHES "HWYBLAS_TARGET=[^ ]* is not available")
  message("${CMAKE_MATCH_0}")
  return()
endif()

if(NOT rc EQUAL 0)
  message(FATAL_ERROR "${report}\nTester exited with ${rc}")
endif()
# Failure messages of the testers. Routines or error exits that were not
# tested count as failures too.
string(REGEX MATCH
       "FAIL|SUSPECT|FATAL ERROR|NOT DETECTED|XERBLA WAS CALLED|NOT BE TESTED|WAS NOT TESTED"
       failure "${report}")
if(failure)
  message(FATAL_ERROR "${report}\nTester reported a failure (\"${failure}\")")
endif()
if(NOT report MATCHES "PASS")
  message(FATAL_ERROR "${report}\nTester did not report any passed tests")
endif()
