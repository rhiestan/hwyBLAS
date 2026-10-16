# Runs a Netlib CBLAS tester: cmake -DEXE=<tester> [-DINPUT=<file>] -P ...
#
# The Level 2 and 3 testers read their parameters from stdin. A tester fails
# with a nonzero exit code or, for some failures, only with a message in its
# report; both fail the test. The report is printed on failure.

if(DEFINED INPUT)
  execute_process(COMMAND "${EXE}" INPUT_FILE "${INPUT}"
                  OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
else()
  execute_process(COMMAND "${EXE}"
                  OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
endif()

set(report "${out}${err}")
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
