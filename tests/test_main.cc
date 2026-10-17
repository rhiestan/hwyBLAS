// main() of the GoogleTest programs. Prints the SIMD target in use and skips
// all tests (exit code 77) if HWYBLAS_TARGET names a target that this CPU
// cannot run: the library then selects the target automatically, which the
// default test run already covers.
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "gtest/gtest.h"
#include "hwyblas/hwyblas.h"

namespace {

bool EqualsIgnoreCase(const char* a, const char* b) {
  for (; *a != '\0' && *b != '\0'; ++a, ++b) {
    if (tolower(static_cast<unsigned char>(*a)) !=
        tolower(static_cast<unsigned char>(*b))) {
      return false;
    }
  }
  return *a == *b;
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  printf("hwyBLAS target: %s\n", hwyblas_target_name());
#ifdef _MSC_VER
#pragma warning(suppress : 4996)  // getenv is safe here
#endif
  const char* requested = getenv("HWYBLAS_TARGET");
  if (requested != nullptr && *requested != '\0' &&
      !EqualsIgnoreCase(requested, hwyblas_target_name())) {
    printf("Skipping all tests: HWYBLAS_TARGET=%s is not available\n",
           requested);
    return 77;
  }
  return RUN_ALL_TESTS();
}
