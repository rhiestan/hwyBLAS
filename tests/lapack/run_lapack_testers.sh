#!/usr/bin/env bash
# Runs LAPACK's own test programs (TESTING/LIN and TESTING/EIG of the
# reference LAPACK) against hwyBLAS: the ported routines replace the
# reference ones, which provide the routines that are not ported yet.
#
#   tests/lapack/run_lapack_testers.sh <hwyblas build> <LAPACK source> <work dir> [ctest regex]
#
# e.g. tests/lapack/run_lapack_testers.sh build ../LAPACK/lapack-3.12.1 /tmp/lt xlintst
# (only the linear equation tests). Needs a Fortran compiler (FC, or one that
# CMake finds) and Python 3 (PYTHON, default python3) for LAPACK's summary.
# With Visual Studio's flang, the reference LAPACK only builds if flang's
# intrinsic module files are installed (VS 2026 does not include them).
#
# The steps:
#  1. Build the reference LAPACK library, with hwyBLAS as its BLAS.
#  2. Compile lapack_tester_hooks.c, which connects the test programs' XERBLA
#     and ILAENV to hwyBLAS.
#  3. Build the test programs, linked with the hooks, hwyBLAS and the
#     reference library, in that order.
#  4. Run them and print LAPACK's summary (numerical failures are test
#     ratios above the threshold; other errors are wrong INFO values or
#     error exits).
set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,12p' "$0"
  exit 1
fi
hwy_build=$(cd "$1" && pwd)
lapack_src=$(cd "$2" && pwd)
mkdir -p "$3"
work=$(cd "$3" && pwd)
regex=${4:-}
here=$(cd "$(dirname "$0")" && pwd)
python=${PYTHON:-python3}

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    native() { cygpath -m "$1"; }
    hwy_lib=$hwy_build/lib/hwyblas.lib
    export PATH="$hwy_build/bin:$PATH"
    ;;
  Darwin)
    native() { echo "$1"; }
    hwy_lib=$hwy_build/lib/libhwyblas.dylib
    ;;
  *)
    native() { echo "$1"; }
    hwy_lib=$hwy_build/lib/libhwyblas.so
    ;;
esac
[ -f "$hwy_lib" ] || { echo "Not found: $hwy_lib" >&2; exit 1; }

common=(-G Ninja -DCMAKE_BUILD_TYPE=Release
        "-DBLAS_LIBRARIES=$(native "$hwy_lib")")
[ -n "${FC:-}" ] && common+=("-DCMAKE_Fortran_COMPILER=$FC")
[ -n "${CC:-}" ] && common+=("-DCMAKE_C_COMPILER=$CC")

echo "== 1. Reference LAPACK"
cmake -S "$lapack_src" -B "$work/reference" "${common[@]}" -DBUILD_TESTING=OFF \
  > "$work/reference.log"
cmake --build "$work/reference" --target lapack >> "$work/reference.log"
ref_lib=$(find "$work/reference/lib" -name 'liblapack.lib' -o -name 'liblapack.a' | head -1)

echo "== 2. Hooks"
mkdir -p "$work/hooks-src"
cat > "$work/hooks-src/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.20)
project(lapack_tester_hooks C)
add_library(hooks OBJECT "$(native "$here/lapack_tester_hooks.c")")
target_include_directories(hooks PRIVATE "$(native "$here/../../include")")
install(FILES \$<TARGET_OBJECTS:hooks> DESTINATION .)
EOF
cmake -S "$work/hooks-src" -B "$work/hooks-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release ${CC:+"-DCMAKE_C_COMPILER=$CC"} \
  "-DCMAKE_INSTALL_PREFIX=$(native "$work/hooks")" > "$work/hooks.log"
cmake --build "$work/hooks-build" --target install >> "$work/hooks.log"
hooks_obj=$(find "$work/hooks" -name 'lapack_tester_hooks*' | head -1)

echo "== 3. Test programs"
cmake -S "$lapack_src" -B "$work/testing" "${common[@]}" -DBUILD_TESTING=ON \
  "-DPYTHON_EXECUTABLE=$(native "$(command -v "$python")")" \
  "-DLAPACK_LIBRARIES=$(native "$hooks_obj");$(native "$hwy_lib");$(native "$ref_lib")" \
  > "$work/testing.log"
cmake --build "$work/testing" >> "$work/testing.log"

echo "== 4. Tests"
(cd "$work/testing" && ctest ${regex:+-R "$regex"} -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
   > "$work/ctest.log" 2>&1) || true
(cd "$work/testing" && "$python" lapack_testing.py -d TESTING -s)
echo "Detailed results: $work/testing/TESTING/*.out"
