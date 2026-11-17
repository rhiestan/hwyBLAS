#!/usr/bin/env bash
# Compares the ported LAPACK drivers with the reference LAPACK on the same
# inputs (see compare_reference.cc): the real results must be bitwise
# identical (except where a documented kernel rounds differently), the
# complex ones agree to rounding.
#
#   tests/lapack/compare_reference.sh <hwyblas build> <LAPACK source> <work dir> [sizes] [special]
#
# e.g. tests/lapack/compare_reference.sh build ../LAPACK/lapack-3.12.1 /tmp/cr
# hwyBLAS must be built as a shared library (the default). Needs a Fortran
# compiler (FC, or one that CMake finds); CC and CXX select the C and C++
# compilers. sizes (default 13) limits the matrix sizes, from 0 up to 200.
# special is one of huge, tiny, subnormal, zero, nan, inf, adversarial (all matrices get
# such values, see compare_reference.cc), or all (every one of them and then
# the ordinary matrices); the default is the ordinary matrices. fuzz (random
# sizes, shapes and kinds; sizes is how many orders; seed 1, other seeds: run
# compare_reference directly) is not part of all. arguments compares the
# argument validation instead (see compare_arguments.cc: INFO of invalid
# calls, and workspace queries; sizes is the number of trials per routine,
# default 400) and is not part of all either.
#
# The steps:
#  1. Build the reference LAPACK as a shared library, with hwyBLAS as its
#     BLAS and without floating-point contraction (as hwyBLAS is built).
#  2. Build compare_reference.cc.
#  3. Run it: it loads both libraries, calls each driver in both with the
#     same inputs and compares the results.
set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,16p' "$0"
  exit 1
fi
hwy_build=$(cd "$1" && pwd)
lapack_src=$(cd "$2" && pwd)
mkdir -p "$3"
work=$(cd "$3" && pwd)
sizes=${4:-13}
special=${5:-none}
here=$(cd "$(dirname "$0")" && pwd)

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    native() { cygpath -m "$1"; }
    hwy_link=$hwy_build/lib/hwyblas.lib
    hwy_shared=$hwy_build/bin/hwyblas.dll
    ref_pattern='liblapack.dll'
    exe=.exe
    export PATH="$hwy_build/bin:$work/reference/bin:$PATH"
    ;;
  Darwin)
    native() { echo "$1"; }
    hwy_link=$hwy_build/lib/libhwyblas.dylib
    hwy_shared=$hwy_link
    ref_pattern='liblapack*.dylib'
    exe=
    ;;
  *)
    native() { echo "$1"; }
    hwy_link=$hwy_build/lib/libhwyblas.so
    hwy_shared=$hwy_link
    ref_pattern='liblapack.so*'
    exe=
    ;;
esac
for f in "$hwy_link" "$hwy_shared"; do
  [ -f "$f" ] || { echo "Not found: $f" >&2; exit 1; }
done

echo "== 1. Reference LAPACK (shared)"
# The reference's XERBLA stops the program, which would end the comparison at
# the first invalid argument (the drivers' scaling routines report NaN
# arguments, see the special values). LAPACK's XERBLA suggests modifying its
# STOP: in a copy of the sources, it returns after the message, as the
# default error handler of hwyBLAS does.
if [ ! -d "$work/lapack-src" ]; then
  mkdir -p "$work/lapack-src"
  tar -C "$lapack_src" --exclude=./TESTING -cf - . | tar -C "$work/lapack-src" -xf -
  sed -i 's/^      STOP$/      RETURN/' "$work/lapack-src/SRC/xerbla.f"
  grep -q '^      RETURN$' "$work/lapack-src/SRC/xerbla.f" ||
    { echo "XERBLA was not patched" >&2; exit 1; }
fi
cmake -S "$work/lapack-src" -B "$work/reference" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON \
  -DBUILD_TESTING=OFF "-DBLAS_LIBRARIES=$(native "$hwy_link")" \
  -DCMAKE_Fortran_FLAGS=-ffp-contract=off \
  ${FC:+"-DCMAKE_Fortran_COMPILER=$FC"} ${CC:+"-DCMAKE_C_COMPILER=$CC"} \
  > "$work/reference.log"
cmake --build "$work/reference" --target lapack >> "$work/reference.log"
ref_lib=$(find "$work/reference" -name "$ref_pattern" -type f | head -1)
[ -n "$ref_lib" ] || { echo "The reference library was not built" >&2; exit 1; }

echo "== 2. compare_reference"
mkdir -p "$work/compare-src"
cat > "$work/compare-src/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.20)
project(compare_reference CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(compare_reference "$(native "$here/compare_reference.cc")")
target_include_directories(compare_reference PRIVATE
  "$(native "$here/../../include")" "$(native "$here/../../bench")")
find_package(Threads REQUIRED)
target_link_libraries(compare_reference PRIVATE \${CMAKE_DL_LIBS} Threads::Threads)
add_executable(compare_arguments "$(native "$here/compare_arguments.cc")")
target_include_directories(compare_arguments PRIVATE
  "$(native "$here/../../include")" "$(native "$here/../../bench")")
target_link_libraries(compare_arguments PRIVATE \${CMAKE_DL_LIBS} Threads::Threads)
EOF
cmake -S "$work/compare-src" -B "$work/compare-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release ${CXX:+"-DCMAKE_CXX_COMPILER=$CXX"} \
  > "$work/compare.log"
cmake --build "$work/compare-build" >> "$work/compare.log"

if [ "$special" = arguments ]; then
  echo "== 3. Argument validation"
  trials=${4:-400}
  [ "$trials" != 13 ] || trials=400
  header=$(native "$here/../../include/hwyblas/lapack.h")
  args=("$(native "$hwy_shared")" "$(native "$ref_lib")" "$header")
  # Only routines that validate their arguments (the reference calls XERBLA),
  # except the divide and conquer kernels, which need structured data
  # (xLAED1, xLAED2, xLAEDA, xLASD1, xLASD2, xLASQ2), xTGSJA (the reference
  # reads out of bounds when K + L > N, which it does not check), and xGEQRT3,
  # whose reference recurses forever for N = 0 (see doc/deviations.md).
  status=0
  checked=0
  for name in $("$work/compare-build/compare_arguments$exe" "${args[@]}" list | sed "s/[[:space:]]*$//"); do
    base=${name%_}
    case "${base#?}" in laed1 | laed2 | laeda | lasd1 | lasd2 | lasq2 | tgsja | geqrt3) continue ;; esac
    src="$work/lapack-src/SRC/$base.f"
    grep -qi 'CALL XERBLA' "$src" 2>/dev/null || continue
    checked=$((checked + 1))
    code=0
    out=$(timeout 90 "$work/compare-build/compare_arguments$exe" "${args[@]}" run "$name" "$trials" 1 "$(native "$src")" 2>/dev/null | grep -v 'On entry') || code=$?
    if [ "$code" -ne 0 ]; then
      status=1
      echo "$name: exit code $code"
      echo "$out" | grep -v ' 0 mismatches' | head -8
    fi
    # Workspace queries with one null argument at a time (the reference does
    # not read most of them there, and NumPy's lstsq leaves RCOND unset).
    code=0
    out=$(timeout 90 "$work/compare-build/compare_arguments$exe" "${args[@]}" queries "$name" 2>/dev/null | grep -v 'On entry') || code=$?
    if [ "$code" -ne 0 ]; then
      status=1
      echo "$name (null arguments in a workspace query): exit code $code"
      echo "$out" | grep -v ' 0 mismatches' | head -8
    fi
  done
  echo "$checked routines compared: $([ $status = 0 ] && echo PASSED || echo FAILED)"
  exit $status
fi

echo "== 3. Comparison"
if [ "$special" = all ]; then
  modes="huge tiny subnormal zero nan inf adversarial none"
else
  modes=$special
fi
status=0
for mode in $modes; do
  "$work/compare-build/compare_reference$exe" "$(native "$hwy_shared")" \
    "$(native "$ref_lib")" "$sizes" "$mode" || status=1
done
exit $status
