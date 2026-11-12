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
# compare_reference directly) is not part of all.
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
EOF
cmake -S "$work/compare-src" -B "$work/compare-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release ${CXX:+"-DCMAKE_CXX_COMPILER=$CXX"} \
  > "$work/compare.log"
cmake --build "$work/compare-build" >> "$work/compare.log"

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
