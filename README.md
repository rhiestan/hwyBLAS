# hwyBLAS

[![CI](https://github.com/rhiestan/hwyBLAS/actions/workflows/ci.yml/badge.svg)](https://github.com/rhiestan/hwyBLAS/actions/workflows/ci.yml)

A BLAS implementation built on [Google Highway](https://github.com/google/highway).

Goals:

1. **Clean, simple, well-documented code.** One portable source per routine;
   Highway generates code for each SIMD target and picks the best one at runtime.
2. **High performance.** On par with OpenBLAS and MKL (single-threaded).
3. **Pluggable into Eigen** via `EIGEN_USE_BLAS` (and `EIGEN_USE_LAPACKE`).
4. **Correct**, verified by tests against a reference implementation on every
   SIMD target that the CPU supports.
5. **Single-threaded** by design.

hwyBLAS is an independent project. It is not affiliated with Google.

## Status

| Level | Routines | Precisions |
|-------|----------|------------|
| 1 | `?axpy_`, `?scal_`, `?copy_`, `?swap_`, `i?amax_` | s d c z |
| 1 | `?dot_` / `?dotu_`, `?dotc_`, `sdsdot_`, `dsdot_` | s d / c z |
| 1 | `?nrm2_`, `?asum_` (`scnrm2_`, `dznrm2_`, `scasum_`, `dzasum_`) | s d c z |
| 1 | `?rot_` (`csrot_`, `zdrot_`), `?rotg_`, `csscal_`, `zdscal_` | s d c z |
| 1 | `?rotm_`, `?rotmg_`, `?cabs1_` | s d |
| 2 | `?gemv_`, `?trmv_`, `?trsv_` | s d c z |
| 2 | `?symv_` / `?hemv_` | s d / c z |
| 2 | `?ger_` / `?geru_`, `?gerc_` | s d / c z |
| 2 | `?syr_`, `?syr2_` / `?her_`, `?her2_` | s d / c z |
| 2 | banded: `?gbmv_`, `?tbmv_`, `?tbsv_` | s d c z |
| 2 | banded: `?sbmv_` / `?hbmv_` | s d / c z |
| 2 | packed: `?tpmv_`, `?tpsv_` | s d c z |
| 2 | packed: `?spmv_`, `?spr_`, `?spr2_` / `?hpmv_`, `?hpr_`, `?hpr2_` | s d / c z |
| 3 | `?gemm_`, `?symm_`, `?syrk_`, `?syr2k_`, `?trmm_`, `?trsm_` | s d c z |
| 3 | `?gemmtr_` (also as `?gemmt_`) | s d c z |
| 3 | `?hemm_`, `?herk_`, `?her2k_` | c z |

This is the complete reference BLAS, with both the Fortran interface
(`hwyblas/hwyblas.h`) and the CBLAS interface (`hwyblas/cblas.h`, row- and
column-major), and includes every routine that Eigen calls with
`EIGEN_USE_BLAS`. In addition, both interfaces provide these extensions, as
in OpenBLAS:

| Routine | Operation | Precisions |
|---------|-----------|------------|
| `?axpby` | y := alpha * x + beta * y | s d c z |
| `i?amin` | index of the first element of smallest \|x_i\| | s d c z |
| `?omatcopy` | B := alpha * op(A), op = N, T, R (conjugate) or C, row- or column-major | s d c z |
| `?imatcopy` | A := alpha * op(A) in place | s d c z |

All options are supported (transposition, conjugation, side, uplo, unit
diagonal, any vector increments including negative ones).

### LAPACK (in progress)

LAPACK is being ported from the reference implementation (LAPACK 3.12.1):
first the routines that Eigen calls with `EIGEN_USE_LAPACKE`, now the
common drivers (linear systems, eigenproblems, least squares). The
ports follow the Fortran original line by line (same algorithms, INFO
values, error exits and workspace rules, 1-based indexing), and are
documented where they deviate, e.g. in block sizes or optimized kernels. The
interfaces are the Fortran one (`hwyblas/lapack.h`, `dgetrf_` etc.) and
LAPACKE (`hwyblas/lapacke.h`, compatible with the reference LAPACKE and
Eigen). `-DHWYBLAS_LAPACK=OFF` leaves LAPACK out.

| Routines | Precisions |
|----------|------------|
| `?getrf`, `?getrf2`, `?laswp` (LU factorization) | s d c z |
| `?gesv`, `?getrs`, `?getri` (linear systems and inverse with the LU factorization) | s d c z |
| `?gesvx` (expert driver), `?gecon` (condition number), `?gerfs` (iterative refinement), `?geequ`, `?laqge` (equilibration) | s d c z |
| `?trtrs`, `?latrs` (triangular systems), `?trtri`, `?trti2` (inverse), `?trcon`, `?trrfs`, `?lantr` | s d c z |
| `?potrf`, `?potrf2` (Cholesky factorization) | s d c z |
| `?geqrf`, `?geqr2`, `?gelqf`, `?gelq2` (QR and LQ factorizations) | s d c z |
| `?geqp3`, `?laqps`, `?laqp2` (QR factorization with column pivoting) | s d c z |
| `?orgqr`, `?org2r`, `?orglq`, `?orgl2`, `?ormqr`, `?orm2r`, `?ormlq`, `?orml2` (Q of QR / LQ) | s d |
| `?ungqr`, `?ung2r`, `?unglq`, `?ungl2`, `?unmqr`, `?unm2r`, `?unmlq`, `?unml2` | c z |
| `?larfg`, `?larf`, `?larf1f`, `?larf1l`, `?larft`, `?larfb` (Householder reflectors) | s d c z |
| `?syev` (symmetric eigenproblem), `?sytrd`, `?sytd2`, `?orgtr`, `?orgql`, `?org2l` | s d |
| `?heev` (Hermitian eigenproblem), `?hetrd`, `?hetd2`, `?ungtr`, `?ungql`, `?ung2l` | c z |
| `?latrd`, `?steqr`, `?lasr` (tridiagonal reduction and QL / QR iteration) | s d c z |
| `?sterf`, `?lae2`, `?laev2`, `?lartg`, `?lanst`, `?lasrt` | s d |
| `?gesvd` (singular value decomposition), `?gebrd`, `?gebd2`, `?labrd` (bidiagonal reduction), `?bdsqr` (bidiagonal QR iteration) | s d c z |
| `?orgbr`, `?ormbr` (Q and P of the bidiagonal reduction) | s d |
| `?ungbr`, `?unmbr` | c z |
| `?las2`, `?lasv2`, `?lasq1` - `?lasq6` (dqds) | s d |
| `?gesdd` (singular value decomposition by divide and conquer) | s d c z |
| `?bdsdc`, `?lasdq`, `?lasd0` - `?lasd8`, `?lasda`, `?lasdt`, `?lamrg`, `?laed6` (bidiagonal divide and conquer) | s d |
| `?lacp2`, `?lacrm`, `?larcm` (complex matrices with real ones) | c z |
| `?gees` (Schur factorization), `?gebal`, `?gebak` (balancing), `?gehrd`, `?gehd2`, `?lahr2` (Hessenberg reduction) | s d c z |
| `?orghr`, `?ormhr` (Q of the Hessenberg reduction) | s d |
| `?unghr`, `?unmhr` | c z |
| `?hseqr`, `?laqr0` - `?laqr5`, `?lahqr` (Hessenberg QR algorithm), `?trsen`, `?trexc`, `?trsyl`, `?lacn2` (reordering of the Schur form), `?larfx` | s d c z |
| `?lanv2`, `?laexc`, `?lasy2`, `?laln2` | s d |
| `?geev` (eigenvalues and eigenvectors), `?geevx` (with balancing options and condition numbers), `?geesx` (Schur factorization with condition numbers), `?trevc3` (eigenvectors of the Schur form), `?trsna` (their condition numbers) | s d c z |
| `?laqtr` (quasi-triangular systems in real arithmetic) | s d |
| `?lartg`, `?rot` (with complex sine), `icmax1`, `izmax1`, `scsum1`, `dzsum1` | c z |
| `?lange`, `?lansy`, `?lassq` (norms) | s d c z |
| `?lanhe` | c z |
| `?lacpy`, `?laset`, `?lascl`, `ila?lc`, `ila?lr` | s d c z |
| `?rscl`, `csrscl`, `zdrscl` | s d c z |
| `?lacgv` | c z |
| `?lapy2`, `?lapy3`, `?ladiv`, `?isnan`, `?laisnan` | s d |
| `ilaenv`, `iparmq`, `ieeeck`, `lsame`, `?lamch` | |

LAPACKE has the reference LAPACKE's functions for these routines (with
documented fixes in `LAPACKE_?gesvx` and `LAPACKE_?geesx`, see
`hwyblas/lapacke.h`).

Besides hwyBLAS's own tests, the ported routines pass LAPACK's test programs
(`tests/lapack/run_lapack_testers.sh`, which links them with hwyBLAS in
place of the reference routines), and the real drivers give bitwise
identical results to the reference LAPACK on the same BLAS
(`tests/lapack/compare_reference.sh`), except where a documented kernel
rounds differently (the leaves of `?getrf2` and `?potrf2`, which make
small LU and Cholesky factorizations several times faster). Other kernels
keep the reference's arithmetic: the plane rotations of `?lasr`, most of
the time of `?gesvd` and `?syev`, run in a SIMD kernel with the same
operations on each element, in the same order. The LAPACK code keeps
LAPACK's license (`LICENSE-LAPACK`, modified BSD). How the Fortran is translated, and the
scripts used for it, are described in `doc/fortran_port.md`.

## Building

Requirements: CMake ≥ 3.20 and a C++17 compiler. **clang (or clang-cl on
Windows) is recommended**: MSVC works, but its code is 10-35% slower.

Highway is taken from a sibling checkout `../highway` if present (override with
`-DHWYBLAS_HWY_SOURCE_DIR=...`), otherwise from an installed package, otherwise
Highway 1.4.0 is downloaded. It is always linked statically into hwyBLAS.
GoogleTest is downloaded for the tests.

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build
ctest --test-dir build        # or run build/bin/hwyblas_tests directly
```

The build produces `build/bin/hwyblas.dll` (`build/lib/libhwyblas.so` on
Linux, `build/lib/libhwyblas.dylib` on macOS), exporting the Fortran-style
symbols (`dgemm_`, ...). Pass `-DBUILD_SHARED_LIBS=OFF` for a static library.

Supported SIMD targets on x86: SSE2, SSE4, AVX2, AVX-512 and AVX-512 (Zen 4).
Scalable-vector targets (SVE, RVV) are not supported yet. The best target that
the CPU supports is selected at runtime; the environment variable
`HWYBLAS_TARGET` (`SSE2`, `SSE4`, `AVX2`, `AVX3` or `AVX3_ZEN4`) selects a
specific one instead.

### Testing

`ctest` runs the unit tests, the Eigen test and the Netlib CBLAS testers
(from `../OpenBLAS/ctest`, if present; `-DHWYBLAS_NETLIB_CBLAS_TESTS_DIR=...`).
Most unit tests loop over all targets that the CPU supports. In addition,
every test program runs once per target with `HWYBLAS_TARGET` set; these
tests are named after the target (`SSE4.hwyblas_tests.0`, ...) and labeled,
e.g. `ctest -L target_SSE4`. Targets that the CPU cannot run are skipped.

To test the AVX-512 targets on a CPU without AVX-512, use the
[Intel Software Development Emulator](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html):

```sh
cmake build -DHWYBLAS_SDE=/path/to/sde/sde.exe
ctest --test-dir build -j16 -L "target_AVX3"   # AVX3 and AVX3_ZEN4
```

The differential fuzzer `blas_fuzz` calls hwyBLAS and a reference BLAS
(OpenBLAS by default, `-DHWYBLAS_FUZZ_REFERENCE=...`) with the same random
arguments and compares the results: random sizes, transposes, triangles,
alpha/beta (including 0 and 1), padded leading dimensions and negative
increments. It also checks that hwyBLAS does not read elements it must not
reference (they are NaN, e.g. the other triangle or C when beta = 0), does
not modify its inputs, and does not access memory outside of its operands
(each one is placed next to an inaccessible page). ctest runs a few thousand
cases per target; for longer runs:

```sh
build/bin/blas_fuzz --time=600 --seed=7 OpenBLAS=path/to/libopenblas.dll
build/bin/blas_fuzz --repro=dgemm:0x1234...  OpenBLAS=...   # rerun one failure
```

`-DHWYBLAS_EIGEN_TESTSUITE=ON` also builds Eigen's own test suite (from the
Eigen checkout, about 900 programs; this takes a while) with
`EIGEN_USE_BLAS` against hwyBLAS, and with LAPACK also with
`EIGEN_USE_LAPACKE` (`-DHWYBLAS_EIGEN_TESTSUITE_LAPACKE=OFF` leaves it
out), and runs its official tests as the test `EigenTestSuite`, also once
per target. A few Eigen tests are excluded because they fail with any BLAS
(`HWYBLAS_EIGEN_TESTSUITE_EXCLUDE`): Eigen's BLAS wrappers allocate
temporaries (`nomalloc`, `product_notemporary`), and pass negative strides,
which BLAS interprets differently (`product_extra`).
`threads_non_blocking_thread_pool` is also excluded: it occasionally hangs.
With `EIGEN_USE_LAPACKE`, these also fail with any LAPACKE (see
`tests/CMakeLists.txt`): `jacobisvd_41`/`42` do not compile and
`bdcsvd_37`-`40` corrupt the heap (Eigen's `JacobiSVD_LAPACKE.h` takes LDA
from its input instead of its copy of it), `jacobisvd_54` forbids
allocations, and `schur_real_2` and `eigensolver_generic_2`/`complex_2` set
an iteration limit that the LAPACKE path ignores.

### clang-tidy

`.clang-tidy` selects the checks (with the reasons for the disabled ones);
clangd uses it automatically. `-DHWYBLAS_CLANG_TIDY=ON` runs clang-tidy on
hwyBLAS's own sources while building (not on Highway or GoogleTest). The code
has no clang-tidy warnings with these checks.

### Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) builds hwyBLAS and runs `ctest` on
Linux x86-64 (GCC, and Clang with AddressSanitizer), Linux ARM64 (GCC), macOS
ARM64 (Apple Clang) and Windows (MSVC and clang-cl). The GCC and clang-cl jobs
also run the AVX-512 targets in SDE. The jobs fetch Eigen 5.0.1 and OpenBLAS
0.3.34 (the sources for the Netlib testers, and the library as the fuzz
reference) and download Highway 1.4.0. Eigen's own test suite is not run.

## Installing

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl \
      -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build
cmake --install build
```

This installs the library (`bin/hwyblas.dll` and `lib/hwyblas.lib` on
Windows, `lib/libhwyblas.so` elsewhere), the headers (`include/hwyblas/`), a
CMake package (`lib/cmake/hwyblas/`) and a pkg-config file
(`lib/pkgconfig/hwyblas.pc`). The shared library contains Highway. A static
build (`-DBUILD_SHARED_LIBS=OFF`) also installs the Highway it was built
with, as `hwyblas_hwy`, unless it was built against an installed Highway
package; consumers link it automatically through the CMake package. Install
rules are generated when hwyBLAS is the top-level project
(`-DHWYBLAS_INSTALL=ON|OFF` to override).

## Using hwyBLAS from another project

With CMake, either use an installed hwyBLAS:

```cmake
find_package(hwyblas 0.1 REQUIRED)  # with CMAKE_PREFIX_PATH=/path/to/prefix
target_link_libraries(my_app PRIVATE hwyblas::hwyblas)
```

or build it as part of the project (tests, benchmarks and install rules are
then off by default):

```cmake
include(FetchContent)
FetchContent_Declare(hwyblas GIT_REPOSITORY <hwyBLAS repository> GIT_TAG main)
FetchContent_MakeAvailable(hwyblas)  # or add_subdirectory(path/to/hwyBLAS)
target_link_libraries(my_app PRIVATE hwyblas::hwyblas)
```

On Windows, the DLL must be found at run time, e.g. by copying it next to the
executable (CMake 3.26 or newer; this does nothing for a static hwyBLAS):

```cmake
add_custom_command(TARGET my_app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:my_app>
          $<TARGET_RUNTIME_DLLS:my_app>
  COMMAND_EXPAND_LISTS)
```

Without CMake, use `pkg-config --cflags --libs hwyblas`, or add
`<prefix>/include` to the include path and link `hwyblas`.

If the project uses Highway itself, note that a static hwyBLAS, or one built
as part of a project that defines `hwy::hwy`, shares that Highway: then
`hwyblas_force_target` also restricts the project's own dynamic dispatch.

## Using hwyBLAS with Eigen

Define `EIGEN_USE_BLAS` before including Eigen and link against `hwyblas`;
with LAPACK, also `EIGEN_USE_LAPACKE` (Eigen declares the LAPACKE functions
itself, in its copy of `lapacke.h`):

```cmake
target_compile_definitions(my_app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
target_link_libraries(my_app PRIVATE hwyblas::hwyblas)
```

`tests/eigen_test.cc` checks the integration: products, triangular and
self-adjoint views, rank updates and the LLT, LDLT and LU decompositions.
`tests/eigen_lapacke_test.cc` checks that with `EIGEN_USE_LAPACKE` Eigen's
LLT, PartialPivLU, HouseholderQR, ColPivHouseholderQR, JacobiSVD, BDCSVD,
SelfAdjointEigenSolver, RealSchur and ComplexSchur run hwyBLAS's LAPACK, in
both storage orders.

## Using the CBLAS interface

```c
#include "hwyblas/cblas.h"

cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k,
            b, n, 0.0, c, n);
```

`cblas.h` declares the reference CBLAS API, plus `cblas_?gemmt` (the OpenBLAS
and MKL name of `cblas_?gemmtr`) and the by-value complex dot products
`cblas_?dotu`/`cblas_?dotc` of OpenBLAS. Row-major calls are mapped onto the
column-major routines; there is no extra cost except for row-major
`ConjTrans` in the matrix-vector routines and the row-major Hermitian Level 2
routines, which conjugate a copy of the vectors as the reference CBLAS does.

## Performance

Single-threaded GFLOP/s on an AMD Ryzen 9 5950X (Zen 3, AVX2), compared
with OpenBLAS 0.3.34 and MKL, square matrices, `trans = NN`:

| | n | hwyBLAS | OpenBLAS | MKL |
|---|---:|---:|---:|---:|
| sgemm | 128 | 121 | 97 | 129 |
| | 2048 | 122 | 122 | 128 |
| dgemm | 128 | 59 | 53 | 47 |
| | 2048 | 64 | 55 | 59 |
| cgemm | 128 | 127 | 109 | 132 |
| | 2048 | 120 | 115 | 121 |
| zgemm | 128 | 62 | 61 | 64 |
| | 2048 | 63 | 59 | 58 |

The other routines (`symm`, `syrk`, `syr2k`, `gemmt`, `trmm`, `trsm`, `gemv`,
`symv`/`hemv`, `trmv`, `trsv`) perform on par with OpenBLAS at medium and
large sizes, and mostly better than MKL on this AMD CPU. From n = 256 up, the rank updates
(`ger`, `syr`, `syr2`, `her`, `her2`) are mostly the fastest of the four
libraries, up to 2x ahead of the next best (`cher2`); at n = 64, MKL or AOCL
is sometimes ahead. AOCL-BLAS, which has special code paths
for small matrices, is the fastest for small (n ≤ 128) GEMM and TRSM.

The packed routines and banded routines with wide bands are mostly the
fastest of the four libraries (e.g. `dspmv` at n = 1000: 39 GFLOP/s vs. 22
for MKL). For very narrow real bands (a few off-diagonals), MKL is up to 2x
faster: there the columns are too short for the vector kernels to pay off.

To reproduce, pass other BLAS DLLs as `name=path`:

```sh
build/bin/blas_bench gemm --types=sdcz OpenBLAS=path/to/libopenblas.dll MKL=path/to/mkl_rt.2.dll
build/bin/blas_bench trsm --side=R --uplo=U --sizes=256,1024
```

The benchmark forces every library to a single thread. It interleaves the
libraries over several rounds and reports each one's best time.

## Design

```
include/hwyblas/hwyblas.h   public C API (Fortran calling convention)
include/hwyblas/cblas.h     public CBLAS API (src/cblas.cc)
src/common.h, common.cc     scalar traits, scratch buffers, xerbla_
src/api.h                   argument decoding and dispatch helpers
src/gemm.h                  GemmProblem: the description of any level 3 product
src/gemm.cc                 GEMM API and the dispatched GEMM driver
src/gemm-inl.h              GEMM driver: cache blocking (Goto/BLIS algorithm)
src/gemm_kernel-inl.h       GEMM micro-kernels (real and complex)
src/pack-inl.h              packing of (structured) operands into panels
src/level3.cc               SYMM, HEMM, SYRK, HERK, SYR2K, HER2K, GEMMTR and
                            TRMM on top of the driver
src/trsm.cc, trsm-inl.h     recursive TRSM and its small-triangle kernels
src/gemv.cc, gemv-inl.h     GEMV API and kernels
src/level2.cc               SYMV, HEMV (symv-inl.h), TRMV (trmv-inl.h),
                            TRSV (trsv-inl.h), GER, SYR, SYR2, HER, HER2
                            (rank_update-inl.h)
src/band_packed.cc          banded and packed Level 2 routines; storage
                            layouts in band_packed.h, kernels in
                            band_packed-inl.h
src/level1.cc, level1-inl.h AXPY and the vector kernels shared by others
src/matcopy.cc              ?OMATCOPY, ?IMATCOPY
tests/                      GoogleTest tests, Eigen integration test
bench/                      benchmark comparing BLAS libraries at runtime
```

**GEMM** follows the Goto/BLIS algorithm. B is packed into `kc × NR`
panels and A into `MR × kc` panels, so the micro-kernel streams contiguous
memory. The `MC × KC` block of A stays in L2 and the `KC × NR` panel of B in
L1. Transposition and conjugation are applied during packing, so one
micro-kernel per scalar type covers all `op(A)`, `op(B)` combinations.

The micro-kernel keeps an `MR × NR` tile of C in registers. Vectors run down
the columns of C (column-major), and elements of B are broadcast. On AVX2,
`MR × NR` is 8 × 6 for double, giving 12 accumulators out of 16 registers.
Complex kernels store numbers interleaved. They accumulate `a·re(b)` and
`a·im(b)` separately and combine them once at the end with
`AddSub(acc_re, Reverse2(acc_im))`.

**The other level 3 routines reuse the GEMM driver.** A `GemmProblem`
describes each operand by strides plus a structure. Symmetric and Hermitian
operands are read from their stored triangle while packing. For triangular
operands (TRMM), the driver also skips the all-zero parts of k. For a
triangular C (SYRK/HERK), it skips tiles outside the triangle. Packing
classifies each panel, so only panels crossing the diagonal take a slower
element-by-element path.

**TRSM is recursive.** It splits the triangle in half, solves one half,
updates the rest with a single large GEMM, and solves the other half.
Triangles of at most 32 rows go to a substitution kernel. That kernel works
on chunks of rows of `X` in registers (rows of `X` are independent in
`X·T = B`); the left side uses transposed 64-column blocks of B.

**Level 2 kernels stream the matrix once.** SYMV/HEMV process 4-column
strips, and each stored element serves both its own position and its mirror
image ("axpy" and "dot" form in one pass). TRMV uses the GEMV kernels on
strips.

**Files ending in `-inl.h`** are compiled once per SIMD target via Highway's
`foreach_target.h`. `HWY_DYNAMIC_DISPATCH` then selects the best target at
runtime. `hwyblas_force_target()` restricts dispatch, which lets the tests
cover every target.

## Conventions

- 32-bit integers (LP64) for dimensions and increments.
- Complex-valued functions (`cdotu_`, `zdotc_`, ...) return their result by
  value, like gfortran, OpenBLAS and AOCL. (MKL's Fortran interface instead
  passes a hidden result pointer as the first argument.)
- `?scal_` with `alpha = 0` multiplies, as in the reference BLAS, so NaN
  stays NaN.
- `?nrm2_` never overflows or underflows harmfully. Float is accumulated in
  double; double uses a fast sum of squares, and the reference safe-scaling
  algorithm only when that sum is out of range.
- `?rotg_` follows the reference BLAS 3.10+ algorithms (as OpenBLAS does).
- Error handling follows reference BLAS: invalid arguments call `xerbla_`,
  and the routine returns without side effects. Unlike the reference
  `xerbla_`, the default handler prints a message but does not terminate.
  CBLAS functions report through `cblas_xerbla` with the position of the
  CBLAS argument (the layout being argument 1), as the reference CBLAS does;
  it does not terminate either.
- `cblas_i?amax` returns a 0-based index, and 0 if `n <= 0` or `incx <= 0`.
- Extensions: `?axpby` reads neither x if `alpha = 0` nor y if `beta = 0`,
  and `?omatcopy`/`?imatcopy` with `alpha = 0` set the result to zero without
  reading A. `CblasConjNoTrans` is accepted by `cblas_?omatcopy` and
  `cblas_?imatcopy` only.
- `beta == 0` means the output is not read, so NaN/Inf in it do not
  propagate. This is the reference BLAS semantics.
- Elements that BLAS defines as not referenced are never read: the other
  triangle of structured matrices, the diagonal of unit triangular matrices
  and the imaginary parts of the diagonal of Hermitian matrices. The tests
  fill all of them with NaN to verify this. `?her_` and `?her2_` set the
  imaginary parts of the diagonal to zero, as in the reference BLAS.
- Unlike the reference BLAS, zero elements of x or y are not skipped
  (e.g. in `?ger_` or `?trsv_`), as in OpenBLAS. This only makes a
  difference when A contains Inf or NaN.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
