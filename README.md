# hwyBLAS

A BLAS implementation built on [Google Highway](https://github.com/google/highway).

Goals:

1. **Clean, simple, well-documented code.** One portable source per routine;
   Highway generates code for each SIMD target and picks the best one at runtime.
2. **High performance.** On par with OpenBLAS and MKL (single-threaded).
3. **Pluggable into Eigen** via `EIGEN_USE_BLAS`.
4. **Correct**, verified by tests against a reference implementation on every
   SIMD target that the CPU supports.
5. **Single-threaded** by design.

hwyBLAS is an independent project. It is not affiliated with Google.

## Status

| Routine | s | d | c | z | Notes |
|---------|---|---|---|---|-------|
| `?gemm_` | ✓ | ✓ | ✓ | ✓ | `N`/`T`/`C` for both operands |
| `?gemv_` | ✓ | ✓ | ✓ | ✓ | any `incx`/`incy`, including negative |

Next up: the remaining routines that Eigen uses with `EIGEN_USE_BLAS`:
`?symv_`/`?hemv_`, `?trmv_`, `?symm_`/`?hemm_`, `?syrk_`/`?herk_`,
`?trmm_`, `?trsm_`.

## Building

Requirements: CMake ≥ 3.20 and a C++17 compiler. **clang (or clang-cl on
Windows) is recommended**: MSVC works, but its code is 10-35% slower.

Highway is taken from a sibling checkout `../highway` if present (override with
`-DHWYBLAS_HWY_SOURCE_DIR=...`), otherwise from an installed package, otherwise
it is downloaded. GoogleTest is downloaded for the tests.

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build
ctest --test-dir build        # or run build/bin/hwyblas_tests directly
```

The build produces `build/bin/hwyblas.dll` (or `libhwyblas.so`), exporting the
Fortran-style symbols (`dgemm_`, ...). Pass `-DBUILD_SHARED_LIBS=OFF` for a
static library.

Supported SIMD targets on x86: SSE2, SSE4, AVX2, AVX-512 and AVX-512 (Zen 4).
Scalable-vector targets (SVE, RVV) are not supported yet.

## Using hwyBLAS with Eigen

Define `EIGEN_USE_BLAS` before including Eigen and link against `hwyblas`:

```cmake
target_compile_definitions(my_app PRIVATE EIGEN_USE_BLAS)
target_link_libraries(my_app PRIVATE hwyblas::hwyblas)
```

Note that with `EIGEN_USE_BLAS`, every Eigen operation backed by BLAS
references its routine, so currently only matrix-matrix and matrix-vector
products may be used. `tests/eigen_test.cc` checks the integration.

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

GEMV is memory-bound. hwyBLAS matches or exceeds OpenBLAS for all precisions
and both `N` and `T`.

To reproduce, pass other BLAS DLLs as `name=path`:

```sh
build/bin/blas_bench gemm --types=sdcz OpenBLAS=path/to/libopenblas.dll MKL=path/to/mkl_rt.2.dll
build/bin/blas_bench gemv --trans=T --shape=4000x4000
```

The benchmark forces every library to a single thread. It interleaves the
libraries over several rounds and reports each one's best time.

## Design

```
include/hwyblas/hwyblas.h   public C API (Fortran calling convention)
src/common.h, common.cc     argument decoding, scratch buffers, xerbla_
src/gemm.cc                 GEMM API, argument checks, dispatch
src/gemm-inl.h              GEMM driver: cache blocking (Goto/BLIS algorithm)
src/gemm_kernel-inl.h       GEMM micro-kernels (real and complex)
src/pack-inl.h              packing of operands into contiguous panels
src/gemv.cc, gemv-inl.h     GEMV API and kernels
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

**Files ending in `-inl.h`** are compiled once per SIMD target via Highway's
`foreach_target.h`. `HWY_DYNAMIC_DISPATCH` then selects the best target at
runtime. `hwyblas_force_target()` restricts dispatch, which lets the tests
cover every target.

## Conventions

- 32-bit integers (LP64) for dimensions and increments.
- Error handling follows reference BLAS: invalid arguments call `xerbla_`,
  and the routine returns without side effects. Unlike the reference
  `xerbla_`, the default handler prints a message but does not terminate.
- `beta == 0` means the output is not read, so NaN/Inf in it do not
  propagate. This is the reference BLAS semantics.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
