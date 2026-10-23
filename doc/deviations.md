# Differences from the reference implementations

hwyBLAS implements the interfaces of the reference BLAS, CBLAS, LAPACK 3.12.1
and LAPACKE 3.12.1, so their documentation describes what each routine does
(see [Documentation](../README.md#documentation) in the README). This page
lists where hwyBLAS behaves differently. Each LAPACK and LAPACKE item is also
described in a comment at the routine in the source, named in the last
column.

## BLAS and CBLAS

The BLAS routines are not ported from the reference; they compute the same
results up to rounding. Where their behavior differs (error handling, which
elements are read, `?nrm2`, `?rotg`, zero elements of x and y) is described
in [Conventions](../README.md#conventions) in the README.

Additions to the reference interfaces:

- `hwyblas_set_error_handler` (`hwyblas/hwyblas.h`) replaces `xerbla_`
  and `cblas_xerbla`, which cannot be replaced at link time when hwyBLAS
  is a DLL. The default handlers print a message but, unlike the
  reference ones, do not stop the program.
- The extensions `?axpby`, `i?amin`, `?omatcopy`, `?imatcopy` (as in
  OpenBLAS), and `?gemmt` as another name of `?gemmtr`.
- `hwyblas_target_name` and the environment variable `HWYBLAS_TARGET`,
  which select the SIMD target (see [Building](../README.md#building)).

## LAPACK

The routines are line-by-line ports of LAPACK 3.12.1 (see
[fortran_port.md](fortran_port.md)) and give the same results as the
reference, bitwise for the real routines, except in the cases below.

`hwyblas_set_ilaenv` (`hwyblas/lapack.h`) replaces `ilaenv_`, which
cannot be replaced at link time when hwyBLAS is a DLL. `ilaenv_` returns
the reference's values.

### Fixed reference bugs

The results differ from the reference only in the cases described.

| Routine | Reference LAPACK 3.12.1 | hwyBLAS | Source |
|---------|-------------------------|---------|--------|
| `?geqrt3` | N = 0: calls itself with the same arguments until the stack overflows (`?geqrt` never calls it with N = 0) | returns | `Geqrt3`, `geqrt.cc` |
| `sgeqp3`, `dgeqp3` | Decides on blocked code with a workspace of 2*SN for the column norms, which take 2*N; with fixed columns (JPVT) and less than the optimal LWORK, `?laqps` writes up to 2*NFXD elements past WORK | compares with 2*N (blocked code when the workspace suffices for it) | `Geqp3`, `geqp3.cc` |
| `slarf1l`, `dlarf1l` | SIDE = 'R', v zero except its last element (or N = 1): scales column 1 of C instead of column N | scales column N, as `clarf1l` and `zlarf1l` do | `Larf1l`, `householder.cc` |
| `clatbs`, `zlatbs` | Lower triangular, TRANS = 'T' or 'C', in the scaled solve: leaves out A(j+1, j) when one element is below the diagonal (JLEN = 1), so x is wrong | includes it, as `slatbs` and `dlatbs` do | `LatbsComplex`, `gbsv.cc` |
| `?tgsen` | IJOB = 1, 2 or 4: the minimum LWORK (also returned by the workspace query) leaves `?tgsyl` a workspace of 0, which it rejects, so PL, PR or DIF are not computed | minimum and query are one element larger | `Tgsen`, `tgsen.cc` |
| `ctgsen`, `ztgsen` | IJOB = 3 or 5: the transposed solve of the Difl estimate passes B11 and B22 in the wrong order, so DIF(2) is the estimate of a different operator | passes them as `stgsen` and `dtgsen` do | `Tgsen`, `tgsen.cc` |
| `slasy2`, `dlasy2` | When NaNs fill the rest of the 4 x 4 system (N1 = N2 = 2), no pivot is found and the rows and columns of the previous step's pivot are swapped | no swap (X is NaN in both) | `Lasy2`, `trsen.cc` |

### Faster kernels

- `?getrf2` and `?potrf2`, and so `?getrf`, `?potrf` and the drivers that
  use them (`?gesv`, `?posv`, ...), stop their recursion at small blocks,
  which a kernel factors with the unblocked algorithm (`?getf2`, `?potf2`).
  The pivots are the same, but the results are rounded differently. Apart
  from the bug fixes above, this is the only difference in the real
  results from the reference (`factor_kernels.h`, `getrf.cc`, `potrf.cc`).
- `?lasr` (the plane rotations of `?bdsqr`, `?steqr` and `?bdsdc`) runs in
  a SIMD kernel, and `?steqr` applies its rotations in blocks of columns,
  both with the same operations on each element in the same order as the
  reference: the results are identical (`rotation_kernels.cc`,
  `steqr.cc`).

## LAPACKE

The functions have the reference LAPACKE's interface and behavior, except
for these fixes (documented at the function in `src/lapack/lapacke*.cc`,
the ones visible to callers also in `hwyblas/lapacke.h`).

| Function | Reference LAPACKE 3.12.1 | hwyBLAS |
|----------|--------------------------|---------|
| `LAPACKE_?gesvx` | Row-major, FACT = 'E': does not copy the equilibrated B back | copies it back, as in column-major layout |
| `LAPACKE_sgeesx`, `LAPACKE_dgeesx` | Passes the IWORK(1) of the workspace query as LIWORK, which is 0 for N = 1 and SENSE = 'V' or 'B', and `?geesx` rejects it | LIWORK is at least 1 |
| `LAPACKE_strsen`, `LAPACKE_dtrsen` | Passes no IWORK for JOB = 'N' or 'E', although `?trsen` writes IWORK(1): crashes | always passes IWORK |
| `LAPACKE_?syconv` | Row-major: transposes LDA rows of A, reading and writing past its N rows if LDA > N | transposes the N x N matrix |
| `LAPACKE_?lacpy` | Row-major: the elements of B outside the triangle UPLO become undefined | they are kept |
| `LAPACKE_cunmlq`, `LAPACKE_zunmlq` | SIDE = 'R': checks and copies A as K x M | K x N, as `LAPACKE_?ormlq` |
| `LAPACKE_cunmbr`, `LAPACKE_zunmbr` | VECT = 'P', NQ > K: checks and copies only min(NQ, K) of the NQ columns of A | all NQ columns, as `LAPACKE_?ormbr` |
| `LAPACKE_?ormrq`, `?unmrq`, `?ormrz`, `?unmrz` | SIDE = 'R': checks and copies A as K x M (and checks LDA against M in row-major layout) | K x N |
| `LAPACKE_?gemqrt` | Allocates M*NB elements of work, where SIDE = 'L' needs N*NB; in row-major layout, copies LDV x K of V and LDT x NB of T, which reads past V, or passes an invalid LDV or only NB of the K columns of T | allocates N*NB (SIDE = 'L') or M*NB; copies V as Q x K and T as NB x K, as `LAPACKE_?geqrt` returns them |

## Keeping this page current

A new deviation is documented in a comment at the routine (as described in
[fortran_port.md](fortran_port.md)) and, if users can notice it, added
here.
