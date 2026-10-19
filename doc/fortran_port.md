# How the LAPACK routines are translated

hwyBLAS's LAPACK (`src/lapack/`) is a line-by-line C++ port of the reference
LAPACK 3.12.1. The goal is that every routine can be read next to its Fortran
source: same structure, same variable names, same 1-based indexing, same
order of floating-point operations. Deviations are allowed only where they
are documented in the routine's comment (own block sizes, fused kernels,
fixed reference bugs).

The routines of milestones M0–M2 (getrf … syev) were ported by hand. From M3
on (gesvd, gesdd, gees and their helpers, about 80 routines so far), a small
translator produces a first draft that is then finished by hand. The scripts are in
`doc/fortran_port/`:

| Script | Purpose |
|---|---|
| `f2cc.py` | Translates one fixed-form Fortran routine into a C++ function body in the port's style. |
| `assemble.py` | Fills translated bodies into a hand-written C++ skeleton (signatures, templates, exports) and applies per-routine replacements. |

They are tools, not a compiler: the output always needs review, and nothing
in the build depends on them.

## Target style

The translator produces the conventions described in
`src/lapack/lapack_common.h`:

- **Arrays** become 1-based column-major views named like the Fortran array
  (upper case): `const Matrix<T> A(a, lda);` and then `A(i, j)`. A
  subarray passed to a routine, `A(I, J)`, becomes `A.ptr(i, j)`.
- **Scalars** keep their Fortran names in lower case, declared at the top of
  the function as in Fortran (`blasint`, `bool`, `R` = real type,
  `T` = the routine's element type).
- **Constants**: `PARAMETER`s become `const` locals; `1.0D+0` becomes `R(1)`.
- **Control flow**: `IF/ELSE IF/ELSE`, `DO` and `DO WHILE` map directly; a
  `DO` step that is a variable gets a condition for both signs (`m3 > 0 ? k
  <= hi : k >= hi`), since its sign is only known at run time. `CYCLE` and
  `EXIT` become `continue` and `break`. `GO TO 10` becomes `goto L10;` with a
  label `L10:`, so the GOTO-heavy routines (bdsqr, lasq*, lasd*, laqr*) keep
  their exact flow; a computed `GO TO (10, 20, 30) K` becomes a `switch` of
  `goto`s.
- **Statement functions** such as `CABS1(CDUM) = ABS(DBLE(CDUM)) +
  ABS(DIMAG(CDUM))` are dropped, and their uses call a helper of the same
  name (`Cabs1` in `lapack_common.h`).
- **Intrinsics**: `MAX/MIN` (`std::max<blasint>` for integer arguments,
  initializer lists for more than two), `ABS`, `SQRT`, `SIGN`
  (`std::copysign`), `DBLE`/`REAL`, `CONJG`, `MOD`, `NINT`, `LSAME`, `**`
  (`(x * x)` for `**2`, otherwise `std::pow`).
- **Calls**: one template per real/complex pair, named after the complex
  routine (`DORGBR` and `ZUNGBR` both become `Ungbr`). A real routine called
  from a complex one gets `<R>`. `INFO` is not an argument: internal routines
  return it (`info = Gebrd(...)`). `XERBLA`, `ILAENV` and `xLAMCH` map to
  `Xerbla`, `Ilaenv` and `Lamch<R>`.
- **Real and complex**: only one precision of each pair is translated (`d*`
  for real-only routines, `z*` when the complex one exists), and the result
  is instantiated for all four types. Where the real and complex routines
  differ by more than conjugation, the difference is kept with
  `if constexpr`, or the two are ported as separate functions behind one
  template (`GesvdReal` / `GesvdComplex`), or as two overloads of the same
  name where their arguments differ (`WR` and `WI` against `W` in xHSEQR,
  xLAQR*, xTRSEN, xGEES).

## Step 1: translate (`f2cc.py`)

```sh
python doc/fortran_port/f2cc.py ../LAPACK/lapack-3.12.1/SRC/dlamrg.f > tr/dlamrg.cc
```

The script:

1. Joins continuation lines and keeps comments (without the `*>` Doxygen
   block and the `..` separators). Statements are upper-cased (Fortran is
   case-insensitive, and newer routines such as xLAQR0 are partly lower
   case).
2. Collects the declarations: types, array ranks, `PARAMETER`s,
   `EXTERNAL`s and the dummy arguments.
3. Emits the declarations, then translates statement by statement: a regex
   for the statement kind, a tokenizer plus a recursive expression translator
   for expressions.
4. Translates calls using the `SIGS` table, which gives each known routine's
   argument kinds: `v` value, `p` scalar output (passed as a pointer), `a`
   array (pointer, `X.ptr(i, j)` for subarrays), `i` INFO (dropped and
   assigned from the return value). `SIGS_COMPLEX` has the kinds of the
   complex routines whose arguments differ from the real ones'. Routines
   missing from the tables are marked `/*CHECK SIG*/`.

Everything the script cannot decide is marked `/*CHECK ...*/`: local array
sizes, unknown signatures, string concatenation, `ILAENV` options, `DATA`
and `EQUIVALENCE` statements, and whole arrays passed where a pointer is
expected. For example, `dlamrg.f` becomes:

```cpp
  const Vector<R> A(a);
  const Vector<blasint> INDEX(index);
  blasint i, ind1, ind2, n1sv, n2sv;
  n1sv = n1;
  ...
  // while ( (N1SV > 0) & (N2SV > 0) )
L10:
  if (n1sv > 0 && n2sv > 0) {
    if (A(ind1) <= A(ind2)) {
      INDEX(i) = ind1;
  ...
    goto L10;
  }
```

## Step 2: assemble (`assemble.py`)

The parts the translator does not write are written by hand once per file
as a skeleton: the copyright header, includes, template signatures, local
array sizes, explicit instantiations and the `extern "C"` exports. Each
function body is a placeholder line:

```cpp
template <typename T>
blasint GesddReal(char jobz, blasint m, blasint n, T* a, blasint lda, ...) {
  using R = T;
//@@BODY dgesdd.cc|R dum[/*CHECK*/];=>R dum[1];\n  const Vector<R> DUM(dum);|ierr = =>|...@@
}
```

```sh
python doc/fortran_port/assemble.py skel_gesdd.cc src/lapack/gesdd.cc tr
```

replaces each placeholder with the translated body of `tr/<file>`, after
applying the `|`-separated edits:

- `old=>new`: literal replacement (`\n` in either part stands for a newline);
- `~word=>new`: whole-word replacement (e.g. renaming a variable);
- `%regex=>new`: a regular expression replacement;
- `VOID`: the routine is a `void` function (`return info;` becomes
  `return;`, and a trailing `return;` is removed).

It then applies the common clean-ups (`Sign(` → `std::copysign(`,
`R(1.0)` → `R(1)`, complex constants `(R(1.0),R(0.0))` → `T(1)`, removing
the `// End of ...` comment, `&*x` → `x`, and declaring `info` at its first
assignment). Keeping the edits in the skeleton
means a fix to the translator only requires running both steps again.

## Step 3: finish by hand

The assembled file compiles only after a review pass. What usually needs
attention:

- every `/*CHECK*/` mark;
- workspace queries and sizes (`WorkSize<T>` replaces `xROUNDUP_LWORK`);
  `ILAENV` calls (`Ilaenv<T>(...)` with the right options);
- scalar outputs and in/out arguments of the callee, which must be passed
  as pointers;
- local arrays (`R dd[3]`) and `Vector` views on them;
- `Xerbla<T>` in real-only routines (`Xerbla<R>`), and explicit template
  arguments where they cannot be deduced (`Lasdt<R>(...)`);
- variables that the compiler warns may be used uninitialized because the
  Fortran flow initializes them via `GO TO` (initialized, with a comment);
- `2**K` as `blasint{1} << k`, and `INT(LOG(...))`-style mixed expressions,
  checked against the Fortran;
- a matrix named `T` (xLAQR2, xTREXC, xTRSEN …) clashes with the template
  parameter and is renamed `Tm`. A regex that renames `T(` must catch every
  index expression: `T(*ns, *ns)` once stayed the complex number
  `T(ns, ns)`, which compiled and broke xLAQR3's deflation test;
- `**` binds tighter than `*`: `A*B**2` is `A*(B*B)`. Without the
  parentheses, `a * b * b` rounds differently (and `A/B**2` would even be
  wrong). The translator once emitted them so; it changed xLASD4's
  iterations (found by the comparison below) and three other routines;
- `DATA` tables become `static const` arrays with views, `EQUIVALENCE`d
  arrays one array with two views (xLALN2);
- `INT(WORK(1))` of a complex workspace query needs `std::real`;
- `INFO` / `IERR` values the Fortran sets but never reads (the compiler
  warns) are dropped, with a comment where it matters.

Once a file has been finished by hand, it is edited directly and not
assembled again.

## Verification

A ported routine is accepted when:

- **Comparison**: it reads line by line like the Fortran, with every
  difference documented in its comment;
- **Unit tests** (`tests/lapack/*_test.cc`) pass: residuals and
  orthogonality, all code paths (small block sizes and subproblem sizes
  forced through `hwyblas_set_ilaenv`), workspace queries and minimal
  workspaces, error exits, and the LAPACKE layouts;
- **Mutation testing**: deliberately wrong versions of the code are caught
  by the unit tests or, failing that, by LAPACK's testers;
- **LAPACK's own testers** (`TESTING/LIN`, `TESTING/EIG`), run by hand with
  `tests/lapack/run_lapack_testers.sh`, link the ported routines in place of
  the reference ones and show no new failures.

- **Comparison with the reference LAPACK**, run by hand with
  `tests/lapack/compare_reference.sh`: the drivers give the same results
  as the reference.

Residual checks cannot see changes to heuristics that only steer the
convergence (shifts, deflation criteria, the NIBBLE crossover of xLAQR0),
or a different rounding: the result is still a valid factorization. The
comparison sees them. It builds the reference LAPACK as a shared library
(without floating-point contraction) against hwyBLAS's BLAS, loads it next
to hwyBLAS in one process, redirects hwyBLAS's ILAENV to the reference one
and calls every driver (xGETRF, xPOTRF, xGEQRF, xGELQF, xGEQP3,
xSYEV/xHEEV, xGESVD, xGESDD, xGEES, with all job options) in both on the
same matrices of orders 0 to 401. With the same BLAS and parameters, the
real results are bitwise identical, and mutations of the heuristics change
them. The complex results differ in the last bits, because flang evaluates
some complex divisions differently (for COMPLEX / REAL it performs a full
complex division, the standard's conversion to COMPLEX first, while C++
and gfortran divide the two parts); the multishift QR algorithm amplifies
such differences into a different order of the eigenvalues. The tool
therefore compares the complex singular values and eigenvalues with a
tolerance (the nonsymmetric eigenvalues as sets), and only counts the
bitwise identical results otherwise.
