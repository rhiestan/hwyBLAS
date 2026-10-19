// Compares the ported LAPACK drivers with the reference LAPACK on the same
// inputs, loading both libraries at runtime:
//
//   compare_reference <hwyblas library> <reference liblapack> [cases]
//
// The reference library must be a shared library built with hwyBLAS as its
// BLAS (so that both use the same BLAS kernels) and without floating-point
// contraction; tests/lapack/compare_reference.sh builds it and runs this
// program. hwyBLAS's ILAENV is redirected to the reference ilaenv_, so that
// both use the same block sizes and crossover points.
//
// With the same BLAS, the same parameters and IEEE arithmetic, a faithful
// port computes exactly the same results as the reference. The real
// routines are therefore compared bitwise. The complex ones cannot be:
// flang evaluates mixed real and complex operations as complex ones
// (COMPLEX / REAL as a complex division, REAL * COMPLEX as a complex
// multiplication), where the port operates on the two parts, so results
// differ in the last bits (and with infinities, flang's 0 * Inf gives
// NaN). Their eigenvalues and singular values are compared
// with a tolerance instead (as sets for the nonsymmetric eigenvalues, whose
// order can change), and only the number of bitwise identical results is shown
// for the other outputs.
//
// Routines with a documented deviation that changes rounding (the leaf
// kernels of xGETRF2 and xPOTRF2, see src/lapack/factor_kernels.h) compute
// different last bits also in real precision: their factors are compared
// with the tolerance, and their pivots and INFO exactly.
//
// The drivers that factor with xGETRF (xGESV, xGESVX with FACT = 'N' or
// 'E') get only matrices with clear pivots: on graded matrices and ties,
// the different rounding of the leaf kernels can change the pivots. The
// routines that take a factorization get the same one in both libraries.
//
// The comparison catches changes that keep the results valid, which the
// residual tests cannot: a different shift strategy, deflation criterion,
// pivot choice or scaling. It is not part of the build or of ctest.
//
// Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "blas_loader.h"
#include "hwyblas/lapack.h"

namespace {

using hwyblas_bench::BlasLibrary;
using I = hwyblas_int;

template <typename T>
struct RealOf {
  using type = T;
};
template <typename T>
struct RealOf<std::complex<T>> {
  using type = T;
};
template <typename T>
using Real = typename RealOf<T>::type;
template <typename T>
constexpr bool kComplex = !std::is_same_v<T, Real<T>>;

template <typename T>
T Conj(T x) {
  if constexpr (kComplex<T>) {
    return std::conj(x);
  } else {
    return x;
  }
}

template <typename T>
const char* Prefix() {
  if constexpr (std::is_same_v<T, float>) return "s";
  if constexpr (std::is_same_v<T, double>) return "d";
  if constexpr (std::is_same_v<T, std::complex<float>>) return "c";
  return "z";
}

// How an output is compared.
enum class Kind {
  kInteger,      // INFO, pivots, SDIM: always equal
  kValues,       // singular values, eigenvalues: with a tolerance (complex)
  kEigenvalues,  // nonsymmetric eigenvalues: as a set (complex)
  kOther,        // factors, vectors: bitwise identity counted (complex)
  kRounded,      // factors rounded differently by design: with a tolerance
};

struct Output {
  std::string name;
  Kind kind;
  std::vector<std::complex<double>> values;
};

struct Result {
  std::vector<Output> outputs;
  double lwork = 0;  // the workspace query's result
};

template <typename T>
void Add(Result& r, const char* name, Kind kind, const T* x, size_t n) {
  Output o{name, kind, {}};
  o.values.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    o.values.emplace_back(std::complex<double>(x[i]));
  }
  r.outputs.push_back(std::move(o));
}

template <typename T>
void Add(Result& r, const char* name, Kind kind, const std::vector<T>& x) {
  Add(r, name, kind, x.data(), x.size());
}

void AddInt(Result& r, const char* name, I x) {
  Add(r, name, Kind::kInteger, &x, 1);
}

std::mt19937_64 rng(20261019);

template <typename T>
T Random() {
  std::uniform_real_distribution<double> d(-1, 1);
  if constexpr (kComplex<T>) {
    const double re = d(rng);
    return T(static_cast<Real<T>>(re), static_cast<Real<T>>(d(rng)));
  } else {
    return static_cast<T>(d(rng));
  }
}

// An m x n matrix of the given kind: 0 random, 1 cyclic permutation,
// 2 graded, 3 Jordan block with noise, 4 small integers (ties).
template <typename T>
std::vector<T> TestMatrix(int kind, I m, I n) {
  using R = Real<T>;
  std::vector<T> a(static_cast<size_t>(m) * n);
  for (I j = 0; j < n; ++j) {
    for (I i = 0; i < m; ++i) {
      T& x = a[i + static_cast<size_t>(j) * m];
      x = Random<T>();
      switch (kind) {
        case 1:
          x = T(i == j + 1 || (i == 0 && j == n - 1) ? 1 : 0);
          break;
        case 2:
          x *= static_cast<R>(std::ldexp(1.0, (j - i) % 40));
          break;
        case 3:
          x = T(i == j || i + 1 == j ? 1 : 0) + Random<T>() * R(1e-3);
          break;
        case 4:
          x = T(static_cast<R>(std::round(4 * std::real(x))));
          break;
        default:
          break;
      }
    }
  }
  return a;
}

// A Hermitian (symmetric) matrix, both triangles set.
template <typename T>
std::vector<T> Hermitian(I n) {
  std::vector<T> a = TestMatrix<T>(0, n, n);
  for (I j = 0; j < n; ++j) {
    for (I i = 0; i <= j; ++i) {
      T& upper = a[i + static_cast<size_t>(j) * n];
      if (i == j) upper = T(std::real(upper));
      a[j + static_cast<size_t>(i) * n] = Conj(upper);
    }
  }
  return a;
}

// A Hermitian positive definite matrix B^H * B + n * I.
template <typename T>
std::vector<T> PositiveDefinite(I n) {
  const std::vector<T> b = TestMatrix<T>(0, n, n);
  std::vector<T> a(static_cast<size_t>(n) * n);
  for (I j = 0; j < n; ++j) {
    for (I i = 0; i < n; ++i) {
      T s = i == j ? T(Real<T>(n)) : T(0);
      for (I k = 0; k < n; ++k) {
        s += Conj(b[k + static_cast<size_t>(i) * n]) *
             b[k + static_cast<size_t>(j) * n];
      }
      a[i + static_cast<size_t>(j) * n] = s;
    }
  }
  return a;
}

size_t Size(I m, I n) {
  return static_cast<size_t>(std::max<I>(m, 1)) * std::max<I>(n, 1);
}

// Selects eigenvalues by the signs of their real and imaginary parts, so
// that the reordering moves eigenvalues in both directions.
template <typename R>
I SelectReal(const R* wr, const R* wi) {
  return (*wr > 0) != (*wi > 0) ? 1 : 0;
}
template <typename R>
I SelectComplex(const R* w) {
  return (w[0] > 0) != (w[1] > 0) ? 1 : 0;
}

// The drivers. Each takes the library, the inputs and LWORK (-1: query) and
// returns its outputs. Character arguments are followed by their lengths,
// as in the Fortran ABI.

template <typename T>
Result Getrf(const BlasLibrary& lib, std::vector<T> a, I m, I n, I) {
  using F = void (*)(const I*, const I*, T*, const I*, I*, I*);
  const auto f = lib.Get<F>((std::string(Prefix<T>()) + "getrf_").c_str());
  const I lda = std::max<I>(m, 1);
  std::vector<I> ipiv(std::max<I>(std::min(m, n), 1));
  I info = 0;
  f(&m, &n, a.data(), &lda, ipiv.data(), &info);
  Result r;
  AddInt(r, "info", info);
  Add(r, "ipiv", Kind::kInteger, ipiv);
  Add(r, "a", Kind::kRounded, a);
  return r;
}

template <typename T>
Result Potrf(const BlasLibrary& lib, std::vector<T> a, I n, char uplo) {
  using F = void (*)(const char*, const I*, T*, const I*, I*, size_t);
  const auto f = lib.Get<F>((std::string(Prefix<T>()) + "potrf_").c_str());
  const I lda = std::max<I>(n, 1);
  I info = 0;
  f(&uplo, &n, a.data(), &lda, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "a", Kind::kRounded, a);
  return r;
}

// geqrf and gelqf.
template <typename T>
Result Geqrf(const BlasLibrary& lib, const char* name, std::vector<T> a, I m,
             I n, I lwork) {
  using F = void (*)(const I*, const I*, T*, const I*, T*, T*, const I*, I*);
  const auto f = lib.Get<F>((std::string(Prefix<T>()) + name).c_str());
  const I lda = std::max<I>(m, 1);
  std::vector<T> tau(std::max<I>(std::min(m, n), 1));
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  f(&m, &n, a.data(), &lda, tau.data(), work.data(), &lwork, &info);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "tau", Kind::kOther, tau);
  Add(r, "a", Kind::kOther, a);
  return r;
}

template <typename T>
Result Geqp3(const BlasLibrary& lib, std::vector<T> a, I m, I n, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(m, 1);
  // Free columns, except that every third one is moved to the front.
  std::vector<I> jpvt(std::max<I>(n, 1));
  for (I j = 0; j < n; ++j) jpvt[j] = j % 3 == 1 ? 1 : 0;
  std::vector<T> tau(std::max<I>(std::min(m, n), 1));
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "geqp3_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const I*, const I*, T*, const I*, I*, T*, T*, const I*,
                       R*, I*);
    std::vector<R> rwork(2 * std::max<I>(n, 1));
    lib.Get<F>(name.c_str())(&m, &n, a.data(), &lda, jpvt.data(), tau.data(),
                             work.data(), &lwork, rwork.data(), &info);
  } else {
    using F =
        void (*)(const I*, const I*, T*, const I*, I*, T*, T*, const I*, I*);
    lib.Get<F>(name.c_str())(&m, &n, a.data(), &lda, jpvt.data(), tau.data(),
                             work.data(), &lwork, &info);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "jpvt", Kind::kInteger, jpvt);
  Add(r, "tau", Kind::kOther, tau);
  Add(r, "a", Kind::kOther, a);
  return r;
}

// syev and heev.
template <typename T>
Result Syev(const BlasLibrary& lib, std::vector<T> a, I n, char jobz, char uplo,
            I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1);
  std::vector<R> w(std::max<I>(n, 1));
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, const I*, T*, const I*, R*, T*,
                       const I*, R*, I*, size_t, size_t);
    std::vector<R> rwork(std::max<I>(3 * n - 2, 1));
    lib.Get<F>((std::string(Prefix<T>()) + "heev_").c_str())(
        &jobz, &uplo, &n, a.data(), &lda, w.data(), work.data(), &lwork,
        rwork.data(), &info, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const I*, T*, const I*, R*, T*,
                       const I*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "syev_").c_str())(
        &jobz, &uplo, &n, a.data(), &lda, w.data(), work.data(), &lwork, &info,
        1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "w", Kind::kValues, w);
  // With JOBZ = 'N', A is destroyed.
  if (jobz == 'V') Add(r, "z", Kind::kOther, a);
  return r;
}

// gesvd (jobu, jobvt) and gesdd (jobz in jobu).
template <typename T>
Result Svd(const BlasLibrary& lib, bool dc, std::vector<T> a, I m, I n,
           char jobu, char jobvt, I lwork) {
  using R = Real<T>;
  const I mn = std::min(m, n), mx = std::max(m, n);
  const I lda = std::max<I>(m, 1), ldu = std::max<I>(m, 1);
  const I ldvt = std::max<I>(n, 1);
  std::vector<R> s(std::max<I>(mn, 1));
  std::vector<T> u(Size(m, m)), vt(Size(n, n));
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<I> iwork(std::max<I>(8 * mn, 1));
  std::vector<R> rwork(std::max<size_t>(
      {size_t{1}, 7 * static_cast<size_t>(mn),
       5 * static_cast<size_t>(mn) * mn + 5 * static_cast<size_t>(mn),
       2 * static_cast<size_t>(mx) * mn + 2 * static_cast<size_t>(mn) * mn +
           mn}));
  I info = 0;
  const std::string name =
      std::string(Prefix<T>()) + (dc ? "gesdd_" : "gesvd_");
  if (dc) {
    if constexpr (kComplex<T>) {
      using F =
          void (*)(const char*, const I*, const I*, T*, const I*, R*, T*,
                   const I*, T*, const I*, T*, const I*, R*, I*, I*, size_t);
      lib.Get<F>(name.c_str())(&jobu, &m, &n, a.data(), &lda, s.data(),
                               u.data(), &ldu, vt.data(), &ldvt, work.data(),
                               &lwork, rwork.data(), iwork.data(), &info, 1);
    } else {
      using F = void (*)(const char*, const I*, const I*, T*, const I*, R*, T*,
                         const I*, T*, const I*, T*, const I*, I*, I*, size_t);
      lib.Get<F>(name.c_str())(&jobu, &m, &n, a.data(), &lda, s.data(),
                               u.data(), &ldu, vt.data(), &ldvt, work.data(),
                               &lwork, iwork.data(), &info, 1);
    }
  } else {
    if constexpr (kComplex<T>) {
      using F = void (*)(const char*, const char*, const I*, const I*, T*,
                         const I*, R*, T*, const I*, T*, const I*, T*, const I*,
                         R*, I*, size_t, size_t);
      lib.Get<F>(name.c_str())(&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(),
                               u.data(), &ldu, vt.data(), &ldvt, work.data(),
                               &lwork, rwork.data(), &info, 1, 1);
    } else {
      using F = void (*)(const char*, const char*, const I*, const I*, T*,
                         const I*, R*, T*, const I*, T*, const I*, T*, const I*,
                         I*, size_t, size_t);
      lib.Get<F>(name.c_str())(&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(),
                               u.data(), &ldu, vt.data(), &ldvt, work.data(),
                               &lwork, &info, 1, 1);
    }
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "s", Kind::kValues, s);
  Add(r, "u", Kind::kOther, u);
  Add(r, "vt", Kind::kOther, vt);
  // A holds U or V^T with JOB 'O' and is destroyed otherwise.
  if (jobu == 'O' || jobvt == 'O') Add(r, "a", Kind::kOther, a);
  return r;
}

// The eigenvalues are only compared when `compare_w` (the complex ones of
// the graded matrices are too ill-conditioned).
template <typename T>
Result Gees(const BlasLibrary& lib, std::vector<T> a, I n, char sort,
            bool compare_w, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1);
  const char jobvs = 'V';
  std::vector<T> vs(Size(n, n));
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<I> bwork(std::max<I>(n, 1));
  I sdim = 0, info = 0;
  Result r;
  const std::string name = std::string(Prefix<T>()) + "gees_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, I (*)(const R*), const I*, T*,
                       const I*, I*, T*, T*, const I*, T*, const I*, R*, I*, I*,
                       size_t, size_t);
    std::vector<T> w(std::max<I>(n, 1));
    std::vector<R> rwork(std::max<I>(n, 1));
    lib.Get<F>(name.c_str())(&jobvs, &sort, &SelectComplex<R>, &n, a.data(),
                             &lda, &sdim, w.data(), vs.data(), &lda,
                             work.data(), &lwork, rwork.data(), bwork.data(),
                             &info, 1, 1);
    Add(r, "w", compare_w ? Kind::kEigenvalues : Kind::kOther, w.data(),
        static_cast<size_t>(n));
  } else {
    using F = void (*)(const char*, const char*, I (*)(const R*, const R*),
                       const I*, T*, const I*, I*, R*, R*, T*, const I*, T*,
                       const I*, I*, I*, size_t, size_t);
    std::vector<R> wr(std::max<I>(n, 1)), wi(std::max<I>(n, 1));
    lib.Get<F>(name.c_str())(&jobvs, &sort, &SelectReal<R>, &n, a.data(), &lda,
                             &sdim, wr.data(), wi.data(), vs.data(), &lda,
                             work.data(), &lwork, bwork.data(), &info, 1, 1);
    std::vector<std::complex<R>> w(static_cast<size_t>(n));
    for (I i = 0; i < n; ++i) w[i] = std::complex<R>(wr[i], wi[i]);
    Add(r, "w", Kind::kEigenvalues, w);
  }
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "sdim", sdim);
  Add(r, "t", Kind::kOther, a);
  Add(r, "vs", Kind::kOther, vs);
  return r;
}

// ---------------------------------------------------------------------------
// Linear systems. Routines that take a factorization get the same one in
// both libraries (hwyBLAS's xGETRF), so that they are compared bitwise; the
// drivers that factor (xGESV, xGESVX with FACT = 'N' or 'E') compute their
// factors with the leaf kernels of xGETRF2, so their factors and solutions
// are compared with the tolerance.

// The LU factorization of a (by hwyBLAS's xGETRF) and its pivots.
template <typename T>
struct Lu {
  std::vector<T> lu;
  std::vector<I> ipiv;
};

template <typename T>
Lu<T> Factor(const BlasLibrary& hwy, const std::vector<T>& a, I n) {
  using F = void (*)(const I*, const I*, T*, const I*, I*, I*);
  Lu<T> f{a, std::vector<I>(std::max<I>(n, 1))};
  const I lda = std::max<I>(n, 1);
  I info = 0;
  hwy.Get<F>((std::string(Prefix<T>()) + "getrf_").c_str())(
      &n, &n, f.lu.data(), &lda, f.ipiv.data(), &info);
  return f;
}

template <typename T>
Result Gesv(const BlasLibrary& lib, std::vector<T> a, std::vector<T> b, I n,
            I nrhs) {
  using F = void (*)(const I*, const I*, T*, const I*, I*, T*, const I*, I*);
  const I ld = std::max<I>(n, 1);
  std::vector<I> ipiv(ld);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "gesv_").c_str())(
      &n, &nrhs, a.data(), &ld, ipiv.data(), b.data(), &ld, &info);
  Result r;
  AddInt(r, "info", info);
  Add(r, "ipiv", Kind::kInteger, ipiv);
  Add(r, "lu", Kind::kRounded, a);
  Add(r, "x", Kind::kRounded, b);
  return r;
}

template <typename T>
Result Getrs(const BlasLibrary& lib, const Lu<T>& f, std::vector<T> b, I n,
             I nrhs, char trans) {
  using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                     const I*, T*, const I*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "getrs_").c_str())(
      &trans, &n, &nrhs, f.lu.data(), &ld, f.ipiv.data(), b.data(), &ld, &info,
      1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "x", Kind::kOther, b);
  return r;
}

template <typename T>
Result Getri(const BlasLibrary& lib, const Lu<T>& f, I n, I lwork) {
  using F = void (*)(const I*, T*, const I*, const I*, T*, const I*, I*);
  const I ld = std::max<I>(n, 1);
  std::vector<T> a = f.lu, work(std::max<I>(lwork, 1));
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "getri_").c_str())(
      &n, a.data(), &ld, f.ipiv.data(), work.data(), &lwork, &info);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "inv", Kind::kOther, a);
  return r;
}

template <typename T>
Result Gecon(const BlasLibrary& lib, const Lu<T>& f, I n, char norm,
             Real<T> anorm) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  R rcond = -1;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "gecon_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const T*, const I*, const R*, R*,
                       T*, R*, I*, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(2 * ld);
    lib.Get<F>(name.c_str())(&norm, &n, f.lu.data(), &ld, &anorm, &rcond,
                             work.data(), rwork.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const T*, const I*, const R*, R*,
                       T*, I*, I*, size_t);
    std::vector<T> work(4 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&norm, &n, f.lu.data(), &ld, &anorm, &rcond,
                             work.data(), iwork.data(), &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "rcond", Kind::kValues, &rcond, 1);
  return r;
}

template <typename T>
Result Geequ(const BlasLibrary& lib, const std::vector<T>& a, I m, I n) {
  using R = Real<T>;
  using F =
      void (*)(const I*, const I*, const T*, const I*, R*, R*, R*, R*, R*, I*);
  const I lda = std::max<I>(m, 1);
  std::vector<R> rs(std::max<I>(m, 1)), cs(std::max<I>(n, 1));
  R v[3] = {-1, -1, -1};
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "geequ_").c_str())(
      &m, &n, a.data(), &lda, rs.data(), cs.data(), &v[0], &v[1], &v[2], &info);
  Result r;
  AddInt(r, "info", info);
  // R and C are complete only if INFO = 0.
  if (info == 0) {
    Add(r, "r", Kind::kOther, rs);
    Add(r, "c", Kind::kOther, cs);
  }
  Add(r, "rowcnd colcnd amax", Kind::kOther, v, info == 0 ? 3 : 0);
  return r;
}

// xGERFS (with the given factorization) refining the solution x.
template <typename T>
Result Gerfs(const BlasLibrary& lib, const std::vector<T>& a, const Lu<T>& f,
             const std::vector<T>& b, std::vector<T> x, I n, I nrhs,
             char trans) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<R> ferr(std::max<I>(nrhs, 1)), berr(std::max<I>(nrhs, 1));
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "gerfs_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const T*, const I*, const I*, const T*, const I*, T*,
                       const I*, R*, R*, T*, R*, I*, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&trans, &n, &nrhs, a.data(), &ld, f.lu.data(), &ld,
                             f.ipiv.data(), b.data(), &ld, x.data(), &ld,
                             ferr.data(), berr.data(), work.data(),
                             rwork.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const T*, const I*, const I*, const T*, const I*, T*,
                       const I*, R*, R*, T*, I*, I*, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&trans, &n, &nrhs, a.data(), &ld, f.lu.data(), &ld,
                             f.ipiv.data(), b.data(), &ld, x.data(), &ld,
                             ferr.data(), berr.data(), work.data(),
                             iwork.data(), &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "x", Kind::kOther, x);
  Add(r, "ferr", Kind::kValues, ferr);
  Add(r, "berr", Kind::kValues, berr);
  return r;
}

// xGESVX. For FACT = 'F', af and ipiv are the given factorization (of the
// matrix as given, EQUED = 'N'); the results are then compared bitwise.
template <typename T>
Result Gesvx(const BlasLibrary& lib, char fact, char trans, std::vector<T> a,
             const Lu<T>& f, std::vector<T> b, I n, I nrhs) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> af = f.lu, x(Size(n, nrhs));
  std::vector<I> ipiv = f.ipiv;
  std::vector<R> rs(ld, 1), cs(ld, 1), ferr(std::max<I>(nrhs, 1)),
      berr(std::max<I>(nrhs, 1));
  char equed = 'N';
  R rcond = -1, rpvgrw = -1;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "gesvx_";
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const I*, const I*, T*, const I*, T*,
                 const I*, I*, char*, R*, R*, T*, const I*, T*, const I*, R*,
                 R*, R*, T*, R*, I*, size_t, size_t, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(2 * ld);
    lib.Get<F>(name.c_str())(
        &fact, &trans, &n, &nrhs, a.data(), &ld, af.data(), &ld, ipiv.data(),
        &equed, rs.data(), cs.data(), b.data(), &ld, x.data(), &ld, &rcond,
        ferr.data(), berr.data(), work.data(), rwork.data(), &info, 1, 1, 1);
    rpvgrw = rwork[0];
  } else {
    using F =
        void (*)(const char*, const char*, const I*, const I*, T*, const I*, T*,
                 const I*, I*, char*, R*, R*, T*, const I*, T*, const I*, R*,
                 R*, R*, T*, I*, I*, size_t, size_t, size_t);
    std::vector<T> work(4 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(
        &fact, &trans, &n, &nrhs, a.data(), &ld, af.data(), &ld, ipiv.data(),
        &equed, rs.data(), cs.data(), b.data(), &ld, x.data(), &ld, &rcond,
        ferr.data(), berr.data(), work.data(), iwork.data(), &info, 1, 1, 1);
    rpvgrw = work[0];
  }
  const Kind factored = fact == 'F' ? Kind::kOther : Kind::kRounded;
  const Kind value = fact == 'F' ? Kind::kValues : Kind::kRounded;
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "equed", equed);
  Add(r, "ipiv", Kind::kInteger, ipiv);
  Add(r, "r", Kind::kOther, rs);
  Add(r, "c", Kind::kOther, cs);
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  Add(r, "af", factored, af);
  Add(r, "x", factored, x);
  Add(r, "rcond", value, &rcond, 1);
  Add(r, "rpvgrw", value, &rpvgrw, 1);
  Add(r, "ferr", value, ferr);
  Add(r, "berr", value, berr);
  return r;
}

// ---------------------------------------------------------------------------
// Triangular matrices

// The triangle uplo of a (n x n, leading dimension n) with the other
// elements zero; the diagonal elements are moved away from zero.
template <typename T>
std::vector<T> Triangle(std::vector<T> a, I n, char uplo) {
  for (I j = 0; j < n; ++j) {
    for (I i = 0; i < n; ++i) {
      T& x = a[i + static_cast<size_t>(j) * n];
      if (uplo == 'U' ? i > j : i < j) x = T(0);
      if (i == j) x += T(std::real(x) >= 0 ? 1 : -1);
    }
  }
  return a;
}

template <typename T>
Result Trtri(const BlasLibrary& lib, std::vector<T> a, I n, char uplo,
             char diag) {
  using F = void (*)(const char*, const char*, const I*, T*, const I*, I*,
                     size_t, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "trtri_").c_str())(
      &uplo, &diag, &n, a.data(), &ld, &info, 1, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "inv", Kind::kOther, a);
  return r;
}

template <typename T>
Result Trtrs(const BlasLibrary& lib, const std::vector<T>& a, std::vector<T> b,
             I n, I nrhs, char uplo, char trans, char diag) {
  using F =
      void (*)(const char*, const char*, const char*, const I*, const I*,
               const T*, const I*, T*, const I*, I*, size_t, size_t, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "trtrs_").c_str())(
      &uplo, &trans, &diag, &n, &nrhs, a.data(), &ld, b.data(), &ld, &info, 1,
      1, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "x", Kind::kOther, b);
  return r;
}

template <typename T>
Result Latrs(const BlasLibrary& lib, const std::vector<T>& a, std::vector<T> x,
             I n, char uplo, char trans, char diag) {
  using R = Real<T>;
  using F = void (*)(const char*, const char*, const char*, const char*,
                     const I*, const T*, const I*, T*, R*, R*, I*, size_t,
                     size_t, size_t, size_t);
  const I ld = std::max<I>(n, 1);
  std::vector<R> cnorm(ld);
  R scale = -1;
  I info = 0;
  const char normin = 'N';
  lib.Get<F>((std::string(Prefix<T>()) + "latrs_").c_str())(
      &uplo, &trans, &diag, &normin, &n, a.data(), &ld, x.data(), &scale,
      cnorm.data(), &info, 1, 1, 1, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "scale", Kind::kValues, &scale, 1);
  Add(r, "x", Kind::kOther, x);
  Add(r, "cnorm", Kind::kValues, cnorm);
  return r;
}

template <typename T>
Result Trcon(const BlasLibrary& lib, const std::vector<T>& a, I n, char norm,
             char uplo, char diag) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  R rcond = -1;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "trcon_";
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const char*, const I*, const T*,
                 const I*, R*, T*, R*, I*, size_t, size_t, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&norm, &uplo, &diag, &n, a.data(), &ld, &rcond,
                             work.data(), rwork.data(), &info, 1, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const char*, const I*, const T*,
                 const I*, R*, T*, I*, I*, size_t, size_t, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&norm, &uplo, &diag, &n, a.data(), &ld, &rcond,
                             work.data(), iwork.data(), &info, 1, 1, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "rcond", Kind::kValues, &rcond, 1);
  return r;
}

template <typename T>
Result Trrfs(const BlasLibrary& lib, const std::vector<T>& a,
             const std::vector<T>& b, const std::vector<T>& x, I n, I nrhs,
             char uplo, char trans, char diag) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<R> ferr(std::max<I>(nrhs, 1)), berr(std::max<I>(nrhs, 1));
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "trrfs_";
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const char*, const I*, const I*,
                 const T*, const I*, const T*, const I*, const T*, const I*, R*,
                 R*, T*, R*, I*, size_t, size_t, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &trans, &diag, &n, &nrhs, a.data(), &ld,
                             b.data(), &ld, x.data(), &ld, ferr.data(),
                             berr.data(), work.data(), rwork.data(), &info, 1,
                             1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const char*, const I*, const I*,
                 const T*, const I*, const T*, const I*, const T*, const I*, R*,
                 R*, T*, I*, I*, size_t, size_t, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &trans, &diag, &n, &nrhs, a.data(), &ld,
                             b.data(), &ld, x.data(), &ld, ferr.data(),
                             berr.data(), work.data(), iwork.data(), &info, 1,
                             1, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "ferr", Kind::kValues, ferr);
  Add(r, "berr", Kind::kValues, berr);
  return r;
}

template <typename T>
Result Lantr(const BlasLibrary& lib, const std::vector<T>& a, I m, I n,
             char norm, char uplo, char diag) {
  using R = Real<T>;
  using F = R (*)(const char*, const char*, const char*, const I*, const I*,
                  const T*, const I*, R*, size_t, size_t, size_t);
  const I lda = std::max<I>(m, 1);
  std::vector<R> work(std::max<I>(m, 1));
  const R value = lib.Get<F>((std::string(Prefix<T>()) + "lantr_").c_str())(
      &norm, &uplo, &diag, &m, &n, a.data(), &lda, work.data(), 1, 1, 1);
  Result r;
  Add(r, "norm", Kind::kOther, &value, 1);
  return r;
}

// The tally of one driver and precision.
struct Tally {
  int cases = 0;
  int identical = 0;   // all outputs bitwise identical
  int mismatches = 0;  // differences that count as failures
  double worst = 0;    // largest difference of the compared values, in eps
};

bool Same(std::complex<double> x, std::complex<double> y) {
  const auto same = [](double a, double b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0 ||
           (std::isnan(a) && std::isnan(b));
  };
  return same(x.real(), y.real()) && same(x.imag(), y.imag());
}

// The largest difference of `a` and `b`, relative to the largest magnitude
// in `b`, in units of eps; as sets (each value of `a` against the nearest
// unused value of `b`) when `as_set`.
double Difference(const std::vector<std::complex<double>>& a,
                  const std::vector<std::complex<double>>& b, bool as_set,
                  double eps) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double scale = std::numeric_limits<double>::min();
  for (const auto& y : b) scale = std::max(scale, std::abs(y));
  double worst = 0;
  std::vector<bool> used(b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    size_t k = i;
    if (as_set) {
      double best = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < b.size(); ++j) {
        if (!used[j] && std::abs(a[i] - b[j]) < best) {
          best = std::abs(a[i] - b[j]);
          k = j;
        }
      }
      used[k] = true;
    }
    const double d = std::abs(a[i] - b[k]);
    worst = std::max(
        worst, std::isnan(d) ? std::numeric_limits<double>::infinity() : d);
  }
  return worst / scale / eps;
}

// Compares the outputs of one case and records the result in `tally`.
template <typename T>
void Compare(const std::string& routine, const std::string& what, I n,
             const Result& hwy, const Result& ref, Tally& tally) {
  const double eps = std::numeric_limits<Real<T>>::epsilon();
  // Tolerance for the complex values: the differences start in the last bit
  // and are amplified by the conditioning of the problem.
  const double tol = 1e3 * std::max<I>(n, 1);
  ++tally.cases;
  bool identical = hwy.outputs.size() == ref.outputs.size();
  bool failed = !identical;
  for (size_t k = 0; identical && k < hwy.outputs.size(); ++k) {
    const Output& x = hwy.outputs[k];
    const Output& y = ref.outputs[k];
    bool same = x.values.size() == y.values.size();
    for (size_t i = 0; same && i < x.values.size(); ++i) {
      same = Same(x.values[i], y.values[i]);
    }
    if (same) continue;
    identical = false;
    std::string detail;
    if ((!kComplex<T> && x.kind != Kind::kRounded) ||
        x.kind == Kind::kInteger) {
      failed = true;
      size_t i = 0;
      while (i < x.values.size() && i < y.values.size() &&
             Same(x.values[i], y.values[i])) {
        ++i;
      }
      char buf[160];
      if (i < x.values.size() && i < y.values.size()) {
        snprintf(buf, sizeof(buf),
                 "%s(%zu) = %.17g%+.17gi, reference %.17g%+.17gi",
                 x.name.c_str(), i + 1, x.values[i].real(), x.values[i].imag(),
                 y.values[i].real(), y.values[i].imag());
      } else {
        snprintf(buf, sizeof(buf), "%s: size %zu, reference %zu",
                 x.name.c_str(), x.values.size(), y.values.size());
      }
      detail = buf;
    } else if (x.kind != Kind::kOther) {
      const double d =
          Difference(x.values, y.values, x.kind == Kind::kEigenvalues, eps);
      tally.worst = std::max(tally.worst, d);
      if (d > tol) {
        failed = true;
        char buf[120];
        snprintf(buf, sizeof(buf), "%s differs by %.3g eps", x.name.c_str(), d);
        detail = buf;
      }
    }
    if (!detail.empty() && tally.mismatches < 5) {
      printf("  %s%s %s: %s\n", Prefix<T>(), routine.c_str(), what.c_str(),
             detail.c_str());
    }
  }
  if (identical) ++tally.identical;
  if (failed) ++tally.mismatches;
}

std::map<std::string, Tally>* g_tallies;

// Runs one case in both libraries with the same workspace (the larger of
// the two queries) and compares the results.
template <typename T, typename Run>
void Case(const BlasLibrary& hwy, const BlasLibrary& ref,
          const std::string& routine, const std::string& what, I n,
          const Run& run) {
  const double lwork = std::max(run(hwy, -1).lwork, run(ref, -1).lwork);
  const I lw = static_cast<I>(std::max(lwork, 1.0));
  const Result x = run(hwy, lw);
  const Result y = run(ref, lw);
  Compare<T>(routine, what, n, x, y,
             (*g_tallies)[std::string(Prefix<T>()) + routine]);
}

// The linear systems and triangular routines for matrices of order n:
// random, graded (also for equilibration) and with small integers.
template <typename T>
void RunLinear(const BlasLibrary& hwy, const BlasLibrary& ref, I n) {
  using R = Real<T>;
  char what[64];
  const I nrhs = 3;
  const std::vector<T> b = TestMatrix<T>(0, n, nrhs);
  // Kinds 0, 2 and 4 of TestMatrix, and 5: random with rows and columns
  // scaled by powers of 2 (for equilibration). The drivers that factor
  // (with the leaf kernels of xGETRF2, which round differently) only get the
  // matrices with clear pivots (0 and 5): in the graded ones and those with
  // ties, the rounding can change the pivots.
  for (const int kind : {0, 2, 4, 5}) {
    std::vector<T> a = TestMatrix<T>(kind == 5 ? 0 : kind, n, n);
    if (kind == 5) {
      for (I j = 0; j < n; ++j) {
        for (I i = 0; i < n; ++i) {
          a[i + static_cast<size_t>(j) * n] *=
              static_cast<R>(std::ldexp(1.0, 3 * (i % 7) - 2 * (j % 5)));
        }
      }
    }
    const bool clear_pivots = kind == 0 || kind == 5;
    snprintf(what, sizeof(what), "n=%d kind=%d", int(n), kind);
    if (clear_pivots) {
      Case<T>(hwy, ref, "gesv", what, n, [&](const BlasLibrary& lib, I) {
        return Gesv(lib, a, b, n, nrhs);
      });
    }
    Case<T>(hwy, ref, "geequ", what, n,
            [&](const BlasLibrary& lib, I) { return Geequ(lib, a, n, n); });
    const std::vector<T> tall = TestMatrix<T>(kind, 2 * n + 1, n);
    Case<T>(hwy, ref, "geequ", what, n, [&](const BlasLibrary& lib, I) {
      return Geequ(lib, tall, 2 * n + 1, n);
    });
    const Lu<T> f = Factor(hwy, a, n);
    Case<T>(hwy, ref, "getri", what, n,
            [&](const BlasLibrary& lib, I lw) { return Getri(lib, f, n, lw); });
    for (const char norm : {'1', 'I'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d norm=%c", int(n), kind, norm);
      Case<T>(hwy, ref, "gecon", what, n, [&](const BlasLibrary& lib, I) {
        return Gecon(lib, f, n, norm, R(3));
      });
    }
    for (const char trans : {'N', 'T', 'C'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d trans=%c", int(n), kind,
               trans);
      Case<T>(hwy, ref, "getrs", what, n, [&](const BlasLibrary& lib, I) {
        return Getrs(lib, f, b, n, nrhs, trans);
      });
      // Refinement starting from B (several steps).
      Case<T>(hwy, ref, "gerfs", what, n, [&](const BlasLibrary& lib, I) {
        return Gerfs(lib, a, f, b, b, n, nrhs, trans);
      });
      for (const char fact : {'N', 'E', 'F'}) {
        if (fact != 'F' && !clear_pivots) continue;
        snprintf(what, sizeof(what), "n=%d kind=%d trans=%c fact=%c", int(n),
                 kind, trans, fact);
        Case<T>(hwy, ref, "gesvx", what, n, [&](const BlasLibrary& lib, I) {
          return Gesvx(lib, fact, trans, a, f, b, n, nrhs);
        });
      }
    }
  }

  // Triangular matrices.
  const R smlnum =
      std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon();
  for (const char uplo : {'U', 'L'}) {
    for (const char diag : {'N', 'U'}) {
      for (const int kind : {0, 2, 4, 5, 6}) {
        // 5: growth (the solution would overflow), 6: one column near
        // overflow (its norm overflows): the scaling paths of xLATRS.
        std::vector<T> t =
            Triangle(TestMatrix<T>(kind < 5 ? kind : 0, n, n), n, uplo);
        const I jc = uplo == 'U' ? n - 1 : 0;
        for (I j = 0; j < n; ++j) {
          for (I i = 0; i < n; ++i) {
            const bool off = uplo == 'U' ? i < j : i > j;
            T& x = t[i + static_cast<size_t>(j) * n];
            if (kind == 5 && off) {
              x = T(-std::pow(1 / smlnum, R(2) / static_cast<R>(n)));
            }
            if (kind == 5 && i == j) x = T(1);
            if (kind == 6 && j == jc && (off || i == j)) {
              x = T(std::numeric_limits<R>::max() / 4);
              if constexpr (kComplex<T>) x *= T(R(0.6), R(0.8));
            }
          }
        }
        snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c diag=%c", int(n),
                 kind, uplo, diag);
        if (kind < 5) {
          Case<T>(hwy, ref, "trtri", what, n, [&](const BlasLibrary& lib, I) {
            return Trtri(lib, t, n, uplo, diag);
          });
          for (const char norm : {'1', 'I'}) {
            snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c diag=%c norm=%c",
                     int(n), kind, uplo, diag, norm);
            Case<T>(hwy, ref, "trcon", what, n, [&](const BlasLibrary& lib, I) {
              return Trcon(lib, t, n, norm, uplo, diag);
            });
          }
          const std::vector<T> trap = TestMatrix<T>(kind, n + 2, n);
          for (const char norm : {'M', '1', 'I', 'F'}) {
            snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c diag=%c norm=%c",
                     int(n), kind, uplo, diag, norm);
            Case<T>(hwy, ref, "lantr", what, n, [&](const BlasLibrary& lib, I) {
              return Lantr(lib, trap, n + 2, n, norm, uplo, diag);
            });
          }
        }
        for (const char trans : {'N', 'T', 'C'}) {
          snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c diag=%c trans=%c",
                   int(n), kind, uplo, diag, trans);
          Case<T>(hwy, ref, "latrs", what, n, [&](const BlasLibrary& lib, I) {
            return Latrs(lib, t, b, n, uplo, trans, diag);
          });
          if (kind >= 5) continue;
          Case<T>(hwy, ref, "trtrs", what, n, [&](const BlasLibrary& lib, I) {
            return Trtrs(lib, t, b, n, nrhs, uplo, trans, diag);
          });
          // Not for complex graded triangles: their solutions overflow in
          // single precision, and on infinities flang's REAL*COMPLEX (a
          // complex multiplication, 0*Inf = NaN) gives NaN where the port's
          // multiplication of the parts does not.
          if (kComplex<T> && kind == 2) continue;
          Case<T>(hwy, ref, "trrfs", what, n, [&](const BlasLibrary& lib, I) {
            return Trrfs(lib, t, b, b, n, nrhs, uplo, trans, diag);
          });
        }
      }
    }
  }
}

template <typename T>
void RunAll(const BlasLibrary& hwy, const BlasLibrary& ref, int cases) {
  static const I kSizes[] = {0, 1, 2, 3, 5, 8, 16, 31, 64, 65, 100, 150, 200};
  int count = 0;
  for (const I s : kSizes) {
    if (count++ >= cases) break;
    char what[64];
    // Square, tall and wide shapes (tall enough for the QR-first paths of
    // the SVDs, which start at m >= 1.6 n).
    const I shapes[][2] = {{s, s}, {2 * s + 1, s}, {s, 2 * s + 1}};
    for (const auto& shape : shapes) {
      const I m = shape[0], n = shape[1];
      snprintf(what, sizeof(what), "m=%d n=%d", int(m), int(n));
      const std::vector<T> a = TestMatrix<T>(s % 5 == 4 ? 4 : 0, m, n);
      const I mn = std::max(m, n);
      Case<T>(hwy, ref, "getrf", what, mn, [&](const BlasLibrary& lib, I lw) {
        return Getrf(lib, a, m, n, lw);
      });
      Case<T>(hwy, ref, "geqrf", what, mn, [&](const BlasLibrary& lib, I lw) {
        return Geqrf(lib, "geqrf_", a, m, n, lw);
      });
      Case<T>(hwy, ref, "gelqf", what, mn, [&](const BlasLibrary& lib, I lw) {
        return Geqrf(lib, "gelqf_", a, m, n, lw);
      });
      Case<T>(hwy, ref, "geqp3", what, mn, [&](const BlasLibrary& lib, I lw) {
        return Geqp3(lib, a, m, n, lw);
      });
      static const char kSvdJobs[][2] = {{'A', 'A'}, {'S', 'S'}, {'O', 'S'},
                                         {'S', 'O'}, {'N', 'N'}, {'O', 'N'},
                                         {'N', 'O'}, {'A', 'N'}};
      for (const auto& job : kSvdJobs) {
        snprintf(what, sizeof(what), "m=%d n=%d jobu=%c jobvt=%c", int(m),
                 int(n), job[0], job[1]);
        Case<T>(hwy, ref, "gesvd", what, mn, [&](const BlasLibrary& lib, I lw) {
          return Svd(lib, false, a, m, n, job[0], job[1], lw);
        });
      }
      for (const char jobz : {'A', 'S', 'O', 'N'}) {
        snprintf(what, sizeof(what), "m=%d n=%d jobz=%c", int(m), int(n), jobz);
        Case<T>(hwy, ref, "gesdd", what, mn, [&](const BlasLibrary& lib, I lw) {
          return Svd(lib, true, a, m, n, jobz, ' ', lw);
        });
      }
    }
    const I n = s;
    for (const char uplo : {'L', 'U'}) {
      snprintf(what, sizeof(what), "n=%d uplo=%c", int(n), uplo);
      const std::vector<T> spd = PositiveDefinite<T>(n);
      Case<T>(hwy, ref, "potrf", what, n, [&](const BlasLibrary& lib, I) {
        return Potrf(lib, spd, n, uplo);
      });
      const std::vector<T> h = Hermitian<T>(n);
      for (const char jobz : {'V', 'N'}) {
        snprintf(what, sizeof(what), "n=%d jobz=%c uplo=%c", int(n), jobz,
                 uplo);
        Case<T>(hwy, ref, kComplex<T> ? "heev" : "syev", what, n,
                [&](const BlasLibrary& lib, I lw) {
                  return Syev(lib, h, n, jobz, uplo, lw);
                });
      }
    }
    for (int kind = 0; kind < 5; ++kind) {
      const std::vector<T> g = TestMatrix<T>(kind, n, n);
      for (const char sort : {'N', 'S'}) {
        snprintf(what, sizeof(what), "n=%d kind=%d sort=%c", int(n), kind,
                 sort);
        Case<T>(hwy, ref, "gees", what, n, [&](const BlasLibrary& lib, I lw) {
          return Gees(lib, g, n, sort, kind != 2, lw);
        });
      }
    }
    RunLinear<T>(hwy, ref, n);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: compare_reference <hwyblas library> <reference liblapack> "
            "[sizes, at most 13]\n");
    return 2;
  }
  const BlasLibrary hwy("hwyBLAS", argv[1]), ref("reference", argv[2]);
  if (!hwy.ok() || !ref.ok()) return 2;
  const auto set_ilaenv =
      hwy.Get<hwyblas_ilaenv_function (*)(hwyblas_ilaenv_function)>(
          "hwyblas_set_ilaenv");
  const auto ref_ilaenv = ref.Get<hwyblas_ilaenv_function>("ilaenv_");
  if (!set_ilaenv || !ref_ilaenv || !ref.Find("dgees_")) {
    fprintf(stderr, "hwyblas_set_ilaenv or the reference routines not found\n");
    return 2;
  }
  set_ilaenv(ref_ilaenv);
  const int cases = argc > 3 ? atoi(argv[3]) : 13;
  std::map<std::string, Tally> tallies;
  g_tallies = &tallies;
  RunAll<float>(hwy, ref, cases);
  RunAll<double>(hwy, ref, cases);
  RunAll<std::complex<float>>(hwy, ref, cases);
  RunAll<std::complex<double>>(hwy, ref, cases);

  printf("%-8s %6s %10s %11s %13s\n", "routine", "cases", "identical",
         "mismatches", "max diff/eps");
  int mismatches = 0;
  for (const char* p : {"s", "d", "c", "z"}) {
    for (const auto& [name, t] : tallies) {
      if (name[0] != p[0]) continue;
      printf("%-8s %6d %10d %11d", name.c_str(), t.cases, t.identical,
             t.mismatches);
      if (p[0] == 'c' || p[0] == 'z' || t.worst > 0) {
        printf(" %13.3g", t.worst);
      }
      printf("\n");
      mismatches += t.mismatches;
    }
  }
  printf("%s: %d mismatches\n", mismatches ? "FAILED" : "PASSED", mismatches);
  return mismatches ? 1 : 0;
}
