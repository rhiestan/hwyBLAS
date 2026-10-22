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
// xGETRF and the drivers that factor with it (xGESV, xGESVX with FACT =
// 'N' or 'E') get only matrices with clear pivots: on graded matrices and
// ties, the different rounding of the leaf kernels can change the pivots.
// The routines that take a factorization get the same one in both
// libraries. The same holds for xPOTRF: the routines that take a Cholesky
// factor get hwyBLAS's, and the drivers that factor with xPOTRF (xPOSV,
// xPOSVX with FACT = 'N' or 'E', xSYGV, xSYGVD and their xHE* routines)
// are compared with the tolerance. The
// Bunch-Kaufman routines (xSYTRF, xHETRF and the routines and drivers
// built on them) and xPSTRF have no such deviation and are compared
// bitwise.
//
// The eigenvectors and condition numbers of a Schur form (xTREVC3, xTRSNA)
// get the same Schur form (hwyBLAS's xGEES) in both libraries. xSTEIN gets
// the eigenvalues of each library's xSTEBZ (which are the same). xLARNV's
// distributions 3 to 5 use LOG, COS and EXP, of the C runtime in hwyBLAS
// and of flang's in the reference (the real results are the same).
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

// The eigenvalues (as complex numbers, also for real T) and their kind.
template <typename T>
void AddEigenvalues(Result& r, const std::vector<T>& w, bool compare_w) {
  Add(r, "w", compare_w ? Kind::kEigenvalues : Kind::kOther, w);
}
template <typename R>
void AddEigenvalues(Result& r, const std::vector<R>& wr,
                    const std::vector<R>& wi, I n) {
  std::vector<std::complex<R>> w(static_cast<size_t>(n));
  for (I i = 0; i < n; ++i) w[i] = std::complex<R>(wr[i], wi[i]);
  Add(r, "w", Kind::kEigenvalues, w);
}

template <typename T>
Result Geev(const BlasLibrary& lib, std::vector<T> a, I n, char jobvl,
            char jobvr, bool compare_w, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> vl(Size(n, n)), vr(Size(n, n));
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  Result r;
  const std::string name = std::string(Prefix<T>()) + "geev_";
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const I*, T*, const I*, T*, T*,
                 const I*, T*, const I*, T*, const I*, R*, I*, size_t, size_t);
    std::vector<T> w(static_cast<size_t>(n));
    std::vector<R> rwork(std::max<I>(2 * n, 1));
    lib.Get<F>(name.c_str())(&jobvl, &jobvr, &n, a.data(), &ld, w.data(),
                             vl.data(), &ld, vr.data(), &ld, work.data(),
                             &lwork, rwork.data(), &info, 1, 1);
    AddEigenvalues(r, w, compare_w);
  } else {
    using F =
        void (*)(const char*, const char*, const I*, T*, const I*, R*, R*, T*,
                 const I*, T*, const I*, T*, const I*, I*, size_t, size_t);
    std::vector<R> wr(std::max<I>(n, 1)), wi(std::max<I>(n, 1));
    lib.Get<F>(name.c_str())(&jobvl, &jobvr, &n, a.data(), &ld, wr.data(),
                             wi.data(), vl.data(), &ld, vr.data(), &ld,
                             work.data(), &lwork, &info, 1, 1);
    AddEigenvalues(r, wr, wi, n);
  }
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  if (jobvl == 'V') Add(r, "vl", Kind::kOther, vl);
  if (jobvr == 'V') Add(r, "vr", Kind::kOther, vr);
  return r;
}

template <typename T>
Result Geevx(const BlasLibrary& lib, std::vector<T> a, I n, char balanc,
             char jobvl, char jobvr, char sense, bool compare_w, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> vl(Size(n, n)), vr(Size(n, n));
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<R> scale(ld), rconde(ld), rcondv(ld);
  R abnrm = 0;
  I ilo = 0, ihi = 0, info = 0;
  Result r;
  const std::string name = std::string(Prefix<T>()) + "geevx_";
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const char*, const char*, const I*,
                 T*, const I*, T*, T*, const I*, T*, const I*, I*, I*, R*, R*,
                 R*, R*, T*, const I*, R*, I*, size_t, size_t, size_t, size_t);
    std::vector<T> w(static_cast<size_t>(n));
    std::vector<R> rwork(std::max<I>(2 * n, 1));
    lib.Get<F>(name.c_str())(
        &balanc, &jobvl, &jobvr, &sense, &n, a.data(), &ld, w.data(), vl.data(),
        &ld, vr.data(), &ld, &ilo, &ihi, scale.data(), &abnrm, rconde.data(),
        rcondv.data(), work.data(), &lwork, rwork.data(), &info, 1, 1, 1, 1);
    AddEigenvalues(r, w, compare_w);
  } else {
    using F = void (*)(const char*, const char*, const char*, const char*,
                       const I*, T*, const I*, R*, R*, T*, const I*, T*,
                       const I*, I*, I*, R*, R*, R*, R*, T*, const I*, I*, I*,
                       size_t, size_t, size_t, size_t);
    std::vector<R> wr(ld), wi(ld);
    std::vector<I> iwork(std::max<I>(2 * n - 2, 1));
    lib.Get<F>(name.c_str())(&balanc, &jobvl, &jobvr, &sense, &n, a.data(), &ld,
                             wr.data(), wi.data(), vl.data(), &ld, vr.data(),
                             &ld, &ilo, &ihi, scale.data(), &abnrm,
                             rconde.data(), rcondv.data(), work.data(), &lwork,
                             iwork.data(), &info, 1, 1, 1, 1);
    AddEigenvalues(r, wr, wi, n);
  }
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "ilo", ilo);
  AddInt(r, "ihi", ihi);
  Add(r, "scale", Kind::kValues, scale);
  Add(r, "abnrm", Kind::kValues, &abnrm, 1);
  if (jobvl == 'V') Add(r, "vl", Kind::kOther, vl);
  if (jobvr == 'V') Add(r, "vr", Kind::kOther, vr);
  const Kind cond = compare_w ? Kind::kValues : Kind::kOther;
  if (sense == 'E' || sense == 'B') Add(r, "rconde", cond, rconde);
  if (sense == 'V' || sense == 'B') Add(r, "rcondv", cond, rcondv);
  if (sense != 'N' || jobvl == 'V' || jobvr == 'V') {
    Add(r, "t", Kind::kOther, a);
  }
  return r;
}

// xGEESX with JOBVS = 'V' and SORT = 'S'; LIWORK is always large enough.
template <typename T>
Result Geesx(const BlasLibrary& lib, std::vector<T> a, I n, char sense,
             bool compare_w, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  const char jobvs = 'V', sort = 'S';
  std::vector<T> vs(Size(n, n));
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<I> bwork(ld);
  I sdim = 0, info = 0;
  R rconde = 0, rcondv = 0;
  Result r;
  const std::string name = std::string(Prefix<T>()) + "geesx_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, I (*)(const R*), const char*,
                       const I*, T*, const I*, I*, T*, T*, const I*, R*, R*, T*,
                       const I*, R*, I*, I*, size_t, size_t, size_t);
    std::vector<T> w(static_cast<size_t>(n));
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&jobvs, &sort, &SelectComplex<R>, &sense, &n,
                             a.data(), &ld, &sdim, w.data(), vs.data(), &ld,
                             &rconde, &rcondv, work.data(), &lwork,
                             rwork.data(), bwork.data(), &info, 1, 1, 1);
    AddEigenvalues(r, w, compare_w);
  } else {
    using F = void (*)(const char*, const char*, I (*)(const R*, const R*),
                       const char*, const I*, T*, const I*, I*, R*, R*, T*,
                       const I*, R*, R*, T*, const I*, I*, const I*, I*, I*,
                       size_t, size_t, size_t);
    std::vector<R> wr(ld), wi(ld);
    const I liwork = n * n + 1;
    std::vector<I> iwork(liwork);
    lib.Get<F>(name.c_str())(
        &jobvs, &sort, &SelectReal<R>, &sense, &n, a.data(), &ld, &sdim,
        wr.data(), wi.data(), vs.data(), &ld, &rconde, &rcondv, work.data(),
        &lwork, iwork.data(), &liwork, bwork.data(), &info, 1, 1, 1);
    AddEigenvalues(r, wr, wi, n);
  }
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "sdim", sdim);
  Add(r, "t", Kind::kOther, a);
  Add(r, "vs", Kind::kOther, vs);
  const Kind cond = compare_w ? Kind::kValues : Kind::kOther;
  if (sense == 'E' || sense == 'B') Add(r, "rconde", cond, &rconde, 1);
  if (sense == 'V' || sense == 'B') Add(r, "rcondv", cond, &rcondv, 1);
  return r;
}

// The Schur form T and Schur vectors Z of a (hwyBLAS's xGEES), the input of
// xTREVC3 and xTRSNA in both libraries.
template <typename T>
struct SchurForm {
  std::vector<T> t, z;
};

template <typename T>
SchurForm<T> Schur(const BlasLibrary& hwy, const std::vector<T>& a, I n) {
  Result q = Gees(hwy, a, n, 'N', true, -1);
  Result r = Gees(hwy, a, n, 'N', true, static_cast<I>(std::max(q.lwork, 1.0)));
  SchurForm<T> s;
  for (const Output& o : r.outputs) {
    std::vector<T>* v = o.name == "t" ? &s.t : o.name == "vs" ? &s.z : nullptr;
    if (v == nullptr) continue;
    for (const std::complex<double>& x : o.values) {
      if constexpr (kComplex<T>) {
        v->push_back(T(x));
      } else {
        v->push_back(static_cast<T>(x.real()));
      }
    }
  }
  return s;
}

// SELECT of xTREVC3 and xTRSNA: every third eigenvalue.
std::vector<I> EveryThird(I n) {
  std::vector<I> select(std::max<I>(n, 1));
  for (I i = 0; i < n; ++i) select[i] = i % 3 == 1;
  return select;
}

// xTREVC3 with SIDE = 'B': HOWMNY = 'A' and 'S' (every third eigenvalue)
// for the Schur form, 'B' with the Schur vectors in VL and VR.
template <typename T>
Result Trevc3(const BlasLibrary& lib, const SchurForm<T>& s, I n, char howmny,
              I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  const char side = 'B';
  std::vector<T> t = s.t;
  std::vector<T> vl = howmny == 'B' ? s.z : std::vector<T>(Size(n, n));
  std::vector<T> vr = vl;
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<I> select = howmny == 'S' ? EveryThird(n) : std::vector<I>(ld, 0);
  I m = 0, info = 0;
  Result r;
  const std::string name = std::string(Prefix<T>()) + "trevc3_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, const I*, const I*, T*,
                       const I*, T*, const I*, T*, const I*, const I*, I*, T*,
                       const I*, R*, const I*, I*, size_t, size_t);
    std::vector<R> rwork(ld);
    const I lrwork = lwork == -1 ? -1 : ld;
    lib.Get<F>(name.c_str())(&side, &howmny, select.data(), &n, t.data(), &ld,
                             vl.data(), &ld, vr.data(), &ld, &n, &m,
                             work.data(), &lwork, rwork.data(), &lrwork, &info,
                             1, 1);
  } else {
    using F = void (*)(const char*, const char*, I*, const I*, const T*,
                       const I*, T*, const I*, T*, const I*, const I*, I*, T*,
                       const I*, I*, size_t, size_t);
    lib.Get<F>(name.c_str())(&side, &howmny, select.data(), &n, t.data(), &ld,
                             vl.data(), &ld, vr.data(), &ld, &n, &m,
                             work.data(), &lwork, &info, 1, 1);
  }
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  Add(r, "select", Kind::kInteger, select);
  Add(r, "vl", Kind::kOther, vl);
  Add(r, "vr", Kind::kOther, vr);
  Add(r, "t", Kind::kOther, t);
  return r;
}

// xTRSNA with JOB = 'B' for the Schur form and its eigenvectors (hwyBLAS's
// xTREVC3), for all (HOWMNY = 'A') or every third eigenvalue ('S').
template <typename T>
Result Trsna(const BlasLibrary& hwy, const BlasLibrary& lib,
             const SchurForm<T>& s, I n, char howmny, bool compare) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  // The eigenvectors of the selected eigenvalues.
  Result v = Trevc3(hwy, s, n, howmny, 3 * ld);
  std::vector<T> vl, vr;
  for (const Output& o : v.outputs) {
    std::vector<T>* p = o.name == "vl" ? &vl : o.name == "vr" ? &vr : nullptr;
    if (p == nullptr) continue;
    for (const std::complex<double>& x : o.values) {
      if constexpr (kComplex<T>) {
        p->push_back(T(x));
      } else {
        p->push_back(static_cast<T>(x.real()));
      }
    }
  }
  const char job = 'B';
  std::vector<I> select = howmny == 'S' ? EveryThird(n) : std::vector<I>(ld, 0);
  std::vector<R> sv(ld), sep(ld);
  std::vector<T> work(Size(n, n + 6));
  I m = 0, info = 0;
  Result r;
  const std::string name = std::string(Prefix<T>()) + "trsna_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, const I*, const I*, const T*,
                       const I*, const T*, const I*, const T*, const I*, R*, R*,
                       const I*, I*, T*, const I*, R*, I*, size_t, size_t);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&job, &howmny, select.data(), &n, s.t.data(), &ld,
                             vl.data(), &ld, vr.data(), &ld, sv.data(),
                             sep.data(), &n, &m, work.data(), &ld, rwork.data(),
                             &info, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const I*, const I*, const T*,
                       const I*, const T*, const I*, const T*, const I*, R*, R*,
                       const I*, I*, T*, const I*, I*, I*, size_t, size_t);
    std::vector<I> iwork(std::max<I>(2 * n - 2, 1));
    lib.Get<F>(name.c_str())(&job, &howmny, select.data(), &n, s.t.data(), &ld,
                             vl.data(), &ld, vr.data(), &ld, sv.data(),
                             sep.data(), &n, &m, work.data(), &ld, iwork.data(),
                             &info, 1, 1);
  }
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  const Kind cond = compare ? Kind::kValues : Kind::kOther;
  Add(r, "s", cond, sv.data(), static_cast<size_t>(m));
  Add(r, "sep", cond, sep.data(), static_cast<size_t>(m));
  return r;
}

// The eigenvalue routines for matrices of order n of the kinds of
// TestMatrix: xGEEV and xGEEVX (all options for random and graded
// matrices), xGEESX, and xTREVC3 and xTRSNA for the Schur form of hwyBLAS's
// xGEES. The complex eigenvalues of graded matrices are too ill-conditioned
// to be compared with a tolerance.
template <typename T>
void RunEigen(const BlasLibrary& hwy, const BlasLibrary& ref, I n) {
  char what[96];
  for (int kind = 0; kind < 5; ++kind) {
    const std::vector<T> a = TestMatrix<T>(kind, n, n);
    const bool compare_w = !kComplex<T> || kind != 2;
    static const char kJobs[][2] = {
        {'V', 'V'}, {'V', 'N'}, {'N', 'V'}, {'N', 'N'}};
    for (const auto& job : kJobs) {
      snprintf(what, sizeof(what), "n=%d kind=%d jobvl=%c jobvr=%c", int(n),
               kind, job[0], job[1]);
      Case<T>(hwy, ref, "geev", what, n, [&](const BlasLibrary& lib, I lw) {
        return Geev(lib, a, n, job[0], job[1], compare_w, lw);
      });
    }
    for (const char balanc : {'N', 'P', 'S', 'B'}) {
      for (const char sense : {'N', 'E', 'V', 'B'}) {
        if ((kind == 1 || kind == 3 || kind == 4) &&
            (balanc != 'B' || sense != 'B')) {
          continue;
        }
        for (const auto& job : kJobs) {
          if ((sense == 'E' || sense == 'B') && job[0] != job[1]) continue;
          if ((sense == 'E' || sense == 'B') && job[0] == 'N') continue;
          snprintf(what, sizeof(what),
                   "n=%d kind=%d balanc=%c sense=%c jobvl=%c jobvr=%c", int(n),
                   kind, balanc, sense, job[0], job[1]);
          Case<T>(hwy, ref, "geevx", what, n,
                  [&](const BlasLibrary& lib, I lw) {
                    return Geevx(lib, a, n, balanc, job[0], job[1], sense,
                                 compare_w, lw);
                  });
        }
      }
    }
    for (const char sense : {'N', 'E', 'V', 'B'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d sense=%c", int(n), kind,
               sense);
      Case<T>(hwy, ref, "geesx", what, n, [&](const BlasLibrary& lib, I lw) {
        return Geesx(lib, a, n, sense, compare_w, lw);
      });
    }
    const SchurForm<T> s = Schur(hwy, a, n);
    for (const char howmny : {'A', 'S', 'B'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d howmny=%c", int(n), kind,
               howmny);
      // The optimal workspace (blocked back-transform for HOWMNY = 'B'),
      // and the minimal one.
      Case<T>(hwy, ref, "trevc3", what, n, [&](const BlasLibrary& lib, I lw) {
        return Trevc3(lib, s, n, howmny, lw);
      });
      if (howmny == 'B') {
        snprintf(what, sizeof(what), "n=%d kind=%d howmny=B lwork=3n", int(n),
                 kind);
        Case<T>(hwy, ref, "trevc3", what, n, [&](const BlasLibrary& lib, I lw) {
          return Trevc3(lib, s, n, howmny,
                        lw == -1 ? -1 : 3 * std::max<I>(n, 1));
        });
      } else {
        Case<T>(hwy, ref, "trsna", what, n, [&](const BlasLibrary& lib, I) {
          return Trsna(hwy, lib, s, n, howmny, compare_w);
        });
      }
    }
  }
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

// ---------------------------------------------------------------------------
// Symmetric (Hermitian) linear systems. The routines that take a Cholesky
// factorization get the same one (hwyBLAS's xPOTRF) in both libraries; the
// drivers that factor with xPOTRF (xPOSV, xPOSVX with FACT = 'N' or 'E')
// round differently in its leaf kernels, so their factors and solutions are
// compared with the tolerance. The Bunch-Kaufman routines have no such
// deviation and are all compared bitwise (also the drivers).

// The name of the symmetric (sy) or Hermitian (he) routine, e.g.
// SyName<double>(false, "trf_") = "dsytrf_".
template <typename T>
std::string SyName(bool herm, const char* rest) {
  return std::string(Prefix<T>()) + (herm ? "he" : "sy") + rest;
}

// A complex symmetric (real symmetric) matrix, both triangles set.
template <typename T>
std::vector<T> Symmetric(I n) {
  std::vector<T> a = TestMatrix<T>(0, n, n);
  for (I j = 0; j < n; ++j) {
    for (I i = 0; i < j; ++i) {
      a[j + static_cast<size_t>(i) * n] = a[i + static_cast<size_t>(j) * n];
    }
  }
  return a;
}

// [0 B; B**H 0] (B**T for !herm) with random B (and 1 at the end for odd
// n): Bunch-Kaufman chooses mostly 2-by-2 pivots.
template <typename T>
std::vector<T> Saddle(I n, bool herm) {
  std::vector<T> a(Size(n, n));
  const I p = n / 2;
  for (I j = 0; j < p; ++j) {
    for (I i = 0; i < p; ++i) {
      const T v = Random<T>();
      a[i + static_cast<size_t>(p + j) * n] = v;
      a[p + j + static_cast<size_t>(i) * n] = herm ? Conj(v) : v;
    }
  }
  if (n % 2 == 1) a[Size(n, n) - 1] = T(1);
  return a;
}

// A positive semidefinite matrix of rank r: B * B**H for an n x r matrix B.
template <typename T>
std::vector<T> Semidefinite(I n, I r) {
  const std::vector<T> b = TestMatrix<T>(0, n, r);
  std::vector<T> a(Size(n, n));
  for (I j = 0; j < n; ++j) {
    for (I i = 0; i < n; ++i) {
      T s = T(0);
      for (I k = 0; k < r; ++k) {
        s += b[i + static_cast<size_t>(k) * n] *
             Conj(b[j + static_cast<size_t>(k) * n]);
      }
      a[i + static_cast<size_t>(j) * n] = i == j ? T(std::real(s)) : s;
    }
  }
  return a;
}

// The Cholesky factor of a by hwyBLAS's xPOTRF.
template <typename T>
std::vector<T> Cholesky(const BlasLibrary& hwy, std::vector<T> a, I n,
                        char uplo) {
  using F = void (*)(const char*, const I*, T*, const I*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  hwy.Get<F>((std::string(Prefix<T>()) + "potrf_").c_str())(&uplo, &n, a.data(),
                                                            &ld, &info, 1);
  return a;
}

template <typename T>
Result Potrs(const BlasLibrary& lib, const std::vector<T>& f, std::vector<T> b,
             I n, I nrhs, char uplo) {
  using F = void (*)(const char*, const I*, const I*, const T*, const I*, T*,
                     const I*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "potrs_").c_str())(
      &uplo, &n, &nrhs, f.data(), &ld, b.data(), &ld, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "x", Kind::kOther, b);
  return r;
}

template <typename T>
Result Posv(const BlasLibrary& lib, std::vector<T> a, std::vector<T> b, I n,
            I nrhs, char uplo) {
  using F = void (*)(const char*, const I*, const I*, T*, const I*, T*,
                     const I*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "posv_").c_str())(
      &uplo, &n, &nrhs, a.data(), &ld, b.data(), &ld, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "a", Kind::kRounded, a);
  Add(r, "x", Kind::kRounded, b);
  return r;
}

// xPOTRI or xLAUUM (name) on the factor f.
template <typename T>
Result Potri(const BlasLibrary& lib, const char* name, std::vector<T> f, I n,
             char uplo) {
  using F = void (*)(const char*, const I*, T*, const I*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + name).c_str())(&uplo, &n, f.data(),
                                                        &ld, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "a", Kind::kOther, f);
  return r;
}

template <typename T>
Result Pocon(const BlasLibrary& lib, const std::vector<T>& f, I n, char uplo,
             Real<T> anorm) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  R rcond = -1;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "pocon_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const T*, const I*, const R*, R*,
                       T*, R*, I*, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &n, f.data(), &ld, &anorm, &rcond,
                             work.data(), rwork.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const T*, const I*, const R*, R*,
                       T*, I*, I*, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &n, f.data(), &ld, &anorm, &rcond,
                             work.data(), iwork.data(), &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "rcond", Kind::kValues, &rcond, 1);
  return r;
}

template <typename T>
Result Poequ(const BlasLibrary& lib, const std::vector<T>& a, I n) {
  using R = Real<T>;
  using F = void (*)(const I*, const T*, const I*, R*, R*, R*, I*);
  const I ld = std::max<I>(n, 1);
  std::vector<R> s(ld);
  R v[2] = {-1, -1};
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "poequ_").c_str())(
      &n, a.data(), &ld, s.data(), &v[0], &v[1], &info);
  Result r;
  AddInt(r, "info", info);
  if (info == 0) Add(r, "s", Kind::kOther, s);
  Add(r, "scond amax", Kind::kOther, v, info == 0 ? 2 : 0);
  return r;
}

// xPORFS (with the factor f) refining the solution x.
template <typename T>
Result Porfs(const BlasLibrary& lib, const std::vector<T>& a,
             const std::vector<T>& f, const std::vector<T>& b, std::vector<T> x,
             I n, I nrhs, char uplo) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<R> ferr(std::max<I>(nrhs, 1)), berr(std::max<I>(nrhs, 1));
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "porfs_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const T*, const I*, const T*, const I*, T*, const I*, R*,
                       R*, T*, R*, I*, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &n, &nrhs, a.data(), &ld, f.data(), &ld,
                             b.data(), &ld, x.data(), &ld, ferr.data(),
                             berr.data(), work.data(), rwork.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const T*, const I*, const T*, const I*, T*, const I*, R*,
                       R*, T*, I*, I*, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &n, &nrhs, a.data(), &ld, f.data(), &ld,
                             b.data(), &ld, x.data(), &ld, ferr.data(),
                             berr.data(), work.data(), iwork.data(), &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "x", Kind::kOther, x);
  Add(r, "ferr", Kind::kValues, ferr);
  Add(r, "berr", Kind::kValues, berr);
  return r;
}

// xPOSVX. For FACT = 'F', f is the given factor (EQUED = 'N'); the results
// are then compared bitwise.
template <typename T>
Result Posvx(const BlasLibrary& lib, char fact, char uplo, std::vector<T> a,
             const std::vector<T>& f, std::vector<T> b, I n, I nrhs) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> af = f, x(Size(n, nrhs));
  std::vector<R> s(ld, 1), ferr(std::max<I>(nrhs, 1)),
      berr(std::max<I>(nrhs, 1));
  char equed = 'N';
  R rcond = -1;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "posvx_";
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const I*, const I*, T*, const I*, T*,
                 const I*, char*, R*, T*, const I*, T*, const I*, R*, R*, R*,
                 T*, R*, I*, size_t, size_t, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&fact, &uplo, &n, &nrhs, a.data(), &ld, af.data(),
                             &ld, &equed, s.data(), b.data(), &ld, x.data(),
                             &ld, &rcond, ferr.data(), berr.data(), work.data(),
                             rwork.data(), &info, 1, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const I*, const I*, T*, const I*, T*,
                 const I*, char*, R*, T*, const I*, T*, const I*, R*, R*, R*,
                 T*, I*, I*, size_t, size_t, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&fact, &uplo, &n, &nrhs, a.data(), &ld, af.data(),
                             &ld, &equed, s.data(), b.data(), &ld, x.data(),
                             &ld, &rcond, ferr.data(), berr.data(), work.data(),
                             iwork.data(), &info, 1, 1, 1);
  }
  const Kind factored = fact == 'F' ? Kind::kOther : Kind::kRounded;
  const Kind value = fact == 'F' ? Kind::kValues : Kind::kRounded;
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "equed", equed);
  Add(r, "s", Kind::kOther, s);
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  Add(r, "af", factored, af);
  Add(r, "x", factored, x);
  Add(r, "rcond", value, &rcond, 1);
  Add(r, "ferr", value, ferr);
  Add(r, "berr", value, berr);
  return r;
}

template <typename T>
Result Pstrf(const BlasLibrary& lib, std::vector<T> a, I n, char uplo) {
  using R = Real<T>;
  using F = void (*)(const char*, const I*, T*, const I*, I*, I*, const R*, R*,
                     I*, size_t);
  const I ld = std::max<I>(n, 1);
  std::vector<I> piv(ld);
  std::vector<R> work(2 * ld);
  I rank = -1, info = 0;
  const R tol = -1;
  lib.Get<F>((std::string(Prefix<T>()) + "pstrf_").c_str())(
      &uplo, &n, a.data(), &ld, piv.data(), &rank, &tol, work.data(), &info, 1);
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "rank", rank);
  Add(r, "piv", Kind::kInteger, piv);
  Add(r, "a", Kind::kOther, a);
  return r;
}

// A Bunch-Kaufman factorization and its pivots.
template <typename T>
struct Bk {
  std::vector<T> f;
  std::vector<I> ipiv;
};

template <typename T>
Result Sytrf(const BlasLibrary& lib, bool herm, std::vector<T> a, I n,
             char uplo, I lwork) {
  using F = void (*)(const char*, const I*, T*, const I*, I*, T*, const I*, I*,
                     size_t);
  const I ld = std::max<I>(n, 1);
  std::vector<I> ipiv(ld);
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  lib.Get<F>(SyName<T>(herm, "trf_").c_str())(
      &uplo, &n, a.data(), &ld, ipiv.data(), work.data(), &lwork, &info, 1);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "ipiv", Kind::kInteger, ipiv);
  Add(r, "a", Kind::kOther, a);
  return r;
}

// The factorization of a by hwyBLAS's xSYTRF / xHETRF.
template <typename T>
Bk<T> Bunch(const BlasLibrary& hwy, bool herm, std::vector<T> a, I n,
            char uplo) {
  using F = void (*)(const char*, const I*, T*, const I*, I*, T*, const I*, I*,
                     size_t);
  const auto f = hwy.Get<F>(SyName<T>(herm, "trf_").c_str());
  const I ld = std::max<I>(n, 1);
  Bk<T> bk{std::move(a), std::vector<I>(ld)};
  T q = T(0);
  I lwork = -1, info = 0;
  f(&uplo, &n, bk.f.data(), &ld, bk.ipiv.data(), &q, &lwork, &info, 1);
  lwork = std::max<I>(1, static_cast<I>(std::real(q)));
  std::vector<T> work(lwork);
  f(&uplo, &n, bk.f.data(), &ld, bk.ipiv.data(), work.data(), &lwork, &info, 1);
  return bk;
}

// xSYTRS / xHETRS, or xSYTRS2 / xHETRS2 if level3.
template <typename T>
Result Sytrs(const BlasLibrary& lib, bool herm, bool level3, Bk<T> bk,
             std::vector<T> b, I n, I nrhs, char uplo) {
  const I ld = std::max<I>(n, 1);
  I info = 0;
  if (level3) {
    using F = void (*)(const char*, const I*, const I*, T*, const I*, const I*,
                       T*, const I*, T*, I*, size_t);
    std::vector<T> work(ld);
    lib.Get<F>(SyName<T>(herm, "trs2_").c_str())(&uplo, &n, &nrhs, bk.f.data(),
                                                 &ld, bk.ipiv.data(), b.data(),
                                                 &ld, work.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const I*, T*, const I*, I*, size_t);
    lib.Get<F>(SyName<T>(herm, "trs_").c_str())(&uplo, &n, &nrhs, bk.f.data(),
                                                &ld, bk.ipiv.data(), b.data(),
                                                &ld, &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "x", Kind::kOther, b);
  Add(r, "a", Kind::kOther, bk.f);  // restored by xSYTRS2
  return r;
}

template <typename T>
Result Sytri(const BlasLibrary& lib, bool herm, Bk<T> bk, I n, char uplo) {
  using F =
      void (*)(const char*, const I*, T*, const I*, const I*, T*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  std::vector<T> work(2 * ld);
  I info = 0;
  lib.Get<F>(SyName<T>(herm, "tri_").c_str())(
      &uplo, &n, bk.f.data(), &ld, bk.ipiv.data(), work.data(), &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "a", Kind::kOther, bk.f);
  return r;
}

template <typename T>
Result Sycon(const BlasLibrary& lib, bool herm, const Bk<T>& bk, I n, char uplo,
             Real<T> anorm) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  R rcond = -1;
  I info = 0;
  std::vector<T> work(2 * ld);
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const T*, const I*, const I*,
                       const R*, R*, T*, I*, size_t);
    lib.Get<F>(SyName<T>(herm, "con_").c_str())(&uplo, &n, bk.f.data(), &ld,
                                                bk.ipiv.data(), &anorm, &rcond,
                                                work.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const T*, const I*, const I*,
                       const R*, R*, T*, I*, I*, size_t);
    std::vector<I> iwork(ld);
    lib.Get<F>(SyName<T>(herm, "con_").c_str())(
        &uplo, &n, bk.f.data(), &ld, bk.ipiv.data(), &anorm, &rcond,
        work.data(), iwork.data(), &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "rcond", Kind::kValues, &rcond, 1);
  return r;
}

template <typename T>
Result Syrfs(const BlasLibrary& lib, bool herm, const std::vector<T>& a,
             const Bk<T>& bk, const std::vector<T>& b, std::vector<T> x, I n,
             I nrhs, char uplo) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<R> ferr(std::max<I>(nrhs, 1)), berr(std::max<I>(nrhs, 1));
  I info = 0;
  const std::string name = SyName<T>(herm, "rfs_");
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const T*, const I*, const I*, const T*, const I*, T*,
                       const I*, R*, R*, T*, R*, I*, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &n, &nrhs, a.data(), &ld, bk.f.data(), &ld,
                             bk.ipiv.data(), b.data(), &ld, x.data(), &ld,
                             ferr.data(), berr.data(), work.data(),
                             rwork.data(), &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const I*, const T*, const I*,
                       const T*, const I*, const I*, const T*, const I*, T*,
                       const I*, R*, R*, T*, I*, I*, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&uplo, &n, &nrhs, a.data(), &ld, bk.f.data(), &ld,
                             bk.ipiv.data(), b.data(), &ld, x.data(), &ld,
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

template <typename T>
Result Sysv(const BlasLibrary& lib, bool herm, std::vector<T> a,
            std::vector<T> b, I n, I nrhs, char uplo, I lwork) {
  using F = void (*)(const char*, const I*, const I*, T*, const I*, I*, T*,
                     const I*, T*, const I*, I*, size_t);
  const I ld = std::max<I>(n, 1);
  std::vector<I> ipiv(ld);
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  lib.Get<F>(SyName<T>(herm, "sv_").c_str())(&uplo, &n, &nrhs, a.data(), &ld,
                                             ipiv.data(), b.data(), &ld,
                                             work.data(), &lwork, &info, 1);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "ipiv", Kind::kInteger, ipiv);
  Add(r, "a", Kind::kOther, a);
  Add(r, "x", Kind::kOther, b);
  return r;
}

// xSYSVX / xHESVX; for FACT = 'F', bk is the given factorization.
template <typename T>
Result Sysvx(const BlasLibrary& lib, bool herm, char fact, char uplo,
             const std::vector<T>& a, const Bk<T>& bk, const std::vector<T>& b,
             I n, I nrhs, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> af = bk.f, x(Size(n, nrhs)), work(std::max<I>(lwork, 1));
  std::vector<I> ipiv = bk.ipiv;
  std::vector<R> ferr(std::max<I>(nrhs, 1)), berr(std::max<I>(nrhs, 1));
  R rcond = -1;
  I info = 0;
  const std::string name = SyName<T>(herm, "svx_");
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const I*, const I*, const T*,
                 const I*, T*, const I*, I*, const T*, const I*, T*, const I*,
                 R*, R*, R*, T*, const I*, R*, I*, size_t, size_t);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&fact, &uplo, &n, &nrhs, a.data(), &ld, af.data(),
                             &ld, ipiv.data(), b.data(), &ld, x.data(), &ld,
                             &rcond, ferr.data(), berr.data(), work.data(),
                             &lwork, rwork.data(), &info, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const I*, const I*, const T*,
                 const I*, T*, const I*, I*, const T*, const I*, T*, const I*,
                 R*, R*, R*, T*, const I*, I*, I*, size_t, size_t);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&fact, &uplo, &n, &nrhs, a.data(), &ld, af.data(),
                             &ld, ipiv.data(), b.data(), &ld, x.data(), &ld,
                             &rcond, ferr.data(), berr.data(), work.data(),
                             &lwork, iwork.data(), &info, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "ipiv", Kind::kInteger, ipiv);
  Add(r, "af", Kind::kOther, af);
  Add(r, "x", Kind::kOther, x);
  Add(r, "rcond", Kind::kValues, &rcond, 1);
  Add(r, "ferr", Kind::kValues, ferr);
  Add(r, "berr", Kind::kValues, berr);
  return r;
}

template <typename T>
Result Syconv(const BlasLibrary& lib, Bk<T> bk, I n, char uplo, char way) {
  using F = void (*)(const char*, const char*, const I*, T*, const I*, const I*,
                     T*, I*, size_t, size_t);
  const I ld = std::max<I>(n, 1);
  std::vector<T> e(ld);
  I info = 0;
  const auto f = lib.Get<F>((std::string(Prefix<T>()) + "syconv_").c_str());
  const char convert = 'C';
  f(&uplo, &convert, &n, bk.f.data(), &ld, bk.ipiv.data(), e.data(), &info, 1,
    1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "a", Kind::kOther, bk.f);
  Add(r, "e", Kind::kOther, e);
  if (way == 'R') {
    f(&uplo, &way, &n, bk.f.data(), &ld, bk.ipiv.data(), e.data(), &info, 1, 1);
    AddInt(r, "info", info);
    Add(r, "a reverted", Kind::kOther, bk.f);
  }
  return r;
}

// The symmetric (Hermitian) linear systems routines for matrices of order n.
template <typename T>
void RunSymmetric(const BlasLibrary& hwy, const BlasLibrary& ref, I n) {
  using R = Real<T>;
  char what[64];
  const I nrhs = 3;
  const std::vector<T> b = TestMatrix<T>(0, n, nrhs);

  // Positive definite: random, and with rows and columns scaled by powers
  // of 2 (for equilibration).
  for (const int kind : {0, 1}) {
    std::vector<T> a = PositiveDefinite<T>(n);
    if (kind == 1) {
      for (I j = 0; j < n; ++j) {
        for (I i = 0; i < n; ++i) {
          a[i + static_cast<size_t>(j) * n] *=
              static_cast<R>(std::ldexp(1.0, 3 * (i % 5) + 3 * (j % 5)));
        }
      }
    }
    for (const char uplo : {'U', 'L'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c", int(n), kind, uplo);
      const std::vector<T> f = Cholesky(hwy, a, n, uplo);
      Case<T>(hwy, ref, "posv", what, n, [&](const BlasLibrary& lib, I) {
        return Posv(lib, a, b, n, nrhs, uplo);
      });
      Case<T>(hwy, ref, "potrs", what, n, [&](const BlasLibrary& lib, I) {
        return Potrs(lib, f, b, n, nrhs, uplo);
      });
      Case<T>(hwy, ref, "potri", what, n, [&](const BlasLibrary& lib, I) {
        return Potri(lib, "potri_", f, n, uplo);
      });
      Case<T>(hwy, ref, "lauum", what, n, [&](const BlasLibrary& lib, I) {
        return Potri(lib, "lauum_", f, n, uplo);
      });
      Case<T>(hwy, ref, "pocon", what, n, [&](const BlasLibrary& lib, I) {
        return Pocon(lib, f, n, uplo, R(3));
      });
      Case<T>(hwy, ref, "porfs", what, n, [&](const BlasLibrary& lib, I) {
        return Porfs(lib, a, f, b, b, n, nrhs, uplo);
      });
      for (const char fact : {'N', 'E', 'F'}) {
        snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c fact=%c", int(n),
                 kind, uplo, fact);
        Case<T>(hwy, ref, "posvx", what, n, [&](const BlasLibrary& lib, I) {
          return Posvx(lib, fact, uplo, a, f, b, n, nrhs);
        });
      }
    }
    snprintf(what, sizeof(what), "n=%d kind=%d", int(n), kind);
    Case<T>(hwy, ref, "poequ", what, n,
            [&](const BlasLibrary& lib, I) { return Poequ(lib, a, n); });
  }

  // Pivoted Cholesky: positive definite, and semidefinite of rank n/2.
  for (const I rank : {n, n / 2}) {
    const std::vector<T> a =
        rank == n ? PositiveDefinite<T>(n) : Semidefinite<T>(n, rank);
    for (const char uplo : {'U', 'L'}) {
      snprintf(what, sizeof(what), "n=%d rank=%d uplo=%c", int(n), int(rank),
               uplo);
      Case<T>(hwy, ref, "pstrf", what, n, [&](const BlasLibrary& lib, I) {
        return Pstrf(lib, a, n, uplo);
      });
    }
  }

  // Indefinite: random, saddle point (2-by-2 pivots) and small integers
  // (ties), symmetric and (complex T) Hermitian.
  for (const bool herm : kComplex<T> ? std::vector<bool>{false, true}
                                     : std::vector<bool>{false}) {
    const std::string sy = herm ? "he" : "sy";
    for (const int kind : {0, 1, 2}) {
      std::vector<T> a;
      if (kind == 1) {
        a = Saddle<T>(n, herm);
      } else {
        a = herm ? Hermitian<T>(n) : Symmetric<T>(n);
        if (kind == 2) {
          for (T& x : a) x = T(static_cast<R>(std::round(4 * std::real(x))));
        }
      }
      for (const char uplo : {'U', 'L'}) {
        snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c", int(n), kind,
                 uplo);
        Case<T>(hwy, ref, sy + "trf", what, n,
                [&](const BlasLibrary& lib, I lw) {
                  return Sytrf(lib, herm, a, n, uplo, lw);
                });
        // A workspace for panels of 10 columns (xLASYF / xLAHEF with a
        // smaller NB, for n > 64).
        snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c lwork=10n", int(n),
                 kind, uplo);
        Case<T>(hwy, ref, sy + "trf", what, n, [&](const BlasLibrary& lib, I) {
          return Sytrf(lib, herm, a, n, uplo, std::max<I>(1, 10 * n));
        });
        snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c", int(n), kind,
                 uplo);
        Case<T>(hwy, ref, sy + "sv", what, n,
                [&](const BlasLibrary& lib, I lw) {
                  return Sysv(lib, herm, a, b, n, nrhs, uplo, lw);
                });
        // LWORK < N: the factorization unblocked, the solution by xSYTRS.
        if (n > 1) {
          snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c lwork=n-1", int(n),
                   kind, uplo);
          Case<T>(hwy, ref, sy + "sv", what, n, [&](const BlasLibrary& lib, I) {
            return Sysv(lib, herm, a, b, n, nrhs, uplo, n - 1);
          });
        }
        snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c", int(n), kind,
                 uplo);
        const Bk<T> bk = Bunch(hwy, herm, a, n, uplo);
        for (const bool level3 : {false, true}) {
          Case<T>(hwy, ref, sy + (level3 ? "trs2" : "trs"), what, n,
                  [&](const BlasLibrary& lib, I) {
                    return Sytrs(lib, herm, level3, bk, b, n, nrhs, uplo);
                  });
        }
        Case<T>(hwy, ref, sy + "tri", what, n, [&](const BlasLibrary& lib, I) {
          return Sytri(lib, herm, bk, n, uplo);
        });
        Case<T>(hwy, ref, sy + "con", what, n, [&](const BlasLibrary& lib, I) {
          return Sycon(lib, herm, bk, n, uplo, R(3));
        });
        Case<T>(hwy, ref, sy + "rfs", what, n, [&](const BlasLibrary& lib, I) {
          return Syrfs(lib, herm, a, bk, b, b, n, nrhs, uplo);
        });
        for (const char fact : {'N', 'F'}) {
          snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c fact=%c", int(n),
                   kind, uplo, fact);
          Case<T>(hwy, ref, sy + "svx", what, n,
                  [&](const BlasLibrary& lib, I lw) {
                    return Sysvx(lib, herm, fact, uplo, a, bk, b, n, nrhs, lw);
                  });
        }
        if (!herm) {
          for (const char way : {'C', 'R'}) {
            snprintf(what, sizeof(what), "n=%d kind=%d uplo=%c way=%c", int(n),
                     kind, uplo, way);
            Case<T>(hwy, ref, "syconv", what, n,
                    [&](const BlasLibrary& lib, I) {
                      return Syconv(lib, bk, n, uplo, way);
                    });
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Divide and conquer eigensolvers and the generalized symmetric-definite
// problem

// The workspaces of the divide and conquer routines that the comparison
// does not query (the largest that xSTEDC needs; Case only varies LWORK).
I LargeRwork(I n) { return 1 + 3 * n + 2 * n * 32 + 4 * n * n + 5 * n; }
I LargeIwork(I n) { return 6 + 6 * n + 5 * n * 32; }

// syevd and heevd.
template <typename T>
Result Syevd(const BlasLibrary& lib, std::vector<T> a, I n, char jobz,
             char uplo, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1);
  std::vector<R> w(std::max<I>(n, 1));
  std::vector<T> work(std::max<I>(lwork, 1));
  const I lrwork = lwork == -1 ? -1 : LargeRwork(n);
  const I liwork = lwork == -1 ? -1 : LargeIwork(n);
  std::vector<R> rwork(std::max<I>(lrwork, 1));
  std::vector<I> iwork(std::max<I>(liwork, 1));
  I info = 0;
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const char*, const char*, const I*, T*, const I*, R*, T*,
                 const I*, R*, const I*, I*, const I*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "heevd_").c_str())(
        &jobz, &uplo, &n, a.data(), &lda, w.data(), work.data(), &lwork,
        rwork.data(), &lrwork, iwork.data(), &liwork, &info, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const I*, T*, const I*, R*, T*,
                       const I*, I*, const I*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "syevd_").c_str())(
        &jobz, &uplo, &n, a.data(), &lda, w.data(), work.data(), &lwork,
        iwork.data(), &liwork, &info, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "w", Kind::kValues, w);
  // With JOBZ = 'N', A is destroyed.
  if (jobz == 'V') Add(r, "z", Kind::kOther, a);
  return r;
}

// stedc (compz 'I', 'V' with the given z, 'N') and, for real T, stevd
// (jobz 'V' for compz 'I', 'N').
template <typename T>
Result Stedc(const BlasLibrary& lib, bool stevd, std::vector<Real<T>> d,
             std::vector<Real<T>> e, std::vector<T> z, I n, char compz,
             I lwork) {
  using R = Real<T>;
  const I ldz = std::max<I>(n, 1);
  e.resize(std::max<I>(n, 1));
  std::vector<T> work(std::max<I>(lwork, 1));
  const I lrwork = lwork == -1 ? -1 : LargeRwork(n);
  const I liwork = lwork == -1 ? -1 : LargeIwork(n);
  std::vector<R> rwork(std::max<I>(lrwork, 1));
  std::vector<I> iwork(std::max<I>(liwork, 1));
  I info = 0;
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, R*, R*, T*, const I*, T*,
                       const I*, R*, const I*, I*, const I*, I*, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "stedc_").c_str())(
        &compz, &n, d.data(), e.data(), z.data(), &ldz, work.data(), &lwork,
        rwork.data(), &lrwork, iwork.data(), &liwork, &info, 1);
  } else {
    using F = void (*)(const char*, const I*, R*, R*, T*, const I*, T*,
                       const I*, I*, const I*, I*, size_t);
    const char jobz = compz == 'N' ? 'N' : 'V';
    lib.Get<F>(
        (std::string(Prefix<T>()) + (stevd ? "stevd_" : "stedc_")).c_str())(
        stevd ? &jobz : &compz, &n, d.data(), e.data(), z.data(), &ldz,
        work.data(), &lwork, iwork.data(), &liwork, &info, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "d", Kind::kValues, d);
  // E is destroyed; it is the same in a faithful port (the deflation
  // passes parts of it as RHO, which xLAED2 / xLAED8 modify).
  Add(r, "e", Kind::kOther, e);
  if (compz != 'N') Add(r, "z", Kind::kOther, z);
  return r;
}

// The reflectors of the reduction to tridiagonal form by hwyBLAS's xSYTRD /
// xHETRD.
template <typename T>
struct Reduction {
  std::vector<T> a, tau;
};

template <typename T>
Reduction<T> Tridiagonalize(const BlasLibrary& hwy, std::vector<T> a, I n,
                            char uplo) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1), lwork = 64 * lda;
  std::vector<R> d(lda), e(lda);
  std::vector<T> tau(lda), work(lwork);
  I info = 0;
  using F = void (*)(const char*, const I*, T*, const I*, R*, R*, T*, T*,
                     const I*, I*, size_t);
  hwy.Get<F>(
      (std::string(Prefix<T>()) + (kComplex<T> ? "hetrd_" : "sytrd_")).c_str())(
      &uplo, &n, a.data(), &lda, d.data(), e.data(), tau.data(), work.data(),
      &lwork, &info, 1);
  return {a, tau};
}

// ormtr / unmtr, and ormql / unmql with the reflectors of uplo = 'U'.
template <typename T>
Result Ormtr(const BlasLibrary& lib, bool ql, const Reduction<T>& red,
             std::vector<T> c, I n, I nc, char side, char uplo, char trans,
             I lwork) {
  const I lda = std::max<I>(n, 1);
  const I m = side == 'L' ? n : nc, ncol = side == 'L' ? nc : n;
  const I ldc = std::max<I>(m, 1);
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  const std::string name =
      std::string(Prefix<T>()) + (kComplex<T> ? "unm" : "orm");
  if (ql) {
    // The k = n-1 reflectors in columns 2..n, of order n-1.
    const I k = std::max<I>(n - 1, 0);
    const I mq = side == 'L' ? k : nc, nq = side == 'L' ? nc : k;
    const I ldcq = std::max<I>(mq, 1);
    using F = void (*)(const char*, const char*, const I*, const I*, const I*,
                       const T*, const I*, const T*, T*, const I*, T*, const I*,
                       I*, size_t, size_t);
    lib.Get<F>((name + "ql_").c_str())(
        &side, &trans, &mq, &nq, &k, red.a.data() + (n > 1 ? lda : 0), &lda,
        red.tau.data(), c.data(), &ldcq, work.data(), &lwork, &info, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const char*, const I*,
                       const I*, const T*, const I*, const T*, T*, const I*, T*,
                       const I*, I*, size_t, size_t, size_t);
    lib.Get<F>((name + "tr_").c_str())(
        &side, &uplo, &trans, &m, &ncol, red.a.data(), &lda, red.tau.data(),
        c.data(), &ldc, work.data(), &lwork, &info, 1, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "c", Kind::kOther, c);
  return r;
}

// sygst / hegst with hwyBLAS's Cholesky factor f.
template <typename T>
Result Sygst(const BlasLibrary& lib, std::vector<T> a, const std::vector<T>& f,
             I n, I itype, char uplo) {
  const I lda = std::max<I>(n, 1);
  I info = 0;
  using F = void (*)(const I*, const char*, const I*, T*, const I*, const T*,
                     const I*, I*, size_t);
  lib.Get<F>(
      (std::string(Prefix<T>()) + (kComplex<T> ? "hegst_" : "sygst_")).c_str())(
      &itype, &uplo, &n, a.data(), &lda, f.data(), &lda, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "a", Kind::kOther, a);
  return r;
}

// sygv / hegv and sygvd / hegvd. They factor B with xPOTRF, whose leaf
// kernels round differently (see the top): all results with a tolerance.
template <typename T>
Result Sygv(const BlasLibrary& lib, bool dc, std::vector<T> a, std::vector<T> b,
            I n, I itype, char jobz, char uplo, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1);
  std::vector<R> w(lda);
  std::vector<T> work(std::max<I>(lwork, 1));
  const I lrwork = lwork == -1 ? -1 : LargeRwork(n);
  const I liwork = lwork == -1 ? -1 : LargeIwork(n);
  std::vector<R> rwork(std::max<I>(std::max<I>(lrwork, 3 * n), 1));
  std::vector<I> iwork(std::max<I>(liwork, 1));
  I info = 0;
  const std::string name = std::string(Prefix<T>()) +
                           (kComplex<T> ? "hegv" : "sygv") + (dc ? "d_" : "_");
  if (dc) {
    if constexpr (kComplex<T>) {
      using F = void (*)(const I*, const char*, const char*, const I*, T*,
                         const I*, T*, const I*, R*, T*, const I*, R*, const I*,
                         I*, const I*, I*, size_t, size_t);
      lib.Get<F>(name.c_str())(&itype, &jobz, &uplo, &n, a.data(), &lda,
                               b.data(), &lda, w.data(), work.data(), &lwork,
                               rwork.data(), &lrwork, iwork.data(), &liwork,
                               &info, 1, 1);
    } else {
      using F = void (*)(const I*, const char*, const char*, const I*, T*,
                         const I*, T*, const I*, R*, T*, const I*, I*, const I*,
                         I*, size_t, size_t);
      lib.Get<F>(name.c_str())(&itype, &jobz, &uplo, &n, a.data(), &lda,
                               b.data(), &lda, w.data(), work.data(), &lwork,
                               iwork.data(), &liwork, &info, 1, 1);
    }
  } else {
    if constexpr (kComplex<T>) {
      using F =
          void (*)(const I*, const char*, const char*, const I*, T*, const I*,
                   T*, const I*, R*, T*, const I*, R*, I*, size_t, size_t);
      lib.Get<F>(name.c_str())(&itype, &jobz, &uplo, &n, a.data(), &lda,
                               b.data(), &lda, w.data(), work.data(), &lwork,
                               rwork.data(), &info, 1, 1);
    } else {
      using F =
          void (*)(const I*, const char*, const char*, const I*, T*, const I*,
                   T*, const I*, R*, T*, const I*, I*, size_t, size_t);
      lib.Get<F>(name.c_str())(&itype, &jobz, &uplo, &n, a.data(), &lda,
                               b.data(), &lda, w.data(), work.data(), &lwork,
                               &info, 1, 1);
    }
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "w", Kind::kRounded, w);
  if (jobz == 'V') Add(r, "z", Kind::kRounded, a);
  Add(r, "b", Kind::kRounded, b);
  return r;
}

// The divide and conquer routines for matrices of order n: Hermitian
// random and with a multiple eigenvalue (I + ones: deflation), tridiagonal
// random, split, graded and with clustered eigenvalues (Wilkinson), and
// the generalized problems.
template <typename T>
void RunSyevd(const BlasLibrary& hwy, const BlasLibrary& ref, I n) {
  using R = Real<T>;
  char what[80];
  const char ctrans = kComplex<T> ? 'C' : 'T';
  for (const int kind : {0, 1}) {
    std::vector<T> a = Hermitian<T>(n);
    if (kind == 1) {
      for (I j = 0; j < n; ++j) {
        for (I i = 0; i < n; ++i) {
          a[i + static_cast<size_t>(j) * n] = T(i == j ? 2 : 1);
        }
      }
    }
    for (const char uplo : {'U', 'L'}) {
      for (const char jobz : {'V', 'N'}) {
        snprintf(what, sizeof(what), "n=%d kind=%d jobz=%c uplo=%c", int(n),
                 kind, jobz, uplo);
        Case<T>(hwy, ref, kComplex<T> ? "heevd" : "syevd", what, n,
                [&](const BlasLibrary& lib, I lw) {
                  return Syevd(lib, a, n, jobz, uplo, lw);
                });
      }
      if (kind == 1) continue;
      const Reduction<T> red = Tridiagonalize(hwy, a, n, uplo);
      const I nc = 3;
      for (const char side : {'L', 'R'}) {
        for (const char trans : {'N', ctrans}) {
          const std::vector<T> c =
              TestMatrix<T>(0, side == 'L' ? n : nc, side == 'L' ? nc : n);
          snprintf(what, sizeof(what), "n=%d side=%c uplo=%c trans=%c", int(n),
                   side, uplo, trans);
          Case<T>(hwy, ref, kComplex<T> ? "unmtr" : "ormtr", what, n,
                  [&](const BlasLibrary& lib, I lw) {
                    return Ormtr(lib, false, red, c, n, nc, side, uplo, trans,
                                 lw);
                  });
          if (uplo == 'U' && n > 1) {
            const std::vector<T> cq = TestMatrix<T>(0, side == 'L' ? n - 1 : nc,
                                                    side == 'L' ? nc : n - 1);
            Case<T>(hwy, ref, kComplex<T> ? "unmql" : "ormql", what, n,
                    [&](const BlasLibrary& lib, I lw) {
                      return Ormtr(lib, true, red, cq, n, nc, side, uplo, trans,
                                   lw);
                    });
          }
        }
      }
    }
  }

  // Tridiagonal: random, split (zeros on the off-diagonal), graded, and
  // Wilkinson (clustered eigenvalues).
  for (const int kind : {0, 1, 2, 3}) {
    std::vector<R> d(std::max<I>(n, 1)), e(std::max<I>(n, 1));
    for (I i = 0; i < n; ++i) {
      d[i] = std::real(Random<T>());
      e[i] = std::real(Random<T>());
      if (kind == 1 && i % 7 == 6) e[i] = 0;
      if (kind == 2) {
        d[i] *= static_cast<R>(std::ldexp(1.0, -(i % 20)));
        e[i] *= static_cast<R>(std::ldexp(1.0, -(i % 20)));
      }
      if (kind == 3) {
        d[i] = static_cast<R>(std::abs(n / 2 - i));
        e[i] = 1;
      }
    }
    const std::vector<T> z = TestMatrix<T>(0, n, n);
    for (const char compz : {'I', 'V', 'N'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d compz=%c", int(n), kind,
               compz);
      Case<T>(hwy, ref, "stedc", what, n, [&](const BlasLibrary& lib, I lw) {
        return Stedc(lib, false, d, e, z, n, compz, lw);
      });
      if (!kComplex<T> && compz != 'V') {
        Case<T>(hwy, ref, "stevd", what, n, [&](const BlasLibrary& lib, I lw) {
          return Stedc(lib, true, d, e, z, n, compz, lw);
        });
      }
    }
  }

  // The generalized problem.
  const std::vector<T> a = Hermitian<T>(n);
  const std::vector<T> b = PositiveDefinite<T>(n);
  for (const char uplo : {'U', 'L'}) {
    const std::vector<T> f = Cholesky(hwy, b, n, uplo);
    for (const I itype : {1, 2, 3}) {
      snprintf(what, sizeof(what), "n=%d itype=%d uplo=%c", int(n), int(itype),
               uplo);
      Case<T>(hwy, ref, kComplex<T> ? "hegst" : "sygst", what, n,
              [&](const BlasLibrary& lib, I) {
                return Sygst(lib, a, f, n, itype, uplo);
              });
      for (const char jobz : {'V', 'N'}) {
        snprintf(what, sizeof(what), "n=%d itype=%d jobz=%c uplo=%c", int(n),
                 int(itype), jobz, uplo);
        for (const bool dc : {false, true}) {
          Case<T>(hwy, ref,
                  std::string(kComplex<T> ? "hegv" : "sygv") + (dc ? "d" : ""),
                  what, n, [&](const BlasLibrary& lib, I lw) {
                    return Sygv(lib, dc, a, b, n, itype, jobz, uplo, lw);
                  });
        }
      }
    }
  }
}

// The wanted eigenvalues of the M10b drivers: RANGE with VL, VU (VALUE) or
// IL, IU.
template <typename R>
struct Wanted {
  char range;
  R vl, vu;
  I il, iu;
};

// All; by index the first half and the last two thirds; by value an
// interval in the middle of the Gershgorin bound g.
template <typename R>
std::vector<Wanted<R>> WantedSets(I n, R g) {
  std::vector<Wanted<R>> s = {{'A', 0, 0, 1, 1}};
  if (n == 0) return s;
  s.push_back({'I', 0, 0, 1, std::max<I>(1, n / 2)});
  s.push_back({'I', 0, 0, n / 3 + 1, n});
  s.push_back({'V', R(-0.3) * g, R(0.4) * g, 1, 1});
  return s;
}

// syevr and heevr. The first m columns of Z and of ISUPPZ are compared
// (ISUPPZ is only set by xSTEMR, for all eigenvalues).
template <typename T>
Result Syevr(const BlasLibrary& lib, std::vector<T> a, I n, char jobz,
             char uplo, const Wanted<Real<T>>& s, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1);
  const R abstol = 0;
  std::vector<R> w(lda);
  std::vector<T> z(static_cast<size_t>(lda) * lda), work(std::max<I>(lwork, 1));
  std::vector<I> isuppz(2 * lda);
  const I lrwork = lwork == -1 ? -1 : LargeRwork(n);
  const I liwork = lwork == -1 ? -1 : LargeIwork(n);
  std::vector<R> rwork(std::max<I>(lrwork, 1));
  std::vector<I> iwork(std::max<I>(liwork, 1));
  I m = -1, info = 0;
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, const char*, const I*, T*,
                       const I*, const R*, const R*, const I*, const I*,
                       const R*, I*, R*, T*, const I*, I*, T*, const I*, R*,
                       const I*, I*, const I*, I*, size_t, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "heevr_").c_str())(
        &jobz, &s.range, &uplo, &n, a.data(), &lda, &s.vl, &s.vu, &s.il, &s.iu,
        &abstol, &m, w.data(), z.data(), &lda, isuppz.data(), work.data(),
        &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info, 1, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const char*, const I*, T*,
                       const I*, const R*, const R*, const I*, const I*,
                       const R*, I*, R*, T*, const I*, I*, T*, const I*, I*,
                       const I*, I*, size_t, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "syevr_").c_str())(
        &jobz, &s.range, &uplo, &n, a.data(), &lda, &s.vl, &s.vu, &s.il, &s.iu,
        &abstol, &m, w.data(), z.data(), &lda, isuppz.data(), work.data(),
        &lwork, iwork.data(), &liwork, &info, 1, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  if (lwork == -1 || info != 0) return r;
  Add(r, "w", Kind::kValues, w.data(), m);
  if (jobz == 'V') {
    Add(r, "z", Kind::kOther, z.data(), static_cast<size_t>(lda) * m);
    if (s.range == 'A') Add(r, "isuppz", Kind::kInteger, isuppz.data(), 2 * m);
  }
  return r;
}

// stevr (stemr = false, real T) or stemr (with tryrac).
template <typename T>
Result Stemr(const BlasLibrary& lib, bool stemr, std::vector<Real<T>> d,
             std::vector<Real<T>> e, I n, char jobz, const Wanted<Real<T>>& s,
             I tryrac, I lwork) {
  using R = Real<T>;
  const I ldz = std::max<I>(n, 1);
  e.resize(ldz);
  const R abstol = 0;
  std::vector<R> w(ldz), work(std::max<I>(lwork, 1));
  std::vector<T> z(static_cast<size_t>(ldz) * ldz);
  std::vector<I> isuppz(2 * ldz);
  const I liwork = lwork == -1 ? -1 : LargeIwork(n);
  std::vector<I> iwork(std::max<I>(liwork, 1));
  I m = -1, info = 0;
  if (stemr) {
    using F =
        void (*)(const char*, const char*, const I*, R*, R*, const R*, const R*,
                 const I*, const I*, I*, R*, T*, const I*, const I*, I*, I*, R*,
                 const I*, I*, const I*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "stemr_").c_str())(
        &jobz, &s.range, &n, d.data(), e.data(), &s.vl, &s.vu, &s.il, &s.iu, &m,
        w.data(), z.data(), &ldz, &n, isuppz.data(), &tryrac, work.data(),
        &lwork, iwork.data(), &liwork, &info, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const I*, R*, R*, const R*, const R*,
                 const I*, const I*, const R*, I*, R*, T*, const I*, I*, R*,
                 const I*, I*, const I*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "stevr_").c_str())(
        &jobz, &s.range, &n, d.data(), e.data(), &s.vl, &s.vu, &s.il, &s.iu,
        &abstol, &m, w.data(), z.data(), &ldz, isuppz.data(), work.data(),
        &lwork, iwork.data(), &liwork, &info, 1, 1);
  }
  Result r;
  r.lwork = work[0];
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  if (lwork == -1 || info != 0) return r;
  if (stemr) AddInt(r, "tryrac", tryrac);
  Add(r, "w", Kind::kValues, w.data(), m);
  if (jobz == 'V') {
    Add(r, "z", Kind::kOther, z.data(), static_cast<size_t>(ldz) * m);
    if (stemr || s.range == 'A') {
      Add(r, "isuppz", Kind::kInteger, isuppz.data(), 2 * m);
    }
  }
  return r;
}

// The scaling thresholds of the eigenvalue drivers: RMIN = sqrt(SMLNUM)
// (upper = false) or RMAX = min(sqrt(BIGNUM), 1 / sqrt(sqrt(SAFMIN))).
template <typename R>
R ScaleThreshold(bool upper) {
  const R safmin = std::numeric_limits<R>::min();
  const R smlnum = safmin / std::numeric_limits<R>::epsilon();
  if (!upper) return std::sqrt(smlnum);
  return std::min(std::sqrt(1 / smlnum), 1 / std::sqrt(std::sqrt(safmin)));
}

// syevx and heevx with ABSTOL. The first m columns of Z and IFAIL are
// compared.
template <typename T>
Result Syevx(const BlasLibrary& lib, std::vector<T> a, I n, char jobz,
             char uplo, const Wanted<Real<T>>& s, Real<T> abstol, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(n, 1);
  std::vector<R> w(lda), rwork(7 * lda);
  std::vector<T> z(static_cast<size_t>(lda) * lda), work(std::max<I>(lwork, 1));
  std::vector<I> iwork(5 * lda), ifail(lda);
  I m = -1, info = 0;
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const char*, const char*, const I*, T*,
                       const I*, const R*, const R*, const I*, const I*,
                       const R*, I*, R*, T*, const I*, T*, const I*, R*, I*, I*,
                       I*, size_t, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "heevx_").c_str())(
        &jobz, &s.range, &uplo, &n, a.data(), &lda, &s.vl, &s.vu, &s.il, &s.iu,
        &abstol, &m, w.data(), z.data(), &lda, work.data(), &lwork,
        rwork.data(), iwork.data(), ifail.data(), &info, 1, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const char*, const I*, T*, const I*,
                 const R*, const R*, const I*, const I*, const R*, I*, R*, T*,
                 const I*, T*, const I*, I*, I*, I*, size_t, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "syevx_").c_str())(
        &jobz, &s.range, &uplo, &n, a.data(), &lda, &s.vl, &s.vu, &s.il, &s.iu,
        &abstol, &m, w.data(), z.data(), &lda, work.data(), &lwork,
        iwork.data(), ifail.data(), &info, 1, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  if (lwork == -1 || info != 0) return r;
  Add(r, "w", Kind::kValues, w.data(), m);
  if (jobz == 'V') {
    Add(r, "z", Kind::kOther, z.data(), static_cast<size_t>(lda) * m);
    Add(r, "ifail", Kind::kInteger, ifail.data(), m);
  }
  return r;
}

// stevx (real T) with ABSTOL; D and E are compared too (scaled on exit).
template <typename T>
Result Stevx(const BlasLibrary& lib, std::vector<T> d, std::vector<T> e, I n,
             char jobz, const Wanted<T>& s, T abstol) {
  const I ldz = std::max<I>(n, 1);
  e.resize(ldz);
  std::vector<T> w(ldz), work(5 * ldz), z(static_cast<size_t>(ldz) * ldz);
  std::vector<I> iwork(5 * ldz), ifail(ldz);
  I m = -1, info = 0;
  using F = void (*)(const char*, const char*, const I*, T*, T*, const T*,
                     const T*, const I*, const I*, const T*, I*, T*, T*,
                     const I*, T*, I*, I*, I*, size_t, size_t);
  lib.Get<F>((std::string(Prefix<T>()) + "stevx_").c_str())(
      &jobz, &s.range, &n, d.data(), e.data(), &s.vl, &s.vu, &s.il, &s.iu,
      &abstol, &m, w.data(), z.data(), &ldz, work.data(), iwork.data(),
      ifail.data(), &info, 1, 1);
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  if (info != 0) return r;
  Add(r, "w", Kind::kValues, w.data(), m);
  Add(r, "d", Kind::kOther, d.data(), n);
  Add(r, "e", Kind::kOther, e.data(), std::max<I>(n - 1, 0));
  if (jobz == 'V') {
    Add(r, "z", Kind::kOther, z.data(), static_cast<size_t>(ldz) * m);
    Add(r, "ifail", Kind::kInteger, ifail.data(), m);
  }
  return r;
}

// stebz (real; order 'B' or 'E') and, for order 'B', stein of T's
// precision with its eigenvalues (from the reference stebz in both
// libraries, so that stein gets the same input).
template <typename T>
Result Stebz(const BlasLibrary& lib, std::vector<Real<T>> d,
             std::vector<Real<T>> e, I n, char order,
             const Wanted<Real<T>>& s) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  e.resize(ld);
  const R abstol = 0;
  std::vector<R> w(ld), work(4 * ld);
  std::vector<I> iblock(ld), isplit(ld), iwork(3 * ld);
  I m = -1, nsplit = -1, info = 0;
  using F = void (*)(const char*, const char*, const I*, const R*, const R*,
                     const I*, const I*, const R*, const R*, const R*, I*, I*,
                     R*, I*, I*, R*, I*, I*, size_t, size_t);
  lib.Get<F>((std::string(kComplex<T> ? (std::is_same_v<R, float> ? "s" : "d")
                                      : Prefix<T>()) +
              "stebz_")
                 .c_str())(&s.range, &order, &n, &s.vl, &s.vu, &s.il, &s.iu,
                           &abstol, d.data(), e.data(), &m, &nsplit, w.data(),
                           iblock.data(), isplit.data(), work.data(),
                           iwork.data(), &info, 1, 1);
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  AddInt(r, "nsplit", nsplit);
  Add(r, "w", Kind::kValues, w.data(), std::max<I>(m, 0));
  Add(r, "iblock", Kind::kInteger, iblock.data(), std::max<I>(m, 0));
  Add(r, "isplit", Kind::kInteger, isplit.data(), std::max<I>(nsplit, 0));
  if (order != 'B' || info != 0 || m <= 0) return r;
  std::vector<T> z(static_cast<size_t>(ld) * m);
  std::vector<R> swork(5 * ld);
  std::vector<I> siwork(ld), ifail(m);
  using G = void (*)(const I*, const R*, const R*, const I*, const R*, const I*,
                     const I*, T*, const I*, R*, I*, I*, I*);
  lib.Get<G>((std::string(Prefix<T>()) + "stein_").c_str())(
      &n, d.data(), e.data(), &m, w.data(), iblock.data(), isplit.data(),
      z.data(), &ld, swork.data(), siwork.data(), ifail.data(), &info);
  AddInt(r, "stein info", info);
  Add(r, "ifail", Kind::kInteger, ifail);
  Add(r, "z", Kind::kOther, z);
  return r;
}

// larnv: the random numbers and the new seed.
template <typename T>
Result Larnv(const BlasLibrary& lib, I idist, I n) {
  std::vector<I> iseed = {11, 22, 33, 45};
  std::vector<T> x(std::max<I>(n, 1));
  using F = void (*)(const I*, I*, const I*, T*);
  lib.Get<F>((std::string(Prefix<T>()) + "larnv_").c_str())(
      &idist, iseed.data(), &n, x.data());
  Result r;
  Add(r, "iseed", Kind::kInteger, iseed);
  // IDIST = 3 to 5 use LOG, COS, EXP of the C runtime and flang.
  Add(r, "x", idist <= 2 ? Kind::kOther : Kind::kValues, x.data(), n);
  return r;
}

// The MRRR, bisection and inverse iteration eigensolvers for matrices of
// order n: xSYEVR / xHEEVR on a random matrix, xSYEVX / xHEEVX also on
// matrices that it scales (norm near the underflow and overflow
// thresholds, and just beyond its scaling thresholds), xSTEVR, xSTEVX, xSTEMR,
// xSTEBZ and xSTEIN on tridiagonal matrices (random, split, graded, clustered
// (glued Wilkinson), near the underflow threshold, scaled diagonally dominant,
// with norms just beyond the scaling thresholds), for all ranges, and xSYEVX /
// xHEEVX and xSTEVX with ABSTOL = 0 (xSTERF or xSTEQR for all eigenvalues) and
// > 0 (xSTEBZ and xSTEIN); and xLARNV.
template <typename T>
void RunSyevr(const BlasLibrary& hwy, const BlasLibrary& ref, I n) {
  using R = Real<T>;
  char what[96];
  const std::vector<T> a = Hermitian<T>(n);
  // The Gershgorin bound of A.
  R g = 0;
  for (I i = 0; i < n; ++i) {
    R s = 0;
    for (I j = 0; j < n; ++j) s += std::abs(a[i + static_cast<size_t>(j) * n]);
    g = std::max(g, s);
  }
  for (const Wanted<R>& s : WantedSets<R>(n, g)) {
    for (const char uplo : {'U', 'L'}) {
      for (const char jobz : {'V', 'N'}) {
        snprintf(what, sizeof(what), "n=%d jobz=%c range=%c uplo=%c", int(n),
                 jobz, s.range, uplo);
        Case<T>(hwy, ref, kComplex<T> ? "heevr" : "syevr", what, n,
                [&](const BlasLibrary& lib, I lw) {
                  return Syevr(lib, a, n, jobz, uplo, s, lw);
                });
      }
    }
  }
  // Far beyond and just beyond the drivers' scaling thresholds RMIN and
  // RMAX (of the largest element; by factors that are not powers of 2,
  // where the scaling would be exact).
  const R tiny = std::sqrt(std::numeric_limits<R>::min());
  R amax = 0;
  for (const T& x : a) amax = std::max(amax, std::abs(x));
  const R near_rmin =
      ScaleThreshold<R>(false) * R(0.3) / (amax > 0 ? amax : R(1));
  const R near_rmax = ScaleThreshold<R>(true) * 3 / (amax > 0 ? amax : R(1));
  for (const R scale : {R(1), tiny / 4, 1 / tiny, near_rmin, near_rmax}) {
    std::vector<T> as = a;
    for (T& x : as) x *= scale;
    for (const Wanted<R>& s : WantedSets<R>(n, g * scale)) {
      for (const char uplo : {'U', 'L'}) {
        for (const char jobz : {'V', 'N'}) {
          for (const R abstol : {R(0), 2 * std::numeric_limits<R>::min()}) {
            snprintf(what, sizeof(what),
                     "n=%d scale=%g jobz=%c range=%c uplo=%c abstol=%g", int(n),
                     double(scale), jobz, s.range, uplo, double(abstol));
            Case<T>(hwy, ref, kComplex<T> ? "heevx" : "syevx", what, n,
                    [&](const BlasLibrary& lib, I lw) {
                      return Syevx(lib, as, n, jobz, uplo, s, abstol, lw);
                    });
          }
        }
      }
    }
  }

  for (const int kind : {0, 1, 2, 3, 4, 5, 6, 7}) {
    std::vector<R> d(std::max<I>(n, 1)), e(std::max<I>(n, 1));
    for (I i = 0; i < n; ++i) {
      d[i] = std::real(Random<T>());
      e[i] = std::real(Random<T>());
      if (kind == 1 && i % 7 == 6) e[i] = 0;
      if (kind == 2) {
        d[i] *= static_cast<R>(std::ldexp(1.0, -(i % 20)));
        e[i] *= static_cast<R>(std::ldexp(1.0, -(i % 20)));
      }
      if (kind == 3) {
        d[i] = static_cast<R>(std::abs(i % 11 - 5));
        e[i] = i % 11 == 10 ? R(1e-6) : R(1);
      }
      if (kind == 4) {
        const R tiny = std::sqrt(std::numeric_limits<R>::min()) / 4;
        d[i] *= tiny;
        e[i] *= tiny;
      }
      if (kind == 5) {
        d[i] =
            (R(1.5) + d[i] / 2) * static_cast<R>(std::ldexp(1.0, -3 * (i % 6)));
      }
    }
    if (kind == 5) {
      for (I i = 0; i + 1 < n; ++i) {
        e[i] *= R(0.2) * std::sqrt(std::abs(d[i] * d[i + 1]));
      }
    }
    if (kind >= 6) {
      // Norm just below RMIN (6) or above RMAX (7) of xSTEVR / xSTEVX.
      R tmax = 0;
      for (I i = 0; i < n; ++i) {
        tmax = std::max(tmax, std::abs(d[i]));
        if (i + 1 < n) tmax = std::max(tmax, std::abs(e[i]));
      }
      const R target = kind == 6 ? ScaleThreshold<R>(false) * R(0.3)
                                 : ScaleThreshold<R>(true) * 3;
      const R f = tmax > 0 ? target / tmax : R(1);
      for (I i = 0; i < n; ++i) {
        d[i] *= f;
        e[i] *= f;
      }
    }
    R gt = 0;
    for (I i = 0; i < n; ++i) {
      gt = std::max(gt, std::abs(d[i]) + std::abs(e[i]) +
                            (i > 0 ? std::abs(e[i - 1]) : R(0)));
    }
    for (const Wanted<R>& s : WantedSets<R>(n, gt)) {
      for (const char jobz : {'V', 'N'}) {
        for (const I tryrac : {1, 0}) {
          snprintf(what, sizeof(what), "n=%d kind=%d jobz=%c range=%c rac=%d",
                   int(n), kind, jobz, s.range, int(tryrac));
          Case<T>(hwy, ref, "stemr", what, n,
                  [&](const BlasLibrary& lib, I lw) {
                    return Stemr<T>(lib, true, d, e, n, jobz, s, tryrac, lw);
                  });
        }
        if constexpr (!kComplex<T>) {
          snprintf(what, sizeof(what), "n=%d kind=%d jobz=%c range=%c", int(n),
                   kind, jobz, s.range);
          Case<T>(hwy, ref, "stevr", what, n,
                  [&](const BlasLibrary& lib, I lw) {
                    return Stemr<T>(lib, false, d, e, n, jobz, s, 0, lw);
                  });
          for (const R abstol : {R(0), 2 * std::numeric_limits<R>::min()}) {
            snprintf(what, sizeof(what),
                     "n=%d kind=%d jobz=%c range=%c abstol=%g", int(n), kind,
                     jobz, s.range, double(abstol));
            Case<T>(hwy, ref, "stevx", what, n, [&](const BlasLibrary& lib, I) {
              return Stevx<T>(lib, d, e, n, jobz, s, abstol);
            });
          }
        }
      }
      for (const char order : {'B', 'E'}) {
        // xSTEBZ is real: alone only for real T.
        if (kComplex<T> && order == 'E') continue;
        snprintf(what, sizeof(what), "n=%d kind=%d order=%c range=%c", int(n),
                 kind, order, s.range);
        Case<T>(hwy, ref, order == 'B' ? "stein" : "stebz", what, n,
                [&](const BlasLibrary& lib, I) {
                  return Stebz<T>(lib, d, e, n, order, s);
                });
      }
    }
  }

  for (const I idist : {1, 2, 3, 4, 5}) {
    if (!kComplex<T> && idist > 3) continue;
    snprintf(what, sizeof(what), "n=%d idist=%d", int(n), int(idist));
    Case<T>(hwy, ref, "larnv", what, n,
            [&](const BlasLibrary& lib, I) { return Larnv<T>(lib, idist, n); });
  }
}

// ---------------------------------------------------------------------------
// Least squares (M11)

// An m x n matrix of rank min(r, m, n): the product of random m x r and r x n
// matrices.
template <typename T>
std::vector<T> LowRank(I m, I n, I r) {
  const std::vector<T> x = TestMatrix<T>(0, m, r);
  const std::vector<T> y = TestMatrix<T>(0, r, n);
  std::vector<T> a(static_cast<size_t>(m) * n, T(0));
  for (I j = 0; j < n; ++j) {
    for (I l = 0; l < r; ++l) {
      for (I i = 0; i < m; ++i) {
        a[i + static_cast<size_t>(j) * m] += x[i + static_cast<size_t>(l) * m] *
                                             y[l + static_cast<size_t>(j) * r];
      }
    }
  }
  return a;
}

// The right-hand sides of a least squares problem with A m x n: max(m, n)
// rows (LDB), nrhs columns.
template <typename T>
std::vector<T> RightHandSides(I m, I n, I nrhs) {
  return TestMatrix<T>(0, std::max<I>({m, n, 1}), nrhs);
}

// gels with TRANS = 'N' or 'T' / 'C'.
template <typename T>
Result Gels(const BlasLibrary& lib, char trans, std::vector<T> a, I m, I n,
            std::vector<T> b, I nrhs, I lwork) {
  const I lda = std::max<I>(m, 1);
  const I ldb = std::max<I>({m, n, 1});
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  using F = void (*)(const char*, const I*, const I*, const I*, T*, const I*,
                     T*, const I*, T*, const I*, I*, size_t);
  lib.Get<F>((std::string(Prefix<T>()) + "gels_").c_str())(
      &trans, &m, &n, &nrhs, a.data(), &lda, b.data(), &ldb, work.data(),
      &lwork, &info, 1);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  if (lwork == -1) return r;
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  return r;
}

// gelss (dc = false) or gelsd (dc = true) with RCOND.
template <typename T>
Result GelssD(const BlasLibrary& lib, bool dc, std::vector<T> a, I m, I n,
              std::vector<T> b, I nrhs, Real<T> rcond, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(m, 1);
  const I ldb = std::max<I>({m, n, 1});
  const I mn = std::max<I>(std::min(m, n), 1);
  std::vector<R> s(mn);
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<R> rwork(128 * static_cast<size_t>(std::max(m, n)) + 2048);
  std::vector<I> iwork(64 * static_cast<size_t>(mn) + 64);
  I rank = -1, info = 0;
  const std::string name =
      std::string(Prefix<T>()) + (dc ? "gelsd_" : "gelss_");
  if constexpr (kComplex<T>) {
    if (dc) {
      using F = void (*)(const I*, const I*, const I*, T*, const I*, T*,
                         const I*, R*, const R*, I*, T*, const I*, R*, I*, I*);
      lib.Get<F>(name.c_str())(&m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                               s.data(), &rcond, &rank, work.data(), &lwork,
                               rwork.data(), iwork.data(), &info);
    } else {
      using F = void (*)(const I*, const I*, const I*, T*, const I*, T*,
                         const I*, R*, const R*, I*, T*, const I*, R*, I*);
      lib.Get<F>(name.c_str())(&m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                               s.data(), &rcond, &rank, work.data(), &lwork,
                               rwork.data(), &info);
    }
  } else {
    if (dc) {
      using F = void (*)(const I*, const I*, const I*, T*, const I*, T*,
                         const I*, R*, const R*, I*, T*, const I*, I*, I*);
      lib.Get<F>(name.c_str())(&m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                               s.data(), &rcond, &rank, work.data(), &lwork,
                               iwork.data(), &info);
    } else {
      using F = void (*)(const I*, const I*, const I*, T*, const I*, T*,
                         const I*, R*, const R*, I*, T*, const I*, I*);
      lib.Get<F>(name.c_str())(&m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                               s.data(), &rcond, &rank, work.data(), &lwork,
                               &info);
    }
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  if (lwork == -1) {
    if (dc) AddInt(r, "liwork", iwork[0]);
    if (dc && kComplex<T>) Add(r, "lrwork", Kind::kInteger, rwork.data(), 1);
    return r;
  }
  AddInt(r, "rank", rank);
  Add(r, "s", Kind::kValues, s);
  Add(r, "b", Kind::kOther, b);
  if (!dc) Add(r, "a", Kind::kOther, a);
  return r;
}

// gelsy with RCOND; every third column is moved to the front.
template <typename T>
Result Gelsy(const BlasLibrary& lib, std::vector<T> a, I m, I n,
             std::vector<T> b, I nrhs, Real<T> rcond, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(m, 1);
  const I ldb = std::max<I>({m, n, 1});
  std::vector<I> jpvt(std::max<I>(n, 1));
  for (I j = 0; j < n; ++j) jpvt[j] = j % 3 == 1 ? 1 : 0;
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<R> rwork(2 * std::max<I>(n, 1));
  I rank = -1, info = 0;
  const std::string name = std::string(Prefix<T>()) + "gelsy_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const I*, const I*, const I*, T*, const I*, T*, const I*,
                       I*, const R*, I*, T*, const I*, R*, I*);
    lib.Get<F>(name.c_str())(&m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                             jpvt.data(), &rcond, &rank, work.data(), &lwork,
                             rwork.data(), &info);
  } else {
    using F = void (*)(const I*, const I*, const I*, T*, const I*, T*, const I*,
                       I*, const R*, I*, T*, const I*, I*);
    lib.Get<F>(name.c_str())(&m, &n, &nrhs, a.data(), &lda, b.data(), &ldb,
                             jpvt.data(), &rcond, &rank, work.data(), &lwork,
                             &info);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  if (lwork == -1) return r;
  AddInt(r, "rank", rank);
  Add(r, "jpvt", Kind::kInteger, jpvt);
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  return r;
}

// gerq2 (unblocked; WORK has M elements).
template <typename T>
Result Gerq2(const BlasLibrary& lib, std::vector<T> a, I m, I n) {
  using F = void (*)(const I*, const I*, T*, const I*, T*, T*, I*);
  const I lda = std::max<I>(m, 1);
  std::vector<T> tau(std::max<I>(std::min(m, n), 1));
  std::vector<T> work(std::max<I>(m, 1));
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "gerq2_").c_str())(
      &m, &n, a.data(), &lda, tau.data(), work.data(), &info);
  Result r;
  AddInt(r, "info", info);
  Add(r, "tau", Kind::kOther, tau);
  Add(r, "a", Kind::kOther, a);
  return r;
}

// gglse (A m x n, B p x n) or ggglm (A n x m, B n x p); `rows` and `cols`
// are those of A, bcols the columns of B (or its rows for gglse).
template <typename T>
Result Gglse(const BlasLibrary& lib, bool glm, std::vector<T> a, I m, I n, I p,
             std::vector<T> b, std::vector<T> c, std::vector<T> d, I lwork) {
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  using F = void (*)(const I*, const I*, const I*, T*, const I*, T*, const I*,
                     T*, T*, T*, T*, const I*, I*);
  Result r;
  if (glm) {
    // GGGLM( N = m, M = n, P = p ): A is m x n, B m x p, D(m), X(n), Y(p).
    const I lda = std::max<I>(m, 1), ldb = std::max<I>(m, 1);
    std::vector<T> x(std::max<I>(n, 1)), y(std::max<I>(p, 1));
    lib.Get<F>((std::string(Prefix<T>()) + "ggglm_").c_str())(
        &m, &n, &p, a.data(), &lda, b.data(), &ldb, d.data(), x.data(),
        y.data(), work.data(), &lwork, &info);
    r.lwork = std::real(work[0]);
    AddInt(r, "info", info);
    if (lwork == -1) return r;
    Add(r, "x", Kind::kOther, x);
    Add(r, "y", Kind::kOther, y);
  } else {
    // GGLSE( M = m, N = n, P = p ): A is m x n, B p x n, C(m), D(p), X(n).
    const I lda = std::max<I>(m, 1), ldb = std::max<I>(p, 1);
    std::vector<T> x(std::max<I>(n, 1));
    lib.Get<F>((std::string(Prefix<T>()) + "gglse_").c_str())(
        &m, &n, &p, a.data(), &lda, b.data(), &ldb, c.data(), d.data(),
        x.data(), work.data(), &lwork, &info);
    r.lwork = std::real(work[0]);
    AddInt(r, "info", info);
    if (lwork == -1) return r;
    Add(r, "x", Kind::kOther, x);
    Add(r, "c", Kind::kOther, c);
  }
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  return r;
}

// ggqrf (rq = false; A n x m, B n x p) or ggrqf (rq = true; A m x n, B p x
// n), with the dimensions in the routine's order (d1, d2, d3).
template <typename T>
Result Ggqrf(const BlasLibrary& lib, bool rq, std::vector<T> a, I d1, I d2,
             I d3, std::vector<T> b, I lwork) {
  // ggqrf( N, M, P ): lda, ldb >= N. ggrqf( M, P, N ): lda >= M, ldb >= P.
  const I lda = std::max<I>(d1, 1), ldb = std::max<I>(rq ? d2 : d1, 1);
  std::vector<T> taua(std::max<I>({d1, d2, d3, 1})), taub(taua.size());
  std::vector<T> work(std::max<I>(lwork, 1));
  I info = 0;
  using F = void (*)(const I*, const I*, const I*, T*, const I*, T*, T*,
                     const I*, T*, T*, const I*, I*);
  lib.Get<F>((std::string(Prefix<T>()) + (rq ? "ggrqf_" : "ggqrf_")).c_str())(
      &d1, &d2, &d3, a.data(), &lda, taua.data(), b.data(), &ldb, taub.data(),
      work.data(), &lwork, &info);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  if (lwork == -1) return r;
  Add(r, "a", Kind::kOther, a);
  Add(r, "taua", Kind::kOther, taua);
  Add(r, "b", Kind::kOther, b);
  Add(r, "taub", Kind::kOther, taub);
  return r;
}

// The orthogonal factor of an RQ (rz = false; ormrq / unmrq, A k x nq with
// TAU from gerqf) or RZ factorization (rz = true; ormrz / unmrz, A k x nq
// from tzrzf, l = nq - k) applied to the m x n matrix C.
template <typename T>
Result Ormrq(const BlasLibrary& lib, bool rz, const std::vector<T>& a,
             const std::vector<T>& tau, I k, I nq, std::vector<T> c, I m, I n,
             char side, char trans, I lwork) {
  const I lda = std::max<I>(k, 1), ldc = std::max<I>(m, 1);
  const I l = nq - k;
  // As LAPACK, the query gives 1 for an empty C, but LWORK >= max(1, N) or
  // max(1, M) is required.
  if (lwork != -1) lwork = std::max<I>({lwork, m, n, 1});
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<T> af = a;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) +
                           (kComplex<T> ? "unm" : "orm") + (rz ? "rz_" : "rq_");
  if (rz) {
    using F = void (*)(const char*, const char*, const I*, const I*, const I*,
                       const I*, const T*, const I*, const T*, T*, const I*, T*,
                       const I*, I*, size_t, size_t);
    lib.Get<F>(name.c_str())(&side, &trans, &m, &n, &k, &l, af.data(), &lda,
                             tau.data(), c.data(), &ldc, work.data(), &lwork,
                             &info, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const I*, const I*, const I*,
                       const T*, const I*, const T*, T*, const I*, T*, const I*,
                       I*, size_t, size_t);
    lib.Get<F>(name.c_str())(&side, &trans, &m, &n, &k, af.data(), &lda,
                             tau.data(), c.data(), &ldc, work.data(), &lwork,
                             &info, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  if (lwork == -1) return r;
  Add(r, "c", Kind::kOther, c);
  Add(r, "a", Kind::kOther, af);  // restored by the complex routines
  return r;
}

// The least squares drivers for problems of order s: square, tall and wide
// matrices (random, graded, small integers, rank deficient), one and three
// right-hand sides, and RCOND below the machine precision and 1e-3
// (xGELSS, xGELSD, xGELSY); the generalized problems (xGGLSE, xGGGLM); and
// the factorizations they use: xGERQF (and xGERQ2), xORMRQ / xUNMRQ,
// xTZRZF, xORMRZ / xUNMRZ, xGGQRF, xGGRQF. The routines that take a
// factorization get hwyBLAS's in both libraries.
template <typename T>
void RunGels(const BlasLibrary& hwy, const BlasLibrary& ref, I s) {
  using R = Real<T>;
  char what[96];
  const char ctrans = kComplex<T> ? 'C' : 'T';
  const I shapes[][2] = {{s, s}, {2 * s + 1, s}, {s, 2 * s + 1}};
  for (const auto& shape : shapes) {
    const I m = shape[0], n = shape[1];
    const I mn = std::max(m, n);
    for (const int kind : {0, 2, 4, 5}) {
      const std::vector<T> a =
          kind == 5 ? LowRank<T>(m, n, std::max<I>(std::min(m, n) / 2, 1))
                    : TestMatrix<T>(kind, m, n);
      for (const I nrhs : {1, 3}) {
        const std::vector<T> b = RightHandSides<T>(m, n, nrhs);
        // xGELS needs full rank.
        if (kind != 5) {
          for (const char trans : {'N', ctrans}) {
            snprintf(what, sizeof(what), "m=%d n=%d kind=%d nrhs=%d trans=%c",
                     int(m), int(n), kind, int(nrhs), trans);
            Case<T>(hwy, ref, "gels", what, mn,
                    [&](const BlasLibrary& lib, I lw) {
                      return Gels(lib, trans, a, m, n, b, nrhs, lw);
                    });
          }
        }
        for (const R rcond : {R(-1), R(1e-3)}) {
          snprintf(what, sizeof(what), "m=%d n=%d kind=%d nrhs=%d rcond=%g",
                   int(m), int(n), kind, int(nrhs), double(rcond));
          Case<T>(hwy, ref, "gelss", what, mn,
                  [&](const BlasLibrary& lib, I lw) {
                    return GelssD(lib, false, a, m, n, b, nrhs, rcond, lw);
                  });
          Case<T>(hwy, ref, "gelsd", what, mn,
                  [&](const BlasLibrary& lib, I lw) {
                    return GelssD(lib, true, a, m, n, b, nrhs, rcond, lw);
                  });
          Case<T>(hwy, ref, "gelsy", what, mn,
                  [&](const BlasLibrary& lib, I lw) {
                    return Gelsy(lib, a, m, n, b, nrhs, rcond, lw);
                  });
        }
      }
      if (kind == 5) continue;

      // RQ factorization and its Q, applied from both sides.
      snprintf(what, sizeof(what), "m=%d n=%d kind=%d", int(m), int(n), kind);
      Case<T>(hwy, ref, "gerqf", what, mn, [&](const BlasLibrary& lib, I lw) {
        return Geqrf(lib, "gerqf_", a, m, n, lw);
      });
      Case<T>(hwy, ref, "gerq2", what, mn,
              [&](const BlasLibrary& lib, I) { return Gerq2(lib, a, m, n); });
    }
  }

  // The factors of RQ (A k x nq, k <= nq) and RZ (A k x nq upper
  // trapezoidal) factorizations, applied to C.
  for (const I k : {s, s / 2}) {
    const I nq = s + (s + 1) / 2;
    if (k > nq) continue;
    std::vector<T> a = TestMatrix<T>(0, k, nq);
    std::vector<T> tau(std::max<I>(k, 1));
    {
      using F =
          void (*)(const I*, const I*, T*, const I*, T*, T*, const I*, I*);
      const I lda = std::max<I>(k, 1), lwork = 64 * std::max<I>(nq, 1);
      std::vector<T> work(lwork);
      I info = 0;
      hwy.Get<F>((std::string(Prefix<T>()) + "gerqf_").c_str())(
          &k, &nq, a.data(), &lda, tau.data(), work.data(), &lwork, &info);
    }
    // A k x nq upper trapezoidal matrix.
    std::vector<T> z = TestMatrix<T>(0, k, nq);
    for (I j = 0; j < nq; ++j) {
      for (I i = j + 1; i < k; ++i) z[i + static_cast<size_t>(j) * k] = T(0);
    }
    snprintf(what, sizeof(what), "m=%d n=%d", int(k), int(nq));
    Case<T>(hwy, ref, "tzrzf", what, nq, [&](const BlasLibrary& lib, I lw) {
      return Geqrf(lib, "tzrzf_", z, k, nq, lw);
    });
    std::vector<T> ztau(std::max<I>(k, 1));
    {
      using F =
          void (*)(const I*, const I*, T*, const I*, T*, T*, const I*, I*);
      const I lda = std::max<I>(k, 1), lwork = 64 * std::max<I>(nq, 1);
      std::vector<T> work(lwork);
      I info = 0;
      hwy.Get<F>((std::string(Prefix<T>()) + "tzrzf_").c_str())(
          &k, &nq, z.data(), &lda, ztau.data(), work.data(), &lwork, &info);
    }
    const I nc = 3;
    for (const char side : {'L', 'R'}) {
      for (const char trans : {'N', ctrans}) {
        const I cm = side == 'L' ? nq : nc, cn = side == 'L' ? nc : nq;
        const std::vector<T> c = TestMatrix<T>(0, cm, cn);
        snprintf(what, sizeof(what), "k=%d nq=%d side=%c trans=%c", int(k),
                 int(nq), side, trans);
        Case<T>(hwy, ref, kComplex<T> ? "unmrq" : "ormrq", what, nq,
                [&](const BlasLibrary& lib, I lw) {
                  return Ormrq(lib, false, a, tau, k, nq, c, cm, cn, side,
                               trans, lw);
                });
        Case<T>(hwy, ref, kComplex<T> ? "unmrz" : "ormrz", what, nq,
                [&](const BlasLibrary& lib, I lw) {
                  return Ormrq(lib, true, z, ztau, k, nq, c, cm, cn, side,
                               trans, lw);
                });
      }
    }
  }

  // Generalized problems: GGLSE with A m x n, B p x n (p <= n <= m + p),
  // GGGLM with A n x m, B n x p (m <= n <= m + p), and the generalized QR
  // and RQ factorizations.
  const I sizes[][3] = {{s + 2, s, s / 2}, {s / 2, s, s - s / 3}, {s, s, s}};
  for (const auto& d : sizes) {
    const I m = d[0], n = d[1], p = d[2];
    if (p <= n && n <= m + p) {
      const std::vector<T> a = TestMatrix<T>(0, m, n);
      const std::vector<T> b = TestMatrix<T>(0, p, n);
      const std::vector<T> c = TestMatrix<T>(0, std::max<I>(m, 1), 1);
      const std::vector<T> dd = TestMatrix<T>(0, std::max<I>(p, 1), 1);
      snprintf(what, sizeof(what), "m=%d n=%d p=%d", int(m), int(n), int(p));
      Case<T>(hwy, ref, "gglse", what, n, [&](const BlasLibrary& lib, I lw) {
        return Gglse(lib, false, a, m, n, p, b, c, dd, lw);
      });
      Case<T>(hwy, ref, "ggrqf", what, n, [&](const BlasLibrary& lib, I lw) {
        return Ggqrf(lib, true, a, m, p, n, b, lw);
      });
    }
    // GGGLM( N, M, P ) with N = n, M = m', P = p': here m' = n - p / 2.
    const I gn = n, gm = std::max<I>(n - p / 2, 0), gp = p;
    if (gm <= gn && gn <= gm + gp) {
      const std::vector<T> a = TestMatrix<T>(0, gn, gm);
      const std::vector<T> b = TestMatrix<T>(0, gn, gp);
      const std::vector<T> dd = TestMatrix<T>(0, std::max<I>(gn, 1), 1);
      snprintf(what, sizeof(what), "n=%d m=%d p=%d", int(gn), int(gm), int(gp));
      Case<T>(hwy, ref, "ggglm", what, gn, [&](const BlasLibrary& lib, I lw) {
        return Gglse(lib, true, a, gn, gm, gp, b, std::vector<T>(), dd, lw);
      });
      Case<T>(hwy, ref, "ggqrf", what, gn, [&](const BlasLibrary& lib, I lw) {
        return Ggqrf(lib, false, a, gn, gm, gp, b, lw);
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Band and tridiagonal systems

// The band storage (leading dimension ldab, kv superdiagonals in it) of the
// band of the m x n matrix a with kl subdiagonals and ku superdiagonals;
// zero elsewhere.
template <typename T>
std::vector<T> BandOf(const std::vector<T>& a, I m, I n, I kl, I ku, I kv,
                      I ldab) {
  std::vector<T> ab(static_cast<size_t>(ldab) * std::max<I>(n, 1));
  for (I j = 0; j < n; ++j) {
    for (I i = std::max<I>(0, j - ku); i < std::min(m, j + kl + 1); ++i) {
      ab[kv + i - j + static_cast<size_t>(j) * ldab] =
          a[i + static_cast<size_t>(j) * m];
    }
  }
  return ab;
}

// How the pivots of the band LU factorization are compared: exactly for
// real T; for complex T only counted, as the factors, since flang's complex
// division of the multipliers rounds differently and can flip near-ties
// (seen in CGBTRF of order 100 and 150).
template <typename T>
constexpr Kind BandPivots() {
  return kComplex<T> ? Kind::kOther : Kind::kInteger;
}

// gbtrf or gbtf2 (unblocked).
template <typename T>
Result Gbtrf(const BlasLibrary& lib, bool unblocked, std::vector<T> ab, I m,
             I n, I kl, I ku) {
  using F =
      void (*)(const I*, const I*, const I*, const I*, T*, const I*, I*, I*);
  const I ldab = 2 * kl + ku + 1;
  std::vector<I> ipiv(std::max<I>(std::min(m, n), 1));
  I info = 0;
  lib.Get<F>(
      (std::string(Prefix<T>()) + (unblocked ? "gbtf2_" : "gbtrf_")).c_str())(
      &m, &n, &kl, &ku, ab.data(), &ldab, ipiv.data(), &info);
  Result r;
  AddInt(r, "info", info);
  Add(r, "ipiv", BandPivots<T>(), ipiv);
  Add(r, "ab", Kind::kOther, ab);
  return r;
}

template <typename T>
Result Gbtrs(const BlasLibrary& lib, char trans, const std::vector<T>& lu,
             const std::vector<I>& ipiv, I n, I kl, I ku, std::vector<T> b,
             I nrhs) {
  using F = void (*)(const char*, const I*, const I*, const I*, const I*,
                     const T*, const I*, const I*, T*, const I*, I*, size_t);
  const I ldab = 2 * kl + ku + 1, ldb = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "gbtrs_").c_str())(
      &trans, &n, &kl, &ku, &nrhs, lu.data(), &ldab, ipiv.data(), b.data(),
      &ldb, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "b", Kind::kOther, b);
  return r;
}

template <typename T>
Result Gbsv(const BlasLibrary& lib, std::vector<T> ab, I n, I kl, I ku,
            std::vector<T> b, I nrhs) {
  using F = void (*)(const I*, const I*, const I*, const I*, T*, const I*, I*,
                     T*, const I*, I*);
  const I ldab = 2 * kl + ku + 1, ldb = std::max<I>(n, 1);
  std::vector<I> ipiv(std::max<I>(n, 1));
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "gbsv_").c_str())(
      &n, &kl, &ku, &nrhs, ab.data(), &ldab, ipiv.data(), b.data(), &ldb,
      &info);
  Result r;
  AddInt(r, "info", info);
  Add(r, "ipiv", BandPivots<T>(), ipiv);
  Add(r, "ab", Kind::kOther, ab);
  Add(r, "b", Kind::kOther, b);
  return r;
}

template <typename T>
Result Gbcon(const BlasLibrary& lib, char norm, const std::vector<T>& lu,
             const std::vector<I>& ipiv, I n, I kl, I ku, Real<T> anorm) {
  using R = Real<T>;
  const I ldab = 2 * kl + ku + 1, ld = std::max<I>(n, 1);
  R rcond = -1;
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "gbcon_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const I*, const I*, const T*,
                       const I*, const I*, const R*, R*, T*, R*, I*, size_t);
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(ld);
    lib.Get<F>(name.c_str())(&norm, &n, &kl, &ku, lu.data(), &ldab, ipiv.data(),
                             &anorm, &rcond, work.data(), rwork.data(), &info,
                             1);
  } else {
    using F = void (*)(const char*, const I*, const I*, const I*, const T*,
                       const I*, const I*, const R*, R*, T*, I*, I*, size_t);
    std::vector<T> work(3 * ld);
    std::vector<I> iwork(ld);
    lib.Get<F>(name.c_str())(&norm, &n, &kl, &ku, lu.data(), &ldab, ipiv.data(),
                             &anorm, &rcond, work.data(), iwork.data(), &info,
                             1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "rcond", Kind::kValues, &rcond, 1);
  return r;
}

// latbs with the triangular band matrix ab (kd + 1 rows) and normin = 'N'.
template <typename T>
Result Latbs(const BlasLibrary& lib, char uplo, char trans, char diag,
             const std::vector<T>& ab, I n, I kd, std::vector<T> x,
             Real<T> xscale) {
  for (T& v : x) v *= xscale;
  using R = Real<T>;
  using F = void (*)(const char*, const char*, const char*, const char*,
                     const I*, const I*, const T*, const I*, T*, R*, R*, I*,
                     size_t, size_t, size_t, size_t);
  const I ldab = kd + 1;
  const char normin = 'N';
  std::vector<R> cnorm(std::max<I>(n, 1));
  R scale = -1;
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "latbs_").c_str())(
      &uplo, &trans, &diag, &normin, &n, &kd, ab.data(), &ldab, x.data(),
      &scale, cnorm.data(), &info, 1, 1, 1, 1);
  Result r;
  AddInt(r, "info", info);
  // SCALE and CNORM with the tolerance for complex T (they show the scaling
  // decisions). The complex X is not, nor SCALE of the lower triangular
  // A**T and A**H solves: C/ZLATBS in the reference drop an element of A
  // there in scaled solves (see LatbsComplex in src/lapack/gbsv.cc).
  Add(r, "scale", uplo == 'L' && trans != 'N' ? Kind::kOther : Kind::kValues,
      &scale, 1);
  Add(r, "x", Kind::kOther, x);
  Add(r, "cnorm", Kind::kValues, cnorm);
  return r;
}

// pbtrf, pbtf2 (unblocked), potf2 (kd = -1: the full matrix ab of order n)
// or pbsv (b given).
template <typename T>
Result Pbtrf(const BlasLibrary& lib, const char* name, char uplo,
             std::vector<T> ab, I n, I kd, std::vector<T> b = {}, I nrhs = 0) {
  const I ldab = kd < 0 ? std::max<I>(n, 1) : kd + 1, ldb = std::max<I>(n, 1);
  const std::string f = std::string(Prefix<T>()) + name;
  I info = 0;
  if (kd < 0) {
    using F = void (*)(const char*, const I*, T*, const I*, I*, size_t);
    lib.Get<F>(f.c_str())(&uplo, &n, ab.data(), &ldab, &info, 1);
  } else if (b.empty()) {
    using F =
        void (*)(const char*, const I*, const I*, T*, const I*, I*, size_t);
    lib.Get<F>(f.c_str())(&uplo, &n, &kd, ab.data(), &ldab, &info, 1);
  } else {
    using F = void (*)(const char*, const I*, const I*, const I*, T*, const I*,
                       T*, const I*, I*, size_t);
    lib.Get<F>(f.c_str())(&uplo, &n, &kd, &nrhs, ab.data(), &ldab, b.data(),
                          &ldb, &info, 1);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "ab", Kind::kOther, ab);
  if (!b.empty()) Add(r, "b", Kind::kOther, b);
  return r;
}

template <typename T>
Result Pbtrs(const BlasLibrary& lib, char uplo, const std::vector<T>& f, I n,
             I kd, std::vector<T> b, I nrhs) {
  using F = void (*)(const char*, const I*, const I*, const I*, const T*,
                     const I*, T*, const I*, I*, size_t);
  const I ldab = kd + 1, ldb = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "pbtrs_").c_str())(
      &uplo, &n, &kd, &nrhs, f.data(), &ldab, b.data(), &ldb, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "b", Kind::kOther, b);
  return r;
}

// A tridiagonal matrix and its factorization.
template <typename T>
struct Tri {
  std::vector<T> dl, d, du, du2;
  std::vector<I> ipiv;
};

// gttrf (b empty) or gtsv.
template <typename T>
Result Gttrf(const BlasLibrary& lib, Tri<T> t, I n, std::vector<T> b = {},
             I nrhs = 0, Tri<T>* factor = nullptr) {
  I info = 0;
  const I ldb = std::max<I>(n, 1);
  t.du2.assign(std::max<I>(n, 1), T(0));
  t.ipiv.assign(std::max<I>(n, 1), 0);
  if (b.empty()) {
    using F = void (*)(const I*, T*, T*, T*, T*, I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "gttrf_").c_str())(
        &n, t.dl.data(), t.d.data(), t.du.data(), t.du2.data(), t.ipiv.data(),
        &info);
  } else {
    using F = void (*)(const I*, const I*, T*, T*, T*, T*, const I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "gtsv_").c_str())(
        &n, &nrhs, t.dl.data(), t.d.data(), t.du.data(), b.data(), &ldb, &info);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "dl", Kind::kOther, t.dl);
  Add(r, "d", Kind::kOther, t.d);
  Add(r, "du", Kind::kOther, t.du);
  if (b.empty()) {
    Add(r, "du2", Kind::kOther, t.du2);
    Add(r, "ipiv", Kind::kInteger, t.ipiv);
  } else {
    Add(r, "b", Kind::kOther, b);
  }
  if (factor != nullptr) *factor = t;
  return r;
}

template <typename T>
Result Gttrs(const BlasLibrary& lib, char trans, const Tri<T>& f, I n,
             std::vector<T> b, I nrhs) {
  using F = void (*)(const char*, const I*, const I*, const T*, const T*,
                     const T*, const T*, const I*, T*, const I*, I*, size_t);
  const I ldb = std::max<I>(n, 1);
  I info = 0;
  lib.Get<F>((std::string(Prefix<T>()) + "gttrs_").c_str())(
      &trans, &n, &nrhs, f.dl.data(), f.d.data(), f.du.data(), f.du2.data(),
      f.ipiv.data(), b.data(), &ldb, &info, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "b", Kind::kOther, b);
  return r;
}

// pttrf (b empty) or ptsv.
template <typename T>
Result Pttrf(const BlasLibrary& lib, std::vector<Real<T>> d, std::vector<T> e,
             I n, std::vector<T> b = {}, I nrhs = 0) {
  using R = Real<T>;
  I info = 0;
  const I ldb = std::max<I>(n, 1);
  if (b.empty()) {
    using F = void (*)(const I*, R*, T*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "pttrf_").c_str())(&n, d.data(),
                                                              e.data(), &info);
  } else {
    using F = void (*)(const I*, const I*, R*, T*, T*, const I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "ptsv_").c_str())(
        &n, &nrhs, d.data(), e.data(), b.data(), &ldb, &info);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "d", Kind::kOther, d);
  Add(r, "e", Kind::kOther, e);
  if (!b.empty()) Add(r, "b", Kind::kOther, b);
  return r;
}

// pttrs (uplo for complex T only).
template <typename T>
Result Pttrs(const BlasLibrary& lib, char uplo, const std::vector<Real<T>>& d,
             const std::vector<T>& e, I n, std::vector<T> b, I nrhs) {
  using R = Real<T>;
  const I ldb = std::max<I>(n, 1);
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "pttrs_";
  if constexpr (kComplex<T>) {
    using F = void (*)(const char*, const I*, const I*, const R*, const T*, T*,
                       const I*, I*, size_t);
    lib.Get<F>(name.c_str())(&uplo, &n, &nrhs, d.data(), e.data(), b.data(),
                             &ldb, &info, 1);
  } else {
    using F =
        void (*)(const I*, const I*, const R*, const T*, T*, const I*, I*);
    lib.Get<F>(name.c_str())(&n, &nrhs, d.data(), e.data(), b.data(), &ldb,
                             &info);
  }
  Result r;
  AddInt(r, "info", info);
  Add(r, "b", Kind::kOther, b);
  return r;
}

// The band and tridiagonal routines for matrices of order s: general band
// matrices (square, and with three more rows or columns for the
// factorization) with several bandwidths, also wide enough for the blocked
// code (ILAENV's block size for xGBTRF and xPBTRF is 32 only for more than
// 64 superdiagonals), random, graded and with small integers (ties);
// xLATBS with matrices that need scaling; positive definite band matrices;
// general and positive definite tridiagonal matrices. The routines that take
// a factorization get hwyBLAS's in both libraries.
template <typename T>
void RunBand(const BlasLibrary& hwy, const BlasLibrary& ref, I s) {
  using R = Real<T>;
  // The other drivers get the same matrices with and without these cases.
  const std::mt19937_64 saved = rng;
  char what[96];
  const I nrhs = 3;
  const std::vector<T> b = TestMatrix<T>(0, std::max<I>(s, 1), nrhs);
  const I bands[][2] = {{0, 0},
                        {1, 1},
                        {s / 4 + 1, s / 3},
                        {s / 2, 2},
                        {std::min<I>(s, 40), std::min<I>(s, 70)}};
  for (const auto& band : bands) {
    const I kl = band[0], ku = band[1], ldab = 2 * kl + ku + 1;
    for (const int kind : {0, 2, 4}) {
      for (const I dm : {0, 3, -3}) {
        const I m = dm > 0 ? s + dm : s, n = dm < 0 ? s - dm : s;
        const std::vector<T> ab =
            BandOf(TestMatrix<T>(kind, m, n), m, n, kl, ku, kl + ku, ldab);
        snprintf(what, sizeof(what), "m=%d n=%d kl=%d ku=%d kind=%d", int(m),
                 int(n), int(kl), int(ku), kind);
        for (const bool unblocked : {false, true}) {
          Case<T>(hwy, ref, unblocked ? "gbtf2" : "gbtrf", what, n,
                  [&](const BlasLibrary& lib, I) {
                    return Gbtrf(lib, unblocked, ab, m, n, kl, ku);
                  });
        }
        if (m != n) continue;
        Case<T>(hwy, ref, "gbsv", what, n, [&](const BlasLibrary& lib, I) {
          return Gbsv(lib, ab, n, kl, ku, b, nrhs);
        });
        // hwyBLAS's factorization for xGBTRS and xGBCON.
        std::vector<T> lu = ab;
        std::vector<I> ipiv(std::max<I>(n, 1));
        {
          using F = void (*)(const I*, const I*, const I*, const I*, T*,
                             const I*, I*, I*);
          I info = 0;
          hwy.Get<F>((std::string(Prefix<T>()) + "gbtrf_").c_str())(
              &n, &n, &kl, &ku, lu.data(), &ldab, ipiv.data(), &info);
          if (info != 0) continue;
        }
        for (const char trans : {'N', 'T', 'C'}) {
          snprintf(what, sizeof(what), "n=%d kl=%d ku=%d kind=%d trans=%c",
                   int(n), int(kl), int(ku), kind, trans);
          Case<T>(hwy, ref, "gbtrs", what, n, [&](const BlasLibrary& lib, I) {
            return Gbtrs(lib, trans, lu, ipiv, n, kl, ku, b, nrhs);
          });
        }
        for (const char norm : {'1', 'I'}) {
          snprintf(what, sizeof(what), "n=%d kl=%d ku=%d kind=%d norm=%c",
                   int(n), int(kl), int(ku), kind, norm);
          Case<T>(hwy, ref, "gbcon", what, n, [&](const BlasLibrary& lib, I) {
            return Gbcon(lib, norm, lu, ipiv, n, kl, ku, R(n + 1));
          });
        }
      }
    }
  }

  // xGBCON with solves that scale: an upper triangular band matrix (KL =
  // 0, so that U = A) with a tiny diagonal.
  for (const I ku : {I(1), I(3)}) {
    const I ldab = ku + 1;
    std::vector<T> u = TestMatrix<T>(0, ldab, std::max<I>(s, 1));
    for (I j = 0; j < s; ++j) {
      u[ku + static_cast<size_t>(j) * ldab] *=
          std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon() *
          R(1 << 20);
    }
    std::vector<I> ipiv(std::max<I>(s, 1));
    for (I j = 0; j < s; ++j) ipiv[j] = j + 1;
    for (const char norm : {'1', 'I'}) {
      snprintf(what, sizeof(what), "n=%d ku=%d tiny diagonal norm=%c", int(s),
               int(ku), norm);
      Case<T>(hwy, ref, "gbcon", what, s, [&](const BlasLibrary& lib, I) {
        return Gbcon(lib, norm, u, ipiv, s, 0, ku, R(2));
      });
    }
  }

  // xLATBS: kind 0 random with a dominant diagonal (xTBSV), 1 a tiny
  // diagonal (growth: the scaled solve), 2 huge off-diagonal elements
  // (CNORM scaled by TSCAL), 3 a zero on the diagonal (SCALE = 0); b of
  // order 1 or near the overflow threshold.
  const R smlnum =
      std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon();
  for (const I kd : {I(0), I(1), I(3), std::min<I>(s, 20)}) {
    for (const int kind : {0, 1, 2, 3}) {
      std::vector<T> ab = TestMatrix<T>(0, kd + 1, s);
      for (const char uplo : {'U', 'L'}) {
        const I dr = uplo == 'U' ? kd : 0;
        std::vector<T> t = ab;
        for (I j = 0; j < s; ++j) {
          for (I i = 0; i <= kd; ++i) {
            T& x = t[i + static_cast<size_t>(j) * (kd + 1)];
            if (i == dr) {
              if (kind == 0) x += T(R(kd + 2));
              if (kind == 1) x *= smlnum * R(64);
            } else if (kind == 2) {
              x *= std::numeric_limits<R>::max() / R(16 * (kd + 1));
            }
          }
        }
        if (kind == 3 && s > 0)
          t[dr + static_cast<size_t>(s / 2) * (kd + 1)] = T(0);
        for (const char trans : {'N', 'T', 'C'}) {
          for (const char diag : {'N', 'U'}) {
            snprintf(what, sizeof(what), "n=%d kd=%d kind=%d %c%c%c", int(s),
                     int(kd), kind, uplo, trans, diag);
            for (const R xscale : {R(1), R(0.25) / smlnum}) {
              Case<T>(
                  hwy, ref, "latbs", what, s, [&](const BlasLibrary& lib, I) {
                    return Latbs(lib, uplo, trans, diag, t, s, kd,
                                 std::vector<T>(b.begin(),
                                                b.begin() + std::max<I>(s, 1)),
                                 xscale);
                  });
            }
          }
        }
      }
    }
  }

  // Positive definite band matrices, and xPOTF2.
  const std::vector<T> spd = PositiveDefinite<T>(s);
  for (const char uplo : {'U', 'L'}) {
    snprintf(what, sizeof(what), "n=%d uplo=%c", int(s), uplo);
    Case<T>(hwy, ref, "potf2", what, s, [&](const BlasLibrary& lib, I) {
      return Pbtrf(lib, "potf2_", uplo, spd, s, -1);
    });
    for (const I kd : {I(0), I(1), s / 3, std::min<I>(s, 70)}) {
      const I ld = kd + 1;
      // Random, with a dominant real diagonal.
      std::vector<T> pb = TestMatrix<T>(0, ld, std::max<I>(s, 1));
      for (I j = 0; j < s; ++j) {
        T& x = pb[(uplo == 'U' ? kd : 0) + static_cast<size_t>(j) * ld];
        x = T(R(2 * kd + 2) + std::real(x));
      }
      snprintf(what, sizeof(what), "n=%d kd=%d uplo=%c", int(s), int(kd), uplo);
      for (const char* name : {"pbtrf_", "pbtf2_"}) {
        Case<T>(hwy, ref, std::string(name).substr(0, 5), what, s,
                [&](const BlasLibrary& lib, I) {
                  return Pbtrf(lib, name, uplo, pb, s, kd);
                });
      }
      Case<T>(hwy, ref, "pbsv", what, s, [&](const BlasLibrary& lib, I) {
        return Pbtrf(lib, "pbsv_", uplo, pb, s, kd, b, nrhs);
      });
      std::vector<T> f = pb;
      {
        using F =
            void (*)(const char*, const I*, const I*, T*, const I*, I*, size_t);
        I info = 0;
        hwy.Get<F>((std::string(Prefix<T>()) + "pbtrf_").c_str())(
            &uplo, &s, &kd, f.data(), &ld, &info, 1);
        if (info != 0) continue;
      }
      Case<T>(hwy, ref, "pbtrs", what, s, [&](const BlasLibrary& lib, I) {
        return Pbtrs(lib, uplo, f, s, kd, b, nrhs);
      });
    }
  }

  // Tridiagonal matrices: random, with small diagonals (row interchanges),
  // and with small integers (ties, zero subdiagonals).
  for (const int kind : {0, 1, 4}) {
    const std::vector<T> g =
        TestMatrix<T>(kind == 4 ? 4 : 0, 3, std::max<I>(s, 1));
    Tri<T> t;
    for (I i = 0; i < std::max<I>(s, 1); ++i) {
      t.dl.push_back(g[3 * i]);
      t.d.push_back(kind == 1 ? g[3 * i + 1] * R(0.01) : g[3 * i + 1]);
      t.du.push_back(g[3 * i + 2]);
    }
    for (const I k : {I(1), I(2), nrhs}) {
      snprintf(what, sizeof(what), "n=%d kind=%d nrhs=%d", int(s), kind,
               int(k));
      Case<T>(hwy, ref, "gtsv", what, s, [&](const BlasLibrary& lib, I) {
        return Gttrf(lib, t, s, b, k);
      });
    }
    snprintf(what, sizeof(what), "n=%d kind=%d", int(s), kind);
    Tri<T> f;
    Case<T>(hwy, ref, "gttrf", what, s, [&](const BlasLibrary& lib, I) {
      return Gttrf(lib, t, s, {}, 0, &lib == &hwy ? &f : nullptr);
    });
    for (const char trans : {'N', 'T', 'C'}) {
      for (const I k : {I(1), nrhs}) {
        snprintf(what, sizeof(what), "n=%d kind=%d trans=%c nrhs=%d", int(s),
                 kind, trans, int(k));
        Case<T>(hwy, ref, "gttrs", what, s, [&](const BlasLibrary& lib, I) {
          return Gttrs(lib, trans, f, s, b, k);
        });
      }
    }
  }
  // Positive definite tridiagonal matrices.
  {
    std::vector<R> d;
    std::vector<T> e;
    for (I i = 0; i < std::max<I>(s, 1); ++i) {
      d.push_back(R(4) + std::real(Random<T>()));
      e.push_back(Random<T>());
    }
    snprintf(what, sizeof(what), "n=%d", int(s));
    Case<T>(hwy, ref, "pttrf", what, s,
            [&](const BlasLibrary& lib, I) { return Pttrf<T>(lib, d, e, s); });
    for (const I k : {I(1), I(2), nrhs}) {
      snprintf(what, sizeof(what), "n=%d nrhs=%d", int(s), int(k));
      Case<T>(hwy, ref, "ptsv", what, s, [&](const BlasLibrary& lib, I) {
        return Pttrf<T>(lib, d, e, s, b, k);
      });
    }
    std::vector<R> df = d;
    std::vector<T> ef = e;
    {
      using F = void (*)(const I*, R*, T*, I*);
      I info = 0;
      hwy.Get<F>((std::string(Prefix<T>()) + "pttrf_").c_str())(
          &s, df.data(), ef.data(), &info);
    }
    for (const char uplo : {'U', 'L'}) {
      for (const I k : {I(1), I(2), nrhs}) {
        snprintf(what, sizeof(what), "n=%d uplo=%c nrhs=%d", int(s), uplo,
                 int(k));
        Case<T>(hwy, ref, "pttrs", what, s, [&](const BlasLibrary& lib, I) {
          return Pttrs<T>(lib, uplo, df, ef, s, b, k);
        });
      }
    }
  }
  rng = saved;
}

// ---------------------------------------------------------------------------
// Generalized nonsymmetric eigenproblem and GSVD. The computational routines
// get hwyBLAS's results of the preceding steps (xGGHRD's Hessenberg form
// for xHGEQZ, xGGES's Schur form for xTGEVC, xTGSEN, xTGEXC, xTGSYL) in
// both libraries.

// Selects the eigenvalues alpha / beta with a negative real part.
template <typename R>
I SelectNegative3(const R* ar, const R* /*ai*/, const R* b) {
  return *ar * *b < 0 ? 1 : 0;
}
template <typename R>
I SelectNegative2(const R* a, const R* b) {
  return a[0] * b[0] + a[1] * b[1] < 0 ? 1 : 0;
}

// The eigenvalues: real T, ALPHAR, ALPHAI and BETA (bitwise); complex T,
// ALPHA and BETA (counted) and the set of ALPHA / BETA (when `compare_w`
// and all BETA are nonzero).
template <typename T>
void AddPencil(Result& r, const std::vector<T>& alpha,
               const std::vector<T>& alphai, const std::vector<T>& beta, I n,
               bool compare_w) {
  const size_t m = static_cast<size_t>(n);
  Add(r, "alpha", Kind::kOther, alpha.data(), m);
  if (!kComplex<T>) Add(r, "alphai", Kind::kOther, alphai.data(), m);
  Add(r, "beta", Kind::kOther, beta.data(), m);
  if constexpr (kComplex<T>) {
    bool finite = compare_w;
    for (I i = 0; i < n; ++i) finite = finite && beta[i] != T(0);
    if (finite) {
      std::vector<T> w(m);
      for (I i = 0; i < n; ++i) w[i] = alpha[i] / beta[i];
      Add(r, "lambda", Kind::kEigenvalues, w);
    }
  }
}

template <typename T>
Result Ggbal(const BlasLibrary& lib, std::vector<T> a, std::vector<T> b, I n,
             char job, const std::vector<T>& v, I m) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<R> ls(ld), rs(ld), work(std::max<I>(6 * n, 1));
  I ilo = 0, ihi = 0, info = 0;
  const std::string p = Prefix<T>();
  using F = void (*)(const char*, const I*, T*, const I*, T*, const I*, I*, I*,
                     R*, R*, R*, I*, size_t);
  lib.Get<F>((p + "ggbal_").c_str())(&job, &n, a.data(), &ld, b.data(), &ld,
                                     &ilo, &ihi, ls.data(), rs.data(),
                                     work.data(), &info, 1);
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "ilo", ilo);
  AddInt(r, "ihi", ihi);
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  Add(r, "lscale", Kind::kOther, ls);
  Add(r, "rscale", Kind::kOther, rs);
  // xGGBAK with the balancing of this library.
  using G =
      void (*)(const char*, const char*, const I*, const I*, const I*, const R*,
               const R*, const I*, T*, const I*, I*, size_t, size_t);
  for (const char side : {'L', 'R'}) {
    std::vector<T> x = v;
    lib.Get<G>((p + "ggbak_").c_str())(&job, &side, &n, &ilo, &ihi, ls.data(),
                                       rs.data(), &m, x.data(), &ld, &info, 1,
                                       1);
    AddInt(r, "info", info);
    Add(r, side == 'L' ? "vl" : "vr", Kind::kOther, x);
  }
  return r;
}

template <typename T>
Result Gghrd(const BlasLibrary& lib, std::vector<T> a, std::vector<T> b, I n,
             I ilo, I ihi, char comp, const std::vector<T>& q0) {
  const I ld = std::max<I>(n, 1);
  std::vector<T> q = q0, z = q0;
  I info = 0;
  using F = void (*)(const char*, const char*, const I*, const I*, const I*, T*,
                     const I*, T*, const I*, T*, const I*, T*, const I*, I*,
                     size_t, size_t);
  lib.Get<F>((std::string(Prefix<T>()) + "gghrd_").c_str())(
      &comp, &comp, &n, &ilo, &ihi, a.data(), &ld, b.data(), &ld, q.data(), &ld,
      z.data(), &ld, &info, 1, 1);
  Result r;
  AddInt(r, "info", info);
  Add(r, "h", Kind::kOther, a);
  Add(r, "t", Kind::kOther, b);
  if (comp != 'N') {
    Add(r, "q", Kind::kOther, q);
    Add(r, "z", Kind::kOther, z);
  }
  return r;
}

template <typename T>
Result Hgeqz(const BlasLibrary& lib, std::vector<T> h, std::vector<T> t,
             std::vector<T> q, std::vector<T> z, I n, I ilo, I ihi, char job,
             char comp, bool compare_w, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> work(std::max<I>(lwork, 1)), alpha(ld), alphai(ld), beta(ld);
  I info = 0;
  const std::string name = std::string(Prefix<T>()) + "hgeqz_";
  if constexpr (kComplex<T>) {
    std::vector<R> rwork(ld);
    using F =
        void (*)(const char*, const char*, const char*, const I*, const I*,
                 const I*, T*, const I*, T*, const I*, T*, T*, T*, const I*, T*,
                 const I*, T*, const I*, R*, I*, size_t, size_t, size_t);
    lib.Get<F>(name.c_str())(&job, &comp, &comp, &n, &ilo, &ihi, h.data(), &ld,
                             t.data(), &ld, alpha.data(), beta.data(), q.data(),
                             &ld, z.data(), &ld, work.data(), &lwork,
                             rwork.data(), &info, 1, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const char*, const I*, const I*,
                 const I*, T*, const I*, T*, const I*, T*, T*, T*, T*, const I*,
                 T*, const I*, T*, const I*, I*, size_t, size_t, size_t);
    lib.Get<F>(name.c_str())(&job, &comp, &comp, &n, &ilo, &ihi, h.data(), &ld,
                             t.data(), &ld, alpha.data(), alphai.data(),
                             beta.data(), q.data(), &ld, z.data(), &ld,
                             work.data(), &lwork, &info, 1, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddPencil(r, alpha, alphai, beta, n, compare_w);
  if (job == 'S') {
    Add(r, "s", Kind::kOther, h);
    Add(r, "p", Kind::kOther, t);
  }
  if (comp != 'N') {
    Add(r, "q", Kind::kOther, q);
    Add(r, "z", Kind::kOther, z);
  }
  return r;
}

// xGGES (SORT = 'S' selects the eigenvalues with a negative real part) and
// xGGEV (JOBVSL / JOBVSR are JOBVL / JOBVR).
template <typename T>
Result Gges(const BlasLibrary& lib, bool ggev, std::vector<T> a,
            std::vector<T> b, I n, char jobl, char jobr, char sort,
            bool compare_w, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  std::vector<T> work(std::max<I>(lwork, 1)), alpha(ld), alphai(ld), beta(ld),
      vl(Size(n, n)), vr(Size(n, n));
  std::vector<I> bwork(ld);
  I sdim = 0, info = 0;
  const std::string p = Prefix<T>();
  if constexpr (kComplex<T>) {
    std::vector<R> rwork(std::max<I>(8 * n, 1));
    if (ggev) {
      using F = void (*)(const char*, const char*, const I*, T*, const I*, T*,
                         const I*, T*, T*, T*, const I*, T*, const I*, T*,
                         const I*, R*, I*, size_t, size_t);
      lib.Get<F>((p + "ggev_").c_str())(
          &jobl, &jobr, &n, a.data(), &ld, b.data(), &ld, alpha.data(),
          beta.data(), vl.data(), &ld, vr.data(), &ld, work.data(), &lwork,
          rwork.data(), &info, 1, 1);
    } else {
      using F = void (*)(const char*, const char*, const char*,
                         I (*)(const R*, const R*), const I*, T*, const I*, T*,
                         const I*, I*, T*, T*, T*, const I*, T*, const I*, T*,
                         const I*, R*, I*, I*, size_t, size_t, size_t);
      lib.Get<F>((p + "gges_").c_str())(
          &jobl, &jobr, &sort, &SelectNegative2<R>, &n, a.data(), &ld, b.data(),
          &ld, &sdim, alpha.data(), beta.data(), vl.data(), &ld, vr.data(), &ld,
          work.data(), &lwork, rwork.data(), bwork.data(), &info, 1, 1, 1);
    }
  } else {
    if (ggev) {
      using F = void (*)(const char*, const char*, const I*, T*, const I*, T*,
                         const I*, T*, T*, T*, T*, const I*, T*, const I*, T*,
                         const I*, I*, size_t, size_t);
      lib.Get<F>((p + "ggev_").c_str())(
          &jobl, &jobr, &n, a.data(), &ld, b.data(), &ld, alpha.data(),
          alphai.data(), beta.data(), vl.data(), &ld, vr.data(), &ld,
          work.data(), &lwork, &info, 1, 1);
    } else {
      using F =
          void (*)(const char*, const char*, const char*,
                   I (*)(const R*, const R*, const R*), const I*, T*, const I*,
                   T*, const I*, I*, T*, T*, T*, T*, const I*, T*, const I*, T*,
                   const I*, I*, I*, size_t, size_t, size_t);
      lib.Get<F>((p + "gges_").c_str())(
          &jobl, &jobr, &sort, &SelectNegative3<R>, &n, a.data(), &ld, b.data(),
          &ld, &sdim, alpha.data(), alphai.data(), beta.data(), vl.data(), &ld,
          vr.data(), &ld, work.data(), &lwork, bwork.data(), &info, 1, 1, 1);
    }
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddPencil(r, alpha, alphai, beta, n, compare_w);
  if (!ggev) {
    AddInt(r, "sdim", sdim);
    Add(r, "s", Kind::kOther, a);
    Add(r, "t", Kind::kOther, b);
  }
  if (jobl == 'V') Add(r, ggev ? "vl" : "vsl", Kind::kOther, vl);
  if (jobr == 'V') Add(r, ggev ? "vr" : "vsr", Kind::kOther, vr);
  return r;
}

// A generalized Schur form (S, P) = Q**H (A, B) Z (hwyBLAS's xGGES).
template <typename T>
struct GSchur {
  std::vector<T> s, p, q, z;
};

template <typename T>
GSchur<T> GeneralizedSchur(const BlasLibrary& hwy, const std::vector<T>& a,
                           const std::vector<T>& b, I n) {
  using R = Real<T>;
  GSchur<T> g{a, b, std::vector<T>(Size(n, n)), std::vector<T>(Size(n, n))};
  const I ld = std::max<I>(n, 1), lwork = 8 * ld + 16;
  std::vector<T> work(lwork), alpha(ld), alphai(ld), beta(ld);
  std::vector<I> bwork(ld);
  I sdim = 0, info = 0;
  const char v = 'V', sort = 'N';
  if constexpr (kComplex<T>) {
    std::vector<R> rwork(8 * ld);
    using F = void (*)(const char*, const char*, const char*,
                       I (*)(const R*, const R*), const I*, T*, const I*, T*,
                       const I*, I*, T*, T*, T*, const I*, T*, const I*, T*,
                       const I*, R*, I*, I*, size_t, size_t, size_t);
    hwy.Get<F>((std::string(Prefix<T>()) + "gges_").c_str())(
        &v, &v, &sort, &SelectNegative2<R>, &n, g.s.data(), &ld, g.p.data(),
        &ld, &sdim, alpha.data(), beta.data(), g.q.data(), &ld, g.z.data(), &ld,
        work.data(), &lwork, rwork.data(), bwork.data(), &info, 1, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const char*,
                       I (*)(const R*, const R*, const R*), const I*, T*,
                       const I*, T*, const I*, I*, T*, T*, T*, T*, const I*, T*,
                       const I*, T*, const I*, I*, I*, size_t, size_t, size_t);
    hwy.Get<F>((std::string(Prefix<T>()) + "gges_").c_str())(
        &v, &v, &sort, &SelectNegative3<R>, &n, g.s.data(), &ld, g.p.data(),
        &ld, &sdim, alpha.data(), alphai.data(), beta.data(), g.q.data(), &ld,
        g.z.data(), &ld, work.data(), &lwork, bwork.data(), &info, 1, 1, 1);
  }
  return g;
}

template <typename T>
Result Tgevc(const BlasLibrary& lib, const GSchur<T>& g, I n, char howmny) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  const std::vector<I> select = EveryThird(n);
  std::vector<T> vl = howmny == 'B' ? g.q : std::vector<T>(Size(n, n));
  std::vector<T> vr = howmny == 'B' ? g.z : std::vector<T>(Size(n, n));
  I m = 0, info = 0;
  const char side = 'B';
  if constexpr (kComplex<T>) {
    std::vector<T> work(2 * ld);
    std::vector<R> rwork(2 * ld);
    using F = void (*)(const char*, const char*, const I*, const I*, const T*,
                       const I*, const T*, const I*, T*, const I*, T*, const I*,
                       const I*, I*, T*, R*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "tgevc_").c_str())(
        &side, &howmny, select.data(), &n, g.s.data(), &ld, g.p.data(), &ld,
        vl.data(), &ld, vr.data(), &ld, &n, &m, work.data(), rwork.data(),
        &info, 1, 1);
  } else {
    std::vector<T> work(6 * ld);
    using F = void (*)(const char*, const char*, const I*, const I*, const T*,
                       const I*, const T*, const I*, T*, const I*, T*, const I*,
                       const I*, I*, T*, I*, size_t, size_t);
    lib.Get<F>((std::string(Prefix<T>()) + "tgevc_").c_str())(
        &side, &howmny, select.data(), &n, g.s.data(), &ld, g.p.data(), &ld,
        vl.data(), &ld, vr.data(), &ld, &n, &m, work.data(), &info, 1, 1);
  }
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  Add(r, "vl", Kind::kOther, vl);
  Add(r, "vr", Kind::kOther, vr);
  return r;
}

// xTGSEN with every third eigenvalue selected.
template <typename T>
Result Tgsen(const BlasLibrary& lib, GSchur<T> g, I n, I ijob, I lwork) {
  using R = Real<T>;
  const I ld = std::max<I>(n, 1);
  const std::vector<I> select = EveryThird(n);
  const I liwork = std::max<I>(n * n + 6, 1);
  std::vector<T> work(std::max<I>(lwork, 1)), alpha(ld), alphai(ld), beta(ld);
  std::vector<I> iwork(liwork);
  const I one = 1;
  I m = 0, info = 0;
  R pl = 0, pr = 0, dif[2] = {0, 0};
  if constexpr (kComplex<T>) {
    using F =
        void (*)(const I*, const I*, const I*, const I*, const I*, T*, const I*,
                 T*, const I*, T*, T*, T*, const I*, T*, const I*, I*, R*, R*,
                 R*, T*, const I*, I*, const I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "tgsen_").c_str())(
        &ijob, &one, &one, select.data(), &n, g.s.data(), &ld, g.p.data(), &ld,
        alpha.data(), beta.data(), g.q.data(), &ld, g.z.data(), &ld, &m, &pl,
        &pr, dif, work.data(), &lwork, iwork.data(), &liwork, &info);
  } else {
    using F =
        void (*)(const I*, const I*, const I*, const I*, const I*, T*, const I*,
                 T*, const I*, T*, T*, T*, T*, const I*, T*, const I*, I*, R*,
                 R*, R*, T*, const I*, I*, const I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "tgsen_").c_str())(
        &ijob, &one, &one, select.data(), &n, g.s.data(), &ld, g.p.data(), &ld,
        alpha.data(), alphai.data(), beta.data(), g.q.data(), &ld, g.z.data(),
        &ld, &m, &pl, &pr, dif, work.data(), &lwork, iwork.data(), &liwork,
        &info);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "m", m);
  AddPencil(r, alpha, alphai, beta, n, false);
  Add(r, "s", Kind::kOther, g.s);
  Add(r, "p", Kind::kOther, g.p);
  Add(r, "q", Kind::kOther, g.q);
  Add(r, "z", Kind::kOther, g.z);
  if (ijob == 1 || ijob >= 4) {
    Add(r, "pl", Kind::kOther, &pl, 1);
    Add(r, "pr", Kind::kOther, &pr, 1);
  }
  if (ijob >= 2) Add(r, "dif", Kind::kOther, dif, 2);
  return r;
}

template <typename T>
Result Tgexc(const BlasLibrary& lib, GSchur<T> g, I n, I ifst, I ilst) {
  const I ld = std::max<I>(n, 1);
  const I one = 1;
  I info = 0;
  if constexpr (kComplex<T>) {
    using F = void (*)(const I*, const I*, const I*, T*, const I*, T*, const I*,
                       T*, const I*, T*, const I*, const I*, I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "tgexc_").c_str())(
        &one, &one, &n, g.s.data(), &ld, g.p.data(), &ld, g.q.data(), &ld,
        g.z.data(), &ld, &ifst, &ilst, &info);
  } else {
    const I lwork = 4 * ld + 16;
    std::vector<T> work(lwork);
    using F = void (*)(const I*, const I*, const I*, T*, const I*, T*, const I*,
                       T*, const I*, T*, const I*, I*, I*, T*, const I*, I*);
    lib.Get<F>((std::string(Prefix<T>()) + "tgexc_").c_str())(
        &one, &one, &n, g.s.data(), &ld, g.p.data(), &ld, g.q.data(), &ld,
        g.z.data(), &ld, &ifst, &ilst, work.data(), &lwork, &info);
  }
  Result r;
  AddInt(r, "info", info);
  AddInt(r, "ifst", ifst);
  AddInt(r, "ilst", ilst);
  Add(r, "s", Kind::kOther, g.s);
  Add(r, "p", Kind::kOther, g.p);
  Add(r, "q", Kind::kOther, g.q);
  Add(r, "z", Kind::kOther, g.z);
  return r;
}

// xTGSYL with (A, D) = g's Schur form (order m) and (B, E) = h's (order n).
template <typename T>
Result Tgsyl(const BlasLibrary& lib, const GSchur<T>& g, I m,
             const GSchur<T>& h, I n, std::vector<T> c, std::vector<T> f,
             char trans, I ijob, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(m, 1), ldb = std::max<I>(n, 1);
  std::vector<T> work(std::max<I>(lwork, 1));
  std::vector<I> iwork(m + n + 6);
  R scale = 0, dif = 0;
  I info = 0;
  using F =
      void (*)(const char*, const I*, const I*, const I*, const T*, const I*,
               const T*, const I*, T*, const I*, const T*, const I*, const T*,
               const I*, T*, const I*, R*, R*, T*, const I*, I*, I*, size_t);
  lib.Get<F>((std::string(Prefix<T>()) + "tgsyl_").c_str())(
      &trans, &ijob, &m, &n, g.s.data(), &lda, h.s.data(), &ldb, c.data(), &lda,
      g.p.data(), &lda, h.p.data(), &ldb, f.data(), &lda, &scale, &dif,
      work.data(), &lwork, iwork.data(), &info, 1);
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  Add(r, "scale", Kind::kOther, &scale, 1);
  if (ijob >= 1) Add(r, "dif", Kind::kOther, &dif, 1);
  if (ijob <= 2) {
    Add(r, "c", Kind::kOther, c);
    Add(r, "f", Kind::kOther, f);
  }
  return r;
}

// xGGSVD3 (JOBU = JOBV = JOBQ = job) of the m x n A and the p x n B. The
// complex ALPHA and BETA are compared with the tolerance as sets (they are
// not sorted, and rounding can change their order), when `compare_values`
// (they are too ill-conditioned for graded pairs and pairs with ties).
template <typename T>
Result Ggsvd3(const BlasLibrary& lib, std::vector<T> a, std::vector<T> b, I m,
              I n, I p, char job, bool compare_values, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(m, 1), ldb = std::max<I>(p, 1),
          ldq = std::max<I>(n, 1);
  const char ju = job == 'N' ? 'N' : 'U', jv = job == 'N' ? 'N' : 'V',
             jq = job == 'N' ? 'N' : 'Q';
  std::vector<T> u(Size(m, m)), v(Size(p, p)), q(Size(n, n)),
      work(std::max<I>(lwork, 1));
  std::vector<R> alpha(ldq), beta(ldq);
  std::vector<I> iwork(ldq);
  I k = 0, l = 0, info = 0;
  const std::string name = std::string(Prefix<T>()) + "ggsvd3_";
  if constexpr (kComplex<T>) {
    std::vector<R> rwork(2 * ldq);
    using F = void (*)(const char*, const char*, const char*, const I*,
                       const I*, const I*, I*, I*, T*, const I*, T*, const I*,
                       R*, R*, T*, const I*, T*, const I*, T*, const I*, T*,
                       const I*, R*, I*, I*, size_t, size_t, size_t);
    lib.Get<F>(name.c_str())(
        &ju, &jv, &jq, &m, &n, &p, &k, &l, a.data(), &lda, b.data(), &ldb,
        alpha.data(), beta.data(), u.data(), &lda, v.data(), &ldb, q.data(),
        &ldq, work.data(), &lwork, rwork.data(), iwork.data(), &info, 1, 1, 1);
  } else {
    using F = void (*)(const char*, const char*, const char*, const I*,
                       const I*, const I*, I*, I*, T*, const I*, T*, const I*,
                       R*, R*, T*, const I*, T*, const I*, T*, const I*, T*,
                       const I*, I*, I*, size_t, size_t, size_t);
    lib.Get<F>(name.c_str())(&ju, &jv, &jq, &m, &n, &p, &k, &l, a.data(), &lda,
                             b.data(), &ldb, alpha.data(), beta.data(),
                             u.data(), &lda, v.data(), &ldb, q.data(), &ldq,
                             work.data(), &lwork, iwork.data(), &info, 1, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "k", k);
  AddInt(r, "l", l);
  const Kind values = compare_values ? Kind::kEigenvalues : Kind::kOther;
  Add(r, "alpha", values, alpha.data(), static_cast<size_t>(n));
  Add(r, "beta", values, beta.data(), static_cast<size_t>(n));
  Add(r, "iwork", Kind::kInteger, iwork.data(), static_cast<size_t>(n));
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  if (job != 'N') {
    Add(r, "u", Kind::kOther, u);
    Add(r, "v", Kind::kOther, v);
    Add(r, "q", Kind::kOther, q);
  }
  return r;
}

// xGGSVP3 with the given tolerances.
template <typename T>
Result Ggsvp3(const BlasLibrary& lib, std::vector<T> a, std::vector<T> b, I m,
              I n, I p, Real<T> tola, Real<T> tolb, I lwork) {
  using R = Real<T>;
  const I lda = std::max<I>(m, 1), ldb = std::max<I>(p, 1),
          ldq = std::max<I>(n, 1);
  const char ju = 'U', jv = 'V', jq = 'Q';
  std::vector<T> u(Size(m, m)), v(Size(p, p)), q(Size(n, n)), tau(ldq),
      work(std::max<I>(lwork, 1));
  std::vector<I> iwork(ldq);
  I k = 0, l = 0, info = 0;
  const std::string name = std::string(Prefix<T>()) + "ggsvp3_";
  if constexpr (kComplex<T>) {
    std::vector<R> rwork(2 * ldq);
    using F =
        void (*)(const char*, const char*, const char*, const I*, const I*,
                 const I*, T*, const I*, T*, const I*, const R*, const R*, I*,
                 I*, T*, const I*, T*, const I*, T*, const I*, I*, R*, T*, T*,
                 const I*, I*, size_t, size_t, size_t);
    lib.Get<F>(name.c_str())(
        &ju, &jv, &jq, &m, &p, &n, a.data(), &lda, b.data(), &ldb, &tola, &tolb,
        &k, &l, u.data(), &lda, v.data(), &ldb, q.data(), &ldq, iwork.data(),
        rwork.data(), tau.data(), work.data(), &lwork, &info, 1, 1, 1);
  } else {
    using F =
        void (*)(const char*, const char*, const char*, const I*, const I*,
                 const I*, T*, const I*, T*, const I*, const R*, const R*, I*,
                 I*, T*, const I*, T*, const I*, T*, const I*, I*, T*, T*,
                 const I*, I*, size_t, size_t, size_t);
    lib.Get<F>(name.c_str())(&ju, &jv, &jq, &m, &p, &n, a.data(), &lda,
                             b.data(), &ldb, &tola, &tolb, &k, &l, u.data(),
                             &lda, v.data(), &ldb, q.data(), &ldq, iwork.data(),
                             tau.data(), work.data(), &lwork, &info, 1, 1, 1);
  }
  Result r;
  r.lwork = std::real(work[0]);
  AddInt(r, "info", info);
  AddInt(r, "k", k);
  AddInt(r, "l", l);
  Add(r, "a", Kind::kOther, a);
  Add(r, "b", Kind::kOther, b);
  Add(r, "u", Kind::kOther, u);
  Add(r, "v", Kind::kOther, v);
  Add(r, "q", Kind::kOther, q);
  return r;
}

// The generalized eigenproblem and GSVD routines for pairs of order n:
// random, graded, with small integers (ties), and random A with a singular
// B (a zero row and column). The complex eigenvalues of the graded and
// singular pairs are not compared (too ill-conditioned or infinite).
template <typename T>
void RunGeneralized(const BlasLibrary& hwy, const BlasLibrary& ref, I n) {
  // The other drivers get the same matrices with and without these cases.
  const std::mt19937_64 saved = rng;
  char what[96];
  for (const int kind : {0, 2, 4, 5}) {
    const std::vector<T> a = TestMatrix<T>(kind == 5 ? 0 : kind, n, n);
    std::vector<T> b = TestMatrix<T>(kind == 5 ? 0 : kind, n, n);
    if (kind == 5 && n > 0) {
      for (I i = 0; i < n; ++i) {
        b[n / 2 + static_cast<size_t>(i) * n] = T(0);
        b[i + static_cast<size_t>(n / 2) * n] = T(0);
      }
    }
    const bool compare_w = kind == 0;
    for (const char jobl : {'V', 'N'}) {
      for (const char jobr : {'V', 'N'}) {
        snprintf(what, sizeof(what), "n=%d kind=%d jobvl=%c jobvr=%c", int(n),
                 kind, jobl, jobr);
        Case<T>(hwy, ref, "ggev", what, n, [&](const BlasLibrary& lib, I lw) {
          return Gges(lib, true, a, b, n, jobl, jobr, 'N', compare_w, lw);
        });
      }
    }
    for (const char sort : {'N', 'S'}) {
      for (const char job : {'V', 'N'}) {
        snprintf(what, sizeof(what), "n=%d kind=%d jobvs=%c sort=%c", int(n),
                 kind, job, sort);
        Case<T>(hwy, ref, "gges", what, n, [&](const BlasLibrary& lib, I lw) {
          return Gges(lib, false, a, b, n, job, job, sort, compare_w, lw);
        });
      }
    }
    const std::vector<T> v = TestMatrix<T>(0, n, 3);
    for (const char job : {'N', 'P', 'S', 'B'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d job=%c", int(n), kind, job);
      Case<T>(hwy, ref, "ggbal", what, n, [&](const BlasLibrary& lib, I) {
        return Ggbal(lib, a, b, n, job, v, 3);
      });
    }
    // xGGHRD on (A, triu(B)) (also with ILO, IHI of a block), then xHGEQZ on
    // hwyBLAS's Hessenberg form.
    std::vector<T> bt = b;
    for (I j = 0; j < n; ++j) {
      for (I i = j + 1; i < n; ++i) bt[i + static_cast<size_t>(j) * n] = T(0);
    }
    const std::vector<T> q0 = TestMatrix<T>(0, n, n);
    for (const char comp : {'N', 'I', 'V'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d comp=%c", int(n), kind, comp);
      Case<T>(hwy, ref, "gghrd", what, n, [&](const BlasLibrary& lib, I) {
        return Gghrd(lib, a, bt, n, 1, n, comp, q0);
      });
    }
    const Result hess = Gghrd(hwy, a, bt, n, 1, n, 'I', q0);
    std::vector<T> h(Size(n, n)), t(Size(n, n)), q(Size(n, n)), z(Size(n, n));
    const auto from = [](std::complex<double> x) {
      if constexpr (kComplex<T>) {
        return static_cast<T>(x);
      } else {
        return static_cast<T>(x.real());
      }
    };
    for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i) {
      h[i] = from(hess.outputs[1].values[i]);
      t[i] = from(hess.outputs[2].values[i]);
      q[i] = from(hess.outputs[3].values[i]);
      z[i] = from(hess.outputs[4].values[i]);
    }
    for (const char job : {'E', 'S'}) {
      for (const char comp : {'N', 'I', 'V'}) {
        if (job == 'E' && comp != 'N') continue;
        snprintf(what, sizeof(what), "n=%d kind=%d job=%c comp=%c", int(n),
                 kind, job, comp);
        Case<T>(hwy, ref, "hgeqz", what, n, [&](const BlasLibrary& lib, I lw) {
          return Hgeqz(lib, h, t, q, z, n, 1, n, job, comp, compare_w, lw);
        });
      }
    }
    // The routines on hwyBLAS's generalized Schur form.
    const GSchur<T> g = GeneralizedSchur(hwy, a, b, n);
    for (const char howmny : {'A', 'S', 'B'}) {
      snprintf(what, sizeof(what), "n=%d kind=%d howmny=%c", int(n), kind,
               howmny);
      Case<T>(hwy, ref, "tgevc", what, n, [&](const BlasLibrary& lib, I) {
        return Tgevc(lib, g, n, howmny);
      });
    }
    for (I ijob = 0; ijob <= 5; ++ijob) {
      snprintf(what, sizeof(what), "n=%d kind=%d ijob=%d", int(n), kind,
               int(ijob));
      Case<T>(hwy, ref, "tgsen", what, n, [&](const BlasLibrary& lib, I lw) {
        return Tgsen(lib, g, n, ijob, lw);
      });
    }
    if (n > 1) {
      for (const auto& move : {std::make_pair(I{1}, n), std::make_pair(n, I{1}),
                               std::make_pair(I{2}, n / 2 + 1)}) {
        snprintf(what, sizeof(what), "n=%d kind=%d ifst=%d ilst=%d", int(n),
                 kind, int(move.first), int(move.second));
        Case<T>(hwy, ref, "tgexc", what, n, [&](const BlasLibrary& lib, I) {
          return Tgexc(lib, g, n, move.first, move.second);
        });
      }
    }
    // xTGSYL with g and the Schur form of another (random) pair of order
    // n2.
    if (n > 0 && kind != 5) {
      const I n2 = std::max<I>(1, n / 2 + 1);
      const GSchur<T> g2 = GeneralizedSchur(hwy, TestMatrix<T>(0, n2, n2),
                                            TestMatrix<T>(0, n2, n2), n2);
      const std::vector<T> c = TestMatrix<T>(0, n, n2),
                           f = TestMatrix<T>(0, n, n2);
      for (const char trans : {'N', kComplex<T> ? 'C' : 'T'}) {
        for (I ijob = 0; ijob <= 4; ++ijob) {
          if (trans != 'N' && ijob > 0) continue;
          snprintf(what, sizeof(what), "m=%d n=%d kind=%d trans=%c ijob=%d",
                   int(n), int(n2), kind, trans, int(ijob));
          Case<T>(hwy, ref, "tgsyl", what, n,
                  [&](const BlasLibrary& lib, I lw) {
                    return Tgsyl(lib, g, n, g2, n2, c, f, trans, ijob, lw);
                  });
        }
      }
    }
    // The GSVD of A (n x n, and its first rows) and B (n x n or fewer rows).
    const I shapes[][2] = {{n, n}, {n / 2, n}, {n, n / 2 + 1}};
    for (const auto& shape : shapes) {
      const I m = shape[0], p = shape[1];
      std::vector<T> am(Size(m, n)), bp(Size(p, n));
      for (I j = 0; j < n; ++j) {
        for (I i = 0; i < m; ++i) {
          am[i + static_cast<size_t>(j) * m] =
              a[i + static_cast<size_t>(j) * n];
        }
        for (I i = 0; i < p; ++i) {
          bp[i + static_cast<size_t>(j) * p] =
              b[i + static_cast<size_t>(j) * n];
        }
      }
      for (const char job : {'U', 'N'}) {
        snprintf(what, sizeof(what), "m=%d n=%d p=%d kind=%d job=%c", int(m),
                 int(n), int(p), kind, job);
        Case<T>(hwy, ref, "ggsvd3", what, n, [&](const BlasLibrary& lib, I lw) {
          return Ggsvd3(lib, am, bp, m, n, p, job, kind == 0 || kind == 5, lw);
        });
      }
      using R = Real<T>;
      const R tol = std::max<I>(n, 1) * std::numeric_limits<R>::epsilon();
      snprintf(what, sizeof(what), "m=%d n=%d p=%d kind=%d", int(m), int(n),
               int(p), kind);
      Case<T>(hwy, ref, "ggsvp3", what, n, [&](const BlasLibrary& lib, I lw) {
        return Ggsvp3(lib, am, bp, m, n, p, tol, tol, lw);
      });
    }
  }
  rng = saved;
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
      // xGETRF only on matrices with clear pivots (see the top).
      if (s % 5 != 4) {
        Case<T>(hwy, ref, "getrf", what, mn, [&](const BlasLibrary& lib, I lw) {
          return Getrf(lib, a, m, n, lw);
        });
      }
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
    RunEigen<T>(hwy, ref, n);
    RunLinear<T>(hwy, ref, n);
    RunSymmetric<T>(hwy, ref, n);
    RunSyevd<T>(hwy, ref, n);
    RunSyevr<T>(hwy, ref, n);
    RunGels<T>(hwy, ref, n);
    RunBand<T>(hwy, ref, n);
    RunGeneralized<T>(hwy, ref, n);
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
