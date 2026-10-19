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
// flang evaluates some complex divisions differently (COMPLEX / REAL as a
// complex division, where the port divides the two parts), so results
// differ in the last bits. Their eigenvalues and singular values are compared
// with a tolerance instead (as sets for the nonsymmetric eigenvalues, whose
// order can change), and only the number of bitwise identical results is shown
// for the other outputs.
//
// Routines with a documented deviation that changes rounding (the leaf
// kernels of xGETRF2 and xPOTRF2, see src/lapack/factor_kernels.h) compute
// different last bits also in real precision: their factors are compared
// with the tolerance, and their pivots and INFO exactly.
//
// The comparison catches changes that keep the results valid, which the
// residual tests cannot: a different shift strategy, deflation criterion or
// pivot choice. It is not part of the build or of ctest.
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
