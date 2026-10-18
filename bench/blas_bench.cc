// BLAS benchmark: compares single-threaded GFLOP/s of BLAS libraries that are
// loaded at runtime.
//
// Usage:
//   blas_bench <routine> [options] [name=path/to/blas.dll ...]
//
// Routines: gemm, gemv, symm (hemm for complex), syrk, syr2k (her2k for
// complex), gemmt (C := A * B for one triangle of C), trmm, trsm,
// symv (hemv for complex), trmv, trsv, ger (geru for complex), syr (her),
// syr2 (her2), banded and packed: gbmv, sbmv (hbmv), spmv (hpmv), tbmv,
// tbsv, tpmv, tpsv, spr (hpr), spr2 (hpr2), and level 1: axpy, scal, dot
// (dotu for complex), nrm2, asum, iamax, rot. LAPACK: getrf, potrf, geqrf,
// geqp3, syev (heev for complex), gesvd, gesdd, gees (their times include
// copying the input matrix, which they overwrite; for syev, gesvd, gesdd and
// gees, the outputs of the libraries differ, as eigenvectors and singular
// vectors are unique only up to their sign, and the order of the
// eigenvalues in the Schur form depends on rounding). A square
// size n means m = n = k; for syrk, C is m x m and A is m x k.
//
// Options:
//   --types=sdcz        precisions to run (default: d)
//   --trans=NN          gemm: transa/transb (default: NN); others: one letter
//   --side=L --uplo=L --diag=N   options of the structured routines
//   --sizes=64,128,...  square sizes (m = n = k)
//   --shape=MxNxK       a single, possibly non-square shape (repeatable;
//                       gemv uses M x N)
//   --min-time=0.5      seconds per library and shape
//   --rounds=3          libraries take turns this many times
//   --pin=N             pin the benchmark thread to logical CPU N
//   --band=K            banded routines: K off-diagonals (gbmv: kl = ku = K;
//                       default: n - 1, i.e. a full band)
//
// hwyBLAS (the library built with the benchmark) is always included first.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "blas_loader.h"

namespace hwyblas_bench {
namespace {

struct Shape {
  int m, n, k;
};

struct Options {
  std::string routine;
  std::string types = "d";
  std::string trans;  // empty: routine default
  char side = 'L', uplo = 'L', diag = 'N';
  std::vector<Shape> shapes;
  double min_time = 0.5;  // per library and shape, over all rounds
  int rounds = 3;
  int pin_cpu = -1;  // -1: no pinning
  int band = -1;     // -1: n - 1
};

using Libraries = std::vector<std::unique_ptr<BlasLibrary>>;

template <typename T>
constexpr bool kIsComplex =
    !std::is_same_v<T, float> && !std::is_same_v<T, double>;

// Real type of T: float for float and std::complex<float>, etc.
template <typename T>
struct RealType {
  using type = T;
};
template <typename T>
struct RealType<std::complex<T>> {
  using type = T;
};

// BLAS symbol name for the precision of T, e.g. Symbol<double>("gemm").
template <typename T>
std::string Symbol(const char* routine) {
  const char prefix = std::is_same_v<T, float>                 ? 's'
                      : std::is_same_v<T, double>              ? 'd'
                      : std::is_same_v<T, std::complex<float>> ? 'c'
                                                               : 'z';
  return prefix + std::string(routine) + "_";
}

template <typename T>
std::vector<T> RandomVector(size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> v(n);
  for (T& x : v) {
    if constexpr (kIsComplex<T>) {
      using R = typename T::value_type;
      x = T(static_cast<R>(dist(rng)), static_cast<R>(dist(rng)));
    } else {
      x = static_cast<T>(dist(rng));
    }
  }
  return v;
}

using Clock = std::chrono::steady_clock;

// Returns the best time in seconds of one call, from repeated samples
// running at least `min_time` seconds in total (and at least 2 samples).
// Fast calls are batched so that each sample is well above the clock
// resolution.
template <class Func>
double BestTime(const Func& func, double min_time) {
  const auto time = [&func](int calls) {
    const auto t0 = Clock::now();
    for (int i = 0; i < calls; ++i) func();
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  func();  // warm-up: page faults, library initialization, caches
  int calls = 1;
  while (calls < (1 << 24) && time(calls) < 1e-4) calls *= 2;

  double best = 1e30, total = 0;
  for (int samples = 0; samples < 2 || total < min_time; ++samples) {
    const double t = time(calls);
    best = std::min(best, t / calls);
    total += t;
  }
  return best;
}

template <typename T>
double MaxRelDiff(const std::vector<T>& x, const std::vector<T>& ref) {
  double max_diff = 0, max_ref = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::abs(x[i] - ref[i])));
    max_ref = std::max(max_ref, static_cast<double>(std::abs(ref[i])));
  }
  return max_ref > 0 ? max_diff / max_ref : max_diff;
}

// One benchmark case: `call(fn, out)` invokes the routine at address `fn`
// writing to `out`, which starts as a copy of `initial_out`.
template <typename T>
struct Case {
  std::string label;
  double flops;
  std::vector<T> initial_out;
  std::function<void(void* fn, T* out)> call;
};

// Times one case for every library and prints a table row: GFLOP/s per
// library and the largest relative difference between their results.
template <typename T>
void RunCase(const Options& opt, const Libraries& libs,
             const std::string& symbol, const Case<T>& c) {
  printf("%-20s", c.label.c_str());

  // Cross-library correctness check: one clean call each.
  std::vector<T> reference;
  double max_diff = 0;
  for (const auto& lib : libs) {
    void* fn = lib->Find(symbol.c_str());
    if (!fn) continue;
    std::vector<T> out = c.initial_out;
    c.call(fn, out.data());
    if (reference.empty()) {
      reference = out;
    } else {
      max_diff = std::max(max_diff, MaxRelDiff(out, reference));
    }
  }

  // Timing: libraries take turns over several rounds and each keeps its
  // best time, so slow drifts (clock boost, background load) affect all of
  // them alike. Calls keep accumulating into `out`; fine for timing.
  std::vector<double> best(libs.size(), 1e30);
  std::vector<T> out = c.initial_out;
  for (int round = 0; round < opt.rounds; ++round) {
    for (size_t i = 0; i < libs.size(); ++i) {
      void* fn = libs[i]->Find(symbol.c_str());
      if (!fn) continue;
      const double t =
          BestTime([&] { c.call(fn, out.data()); }, opt.min_time / opt.rounds);
      best[i] = std::min(best[i], t);
    }
  }
  for (double t : best) {
    if (t >= 1e30) {
      printf("%12s", "-");
    } else {
      printf("%12.2f", c.flops / t * 1e-9);
    }
  }
  printf("%14.2e\n", max_diff);
  fflush(stdout);
}

template <typename T>
void PrintHeader(const Libraries& libs, const std::string& title,
                 const char* shape_label) {
  printf("\n%s  (GFLOP/s, single-threaded)\n", title.c_str());
  printf("%-20s", shape_label);
  for (const auto& lib : libs) printf("%12s", lib->name().c_str());
  printf("%14s\n", "max rel diff");
}

// Fortran-style signatures; complex arguments are interleaved pairs.
template <typename T>
using GemmFn = void (*)(const char*, const char*, const int*, const int*,
                        const int*, const T*, const T*, const int*, const T*,
                        const int*, const T*, T*, const int*);
template <typename T>
using GemvFn = void (*)(const char*, const int*, const int*, const T*, const T*,
                        const int*, const T*, const int*, const T*, T*,
                        const int*);

template <typename T>
void BenchGemm(const Options& opt, const Libraries& libs) {
  const std::string trans = opt.trans.empty() ? "NN" : opt.trans;
  const char ta = trans[0], tb = trans[1];
  const std::string symbol = Symbol<T>("gemm");
  PrintHeader<T>(libs, symbol + "  trans=" + trans, "m x n x k");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int lda = std::max(1, ta == 'N' ? s.m : s.k);
    const int ldb = std::max(1, tb == 'N' ? s.k : s.n);
    const int ldc = std::max(1, s.m);
    const auto a = std::make_shared<std::vector<T>>(RandomVector<T>(
        static_cast<size_t>(lda) * (ta == 'N' ? s.k : s.m), rng));
    const auto b = std::make_shared<std::vector<T>>(RandomVector<T>(
        static_cast<size_t>(ldb) * (tb == 'N' ? s.n : s.k), rng));

    Case<T> c;
    c.label = std::to_string(s.m) + " x " + std::to_string(s.n) + " x " +
              std::to_string(s.k);
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * s.m * s.n * s.k;
    c.initial_out = RandomVector<T>(static_cast<size_t>(ldc) * s.n, rng);
    c.call = [=](void* fn, T* out) {
      const T alpha = T(1), beta = T(0.5);
      reinterpret_cast<GemmFn<T>>(fn)(&ta, &tb, &s.m, &s.n, &s.k, &alpha,
                                      a->data(), &lda, b->data(), &ldb, &beta,
                                      out, &ldc);
    };
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
void BenchGemv(const Options& opt, const Libraries& libs) {
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  const std::string symbol = Symbol<T>("gemv");
  PrintHeader<T>(libs, symbol + "  trans=" + trans, "m x n");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int lda = std::max(1, s.m), inc = 1;
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * s.n, rng));
    const auto x = std::make_shared<std::vector<T>>(
        RandomVector<T>(trans == 'N' ? s.n : s.m, rng));

    Case<T> c;
    c.label = std::to_string(s.m) + " x " + std::to_string(s.n);
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * s.m * s.n;
    c.initial_out = RandomVector<T>(trans == 'N' ? s.m : s.n, rng);
    c.call = [=](void* fn, T* out) {
      const T alpha = T(1), beta = T(0.5);
      reinterpret_cast<GemvFn<T>>(fn)(&trans, &s.m, &s.n, &alpha, a->data(),
                                      &lda, x->data(), &inc, &beta, out, &inc);
    };
    RunCase(opt, libs, symbol, c);
  }
}

// Random n x n matrix with a dominant diagonal (|diag| ~ n), so that
// triangular solves stay well conditioned. Both triangles are filled.
template <typename T>
std::shared_ptr<std::vector<T>> DominantMatrix(int n, std::mt19937& rng) {
  auto a = std::make_shared<std::vector<T>>(
      RandomVector<T>(static_cast<size_t>(n) * n, rng));
  for (int i = 0; i < n; ++i)
    (*a)[i + static_cast<size_t>(i) * n] +=
        T(static_cast<typename RealType<T>::type>(n));
  return a;
}

std::string Label(int m, int n) {
  return std::to_string(m) + " x " + std::to_string(n);
}

template <typename T>
void BenchSymm(const Options& opt, const Libraries& libs) {
  const std::string symbol = Symbol<T>(kIsComplex<T> ? "hemm" : "symm");
  PrintHeader<T>(libs, symbol + "  side=" + opt.side + " uplo=" + opt.uplo,
                 "m x n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int ka = (opt.side == 'L') ? s.m : s.n;
    const auto a = DominantMatrix<T>(ka, rng);
    const auto b = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(s.m) * s.n, rng));
    Case<T> c;
    c.label = Label(s.m, s.n);
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * s.m * s.n * ka;
    c.initial_out = RandomVector<T>(static_cast<size_t>(s.m) * s.n, rng);
    const char side = opt.side, uplo = opt.uplo;
    c.call = [=](void* fn, T* out) {
      using Fn = void (*)(const char*, const char*, const int*, const int*,
                          const T*, const T*, const int*, const T*, const int*,
                          const T*, T*, const int*);
      const T alpha = T(1), beta = T(0.5);
      reinterpret_cast<Fn>(fn)(&side, &uplo, &s.m, &s.n, &alpha, a->data(), &ka,
                               b->data(), &s.m, &beta, out, &s.m);
    };
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
void BenchSyrk(const Options& opt, const Libraries& libs) {
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  const std::string symbol = Symbol<T>("syrk");
  PrintHeader<T>(libs, symbol + "  uplo=" + opt.uplo + " trans=" + trans,
                 "n x k");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    // C is n x n; A is n x k (trans = N) or k x n.
    const int n = s.m, k = s.k;
    const int lda = (trans == 'N') ? n : k;
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(n) * k, rng));
    Case<T> c;
    c.label = Label(n, k);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * n * (n + 1.0) * k;
    c.initial_out = RandomVector<T>(static_cast<size_t>(n) * n, rng);
    const char uplo = opt.uplo;
    c.call = [=](void* fn, T* out) {
      using Fn =
          void (*)(const char*, const char*, const int*, const int*, const T*,
                   const T*, const int*, const T*, T*, const int*);
      const T alpha = T(1), beta = T(0.5);
      reinterpret_cast<Fn>(fn)(&uplo, &trans, &n, &k, &alpha, a->data(), &lda,
                               &beta, out, &n);
    };
    RunCase(opt, libs, symbol, c);
  }
}

// SYR2K (HER2K for complex types) and GEMMT, which update the uplo triangle
// of the n x n matrix C. For gemmt, op(A) is n x k and op(B) k x n; for
// syr2k, A and B are n x k (trans = N) or k x n.
template <typename T>
void BenchTriangleUpdate(const Options& opt, const Libraries& libs) {
  const bool gemmt = (opt.routine == "gemmt");
  const char ta = opt.trans.empty() ? 'N' : opt.trans[0];
  const char tb = opt.trans.size() > 1 ? opt.trans[1] : 'N';
  const std::string symbol =
      Symbol<T>(gemmt ? "gemmt" : (kIsComplex<T> ? "her2k" : "syr2k"));
  const std::string trans = gemmt ? std::string{ta, tb} : std::string{ta};
  PrintHeader<T>(libs, symbol + "  uplo=" + opt.uplo + " trans=" + trans,
                 "n x k");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.m, k = s.k;
    const int lda = (ta == 'N') ? n : k;
    const int ldb = gemmt ? ((tb == 'N') ? k : n) : lda;
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(n) * k, rng));
    const auto b = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(n) * k, rng));
    Case<T> c;
    c.label = Label(n, k);
    // gemmt: half of a GEMM; syr2k: two halves.
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * n * (n + 1.0) * k * (gemmt ? 1 : 2);
    c.initial_out = RandomVector<T>(static_cast<size_t>(n) * n, rng);
    const char uplo = opt.uplo;
    if (gemmt) {
      c.call = [=](void* fn, T* out) {
        using Fn = void (*)(const char*, const char*, const char*, const int*,
                            const int*, const T*, const T*, const int*,
                            const T*, const int*, const T*, T*, const int*);
        const T alpha = T(1), beta = T(0.5);
        reinterpret_cast<Fn>(fn)(&uplo, &ta, &tb, &n, &k, &alpha, a->data(),
                                 &lda, b->data(), &ldb, &beta, out, &n);
      };
    } else {
      c.call = [=](void* fn, T* out) {
        // For her2k, beta is real: its pointer reads the real part.
        using Fn = void (*)(const char*, const char*, const int*, const int*,
                            const T*, const T*, const int*, const T*,
                            const int*, const T*, T*, const int*);
        const T alpha = T(1), beta = T(0.5);
        reinterpret_cast<Fn>(fn)(&uplo, &ta, &n, &k, &alpha, a->data(), &lda,
                                 b->data(), &ldb, &beta, out, &n);
      };
    }
    RunCase(opt, libs, symbol, c);
  }
}

// TRMM and TRSM. Both overwrite B, so every call starts from a fresh copy of
// the input (the copy is cheap compared to the O(m^2 n) work).
template <typename T>
void BenchTriangular3(const Options& opt, const Libraries& libs,
                      const char* routine) {
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  const std::string symbol = Symbol<T>(routine);
  PrintHeader<T>(libs,
                 symbol + "  side=" + opt.side + " uplo=" + opt.uplo +
                     " trans=" + trans + " diag=" + opt.diag,
                 "m x n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int ka = (opt.side == 'L') ? s.m : s.n;
    const auto a = DominantMatrix<T>(ka, rng);
    const auto b0 = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(s.m) * s.n, rng));
    Case<T> c;
    c.label = Label(s.m, s.n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * s.m * s.n * ka;
    c.initial_out = *b0;
    const char side = opt.side, uplo = opt.uplo, diag = opt.diag;
    c.call = [=](void* fn, T* out) {
      using Fn = void (*)(const char*, const char*, const char*, const char*,
                          const int*, const int*, const T*, const T*,
                          const int*, T*, const int*);
      const T alpha = T(1);
      std::copy(b0->begin(), b0->end(), out);
      reinterpret_cast<Fn>(fn)(&side, &uplo, &trans, &diag, &s.m, &s.n, &alpha,
                               a->data(), &ka, out, &s.m);
    };
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
void BenchSymv(const Options& opt, const Libraries& libs) {
  const std::string symbol = Symbol<T>(kIsComplex<T> ? "hemv" : "symv");
  PrintHeader<T>(libs, symbol + "  uplo=" + opt.uplo, "n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.m, inc = 1;
    const auto a = DominantMatrix<T>(n, rng);
    const auto x = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));
    Case<T> c;
    c.label = std::to_string(n);
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * n * n;
    c.initial_out = RandomVector<T>(n, rng);
    const char uplo = opt.uplo;
    c.call = [=](void* fn, T* out) {
      using Fn =
          void (*)(const char*, const int*, const T*, const T*, const int*,
                   const T*, const int*, const T*, T*, const int*);
      const T alpha = T(1), beta = T(0.5);
      reinterpret_cast<Fn>(fn)(&uplo, &n, &alpha, a->data(), &n, x->data(),
                               &inc, &beta, out, &inc);
    };
    RunCase(opt, libs, symbol, c);
  }
}

// TRMV and TRSV.
template <typename T>
void BenchTrmv(const Options& opt, const Libraries& libs, const char* name) {
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  const std::string symbol = Symbol<T>(name);
  PrintHeader<T>(
      libs,
      symbol + "  uplo=" + opt.uplo + " trans=" + trans + " diag=" + opt.diag,
      "n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.m, inc = 1;
    const auto a = DominantMatrix<T>(n, rng);
    const auto x0 = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));
    Case<T> c;
    c.label = std::to_string(n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * n * n;
    c.initial_out = *x0;
    const char uplo = opt.uplo, diag = opt.diag;
    c.call = [=](void* fn, T* out) {
      using Fn = void (*)(const char*, const char*, const char*, const int*,
                          const T*, const int*, T*, const int*);
      // Restart from x0 so that repeated calls stay bounded.
      std::copy(x0->begin(), x0->end(), out);
      reinterpret_cast<Fn>(fn)(&uplo, &trans, &diag, &n, a->data(), &n, out,
                               &inc);
    };
    RunCase(opt, libs, symbol, c);
  }
}

// Rank-1 and rank-2 updates of an m x n (ger) or n x n (syr, syr2) matrix.
template <typename T>
void BenchRankUpdate(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  const std::string& routine = opt.routine;
  std::string symbol;
  if (routine == "ger") {
    symbol = Symbol<T>(kIsComplex<T> ? "geru" : "ger");
  } else if (routine == "syr") {
    symbol = Symbol<T>(kIsComplex<T> ? "her" : "syr");
  } else {
    symbol = Symbol<T>(kIsComplex<T> ? "her2" : "syr2");
  }
  PrintHeader<T>(libs,
                 routine == "ger" ? symbol : symbol + "  uplo=" + opt.uplo,
                 routine == "ger" ? "m x n" : "n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int m = s.m, n = (routine == "ger") ? s.n : s.m, inc = 1;
    const auto x = std::make_shared<std::vector<T>>(RandomVector<T>(m, rng));
    const auto y = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));
    Case<T> c;
    c.label = (routine == "ger") ? std::to_string(m) + "x" + std::to_string(n)
                                 : std::to_string(n);
    // Multiply-adds per element: 1 (ger), 1/2 (syr), 1 (syr2).
    const double elements = (routine == "ger")   ? static_cast<double>(m) * n
                            : (routine == "syr") ? 0.5 * n * n
                                                 : 1.0 * n * n;
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * elements;
    c.initial_out = RandomVector<T>(static_cast<size_t>(m) * n, rng);
    const char uplo = opt.uplo;
    // Small alpha keeps the repeatedly updated matrix bounded.
    if (routine == "ger") {
      c.call = [=](void* fn, T* out) {
        const T alpha = T(R(1e-3));
        reinterpret_cast<void (*)(const int*, const int*, const T*, const T*,
                                  const int*, const T*, const int*, T*,
                                  const int*)>(fn)(
            &m, &n, &alpha, x->data(), &inc, y->data(), &inc, out, &m);
      };
    } else if (routine == "syr") {
      c.call = [=](void* fn, T* out) {
        const R alpha = R(1e-3);
        reinterpret_cast<void (*)(const char*, const int*, const R*, const T*,
                                  const int*, T*, const int*)>(fn)(
            &uplo, &n, &alpha, x->data(), &inc, out, &n);
      };
    } else {
      c.call = [=](void* fn, T* out) {
        const T alpha = T(R(1e-3));
        reinterpret_cast<void (*)(const char*, const int*, const T*, const T*,
                                  const int*, const T*, const int*, T*,
                                  const int*)>(fn)(
            &uplo, &n, &alpha, x->data(), &inc, y->data(), &inc, out, &n);
      };
    }
    RunCase(opt, libs, symbol, c);
  }
}

// Banded and packed routines on n x n matrices (gbmv: m x n). The matrices
// have a dominant diagonal, so that the triangular solves stay bounded.
template <typename T>
void BenchBandPacked(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  const std::string& routine = opt.routine;
  const bool packed = (routine[1] == 'p');  // spmv, tpmv, tpsv, spr, spr2
  const bool triangular = (routine[0] == 't');
  const bool update = (routine == "spr" || routine == "spr2");
  // Hermitian names for complex types: sbmv -> hbmv, spmv -> hpmv, ...
  std::string name = routine;
  if (kIsComplex<T> && routine[0] == 's') name[0] = 'h';
  const std::string symbol = Symbol<T>(name.c_str());
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  std::string title = symbol;
  if (routine != "gbmv") title += std::string("  uplo=") + opt.uplo;
  if (routine == "gbmv" || triangular) title += std::string(" trans=") + trans;
  if (!packed) {
    title += " band=" +
             (opt.band < 0 ? std::string("n-1") : std::to_string(opt.band));
  }
  PrintHeader<T>(libs, title, routine == "gbmv" ? "m x n" : "n");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int m = (routine == "gbmv") ? s.m : s.n, n = s.n, inc = 1;
    const int k = (opt.band < 0) ? std::max(m, n) - 1 : opt.band;
    const char uplo = opt.uplo, diag = opt.diag;
    const bool lower = (uplo == 'L' || uplo == 'l');
    // Stored matrix: kl, ku for band storage.
    const int kl = (routine == "gbmv") ? k : (lower ? k : 0);
    const int ku = (routine == "gbmv") ? k : (lower ? 0 : k);
    const int lda = kl + ku + 1;
    const size_t size = packed ? static_cast<size_t>(n) * (n + 1) / 2
                               : static_cast<size_t>(lda) * n;
    auto a = std::make_shared<std::vector<T>>(RandomVector<T>(size, rng));
    // Scale the off-diagonals and make the diagonal dominant. Counts the
    // elements that are part of the matrix.
    double elements = 0;
    const R scale = R(1) / R(std::min(k, std::max(m, n)) + 1);
    for (int j = 0; j < n; ++j) {
      int begin, end;
      size_t offset;  // of element (begin, j)
      if (packed) {
        begin = lower ? j : 0;
        end = lower ? n : j + 1;
        offset = lower ? static_cast<size_t>(j) * (2 * n - j + 1) / 2
                       : static_cast<size_t>(j) * (j + 1) / 2;
      } else {
        begin = std::max(0, j - ku);
        end = std::max(begin, std::min(m, j + kl + 1));
        offset = static_cast<size_t>(j) * lda + ku + begin - j;
      }
      elements += end - begin;
      for (int i = begin; i < end; ++i) {
        T& v = (*a)[offset + (i - begin)];
        v = (i == j) ? T(R(2)) + v * R(0.5) : v * scale;
      }
    }
    const auto x = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));
    const auto y = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));

    Case<T> c;
    c.label = (routine == "gbmv") ? std::to_string(m) + "x" + std::to_string(n)
                                  : std::to_string(n);
    // Multiply-adds per stored element: 1, 2 for symmetric matrix-vector
    // products (each off-diagonal element stands for two) and spr2.
    const bool twice =
        (routine == "sbmv" || routine == "spmv" || routine == "spr2");
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * elements * (twice ? 2 : 1);
    if (update) {
      c.initial_out = *a;
    } else {
      c.initial_out =
          RandomVector<T>(routine == "gbmv" && trans == 'N' ? m : n, rng);
    }
    const auto x0 = std::make_shared<std::vector<T>>(c.initial_out);
    const T alpha = T(R(1e-3)), beta = T(R(0.5));
    if (routine == "gbmv") {
      const int len_x = (trans == 'N') ? n : m;
      const auto xg =
          std::make_shared<std::vector<T>>(RandomVector<T>(len_x, rng));
      c.call = [=](void* fn, T* out) {
        reinterpret_cast<void (*)(
            const char*, const int*, const int*, const int*, const int*,
            const T*, const T*, const int*, const T*, const int*, const T*, T*,
            const int*)>(fn)(&trans, &m, &n, &kl, &ku, &alpha, a->data(), &lda,
                             xg->data(), &inc, &beta, out, &inc);
      };
    } else if (routine == "sbmv") {
      c.call = [=](void* fn, T* out) {
        reinterpret_cast<void (*)(const char*, const int*, const int*, const T*,
                                  const T*, const int*, const T*, const int*,
                                  const T*, T*, const int*)>(fn)(
            &uplo, &n, &k, &alpha, a->data(), &lda, x->data(), &inc, &beta, out,
            &inc);
      };
    } else if (routine == "spmv") {
      c.call = [=](void* fn, T* out) {
        reinterpret_cast<void (*)(const char*, const int*, const T*, const T*,
                                  const T*, const int*, const T*, T*,
                                  const int*)>(fn)(
            &uplo, &n, &alpha, a->data(), x->data(), &inc, &beta, out, &inc);
      };
    } else if (routine == "tbmv" || routine == "tbsv") {
      c.call = [=](void* fn, T* out) {
        // Restart from x0 so that repeated calls stay bounded.
        std::copy(x0->begin(), x0->end(), out);
        reinterpret_cast<void (*)(const char*, const char*, const char*,
                                  const int*, const int*, const T*, const int*,
                                  T*, const int*)>(fn)(
            &uplo, &trans, &diag, &n, &k, a->data(), &lda, out, &inc);
      };
    } else if (routine == "tpmv" || routine == "tpsv") {
      c.call = [=](void* fn, T* out) {
        std::copy(x0->begin(), x0->end(), out);
        reinterpret_cast<void (*)(const char*, const char*, const char*,
                                  const int*, const T*, T*, const int*)>(fn)(
            &uplo, &trans, &diag, &n, a->data(), out, &inc);
      };
    } else if (routine == "spr") {
      c.call = [=](void* fn, T* out) {
        const R real_alpha = R(1e-3);
        reinterpret_cast<void (*)(const char*, const int*, const R*, const T*,
                                  const int*, T*)>(fn)(&uplo, &n, &real_alpha,
                                                       x->data(), &inc, out);
      };
    } else {  // spr2
      c.call = [=](void* fn, T* out) {
        reinterpret_cast<void (*)(const char*, const int*, const T*, const T*,
                                  const int*, const T*, const int*, T*)>(fn)(
            &uplo, &n, &alpha, x->data(), &inc, y->data(), &inc, out);
      };
    }
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
using GetrfFn = void (*)(const int*, const int*, T*, const int*, int*, int*);

// LU factorization of an m x n matrix (LAPACK). Each call factors a fresh
// copy of the same random matrix.
template <typename T>
void BenchGetrf(const Options& opt, const Libraries& libs) {
  const std::string symbol = Symbol<T>("getrf");
  PrintHeader<T>(libs, symbol, "m x n");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int lda = std::max(1, s.m);
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * s.n, rng));
    const auto ipiv =
        std::make_shared<std::vector<int>>(std::max(1, std::min(s.m, s.n)));

    Case<T> c;
    c.label = std::to_string(s.m) + " x " + std::to_string(s.n);
    // m n^2 - n^3 / 3 multiply-adds for m >= n (m and n swapped otherwise),
    // about 4 times as many flops for complex numbers.
    const double lo = std::min(s.m, s.n), hi = std::max(s.m, s.n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * (hi * lo * lo - lo * lo * lo / 3);
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      int info = 0;
      reinterpret_cast<GetrfFn<T>>(fn)(&s.m, &s.n, out, &lda, ipiv->data(),
                                       &info);
    };
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
using PotrfFn = void (*)(const char*, const int*, T*, const int*, int*);

// Cholesky factorization of an n x n symmetric (Hermitian) positive definite
// matrix (LAPACK), the triangle --uplo (default: L). Each call factors a
// fresh copy of the same matrix.
template <typename T>
void BenchPotrf(const Options& opt, const Libraries& libs) {
  const char uplo = opt.uplo;
  const std::string symbol = Symbol<T>("potrf");
  PrintHeader<T>(libs, symbol + "  uplo=" + uplo, "n");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.n, lda = std::max(1, n);
    // Random, with a dominant real diagonal: the Hermitian matrix of either
    // triangle is positive definite.
    auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * n, rng));
    for (int j = 0; j < n; ++j) {
      (*a)[j + static_cast<size_t>(j) * lda] =
          T(static_cast<typename RealType<T>::type>(2 * n));
    }

    Case<T> c;
    c.label = std::to_string(n);
    // n^3 / 3 flops, about 4 times as many for complex numbers.
    c.flops =
        (kIsComplex<T> ? 4.0 : 1.0) * (static_cast<double>(n) * n * n / 3);
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      int info = 0;
      reinterpret_cast<PotrfFn<T>>(fn)(&uplo, &n, out, &lda, &info);
    };
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
using GeqrfFn = void (*)(const int*, const int*, T*, const int*, T*, T*,
                         const int*, int*);

// QR factorization of an m x n matrix (LAPACK), with a workspace that is
// large enough for any block size up to 256. Each call factors a fresh copy
// of the same random matrix.
template <typename T>
void BenchGeqrf(const Options& opt, const Libraries& libs) {
  const std::string symbol = Symbol<T>("geqrf");
  PrintHeader<T>(libs, symbol, "m x n");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int lda = std::max(1, s.m);
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * s.n, rng));
    const int lwork = 256 * std::max(1, s.n);
    const auto tau =
        std::make_shared<std::vector<T>>(std::max(1, std::min(s.m, s.n)));
    const auto work = std::make_shared<std::vector<T>>(lwork);

    Case<T> c;
    c.label = std::to_string(s.m) + " x " + std::to_string(s.n);
    // 2 m n^2 - 2 n^3 / 3 flops for m >= n (m and n swapped otherwise),
    // about 4 times as many for complex numbers.
    const double lo = std::min(s.m, s.n), hi = std::max(s.m, s.n);
    c.flops =
        (kIsComplex<T> ? 4.0 : 1.0) * (2 * hi * lo * lo - 2 * lo * lo * lo / 3);
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      int info = 0;
      reinterpret_cast<GeqrfFn<T>>(fn)(&s.m, &s.n, out, &lda, tau->data(),
                                       work->data(), &lwork, &info);
    };
    RunCase(opt, libs, symbol, c);
  }
}

// xGEQP3: the real routines without RWORK, the complex ones with it.
template <typename T>
using Geqp3RealFn = void (*)(const int*, const int*, T*, const int*, int*, T*,
                             T*, const int*, int*);
template <typename T>
using Geqp3ComplexFn = void (*)(const int*, const int*, T*, const int*, int*,
                                T*, T*, const int*, typename RealType<T>::type*,
                                int*);

// QR factorization with column pivoting of an m x n matrix (LAPACK), with
// all columns free; the flops are counted as for geqrf. Each call factors a
// fresh copy of the same random matrix.
template <typename T>
void BenchGeqp3(const Options& opt, const Libraries& libs) {
  const std::string symbol = Symbol<T>("geqp3");
  PrintHeader<T>(libs, symbol, "m x n");

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int lda = std::max(1, s.m);
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * s.n, rng));
    const int lwork = 256 * (std::max(1, s.n) + 1) + 2 * s.n;
    const auto tau =
        std::make_shared<std::vector<T>>(std::max(1, std::min(s.m, s.n)));
    const auto jpvt = std::make_shared<std::vector<int>>(std::max(1, s.n));
    const auto work = std::make_shared<std::vector<T>>(lwork);
    const auto rwork =
        std::make_shared<std::vector<typename RealType<T>::type>>(
            2 * std::max(1, s.n));

    Case<T> c;
    c.label = std::to_string(s.m) + " x " + std::to_string(s.n);
    const double lo = std::min(s.m, s.n), hi = std::max(s.m, s.n);
    c.flops =
        (kIsComplex<T> ? 4.0 : 1.0) * (2 * hi * lo * lo - 2 * lo * lo * lo / 3);
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      std::fill(jpvt->begin(), jpvt->end(), 0);
      int info = 0;
      if constexpr (kIsComplex<T>) {
        reinterpret_cast<Geqp3ComplexFn<T>>(fn)(
            &s.m, &s.n, out, &lda, jpvt->data(), tau->data(), work->data(),
            &lwork, rwork->data(), &info);
      } else {
        reinterpret_cast<Geqp3RealFn<T>>(fn)(&s.m, &s.n, out, &lda,
                                             jpvt->data(), tau->data(),
                                             work->data(), &lwork, &info);
      }
    };
    RunCase(opt, libs, symbol, c);
  }
}

// xSYEV (real) and xHEEV (complex, with RWORK).
template <typename T>
using SyevFn = void (*)(const char*, const char*, const int*, T*, const int*,
                        typename RealType<T>::type*, T*, const int*, int*);
template <typename T>
using HeevFn = void (*)(const char*, const char*, const int*, T*, const int*,
                        typename RealType<T>::type*, T*, const int*,
                        typename RealType<T>::type*, int*);

// Eigenvalues and eigenvectors of an n x n symmetric (Hermitian) matrix
// (LAPACK, jobz = 'V', the triangle --uplo). Reports the time as GFLOP/s
// of the nominal 9 n^3 flops (4 times as many for complex numbers); each
// call works on a fresh copy of the same random matrix.
template <typename T>
void BenchSyev(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  const char jobz = 'V', uplo = opt.uplo;
  const std::string symbol = Symbol<T>(kIsComplex<T> ? "heev" : "syev");
  PrintHeader<T>(libs, symbol + "  jobz=V uplo=" + uplo + "  (9 n^3 flops)",
                 "n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.n, lda = std::max(1, n);
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * n, rng));
    for (int j = 0; j < n; ++j) {
      // Real diagonal; the other triangle is not referenced.
      (*a)[j + static_cast<size_t>(j) * lda] =
          T(static_cast<R>(std::real((*a)[j + static_cast<size_t>(j) * lda])));
    }
    const int lwork = 256 * std::max(1, n) + 3 * n;
    const auto w = std::make_shared<std::vector<R>>(std::max(1, n));
    const auto work = std::make_shared<std::vector<T>>(lwork);
    const auto rwork = std::make_shared<std::vector<R>>(std::max(1, 3 * n));

    Case<T> c;
    c.label = std::to_string(n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * 9.0 * n * n * n;
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      int info = 0;
      if constexpr (kIsComplex<T>) {
        reinterpret_cast<HeevFn<T>>(fn)(&jobz, &uplo, &n, out, &lda, w->data(),
                                        work->data(), &lwork, rwork->data(),
                                        &info);
      } else {
        reinterpret_cast<SyevFn<T>>(fn)(&jobz, &uplo, &n, out, &lda, w->data(),
                                        work->data(), &lwork, &info);
      }
    };
    RunCase(opt, libs, symbol, c);
  }
}

// xGESVD: the real routines without RWORK, the complex ones with it.
template <typename T>
using GesvdRealFn = void (*)(const char*, const char*, const int*, const int*,
                             T*, const int*, typename RealType<T>::type*, T*,
                             const int*, T*, const int*, T*, const int*, int*);
template <typename T>
using GesvdComplexFn = void (*)(const char*, const char*, const int*,
                                const int*, T*, const int*,
                                typename RealType<T>::type*, T*, const int*, T*,
                                const int*, T*, const int*,
                                typename RealType<T>::type*, int*);

// Singular value decomposition of an m x n matrix (LAPACK, jobu = jobvt =
// 'S': the singular values and min(m, n) left and right singular vectors).
// Reports the time as GFLOP/s of the nominal min(14 m n^2 + 8 n^3, 6 m n^2
// + 20 n^3) flops for m >= n (Golub and Van Loan's counts for the
// Golub-Kahan-Reinsch SVD and the R-SVD with U1 and V; m and n swapped for
// m < n; 4 times as many for complex numbers); each call works on a fresh
// copy of the same random matrix.
template <typename T>
void BenchGesvd(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  const char jobu = 'S', jobvt = 'S';
  const std::string symbol = Symbol<T>("gesvd");
  PrintHeader<T>(libs, symbol + "  jobu=S jobvt=S", "m x n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int m = s.m, n = s.n, lda = std::max(1, m);
    const int mn = std::max(1, std::min(m, n));
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * n, rng));
    const int lwork = 256 * (m + n) + 5 * mn + std::max(m, n) + 2 * mn * mn;
    const auto sv = std::make_shared<std::vector<R>>(mn);
    const auto u =
        std::make_shared<std::vector<T>>(static_cast<size_t>(lda) * mn);
    const auto vt =
        std::make_shared<std::vector<T>>(static_cast<size_t>(mn) * n);
    const auto work = std::make_shared<std::vector<T>>(lwork);
    const auto rwork = std::make_shared<std::vector<R>>(5 * mn);

    Case<T> c;
    c.label = std::to_string(m) + " x " + std::to_string(n);
    const double hi = std::max(m, n), lo = std::min(m, n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) *
              std::min(14 * hi * lo * lo + 8 * lo * lo * lo,
                       6 * hi * lo * lo + 20 * lo * lo * lo);
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      const int ldvt = mn;
      int info = 0;
      if constexpr (kIsComplex<T>) {
        reinterpret_cast<GesvdComplexFn<T>>(fn)(
            &jobu, &jobvt, &m, &n, out, &lda, sv->data(), u->data(), &lda,
            vt->data(), &ldvt, work->data(), &lwork, rwork->data(), &info);
      } else {
        reinterpret_cast<GesvdRealFn<T>>(fn)(
            &jobu, &jobvt, &m, &n, out, &lda, sv->data(), u->data(), &lda,
            vt->data(), &ldvt, work->data(), &lwork, &info);
      }
    };
    RunCase(opt, libs, symbol, c);
  }
}

// xGESDD: the real routines without RWORK, the complex ones with it.
template <typename T>
using GesddRealFn = void (*)(const char*, const int*, const int*, T*,
                             const int*, typename RealType<T>::type*, T*,
                             const int*, T*, const int*, T*, const int*, int*,
                             int*);
template <typename T>
using GesddComplexFn = void (*)(const char*, const int*, const int*, T*,
                                const int*, typename RealType<T>::type*, T*,
                                const int*, T*, const int*, T*, const int*,
                                typename RealType<T>::type*, int*, int*);

// Singular value decomposition by divide and conquer of an m x n matrix
// (LAPACK, jobz = 'S'), reported as GFLOP/s of the nominal flops of gesvd
// (see BenchGesvd), so that the two compare by their times.
template <typename T>
void BenchGesdd(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  const char jobz = 'S';
  const std::string symbol = Symbol<T>("gesdd");
  PrintHeader<T>(libs, symbol + "  jobz=S", "m x n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int m = s.m, n = s.n, lda = std::max(1, m);
    const int mn = std::max(1, std::min(m, n)), mx = std::max(m, n);
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * n, rng));
    // More than the optimal workspace of all paths.
    const int lwork = 4 * mn * mn + 7 * mn + 256 * (m + n) + mx * mn;
    const auto sv = std::make_shared<std::vector<R>>(mn);
    const auto u =
        std::make_shared<std::vector<T>>(static_cast<size_t>(lda) * mn);
    const auto vt =
        std::make_shared<std::vector<T>>(static_cast<size_t>(mn) * n);
    const auto work = std::make_shared<std::vector<T>>(lwork);
    const auto rwork = std::make_shared<std::vector<R>>(
        std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn));
    const auto iwork = std::make_shared<std::vector<int>>(8 * mn);

    Case<T> c;
    c.label = std::to_string(m) + " x " + std::to_string(n);
    const double hi = mx, lo = std::min(m, n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) *
              std::min(14 * hi * lo * lo + 8 * lo * lo * lo,
                       6 * hi * lo * lo + 20 * lo * lo * lo);
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      const int ldvt = mn;
      int info = 0;
      if constexpr (kIsComplex<T>) {
        reinterpret_cast<GesddComplexFn<T>>(fn)(
            &jobz, &m, &n, out, &lda, sv->data(), u->data(), &lda, vt->data(),
            &ldvt, work->data(), &lwork, rwork->data(), iwork->data(), &info);
      } else {
        reinterpret_cast<GesddRealFn<T>>(fn)(
            &jobz, &m, &n, out, &lda, sv->data(), u->data(), &lda, vt->data(),
            &ldvt, work->data(), &lwork, iwork->data(), &info);
      }
    };
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
using GeesRealFn = void (*)(const char*, const char*, void*, const int*, T*,
                            const int*, int*, T*, T*, T*, const int*, T*,
                            const int*, int*, int*);
template <typename T>
using GeesComplexFn = void (*)(const char*, const char*, void*, const int*, T*,
                               const int*, int*, T*, T*, const int*, T*,
                               const int*, typename RealType<T>::type*, int*,
                               int*);

// Schur factorization A = Z * T * Z**H of an n x n matrix with the Schur
// vectors (LAPACK, jobvs = 'V', sort = 'N'), reported as GFLOP/s of the
// nominal 25 n**3 flops (4 times as many real flops for complex T).
template <typename T>
void BenchGees(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  const char jobvs = 'V', sort = 'N';
  const std::string symbol = Symbol<T>("gees");
  PrintHeader<T>(libs, symbol + "  jobvs=V sort=N", "n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.n, lda = std::max(1, n);
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(lda) * n, rng));
    // More than the optimal workspace.
    const int lwork = 64 * n + 4096;
    const auto w = std::make_shared<std::vector<T>>(2 * lda);
    const auto vs =
        std::make_shared<std::vector<T>>(static_cast<size_t>(lda) * n);
    const auto work = std::make_shared<std::vector<T>>(lwork);
    const auto rwork = std::make_shared<std::vector<R>>(lda);
    const auto bwork = std::make_shared<std::vector<int>>(lda);

    Case<T> c;
    c.label = std::to_string(n);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * 25.0 * n * n * n;
    c.initial_out = *a;
    c.call = [=](void* fn, T* out) {
      std::copy(a->begin(), a->end(), out);
      int sdim = 0, info = 0;
      if constexpr (kIsComplex<T>) {
        reinterpret_cast<GeesComplexFn<T>>(fn)(
            &jobvs, &sort, nullptr, &n, out, &lda, &sdim, w->data(), vs->data(),
            &lda, work->data(), &lwork, rwork->data(), bwork->data(), &info);
      } else {
        reinterpret_cast<GeesRealFn<T>>(fn)(
            &jobvs, &sort, nullptr, &n, out, &lda, &sdim, w->data(),
            w->data() + lda, vs->data(), &lda, work->data(), &lwork,
            bwork->data(), &info);
      }
    };
    RunCase(opt, libs, symbol, c);
  }
}

// OMATCOPY: B := alpha * op(A) for an m x n column-major A. Reports
// G elements per second in the GFLOP/s columns.
template <typename T>
void BenchOmatcopy(const Options& opt, const Libraries& libs) {
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  const std::string symbol = Symbol<T>("omatcopy");
  PrintHeader<T>(libs,
                 symbol + "  trans=" + trans + "  (G elements/s, not GFLOP/s)",
                 "m x n");
  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int m = s.m, n = s.n;
    const int ldb = (trans == 'N' || trans == 'R') ? m : n;
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(static_cast<size_t>(m) * n, rng));
    Case<T> c;
    c.label = std::to_string(m) + "x" + std::to_string(n);
    c.flops = static_cast<double>(m) * n;
    c.initial_out = std::vector<T>(static_cast<size_t>(m) * n);
    c.call = [=](void* fn, T* out) {
      const char order = 'C';
      const T alpha = T(2);
      reinterpret_cast<void (*)(const char*, const char*, const int*,
                                const int*, const T*, const T*, const int*, T*,
                                const int*)>(fn)(&order, &trans, &m, &n, &alpha,
                                                 a->data(), &m, out, &ldb);
    };
    RunCase(opt, libs, symbol, c);
  }
}

// Level 1 routines on vectors of length n (from the m of each shape). Scalar
// results are written to out[0] so that they take part in the cross-library
// comparison; this also checks that complex results are returned with the
// same calling convention as the other libraries.
template <typename T>
void BenchLevel1(const Options& opt, const Libraries& libs) {
  using R = typename RealType<T>::type;
  struct Complex {
    R re, im;
  };
  const std::string& routine = opt.routine;
  // dot means dotu for complex types; nrm2/asum/iamax have special names.
  std::string symbol;
  if (routine == "nrm2" || routine == "asum") {
    symbol = std::is_same_v<T, std::complex<float>> ? "sc" + routine + "_"
             : std::is_same_v<T, std::complex<double>>
                 ? "dz" + routine + "_"
                 : Symbol<T>(routine.c_str());
  } else if (routine == "iamax") {
    symbol = "i" + Symbol<T>("amax");
  } else if (routine == "rot" && kIsComplex<T>) {
    // Complex vectors with real c and s: csrot/zdrot (crot/zrot in LAPACK
    // have a complex s).
    symbol = std::is_same_v<T, std::complex<float>> ? "csrot_" : "zdrot_";
  } else {
    symbol = Symbol<T>((routine == "dot" && kIsComplex<T>) ? "dotu"
                                                           : routine.c_str());
  }
  PrintHeader<T>(libs, symbol, "n");

  // Flops per element (real; complex counts the real operations).
  double flops_per_element = 2;  // axpy, dot, nrm2
  if (routine == "axpby") flops_per_element = 3;
  if (routine == "scal" || routine == "asum" || routine == "iamax") {
    flops_per_element = 1;
  } else if (routine == "rot") {
    flops_per_element = 6;
  }
  if (kIsComplex<T>) {
    flops_per_element *=
        (routine == "axpy" || routine == "axpby" || routine == "dot") ? 4 : 2;
  }

  std::mt19937 rng(42);
  for (const Shape& s : opt.shapes) {
    const int n = s.m, inc = 1;
    const auto x = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));
    Case<T> c;
    c.label = std::to_string(n);
    c.flops = flops_per_element * n;
    c.initial_out = RandomVector<T>(n, rng);
    if (routine == "axpby") {
      c.call = [=](void* fn, T* out) {
        // beta close to 1 keeps the values bounded over repeated calls.
        const T alpha = T(0.5), beta = T(0.75);
        reinterpret_cast<void (*)(const int*, const T*, const T*, const int*,
                                  const T*, T*, const int*)>(fn)(
            &n, &alpha, x->data(), &inc, &beta, out, &inc);
      };
    } else if (routine == "axpy") {
      c.call = [=](void* fn, T* out) {
        const T alpha = T(0.5);
        reinterpret_cast<void (*)(const int*, const T*, const T*, const int*,
                                  T*, const int*)>(fn)(&n, &alpha, x->data(),
                                                       &inc, out, &inc);
      };
    } else if (routine == "scal") {
      c.call = [=](void* fn, T* out) {
        // Close to 1, so that repeated calls keep the values bounded.
        const T alpha = T(static_cast<R>(0.9999999));
        reinterpret_cast<void (*)(const int*, const T*, T*, const int*)>(fn)(
            &n, &alpha, out, &inc);
      };
    } else if (routine == "rot") {
      c.call = [=](void* fn, T* out) {
        // y is rotated in place too; start from the same y in every call.
        const R cs = R(0.6), sn = R(0.8);
        std::vector<T> y = *x;
        reinterpret_cast<void (*)(const int*, T*, const int*, T*, const int*,
                                  const R*, const R*)>(fn)(
            &n, out, &inc, y.data(), &inc, &cs, &sn);
      };
    } else if (routine == "dot") {
      const auto y = std::make_shared<std::vector<T>>(RandomVector<T>(n, rng));
      c.call = [=](void* fn, T* out) {
        if constexpr (kIsComplex<T>) {
          const Complex r = reinterpret_cast<Complex (*)(
              const int*, const T*, const int*, const T*, const int*)>(fn)(
              &n, x->data(), &inc, y->data(), &inc);
          out[0] = T(r.re, r.im);
        } else {
          out[0] = reinterpret_cast<T (*)(const int*, const T*, const int*,
                                          const T*, const int*)>(fn)(
              &n, x->data(), &inc, y->data(), &inc);
        }
      };
    } else if (routine == "nrm2" || routine == "asum") {
      c.call = [=](void* fn, T* out) {
        out[0] = T(reinterpret_cast<R (*)(const int*, const T*, const int*)>(
            fn)(&n, x->data(), &inc));
      };
    } else if (routine == "iamax") {
      c.call = [=](void* fn, T* out) {
        out[0] = T(static_cast<R>(
            reinterpret_cast<int (*)(const int*, const T*, const int*)>(fn)(
                &n, x->data(), &inc)));
      };
    } else {
      fprintf(stderr, "Unknown routine: %s\n", routine.c_str());
      return;
    }
    RunCase(opt, libs, symbol, c);
  }
}

template <typename T>
bool RunRoutine(const Options& opt, const Libraries& libs) {
  if (opt.routine == "gemm") {
    BenchGemm<T>(opt, libs);
  } else if (opt.routine == "gemv") {
    BenchGemv<T>(opt, libs);
  } else if (opt.routine == "symm") {
    BenchSymm<T>(opt, libs);
  } else if (opt.routine == "syrk") {
    BenchSyrk<T>(opt, libs);
  } else if (opt.routine == "syr2k" || opt.routine == "gemmt") {
    BenchTriangleUpdate<T>(opt, libs);
  } else if (opt.routine == "trmm" || opt.routine == "trsm") {
    BenchTriangular3<T>(opt, libs, opt.routine.c_str());
  } else if (opt.routine == "symv") {
    BenchSymv<T>(opt, libs);
  } else if (opt.routine == "trmv" || opt.routine == "trsv") {
    BenchTrmv<T>(opt, libs, opt.routine.c_str());
  } else if (opt.routine == "gbmv" || opt.routine == "sbmv" ||
             opt.routine == "spmv" || opt.routine == "tbmv" ||
             opt.routine == "tbsv" || opt.routine == "tpmv" ||
             opt.routine == "tpsv" || opt.routine == "spr" ||
             opt.routine == "spr2") {
    BenchBandPacked<T>(opt, libs);
  } else if (opt.routine == "ger" || opt.routine == "syr" ||
             opt.routine == "syr2") {
    BenchRankUpdate<T>(opt, libs);
  } else if (opt.routine == "omatcopy") {
    BenchOmatcopy<T>(opt, libs);
  } else if (opt.routine == "getrf") {
    BenchGetrf<T>(opt, libs);
  } else if (opt.routine == "potrf") {
    BenchPotrf<T>(opt, libs);
  } else if (opt.routine == "geqrf") {
    BenchGeqrf<T>(opt, libs);
  } else if (opt.routine == "geqp3") {
    BenchGeqp3<T>(opt, libs);
  } else if (opt.routine == "syev") {
    BenchSyev<T>(opt, libs);
  } else if (opt.routine == "gesvd") {
    BenchGesvd<T>(opt, libs);
  } else if (opt.routine == "gesdd") {
    BenchGesdd<T>(opt, libs);
  } else if (opt.routine == "gees") {
    BenchGees<T>(opt, libs);
  } else if (opt.routine == "axpy" || opt.routine == "axpby" ||
             opt.routine == "scal" || opt.routine == "dot" ||
             opt.routine == "nrm2" || opt.routine == "asum" ||
             opt.routine == "iamax" || opt.routine == "rot") {
    BenchLevel1<T>(opt, libs);
  } else {
    fprintf(stderr, "Unknown routine: %s\n", opt.routine.c_str());
    return false;
  }
  return true;
}

std::vector<int> ParseInts(const char* s, char sep) {
  std::vector<int> v;
  while (*s) {
    v.push_back(atoi(s));
    const char* next = strchr(s, sep);
    if (!next) break;
    s = next + 1;
  }
  return v;
}

int Main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '-') {
    fprintf(stderr,
            "Usage: %s <routine> [--types=sdcz] [--trans=..] "
            "[--sizes=..] [--shape=MxNxK] [name=library ...]\n",
            argv[0]);
    return 1;
  }
  Options opt;
  opt.routine = argv[1];
  std::vector<std::pair<std::string, std::string>> lib_specs;
  lib_specs.emplace_back("hwyBLAS", HWYBLAS_LIBRARY);

  for (int i = 2; i < argc; ++i) {
    const char* arg = argv[i];
    if (strncmp(arg, "--types=", 8) == 0) {
      opt.types = arg + 8;
    } else if (strncmp(arg, "--trans=", 8) == 0) {
      opt.trans = arg + 8;
    } else if (strncmp(arg, "--side=", 7) == 0) {
      opt.side = arg[7];
    } else if (strncmp(arg, "--uplo=", 7) == 0) {
      opt.uplo = arg[7];
    } else if (strncmp(arg, "--diag=", 7) == 0) {
      opt.diag = arg[7];
    } else if (strncmp(arg, "--sizes=", 8) == 0) {
      for (int n : ParseInts(arg + 8, ',')) opt.shapes.push_back({n, n, n});
    } else if (strncmp(arg, "--shape=", 8) == 0) {
      std::vector<int> d = ParseInts(arg + 8, 'x');
      if (d.size() < 2 || d.size() > 3) {
        fprintf(stderr, "Bad shape: %s\n", arg);
        return 1;
      }
      if (d.size() == 2) d.push_back(1);
      opt.shapes.push_back({d[0], d[1], d[2]});
    } else if (strncmp(arg, "--min-time=", 11) == 0) {
      opt.min_time = atof(arg + 11);
    } else if (strncmp(arg, "--rounds=", 9) == 0) {
      opt.rounds = std::max(1, atoi(arg + 9));
    } else if (strncmp(arg, "--pin=", 6) == 0) {
      opt.pin_cpu = atoi(arg + 6);
    } else if (strncmp(arg, "--band=", 7) == 0) {
      opt.band = atoi(arg + 7);
    } else if (const char* eq = strchr(arg, '=')) {
      lib_specs.emplace_back(std::string(arg, eq), eq + 1);
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg);
      return 1;
    }
  }
  if (opt.shapes.empty()) {
    for (int n : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
      opt.shapes.push_back({n, n, n});
    }
  }

#if defined(_WIN32)
  // Reduce noise: outrank background work, optionally stay on one core.
  if (opt.pin_cpu >= 0) {
    SetThreadAffinityMask(GetCurrentThread(), 1ull << opt.pin_cpu);
  }
  SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

  Libraries libs;
  for (const auto& spec : lib_specs) {
    auto lib = std::make_unique<BlasLibrary>(spec.first, spec.second);
    if (lib->ok()) libs.push_back(std::move(lib));
  }
  if (libs.empty()) return 1;
  if (auto target = libs[0]->Get<const char* (*)()>("hwyblas_target_name")) {
    printf("hwyBLAS target: %s\n", target());
  }

  for (char t : opt.types) {
    bool ok = true;
    switch (t) {
      case 's':
        ok = RunRoutine<float>(opt, libs);
        break;
      case 'd':
        ok = RunRoutine<double>(opt, libs);
        break;
      case 'c':
        ok = RunRoutine<std::complex<float>>(opt, libs);
        break;
      case 'z':
        ok = RunRoutine<std::complex<double>>(opt, libs);
        break;
      default:
        fprintf(stderr, "Unknown type %c\n", t);
        return 1;
    }
    if (!ok) return 1;
  }
  return 0;
}

}  // namespace
}  // namespace hwyblas_bench

int main(int argc, char** argv) { return hwyblas_bench::Main(argc, argv); }
