// BLAS benchmark: compares single-threaded GFLOP/s of BLAS libraries that are
// loaded at runtime.
//
// Usage:
//   blas_bench <routine> [options] [name=path/to/blas.dll ...]
//
// Routines: gemm, gemv, symm (hemm for complex), syrk, trmm, trsm,
// symv (hemv for complex), trmv. A square size n means m = n = k; for
// syrk, C is m x m and A is m x k.
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
//
// hwyBLAS from the benchmark's own directory is always included first.
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

// Returns the best time in seconds of repeated calls, running at least
// `min_time` seconds in total (and at least 2 repetitions).
template <class Func>
double BestTime(const Func& func, double min_time) {
  func();  // warm-up: page faults, library initialization, caches
  double best = 1e30, total = 0;
  for (int reps = 0; reps < 2 || total < min_time; ++reps) {
    const auto t0 = Clock::now();
    func();
    const double t = std::chrono::duration<double>(Clock::now() - t0).count();
    best = std::min(best, t);
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
    const auto a = std::make_shared<std::vector<T>>(
        RandomVector<T>(size_t(lda) * (ta == 'N' ? s.k : s.m), rng));
    const auto b = std::make_shared<std::vector<T>>(
        RandomVector<T>(size_t(ldb) * (tb == 'N' ? s.n : s.k), rng));

    Case<T> c;
    c.label = std::to_string(s.m) + " x " + std::to_string(s.n) + " x " +
              std::to_string(s.k);
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * s.m * s.n * s.k;
    c.initial_out = RandomVector<T>(size_t(ldc) * s.n, rng);
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
        RandomVector<T>(size_t(lda) * s.n, rng));
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
  auto a =
      std::make_shared<std::vector<T>>(RandomVector<T>(size_t(n) * n, rng));
  for (int i = 0; i < n; ++i)
    (*a)[i + size_t(i) * n] += T(static_cast<typename RealType<T>::type>(n));
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
        RandomVector<T>(size_t(s.m) * s.n, rng));
    Case<T> c;
    c.label = Label(s.m, s.n);
    c.flops = (kIsComplex<T> ? 8.0 : 2.0) * s.m * s.n * ka;
    c.initial_out = RandomVector<T>(size_t(s.m) * s.n, rng);
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
    const auto a =
        std::make_shared<std::vector<T>>(RandomVector<T>(size_t(n) * k, rng));
    Case<T> c;
    c.label = Label(n, k);
    c.flops = (kIsComplex<T> ? 4.0 : 1.0) * n * (n + 1.0) * k;
    c.initial_out = RandomVector<T>(size_t(n) * n, rng);
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
        RandomVector<T>(size_t(s.m) * s.n, rng));
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

template <typename T>
void BenchTrmv(const Options& opt, const Libraries& libs) {
  const char trans = opt.trans.empty() ? 'N' : opt.trans[0];
  const std::string symbol = Symbol<T>("trmv");
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
      // Restart from x0 so that repeated products stay bounded.
      std::copy(x0->begin(), x0->end(), out);
      reinterpret_cast<Fn>(fn)(&uplo, &trans, &diag, &n, a->data(), &n, out,
                               &inc);
    };
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
  } else if (opt.routine == "trmm" || opt.routine == "trsm") {
    BenchTriangular3<T>(opt, libs, opt.routine.c_str());
  } else if (opt.routine == "symv") {
    BenchSymv<T>(opt, libs);
  } else if (opt.routine == "trmv") {
    BenchTrmv<T>(opt, libs);
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

std::string ExeDirectory(const char* argv0) {
  std::string path = argv0;
  const size_t pos = path.find_last_of("/\\");
  return pos == std::string::npos ? "." : path.substr(0, pos);
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
#if defined(_WIN32)
  lib_specs.emplace_back("hwyBLAS", ExeDirectory(argv[0]) + "/hwyblas.dll");
#else
  lib_specs.emplace_back("hwyBLAS", ExeDirectory(argv[0]) + "/libhwyblas.so");
#endif

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
