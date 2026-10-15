// BLAS benchmark: compares single-threaded GFLOP/s of BLAS libraries that are
// loaded at runtime.
//
// Usage:
//   blas_bench <routine> [options] [name=path/to/blas.dll ...]
//
// Routines: gemm, gemv
//
// Options:
//   --types=sdcz        precisions to run (default: d)
//   --trans=NN          gemm: transa/transb (default: NN); gemv: one letter
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
  std::vector<Shape> shapes;
  double min_time = 0.5;  // per library and shape, over all rounds
  int rounds = 3;
  int pin_cpu = -1;  // -1: no pinning
};

using Libraries = std::vector<std::unique_ptr<BlasLibrary>>;

template <typename T>
constexpr bool kIsComplex =
    !std::is_same_v<T, float> && !std::is_same_v<T, double>;

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

template <typename T>
bool RunRoutine(const Options& opt, const Libraries& libs) {
  if (opt.routine == "gemm") {
    BenchGemm<T>(opt, libs);
  } else if (opt.routine == "gemv") {
    BenchGemv<T>(opt, libs);
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
            "Usage: %s <gemm|gemv> [--types=sdcz] [--trans=..] "
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
