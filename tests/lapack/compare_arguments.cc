// Compares the argument validation of the ported LAPACK routines with the
// reference LAPACK's, loading both libraries at runtime:
//
//   compare_arguments <hwyblas library> <reference liblapack> <lapack.h> list
//   compare_arguments <hwyblas library> <reference liblapack> <lapack.h> run
//                     <routine> [trials] [seed] [Fortran source]
//   compare_arguments <hwyblas library> <reference liblapack> <lapack.h>
//                     queries <routine>
//
// The routines and their arguments come from the declarations in lapack.h:
// every routine that returns void and takes only integers, characters and
// real arrays (no callbacks) and has an INFO argument. For each, a valid call
// is found by changing the arguments that the reference reports as invalid
// (INFO = -k) until it accepts them; then trials change one to three of the
// character and integer arguments (dimensions, leading dimensions,
// workspace sizes; given the Fortran source, only those it checks) to random
// values, valid or not, and compare INFO. A workspace query (LWORK = -1) that
// both libraries accept also compares WORK(1). The reference is the oracle: its
// INFO is the one hwyBLAS must return, and an argument checked in another order
// or not at all shows up as a different INFO.
//
// Both libraries run on copies of the same buffers (random reals, identity
// permutations); the dimensions stay below 10 and the buffers are generous,
// so that a valid call stays in bounds. Routines that need structured input
// (the divide and conquer kernels, for example) can still crash the
// reference or hwyBLAS: a crash is caught and reported, the program names
// each call before it runs with COMPARE_ARGUMENTS_TRACE=1, a case that takes
// longer than 20 seconds ends it with exit code 3 and its name, and
// compare_reference.sh (mode "arguments") runs one process per routine.
//
// "queries" is a second check: a workspace query with one argument at a time
// replaced by a null pointer. Callers leave arguments that the reference does
// not read in a query unset (NumPy's lstsq hands xGELSD an uninitialized
// RCOND), so where the reference survives a null argument, hwyBLAS must too
// and return the same INFO. Not part of the build or of ctest.
//
// Copyright (c) 2026, Roman Hiestand. BSD 3-Clause License.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "blas_loader.h"
#include "hwyblas/lapack.h"

namespace {

using hwyblas_bench::BlasLibrary;
using I = hwyblas_int;

enum class Kind {
  kChar,    // const char*
  kInt,     // const hwyblas_int*: a dimension, a leading dimension or a size
  kBuffer,  // everything else: an array (or scalar) that is not changed
};
enum class IntClass { kDimension, kLeading, kWorkspace };

struct Param {
  std::string name;
  Kind kind = Kind::kBuffer;
  IntClass int_class = IntClass::kDimension;
  bool is_integer_buffer = false;  // hwyblas_int* (INFO, IPIV, IWORK, ...)
  size_t element = 4;              // float or double
};

struct Routine {
  std::string name;
  std::vector<Param> params;
  int info = -1;  // index of INFO
  int work = -1;  // index of WORK, for the workspace query
  int lwork = -1;
};

// The routines that lapack.h declares and this program can call.
std::vector<Routine> ParseHeader(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  // Without comments.
  const std::string raw = text.str();
  std::string s;
  for (size_t i = 0; i < raw.size();) {
    if (raw.compare(i, 2, "/*") == 0) {
      const size_t e = raw.find("*/", i + 2);
      i = e == std::string::npos ? raw.size() : e + 2;
      s += ' ';
    } else {
      s += raw[i++];
    }
  }
  std::vector<Routine> routines;
  for (size_t pos = s.find("HWYBLAS_API"); pos != std::string::npos;
       pos = s.find("HWYBLAS_API", pos + 1)) {
    // "HWYBLAS_API <return type> <name>(<parameters>);"
    const size_t open = s.find('(', pos), semicolon = s.find(';', pos);
    if (open == std::string::npos || semicolon == std::string::npos ||
        open > semicolon) {
      continue;
    }
    const size_t close = s.rfind(')', semicolon);
    std::stringstream head(s.substr(pos + 11, open - pos - 11));
    std::vector<std::string> hw;
    for (std::string x; head >> x;) hw.push_back(x);
    if (hw.size() != 2 || hw[0] != "void") continue;
    const std::string parameters = s.substr(open + 1, close - open - 1);
    {
      Routine r;
      r.name = hw[1];
      bool supported = true;
      std::stringstream params(parameters);
      std::string p;
      while (std::getline(params, p, ',')) {
        std::stringstream words(p);
        std::vector<std::string> w;
        std::string x;
        while (words >> x) w.push_back(x);
        if (w.empty()) continue;
        // Split "type*" and the name ("const double* a", "double *a").
        std::string name = w.back();
        std::string type;
        for (size_t i = 0; i + 1 < w.size(); ++i) type += w[i] + " ";
        while (!name.empty() && name[0] == '*') {
          type += "*";
          name.erase(0, 1);
        }
        type.erase(std::remove(type.begin(), type.end(), ' '), type.end());
        Param q;
        q.name = name;
        if (type == "constchar*") {
          q.kind = Kind::kChar;
        } else if (type == "consthwyblas_int*" &&
                   (name == "ipiv" || name == "select" || name == "jpiv" ||
                    name == "idxc" || name == "ctot")) {
          // Input arrays of integers (permutations, selections).
          q.is_integer_buffer = true;
          q.element = sizeof(I);
        } else if (type == "consthwyblas_int*") {
          q.kind = Kind::kInt;
          if (name.rfind("ld", 0) == 0) {
            q.int_class = IntClass::kLeading;
          } else if (name == "lwork" || name == "liwork" || name == "lrwork") {
            q.int_class = IntClass::kWorkspace;
            if (name == "lwork") r.lwork = static_cast<int>(r.params.size());
          }
        } else if (type == "hwyblas_int*") {
          q.is_integer_buffer = true;
          q.element = sizeof(I);
          if (name == "info") r.info = static_cast<int>(r.params.size());
        } else if (type == "float*" || type == "constfloat*") {
          q.element = 4;
        } else if (type == "double*" || type == "constdouble*") {
          q.element = 8;
        } else {
          supported = false;  // callbacks, size_t, char*
        }
        if (name == "work" && !q.is_integer_buffer) {
          r.work = static_cast<int>(r.params.size());
        }
        r.params.push_back(q);
      }
      if (supported && r.info >= 0) routines.push_back(std::move(r));
    }
  }
  return routines;
}

// Calls fn with n pointer arguments.
template <size_t... Is>
void CallWith(void* fn, void** a, std::index_sequence<Is...>) {
  reinterpret_cast<void (*)(
      decltype((void)Is, static_cast<void*>(nullptr))...)>(fn)(a[Is]...);
}
template <size_t N>
void Dispatch(size_t n, void* fn, void** a) {
  if (n == N) {
    CallWith(fn, a, std::make_index_sequence<N>{});
  } else if constexpr (N > 1) {
    Dispatch<N - 1>(n, fn, a);
  }
}

// Calls the routine; false if it crashed (an access violation or a signal),
// so that a routine that dereferences an argument it should not is reported
// instead of ending the program.
#if defined(_WIN32)
bool GuardedDispatch(size_t n, void* fn, void** a) {
  __try {
    Dispatch<80>(n, fn, a);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return false;
  }
  return true;
}
#else
sigjmp_buf g_jump;
void OnSignal(int) { siglongjmp(g_jump, 1); }
bool GuardedDispatch(size_t n, void* fn, void** a) {
  struct sigaction action{};
  struct sigaction old_segv{}, old_bus{};
  action.sa_handler = OnSignal;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_NODEFER;
  sigaction(SIGSEGV, &action, &old_segv);
  sigaction(SIGBUS, &action, &old_bus);
  bool ok = true;
  if (sigsetjmp(g_jump, 1) == 0) {
    Dispatch<80>(n, fn, a);
  } else {
    ok = false;
  }
  sigaction(SIGSEGV, &old_segv, nullptr);
  sigaction(SIGBUS, &old_bus, nullptr);
  return ok;
}
#endif

std::mt19937_64 rng;

constexpr size_t kBufferBytes = 4096 * 16;
constexpr I kInfoUnset = 12345;

struct Value {
  char c = 'N';
  I i = 0;
};

const char kChars[] = "NUTLCVASORBEPQGIDFHKXMnutlcvasZ?";
const I kDimensions[] = {-1, 0, 1, 2, 3, 4, 5, 6, 8, 9};
const I kLeadings[] = {-1, 0, 1, 2, 3, 4, 8, 9, 16};
const I kWorkspaces[] = {-1, 0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1024};

Value RandomValue(const Param& p) {
  Value v;
  if (p.kind == Kind::kChar) {
    v.c = kChars[rng() % (sizeof(kChars) - 1)];
  } else if (p.int_class == IntClass::kLeading) {
    v.i = kLeadings[rng() % std::size(kLeadings)];
  } else if (p.int_class == IntClass::kWorkspace) {
    v.i = kWorkspaces[rng() % std::size(kWorkspaces)];
  } else {
    v.i = kDimensions[rng() % std::size(kDimensions)];
  }
  return v;
}

// The candidates of a parameter for the search of a valid call, best first.
std::vector<Value> Candidates(const Param& p) {
  std::vector<Value> out;
  if (p.kind == Kind::kChar) {
    std::string pool = "NUTLCVASORBEPQGIDFHKXM";
    std::shuffle(pool.begin(), pool.end(), rng);
    for (const char c : pool) {
      Value v;
      v.c = c;
      out.push_back(v);
    }
  } else if (p.int_class == IntClass::kLeading) {
    for (const I x : {16, 9, 8, 4}) out.push_back(Value{'N', x});
  } else if (p.int_class == IntClass::kWorkspace) {
    for (const I x : {1024, 513, 257, 129, 65, 33, 17, 9, 5, 3, 2, 1}) {
      out.push_back(Value{'N', x});
    }
  } else {
    std::vector<I> pool = {4, 3, 5, 2, 1, 6, 8, 9, 0};
    std::shuffle(pool.begin() + 1, pool.end(), rng);
    for (const I x : pool) out.push_back(Value{'N', x});
  }
  return out;
}

// One call: the values of the characters and integers, and the buffers.
struct Call {
  const Routine* routine;
  std::vector<Value> values;
};

struct Outcome {
  bool crashed = false;
  I info = kInfoUnset;
  double work = 0;
};

class Buffers {
 public:
  explicit Buffers(const Routine& r) : routine_(&r) {
    for (size_t k = 0; k < r.params.size(); ++k) {
      pristine_.emplace_back(kBufferBytes, 0);
      Fill(k, InitialStyle(r.params[k]));
    }
  }

  // Fill styles of an array of reals: 0 random in [-1, 1], 1 identity (for
  // leading dimension 16), 2 ones, 3 random in [0.5, 1.5]. Routines reject
  // arrays that are not what they expect (a quasi-triangular matrix, a
  // negative norm).
  static constexpr int kStyles = 4;

  static int InitialStyle(const Param& p) {
    if (p.name == "anorm") return 3;
    // Permutation indexes and scale factors of xGEBAK and xGGBAK.
    if (p.name == "scale" || p.name == "lscale" || p.name == "rscale") return 2;
    return 0;
  }

  bool IsReal(size_t k) const {
    const Param& p = routine_->params[k];
    return p.kind == Kind::kBuffer && !p.is_integer_buffer;
  }

  void Fill(size_t k, int style) {
    const Param& p = routine_->params[k];
    std::vector<unsigned char>& b = pristine_[k];
    if (p.kind != Kind::kBuffer) return;
    const size_t count = kBufferBytes / p.element;
    std::uniform_real_distribution<double> d(-1, 1), positive(0.5, 1.5);
    for (size_t i = 0; i < count; ++i) {
      if (p.is_integer_buffer) {
        const I x = p.name == "info" ? kInfoUnset : static_cast<I>(i + 1);
        memcpy(&b[i * sizeof(I)], &x, sizeof(I));
        continue;
      }
      double v = 0;
      switch (style) {
        case 0:
          v = d(rng);
          break;
        case 1:
          v = i % 17 == 0 ? 1 : 0;
          break;
        case 2:
          v = 1;
          break;
        default:
          v = positive(rng);
          break;
      }
      if (p.element == 4) {
        const float x = static_cast<float>(v);
        memcpy(&b[i * 4], &x, 4);
      } else {
        memcpy(&b[i * 8], &v, 8);
      }
    }
  }

  // Runs the call in the library on copies of the buffers.
  Outcome Run(const BlasLibrary& lib, const Call& call,
              int null_param = -1) const {
    const Routine& r = *call.routine;
    std::vector<std::vector<unsigned char>> work = pristine_;
    std::vector<char> chars(r.params.size());
    std::vector<I> ints(r.params.size());
    std::vector<void*> args(r.params.size());
    for (size_t k = 0; k < r.params.size(); ++k) {
      if (r.params[k].kind == Kind::kChar) {
        chars[k] = call.values[k].c;
        args[k] = &chars[k];
      } else if (r.params[k].kind == Kind::kInt) {
        ints[k] = call.values[k].i;
        args[k] = &ints[k];
      } else {
        args[k] = work[k].data();
      }
    }
    if (null_param >= 0) args[null_param] = nullptr;
    void* fn = lib.Find(r.name.c_str());
    Outcome o;
    if (!GuardedDispatch(r.params.size(), fn, args.data())) {
      o.crashed = true;
      return o;
    }
    memcpy(&o.info, work[r.info].data(), sizeof(I));
    if (r.work >= 0) {
      if (r.params[r.work].element == 4) {
        float x;
        memcpy(&x, work[r.work].data(), 4);
        o.work = x;
      } else {
        memcpy(&o.work, work[r.work].data(), 8);
      }
    }
    return o;
  }

 private:
  const Routine* routine_;
  std::vector<std::vector<unsigned char>> pristine_;
};

std::string Describe(const Call& call) {
  const Routine& r = *call.routine;
  std::string s;
  for (size_t k = 0; k < r.params.size(); ++k) {
    if (r.params[k].kind == Kind::kBuffer) continue;
    if (!s.empty()) s += " ";
    s += r.params[k].name + "=";
    s += r.params[k].kind == Kind::kChar ? std::string(1, call.values[k].c)
                                         : std::to_string(call.values[k].i);
  }
  return s;
}

std::atomic<long long> g_case_start{0};
char g_case[400];

long long Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count() +
         1;
}

void StartWatchdog() {
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const long long start = g_case_start.load();
      if (start != 0 && Now() - start > 20) {
        printf("HANG: %s\n", g_case);
        fflush(stdout);
        _Exit(3);
      }
    }
  }).detach();
}

void NoHandler(const char*, hwyblas_int) {}

struct Runner {
  const BlasLibrary& hwy;
  const BlasLibrary& ref;
  bool trace = getenv("COMPARE_ARGUMENTS_TRACE") != nullptr;

  Outcome Execute(const BlasLibrary& lib, const Buffers& b, const Call& call,
                  int null_param = -1) {
    snprintf(g_case, sizeof(g_case), "%s %s: %s", lib.name().c_str(),
             call.routine->name.c_str(), Describe(call).c_str());
    if (trace) {
      fprintf(stderr, "%s\n", g_case);
      fflush(stderr);
    }
    g_case_start = Now();
    const Outcome o = b.Run(lib, call, null_param);
    g_case_start = 0;
    return o;
  }

  // Changes the arguments that the reference reports (INFO = -k) until it
  // accepts the call; false if it does not.
  bool FindValid(Buffers& b, Call& call) {
    const Routine& r = *call.routine;
    std::vector<std::vector<Value>> candidates(r.params.size());
    std::vector<size_t> next(r.params.size(), 1);
    std::vector<int> style(r.params.size(), 0);
    for (size_t k = 0; k < r.params.size(); ++k) {
      style[k] = Buffers::InitialStyle(r.params[k]);
    }
    for (size_t k = 0; k < r.params.size(); ++k) {
      if (r.params[k].kind == Kind::kBuffer) continue;
      candidates[k] = Candidates(r.params[k]);
      call.values[k] = candidates[k][0];
    }
    for (int step = 0; step < 300; ++step) {
      const Outcome o = Execute(ref, b, call);
      if (o.crashed) return false;
      if (o.info >= 0) return true;
      const long k = -static_cast<long>(o.info) - 1;
      if (k < 0 || k >= static_cast<long>(r.params.size())) return false;
      if (b.IsReal(k)) {
        // An array that the routine rejects: try the next kind of fill.
        if (++style[k] >= Buffers::kStyles) return false;
        b.Fill(k, style[k]);
        continue;
      }
      if (next[k] >= candidates[k].size()) return false;
      call.values[k] = candidates[k][next[k]++];
    }
    return false;
  }
};

int List(const std::vector<Routine>& routines, const BlasLibrary& hwy,
         const BlasLibrary& ref) {
  for (const Routine& r : routines) {
    if (hwy.Find(r.name.c_str()) && ref.Find(r.name.c_str()) &&
        r.params.size() <= 80) {
      printf("%s\n", r.name.c_str());
    }
  }
  return 0;
}

// The positions of the arguments that the Fortran source checks: the N of
// every "INFO = -N". Empty without a source (then all are changed).
std::set<int> CheckedPositions(const std::string& path) {
  std::set<int> positions;
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  std::string s = text.str();
  for (char& c : s)
    c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
  for (size_t i = s.find("INFO"); i != std::string::npos;
       i = s.find("INFO", i + 1)) {
    size_t j = i + 4;
    while (j < s.size() && s[j] == ' ') ++j;
    if (j >= s.size() || s[j] != '=') continue;
    ++j;
    while (j < s.size() && s[j] == ' ') ++j;
    if (j >= s.size() || s[j] != '-') continue;
    ++j;
    if (j < s.size() && isdigit(static_cast<unsigned char>(s[j]))) {
      positions.insert(atoi(s.c_str() + j));
    }
  }
  return positions;
}

// Calls whose results differ on purpose (doc/deviations.md): xGEQP3 with M = 0
// and N > 0 returns the INFO = -12 of the reference's xUNMQR call there.
bool KnownDeviation(const Routine& r, const Call& call) {
  if (r.name.size() < 7 || r.name.compare(1, 6, "geqp3_") != 0) return false;
  I m = 1, n = 0;
  for (size_t k = 0; k < r.params.size(); ++k) {
    if (r.params[k].name == "m") m = call.values[k].i;
    if (r.params[k].name == "n") n = call.values[k].i;
  }
  return m == 0 && n > 0;
}

int Run(const std::vector<Routine>& routines, const BlasLibrary& hwy,
        const BlasLibrary& ref, const std::string& name, int trials,
        unsigned seed, const std::string& source) {
  const Routine* routine = nullptr;
  for (const Routine& r : routines) {
    if (r.name == name) routine = &r;
  }
  if (routine == nullptr || !hwy.Find(name.c_str()) ||
      !ref.Find(name.c_str())) {
    printf("%s: not found\n", name.c_str());
    return 2;
  }
  rng.seed(seed);
  const std::set<int> checked =
      source.empty() ? std::set<int>() : CheckedPositions(source);
  StartWatchdog();
  Runner runner{hwy, ref};
  int mismatches = 0, calls = 0, accepted = 0, bases = 0;
  Buffers buffers(*routine);
  for (int base = 0; base < 4; ++base) {
    Call valid{routine, std::vector<Value>(routine->params.size())};
    if (!runner.FindValid(buffers, valid)) continue;
    ++bases;
    for (int t = 0; t < trials / 4; ++t) {
      Call call = valid;
      const int changes = 1 + static_cast<int>(rng() % 3);
      for (int c = 0; c < changes; ++c) {
        const size_t k = rng() % routine->params.size();
        if (routine->params[k].kind != Kind::kBuffer &&
            (checked.empty() || checked.count(static_cast<int>(k) + 1))) {
          call.values[k] = RandomValue(routine->params[k]);
        }
      }
      const Outcome x = runner.Execute(hwy, buffers, call);
      const Outcome y = runner.Execute(ref, buffers, call);
      ++calls;
      if (y.crashed) continue;  // the reference rejects nothing here: skip
      if (x.crashed) {
        ++mismatches;
        if (mismatches <= 5) {
          printf("%s: hwyBLAS crashed, the reference returned INFO %d: %s\n",
                 name.c_str(), int(y.info), Describe(call).c_str());
        }
        continue;
      }
      if (y.info >= 0) ++accepted;
      if (KnownDeviation(*routine, call)) continue;
      bool query = routine->lwork >= 0 && call.values[routine->lwork].i == -1 &&
                   x.info == 0 && y.info == 0 && routine->work >= 0;
      if (x.info != y.info) {
        ++mismatches;
        if (mismatches <= 5) {
          printf("%s: INFO %d, reference %d: %s\n", name.c_str(), int(x.info),
                 int(y.info), Describe(call).c_str());
        }
      } else if (query && x.work != y.work) {
        ++mismatches;
        if (mismatches <= 5) {
          printf("%s: workspace query %.17g, reference %.17g: %s\n",
                 name.c_str(), x.work, y.work, Describe(call).c_str());
        }
      }
    }
  }
  if (bases == 0) {
    printf("%s: no valid call found\n", name.c_str());
    return 0;
  }
  printf("%s: %d calls (%d accepted by the reference), %d mismatches\n",
         name.c_str(), calls, accepted, mismatches);
  return mismatches ? 1 : 0;
}

// Workspace queries (LWORK = -1) with one argument at a time replaced by a
// null pointer: callers (NumPy's lstsq, for one) leave arguments that the
// reference does not reference in a query unset. If the reference survives a
// null argument, hwyBLAS must too and return the same INFO.
int RunQueries(const std::vector<Routine>& routines, const BlasLibrary& hwy,
               const BlasLibrary& ref, const std::string& name) {
  const Routine* routine = nullptr;
  for (const Routine& r : routines) {
    if (r.name == name) routine = &r;
  }
  if (routine == nullptr || !hwy.Find(name.c_str()) ||
      !ref.Find(name.c_str())) {
    printf("%s: not found\n", name.c_str());
    return 2;
  }
  std::vector<size_t> sizes;  // the workspace size arguments
  for (size_t k = 0; k < routine->params.size(); ++k) {
    if (routine->params[k].kind == Kind::kInt &&
        routine->params[k].int_class == IntClass::kWorkspace) {
      sizes.push_back(k);
    }
  }
  if (sizes.empty()) return 0;  // no workspace query
  rng.seed(1);
  StartWatchdog();
  Runner runner{hwy, ref};
  Buffers buffers(*routine);
  Call query{routine, std::vector<Value>(routine->params.size())};
  if (!runner.FindValid(buffers, query)) {
    printf("%s: no valid call found\n", name.c_str());
    return 0;
  }
  for (const size_t k : sizes) query.values[k].i = -1;
  int mismatches = 0, tried = 0;
  for (size_t k = 0; k < routine->params.size(); ++k) {
    if (static_cast<int>(k) == routine->info) continue;
    const Outcome y = runner.Execute(ref, buffers, query, static_cast<int>(k));
    if (y.crashed) continue;  // the reference references it
    ++tried;
    const Outcome x = runner.Execute(hwy, buffers, query, static_cast<int>(k));
    if (x.crashed || x.info != y.info) {
      ++mismatches;
      printf(
          "%s: %s is null in a workspace query: hwyBLAS %s, reference INFO "
          "%d\n",
          name.c_str(), routine->params[k].name.c_str(),
          x.crashed ? "crashed"
                    : ("returned INFO " + std::to_string(int(x.info))).c_str(),
          int(y.info));
    }
  }
  printf("%s: %d null arguments survived by the reference, %d mismatches\n",
         name.c_str(), tried, mismatches);
  return mismatches ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: compare_arguments <hwyblas library> <reference liblapack> "
            "<lapack.h> list | run <routine> [trials] [seed]\n");
    return 2;
  }
  const BlasLibrary hwy("hwyBLAS", argv[1]), ref("reference", argv[2]);
  if (!hwy.ok() || !ref.ok()) return 2;
  const auto set_ilaenv =
      hwy.Get<hwyblas_ilaenv_function (*)(hwyblas_ilaenv_function)>(
          "hwyblas_set_ilaenv");
  const auto ref_ilaenv = ref.Get<hwyblas_ilaenv_function>("ilaenv_");
  const auto set_handler =
      hwy.Get<hwyblas_error_handler (*)(hwyblas_error_handler)>(
          "hwyblas_set_error_handler");
  if (!set_ilaenv || !ref_ilaenv || !set_handler) {
    fprintf(stderr, "hwyblas_set_ilaenv or ilaenv_ not found\n");
    return 2;
  }
  set_ilaenv(ref_ilaenv);
  set_handler(NoHandler);
  const std::vector<Routine> routines = ParseHeader(argv[3]);
  if (strcmp(argv[4], "list") == 0) return List(routines, hwy, ref);
  if (strcmp(argv[4], "run") == 0 && argc > 5) {
    return Run(routines, hwy, ref, argv[5], argc > 6 ? atoi(argv[6]) : 400,
               argc > 7 ? static_cast<unsigned>(atoll(argv[7])) : 1,
               argc > 8 ? argv[8] : "");
  }
  if (strcmp(argv[4], "queries") == 0 && argc > 5) {
    return RunQueries(routines, hwy, ref, argv[5]);
  }
  fprintf(stderr, "unknown command\n");
  return 2;
}
