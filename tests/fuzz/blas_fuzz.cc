// Differential fuzzer: calls hwyBLAS and a reference BLAS (e.g. OpenBLAS)
// with the same random arguments and compares the results.
//
// Usage:
//   blas_fuzz [options] Name=path/to/reference/blas.dll
//
// Options:
//   --seed=N            seed of the case sequence (default 1)
//   --iterations=N      number of cases (default 1000)
//   --time=S            run for S seconds instead (soak testing)
//   --types=sdcz        precisions (default: all)
//   --routines=a,b      only these routines, e.g. dgemm,ztrsm
//   --repro=name:seed   run one case, as printed for a failure
//   --max-failures=N    stop after N failures (default 10)
//   --verbose           print every case
//   --threads=N         run the cases on N concurrent threads, replacing the
//                       threads every few hundred cases (for ThreadSanitizer)
//   --netlib            the reference is the Netlib reference BLAS and CBLAS
//                       (see netlib/CMakeLists.txt): also zero and negative
//                       increments where they are defined or a no-op
//   --special=KIND      some inputs are special: nan, inf, zero, subnormal or
//                       huge. Meant for --netlib; differences are expected
//                       and have to be judged, see the category in brackets.
//   --fp-flags          fail if hwyBLAS raises an invalid, divide-by-zero or
//                       overflow exception that the reference does not
//
// Each case picks a routine and random arguments: the Fortran or the CBLAS
// interface (in column- or row-major layout), sizes from 0 to beyond the
// kernels' blocking, transposes, sides, triangles, alpha/beta including 0
// and 1, leading dimensions with padding, positive or negative increments,
// and complex operands aligned only to their real type. Besides the results, a
// case checks that hwyBLAS
//  - leaves every element it must not reference unchanged and unused: these
//    elements (padding, gaps between vector elements, the other triangle, a
//    unit diagonal, the imaginary part of a Hermitian diagonal) are NaN for
//    hwyBLAS and zero for the reference library;
//  - does not read inputs that BLAS defines to be ignored, such as C when
//    beta = 0 or A and B when alpha = 0 (also NaN for hwyBLAS);
//  - does not modify its inputs;
//  - does not access memory outside its operands: each operand starts or
//    ends at an inaccessible page.
// Results must agree within a bound of the form c * eps * (k + 1) * scale,
// where k is the length of the inner products and scale the size of the
// result. hwyBLAS is loaded at runtime (the library built with the fuzzer),
// so the environment variable HWYBLAS_TARGET selects the target that is
// tested.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "blas_loader.h"

#if !defined(_WIN32)
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace hwyblas_fuzz {
namespace {

using Lib = hwyblas_bench::BlasLibrary;
using Rng = std::mt19937_64;

template <typename T>
constexpr bool kIsComplex = !std::is_floating_point_v<T>;

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
T Poison() {
  const Real<T> nan = std::numeric_limits<Real<T>>::quiet_NaN();
  if constexpr (kIsComplex<T>) {
    return T(nan, nan);
  } else {
    return nan;
  }
}

template <typename T>
bool IsNan(T x) {
  if constexpr (kIsComplex<T>) {
    return std::isnan(x.real()) || std::isnan(x.imag());
  } else {
    return std::isnan(x);
  }
}

template <typename T>
bool SameBits(const T& a, const T& b) {
  return memcmp(&a, &b, sizeof(T)) == 0;
}

struct Options {
  uint64_t seed = 1;
  int64_t iterations = 1000;
  double time = 0;  // > 0: run for this many seconds
  std::string types = "sdcz";
  std::vector<std::string> routines;  // empty: all
  std::string repro;                  // name:seed
  int max_failures = 10;
  bool verbose = false;
  bool netlib = false;
  std::string special;  // empty: ordinary values
  bool fp_flags = false;
  int threads = 1;
};

// A character argument and its CBLAS enumerator.
struct Flag {
  char letter;
  int cblas;
  // NOLINTNEXTLINE(google-explicit-constructor): used as a char.
  operator char() const { return letter; }
};

// Placeholder for the layout, the first argument of CBLAS Level 2 and 3.
struct Layout {};

enum class Api { kFortran, kCblasCol, kCblasRow };

std::string ToString(int v) { return std::to_string(v); }
std::string ToString(char v) { return std::string(1, v); }
std::string ToString(double v) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}
std::string ToString(float v) { return ToString(static_cast<double>(v)); }
template <typename R>
std::string ToString(std::complex<R> v) {
  return "(" + ToString(v.real()) + "," + ToString(v.imag()) + ")";
}

// ---------------------------------------------------------------------------
// Memory that starts or ends at an inaccessible page, so that reading or
// writing outside of it crashes.

size_t PageSize() {
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize;
#else
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

class GuardedBuffer {
 public:
  // `bytes` bytes, directly before (`guard_after`) or after an inaccessible
  // page. Both ends are guarded if `bytes` is a multiple of the page size.
  GuardedBuffer(size_t bytes, bool guard_after) {
    const size_t page = PageSize();
    const size_t data_pages = (bytes + page - 1) / page;
    size_ = (data_pages + 2) * page;
#if defined(_WIN32)
    base_ = static_cast<char*>(
        VirtualAlloc(nullptr, size_, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS));
    if (base_ == nullptr) abort();
    DWORD old;
    if (data_pages > 0 && !VirtualProtect(base_ + page, data_pages * page,
                                          PAGE_READWRITE, &old)) {
      abort();
    }
#else
    void* p =
        mmap(nullptr, size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) abort();
    base_ = static_cast<char*>(p);
    if (data_pages > 0 && mprotect(base_ + page, data_pages * page,
                                   PROT_READ | PROT_WRITE) != 0) {
      abort();
    }
#endif
    data_ =
        guard_after ? base_ + page + data_pages * page - bytes : base_ + page;
  }
  ~GuardedBuffer() {
#if defined(_WIN32)
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, size_);
#endif
  }
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;

  void* data() const { return data_; }

 private:
  char* base_ = nullptr;
  char* data_ = nullptr;
  size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Test cases

// How a routine uses an element of an operand.
enum class Use : uint8_t {
  kUnused,    // must not be referenced
  kUsed,      // read and/or written
  kRealOnly,  // Hermitian diagonal: the imaginary part is not referenced
};

template <typename T>
struct Operand {
  std::string name;
  bool output = false;
  bool exact = false;     // integer result: must match exactly
  bool relative = false;  // tolerance relative to each element
  bool misalign = false;  // aligned only to the real type
  int ld = 0;             // leading dimension, for messages (0: vector)
  bool guard_after = true;
  std::vector<Use> use;
  std::vector<T> hwy_init;  // initial values for hwyBLAS
  std::vector<T> ref_init;  // initial values for the reference library
};

// The routine to call, of one of the libraries.
struct Target {
  const Lib& lib;
  const std::string& symbol;
  Api api;
};

template <typename T>
struct Case {
  std::string name;    // routine, e.g. "dgemm"
  std::string symbol;  // symbol that both libraries must export
  std::string params;  // description of the arguments
  Api api = Api::kFortran;
  std::vector<Operand<T>> ops;
  int error_dim = 0;  // length of the inner products / of the solve
  // Calls the routine `symbol` of a library with the operands.
  std::function<void(const Target&, T* const*)> call;
  // If set, computes the expected results instead of the reference library.
  std::function<void(T* const*)> expected;
  // If set, checks hwyBLAS's results instead of comparing them: returns the
  // problem, or an empty string.
  std::function<std::string(const std::vector<std::vector<T>>&)> verify;
};

// The libraries' functions are declared with other (const, void*) argument
// types than the ones of the call: the same ABI, but clang's
// -fsanitize=function (UBSan) reports it.
#if defined(__clang__)
#define FUZZ_NO_SANITIZE_FUNCTION __attribute__((no_sanitize("function")))
#else
#define FUZZ_NO_SANITIZE_FUNCTION
#endif
template <typename R = void, typename... Args>
FUZZ_NO_SANITIZE_FUNCTION R Call(const Lib& lib, const std::string& symbol,
                                 Args... args) {
  return lib.Get<R (*)(Args...)>(symbol.c_str())(args...);
}

// The CBLAS form of an argument given as for Fortran: dimensions and real
// scalars (pointers to const) are passed by value, characters as enumerators;
// complex scalars and arrays (pointers to non-const) stay pointers.
std::tuple<int> CblasArg(Api api, Layout /*layout*/) {
  return {api == Api::kCblasRow ? 101 : 102};
}
std::tuple<int> CblasArg(Api /*api*/, const Flag* f) { return {f->cblas}; }
std::tuple<int> CblasArg(Api /*api*/, const int* v) { return {*v}; }
template <typename V>
auto CblasArg(Api /*api*/, const V* v) {
  if constexpr (std::is_floating_point_v<V>) {
    return std::tuple<V>(*v);
  } else {
    return std::tuple<const V*>(v);
  }
}
template <typename V>
std::tuple<V*> CblasArg(Api /*api*/, V* v) {
  return {v};
}

std::tuple<> FortranArg(Layout /*layout*/) { return {}; }
std::tuple<const char*> FortranArg(const Flag* f) { return {&f->letter}; }
template <typename V>
std::tuple<V*> FortranArg(V* v) {
  return {v};
}

// Calls the target routine with arguments given as for Fortran.
template <typename R = void, typename... Args>
R Blas(const Target& t, Args... args) {
  const auto invoke = [&t](auto... converted) -> R {
    return Call<R>(t.lib, t.symbol, converted...);
  };
  if (t.api == Api::kFortran) {
    return std::apply(invoke, std::tuple_cat(FortranArg(args)...));
  }
  return std::apply(invoke, std::tuple_cat(CblasArg(t.api, args)...));
}

// Builds one case: chooses (and records) the arguments and operands.
template <typename T>
class Builder {
 public:
  using R = Real<T>;

  Builder(Case<T>& c, uint64_t seed, const Options& opt)
      : c_(c), opt_(opt), rng_(seed) {
    const int r = Uniform(0, 3);
    UseApi(r < 2 ? Api::kFortran : r == 2 ? Api::kCblasCol : Api::kCblasRow);
  }

  const Options& options() const { return opt_; }

  // Overrides the interface chosen at random.
  void UseApi(Api api) {
    c_.api = api;
    row_major_ = api == Api::kCblasRow;
  }
  // Operands are column-major whatever the interface.
  void ColumnMajorShapes() { row_major_ = false; }

  // The reference CBLAS (3.12.1) never returns from a complex ?gemv or ?gbmv
  // in row-major layout with a conjugate transpose and m = 0.
  bool HangsInNetlib(char trans, int m) const {
    return kIsComplex<T> && opt_.netlib && row_major_ && trans == 'C' && m == 0;
  }

  // The dimension that the leading dimension of a rows x cols matrix covers.
  int Major(int rows, int cols) const { return row_major_ ? cols : rows; }

  template <typename V>
  void Param(const char* key, const V& v) {
    c_.params += ' ';
    c_.params += key;
    c_.params += '=';
    c_.params += ToString(v);
  }

  int Uniform(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng_);
  }
  double Unit() { return std::uniform_real_distribution<double>(-1, 1)(rng_); }

  // A dimension; `level` 1, 2 or 3 selects the range of large sizes. At
  // most one dimension of a Level 3 case is huge, beyond the GEMM column
  // blocking (NC).
  int Dim(const char* key, int level) {
    const int r = Uniform(0, 199);
    static constexpr int kLarge[4] = {0, 20000, 400, 700};
    static constexpr int kMedium[4] = {0, 2000, 100, 300};
    int d;
    // (The reference BLAS is unblocked: huge cases take too long with it.)
    if (level == 3 && r < 2 && !huge_ && !opt_.netlib) {
      huge_ = true;
      d = Uniform(2000, 4300);
    } else if (r < 100) {
      d = Uniform(0, 16);
    } else if (r < 160) {
      d = Uniform(17, 80);
    } else if (r < 190) {
      d = Uniform(81, kMedium[level]);
    } else {
      d = Uniform(kMedium[level] + 1, kLarge[level]);
    }
    Param(key, d);
    return d;
  }

  // A band width up to about `n`.
  int Bandwidth(const char* key, int n) {
    const int r = Uniform(0, 9);
    const int k = r < 4 ? Uniform(0, 3) : r < 8 ? Uniform(0, n + 1) : n + 3;
    Param(key, k);
    return k;
  }

  // An increment; `zero`: 0 is defined (only tested against Netlib, other
  // libraries differ).
  int Inc(const char* key, bool zero = false) {
    static constexpr int kIncs[] = {1, 1, 1, 1, 1, -1, 2, -2, 3, -5, 8, 17};
    int inc = kIncs[Uniform(
        0, static_cast<int>(sizeof(kIncs) / sizeof(kIncs[0])) - 1)];
    if (zero && opt_.netlib && Uniform(0, 7) == 0) inc = 0;
    Param(key, inc);
    return inc;
  }
  // The increment of a routine for which a negative or zero one is a no-op or
  // differs between libraries: only against Netlib.
  int ForwardInc(const char* key) {
    const int inc = Inc(key, true);
    return opt_.netlib ? inc : std::abs(inc);
  }

  // A leading dimension for `rows` rows.
  int Ld(const char* key, int rows) {
    const int min = std::max(1, rows);
    const int r = Uniform(0, 9);
    int ld = min;
    if (r >= 5 && r < 8) {
      ld = min + Uniform(1, 8);
    } else if (r == 8) {
      // A multiple of 1 KiB, as in power-of-two sized matrices.
      const int step = static_cast<int>(1024 / sizeof(T));
      ld = (min + step - 1) / step * step;
    } else if (r == 9) {
      ld = min + Uniform(0, 100);
    }
    Param(key, ld);
    return ld;
  }

  Flag Letter(const char* key, const char* choices) {
    const char ch = choices[Uniform(0, static_cast<int>(strlen(choices)) - 1)];
    Param(key, ch);
    return MakeFlag(key, ch);
  }

  static Flag MakeFlag(const char* key, char ch) {
    int e = 0;
    if (strncmp(key, "trans", 5) == 0) {
      e = ch == 'N' ? 111 : ch == 'T' ? 112 : ch == 'C' ? 113 : 114;
    } else if (strcmp(key, "uplo") == 0) {
      e = ch == 'U' ? 121 : 122;
    } else if (strcmp(key, "diag") == 0) {
      e = ch == 'N' ? 131 : 132;
    } else if (strcmp(key, "side") == 0) {
      e = ch == 'L' ? 141 : 142;
    } else if (strcmp(key, "order") == 0) {
      e = ch == 'C' ? 102 : 101;
    }
    return Flag{ch, e};
  }

  // alpha (or beta, with more zeros): 0, 1, -1, a real or a general value.
  T Scalar(const char* key, bool beta = false) {
    const int r = Uniform(0, 99);
    T v;
    if (r < (beta ? 20 : 8)) {
      v = T(0);
    } else if (r < (beta ? 45 : 30)) {
      v = T(1);
    } else if (r < (beta ? 52 : 38)) {
      v = T(-1);
    } else if (r < 55) {
      v = T(static_cast<R>(2 * Unit()));
    } else {
      v = Random(2);
    }
    if (!opt_.special.empty() && Uniform(0, 19) == 0) v = Special(v);
    Param(key, v);
    return v;
  }
  R RealScalar(const char* key, bool beta = false) {
    const int r = Uniform(0, 99);
    R v;
    if (r < (beta ? 20 : 8)) {
      v = 0;
    } else if (r < (beta ? 45 : 30)) {
      v = 1;
    } else if (r < (beta ? 52 : 38)) {
      v = -1;
    } else {
      v = static_cast<R>(2 * Unit());
    }
    if (!opt_.special.empty() && Uniform(0, 19) == 0) v = SpecialReal();
    Param(key, v);
    return v;
  }

  // A random value with real and imaginary parts in [-scale, scale].
  T Random(double scale = 1) {
    if constexpr (kIsComplex<T>) {
      const R re = static_cast<R>(scale * Unit());
      return T(re, static_cast<R>(scale * Unit()));
    } else {
      return static_cast<T>(scale * Unit());
    }
  }

  // A random value with a random binary exponent, or zero.
  T Wide() {
    const auto part = [this]() {
      if (Uniform(0, 9) == 0) return R(0);
      return static_cast<R>(std::ldexp(Unit(), Uniform(-40, 40)));
    };
    if constexpr (kIsComplex<T>) {
      const R re = part();
      return T(re, part());
    } else {
      return part();
    }
  }

  // A special value of the kind given by --special.
  R SpecialReal() {
    using L = std::numeric_limits<R>;
    const bool negative = Uniform(0, 1) != 0;
    R v = 0;
    if (opt_.special == "nan") {
      v = L::quiet_NaN();
    } else if (opt_.special == "inf") {
      v = L::infinity();
    } else if (opt_.special == "subnormal") {
      v = std::max(L::denorm_min(),
                   static_cast<R>(std::abs(Unit())) * L::min());
    } else if (opt_.special == "huge") {
      v = static_cast<R>(0.5 + 0.25 * Unit()) * L::max();
    }
    return negative ? -v : v;
  }
  // `v` with one or both of its parts replaced by special values.
  T Special(T v) {
    if constexpr (kIsComplex<T>) {
      const int which = Uniform(0, 2);
      const R re = which != 1 ? SpecialReal() : v.real();
      return T(re, which != 0 ? SpecialReal() : v.imag());
    } else {
      return SpecialReal();
    }
  }

  // An element of an operand: index(i, j) is its position, or -1 for
  // elements that are not stored.
  struct Shape {
    int rows = 0, cols = 0;
    int ld = 0;  // for messages
    std::function<int64_t(int, int)> index;
    bool unit_diag = false;  // diagonal not referenced (assumed 1)
    bool hermitian = false;  // imaginary part of the diagonal not referenced
    bool solve = false;      // dominant diagonal: a well-conditioned solve
  };

  // Adds an operand and returns its number.
  int Add(const char* name, const Shape& s, bool output) {
    Operand<T> op;
    op.name = name;
    op.output = output;
    op.ld = s.ld;
    op.guard_after = Uniform(0, 3) != 0;
    op.misalign = kIsComplex<T> && Uniform(0, 3) == 0;
    int64_t size = 0;
    for (int j = 0; j < s.cols; ++j) {
      for (int i = 0; i < s.rows; ++i) size = std::max(size, s.index(i, j) + 1);
    }
    op.use.assign(size, Use::kUnused);
    op.hwy_init.assign(size, Poison<T>());
    op.ref_init.assign(size, T(0));
    // Off-diagonal elements of triangular solves are small, so that the
    // solution does not grow.
    const double off_scale =
        s.solve ? 1.0 / std::max(1, std::max(s.rows, s.cols)) : 1.0;
    for (int j = 0; j < s.cols; ++j) {
      for (int i = 0; i < s.rows; ++i) {
        const int64_t idx = s.index(i, j);
        if (idx < 0) continue;
        if (i == j && s.unit_diag) continue;
        T v = Random(i == j ? 1.0 : off_scale);
        if (i == j && s.solve) {
          // |v| in [1, 2], with a random sign or phase.
          const double abs = static_cast<double>(std::abs(v));
          if (abs < 0.1) v = T(1);
          v = v * static_cast<R>((1.5 + 0.5 * Unit()) / std::max(abs, 0.1));
        }
        op.use[idx] = Use::kUsed;
        op.hwy_init[idx] = v;
        op.ref_init[idx] = v;
        if constexpr (kIsComplex<T>) {
          if (i == j && s.hermitian) {
            op.use[idx] = Use::kRealOnly;
            op.hwy_init[idx] = T(v.real(), Poison<R>());
            op.ref_init[idx] = T(v.real(), 0);
          }
        }
      }
    }
    // Special values in some of the operands: a few elements, or for zeros
    // a larger part of them.
    if (!opt_.special.empty() && size > 0 && Uniform(0, 1) == 0) {
      const int64_t count =
          opt_.special == "zero" ? 1 + size / 4 : Uniform(1, 3);
      for (int64_t n = 0; n < count; ++n) {
        const int64_t idx = Uniform(0, static_cast<int>(size) - 1);
        if (op.use[idx] == Use::kUnused) continue;
        T v = Special(op.ref_init[idx]);
        op.hwy_init[idx] = v;
        op.ref_init[idx] = v;
        if constexpr (kIsComplex<T>) {
          if (op.use[idx] == Use::kRealOnly) {
            op.hwy_init[idx] = T(v.real(), Poison<R>());
            op.ref_init[idx] = T(v.real(), 0);
          }
        }
      }
    }
    c_.ops.push_back(std::move(op));
    return static_cast<int>(c_.ops.size()) - 1;
  }

  // Sets an element of operand `op` for both libraries.
  void Set(int op, int64_t idx, T v) {
    c_.ops[op].hwy_init[idx] = v;
    c_.ops[op].ref_init[idx] = v;
  }

  // Common shapes.
  // (In row-major layout, element (i, j) of a matrix is at i * ld + j, and
  // the band and packed layouts are those of the CBLAS rows.)
  Shape General(int rows, int cols, int ld) {
    Shape s;
    s.rows = rows;
    s.cols = cols;
    s.ld = ld;
    const bool row = row_major_;
    s.index = [ld, row](int i, int j) -> int64_t {
      return row ? static_cast<int64_t>(i) * ld + j
                 : i + static_cast<int64_t>(j) * ld;
    };
    return s;
  }
  Shape Vector(int n, int inc) {
    Shape s;
    s.rows = n;
    s.cols = 1;
    const int64_t step = std::abs(inc);
    s.index = [step](int i, int) { return i * step; };
    return s;
  }
  // The `uplo` triangle of an n x n matrix.
  Shape Triangle(int n, int ld, char uplo) {
    Shape s = General(n, n, ld);
    const auto full = s.index;
    s.index = [full, uplo](int i, int j) -> int64_t {
      if (uplo == 'U' ? i > j : i < j) return -1;
      return full(i, j);
    };
    return s;
  }
  // General band matrix with kl sub- and ku superdiagonals.
  Shape Band(int m, int n, int kl, int ku, int ld) {
    Shape s = General(m, n, ld);
    const bool row = row_major_;
    s.index = [kl, ku, ld, row](int i, int j) -> int64_t {
      if (i < j - ku || i > j + kl) return -1;
      return row ? static_cast<int64_t>(i) * ld + kl + j - i
                 : ku + i - j + static_cast<int64_t>(j) * ld;
    };
    return s;
  }
  // The `uplo` triangle of an n x n band matrix with k off-diagonals.
  Shape TriangleBand(int n, int k, int ld, char uplo) {
    Shape s = General(n, n, ld);
    const bool row = row_major_;
    s.index = [k, ld, uplo, row](int i, int j) -> int64_t {
      const bool upper = uplo == 'U';
      if (upper ? (i > j || i < j - k) : (i < j || i > j + k)) return -1;
      if (row) {
        return static_cast<int64_t>(i) * ld + (upper ? 0 : k) + j - i;
      }
      return (upper ? k : 0) + i - j + static_cast<int64_t>(j) * ld;
    };
    return s;
  }
  // The `uplo` triangle of an n x n matrix in packed storage.
  Shape Packed(int n, char uplo) {
    Shape s;
    s.rows = n;
    s.cols = n;
    const bool row = row_major_;
    s.index = [n, uplo, row](int i, int j) -> int64_t {
      const bool upper = uplo == 'U';
      if (upper ? i > j : i < j) return -1;
      // Packed by rows is packed by columns with the other triangle of the
      // transpose.
      const int64_t a = row ? j : i, b = row ? i : j;
      if (upper != row) return a + b * (b + 1) / 2;
      return a + b * (2 * n - b - 1) / 2;
    };
    return s;
  }

  // A scalar result (written by `call`).
  int Result(const char* name, bool exact = false) {
    Operand<T> op;
    op.name = name;
    op.output = true;
    op.exact = exact;
    op.guard_after = Uniform(0, 1) != 0;
    op.misalign = kIsComplex<T> && Uniform(0, 3) == 0;
    op.use.assign(1, Use::kUsed);
    op.hwy_init.assign(1, Poison<T>());
    op.ref_init.assign(1, Poison<T>());
    c_.ops.push_back(std::move(op));
    return static_cast<int>(c_.ops.size()) - 1;
  }

  // The routine must ignore the values of operand `i` (e.g. C if beta = 0):
  // NaN for hwyBLAS, random values for the reference.
  void Ignore(int i) {
    Operand<T>& op = c_.ops[i];
    for (size_t k = 0; k < op.use.size(); ++k) {
      if (op.use[k] != Use::kUnused) op.hwy_init[k] = Poison<T>();
    }
  }

 private:
  Case<T>& c_;
  const Options& opt_;
  Rng rng_;
  bool huge_ = false;  // a huge dimension was chosen
  bool row_major_ = false;
};

// ---------------------------------------------------------------------------
// Routines. Each generator fills a case from a seed.

template <typename T>
using Generator = std::function<void(Case<T>&, Builder<T>&)>;

template <typename T>
struct Routine {
  std::string name;
  Generator<T> generate;
};

template <typename T>
char Prefix() {
  if constexpr (std::is_same_v<T, float>) return 's';
  if constexpr (std::is_same_v<T, double>) return 'd';
  if constexpr (std::is_same_v<T, std::complex<float>>) return 'c';
  return 'z';
}

// Prefix of the real type: s or d.
template <typename T>
char RealPrefix() {
  return std::is_same_v<Real<T>, float> ? 's' : 'd';
}

bool IsZero(double v) { return v == 0; }
template <typename R>
bool IsZero(std::complex<R> v) {
  return v == std::complex<R>(0);
}

// Checks the results of ?rotmg: with H given by `param`,
//   H (x1, y1)^T = (z1, 0)^T  and  H^T diag(e1, e2) H = diag(d1, d2),
// relative to the size of the terms.
template <typename R>
std::string CheckRotmg(R d1, R d2, R x1, R y1, R e1, R e2, R z1,
                       const R* param) {
  const R flag = param[0];
  if (flag != -2 && flag != -1 && flag != 0 && flag != 1) {
    return "flag " + ToString(flag);
  }
  // Elements of param that the flag leaves unset are unchanged (7).
  const bool unset[5] = {false, flag == -2 || flag == 0,
                         flag == -2 || flag == 1, flag == -2 || flag == 1,
                         flag == -2 || flag == 0};
  for (int i = 1; i < 5; ++i) {
    if (unset[i] && param[i] != 7) {
      return "param[" + std::to_string(i) + "] was set for flag " +
             ToString(flag);
    }
  }
  // Skip inputs without a rotation, with strong cancellation (d2 < 0), or
  // whose products underflow.
  const double a = d1, b = d2, x = x1, y = y1;
  const double q1 = a * x * x, q2 = std::abs(b * y * y);
  if (d1 < 0 || (d2 < 0 && q2 > 0.5 * q1)) return "";
  const double tiny = std::ldexp(1.0, sizeof(R) == 4 ? -90 : -900);
  for (double v : {a * x, b * y, q1, q2}) {
    if (v != 0 && std::abs(v) < tiny) return "";
  }
  // The elements of H before rescaling are these ratios; one in the
  // subnormal range has lost precision.
  for (double v : {y / x, x / y, (b * y) / (a * x), (a * x) / (b * y)}) {
    if (v != 0 && std::abs(v) < std::numeric_limits<R>::min()) return "";
  }
  if (flag == -2) {
    return b * y == 0 ? "" : "flag -2, but d2 * y1 is not 0";
  }
  const double h11 = flag == 0 ? 1 : param[1];
  const double h21 = flag == 1 ? -1 : param[2];
  const double h12 = flag == 1 ? 1 : param[3];
  const double h22 = flag == 0 ? 1 : param[4];
  if (e1 == 0 && e2 == 0 && z1 == 0 && h11 == 0 && h12 == 0) return "";
  // An element of H in the subnormal range has lost precision.
  for (double v : {h11, h12, h21, h22}) {
    if (v != 0 && std::abs(v) < std::numeric_limits<R>::min()) return "";
  }
  const double tol = 100 * std::numeric_limits<R>::epsilon();
  const auto differs = [tol](double p, double q, double want) {
    const double size = std::abs(p) + std::abs(q) + std::abs(want);
    return !(std::abs(p + q - want) <= tol * size);
  };
  const double f1 = e1, f2 = e2;
  if (differs(h11 * x, h12 * y, z1)) return "H x is not the new x1";
  if (differs(h21 * x, h22 * y, 0)) return "H does not annihilate y1";
  if (differs(f1 * h11 * h11, f2 * h21 * h21, a) ||
      differs(f1 * h11 * h12, f2 * h21 * h22, 0) ||
      differs(f1 * h12 * h12, f2 * h22 * h22, b)) {
    return "H^T D' H differs from D";
  }
  return "";
}

// Error length of sdsdot and dsdot: hwyBLAS accumulates in double, but a
// reference may not for all of its sum (OpenBLAS 0.3.34's x86 kernel loses
// precision for long vectors).
int DoubleDotErrorDim(int n) { return 1 + n / 1000; }

template <typename T>
std::vector<Routine<T>> Routines() {
  using R = Real<T>;
  const std::string p(1, Prefix<T>());
  const std::string rp(1, RealPrefix<T>());
  static constexpr bool cplx = kIsComplex<T>;
  std::vector<Routine<T>> list;
  const auto add = [&list](std::string name, Generator<T> gen) {
    list.push_back({std::move(name), std::move(gen)});
  };
  const char* const kTrans = "NTC";
  const char* const kUplo = "UL";

  // ----- Level 1
  add(p + "axpy", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const T alpha = b.Scalar("alpha");
    const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int y = b.Add("y", b.Vector(n, incy), true);
    if (IsZero(alpha)) b.Ignore(x);
    c.error_dim = incy == 0 ? n : 1;  // incy = 0: n updates of one element
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &n, &alpha, o[x], &incx, o[y], &incy);
    };
  });
  add(p + "axpby", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int y = b.Add("y", b.Vector(n, incy), true);
    if (IsZero(alpha)) b.Ignore(x);
    if (IsZero(beta)) b.Ignore(y);
    c.error_dim = 2;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &n, &alpha, o[x], &incx, &beta, o[y], &incy);
    };
  });
  add(p + "scal", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const T alpha = b.Scalar("alpha");
    const int incx = b.ForwardInc("incx");
    const int x = b.Add("x", b.Vector(n, incx), true);
    c.error_dim = 1;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &n, &alpha, o[x], &incx);
    };
  });
  if constexpr (cplx) {
    // csscal, zdscal: real alpha.
    add(p + rp + "scal", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const R alpha = b.RealScalar("alpha");
      const int incx = b.ForwardInc("incx");
      const int x = b.Add("x", b.Vector(n, incx), true);
      c.error_dim = 1;
      c.call = [=](const Target& t, T* const* o) {
        Blas(t, &n, &alpha, o[x], &incx);
      };
    });
  }
  add(p + "copy", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int y = b.Add("y", b.Vector(n, incy), true);
    b.Ignore(y);
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &n, o[x], &incx, o[y], &incy);
    };
  });
  add(p + "swap", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
    const int x = b.Add("x", b.Vector(n, incx), true);
    const int y = b.Add("y", b.Vector(n, incy), true);
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &n, o[x], &incx, o[y], &incy);
    };
  });
  // Dot products: real ones return the result, complex ones use the CBLAS
  // *_sub functions (the Fortran ABI for complex return values varies).
  const auto add_dot = [&](std::string name, bool conj) {
    add(std::move(name), [conj](Case<T>& c, Builder<T>& b) {
      (void)conj;  // only used for complex types
      const int n = b.Dim("n", 1);
      const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int r = b.Result("result");
      c.error_dim = n;
      if constexpr (cplx) {
        b.UseApi(Api::kCblasCol);
        c.symbol = std::string("cblas_") + Prefix<T>() +
                   (conj ? "dotc_sub" : "dotu_sub");
        c.call = [=](const Target& t, T* const* o) {
          Blas(t, &n, o[x], &incx, o[y], &incy, o[r]);
        };
      } else {
        c.call = [=](const Target& t, T* const* o) {
          *o[r] = Blas<T>(t, &n, o[x], &incx, o[y], &incy);
        };
      }
    });
  };
  if constexpr (cplx) {
    add_dot(p + "dotu", false);
    add_dot(p + "dotc", true);
  } else {
    add_dot(p + "dot", false);
  }
  if constexpr (std::is_same_v<T, float>) {
    add("sdsdot", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const float sb = static_cast<float>(b.Scalar("sb"));
      const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int r = b.Result("result");
      c.error_dim = DoubleDotErrorDim(n);
      c.call = [=](const Target& t, T* const* o) {
        *o[r] = Blas<float>(t, &n, &sb, o[x], &incx, o[y], &incy);
      };
    });
    add("dsdot", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int r = b.Result("result");
      c.error_dim = DoubleDotErrorDim(n);
      c.call = [=](const Target& t, T* const* o) {
        *o[r] =
            static_cast<float>(Blas<double>(t, &n, o[x], &incx, o[y], &incy));
      };
    });
  }
  // nrm2, asum, iamax: incx > 0 (negative increments are no-ops or
  // differ between libraries, and are tested by the unit tests).
  const std::string rprefix = cplx ? rp + p : p;  // scnrm2, dznrm2, ...
  for (const char* what : {"nrm2", "asum"}) {
    const bool asum = what[0] == 'a';
    add(rprefix + what, [asum](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const int incx = b.ForwardInc("incx");
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int r = b.Result("result");
      // With increments, hwyBLAS adds the squares one by one, and the
      // rounding errors grow slowly with n.
      c.error_dim = asum || incx == 0 ? n : 2 + n / 1000;
      c.call = [=](const Target& t, T* const* o) {
        *o[r] = T(Blas<R>(t, &n, o[x], &incx));
      };
    });
  }
  add(std::string("i") + p + "amax", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.ForwardInc("incx");
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int r = b.Result("index", true);
    c.call = [=](const Target& t, T* const* o) {
      // (CBLAS returns a size_t, 0-based.)
      const int index =
          t.api == Api::kFortran
              ? Blas<int>(t, &n, o[x], &incx)
              : static_cast<int>(Blas<size_t>(t, &n, o[x], &incx));
      *o[r] = T(static_cast<R>(index));
    };
  });
  // rot with real c and s: srot, drot, csrot, zdrot.
  add((cplx ? p + rp : p) + "rot", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
    const R angle = static_cast<R>(3.2 * b.Unit());
    const R cs = std::cos(angle), sn = std::sin(angle);
    b.Param("c", cs);
    b.Param("s", sn);
    const int x = b.Add("x", b.Vector(n, incx), true);
    const int y = b.Add("y", b.Vector(n, incy), true);
    c.error_dim = incx == 0 || incy == 0 ? 4 * n + 2 : 2;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &n, o[x], &incx, o[y], &incy, &cs, &sn);
    };
  });
  if constexpr (!cplx) {
    add(p + "rotm", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const int incx = b.Inc("incx", true), incy = b.Inc("incy", true);
      const int flag = b.Uniform(-2, 1);
      b.Param("flag", flag);
      const int x = b.Add("x", b.Vector(n, incx), true);
      const int y = b.Add("y", b.Vector(n, incy), true);
      // param = {flag, h11, h21, h12, h22}; unused entries are NaN.
      typename Builder<T>::Shape s = b.Vector(5, 1);
      s.index = [flag](int i, int) -> int64_t {
        if (i == 0) return 0;
        if (flag == -2) return -1;
        if (flag == 0 && (i == 1 || i == 4)) return -1;
        if (flag == 1 && (i == 2 || i == 3)) return -1;
        return i;
      };
      const int param = b.Add("param", s, false);
      // param must have all 5 elements even if the last ones are unused.
      Operand<T>& op = c.ops[param];
      op.use.resize(5, Use::kUnused);
      op.hwy_init.resize(5, Poison<T>());
      op.ref_init.resize(5, T(0));
      op.hwy_init[0] = op.ref_init[0] = static_cast<T>(flag);
      // (An increment of 0 repeats the rotation of one element.)
      c.error_dim = incx == 0 || incy == 0 ? 4 * n + 2 : 2;
      c.call = [=](const Target& t, T* const* o) {
        Blas(t, &n, o[x], &incx, o[y], &incy, o[param]);
      };
    });
  }

  // ----- Level 2
  add(p + "gemv", [kTrans](Case<T>& c, Builder<T>& b) {
    const Flag trans = b.Letter("trans", kTrans);
    const int m = b.Dim("m", 2), n = b.Dim("n", 2);
    if (b.HangsInNetlib(trans, m)) return;
    const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
    const int lda = b.Ld("lda", b.Major(m, n));
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int lx = trans == 'N' ? n : m, ly = trans == 'N' ? m : n;
    const int a = b.Add("A", b.General(m, n, lda), false);
    const int x = b.Add("x", b.Vector(lx, incx), false);
    const int y = b.Add("y", b.Vector(ly, incy), true);
    if (IsZero(alpha)) b.Ignore(a), b.Ignore(x);
    // (BLAS returns at once, without scaling y, if m or n is 0.)
    if (IsZero(beta) && m > 0 && n > 0) b.Ignore(y);
    c.error_dim = lx + 1;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, Layout{}, &trans, &m, &n, &alpha, o[a], &lda, o[x], &incx, &beta,
           o[y], &incy);
    };
  });
  add(p + "gbmv", [kTrans](Case<T>& c, Builder<T>& b) {
    const Flag trans = b.Letter("trans", kTrans);
    const int m = b.Dim("m", 2), n = b.Dim("n", 2);
    if (b.HangsInNetlib(trans, m)) return;
    const int kl = b.Bandwidth("kl", m), ku = b.Bandwidth("ku", n);
    const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
    const int lda = b.Ld("lda", kl + ku + 1);
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int lx = trans == 'N' ? n : m, ly = trans == 'N' ? m : n;
    const int a = b.Add("A", b.Band(m, n, kl, ku, lda), false);
    const int x = b.Add("x", b.Vector(lx, incx), false);
    const int y = b.Add("y", b.Vector(ly, incy), true);
    if (IsZero(alpha)) b.Ignore(a), b.Ignore(x);
    // (BLAS returns at once, without scaling y, if m or n is 0.)
    if (IsZero(beta) && m > 0 && n > 0) b.Ignore(y);
    c.error_dim = kl + ku + 2;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, Layout{}, &trans, &m, &n, &kl, &ku, &alpha, o[a], &lda, o[x],
           &incx, &beta, o[y], &incy);
    };
  });
  // symv/hemv, sbmv/hbmv, spmv/hpmv: kind 0 full, 1 band, 2 packed.
  for (int kind = 0; kind < 3; ++kind) {
    static const char* const kNames[2][3] = {{"symv", "sbmv", "spmv"},
                                             {"hemv", "hbmv", "hpmv"}};
    add(p + kNames[cplx][kind], [kind, kUplo](Case<T>& c, Builder<T>& b) {
      const Flag uplo = b.Letter("uplo", kUplo);
      const int n = b.Dim("n", 2);
      const int k = kind == 1 ? b.Bandwidth("k", n) : 0;
      const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
      const int lda = kind == 0   ? b.Ld("lda", n)
                      : kind == 1 ? b.Ld("lda", k + 1)
                                  : 0;
      const int incx = b.Inc("incx"), incy = b.Inc("incy");
      typename Builder<T>::Shape s = kind == 0 ? b.Triangle(n, lda, uplo)
                                     : kind == 1
                                         ? b.TriangleBand(n, k, lda, uplo)
                                         : b.Packed(n, uplo);
      s.hermitian = true;
      const int a = b.Add("A", s, false);
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), true);
      if (IsZero(alpha)) b.Ignore(a), b.Ignore(x);
      if (IsZero(beta)) b.Ignore(y);
      c.error_dim = (kind == 1 ? 2 * k : n) + 2;
      c.call = [=](const Target& t, T* const* o) {
        if (kind == 0) {
          Blas(t, Layout{}, &uplo, &n, &alpha, o[a], &lda, o[x], &incx, &beta,
               o[y], &incy);
        } else if (kind == 1) {
          Blas(t, Layout{}, &uplo, &n, &k, &alpha, o[a], &lda, o[x], &incx,
               &beta, o[y], &incy);
        } else {
          Blas(t, Layout{}, &uplo, &n, &alpha, o[a], o[x], &incx, &beta, o[y],
               &incy);
        }
      };
    });
  }
  // trmv, tbmv, tpmv, trsv, tbsv, tpsv.
  for (int solve = 0; solve < 2; ++solve) {
    for (int kind = 0; kind < 3; ++kind) {
      static const char* const kNames[2][3] = {{"trmv", "tbmv", "tpmv"},
                                               {"trsv", "tbsv", "tpsv"}};
      add(p + kNames[solve][kind],
          [solve, kind, kTrans, kUplo](Case<T>& c, Builder<T>& b) {
            const Flag uplo = b.Letter("uplo", kUplo);
            const Flag trans = b.Letter("trans", kTrans);
            const Flag diag = b.Letter("diag", "NU");
            const int n = b.Dim("n", 2);
            const int k = kind == 1 ? b.Bandwidth("k", n) : 0;
            const int lda = kind == 0   ? b.Ld("lda", n)
                            : kind == 1 ? b.Ld("lda", k + 1)
                                        : 0;
            const int incx = b.Inc("incx");
            typename Builder<T>::Shape s = kind == 0 ? b.Triangle(n, lda, uplo)
                                           : kind == 1
                                               ? b.TriangleBand(n, k, lda, uplo)
                                               : b.Packed(n, uplo);
            s.unit_diag = diag == 'U';
            s.solve = solve != 0;
            const int a = b.Add("A", s, false);
            const int x = b.Add("x", b.Vector(n, incx), true);
            c.error_dim = (kind == 1 ? k : n) + 2;
            c.call = [=](const Target& t, T* const* o) {
              if (kind == 0) {
                Blas(t, Layout{}, &uplo, &trans, &diag, &n, o[a], &lda, o[x],
                     &incx);
              } else if (kind == 1) {
                Blas(t, Layout{}, &uplo, &trans, &diag, &n, &k, o[a], &lda,
                     o[x], &incx);
              } else {
                Blas(t, Layout{}, &uplo, &trans, &diag, &n, o[a], o[x], &incx);
              }
            };
          });
    }
  }
  // ger (geru, gerc).
  for (int conj = 0; conj < (cplx ? 2 : 1); ++conj) {
    add(p + (cplx ? (conj ? "gerc" : "geru") : "ger"), [](Case<T>& c,
                                                          Builder<T>& b) {
      const int m = b.Dim("m", 2), n = b.Dim("n", 2);
      const T alpha = b.Scalar("alpha");
      const int incx = b.Inc("incx"), incy = b.Inc("incy");
      const int lda = b.Ld("lda", b.Major(m, n));
      const int x = b.Add("x", b.Vector(m, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int a = b.Add("A", b.General(m, n, lda), true);
      if (IsZero(alpha)) b.Ignore(x), b.Ignore(y);
      c.error_dim = 2;
      c.call = [=](const Target& t, T* const* o) {
        Blas(t, Layout{}, &m, &n, &alpha, o[x], &incx, o[y], &incy, o[a], &lda);
      };
    });
  }
  // syr/her, spr/hpr, syr2/her2, spr2/hpr2.
  for (int two = 0; two < 2; ++two) {
    for (int packed = 0; packed < 2; ++packed) {
      std::string name = p + (cplx ? "h" : "s") + (packed ? "pr" : "yr");
      if (cplx && !packed) name = p + "her";
      if (two) name += "2";
      add(name, [two, packed, kUplo](Case<T>& c, Builder<T>& b) {
        const Flag uplo = b.Letter("uplo", kUplo);
        const int n = b.Dim("n", 2);
        // her and hpr take a real alpha.
        const bool real_alpha = cplx && !two;
        T alpha;
        R ralpha = 0;
        if (real_alpha) {
          ralpha = b.RealScalar("alpha");
          alpha = T(ralpha);
        } else {
          alpha = b.Scalar("alpha");
        }
        const int incx = b.Inc("incx");
        const int incy = two ? b.Inc("incy") : 0;
        const int lda = packed ? 0 : b.Ld("lda", n);
        const int x = b.Add("x", b.Vector(n, incx), false);
        const int y = two ? b.Add("y", b.Vector(n, incy), false) : -1;
        typename Builder<T>::Shape s =
            packed ? b.Packed(n, uplo) : b.Triangle(n, lda, uplo);
        s.hermitian = true;
        const int a = b.Add("A", s, true);
        if (IsZero(alpha)) {
          b.Ignore(x);
          if (two) b.Ignore(y);
        }
        c.error_dim = 3;
        c.call = [=](const Target& t, T* const* o) {
          const auto call = [&](const auto* al) {
            if (two && packed) {
              Blas(t, Layout{}, &uplo, &n, al, o[x], &incx, o[y], &incy, o[a]);
            } else if (two) {
              Blas(t, Layout{}, &uplo, &n, al, o[x], &incx, o[y], &incy, o[a],
                   &lda);
            } else if (packed) {
              Blas(t, Layout{}, &uplo, &n, al, o[x], &incx, o[a]);
            } else {
              Blas(t, Layout{}, &uplo, &n, al, o[x], &incx, o[a], &lda);
            }
          };
          if (real_alpha) {
            call(&ralpha);
          } else {
            call(&alpha);
          }
        };
      });
    }
  }

  // ----- Level 3
  // gemm and gemmt (one triangle of C).
  // (gemmtr is the name of gemmt in the reference BLAS.)
  for (int variant = 0; variant < 3; ++variant) {
    const int tri = variant != 0;
    add(p + (variant == 0   ? "gemm"
             : variant == 1 ? "gemmt"
                            : "gemmtr"),
        [tri, kTrans, kUplo](Case<T>& c, Builder<T>& b) {
          const Flag uplo = tri ? b.Letter("uplo", kUplo) : Flag{'N', 0};
          const Flag ta = b.Letter("transa", kTrans);
          const Flag tb = b.Letter("transb", kTrans);
          const int m = b.Dim("m", 3);
          const int n = tri ? m : b.Dim("n", 3);
          const int k = b.Dim("k", 3);
          const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
          const int ar = ta == 'N' ? m : k, ac = ta == 'N' ? k : m;
          const int br = tb == 'N' ? k : n, bc = tb == 'N' ? n : k;
          const int lda = b.Ld("lda", b.Major(ar, ac));
          const int ldb = b.Ld("ldb", b.Major(br, bc));
          const int ldc = b.Ld("ldc", b.Major(m, n));
          const int a = b.Add("A", b.General(ar, ac, lda), false);
          const int bb = b.Add("B", b.General(br, bc, ldb), false);
          const int cc = b.Add(
              "C", tri ? b.Triangle(n, ldc, uplo) : b.General(m, n, ldc), true);
          if (IsZero(alpha)) b.Ignore(a), b.Ignore(bb);
          if (IsZero(beta)) b.Ignore(cc);
          c.error_dim = k + 1;
          c.call = [=](const Target& t, T* const* o) {
            if (tri) {
              Blas(t, Layout{}, &uplo, &ta, &tb, &n, &k, &alpha, o[a], &lda,
                   o[bb], &ldb, &beta, o[cc], &ldc);
            } else {
              Blas(t, Layout{}, &ta, &tb, &m, &n, &k, &alpha, o[a], &lda, o[bb],
                   &ldb, &beta, o[cc], &ldc);
            }
          };
        });
  }
  // symm (and hemm).
  for (int herm = 0; herm < (cplx ? 2 : 1); ++herm) {
    add(p + (herm ? "hemm" : "symm"), [herm, kUplo](Case<T>& c, Builder<T>& b) {
      const Flag side = b.Letter("side", "LR");
      const Flag uplo = b.Letter("uplo", kUplo);
      const int m = b.Dim("m", 3), n = b.Dim("n", 3);
      const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
      const int ka = side == 'L' ? m : n;
      const int lda = b.Ld("lda", ka), ldb = b.Ld("ldb", b.Major(m, n)),
                ldc = b.Ld("ldc", b.Major(m, n));
      typename Builder<T>::Shape s = b.Triangle(ka, lda, uplo);
      s.hermitian = herm != 0;
      const int a = b.Add("A", s, false);
      const int bb = b.Add("B", b.General(m, n, ldb), false);
      const int cc = b.Add("C", b.General(m, n, ldc), true);
      if (IsZero(alpha)) b.Ignore(a), b.Ignore(bb);
      if (IsZero(beta)) b.Ignore(cc);
      c.error_dim = ka + 1;
      c.call = [=](const Target& t, T* const* o) {
        Blas(t, Layout{}, &side, &uplo, &m, &n, &alpha, o[a], &lda, o[bb], &ldb,
             &beta, o[cc], &ldc);
      };
    });
  }
  // syrk, syr2k (and herk, her2k: real beta, herk also a real alpha).
  for (int herm = 0; herm < (cplx ? 2 : 1); ++herm) {
    for (int two = 0; two < 2; ++two) {
      std::string name = p + (herm ? "her" : "syr") + (two ? "2k" : "k");
      add(name, [herm, two, kUplo](Case<T>& c, Builder<T>& b) {
        const Flag uplo = b.Letter("uplo", kUplo);
        const Flag trans =
            b.Letter("trans", herm ? "NC" : (cplx ? "NT" : "NTC"));
        const int n = b.Dim("n", 3), k = b.Dim("k", 3);
        T alpha, beta;
        R ralpha = 0, rbeta = 0;
        if (herm && !two) {
          ralpha = b.RealScalar("alpha");
          alpha = T(ralpha);
        } else {
          alpha = b.Scalar("alpha");
        }
        if (herm) {
          rbeta = b.RealScalar("beta", true);
          beta = T(rbeta);
        } else {
          beta = b.Scalar("beta", true);
        }
        const int ar = trans == 'N' ? n : k, ac = trans == 'N' ? k : n;
        const int lda = b.Ld("lda", b.Major(ar, ac));
        const int ldb = two ? b.Ld("ldb", b.Major(ar, ac)) : 0;
        const int ldc = b.Ld("ldc", n);
        const int a = b.Add("A", b.General(ar, ac, lda), false);
        const int bb = two ? b.Add("B", b.General(ar, ac, ldb), false) : -1;
        typename Builder<T>::Shape s = b.Triangle(n, ldc, uplo);
        s.hermitian = herm != 0;
        const int cc = b.Add("C", s, true);
        if (IsZero(alpha)) {
          b.Ignore(a);
          if (two) b.Ignore(bb);
        }
        if (IsZero(beta)) b.Ignore(cc);
        c.error_dim = 2 * k + 1;
        c.call = [=](const Target& t, T* const* o) {
          const auto call = [&](const auto* al, const auto* be) {
            if (two) {
              Blas(t, Layout{}, &uplo, &trans, &n, &k, al, o[a], &lda, o[bb],
                   &ldb, be, o[cc], &ldc);
            } else {
              Blas(t, Layout{}, &uplo, &trans, &n, &k, al, o[a], &lda, be,
                   o[cc], &ldc);
            }
          };
          if (herm && two) {
            call(&alpha, &rbeta);
          } else if (herm) {
            call(&ralpha, &rbeta);
          } else {
            call(&alpha, &beta);
          }
        };
      });
    }
  }
  // trmm, trsm.
  for (int solve = 0; solve < 2; ++solve) {
    add(p + (solve ? "trsm" : "trmm"),
        [solve, kTrans, kUplo](Case<T>& c, Builder<T>& b) {
          const Flag side = b.Letter("side", "LR");
          const Flag uplo = b.Letter("uplo", kUplo);
          const Flag transa = b.Letter("transa", kTrans);
          const Flag diag = b.Letter("diag", "NU");
          const int m = b.Dim("m", 3), n = b.Dim("n", 3);
          const T alpha = b.Scalar("alpha");
          const int ka = side == 'L' ? m : n;
          const int lda = b.Ld("lda", ka);
          const int ldb = b.Ld("ldb", b.Major(m, n));
          typename Builder<T>::Shape s = b.Triangle(ka, lda, uplo);
          s.unit_diag = diag == 'U';
          s.solve = solve != 0;
          const int a = b.Add("A", s, false);
          const int bb = b.Add("B", b.General(m, n, ldb), true);
          if (IsZero(alpha)) b.Ignore(a), b.Ignore(bb);
          c.error_dim = ka + 1;
          c.call = [=](const Target& t, T* const* o) {
            Blas(t, Layout{}, &side, &uplo, &transa, &diag, &m, &n, &alpha,
                 o[a], &lda, o[bb], &ldb);
          };
        });
  }
  // omatcopy (OpenBLAS extension): B := alpha * op(A).
  add(p + "omatcopy", [](Case<T>& c, Builder<T>& b) {
    b.ColumnMajorShapes();  // `order` decides the layout
    const Flag order = b.Letter("order", "CR");
    const Flag trans = b.Letter("trans", "NTRC");
    const int rows = b.Dim("rows", 2), cols = b.Dim("cols", 2);
    const T alpha = b.Scalar("alpha");
    // In column-major terms, A is ar x ac and B is br x bc.
    const bool transpose = trans == 'T' || trans == 'C';
    const int ar = order == 'C' ? rows : cols, ac = order == 'C' ? cols : rows;
    const int br = transpose ? ac : ar, bc = transpose ? ar : ac;
    const int lda = b.Ld("lda", ar), ldb = b.Ld("ldb", br);
    const int a = b.Add("A", b.General(ar, ac, lda), false);
    const int bb = b.Add("B", b.General(br, bc, ldb), true);
    b.Ignore(bb);
    c.error_dim = 1;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &order, &trans, &rows, &cols, &alpha, o[a], &lda, o[bb], &ldb);
    };
  });

  // ----- Routines added for coverage
  // imatcopy: A := alpha * op(A) in place. Libraries differ in the details
  // (and Netlib has none), so the expected result is computed here.
  add(p + "imatcopy", [](Case<T>& c, Builder<T>& b) {
    if (!b.options().special.empty()) return;
    b.ColumnMajorShapes();
    const Flag order = b.Letter("order", "CR");
    const Flag trans = b.Letter("trans", "NTRC");
    const int rows = b.Dim("rows", 2), cols = b.Dim("cols", 2);
    const T alpha = b.Scalar("alpha");
    const bool transpose = trans == 'T' || trans == 'C';
    const bool conj = trans == 'C' || trans == 'R';
    const int ar = order == 'C' ? rows : cols, ac = order == 'C' ? cols : rows;
    const int br = transpose ? ac : ar, bc = transpose ? ar : ac;
    const int lda = b.Ld("lda", ar), ldb = b.Ld("ldb", br);
    // The array holds A (ar x ac, lda) before and B (br x bc, ldb) after.
    const int64_t a_end =
        ar > 0 && ac > 0 ? static_cast<int64_t>(ac - 1) * lda + ar : 0;
    const int64_t b_end =
        br > 0 && bc > 0 ? static_cast<int64_t>(bc - 1) * ldb + br : 0;
    const int size = static_cast<int>(std::max(a_end, b_end));
    typename Builder<T>::Shape s;
    s.rows = size;
    s.cols = 1;
    s.index = [=](int i, int) -> int64_t {
      const bool in_a = i < a_end && i % lda < ar;
      const bool in_b = i < b_end && i % ldb < br;
      return in_a || in_b ? i : -1;
    };
    const int a = b.Add("A", s, true);
    // Both start from the same values (those outside A are overwritten).
    for (int i = 0; i < size; ++i) {
      const bool in_a = i < a_end && i % lda < ar;
      const bool in_b = i < b_end && i % ldb < br;
      if (in_a || in_b) {
        c.ops[a].ref_init[i] = c.ops[a].hwy_init[i];
      } else {
        c.ops[a].ref_init[i] = c.ops[a].hwy_init[i] = Poison<T>();
      }
    }
    c.error_dim = 2;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, &order, &trans, &rows, &cols, &alpha, o[a], &lda, &ldb);
    };
    const std::vector<T> before = c.ops[a].hwy_init;
    c.expected = [=](T* const* o) {
      for (int j = 0; j < bc; ++j) {
        for (int i = 0; i < br; ++i) {
          const int64_t from = transpose ? j + static_cast<int64_t>(i) * lda
                                         : i + static_cast<int64_t>(j) * lda;
          T v{};
          if (!IsZero(alpha)) {
            v = before[from];
            if constexpr (cplx) {
              if (conj) v = std::conj(v);
            }
            v *= alpha;
          }
          o[a][i + static_cast<int64_t>(j) * ldb] = v;
        }
      }
    };
  });
  // i?amin (an extension, so the expected result is computed here).
  add(std::string("i") + p + "amin", [](Case<T>& c, Builder<T>& b) {
    if (!b.options().special.empty()) return;
    const int n = b.Dim("n", 1);
    const int incx = std::abs(b.Inc("incx"));
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int r = b.Result("index", true);
    c.call = [=](const Target& t, T* const* o) {
      const int index =
          t.api == Api::kFortran
              ? Blas<int>(t, &n, o[x], &incx)
              : static_cast<int>(Blas<size_t>(t, &n, o[x], &incx));
      *o[r] = T(static_cast<R>(index));
    };
    const bool one_based = c.api == Api::kFortran;
    c.expected = [=](T* const* o) {
      int best = 0;  // 0-based
      double smallest = 0;
      for (int i = 0; i < n; ++i) {
        const T v = o[x][static_cast<int64_t>(i) * incx];
        double abs = std::abs(static_cast<double>(std::real(v)));
        if constexpr (cplx) {
          abs += std::abs(static_cast<double>(std::imag(v)));
        }
        if (i == 0 || abs < smallest) {
          smallest = abs;
          best = i;
        }
      }
      *o[r] = T(static_cast<R>(n > 0 && one_based ? best + 1 : best));
    };
  });
  if constexpr (cplx) {
    // scabs1, dcabs1 (Fortran only).
    add(rp + "cabs1", [](Case<T>& c, Builder<T>& b) {
      b.UseApi(Api::kFortran);
      const int x = b.Add("z", b.Vector(1, 1), false);
      b.Set(x, 0, b.Wide());
      const int r = b.Result("result");
      c.error_dim = 1;
      c.ops[r].relative = true;
      c.call = [=](const Target& t, T* const* o) {
        *o[r] = T(Blas<R>(t, o[x]));
      };
    });
  }
  // rotg: constructs a plane rotation from a and b.
  add(p + "rotg", [](Case<T>& c, Builder<T>& b) {
    const int x = b.Add("a", b.Vector(1, 1), true);
    const int y = b.Add("b", b.Vector(1, 1), !cplx);
    // For complex types, values of very different sizes only against Netlib:
    // OpenBLAS 0.3.34 is inaccurate for them.
    if (!cplx || b.options().netlib) {
      b.Set(x, 0, b.Wide());
      b.Set(y, 0, b.Wide());
    }
    const int cs = b.Result("c"), sn = b.Result("s");
    // For complex types, c is real.
    if (cplx) c.ops[cs].use[0] = Use::kRealOnly;
    for (int op : {x, y, cs, sn}) c.ops[op].relative = true;
    c.error_dim = 4;
    c.call = [=](const Target& t, T* const* o) {
      Blas(t, o[x], o[y], o[cs], o[sn]);
    };
  });
  if constexpr (!cplx) {
    // rotmg: constructs a modified plane rotation.
    add(p + "rotmg", [](Case<T>& c, Builder<T>& b) {
      const int d1 = b.Add("d1", b.Vector(1, 1), true);
      const int d2 = b.Add("d2", b.Vector(1, 1), true);
      const int x1 = b.Add("x1", b.Vector(1, 1), true);
      const int param = b.Add("param", b.Vector(5, 1), true);
      b.Set(d1, 0, std::abs(b.Wide()) * (b.Uniform(0, 9) == 0 ? -1 : 1));
      b.Set(d2, 0, b.Wide());
      b.Set(x1, 0, b.Wide());
      // Elements of param that a flag leaves unset keep this value.
      for (int i = 0; i < 5; ++i) b.Set(param, i, T(7));
      const T y1 = b.Wide();
      b.Param("y1", y1);
      const T in1 = c.ops[d1].hwy_init[0], in2 = c.ops[d2].hwy_init[0];
      const T in_x = c.ops[x1].hwy_init[0];
      // The reference BLAS (3.12.1) rescales H wrongly, so the results are
      // checked against the defining equations instead.
      c.verify = [=](const std::vector<std::vector<T>>& out) {
        std::string problem =
            CheckRotmg(in1, in2, in_x, y1, out[d1][0], out[d2][0], out[x1][0],
                       out[param].data());
        if (!problem.empty()) {
          problem += ": d1=" + ToString(in1) + " d2=" + ToString(in2) +
                     " x1=" + ToString(in_x) +
                     " -> d1=" + ToString(out[d1][0]) +
                     " d2=" + ToString(out[d2][0]) +
                     " x1=" + ToString(out[x1][0]) + " param";
          for (const T& v : out[param]) problem += ' ' + ToString(v);
        }
        return problem;
      };
      c.call = [=](const Target& t, T* const* o) {
        Blas(t, o[d1], o[d2], o[x1], &y1, o[param]);
      };
    });
  }
  return list;
}

// ---------------------------------------------------------------------------
// Running and checking cases

// The current case and library, printed if it crashes.
thread_local std::string g_current_case;
thread_local const char* g_current_library = "";

void PrintCrash(const char* what) {
  fprintf(stderr, "\nCRASH (%s) in %s, case: %s\n", what, g_current_library,
          g_current_case.c_str());
  fflush(stderr);
}

#if defined(_WIN32)
LONG WINAPI CrashHandler(EXCEPTION_POINTERS* info) {
  char what[64];
  snprintf(what, sizeof(what), "exception 0x%08lx",
           static_cast<unsigned long>(info->ExceptionRecord->ExceptionCode));
  PrintCrash(what);
  return EXCEPTION_EXECUTE_HANDLER;
}
void InstallCrashHandler() { SetUnhandledExceptionFilter(CrashHandler); }
#else
void CrashHandler(int sig) {
  PrintCrash(sig == SIGSEGV ? "SIGSEGV" : "signal");
  _exit(2);
}
void InstallCrashHandler() {
  signal(SIGSEGV, CrashHandler);
  signal(SIGBUS, CrashHandler);
  signal(SIGFPE, CrashHandler);
}
#endif

// The size of a value; 0 for NaN and infinity, which must not widen the
// tolerance.
template <typename T>
double Magnitude(const T& v) {
  const double m = static_cast<double>(std::abs(v));
  return std::isfinite(m) ? m : 0;
}

template <typename T>
const char* Category(const T& v) {
  if (IsNan(v)) return "NaN";
  return std::isfinite(static_cast<double>(std::abs(v))) ? "finite" : "Inf";
}

// Returns the first problem with the results, or an empty string.
template <typename T>
std::string Check(const Case<T>& c, const std::vector<std::vector<T>>& hwy,
                  const std::vector<std::vector<T>>& ref,
                  const std::string& ref_name) {
  using R = Real<T>;
  const double eps = std::numeric_limits<R>::epsilon();
  for (size_t k = 0; k < c.ops.size(); ++k) {
    const Operand<T>& op = c.ops[k];
    const auto where = [&](size_t i) {
      std::string s = op.name + "[" + std::to_string(i) + "]";
      if (op.ld > 0) {
        s += " (row " + std::to_string(i % op.ld) + ", column " +
             std::to_string(i / op.ld) + ")";
      }
      return s;
    };
    double scale = 1;
    if (op.output) {
      for (size_t i = 0; i < op.use.size(); ++i) {
        if (op.use[i] == Use::kUnused) continue;
        scale =
            std::max({scale, Magnitude(ref[k][i]), Magnitude(op.ref_init[i])});
      }
    }
    const double factor = 16 * eps * (c.error_dim + 1);
    for (size_t i = 0; i < op.use.size(); ++i) {
      const T h = hwy[k][i], r = ref[k][i], init = op.hwy_init[i];
      if (!op.output || op.use[i] == Use::kUnused) {
        if (!SameBits(h, init)) {
          return where(i) +
                 (op.output ? " must not be referenced" : " is an input") +
                 " but changed from " + ToString(init) + " to " + ToString(h);
        }
        continue;
      }
      T hc = h, rc = r;
      if constexpr (kIsComplex<T>) {
        if (op.use[i] == Use::kRealOnly) {
          // The imaginary part is either set to zero or left unchanged.
          if (h.imag() != 0 && !SameBits(h.imag(), init.imag())) {
            return where(i) + ": the imaginary part of a Hermitian diagonal " +
                   "element is " + ToString(h.imag());
          }
          hc = T(h.real(), 0);
          rc = T(r.real(), 0);
        }
      }
      if (IsNan(hc) || IsNan(rc)) {
        if (IsNan(hc) != IsNan(rc)) {
          return where(i) + ": hwyBLAS " + ToString(h) + ", " + ref_name + " " +
                 ToString(r) + " [" + Category(hc) + " vs " + Category(rc) +
                 "]";
        }
        continue;
      }
      if (hc == rc) continue;  // also equal infinities
      const double diff = static_cast<double>(std::abs(hc - rc));
      const double tol =
          factor * (op.relative ? static_cast<double>(std::abs(rc)) : scale);
      if (op.exact || !(diff <= tol) || std::isinf(diff)) {
        return where(i) + ": hwyBLAS " + ToString(h) + ", " + ref_name + " " +
               ToString(r) + " (difference " + ToString(diff) + ", tolerance " +
               ToString(tol) + ") [" + Category(hc) + " vs " + Category(rc) +
               "]";
      }
    }
  }
  return "";
}

template <typename T>
std::string RunCase(const Case<T>& c, const Lib& hwy, const Lib& ref,
                    const Options& opt) {
  // Runs the case with library `lib` on copies of the operands (`init` of
  // each), and returns the operands afterwards. hwyBLAS gets guarded
  // copies; the reference gets padded ones, because reference libraries
  // may access a little beyond the operands (e.g. OpenBLAS 0.3.34 cscal
  // with alpha = 0).
  const auto run = [&c](const Lib& lib, std::vector<T> Operand<T>::* init,
                        bool guarded, int* exceptions) {
    constexpr size_t kPadding = 64;  // elements before and after
    const size_t num = c.ops.size();
    std::vector<std::unique_ptr<GuardedBuffer>> guarded_buffers;
    std::vector<std::vector<T>> padded_buffers(num);
    std::vector<T*> ptrs(num);
    for (size_t k = 0; k < num; ++k) {
      const std::vector<T>& values = c.ops[k].*init;
      if (guarded) {
        // A misaligned operand starts half an element later; the buffer is
        // that much larger, so one end still touches the inaccessible page.
        const size_t shift = c.ops[k].misalign ? sizeof(Real<T>) : 0;
        guarded_buffers.push_back(std::make_unique<GuardedBuffer>(
            values.size() * sizeof(T) + shift, c.ops[k].guard_after));
        char* data = static_cast<char*>(guarded_buffers.back()->data());
        if (!c.ops[k].guard_after) data += shift;
        ptrs[k] = reinterpret_cast<T*>(data);
      } else {
        padded_buffers[k].assign(values.size() + 2 * kPadding, T(0));
        ptrs[k] = padded_buffers[k].data() + kPadding;
      }
      if (!values.empty()) {
        memcpy(ptrs[k], values.data(), values.size() * sizeof(T));
      }
    }
    g_current_library = lib.name().c_str();
    std::feclearexcept(FE_ALL_EXCEPT);
    if (!guarded && c.expected) {
      c.expected(ptrs.data());
    } else {
      c.call(Target{lib, c.symbol, c.api}, ptrs.data());
    }
    *exceptions = std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
    std::vector<std::vector<T>> data(num);
    for (size_t k = 0; k < num; ++k) {
      data[k].assign(ptrs[k], ptrs[k] + (c.ops[k].*init).size());
    }
    return data;
  };
  int ref_exceptions = 0, hwy_exceptions = 0;
  if (c.verify) {
    return c.verify(run(hwy, &Operand<T>::hwy_init, true, &hwy_exceptions));
  }
  const std::vector<std::vector<T>> ref_data =
      run(ref, &Operand<T>::ref_init, false, &ref_exceptions);
  const std::vector<std::vector<T>> hwy_data =
      run(hwy, &Operand<T>::hwy_init, true, &hwy_exceptions);
  std::string problem = Check(c, hwy_data, ref_data, ref.name());
  const int extra = hwy_exceptions & ~ref_exceptions;
  if (problem.empty() && opt.fp_flags && extra != 0 && !c.expected) {
    problem = "floating-point exceptions only in hwyBLAS:";
    if (extra & FE_INVALID) problem += " invalid";
    if (extra & FE_DIVBYZERO) problem += " divide-by-zero";
    if (extra & FE_OVERFLOW) problem += " overflow";
  }
  return problem;
}

uint64_t SplitMix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

struct Stats {
  int64_t cases = 0, failures = 0;
  std::map<std::string, int64_t> per_routine;
};

// All routines of all types, type-erased: generate and run one case.
struct AnyRoutine {
  std::string name;
  std::function<bool(uint64_t seed, const Lib&, const Lib&, const Options&,
                     Stats&)>
      run;
};

template <typename T>
void AddRoutines(std::vector<AnyRoutine>& all) {
  for (Routine<T>& r : Routines<T>()) {
    Generator<T> gen = r.generate;
    const std::string name = r.name;
    all.push_back(
        {name, [gen, name](uint64_t seed, const Lib& hwy, const Lib& ref,
                           const Options& opt, Stats& stats) {
           Case<T> c;
           c.name = name;
           c.symbol = name + "_";
           Builder<T> b(c, seed, opt);
           gen(c, b);
           if (!c.call) return true;  // not tested with these options
           // (Generators may set a symbol, or change the interface.)
           if (c.api != Api::kFortran && c.symbol == name + "_") {
             c.symbol = "cblas_" + name;
           }
           if (!hwy.Find(c.symbol.c_str()) ||
               (!c.expected && !c.verify && !ref.Find(c.symbol.c_str()))) {
             return true;  // not provided by one of the libraries
           }
           char seed_str[32];
           snprintf(seed_str, sizeof(seed_str), "0x%016" PRIx64, seed);
           const std::string repro = name + ":" + seed_str;
           if (c.api != Api::kFortran) {
             c.params +=
                 c.api == Api::kCblasRow ? " api=cblas-row" : " api=cblas-col";
           }
           g_current_case = repro + " " + c.params;
           if (opt.verbose) {
             printf("%s\n", g_current_case.c_str());
             fflush(stdout);
           }
           ++stats.cases;
           ++stats.per_routine[name];
           const std::string problem = RunCase(c, hwy, ref, opt);
           if (problem.empty()) return true;
           ++stats.failures;
           std::string options;
           if (opt.netlib) options += " --netlib";
           if (!opt.special.empty()) options += " --special=" + opt.special;
           if (opt.fp_flags) options += " --fp-flags";
           printf("FAIL %s:%s\n  %s\n  reproduce: --repro=%s%s\n", name.c_str(),
                  c.params.c_str(), problem.c_str(), repro.c_str(),
                  options.c_str());
           fflush(stdout);
           return false;
         }});
  }
}

int Main(int argc, char** argv) {
  Options opt;
  std::string ref_name, ref_path;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (strncmp(arg, "--seed=", 7) == 0) {
      opt.seed = strtoull(arg + 7, nullptr, 0);
    } else if (strncmp(arg, "--iterations=", 13) == 0) {
      opt.iterations = strtoll(arg + 13, nullptr, 10);
    } else if (strncmp(arg, "--time=", 7) == 0) {
      opt.time = atof(arg + 7);
    } else if (strncmp(arg, "--types=", 8) == 0) {
      opt.types = arg + 8;
    } else if (strncmp(arg, "--routines=", 11) == 0) {
      std::string list = arg + 11;
      size_t start = 0;
      while (start <= list.size()) {
        const size_t comma = std::min(list.find(',', start), list.size());
        if (comma > start)
          opt.routines.push_back(list.substr(start, comma - start));
        start = comma + 1;
      }
    } else if (strncmp(arg, "--repro=", 8) == 0) {
      opt.repro = arg + 8;
    } else if (strncmp(arg, "--max-failures=", 15) == 0) {
      opt.max_failures = atoi(arg + 15);
    } else if (strcmp(arg, "--verbose") == 0) {
      opt.verbose = true;
    } else if (strcmp(arg, "--netlib") == 0) {
      opt.netlib = true;
    } else if (strncmp(arg, "--special=", 10) == 0) {
      opt.special = arg + 10;
      if (opt.special != "nan" && opt.special != "inf" &&
          opt.special != "zero" && opt.special != "subnormal" &&
          opt.special != "huge") {
        fprintf(stderr, "Unknown special values: %s\n", arg);
        return 1;
      }
    } else if (strncmp(arg, "--threads=", 10) == 0) {
      opt.threads = std::max(1, atoi(arg + 10));
    } else if (strcmp(arg, "--fp-flags") == 0) {
      opt.fp_flags = true;
    } else if (const char* eq = strchr(arg, '=');
               eq != nullptr && arg[0] != '-') {
      ref_name.assign(arg, eq);
      ref_path = eq + 1;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg);
      return 1;
    }
  }
  if (ref_path.empty()) {
    fprintf(stderr, "Usage: %s [options] Name=path/to/reference/blas\n",
            argv[0]);
    return 1;
  }
  Lib hwy("hwyBLAS", HWYBLAS_LIBRARY);
  Lib ref(ref_name, ref_path);
  if (!hwy.ok() || !ref.ok()) return 1;
  if (auto target = hwy.Get<const char* (*)()>("hwyblas_target_name")) {
    printf("hwyBLAS target: %s, reference: %s\n", target(), ref_name.c_str());
  }
  InstallCrashHandler();

  std::vector<AnyRoutine> all;
  for (char t : opt.types) {
    switch (t) {
      case 's':
        AddRoutines<float>(all);
        break;
      case 'd':
        AddRoutines<double>(all);
        break;
      case 'c':
        AddRoutines<std::complex<float>>(all);
        break;
      case 'z':
        AddRoutines<std::complex<double>>(all);
        break;
      default:
        fprintf(stderr, "Unknown type %c\n", t);
        return 1;
    }
  }
  Stats stats;
  if (!opt.repro.empty()) {
    const size_t colon = opt.repro.find(':');
    const std::string name = opt.repro.substr(0, colon);
    const uint64_t seed =
        colon == std::string::npos
            ? 0
            : strtoull(opt.repro.c_str() + colon + 1, nullptr, 0);
    opt.verbose = true;
    for (const AnyRoutine& r : all) {
      if (r.name == name) return r.run(seed, hwy, ref, opt, stats) ? 0 : 1;
    }
    fprintf(stderr, "Unknown routine %s\n", name.c_str());
    return 1;
  }
  if (!opt.routines.empty()) {
    std::vector<AnyRoutine> selected;
    for (const AnyRoutine& r : all) {
      if (std::find(opt.routines.begin(), opt.routines.end(), r.name) !=
          opt.routines.end()) {
        selected.push_back(r);
      }
    }
    all.swap(selected);
  }
  if (all.empty()) {
    fprintf(stderr, "No routines selected\n");
    return 1;
  }

  const auto start = std::chrono::steady_clock::now();
  const auto done = [&opt, start](int64_t i) {
    if (opt.time <= 0) return i >= opt.iterations;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return elapsed >= opt.time;
  };
  const auto run_case = [&](int64_t i, Stats& s) {
    const uint64_t seed = SplitMix(opt.seed * 0x100000001B3ull + i);
    const AnyRoutine& r = all[seed % all.size()];
    r.run(SplitMix(seed), hwy, ref, opt, s);
  };
  if (opt.threads == 1) {
    for (int64_t i = 0; !done(i) && stats.failures < opt.max_failures; ++i) {
      run_case(i, stats);
    }
  } else {
    // Thread t runs every threads-th case of a block, so that every case is
    // the same as with one thread.
    const int64_t block = 200 * static_cast<int64_t>(opt.threads);
    for (int64_t first = 0; !done(first) && stats.failures < opt.max_failures;
         first += block) {
      std::vector<Stats> per_thread(opt.threads);
      std::vector<std::thread> threads;
      for (int t = 0; t < opt.threads; ++t) {
        threads.emplace_back([&, t] {
          for (int64_t i = first + t; i < first + block; i += opt.threads) {
            if (opt.time <= 0 && i >= opt.iterations) break;
            run_case(i, per_thread[t]);
          }
        });
      }
      for (std::thread& t : threads) t.join();
      for (const Stats& s : per_thread) {
        stats.cases += s.cases;
        stats.failures += s.failures;
        for (const auto& [name, count] : s.per_routine) {
          stats.per_routine[name] += count;
        }
      }
    }
  }
  printf("%" PRId64 " cases of %zu routines, %" PRId64 " failures\n",
         stats.cases, stats.per_routine.size(), stats.failures);
  return stats.failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace hwyblas_fuzz

int main(int argc, char** argv) { return hwyblas_fuzz::Main(argc, argv); }
