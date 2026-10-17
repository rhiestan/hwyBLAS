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
//
// Each case picks a routine and random arguments: sizes from 0 to beyond
// the kernels' blocking, transposes, sides, triangles, alpha/beta including
// 0 and 1, leading dimensions with padding, and positive or negative
// increments. Besides the results, a case checks that hwyBLAS
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
// result. hwyBLAS is loaded from the fuzzer's own directory, so the
// environment variable HWYBLAS_TARGET selects the target that is tested.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
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
  bool exact = false;  // integer result: must match exactly
  int ld = 0;          // leading dimension, for messages (0: vector)
  bool guard_after = true;
  std::vector<Use> use;
  std::vector<T> hwy_init;  // initial values for hwyBLAS
  std::vector<T> ref_init;  // initial values for the reference library
};

template <typename T>
struct Case {
  std::string name;    // routine, e.g. "dgemm"
  std::string symbol;  // symbol that both libraries must export
  std::string params;  // description of the arguments
  std::vector<Operand<T>> ops;
  int error_dim = 0;  // length of the inner products / of the solve
  // Calls the routine `symbol` of a library with the operands.
  std::function<void(const Lib&, const std::string& symbol, T* const*)> call;
};

template <typename R = void, typename... Args>
R Call(const Lib& lib, const std::string& symbol, Args... args) {
  return lib.Get<R (*)(Args...)>(symbol.c_str())(args...);
}

// Builds one case: chooses (and records) the arguments and operands.
template <typename T>
class Builder {
 public:
  using R = Real<T>;

  Builder(Case<T>& c, uint64_t seed) : c_(c), rng_(seed) {}

  Rng& rng() { return rng_; }

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
    if (level == 3 && r < 2 && !huge_) {
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

  int Inc(const char* key) {
    static constexpr int kIncs[] = {1, 1, 1, 1, 1, -1, 2, -2, 3, -5, 8, 17};
    const int inc = kIncs[Uniform(
        0, static_cast<int>(sizeof(kIncs) / sizeof(kIncs[0])) - 1)];
    Param(key, inc);
    return inc;
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

  char Letter(const char* key, const char* choices) {
    const char ch = choices[Uniform(0, static_cast<int>(strlen(choices)) - 1)];
    Param(key, ch);
    return ch;
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

  // An element of an operand: index(i, j) is its position, or -1 for
  // elements that are not stored.
  struct Shape {
    int rows = 0, cols = 0;
    int ld = 0;  // for messages
    std::function<long(int, int)> index;
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
    long size = 0;
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
        const long idx = s.index(i, j);
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
    c_.ops.push_back(std::move(op));
    return static_cast<int>(c_.ops.size()) - 1;
  }

  // Common shapes.
  Shape General(int rows, int cols, int ld) {
    Shape s;
    s.rows = rows;
    s.cols = cols;
    s.ld = ld;
    s.index = [ld](int i, int j) { return i + static_cast<long>(j) * ld; };
    return s;
  }
  Shape Vector(int n, int inc) {
    Shape s;
    s.rows = n;
    s.cols = 1;
    const long step = std::abs(inc);
    s.index = [step](int i, int) { return i * step; };
    return s;
  }
  // The `uplo` triangle of an n x n matrix.
  Shape Triangle(int n, int ld, char uplo) {
    Shape s = General(n, n, ld);
    s.index = [ld, uplo](int i, int j) -> long {
      if (uplo == 'U' ? i > j : i < j) return -1;
      return i + static_cast<long>(j) * ld;
    };
    return s;
  }
  // General band matrix with kl sub- and ku superdiagonals.
  Shape Band(int m, int n, int kl, int ku, int ld) {
    Shape s = General(m, n, ld);
    s.index = [kl, ku, ld](int i, int j) -> long {
      if (i < j - ku || i > j + kl) return -1;
      return ku + i - j + static_cast<long>(j) * ld;
    };
    return s;
  }
  // The `uplo` triangle of an n x n band matrix with k off-diagonals.
  Shape TriangleBand(int n, int k, int ld, char uplo) {
    Shape s = General(n, n, ld);
    s.index = [k, ld, uplo](int i, int j) -> long {
      if (uplo == 'U') {
        if (i > j || i < j - k) return -1;
        return k + i - j + static_cast<long>(j) * ld;
      }
      if (i < j || i > j + k) return -1;
      return i - j + static_cast<long>(j) * ld;
    };
    return s;
  }
  // The `uplo` triangle of an n x n matrix in packed storage.
  Shape Packed(int n, char uplo) {
    Shape s;
    s.rows = n;
    s.cols = n;
    s.index = [n, uplo](int i, int j) -> long {
      if (uplo == 'U') {
        return i > j ? -1 : i + static_cast<long>(j) * (j + 1) / 2;
      }
      return i < j ? -1 : i + static_cast<long>(j) * (2 * n - j - 1) / 2;
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
  Rng rng_;
  bool huge_ = false;  // a huge dimension was chosen
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
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int y = b.Add("y", b.Vector(n, incy), true);
    if (IsZero(alpha)) b.Ignore(x);
    c.error_dim = 1;
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &n, &alpha, o[x], &incx, o[y], &incy);
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
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &n, &alpha, o[x], &incx, &beta, o[y], &incy);
    };
  });
  add(p + "scal", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const T alpha = b.Scalar("alpha");
    const int incx = std::abs(b.Inc("incx"));
    const int x = b.Add("x", b.Vector(n, incx), true);
    c.error_dim = 1;
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &n, &alpha, o[x], &incx);
    };
  });
  if constexpr (cplx) {
    // csscal, zdscal: real alpha.
    add(p + rp + "scal", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const R alpha = b.RealScalar("alpha");
      const int incx = std::abs(b.Inc("incx"));
      const int x = b.Add("x", b.Vector(n, incx), true);
      c.error_dim = 1;
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        Call(lib, sym, &n, &alpha, o[x], &incx);
      };
    });
  }
  add(p + "copy", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int y = b.Add("y", b.Vector(n, incy), true);
    b.Ignore(y);
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &n, o[x], &incx, o[y], &incy);
    };
  });
  add(p + "swap", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int x = b.Add("x", b.Vector(n, incx), true);
    const int y = b.Add("y", b.Vector(n, incy), true);
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &n, o[x], &incx, o[y], &incy);
    };
  });
  // Dot products: real ones return the result, complex ones use the CBLAS
  // *_sub functions (the Fortran ABI for complex return values varies).
  const auto add_dot = [&](std::string name, bool conj) {
    add(name, [conj](Case<T>& c, Builder<T>& b) {
      (void)conj;  // only used for complex types
      const int n = b.Dim("n", 1);
      const int incx = b.Inc("incx"), incy = b.Inc("incy");
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int r = b.Result("result");
      c.error_dim = n;
      if constexpr (cplx) {
        c.symbol = std::string("cblas_") + Prefix<T>() +
                   (conj ? "dotc_sub" : "dotu_sub");
        c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
          Call(lib, sym, n, static_cast<const T*>(o[x]), incx,
               static_cast<const T*>(o[y]), incy, o[r]);
        };
      } else {
        c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
          *o[r] = Call<T>(lib, sym, &n, o[x], &incx, o[y], &incy);
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
      const int incx = b.Inc("incx"), incy = b.Inc("incy");
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int r = b.Result("result");
      c.error_dim = 1;  // accumulates in double
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        *o[r] = Call<float>(lib, sym, &n, &sb, o[x], &incx, o[y], &incy);
      };
    });
    add("dsdot", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const int incx = b.Inc("incx"), incy = b.Inc("incy");
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int r = b.Result("result");
      c.error_dim = 1;  // accumulates in double
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        *o[r] = static_cast<float>(
            Call<double>(lib, sym, &n, o[x], &incx, o[y], &incy));
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
      const int incx = std::abs(b.Inc("incx"));
      const int x = b.Add("x", b.Vector(n, incx), false);
      const int r = b.Result("result");
      c.error_dim = asum ? n : 2;
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        *o[r] = T(Call<R>(lib, sym, &n, o[x], &incx));
      };
    });
  }
  add(std::string("i") + p + "amax", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = std::abs(b.Inc("incx"));
    const int x = b.Add("x", b.Vector(n, incx), false);
    const int r = b.Result("index", true);
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      *o[r] = T(static_cast<R>(Call<int>(lib, sym, &n, o[x], &incx)));
    };
  });
  // rot with real c and s: srot, drot, csrot, zdrot.
  add((cplx ? p + rp : p) + "rot", [](Case<T>& c, Builder<T>& b) {
    const int n = b.Dim("n", 1);
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const R angle = static_cast<R>(3.2 * b.Unit());
    const R cs = std::cos(angle), sn = std::sin(angle);
    b.Param("c", cs);
    b.Param("s", sn);
    const int x = b.Add("x", b.Vector(n, incx), true);
    const int y = b.Add("y", b.Vector(n, incy), true);
    c.error_dim = 2;
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &n, o[x], &incx, o[y], &incy, &cs, &sn);
    };
  });
  if constexpr (!cplx) {
    add(p + "rotm", [](Case<T>& c, Builder<T>& b) {
      const int n = b.Dim("n", 1);
      const int incx = b.Inc("incx"), incy = b.Inc("incy");
      const int flag = b.Uniform(-2, 1);
      b.Param("flag", flag);
      const int x = b.Add("x", b.Vector(n, incx), true);
      const int y = b.Add("y", b.Vector(n, incy), true);
      // param = {flag, h11, h21, h12, h22}; unused entries are NaN.
      typename Builder<T>::Shape s = b.Vector(5, 1);
      s.index = [flag](int i, int) -> long {
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
      c.error_dim = 2;
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        Call(lib, sym, &n, o[x], &incx, o[y], &incy,
             static_cast<const T*>(o[param]));
      };
    });
  }

  // ----- Level 2
  add(p + "gemv", [kTrans](Case<T>& c, Builder<T>& b) {
    const char trans = b.Letter("trans", kTrans);
    const int m = b.Dim("m", 2), n = b.Dim("n", 2);
    const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
    const int lda = b.Ld("lda", m);
    const int incx = b.Inc("incx"), incy = b.Inc("incy");
    const int lx = trans == 'N' ? n : m, ly = trans == 'N' ? m : n;
    const int a = b.Add("A", b.General(m, n, lda), false);
    const int x = b.Add("x", b.Vector(lx, incx), false);
    const int y = b.Add("y", b.Vector(ly, incy), true);
    if (IsZero(alpha)) b.Ignore(a), b.Ignore(x);
    // (BLAS returns at once, without scaling y, if m or n is 0.)
    if (IsZero(beta) && m > 0 && n > 0) b.Ignore(y);
    c.error_dim = lx + 1;
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &trans, &m, &n, &alpha, o[a], &lda, o[x], &incx, &beta,
           o[y], &incy);
    };
  });
  add(p + "gbmv", [kTrans](Case<T>& c, Builder<T>& b) {
    const char trans = b.Letter("trans", kTrans);
    const int m = b.Dim("m", 2), n = b.Dim("n", 2);
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
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &trans, &m, &n, &kl, &ku, &alpha, o[a], &lda, o[x], &incx,
           &beta, o[y], &incy);
    };
  });
  // symv/hemv, sbmv/hbmv, spmv/hpmv: kind 0 full, 1 band, 2 packed.
  for (int kind = 0; kind < 3; ++kind) {
    static const char* const kNames[2][3] = {{"symv", "sbmv", "spmv"},
                                             {"hemv", "hbmv", "hpmv"}};
    add(p + kNames[cplx][kind], [kind, kUplo](Case<T>& c, Builder<T>& b) {
      const char uplo = b.Letter("uplo", kUplo);
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
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        if (kind == 0) {
          Call(lib, sym, &uplo, &n, &alpha, o[a], &lda, o[x], &incx, &beta,
               o[y], &incy);
        } else if (kind == 1) {
          Call(lib, sym, &uplo, &n, &k, &alpha, o[a], &lda, o[x], &incx, &beta,
               o[y], &incy);
        } else {
          Call(lib, sym, &uplo, &n, &alpha, o[a], o[x], &incx, &beta, o[y],
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
      add(p + kNames[solve][kind], [solve, kind, kTrans, kUplo](Case<T>& c,
                                                                Builder<T>& b) {
        const char uplo = b.Letter("uplo", kUplo);
        const char trans = b.Letter("trans", kTrans);
        const char diag = b.Letter("diag", "NU");
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
        c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
          if (kind == 0) {
            Call(lib, sym, &uplo, &trans, &diag, &n, o[a], &lda, o[x], &incx);
          } else if (kind == 1) {
            Call(lib, sym, &uplo, &trans, &diag, &n, &k, o[a], &lda, o[x],
                 &incx);
          } else {
            Call(lib, sym, &uplo, &trans, &diag, &n, o[a], o[x], &incx);
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
      const int lda = b.Ld("lda", m);
      const int x = b.Add("x", b.Vector(m, incx), false);
      const int y = b.Add("y", b.Vector(n, incy), false);
      const int a = b.Add("A", b.General(m, n, lda), true);
      if (IsZero(alpha)) b.Ignore(x), b.Ignore(y);
      c.error_dim = 2;
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        Call(lib, sym, &m, &n, &alpha, o[x], &incx, o[y], &incy, o[a], &lda);
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
        const char uplo = b.Letter("uplo", kUplo);
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
        c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
          const void* al = real_alpha ? static_cast<const void*>(&ralpha)
                                      : static_cast<const void*>(&alpha);
          if (two && packed) {
            Call(lib, sym, &uplo, &n, al, o[x], &incx, o[y], &incy, o[a]);
          } else if (two) {
            Call(lib, sym, &uplo, &n, al, o[x], &incx, o[y], &incy, o[a], &lda);
          } else if (packed) {
            Call(lib, sym, &uplo, &n, al, o[x], &incx, o[a]);
          } else {
            Call(lib, sym, &uplo, &n, al, o[x], &incx, o[a], &lda);
          }
        };
      });
    }
  }

  // ----- Level 3
  // gemm and gemmt (one triangle of C).
  for (int tri = 0; tri < 2; ++tri) {
    add(p + (tri ? "gemmt" : "gemm"),
        [tri, kTrans, kUplo](Case<T>& c, Builder<T>& b) {
          const char uplo = tri ? b.Letter("uplo", kUplo) : 'N';
          const char ta = b.Letter("transa", kTrans);
          const char tb = b.Letter("transb", kTrans);
          const int m = b.Dim("m", 3);
          const int n = tri ? m : b.Dim("n", 3);
          const int k = b.Dim("k", 3);
          const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
          const int ar = ta == 'N' ? m : k, ac = ta == 'N' ? k : m;
          const int br = tb == 'N' ? k : n, bc = tb == 'N' ? n : k;
          const int lda = b.Ld("lda", ar), ldb = b.Ld("ldb", br);
          const int ldc = b.Ld("ldc", m);
          const int a = b.Add("A", b.General(ar, ac, lda), false);
          const int bb = b.Add("B", b.General(br, bc, ldb), false);
          const int cc = b.Add(
              "C", tri ? b.Triangle(n, ldc, uplo) : b.General(m, n, ldc), true);
          if (IsZero(alpha)) b.Ignore(a), b.Ignore(bb);
          if (IsZero(beta)) b.Ignore(cc);
          c.error_dim = k + 1;
          c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
            if (tri) {
              Call(lib, sym, &uplo, &ta, &tb, &n, &k, &alpha, o[a], &lda, o[bb],
                   &ldb, &beta, o[cc], &ldc);
            } else {
              Call(lib, sym, &ta, &tb, &m, &n, &k, &alpha, o[a], &lda, o[bb],
                   &ldb, &beta, o[cc], &ldc);
            }
          };
        });
  }
  // symm (and hemm).
  for (int herm = 0; herm < (cplx ? 2 : 1); ++herm) {
    add(p + (herm ? "hemm" : "symm"), [herm, kUplo](Case<T>& c, Builder<T>& b) {
      const char side = b.Letter("side", "LR");
      const char uplo = b.Letter("uplo", kUplo);
      const int m = b.Dim("m", 3), n = b.Dim("n", 3);
      const T alpha = b.Scalar("alpha"), beta = b.Scalar("beta", true);
      const int ka = side == 'L' ? m : n;
      const int lda = b.Ld("lda", ka), ldb = b.Ld("ldb", m),
                ldc = b.Ld("ldc", m);
      typename Builder<T>::Shape s = b.Triangle(ka, lda, uplo);
      s.hermitian = herm != 0;
      const int a = b.Add("A", s, false);
      const int bb = b.Add("B", b.General(m, n, ldb), false);
      const int cc = b.Add("C", b.General(m, n, ldc), true);
      if (IsZero(alpha)) b.Ignore(a), b.Ignore(bb);
      if (IsZero(beta)) b.Ignore(cc);
      c.error_dim = ka + 1;
      c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
        Call(lib, sym, &side, &uplo, &m, &n, &alpha, o[a], &lda, o[bb], &ldb,
             &beta, o[cc], &ldc);
      };
    });
  }
  // syrk, syr2k (and herk, her2k: real beta, herk also a real alpha).
  for (int herm = 0; herm < (cplx ? 2 : 1); ++herm) {
    for (int two = 0; two < 2; ++two) {
      std::string name = p + (herm ? "her" : "syr") + (two ? "2k" : "k");
      add(name, [herm, two, kUplo](Case<T>& c, Builder<T>& b) {
        const char uplo = b.Letter("uplo", kUplo);
        const char trans =
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
        const int lda = b.Ld("lda", ar);
        const int ldb = two ? b.Ld("ldb", ar) : 0;
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
        c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
          const void* al = herm && !two ? static_cast<const void*>(&ralpha)
                                        : static_cast<const void*>(&alpha);
          const void* be = herm ? static_cast<const void*>(&rbeta)
                                : static_cast<const void*>(&beta);
          if (two) {
            Call(lib, sym, &uplo, &trans, &n, &k, al, o[a], &lda, o[bb], &ldb,
                 be, o[cc], &ldc);
          } else {
            Call(lib, sym, &uplo, &trans, &n, &k, al, o[a], &lda, be, o[cc],
                 &ldc);
          }
        };
      });
    }
  }
  // trmm, trsm.
  for (int solve = 0; solve < 2; ++solve) {
    add(p + (solve ? "trsm" : "trmm"),
        [solve, kTrans, kUplo](Case<T>& c, Builder<T>& b) {
          const char side = b.Letter("side", "LR");
          const char uplo = b.Letter("uplo", kUplo);
          const char transa = b.Letter("transa", kTrans);
          const char diag = b.Letter("diag", "NU");
          const int m = b.Dim("m", 3), n = b.Dim("n", 3);
          const T alpha = b.Scalar("alpha");
          const int ka = side == 'L' ? m : n;
          const int lda = b.Ld("lda", ka), ldb = b.Ld("ldb", m);
          typename Builder<T>::Shape s = b.Triangle(ka, lda, uplo);
          s.unit_diag = diag == 'U';
          s.solve = solve != 0;
          const int a = b.Add("A", s, false);
          const int bb = b.Add("B", b.General(m, n, ldb), true);
          if (IsZero(alpha)) b.Ignore(a), b.Ignore(bb);
          c.error_dim = ka + 1;
          c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
            Call(lib, sym, &side, &uplo, &transa, &diag, &m, &n, &alpha, o[a],
                 &lda, o[bb], &ldb);
          };
        });
  }
  // omatcopy (OpenBLAS extension): B := alpha * op(A).
  add(p + "omatcopy", [](Case<T>& c, Builder<T>& b) {
    const char order = b.Letter("order", "CR");
    const char trans = b.Letter("trans", "NTRC");
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
    c.call = [=](const Lib& lib, const std::string& sym, T* const* o) {
      Call(lib, sym, &order, &trans, &rows, &cols, &alpha, o[a], &lda, o[bb],
           &ldb);
    };
  });
  return list;
}

// ---------------------------------------------------------------------------
// Running and checking cases

// The current case and library, printed if it crashes.
std::string g_current_case;
const char* g_current_library = "";

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

template <typename T>
double Magnitude(const T& v) {
  return IsNan(v) ? 0 : static_cast<double>(std::abs(v));
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
    const double tol = 16 * eps * (c.error_dim + 1) * scale;
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
                 ToString(r);
        }
        continue;
      }
      const double diff = static_cast<double>(std::abs(hc - rc));
      if (op.exact ? diff != 0 : !(diff <= tol)) {
        return where(i) + ": hwyBLAS " + ToString(h) + ", " + ref_name + " " +
               ToString(r) + " (difference " + ToString(diff) + ", tolerance " +
               ToString(tol) + ")";
      }
    }
  }
  return "";
}

template <typename T>
std::string RunCase(const Case<T>& c, const Lib& hwy, const Lib& ref) {
  // Runs the case with library `lib` on copies of the operands (`init` of
  // each), and returns the operands afterwards. hwyBLAS gets guarded
  // copies; the reference gets padded ones, because reference libraries
  // may access a little beyond the operands (e.g. OpenBLAS 0.3.34 cscal
  // with alpha = 0).
  const auto run = [&c](const Lib& lib, std::vector<T> Operand<T>::* init,
                        bool guarded) {
    constexpr size_t kPadding = 64;  // elements before and after
    const size_t num = c.ops.size();
    std::vector<std::unique_ptr<GuardedBuffer>> guarded_buffers;
    std::vector<std::vector<T>> padded_buffers(num);
    std::vector<T*> ptrs(num);
    for (size_t k = 0; k < num; ++k) {
      const std::vector<T>& values = c.ops[k].*init;
      if (guarded) {
        guarded_buffers.push_back(std::make_unique<GuardedBuffer>(
            values.size() * sizeof(T), c.ops[k].guard_after));
        ptrs[k] = static_cast<T*>(guarded_buffers.back()->data());
      } else {
        padded_buffers[k].assign(values.size() + 2 * kPadding, T(0));
        ptrs[k] = padded_buffers[k].data() + kPadding;
      }
      if (!values.empty()) {
        memcpy(ptrs[k], values.data(), values.size() * sizeof(T));
      }
    }
    g_current_library = lib.name().c_str();
    c.call(lib, c.symbol, ptrs.data());
    std::vector<std::vector<T>> data(num);
    for (size_t k = 0; k < num; ++k) {
      data[k].assign(ptrs[k], ptrs[k] + (c.ops[k].*init).size());
    }
    return data;
  };
  const std::vector<std::vector<T>> ref_data =
      run(ref, &Operand<T>::ref_init, false);
  const std::vector<std::vector<T>> hwy_data =
      run(hwy, &Operand<T>::hwy_init, true);
  return Check(c, hwy_data, ref_data, ref.name());
}

uint64_t SplitMix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

struct Options {
  uint64_t seed = 1;
  long iterations = 1000;
  double time = 0;  // > 0: run for this many seconds
  std::string types = "sdcz";
  std::vector<std::string> routines;  // empty: all
  std::string repro;                  // name:seed
  int max_failures = 10;
  bool verbose = false;
};

struct Stats {
  long cases = 0, failures = 0;
  std::map<std::string, long> per_routine;
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
           Builder<T> b(c, seed);
           gen(c, b);
           if (!hwy.Find(c.symbol.c_str()) || !ref.Find(c.symbol.c_str())) {
             return true;  // not provided by one of the libraries
           }
           char seed_str[32];
           snprintf(seed_str, sizeof(seed_str), "0x%016llx",
                    static_cast<unsigned long long>(seed));
           const std::string repro = name + ":" + seed_str;
           g_current_case = repro + " " + c.params;
           if (opt.verbose) {
             printf("%s\n", g_current_case.c_str());
             fflush(stdout);
           }
           ++stats.cases;
           ++stats.per_routine[name];
           const std::string problem = RunCase(c, hwy, ref);
           if (problem.empty()) return true;
           ++stats.failures;
           printf("FAIL %s:%s\n  %s\n  reproduce: --repro=%s\n", name.c_str(),
                  c.params.c_str(), problem.c_str(), repro.c_str());
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
      opt.iterations = atol(arg + 13);
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
  for (long i = 0;; ++i) {
    if (opt.time > 0) {
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      if (elapsed >= opt.time) break;
    } else if (i >= opt.iterations) {
      break;
    }
    const uint64_t seed = SplitMix(opt.seed * 0x100000001B3ull + i);
    const AnyRoutine& r = all[seed % all.size()];
    r.run(SplitMix(seed), hwy, ref, opt, stats);
    if (stats.failures >= opt.max_failures) break;
  }
  printf("%ld cases of %zu routines, %ld failures\n", stats.cases,
         stats.per_routine.size(), stats.failures);
  return stats.failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace hwyblas_fuzz

int main(int argc, char** argv) { return hwyblas_fuzz::Main(argc, argv); }
