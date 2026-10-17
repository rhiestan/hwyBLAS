// Semantics beyond the results for ordinary numbers: propagation of NaN and
// Inf, subnormal numbers, the floating-point environment, reproducibility,
// and element offsets beyond 2^31.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/cblas.h"
#include "hwyblas/hwyblas.h"
#include "test_util.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#include <immintrin.h>
#define HWYBLAS_TEST_X86 1
#else
#define HWYBLAS_TEST_X86 0
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace hwyblas_test {
namespace {

template <typename T>
class Semantics : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(Semantics, ScalarTypes);

template <typename T>
T Inf() {
  return T(std::numeric_limits<RealT<T>>::infinity());
}

template <typename T>
bool IsNan(T x) {
  return std::isnan(std::real(x)) || std::isnan(std::imag(x));
}

template <typename T>
bool IsFinite(T x) {
  return std::isfinite(std::real(x)) && std::isfinite(std::imag(x));
}

// Checks a special (NaN or Inf) result: NaN stays NaN, Inf may become NaN.
template <typename T>
bool IsSpecial(T x, bool nan) {
  return nan ? IsNan(x) : !IsFinite(x);
}

// A triangular matrix with a dominant diagonal and no zeros in the triangle,
// so that every solution element depends on all earlier ones.
template <typename T>
std::vector<T> DominantTriangle(int n, char uplo, Random<T>& rng) {
  std::vector<T> a(static_cast<size_t>(n) * n, T(0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (i == j) {
        a[i + j * n] = T(2) + T(std::abs(rng.Next()));
      } else if (uplo == 'U' ? i < j : i > j) {
        a[i + j * n] = rng.Next() / T(static_cast<RealT<T>>(n));
      }
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// NaN and Inf

// A NaN or Inf in op(A)(i0, l) reaches all of row i0 of C (B has no zeros)
// and nothing else; likewise op(B)(l, j0) and column j0, and C(i0, j0) for
// beta = 1.
TYPED_TEST(Semantics, GemmPropagatesNanAndInf) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(1);
    for (int size : {1, 7, 33, 130}) {
      const int m = size, n = size + 3, k = size + 1;
      const int i0 = m / 2, j0 = n - 1, l0 = k / 3;
      for (char ta : Ops<T>()) {
        for (char tb : {'N', 'T'}) {
          for (int where = 0; where < 3; ++where) {  // A, B, C
            for (bool nan : {true, false}) {
              const T value = nan ? NaN<T>() : Inf<T>();
              const int lda = ta == 'N' ? m : k, ldb = tb == 'N' ? k : n;
              std::vector<T> a = rng.Vector(static_cast<size_t>(m) * k);
              std::vector<T> b = rng.Vector(static_cast<size_t>(k) * n);
              std::vector<T> c = rng.Vector(static_cast<size_t>(m) * n);
              if (where == 0) {
                a[ta == 'N' ? i0 + l0 * lda : l0 + i0 * lda] = value;
              } else if (where == 1) {
                b[tb == 'N' ? l0 + j0 * ldb : j0 + l0 * ldb] = value;
              } else {
                c[i0 + j0 * m] = value;
              }
              const T alpha = MakeScalar<T>(0.5, 0.25);
              const T beta = where == 2 ? T(1) : T(0);
              HWYBLAS_CALL(T, gemm, &ta, &tb, &m, &n, &k, P(&alpha),
                           P(a.data()), &lda, P(b.data()), &ldb, P(&beta),
                           P(c.data()), &m);
              for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                  const bool hit = where == 0   ? i == i0
                                   : where == 1 ? j == j0
                                                : i == i0 && j == j0;
                  const T x = c[i + j * m];
                  ASSERT_TRUE(hit ? IsSpecial(x, nan) : IsFinite(x))
                      << "transa=" << ta << " transb=" << tb << " m=" << m
                      << " where=" << where << " nan=" << nan << " C(" << i
                      << "," << j << ")=" << x;
                }
              }
            }
          }
        }
      }
    }
  });
}

// y := alpha op(A) x + beta y: A(i0, j0) reaches y(i0) (or y(j0) for a
// transpose), x(j0) all of y, and y(i0) (beta = 1) only itself.
TYPED_TEST(Semantics, GemvPropagatesNanAndInf) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(2);
    for (int size : {1, 5, 17, 70}) {
      const int m = size + 2, n = size;
      const int i0 = m - 1, j0 = n / 2;
      for (char trans : Ops<T>()) {
        const int lx = trans == 'N' ? n : m, ly = trans == 'N' ? m : n;
        for (int where = 0; where < 3; ++where) {  // A, x, y
          for (bool nan : {true, false}) {
            const T value = nan ? NaN<T>() : Inf<T>();
            std::vector<T> a = rng.Vector(static_cast<size_t>(m) * n);
            std::vector<T> x = rng.Vector(lx), y = rng.Vector(ly);
            int hit = -1;  // the affected element of y; -1: all
            if (where == 0) {
              a[i0 + j0 * m] = value;
              hit = trans == 'N' ? i0 : j0;
            } else if (where == 1) {
              x[lx - 1] = value;
            } else {
              hit = ly / 2;
              y[hit] = value;
            }
            const T alpha = MakeScalar<T>(-0.75, 0.5);
            const T beta = where == 2 ? T(1) : T(0);
            const int inc = 1;
            HWYBLAS_CALL(T, gemv, &trans, &m, &n, P(&alpha), P(a.data()), &m,
                         P(x.data()), &inc, P(&beta), P(y.data()), &inc);
            for (int i = 0; i < ly; ++i) {
              const bool affected = hit < 0 || i == hit;
              ASSERT_TRUE(affected ? IsSpecial(y[i], nan) : IsFinite(y[i]))
                  << "trans=" << trans << " m=" << m << " where=" << where
                  << " nan=" << nan << " y(" << i << ")=" << y[i];
            }
          }
        }
      }
    }
  });
}

// X := inv(op(A)) B or B inv(op(A)): B(i0, j0) reaches exactly the
// elements of X that depend on it: in column j0 (left side) the rows after
// i0 for a lower triangular op(A), before i0 for an upper one; in row i0
// (right side) the columns after j0 for upper, before j0 for lower.
TYPED_TEST(Semantics, TrsmPropagatesNanAndInf) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(3);
    for (int size : {1, 5, 40, 100}) {
      const int m = size, n = size / 2 + 3;
      const int i0 = m / 2, j0 = n / 3;
      for (char side : {'L', 'R'}) {
        for (char uplo : {'U', 'L'}) {
          for (char trans : {'N', 'T'}) {
            for (bool nan : {true, false}) {
              const int ka = side == 'L' ? m : n;
              const std::vector<T> a = DominantTriangle(ka, uplo, rng);
              std::vector<T> b = rng.Vector(static_cast<size_t>(m) * n);
              b[i0 + j0 * m] = nan ? NaN<T>() : Inf<T>();
              const T alpha(1);
              const char diag = 'N';
              HWYBLAS_CALL(T, trsm, &side, &uplo, &trans, &diag, &m, &n,
                           P(&alpha), P(a.data()), &ka, P(b.data()), &m);
              const bool op_lower = (uplo == 'L') == (trans == 'N');
              for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                  bool hit;
                  if (side == 'L') {
                    hit = j == j0 && (op_lower ? i >= i0 : i <= i0);
                  } else {
                    hit = i == i0 && (op_lower ? j <= j0 : j >= j0);
                  }
                  const T x = b[i + j * m];
                  ASSERT_TRUE(hit ? IsSpecial(x, nan) : IsFinite(x))
                      << "side=" << side << " uplo=" << uplo
                      << " trans=" << trans << " m=" << m << " nan=" << nan
                      << " X(" << i << "," << j << ")=" << x;
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
RealT<T> Nrm2(int n, const T* x, int inc) {
  if constexpr (std::is_same_v<T, float>) return snrm2_(&n, x, &inc);
  if constexpr (std::is_same_v<T, double>) return dnrm2_(&n, x, &inc);
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return scnrm2_(&n, P(x), &inc);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    return dznrm2_(&n, P(x), &inc);
  }
}

template <typename T>
RealT<T> Asum(int n, const T* x, int inc) {
  if constexpr (std::is_same_v<T, float>) return sasum_(&n, x, &inc);
  if constexpr (std::is_same_v<T, double>) return dasum_(&n, x, &inc);
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    return scasum_(&n, P(x), &inc);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    return dzasum_(&n, P(x), &inc);
  }
}

template <typename T>
T Dot(int n, const T* x, const T* y) {
  const int inc = 1;
  if constexpr (std::is_same_v<T, float>) return sdot_(&n, x, &inc, y, &inc);
  if constexpr (std::is_same_v<T, double>) return ddot_(&n, x, &inc, y, &inc);
  T result;
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    cblas_cdotu_sub(n, x, 1, y, 1, &result);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    cblas_zdotu_sub(n, x, 1, y, 1, &result);
  }
  return result;
}

// dot, nrm2 and asum of a vector with a NaN are NaN, with an Inf Inf (dot:
// not finite); axpy only changes the element with the NaN or Inf.
TYPED_TEST(Semantics, Level1PropagatesNanAndInf) {
  using T = TypeParam;
  using R = RealT<T>;
  ForEachTarget([] {
    Random<T> rng(4);
    for (int n : {1, 4, 9, 33, 1000}) {
      for (int i0 : {0, n / 2, n - 1}) {
        for (bool nan : {true, false}) {
          std::vector<T> x = rng.Vector(n), y = rng.Vector(n);
          x[i0] = nan ? NaN<T>() : Inf<T>();
          const R inf = std::numeric_limits<R>::infinity();
          const R nrm2 = Nrm2(n, x.data(), 1), asum = Asum(n, x.data(), 1);
          EXPECT_TRUE(nan ? std::isnan(nrm2) : nrm2 == inf)
              << "nrm2 n=" << n << " i0=" << i0 << ": " << nrm2;
          EXPECT_TRUE(nan ? std::isnan(asum) : asum == inf)
              << "asum n=" << n << " i0=" << i0 << ": " << asum;
          const T dot = Dot(n, x.data(), y.data());
          EXPECT_TRUE(IsSpecial(dot, nan))
              << "dot n=" << n << " i0=" << i0 << ": " << dot;

          const T alpha = MakeScalar<T>(0.5, -1);
          const int inc = 1;
          HWYBLAS_CALL(T, axpy, &n, P(&alpha), P(x.data()), &inc, P(y.data()),
                       &inc);
          for (int i = 0; i < n; ++i) {
            ASSERT_TRUE(i == i0 ? IsSpecial(y[i], nan) : IsFinite(y[i]))
                << "axpy n=" << n << " i0=" << i0 << " y(" << i << ")=" << y[i];
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Subnormal numbers are neither flushed to zero as results nor treated as
// zero as inputs: s * s and sums of a few of them are exact subnormals.

TYPED_TEST(Semantics, SubnormalsAreExact) {
  using T = TypeParam;
  using R = RealT<T>;
  // s^2 = 2^-140 (float) or 2^-1060 (double): subnormal and exact.
  const R s = std::ldexp(R(1), std::is_same_v<R, float> ? -70 : -530);
  const R s2 = s * s;
  ASSERT_EQ(std::fpclassify(s2), FP_SUBNORMAL);
  ForEachTarget([&] {
    for (int n : {1, 5, 64, 200}) {
      const R expected = static_cast<R>(n) * s2;
      ASSERT_EQ(std::fpclassify(expected), FP_SUBNORMAL);
      // dot: subnormal products.
      const std::vector<T> x(n, T(s));
      EXPECT_EQ(Dot(n, x.data(), x.data()), T(expected)) << "dot n=" << n;

      // gemv and gemm: subnormal products.
      const std::vector<T> a(static_cast<size_t>(n) * n, T(s));
      std::vector<T> y(n, T(1));
      const T one(1), zero(0);
      const int inc = 1;
      for (char trans : {'N', 'T'}) {
        HWYBLAS_CALL(T, gemv, &trans, &n, &n, P(&one), P(a.data()), &n,
                     P(x.data()), &inc, P(&zero), P(y.data()), &inc);
        for (int i = 0; i < n; ++i) {
          ASSERT_EQ(y[i], T(expected)) << "gemv trans=" << trans << " n=" << n;
        }
      }
      std::vector<T> c(static_cast<size_t>(n) * n, T(1));
      const char no = 'N';
      HWYBLAS_CALL(T, gemm, &no, &no, &n, &n, &n, P(&one), P(a.data()), &n,
                   P(a.data()), &n, P(&zero), P(c.data()), &n);
      for (const T& v : c) ASSERT_EQ(v, T(expected)) << "gemm n=" << n;

      // axpy and trsm with the identity: subnormal inputs.
      std::vector<T> sub(n, T(s2)), z(n, T(0));
      HWYBLAS_CALL(T, axpy, &n, P(&one), P(sub.data()), &inc, P(z.data()),
                   &inc);
      for (const T& v : z) ASSERT_EQ(v, T(s2)) << "axpy n=" << n;
      std::vector<T> eye(static_cast<size_t>(n) * n, T(0));
      for (int i = 0; i < n; ++i) eye[i + i * n] = T(1);
      std::vector<T> b(static_cast<size_t>(n) * 3, T(s2));
      const int three = 3;
      const char left = 'L', lower = 'L', diag = 'N';
      HWYBLAS_CALL(T, trsm, &left, &lower, &no, &diag, &n, &three, P(&one),
                   P(eye.data()), &n, P(b.data()), &n);
      for (const T& v : b) ASSERT_EQ(v, T(s2)) << "trsm n=" << n;
    }
  });
}

// ---------------------------------------------------------------------------
// The floating-point environment

#if HWYBLAS_TEST_X86
// Calls a few routines that use all kinds of kernels.
template <typename T>
void CallSomeRoutines() {
  Random<T> rng(5);
  const int n = 150, inc = 1;
  const std::vector<T> a = DominantTriangle(n, 'L', rng);
  std::vector<T> b = rng.Vector(static_cast<size_t>(n) * n);
  std::vector<T> c(static_cast<size_t>(n) * n);
  const T one(1), zero(0);
  const char no = 'N', left = 'L', lower = 'L', diag = 'N';
  HWYBLAS_CALL(T, gemm, &no, &no, &n, &n, &n, P(&one), P(a.data()), &n,
               P(b.data()), &n, P(&zero), P(c.data()), &n);
  HWYBLAS_CALL(T, trsm, &left, &lower, &no, &diag, &n, &n, P(&one), P(a.data()),
               &n, P(b.data()), &n);
  HWYBLAS_CALL(T, gemv, &no, &n, &n, P(&one), P(a.data()), &n, P(b.data()),
               &inc, P(&zero), P(c.data()), &inc);
  Nrm2(n * n, b.data(), 1);
  Dot(n * n, b.data(), c.data());
}

// The routines keep the caller's floating-point modes (the control bits of
// MXCSR: rounding, flush-to-zero, denormals-are-zero, exception masks).
TYPED_TEST(Semantics, FloatingPointModesArePreserved) {
  using T = TypeParam;
  constexpr unsigned kControlBits = 0xFFC0;
  const unsigned saved = _mm_getcsr();
  ForEachTarget([&] {
    // Default modes, then flush-to-zero and denormals-are-zero.
    for (unsigned modes :
         {saved & kControlBits, (saved & kControlBits) | 0x8040u}) {
      _mm_setcsr((saved & ~kControlBits) | modes);
      CallSomeRoutines<T>();
      const unsigned after = _mm_getcsr() & kControlBits;
      _mm_setcsr(saved);
      EXPECT_EQ(after, modes) << std::hex << "MXCSR control bits";
    }
  });
}
#endif

// ---------------------------------------------------------------------------
// Reproducibility: repeated calls, and calls with operands at other
// addresses (alignments), give bitwise identical results.

// A copy of `values` at `offset` elements from a 64-byte boundary.
template <typename T>
class OffsetCopy {
 public:
  OffsetCopy(const std::vector<T>& values, int offset)
      : storage_(values.size() + 64 / sizeof(T) + offset + 1) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(storage_.data());
    data_ = reinterpret_cast<T*>((base + 63) & ~uintptr_t{63}) + offset;
    if (!values.empty())
      memcpy(data_, values.data(), values.size() * sizeof(T));
    size_ = values.size();
  }
  T* data() { return data_; }
  std::vector<T> values() const { return std::vector<T>(data_, data_ + size_); }

 private:
  std::vector<T> storage_;
  T* data_;
  size_t size_;
};

template <typename T>
bool BitwiseEqual(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

TYPED_TEST(Semantics, ResultsAreReproducible) {
  using T = TypeParam;
  Random<T> rng(6);
  const T alpha = MakeScalar<T>(0.75, -0.5), beta = MakeScalar<T>(-0.5, 1);
  const int inc = 1;
  // Each case maps an offset to its result (all operands at that offset).
  struct Case {
    const char* name;
    std::function<std::vector<T>(int offset)> run;
  };
  std::vector<Case> cases;
  for (int n : {37, 150}) {
    const auto a = std::make_shared<std::vector<T>>(
        rng.Vector(static_cast<size_t>(n) * n));
    const auto b = std::make_shared<std::vector<T>>(
        rng.Vector(static_cast<size_t>(n) * n));
    const auto c = std::make_shared<std::vector<T>>(
        rng.Vector(static_cast<size_t>(n) * n));
    const auto tri =
        std::make_shared<std::vector<T>>(DominantTriangle(n, 'U', rng));
    cases.push_back(
        {"gemm", [=](int offset) {
           OffsetCopy<T> ca(*a, offset), cb(*b, offset), cc(*c, offset);
           const char no = 'N', t = 'T';
           HWYBLAS_CALL(T, gemm, &no, &t, &n, &n, &n, P(&alpha), P(ca.data()),
                        &n, P(cb.data()), &n, P(&beta), P(cc.data()), &n);
           return cc.values();
         }});
    cases.push_back({"trsm", [=](int offset) {
                       OffsetCopy<T> ca(*tri, offset), cb(*b, offset);
                       const char right = 'R', up = 'U', no = 'N', diag = 'N';
                       HWYBLAS_CALL(T, trsm, &right, &up, &no, &diag, &n, &n,
                                    P(&alpha), P(ca.data()), &n, P(cb.data()),
                                    &n);
                       return cb.values();
                     }});
    cases.push_back({"syrk", [=](int offset) {
                       OffsetCopy<T> ca(*a, offset), cc(*c, offset);
                       const char lo = 'L', no = 'N';
                       HWYBLAS_CALL(T, syrk, &lo, &no, &n, &n, P(&alpha),
                                    P(ca.data()), &n, P(&beta), P(cc.data()),
                                    &n);
                       return cc.values();
                     }});
    for (char trans : {'N', 'T'}) {
      cases.push_back(
          {trans == 'N' ? "gemv N" : "gemv T", [=](int offset) {
             OffsetCopy<T> ca(*a, offset), cx(*b, offset), cy(*c, offset);
             HWYBLAS_CALL(T, gemv, &trans, &n, &n, P(&alpha), P(ca.data()), &n,
                          P(cx.data()), &inc, P(&beta), P(cy.data()), &inc);
             return std::vector<T>(cy.data(), cy.data() + n);
           }});
    }
    const int len = n * n;
    cases.push_back({"dot", [=](int offset) {
                       OffsetCopy<T> cx(*a, offset), cy(*b, offset);
                       return std::vector<T>{Dot(len, cx.data(), cy.data())};
                     }});
    cases.push_back({"nrm2", [=](int offset) {
                       OffsetCopy<T> cx(*a, offset);
                       return std::vector<T>{T(Nrm2(len, cx.data(), 1))};
                     }});
    cases.push_back({"asum", [=](int offset) {
                       OffsetCopy<T> cx(*a, offset);
                       return std::vector<T>{T(Asum(len, cx.data(), 1))};
                     }});
  }
  ForEachTarget([&] {
    for (const Case& c : cases) {
      const std::vector<T> expected = c.run(0);
      EXPECT_TRUE(BitwiseEqual(c.run(0), expected)) << c.name << " repeated";
      for (int offset = 1; offset < static_cast<int>(64 / sizeof(T));
           ++offset) {
        EXPECT_TRUE(BitwiseEqual(c.run(offset), expected))
            << c.name << " at offset " << offset;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Element offsets beyond 2^31: leading dimensions and increments of about
// 2^30 with a few rows and columns. Only the pages that hold elements are
// committed; any other access crashes.

#if INTPTR_MAX > INT32_MAX
class SparseMemory {
 public:
  explicit SparseMemory(size_t bytes) : size_(bytes) {
#if defined(_WIN32)
    base_ = static_cast<char*>(
        VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS));
#else
    void* p = mmap(nullptr, bytes, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    base_ = p == MAP_FAILED ? nullptr : static_cast<char*>(p);
#endif
  }
  ~SparseMemory() {
    if (base_ == nullptr) return;
#if defined(_WIN32)
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, size_);
#endif
  }
  SparseMemory(const SparseMemory&) = delete;
  SparseMemory& operator=(const SparseMemory&) = delete;

  bool ok() const { return base_ != nullptr; }
  char* base() const { return base_; }

  // Makes [offset, offset + bytes) accessible.
  void Commit(size_t offset, size_t bytes) {
#if defined(_WIN32)
    ASSERT_NE(VirtualAlloc(base_ + offset, bytes, MEM_COMMIT, PAGE_READWRITE),
              nullptr);
#else
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t begin = offset / page * page;
    ASSERT_EQ(
        mprotect(base_ + begin, offset + bytes - begin, PROT_READ | PROT_WRITE),
        0);
#endif
  }

 private:
  char* base_ = nullptr;
  [[maybe_unused]] size_t size_;  // for munmap
};

// A rows x cols matrix (or a vector: cols = 1, ld = increment) with
// leading dimension `ld` in its own sparse memory.
template <typename T>
class SparseMatrix {
 public:
  SparseMatrix(const std::vector<T>& compact, int rows, int cols, size_t ld)
      : rows_(rows),
        cols_(cols),
        ld_(ld),
        memory_(((cols - 1) * ld + rows) * sizeof(T)) {
    if (!memory_.ok()) return;
    for (int j = 0; j < cols; ++j) {
      memory_.Commit(j * ld * sizeof(T), rows * sizeof(T));
      memcpy(data() + j * ld, compact.data() + static_cast<size_t>(j) * rows,
             rows * sizeof(T));
    }
  }
  bool ok() const { return memory_.ok(); }
  T* data() const { return reinterpret_cast<T*>(memory_.base()); }
  std::vector<T> Compact() const {
    std::vector<T> v(static_cast<size_t>(rows_) * cols_);
    for (int j = 0; j < cols_; ++j) {
      memcpy(v.data() + static_cast<size_t>(j) * rows_, data() + j * ld_,
             rows_ * sizeof(T));
    }
    return v;
  }

 private:
  int rows_, cols_;
  size_t ld_;
  SparseMemory memory_;
};

// Expects results computed with huge strides to match the compact ones.
template <typename T>
void ExpectSame(const std::vector<T>& actual, const std::vector<T>& expected,
                const char* what) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_LE(Abs(actual[i] - expected[i]),
              1e3 * Eps<T>() * (1 + Abs(expected[i])))
        << what << " element " << i;
  }
}

template <typename T>
class LargeOffsets : public ::testing::Test {};
using LargeOffsetTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(LargeOffsets, LargeOffsetTypes);

// 2^30 (a multiple of 1 KiB, as in power-of-two sized matrices) and an odd
// number; with 3 or more columns the offsets exceed 2^31.
constexpr int kHugeLds[] = {1 << 30, (1 << 30) + 3};

TYPED_TEST(LargeOffsets, Level3) {
  using T = TypeParam;
  Random<T> rng(7);
  const int m = 5, n = 4, k = 3;
  const T alpha = MakeScalar<T>(0.5, 0.25), beta = MakeScalar<T>(-1, 0.5);
  ForEachTarget([&] {
    for (int ld : kHugeLds) {
      for (char ta : {'N', 'T'}) {
        for (char tb : {'N', 'T'}) {
          const int ar = ta == 'N' ? m : k, ac = ta == 'N' ? k : m;
          const int br = tb == 'N' ? k : n, bc = tb == 'N' ? n : k;
          const std::vector<T> a = rng.Vector(ar * ac), b = rng.Vector(br * bc);
          std::vector<T> c = rng.Vector(m * n);
          SparseMatrix<T> sa(a, ar, ac, ld), sb(b, br, bc, ld), sc(c, m, n, ld);
          ASSERT_TRUE(sa.ok() && sb.ok() && sc.ok());
          HWYBLAS_CALL(T, gemm, &ta, &tb, &m, &n, &k, P(&alpha), P(sa.data()),
                       &ld, P(sb.data()), &ld, P(&beta), P(sc.data()), &ld);
          HWYBLAS_CALL(T, gemm, &ta, &tb, &m, &n, &k, P(&alpha), P(a.data()),
                       &ar, P(b.data()), &br, P(&beta), P(c.data()), &m);
          ExpectSame(sc.Compact(), c, "gemm");
        }
      }
      for (char side : {'L', 'R'}) {
        for (char uplo : {'U', 'L'}) {
          const int ka = side == 'L' ? m : n;
          const std::vector<T> a = DominantTriangle(ka, uplo, rng);
          std::vector<T> b = rng.Vector(m * n);
          SparseMatrix<T> sa(a, ka, ka, ld), sb(b, m, n, ld);
          ASSERT_TRUE(sa.ok() && sb.ok());
          const char trans = 'T', diag = 'N';
          for (int solve = 0; solve < 2; ++solve) {
            if (solve) {
              HWYBLAS_CALL(T, trsm, &side, &uplo, &trans, &diag, &m, &n,
                           P(&alpha), P(sa.data()), &ld, P(sb.data()), &ld);
              HWYBLAS_CALL(T, trsm, &side, &uplo, &trans, &diag, &m, &n,
                           P(&alpha), P(a.data()), &ka, P(b.data()), &m);
            } else {
              HWYBLAS_CALL(T, trmm, &side, &uplo, &trans, &diag, &m, &n,
                           P(&alpha), P(sa.data()), &ld, P(sb.data()), &ld);
              HWYBLAS_CALL(T, trmm, &side, &uplo, &trans, &diag, &m, &n,
                           P(&alpha), P(a.data()), &ka, P(b.data()), &m);
            }
            ExpectSame(sb.Compact(), b, solve ? "trsm" : "trmm");
          }
        }
      }
      {
        const std::vector<T> a = rng.Vector(m * k);
        std::vector<T> c = rng.Vector(m * m);
        SparseMatrix<T> sa(a, m, k, ld), sc(c, m, m, ld);
        ASSERT_TRUE(sa.ok() && sc.ok());
        const char uplo = 'U', no = 'N';
        HWYBLAS_CALL(T, syrk, &uplo, &no, &m, &k, P(&alpha), P(sa.data()), &ld,
                     P(&beta), P(sc.data()), &ld);
        HWYBLAS_CALL(T, syrk, &uplo, &no, &m, &k, P(&alpha), P(a.data()), &m,
                     P(&beta), P(c.data()), &m);
        ExpectSame(sc.Compact(), c, "syrk");
      }
    }
  });
}

TYPED_TEST(LargeOffsets, Level2) {
  using T = TypeParam;
  Random<T> rng(8);
  const int m = 6, n = 4;
  const T alpha = MakeScalar<T>(0.5, -0.25), beta = MakeScalar<T>(2, 0.5);
  ForEachTarget([&] {
    for (int ld : kHugeLds) {
      // Huge lda, and huge increments (positive and negative).
      for (int inc : {1, ld, -ld}) {
        const int one = 1;
        for (char trans : {'N', 'T'}) {
          const int lx = trans == 'N' ? n : m, ly = trans == 'N' ? m : n;
          const std::vector<T> a = rng.Vector(m * n), x = rng.Vector(lx);
          std::vector<T> y = rng.Vector(ly);
          const size_t step = std::abs(inc);
          SparseMatrix<T> sa(a, m, n, ld), sx(x, 1, lx, step),
              sy(y, 1, ly, step);
          ASSERT_TRUE(sa.ok() && sx.ok() && sy.ok());
          HWYBLAS_CALL(T, gemv, &trans, &m, &n, P(&alpha), P(sa.data()), &ld,
                       P(sx.data()), &inc, P(&beta), P(sy.data()), &inc);
          // With a negative increment, the vectors are traversed backwards.
          std::vector<T> xr = x, yr = y;
          if (inc < 0) {
            std::reverse(xr.begin(), xr.end());
            std::reverse(yr.begin(), yr.end());
          }
          HWYBLAS_CALL(T, gemv, &trans, &m, &n, P(&alpha), P(a.data()), &m,
                       P(xr.data()), &one, P(&beta), P(yr.data()), &one);
          if (inc < 0) std::reverse(yr.begin(), yr.end());
          ExpectSame(sy.Compact(), yr, "gemv");
        }
        {
          const std::vector<T> x = rng.Vector(m), y = rng.Vector(n);
          std::vector<T> a = rng.Vector(m * n);
          const size_t step = std::abs(inc);
          SparseMatrix<T> sa(a, m, n, ld), sx(x, 1, m, step), sy(y, 1, n, step);
          ASSERT_TRUE(sa.ok() && sx.ok() && sy.ok());
          std::vector<T> xr = x, yr = y;
          if (inc < 0) {
            std::reverse(xr.begin(), xr.end());
            std::reverse(yr.begin(), yr.end());
          }
          if constexpr (IsComplex<T>::value) {
            zgeru_(&m, &n, P(&alpha), P(sx.data()), &inc, P(sy.data()), &inc,
                   P(sa.data()), &ld);
            zgeru_(&m, &n, P(&alpha), P(xr.data()), &one, P(yr.data()), &one,
                   P(a.data()), &m);
          } else {
            dger_(&m, &n, &alpha, sx.data(), &inc, sy.data(), &inc, sa.data(),
                  &ld);
            dger_(&m, &n, &alpha, xr.data(), &one, yr.data(), &one, a.data(),
                  &m);
          }
          ExpectSame(sa.Compact(), a, "ger");
        }
        for (char uplo : {'U', 'L'}) {
          const std::vector<T> a = DominantTriangle(m, uplo, rng);
          std::vector<T> x = rng.Vector(m);
          const size_t step = std::abs(inc);
          SparseMatrix<T> sa(a, m, m, ld), sx(x, 1, m, step);
          ASSERT_TRUE(sa.ok() && sx.ok());
          std::vector<T> xr = x;
          if (inc < 0) std::reverse(xr.begin(), xr.end());
          const char trans = 'N', diag = 'N';
          HWYBLAS_CALL(T, trsv, &uplo, &trans, &diag, &m, P(sa.data()), &ld,
                       P(sx.data()), &inc);
          HWYBLAS_CALL(T, trsv, &uplo, &trans, &diag, &m, P(a.data()), &m,
                       P(xr.data()), &one);
          if (inc < 0) std::reverse(xr.begin(), xr.end());
          ExpectSame(sx.Compact(), xr, "trsv");
        }
      }
    }
  });
}

TYPED_TEST(LargeOffsets, Level1) {
  using T = TypeParam;
  Random<T> rng(9);
  const int n = 5;
  const T alpha = MakeScalar<T>(-0.5, 0.75);
  ForEachTarget([&] {
    for (int inc : {kHugeLds[0], kHugeLds[1], -kHugeLds[1]}) {
      const size_t step = std::abs(inc);
      const int one = 1;
      const std::vector<T> x = rng.Vector(n);
      std::vector<T> y = rng.Vector(n);
      std::vector<T> xr = x, yr = y;
      if (inc < 0) {
        std::reverse(xr.begin(), xr.end());
        std::reverse(yr.begin(), yr.end());
      }
      SparseMatrix<T> sx(x, 1, n, step), sy(y, 1, n, step);
      ASSERT_TRUE(sx.ok() && sy.ok());
      HWYBLAS_CALL(T, axpy, &n, P(&alpha), P(sx.data()), &inc, P(sy.data()),
                   &inc);
      HWYBLAS_CALL(T, axpy, &n, P(&alpha), P(xr.data()), &one, P(yr.data()),
                   &one);
      if (inc < 0) std::reverse(yr.begin(), yr.end());
      ExpectSame(sy.Compact(), yr, "axpy");
      // dot: both libraries' order is the same.
      ExpectSame(std::vector<T>{[&] {
                   if constexpr (IsComplex<T>::value) {
                     T r;
                     cblas_zdotu_sub(n, sx.data(), inc, sy.data(), inc, &r);
                     return r;
                   } else {
                     return ddot_(&n, sx.data(), &inc, sy.data(), &inc);
                   }
                 }()},
                 std::vector<T>{[&] {
                   std::vector<T> xc = sx.Compact(), yc = sy.Compact();
                   if (inc < 0) {
                     std::reverse(xc.begin(), xc.end());
                     std::reverse(yc.begin(), yc.end());
                   }
                   return Dot(n, xc.data(), yc.data());
                 }()},
                 "dot");
      if (inc > 0) {  // nrm2 and asum: positive increments
        ExpectSame(std::vector<T>{T(Nrm2(n, sx.data(), inc))},
                   std::vector<T>{T(Nrm2(n, x.data(), 1))}, "nrm2");
        ExpectSame(std::vector<T>{T(Asum(n, sx.data(), inc))},
                   std::vector<T>{T(Asum(n, x.data(), 1))}, "asum");
        HWYBLAS_CALL(T, scal, &n, P(&alpha), P(sx.data()), &inc);
        std::vector<T> xs = x;
        HWYBLAS_CALL(T, scal, &n, P(&alpha), P(xs.data()), &one);
        ExpectSame(sx.Compact(), xs, "scal");
      }
    }
  });
}
#endif  // INTPTR_MAX > INT32_MAX

}  // namespace
}  // namespace hwyblas_test
