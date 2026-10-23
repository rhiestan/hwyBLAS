// Tests of the Schur factorization xGEES and its steps: balancing (xGEBAL,
// xGEBAK), the Hessenberg reduction (xGEHRD, xGEHD2, xLAHR2, xORGHR /
// xUNGHR, xORMHR / xUNMHR), the Hessenberg QR algorithm (xHSEQR, xLAQR0 -
// xLAQR5, xLAHQR, xLANV2), the reordering of the Schur form (xTRSEN,
// xTREXC, xLAEXC, xTRSYL, xLASY2, xLALN2, xLACN2), xLARFX, the complex
// helpers (xLARTG, xROT, IxMAX1, xSUM1 for complex T), and LAPACKE.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/lapack.h"
#include "hwyblas/lapacke.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class GeesTest : public ::testing::Test {};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GeesTest, ScalarTypes);

template <typename R>
class GeesRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GeesRealTest, RealTypes);

template <typename T>
class GeesComplexTest : public ::testing::Test {};

using ComplexTypes =
    ::testing::Types<std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(GeesComplexTest, ComplexTypes);

// Calls the real routine `rname` or the complex routine `cname`.
#define CALL_RC(T, rname, cname, ...)                              \
  do {                                                             \
    if constexpr (std::is_same_v<T, float>) {                      \
      s##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, double>) {              \
      d##rname##_(__VA_ARGS__);                                    \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      c##cname##_(__VA_ARGS__);                                    \
    } else {                                                       \
      z##cname##_(__VA_ARGS__);                                    \
    }                                                              \
  } while (0)

// The eigenvalues, as complex numbers also for real T.
template <typename T>
using Eig = std::complex<RealT<T>>;

size_t Sz(hwyblas_int n) {
  return static_cast<size_t>(std::max<hwyblas_int>(1, n));
}

size_t Sz2(hwyblas_int m, hwyblas_int n) { return Sz(m) * Sz(n); }

// Forces the parameters of xHSEQR (ILAENV ispec 12 - 16: the crossover
// point NMIN to xLAQR0, the deflation window NW, NIBBLE, the number of
// shifts NS and KACC22) while it exists; 0 keeps ILAENV's choice.
class ScopedHseqrParameters {
 public:
  ScopedHseqrParameters(hwyblas_int nmin, hwyblas_int nw, hwyblas_int nibble,
                        hwyblas_int ns, hwyblas_int kacc22) {
    values_[0] = nmin;
    values_[1] = nw;
    values_[2] = nibble;
    values_[3] = ns;
    values_[4] = kacc22;
    previous_ = hwyblas_set_ilaenv(&Ilaenv);
  }
  ~ScopedHseqrParameters() {
    hwyblas_set_ilaenv(previous_);
    for (hwyblas_int& v : values_) v = -1;
  }
  ScopedHseqrParameters(const ScopedHseqrParameters&) = delete;
  ScopedHseqrParameters& operator=(const ScopedHseqrParameters&) = delete;

 private:
  static hwyblas_int Ilaenv(const hwyblas_int* ispec, const char* name,
                            const char* opts, const hwyblas_int* n1,
                            const hwyblas_int* n2, const hwyblas_int* n3,
                            const hwyblas_int* n4, size_t name_len,
                            size_t opts_len) {
    if (*ispec >= 12 && *ispec <= 16 && values_[*ispec - 12] >= 0) {
      // NIBBLE and KACC22 may be 0; the others must be positive.
      const hwyblas_int v = values_[*ispec - 12];
      if (v > 0 || *ispec == 14 || *ispec == 16) return v;
    }
    return ilaenv_(ispec, name, opts, n1, n2, n3, n4, name_len, opts_len);
  }

  static inline hwyblas_int values_[5] = {-1, -1, -1, -1, -1};
  hwyblas_ilaenv_function previous_ = nullptr;
};

// The parameter sets of the tests: ILAENV's own, and ones that use xLAQR0
// (and xLAQR4 in the deflation window) for small matrices, with all three
// KACC22 modes (of the input file of LAPACK's test program, nep.in); the
// last one uses xLAQR4's multishift code (with xLAQR2) in the deflation
// window for N = 50.
struct HseqrParameters {
  hwyblas_int nmin, nw, nibble, ns, kacc22;
};
const HseqrParameters kHseqrParameters[] = {
    {-1, -1, -1, -1, -1}, {11, 2, 0, 1, 0},  {12, 3, 5, 2, 1},
    {11, 5, 7, 4, 2},     {15, 3, 3, 2, 0},  {11, 2, 200, 1, 1},
    {11, 12, 0, 6, 2},    {11, 16, 3, 2, 1},
};

// ---------------------------------------------------------------------------
// Test matrices

enum class Kind {
  kRandom,
  kZero,
  kIdentity,
  kJordan,      // a Jordan block for 1
  kCyclic,      // the cyclic permutation (eigenvalues: roots of unity)
  kTriangular,  // upper triangular
  kGraded,      // random, A(i,j) scaled by 2**(j - i)
  kPairs,       // block diagonal with 2 x 2 rotations (complex pairs)
};
const Kind kKinds[] = {Kind::kRandom, Kind::kZero,   Kind::kIdentity,
                       Kind::kJordan, Kind::kCyclic, Kind::kTriangular,
                       Kind::kGraded, Kind::kPairs};

template <typename T>
std::vector<T> TestMatrix(Kind kind, hwyblas_int n, hwyblas_int lda,
                          Random<T>& rng) {
  using R = RealT<T>;
  std::vector<T> a = rng.Vector(Sz2(lda, n));
  const T zero = T(0);
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = 0; i < n; ++i) {
      T& x = At(a, i, j, lda);
      switch (kind) {
        case Kind::kRandom:
          break;
        case Kind::kZero:
          x = zero;
          break;
        case Kind::kIdentity:
          x = T(R(i == j ? 1 : 0));
          break;
        case Kind::kJordan:
          x = T(R(i == j || i + 1 == j ? 1 : 0));
          break;
        case Kind::kCyclic:
          x = T(R(i == j + 1 || (i == 0 && j == n - 1) ? 1 : 0));
          break;
        case Kind::kTriangular:
          if (i > j) x = zero;
          break;
        case Kind::kGraded:
          x *= static_cast<R>(std::ldexp(1.0, j - i));
          break;
        case Kind::kPairs: {
          const hwyblas_int bi = i / 2, bj = j / 2;
          if (bi != bj) {
            x = zero;
          } else if (i == j) {
            x = T(static_cast<R>(bi + 1));
          } else {
            x = T(static_cast<R>(i < j ? 2 : -3));
          }
          break;
        }
      }
    }
  }
  return a;
}

const char* KindName(Kind kind) {
  switch (kind) {
    case Kind::kRandom:
      return "random";
    case Kind::kZero:
      return "zero";
    case Kind::kIdentity:
      return "identity";
    case Kind::kJordan:
      return "jordan";
    case Kind::kCyclic:
      return "cyclic";
    case Kind::kTriangular:
      return "triangular";
    case Kind::kGraded:
      return "graded";
    case Kind::kPairs:
      return "pairs";
  }
  return "";
}

// Zeroes a below its first subdiagonal (upper Hessenberg).
template <typename T>
void MakeHessenberg(std::vector<T>& a, hwyblas_int n, hwyblas_int lda) {
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j + 2; i < n; ++i) At(a, i, j, lda) = T(0);
  }
}

// ---------------------------------------------------------------------------
// Checks

// Checks that T (n x n) is in Schur form, upper triangular (complex T) or
// upper quasi-triangular with 2 x 2 blocks in standard form (real T: equal
// diagonal elements and off-diagonal elements of opposite signs), and that
// w are its eigenvalues in the order of the diagonal (rows and columns
// lo:hi, 0-based, of T; outside them T only needs to be upper triangular).
template <typename T>
void ExpectSchurForm(const std::vector<T>& t, hwyblas_int n, hwyblas_int ld,
                     const std::vector<Eig<T>>& w, hwyblas_int lo = 0,
                     hwyblas_int hi = -1) {
  using R = RealT<T>;
  if (hi < 0) hi = n - 1;
  for (hwyblas_int j = 0; j < n; ++j) {
    for (hwyblas_int i = j + 2; i < n; ++i) {
      ASSERT_EQ(At(t, i, j, ld), T(0)) << i << ", " << j;
    }
  }
  hwyblas_int i = 0;
  while (i < n) {
    const bool block =
        !IsComplex<T>::value && i + 1 < n && At(t, i + 1, i, ld) != T(0);
    if (!block) {
      if (i + 1 < n) {
        ASSERT_EQ(At(t, i + 1, i, ld), T(0)) << i;
      }
      ASSERT_EQ(w[i], Eig<T>(At(t, i, i, ld))) << i;
      ++i;
      continue;
    }
    ASSERT_TRUE(i >= lo && i + 1 <= hi) << i;
    const R a = std::real(At(t, i, i, ld)), b = std::real(At(t, i, i + 1, ld));
    const R c = std::real(At(t, i + 1, i, ld));
    const R d = std::real(At(t, i + 1, i + 1, ld));
    ASSERT_EQ(a, d) << i;
    ASSERT_TRUE((b > 0 && c < 0) || (b < 0 && c > 0)) << i;
    if (i + 2 < n) {
      ASSERT_EQ(At(t, i + 2, i + 1, ld), T(0)) << i;
    }
    const R im = std::sqrt(std::abs(b)) * std::sqrt(std::abs(c));
    const R tol = 8 * std::numeric_limits<R>::epsilon() * (std::abs(a) + im);
    EXPECT_EQ(w[i].real(), a) << i;
    EXPECT_EQ(w[i + 1].real(), a) << i;
    EXPECT_NEAR(w[i].imag(), im, tol) << i;
    EXPECT_EQ(w[i + 1].imag(), -w[i].imag()) << i;
    i += 2;
  }
}

// ||A - Z * T * Z**H||_1 / (n ||A||_1 eps) and ||I - Z**H Z|| / (n eps).
template <typename T>
void ExpectSchurFactorization(const std::vector<T>& a, const std::vector<T>& z,
                              const std::vector<T>& t, hwyblas_int n,
                              hwyblas_int lda, hwyblas_int ldz,
                              hwyblas_int ldt) {
  if (n == 0) return;
  const Dense<T> ad = ToDense(a, n, n, lda);
  const Dense<T> zd = ToDense(z, n, n, ldz);
  const Dense<T> td = ToDense(t, n, n, ldt);
  EXPECT_LT(
      ResidualRatio(ad, Mul(Mul(zd, td), Adjoint(zd)), static_cast<double>(n)),
      kLapackThreshold);
  EXPECT_LT(OrthogonalityRatio(zd), kLapackThreshold);
}

// ---------------------------------------------------------------------------
// Calls with vectors; the eigenvalues as complex numbers.

template <typename T>
hwyblas_int Hseqr(char job, char compz, hwyblas_int n, hwyblas_int ilo,
                  hwyblas_int ihi, std::vector<T>& h, hwyblas_int ldh,
                  std::vector<Eig<T>>& w, std::vector<T>& z, hwyblas_int ldz,
                  hwyblas_int lwork) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  hwyblas_int info = -99;
  w.assign(Sz(n), Eig<T>(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, hseqr, &job, &compz, &n, &ilo, &ihi, P(h.data()),
                         &ldh, P(w.data()), P(z.data()), &ldz, P(work.data()),
                         &lwork, &info);
  } else {
    std::vector<R> wr(Sz(n), NAN), wi(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, hseqr, &job, &compz, &n, &ilo, &ihi, h.data(), &ldh,
                      wr.data(), wi.data(), z.data(), &ldz, work.data(), &lwork,
                      &info);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
  }
  w.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (lwork == -1) w.assign(1, Eig<T>(std::real(work[0]), 0));
  return info;
}

// SELECT functions of xGEES: eigenvalues with a positive real part, or
// none, or all.
template <typename R>
hwyblas_int PositiveReal(const R* wr, const R* wi) {
  (void)wi;
  return *wr > 0 ? 1 : 0;
}
template <typename R>
hwyblas_int PositiveRealC(const R* w) {
  return w[0] > 0 ? 1 : 0;
}
template <typename R>
hwyblas_int PositiveImag(const R* wr, const R* wi) {
  (void)wr;
  return *wi > 0 ? 1 : 0;
}
template <typename R>
hwyblas_int PositiveImagC(const R* w) {
  return w[1] > 0 ? 1 : 0;
}
template <typename R>
hwyblas_int Never(const R* /*wr*/, const R* /*wi*/) {
  return 0;
}
template <typename R>
hwyblas_int NeverC(const R* /*w*/) {
  return 0;
}
template <typename R>
hwyblas_int Always(const R* /*wr*/, const R* /*wi*/) {
  return 1;
}
template <typename R>
hwyblas_int AlwaysC(const R* /*w*/) {
  return 1;
}

// kPositiveImag selects only one eigenvalue of each complex conjugate pair
// of a real matrix: xGEES then selects both.
enum class Select { kPositiveReal, kPositiveImag, kNone, kAll };

// Whether eigenvalue i of w (in xGEES's order) is selected; for real T also
// if the other eigenvalue of its pair is.
template <typename T>
bool Selected(Select select, const std::vector<Eig<T>>& w, size_t i) {
  const auto one = [&](size_t k) {
    switch (select) {
      case Select::kPositiveReal:
        return w[k].real() > 0;
      case Select::kPositiveImag:
        return w[k].imag() > 0;
      case Select::kNone:
        return false;
      case Select::kAll:
        return true;
    }
    return false;
  };
  if (IsComplex<T>::value || w[i].imag() == 0) return one(i);
  const size_t other = w[i].imag() > 0 ? i + 1 : i - 1;
  return one(i) || one(other);
}

// xGEES: returns INFO; sdim, w, vs as computed.
template <typename T>
hwyblas_int Gees(char jobvs, char sort, Select select, hwyblas_int n,
                 std::vector<T>& a, hwyblas_int lda, hwyblas_int* sdim,
                 std::vector<Eig<T>>& w, std::vector<T>& vs, hwyblas_int ldvs,
                 hwyblas_int lwork) {
  using R = RealT<T>;
  std::vector<T> work(Sz(lwork), NaN<T>());
  std::vector<hwyblas_int> bwork(Sz(n), -7);
  hwyblas_int info = -99;
  w.assign(Sz(n), Eig<T>(NAN, NAN));
  if constexpr (IsComplex<T>::value) {
    hwyblas_int (*sel)(const R*) =
        select == Select::kPositiveReal   ? &PositiveRealC<R>
        : select == Select::kPositiveImag ? &PositiveImagC<R>
        : select == Select::kNone         ? &NeverC<R>
                                          : &AlwaysC<R>;
    std::vector<R> rwork(Sz(n), NAN);
    HWYBLAS_CALL_COMPLEX(T, gees, &jobvs, &sort, sel, &n, P(a.data()), &lda,
                         sdim, P(w.data()), P(vs.data()), &ldvs, P(work.data()),
                         &lwork, rwork.data(), bwork.data(), &info);
  } else {
    hwyblas_int (*sel)(const R*, const R*) =
        select == Select::kPositiveReal   ? &PositiveReal<R>
        : select == Select::kPositiveImag ? &PositiveImag<R>
        : select == Select::kNone         ? &Never<R>
                                          : &Always<R>;
    std::vector<R> wr(Sz(n), NAN), wi(Sz(n), NAN);
    HWYBLAS_CALL_REAL(T, gees, &jobvs, &sort, sel, &n, a.data(), &lda, sdim,
                      wr.data(), wi.data(), vs.data(), &ldvs, work.data(),
                      &lwork, bwork.data(), &info);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
  }
  w.resize(static_cast<size_t>(std::max<hwyblas_int>(0, n)));
  if (lwork == -1) w.assign(1, Eig<T>(std::real(work[0]), 0));
  return info;
}

// The minimum LWORK of xGEES.
template <typename T>
hwyblas_int MinimumGeesLwork(hwyblas_int n) {
  return std::max<hwyblas_int>(1, (IsComplex<T>::value ? 2 : 3) * n);
}

// ---------------------------------------------------------------------------
// Balancing

// xGEBAL and xGEBAK: the balanced matrix A' = D**-1 * P**T * A * P * D
// (exact: the scale factors are powers of 2), with A'(i,j) = 0 for i > j
// and j < ILO or i > IHI; X = xGEBAK('R', I) = P * D and xGEBAK('L', I) =
// P * D**-1, so A * X = X * A' and X_L**T * X = I.
TYPED_TEST(GeesTest, Gebal) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(7);
  Random<R> rngr(8);
  for (const hwyblas_int n : {0, 1, 2, 5, 12, 23}) {
    for (const char job : {'N', 'P', 'S', 'B'}) {
      for (const int structure : {0, 1, 2}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " job=" << job
                                          << " structure=" << structure);
        const hwyblas_int lda = n + 1;
        std::vector<T> a = rng.Vector(Sz2(lda, n));
        // Structure 1: a permuted block upper triangular matrix with
        // isolated eigenvalues at both ends; 2: also badly scaled.
        if (structure > 0 && n > 0) {
          std::vector<hwyblas_int> perm(static_cast<size_t>(n));
          for (hwyblas_int i = 0; i < n; ++i) perm[i] = i;
          for (hwyblas_int i = n - 1; i > 0; --i) {
            std::swap(perm[i],
                      perm[static_cast<size_t>(std::abs(rngr.Next()) * i)]);
          }
          const hwyblas_int k1 = n / 3, k2 = n - n / 4;
          std::vector<T> b = a;
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              const bool zero = (i > j && (j < k1 || i >= k2)) ||
                                (i >= k1 && i < k2 && j < k1);
              T x = zero ? T(0) : At(a, i, j, lda);
              if (structure == 2) {
                x *= static_cast<R>(std::ldexp(1.0, 6 * (j % 4) - 6 * (i % 3)));
              }
              At(b, perm[i], perm[j], lda) = x;
            }
          }
          a = b;
        }
        std::vector<T> ab = a;
        std::vector<R> scale(Sz(n), NAN);
        hwyblas_int ilo = -1, ihi = -1, info = -99;
        HWYBLAS_CALL(T, gebal, &job, &n, P(ab.data()), &lda, &ilo, &ihi,
                     scale.data(), &info);
        ASSERT_EQ(info, 0);
        if (n == 0) {
          EXPECT_EQ(ilo, 1);
          EXPECT_EQ(ihi, 0);
          continue;
        }
        ASSERT_GE(ilo, 1);
        ASSERT_LE(ihi, n);
        ASSERT_LE(ilo, std::max<hwyblas_int>(ihi, 1));
        if (job == 'N' || job == 'S') {
          EXPECT_EQ(ilo, 1);
          EXPECT_EQ(ihi, n);
        }
        if (structure > 0 && (job == 'P' || job == 'B')) {
          EXPECT_GE(ilo - 1, n / 3);
          EXPECT_GE(n - ihi, n / 4);
        }
        if (job == 'N') {
          EXPECT_EQ(ab, a);
        }
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = j + 1; i < n; ++i) {
            if (j < ilo - 1 || i > ihi - 1) {
              ASSERT_EQ(At(ab, i, j, lda), T(0)) << i << ", " << j;
            }
          }
        }
        // X = P * D and X_L = P * D**-1.
        std::vector<T> x(Sz2(n, n), T(0)), xl(Sz2(n, n), T(0));
        for (hwyblas_int i = 0; i < n; ++i)
          At(x, i, i, n) = At(xl, i, i, n) = T(1);
        HWYBLAS_CALL(T, gebak, &job, "R", &n, &ilo, &ihi, scale.data(), &n,
                     P(x.data()), &n, &info);
        ASSERT_EQ(info, 0);
        HWYBLAS_CALL(T, gebak, &job, "L", &n, &ilo, &ihi, scale.data(), &n,
                     P(xl.data()), &n, &info);
        ASSERT_EQ(info, 0);
        const Dense<T> ad = ToDense(a, n, n, lda), abd = ToDense(ab, n, n, lda);
        const Dense<T> xd = ToDense(x, n, n, n), xld = ToDense(xl, n, n, n);
        const Dense<T> lhs = Mul(ad, xd), rhs = Mul(xd, abd);
        for (size_t k = 0; k < lhs.v.size(); ++k)
          ASSERT_EQ(lhs.v[k], rhs.v[k]) << k;
        const Dense<T> id = Mul(Adjoint(xld), xd);
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i < n; ++i) {
            ASSERT_EQ(id(i, j), RefT<T>(i == j ? 1 : 0)) << i << ", " << j;
          }
        }
        // Balancing converged: for each i in ILO:IHI, scaling row and
        // column i by 2 or 1/2 would not reduce c + r (the 2-norms of
        // column and row i in rows and columns ILO:IHI) by 5%, so c/r is in
        // [0.43, 2.33] (or either is zero).
        if (job == 'S' || job == 'B') {
          for (hwyblas_int i = ilo - 1; i <= ihi - 1; ++i) {
            double c = 0, r = 0;
            for (hwyblas_int k = ilo - 1; k <= ihi - 1; ++k) {
              c += std::norm(std::complex<double>(At(ab, k, i, lda)));
              r += std::norm(std::complex<double>(At(ab, i, k, lda)));
            }
            if (c == 0 || r == 0) continue;
            EXPECT_GE(std::sqrt(c / r), 0.42) << i;
            EXPECT_LE(std::sqrt(c / r), 2.4) << i;
          }
        }
        // Scale factors are powers of 2, permutation indices integers.
        for (hwyblas_int i = 0; i < n; ++i) {
          if (i >= ilo - 1 && i <= ihi - 1) {
            int e = 0;
            EXPECT_EQ(std::frexp(scale[i], &e), R(0.5)) << i;
          } else {
            EXPECT_EQ(scale[i], std::round(scale[i])) << i;
          }
        }
      }
    }
  }
}

// xGEBAL: a NaN in the part to scale gives INFO = -3 (and an error report).
TYPED_TEST(GeesTest, GebalNaN) {
  using T = TypeParam;
  using R = RealT<T>;
  const hwyblas_int n = 4;
  std::vector<T> a = Random<T>(3).Vector(16);
  a[5] = NaN<T>();
  std::vector<R> scale(n);
  hwyblas_int ilo = 0, ihi = 0, info = 0;
  ErrorCapture errors;
  HWYBLAS_CALL(T, gebal, "B", &n, P(a.data()), &n, &ilo, &ihi, scale.data(),
               &info);
  EXPECT_EQ(info, -3);
  EXPECT_EQ(errors.Take(), std::vector<std::string>{Name<T>("GEBAL") + ":3"});
}

// ---------------------------------------------------------------------------
// Hessenberg reduction

struct Blocking {
  hwyblas_int nb, nx;
};
const Blocking kBlockings[] = {{0, 0}, {1, 0}, {2, 0}, {3, 4}, {8, 0}, {5, 7}};

// xGEHRD with xORGHR / xUNGHR and xORMHR / xUNMHR: A = Q * H * Q**H with
// upper Hessenberg H (in rows and columns ILO:IHI), with blocked and
// unblocked code, reduced workspaces and the unblocked xGEHD2.
TYPED_TEST(GeesTest, Hessenberg) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(29);
    for (const Blocking& b : kBlockings) {
      const ScopedBlockSize block_size(b.nb, b.nx);
      for (const hwyblas_int n : {0, 1, 2, 3, 8, 17, 40}) {
        for (const int range : {0, 1}) {
          for (const int variant : {0, 1, 2}) {  // optimal, minimal, xGEHD2
            const hwyblas_int ilo = range == 0 || n < 4 ? 1 : 2;
            const hwyblas_int ihi = range == 0 || n < 4 ? n : n - 1;
            SCOPED_TRACE(::testing::Message()
                         << "n=" << n << " ilo=" << ilo << " ihi=" << ihi
                         << " nb=" << b.nb << " nx=" << b.nx
                         << " variant=" << variant);
            const hwyblas_int lda = n + 2;
            // Upper triangular outside rows and columns ilo:ihi, as after
            // xGEBAL.
            std::vector<T> a = rng.Vector(Sz2(lda, n));
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = j + 1; i < n; ++i) {
                if (j < ilo - 1 || i > ihi - 1) At(a, i, j, lda) = T(0);
              }
            }
            std::vector<T> h = a;
            std::vector<T> tau(Sz(n), NaN<T>());
            hwyblas_int lwork =
                variant == 1 ? std::max<hwyblas_int>(1, n) : n * 64 + 65 * 64;
            std::vector<T> work(Sz(lwork), NaN<T>());
            hwyblas_int info = -99;
            if (variant == 2) {
              HWYBLAS_CALL(T, gehd2, &n, &ilo, &ihi, P(h.data()), &lda,
                           P(tau.data()), P(work.data()), &info);
            } else {
              HWYBLAS_CALL(T, gehrd, &n, &ilo, &ihi, P(h.data()), &lda,
                           P(tau.data()), P(work.data()), &lwork, &info);
            }
            ASSERT_EQ(info, 0);
            // xGEHRD sets TAU(1:ILO-1) and TAU(IHI:N-1) to zero.
            for (hwyblas_int i = 0; i + 1 < n && variant != 2; ++i) {
              if (i < ilo - 1 || i >= ihi - 1) {
                ASSERT_EQ(tau[i], T(0)) << i;
              }
            }
            const std::vector<T> reflectors = h;
            std::vector<T> q = h;
            CALL_RC(T, orghr, unghr, &n, &ilo, &ihi, P(q.data()), &lda,
                    P(tau.data()), P(work.data()), &lwork, &info);
            ASSERT_EQ(info, 0);
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = j + 2; i < n; ++i) At(h, i, j, lda) = T(0);
            }
            ExpectSchurFactorization(a, q, h, n, lda, lda, lda);

            // Q * C, Q**H * C, C * Q and C * Q**H with xORMHR / xUNMHR.
            const char ct = IsComplex<T>::value ? 'C' : 'T';
            for (const char side : {'L', 'R'}) {
              for (const char trans : {'N', ct}) {
                const hwyblas_int m = side == 'L' ? n : 3;
                const hwyblas_int nc = side == 'L' ? 4 : n;
                const hwyblas_int ldc = m + 1;
                const std::vector<T> c = rng.Vector(Sz2(ldc, nc));
                std::vector<T> qc = c;
                // The minimum workspace, or the optimal one.
                hwyblas_int lw = variant == 1 ? std::max<hwyblas_int>(
                                                    1, side == 'L' ? nc : m)
                                              : 64 * (n + 4);
                std::vector<T> w2(Sz(lw), NaN<T>());
                CALL_RC(T, ormhr, unmhr, &side, &trans, &m, &nc, &ilo, &ihi,
                        P(reflectors.data()), &lda, P(tau.data()), P(qc.data()),
                        &ldc, P(w2.data()), &lw, &info);
                ASSERT_EQ(info, 0);
                if (m == 0 || nc == 0) continue;
                const Dense<T> qd = ToDense(q, n, n, lda);
                const Dense<T> op = trans == 'N' ? qd : Adjoint(qd);
                const Dense<T> cd = ToDense(c, m, nc, ldc);
                const Dense<T> expect = side == 'L' ? Mul(op, cd) : Mul(cd, op);
                EXPECT_LT(ResidualRatio(expect, ToDense(qc, m, nc, ldc),
                                        static_cast<double>(n)),
                          kLapackThreshold)
                    << side << trans;
              }
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Hessenberg QR

// xHSEQR: H = Z * T * Z**H (COMPZ = 'I'), with T in Schur form and the
// eigenvalues W; COMPZ = 'V' with Z = Q of xGEHRD gives A = Z * T * Z**H;
// JOB = 'E' gives the same eigenvalues without T; also for rows and
// columns ILO:IHI only, with xLAHQR (small matrices) and xLAQR0 - xLAQR5.
TYPED_TEST(GeesTest, Hseqr) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(41);
    for (const HseqrParameters& p : kHseqrParameters) {
      const ScopedHseqrParameters params(p.nmin, p.nw, p.nibble, p.ns,
                                         p.kacc22);
      for (const hwyblas_int n : {0, 1, 2, 3, 6, 16, 31, 50}) {
        for (const Kind kind : kKinds) {
          for (const int range : {0, 1}) {
            const hwyblas_int ilo = range == 0 || n < 4 ? 1 : 2;
            const hwyblas_int ihi = range == 0 || n < 4 ? n : n - 1;
            SCOPED_TRACE(::testing::Message()
                         << "n=" << n << " kind=" << KindName(kind) << " ilo="
                         << ilo << " ihi=" << ihi << " nmin=" << p.nmin
                         << " nw=" << p.nw << " nibble=" << p.nibble
                         << " ns=" << p.ns << " kacc22=" << p.kacc22);
            const hwyblas_int ldh = n + 1;
            std::vector<T> h = TestMatrix<T>(kind, n, ldh, rng);
            MakeHessenberg(h, n, ldh);
            for (hwyblas_int j = 0; j < n; ++j) {
              for (hwyblas_int i = j + 1; i < n; ++i) {
                if (j < ilo - 1 || i > ihi - 1) At(h, i, j, ldh) = T(0);
              }
            }
            const hwyblas_int lwork = std::max<hwyblas_int>(1, 11 * n);
            std::vector<T> t = h, z(Sz2(ldh, n), NaN<T>());
            std::vector<Eig<T>> w;
            ASSERT_EQ(Hseqr<T>('S', 'I', n, ilo, ihi, t, ldh, w, z, ldh, lwork),
                      0);
            ASSERT_NO_FATAL_FAILURE(
                ExpectSchurForm(t, n, ldh, w, ilo - 1, ihi - 1));
            ExpectSchurFactorization(h, z, t, n, ldh, ldh, ldh);
            // JOB = 'E': the same eigenvalues (they may differ by rounding:
            // without T, fewer elements are updated).
            std::vector<T> te = h, zn(1);
            std::vector<Eig<T>> we;
            ASSERT_EQ(
                Hseqr<T>('E', 'N', n, ilo, ihi, te, ldh, we, zn, 1, lwork), 0);
            double hnorm = Norm1(h, n, n, ldh);
            for (hwyblas_int i = 0; i < n; ++i) {
              bool found = false;
              for (hwyblas_int k = 0; k < n && !found; ++k) {
                found = std::abs(std::complex<double>(we[i]) -
                                 std::complex<double>(w[k])) <=
                        1e3 * n * Eps<T>() * std::max(1.0, hnorm);
              }
              // (Defective eigenvalues, as of the Jordan block, are only
              // determined to about sqrt(eps).)
              if (kind != Kind::kJordan) {
                EXPECT_TRUE(found) << i;
              }
            }
          }
        }
      }
    }
  });
}

// xGEHRD + xORGHR + xHSEQR (COMPZ = 'V'): A = Z * T * Z**H.
TYPED_TEST(GeesTest, HseqrAccumulatesZ) {
  using T = TypeParam;
  Random<T> rng(43);
  const ScopedHseqrParameters params(11, 5, 7, 4, 2);
  for (const hwyblas_int n : {5, 20, 45}) {
    SCOPED_TRACE(n);
    const hwyblas_int lda = n;
    const std::vector<T> a = rng.Vector(Sz2(lda, n));
    std::vector<T> h = a, tau(Sz(n));
    hwyblas_int lwork = 64 * n + 65 * 64, info = -99, ilo = 1, ihi = n;
    std::vector<T> work(Sz(lwork));
    HWYBLAS_CALL(T, gehrd, &n, &ilo, &ihi, P(h.data()), &lda, P(tau.data()),
                 P(work.data()), &lwork, &info);
    ASSERT_EQ(info, 0);
    std::vector<T> z = h;
    CALL_RC(T, orghr, unghr, &n, &ilo, &ihi, P(z.data()), &lda, P(tau.data()),
            P(work.data()), &lwork, &info);
    MakeHessenberg(h, n, lda);
    std::vector<Eig<T>> w;
    ASSERT_EQ(Hseqr<T>('S', 'V', n, 1, n, h, lda, w, z, lda, lwork), 0);
    ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(h, n, lda, w));
    ExpectSchurFactorization(a, z, h, n, lda, lda, lda);
  }
}

// The workspace queries of xHSEQR: at least max(1, N), and enough.
TYPED_TEST(GeesTest, HseqrWorkspace) {
  using T = TypeParam;
  Random<T> rng(47);
  for (const HseqrParameters& p : kHseqrParameters) {
    const ScopedHseqrParameters params(p.nmin, p.nw, p.nibble, p.ns, p.kacc22);
    for (const hwyblas_int n : {0, 1, 10, 30, 90}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " nmin=" << p.nmin);
      std::vector<T> h = rng.Vector(Sz2(n, n)), z(Sz2(n, n));
      MakeHessenberg(h, n, std::max<hwyblas_int>(1, n));
      const hwyblas_int ld = std::max<hwyblas_int>(1, n);
      std::vector<Eig<T>> q;
      std::vector<T> h0 = h;
      ASSERT_EQ(Hseqr<T>('S', 'I', n, 1, n, h, ld, q, z, ld, -1), 0);
      const auto lwork = static_cast<hwyblas_int>(q[0].real());
      EXPECT_GE(lwork, std::max<hwyblas_int>(1, n));
      for (const hwyblas_int lw : {lwork, std::max<hwyblas_int>(1, n)}) {
        std::vector<T> t = h0;
        std::vector<Eig<T>> w;
        ASSERT_EQ(Hseqr<T>('S', 'I', n, 1, n, t, ld, w, z, ld, lw), 0);
        ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, n, ld, w));
        ExpectSchurFactorization(h0, z, t, n, ld, ld, ld);
      }
    }
  }
}

// xLAHQR directly, without T (WANTT false: only the active block is
// transformed, so H is not a Schur form) and with Z for a subrange of rows.
TYPED_TEST(GeesTest, Lahqr) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(53);
  for (const hwyblas_int n : {1, 2, 4, 9, 14}) {
    for (const bool wantt : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " wantt=" << wantt);
      std::vector<T> h = rng.Vector(Sz2(n, n));
      MakeHessenberg(h, n, n);
      std::vector<T> t = h, z(Sz2(n, n), T(0));
      for (hwyblas_int i = 0; i < n; ++i) At(z, i, i, n) = T(1);
      const hwyblas_int one = 1, wt = wantt ? 1 : 0, wz = 1;
      hwyblas_int info = -99;
      std::vector<Eig<T>> w(static_cast<size_t>(n));
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, lahqr, &wt, &wz, &n, &one, &n, P(t.data()), &n,
                             P(w.data()), &one, &n, P(z.data()), &n, &info);
      } else {
        std::vector<R> wr(n), wi(n);
        HWYBLAS_CALL_REAL(T, lahqr, &wt, &wz, &n, &one, &n, t.data(), &n,
                          wr.data(), wi.data(), &one, &n, z.data(), &n, &info);
        for (hwyblas_int i = 0; i < n; ++i) w[i] = Eig<T>(wr[i], wi[i]);
      }
      ASSERT_EQ(info, 0);
      if (wantt) {
        ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, n, n, w));
        ExpectSchurFactorization(h, z, t, n, n, n, n);
      }
      // The eigenvalues: the trace and (for real T) conjugate pairs.
      std::complex<double> sum = 0, trace = 0;
      for (hwyblas_int i = 0; i < n; ++i) {
        sum += std::complex<double>(w[i]);
        trace += std::complex<double>(At(h, i, i, n));
      }
      EXPECT_NEAR(std::abs(sum - trace), 0, 100 * n * Eps<T>());
    }
  }
}

// xLANV2: the standardized Schur factorization of 2 x 2 matrices.
TYPED_TEST(GeesRealTest, Lanv2) {
  using R = TypeParam;
  Random<R> rng(59);
  const R tiny = std::numeric_limits<R>::min();
  const R huge = std::numeric_limits<R>::max() / 8;
  std::vector<std::vector<R>> cases = {{1, 2, 3, 4},
                                       {1, 0, 3, 4},
                                       {1, 2, 0, 4},
                                       {1, -2, 3, 1},
                                       {2, 5, -5, 2},
                                       {0, 0, 0, 0},
                                       {1, 1e-20f, -1e-20f, 1},
                                       {3, 1, 1, 3},
                                       {1, 2, -2, 1},
                                       {tiny, tiny, -tiny, tiny},
                                       {huge, huge, -huge, -huge},
                                       {1, -3, 3, 5}};
  for (int k = 0; k < 200; ++k) {
    cases.push_back({rng.Next(), rng.Next(), rng.Next(), rng.Next()});
  }
  for (const std::vector<R>& m : cases) {
    SCOPED_TRACE(::testing::Message()
                 << m[0] << " " << m[1] << " " << m[2] << " " << m[3]);
    R a = m[0], b = m[1], c = m[2], d = m[3];
    R rt1r, rt1i, rt2r, rt2i, cs, sn;
    HWYBLAS_CALL_REAL(R, lanv2, &a, &b, &c, &d, &rt1r, &rt1i, &rt2r, &rt2i, &cs,
                      &sn);
    EXPECT_NEAR(cs * cs + sn * sn, 1, 4 * Eps<R>());
    if (c != 0) {
      EXPECT_EQ(a, d);
      EXPECT_TRUE((b > 0) != (c > 0));
      EXPECT_EQ(rt1r, a);
      EXPECT_EQ(rt2r, d);
      EXPECT_EQ(rt1i, std::sqrt(std::abs(b)) * std::sqrt(std::abs(c)));
      EXPECT_EQ(rt2i, -rt1i);
    } else {
      EXPECT_EQ(rt1r, a);
      EXPECT_EQ(rt2r, d);
      EXPECT_EQ(rt1i, 0);
      EXPECT_EQ(rt2i, 0);
    }
    // [A B; C D] = [CS -SN; SN CS] [AA BB; CC DD] [CS SN; -SN CS].
    const double s = std::max({std::abs(double(m[0])), std::abs(double(m[1])),
                               std::abs(double(m[2])), std::abs(double(m[3])),
                               double(tiny)});
    const double r[4] = {
        double(cs) * a - double(sn) * c, double(cs) * b - double(sn) * d,
        double(sn) * a + double(cs) * c, double(sn) * b + double(cs) * d};
    const double e[4] = {r[0] * cs - r[1] * sn, r[0] * sn + r[1] * cs,
                         r[2] * cs - r[3] * sn, r[2] * sn + r[3] * cs};
    EXPECT_NEAR(e[0], m[0], 20 * Eps<R>() * s);
    EXPECT_NEAR(e[1], m[1], 20 * Eps<R>() * s);
    EXPECT_NEAR(e[2], m[2], 20 * Eps<R>() * s);
    EXPECT_NEAR(e[3], m[3], 20 * Eps<R>() * s);
  }
}

// xLAQR1: a multiple of the first column of (H - s1 I) (H - s2 I).
TYPED_TEST(GeesTest, Laqr1) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(61);
  for (int k = 0; k < 50; ++k) {
    for (const hwyblas_int n : {2, 3}) {
      std::vector<T> h = rng.Vector(9);
      MakeHessenberg(h, n, 3);
      const hwyblas_int ld = 3;
      std::vector<T> v(3, NaN<T>());
      std::complex<double> s1, s2;
      if constexpr (IsComplex<T>::value) {
        const T a = rng.Next(), b = rng.Next();
        s1 = std::complex<double>(a);
        s2 = std::complex<double>(b);
        HWYBLAS_CALL_COMPLEX(T, laqr1, &n, P(h.data()), &ld, P(&a), P(&b),
                             P(v.data()));
      } else {
        // Both real or a conjugate pair.
        const R re1 = rng.Next(), im = k % 2 ? rng.Next() : R(0);
        const R re2 = k % 2 ? re1 : rng.Next(), im2 = -im;
        s1 = {re1, im};
        s2 = {re2, im2};
        HWYBLAS_CALL_REAL(T, laqr1, &n, h.data(), &ld, &re1, &im, &re2, &im2,
                          v.data());
      }
      // The first column of (H - s1)(H - s2) in double precision.
      std::complex<double> hd[3][3] = {};
      for (hwyblas_int i = 0; i < n; ++i) {
        for (hwyblas_int j = 0; j < n; ++j) {
          hd[i][j] = std::complex<double>(At(h, i, j, ld));
        }
      }
      std::complex<double> x[3] = {}, y[3] = {};
      x[0] = 1;
      for (hwyblas_int i = 0; i < n; ++i) {
        for (hwyblas_int j = 0; j < n; ++j) {
          y[i] += (hd[i][j] - (i == j ? s2 : 0.0)) * x[j];
        }
      }
      std::complex<double> col[3] = {};
      for (hwyblas_int i = 0; i < n; ++i) {
        for (hwyblas_int j = 0; j < n; ++j) {
          col[i] += (hd[i][j] - (i == j ? s1 : 0.0)) * y[j];
        }
      }
      // v = scale * col with a real or complex scale: v x col = 0.
      double cn = 0, vn = 0;
      for (hwyblas_int i = 0; i < n; ++i) {
        cn = std::max(cn, std::abs(col[i]));
        vn = std::max(vn, std::abs(std::complex<double>(v[i])));
      }
      for (hwyblas_int i = 0; i < n; ++i) {
        for (hwyblas_int j = i + 1; j < n; ++j) {
          const std::complex<double> cross =
              std::complex<double>(v[i]) * col[j] -
              std::complex<double>(v[j]) * col[i];
          EXPECT_LE(std::abs(cross), 100 * Eps<T>() * cn * vn + 1e-300)
              << i << j;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Schur factorization

// xGEES: A = VS * T * VS**H with T in Schur form, sorted (SORT = 'S') so
// that the selected eigenvalues come first, for all kinds of test matrices,
// with the minimum and the optimal workspace, and on matrices that need
// scaling.
TYPED_TEST(GeesTest, Gees) {
  using T = TypeParam;
  ForEachTarget([] {
    Random<T> rng(67);
    for (const hwyblas_int n : {0, 1, 2, 3, 7, 16, 33}) {
      for (const Kind kind : kKinds) {
        for (const char jobvs : {'V', 'N'}) {
          for (const char sort : {'N', 'S'}) {
            for (const Select select :
                 {Select::kPositiveReal, Select::kPositiveImag, Select::kNone,
                  Select::kAll}) {
              if (sort == 'N' && select != Select::kPositiveReal) continue;
              for (const int variant : {0, 1}) {  // optimal / minimal work
                SCOPED_TRACE(::testing::Message()
                             << "n=" << n << " kind=" << KindName(kind)
                             << " jobvs=" << jobvs << " sort=" << sort
                             << " select=" << static_cast<int>(select)
                             << " variant=" << variant);
                const hwyblas_int lda = n + 2;
                const std::vector<T> a = TestMatrix<T>(kind, n, lda, rng);
                std::vector<T> t = a, vs(Sz2(lda, n), NaN<T>());
                std::vector<Eig<T>> w;
                hwyblas_int sdim = -1;
                hwyblas_int lwork = MinimumGeesLwork<T>(n);
                if (variant == 0) {
                  std::vector<T> tq = a;
                  ASSERT_EQ(Gees<T>(jobvs, sort, select, n, tq, lda, &sdim, w,
                                    vs, lda, -1),
                            0);
                  lwork = static_cast<hwyblas_int>(w[0].real());
                  EXPECT_GE(lwork, MinimumGeesLwork<T>(n));
                }
                const hwyblas_int info = Gees<T>(jobvs, sort, select, n, t, lda,
                                                 &sdim, w, vs, lda, lwork);
                ASSERT_EQ(info, 0);
                ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, n, lda, w));
                if (jobvs == 'V') {
                  ExpectSchurFactorization(a, vs, t, n, lda, lda, lda);
                }
                if (sort == 'N') {
                  EXPECT_EQ(sdim, 0);
                  continue;
                }
                hwyblas_int expected = 0;
                for (hwyblas_int i = 0; i < n; ++i) {
                  const bool sel = Selected<T>(select, w, i);
                  if (sel) ++expected;
                  EXPECT_EQ(sel, i < sdim) << i;
                }
                EXPECT_EQ(sdim, expected);
              }
            }
          }
        }
      }
    }
  });
}

// xGEES on matrices with tiny or huge elements (scaled before and after
// the QR algorithm), and with a 2 x 2 block of complex eigenvalues that
// scaling back towards underflow turns into a pair of real ones.
TYPED_TEST(GeesTest, GeesScaling) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(71);
  const R tiny = std::sqrt(std::numeric_limits<R>::min()) *
                 std::numeric_limits<R>::epsilon() / 64;
  const R huge = 1 / tiny;
  for (const R factor : {tiny, huge}) {
    for (const char sort : {'N', 'S'}) {
      for (const hwyblas_int n : {4, 12}) {
        SCOPED_TRACE(::testing::Message()
                     << "factor=" << factor << " sort=" << sort << " n=" << n);
        std::vector<T> a = TestMatrix<T>(Kind::kRandom, n, n, rng);
        for (T& x : a) x *= factor;
        std::vector<T> t = a, vs(Sz2(n, n));
        std::vector<Eig<T>> w;
        hwyblas_int sdim = -1;
        ASSERT_EQ(Gees<T>('V', sort, Select::kPositiveReal, n, t, n, &sdim, w,
                          vs, n, 64 * n),
                  0);
        ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, n, n, w));
        ExpectSchurFactorization(a, vs, t, n, n, n, n);
      }
    }
  }
  if constexpr (!IsComplex<T>::value) {
    // [0 a; -b 0] * small: a complex pair whose off-diagonal elements
    // underflow differently when scaled back: x * [0 1; -1e-30 0].
    const R small = std::numeric_limits<R>::min() * 64;
    std::vector<T> a = {0, -small * std::numeric_limits<R>::epsilon() / 4,
                        small, 0};
    std::vector<T> t = a, vs(4);
    std::vector<Eig<T>> w;
    hwyblas_int sdim = -1;
    ASSERT_EQ(Gees<T>('V', 'N', Select::kNone, 2, t, 2, &sdim, w, vs, 2, 64),
              0);
    ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, 2, 2, w));
  }
}

// ---------------------------------------------------------------------------
// Reordering of the Schur form

// A random n x n matrix in Schur form, with its Schur vectors Q of a random
// matrix (xGEES), and the eigenvalues.
template <typename T>
void RandomSchur(hwyblas_int n, Random<T>& rng, std::vector<T>& a,
                 std::vector<T>& t, std::vector<T>& q, std::vector<Eig<T>>& w) {
  a = rng.Vector(Sz2(n, n));
  // Some repeated diagonal blocks of complex pairs for real T.
  t = a;
  q.assign(Sz2(n, n), T(0));
  hwyblas_int sdim = 0;
  ASSERT_EQ(Gees<T>('V', 'N', Select::kNone, n, t, n, &sdim, w, q, n, 64 * n),
            0);
}

// xTREXC: moves the diagonal block at IFST to ILST; A = Q * T * Q**H
// stays a Schur factorization, with the moved eigenvalue(s) at ILST.
TYPED_TEST(GeesTest, Trexc) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(73);
  for (const hwyblas_int n : {1, 2, 5, 11}) {
    std::vector<T> a, t0, q0;
    std::vector<Eig<T>> w0;
    RandomSchur(n, rng, a, t0, q0, w0);
    for (hwyblas_int ifst = 1; ifst <= n; ++ifst) {
      for (hwyblas_int ilst = 1; ilst <= n; ++ilst) {
        for (const char compq : {'V', 'N'}) {
          SCOPED_TRACE(::testing::Message()
                       << "n=" << n << " ifst=" << ifst << " ilst=" << ilst
                       << " " << compq);
          std::vector<T> t = t0, q = q0;
          hwyblas_int info = -99;
          hwyblas_int f = ifst, l = ilst;
          if constexpr (IsComplex<T>::value) {
            HWYBLAS_CALL_COMPLEX(T, trexc, &compq, &n, P(t.data()), &n,
                                 P(q.data()), &n, &f, &l, &info);
          } else {
            std::vector<R> work(Sz(n), NAN);
            HWYBLAS_CALL_REAL(T, trexc, &compq, &n, t.data(), &n, q.data(), &n,
                              &f, &l, work.data(), &info);
          }
          ASSERT_EQ(info, 0);
          // The eigenvalues of t, from its diagonal.
          std::vector<Eig<T>> w(static_cast<size_t>(n));
          for (hwyblas_int i = 0; i < n; ++i) {
            if (!IsComplex<T>::value && i + 1 < n &&
                At(t, i + 1, i, n) != T(0)) {
              const R re = std::real(At(t, i, i, n));
              const R im = std::sqrt(std::abs(std::real(At(t, i, i + 1, n)))) *
                           std::sqrt(std::abs(std::real(At(t, i + 1, i, n))));
              w[i] = {re, im};
              w[i + 1] = {re, -im};
              ++i;
            } else {
              w[i] = Eig<T>(At(t, i, i, n));
            }
          }
          ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, n, n, w));
          if (compq == 'V') ExpectSchurFactorization(a, q, t, n, n, n, n);
          // The moved eigenvalue (pair) is at ILST, IFST and ILST adjusted
          // by xTREXC to the first rows of their blocks.
          const Eig<T> moved = w0[f - 1];
          const double tol = 1e3 * Eps<T>() * Norm1(a, n, n, n);
          EXPECT_LE(std::abs(std::complex<double>(w[l - 1]) -
                             std::complex<double>(moved)),
                    tol);
          if (!IsComplex<T>::value) {
            // IFST was moved to the first row of its block.
            EXPECT_TRUE(f == ifst || f == ifst - 1);
          }
        }
      }
    }
  }
}

// xLAEXC (real only): swaps adjacent blocks of 1 x 1 and 2 x 2.
TYPED_TEST(GeesRealTest, Laexc) {
  using R = TypeParam;
  Random<R> rng(79);
  for (const hwyblas_int n1 : {1, 2}) {
    for (const hwyblas_int n2 : {1, 2}) {
      for (int k = 0; k < 20; ++k) {
        SCOPED_TRACE(::testing::Message() << n1 << n2 << " " << k);
        // T = [T11 T12; 0 T22] with standardized blocks, inside a 6 x 6
        // matrix at J1 = 2.
        const hwyblas_int n = 6, j1 = 2;
        std::vector<R> t(36, R(0));
        for (hwyblas_int j = 0; j < n; ++j) {
          for (hwyblas_int i = 0; i <= j; ++i) At(t, i, j, n) = rng.Next();
        }
        auto block = [&](hwyblas_int i0, hwyblas_int sz) {
          if (sz == 2) {
            const R d = rng.Next(), b = 1 + std::abs(rng.Next());
            At(t, i0, i0, n) = At(t, i0 + 1, i0 + 1, n) = d;
            At(t, i0, i0 + 1, n) = b;
            At(t, i0 + 1, i0, n) = -(R(0.5) + std::abs(rng.Next()));
          }
        };
        block(j1 - 1, n1);
        block(j1 - 1 + n1, n2);
        std::vector<R> q(36, R(0)), work(n);
        for (hwyblas_int i = 0; i < n; ++i) At(q, i, i, n) = 1;
        std::vector<R> tt = t;
        const hwyblas_int wantq = 1;
        hwyblas_int info = -99;
        HWYBLAS_CALL_REAL(R, laexc, &wantq, &n, tt.data(), &n, q.data(), &n,
                          &j1, &n1, &n2, work.data(), &info);
        ASSERT_EQ(info, 0);
        ExpectSchurFactorization(t, q, tt, n, n, n, n);
        // The blocks are swapped: the eigenvalues of T22 now come first.
        const auto eig = [&](const std::vector<R>& m, hwyblas_int i0,
                             hwyblas_int sz) {
          const R a = At(m, i0, i0, n);
          if (sz == 1) return std::complex<double>(a);
          return std::complex<double>(
              a,
              std::sqrt(std::abs(At(m, i0, i0 + 1, n) * At(m, i0 + 1, i0, n))));
        };
        EXPECT_NEAR(std::abs(eig(tt, j1 - 1, n2) - eig(t, j1 - 1 + n1, n2)), 0,
                    1e3 * Eps<R>());
        EXPECT_NEAR(std::abs(eig(tt, j1 - 1 + n2, n1) - eig(t, j1 - 1, n1)), 0,
                    1e3 * Eps<R>());
      }
    }
  }
}

// The solution of op(A) X + isgn X op(B) = scale C for the dense matrices
// A (m x m), B (n x n), C, in double precision by Gaussian elimination on
// the Kronecker form.
template <typename T>
Dense<T> SolveSylvester(char trana, char tranb, int isgn, const Dense<T>& a,
                        const Dense<T>& b, const Dense<T>& c) {
  using C = RefT<T>;
  const size_t m = a.rows, n = b.rows, mn = m * n;
  const Dense<T> oa = trana == 'N' ? a : Adjoint(a);
  const Dense<T> ob = tranb == 'N' ? b : Adjoint(b);
  // K vec(X) = vec(C), K = I (x) op(A) + isgn op(B)**T (x) I.
  std::vector<C> k(mn * mn), x(mn);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      const size_t row = i + j * m;
      x[row] = c(i, j);
      for (size_t l = 0; l < m; ++l) k[row + (l + j * m) * mn] += oa(i, l);
      for (size_t l = 0; l < n; ++l) {
        k[row + (i + l * m) * mn] += static_cast<double>(isgn) * ob(l, j);
      }
    }
  }
  for (size_t col = 0; col < mn; ++col) {
    size_t piv = col;
    for (size_t r = col + 1; r < mn; ++r) {
      if (std::abs(k[r + col * mn]) > std::abs(k[piv + col * mn])) piv = r;
    }
    for (size_t l = 0; l < mn; ++l) std::swap(k[col + l * mn], k[piv + l * mn]);
    std::swap(x[col], x[piv]);
    for (size_t r = col + 1; r < mn; ++r) {
      const C f = k[r + col * mn] / k[col + col * mn];
      for (size_t l = col; l < mn; ++l) k[r + l * mn] -= f * k[col + l * mn];
      x[r] -= f * x[col];
    }
  }
  for (size_t col = mn; col-- > 0;) {
    for (size_t l = col + 1; l < mn; ++l) x[col] -= k[col + l * mn] * x[l];
    x[col] /= k[col + col * mn];
  }
  Dense<T> xd(m, n);
  xd.v = x;
  return xd;
}

// xTRSYL: op(A) X + ISGN X op(B) = SCALE C for A and B in Schur form, all
// operations and signs, compared with the solution of the Kronecker form.
TYPED_TEST(GeesTest, Trsyl) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(83);
  const char ct = IsComplex<T>::value ? 'C' : 'T';
  for (const hwyblas_int m : {0, 1, 2, 5, 8}) {
    for (const hwyblas_int n : {0, 1, 3, 6}) {
      std::vector<T> a0, a, qa, b0, b, qb;
      std::vector<Eig<T>> wa, wb;
      if (m > 0) RandomSchur(m, rng, a0, a, qa, wa);
      if (n > 0) RandomSchur(n, rng, b0, b, qb, wb);
      // Shift B's eigenvalues away from -A's and A's.
      for (hwyblas_int i = 0; i < n; ++i) At(b, i, i, n) += T(3);
      for (const char trana : {'N', ct}) {
        for (const char tranb : {'N', ct}) {
          for (const hwyblas_int isgn : {1, -1}) {
            SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n << " "
                                              << trana << tranb << isgn);
            const hwyblas_int ldc = m + 1;
            const std::vector<T> c = rng.Vector(Sz2(ldc, n));
            std::vector<T> x = c;
            R scale = -1;
            hwyblas_int info = -99;
            const hwyblas_int lda = std::max<hwyblas_int>(1, m);
            const hwyblas_int ldb = std::max<hwyblas_int>(1, n);
            if (m == 0) a.assign(1, T(0));
            if (n == 0) b.assign(1, T(0));
            HWYBLAS_CALL(T, trsyl, &trana, &tranb, &isgn, &m, &n, P(a.data()),
                         &lda, P(b.data()), &ldb, P(x.data()), &ldc, &scale,
                         &info);
            ASSERT_EQ(info, 0);
            EXPECT_EQ(scale, R(1));
            if (m == 0 || n == 0) continue;
            const Dense<T> expect = SolveSylvester<T>(
                trana, tranb, static_cast<int>(isgn), ToDense(a, m, m, m),
                ToDense(b, n, n, n), ToDense(c, m, n, ldc));
            EXPECT_LT(ResidualRatio(expect, ToDense(x, m, n, ldc),
                                    static_cast<double>(m + n)),
                      100 * kLapackThreshold);
          }
        }
      }
    }
  }
}

// xTRSYL on a nearly singular equation with a huge right-hand side: SCALE <
// 1 avoids overflow, and op(A) X + ISGN X op(B) = SCALE C still holds.
TYPED_TEST(GeesTest, TrsylScaling) {
  using T = TypeParam;
  using R = RealT<T>;
  const R eps = std::numeric_limits<R>::epsilon();
  const R huge = std::numeric_limits<R>::max() / 16;
  const char ct = IsComplex<T>::value ? 'C' : 'T';
  for (const hwyblas_int m : {1, 2}) {
    for (const char trana : {'N', ct}) {
      for (const char tranb : {'N', ct}) {
        SCOPED_TRACE(::testing::Message() << m << trana << tranb);
        // A = [1 x; 0 2], B = -diag(A) + 64 eps**2 I: A + B is nearly
        // singular.
        std::vector<T> a(Sz2(m, m), T(0)), b(Sz2(m, m), T(0));
        for (hwyblas_int i = 0; i < m; ++i) {
          At(a, i, i, m) = T(R(i + 1));
          At(b, i, i, m) = T(-R(i + 1) + 64 * eps * eps);
        }
        if (m == 2) At(a, 0, 1, m) = T(R(0.5));
        std::vector<T> c(Sz2(m, m), T(huge));
        std::vector<T> x = c;
        R scale = -1;
        hwyblas_int info = -99;
        const hwyblas_int isgn = 1;
        HWYBLAS_CALL(T, trsyl, &trana, &tranb, &isgn, &m, &m, P(a.data()), &m,
                     P(b.data()), &m, P(x.data()), &m, &scale, &info);
        ASSERT_GE(info, 0);
        EXPECT_LT(scale, R(1));
        EXPECT_GT(scale, R(0));
        for (const T& v : x) {
          ASSERT_TRUE(std::isfinite(std::abs(std::complex<double>(v))));
        }
        // The residual relative to the sizes of the terms.
        const Dense<T> ad = ToDense(a, m, m, m), bd = ToDense(b, m, m, m);
        const Dense<T> xd = ToDense(x, m, m, m);
        const Dense<T> lhs1 = trana == 'N' ? Mul(ad, xd) : Mul(Adjoint(ad), xd);
        const Dense<T> lhs2 = tranb == 'N' ? Mul(xd, bd) : Mul(xd, Adjoint(bd));
        double size = 0, resid = 0;
        for (hwyblas_int j = 0; j < m; ++j) {
          for (hwyblas_int i = 0; i < m; ++i) {
            const RefT<T> r = lhs1(i, j) + lhs2(i, j) -
                              static_cast<double>(scale) *
                                  static_cast<RefT<T>>(At(c, i, j, m));
            resid = std::max(resid, std::abs(r));
            size = std::max({size, std::abs(lhs1(i, j)), std::abs(lhs2(i, j))});
          }
        }
        EXPECT_LE(resid, 100 * eps * size);
      }
    }
  }
}

// xLASY2 (real only): the 1 x 1 to 2 x 2 Sylvester equations.
TYPED_TEST(GeesRealTest, Lasy2) {
  using R = TypeParam;
  Random<R> rng(89);
  for (int k = 0; k < 30; ++k) {
    for (const hwyblas_int n1 : {1, 2}) {
      for (const hwyblas_int n2 : {1, 2}) {
        for (const int ltranl : {0, 1}) {
          for (const int ltranr : {0, 1}) {
            for (const hwyblas_int isgn : {1, -1}) {
              SCOPED_TRACE(::testing::Message()
                           << n1 << n2 << ltranl << ltranr << isgn << " " << k);
              std::vector<R> tl = rng.Vector(4), tr = rng.Vector(4);
              std::vector<R> b = rng.Vector(4), x(4, NAN);
              // Keep op(TL) + isgn op(TR) away from singular.
              tl[0] += 3;
              tl[3] += 3;
              R scale = -1, xnorm = -1;
              hwyblas_int info = -99;
              const hwyblas_int ld = 2, lt = ltranl, rt = ltranr;
              HWYBLAS_CALL_REAL(R, lasy2, &lt, &rt, &isgn, &n1, &n2, tl.data(),
                                &ld, tr.data(), &ld, b.data(), &ld, &scale,
                                x.data(), &ld, &xnorm, &info);
              ASSERT_EQ(info, 0);
              const auto sub = [](const std::vector<R>& v, hwyblas_int s) {
                Dense<R> d(static_cast<size_t>(s), static_cast<size_t>(s));
                for (hwyblas_int j = 0; j < s; ++j) {
                  for (hwyblas_int i = 0; i < s; ++i) d(i, j) = At(v, i, j, 2);
                }
                return d;
              };
              Dense<R> bd(static_cast<size_t>(n1), static_cast<size_t>(n2));
              for (hwyblas_int j = 0; j < n2; ++j) {
                for (hwyblas_int i = 0; i < n1; ++i) {
                  bd(i, j) = static_cast<double>(scale) * At(b, i, j, 2);
                }
              }
              const Dense<R> expect = SolveSylvester<R>(
                  ltranl ? 'T' : 'N', ltranr ? 'T' : 'N',
                  static_cast<int>(isgn), sub(tl, n1), sub(tr, n2), bd);
              double xmax = 0;
              for (hwyblas_int i = 0; i < n1; ++i) {
                double row = 0;
                for (hwyblas_int j = 0; j < n2; ++j) {
                  EXPECT_NEAR(At(x, i, j, 2), expect(i, j),
                              1e3 * Eps<R>() * (1 + std::abs(expect(i, j))));
                  row += std::abs(At(x, i, j, 2));
                }
                xmax = std::max(xmax, row);
              }
              EXPECT_NEAR(xnorm, xmax, 8 * Eps<R>() * xmax);
              EXPECT_EQ(scale, 1);
            }
          }
        }
      }
    }
  }
  // A NaN in TL or TR: the 4 x 4 elimination of N1 = N2 = 2 finds no
  // pivot among NaNs (the reference then swaps with the previous step's).
  for (int pos = 0; pos < 8; ++pos) {
    SCOPED_TRACE(pos);
    std::vector<R> tl = rng.Vector(4), tr = rng.Vector(4), b = rng.Vector(4);
    std::vector<R> x(4, 0);
    (pos < 4 ? tl : tr)[static_cast<size_t>(pos % 4)] = NAN;
    R scale = -1, xnorm = -1;
    hwyblas_int info = -99;
    const hwyblas_int ld = 2, two = 2, lt = 0, rt = 0, isgn = 1;
    HWYBLAS_CALL_REAL(R, lasy2, &lt, &rt, &isgn, &two, &two, tl.data(), &ld,
                      tr.data(), &ld, b.data(), &ld, &scale, x.data(), &ld,
                      &xnorm, &info);
    EXPECT_TRUE(
        std::any_of(x.begin(), x.end(), [](R v) { return std::isnan(v); }));
  }
}

// xLALN2 (real only): (ca A - w D) X = s B and (ca A**T - w D) X = s B.
TYPED_TEST(GeesRealTest, Laln2) {
  using R = TypeParam;
  Random<R> rng(97);
  for (int k = 0; k < 30; ++k) {
    for (const hwyblas_int na : {1, 2}) {
      for (const hwyblas_int nw : {1, 2}) {
        for (const int ltrans : {0, 1}) {
          SCOPED_TRACE(::testing::Message() << na << nw << ltrans << " " << k);
          std::vector<R> a = rng.Vector(4), b = rng.Vector(4), x(4, NAN);
          const R ca = 1 + std::abs(rng.Next()), d1 = rng.Next(),
                  d2 = rng.Next(), wr = rng.Next() + 3,
                  wi = nw == 2 ? rng.Next() : R(0);
          const R smin = static_cast<R>(Eps<R>());
          R scale = -1, xnorm = -1;
          hwyblas_int info = -99;
          const hwyblas_int ld = 2, lt = ltrans;
          HWYBLAS_CALL_REAL(R, laln2, &lt, &na, &nw, &smin, &ca, a.data(), &ld,
                            &d1, &d2, b.data(), &ld, &wr, &wi, x.data(), &ld,
                            &scale, &xnorm, &info);
          ASSERT_EQ(info, 0);
          // Residual of (ca op(A) - (wr + i wi) D) x = scale b.
          for (hwyblas_int i = 0; i < na; ++i) {
            std::complex<double> lhs = 0;
            for (hwyblas_int j = 0; j < na; ++j) {
              const double aij = ltrans ? At(a, j, i, 2) : At(a, i, j, 2);
              const std::complex<double> xj(At(x, j, 0, 2),
                                            nw == 2 ? At(x, j, 1, 2) : 0.0);
              const double dj = j == 0 ? d1 : d2;
              lhs += (static_cast<double>(ca) * aij -
                      (i == j ? std::complex<double>(wr, wi) * dj : 0.0)) *
                     xj;
            }
            const std::complex<double> rhs(
                static_cast<double>(scale) * At(b, i, 0, 2),
                nw == 2 ? static_cast<double>(scale) * At(b, i, 1, 2) : 0.0);
            EXPECT_LE(std::abs(lhs - rhs),
                      1e3 * Eps<R>() * (std::abs(rhs) + 1));
          }
          EXPECT_GT(xnorm, 0);
        }
      }
    }
  }
}

// xTRSEN: the selected eigenvalues move to the top (T11 is M x M), A = Q *
// T * Q**H stays a Schur factorization; S is the reciprocal condition
// number of the cluster, 1 / sqrt(1 + ||X||_F**2) for T11 X - X T22 =
// T12, and SEP estimates sep(T11, T22) (between the smallest singular
// value of the Sylvester operator over sqrt(M (N-M)) and three times it).
TYPED_TEST(GeesTest, Trsen) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(101);
  for (const hwyblas_int n : {1, 2, 6, 13}) {
    std::vector<T> a, t0, q0;
    std::vector<Eig<T>> w0;
    RandomSchur(n, rng, a, t0, q0, w0);
    for (int pattern = 0; pattern < 4; ++pattern) {
      for (const char job : {'N', 'E', 'V', 'B'}) {
        for (const char compq : {'V', 'N'}) {
          SCOPED_TRACE(::testing::Message() << "n=" << n << " pattern="
                                            << pattern << " " << job << compq);
          std::vector<hwyblas_int> select(Sz(n));
          for (hwyblas_int i = 0; i < n; ++i) {
            select[i] = pattern == 0   ? (i % 3 == 1)
                        : pattern == 1 ? (i == n - 1)
                        : pattern == 2 ? 1
                                       : (w0[i].real() < 0);
          }
          std::vector<T> t = t0, q = q0;
          hwyblas_int m = -1, info = -99;
          R s = -1, sep = -1;
          std::vector<Eig<T>> w(Sz(n));
          // Workspace query, then the call with that workspace.
          if constexpr (IsComplex<T>::value) {
            T wq;
            hwyblas_int lq = -1;
            HWYBLAS_CALL_COMPLEX(T, trsen, &job, &compq, select.data(), &n,
                                 P(t.data()), &n, P(q.data()), &n, P(w.data()),
                                 &m, &s, &sep, P(&wq), &lq, &info);
            ASSERT_EQ(info, 0);
            hwyblas_int lwork = static_cast<hwyblas_int>(std::real(wq));
            std::vector<T> work(Sz(lwork));
            HWYBLAS_CALL_COMPLEX(T, trsen, &job, &compq, select.data(), &n,
                                 P(t.data()), &n, P(q.data()), &n, P(w.data()),
                                 &m, &s, &sep, P(work.data()), &lwork, &info);
          } else {
            std::vector<R> wr(Sz(n)), wi(Sz(n));
            R wq;
            hwyblas_int iwq, lq = -1;
            HWYBLAS_CALL_REAL(T, trsen, &job, &compq, select.data(), &n,
                              t.data(), &n, q.data(), &n, wr.data(), wi.data(),
                              &m, &s, &sep, &wq, &lq, &iwq, &lq, &info);
            ASSERT_EQ(info, 0);
            hwyblas_int lwork = static_cast<hwyblas_int>(wq), liwork = iwq;
            std::vector<R> work(Sz(lwork));
            std::vector<hwyblas_int> iwork(Sz(liwork));
            HWYBLAS_CALL_REAL(T, trsen, &job, &compq, select.data(), &n,
                              t.data(), &n, q.data(), &n, wr.data(), wi.data(),
                              &m, &s, &sep, work.data(), &lwork, iwork.data(),
                              &liwork, &info);
            for (hwyblas_int i = 0; i < n; ++i) w[i] = Eig<T>(wr[i], wi[i]);
          }
          ASSERT_EQ(info, 0);
          w.resize(n);
          ASSERT_NO_FATAL_FAILURE(ExpectSchurForm(t, n, n, w));
          if (compq == 'V') ExpectSchurFactorization(a, q, t, n, n, n, n);
          // M: the selected eigenvalues (with both of a real pair).
          hwyblas_int expected = 0;
          for (hwyblas_int i = 0; i < n; ++i) {
            const bool pair = !IsComplex<T>::value && w0[i].imag() != 0;
            const bool sel = select[i] != 0 ||
                             (pair && w0[i].imag() > 0 && select[i + 1]) ||
                             (pair && w0[i].imag() < 0 && select[i - 1]);
            if (sel) ++expected;
          }
          EXPECT_EQ(m, expected);
          // The leading M eigenvalues are the selected ones.
          for (hwyblas_int i = 0; i < m; ++i) {
            bool found = false;
            for (hwyblas_int k = 0; k < n && !found; ++k) {
              const bool pair = !IsComplex<T>::value && w0[k].imag() != 0;
              const bool sel = select[k] != 0 ||
                               (pair && w0[k].imag() > 0 && select[k + 1]) ||
                               (pair && w0[k].imag() < 0 && select[k - 1]);
              found = sel && std::abs(std::complex<double>(w[i]) -
                                      std::complex<double>(w0[k])) <=
                                 1e3 * Eps<T>() * (1 + std::abs(w0[k]));
            }
            EXPECT_TRUE(found) << i;
          }
          if (m == 0 || m == n) {
            if (job == 'E' || job == 'B') {
              EXPECT_EQ(s, R(1));
            }
            continue;
          }
          const hwyblas_int n1 = m, n2 = n - m;
          Dense<T> t11(n1, n1), t22(n2, n2), t12(n1, n2);
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              const RefT<T> v = static_cast<RefT<T>>(At(t, i, j, n));
              if (i < n1 && j < n1) t11(i, j) = v;
              if (i >= n1 && j >= n1) t22(i - n1, j - n1) = v;
              if (i < n1 && j >= n1) t12(i, j - n1) = v;
            }
          }
          if (job == 'E' || job == 'B') {
            const Dense<T> x = SolveSylvester<T>('N', 'N', -1, t11, t22, t12);
            double fro = 0;
            for (const RefT<T>& v : x.v) fro += std::norm(v);
            EXPECT_NEAR(s, 1 / std::sqrt(1 + fro), 1e3 * Eps<T>());
          }
          if (job == 'V' || job == 'B') {
            // The smallest singular value of the Sylvester operator K, by
            // inverse iteration with K**H K.
            double lo = 0;
            Dense<T> v(n1, n2);
            for (RefT<T>& e : v.v) e = 1;
            for (int it = 0; it < 200; ++it) {
              const Dense<T> y = SolveSylvester<T>('N', 'N', -1, t11, t22, v);
              const Dense<T> z = SolveSylvester<T>('C', 'C', -1, t11, t22, y);
              double nz = 0;
              for (const RefT<T>& e : z.v) nz += std::norm(e);
              nz = std::sqrt(nz);
              for (size_t k = 0; k < v.v.size(); ++k) v.v[k] = z.v[k] / nz;
              lo = 1 / std::sqrt(nz);
            }
            const double dim = std::sqrt(static_cast<double>(n1 * n2));
            EXPECT_GE(sep, lo / dim / 1.01);
            EXPECT_LE(sep, 3.01 * lo * dim);
          }
        }
      }
    }
  }
}

// xLACN2: estimates the 1-norm of random matrices by reverse
// communication: never more than the norm, and here (small matrices)
// usually equal.
TYPED_TEST(GeesTest, Lacn2) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(103);
  int exact = 0, total = 0;
  for (const hwyblas_int n : {1, 2, 5, 10, 30}) {
    for (int k = 0; k < 10; ++k) {
      SCOPED_TRACE(::testing::Message() << n << " " << k);
      const std::vector<T> a = rng.Vector(Sz2(n, n));
      std::vector<T> v(Sz(n)), x(Sz(n));
      std::vector<hwyblas_int> isgn(Sz(n));
      hwyblas_int kase = 0, isave[3] = {0, 0, 0};
      R est = 0;
      int calls = 0;
      do {
        if constexpr (IsComplex<T>::value) {
          HWYBLAS_CALL_COMPLEX(T, lacn2, &n, P(v.data()), P(x.data()), &est,
                               &kase, isave);
        } else {
          HWYBLAS_CALL_REAL(T, lacn2, &n, v.data(), x.data(), isgn.data(), &est,
                            &kase, isave);
        }
        if (kase != 0) {
          // x := A * x (kase = 1) or A**H * x (kase = 2).
          std::vector<T> y(Sz(n), T(0));
          for (hwyblas_int j = 0; j < n; ++j) {
            for (hwyblas_int i = 0; i < n; ++i) {
              if (kase == 1) {
                y[i] += At(a, i, j, n) * x[j];
              } else {
                y[j] += Conj(At(a, i, j, n)) * x[i];
              }
            }
          }
          x = y;
        }
        ASSERT_LT(++calls, 20);
      } while (kase != 0);
      const double norm = Norm1(a, n, n, n);
      EXPECT_LE(est, norm * (1 + 100 * Eps<T>()));
      EXPECT_GE(est, norm / (3 * n));
      ++total;
      if (std::abs(est - norm) <= 100 * Eps<T>() * norm) ++exact;
    }
  }
  EXPECT_GE(exact, total / 2);
}

// xLARFX: H * C and C * H with the unrolled code (orders <= 10) and
// xLARF, compared with the dense products.
TYPED_TEST(GeesTest, Larfx) {
  using T = TypeParam;
  Random<T> rng(107);
  for (const char side : {'L', 'R'}) {
    for (hwyblas_int order = 1; order <= 12; ++order) {
      SCOPED_TRACE(::testing::Message() << side << order);
      const hwyblas_int m = side == 'L' ? order : 4;
      const hwyblas_int n = side == 'L' ? 3 : order;
      const hwyblas_int ldc = m + 1;
      const std::vector<T> v = rng.Vector(Sz(order));
      const T tau = rng.Next();
      const std::vector<T> c = rng.Vector(Sz2(ldc, n));
      std::vector<T> hc = c, work(Sz(side == 'L' ? n : m));
      HWYBLAS_CALL(T, larfx, &side, &m, &n, P(v.data()), P(&tau), P(hc.data()),
                   &ldc, P(work.data()));
      // H = I - tau v v**H.
      Dense<T> h = Identity<T>(order);
      for (hwyblas_int j = 0; j < order; ++j) {
        for (hwyblas_int i = 0; i < order; ++i) {
          h(i, j) -= static_cast<RefT<T>>(tau) * static_cast<RefT<T>>(v[i]) *
                     Conj(static_cast<RefT<T>>(v[j]));
        }
      }
      const Dense<T> cd = ToDense(c, m, n, ldc);
      const Dense<T> expect = side == 'L' ? Mul(h, cd) : Mul(cd, h);
      EXPECT_LT(ResidualRatio(expect, ToDense(hc, m, n, ldc),
                              static_cast<double>(order)),
                kLapackThreshold);
    }
  }
}

// ---------------------------------------------------------------------------
// Complex helpers

// CLARTG / ZLARTG: [c s; -conj(s) c] [f; g] = [r; 0] with real c >= 0 and
// c**2 + |s|**2 = 1, also for extreme magnitudes.
TYPED_TEST(GeesComplexTest, Lartg) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(109);
  const R tiny = std::numeric_limits<R>::min();
  const R huge = std::numeric_limits<R>::max() / 4;
  std::vector<std::pair<T, T>> cases = {{T(0), T(0)},
                                        {T(1, 2), T(0)},
                                        {T(0), T(3, -4)},
                                        {T(0), T(5, 0)},
                                        {T(0), T(0, -2)},
                                        {T(tiny, tiny), T(1, 1)},
                                        {T(1, 1), T(tiny, 0)},
                                        {T(huge, huge), T(1, 0)},
                                        {T(1, 0), T(huge, -huge)},
                                        {T(tiny, 0), T(huge, 0)},
                                        {T(huge, 0), T(tiny, tiny)},
                                        {T(0), T(huge, huge)},
                                        {T(0), T(tiny, tiny)}};
  for (int k = 0; k < 100; ++k) cases.emplace_back(rng.Next(), rng.Next());
  for (const auto& [f, g] : cases) {
    SCOPED_TRACE(::testing::Message() << f << " " << g);
    R c = -1;
    T s, r;
    HWYBLAS_CALL_COMPLEX(T, lartg, P(&f), P(&g), &c, P(&s), P(&r));
    EXPECT_GE(c, 0);
    EXPECT_NEAR(c * c + std::norm(s), 1, 8 * Eps<T>());
    const std::complex<double> fd(f), gd(g), sd(s), rd(r);
    const double scale = std::max(std::abs(fd), std::abs(gd));
    EXPECT_LE(std::abs(double(c) * fd + sd * gd - rd), 8 * Eps<T>() * scale);
    EXPECT_LE(std::abs(-std::conj(sd) * fd + double(c) * gd),
              8 * Eps<T>() * scale);
    if (g == T(0)) {
      EXPECT_EQ(c, 1);
      EXPECT_EQ(s, T(0));
      EXPECT_EQ(r, f);
    }
    if (f == T(0) && g != T(0)) {
      EXPECT_EQ(c, 0);
      EXPECT_EQ(std::imag(r), 0);
    }
  }
}

// CROT / ZROT, ICMAX1 / IZMAX1 and SCSUM1 / DZSUM1, also with other
// increments.
TYPED_TEST(GeesComplexTest, RotImax1Sum1) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(113);
  for (const hwyblas_int n : {0, 1, 4, 9}) {
    for (const hwyblas_int incx : {1, 2, -1}) {
      for (const hwyblas_int incy : {1, -2}) {
        SCOPED_TRACE(::testing::Message() << n << " " << incx << " " << incy);
        const size_t sx = Sz(n) * std::abs(incx), sy = Sz(n) * std::abs(incy);
        const std::vector<T> x0 = rng.Vector(sx), y0 = rng.Vector(sy);
        std::vector<T> x = x0, y = y0;
        const R c = R(0.6);
        const T s = T(R(0.48), R(0.64));
        HWYBLAS_CALL_COMPLEX(T, rot, &n, P(x.data()), &incx, P(y.data()), &incy,
                             &c, P(&s));
        for (hwyblas_int i = 0; i < n; ++i) {
          const size_t ix = incx > 0 ? i * incx : (n - 1 - i) * -incx;
          const size_t iy = incy > 0 ? i * incy : (n - 1 - i) * -incy;
          const T ex = c * x0[ix] + s * y0[iy];
          const T ey = c * y0[iy] - std::conj(s) * x0[ix];
          EXPECT_LE(std::abs(x[ix] - ex), 4 * Eps<T>());
          EXPECT_LE(std::abs(y[iy] - ey), 4 * Eps<T>());
        }
      }
      if (incx < 0) continue;
      const std::vector<T> x = rng.Vector(Sz(n) * incx);
      hwyblas_int imax = -1;
      R sum = -1;
      if constexpr (std::is_same_v<T, std::complex<float>>) {
        imax = icmax1_(&n, P(x.data()), &incx);
        sum = scsum1_(&n, P(x.data()), &incx);
      } else {
        imax = izmax1_(&n, P(x.data()), &incx);
        sum = dzsum1_(&n, P(x.data()), &incx);
      }
      hwyblas_int expect_max = 0;
      double expect_sum = 0, best = -1;
      for (hwyblas_int i = 0; i < n; ++i) {
        const double a = std::abs(x[i * incx]);
        expect_sum += a;
        if (a > best) {
          best = a;
          expect_max = i + 1;
        }
      }
      EXPECT_EQ(imax, expect_max);
      EXPECT_NEAR(sum, expect_sum, 4 * n * Eps<T>() * (1 + expect_sum));
    }
  }
}

// ---------------------------------------------------------------------------
// Workspace queries and invalid arguments

TYPED_TEST(GeesTest, WorkspaceQueriesAndErrors) {
  using T = TypeParam;
  const hwyblas_int n = 10;
  std::vector<T> a(Sz2(n, n), T(1)), vs(Sz2(n, n));
  std::vector<Eig<T>> w;
  hwyblas_int sdim = 0;
  // xGEES: the reference's optimal sizes.
  ASSERT_EQ(Gees<T>('V', 'N', Select::kNone, n, a, n, &sdim, w, vs, n, -1), 0);
  EXPECT_GE(static_cast<hwyblas_int>(w[0].real()), MinimumGeesLwork<T>(n));
  ASSERT_EQ(Gees<T>('N', 'N', Select::kNone, 0, a, 1, &sdim, w, vs, 1, -1), 0);
  EXPECT_EQ(w[0].real(), 1);

  ErrorCapture errors;
  const std::string gees = Name<T>("GEES");
  EXPECT_EQ(Gees<T>('X', 'N', Select::kNone, n, a, n, &sdim, w, vs, n, 64 * n),
            -1);
  EXPECT_EQ(Gees<T>('V', 'X', Select::kNone, n, a, n, &sdim, w, vs, n, 64 * n),
            -2);
  EXPECT_EQ(Gees<T>('V', 'N', Select::kNone, -1, a, n, &sdim, w, vs, n, 64 * n),
            -4);
  EXPECT_EQ(
      Gees<T>('V', 'N', Select::kNone, n, a, n - 1, &sdim, w, vs, n, 64 * n),
      -6);
  const hwyblas_int ldvs_pos = IsComplex<T>::value ? 10 : 11;
  const hwyblas_int lwork_pos = IsComplex<T>::value ? 12 : 13;
  EXPECT_EQ(
      Gees<T>('V', 'N', Select::kNone, n, a, n, &sdim, w, vs, n - 1, 64 * n),
      -ldvs_pos);
  EXPECT_EQ(Gees<T>('N', 'N', Select::kNone, n, a, n, &sdim, w, vs, 0, 64 * n),
            -ldvs_pos);
  EXPECT_EQ(Gees<T>('V', 'N', Select::kNone, n, a, n, &sdim, w, vs, n,
                    MinimumGeesLwork<T>(n) - 1),
            -lwork_pos);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{gees + ":1", gees + ":2", gees + ":4",
                                      gees + ":6",
                                      gees + ":" + std::to_string(ldvs_pos),
                                      gees + ":" + std::to_string(ldvs_pos),
                                      gees + ":" + std::to_string(lwork_pos)}));

  // xHSEQR.
  std::vector<T> h(Sz2(n, n)), z(Sz2(n, n));
  const std::string hseqr = Name<T>("HSEQR");
  const hwyblas_int hz = IsComplex<T>::value ? 10 : 11;
  const hwyblas_int hw = IsComplex<T>::value ? 12 : 13;
  EXPECT_EQ(Hseqr<T>('X', 'N', n, 1, n, h, n, w, z, n, n), -1);
  EXPECT_EQ(Hseqr<T>('E', 'X', n, 1, n, h, n, w, z, n, n), -2);
  EXPECT_EQ(Hseqr<T>('E', 'N', -1, 1, n, h, n, w, z, n, n), -3);
  EXPECT_EQ(Hseqr<T>('E', 'N', n, 0, n, h, n, w, z, n, n), -4);
  EXPECT_EQ(Hseqr<T>('E', 'N', n, 2, 1, h, n, w, z, n, n), -5);
  EXPECT_EQ(Hseqr<T>('E', 'N', n, 1, n, h, n - 1, w, z, n, n), -7);
  EXPECT_EQ(Hseqr<T>('E', 'V', n, 1, n, h, n, w, z, n - 1, n), -hz);
  EXPECT_EQ(Hseqr<T>('E', 'N', n, 1, n, h, n, w, z, 1, n - 1), -hw);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{hseqr + ":1", hseqr + ":2", hseqr + ":3",
                                      hseqr + ":4", hseqr + ":5", hseqr + ":7",
                                      hseqr + ":" + std::to_string(hz),
                                      hseqr + ":" + std::to_string(hw)}));

  // xGEHRD, xORGHR / xUNGHR, xORMHR / xUNMHR, xGEBAK.
  std::vector<T> tau(Sz(n)), work(Sz(64 * n)), c(Sz2(n, n));
  hwyblas_int info = 0, lwork = 64 * n, ilo = 1, ihi = n, mm = n;
  const hwyblas_int nm1 = n - 1, zero = 0, two = 2, neg = -1;
  HWYBLAS_CALL(T, gehrd, &n, &ilo, &ihi, P(a.data()), &nm1, P(tau.data()),
               P(work.data()), &lwork, &info);
  EXPECT_EQ(info, -5);
  HWYBLAS_CALL(T, gehrd, &n, &two, &ilo, P(a.data()), &n, P(tau.data()),
               P(work.data()), &lwork, &info);
  EXPECT_EQ(info, -3);
  HWYBLAS_CALL(T, gehrd, &n, &ilo, &ihi, P(a.data()), &n, P(tau.data()),
               P(work.data()), &nm1, &info);
  EXPECT_EQ(info, -8);
  CALL_RC(T, orghr, unghr, &n, &ilo, &ihi, P(a.data()), &n, P(tau.data()),
          P(work.data()), &zero, &info);
  EXPECT_EQ(info, -8);
  const char ct = IsComplex<T>::value ? 'C' : 'T';
  const char wrong = IsComplex<T>::value ? 'T' : 'C';
  CALL_RC(T, ormhr, unmhr, "L", &wrong, &mm, &n, &ilo, &ihi, P(a.data()), &n,
          P(tau.data()), P(c.data()), &n, P(work.data()), &lwork, &info);
  EXPECT_EQ(info, -2);
  CALL_RC(T, ormhr, unmhr, "L", &ct, &mm, &n, &ilo, &ihi, P(a.data()), &n,
          P(tau.data()), P(c.data()), &n, P(work.data()), &nm1, &info);
  EXPECT_EQ(info, -13);
  std::vector<RealT<T>> scale(Sz(n), 1);
  HWYBLAS_CALL(T, gebak, "B", "R", &n, &ilo, &ihi, scale.data(), &neg,
               P(c.data()), &n, &info);
  EXPECT_EQ(info, -7);
  const std::string og = Name<T>(IsComplex<T>::value ? "UNGHR" : "ORGHR");
  const std::string om = Name<T>(IsComplex<T>::value ? "UNMHR" : "ORMHR");
  EXPECT_EQ(errors.Take(), (std::vector<std::string>{
                               Name<T>("GEHRD") + ":5", Name<T>("GEHRD") + ":3",
                               Name<T>("GEHRD") + ":8", og + ":8", om + ":2",
                               om + ":13", Name<T>("GEBAK") + ":7"}));

  // xTRSEN, xTREXC, xTRSYL.
  std::vector<hwyblas_int> select(Sz(n), 0);
  hwyblas_int m = 0;
  RealT<T> s = 0, sep = 0;
  if constexpr (IsComplex<T>::value) {
    HWYBLAS_CALL_COMPLEX(T, trsen, "X", "N", select.data(), &n, P(a.data()), &n,
                         P(z.data()), &n, P(tau.data()), &m, &s, &sep,
                         P(work.data()), &lwork, &info);
    EXPECT_EQ(info, -1);
    HWYBLAS_CALL_COMPLEX(T, trexc, "V", &n, P(a.data()), &n, P(z.data()), &nm1,
                         &ilo, &ihi, &info);
    EXPECT_EQ(info, -6);
  } else {
    std::vector<hwyblas_int> iwork(Sz(n));
    std::vector<T> wi(Sz(n));
    HWYBLAS_CALL_REAL(T, trsen, "B", "N", select.data(), &n, a.data(), &n,
                      z.data(), &n, tau.data(), wi.data(), &m, &s, &sep,
                      work.data(), &lwork, iwork.data(), &zero, &info);
    EXPECT_EQ(info, -17);
    hwyblas_int f = n + 1, l = 1;
    HWYBLAS_CALL_REAL(T, trexc, "V", &n, a.data(), &n, z.data(), &n, &f, &l,
                      work.data(), &info);
    EXPECT_EQ(info, -7);
  }
  RealT<T> sc = 0;
  HWYBLAS_CALL(T, trsyl, "N", "N", &two, &n, &n, P(a.data()), &n, P(h.data()),
               &n, P(c.data()), &n, &sc, &info);
  EXPECT_EQ(info, -3);
  EXPECT_EQ(errors.Take(),
            (std::vector<std::string>{
                Name<T>("TRSEN") + (IsComplex<T>::value ? ":1" : ":17"),
                Name<T>("TREXC") + (IsComplex<T>::value ? ":6" : ":7"),
                Name<T>("TRSYL") + ":3"}));
}

// ---------------------------------------------------------------------------
// LAPACKE

template <typename T>
std::vector<T> ToRowMajor(const std::vector<T>& col, lapack_int ld) {
  std::vector<T> row(col.size());
  for (lapack_int j = 0; j < ld; ++j) {
    for (lapack_int i = 0; i < ld; ++i) row[i * ld + j] = col[i + j * ld];
  }
  return row;
}

template <typename T>
void ExpectSameRowMajor(const std::vector<T>& row, const std::vector<T>& col,
                        lapack_int m, lapack_int n, lapack_int ld) {
  for (lapack_int i = 0; i < m; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      ASSERT_EQ(row[i * ld + j], col[i + j * ld]) << i << ", " << j;
    }
  }
}

#define LAPACKE_CALL(T, rname, cname, ...)                         \
  [&] {                                                            \
    if constexpr (std::is_same_v<T, float>) {                      \
      return LAPACKE_s##rname(__VA_ARGS__);                        \
    } else if constexpr (std::is_same_v<T, double>) {              \
      return LAPACKE_d##rname(__VA_ARGS__);                        \
    } else if constexpr (std::is_same_v<T, std::complex<float>>) { \
      return LAPACKE_c##cname(__VA_ARGS__);                        \
    } else {                                                       \
      return LAPACKE_z##cname(__VA_ARGS__);                        \
    }                                                              \
  }()

// LAPACKE_?gees and ?hseqr with the eigenvalues as complex numbers.
template <typename T>
lapack_int LapackeGees(int layout, char jobvs, char sort, lapack_int n, T* a,
                       lapack_int lda, lapack_int* sdim, std::vector<Eig<T>>& w,
                       T* vs, lapack_int ldvs) {
  using R = RealT<T>;
  w.assign(Sz(n), Eig<T>());
  if constexpr (IsComplex<T>::value) {
    lapack_logical (*sel)(const T*) = [](const T* x) -> lapack_logical {
      return x->real() > 0;
    };
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      return LAPACKE_cgees(layout, jobvs, sort, sel, n, a, lda, sdim, w.data(),
                           vs, ldvs);
    } else {
      return LAPACKE_zgees(layout, jobvs, sort, sel, n, a, lda, sdim, w.data(),
                           vs, ldvs);
    }
  } else {
    lapack_logical (*sel)(const R*, const R*) =
        [](const R* re, const R*) -> lapack_logical { return *re > 0; };
    std::vector<R> wr(Sz(n)), wi(Sz(n));
    lapack_int info;
    if constexpr (std::is_same_v<T, float>) {
      info = LAPACKE_sgees(layout, jobvs, sort, sel, n, a, lda, sdim, wr.data(),
                           wi.data(), vs, ldvs);
    } else {
      info = LAPACKE_dgees(layout, jobvs, sort, sel, n, a, lda, sdim, wr.data(),
                           wi.data(), vs, ldvs);
    }
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
    return info;
  }
}

template <typename T>
lapack_int LapackeHseqr(int layout, char job, char compz, lapack_int n,
                        lapack_int ilo, lapack_int ihi, T* h, lapack_int ldh,
                        std::vector<Eig<T>>& w, T* z, lapack_int ldz) {
  using R = RealT<T>;
  w.assign(Sz(n), Eig<T>());
  if constexpr (IsComplex<T>::value) {
    return LAPACKE_CALL(T, hseqr, hseqr, layout, job, compz, n, ilo, ihi, h,
                        ldh, w.data(), z, ldz);
  } else {
    std::vector<R> wr(Sz(n)), wi(Sz(n));
    const lapack_int info =
        LAPACKE_CALL(T, hseqr, hseqr, layout, job, compz, n, ilo, ihi, h, ldh,
                     wr.data(), wi.data(), z, ldz);
    for (size_t i = 0; i < wr.size(); ++i) w[i] = Eig<T>(wr[i], wi[i]);
    return info;
  }
}

// Row-major results equal the column-major ones.
TYPED_TEST(GeesTest, LapackeLayouts) {
  using T = TypeParam;
  using R = RealT<T>;
  Random<T> rng(127);
  const lapack_int n = 9, ld = 11;
  const std::vector<T> a0 = rng.Vector(static_cast<size_t>(ld) * ld);
  const std::vector<T> ar0 = ToRowMajor(a0, ld);
  // ?gees.
  for (const char jobvs : {'V', 'N'}) {
    for (const char sort : {'N', 'S'}) {
      SCOPED_TRACE(::testing::Message() << "gees " << jobvs << sort);
      std::vector<T> col = a0, row = ar0;
      std::vector<T> vc(static_cast<size_t>(ld) * ld), vr(vc.size());
      std::vector<Eig<T>> wc, wr;
      lapack_int sc = -1, sr = -1;
      ASSERT_EQ(LapackeGees<T>(LAPACK_COL_MAJOR, jobvs, sort, n, col.data(), ld,
                               &sc, wc, vc.data(), ld),
                0);
      ASSERT_EQ(LapackeGees<T>(LAPACK_ROW_MAJOR, jobvs, sort, n, row.data(), ld,
                               &sr, wr, vr.data(), ld),
                0);
      EXPECT_EQ(wr, wc);
      EXPECT_EQ(sr, sc);
      ExpectSameRowMajor(row, col, n, n, ld);
      if (jobvs == 'V') ExpectSameRowMajor(vr, vc, n, n, ld);
    }
  }
  // ?gebal, ?gebak, ?gehrd, ?orghr / ?unghr, ?ormhr / ?unmhr, ?hseqr.
  std::vector<T> col = a0, row = ar0;
  std::vector<R> sc(n), sr(n);
  lapack_int ilo_c, ihi_c, ilo_r, ihi_r;
  ASSERT_EQ(LAPACKE_CALL(T, gebal, gebal, LAPACK_COL_MAJOR, 'B', n, col.data(),
                         ld, &ilo_c, &ihi_c, sc.data()),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, gebal, gebal, LAPACK_ROW_MAJOR, 'B', n, row.data(),
                         ld, &ilo_r, &ihi_r, sr.data()),
            0);
  EXPECT_EQ(sr, sc);
  EXPECT_EQ(ilo_r, ilo_c);
  ExpectSameRowMajor(row, col, n, n, ld);
  std::vector<T> vc = rng.Vector(static_cast<size_t>(ld) * ld);
  std::vector<T> vr = ToRowMajor(vc, ld);
  ASSERT_EQ(LAPACKE_CALL(T, gebak, gebak, LAPACK_COL_MAJOR, 'B', 'R', n, ilo_c,
                         ihi_c, sc.data(), 5, vc.data(), ld),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, gebak, gebak, LAPACK_ROW_MAJOR, 'B', 'R', n, ilo_c,
                         ihi_c, sc.data(), 5, vr.data(), ld),
            0);
  ExpectSameRowMajor(vr, vc, n, 5, ld);
  std::vector<T> tc(n), tr(n);
  ASSERT_EQ(LAPACKE_CALL(T, gehrd, gehrd, LAPACK_COL_MAJOR, n, 1, n, col.data(),
                         ld, tc.data()),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, gehrd, gehrd, LAPACK_ROW_MAJOR, n, 1, n, row.data(),
                         ld, tr.data()),
            0);
  EXPECT_EQ(tr, tc);
  ExpectSameRowMajor(row, col, n, n, ld);
  std::vector<T> cc = rng.Vector(static_cast<size_t>(ld) * ld);
  std::vector<T> cr = ToRowMajor(cc, ld);
  const char ct = IsComplex<T>::value ? 'C' : 'T';
  ASSERT_EQ(LAPACKE_CALL(T, ormhr, unmhr, LAPACK_COL_MAJOR, 'R', ct, 4, n, 1, n,
                         col.data(), ld, tc.data(), cc.data(), ld),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, ormhr, unmhr, LAPACK_ROW_MAJOR, 'R', ct, 4, n, 1, n,
                         row.data(), ld, tr.data(), cr.data(), ld),
            0);
  ExpectSameRowMajor(cr, cc, 4, n, ld);
  std::vector<T> qc = col, qr = row;
  ASSERT_EQ(LAPACKE_CALL(T, orghr, unghr, LAPACK_COL_MAJOR, n, 1, n, qc.data(),
                         ld, tc.data()),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, orghr, unghr, LAPACK_ROW_MAJOR, n, 1, n, qr.data(),
                         ld, tr.data()),
            0);
  ExpectSameRowMajor(qr, qc, n, n, ld);
  for (lapack_int j = 0; j < n; ++j) {
    for (lapack_int i = j + 2; i < n; ++i) {
      col[i + j * ld] = T(0);
      row[i * ld + j] = T(0);
    }
  }
  for (const char compz : {'V', 'I', 'N'}) {
    SCOPED_TRACE(compz);
    std::vector<T> hc = col, hr = row, zc = qc, zr = qr;
    std::vector<Eig<T>> wc, wr;
    ASSERT_EQ(LapackeHseqr<T>(LAPACK_COL_MAJOR, 'S', compz, n, 1, n, hc.data(),
                              ld, wc, zc.data(), ld),
              0);
    ASSERT_EQ(LapackeHseqr<T>(LAPACK_ROW_MAJOR, 'S', compz, n, 1, n, hr.data(),
                              ld, wr, zr.data(), ld),
              0);
    EXPECT_EQ(wr, wc);
    ExpectSameRowMajor(hr, hc, n, n, ld);
    ExpectSameRowMajor(zr, zc, n, n, ld);

    // ?trexc, ?trsen on the Schur form, ?trsyl.
    if (compz != 'V') continue;
    std::vector<T> t2c = hc, t2r = hr, q2c = zc, q2r = zr;
    lapack_int f = 1, l = n;
    if constexpr (IsComplex<T>::value) {
      ASSERT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_COL_MAJOR, 'V', n,
                             t2c.data(), ld, q2c.data(), ld, f, l),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_ROW_MAJOR, 'V', n,
                             t2r.data(), ld, q2r.data(), ld, f, l),
                0);
    } else {
      lapack_int f2 = 1, l2 = n;
      ASSERT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_COL_MAJOR, 'V', n,
                             t2c.data(), ld, q2c.data(), ld, &f, &l),
                0);
      ASSERT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_ROW_MAJOR, 'V', n,
                             t2r.data(), ld, q2r.data(), ld, &f2, &l2),
                0);
      EXPECT_EQ(f2, f);
      EXPECT_EQ(l2, l);
    }
    ExpectSameRowMajor(t2r, t2c, n, n, ld);
    ExpectSameRowMajor(q2r, q2c, n, n, ld);
    for (const char job : {'N', 'B'}) {
      std::vector<lapack_logical> select(n, 0);
      select[n - 1] = select[2] = 1;
      std::vector<T> t3c = hc, t3r = hr, q3c = zc, q3r = zr;
      lapack_int mc = -1, mr = -1;
      R s_c = 0, s_r = 0, sep_c = 0, sep_r = 0;
      if constexpr (IsComplex<T>::value) {
        std::vector<T> w3c(n), w3r(n);
        ASSERT_EQ(LAPACKE_CALL(T, trsen, trsen, LAPACK_COL_MAJOR, job, 'V',
                               select.data(), n, t3c.data(), ld, q3c.data(), ld,
                               w3c.data(), &mc, &s_c, &sep_c),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, trsen, trsen, LAPACK_ROW_MAJOR, job, 'V',
                               select.data(), n, t3r.data(), ld, q3r.data(), ld,
                               w3r.data(), &mr, &s_r, &sep_r),
                  0);
        EXPECT_EQ(w3r, w3c);
      } else {
        std::vector<R> w3c(n), w3r(n), wi3c(n), wi3r(n);
        ASSERT_EQ(LAPACKE_CALL(T, trsen, trsen, LAPACK_COL_MAJOR, job, 'V',
                               select.data(), n, t3c.data(), ld, q3c.data(), ld,
                               w3c.data(), wi3c.data(), &mc, &s_c, &sep_c),
                  0);
        ASSERT_EQ(LAPACKE_CALL(T, trsen, trsen, LAPACK_ROW_MAJOR, job, 'V',
                               select.data(), n, t3r.data(), ld, q3r.data(), ld,
                               w3r.data(), wi3r.data(), &mr, &s_r, &sep_r),
                  0);
        EXPECT_EQ(w3r, w3c);
        EXPECT_EQ(wi3r, wi3c);
      }
      EXPECT_EQ(mr, mc);
      EXPECT_EQ(s_r, s_c);
      EXPECT_EQ(sep_r, sep_c);
      ExpectSameRowMajor(t3r, t3c, n, n, ld);
      ExpectSameRowMajor(q3r, q3c, n, n, ld);
    }
    // ?trsyl with A = T (n x n) and B = T(1:3, 1:3) + 3 I.
    std::vector<T> bc = hc, br = hr;
    for (lapack_int i = 0; i < 3; ++i) {
      bc[i + i * ld] += T(3);
      br[i * ld + i] += T(3);
    }
    std::vector<T> xc = rng.Vector(static_cast<size_t>(ld) * ld);
    std::vector<T> xr = ToRowMajor(xc, ld);
    R scl_c = 0, scl_r = 0;
    ASSERT_EQ(LAPACKE_CALL(T, trsyl, trsyl, LAPACK_COL_MAJOR, 'N', ct, 1, n, 3,
                           hc.data(), ld, bc.data(), ld, xc.data(), ld, &scl_c),
              0);
    ASSERT_EQ(LAPACKE_CALL(T, trsyl, trsyl, LAPACK_ROW_MAJOR, 'N', ct, 1, n, 3,
                           hr.data(), ld, br.data(), ld, xr.data(), ld, &scl_r),
              0);
    EXPECT_EQ(scl_r, scl_c);
    ExpectSameRowMajor(xr, xc, n, 3, ld);
  }
  // ?larfx.
  const std::vector<T> v = rng.Vector(n);
  const T tau = rng.Next();
  std::vector<T> lc = rng.Vector(static_cast<size_t>(ld) * ld);
  std::vector<T> lr = ToRowMajor(lc, ld), work(n);
  ASSERT_EQ(LAPACKE_CALL(T, larfx, larfx, LAPACK_COL_MAJOR, 'L', 5, 4, v.data(),
                         tau, lc.data(), ld, work.data()),
            0);
  ASSERT_EQ(LAPACKE_CALL(T, larfx, larfx, LAPACK_ROW_MAJOR, 'L', 5, 4, v.data(),
                         tau, lr.data(), ld, work.data()),
            0);
  ExpectSameRowMajor(lr, lc, 5, 4, ld);
}

TYPED_TEST(GeesTest, LapackeErrors) {
  using T = TypeParam;
  using R = RealT<T>;
  std::vector<T> a(16, T(1)), vs(16);
  std::vector<Eig<T>> w;
  lapack_int sdim = 0;
  ::testing::internal::CaptureStdout();
  EXPECT_EQ(LapackeGees<T>(0, 'V', 'N', 4, a.data(), 4, &sdim, w, vs.data(), 4),
            -1);
  EXPECT_EQ(LapackeGees<T>(LAPACK_ROW_MAJOR, 'V', 'N', 4, a.data(), 3, &sdim, w,
                           vs.data(), 4),
            -7);
  const lapack_int vs_pos = IsComplex<T>::value ? 11 : 12;
  EXPECT_EQ(LapackeGees<T>(LAPACK_ROW_MAJOR, 'N', 'N', 4, a.data(), 4, &sdim, w,
                           vs.data(), 3),
            -vs_pos);
  std::vector<T> z(16);
  const lapack_int hz = IsComplex<T>::value ? 11 : 12;
  EXPECT_EQ(LapackeHseqr<T>(LAPACK_ROW_MAJOR, 'E', 'N', 4, 1, 4, a.data(), 4, w,
                            z.data(), 3),
            -hz);
  // ?trexc checks LDQ only for COMPQ = 'V'.
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_ROW_MAJOR, 'N', 4, a.data(),
                           4, z.data(), 1, 1, 2),
              0);
    EXPECT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_ROW_MAJOR, 'V', 4, a.data(),
                           4, z.data(), 1, 1, 2),
              -7);
  } else {
    lapack_int f = 1, l = 2;
    std::vector<T> t = {1, 0, 0, 0, 1, 2, 0, 0, 1, 1, 3, 0, 1, 1, 1, 4};
    EXPECT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_ROW_MAJOR, 'N', 4, t.data(),
                           4, z.data(), 1, &f, &l),
              0);
    EXPECT_EQ(LAPACKE_CALL(T, trexc, trexc, LAPACK_ROW_MAJOR, 'V', 4, t.data(),
                           4, z.data(), 1, &f, &l),
              -7);
  }
  const std::string gees = LapackeName<T>("gees");
  EXPECT_EQ(::testing::internal::GetCapturedStdout(),
            "Wrong parameter 1 in " + gees + "\nWrong parameter 7 in " + gees +
                "_work\nWrong parameter " + std::to_string(vs_pos) + " in " +
                gees + "_work\nWrong parameter " + std::to_string(hz) + " in " +
                LapackeName<T>("hseqr_work") + "\nWrong parameter 7 in " +
                LapackeName<T>("trexc_work") + "\n");
  // NaN checks.
  a[5] = NaN<T>();
  EXPECT_EQ(LapackeGees<T>(LAPACK_COL_MAJOR, 'V', 'N', 4, a.data(), 4, &sdim, w,
                           vs.data(), 4),
            -6);
  EXPECT_EQ(LapackeHseqr<T>(LAPACK_COL_MAJOR, 'E', 'N', 4, 1, 4, a.data(), 4, w,
                            z.data(), 4),
            -7);
  std::vector<R> scale(4);
  lapack_int ilo = 0, ihi = 0;
  EXPECT_EQ(LAPACKE_CALL(T, gebal, gebal, LAPACK_COL_MAJOR, 'B', 4, a.data(), 4,
                         &ilo, &ihi, scale.data()),
            -4);
  EXPECT_EQ(LAPACKE_CALL(T, gebal, gebal, LAPACK_COL_MAJOR, 'N', 4, a.data(), 4,
                         &ilo, &ihi, scale.data()),
            0);
  std::vector<T> tau(4, T(0));
  EXPECT_EQ(LAPACKE_CALL(T, gehrd, gehrd, LAPACK_COL_MAJOR, 4, 1, 4, a.data(),
                         4, tau.data()),
            -5);
  R sc = 0;
  EXPECT_EQ(LAPACKE_CALL(T, trsyl, trsyl, LAPACK_COL_MAJOR, 'N', 'N', 1, 4, 4,
                         a.data(), 4, vs.data(), 4, z.data(), 4, &sc),
            -7);
  // ?lacn2: NaN in EST (-5, also for complex types, as in the reference) and
  // X (-3).
  std::vector<T> v(4), x(4, T(1));
  R est = NAN;
  lapack_int kase = 0, isave[3] = {};
  std::vector<lapack_int> isgn(4);
  if constexpr (IsComplex<T>::value) {
    EXPECT_EQ(LAPACKE_CALL(T, lacn2, lacn2, 4, v.data(), x.data(), &est, &kase,
                           isave),
              -5);
    est = 0;
    x[2] = NaN<T>();
    EXPECT_EQ(LAPACKE_CALL(T, lacn2, lacn2, 4, v.data(), x.data(), &est, &kase,
                           isave),
              -3);
  } else {
    EXPECT_EQ(LAPACKE_CALL(T, lacn2, lacn2, 4, v.data(), x.data(), isgn.data(),
                           &est, &kase, isave),
              -5);
    est = 0;
    x[2] = NaN<T>();
    EXPECT_EQ(LAPACKE_CALL(T, lacn2, lacn2, 4, v.data(), x.data(), isgn.data(),
                           &est, &kase, isave),
              -3);
  }
}

}  // namespace
}  // namespace hwyblas_test
