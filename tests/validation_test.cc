// Argument validation of the BLAS routines: for each routine, calls with
// each argument invalid in turn must report exactly that argument (the first
// invalid one), and calls with the smallest valid values (zero dimensions
// with a leading dimension of 1, a leading dimension equal to the number of
// rows, an increment of -1) must not report anything.
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "hwyblas/cblas.h"
#include "hwyblas/hwyblas.h"
#include "lapack/lapack_test_util.h"
#include "test_util.h"

namespace hwyblas_test {
namespace {

template <typename T>
class ValidationTest : public ::testing::Test {
 protected:
  // The errors reported by `call`, as "DGEMM:13".
  template <typename F>
  static std::vector<std::string> Reported(F&& call) {
    ErrorCapture capture;
    call();
    return capture.Take();
  }

  // `call` reports argument `position` of `routine` (0: nothing).
  template <typename F>
  static void Expect(const char* routine, int position, F&& call) {
    std::vector<std::string> expected;
    if (position != 0) {
      expected.push_back(Name<T>(routine) + ":" + std::to_string(position));
    }
    EXPECT_EQ(Reported(std::forward<F>(call)), expected)
        << routine << " argument " << position;
  }

  std::vector<T> a = std::vector<T>(256, T(0));
  std::vector<T> b = std::vector<T>(256, T(0));
  std::vector<T> c = std::vector<T>(256, T(0));
  T alpha = T(1), beta = T(1);
  RealT<T> ralpha = 1, rbeta = 1;
};

using ScalarTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ValidationTest, ScalarTypes);

// ---------------------------------------------------------------------------
// Level 3

TYPED_TEST(ValidationTest, Gemm) {
  using T = TypeParam;
  auto gemm = [&](char ta, char tb, int m, int n, int k, int lda, int ldb,
                  int ldc) {
    return [=] {
      HWYBLAS_CALL(T, gemm, &ta, &tb, &m, &n, &k, P(&this->alpha),
                   P(this->a.data()), &lda, P(this->b.data()), &ldb,
                   P(&this->beta), P(this->c.data()), &ldc);
    };
  };
  const auto E = [](int position, auto&& call) {
    TestFixture::Expect("GEMM", position, call);
  };
  E(0, gemm('N', 'N', 0, 0, 0, 1, 1, 1));
  E(0, gemm('N', 'N', 1, 1, 1, 1, 1, 1));
  E(0, gemm('T', 'T', 3, 2, 4, 4, 2, 3));
  E(1, gemm('X', 'N', 1, 1, 1, 1, 1, 1));
  E(2, gemm('N', 'X', 1, 1, 1, 1, 1, 1));
  E(3, gemm('N', 'N', -1, 1, 1, 1, 1, 1));
  E(4, gemm('N', 'N', 1, -1, 1, 1, 1, 1));
  E(5, gemm('N', 'N', 1, 1, -1, 1, 1, 1));
  // A is m x k if not transposed, k x m if transposed.
  E(8, gemm('N', 'N', 3, 1, 1, 2, 1, 3));
  E(8, gemm('T', 'N', 1, 1, 3, 2, 3, 1));
  E(8, gemm('N', 'N', 0, 0, 0, 0, 1, 1));
  E(10, gemm('N', 'N', 1, 1, 3, 1, 2, 1));
  E(10, gemm('N', 'T', 1, 3, 1, 1, 2, 1));
  E(10, gemm('N', 'N', 0, 0, 0, 1, 0, 1));
  E(13, gemm('N', 'N', 3, 1, 1, 3, 1, 2));
  E(13, gemm('N', 'N', 0, 0, 0, 1, 1, 0));
}

TYPED_TEST(ValidationTest, SymmHemm) {
  using T = TypeParam;
  auto symm = [&](bool hermitian, char side, char uplo, int m, int n, int lda,
                  int ldb, int ldc) {
    return [=] {
      if (hermitian) {
        HWYBLAS_CALL_COMPLEX(T, hemm, &side, &uplo, &m, &n, P(&this->alpha),
                             P(this->a.data()), &lda, P(this->b.data()), &ldb,
                             P(&this->beta), P(this->c.data()), &ldc);
      } else {
        HWYBLAS_CALL(T, symm, &side, &uplo, &m, &n, P(&this->alpha),
                     P(this->a.data()), &lda, P(this->b.data()), &ldb,
                     P(&this->beta), P(this->c.data()), &ldc);
      }
    };
  };
  for (const bool hermitian : {false, true}) {
    if (hermitian && !IsComplex<T>::value) continue;
    const char* name = hermitian ? "HEMM" : "SYMM";
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, symm(hermitian, 'L', 'U', 0, 0, 1, 1, 1));
    E(0, symm(hermitian, 'L', 'L', 1, 1, 1, 1, 1));
    E(0, symm(hermitian, 'R', 'U', 2, 3, 3, 2, 2));
    E(1, symm(hermitian, 'X', 'U', 1, 1, 1, 1, 1));
    E(2, symm(hermitian, 'L', 'X', 1, 1, 1, 1, 1));
    E(3, symm(hermitian, 'L', 'U', -1, 1, 1, 1, 1));
    E(4, symm(hermitian, 'L', 'U', 1, -1, 1, 1, 1));
    // A is m x m on the left and n x n on the right.
    E(7, symm(hermitian, 'L', 'U', 3, 1, 2, 3, 3));
    E(7, symm(hermitian, 'R', 'U', 1, 3, 2, 1, 1));
    E(7, symm(hermitian, 'L', 'U', 0, 0, 0, 1, 1));
    E(9, symm(hermitian, 'L', 'U', 3, 1, 3, 2, 3));
    E(9, symm(hermitian, 'L', 'U', 0, 0, 1, 0, 1));
    E(12, symm(hermitian, 'L', 'U', 3, 1, 3, 3, 2));
    E(12, symm(hermitian, 'L', 'U', 0, 0, 1, 1, 0));
  }
}

TYPED_TEST(ValidationTest, SyrkHerk) {
  using T = TypeParam;
  auto syrk = [&](bool hermitian, char uplo, char trans, int n, int k, int lda,
                  int ldc) {
    return [=] {
      if (hermitian) {
        HWYBLAS_CALL_COMPLEX(T, herk, &uplo, &trans, &n, &k, &this->ralpha,
                             P(this->a.data()), &lda, &this->rbeta,
                             P(this->c.data()), &ldc);
      } else {
        HWYBLAS_CALL(T, syrk, &uplo, &trans, &n, &k, P(&this->alpha),
                     P(this->a.data()), &lda, P(&this->beta), P(this->c.data()),
                     &ldc);
      }
    };
  };
  for (const bool hermitian : {false, true}) {
    if (hermitian && !IsComplex<T>::value) continue;
    const char* name = hermitian ? "HERK" : "SYRK";
    const char other = hermitian ? 'T' : 'C';  // invalid for complex T
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, syrk(hermitian, 'U', 'N', 0, 0, 1, 1));
    E(0, syrk(hermitian, 'L', 'N', 1, 1, 1, 1));
    E(0, syrk(hermitian, 'L', hermitian ? 'C' : 'T', 2, 3, 3, 2));
    E(1, syrk(hermitian, 'X', 'N', 1, 1, 1, 1));
    E(2, syrk(hermitian, 'U', 'X', 1, 1, 1, 1));
    if (IsComplex<T>::value) E(2, syrk(hermitian, 'U', other, 1, 1, 1, 1));
    E(3, syrk(hermitian, 'U', 'N', -1, 1, 1, 1));
    E(4, syrk(hermitian, 'U', 'N', 1, -1, 1, 1));
    // A is n x k if not transposed, k x n if transposed.
    E(7, syrk(hermitian, 'U', 'N', 3, 1, 2, 3));
    E(7, syrk(hermitian, 'U', hermitian ? 'C' : 'T', 1, 3, 2, 1));
    E(7, syrk(hermitian, 'U', 'N', 0, 0, 0, 1));
    E(10, syrk(hermitian, 'U', 'N', 3, 1, 3, 2));
    E(10, syrk(hermitian, 'U', 'N', 0, 0, 1, 0));
  }
}

TYPED_TEST(ValidationTest, Syr2kHer2k) {
  using T = TypeParam;
  auto syr2k = [&](bool hermitian, char uplo, char trans, int n, int k, int lda,
                   int ldb, int ldc) {
    return [=] {
      if (hermitian) {
        HWYBLAS_CALL_COMPLEX(T, her2k, &uplo, &trans, &n, &k, P(&this->alpha),
                             P(this->a.data()), &lda, P(this->b.data()), &ldb,
                             &this->rbeta, P(this->c.data()), &ldc);
      } else {
        HWYBLAS_CALL(T, syr2k, &uplo, &trans, &n, &k, P(&this->alpha),
                     P(this->a.data()), &lda, P(this->b.data()), &ldb,
                     P(&this->beta), P(this->c.data()), &ldc);
      }
    };
  };
  for (const bool hermitian : {false, true}) {
    if (hermitian && !IsComplex<T>::value) continue;
    const char* name = hermitian ? "HER2K" : "SYR2K";
    const char other = hermitian ? 'T' : 'C';
    const char trans = hermitian ? 'C' : 'T';
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, syr2k(hermitian, 'U', 'N', 0, 0, 1, 1, 1));
    E(0, syr2k(hermitian, 'L', 'N', 1, 1, 1, 1, 1));
    E(0, syr2k(hermitian, 'L', trans, 2, 3, 3, 3, 2));
    E(1, syr2k(hermitian, 'X', 'N', 1, 1, 1, 1, 1));
    E(2, syr2k(hermitian, 'U', 'X', 1, 1, 1, 1, 1));
    if (IsComplex<T>::value) {
      E(2, syr2k(hermitian, 'U', other, 1, 1, 1, 1, 1));
    }
    E(3, syr2k(hermitian, 'U', 'N', -1, 1, 1, 1, 1));
    E(4, syr2k(hermitian, 'U', 'N', 1, -1, 1, 1, 1));
    E(7, syr2k(hermitian, 'U', 'N', 3, 1, 2, 3, 3));
    E(7, syr2k(hermitian, 'U', trans, 1, 3, 2, 3, 1));
    E(7, syr2k(hermitian, 'U', 'N', 0, 0, 0, 1, 1));
    E(9, syr2k(hermitian, 'U', 'N', 3, 1, 3, 2, 3));
    E(9, syr2k(hermitian, 'U', trans, 1, 3, 3, 2, 1));
    E(9, syr2k(hermitian, 'U', 'N', 0, 0, 1, 0, 1));
    E(12, syr2k(hermitian, 'U', 'N', 3, 1, 3, 3, 2));
    E(12, syr2k(hermitian, 'U', 'N', 0, 0, 1, 1, 0));
  }
}

TYPED_TEST(ValidationTest, Gemmtr) {
  using T = TypeParam;
  auto gemmtr = [&](char uplo, char ta, char tb, int n, int k, int lda, int ldb,
                    int ldc) {
    return [=] {
      HWYBLAS_CALL(T, gemmtr, &uplo, &ta, &tb, &n, &k, P(&this->alpha),
                   P(this->a.data()), &lda, P(this->b.data()), &ldb,
                   P(&this->beta), P(this->c.data()), &ldc);
    };
  };
  const auto E = [](int position, auto&& call) {
    TestFixture::Expect("GEMMTR", position, call);
  };
  E(0, gemmtr('U', 'N', 'N', 0, 0, 1, 1, 1));
  E(0, gemmtr('L', 'N', 'N', 1, 1, 1, 1, 1));
  E(0, gemmtr('L', 'T', 'T', 3, 2, 2, 3, 3));
  E(1, gemmtr('X', 'N', 'N', 1, 1, 1, 1, 1));
  E(2, gemmtr('U', 'X', 'N', 1, 1, 1, 1, 1));
  E(3, gemmtr('U', 'N', 'X', 1, 1, 1, 1, 1));
  E(4, gemmtr('U', 'N', 'N', -1, 1, 1, 1, 1));
  E(5, gemmtr('U', 'N', 'N', 1, -1, 1, 1, 1));
  E(8, gemmtr('U', 'N', 'N', 3, 1, 2, 1, 3));
  E(8, gemmtr('U', 'T', 'N', 1, 3, 2, 3, 1));
  E(8, gemmtr('U', 'N', 'N', 0, 0, 0, 1, 1));
  E(10, gemmtr('U', 'N', 'N', 1, 3, 1, 2, 1));
  E(10, gemmtr('U', 'N', 'T', 3, 1, 3, 2, 3));
  E(10, gemmtr('U', 'N', 'N', 0, 0, 1, 0, 1));
  E(13, gemmtr('U', 'N', 'N', 3, 1, 3, 1, 2));
  E(13, gemmtr('U', 'N', 'N', 0, 0, 1, 1, 0));
}

TYPED_TEST(ValidationTest, TrmmTrsm) {
  using T = TypeParam;
  auto tr = [&](bool solve, char side, char uplo, char trans, char diag, int m,
                int n, int lda, int ldb) {
    return [=] {
      if (solve) {
        HWYBLAS_CALL(T, trsm, &side, &uplo, &trans, &diag, &m, &n,
                     P(&this->alpha), P(this->a.data()), &lda,
                     P(this->b.data()), &ldb);
      } else {
        HWYBLAS_CALL(T, trmm, &side, &uplo, &trans, &diag, &m, &n,
                     P(&this->alpha), P(this->a.data()), &lda,
                     P(this->b.data()), &ldb);
      }
    };
  };
  for (const bool solve : {false, true}) {
    const char* name = solve ? "TRSM" : "TRMM";
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, tr(solve, 'L', 'U', 'N', 'N', 0, 0, 1, 1));
    E(0, tr(solve, 'L', 'L', 'N', 'U', 1, 1, 1, 1));
    E(0, tr(solve, 'R', 'L', 'T', 'N', 2, 3, 3, 2));
    E(1, tr(solve, 'X', 'U', 'N', 'N', 1, 1, 1, 1));
    E(2, tr(solve, 'L', 'X', 'N', 'N', 1, 1, 1, 1));
    E(3, tr(solve, 'L', 'U', 'X', 'N', 1, 1, 1, 1));
    E(4, tr(solve, 'L', 'U', 'N', 'X', 1, 1, 1, 1));
    E(5, tr(solve, 'L', 'U', 'N', 'N', -1, 1, 1, 1));
    E(6, tr(solve, 'L', 'U', 'N', 'N', 1, -1, 1, 1));
    // A is m x m on the left and n x n on the right.
    E(9, tr(solve, 'L', 'U', 'N', 'N', 3, 1, 2, 3));
    E(9, tr(solve, 'R', 'U', 'N', 'N', 1, 3, 2, 1));
    E(9, tr(solve, 'L', 'U', 'N', 'N', 0, 0, 0, 1));
    E(11, tr(solve, 'L', 'U', 'N', 'N', 3, 1, 3, 2));
    E(11, tr(solve, 'L', 'U', 'N', 'N', 0, 0, 1, 0));
  }
}

// ---------------------------------------------------------------------------
// Level 2

TYPED_TEST(ValidationTest, GemvGer) {
  using T = TypeParam;
  auto gemv = [&](char trans, int m, int n, int lda, int incx, int incy) {
    return [=] {
      HWYBLAS_CALL(T, gemv, &trans, &m, &n, P(&this->alpha), P(this->a.data()),
                   &lda, P(this->b.data()), &incx, P(&this->beta),
                   P(this->c.data()), &incy);
    };
  };
  const auto G = [](int position, auto&& call) {
    TestFixture::Expect("GEMV", position, call);
  };
  G(0, gemv('N', 0, 0, 1, 1, 1));
  G(0, gemv('N', 1, 1, 1, -1, -1));
  G(0, gemv('T', 3, 2, 3, 1, 1));
  G(1, gemv('X', 1, 1, 1, 1, 1));
  G(2, gemv('N', -1, 1, 1, 1, 1));
  G(3, gemv('N', 1, -1, 1, 1, 1));
  G(6, gemv('N', 3, 1, 2, 1, 1));
  G(6, gemv('T', 3, 1, 2, 1, 1));
  G(6, gemv('N', 0, 0, 0, 1, 1));
  G(8, gemv('N', 1, 1, 1, 0, 1));
  G(11, gemv('N', 1, 1, 1, 1, 0));

  auto ger = [&](int m, int n, int incx, int incy, int lda) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, gerc, &m, &n, P(&this->alpha),
                             P(this->b.data()), &incx, P(this->c.data()), &incy,
                             P(this->a.data()), &lda);
      } else {
        HWYBLAS_CALL_REAL(T, ger, &m, &n, P(&this->alpha), P(this->b.data()),
                          &incx, P(this->c.data()), &incy, P(this->a.data()),
                          &lda);
      }
    };
  };
  const char* name = IsComplex<T>::value ? "GERC" : "GER";
  const auto R = [&](int position, auto&& call) {
    TestFixture::Expect(name, position, call);
  };
  R(0, ger(0, 0, 1, 1, 1));
  R(0, ger(1, 1, -1, -1, 1));
  R(1, ger(-1, 1, 1, 1, 1));
  R(2, ger(1, -1, 1, 1, 1));
  R(5, ger(1, 1, 0, 1, 1));
  R(7, ger(1, 1, 1, 0, 1));
  R(9, ger(3, 1, 1, 1, 2));
  R(9, ger(0, 0, 1, 1, 0));
}

TYPED_TEST(ValidationTest, SymvSyrSyr2) {
  using T = TypeParam;
  auto symv = [&](char uplo, int n, int lda, int incx, int incy) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hemv, &uplo, &n, P(&this->alpha),
                             P(this->a.data()), &lda, P(this->b.data()), &incx,
                             P(&this->beta), P(this->c.data()), &incy);
      } else {
        HWYBLAS_CALL_REAL(T, symv, &uplo, &n, P(&this->alpha),
                          P(this->a.data()), &lda, P(this->b.data()), &incx,
                          P(&this->beta), P(this->c.data()), &incy);
      }
    };
  };
  const char* sv = IsComplex<T>::value ? "HEMV" : "SYMV";
  const auto V = [&](int position, auto&& call) {
    TestFixture::Expect(sv, position, call);
  };
  V(0, symv('U', 0, 1, 1, 1));
  V(0, symv('L', 1, 1, -1, -1));
  V(1, symv('X', 1, 1, 1, 1));
  V(2, symv('U', -1, 1, 1, 1));
  V(5, symv('U', 3, 2, 1, 1));
  V(5, symv('U', 0, 0, 1, 1));
  V(7, symv('U', 1, 1, 0, 1));
  V(10, symv('U', 1, 1, 1, 0));

  auto syr = [&](char uplo, int n, int incx, int lda) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, her, &uplo, &n, &this->ralpha,
                             P(this->b.data()), &incx, P(this->a.data()), &lda);
      } else {
        HWYBLAS_CALL_REAL(T, syr, &uplo, &n, P(&this->alpha), P(this->b.data()),
                          &incx, P(this->a.data()), &lda);
      }
    };
  };
  const char* sr = IsComplex<T>::value ? "HER" : "SYR";
  const auto S = [&](int position, auto&& call) {
    TestFixture::Expect(sr, position, call);
  };
  S(0, syr('U', 0, 1, 1));
  S(0, syr('L', 1, -1, 1));
  S(1, syr('X', 1, 1, 1));
  S(2, syr('U', -1, 1, 1));
  S(5, syr('U', 1, 0, 1));
  S(7, syr('U', 3, 1, 2));
  S(7, syr('U', 0, 1, 0));

  auto syr2 = [&](char uplo, int n, int incx, int incy, int lda) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, her2, &uplo, &n, P(&this->alpha),
                             P(this->b.data()), &incx, P(this->c.data()), &incy,
                             P(this->a.data()), &lda);
      } else {
        HWYBLAS_CALL_REAL(T, syr2, &uplo, &n, P(&this->alpha),
                          P(this->b.data()), &incx, P(this->c.data()), &incy,
                          P(this->a.data()), &lda);
      }
    };
  };
  const char* s2 = IsComplex<T>::value ? "HER2" : "SYR2";
  const auto S2 = [&](int position, auto&& call) {
    TestFixture::Expect(s2, position, call);
  };
  S2(0, syr2('U', 0, 1, 1, 1));
  S2(0, syr2('L', 1, -1, -1, 1));
  S2(1, syr2('X', 1, 1, 1, 1));
  S2(2, syr2('U', -1, 1, 1, 1));
  S2(5, syr2('U', 1, 0, 1, 1));
  S2(7, syr2('U', 1, 1, 0, 1));
  S2(9, syr2('U', 3, 1, 1, 2));
  S2(9, syr2('U', 0, 1, 1, 0));
}

TYPED_TEST(ValidationTest, TrmvTrsv) {
  using T = TypeParam;
  auto tr = [&](bool solve, char uplo, char trans, char diag, int n, int lda,
                int incx) {
    return [=] {
      if (solve) {
        HWYBLAS_CALL(T, trsv, &uplo, &trans, &diag, &n, P(this->a.data()), &lda,
                     P(this->b.data()), &incx);
      } else {
        HWYBLAS_CALL(T, trmv, &uplo, &trans, &diag, &n, P(this->a.data()), &lda,
                     P(this->b.data()), &incx);
      }
    };
  };
  for (const bool solve : {false, true}) {
    const char* name = solve ? "TRSV" : "TRMV";
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, tr(solve, 'U', 'N', 'N', 0, 1, 1));
    E(0, tr(solve, 'L', 'T', 'U', 1, 1, -1));
    E(1, tr(solve, 'X', 'N', 'N', 1, 1, 1));
    E(2, tr(solve, 'U', 'X', 'N', 1, 1, 1));
    E(3, tr(solve, 'U', 'N', 'X', 1, 1, 1));
    E(4, tr(solve, 'U', 'N', 'N', -1, 1, 1));
    E(6, tr(solve, 'U', 'N', 'N', 3, 2, 1));
    E(6, tr(solve, 'U', 'N', 'N', 0, 0, 1));
    E(8, tr(solve, 'U', 'N', 'N', 1, 1, 0));
  }
}

// ---------------------------------------------------------------------------
// Banded and packed

TYPED_TEST(ValidationTest, Gbmv) {
  using T = TypeParam;
  auto gbmv = [&](char trans, int m, int n, int kl, int ku, int lda, int incx,
                  int incy) {
    return [=] {
      HWYBLAS_CALL(T, gbmv, &trans, &m, &n, &kl, &ku, P(&this->alpha),
                   P(this->a.data()), &lda, P(this->b.data()), &incx,
                   P(&this->beta), P(this->c.data()), &incy);
    };
  };
  const auto E = [](int position, auto&& call) {
    TestFixture::Expect("GBMV", position, call);
  };
  E(0, gbmv('N', 0, 0, 0, 0, 1, 1, 1));
  E(0, gbmv('N', 2, 2, 0, 0, 1, 1, 1));
  E(0, gbmv('T', 3, 3, 1, 2, 4, -1, -1));
  E(1, gbmv('X', 1, 1, 0, 0, 1, 1, 1));
  E(2, gbmv('N', -1, 1, 0, 0, 1, 1, 1));
  E(3, gbmv('N', 1, -1, 0, 0, 1, 1, 1));
  E(4, gbmv('N', 1, 1, -1, 0, 1, 1, 1));
  E(5, gbmv('N', 1, 1, 0, -1, 1, 1, 1));
  E(8, gbmv('N', 3, 3, 1, 2, 3, 1, 1));
  E(8, gbmv('N', 1, 1, 0, 0, 0, 1, 1));
  E(10, gbmv('N', 1, 1, 0, 0, 1, 0, 1));
  E(13, gbmv('N', 1, 1, 0, 0, 1, 1, 0));
}

TYPED_TEST(ValidationTest, SbmvTbmvTbsv) {
  using T = TypeParam;
  auto sbmv = [&](char uplo, int n, int k, int lda, int incx, int incy) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hbmv, &uplo, &n, &k, P(&this->alpha),
                             P(this->a.data()), &lda, P(this->b.data()), &incx,
                             P(&this->beta), P(this->c.data()), &incy);
      } else {
        HWYBLAS_CALL_REAL(T, sbmv, &uplo, &n, &k, P(&this->alpha),
                          P(this->a.data()), &lda, P(this->b.data()), &incx,
                          P(&this->beta), P(this->c.data()), &incy);
      }
    };
  };
  const char* sb = IsComplex<T>::value ? "HBMV" : "SBMV";
  const auto S = [&](int position, auto&& call) {
    TestFixture::Expect(sb, position, call);
  };
  S(0, sbmv('U', 0, 0, 1, 1, 1));
  S(0, sbmv('L', 3, 1, 2, -1, -1));
  S(1, sbmv('X', 1, 0, 1, 1, 1));
  S(2, sbmv('U', -1, 0, 1, 1, 1));
  S(3, sbmv('U', 1, -1, 1, 1, 1));
  S(6, sbmv('U', 3, 1, 1, 1, 1));
  S(6, sbmv('U', 0, 0, 0, 1, 1));
  S(8, sbmv('U', 1, 0, 1, 0, 1));
  S(11, sbmv('U', 1, 0, 1, 1, 0));

  auto tbmv = [&](bool solve, char uplo, char trans, char diag, int n, int k,
                  int lda, int incx) {
    return [=] {
      if (solve) {
        HWYBLAS_CALL(T, tbsv, &uplo, &trans, &diag, &n, &k, P(this->a.data()),
                     &lda, P(this->b.data()), &incx);
      } else {
        HWYBLAS_CALL(T, tbmv, &uplo, &trans, &diag, &n, &k, P(this->a.data()),
                     &lda, P(this->b.data()), &incx);
      }
    };
  };
  for (const bool solve : {false, true}) {
    const char* name = solve ? "TBSV" : "TBMV";
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, tbmv(solve, 'U', 'N', 'N', 0, 0, 1, 1));
    E(0, tbmv(solve, 'L', 'T', 'U', 3, 1, 2, -1));
    E(1, tbmv(solve, 'X', 'N', 'N', 1, 0, 1, 1));
    E(2, tbmv(solve, 'U', 'X', 'N', 1, 0, 1, 1));
    E(3, tbmv(solve, 'U', 'N', 'X', 1, 0, 1, 1));
    E(4, tbmv(solve, 'U', 'N', 'N', -1, 0, 1, 1));
    E(5, tbmv(solve, 'U', 'N', 'N', 1, -1, 1, 1));
    E(7, tbmv(solve, 'U', 'N', 'N', 3, 1, 1, 1));
    E(7, tbmv(solve, 'U', 'N', 'N', 0, 0, 0, 1));
    E(9, tbmv(solve, 'U', 'N', 'N', 1, 0, 1, 0));
  }
}

TYPED_TEST(ValidationTest, PackedRoutines) {
  using T = TypeParam;
  auto spmv = [&](char uplo, int n, int incx, int incy) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hpmv, &uplo, &n, P(&this->alpha),
                             P(this->a.data()), P(this->b.data()), &incx,
                             P(&this->beta), P(this->c.data()), &incy);
      } else {
        HWYBLAS_CALL_REAL(T, spmv, &uplo, &n, P(&this->alpha),
                          P(this->a.data()), P(this->b.data()), &incx,
                          P(&this->beta), P(this->c.data()), &incy);
      }
    };
  };
  const char* sp = IsComplex<T>::value ? "HPMV" : "SPMV";
  const auto S = [&](int position, auto&& call) {
    TestFixture::Expect(sp, position, call);
  };
  S(0, spmv('U', 0, 1, 1));
  S(0, spmv('L', 3, -1, -1));
  S(1, spmv('X', 1, 1, 1));
  S(2, spmv('U', -1, 1, 1));
  S(6, spmv('U', 1, 0, 1));
  S(9, spmv('U', 1, 1, 0));

  auto tpmv = [&](bool solve, char uplo, char trans, char diag, int n,
                  int incx) {
    return [=] {
      if (solve) {
        HWYBLAS_CALL(T, tpsv, &uplo, &trans, &diag, &n, P(this->a.data()),
                     P(this->b.data()), &incx);
      } else {
        HWYBLAS_CALL(T, tpmv, &uplo, &trans, &diag, &n, P(this->a.data()),
                     P(this->b.data()), &incx);
      }
    };
  };
  for (const bool solve : {false, true}) {
    const char* name = solve ? "TPSV" : "TPMV";
    const auto E = [&](int position, auto&& call) {
      TestFixture::Expect(name, position, call);
    };
    E(0, tpmv(solve, 'U', 'N', 'N', 0, 1));
    E(0, tpmv(solve, 'L', 'T', 'U', 3, -1));
    E(1, tpmv(solve, 'X', 'N', 'N', 1, 1));
    E(2, tpmv(solve, 'U', 'X', 'N', 1, 1));
    E(3, tpmv(solve, 'U', 'N', 'X', 1, 1));
    E(4, tpmv(solve, 'U', 'N', 'N', -1, 1));
    E(7, tpmv(solve, 'U', 'N', 'N', 1, 0));
  }

  auto spr = [&](char uplo, int n, int incx) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hpr, &uplo, &n, &this->ralpha,
                             P(this->b.data()), &incx, P(this->a.data()));
      } else {
        HWYBLAS_CALL_REAL(T, spr, &uplo, &n, P(&this->alpha), P(this->b.data()),
                          &incx, P(this->a.data()));
      }
    };
  };
  const char* sr = IsComplex<T>::value ? "HPR" : "SPR";
  const auto R = [&](int position, auto&& call) {
    TestFixture::Expect(sr, position, call);
  };
  R(0, spr('U', 0, 1));
  R(0, spr('L', 3, -1));
  R(1, spr('X', 1, 1));
  R(2, spr('U', -1, 1));
  R(5, spr('U', 1, 0));

  auto spr2 = [&](char uplo, int n, int incx, int incy) {
    return [=] {
      if constexpr (IsComplex<T>::value) {
        HWYBLAS_CALL_COMPLEX(T, hpr2, &uplo, &n, P(&this->alpha),
                             P(this->b.data()), &incx, P(this->c.data()), &incy,
                             P(this->a.data()));
      } else {
        HWYBLAS_CALL_REAL(T, spr2, &uplo, &n, P(&this->alpha),
                          P(this->b.data()), &incx, P(this->c.data()), &incy,
                          P(this->a.data()));
      }
    };
  };
  const char* s2 = IsComplex<T>::value ? "HPR2" : "SPR2";
  const auto R2 = [&](int position, auto&& call) {
    TestFixture::Expect(s2, position, call);
  };
  R2(0, spr2('U', 0, 1, 1));
  R2(0, spr2('L', 3, -1, -1));
  R2(1, spr2('X', 1, 1, 1));
  R2(2, spr2('U', -1, 1, 1));
  R2(5, spr2('U', 1, 0, 1));
  R2(7, spr2('U', 1, 1, 0));
}

// ---------------------------------------------------------------------------
// ?omatcopy and ?imatcopy

TYPED_TEST(ValidationTest, Matcopy) {
  using T = TypeParam;
  auto omatcopy = [&](char order, char trans, int rows, int cols, int lda,
                      int ldb) {
    return [=] {
      HWYBLAS_CALL(T, omatcopy, &order, &trans, &rows, &cols, P(&this->alpha),
                   P(this->a.data()), &lda, P(this->b.data()), &ldb);
    };
  };
  const auto O = [](int position, auto&& call) {
    TestFixture::Expect("OMATCOPY", position, call);
  };
  O(0, omatcopy('C', 'N', 0, 0, 1, 1));
  O(0, omatcopy('R', 'N', 0, 0, 1, 1));
  O(0, omatcopy('C', 'T', 3, 2, 3, 2));
  O(0, omatcopy('R', 'C', 3, 2, 2, 3));
  O(1, omatcopy('X', 'N', 1, 1, 1, 1));
  O(2, omatcopy('C', 'X', 1, 1, 1, 1));
  O(3, omatcopy('C', 'N', -1, 1, 1, 1));
  O(4, omatcopy('C', 'N', 1, -1, 1, 1));
  // A is rows x cols; column-major: lda >= rows, row-major: lda >= cols.
  O(7, omatcopy('C', 'N', 3, 1, 2, 3));
  O(7, omatcopy('R', 'N', 1, 3, 2, 1));
  O(7, omatcopy('C', 'N', 0, 0, 0, 1));
  O(7, omatcopy('R', 'N', 0, 0, 0, 1));
  // B is op(A): ldb >= rows for N and R, cols for T and C (column-major).
  O(9, omatcopy('C', 'N', 3, 1, 3, 2));
  O(9, omatcopy('C', 'T', 1, 3, 1, 2));
  O(9, omatcopy('R', 'N', 1, 3, 3, 2));
  O(9, omatcopy('R', 'T', 3, 1, 1, 2));
  O(9, omatcopy('C', 'N', 0, 0, 1, 0));

  auto imatcopy = [&](char order, char trans, int rows, int cols, int lda,
                      int ldb) {
    return [=] {
      HWYBLAS_CALL(T, imatcopy, &order, &trans, &rows, &cols, P(&this->alpha),
                   P(this->a.data()), &lda, &ldb);
    };
  };
  const auto I = [](int position, auto&& call) {
    TestFixture::Expect("IMATCOPY", position, call);
  };
  I(0, imatcopy('C', 'N', 0, 0, 1, 1));
  I(0, imatcopy('R', 'T', 2, 3, 3, 2));
  I(1, imatcopy('X', 'N', 1, 1, 1, 1));
  I(2, imatcopy('C', 'X', 1, 1, 1, 1));
  I(3, imatcopy('C', 'N', -1, 1, 1, 1));
  I(4, imatcopy('C', 'N', 1, -1, 1, 1));
  I(7, imatcopy('C', 'N', 3, 1, 2, 3));
  I(7, imatcopy('R', 'N', 0, 0, 0, 1));
  I(8, imatcopy('C', 'N', 3, 1, 3, 2));
  I(8, imatcopy('C', 'T', 1, 3, 1, 2));
  I(8, imatcopy('C', 'N', 0, 0, 1, 0));
}

}  // namespace
}  // namespace hwyblas_test
