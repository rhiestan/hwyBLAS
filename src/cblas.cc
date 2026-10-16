// CBLAS interface: C functions with value arguments and a choice of row- or
// column-major storage, implemented on top of the Fortran-style routines.
//
// Column-major calls are forwarded unchanged. A row-major matrix is the
// column-major storage of its transpose, so row-major calls are mapped with
// the usual identities, e.g. C = A * B  <=>  C^T = B^T * A^T for GEMM, or a
// flipped uplo and transposition for triangular matrices. Where the mapped
// operation would need a conjugated but not transposed matrix (row-major
// ConjTrans in GEMV and the triangular routines, and the row-major
// Hermitian Level 2 routines), vectors are conjugated around the call
// instead, as in the reference CBLAS:  conj(B) * x = conj(B * conj(x)).
//
// Invalid arguments are reported through cblas_xerbla with the position of
// the CBLAS argument (counting the layout as argument 1). The Fortran
// routines detect them; each wrapper tells ReportError which CBLAS argument
// each Fortran argument came from.
#include "hwyblas/cblas.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <complex>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include "hwyblas/hwyblas.h"
#include "src/common.h"

namespace hwyblas {
namespace {

using CblasInt = hwyblas_int;

// ---------------------------------------------------------------------------
// Argument conversion

// Array arguments as the pointer types of the Fortran routines (complex
// numbers as interleaved reals).
template <typename T>
const RealT<T>* F(const void* p) {
  return static_cast<const RealT<T>*>(p);
}
template <typename T>
RealT<T>* F(void* p) {
  return static_cast<RealT<T>*>(p);
}
// Scalars passed to the Fortran routines by pointer.
template <typename T>
const RealT<T>* F(const T& v) {
  return reinterpret_cast<const RealT<T>*>(&v);
}

char TransChar(CBLAS_TRANSPOSE t) {
  switch (t) {
    case CblasNoTrans:
      return 'N';
    case CblasTrans:
      return 'T';
    case CblasConjTrans:
      return 'C';
    default:
      return '?';  // rejected by the Fortran routine
  }
}

char UploChar(CBLAS_UPLO u) {
  return u == CblasUpper ? 'U' : u == CblasLower ? 'L' : '?';
}
char DiagChar(CBLAS_DIAG d) {
  return d == CblasNonUnit ? 'N' : d == CblasUnit ? 'U' : '?';
}
char SideChar(CBLAS_SIDE s) {
  return s == CblasLeft ? 'L' : s == CblasRight ? 'R' : '?';
}

// ORDER and TRANS of ?omatcopy_/?imatcopy_ ('?' if invalid).
char MatcopyOrder(CBLAS_LAYOUT layout) {
  return layout == CblasColMajor ? 'C' : layout == CblasRowMajor ? 'R' : '?';
}
char MatcopyTrans(CBLAS_TRANSPOSE t) {
  return t == CblasConjNoTrans ? 'R' : TransChar(t);
}

// The opposite triangle or side (for row-major storage).
char FlipUplo(char c) { return c == 'U' ? 'L' : c == 'L' ? 'U' : c; }
char FlipSide(char c) { return c == 'L' ? 'R' : c == 'R' ? 'L' : c; }

// Transposition for the transposed (row-major) matrix: N <-> T, and C -> N
// with `*conj` set (the caller conjugates instead). For real types, C is T.
char FlipTrans(char c, bool* conj) {
  *conj = false;
  switch (c) {
    case 'N':
      return 'T';
    case 'T':
      return 'N';
    case 'C':
      *conj = true;
      return 'N';
    default:
      return c;
  }
}

// ---------------------------------------------------------------------------
// Error reporting

// Maps the positions of the Fortran arguments to those of the CBLAS
// arguments for the duration of a call.
class CblasScope {
 public:
  CblasScope(const char* name, std::initializer_list<int> positions) {
    context_.name = name;
    size_t i = 0;
    for (int p : positions) positions_[i++] = p;
    context_.positions = positions_;
    SetCblasContext(&context_);
  }
  ~CblasScope() { SetCblasContext(nullptr); }
  CblasScope(const CblasScope&) = delete;
  CblasScope& operator=(const CblasScope&) = delete;

 private:
  CblasContext context_;
  int positions_[16];
};

// Returns false and reports an invalid layout (argument 1).
bool ValidLayout(const char* name, CBLAS_LAYOUT layout) {
  if (layout == CblasRowMajor || layout == CblasColMajor) return true;
  cblas_xerbla(1, name, "Illegal layout setting, %d\n",
               static_cast<int>(layout));
  return false;
}

// ---------------------------------------------------------------------------
// Conjugation of strided vectors

template <typename T>
void ConjugateInPlace(CblasInt n, T* x, CblasInt inc) {
  if constexpr (IsComplex<T>::value) {
    if (n <= 0 || inc == 0) return;
    T* start = VectorStart(x, n, inc);
    for (CblasInt i = 0; i < n; ++i) {
      T& v = start[static_cast<ptrdiff_t>(i) * inc];
      v = std::conj(v);
    }
  }
}

// Contiguous conjugated copy of x (empty if n <= 0 or inc == 0).
template <typename T>
std::vector<T> ConjugatedCopy(CblasInt n, const T* x, CblasInt inc) {
  std::vector<T> c;
  if (n <= 0 || inc == 0) return c;
  const T* start = VectorStart(x, n, inc);
  c.resize(n);
  for (CblasInt i = 0; i < n; ++i) {
    c[i] = Conj(start[static_cast<ptrdiff_t>(i) * inc]);
  }
  return c;
}

// Runs `call(x, incx)` with x replaced by its conjugate if `conj` (and the
// vector is valid; otherwise the original arguments go to the Fortran
// routine, which reports the error).
template <typename T, class Call>
void WithConjugatedInput(bool conj, CblasInt n, const void* x, CblasInt incx,
                         const Call& call) {
  if (conj) {
    const std::vector<T> c = ConjugatedCopy(n, static_cast<const T*>(x), incx);
    if (!c.empty()) return call(static_cast<const void*>(c.data()), 1);
  }
  call(x, incx);
}

// ---------------------------------------------------------------------------
// Fortran routines per scalar type.

template <typename T>
struct Fortran;

#define HWYBLAS_COMMON_ROUTINES(p)            \
  static constexpr auto gemv = &p##gemv_;     \
  static constexpr auto gbmv = &p##gbmv_;     \
  static constexpr auto trmv = &p##trmv_;     \
  static constexpr auto trsv = &p##trsv_;     \
  static constexpr auto tbmv = &p##tbmv_;     \
  static constexpr auto tbsv = &p##tbsv_;     \
  static constexpr auto tpmv = &p##tpmv_;     \
  static constexpr auto tpsv = &p##tpsv_;     \
  static constexpr auto gemm = &p##gemm_;     \
  static constexpr auto gemmtr = &p##gemmtr_; \
  static constexpr auto symm = &p##symm_;     \
  static constexpr auto syrk = &p##syrk_;     \
  static constexpr auto syr2k = &p##syr2k_;   \
  static constexpr auto trmm = &p##trmm_;     \
  static constexpr auto trsm = &p##trsm_

// Real types: the symmetric Level 2 routines.
#define HWYBLAS_REAL_ROUTINES(p)          \
  HWYBLAS_COMMON_ROUTINES(p);             \
  static constexpr auto symv = &p##symv_; \
  static constexpr auto sbmv = &p##sbmv_; \
  static constexpr auto spmv = &p##spmv_; \
  static constexpr auto ger = &p##ger_;   \
  static constexpr auto syr = &p##syr_;   \
  static constexpr auto spr = &p##spr_;   \
  static constexpr auto syr2 = &p##syr2_; \
  static constexpr auto spr2 = &p##spr2_

// Complex types: the Hermitian routines. They are listed under the names of
// the corresponding real routines, so that the wrappers below cover both.
#define HWYBLAS_COMPLEX_ROUTINES(p)       \
  HWYBLAS_COMMON_ROUTINES(p);             \
  static constexpr auto symv = &p##hemv_; \
  static constexpr auto sbmv = &p##hbmv_; \
  static constexpr auto spmv = &p##hpmv_; \
  static constexpr auto ger = &p##geru_;  \
  static constexpr auto gerc = &p##gerc_; \
  static constexpr auto syr = &p##her_;   \
  static constexpr auto spr = &p##hpr_;   \
  static constexpr auto syr2 = &p##her2_; \
  static constexpr auto spr2 = &p##hpr2_; \
  static constexpr auto hemm = &p##hemm_; \
  static constexpr auto herk = &p##herk_; \
  static constexpr auto her2k = &p##her2k_

template <>
struct Fortran<float> {
  HWYBLAS_REAL_ROUTINES(s);
};
template <>
struct Fortran<double> {
  HWYBLAS_REAL_ROUTINES(d);
};
template <>
struct Fortran<c32> {
  HWYBLAS_COMPLEX_ROUTINES(c);
};
template <>
struct Fortran<c64> {
  HWYBLAS_COMPLEX_ROUTINES(z);
};

#undef HWYBLAS_COMMON_ROUTINES
#undef HWYBLAS_REAL_ROUTINES
#undef HWYBLAS_COMPLEX_ROUTINES

// ---------------------------------------------------------------------------
// Level 2

// GEMV (band = false) and GBMV: y := alpha * op(A) * x + beta * y.
// Positions: layout 1, trans 2, m 3, n 4, [kl 5, ku 6,] alpha, a, lda, x,
// incx, beta, y, incy.
template <typename T>
void Gemv(const char* name, bool band, CBLAS_LAYOUT layout,
          CBLAS_TRANSPOSE trans, CblasInt m, CblasInt n, CblasInt kl,
          CblasInt ku, T alpha, const void* a, CblasInt lda, const void* x,
          CblasInt incx, T beta, void* y, CblasInt incy) {
  if (!ValidLayout(name, layout)) return;
  char tc = TransChar(trans);
  bool conj = false;
  const bool row = (layout == CblasRowMajor);
  if (row) {
    // op(A) for row-major A is op'(A^T) for column-major A^T (n x m).
    tc = FlipTrans(tc, &conj);
    conj = conj && IsComplex<T>::value;
    std::swap(m, n);
    std::swap(kl, ku);
  }
  CblasScope scope(
      name,
      band && row ? std::initializer_list<int>{2, 4, 3, 6, 5, 7, 8, 9, 10, 11,
                                               12, 13, 14}
      : band ? std::initializer_list<int>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                          13, 14}
      : row  ? std::initializer_list<int>{2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12}
             : std::initializer_list<int>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // With conj: y := conj(conj(alpha) * A' * conj(x) + conj(beta) * conj(y)).
  const CblasInt len_x = (tc == 'N') ? n : m;
  const CblasInt len_y = (tc == 'N') ? m : n;
  if (conj) {
    alpha = Conj(alpha);
    beta = Conj(beta);
    ConjugateInPlace(len_y, static_cast<T*>(y), incy);
  }
  WithConjugatedInput<T>(
      conj, len_x, x, incx, [&](const void* xv, CblasInt ix) {
        if (band) {
          Fortran<T>::gbmv(&tc, &m, &n, &kl, &ku, F<T>(alpha), F<T>(a), &lda,
                           F<T>(xv), &ix, F<T>(beta), F<T>(y), &incy);
        } else {
          Fortran<T>::gemv(&tc, &m, &n, F<T>(alpha), F<T>(a), &lda, F<T>(xv),
                           &ix, F<T>(beta), F<T>(y), &incy);
        }
      });
  if (conj) ConjugateInPlace(len_y, static_cast<T*>(y), incy);
}

// The triangular Level 2 routines: x := op(A) * x or solves op(A) x = b.
enum class TriangularKind { kTrmv, kTrsv, kTbmv, kTbsv, kTpmv, kTpsv };

// Positions: layout 1, uplo 2, trans 3, diag 4, n 5, [k 6,] a, [lda,] x, incx.
template <typename T>
void Triangular(const char* name, TriangularKind kind, CBLAS_LAYOUT layout,
                CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                CblasInt n, CblasInt k, const void* a, CblasInt lda, void* x,
                CblasInt incx) {
  if (!ValidLayout(name, layout)) return;
  char uc = UploChar(uplo), tc = TransChar(trans);
  const char dc = DiagChar(diag);
  bool conj = false;
  if (layout == CblasRowMajor) {
    // Row-major A is column-major A^T, whose other triangle is stored.
    uc = FlipUplo(uc);
    tc = FlipTrans(tc, &conj);
    conj = conj && IsComplex<T>::value;
  }
  CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10});
  // With conj: conj(A') * x = conj(A' * conj(x)), and the same for solves.
  if (conj) ConjugateInPlace(n, static_cast<T*>(x), incx);
  RealT<T>* xr = F<T>(x);
  switch (kind) {
    case TriangularKind::kTrmv:
      Fortran<T>::trmv(&uc, &tc, &dc, &n, F<T>(a), &lda, xr, &incx);
      break;
    case TriangularKind::kTrsv:
      Fortran<T>::trsv(&uc, &tc, &dc, &n, F<T>(a), &lda, xr, &incx);
      break;
    case TriangularKind::kTbmv:
      Fortran<T>::tbmv(&uc, &tc, &dc, &n, &k, F<T>(a), &lda, xr, &incx);
      break;
    case TriangularKind::kTbsv:
      Fortran<T>::tbsv(&uc, &tc, &dc, &n, &k, F<T>(a), &lda, xr, &incx);
      break;
    case TriangularKind::kTpmv:
      Fortran<T>::tpmv(&uc, &tc, &dc, &n, F<T>(a), xr, &incx);
      break;
    case TriangularKind::kTpsv:
      Fortran<T>::tpsv(&uc, &tc, &dc, &n, F<T>(a), xr, &incx);
      break;
  }
  if (conj) ConjugateInPlace(n, static_cast<T*>(x), incx);
}

// SYMV, SBMV, SPMV (HEMV, HBMV, HPMV for complex types):
// y := alpha * A * x + beta * y.
enum class SymmetricKind { kFull, kBand, kPacked };

// Positions: layout 1, uplo 2, n 3, [k 4,] alpha, a, [lda,] x, incx, beta, y,
// incy.
template <typename T>
void Symmetric(const char* name, SymmetricKind kind, CBLAS_LAYOUT layout,
               CBLAS_UPLO uplo, CblasInt n, CblasInt k, T alpha, const void* a,
               CblasInt lda, const void* x, CblasInt incx, T beta, void* y,
               CblasInt incy) {
  if (!ValidLayout(name, layout)) return;
  char uc = UploChar(uplo);
  // Row-major A is column-major A^T: the other triangle of the same matrix
  // if symmetric, of conj(A) if Hermitian.
  const bool row = (layout == CblasRowMajor);
  const bool conj = row && IsComplex<T>::value;
  if (row) uc = FlipUplo(uc);
  CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  if (conj) {
    alpha = Conj(alpha);
    beta = Conj(beta);
    ConjugateInPlace(n, static_cast<T*>(y), incy);
  }
  WithConjugatedInput<T>(conj, n, x, incx, [&](const void* xv, CblasInt ix) {
    switch (kind) {
      case SymmetricKind::kFull:
        Fortran<T>::symv(&uc, &n, F<T>(alpha), F<T>(a), &lda, F<T>(xv), &ix,
                         F<T>(beta), F<T>(y), &incy);
        break;
      case SymmetricKind::kBand:
        Fortran<T>::sbmv(&uc, &n, &k, F<T>(alpha), F<T>(a), &lda, F<T>(xv), &ix,
                         F<T>(beta), F<T>(y), &incy);
        break;
      case SymmetricKind::kPacked:
        Fortran<T>::spmv(&uc, &n, F<T>(alpha), F<T>(a), F<T>(xv), &ix,
                         F<T>(beta), F<T>(y), &incy);
        break;
    }
  });
  if (conj) ConjugateInPlace(n, static_cast<T*>(y), incy);
}

// GER, GERU (conj = false), GERC (conj = true): A := alpha * x * op(y)^T + A.
// Positions: layout 1, m 2, n 3, alpha 4, x 5, incx 6, y 7, incy 8, a 9,
// lda 10.
template <typename T>
void Ger(const char* name, bool conj, CBLAS_LAYOUT layout, CblasInt m,
         CblasInt n, T alpha, const void* x, CblasInt incx, const void* y,
         CblasInt incy, void* a, CblasInt lda) {
  if (!ValidLayout(name, layout)) return;
  if (layout == CblasColMajor) {
    CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10});
    if constexpr (IsComplex<T>::value) {
      if (conj) {
        Fortran<T>::gerc(&m, &n, F<T>(alpha), F<T>(x), &incx, F<T>(y), &incy,
                         F<T>(a), &lda);
        return;
      }
    }
    Fortran<T>::ger(&m, &n, F<T>(alpha), F<T>(x), &incx, F<T>(y), &incy,
                    F<T>(a), &lda);
    return;
  }
  // Row-major A is column-major A^T += alpha * op(y) * x^T (n x m), an
  // unconjugated update with the vectors swapped and y conjugated for GERC.
  CblasScope scope(name, {3, 2, 4, 7, 8, 5, 6, 9, 10});
  WithConjugatedInput<T>(conj && IsComplex<T>::value, n, y, incy,
                         [&](const void* yv, CblasInt iy) {
                           Fortran<T>::ger(&n, &m, F<T>(alpha), F<T>(yv), &iy,
                                           F<T>(x), &incx, F<T>(a), &lda);
                         });
}

// SYR, SPR (HER, HPR for complex types, real alpha): A := alpha x x^H + A.
// Positions: layout 1, uplo 2, n 3, alpha 4, x 5, incx 6, a 7, [lda 8].
template <typename T>
void Syr(const char* name, bool packed, CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
         CblasInt n, RealT<T> alpha, const void* x, CblasInt incx, void* a,
         CblasInt lda) {
  if (!ValidLayout(name, layout)) return;
  char uc = UploChar(uplo);
  // Row-major: the other triangle of A (symmetric) or of conj(A) (Hermitian),
  // and conj(A) += alpha * conj(x) * conj(x)^H.
  const bool row = (layout == CblasRowMajor);
  if (row) uc = FlipUplo(uc);
  CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8});
  WithConjugatedInput<T>(
      row && IsComplex<T>::value, n, x, incx, [&](const void* xv, CblasInt ix) {
        if (packed) {
          Fortran<T>::spr(&uc, &n, &alpha, F<T>(xv), &ix, F<T>(a));
        } else {
          Fortran<T>::syr(&uc, &n, &alpha, F<T>(xv), &ix, F<T>(a), &lda);
        }
      });
}

// SYR2, SPR2 (HER2, HPR2): A := alpha x y^H + conj(alpha) y x^H + A.
// Positions: layout 1, uplo 2, n 3, alpha 4, x 5, incx 6, y 7, incy 8, a 9,
// [lda 10].
template <typename T>
void Syr2(const char* name, bool packed, CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
          CblasInt n, T alpha, const void* x, CblasInt incx, const void* y,
          CblasInt incy, void* a, CblasInt lda) {
  if (!ValidLayout(name, layout)) return;
  char uc = UploChar(uplo);
  // Row-major Hermitian: conj(A) += conj(alpha) conj(x) conj(y)^H +
  // alpha conj(y) conj(x)^H, i.e. the same update with all conjugated.
  const bool row = (layout == CblasRowMajor);
  const bool conj = row && IsComplex<T>::value;
  if (row) uc = FlipUplo(uc);
  if (conj) alpha = Conj(alpha);
  CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10});
  WithConjugatedInput<T>(conj, n, x, incx, [&](const void* xv, CblasInt ix) {
    WithConjugatedInput<T>(conj, n, y, incy, [&](const void* yv, CblasInt iy) {
      if (packed) {
        Fortran<T>::spr2(&uc, &n, F<T>(alpha), F<T>(xv), &ix, F<T>(yv), &iy,
                         F<T>(a));
      } else {
        Fortran<T>::syr2(&uc, &n, F<T>(alpha), F<T>(xv), &ix, F<T>(yv), &iy,
                         F<T>(a), &lda);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Level 3

// GEMM: C := alpha * op(A) * op(B) + beta * C. Row-major: C^T = op(B)^T *
// op(A)^T, i.e. the column-major product with A and B swapped.
// Positions: layout 1, transa 2, transb 3, m 4, n 5, k 6, alpha 7, a 8,
// lda 9, b 10, ldb 11, beta 12, c 13, ldc 14.
template <typename T>
void Gemm(const char* name, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa,
          CBLAS_TRANSPOSE transb, CblasInt m, CblasInt n, CblasInt k, T alpha,
          const void* a, CblasInt lda, const void* b, CblasInt ldb, T beta,
          void* c, CblasInt ldc) {
  if (!ValidLayout(name, layout)) return;
  const char ta = TransChar(transa), tb = TransChar(transb);
  if (layout == CblasColMajor) {
    CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    Fortran<T>::gemm(&ta, &tb, &m, &n, &k, F<T>(alpha), F<T>(a), &lda, F<T>(b),
                     &ldb, F<T>(beta), F<T>(c), &ldc);
  } else {
    CblasScope scope(name, {3, 2, 5, 4, 6, 7, 10, 11, 8, 9, 12, 13, 14});
    Fortran<T>::gemm(&tb, &ta, &n, &m, &k, F<T>(alpha), F<T>(b), &ldb, F<T>(a),
                     &lda, F<T>(beta), F<T>(c), &ldc);
  }
}

// GEMMTR: as GEMM for the uplo triangle of the n x n matrix C.
// Positions: layout 1, uplo 2, transa 3, transb 4, n 5, k 6, alpha 7, a 8,
// lda 9, b 10, ldb 11, beta 12, c 13, ldc 14.
template <typename T>
void Gemmtr(const char* name, CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
            CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CblasInt n,
            CblasInt k, T alpha, const void* a, CblasInt lda, const void* b,
            CblasInt ldb, T beta, void* c, CblasInt ldc) {
  if (!ValidLayout(name, layout)) return;
  const char ta = TransChar(transa), tb = TransChar(transb);
  const char uc = UploChar(uplo);
  if (layout == CblasColMajor) {
    CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    Fortran<T>::gemmtr(&uc, &ta, &tb, &n, &k, F<T>(alpha), F<T>(a), &lda,
                       F<T>(b), &ldb, F<T>(beta), F<T>(c), &ldc);
  } else {
    const char ur = FlipUplo(uc);
    CblasScope scope(name, {2, 4, 3, 5, 6, 7, 10, 11, 8, 9, 12, 13, 14});
    Fortran<T>::gemmtr(&ur, &tb, &ta, &n, &k, F<T>(alpha), F<T>(b), &ldb,
                       F<T>(a), &lda, F<T>(beta), F<T>(c), &ldc);
  }
}

// SYMM, HEMM: C := alpha * A * B + beta * C (side L) or alpha * B * A + beta
// * C. Row-major: C^T = B^T * A^T, and the column-major storage of A is the
// other triangle of A^T (= A if symmetric, and the matrix A^T itself if
// Hermitian).
// Positions: layout 1, side 2, uplo 3, m 4, n 5, alpha 6, a 7, lda 8, b 9,
// ldb 10, beta 11, c 12, ldc 13.
template <typename T>
void Symm(const char* name, bool hermitian, CBLAS_LAYOUT layout,
          CBLAS_SIDE side, CBLAS_UPLO uplo, CblasInt m, CblasInt n, T alpha,
          const void* a, CblasInt lda, const void* b, CblasInt ldb, T beta,
          void* c, CblasInt ldc) {
  if (!ValidLayout(name, layout)) return;
  char sc = SideChar(side), uc = UploChar(uplo);
  const bool row = (layout == CblasRowMajor);
  if (row) {
    sc = FlipSide(sc);
    uc = FlipUplo(uc);
    std::swap(m, n);
  }
  CblasScope scope(
      name,
      row ? std::initializer_list<int>{2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13}
          : std::initializer_list<int>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
  if constexpr (IsComplex<T>::value) {
    if (hermitian) {
      Fortran<T>::hemm(&sc, &uc, &m, &n, F<T>(alpha), F<T>(a), &lda, F<T>(b),
                       &ldb, F<T>(beta), F<T>(c), &ldc);
      return;
    }
  }
  Fortran<T>::symm(&sc, &uc, &m, &n, F<T>(alpha), F<T>(a), &lda, F<T>(b), &ldb,
                   F<T>(beta), F<T>(c), &ldc);
}

// Transposition for SYRK and SYR2K (hermitian = false: N <-> T, and C
// is T for real types) or HERK and HER2K (N <-> C) of row-major matrices.
char FlipRankTrans(char c, bool hermitian) {
  if (c == 'N') return hermitian ? 'C' : 'T';
  if (c == 'T' || c == 'C') return 'N';
  return c;
}

// SYRK (S = T) and HERK (S real): C := alpha * op(A) * op(A)' + beta * C.
// Row-major: C^T is the same product of the column-major A^T, with the
// other triangle and transposition.
// Positions: layout 1, uplo 2, trans 3, n 4, k 5, alpha 6, a 7, lda 8,
// beta 9, c 10, ldc 11.
template <typename T, typename S>
void Syrk(const char* name, CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
          CBLAS_TRANSPOSE trans, CblasInt n, CblasInt k, S alpha, const void* a,
          CblasInt lda, S beta, void* c, CblasInt ldc) {
  constexpr bool kHermitian = IsComplex<T>::value && !std::is_same_v<S, T>;
  if (!ValidLayout(name, layout)) return;
  char uc = UploChar(uplo), tc = TransChar(trans);
  if (layout == CblasRowMajor) {
    uc = FlipUplo(uc);
    // Complex SYRK rejects C, HERK rejects T: keep invalid ones invalid.
    const bool invalid = IsComplex<T>::value && tc == (kHermitian ? 'T' : 'C');
    if (!invalid) tc = FlipRankTrans(tc, kHermitian);
  }
  CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  if constexpr (kHermitian) {
    Fortran<T>::herk(&uc, &tc, &n, &k, &alpha, F<T>(a), &lda, &beta, F<T>(c),
                     &ldc);
  } else {
    Fortran<T>::syrk(&uc, &tc, &n, &k, F<T>(alpha), F<T>(a), &lda, F<T>(beta),
                     F<T>(c), &ldc);
  }
}

// SYR2K (B = T) and HER2K (B real): C := alpha * op(A) * op(B)' +
// alpha2 * op(B) * op(A)' + beta * C. Row-major as SYRK; for HER2K the
// transposed problem has conj(alpha) in place of alpha.
// Positions: layout 1, uplo 2, trans 3, n 4, k 5, alpha 6, a 7, lda 8, b 9,
// ldb 10, beta 11, c 12, ldc 13.
template <typename T, typename B>
void Syr2k(const char* name, CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
           CBLAS_TRANSPOSE trans, CblasInt n, CblasInt k, T alpha,
           const void* a, CblasInt lda, const void* b, CblasInt ldb, B beta,
           void* c, CblasInt ldc) {
  constexpr bool hermitian = IsComplex<T>::value && !std::is_same_v<B, T>;
  if (!ValidLayout(name, layout)) return;
  char uc = UploChar(uplo), tc = TransChar(trans);
  if (layout == CblasRowMajor) {
    uc = FlipUplo(uc);
    const bool invalid = IsComplex<T>::value && tc == (hermitian ? 'T' : 'C');
    if (!invalid) tc = FlipRankTrans(tc, hermitian);
    if (hermitian) alpha = Conj(alpha);
  }
  CblasScope scope(name, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
  if constexpr (hermitian) {
    Fortran<T>::her2k(&uc, &tc, &n, &k, F<T>(alpha), F<T>(a), &lda, F<T>(b),
                      &ldb, &beta, F<T>(c), &ldc);
  } else {
    Fortran<T>::syr2k(&uc, &tc, &n, &k, F<T>(alpha), F<T>(a), &lda, F<T>(b),
                      &ldb, F<T>(beta), F<T>(c), &ldc);
  }
}

// TRMM, TRSM: B := alpha * op(A) * B or the solve, side L or R. Row-major:
// B^T = alpha * B^T * op(A)^T with the other side and triangle; op is
// unchanged, as the column-major storage of A is A^T.
// Positions: layout 1, side 2, uplo 3, transa 4, diag 5, m 6, n 7, alpha 8,
// a 9, lda 10, b 11, ldb 12.
template <typename T>
void Trmm(const char* name, bool solve, CBLAS_LAYOUT layout, CBLAS_SIDE side,
          CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, CblasInt m,
          CblasInt n, T alpha, const void* a, CblasInt lda, void* b,
          CblasInt ldb) {
  if (!ValidLayout(name, layout)) return;
  char sc = SideChar(side), uc = UploChar(uplo);
  const char tc = TransChar(transa), dc = DiagChar(diag);
  const bool row = (layout == CblasRowMajor);
  if (row) {
    sc = FlipSide(sc);
    uc = FlipUplo(uc);
    std::swap(m, n);
  }
  CblasScope scope(
      name,
      row ? std::initializer_list<int>{2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12}
          : std::initializer_list<int>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  if (solve) {
    Fortran<T>::trsm(&sc, &uc, &tc, &dc, &m, &n, F<T>(alpha), F<T>(a), &lda,
                     F<T>(b), &ldb);
  } else {
    Fortran<T>::trmm(&sc, &uc, &tc, &dc, &m, &n, F<T>(alpha), F<T>(a), &lda,
                     F<T>(b), &ldb);
  }
}

}  // namespace
}  // namespace hwyblas

// ---------------------------------------------------------------------------
// Entry points

namespace hwyblas {

// Complex scalar arguments are passed by pointer.
template <typename T>
T Scalar(const void* p) {
  return *static_cast<const T*>(p);
}

}  // namespace hwyblas

using hwyblas::SymmetricKind;
using hwyblas::TriangularKind;

extern "C" {

HWYBLAS_API void cblas_xerbla(hwyblas_int p, const char* rout, const char* form,
                              ...) {
  if (p != 0) {
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n",
            static_cast<int>(p), rout);
  }
  if (form != nullptr && *form != '\0') {
    va_list args;
    va_start(args, form);
    vfprintf(stderr, form, args);
    va_end(args);
  }
}

HWYBLAS_API float cblas_sdot(const hwyblas_int n, const float* x,
                             const hwyblas_int incx, const float* y,
                             const hwyblas_int incy) {
  return sdot_(&n, x, &incx, y, &incy);
}
HWYBLAS_API double cblas_ddot(const hwyblas_int n, const double* x,
                              const hwyblas_int incx, const double* y,
                              const hwyblas_int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}
HWYBLAS_API float cblas_sdsdot(const hwyblas_int n, const float alpha,
                               const float* x, const hwyblas_int incx,
                               const float* y, const hwyblas_int incy) {
  return sdsdot_(&n, &alpha, x, &incx, y, &incy);
}
HWYBLAS_API double cblas_dsdot(const hwyblas_int n, const float* x,
                               const hwyblas_int incx, const float* y,
                               const hwyblas_int incy) {
  return dsdot_(&n, x, &incx, y, &incy);
}
HWYBLAS_API hwyblas_complex_float cblas_cdotu(const hwyblas_int n,
                                              const void* x,
                                              const hwyblas_int incx,
                                              const void* y,
                                              const hwyblas_int incy) {
  return cdotu_(&n, hwyblas::F<hwyblas::c32>(x), &incx,
                hwyblas::F<hwyblas::c32>(y), &incy);
}
HWYBLAS_API void cblas_cdotu_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret) {
  *static_cast<hwyblas_complex_float*>(ret) =
      cdotu_(&n, hwyblas::F<hwyblas::c32>(x), &incx,
             hwyblas::F<hwyblas::c32>(y), &incy);
}
HWYBLAS_API hwyblas_complex_float cblas_cdotc(const hwyblas_int n,
                                              const void* x,
                                              const hwyblas_int incx,
                                              const void* y,
                                              const hwyblas_int incy) {
  return cdotc_(&n, hwyblas::F<hwyblas::c32>(x), &incx,
                hwyblas::F<hwyblas::c32>(y), &incy);
}
HWYBLAS_API void cblas_cdotc_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret) {
  *static_cast<hwyblas_complex_float*>(ret) =
      cdotc_(&n, hwyblas::F<hwyblas::c32>(x), &incx,
             hwyblas::F<hwyblas::c32>(y), &incy);
}
HWYBLAS_API hwyblas_complex_double cblas_zdotu(const hwyblas_int n,
                                               const void* x,
                                               const hwyblas_int incx,
                                               const void* y,
                                               const hwyblas_int incy) {
  return zdotu_(&n, hwyblas::F<hwyblas::c64>(x), &incx,
                hwyblas::F<hwyblas::c64>(y), &incy);
}
HWYBLAS_API void cblas_zdotu_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret) {
  *static_cast<hwyblas_complex_double*>(ret) =
      zdotu_(&n, hwyblas::F<hwyblas::c64>(x), &incx,
             hwyblas::F<hwyblas::c64>(y), &incy);
}
HWYBLAS_API hwyblas_complex_double cblas_zdotc(const hwyblas_int n,
                                               const void* x,
                                               const hwyblas_int incx,
                                               const void* y,
                                               const hwyblas_int incy) {
  return zdotc_(&n, hwyblas::F<hwyblas::c64>(x), &incx,
                hwyblas::F<hwyblas::c64>(y), &incy);
}
HWYBLAS_API void cblas_zdotc_sub(const hwyblas_int n, const void* x,
                                 const hwyblas_int incx, const void* y,
                                 const hwyblas_int incy, void* ret) {
  *static_cast<hwyblas_complex_double*>(ret) =
      zdotc_(&n, hwyblas::F<hwyblas::c64>(x), &incx,
             hwyblas::F<hwyblas::c64>(y), &incy);
}
HWYBLAS_API float cblas_snrm2(const hwyblas_int n, const float* x,
                              const hwyblas_int incx) {
  return snrm2_(&n, x, &incx);
}
HWYBLAS_API double cblas_dnrm2(const hwyblas_int n, const double* x,
                               const hwyblas_int incx) {
  return dnrm2_(&n, x, &incx);
}
HWYBLAS_API float cblas_scnrm2(const hwyblas_int n, const void* x,
                               const hwyblas_int incx) {
  return scnrm2_(&n, hwyblas::F<hwyblas::c32>(x), &incx);
}
HWYBLAS_API double cblas_dznrm2(const hwyblas_int n, const void* x,
                                const hwyblas_int incx) {
  return dznrm2_(&n, hwyblas::F<hwyblas::c64>(x), &incx);
}
HWYBLAS_API float cblas_sasum(const hwyblas_int n, const float* x,
                              const hwyblas_int incx) {
  return sasum_(&n, x, &incx);
}
HWYBLAS_API double cblas_dasum(const hwyblas_int n, const double* x,
                               const hwyblas_int incx) {
  return dasum_(&n, x, &incx);
}
HWYBLAS_API float cblas_scasum(const hwyblas_int n, const void* x,
                               const hwyblas_int incx) {
  return scasum_(&n, hwyblas::F<hwyblas::c32>(x), &incx);
}
HWYBLAS_API double cblas_dzasum(const hwyblas_int n, const void* x,
                                const hwyblas_int incx) {
  return dzasum_(&n, hwyblas::F<hwyblas::c64>(x), &incx);
}
HWYBLAS_API CBLAS_INDEX cblas_isamax(const hwyblas_int n, const float* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = isamax_(&n, x, &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API CBLAS_INDEX cblas_idamax(const hwyblas_int n, const double* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = idamax_(&n, x, &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API CBLAS_INDEX cblas_icamax(const hwyblas_int n, const void* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = icamax_(&n, hwyblas::F<hwyblas::c32>(x), &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API CBLAS_INDEX cblas_izamax(const hwyblas_int n, const void* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = izamax_(&n, hwyblas::F<hwyblas::c64>(x), &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API void cblas_sswap(const hwyblas_int n, float* x,
                             const hwyblas_int incx, float* y,
                             const hwyblas_int incy) {
  sswap_(&n, x, &incx, y, &incy);
}
HWYBLAS_API void cblas_dswap(const hwyblas_int n, double* x,
                             const hwyblas_int incx, double* y,
                             const hwyblas_int incy) {
  dswap_(&n, x, &incx, y, &incy);
}
HWYBLAS_API void cblas_cswap(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy) {
  cswap_(&n, hwyblas::F<hwyblas::c32>(x), &incx, hwyblas::F<hwyblas::c32>(y),
         &incy);
}
HWYBLAS_API void cblas_zswap(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy) {
  zswap_(&n, hwyblas::F<hwyblas::c64>(x), &incx, hwyblas::F<hwyblas::c64>(y),
         &incy);
}
HWYBLAS_API void cblas_scopy(const hwyblas_int n, const float* x,
                             const hwyblas_int incx, float* y,
                             const hwyblas_int incy) {
  scopy_(&n, x, &incx, y, &incy);
}
HWYBLAS_API void cblas_dcopy(const hwyblas_int n, const double* x,
                             const hwyblas_int incx, double* y,
                             const hwyblas_int incy) {
  dcopy_(&n, x, &incx, y, &incy);
}
HWYBLAS_API void cblas_ccopy(const hwyblas_int n, const void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy) {
  ccopy_(&n, hwyblas::F<hwyblas::c32>(x), &incx, hwyblas::F<hwyblas::c32>(y),
         &incy);
}
HWYBLAS_API void cblas_zcopy(const hwyblas_int n, const void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy) {
  zcopy_(&n, hwyblas::F<hwyblas::c64>(x), &incx, hwyblas::F<hwyblas::c64>(y),
         &incy);
}
HWYBLAS_API void cblas_saxpy(const hwyblas_int n, const float alpha,
                             const float* x, const hwyblas_int incx, float* y,
                             const hwyblas_int incy) {
  saxpy_(&n, &alpha, x, &incx, y, &incy);
}
HWYBLAS_API void cblas_daxpy(const hwyblas_int n, const double alpha,
                             const double* x, const hwyblas_int incx, double* y,
                             const hwyblas_int incy) {
  daxpy_(&n, &alpha, x, &incx, y, &incy);
}
HWYBLAS_API void cblas_caxpy(const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx, void* y,
                             const hwyblas_int incy) {
  caxpy_(&n, hwyblas::F<hwyblas::c32>(alpha), hwyblas::F<hwyblas::c32>(x),
         &incx, hwyblas::F<hwyblas::c32>(y), &incy);
}
HWYBLAS_API void cblas_zaxpy(const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx, void* y,
                             const hwyblas_int incy) {
  zaxpy_(&n, hwyblas::F<hwyblas::c64>(alpha), hwyblas::F<hwyblas::c64>(x),
         &incx, hwyblas::F<hwyblas::c64>(y), &incy);
}
HWYBLAS_API void cblas_sscal(const hwyblas_int n, const float alpha, float* x,
                             const hwyblas_int incx) {
  sscal_(&n, &alpha, x, &incx);
}
HWYBLAS_API void cblas_dscal(const hwyblas_int n, const double alpha, double* x,
                             const hwyblas_int incx) {
  dscal_(&n, &alpha, x, &incx);
}
HWYBLAS_API void cblas_cscal(const hwyblas_int n, const void* alpha, void* x,
                             const hwyblas_int incx) {
  cscal_(&n, hwyblas::F<hwyblas::c32>(alpha), hwyblas::F<hwyblas::c32>(x),
         &incx);
}
HWYBLAS_API void cblas_zscal(const hwyblas_int n, const void* alpha, void* x,
                             const hwyblas_int incx) {
  zscal_(&n, hwyblas::F<hwyblas::c64>(alpha), hwyblas::F<hwyblas::c64>(x),
         &incx);
}
HWYBLAS_API void cblas_csscal(const hwyblas_int n, const float alpha, void* x,
                              const hwyblas_int incx) {
  csscal_(&n, &alpha, hwyblas::F<hwyblas::c32>(x), &incx);
}
HWYBLAS_API void cblas_zdscal(const hwyblas_int n, const double alpha, void* x,
                              const hwyblas_int incx) {
  zdscal_(&n, &alpha, hwyblas::F<hwyblas::c64>(x), &incx);
}
HWYBLAS_API void cblas_srot(const hwyblas_int n, float* x,
                            const hwyblas_int incx, float* y,
                            const hwyblas_int incy, const float c,
                            const float s) {
  srot_(&n, x, &incx, y, &incy, &c, &s);
}
HWYBLAS_API void cblas_drot(const hwyblas_int n, double* x,
                            const hwyblas_int incx, double* y,
                            const hwyblas_int incy, const double c,
                            const double s) {
  drot_(&n, x, &incx, y, &incy, &c, &s);
}
HWYBLAS_API void cblas_csrot(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy, const float c,
                             const float s) {
  csrot_(&n, hwyblas::F<hwyblas::c32>(x), &incx, hwyblas::F<hwyblas::c32>(y),
         &incy, &c, &s);
}
HWYBLAS_API void cblas_zdrot(const hwyblas_int n, void* x,
                             const hwyblas_int incx, void* y,
                             const hwyblas_int incy, const double c,
                             const double s) {
  zdrot_(&n, hwyblas::F<hwyblas::c64>(x), &incx, hwyblas::F<hwyblas::c64>(y),
         &incy, &c, &s);
}
HWYBLAS_API void cblas_srotg(float* a, float* b, float* c, float* s) {
  srotg_(a, b, c, s);
}
HWYBLAS_API void cblas_drotg(double* a, double* b, double* c, double* s) {
  drotg_(a, b, c, s);
}
HWYBLAS_API void cblas_crotg(void* a, const void* b, float* c, void* s) {
  crotg_(hwyblas::F<hwyblas::c32>(a), hwyblas::F<hwyblas::c32>(b), c,
         hwyblas::F<hwyblas::c32>(s));
}
HWYBLAS_API void cblas_zrotg(void* a, const void* b, double* c, void* s) {
  zrotg_(hwyblas::F<hwyblas::c64>(a), hwyblas::F<hwyblas::c64>(b), c,
         hwyblas::F<hwyblas::c64>(s));
}
HWYBLAS_API void cblas_srotm(const hwyblas_int n, float* x,
                             const hwyblas_int incx, float* y,
                             const hwyblas_int incy, const float* p) {
  srotm_(&n, x, &incx, y, &incy, p);
}
HWYBLAS_API void cblas_drotm(const hwyblas_int n, double* x,
                             const hwyblas_int incx, double* y,
                             const hwyblas_int incy, const double* p) {
  drotm_(&n, x, &incx, y, &incy, p);
}
HWYBLAS_API void cblas_srotmg(float* d1, float* d2, float* b1, const float b2,
                              float* p) {
  srotmg_(d1, d2, b1, &b2, p);
}
HWYBLAS_API void cblas_drotmg(double* d1, double* d2, double* b1,
                              const double b2, double* p) {
  drotmg_(d1, d2, b1, &b2, p);
}
HWYBLAS_API void cblas_sgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* x, const hwyblas_int incx,
                             const float beta, float* y,
                             const hwyblas_int incy) {
  hwyblas::Gemv<float>("cblas_sgemv", false, layout, trans, m, n, 0, 0, alpha,
                       a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void cblas_dgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* x, const hwyblas_int incx,
                             const double beta, double* y,
                             const hwyblas_int incy) {
  hwyblas::Gemv<double>("cblas_dgemv", false, layout, trans, m, n, 0, 0, alpha,
                        a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void cblas_cgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Gemv<hwyblas::c32>("cblas_cgemv", false, layout, trans, m, n, 0, 0,
                              hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, x,
                              incx, hwyblas::Scalar<hwyblas::c32>(beta), y,
                              incy);
}
HWYBLAS_API void cblas_zgemv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Gemv<hwyblas::c64>("cblas_zgemv", false, layout, trans, m, n, 0, 0,
                              hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, x,
                              incx, hwyblas::Scalar<hwyblas::c64>(beta), y,
                              incy);
}
HWYBLAS_API void cblas_sgbmv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int kl,
                             const hwyblas_int ku, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* x, const hwyblas_int incx,
                             const float beta, float* y,
                             const hwyblas_int incy) {
  hwyblas::Gemv<float>("cblas_sgbmv", true, layout, trans, m, n, kl, ku, alpha,
                       a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void cblas_dgbmv(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int kl,
                             const hwyblas_int ku, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* x, const hwyblas_int incx,
                             const double beta, double* y,
                             const hwyblas_int incy) {
  hwyblas::Gemv<double>("cblas_dgbmv", true, layout, trans, m, n, kl, ku, alpha,
                        a, lda, x, incx, beta, y, incy);
}
HWYBLAS_API void cblas_cgbmv(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, const hwyblas_int m,
    const hwyblas_int n, const hwyblas_int kl, const hwyblas_int ku,
    const void* alpha, const void* a, const hwyblas_int lda, const void* x,
    const hwyblas_int incx, const void* beta, void* y, const hwyblas_int incy) {
  hwyblas::Gemv<hwyblas::c32>("cblas_cgbmv", true, layout, trans, m, n, kl, ku,
                              hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, x,
                              incx, hwyblas::Scalar<hwyblas::c32>(beta), y,
                              incy);
}
HWYBLAS_API void cblas_zgbmv(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, const hwyblas_int m,
    const hwyblas_int n, const hwyblas_int kl, const hwyblas_int ku,
    const void* alpha, const void* a, const hwyblas_int lda, const void* x,
    const hwyblas_int incx, const void* beta, void* y, const hwyblas_int incy) {
  hwyblas::Gemv<hwyblas::c64>("cblas_zgbmv", true, layout, trans, m, n, kl, ku,
                              hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, x,
                              incx, hwyblas::Scalar<hwyblas::c64>(beta), y,
                              incy);
}
HWYBLAS_API void cblas_strmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* a,
                             const hwyblas_int lda, float* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<float>("cblas_strmv", hwyblas::TriangularKind::kTrmv,
                             layout, uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_dtrmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* a,
                             const hwyblas_int lda, double* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<double>("cblas_dtrmv", hwyblas::TriangularKind::kTrmv,
                              layout, uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_ctrmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c32>("cblas_ctrmv",
                                    hwyblas::TriangularKind::kTrmv, layout,
                                    uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_ztrmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c64>("cblas_ztrmv",
                                    hwyblas::TriangularKind::kTrmv, layout,
                                    uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_strsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* a,
                             const hwyblas_int lda, float* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<float>("cblas_strsv", hwyblas::TriangularKind::kTrsv,
                             layout, uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_dtrsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* a,
                             const hwyblas_int lda, double* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<double>("cblas_dtrsv", hwyblas::TriangularKind::kTrsv,
                              layout, uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_ctrsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c32>("cblas_ctrsv",
                                    hwyblas::TriangularKind::kTrsv, layout,
                                    uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_ztrsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* a,
                             const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c64>("cblas_ztrsv",
                                    hwyblas::TriangularKind::kTrsv, layout,
                                    uplo, trans, diag, n, 0, a, lda, x, incx);
}
HWYBLAS_API void cblas_stbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const float* a, const hwyblas_int lda, float* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<float>("cblas_stbmv", hwyblas::TriangularKind::kTbmv,
                             layout, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_dtbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const double* a, const hwyblas_int lda, double* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<double>("cblas_dtbmv", hwyblas::TriangularKind::kTbmv,
                              layout, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_ctbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c32>("cblas_ctbmv",
                                    hwyblas::TriangularKind::kTbmv, layout,
                                    uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_ztbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c64>("cblas_ztbmv",
                                    hwyblas::TriangularKind::kTbmv, layout,
                                    uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_stbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const float* a, const hwyblas_int lda, float* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<float>("cblas_stbsv", hwyblas::TriangularKind::kTbsv,
                             layout, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_dtbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const double* a, const hwyblas_int lda, double* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<double>("cblas_dtbsv", hwyblas::TriangularKind::kTbsv,
                              layout, uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_ctbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c32>("cblas_ctbsv",
                                    hwyblas::TriangularKind::kTbsv, layout,
                                    uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_ztbsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* a, const hwyblas_int lda, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c64>("cblas_ztbsv",
                                    hwyblas::TriangularKind::kTbsv, layout,
                                    uplo, trans, diag, n, k, a, lda, x, incx);
}
HWYBLAS_API void cblas_stpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* ap, float* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<float>("cblas_stpmv", hwyblas::TriangularKind::kTpmv,
                             layout, uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_dtpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* ap, double* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<double>("cblas_dtpmv", hwyblas::TriangularKind::kTpmv,
                              layout, uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_ctpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c32>("cblas_ctpmv",
                                    hwyblas::TriangularKind::kTpmv, layout,
                                    uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_ztpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c64>("cblas_ztpmv",
                                    hwyblas::TriangularKind::kTpmv, layout,
                                    uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_stpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const float* ap, float* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<float>("cblas_stpsv", hwyblas::TriangularKind::kTpsv,
                             layout, uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_dtpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const double* ap, double* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<double>("cblas_dtpsv", hwyblas::TriangularKind::kTpsv,
                              layout, uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_ctpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c32>("cblas_ctpsv",
                                    hwyblas::TriangularKind::kTpsv, layout,
                                    uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_ztpsv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const CBLAS_DIAG diag,
                             const hwyblas_int n, const void* ap, void* x,
                             const hwyblas_int incx) {
  hwyblas::Triangular<hwyblas::c64>("cblas_ztpsv",
                                    hwyblas::TriangularKind::kTpsv, layout,
                                    uplo, trans, diag, n, 0, ap, 0, x, incx);
}
HWYBLAS_API void cblas_ssymv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* x, const hwyblas_int incx,
                             const float beta, float* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<float>("cblas_ssymv", hwyblas::SymmetricKind::kFull,
                            layout, uplo, n, 0, alpha, a, lda, x, incx, beta, y,
                            incy);
}
HWYBLAS_API void cblas_dsymv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* x, const hwyblas_int incx,
                             const double beta, double* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<double>("cblas_dsymv", hwyblas::SymmetricKind::kFull,
                             layout, uplo, n, 0, alpha, a, lda, x, incx, beta,
                             y, incy);
}
HWYBLAS_API void cblas_chemv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<hwyblas::c32>(
      "cblas_chemv", hwyblas::SymmetricKind::kFull, layout, uplo, n, 0,
      hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, x, incx,
      hwyblas::Scalar<hwyblas::c32>(beta), y, incy);
}
HWYBLAS_API void cblas_zhemv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* x, const hwyblas_int incx,
                             const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<hwyblas::c64>(
      "cblas_zhemv", hwyblas::SymmetricKind::kFull, layout, uplo, n, 0,
      hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, x, incx,
      hwyblas::Scalar<hwyblas::c64>(beta), y, incy);
}
HWYBLAS_API void cblas_ssbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const float alpha, const float* a,
                             const hwyblas_int lda, const float* x,
                             const hwyblas_int incx, const float beta, float* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<float>("cblas_ssbmv", hwyblas::SymmetricKind::kBand,
                            layout, uplo, n, k, alpha, a, lda, x, incx, beta, y,
                            incy);
}
HWYBLAS_API void cblas_dsbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const double alpha, const double* a,
                             const hwyblas_int lda, const double* x,
                             const hwyblas_int incx, const double beta,
                             double* y, const hwyblas_int incy) {
  hwyblas::Symmetric<double>("cblas_dsbmv", hwyblas::SymmetricKind::kBand,
                             layout, uplo, n, k, alpha, a, lda, x, incx, beta,
                             y, incy);
}
HWYBLAS_API void cblas_chbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<hwyblas::c32>(
      "cblas_chbmv", hwyblas::SymmetricKind::kBand, layout, uplo, n, k,
      hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, x, incx,
      hwyblas::Scalar<hwyblas::c32>(beta), y, incy);
}
HWYBLAS_API void cblas_zhbmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<hwyblas::c64>(
      "cblas_zhbmv", hwyblas::SymmetricKind::kBand, layout, uplo, n, k,
      hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, x, incx,
      hwyblas::Scalar<hwyblas::c64>(beta), y, incy);
}
HWYBLAS_API void cblas_sspmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* ap, const float* x,
                             const hwyblas_int incx, const float beta, float* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<float>("cblas_sspmv", hwyblas::SymmetricKind::kPacked,
                            layout, uplo, n, 0, alpha, ap, 0, x, incx, beta, y,
                            incy);
}
HWYBLAS_API void cblas_dspmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* ap, const double* x,
                             const hwyblas_int incx, const double beta,
                             double* y, const hwyblas_int incy) {
  hwyblas::Symmetric<double>("cblas_dspmv", hwyblas::SymmetricKind::kPacked,
                             layout, uplo, n, 0, alpha, ap, 0, x, incx, beta, y,
                             incy);
}
HWYBLAS_API void cblas_chpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* ap, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<hwyblas::c32>(
      "cblas_chpmv", hwyblas::SymmetricKind::kPacked, layout, uplo, n, 0,
      hwyblas::Scalar<hwyblas::c32>(alpha), ap, 0, x, incx,
      hwyblas::Scalar<hwyblas::c32>(beta), y, incy);
}
HWYBLAS_API void cblas_zhpmv(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* ap, const void* x,
                             const hwyblas_int incx, const void* beta, void* y,
                             const hwyblas_int incy) {
  hwyblas::Symmetric<hwyblas::c64>(
      "cblas_zhpmv", hwyblas::SymmetricKind::kPacked, layout, uplo, n, 0,
      hwyblas::Scalar<hwyblas::c64>(alpha), ap, 0, x, incx,
      hwyblas::Scalar<hwyblas::c64>(beta), y, incy);
}
HWYBLAS_API void cblas_sger(const CBLAS_LAYOUT layout, const hwyblas_int m,
                            const hwyblas_int n, const float alpha,
                            const float* x, const hwyblas_int incx,
                            const float* y, const hwyblas_int incy, float* a,
                            const hwyblas_int lda) {
  hwyblas::Ger<float>("cblas_sger", false, layout, m, n, alpha, x, incx, y,
                      incy, a, lda);
}
HWYBLAS_API void cblas_dger(const CBLAS_LAYOUT layout, const hwyblas_int m,
                            const hwyblas_int n, const double alpha,
                            const double* x, const hwyblas_int incx,
                            const double* y, const hwyblas_int incy, double* a,
                            const hwyblas_int lda) {
  hwyblas::Ger<double>("cblas_dger", false, layout, m, n, alpha, x, incx, y,
                       incy, a, lda);
}
HWYBLAS_API void cblas_cgeru(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda) {
  hwyblas::Ger<hwyblas::c32>("cblas_cgeru", false, layout, m, n,
                             hwyblas::Scalar<hwyblas::c32>(alpha), x, incx, y,
                             incy, a, lda);
}
HWYBLAS_API void cblas_zgeru(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda) {
  hwyblas::Ger<hwyblas::c64>("cblas_zgeru", false, layout, m, n,
                             hwyblas::Scalar<hwyblas::c64>(alpha), x, incx, y,
                             incy, a, lda);
}
HWYBLAS_API void cblas_cgerc(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda) {
  hwyblas::Ger<hwyblas::c32>("cblas_cgerc", true, layout, m, n,
                             hwyblas::Scalar<hwyblas::c32>(alpha), x, incx, y,
                             incy, a, lda);
}
HWYBLAS_API void cblas_zgerc(const CBLAS_LAYOUT layout, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda) {
  hwyblas::Ger<hwyblas::c64>("cblas_zgerc", true, layout, m, n,
                             hwyblas::Scalar<hwyblas::c64>(alpha), x, incx, y,
                             incy, a, lda);
}
HWYBLAS_API void cblas_ssyr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const float* x, const hwyblas_int incx, float* a,
                            const hwyblas_int lda) {
  hwyblas::Syr<float>("cblas_ssyr", false, layout, uplo, n, alpha, x, incx, a,
                      lda);
}
HWYBLAS_API void cblas_dsyr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const double* x, const hwyblas_int incx, double* a,
                            const hwyblas_int lda) {
  hwyblas::Syr<double>("cblas_dsyr", false, layout, uplo, n, alpha, x, incx, a,
                       lda);
}
HWYBLAS_API void cblas_cher(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const void* x, const hwyblas_int incx, void* a,
                            const hwyblas_int lda) {
  hwyblas::Syr<hwyblas::c32>("cblas_cher", false, layout, uplo, n, alpha, x,
                             incx, a, lda);
}
HWYBLAS_API void cblas_zher(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const void* x, const hwyblas_int incx, void* a,
                            const hwyblas_int lda) {
  hwyblas::Syr<hwyblas::c64>("cblas_zher", false, layout, uplo, n, alpha, x,
                             incx, a, lda);
}
HWYBLAS_API void cblas_sspr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const float* x, const hwyblas_int incx, float* ap) {
  hwyblas::Syr<float>("cblas_sspr", true, layout, uplo, n, alpha, x, incx, ap,
                      0);
}
HWYBLAS_API void cblas_dspr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const double* x, const hwyblas_int incx,
                            double* ap) {
  hwyblas::Syr<double>("cblas_dspr", true, layout, uplo, n, alpha, x, incx, ap,
                       0);
}
HWYBLAS_API void cblas_chpr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const float alpha,
                            const void* x, const hwyblas_int incx, void* ap) {
  hwyblas::Syr<hwyblas::c32>("cblas_chpr", true, layout, uplo, n, alpha, x,
                             incx, ap, 0);
}
HWYBLAS_API void cblas_zhpr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                            const hwyblas_int n, const double alpha,
                            const void* x, const hwyblas_int incx, void* ap) {
  hwyblas::Syr<hwyblas::c64>("cblas_zhpr", true, layout, uplo, n, alpha, x,
                             incx, ap, 0);
}
HWYBLAS_API void cblas_ssyr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* x, const hwyblas_int incx,
                             const float* y, const hwyblas_int incy, float* a,
                             const hwyblas_int lda) {
  hwyblas::Syr2<float>("cblas_ssyr2", false, layout, uplo, n, alpha, x, incx, y,
                       incy, a, lda);
}
HWYBLAS_API void cblas_dsyr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* x, const hwyblas_int incx,
                             const double* y, const hwyblas_int incy, double* a,
                             const hwyblas_int lda) {
  hwyblas::Syr2<double>("cblas_dsyr2", false, layout, uplo, n, alpha, x, incx,
                        y, incy, a, lda);
}
HWYBLAS_API void cblas_cher2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda) {
  hwyblas::Syr2<hwyblas::c32>("cblas_cher2", false, layout, uplo, n,
                              hwyblas::Scalar<hwyblas::c32>(alpha), x, incx, y,
                              incy, a, lda);
}
HWYBLAS_API void cblas_zher2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* a,
                             const hwyblas_int lda) {
  hwyblas::Syr2<hwyblas::c64>("cblas_zher2", false, layout, uplo, n,
                              hwyblas::Scalar<hwyblas::c64>(alpha), x, incx, y,
                              incy, a, lda);
}
HWYBLAS_API void cblas_sspr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const float alpha,
                             const float* x, const hwyblas_int incx,
                             const float* y, const hwyblas_int incy,
                             float* ap) {
  hwyblas::Syr2<float>("cblas_sspr2", true, layout, uplo, n, alpha, x, incx, y,
                       incy, ap, 0);
}
HWYBLAS_API void cblas_dspr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const double alpha,
                             const double* x, const hwyblas_int incx,
                             const double* y, const hwyblas_int incy,
                             double* ap) {
  hwyblas::Syr2<double>("cblas_dspr2", true, layout, uplo, n, alpha, x, incx, y,
                        incy, ap, 0);
}
HWYBLAS_API void cblas_chpr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* ap) {
  hwyblas::Syr2<hwyblas::c32>("cblas_chpr2", true, layout, uplo, n,
                              hwyblas::Scalar<hwyblas::c32>(alpha), x, incx, y,
                              incy, ap, 0);
}
HWYBLAS_API void cblas_zhpr2(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const hwyblas_int n, const void* alpha,
                             const void* x, const hwyblas_int incx,
                             const void* y, const hwyblas_int incy, void* ap) {
  hwyblas::Syr2<hwyblas::c64>("cblas_zhpr2", true, layout, uplo, n,
                              hwyblas::Scalar<hwyblas::c64>(alpha), x, incx, y,
                              incy, ap, 0);
}
HWYBLAS_API void cblas_sgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const float alpha, const float* a,
                             const hwyblas_int lda, const float* b,
                             const hwyblas_int ldb, const float beta, float* c,
                             const hwyblas_int ldc) {
  hwyblas::Gemm<float>("cblas_sgemm", layout, transa, transb, m, n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_dgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const double alpha, const double* a,
                             const hwyblas_int lda, const double* b,
                             const hwyblas_int ldb, const double beta,
                             double* c, const hwyblas_int ldc) {
  hwyblas::Gemm<double>("cblas_dgemm", layout, transa, transb, m, n, k, alpha,
                        a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_cgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* b,
                             const hwyblas_int ldb, const void* beta, void* c,
                             const hwyblas_int ldc) {
  hwyblas::Gemm<hwyblas::c32>("cblas_cgemm", layout, transa, transb, m, n, k,
                              hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, b,
                              ldb, hwyblas::Scalar<hwyblas::c32>(beta), c, ldc);
}
HWYBLAS_API void cblas_zgemm(const CBLAS_LAYOUT layout,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_TRANSPOSE transb, const hwyblas_int m,
                             const hwyblas_int n, const hwyblas_int k,
                             const void* alpha, const void* a,
                             const hwyblas_int lda, const void* b,
                             const hwyblas_int ldb, const void* beta, void* c,
                             const hwyblas_int ldc) {
  hwyblas::Gemm<hwyblas::c64>("cblas_zgemm", layout, transa, transb, m, n, k,
                              hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, b,
                              ldb, hwyblas::Scalar<hwyblas::c64>(beta), c, ldc);
}
HWYBLAS_API void cblas_sgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const float alpha, const float* a,
                               const hwyblas_int lda, const float* b,
                               const hwyblas_int ldb, const float beta,
                               float* c, const hwyblas_int ldc) {
  hwyblas::Gemmtr<float>("cblas_sgemmtr", layout, uplo, transa, transb, n, k,
                         alpha, a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_dgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const double alpha, const double* a,
                               const hwyblas_int lda, const double* b,
                               const hwyblas_int ldb, const double beta,
                               double* c, const hwyblas_int ldc) {
  hwyblas::Gemmtr<double>("cblas_dgemmtr", layout, uplo, transa, transb, n, k,
                          alpha, a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_cgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const void* alpha, const void* a,
                               const hwyblas_int lda, const void* b,
                               const hwyblas_int ldb, const void* beta, void* c,
                               const hwyblas_int ldc) {
  hwyblas::Gemmtr<hwyblas::c32>("cblas_cgemmtr", layout, uplo, transa, transb,
                                n, k, hwyblas::Scalar<hwyblas::c32>(alpha), a,
                                lda, b, ldb,
                                hwyblas::Scalar<hwyblas::c32>(beta), c, ldc);
}
HWYBLAS_API void cblas_zgemmtr(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const hwyblas_int n, const hwyblas_int k,
                               const void* alpha, const void* a,
                               const hwyblas_int lda, const void* b,
                               const hwyblas_int ldb, const void* beta, void* c,
                               const hwyblas_int ldc) {
  hwyblas::Gemmtr<hwyblas::c64>("cblas_zgemmtr", layout, uplo, transa, transb,
                                n, k, hwyblas::Scalar<hwyblas::c64>(alpha), a,
                                lda, b, ldb,
                                hwyblas::Scalar<hwyblas::c64>(beta), c, ldc);
}
HWYBLAS_API void cblas_sgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const float alpha,
                              const float* a, const hwyblas_int lda,
                              const float* b, const hwyblas_int ldb,
                              const float beta, float* c,
                              const hwyblas_int ldc) {
  hwyblas::Gemmtr<float>("cblas_sgemmt", layout, uplo, transa, transb, n, k,
                         alpha, a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_dgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const double alpha,
                              const double* a, const hwyblas_int lda,
                              const double* b, const hwyblas_int ldb,
                              const double beta, double* c,
                              const hwyblas_int ldc) {
  hwyblas::Gemmtr<double>("cblas_dgemmt", layout, uplo, transa, transb, n, k,
                          alpha, a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_cgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c,
                              const hwyblas_int ldc) {
  hwyblas::Gemmtr<hwyblas::c32>("cblas_cgemmt", layout, uplo, transa, transb, n,
                                k, hwyblas::Scalar<hwyblas::c32>(alpha), a, lda,
                                b, ldb, hwyblas::Scalar<hwyblas::c32>(beta), c,
                                ldc);
}
HWYBLAS_API void cblas_zgemmt(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE transa,
                              const CBLAS_TRANSPOSE transb, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c,
                              const hwyblas_int ldc) {
  hwyblas::Gemmtr<hwyblas::c64>("cblas_zgemmt", layout, uplo, transa, transb, n,
                                k, hwyblas::Scalar<hwyblas::c64>(alpha), a, lda,
                                b, ldb, hwyblas::Scalar<hwyblas::c64>(beta), c,
                                ldc);
}
HWYBLAS_API void cblas_ssymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float* b, const hwyblas_int ldb,
                             const float beta, float* c,
                             const hwyblas_int ldc) {
  hwyblas::Symm<float>("cblas_ssymm", false, layout, side, uplo, m, n, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_dsymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double* b, const hwyblas_int ldb,
                             const double beta, double* c,
                             const hwyblas_int ldc) {
  hwyblas::Symm<double>("cblas_dsymm", false, layout, side, uplo, m, n, alpha,
                        a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_csymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc) {
  hwyblas::Symm<hwyblas::c32>("cblas_csymm", false, layout, side, uplo, m, n,
                              hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, b,
                              ldb, hwyblas::Scalar<hwyblas::c32>(beta), c, ldc);
}
HWYBLAS_API void cblas_zsymm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc) {
  hwyblas::Symm<hwyblas::c64>("cblas_zsymm", false, layout, side, uplo, m, n,
                              hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, b,
                              ldb, hwyblas::Scalar<hwyblas::c64>(beta), c, ldc);
}
HWYBLAS_API void cblas_chemm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc) {
  hwyblas::Symm<hwyblas::c32>("cblas_chemm", true, layout, side, uplo, m, n,
                              hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, b,
                              ldb, hwyblas::Scalar<hwyblas::c32>(beta), c, ldc);
}
HWYBLAS_API void cblas_zhemm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* b, const hwyblas_int ldb,
                             const void* beta, void* c, const hwyblas_int ldc) {
  hwyblas::Symm<hwyblas::c64>("cblas_zhemm", true, layout, side, uplo, m, n,
                              hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, b,
                              ldb, hwyblas::Scalar<hwyblas::c64>(beta), c, ldc);
}
HWYBLAS_API void cblas_ssyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const float alpha,
                             const float* a, const hwyblas_int lda,
                             const float beta, float* c,
                             const hwyblas_int ldc) {
  hwyblas::Syrk<float, float>("cblas_ssyrk", layout, uplo, trans, n, k, alpha,
                              a, lda, beta, c, ldc);
}
HWYBLAS_API void cblas_dsyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const double alpha,
                             const double* a, const hwyblas_int lda,
                             const double beta, double* c,
                             const hwyblas_int ldc) {
  hwyblas::Syrk<double, double>("cblas_dsyrk", layout, uplo, trans, n, k, alpha,
                                a, lda, beta, c, ldc);
}
HWYBLAS_API void cblas_csyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* beta, void* c, const hwyblas_int ldc) {
  hwyblas::Syrk<hwyblas::c32, hwyblas::c32>(
      "cblas_csyrk", layout, uplo, trans, n, k,
      hwyblas::Scalar<hwyblas::c32>(alpha), a, lda,
      hwyblas::Scalar<hwyblas::c32>(beta), c, ldc);
}
HWYBLAS_API void cblas_zsyrk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const void* alpha,
                             const void* a, const hwyblas_int lda,
                             const void* beta, void* c, const hwyblas_int ldc) {
  hwyblas::Syrk<hwyblas::c64, hwyblas::c64>(
      "cblas_zsyrk", layout, uplo, trans, n, k,
      hwyblas::Scalar<hwyblas::c64>(alpha), a, lda,
      hwyblas::Scalar<hwyblas::c64>(beta), c, ldc);
}
HWYBLAS_API void cblas_cherk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const float alpha,
                             const void* a, const hwyblas_int lda,
                             const float beta, void* c, const hwyblas_int ldc) {
  hwyblas::Syrk<hwyblas::c32, float>("cblas_cherk", layout, uplo, trans, n, k,
                                     alpha, a, lda, beta, c, ldc);
}
HWYBLAS_API void cblas_zherk(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                             const hwyblas_int k, const double alpha,
                             const void* a, const hwyblas_int lda,
                             const double beta, void* c,
                             const hwyblas_int ldc) {
  hwyblas::Syrk<hwyblas::c64, double>("cblas_zherk", layout, uplo, trans, n, k,
                                      alpha, a, lda, beta, c, ldc);
}
HWYBLAS_API void cblas_ssyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const float alpha,
                              const float* a, const hwyblas_int lda,
                              const float* b, const hwyblas_int ldb,
                              const float beta, float* c,
                              const hwyblas_int ldc) {
  hwyblas::Syr2k<float, float>("cblas_ssyr2k", layout, uplo, trans, n, k, alpha,
                               a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_dsyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const double alpha,
                              const double* a, const hwyblas_int lda,
                              const double* b, const hwyblas_int ldb,
                              const double beta, double* c,
                              const hwyblas_int ldc) {
  hwyblas::Syr2k<double, double>("cblas_dsyr2k", layout, uplo, trans, n, k,
                                 alpha, a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_csyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c,
                              const hwyblas_int ldc) {
  hwyblas::Syr2k<hwyblas::c32, hwyblas::c32>(
      "cblas_csyr2k", layout, uplo, trans, n, k,
      hwyblas::Scalar<hwyblas::c32>(alpha), a, lda, b, ldb,
      hwyblas::Scalar<hwyblas::c32>(beta), c, ldc);
}
HWYBLAS_API void cblas_zsyr2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const void* beta, void* c,
                              const hwyblas_int ldc) {
  hwyblas::Syr2k<hwyblas::c64, hwyblas::c64>(
      "cblas_zsyr2k", layout, uplo, trans, n, k,
      hwyblas::Scalar<hwyblas::c64>(alpha), a, lda, b, ldb,
      hwyblas::Scalar<hwyblas::c64>(beta), c, ldc);
}
HWYBLAS_API void cblas_cher2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const float beta, void* c,
                              const hwyblas_int ldc) {
  hwyblas::Syr2k<hwyblas::c32, float>("cblas_cher2k", layout, uplo, trans, n, k,
                                      hwyblas::Scalar<hwyblas::c32>(alpha), a,
                                      lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_zher2k(const CBLAS_LAYOUT layout, const CBLAS_UPLO uplo,
                              const CBLAS_TRANSPOSE trans, const hwyblas_int n,
                              const hwyblas_int k, const void* alpha,
                              const void* a, const hwyblas_int lda,
                              const void* b, const hwyblas_int ldb,
                              const double beta, void* c,
                              const hwyblas_int ldc) {
  hwyblas::Syr2k<hwyblas::c64, double>("cblas_zher2k", layout, uplo, trans, n,
                                       k, hwyblas::Scalar<hwyblas::c64>(alpha),
                                       a, lda, b, ldb, beta, c, ldc);
}
HWYBLAS_API void cblas_strmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda, float* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<float>("cblas_strmm", false, layout, side, uplo, transa, diag,
                       m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void cblas_dtrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda, double* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<double>("cblas_dtrmm", false, layout, side, uplo, transa, diag,
                        m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void cblas_ctrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<hwyblas::c32>("cblas_ctrmm", false, layout, side, uplo, transa,
                              diag, m, n, hwyblas::Scalar<hwyblas::c32>(alpha),
                              a, lda, b, ldb);
}
HWYBLAS_API void cblas_ztrmm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<hwyblas::c64>("cblas_ztrmm", false, layout, side, uplo, transa,
                              diag, m, n, hwyblas::Scalar<hwyblas::c64>(alpha),
                              a, lda, b, ldb);
}
HWYBLAS_API void cblas_strsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const float alpha,
                             const float* a, const hwyblas_int lda, float* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<float>("cblas_strsm", true, layout, side, uplo, transa, diag, m,
                       n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void cblas_dtrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const double alpha,
                             const double* a, const hwyblas_int lda, double* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<double>("cblas_dtrsm", true, layout, side, uplo, transa, diag,
                        m, n, alpha, a, lda, b, ldb);
}
HWYBLAS_API void cblas_ctrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<hwyblas::c32>("cblas_ctrsm", true, layout, side, uplo, transa,
                              diag, m, n, hwyblas::Scalar<hwyblas::c32>(alpha),
                              a, lda, b, ldb);
}
HWYBLAS_API void cblas_ztrsm(const CBLAS_LAYOUT layout, const CBLAS_SIDE side,
                             const CBLAS_UPLO uplo,
                             const CBLAS_TRANSPOSE transa,
                             const CBLAS_DIAG diag, const hwyblas_int m,
                             const hwyblas_int n, const void* alpha,
                             const void* a, const hwyblas_int lda, void* b,
                             const hwyblas_int ldb) {
  hwyblas::Trmm<hwyblas::c64>("cblas_ztrsm", true, layout, side, uplo, transa,
                              diag, m, n, hwyblas::Scalar<hwyblas::c64>(alpha),
                              a, lda, b, ldb);
}

// Extensions (as in OpenBLAS)

HWYBLAS_API void cblas_saxpby(const hwyblas_int n, const float alpha,
                              const float* x, const hwyblas_int incx,
                              const float beta, float* y,
                              const hwyblas_int incy) {
  saxpby_(&n, &alpha, x, &incx, &beta, y, &incy);
}
HWYBLAS_API void cblas_daxpby(const hwyblas_int n, const double alpha,
                              const double* x, const hwyblas_int incx,
                              const double beta, double* y,
                              const hwyblas_int incy) {
  daxpby_(&n, &alpha, x, &incx, &beta, y, &incy);
}
HWYBLAS_API void cblas_caxpby(const hwyblas_int n, const void* alpha,
                              const void* x, const hwyblas_int incx,
                              const void* beta, void* y,
                              const hwyblas_int incy) {
  caxpby_(&n, hwyblas::F<hwyblas::c32>(alpha), hwyblas::F<hwyblas::c32>(x),
          &incx, hwyblas::F<hwyblas::c32>(beta), hwyblas::F<hwyblas::c32>(y),
          &incy);
}
HWYBLAS_API void cblas_zaxpby(const hwyblas_int n, const void* alpha,
                              const void* x, const hwyblas_int incx,
                              const void* beta, void* y,
                              const hwyblas_int incy) {
  zaxpby_(&n, hwyblas::F<hwyblas::c64>(alpha), hwyblas::F<hwyblas::c64>(x),
          &incx, hwyblas::F<hwyblas::c64>(beta), hwyblas::F<hwyblas::c64>(y),
          &incy);
}
HWYBLAS_API CBLAS_INDEX cblas_isamin(const hwyblas_int n, const float* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = isamin_(&n, x, &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API CBLAS_INDEX cblas_idamin(const hwyblas_int n, const double* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = idamin_(&n, x, &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API CBLAS_INDEX cblas_icamin(const hwyblas_int n, const void* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = icamin_(&n, hwyblas::F<hwyblas::c32>(x), &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API CBLAS_INDEX cblas_izamin(const hwyblas_int n, const void* x,
                                     const hwyblas_int incx) {
  const hwyblas_int i = izamin_(&n, hwyblas::F<hwyblas::c64>(x), &incx);
  return i > 0 ? static_cast<CBLAS_INDEX>(i - 1) : 0;
}
HWYBLAS_API void cblas_somatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const float alpha, const float* a,
                                 const hwyblas_int lda, float* b,
                                 const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_somatcopy", {1, 2, 3, 4, 5, 6, 7, 8, 9});
  somatcopy_(&order, &tc, &rows, &cols, &alpha, a, &lda, b, &ldb);
}
HWYBLAS_API void cblas_domatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const double alpha, const double* a,
                                 const hwyblas_int lda, double* b,
                                 const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_domatcopy", {1, 2, 3, 4, 5, 6, 7, 8, 9});
  domatcopy_(&order, &tc, &rows, &cols, &alpha, a, &lda, b, &ldb);
}
HWYBLAS_API void cblas_comatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const void* alpha, const void* a,
                                 const hwyblas_int lda, void* b,
                                 const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_comatcopy", {1, 2, 3, 4, 5, 6, 7, 8, 9});
  comatcopy_(&order, &tc, &rows, &cols, hwyblas::F<hwyblas::c32>(alpha),
             hwyblas::F<hwyblas::c32>(a), &lda, hwyblas::F<hwyblas::c32>(b),
             &ldb);
}
HWYBLAS_API void cblas_zomatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const void* alpha, const void* a,
                                 const hwyblas_int lda, void* b,
                                 const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_zomatcopy", {1, 2, 3, 4, 5, 6, 7, 8, 9});
  zomatcopy_(&order, &tc, &rows, &cols, hwyblas::F<hwyblas::c64>(alpha),
             hwyblas::F<hwyblas::c64>(a), &lda, hwyblas::F<hwyblas::c64>(b),
             &ldb);
}
HWYBLAS_API void cblas_simatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const float alpha, float* a,
                                 const hwyblas_int lda, const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_simatcopy", {1, 2, 3, 4, 5, 6, 7, 8});
  simatcopy_(&order, &tc, &rows, &cols, &alpha, a, &lda, &ldb);
}
HWYBLAS_API void cblas_dimatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const double alpha, double* a,
                                 const hwyblas_int lda, const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_dimatcopy", {1, 2, 3, 4, 5, 6, 7, 8});
  dimatcopy_(&order, &tc, &rows, &cols, &alpha, a, &lda, &ldb);
}
HWYBLAS_API void cblas_cimatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const void* alpha, void* a,
                                 const hwyblas_int lda, const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_cimatcopy", {1, 2, 3, 4, 5, 6, 7, 8});
  cimatcopy_(&order, &tc, &rows, &cols, hwyblas::F<hwyblas::c32>(alpha),
             hwyblas::F<hwyblas::c32>(a), &lda, &ldb);
}
HWYBLAS_API void cblas_zimatcopy(const CBLAS_LAYOUT layout,
                                 const CBLAS_TRANSPOSE trans,
                                 const hwyblas_int rows, const hwyblas_int cols,
                                 const void* alpha, void* a,
                                 const hwyblas_int lda, const hwyblas_int ldb) {
  const char order = hwyblas::MatcopyOrder(layout),
             tc = hwyblas::MatcopyTrans(trans);
  hwyblas::CblasScope scope("cblas_zimatcopy", {1, 2, 3, 4, 5, 6, 7, 8});
  zimatcopy_(&order, &tc, &rows, &cols, hwyblas::F<hwyblas::c64>(alpha),
             hwyblas::F<hwyblas::c64>(a), &lda, &ldb);
}

}  // extern "C"
