// Tridiagonal linear systems: general (the LU factorization with partial
// pivoting xGTTRF, the solvers xGTTRS and xGTTS2, the driver xGTSV) and
// symmetric (Hermitian) positive definite (the L*D*L**H factorization
// xPTTRF, the solvers xPTTRS and xPTTS2, the driver xPTSV).
//
// A general tridiagonal matrix is given by its subdiagonal DL (N-1
// elements), diagonal D (N) and superdiagonal DU (N-1); a positive definite
// one by its real diagonal D (N) and its subdiagonal E (N-1, complex for
// complex T).
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <cmath>

#include "hwyblas/lapack.h"
#include "src/lapack/blas.h"
#include "src/lapack/lapack_common.h"
#include "src/lapack/lapack_routines.h"

namespace hwyblas {
namespace lapack {

// xGTSV for real T (DGTSV / SGTSV); see Gtsv.
template <typename T>
blasint GtsvReal(blasint n, blasint nrhs, T* dl, T* d, T* du, T* b,
                 blasint ldb) {
  using R = T;
  const R zero = R(0);
  const Matrix<R> B(b, ldb);
  const Vector<R> D(d);
  const Vector<R> DL(dl);
  const Vector<R> DU(du);
  blasint i, j;
  R fact, temp;

  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (nrhs < 0) {
    info = -2;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<R>("GTSV", -info);
    return info;
  }

  if (n == 0) return info;

  if (nrhs == 1) {
    for (i = 1; i <= n - 2; ++i) {
      if (std::abs(D(i)) >= std::abs(DL(i))) {
        // No row interchange required.
        if (D(i) != zero) {
          fact = DL(i) / D(i);
          D(i + 1) = D(i + 1) - fact * DU(i);
          B(i + 1, 1) = B(i + 1, 1) - fact * B(i, 1);
        } else {
          info = i;
          return info;
        }
        DL(i) = zero;
      } else {
        // Interchange rows I and I+1.
        fact = D(i) / DL(i);
        D(i) = DL(i);
        temp = D(i + 1);
        D(i + 1) = DU(i) - fact * temp;
        DL(i) = DU(i + 1);
        DU(i + 1) = -fact * DL(i);
        DU(i) = temp;
        temp = B(i, 1);
        B(i, 1) = B(i + 1, 1);
        B(i + 1, 1) = temp - fact * B(i + 1, 1);
      }
    }
    if (n > 1) {
      i = n - 1;
      if (std::abs(D(i)) >= std::abs(DL(i))) {
        if (D(i) != zero) {
          fact = DL(i) / D(i);
          D(i + 1) = D(i + 1) - fact * DU(i);
          B(i + 1, 1) = B(i + 1, 1) - fact * B(i, 1);
        } else {
          info = i;
          return info;
        }
      } else {
        fact = D(i) / DL(i);
        D(i) = DL(i);
        temp = D(i + 1);
        D(i + 1) = DU(i) - fact * temp;
        DU(i) = temp;
        temp = B(i, 1);
        B(i, 1) = B(i + 1, 1);
        B(i + 1, 1) = temp - fact * B(i + 1, 1);
      }
    }
    if (D(n) == zero) {
      info = n;
      return info;
    }
  } else {
    for (i = 1; i <= n - 2; ++i) {
      if (std::abs(D(i)) >= std::abs(DL(i))) {
        // No row interchange required.
        if (D(i) != zero) {
          fact = DL(i) / D(i);
          D(i + 1) = D(i + 1) - fact * DU(i);
          for (j = 1; j <= nrhs; ++j) {
            B(i + 1, j) = B(i + 1, j) - fact * B(i, j);
          }
        } else {
          info = i;
          return info;
        }
        DL(i) = zero;
      } else {
        // Interchange rows I and I+1.
        fact = D(i) / DL(i);
        D(i) = DL(i);
        temp = D(i + 1);
        D(i + 1) = DU(i) - fact * temp;
        DL(i) = DU(i + 1);
        DU(i + 1) = -fact * DL(i);
        DU(i) = temp;
        for (j = 1; j <= nrhs; ++j) {
          temp = B(i, j);
          B(i, j) = B(i + 1, j);
          B(i + 1, j) = temp - fact * B(i + 1, j);
        }
      }
    }
    if (n > 1) {
      i = n - 1;
      if (std::abs(D(i)) >= std::abs(DL(i))) {
        if (D(i) != zero) {
          fact = DL(i) / D(i);
          D(i + 1) = D(i + 1) - fact * DU(i);
          for (j = 1; j <= nrhs; ++j) {
            B(i + 1, j) = B(i + 1, j) - fact * B(i, j);
          }
        } else {
          info = i;
          return info;
        }
      } else {
        fact = D(i) / DL(i);
        D(i) = DL(i);
        temp = D(i + 1);
        D(i + 1) = DU(i) - fact * temp;
        DU(i) = temp;
        for (j = 1; j <= nrhs; ++j) {
          temp = B(i, j);
          B(i, j) = B(i + 1, j);
          B(i + 1, j) = temp - fact * B(i + 1, j);
        }
      }
    }
    if (D(n) == zero) {
      info = n;
      return info;
    }
  }

  // Back solve with the matrix U from the factorization.
  if (nrhs <= 2) {
    j = 1;
  L70:
    B(n, j) = B(n, j) / D(n);
    if (n > 1) B(n - 1, j) = (B(n - 1, j) - DU(n - 1) * B(n, j)) / D(n - 1);
    for (i = n - 2; i >= 1; --i) {
      B(i, j) = (B(i, j) - DU(i) * B(i + 1, j) - DL(i) * B(i + 2, j)) / D(i);
    }
    if (j < nrhs) {
      j = j + 1;
      goto L70;
    }
  } else {
    for (j = 1; j <= nrhs; ++j) {
      B(n, j) = B(n, j) / D(n);
      if (n > 1) {
        B(n - 1, j) = (B(n - 1, j) - DU(n - 1) * B(n, j)) / D(n - 1);
      }
      for (i = n - 2; i >= 1; --i) {
        B(i, j) =
            (B(i, j) - DU(i) * B(i + 1, j) - DL(i) * B(i + 2, j)) / D(i);
      }
    }
  }
  return info;
}

// xGTSV for complex T (CGTSV / ZGTSV); see Gtsv. Unlike the real routine,
// it skips the elimination where DL(K) = 0 and treats all right-hand sides
// alike.
template <typename T>
blasint GtsvComplex(blasint n, blasint nrhs, T* dl, T* d, T* du, T* b,
                    blasint ldb) {
  const T zero = T(0);
  const Matrix<T> B(b, ldb);
  const Vector<T> D(d);
  const Vector<T> DL(dl);
  const Vector<T> DU(du);
  blasint j, k;
  T mult, temp;

  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (nrhs < 0) {
    info = -2;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("GTSV", -info);
    return info;
  }

  if (n == 0) return info;

  for (k = 1; k <= n - 1; ++k) {
    if (DL(k) == zero) {
      // Subdiagonal is zero, no elimination is required.
      if (D(k) == zero) {
        // Diagonal is zero: set INFO = K and return; a unique solution can
        // not be found.
        info = k;
        return info;
      }
    } else if (Cabs1(D(k)) >= Cabs1(DL(k))) {
      // No row interchange required.
      mult = DL(k) / D(k);
      D(k + 1) = D(k + 1) - mult * DU(k);
      for (j = 1; j <= nrhs; ++j) B(k + 1, j) = B(k + 1, j) - mult * B(k, j);
      if (k < (n - 1)) DL(k) = zero;
    } else {
      // Interchange rows K and K+1.
      mult = D(k) / DL(k);
      D(k) = DL(k);
      temp = D(k + 1);
      D(k + 1) = DU(k) - mult * temp;
      if (k < (n - 1)) {
        DL(k) = DU(k + 1);
        DU(k + 1) = -mult * DL(k);
      }
      DU(k) = temp;
      for (j = 1; j <= nrhs; ++j) {
        temp = B(k, j);
        B(k, j) = B(k + 1, j);
        B(k + 1, j) = temp - mult * B(k + 1, j);
      }
    }
  }
  if (D(n) == zero) {
    info = n;
    return info;
  }

  // Back solve with the matrix U from the factorization.
  for (j = 1; j <= nrhs; ++j) {
    B(n, j) = B(n, j) / D(n);
    if (n > 1) B(n - 1, j) = (B(n - 1, j) - DU(n - 1) * B(n, j)) / D(n - 1);
    for (k = n - 2; k >= 1; --k) {
      B(k, j) = (B(k, j) - DU(k) * B(k + 1, j) - DL(k) * B(k + 2, j)) / D(k);
    }
  }
  return info;
}

// xGTSV: solves A * X = B for the N-by-N tridiagonal matrix A (DL, D, DU)
// and the N-by-NRHS matrix B, overwritten with X, by Gaussian elimination
// with partial pivoting. On exit, D holds the diagonal of U, DU its first
// superdiagonal and DL (N-2 elements) its second superdiagonal (the real
// and the complex routine differ in how they get there, see GtsvComplex).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero (the solution has not been computed).
template <typename T>
blasint Gtsv(blasint n, blasint nrhs, T* dl, T* d, T* du, T* b, blasint ldb) {
  if constexpr (IsComplex<T>::value) {
    return GtsvComplex(n, nrhs, dl, d, du, b, ldb);
  } else {
    return GtsvReal(n, nrhs, dl, d, du, b, ldb);
  }
}

// xGTTRF: computes the LU factorization A = L * U of the N-by-N tridiagonal
// matrix A (DL, D, DU) with partial pivoting and row interchanges. On exit,
// DL holds the multipliers of L, D the diagonal of U, DU its first and DU2
// (N-2 elements) its second superdiagonal, and IPIV(i) the row interchanged
// with row i (i or i+1).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if U(i, i) is exactly zero (the factorization is completed).
template <typename T>
blasint Gttrf(blasint n, T* dl, T* d, T* du, T* du2, blasint* ipiv) {
  using R = RealT<T>;
  const R zero = R(0);
  const Vector<T> D(d);
  const Vector<T> DL(dl);
  const Vector<T> DU(du);
  const Vector<T> DU2(du2);
  const Vector<blasint> IPIV(ipiv);
  blasint i;
  T fact, temp;

  blasint info = 0;
  if (n < 0) {
    info = -1;
    Xerbla<T>("GTTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Initialize IPIV(i) = i and DU2(i) = 0.
  for (i = 1; i <= n; ++i) IPIV(i) = i;
  for (i = 1; i <= n - 2; ++i) DU2(i) = zero;

  // (The real routine compares |D(I)| and |DL(I)|, and D(I) with zero, which
  // Cabs1 computes for real T.)
  for (i = 1; i <= n - 2; ++i) {
    if (Cabs1(D(i)) >= Cabs1(DL(i))) {
      // No row interchange required, eliminate DL(I).
      if (Cabs1(D(i)) != zero) {
        fact = DL(i) / D(i);
        DL(i) = fact;
        D(i + 1) = D(i + 1) - fact * DU(i);
      }
    } else {
      // Interchange rows I and I+1, eliminate DL(I).
      fact = D(i) / DL(i);
      D(i) = DL(i);
      DL(i) = fact;
      temp = DU(i);
      DU(i) = D(i + 1);
      D(i + 1) = temp - fact * D(i + 1);
      DU2(i) = DU(i + 1);
      DU(i + 1) = -fact * DU(i + 1);
      IPIV(i) = i + 1;
    }
  }
  if (n > 1) {
    i = n - 1;
    if (Cabs1(D(i)) >= Cabs1(DL(i))) {
      if (Cabs1(D(i)) != zero) {
        fact = DL(i) / D(i);
        DL(i) = fact;
        D(i + 1) = D(i + 1) - fact * DU(i);
      }
    } else {
      fact = D(i) / DL(i);
      D(i) = DL(i);
      DL(i) = fact;
      temp = DU(i);
      DU(i) = D(i + 1);
      D(i + 1) = temp - fact * D(i + 1);
      IPIV(i) = i + 1;
    }
  }

  // Check for a zero on the diagonal of U.
  for (i = 1; i <= n; ++i) {
    if (Cabs1(D(i)) == zero) {
      info = i;
      break;
    }
  }
  return info;
}

// xGTTS2: solves A * X = B (ITRANS = 0), A**T * X = B (1) or A**H * X = B
// (2, complex T) for the N-by-NRHS matrix B, overwritten with X, with the
// LU factorization of the tridiagonal matrix A computed by xGTTRF, without
// checking the arguments.
//
// The real routine solves with L and L**T for a single right-hand side with
// IPIV(I) as an index instead of a branch; this computes the same.
template <typename T>
void Gtts2(blasint itrans, blasint n, blasint nrhs, const T* dl, const T* d,
           const T* du, const T* du2, const blasint* ipiv, T* b,
           blasint ldb) {
  const Matrix<T> B(b, ldb);
  const Vector<const T> D(d);
  const Vector<const T> DL(dl);
  const Vector<const T> DU(du);
  const Vector<const T> DU2(du2);
  const Vector<const blasint> IPIV(ipiv);
  blasint i, j;
  T temp;

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return;

  if (itrans == 0) {
    // Solve A*X = B using the LU factorization of A, overwriting each right
    // hand side vector with its solution.
    if (nrhs <= 1) {
      j = 1;
    L10:

      // Solve L*x = b.
      for (i = 1; i <= n - 1; ++i) {
        if (IPIV(i) == i) {
          B(i + 1, j) = B(i + 1, j) - DL(i) * B(i, j);
        } else {
          temp = B(i, j);
          B(i, j) = B(i + 1, j);
          B(i + 1, j) = temp - DL(i) * B(i, j);
        }
      }

      // Solve U*x = b.
      B(n, j) = B(n, j) / D(n);
      if (n > 1) B(n - 1, j) = (B(n - 1, j) - DU(n - 1) * B(n, j)) / D(n - 1);
      for (i = n - 2; i >= 1; --i) {
        B(i, j) =
            (B(i, j) - DU(i) * B(i + 1, j) - DU2(i) * B(i + 2, j)) / D(i);
      }
      if (j < nrhs) {
        j = j + 1;
        goto L10;
      }
    } else {
      for (j = 1; j <= nrhs; ++j) {
        // Solve L*x = b.
        for (i = 1; i <= n - 1; ++i) {
          if (IPIV(i) == i) {
            B(i + 1, j) = B(i + 1, j) - DL(i) * B(i, j);
          } else {
            temp = B(i, j);
            B(i, j) = B(i + 1, j);
            B(i + 1, j) = temp - DL(i) * B(i, j);
          }
        }

        // Solve U*x = b.
        B(n, j) = B(n, j) / D(n);
        if (n > 1) {
          B(n - 1, j) = (B(n - 1, j) - DU(n - 1) * B(n, j)) / D(n - 1);
        }
        for (i = n - 2; i >= 1; --i) {
          B(i, j) =
              (B(i, j) - DU(i) * B(i + 1, j) - DU2(i) * B(i + 2, j)) / D(i);
        }
      }
    }
  } else if (itrans == 1) {
    // Solve A**T * X = B.
    if (nrhs <= 1) {
      j = 1;
    L70:

      // Solve U**T * x = b.
      B(1, j) = B(1, j) / D(1);
      if (n > 1) B(2, j) = (B(2, j) - DU(1) * B(1, j)) / D(2);
      for (i = 3; i <= n; ++i) {
        B(i, j) = (B(i, j) - DU(i - 1) * B(i - 1, j) -
                   DU2(i - 2) * B(i - 2, j)) /
                  D(i);
      }

      // Solve L**T * x = b.
      for (i = n - 1; i >= 1; --i) {
        if (IPIV(i) == i) {
          B(i, j) = B(i, j) - DL(i) * B(i + 1, j);
        } else {
          temp = B(i + 1, j);
          B(i + 1, j) = B(i, j) - DL(i) * temp;
          B(i, j) = temp;
        }
      }
      if (j < nrhs) {
        j = j + 1;
        goto L70;
      }
    } else {
      for (j = 1; j <= nrhs; ++j) {
        // Solve U**T * x = b.
        B(1, j) = B(1, j) / D(1);
        if (n > 1) B(2, j) = (B(2, j) - DU(1) * B(1, j)) / D(2);
        for (i = 3; i <= n; ++i) {
          B(i, j) = (B(i, j) - DU(i - 1) * B(i - 1, j) -
                     DU2(i - 2) * B(i - 2, j)) /
                    D(i);
        }

        // Solve L**T * x = b.
        for (i = n - 1; i >= 1; --i) {
          if (IPIV(i) == i) {
            B(i, j) = B(i, j) - DL(i) * B(i + 1, j);
          } else {
            temp = B(i + 1, j);
            B(i + 1, j) = B(i, j) - DL(i) * temp;
            B(i, j) = temp;
          }
        }
      }
    }
  } else {
    // Solve A**H * X = B.
    if (nrhs <= 1) {
      j = 1;
    L130:

      // Solve U**H * x = b.
      B(1, j) = B(1, j) / Conj(D(1));
      if (n > 1) B(2, j) = (B(2, j) - Conj(DU(1)) * B(1, j)) / Conj(D(2));
      for (i = 3; i <= n; ++i) {
        B(i, j) = (B(i, j) - Conj(DU(i - 1)) * B(i - 1, j) -
                   Conj(DU2(i - 2)) * B(i - 2, j)) /
                  Conj(D(i));
      }

      // Solve L**H * x = b.
      for (i = n - 1; i >= 1; --i) {
        if (IPIV(i) == i) {
          B(i, j) = B(i, j) - Conj(DL(i)) * B(i + 1, j);
        } else {
          temp = B(i + 1, j);
          B(i + 1, j) = B(i, j) - Conj(DL(i)) * temp;
          B(i, j) = temp;
        }
      }
      if (j < nrhs) {
        j = j + 1;
        goto L130;
      }
    } else {
      for (j = 1; j <= nrhs; ++j) {
        // Solve U**H * x = b.
        B(1, j) = B(1, j) / Conj(D(1));
        if (n > 1) B(2, j) = (B(2, j) - Conj(DU(1)) * B(1, j)) / Conj(D(2));
        for (i = 3; i <= n; ++i) {
          B(i, j) = (B(i, j) - Conj(DU(i - 1)) * B(i - 1, j) -
                     Conj(DU2(i - 2)) * B(i - 2, j)) /
                    Conj(D(i));
        }

        // Solve L**H * x = b.
        for (i = n - 1; i >= 1; --i) {
          if (IPIV(i) == i) {
            B(i, j) = B(i, j) - Conj(DL(i)) * B(i + 1, j);
          } else {
            temp = B(i + 1, j);
            B(i + 1, j) = B(i, j) - Conj(DL(i)) * temp;
            B(i, j) = temp;
          }
        }
      }
    }
  }
}

// xGTTRS: solves A * X = B, A**T * X = B or A**H * X = B (TRANS = 'N', 'T'
// or 'C') for the N-by-NRHS matrix B, overwritten with X, with the LU
// factorization of the tridiagonal matrix A computed by xGTTRF, in blocks
// of NB right-hand sides (ILAENV) with xGTTS2.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Gttrs(char trans, blasint n, blasint nrhs, const T* dl, const T* d,
              const T* du, const T* du2, const blasint* ipiv, T* b,
              blasint ldb) {
  const Matrix<T> B(b, ldb);
  blasint itrans, j, jb, nb;
  bool notran;

  blasint info = 0;
  notran = (trans == 'N' || trans == 'n');
  // NOLINTBEGIN(readability-simplify-boolean-expr): as LAPACK
  if (!notran && !(trans == 'T' || trans == 't') &&
      !(trans == 'C' || trans == 'c')) {
    // NOLINTEND(readability-simplify-boolean-expr)
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (ldb < Max<blasint>(n, 1)) {
    info = -10;
  }
  if (info != 0) {
    Xerbla<T>("GTTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  // Decode TRANS. (The real routine solves with A**T also for TRANS = 'C'.)
  if (notran) {
    itrans = 0;
  } else if (!IsComplex<T>::value || trans == 'T' || trans == 't') {
    itrans = 1;
  } else {
    itrans = 2;
  }

  // Determine the number of right-hand sides to solve at a time.
  if (nrhs == 1) {
    nb = 1;
  } else {
    const char opts[2] = {trans, '\0'};
    nb = Max<blasint>(1, Ilaenv<T>(1, "GTTRS", opts, n, nrhs, -1, -1));
  }

  if (nb >= nrhs) {
    Gtts2(itrans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
  } else {
    for (j = 1; j <= nrhs; j += nb) {
      jb = Min(nrhs - j + 1, nb);
      Gtts2(itrans, n, jb, dl, d, du, du2, ipiv, B.ptr(1, j), ldb);
    }
  }
  return info;
}

// xPTTRF for real T (DPTTRF / SPTTRF); see Pttrf.
template <typename T>
blasint PttrfReal(blasint n, T* d, T* e) {
  using R = T;
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<R> E(e);
  blasint i, i4;
  R ei;

  blasint info = 0;
  if (n < 0) {
    info = -1;
    Xerbla<R>("PTTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Compute the L*D*L**T (or U**T*D*U) factorization of A.
  i4 = (n - 1) % 4;
  for (i = 1; i <= i4; ++i) {
    if (D(i) <= zero) {
      info = i;
      return info;
    }
    ei = E(i);
    E(i) = ei / D(i);
    D(i + 1) = D(i + 1) - E(i) * ei;
  }

  for (i = i4 + 1; i <= n - 4; i += 4) {
    // Drop out of the loop if d(i) <= 0: the matrix is not positive
    // definite.
    if (D(i) <= zero) {
      info = i;
      return info;
    }

    // Solve for e(i) and d(i+1).
    ei = E(i);
    E(i) = ei / D(i);
    D(i + 1) = D(i + 1) - E(i) * ei;

    if (D(i + 1) <= zero) {
      info = i + 1;
      return info;
    }

    // Solve for e(i+1) and d(i+2).
    ei = E(i + 1);
    E(i + 1) = ei / D(i + 1);
    D(i + 2) = D(i + 2) - E(i + 1) * ei;

    if (D(i + 2) <= zero) {
      info = i + 2;
      return info;
    }

    // Solve for e(i+2) and d(i+3).
    ei = E(i + 2);
    E(i + 2) = ei / D(i + 2);
    D(i + 3) = D(i + 3) - E(i + 2) * ei;

    if (D(i + 3) <= zero) {
      info = i + 3;
      return info;
    }

    // Solve for e(i+3) and d(i+4).
    ei = E(i + 3);
    E(i + 3) = ei / D(i + 3);
    D(i + 4) = D(i + 4) - E(i + 3) * ei;
  }

  // Check d(n) for positive definiteness.
  if (D(n) <= zero) info = n;
  return info;
}

// xPTTRF for complex T (CPTTRF / ZPTTRF); see Pttrf. It updates D with the
// real and imaginary parts of E separately.
template <typename T>
blasint PttrfComplex(blasint n, RealT<T>* d, T* e) {
  using R = RealT<T>;
  const R zero = R(0);
  const Vector<R> D(d);
  const Vector<T> E(e);
  blasint i, i4;
  R eii, eir, f, g;

  blasint info = 0;
  if (n < 0) {
    info = -1;
    Xerbla<T>("PTTRF", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0) return info;

  // Compute the L*D*L**H (or U**H *D*U) factorization of A.
  i4 = (n - 1) % 4;
  for (i = 1; i <= i4; ++i) {
    if (D(i) <= zero) {
      info = i;
      return info;
    }
    eir = std::real(E(i));
    eii = std::imag(E(i));
    f = eir / D(i);
    g = eii / D(i);
    E(i) = T(f, g);
    D(i + 1) = D(i + 1) - f * eir - g * eii;
  }

  for (i = i4 + 1; i <= n - 4; i += 4) {
    // Drop out of the loop if d(i) <= 0: the matrix is not positive
    // definite.
    if (D(i) <= zero) {
      info = i;
      return info;
    }

    // Solve for e(i) and d(i+1).
    eir = std::real(E(i));
    eii = std::imag(E(i));
    f = eir / D(i);
    g = eii / D(i);
    E(i) = T(f, g);
    D(i + 1) = D(i + 1) - f * eir - g * eii;

    if (D(i + 1) <= zero) {
      info = i + 1;
      return info;
    }

    // Solve for e(i+1) and d(i+2).
    eir = std::real(E(i + 1));
    eii = std::imag(E(i + 1));
    f = eir / D(i + 1);
    g = eii / D(i + 1);
    E(i + 1) = T(f, g);
    D(i + 2) = D(i + 2) - f * eir - g * eii;

    if (D(i + 2) <= zero) {
      info = i + 2;
      return info;
    }

    // Solve for e(i+2) and d(i+3).
    eir = std::real(E(i + 2));
    eii = std::imag(E(i + 2));
    f = eir / D(i + 2);
    g = eii / D(i + 2);
    E(i + 2) = T(f, g);
    D(i + 3) = D(i + 3) - f * eir - g * eii;

    if (D(i + 3) <= zero) {
      info = i + 3;
      return info;
    }

    // Solve for e(i+3) and d(i+4).
    eir = std::real(E(i + 3));
    eii = std::imag(E(i + 3));
    f = eir / D(i + 3);
    g = eii / D(i + 3);
    E(i + 3) = T(f, g);
    D(i + 4) = D(i + 4) - f * eir - g * eii;
  }

  // Check d(n) for positive definiteness.
  if (D(n) <= zero) info = n;
  return info;
}

// xPTTRF: computes the factorization A = L * D * L**H of the symmetric
// (Hermitian) positive definite tridiagonal matrix A with diagonal D and
// subdiagonal E (overwritten with D and the subdiagonal of the unit lower
// bidiagonal L; A = U**H * D * U with U = L**H as well).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the leading principal minor of order i is not positive (i < N:
// the factorization could not be completed; i = N: it was completed, but
// D(N) <= 0).
template <typename T>
blasint Pttrf(blasint n, RealT<T>* d, T* e) {
  if constexpr (IsComplex<T>::value) {
    return PttrfComplex(n, d, e);
  } else {
    return PttrfReal(n, d, e);
  }
}

// DPTTS2 / SPTTS2: solves A * X = B for the N-by-NRHS matrix B,
// overwritten with X, with the factorization A = L * D * L**T of the real
// symmetric positive definite tridiagonal matrix A computed by xPTTRF,
// without checking the arguments. Real T only.
template <typename T>
void Ptts2(blasint n, blasint nrhs, const T* d, const T* e, T* b,
           blasint ldb) {
  using R = T;
  const Matrix<R> B(b, ldb);
  const Vector<const R> D(d);
  const Vector<const R> E(e);
  blasint i, j;

  // Quick return if possible.
  if (n <= 1) {
    if (n == 1) Scal(nrhs, R(1) / D(1), b, ldb);
    return;
  }

  // Solve A * X = B using the factorization A = L*D*L**T, overwriting each
  // right hand side vector with its solution.
  for (j = 1; j <= nrhs; ++j) {
    // Solve L * x = b.
    for (i = 2; i <= n; ++i) B(i, j) = B(i, j) - B(i - 1, j) * E(i - 1);

    // Solve D * L**T * x = b.
    B(n, j) = B(n, j) / D(n);
    for (i = n - 1; i >= 1; --i) {
      B(i, j) = B(i, j) / D(i) - B(i + 1, j) * E(i);
    }
  }
}

// CPTTS2 / ZPTTS2: solves A * X = B as the real routine, with the
// factorization A = U**H * D * U (IUPLO = 1) or A = L * D * L**H (IUPLO =
// 0) of the Hermitian positive definite tridiagonal matrix A computed by
// xPTTRF. Complex T only.
template <typename T>
void Ptts2(blasint iuplo, blasint n, blasint nrhs, const RealT<T>* d,
           const T* e, T* b, blasint ldb) {
  using R = RealT<T>;
  const Matrix<T> B(b, ldb);
  const Vector<const R> D(d);
  const Vector<const T> E(e);
  blasint i, j;

  // Quick return if possible.
  if (n <= 1) {
    if (n == 1) Rscal(nrhs, R(1) / D(1), b, ldb);
    return;
  }

  if (iuplo == 1) {
    // Solve A * X = B using the factorization A = U**H *D*U, overwriting
    // each right hand side vector with its solution.
    if (nrhs <= 2) {
      j = 1;
    L10:

      // Solve U**H * x = b.
      for (i = 2; i <= n; ++i) {
        B(i, j) = B(i, j) - B(i - 1, j) * std::conj(E(i - 1));
      }

      // Solve D * U * x = b.
      for (i = 1; i <= n; ++i) B(i, j) = B(i, j) / D(i);
      for (i = n - 1; i >= 1; --i) B(i, j) = B(i, j) - B(i + 1, j) * E(i);
      if (j < nrhs) {
        j = j + 1;
        goto L10;
      }
    } else {
      for (j = 1; j <= nrhs; ++j) {
        // Solve U**H * x = b.
        for (i = 2; i <= n; ++i) {
          B(i, j) = B(i, j) - B(i - 1, j) * std::conj(E(i - 1));
        }

        // Solve D * U * x = b.
        B(n, j) = B(n, j) / D(n);
        for (i = n - 1; i >= 1; --i) {
          B(i, j) = B(i, j) / D(i) - B(i + 1, j) * E(i);
        }
      }
    }
  } else {
    // Solve A * X = B using the factorization A = L*D*L**H, overwriting
    // each right hand side vector with its solution.
    if (nrhs <= 2) {
      j = 1;
    L80:

      // Solve L * x = b.
      for (i = 2; i <= n; ++i) B(i, j) = B(i, j) - B(i - 1, j) * E(i - 1);

      // Solve D * L**H * x = b.
      for (i = 1; i <= n; ++i) B(i, j) = B(i, j) / D(i);
      for (i = n - 1; i >= 1; --i) {
        B(i, j) = B(i, j) - B(i + 1, j) * std::conj(E(i));
      }
      if (j < nrhs) {
        j = j + 1;
        goto L80;
      }
    } else {
      for (j = 1; j <= nrhs; ++j) {
        // Solve L * x = b.
        for (i = 2; i <= n; ++i) B(i, j) = B(i, j) - B(i - 1, j) * E(i - 1);

        // Solve D * L**H * x = b.
        B(n, j) = B(n, j) / D(n);
        for (i = n - 1; i >= 1; --i) {
          B(i, j) = B(i, j) / D(i) - B(i + 1, j) * std::conj(E(i));
        }
      }
    }
  }
}

// DPTTRS / SPTTRS: solves A * X = B for the N-by-NRHS matrix B, overwritten
// with X, with the factorization of the real symmetric positive definite
// tridiagonal matrix A computed by xPTTRF, in blocks of NB right-hand
// sides (ILAENV) with xPTTS2. Real T only.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Pttrs(blasint n, blasint nrhs, const T* d, const T* e, T* b,
              blasint ldb) {
  const Matrix<T> B(b, ldb);
  blasint j, jb, nb;

  // Test the input arguments.
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (nrhs < 0) {
    info = -2;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("PTTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  // Determine the number of right-hand sides to solve at a time.
  if (nrhs == 1) {
    nb = 1;
  } else {
    nb = Max<blasint>(1, Ilaenv<T>(1, "PTTRS", " ", n, nrhs, -1, -1));
  }

  if (nb >= nrhs) {
    Ptts2(n, nrhs, d, e, b, ldb);
  } else {
    for (j = 1; j <= nrhs; j += nb) {
      jb = Min(nrhs - j + 1, nb);
      Ptts2(n, jb, d, e, B.ptr(1, j), ldb);
    }
  }
  return info;
}

// CPTTRS / ZPTTRS: solves A * X = B as the real routine, with the
// factorization A = U**H * D * U (UPLO = 'U') or A = L * D * L**H ('L') of
// the Hermitian positive definite tridiagonal matrix A computed by xPTTRF
// (E is the superdiagonal of U or the subdiagonal of L). Complex T only.
//
// Returns INFO: 0 on success, or -i if argument i had an illegal value.
template <typename T>
blasint Pttrs(char uplo, blasint n, blasint nrhs, const RealT<T>* d,
              const T* e, T* b, blasint ldb) {
  const Matrix<T> B(b, ldb);
  blasint iuplo, j, jb, nb;
  bool upper;

  // Test the input arguments.
  blasint info = 0;
  upper = (uplo == 'U' || uplo == 'u');
  // NOLINTNEXTLINE(readability-simplify-boolean-expr): as LAPACK
  if (!upper && !(uplo == 'L' || uplo == 'l')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (nrhs < 0) {
    info = -3;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -7;
  }
  if (info != 0) {
    Xerbla<T>("PTTRS", -info);
    return info;
  }

  // Quick return if possible.
  if (n == 0 || nrhs == 0) return info;

  // Determine the number of right-hand sides to solve at a time.
  if (nrhs == 1) {
    nb = 1;
  } else {
    const char opts[2] = {uplo, '\0'};
    nb = Max<blasint>(1, Ilaenv<T>(1, "PTTRS", opts, n, nrhs, -1, -1));
  }

  // Decode UPLO.
  if (upper) {
    iuplo = 1;
  } else {
    iuplo = 0;
  }

  if (nb >= nrhs) {
    Ptts2(iuplo, n, nrhs, d, e, b, ldb);
  } else {
    for (j = 1; j <= nrhs; j += nb) {
      jb = Min(nrhs - j + 1, nb);
      Ptts2(iuplo, n, jb, d, e, B.ptr(1, j), ldb);
    }
  }
  return info;
}

// xPTSV: solves A * X = B for the symmetric (Hermitian) positive definite
// tridiagonal matrix A (diagonal D, subdiagonal E; overwritten with its
// factorization A = L * D * L**H by xPTTRF) and the N-by-NRHS matrix B,
// overwritten with X.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if the leading principal minor of order i is not positive (the
// solution has not been computed).
template <typename T>
blasint Ptsv(blasint n, blasint nrhs, RealT<T>* d, T* e, T* b, blasint ldb) {
  // Test the input parameters.
  blasint info = 0;
  if (n < 0) {
    info = -1;
  } else if (nrhs < 0) {
    info = -2;
  } else if (ldb < Max<blasint>(1, n)) {
    info = -6;
  }
  if (info != 0) {
    Xerbla<T>("PTSV", -info);
    return info;
  }

  // Compute the L*D*L**H (or U**H*D*U) factorization of A.
  info = Pttrf(n, d, e);
  if (info == 0) {
    // Solve the system A*X = B, overwriting B with X.
    if constexpr (IsComplex<T>::value) {
      info = Pttrs('L', n, nrhs, d, e, b, ldb);
    } else {
      info = Pttrs(n, nrhs, d, e, b, ldb);
    }
  }
  return info;
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                         \
  template blasint Gtsv<T>(blasint, blasint, T*, T*, T*, T*, blasint);        \
  template blasint Gttrf<T>(blasint, T*, T*, T*, T*, blasint*);               \
  template void Gtts2<T>(blasint, blasint, blasint, const T*, const T*,       \
                         const T*, const T*, const blasint*, T*, blasint);    \
  template blasint Gttrs<T>(char, blasint, blasint, const T*, const T*,       \
                            const T*, const T*, const blasint*, T*, blasint); \
  template blasint Pttrf<T>(blasint, RealT<T>*, T*);                          \
  template blasint Ptsv<T>(blasint, blasint, RealT<T>*, T*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

#define HWYBLAS_LAPACK_INSTANTIATE_REAL(T)                                     \
  template void Ptts2<T>(blasint, blasint, const T*, const T*, T*, blasint);   \
  template blasint Pttrs<T>(blasint, blasint, const T*, const T*, T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_REAL(float)
HWYBLAS_LAPACK_INSTANTIATE_REAL(double)
#undef HWYBLAS_LAPACK_INSTANTIATE_REAL

#define HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(T)                                \
  template void Ptts2<T>(blasint, blasint, blasint, const RealT<T>*,         \
                         const T*, T*, blasint);                             \
  template blasint Pttrs<T>(char, blasint, blasint, const RealT<T>*,         \
                            const T*, T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c32)
HWYBLAS_LAPACK_INSTANTIATE_COMPLEX(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE_COMPLEX

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_GTSV(prefix, T, R)                                      \
  HWYBLAS_API void prefix##gtsv_(const hwyblas_int* n,                         \
                                 const hwyblas_int* nrhs, R* dl, R* d, R* du,  \
                                 R* b, const hwyblas_int* ldb,                 \
                                 hwyblas_int* info) {                          \
    *info = hwyblas::lapack::Gtsv(*n, *nrhs, As<T>(dl), As<T>(d), As<T>(du),   \
                                  As<T>(b), *ldb);                             \
  }                                                                            \
  HWYBLAS_API void prefix##gttrf_(const hwyblas_int* n, R* dl, R* d, R* du,    \
                                  R* du2, hwyblas_int* ipiv,                   \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Gttrf(*n, As<T>(dl), As<T>(d), As<T>(du),         \
                                   As<T>(du2), ipiv);                          \
  }                                                                            \
  HWYBLAS_API void prefix##gtts2_(                                             \
      const hwyblas_int* itrans, const hwyblas_int* n,                         \
      const hwyblas_int* nrhs, const R* dl, const R* d, const R* du,           \
      const R* du2, const hwyblas_int* ipiv, R* b, const hwyblas_int* ldb) {   \
    hwyblas::lapack::Gtts2(*itrans, *n, *nrhs, As<const T>(dl),                \
                           As<const T>(d), As<const T>(du), As<const T>(du2),  \
                           ipiv, As<T>(b), *ldb);                              \
  }                                                                            \
  HWYBLAS_API void prefix##gttrs_(                                             \
      const char* trans, const hwyblas_int* n, const hwyblas_int* nrhs,        \
      const R* dl, const R* d, const R* du, const R* du2,                      \
      const hwyblas_int* ipiv, R* b, const hwyblas_int* ldb,                   \
      hwyblas_int* info) {                                                     \
    *info = hwyblas::lapack::Gttrs(*trans, *n, *nrhs, As<const T>(dl),         \
                                   As<const T>(d), As<const T>(du),            \
                                   As<const T>(du2), ipiv, As<T>(b), *ldb);    \
  }                                                                            \
  HWYBLAS_API void prefix##pttrf_(const hwyblas_int* n, R* d, R* e,            \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Pttrf(*n, d, As<T>(e));                           \
  }                                                                            \
  HWYBLAS_API void prefix##ptsv_(const hwyblas_int* n,                         \
                                 const hwyblas_int* nrhs, R* d, R* e, R* b,    \
                                 const hwyblas_int* ldb, hwyblas_int* info) {  \
    *info =                                                                    \
        hwyblas::lapack::Ptsv(*n, *nrhs, d, As<T>(e), As<T>(b), *ldb);         \
  }

HWYBLAS_LAPACK_GTSV(s, float, float)
HWYBLAS_LAPACK_GTSV(d, double, double)
HWYBLAS_LAPACK_GTSV(c, c32, float)
HWYBLAS_LAPACK_GTSV(z, c64, double)

#undef HWYBLAS_LAPACK_GTSV

// xPTTRS and xPTTS2 take UPLO / IUPLO for complex T only.
#define HWYBLAS_LAPACK_PTTRS_REAL(prefix, R)                                   \
  HWYBLAS_API void prefix##pttrs_(const hwyblas_int* n,                        \
                                  const hwyblas_int* nrhs, const R* d,         \
                                  const R* e, R* b, const hwyblas_int* ldb,    \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Pttrs(*n, *nrhs, d, e, b, *ldb);                  \
  }                                                                            \
  HWYBLAS_API void prefix##ptts2_(const hwyblas_int* n,                        \
                                  const hwyblas_int* nrhs, const R* d,         \
                                  const R* e, R* b, const hwyblas_int* ldb) {  \
    hwyblas::lapack::Ptts2(*n, *nrhs, d, e, b, *ldb);                          \
  }

#define HWYBLAS_LAPACK_PTTRS_COMPLEX(prefix, T, R)                             \
  HWYBLAS_API void prefix##pttrs_(const char* uplo, const hwyblas_int* n,      \
                                  const hwyblas_int* nrhs, const R* d,         \
                                  const R* e, R* b, const hwyblas_int* ldb,    \
                                  hwyblas_int* info) {                         \
    *info = hwyblas::lapack::Pttrs(*uplo, *n, *nrhs, d, As<const T>(e),        \
                                   As<T>(b), *ldb);                            \
  }                                                                            \
  HWYBLAS_API void prefix##ptts2_(const hwyblas_int* iuplo,                    \
                                  const hwyblas_int* n,                        \
                                  const hwyblas_int* nrhs, const R* d,         \
                                  const R* e, R* b, const hwyblas_int* ldb) {  \
    hwyblas::lapack::Ptts2(*iuplo, *n, *nrhs, d, As<const T>(e), As<T>(b),     \
                           *ldb);                                              \
  }

HWYBLAS_LAPACK_PTTRS_REAL(s, float)
HWYBLAS_LAPACK_PTTRS_REAL(d, double)
HWYBLAS_LAPACK_PTTRS_COMPLEX(c, c32, float)
HWYBLAS_LAPACK_PTTRS_COMPLEX(z, c64, double)

#undef HWYBLAS_LAPACK_PTTRS_REAL
#undef HWYBLAS_LAPACK_PTTRS_COMPLEX

}  // extern "C"
