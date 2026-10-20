// The Bunch-Kaufman factorization of symmetric indefinite matrices,
// A = U * D * U**T or L * D * L**T, with xSYTRF (blocked, with the panel
// factorization xLASYF) and xSYTF2 (unblocked), and of Hermitian indefinite
// matrices, A = U * D * U**H or L * D * L**H, with xHETRF, xLAHEF and
// xHETF2 (complex T only). D is block diagonal with 1-by-1 and 2-by-2
// blocks; IPIV describes the blocks and the interchanges. Also LAPACK's
// complex symmetric rank-1 update and matrix-vector product CSYR / ZSYR and
// CSYMV / ZSYMV (the BLAS has only the real and Hermitian ones).
//
// The real routines (xSYTF2, xLASYF, xSYTRF) differ from the complex
// symmetric ones only in |.| against CABS1 (Cabs1 is |.| for real T); one
// template serves both. The Hermitian routines are separate templates.
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

// CSYMV / ZSYMV (complex T): y := alpha * A * x + beta * y for the complex
// symmetric matrix A of order N (triangle UPLO), like the BLAS's xSYMV.
// For real T, this calls the BLAS's xSYMV.
template <typename T>
void Symv(char uplo, blasint n, T alpha, const T* a, blasint lda, const T* x,
          blasint incx, T beta, T* y, blasint incy) {
  if constexpr (!IsComplex<T>::value) {
    Hemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
  } else {
    const T one = T(1), zero = T(0);
    const Matrix<const T> A(a, lda);
    const Vector<const T> X(x);
    const Vector<T> Y(y);

    // Test the input parameters.
    blasint info = 0;
    if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
      info = 1;
    } else if (n < 0) {
      info = 2;
    } else if (lda < Max<blasint>(1, n)) {
      info = 5;
    } else if (incx == 0) {
      info = 7;
    } else if (incy == 0) {
      info = 10;
    }
    if (info != 0) {
      Xerbla<T>("SYMV", info);
      return;
    }

    // Quick return if possible.
    if (n == 0 || (alpha == zero && beta == one)) return;

    // Set up the start points in X and Y.
    const blasint kx = incx > 0 ? 1 : 1 - (n - 1) * incx;
    const blasint ky = incy > 0 ? 1 : 1 - (n - 1) * incy;

    // Start the operations. In this version the elements of A are accessed
    // sequentially with one pass through the triangular part of A.
    //
    // First form y := beta*y.
    if (beta != one) {
      if (incy == 1) {
        if (beta == zero) {
          for (blasint i = 1; i <= n; ++i) Y(i) = zero;
        } else {
          for (blasint i = 1; i <= n; ++i) Y(i) = beta * Y(i);
        }
      } else {
        blasint iy = ky;
        if (beta == zero) {
          for (blasint i = 1; i <= n; ++i) {
            Y(iy) = zero;
            iy = iy + incy;
          }
        } else {
          for (blasint i = 1; i <= n; ++i) {
            Y(iy) = beta * Y(iy);
            iy = iy + incy;
          }
        }
      }
    }
    if (alpha == zero) return;
    if (Lsame(uplo, 'U')) {
      // Form y when A is stored in upper triangle.
      if (incx == 1 && incy == 1) {
        for (blasint j = 1; j <= n; ++j) {
          const T temp1 = alpha * X(j);
          T temp2 = zero;
          for (blasint i = 1; i <= j - 1; ++i) {
            Y(i) = Y(i) + temp1 * A(i, j);
            temp2 = temp2 + A(i, j) * X(i);
          }
          Y(j) = Y(j) + temp1 * A(j, j) + alpha * temp2;
        }
      } else {
        blasint jx = kx;
        blasint jy = ky;
        for (blasint j = 1; j <= n; ++j) {
          const T temp1 = alpha * X(jx);
          T temp2 = zero;
          blasint ix = kx;
          blasint iy = ky;
          for (blasint i = 1; i <= j - 1; ++i) {
            Y(iy) = Y(iy) + temp1 * A(i, j);
            temp2 = temp2 + A(i, j) * X(ix);
            ix = ix + incx;
            iy = iy + incy;
          }
          Y(jy) = Y(jy) + temp1 * A(j, j) + alpha * temp2;
          jx = jx + incx;
          jy = jy + incy;
        }
      }
    } else {
      // Form y when A is stored in lower triangle.
      if (incx == 1 && incy == 1) {
        for (blasint j = 1; j <= n; ++j) {
          const T temp1 = alpha * X(j);
          T temp2 = zero;
          Y(j) = Y(j) + temp1 * A(j, j);
          for (blasint i = j + 1; i <= n; ++i) {
            Y(i) = Y(i) + temp1 * A(i, j);
            temp2 = temp2 + A(i, j) * X(i);
          }
          Y(j) = Y(j) + alpha * temp2;
        }
      } else {
        blasint jx = kx;
        blasint jy = ky;
        for (blasint j = 1; j <= n; ++j) {
          const T temp1 = alpha * X(jx);
          T temp2 = zero;
          Y(jy) = Y(jy) + temp1 * A(j, j);
          blasint ix = jx;
          blasint iy = jy;
          for (blasint i = j + 1; i <= n; ++i) {
            ix = ix + incx;
            iy = iy + incy;
            Y(iy) = Y(iy) + temp1 * A(i, j);
            temp2 = temp2 + A(i, j) * X(ix);
          }
          Y(jy) = Y(jy) + alpha * temp2;
          jx = jx + incx;
          jy = jy + incy;
        }
      }
    }
  }
}

// CSYR / ZSYR (complex T): A := alpha * x * x**T + A for the complex
// symmetric matrix A of order N (triangle UPLO), like the BLAS's xSYR. For
// real T, this calls the BLAS's xSYR.
template <typename T>
void Syr(char uplo, blasint n, T alpha, const T* x, blasint incx, T* a,
         blasint lda) {
  if constexpr (!IsComplex<T>::value) {
    Her(uplo, n, alpha, x, incx, a, lda);
  } else {
    const T zero = T(0);
    const Matrix<T> A(a, lda);
    const Vector<const T> X(x);

    // Test the input parameters.
    blasint info = 0;
    if (!Lsame(uplo, 'U') && !Lsame(uplo, 'L')) {
      info = 1;
    } else if (n < 0) {
      info = 2;
    } else if (incx == 0) {
      info = 5;
    } else if (lda < Max<blasint>(1, n)) {
      info = 7;
    }
    if (info != 0) {
      Xerbla<T>("SYR", info);
      return;
    }

    // Quick return if possible.
    if (n == 0 || alpha == zero) return;

    // Set the start point in X if the increment is not unity. (KX is used
    // only then.)
    const blasint kx = incx <= 0 ? 1 - (n - 1) * incx : 1;

    // Start the operations. In this version the elements of A are accessed
    // sequentially with one pass through the triangular part of A.
    if (Lsame(uplo, 'U')) {
      // Form A when A is stored in upper triangle.
      if (incx == 1) {
        for (blasint j = 1; j <= n; ++j) {
          if (X(j) != zero) {
            const T temp = alpha * X(j);
            for (blasint i = 1; i <= j; ++i) A(i, j) = A(i, j) + X(i) * temp;
          }
        }
      } else {
        blasint jx = kx;
        for (blasint j = 1; j <= n; ++j) {
          if (X(jx) != zero) {
            const T temp = alpha * X(jx);
            blasint ix = kx;
            for (blasint i = 1; i <= j; ++i) {
              A(i, j) = A(i, j) + X(ix) * temp;
              ix = ix + incx;
            }
          }
          jx = jx + incx;
        }
      }
    } else {
      // Form A when A is stored in lower triangle.
      if (incx == 1) {
        for (blasint j = 1; j <= n; ++j) {
          if (X(j) != zero) {
            const T temp = alpha * X(j);
            for (blasint i = j; i <= n; ++i) A(i, j) = A(i, j) + X(i) * temp;
          }
        }
      } else {
        blasint jx = kx;
        for (blasint j = 1; j <= n; ++j) {
          if (X(jx) != zero) {
            const T temp = alpha * X(jx);
            blasint ix = jx;
            for (blasint i = j; i <= n; ++i) {
              A(i, j) = A(i, j) + X(ix) * temp;
              ix = ix + incx;
            }
          }
          jx = jx + incx;
        }
      }
    }
  }
}

// xSYTF2: computes the factorization A = U * D * U**T (UPLO = 'U') or
// A = L * D * L**T ('L') of the symmetric matrix A of order N with the
// Bunch-Kaufman diagonal pivoting method, unblocked (Level 2 BLAS). U (L)
// is a product of permutation and unit upper (lower) triangular matrices,
// D is block diagonal with 1-by-1 and 2-by-2 blocks. A returns D and the
// multipliers in the triangle UPLO. IPIV(k) > 0: rows and columns k and
// IPIV(k) were interchanged and D(k, k) is a 1-by-1 block; IPIV(k) =
// IPIV(k-1) < 0 (UPLO = 'U') or IPIV(k) = IPIV(k+1) < 0 ('L'): rows and
// columns k-1 (k+1) and -IPIV(k) were interchanged and D(k-1:k, k-1:k)
// (D(k:k+1, k:k+1)) is a 2-by-2 block.
//
// The real routine names the reciprocal of the 1-by-1 pivot D11 in the
// lower case, the complex one R1; the template uses R1.
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if D(i, i) is exactly zero (or column i contains a NaN): the
// factorization has been completed, but D is singular.
template <typename T>
blasint Sytf2(char uplo, blasint n, T* a, blasint lda, blasint* ipiv) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1), eight = R(8), sevten = R(17);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("SYTF2", -info);
    return info;
  }

  // Initialize ALPHA for use in choosing pivot block size.
  const R alpha = (one + std::sqrt(sevten)) / eight;

  blasint imax = 0;  // set (and used) if COLMAX > 0
  if (upper) {
    // Factorize A as U*D*U**T using the upper triangle of A.
    //
    // K is the main loop index, decreasing from N to 1 in steps of 1 or 2.
    blasint k = n;
    while (k >= 1) {
      blasint kstep = 1;

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = Cabs1(A(k, k));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k > 1) {
        imax = Iamax(k - 1, A.ptr(1, k), 1);
        colmax = Cabs1(A(imax, k));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero || std::isnan(absakk)) {
        // Column K is zero or underflow, or contains a NaN: set INFO and
        // continue.
        if (info == 0) info = k;
        kp = k;
      } else {
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value.
          blasint jmax = imax + Iamax(k - imax, A.ptr(imax, imax + 1), lda);
          R rowmax = Cabs1(A(imax, jmax));
          if (imax > 1) {
            jmax = Iamax(imax - 1, A.ptr(1, imax), 1);
            rowmax = Max(rowmax, Cabs1(A(jmax, imax)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (Cabs1(A(imax, imax)) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;
          } else {
            // Interchange rows and columns K-1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        const blasint kk = k - kstep + 1;
        if (kp != kk) {
          // Interchange rows and columns KK and KP in the leading submatrix
          // A(1:k,1:k).
          Swap(kp - 1, A.ptr(1, kk), 1, A.ptr(1, kp), 1);
          Swap(kk - kp - 1, A.ptr(kp + 1, kk), 1, A.ptr(kp, kp + 1), lda);
          T t = A(kk, kk);
          A(kk, kk) = A(kp, kp);
          A(kp, kp) = t;
          if (kstep == 2) {
            t = A(k - 1, k);
            A(k - 1, k) = A(kp, k);
            A(kp, k) = t;
          }
        }

        // Update the leading submatrix.
        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column k now holds
          //
          // W(k) = U(k)*D(k)
          //
          // where U(k) is the k-th column of U.
          //
          // Perform a rank-1 update of A(1:k-1,1:k-1) as
          //
          // A := A - U(k)*D(k)*U(k)**T = A - W(k)*1/D(k)*W(k)**T
          const T r1 = cone / A(k, k);
          Syr(uplo, k - 1, -r1, A.ptr(1, k), 1, a, lda);

          // Store U(k) in column k.
          Scal(k - 1, r1, A.ptr(1, k), 1);
        } else {
          // 2-by-2 pivot block D(k): columns k and k-1 now hold
          //
          // ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
          //
          // where U(k) and U(k-1) are the k-th and (k-1)-th columns of U.
          //
          // Perform a rank-2 update of A(1:k-2,1:k-2) as
          //
          // A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**T
          //    = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )**T
          if (k > 2) {
            T d12 = A(k - 1, k);
            const T d22 = A(k - 1, k - 1) / d12;
            const T d11 = A(k, k) / d12;
            const T t = cone / (d11 * d22 - cone);
            d12 = t / d12;

            for (blasint j = k - 2; j >= 1; --j) {
              const T wkm1 = d12 * (d11 * A(j, k - 1) - A(j, k));
              const T wk = d12 * (d22 * A(j, k) - A(j, k - 1));
              for (blasint i = j; i >= 1; --i) {
                A(i, j) = A(i, j) - A(i, k) * wk - A(i, k - 1) * wkm1;
              }
              A(j, k) = wk;
              A(j, k - 1) = wkm1;
            }
          }
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k - 1) = -kp;
      }

      // Decrease K and return to the start of the main loop.
      k = k - kstep;
    }
  } else {
    // Factorize A as L*D*L**T using the lower triangle of A.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2.
    blasint k = 1;
    while (k <= n) {
      blasint kstep = 1;

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = Cabs1(A(k, k));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k < n) {
        imax = k + Iamax(n - k, A.ptr(k + 1, k), 1);
        colmax = Cabs1(A(imax, k));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero || std::isnan(absakk)) {
        // Column K is zero or underflow, or contains a NaN: set INFO and
        // continue.
        if (info == 0) info = k;
        kp = k;
      } else {
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value.
          blasint jmax = k - 1 + Iamax(imax - k, A.ptr(imax, k), lda);
          R rowmax = Cabs1(A(imax, jmax));
          if (imax < n) {
            jmax = imax + Iamax(n - imax, A.ptr(imax + 1, imax), 1);
            rowmax = Max(rowmax, Cabs1(A(jmax, imax)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (Cabs1(A(imax, imax)) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;
          } else {
            // Interchange rows and columns K+1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        const blasint kk = k + kstep - 1;
        if (kp != kk) {
          // Interchange rows and columns KK and KP in the trailing
          // submatrix A(k:n,k:n).
          if (kp < n) {
            Swap(n - kp, A.ptr(kp + 1, kk), 1, A.ptr(kp + 1, kp), 1);
          }
          Swap(kp - kk - 1, A.ptr(kk + 1, kk), 1, A.ptr(kp, kk + 1), lda);
          T t = A(kk, kk);
          A(kk, kk) = A(kp, kp);
          A(kp, kp) = t;
          if (kstep == 2) {
            t = A(k + 1, k);
            A(k + 1, k) = A(kp, k);
            A(kp, k) = t;
          }
        }

        // Update the trailing submatrix.
        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column k now holds
          //
          // W(k) = L(k)*D(k)
          //
          // where L(k) is the k-th column of L.
          if (k < n) {
            // Perform a rank-1 update of A(k+1:n,k+1:n) as
            //
            // A := A - L(k)*D(k)*L(k)**T = A - W(k)*(1/D(k))*W(k)**T
            const T r1 = cone / A(k, k);
            Syr(uplo, n - k, -r1, A.ptr(k + 1, k), 1, A.ptr(k + 1, k + 1), lda);

            // Store L(k) in column K.
            Scal(n - k, r1, A.ptr(k + 1, k), 1);
          }
        } else {
          // 2-by-2 pivot block D(k).
          if (k < n - 1) {
            // Perform a rank-2 update of A(k+2:n,k+2:n) as
            //
            // A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**T
            //    = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**T
            //
            // where L(k) and L(k+1) are the k-th and (k+1)-th columns of
            // L.
            T d21 = A(k + 1, k);
            const T d11 = A(k + 1, k + 1) / d21;
            const T d22 = A(k, k) / d21;
            const T t = cone / (d11 * d22 - cone);
            d21 = t / d21;

            for (blasint j = k + 2; j <= n; ++j) {
              const T wk = d21 * (d11 * A(j, k) - A(j, k + 1));
              const T wkp1 = d21 * (d22 * A(j, k + 1) - A(j, k));
              for (blasint i = j; i <= n; ++i) {
                A(i, j) = A(i, j) - A(i, k) * wk - A(i, k + 1) * wkp1;
              }
              A(j, k) = wk;
              A(j, k + 1) = wkp1;
            }
          }
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k + 1) = -kp;
      }

      // Increase K and return to the start of the main loop.
      k = k + kstep;
    }
  }
  return info;
}

// xLASYF: factors NB columns of the symmetric matrix A of order N with the
// Bunch-Kaufman diagonal pivoting method, for xSYTRF: the last NB columns
// (UPLO = 'U') or the first NB ('L'), and updates the rest of A with Level
// 3 BLAS (A11 := A11 - U12*D*U12**T or A22 := A22 - L21*D*L21**T). W
// (LDW-by-NB) is workspace. KB returns the number of columns factored: NB
// or NB-1 (a 2-by-2 block would straddle the boundary), or N if N <= NB.
// IPIV as in xSYTF2.
//
// Returns INFO: 0 on success, or k > 0 if D(k, k) is exactly zero (the
// factorization goes on, but D is singular).
template <typename T>
blasint Lasyf(char uplo, blasint n, blasint nb, blasint* kb, T* a, blasint lda,
              blasint* ipiv, T* w, blasint ldw) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1), eight = R(8), sevten = R(17);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);
  const Matrix<T> W(w, ldw);

  blasint info = 0;

  // Initialize ALPHA for use in choosing pivot block size.
  const R alpha = (one + std::sqrt(sevten)) / eight;

  blasint imax = 0;  // set (and used) if COLMAX > 0
  if (Lsame(uplo, 'U')) {
    // Factorize the trailing columns of A using the upper triangle of A and
    // working backwards, and compute the matrix W = U12*D for use in
    // updating A11.
    //
    // K is the main loop index, decreasing from N in steps of 1 or 2.
    //
    // KW is the column of W which corresponds to column K of A.
    blasint k = n;
    blasint kw;
    for (;;) {
      kw = nb + k - n;

      // Exit from loop.
      if ((k <= n - nb + 1 && nb < n) || k < 1) break;

      // Copy column K of A to column KW of W and update it.
      Copy(k, A.ptr(1, k), 1, W.ptr(1, kw), 1);
      if (k < n) {
        Gemv('N', k, n - k, -cone, A.ptr(1, k + 1), lda, W.ptr(k, kw + 1), ldw,
             cone, W.ptr(1, kw), 1);
      }

      blasint kstep = 1;

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = Cabs1(W(k, kw));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k > 1) {
        imax = Iamax(k - 1, W.ptr(1, kw), 1);
        colmax = Cabs1(W(imax, kw));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero) {
        // Column K is zero or underflow: set INFO and continue.
        if (info == 0) info = k;
        kp = k;
      } else {
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // Copy column IMAX to column KW-1 of W and update it.
          Copy(imax, A.ptr(1, imax), 1, W.ptr(1, kw - 1), 1);
          Copy(k - imax, A.ptr(imax, imax + 1), lda, W.ptr(imax + 1, kw - 1),
               1);
          if (k < n) {
            Gemv('N', k, n - k, -cone, A.ptr(1, k + 1), lda,
                 W.ptr(imax, kw + 1), ldw, cone, W.ptr(1, kw - 1), 1);
          }

          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value.
          blasint jmax = imax + Iamax(k - imax, W.ptr(imax + 1, kw - 1), 1);
          R rowmax = Cabs1(W(jmax, kw - 1));
          if (imax > 1) {
            jmax = Iamax(imax - 1, W.ptr(1, kw - 1), 1);
            rowmax = Max(rowmax, Cabs1(W(jmax, kw - 1)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (Cabs1(W(imax, kw - 1)) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;

            // Copy column KW-1 of W to column KW of W.
            Copy(k, W.ptr(1, kw - 1), 1, W.ptr(1, kw), 1);
          } else {
            // Interchange rows and columns K-1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        // KK is the column of A where pivoting step stopped.
        const blasint kk = k - kstep + 1;

        // KKW is the column of W which corresponds to column KK of A.
        const blasint kkw = nb + kk - n;

        // Interchange rows and columns KP and KK. Updated column KP is
        // already stored in column KKW of W.
        if (kp != kk) {
          // Copy non-updated column KK to column KP of submatrix A at step
          // K. No need to copy element into column K (or K and K-1 for
          // 2-by-2 pivot) of A, since these columns will be later
          // overwritten.
          A(kp, kp) = A(kk, kk);
          Copy(kk - 1 - kp, A.ptr(kp + 1, kk), 1, A.ptr(kp, kp + 1), lda);
          if (kp > 1) Copy(kp - 1, A.ptr(1, kk), 1, A.ptr(1, kp), 1);

          // Interchange rows KK and KP in last K+1 to N columns of A
          // (columns K (or K and K-1 for 2-by-2 pivot) of A will be later
          // overwritten). Interchange rows KK and KP in last KKW to NB
          // columns of W.
          if (k < n) {
            Swap(n - k, A.ptr(kk, k + 1), lda, A.ptr(kp, k + 1), lda);
          }
          Swap(n - kk + 1, W.ptr(kk, kkw), ldw, W.ptr(kp, kkw), ldw);
        }

        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column kw of W now holds
          //
          // W(kw) = U(k)*D(k),
          //
          // where U(k) is the k-th column of U.
          //
          // Store subdiag. elements of column U(k) and 1-by-1 block D(k) in
          // column k of A. NOTE: Diagonal element U(k,k) is a UNIT element
          // and not stored.
          //    A(k,k) := D(k,k) = W(k,kw)
          //    A(1:k-1,k) := U(1:k-1,k) = W(1:k-1,kw)/D(k,k)
          Copy(k, W.ptr(1, kw), 1, A.ptr(1, k), 1);
          const T r1 = cone / A(k, k);
          Scal(k - 1, r1, A.ptr(1, k), 1);
        } else {
          // 2-by-2 pivot block D(k): columns kw and kw-1 of W now hold
          //
          // ( W(kw-1) W(kw) ) = ( U(k-1) U(k) )*D(k)
          //
          // where U(k) and U(k-1) are the k-th and (k-1)-th columns of U.
          //
          // Store U(1:k-2,k-1) and U(1:k-2,k) and 2-by-2 block
          // D(k-1:k,k-1:k) in columns k-1 and k of A. NOTE: 2-by-2 diagonal
          // block U(k-1:k,k-1:k) is a UNIT block and not stored.
          //    A(k-1:k,k-1:k) := D(k-1:k,k-1:k) = W(k-1:k,kw-1:kw)
          //    A(1:k-2,k-1:k) := U(1:k-2,k:k-1:k) =
          //    = W(1:k-2,kw-1:kw) * ( D(k-1:k,k-1:k)**(-1) )
          if (k > 2) {
            // Compose the columns of the inverse of 2-by-2 pivot block D in
            // the following way to reduce the number of FLOPS when we
            // multiply panel ( W(kw-1) W(kw) ) by this inverse
            //
            // D**(-1) = ( d11 d21 )**(-1) =
            //           ( d21 d22 )
            //
            // = 1/(d11*d22-d21**2) * ( ( d22 ) (-d21 ) ) =
            //                        ( (-d21 ) ( d11 ) )
            //
            // = 1/d21 * 1/((d11/d21)*(d22/d21)-1) *
            //
            //   * ( ( d22/d21 ) (      -1 ) ) =
            //     ( (      -1 ) ( d11/d21 ) )
            //
            // = 1/d21 * 1/(D22*D11-1) * ( ( D11 ) (  -1 ) ) =
            //                           ( ( -1  ) ( D22 ) )
            //
            // = 1/d21 * T * ( ( D11 ) (  -1 ) )
            //               ( (  -1 ) ( D22 ) )
            //
            // = D21 * ( ( D11 ) (  -1 ) )
            //         ( (  -1 ) ( D22 ) )
            T d21 = W(k - 1, kw);
            const T d11 = W(k, kw) / d21;
            const T d22 = W(k - 1, kw - 1) / d21;
            const T t = cone / (d11 * d22 - cone);
            d21 = t / d21;

            // Update elements in columns A(k-1) and A(k) as dot products of
            // rows of ( W(kw-1) W(kw) ) and columns of D**(-1).
            for (blasint j = 1; j <= k - 2; ++j) {
              A(j, k - 1) = d21 * (d11 * W(j, kw - 1) - W(j, kw));
              A(j, k) = d21 * (d22 * W(j, kw) - W(j, kw - 1));
            }
          }

          // Copy D(k) to A.
          A(k - 1, k - 1) = W(k - 1, kw - 1);
          A(k - 1, k) = W(k - 1, kw);
          A(k, k) = W(k, kw);
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k - 1) = -kp;
      }

      // Decrease K and return to the start of the main loop.
      k = k - kstep;
    }

    // Update the upper triangle of A11 (= A(1:k,1:k)) as
    //
    // A11 := A11 - U12*D*U12**T = A11 - U12*W**T
    Gemmtr('U', 'N', 'T', k, n - k, -cone, A.ptr(1, k + 1), lda,
           W.ptr(1, kw + 1), ldw, cone, A.ptr(1, 1), lda);

    // Put U12 in standard form by partially undoing the interchanges in
    // columns k+1:n looping backwards from k+1 to n.
    blasint j = k + 1;
    do {
      // Undo the interchanges (if any) of rows JJ and JP at each step J.
      //
      // (Here, J is a diagonal index.)
      const blasint jj = j;
      blasint jp = IPIV(j);
      if (jp < 0) {
        jp = -jp;
        // (Here, J is a diagonal index.)
        j = j + 1;
      }
      // (NOTE: Here, J is used to determine row length. Length N-J+1 of the
      // rows to swap back doesn't include diagonal element.)
      j = j + 1;
      if (jp != jj && j <= n) {
        Swap(n - j + 1, A.ptr(jp, j), lda, A.ptr(jj, j), lda);
      }
    } while (j < n);

    // Set KB to the number of columns factorized.
    *kb = n - k;
  } else {
    // Factorize the leading columns of A using the lower triangle of A and
    // working forwards, and compute the matrix W = L21*D for use in
    // updating A22.
    //
    // K is the main loop index, increasing from 1 in steps of 1 or 2.
    blasint k = 1;
    for (;;) {
      // Exit from loop.
      if ((k >= nb && nb < n) || k > n) break;

      // Copy column K of A to column K of W and update it.
      Copy(n - k + 1, A.ptr(k, k), 1, W.ptr(k, k), 1);
      Gemv('N', n - k + 1, k - 1, -cone, A.ptr(k, 1), lda, W.ptr(k, 1), ldw,
           cone, W.ptr(k, k), 1);

      blasint kstep = 1;

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = Cabs1(W(k, k));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k < n) {
        imax = k + Iamax(n - k, W.ptr(k + 1, k), 1);
        colmax = Cabs1(W(imax, k));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero) {
        // Column K is zero or underflow: set INFO and continue.
        if (info == 0) info = k;
        kp = k;
      } else {
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // Copy column IMAX to column K+1 of W and update it.
          Copy(imax - k, A.ptr(imax, k), lda, W.ptr(k, k + 1), 1);
          Copy(n - imax + 1, A.ptr(imax, imax), 1, W.ptr(imax, k + 1), 1);
          Gemv('N', n - k + 1, k - 1, -cone, A.ptr(k, 1), lda, W.ptr(imax, 1),
               ldw, cone, W.ptr(k, k + 1), 1);

          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value.
          blasint jmax = k - 1 + Iamax(imax - k, W.ptr(k, k + 1), 1);
          R rowmax = Cabs1(W(jmax, k + 1));
          if (imax < n) {
            jmax = imax + Iamax(n - imax, W.ptr(imax + 1, k + 1), 1);
            rowmax = Max(rowmax, Cabs1(W(jmax, k + 1)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (Cabs1(W(imax, k + 1)) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;

            // Copy column K+1 of W to column K of W.
            Copy(n - k + 1, W.ptr(k, k + 1), 1, W.ptr(k, k), 1);
          } else {
            // Interchange rows and columns K+1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        // KK is the column of A where pivoting step stopped.
        const blasint kk = k + kstep - 1;

        // Interchange rows and columns KP and KK. Updated column KP is
        // already stored in column KK of W.
        if (kp != kk) {
          // Copy non-updated column KK to column KP of submatrix A at step
          // K. No need to copy element into column K (or K and K+1 for
          // 2-by-2 pivot) of A, since these columns will be later
          // overwritten.
          A(kp, kp) = A(kk, kk);
          Copy(kp - kk - 1, A.ptr(kk + 1, kk), 1, A.ptr(kp, kk + 1), lda);
          if (kp < n) {
            Copy(n - kp, A.ptr(kp + 1, kk), 1, A.ptr(kp + 1, kp), 1);
          }

          // Interchange rows KK and KP in first K-1 columns of A (columns K
          // (or K and K+1 for 2-by-2 pivot) of A will be later
          // overwritten). Interchange rows KK and KP in first KK columns of
          // W.
          if (k > 1) Swap(k - 1, A.ptr(kk, 1), lda, A.ptr(kp, 1), lda);
          Swap(kk, W.ptr(kk, 1), ldw, W.ptr(kp, 1), ldw);
        }

        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column k of W now holds
          //
          // W(k) = L(k)*D(k),
          //
          // where L(k) is the k-th column of L.
          //
          // Store subdiag. elements of column L(k) and 1-by-1 block D(k) in
          // column k of A. (NOTE: Diagonal element L(k,k) is a UNIT element
          // and not stored.)
          //    A(k,k) := D(k,k) = W(k,k)
          //    A(k+1:N,k) := L(k+1:N,k) = W(k+1:N,k)/D(k,k)
          Copy(n - k + 1, W.ptr(k, k), 1, A.ptr(k, k), 1);
          if (k < n) {
            const T r1 = cone / A(k, k);
            Scal(n - k, r1, A.ptr(k + 1, k), 1);
          }
        } else {
          // 2-by-2 pivot block D(k): columns k and k+1 of W now hold
          //
          // ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
          //
          // where L(k) and L(k+1) are the k-th and (k+1)-th columns of L.
          //
          // Store L(k+2:N,k) and L(k+2:N,k+1) and 2-by-2 block
          // D(k:k+1,k:k+1) in columns k and k+1 of A. (NOTE: 2-by-2
          // diagonal block L(k:k+1,k:k+1) is a UNIT block and not stored.)
          //    A(k:k+1,k:k+1) := D(k:k+1,k:k+1) = W(k:k+1,k:k+1)
          //    A(k+2:N,k:k+1) := L(k+2:N,k:k+1) =
          //    = W(k+2:N,k:k+1) * ( D(k:k+1,k:k+1)**(-1) )
          if (k < n - 1) {
            // Compose the columns of the inverse of 2-by-2 pivot block D in
            // the following way to reduce the number of FLOPS when we
            // multiply panel ( W(k) W(k+1) ) by this inverse (see the
            // upper case).
            T d21 = W(k + 1, k);
            const T d11 = W(k + 1, k + 1) / d21;
            const T d22 = W(k, k) / d21;
            const T t = cone / (d11 * d22 - cone);
            d21 = t / d21;

            // Update elements in columns A(k) and A(k+1) as dot products of
            // rows of ( W(k) W(k+1) ) and columns of D**(-1).
            for (blasint j = k + 2; j <= n; ++j) {
              A(j, k) = d21 * (d11 * W(j, k) - W(j, k + 1));
              A(j, k + 1) = d21 * (d22 * W(j, k + 1) - W(j, k));
            }
          }

          // Copy D(k) to A.
          A(k, k) = W(k, k);
          A(k + 1, k) = W(k + 1, k);
          A(k + 1, k + 1) = W(k + 1, k + 1);
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k + 1) = -kp;
      }

      // Increase K and return to the start of the main loop.
      k = k + kstep;
    }

    // Update the lower triangle of A22 (= A(k:n,k:n)) as
    //
    // A22 := A22 - L21*D*L21**T = A22 - L21*W**T
    Gemmtr('L', 'N', 'T', n - k + 1, k - 1, -cone, A.ptr(k, 1), lda,
           W.ptr(k, 1), ldw, cone, A.ptr(k, k), lda);

    // Put L21 in standard form by partially undoing the interchanges of
    // rows in columns 1:k-1 looping backwards from k-1 to 1.
    blasint j = k - 1;
    do {
      // Undo the interchanges (if any) of rows JJ and JP at each step J.
      //
      // (Here, J is a diagonal index.)
      const blasint jj = j;
      blasint jp = IPIV(j);
      if (jp < 0) {
        jp = -jp;
        // (Here, J is a diagonal index.)
        j = j - 1;
      }
      // (NOTE: Here, J is used to determine row length. Length J of the
      // rows to swap back doesn't include diagonal element.)
      j = j - 1;
      if (jp != jj && j >= 1) Swap(j, A.ptr(jp, 1), lda, A.ptr(jj, 1), lda);
    } while (j > 1);

    // Set KB to the number of columns factorized.
    *kb = k - 1;
  }
  return info;
}

// The driver of xSYTRF and xHETRF: Panel is xLASYF or xLAHEF, Unblocked
// xSYTF2 or xHETF2. The two routines are the same but for these calls and
// their names (and the comments: U**T against U**H).
template <typename T, typename Panel, typename Unblocked>
blasint SytrfDriver(const char* name, char uplo, blasint n, T* a, blasint lda,
                    blasint* ipiv, T* work, blasint lwork, Panel panel,
                    Unblocked unblocked) {
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);
  const Vector<T> WORK(work);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  const bool lquery = lwork == -1;
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  } else if (lwork < 1 && !lquery) {
    info = -7;
  }

  const char opts[2] = {uplo, '\0'};
  blasint nb = 0, lwkopt = 0;
  if (info == 0) {
    // Determine the block size.
    nb = Ilaenv<T>(1, name, opts, n, -1, -1, -1);
    lwkopt = Max<blasint>(1, n * nb);
    WORK(1) = WorkSize<T>(lwkopt);
  }

  if (info != 0) {
    Xerbla<T>(name, -info);
    return info;
  } else if (lquery) {
    return info;
  }

  blasint nbmin = 2;
  const blasint ldwork = n;
  if (nb > 1 && nb < n) {
    const blasint iws = ldwork * nb;
    if (lwork < iws) {
      nb = Max<blasint>(lwork / ldwork, 1);
      nbmin = Max<blasint>(2, Ilaenv<T>(2, name, opts, n, -1, -1, -1));
    }
  }
  // (The reference's IWS, the workspace used, is not returned.)
  if (nb < nbmin) nb = n;

  if (upper) {
    // Factorize A as U*D*U**T (U**H) using the upper triangle of A.
    //
    // K is the main loop index, decreasing from N to 1 in steps of KB,
    // where KB is the number of columns factorized by xLASYF (xLAHEF); KB
    // is either NB or NB-1, or K for the last block.
    blasint k = n;
    while (k >= 1) {
      blasint kb, iinfo;
      if (k > nb) {
        // Factorize columns k-kb+1:k of A and use blocked code to update
        // columns 1:k-kb.
        iinfo = panel(uplo, k, nb, &kb, a, lda, ipiv, work, n);
      } else {
        // Use unblocked code to factorize columns 1:k of A.
        iinfo = unblocked(uplo, k, a, lda, ipiv);
        kb = k;
      }

      // Set INFO on the first occurrence of a zero pivot.
      if (info == 0 && iinfo > 0) info = iinfo;

      // Decrease K and return to the start of the main loop.
      k = k - kb;
    }
  } else {
    // Factorize A as L*D*L**T (L**H) using the lower triangle of A.
    //
    // K is the main loop index, increasing from 1 to N in steps of KB,
    // where KB is the number of columns factorized by xLASYF (xLAHEF); KB
    // is either NB or NB-1, or N-K+1 for the last block.
    blasint k = 1;
    while (k <= n) {
      blasint kb, iinfo;
      if (k <= n - nb) {
        // Factorize columns k:k+kb-1 of A and use blocked code to update
        // columns k+kb:n.
        iinfo = panel(uplo, n - k + 1, nb, &kb, A.ptr(k, k), lda, IPIV.ptr(k),
                      work, n);
      } else {
        // Use unblocked code to factorize columns k:n of A.
        iinfo = unblocked(uplo, n - k + 1, A.ptr(k, k), lda, IPIV.ptr(k));
        kb = n - k + 1;
      }

      // Set INFO on the first occurrence of a zero pivot.
      if (info == 0 && iinfo > 0) info = iinfo + k - 1;

      // Adjust IPIV.
      for (blasint j = k; j <= k + kb - 1; ++j) {
        if (IPIV(j) > 0) {
          IPIV(j) = IPIV(j) + k - 1;
        } else {
          IPIV(j) = IPIV(j) - k + 1;
        }
      }

      // Increase K and return to the start of the main loop.
      k = k + kb;
    }
  }

  WORK(1) = WorkSize<T>(lwkopt);
  return info;
}

// xSYTRF: computes the Bunch-Kaufman factorization A = U * D * U**T or
// L * D * L**T of the symmetric matrix A of order N (triangle UPLO; see
// xSYTF2 for the result), blocked: xLASYF factors NB columns at a time (NB
// from ILAENV, reduced to fit LWORK), xSYTF2 the last ones. WORK has
// LWORK >= 1 elements, N*NB for the optimal block size; LWORK = -1 is a
// workspace query (the optimal size is returned in WORK(1)).
//
// Returns INFO: 0 on success, -i if argument i had an illegal value, or
// i > 0 if D(i, i) is exactly zero (the factorization has been completed,
// but D is singular).
template <typename T>
blasint Sytrf(char uplo, blasint n, T* a, blasint lda, blasint* ipiv, T* work,
              blasint lwork) {
  return SytrfDriver("SYTRF", uplo, n, a, lda, ipiv, work, lwork, &Lasyf<T>,
                     &Sytf2<T>);
}

// xHETF2 (complex T only): computes the factorization A = U * D * U**H
// (UPLO = 'U') or A = L * D * L**H ('L') of the Hermitian matrix A of order
// N with the Bunch-Kaufman diagonal pivoting method, unblocked, like
// xSYTF2 (see there for the result and INFO). The diagonal of D is real.
template <typename T>
blasint Hetf2(char uplo, blasint n, T* a, blasint lda, blasint* ipiv) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1), eight = R(8), sevten = R(17);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);

  // Test the input parameters.
  blasint info = 0;
  const bool upper = Lsame(uplo, 'U');
  if (!upper && !Lsame(uplo, 'L')) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Max<blasint>(1, n)) {
    info = -4;
  }
  if (info != 0) {
    Xerbla<T>("HETF2", -info);
    return info;
  }

  // Initialize ALPHA for use in choosing pivot block size.
  const R alpha = (one + std::sqrt(sevten)) / eight;

  blasint imax = 0;  // set (and used) if COLMAX > 0
  if (upper) {
    // Factorize A as U*D*U**H using the upper triangle of A.
    //
    // K is the main loop index, decreasing from N to 1 in steps of 1 or 2.
    blasint k = n;
    while (k >= 1) {
      blasint kstep = 1;

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = std::abs(std::real(A(k, k)));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k > 1) {
        imax = Iamax(k - 1, A.ptr(1, k), 1);
        colmax = Cabs1(A(imax, k));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero || std::isnan(absakk)) {
        // Column K is zero or underflow, or contains a NaN: set INFO and
        // continue.
        if (info == 0) info = k;
        kp = k;
        A(k, k) = std::real(A(k, k));
      } else {
        // Test for interchange.
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value. Determine only
          // ROWMAX.
          blasint jmax = imax + Iamax(k - imax, A.ptr(imax, imax + 1), lda);
          R rowmax = Cabs1(A(imax, jmax));
          if (imax > 1) {
            jmax = Iamax(imax - 1, A.ptr(1, imax), 1);
            rowmax = Max(rowmax, Cabs1(A(jmax, imax)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (std::abs(std::real(A(imax, imax))) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;
          } else {
            // Interchange rows and columns K-1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        const blasint kk = k - kstep + 1;
        if (kp != kk) {
          // Interchange rows and columns KK and KP in the leading submatrix
          // A(1:k,1:k).
          Swap(kp - 1, A.ptr(1, kk), 1, A.ptr(1, kp), 1);
          for (blasint j = kp + 1; j <= kk - 1; ++j) {
            const T t = Conj(A(j, kk));
            A(j, kk) = Conj(A(kp, j));
            A(kp, j) = t;
          }
          A(kp, kk) = Conj(A(kp, kk));
          const R r1 = std::real(A(kk, kk));
          A(kk, kk) = std::real(A(kp, kp));
          A(kp, kp) = r1;
          if (kstep == 2) {
            A(k, k) = std::real(A(k, k));
            const T t = A(k - 1, k);
            A(k - 1, k) = A(kp, k);
            A(kp, k) = t;
          }
        } else {
          A(k, k) = std::real(A(k, k));
          if (kstep == 2) A(k - 1, k - 1) = std::real(A(k - 1, k - 1));
        }

        // Update the leading submatrix.
        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column k now holds
          //
          // W(k) = U(k)*D(k)
          //
          // where U(k) is the k-th column of U.
          //
          // Perform a rank-1 update of A(1:k-1,1:k-1) as
          //
          // A := A - U(k)*D(k)*U(k)**H = A - W(k)*1/D(k)*W(k)**H
          const R r1 = one / std::real(A(k, k));
          Her(uplo, k - 1, -r1, A.ptr(1, k), 1, a, lda);

          // Store U(k) in column k.
          Rscal(k - 1, r1, A.ptr(1, k), 1);
        } else {
          // 2-by-2 pivot block D(k): columns k and k-1 now hold
          //
          // ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
          //
          // where U(k) and U(k-1) are the k-th and (k-1)-th columns of U.
          //
          // Perform a rank-2 update of A(1:k-2,1:k-2) as
          //
          // A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**H
          //    = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )**H
          if (k > 2) {
            R d = Lapy2(std::real(A(k - 1, k)), std::imag(A(k - 1, k)));
            const R d22 = std::real(A(k - 1, k - 1)) / d;
            const R d11 = std::real(A(k, k)) / d;
            const R tt = one / (d11 * d22 - one);
            const T d12 = A(k - 1, k) / d;
            d = tt / d;

            for (blasint j = k - 2; j >= 1; --j) {
              const T wkm1 = d * (d11 * A(j, k - 1) - Conj(d12) * A(j, k));
              const T wk = d * (d22 * A(j, k) - d12 * A(j, k - 1));
              for (blasint i = j; i >= 1; --i) {
                A(i, j) =
                    A(i, j) - A(i, k) * Conj(wk) - A(i, k - 1) * Conj(wkm1);
              }
              A(j, k) = wk;
              A(j, k - 1) = wkm1;
              A(j, j) = T(std::real(A(j, j)), R(0));
            }
          }
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k - 1) = -kp;
      }

      // Decrease K and return to the start of the main loop.
      k = k - kstep;
    }
  } else {
    // Factorize A as L*D*L**H using the lower triangle of A.
    //
    // K is the main loop index, increasing from 1 to N in steps of 1 or 2.
    blasint k = 1;
    while (k <= n) {
      blasint kstep = 1;

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = std::abs(std::real(A(k, k)));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k < n) {
        imax = k + Iamax(n - k, A.ptr(k + 1, k), 1);
        colmax = Cabs1(A(imax, k));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero || std::isnan(absakk)) {
        // Column K is zero or underflow, or contains a NaN: set INFO and
        // continue.
        if (info == 0) info = k;
        kp = k;
        A(k, k) = std::real(A(k, k));
      } else {
        // Test for interchange.
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value. Determine only
          // ROWMAX.
          blasint jmax = k - 1 + Iamax(imax - k, A.ptr(imax, k), lda);
          R rowmax = Cabs1(A(imax, jmax));
          if (imax < n) {
            jmax = imax + Iamax(n - imax, A.ptr(imax + 1, imax), 1);
            rowmax = Max(rowmax, Cabs1(A(jmax, imax)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (std::abs(std::real(A(imax, imax))) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;
          } else {
            // Interchange rows and columns K+1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        const blasint kk = k + kstep - 1;
        if (kp != kk) {
          // Interchange rows and columns KK and KP in the trailing
          // submatrix A(k:n,k:n).
          if (kp < n) {
            Swap(n - kp, A.ptr(kp + 1, kk), 1, A.ptr(kp + 1, kp), 1);
          }
          for (blasint j = kk + 1; j <= kp - 1; ++j) {
            const T t = Conj(A(j, kk));
            A(j, kk) = Conj(A(kp, j));
            A(kp, j) = t;
          }
          A(kp, kk) = Conj(A(kp, kk));
          const R r1 = std::real(A(kk, kk));
          A(kk, kk) = std::real(A(kp, kp));
          A(kp, kp) = r1;
          if (kstep == 2) {
            A(k, k) = std::real(A(k, k));
            const T t = A(k + 1, k);
            A(k + 1, k) = A(kp, k);
            A(kp, k) = t;
          }
        } else {
          A(k, k) = std::real(A(k, k));
          if (kstep == 2) A(k + 1, k + 1) = std::real(A(k + 1, k + 1));
        }

        // Update the trailing submatrix.
        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column k now holds
          //
          // W(k) = L(k)*D(k)
          //
          // where L(k) is the k-th column of L.
          if (k < n) {
            // Perform a rank-1 update of A(k+1:n,k+1:n) as
            //
            // A := A - L(k)*D(k)*L(k)**H = A - W(k)*(1/D(k))*W(k)**H
            const R r1 = one / std::real(A(k, k));
            Her(uplo, n - k, -r1, A.ptr(k + 1, k), 1, A.ptr(k + 1, k + 1), lda);

            // Store L(k) in column K.
            Rscal(n - k, r1, A.ptr(k + 1, k), 1);
          }
        } else {
          // 2-by-2 pivot block D(k).
          if (k < n - 1) {
            // Perform a rank-2 update of A(k+2:n,k+2:n) as
            //
            // A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**H
            //    = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**H
            //
            // where L(k) and L(k+1) are the k-th and (k+1)-th columns of
            // L.
            R d = Lapy2(std::real(A(k + 1, k)), std::imag(A(k + 1, k)));
            const R d11 = std::real(A(k + 1, k + 1)) / d;
            const R d22 = std::real(A(k, k)) / d;
            const R tt = one / (d11 * d22 - one);
            const T d21 = A(k + 1, k) / d;
            d = tt / d;

            for (blasint j = k + 2; j <= n; ++j) {
              const T wk = d * (d11 * A(j, k) - d21 * A(j, k + 1));
              const T wkp1 = d * (d22 * A(j, k + 1) - Conj(d21) * A(j, k));
              for (blasint i = j; i <= n; ++i) {
                A(i, j) =
                    A(i, j) - A(i, k) * Conj(wk) - A(i, k + 1) * Conj(wkp1);
              }
              A(j, k) = wk;
              A(j, k + 1) = wkp1;
              A(j, j) = T(std::real(A(j, j)), R(0));
            }
          }
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k + 1) = -kp;
      }

      // Increase K and return to the start of the main loop.
      k = k + kstep;
    }
  }
  return info;
}

// xLAHEF (complex T only): factors NB columns of the Hermitian matrix A of
// order N with the Bunch-Kaufman diagonal pivoting method, for xHETRF, like
// xLASYF (see there for the arguments and INFO). W holds the conjugates of
// the updated columns.
template <typename T>
blasint Lahef(char uplo, blasint n, blasint nb, blasint* kb, T* a, blasint lda,
              blasint* ipiv, T* w, blasint ldw) {
  using R = RealT<T>;
  const R zero = R(0), one = R(1), eight = R(8), sevten = R(17);
  const T cone = T(1);
  const Matrix<T> A(a, lda);
  const Vector<blasint> IPIV(ipiv);
  const Matrix<T> W(w, ldw);

  blasint info = 0;

  // Initialize ALPHA for use in choosing pivot block size.
  const R alpha = (one + std::sqrt(sevten)) / eight;

  blasint imax = 0;  // set (and used) if COLMAX > 0
  if (Lsame(uplo, 'U')) {
    // Factorize the trailing columns of A using the upper triangle of A and
    // working backwards, and compute the matrix W = U12*D for use in
    // updating A11 (note that conjg(W) is actually stored).
    //
    // K is the main loop index, decreasing from N in steps of 1 or 2.
    //
    // KW is the column of W which corresponds to column K of A.
    blasint k = n;
    blasint kw;
    for (;;) {
      kw = nb + k - n;

      // Exit from loop.
      if ((k <= n - nb + 1 && nb < n) || k < 1) break;

      blasint kstep = 1;

      // Copy column K of A to column KW of W and update it.
      Copy(k - 1, A.ptr(1, k), 1, W.ptr(1, kw), 1);
      W(k, kw) = std::real(A(k, k));
      if (k < n) {
        Gemv('N', k, n - k, -cone, A.ptr(1, k + 1), lda, W.ptr(k, kw + 1), ldw,
             cone, W.ptr(1, kw), 1);
        W(k, kw) = std::real(W(k, kw));
      }

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = std::abs(std::real(W(k, kw)));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k > 1) {
        imax = Iamax(k - 1, W.ptr(1, kw), 1);
        colmax = Cabs1(W(imax, kw));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero) {
        // Column K is zero or underflow: set INFO and continue.
        if (info == 0) info = k;
        kp = k;
        A(k, k) = std::real(A(k, k));
      } else {
        // Test for interchange.
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // Copy column IMAX to column KW-1 of W and update it.
          Copy(imax - 1, A.ptr(1, imax), 1, W.ptr(1, kw - 1), 1);
          W(imax, kw - 1) = std::real(A(imax, imax));
          Copy(k - imax, A.ptr(imax, imax + 1), lda, W.ptr(imax + 1, kw - 1),
               1);
          Lacgv(k - imax, W.ptr(imax + 1, kw - 1), 1);
          if (k < n) {
            Gemv('N', k, n - k, -cone, A.ptr(1, k + 1), lda,
                 W.ptr(imax, kw + 1), ldw, cone, W.ptr(1, kw - 1), 1);
            W(imax, kw - 1) = std::real(W(imax, kw - 1));
          }

          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value. Determine only
          // ROWMAX.
          blasint jmax = imax + Iamax(k - imax, W.ptr(imax + 1, kw - 1), 1);
          R rowmax = Cabs1(W(jmax, kw - 1));
          if (imax > 1) {
            jmax = Iamax(imax - 1, W.ptr(1, kw - 1), 1);
            rowmax = Max(rowmax, Cabs1(W(jmax, kw - 1)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (std::abs(std::real(W(imax, kw - 1))) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;

            // Copy column KW-1 of W to column KW of W.
            Copy(k, W.ptr(1, kw - 1), 1, W.ptr(1, kw), 1);
          } else {
            // Interchange rows and columns K-1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        // KK is the column of A where pivoting step stopped.
        const blasint kk = k - kstep + 1;

        // KKW is the column of W which corresponds to column KK of A.
        const blasint kkw = nb + kk - n;

        // Interchange rows and columns KP and KK. Updated column KP is
        // already stored in column KKW of W.
        if (kp != kk) {
          // Copy non-updated column KK to column KP of submatrix A at step
          // K. No need to copy element into column K (or K and K-1 for
          // 2-by-2 pivot) of A, since these columns will be later
          // overwritten.
          A(kp, kp) = std::real(A(kk, kk));
          Copy(kk - 1 - kp, A.ptr(kp + 1, kk), 1, A.ptr(kp, kp + 1), lda);
          Lacgv(kk - 1 - kp, A.ptr(kp, kp + 1), lda);
          if (kp > 1) Copy(kp - 1, A.ptr(1, kk), 1, A.ptr(1, kp), 1);

          // Interchange rows KK and KP in last K+1 to N columns of A
          // (columns K (or K and K-1 for 2-by-2 pivot) of A will be later
          // overwritten). Interchange rows KK and KP in last KKW to NB
          // columns of W.
          if (k < n) {
            Swap(n - k, A.ptr(kk, k + 1), lda, A.ptr(kp, k + 1), lda);
          }
          Swap(n - kk + 1, W.ptr(kk, kkw), ldw, W.ptr(kp, kkw), ldw);
        }

        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column kw of W now holds
          //
          // W(kw) = U(k)*D(k),
          //
          // where U(k) is the k-th column of U.
          //
          // (1) Store subdiag. elements of column U(k) and 1-by-1 block D(k)
          //     in column k of A. (NOTE: Diagonal element U(k,k) is a UNIT
          //     element and not stored.)
          //        A(k,k) := D(k,k) = W(k,kw)
          //        A(1:k-1,k) := U(1:k-1,k) = W(1:k-1,kw)/D(k,k)
          //
          //     (NOTE: Diagonal element D(k,k) is a REAL element, but we
          //     use a complex copy to store it.)
          Copy(k, W.ptr(1, kw), 1, A.ptr(1, k), 1);
          if (k > 1) {
            // Store U(k) in column k of A.
            const R r1 = one / std::real(A(k, k));
            Rscal(k - 1, r1, A.ptr(1, k), 1);

            // (2) Conjugate column W(kw).
            Lacgv(k - 1, W.ptr(1, kw), 1);
          }
        } else {
          // 2-by-2 pivot block D(k): columns kw and kw-1 of W now hold
          //
          // ( W(kw-1) W(kw) ) = ( U(k-1) U(k) )*D(k)
          //
          // where U(k) and U(k-1) are the k-th and (k-1)-th columns of U.
          //
          // (1) Store U(1:k-2,k-1) and U(1:k-2,k) and 2-by-2 block
          //     D(k-1:k,k-1:k) in columns k-1 and k of A. (NOTE: 2-by-2
          //     diagonal block U(k-1:k,k-1:k) is a UNIT block and not
          //     stored.)
          //        A(k-1:k,k-1:k) := D(k-1:k,k-1:k) = W(k-1:k,kw-1:kw)
          //        A(1:k-2,k-1:k) := U(1:k-2,k:k-1:k) =
          //        = W(1:k-2,kw-1:kw) * ( D(k-1:k,k-1:k)**(-1) )
          if (k > 2) {
            // Factor out the columns of the inverse of 2-by-2 pivot block D,
            // so that each column contains 1, to reduce the number of FLOPS
            // when we multiply panel ( W(kw-1) W(kw) ) by this inverse, i.e.
            // by D**(-1):
            //
            // D**(-1) = ( d11 cj(d21) )**(-1) =
            //           ( d21    d22 )
            //
            // = ( conj(D21)*( D11 ) D21*(  -1 ) )
            //   (           (  -1 )     ( D22 ) ),
            //
            // where D11 = d22/d21,
            //       D22 = d11/conj(d21),
            //       D21 = T/d21,
            //       T = 1/(D22*D11-1).
            //
            // (NOTE: No need to check for division by ZERO, since that was
            // ensured earlier in pivot search: (a) d21 != 0, since in 2x2
            // pivot case(4) |d21| should be larger than |d11| and |d22|;
            // (b) (D22*D11 - 1) != 0, since from (a), both |D11| < 1,
            // |D22| < 1, hence |D22*D11| << 1.)
            T d21 = W(k - 1, kw);
            const T d11 = W(k, kw) / Conj(d21);
            const T d22 = W(k - 1, kw - 1) / d21;
            const R t = one / (std::real(d11 * d22) - one);
            d21 = t / d21;

            // Update elements in columns A(k-1) and A(k) as dot products of
            // rows of ( W(kw-1) W(kw) ) and columns of D**(-1).
            for (blasint j = 1; j <= k - 2; ++j) {
              A(j, k - 1) = d21 * (d11 * W(j, kw - 1) - W(j, kw));
              A(j, k) = Conj(d21) * (d22 * W(j, kw) - W(j, kw - 1));
            }
          }

          // Copy D(k) to A.
          A(k - 1, k - 1) = W(k - 1, kw - 1);
          A(k - 1, k) = W(k - 1, kw);
          A(k, k) = W(k, kw);

          // (2) Conjugate columns W(kw) and W(kw-1).
          Lacgv(k - 1, W.ptr(1, kw), 1);
          Lacgv(k - 2, W.ptr(1, kw - 1), 1);
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k - 1) = -kp;
      }

      // Decrease K and return to the start of the main loop.
      k = k - kstep;
    }

    // Update the upper triangle of A11 (= A(1:k,1:k)) as
    //
    // A11 := A11 - U12*D*U12**H = A11 - U12*W**H
    //
    // (note that conjg(W) is actually stored)
    Gemmtr('U', 'N', 'T', k, n - k, -cone, A.ptr(1, k + 1), lda,
           W.ptr(1, kw + 1), ldw, cone, A.ptr(1, 1), lda);

    // Put U12 in standard form by partially undoing the interchanges in
    // columns k+1:n looping backwards from k+1 to n.
    blasint j = k + 1;
    do {
      // Undo the interchanges (if any) of rows JJ and JP at each step J.
      //
      // (Here, J is a diagonal index.)
      const blasint jj = j;
      blasint jp = IPIV(j);
      if (jp < 0) {
        jp = -jp;
        // (Here, J is a diagonal index.)
        j = j + 1;
      }
      // (NOTE: Here, J is used to determine row length. Length N-J+1 of the
      // rows to swap back doesn't include diagonal element.)
      j = j + 1;
      if (jp != jj && j <= n) {
        Swap(n - j + 1, A.ptr(jp, j), lda, A.ptr(jj, j), lda);
      }
    } while (j < n);

    // Set KB to the number of columns factorized.
    *kb = n - k;
  } else {
    // Factorize the leading columns of A using the lower triangle of A and
    // working forwards, and compute the matrix W = L21*D for use in
    // updating A22 (note that conjg(W) is actually stored).
    //
    // K is the main loop index, increasing from 1 in steps of 1 or 2.
    blasint k = 1;
    for (;;) {
      // Exit from loop.
      if ((k >= nb && nb < n) || k > n) break;

      blasint kstep = 1;

      // Copy column K of A to column K of W and update it.
      W(k, k) = std::real(A(k, k));
      if (k < n) Copy(n - k, A.ptr(k + 1, k), 1, W.ptr(k + 1, k), 1);
      Gemv('N', n - k + 1, k - 1, -cone, A.ptr(k, 1), lda, W.ptr(k, 1), ldw,
           cone, W.ptr(k, k), 1);
      W(k, k) = std::real(W(k, k));

      // Determine rows and columns to be interchanged and whether a 1-by-1
      // or 2-by-2 pivot block will be used.
      const R absakk = std::abs(std::real(W(k, k)));

      // IMAX is the row-index of the largest off-diagonal element in column
      // K, and COLMAX is its absolute value. Determine both COLMAX and
      // IMAX.
      R colmax;
      if (k < n) {
        imax = k + Iamax(n - k, W.ptr(k + 1, k), 1);
        colmax = Cabs1(W(imax, k));
      } else {
        colmax = zero;
      }

      blasint kp;
      if (Max(absakk, colmax) == zero) {
        // Column K is zero or underflow: set INFO and continue.
        if (info == 0) info = k;
        kp = k;
        A(k, k) = std::real(A(k, k));
      } else {
        // Test for interchange.
        if (absakk >= alpha * colmax) {
          // No interchange, use 1-by-1 pivot block.
          kp = k;
        } else {
          // Copy column IMAX to column K+1 of W and update it.
          Copy(imax - k, A.ptr(imax, k), lda, W.ptr(k, k + 1), 1);
          Lacgv(imax - k, W.ptr(k, k + 1), 1);
          W(imax, k + 1) = std::real(A(imax, imax));
          if (imax < n) {
            Copy(n - imax, A.ptr(imax + 1, imax), 1, W.ptr(imax + 1, k + 1), 1);
          }
          Gemv('N', n - k + 1, k - 1, -cone, A.ptr(k, 1), lda, W.ptr(imax, 1),
               ldw, cone, W.ptr(k, k + 1), 1);
          W(imax, k + 1) = std::real(W(imax, k + 1));

          // JMAX is the column-index of the largest off-diagonal element in
          // row IMAX, and ROWMAX is its absolute value. Determine only
          // ROWMAX.
          blasint jmax = k - 1 + Iamax(imax - k, W.ptr(k, k + 1), 1);
          R rowmax = Cabs1(W(jmax, k + 1));
          if (imax < n) {
            jmax = imax + Iamax(n - imax, W.ptr(imax + 1, k + 1), 1);
            rowmax = Max(rowmax, Cabs1(W(jmax, k + 1)));
          }

          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            // No interchange, use 1-by-1 pivot block.
            kp = k;
          } else if (std::abs(std::real(W(imax, k + 1))) >= alpha * rowmax) {
            // Interchange rows and columns K and IMAX, use 1-by-1 pivot
            // block.
            kp = imax;

            // Copy column K+1 of W to column K of W.
            Copy(n - k + 1, W.ptr(k, k + 1), 1, W.ptr(k, k), 1);
          } else {
            // Interchange rows and columns K+1 and IMAX, use 2-by-2 pivot
            // block.
            kp = imax;
            kstep = 2;
          }
        }

        // KK is the column of A where pivoting step stopped.
        const blasint kk = k + kstep - 1;

        // Interchange rows and columns KP and KK. Updated column KP is
        // already stored in column KK of W.
        if (kp != kk) {
          // Copy non-updated column KK to column KP of submatrix A at step
          // K. No need to copy element into column K (or K and K+1 for
          // 2-by-2 pivot) of A, since these columns will be later
          // overwritten.
          A(kp, kp) = std::real(A(kk, kk));
          Copy(kp - kk - 1, A.ptr(kk + 1, kk), 1, A.ptr(kp, kk + 1), lda);
          Lacgv(kp - kk - 1, A.ptr(kp, kk + 1), lda);
          if (kp < n) {
            Copy(n - kp, A.ptr(kp + 1, kk), 1, A.ptr(kp + 1, kp), 1);
          }

          // Interchange rows KK and KP in first K-1 columns of A (columns K
          // (or K and K+1 for 2-by-2 pivot) of A will be later
          // overwritten). Interchange rows KK and KP in first KK columns of
          // W.
          if (k > 1) Swap(k - 1, A.ptr(kk, 1), lda, A.ptr(kp, 1), lda);
          Swap(kk, W.ptr(kk, 1), ldw, W.ptr(kp, 1), ldw);
        }

        if (kstep == 1) {
          // 1-by-1 pivot block D(k): column k of W now holds
          //
          // W(k) = L(k)*D(k),
          //
          // where L(k) is the k-th column of L.
          //
          // (1) Store subdiag. elements of column L(k) and 1-by-1 block D(k)
          //     in column k of A. (NOTE: Diagonal element L(k,k) is a UNIT
          //     element and not stored.)
          //        A(k,k) := D(k,k) = W(k,k)
          //        A(k+1:N,k) := L(k+1:N,k) = W(k+1:N,k)/D(k,k)
          //
          //     (NOTE: Diagonal element D(k,k) is a REAL element, but we
          //     use a complex copy to store it.)
          Copy(n - k + 1, W.ptr(k, k), 1, A.ptr(k, k), 1);
          if (k < n) {
            // Store L(k) in column k of A.
            const R r1 = one / std::real(A(k, k));
            Rscal(n - k, r1, A.ptr(k + 1, k), 1);

            // (2) Conjugate column W(k).
            Lacgv(n - k, W.ptr(k + 1, k), 1);
          }
        } else {
          // 2-by-2 pivot block D(k): columns k and k+1 of W now hold
          //
          // ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
          //
          // where L(k) and L(k+1) are the k-th and (k+1)-th columns of L.
          //
          // (1) Store L(k+2:N,k) and L(k+2:N,k+1) and 2-by-2 block
          //     D(k:k+1,k:k+1) in columns k and k+1 of A. (NOTE: 2-by-2
          //     diagonal block L(k:k+1,k:k+1) is a UNIT block and not
          //     stored.)
          //        A(k:k+1,k:k+1) := D(k:k+1,k:k+1) = W(k:k+1,k:k+1)
          //        A(k+2:N,k:k+1) := L(k+2:N,k:k+1) =
          //        = W(k+2:N,k:k+1) * ( D(k:k+1,k:k+1)**(-1) )
          if (k < n - 1) {
            // Factor out the columns of the inverse of 2-by-2 pivot block D,
            // so that each column contains 1, to reduce the number of FLOPS
            // when we multiply panel ( W(k) W(k+1) ) by this inverse (as in
            // the upper case, here with D11 = W(k+1,k+1)/d21 and
            // D22 = W(k,k)/conj(d21)).
            T d21 = W(k + 1, k);
            const T d11 = W(k + 1, k + 1) / d21;
            const T d22 = W(k, k) / Conj(d21);
            const R t = one / (std::real(d11 * d22) - one);
            d21 = t / d21;

            // Update elements in columns A(k) and A(k+1) as dot products of
            // rows of ( W(k) W(k+1) ) and columns of D**(-1).
            for (blasint j = k + 2; j <= n; ++j) {
              A(j, k) = Conj(d21) * (d11 * W(j, k) - W(j, k + 1));
              A(j, k + 1) = d21 * (d22 * W(j, k + 1) - W(j, k));
            }
          }

          // Copy D(k) to A.
          A(k, k) = W(k, k);
          A(k + 1, k) = W(k + 1, k);
          A(k + 1, k + 1) = W(k + 1, k + 1);

          // (2) Conjugate columns W(k) and W(k+1).
          Lacgv(n - k, W.ptr(k + 1, k), 1);
          Lacgv(n - k - 1, W.ptr(k + 2, k + 1), 1);
        }
      }

      // Store details of the interchanges in IPIV.
      if (kstep == 1) {
        IPIV(k) = kp;
      } else {
        IPIV(k) = -kp;
        IPIV(k + 1) = -kp;
      }

      // Increase K and return to the start of the main loop.
      k = k + kstep;
    }

    // Update the lower triangle of A22 (= A(k:n,k:n)) as
    //
    // A22 := A22 - L21*D*L21**H = A22 - L21*W**H
    //
    // (note that conjg(W) is actually stored)
    Gemmtr('L', 'N', 'T', n - k + 1, k - 1, -cone, A.ptr(k, 1), lda,
           W.ptr(k, 1), ldw, cone, A.ptr(k, k), lda);

    // Put L21 in standard form by partially undoing the interchanges of
    // rows in columns 1:k-1 looping backwards from k-1 to 1.
    blasint j = k - 1;
    do {
      // Undo the interchanges (if any) of rows JJ and JP at each step J.
      //
      // (Here, J is a diagonal index.)
      const blasint jj = j;
      blasint jp = IPIV(j);
      if (jp < 0) {
        jp = -jp;
        // (Here, J is a diagonal index.)
        j = j - 1;
      }
      // (NOTE: Here, J is used to determine row length. Length J of the
      // rows to swap back doesn't include diagonal element.)
      j = j - 1;
      if (jp != jj && j >= 1) Swap(j, A.ptr(jp, 1), lda, A.ptr(jj, 1), lda);
    } while (j > 1);

    // Set KB to the number of columns factorized.
    *kb = k - 1;
  }
  return info;
}

// xHETRF (complex T only): computes the Bunch-Kaufman factorization
// A = U * D * U**H or L * D * L**H of the Hermitian matrix A of order N,
// blocked with xLAHEF and xHETF2, like xSYTRF (see there for the arguments
// and INFO).
template <typename T>
blasint Hetrf(char uplo, blasint n, T* a, blasint lda, blasint* ipiv, T* work,
              blasint lwork) {
  return SytrfDriver("HETRF", uplo, n, a, lda, ipiv, work, lwork, &Lahef<T>,
                     &Hetf2<T>);
}

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                      \
  template void Symv<T>(char, blasint, T, const T*, blasint, const T*,     \
                        blasint, T, T*, blasint);                          \
  template void Syr<T>(char, blasint, T, const T*, blasint, T*, blasint);  \
  template blasint Sytf2<T>(char, blasint, T*, blasint, blasint*);         \
  template blasint Lasyf<T>(char, blasint, blasint, blasint*, T*, blasint, \
                            blasint*, T*, blasint);                        \
  template blasint Sytrf<T>(char, blasint, T*, blasint, blasint*, T*, blasint);
HWYBLAS_LAPACK_FOR_EACH_TYPE(HWYBLAS_LAPACK_INSTANTIATE)
#undef HWYBLAS_LAPACK_INSTANTIATE

#define HWYBLAS_LAPACK_INSTANTIATE(T)                                      \
  template blasint Hetf2<T>(char, blasint, T*, blasint, blasint*);         \
  template blasint Lahef<T>(char, blasint, blasint, blasint*, T*, blasint, \
                            blasint*, T*, blasint);                        \
  template blasint Hetrf<T>(char, blasint, T*, blasint, blasint*, T*, blasint);
HWYBLAS_LAPACK_INSTANTIATE(c32)
HWYBLAS_LAPACK_INSTANTIATE(c64)
#undef HWYBLAS_LAPACK_INSTANTIATE

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::c32;
using hwyblas::c64;
using hwyblas::lapack::As;

extern "C" {

#define HWYBLAS_LAPACK_SYTRF(prefix, T, R)                                    \
  HWYBLAS_API void prefix##sytf2_(const char* uplo, const hwyblas_int* n,     \
                                  R* a, const hwyblas_int* lda,               \
                                  hwyblas_int* ipiv, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Sytf2(*uplo, *n, As<T>(a), *lda, ipiv);          \
  }                                                                           \
  HWYBLAS_API void prefix##lasyf_(                                            \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nb,          \
      hwyblas_int* kb, R* a, const hwyblas_int* lda, hwyblas_int* ipiv, R* w, \
      const hwyblas_int* ldw, hwyblas_int* info) {                            \
    *info = hwyblas::lapack::Lasyf(*uplo, *n, *nb, kb, As<T>(a), *lda, ipiv,  \
                                   As<T>(w), *ldw);                           \
  }                                                                           \
  HWYBLAS_API void prefix##sytrf_(                                            \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,   \
      hwyblas_int* ipiv, R* work, const hwyblas_int* lwork,                   \
      hwyblas_int* info) {                                                    \
    *info = hwyblas::lapack::Sytrf(*uplo, *n, As<T>(a), *lda, ipiv,           \
                                   As<T>(work), *lwork);                      \
  }

HWYBLAS_LAPACK_SYTRF(s, float, float)
HWYBLAS_LAPACK_SYTRF(d, double, double)
HWYBLAS_LAPACK_SYTRF(c, c32, float)
HWYBLAS_LAPACK_SYTRF(z, c64, double)

#undef HWYBLAS_LAPACK_SYTRF

#define HWYBLAS_LAPACK_HETRF(prefix, T, R)                                    \
  HWYBLAS_API void prefix##symv_(                                             \
      const char* uplo, const hwyblas_int* n, const R* alpha, const R* a,     \
      const hwyblas_int* lda, const R* x, const hwyblas_int* incx,            \
      const R* beta, R* y, const hwyblas_int* incy) {                         \
    hwyblas::lapack::Symv(*uplo, *n, *As<const T>(alpha), As<const T>(a),     \
                          *lda, As<const T>(x), *incx, *As<const T>(beta),    \
                          As<T>(y), *incy);                                   \
  }                                                                           \
  HWYBLAS_API void prefix##syr_(                                              \
      const char* uplo, const hwyblas_int* n, const R* alpha, const R* x,     \
      const hwyblas_int* incx, R* a, const hwyblas_int* lda) {                \
    hwyblas::lapack::Syr(*uplo, *n, *As<const T>(alpha), As<const T>(x),      \
                         *incx, As<T>(a), *lda);                              \
  }                                                                           \
  HWYBLAS_API void prefix##hetf2_(const char* uplo, const hwyblas_int* n,     \
                                  R* a, const hwyblas_int* lda,               \
                                  hwyblas_int* ipiv, hwyblas_int* info) {     \
    *info = hwyblas::lapack::Hetf2(*uplo, *n, As<T>(a), *lda, ipiv);          \
  }                                                                           \
  HWYBLAS_API void prefix##lahef_(                                            \
      const char* uplo, const hwyblas_int* n, const hwyblas_int* nb,          \
      hwyblas_int* kb, R* a, const hwyblas_int* lda, hwyblas_int* ipiv, R* w, \
      const hwyblas_int* ldw, hwyblas_int* info) {                            \
    *info = hwyblas::lapack::Lahef(*uplo, *n, *nb, kb, As<T>(a), *lda, ipiv,  \
                                   As<T>(w), *ldw);                           \
  }                                                                           \
  HWYBLAS_API void prefix##hetrf_(                                            \
      const char* uplo, const hwyblas_int* n, R* a, const hwyblas_int* lda,   \
      hwyblas_int* ipiv, R* work, const hwyblas_int* lwork,                   \
      hwyblas_int* info) {                                                    \
    *info = hwyblas::lapack::Hetrf(*uplo, *n, As<T>(a), *lda, ipiv,           \
                                   As<T>(work), *lwork);                      \
  }

HWYBLAS_LAPACK_HETRF(c, c32, float)
HWYBLAS_LAPACK_HETRF(z, c64, double)

#undef HWYBLAS_LAPACK_HETRF

}  // extern "C"
