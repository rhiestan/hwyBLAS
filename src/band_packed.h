// Storage layouts of banded and packed matrices, column by column.
#ifndef HWYBLAS_SRC_BAND_PACKED_H_
#define HWYBLAS_SRC_BAND_PACKED_H_

#include <stddef.h>

#include "hwy/base.h"

namespace hwyblas {

// Where the stored elements of each column of a banded or packed matrix are.
// In both formats they are contiguous: column j stores rows [begin, end) and
// element (i, j) is at offset + (i - begin).
struct ColumnLayout {
  struct Column {
    size_t begin, end;  // stored rows
    size_t offset;      // of element (begin, j)
  };

  // Band storage (GBMV, SBMV, HBMV, TBMV, TBSV): a matrix with `rows` rows,
  // kl sub- and ku super-diagonals, stored with leading dimension lda >=
  // kl + ku + 1. Element (i, j) is at (ku + i - j) + j * lda. Symmetric and
  // triangular band matrices with k off-diagonals have kl = k, ku = 0
  // (lower) or kl = 0, ku = k (upper).
  static ColumnLayout Band(size_t rows, size_t kl, size_t ku, size_t lda) {
    ColumnLayout l;
    l.packed_ = false;
    l.rows_ = rows;
    l.kl_ = kl;
    l.ku_ = ku;
    l.lda_ = lda;
    return l;
  }

  // Packed storage (SPMV, HPMV, TPMV, TPSV, SPR, ...) of the `lower` or upper
  // triangle of an n x n matrix: the stored parts of the columns follow each
  // other without gaps.
  static ColumnLayout Packed(bool lower, size_t n) {
    ColumnLayout l;
    l.packed_ = true;
    l.lower_ = lower;
    l.rows_ = n;
    return l;
  }

  // Number of stored off-diagonals (n - 1 for packed storage).
  size_t OffDiagonals() const {
    return packed_ ? (rows_ == 0 ? 0 : rows_ - 1) : kl_ + ku_;
  }

  Column At(size_t j) const {
    Column c;
    if (packed_) {
      if (lower_) {  // rows [j, n); columns before hold n, n - 1, ...
        c.begin = j;
        c.end = rows_;
        c.offset = j * (2 * rows_ - j + 1) / 2;
      } else {  // rows [0, j]; columns before hold 1, 2, ..., j elements
        c.begin = 0;
        c.end = j + 1;
        c.offset = j * (j + 1) / 2;
      }
    } else {
      c.begin = (j > ku_) ? j - ku_ : 0;
      c.end = HWY_MAX(c.begin, HWY_MIN(rows_, j + kl_ + 1));
      c.offset = j * lda_ + ku_ + c.begin - j;
    }
    return c;
  }

 private:
  bool packed_ = false;
  bool lower_ = false;
  size_t rows_ = 0, kl_ = 0, ku_ = 0, lda_ = 0;
};

}  // namespace hwyblas

#endif  // HWYBLAS_SRC_BAND_PACKED_H_
