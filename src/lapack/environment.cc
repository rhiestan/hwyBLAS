// LAPACK's environment and machine-parameter routines: ILAENV (block sizes
// and other tuning parameters), IPARMQ (parameters of the nonsymmetric
// eigenvalue routines), IEEECK, LSAME and xLAMCH.
//
// Derived from LAPACK 3.12.1, Copyright (c) 1992-2023 The University of
// Tennessee and The University of Tennessee Research Foundation, The
// University of California Berkeley, and The University of Colorado Denver.
// Modified BSD license, see LICENSE-LAPACK.
#include <math.h>
#include <string.h>

#include <algorithm>
#include <atomic>

#include "hwyblas/lapack.h"
#include "src/lapack/lapack_common.h"

namespace hwyblas {
namespace lapack {
namespace {

std::atomic<hwyblas_ilaenv_function> g_ilaenv{nullptr};

// A Fortran CHARACTER*(len) variable: assigning a string truncates it or
// pads it with blanks.
template <size_t kLen>
class FortranString {
 public:
  FortranString(const char* s, size_t len) {
    memset(str_, ' ', kLen);
    memcpy(str_, s, std::min(len, kLen));
  }
  // The character at the 1-based position i.
  char& operator()(size_t i) { return str_[i - 1]; }
  // Whether the substring (first:last) (1-based, inclusive) equals `s`.
  bool Equals(size_t first, size_t last, const char* s) const {
    return strlen(s) == last - first + 1 &&
           memcmp(str_ + first - 1, s, last - first + 1) == 0;
  }

 private:
  char str_[kLen];
};

// Converts the first six characters of a routine name to upper case, as
// ILAENV and IPARMQ do. (They also handle the EBCDIC and Prime character
// sets, which C++ implementations do not use.)
template <size_t kLen>
void UpperCaseName(FortranString<kLen>& subnam) {
  const int ic = static_cast<unsigned char>(subnam(1));
  if (ic >= 97 && ic <= 122) {
    subnam(1) = static_cast<char>(ic - 32);
    for (size_t i = 2; i <= 6; ++i) {
      const int c = static_cast<unsigned char>(subnam(i));
      if (c >= 97 && c <= 122) subnam(i) = static_cast<char>(c - 32);
    }
  }
}

// ILAENV for ISPEC = 1, 2, 3 (block size, minimum block size, crossover
// point), for a name that starts with the precision letter. The table keeps
// LAPACK's structure, which has the same values in different branches.
// NOLINTBEGIN(bugprone-branch-clone)
blasint BlockParameter(blasint ispec, FortranString<16>& subnam, blasint n1,
                       blasint n2, blasint n3, blasint n4) {
  UpperCaseName(subnam);
  const char c1 = subnam(1);
  const bool sname = c1 == 'S' || c1 == 'D';
  const bool cname = c1 == 'C' || c1 == 'Z';
  if (!(cname || sname)) return 1;
  // C2 = SUBNAM(2:3), C3 = SUBNAM(4:6), C4 = C3(2:3).
  const auto c2 = [&](const char* s) { return subnam.Equals(2, 3, s); };
  const auto c3 = [&](const char* s) { return subnam.Equals(4, 6, s); };
  const auto c4 = [&](const char* s) { return subnam.Equals(5, 6, s); };
  const bool twostage = subnam(11) == '2';
  // The orthogonal/unitary routines whose block parameters are set below.
  const bool c4_in_list = c4("QR") || c4("RQ") || c4("LQ") || c4("QL") ||
                          c4("HR") || c4("TR") || c4("BR");

  if (ispec == 1) {
    // Optimal block size. (LAPACK distinguishes real and complex names
    // here, but uses the same values for both.)
    blasint nb = 1;
    if (subnam.Equals(2, 6, "LAORH")) {
      nb = 32;
    } else if (c2("GE")) {
      if (c3("TRF")) {
        nb = 64;
      } else if (c3("QRF") || c3("RQF") || c3("LQF") || c3("QLF")) {
        nb = 32;
      } else if (c3("QR ")) {
        if (n3 == 1) {
          nb = (n1 * n2 <= 131072 || n1 <= 8192) ? n1 : 32768 / n2;
        } else {
          nb = 1;
        }
      } else if (c3("LQ ")) {
        if (n3 == 2) {
          nb = (n1 * n2 <= 131072 || n1 <= 8192) ? n1 : 32768 / n2;
        } else {
          nb = 1;
        }
      } else if (c3("HRD") || c3("BRD")) {
        nb = 32;
      } else if (c3("TRI")) {
        nb = 64;
      } else if (subnam.Equals(4, 7, "QP3RK")) {
        nb = 32;
      }
    } else if (c2("PO")) {
      if (c3("TRF")) nb = 64;
    } else if (c2("SY")) {
      if (c3("TRF")) {
        nb = twostage ? 192 : 64;
      } else if (sname && c3("TRD")) {
        nb = 32;
      } else if (sname && c3("GST")) {
        nb = 64;
      }
    } else if (cname && c2("HE")) {
      if (c3("TRF")) {
        nb = twostage ? 192 : 64;
      } else if (c3("TRD")) {
        nb = 32;
      } else if (c3("GST")) {
        nb = 64;
      }
    } else if ((sname && c2("OR")) || (cname && c2("UN"))) {
      if ((subnam(4) == 'G' || subnam(4) == 'M') && c4_in_list) nb = 32;
    } else if (c2("GB")) {
      if (c3("TRF")) nb = n4 <= 64 ? 1 : 32;
    } else if (c2("PB")) {
      if (c3("TRF")) nb = n2 <= 64 ? 1 : 32;
    } else if (c2("TR")) {
      if (c3("TRI") || c3("EVC")) {
        nb = 64;
      } else if (c3("SYL")) {
        // INT(...) of an integer expression: integer division.
        if (sname) {
          nb = std::min(std::max<blasint>(48, (std::min(n1, n2) * 16) / 100),
                        blasint{240});
        } else {
          nb = std::min(std::max<blasint>(24, (std::min(n1, n2) * 8) / 100),
                        blasint{80});
        }
      }
    } else if (c2("LA")) {
      if (c3("UUM")) {
        nb = 64;
      } else if (c3("TRS")) {
        nb = 32;
      }
    } else if (sname && c2("ST")) {
      if (c3("EBZ")) nb = 1;
    } else if (c2("GG")) {
      nb = 32;  // also for xGGHD3
    }
    return nb;
  }

  if (ispec == 2) {
    // Minimum block size.
    blasint nbmin = 2;
    if (c2("GE")) {
      // QRF, RQF, LQF, QLF, HRD, BRD, TRI, QP3RK: 2.
    } else if (c2("SY")) {
      if (c3("TRF")) {
        nbmin = 8;
      } else if (sname && c3("TRD")) {
        nbmin = 2;
      }
    }
    // HE (TRD), OR/UN and GG: 2.
    return nbmin;
  }

  // ispec == 3: crossover point, below which an unblocked routine is used.
  blasint nx = 0;
  if (c2("GE")) {
    if (c3("QRF") || c3("RQF") || c3("LQF") || c3("QLF") || c3("HRD") ||
        c3("BRD") || subnam.Equals(4, 7, "QP3RK")) {
      nx = 128;
    }
  } else if (c2("SY")) {
    if (sname && c3("TRD")) nx = 32;
  } else if (cname && c2("HE")) {
    if (c3("TRD")) nx = 32;
  } else if ((sname && c2("OR")) || (cname && c2("UN"))) {
    if (subnam(4) == 'G' && c4_in_list) nx = 128;
  } else if (c2("GG")) {
    nx = 128;  // also for xGGHD3
  }
  return nx;
}
// NOLINTEND(bugprone-branch-clone)

// IEEECK(ispec, zero, one): 1 if infinity arithmetic (ispec = 0), or also
// NaN arithmetic (ispec = 1), works as IEEE 754 specifies; 0 otherwise.
// `zero` and `one` are arguments so that the computations are not folded.
blasint Ieeeck(blasint ispec, float zero, float one) {
  float posinf = one / zero;
  if (posinf <= one) return 0;
  float neginf = -one / zero;
  if (neginf >= zero) return 0;
  const float negzro = one / (neginf + one);
  if (negzro != zero) return 0;
  neginf = one / negzro;
  if (neginf >= zero) return 0;
  const float newzro = negzro + zero;
  if (newzro != zero) return 0;
  posinf = one / newzro;
  if (posinf <= one) return 0;
  neginf = neginf * posinf;
  if (neginf >= zero) return 0;
  posinf = posinf * posinf;
  if (posinf <= one) return 0;
  if (ispec == 0) return 1;
  // Each of these must be NaN, i.e. not equal to itself.
  const float nan1 = posinf + neginf;
  const float nan2 = posinf / neginf;
  const float nan3 = posinf / posinf;  // NOLINT(misc-redundant-expression)
  const float nan4 = posinf * zero;
  const float nan5 = neginf * negzro;
  const float nan6 = nan5 * zero;
  for (const float x : {nan1, nan2, nan3, nan4, nan5, nan6}) {
    // NOLINTNEXTLINE(misc-redundant-expression): the NaN test of IEEECK
    if (x == x) return 0;
  }
  return 1;
}

// IPARMQ for ISPEC = 12 to 17 (parameters of xHSEQR and xLAQR*): `n`,
// `ilo`, `ihi` and `lwork` are those of the calling routine.
blasint Iparmq(blasint ispec, const char* name, size_t name_len, blasint ilo,
               blasint ihi) {
  constexpr blasint kInmin = 12, kInwin = 13, kInibl = 14, kIshfts = 15,
                    kIacc22 = 16, kIcost = 17;
  // Crossover to the small-matrix code, deflation window size, nibble
  // crossover, and the remaining tuning constants.
  constexpr blasint kNmin = 75, kK22min = 14, kKacmin = 14, kNibble = 14,
                    kKnwswp = 500, kRcost = 10;
  blasint nh = 0, ns = 0;
  if (ispec == kIshfts || ispec == kInwin || ispec == kIacc22) {
    // Number of simultaneous shifts.
    nh = ihi - ilo + 1;
    ns = 2;
    if (nh >= 30) ns = 4;
    if (nh >= 60) ns = 10;
    if (nh >= 150) {
      // NINT(LOG(REAL(NH)) / LOG(TWO)), in single precision.
      const blasint log2nh = static_cast<blasint>(
          lroundf(logf(static_cast<float>(nh)) / logf(2.0f)));
      ns = std::max<blasint>(10, nh / log2nh);
    }
    if (nh >= 590) ns = 64;
    if (nh >= 3000) ns = 128;
    if (nh >= 6000) ns = 256;
    ns = std::max<blasint>(2, ns - ns % 2);
  }

  if (ispec == kInmin) return kNmin;
  if (ispec == kInibl) return kNibble;
  if (ispec == kIshfts) return ns;
  if (ispec == kInwin) return nh <= kKnwswp ? ns : 3 * ns / 2;
  if (ispec == kIacc22) {
    // Whether and how to use matrix multiplication to accumulate the
    // reflections of a multishift QR sweep.
    blasint iparmq = 0;
    FortranString<6> subnam(name, name_len);
    UpperCaseName(subnam);
    if (subnam.Equals(2, 6, "GGHRD") || subnam.Equals(2, 6, "GGHD3")) {
      iparmq = 1;
      if (nh >= kK22min) iparmq = 2;
    } else if (subnam.Equals(4, 6, "EXC")) {
      if (nh >= kKacmin) iparmq = 1;
      if (nh >= kK22min) iparmq = 2;
    } else if (subnam.Equals(2, 6, "HSEQR") || subnam.Equals(2, 5, "LAQR")) {
      if (ns >= kKacmin) iparmq = 1;
      if (ns >= kK22min) iparmq = 2;
    }
    return iparmq;
  }
  if (ispec == kIcost) return kRcost;
  return -1;
}

// ILAENV with the name given as a Fortran string (not NUL-terminated).
blasint IlaenvImpl(blasint ispec, const char* name, size_t name_len, blasint n1,
                   blasint n2, blasint n3, blasint n4) {
  switch (ispec) {
    case 1:
    case 2:
    case 3: {
      FortranString<16> subnam(name, name_len);
      return BlockParameter(ispec, subnam, n1, n2, n3, n4);
    }
    case 4:
      return 6;  // number of shifts (used by xHSEQR before LAPACK 3.1)
    case 5:
      return 2;  // minimum column dimension (not used)
    case 6:
      // Crossover point for the SVD (not used): INT(REAL(MIN(N1, N2)) * 1.6).
      return static_cast<blasint>(static_cast<float>(std::min(n1, n2)) * 1.6f);
    case 7:
      return 1;  // number of processors (not used)
    case 8:
      return 50;  // crossover point for the multishift QR method
    case 9:
      return 25;  // maximum size of the subproblems of divide and conquer
    case 10:
      return Ieeeck(1, 0.0f, 1.0f);  // whether NaN arithmetic is trusted
    case 11:
      return Ieeeck(0, 0.0f, 1.0f);  // whether infinity arithmetic is trusted
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
      // n2 and n3 are ILO and IHI.
      return Iparmq(ispec, name, name_len, n2, n3);
    default:
      return -1;  // invalid ISPEC
  }
}

}  // namespace

blasint Ilaenv(blasint ispec, const char* name, const char* opts, blasint n1,
               blasint n2, blasint n3, blasint n4) {
  if (const hwyblas_ilaenv_function f =
          g_ilaenv.load(std::memory_order_acquire)) {
    return f(&ispec, name, opts, &n1, &n2, &n3, &n4, strlen(name),
             strlen(opts));
  }
  return IlaenvImpl(ispec, name, strlen(name), n1, n2, n3, n4);
}

}  // namespace lapack
}  // namespace hwyblas

using hwyblas::blasint;

extern "C" {

HWYBLAS_API hwyblas_int ilaenv_(const hwyblas_int* ispec, const char* name,
                                const char* /*opts*/, const hwyblas_int* n1,
                                const hwyblas_int* n2, const hwyblas_int* n3,
                                const hwyblas_int* n4, size_t name_len,
                                size_t /*opts_len*/) {
  return hwyblas::lapack::IlaenvImpl(*ispec, name, name_len, *n1, *n2, *n3,
                                     *n4);
}

HWYBLAS_API hwyblas_int iparmq_(const hwyblas_int* ispec, const char* name,
                                const char* /*opts*/, const hwyblas_int* /*n*/,
                                const hwyblas_int* ilo, const hwyblas_int* ihi,
                                const hwyblas_int* /*lwork*/, size_t name_len,
                                size_t /*opts_len*/) {
  return hwyblas::lapack::Iparmq(*ispec, name, name_len, *ilo, *ihi);
}

HWYBLAS_API hwyblas_int ieeeck_(const hwyblas_int* ispec, const float* zero,
                                const float* one) {
  return hwyblas::lapack::Ieeeck(*ispec, *zero, *one);
}

HWYBLAS_API hwyblas_int lsame_(const char* ca, const char* cb,
                               size_t /*ca_len*/, size_t /*cb_len*/) {
  return hwyblas::ToUpper(*ca) == hwyblas::ToUpper(*cb) ? 1 : 0;
}

HWYBLAS_API float slamch_(const char* cmach, size_t /*cmach_len*/) {
  return hwyblas::lapack::Lamch<float>(*cmach);
}

HWYBLAS_API double dlamch_(const char* cmach, size_t /*cmach_len*/) {
  return hwyblas::lapack::Lamch<double>(*cmach);
}

// xLAMC3(a, b) = a + b, computed out of line (the reference implementation
// uses it to keep compilers from optimizing machine-parameter computations).
HWYBLAS_API float slamc3_(const float* a, const float* b) { return *a + *b; }
HWYBLAS_API double dlamc3_(const double* a, const double* b) { return *a + *b; }

HWYBLAS_API hwyblas_ilaenv_function
hwyblas_set_ilaenv(hwyblas_ilaenv_function function) {
  return hwyblas::lapack::g_ilaenv.exchange(function,
                                            std::memory_order_acq_rel);
}

}  // extern "C"
