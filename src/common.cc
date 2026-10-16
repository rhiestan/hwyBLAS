// Target-independent support code: error reporting and scratch memory.
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hwy/aligned_allocator.h"
#include "hwy/targets.h"
#include "hwyblas/cblas.h"

namespace hwyblas {

namespace {
thread_local const CblasContext* g_cblas_context = nullptr;
}  // namespace

void SetCblasContext(const CblasContext* context) { g_cblas_context = context; }

void ReportError(const char* routine, blasint info) {
  if (const CblasContext* c = g_cblas_context) {
    cblas_xerbla(c->positions[info - 1], c->name, "");
    return;
  }
  xerbla_(routine, &info, strlen(routine));
}

void* ScratchBuffer(size_t slot, size_t bytes) {
  struct Buffer {
    hwy::AlignedFreeUniquePtr<uint8_t[]> data;
    size_t capacity = 0;
  };
  thread_local Buffer buffers[kNumScratchSlots];

  Buffer& buf = buffers[slot];
  if (buf.capacity < bytes) {
    buf.data.reset();  // free first to reduce peak memory
    buf.data = hwy::AllocateAligned<uint8_t>(bytes);
    if (!buf.data) {
      fprintf(stderr, "hwyBLAS: failed to allocate %zu bytes\n", bytes);
      abort();
    }
    buf.capacity = bytes;
  }
  return buf.data.get();
}

}  // namespace hwyblas

// Reference-BLAS compatible error handler. Unlike the reference version it
// does not stop the program: the routine that detected the error returns
// without modifying its outputs.
extern "C" HWYBLAS_API void xerbla_(const char* srname, const hwyblas_int* info,
                                    size_t srname_len) {
  // srname is a Fortran string: not necessarily NUL-terminated.
  size_t len = 0;
  while (len < srname_len && srname[len] != '\0' && srname[len] != ' ') ++len;
  fprintf(stderr,
          " ** On entry to %.*s parameter number %d had an illegal value\n",
          static_cast<int>(len), srname, static_cast<int>(*info));
}

extern "C" HWYBLAS_API int64_t hwyblas_available_targets(void) {
  // This translation unit is compiled with the same HWY_DISABLED_TARGETS as
  // the kernels, so HWY_TARGETS matches the set of compiled targets.
  return hwy::SupportedTargets() & HWY_TARGETS;
}

extern "C" HWYBLAS_API void hwyblas_force_target(int64_t target) {
  hwy::SetSupportedTargetsForTest(target);
}
