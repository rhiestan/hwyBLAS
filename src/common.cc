// Target-independent support code: error reporting, scratch memory and
// target selection.
#include "common.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#include "hwy/aligned_allocator.h"
#include "hwy/targets.h"
#include "hwyblas/cblas.h"

namespace hwyblas {

namespace {
thread_local const CblasContext* g_cblas_context = nullptr;
std::atomic<hwyblas_error_handler> g_error_handler{nullptr};
}  // namespace

void SetCblasContext(const CblasContext* context) { g_cblas_context = context; }

bool CallErrorHandler(const char* name, blasint position) {
  const hwyblas_error_handler handler =
      g_error_handler.load(std::memory_order_acquire);
  if (handler == nullptr) return false;
  handler(name, position);
  return true;
}

void ReportError(const char* routine, blasint info) {
  if (const CblasContext* c = g_cblas_context) {
    const blasint position = c->positions[info - 1];
    if (!CallErrorHandler(c->name, position)) {
      cblas_xerbla(position, c->name, "");
    }
    return;
  }
  if (!CallErrorHandler(routine, info)) {
    xerbla_(routine, &info, strlen(routine));
  }
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

extern "C" HWYBLAS_API hwyblas_error_handler
hwyblas_set_error_handler(hwyblas_error_handler handler) {
  return hwyblas::g_error_handler.exchange(handler, std::memory_order_acq_rel);
}

namespace {

// Targets that are compiled into the library and that this CPU can run.
int64_t CompiledAndSupportedTargets() {
  static const int64_t targets = [] {
    int64_t supported = hwy::SupportedTargets();
    // Highway only selects AVX3_ZEN4 on AMD CPUs, but its code also runs on
    // Intel CPUs with the AVX-512 extensions of Sapphire Rapids (or under
    // Intel SDE), which is useful for testing.
    if (supported & HWY_AVX3_SPR) supported |= HWY_AVX3_ZEN4;
    // This translation unit is compiled with the same HWY_DISABLED_TARGETS
    // as the kernels, so HWY_TARGETS matches the set of compiled targets.
    return supported & HWY_TARGETS;
  }();
  return targets;
}

bool EqualsIgnoreCase(const char* a, const char* b) {
  for (; *a != '\0' && *b != '\0'; ++a, ++b) {
    if (tolower(static_cast<unsigned char>(*a)) !=
        tolower(static_cast<unsigned char>(*b))) {
      return false;
    }
  }
  return *a == *b;
}

// The target named by the environment variable HWYBLAS_TARGET (e.g. "SSE4"
// or "AVX3_ZEN4"), or 0 if it is not set or that target is not available.
int64_t TargetFromEnvironment() {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)  // getenv is safe here
#endif
  const char* name = getenv("HWYBLAS_TARGET");
  if (name == nullptr || *name == '\0') return 0;
  const int64_t available = CompiledAndSupportedTargets();
  for (int64_t bits = available; bits != 0; bits &= bits - 1) {
    const int64_t target = bits & -bits;
    if (EqualsIgnoreCase(hwy::TargetName(target), name)) return target;
  }
  fprintf(stderr,
          "hwyBLAS: HWYBLAS_TARGET=%s is not available on this CPU, using "
          "automatic target selection\n",
          name);
  return 0;
}

// The targets that automatic selection chooses from: the one named by
// HWYBLAS_TARGET, or 0 for all available targets. Set when the library is
// loaded, before any dispatch.
const int64_t g_default_targets = [] {
  const int64_t target = TargetFromEnvironment();
  if (target != 0) hwy::SetSupportedTargetsForTest(target);
  return target;
}();

}  // namespace

extern "C" HWYBLAS_API int64_t hwyblas_available_targets(void) {
  return g_default_targets != 0 ? g_default_targets
                                : CompiledAndSupportedTargets();
}

extern "C" HWYBLAS_API void hwyblas_force_target(int64_t target) {
  hwy::SetSupportedTargetsForTest(target != 0 ? target : g_default_targets);
}
