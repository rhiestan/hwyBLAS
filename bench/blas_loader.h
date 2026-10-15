// Loads BLAS libraries at runtime so that several implementations (hwyBLAS,
// OpenBLAS, MKL, ...) can be benchmarked side by side in one process.
#ifndef HWYBLAS_BENCH_BLAS_LOADER_H_
#define HWYBLAS_BENCH_BLAS_LOADER_H_

#include <stdio.h>

#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace hwyblas_bench {

class BlasLibrary {
 public:
  BlasLibrary(std::string name, const std::string& path)
      : name_(std::move(name)) {
#if defined(_WIN32)
    // Search the DLL's own directory for its dependencies (e.g. MKL).
    handle_ =
        LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!handle_) {
      fprintf(stderr, "Failed to load %s from %s\n", name_.c_str(),
              path.c_str());
      return;
    }
    SetSingleThreaded();
  }

  BlasLibrary(const BlasLibrary&) = delete;
  BlasLibrary& operator=(const BlasLibrary&) = delete;

  bool ok() const { return handle_ != nullptr; }
  const std::string& name() const { return name_; }

  // Returns the address of `symbol`, or nullptr.
  void* Find(const char* symbol) const {
#if defined(_WIN32)
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle_), symbol));
#else
    return dlsym(handle_, symbol);
#endif
  }

  template <typename Fn>
  Fn Get(const char* symbol) const {
    return reinterpret_cast<Fn>(Find(symbol));
  }

 private:
  // Benchmarks compare single-threaded performance: switch off threading in
  // libraries that support it.
  void SetSingleThreaded() {
    if (auto f = Get<void (*)(int)>("openblas_set_num_threads")) f(1);
    if (auto f = Get<int (*)(int)>("MKL_Set_Num_Threads_Local")) f(1);
    if (auto f = Get<void (*)(int)>("MKL_Set_Num_Threads")) f(1);
    if (auto f = Get<void (*)(int)>("bli_thread_set_num_threads")) f(1);
  }

  std::string name_;
  void* handle_ = nullptr;
};

}  // namespace hwyblas_bench

#endif  // HWYBLAS_BENCH_BLAS_LOADER_H_
