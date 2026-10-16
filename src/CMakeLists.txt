add_library(hwyblas
  band_packed.cc
  cblas.cc
  common.cc
  gemm.cc
  gemv.cc
  level1.cc
  level2.cc
  level3.cc
  matcopy.cc
  trsm.cc
)
add_library(hwyblas::hwyblas ALIAS hwyblas)

target_include_directories(hwyblas
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  # Highway's foreach_target re-includes sources as "src/<file>".
  PRIVATE ${PROJECT_SOURCE_DIR}
)
target_link_libraries(hwyblas PRIVATE hwy::hwy hwyblas_options)
target_compile_definitions(hwyblas
  PRIVATE HWYBLAS_BUILDING
  # Skip SIMD targets that bring no benefit for floating-point BLAS (they
  # would only cost compile time) and scalable-vector targets, which the
  # fixed-size micro-kernels do not support.
  PRIVATE "HWY_DISABLED_TARGETS=(HWY_SSSE3|HWY_AVX3_DL|HWY_AVX3_SPR|HWY_AVX10_2|HWY_SVE|HWY_SVE2|HWY_SVE_256|HWY_SVE2_128|HWY_RVV)"
)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(hwyblas PRIVATE HWYBLAS_SHARED_BUILD)
endif()
set_target_properties(hwyblas PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
