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

# LAPACK, ported from the reference implementation (see src/lapack/).
if(HWYBLAS_LAPACK)
  target_sources(hwyblas PRIVATE
    lapack/auxiliary.cc
    lapack/environment.cc
    lapack/bdsdc.cc
    lapack/bdsqr.cc
    lapack/bidiagonal.cc
    lapack/geqp3.cc
    lapack/geqrf.cc
    lapack/gesdd.cc
    lapack/gesvd.cc
    lapack/gees.cc
    lapack/getrf.cc
    lapack/hessenberg.cc
    lapack/hseqr.cc
    lapack/householder.cc
    lapack/lapacke.cc
    lapack/lasq.cc
    lapack/norms.cc
    lapack/potrf.cc
    lapack/steqr.cc
    lapack/syev.cc
    lapack/tridiagonal.cc
    lapack/trsen.cc
    lapack/ungqr.cc
    lapack/ungtr.cc
  )
endif()

target_include_directories(hwyblas
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  # Highway's foreach_target re-includes sources as "src/<file>".
  PRIVATE ${PROJECT_SOURCE_DIR}
)
# hwyblas_options only applies to building hwyblas, not to its users (and
# is not exported).
target_link_libraries(hwyblas PRIVATE hwy::hwy
                                      $<BUILD_INTERFACE:hwyblas_options>)
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
