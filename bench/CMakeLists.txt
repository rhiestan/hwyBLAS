# The benchmark loads BLAS libraries at runtime (including hwyblas itself), so
# it does not link against them.
add_executable(blas_bench blas_bench.cc)
target_link_libraries(blas_bench PRIVATE hwyblas_options ${CMAKE_DL_LIBS})
add_dependencies(blas_bench hwyblas)
