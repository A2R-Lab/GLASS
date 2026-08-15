#include "../../glass-cgrps.cuh"
#include <cstdio>

// Compile-only overload canary; see test_api_vector.cu. Numerical coverage is
// supplied by the focused L2/L3 pytest modules.
__global__ void compile_dense_contracts(bool run, float* p, uint32_t* u) {
    if (!run) return;

    glass::block::gemv(2u, 3u, 1.0f, p, p, p);
    glass::block::gemv_reduced<float, 2, 3>(1.0f, p, p, p);
    glass::warp::gemv_reduced<float, 2, 3>(1.0f, p, p, p);
    glass::block::gemv_strided<float, 2, 3>(1.0f, p, p, p);
    glass::block::ger<float, 2, 3>(1.0f, p, p, p);
    glass::block::trmv(4u, p, p, p);
    glass::block::trmv<float, 4>(p, p, p);
    glass::block::dimm<float, 2, 3>(1.0f, p, p, p);
    glass::block::gemm_strided<float, 2, 3, 4>(1.0f, p, p, p);

    glass::block::syrk(2u, 3u, 1.0f, p, p);
    glass::block::syrk<float, 2, 3>(1.0f, p, 1.0f, p);
    glass::block::syr2k(2u, 3u, 1.0f, p, p, p);
    glass::block::syr2k<float, 2, 3>(1.0f, p, p, 1.0f, p);
    glass::warp::syrk<float, 2, 3>(1.0f, p, p);
    glass::warp::syr2k<float, 2, 3>(1.0f, p, p, p);
    glass::thread::gemv<float, 2, 3>(1.0f, p, p, 1.0f, p);
    glass::thread::gemm<float, 2, 3, 4>(1.0f, p, p, 1.0f, p);
    glass::thread::syrk<float, 2, 3>(1.0f, p, 1.0f, p);
    glass::thread::syr2k<float, 2, 3>(1.0f, p, p, 1.0f, p);
    glass::warp::syrk_reduced<float, 2, 3>(1.0f, p, p);
    glass::block::syrk_reduced<float, 2, 3>(1.0f, p, p);

    glass::dot(4u, p, p);
    glass::gemv(2u, 3u, 1.0f, p, p, 1.0f, p);
    glass::gemv(2u, 3u, 1.0f, p, p, p);
    glass::gemv<float, 2, 3>(1.0f, p, p, p);

    glass::cgrps::gemv<float, 2, 3>(1.0f, p, p, 1.0f, p);
    glass::cgrps::gemv<float, 2, 3>(1.0f, p, p, p);
    glass::cgrps::ger<float, 2, 3>(1.0f, p, p, p);
    glass::cgrps::gemv_reduced<float, 2, 3>(1.0f, p, p, p);
    glass::cgrps::gemm<float, 2, 3, 4>(1.0f, p, p, p);
    glass::cgrps::syrk_reduced<float, 2, 3>(1.0f, p, p);
}

int main() {
    std::puts("1");
    return 0;
}
