#include "../../glass.cuh"
#include <cstdio>

// Compile-only overload canary; numerical factor/solve coverage lives in the
// dedicated reconstruction, residual, conditioning, and thread-sweep tests.
__global__ void compile_factor_contracts(bool run, float* p, uint32_t* u) {
    if (!run) return;
    int* fail = nullptr;
    float** mats = nullptr;

    glass::block::laswp(4u, 2u, p, u, 0u, 4u);
    glass::block::inv_dense<float, 4>(p, p, p);
    glass::block::potrs<float, 4, 2>(p, p);
    glass::block::potrf(2u, 3u, 4u, p, p);
    glass::block::potrf(2u, 3u, 4u, 4u, p, p, p);
    glass::block::eig_clamp<float, 4>(p, 0.0f, p);

    glass::potrf(4u, p, fail);
    glass::potrf(2u, u, 4u, mats);
    glass::potrf(2u, 3u, 4u, p, p);
    glass::potrf(2u, 3u, 4u, 4u, p, p, p);
    glass::trsv(4u, p, p);
    glass::posv(4u, p, p);
    glass::posv(4u, 2u, p, p, 0.0f, fail);
    glass::posv<float, 4, 2>(p, p);
}

int main() {
    std::puts("1");
    return 0;
}
