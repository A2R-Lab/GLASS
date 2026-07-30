// test_symmetrize.cu — dispatch glass::symmetrize (block / warp / cgrps),
// in-place A := 0.5*(A + Aᵀ), and print the float32 result.
//
// Usage:
//   sym <surface> <THREADS> <n> <ct> <A.bin>   -> A (n*n, column-major)
//     surface : block | warp | cgrps
//     ct      : 0 = runtime-n overload, 1 = compile-time <T, N> overload
//               (cgrps has runtime only; ct is ignored there)
//   n ∈ {1, 4, 7, 16}.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

enum { SURF_BLOCK = 0, SURF_WARP = 1, SURF_CGRPS = 2 };

template <int SURF, uint32_t N, bool CT>
__global__ void k_sym(uint32_t n, float* A) {
    if constexpr (SURF == SURF_BLOCK) {
        if constexpr (CT) glass::block::symmetrize<float, N>(A);
        else              glass::block::symmetrize<float>(n, A);
    } else if constexpr (SURF == SURF_WARP) {
        if constexpr (CT) glass::warp::symmetrize<float, N>(A);
        else              glass::warp::symmetrize<float>(n, A);
    } else {
        glass::cgrps::symmetrize<float>(n, A);
    }
}

template <uint32_t N>
static void launch(int surf, int th, bool ct, uint32_t n, float* dA) {
    if (surf == SURF_BLOCK) {
        if (ct) k_sym<SURF_BLOCK, N, true><<<1, th>>>(n, dA);
        else    k_sym<SURF_BLOCK, N, false><<<1, th>>>(n, dA);
    } else if (surf == SURF_WARP) {
        if (ct) k_sym<SURF_WARP, N, true><<<1, th>>>(n, dA);
        else    k_sym<SURF_WARP, N, false><<<1, th>>>(n, dA);
    } else {
        k_sym<SURF_CGRPS, N, false><<<1, th>>>(n, dA);
    }
}

#define SYM_SHAPES(_) _(1) _(4) _(7) _(16)

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <block|warp|cgrps> <threads> <n> <ct> <A.bin>\n", argv[0]);
        return 1;
    }
    int surf = (strcmp(argv[1], "warp") == 0) ? SURF_WARP
             : (strcmp(argv[1], "cgrps") == 0) ? SURF_CGRPS : SURF_BLOCK;
    int th = atoi(argv[2]);
    uint32_t n = atoi(argv[3]);
    bool ct = atoi(argv[4]) != 0;
    float* dA = read_device_vec(argv[5], n * n);

    bool ok = false;
    #define DS(NN) if (!ok && n == NN) { launch<NN>(surf, th, ct, n, dA); ok = true; }
    SYM_SHAPES(DS)
    #undef DS
    if (!ok) { fprintf(stderr, "bad shape n=%u\n", n); return 1; }

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "err %s\n", cudaGetErrorString(e)); return 1; }
    print_device_vec(dA, n * n);
    return 0;
}
