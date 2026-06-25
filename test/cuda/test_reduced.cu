// test_reduced.cu — dispatch glass::gemm_reduced (block / warp / cgrps) and
// print float32 C. Validates the contraction-parallel engine vs the serial gemm
// oracle and across thread counts.
//
// Usage:
//   ./test_reduced <surface> <THREADS> <M> <N> <K> <TRANSPOSE_B> <ROW_MAJOR>
//                  <alpha> <beta> <A.bin> <B.bin> <C.bin>
//     surface     : block | warp | cgrps
//     THREADS     : block thread count (launched as <<<1, THREADS>>>)
//     M,N,K       : A is M*N, B is N*K (N*N when TRANSPOSE_B), C is M*(TRANSPOSE_B?N:K)
//     TRANSPOSE_B : 0 | 1   (1 requires N == K)
//     ROW_MAJOR   : 0 | 1
//
// Prints C (M * Ccols) on one line, where Ccols = TRANSPOSE_B ? N : K.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"   // pulls glass.cuh (block+warp) + glass::cgrps

enum { SURF_BLOCK = 0, SURF_WARP = 1, SURF_CGRPS = 2 };

// has_beta selects the overload at runtime: true -> 4-arg (reads C, C = a*AB +
// beta*C); false -> 3-arg (overwrites C, never reads it). The python harness
// passes has_beta = (beta != 0) so beta=0 genuinely exercises the skip-C path.
template <int SURF, uint32_t M, uint32_t N, uint32_t K, bool TB, bool RM>
__global__ void k_reduced(float alpha, float* A, float* B, float beta, float* C, bool has_beta) {
    if (SURF == SURF_BLOCK) {
        if (has_beta) glass::gemm_reduced<float, M, N, K, TB, RM>(alpha, A, B, beta, C);
        else          glass::gemm_reduced<float, M, N, K, TB, RM>(alpha, A, B, C);
    } else if (SURF == SURF_WARP) {
        if (has_beta) glass::warp::gemm_reduced<float, M, N, K, TB, RM>(alpha, A, B, beta, C);
        else          glass::warp::gemm_reduced<float, M, N, K, TB, RM>(alpha, A, B, C);
    } else {
        if (has_beta) glass::cgrps::gemm_reduced<float, M, N, K, TB, RM>(alpha, A, B, beta, C);
        else          glass::cgrps::gemm_reduced<float, M, N, K, TB, RM>(alpha, A, B, C);
    }
}

template <uint32_t M, uint32_t N, uint32_t K, bool TB, bool RM>
static void launch(int surf, int threads, float alpha, float* dA, float* dB, float beta, float* dC, bool hb) {
    if      (surf == SURF_BLOCK) k_reduced<SURF_BLOCK, M, N, K, TB, RM><<<1, threads>>>(alpha, dA, dB, beta, dC, hb);
    else if (surf == SURF_WARP)  k_reduced<SURF_WARP,  M, N, K, TB, RM><<<1, threads>>>(alpha, dA, dB, beta, dC, hb);
    else                         k_reduced<SURF_CGRPS, M, N, K, TB, RM><<<1, threads>>>(alpha, dA, dB, beta, dC, hb);
}

template <uint32_t M, uint32_t N, uint32_t K>
static bool launch_flags(int surf, int threads, bool tb, bool rm,
                         float alpha, float* dA, float* dB, float beta, float* dC, bool hb) {
    if (!tb && !rm)      launch<M, N, K, false, false>(surf, threads, alpha, dA, dB, beta, dC, hb);
    else if (!tb && rm)  launch<M, N, K, false, true >(surf, threads, alpha, dA, dB, beta, dC, hb);
    else if (tb && !rm)  launch<M, N, K, true,  false>(surf, threads, alpha, dA, dB, beta, dC, hb);
    else                 launch<M, N, K, true,  true >(surf, threads, alpha, dA, dB, beta, dC, hb);
    return true;
}

// Compile-time shapes the runner understands. Includes consumer dims (14/7/21),
// partial-warp output counts, and contraction lengths spanning the 32 boundary
// (N = 33, 64) to exercise the multi-iteration lane loop + register fallback.
#define SHAPES(_) \
    _(1,1,1)   _(3,5,4)   _(5,3,6)   _(7,2,9)   _(8,8,8) \
    _(14,14,14) _(21,21,21) _(14,21,7) _(21,14,21) _(7,14,7) \
    _(2,33,2)  _(4,64,4)

int main(int argc, char** argv) {
    if (argc < 13) {
        fprintf(stderr,
            "Usage: %s <block|warp|cgrps> <THREADS> <M> <N> <K> <TRANSPOSE_B> "
            "<ROW_MAJOR> <alpha> <beta> <A.bin> <B.bin> <C.bin>\n", argv[0]);
        return 1;
    }
    const char* surf_s = argv[1];
    int surf = (strcmp(surf_s, "warp") == 0) ? SURF_WARP
             : (strcmp(surf_s, "cgrps") == 0) ? SURF_CGRPS : SURF_BLOCK;
    int threads  = atoi(argv[2]);
    uint32_t M   = (uint32_t)atoi(argv[3]);
    uint32_t N   = (uint32_t)atoi(argv[4]);
    uint32_t K   = (uint32_t)atoi(argv[5]);
    bool tb      = atoi(argv[6]) != 0;
    bool rm      = atoi(argv[7]) != 0;
    float alpha  = atof(argv[8]);
    float beta   = atof(argv[9]);

    uint32_t Ccols = tb ? N : K;
    uint32_t b_len = tb ? N * N : N * K;
    float* dA = read_device_vec(argv[10], M * N);
    float* dB = read_device_vec(argv[11], b_len);
    float* dC = read_device_vec(argv[12], M * Ccols);

    bool has_beta = (beta != 0.0f);
    bool ok = false;
    #define DISPATCH(MM,NN,KK) \
        if (!ok && M==MM && N==NN && K==KK) \
            ok = launch_flags<MM,NN,KK>(surf, threads, tb, rm, alpha, dA, dB, beta, dC, has_beta);
    SHAPES(DISPATCH)
    #undef DISPATCH

    if (!ok) { fprintf(stderr, "unsupported shape %u %u %u\n", M, N, K); return 1; }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err)); return 1; }
    print_device_vec(dC, M * Ccols);
    return 0;
}
