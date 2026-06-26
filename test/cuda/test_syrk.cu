// test_syrk.cu — dispatch glass::syrk / glass::syr2k and print float32 C.
//
// Usage:
//   ./test_syrk <op> <THREADS> <n> <k> <FILL> <TRANSPOSE> <ROW_MAJOR> <alpha> <beta> <A.bin> [<B.bin>] <C.bin>
//     op        : syrk | syr2k
//     THREADS   : block thread count (launched as <<<1, THREADS>>>)
//     FILL      : 0=Lower 1=Upper 2=Full
//     TRANSPOSE     : 0 | 1
//     ROW_MAJOR : 0 | 1
//   syrk  : A is n*k (TRANSPOSE=0) or k*n (TRANSPOSE=1); C is n*n. files: A C
//   syr2k : same shapes for A and B; C is n*n.       files: A B C
//
// Prints C (n*n) on one line.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

// ─── syrk kernel (runtime n,k; flags as template params) ─────────────────────
template <glass::FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__global__ void k_syrk(uint32_t n, uint32_t k, float alpha, float* A, float beta, float* C) {
    glass::syrk<float, FILL, TRANSPOSE, ROW_MAJOR>(n, k, alpha, A, beta, C);
}

template <glass::FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__global__ void k_syr2k(uint32_t n, uint32_t k, float alpha, float* A, float* B, float beta, float* C) {
    glass::syr2k<float, FILL, TRANSPOSE, ROW_MAJOR>(n, k, alpha, A, B, beta, C);
}

// ─── warp forms (compile-time N,K; one 32-lane warp) ─────────────────────────
template <uint32_t N, uint32_t K, glass::FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__global__ void k_syrk_warp(float alpha, float* A, float beta, float* C) {
    glass::warp::syrk<float, N, K, FILL, TRANSPOSE, ROW_MAJOR>(alpha, A, beta, C);
}
template <uint32_t N, uint32_t K, glass::FillMode FILL, bool TRANSPOSE, bool ROW_MAJOR>
__global__ void k_syr2k_warp(float alpha, float* A, float* B, float beta, float* C) {
    glass::warp::syr2k<float, N, K, FILL, TRANSPOSE, ROW_MAJOR>(alpha, A, B, beta, C);
}

// ─── flag dispatch: select template instantiation from runtime ints ──────────
static glass::FillMode fill_of(int f) {
    return (f == 0) ? glass::FillMode::Lower : (f == 1) ? glass::FillMode::Upper : glass::FillMode::Full;
}

#define SYRK_CASE(FE, TR, RM)                                                       \
    if (fill == FE && trans == (TR) && rowmajor == (RM)) {                         \
        k_syrk<FE, TR, RM><<<1, THREADS>>>(n, k, alpha, dA, beta, dC);             \
        dispatched = true;                                                         \
    }
#define SYR2K_CASE(FE, TR, RM)                                                      \
    if (fill == FE && trans == (TR) && rowmajor == (RM)) {                         \
        k_syr2k<FE, TR, RM><<<1, THREADS>>>(n, k, alpha, dA, dB, beta, dC);        \
        dispatched = true;                                                         \
    }

#define FOR_ALL_FLAGS(MACRO)                                                       \
    MACRO(glass::FillMode::Lower, false, false) MACRO(glass::FillMode::Lower, false, true)        \
    MACRO(glass::FillMode::Lower, true,  false) MACRO(glass::FillMode::Lower, true,  true)        \
    MACRO(glass::FillMode::Upper, false, false) MACRO(glass::FillMode::Upper, false, true)        \
    MACRO(glass::FillMode::Upper, true,  false) MACRO(glass::FillMode::Upper, true,  true)        \
    MACRO(glass::FillMode::Full,  false, false) MACRO(glass::FillMode::Full,  false, true)        \
    MACRO(glass::FillMode::Full,  true,  false) MACRO(glass::FillMode::Full,  true,  true)

// warp dispatch: compile-time (N,K) from a fixed set, always one 32-lane warp.
#define WSHAPES(KER, FE, TR, RM, ...)                                              \
    if      (n==4 && k==6) KER<4,6,FE,TR,RM><<<1,32>>>(__VA_ARGS__);               \
    else if (n==6 && k==4) KER<6,4,FE,TR,RM><<<1,32>>>(__VA_ARGS__);               \
    else if (n==7 && k==6) KER<7,6,FE,TR,RM><<<1,32>>>(__VA_ARGS__);               \
    else { fprintf(stderr,"warp shape n=%u k=%u unsupported (use 4x6,6x4,7x6)\n",n,k); return 1; }
#define SYRKW_CASE(FE, TR, RM)                                                      \
    if (fill==FE && trans==(TR) && rowmajor==(RM)) { WSHAPES(k_syrk_warp,FE,TR,RM,alpha,dA,beta,dC);  dispatched=true; }
#define SYR2KW_CASE(FE, TR, RM)                                                     \
    if (fill==FE && trans==(TR) && rowmajor==(RM)) { WSHAPES(k_syr2k_warp,FE,TR,RM,alpha,dA,dB,beta,dC); dispatched=true; }

int main(int argc, char** argv) {
    if (argc < 11) {
        fprintf(stderr,
            "Usage: %s <syrk|syr2k> <THREADS> <n> <k> <FILL> <TRANSPOSE> <ROW_MAJOR> "
            "<alpha> <beta> <A.bin> [<B.bin>] <C.bin>\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];
    int THREADS    = atoi(argv[2]);
    uint32_t n     = (uint32_t)atoi(argv[3]);
    uint32_t k     = (uint32_t)atoi(argv[4]);
    glass::FillMode fill  = fill_of(atoi(argv[5]));
    bool trans     = atoi(argv[6]) != 0;
    bool rowmajor  = atoi(argv[7]) != 0;
    float alpha    = atof(argv[8]);
    float beta     = atof(argv[9]);

    bool is2  = (strcmp(op, "syr2k") == 0 || strcmp(op, "syr2k_warp") == 0);
    bool warp = (strstr(op, "_warp") != nullptr);
    bool dispatched = false;

    if (!is2) {
        float* dA = read_device_vec(argv[10], n * k);
        float* dC = read_device_vec(argv[11], n * n);
        if (warp) { FOR_ALL_FLAGS(SYRKW_CASE) } else { FOR_ALL_FLAGS(SYRK_CASE) }
        if (!dispatched) { fprintf(stderr, "bad flags\n"); return 1; }
        cudaDeviceSynchronize();
        print_device_vec(dC, n * n);
    } else {
        float* dA = read_device_vec(argv[10], n * k);
        float* dB = read_device_vec(argv[11], n * k);
        float* dC = read_device_vec(argv[12], n * n);
        if (warp) { FOR_ALL_FLAGS(SYR2KW_CASE) } else { FOR_ALL_FLAGS(SYR2K_CASE) }
        if (!dispatched) { fprintf(stderr, "bad flags\n"); return 1; }
        cudaDeviceSynchronize();
        print_device_vec(dC, n * n);
    }
    return 0;
}
