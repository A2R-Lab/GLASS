// test_syrk.cu — dispatch glass::syrk / glass::syr2k and print float32 C.
//
// Usage:
//   ./test_syrk <op> <THREADS> <n> <k> <FILL> <TRANS> <ROW_MAJOR> <alpha> <beta> <A.bin> [<B.bin>] <C.bin>
//     op        : syrk | syr2k
//     THREADS   : block thread count (launched as <<<1, THREADS>>>)
//     FILL      : 0=Lower 1=Upper 2=Full
//     TRANS     : 0 | 1
//     ROW_MAJOR : 0 | 1
//   syrk  : A is n*k (TRANS=0) or k*n (TRANS=1); C is n*n. files: A C
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
template <glass::FillMode FILL, bool TRANS, bool ROW_MAJOR>
__global__ void k_syrk(uint32_t n, uint32_t k, float alpha, float* A, float beta, float* C) {
    glass::syrk<float, FILL, TRANS, ROW_MAJOR>(n, k, alpha, A, beta, C);
}

template <glass::FillMode FILL, bool TRANS, bool ROW_MAJOR>
__global__ void k_syr2k(uint32_t n, uint32_t k, float alpha, float* A, float* B, float beta, float* C) {
    glass::syr2k<float, FILL, TRANS, ROW_MAJOR>(n, k, alpha, A, B, beta, C);
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

int main(int argc, char** argv) {
    if (argc < 11) {
        fprintf(stderr,
            "Usage: %s <syrk|syr2k> <THREADS> <n> <k> <FILL> <TRANS> <ROW_MAJOR> "
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

    bool is2 = (strcmp(op, "syr2k") == 0);
    bool dispatched = false;

    if (!is2) {
        float* dA = read_device_vec(argv[10], n * k);
        float* dC = read_device_vec(argv[11], n * n);
        FOR_ALL_FLAGS(SYRK_CASE)
        if (!dispatched) { fprintf(stderr, "bad flags\n"); return 1; }
        cudaDeviceSynchronize();
        print_device_vec(dC, n * n);
    } else {
        float* dA = read_device_vec(argv[10], n * k);
        float* dB = read_device_vec(argv[11], n * k);
        float* dC = read_device_vec(argv[12], n * n);
        FOR_ALL_FLAGS(SYR2K_CASE)
        if (!dispatched) { fprintf(stderr, "bad flags\n"); return 1; }
        cudaDeviceSynchronize();
        print_device_vec(dC, n * n);
    }
    return 0;
}
