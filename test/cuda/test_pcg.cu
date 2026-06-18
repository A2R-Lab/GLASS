// test_pcg.cu — dispatch glass::pcg and print the solution + iters.
// Usage: ./test_pcg solve simple <SS> <KP> <threads> <max_iters> <rel_tol> <abs_tol>
//                    <S.bin> <Pinv.bin> <b.bin>
//   S.bin, Pinv.bin : KP * 3*SS*SS floats ([L|D|R] row-major strips)
//   b.bin           : (KP+2)*SS floats (padded rhs)
// Prints two lines: the padded solution x, then the iteration count.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

#define DEFINE_PCG(SS, KP)                                                        \
    __global__ void k_pcg_##SS##_##KP(float* x, float* S, float* Pinv, float* b, \
                                      int max_iters, float rel_tol, float abs_tol,\
                                      int* d_iters) {                             \
        extern __shared__ float s_mem[];                                          \
        uint32_t it = 0;                                                          \
        glass::pcg<float, SS, KP>(x, S, Pinv, b, s_mem,                    \
            (uint32_t)max_iters, rel_tol, abs_tol, &it);                          \
        if (threadIdx.x == 0) *d_iters = (int)it;                                 \
    }

DEFINE_PCG(2, 3)
DEFINE_PCG(6, 4)

int main(int argc, char** argv) {
    if (argc < 12) {
        fprintf(stderr,
            "Usage: %s solve simple <SS> <KP> <threads> <max_iters> <rel_tol> "
            "<abs_tol> <S.bin> <Pinv.bin> <b.bin>\n", argv[0]);
        return 1;
    }
    int   SS        = atoi(argv[3]);
    int   KP        = atoi(argv[4]);
    int   threads   = atoi(argv[5]);
    int   max_iters = atoi(argv[6]);
    float rel_tol   = atof(argv[7]);
    float abs_tol   = atof(argv[8]);
    const char* S_path    = argv[9];
    const char* Pinv_path = argv[10];
    const char* b_path    = argv[11];

    int VEC    = (KP + 2) * SS;
    int band_n = KP * 3 * SS * SS;

    float* dS    = read_device_vec(S_path, band_n);
    float* dPinv = read_device_vec(Pinv_path, band_n);
    float* db    = read_device_vec(b_path, VEC);
    float* dx    = alloc_device_vec(VEC);          // zero initial guess
    int*   d_iters;
    cudaMalloc(&d_iters, sizeof(int));

    size_t smem = 0;
#define DISPATCH(SS_, KP_)                                                        \
    if (SS == SS_ && KP == KP_) {                                                 \
        smem = (size_t)glass::pcg_smem_size<float, SS_, KP_>((uint32_t)threads) \
               * sizeof(float);                                                   \
        k_pcg_##SS_##_##KP_<<<1, threads, smem>>>(dx, dS, dPinv, db,              \
            max_iters, rel_tol, abs_tol, d_iters);                               \
    }

    DISPATCH(2, 3)
    else DISPATCH(6, 4)
    else { fprintf(stderr, "unsupported size %d,%d\n", SS, KP); return 1; }

    cudaDeviceSynchronize();
    print_device_vec(dx, VEC);
    int h_iters = 0;
    cudaMemcpy(&h_iters, d_iters, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_iters);
    return 0;
}
