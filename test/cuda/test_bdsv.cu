// test_bdsv.cu — glass::bdsv (direct block-tridiagonal SPD factor/solve).
//
// Ops:
//   bdsv     <SS> <KP> <threads> <M.bin> <b.bin>            → padded x
//   two_rhs  <SS> <KP> <threads> <M.bin> <b1.bin> <b2.bin>  → padded x1 ++ padded x2
//            (factor ONCE, bdsv_solve per RHS — proves factor reuse)
//   check    <SS> <KP> <threads> <M.bin> <b.bin>            → [fail] ++ padded x
//            (CHECK=true; M is expected non-SPD → fail=1)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

template <uint32_t KP, uint32_t SS, bool CHECK>
__global__ void k_bdsv(float* M, float* v, int* fail) {
    extern __shared__ float s[];
    glass::block::bdsv<float, KP, SS, CHECK>(M, v, s, fail);
}

template <uint32_t KP, uint32_t SS>
__global__ void k_bdsv_two_rhs(float* M, float* v1, float* v2) {
    extern __shared__ float s[];
    glass::block::bdsv_factor<float, KP, SS>(M, s);
    glass::block::bdsv_solve<float, KP, SS>(M, v1, s);
    glass::block::bdsv_solve<float, KP, SS>(M, v2, s);
}

#define BDSV_SHAPES(F) F(2,3) F(6,4) F(3,1) F(1,5) F(4,7)

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <op> <version> <SS> <KP> <threads> <M.bin> <b.bin> [b2.bin]\n", argv[0]); return 1; }
    const char* op = argv[1];
    int SS = atoi(argv[3]);
    int KP = atoi(argv[4]);
    int threads = atoi(argv[5]);
    int band_n = KP * SS * 3 * SS;
    int vec_n  = (KP + 2) * SS;
    float* dM = read_device_vec(argv[6], band_n);
    float* dv = read_device_vec(argv[7], vec_n);
    bool ok = false;

    if (strcmp(op, "bdsv") == 0 || strcmp(op, "check") == 0) {
        bool check = (strcmp(op, "check") == 0);
        int* dFail; cudaMalloc(&dFail, sizeof(int)); cudaMemset(dFail, 0, sizeof(int));
        #define DB(ss, kp) if (!ok && SS==ss && KP==kp) { \
            int sm = (int)glass::block::bdsv_scratch_bytes<float, ss>(); \
            if (check) k_bdsv<kp, ss, true ><<<1, threads, sm>>>(dM, dv, dFail); \
            else       k_bdsv<kp, ss, false><<<1, threads, sm>>>(dM, dv, dFail); ok = true; }
        BDSV_SHAPES(DB)
        #undef DB
        cudaDeviceSynchronize();
        if (check) {
            int h_fail = 0; cudaMemcpy(&h_fail, dFail, sizeof(int), cudaMemcpyDeviceToHost);
            printf("%d\n", h_fail);
        }
        print_device_vec(dv, vec_n);
    } else if (strcmp(op, "two_rhs") == 0) {
        float* dv2 = read_device_vec(argv[8], vec_n);
        #define DB2(ss, kp) if (!ok && SS==ss && KP==kp) { \
            int sm = (int)glass::block::bdsv_scratch_bytes<float, ss>(); \
            k_bdsv_two_rhs<kp, ss><<<1, threads, sm>>>(dM, dv, dv2); ok = true; }
        BDSV_SHAPES(DB2)
        #undef DB2
        cudaDeviceSynchronize();
        print_device_vec(dv, vec_n);
        print_device_vec(dv2, vec_n);
    }
    if (!ok) { fprintf(stderr, "unsupported shape SS=%d KP=%d\n", SS, KP); return 2; }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 3; }
    return 0;
}
