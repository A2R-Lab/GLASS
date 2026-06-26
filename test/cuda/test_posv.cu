// test_posv.cu — dispatch glass::posv / glass::potrs and print float32 x.
//
// Usage:
//   ./test_posv <op> <n> <threads> <A_or_L.bin> <b.bin>
//     op : posv  — A is SPD (n*n col-major); factor + solve; b -> x
//          potrs — L is the lower Cholesky factor (n*n col-major); solve; b -> x
//     Prints x (length n) on one line.
//
//   ./test_posv <op> <n> <nrhs> <threads> <A_or_L.bin> <B.bin>
//     op : posv_m  — A is SPD (n*n col-major); factor + solve; B -> X
//          potrs_m — L is the lower factor (n*n col-major); solve; B -> X
//     B is n*nrhs col-major; prints X (length n*nrhs) on one line.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

__global__ void k_posv(uint32_t n, float* A, float* b) {
    glass::posv<float>(n, A, b);
}
__global__ void k_potrs(uint32_t n, const float* L, float* b) {
    glass::potrs<float>(n, L, b);
}
__global__ void k_posv_m(uint32_t n, uint32_t nrhs, float* A, float* B) {
    glass::posv<float>(n, nrhs, A, B);
}
__global__ void k_potrs_m(uint32_t n, uint32_t nrhs, const float* L, float* B) {
    glass::potrs<float>(n, nrhs, L, B);
}

// ─── flagged solves (compile-time N=7, NRHS=1) — REGULARIZE/CHECK/REG_DIAG ─────
// b is an n×1 column-major B, so the single-RHS flagged solve is the multi-RHS
// overload with NRHS=1. Block (thread-count invariant) + warp (single-warp).
static constexpr uint32_t FN = 7;
template <bool REG, bool CHK, bool DIAG>
__global__ void k_posv_flag(float* A, float* b, float rho, int* s_fail) {
    glass::posv<float, FN, 1, REG, CHK, DIAG>(A, b, rho, s_fail);
}
template <bool REG, bool CHK, bool DIAG>
__global__ void k_posv_warp_flag(float* A, float* b, float rho, int* s_fail) {
    glass::warp::posv<float, FN, 1, REG, CHK, DIAG>(A, b, rho, s_fail);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <posv|potrs> <n> <threads> <A_or_L.bin> <b.bin>\n"
            "       %s <posv_m|potrs_m> <n> <nrhs> <threads> <A_or_L.bin> <B.bin>\n",
            argv[0], argv[0]);
        return 1;
    }
    const char* op = argv[1];

    // Flagged path: <flagop> <n=7> <threads> <A.bin> <b.bin> <rho>
    //   prints x (length 7) on line 1 and s_fail (0/1) on line 2.
    //   flagops: posv_reg / posv_regdiag / posv_check  (block)
    //            posv_warp_reg / posv_warp_regdiag / posv_warp_check  (warp)
    if (strncmp(op, "posv_reg", 8) == 0 || strncmp(op, "posv_check", 10) == 0 ||
        strncmp(op, "posv_warp_", 10) == 0) {
        if (argc < 7) { fprintf(stderr, "flagged needs <n=7> <threads> <A.bin> <b.bin> <rho>\n"); return 1; }
        uint32_t n  = (uint32_t)atoi(argv[2]);
        int THREADS = atoi(argv[3]);
        if (n != FN) { fprintf(stderr, "flagged kernels are compiled for n=%u only\n", FN); return 1; }
        float* dA = read_device_vec(argv[4], n * n);
        float* db = read_device_vec(argv[5], n);
        float rho = (float)atof(argv[6]);
        int* d_fail; cudaMalloc(&d_fail, sizeof(int));
        cudaMemset(d_fail, 0, sizeof(int));
        bool warp = (strncmp(op, "posv_warp_", 10) == 0);
        const char* tail = warp ? op + 10 : op + 5;   // strip "posv_warp_" / "posv_"
        // tail ∈ {reg, regdiag, check}
        if      (strcmp(tail, "reg") == 0)     { warp ? k_posv_warp_flag<true,false,false><<<1,THREADS>>>(dA,db,rho,d_fail)
                                                      : k_posv_flag<true,false,false><<<1,THREADS>>>(dA,db,rho,d_fail); }
        else if (strcmp(tail, "regdiag") == 0) { warp ? k_posv_warp_flag<true,false,true><<<1,THREADS>>>(dA,db,rho,d_fail)
                                                      : k_posv_flag<true,false,true><<<1,THREADS>>>(dA,db,rho,d_fail); }
        else if (strcmp(tail, "check") == 0)   { warp ? k_posv_warp_flag<false,true,false><<<1,THREADS>>>(dA,db,rho,d_fail)
                                                      : k_posv_flag<false,true,false><<<1,THREADS>>>(dA,db,rho,d_fail); }
        else { fprintf(stderr, "bad flag op '%s'\n", op); return 1; }
        cudaDeviceSynchronize();
        print_device_vec(db, n);
        int h_fail = 0; cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
        printf("%d\n", h_fail);
        cudaFree(d_fail);
        return 0;
    }

    // Multi-RHS path: posv_m / potrs_m take an extra <nrhs> arg before <threads>.
    if (strcmp(op, "posv_m") == 0 || strcmp(op, "potrs_m") == 0) {
        if (argc < 7) { fprintf(stderr, "multi-RHS needs <n> <nrhs> <threads> <M.bin> <B.bin>\n"); return 1; }
        uint32_t n    = (uint32_t)atoi(argv[2]);
        uint32_t nrhs = (uint32_t)atoi(argv[3]);
        int THREADS   = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], n * n);
        float* dB = read_device_vec(argv[6], n * nrhs);
        if (strcmp(op, "posv_m") == 0) {
            k_posv_m<<<1, THREADS>>>(n, nrhs, dA, dB);
        } else {
            k_potrs_m<<<1, THREADS>>>(n, nrhs, dA, dB);
        }
        cudaDeviceSynchronize();
        print_device_vec(dB, n * nrhs);
        return 0;
    }

    uint32_t n   = (uint32_t)atoi(argv[2]);
    int THREADS  = atoi(argv[3]);
    float* dA = read_device_vec(argv[4], n * n);
    float* db = read_device_vec(argv[5], n);

    if (strcmp(op, "posv") == 0) {
        k_posv<<<1, THREADS>>>(n, dA, db);
    } else if (strcmp(op, "potrs") == 0) {
        k_potrs<<<1, THREADS>>>(n, dA, db);
    } else {
        fprintf(stderr, "bad op\n"); return 1;
    }
    cudaDeviceSynchronize();
    print_device_vec(db, n);
    return 0;
}
