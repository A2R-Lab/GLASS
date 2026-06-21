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

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <posv|potrs> <n> <threads> <A_or_L.bin> <b.bin>\n"
            "       %s <posv_m|potrs_m> <n> <nrhs> <threads> <A_or_L.bin> <B.bin>\n",
            argv[0], argv[0]);
        return 1;
    }
    const char* op = argv[1];

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
