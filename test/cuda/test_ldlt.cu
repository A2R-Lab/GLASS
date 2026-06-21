// test_ldlt.cu — dispatch glass::ldlt / glass::ldlt_solve and print results.
//
// Usage:
//   ./test_ldlt ldlt        <n> <threads> <A.bin>
//       Factors the symmetric n x n matrix A (column-major) in place and prints
//       the n*n factored buffer (diagonal = D, strict-lower = unit-L).
//
//   ./test_ldlt ldlt_solve  <n> <threads> <A.bin> <b.bin>
//       Factors A then solves A x = b and prints the n-vector x.
//
// A.bin : n*n float32 (column-major). b.bin : n float32.
// Scratch is allocated as (n+1) floats (reserved for the pivot path).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

__global__ void k_ldlt(uint32_t n, float* A, float* s_temp) {
    glass::ldlt<float>(n, A, s_temp);
}

__global__ void k_ldlt_solve(uint32_t n, float* A, float* s_temp, float* b) {
    glass::ldlt<float>(n, A, s_temp);
    glass::ldlt_solve<float>(n, A, b);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <ldlt|ldlt_solve> <n> <threads> <A.bin> [b.bin]\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];
    int n       = atoi(argv[2]);
    int threads = atoi(argv[3]);
    const char* A_path = argv[4];

    float* dA     = read_device_vec(A_path, n * n);
    float* dscr;  cudaMalloc(&dscr, (n + 1) * sizeof(float));

    if (strcmp(op, "ldlt") == 0) {
        k_ldlt<<<1, threads>>>((uint32_t)n, dA, dscr);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);
    } else if (strcmp(op, "ldlt_solve") == 0) {
        if (argc < 6) { fprintf(stderr, "ldlt_solve needs <b.bin>\n"); return 1; }
        float* db = read_device_vec(argv[5], n);
        k_ldlt_solve<<<1, threads>>>((uint32_t)n, dA, dscr, db);
        cudaDeviceSynchronize();
        print_device_vec(db, n);
    } else {
        fprintf(stderr, "unknown op %s\n", op);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
