// test_ldlt.cu — dispatch glass::ldlt / glass::ldlt_solve and print results.
//
// Usage:
//   ./test_ldlt ldlt        <n> <threads> <pivot> <A.bin>
//       Factors the symmetric n x n matrix A (column-major) in place and prints
//       the n*n factored buffer (diagonal = D, strict-lower = unit-L). When
//       <pivot> != 0, a SECOND line is printed: the n recorded pivot indices
//       piv[0..n-1] (as floats), so the test can rebuild P for the
//       L@diag(D)@L.T == P A Pᵀ reconstruction check.
//
//   ./test_ldlt ldlt_solve  <n> <threads> <pivot> <A.bin> <b.bin>
//       Factors A (pivoted if <pivot> != 0) then solves A x = b and prints x.
//
// A.bin : n*n float32 (column-major). b.bin : n float32.
// <pivot> : 0 = non-pivoted (piv = nullptr), 1 = symmetric 1x1 pivoting.
// Scratch is allocated as (n+1) floats (used by the pivot path's argmax).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

__global__ void k_ldlt(uint32_t n, float* A, float* s_temp, int pivot, uint32_t* piv) {
    glass::ldlt<float>(n, A, s_temp, pivot != 0, pivot != 0 ? piv : nullptr);
}

__global__ void k_ldlt_solve(uint32_t n, float* A, float* s_temp, float* b,
                             int pivot, uint32_t* piv) {
    glass::ldlt<float>(n, A, s_temp, pivot != 0, pivot != 0 ? piv : nullptr);
    glass::ldlt_solve<float>(n, A, b, pivot != 0 ? piv : nullptr);
}

// Print n device uint32 values (the pivot array) as space-separated ints.
__global__ void print_piv_kernel(uint32_t* d, int n) {
    for (int i = 0; i < n; i++) {
        printf("%u", d[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <ldlt|ldlt_solve> <n> <threads> <pivot> <A.bin> [b.bin]\n",
            argv[0]);
        return 1;
    }
    const char* op = argv[1];
    int n       = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int pivot   = atoi(argv[4]);
    const char* A_path = argv[5];

    float* dA     = read_device_vec(A_path, n * n);
    float* dscr;  cudaMalloc(&dscr, (n + 1) * sizeof(float));
    uint32_t* dpiv; cudaMalloc(&dpiv, n * sizeof(uint32_t));

    if (strcmp(op, "ldlt") == 0) {
        k_ldlt<<<1, threads>>>((uint32_t)n, dA, dscr, pivot, dpiv);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);
        if (pivot) {
            print_piv_kernel<<<1, 1>>>(dpiv, n);
            cudaDeviceSynchronize();
        }
    } else if (strcmp(op, "ldlt_solve") == 0) {
        if (argc < 7) { fprintf(stderr, "ldlt_solve needs <b.bin>\n"); return 1; }
        float* db = read_device_vec(argv[6], n);
        k_ldlt_solve<<<1, threads>>>((uint32_t)n, dA, dscr, db, pivot, dpiv);
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
