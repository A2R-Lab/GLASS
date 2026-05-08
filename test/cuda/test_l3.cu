// test_l3.cu — dispatch L3 GLASS operations and print float32 results to stdout
// Usage: ./test_l3 <op> <cg|simple> <m> [n] [k] [args...] [files...]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

#include "helpers.cuh"
#include "../../glass.cuh"

static int THREADS = 256;

// ─── gemm kernels ─────────────────────────────────────────────────────────────
__global__ void k_gemm_cg(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm<float, false>(m, n, k, alpha, A, B, beta, C, cgrps::this_thread_block());
}
__global__ void k_gemm_simple(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::simple::gemm<float, false>(m, n, k, alpha, A, B, beta, C);
}
__global__ void k_gemm_t_cg(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm<float, true>(m, n, k, alpha, A, B, beta, C, cgrps::this_thread_block());
}
__global__ void k_gemm_t_simple(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::simple::gemm<float, true>(m, n, k, alpha, A, B, beta, C);
}

// ─── inv kernels ──────────────────────────────────────────────────────────────
__global__ void k_inv_cg(int n, float* A, float* scratch) {
    glass::invertMatrix(n, A, scratch);
}
__global__ void k_inv_simple(int n, float* A, float* scratch) {
    glass::simple::invertMatrix(n, A, scratch);
}

// ─── chol kernels ─────────────────────────────────────────────────────────────
__global__ void k_chol_cg(int n, float* A) {
    glass::cholDecomp_InPlace(n, A);
}
__global__ void k_chol_simple(int n, float* A) {
    glass::simple::cholDecomp_InPlace(n, A);
}

// ─── trsm kernels ─────────────────────────────────────────────────────────────
__global__ void k_trsm_cg(int n, float* L, float* b) {
    glass::trsm(n, L, b);
}
__global__ void k_trsm_simple(int n, float* L, float* b) {
    glass::simple::trsm(n, L, b);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <op> <cg|simple> <dims...> [args...] [files...]\n", argv[0]);
        return 1;
    }
    const char* op  = argv[1];
    const char* ver = argv[2];
    bool cg = (strcmp(ver, "cg") == 0);

    if (strcmp(op, "gemm") == 0) {
        // argv: op ver m n k alpha beta A.bin B.bin C.bin
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        float* dA = read_device_vec(argv[8], m * n);
        float* dB = read_device_vec(argv[9], n * k);
        float* dC = read_device_vec(argv[10], m * k);
        if (cg) k_gemm_cg<<<1, THREADS>>>(m, n, k, alpha, dA, dB, beta, dC);
        else    k_gemm_simple<<<1, THREADS>>>(m, n, k, alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, m * k);

    } else if (strcmp(op, "gemm_t") == 0) {
        // C(mxn) = alpha * A(mxn) * B(nxn)^T + beta * C(mxn)
        // B must be square nxn (GLASS TRANSPOSE_B convention).
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = n;  // k == n for TRANSPOSE_B in GLASS
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        float* dA = read_device_vec(argv[8], m * n);
        float* dB = read_device_vec(argv[9], n * n);
        float* dC = read_device_vec(argv[10], m * n);
        if (cg) k_gemm_t_cg<<<1, THREADS>>>(m, n, k, alpha, dA, dB, beta, dC);
        else    k_gemm_t_simple<<<1, THREADS>>>(m, n, k, alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, m * n);

    } else if (strcmp(op, "inv") == 0) {
        // argv: op ver n A.bin
        // A must be laid out as [A | I] (2n*n floats) for Gauss-Jordan; we just invert n x n
        int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], 2 * n * n);
        float* scratch; cudaMalloc(&scratch, (2 * n + 1) * sizeof(float));
        if (cg) k_inv_cg<<<1, THREADS>>>(n, dA, scratch);
        else    k_inv_simple<<<1, THREADS>>>(n, dA, scratch);
        cudaDeviceSynchronize();
        // The right half of dA (offset n*n) is the inverse after Gauss-Jordan
        print_device_vec(dA + n * n, n * n);
        cudaFree(scratch);

    } else if (strcmp(op, "chol") == 0) {
        // argv: op ver n A.bin
        int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], n * n);
        if (cg) k_chol_cg<<<1, THREADS>>>(n, dA);
        else    k_chol_simple<<<1, THREADS>>>(n, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "trsm") == 0) {
        // argv: op ver n L.bin b.bin
        int n = atoi(argv[3]);
        float* dL = read_device_vec(argv[4], n * n);
        float* db = read_device_vec(argv[5], n);
        if (cg) k_trsm_cg<<<1, THREADS>>>(n, dL, db);
        else    k_trsm_simple<<<1, THREADS>>>(n, dL, db);
        cudaDeviceSynchronize();
        print_device_vec(db, n);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
