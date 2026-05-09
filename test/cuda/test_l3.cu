// test_l3.cu — dispatch L3 GLASS operations and print float32 results to stdout
// Usage: ./test_l3 <op> <cg|simple> <m> [n] [k] [args...] [files...]
//
// Versions:
//   cg     — glass::cgrps:: (cooperative groups)
//   simple — glass::        (threadIdx, default)

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

static int THREADS = 256;

// ─── gemm kernels ─────────────────────────────────────────────────────────────
__global__ void k_gemm_cg(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::cgrps::gemm<float, false>(m, n, k, alpha, A, B, beta, C);
}
__global__ void k_gemm_simple(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm<float, false>(m, n, k, alpha, A, B, beta, C);
}
__global__ void k_gemm_t_cg(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::cgrps::gemm<float, true>(m, n, k, alpha, A, B, beta, C);
}
__global__ void k_gemm_t_simple(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm<float, true>(m, n, k, alpha, A, B, beta, C);
}

// ─── inv kernels ──────────────────────────────────────────────────────────────
__global__ void k_inv_cg(int n, float* A, float* scratch) {
    glass::cgrps::invertMatrix(n, A, scratch);
}
__global__ void k_inv_simple(int n, float* A, float* scratch) {
    glass::invertMatrix(n, A, scratch);
}

// ─── chol kernels ─────────────────────────────────────────────────────────────
__global__ void k_chol_cg(int n, float* A) {
    glass::cgrps::cholDecomp_InPlace(n, A);
}
__global__ void k_chol_simple(int n, float* A) {
    glass::cholDecomp_InPlace(n, A);
}

// ─── trsm kernels ─────────────────────────────────────────────────────────────
__global__ void k_trsm_cg(int n, float* L, float* b) {
    glass::cgrps::trsm(n, L, b);
}
__global__ void k_trsm_simple(int n, float* L, float* b) {
    glass::trsm(n, L, b);
}

// ─── gemm row-major kernels ───────────────────────────────────────────────────
__global__ void k_gemm_rowmajor(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm<float, false, true>(m, n, k, alpha, A, B, beta, C);
}
__global__ void k_gemm_ex_mixed(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm_ex<float, false, true, false, true>(m, n, k, alpha, A, B, beta, C);
}

// ─── gemm_tiled kernels ───────────────────────────────────────────────────────
__global__ void k_gemm_tiled(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_B = smem + m * 8;
    glass::gemm_tiled<float, 8>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
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
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = n;
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
        int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], 2 * n * n);
        float* scratch; cudaMalloc(&scratch, (2 * n + 1) * sizeof(float));
        if (cg) k_inv_cg<<<1, THREADS>>>(n, dA, scratch);
        else    k_inv_simple<<<1, THREADS>>>(n, dA, scratch);
        cudaDeviceSynchronize();
        print_device_vec(dA + n * n, n * n);
        cudaFree(scratch);

    } else if (strcmp(op, "chol") == 0) {
        int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], n * n);
        if (cg) k_chol_cg<<<1, THREADS>>>(n, dA);
        else    k_chol_simple<<<1, THREADS>>>(n, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "trsm") == 0) {
        int n = atoi(argv[3]);
        float* dL = read_device_vec(argv[4], n * n);
        float* db = read_device_vec(argv[5], n);
        if (cg) k_trsm_cg<<<1, THREADS>>>(n, dL, db);
        else    k_trsm_simple<<<1, THREADS>>>(n, dL, db);
        cudaDeviceSynchronize();
        print_device_vec(db, n);

    } else if (strcmp(op, "gemm_tiled") == 0) {
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        float* dA = read_device_vec(argv[8], m * n);
        float* dB = read_device_vec(argv[9], n * k);
        float* dC = read_device_vec(argv[10], m * k);
        int smem_bytes = (m * 8 + 8 * k) * sizeof(float);
        k_gemm_tiled<<<1, THREADS, smem_bytes>>>(m, n, k, alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, m * k);

    } else if (strcmp(op, "gemm_rowmajor") == 0) {
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        float* dA = read_device_vec(argv[8], m * n);
        float* dB = read_device_vec(argv[9], n * k);
        float* dC = read_device_vec(argv[10], m * k);
        k_gemm_rowmajor<<<1, THREADS>>>(m, n, k, alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, m * k);

    } else if (strcmp(op, "gemm_ex") == 0) {
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        float* dA = read_device_vec(argv[8], m * n);
        float* dB = read_device_vec(argv[9], n * k);
        float* dC = read_device_vec(argv[10], m * k);
        k_gemm_ex_mixed<<<1, THREADS>>>(m, n, k, alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, m * k);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
