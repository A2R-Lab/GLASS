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

// Read n float32 values from a .bin, round to int, upload to device int array.
static int* read_device_ivec(const char* path, int n) {
    float* h = read_host_vec(path, n);
    int* hi = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) hi[i] = (int)lroundf(h[i]);
    int* d; cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, hi, n * sizeof(int), cudaMemcpyHostToDevice);
    free(h); free(hi);
    return d;
}

// ─── indexed_batched_gemm kernel (DIM=4) ─────────────────────────────────────
__global__ void k_indexed_bgemm_4(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                   float* A, float* B, float* C) {
    glass::indexed_batched_gemm<float, 4>(pairs, a_idx, b_idx, c_idx, A, B, C);
}

// ─── indexed_batched_gemm: TRANSPOSE_A / TRANSPOSE_B / ATOMIC_C variants ──────
// TRANSPOSE_A: C_p = A_pᵀ · B_p (distinct c_idx, plain overwrite).
__global__ void k_indexed_bgemm_4_ta(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                      float* A, float* B, float* C) {
    glass::indexed_batched_gemm<float, 4, /*TA*/true, /*TB*/false, /*ATOMIC*/false>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}
// TRANSPOSE_B: C_p = A_p · B_pᵀ.
__global__ void k_indexed_bgemm_4_tb(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                      float* A, float* B, float* C) {
    glass::indexed_batched_gemm<float, 4, /*TA*/false, /*TB*/true, /*ATOMIC*/false>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}
// ATOMIC_C: pairs may share c_idx; products are scatter-summed into pre-zeroed C.
__global__ void k_indexed_bgemm_4_atomic(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                         float* A, float* B, float* C) {
    glass::indexed_batched_gemm<float, 4, /*TA*/false, /*TB*/false, /*ATOMIC*/true>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}
// TRANSPOSE_A + ATOMIC_C (the backward-pass case Xᵀ·M·X → shared parent): each
// child computes A_pᵀ·B_p and atomically accumulates into a SHARED parent C slot.
__global__ void k_indexed_bgemm_4_ta_atomic(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                            float* A, float* B, float* C) {
    glass::indexed_batched_gemm<float, 4, /*TA*/true, /*TB*/false, /*ATOMIC*/true>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}

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

// ─── row_strided_gemm kernels (compile-time M, N, K, A_RS, B_RS) ─────────────
// A[i][j] = A[i + j*A_RS] (col-major), B[j][l] = B[j + l*B_RS] (col-major), C standard.
__global__ void k_gemm_strided_6x6x6_6_6(float alpha, float* A, float* B, float beta, float* C) {
    glass::row_strided_gemm<float, 6, 6, 6, 6, 6>(A, B, C, alpha, beta);
}
__global__ void k_gemm_strided_6x6x6_8_8(float alpha, float* A, float* B, float beta, float* C) {
    glass::row_strided_gemm<float, 6, 6, 6, 8, 8>(A, B, C, alpha, beta);
}
__global__ void k_gemm_strided_4x4x4_4_4(float alpha, float* A, float* B, float beta, float* C) {
    glass::row_strided_gemm<float, 4, 4, 4, 4, 4>(A, B, C, alpha, beta);
}
__global__ void k_gemm_strided_4x4x4_6_6(float alpha, float* A, float* B, float beta, float* C) {
    glass::row_strided_gemm<float, 4, 4, 4, 6, 6>(A, B, C, alpha, beta);
}

// ─── packed GEMM CT kernels (4×4×{16,32,48,64}) ──────────────────────────────
#define DEFINE_PACKED_GEMM_KERNEL(M, N, K)                                              \
    __global__ void k_packed_gemm_##M##x##N##x##K(                                     \
            float alpha, float* A, float* B, float beta, float* C) {                   \
        glass::gemm<float, M, N, K>(alpha, A, B, beta, C);                             \
    }
DEFINE_PACKED_GEMM_KERNEL(4, 4, 16)
DEFINE_PACKED_GEMM_KERNEL(4, 4, 32)
DEFINE_PACKED_GEMM_KERNEL(4, 4, 48)
DEFINE_PACKED_GEMM_KERNEL(4, 4, 64)

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

    } else if (strcmp(op, "gemm_strided_6x6x6_6_6") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 6 * 6);
        float* dB = read_device_vec(argv[6], 6 * 6);
        float* dC = read_device_vec(argv[7], 6 * 6);
        k_gemm_strided_6x6x6_6_6<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 6 * 6);

    } else if (strcmp(op, "gemm_strided_6x6x6_8_8") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 6 * 8);   // LDA_A=8
        float* dB = read_device_vec(argv[6], 6 * 8);   // LDA_B=8 (N*B_RS where N=6, B_RS=8)
        float* dC = read_device_vec(argv[7], 6 * 6);
        k_gemm_strided_6x6x6_8_8<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 6 * 6);

    } else if (strcmp(op, "gemm_strided_4x4x4_4_4") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 4);
        float* dB = read_device_vec(argv[6], 4 * 4);
        float* dC = read_device_vec(argv[7], 4 * 4);
        k_gemm_strided_4x4x4_4_4<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 4);

    } else if (strcmp(op, "gemm_strided_4x4x4_6_6") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 6);   // LDA_A=6
        float* dB = read_device_vec(argv[6], 4 * 6);   // LDA_B=6
        float* dC = read_device_vec(argv[7], 4 * 4);
        k_gemm_strided_4x4x4_6_6<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 4);

    } else if (strcmp(op, "packed_gemm_4x4x16") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 4);
        float* dB = read_device_vec(argv[6], 4 * 16);
        float* dC = read_device_vec(argv[7], 4 * 16);
        k_packed_gemm_4x4x16<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 16);

    } else if (strcmp(op, "packed_gemm_4x4x32") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 4);
        float* dB = read_device_vec(argv[6], 4 * 32);
        float* dC = read_device_vec(argv[7], 4 * 32);
        k_packed_gemm_4x4x32<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 32);

    } else if (strcmp(op, "packed_gemm_4x4x48") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 4);
        float* dB = read_device_vec(argv[6], 4 * 48);
        float* dC = read_device_vec(argv[7], 4 * 48);
        k_packed_gemm_4x4x48<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 48);

    } else if (strcmp(op, "packed_gemm_4x4x64") == 0) {
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 4);
        float* dB = read_device_vec(argv[6], 4 * 64);
        float* dC = read_device_vec(argv[7], 4 * 64);
        k_packed_gemm_4x4x64<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 64);

    } else if (strcmp(op, "indexed_bgemm_4") == 0) {
        // argv: m n k pairs A_mats B_mats C_mats  a_idx b_idx c_idx A B C
        uint32_t pairs = (uint32_t)atoi(argv[6]);
        int A_mats = atoi(argv[7]);
        int B_mats = atoi(argv[8]);
        int C_mats = atoi(argv[9]);
        int MAT = 4 * 4;
        int* a_idx = read_device_ivec(argv[10], pairs);
        int* b_idx = read_device_ivec(argv[11], pairs);
        int* c_idx = read_device_ivec(argv[12], pairs);
        float* dA = read_device_vec(argv[13], A_mats * MAT);
        float* dB = read_device_vec(argv[14], B_mats * MAT);
        float* dC = alloc_device_vec(C_mats * MAT);
        k_indexed_bgemm_4<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, C_mats * MAT);

    } else if (strcmp(op, "indexed_bgemm_4_ta") == 0 ||
               strcmp(op, "indexed_bgemm_4_tb") == 0 ||
               strcmp(op, "indexed_bgemm_4_atomic") == 0 ||
               strcmp(op, "indexed_bgemm_4_ta_atomic") == 0) {
        // argv: m n k pairs A_mats B_mats C_mats  a_idx b_idx c_idx A B C
        // The atomic variants expect C to be PRE-ZEROED here (alloc_device_vec
        // zero-inits) so the scatter-add reference matches.
        uint32_t pairs = (uint32_t)atoi(argv[6]);
        int A_mats = atoi(argv[7]);
        int B_mats = atoi(argv[8]);
        int C_mats = atoi(argv[9]);
        int MAT = 4 * 4;
        int* a_idx = read_device_ivec(argv[10], pairs);
        int* b_idx = read_device_ivec(argv[11], pairs);
        int* c_idx = read_device_ivec(argv[12], pairs);
        float* dA = read_device_vec(argv[13], A_mats * MAT);
        float* dB = read_device_vec(argv[14], B_mats * MAT);
        float* dC = alloc_device_vec(C_mats * MAT);
        if (strcmp(op, "indexed_bgemm_4_ta") == 0)
            k_indexed_bgemm_4_ta<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        else if (strcmp(op, "indexed_bgemm_4_tb") == 0)
            k_indexed_bgemm_4_tb<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        else if (strcmp(op, "indexed_bgemm_4_atomic") == 0)
            k_indexed_bgemm_4_atomic<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        else
            k_indexed_bgemm_4_ta_atomic<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, C_mats * MAT);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
