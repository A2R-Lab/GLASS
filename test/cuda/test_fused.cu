// test_fused.cu — dedicated runner for the K-way fused inv / potrf, plus the
// single-warp glass::warp::inv (and its block glass::inv baseline).
//
// CLI:
//   test_fused <inv|chol|winv|binv> <threads> K d0 d1 ... d_{K-1} MAX_DIM <file0> <file1> ... <file_{K-1}>
//
// inv:  each fileM holds an augmented column-major [A_m | I] buffer (dim_m x 2*dim_m).
//       Prints K lines, each the right half (the inverse, dim_m*dim_m) of matrix m.
// chol: each fileM holds a column-major SPD buffer (dim_m x dim_m).
//       Prints K lines, each the full matrix (dim_m*dim_m); the oracle compares np.tril.
// winv: warp-packed inversion — SAME CLI/layout as inv, but all K dims must be
//       EQUAL and in the compile-time set {4, 8, 12}. Launches ONE block of
//       dim3(32, K): warp w inverts mats[w] via glass::warp::inv<float, N> with
//       its own (2*N+1)-element span of dynamic shared scratch. <threads> is
//       ignored (geometry is fixed by K). Prints K lines like inv.
// binv: block single-matrix glass::inv(dim, A, s) baseline, K=1, at <threads>
//       threads (32 ⇒ byte-comparable against winv). Prints 1 line like inv.
//
// Builds a device float** of per-matrix device pointers and a device uint32_t* dims[],
// then launches <<<1, threads>>>.  Local uint32 upload here (does NOT touch helpers.cuh).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "glass.cuh"
#include "helpers.cuh"

__global__ void k_inv_fused(uint32_t K, const uint32_t* dims, uint32_t MAX_DIM, float** mats, float* s_temp) {
    glass::block::inv<float>(K, dims, MAX_DIM, mats, s_temp);
}
__global__ void k_chol_fused(uint32_t K, const uint32_t* dims, uint32_t MAX_DIM, float** mats) {
    glass::block::potrf<float>(K, dims, MAX_DIM, mats);
}
// Warp-packed: block of dim3(32, W); warp w = threadIdx.y inverts mats[w] with
// its OWN (2*N+1)-element span of the dynamic shared scratch.
template <uint32_t N>
__global__ void k_inv_warp(float** mats) {
    extern __shared__ float sh[];
    uint32_t w = threadIdx.y;
    glass::warp::inv<float, N>(mats[w], &sh[w * (2*N + 1)]);
}
// Block single-matrix baseline (glass::inv on one matrix at <threads> threads).
__global__ void k_inv_block(uint32_t dim, float* A, float* s_scratch) {
    glass::block::inv<float>(dim, A, s_scratch);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <inv|chol|winv|binv> <threads> K d0..d_{K-1} MAX_DIM <files...>\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];
    int threads = atoi(argv[2]);
    uint32_t K  = (uint32_t)atoi(argv[3]);

    // argv layout: [0]=prog [1]=op [2]=threads [3]=K [4..4+K-1]=dims [4+K]=MAX_DIM [5+K..]=files
    uint32_t* h_dims = (uint32_t*)malloc(K * sizeof(uint32_t));
    for (uint32_t m = 0; m < K; m++) h_dims[m] = (uint32_t)atoi(argv[4 + m]);
    uint32_t MAX_DIM = (uint32_t)atoi(argv[4 + K]);
    char** files = &argv[5 + K];

    bool is_inv  = (strcmp(op, "inv") == 0);
    bool is_winv = (strcmp(op, "winv") == 0);
    bool is_binv = (strcmp(op, "binv") == 0);
    bool aug = is_inv || is_winv || is_binv;   // all inverse ops use [A | I]

    // Upload per-matrix device buffers + collect their device pointers.
    float** h_ptrs = (float**)malloc(K * sizeof(float*));
    for (uint32_t m = 0; m < K; m++) {
        int n = (int)h_dims[m];
        int count = aug ? (2 * n * n) : (n * n);   // augmented vs plain
        h_ptrs[m] = read_device_vec(files[m], count);
    }

    // Device float** of the per-matrix pointers.
    float** d_ptrs;
    cudaMalloc(&d_ptrs, K * sizeof(float*));
    cudaMemcpy(d_ptrs, h_ptrs, K * sizeof(float*), cudaMemcpyHostToDevice);

    // Local device uint32_t* dims upload (does not use helpers.cuh).
    uint32_t* d_dims;
    cudaMalloc(&d_dims, K * sizeof(uint32_t));
    cudaMemcpy(d_dims, h_dims, K * sizeof(uint32_t), cudaMemcpyHostToDevice);

    if (is_winv) {
        // All dims equal + compile-time N; one warp per matrix, per-warp scratch span.
        uint32_t N = h_dims[0];
        for (uint32_t m = 1; m < K; m++) {
            if (h_dims[m] != N) { fprintf(stderr, "winv: all dims must be equal\n"); return 1; }
        }
        size_t smem = (size_t)K * (2 * N + 1) * sizeof(float);
        dim3 block(32, K);
        if      (N == 4)  k_inv_warp<4><<<1, block, smem>>>(d_ptrs);
        else if (N == 8)  k_inv_warp<8><<<1, block, smem>>>(d_ptrs);
        else if (N == 12) k_inv_warp<12><<<1, block, smem>>>(d_ptrs);
        else { fprintf(stderr, "winv: unsupported dim %u (need 4, 8, or 12)\n", N); return 1; }
        cudaDeviceSynchronize();
        for (uint32_t m = 0; m < K; m++) {
            print_device_vec(h_ptrs[m] + N * N, N * N);
        }
    } else if (is_binv) {
        if (K != 1) { fprintf(stderr, "binv: K must be 1\n"); return 1; }
        uint32_t n = h_dims[0];
        float* d_scratch; cudaMalloc(&d_scratch, (2 * n + 1) * sizeof(float));
        k_inv_block<<<1, threads>>>(n, h_ptrs[0], d_scratch);
        cudaDeviceSynchronize();
        print_device_vec(h_ptrs[0] + n * n, n * n);
        cudaFree(d_scratch);
    } else if (is_inv) {
        // total scratch = sum_m (2*dim_m + 1)
        uint32_t scratch_n = 0;
        for (uint32_t m = 0; m < K; m++) scratch_n += 2 * h_dims[m] + 1;
        float* d_scratch; cudaMalloc(&d_scratch, scratch_n * sizeof(float));
        k_inv_fused<<<1, threads>>>(K, d_dims, MAX_DIM, d_ptrs, d_scratch);
        cudaDeviceSynchronize();
        // print right half (inverse) of each matrix
        for (uint32_t m = 0; m < K; m++) {
            int n = (int)h_dims[m];
            print_device_vec(h_ptrs[m] + n * n, n * n);
        }
        cudaFree(d_scratch);
    } else {
        k_chol_fused<<<1, threads>>>(K, d_dims, MAX_DIM, d_ptrs);
        cudaDeviceSynchronize();
        // print full matrix of each (oracle compares np.tril)
        for (uint32_t m = 0; m < K; m++) {
            int n = (int)h_dims[m];
            print_device_vec(h_ptrs[m], n * n);
        }
    }

    cudaDeviceSynchronize();
    return 0;
}
