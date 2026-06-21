// test_iamax.cu — dispatch GLASS iamax (BLAS i_amax) and print results to stdout.
// Usage: ./test_iamax <op> <version> <n> <threads> <input.bin>
//
// ops:
//   iamax      — glass::iamax            (default, threadIdx-strided + scratch)
//   iamax_lm   — glass::low_memory::iamax (serial on thread 0, no scratch)
//   iamax_hs   — glass::high_speed::iamax (warp-shuffle + per-warp scratch)
//   iamax_val  — glass::iamax value form: prints index THEN max|x| (two lines)
//   iamax_warp — glass::warp::iamax: WARPS independent vectors, one 32-lane warp
//                each (launched <<<1, dim3(32, WARPS)>>>); arg 4 carries WARPS
//                (not a thread count) and the driver prints W indices on one line.
//
// THREADS is taken explicitly (arg 4) so the python side can sweep block sizes
// and assert thread-count invariance. The output index is a uint32_t; it is
// printed via a float cast (exact for the small n used here, all < 2^24).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

// ─── uint output print (index is exact in float for n < 2^24) ────────────────
__global__ void k_print_uint(uint32_t* d, int n) {
    for (int i = 0; i < n; i++) {
        printf("%u", d[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
}
static void print_device_uint(uint32_t* d, int n) {
    k_print_uint<<<1,1>>>(d, n);
    cudaDeviceSynchronize();
}

// ─── kernel wrappers ─────────────────────────────────────────────────────────
__global__ void k_iamax_simple(int n, float* x, uint32_t* out, float* s_temp) {
    glass::iamax(static_cast<uint32_t>(n), x, out, s_temp);
}
__global__ void k_iamax_lm(int n, float* x, uint32_t* out) {
    glass::low_memory::iamax(static_cast<uint32_t>(n), x, out);
}
__global__ void k_iamax_hs(int n, float* x, uint32_t* out, float* s_temp) {
    glass::high_speed::iamax(static_cast<uint32_t>(n), x, out, s_temp);
}
__global__ void k_iamax_val(int n, float* x, uint32_t* out, float* out_val, float* s_temp) {
    glass::iamax(static_cast<uint32_t>(n), x, out, out_val, s_temp);
}
// Warp variant: WARPS independent vectors, one 32-lane warp (threadIdx.y) each.
// Warp w owns x + w*n; the returned index (broadcast to all lanes) is written by
// lane 0 to out[w]. Many warps run independently in one block (no shared, no sync).
__global__ void k_iamax_warp(int n, int W, float* x, uint32_t* out) {
    int w = threadIdx.y;
    if (w >= W) return;
    uint32_t idx = glass::warp::iamax<float>(static_cast<uint32_t>(n), x + w*n);
    uint32_t lane = threadIdx.x & 31;
    if (lane == 0) out[w] = idx;
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <op> <version> <n> <threads> <input.bin>\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];
    // argv[2] (version) is unused — the op name already selects the variant —
    // but kept for run_op's positional contract.
    int n       = atoi(argv[3]);
    int threads = atoi(argv[4]);
    const char* infile = argv[5];

    float* dx = read_device_vec(infile, n);
    uint32_t* d_out;  cudaMalloc(&d_out, sizeof(uint32_t));
    cudaMemset(d_out, 0, sizeof(uint32_t));
    // Generous scratch: default variant needs ~2*threads floats, hs far less.
    float* d_scratch = alloc_device_vec(2 * threads + 64);

    if (strcmp(op, "iamax") == 0) {
        k_iamax_simple<<<1, threads>>>(n, dx, d_out, d_scratch);
        cudaDeviceSynchronize();
        print_device_uint(d_out, 1);

    } else if (strcmp(op, "iamax_lm") == 0) {
        k_iamax_lm<<<1, threads>>>(n, dx, d_out);
        cudaDeviceSynchronize();
        print_device_uint(d_out, 1);

    } else if (strcmp(op, "iamax_hs") == 0) {
        k_iamax_hs<<<1, threads>>>(n, dx, d_out, d_scratch);
        cudaDeviceSynchronize();
        print_device_uint(d_out, 1);

    } else if (strcmp(op, "iamax_val") == 0) {
        float* d_val; cudaMalloc(&d_val, sizeof(float));
        cudaMemset(d_val, 0, sizeof(float));
        k_iamax_val<<<1, threads>>>(n, dx, d_out, d_val, d_scratch);
        cudaDeviceSynchronize();
        print_device_uint(d_out, 1);   // line 1: index
        print_device_vec(d_val, 1);    // line 2: max|x|
        cudaFree(d_val);

    } else if (strcmp(op, "iamax_warp") == 0) {
        // arg 4 ("threads") carries WARPS for this op; data holds W*n elements.
        int W = threads;
        cudaFree(dx);
        cudaFree(d_out);
        dx = read_device_vec(infile, n * W);
        uint32_t* d_outw; cudaMalloc(&d_outw, sizeof(uint32_t) * W);
        cudaMemset(d_outw, 0, sizeof(uint32_t) * W);
        k_iamax_warp<<<1, dim3(32, W)>>>(n, W, dx, d_outw);
        cudaDeviceSynchronize();
        print_device_uint(d_outw, W);
        cudaFree(d_outw);
        cudaFree(dx);
        cudaFree(d_scratch);
        return 0;

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    cudaFree(dx);
    cudaFree(d_out);
    cudaFree(d_scratch);
    return 0;
}
