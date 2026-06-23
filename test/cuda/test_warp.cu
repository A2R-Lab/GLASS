// test_warp.cu — multi-warp driver for the glass::warp:: surface.
//
// Launches <<<1, dim3(32, WARPS)>>> so WARPS>=2 distinct problems run in one
// block, each owned by one warp (threadIdx.y). Each warp's data is packed
// contiguously: matrices at A + w*N*N, vectors at x/b + w*N. This is the layout
// that catches cross-warp bugs the existing single-warp tests cannot (a stray
// __syncthreads, a shared re-read shared across warps, a lane-mask leak).
//
// The trsv/posv kernels mark their pointer params __restrict__ on purpose, to
// exercise the §1g shared-reread broadcast miscompile path under -O3.
//
// Usage: ./test_warp <op> <n> <WARPS> [flags] <files...>
//   ops: dot axpy copy scal gemv gemv_t trsv posv
//   flags (trsv): <lower> <unit> <trans>  (each 0/1)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass.cuh"

// ─── L1 kernels (runtime n; one warp per problem) ────────────────────────────
__global__ void k_dot_warp(int n, int W, float* x, float* y, float* out) {
    int w = threadIdx.y;
    if (w >= W) return;
    float r = glass::warp::dot<float>((uint32_t)n, x + w*n, y + w*n);
    // every lane holds r (broadcast); lane 0 writes the per-warp result slot
    uint32_t lane = threadIdx.x & 31;
    if (lane == 0) out[w] = r;
}
__global__ void k_axpy_warp(int n, int W, float alpha, float* x, float* y) {
    int w = threadIdx.y;
    if (w >= W) return;
    glass::warp::axpy<float>((uint32_t)n, alpha, x + w*n, y + w*n);
}
__global__ void k_copy_warp(int n, int W, float* x, float* y) {
    int w = threadIdx.y;
    if (w >= W) return;
    glass::warp::copy<float>((uint32_t)n, x + w*n, y + w*n);
}
__global__ void k_scal_warp(int n, int W, float alpha, float* x) {
    int w = threadIdx.y;
    if (w >= W) return;
    glass::warp::scal<float>((uint32_t)n, alpha, x + w*n);
}

// ─── L2 gemv kernels (compile-time square N; one warp per problem) ───────────
// gemv: y = alpha*A@x (implicit beta=0). gemv_t: y = alpha*A.T@x. Square A (M=N=Nc).
#define DEFINE_GEMV_KERNEL(Nc)                                                            \
    __global__ void k_gemv_warp_##Nc(int W, float alpha, float* A, float* x, float* y) { \
        int w = threadIdx.y; if (w >= W) return;                                         \
        glass::warp::gemv<float, Nc, Nc>(alpha, A + w*Nc*Nc, x + w*Nc, y + w*Nc);        \
    }                                                                                     \
    __global__ void k_gemvt_warp_##Nc(int W, float alpha, float* A, float* x, float* y) {\
        int w = threadIdx.y; if (w >= W) return;                                         \
        glass::warp::gemv<float, Nc, Nc, true>(alpha, A + w*Nc*Nc, x + w*Nc, y + w*Nc);  \
    }

// ─── L3 gemm kernel (compile-time square N; one warp per problem) ────────────
// C = alpha*A@B (beta=0). Square A,B,C (M=N=K=Nc).
#define DEFINE_GEMM_KERNEL(Nc)                                                              \
    __global__ void k_gemm_warp_##Nc(int W, float alpha, float* A, float* B, float* C) {    \
        int w = threadIdx.y; if (w >= W) return;                                           \
        glass::warp::gemm<float, Nc, Nc, Nc>(alpha, A + w*Nc*Nc, B + w*Nc*Nc, 0.f, C + w*Nc*Nc); \
    }

// ─── L3 trsv (flagged) + posv kernels (compile-time N; __restrict__ params) ──
#define DEFINE_TRI_KERNEL(Nc)                                                              \
    template <bool LOWER, bool UNIT, bool TRANS>                                           \
    __global__ void k_trsv_warp_##Nc(int W, float* __restrict__ A, float* __restrict__ b){\
        int w = threadIdx.y; if (w >= W) return;                                          \
        glass::warp::trsv<float, Nc, LOWER, UNIT, TRANS>(A + w*Nc*Nc, b + w*Nc);          \
    }                                                                                       \
    __global__ void k_posv_warp_##Nc(int W, float* __restrict__ A, float* __restrict__ b){\
        int w = threadIdx.y; if (w >= W) return;                                          \
        glass::warp::posv<float, Nc>(A + w*Nc*Nc, b + w*Nc);                              \
    }

#define DEFINE_ALL(Nc) DEFINE_GEMV_KERNEL(Nc) DEFINE_GEMM_KERNEL(Nc) DEFINE_TRI_KERNEL(Nc)
DEFINE_ALL(5)
DEFINE_ALL(7)
DEFINE_ALL(16)
DEFINE_ALL(33)
DEFINE_ALL(40)
DEFINE_ALL(64)

// ─── dispatch helpers (runtime n -> compile-time kernel) ─────────────────────
static void launch_gemv(int n, int W, bool trans, float alpha, float* A, float* x, float* y) {
    dim3 blk(32, W);
    #define GEMV_CASE(Nc) case Nc: \
        if (trans) k_gemvt_warp_##Nc<<<1, blk>>>(W, alpha, A, x, y); \
        else       k_gemv_warp_##Nc<<<1, blk>>>(W, alpha, A, x, y);  break;
    switch (n) { GEMV_CASE(5) GEMV_CASE(7) GEMV_CASE(16) GEMV_CASE(33) GEMV_CASE(40) GEMV_CASE(64)
                 default: fprintf(stderr, "unsupported n=%d for gemv\n", n); exit(1); }
    #undef GEMV_CASE
}

static void launch_gemm(int n, int W, float alpha, float* A, float* B, float* C) {
    dim3 blk(32, W);
    #define GEMM_CASE(Nc) case Nc: k_gemm_warp_##Nc<<<1, blk>>>(W, alpha, A, B, C); break;
    switch (n) { GEMM_CASE(5) GEMM_CASE(7) GEMM_CASE(16) GEMM_CASE(33) GEMM_CASE(40) GEMM_CASE(64)
                 default: fprintf(stderr, "unsupported n=%d for gemm\n", n); exit(1); }
    #undef GEMM_CASE
}

// Per-N dispatch of the 8 {lower,unit,trans} bool combos. Generated by macro so
// the `k_trsv_warp_<Nc>` token-paste happens in macro context (templates can't
// paste). Each maps the runtime flag triple to a compile-time instantiation.
#define DEFINE_LAUNCH_TRSV_N(Nc)                                                       \
    static void launch_trsv_##Nc(int W, bool lower, bool unit, bool trans,             \
                                 float* A, float* b) {                                 \
        dim3 blk(32, W);                                                               \
        int key = (lower?4:0) | (unit?2:0) | (trans?1:0);                              \
        switch (key) {                                                                 \
            case 0: k_trsv_warp_##Nc<false,false,false><<<1,blk>>>(W,A,b); break;      \
            case 1: k_trsv_warp_##Nc<false,false,true ><<<1,blk>>>(W,A,b); break;      \
            case 2: k_trsv_warp_##Nc<false,true ,false><<<1,blk>>>(W,A,b); break;      \
            case 3: k_trsv_warp_##Nc<false,true ,true ><<<1,blk>>>(W,A,b); break;      \
            case 4: k_trsv_warp_##Nc<true ,false,false><<<1,blk>>>(W,A,b); break;      \
            case 5: k_trsv_warp_##Nc<true ,false,true ><<<1,blk>>>(W,A,b); break;      \
            case 6: k_trsv_warp_##Nc<true ,true ,false><<<1,blk>>>(W,A,b); break;      \
            case 7: k_trsv_warp_##Nc<true ,true ,true ><<<1,blk>>>(W,A,b); break;      \
        }                                                                             \
    }
DEFINE_LAUNCH_TRSV_N(5)
DEFINE_LAUNCH_TRSV_N(7)
DEFINE_LAUNCH_TRSV_N(16)
DEFINE_LAUNCH_TRSV_N(33)
DEFINE_LAUNCH_TRSV_N(40)
DEFINE_LAUNCH_TRSV_N(64)

static void launch_trsv(int n, int W, bool lower, bool unit, bool trans, float* A, float* b) {
    switch (n) {
        case 5:  launch_trsv_5 (W,lower,unit,trans,A,b); break;
        case 7:  launch_trsv_7 (W,lower,unit,trans,A,b); break;
        case 16: launch_trsv_16(W,lower,unit,trans,A,b); break;
        case 33: launch_trsv_33(W,lower,unit,trans,A,b); break;
        case 40: launch_trsv_40(W,lower,unit,trans,A,b); break;
        case 64: launch_trsv_64(W,lower,unit,trans,A,b); break;
        default: fprintf(stderr, "unsupported n=%d for trsv\n", n); exit(1);
    }
}
static void launch_posv(int n, int W, float* A, float* b) {
    dim3 blk(32, W);
    switch (n) {
        case 5:  k_posv_warp_5 <<<1,blk>>>(W,A,b); break;
        case 7:  k_posv_warp_7 <<<1,blk>>>(W,A,b); break;
        case 16: k_posv_warp_16<<<1,blk>>>(W,A,b); break;
        case 33: k_posv_warp_33<<<1,blk>>>(W,A,b); break;
        case 40: k_posv_warp_40<<<1,blk>>>(W,A,b); break;
        case 64: k_posv_warp_64<<<1,blk>>>(W,A,b); break;
        default: fprintf(stderr, "unsupported n=%d for posv\n", n); exit(1);
    }
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <op> <n> <WARPS> [flags] <files...>\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];
    int n = atoi(argv[2]);
    int W = atoi(argv[3]);

    if (strcmp(op, "dot") == 0) {
        float* x = read_device_vec(argv[4], n*W);
        float* y = read_device_vec(argv[5], n*W);
        float* out = alloc_device_vec(W);
        k_dot_warp<<<1, dim3(32, W)>>>(n, W, x, y, out);
        cudaDeviceSynchronize();
        print_device_vec(out, W);

    } else if (strcmp(op, "axpy") == 0) {
        float alpha = atof(argv[4]);
        float* x = read_device_vec(argv[5], n*W);
        float* y = read_device_vec(argv[6], n*W);
        k_axpy_warp<<<1, dim3(32, W)>>>(n, W, alpha, x, y);
        cudaDeviceSynchronize();
        print_device_vec(y, n*W);

    } else if (strcmp(op, "copy") == 0) {
        float* x = read_device_vec(argv[4], n*W);
        float* y = alloc_device_vec(n*W);
        k_copy_warp<<<1, dim3(32, W)>>>(n, W, x, y);
        cudaDeviceSynchronize();
        print_device_vec(y, n*W);

    } else if (strcmp(op, "scal") == 0) {
        float alpha = atof(argv[4]);
        float* x = read_device_vec(argv[5], n*W);
        k_scal_warp<<<1, dim3(32, W)>>>(n, W, alpha, x);
        cudaDeviceSynchronize();
        print_device_vec(x, n*W);

    } else if (strcmp(op, "gemv") == 0 || strcmp(op, "gemv_t") == 0) {
        bool trans = (strcmp(op, "gemv_t") == 0);
        float alpha = atof(argv[4]);
        float* A = read_device_vec(argv[5], n*n*W);
        float* x = read_device_vec(argv[6], n*W);
        float* y = alloc_device_vec(n*W);
        launch_gemv(n, W, trans, alpha, A, x, y);
        cudaDeviceSynchronize();
        print_device_vec(y, n*W);

    } else if (strcmp(op, "gemm") == 0) {
        float alpha = atof(argv[4]);
        float* A = read_device_vec(argv[5], n*n*W);
        float* B = read_device_vec(argv[6], n*n*W);
        float* C = alloc_device_vec(n*n*W);
        launch_gemm(n, W, alpha, A, B, C);
        cudaDeviceSynchronize();
        print_device_vec(C, n*n*W);

    } else if (strcmp(op, "trsv") == 0) {
        bool lower = atoi(argv[4]) != 0;
        bool unit  = atoi(argv[5]) != 0;
        bool trans = atoi(argv[6]) != 0;
        float* A = read_device_vec(argv[7], n*n*W);
        float* b = read_device_vec(argv[8], n*W);
        launch_trsv(n, W, lower, unit, trans, A, b);
        cudaDeviceSynchronize();
        print_device_vec(b, n*W);

    } else if (strcmp(op, "posv") == 0) {
        float* A = read_device_vec(argv[4], n*n*W);
        float* b = read_device_vec(argv[5], n*W);
        launch_posv(n, W, A, b);
        cudaDeviceSynchronize();
        print_device_vec(b, n*W);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }
    return 0;
}
