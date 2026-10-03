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
//   ops: dot axpy copy scal gemv gemv_t trsv posv potrs
//   flags (trsv): <lower> <unit> <trans>  (each 0/1)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass.cuh"

// bool → enum translators for the flag-templated test kernels
__host__ __device__ constexpr glass::block::FillMode FM(bool lower) { return lower ? glass::block::FillMode::Lower : glass::block::FillMode::Upper; }
__host__ __device__ constexpr glass::block::Diag     DG(bool unit)  { return unit ? glass::block::Diag::Unit : glass::block::Diag::NonUnit; }


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
    template <bool LOWER, bool UNIT, bool TRANSPOSE>                                           \
    __global__ void k_trsv_warp_##Nc(int W, float* __restrict__ A, float* __restrict__ b){\
        int w = threadIdx.y; if (w >= W) return;                                          \
        glass::warp::trsv<float, Nc, FM(LOWER), DG(UNIT), TRANSPOSE>(A + w*Nc*Nc, b + w*Nc);          \
    }                                                                                       \
    __global__ void k_posv_warp_##Nc(int W, float* __restrict__ A, float* __restrict__ b){\
        int w = threadIdx.y; if (w >= W) return;                                          \
        glass::warp::posv<float, Nc>(A + w*Nc*Nc, b + w*Nc);                              \
    }                                                                                     \
    __global__ void k_potrs_warp_##Nc(int W, float* __restrict__ L, float* __restrict__ b){\
        int w = threadIdx.y; if (w >= W) return;                                          \
        glass::warp::potrs<float, Nc>(L + w*Nc*Nc, b + w*Nc);                             \
    }

// ─── geometry: transform_points (one warp per problem; X float or double) ─────
// Each warp w owns n points (pts + w*3n), its own transform(s) and its own
// output slice. The reference kernels inline HJCD-IK's `warp_config_free`
// placement expression verbatim (the line GLASS replaces) so pytest can assert
// BIT-identity on device: `mism[w]` counts output words whose bits differ.
// Transforms arrive as float32 files; the double variants promote on device
// (exact), so the numpy oracle sees the same values as float64.
template <typename T>
__global__ void k_promote(int n, const float* src, T* dst) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = static_cast<T>(src[i]);
}
template <typename T>
__global__ void k_xpts_warp(int n, int W, const T* X, const float* pts, float* out, float* ref, int* mism) {
    int w = threadIdx.y; if (w >= W) return;
    const int lane = threadIdx.x & 31;
    const T* Xw = X + w*16;
    const float* p = pts + w*3*n;
    glass::warp::transform_points<T>(Xw, p, out + w*3*n, n);
    // reference: HJCD-IK csrc/kernel/hjcd_kernel.cu warp_config_free, per-lane strided
    for (int s = lane; s < n; s += 32) {
        const float ox = p[3*s], oy = p[3*s + 1], oz = p[3*s + 2];
        ref[w*3*n + 3*s]     = (float)(Xw[0] * ox + Xw[4] * oy + Xw[8]  * oz + Xw[12]);
        ref[w*3*n + 3*s + 1] = (float)(Xw[1] * ox + Xw[5] * oy + Xw[9]  * oz + Xw[13]);
        ref[w*3*n + 3*s + 2] = (float)(Xw[2] * ox + Xw[6] * oy + Xw[10] * oz + Xw[14]);
    }
    __syncwarp();
    int bad = 0;
    for (int i = lane; i < 3*n; i += 32)
        bad += (__float_as_int(out[w*3*n + i]) != __float_as_int(ref[w*3*n + i]));
    bad = glass::warp::reduce<int>(bad);
    if (lane == 0) mism[w] = bad;
}
// indexed form: NX transforms per warp (Xs + w*16*NX), idx[i] in [0, NX)
template <typename T>
__global__ void k_xpts_idx_warp(int n, int W, int NX, const T* Xs, const int* idx, const float* pts,
                                float* out, float* ref, int* mism) {
    int w = threadIdx.y; if (w >= W) return;
    const int lane = threadIdx.x & 31;
    const T* Xw = Xs + w*16*NX;
    const int* iw = idx + w*n;
    const float* p = pts + w*3*n;
    glass::warp::transform_points<T>(Xw, iw, p, out + w*3*n, n);
    for (int s = lane; s < n; s += 32) {
        const T* X = &Xw[16 * iw[s]];
        const float ox = p[3*s], oy = p[3*s + 1], oz = p[3*s + 2];
        ref[w*3*n + 3*s]     = (float)(X[0] * ox + X[4] * oy + X[8]  * oz + X[12]);
        ref[w*3*n + 3*s + 1] = (float)(X[1] * ox + X[5] * oy + X[9]  * oz + X[13]);
        ref[w*3*n + 3*s + 2] = (float)(X[2] * ox + X[6] * oy + X[10] * oz + X[14]);
    }
    __syncwarp();
    int bad = 0;
    for (int i = lane; i < 3*n; i += 32)
        bad += (__float_as_int(out[w*3*n + i]) != __float_as_int(ref[w*3*n + i]));
    bad = glass::warp::reduce<int>(bad);
    if (lane == 0) mism[w] = bad;
}

// ─── votes: warp::any / warp::all on diverged-then-reconverged lanes ─────────
// pred[w*32 + lane] in {0,1}. Odd lanes take a data-dependent detour (so the
// warp is genuinely diverged) before everyone reconverges and votes; the
// per-warp result is {any, all} as floats.
__global__ void k_vote_warp(int W, const float* pred, float* out) {
    int w = threadIdx.y; if (w >= W) return;
    const int lane = threadIdx.x & 31;
    bool p = pred[w*32 + lane] != 0.f;
    float junk = 0.f;
    if (lane & 1) { for (int k = 0; k < lane; ++k) junk += sinf((float)k) * (p ? 1.f : -1.f); }
    __syncwarp();
    const bool a = glass::warp::any(p);
    const bool l = glass::warp::all(p);
    if (lane == 0) { out[2*w] = a ? 1.f : 0.f; out[2*w + 1] = l ? 1.f : 0.f; }
    if (junk == 12345.f) out[2*w] = -1.f;   // keep the detour live
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
static void launch_potrs(int n, int W, float* L, float* b) {
    dim3 blk(32, W);
    switch (n) {
        case 5:  k_potrs_warp_5 <<<1,blk>>>(W,L,b); break;
        case 7:  k_potrs_warp_7 <<<1,blk>>>(W,L,b); break;
        case 16: k_potrs_warp_16<<<1,blk>>>(W,L,b); break;
        case 33: k_potrs_warp_33<<<1,blk>>>(W,L,b); break;
        case 40: k_potrs_warp_40<<<1,blk>>>(W,L,b); break;
        case 64: k_potrs_warp_64<<<1,blk>>>(W,L,b); break;
        default: fprintf(stderr, "unsupported n=%d for potrs\n", n); exit(1);
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

    } else if (strcmp(op, "potrs") == 0) {
        float* L = read_device_vec(argv[4], n*n*W);
        float* b = read_device_vec(argv[5], n*W);
        launch_potrs(n, W, L, b);
        cudaDeviceSynchronize();
        print_device_vec(b, n*W);

    } else if (strcmp(op, "xpts") == 0 || strcmp(op, "xpts64") == 0) {
        // files: X (16*W), pts (3n*W). Output: mismatch count per warp, then out.
        const bool f64 = (strcmp(op, "xpts64") == 0);
        float* X32 = read_device_vec(argv[4], 16*W);
        float* pts = read_device_vec(argv[5], 3*n*W);
        float* out = alloc_device_vec(3*n*W);
        float* ref = alloc_device_vec(3*n*W);
        int* mism; cudaMalloc(&mism, W*sizeof(int));
        if (f64) {
            double* X64; cudaMalloc(&X64, 16*W*sizeof(double));
            k_promote<double><<<1, 128>>>(16*W, X32, X64);
            k_xpts_warp<double><<<1, dim3(32, W)>>>(n, W, X64, pts, out, ref, mism);
        } else {
            k_xpts_warp<float><<<1, dim3(32, W)>>>(n, W, X32, pts, out, ref, mism);
        }
        cudaDeviceSynchronize();
        {
            int* h = (int*)malloc(W*sizeof(int)); cudaMemcpy(h, mism, W*sizeof(int), cudaMemcpyDeviceToHost);
            for (int w = 0; w < W; ++w) printf("%d%s", h[w], w + 1 < W ? " " : "\n");
            free(h);
        }
        print_device_vec(out, 3*n*W);

    } else if (strcmp(op, "xpts_idx") == 0 || strcmp(op, "xpts_idx64") == 0) {
        // argv[4] = NX transforms per warp; files: Xs (16*NX*W), idx (n*W, as floats), pts (3n*W)
        const bool f64 = (strcmp(op, "xpts_idx64") == 0);
        int NX = atoi(argv[4]);
        float* X32 = read_device_vec(argv[5], 16*NX*W);
        float* idxf = read_host_vec(argv[6], n*W);
        float* pts = read_device_vec(argv[7], 3*n*W);
        int* hidx = (int*)malloc(n*W*sizeof(int));
        for (int i = 0; i < n*W; ++i) hidx[i] = (int)idxf[i];
        int* idx; cudaMalloc(&idx, n*W*sizeof(int)); cudaMemcpy(idx, hidx, n*W*sizeof(int), cudaMemcpyHostToDevice);
        float* out = alloc_device_vec(3*n*W);
        float* ref = alloc_device_vec(3*n*W);
        int* mism; cudaMalloc(&mism, W*sizeof(int));
        if (f64) {
            double* X64; cudaMalloc(&X64, 16*NX*W*sizeof(double));
            k_promote<double><<<1, 128>>>(16*NX*W, X32, X64);
            k_xpts_idx_warp<double><<<1, dim3(32, W)>>>(n, W, NX, X64, idx, pts, out, ref, mism);
        } else {
            k_xpts_idx_warp<float><<<1, dim3(32, W)>>>(n, W, NX, X32, idx, pts, out, ref, mism);
        }
        cudaDeviceSynchronize();
        {
            int* h = (int*)malloc(W*sizeof(int)); cudaMemcpy(h, mism, W*sizeof(int), cudaMemcpyDeviceToHost);
            for (int w = 0; w < W; ++w) printf("%d%s", h[w], w + 1 < W ? " " : "\n");
            free(h);
        }
        print_device_vec(out, 3*n*W);

    } else if (strcmp(op, "vote") == 0) {
        // n unused; file: pred (32*W floats in {0,1}). Output: {any, all} per warp.
        float* pred = read_device_vec(argv[4], 32*W);
        float* out = alloc_device_vec(2*W);
        k_vote_warp<<<1, dim3(32, W)>>>(W, pred, out);
        cudaDeviceSynchronize();
        print_device_vec(out, 2*W);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }
    return 0;
}
