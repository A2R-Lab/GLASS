// test_l2.cu — dispatch L2 GLASS operations and print float32 results to stdout
// Usage: ./test_l2 <op> <cg|simple> <m> <n> [args...] [files...]
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

// ─── gemv kernels ─────────────────────────────────────────────────────────────
__global__ void k_gemv_cg(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::cgrps::gemv(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_simple(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::gemv(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_t_cg(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::cgrps::gemv<float, true>(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_t_simple(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::gemv<float, true>(m, n, alpha, A, x, beta, y);
}

// ─── gemv row-major kernels ───────────────────────────────────────────────────
__global__ void k_gemv_rowmajor(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::gemv<float, false, true>(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_ex_rowA(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::gemv_ex<float, false, true>(m, n, alpha, A, x, beta, y);
}

// ─── row_strided_gemv kernels (compile-time M, N, ROW_STRIDE) ────────────────
// A has M*ROW_STRIDE elements; A[i][j] = A[i + j*ROW_STRIDE] (col-major, LDA=ROW_STRIDE)
__global__ void k_gemv_strided_6x6_6(float alpha, float* A, float* x, float beta, float* y) {
    glass::row_strided_gemv<float, 6, 6, 6>(A, x, y, alpha, beta);
}
__global__ void k_gemv_strided_6x6_8(float alpha, float* A, float* x, float beta, float* y) {
    glass::row_strided_gemv<float, 6, 6, 8>(A, x, y, alpha, beta);
}
__global__ void k_gemv_strided_4x4_4(float alpha, float* A, float* x, float beta, float* y) {
    glass::row_strided_gemv<float, 4, 4, 4>(A, x, y, alpha, beta);
}
__global__ void k_gemv_strided_4x4_6(float alpha, float* A, float* x, float beta, float* y) {
    glass::row_strided_gemv<float, 4, 4, 6>(A, x, y, alpha, beta);
}

// ─── segmented_row_strided_gemv kernels ──────────────────────────────────────
// Descriptor arrays arrive as float32 .bin; cast to int on host then upload.
__global__ void k_seg_gemv_6x6_nofuse(uint32_t segs, int* a_off, int* x_off, int* y_off,
                                      float* A, float* x, float* y, float alpha, float beta) {
    glass::segmented_row_strided_gemv<float, 6, 6, 6, false>(
        segs, a_off, x_off, y_off, A, x, y, alpha, beta);
}
__global__ void k_seg_gemv_6x6_fuse(uint32_t segs, int* a_off, int* x_off, int* y_off,
                                     float* A, float* x, float* y, float alpha, float beta,
                                     int* s_off, float* S, float* scalar) {
    glass::segmented_row_strided_gemv<float, 6, 6, 6, true>(
        segs, a_off, x_off, y_off, A, x, y, alpha, beta, s_off, S, scalar);
}

// ─── segmented_row_strided_gemv: TRANSPOSE / ATOMIC_Y variants ───────────────
// TRANSPOSE: per segment y_seg(N) = alpha * Aᵀ_seg(N×M) * x_seg(M) + beta*y_seg.
// A_seg is M×N col-major LDA=ROW_STRIDE; here M=6,N=4,ROW_STRIDE=6.
__global__ void k_seg_gemv_transpose(uint32_t segs, int* a_off, int* x_off, int* y_off,
                                     float* A, float* x, float* y, float alpha, float beta) {
    glass::segmented_row_strided_gemv<float, 6, 4, 6,
        /*FUSE*/false, /*TRANSPOSE*/true, /*ATOMIC_Y*/false>(
        segs, a_off, x_off, y_off, A, x, y, alpha, beta);
}
// ATOMIC_Y (no transpose): per segment y_seg(M) += alpha * A_seg(M×N) * x_seg(N).
// Overlapping y ranges are summed atomically; caller pre-zeros/pre-scales y.
__global__ void k_seg_gemv_atomic(uint32_t segs, int* a_off, int* x_off, int* y_off,
                                  float* A, float* x, float* y, float alpha) {
    glass::segmented_row_strided_gemv<float, 6, 6, 6,
        /*FUSE*/false, /*TRANSPOSE*/false, /*ATOMIC_Y*/true>(
        segs, a_off, x_off, y_off, A, x, y, alpha);
}
// TRANSPOSE + ATOMIC_Y (the backward-pass case): children compute Xᵀ·f and
// atomically accumulate into shared parent y ranges. M=6,N=6,ROW_STRIDE=6.
__global__ void k_seg_gemv_transpose_atomic(uint32_t segs, int* a_off, int* x_off, int* y_off,
                                            float* A, float* x, float* y, float alpha) {
    glass::segmented_row_strided_gemv<float, 6, 6, 6,
        /*FUSE*/false, /*TRANSPOSE*/true, /*ATOMIC_Y*/true>(
        segs, a_off, x_off, y_off, A, x, y, alpha);
}

// ─── ger kernels ──────────────────────────────────────────────────────────────
__global__ void k_ger_cg(int m, int n, float alpha, float* x, float* y, float* A) {
    glass::cgrps::ger(m, n, alpha, x, y, A);
}
__global__ void k_ger_simple(int m, int n, float alpha, float* x, float* y, float* A) {
    glass::ger(m, n, alpha, x, y, A);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <op> <cg|simple> <m> <n> [args...] [files...]\n", argv[0]);
        return 1;
    }
    const char* op  = argv[1];
    const char* ver = argv[2];
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);

    bool cg = (strcmp(ver, "cg") == 0);

    if (strcmp(op, "gemv") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], m * n);
        float* dx = read_device_vec(argv[8], n);
        float* dy = read_device_vec(argv[9], m);
        if (cg) k_gemv_cg<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        else    k_gemv_simple<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, m);

    } else if (strcmp(op, "gemv_t") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], m * n);
        float* dx = read_device_vec(argv[8], m);
        float* dy = read_device_vec(argv[9], n);
        if (cg) k_gemv_t_cg<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        else    k_gemv_t_simple<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, n);

    } else if (strcmp(op, "ger") == 0) {
        float alpha = atof(argv[5]);
        float* dx = read_device_vec(argv[6], m);
        float* dy = read_device_vec(argv[7], n);
        float* dA = read_device_vec(argv[8], m * n);
        if (cg) k_ger_cg<<<1, THREADS>>>(m, n, alpha, dx, dy, dA);
        else    k_ger_simple<<<1, THREADS>>>(m, n, alpha, dx, dy, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, m * n);

    } else if (strcmp(op, "gemv_rowmajor") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], m * n);
        float* dx = read_device_vec(argv[8], n);
        float* dy = read_device_vec(argv[9], m);
        k_gemv_rowmajor<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, m);

    } else if (strcmp(op, "gemv_ex") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], m * n);
        float* dx = read_device_vec(argv[8], n);
        float* dy = read_device_vec(argv[9], m);
        k_gemv_ex_rowA<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, m);

    } else if (strcmp(op, "gemv_strided_6x6_6") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], 6 * 6);
        float* dx = read_device_vec(argv[8], 6);
        float* dy = read_device_vec(argv[9], 6);
        k_gemv_strided_6x6_6<<<1, THREADS>>>(alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, 6);

    } else if (strcmp(op, "gemv_strided_6x6_8") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], 6 * 8);  // LDA=8
        float* dx = read_device_vec(argv[8], 6);
        float* dy = read_device_vec(argv[9], 6);
        k_gemv_strided_6x6_8<<<1, THREADS>>>(alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, 6);

    } else if (strcmp(op, "gemv_strided_4x4_4") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], 4 * 4);
        float* dx = read_device_vec(argv[8], 4);
        float* dy = read_device_vec(argv[9], 4);
        k_gemv_strided_4x4_4<<<1, THREADS>>>(alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, 4);

    } else if (strcmp(op, "gemv_strided_4x4_6") == 0) {
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], 4 * 6);  // LDA=6
        float* dx = read_device_vec(argv[8], 4);
        float* dy = read_device_vec(argv[9], 4);
        k_gemv_strided_4x4_6<<<1, THREADS>>>(alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, 4);

    } else if (strcmp(op, "seg_gemv_6x6_nofuse") == 0) {
        // argv: m n segments alpha beta A_size x_size y_size  a_off x_off y_off A x y
        uint32_t segs = (uint32_t)atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        int A_size = atoi(argv[8]);
        int x_size = atoi(argv[9]);
        int y_size = atoi(argv[10]);
        int* a_off = read_device_ivec(argv[11], segs);
        int* x_off = read_device_ivec(argv[12], segs);
        int* y_off = read_device_ivec(argv[13], segs);
        float* dA = read_device_vec(argv[14], A_size);
        float* dx = read_device_vec(argv[15], x_size);
        float* dy = read_device_vec(argv[16], y_size);
        k_seg_gemv_6x6_nofuse<<<1, THREADS>>>(segs, a_off, x_off, y_off, dA, dx, dy, alpha, beta);
        cudaDeviceSynchronize();
        print_device_vec(dy, y_size);

    } else if (strcmp(op, "seg_gemv_6x6_fuse") == 0) {
        // argv: m n segments alpha beta A_size x_size y_size S_size
        //       a_off x_off y_off A x y s_off S scalar
        uint32_t segs = (uint32_t)atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        int A_size = atoi(argv[8]);
        int x_size = atoi(argv[9]);
        int y_size = atoi(argv[10]);
        int S_size = atoi(argv[11]);
        int* a_off = read_device_ivec(argv[12], segs);
        int* x_off = read_device_ivec(argv[13], segs);
        int* y_off = read_device_ivec(argv[14], segs);
        float* dA = read_device_vec(argv[15], A_size);
        float* dx = read_device_vec(argv[16], x_size);
        float* dy = read_device_vec(argv[17], y_size);
        int* s_off = read_device_ivec(argv[18], segs);
        float* dS = read_device_vec(argv[19], S_size);
        float* dscalar = read_device_vec(argv[20], segs);
        k_seg_gemv_6x6_fuse<<<1, THREADS>>>(segs, a_off, x_off, y_off, dA, dx, dy,
                                            alpha, beta, s_off, dS, dscalar);
        cudaDeviceSynchronize();
        print_device_vec(dy, y_size);

    } else if (strcmp(op, "seg_gemv_transpose") == 0) {
        // M=6,N=4,RS=6. argv: m n segments alpha beta A_size x_size y_size
        //                     a_off x_off y_off A x y
        uint32_t segs = (uint32_t)atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        int A_size = atoi(argv[8]);
        int x_size = atoi(argv[9]);
        int y_size = atoi(argv[10]);
        int* a_off = read_device_ivec(argv[11], segs);
        int* x_off = read_device_ivec(argv[12], segs);
        int* y_off = read_device_ivec(argv[13], segs);
        float* dA = read_device_vec(argv[14], A_size);
        float* dx = read_device_vec(argv[15], x_size);
        float* dy = read_device_vec(argv[16], y_size);
        k_seg_gemv_transpose<<<1, THREADS>>>(segs, a_off, x_off, y_off, dA, dx, dy, alpha, beta);
        cudaDeviceSynchronize();
        print_device_vec(dy, y_size);

    } else if (strcmp(op, "seg_gemv_atomic") == 0) {
        // M=6,N=6,RS=6, no-beta (pure accumulate). argv: m n segments alpha
        //   A_size x_size y_size  a_off x_off y_off A x y
        uint32_t segs = (uint32_t)atoi(argv[5]);
        float alpha = atof(argv[6]);
        int A_size = atoi(argv[7]);
        int x_size = atoi(argv[8]);
        int y_size = atoi(argv[9]);
        int* a_off = read_device_ivec(argv[10], segs);
        int* x_off = read_device_ivec(argv[11], segs);
        int* y_off = read_device_ivec(argv[12], segs);
        float* dA = read_device_vec(argv[13], A_size);
        float* dx = read_device_vec(argv[14], x_size);
        float* dy = read_device_vec(argv[15], y_size);
        k_seg_gemv_atomic<<<1, THREADS>>>(segs, a_off, x_off, y_off, dA, dx, dy, alpha);
        cudaDeviceSynchronize();
        print_device_vec(dy, y_size);

    } else if (strcmp(op, "seg_gemv_transpose_atomic") == 0) {
        // M=6,N=6,RS=6, transpose + atomic accumulate (no beta).
        // argv: m n segments alpha A_size x_size y_size a_off x_off y_off A x y
        uint32_t segs = (uint32_t)atoi(argv[5]);
        float alpha = atof(argv[6]);
        int A_size = atoi(argv[7]);
        int x_size = atoi(argv[8]);
        int y_size = atoi(argv[9]);
        int* a_off = read_device_ivec(argv[10], segs);
        int* x_off = read_device_ivec(argv[11], segs);
        int* y_off = read_device_ivec(argv[12], segs);
        float* dA = read_device_vec(argv[13], A_size);
        float* dx = read_device_vec(argv[14], x_size);
        float* dy = read_device_vec(argv[15], y_size);
        k_seg_gemv_transpose_atomic<<<1, THREADS>>>(segs, a_off, x_off, y_off, dA, dx, dy, alpha);
        cudaDeviceSynchronize();
        print_device_vec(dy, y_size);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
