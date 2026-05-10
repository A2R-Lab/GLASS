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

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
