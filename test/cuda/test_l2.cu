// test_l2.cu — dispatch L2 GLASS operations and print float32 results to stdout
// Usage: ./test_l2 <op> <cg|simple> <m> <n> [args...] [files...]

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.cuh"
#include "../../glass.cuh"

static int THREADS = 256;

// ─── gemv kernels ─────────────────────────────────────────────────────────────
__global__ void k_gemv_cg(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::gemv(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_simple(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::simple::gemv(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_t_cg(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::gemv<float, true>(m, n, alpha, A, x, beta, y);
}
__global__ void k_gemv_t_simple(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::simple::gemv<float, true>(m, n, alpha, A, x, beta, y);
}

// ─── gemv row-major kernels ───────────────────────────────────────────────────
// ROW_MAJOR=true: A stored in C row-major order (numpy default)
__global__ void k_gemv_rowmajor(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::simple::gemv<float, false, true>(m, n, alpha, A, x, beta, y);
}
// gemv_ex: row-major A only (per-matrix explicit flag)
__global__ void k_gemv_ex_rowA(int m, int n, float alpha, float* A, float* x, float beta, float* y) {
    glass::simple::gemv_ex<float, false, true>(m, n, alpha, A, x, beta, y);
}

// ─── ger kernels ──────────────────────────────────────────────────────────────
__global__ void k_ger_cg(int m, int n, float alpha, float* x, float* y, float* A) {
    glass::ger(m, n, alpha, x, y, A);
}
__global__ void k_ger_simple(int m, int n, float alpha, float* x, float* y, float* A) {
    glass::simple::ger(m, n, alpha, x, y, A);
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
        // argv: op ver m n alpha beta A.bin x.bin y.bin
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
        // transposed: y = alpha * A^T * x + beta * y  (A is mxn, output is n)
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
        // argv: op ver m n alpha x.bin y.bin A.bin
        float alpha = atof(argv[5]);
        float* dx = read_device_vec(argv[6], m);
        float* dy = read_device_vec(argv[7], n);
        float* dA = read_device_vec(argv[8], m * n);
        if (cg) k_ger_cg<<<1, THREADS>>>(m, n, alpha, dx, dy, dA);
        else    k_ger_simple<<<1, THREADS>>>(m, n, alpha, dx, dy, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, m * n);

    } else if (strcmp(op, "gemv_rowmajor") == 0) {
        // Row-major A (C-order): y = alpha * A * x + beta * y
        // ver ignored (simple only — cg has same result)
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], m * n);
        float* dx = read_device_vec(argv[8], n);
        float* dy = read_device_vec(argv[9], m);
        k_gemv_rowmajor<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, m);

    } else if (strcmp(op, "gemv_ex") == 0) {
        // gemv_ex with ROW_MAJOR_A=true (per-matrix flag)
        float alpha = atof(argv[5]);
        float beta  = atof(argv[6]);
        float* dA = read_device_vec(argv[7], m * n);
        float* dx = read_device_vec(argv[8], n);
        float* dy = read_device_vec(argv[9], m);
        k_gemv_ex_rowA<<<1, THREADS>>>(m, n, alpha, dA, dx, beta, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, m);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
