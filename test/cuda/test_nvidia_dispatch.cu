// test_nvidia_dispatch.cu — exercise the round-2 auto-dispatch.
//
// Companion to test_l3_nvidia.cu (which exercises the SIMT-only batched APIs).
// This file targets the round-2 additions:
//   * Gap A — glass::nvidia::gemv<>     auto-dispatches SIMT vs cuBLASDx
//   * Gap B — row_strided_gemv<>        auto-dispatches; uses stride directly on SIMT
//   * Gap C — row_strided_gemm<>        auto-dispatches; skips compact-pack on SIMT
//   * Gap D — gemm<T,...,col,row,col>   maps onto SIMT TRANSPOSE_B=true
//   * print_dispatch<>                  query helper from query_simt.cuh
//
// Usage:  ./test_nvidia_dispatch <op>
//   ops:  gemm_simt, gemm_cublas, gemm_transb, gemv_simt, strided_gemv,
//         strided_gemm, dispatch_q
//
// Returns 0 + "PASS" on stdout if the result matches the reference within
// 1e-4 max abs error; returns 1 + "FAIL" otherwise.

#include "../../glass-nvidia.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA: %s @ %d: %s\n", cudaGetErrorString(e), __LINE__, #x); \
    return 1; } } while (0)

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// ─── kernels ────────────────────────────────────────────────────────────────

__global__ void k_gemm_6x6x6(float* A, float* B, float* C) {
    // 6x6x6 has no DEFINE_NVIDIA_GEMM → SIMT route via primary template.
    glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, nullptr);
}

__global__ void k_gemm_16x16x16_dx(float* A, float* B, float* C) {
    // 16x16x16 has DEFINE_NVIDIA_GEMM(16,16,16) in glass-nvidia.cuh; cuBLASDx.
    extern __shared__ __align__(16) char smem[];
    glass::nvidia::gemm<float, 16, 16, 16>(1.f, A, B, 0.f, C, smem);
}

__global__ void k_gemm_16x16x16_simt(float* A, float* B, float* C) {
    // SIMT direct call for bit-parity reference.
    ::glass::gemm<float, 16, 16, 16>(1.f, A, B, 0.f, C);
}

__global__ void k_gemm_6x6x6_transb(float* A, float* B, float* C) {
    // Gap D: LB=row_major maps to TRANSPOSE_B=true in the SIMT branch.
    using L = glass::nvidia::layout;
    glass::nvidia::gemm<float, 6, 6, 6, 0, L::col_major, L::row_major, L::col_major>(
        1.f, A, B, 0.f, C, nullptr);
}

__global__ void k_gemm_6x6x6_transb_simt(float* A, float* B, float* C) {
    ::glass::gemm<float, 6, 6, 6, /*TRANSPOSE_B=*/true>(1.f, A, B, 0.f, C);
}

__global__ void k_gemv_5x5(float* A, float* x, float* y) {
    // 5x5 gemv has no pre-instantiated DEFINE (those are 4,6,8,12,14,24,64).
    // Heuristic max<32 → SIMT.
    glass::nvidia::gemv<float, 5, 5>(1.f, A, x, 0.f, y, nullptr);
}

__global__ void k_strided_gemv_5x5_rs8(float* A, float* x, float* y) {
    // Gap B: SIMT uses stride directly, no smem packing.
    glass::nvidia::row_strided_gemv<float, 5, 5, 8>(1.f, A, x, 0.f, y, nullptr);
}

__global__ void k_strided_gemm_6x6x6_rs8(float* A, float* B, float* C) {
    // Gap C: SIMT uses A_RS=8, B_RS=8 directly.
    glass::nvidia::row_strided_gemm<float, 6, 6, 6, 8, 8>(1.f, A, B, 0.f, C, nullptr);
}

// ─── ops ────────────────────────────────────────────────────────────────────

static int op_gemm_simt() {
    constexpr int M=6, N=6, K=6;
    std::vector<float> A(M*N), B(N*K), Cref(M*K, 0.f), Cdev(M*K);
    for (int i = 0; i < M*N; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < N*K; i++) B[i] = 0.02f * (i+1);
    for (int j = 0; j < K; j++) for (int i = 0; i < M; i++) {
        float r = 0; for (int p = 0; p < N; p++) r += A[i + p*M]*B[p + j*N];
        Cref[i + j*M] = r;
    }
    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA, M*N*4)); CUDA_CHECK(cudaMalloc(&dB, N*K*4));
    CUDA_CHECK(cudaMalloc(&dC, M*K*4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N*K*4, cudaMemcpyHostToDevice));
    k_gemm_6x6x6<<<1, 64>>>(dA, dB, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Cdev.data(), dC, M*K*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(Cref, Cdev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return err > 1e-4f;
}

static int op_gemm_cublas() {
    constexpr int M=16, N=16, K=16;
    std::vector<float> A(M*N), B(N*K), Cref(M*K), Cdev(M*K);
    for (int i = 0; i < M*N; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < N*K; i++) B[i] = 0.02f * (i+1);
    float *dA,*dB,*dC,*dCref;
    constexpr size_t smemsz = glass::nvidia::gemm_smem_size<float, 16, 16, 16>();
    constexpr uint32_t tc = glass::nvidia::gemm_threads<float, 16, 16, 16>();
    CUDA_CHECK(cudaMalloc(&dA, M*N*4)); CUDA_CHECK(cudaMalloc(&dB, N*K*4));
    CUDA_CHECK(cudaMalloc(&dC, M*K*4)); CUDA_CHECK(cudaMalloc(&dCref, M*K*4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N*K*4, cudaMemcpyHostToDevice));
    k_gemm_16x16x16_dx<<<1, tc, smemsz>>>(dA, dB, dC);
    k_gemm_16x16x16_simt<<<1, 64>>>(dA, dB, dCref);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Cdev.data(), dC, M*K*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Cref.data(), dCref, M*K*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(Cref, Cdev);
    std::printf("err=%.3e tc=%u smem=%zu %s\n",
                err, tc, smemsz, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCref);
    return err > 1e-4f;
}

static int op_gemm_transb() {
    constexpr int M=6, N=6, K=6;
    std::vector<float> A(M*N), B(N*N), Cref(M*K), Cdev(M*K);
    for (int i = 0; i < M*N; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < N*N; i++) B[i] = 0.02f * (i+1);
    float *dA,*dB,*dC,*dCref;
    CUDA_CHECK(cudaMalloc(&dA, M*N*4)); CUDA_CHECK(cudaMalloc(&dB, N*N*4));
    CUDA_CHECK(cudaMalloc(&dC, M*K*4)); CUDA_CHECK(cudaMalloc(&dCref, M*K*4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N*N*4, cudaMemcpyHostToDevice));
    k_gemm_6x6x6_transb<<<1, 64>>>(dA, dB, dC);
    k_gemm_6x6x6_transb_simt<<<1, 64>>>(dA, dB, dCref);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Cdev.data(), dC, M*K*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Cref.data(), dCref, M*K*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(Cref, Cdev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCref);
    return err > 1e-4f;
}

static int op_gemv_simt() {
    constexpr int M=5, N=5;
    std::vector<float> A(M*N), x(N), yref(M), ydev(M);
    for (int i = 0; i < M*N; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < N; i++) x[i] = 0.1f*(i+1);
    for (int i = 0; i < M; i++) {
        float r = 0; for (int j = 0; j < N; j++) r += A[i + j*M] * x[j];
        yref[i] = r;
    }
    float *dA,*dx,*dy;
    CUDA_CHECK(cudaMalloc(&dA, M*N*4)); CUDA_CHECK(cudaMalloc(&dx, N*4));
    CUDA_CHECK(cudaMalloc(&dy, M*4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    k_gemv_5x5<<<1, 64>>>(dA, dx, dy);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ydev.data(), dy, M*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(yref, ydev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    return err > 1e-4f;
}

static int op_strided_gemv() {
    constexpr int M=5, N=5, RS=8;
    std::vector<float> Abuf(RS*N, 0.f), x(N), yref(M, 0.f), ydev(M, 0.f);
    for (int j = 0; j < N; j++) for (int i = 0; i < M; i++)
        Abuf[i + j*RS] = 0.01f*(i+1) + 0.05f*(j+1);
    for (int i = 0; i < N; i++) x[i] = 0.1f*(i+1);
    for (int i = 0; i < M; i++) {
        float r = 0; for (int j = 0; j < N; j++) r += Abuf[i + j*RS] * x[j];
        yref[i] = r;
    }
    float *dA,*dx,*dy;
    CUDA_CHECK(cudaMalloc(&dA, RS*N*4)); CUDA_CHECK(cudaMalloc(&dx, N*4));
    CUDA_CHECK(cudaMalloc(&dy, M*4));
    CUDA_CHECK(cudaMemcpy(dA, Abuf.data(), RS*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    k_strided_gemv_5x5_rs8<<<1, 64>>>(dA, dx, dy);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ydev.data(), dy, M*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(yref, ydev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    return err > 1e-4f;
}

static int op_strided_gemm() {
    constexpr int M=6, N=6, K=6, A_RS=8, B_RS=8;
    std::vector<float> Abuf(A_RS*N, 0.f), Bbuf(B_RS*K, 0.f),
                       Cref(M*K, 0.f), Cdev(M*K, 0.f);
    for (int j = 0; j < N; j++) for (int i = 0; i < M; i++)
        Abuf[i + j*A_RS] = 0.01f*(i+1) + 0.05f*(j+1);
    for (int j = 0; j < K; j++) for (int i = 0; i < N; i++)
        Bbuf[i + j*B_RS] = 0.02f*(i+1) - 0.03f*(j+1);
    for (int j = 0; j < K; j++) for (int i = 0; i < M; i++) {
        float r = 0;
        for (int p = 0; p < N; p++) r += Abuf[i + p*A_RS] * Bbuf[p + j*B_RS];
        Cref[i + j*M] = r;
    }
    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA, A_RS*N*4)); CUDA_CHECK(cudaMalloc(&dB, B_RS*K*4));
    CUDA_CHECK(cudaMalloc(&dC, M*K*4));
    CUDA_CHECK(cudaMemcpy(dA, Abuf.data(), A_RS*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, Bbuf.data(), B_RS*K*4, cudaMemcpyHostToDevice));
    k_strided_gemm_6x6x6_rs8<<<1, 64>>>(dA, dB, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Cdev.data(), dC, M*K*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(Cref, Cdev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return err > 1e-4f;
}

static int op_dispatch_q() {
    // print_dispatch is host-callable per query_simt.cuh.
    glass::nvidia::print_dispatch<float, 6, 6, 6>();
    glass::nvidia::print_dispatch<float, 16, 16, 16>();
    glass::nvidia::print_dispatch<float, 32, 32, 32>();
    glass::nvidia::print_dispatch<float, 64, 64, 64>();
    std::printf("PASS\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <op>\n", argv[0]); return 2; }
    const char* op = argv[1];
    if (!std::strcmp(op, "gemm_simt"))    return op_gemm_simt();
    if (!std::strcmp(op, "gemm_cublas"))  return op_gemm_cublas();
    if (!std::strcmp(op, "gemm_transb"))  return op_gemm_transb();
    if (!std::strcmp(op, "gemv_simt"))    return op_gemv_simt();
    if (!std::strcmp(op, "strided_gemv")) return op_strided_gemv();
    if (!std::strcmp(op, "strided_gemm")) return op_strided_gemm();
    if (!std::strcmp(op, "dispatch_q"))   return op_dispatch_q();
    std::fprintf(stderr, "unknown op: %s\n", op);
    return 2;
}
