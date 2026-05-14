// test_nvidia_dispatch.cu — exercise the glass::nvidia auto-dispatch.
//
// Usage:
//   ./test_nvidia_dispatch <op>
//
// Each <op> runs a self-contained correctness check that returns 0 on success,
// non-zero on failure. The Python harness greps for "PASS" / "FAIL" on stdout.
//
// Ops covered:
//   gemm_simt    — 6x6x6, no DEFINE, auto-routes to ::glass::gemm
//   gemm_cublas  — 8x8x8 with DEFINE_NVIDIA_GEMM(8,8,8); checks bit-parity
//                  against ::glass::gemm
//   gemm_transb  — 6x6x6 with LB=row_major; checks parity with
//                  ::glass::gemm<TRANSPOSE_B=true>
//   strided_gemv — row_strided_gemv<6,6,RS=8> SIMT path, vs CPU reference
//   batched_simt — gemm_batched_1d<6,6,6,BATCH=4> SIMT vs loop of single SIMT
//   batched_dx   — gemm_batched_1d<8,8,8,BATCH=16,TC=64> cuBLASDx vs SIMT
//   dispatch_q   — print which path each shape takes (no PASS/FAIL — info only)

#include "/home/plancher/Desktop/GRiD/GLASS/glass-nvidia.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM(8, 8, 8)
    DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM(8, 8, 8, /*BATCH=*/16, /*TC=*/64)
}}

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA: %s @ %d: %s\n", cudaGetErrorString(e), __LINE__, #x); \
    return 1; } } while (0)

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// ─── kernels (one per call site so smem sizes are constexpr-clean) ─────────

__global__ void k_gemm_6x6x6(float* A, float* B, float* C) {
    glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, nullptr);
}
__global__ void k_gemm_8x8x8_dx(float* A, float* B, float* C) {
    extern __shared__ __align__(16) char smem[];
    glass::nvidia::gemm<float, 8, 8, 8>(1.f, A, B, 0.f, C, smem);
}
__global__ void k_gemm_8x8x8_simt(float* A, float* B, float* C) {
    ::glass::gemm<float, 8, 8, 8>(1.f, A, B, 0.f, C);
}
__global__ void k_gemm_6x6x6_transb(float* A, float* B, float* C) {
    using L = glass::nvidia::layout;
    glass::nvidia::gemm<float, 6, 6, 6, 0, L::col_major, L::row_major, L::col_major>(
        1.f, A, B, 0.f, C, nullptr);
}
__global__ void k_gemm_6x6x6_transb_simt(float* A, float* B, float* C) {
    ::glass::gemm<float, 6, 6, 6, /*TRANSPOSE_B=*/true>(1.f, A, B, 0.f, C);
}
__global__ void k_strided_gemv_6x6_rs8(float* A, float* x, float* y) {
    glass::nvidia::row_strided_gemv<float, 6, 6, 8>(1.f, A, x, 0.f, y, nullptr);
}
__global__ void k_batched_simt_6x6x6(float* const* A, float* const* B, float* const* C) {
    glass::nvidia::gemm_batched_1d<float, 6, 6, 6, 4>(1.f, A, B, 0.f, C, nullptr);
}
__global__ void k_batched_loop_6x6x6(float* const* A, float* const* B, float* const* C) {
    for (int b = 0; b < 4; b++)
        ::glass::gemm<float, 6, 6, 6>(1.f, A[b], B[b], 0.f, C[b]);
}
__global__ void k_batched_dx_8x8x8(float* const* A, float* const* B, float* const* C) {
    extern __shared__ __align__(16) char smem[];
    glass::nvidia::gemm_batched_1d<float, 8, 8, 8, 16, 64>(1.f, A, B, 0.f, C, smem);
}
__global__ void k_batched_simt_8x8x8(float* const* A, float* const* B, float* const* C) {
    ::glass::gemm_batched_1d<float, 8, 8, 8, 16>(1.f, A, B, 0.f, C);
}

// ─── op handlers ───────────────────────────────────────────────────────────

static int op_gemm_simt() {
    constexpr int M=6, N=6, K=6;
    std::vector<float> A(M*N), B(N*K), Cref(M*K, 0.f), Cdev(M*K);
    for (int i = 0; i < M*N; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < N*K; i++) B[i] = 0.02f * (i+1);
    // CPU reference, col-major
    for (int j = 0; j < K; j++) for (int i = 0; i < M; i++) {
        float r = 0; for (int p = 0; p < N; p++) r += A[i+p*M]*B[p+j*N];
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
    constexpr int M=8, N=8, K=8;
    std::vector<float> A(M*N), B(N*K), Cref(M*K), Cdev(M*K);
    for (int i = 0; i < M*N; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < N*K; i++) B[i] = 0.02f * (i+1);
    float *dA,*dB,*dC,*dCref;
    constexpr size_t smemsz = glass::nvidia::gemm_smem_size<float, 8, 8, 8>();
    constexpr uint32_t tc = glass::nvidia::gemm_threads<float, 8, 8, 8>();
    CUDA_CHECK(cudaMalloc(&dA, M*N*4)); CUDA_CHECK(cudaMalloc(&dB, N*K*4));
    CUDA_CHECK(cudaMalloc(&dC, M*K*4)); CUDA_CHECK(cudaMalloc(&dCref, M*K*4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N*K*4, cudaMemcpyHostToDevice));
    k_gemm_8x8x8_dx<<<1, tc, smemsz>>>(dA, dB, dC);
    k_gemm_8x8x8_simt<<<1, 64>>>(dA, dB, dCref);
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

static int op_strided_gemv() {
    constexpr int M=6, N=6, RS=8;
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
    k_strided_gemv_6x6_rs8<<<1, 64>>>(dA, dx, dy);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ydev.data(), dy, M*4, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(yref, ydev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    return err > 1e-4f;
}

static int run_batched(int M, int N, int K, int BATCH, bool cublas) {
    std::vector<std::vector<float>> Av(BATCH, std::vector<float>(M*N)),
                                    Bv(BATCH, std::vector<float>(N*K));
    std::vector<float> Cref(BATCH*M*K, 0.f), Cdev(BATCH*M*K, 0.f);
    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < M*N; i++) Av[b][i] = 0.01f * (i+1) + 0.1f*b;
        for (int i = 0; i < N*K; i++) Bv[b][i] = 0.02f * (i+1) - 0.05f*b;
    }
    std::vector<float*> dA(BATCH), dB(BATCH), dCref(BATCH), dCdev(BATCH);
    for (int b = 0; b < BATCH; b++) {
        CUDA_CHECK(cudaMalloc(&dA[b], M*N*4)); CUDA_CHECK(cudaMalloc(&dB[b], N*K*4));
        CUDA_CHECK(cudaMalloc(&dCref[b], M*K*4)); CUDA_CHECK(cudaMalloc(&dCdev[b], M*K*4));
        CUDA_CHECK(cudaMemcpy(dA[b], Av[b].data(), M*N*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB[b], Bv[b].data(), N*K*4, cudaMemcpyHostToDevice));
    }
    float **dAp,**dBp,**dCrp,**dCdp;
    CUDA_CHECK(cudaMalloc(&dAp,  BATCH*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&dBp,  BATCH*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&dCrp, BATCH*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&dCdp, BATCH*sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(dAp,  dA.data(),    BATCH*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBp,  dB.data(),    BATCH*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCrp, dCref.data(), BATCH*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCdp, dCdev.data(), BATCH*sizeof(float*), cudaMemcpyHostToDevice));

    if (cublas) {
        constexpr size_t smemsz = glass::nvidia::gemm_batched_1d_smem_size<float, 8, 8, 8, 16, 64>();
        k_batched_simt_8x8x8<<<1, 64>>>(dAp, dBp, dCrp);
        k_batched_dx_8x8x8<<<1, 64, smemsz>>>(dAp, dBp, dCdp);
    } else {
        k_batched_simt_6x6x6<<<1, 64>>>(dAp, dBp, dCdp);
        k_batched_loop_6x6x6<<<1, 64>>>(dAp, dBp, dCrp);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int b = 0; b < BATCH; b++) {
        CUDA_CHECK(cudaMemcpy(&Cdev[b*M*K], dCdev[b], M*K*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&Cref[b*M*K], dCref[b], M*K*4, cudaMemcpyDeviceToHost));
    }
    float err = max_abs_diff(Cref, Cdev);
    std::printf("err=%.3e %s\n", err, err > 1e-4f ? "FAIL" : "PASS");
    for (int b = 0; b < BATCH; b++) {
        cudaFree(dA[b]); cudaFree(dB[b]); cudaFree(dCref[b]); cudaFree(dCdev[b]);
    }
    cudaFree(dAp); cudaFree(dBp); cudaFree(dCrp); cudaFree(dCdp);
    return err > 1e-4f;
}

static int op_batched_simt()   { return run_batched(6, 6, 6, 4,  /*cublas=*/false); }
static int op_batched_dx()     { return run_batched(8, 8, 8, 16, /*cublas=*/true ); }

static int op_dispatch_q() {
    glass::nvidia::print_dispatch_full_gemm<float, 6, 6, 6>();
    glass::nvidia::print_dispatch_full_gemm<float, 24, 24, 24>();
    glass::nvidia::print_dispatch_full_gemv<float, 6, 6>();
    glass::nvidia::print_dispatch_full_gemv<float, 64, 64>();
    glass::nvidia::print_dispatch_full_batched<float, 6, 6, 6, 4>();
    glass::nvidia::print_dispatch_full_batched<float, 8, 8, 8, 16>();
    std::printf("PASS\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <op>\n", argv[0]); return 2; }
    const char* op = argv[1];
    if (!std::strcmp(op, "gemm_simt"))    return op_gemm_simt();
    if (!std::strcmp(op, "gemm_cublas"))  return op_gemm_cublas();
    if (!std::strcmp(op, "gemm_transb"))  return op_gemm_transb();
    if (!std::strcmp(op, "strided_gemv")) return op_strided_gemv();
    if (!std::strcmp(op, "batched_simt")) return op_batched_simt();
    if (!std::strcmp(op, "batched_dx"))   return op_batched_dx();
    if (!std::strcmp(op, "dispatch_q"))   return op_dispatch_q();
    std::fprintf(stderr, "unknown op: %s\n", op);
    return 2;
}
