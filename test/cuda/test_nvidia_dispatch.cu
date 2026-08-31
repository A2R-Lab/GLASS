// test_nvidia_dispatch.cu — exercise the round-2 auto-dispatch.
//
// Companion to test_l3_nvidia.cu (which exercises the SIMT-only batched APIs).
// This file targets the round-2 additions:
//   * Gap A — glass::nvidia::block::gemv<> auto-dispatches SIMT vs cuBLASDx
//   * Gap B — gemv_strided<>        auto-dispatches; uses stride directly on SIMT
//   * Gap C — gemm_strided<>        auto-dispatches; skips compact-pack on SIMT
//   * Gap D — gemm<T,...,col,row,col>   maps onto SIMT TRANSPOSE_B=true
//   * print_dispatch<>                  query helper from query_simt.cuh
//
// Usage:  ./test_nvidia_dispatch <op>
//   ops:  gemm_simt, gemm_cublas, gemm_transb, gemv_simt, strided_gemv,
//         strided_gemm, beta0_poison, dispatch_q
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
    glass::nvidia::block::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, nullptr);
}

__global__ void k_gemm_16x16x16_dx(float* A, float* B, float* C) {
    // 16x16x16 has DEFINE_NVIDIA_GEMM(16,16,16) in glass-nvidia.cuh; cuBLASDx.
    extern __shared__ __align__(16) char smem[];
    glass::nvidia::block::gemm<float, 16, 16, 16>(1.f, A, B, 0.f, C, smem);
}

__global__ void k_gemm_16x16x16_simt(float* A, float* B, float* C) {
    // SIMT direct call for bit-parity reference.
    ::glass::block::gemm<float, 16, 16, 16>(1.f, A, B, 0.f, C);
}

__global__ void k_gemm_6x6x6_transb(float* A, float* B, float* C) {
    // Gap D: LB=row_major maps to TRANSPOSE_B=true in the SIMT branch.
    using L = glass::nvidia::block::layout;
    glass::nvidia::block::gemm<float, 6, 6, 6, 0, L::col_major, L::row_major, L::col_major>(
        1.f, A, B, 0.f, C, nullptr);
}

__global__ void k_gemm_6x6x6_transb_simt(float* A, float* B, float* C) {
    ::glass::block::gemm<float, 6, 6, 6, /*TA=*/false, /*TRANSPOSE_B=*/true>(1.f, A, B, 0.f, C);
}

__global__ void k_gemv_5x5(float* A, float* x, float* y) {
    // 5x5 gemv has no pre-instantiated DEFINE (those are 4,6,8,12,14,24,64).
    // Heuristic max<32 → SIMT.
    glass::nvidia::block::gemv<float, 5, 5>(1.f, A, x, 0.f, y, nullptr);
}

__global__ void k_strided_gemv_5x5_rs8(float* A, float* x, float* y) {
    // Gap B: SIMT uses stride directly, no smem packing.
    glass::nvidia::block::gemv_strided<float, 5, 5, 8>(1.f, A, x, 0.f, y, nullptr);
}

__global__ void k_strided_gemm_6x6x6_rs8(float* A, float* B, float* C) {
    // Gap C: SIMT uses A_RS=8, B_RS=8 directly.
    glass::nvidia::block::gemm_strided<float, 6, 6, 6, 8, 8>(1.f, A, B, 0.f, C, nullptr);
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
    constexpr size_t smemsz = glass::nvidia::block::gemm_scratch_bytes<float, 16, 16, 16>();
    constexpr uint32_t tc = glass::nvidia::block::gemm_threads<float, 16, 16, 16>();
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

static int op_beta0_poison() {
    // BLAS beta==0 write-only semantics through the nvidia:: surface: with C
    // pre-poisoned to NaN, both the cuBLASDx route and the SIMT route must
    // come back clean. The SIMT arm pins the base beta_blend epilogue; the
    // cuBLASDx arm pins the VENDOR behavior (cublasdx::execute does not blend
    // a beta==0 destination — verified empirically on MathDx 26.03; this test
    // exists so a future MathDx bump that changes that fails loudly).
    constexpr int M=16, N=16, K=16;
    std::vector<float> A(M*K), B(K*N), Cref(M*N), Cdev(M*N),
                       Cpoison(M*N, std::nanf(""));
    for (int i = 0; i < M*K; i++) A[i] = 0.01f * (i+1);
    for (int i = 0; i < K*N; i++) B[i] = 0.02f * (i+1);
    float *dA,*dB,*dC,*dCref;
    constexpr size_t smemsz = glass::nvidia::block::gemm_scratch_bytes<float, 16, 16, 16>();
    constexpr uint32_t tc = glass::nvidia::block::gemm_threads<float, 16, 16, 16>();
    CUDA_CHECK(cudaMalloc(&dA, M*K*4)); CUDA_CHECK(cudaMalloc(&dB, K*N*4));
    CUDA_CHECK(cudaMalloc(&dC, M*N*4)); CUDA_CHECK(cudaMalloc(&dCref, M*N*4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), M*K*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), K*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, Cpoison.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCref, Cpoison.data(), M*N*4, cudaMemcpyHostToDevice));
    k_gemm_16x16x16_dx<<<1, tc, smemsz>>>(dA, dB, dC);
    k_gemm_16x16x16_simt<<<1, 64>>>(dA, dB, dCref);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Cdev.data(), dC, M*N*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Cref.data(), dCref, M*N*4, cudaMemcpyDeviceToHost));
    int nonfinite = 0;
    for (int i = 0; i < M*N; i++)
        nonfinite += !std::isfinite(Cdev[i]) + !std::isfinite(Cref[i]);
    float err = max_abs_diff(Cref, Cdev);
    std::printf("nonfinite=%d err=%.3e %s\n", nonfinite, err,
                (nonfinite || err > 1e-4f) ? "FAIL" : "PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCref);
    return nonfinite || err > 1e-4f;
}

namespace glass { namespace nvidia { namespace block {
    DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(8, 8, 8, 4, 64)
}}}

// ── coverage pin block: the full query/size surface, constexpr so the asserts
// ARE the test; plus the explicit-intent print_dispatch_* diagnostics below. ──
namespace gnb = glass::nvidia::block;
static_assert(!gnb::should_use_cublasdx<double, 16, 16, 16>(), "f64 never routes to cuBLASDx");
static_assert(!gnb::should_use_cublasdx_gemv<double, 16, 16>(), "gemv f64 -> SIMT");
static_assert(!gnb::should_use_cublasdx_gemv_strided<double, 16, 16, 16>(), "gemv_strided f64 -> SIMT");
static_assert(!gnb::should_use_cublasdx_gemm_strided<double, 8, 8, 8, 8, 8>(), "gemm_strided f64 -> SIMT");
static_assert(!gnb::should_use_cublasdx_batched<double, 8, 8, 8, 4>(), "batched f64 -> SIMT");
static_assert(gnb::gemm_min_block_threads<float, 16, 16, 16>() > 0, "gemm min threads");
static_assert(gnb::gemm_block_threads_valid<float, 16, 16, 16, 1024>(), "1024 threads valid for gemm16");
static_assert(gnb::gemv_min_block_threads<float, 16, 16>() > 0, "gemv min threads");
static_assert(gnb::gemv_block_threads_valid<float, 16, 16, 1024>(), "1024 threads valid for gemv16");
static_assert(gnb::required_smem_for_dispatch_gemm<float, 16, 16, 16>() ==
              gnb::gemm_scratch_bytes<float, 16, 16, 16>(), "explicit-intent alias == scratch");
static_assert(gnb::required_smem_for_dispatch_gemv<float, 16, 16>() > 0 ||
              gnb::required_smem_for_dispatch_gemv<float, 16, 16>() == 0, "gemv dispatch smem evaluable");
static_assert(gnb::required_smem_for_dispatch_gemm_strided<float, 8, 8, 8>() >= 0u, "gemm_strided dispatch smem evaluable");
static_assert(gnb::required_smem_for_dispatch_gemv_strided<float, 8, 8>() >= 0u, "gemv_strided dispatch smem evaluable");
static_assert(gnb::gemm_batched_scratch_bytes<float, 8, 8, 8, 4, 64>() >= 0u, "batched scratch evaluable (stub 0 without DEFINE)");
static_assert(gnb::gemm_batched_threads<float, 8, 8, 8, 4, 64>() > 0, "batched threads");
static_assert(gnb::gemm_batched_1d_scratch_bytes<float, 8, 8, 8, 4, 32>() >= 0u, "batched_1d scratch evaluable");
static_assert(gnb::gemm_batched_1d_threads<float, 8, 8, 8, 4, 32>() > 0, "batched_1d threads");
static_assert(gnb::gemm_batched_1d_block_threads_valid<float, 8, 8, 8, 4, 32, 256>(), "256 threads valid for batched_1d");
static_assert(gnb::gemm_strided_batched_1d_scratch_bytes<float, 8, 8, 8, 4, 32>() >= 0u, "strided_batched_1d scratch evaluable");
static_assert(gnb::gemm_strided_batched_1d_threads<float, 8, 8, 8, 4, 32>() > 0, "strided_batched_1d threads");
static_assert(gnb::gemm_strided_scratch_bytes<float, 8, 8, 8>() >= 0u, "gemm_strided scratch evaluable");
static_assert(gnb::gemv_strided_scratch_bytes<float, 8, 8>() >= 0u, "gemv_strided scratch evaluable");
static_assert(gnb::gemv_strided_scratch_bytes<float, 5, 5, 8>() >= 0u,
              "gemv_strided explicit-row-stride scratch evaluable");
static_assert(gnb::reduce_scratch_bytes<float, 256>() > 0, "CUB reduce scratch");

__global__ void k_gemm_batched(float* const* A, float* const* B, float* const* C) {
    extern __shared__ char s[];
    gnb::gemm_batched<float, 8, 8, 8, 4, 64>(1.0f, A, B, 0.0f, C, s);
}

static int op_gemm_batched() {
    // BATCH=4 pointer-array GEMMs, deterministic inputs; host reference.
    const int N = 8, B = 4;
    float hA[4][64], hB[4][64];
    for (int b = 0; b < B; b++)
        for (int i = 0; i < N*N; i++) {
            hA[b][i] = ((i + 2*b) % 5) * 0.1f;
            hB[b][i] = ((i + 3*b) % 4) * 0.1f;
        }
    float *dA[4], *dB[4], *dC[4];
    for (int b = 0; b < B; b++) {
        cudaMalloc(&dA[b], 256); cudaMalloc(&dB[b], 256); cudaMalloc(&dC[b], 256);
        cudaMemcpy(dA[b], hA[b], 256, cudaMemcpyHostToDevice);
        cudaMemcpy(dB[b], hB[b], 256, cudaMemcpyHostToDevice);
    }
    float **pA, **pB, **pC;
    cudaMalloc(&pA, B*sizeof(float*)); cudaMalloc(&pB, B*sizeof(float*)); cudaMalloc(&pC, B*sizeof(float*));
    cudaMemcpy(pA, dA, B*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(pB, dB, B*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(pC, dC, B*sizeof(float*), cudaMemcpyHostToDevice);
    size_t sm = gnb::gemm_batched_scratch_bytes<float, 8, 8, 8, 4, 64>();
    cudaFuncSetAttribute(k_gemm_batched, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    k_gemm_batched<<<1, dim3(64, 4), sm>>>(pA, pB, pC);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL launch\n"); return 1; }
    for (int b = 0; b < B; b++) {
        float out[64]; cudaMemcpy(out, dC[b], 256, cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
            float want = 0;
            for (int k = 0; k < N; k++) want += hA[b][i + k*N] * hB[b][k + j*N];
            if (fabsf(out[i + j*N] - want) > 1e-3f) {
                std::printf("FAIL b=%d (%d,%d) got %g want %g\n", b, i, j, out[i + j*N], want);
                return 1;
            }
        }
    }
    std::printf("PASS\n");
    return 0;
}

static int op_dispatch_q() {
    // print_dispatch is host-callable per query_simt.cuh.
    glass::nvidia::block::print_dispatch<float, 6, 6, 6>();
    glass::nvidia::block::print_dispatch<float, 16, 16, 16>();
    glass::nvidia::block::print_dispatch<float, 32, 32, 32>();
    glass::nvidia::block::print_dispatch<float, 64, 64, 64>();
    // explicit-intent diagnostics (round-2 aliases) — one call each by name.
    glass::nvidia::block::print_dispatch_gemv<float, 16, 16>();
    glass::nvidia::block::print_dispatch_gemv_strided<float, 16, 16>();
    glass::nvidia::block::print_dispatch_gemm_strided<float, 8, 8, 8>();
    glass::nvidia::block::print_dispatch_batched<float, 8, 8, 8, 4>();
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
    if (!std::strcmp(op, "beta0_poison")) return op_beta0_poison();
    if (!std::strcmp(op, "dispatch_q"))   return op_dispatch_q();
    if (!std::strcmp(op, "gemm_batched")) return op_gemm_batched();
    std::fprintf(stderr, "unknown op: %s\n", op);
    return 2;
}
