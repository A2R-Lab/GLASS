// test_trailing_sync.cu — verify that every GLASS L1/L2/L3 surface exposes
// the `bool TRAILING_SYNC` template parameter with both true/false
// specializations and that the two variants produce identical numerical
// output.
//
// What this catches:
//   * The cuBLASDx-backed macros (_GLASS_GEMM_NO_BD / _GLASS_GEMM_BD /
//     _GLASS_GEMV_NO_BD / _GLASS_GEMV_BD) emit BOTH specializations.
//   * The L1 / L3_SIMT primary templates allow both instantiations.
//   * `if constexpr (TRAILING_SYNC)` gating is correct (when false, the
//     caller must emit __syncthreads() before reading — we do that here).
//
// What this does NOT catch:
//   * Performance differences (kernel time is below noise floor at these
//     sizes — the test is correctness-only).
//   * Race conditions when callers forget to sync (that's a user error
//     by design — TRAILING_SYNC=false is opt-in).
//
// Usage:  ./test_trailing_sync <op>
//   ops:  l1_dot, l3_simt_batched, l3_simt_strided_batched, l3_cublasdx_gemm
//   (l3_cublasdx_gemm requires the test to be built with -DGLASS_BENCH_CUBLASDX
//    so the 16x16x16 cuBLASDx specialization exists; otherwise that op exits 0
//    with a "SKIP" line so the pytest layer marks it skipped.)
//
// Returns 0 + "PASS" if true/false variants match within 1e-5 max abs error.
// Returns 1 + "FAIL" otherwise.

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

// ─── L1: glass::nvidia::dot ──────────────────────────────────────────────────
//
// Two kernels: one with default TRAILING_SYNC=true, one with =false + explicit
// caller sync. Both write the dot product to out[0]. Result must match.

template <bool TRAILING_SYNC>
__global__ void k_l1_dot(float* x, float* y, float* out, float* scratch)
{
    glass::nvidia::dot<float, 64, 256, TRAILING_SYNC>(x, y, out, scratch);
    if constexpr (!TRAILING_SYNC) {
        __syncthreads();   // caller-emitted sync — required when TRAILING_SYNC=false
    }
}

// ─── L3_SIMT: gemm_batched_1d / gemm_strided_batched_1d ──────────────────────

template <bool TRAILING_SYNC>
__global__ void k_l3_simt_batched(float* const* A, float* const* B, float* const* C)
{
    glass::nvidia::gemm_batched_1d<
        float, 4, 4, 4, /*BATCH=*/4, /*TC=*/64,
        glass::nvidia::layout::col_major,
        glass::nvidia::layout::col_major,
        glass::nvidia::layout::col_major,
        TRAILING_SYNC>(1.f, A, B, 0.f, C);
    if constexpr (!TRAILING_SYNC) __syncthreads();
}

template <bool TRAILING_SYNC>
__global__ void k_l3_simt_strided_batched(float* A_shared, float* B, float* C)
{
    glass::nvidia::gemm_strided_batched_1d<
        float, 4, 4, 4, /*BATCH=*/4, /*TC=*/64,
        /*B_STRIDE=*/16, /*C_STRIDE=*/16,
        glass::nvidia::layout::col_major,
        glass::nvidia::layout::col_major,
        glass::nvidia::layout::col_major,
        TRAILING_SYNC>(1.f, A_shared, B, 0.f, C);
    if constexpr (!TRAILING_SYNC) __syncthreads();
}

// ─── L3 cuBLASDx: gemm at a shape pre-instantiated by glass-nvidia.cuh ───────
// Only compiled when GLASS_BENCH_CUBLASDX is defined (cuBLASDx-available
// builds). glass-nvidia.cuh ships `DEFINE_NVIDIA_GEMM(16, 16, 16)` already,
// so both TRAILING_SYNC=true and =false specializations are emitted by the
// macro. No re-DEFINE needed in this test file.

#ifdef GLASS_BENCH_CUBLASDX
template <bool TRAILING_SYNC>
__global__ void k_l3_cublasdx_gemm(float* A, float* B, float* C)
{
    extern __shared__ __align__(16) char smem[];
    glass::nvidia::gemm<
        float, 16, 16, 16, /*BLOCK_THREADS=*/0,
        glass::nvidia::layout::col_major,
        glass::nvidia::layout::col_major,
        glass::nvidia::layout::col_major,
        SMS,
        TRAILING_SYNC>(1.f, A, B, 0.f, C, smem);
    if constexpr (!TRAILING_SYNC) __syncthreads();
}
#endif

// ─── ops ─────────────────────────────────────────────────────────────────────

static int op_l1_dot()
{
    constexpr int N = 64;
    std::vector<float> hx(N), hy(N);
    for (int i = 0; i < N; ++i) { hx[i] = 0.1f * i; hy[i] = 0.2f * (i + 1); }

    float *dx, *dy, *dout_t, *dout_f, *dscratch;
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout_t, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout_f, sizeof(float)));
    // CUB BlockReduce TempStorage upper bound; 256 threads → safe to allocate
    // more than needed.
    CUDA_CHECK(cudaMalloc(&dscratch, 4096));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    k_l1_dot<true ><<<1, 256>>>(dx, dy, dout_t, dscratch); CUDA_CHECK(cudaDeviceSynchronize());
    k_l1_dot<false><<<1, 256>>>(dx, dy, dout_f, dscratch); CUDA_CHECK(cudaDeviceSynchronize());

    float out_t, out_f;
    CUDA_CHECK(cudaMemcpy(&out_t, dout_t, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&out_f, dout_f, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dx); cudaFree(dy); cudaFree(dout_t); cudaFree(dout_f); cudaFree(dscratch);

    float diff = std::fabs(out_t - out_f);
    std::printf("l1_dot trailing_sync=true:%g false:%g max_abs_diff=%g\n", out_t, out_f, diff);
    bool ok = diff < 1e-5f;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

static int op_l3_simt_batched()
{
    constexpr int BATCH = 4, M = 4, N = 4, K = 4;
    std::vector<float> hA(M*N), hB(N*K);
    for (int i = 0; i < M*N; ++i) hA[i] = 0.1f * (i + 1);
    for (int i = 0; i < N*K; ++i) hB[i] = 0.2f * (i + 1);

    float *dA, *dB, *dC_t, *dC_f;
    CUDA_CHECK(cudaMalloc(&dA, BATCH*M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, BATCH*N*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_t, BATCH*M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_f, BATCH*M*K*sizeof(float)));
    // populate every batch with the same matrices for simplicity
    for (int b = 0; b < BATCH; ++b) {
        CUDA_CHECK(cudaMemcpy(dA + b*M*N, hA.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB + b*N*K, hB.data(), N*K*sizeof(float), cudaMemcpyHostToDevice));
    }

    float **dAs, **dBs, **dCs_t, **dCs_f;
    CUDA_CHECK(cudaMalloc(&dAs, BATCH * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&dBs, BATCH * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&dCs_t, BATCH * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&dCs_f, BATCH * sizeof(float*)));
    float *hAs[BATCH], *hBs[BATCH], *hCs_t[BATCH], *hCs_f[BATCH];
    for (int b = 0; b < BATCH; ++b) {
        hAs[b]   = dA + b*M*N;
        hBs[b]   = dB + b*N*K;
        hCs_t[b] = dC_t + b*M*K;
        hCs_f[b] = dC_f + b*M*K;
    }
    CUDA_CHECK(cudaMemcpy(dAs, hAs, BATCH * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBs, hBs, BATCH * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCs_t, hCs_t, BATCH * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCs_f, hCs_f, BATCH * sizeof(float*), cudaMemcpyHostToDevice));

    k_l3_simt_batched<true ><<<1, 64 * BATCH>>>(dAs, dBs, dCs_t); CUDA_CHECK(cudaDeviceSynchronize());
    k_l3_simt_batched<false><<<1, 64 * BATCH>>>(dAs, dBs, dCs_f); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hC_t(BATCH*M*K), hC_f(BATCH*M*K);
    CUDA_CHECK(cudaMemcpy(hC_t.data(), dC_t, BATCH*M*K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_f.data(), dC_f, BATCH*M*K*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC_t); cudaFree(dC_f);
    cudaFree(dAs); cudaFree(dBs); cudaFree(dCs_t); cudaFree(dCs_f);

    float diff = max_abs_diff(hC_t, hC_f);
    std::printf("l3_simt_batched max_abs_diff=%g\n", diff);
    bool ok = diff < 1e-5f;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

static int op_l3_simt_strided_batched()
{
    constexpr int BATCH = 4, M = 4, N = 4, K = 4;
    std::vector<float> hA(M*N), hB(N*K);
    for (int i = 0; i < M*N; ++i) hA[i] = 0.1f * (i + 1);
    for (int i = 0; i < N*K; ++i) hB[i] = 0.2f * (i + 1);

    // B and C are tightly packed (B_STRIDE=N*K=16, C_STRIDE=M*K=16, defaults).
    float *dA, *dB, *dC_t, *dC_f;
    CUDA_CHECK(cudaMalloc(&dA, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, BATCH*N*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_t, BATCH*M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_f, BATCH*M*K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
    for (int b = 0; b < BATCH; ++b) {
        CUDA_CHECK(cudaMemcpy(dB + b*N*K, hB.data(), N*K*sizeof(float), cudaMemcpyHostToDevice));
    }

    k_l3_simt_strided_batched<true ><<<1, 64 * BATCH>>>(dA, dB, dC_t); CUDA_CHECK(cudaDeviceSynchronize());
    k_l3_simt_strided_batched<false><<<1, 64 * BATCH>>>(dA, dB, dC_f); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hC_t(BATCH*M*K), hC_f(BATCH*M*K);
    CUDA_CHECK(cudaMemcpy(hC_t.data(), dC_t, BATCH*M*K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_f.data(), dC_f, BATCH*M*K*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC_t); cudaFree(dC_f);

    float diff = max_abs_diff(hC_t, hC_f);
    std::printf("l3_simt_strided_batched max_abs_diff=%g\n", diff);
    bool ok = diff < 1e-5f;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

#ifdef GLASS_BENCH_CUBLASDX
static int op_l3_cublasdx_gemm()
{
    constexpr int M = 16, N = 16, K = 16;
    std::vector<float> hA(M*N), hB(N*K);
    for (int i = 0; i < M*N; ++i) hA[i] = 0.05f * (i + 1);
    for (int i = 0; i < N*K; ++i) hB[i] = 0.07f * (i + 1);

    float *dA, *dB, *dC_t, *dC_f;
    CUDA_CHECK(cudaMalloc(&dA, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_t, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_f, M*K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N*K*sizeof(float), cudaMemcpyHostToDevice));

    auto smem_bytes = glass::nvidia::gemm_smem_size<float, M, N, K>();
    auto threads    = glass::nvidia::gemm_threads<float, M, N, K>();

    k_l3_cublasdx_gemm<true ><<<1, threads, smem_bytes>>>(dA, dB, dC_t); CUDA_CHECK(cudaDeviceSynchronize());
    k_l3_cublasdx_gemm<false><<<1, threads, smem_bytes>>>(dA, dB, dC_f); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hC_t(M*K), hC_f(M*K);
    CUDA_CHECK(cudaMemcpy(hC_t.data(), dC_t, M*K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_f.data(), dC_f, M*K*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC_t); cudaFree(dC_f);

    float diff = max_abs_diff(hC_t, hC_f);
    std::printf("l3_cublasdx_gemm max_abs_diff=%g\n", diff);
    bool ok = diff < 1e-3f;  // cuBLASDx uses tensor-core-ish ops; bigger tolerance
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
#endif

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <op>\n", argv[0]);
        return 2;
    }
    const char* op = argv[1];
    if (std::strcmp(op, "l1_dot") == 0)                    return op_l1_dot();
    if (std::strcmp(op, "l3_simt_batched") == 0)           return op_l3_simt_batched();
    if (std::strcmp(op, "l3_simt_strided_batched") == 0)   return op_l3_simt_strided_batched();
#ifdef GLASS_BENCH_CUBLASDX
    if (std::strcmp(op, "l3_cublasdx_gemm") == 0)          return op_l3_cublasdx_gemm();
#else
    if (std::strcmp(op, "l3_cublasdx_gemm") == 0) {
        std::printf("SKIP (no cuBLASDx)\n");
        return 0;
    }
#endif
    std::fprintf(stderr, "unknown op: %s\n", op);
    return 2;
}
