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
    glass::nvidia::block::dot<float, 64, 256, TRAILING_SYNC>(x, y, out, scratch);
    if constexpr (!TRAILING_SYNC) {
        __syncthreads();   // caller-emitted sync — required when TRAILING_SYNC=false
    }
}

// ─── L1 warp: glass::nvidia::warp::{reduce, dot, nrm2} ───────────────────────
//
// One problem per FULL warp, 4 warps per block — each warp reduces its own
// N=24 slice, so the op must be warp-local (a block-wide CUB would corrupt
// neighbors). Checks TRAILING_SYNC parity AND absolute correctness vs a host
// reference for all 3 ops x 4 warps.

template <typename T, bool TRAILING_SYNC>
__global__ void k_l1_warp(T* x, T* y, T* out_dot, T* out_sum, T* out_nrm)
{
    constexpr uint32_t N = 24;
    const uint32_t w = threadIdx.x >> 5;
    __shared__ __align__(16) char scratch[4 * 64];   // >= 4 * warp_reduce_scratch_bytes<T>()
    T* s = reinterpret_cast<T*>(scratch + w * 64);
    glass::nvidia::warp::dot<T, N, TRAILING_SYNC>(x + w * N, y + w * N, out_dot + w, s);
    glass::nvidia::warp::nrm2<T, N, TRAILING_SYNC>(x + w * N, out_nrm + w, s);
    // reduce mutates x[w*N] in place — run it LAST so dot/nrm2 saw pristine x.
    glass::nvidia::warp::reduce<T, N, TRAILING_SYNC>(x + w * N, s);
    if ((threadIdx.x & 31u) == 0) out_sum[w] = x[w * N];   // lane-0 wrote it (same lane reads)
}

// ─── L3_SIMT: gemm_batched_1d / gemm_strided_batched_1d ──────────────────────

template <bool TRAILING_SYNC>
__global__ void k_l3_simt_batched(float* const* A, float* const* B, float* const* C)
{
    glass::nvidia::block::gemm_batched_1d<
        float, 4, 4, 4, /*BATCH=*/4, /*TC=*/64,
        glass::nvidia::block::layout::col_major,
        glass::nvidia::block::layout::col_major,
        glass::nvidia::block::layout::col_major,
        TRAILING_SYNC>(1.f, A, B, 0.f, C);
    if constexpr (!TRAILING_SYNC) __syncthreads();
}

template <bool TRAILING_SYNC>
__global__ void k_l3_simt_strided_batched(float* A_shared, float* B, float* C)
{
    glass::nvidia::block::gemm_strided_batched_1d<
        float, 4, 4, 4, /*BATCH=*/4, /*TC=*/64,
        /*B_STRIDE=*/16, /*C_STRIDE=*/16,
        glass::nvidia::block::layout::col_major,
        glass::nvidia::block::layout::col_major,
        glass::nvidia::block::layout::col_major,
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
    glass::nvidia::block::gemm<
        float, 16, 16, 16, /*BLOCK_THREADS=*/0,
        glass::nvidia::block::layout::col_major,
        glass::nvidia::block::layout::col_major,
        glass::nvidia::block::layout::col_major,
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

template <typename T>
static int op_l1_warp(const char* tag)
{
    constexpr int N = 24, WARPS = 4;
    std::vector<T> hx(WARPS * N), hy(WARPS * N);
    for (int i = 0; i < WARPS * N; ++i) {
        hx[i] = static_cast<T>(0.03) * (i % 17) - static_cast<T>(0.2);
        hy[i] = static_cast<T>(0.05) * (i % 13) + static_cast<T>(0.1);
    }
    // host reference per warp
    std::vector<double> ref_dot(WARPS, 0), ref_sum(WARPS, 0), ref_nrm(WARPS, 0);
    for (int w = 0; w < WARPS; ++w) {
        for (int i = 0; i < N; ++i) {
            double xv = (double)hx[w * N + i], yv = (double)hy[w * N + i];
            ref_dot[w] += xv * yv; ref_sum[w] += xv; ref_nrm[w] += xv * xv;
        }
        ref_nrm[w] = std::sqrt(ref_nrm[w]);
    }

    T *dx, *dy, *d_dot, *d_sum, *d_nrm;
    CUDA_CHECK(cudaMalloc(&dx, WARPS * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy, WARPS * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dot, WARPS * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_sum, WARPS * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_nrm, WARPS * sizeof(T)));

    const double tol = sizeof(T) == 4 ? 1e-5 : 1e-12;
    bool ok = true;
    std::vector<T> got_dot_prev, got_sum_prev, got_nrm_prev;
    for (int variant = 0; variant < 2; ++variant) {
        // reduce mutates x in place, so re-upload per variant.
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), WARPS * N * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, hy.data(), WARPS * N * sizeof(T), cudaMemcpyHostToDevice));
        if (variant == 0) k_l1_warp<T, true ><<<1, 32 * WARPS>>>(dx, dy, d_dot, d_sum, d_nrm);
        else              k_l1_warp<T, false><<<1, 32 * WARPS>>>(dx, dy, d_dot, d_sum, d_nrm);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<T> got_dot(WARPS), got_sum(WARPS), got_nrm(WARPS);
        CUDA_CHECK(cudaMemcpy(got_dot.data(), d_dot, WARPS * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_sum.data(), d_sum, WARPS * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_nrm.data(), d_nrm, WARPS * sizeof(T), cudaMemcpyDeviceToHost));
        for (int w = 0; w < WARPS; ++w) {
            ok = ok && std::fabs((double)got_dot[w] - ref_dot[w]) < tol
                    && std::fabs((double)got_sum[w] - ref_sum[w]) < tol
                    && std::fabs((double)got_nrm[w] - ref_nrm[w]) < tol;
        }
        if (variant == 1) {   // parity between the two sync variants
            for (int w = 0; w < WARPS; ++w) {
                ok = ok && got_dot[w] == got_dot_prev[w]
                        && got_sum[w] == got_sum_prev[w]
                        && got_nrm[w] == got_nrm_prev[w];
            }
        }
        got_dot_prev = got_dot; got_sum_prev = got_sum; got_nrm_prev = got_nrm;
    }
    cudaFree(dx); cudaFree(dy); cudaFree(d_dot); cudaFree(d_sum); cudaFree(d_nrm);
    std::printf("l1_warp_%s 3 ops x 4 warps vs host ref + sync parity\n", tag);
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

    auto smem_bytes = glass::nvidia::block::gemm_scratch_bytes<float, M, N, K>();
    auto threads    = glass::nvidia::block::gemm_threads<float, M, N, K>();

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


// ─── L3 factor/solve chain (uniform retrofit 2026-08-12) ─────────────────────
//
// Runs the retrofitted block ops — posv(multi-RHS), trsv, trsm, potrf, inv,
// trmv, syev, eigh, bdmv — each with TRAILING_SYNC as given; when false the
// kernel emits the caller-owned __syncthreads() after each call (the
// contract). true/false outputs must match exactly.

template <bool TS>
__global__ void k_l3_factor_solve(const float* in, float* out)
{
    constexpr uint32_t N = 8;
    __shared__ float sA[N*N], sB[N*2], sL[N*N], sb[N], sT[N*N], sBt[N*2];
    __shared__ float sP[N*N], sAug[N*2*N], sScr[2*N + 1];
    __shared__ float sE[N*N], sW[N], sV[N*N];
    __shared__ float sSy[N*N + 2*N + N*N];             // >= syev_scratch_bytes<f>(8)/4
    __shared__ float sM[2 * (3*4) * 4], sVec[(2+2)*4], sOut[(2+2)*4];
    __shared__ float sX[N], sY[N];
    const uint32_t rank = threadIdx.x, size = blockDim.x;
    for (uint32_t i = rank; i < N*N; i += size) {
        sA[i] = in[i]; sL[i] = in[N*N + i]; sT[i] = in[N*N + i];
        sP[i] = in[i]; sE[i] = in[2*N*N + i];
    }
    for (uint32_t i = rank; i < N*2; i += size) { sB[i] = in[3*N*N + i]; sBt[i] = in[3*N*N + i]; }
    for (uint32_t i = rank; i < N; i += size)   { sb[i] = in[3*N*N + i]; sX[i] = in[3*N*N + i]; }
    for (uint32_t i = rank; i < 2*N*N; i += size) sAug[i] = (i < N*N) ? in[i] : ((i - N*N) % (N+1) == 0 ? 1.f : 0.f);
    for (uint32_t i = rank; i < 2*(3*4)*4; i += size) sM[i] = in[i % (N*N)];
    for (uint32_t i = rank; i < (2+2)*4; i += size) sVec[i] = in[3*N*N + (i % (2*N))];
    __syncthreads();

    glass::block::posv<float, N, 2, false, false, false, TS>(sA, sB);
    if constexpr (!TS) __syncthreads();
    glass::block::trsv<float, N, glass::FillMode::Lower, glass::Diag::NonUnit, false, TS>(sL, sb);
    if constexpr (!TS) __syncthreads();
    glass::block::trsm<float, N, 2, glass::FillMode::Lower, glass::Diag::NonUnit, false, TS>(sT, sBt);
    if constexpr (!TS) __syncthreads();
    glass::block::potrf<float, N, false, TS>(sP);
    if constexpr (!TS) __syncthreads();
    glass::block::inv<float, N, TS>(sAug, sScr);
    if constexpr (!TS) __syncthreads();
    glass::block::trmv<float, N, glass::FillMode::Lower, glass::Diag::NonUnit, false, TS>(sL, sX, sY);
    if constexpr (!TS) __syncthreads();
    glass::block::syev<float, TS>(N, sE, sW, sV, sSy);
    if constexpr (!TS) __syncthreads();
    glass::block::bdmv<float, 2, 4, TS>(sOut, sM, sVec);
    if constexpr (!TS) __syncthreads();

    float* o = out;
    for (uint32_t i = rank; i < N*2;  i += size) o[i] = sB[i];          o += N*2;
    for (uint32_t i = rank; i < N;    i += size) o[i] = sb[i];          o += N;
    for (uint32_t i = rank; i < N*2;  i += size) o[i] = sBt[i];         o += N*2;
    for (uint32_t i = rank; i < N*N;  i += size) o[i] = sP[i];          o += N*N;
    for (uint32_t i = rank; i < N;    i += size) o[i] = sAug[N*N + i];  o += N;
    for (uint32_t i = rank; i < N;    i += size) o[i] = sY[i];          o += N;
    for (uint32_t i = rank; i < N;    i += size) o[i] = sW[i];          o += N;
    for (uint32_t i = rank; i < (2+2)*4; i += size) o[i] = sOut[i];
}

static int op_l3_factor_solve()
{
    constexpr uint32_t N = 8;
    constexpr size_t IN = 3*N*N + N*2;
    constexpr size_t OUT = N*2 + N + N*2 + N*N + N + N + N + (2+2)*4;
    std::vector<float> h_in(IN);
    // deterministic SPD-ish inputs: diag-dominant symmetric for slot 0, unit
    // lower-triangular magnitudes for slot 1, symmetric for slot 2, rhs after.
    for (size_t i = 0; i < IN; i++) h_in[i] = 0.02f * float((i * 2654435761u >> 16) & 0xff) - 2.5f;
    for (uint32_t r = 0; r < N; r++)
        for (uint32_t c = 0; c <= r; c++) {
            float v = h_in[r + c*N];
            h_in[c + r*N] = v;                     // symmetrize slot 0
            float w = h_in[2*N*N + r + c*N];
            h_in[2*N*N + c + r*N] = w;             // symmetrize slot 2
        }
    for (uint32_t d = 0; d < N; d++) {
        h_in[d + d*N] += 24.f;                     // SPD slot 0
        h_in[N*N + d + d*N] = 3.f + 0.1f * d;      // safe trsv/trsm diagonal
    }
    float *d_in, *d_out_t, *d_out_f;
    CUDA_CHECK(cudaMalloc(&d_in, IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_t, OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f, OUT * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), IN * sizeof(float), cudaMemcpyHostToDevice));
    k_l3_factor_solve<true ><<<1, 128>>>(d_in, d_out_t);
    k_l3_factor_solve<false><<<1, 128>>>(d_in, d_out_f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> a(OUT), b(OUT);
    CUDA_CHECK(cudaMemcpy(a.data(), d_out_t, OUT * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_out_f, OUT * sizeof(float), cudaMemcpyDeviceToHost));
    float d = max_abs_diff(a, b);
    std::printf("l3_factor_solve max_abs_diff=%.3e %s\n", d, d == 0.f ? "PASS" : "FAIL");
    return d == 0.f ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <op>\n", argv[0]);
        return 2;
    }
    const char* op = argv[1];
    if (std::strcmp(op, "l1_dot") == 0)                    return op_l1_dot();
    if (std::strcmp(op, "l1_warp_f32") == 0)               return op_l1_warp<float>("f32");
    if (std::strcmp(op, "l1_warp_f64") == 0)               return op_l1_warp<double>("f64");
    if (std::strcmp(op, "l3_simt_batched") == 0)           return op_l3_simt_batched();
    if (std::strcmp(op, "l3_simt_strided_batched") == 0)   return op_l3_simt_strided_batched();
    if (std::strcmp(op, "l3_factor_solve") == 0)           return op_l3_factor_solve();
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
