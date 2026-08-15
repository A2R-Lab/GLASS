// bench_nvwarp_l1.cu — warp-scope vendor A/B: glass::warp:: (SIMT shuffles) vs
// glass::nvidia::warp:: (cub::WarpReduce) on the three ops the vendor warp tier
// ships (reduce / dot / nrm2). Same launch shape for both legs: 8 warps per
// block, ONE problem per warp (the warp packing model), full warps only.
//
// Compilation: nvcc -std=c++17 -arch=native -O3 --expt-relaxed-constexpr -I.. bench_nvwarp_l1.cu -o bench_nvwarp_l1
// Usage: ./bench_nvwarp_l1
//
// Every leg is correctness-gated against a host reference before it is timed —
// a wrong result or failed launch prints FAIL and exits nonzero rather than
// timing garbage.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../glass.cuh"
#include "../glass-nvidia.cuh"   // glass::nvidia::warp::{reduce,dot,nrm2}
#include "timing_common.cuh"

static const int WARPS = 8;      // warps (= problems) per block
static int NPROB = 8192;         // set per section in main

// ─── kernels: p = global warp index = one problem per warp ───────────────────
template<typename T, uint32_t N>
__global__ void kw_reduce(T* x, int np) {
    int p = blockIdx.x * blockDim.y + threadIdx.y; if (p >= np) return;
    glass::warp::reduce<T, N>(x + (size_t)p * N);
}
template<typename T, uint32_t N>
__global__ void kw_dot(T* x, T* y, T* out, int np) {
    int p = blockIdx.x * blockDim.y + threadIdx.y; if (p >= np) return;
    T r = glass::warp::dot<T, N>(x + (size_t)p * N, y + (size_t)p * N);
    if ((threadIdx.x & 31) == 0) out[p] = r;
}
template<typename T, uint32_t N>
__global__ void kw_nrm2(T* x, T* out, int np) {
    int p = blockIdx.x * blockDim.y + threadIdx.y; if (p >= np) return;
    T r = glass::warp::nrm2<T, N>(x + (size_t)p * N);
    if ((threadIdx.x & 31) == 0) out[p] = r;
}

template<typename T>
__device__ T* warp_scratch(char* s) {
    constexpr size_t B = glass::nvidia::warp::warp_reduce_scratch_bytes<T>();
    return reinterpret_cast<T*>(s + threadIdx.y * (B ? B : sizeof(T)));
}
template<typename T, uint32_t N>
__global__ void kn_reduce(T* x, int np) {
    extern __shared__ char s[];
    int p = blockIdx.x * blockDim.y + threadIdx.y; if (p >= np) return;
    glass::nvidia::warp::reduce<T, N>(x + (size_t)p * N, warp_scratch<T>(s));
}
template<typename T, uint32_t N>
__global__ void kn_dot(T* x, T* y, T* out, int np) {
    extern __shared__ char s[];
    int p = blockIdx.x * blockDim.y + threadIdx.y; if (p >= np) return;
    glass::nvidia::warp::dot<T, N>(x + (size_t)p * N, y + (size_t)p * N,
                                   out + p, warp_scratch<T>(s));
}
template<typename T, uint32_t N>
__global__ void kn_nrm2(T* x, T* out, int np) {
    extern __shared__ char s[];
    int p = blockIdx.x * blockDim.y + threadIdx.y; if (p >= np) return;
    glass::nvidia::warp::nrm2<T, N>(x + (size_t)p * N, out + p, warp_scratch<T>(s));
}

// ─── harness ─────────────────────────────────────────────────────────────────
template<typename T>
static void fill(std::vector<T>& h) {
    for (size_t i = 0; i < h.size(); i++)
        h[i] = (T)0.25 * (T)((int)(i % 7) - 3);
}

template<typename T>
static bool close(T a, T b) {
    double tol = sizeof(T) == 4 ? 1e-4 : 1e-10;
    double d = std::fabs((double)a - (double)b);
    return d <= tol * (1.0 + std::fabs((double)b));
}

template<typename T, uint32_t N>
static void run_op(const char* dt) {
    const size_t elems = (size_t)NPROB * N;
    std::vector<T> hx(elems), hy(elems), hout(NPROB);
    fill(hx); fill(hy);
    for (size_t i = 0; i < elems; i++) hy[i] = hy[(i * 31 + 7) % elems];
    T *dx, *dy, *dout;
    cudaMalloc(&dx, elems * sizeof(T)); cudaMalloc(&dy, elems * sizeof(T));
    cudaMalloc(&dout, NPROB * sizeof(T));
    const dim3 blk(32, WARPS);
    const int grid = (NPROB + WARPS - 1) / WARPS;
    constexpr size_t SB = glass::nvidia::warp::warp_reduce_scratch_bytes<T>();
    const size_t smem = WARPS * (SB ? SB : sizeof(T));
    const int reps = NPROB >= 8192 ? 250 : NPROB >= 1024 ? 500 : 2000;

    // host references (problem 0 and NPROB-1 spot-checked per problem below)
    auto refs = [&](int p, double& rsum, double& rdot, double& rnrm) {
        rsum = rdot = rnrm = 0.0;
        for (uint32_t i = 0; i < N; i++) {
            double xi = (double)hx[(size_t)p * N + i], yi = (double)hy[(size_t)p * N + i];
            rsum += xi; rdot += xi * yi; rnrm += xi * xi;
        }
        rnrm = std::sqrt(rnrm);
    };
    auto check = [&](const char* leg, int which) -> bool {   // 0=reduce 1=dot 2=nrm2
        cudaMemcpy(dx, hx.data(), elems * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy.data(), elems * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemset(dout, 0, NPROB * sizeof(T));
        switch (which) {
            case 0: strcmp(leg, "cub") ? kw_reduce<T, N><<<grid, blk>>>(dx, NPROB)
                                       : kn_reduce<T, N><<<grid, blk, smem>>>(dx, NPROB); break;
            case 1: strcmp(leg, "cub") ? kw_dot<T, N><<<grid, blk>>>(dx, dy, dout, NPROB)
                                       : kn_dot<T, N><<<grid, blk, smem>>>(dx, dy, dout, NPROB); break;
            case 2: strcmp(leg, "cub") ? kw_nrm2<T, N><<<grid, blk>>>(dx, dout, NPROB)
                                       : kn_nrm2<T, N><<<grid, blk, smem>>>(dx, dout, NPROB); break;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { printf("FAIL launch %s\n", leg); return false; }
        std::vector<T> gx(elems), gout(NPROB);
        cudaMemcpy(gx.data(), dx, elems * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(gout.data(), dout, NPROB * sizeof(T), cudaMemcpyDeviceToHost);
        for (int p : {0, NPROB - 1}) {
            double rs, rd, rn; refs(p, rs, rd, rn);
            T got = which == 0 ? gx[(size_t)p * N] : gout[p];
            double want = which == 0 ? rs : which == 1 ? rd : rn;
            if (!close(got, (T)want)) {
                printf("FAIL check op=%d leg=%s N=%u p=%d got=%g want=%g\n",
                       which, leg, N, p, (double)got, want);
                return false;
            }
        }
        return true;
    };

    const char* names[3] = {"reduce", "dot", "nrm2"};
    for (int which = 0; which < 3; which++) {
        if (!check("simt", which) || !check("cub", which)) exit(1);
        cudaMemcpy(dx, hx.data(), elems * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy.data(), elems * sizeof(T), cudaMemcpyHostToDevice);
        double simt = -1, cub = -1, simt_spread = 0, cub_spread = 0;
        switch (which) {
            case 0:
                simt = tc_time_ns_per_prob([&]{ kw_reduce<T, N><<<grid, blk>>>(dx, NPROB); }, reps, NPROB);
                simt_spread = tc_last_spread_pct();
                cub  = tc_time_ns_per_prob([&]{ kn_reduce<T, N><<<grid, blk, smem>>>(dx, NPROB); }, reps, NPROB);
                cub_spread = tc_last_spread_pct();
                break;
            case 1:
                simt = tc_time_ns_per_prob([&]{ kw_dot<T, N><<<grid, blk>>>(dx, dy, dout, NPROB); }, reps, NPROB);
                simt_spread = tc_last_spread_pct();
                cub  = tc_time_ns_per_prob([&]{ kn_dot<T, N><<<grid, blk, smem>>>(dx, dy, dout, NPROB); }, reps, NPROB);
                cub_spread = tc_last_spread_pct();
                break;
            case 2:
                simt = tc_time_ns_per_prob([&]{ kw_nrm2<T, N><<<grid, blk>>>(dx, dout, NPROB); }, reps, NPROB);
                simt_spread = tc_last_spread_pct();
                cub  = tc_time_ns_per_prob([&]{ kn_nrm2<T, N><<<grid, blk, smem>>>(dx, dout, NPROB); }, reps, NPROB);
                cub_spread = tc_last_spread_pct();
                break;
        }
        printf("%-6s N=%-3u | simt=%8.3f spread=%5.2f%%  cub=%8.3f spread=%5.2f%% | cub/simt=%.3f -> %s\n",
               names[which], N, simt, simt_spread, cub, cub_spread, cub / simt,
               cub < simt * 0.98 ? "CUB" : simt < cub * 0.98 ? "SIMT" : "tie");
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dout);
}

template<typename T>
static void run_dtype(const char* dt) {
    printf("#### NPROB=%d  dtype=%s  (8 warps/block, 1 problem/warp) ####\n", NPROB, dt);
    run_op<T, 4>(dt);  run_op<T, 8>(dt);   run_op<T, 16>(dt);  run_op<T, 32>(dt);
    run_op<T, 64>(dt); run_op<T, 128>(dt); run_op<T, 256>(dt);
}

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    tc_warm_gpu();
    for (int np : {64, 1024, 8192}) {
        NPROB = np;
        run_dtype<float>("f32");
        run_dtype<double>("f64");
    }
    return 0;
}
