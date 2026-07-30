// bench_rect.cu — warp/block sweep over RECTANGULAR gemv/gemm shapes (the mega
// sweep is square-only, but consumers' Jacobians are rectangular):
//   gemv (M,N):   tall {64x8, 128x16, 256x32}  +  wide {8x64, 16x128, 32x256}
//   gemm (M,K,N): {(32,8,32),(8,32,8),(64,16,16),(16,64,16),(6,6,64),(64,6,6)}
//                 (C is MxN, contraction K — glass::gemm<T,M,N,K> template order)
//
// A SEPARATE harness (not an extension of bench_mega_sweep.cu) on purpose:
// the mega sweep is the MathDx-heavy source whose hash keys the prebuilt ladder
// cache and whose square-N grammar feeds glass-defaults.cuh regeneration —
// touching it would invalidate that cache and risk the ladder parser, while
// this leg needs no MathDx and compiles in seconds.
//
// NVIDIA leg SKIPPED for rectangular shapes: forcing cuBLASDx here would need
// new per-(M,N,K)/(M,N) DEFINE_NVIDIA_* descriptor instantiations; per-shape
// cuBLASDx-vs-SIMT decisions already live in the `shapes` leg (bench/autotune.py
// → src/nvidia/tuning_table.cuh) and rectangular vendor coverage belongs there.
//
// Same methodology + output grammar as bench_mega_sweep.cu (ns/problem =
// wall/(reps*NPROB), min of 3 trials; BLOCK TB∈{32,64,128,256} vs WARP
// WPB∈{1..32}; NPROB batching), rows keyed by shape instead of square N.
//
// Compile: nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1
//          -I.. -I../src bench_rect.cu -o bench_rect        (no MathDx needed)
// Usage:   ./bench_rect [nprob=8192] [reps=500] [dtype=f32|f64]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>

#include "../glass.cuh"

static int NPROB = 8192;

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
}

template<typename F>
static double time_ns_per_prob(F launch, int reps) {
    launch(); cudaDeviceSynchronize();
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < reps; r++) launch();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = elapsed_ms(t0, t1) * 1e6 / ((double)reps * NPROB);
        if (ns < best) best = ns;
    }
    return best;
}

// ─── gemv: y(M) = A(MxN)·x(N), one problem per block / per warp ──────────────
template<typename T,int M,int N> __global__ void kb_gemv(T* A, T* x, T* y) {
    int p = blockIdx.x;
    glass::block::gemv<T,M,N>((T)1, A+(size_t)p*M*N, x+(size_t)p*N, (T)0, y+(size_t)p*M);
}
template<typename T,int M,int N> __global__ void kw_gemv(T* A, T* x, T* y, int np) {
    int p = blockIdx.x*blockDim.y+threadIdx.y; if (p>=np) return;
    glass::warp::gemv<T,M,N>((T)1, A+(size_t)p*M*N, x+(size_t)p*N, (T)0, y+(size_t)p*M);
}

// ─── gemm: C(MxN) = A(MxK)·B(KxN) — glass template order <T,M,N,K> ──────────
template<typename T,int M,int K,int N> __global__ void kb_gemm(T* A, T* B, T* C) {
    int p = blockIdx.x;
    glass::block::gemm<T,M,N,K>((T)1, A+(size_t)p*M*K, B+(size_t)p*K*N, (T)0, C+(size_t)p*M*N);
}
template<typename T,int M,int K,int N> __global__ void kw_gemm(T* A, T* B, T* C, int np) {
    int p = blockIdx.x*blockDim.y+threadIdx.y; if (p>=np) return;
    glass::warp::gemm<T,M,N,K>((T)1, A+(size_t)p*M*K, B+(size_t)p*K*N, (T)0, C+(size_t)p*M*N);
}

// Shared 2-way sweep + report: prints "| BLOCK tb..=  | WARP w..=  || block tbX=..
// warp wY=..  -> WINNER (m.mmx)" — same summary grammar as the mega sweep.
template<typename LB, typename LW>
static void sweep_and_report(LB launch_block, LW launch_warp, int reps) {
    double best_block=1e30, best_warp=1e30; int best_tb=0, best_wpb=0;
    printf(" | BLOCK");
    for (int TB : {32, 64, 128, 256}) {
        double ns = time_ns_per_prob([&]{ launch_block(TB); }, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best_block) { best_block = ns; best_tb = TB; }
    }
    printf("  | WARP");
    for (int WPB : {1, 2, 4, 8, 16, 32}) {
        if (WPB > NPROB) break;
        double ns = time_ns_per_prob([&]{ launch_warp(WPB); }, reps);
        printf("  w%d=%.2f", WPB, ns);
        if (ns < best_warp) { best_warp = ns; best_wpb = WPB; }
    }
    const char* winner = best_warp < best_block ? "WARP" : "BLOCK";
    double margin = best_warp < best_block ? best_block/best_warp : best_warp/best_block;
    printf("  || block tb%d=%.2f  warp w%d=%.2f  -> %s (%.2fx)\n",
           best_tb, best_block, best_wpb, best_warp, winner, margin);
}

// Fill dst with NPROB copies of a deterministic elems-long tile.
template<typename T>
static void fill_tiled(T* dst, size_t elems) {
    T* h = (T*)malloc(elems*sizeof(T));
    for (size_t i = 0; i < elems; i++) h[i] = (T)(0.25 + 0.1*(double)((i*7+3) % 11));
    cudaMemcpy(dst, h, elems*sizeof(T), cudaMemcpyHostToDevice);
    for (size_t p = 1; p < (size_t)NPROB; p++)
        cudaMemcpy(dst + p*elems, dst, elems*sizeof(T), cudaMemcpyDeviceToDevice);
    free(h);
}

template<typename T,int M,int N>
static void bench_gemv_shape(int reps) {
    T *A, *x, *y;
    cudaMalloc(&A, (size_t)NPROB*M*N*sizeof(T));
    cudaMalloc(&x, (size_t)NPROB*N*sizeof(T));
    cudaMalloc(&y, (size_t)NPROB*M*sizeof(T));
    fill_tiled(A, (size_t)M*N);
    cudaMemset(x, 1, (size_t)NPROB*N*sizeof(T)); cudaMemset(y, 0, (size_t)NPROB*M*sizeof(T));
    printf("gemv  M=%-3d N=%-3d", M, N);
    sweep_and_report(
        [&](int TB){ kb_gemv<T,M,N><<<dim3(NPROB),dim3(TB)>>>(A, x, y); },
        [&](int WPB){ kw_gemv<T,M,N><<<dim3((NPROB+WPB-1)/WPB),dim3(32,WPB)>>>(A, x, y, NPROB); },
        reps);
    cudaFree(A); cudaFree(x); cudaFree(y);
}

template<typename T,int M,int K,int N>
static void bench_gemm_shape(int reps) {
    T *A, *B, *C;
    cudaMalloc(&A, (size_t)NPROB*M*K*sizeof(T));
    cudaMalloc(&B, (size_t)NPROB*K*N*sizeof(T));
    cudaMalloc(&C, (size_t)NPROB*M*N*sizeof(T));
    fill_tiled(A, (size_t)M*K); fill_tiled(B, (size_t)K*N);
    cudaMemset(C, 0, (size_t)NPROB*M*N*sizeof(T));
    printf("gemm  M=%-3d K=%-3d N=%-3d", M, K, N);
    sweep_and_report(
        [&](int TB){ kb_gemm<T,M,K,N><<<dim3(NPROB),dim3(TB)>>>(A, B, C); },
        [&](int WPB){ kw_gemm<T,M,K,N><<<dim3((NPROB+WPB-1)/WPB),dim3(32,WPB)>>>(A, B, C, NPROB); },
        reps);
    cudaFree(A); cudaFree(B); cudaFree(C);
}

template<typename T> static void run_all(int reps) {
    // gemv: tall then wide
    bench_gemv_shape<T,64,8>(reps);   bench_gemv_shape<T,128,16>(reps); bench_gemv_shape<T,256,32>(reps);
    bench_gemv_shape<T,8,64>(reps);   bench_gemv_shape<T,16,128>(reps); bench_gemv_shape<T,32,256>(reps);
    printf("\n");
    // gemm (M,K,N): C(MxN) = A(MxK)·B(KxN)
    bench_gemm_shape<T,32,8,32>(reps);  bench_gemm_shape<T,8,32,8>(reps);
    bench_gemm_shape<T,64,16,16>(reps); bench_gemm_shape<T,16,64,16>(reps);
    bench_gemm_shape<T,6,6,64>(reps);   bench_gemm_shape<T,64,6,6>(reps);
    printf("\n");
}

int main(int argc, char** argv) {
    NPROB    = (argc > 1) ? atoi(argv[1]) : 8192;
    int reps = (argc > 2) ? atoi(argv[2]) : 500;
    const char* dt = (argc > 3) ? argv[3] : "f32";
    bool f64 = (strcmp(dt, "f64") == 0 || strcmp(dt, "fp64") == 0 || strcmp(dt, "double") == 0);
    printf("# rect sweep | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n", NPROB, reps, f64 ? "f64" : "f32");
    printf("# contenders: BLOCK(SIMT, TB swept) | WARP(WPB swept) — nvidia leg skipped for rectangular shapes (see header)\n");
    if (f64) run_all<double>(reps);
    else     run_all<float>(reps);
    return 0;
}
