// bench_blas2.cu — warp/block scaling sweep for the ops the mega sweep misses:
//   syrk    C = A·Aᵀ                (L3, warp:: variant exists)
//   syr2k   C = A·Bᵀ + B·Aᵀ        (L3, warp:: variant exists)
//   ldlt    A = L·D·Lᵀ in place     (L3, warp:: variant exists)
//   ldlt_solve  ldlt + solve        (L3 factor+solve, warp:: variant exists)
//   inv     Gauss-Jordan on [A|I]   (L3, BLOCK-ONLY — no warp:: variant)
//   trmv    y = tril(A)·x           (L2, BLOCK-ONLY)
//   ger     A += α·x·yᵀ             (L2, BLOCK-ONLY)
//
// Same methodology + output grammar as bench_mega_sweep.cu (ns/problem =
// wall/(reps*NPROB), min of 3 trials; one problem per block / per warp; NPROB
// batching) so tune.py's parser conventions carry over. TWO contenders only:
// none of these ops has a glass::nvidia::{block,thread} counterpart, so there is no vendor leg
// (2-way BLOCK TB∈{32,64,128,256} vs WARP WPB∈{1..32} where warp:: exists).
//
// Timing-only: inputs are factored/overwritten in place across reps (no per-rep
// reload) — identical policy for both contenders, apples-to-apples (same as the
// mega sweep's chol/posv rows).
//
// inv uses the augmented [A | I] layout (column-major N x 2N per problem, see
// src/base/L3/inv.cuh) with (2N+1)-element shared scratch; ldlt uses the
// (N+1)-element scratch from ldlt_scratch_bytes (unread on the non-pivoted path).
//
// Compile: nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1
//          -I.. -I../src bench_blas2.cu -o bench_blas2      (no MathDx needed)
// Usage:   ./bench_blas2 [nprob=8192] [reps=500] [dtype=f32|f64]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include "timing_common.cuh"

#include "../glass.cuh"

static int NPROB = 8192;


// ─── BLOCK model: block b owns problem b ─────────────────────────────────────
template<typename T,int N> __global__ void kb_syrk (T* A, T* C) { int p=blockIdx.x; glass::block::syrk<T,N,N>((T)1, A+(size_t)p*N*N, (T)0, C+(size_t)p*N*N); }
template<typename T,int N> __global__ void kb_syr2k(T* A, T* B, T* C) { int p=blockIdx.x; glass::block::syr2k<T,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N); }
template<typename T,int N> __global__ void kb_ldlt (T* A) { __shared__ T s[N+1]; int p=blockIdx.x; glass::block::ldlt<T,N>(A+(size_t)p*N*N, s); }
template<typename T,int N> __global__ void kb_ldltsv(T* A, T* x) { __shared__ T s[N+1]; int p=blockIdx.x; glass::block::ldlt<T,N>(A+(size_t)p*N*N, s); glass::block::ldlt_solve<T,N>(A+(size_t)p*N*N, x+p*N); }
template<typename T,int N> __global__ void kb_inv  (T* G) { __shared__ T s[2*N+1]; int p=blockIdx.x; glass::block::inv<T,N>(G+(size_t)p*2*N*N, s); }
template<typename T,int N> __global__ void kb_trmv (T* A, T* x, T* y) { int p=blockIdx.x; glass::block::trmv<T,N>(A+(size_t)p*N*N, x+p*N, y+p*N); }
template<typename T,int N> __global__ void kb_ger  (T* A, T* x, T* y) { int p=blockIdx.x; glass::block::ger<T,N,N>((T)1, x+p*N, y+p*N, A+(size_t)p*N*N); }

// ─── WARP model: warp (blockIdx.x*WPB + threadIdx.y) owns its problem ─────────
// Only syrk/syr2k/ldlt/ldlt_solve have warp variants (inv/trmv/ger are block-only).
template<typename T,int N> __global__ void kw_syrk (T* A, T* C, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::syrk<T,N,N>((T)1, A+(size_t)p*N*N, (T)0, C+(size_t)p*N*N); }
template<typename T,int N> __global__ void kw_syr2k(T* A, T* B, T* C, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::syr2k<T,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N); }
template<typename T,int N> __global__ void kw_ldlt (T* A, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::ldlt<T,N>(A+(size_t)p*N*N); }
template<typename T,int N> __global__ void kw_ldltsv(T* A, T* x, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::ldlt<T,N>(A+(size_t)p*N*N); glass::warp::ldlt_solve<T,N>(A+(size_t)p*N*N, x+p*N); }

enum Op { SYRK, SYR2K, LDLT, LDLTSV, INV, TRMV, GER, NOP };
static const char* op_name(Op o) {
    const char* n[] = {"syrk","syr2k","ldlt","ldlt_solve","inv","trmv","ger"};
    return n[o];
}
static bool has_warp(Op o) { return o == SYRK || o == SYR2K || o == LDLT || o == LDLTSV; }

template<typename T,int N>
static void launch_block(Op op, int TB, T* A, T* B, T* C, T* G, T* x, T* y) {
    dim3 grid(NPROB), blk(TB);
    switch (op) {
        case SYRK:   kb_syrk  <T,N><<<grid,blk>>>(A, C); break;
        case SYR2K:  kb_syr2k <T,N><<<grid,blk>>>(A, B, C); break;
        case LDLT:   kb_ldlt  <T,N><<<grid,blk>>>(A); break;
        case LDLTSV: kb_ldltsv<T,N><<<grid,blk>>>(A, x); break;
        case INV:    kb_inv   <T,N><<<grid,blk>>>(G); break;
        case TRMV:   kb_trmv  <T,N><<<grid,blk>>>(A, x, y); break;
        case GER:    kb_ger   <T,N><<<grid,blk>>>(A, x, y); break;
        default: break;
    }
}
template<typename T,int N>
static void launch_warp(Op op, int WPB, T* A, T* B, T* C, T* x) {
    dim3 grid((NPROB + WPB - 1) / WPB), blk(32, WPB);
    switch (op) {
        case SYRK:   kw_syrk  <T,N><<<grid,blk>>>(A, C, NPROB); break;
        case SYR2K:  kw_syr2k <T,N><<<grid,blk>>>(A, B, C, NPROB); break;
        case LDLT:   kw_ldlt  <T,N><<<grid,blk>>>(A, NPROB); break;
        case LDLTSV: kw_ldltsv<T,N><<<grid,blk>>>(A, x, NPROB); break;
        default: break;
    }
}

// Measurement core lives in timing_common.cuh (min-of-3 + trial spread +
// mutation invariant). g_row_spread accumulates the worst trial spread across
// a row; the row printer emits `spread<=X%` so tune.py can flag jittery rows.
static double g_row_spread = 0.0;

template<typename F>
static double time_ns_per_prob(F launch, int reps) {
    double ns = tc_time_ns_per_prob(launch, reps, NPROB);
    if (tc_last_spread_pct() > g_row_spread) g_row_spread = tc_last_spread_pct();
    return ns;
}

template<typename T,int N>
static void bench_size(Op op, int reps) {
    T *A, *B, *C, *x, *y, *G = nullptr;
    size_t mm = (size_t)NPROB * N * N, vv = (size_t)NPROB * N;
    cudaMalloc(&A, mm*sizeof(T)); cudaMalloc(&B, mm*sizeof(T)); cudaMalloc(&C, mm*sizeof(T));
    cudaMalloc(&x, vv*sizeof(T)); cudaMalloc(&y, vv*sizeof(T));
    // diagonally-dominant A (valid for ldlt/inv/trmv); broadcast one tile to all problems.
    T* hA = (T*)malloc((size_t)N*N*sizeof(T));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) hA[i+j*N] = (i==j)?(T)(N+2):(T)(0.1*((i+2*j)%5));
    cudaMemcpy(A, hA, (size_t)N*N*sizeof(T), cudaMemcpyHostToDevice);
    for (size_t p=1;p<(size_t)NPROB;p++) cudaMemcpy(A+p*N*N, A, (size_t)N*N*sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemset(B, 1, mm*sizeof(T)); cudaMemset(C, 0, mm*sizeof(T));
    cudaMemset(x, 1, vv*sizeof(T)); cudaMemset(y, 1, vv*sizeof(T));
    if (op == INV) {   // augmented [A | I], column-major N x 2N per problem
        size_t gg = (size_t)NPROB * 2 * N * N;
        cudaMalloc(&G, gg*sizeof(T));
        T* hG = (T*)malloc((size_t)2*N*N*sizeof(T));
        for (int j=0;j<N;j++) for (int i=0;i<N;i++) { hG[i+j*N] = hA[i+j*N]; hG[i+(size_t)(N+j)*N] = (i==j)?(T)1:(T)0; }
        cudaMemcpy(G, hG, (size_t)2*N*N*sizeof(T), cudaMemcpyHostToDevice);
        for (size_t p=1;p<(size_t)NPROB;p++) cudaMemcpy(G+p*2*N*N, G, (size_t)2*N*N*sizeof(T), cudaMemcpyDeviceToDevice);
        free(hG);
    }
    free(hA);

    double best_block=1e30, best_warp=1e30; int best_tb=0, best_wpb=0;
    g_row_spread = 0.0;
    printf("%-6s N=%-3d | BLOCK", op_name(op), N);
    for (int TB : {32, 64, 128, 256}) {
        double ns = time_ns_per_prob([&]{ launch_block<T,N>(op, TB, A, B, C, G, x, y); }, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best_block) { best_block = ns; best_tb = TB; }
    }
    if (has_warp(op)) {
        printf("  | WARP");
        for (int WPB : {1, 2, 4, 8, 16, 32}) {
            if (WPB > NPROB) break;
            double ns = time_ns_per_prob([&]{ launch_warp<T,N>(op, WPB, A, B, C, x); }, reps);
            printf("  w%d=%.2f", WPB, ns);
            if (ns < best_warp) { best_warp = ns; best_wpb = WPB; }
        }
    }
    // 2-way winner (raw argmin; the tune.py picker re-decides under the shared margin)
    printf("  || block tb%d=%.2f", best_tb, best_block);
    if (has_warp(op)) {
        printf("  warp w%d=%.2f", best_wpb, best_warp);
        const char* winner = best_warp < best_block ? "WARP" : "BLOCK";
        double margin = best_warp < best_block ? best_block/best_warp : best_warp/best_block;
        printf("  spread<=%.1f%%  -> %s (%.2fx)\n", g_row_spread, winner, margin);
    } else {
        printf("  spread<=%.1f%%  -> BLOCK (1.00x)\n", g_row_spread);
    }
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(x); cudaFree(y);
    if (G) cudaFree(G);
}

template<typename T> static void run_all(int reps) {
    for (Op op : {SYRK, SYR2K, LDLT, LDLTSV, INV, TRMV, GER}) {
        bench_size<T,4>(op, reps);  bench_size<T,6>(op, reps);  bench_size<T,8>(op, reps);
        bench_size<T,12>(op, reps); bench_size<T,16>(op, reps); bench_size<T,24>(op, reps);
        bench_size<T,32>(op, reps); bench_size<T,48>(op, reps); bench_size<T,64>(op, reps);
        bench_size<T,96>(op, reps); bench_size<T,128>(op, reps);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    NPROB    = (argc > 1) ? atoi(argv[1]) : 8192;
    int reps = (argc > 2) ? atoi(argv[2]) : 500;
    const char* dt = (argc > 3) ? argv[3] : "f32";
    bool f64 = (strcmp(dt, "f64") == 0 || strcmp(dt, "fp64") == 0 || strcmp(dt, "double") == 0);
    printf("# blas2 sweep | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n", NPROB, reps, f64 ? "f64" : "f32");
    printf("# contenders: BLOCK(SIMT, TB swept) | WARP(WPB swept; syrk/syr2k/ldlt/ldlt_solve only) — no NVIDIA counterparts\n");
    tc_warm_gpu();                    // steady boost clocks before the first timed cell
    if (f64) run_all<double>(reps);
    else     run_all<float>(reps);
    return 0;
}
