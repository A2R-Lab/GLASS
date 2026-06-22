// bench_warp_vs_block.cu — warp-per-problem vs one-block-per-problem THROUGHPUT.
//
// The other benches measure single-block LATENCY (one problem, <<<1,256>>>). This
// one measures THROUGHPUT: launch a grid of NPROB independent small problems and
// compare two packing models as a function of problem size N:
//
//   BLOCK model — one block per problem, <<<NPROB, TB>>>, TB ∈ {32,64,128,256}
//   WARP  model — one warp per problem,  <<<NPROB/WPB, dim3(32,WPB)>>>,
//                 WPB warps/block ∈ {1,2,4,8,16,32}
//
// Metric: nanoseconds per problem (wall time / (reps*NPROB)). Lower = better.
// The crossover (small N → warp wins by filling the SMs; larger N → block wins as
// one problem saturates a block) is the thing we want to locate per op.
//
// Ops: gemm (NxN*NxN), chol (NxN SPD factor), posv (NxN SPD solve = chol+2 trsv).
// All have matching glass::<op> and glass::warp::<op> compile-time-N forms.
//
// Compile (pure-SIMT, no MathDx needed):
//   nvcc -std=c++17 -arch=sm_XX -O3 -Xptxas -O1 -I.. -I../src
//        bench_warp_vs_block.cu -o bench_warp_vs_block
// Usage: ./bench_warp_vs_block [reps]   (default 200)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include "../glass.cuh"

static const int NPROB = 8192;   // total independent problems per launch

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
}

// ─── BLOCK model: block b owns problem b ─────────────────────────────────────
template<int N>
__global__ void kb_gemm(const float* A, const float* B, float* C) {
    int p = blockIdx.x;
    glass::gemm<float, N, N, N>(1.f, (float*)(A + p*N*N), (float*)(B + p*N*N), 0.f, C + p*N*N);
}
template<int N>
__global__ void kb_chol(float* A) {
    int p = blockIdx.x;
    glass::cholDecomp_InPlace<float, N>(A + p*N*N);
}
template<int N>
__global__ void kb_posv(float* A, float* b) {
    int p = blockIdx.x;
    glass::posv<float, N>(A + p*N*N, b + p*N);
}

// ─── WARP model: warp (blockIdx.x*WPB + threadIdx.y) owns its problem ─────────
template<int N>
__global__ void kw_gemm(const float* A, const float* B, float* C, int nprob) {
    int p = blockIdx.x * blockDim.y + threadIdx.y;
    if (p >= nprob) return;
    glass::warp::gemm<float, N, N, N>(1.f, (float*)(A + p*N*N), (float*)(B + p*N*N), 0.f, C + p*N*N);
}
template<int N>
__global__ void kw_chol(float* A, int nprob) {
    int p = blockIdx.x * blockDim.y + threadIdx.y;
    if (p >= nprob) return;
    glass::warp::cholDecomp_InPlace<float, N>(A + p*N*N);
}
template<int N>
__global__ void kw_posv(float* A, float* b, int nprob) {
    int p = blockIdx.x * blockDim.y + threadIdx.y;
    if (p >= nprob) return;
    glass::warp::posv<float, N>(A + p*N*N, b + p*N);
}

enum Op { GEMM, CHOL, POSV };
static const char* op_name(Op o) { return o == GEMM ? "gemm" : o == CHOL ? "chol" : "posv"; }

template<int N>
static void launch_block(Op op, int TB, float* A, float* B, float* C, float* b) {
    dim3 grid(NPROB), blk(TB);
    if (op == GEMM)      kb_gemm<N><<<grid, blk>>>(A, B, C);
    else if (op == CHOL) kb_chol<N><<<grid, blk>>>(A);
    else                 kb_posv<N><<<grid, blk>>>(A, b);
}
template<int N>
static void launch_warp(Op op, int WPB, float* A, float* B, float* C, float* b) {
    dim3 grid((NPROB + WPB - 1) / WPB), blk(32, WPB);
    if (op == GEMM)      kw_gemm<N><<<grid, blk>>>(A, B, C, NPROB);
    else if (op == CHOL) kw_chol<N><<<grid, blk>>>(A, NPROB);
    else                 kw_posv<N><<<grid, blk>>>(A, b, NPROB);
}

// time a launch closure: reps back-to-back, wall clock, ns/problem (min of 3 trials)
template<typename F>
static double time_ns_per_prob(F launch, int reps) {
    launch(); cudaDeviceSynchronize();           // warm up
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

template<int N>
static void bench_size(Op op, int reps) {
    float *A, *B, *C, *b;
    size_t mm = (size_t)NPROB * N * N;
    cudaMalloc(&A, mm * sizeof(float));
    cudaMalloc(&B, mm * sizeof(float));
    cudaMalloc(&C, mm * sizeof(float));
    cudaMalloc(&b, (size_t)NPROB * N * sizeof(float));
    // init: SPD-ish A (diagonally dominant) replicated; B/b arbitrary nonzero.
    float* hA = (float*)malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            hA[i + j*N] = (i == j) ? (float)(N + 2) : 0.1f * ((i + 2*j) % 5);
    // broadcast the one tile to all problems via a small staging copy
    cudaMemcpy(A, hA, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice);
    for (size_t p = 1; p < (size_t)NPROB; p++)
        cudaMemcpy(A + p*N*N, A, (size_t)N*N*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(B, 1, mm * sizeof(float));
    cudaMemset(b, 1, (size_t)NPROB * N * sizeof(float));
    free(hA);

    double best_block = 1e30, best_warp = 1e30;
    int best_tb = 0, best_wpb = 0;
    printf("%-5s N=%-3d | BLOCK", op_name(op), N);
    for (int TB : {32, 64, 128, 256}) {
        // reset A each config for the destructive ops so we factor real SPD once;
        // (timing is control-flow-identical regardless, but keep it clean)
        double ns = time_ns_per_prob([&]{ launch_block<N>(op, TB, A, B, C, b); }, reps);
        printf("  tb%d=%.1f", TB, ns);
        if (ns < best_block) { best_block = ns; best_tb = TB; }
    }
    printf("  | WARP");
    for (int WPB : {1, 2, 4, 8, 16, 32}) {
        double ns = time_ns_per_prob([&]{ launch_warp<N>(op, WPB, A, B, C, b); }, reps);
        printf("  w%d=%.1f", WPB, ns);
        if (ns < best_warp) { best_warp = ns; best_wpb = WPB; }
    }
    const char* winner = best_warp < best_block ? "WARP" : "BLOCK";
    printf("  || best block tb%d=%.1f  best warp w%d=%.1f  -> %s (%.2fx)\n",
           best_tb, best_block, best_wpb, best_warp, winner,
           best_warp < best_block ? best_block / best_warp : best_warp / best_block);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(b);
}

int main(int argc, char** argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 200;
    printf("# warp-per-problem vs one-block-per-problem, NPROB=%d, reps=%d, ns/problem\n", NPROB, reps);
    for (Op op : {GEMM, CHOL, POSV}) {
        bench_size<4>(op, reps);
        bench_size<6>(op, reps);
        bench_size<8>(op, reps);
        bench_size<12>(op, reps);
        bench_size<16>(op, reps);
        bench_size<24>(op, reps);
        bench_size<32>(op, reps);
        printf("\n");
    }
    return 0;
}
