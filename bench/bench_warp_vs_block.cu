// bench_warp_vs_block.cu — warp-per-problem vs one-block-per-problem.
//
// Two regimes, selected by NPROB (CLI arg 1):
//   NPROB large (throughput) — many independent small problems fill the GPU.
//   NPROB = 1   (latency)    — a SINGLE problem: <<<1,TB>>> block vs <<<1,32>>> warp.
//                              (answers "if there's only one op, warp or block?")
//
// Models:
//   BLOCK — one block per problem, <<<NPROB, TB>>>, TB ∈ {32,64,128,256}
//   WARP  — one warp per problem,  <<<ceil(NPROB/WPB), dim3(32,WPB)>>>,
//           WPB warps/block ∈ {1,2,4,8,16,32}
//
// Metric: ns per problem (wall / (reps*NPROB)), min of 3 trials. Lower = better.
//
// Ops (each has matching glass::<op> and glass::warp::<op> compile-time-N forms):
//   dot (L1)  gemv (L2)  gemm (L3)  chol (L3)  trsv (L3)  posv (L3)
//
// Compile (pure-SIMT, no MathDx):
//   nvcc -std=c++17 -arch=sm_XX -O3 -Xptxas -O1 -I.. -I../src
//        bench_warp_vs_block.cu -o bench_warp_vs_block
// Usage: ./bench_warp_vs_block [nprob=8192] [reps=500]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include "../glass.cuh"

static int NPROB = 8192;

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
}

// ─── BLOCK model: block b owns problem b ─────────────────────────────────────
template<int N> __global__ void kb_dot (float* x, float* y) { int p=blockIdx.x; glass::dot<float,N>(x+p*N, y+p*N); }
template<int N> __global__ void kb_gemv(float* A, float* x, float* y) { int p=blockIdx.x; glass::gemv<float,N,N>(1.f, A+(size_t)p*N*N, x+p*N, 0.f, y+p*N); }
template<int N> __global__ void kb_gemm(float* A, float* B, float* C) { int p=blockIdx.x; glass::gemm<float,N,N,N>(1.f, A+(size_t)p*N*N, B+(size_t)p*N*N, 0.f, C+(size_t)p*N*N); }
template<int N> __global__ void kb_chol(float* A) { int p=blockIdx.x; glass::cholDecomp_InPlace<float,N>(A+(size_t)p*N*N); }
template<int N> __global__ void kb_trsv(float* A, float* x) { int p=blockIdx.x; glass::trsv<float,N>(A+(size_t)p*N*N, x+p*N); }
template<int N> __global__ void kb_posv(float* A, float* b) { int p=blockIdx.x; glass::posv<float,N>(A+(size_t)p*N*N, b+p*N); }

// ─── WARP model: warp (blockIdx.x*WPB + threadIdx.y) owns its problem ─────────
template<int N> __global__ void kw_dot (float* x, float* y, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; float r=glass::warp::dot<float,N>(x+p*N, y+p*N); if((threadIdx.x&31)==0) y[p*N]=r; }
template<int N> __global__ void kw_gemv(float* A, float* x, float* y, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::gemv<float,N,N>(1.f, A+(size_t)p*N*N, x+p*N, 0.f, y+p*N); }
template<int N> __global__ void kw_gemm(float* A, float* B, float* C, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::gemm<float,N,N,N>(1.f, A+(size_t)p*N*N, B+(size_t)p*N*N, 0.f, C+(size_t)p*N*N); }
template<int N> __global__ void kw_chol(float* A, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::cholDecomp_InPlace<float,N>(A+(size_t)p*N*N); }
template<int N> __global__ void kw_trsv(float* A, float* x, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::trsv<float,N>(A+(size_t)p*N*N, x+p*N); }
template<int N> __global__ void kw_posv(float* A, float* b, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::posv<float,N>(A+(size_t)p*N*N, b+p*N); }

enum Op { DOT, GEMV, GEMM, CHOL, TRSV, POSV, NOP };
static const char* op_name(Op o) {
    const char* n[] = {"dot","gemv","gemm","chol","trsv","posv"};
    return n[o];
}

template<int N>
static void launch_block(Op op, int TB, float* A, float* B, float* C, float* x, float* y) {
    dim3 grid(NPROB), blk(TB);
    switch (op) {
        case DOT:  kb_dot <N><<<grid,blk>>>(x, y); break;
        case GEMV: kb_gemv<N><<<grid,blk>>>(A, x, y); break;
        case GEMM: kb_gemm<N><<<grid,blk>>>(A, B, C); break;
        case CHOL: kb_chol<N><<<grid,blk>>>(A); break;
        case TRSV: kb_trsv<N><<<grid,blk>>>(A, x); break;
        case POSV: kb_posv<N><<<grid,blk>>>(A, x); break;
        default: break;
    }
}
template<int N>
static void launch_warp(Op op, int WPB, float* A, float* B, float* C, float* x, float* y) {
    dim3 grid((NPROB + WPB - 1) / WPB), blk(32, WPB);
    switch (op) {
        case DOT:  kw_dot <N><<<grid,blk>>>(x, y, NPROB); break;
        case GEMV: kw_gemv<N><<<grid,blk>>>(A, x, y, NPROB); break;
        case GEMM: kw_gemm<N><<<grid,blk>>>(A, B, C, NPROB); break;
        case CHOL: kw_chol<N><<<grid,blk>>>(A, NPROB); break;
        case TRSV: kw_trsv<N><<<grid,blk>>>(A, x, NPROB); break;
        case POSV: kw_posv<N><<<grid,blk>>>(A, x, NPROB); break;
        default: break;
    }
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

template<int N>
static void bench_size(Op op, int reps) {
    float *A, *B, *C, *x, *y;
    size_t mm = (size_t)NPROB * N * N, vv = (size_t)NPROB * N;
    cudaMalloc(&A, mm*sizeof(float)); cudaMalloc(&B, mm*sizeof(float)); cudaMalloc(&C, mm*sizeof(float));
    cudaMalloc(&x, vv*sizeof(float)); cudaMalloc(&y, vv*sizeof(float));
    // diagonally-dominant A (valid for chol/trsv/posv); broadcast one tile to all problems.
    float* hA = (float*)malloc((size_t)N*N*sizeof(float));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) hA[i+j*N] = (i==j)?(float)(N+2):0.1f*((i+2*j)%5);
    cudaMemcpy(A, hA, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice);
    for (size_t p=1;p<(size_t)NPROB;p++) cudaMemcpy(A+p*N*N, A, (size_t)N*N*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(B, 1, mm*sizeof(float)); cudaMemset(C, 0, mm*sizeof(float));
    cudaMemset(x, 1, vv*sizeof(float)); cudaMemset(y, 1, vv*sizeof(float));
    free(hA);

    double best_block=1e30, best_warp=1e30; int best_tb=0, best_wpb=0;
    printf("%-5s N=%-3d | BLOCK", op_name(op), N);
    for (int TB : {32, 64, 128, 256}) {
        double ns = time_ns_per_prob([&]{ launch_block<N>(op, TB, A, B, C, x, y); }, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best_block) { best_block = ns; best_tb = TB; }
    }
    printf("  | WARP");
    for (int WPB : {1, 2, 4, 8, 16, 32}) {
        if (WPB > NPROB) break;
        double ns = time_ns_per_prob([&]{ launch_warp<N>(op, WPB, A, B, C, x, y); }, reps);
        printf("  w%d=%.2f", WPB, ns);
        if (ns < best_warp) { best_warp = ns; best_wpb = WPB; }
    }
    const char* winner = best_warp < best_block ? "WARP" : "BLOCK";
    printf("  || block tb%d=%.2f  warp w%d=%.2f  -> %s (%.2fx)\n",
           best_tb, best_block, best_wpb, best_warp, winner,
           best_warp < best_block ? best_block/best_warp : best_warp/best_block);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(x); cudaFree(y);
}

template<int N> static void bench_all_ops(int reps, Op only) {
    for (Op op : {DOT, GEMV, GEMM, CHOL, TRSV, POSV})
        if (only == NOP || op == only) bench_size<N>(op, reps);
}

int main(int argc, char** argv) {
    NPROB    = (argc > 1) ? atoi(argv[1]) : 8192;
    int reps = (argc > 2) ? atoi(argv[2]) : 500;
    printf("# warp-per-problem vs one-block-per-problem | NPROB=%d reps=%d | ns/problem (lower=better)\n", NPROB, reps);
    // group by size so each op's size-scaling is easy to read; ops interleaved per size.
    for (Op op : {DOT, GEMV, GEMM, CHOL, TRSV, POSV}) {
        bench_size<4>(op, reps);  bench_size<6>(op, reps);  bench_size<8>(op, reps);
        bench_size<12>(op, reps); bench_size<16>(op, reps); bench_size<24>(op, reps);
        bench_size<32>(op, reps);
        printf("\n");
    }
    return 0;
}
