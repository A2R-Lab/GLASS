// bench_mega_sweep.cu — three-contender scaling sweep:
//   WARP   — one warp per problem,   <<<ceil(NPROB/WPB), dim3(32,WPB)>>>, WPB ∈ {1..32}
//   BLOCK  — one block per problem,   <<<NPROB, TB>>>, TB ∈ {32,64,128,256} (pure-SIMT glass::)
//   NVIDIA — cuBLASDx/cuSOLVERDx,     <<<NPROB, nv_threads(N)>>>, descriptor-fixed (f32 to N128; f64 to N64)
//
// Answers "where do the breakevens fall on the warp → SIMT-block → MathDx ladder?" across
// problem size N and batch count NPROB (single-problem latency → GPU-saturating throughput).
//
// Ops (each has glass::<op>, glass::warp::<op>, and a glass::nvidia::<op> form):
//   dot (L1)  gemv (L2)  gemm (L3)  chol (L3)  trsv (L3, nvidia=trsm)  posv (L3)
//
// dtype: f32 (3-way, nvidia to N128) or f64 (3-way, nvidia to N64 — f64 vendor descriptors fit a lower smem cap).
// The nvidia leg is FORCED at every N: a DEFINE_NVIDIA_* macro is in scope for each N, so
// glass::nvidia::<op><float,N,...> resolves to the cuBLASDx/cuSOLVERDx specialization
// unconditionally (bypassing the shipped size-heuristic auto-dispatch) — we want the full
// vendor curve so the crossover with block/warp is visible, not just the heuristic's verdict.
//
// Metric: ns per problem (wall / (reps*NPROB)), min of 3 trials. Lower = better. Timing-only:
// inputs are factored/overwritten in place across reps (no per-rep reload) — uniform across
// all three contenders, so the comparison is apples-to-apples.
//
// Compile (3-way, needs MathDx — set MATHDX_ROOT):
//   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 -I.. -I../src
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
//        -DGLASS_BENCH_CUBLASDX -DGLASS_BENCH_CUSOLVERDX -DSMS=1200
//        -DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT -dlto
//        -lcusolverdx -lcublas -lcusolver -lcudart bench_mega_sweep.cu -o bench_mega_sweep
//   (omit the MathDx -I / -D / -l flags → compiles 2-way warp/block only, both dtypes.)
// Usage: ./bench_mega_sweep [nprob=8192] [reps=500] [dtype=f32|f64]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <type_traits>

#if defined(GLASS_BENCH_CUBLASDX)
#include <cublasdx.hpp>
#include "../glass-nvidia.cuh"     // pulls glass.cuh; CUB-backed L1 + cuBLASDx L2/L3
#define MEGA_NV_BLAS 1             // dot (CUB), gemv, gemm (cuBLASDx)
#else
#include "../glass.cuh"
#define MEGA_NV_BLAS 0
#endif

#if defined(GLASS_BENCH_CUSOLVERDX)
#define MEGA_NV_LAPACK 1           // chol, trsm, posv (cuSOLVERDx)
#else
#define MEGA_NV_LAPACK 0
#endif

static int NPROB = 8192;

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
}

// ─── BLOCK model: block b owns problem b ─────────────────────────────────────
template<typename T,int N> __global__ void kb_dot (T* x, T* y) { int p=blockIdx.x; glass::dot<T,N>(x+p*N, y+p*N); }
template<typename T,int N> __global__ void kb_gemv(T* A, T* x, T* y) { int p=blockIdx.x; glass::gemv<T,N,N>((T)1, A+(size_t)p*N*N, x+p*N, (T)0, y+p*N); }
template<typename T,int N> __global__ void kb_gemm(T* A, T* B, T* C) { int p=blockIdx.x; glass::gemm<T,N,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N); }
template<typename T,int N> __global__ void kb_chol(T* A) { int p=blockIdx.x; glass::cholDecomp_InPlace<T,N>(A+(size_t)p*N*N); }
template<typename T,int N> __global__ void kb_trsv(T* A, T* x) { int p=blockIdx.x; glass::trsv<T,N>(A+(size_t)p*N*N, x+p*N); }
template<typename T,int N> __global__ void kb_posv(T* A, T* b) { int p=blockIdx.x; glass::posv<T,N>(A+(size_t)p*N*N, b+p*N); }

// ─── WARP model: warp (blockIdx.x*WPB + threadIdx.y) owns its problem ─────────
template<typename T,int N> __global__ void kw_dot (T* x, T* y, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; T r=glass::warp::dot<T,N>(x+p*N, y+p*N); if((threadIdx.x&31)==0) y[p*N]=r; }
template<typename T,int N> __global__ void kw_gemv(T* A, T* x, T* y, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::gemv<T,N,N>((T)1, A+(size_t)p*N*N, x+p*N, (T)0, y+p*N); }
template<typename T,int N> __global__ void kw_gemm(T* A, T* B, T* C, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::gemm<T,N,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N); }
template<typename T,int N> __global__ void kw_chol(T* A, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::cholDecomp_InPlace<T,N>(A+(size_t)p*N*N); }
template<typename T,int N> __global__ void kw_trsv(T* A, T* x, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::trsv<T,N>(A+(size_t)p*N*N, x+p*N); }
template<typename T,int N> __global__ void kw_posv(T* A, T* b, int np) { int p=blockIdx.x*blockDim.y+threadIdx.y; if(p>=np)return; glass::warp::posv<T,N>(A+(size_t)p*N*N, b+p*N); }

enum Op { DOT, GEMV, GEMM, CHOL, TRSV, POSV, NOP };
static const char* op_name(Op o) {
    const char* n[] = {"dot","gemv","gemm","chol","trsv","posv"};
    return n[o];
}

template<typename T,int N>
static void launch_block(Op op, int TB, T* A, T* B, T* C, T* x, T* y) {
    dim3 grid(NPROB), blk(TB);
    switch (op) {
        case DOT:  kb_dot <T,N><<<grid,blk>>>(x, y); break;
        case GEMV: kb_gemv<T,N><<<grid,blk>>>(A, x, y); break;
        case GEMM: kb_gemm<T,N><<<grid,blk>>>(A, B, C); break;
        case CHOL: kb_chol<T,N><<<grid,blk>>>(A); break;
        case TRSV: kb_trsv<T,N><<<grid,blk>>>(A, x); break;
        case POSV: kb_posv<T,N><<<grid,blk>>>(A, x); break;
        default: break;
    }
}
template<typename T,int N>
static void launch_warp(Op op, int WPB, T* A, T* B, T* C, T* x, T* y) {
    dim3 grid((NPROB + WPB - 1) / WPB), blk(32, WPB);
    switch (op) {
        case DOT:  kw_dot <T,N><<<grid,blk>>>(x, y, NPROB); break;
        case GEMV: kw_gemv<T,N><<<grid,blk>>>(A, x, y, NPROB); break;
        case GEMM: kw_gemm<T,N><<<grid,blk>>>(A, B, C, NPROB); break;
        case CHOL: kw_chol<T,N><<<grid,blk>>>(A, NPROB); break;
        case TRSV: kw_trsv<T,N><<<grid,blk>>>(A, x, NPROB); break;
        case POSV: kw_posv<T,N><<<grid,blk>>>(A, x, NPROB); break;
        default: break;
    }
}

// ─── NVIDIA model: cuBLASDx / cuSOLVERDx, one block per problem ──────────────
// DEFINE_NVIDIA_* emit explicit specializations so the glass::nvidia::<op> call
// resolves to the vendor path unconditionally (forced, no size-heuristic dispatch).
// FLOAT: gemm 16/24/32/64 + gemv 4..64 are already cuBLASDx-specialized by
// glass-nvidia.cuh/tuning_table.cuh (those shipped specializations ARE the forced
// path) — we only add the float gaps. DOUBLE: nothing is shipped, so every size is
// defined via the *_PREC(..., double) macros. Double caps at N<=64 (smem: a 99KB
// opt-in limit fits f64 gemm only to 64, f64 chol/posv to ~96; we define <=64).
#if MEGA_NV_BLAS
static const int NV_DOT_TB = 256;   // CUB BlockReduce thread count for nvidia::dot
namespace glass { namespace nvidia {
    // float gaps (shipped: gemm 16/24/32/64, gemv 4..64)
    DEFINE_NVIDIA_GEMM(4, 4, 4)  DEFINE_NVIDIA_GEMM(6, 6, 6)
    DEFINE_NVIDIA_GEMM(8, 8, 8)  DEFINE_NVIDIA_GEMM(12, 12, 12)
    DEFINE_NVIDIA_GEMM(48, 48, 48)
    DEFINE_NVIDIA_GEMM(96, 96, 96)  DEFINE_NVIDIA_GEMM(128, 128, 128)
    DEFINE_NVIDIA_GEMV(32, 32)  DEFINE_NVIDIA_GEMV(48, 48)
    DEFINE_NVIDIA_GEMV(96, 96)  DEFINE_NVIDIA_GEMV(128, 128)
    // double — all bench sizes <=64 (none shipped)
    #define MEGA_GEMM_F64(N) DEFINE_NVIDIA_GEMM_PREC(N, N, N, double)
    #define MEGA_GEMV_F64(N) DEFINE_NVIDIA_GEMV_PREC(N, N, double)
    MEGA_GEMM_F64(4) MEGA_GEMM_F64(6) MEGA_GEMM_F64(8) MEGA_GEMM_F64(12)
    MEGA_GEMM_F64(16) MEGA_GEMM_F64(24) MEGA_GEMM_F64(32) MEGA_GEMM_F64(48) MEGA_GEMM_F64(64)
    MEGA_GEMV_F64(4) MEGA_GEMV_F64(6) MEGA_GEMV_F64(8) MEGA_GEMV_F64(12)
    MEGA_GEMV_F64(16) MEGA_GEMV_F64(24) MEGA_GEMV_F64(32) MEGA_GEMV_F64(48) MEGA_GEMV_F64(64)
}}
template<typename T,int N> __global__ void kn_dot (T* x, T* y) {
    extern __shared__ char s[]; int p=blockIdx.x;
    glass::nvidia::dot<T,N,NV_DOT_TB>(x+p*N, y+p*N, y+p*N, reinterpret_cast<T*>(s));
}
template<typename T,int N> __global__ void kn_gemv(T* A, T* x, T* y) {
    extern __shared__ char s[]; int p=blockIdx.x;
    glass::nvidia::gemv<T,N,N>((T)1, A+(size_t)p*N*N, x+p*N, (T)0, y+p*N, s);
}
template<typename T,int N> __global__ void kn_gemm(T* A, T* B, T* C) {
    extern __shared__ char s[]; int p=blockIdx.x;
    glass::nvidia::gemm<T,N,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N, s);
}
#endif
#if MEGA_NV_LAPACK
static const int NV_LP_TB = 256;    // cuSOLVERDx pinned block dim
namespace glass { namespace nvidia {
    #define MEGA_CHOL_DEF(N) DEFINE_NVIDIA_CHOL_BLOCKDIM(N, NV_LP_TB)
    #define MEGA_TRSM_DEF(N) DEFINE_NVIDIA_TRSM_BLOCKDIM(N, 1, NV_LP_TB)
    #define MEGA_POSV_DEF(N) DEFINE_NVIDIA_POSV_BLOCKDIM(N, 1, NV_LP_TB)
    #define MEGA_CHOL_F64(N) DEFINE_NVIDIA_CHOL_BLOCKDIM_PREC(N, NV_LP_TB, double)
    #define MEGA_TRSM_F64(N) DEFINE_NVIDIA_TRSM_BLOCKDIM_PREC(N, 1, NV_LP_TB, double)
    #define MEGA_POSV_F64(N) DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(N, 1, NV_LP_TB, double)
    MEGA_CHOL_DEF(4)  MEGA_CHOL_DEF(6)  MEGA_CHOL_DEF(8)  MEGA_CHOL_DEF(12)
    MEGA_CHOL_DEF(16) MEGA_CHOL_DEF(24) MEGA_CHOL_DEF(32) MEGA_CHOL_DEF(48) MEGA_CHOL_DEF(64)
    MEGA_CHOL_DEF(96) MEGA_CHOL_DEF(128)
    MEGA_TRSM_DEF(4)  MEGA_TRSM_DEF(6)  MEGA_TRSM_DEF(8)  MEGA_TRSM_DEF(12)
    MEGA_TRSM_DEF(16) MEGA_TRSM_DEF(24) MEGA_TRSM_DEF(32) MEGA_TRSM_DEF(48) MEGA_TRSM_DEF(64)
    MEGA_TRSM_DEF(96) MEGA_TRSM_DEF(128)
    MEGA_POSV_DEF(4)  MEGA_POSV_DEF(6)  MEGA_POSV_DEF(8)  MEGA_POSV_DEF(12)
    MEGA_POSV_DEF(16) MEGA_POSV_DEF(24) MEGA_POSV_DEF(32) MEGA_POSV_DEF(48) MEGA_POSV_DEF(64)
    MEGA_POSV_DEF(96) MEGA_POSV_DEF(128)
    // double — bench sizes <=64
    MEGA_CHOL_F64(4)  MEGA_CHOL_F64(6)  MEGA_CHOL_F64(8)  MEGA_CHOL_F64(12)
    MEGA_CHOL_F64(16) MEGA_CHOL_F64(24) MEGA_CHOL_F64(32) MEGA_CHOL_F64(48) MEGA_CHOL_F64(64)
    MEGA_TRSM_F64(4)  MEGA_TRSM_F64(6)  MEGA_TRSM_F64(8)  MEGA_TRSM_F64(12)
    MEGA_TRSM_F64(16) MEGA_TRSM_F64(24) MEGA_TRSM_F64(32) MEGA_TRSM_F64(48) MEGA_TRSM_F64(64)
    MEGA_POSV_F64(4)  MEGA_POSV_F64(6)  MEGA_POSV_F64(8)  MEGA_POSV_F64(12)
    MEGA_POSV_F64(16) MEGA_POSV_F64(24) MEGA_POSV_F64(32) MEGA_POSV_F64(48) MEGA_POSV_F64(64)
}}
template<typename T,int N> __global__ void kn_chol(T* A) {
    extern __shared__ char s[]; int p=blockIdx.x;
    glass::nvidia::chol_inplace<T,N,NV_LP_TB>(A+(size_t)p*N*N, s);
}
template<typename T,int N> __global__ void kn_trsv(T* A, T* x) {
    extern __shared__ char s[]; int p=blockIdx.x;
    glass::nvidia::trsm<T,N,1,NV_LP_TB>((T)1, A+(size_t)p*N*N, x+p*N, s);
}
template<typename T,int N> __global__ void kn_posv(T* A, T* b) {
    extern __shared__ char s[]; int p=blockIdx.x;
    glass::nvidia::posv<T,N,1,NV_LP_TB>(A+(size_t)p*N*N, b+p*N, s);
}
#endif

// True at compile time iff op@N has a forced nvidia variant defined above.
// Double is defined only up to 64 (f64 descriptors/smem cap lower than float).
template<typename T,int N> static constexpr bool nv_blas_ok()
{ return MEGA_NV_BLAS   && (std::is_same_v<T,float> ? N <= 128 : N <= 64); }
template<typename T,int N> static constexpr bool nv_lapack_ok()
{ return MEGA_NV_LAPACK && (std::is_same_v<T,float> ? N <= 128 : N <= 64); }

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

static size_t g_optin_smem = 48 * 1024;   // device opt-in dynamic-smem cap (queried in main)

#if MEGA_NV_BLAS || MEGA_NV_LAPACK
// Checked nvidia launch: skip (return -1) if the descriptor's smem exceeds the
// device opt-in cap, if the attribute opt-in fails, or if a verification launch
// errors out. Without this, a failed launch (e.g. gemm smem > cap) would time at
// ~1ns/problem and masquerade as an absurd "win" — corrupting the comparison.
template<typename Kern, typename Launch>
static double nv_timed(Kern kern, size_t smem, Launch launch, int reps) {
    if (smem > g_optin_smem) return -1.0;
    cudaGetLastError();                                   // clear any prior error
    if (smem > 48u * 1024u &&
        cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem) != cudaSuccess) { cudaGetLastError(); return -1.0; }
    launch();                                             // verification launch
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return -1.0; }
    if (cudaGetLastError() != cudaSuccess)       { cudaGetLastError(); return -1.0; }
    return time_ns_per_prob(launch, reps);
}
#endif

// nvidia leg: needs dynamic smem opt-in for the larger descriptors (>48KB).
// Returns best ns/problem for (op,N) at precision T, or -1 if no nvidia variant
// (or the launch can't fit / isn't defined for this T,N — see nv_*_ok<T,N>).
template<typename T,int N>
static double nv_op_time(Op op, T* A, T* B, T* C, T* x, T* y, int reps) {
    (void)A;(void)B;(void)C;(void)x;(void)y;(void)reps;(void)op;
    dim3 grid(NPROB);
#if MEGA_NV_BLAS
    if constexpr (nv_blas_ok<T,N>()) {
        if (op == DOT) {
            size_t smem = glass::nvidia::reduce_smem_size<T,NV_DOT_TB>();
            return nv_timed(kn_dot<T,N>, smem, [&]{ kn_dot<T,N><<<grid,NV_DOT_TB,smem>>>(x, y); }, reps);
        }
        if (op == GEMV) {
            size_t smem = glass::nvidia::gemv_smem_size<T,N,N>();
            int tb = (int)glass::nvidia::gemv_threads<T,N,N>();
            return nv_timed(kn_gemv<T,N>, smem, [&]{ kn_gemv<T,N><<<grid,tb,smem>>>(A, x, y); }, reps);
        }
        if (op == GEMM) {
            size_t smem = glass::nvidia::gemm_smem_size<T,N,N,N>();
            int tb = (int)glass::nvidia::gemm_threads<T,N,N,N>();
            return nv_timed(kn_gemm<T,N>, smem, [&]{ kn_gemm<T,N><<<grid,tb,smem>>>(A, B, C); }, reps);
        }
    }
#endif
#if MEGA_NV_LAPACK
    if constexpr (nv_lapack_ok<T,N>()) {
        if (op == CHOL) {
            size_t smem = glass::nvidia::chol_inplace_smem_size<T,N,NV_LP_TB>();
            return nv_timed(kn_chol<T,N>, smem, [&]{ kn_chol<T,N><<<grid,NV_LP_TB,smem>>>(A); }, reps);
        }
        if (op == TRSV) {
            size_t smem = glass::nvidia::trsm_smem_size<T,N,1,NV_LP_TB>();
            return nv_timed(kn_trsv<T,N>, smem, [&]{ kn_trsv<T,N><<<grid,NV_LP_TB,smem>>>(A, x); }, reps);
        }
        if (op == POSV) {
            size_t smem = glass::nvidia::posv_smem_size<T,N,1,NV_LP_TB>();
            return nv_timed(kn_posv<T,N>, smem, [&]{ kn_posv<T,N><<<grid,NV_LP_TB,smem>>>(A, x); }, reps);
        }
    }
#endif
    return -1.0;
}

// Dispatch the nvidia leg for the active precision (float full, double <=N64).
template<typename T,int N>
static double nv_dispatch(Op op, T* A, T* B, T* C, T* x, T* y, int reps) {
    return nv_op_time<T,N>(op, A, B, C, x, y, reps);
}

template<typename T,int N>
static void bench_size(Op op, int reps) {
    T *A, *B, *C, *x, *y;
    size_t mm = (size_t)NPROB * N * N, vv = (size_t)NPROB * N;
    cudaMalloc(&A, mm*sizeof(T)); cudaMalloc(&B, mm*sizeof(T)); cudaMalloc(&C, mm*sizeof(T));
    cudaMalloc(&x, vv*sizeof(T)); cudaMalloc(&y, vv*sizeof(T));
    // diagonally-dominant A (valid for chol/trsv/posv); broadcast one tile to all problems.
    T* hA = (T*)malloc((size_t)N*N*sizeof(T));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) hA[i+j*N] = (i==j)?(T)(N+2):(T)(0.1*((i+2*j)%5));
    cudaMemcpy(A, hA, (size_t)N*N*sizeof(T), cudaMemcpyHostToDevice);
    for (size_t p=1;p<(size_t)NPROB;p++) cudaMemcpy(A+p*N*N, A, (size_t)N*N*sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemset(B, 1, mm*sizeof(T)); cudaMemset(C, 0, mm*sizeof(T));
    cudaMemset(x, 1, vv*sizeof(T)); cudaMemset(y, 1, vv*sizeof(T));
    free(hA);

    double best_block=1e30, best_warp=1e30; int best_tb=0, best_wpb=0;
    printf("%-5s N=%-3d | BLOCK", op_name(op), N);
    for (int TB : {32, 64, 128, 256}) {
        double ns = time_ns_per_prob([&]{ launch_block<T,N>(op, TB, A, B, C, x, y); }, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best_block) { best_block = ns; best_tb = TB; }
    }
    printf("  | WARP");
    for (int WPB : {1, 2, 4, 8, 16, 32}) {
        if (WPB > NPROB) break;
        double ns = time_ns_per_prob([&]{ launch_warp<T,N>(op, WPB, A, B, C, x, y); }, reps);
        printf("  w%d=%.2f", WPB, ns);
        if (ns < best_warp) { best_warp = ns; best_wpb = WPB; }
    }
    double nv = nv_dispatch<T,N>(op, A, B, C, x, y, reps);

    // 3-way winner
    double base = best_warp < best_block ? best_warp : best_block;
    const char* base_winner = best_warp < best_block ? "WARP" : "BLOCK";
    const char* winner; double margin;
    if (nv > 0 && nv < base) { winner = "NVIDIA"; margin = base / nv; }
    else if (nv > 0)         { winner = base_winner; margin = nv / base; }   // margin = how much NV trails
    else                     { winner = base_winner; margin = best_warp < best_block ? best_block/best_warp : best_warp/best_block; }
    printf("  || block tb%d=%.2f  warp w%d=%.2f", best_tb, best_block, best_wpb, best_warp);
    if (nv > 0) printf("  nv=%.2f", nv);
    printf("  -> %s (%.2fx)\n", winner, margin);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(x); cudaFree(y);
}

template<typename T> static void run_all(int reps) {
    for (Op op : {DOT, GEMV, GEMM, CHOL, TRSV, POSV}) {
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
    { int v = 48*1024; cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0); g_optin_smem = (size_t)v; }
    printf("# mega sweep | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better) | optin_smem=%zuKB\n", NPROB, reps, f64 ? "f64" : "f32", g_optin_smem/1024);
    printf("# contenders: BLOCK(SIMT, TB swept) | WARP(WPB swept) | NV(cuBLASDx/cuSOLVERDx, forced; f32<=128, f64<=64)\n");
    if (f64) run_all<double>(reps);
    else     run_all<float>(reps);
    return 0;
}
