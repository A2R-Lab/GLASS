// bench_lapack.cu — LAPACK timings: pure-SIMT glass vs. cuSOLVERDx-backed glass::nvidia.
//
// Variants per size N (square SPD problem with NRHS=1):
//   pure-SIMT chol_InPlace        — glass::cholDecomp_InPlace<T, N>
//   pure-SIMT chol+trsm           — chol_InPlace then glass::trsm<T, N>
//   glass::nvidia cholDecomp_InPlace    — cuSOLVERDx potrf via glass::nvidia::cholDecomp_InPlace<T, N, TC>
//   glass::nvidia chol+trsm       — cholDecomp_InPlace then glass::nvidia::trsm<T, N, 1, TC>
//   glass::nvidia posv            — fused factor+solve via glass::nvidia::posv<T, N, 1, TC>
//
// Anti-optimization safeguards:
//   - Compile with -Xptxas -O1 (set in run_bench.py)
//   - Reload destructive inputs (A, B) from a master copy each iteration
//   - Write a per-iteration sink to defeat dead-store elimination
//   - --expt-relaxed-constexpr for cuBLASDx/cuSOLVERDx
//
// Compilation (with cuSOLVERDx):
//   nvcc -std=c++17 -arch=sm_XX -O3 --expt-relaxed-constexpr -Xptxas -O1
//        -I.. -I../src
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
//        -DGLASS_BENCH_CUBLASDX -DGLASS_BENCH_CUSOLVERDX -DSMS=XX0
//        -DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT
//        -lcusolverdx -lcublas -lcusolver
//        bench_lapack.cu -o bench_lapack
// Usage: ./bench_lapack [n [iters]]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif
#include "../glass-nvidia.cuh"

static const int THREADS = 256;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// ─── Pure-SIMT baselines (compile-time N) ────────────────────────────────────
// Reload A from master each iteration since chol overwrites it.

template<int N>
__global__ void k_simt_chol(const float* A_master, float* A, volatile float* sink, int iters) {
    extern __shared__ float smem[];
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (int rep = 0; rep < iters; rep++) {
        for (uint32_t i = rank; i < N*N; i += size) smem[i] = A_master[i];
        __syncthreads();
        glass::cholDecomp_InPlace<float, N>(smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = smem[0];
        __syncthreads();
    }
    if (rank < N*N) A[rank] = smem[rank];
}

template<int N>
__global__ void k_simt_chol_trsm(const float* A_master, const float* b_master,
                                  float* A, float* b, volatile float* sink, int iters) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_b = smem + N*N;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (int rep = 0; rep < iters; rep++) {
        for (uint32_t i = rank; i < N*N; i += size) s_A[i] = A_master[i];
        for (uint32_t i = rank; i < N;   i += size) s_b[i] = b_master[i];
        __syncthreads();
        glass::cholDecomp_InPlace<float, N>(s_A);
        __syncthreads();
        glass::trsm<float, N>(s_A, s_b);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = s_b[0];
        __syncthreads();
    }
    if (rank < N) b[rank] = s_b[rank];
    if (rank < N*N) A[rank] = s_A[rank];
}

// ─── glass::nvidia (cuSOLVERDx-backed) variants ──────────────────────────────
// Pre-instantiate sizes at TC=THREADS (256). NRHS=1 for solve operations.

namespace glass { namespace nvidia {
    DEFINE_NVIDIA_CHOL_BLOCKDIM(4,  THREADS)
    DEFINE_NVIDIA_CHOL_BLOCKDIM(6,  THREADS)
    DEFINE_NVIDIA_CHOL_BLOCKDIM(8,  THREADS)
    DEFINE_NVIDIA_CHOL_BLOCKDIM(12, THREADS)
    DEFINE_NVIDIA_CHOL_BLOCKDIM(14, THREADS)
    DEFINE_NVIDIA_CHOL_BLOCKDIM(24, THREADS)
    DEFINE_NVIDIA_CHOL_BLOCKDIM(64, THREADS)

    DEFINE_NVIDIA_TRSM_BLOCKDIM(4,  1, THREADS)
    DEFINE_NVIDIA_TRSM_BLOCKDIM(6,  1, THREADS)
    DEFINE_NVIDIA_TRSM_BLOCKDIM(8,  1, THREADS)
    DEFINE_NVIDIA_TRSM_BLOCKDIM(12, 1, THREADS)
    DEFINE_NVIDIA_TRSM_BLOCKDIM(14, 1, THREADS)
    DEFINE_NVIDIA_TRSM_BLOCKDIM(24, 1, THREADS)
    DEFINE_NVIDIA_TRSM_BLOCKDIM(64, 1, THREADS)

    DEFINE_NVIDIA_POSV_BLOCKDIM(4,  1, THREADS)
    DEFINE_NVIDIA_POSV_BLOCKDIM(6,  1, THREADS)
    DEFINE_NVIDIA_POSV_BLOCKDIM(8,  1, THREADS)
    DEFINE_NVIDIA_POSV_BLOCKDIM(12, 1, THREADS)
    DEFINE_NVIDIA_POSV_BLOCKDIM(14, 1, THREADS)
    DEFINE_NVIDIA_POSV_BLOCKDIM(24, 1, THREADS)
    DEFINE_NVIDIA_POSV_BLOCKDIM(64, 1, THREADS)
}}

template<int N>
__global__ void k_nv_chol(const float* A_master, float* A, volatile float* sink, int iters) {
    extern __shared__ char nv_smem[];
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (int rep = 0; rep < iters; rep++) {
        // glass::nvidia::cholDecomp_InPlace works on a global pointer; reload from master.
        if (rank < N*N) {
            for (uint32_t i = rank; i < N*N; i += size) A[i] = A_master[i];
        }
        __syncthreads();
        glass::nvidia::cholDecomp_InPlace<float, N, THREADS>(A, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = A[0];
        __syncthreads();
    }
}

template<int N>
__global__ void k_nv_chol_trsm(const float* A_master, const float* b_master,
                                float* A, float* b, volatile float* sink, int iters) {
    extern __shared__ char nv_smem[];
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (int rep = 0; rep < iters; rep++) {
        for (uint32_t i = rank; i < N*N; i += size) A[i] = A_master[i];
        for (uint32_t i = rank; i < N;   i += size) b[i] = b_master[i];
        __syncthreads();
        glass::nvidia::cholDecomp_InPlace<float, N, THREADS>(A, nv_smem);
        __syncthreads();
        glass::nvidia::trsm<float, N, 1, THREADS>(1.f, A, b, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = b[0];
        __syncthreads();
    }
}

template<int N>
__global__ void k_nv_posv(const float* A_master, const float* b_master,
                           float* A, float* b, volatile float* sink, int iters) {
    extern __shared__ char nv_smem[];
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (int rep = 0; rep < iters; rep++) {
        for (uint32_t i = rank; i < N*N; i += size) A[i] = A_master[i];
        for (uint32_t i = rank; i < N;   i += size) b[i] = b_master[i];
        __syncthreads();
        glass::nvidia::posv<float, N, 1, THREADS>(A, b, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = b[0];
        __syncthreads();
    }
}

// ─── Test-data generation: build SPD matrix on host ──────────────────────────
static void make_spd(int n, float* A) {
    // A = M·M^T + n·I (column-major)
    float* M = (float*)malloc(sizeof(float) * n * n);
    for (int i = 0; i < n*n; i++) M[i] = (float)((rand() % 1000) / 1000.0 - 0.5);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.f;
            for (int k = 0; k < n; k++) s += M[i + k*n] * M[j + k*n];
            A[i + j*n] = s + (i == j ? (float)n : 0.f);
        }
    }
    free(M);
}

// ─── Bench harness ───────────────────────────────────────────────────────────
template<int N>
static void bench_size_ct(int iters) {
    float *dA_master, *db_master, *dA, *db, *dSink;
    cudaMalloc(&dA_master, N*N*sizeof(float));
    cudaMalloc(&db_master, N*sizeof(float));
    cudaMalloc(&dA,        N*N*sizeof(float));
    cudaMalloc(&db,        N*sizeof(float));
    cudaMalloc(&dSink,     256*sizeof(float));

    float* hA = (float*)malloc(N*N*sizeof(float));
    float* hb = (float*)malloc(N*sizeof(float));
    make_spd(N, hA);
    for (int i = 0; i < N; i++) hb[i] = (float)((rand() % 100) / 10.0);
    cudaMemcpy(dA_master, hA, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db_master, hb, N*sizeof(float),   cudaMemcpyHostToDevice);
    free(hA); free(hb);

    struct timespec t0, t1;

    // pure-SIMT chol
    int simt_chol_smem = N*N*sizeof(float);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_simt_chol<N><<<1, THREADS, simt_chol_smem>>>(dA_master, dA, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::cholDecomp_InPlace    n=%2d  %.3f us/op\n",
           N, elapsed_us(t0, t1) / iters);

    // pure-SIMT chol+trsm
    int simt_csb_smem = (N*N + N) * sizeof(float);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_simt_chol_trsm<N><<<1, THREADS, simt_csb_smem>>>(dA_master, db_master, dA, db, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::chol+trsm             n=%2d  %.3f us/op\n",
           N, elapsed_us(t0, t1) / iters);

    // glass::nvidia chol
    constexpr size_t nv_chol_smem = glass::nvidia::cholDecomp_InPlace_smem_size<float, N, THREADS>();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_nv_chol<N><<<1, THREADS, nv_chol_smem>>>(dA_master, dA, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia::cholDecomp_InPlace  n=%2d  %.3f us/op\n",
           N, elapsed_us(t0, t1) / iters);

    // glass::nvidia chol+trsm
    constexpr size_t nv_trsm_smem = glass::nvidia::trsm_smem_size<float, N, 1, THREADS>();
    constexpr size_t nv_chol_trsm_smem = nv_chol_smem > nv_trsm_smem ? nv_chol_smem : nv_trsm_smem;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_nv_chol_trsm<N><<<1, THREADS, nv_chol_trsm_smem>>>(dA_master, db_master, dA, db, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia::chol+trsm     n=%2d  %.3f us/op\n",
           N, elapsed_us(t0, t1) / iters);

    // glass::nvidia posv (fused)
    constexpr size_t nv_posv_smem = glass::nvidia::posv_smem_size<float, N, 1, THREADS>();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_nv_posv<N><<<1, THREADS, nv_posv_smem>>>(dA_master, db_master, dA, db, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia::posv (fused)  n=%2d  %.3f us/op\n",
           N, elapsed_us(t0, t1) / iters);

    cudaFree(dA_master); cudaFree(db_master);
    cudaFree(dA); cudaFree(db); cudaFree(dSink);
}

int main(int argc, char** argv) {
    int n     = (argc > 1) ? atoi(argv[1]) : 0;
    int iters = (argc > 2) ? atoi(argv[2]) : 1000;

    srand(42);

    if (n == 4)       bench_size_ct<4>(iters);
    else if (n == 6)  bench_size_ct<6>(iters);
    else if (n == 8)  bench_size_ct<8>(iters);
    else if (n == 12) bench_size_ct<12>(iters);
    else if (n == 14) bench_size_ct<14>(iters);
    else if (n == 24) bench_size_ct<24>(iters);
    else if (n == 64) bench_size_ct<64>(iters);
    else {
        bench_size_ct<4>(iters);
        bench_size_ct<6>(iters);
        bench_size_ct<8>(iters);
        bench_size_ct<12>(iters);
        bench_size_ct<14>(iters);
        bench_size_ct<24>(iters);
        bench_size_ct<64>(iters);
    }
    return 0;
}
