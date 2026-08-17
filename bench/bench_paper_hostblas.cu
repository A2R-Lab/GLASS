// bench_paper_hostblas.cu — GLASS one-block/one-warp-per-problem vs HOST-side
// batched cuBLAS/cuSOLVER, for the paper's F2 (throughput vs batch) and F4
// (batch=1 latency) figures.
//
// Ops (col-major, square N):
//   gemm   C = A·B                     glass::gemm<T,N,N,N>          vs cublas<X>gemmStridedBatched
//   potrf  A = L·Lᵀ (lower, in place)  glass::potrf<T,N>             vs cusolverDn<X>potrfBatched
//   posv   A x = b (nrhs=1, in place)  glass::posv<T,N>              vs potrfBatched + potrsBatched
//
// Contenders per (op, N, B): glass block (TB ∈ {32,128}), glass warp
// (WPB=8, gemm/potrf only — no warp posv in the library), vendor host-batched,
// and vendor_tf32 (gemm f32 only): the SAME strided-batched call on a handle
// with CUBLAS_TF32_TENSOR_OP_MATH — i.e. the vendor's best with tensor cores
// ALLOWED at relaxed numerics (cuBLAS heuristics still pick the kernel; there
// is no TF32 Cholesky in cuSOLVER, which is itself a paper point). Its
// correctness CHECK line is REPORT-ONLY: the maxerr vs the double reference
// IS the measured TF32 rounding cost (sanity-bounded at 0.1, never eps-tight).
//
// THROUGHPUT PROTOCOL (mirrors bench_solvers.cu): B independent problems in
// global memory; each rep = ONE launch / ONE host API chain spanning all B,
// bracketed by cudaEvents; mutated buffers RESTORED from pristine device
// copies OUTSIDE the event window. ns/problem = ms_sum*1e6/(reps*B), min of
// 3 trials, with trial spread reported. GPU-event timing excludes host API overhead — conservative
// TOWARD the vendor (their per-call CPU cost is amortized 1/B anyway).
//
// LATENCY PROTOCOL (batch=1, the MPC regime): R=200 pre-initialized problem
// copies; wall-clock a loop of {one call on its own pristine data +
// cudaDeviceSynchronize}, µs/call = wall/R. The vendor contender here is the
// NON-batched call a user would actually write (cublas<X>gemm,
// cusolverDn<X>potrf(+potrs) with a preallocated workspace).
//
// CORRECTNESS GUARD (no silent caps): before any timing, every (op, N, impl)
// runs on problem 0 and compares against a host double reference
// (max|Δ| < tol, lower-triangle-only for potrf). Mismatch aborts.
//
// Compile: nvcc -std=c++17 -arch=sm_XX -O3 --expt-relaxed-constexpr
//          -I.. -I../src bench_paper_hostblas.cu -o bench_paper_hostblas
//          -lcublas -lcusolver          (no MathDx needed)
// Usage:   ./bench_paper_hostblas [reps=50] [dtype=f32|f64|both] [section=all|thru|lat]
//
// Output grammar (parsed by paper_sweeps.py):
//   CHECK  op=<op> dtype=<dt> N=<n> impl=<impl> maxerr=<e>
//   RESULT section=thru ... ns=<ns_per_problem> spread=<worst/best-1 percent>
//   RESULT section=lat  ... us=<us_per_call> spread=<worst/best-1 percent>
//   SKIP   ... reason=...

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <functional>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "timing_common.cuh"
#include "../glass.cuh"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(3); } } while (0)
#define CB(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(3); } } while (0)
#define CS(call) do { cusolverStatus_t s_ = (call); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(3); } } while (0)

static const int    BGRID[]   = {1, 4, 16, 64, 256, 1024, 8192};
static const int    NBGRID    = 7;
static const size_t MEM_CAP   = 4ull << 30;   // skip cells whose buffers exceed 4 GB (Jetson-safe)
static const int    LAT_CALLS = 200;
static int REPS = 50;
static double last_spread_pct = 0.0;

// ─── deterministic host RNG (xorshift64* + Box-Muller), as bench_solvers.cu ──
static uint64_t rng_state = 1;
static void rng_seed(uint64_t s) { rng_state = s ? s : 0x9E3779B97F4A7C15ull; }
static double urand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
}
static double nrand() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ─── host double references ──────────────────────────────────────────────────
static void ref_gemm(int n, const double* A, const double* B, double* C) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int k = 0; k < n; k++) s += A[i + k*n] * B[k + j*n];
            C[i + j*n] = s;
        }
}
static bool ref_chol(int n, double* A) {          // lower, in place
    for (int j = 0; j < n; j++) {
        double d = A[j + j*n];
        for (int k = 0; k < j; k++) d -= A[j + k*n] * A[j + k*n];
        if (d <= 0.0) return false;
        d = sqrt(d); A[j + j*n] = d;
        for (int i = j + 1; i < n; i++) {
            double s = A[i + j*n];
            for (int k = 0; k < j; k++) s -= A[i + k*n] * A[j + k*n];
            A[i + j*n] = s / d;
        }
    }
    return true;
}
static bool ref_chol_solve(int n, double* A, double* b) {   // A destroyed, b→x
    if (!ref_chol(n, A)) return false;
    for (int i = 0; i < n; i++) {                     // L y = b
        double s = b[i];
        for (int k = 0; k < i; k++) s -= A[i + k*n] * b[k];
        b[i] = s / A[i + i*n];
    }
    for (int i = n - 1; i >= 0; i--) {                // Lᵀ x = y
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= A[k + i*n] * b[k];
        b[i] = s / A[i + i*n];
    }
    return true;
}

// ─── optional MAGMA columns (research host-batched library; BENCH-ONLY dep) ──
// Enabled with -DGLASS_BENCH_MAGMA + MAGMA include/lib flags. The MAGMA queue
// is created on the DEFAULT stream so the same event bracketing times it.
// gemm gets two rows: "magma" = gemm_batched_strided (exact mirror of the
// cuBLAS strided call) and "magma_pa" = pointer-array gemm_batched (their
// canonical entry, which dispatches to their tuned small-square kernels);
// potrf/posv use the pointer-array batched routines (posv is MAGMA's fused
// one-call form — favorable to MAGMA vs the vendor's potrf+potrs pair;
// disclosed). Disclosure for the paper: MAGMA is a research library compared
// here IN ADDITION to our documented default-vendor scope.
#if defined(GLASS_BENCH_MAGMA)
#include <magma_v2.h>
static void mgemm_sb(magma_queue_t q, int n, float alpha, const float* A, long long sA,
                     const float* B, long long sB, float beta, float* C, long long sC, int batch) {
    magmablas_sgemm_batched_strided(MagmaNoTrans, MagmaNoTrans, n, n, n, alpha,
                                    A, n, sA, B, n, sB, beta, C, n, sC, batch, q);
}
static void mgemm_sb(magma_queue_t q, int n, double alpha, const double* A, long long sA,
                     const double* B, long long sB, double beta, double* C, long long sC, int batch) {
    magmablas_dgemm_batched_strided(MagmaNoTrans, MagmaNoTrans, n, n, n, alpha,
                                    A, n, sA, B, n, sB, beta, C, n, sC, batch, q);
}
static void mgemm_pa(magma_queue_t q, int n, float alpha, float** Ap, float** Bp,
                     float beta, float** Cp, int batch) {
    magmablas_sgemm_batched(MagmaNoTrans, MagmaNoTrans, n, n, n, alpha,
                            Ap, n, Bp, n, beta, Cp, n, batch, q);
}
static void mgemm_pa(magma_queue_t q, int n, double alpha, double** Ap, double** Bp,
                     double beta, double** Cp, int batch) {
    magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, n, n, n, alpha,
                            Ap, n, Bp, n, beta, Cp, n, batch, q);
}
static void mpotrf_batched(magma_queue_t q, int n, float** Ap, int* info, int batch) {
    magma_spotrf_batched(MagmaLower, n, Ap, n, info, batch, q);
}
static void mpotrf_batched(magma_queue_t q, int n, double** Ap, int* info, int batch) {
    magma_dpotrf_batched(MagmaLower, n, Ap, n, info, batch, q);
}
static void mposv_batched(magma_queue_t q, int n, float** Ap, float** bp, int* info, int batch) {
    magma_sposv_batched(MagmaLower, n, 1, Ap, n, bp, n, info, batch, q);
}
static void mposv_batched(magma_queue_t q, int n, double** Ap, double** bp, int* info, int batch) {
    magma_dposv_batched(MagmaLower, n, 1, Ap, n, bp, n, info, batch, q);
}
#endif

// ─── vendor call wrappers, overloaded on dtype ───────────────────────────────
static void xgemm_sb(cublasHandle_t h, int n, float alpha, const float* A, long long sA,
                     const float* B, long long sB, float beta, float* C, long long sC, int batch) {
    CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                 A, n, sA, B, n, sB, &beta, C, n, sC, batch));
}
static void xgemm_sb(cublasHandle_t h, int n, double alpha, const double* A, long long sA,
                     const double* B, long long sB, double beta, double* C, long long sC, int batch) {
    CB(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                 A, n, sA, B, n, sB, &beta, C, n, sC, batch));
}
static void xgemm(cublasHandle_t h, int n, float alpha, const float* A, const float* B,
                  float beta, float* C) {
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n));
}
static void xgemm(cublasHandle_t h, int n, double alpha, const double* A, const double* B,
                  double beta, double* C) {
    CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n));
}
static void xpotrf_batched(cusolverDnHandle_t h, int n, float** Aarr, int* info, int batch) {
    CS(cusolverDnSpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, Aarr, n, info, batch));
}
static void xpotrf_batched(cusolverDnHandle_t h, int n, double** Aarr, int* info, int batch) {
    CS(cusolverDnDpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, Aarr, n, info, batch));
}
static void xpotrs_batched(cusolverDnHandle_t h, int n, float** Aarr, float** barr, int* info, int batch) {
    CS(cusolverDnSpotrsBatched(h, CUBLAS_FILL_MODE_LOWER, n, 1, Aarr, n, barr, n, info, batch));
}
static void xpotrs_batched(cusolverDnHandle_t h, int n, double** Aarr, double** barr, int* info, int batch) {
    CS(cusolverDnDpotrsBatched(h, CUBLAS_FILL_MODE_LOWER, n, 1, Aarr, n, barr, n, info, batch));
}
static int xpotrf_bufsize(cusolverDnHandle_t h, int n, float* A) {
    int lw = 0; CS(cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, n, &lw)); return lw;
}
static int xpotrf_bufsize(cusolverDnHandle_t h, int n, double* A) {
    int lw = 0; CS(cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, n, &lw)); return lw;
}
static void xpotrf(cusolverDnHandle_t h, int n, float* A, float* work, int lw, int* info) {
    CS(cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, n, work, lw, info));
}
static void xpotrf(cusolverDnHandle_t h, int n, double* A, double* work, int lw, int* info) {
    CS(cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, n, work, lw, info));
}
static void xpotrs(cusolverDnHandle_t h, int n, const float* A, float* b, int* info) {
    CS(cusolverDnSpotrs(h, CUBLAS_FILL_MODE_LOWER, n, 1, A, n, b, n, info));
}
static void xpotrs(cusolverDnHandle_t h, int n, const double* A, double* b, int* info) {
    CS(cusolverDnDpotrs(h, CUBLAS_FILL_MODE_LOWER, n, 1, A, n, b, n, info));
}

// ─── GLASS kernels: one block / one warp per problem, operands in global ─────
template <typename T, uint32_t N>
__global__ void k_block_gemm(const T* A, const T* B, T* C, int nprob) {
    size_t p = blockIdx.x;
    glass::block::gemm<T, N, N, N>((T)1, A + p*N*N, B + p*N*N, (T)0, C + p*N*N);
}
template <typename T, uint32_t N>
__global__ void k_block_potrf(T* A, int nprob) {
    glass::block::potrf<T, N>(A + (size_t)blockIdx.x * N*N);
}
template <typename T, uint32_t N>
__global__ void k_block_posv(T* A, T* b, int nprob) {
    size_t p = blockIdx.x;
    glass::block::posv<T, N>(A + p*N*N, b + p*N);
}
template <typename T, uint32_t N>
__global__ void k_warp_gemm(const T* A, const T* B, T* C, int nprob) {
    size_t w = blockIdx.x * blockDim.y + threadIdx.y;
    if (w >= (size_t)nprob) return;
    glass::warp::gemm<T, N, N, N>((T)1, A + w*N*N, B + w*N*N, (T)0, C + w*N*N);
}
template <typename T, uint32_t N>
__global__ void k_warp_potrf(T* A, int nprob) {
    size_t w = blockIdx.x * blockDim.y + threadIdx.y;
    if (w >= (size_t)nprob) return;
    glass::warp::potrf<T, N>(A + w*N*N);
}
// glass thread tier: one problem per thread, operands staged global ->
// registers -> global in the ladder's per-problem-contiguous layout (the
// layout tax is IN the timing, mirroring bench_mega_sweep's THREAD model).
// Included because the shipped ladder picks route chol/posv f64 (and posv
// f32) small-N cells to THIS tier — without it, host-batched factorization
// baselines (vendor, MAGMA) would be compared against the wrong GLASS tier
// exactly where they are strongest in f64.
template <typename T, uint32_t N>
__global__ void k_thread_gemm(const T* A, const T* B, T* C, int nprob) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= nprob) return;
    T a[N*N], b[N*N], c[N*N];
    for (uint32_t i = 0; i < N*N; i++) { a[i] = A[(size_t)p*N*N+i]; b[i] = B[(size_t)p*N*N+i]; }
    glass::thread::gemm<T,N,N,N>((T)1, a, b, c);
    for (uint32_t i = 0; i < N*N; i++) C[(size_t)p*N*N+i] = c[i];
}
template <typename T, uint32_t N>
__global__ void k_thread_potrf(T* A, int nprob) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= nprob) return;
    T a[N*N];
    for (uint32_t i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N+i];
    glass::thread::potrf<T,N>(a);
    for (uint32_t i = 0; i < N*N; i++) A[(size_t)p*N*N+i] = a[i];
}
template <typename T, uint32_t N>
__global__ void k_thread_posv(T* A, T* b, int nprob) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= nprob) return;
    T a[N*N], bv[N];
    for (uint32_t i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N+i];
    for (uint32_t i = 0; i < N; i++)   bv[i] = b[(size_t)p*N+i];
    glass::thread::posv<T,N>(a, bv);
    for (uint32_t i = 0; i < N; i++)   b[(size_t)p*N+i] = bv[i];
}

// ─── timing helpers ──────────────────────────────────────────────────────────
static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6 + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// One throughput measurement: min over 3 trials of (sum over REPS of one
// event-bracketed `work`, with `restore` run before it OUTSIDE the events).
static double time_thru_ns_per_prob(int B, const std::function<void()>& restore,
                                    const std::function<void()>& work) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    restore(); work(); CK(cudaDeviceSynchronize());   // warm-up (JIT, clocks)
    double best_ms = 1e300, worst_ms = 0.0;
    for (int trial = 0; trial < 3; trial++) {
        double ms_sum = 0;
        for (int rep = 0; rep < REPS; rep++) {
            restore();
            CK(cudaEventRecord(s));
            work();
            CK(cudaEventRecord(e));
            CK(cudaEventSynchronize(e));
            float ms; CK(cudaEventElapsedTime(&ms, s, e));
            ms_sum += ms;
        }
        if (ms_sum < best_ms) best_ms = ms_sum;
        if (ms_sum > worst_ms) worst_ms = ms_sum;
    }
    last_spread_pct = (worst_ms / best_ms - 1.0) * 100.0;
    CK(cudaEventDestroy(s)); CK(cudaEventDestroy(e));
    return best_ms * 1e6 / ((double)REPS * B);
}

// ─── per-(dtype, N) driver ───────────────────────────────────────────────────
struct Handles { cublasHandle_t cb; cublasHandle_t cb_tf32; cusolverDnHandle_t cs;
#if defined(GLASS_BENCH_MAGMA)
                 magma_queue_t mq;
#endif
};

template <typename T>
static double upload_and_maxerr(const T* d_out, const double* ref, int cnt, bool lower_only, int n) {
    std::vector<T> h(cnt);
    CK(cudaMemcpy(h.data(), d_out, cnt * sizeof(T), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (int idx = 0; idx < cnt; idx++) {
        if (lower_only) { int i = idx % n, j = idx / n; if (i < j) continue; }
        double d = fabs((double)h[idx] - ref[idx]);
        if (d > maxerr) maxerr = d;
    }
    return maxerr;
}

static void check_or_die(const char* op, const char* dt, int n, const char* impl,
                         double maxerr, double tol) {
    printf("CHECK op=%s dtype=%s N=%d impl=%s maxerr=%.3e\n", op, dt, n, impl, maxerr);
    if (!(maxerr < tol)) {
        fprintf(stderr, "FATAL: %s/%s N=%d impl=%s maxerr %.3e >= tol %.3e\n",
                op, dt, n, impl, maxerr, tol);
        exit(4);
    }
}

template <typename T, uint32_t N>
static void run_for_N(Handles H, const char* dt, bool do_thru, bool do_lat) {
    const double tol = (sizeof(T) == 4 ? 2e-3 : 1e-9) * N;
    const size_t MM = (size_t)N * N;
    rng_seed(0xC0FFEE ^ (N * 2654435761u));

    // Problem-0 host data (double masters), shared by every correctness check.
    std::vector<double> hA(MM), hB(MM), hSPD(MM), hb(N), hM(MM);
    for (size_t i = 0; i < MM; i++) hA[i] = nrand() / sqrt((double)N);
    for (size_t i = 0; i < MM; i++) hB[i] = nrand() / sqrt((double)N);
    for (size_t i = 0; i < MM; i++) hM[i] = nrand();
    for (uint32_t j = 0; j < N; j++)
        for (uint32_t i = 0; i < N; i++) {
            double s = 0;
            for (uint32_t k = 0; k < N; k++) s += hM[i + k*N] * hM[j + k*N];
            hSPD[i + j*N] = s + (i == j ? (double)N : 0.0);
        }
    for (uint32_t i = 0; i < N; i++) hb[i] = nrand();

    std::vector<double> refC(MM), refL(hSPD), refX(hb);
    ref_gemm(N, hA.data(), hB.data(), refC.data());
    if (!ref_chol(N, refL.data())) { fprintf(stderr, "ref_chol failed N=%u\n", N); exit(4); }
    { std::vector<double> Acopy(hSPD); if (!ref_chol_solve(N, Acopy.data(), refX.data())) exit(4); }

    // ── correctness pass (B=4; problem 0 carries the reference data) ────────
    {
        const int B = 4;
        std::vector<T> stage(MM * B);
        T *dA, *dB, *dC, *dS, *dS0, *db, *db0;
        CK(cudaMalloc(&dA, MM*B*sizeof(T))); CK(cudaMalloc(&dB, MM*B*sizeof(T)));
        CK(cudaMalloc(&dC, MM*B*sizeof(T))); CK(cudaMalloc(&dS, MM*B*sizeof(T)));
        CK(cudaMalloc(&dS0, MM*B*sizeof(T)));
        CK(cudaMalloc(&db, N*B*sizeof(T))); CK(cudaMalloc(&db0, N*B*sizeof(T)));
        for (int p = 0; p < B; p++)
            for (size_t i = 0; i < MM; i++) stage[p*MM + i] = (T)hA[i];
        CK(cudaMemcpy(dA, stage.data(), MM*B*sizeof(T), cudaMemcpyHostToDevice));
        for (int p = 0; p < B; p++)
            for (size_t i = 0; i < MM; i++) stage[p*MM + i] = (T)hB[i];
        CK(cudaMemcpy(dB, stage.data(), MM*B*sizeof(T), cudaMemcpyHostToDevice));
        for (int p = 0; p < B; p++)
            for (size_t i = 0; i < MM; i++) stage[p*MM + i] = (T)hSPD[i];
        CK(cudaMemcpy(dS0, stage.data(), MM*B*sizeof(T), cudaMemcpyHostToDevice));
        { std::vector<T> sb(N*B);
          for (int p = 0; p < B; p++) for (uint32_t i = 0; i < N; i++) sb[p*N + i] = (T)hb[i];
          CK(cudaMemcpy(db0, sb.data(), N*B*sizeof(T), cudaMemcpyHostToDevice)); }

        std::vector<T*> hSp(B), hbp(B);
        for (int p = 0; p < B; p++) { hSp[p] = dS + p*MM; hbp[p] = db + p*N; }
        T **dSp, **dbp; int* dinfo;
        CK(cudaMalloc(&dSp, B*sizeof(T*))); CK(cudaMalloc(&dbp, B*sizeof(T*)));
        CK(cudaMalloc(&dinfo, (B+1)*sizeof(int)));
        CK(cudaMemcpy(dSp, hSp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbp, hbp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));

        auto restoreS  = [&]{ CK(cudaMemcpy(dS, dS0, MM*B*sizeof(T), cudaMemcpyDeviceToDevice)); };
        auto restoreSb = [&]{ restoreS();
                              CK(cudaMemcpy(db, db0, N*B*sizeof(T), cudaMemcpyDeviceToDevice)); };

        // gemm
        k_block_gemm<T, N><<<B, 128>>>(dA, dB, dC, B); CK(cudaDeviceSynchronize());
        check_or_die("gemm", dt, N, "block", upload_and_maxerr(dC, refC.data(), MM, false, N), tol);
        CK(cudaMemset(dC, 0, MM*B*sizeof(T)));
        k_warp_gemm<T, N><<<1, dim3(32, B)>>>(dA, dB, dC, B); CK(cudaDeviceSynchronize());
        check_or_die("gemm", dt, N, "warp", upload_and_maxerr(dC, refC.data(), MM, false, N), tol);
        CK(cudaMemset(dC, 0, MM*B*sizeof(T)));
        k_thread_gemm<T, N><<<1, B>>>(dA, dB, dC, B); CK(cudaDeviceSynchronize());
        check_or_die("gemm", dt, N, "thread", upload_and_maxerr(dC, refC.data(), MM, false, N), tol);
        CK(cudaMemset(dC, 0, MM*B*sizeof(T)));
        xgemm_sb(H.cb, N, (T)1, dA, MM, dB, MM, (T)0, dC, MM, B); CK(cudaDeviceSynchronize());
        check_or_die("gemm", dt, N, "vendor", upload_and_maxerr(dC, refC.data(), MM, false, N), tol);
        if (sizeof(T) == 4) {   // TF32 rounding cost — recorded, sanity-bounded only
            CK(cudaMemset(dC, 0, MM*B*sizeof(T)));
            xgemm_sb(H.cb_tf32, N, (T)1, dA, MM, dB, MM, (T)0, dC, MM, B); CK(cudaDeviceSynchronize());
            check_or_die("gemm", dt, N, "vendor_tf32", upload_and_maxerr(dC, refC.data(), MM, false, N), 0.1);
        }

        // potrf
        restoreS(); k_block_potrf<T, N><<<B, 128>>>(dS, B); CK(cudaDeviceSynchronize());
        check_or_die("potrf", dt, N, "block", upload_and_maxerr(dS, refL.data(), MM, true, N), tol);
        restoreS(); k_warp_potrf<T, N><<<1, dim3(32, B)>>>(dS, B); CK(cudaDeviceSynchronize());
        check_or_die("potrf", dt, N, "warp", upload_and_maxerr(dS, refL.data(), MM, true, N), tol);
        restoreS(); k_thread_potrf<T, N><<<1, B>>>(dS, B); CK(cudaDeviceSynchronize());
        check_or_die("potrf", dt, N, "thread", upload_and_maxerr(dS, refL.data(), MM, true, N), tol);
        restoreS(); xpotrf_batched(H.cs, N, dSp, dinfo, B); CK(cudaDeviceSynchronize());
        check_or_die("potrf", dt, N, "vendor", upload_and_maxerr(dS, refL.data(), MM, true, N), tol);

        // posv (nrhs=1)
        restoreSb(); k_block_posv<T, N><<<B, 128>>>(dS, db, B); CK(cudaDeviceSynchronize());
        check_or_die("posv", dt, N, "block", upload_and_maxerr(db, refX.data(), N, false, N), tol);
        restoreSb(); k_thread_posv<T, N><<<1, B>>>(dS, db, B); CK(cudaDeviceSynchronize());
        check_or_die("posv", dt, N, "thread", upload_and_maxerr(db, refX.data(), N, false, N), tol);
        restoreSb(); xpotrf_batched(H.cs, N, dSp, dinfo, B);
        xpotrs_batched(H.cs, N, dSp, dbp, dinfo + B, B); CK(cudaDeviceSynchronize());
        check_or_die("posv", dt, N, "vendor", upload_and_maxerr(db, refX.data(), N, false, N), tol);

#if defined(GLASS_BENCH_MAGMA)
        {   // MAGMA correctness (same refs, same tolerance)
            std::vector<T*> hp(B);
            T **dAp, **dBp, **dCp;
            CK(cudaMalloc(&dAp, B*sizeof(T*))); CK(cudaMalloc(&dBp, B*sizeof(T*)));
            CK(cudaMalloc(&dCp, B*sizeof(T*)));
            for (int p = 0; p < B; p++) hp[p] = dA + p*MM;
            CK(cudaMemcpy(dAp, hp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
            for (int p = 0; p < B; p++) hp[p] = dB + p*MM;
            CK(cudaMemcpy(dBp, hp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
            for (int p = 0; p < B; p++) hp[p] = dC + p*MM;
            CK(cudaMemcpy(dCp, hp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));

            CK(cudaMemset(dC, 0, MM*B*sizeof(T)));
            mgemm_sb(H.mq, N, (T)1, dA, MM, dB, MM, (T)0, dC, MM, B); CK(cudaDeviceSynchronize());
            check_or_die("gemm", dt, N, "magma", upload_and_maxerr(dC, refC.data(), MM, false, N), tol);
            CK(cudaMemset(dC, 0, MM*B*sizeof(T)));
            mgemm_pa(H.mq, N, (T)1, dAp, dBp, (T)0, dCp, B); CK(cudaDeviceSynchronize());
            check_or_die("gemm", dt, N, "magma_pa", upload_and_maxerr(dC, refC.data(), MM, false, N), tol);
            restoreS(); mpotrf_batched(H.mq, N, dSp, dinfo, B); CK(cudaDeviceSynchronize());
            check_or_die("potrf", dt, N, "magma", upload_and_maxerr(dS, refL.data(), MM, true, N), tol);
            restoreSb(); mposv_batched(H.mq, N, dSp, dbp, dinfo, B); CK(cudaDeviceSynchronize());
            check_or_die("posv", dt, N, "magma", upload_and_maxerr(db, refX.data(), N, false, N), tol);
            CK(cudaFree(dAp)); CK(cudaFree(dBp)); CK(cudaFree(dCp));
        }
#endif

        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dS));
        CK(cudaFree(dS0)); CK(cudaFree(db)); CK(cudaFree(db0));
        CK(cudaFree(dSp)); CK(cudaFree(dbp)); CK(cudaFree(dinfo));
    }

    // ── throughput sweep over the batch grid ────────────────────────────────
    if (do_thru) for (int bi = 0; bi < NBGRID; bi++) {
        const int B = BGRID[bi];
        const size_t bytes = 5 * MM * B * sizeof(T);   // A,B,C,S,S0 dominate
        if (bytes > MEM_CAP) {
            printf("SKIP op=all dtype=%s N=%u B=%d reason=mem_cap_%zuMB\n",
                   dt, N, B, bytes >> 20);
            continue;
        }
        T *dA, *dB, *dC, *dS, *dS0, *db, *db0;
        CK(cudaMalloc(&dA, MM*B*sizeof(T))); CK(cudaMalloc(&dB, MM*B*sizeof(T)));
        CK(cudaMalloc(&dC, MM*B*sizeof(T))); CK(cudaMalloc(&dS, MM*B*sizeof(T)));
        CK(cudaMalloc(&dS0, MM*B*sizeof(T)));
        CK(cudaMalloc(&db, (size_t)N*B*sizeof(T))); CK(cudaMalloc(&db0, (size_t)N*B*sizeof(T)));
        {   // per-problem randomized fills (host once, upload)
            std::vector<T> st(MM * (size_t)B);
            for (size_t i = 0; i < st.size(); i++) st[i] = (T)(nrand() / sqrt((double)N));
            CK(cudaMemcpy(dA, st.data(), st.size()*sizeof(T), cudaMemcpyHostToDevice));
            for (size_t i = 0; i < st.size(); i++) st[i] = (T)(nrand() / sqrt((double)N));
            CK(cudaMemcpy(dB, st.data(), st.size()*sizeof(T), cudaMemcpyHostToDevice));
            // SPD problems: reuse problem-0's SPD for all (values don't affect
            // chol's fixed flop count; restore keeps every rep pristine)
            for (int p = 0; p < B; p++)
                for (size_t i = 0; i < MM; i++) st[p*MM + i] = (T)hSPD[i];
            CK(cudaMemcpy(dS0, st.data(), st.size()*sizeof(T), cudaMemcpyHostToDevice));
            std::vector<T> sb((size_t)N*B);
            for (size_t i = 0; i < sb.size(); i++) sb[i] = (T)nrand();
            CK(cudaMemcpy(db0, sb.data(), sb.size()*sizeof(T), cudaMemcpyHostToDevice));
        }
        std::vector<T*> hSp(B), hbp(B);
        for (int p = 0; p < B; p++) { hSp[p] = dS + (size_t)p*MM; hbp[p] = db + (size_t)p*N; }
        T **dSp, **dbp; int* dinfo;
        CK(cudaMalloc(&dSp, B*sizeof(T*))); CK(cudaMalloc(&dbp, B*sizeof(T*)));
        CK(cudaMalloc(&dinfo, (B+1)*sizeof(int)));
        CK(cudaMemcpy(dSp, hSp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbp, hbp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));

        auto nop       = []{};
        auto restoreS  = [&]{ CK(cudaMemcpyAsync(dS, dS0, MM*(size_t)B*sizeof(T), cudaMemcpyDeviceToDevice)); };
        auto restoreSb = [&]{ restoreS();
                              CK(cudaMemcpyAsync(db, db0, (size_t)N*B*sizeof(T), cudaMemcpyDeviceToDevice)); };
        const int wgrid = (B + 7) / 8;     // WPB=8
        const int tgrid = (B + 127) / 128; // thread tier, TPB=128

        struct Row { const char* op; const char* impl; std::function<void()> restore, work; bool on; };
        std::vector<Row> rows = {
            {"gemm",  "block32",  nop,       [&]{ k_block_gemm<T,N><<<B, 32>>>(dA, dB, dC, B); },  true},
            {"gemm",  "block128", nop,       [&]{ k_block_gemm<T,N><<<B, 128>>>(dA, dB, dC, B); }, true},
            {"gemm",  "warp8",    nop,       [&]{ k_warp_gemm<T,N><<<wgrid, dim3(32,8)>>>(dA, dB, dC, B); }, true},
            {"gemm",  "thread128", nop,      [&]{ k_thread_gemm<T,N><<<tgrid, 128>>>(dA, dB, dC, B); }, true},
            {"gemm",  "vendor",   nop,       [&]{ xgemm_sb(H.cb, N, (T)1, dA, MM, dB, MM, (T)0, dC, MM, B); }, true},
            {"potrf", "block32",  restoreS,  [&]{ k_block_potrf<T,N><<<B, 32>>>(dS, B); },  true},
            {"potrf", "block128", restoreS,  [&]{ k_block_potrf<T,N><<<B, 128>>>(dS, B); }, true},
            {"potrf", "warp8",    restoreS,  [&]{ k_warp_potrf<T,N><<<wgrid, dim3(32,8)>>>(dS, B); }, true},
            {"potrf", "thread128", restoreS, [&]{ k_thread_potrf<T,N><<<tgrid, 128>>>(dS, B); }, true},
            {"potrf", "vendor",   restoreS,  [&]{ xpotrf_batched(H.cs, N, dSp, dinfo, B); }, true},
            {"posv",  "block32",  restoreSb, [&]{ k_block_posv<T,N><<<B, 32>>>(dS, db, B); },  true},
            {"posv",  "block128", restoreSb, [&]{ k_block_posv<T,N><<<B, 128>>>(dS, db, B); }, true},
            {"posv",  "thread128", restoreSb,[&]{ k_thread_posv<T,N><<<tgrid, 128>>>(dS, db, B); }, true},
            {"posv",  "vendor",   restoreSb, [&]{ xpotrf_batched(H.cs, N, dSp, dinfo, B);
                                                  xpotrs_batched(H.cs, N, dSp, dbp, dinfo + B, B); }, true},
        };
        if (sizeof(T) == 4)
            rows.push_back({"gemm", "vendor_tf32", nop,
                            [&]{ xgemm_sb(H.cb_tf32, N, (T)1, dA, MM, dB, MM, (T)0, dC, MM, B); }, true});
#if defined(GLASS_BENCH_MAGMA)
        T **dAp = nullptr, **dBp = nullptr, **dCp = nullptr;
        {
            std::vector<T*> hp(B);
            CK(cudaMalloc(&dAp, B*sizeof(T*))); CK(cudaMalloc(&dBp, B*sizeof(T*)));
            CK(cudaMalloc(&dCp, B*sizeof(T*)));
            for (int p = 0; p < B; p++) hp[p] = dA + (size_t)p*MM;
            CK(cudaMemcpy(dAp, hp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
            for (int p = 0; p < B; p++) hp[p] = dB + (size_t)p*MM;
            CK(cudaMemcpy(dBp, hp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
            for (int p = 0; p < B; p++) hp[p] = dC + (size_t)p*MM;
            CK(cudaMemcpy(dCp, hp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
        }
        rows.push_back({"gemm",  "magma",    nop,       [&]{ mgemm_sb(H.mq, N, (T)1, dA, MM, dB, MM, (T)0, dC, MM, B); }, true});
        rows.push_back({"gemm",  "magma_pa", nop,       [&]{ mgemm_pa(H.mq, N, (T)1, dAp, dBp, (T)0, dCp, B); }, true});
        rows.push_back({"potrf", "magma",    restoreS,  [&]{ mpotrf_batched(H.mq, N, dSp, dinfo, B); }, true});
        rows.push_back({"posv",  "magma",    restoreSb, [&]{ mposv_batched(H.mq, N, dSp, dbp, dinfo, B); }, true});
#endif
        std::mt19937 order_rng(0x474c4153u ^ (N << 16) ^ (uint32_t)B ^ (uint32_t)sizeof(T));
        std::shuffle(rows.begin(), rows.end(), order_rng);
        for (auto& r : rows) {
            double ns = time_thru_ns_per_prob(B, r.restore, r.work);
            printf("RESULT section=thru op=%s dtype=%s N=%u B=%d impl=%s ns=%.2f spread=%.2f%%\n",
                   r.op, dt, N, B, r.impl, ns, last_spread_pct);
            fflush(stdout);
        }
#if defined(GLASS_BENCH_MAGMA)
        CK(cudaFree(dAp)); CK(cudaFree(dBp)); CK(cudaFree(dCp));
#endif
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dS));
        CK(cudaFree(dS0)); CK(cudaFree(db)); CK(cudaFree(db0));
        CK(cudaFree(dSp)); CK(cudaFree(dbp)); CK(cudaFree(dinfo));
    }

    // ── latency, batch=1: LAT_CALLS pristine copies, wall-clock incl. sync ──
    if (do_lat) {
        const int R = LAT_CALLS;
        T *dA, *dB, *dC, *dS, *dS0, *db, *db0, *dwork; int* dinfo;
        CK(cudaMalloc(&dA, MM*R*sizeof(T))); CK(cudaMalloc(&dB, MM*R*sizeof(T)));
        CK(cudaMalloc(&dC, MM*R*sizeof(T))); CK(cudaMalloc(&dS, MM*R*sizeof(T)));
        CK(cudaMalloc(&dS0, MM*R*sizeof(T)));
        CK(cudaMalloc(&db, (size_t)N*R*sizeof(T))); CK(cudaMalloc(&db0, (size_t)N*R*sizeof(T)));
        CK(cudaMalloc(&dinfo, sizeof(int)));
        {   std::vector<T> st(MM * (size_t)R);
            for (int p = 0; p < R; p++)
                for (size_t i = 0; i < MM; i++) st[p*MM + i] = (T)hA[i];
            CK(cudaMemcpy(dA, st.data(), st.size()*sizeof(T), cudaMemcpyHostToDevice));
            for (int p = 0; p < R; p++)
                for (size_t i = 0; i < MM; i++) st[p*MM + i] = (T)hB[i];
            CK(cudaMemcpy(dB, st.data(), st.size()*sizeof(T), cudaMemcpyHostToDevice));
            for (int p = 0; p < R; p++)
                for (size_t i = 0; i < MM; i++) st[p*MM + i] = (T)hSPD[i];
            CK(cudaMemcpy(dS0, st.data(), st.size()*sizeof(T), cudaMemcpyHostToDevice));
            std::vector<T> sb((size_t)N*R);
            for (int p = 0; p < R; p++) for (uint32_t i = 0; i < N; i++) sb[p*N + i] = (T)hb[i];
            CK(cudaMemcpy(db0, sb.data(), sb.size()*sizeof(T), cudaMemcpyHostToDevice)); }
        int lwork = xpotrf_bufsize(H.cs, N, dS);
        CK(cudaMalloc(&dwork, (size_t)lwork * sizeof(T)));

#if defined(GLASS_BENCH_MAGMA)
        // Per-call pointer arrays so the batched MAGMA entry points run at
        // batch=1 on call r's pristine copy (measures MAGMA's real
        // single-call API floor instead of asserting it).
        T **dSp_lat, **dbp_lat;
        {
            std::vector<T*> hp(R);
            CK(cudaMalloc(&dSp_lat, R*sizeof(T*))); CK(cudaMalloc(&dbp_lat, R*sizeof(T*)));
            for (int r = 0; r < R; r++) hp[r] = dS + (size_t)r*MM;
            CK(cudaMemcpy(dSp_lat, hp.data(), R*sizeof(T*), cudaMemcpyHostToDevice));
            for (int r = 0; r < R; r++) hp[r] = db + (size_t)r*N;
            CK(cudaMemcpy(dbp_lat, hp.data(), R*sizeof(T*), cudaMemcpyHostToDevice));
        }
#endif
        struct Lat { const char* op; const char* impl; std::function<void(int)> call; };
        std::vector<Lat> lats = {
            {"gemm",  "block", [&](int r){ k_block_gemm<T,N><<<1, 64>>>(dA + (size_t)r*MM, dB + (size_t)r*MM, dC + (size_t)r*MM, 1); }},
            {"gemm",  "thread",[&](int r){ k_thread_gemm<T,N><<<1, 1>>>(dA + (size_t)r*MM, dB + (size_t)r*MM, dC + (size_t)r*MM, 1); }},
            {"gemm",  "vendor",[&](int r){ xgemm(H.cb, N, (T)1, dA + (size_t)r*MM, dB + (size_t)r*MM, (T)0, dC + (size_t)r*MM); }},
            {"potrf", "block", [&](int r){ k_block_potrf<T,N><<<1, 32>>>(dS + (size_t)r*MM, 1); }},
            {"potrf", "thread",[&](int r){ k_thread_potrf<T,N><<<1, 1>>>(dS + (size_t)r*MM, 1); }},
            {"potrf", "vendor",[&](int r){ xpotrf(H.cs, N, dS + (size_t)r*MM, dwork, lwork, dinfo); }},
            {"posv",  "block", [&](int r){ k_block_posv<T,N><<<1, 32>>>(dS + (size_t)r*MM, db + (size_t)r*N, 1); }},
            {"posv",  "thread",[&](int r){ k_thread_posv<T,N><<<1, 1>>>(dS + (size_t)r*MM, db + (size_t)r*N, 1); }},
            {"posv",  "vendor",[&](int r){ xpotrf(H.cs, N, dS + (size_t)r*MM, dwork, lwork, dinfo);
                                           xpotrs(H.cs, N, dS + (size_t)r*MM, db + (size_t)r*N, dinfo); }},
        };
#if defined(GLASS_BENCH_MAGMA)
        lats.push_back({"gemm",  "magma", [&](int r){ mgemm_sb(H.mq, N, (T)1, dA + (size_t)r*MM, MM, dB + (size_t)r*MM, MM, (T)0, dC + (size_t)r*MM, MM, 1); }});
        lats.push_back({"potrf", "magma", [&](int r){ mpotrf_batched(H.mq, N, dSp_lat + r, dinfo, 1); }});
        lats.push_back({"posv",  "magma", [&](int r){ mposv_batched(H.mq, N, dSp_lat + r, dbp_lat + r, dinfo, 1); }});
#endif
        if (sizeof(T) == 4)
            lats.push_back({"gemm", "vendor_tf32",
                            [&](int r){ xgemm(H.cb_tf32, N, (T)1, dA + (size_t)r*MM,
                                              dB + (size_t)r*MM, (T)0, dC + (size_t)r*MM); }});
        std::mt19937 latency_order_rng(0x4c41544eu ^ (N << 8) ^ (uint32_t)sizeof(T));
        std::shuffle(lats.begin(), lats.end(), latency_order_rng);
        for (auto& L : lats) {
            double best_us = 1e300, worst_us = 0.0;
            for (int trial = 0; trial < 3; trial++) {
                // Pristine pool per trial; restoration is outside the timed window.
                CK(cudaMemcpy(dS, dS0, MM*(size_t)R*sizeof(T), cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(db, db0, (size_t)N*R*sizeof(T), cudaMemcpyDeviceToDevice));
                for (int r = 0; r < 10; r++) L.call(r);
                CK(cudaDeviceSynchronize());
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int r = 10; r < R; r++) { L.call(r); CK(cudaDeviceSynchronize()); }
                clock_gettime(CLOCK_MONOTONIC, &t1);
                double us = elapsed_us(t0, t1) / (R - 10);
                if (us < best_us) best_us = us;
                if (us > worst_us) worst_us = us;
            }
            printf("RESULT section=lat op=%s dtype=%s N=%u impl=%s us=%.3f spread=%.2f%%\n",
                   L.op, dt, N, L.impl, best_us, (worst_us / best_us - 1.0) * 100.0);
            fflush(stdout);
        }
        CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dS));
        CK(cudaFree(dS0)); CK(cudaFree(db)); CK(cudaFree(db0));
        CK(cudaFree(dwork)); CK(cudaFree(dinfo));
#if defined(GLASS_BENCH_MAGMA)
        CK(cudaFree(dSp_lat)); CK(cudaFree(dbp_lat));
#endif
    }
}

// ─── size dispatch ───────────────────────────────────────────────────────────
#define FOREACH_N(X) X(4) X(6) X(8) X(12) X(16) X(24) X(32) X(48) X(64)

template <typename T>
static void run_dtype(Handles H, const char* dt, bool do_thru, bool do_lat) {
#define RUN_ONE(NN) run_for_N<T, NN>(H, dt, do_thru, do_lat);
    FOREACH_N(RUN_ONE)
#undef RUN_ONE
}

int main(int argc, char** argv) {
    REPS = (argc > 1) ? atoi(argv[1]) : 50;
    std::string dt  = (argc > 2) ? argv[2] : "both";
    std::string sec = (argc > 3) ? argv[3] : "all";
    bool do_thru = (sec == "all" || sec == "thru");
    bool do_lat  = (sec == "all" || sec == "lat");

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# bench_paper_hostblas — %s, reps=%d, dtype=%s, section=%s\n",
           prop.name, REPS, dt.c_str(), sec.c_str());
    tc_warm_gpu();

    Handles H;
    CB(cublasCreate(&H.cb));
    CB(cublasCreate(&H.cb_tf32));
    CB(cublasSetMathMode(H.cb_tf32, CUBLAS_TF32_TENSOR_OP_MATH));
    CS(cusolverDnCreate(&H.cs));
#if defined(GLASS_BENCH_MAGMA)
    magma_init();
    // Queue on the DEFAULT stream so the shared event bracketing times MAGMA
    // work exactly like the cuBLAS/cuSOLVER rows.
    magma_queue_create_from_cuda(0, nullptr, H.cb, nullptr, &H.mq);
    printf("# magma columns enabled (version %d.%d.%d)\n",
           MAGMA_VERSION_MAJOR, MAGMA_VERSION_MINOR, MAGMA_VERSION_MICRO);
#endif
    if (dt == "f32" || dt == "both") run_dtype<float >(H, "f32", do_thru, do_lat);
    if (dt == "f64" || dt == "both") run_dtype<double>(H, "f64", do_thru, do_lat);
#if defined(GLASS_BENCH_MAGMA)
    magma_queue_destroy(H.mq);
    magma_finalize();
#endif
    CB(cublasDestroy(H.cb));
    CB(cublasDestroy(H.cb_tf32));
    CS(cusolverDnDestroy(H.cs));
    printf("# done\n");
    return 0;
}
