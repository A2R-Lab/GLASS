// bench_paper_fusion.cu — the paper's F3 figure: a FUSED device-side Riccati
// gain step vs the same math as a chain of host-batched cuBLAS/cuSOLVER calls.
//
//   K = (R + BᵀPB)⁻¹ (BᵀPA)      P: NX×NX SPD, A: NX×NX, B: NX×NU, R: NU×NU SPD
//
// FUSED (GLASS): one kernel, one block per problem — stage P/A/B/R into smem,
//   glass::riccati_gain<T,NX,NU> (congruence + bilinear + multi-RHS posv, all
//   in smem), write only K back. Intermediates (PB, PA, S, G) never touch
//   global memory.
// CHAIN (vendor): 7 host calls per step, batched over all problems, every
//   intermediate in global memory:
//     1. S ← R                    (D2D memcpy, strided)
//     2. PB = P·B                 cublas<X>gemmStridedBatched
//     3. S += Bᵀ·PB               gemmStridedBatched (OP_T, beta=1)
//     4. PA = P·A                 gemmStridedBatched
//     5. G = Bᵀ·PA                gemmStridedBatched (OP_T)
//     6. S = L·Lᵀ                 cusolverDn<X>potrfBatched
//     7. L·Y=G, LᵀK=Y             cublas<X>trsmBatched ×2  (K lands in G)
//   The chain is SELF-RESTORING (S is rebuilt from pristine R every rep), so
//   the throughput protocol needs no out-of-window restore.
//
// PROTOCOL: as bench_solvers.cu — each rep = one fused launch / one full chain
// spanning all B problems, cudaEvent-bracketed; ns/problem = ms*1e6/(reps*B),
// min of 3 trials with spread reported. CORRECTNESS: fused AND chain vs host double reference on
// problem 0 before any timing (mismatch aborts).
//
// Compile: nvcc -std=c++17 -arch=sm_XX -O3 --expt-relaxed-constexpr
//          -I.. -I../src bench_paper_fusion.cu -o bench_paper_fusion
//          -lcublas -lcusolver          (no MathDx needed)
// Usage:   ./bench_paper_fusion [reps=50] [dtype=f32|f64|both]
//
// Output grammar (parsed by paper_sweeps.py):
//   CHECK  shape=<NX>x<NU> dtype=<dt> impl=<fused|chain> maxerr=<e>
//   RESULT op=riccati dtype=<dt> NX=<nx> NU=<nu> B=<b> impl=<impl> ns=<ns_per_problem>
//   SKIP   ... reason=...

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
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

static const int BGRID[]  = {1, 16, 64, 256, 1024, 4096};
static const int NBGRID   = 6;
static const size_t MEM_CAP = 4ull << 30;
static int REPS = 50;

// ─── deterministic host RNG ──────────────────────────────────────────────────
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

// ─── host double reference: K = (R + BᵀPB)⁻¹ BᵀPA ────────────────────────────
static void ref_riccati(int nx, int nu, const double* P, const double* A,
                        const double* B, const double* R, double* K) {
    std::vector<double> PB((size_t)nx*nu), PA((size_t)nx*nx),
                        S((size_t)nu*nu), G((size_t)nu*nx);
    for (int j = 0; j < nu; j++)
        for (int i = 0; i < nx; i++) {
            double s = 0;
            for (int k = 0; k < nx; k++) s += P[i + k*nx] * B[k + j*nx];
            PB[i + j*nx] = s;
        }
    for (int j = 0; j < nu; j++)
        for (int i = 0; i < nu; i++) {
            double s = R[i + j*nu];
            for (int k = 0; k < nx; k++) s += B[k + i*nx] * PB[k + j*nx];
            S[i + j*nu] = s;
        }
    for (int j = 0; j < nx; j++)
        for (int i = 0; i < nx; i++) {
            double s = 0;
            for (int k = 0; k < nx; k++) s += P[i + k*nx] * A[k + j*nx];
            PA[i + j*nx] = s;
        }
    for (int j = 0; j < nx; j++)
        for (int i = 0; i < nu; i++) {
            double s = 0;
            for (int k = 0; k < nx; k++) s += B[k + i*nx] * PA[k + j*nx];
            G[i + j*nu] = s;
        }
    // chol(S) in place, then column-wise solve S K = G
    for (int j = 0; j < nu; j++) {
        double d = S[j + j*nu];
        for (int k = 0; k < j; k++) d -= S[j + k*nu] * S[j + k*nu];
        if (d <= 0.0) { fprintf(stderr, "ref_riccati: S not PD\n"); exit(4); }
        d = sqrt(d); S[j + j*nu] = d;
        for (int i = j + 1; i < nu; i++) {
            double s = S[i + j*nu];
            for (int k = 0; k < j; k++) s -= S[i + k*nu] * S[j + k*nu];
            S[i + j*nu] = s / d;
        }
    }
    for (int c = 0; c < nx; c++) {
        double* g = &G[(size_t)c*nu];
        for (int i = 0; i < nu; i++) {
            double s = g[i];
            for (int k = 0; k < i; k++) s -= S[i + k*nu] * g[k];
            g[i] = s / S[i + i*nu];
        }
        for (int i = nu - 1; i >= 0; i--) {
            double s = g[i];
            for (int k = i + 1; k < nu; k++) s -= S[k + i*nu] * g[k];
            g[i] = s / S[i + i*nu];
        }
        for (int i = 0; i < nu; i++) K[i + (size_t)c*nu] = g[i];
    }
}

// ─── vendor wrappers, overloaded on dtype ────────────────────────────────────
static void xgemm_sb(cublasHandle_t h, cublasOperation_t ta, int m, int n, int k,
                     float alpha, const float* A, int lda, long long sA,
                     const float* B, int ldb, long long sB,
                     float beta, float* C, int ldc, long long sC, int batch) {
    CB(cublasSgemmStridedBatched(h, ta, CUBLAS_OP_N, m, n, k, &alpha,
                                 A, lda, sA, B, ldb, sB, &beta, C, ldc, sC, batch));
}
static void xgemm_sb(cublasHandle_t h, cublasOperation_t ta, int m, int n, int k,
                     double alpha, const double* A, int lda, long long sA,
                     const double* B, int ldb, long long sB,
                     double beta, double* C, int ldc, long long sC, int batch) {
    CB(cublasDgemmStridedBatched(h, ta, CUBLAS_OP_N, m, n, k, &alpha,
                                 A, lda, sA, B, ldb, sB, &beta, C, ldc, sC, batch));
}
static void xpotrf_batched(cusolverDnHandle_t h, int n, float** Aarr, int* info, int batch) {
    CS(cusolverDnSpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, Aarr, n, info, batch));
}
static void xpotrf_batched(cusolverDnHandle_t h, int n, double** Aarr, int* info, int batch) {
    CS(cusolverDnDpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, Aarr, n, info, batch));
}
static void xtrsm_batched(cublasHandle_t h, cublasOperation_t trans, int m, int n,
                          float alpha, const float* const* Aarr, int lda,
                          float* const* Barr, int ldb, int batch) {
    CB(cublasStrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans,
                          CUBLAS_DIAG_NON_UNIT, m, n, &alpha, Aarr, lda, Barr, ldb, batch));
}
static void xtrsm_batched(cublasHandle_t h, cublasOperation_t trans, int m, int n,
                          double alpha, const double* const* Aarr, int lda,
                          double* const* Barr, int ldb, int batch) {
    CB(cublasDtrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans,
                          CUBLAS_DIAG_NON_UNIT, m, n, &alpha, Aarr, lda, Barr, ldb, batch));
}

// ─── fused GLASS kernel ──────────────────────────────────────────────────────
template <typename T, uint32_t NX, uint32_t NU>
__host__ __device__ constexpr size_t fused_smem_bytes() {
    // staged inputs + K + riccati_gain scratch (NU² + NX·max(NX,NU))
    return (2*NX*NX + NX*NU + NU*NU + NU*NX
            + NU*NU + NX * (NX >= NU ? NX : NU)) * sizeof(T);
}

template <typename T, uint32_t NX, uint32_t NU>
__global__ void k_fused_riccati(const T* gP, const T* gA, const T* gB, const T* gR,
                                T* gK, int nprob) {
    extern __shared__ __align__(16) char smem_raw[];
    T* sP = reinterpret_cast<T*>(smem_raw);
    T* sA = sP + NX*NX;
    T* sB = sA + NX*NX;
    T* sR = sB + NX*NU;
    T* sK = sR + NU*NU;
    T* scr = sK + NU*NX;

    const size_t p = blockIdx.x;
    const uint32_t rank = threadIdx.x, size = blockDim.x;
    for (uint32_t i = rank; i < NX*NX; i += size) { sP[i] = gP[p*NX*NX + i];
                                                    sA[i] = gA[p*NX*NX + i]; }
    for (uint32_t i = rank; i < NX*NU; i += size)   sB[i] = gB[p*NX*NU + i];
    for (uint32_t i = rank; i < NU*NU; i += size)   sR[i] = gR[p*NU*NU + i];
    __syncthreads();

    glass::block::riccati_gain<T, NX, NU>(sP, sA, sB, sR, sK, scr);

    for (uint32_t i = rank; i < NU*NX; i += size)   gK[p*NU*NX + i] = sK[i];
}

// ─── per-(dtype, shape) driver ───────────────────────────────────────────────
struct Handles { cublasHandle_t cb; cusolverDnHandle_t cs; };

template <typename T, uint32_t NX, uint32_t NU>
static void run_shape(Handles H, const char* dt) {
    const double tol = (sizeof(T) == 4 ? 5e-3 : 1e-8) * NX;
    const size_t PP = (size_t)NX*NX, BB = (size_t)NX*NU,
                 RR = (size_t)NU*NU, KK = (size_t)NU*NX;
    rng_seed(0x51CC ^ (NX*131u + NU));

    // opt-in dynamic smem when the staged footprint exceeds the 48KB default
    const size_t smem = fused_smem_bytes<T, NX, NU>();
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    if (smem > (size_t)prop.sharedMemPerBlockOptin) {
        printf("SKIP op=riccati dtype=%s NX=%u NU=%u reason=smem_%zuKB_over_optin\n",
               dt, NX, NU, smem >> 10);
        return;
    }
    if (smem > 48*1024)
        CK(cudaFuncSetAttribute(k_fused_riccati<T, NX, NU>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    // problem-0 host masters (double)
    std::vector<double> hP(PP), hA(PP), hB(BB), hR(RR), hM(PP), refK(KK);
    for (size_t i = 0; i < PP; i++) hM[i] = nrand();
    for (uint32_t j = 0; j < NX; j++)                       // P = MMᵀ/NX + I  (SPD)
        for (uint32_t i = 0; i < NX; i++) {
            double s = 0;
            for (uint32_t k = 0; k < NX; k++) s += hM[i + k*NX] * hM[j + k*NX];
            hP[i + j*NX] = s / NX + (i == j ? 1.0 : 0.0);
        }
    for (size_t i = 0; i < PP; i++) hA[i] = nrand() / sqrt((double)NX);
    for (size_t i = 0; i < BB; i++) hB[i] = nrand() / sqrt((double)NX);
    for (size_t i = 0; i < RR; i++) hM[i] = nrand();
    for (uint32_t j = 0; j < NU; j++)                       // R = MMᵀ/NU + I  (SPD)
        for (uint32_t i = 0; i < NU; i++) {
            double s = 0;
            for (uint32_t k = 0; k < NU; k++) s += hM[i + k*NU] * hM[j + k*NU];
            hR[i + j*NU] = s / NU + (i == j ? 1.0 : 0.0);
        }
    ref_riccati(NX, NU, hP.data(), hA.data(), hB.data(), hR.data(), refK.data());

    for (int bi = 0; bi < NBGRID; bi++) {
        const int B = BGRID[bi];
        const size_t bytes = ((2*PP + BB + RR + KK)              // inputs + K
                            + (PP + BB + RR + KK)                // chain PA/PB/S/G
                             ) * (size_t)B * sizeof(T);
        if (bytes > MEM_CAP) {
            printf("SKIP op=riccati dtype=%s NX=%u NU=%u B=%d reason=mem_cap_%zuMB\n",
                   dt, NX, NU, B, bytes >> 20);
            continue;
        }
        T *dP, *dA, *dB_, *dR, *dK, *dPB, *dPA, *dS, *dG;
        CK(cudaMalloc(&dP,  PP*B*sizeof(T))); CK(cudaMalloc(&dA,  PP*B*sizeof(T)));
        CK(cudaMalloc(&dB_, BB*B*sizeof(T))); CK(cudaMalloc(&dR,  RR*B*sizeof(T)));
        CK(cudaMalloc(&dK,  KK*B*sizeof(T)));
        CK(cudaMalloc(&dPB, BB*B*sizeof(T))); CK(cudaMalloc(&dPA, PP*B*sizeof(T)));
        CK(cudaMalloc(&dS,  RR*B*sizeof(T))); CK(cudaMalloc(&dG,  KK*B*sizeof(T)));
        {   // every problem gets problem-0's data (correctness reference holds
            // for all; identical flop count regardless of values)
            std::vector<T> st(PP*(size_t)B);
            for (int p = 0; p < B; p++) for (size_t i = 0; i < PP; i++) st[p*PP+i] = (T)hP[i];
            CK(cudaMemcpy(dP, st.data(), PP*B*sizeof(T), cudaMemcpyHostToDevice));
            for (int p = 0; p < B; p++) for (size_t i = 0; i < PP; i++) st[p*PP+i] = (T)hA[i];
            CK(cudaMemcpy(dA, st.data(), PP*B*sizeof(T), cudaMemcpyHostToDevice));
            st.resize(BB*(size_t)B);
            for (int p = 0; p < B; p++) for (size_t i = 0; i < BB; i++) st[p*BB+i] = (T)hB[i];
            CK(cudaMemcpy(dB_, st.data(), BB*B*sizeof(T), cudaMemcpyHostToDevice));
            st.resize(RR*(size_t)B);
            for (int p = 0; p < B; p++) for (size_t i = 0; i < RR; i++) st[p*RR+i] = (T)hR[i];
            CK(cudaMemcpy(dR, st.data(), RR*B*sizeof(T), cudaMemcpyHostToDevice)); }

        std::vector<T*> hSp(B), hGp(B);
        for (int p = 0; p < B; p++) { hSp[p] = dS + (size_t)p*RR; hGp[p] = dG + (size_t)p*KK; }
        T **dSp, **dGp; int* dinfo;
        CK(cudaMalloc(&dSp, B*sizeof(T*))); CK(cudaMalloc(&dGp, B*sizeof(T*)));
        CK(cudaMalloc(&dinfo, B*sizeof(int)));
        CK(cudaMemcpy(dSp, hSp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dGp, hGp.data(), B*sizeof(T*), cudaMemcpyHostToDevice));

        auto fused = [&](int tb) {
            k_fused_riccati<T, NX, NU><<<B, tb, smem>>>(dP, dA, dB_, dR, dK, B);
        };
        auto chain = [&] {
            CK(cudaMemcpyAsync(dS, dR, RR*(size_t)B*sizeof(T), cudaMemcpyDeviceToDevice));
            xgemm_sb(H.cb, CUBLAS_OP_N, NX, NU, NX, (T)1, dP, NX, PP, dB_, NX, BB, (T)0, dPB, NX, BB, B);
            xgemm_sb(H.cb, CUBLAS_OP_T, NU, NU, NX, (T)1, dB_, NX, BB, dPB, NX, BB, (T)1, dS, NU, RR, B);
            xgemm_sb(H.cb, CUBLAS_OP_N, NX, NX, NX, (T)1, dP, NX, PP, dA, NX, PP, (T)0, dPA, NX, PP, B);
            xgemm_sb(H.cb, CUBLAS_OP_T, NU, NX, NX, (T)1, dB_, NX, BB, dPA, NX, PP, (T)0, dG, NU, KK, B);
            xpotrf_batched(H.cs, NU, dSp, dinfo, B);
            xtrsm_batched(H.cb, CUBLAS_OP_N, NU, NX, (T)1, dSp, NU, dGp, NU, B);
            xtrsm_batched(H.cb, CUBLAS_OP_T, NU, NX, (T)1, dSp, NU, dGp, NU, B);
        };

        // correctness (once, at the smallest B)
        if (bi == 0) {
            fused(128); CK(cudaDeviceSynchronize());
            std::vector<T> hK(KK);
            CK(cudaMemcpy(hK.data(), dK, KK*sizeof(T), cudaMemcpyDeviceToHost));
            double me = 0;
            for (size_t i = 0; i < KK; i++) me = fmax(me, fabs((double)hK[i] - refK[i]));
            printf("CHECK shape=%ux%u dtype=%s impl=fused maxerr=%.3e\n", NX, NU, dt, me);
            if (!(me < tol)) { fprintf(stderr, "FATAL fused mismatch\n"); exit(4); }
            chain(); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hK.data(), dG, KK*sizeof(T), cudaMemcpyDeviceToHost));
            me = 0;
            for (size_t i = 0; i < KK; i++) me = fmax(me, fabs((double)hK[i] - refK[i]));
            printf("CHECK shape=%ux%u dtype=%s impl=chain maxerr=%.3e\n", NX, NU, dt, me);
            if (!(me < tol)) { fprintf(stderr, "FATAL chain mismatch\n"); exit(4); }
        }

        // timing: min over 3 trials of REPS event-bracketed reps
        cudaEvent_t ev_s, ev_e;
        CK(cudaEventCreate(&ev_s)); CK(cudaEventCreate(&ev_e));
        struct Row { const char* impl; std::function<void()> work; };
        std::vector<Row> rows = {
            {"fused_tb32",  [&]{ fused(32); }},
            {"fused_tb128", [&]{ fused(128); }},
            {"fused_tb256", [&]{ fused(256); }},   // wide-block probe: the (36,12)/(48,16)
            {"fused_tb512", [&]{ fused(512); }},   // chain-wins cells may be TB-starved
            {"chain",       chain},
        };
        std::mt19937 order_rng(0x46555345u ^ (NX << 20) ^ (NU << 12) ^
                               (uint32_t)B ^ (uint32_t)sizeof(T));
        std::shuffle(rows.begin(), rows.end(), order_rng);
        for (auto& r : rows) {
            r.work(); CK(cudaDeviceSynchronize());   // warm-up
            double best_ms = 1e300, worst_ms = 0.0;
            for (int trial = 0; trial < 3; trial++) {
                double ms_sum = 0;
                for (int rep = 0; rep < REPS; rep++) {
                    CK(cudaEventRecord(ev_s));
                    r.work();
                    CK(cudaEventRecord(ev_e));
                    CK(cudaEventSynchronize(ev_e));
                    float ms; CK(cudaEventElapsedTime(&ms, ev_s, ev_e));
                    ms_sum += ms;
                }
                if (ms_sum < best_ms) best_ms = ms_sum;
                if (ms_sum > worst_ms) worst_ms = ms_sum;
            }
            printf("RESULT op=riccati dtype=%s NX=%u NU=%u B=%d impl=%s ns=%.2f spread=%.2f%%\n",
                   dt, NX, NU, B, r.impl, best_ms * 1e6 / ((double)REPS * B),
                   (worst_ms / best_ms - 1.0) * 100.0);
            fflush(stdout);
        }
        CK(cudaEventDestroy(ev_s)); CK(cudaEventDestroy(ev_e));
        CK(cudaFree(dP)); CK(cudaFree(dA)); CK(cudaFree(dB_)); CK(cudaFree(dR));
        CK(cudaFree(dK)); CK(cudaFree(dPB)); CK(cudaFree(dPA)); CK(cudaFree(dS));
        CK(cudaFree(dG)); CK(cudaFree(dSp)); CK(cudaFree(dGp)); CK(cudaFree(dinfo));
    }
}

template <typename T>
static void run_dtype(Handles H, const char* dt) {
    run_shape<T, 12,  4>(H, dt);   // quadrotor-ish
    run_shape<T, 14,  7>(H, dt);   // iiwa (nx=2*nq, nu=nq)
    run_shape<T, 36, 12>(H, dt);   // quadruped-ish
    run_shape<T, 48, 16>(H, dt);   // humanoid-ish
}

int main(int argc, char** argv) {
    REPS = (argc > 1) ? atoi(argv[1]) : 50;
    std::string dt = (argc > 2) ? argv[2] : "both";

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    tc_warm_gpu();
    printf("# bench_paper_fusion — %s, reps=%d, dtype=%s\n", prop.name, REPS, dt.c_str());

    Handles H;
    CB(cublasCreate(&H.cb));
    CS(cusolverDnCreate(&H.cs));
    if (dt == "f32" || dt == "both") run_dtype<float >(H, "f32");
    if (dt == "f64" || dt == "both") run_dtype<double>(H, "f64");
    CB(cublasDestroy(H.cb));
    CS(cusolverDnDestroy(H.cs));
    printf("# done\n");
    return 0;
}
