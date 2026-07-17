// bench_solvers.cu — characterization sweep for the SOLVER-level ops:
//
//   A. bdsv vs pcg   — glass::bdsv (direct block-Cholesky sweep) vs glass::pcg
//                      (block-Jacobi-preconditioned CG, rel_tol=1e-6, abs_tol=
//                      1e-12, max 200 iters, warm zero start) on the IDENTICAL
//                      block-tridiagonal SPD input, (BlockSize,Knots) ∈
//                      {(2,8),(2,32),(6,8),(6,32),(6,64),(12,16)}.
//   B. gesv vs posv vs inv+gemv — the three ways to solve one SPD system
//                      (N ∈ {4,8,16,32,64}, single RHS): prices the pivoted-LU
//                      overhead where Cholesky suffices, and the
//                      invert-then-multiply anti-pattern.
//   C. syev + eig_clamp — timing only (no contender), N ∈ {4,8,16,32}.
//
// PROTOCOL (differs from bench_blas2.cu because these ops MUTATE their input):
// NPROB independent problem copies live in global memory; each rep launches one
// kernel over all NPROB problems (one block per problem) bracketed by cudaEvents,
// and the mutated state is RESTORED from pristine device copies by device-to-
// device memcpy/memset OUTSIDE the event window (stream-ordered before the next
// launch, so the events time the kernel only). ns/problem = event_ms summed over
// reps / (reps*NPROB), min of 3 trials. reps can be modest (default 50) since
// each rep already spans NPROB problems. Same CLI + section grammar as
// bench_blas2.cu so tune.py's runner/parser conventions carry over.
//
// CORRECTNESS GUARD (no silent caps): before timing, every section-A shape runs
// bdsv AND pcg on problem 0 and compares both against a host double-precision
// dense Cholesky solve (max|Δ| < 1e-3, pcg must CONVERGE: 0 < iters < 200);
// every section-B N compares gesv/posv/inv+gemv the same way. Mismatch aborts.
//
// Inputs mirror test/test_pcg.py::make_spd_banded — per knot D = M·Mᵀ + SS·I
// (M standard normal), ±0.1-scaled symmetric off-diagonal blocks — ported to
// host C++ (double, cast to the bench dtype at upload). pcg's Pinv strips are
// the block-Jacobi inverses of the D blocks.
//
// Compile: nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1
//          -I.. -I../src bench_solvers.cu -o bench_solvers   (no MathDx needed)
// Usage:   ./bench_solvers [nprob=8192] [reps=50] [dtype=f32|f64]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "../glass.cuh"

static int NPROB = 8192;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(3); } } while (0)

// ─── deterministic host RNG (xorshift64* + Box-Muller) ───────────────────────
static uint64_t rng_state = 1;
static void rng_seed(uint64_t s) { rng_state = s ? s : 0x9E3779B97F4A7C15ull; }
static double urand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
}
static double nrand() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ─── host reference linear algebra (double) ──────────────────────────────────

// In-place dense Cholesky solve: A (n×n col-major, SPD, destroyed), b → x.
static bool host_chol_solve(int n, double* A, double* b) {
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

// Gauss-Jordan inverse with partial pivoting (m ≤ 12 here): Ainv = A⁻¹, col-major.
static void host_inv(int m, const double* A, double* Ainv) {
    double* W = (double*)malloc((size_t)m * 2*m * sizeof(double));  // [A | I] col-major
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
            W[i + j*m] = A[i + j*m];
            W[i + (m + j)*m] = (i == j) ? 1.0 : 0.0;
        }
    for (int k = 0; k < m; k++) {
        int p = k; double best = fabs(W[k + k*m]);
        for (int i = k + 1; i < m; i++)
            if (fabs(W[i + k*m]) > best) { best = fabs(W[i + k*m]); p = i; }
        if (p != k)
            for (int j = 0; j < 2*m; j++) { double t = W[k + j*m]; W[k + j*m] = W[p + j*m]; W[p + j*m] = t; }
        double piv = W[k + k*m];
        for (int j = 0; j < 2*m; j++) W[k + j*m] /= piv;
        for (int i = 0; i < m; i++) {
            if (i == k) continue;
            double f = W[i + k*m];
            if (f != 0.0) for (int j = 0; j < 2*m; j++) W[i + j*m] -= f * W[k + j*m];
        }
    }
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) Ainv[i + j*m] = W[i + (m + j)*m];
    free(W);
}

// Port of test/test_pcg.py::make_spd_banded. Fills the [L|D|R] strips (row-major
// BS×3BS tiles, KP of them), the block-Jacobi Pinv strips (MAIN = D_k⁻¹, rest 0),
// and the dense col-major n×n system (n = BS*KP) for the CPU reference.
static void make_spd_banded(int BS, int KP, double* hBand, double* hPinv, double* hDense) {
    int n = BS * KP;
    size_t stripe = (size_t)3 * BS * BS;
    memset(hBand, 0, (size_t)KP * stripe * sizeof(double));
    memset(hPinv, 0, (size_t)KP * stripe * sizeof(double));
    memset(hDense, 0, (size_t)n * n * sizeof(double));
    double* M     = (double*)malloc((size_t)BS*BS*sizeof(double));
    double* D     = (double*)malloc((size_t)BS*BS*sizeof(double));
    double* R     = (double*)malloc((size_t)BS*BS*sizeof(double));
    double* Rprev = (double*)malloc((size_t)BS*BS*sizeof(double));
    double* Dinv  = (double*)malloc((size_t)BS*BS*sizeof(double));
    for (int k = 0; k < KP; k++) {
        for (int i = 0; i < BS*BS; i++) M[i] = nrand();
        for (int j = 0; j < BS; j++)                        // D = M·Mᵀ + BS·I (col-major,
            for (int i = 0; i < BS; i++) {                  // exactly symmetric by construction)
                double s = (i == j) ? (double)BS : 0.0;
                for (int c = 0; c < BS; c++) s += M[i + c*BS] * M[j + c*BS];
                D[i + j*BS] = s;
            }
        if (k < KP - 1)                                     // R_k = block (k, k+1), col-major
            for (int i = 0; i < BS*BS; i++) R[i] = 0.1 * nrand();
        double* strip = hBand + (size_t)k * stripe;         // row-major BS×3BS: [L|D|R]
        for (int r = 0; r < BS; r++)
            for (int c = 0; c < BS; c++) {
                if (k > 0)      strip[r*3*BS + c]        = Rprev[c + r*BS];   // L = R_{k-1}ᵀ
                strip[r*3*BS + BS + c]                   = D[r + c*BS];       // D
                if (k < KP - 1) strip[r*3*BS + 2*BS + c] = R[r + c*BS];       // R
            }
        host_inv(BS, D, Dinv);                              // block-Jacobi Pinv (MAIN only)
        double* pstrip = hPinv + (size_t)k * stripe;
        for (int r = 0; r < BS; r++)
            for (int c = 0; c < BS; c++) pstrip[r*3*BS + BS + c] = Dinv[r + c*BS];
        for (int j = 0; j < BS; j++)                        // dense mirror
            for (int i = 0; i < BS; i++) {
                hDense[(k*BS + i) + (size_t)(k*BS + j)*n] = D[i + j*BS];
                if (k < KP - 1) {
                    hDense[(k*BS + i) + (size_t)((k+1)*BS + j)*n] = R[i + j*BS];
                    hDense[((k+1)*BS + i) + (size_t)(k*BS + j)*n] = R[j + i*BS];
                }
            }
        double* t = Rprev; Rprev = R; R = t;
    }
    free(M); free(D); free(R); free(Rprev); free(Dinv);
}

// ─── device buffer helpers ────────────────────────────────────────────────────

// Doubling replicate: buffer already holds problem 0's `elems`-long tile.
template<typename T>
static void replicate(T* d, size_t elems, int nprob) {
    size_t have = 1;
    while (have < (size_t)nprob) {
        size_t cnt = have < (size_t)nprob - have ? have : (size_t)nprob - have;
        CK(cudaMemcpy(d + have*elems, d, cnt*elems*sizeof(T), cudaMemcpyDeviceToDevice));
        have += cnt;
    }
}

// Alloc NPROB copies of a double host tile, cast to T.
template<typename T>
static T* upload_tile(const double* h, size_t elems, int nprob) {
    T* d; CK(cudaMalloc(&d, (size_t)nprob*elems*sizeof(T)));
    T* tmp = (T*)malloc(elems*sizeof(T));
    for (size_t i = 0; i < elems; i++) tmp[i] = (T)h[i];
    CK(cudaMemcpy(d, tmp, elems*sizeof(T), cudaMemcpyHostToDevice));
    free(tmp);
    replicate(d, elems, nprob);
    return d;
}

template<typename T>
static double max_abs_diff_dev(const T* dptr, const double* ref, int n) {
    T* h = (T*)malloc((size_t)n*sizeof(T));
    CK(cudaMemcpy(h, dptr, (size_t)n*sizeof(T), cudaMemcpyDeviceToHost));
    double md = 0.0;
    for (int i = 0; i < n; i++) { double d = fabs((double)h[i] - ref[i]); if (d > md) md = d; }
    free(h);
    return md;
}

static void guard_fail(const char* what, double maxd) {
    fprintf(stderr, "SOLVERS GUARD FAILED: %s max|Δ|=%.3e (limit 1e-3) — fix the "
                    "harness/system generation, do not loosen the gate.\n", what, maxd);
    exit(1);
}

// ─── timed protocol: restore OUTSIDE the cudaEvent window ─────────────────────
template<typename L, typename R>
static double time_restored_ns(L launch, R restore, int reps) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    restore(); launch(); CK(cudaDeviceSynchronize());        // warmup (untimed)
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        double total_ms = 0.0;
        for (int r = 0; r < reps; r++) {
            restore();                                       // stream-ordered, before e0
            CK(cudaEventRecord(e0));
            launch();
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            total_ms += ms;
        }
        double ns = total_ms * 1e6 / ((double)reps * NPROB);
        if (ns < best) best = ns;
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return best;
}

// ─── kernels: one block per problem ───────────────────────────────────────────
// All dynamic shared memory goes through one extern symbol (double-aligned).

template<typename T, int BS, int KP>
__global__ void k_bdsv(T* strips, T* vecs) {
    extern __shared__ double solvers_smem[];
    T* s = reinterpret_cast<T*>(solvers_smem);
    size_t p = blockIdx.x;
    glass::bdsv<T, KP, BS>(strips + p * (size_t)KP*3*BS*BS,
                           vecs   + p * (size_t)(KP + 2)*BS, s);
}

template<typename T, int BS, int KP>
__global__ void k_pcg(T* x, T* S, T* Pinv, T* b, uint32_t max_iters,
                      T rel_tol, T abs_tol, uint32_t* iters) {
    extern __shared__ double solvers_smem[];
    T* s_mem = reinterpret_cast<T*>(solvers_smem);
    size_t p = blockIdx.x, band = (size_t)KP*3*BS*BS, vec = (size_t)(KP + 2)*BS;
    glass::pcg<T, BS, KP>(x + p*vec, S + p*band, Pinv + p*band, b + p*vec,
                          s_mem, max_iters, rel_tol, abs_tol, iters + p);
}

template<typename T, int N>
__global__ void k_gesv(T* A, T* b) {
    __shared__ uint32_t piv[N];
    size_t p = blockIdx.x;
    glass::gesv<T, N, 1>(A + p * (size_t)N*N, piv, b + p * (size_t)N);
}

template<typename T, int N>
__global__ void k_posv(T* A, T* b) {
    size_t p = blockIdx.x;
    glass::posv<T, N>(A + p * (size_t)N*N, b + p * (size_t)N);
}

// glass::thread:: tier: one problem per THREAD (32 packed per warp), operands
// register-resident. A whole grid solves all NPROB problems in one launch — the
// same unit time_restored_ns already normalises to (so ns/problem is directly
// comparable to the block-per-problem contenders). Reads A/b from global into
// registers (leaving global A untouched) and writes the solution back to b so the
// correctness guard and restore protocol behave exactly as for the block solvers.
// Only valid below the N<=7 register-residency ceiling (CLAUDE.md).
template<typename T, int N>
__global__ void k_thread_posv(T* A, T* b, int nprob) {
    size_t p = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= (size_t)nprob) return;
    T a[N*N], bv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[p*(size_t)N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[p*(size_t)N + i];
    glass::thread::posv<T, N>(a, bv);
    for (int i = 0; i < N;   i++) b[p*(size_t)N + i] = bv[i];
}

template<typename T, int N>
__global__ void k_invsolve(T* G, T* b, T* y) {   // Gauss-Jordan inv on [A|I], then x = A⁻¹·b
    extern __shared__ double solvers_smem[];
    T* s = reinterpret_cast<T*>(solvers_smem);   // (2N+1) elems
    size_t p = blockIdx.x;
    T* Gp = G + p * (size_t)2*N*N;
    glass::inv<T, N>(Gp, s);
    glass::gemv<T, N, N>((T)1, Gp + (size_t)N*N, b + p * (size_t)N, (T)0, y + p * (size_t)N);
}

template<typename T, int N>
__global__ void k_syev(const T* A, T* W, T* V) {
    extern __shared__ double solvers_smem[];
    T* s = reinterpret_cast<T*>(solvers_smem);
    size_t p = blockIdx.x;
    glass::syev<T, N>(A + p * (size_t)N*N, W + p * (size_t)N, V + p * (size_t)N*N, s);
}

template<typename T, int N>
__global__ void k_eig_clamp(T* A) {
    extern __shared__ double solvers_smem[];
    T* s = reinterpret_cast<T*>(solvers_smem);
    glass::eig_clamp<T, N>(A + (size_t)blockIdx.x*N*N, (T)1e-3, s);
}

// ─── section A: bdsv vs pcg ───────────────────────────────────────────────────

template<typename T, int BS, int KP>
static void bench_bdsv_pcg(int reps) {
    const int n = BS * KP;
    const size_t band = (size_t)KP*3*BS*BS, vec = (size_t)(KP + 2)*BS;
    const uint32_t PCG_MAX = 200;
    const T REL = (T)1e-6, ABS = (T)1e-12;

    rng_seed((uint64_t)BS * 1000 + KP);
    double* hBand  = (double*)malloc(band * sizeof(double));
    double* hPinv  = (double*)malloc(band * sizeof(double));
    double* hDense = (double*)malloc((size_t)n*n*sizeof(double));
    make_spd_banded(BS, KP, hBand, hPinv, hDense);
    double* hVec = (double*)calloc(vec, sizeof(double));       // padded rhs
    double* hRef = (double*)malloc((size_t)n*sizeof(double));
    for (int i = 0; i < n; i++) { hRef[i] = nrand(); hVec[BS + i] = hRef[i]; }
    if (!host_chol_solve(n, hDense, hRef)) {                   // hRef → x_ref
        fprintf(stderr, "SOLVERS GUARD FAILED: host Cholesky says the generated "
                        "system BS=%d KP=%d is not SPD.\n", BS, KP);
        exit(1);
    }

    T* dWork        = upload_tile<T>(hBand, band, NPROB);      // bdsv factors in place
    T* dPristine    = upload_tile<T>(hBand, band, NPROB);      // restore source = pcg's S
    T* dPinv        = upload_tile<T>(hPinv, band, NPROB);
    T* dVecWork     = upload_tile<T>(hVec, vec, NPROB);        // bdsv solves in place
    T* dVecPristine = upload_tile<T>(hVec, vec, NPROB);        // restore source = pcg's b
    T* dX; CK(cudaMalloc(&dX, (size_t)NPROB*vec*sizeof(T)));   // pcg warm zero start
    CK(cudaMemset(dX, 0, (size_t)NPROB*vec*sizeof(T)));
    uint32_t* dIters; CK(cudaMalloc(&dIters, (size_t)NPROB*sizeof(uint32_t)));

    const size_t sm_bdsv = glass::bdsv_scratch_bytes<T, BS>();

    // Correctness guard (problem 0, both solvers vs the CPU double reference).
    k_bdsv<T, BS, KP><<<1, 256, sm_bdsv>>>(dWork, dVecWork);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    double md_bdsv = max_abs_diff_dev(dVecWork + BS, hRef, n);
    if (md_bdsv >= 1e-3) guard_fail("bdsv vs CPU chol", md_bdsv);
    k_pcg<T, BS, KP><<<1, 256, glass::pcg_scratch_bytes<T, BS, KP>(256)>>>(
        dX, dPristine, dPinv, dVecPristine, PCG_MAX, REL, ABS, dIters);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    double md_pcg = max_abs_diff_dev(dX + BS, hRef, n);
    uint32_t iters0 = 0;
    CK(cudaMemcpy(&iters0, dIters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (md_pcg >= 1e-3) guard_fail("pcg vs CPU chol", md_pcg);
    if (iters0 == 0 || iters0 >= PCG_MAX) {
        fprintf(stderr, "SOLVERS GUARD FAILED: pcg did not converge (iters=%u, max=%u) "
                        "on BS=%d KP=%d — fix the system generation, not the tolerance.\n",
                iters0, PCG_MAX, BS, KP);
        exit(1);
    }
    printf("# guard BS=%d KP=%d: bdsv maxD=%.2e  pcg maxD=%.2e it=%u — OK\n",
           BS, KP, md_bdsv, md_pcg, iters0);

    auto restore_bdsv = [&] {
        CK(cudaMemcpyAsync(dWork, dPristine, (size_t)NPROB*band*sizeof(T), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(dVecWork, dVecPristine, (size_t)NPROB*vec*sizeof(T), cudaMemcpyDeviceToDevice));
    };
    auto restore_pcg = [&] {
        CK(cudaMemsetAsync(dX, 0, (size_t)NPROB*vec*sizeof(T)));
    };

    printf("bdsv_pcg BS=%-3d KP=%-3d | BDSV", BS, KP);
    double best_b = 1e30; int tb_b = 0;
    for (int TB : {32, 256}) {
        double ns = time_restored_ns(
            [&] { k_bdsv<T, BS, KP><<<NPROB, TB, sm_bdsv>>>(dWork, dVecWork); },
            restore_bdsv, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best_b) { best_b = ns; tb_b = TB; }
    }
    printf("  | PCG");
    double best_p = 1e30; int tb_p = 0;
    for (int TB : {32, 256}) {
        size_t sm_pcg = glass::pcg_scratch_bytes<T, BS, KP>((uint32_t)TB);
        double ns = time_restored_ns(
            [&] { k_pcg<T, BS, KP><<<NPROB, TB, sm_pcg>>>(dX, dPristine, dPinv,
                      dVecPristine, PCG_MAX, REL, ABS, dIters); },
            restore_pcg, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best_p) { best_p = ns; tb_p = TB; }
    }
    const char* winner = best_b <= best_p ? "BDSV" : "PCG";
    double ratio = best_b <= best_p ? best_p / best_b : best_b / best_p;
    printf("  it=%u  || bdsv tb%d=%.2f  pcg tb%d=%.2f  -> %s (%.2fx)\n",
           iters0, tb_b, best_b, tb_p, best_p, winner, ratio);
    CK(cudaGetLastError());

    cudaFree(dWork); cudaFree(dPristine); cudaFree(dPinv);
    cudaFree(dVecWork); cudaFree(dVecPristine); cudaFree(dX); cudaFree(dIters);
    free(hBand); free(hPinv); free(hDense); free(hVec); free(hRef);
}

// ─── section B: gesv vs posv vs inv+gemv on one SPD system ───────────────────

template<typename T, int N>
static void bench_spdsv(int reps) {
    const size_t mm = (size_t)N*N, gg = (size_t)2*N*N;
    rng_seed(50000 + N);
    double* M  = (double*)malloc(mm*sizeof(double));
    double* hA = (double*)malloc(mm*sizeof(double));
    double* hG = (double*)malloc(gg*sizeof(double));
    double* hB = (double*)malloc((size_t)N*sizeof(double));
    double* hRef = (double*)malloc((size_t)N*sizeof(double));
    double* hAcopy = (double*)malloc(mm*sizeof(double));
    for (size_t i = 0; i < mm; i++) M[i] = nrand();
    for (int j = 0; j < N; j++)                     // A = M·Mᵀ + N·I (SPD, col-major)
        for (int i = 0; i < N; i++) {
            double s = (i == j) ? (double)N : 0.0;
            for (int c = 0; c < N; c++) s += M[i + c*N] * M[j + c*N];
            hA[i + j*N] = s;
        }
    for (int j = 0; j < N; j++)                     // G = [A | I] col-major N×2N
        for (int i = 0; i < N; i++) {
            hG[i + j*N] = hA[i + j*N];
            hG[i + (size_t)(N + j)*N] = (i == j) ? 1.0 : 0.0;
        }
    for (int i = 0; i < N; i++) { hB[i] = nrand(); hRef[i] = hB[i]; }
    memcpy(hAcopy, hA, mm*sizeof(double));
    host_chol_solve(N, hAcopy, hRef);               // hRef → x_ref

    T* dA  = upload_tile<T>(hA, mm, NPROB);
    T* dA0 = upload_tile<T>(hA, mm, NPROB);
    T* dB  = upload_tile<T>(hB, N, NPROB);
    T* dB0 = upload_tile<T>(hB, N, NPROB);
    T* dG  = upload_tile<T>(hG, gg, NPROB);
    T* dG0 = upload_tile<T>(hG, gg, NPROB);
    T* dY; CK(cudaMalloc(&dY, (size_t)NPROB*N*sizeof(T)));

    auto restore_ab = [&] {
        CK(cudaMemcpyAsync(dA, dA0, (size_t)NPROB*mm*sizeof(T), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(dB, dB0, (size_t)NPROB*N*sizeof(T), cudaMemcpyDeviceToDevice));
    };
    auto restore_g = [&] {
        CK(cudaMemcpyAsync(dG, dG0, (size_t)NPROB*gg*sizeof(T), cudaMemcpyDeviceToDevice));
    };
    const size_t sm_inv = (size_t)(2*N + 1) * sizeof(T);

    // Correctness guard (problem 0, all three contenders vs the CPU reference).
    k_gesv<T, N><<<1, 256>>>(dA, dB);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    double md = max_abs_diff_dev(dB, hRef, N);
    if (md >= 1e-3) guard_fail("gesv vs CPU chol", md);
    restore_ab(); CK(cudaDeviceSynchronize());
    k_posv<T, N><<<1, 256>>>(dA, dB);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    md = max_abs_diff_dev(dB, hRef, N);
    if (md >= 1e-3) guard_fail("posv vs CPU chol", md);
    restore_ab();
    k_invsolve<T, N><<<1, 256, sm_inv>>>(dG, dB, dY);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    md = max_abs_diff_dev(dY, hRef, N);
    if (md >= 1e-3) guard_fail("inv+gemv vs CPU chol", md);
    restore_g(); CK(cudaDeviceSynchronize());
    if constexpr (N <= 7) {                          // thread-tier guard (problem 0)
        k_thread_posv<T, N><<<(NPROB + 255) / 256, 256>>>(dA, dB, NPROB);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        md = max_abs_diff_dev(dB, hRef, N);
        if (md >= 1e-3) guard_fail("thread::posv vs CPU chol", md);
        restore_ab(); CK(cudaDeviceSynchronize());
    }

    printf("spdsv  N=%-3d | GESV", N);
    double best[3] = {1e30, 1e30, 1e30};
    for (int TB : {32, 256}) {
        double ns = time_restored_ns([&] { k_gesv<T, N><<<NPROB, TB>>>(dA, dB); },
                                     restore_ab, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best[0]) best[0] = ns;
    }
    printf("  | POSV");
    for (int TB : {32, 256}) {
        double ns = time_restored_ns([&] { k_posv<T, N><<<NPROB, TB>>>(dA, dB); },
                                     restore_ab, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best[1]) best[1] = ns;
    }
    printf("  | INVSV");
    for (int TB : {32, 256}) {
        double ns = time_restored_ns([&] { k_invsolve<T, N><<<NPROB, TB, sm_inv>>>(dG, dB, dY); },
                                     restore_g, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best[2]) best[2] = ns;
    }
    double best_thr = 1e30;
    if constexpr (N <= 7) {                          // thread-tier posv (one problem/thread)
        printf("  | THR-POSV");
        for (int TB : {32, 256}) {
            double ns = time_restored_ns(
                [&] { k_thread_posv<T, N><<<(NPROB + TB - 1) / TB, TB>>>(dA, dB, NPROB); },
                restore_ab, reps);
            printf("  tb%d=%.2f", TB, ns);
            if (ns < best_thr) best_thr = ns;
        }
    }
    const char* names[3] = {"GESV", "POSV", "INVSV"};
    int w = 0;
    for (int i = 1; i < 3; i++) if (best[i] < best[w]) w = i;
    if constexpr (N <= 7)
        printf("  || gesv=%.2f  posv=%.2f  invsv=%.2f  thr-posv=%.2f  -> block %s (",
               best[0], best[1], best[2], best_thr, names[w]);
    else
    printf("  || gesv=%.2f  posv=%.2f  invsv=%.2f  -> %s (", best[0], best[1], best[2], names[w]);
    bool first = true;
    for (int i = 0; i < 3; i++) {
        if (i == w) continue;
        printf("%s%s %.2fx", first ? "" : ", ", i == 0 ? "gesv" : i == 1 ? "posv" : "invsv",
               best[i] / best[w]);
        first = false;
    }
    printf(")\n");
    CK(cudaGetLastError());

    cudaFree(dA); cudaFree(dA0); cudaFree(dB); cudaFree(dB0);
    cudaFree(dG); cudaFree(dG0); cudaFree(dY);
    free(M); free(hA); free(hG); free(hB); free(hRef); free(hAcopy);
}

// ─── section C: syev + eig_clamp (timing only) ───────────────────────────────

template<typename T, int N>
static void bench_eig(int reps) {
    const size_t mm = (size_t)N*N;
    rng_seed(90000 + N);
    double* hA = (double*)malloc(mm*sizeof(double));    // symmetric INDEFINITE
    for (int j = 0; j < N; j++)
        for (int i = 0; i <= j; i++) {
            double v = nrand();
            hA[i + j*N] = v; hA[j + (size_t)i*N] = v;
        }
    T* dA = upload_tile<T>(hA, mm, NPROB);              // syev input (const, preserved)
    T* dC  = upload_tile<T>(hA, mm, NPROB);             // eig_clamp works in place
    T* dC0 = upload_tile<T>(hA, mm, NPROB);
    T* dW; CK(cudaMalloc(&dW, (size_t)NPROB*N*sizeof(T)));
    T* dV; CK(cudaMalloc(&dV, (size_t)NPROB*mm*sizeof(T)));

    const size_t sm_syev  = glass::syev_scratch_bytes<T>(N);
    const size_t sm_clamp = glass::eig_clamp_scratch_bytes<T>(N);

    printf("syev      N=%-3d | BLOCK", N);
    double best = 1e30; int tb = 0;
    for (int TB : {32, 256}) {
        double ns = time_restored_ns([&] { k_syev<T, N><<<NPROB, TB, sm_syev>>>(dA, dW, dV); },
                                     [] {}, reps);      // A preserved — no restore needed
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best) { best = ns; tb = TB; }
    }
    printf("  || best tb%d=%.2f\n", tb, best);

    printf("eig_clamp N=%-3d | BLOCK", N);
    best = 1e30; tb = 0;
    auto restore_c = [&] {
        CK(cudaMemcpyAsync(dC, dC0, (size_t)NPROB*mm*sizeof(T), cudaMemcpyDeviceToDevice));
    };
    for (int TB : {32, 256}) {
        double ns = time_restored_ns([&] { k_eig_clamp<T, N><<<NPROB, TB, sm_clamp>>>(dC); },
                                     restore_c, reps);
        printf("  tb%d=%.2f", TB, ns);
        if (ns < best) { best = ns; tb = TB; }
    }
    printf("  || best tb%d=%.2f\n", tb, best);
    CK(cudaGetLastError());

    cudaFree(dA); cudaFree(dC); cudaFree(dC0); cudaFree(dW); cudaFree(dV);
    free(hA);
}

// ─── driver ───────────────────────────────────────────────────────────────────

template<typename T> static void run_all(int reps) {
    printf("# section A: bdsv (direct) vs pcg (block-Jacobi PCG, rel_tol=1e-6 abs_tol=1e-12 max=200, zero start) — IDENTICAL block-tridiag SPD input\n");
    bench_bdsv_pcg<T, 2, 8>(reps);   bench_bdsv_pcg<T, 2, 32>(reps);
    bench_bdsv_pcg<T, 6, 8>(reps);   bench_bdsv_pcg<T, 6, 32>(reps);
    bench_bdsv_pcg<T, 6, 64>(reps);  bench_bdsv_pcg<T, 12, 16>(reps);
    printf("\n# section B: gesv (pivoted LU) vs posv (Cholesky) vs inv+gemv (anti-pattern) — same SPD system, single RHS\n");
    bench_spdsv<T, 4>(reps);  bench_spdsv<T, 8>(reps);  bench_spdsv<T, 16>(reps);
    bench_spdsv<T, 32>(reps); bench_spdsv<T, 64>(reps);
    printf("\n# section C: syev (cyclic Jacobi eigensolver) + eig_clamp (decompose-clamp-reconstruct) — timing only\n");
    bench_eig<T, 4>(reps);  bench_eig<T, 8>(reps);
    bench_eig<T, 16>(reps); bench_eig<T, 32>(reps);
    printf("\n");
}

int main(int argc, char** argv) {
    NPROB    = (argc > 1) ? atoi(argv[1]) : 8192;
    int reps = (argc > 2) ? atoi(argv[2]) : 50;
    if (reps < 1) reps = 1;
    const char* dt = (argc > 3) ? argv[3] : "f32";
    bool f64 = (strcmp(dt, "f64") == 0 || strcmp(dt, "fp64") == 0 || strcmp(dt, "double") == 0);
    printf("# solvers sweep | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n",
           NPROB, reps, f64 ? "f64" : "f32");
    printf("# protocol: in-place solvers run on NPROB independent global-memory copies; "
           "mutated state restored from pristine device copies OUTSIDE the cudaEvent window "
           "(per-launch event timing, min of 3 trials); CPU-checked correctness guard per shape\n");
    if (f64) run_all<double>(reps);
    else     run_all<float>(reps);
    return 0;
}
