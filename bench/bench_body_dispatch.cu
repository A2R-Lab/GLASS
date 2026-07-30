// bench_body_dispatch.cu — the Phase-2 "in-block body" sweep for the bare
// glass::op face.
//
// The bare `glass::op` face carries a BLOCK-SCOPE calling contract: a kernel of
// TB threads calls the op once per block, ALL threads enter, the result is
// valid after return (glass.cuh header; `glass::dispatch_body()` in
// glass-defaults.cuh). This bench answers the dispatch question: inside that
// FIXED contract, which BODY is fastest —
//
//   BLOCKBODY   glass::block::op(...) executed by all TB threads
//               (the current Phase-1 pin);
//   WARPBODY    if (threadIdx.x < 32) { glass::warp::op(...); } __syncthreads();
//               — warp 0 does the work, the other warps wait;
//   THREADBODY  if (threadIdx.x == 0) { glass::thread::op(...); } __syncthreads();
//               — thread 0 runs the serial register body, loading from and
//               storing to the SAME memory the block body uses (no register
//               staging of operands), so memory effects match.
//
// This is NOT the packed-tier ladder (bench_mega_sweep / bench_robotics pack
// one problem per warp/thread): the launch shape here is one problem per
// BLOCK, NPROB blocks, IDENTICAL for all three bodies. The only difference is
// who executes inside the block. Data placement is identical across the three
// bodies for each cell (same global/shared staging), so surrounding costs
// cancel and the delta is the body itself.
//
// Grid: ops dot/gemv/gemm/chol(=potrf)/trsv/posv at N in {4,8,16,32,64}, plus
// eig3 (fixed 3x3) and softmax_n16 (robotics representatives — their block/
// warp/thread forms share one serial core; see src/base/est/svd3.cuh and
// src/base/L1/softmax.cuh). THREADBODY register ceiling (measured ~7, see
// CLAUDE.md): the matrix ops (gemm/chol/trsv/posv) carry a thread body only
// for N <= 8, dot/gemv up to N = 32 — template-guarded instantiation, so
// unsupported cells are simply absent from that body's column (mirrors
// bench_mega_sweep's missing thread cells). TB (the caller's kernel width) in
// {32, 64, 128, 256}, one column each.
//
// Correctness gate BEFORE timing: every (op, N, body, TB) cell runs once and
// the bodies' outputs are compared pairwise (rel/abs hybrid tolerance 1e-5 f32
// / 1e-12 f64; bit-identity is NOT expected across bodies — different
// summation orders and FMA contraction). A body whose result mismatches or
// whose launch fails prints FAIL for that cell and is excluded from the winner
// argmin (the bench_mega_sweep lesson: a failed launch times as ~ns and would
// otherwise poison the argmin).
//
// Metric: ns per problem (wall / (reps*NPROB)), min of 3 trials, FAIL-guarded.
// Timing-only: outputs are overwritten across reps (in-place ops refactor
// their own output) — uniform across all three bodies, apples-to-apples.
//
// Output grammar (parseable; mirrors the robotics sweep):
//   <op>_n<N> | BLOCKBODY tb32=.. tb64=.. tb128=.. tb256=.. | WARPBODY .. \
//             | THREADBODY .. || -> <WINNERBODY>
// The wrapper script adds `== NPROB=... ==` section headers; this binary
// prints one sweep for its (nprob, reps, dtype) args.
//
// PROTOCOL: timed runs ONLY on a quiet GPU (verify EMPTY via nvidia-smi
// first), serially, no concurrent CPU work. Compile/smoke anytime. SIMT
// bodies only — no MathDx: the nvidia block body is geometry-constrained
// (descriptor-pinned thread counts) and deliberately out of scope for body
// dispatch.
//   nvcc -std=c++17 -arch=<sm> -O3 -I.. -I../src bench_body_dispatch.cu \
//        -o build/bench_body_dispatch_<sm>
// Usage: ./bench_body_dispatch [nprob=8192] [reps=250] [dtype=f32|f64]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#include "../glass.cuh"

static int NPROB = 8192;

enum { BODY_BLOCK = 0, BODY_WARP = 1, BODY_THREAD = 2 };
static const char* BODY_NAMES[3] = {"BLOCKBODY", "WARPBODY", "THREADBODY"};
static const int TBS[4] = {32, 64, 128, 256};

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e3 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
}

template <typename F>
static double time_ns_per_prob(F launch, int reps) {
    cudaGetLastError();                          // clear any sticky prior error
    launch(); cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 1e30;   // FAIL guard
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

// ─── thread-body support gates ───────────────────────────────────────────────
// Measured register ceiling ~7 for the register-resident tier (CLAUDE.md); the
// if-constexpr guards below keep unsupported glass::thread:: instantiations out
// of the binary and thread_body_present() keeps those cells out of the column.
template <int N> constexpr bool T_VEC = (N <= 32);   // dot / gemv thread bodies
template <int N> constexpr bool T_MAT = (N <= 8);    // gemm / chol / trsv / posv

// ─── kernels: one problem per BLOCK, BODY selects who executes inside ────────
// All bodies operate directly on the per-problem global slices — identical data
// placement, so the measured delta is the body, not the staging.

template <typename T, int N, int BODY>
__global__ void k_dot(T* X, T* Y, T* out) {
    int p = blockIdx.x;
    T* x = X + (size_t)p * N;
    T* y = Y + (size_t)p * N;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::dot<T, N>(x, y);           // destructive halving tree; result in y[0]
        if (threadIdx.x == 0) out[p] = y[0];
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) {
            T r = glass::warp::dot<T, N>(x, y);
            if (threadIdx.x == 0) out[p] = r;
        }
        __syncthreads();
    } else if constexpr (T_VEC<N>) {
        if (threadIdx.x == 0) out[p] = glass::thread::dot<T, N>(x, y);
        __syncthreads();
    }
}

template <typename T, int N, int BODY>
__global__ void k_gemv(T* A, T* x, T* y) {
    int p = blockIdx.x;
    T* a  = A + (size_t)p * N * N;
    T* xv = x + (size_t)p * N;
    T* yv = y + (size_t)p * N;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::gemv<T, N, N>((T)1, a, xv, (T)0, yv);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::gemv<T, N, N>((T)1, a, xv, (T)0, yv);
        __syncthreads();
    } else if constexpr (T_VEC<N>) {
        if (threadIdx.x == 0) glass::thread::gemv<T, N, N>((T)1, a, xv, yv);
        __syncthreads();
    }
}

template <typename T, int N, int BODY>
__global__ void k_gemm(T* A, T* B, T* C) {
    int p = blockIdx.x;
    T* a = A + (size_t)p * N * N;
    T* b = B + (size_t)p * N * N;
    T* c = C + (size_t)p * N * N;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::gemm<T, N, N, N>((T)1, a, b, (T)0, c);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::gemm<T, N, N, N>((T)1, a, b, (T)0, c);
        __syncthreads();
    } else if constexpr (T_MAT<N>) {
        if (threadIdx.x == 0) glass::thread::gemm<T, N, N, N>((T)1, a, b, c);
        __syncthreads();
    }
}

template <typename T, int N, int BODY>
__global__ void k_chol(T* A) {
    int p = blockIdx.x;
    T* a = A + (size_t)p * N * N;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::potrf<T, N>(a);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::potrf<T, N>(a);
        __syncthreads();
    } else if constexpr (T_MAT<N>) {
        if (threadIdx.x == 0) glass::thread::potrf<T, N>(a);
        __syncthreads();
    }
}

template <typename T, int N, int BODY>
__global__ void k_trsv(T* A, T* x) {
    int p = blockIdx.x;
    T* a  = A + (size_t)p * N * N;
    T* xv = x + (size_t)p * N;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::trsv<T, N>(a, xv);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::trsv<T, N>(a, xv);
        __syncthreads();
    } else if constexpr (T_MAT<N>) {
        if (threadIdx.x == 0) glass::thread::trsv<T, N>(a, xv);
        __syncthreads();
    }
}

template <typename T, int N, int BODY>
__global__ void k_posv(T* A, T* b) {
    int p = blockIdx.x;
    T* a  = A + (size_t)p * N * N;
    T* bv = b + (size_t)p * N;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::posv<T, N>(a, bv);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::posv<T, N>(a, bv);
        __syncthreads();
    } else if constexpr (T_MAT<N>) {
        if (threadIdx.x == 0) glass::thread::posv<T, N>(a, bv);
        __syncthreads();
    }
}

// eig3: symmetric 3x3 in (9 col-major), out = [W(3) | V(9)] per problem.
template <typename T, int BODY>
__global__ void k_eig3(const T* S, T* out) {
    int p = blockIdx.x;
    const T* a = S + (size_t)p * 9;
    T* o = out + (size_t)p * 12;
    if constexpr (BODY == BODY_BLOCK) {
        glass::block::eig3<T>(a, o, o + 3);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::eig3<T>(a, o, o + 3);
        __syncthreads();
    } else {
        if (threadIdx.x == 0) glass::thread::eig3<T>(a, o, o + 3);
        __syncthreads();
    }
}

// softmax, fixed n=16, MPPI-style alpha=-0.75. Block form needs an n-element
// shared scratch (softmax_scratch_bytes); warp/thread forms are scratch-free —
// intrinsic to the bodies, the operand staging (global x -> global y) is
// identical.
#define SM_N 16
template <typename T, int BODY>
__global__ void k_softmax(const T* X, T* Y) {
    int p = blockIdx.x;
    const T* x = X + (size_t)p * SM_N;
    T* y = Y + (size_t)p * SM_N;
    if constexpr (BODY == BODY_BLOCK) {
        __shared__ T scr[SM_N];
        glass::block::softmax<T>(SM_N, (T)-0.75, x, y, scr);
    } else if constexpr (BODY == BODY_WARP) {
        if (threadIdx.x < 32) glass::warp::softmax<T>(SM_N, (T)-0.75, x, y);
        __syncthreads();
    } else {
        if (threadIdx.x == 0) glass::thread::softmax<T>(SM_N, (T)-0.75, x, y);
        __syncthreads();
    }
}

// ─── correctness gate ────────────────────────────────────────────────────────

template <typename T>
static bool buffers_match(const T* a, const T* b, size_t n) {
    const double tol = (sizeof(T) == 8) ? 1e-12 : 1e-5;
    for (size_t i = 0; i < n; i++) {
        double u = (double)a[i], v = (double)b[i];
        if (u != u || v != v) return false;      // NaN anywhere = genuine failure
        double d = fabs(u - v);
        double m = fmax(1.0, fmax(fabs(u), fabs(v)));
        if (d > tol * m) return false;
    }
    return true;
}

// One output row: correctness-gate every (body, TB) cell (single run, pairwise
// compare — a body passes iff it launched cleanly AND agrees with at least one
// other launched body; a 2-body disagreement fails both, no referee), then time
// the passing cells and print per-body tbXX columns with a winner verdict.
template <typename T, typename LaunchF, typename RestoreF>
static void run_row(const char* name, bool has_thread, LaunchF launch, RestoreF restore,
                    const T* d_out, size_t out_elems, int reps)
{
    const bool present[3] = {true, true, has_thread};
    bool pass[3][4];
    T* h[3];
    for (int b = 0; b < 3; b++) h[b] = (T*)malloc(out_elems * sizeof(T));

    for (int t = 0; t < 4; t++) {
        bool ok[3] = {false, false, false};
        for (int b = 0; b < 3; b++) {
            pass[b][t] = false;
            if (!present[b]) continue;
            restore();
            cudaGetLastError();                  // clear any sticky prior error
            launch(b, TBS[t]);
            ok[b] = (cudaDeviceSynchronize() == cudaSuccess &&
                     cudaGetLastError() == cudaSuccess);
            if (ok[b]) cudaMemcpy(h[b], d_out, out_elems * sizeof(T), cudaMemcpyDeviceToHost);
        }
        int nok = (int)ok[0] + (int)ok[1] + (int)ok[2];
        for (int b = 0; b < 3; b++) {
            if (!ok[b]) continue;
            if (nok == 1) { pass[b][t] = true; continue; }   // sole survivor, unopposed
            for (int o = 0; o < 3; o++)
                if (o != b && ok[o] && buffers_match(h[b], h[o], out_elems)) {
                    pass[b][t] = true; break;
                }
        }
    }
    for (int b = 0; b < 3; b++) free(h[b]);

    printf("%-12s", name);
    double best[3] = {1e30, 1e30, 1e30};
    for (int b = 0; b < 3; b++) {
        if (!present[b]) continue;
        printf("| %s", BODY_NAMES[b]);
        for (int t = 0; t < 4; t++) {
            double ns = 1e30;
            if (pass[b][t]) {
                restore();                       // first timed rep sees valid inputs
                ns = time_ns_per_prob([&]{ launch(b, TBS[t]); }, reps);
            }
            if (ns < 1e29) printf(" tb%d=%.4f", TBS[t], ns);
            else           printf(" tb%d=FAIL", TBS[t]);
            if (ns < best[b]) best[b] = ns;
        }
        printf(" ");
    }
    int win = -1; double wns = 1e30;
    for (int b = 0; b < 3; b++)
        if (best[b] < wns) { wns = best[b]; win = b; }
    printf("|| -> %s\n", win >= 0 ? BODY_NAMES[win] : "NONE");
    fflush(stdout);
}

// ─── rows ────────────────────────────────────────────────────────────────────

enum Op { DOT, GEMV, GEMM, CHOL, TRSV, POSV };
static const char* OP_NAMES[6] = {"dot", "gemv", "gemm", "chol", "trsv", "posv"};

static bool thread_body_present(Op op, int N) {
    return (op == DOT || op == GEMV) ? (N <= 32) : (N <= 8);
}

template <typename T, int N>
static void bench_blas_row(Op op, int reps)
{
    size_t mm = (size_t)NPROB * N * N, vv = (size_t)NPROB * N;
    T *A, *A0, *B, *C, *x, *x0, *y, *y0, *out;
    cudaMalloc(&A, mm * sizeof(T));  cudaMalloc(&A0, mm * sizeof(T));
    cudaMalloc(&B, mm * sizeof(T));  cudaMalloc(&C, mm * sizeof(T));
    cudaMalloc(&x, vv * sizeof(T));  cudaMalloc(&x0, vv * sizeof(T));
    cudaMalloc(&y, vv * sizeof(T));  cudaMalloc(&y0, vv * sizeof(T));
    cudaMalloc(&out, (size_t)NPROB * sizeof(T));

    // A: diagonally-dominant nonneg tile (valid chol/trsv/posv input), broadcast
    // to all problems (mega-sweep precedent). Vectors / B: POSITIVE per-problem
    // patterns — no cancellation, so cross-body summation-order differences stay
    // well inside the gate tolerance.
    {
        T* hA = (T*)malloc(mm * sizeof(T));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                hA[i + (size_t)j * N] = (i == j) ? (T)(N + 2) : (T)(0.1 * ((i + 2*j) % 5));
        for (size_t p = 1; p < (size_t)NPROB; p++)
            memcpy(hA + p * N * N, hA, (size_t)N * N * sizeof(T));
        cudaMemcpy(A0, hA, mm * sizeof(T), cudaMemcpyHostToDevice);
        for (size_t p = 0; p < (size_t)NPROB; p++)
            for (int i = 0; i < N * N; i++)
                hA[p * N * N + i] = (T)(0.25 + 0.2 * cos(0.13 * (double)p + 0.7 * i));
        cudaMemcpy(B, hA, mm * sizeof(T), cudaMemcpyHostToDevice);
        for (size_t p = 0; p < (size_t)NPROB; p++)
            for (int i = 0; i < N; i++)
                hA[p * N + i] = (T)(0.25 + 0.2 * sin(0.31 * (double)p + 1.3 * i));
        cudaMemcpy(x0, hA, vv * sizeof(T), cudaMemcpyHostToDevice);
        for (size_t p = 0; p < (size_t)NPROB; p++)
            for (int i = 0; i < N; i++)
                hA[p * N + i] = (T)(0.25 + 0.2 * sin(0.17 * (double)p + 0.9 * i + 1.0));
        cudaMemcpy(y0, hA, vv * sizeof(T), cudaMemcpyHostToDevice);
        free(hA);
    }
    cudaMemcpy(A, A0, mm * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(x, x0, vv * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(y, y0, vv * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemset(C, 0, mm * sizeof(T));

    // Restore the buffers the op mutates in place (gemv/gemm fully overwrite
    // their outputs, nothing to restore).
    auto restore = [&]() {
        switch (op) {
            case DOT:  cudaMemcpy(y, y0, vv * sizeof(T), cudaMemcpyDeviceToDevice); break;
            case CHOL: cudaMemcpy(A, A0, mm * sizeof(T), cudaMemcpyDeviceToDevice); break;
            case TRSV: cudaMemcpy(x, x0, vv * sizeof(T), cudaMemcpyDeviceToDevice); break;
            case POSV: cudaMemcpy(A, A0, mm * sizeof(T), cudaMemcpyDeviceToDevice);
                       cudaMemcpy(x, x0, vv * sizeof(T), cudaMemcpyDeviceToDevice); break;
            default: break;
        }
    };
    auto launch = [&](int body, int tb) {
        dim3 g(NPROB), blk(tb);
        #define BD_LAUNCH(K, ...) do { \
            if      (body == BODY_BLOCK) K<T, N, BODY_BLOCK ><<<g, blk>>>(__VA_ARGS__); \
            else if (body == BODY_WARP)  K<T, N, BODY_WARP  ><<<g, blk>>>(__VA_ARGS__); \
            else                         K<T, N, BODY_THREAD><<<g, blk>>>(__VA_ARGS__); \
        } while (0)
        switch (op) {
            case DOT:  BD_LAUNCH(k_dot,  x, y, out); break;
            case GEMV: BD_LAUNCH(k_gemv, A, x, y);   break;
            case GEMM: BD_LAUNCH(k_gemm, A, B, C);   break;
            case CHOL: BD_LAUNCH(k_chol, A);         break;
            case TRSV: BD_LAUNCH(k_trsv, A, x);      break;
            case POSV: BD_LAUNCH(k_posv, A, x);      break;
        }
        #undef BD_LAUNCH
    };
    const T* cmp = out; size_t nc = (size_t)NPROB;   // buffer the gate compares
    switch (op) {
        case DOT:  cmp = out; nc = (size_t)NPROB; break;
        case GEMV: cmp = y;   nc = vv; break;
        case GEMM: cmp = C;   nc = mm; break;
        case CHOL: cmp = A;   nc = mm; break;
        case TRSV: cmp = x;   nc = vv; break;
        case POSV: cmp = x;   nc = vv; break;
    }
    char name[32];
    snprintf(name, sizeof(name), "%s_n%d", OP_NAMES[op], N);
    run_row<T>(name, thread_body_present(op, N), launch, restore, cmp, nc, reps);

    cudaFree(A); cudaFree(A0); cudaFree(B); cudaFree(C);
    cudaFree(x); cudaFree(x0); cudaFree(y); cudaFree(y0); cudaFree(out);
}

template <typename T>
static void bench_eig3_row(int reps)
{
    size_t ni = (size_t)NPROB * 9, no = (size_t)NPROB * 12;
    T *S, *out;
    cudaMalloc(&S, ni * sizeof(T));
    cudaMalloc(&out, no * sizeof(T));
    // Symmetric 3x3 with well-separated eigenvalues (~2/3/4 + small off-diag):
    // stable eigenvectors, so the cross-body compare sits inside the tolerance.
    T* h = (T*)malloc(ni * sizeof(T));
    for (size_t p = 0; p < (size_t)NPROB; p++) {
        T* s = h + p * 9;
        double o01 = 0.3 * sin(0.7 * (double)p);
        double o02 = 0.2 * cos(1.3 * (double)p);
        double o12 = 0.1 * sin(2.1 * (double)p);
        s[0] = (T)(2.0 + 0.1 * sin(0.3 * (double)p));
        s[4] = (T)(3.0 + 0.1 * cos(0.5 * (double)p));
        s[8] = (T)(4.0 + 0.1 * sin(0.9 * (double)p));
        s[1] = s[3] = (T)o01;
        s[2] = s[6] = (T)o02;
        s[5] = s[7] = (T)o12;
    }
    cudaMemcpy(S, h, ni * sizeof(T), cudaMemcpyHostToDevice);
    free(h);
    auto restore = []() {};                       // non-destructive
    auto launch = [&](int body, int tb) {
        dim3 g(NPROB), blk(tb);
        if      (body == BODY_BLOCK) k_eig3<T, BODY_BLOCK ><<<g, blk>>>(S, out);
        else if (body == BODY_WARP)  k_eig3<T, BODY_WARP  ><<<g, blk>>>(S, out);
        else                         k_eig3<T, BODY_THREAD><<<g, blk>>>(S, out);
    };
    run_row<T>("eig3_n3", true, launch, restore, (const T*)out, no, reps);
    cudaFree(S); cudaFree(out);
}

template <typename T>
static void bench_softmax_row(int reps)
{
    size_t nn = (size_t)NPROB * SM_N;
    T *X, *Y;
    cudaMalloc(&X, nn * sizeof(T));
    cudaMalloc(&Y, nn * sizeof(T));
    T* h = (T*)malloc(nn * sizeof(T));
    for (size_t p = 0; p < (size_t)NPROB; p++)
        for (int i = 0; i < SM_N; i++)
            h[p * SM_N + i] = (T)(0.9 * sin(0.7 * (double)p + 0.4 * i));
    cudaMemcpy(X, h, nn * sizeof(T), cudaMemcpyHostToDevice);
    free(h);
    auto restore = []() {};                       // y separate from x, fully overwritten
    auto launch = [&](int body, int tb) {
        dim3 g(NPROB), blk(tb);
        if      (body == BODY_BLOCK) k_softmax<T, BODY_BLOCK ><<<g, blk>>>(X, Y);
        else if (body == BODY_WARP)  k_softmax<T, BODY_WARP  ><<<g, blk>>>(X, Y);
        else                         k_softmax<T, BODY_THREAD><<<g, blk>>>(X, Y);
    };
    run_row<T>("softmax_n16", true, launch, restore, (const T*)Y, nn, reps);
    cudaFree(X); cudaFree(Y);
}

template <typename T>
static void bench_all(int reps)
{
    for (Op op : {DOT, GEMV, GEMM, CHOL, TRSV, POSV}) {
        bench_blas_row<T, 4 >(op, reps);
        bench_blas_row<T, 8 >(op, reps);
        bench_blas_row<T, 16>(op, reps);
        bench_blas_row<T, 32>(op, reps);
        bench_blas_row<T, 64>(op, reps);
    }
    bench_eig3_row<T>(reps);
    bench_softmax_row<T>(reps);
}

int main(int argc, char** argv)
{
    NPROB    = (argc > 1) ? atoi(argv[1]) : 8192;
    int reps = (argc > 2) ? atoi(argv[2]) : 250;
    const char* dt = (argc > 3) ? argv[3] : "f32";
    bool f64 = (strcmp(dt, "f64") == 0);
    printf("# body-dispatch sweep | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n",
           NPROB, reps, f64 ? "f64" : "f32");
    printf("# fixed block-scope contract: one problem per BLOCK; bodies = block SIMT / warp0 / thread0.\n");
    printf("# TIMED RUNS REQUIRE A QUIET GPU (see file header).\n");
    if (f64) bench_all<double>(reps);
    else     bench_all<float>(reps);
    return 0;
}
