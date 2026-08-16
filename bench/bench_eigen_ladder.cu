// bench_eigen_ladder.cu — Eigen-in-kernel baseline for the paper's
// "other device-side alternatives" comparison (BENCH-ONLY; never a library dep).
//
// QUESTION ANSWERED: "why not just use Eigen inside kernels?" Eigen's fixed-size
// matrices compile in device code but execute strictly THREAD-SERIALLY — one
// thread evaluates the whole expression (Eigen's own CUDA doc). So the honest
// comparison is Eigen-per-thread vs glass::thread:: per-thread at the SAME
// launch shapes with the SAME operand staging: the delta isolates the math
// library, everything else held equal. glass::block:: rows are timed alongside
// as the ladder anchor (what a caller actually ships at larger N).
//
// SCOPE / DISCLOSURE (state in the paper wherever these numbers appear):
//   * ops dot/gemv/gemm ONLY. Eigen device code has NO cooperative execution
//     and (per its docs) essentially NO device-side decompositions (only the
//     2x2/3x3 direct self-adjoint eigensolver) — the ABSENCE of potrf/trsv/posv
//     columns is itself the finding, not a harness limitation.
//   * both thread contenders stage operands global -> registers -> global in
//     the per-problem-contiguous layout (uncoalesced; the layout tax is IN the
//     timing) — identical loops, mirroring bench_mega_sweep's THREAD model.
//   * N capped at 32: per-thread N*N staging beyond that is the documented
//     spill regime for ANY per-thread library (mega sweep covers it); the
//     robot band is N<=32. The cap is printed, never silent.
//   * host Eigen SIMD is disabled in this TU (EIGEN_DONT_VECTORIZE) per
//     Eigen's CUDA guidance; device codegen is unaffected.
//
// CORRECTNESS CROSS-CHECK BEFORE TIMING (bench_common pattern): every
// (op, N, dtype) contender is run once on pristine inputs and compared against
// a double-precision host reference before any timed trial; a failing check
// prints CHECK=FAIL and poisons the row with the 1e30 sentinel.
//
// Compile (Eigen is header-only; apt: libeigen3-dev, or set EIGEN_ROOT):
//   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -I.. -I../src \
//        -I/usr/include/eigen3 -DEIGEN_DONT_VECTORIZE -DEIGEN_NO_DEBUG \
//        -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int \
//        bench_eigen_ladder.cu -o bench_eigen_ladder
// Usage: ./bench_eigen_ladder [reps=500] [dtype=f32|f64|both]
//   (sweeps NPROB over {64, 1024, 8192} internally — the ladder's three
//    concurrency regimes — so one run is a complete capture.)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include "timing_common.cuh"

#if !__has_include(<Eigen/Dense>)
#error "Eigen not found. Install libeigen3-dev or add -I$EIGEN_ROOT (bench-only dependency)."
#endif
#include <Eigen/Dense>

#include "../glass.cuh"

static int NPROB = 8192;
static std::function<void()> g_pre_trial;

// ─── EIGEN thread model: thread p owns problem p; staged like kt_* ───────────
template<typename T,int N> __global__ void ke_dot(const T* x, const T* y, T* r, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= np) return;
    Eigen::Matrix<T,N,1> xv, yv;
    for (int i = 0; i < N; i++) { xv(i) = x[(size_t)p*N+i]; yv(i) = y[(size_t)p*N+i]; }
    r[p] = xv.dot(yv);
}
template<typename T,int N> __global__ void ke_gemv(const T* A, const T* x, T* y, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= np) return;
    Eigen::Matrix<T,N,N> a; Eigen::Matrix<T,N,1> xv, yv;
    for (int i = 0; i < N*N; i++) a.data()[i] = A[(size_t)p*N*N+i];   // column-major both sides
    for (int i = 0; i < N; i++)   xv(i) = x[(size_t)p*N+i];
    yv.noalias() = a * xv;
    for (int i = 0; i < N; i++)   y[(size_t)p*N+i] = yv(i);
}
template<typename T,int N> __global__ void ke_gemm(const T* A, const T* B, T* C, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= np) return;
    Eigen::Matrix<T,N,N> a, b, c;
    for (int i = 0; i < N*N; i++) { a.data()[i] = A[(size_t)p*N*N+i]; b.data()[i] = B[(size_t)p*N*N+i]; }
    c.noalias() = a * b;
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N+i] = c.data()[i];
}

// ─── GLASS thread model: identical staging, glass::thread:: arithmetic ───────
template<typename T,int N> __global__ void kg_dot(const T* x, const T* y, T* r, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= np) return;
    T xv[N], yv[N];
    for (int i = 0; i < N; i++) { xv[i] = x[(size_t)p*N+i]; yv[i] = y[(size_t)p*N+i]; }
    r[p] = glass::thread::dot<T,N>(xv, yv);
}
template<typename T,int N> __global__ void kg_gemv(const T* A, const T* x, T* y, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= np) return;
    T a[N*N], xv[N], yv[N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N+i];
    for (int i = 0; i < N; i++)   xv[i] = x[(size_t)p*N+i];
    glass::thread::gemv<T,N,N>((T)1, a, xv, yv);
    for (int i = 0; i < N; i++)   y[(size_t)p*N+i] = yv[i];
}
template<typename T,int N> __global__ void kg_gemm(const T* A, const T* B, T* C, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= np) return;
    T a[N*N], b[N*N], c[N*N];
    for (int i = 0; i < N*N; i++) { a[i] = A[(size_t)p*N*N+i]; b[i] = B[(size_t)p*N*N+i]; }
    glass::thread::gemm<T,N,N,N>((T)1, a, b, c);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N+i] = c[i];
}

// ─── GLASS block anchor: block p owns problem p (pure-SIMT ladder entry) ─────
// block::dot is in-place: result lands in y[p*N] (y is scratch); thread 0
// publishes it to r[p] so the correctness check reads one location for all
// contenders. y drifts — restored by the per-trial hook, same as the others.
template<typename T,int N> __global__ void kb_dot(T* x, T* y, T* r) {
    int p = blockIdx.x;
    glass::block::dot<T,N>(x+(size_t)p*N, y+(size_t)p*N);
    if (threadIdx.x == 0) r[p] = y[(size_t)p*N];
}
template<typename T,int N> __global__ void kb_gemv(T* A, T* x, T* y) {
    int p = blockIdx.x;
    glass::block::gemv<T,N,N>((T)1, A+(size_t)p*N*N, x+(size_t)p*N, (T)0, y+(size_t)p*N);
}
template<typename T,int N> __global__ void kb_gemm(T* A, T* B, T* C) {
    int p = blockIdx.x;
    glass::block::gemm<T,N,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N);
}

enum Op { DOT, GEMV, GEMM };
static const char* op_name(Op o) { const char* n[] = {"dot","gemv","gemm"}; return n[o]; }

// Host double-precision reference over the first `k` problems.
template<typename T>
static double ref_maxerr(Op op, int N, int k, const T* hA, const T* hB, const T* hx,
                         const T* hy_in, const T* out_r, const T* out_y, const T* out_C) {
    double maxerr = 0.0;
    for (int p = 0; p < k; p++) {
        if (op == DOT) {
            double acc = 0.0;
            for (int i = 0; i < N; i++) acc += (double)hx[(size_t)p*N+i] * (double)hy_in[(size_t)p*N+i];
            double got = (double)out_r[p];
            double err = std::fabs(got - acc) / (std::fabs(acc) + 1e-30);
            if (err > maxerr) maxerr = err;
        } else if (op == GEMV) {
            for (int i = 0; i < N; i++) {
                double acc = 0.0;
                for (int j = 0; j < N; j++) acc += (double)hA[(size_t)p*N*N + i + j*N] * (double)hx[(size_t)p*N+j];
                double got = (double)out_y[(size_t)p*N+i];
                double err = std::fabs(got - acc) / (std::fabs(acc) + 1e-30);
                if (err > maxerr) maxerr = err;
            }
        } else {
            for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
                double acc = 0.0;
                for (int l = 0; l < N; l++) acc += (double)hA[(size_t)p*N*N + i + l*N] * (double)hB[(size_t)p*N*N + l + j*N];
                double got = (double)out_C[(size_t)p*N*N + i + j*N];
                double err = std::fabs(got - acc) / (std::fabs(acc) + 1e-30);
                if (err > maxerr) maxerr = err;
            }
        }
    }
    return maxerr;
}

template<typename T,int N>
static void bench_size(Op op, int reps, const char* dt) {
    size_t mm = (size_t)NPROB*N*N, vv = (size_t)NPROB*N;
    T *A, *B, *C, *x, *y, *r, *x0, *y0;
    cudaMalloc(&A, mm*sizeof(T)); cudaMalloc(&B, mm*sizeof(T)); cudaMalloc(&C, mm*sizeof(T));
    cudaMalloc(&x, vv*sizeof(T)); cudaMalloc(&y, vv*sizeof(T)); cudaMalloc(&r, NPROB*sizeof(T));
    cudaMalloc(&x0, vv*sizeof(T)); cudaMalloc(&y0, vv*sizeof(T));

    // Real-magnitude host init (NOT memset byte patterns — those are denormals
    // in fp and would make the relative-error check meaningless).
    T* hA = (T*)malloc(mm*sizeof(T)); T* hB = (T*)malloc(mm*sizeof(T));
    T* hx = (T*)malloc(vv*sizeof(T)); T* hy = (T*)malloc(vv*sizeof(T));
    for (size_t i = 0; i < mm; i++) { hA[i] = (T)(0.25 + (double)(i % 11) * 0.125); hB[i] = (T)(0.5 - (double)(i % 7) * 0.0625); }
    for (size_t i = 0; i < vv; i++) { hx[i] = (T)(0.5 + (double)(i % 5) * 0.25); hy[i] = (T)(1.0 - (double)(i % 3) * 0.125); }
    cudaMemcpy(A, hA, mm*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, mm*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(x0, hx, vv*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(y0, hy, vv*sizeof(T), cudaMemcpyHostToDevice);
    g_pre_trial = [=]{   // dot/gemv write y (and r); restore per TRIAL, untimed
        cudaMemcpy(x, x0, vv*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(y, y0, vv*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    };

    // ── correctness cross-check BEFORE timing (each contender, pristine inputs)
    const int kcheck = NPROB < 64 ? NPROB : 64;
    const double tol = (sizeof(T) == 4) ? 1e-4 * N : 1e-12 * N;
    T* out_r = (T*)malloc(NPROB*sizeof(T)); T* out_y = (T*)malloc(vv*sizeof(T));
    T* out_C = (T*)malloc(mm*sizeof(T));
    bool ok[3] = {true, true, true};   // eigen, gthread, gblock
    const char* impls[3] = {"eigen_thread", "glass_thread", "glass_block"};
    for (int c = 0; c < 3; c++) {
        g_pre_trial(); cudaMemset(C, 0, mm*sizeof(T)); cudaMemset(r, 0, NPROB*sizeof(T));
        cudaGetLastError();
        int TPB = 128;
        if (c == 0) {
            if (op == DOT)  ke_dot <T,N><<<(NPROB+TPB-1)/TPB,TPB>>>(x, y, r, NPROB);
            if (op == GEMV) ke_gemv<T,N><<<(NPROB+TPB-1)/TPB,TPB>>>(A, x, y, NPROB);
            if (op == GEMM) ke_gemm<T,N><<<(NPROB+TPB-1)/TPB,TPB>>>(A, B, C, NPROB);
        } else if (c == 1) {
            if (op == DOT)  kg_dot <T,N><<<(NPROB+TPB-1)/TPB,TPB>>>(x, y, r, NPROB);
            if (op == GEMV) kg_gemv<T,N><<<(NPROB+TPB-1)/TPB,TPB>>>(A, x, y, NPROB);
            if (op == GEMM) kg_gemm<T,N><<<(NPROB+TPB-1)/TPB,TPB>>>(A, B, C, NPROB);
        } else {
            if (op == DOT)  kb_dot <T,N><<<NPROB,128>>>(x, y, r);
            if (op == GEMV) kb_gemv<T,N><<<NPROB,128>>>(A, x, y);
            if (op == GEMM) kb_gemm<T,N><<<NPROB,128>>>(A, B, C);
        }
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { ok[c] = false; printf("# CHECK op=%s N=%d %s LAUNCH-FAIL\n", op_name(op), N, impls[c]); continue; }
        cudaMemcpy(out_r, r, NPROB*sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_y, y, vv*sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_C, C, mm*sizeof(T), cudaMemcpyDeviceToHost);
        double err = ref_maxerr<T>(op, N, kcheck, hA, hB, hx, hy, out_r, out_y, out_C);
        ok[c] = std::isfinite(err) && err <= tol;
        printf("# CHECK op=%s N=%d %s maxerr=%.3e tol=%.1e %s\n",
               op_name(op), N, impls[c], err, tol, ok[c] ? "OK" : "FAIL");
    }
    free(out_r); free(out_y); free(out_C);

    // ── timing: TPB/TB swept per contender, min-of-3 trials, per-trial reset
    auto time_thread = [&](int c, int TPB) -> double {
        auto launch = [&]{
            dim3 g((NPROB+TPB-1)/TPB), b(TPB);
            if (c == 0) {
                if (op == DOT)  ke_dot <T,N><<<g,b>>>(x, y, r, NPROB);
                if (op == GEMV) ke_gemv<T,N><<<g,b>>>(A, x, y, NPROB);
                if (op == GEMM) ke_gemm<T,N><<<g,b>>>(A, B, C, NPROB);
            } else {
                if (op == DOT)  kg_dot <T,N><<<g,b>>>(x, y, r, NPROB);
                if (op == GEMV) kg_gemv<T,N><<<g,b>>>(A, x, y, NPROB);
                if (op == GEMM) kg_gemm<T,N><<<g,b>>>(A, B, C, NPROB);
            }
        };
        return tc_time_ns_per_prob_pre(launch, []{ if (g_pre_trial) g_pre_trial(); }, reps, NPROB);
    };
    auto time_block = [&](int TB) -> double {
        auto launch = [&]{
            if (op == DOT)  kb_dot <T,N><<<NPROB,TB>>>(x, y, r);
            if (op == GEMV) kb_gemv<T,N><<<NPROB,TB>>>(A, x, y);
            if (op == GEMM) kb_gemm<T,N><<<NPROB,TB>>>(A, B, C);
        };
        return tc_time_ns_per_prob_pre(launch, []{ if (g_pre_trial) g_pre_trial(); }, reps, NPROB);
    };

    for (int c = 0; c < 2; c++) {   // 0=eigen_thread, 1=glass_thread
        for (int TPB : {32, 64, 128, 256}) {
            double ns = ok[c] ? time_thread(c, TPB) : 1e30;
            if (ns < 1e29)
                printf("RESULT section=eigen op=%s dtype=%s N=%d NPROB=%d impl=%s cfg=t%d ns=%.4f spread=%.2f%%\n",
                       op_name(op), dt, N, NPROB, impls[c], TPB, ns, tc_last_spread_pct());
            else
                printf("RESULT section=eigen op=%s dtype=%s N=%d NPROB=%d impl=%s cfg=t%d ns=FAIL\n",
                       op_name(op), dt, N, NPROB, impls[c], TPB);
        }
    }
    for (int TB : {32, 64, 128, 256}) {
        double ns = ok[2] ? time_block(TB) : 1e30;
        if (ns < 1e29)
            printf("RESULT section=eigen op=%s dtype=%s N=%d NPROB=%d impl=%s cfg=tb%d ns=%.4f spread=%.2f%%\n",
                   op_name(op), dt, N, NPROB, impls[2], TB, ns, tc_last_spread_pct());
        else
            printf("RESULT section=eigen op=%s dtype=%s N=%d NPROB=%d impl=%s cfg=tb%d ns=FAIL\n",
                   op_name(op), dt, N, NPROB, impls[2], TB);
    }
    fflush(stdout);
    g_pre_trial = nullptr;
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(x); cudaFree(y); cudaFree(r);
    cudaFree(x0); cudaFree(y0);
    free(hA); free(hB); free(hx); free(hy);
}

template<typename T> static void run_all(int reps, const char* dt) {
    for (int np : {64, 1024, 8192}) {
        NPROB = np;
        printf("# eigen ladder | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n", NPROB, reps, dt);
        for (Op op : {DOT, GEMV, GEMM}) {
            bench_size<T,4>(op, reps, dt);  bench_size<T,6>(op, reps, dt);
            bench_size<T,8>(op, reps, dt);  bench_size<T,12>(op, reps, dt);
            bench_size<T,16>(op, reps, dt); bench_size<T,24>(op, reps, dt);
            bench_size<T,32>(op, reps, dt);
        }
    }
}

int main(int argc, char** argv) {
    int reps       = (argc > 1) ? atoi(argv[1]) : 500;
    const char* dt = (argc > 2) ? argv[2] : "both";
    printf("# eigen-in-kernel baseline | contenders: EIGEN(thread-serial, staged, TPB swept) | "
           "GLASS-THREAD(identical staging) | GLASS-BLOCK(anchor, TB swept)\n");
    printf("# ops dot/gemv/gemm only — Eigen device code has no cooperative path and no "
           "factorizations (the absence is the finding); N capped at 32 (robot band).\n");
    printf("# eigen_version=%d.%d.%d\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    tc_warm_gpu();
    if (strcmp(dt, "f64") == 0 || strcmp(dt, "both") == 0) run_all<double>(reps, "f64");
    if (strcmp(dt, "f32") == 0 || strcmp(dt, "both") == 0) run_all<float>(reps, "f32");
    return 0;
}
