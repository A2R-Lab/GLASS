// bench_kokkos_ladder.cu — Kokkos Kernels (KokkosBatched) baseline for the
// paper's "other device-side alternatives" comparison (BENCH-ONLY dep).
//
// Kokkos Kernels is the nearest cross-platform analog to GLASS: genuinely
// device-callable, team(≈block)-cooperative batched routines invoked inside a
// user's parallel_for. This harness times its three execution modes at the
// ladder shapes, one problem per team/thread, against glass::block:: and
// glass::warp:: anchors in the SAME translation unit under the SAME timing
// core, data, and reset discipline.
//
// Contenders per (op, N, NPROB, dtype):
//   kk_serial      one problem per THREAD (RangePolicy; Serial*::invoke)
//   kk_team        one problem per TEAM   (TeamPolicy ts∈{32,64,128,256}, vlen=1)
//   kk_teamvector  one problem per TEAM   (TeamPolicy ts∈{1,2,4,8}, vlen=32)
//   glass_block    one problem per BLOCK  (TB swept — raw kernel)
//   glass_warp     one problem per WARP   (WPB swept — raw kernel)
// Kokkos algorithm tags: Unblocked everywhere; gemm additionally sweeps
// Blocked (their register-blocking variant). Their best tag/shape is reported
// per cell — we compare against the BEST Kokkos configuration, disclosed.
//
// SCOPE / DISCLOSURE (state wherever these numbers appear):
//   * ops gemm/gemv/trsv ONLY: the KokkosBatched team-scope set has no
//     Cholesky/posv (LU is unpivoted, no symmetric factorization) — measured
//     where comparable, prose where absent.
//   * N ≤ 64 (compile-time bound on instantiation count; the robot band).
//   * one problem per team with subviews of a problem-contiguous column-major
//     buffer — identical bytes to the glass anchors; Kokkos launches go
//     through Kokkos::parallel_for (its natural dispatch path), glass anchors
//     through raw kernel launches (theirs). Both are enqueue-async and timed
//     by the same reps-then-sync wall-clock core.
//   * correctness: every contender is cross-checked against a host double
//     reference before any timed trial (FAIL poisons the row with 1e30).
//
// Compile (Kokkos + Kokkos Kernels BATCHED installed, e.g. ~/opt):
//   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr --expt-extended-lambda \
//        -I.. -I../src -I$KOKKOS_ROOT/include -I$KOKKOSKERNELS_ROOT/include \
//        bench_kokkos_ladder.cu -o bench_kokkos_ladder \
//        -L$KOKKOS_ROOT/lib -L$KOKKOSKERNELS_ROOT/lib \
//        -lkokkoskernels -lkokkoscore -lkokkoscontainers -lkokkossimd -ldl
// Usage: ./bench_kokkos_ladder [reps=500] [dtype=f32|f64|both]
//   (sweeps NPROB over {64, 1024, 8192} internally.)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include "timing_common.cuh"

#include <Kokkos_Core.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
// KokkosBatched's Team/TeamVector gemv takes a RANK-3 (multi-problem-per-team)
// A view; the single-matrix team-cooperative gemv lives in KokkosBlas (it is
// also what the batched form delegates to at batch extent 1). We use the
// KokkosBlas form so gemv keeps the same one-problem-per-team mapping as
// gemm/trsv.
#include <KokkosBlas2_team_gemv.hpp>
#include <KokkosBlas2_serial_gemv_impl.hpp>   // KokkosBatched::SerialGemv is deprecated (device-aborts)

#include "../glass.cuh"

static int NPROB = 8192;
static std::function<void()> g_pre_trial;

using ExecSpace = Kokkos::Cuda;
using TeamPol   = Kokkos::TeamPolicy<ExecSpace>;
using Member    = TeamPol::member_type;
template<typename T> using UView2 =
    Kokkos::View<T**, Kokkos::LayoutLeft, Kokkos::CudaSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template<typename T> using UView1 =
    Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

namespace KB = KokkosBatched;

enum Op { GEMM, GEMV, TRSV };
static const char* op_name(Op o) { const char* n[] = {"gemm","gemv","trsv"}; return n[o]; }

// ─── Kokkos contenders: MODE 0=Serial-per-thread, 1=Team, 2=TeamVector ───────
// Compile-time op + mode (if constexpr) so only valid (op, algo, mode)
// combinations instantiate — Algo::Gemm is Level3 while Algo::Gemv/Trsv are
// Level2, so the tags are NOT interchangeable across ops. Problem p gets
// column-major unmanaged views over the shared problem-contiguous buffers.
template<typename T, int N, int MODE, typename Algo>
static void kk_gemm(int ts, int vlen, T* A, T* B, T* C) {
    const int np = NPROB;
    if constexpr (MODE == 0) {
        Kokkos::parallel_for("kkg", Kokkos::RangePolicy<ExecSpace>(0, np),
            KOKKOS_LAMBDA(const int p) {
                UView2<T> a(A + (size_t)p*N*N, N, N), b(B + (size_t)p*N*N, N, N), c(C + (size_t)p*N*N, N, N);
                KB::SerialGemm<KB::Trans::NoTranspose, KB::Trans::NoTranspose, Algo>::invoke((T)1, a, b, (T)0, c);
            });
    } else {
        Kokkos::parallel_for("kkg", TeamPol(np, ts, vlen),
            KOKKOS_LAMBDA(const Member& m) {
                const int p = m.league_rank();
                UView2<T> a(A + (size_t)p*N*N, N, N), b(B + (size_t)p*N*N, N, N), c(C + (size_t)p*N*N, N, N);
                if constexpr (MODE == 1)
                    KB::TeamGemm<Member, KB::Trans::NoTranspose, KB::Trans::NoTranspose, Algo>::invoke(m, (T)1, a, b, (T)0, c);
                else
                    KB::TeamVectorGemm<Member, KB::Trans::NoTranspose, KB::Trans::NoTranspose, Algo>::invoke(m, (T)1, a, b, (T)0, c);
            });
    }
}
template<typename T, int N, int MODE, typename Algo>
static void kk_gemv(int ts, int vlen, T* A, T* x, T* y) {
    const int np = NPROB;
    if constexpr (MODE == 0) {
        Kokkos::parallel_for("kkv", Kokkos::RangePolicy<ExecSpace>(0, np),
            KOKKOS_LAMBDA(const int p) {
                UView2<T> a(A + (size_t)p*N*N, N, N);
                UView1<T> xv(x + (size_t)p*N, N), yv(y + (size_t)p*N, N);
                KokkosBlas::SerialGemv<KB::Trans::NoTranspose, Algo>::invoke((T)1, a, xv, (T)0, yv);
            });
    } else {
        Kokkos::parallel_for("kkv", TeamPol(np, ts, vlen),
            KOKKOS_LAMBDA(const Member& m) {
                const int p = m.league_rank();
                UView2<T> a(A + (size_t)p*N*N, N, N);
                UView1<T> xv(x + (size_t)p*N, N), yv(y + (size_t)p*N, N);
                if constexpr (MODE == 1)
                    KokkosBlas::TeamGemv<Member, KB::Trans::NoTranspose, Algo>::invoke(m, (T)1, a, xv, (T)0, yv);
                else
                    KokkosBlas::TeamVectorGemv<Member, KB::Trans::NoTranspose, Algo>::invoke(m, (T)1, a, xv, (T)0, yv);
            });
    }
}
template<typename T, int N, int MODE, typename Algo>
static void kk_trsv(int ts, int vlen, T* A, T* x) {
    const int np = NPROB;
    if constexpr (MODE == 0) {
        Kokkos::parallel_for("kkr", Kokkos::RangePolicy<ExecSpace>(0, np),
            KOKKOS_LAMBDA(const int p) {
                UView2<T> a(A + (size_t)p*N*N, N, N);
                UView1<T> bv(x + (size_t)p*N, N);
                KB::SerialTrsv<KB::Uplo::Lower, KB::Trans::NoTranspose, KB::Diag::NonUnit, Algo>::invoke((T)1, a, bv);
            });
    } else {
        Kokkos::parallel_for("kkr", TeamPol(np, ts, vlen),
            KOKKOS_LAMBDA(const Member& m) {
                const int p = m.league_rank();
                UView2<T> a(A + (size_t)p*N*N, N, N);
                UView1<T> bv(x + (size_t)p*N, N);
                if constexpr (MODE == 1)
                    KB::TeamTrsv<Member, KB::Uplo::Lower, KB::Trans::NoTranspose, KB::Diag::NonUnit, Algo>::invoke(m, (T)1, a, bv);
                else
                    KB::TeamVectorTrsv<Member, KB::Uplo::Lower, KB::Trans::NoTranspose, KB::Diag::NonUnit, Algo>::invoke(m, (T)1, a, bv);
            });
    }
}

// Runtime-op front end over the compile-time kernels (algo: 0=Unblocked,
// 1=Blocked — gemm Serial/Team only; TeamVectorGemm<Blocked> is declared but
// unimplemented upstream, so MODE 2 pins Unblocked at compile time).
template<typename T, int N, int MODE>
static void kk_launch(Op op, int algo, int ts, int vlen, T* A, T* B, T* C, T* x) {
    if (op == GEMM) {
        if constexpr (MODE == 2) {
            (void)algo;
            kk_gemm<T,N,MODE,KB::Algo::Gemm::Unblocked>(ts, vlen, A, B, C);
        } else {
            if (algo) kk_gemm<T,N,MODE,KB::Algo::Gemm::Blocked>(ts, vlen, A, B, C);
            else      kk_gemm<T,N,MODE,KB::Algo::Gemm::Unblocked>(ts, vlen, A, B, C);
        }
    } else if (op == GEMV) {
        kk_gemv<T,N,MODE,KB::Algo::Gemv::Unblocked>(ts, vlen, A, x, C);
    } else {
        kk_trsv<T,N,MODE,KB::Algo::Trsv::Unblocked>(ts, vlen, A, x);
    }
}

// ─── glass anchors (raw launches, same buffers) ──────────────────────────────
template<typename T,int N> __global__ void gb_gemm(T* A, T* B, T* C) {
    int p = blockIdx.x;
    glass::block::gemm<T,N,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N);
}
template<typename T,int N> __global__ void gb_gemv(T* A, T* x, T* y) {
    int p = blockIdx.x;
    glass::block::gemv<T,N,N>((T)1, A+(size_t)p*N*N, x+(size_t)p*N, (T)0, y+(size_t)p*N);
}
template<typename T,int N> __global__ void gb_trsv(T* A, T* x) {
    int p = blockIdx.x;
    glass::block::trsv<T,N>(A+(size_t)p*N*N, x+(size_t)p*N);
}
template<typename T,int N> __global__ void gw_gemm(T* A, T* B, T* C, int np) {
    int p = blockIdx.x*blockDim.y+threadIdx.y; if (p>=np) return;
    glass::warp::gemm<T,N,N,N>((T)1, A+(size_t)p*N*N, B+(size_t)p*N*N, (T)0, C+(size_t)p*N*N);
}
template<typename T,int N> __global__ void gw_gemv(T* A, T* x, T* y, int np) {
    int p = blockIdx.x*blockDim.y+threadIdx.y; if (p>=np) return;
    glass::warp::gemv<T,N,N>((T)1, A+(size_t)p*N*N, x+(size_t)p*N, (T)0, y+(size_t)p*N);
}
template<typename T,int N> __global__ void gw_trsv(T* A, T* x, int np) {
    int p = blockIdx.x*blockDim.y+threadIdx.y; if (p>=np) return;
    glass::warp::trsv<T,N>(A+(size_t)p*N*N, x+(size_t)p*N);
}

// ─── host double reference over the first k problems ─────────────────────────
template<typename T>
static double ref_maxerr(Op op, int N, int k, const T* hA, const T* hB, const T* hx,
                         const T* out_C, const T* out_x) {
    double maxerr = 0.0;
    for (int p = 0; p < k; p++) {
        if (op == GEMM) {
            for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
                double acc = 0.0;
                for (int l = 0; l < N; l++) acc += (double)hA[(size_t)p*N*N + i + l*N] * (double)hB[(size_t)p*N*N + l + j*N];
                double e = std::fabs((double)out_C[(size_t)p*N*N + i + j*N] - acc) / (std::fabs(acc) + 1e-30);
                if (e > maxerr) maxerr = e;
            }
        } else if (op == GEMV) {
            for (int i = 0; i < N; i++) {
                double acc = 0.0;
                for (int j = 0; j < N; j++) acc += (double)hA[(size_t)p*N*N + i + j*N] * (double)hx[(size_t)p*N+j];
                double e = std::fabs((double)out_C[(size_t)p*N+i] - acc) / (std::fabs(acc) + 1e-30);
                if (e > maxerr) maxerr = e;
            }
        } else {   // forward substitution: L y = x, L = lower(A) incl. diag
            double y[64];
            for (int i = 0; i < N; i++) {
                double acc = (double)hx[(size_t)p*N+i];
                for (int j = 0; j < i; j++) acc -= (double)hA[(size_t)p*N*N + i + j*N] * y[j];
                y[i] = acc / (double)hA[(size_t)p*N*N + i + i*N];
                double e = std::fabs((double)out_x[(size_t)p*N+i] - y[i]) / (std::fabs(y[i]) + 1e-30);
                if (e > maxerr) maxerr = e;
            }
        }
    }
    return maxerr;
}

// contender ids: 0 kk_serial 1 kk_team 2 kk_teamvector 3 glass_block 4 glass_warp
static const char* impls[5] = {"kk_serial","kk_team","kk_teamvector","glass_block","glass_warp"};

template<typename T,int N>
static void bench_size(Op op, int reps, const char* dt) {
    size_t mm = (size_t)NPROB*N*N, vv = (size_t)NPROB*N;
    T *A, *B, *C, *x, *x0;
    cudaMalloc(&A, mm*sizeof(T)); cudaMalloc(&B, mm*sizeof(T)); cudaMalloc(&C, mm*sizeof(T));
    cudaMalloc(&x, vv*sizeof(T)); cudaMalloc(&x0, vv*sizeof(T));

    // diagonally-dominant A (valid for trsv), real-magnitude entries.
    T* hA = (T*)malloc(mm*sizeof(T)); T* hB = (T*)malloc(mm*sizeof(T)); T* hx = (T*)malloc(vv*sizeof(T));
    for (size_t q = 0; q < mm; q++) hB[q] = (T)(0.5 - (double)(q % 7) * 0.0625);
    for (int p = 0; p < NPROB; p++)
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++)
            hA[(size_t)p*N*N + i + j*N] = (i==j) ? (T)(N+2) : (T)(0.1*((i+2*j)%5));
    for (size_t q = 0; q < vv; q++) hx[q] = (T)(0.5 + (double)(q % 5) * 0.25);
    cudaMemcpy(A, hA, mm*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, mm*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(x0, hx, vv*sizeof(T), cudaMemcpyHostToDevice);
    g_pre_trial = [=]{   // gemv writes C(as y); trsv solves in-place on x
        cudaMemcpy(x, x0, vv*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    };

    // ── correctness cross-check BEFORE timing ────────────────────────────────
    const int kcheck = NPROB < 64 ? NPROB : 64;
    const double tol = (sizeof(T) == 4) ? 1e-4 * N : 1e-12 * N;
    T* outC = (T*)malloc(mm*sizeof(T)); T* outx = (T*)malloc(vv*sizeof(T));
    bool ok[5];
    for (int c = 0; c < 5; c++) {
        g_pre_trial(); cudaMemset(C, 0, mm*sizeof(T)); cudaGetLastError();
        switch (c) {
            case 0: kk_launch<T,N,0>(op, 0, 0, 0, A, B, C, x); break;
            case 1: kk_launch<T,N,1>(op, 0, 64, 1, A, B, C, x); break;
            case 2: kk_launch<T,N,2>(op, 0, 4, 32, A, B, C, x); break;
            case 3:
                if (op == GEMM) gb_gemm<T,N><<<NPROB,128>>>(A, B, C);
                if (op == GEMV) gb_gemv<T,N><<<NPROB,128>>>(A, x, C);
                if (op == TRSV) gb_trsv<T,N><<<NPROB,128>>>(A, x);
                break;
            case 4: {
                dim3 g((NPROB+3)/4), b(32,4);
                if (op == GEMM) gw_gemm<T,N><<<g,b>>>(A, B, C, NPROB);
                if (op == GEMV) gw_gemv<T,N><<<g,b>>>(A, x, C, NPROB);
                if (op == TRSV) gw_trsv<T,N><<<g,b>>>(A, x, NPROB);
                break;
            }
        }
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            ok[c] = false;
            printf("# CHECK op=%s N=%d %s LAUNCH-FAIL\n", op_name(op), N, impls[c]);
            continue;
        }
        cudaMemcpy(outC, C, mm*sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(outx, x, vv*sizeof(T), cudaMemcpyDeviceToHost);
        double err = ref_maxerr<T>(op, N, kcheck, hA, hB, hx, outC, outx);
        ok[c] = std::isfinite(err) && err <= tol;
        printf("# CHECK op=%s N=%d %s maxerr=%.3e tol=%.1e %s\n",
               op_name(op), N, impls[c], err, tol, ok[c] ? "OK" : "FAIL");
    }
    free(outC); free(outx);

    // ── timing ───────────────────────────────────────────────────────────────
    auto pre = []{ if (g_pre_trial) g_pre_trial(); };
    auto emit = [&](int c, const char* cfg, double ns) {
        if (ns < 1e29)
            printf("RESULT section=kokkos op=%s dtype=%s N=%d NPROB=%d impl=%s cfg=%s ns=%.4f spread=%.2f%%\n",
                   op_name(op), dt, N, NPROB, impls[c], cfg, ns, tc_last_spread_pct());
        else
            printf("RESULT section=kokkos op=%s dtype=%s N=%d NPROB=%d impl=%s cfg=%s ns=FAIL\n",
                   op_name(op), dt, N, NPROB, impls[c], cfg);
    };
    char cfg[32];
    // kk_serial: Unblocked (+Blocked for gemm)
    double ns = ok[0] ? tc_time_ns_per_prob_pre([&]{ kk_launch<T,N,0>(op,0,0,0,A,B,C,x); }, pre, reps, NPROB) : 1e30;
    emit(0, "unb", ns);
    if (op == GEMM) {
        ns = ok[0] ? tc_time_ns_per_prob_pre([&]{ kk_launch<T,N,0>(op,1,0,0,A,B,C,x); }, pre, reps, NPROB) : 1e30;
        emit(0, "blk", ns);
    }
    // kk_team: ts swept, vlen=1
    for (int ts : {32, 64, 128, 256}) {
        snprintf(cfg, sizeof cfg, "ts%d_unb", ts);
        ns = ok[1] ? tc_time_ns_per_prob_pre([&]{ kk_launch<T,N,1>(op,0,ts,1,A,B,C,x); }, pre, reps, NPROB) : 1e30;
        emit(1, cfg, ns);
        if (op == GEMM) {
            snprintf(cfg, sizeof cfg, "ts%d_blk", ts);
            ns = ok[1] ? tc_time_ns_per_prob_pre([&]{ kk_launch<T,N,1>(op,1,ts,1,A,B,C,x); }, pre, reps, NPROB) : 1e30;
            emit(1, cfg, ns);
        }
    }
    // kk_teamvector: ts×32 lanes
    for (int ts : {1, 2, 4, 8}) {
        snprintf(cfg, sizeof cfg, "ts%dv32_unb", ts);
        ns = ok[2] ? tc_time_ns_per_prob_pre([&]{ kk_launch<T,N,2>(op,0,ts,32,A,B,C,x); }, pre, reps, NPROB) : 1e30;
        emit(2, cfg, ns);
    }
    // glass_block
    for (int TB : {32, 64, 128, 256}) {
        auto launch = [&]{
            if (op == GEMM) gb_gemm<T,N><<<NPROB,TB>>>(A, B, C);
            if (op == GEMV) gb_gemv<T,N><<<NPROB,TB>>>(A, x, C);
            if (op == TRSV) gb_trsv<T,N><<<NPROB,TB>>>(A, x);
        };
        snprintf(cfg, sizeof cfg, "tb%d", TB);
        ns = ok[3] ? tc_time_ns_per_prob_pre(launch, pre, reps, NPROB) : 1e30;
        emit(3, cfg, ns);
    }
    // glass_warp
    for (int WPB : {1, 2, 4, 8}) {
        auto launch = [&]{
            dim3 g((NPROB+WPB-1)/WPB), b(32,WPB);
            if (op == GEMM) gw_gemm<T,N><<<g,b>>>(A, B, C, NPROB);
            if (op == GEMV) gw_gemv<T,N><<<g,b>>>(A, x, C, NPROB);
            if (op == TRSV) gw_trsv<T,N><<<g,b>>>(A, x, NPROB);
        };
        snprintf(cfg, sizeof cfg, "w%d", WPB);
        ns = ok[4] ? tc_time_ns_per_prob_pre(launch, pre, reps, NPROB) : 1e30;
        emit(4, cfg, ns);
    }
    fflush(stdout);
    g_pre_trial = nullptr;
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(x); cudaFree(x0);
    free(hA); free(hB); free(hx);
}

template<typename T> static void run_all(int reps, const char* dt) {
    for (int np : {64, 1024, 8192}) {
        NPROB = np;
        printf("# kokkos ladder | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n", NPROB, reps, dt);
        for (Op op : {GEMM, GEMV, TRSV}) {
            bench_size<T,4>(op, reps, dt);  bench_size<T,6>(op, reps, dt);
            bench_size<T,8>(op, reps, dt);  bench_size<T,12>(op, reps, dt);
            bench_size<T,16>(op, reps, dt); bench_size<T,24>(op, reps, dt);
            bench_size<T,32>(op, reps, dt); bench_size<T,48>(op, reps, dt);
            bench_size<T,64>(op, reps, dt);
        }
    }
}

int main(int argc, char** argv) {
    int reps       = (argc > 1) ? atoi(argv[1]) : 500;
    const char* dt = (argc > 2) ? argv[2] : "both";
    Kokkos::initialize();
    {
        printf("# kokkos-kernels team-scope baseline | contenders: KK-SERIAL(thread) | "
               "KK-TEAM(ts swept, vlen=1) | KK-TEAMVECTOR(ts x 32 lanes) | GLASS-BLOCK | GLASS-WARP\n");
        printf("# ops gemm/gemv/trsv only — the KokkosBatched team set has no Cholesky/posv "
               "(measured-where-comparable, prose-where-absent); N<=64; best Kokkos tag/shape reported.\n");
        printf("# kokkos_version=%d.%d.%d\n", KOKKOS_VERSION / 10000,
               (KOKKOS_VERSION / 100) % 100, KOKKOS_VERSION % 100);
        tc_warm_gpu();
        if (strcmp(dt, "f64") == 0 || strcmp(dt, "both") == 0) run_all<double>(reps, "f64");
        if (strcmp(dt, "f32") == 0 || strcmp(dt, "both") == 0) run_all<float>(reps, "f32");
    }
    Kokkos::finalize();
    return 0;
}
