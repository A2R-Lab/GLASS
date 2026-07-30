// bench_robotics.cu — micro-op timing legs for the robotics families:
//
//   1. FUSED vs COMPOSED — each fused apply (motion_cross_mul, force_cross_mul,
//      motion_transform_mul, spatial_inertia_mul) against the composition it
//      replaces (materialize the 6x6 into scratch + glass::gemv), same tier.
//      The claim under test is correctness economics, not FLOPs: a wash
//      CONFIRMS "hand-rolled matches perf but not guarantees".
//   2. TIER PACKING — batched robotics workloads (quat_retract, se3_retract,
//      quat_error, eig3, svd3, closest_rotation, softmax) across the
//      block / warp / thread packings (the 17_thread_pack story replayed).
//   3. ARGREDUCE strategies — argmax default vs argmax_fast (block tier).
//
// Metric: ns per problem (wall / (reps*NPROB)), min of 3 trials, FAIL-guarded
// (a failed launch times as ~ns and would otherwise poison the argmin — the
// bench_mega_sweep lesson). Timing-only: outputs are overwritten across reps.
//
// PROTOCOL: timed runs ONLY on a quiet GPU (verify EMPTY via nvidia-smi
// first), serially, no concurrent CPU work. Compile/smoke anytime:
//   nvcc -std=c++17 -arch=<sm> -O3 -I.. -I../src bench_robotics.cu -o bench_robotics
// Usage: ./bench_robotics [nprob=8192] [reps=500] [dtype=f32|f64]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#include "../glass.cuh"

static int NPROB = 8192;

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

// ─── ops ─────────────────────────────────────────────────────────────────────
// Per-problem input strides: in0 = 12 (E|r, pose|-, quat|quat_des, sym3, pi),
// in1 = 8 (v6 / phi3 / rho3+phi3 / softmax uses in0 rows of N16), out = 36.
enum Op {
    QUAT_RETRACT, SE3_RETRACT, QUAT_ERROR,
    MCROSS_FUSED, MCROSS_COMPOSED, FCROSS_FUSED, FCROSS_COMPOSED,
    MXFORM_FUSED, MXFORM_COMPOSED, SINERTIA_FUSED, SINERTIA_COMPOSED,
    EIG3, SVD3, CLOSEST_ROT, ARGMAX_DEFAULT, ARGMAX_FAST, SOFTMAX16,
    OP_N
};
static const char* OP_NAMES[OP_N] = {
    "quat_retract", "se3_retract", "quat_error",
    "mcross_fused", "mcross_composed", "fcross_fused", "fcross_composed",
    "mxform_fused", "mxform_composed", "sinertia_fused", "sinertia_composed",
    "eig3", "svd3", "closest_rot", "argmax_n16", "argmax_fast_n16", "softmax_n16",
};
#define IN0 12
#define IN1 8
#define OUTS 36
#define SM_N 16   // softmax/argmax vector length (in0 + in1 rows read as one 16-vec... kept separate: uses in0's 12 + in1's first 4)

// ─── BLOCK tier: block p owns problem p ──────────────────────────────────────
template <typename T>
__device__ __forceinline__ void run_block_op(int op, const T* a, const T* b, T* o, T* smem) {
    switch (op) {
        case QUAT_RETRACT: glass::block::quat_retract<T>(a, b, o); break;
        case SE3_RETRACT:  glass::block::se3_retract<T>(a, b, b + 3, o); break;
        case QUAT_ERROR:   glass::block::quat_error<T>(a, a + 4, o); break;
        case MCROSS_FUSED: glass::block::motion_cross_mul<T>((T)1, a, b, (T)0, o); break;
        case MCROSS_COMPOSED:
            glass::block::motion_cross<T>(a, smem);
            glass::block::gemv<T, 6, 6>((T)1, smem, b, (T)0, o);
            break;
        case FCROSS_FUSED: glass::block::force_cross_mul<T>((T)1, a, b, (T)0, o); break;
        case FCROSS_COMPOSED:
            glass::block::force_cross<T>(a, smem);
            glass::block::gemv<T, 6, 6>((T)1, smem, b, (T)0, o);
            break;
        case MXFORM_FUSED: glass::block::motion_transform_mul<T>((T)1, a, a + 9, b, (T)0, o); break;
        case MXFORM_COMPOSED:
            glass::block::motion_transform<T>(a, a + 9, smem);
            glass::block::gemv<T, 6, 6>((T)1, smem, b, (T)0, o);
            break;
        case SINERTIA_FUSED: glass::block::spatial_inertia_mul<T>((T)1, a, b, (T)0, o); break;
        case SINERTIA_COMPOSED:
            glass::block::spatial_inertia<T>(a, smem);
            glass::block::gemv<T, 6, 6>((T)1, smem, b, (T)0, o);
            break;
        case EIG3:        glass::block::eig3<T>(a, o, o + 3); break;
        case SVD3:        glass::block::svd3<T>(a, o, o + 9, o + 12); break;
        case CLOSEST_ROT: glass::block::closest_rotation<T>(a, o); break;
        case ARGMAX_DEFAULT: {
            __shared__ uint32_t idx[1];
            glass::block::argmax<T>(SM_N, a, idx, smem);
            if (threadIdx.x == 0) o[0] = (T)idx[0];
            break;
        }
        case ARGMAX_FAST: {
            __shared__ uint32_t idx[1];
            glass::block::argmax_fast<T>(SM_N, a, idx, smem);
            if (threadIdx.x == 0) o[0] = (T)idx[0];
            break;
        }
        case SOFTMAX16: glass::block::softmax<T>(SM_N, (T)-0.75, a, o, smem); break;
        default: break;
    }
}
template <typename T>
__global__ void k_block(int op, const T* i0, const T* i1, T* out) {
    extern __shared__ char raw[];
    T* smem = reinterpret_cast<T*>(raw);
    int p = blockIdx.x;
    run_block_op<T>(op, i0 + (size_t)p*IN0, i1 + (size_t)p*IN1, out + (size_t)p*OUTS, smem);
}

// ─── WARP tier: warp owns problem (no smem-dependent composed/argmax legs) ──
template <typename T>
__global__ void k_warp(int op, const T* i0, const T* i1, T* out, int np) {
    int p = blockIdx.x*blockDim.y + threadIdx.y;
    if (p >= np) return;
    const T* a = i0 + (size_t)p*IN0;
    const T* b = i1 + (size_t)p*IN1;
    T* o = out + (size_t)p*OUTS;
    switch (op) {
        case QUAT_RETRACT: glass::warp::quat_retract<T>(a, b, o); break;
        case SE3_RETRACT:  glass::warp::se3_retract<T>(a, b, b + 3, o); break;
        case QUAT_ERROR:   glass::warp::quat_error<T>(a, a + 4, o); break;
        case MCROSS_FUSED: glass::warp::motion_cross_mul<T>((T)1, a, b, (T)0, o); break;
        case FCROSS_FUSED: glass::warp::force_cross_mul<T>((T)1, a, b, (T)0, o); break;
        case MXFORM_FUSED: glass::warp::motion_transform_mul<T>((T)1, a, a + 9, b, (T)0, o); break;
        case SINERTIA_FUSED: glass::warp::spatial_inertia_mul<T>((T)1, a, b, (T)0, o); break;
        case EIG3:        glass::warp::eig3<T>(a, o, o + 3); break;
        case SVD3:        glass::warp::svd3<T>(a, o, o + 9, o + 12); break;
        case CLOSEST_ROT: glass::warp::closest_rotation<T>(a, o); break;
        case ARGMAX_DEFAULT: { uint32_t i = glass::warp::argmax<T>(SM_N, a); if ((threadIdx.x & 31) == 0) o[0] = (T)i; break; }
        case SOFTMAX16:   glass::warp::softmax<T>(SM_N, (T)-0.75, a, o); break;
        default: break;
    }
}

// ─── THREAD tier: thread owns problem ────────────────────────────────────────
template <typename T>
__global__ void k_thread(int op, const T* i0, const T* i1, T* out, int np) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= np) return;
    const T* a = i0 + (size_t)p*IN0;
    const T* b = i1 + (size_t)p*IN1;
    T* o = out + (size_t)p*OUTS;
    switch (op) {
        case QUAT_RETRACT: glass::thread::quat_retract<T>(a, b, o); break;
        case SE3_RETRACT:  glass::thread::se3_retract<T>(a, b, b + 3, o); break;
        case QUAT_ERROR:   glass::thread::quat_error<T>(a, a + 4, o); break;
        case MCROSS_FUSED: glass::thread::motion_cross_mul<T>((T)1, a, b, (T)0, o); break;
        case MCROSS_COMPOSED: {
            T M[36];
            glass::thread::motion_cross<T>(a, M);
            glass::thread::gemv<T, 6, 6>((T)1, M, b, o);
            break;
        }
        case FCROSS_FUSED: glass::thread::force_cross_mul<T>((T)1, a, b, (T)0, o); break;
        case FCROSS_COMPOSED: {
            T M[36];
            glass::thread::force_cross<T>(a, M);
            glass::thread::gemv<T, 6, 6>((T)1, M, b, o);
            break;
        }
        case MXFORM_FUSED: glass::thread::motion_transform_mul<T>((T)1, a, a + 9, b, (T)0, o); break;
        case MXFORM_COMPOSED: {
            T M[36];
            glass::thread::motion_transform<T>(a, a + 9, M);
            glass::thread::gemv<T, 6, 6>((T)1, M, b, o);
            break;
        }
        case SINERTIA_FUSED: glass::thread::spatial_inertia_mul<T>((T)1, a, b, (T)0, o); break;
        case SINERTIA_COMPOSED: {
            T M[36];
            glass::thread::spatial_inertia<T>(a, M);
            glass::thread::gemv<T, 6, 6>((T)1, M, b, o);
            break;
        }
        case EIG3:        glass::thread::eig3<T>(a, o, o + 3); break;
        case SVD3:        glass::thread::svd3<T>(a, o, o + 9, o + 12); break;
        case CLOSEST_ROT: glass::thread::closest_rotation<T>(a, o); break;
        case ARGMAX_DEFAULT: o[0] = (T)glass::thread::argmax<T>(SM_N, a); break;
        case SOFTMAX16:   glass::thread::softmax<T>(SM_N, (T)-0.75, a, o); break;
        default: break;
    }
}

static bool has_warp(int op) {
    switch (op) {
        case MCROSS_COMPOSED: case FCROSS_COMPOSED: case MXFORM_COMPOSED:
        case SINERTIA_COMPOSED: case ARGMAX_FAST: return false;
        default: return true;
    }
}
static bool has_thread(int op) { return op != ARGMAX_FAST; }

// ─── host-side valid inputs (unit quats, rotations, safe params) ─────────────
template <typename T>
static void fill_inputs(T* h0, T* h1) {
    for (int p = 0; p < NPROB; p++) {
        double th = 0.1 + 2.8 * ((p * 2654435761u % 1000) / 1000.0);
        double ax[3] = {sin(1.0 + p), cos(0.5 + 2.0*p), sin(2.0 + 0.7*p)};
        double an = sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2]);
        for (int i = 0; i < 3; i++) ax[i] /= an;
        // in0: a unit quat [x y z w] at 0, a second at 4, then 4 fillers —
        //      ALSO read as (E|r) 12, sym3 9, pi 10, pose 7, argmax/softmax 12.
        double s = sin(th/2), c = cos(th/2);
        T* a = h0 + (size_t)p*IN0;
        a[0] = (T)(s*ax[0]); a[1] = (T)(s*ax[1]); a[2] = (T)(s*ax[2]); a[3] = (T)c;
        double s2 = sin(th/3), c2 = cos(th/3);
        a[4] = (T)(s2*ax[1]); a[5] = (T)(s2*ax[2]); a[6] = (T)(s2*ax[0]); a[7] = (T)c2;
        a[8] = (T)(0.3*sin(3.0*p)); a[9] = (T)(0.2*cos(1.7*p));
        a[10] = (T)(0.1*sin(0.9*p)); a[11] = (T)(0.4*cos(2.3*p));
        // For the (E|r) / sym3 / pi readings the quat-ish values are merely
        // finite well-scaled numbers — valid for TIMING (not correctness).
        T* b = h1 + (size_t)p*IN1;
        for (int i = 0; i < IN1; i++) b[i] = (T)(0.5*sin(0.31*p + 1.3*i));
    }
}

template <typename T>
static void bench_all(int reps) {
    T *i0, *i1, *out, *h0, *h1;
    size_t n0 = (size_t)NPROB*IN0, n1 = (size_t)NPROB*IN1, no = (size_t)NPROB*OUTS;
    h0 = (T*)malloc(n0*sizeof(T)); h1 = (T*)malloc(n1*sizeof(T));
    fill_inputs<T>(h0, h1);
    cudaMalloc(&i0, n0*sizeof(T)); cudaMalloc(&i1, n1*sizeof(T)); cudaMalloc(&out, no*sizeof(T));
    cudaMemcpy(i0, h0, n0*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(i1, h1, n1*sizeof(T), cudaMemcpyHostToDevice);
    free(h0); free(h1);
    const size_t smem = 4096;

    for (int op = 0; op < OP_N; op++) {
        printf("%-18s | BLOCK", OP_NAMES[op]);
        double best_b = 1e30, best_w = 1e30, best_t = 1e30;
        for (int TB : {32, 64, 128, 256}) {
            double ns = time_ns_per_prob([&]{ k_block<T><<<NPROB, TB, smem>>>(op, i0, i1, out); }, reps);
            if (ns < 1e29) printf("  tb%d=%.4f", TB, ns); else printf("  tb%d=FAIL", TB);
            if (ns < best_b) best_b = ns;
        }
        if (has_warp(op)) {
            printf("  | WARP");
            for (int WPB : {2, 4, 8}) {
                dim3 grid((NPROB + WPB - 1)/WPB), blk(32, WPB);
                double ns = time_ns_per_prob([&]{ k_warp<T><<<grid, blk>>>(op, i0, i1, out, NPROB); }, reps);
                if (ns < 1e29) printf("  w%d=%.4f", WPB, ns); else printf("  w%d=FAIL", WPB);
                if (ns < best_w) best_w = ns;
            }
        }
        if (has_thread(op)) {
            printf("  | THREAD");
            for (int TPB : {64, 128, 256}) {
                int blocks = (NPROB + TPB - 1)/TPB;
                double ns = time_ns_per_prob([&]{ k_thread<T><<<blocks, TPB>>>(op, i0, i1, out, NPROB); }, reps);
                if (ns < 1e29) printf("  t%d=%.4f", TPB, ns); else printf("  t%d=FAIL", TPB);
                if (ns < best_t) best_t = ns;
            }
        }
        const char* win = (best_t <= best_b && best_t <= best_w) ? "THREAD"
                        : (best_w <= best_b) ? "WARP" : "BLOCK";
        printf("  || -> %s\n", win);
    }
    cudaFree(i0); cudaFree(i1); cudaFree(out);
}

int main(int argc, char** argv) {
    NPROB    = (argc > 1) ? atoi(argv[1]) : 8192;
    int reps = (argc > 2) ? atoi(argv[2]) : 500;
    const char* dt = (argc > 3) ? argv[3] : "f32";
    bool f64 = (strcmp(dt, "f64") == 0);
    printf("# robotics micro-op sweep | NPROB=%d reps=%d dtype=%s | ns/problem (lower=better)\n",
           NPROB, reps, f64 ? "f64" : "f32");
    printf("# fused-vs-composed pairs share a tier; composed = materialize 6x6 + gemv.\n");
    printf("# TIMED RUNS REQUIRE A QUIET GPU (see file header).\n");
    if (f64) bench_all<double>(reps);
    else     bench_all<float>(reps);
    return 0;
}
