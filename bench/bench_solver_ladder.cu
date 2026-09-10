// bench_solver_ladder.cu — authoritative fresh-input solver placement sweep.
//
// The general ladder measures back-to-back launches and is appropriate for
// non-destructive operations.  POTRF, TRSV, and POSV overwrite their inputs,
// so this companion leg is authoritative for *every* solver contender:
// native block/warp/thread, NVIDIA block, and NVIDIA thread.  Every timed
// launch consumes a distinct valid system from a bounded ring; initialization
// is outside the timed region.
//
// Measurements are organized into paired rounds.  Each contender is measured
// once per round, contender order is shuffled deterministically within the
// round, and every raw sample is emitted.  This supports both the tuner's
// robust median decision and independent/held-out paper analysis.
//
// Usage:
//   ./bench_solver_ladder [nprob=8192] [requested_slots=64]
//                         [dtype=f32|f64] [rounds=9] [seed=1]

// Correctness remains a separate signed-receipt gate.  This timing harness
// checks launch success and supplies mathematically valid inputs; it does not
// turn timing agreement into a numerical oracle.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "timing_common.cuh"

#if defined(GLASS_BENCH_CUSOLVERDX)
#include "../glass-nvidia.cuh"
#define SOLVER_HAVE_NVIDIA_BLOCK 1
#else
#include "../glass.cuh"
#define SOLVER_HAVE_NVIDIA_BLOCK 0
#endif

#if defined(GLASS_HAVE_CUSOLVERDX_THREAD) && GLASS_HAVE_CUSOLVERDX_THREAD
#define SOLVER_HAVE_NVIDIA_THREAD 1
#else
#define SOLVER_HAVE_NVIDIA_THREAD 0
#endif

#define CUDA_OK(call) do {                                                     \
    cudaError_t glass_cuda_error_ = (call);                                    \
    if (glass_cuda_error_ != cudaSuccess) {                                    \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                     cudaGetErrorString(glass_cuda_error_), __FILE__, __LINE__); \
        std::exit(2);                                                          \
    }                                                                          \
} while (0)

enum class Op { potrf, trsv, posv };

static int g_nprob = 8192;
static int g_rounds = 9;
static unsigned long long g_seed = 1;

#if SOLVER_HAVE_NVIDIA_BLOCK
static constexpr int NV_BLOCK_THREADS = 256;
namespace glass { namespace nvidia { namespace block {
    #define SOLVER_CHOL_F32(N) DEFINE_NVIDIA_CHOL_BLOCKDIM(N, NV_BLOCK_THREADS)
    #define SOLVER_TRSM_F32(N) DEFINE_NVIDIA_TRSM_BLOCKDIM(N, 1, NV_BLOCK_THREADS)
    #define SOLVER_POSV_F32(N) DEFINE_NVIDIA_POSV_BLOCKDIM(N, 1, NV_BLOCK_THREADS)
    #define SOLVER_CHOL_F64(N) DEFINE_NVIDIA_CHOL_BLOCKDIM_PREC(N, NV_BLOCK_THREADS, double)
    #define SOLVER_TRSM_F64(N) DEFINE_NVIDIA_TRSM_BLOCKDIM_PREC(N, 1, NV_BLOCK_THREADS, double)
    #define SOLVER_POSV_F64(N) DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(N, 1, NV_BLOCK_THREADS, double)
    #define SOLVER_F32(N) SOLVER_CHOL_F32(N) SOLVER_TRSM_F32(N) SOLVER_POSV_F32(N)
    #define SOLVER_F64(N) SOLVER_CHOL_F64(N) SOLVER_TRSM_F64(N) SOLVER_POSV_F64(N)
    SOLVER_F32(4) SOLVER_F32(6) SOLVER_F32(8) SOLVER_F32(12)
    SOLVER_F32(16) SOLVER_F32(24) SOLVER_F32(32) SOLVER_F32(48)
    SOLVER_F32(64) SOLVER_F32(96) SOLVER_F32(128)
    SOLVER_F64(4) SOLVER_F64(6) SOLVER_F64(8) SOLVER_F64(12)
    SOLVER_F64(16) SOLVER_F64(24) SOLVER_F64(32) SOLVER_F64(48)
    SOLVER_F64(64)
}}}
#endif

template <typename T, int N>
__global__ void init_valid(T* A, T* b, int problems) {
    const size_t matrices = static_cast<size_t>(problems) * N * N;
    for (size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        q < matrices; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int e = static_cast<int>(q % (N * N));
        const size_t p = q / (N * N);
        const int i = e % N, j = e / N;
        if (i == j) {
            A[q] = static_cast<T>(N + 2 + 0.01 * (p % 5));
        } else {
            const int lo = i < j ? i : j, hi = i < j ? j : i;
            A[q] = static_cast<T>(0.02 *
                (1 + ((3 * lo + 5 * hi + static_cast<int>(p % 11)) % 7)));
        }
    }
    const size_t vectors = static_cast<size_t>(problems) * N;
    for (size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         q < vectors; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t p = q / N;
        b[q] = static_cast<T>(1 + (q % N) * 0.03125 + (p % 7) * 0.015625);
    }
}

template <typename T, int N>
__global__ void block_kernel(Op op, T* A, T* b) {
    const int p = blockIdx.x;
    T* Ap = A + static_cast<size_t>(p) * N * N;
    T* bp = b + static_cast<size_t>(p) * N;
    if (op == Op::potrf) glass::block::potrf<T, N>(Ap);
    if (op == Op::trsv)  glass::block::trsv<T, N>(Ap, bp);
    if (op == Op::posv)  glass::block::posv<T, N>(Ap, bp);
}

template <typename T, int N>
__global__ void warp_kernel(Op op, T* A, T* b, int problems) {
    const int p = blockIdx.x * blockDim.y + threadIdx.y;
    if (p >= problems) return;
    T* Ap = A + static_cast<size_t>(p) * N * N;
    T* bp = b + static_cast<size_t>(p) * N;
    if (op == Op::potrf) glass::warp::potrf<T, N>(Ap);
    if (op == Op::trsv)  glass::warp::trsv<T, N>(Ap, bp);
    if (op == Op::posv)  glass::warp::posv<T, N>(Ap, bp);
}

template <typename T, int N>
__global__ void thread_kernel(Op op, T* A, T* b, int problems) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= problems) return;
    T a[N * N], bv[N];
    T* Ap = A + static_cast<size_t>(p) * N * N;
    T* bp = b + static_cast<size_t>(p) * N;
    for (int i = 0; i < N * N; ++i) a[i] = Ap[i];
    if (op != Op::potrf)
        for (int i = 0; i < N; ++i) bv[i] = bp[i];
    if (op == Op::potrf) glass::thread::potrf<T, N>(a);
    if (op == Op::trsv)  glass::thread::trsv<T, N>(a, bv);
    if (op == Op::posv)  glass::thread::posv<T, N>(a, bv);
    if (op == Op::potrf)
        for (int i = 0; i < N * N; ++i) Ap[i] = a[i];
    else
        for (int i = 0; i < N; ++i) bp[i] = bv[i];
}

#if SOLVER_HAVE_NVIDIA_BLOCK
template <typename T, int N>
__global__ void nvidia_block_kernel(Op op, T* A, T* b) {
    extern __shared__ char scratch[];
    const int p = blockIdx.x;
    T* Ap = A + static_cast<size_t>(p) * N * N;
    T* bp = b + static_cast<size_t>(p) * N;
    if (op == Op::potrf)
        glass::nvidia::block::potrf<T, N, NV_BLOCK_THREADS>(Ap, scratch);
    if (op == Op::trsv)
        glass::nvidia::block::trsm<T, N, 1, NV_BLOCK_THREADS>(T(1), Ap, bp, scratch);
    if (op == Op::posv)
        glass::nvidia::block::posv<T, N, 1, NV_BLOCK_THREADS>(Ap, bp, scratch);
}
#endif

#if SOLVER_HAVE_NVIDIA_THREAD
template <typename T, int N>
__global__ void nvidia_thread_kernel(Op op, T* A, T* b, int problems) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= problems) return;
    T* Ap = A + static_cast<size_t>(p) * N * N;
    T* bp = b + static_cast<size_t>(p) * N;
    if (op == Op::potrf) glass::nvidia::thread::potrf<T, N>(Ap);
    if (op == Op::trsv)  glass::nvidia::thread::trsm<T, N, 1>(T(1), Ap, bp);
    if (op == Op::posv)  glass::nvidia::thread::posv<T, N, 1>(Ap, bp);
}
#endif

struct Candidate {
    std::string impl;
    std::string cfg;
    std::function<void(int)> launch;
    std::vector<double> samples;
};

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}

template <typename Prepare>
static double time_once(Prepare prepare, Candidate& candidate, int slots) {
    prepare();
    CUDA_OK(cudaDeviceSynchronize());
    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int slot = 0; slot < slots; ++slot) candidate.launch(slot);
    CUDA_OK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return tc_elapsed_ms(t0, t1) * 1e6 /
           (static_cast<double>(slots) * g_nprob);
}

template <typename T, int N>
static void run_case(Op op, int requested_slots) {
    const size_t bytes_per_slot = static_cast<size_t>(g_nprob) *
        (N * N + N) * sizeof(T);
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = std::min<size_t>(size_t(1) << 30, free_bytes / 3);
    const int slots = std::max(1, std::min(requested_slots,
        static_cast<int>(budget / bytes_per_slot)));
    const size_t problems = static_cast<size_t>(g_nprob) * slots;
    T *A = nullptr, *b = nullptr;
    CUDA_OK(cudaMalloc(&A, problems * N * N * sizeof(T)));
    CUDA_OK(cudaMalloc(&b, problems * N * sizeof(T)));

    auto prepare = [&] {
        const size_t elements = problems * N * N;
        const int blocks = static_cast<int>(std::min<size_t>(4096, (elements + 255) / 256));
        init_valid<T, N><<<blocks, 256>>>(A, b, static_cast<int>(problems));
    };
    auto slot_A = [&](int slot) {
        return A + static_cast<size_t>(slot) * g_nprob * N * N;
    };
    auto slot_b = [&](int slot) {
        return b + static_cast<size_t>(slot) * g_nprob * N;
    };

    std::vector<Candidate> candidates;
    for (int tb : {32, 64, 128, 256}) {
        candidates.push_back({"block", "tb" + std::to_string(tb), [=](int slot) {
            block_kernel<T, N><<<g_nprob, tb>>>(op, slot_A(slot), slot_b(slot));
        }, {}});
    }
    for (int wpb : {1, 2, 4, 8, 16, 32}) {
        candidates.push_back({"warp", "w" + std::to_string(wpb), [=](int slot) {
            const int grid = (g_nprob + wpb - 1) / wpb;
            warp_kernel<T, N><<<grid, dim3(32, wpb)>>>(
                op, slot_A(slot), slot_b(slot), g_nprob);
        }, {}});
    }
    if constexpr (N <= 64) {
        for (int tb : {32, 64, 128, 256}) {
            candidates.push_back({"thread", "t" + std::to_string(tb), [=](int slot) {
                const int grid = (g_nprob + tb - 1) / tb;
                thread_kernel<T, N><<<grid, tb>>>(
                    op, slot_A(slot), slot_b(slot), g_nprob);
            }, {}});
        }
    }

#if SOLVER_HAVE_NVIDIA_BLOCK
    if constexpr (std::is_same_v<T, float> || N <= 64) {
        size_t smem = 0;
        if (op == Op::potrf)
            smem = glass::nvidia::block::potrf_scratch_bytes<T, N, NV_BLOCK_THREADS>();
        if (op == Op::trsv)
            smem = glass::nvidia::block::trsm_scratch_bytes<T, N, 1, NV_BLOCK_THREADS>();
        if (op == Op::posv)
            smem = glass::nvidia::block::posv_scratch_bytes<T, N, 1, NV_BLOCK_THREADS>();
        int device = 0, optin = 0;
        CUDA_OK(cudaGetDevice(&device));
        CUDA_OK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (smem <= static_cast<size_t>(optin) &&
            (smem <= 48u * 1024u ||
             cudaFuncSetAttribute(nvidia_block_kernel<T, N>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)) == cudaSuccess)) {
            candidates.push_back({"nvidia", "tb256", [=](int slot) {
                nvidia_block_kernel<T, N><<<g_nprob, NV_BLOCK_THREADS, smem>>>(
                    op, slot_A(slot), slot_b(slot));
            }, {}});
        } else {
            cudaGetLastError();
        }
    }
#endif

#if SOLVER_HAVE_NVIDIA_THREAD
    if constexpr (N <= 32) {
        for (int tb : {32, 64, 128, 256}) {
            candidates.push_back({"nvidia_thread", "t" + std::to_string(tb), [=](int slot) {
                const int grid = (g_nprob + tb - 1) / tb;
                nvidia_thread_kernel<T, N><<<grid, tb>>>(
                    op, slot_A(slot), slot_b(slot), g_nprob);
            }, {}});
        }
    }
#endif

    // Probe every candidate before timing; an infeasible launch is omitted and
    // cannot masquerade as an implausibly fast result.
    std::vector<Candidate> valid;
    for (auto& candidate : candidates) {
        cudaGetLastError();
        prepare();
        candidate.launch(0);
        cudaError_t sync = cudaDeviceSynchronize();
        cudaError_t launch = cudaGetLastError();
        if (sync == cudaSuccess && launch == cudaSuccess) {
            valid.push_back(std::move(candidate));
        } else {
            std::printf("SOLVER_SKIP op=%s N=%d dtype=%s nprob=%d impl=%s cfg=%s reason=launch\n",
                        op == Op::potrf ? "potrf" : op == Op::trsv ? "trsv" : "posv",
                        N, std::is_same_v<T, float> ? "f32" : "f64", g_nprob,
                        candidate.impl.c_str(), candidate.cfg.c_str());
            cudaGetLastError();
        }
    }
    candidates = std::move(valid);

    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::mt19937_64 rng(g_seed ^ (static_cast<unsigned long long>(N) << 32) ^
                        (static_cast<unsigned long long>(g_nprob) << 4) ^
                        static_cast<unsigned long long>(op));
    for (int round = 0; round < g_rounds; ++round) {
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t index : order) {
            candidates[index].samples.push_back(
                time_once(prepare, candidates[index], slots));
        }
    }

    const char* opname = op == Op::potrf ? "potrf" :
                         op == Op::trsv ? "trsv" : "posv";
    for (const auto& candidate : candidates) {
        const auto bounds = std::minmax_element(candidate.samples.begin(),
                                                candidate.samples.end());
        const double med = median(candidate.samples);
        const double spread = (*bounds.second / *bounds.first - 1.0) * 100.0;
        std::printf("SOLVER_RESULT op=%s N=%d dtype=%s nprob=%d slots=%d "
                    "impl=%s cfg=%s ns=%.6f spread=%.2f samples=",
                    opname, N, std::is_same_v<T, float> ? "f32" : "f64",
                    g_nprob, slots, candidate.impl.c_str(), candidate.cfg.c_str(),
                    med, spread);
        for (size_t i = 0; i < candidate.samples.size(); ++i)
            std::printf("%s%.6f", i ? "," : "", candidate.samples[i]);
        std::printf("\n");
    }

    CUDA_OK(cudaFree(A));
    CUDA_OK(cudaFree(b));
}

template <typename T, int N>
static void run_size(int slots) {
    run_case<T, N>(Op::potrf, slots);
    run_case<T, N>(Op::trsv, slots);
    run_case<T, N>(Op::posv, slots);
}

template <typename T>
static void run_type(int slots) {
    run_size<T, 4>(slots);   run_size<T, 6>(slots);
    run_size<T, 8>(slots);   run_size<T, 12>(slots);
    run_size<T, 16>(slots);  run_size<T, 24>(slots);
    run_size<T, 32>(slots);  run_size<T, 48>(slots);
    run_size<T, 64>(slots);  run_size<T, 96>(slots);
    run_size<T, 128>(slots);
}

int main(int argc, char** argv) {
    g_nprob = argc > 1 ? std::atoi(argv[1]) : 8192;
    const int slots = argc > 2 ? std::atoi(argv[2]) : 64;
    const char* dtype = argc > 3 ? argv[3] : "f32";
    g_rounds = argc > 4 ? std::atoi(argv[4]) : 9;
    g_seed = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1;
    if (g_nprob <= 0 || slots <= 0 || g_rounds < 3) {
        std::fprintf(stderr, "nprob/slots must be positive and rounds >= 3\n");
        return 2;
    }
    std::printf("# fresh-input solver ladder | NPROB=%d requested_slots=%d "
                "dtype=%s rounds=%d seed=%llu | median ns/problem\n",
                g_nprob, slots, dtype, g_rounds, g_seed);
    std::printf("# paired randomized rounds; every launch consumes a distinct valid batch; "
                "initialization is outside timing; raw round samples follow each row\n");
    tc_warm_gpu();
    if (std::strcmp(dtype, "f64") == 0) run_type<double>(slots);
    else run_type<float>(slots);
    return 0;
}
