// bench_nvt_valid.cu — valid-input confirmation for NVIDIA-thread defaults.
//
// The main ladder intentionally measures many back-to-back launches between
// restores. That is a useful steady-throughput workload, but in-place solvers
// see their own output after the first launch. This companion benchmark gives
// every timed launch a distinct, valid input batch. It is a veto gate only:
// tune.py may retain an NVIDIA-thread ladder winner when it also clears the
// native thread/warp/block winner here, but this leg never promotes a vendor
// backend that did not win the main ladder.
//
// Usage: ./bench_nvt_valid [nprob=8192] [requested_slots=64] [dtype=f32|f64]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <type_traits>

#include "timing_common.cuh"
#include "../glass-nvidia.cuh"

#define CUDA_OK(call) do {                                                    \
    cudaError_t glass_cuda_error_ = (call);                                   \
    if (glass_cuda_error_ != cudaSuccess) {                                   \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                     cudaGetErrorString(glass_cuda_error_), __FILE__, __LINE__); \
        std::exit(2);                                                         \
    }                                                                         \
} while (0)

enum class Op { potrf, trsv, posv };

static int g_nprob = 8192;

template <typename T, int N>
__global__ void init_valid(T* A, T* b, int problems) {
    const size_t matrices = static_cast<size_t>(problems) * N * N;
    for (size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         q < matrices; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int e = static_cast<int>(q % (N * N));
        const int i = e % N, j = e / N;
        if (i == j) {
            A[q] = static_cast<T>(N + 2);
        } else {
            const int lo = i < j ? i : j, hi = i < j ? j : i;
            A[q] = static_cast<T>(0.02 * (1 + ((3 * lo + 5 * hi) % 7)));
        }
    }
    const size_t vectors = static_cast<size_t>(problems) * N;
    for (size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         q < vectors; q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        b[q] = static_cast<T>(1 + (q % N) * 0.03125);
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

template <typename T, int N>
__global__ void nvt_kernel(Op op, T* A, T* b, int problems) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= problems) return;
    T* Ap = A + static_cast<size_t>(p) * N * N;
    T* bp = b + static_cast<size_t>(p) * N;
    if (op == Op::potrf) glass::nvidia::thread::potrf<T, N>(Ap);
    if (op == Op::trsv)  glass::nvidia::thread::trsm<T, N, 1>(T(1), Ap, bp);
    if (op == Op::posv)  glass::nvidia::thread::posv<T, N, 1>(Ap, bp);
}

struct Timing {
    double ns = 1e30;
    double spread = 0.0;
    int shape = 0;
};

template <typename Prepare, typename LaunchSlot>
static Timing time_slots(Prepare prepare, LaunchSlot launch, int slots) {
    cudaGetLastError();
    prepare();
    launch(0);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaGetLastError() != cudaSuccess) return {};

    double best = 1e30, worst = 0.0;
    for (int trial = 0; trial < 3; ++trial) {
        prepare();
        CUDA_OK(cudaDeviceSynchronize());
        timespec t0{}, t1{};
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int slot = 0; slot < slots; ++slot) launch(slot);
        CUDA_OK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ns = tc_elapsed_ms(t0, t1) * 1e6 /
                          (static_cast<double>(slots) * g_nprob);
        best = std::min(best, ns);
        worst = std::max(worst, ns);
    }
    return {best, (worst / best - 1.0) * 100.0, 0};
}

static void retain(Timing& best, Timing candidate, int shape) {
    if (candidate.ns < best.ns) {
        best = candidate;
        best.shape = shape;
    }
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
        const int blocks = std::min<size_t>(4096, (elements + 255) / 256);
        init_valid<T, N><<<blocks, 256>>>(A, b, static_cast<int>(problems));
    };
    auto slot_A = [&](int slot) {
        return A + static_cast<size_t>(slot) * g_nprob * N * N;
    };
    auto slot_b = [&](int slot) {
        return b + static_cast<size_t>(slot) * g_nprob * N;
    };

    constexpr int TBS[] = {32, 64, 128, 256};
    constexpr int WPBS[] = {1, 2, 4, 8, 16, 32};
    Timing block, warp, thread, nvt;
    for (int tb : TBS) {
        retain(block, time_slots(prepare, [&](int s) {
            block_kernel<T, N><<<g_nprob, tb>>>(op, slot_A(s), slot_b(s));
        }, slots), tb);
        retain(thread, time_slots(prepare, [&](int s) {
            const int grid = (g_nprob + tb - 1) / tb;
            thread_kernel<T, N><<<grid, tb>>>(op, slot_A(s), slot_b(s), g_nprob);
        }, slots), tb);
        retain(nvt, time_slots(prepare, [&](int s) {
            const int grid = (g_nprob + tb - 1) / tb;
            nvt_kernel<T, N><<<grid, tb>>>(op, slot_A(s), slot_b(s), g_nprob);
        }, slots), tb);
    }
    for (int wpb : WPBS) {
        retain(warp, time_slots(prepare, [&](int s) {
            const int grid = (g_nprob + wpb - 1) / wpb;
            warp_kernel<T, N><<<grid, dim3(32, wpb)>>>(
                op, slot_A(s), slot_b(s), g_nprob);
        }, slots), wpb);
    }

    const char* opname = op == Op::potrf ? "potrf" :
                         op == Op::trsv ? "trsv" : "posv";
    std::printf("NVT_VALID op=%s N=%d dtype=%s nprob=%d slots=%d "
                "block=%.4f block_shape=%d block_spread=%.2f "
                "warp=%.4f warp_shape=%d warp_spread=%.2f "
                "thread=%.4f thread_shape=%d thread_spread=%.2f "
                "nvidia_thread=%.4f nvt_shape=%d nvt_spread=%.2f\n",
                opname, N, std::is_same_v<T, float> ? "f32" : "f64",
                g_nprob, slots,
                block.ns, block.shape, block.spread,
                warp.ns, warp.shape, warp.spread,
                thread.ns, thread.shape, thread.spread,
                nvt.ns, nvt.shape, nvt.spread);
    CUDA_OK(cudaFree(A));
    CUDA_OK(cudaFree(b));
}

template <typename T, int N>
static void run_size(int requested_slots) {
    run_case<T, N>(Op::potrf, requested_slots);
    run_case<T, N>(Op::trsv, requested_slots);
    run_case<T, N>(Op::posv, requested_slots);
}

template <typename T>
static void run_type(int requested_slots) {
    run_size<T, 4>(requested_slots);
    run_size<T, 6>(requested_slots);
    run_size<T, 8>(requested_slots);
    run_size<T, 12>(requested_slots);
    run_size<T, 16>(requested_slots);
    run_size<T, 24>(requested_slots);
    run_size<T, 32>(requested_slots);
}

int main(int argc, char** argv) {
    g_nprob = argc > 1 ? std::atoi(argv[1]) : 8192;
    const int slots = argc > 2 ? std::atoi(argv[2]) : 64;
    const char* dtype = argc > 3 ? argv[3] : "f32";
    if (g_nprob <= 0 || slots <= 0) {
        std::fprintf(stderr, "nprob and requested_slots must be positive\n");
        return 2;
    }
    std::printf("# nvidia-thread valid-input confirmation | NPROB=%d "
                "requested_slots=%d dtype=%s | ns/problem, min of 3\n",
                g_nprob, slots, dtype);
    std::printf("# each timed launch consumes one independent valid batch; "
                "initialization is outside the timed region; memory cap=1GiB\n");
    tc_warm_gpu();
    if (std::strcmp(dtype, "f64") == 0) run_type<double>(slots);
    else run_type<float>(slots);
    return 0;
}
