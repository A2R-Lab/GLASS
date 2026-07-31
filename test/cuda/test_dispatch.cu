// test_dispatch.cu — functional validation of the bare-face body dispatch
// (src/base/dispatch.cuh + glass-dispatch.cuh, 2026-07-30 Phase 2).
//
// For every op family with moved cells, runs the BARE glass::op spelling and
// the explicit glass::block:: contract twin on identical inputs across block
// widths {16, 32, 64, 256} and asserts the outputs agree within reduction-
// order tolerance. TB=16 exercises the narrow-block fallback (warp bodies
// require a full x-major first warp and must fall back to the block body);
// TB=256 is where the moved bodies actually win. The host side also prints
// which body the compiled table picked per cell, so a failure names the body
// it came from.
//
// Bit-exactness is NOT asserted — the bare face is the performance tier; the
// contract tier (glass::block::) keeps bit-exact thread-count invariance and
// is tested elsewhere.
//
// Usage: ./test_dispatch <op>   ops: dot_f32 dot_f64 gemv_f32 chol_f32
//                                    trsv_f32 posv_f32 eig3_f64 softmax_f32
// Prints PASS / FAIL per (cell, TB); returns 0 iff all pass.

#include "glass.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA: %s @ %d: %s\n", cudaGetErrorString(e), __LINE__, #x); \
    std::exit(1); } } while (0)

static const int TBS[] = {16, 32, 64, 256};

template <typename T>
static double tol() { return sizeof(T) == 8 ? 1e-10 : 1e-4; }

template <typename T>
static double max_rel_diff(const std::vector<T>& a, const std::vector<T>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double denom = std::max(1.0, std::fabs((double)b[i]));
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]) / denom);
    }
    return m;
}

static const char* body_name(glass::body b) {
    switch (b) {
        case glass::body::warp_in_block:   return "warp_in_block";
        case glass::body::thread_in_block: return "thread_in_block";
        default:                           return "block";
    }
}

static int g_fail = 0;
template <typename T>
static void report(const char* cell, glass::body b, int tb,
                   const std::vector<T>& bare, const std::vector<T>& blk) {
    double d = max_rel_diff(bare, blk);
    bool ok = d <= tol<T>();
    std::printf("%-14s body=%-15s tb=%-3d max_rel=%.3e %s\n",
                cell, body_name(b), tb, d, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}

// ─── kernels: bare face vs block contract twin ───────────────────────────────

template <typename T, uint32_t N, bool BARE>
__global__ void k_dot(const T* X, const T* Y, T* out) {
    __shared__ T x[N], y[N];
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) { x[i] = X[i]; y[i] = Y[i]; }
    __syncthreads();
    if constexpr (BARE) glass::dot<T, N>(x, y);
    else                glass::block::dot<T, N>(x, y);
    if (threadIdx.x == 0) out[0] = y[0];
}

template <typename T, uint32_t N, bool BARE>
__global__ void k_gemv(const T* A, const T* X, T* out) {
    __shared__ T y[N];
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) y[i] = T(0);
    __syncthreads();
    if constexpr (BARE) glass::gemv<T, N, N>(T(1), A, X, T(0), y);
    else                glass::block::gemv<T, N, N>(T(1), A, X, T(0), y);
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) out[i] = y[i];
}

template <typename T, uint32_t N, bool BARE>
__global__ void k_chol(const T* A0, T* out) {
    __shared__ T a[N * N];
    for (uint32_t i = threadIdx.x; i < N * N; i += blockDim.x) a[i] = A0[i];
    __syncthreads();
    if constexpr (BARE) glass::potrf<T, N>(a);
    else                glass::block::potrf<T, N>(a);
    for (uint32_t i = threadIdx.x; i < N * N; i += blockDim.x) out[i] = a[i];
}

template <typename T, uint32_t N, bool BARE>
__global__ void k_trsv(const T* A, const T* B, T* out) {
    __shared__ T x[N];
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) x[i] = B[i];
    __syncthreads();
    if constexpr (BARE) glass::trsv<T, N>(A, x);
    else                glass::block::trsv<T, N>(A, x);
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) out[i] = x[i];
}

template <typename T, uint32_t N, bool BARE>
__global__ void k_posv(const T* A0, const T* B, T* out) {
    __shared__ T a[N * N], b[N];
    for (uint32_t i = threadIdx.x; i < N * N; i += blockDim.x) a[i] = A0[i];
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) b[i] = B[i];
    __syncthreads();
    if constexpr (BARE) glass::posv<T, N>(a, b);
    else                glass::block::posv<T, N>(a, b);
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) out[i] = b[i];
}

template <typename T, bool BARE>
__global__ void k_eig3(const T* A, T* out) {   // out = [W(3) | V(9)]
    if constexpr (BARE) glass::eig3<T>(A, out, out + 3);
    else                glass::block::eig3<T>(A, out, out + 3);
}

template <typename T, bool BARE>
__global__ void k_softmax(uint32_t n, const T* X, T* out) {
    extern __shared__ unsigned char smem[];
    T* scr = reinterpret_cast<T*>(smem);
    if constexpr (BARE) glass::softmax<T>(n, T(-0.75), X, out, scr);
    else                glass::block::softmax<T>(n, T(-0.75), X, out, scr);
}

// ─── host drivers ────────────────────────────────────────────────────────────

template <typename T>
static std::vector<T> randv(size_t n, unsigned seed) {
    std::vector<T> v(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = T((double)(s >> 8) / (double)(1u << 24)) - T(0.5);
    }
    return v;
}

// SPD: A = M Mᵀ + n·I  (column-major, symmetric so layout is moot)
template <typename T>
static std::vector<T> spd(uint32_t n, unsigned seed) {
    auto m = randv<T>((size_t)n * n, seed);
    std::vector<T> a((size_t)n * n, T(0));
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++) {
            double s = (i == j) ? (double)n : 0.0;
            for (uint32_t k = 0; k < n; k++)
                s += (double)m[i + k * n] * (double)m[j + k * n];
            a[i + j * n] = T(s);
        }
    return a;
}

// Lower-triangular nonsingular (unit-dominant diagonal), column-major.
template <typename T>
static std::vector<T> lower(uint32_t n, unsigned seed) {
    auto a = randv<T>((size_t)n * n, seed);
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < n; i++) {
            if (i < j) a[i + j * n] = T(0);
            if (i == j) a[i + j * n] = T(2) + std::fabs((double)a[i + j * n]);
        }
    return a;
}

template <typename T, uint32_t N>
static void drive_dot(const char* cell) {
    auto hx = randv<T>(N, 11), hy = randv<T>(N, 22);
    T *dx, *dy, *dout;
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    glass::body b = glass::dispatch_body(glass::op::dot, N, sizeof(T) == 8);
    for (int tb : TBS) {
        std::vector<T> r_bare(1), r_blk(1);
        k_dot<T, N, true><<<1, tb>>>(dx, dy, dout);
        CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, sizeof(T), cudaMemcpyDeviceToHost));
        k_dot<T, N, false><<<1, tb>>>(dx, dy, dout);
        CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, sizeof(T), cudaMemcpyDeviceToHost));
        report(cell, b, tb, r_bare, r_blk);
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dout);
}

template <typename T, uint32_t N>
static void drive_gemv(const char* cell) {
    auto ha = randv<T>((size_t)N * N, 33);
    auto hx = randv<T>(N, 44);
    T *da, *dx, *dout;
    CUDA_CHECK(cudaMalloc(&da, N * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), N * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    glass::body b = glass::dispatch_body(glass::op::gemv, N, sizeof(T) == 8);
    for (int tb : TBS) {
        std::vector<T> r_bare(N), r_blk(N);
        k_gemv<T, N, true><<<1, tb>>>(da, dx, dout);
        CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, N * sizeof(T), cudaMemcpyDeviceToHost));
        k_gemv<T, N, false><<<1, tb>>>(da, dx, dout);
        CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, N * sizeof(T), cudaMemcpyDeviceToHost));
        report(cell, b, tb, r_bare, r_blk);
    }
    cudaFree(da); cudaFree(dx); cudaFree(dout);
}

template <typename T, uint32_t N>
static void drive_chol(const char* cell) {
    auto ha = spd<T>(N, 55);
    T *da, *dout;
    CUDA_CHECK(cudaMalloc(&da, N * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, N * N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), N * N * sizeof(T), cudaMemcpyHostToDevice));
    glass::body b = glass::dispatch_body(glass::op::chol, N, sizeof(T) == 8);
    for (int tb : TBS) {
        std::vector<T> r_bare(N * N), r_blk(N * N);
        k_chol<T, N, true><<<1, tb>>>(da, dout);
        CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, N * N * sizeof(T), cudaMemcpyDeviceToHost));
        k_chol<T, N, false><<<1, tb>>>(da, dout);
        CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, N * N * sizeof(T), cudaMemcpyDeviceToHost));
        report(cell, b, tb, r_bare, r_blk);
    }
    cudaFree(da); cudaFree(dout);
}

template <typename T, uint32_t N>
static void drive_trsv(const char* cell) {
    auto ha = lower<T>(N, 66);
    auto hb = randv<T>(N, 77);
    T *da, *db, *dout;
    CUDA_CHECK(cudaMalloc(&da, N * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&db, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), N * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    glass::body b = glass::dispatch_body(glass::op::trsv, N, sizeof(T) == 8);
    for (int tb : TBS) {
        std::vector<T> r_bare(N), r_blk(N);
        k_trsv<T, N, true><<<1, tb>>>(da, db, dout);
        CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, N * sizeof(T), cudaMemcpyDeviceToHost));
        k_trsv<T, N, false><<<1, tb>>>(da, db, dout);
        CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, N * sizeof(T), cudaMemcpyDeviceToHost));
        report(cell, b, tb, r_bare, r_blk);
    }
    cudaFree(da); cudaFree(db); cudaFree(dout);
}

template <typename T, uint32_t N>
static void drive_posv(const char* cell) {
    auto ha = spd<T>(N, 88);
    auto hb = randv<T>(N, 99);
    T *da, *db, *dout;
    CUDA_CHECK(cudaMalloc(&da, N * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&db, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), N * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    glass::body b = glass::dispatch_body(glass::op::posv, N, sizeof(T) == 8);
    for (int tb : TBS) {
        std::vector<T> r_bare(N), r_blk(N);
        k_posv<T, N, true><<<1, tb>>>(da, db, dout);
        CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, N * sizeof(T), cudaMemcpyDeviceToHost));
        k_posv<T, N, false><<<1, tb>>>(da, db, dout);
        CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, N * sizeof(T), cudaMemcpyDeviceToHost));
        report(cell, b, tb, r_bare, r_blk);
    }
    cudaFree(da); cudaFree(db); cudaFree(dout);
}

template <typename T>
static void drive_eig3(const char* cell) {
    // symmetric 3x3
    auto m = randv<T>(9, 123);
    std::vector<T> ha(9);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ha[i + 3 * j] = T(0.5) * (m[i + 3 * j] + m[j + 3 * i]) + (i == j ? T(2) : T(0));
    T *da, *dout;
    CUDA_CHECK(cudaMalloc(&da, 9 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, 12 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), 9 * sizeof(T), cudaMemcpyHostToDevice));
    glass::body b = glass::dispatch_body(glass::op::eig3, 3, sizeof(T) == 8);
    for (int tb : TBS) {
        std::vector<T> r_bare(12), r_blk(12);
        k_eig3<T, true><<<1, tb>>>(da, dout);
        CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, 12 * sizeof(T), cudaMemcpyDeviceToHost));
        k_eig3<T, false><<<1, tb>>>(da, dout);
        CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, 12 * sizeof(T), cudaMemcpyDeviceToHost));
        report(cell, b, tb, r_bare, r_blk);
    }
    cudaFree(da); cudaFree(dout);
}

template <typename T>
static void drive_softmax() {
    // n=16: the moved f32 cell; n=64: beyond the measured bound -> block body.
    for (uint32_t n : {16u, 64u}) {
        auto hx = randv<T>(n, 321);
        T *dx, *dout;
        CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dout, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        glass::body b = glass::dispatch_body(glass::op::softmax, n, sizeof(T) == 8);
        size_t smem = n * sizeof(T);
        char cell[32];
        std::snprintf(cell, sizeof(cell), "softmax_n%u", n);
        for (int tb : TBS) {
            std::vector<T> r_bare(n), r_blk(n);
            k_softmax<T, true><<<1, tb, smem>>>(n, dx, dout);
            CUDA_CHECK(cudaMemcpy(r_bare.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
            k_softmax<T, false><<<1, tb, smem>>>(n, dx, dout);
            CUDA_CHECK(cudaMemcpy(r_blk.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
            report(cell, b, tb, r_bare, r_blk);
        }
        cudaFree(dx); cudaFree(dout);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <op>\n", argv[0]); return 2; }
    const char* op = argv[1];
    if      (!std::strcmp(op, "dot_f32"))  { drive_dot<float, 8>("dot_n8_f32");
                                             drive_dot<float, 32>("dot_n32_f32");
                                             drive_dot<float, 64>("dot_n64_f32"); }
    else if (!std::strcmp(op, "dot_f64"))  { drive_dot<double, 8>("dot_n8_f64");
                                             drive_dot<double, 32>("dot_n32_f64"); }
    else if (!std::strcmp(op, "gemv_f32")) { drive_gemv<float, 16>("gemv_n16_f32"); }
    else if (!std::strcmp(op, "chol_f32")) { drive_chol<float, 4>("chol_n4_f32"); }
    else if (!std::strcmp(op, "trsv_f32")) { drive_trsv<float, 4>("trsv_n4_f32");
                                             drive_trsv<float, 16>("trsv_n16_f32"); }
    else if (!std::strcmp(op, "posv_f32")) { drive_posv<float, 4>("posv_n4_f32");
                                             drive_posv<float, 32>("posv_n32_f32"); }
    else if (!std::strcmp(op, "eig3_f64")) { drive_eig3<double>("eig3_f64"); }
    else if (!std::strcmp(op, "softmax_f32")) { drive_softmax<float>(); }
    else { std::fprintf(stderr, "unknown op %s\n", op); return 2; }
    CUDA_CHECK(cudaDeviceSynchronize());
    return g_fail;
}
