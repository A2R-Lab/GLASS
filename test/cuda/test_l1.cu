// test_l1.cu — dispatch L1 GLASS operations and print float32 results to stdout
// Usage: ./test_l1 <op> <cg|simple|simple_lm|simple_hs|warp> <threads> <n> [extra args] [input.bin ...]
//
// <threads> is the RUNTIME block size (launched as <<<1, threads>>>) so the
// Python driver can sweep the canonical thread counts. warp:: ops ignore it
// and always launch one 32-lane warp.
//
// Versions:
//   cg        — glass::cgrps:: (cooperative groups)
//   simple    — glass::        (threadIdx, default)
//   simple_lm — glass::*_lowmem
//   simple_hs — glass::*_fast
//   warp      — glass::warp::  (single 32-lane warp)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

// ─── kernel wrappers ─────────────────────────────────────────────────────────

__global__ void k_axpy_cg(int n, float alpha, float* x, float* y) {
    glass::cgrps::axpy(n, alpha, x, y);
}
__global__ void k_axpy_simple(int n, float alpha, float* x, float* y) {
    glass::axpy(n, alpha, x, y);
}
__global__ void k_axpy3_cg(int n, float alpha, float* x, float* y, float* z) {
    glass::cgrps::axpy(n, alpha, x, y, z);
}
__global__ void k_axpy3_simple(int n, float alpha, float* x, float* y, float* z) {
    glass::axpy(n, alpha, x, y, z);
}
__global__ void k_axpby_cg(int n, float alpha, float* x, float beta, float* y, float* z) {
    glass::cgrps::axpby(n, alpha, x, beta, y, z);
}
__global__ void k_axpby_simple(int n, float alpha, float* x, float beta, float* y, float* z) {
    glass::axpby(n, alpha, x, beta, y, z);
}

__global__ void k_copy_cg(int n, float* x, float* y) { glass::cgrps::copy(n, x, y); }
__global__ void k_copy_simple(int n, float* x, float* y) { glass::copy(n, x, y); }

__global__ void k_scal_cg(int n, float alpha, float* x) { glass::cgrps::scal(n, alpha, x); }
__global__ void k_scal_simple(int n, float alpha, float* x) { glass::scal(n, alpha, x); }

__global__ void k_swap_cg(int n, float* x, float* y) { glass::cgrps::swap(n, x, y); }
__global__ void k_swap_simple(int n, float* x, float* y) { glass::swap(n, x, y); }

__global__ void k_dot_cg(int n, float* x, float* y) {
    glass::cgrps::dot(n, x, y);
}
__global__ void k_dot_simple_lm(int n, float* x, float* y, float* out) {
    glass::dot_lowmem(n, x, y, out);
}
__global__ void k_dot_simple_hs(int n, float* x, float* y, float* out, float* scratch) {
    glass::dot_fast(n, x, y, out, scratch);
}

__global__ void k_reduce_cg(int n, float* x) { glass::cgrps::reduce(n, x); }
__global__ void k_reduce_simple_lm(int n, float* x) { glass::reduce_lowmem(n, x); }
__global__ void k_reduce_simple_hs(int n, float* x, float* scratch) {
    glass::reduce_fast(n, x, scratch);
}
// Single-warp reduce (launch <<<1,32>>>): raw __shfl, no scratch, no inter-warp combine.
__global__ void k_reduce_warp(int n, float* x) { glass::warp::reduce(n, x); }
// Single-warp register-partial reduce: each lane forms a strided partial, gets the warp total back.
__global__ void k_reduce_partial_warp(int n, float* x, float* out) {
    uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
    float partial = 0.0f;
    for (uint32_t i = lane; i < (uint32_t)n; i += 32) partial += x[i];
    float total = glass::warp::reduce(partial);
    out[0] = total;   // broadcast check: every lane holds the same `total`
}
// Register-partial -> block-sum overload: each thread forms a per-thread partial
// (here a strided slice of x, mirroring a cost/barrier kernel's per-thread term),
// passes it directly to reduce(partial, scratch), and gets the block total back.
// We write the returned total from EVERY thread to verify the broadcast (out[0]
// is the last writer's value; all threads hold the same total).
__global__ void k_reduce_partial_hs(int n, float* x, float* out, float* scratch) {
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    float partial = 0.0f;
    for (uint32_t i = rank; i < (uint32_t)n; i += size) partial += x[i];
    float total = glass::reduce_fast(partial, scratch);
    out[0] = total;   // broadcast check: every thread holds the same `total`
}

// Register-partial -> block-MIN overload (reduce_fast_min): the min twin of
// k_reduce_partial_hs. Each thread folds a strided slice of x to a per-thread
// minimum, passes it to reduce_fast_min(partial, scratch), and gets the block
// minimum back. Threads with no element seed with +inf (the min identity).
// Written from EVERY thread to verify the broadcast.
__global__ void k_reduce_min_partial_hs(int n, float* x, float* out, float* scratch) {
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    float partial = INFINITY;
    for (uint32_t i = rank; i < (uint32_t)n; i += size) partial = fminf(partial, x[i]);
    float total = glass::reduce_fast_min(partial, scratch);
    out[0] = total;   // broadcast check: every thread holds the same `total`
}

__global__ void k_nrm2_cg(int n, float* x) { glass::cgrps::nrm2(n, x); }
__global__ void k_nrm2_simple_lm(int n, float* x) { glass::nrm2_lowmem(n, x); }
__global__ void k_nrm2_simple_hs(int n, float* x, float* scratch) {
    glass::nrm2_fast(n, x, scratch);
}

__global__ void k_infnorm_cg(int n, float* x) { glass::cgrps::infnorm(n, x); }
__global__ void k_infnorm_simple(int n, float* x) { glass::infnorm(n, x); }

// ── vector_norm family (non-destructive ‖a‖₂ into out[0]; a untouched) ───────
__global__ void k_vector_norm_simple(int n, float* a, float* out) {
    glass::vector_norm(n, a, out);
}
__global__ void k_vector_norm_simple_lm(int n, float* a, float* out) {
    glass::vector_norm_lowmem(n, a, out);
}
__global__ void k_vector_norm_simple_hs(int n, float* a, float* out, float* scratch) {
    glass::vector_norm_fast(n, a, out, scratch);
}

__global__ void k_asum_cg(int n, float* x, float* out) { glass::cgrps::asum(n, x, out); }
__global__ void k_asum_simple_lm(int n, float* x, float* out) {
    glass::asum_lowmem(n, x, out);
}
__global__ void k_asum_simple_hs(int n, float* x, float* scratch) {
    glass::asum_fast(n, x, scratch);
}

__global__ void k_clip_cg(int n, float* x, float* l, float* u) { glass::cgrps::clip(n, x, l, u); }
__global__ void k_clip_simple(int n, float* x, float* l, float* u) {
    glass::clip(n, x, l, u);
}

__global__ void k_set_const_cg(int n, float alpha, float* x) { glass::cgrps::set_const(n, alpha, x); }
__global__ void k_set_const_simple(int n, float alpha, float* x) {
    glass::set_const(n, alpha, x);
}

__global__ void k_loadIdentity_cg(int n, float* A) { glass::cgrps::set_identity(n, A); }
__global__ void k_loadIdentity_simple(int n, float* A) { glass::set_identity(n, A); }

__global__ void k_addI_cg(int n, float alpha, float* A) { glass::cgrps::add_identity(n, A, alpha); }
__global__ void k_addI_simple(int n, float alpha, float* A) { glass::add_identity(n, A, alpha); }

__global__ void k_transpose_cg(int N, int M, float* a, float* b) {
    glass::cgrps::transpose(N, M, a, b);
}
__global__ void k_transpose_simple(int N, int M, float* a, float* b) {
    glass::transpose(N, M, a, b);
}

__global__ void k_elementwise_add_cg(int n, float* a, float* b, float* c) {
    glass::cgrps::elementwise_add(n, a, b, c);
}
__global__ void k_elementwise_add_simple(int n, float* a, float* b, float* c) {
    glass::elementwise_add(n, a, b, c);
}

__global__ void k_elementwise_sub_cg(int n, float* a, float* b, float* c) {
    glass::cgrps::elementwise_sub(n, a, b, c);
}
__global__ void k_elementwise_sub_simple(int n, float* a, float* b, float* c) {
    glass::elementwise_sub(n, a, b, c);
}

__global__ void k_elementwise_mult_cg(int n, float* a, float* b, float* c) {
    glass::cgrps::elementwise_mult(n, a, b, c);
}
__global__ void k_elementwise_mult_simple(int n, float* a, float* b, float* c) {
    glass::elementwise_mult(n, a, b, c);
}

__global__ void k_elementwise_abs_cg(int n, float* a, float* b) {
    glass::cgrps::elementwise_abs(n, a, b);
}
__global__ void k_elementwise_abs_simple(int n, float* a, float* b) {
    glass::elementwise_abs(n, a, b);
}

__global__ void k_elementwise_max_cg(int n, float* a, float* b, float* c) {
    glass::cgrps::elementwise_max(n, a, b, c);
}
__global__ void k_elementwise_max_simple(int n, float* a, float* b, float* c) {
    glass::elementwise_max(n, a, b, c);
}

__global__ void k_elementwise_min_cg(int n, float* a, float* b, float* c) {
    glass::cgrps::elementwise_min(n, a, b, c);
}
__global__ void k_elementwise_min_simple(int n, float* a, float* b, float* c) {
    glass::elementwise_min(n, a, b, c);
}

// ── comparison / logic / scalar elementwise (cg + simple) ───────────────────
#define EW3(NAME) \
  __global__ void k_##NAME##_cg(int n, float* a, float* b, float* c)     { glass::cgrps::NAME(n,a,b,c); } \
  __global__ void k_##NAME##_simple(int n, float* a, float* b, float* c) { glass::NAME(n,a,b,c); }
EW3(elementwise_less_than)
EW3(elementwise_more_than)
EW3(elementwise_less_than_or_eq)
EW3(elementwise_and)
__global__ void k_elementwise_not_cg(int n, float* a, float* c)     { glass::cgrps::elementwise_not(n,a,c); }
__global__ void k_elementwise_not_simple(int n, float* a, float* c) { glass::elementwise_not(n,a,c); }
#define EWS(NAME) \
  __global__ void k_##NAME##_cg(int n, float* a, float b, float* c)     { glass::cgrps::NAME(n,a,b,c); } \
  __global__ void k_##NAME##_simple(int n, float* a, float b, float* c) { glass::NAME(n,a,b,c); }
EWS(elementwise_mult_scalar)
EWS(elementwise_max_scalar)
EWS(elementwise_min_scalar)
// less_than_scalar has no cgrps variant — simple only.
__global__ void k_elementwise_less_than_scalar_simple(int n, float* a, float b, float* c) { glass::elementwise_less_than_scalar(n,a,b,c); }

__global__ void k_prefix_sum_excl_cg(int n, float* x, float* out) {
    glass::prefix_sum_exclusive(x, out, n);
}
__global__ void k_prefix_sum_excl_simple(int n, float* x, float* out) {
    glass::prefix_sum_exclusive(x, out, n);
}

__global__ void k_prefix_sum_incl_cg(int n, float* x, float* out) {
    glass::prefix_sum_inclusive(x, out, n);
}
__global__ void k_prefix_sum_incl_simple(int n, float* x, float* out) {
    glass::prefix_sum_inclusive(x, out, n);
}

// ─── dot_strided kernels (compile-time N, SX, SY; per-thread, no reduction) ──
// x has N*SX elements, y has N*SY elements; result = sum(x[i*SX]*y[i*SY]).
// Launch with 1 thread — the operation is intentionally not block-parallel.
#define DEFINE_DOT_STRIDED_KERNEL(N, SX, SY)                                           \
    __global__ void k_dot_strided_##N##_##SX##_##SY(float* x, float* y, float* out) { \
        glass::dot_strided<float, N, SX, SY>(x, y, out);                               \
    }
DEFINE_DOT_STRIDED_KERNEL(4, 4, 1)
DEFINE_DOT_STRIDED_KERNEL(6, 1, 1)
DEFINE_DOT_STRIDED_KERNEL(6, 6, 1)
DEFINE_DOT_STRIDED_KERNEL(6, 6, 6)

// ─── dot_strided_coalesced kernels (block-cooperative, coalesced loads) ──────
// Block-wide reduction sibling of dot_strided: same result, launched <<<1,T>>>.
// Also a reference per-thread dot_strided kernel that writes from thread 0 only,
// so the test can assert numerical equivalence for a large stride.
#define DEFINE_DOT_COALESCED_KERNEL(N, SX, SY)                                          \
    __global__ void k_dot_coalesced_##N##_##SX##_##SY(float* x, float* y, float* out,  \
                                                      float* scratch) {                 \
        glass::dot_strided_coalesced<float, N, SX, SY>(x, y, out, scratch);            \
    }                                                                                    \
    __global__ void k_dot_strided_ref_##N##_##SX##_##SY(float* x, float* y, float* out){\
        uint32_t r = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;\
        float v = glass::dot_strided<float, N, SX, SY>(x, y);                           \
        if (r == 0) *out = v;                                                           \
    }
DEFINE_DOT_COALESCED_KERNEL(64, 64, 64)
DEFINE_DOT_COALESCED_KERNEL(256, 256, 1)

// ─── helpers ─────────────────────────────────────────────────────────────────

static int THREADS = 256;

static bool is_cg(const char* v)     { return strcmp(v, "cg") == 0; }
static bool is_simple(const char* v) { return strcmp(v, "simple") == 0; }
static bool is_lm(const char* v)     { return strcmp(v, "simple_lm") == 0; }
static bool is_hs(const char* v)     { return strcmp(v, "simple_hs") == 0; }
static bool is_warp(const char* v)   { return strcmp(v, "warp") == 0; }

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <op> <cg|simple|simple_lm|simple_hs|warp> <threads> <n> [args...] [files...]\n", argv[0]);
        return 1;
    }
    const char* op  = argv[1];
    const char* ver = argv[2];
    // Runtime block size: argv[3] is <threads>. Splice it out of argv so every
    // op handler below keeps its historical argument indices (n at argv[3], ...).
    THREADS = atoi(argv[3]);
    if (THREADS < 1) { fprintf(stderr, "bad thread count %d\n", THREADS); return 1; }
    for (int i = 3; i + 1 < argc; i++) argv[i] = argv[i + 1];
    argc--;
    int n = atoi(argv[3]);

    // Inter-warp scratch for the *_fast reductions and dot_strided_coalesced:
    // ceil(threads/32) elements of T (see glass::reduce_fast_scratch_bytes).
    float* d_scratch;
    cudaMalloc(&d_scratch, glass::reduce_fast_scratch_bytes<float>(THREADS));
    cudaMemset(d_scratch, 0, glass::reduce_fast_scratch_bytes<float>(THREADS));

    if (strcmp(op, "axpy") == 0) {
        float alpha = atof(argv[4]);
        float* dx = read_device_vec(argv[5], n);
        float* dy = read_device_vec(argv[6], n);
        if (is_cg(ver))     k_axpy_cg<<<1, THREADS>>>(n, alpha, dx, dy);
        else                k_axpy_simple<<<1, THREADS>>>(n, alpha, dx, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, n);

    } else if (strcmp(op, "axpby") == 0) {
        float alpha = atof(argv[4]);
        float beta  = atof(argv[5]);
        float* dx = read_device_vec(argv[6], n);
        float* dy = read_device_vec(argv[7], n);
        float* dz = alloc_device_vec(n);
        if (is_cg(ver))  k_axpby_cg<<<1, THREADS>>>(n, alpha, dx, beta, dy, dz);
        else             k_axpby_simple<<<1, THREADS>>>(n, alpha, dx, beta, dy, dz);
        cudaDeviceSynchronize();
        print_device_vec(dz, n);

    } else if (strcmp(op, "copy") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dy = alloc_device_vec(n);
        if (is_cg(ver))  k_copy_cg<<<1, THREADS>>>(n, dx, dy);
        else             k_copy_simple<<<1, THREADS>>>(n, dx, dy);
        cudaDeviceSynchronize();
        print_device_vec(dy, n);

    } else if (strcmp(op, "scal") == 0) {
        float alpha = atof(argv[4]);
        float* dx = read_device_vec(argv[5], n);
        if (is_cg(ver))  k_scal_cg<<<1, THREADS>>>(n, alpha, dx);
        else             k_scal_simple<<<1, THREADS>>>(n, alpha, dx);
        cudaDeviceSynchronize();
        print_device_vec(dx, n);

    } else if (strcmp(op, "swap") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dy = read_device_vec(argv[5], n);
        if (is_cg(ver))  k_swap_cg<<<1, THREADS>>>(n, dx, dy);
        else             k_swap_simple<<<1, THREADS>>>(n, dx, dy);
        cudaDeviceSynchronize();
        print_device_vec(dx, n);
        print_device_vec(dy, n);

    } else if (strcmp(op, "dot") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dy = read_device_vec(argv[5], n);
        float* dout = alloc_device_vec(n);
        if (is_cg(ver)) {
            k_dot_cg<<<1, THREADS>>>(n, dx, dy);
            cudaDeviceSynchronize();
            print_device_vec(dy, 1);
        } else if (is_lm(ver)) {
            k_dot_simple_lm<<<1, THREADS>>>(n, dx, dy, dout);
            cudaDeviceSynchronize();
            print_device_vec(dout, 1);
        } else {
            k_dot_simple_hs<<<1, THREADS>>>(n, dx, dy, dout, d_scratch);
            cudaDeviceSynchronize();
            print_device_vec(dout, 1);
        }

    } else if (strcmp(op, "reduce") == 0) {
        float* dx = read_device_vec(argv[4], n);
        if (is_cg(ver)) {
            k_reduce_cg<<<1, THREADS>>>(n, dx);
        } else if (is_lm(ver)) {
            k_reduce_simple_lm<<<1, THREADS>>>(n, dx);
        } else if (is_warp(ver)) {
            k_reduce_warp<<<1, 32>>>(n, dx);
        } else {
            k_reduce_simple_hs<<<1, THREADS>>>(n, dx, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dx, 1);

    } else if (strcmp(op, "reduce_partial") == 0) {
        float* dx  = read_device_vec(argv[4], n);
        float* dout = alloc_device_vec(1);
        if (is_warp(ver)) {
            k_reduce_partial_warp<<<1, 32>>>(n, dx, dout);
        } else {
            k_reduce_partial_hs<<<1, THREADS>>>(n, dx, dout, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "reduce_min_partial") == 0) {
        float* dx  = read_device_vec(argv[4], n);
        float* dout = alloc_device_vec(1);
        k_reduce_min_partial_hs<<<1, THREADS>>>(n, dx, dout, d_scratch);
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "nrm2") == 0) {
        float* dx = read_device_vec(argv[4], n);
        if (is_cg(ver)) {
            k_nrm2_cg<<<1, THREADS>>>(n, dx);
        } else if (is_lm(ver)) {
            k_nrm2_simple_lm<<<1, THREADS>>>(n, dx);
        } else {
            k_nrm2_simple_hs<<<1, THREADS>>>(n, dx, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dx, 1);

    } else if (strcmp(op, "infnorm") == 0) {
        float* dx = read_device_vec(argv[4], n);
        if (is_cg(ver))  k_infnorm_cg<<<1, THREADS>>>(n, dx);
        else             k_infnorm_simple<<<1, THREADS>>>(n, dx);
        cudaDeviceSynchronize();
        print_device_vec(dx, 1);

    } else if (strcmp(op, "vector_norm") == 0) {
        // Non-destructive norm: prints out[0] on line 1 and the (untouched)
        // input vector on line 2 so the driver can assert a is not clobbered.
        float* da = read_device_vec(argv[4], n);
        float* dout = alloc_device_vec(n);   // lowmem/default use out as length-n scratch
        if (is_lm(ver)) {
            k_vector_norm_simple_lm<<<1, THREADS>>>(n, da, dout);
        } else if (is_hs(ver)) {
            k_vector_norm_simple_hs<<<1, THREADS>>>(n, da, dout, d_scratch);
        } else {
            k_vector_norm_simple<<<1, THREADS>>>(n, da, dout);
        }
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);
        print_device_vec(da, n);

    } else if (strcmp(op, "asum") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dout = alloc_device_vec(n);
        if (is_cg(ver)) {
            k_asum_cg<<<1, THREADS>>>(n, dx, dout);
            cudaDeviceSynchronize();
            print_device_vec(dout, 1);
        } else if (is_lm(ver)) {
            k_asum_simple_lm<<<1, THREADS>>>(n, dx, dout);
            cudaDeviceSynchronize();
            print_device_vec(dout, 1);
        } else {
            k_asum_simple_hs<<<1, THREADS>>>(n, dx, d_scratch);
            cudaDeviceSynchronize();
            print_device_vec(dx, 1);
        }

    } else if (strcmp(op, "clip") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dl = read_device_vec(argv[5], n);
        float* du = read_device_vec(argv[6], n);
        if (is_cg(ver))  k_clip_cg<<<1, THREADS>>>(n, dx, dl, du);
        else             k_clip_simple<<<1, THREADS>>>(n, dx, dl, du);
        cudaDeviceSynchronize();
        print_device_vec(dx, n);

    } else if (strcmp(op, "set_const") == 0) {
        float alpha = atof(argv[4]);
        float* dx = alloc_device_vec(n);
        if (is_cg(ver))  k_set_const_cg<<<1, THREADS>>>(n, alpha, dx);
        else             k_set_const_simple<<<1, THREADS>>>(n, alpha, dx);
        cudaDeviceSynchronize();
        print_device_vec(dx, n);

    } else if (strcmp(op, "set_identity") == 0) {
        float* dA = alloc_device_vec(n * n);
        if (is_cg(ver))  k_loadIdentity_cg<<<1, THREADS>>>(n, dA);
        else             k_loadIdentity_simple<<<1, THREADS>>>(n, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "add_identity") == 0) {
        float alpha = atof(argv[4]);
        float* dA = read_device_vec(argv[5], n * n);
        if (is_cg(ver))  k_addI_cg<<<1, THREADS>>>(n, alpha, dA);
        else             k_addI_simple<<<1, THREADS>>>(n, alpha, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "transpose") == 0) {
        int m = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], n * m);
        float* dB = alloc_device_vec(n * m);
        if (is_cg(ver))  k_transpose_cg<<<1, THREADS>>>(n, m, dA, dB);
        else             k_transpose_simple<<<1, THREADS>>>(n, m, dA, dB);
        cudaDeviceSynchronize();
        print_device_vec(dB, n * m);

    } else if (strcmp(op, "elementwise_add") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* db = read_device_vec(argv[5], n);
        float* dc = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_add_cg<<<1, THREADS>>>(n, da, db, dc);
        else             k_elementwise_add_simple<<<1, THREADS>>>(n, da, db, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

    } else if (strcmp(op, "elementwise_sub") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* db = read_device_vec(argv[5], n);
        float* dc = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_sub_cg<<<1, THREADS>>>(n, da, db, dc);
        else             k_elementwise_sub_simple<<<1, THREADS>>>(n, da, db, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

    } else if (strcmp(op, "elementwise_mult") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* db = read_device_vec(argv[5], n);
        float* dc = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_mult_cg<<<1, THREADS>>>(n, da, db, dc);
        else             k_elementwise_mult_simple<<<1, THREADS>>>(n, da, db, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

    } else if (strcmp(op, "elementwise_abs") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* db = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_abs_cg<<<1, THREADS>>>(n, da, db);
        else             k_elementwise_abs_simple<<<1, THREADS>>>(n, da, db);
        cudaDeviceSynchronize();
        print_device_vec(db, n);

    } else if (strcmp(op, "elementwise_max") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* db = read_device_vec(argv[5], n);
        float* dc = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_max_cg<<<1, THREADS>>>(n, da, db, dc);
        else             k_elementwise_max_simple<<<1, THREADS>>>(n, da, db, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

    } else if (strcmp(op, "elementwise_min") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* db = read_device_vec(argv[5], n);
        float* dc = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_min_cg<<<1, THREADS>>>(n, da, db, dc);
        else             k_elementwise_min_simple<<<1, THREADS>>>(n, da, db, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

#define EW3_DISPATCH(NAME) \
    } else if (strcmp(op, #NAME) == 0) { \
        float* da = read_device_vec(argv[4], n); \
        float* db = read_device_vec(argv[5], n); \
        float* dc = alloc_device_vec(n); \
        if (is_cg(ver))  k_##NAME##_cg<<<1, THREADS>>>(n, da, db, dc); \
        else             k_##NAME##_simple<<<1, THREADS>>>(n, da, db, dc); \
        cudaDeviceSynchronize(); \
        print_device_vec(dc, n);
    EW3_DISPATCH(elementwise_less_than)
    EW3_DISPATCH(elementwise_more_than)
    EW3_DISPATCH(elementwise_less_than_or_eq)
    EW3_DISPATCH(elementwise_and)

    } else if (strcmp(op, "elementwise_not") == 0) {
        float* da = read_device_vec(argv[4], n);
        float* dc = alloc_device_vec(n);
        if (is_cg(ver))  k_elementwise_not_cg<<<1, THREADS>>>(n, da, dc);
        else             k_elementwise_not_simple<<<1, THREADS>>>(n, da, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

#define EWS_DISPATCH(NAME) \
    } else if (strcmp(op, #NAME) == 0) { \
        float scalar = atof(argv[4]); \
        float* da = read_device_vec(argv[5], n); \
        float* dc = alloc_device_vec(n); \
        if (is_cg(ver))  k_##NAME##_cg<<<1, THREADS>>>(n, da, scalar, dc); \
        else             k_##NAME##_simple<<<1, THREADS>>>(n, da, scalar, dc); \
        cudaDeviceSynchronize(); \
        print_device_vec(dc, n);
    EWS_DISPATCH(elementwise_mult_scalar)
    EWS_DISPATCH(elementwise_max_scalar)
    EWS_DISPATCH(elementwise_min_scalar)

    } else if (strcmp(op, "elementwise_less_than_scalar") == 0) {
        float scalar = atof(argv[4]);
        float* da = read_device_vec(argv[5], n);
        float* dc = alloc_device_vec(n);
        k_elementwise_less_than_scalar_simple<<<1, THREADS>>>(n, da, scalar, dc);
        cudaDeviceSynchronize();
        print_device_vec(dc, n);

    } else if (strcmp(op, "prefix_sum_excl") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dout = alloc_device_vec(n);
        if (is_cg(ver))  k_prefix_sum_excl_cg<<<1, THREADS>>>(n, dx, dout);
        else             k_prefix_sum_excl_simple<<<1, THREADS>>>(n, dx, dout);
        cudaDeviceSynchronize();
        print_device_vec(dout, n);

    } else if (strcmp(op, "prefix_sum_incl") == 0) {
        float* dx = read_device_vec(argv[4], n);
        float* dout = alloc_device_vec(n);
        if (is_cg(ver))  k_prefix_sum_incl_cg<<<1, THREADS>>>(n, dx, dout);
        else             k_prefix_sum_incl_simple<<<1, THREADS>>>(n, dx, dout);
        cudaDeviceSynchronize();
        print_device_vec(dout, n);

    } else if (strcmp(op, "dot_strided_4_4_1") == 0) {
        float* dx = read_device_vec(argv[4], 4 * 4);  // x: N*SX = 16 elements
        float* dy = read_device_vec(argv[5], 4 * 1);  // y: N*SY = 4 elements
        float* dout = alloc_device_vec(1);
        k_dot_strided_4_4_1<<<1, 1>>>(dx, dy, dout);
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "dot_strided_6_1_1") == 0) {
        float* dx = read_device_vec(argv[4], 6 * 1);  // x: 6 elements
        float* dy = read_device_vec(argv[5], 6 * 1);  // y: 6 elements
        float* dout = alloc_device_vec(1);
        k_dot_strided_6_1_1<<<1, 1>>>(dx, dy, dout);
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "dot_strided_6_6_1") == 0) {
        float* dx = read_device_vec(argv[4], 6 * 6);  // x: N*SX = 36 elements
        float* dy = read_device_vec(argv[5], 6 * 1);  // y: N*SY = 6 elements
        float* dout = alloc_device_vec(1);
        k_dot_strided_6_6_1<<<1, 1>>>(dx, dy, dout);
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "dot_strided_6_6_6") == 0) {
        float* dx = read_device_vec(argv[4], 6 * 6);  // x: N*SX = 36 elements
        float* dy = read_device_vec(argv[5], 6 * 6);  // y: N*SY = 36 elements
        float* dout = alloc_device_vec(1);
        k_dot_strided_6_6_6<<<1, 1>>>(dx, dy, dout);
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "dot_coalesced_64_64_64") == 0) {
        float* dx = read_device_vec(argv[4], 64 * 64);
        float* dy = read_device_vec(argv[5], 64 * 64);
        float* dout = alloc_device_vec(1);
        if (is_simple(ver)) {
            k_dot_strided_ref_64_64_64<<<1, THREADS>>>(dx, dy, dout);
        } else {
            k_dot_coalesced_64_64_64<<<1, THREADS>>>(dx, dy, dout, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else if (strcmp(op, "dot_coalesced_256_256_1") == 0) {
        float* dx = read_device_vec(argv[4], 256 * 256);
        float* dy = read_device_vec(argv[5], 256 * 1);
        float* dout = alloc_device_vec(1);
        if (is_simple(ver)) {
            k_dot_strided_ref_256_256_1<<<1, THREADS>>>(dx, dy, dout);
        } else {
            k_dot_coalesced_256_256_1<<<1, THREADS>>>(dx, dy, dout, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dout, 1);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    cudaFree(d_scratch);
    return 0;
}
