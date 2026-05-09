// test_l1.cu — dispatch L1 GLASS operations and print float32 results to stdout
// Usage: ./test_l1 <op> <cg|simple|simple_lm|simple_hs> <n> [extra args] [input.bin ...]
//
// Versions:
//   cg        — glass::cgrps:: (cooperative groups)
//   simple    — glass::        (threadIdx, default)
//   simple_lm — glass::low_memory::
//   simple_hs — glass::high_speed::

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
    glass::low_memory::dot(n, x, y, out);
}
__global__ void k_dot_simple_hs(int n, float* x, float* y, float* out, float* scratch) {
    glass::high_speed::dot(n, x, y, out, scratch);
}

__global__ void k_reduce_cg(int n, float* x) { glass::cgrps::reduce(n, x); }
__global__ void k_reduce_simple_lm(int n, float* x) { glass::low_memory::reduce(n, x); }
__global__ void k_reduce_simple_hs(int n, float* x, float* scratch) {
    glass::high_speed::reduce(n, x, scratch);
}

__global__ void k_l2norm_cg(int n, float* x) { glass::cgrps::l2norm(n, x); }
__global__ void k_l2norm_simple_lm(int n, float* x) { glass::low_memory::l2norm(n, x); }
__global__ void k_l2norm_simple_hs(int n, float* x, float* scratch) {
    glass::high_speed::l2norm(n, x, scratch);
}

__global__ void k_infnorm_cg(int n, float* x) { glass::cgrps::infnorm(n, x); }
__global__ void k_infnorm_simple(int n, float* x) { glass::infnorm(n, x); }

__global__ void k_asum_cg(int n, float* x, float* out) { glass::cgrps::asum(n, x, out); }
__global__ void k_asum_simple_lm(int n, float* x, float* out) {
    glass::low_memory::asum(n, x, out);
}
__global__ void k_asum_simple_hs(int n, float* x, float* scratch) {
    glass::high_speed::asum(n, x, scratch);
}

__global__ void k_clip_cg(int n, float* x, float* l, float* u) { glass::cgrps::clip(n, x, l, u); }
__global__ void k_clip_simple(int n, float* x, float* l, float* u) {
    glass::clip(n, x, l, u);
}

__global__ void k_set_const_cg(int n, float alpha, float* x) { glass::cgrps::set_const(n, alpha, x); }
__global__ void k_set_const_simple(int n, float alpha, float* x) {
    glass::set_const(n, alpha, x);
}

__global__ void k_loadIdentity_cg(int n, float* A) { glass::cgrps::loadIdentity(n, A); }
__global__ void k_loadIdentity_simple(int n, float* A) { glass::loadIdentity(n, A); }

__global__ void k_addI_cg(int n, float alpha, float* A) { glass::cgrps::addI(n, A, alpha); }
__global__ void k_addI_simple(int n, float alpha, float* A) { glass::addI(n, A, alpha); }

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

// ─── helpers ─────────────────────────────────────────────────────────────────

static int THREADS = 256;

static bool is_cg(const char* v)     { return strcmp(v, "cg") == 0; }
static bool is_simple(const char* v) { return strcmp(v, "simple") == 0; }
static bool is_lm(const char* v)     { return strcmp(v, "simple_lm") == 0; }
static bool is_hs(const char* v)     { return strcmp(v, "simple_hs") == 0; }

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <op> <cg|simple|simple_lm|simple_hs> <n> [args...] [files...]\n", argv[0]);
        return 1;
    }
    const char* op  = argv[1];
    const char* ver = argv[2];
    int n = atoi(argv[3]);

    float* d_scratch = alloc_device_vec(32);

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
        } else {
            k_reduce_simple_hs<<<1, THREADS>>>(n, dx, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dx, 1);

    } else if (strcmp(op, "l2norm") == 0) {
        float* dx = read_device_vec(argv[4], n);
        if (is_cg(ver)) {
            k_l2norm_cg<<<1, THREADS>>>(n, dx);
        } else if (is_lm(ver)) {
            k_l2norm_simple_lm<<<1, THREADS>>>(n, dx);
        } else {
            k_l2norm_simple_hs<<<1, THREADS>>>(n, dx, d_scratch);
        }
        cudaDeviceSynchronize();
        print_device_vec(dx, 1);

    } else if (strcmp(op, "infnorm") == 0) {
        float* dx = read_device_vec(argv[4], n);
        if (is_cg(ver))  k_infnorm_cg<<<1, THREADS>>>(n, dx);
        else             k_infnorm_simple<<<1, THREADS>>>(n, dx);
        cudaDeviceSynchronize();
        print_device_vec(dx, 1);

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

    } else if (strcmp(op, "loadIdentity") == 0) {
        float* dA = alloc_device_vec(n * n);
        if (is_cg(ver))  k_loadIdentity_cg<<<1, THREADS>>>(n, dA);
        else             k_loadIdentity_simple<<<1, THREADS>>>(n, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "addI") == 0) {
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

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    cudaFree(d_scratch);
    return 0;
}
