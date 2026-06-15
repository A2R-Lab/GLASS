// test_qp.cu — runner for the internal box-constrained QP solver.
//
// Usage:
//   test_qp box_qp <f32|f64> <n> <threads> <max_iter> <tol> <P.bin> <q.bin> <l.bin> <u.bin> <x0.bin>
//
// Reads column-major P (n*n) and vectors q,l,u,x0 (n) of the chosen dtype from
// raw binary files, runs glass::internal::box_qp in one block of <threads>, then
// prints two lines to stdout:
//   line 1: the solution x (n values)
//   line 2: "<converged> <iters> <grad_norm>"
//
// Unlike the shared L1/L2/L3 runners, this one takes the block thread count as
// an argument (so tests can sweep it for thread-count invariance) and supports
// both float32 and float64 (the solver is sensitive to precision).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "../../src/L3/box_qp.cuh"

// --- local dtype-generic IO helpers (helpers.cuh is float32-only) -----------
template <typename T>
static T *read_device_vec(const char *path, int n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    T *h = (T *)malloc(n * sizeof(T));
    if (fread(h, sizeof(T), n, f) != (size_t)n) {
        fprintf(stderr, "short read from %s\n", path); exit(1);
    }
    fclose(f);
    T *d;
    cudaMalloc(&d, n * sizeof(T));
    cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice);
    free(h);
    return d;
}

template <typename T>
static void print_host_vec(const T *h, int n)
{
    for (int i = 0; i < n; i++) {
        printf("%.10g", (double)h[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
}

// --- kernel -----------------------------------------------------------------
template <typename T>
__global__ void run_box_qp(std::uint32_t n, T *P, T *q, T *l, T *u, T *x,
                           T *scratch, T *info, int max_iter, T tol)
{
    glass::internal::QPParams<T> params;
    params.max_iter = max_iter;
    params.tol = tol;
    glass::internal::QPResult<T> r =
        glass::internal::box_qp<T>(n, P, q, l, u, x, scratch, params);
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        info[0] = r.converged ? T(1) : T(0);
        info[1] = (T)r.iters;
        info[2] = r.grad_norm;
    }
}

template <typename T>
static int run(int n, int threads, int max_iter, double tol,
               const char *pP, const char *pq, const char *pl,
               const char *pu, const char *px0)
{
    T *dP = read_device_vec<T>(pP, n * n);
    T *dq = read_device_vec<T>(pq, n);
    T *dl = read_device_vec<T>(pl, n);
    T *du = read_device_vec<T>(pu, n);
    T *dx = read_device_vec<T>(px0, n);

    T *dscratch, *dinfo;
    cudaMalloc(&dscratch, glass::internal::box_qp_scratch_size<T>(n) * sizeof(T));
    cudaMalloc(&dinfo, 3 * sizeof(T));

    run_box_qp<T><<<1, threads>>>(n, dP, dq, dl, du, dx, dscratch, dinfo,
                                  max_iter, (T)tol);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    T hx[1024], hinfo[3];
    cudaMemcpy(hx, dx, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hinfo, dinfo, 3 * sizeof(T), cudaMemcpyDeviceToHost);
    print_host_vec<T>(hx, n);         // line 1: solution
    print_host_vec<T>(hinfo, 3);      // line 2: converged iters grad_norm

    cudaFree(dP); cudaFree(dq); cudaFree(dl); cudaFree(du); cudaFree(dx);
    cudaFree(dscratch); cudaFree(dinfo);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 12) {
        fprintf(stderr,
            "Usage: %s box_qp <f32|f64> <n> <threads> <max_iter> <tol> "
            "<P> <q> <l> <u> <x0>\n", argv[0]);
        return 1;
    }
    const char *op = argv[1];
    if (strcmp(op, "box_qp") != 0) { fprintf(stderr, "unknown op %s\n", op); return 1; }
    const char *dtype = argv[2];
    int   n        = atoi(argv[3]);
    int   threads  = atoi(argv[4]);
    int   max_iter = atoi(argv[5]);
    double tol     = atof(argv[6]);
    const char *pP = argv[7], *pq = argv[8], *pl = argv[9], *pu = argv[10], *px0 = argv[11];

    if (n > 1024) { fprintf(stderr, "n too large for this runner\n"); return 1; }

    if (strcmp(dtype, "f64") == 0)
        return run<double>(n, threads, max_iter, tol, pP, pq, pl, pu, px0);
    else
        return run<float>(n, threads, max_iter, tol, pP, pq, pl, pu, px0);
}
