// test_l1_round2.cu — round-2 L1 ops: nrm1_diff (‖x−y‖₁), warp norms (asum/nrm2),
// and the row-strided AXPY/COPY block movers. All take a runtime <threads> arg so
// the Python side can sweep block sizes (incl. partial/odd/non-warp-boundary).
//
// Scalar-reduction ops print the single result on one line.
// Strided ops print the FULL Y buffer (Y_RS*N elements, column-major) on one line.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

// ─── nrm1_diff ────────────────────────────────────────────────────────────────
__global__ void k_nrm1_lm(uint32_t n, const float* x, const float* y, float* out) {
    glass::nrm1_diff_lowmem<float>(n, x, y, out);            // result in out[0]
}
__global__ void k_nrm1_hs(uint32_t n, const float* x, const float* y, float* out, float* scr) {
    glass::nrm1_diff_fast<float>(n, x, y, out, scr);        // result in out[0]
}
__global__ void k_nrm1_warp(uint32_t n, const float* x, const float* y, float* out) {
    float r = glass::warp::nrm1_diff<float>(n, x, y);
    if (((threadIdx.x) & 31) == 0) out[0] = r;                     // every lane has r; lane 0 writes
}

// ─── warp norms ───────────────────────────────────────────────────────────────
__global__ void k_warp_asum(uint32_t n, const float* x, float* out) {
    float r = glass::warp::asum<float>(n, x);
    if (((threadIdx.x) & 31) == 0) out[0] = r;
}
__global__ void k_warp_nrm2(uint32_t n, const float* x, float* out) {
    float r = glass::warp::nrm2<float>(n, x);
    if (((threadIdx.x) & 31) == 0) out[0] = r;
}

// ─── row-strided AXPY / COPY (compile-time shapes) ────────────────────────────
template <uint32_t M, uint32_t N, uint32_t YRS, uint32_t XRS, bool WARP, bool COPY>
__global__ void k_rs(float alpha, const float* X, float* Y) {
    if constexpr (WARP) {
        if constexpr (COPY) glass::warp::copy_strided<float, M, N, YRS, XRS>(alpha, X, Y);
        else                glass::warp::axpy_strided<float, M, N, YRS, XRS>(alpha, X, Y);
    } else {
        if constexpr (COPY) glass::copy_strided<float, M, N, YRS, XRS>(alpha, X, Y);
        else                glass::axpy_strided<float, M, N, YRS, XRS>(alpha, X, Y);
    }
}

// Shape table (id : M, N, Y_RS, X_RS). id 0 = PDDP acceptance (14×14 into 21-lead).
#define DISPATCH_SHAPE(id, WARP, COPY, alpha, dX, dY)                                  \
    do {                                                                               \
        if      (id == 0) k_rs<14,14,21,14,WARP,COPY><<<1,THREADS>>>(alpha,dX,dY);      \
        else if (id == 1) k_rs< 6, 5, 8, 6,WARP,COPY><<<1,THREADS>>>(alpha,dX,dY);      \
        else if (id == 2) k_rs< 4, 4, 4, 4,WARP,COPY><<<1,THREADS>>>(alpha,dX,dY);      \
        else { fprintf(stderr, "bad shape id %d\n", id); return 1; }                    \
    } while (0)

static void shape_dims(int id, int* M, int* N, int* YRS, int* XRS) {
    if (id == 0) { *M=14; *N=14; *YRS=21; *XRS=14; }
    else if (id == 1) { *M=6; *N=5; *YRS=8; *XRS=6; }
    else { *M=4; *N=4; *YRS=4; *XRS=4; }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <op> <threads> ...\n", argv[0]); return 1; }
    const char* op = argv[1];
    int THREADS = atoi(argv[2]);

    // Scalar reductions: <op> <threads> <n> <x.bin> [<y.bin>]
    if (strncmp(op, "nrm1_", 5) == 0 || strncmp(op, "warp_", 5) == 0) {
        int n = atoi(argv[3]);
        bool needs_y = (strncmp(op, "nrm1_", 5) == 0);
        float* dx = read_device_vec(argv[4], n);
        float* dy = needs_y ? read_device_vec(argv[5], n) : nullptr;
        float* out = alloc_device_vec(n);  // low_memory needs n; others use out[0]
        if      (strcmp(op, "nrm1_lm")   == 0) k_nrm1_lm<<<1,THREADS>>>(n, dx, dy, out);
        else if (strcmp(op, "nrm1_hs")   == 0) {
            float* scr; cudaMalloc(&scr, ((THREADS+31)/32)*sizeof(float));
            k_nrm1_hs<<<1,THREADS>>>(n, dx, dy, out, scr);
        }
        else if (strcmp(op, "nrm1_warp") == 0) k_nrm1_warp<<<1,THREADS>>>(n, dx, dy, out);
        else if (strcmp(op, "warp_asum") == 0) k_warp_asum<<<1,THREADS>>>(n, dx, out);
        else if (strcmp(op, "warp_nrm2")==0) k_warp_nrm2<<<1,THREADS>>>(n, dx, out);
        else { fprintf(stderr, "bad op %s\n", op); return 1; }
        cudaDeviceSynchronize();
        print_device_vec(out, 1);
        return 0;
    }

    // Strided movers: <op> <threads> <shape_id> <alpha> <X.bin> <Y.bin>
    // op ∈ {rsaxpy, rscopy, rsaxpy_warp, rscopy_warp}. Prints full Y (Y_RS*N).
    if (strncmp(op, "rs", 2) == 0) {
        int id = atoi(argv[3]);
        float alpha = (float)atof(argv[4]);
        int M,N,YRS,XRS; shape_dims(id, &M, &N, &YRS, &XRS);
        float* dX = read_device_vec(argv[5], XRS * N);
        float* dY = read_device_vec(argv[6], YRS * N);
        bool warp = (strstr(op, "_warp") != nullptr);
        bool copy = (strncmp(op, "rscopy", 6) == 0);
        if      (!warp && !copy) DISPATCH_SHAPE(id, false, false, alpha, dX, dY);
        else if (!warp &&  copy) DISPATCH_SHAPE(id, false, true,  alpha, dX, dY);
        else if ( warp && !copy) DISPATCH_SHAPE(id, true,  false, alpha, dX, dY);
        else                     DISPATCH_SHAPE(id, true,  true,  alpha, dX, dY);
        cudaDeviceSynchronize();
        print_device_vec(dY, YRS * N);
        return 0;
    }

    fprintf(stderr, "bad op %s\n", op);
    return 1;
}
