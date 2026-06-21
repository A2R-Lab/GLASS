// test_trsv.cu — dispatch glass::trsv / glass::trmv and print the result.
// Usage: ./test_trsv <trsv|trmv> <threads> <n> <lower> <unit> <trans> <A.bin> <x.bin>
//   A.bin : n*n floats (column-major triangular matrix)
//   x.bin : n   floats (rhs for trsv / input vector for trmv)
//   lower/unit/trans : 0 or 1
// trsv solves op(A) x = b in place; trmv computes x = op(A) x in place.
// Prints one line: the resulting x vector.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

// ── trsv kernels (templated over the 3 flags) ─────────────────────────────────
template <bool LOWER, bool UNIT, bool TRANS>
__global__ void k_trsv(uint32_t n, const float* A, float* x) {
    glass::trsv<float, LOWER, UNIT, TRANS>(n, A, x);
}

// ── trmv in-place kernels (templated over the 3 flags) ────────────────────────
template <bool LOWER, bool UNIT, bool TRANS>
__global__ void k_trmv(uint32_t n, const float* A, float* x) {
    extern __shared__ float scratch[];
    glass::trmv<float, LOWER, UNIT, TRANS>(n, A, x, scratch);
}

int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr,
            "Usage: %s <trsv|trmv> <threads> <n> <lower> <unit> <trans> "
            "<A.bin> <x.bin>\n", argv[0]);
        return 1;
    }
    const char* op    = argv[1];
    int threads       = atoi(argv[2]);
    int n             = atoi(argv[3]);
    int lower         = atoi(argv[4]);
    int unit          = atoi(argv[5]);
    int trans         = atoi(argv[6]);
    const char* A_path = argv[7];
    const char* x_path = argv[8];

    float* dA = read_device_vec(A_path, n * n);
    float* dx = read_device_vec(x_path, n);

    bool is_trmv = (strcmp(op, "trmv") == 0);

    // dispatch over the 8 (lower,unit,trans) flag combos.
#define DISPATCH(L, U, TR)                                                       \
    if (lower == (L) && unit == (U) && trans == (TR)) {                          \
        if (is_trmv)                                                             \
            k_trmv<(L), (U), (TR)><<<1, threads, n * sizeof(float)>>>(           \
                (uint32_t)n, dA, dx);                                            \
        else                                                                     \
            k_trsv<(L), (U), (TR)><<<1, threads>>>((uint32_t)n, dA, dx);         \
    }

    DISPATCH(0, 0, 0)
    else DISPATCH(0, 0, 1)
    else DISPATCH(0, 1, 0)
    else DISPATCH(0, 1, 1)
    else DISPATCH(1, 0, 0)
    else DISPATCH(1, 0, 1)
    else DISPATCH(1, 1, 0)
    else DISPATCH(1, 1, 1)
    else { fprintf(stderr, "bad flags\n"); return 1; }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    print_device_vec(dx, n);
    return 0;
}
