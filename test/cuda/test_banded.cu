// test_banded.cu — dispatch glass::banded::bdmv and print float32 results.
// Usage: ./test_banded <bdmv|bdmv_dual> simple <BS> <NBR> <threads> <mat.bin> <vec.bin>
//   mat.bin : NBR * 3*BS*BS floats ([L|D|R] row-major strips, one per block-row)
//   vec.bin : (NBR+2)*BS floats (padded; leading/trailing BS pad blocks zeroed)
// Prints the padded output (one line); bdmv_dual prints two identical lines.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.cuh"
#include "../../glass.cuh"

#define DEFINE_BDMV(BS, NBR)                                                       \
    __global__ void k_bdmv_##BS##_##NBR(float* out, float* mat, float* vec) {     \
        glass::banded::bdmv<float, NBR, BS>(out, mat, vec);                        \
    }                                                                              \
    __global__ void k_bdmv_dual_##BS##_##NBR(float* o1, float* o2,                \
                                             float* mat, float* vec) {            \
        glass::banded::bdmv<float, NBR, BS>(o1, o2, mat, vec);                     \
    }

DEFINE_BDMV(2, 3)
DEFINE_BDMV(6, 4)

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr,
            "Usage: %s <bdmv|bdmv_dual> simple <BS> <NBR> <threads> <mat.bin> <vec.bin>\n",
            argv[0]);
        return 1;
    }
    const char* op  = argv[1];
    int BS      = atoi(argv[3]);
    int NBR     = atoi(argv[4]);
    int threads = atoi(argv[5]);
    const char* mat_path = argv[6];
    const char* vec_path = argv[7];

    int mat_n = NBR * 3 * BS * BS;
    int vec_n = (NBR + 2) * BS;
    bool dual = strcmp(op, "bdmv_dual") == 0;

    float* d_mat  = read_device_vec(mat_path, mat_n);
    float* d_vec  = read_device_vec(vec_path, vec_n);
    float* d_out  = alloc_device_vec(vec_n);                  // padded, zero-init
    float* d_out2 = dual ? alloc_device_vec(vec_n) : nullptr;

#define DISPATCH(BS_, NBR_)                                                        \
    if (BS == BS_ && NBR == NBR_) {                                                \
        if (dual) k_bdmv_dual_##BS_##_##NBR_<<<1, threads>>>(d_out, d_out2,        \
                                                             d_mat, d_vec);        \
        else      k_bdmv_##BS_##_##NBR_<<<1, threads>>>(d_out, d_mat, d_vec);      \
    }

    DISPATCH(2, 3)
    else DISPATCH(6, 4)
    else { fprintf(stderr, "unsupported size %d,%d\n", BS, NBR); return 1; }

    cudaDeviceSynchronize();
    print_device_vec(d_out, vec_n);
    if (dual) print_device_vec(d_out2, vec_n);
    return 0;
}
