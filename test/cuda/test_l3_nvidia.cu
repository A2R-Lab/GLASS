// test_l3_nvidia.cu — exercise glass::nvidia 1D-launch SIMT batched GEMMs.
//
// Usage:
//   ./test_l3_nvidia gemm_batched_1d_<M>x<N>x<K>_b<BATCH>_<col|row>  <alpha> <beta>  A.bin B.bin C.bin
//   ./test_l3_nvidia gemm_strided_batched_1d_<M>x<N>x<K>_b<BATCH>    <alpha> <beta>  Ashared.bin B.bin C.bin
//
// All matrices are passed flat. For col-major layout, an M×K matrix is written
// in column-major order. For batched cases, BATCH copies are concatenated.
//
// Output: stdout — full result vector(s), space-separated.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.cuh"
#include "../../glass-nvidia.cuh"

static constexpr int TC = 32;

// ─── gemm_batched_1d kernels (col-major) ─────────────────────────────────────
template <int M_, int N_, int K_, int BATCH_>
__global__ void k_batched_1d(float alpha, float** A, float** B, float beta, float** C) {
    glass::nvidia::gemm_batched_1d<float, M_, N_, K_, BATCH_, TC>(alpha, A, B, beta, C);
}

template <int M_, int N_, int K_, int BATCH_>
__global__ void k_batched_1d_rowmajor(float alpha, float** A, float** B, float beta, float** C) {
    glass::nvidia::gemm_batched_1d<float, M_, N_, K_, BATCH_, TC,
        glass::nvidia::layout::row_major,
        glass::nvidia::layout::row_major,
        glass::nvidia::layout::row_major>(alpha, A, B, beta, C);
}

// ─── gemm_strided_batched_1d kernels ─────────────────────────────────────────
template <int M_, int N_, int K_, int BATCH_>
__global__ void k_strided_1d(float alpha, float* A_shared, float* B, float beta, float* C) {
    glass::nvidia::gemm_strided_batched_1d<float, M_, N_, K_, BATCH_, TC>(
        alpha, A_shared, B, beta, C);
}

// Non-default strides: B and C have padding between batches (B_STRIDE > N*K,
// C_STRIDE > M*K). Exercises the * b * STRIDE indexing inside the kernel.
template <int M_, int N_, int K_, int BATCH_, int B_STRIDE_, int C_STRIDE_>
__global__ void k_strided_1d_padded(float alpha, float* A_shared, float* B, float beta, float* C) {
    glass::nvidia::gemm_strided_batched_1d<float, M_, N_, K_, BATCH_, TC,
                                            B_STRIDE_, C_STRIDE_>(
        alpha, A_shared, B, beta, C);
}

// ─── helpers ────────────────────────────────────────────────────────────────
// Build BATCH-length pointer arrays on host, copy to device, return device ptr.
static float** build_ptr_array(float* base, int batch, int per) {
    float* h_ptrs[64];
    if (batch > 64) { fprintf(stderr, "BATCH > 64 not supported in test\n"); exit(1); }
    for (int b = 0; b < batch; b++) h_ptrs[b] = base + b * per;
    float** d_ptrs;
    cudaMalloc(&d_ptrs, batch * sizeof(float*));
    cudaMemcpy(d_ptrs, h_ptrs, batch * sizeof(float*), cudaMemcpyHostToDevice);
    return d_ptrs;
}

// Dispatch one (M, N, K, BATCH) gemm_batched_1d.
// argv layout (matches conftest.py run_op):
//   [1]=op  [2]=version  [3]=alpha  [4]=beta  [5]=A.bin  [6]=B.bin  [7]=C.bin
template <int M_, int N_, int K_, int BATCH_>
static void do_batched(int argc, char** argv, bool row_major) {
    float alpha = atof(argv[3]);
    float beta  = atof(argv[4]);
    float* dA = read_device_vec(argv[5], BATCH_ * M_ * N_);
    float* dB = read_device_vec(argv[6], BATCH_ * N_ * K_);
    float* dC = read_device_vec(argv[7], BATCH_ * M_ * K_);

    float** dA_ptrs = build_ptr_array(dA, BATCH_, M_ * N_);
    float** dB_ptrs = build_ptr_array(dB, BATCH_, N_ * K_);
    float** dC_ptrs = build_ptr_array(dC, BATCH_, M_ * K_);

    dim3 block(TC * BATCH_, 1, 1);
    if (row_major)
        k_batched_1d_rowmajor<M_, N_, K_, BATCH_><<<1, block>>>(alpha, dA_ptrs, dB_ptrs, beta, dC_ptrs);
    else
        k_batched_1d<M_, N_, K_, BATCH_><<<1, block>>>(alpha, dA_ptrs, dB_ptrs, beta, dC_ptrs);
    cudaDeviceSynchronize();
    print_device_vec(dC, BATCH_ * M_ * K_);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dA_ptrs); cudaFree(dB_ptrs); cudaFree(dC_ptrs);
}

// Dispatch one (M, N, K, BATCH) gemm_strided_batched_1d (shared A, packed B/C).
// argv layout matches do_batched.
template <int M_, int N_, int K_, int BATCH_>
static void do_strided(int argc, char** argv) {
    float alpha = atof(argv[3]);
    float beta  = atof(argv[4]);
    float* dAs = read_device_vec(argv[5], M_ * N_);
    float* dB  = read_device_vec(argv[6], BATCH_ * N_ * K_);
    float* dC  = read_device_vec(argv[7], BATCH_ * M_ * K_);

    dim3 block(TC * BATCH_, 1, 1);
    k_strided_1d<M_, N_, K_, BATCH_><<<1, block>>>(alpha, dAs, dB, beta, dC);
    cudaDeviceSynchronize();
    print_device_vec(dC, BATCH_ * M_ * K_);

    cudaFree(dAs); cudaFree(dB); cudaFree(dC);
}

// Strided variant with explicit (non-default) B_STRIDE, C_STRIDE.
// Caller passes BATCH_ * B_STRIDE_ B-elements and BATCH_ * C_STRIDE_ C-elements;
// only the first N_*K_ / M_*K_ slots of each batch are used by the kernel
// (padding bytes are read-but-not-written by C and not-read-or-written by B).
template <int M_, int N_, int K_, int BATCH_, int B_STRIDE_, int C_STRIDE_>
static void do_strided_padded(int argc, char** argv) {
    float alpha = atof(argv[3]);
    float beta  = atof(argv[4]);
    float* dAs = read_device_vec(argv[5], M_ * N_);
    float* dB  = read_device_vec(argv[6], BATCH_ * B_STRIDE_);
    float* dC  = read_device_vec(argv[7], BATCH_ * C_STRIDE_);

    dim3 block(TC * BATCH_, 1, 1);
    k_strided_1d_padded<M_, N_, K_, BATCH_, B_STRIDE_, C_STRIDE_>
        <<<1, block>>>(alpha, dAs, dB, beta, dC);
    cudaDeviceSynchronize();
    print_device_vec(dC, BATCH_ * C_STRIDE_);

    cudaFree(dAs); cudaFree(dB); cudaFree(dC);
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <op> <alpha> <beta> A.bin B.bin C.bin\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];

    // gemm_batched_1d, col-major
    if (!strcmp(op, "gemm_batched_1d_4x4x4_b1_col"))  return do_batched<4,4,4,1>(argc, argv, false), 0;
    if (!strcmp(op, "gemm_batched_1d_4x4x4_b4_col"))  return do_batched<4,4,4,4>(argc, argv, false), 0;
    if (!strcmp(op, "gemm_batched_1d_6x6x6_b2_col"))  return do_batched<6,6,6,2>(argc, argv, false), 0;
    if (!strcmp(op, "gemm_batched_1d_3x5x7_b3_col"))  return do_batched<3,5,7,3>(argc, argv, false), 0;
    // gemm_batched_1d, row-major
    if (!strcmp(op, "gemm_batched_1d_4x4x4_b4_row"))  return do_batched<4,4,4,4>(argc, argv, true), 0;
    if (!strcmp(op, "gemm_batched_1d_3x5x7_b3_row"))  return do_batched<3,5,7,3>(argc, argv, true), 0;

    // gemm_strided_batched_1d, col-major (defaults: B_STRIDE=N*K, C_STRIDE=M*K)
    if (!strcmp(op, "gemm_strided_batched_1d_4x4x4_b1"))  return do_strided<4,4,4,1>(argc, argv), 0;
    if (!strcmp(op, "gemm_strided_batched_1d_4x4x4_b4"))  return do_strided<4,4,4,4>(argc, argv), 0;
    if (!strcmp(op, "gemm_strided_batched_1d_6x6x6_b2"))  return do_strided<6,6,6,2>(argc, argv), 0;
    if (!strcmp(op, "gemm_strided_batched_1d_3x5x7_b3"))  return do_strided<3,5,7,3>(argc, argv), 0;

    // Strided variant with non-default (padded) strides. Each batch's B has
    // B_STRIDE elements (only the first N*K used); same for C with C_STRIDE.
    // Format: <op> <alpha> <beta> Ashared.bin B_padded.bin C_padded.bin
    if (!strcmp(op, "gemm_strided_padded_4x4x4_b4_bs24_cs20"))
        return do_strided_padded<4,4,4,4,/*B_STRIDE=*/24,/*C_STRIDE=*/20>(argc, argv), 0;
    if (!strcmp(op, "gemm_strided_padded_3x5x7_b3_bs50_cs28"))
        return do_strided_padded<3,5,7,3,/*B_STRIDE=*/50,/*C_STRIDE=*/28>(argc, argv), 0;

    fprintf(stderr, "Unknown op: %s\n", op);
    return 1;
}
