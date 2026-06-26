// test_nvidia_f64.cu — double-precision validation of the glass::nvidia cuSOLVERDx /
// cuBLASDx wrappers (posv = chol+trsm, gemm, gemv). Self-contained: builds a
// DETERMINISTIC problem from a fixed formula that test_nvidia_f64.py reproduces in
// numpy, runs the *double* nvidia op, and prints the result at full double precision.
//
// Needs cuBLASDx + cuSOLVERDx (compiled by conftest with the MathDx + -dlto flags).
//   Usage: ./test_nvidia_f64 <posv|gemm|gemv> <N in {8,16,32}>
//
// Problem (column-major, matches the numpy side exactly):
//   M[i+j*N] = ((i + 2j) % 5) * 0.1          (deterministic dense)
//   A        = M·Mᵀ + N·I                     (SPD)
//   B[i+j*N] = ((i + 3j) % 4) * 0.1           (gemm RHS matrix)
//   b[i]     = 1 + 0.1·i                       (posv/gemv RHS vector)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublasdx.hpp>
#include "glass-nvidia.cuh"

namespace glass { namespace nvidia {
    DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(8,  1, 256, double)
    DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(16, 1, 256, double)
    DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(32, 1, 256, double)
    DEFINE_NVIDIA_GEMM_PREC(8,  8,  8,  double)
    DEFINE_NVIDIA_GEMM_PREC(16, 16, 16, double)
    DEFINE_NVIDIA_GEMM_PREC(32, 32, 32, double)
    DEFINE_NVIDIA_GEMV_PREC(8,  8,  double)
    DEFINE_NVIDIA_GEMV_PREC(16, 16, double)
    DEFINE_NVIDIA_GEMV_PREC(32, 32, double)
}}

template<int N> __global__ void k_posv(double* A, double* b) {
    extern __shared__ char s[]; glass::nvidia::posv<double,N,1,256>(A, b, s);
}
template<int N> __global__ void k_gemm(double* A, double* B, double* C) {
    extern __shared__ char s[]; glass::nvidia::gemm<double,N,N,N>(1.0, A, B, 0.0, C, s);
}
template<int N> __global__ void k_gemv(double* A, double* x, double* y) {
    extern __shared__ char s[]; glass::nvidia::gemv<double,N,N>(1.0, A, x, 0.0, y, s);
}

static void build(int N, double* A, double* B, double* b) {
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) {
        double m=0; for (int k=0;k<N;k++) m += (((i+2*k)%5)*0.1) * (((j+2*k)%5)*0.1);
        A[i+j*N] = m + (i==j ? (double)N : 0.0);
        B[i+j*N] = ((i+3*j)%4)*0.1;
    }
    for (int i=0;i<N;i++) b[i] = 1.0 + 0.1*i;
}

template<int N> static void run(const char* op) {
    double hA[N*N], hB[N*N], hb[N], hout[N*N>N? N*N : N];
    build(N, hA, hB, hb);
    double *dA,*dB,*dC,*db; cudaMalloc(&dA,N*N*8); cudaMalloc(&dB,N*N*8);
    cudaMalloc(&dC,N*N*8); cudaMalloc(&db,N*8);
    cudaMemcpy(dA,hA,N*N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,N*N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*8,cudaMemcpyHostToDevice);
    int nout=N;
    if (!strcmp(op,"posv")) {
        size_t sm=glass::nvidia::posv_scratch_bytes<double,N,1,256>();
        cudaFuncSetAttribute(k_posv<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_posv<N><<<1,256,sm>>>(dA,db); cudaDeviceSynchronize();
        cudaMemcpy(hout,db,N*8,cudaMemcpyDeviceToHost); nout=N;
    } else if (!strcmp(op,"gemm")) {
        size_t sm=glass::nvidia::gemm_scratch_bytes<double,N,N,N>();
        int tb=(int)glass::nvidia::gemm_threads<double,N,N,N>();
        cudaFuncSetAttribute(k_gemm<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_gemm<N><<<1,tb,sm>>>(dA,dB,dC); cudaDeviceSynchronize();
        cudaMemcpy(hout,dC,N*N*8,cudaMemcpyDeviceToHost); nout=N*N;
    } else { // gemv
        size_t sm=glass::nvidia::gemv_scratch_bytes<double,N,N>();
        int tb=(int)glass::nvidia::gemv_threads<double,N,N>();
        cudaFuncSetAttribute(k_gemv<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_gemv<N><<<1,tb,sm>>>(dA,db,dC); cudaDeviceSynchronize();
        cudaMemcpy(hout,dC,N*8,cudaMemcpyDeviceToHost); nout=N;
    }
    for (int i=0;i<nout;i++) printf("%.17g ", hout[i]);
    printf("\n");
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(db);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s <posv|gemm|gemv> <N>\n", argv[0]); return 2; }
    const char* op = argv[1]; int N = atoi(argv[2]);
    if      (N==8)  run<8>(op);
    else if (N==16) run<16>(op);
    else if (N==32) run<32>(op);
    else { fprintf(stderr,"N must be 8|16|32\n"); return 2; }
    return 0;
}
