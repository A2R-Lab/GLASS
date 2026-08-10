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

namespace glass { namespace nvidia { namespace block {
    DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(8,  1, 256, double)
    DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(16, 1, 256, double)
    DEFINE_NVIDIA_POSV_BLOCKDIM_PREC(32, 1, 256, double)
    DEFINE_NVIDIA_GEMM_PREC(8,  8,  8,  double)
    DEFINE_NVIDIA_GEMM_PREC(16, 16, 16, double)
    DEFINE_NVIDIA_GEMM_PREC(32, 32, 32, double)
    DEFINE_NVIDIA_GEMV_PREC(8,  8,  double)
    DEFINE_NVIDIA_GEMV_PREC(16, 16, double)
    DEFINE_NVIDIA_GEMV_PREC(32, 32, double)
    // float LAPACK tail (this is the only cuSOLVERDx-linked TU): the no-pivot
    // LU family + QR/least-squares, one shape each.
    DEFINE_NVIDIA_GETRF(8)
    DEFINE_NVIDIA_GETRS(8, 1)
    DEFINE_NVIDIA_GESV(8, 1)
    DEFINE_NVIDIA_GEQRF(8, 4)
    DEFINE_NVIDIA_GELS(8, 4, 1)
}}}

// Host-side query helpers for the LAPACK tail: constexpr, so the asserts ARE
// the test (positive scratch, sane thread counts).
namespace gnb = glass::nvidia::block;
static_assert(gnb::getrf_no_pivot_scratch_bytes<float, 8>() > 0, "getrf scratch");
static_assert(gnb::getrf_no_pivot_threads<float, 8>() > 0, "getrf threads");
static_assert(gnb::getrs_no_pivot_scratch_bytes<float, 8, 1>() > 0, "getrs scratch");
static_assert(gnb::getrs_no_pivot_threads<float, 8, 1>() > 0, "getrs threads");
static_assert(gnb::gesv_no_pivot_scratch_bytes<float, 8, 1>() > 0, "gesv scratch");
static_assert(gnb::gesv_no_pivot_threads<float, 8, 1>() > 0, "gesv threads");
static_assert(gnb::geqrf_scratch_bytes<float, 8, 4>() > 0, "geqrf scratch");
static_assert(gnb::geqrf_threads<float, 8, 4>() > 0, "geqrf threads");
static_assert(gnb::gels_scratch_bytes<float, 8, 4, 1>() > 0, "gels scratch");
static_assert(gnb::gels_threads<float, 8, 4, 1>() > 0, "gels threads");
// potrf/potrs/trsm have no DEFINE in this TU (posv composes its own chol+trsm
// internally): the un-specialized scratch helpers return the documented stub 0.
static_assert(gnb::potrf_scratch_bytes<double, 8, 256>() == 0, "potrf scratch: stub 0 without a DEFINE");
static_assert(gnb::potrf_threads<double, 8, 256>() > 0, "potrf threads");
static_assert(gnb::potrs_scratch_bytes<double, 8, 1, 256>() == 0, "potrs scratch: stub 0 without a DEFINE");
static_assert(gnb::potrs_threads<double, 8, 1, 256>() > 0, "potrs threads");
static_assert(gnb::trsm_scratch_bytes<double, 8, 1, 256>() == 0, "trsm scratch: stub 0 without a DEFINE");
static_assert(gnb::trsm_threads<double, 8, 1, 256>() > 0, "trsm threads");
static_assert(gnb::posv_threads<double, 8, 1, 256>() > 0, "posv threads");

__global__ void k_gesv(float* A, float* B) {
    extern __shared__ char s[]; gnb::gesv_no_pivot<float, 8, 1>(A, B, s);
}
__global__ void k_getrf_getrs(float* A, float* B) {
    extern __shared__ char s[];
    gnb::getrf_no_pivot<float, 8>(A, s);
    gnb::getrs_no_pivot<float, 8, 1>(A, B, s);
}
__global__ void k_geqrf(float* A, float* tau) {
    extern __shared__ char s[]; gnb::geqrf<float, 8, 4>(A, tau, s);
}
__global__ void k_gels(float* A, float* tau, float* B) {
    extern __shared__ char s[]; gnb::gels<float, 8, 4, 1>(A, tau, B, s);
}

template<int N> __global__ void k_posv(double* A, double* b) {
    extern __shared__ char s[]; glass::nvidia::block::posv<double,N,1,256>(A, b, s);
}
template<int N> __global__ void k_gemm(double* A, double* B, double* C) {
    extern __shared__ char s[]; glass::nvidia::block::gemm<double,N,N,N>(1.0, A, B, 0.0, C, s);
}
template<int N> __global__ void k_gemv(double* A, double* x, double* y) {
    extern __shared__ char s[]; glass::nvidia::block::gemv<double,N,N>(1.0, A, x, 0.0, y, s);
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
        size_t sm=glass::nvidia::block::posv_scratch_bytes<double,N,1,256>();
        cudaFuncSetAttribute(k_posv<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_posv<N><<<1,256,sm>>>(dA,db); cudaDeviceSynchronize();
        cudaMemcpy(hout,db,N*8,cudaMemcpyDeviceToHost); nout=N;
    } else if (!strcmp(op,"gemm")) {
        size_t sm=glass::nvidia::block::gemm_scratch_bytes<double,N,N,N>();
        int tb=(int)glass::nvidia::block::gemm_threads<double,N,N,N>();
        cudaFuncSetAttribute(k_gemm<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_gemm<N><<<1,tb,sm>>>(dA,dB,dC); cudaDeviceSynchronize();
        cudaMemcpy(hout,dC,N*N*8,cudaMemcpyDeviceToHost); nout=N*N;
    } else { // gemv
        size_t sm=glass::nvidia::block::gemv_scratch_bytes<double,N,N>();
        int tb=(int)glass::nvidia::block::gemv_threads<double,N,N>();
        cudaFuncSetAttribute(k_gemv<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_gemv<N><<<1,tb,sm>>>(dA,db,dC); cudaDeviceSynchronize();
        cudaMemcpy(hout,dC,N*8,cudaMemcpyDeviceToHost); nout=N;
    }
    for (int i=0;i<nout;i++) printf("%.17g ", hout[i]);
    printf("\n");
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(db);
}

// float LAPACK-tail driver: fixed shapes (N=8 / M=8,N=4), deterministic
// diag-dominant inputs; prints the result vector/matrix.
static void run_lapack_tail(const char* op) {
    float hA[8*8], hB[8], htau[8];
    for (int i=0;i<8;i++) for (int j=0;j<8;j++)
        hA[i+j*8] = ((i+3*j)%5)*0.1f + (i==j ? 8.0f : 0.0f);
    for (int i=0;i<8;i++) { hB[i] = 1.0f + 0.1f*i; htau[i] = 0.0f; }
    float *dA,*dB,*dtau;
    cudaMalloc(&dA,8*8*4); cudaMalloc(&dB,8*4); cudaMalloc(&dtau,8*4);
    cudaMemcpy(dA,hA,8*8*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,8*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dtau,htau,8*4,cudaMemcpyHostToDevice);
    float hout[8*8]; int nout = 8;
    if (!strcmp(op,"gesv")) {
        size_t sm = gnb::gesv_no_pivot_scratch_bytes<float,8,1>();
        int tb = (int)gnb::gesv_no_pivot_threads<float,8,1>();
        cudaFuncSetAttribute(k_gesv,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_gesv<<<1,tb,sm>>>(dA,dB); cudaDeviceSynchronize();
        cudaMemcpy(hout,dB,8*4,cudaMemcpyDeviceToHost);
    } else if (!strcmp(op,"lu_solve")) {
        size_t sm = gnb::getrf_no_pivot_scratch_bytes<float,8>();
        size_t sm2 = gnb::getrs_no_pivot_scratch_bytes<float,8,1>();
        if (sm2 > sm) sm = sm2;
        int tb = (int)gnb::getrf_no_pivot_threads<float,8>();
        cudaFuncSetAttribute(k_getrf_getrs,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_getrf_getrs<<<1,tb,sm>>>(dA,dB); cudaDeviceSynchronize();
        cudaMemcpy(hout,dB,8*4,cudaMemcpyDeviceToHost);
    } else if (!strcmp(op,"geqrf")) {
        size_t sm = gnb::geqrf_scratch_bytes<float,8,4>();
        int tb = (int)gnb::geqrf_threads<float,8,4>();
        cudaFuncSetAttribute(k_geqrf,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_geqrf<<<1,tb,sm>>>(dA,dtau); cudaDeviceSynchronize();
        cudaMemcpy(hout,dA,8*4*4,cudaMemcpyDeviceToHost); nout = 8*4;
    } else { // gels
        size_t sm = gnb::gels_scratch_bytes<float,8,4,1>();
        int tb = (int)gnb::gels_threads<float,8,4,1>();
        cudaFuncSetAttribute(k_gels,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        k_gels<<<1,tb,sm>>>(dA,dtau,dB); cudaDeviceSynchronize();
        cudaMemcpy(hout,dB,8*4,cudaMemcpyDeviceToHost); nout = 4;
    }
    if (cudaGetLastError() != cudaSuccess) { printf("LAUNCH_FAIL\n"); return; }
    for (int i=0;i<nout;i++) printf("%.9g ", hout[i]);
    printf("\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dtau);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s <posv|gemm|gemv|gesv|lu_solve|geqrf|gels> <N>\n", argv[0]); return 2; }
    const char* op = argv[1]; int N = atoi(argv[2]);
    if (!strcmp(op,"gesv") || !strcmp(op,"lu_solve") || !strcmp(op,"geqrf") || !strcmp(op,"gels")) {
        run_lapack_tail(op); return 0;
    }
    if      (N==8)  run<8>(op);
    else if (N==16) run<16>(op);
    else if (N==32) run<32>(op);
    else { fprintf(stderr,"N must be 8|16|32\n"); return 2; }
    return 0;
}
