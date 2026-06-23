// test_base_f64.cu — DOUBLE-precision validation of the pure-SIMT base ops
// (glass::) and the warp ops (glass::warp::). The rest of the suite is f32-only;
// these ops are templated on T, so this exercises the f64 instantiations.
//
// Self-contained: builds a deterministic problem (mirrored in test_base_f64.py) and
// prints the double result. Oracle-free residual checks on the python side.
//   Usage: ./test_base_f64 <dot|gemv|gemm|chol|trsv|posv> <block|warp> <N in {8,16,32}> <threads>
//          (threads ignored for warp — always one 32-lane warp)
//
// Problem (column-major, matches the numpy side):
//   M[i+j*N] = ((i+2j)%5)*0.1 ; A = M·Mᵀ + N·I (SPD) ; B[i+j*N] = ((i+3j)%4)*0.1
//   L = lower-triangular, L[i+j*N] = (i>j? ((i+2j)%5)*0.1 : 0) + (i==j? N : 0)
//   b[i] = 1 + 0.1·i

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "glass.cuh"

template<typename T,int N,bool WARP> __global__ void k_dot(T* x, T* y) {
    if constexpr (WARP) { T r = glass::warp::dot<T,N>(x, y); if ((threadIdx.x&31)==0) y[0]=r; }
    else glass::dot<T,N>(x, y);                                   // result -> y[0]
}
template<typename T,int N,bool WARP> __global__ void k_gemv(T* A, T* x, T* y) {
    if constexpr (WARP) glass::warp::gemv<T,N,N>((T)1, A, x, (T)0, y);
    else                glass::gemv<T,N,N>((T)1, A, x, (T)0, y);
}
template<typename T,int N,bool WARP> __global__ void k_gemm(T* A, T* B, T* C) {
    if constexpr (WARP) glass::warp::gemm<T,N,N,N>((T)1, A, B, (T)0, C);
    else                glass::gemm<T,N,N,N>((T)1, A, B, (T)0, C);
}
template<typename T,int N,bool WARP> __global__ void k_chol(T* A) {
    if constexpr (WARP) glass::warp::cholDecomp_InPlace<T,N>(A);   // A -> L (lower)
    else                glass::cholDecomp_InPlace<T,N>(A);
}
template<typename T,int N,bool WARP> __global__ void k_trsv(T* L, T* x) {
    if constexpr (WARP) glass::warp::trsv<T,N>(L, x);             // solve L x = b in place
    else                glass::trsv<T,N>(L, x);
}
template<typename T,int N,bool WARP> __global__ void k_posv(T* A, T* b) {
    if constexpr (WARP) glass::warp::posv<T,N>(A, b);            // b -> solution
    else                glass::posv<T,N>(A, b);
}

using T = double;

static void build(int N, T* A, T* B, T* L, T* b) {
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) {
        T m=0; for (int k=0;k<N;k++) m += (((i+2*k)%5)*0.1) * (((j+2*k)%5)*0.1);
        A[i+j*N] = m + (i==j ? (T)N : 0.0);
        B[i+j*N] = ((i+3*j)%4)*0.1;
        L[i+j*N] = (i>j ? ((i+2*j)%5)*0.1 : 0.0) + (i==j ? (T)N : 0.0);
    }
    for (int i=0;i<N;i++) b[i] = 1.0 + 0.1*i;
}

template<int N, bool WARP>
static void run(const char* op, int tb) {
    int threads = WARP ? 32 : tb;
    T hA[N*N], hB[N*N], hL[N*N], hb[N], hout[N*N>N? N*N : N];
    build(N, hA, hB, hL, hb);
    T *dA,*dB,*dC,*dL,*db; cudaMalloc(&dA,N*N*8);cudaMalloc(&dB,N*N*8);
    cudaMalloc(&dC,N*N*8);cudaMalloc(&dL,N*N*8);cudaMalloc(&db,N*8);
    cudaMemcpy(dA,hA,N*N*8,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB,N*N*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dL,hL,N*N*8,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,N*8,cudaMemcpyHostToDevice);
    int nout=N;
    if (!strcmp(op,"dot"))  {            // x=db (b), y=dC (=b) → result b·b in y[0]
        cudaMemcpy(dC,hb,N*8,cudaMemcpyHostToDevice);
        k_dot<T,N,WARP><<<1,threads>>>(db,dC); cudaMemcpy(hout,dC,8,cudaMemcpyDeviceToHost); nout=1; }
    else if (!strcmp(op,"gemv")) { k_gemv<T,N,WARP><<<1,threads>>>(dA,db,dC); cudaMemcpy(hout,dC,N*8,cudaMemcpyDeviceToHost); nout=N; }
    else if (!strcmp(op,"gemm")) { k_gemm<T,N,WARP><<<1,threads>>>(dA,dB,dC); cudaMemcpy(hout,dC,N*N*8,cudaMemcpyDeviceToHost); nout=N*N; }
    else if (!strcmp(op,"chol")) { k_chol<T,N,WARP><<<1,threads>>>(dA); cudaMemcpy(hout,dA,N*N*8,cudaMemcpyDeviceToHost); nout=N*N; }
    else if (!strcmp(op,"trsv")) { k_trsv<T,N,WARP><<<1,threads>>>(dL,db); cudaMemcpy(hout,db,N*8,cudaMemcpyDeviceToHost); nout=N; }
    else /*posv*/                { k_posv<T,N,WARP><<<1,threads>>>(dA,db); cudaMemcpy(hout,db,N*8,cudaMemcpyDeviceToHost); nout=N; }
    cudaDeviceSynchronize();
    for (int i=0;i<nout;i++) printf("%.17g ", hout[i]);
    printf("\n");
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dL);cudaFree(db);
}

template<int N> static void dispatch(const char* op, bool warp, int tb) {
    if (warp) run<N,true>(op, tb); else run<N,false>(op, tb);
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr,"usage: %s op block|warp N threads\n", argv[0]); return 2; }
    const char* op = argv[1];
    bool warp = !strcmp(argv[2], "warp");
    int N = atoi(argv[3]); int tb = atoi(argv[4]);
    if      (N==8)  dispatch<8>(op, warp, tb);
    else if (N==16) dispatch<16>(op, warp, tb);
    else if (N==32) dispatch<32>(op, warp, tb);
    else { fprintf(stderr,"N must be 8|16|32\n"); return 2; }
    return 0;
}
