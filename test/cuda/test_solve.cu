// test_solve.cu — exercise the posv REGULARIZE/CHECK flags and riccati_gain.
//
// Usage:
//   posvreg <THREADS> <N> <NRHS> <REG> <rho> <A.bin> <B.bin>   -> "<fail>\n<X (N*NRHS)>"
//   riccati <THREADS> <NX> <NU> <REG> <rho> <P> <A> <B> <R>    -> "<fail>\n<K (NU*NX)>"
//   CHECK is always on (reports s_fail).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

template <uint32_t N, uint32_t NRHS, bool REG>
__global__ void k_posvreg(float* A, float* B, float rho, int* fail) {
    extern __shared__ float s[];
    float* sA = s; float* sB = s + N*N;
    for (uint32_t i = threadIdx.x; i < N*N; i += blockDim.x) sA[i] = A[i];
    for (uint32_t i = threadIdx.x; i < N*NRHS; i += blockDim.x) sB[i] = B[i];
    __syncthreads();
    glass::posv<float, N, NRHS, REG, true>(sA, sB, rho, fail);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < N*NRHS; i += blockDim.x) B[i] = sB[i];
}

template <uint32_t NX, uint32_t NU, bool REG>
__global__ void k_riccati(const float* P, const float* A, const float* B, const float* R,
                          float* K, float rho, int* fail) {
    extern __shared__ float st[];
    glass::riccati_gain<float, NX, NU, REG>(P, A, B, R, K, st, rho, fail);
}

template <uint32_t NX, uint32_t NU, bool REG>
__global__ void k_riccati_warp(const float* P, const float* A, const float* B, const float* R,
                               float* K, float rho, int* fail) {
    extern __shared__ float st[];
    glass::warp::riccati_gain<float, NX, NU, REG>(P, A, B, R, K, st, rho, fail);
}

#define POSV_SHAPES(_) _(7,7) _(8,5) _(14,7) _(5,1) _(3,3) _(7,14)
#define RIC_SHAPES(_)  _(14,7) _(8,4) _(6,3) _(10,5) _(4,2)

int main(int argc, char** argv) {
    const char* op = argv[1];
    int th = atoi(argv[2]);

    if (strcmp(op, "posvreg") == 0) {
        uint32_t N = atoi(argv[3]), NRHS = atoi(argv[4]); bool reg = atoi(argv[5]) != 0;
        float rho = atof(argv[6]);
        float* dA = read_device_vec(argv[7], N*N);
        float* dB = read_device_vec(argv[8], N*NRHS);
        int* dFail; cudaMalloc(&dFail, sizeof(int));
        int sm = (N*N + N*NRHS) * sizeof(float);
        bool ok = false;
        #define DP(NN,RR) if(!ok && N==NN && NRHS==RR){ \
            if(reg) k_posvreg<NN,RR,true><<<1,th,sm>>>(dA,dB,rho,dFail); \
            else    k_posvreg<NN,RR,false><<<1,th,sm>>>(dA,dB,rho,dFail); ok=true; }
        POSV_SHAPES(DP)
        #undef DP
        if(!ok){fprintf(stderr,"bad posv shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        int fail; cudaMemcpy(&fail,dFail,sizeof(int),cudaMemcpyDeviceToHost);
        printf("%d\n", fail);
        print_device_vec(dB, N*NRHS);
    } else if (strcmp(op, "riccati") == 0 || strcmp(op, "riccati_warp") == 0) {
        bool warp = (strcmp(op, "riccati_warp") == 0);
        uint32_t NX = atoi(argv[3]), NU = atoi(argv[4]); bool reg = atoi(argv[5]) != 0;
        float rho = atof(argv[6]);
        float* dP = read_device_vec(argv[7], NX*NX);
        float* dA = read_device_vec(argv[8], NX*NX);
        float* dB = read_device_vec(argv[9], NX*NU);
        float* dR = read_device_vec(argv[10], NU*NU);
        float* dK; cudaMalloc(&dK, NU*NX*sizeof(float));
        int* dFail; cudaMalloc(&dFail, sizeof(int));
        bool ok = false;
        #define DR(XX,UU) if(!ok && NX==XX && NU==UU){ \
            int sm = glass::riccati_smem_count<XX,UU>()*sizeof(float); \
            if(warp){ if(reg) k_riccati_warp<XX,UU,true><<<1,32,sm>>>(dP,dA,dB,dR,dK,rho,dFail); \
                      else    k_riccati_warp<XX,UU,false><<<1,32,sm>>>(dP,dA,dB,dR,dK,rho,dFail); } \
            else    { if(reg) k_riccati<XX,UU,true><<<1,th,sm>>>(dP,dA,dB,dR,dK,rho,dFail); \
                      else    k_riccati<XX,UU,false><<<1,th,sm>>>(dP,dA,dB,dR,dK,rho,dFail); } ok=true; }
        RIC_SHAPES(DR)
        #undef DR
        if(!ok){fprintf(stderr,"bad riccati shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        int fail; cudaMemcpy(&fail,dFail,sizeof(int),cudaMemcpyDeviceToHost);
        printf("%d\n", fail);
        print_device_vec(dK, NU*NX);
    } else { fprintf(stderr,"bad op\n"); return 1; }
    return 0;
}
