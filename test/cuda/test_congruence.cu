// test_congruence.cu — dispatch glass::congruence_sym / glass::bilinear
// (block / warp / cgrps) and print the float32 result.
//
// Usage:
//   cong <surface> <THREADS> <N> <Kdim> <ACC> <alpha> <beta> <X> <M> <Q>   -> Q (Kdim*Kdim)
//   bil  <surface> <THREADS> <N> <P> <Qd> <ACC> <alpha> <beta> <X> <M> <Y> <R> -> R (P*Qd)
//     surface : block | warp | cgrps   (X,M,Y,Q,R column-major)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

enum { SURF_BLOCK = 0, SURF_WARP = 1, SURF_CGRPS = 2 };

template <int SURF, uint32_t N, uint32_t Kdim, bool ACC>
__global__ void k_cong(float alpha, const float* X, const float* M, float beta, float* Q) {
    extern __shared__ float st[];
    if (SURF == SURF_BLOCK)      glass::congruence_sym<float, N, Kdim, ACC>(alpha, X, M, beta, Q, st);
    else if (SURF == SURF_WARP)  glass::warp::congruence_sym<float, N, Kdim, ACC>(alpha, X, M, beta, Q, st);
    else                         glass::cgrps::congruence_sym<float, N, Kdim, ACC>(alpha, X, M, beta, Q, st);
}

template <int SURF, uint32_t N, uint32_t P, uint32_t Qd, bool ACC>
__global__ void k_bil(float alpha, const float* X, const float* M, const float* Y, float beta, float* R) {
    extern __shared__ float st[];
    if (SURF == SURF_BLOCK)      glass::bilinear<float, N, P, Qd, ACC>(alpha, X, M, Y, beta, R, st);
    else if (SURF == SURF_WARP)  glass::warp::bilinear<float, N, P, Qd, ACC>(alpha, X, M, Y, beta, R, st);
    else                         glass::cgrps::bilinear<float, N, P, Qd, ACC>(alpha, X, M, Y, beta, R, st);
}

template <uint32_t N, uint32_t Kdim>
static void launch_cong(int surf, int th, bool acc, float al, const float* dX, const float* dM, float be, float* dQ) {
    int sm = N * Kdim * sizeof(float);
    if (acc) {
        if (surf==SURF_BLOCK) k_cong<SURF_BLOCK,N,Kdim,true><<<1,th,sm>>>(al,dX,dM,be,dQ);
        else if (surf==SURF_WARP) k_cong<SURF_WARP,N,Kdim,true><<<1,th,sm>>>(al,dX,dM,be,dQ);
        else k_cong<SURF_CGRPS,N,Kdim,true><<<1,th,sm>>>(al,dX,dM,be,dQ);
    } else {
        if (surf==SURF_BLOCK) k_cong<SURF_BLOCK,N,Kdim,false><<<1,th,sm>>>(al,dX,dM,be,dQ);
        else if (surf==SURF_WARP) k_cong<SURF_WARP,N,Kdim,false><<<1,th,sm>>>(al,dX,dM,be,dQ);
        else k_cong<SURF_CGRPS,N,Kdim,false><<<1,th,sm>>>(al,dX,dM,be,dQ);
    }
}

template <int SURF, uint32_t P, uint32_t Q, bool ACC>
__global__ void k_cacc(float alpha, const float* G, const float* M, float beta, float* C) {
    extern __shared__ float st[];
    if (SURF == SURF_WARP) glass::warp::congruence_accum<float, P, Q, ACC>(alpha, G, M, beta, C, st);
    else                   glass::congruence_accum<float, P, Q, ACC>(alpha, G, M, beta, C, st);
}

template <uint32_t P, uint32_t Q>
static void launch_cacc(int surf, int th, bool acc, float al, const float* dG, const float* dM, float be, float* dC) {
    int sm = glass::congruence_accum_smem_count<float,P,Q>() * sizeof(float);
    if (acc) {
        if (surf==SURF_WARP) k_cacc<SURF_WARP,P,Q,true><<<1,th,sm>>>(al,dG,dM,be,dC);
        else                 k_cacc<SURF_BLOCK,P,Q,true><<<1,th,sm>>>(al,dG,dM,be,dC);
    } else {
        if (surf==SURF_WARP) k_cacc<SURF_WARP,P,Q,false><<<1,th,sm>>>(al,dG,dM,be,dC);
        else                 k_cacc<SURF_BLOCK,P,Q,false><<<1,th,sm>>>(al,dG,dM,be,dC);
    }
}

template <uint32_t N, uint32_t P, uint32_t Qd>
static void launch_bil(int surf, int th, bool acc, float al, const float* dX, const float* dM, const float* dY, float be, float* dR) {
    int sm = N * Qd * sizeof(float);
    if (acc) {
        if (surf==SURF_BLOCK) k_bil<SURF_BLOCK,N,P,Qd,true><<<1,th,sm>>>(al,dX,dM,dY,be,dR);
        else if (surf==SURF_WARP) k_bil<SURF_WARP,N,P,Qd,true><<<1,th,sm>>>(al,dX,dM,dY,be,dR);
        else k_bil<SURF_CGRPS,N,P,Qd,true><<<1,th,sm>>>(al,dX,dM,dY,be,dR);
    } else {
        if (surf==SURF_BLOCK) k_bil<SURF_BLOCK,N,P,Qd,false><<<1,th,sm>>>(al,dX,dM,dY,be,dR);
        else if (surf==SURF_WARP) k_bil<SURF_WARP,N,P,Qd,false><<<1,th,sm>>>(al,dX,dM,dY,be,dR);
        else k_bil<SURF_CGRPS,N,P,Qd,false><<<1,th,sm>>>(al,dX,dM,dY,be,dR);
    }
}

#define CONG_SHAPES(_) _(14,21) _(8,8) _(5,3) _(7,14) _(33,5) _(64,3) _(14,14) _(3,4)
#define BIL_SHAPES(_)  _(14,7,21) _(8,5,3) _(5,5,5) _(33,4,6) _(7,14,7)
// congruence_accum: G is P×Q, M is Q×Q, C is P×P (GATO B·R⁻¹·Bᵀ shapes).
#define CACC_SHAPES(_) _(5,3) _(14,7) _(7,7) _(8,4) _(6,5) _(3,3)

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "need op\n"); return 1; }
    const char* op = argv[1];
    int surf = (strcmp(argv[2],"warp")==0)?SURF_WARP:(strcmp(argv[2],"cgrps")==0)?SURF_CGRPS:SURF_BLOCK;
    int th = atoi(argv[3]);

    if (strcmp(op,"cong")==0) {
        uint32_t N=atoi(argv[4]), Kdim=atoi(argv[5]); bool acc=atoi(argv[6])!=0;
        float al=atof(argv[7]), be=atof(argv[8]);
        float* dX=read_device_vec(argv[9], N*Kdim);
        float* dM=read_device_vec(argv[10], N*N);
        float* dQ=read_device_vec(argv[11], Kdim*Kdim);
        bool ok=false;
        #define DC(NN,KK) if(!ok&&N==NN&&Kdim==KK){launch_cong<NN,KK>(surf,th,acc,al,dX,dM,be,dQ);ok=true;}
        CONG_SHAPES(DC)
        #undef DC
        if(!ok){fprintf(stderr,"bad cong shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        print_device_vec(dQ, Kdim*Kdim);
    } else if (strcmp(op,"bil")==0) {
        uint32_t N=atoi(argv[4]), P=atoi(argv[5]), Qd=atoi(argv[6]); bool acc=atoi(argv[7])!=0;
        float al=atof(argv[8]), be=atof(argv[9]);
        float* dX=read_device_vec(argv[10], N*P);
        float* dM=read_device_vec(argv[11], N*N);
        float* dY=read_device_vec(argv[12], N*Qd);
        float* dR=read_device_vec(argv[13], P*Qd);
        bool ok=false;
        #define DB(NN,PP,QQ) if(!ok&&N==NN&&P==PP&&Qd==QQ){launch_bil<NN,PP,QQ>(surf,th,acc,al,dX,dM,dY,be,dR);ok=true;}
        BIL_SHAPES(DB)
        #undef DB
        if(!ok){fprintf(stderr,"bad bil shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        print_device_vec(dR, P*Qd);
    } else if (strcmp(op,"cacc")==0) {
        uint32_t P=atoi(argv[4]), Q=atoi(argv[5]); bool acc=atoi(argv[6])!=0;
        float al=atof(argv[7]), be=atof(argv[8]);
        float* dG=read_device_vec(argv[9], P*Q);
        float* dM=read_device_vec(argv[10], Q*Q);
        float* dC=read_device_vec(argv[11], P*P);
        bool ok=false;
        #define DA(PP,QQ) if(!ok&&P==PP&&Q==QQ){launch_cacc<PP,QQ>(surf,th,acc,al,dG,dM,be,dC);ok=true;}
        CACC_SHAPES(DA)
        #undef DA
        if(!ok){fprintf(stderr,"bad cacc shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        print_device_vec(dC, P*P);
    } else { fprintf(stderr,"bad op\n"); return 1; }
    return 0;
}
