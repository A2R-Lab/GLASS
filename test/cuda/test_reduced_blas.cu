// test_reduced_blas.cu — dispatch glass::gemv_reduced / glass::syrk_reduced
// (block / warp / cgrps) and print the float32 result.
//
// Usage:
//   gemv <surface> <THREADS> <M> <N> <TRANSPOSE> <alpha> <beta> <A> <x> <y>  -> y (TRANSPOSE?N:M)
//   syrk <surface> <THREADS> <ROWS> <COLS> <TRANSPOSE> <alpha> <beta> <A> <C> -> C (OUT*OUT)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

enum { BLOCK = 0, WARP = 1, CGRPS = 2 };

template <int S, uint32_t M, uint32_t N, bool TR>
__global__ void k_gemv(float al, const float* A, const float* x, float be, float* y) {
    if (S == BLOCK)     glass::gemv_reduced<float, M, N, TR>(al, A, x, be, y);
    else if (S == WARP) glass::warp::gemv_reduced<float, M, N, TR>(al, A, x, be, y);
    else                glass::cgrps::gemv_reduced<float, M, N, TR>(al, A, x, be, y);
}
template <int S, uint32_t R, uint32_t C, bool TR>
__global__ void k_syrk(float al, const float* A, float be, float* Cm) {
    if (S == BLOCK)     glass::syrk_reduced<float, R, C, TR>(al, A, be, Cm);
    else if (S == WARP) glass::warp::syrk_reduced<float, R, C, TR>(al, A, be, Cm);
    else                glass::cgrps::syrk_reduced<float, R, C, TR>(al, A, be, Cm);
}

template <uint32_t M, uint32_t N>
static void gemv_tr(int s, int th, bool tr, float al, const float* A, const float* x, float be, float* y) {
    if (tr) { if(s==BLOCK)k_gemv<BLOCK,M,N,true><<<1,th>>>(al,A,x,be,y); else if(s==WARP)k_gemv<WARP,M,N,true><<<1,th>>>(al,A,x,be,y); else k_gemv<CGRPS,M,N,true><<<1,th>>>(al,A,x,be,y); }
    else    { if(s==BLOCK)k_gemv<BLOCK,M,N,false><<<1,th>>>(al,A,x,be,y); else if(s==WARP)k_gemv<WARP,M,N,false><<<1,th>>>(al,A,x,be,y); else k_gemv<CGRPS,M,N,false><<<1,th>>>(al,A,x,be,y); }
}
template <uint32_t R, uint32_t C>
static void syrk_tr(int s, int th, bool tr, float al, const float* A, float be, float* Cm) {
    if (tr) { if(s==BLOCK)k_syrk<BLOCK,R,C,true><<<1,th>>>(al,A,be,Cm); else if(s==WARP)k_syrk<WARP,R,C,true><<<1,th>>>(al,A,be,Cm); else k_syrk<CGRPS,R,C,true><<<1,th>>>(al,A,be,Cm); }
    else    { if(s==BLOCK)k_syrk<BLOCK,R,C,false><<<1,th>>>(al,A,be,Cm); else if(s==WARP)k_syrk<WARP,R,C,false><<<1,th>>>(al,A,be,Cm); else k_syrk<CGRPS,R,C,false><<<1,th>>>(al,A,be,Cm); }
}

#define GEMV_SHAPES(_) _(14,14) _(7,21) _(8,3) _(33,5) _(5,33) _(64,3) _(3,7)
#define SYRK_SHAPES(_) _(14,7) _(8,8) _(5,3) _(33,4) _(7,14) _(64,2)

int main(int argc, char** argv) {
    const char* op = argv[1];
    int s = (strcmp(argv[2],"warp")==0)?WARP:(strcmp(argv[2],"cgrps")==0)?CGRPS:BLOCK;
    int th = atoi(argv[3]);
    if (strcmp(op,"gemv")==0) {
        uint32_t M=atoi(argv[4]),N=atoi(argv[5]); bool tr=atoi(argv[6])!=0; float al=atof(argv[7]),be=atof(argv[8]);
        uint32_t xl = tr?M:N, yl = tr?N:M;
        float* dA=read_device_vec(argv[9],M*N); float* dx=read_device_vec(argv[10],xl); float* dy=read_device_vec(argv[11],yl);
        bool ok=false;
        #define DG(MM,NN) if(!ok&&M==MM&&N==NN){gemv_tr<MM,NN>(s,th,tr,al,dA,dx,be,dy);ok=true;}
        GEMV_SHAPES(DG)
        #undef DG
        if(!ok){fprintf(stderr,"bad gemv shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        print_device_vec(dy, yl);
    } else if (strcmp(op,"syrk")==0) {
        uint32_t R=atoi(argv[4]),C=atoi(argv[5]); bool tr=atoi(argv[6])!=0; float al=atof(argv[7]),be=atof(argv[8]);
        uint32_t OUT = tr?C:R;
        float* dA=read_device_vec(argv[9],R*C); float* dC=read_device_vec(argv[10],OUT*OUT);
        bool ok=false;
        #define DS(RR,CC) if(!ok&&R==RR&&C==CC){syrk_tr<RR,CC>(s,th,tr,al,dA,be,dC);ok=true;}
        SYRK_SHAPES(DS)
        #undef DS
        if(!ok){fprintf(stderr,"bad syrk shape\n");return 1;}
        cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){fprintf(stderr,"err %s\n",cudaGetErrorString(e));return 1;}
        print_device_vec(dC, OUT*OUT);
    } else { fprintf(stderr,"bad op\n"); return 1; }
    return 0;
}
