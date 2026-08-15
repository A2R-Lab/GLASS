// A/B harness for candidate work-mapping changes that are not yet public API.
// Correctness stays in pytest; candidates that win the quiet timing pass must
// be promoted with dedicated oracle tests before landing.
//
// NOTE: the timed region includes each kernel's per-launch shared-memory
// initialization (operand fill), which is identical across the A and B
// variants — so reported ratios are LOWER BOUNDS on the mapping-only delta.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "timing_common.cuh"
#include "../glass.cuh"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(3); } } while (0)

static int NPROB = 8192;

template <typename T, uint32_t M, uint32_t N>
__device__ void ger_flat(T alpha, const T* x, const T* y, T* A) {
    for (uint32_t e = threadIdx.x; e < M*N; e += blockDim.x) {
        uint32_t row = e % M, col = e / M;
        A[e] += alpha * x[row] * y[col];
    }
    __syncthreads();
}

template <typename T, uint32_t M, uint32_t N>
__device__ void ger_legacy(T alpha, const T* x, const T* y, T* A) {
    for (uint32_t col = 0; col < N; ++col) {
        T ay = alpha * y[col];
        for (uint32_t row = threadIdx.x; row < M; row += blockDim.x)
            A[row + col*M] += ay * x[row];
    }
    __syncthreads();
}

template <typename T, uint32_t M, uint32_t N, bool FLAT>
__global__ void k_ger(T* out) {
    extern __shared__ double raw[];
    T* x = reinterpret_cast<T*>(raw); T* y = x + M; T* A = y + N;
    for (uint32_t i=threadIdx.x; i<M; i+=blockDim.x) x[i]=(T)(0.01*(i%13+1));
    for (uint32_t i=threadIdx.x; i<N; i+=blockDim.x) y[i]=(T)(0.02*(i%11+1));
    for (uint32_t i=threadIdx.x; i<M*N; i+=blockDim.x) A[i]=(T)1;
    __syncthreads();
    if constexpr (FLAT) ger_flat<T,M,N>((T)0.5,x,y,A);
    else ger_legacy<T,M,N>((T)0.5,x,y,A);
    if(threadIdx.x==0) out[blockIdx.x]=A[0];
}

template <typename T, uint32_t R, uint32_t C, uint32_t TILE=16>
__device__ void transpose_tiled(const T* a, T* b, T* tile) {
    constexpr uint32_t STRIDE = TILE + 1;
    for (uint32_t r0=0; r0<R; r0+=TILE) {
        for (uint32_t c0=0; c0<C; c0+=TILE) {
            for (uint32_t e=threadIdx.x; e<TILE*TILE; e+=blockDim.x) {
                uint32_t rr=e%TILE, cc=e/TILE;
                if(r0+rr<R && c0+cc<C) tile[rr+cc*STRIDE]=a[(r0+rr)+(c0+cc)*R];
            }
            __syncthreads();
            for (uint32_t e=threadIdx.x; e<TILE*TILE; e+=blockDim.x) {
                uint32_t rr=e%TILE, cc=e/TILE;
                if(c0+rr<C && r0+cc<R) b[(c0+rr)+(r0+cc)*C]=tile[cc+rr*STRIDE];
            }
            __syncthreads();
        }
    }
}

template <typename T, uint32_t R, uint32_t C, bool TILED>
__global__ void k_transpose(T* out) {
    extern __shared__ double raw[];
    T* a=reinterpret_cast<T*>(raw); T* b=a+R*C; T* tile=b+R*C;
    for(uint32_t i=threadIdx.x;i<R*C;i+=blockDim.x) a[i]=(T)(i%29);
    __syncthreads();
    if constexpr(TILED) transpose_tiled<T,R,C>(a,b,tile);
    else glass::block::transpose<T,R,C>(a,b);
    if(threadIdx.x==0) out[blockIdx.x]=b[0];
}

template <typename T, bool EXCLUSIVE>
__device__ void prefix_shuffle(T* input, T* output, int n) {
    uint32_t tid=threadIdx.x, lane=tid&31u, warp=tid>>5;
    T v=(tid<(uint32_t)n) ? (EXCLUSIVE ? (tid ? input[tid-1] : (T)0) : input[tid]) : (T)0;
    unsigned mask=__activemask();
    for(int d=1;d<32;d*=2){T q=__shfl_up_sync(mask,v,d);if((int)lane>=d)v+=q;}
    uint32_t last=(warp+1u)*32u-1u;
    if(last>=(uint32_t)blockDim.x) last=blockDim.x-1u;
    if(tid==last) output[warp]=v;
    __syncthreads();
    uint32_t nw=(blockDim.x+31u)/32u;
    if(warp==0){
        T w=(lane<nw)?output[lane]:(T)0;
        unsigned wmask=__activemask();
        for(int d=1;d<32;d*=2){T q=__shfl_up_sync(wmask,w,d);if((int)lane>=d)w+=q;}
        if(lane<nw) output[lane]=w;
    }
    __syncthreads();
    T offset=warp?output[warp-1]:(T)0;
    __syncthreads();
    if(tid<(uint32_t)n) output[tid]=v+offset;
    __syncthreads();
}

template <typename T, uint32_t N, bool SHUFFLE>
__global__ void k_prefix(T* out) {
    extern __shared__ double raw[];
    T* input=reinterpret_cast<T*>(raw); T* output=input+N;
    if(threadIdx.x<N) input[threadIdx.x]=(T)(1+(threadIdx.x%7));
    __syncthreads();
    if constexpr(SHUFFLE) prefix_shuffle<T,false>(input,output,N);
    else glass::block::prefix_sum_inclusive<T>(input,output,N);
    if(threadIdx.x==0) out[blockIdx.x]=output[N-1];
}

template <typename T, uint32_t N>
__device__ void getrf_parallel(T* A, uint32_t* piv, T* vals, uint32_t* idx) {
    uint32_t rank=threadIdx.x, size=blockDim.x;
    for(uint32_t k=0;k<N;k++){
        T best=(T)-1; uint32_t bi=N;
        for(uint32_t i=k+rank;i<N;i+=size){T v=A[i+k*N];v=v<(T)0?-v:v;
            if(v>best || (v==best && i<bi)){best=v;bi=i;}}
        vals[rank]=best;idx[rank]=bi;__syncthreads();
        for(uint32_t s=(size+1)/2;s>0;s=(s+1)/2){
            if(rank<s && rank+s<size){T ov=vals[rank+s];uint32_t oi=idx[rank+s];
                if(ov>vals[rank] || (ov==vals[rank] && oi<idx[rank])){vals[rank]=ov;idx[rank]=oi;}}
            __syncthreads(); if(s==1) break;
        }
        if(rank==0)piv[k]=idx[0];__syncthreads();uint32_t p=piv[k];
        if(p!=k)for(uint32_t j=rank;j<N;j+=size){T q=A[k+j*N];A[k+j*N]=A[p+j*N];A[p+j*N]=q;}
        __syncthreads();T pv=A[k+k*N];
        for(uint32_t i=k+1+rank;i<N;i+=size)A[i+k*N]/=pv;__syncthreads();
        uint32_t rem=N-1-k;
        for(uint32_t e=rank;e<rem*rem;e+=size){uint32_t i=k+1+e%rem,j=k+1+e/rem;
            A[i+j*N]-=A[i+k*N]*A[k+j*N];}
        __syncthreads();
    }
}

template <typename T, uint32_t N, bool PARALLEL>
__global__ void k_getrf(T* out) {
    extern __shared__ double raw[];
    T* A=reinterpret_cast<T*>(raw); T* vals=A+N*N;
    uint32_t* piv=reinterpret_cast<uint32_t*>(vals+blockDim.x);
    uint32_t* idx=piv+N;
    for(uint32_t i=threadIdx.x;i<N*N;i+=blockDim.x){uint32_t r=i%N,c=i/N;
        // A cyclically permuted, diagonally dominant matrix forces real row
        // exchanges instead of benchmarking the pivot scan on SPD-like data.
        A[i]=(r==(c+1)%N)?(T)2:(T)(0.001*(1+(r*7+c*3)%19));}
    __syncthreads();
    if constexpr(PARALLEL)getrf_parallel<T,N>(A,piv,vals,idx);
    else glass::block::getrf<T,N>(A,piv);
    if(threadIdx.x==0)out[blockIdx.x]=A[0];
}

template <typename A, typename B>
static void report(const char* name,A baseline,B candidate,int reps){
    static bool candidate_first=false;
    candidate_first=!candidate_first;
    double a,as,b,bs;
    if(candidate_first){
        b=tc_time_ns_per_prob(candidate,reps,NPROB);bs=tc_last_spread_pct();
        a=tc_time_ns_per_prob(baseline,reps,NPROB);as=tc_last_spread_pct();
    }else{
        a=tc_time_ns_per_prob(baseline,reps,NPROB);as=tc_last_spread_pct();
        b=tc_time_ns_per_prob(candidate,reps,NPROB);bs=tc_last_spread_pct();
    }
    if(a>=1e29||b>=1e29){   // launch-failure sentinel, not a time
        printf("AB %-20s FAIL baseline=%s candidate=%s (launch failed; no ratio)\n",
               name,a>=1e29?"FAIL":"ok",b>=1e29?"FAIL":"ok");
        exit(4);
    }
    printf("AB %-20s baseline=%.3f spread=%.2f%% candidate=%.3f spread=%.2f%% ratio=%.3f\n",
           name,a,as,b,bs,a/b);
}

template <typename T>
static void run_dtype(int reps){
    printf("# mapping dtype=%s\n",sizeof(T)==sizeof(double)?"f64":"f32");
    T* out;CK(cudaMalloc(&out,(size_t)NPROB*sizeof(T)));
#define GER_CASE(M,N,TB) report("ger_" #M "x" #N, \
    [&]{k_ger<T,M,N,false><<<NPROB,TB,(M+N+M*N)*sizeof(T)>>>(out);}, \
    [&]{k_ger<T,M,N,true ><<<NPROB,TB,(M+N+M*N)*sizeof(T)>>>(out);},reps)
    GER_CASE(4,128,128); GER_CASE(8,64,128); GER_CASE(64,8,128); GER_CASE(32,32,128);
#undef GER_CASE
#define TR_CASE(R,C,TB) report("transpose_" #R "x" #C, \
    [&]{k_transpose<T,R,C,false><<<NPROB,TB,(2*R*C+16*17)*sizeof(T)>>>(out);}, \
    [&]{k_transpose<T,R,C,true ><<<NPROB,TB,(2*R*C+16*17)*sizeof(T)>>>(out);},reps)
    TR_CASE(16,16,128); TR_CASE(32,32,128); TR_CASE(16,64,128); TR_CASE(64,16,128);
#undef TR_CASE
#define PS_CASE(N,TB) report("prefix_" #N, \
    [&]{k_prefix<T,N,false><<<NPROB,TB,2*N*sizeof(T)>>>(out);}, \
    [&]{k_prefix<T,N,true ><<<NPROB,TB,2*N*sizeof(T)>>>(out);},reps)
    PS_CASE(32,32); PS_CASE(64,64); PS_CASE(128,128); PS_CASE(256,256);
#undef PS_CASE
    if constexpr(sizeof(T)==sizeof(double)){
        constexpr int smem96=96*96*sizeof(T)+128*sizeof(T)+(96+128)*sizeof(uint32_t);
        CK(cudaFuncSetAttribute(k_getrf<T,96,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem96));
        CK(cudaFuncSetAttribute(k_getrf<T,96,true>, cudaFuncAttributeMaxDynamicSharedMemorySize,smem96));
    }
#define LU_CASE(N,TB) report("getrf_pivot_" #N, \
    [&]{k_getrf<T,N,false><<<NPROB,TB,N*N*sizeof(T)+TB*sizeof(T)+(N+TB)*sizeof(uint32_t)>>>(out);}, \
    [&]{k_getrf<T,N,true ><<<NPROB,TB,N*N*sizeof(T)+TB*sizeof(T)+(N+TB)*sizeof(uint32_t)>>>(out);},reps)
    LU_CASE(16,32); LU_CASE(32,32); LU_CASE(64,64); LU_CASE(96,128);
#undef LU_CASE
    cudaFree(out);
}

int main(int argc,char**argv){
    NPROB=argc>1?atoi(argv[1]):8192;int reps=argc>2?atoi(argv[2]):20;
    const char* dt=argc>3?argv[3]:"both";
    printf("# mapping A/B NPROB=%d reps=%d dtype=%s\n",NPROB,reps,dt);tc_warm_gpu();
    if(!strcmp(dt,"f64")||!strcmp(dt,"both"))run_dtype<double>(reps);
    if(!strcmp(dt,"f32")||!strcmp(dt,"both"))run_dtype<float>(reps);
    return 0;
}
