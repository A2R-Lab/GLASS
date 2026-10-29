#include <cstdio>
#include <cstdint>
#include <math.h>
#include <cuda_runtime.h>
namespace glass { namespace block {
struct BlockBarrier {
__device__ __forceinline__ uint32_t rank()const{return threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;}
__device__ __forceinline__ uint32_t size()const{return blockDim.x*blockDim.y*blockDim.z;}
__device__ __forceinline__ void sync()const{__syncthreads();}
};
__device__ __forceinline__ uint32_t flat_rank(){return BlockBarrier{}.rank();}
__device__ __forceinline__ uint32_t flat_size(){return BlockBarrier{}.size();}
template <typename Bar, typename T, bool TRAILING_SYNC = true>
__device__ void reduce_impl(Bar bar, uint32_t n, T *x)
{
    uint32_t rank = bar.rank(), size = bar.size();
    uint32_t left = n;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        bar.sync();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
    if constexpr (TRAILING_SYNC) bar.sync();
}

namespace softmax_detail {
    // in-place halving-tree MAX reduce — the reduce_impl tree with `+` replaced
    // by max (same fixed combine order; result in x[0]).
    template <typename Bar, typename T, bool TRAILING_SYNC = true>
    __device__ void max_tree_impl(Bar bar, uint32_t n, T *x)
    {
        uint32_t rank = bar.rank(), size = bar.size();
        uint32_t left = n;
        while (left > 3) {
            bool odd = left % 2;
            left = (left - odd) / 2;
            for (uint32_t i = rank; i < left; i += size)
                x[i] = (x[i + left] > x[i]) ? x[i + left] : x[i];
            if (rank == 0 && odd) x[0] = (x[2*left] > x[0]) ? x[2*left] : x[0];
            bar.sync();
        }
        if (rank == 0) {
            for (uint32_t i = 1; i < left; i++) x[0] = (x[i] > x[0]) ? x[i] : x[0];
        }
        if constexpr (TRAILING_SYNC) bar.sync();
    }

}
template <typename T, bool TRAILING_SYNC = true>
__device__ void softmax(uint32_t n, T alpha, const T *x, T *y, T *s_scratch)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    for (uint32_t i = rank; i < n; i += size) s_scratch[i] = alpha*x[i];
    __syncthreads();
    softmax_detail::max_tree_impl<BlockBarrier, T>(BlockBarrier{}, n, s_scratch);
    const T m = s_scratch[0];
    __syncthreads();                       // everyone read m before scratch is rewritten
    // unroll 1: unroll-copy FMA contraction of alpha*x[i] - m would break
    // bit-identity across thread counts.
    #pragma unroll 1
    for (uint32_t i = rank; i < n; i += size) {
        const T e = exp(alpha*x[i] - m);
        y[i] = e;
        s_scratch[i] = e;
    }
    __syncthreads();
    reduce_impl<BlockBarrier, T>(BlockBarrier{}, n, s_scratch);
    const T inv = static_cast<T>(1)/s_scratch[0];
    for (uint32_t i = rank; i < n; i += size) y[i] *= inv;
    if constexpr (TRAILING_SYNC) __syncthreads();
}

}}
#ifndef AUDIT_RESTRICT
#define AUDIT_RESTRICT __restrict__
#endif
__device__ __forceinline__ void f(const double* AUDIT_RESTRICT x,double* AUDIT_RESTRICT y,double* AUDIT_RESTRICT scratch){glass::block::softmax<double>(3,1.,x,y,scratch);}
__global__ void k(double*out){__shared__ double x[3],y[3],s[3];if(threadIdx.x==0){x[0]=0;x[1]=0;x[2]=0;}__syncthreads();f(x,y,s);__syncthreads();if(threadIdx.x==0)for(int i=0;i<3;i++)out[i]=y[i];}
int main(){double *d,h[3];cudaMalloc(&d,24);for(int t=1;t<=2;t++){k<<<1,t>>>(d);auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){puts(cudaGetErrorString(e));return 1;}cudaMemcpy(h,d,24,cudaMemcpyDeviceToHost);printf("threads=%d: %.17g %.17g %.17g; sum=%.17g (want 1/3 each)\n",t,h[0],h[1],h[2],h[0]+h[1]+h[2]);}cudaFree(d);}
