#include <cstdio>
#include "glass.cuh"
#ifndef AUDIT_RESTRICT
#define AUDIT_RESTRICT __restrict__
#endif
__device__ __forceinline__ void f(const double* AUDIT_RESTRICT x,double* AUDIT_RESTRICT y,double* AUDIT_RESTRICT scratch){glass::block::softmax<double>(3,1.,x,y,scratch);}
__global__ void k(double*out){__shared__ double x[3],y[3],s[3];if(threadIdx.x==0){x[0]=0;x[1]=0;x[2]=0;}__syncthreads();f(x,y,s);__syncthreads();if(threadIdx.x==0)for(int i=0;i<3;i++)out[i]=y[i];}
int main(){double *d,h[3];cudaMalloc(&d,24);for(int t=1;t<=2;t++){k<<<1,t>>>(d);auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){puts(cudaGetErrorString(e));return 1;}cudaMemcpy(h,d,24,cudaMemcpyDeviceToHost);printf("threads=%d: %.17g %.17g %.17g; sum=%.17g (want 1/3 each)\n",t,h[0],h[1],h[2],h[0]+h[1]+h[2]);}cudaFree(d);}
