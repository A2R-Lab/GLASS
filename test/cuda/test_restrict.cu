// Optimized restricted-caller regression; Python supplies independent numerical oracles.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "glass-cgrps.cuh"
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
using glass::FillMode; using glass::Diag;
template<class T,int N,int OP> __device__ __forceinline__ void call(T* RESTRICT a,T* RESTRICT b,T* RESTRICT s,unsigned* RESTRICT p, unsigned runtime_n){
 if constexpr(OP>=31 && OP<=38)glass::block::trsv<T,N,
     ((OP-31)&1)?FillMode::Upper:FillMode::Lower,
     ((OP-31)&4)?Diag::Unit:Diag::NonUnit, bool((OP-31)&2)>(a,b);
 if constexpr(OP==39){glass::block::trsv<T,N,FillMode::Lower,Diag::NonUnit,false,false>(a,b);__syncthreads();}
 if constexpr(OP==40){glass::block::softmax<T,false>(runtime_n,T(1),b,a,s);__syncthreads();}
 if constexpr(OP==0)glass::block::trsv<T,N>(a,b);
 if constexpr(OP==1)glass::block::trsv<T>(runtime_n,a,b);
 if constexpr(OP==2)glass::block::trsm<T,N,2>(a,b);
 if constexpr(OP==3)glass::block::potrf<T>(runtime_n,a);
 if constexpr(OP==4)glass::block::potrs<T,N>(a,b);
 if constexpr(OP==5)glass::block::posv<T,N>(a,b);
 if constexpr(OP==6)glass::block::potrs<T,N,2>(a,b);
 if constexpr(OP==7)glass::block::posv<T,N,2>(a,b);
 if constexpr(OP==8)glass::block::inv<T,N>(a,s);
 if constexpr(OP==9)glass::block::gesv<T,N,2>(a,p,b);
 if constexpr(OP==10){glass::block::ldlt<T,N>(a,s);glass::block::ldlt_solve<T,N>(a,b);}
 if constexpr(OP==11)glass::block::reduce<T,N>(b);
 if constexpr(OP==12)glass::block::nrm2_impl<glass::block::BlockBarrier,T>(glass::block::BlockBarrier{},N,b);
 if constexpr(OP==13)glass::block::softmax<T>(runtime_n,T(1),b,a,s);
 if constexpr(OP==14)glass::cgrps::potrf<T>(runtime_n,a);
 if constexpr(OP==15)glass::cgrps::trsm<T>(N,2,a,b);
 if constexpr(OP==16)glass::warp::trsv<T,N>(a,b);
 if constexpr(OP==17)glass::warp::potrf<T,N>(a);
 if constexpr(OP==18)glass::block::logsumexp<T>(runtime_n,T(1),b,a,s);
 if constexpr(OP==19)glass::block::vector_norm<T>(N,b,a);
 if constexpr(OP==20)glass::block::prefix_sum_inclusive<T>(b,a,N);
 if constexpr(OP==21)glass::cgrps::reduce<T,N>(b);
 if constexpr(OP==22)glass::block::potrf<T,N>(a);
 if constexpr(OP==23)glass::block::inv_pivoted<T,N>(a,s);
 if constexpr(OP==24){glass::block::ldlt<T,N>(a,s,true,reinterpret_cast<int32_t*>(p));glass::block::ldlt_solve<T,N>(a,b,reinterpret_cast<int32_t*>(p));}
 if constexpr(OP==25)glass::block::trsm<T,N,1>(a,b);
 if constexpr(OP==26)glass::block::potrs<T,N,1>(a,b);
 if constexpr(OP==27)glass::block::posv<T,N,1>(a,b);
 if constexpr(OP==29)glass::warp::softmax<T>(runtime_n,T(1),b,a);
 if constexpr(OP==30)glass::block::syev<T,N>(a,b,a+N*N,s);
 if constexpr(OP==28)glass::softmax<T>(runtime_n,T(1),b,a,s);
}
template<class T,int N,int OP> __global__ void kernel(T* inA,T* inB,T* scratch,unsigned* piv,T* out,unsigned runtime_n,bool global){
 __shared__ T sa[2*N*N],sb[2*N],ss[4*N*N]; __shared__ unsigned sp[N];
 unsigned rank=glass::block::flat_rank(),size=glass::block::flat_size();
 T* a=global?inA:sa; T* b=global?inB:sb; T* s=global?scratch:ss; unsigned* p=global?piv:sp;
 for(unsigned i=rank;i<2*N*N;i+=size)a[i]=inA[i];
 for(unsigned i=rank;i<2*N;i+=size)b[i]=inB[i];
 for(unsigned i=rank;i<4*N*N;i+=size)s[i]=T(-777);
 __syncthreads(); call<T,N,OP>(a,b,s,p,runtime_n);
 // Deliberately no caller barrier: default trailing publication is tested.
 // Rotate ownership so output cannot be satisfied by each writer's own register.
 for(unsigned i=rank;i<2*N*N;i+=size){unsigned j=(i+1)%(2*N*N);out[j]=a[j];}
 for(unsigned i=rank;i<2*N;i+=size){unsigned j=(i+1)%(2*N);out[2*N*N+j]=b[j];}
}
template<class T,int N,int OP> __global__ void kernel_repro(const T* inA,const T* inB,T* out){
 __shared__ T a[2*N*N],b[2*N],s[4*N*N]; __shared__ unsigned p[N];
 for(int i=threadIdx.x;i<2*N*N;i+=blockDim.x)a[i]=inA[i];
 for(int i=threadIdx.x;i<2*N;i+=blockDim.x)b[i]=inB[i];
 for(int i=threadIdx.x;i<4*N*N;i+=blockDim.x)s[i]=T(-777);
 __syncthreads(); call<T,N,OP>(a,b,s,p,N); __syncthreads();
 for(int i=threadIdx.x;i<2*N*N;i+=blockDim.x)out[i]=a[i];
 for(int i=threadIdx.x;i<2*N;i+=blockDim.x)out[2*N*N+i]=b[i];
}

void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
template<class T,int N,int OP> void run(int seed,int th,int global,bool multidim){
 std::vector<T> a(2*N*N),b(2*N),out(2*N*N+2*N); std::vector<double>L(N*N),A(N*N),X(2*N);
 for(int j=0;j<N;j++)for(int i=j;i<N;i++)L[i+j*N]=i==j ? 2.0+(i%3) : (((i*7+j*3+seed)%9)-4)*0.13;
 for(int j=0;j<N;j++)for(int i=0;i<N;i++)for(int k=0;k<N;k++)A[i+j*N]+=L[i+k*N]*L[j+k*N];
 if(seed>=10 && (OP==24 || OP==30)) {
  std::fill(A.begin(),A.end(),0.0);A[1]=A[N]=2;
  for(int i=2;i<N;i++)A[i+i*N]=(i%2?-1:1)*(i+1.0);
 }
 if(seed>=10 && OP==23)for(int j=0;j<N;j++)std::swap(A[j*N],A[1+j*N]);
 for(int i=0;i<2*N;i++)X[i]=((i*3+seed)%11-5)*0.17;
 bool tri=OP==0||OP==1||OP==2||OP==4||OP==6||OP==15||OP==16||OP==25||OP==26||OP==39||(OP>=31&&OP<=38);
 for(int i=0;i<N*N;i++)a[i]=tri?L[i]:A[i];
 for(int i=0;i<N;i++)a[N*N+i+i*N]=1;
 for(int c=0;c<2;c++)for(int i=0;i<N;i++)for(int j=0;j<N;j++)b[i+c*N]+=T((OP==0||OP==1||OP==2||OP==15||OP==16||OP==25||OP==39?L[i+j*N]:A[i+j*N])*X[j+c*N]);
 if constexpr((OP>=11&&OP<=13)||(OP>=18&&OP<=21)||OP==28||OP==29)for(int i=0;i<N;i++)b[i]=T(X[i]);
 if constexpr(OP>=31&&OP<=38){
  constexpr bool upper=(OP-31)&1,transpose=(OP-31)&2,unit=(OP-31)&4;
  for(int j=0;j<N;j++)for(int i=0;i<N;i++)a[i+j*N]=i==j&&unit?T(1):T(upper?L[j+i*N]:L[i+j*N]);
  for(int c=0;c<2;c++)for(int i=0;i<N;i++){
   b[i+c*N]=0;for(int j=0;j<N;j++)b[i+c*N]+=T((transpose?a[j+i*N]:a[i+j*N])*X[j+c*N]);
  }
 }
 if constexpr(OP==40)for(int i=0;i<N;i++)b[i]=T(X[i]);
 if(seed>=10 && (OP==13||OP==18||OP==28||OP==29||OP==40))for(int i=0;i<N;i++)b[i]+=T(1000);
 T *da,*db,*dout,*ds; unsigned* dp;ck(cudaMalloc(&ds,4*N*N*sizeof(T)));ck(cudaMalloc(&dp,N*sizeof(unsigned)));ck(cudaMalloc(&da,a.size()*sizeof(T)));ck(cudaMalloc(&db,b.size()*sizeof(T)));ck(cudaMalloc(&dout,out.size()*sizeof(T)));
 ck(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));

 dim3 block=multidim?dim3(1,th,1):dim3(th);
 if(global==2)kernel_repro<T,N,OP><<<1,th>>>(da,db,dout);
 else kernel<T,N,OP><<<1,block>>>(da,db,ds,dp,dout,N,global);
 ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dout,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
 for(T value:out)printf("%.17g ",double(value));puts("");
 ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(ds));ck(cudaFree(dp));ck(cudaFree(dout));
}
template<class T,int N,int O=0>void choose(int op,int seed,int th,int global,bool multidim){if(op==O)run<T,N,O>(seed,th,global,multidim);else if constexpr(O<40)choose<T,N,O+1>(op,seed,th,global,multidim);}
int main(int argc,char**argv){if(argc!=8)return 3;int op=atoi(argv[1]),bits=atoi(argv[2]),n=atoi(argv[3]),seed=atoi(argv[4]),th=atoi(argv[5]);int global=atoi(argv[6]);bool multidim=atoi(argv[7]);if(op<0||op>40||(bits!=32&&bits!=64)||th<1||th>1024)return 3;
if((op==16||op==17||op==29)&&th!=32)return 3;if(op==20&&th<n)return 3;
#define GO(T) if(n==3)choose<T,3>(op,seed,th,global,multidim);else if(n==7)choose<T,7>(op,seed,th,global,multidim);else if(n==17)choose<T,17>(op,seed,th,global,multidim);else return 3;
if(bits==32){GO(float)}else {GO(double)}return 0;}
