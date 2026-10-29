#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "glass-cgrps.cuh"
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
using glass::FillMode; using glass::Diag;
template<class T,int N,int OP> __device__ __forceinline__ void call(T* RESTRICT a,T* RESTRICT b,T* RESTRICT s,unsigned* RESTRICT p){
 if constexpr(OP==0)glass::block::trsv<T,N>(a,b);
 if constexpr(OP==1)glass::block::trsv<T>(N,a,b);
 if constexpr(OP==2)glass::block::trsm<T,N,2>(a,b);
 if constexpr(OP==3)glass::block::potrf<T>(N,a);
 if constexpr(OP==4)glass::block::potrs<T,N>(a,b);
 if constexpr(OP==5)glass::block::posv<T,N>(a,b);
 if constexpr(OP==6)glass::block::potrs<T,N,2>(a,b);
 if constexpr(OP==7)glass::block::posv<T,N,2>(a,b);
 if constexpr(OP==8)glass::block::inv<T,N>(a,s);
 if constexpr(OP==9)glass::block::gesv<T,N,2>(a,p,b);
 if constexpr(OP==10){glass::block::ldlt<T,N>(a,s);glass::block::ldlt_solve<T,N>(a,b);}
 if constexpr(OP==11)glass::block::reduce<T,N>(b);
 if constexpr(OP==12)glass::block::nrm2_impl<glass::block::BlockBarrier,T>(glass::block::BlockBarrier{},N,b);
 if constexpr(OP==13)glass::block::softmax<T>(N,T(1),b,a,s);
 if constexpr(OP==14)glass::cgrps::potrf<T>(N,a);
 if constexpr(OP==15)glass::cgrps::trsm<T>(N,2,a,b);
 if constexpr(OP==16)glass::warp::trsv<T,N>(a,b);
 if constexpr(OP==17)glass::warp::potrf<T,N>(a);
 if constexpr(OP==18)glass::block::logsumexp<T>(N,T(1),b,a,s);
 if constexpr(OP==19)glass::block::vector_norm<T>(N,b,a);
 if constexpr(OP==20)glass::block::prefix_sum_inclusive<T>(b,a,N);
 if constexpr(OP==21)glass::cgrps::reduce<T,N>(b);
 if constexpr(OP==22)glass::block::potrf<T,N>(a);
 if constexpr(OP==23)glass::block::inv_pivoted<T,N>(a,s);
 if constexpr(OP==24){glass::block::ldlt<T,N>(a,s,true,reinterpret_cast<int32_t*>(p));glass::block::ldlt_solve<T,N>(a,b,reinterpret_cast<int32_t*>(p));}
 if constexpr(OP==25)glass::block::trsm<T,N,1>(a,b);
 if constexpr(OP==26)glass::block::potrs<T,N,1>(a,b);
 if constexpr(OP==27)glass::block::posv<T,N,1>(a,b);
 if constexpr(OP==28)glass::softmax<T>(N,T(1),b,a,s);
}
template<class T,int N,int OP> __global__ void kernel(const T* inA,const T* inB,T* out){
 __shared__ T a[2*N*N],b[2*N],s[4*N*N]; __shared__ unsigned p[N];
 for(int i=threadIdx.x;i<2*N*N;i+=blockDim.x)a[i]=inA[i];
 for(int i=threadIdx.x;i<2*N;i+=blockDim.x)b[i]=inB[i];
 for(int i=threadIdx.x;i<4*N*N;i+=blockDim.x)s[i]=T(-777);
 __syncthreads(); call<T,N,OP>(a,b,s,p); __syncthreads();
 for(int i=threadIdx.x;i<2*N*N;i+=blockDim.x)out[i]=a[i];
 for(int i=threadIdx.x;i<2*N;i+=blockDim.x)out[2*N*N+i]=b[i];
}
const char* names[]={"trsv_ct","trsv_rt","trsm","potrf","potrs_1","posv_1","potrs_2","posv_2","inv","gesv","ldlt_solve","reduce","nrm2","softmax","cgrps_potrf","cgrps_trsm","warp_trsv","warp_potrf","logsumexp","vector_norm","prefix_inclusive","cgrps_reduce","potrf_ct","inv_pivoted","ldlt_pivot_solve","trsm_1","potrs_multi1","posv_multi1","bare_softmax"};
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
template<class T,int N,int OP> void run(){
 for(int seed=0;seed<3;seed++){
 std::vector<T> a(2*N*N),b(2*N),out(2*N*N+2*N); std::vector<double>L(N*N),A(N*N),X(2*N);
 for(int j=0;j<N;j++)for(int i=j;i<N;i++)L[i+j*N]=i==j ? 2.0+(i%3) : (((i*7+j*3+seed)%9)-4)*0.13;
 for(int j=0;j<N;j++)for(int i=0;i<N;i++)for(int k=0;k<N;k++)A[i+j*N]+=L[i+k*N]*L[j+k*N];
 for(int i=0;i<2*N;i++)X[i]=((i*3+seed)%11-5)*0.17;
 bool tri=OP==0||OP==1||OP==2||OP==4||OP==6||OP==15||OP==16||OP==25||OP==26;
 for(int i=0;i<N*N;i++)a[i]=tri?L[i]:A[i];
 for(int i=0;i<N;i++)a[N*N+i+i*N]=1;
 for(int c=0;c<2;c++)for(int i=0;i<N;i++)for(int j=0;j<N;j++)b[i+c*N]+=T((OP==0||OP==1||OP==2||OP==15||OP==16||OP==25?L[i+j*N]:A[i+j*N])*X[j+c*N]);
 if constexpr((OP>=11&&OP<=13)||(OP>=18&&OP<=21)||OP==28)for(int i=0;i<N;i++)b[i]=T(X[i]);
 T *da,*db,*dout;ck(cudaMalloc(&da,a.size()*sizeof(T)));ck(cudaMalloc(&db,b.size()*sizeof(T)));ck(cudaMalloc(&dout,out.size()*sizeof(T)));
 ck(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
 for(int th:{1,2,3,7,16,31,32,33,64,96,128}){
 if((OP==16||OP==17)&&th!=32)continue; if(OP==20&&th<N)continue;
 kernel<T,N,OP><<<1,th>>>(da,db,dout);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dout,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
 double err=0;auto check=[&](double actual,double expected){double e=std::abs(actual-expected)/(1+std::abs(expected));if(!std::isfinite(e))e=1e300;err=std::max(err,e);};
 if constexpr(OP==3||OP==14||OP==17||OP==22){for(int j=0;j<N;j++)for(int i=j;i<N;i++)check(out[i+j*N],L[i+j*N]);}
 else if constexpr(OP==8||OP==23){for(int j=0;j<N;j++)for(int i=0;i<N;i++){double v=0;for(int k=0;k<N;k++)v+=A[i+k*N]*out[N*N+k+j*N];check(v,i==j?1:0);}}
 else if constexpr(OP==11||OP==12||OP==21){double v=0;for(int i=0;i<N;i++)v+=(OP==11||OP==21)?double(b[i]):double(b[i])*b[i];check(out[2*N*N],(OP==11||OP==21)?v:sqrt(v));}
 else if constexpr(OP==13||OP==28){double v=0;for(int i=0;i<N;i++)v+=exp(double(b[i]));for(int i=0;i<N;i++)check(out[i],exp(double(b[i]))/v);}
 else if constexpr(OP==18){double v=0;for(int i=0;i<N;i++)v+=exp(double(b[i]));check(out[0],log(v));}
 else if constexpr(OP==19){double v=0;for(int i=0;i<N;i++)v+=double(b[i])*b[i];check(out[0],sqrt(v));}
 else if constexpr(OP==20){double v=0;for(int i=0;i<N;i++){v+=b[i];check(out[i],v);}}
 else {constexpr int C=(OP==2||OP==6||OP==7||OP==9||OP==15)?2:1;for(int i=0;i<N*C;i++)check(out[2*N*N+i],X[i]);}
 printf("%s,%s,%d,%d,%d,%.9g,%s\n",names[OP],sizeof(T)==4?"f32":"f64",N,seed,th,err,err>(sizeof(T)==4?2e-5:1e-11)?"FAIL":"PASS");
 }
 ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(dout));
 }
}
template<class T,int N,int O=0>void ops(){run<T,N,O>();if constexpr(O<28)ops<T,N,O+1>();}
int main(){setvbuf(stdout,nullptr,_IONBF,0);puts("op,dtype,n,seed,threads,error,result");ops<float,3>();ops<double,3>();ops<float,7>();ops<double,7>();ops<float,17>();ops<double,17>();}
