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
 if constexpr(OP==29)glass::warp::softmax<T>(N,T(1),b,a);
 if constexpr(OP==30)glass::block::syev<T,N>(a,b,a+N*N,s);
 if constexpr(OP==28)glass::softmax<T>(N,T(1),b,a,s);
}
template<class T,int N,int OP> __global__ void kernel(const T* inA,const T* inB,T* out){
 out += blockIdx.x*(2*N*N+2*N);
 __shared__ T a[2*N*N],b[2*N],s[4*N*N]; __shared__ unsigned p[N];
 for(int i=threadIdx.x;i<2*N*N;i+=blockDim.x)a[i]=inA[i];
 for(int i=threadIdx.x;i<2*N;i+=blockDim.x)b[i]=inB[i];
 for(int i=threadIdx.x;i<4*N*N;i+=blockDim.x)s[i]=T(-777);
 __syncthreads(); call<T,N,OP>(a,b,s,p); __syncthreads();
 for(int i=threadIdx.x;i<2*N*N;i+=blockDim.x)out[i]=a[i];
 for(int i=threadIdx.x;i<2*N;i+=blockDim.x)out[2*N*N+i]=b[i];
}
const char* names[]={"trsv_ct","trsv_rt","trsm","potrf","potrs_1","posv_1","potrs_2","posv_2","inv","gesv","ldlt_solve","reduce","nrm2","softmax","cgrps_potrf","cgrps_trsm","warp_trsv","warp_potrf","logsumexp","vector_norm","prefix_inclusive","cgrps_reduce","potrf_ct","inv_pivoted","ldlt_pivot_solve","trsm_1","potrs_multi1","posv_multi1","bare_softmax","warp_softmax","syev"};
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
int failures=0; int selected_seed=-1, selected_threads=-1;
template<class T,int N,int OP> void run(){
 for(int seed=0;seed<3;seed++){
 if(selected_seed>=0 && seed!=selected_seed)continue;
 std::vector<T> a(2*N*N),b(2*N),out(2*N*N+2*N); std::vector<double>L(N*N),A(N*N),X(2*N);
 for(int j=0;j<N;j++)for(int i=j;i<N;i++)L[i+j*N]=i==j ? 2.0+(i%3) : (((i*7+j*3+seed)%9)-4)*0.13;
 for(int j=0;j<N;j++)for(int i=0;i<N;i++)for(int k=0;k<N;k++)A[i+j*N]+=L[i+k*N]*L[j+k*N];
 for(int i=0;i<2*N;i++)X[i]=((i*3+seed)%11-5)*0.17;
 bool tri=OP==0||OP==1||OP==2||OP==4||OP==6||OP==15||OP==16||OP==25||OP==26;
 for(int i=0;i<N*N;i++)a[i]=tri?L[i]:A[i];
 for(int i=0;i<N;i++)a[N*N+i+i*N]=1;
 for(int c=0;c<2;c++)for(int i=0;i<N;i++)for(int j=0;j<N;j++)b[i+c*N]+=T((OP==0||OP==1||OP==2||OP==15||OP==16||OP==25?L[i+j*N]:A[i+j*N])*X[j+c*N]);
 if constexpr((OP>=11&&OP<=13)||(OP>=18&&OP<=21)||OP==28||OP==29)for(int i=0;i<N;i++)b[i]=T(X[i]);
 T *da,*db,*dout;ck(cudaMalloc(&da,a.size()*sizeof(T)));ck(cudaMalloc(&db,b.size()*sizeof(T)));ck(cudaMalloc(&dout,4096*out.size()*sizeof(T)));
 ck(cudaMemcpy(da,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice));
 for(int th:{1,2,3,7,16,31,32,33,64,96,128}){
 if(selected_threads>=0 && th!=selected_threads)continue;
 if((OP==16||OP==17||OP==29)&&th!=32)continue; if(OP==20&&th<N)continue;
 kernel<T,N,OP><<<1,th>>>(da,db,dout);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),dout,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
 cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
 for(int warm=0;warm<10;warm++)kernel<T,N,OP><<<4096,th>>>(da,db,dout);
 ck(cudaDeviceSynchronize());
 for(int trial=0;trial<3;trial++){
  ck(cudaEventRecord(begin));for(int rep=0;rep<30;rep++)kernel<T,N,OP><<<4096,th>>>(da,db,dout);
  ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));
  fprintf(stderr,"%s,%s,%d,%d,%d,%.9g\n",names[OP],sizeof(T)==4?"f32":"f64",N,th,trial,double(ms)*1000/(30*4096));
 }
 ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
 double err=0;auto check=[&](double actual,double expected){double e=std::abs(actual-expected)/(1+std::abs(expected));if(!std::isfinite(e))e=1e300;err=std::max(err,e);};
 if constexpr(OP==3||OP==14||OP==17||OP==22){for(int j=0;j<N;j++)for(int i=j;i<N;i++)check(out[i+j*N],L[i+j*N]);}
 else if constexpr(OP==8||OP==23){for(int j=0;j<N;j++)for(int i=0;i<N;i++){double v=0;for(int k=0;k<N;k++)v+=A[i+k*N]*out[N*N+k+j*N];check(v,i==j?1:0);}}
 else if constexpr(OP==11||OP==12||OP==21){double v=0;for(int i=0;i<N;i++)v+=(OP==11||OP==21)?double(b[i]):double(b[i])*b[i];check(out[2*N*N],(OP==11||OP==21)?v:sqrt(v));}
 else if constexpr(OP==13||OP==28||OP==29){double v=0;for(int i=0;i<N;i++)v+=exp(double(b[i]));for(int i=0;i<N;i++)check(out[i],exp(double(b[i]))/v);}
 else if constexpr(OP==18){double v=0;for(int i=0;i<N;i++)v+=exp(double(b[i]));check(out[0],log(v));}
 else if constexpr(OP==19){double v=0;for(int i=0;i<N;i++)v+=double(b[i])*b[i];check(out[0],sqrt(v));}
 else if constexpr(OP==20){double v=0;for(int i=0;i<N;i++){v+=b[i];check(out[i],v);}}
 else if constexpr(OP==30){for(int j=0;j<N;j++)for(int i=0;i<N;i++){double v=0;for(int k=0;k<N;k++)v+=A[i+k*N]*out[N*N+k+j*N];check(v,out[N*N+i+j*N]*out[2*N*N+j]);}}
 else {constexpr int C=(OP==2||OP==6||OP==7||OP==9||OP==15)?2:1;for(int i=0;i<N*C;i++)check(out[2*N*N+i],X[i]);}
 failures += err>(sizeof(T)==4?2e-5:1e-11);
 printf("%s,%s,%d,%d,%d,%.9g,%s\n",names[OP],sizeof(T)==4?"f32":"f64",N,seed,th,err,err>(sizeof(T)==4?2e-5:1e-11)?"FAIL":"PASS");
 }
 ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(dout));
 }
}
template<class T,int N>void choose(int op){switch(op){case 0:run<T,N,0>();break;case 13:run<T,N,13>();break;case 23:run<T,N,23>();break;case 24:run<T,N,24>();break;case 30:run<T,N,30>();break;default:exit(3);}}
int main(int argc,char**argv){if(argc!=6)return 3;setvbuf(stdout,nullptr,_IONBF,0);int op=atoi(argv[1]),n=atoi(argv[3]);selected_seed=atoi(argv[4]);selected_threads=atoi(argv[5]);if(op<0||op>30)return 3;
#define GO(T) if(n==3)choose<T,3>(op);else if(n==7)choose<T,7>(op);else if(n==17)choose<T,17>(op);else return 3;
if(atoi(argv[2])==32){GO(float)}else {GO(double)}return failures?1:0;}
