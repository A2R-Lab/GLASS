#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../glass.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ 
void global_gemv (int m, int n, float alpha, float *A, float *v,
				float beta, float *y, int reps)
{
	for (int i = 0; i < reps; i++) {
		glass::gemv<float, false>(m,n,alpha,A,v,beta,y);
		__syncthreads();
	}
}

__global__ 
void global_gemv_vecshared (int m, int n, float alpha, float *A, 
				float beta, float *y, int reps)
{
	extern __shared__ float vec[];
	if (threadIdx.x == 0) {
		for (int i = 0; i < n; i++) {
			vec[i] = 1.0;
		}
	}
	__syncthreads();
	for (int i = 0; i < reps; i++) {
		glass::gemv<float, false>(m,n,alpha,A,vec,beta,y);
		__syncthreads();
	}
}

__global__ 
void global_gemv_bothshared (int m, int n, float alpha,
				float beta, float *y, int reps)
{
	extern __shared__ float mat_vec[];
	if (threadIdx.x == 0) {
		for (int i = 0; i < m*n; i++) {
			mat_vec[i] = 1.0;
		}
	}
	__syncthreads();
	for (int i = 0; i < reps; i++) {
		glass::gemv<float, false>(m,n,alpha,mat_vec,mat_vec,beta,y);
		__syncthreads();
	}
}



int main(int argc, char *argv[]) {
	double total_time, avg_time, total_time_vec, 
			avg_time_vec, total_time_both, avg_time_both; 
	struct timespec start, end;
	float *d_m, *d_v, *d_res;

	if (argc != 4) {
		printf("Usage: ./executable <m> <n> <reps>\n");
		exit(-1);
	}

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int reps = atoi(argv[3]);
	float *h_m =(float*)malloc(sizeof(float)*m*n);
	float *h_v = (float*)malloc(sizeof(float)*n);
	float *h_res = (float*)malloc(sizeof(float)*m);

	for (int i = 0; i < m*n; i++) {
		h_m[i] = 1;
	}
	for (int i = 0; i < n; i++) {
		h_v[i] = 1;
	}

	gpuErrchk(cudaMalloc(&d_m, m*n * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_v, n * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_res, m * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_m, h_m, m*n * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_v, h_v, n * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	// both in device
	clock_gettime(CLOCK_MONOTONIC, &start);
	global_gemv<<<1,m>>>(m, n, static_cast<float>(1), d_m, d_v, static_cast<float>(1), d_res, reps);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_MONOTONIC, &end);
	total_time = (double)end.tv_sec + (double)end.tv_nsec*1e-9
				- (double)start.tv_sec - (double)start.tv_nsec*1e-9;

	// vec in shared
	clock_gettime(CLOCK_MONOTONIC, &start);
	global_gemv_vecshared<<<1,m, n*sizeof(float)>>>(m, n, static_cast<float>(1), d_m, static_cast<float>(1), d_res, reps);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_MONOTONIC, &end);
	total_time_vec = (double)end.tv_sec + (double)end.tv_nsec*1e-9
				- (double)start.tv_sec - (double)start.tv_nsec*1e-9;

	// both in shared
	cudaFuncSetAttribute(global_gemv_bothshared, cudaFuncAttributeMaxDynamicSharedMemorySize, 99000);
	clock_gettime(CLOCK_MONOTONIC, &start);
	global_gemv_bothshared<<<1,m,(m*n)*sizeof(float)>>>(m, n, static_cast<float>(1), static_cast<float>(1), d_res, reps);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_MONOTONIC, &end);
	total_time_both = (double)end.tv_sec + (double)end.tv_nsec*1e-9
				- (double)start.tv_sec - (double)start.tv_nsec*1e-9;

	cudaFree(d_m); cudaFree(d_v); cudaFree(d_res);
	free(h_m); free(h_v);

	// print results
	avg_time = total_time/reps*1e9;
	avg_time_vec = total_time_vec/reps*1e9;
	avg_time_both = total_time_both/reps*1e9;
	printf("avg time (both in device):   %.6f ns\n", avg_time);
	printf("avg time (vec in shared):    %.6f ns\n", avg_time_vec);
	printf("avg time (both in shared):   %.6f ns\n", avg_time_both);
	return 0;
}
