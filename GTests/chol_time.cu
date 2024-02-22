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
void cholDecomp_InPlace (int n, float *s_A, int reps)
{
	for (int i = 0; i < reps; i++) {
		glass::chol_InPlace(n, s_A);
		__syncthreads();
	}
}


int main(int argc, char *argv[]) {
	float total_time, avg_time;
	struct timespec start, end;
	float *d_d;
	long i;

	if (argc != 3) {
		printf("./<executable> <dim of matrix> <num of reps>\n");
		exit(-1);
	}
	long len = atol(argv[1]);
	long reps = atol(argv[2]);
	float *h_d =(float*)malloc(sizeof(float)*pow(len,2));
    float *res = (float*)malloc(sizeof(float)*pow(len,2));

	for (i = 0; i < pow(len,2); i++) {
		if (i % (len+1) == 0) h_d[i] = 1;
		else h_d[i] = 0;
	}

	gpuErrchk(cudaMalloc(&d_d, pow(len,2) * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_d, h_d, pow(len,2) * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	clock_gettime(CLOCK_MONOTONIC, &start);
	cholDecomp_InPlace<<<1,len>>>(len, d_d, reps);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
	clock_gettime(CLOCK_MONOTONIC, &end);
	total_time = (float)end.tv_sec + (float)end.tv_nsec*1e-9
					- (float)start.tv_sec - (float)start.tv_nsec*1e-9;
	cudaFree(d_d);
	free(h_d);

	avg_time = total_time/reps;
	printf("avg time:   %.10fs\n", avg_time);
	return 0;
}
