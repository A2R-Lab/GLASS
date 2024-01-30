#include <iostream>
#include <cuda_runtime.h>

#include "../glass.cuh"
#include "./global_glass.cuh"
#include "gtest/gtest.h"


class L1Test : public ::testing::Test{

	protected:
		void SetUp() override {
			n = 100;
			h_a = new int[n];
			h_b = new int[n];
			h_c = new int;
			for(int i = 0; i < n; i++){
					h_a[i] = i;
					h_b[i] = 2 * i;
			}
			cudaMalloc(&d_a, n * sizeof(int));
			cudaMalloc(&d_b, n * sizeof(int));
			cudaMalloc(&d_c, sizeof(int));
			cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

	int n;
	int * h_a;
	int * h_b;
	int * h_c;
	int * d_a, *d_b, *d_c;
};

class L2Test : public ::testing::Test{

	protected:
		void SetUp() override {
			m = 5;
			n = 7;
			h_a = new int[m*n];
			h_b = new int[n];
			h_c = new int[m];
			for(int i = 0; i < m*n; i++){
					h_a[i] = i;
			}
			for (size_t i = 0; i < n; i++)
			{
				h_b[i] = 2 * i;
			}
			cudaMalloc(&d_a, m*n * sizeof(int));
			cudaMalloc(&d_b, n * sizeof(int));
			cudaMalloc(&d_c, m * sizeof(int));
			cudaMemcpy(d_a, h_a, m*n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

	int n, m;
	int * h_a;
	int * h_b;
	int * h_c;
	int * d_a, *d_b, *d_c;
};

class L3Test : public ::testing::Test{

	protected:
		void SetUp() override {
			m = 5;
			n = 4;
			k = 3;
			h_a = new int[m*n];
			h_b = new int[n*k];
			h_c = new int[m*k];
			for(int i = 0; i < m*n; i++){
					h_a[i] = i;
			}
			for(int i = 0; i < n*k; i++){
					h_b[i] = 2 * i;
			}
			cudaMalloc(&d_a, m*n * sizeof(int));
			cudaMalloc(&d_b, n*k * sizeof(int));
			cudaMalloc(&d_c, m*k * sizeof(int));
			cudaMemcpy(d_a, h_a, m*n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, h_b, n*k * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

	int n, m, k;
	int * h_a;
	int * h_b;
	int * h_c;
	int * d_a, *d_b, *d_c;
};

TEST_F(L1Test, DotProduct){
	global_dot<<<1, n>>>(d_c, n, d_a, d_b);
	cudaDeviceSynchronize();
	// copy the memory back
	cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	EXPECT_EQ(*h_c, 656700);
}

TEST_F(L1Test, DotProductMultiBlock){
	global_dot<<<dim3(2,2,2), dim3(2,2,2)>>>(d_c, n, d_a, d_b);
	cudaDeviceSynchronize();
	// copy the memory back
	cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	EXPECT_EQ(*h_c, 656700);
}

TEST_F(L1Test, L2norm) {
	int expected_sum = 0;
	for (int i = 0; i < n; i++) {
		expected_sum += h_a[i] * h_a[i];
	}
	printf("%d\n", expected_sum);

	global_l2norm<<<1, n>>>(n, d_a);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, sizeof(*h_a), cudaMemcpyDeviceToHost);
	EXPECT_EQ(h_a[0], floor(sqrtf(expected_sum)));
}

TEST_F(L2Test, gemv){
	int res[] = {910,952,994,1036,1078};
	int res_transpose[] = {182,476,770,1064,1358};
	global_gemv<int, false><<<1, 64>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemv<int, true><<<1, 64>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

TEST_F(L2Test, gemvMultiBlock){
	int res[] = {910,952,994,1036,1078};
	int res_transpose[] = {182,476,770,1064,1358};
	global_gemv<int, false><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemv<int, true><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

TEST_F(L3Test, gemm){
	int res[] = {140,152,164,176,188,380,424,468,512,556,620,696,772,848,924};
	int res_transpose[] = {420,456,492,528,564,480,524,568,612,656,540,592,644,696,748};
	global_gemm<int, false><<<1, 64>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*k*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemm<int, true><<<1, 64>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*k*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

TEST_F(L3Test, gemmMultiBlock){
	int res[] = {140,152,164,176,188,380,424,468,512,556,620,696,772,848,924};
	int res_transpose[] = {420,456,492,528,564,480,524,568,612,656,540,592,644,696,748};
	global_gemm<int, false><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*k*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemm<int, true><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m*k*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

int main(){
        ::testing::InitGoogleTest();
        return RUN_ALL_TESTS();
}




