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
			h_d = new int[m*m];
			h_e = new int[m*m];
			for(int i = 0; i < m*n; i++){
					h_a[i] = i;
			}
			for(int i = 0; i < n*k; i++){
					h_b[i] = 2 * i;
			}
			for(int i = 0; i < m*m; i++){
					h_d[i] = 2 * i + 1;
			}
			cudaMalloc(&d_a, m*n * sizeof(int));
			cudaMalloc(&d_b, n*k * sizeof(int));
			cudaMalloc(&d_c, m*k * sizeof(int));
			cudaMalloc(&d_d, m*m * sizeof(int));
			cudaMalloc(&d_e, m*m * sizeof(int));
			cudaMemcpy(d_a, h_a, m*n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, h_b, n*k * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_d, h_d, m*m * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			cudaFree(d_d);
			cudaFree(d_e);
			delete h_a;
			delete h_b;
			delete h_c;
			delete h_d;
			delete h_e;
		}

	int n, m, k;
	int * h_a, *h_b, *h_c, *h_d, *h_e;
	int * d_a, *d_b, *d_c, *d_d, *d_e;
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

TEST_F(L1Test, reduce) {
	int expected_sum = 0;
	for (int i = 0; i < n; i++) { expected_sum += h_a[i]; }
	global_reduce<<<1, n>>>(n, d_a);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, sizeof(*h_a), cudaMemcpyDeviceToHost);
	EXPECT_EQ(h_a[0], expected_sum);
}

TEST_F(L1Test, l2norm) {
	int expected_sum = 0;
	for (int i = 0; i < n; i++) { expected_sum += h_a[i] * h_a[i]; }
	global_l2norm<<<1, n>>>(n, d_a);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, sizeof(*h_a), cudaMemcpyDeviceToHost);
	EXPECT_EQ(h_a[0], floor(sqrtf(expected_sum)));
}

TEST_F(L1Test, scal) {
	int expected[n];
	for (int i = 0; i < n; i++) { expected[i] = h_a[i] * 2; }
	global_scal<<<1, n>>>(n, 2, d_a);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, sizeof(*h_a) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(h_a[i], expected[i]);
	}
}

TEST_F(L1Test, setConst) {
	global_set_const<<<1, n>>>(n, n, d_a);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, sizeof(*h_a) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(h_a[i], n);
	}
}

TEST_F(L1Test, swap) {
	int expected_a[n], expected_b[n];
	for (int i = 0; i < n; i++) {
		expected_a[i] = h_b[i];
		expected_b[i] = h_a[i];
	}
	global_swap<<<1,n>>>(n, 1, d_a, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, sizeof(*h_a) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, sizeof(*h_b) * n, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(h_a[i], expected_a[i]);
		EXPECT_EQ(h_b[i], expected_b[i]);
	}
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

TEST_F(L3Test, invSingle){
	global_invertMatrix<<<1, m*m>>>(m, d_d, d_e);
	cudaDeviceSynchronize();
	global_invertMatrix<<<1, m*m>>>(m, d_d, d_e);
	cudaDeviceSynchronize();
	cudaMemcpy(h_d, d_d, m*m*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<m*m; i++){
		EXPECT_EQ(h_d[i], 2 * i + 1);
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




