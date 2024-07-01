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
			cudaMalloc(&d_c, n * sizeof(int));
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
	int * h_a, *h_b, *h_c;
	int * d_a, *d_b, *d_c;
};

class L3InvTest : public ::testing::Test{

	protected:
		void SetUp() override {
			m = 5;
			h_a = new double[2*m*m] {
				10, 2,  4,  5,  3,
				11, 6,  12, 7,  13,
				8,  9,  14, 15, 16,
				17, 18, 19, 20, 21,
				22, 23, 24, 25, 26,
			};

			cudaMalloc(&d_a, 2*m*m * sizeof(*d_a));
			cudaMalloc(&d_b, 2*m*m * sizeof(*d_b));
			cudaMalloc(&d_temp, (2*m + 1) * sizeof(*d_temp));
			cudaMemcpy(d_a, h_a, 2*m*m * sizeof(*d_a), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		void TearDown() override {
			cudaFree(d_a);
			cudaFree(d_temp);
			delete h_a;
		}

	int m;
	double *h_a;
	double *d_a, *d_b;
	double *d_temp;
};

class TriangularTest : public ::testing::Test{

protected:
    void SetUp() override {
        n = 3;
        m = 2;
        h_A = new double[n*n] {
                a, b, c,
                0, d, e,
                0, 0, f
        };
        h_B = new double[m*n] {
                b11, b21,
                b12, b22,
                b13, b23
        };
        h_c = new double [n] {c1, c2, c3};

        cudaMalloc(&d_A, n*n * sizeof(*d_A));
        cudaMalloc(&d_B, m*n * sizeof(*d_B));
        cudaMalloc(&d_c, n * sizeof(*d_c));
        cudaMemcpy(d_A, h_A, n*n * sizeof(*d_A), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, m*n * sizeof(*d_B), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c, n * sizeof(*d_c), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    void TearDown() override {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_c);
        delete h_A;
        delete h_B;
        delete h_c;
    }

    int n, m;
    double *h_A, *h_B, *h_c;
    double *d_A, *d_B, *d_c;
    // values can be arbitrary
    double a = 3, b = 2.7, c = 0.8, d = 10, e = 0.4, f = 3.2;
    double b11 = -2, b12 = 1.8, b13 = -3, b21 = -1.9, b22 = -0.8, b23 = 1.2;
    double c1 = 2.8, c2 = -0.9, c3 = -2.1;
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

TEST_F(L1Test, axpy) {
	int res[n];
	int alpha = n;

	for (int i=0; i<n; i++) {
		res[i]= h_a[i]*alpha+h_b[i];
	}
	global_axpy<<<1,n>>>(n, alpha, d_a, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_b, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(h_b[i], res[i]);
	}
}

TEST_F(L1Test, clip) {
	global_clip<<<1,n>>>(n, d_a, d_b, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, copy) {
	global_copy<<<1,n>>>(n, d_a, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_b, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, copyMultiBlock) {
	global_copy<<<dim3(2,2,2), dim3(2,2,2)>>>(n, d_a, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_b, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ(h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, scaledCopy) {
	int alpha = 4;

	global_copy<<<1,n>>>(n, alpha, d_a, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_b, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ(alpha*h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, loadIdentity) {
	int dim = (int)sqrt(n);
	global_loadIdentity<<<1,n>>>(dim, d_a);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ((i%dim == i/dim), h_a[i]);
	}
}

TEST_F(L1Test, loadIdentity2) {
	int dim = (int)sqrt(n);
	global_loadIdentity<<<1,n>>>(dim, d_a, dim, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ((i%dim == i/dim), h_a[i]);
		EXPECT_EQ((i%dim == i/dim), h_b[i]);
	}
}

TEST_F(L1Test, addI) {
	int dim = (int)sqrt(n);
	int alpha = 4;
	int res[n];

	for (int i=0; i<n; i++) {
		if (i%dim == i/dim)
			res[i] = h_a[i] + alpha;
		else
			res[i] = h_a[i];
	}
	global_addI<<<1,n>>>(dim, d_a, alpha);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(res[i], h_a[i]);
	}
}

TEST_F(L1Test, infnorm) {
	global_infnorm<<<1,n>>>(n, d_b);
	cudaDeviceSynchronize();
	cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
	EXPECT_EQ(198, h_b[0]);
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

TEST_F(L3Test, ldl) {
    double h_A[] = {18, 5, 1.5, 5, 3.5, 1.3, 1.5, 1.3, 8.8};
    double h_D[] = {0,0,0};
    
    double res_A[] = {1, 0.27777779, 0.083333336, 5, 1, 0.41842106,1.5, 1.3, 1};
    double res_D[] = {18, 2.1111112, 8.3053951};

    double *d_A;
    double *d_D;

    cudaMalloc(&d_A, 9 * sizeof(double));
    cudaMalloc(&d_D, 3 * sizeof(double));
    cudaMemcpy(d_A, h_A, 9 * sizeof(double), cudaMemcpyHostToDevice);
    global_ldlDecomp_InPlace<<<1,5>>>(3, d_A, d_D);
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_A, 9*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, 3*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(h_A[i], res_A[i]);
    }

    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(h_D[i], res_D[i]);
    }

    cudaFree(d_A);
    cudaFree(d_D);
}

TEST_F(L3Test, chol) {
	double h_d[] = {10, 5, 2, 5, 3, 2, 2, 2, 3};
	double res[] = {pow(10,0.5), 5/pow(10,0.5), 2/pow(10,0.5), 5, 
					1/pow(2,0.5), pow(2,0.5), 2, 2, pow(3,0.5)/pow(5,0.5)};
	double *d_d;

	cudaMalloc(&d_d, 9 * sizeof(double));
	cudaMemcpy(d_d, h_d, 9 * sizeof(double), cudaMemcpyHostToDevice);
	global_cholDecomp_InPlace_c<<<1,9>>>(3, d_d);
	cudaDeviceSynchronize();
	cudaMemcpy(h_d, d_d, 9*sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 9; i++) {
		EXPECT_FLOAT_EQ(h_d[i], res[i]);
	}
	cudaFree(d_d);
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

TEST_F(L3InvTest, invSingle) {

	double res[] = {
		1.0/9, 1.0/33, -1.0/9, 26.0/45, -211.0/495,
		-1.0/9, -1.0/33, -5.0/36, -7.0/90, 349.0/1980,
		-1.0/9, 2.0/33, 13.0/36, -331.0/90, 5407.0/1980,
		1.0/9, -5.0/33, 5.0/36, 7.0/90, -169.0/1980,
		0.0, 1.0/11, -1.0/4, 29.0/10, -483.0/220
	};

	// load identity:	[d_a 	| identity]
	global_loadIdentity<<<1, 64>>>(m, d_a + m*m);
	cudaDeviceSynchronize();

	// invert d_a:		[ident w error? | d_a inv]
	global_invertMatrix<<<1, 64>>>(m, d_a, d_temp);
	cudaDeviceSynchronize();

	cudaMemcpy(h_a, d_a + m*m, m*m * sizeof(*d_a), cudaMemcpyDeviceToHost);

	for (int i = 0; i < m*m; i++) {
		EXPECT_LT(abs(h_a[i] - res[i]), 1e-13);
	}
}

TEST_F(L3InvTest, invSingleAndMultiply) {
	// load identity:	[d_a 	| identity]
	global_loadIdentity<<<1, 64>>>(m, d_a + m*m);
	cudaDeviceSynchronize();

	// invert d_a:		[ident w error? | d_a inv]
	global_invertMatrix<<<1, 64>>>(m, d_a, d_temp);
	cudaDeviceSynchronize();

	// load into d_a again:	[d_a	| d_a inv]
	cudaMemcpy(d_a, h_a, m*m * sizeof(*d_a), cudaMemcpyHostToDevice);

	// multiply d_a * d_a inv, store result in d_b
	global_gemm<double, false><<<1, 64>>>(m, m, m, 1.0, d_a, d_a + m*m, d_b),
	cudaDeviceSynchronize();

	cudaMemcpy(h_a, d_b, m*m * sizeof(*d_b), cudaMemcpyDeviceToHost);

	// result should be identity
	for (int i = 0; i < m*m; i++) {
		EXPECT_LT(abs(h_a[i] - (i%m == i/m)), 1e-13);
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

TEST_F(TriangularTest, trmv){
    // test d = A*c, e = A'*c
    double res_d[] = {a*c1, b*c1+d*c2, c*c1 +e*c2+f*c3};
    double res_e[] = {a*c1+b*c2+c*c3, d*c2+e*c3, f*c3};
    double h_d[] = {0,0,0};
    double h_e[] = {0,0,0};
    double *d_d, *d_e;

    // test d = A*c
    cudaMalloc(&d_d, n*sizeof(double));
//    global_gemv<double, false><<<1,64>>>(n,n,static_cast<double>(1), d_A, d_c, d_d);
    global_trmv<double, false><<<1,64>>>(n,static_cast<double>(1), d_A, d_c, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, n * sizeof(*d_d), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(h_d[i], res_d[i]);
    }

    // test e = A'*c
    cudaMalloc(&d_e, n*sizeof(double));
//    global_gemv<double, true><<<1,64>>>(n,n,static_cast<double>(1), d_A, d_c, d_e);
    global_trmv<double, true><<<1,64>>>(n,static_cast<double>(1), d_A, d_c, d_e);
    cudaDeviceSynchronize();
    cudaMemcpy(h_e, d_e, n * sizeof(*d_e), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(h_e[i], res_e[i]);
    }
}

TEST_F(TriangularTest, trsv){
    // there is actually no trsv function, but trsm applied to a single vector IS trsv
    // test d = inv(A)*c, e = inv(A)'*c
    double res_d[] = {c1/a, -b*c1/a/d + c2/d, (b*e-c*d)*c1/a/d/f + -e*c2/d/f + c3/f};
    double res_e[] = {c1/a + -b*c2/a/d + (b*e-c*d)*c3/a/d/f, c2/d - e*c3/d/f, c3/f};
    double h_d[] = {0,0,0};
    double h_e[] = {0,0,0};
    double *d_d, *d_e;

    // test d = inv(A)*c
    cudaMalloc(&d_d, n*sizeof(double));
    cudaMemcpy(d_d, h_c, n * sizeof(*d_c), cudaMemcpyHostToDevice);
    global_trsm_InPlace<double, false><<<1,64>>>(n,1, d_A, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, n * sizeof(*d_d), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(h_d[i], res_d[i]);
    }

    // test e = inv(A)'*c
    cudaMalloc(&d_e, n*sizeof(double));
    cudaMemcpy(d_e, h_c, n * sizeof(*d_c), cudaMemcpyHostToDevice);
    global_trsm_InPlace<double, true><<<1,64>>>(n,1, d_A, d_e);
    cudaDeviceSynchronize();
    cudaMemcpy(h_e, d_e, n * sizeof(*d_e), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(h_e[i], res_e[i]);
    }
}

TEST_F(TriangularTest, trsm){
    // test C=inv(A)*B, D=inv(A), E = inv(A)'*B, where A is lower triangular

    double res_inv[] = {1/a, -b/a/d, (b*e-c*d)/a/d/f, 0, 1/d, -e/d/f, 0, 0, 1/f};
    double res_C[] = {0,0,0,0,0,0};
    double res_D[] = {0,0,0,0,0,0,0,0,0};
    double *d_inv, *d_C, *d_D;

    cudaMalloc(&d_inv, n*n*sizeof(double));
    cudaMalloc(&d_C, m*n*sizeof(double));
    cudaMemcpy(d_inv, res_inv, n*n * sizeof(double), cudaMemcpyHostToDevice);

    global_gemm<double, false><<<1, 64>>>(n, n, m, static_cast<double>(1), d_inv, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(res_C, d_C, n*m*sizeof(double), cudaMemcpyDeviceToHost);

    // test trsm for C=inv(A)*B
    global_trsm_InPlace<double, false><<<1, 64>>>(n, m, d_A, d_B);
    cudaMemcpy(h_B, d_B, m*n * sizeof(*d_B), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n*m; i++) {
        EXPECT_FLOAT_EQ(h_B[i], res_C[i]);
    }

    // test trsm for D=inv(A)
    cudaMalloc(&d_D, n*n*sizeof(double));
    global_loadIdentity<<<1, 64>>>(n, d_D);
    cudaDeviceSynchronize();

    global_trsm_InPlace<double, false><<<1, 64>>>(n, n, d_A, d_D);
    cudaDeviceSynchronize();

    cudaMemcpy(res_D, d_D, n*n * sizeof(*d_B), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n*n; i++) {
        EXPECT_FLOAT_EQ(res_D[i], res_inv[i]);
    }

    // test E = inv(A)'*B
    // not tested because gemm has no corresponding support
    // but should be working because trsv tested transpose above

    cudaFree(d_inv);
    cudaFree(d_C);
    cudaFree(d_D);
}

int main(){
        ::testing::InitGoogleTest();
        return RUN_ALL_TESTS();
}