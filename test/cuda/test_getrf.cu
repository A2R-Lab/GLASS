// test_getrf.cu — dispatch getrf/getrs/gesv/laswp ops and print results to stdout
// Usage: ./test_getrf <op> <simple> <threads> <dims...> [args...] [files...]
//
// Ops (all launched <<<1, threads>>> so the Python side can sweep block sizes):
//   getrf       <threads> <n> <A>                          → LU line, piv line
//   getrf_check <threads> <n> <A>                          → LU line, piv line, fail line
//   getrs       <threads> <n> <nrhs> <transpose> <LU> <piv> <B> → X line
//   gesv        <threads> <n> <nrhs> <A> <B>               → X line, LU line
//   gesv_ct     <threads> <split> <A(4x4)> <B(4x3)>        → X line  (compile-time overloads;
//                 split=0: glass::gesv<float,4,3>, split=1: getrf<float,4> + getrs<float,4,3>)
//   laswp_vec   <threads> <n> <reverse> <piv> <x>          → x line
//   laswp_mat   <threads> <n> <reverse> <piv> <A(n×n)>     → A line
//
// piv files hold float32 values that are rounded to uint32_t on upload;
// piv outputs are printed as a space-separated integer line.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.cuh"
#include "../../glass.cuh"

// Read n float32 values from a .bin, round, upload to a device uint32_t array.
static uint32_t* read_device_uvec(const char* path, int n) {
    float* h = read_host_vec(path, n);
    uint32_t* hu = (uint32_t*)malloc(n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) hu[i] = (uint32_t)lroundf(h[i]);
    uint32_t* d; cudaMalloc(&d, n * sizeof(uint32_t));
    cudaMemcpy(d, hu, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    free(h); free(hu);
    return d;
}

static uint32_t* alloc_device_uvec(int n) {
    uint32_t* d; cudaMalloc(&d, n * sizeof(uint32_t));
    cudaMemset(d, 0, n * sizeof(uint32_t));
    return d;
}

// Print n device uint32_t values as a space-separated line.
static void print_device_uvec(const uint32_t* d, int n) {
    uint32_t* h = (uint32_t*)malloc(n * sizeof(uint32_t));
    cudaMemcpy(h, d, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%u", h[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
    free(h);
}

// ─── kernels ─────────────────────────────────────────────────────────────────

__global__ void k_getrf(int n, float* A, uint32_t* piv) {
    glass::block::getrf<float>(n, A, piv);
}
__global__ void k_getrf_check(int n, float* A, uint32_t* piv, int* fail) {
    glass::block::getrf<float, true>(n, A, piv, fail);
}
__global__ void k_getrs(int n, int nrhs, const float* LU, const uint32_t* piv, float* B) {
    glass::block::getrs<float>(n, nrhs, LU, piv, B);
}
__global__ void k_getrs_t(int n, int nrhs, const float* LU, const uint32_t* piv, float* B) {
    glass::block::getrs<float, /*TRANSPOSE=*/true>(n, nrhs, LU, piv, B);
}
__global__ void k_gesv(int n, int nrhs, float* A, uint32_t* piv, float* B) {
    glass::block::gesv<float>(n, nrhs, A, piv, B);
}
// Compile-time overloads (N=4, NRHS=3): fused gesv, and split getrf + getrs.
__global__ void k_gesv_ct_4_3(float* A, uint32_t* piv, float* B) {
    glass::block::gesv<float, 4, 3>(A, piv, B);
}
__global__ void k_getrf_getrs_ct_4_3(float* A, uint32_t* piv, float* B) {
    glass::block::getrf<float, 4>(A, piv);
    glass::block::getrs<float, 4, 3>(A, piv, B);
}
__global__ void k_laswp_vec(int n, const uint32_t* piv, float* x) {
    glass::block::laswp<float>(piv, 0, n, x);
}
__global__ void k_laswp_vec_rev(int n, const uint32_t* piv, float* x) {
    glass::block::laswp<float, /*REVERSE=*/true>(piv, 0, n, x);
}
__global__ void k_laswp_mat(int n, const uint32_t* piv, float* A) {
    glass::block::laswp<float>(n, A, piv, 0, n);
}
__global__ void k_laswp_mat_rev(int n, const uint32_t* piv, float* A) {
    glass::block::laswp<float, /*REVERSE=*/true>(n, A, piv, 0, n);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <op> <simple> <threads> <dims...> [args...] [files...]\n", argv[0]);
        return 1;
    }
    const char* op = argv[1];
    int th = atoi(argv[3]);

    if (strcmp(op, "getrf") == 0) {
        // getrf <simple> <threads> <n> <A>
        int n = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], n * n);
        uint32_t* dpiv = alloc_device_uvec(n);
        k_getrf<<<1, th>>>(n, dA, dpiv);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);
        print_device_uvec(dpiv, n);

    } else if (strcmp(op, "getrf_check") == 0) {
        // getrf_check <simple> <threads> <n> <A>
        int n = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], n * n);
        uint32_t* dpiv = alloc_device_uvec(n);
        int* dfail; cudaMalloc(&dfail, sizeof(int));
        cudaMemset(dfail, 0xFF, sizeof(int));   // poison: getrf must write 0/1 itself
        k_getrf_check<<<1, th>>>(n, dA, dpiv, dfail);
        cudaDeviceSynchronize();
        int fail; cudaMemcpy(&fail, dfail, sizeof(int), cudaMemcpyDeviceToHost);
        print_device_vec(dA, n * n);
        print_device_uvec(dpiv, n);
        printf("%d\n", fail);
        cudaFree(dfail);

    } else if (strcmp(op, "getrs") == 0) {
        // getrs <simple> <threads> <n> <nrhs> <transpose> <LU> <piv> <B>
        int n = atoi(argv[4]);
        int nrhs = atoi(argv[5]);
        int transpose = atoi(argv[6]);
        float* dLU = read_device_vec(argv[7], n * n);
        uint32_t* dpiv = read_device_uvec(argv[8], n);
        float* dB = read_device_vec(argv[9], n * nrhs);
        if (transpose) k_getrs_t<<<1, th>>>(n, nrhs, dLU, dpiv, dB);
        else           k_getrs<<<1, th>>>(n, nrhs, dLU, dpiv, dB);
        cudaDeviceSynchronize();
        print_device_vec(dB, n * nrhs);

    } else if (strcmp(op, "gesv") == 0) {
        // gesv <simple> <threads> <n> <nrhs> <A> <B>
        int n = atoi(argv[4]);
        int nrhs = atoi(argv[5]);
        float* dA = read_device_vec(argv[6], n * n);
        float* dB = read_device_vec(argv[7], n * nrhs);
        uint32_t* dpiv = alloc_device_uvec(n);
        k_gesv<<<1, th>>>(n, nrhs, dA, dpiv, dB);
        cudaDeviceSynchronize();
        print_device_vec(dB, n * nrhs);
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "gesv_ct") == 0) {
        // gesv_ct <simple> <threads> <split> <A(4x4)> <B(4x3)>
        int split = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * 4);
        float* dB = read_device_vec(argv[6], 4 * 3);
        uint32_t* dpiv = alloc_device_uvec(4);
        if (split) k_getrf_getrs_ct_4_3<<<1, th>>>(dA, dpiv, dB);
        else       k_gesv_ct_4_3<<<1, th>>>(dA, dpiv, dB);
        cudaDeviceSynchronize();
        print_device_vec(dB, 4 * 3);

    } else if (strcmp(op, "laswp_vec") == 0) {
        // laswp_vec <simple> <threads> <n> <reverse> <piv> <x>
        int n = atoi(argv[4]);
        int reverse = atoi(argv[5]);
        uint32_t* dpiv = read_device_uvec(argv[6], n);
        float* dx = read_device_vec(argv[7], n);
        if (reverse) k_laswp_vec_rev<<<1, th>>>(n, dpiv, dx);
        else         k_laswp_vec<<<1, th>>>(n, dpiv, dx);
        cudaDeviceSynchronize();
        print_device_vec(dx, n);

    } else if (strcmp(op, "laswp_mat") == 0) {
        // laswp_mat <simple> <threads> <n> <reverse> <piv> <A>
        int n = atoi(argv[4]);
        int reverse = atoi(argv[5]);
        uint32_t* dpiv = read_device_uvec(argv[6], n);
        float* dA = read_device_vec(argv[7], n * n);
        if (reverse) k_laswp_mat_rev<<<1, th>>>(n, dpiv, dA);
        else         k_laswp_mat<<<1, th>>>(n, dpiv, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
