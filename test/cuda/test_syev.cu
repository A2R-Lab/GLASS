// test_syev.cu — glass::syev (cyclic-Jacobi symmetric eigendecomposition) and
// glass::eig_clamp (eigenvalue clamp + reconstruct) drivers.
//
// Ops (argv[2] is the usual run_op version slot, ignored):
//   syev      simple <n> <threads> <A.bin>        → W ascending (line 1),
//                                                    V col-major (line 2)
//   syev_ct   simple <n> <threads> <A.bin>        → same, compile-time-N overload
//                                                    (n restricted to SYEV_SIZES)
//   eig_clamp simple <n> <threads> <eps> <A.bin>  → clamped A, n*n col-major (line 1)
//   eigh        simple <n> <threads> <dtype> <A.bin>        → W unsorted (line 1),
//                                                             V col-major (line 2)
//   psd_project simple <n> <threads> <dtype> <eps> <A.bin>  → projected A (line 1)
//     (dtype = f32|f64; compile-time-N ops, n restricted to EIGH_SIZES. The f32
//      .bin input is widened to T host-side; output prints at round-trip
//      precision so bit-equal values render identically — the thread-invariance
//      and run-twice gates compare THIS text.)
//
// A.bin : n*n float32 (column-major, symmetric). Scratch is dynamic shared
// memory sized by glass::syev_scratch_bytes / glass::eig_clamp_scratch_bytes /
// glass::eigh_scratch_bytes / glass::psd_project_scratch_bytes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass.cuh"

__global__ void k_syev(uint32_t n, const float* A, float* W, float* V) {
    extern __shared__ float s[];
    glass::syev<float>(n, A, W, V, s);
}

template <uint32_t N>
__global__ void k_syev_ct(const float* A, float* W, float* V) {
    extern __shared__ float s[];
    glass::syev<float, N>(A, W, V, s);
}

__global__ void k_eig_clamp(uint32_t n, float* A, float eps) {
    extern __shared__ float s[];
    glass::eig_clamp<float>(n, A, eps, s);
}

#define SYEV_SIZES(F) F(1) F(2) F(3) F(4) F(7) F(12) F(16) F(32)

// ─── eigh / psd_project (fixed-sweep Jacobi; dtype-templated) ────────────────
// Consumer sizes 12/14/18/21 (GATO stage blocks) + small even/odd + an odd
// >32 to exercise the pad-index schedule path near the top of the range.
#define EIGH_SIZES(F) F(4) F(7) F(12) F(14) F(18) F(21) F(33)

template <typename T, uint32_t N>
__global__ void k_eigh_ct(const T* A, T* W, T* V) {
    extern __shared__ __align__(16) unsigned char smraw[];
    glass::eigh<T, N>(A, W, V, reinterpret_cast<T*>(smraw));
}

template <typename T, uint32_t N>
__global__ void k_psd_project_ct(T* A, T eps) {
    extern __shared__ __align__(16) unsigned char smraw[];
    glass::psd_project<T, N>(A, eps, reinterpret_cast<T*>(smraw));
}

// Read an n-element float32 .bin and widen to T on the host; upload.
template <typename T>
static T* read_device_vec_w(const char* path, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float* hf = (float*)malloc((size_t)n * sizeof(float));
    if (fread(hf, sizeof(float), (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    T* ht = (T*)malloc((size_t)n * sizeof(T));
    for (int i = 0; i < n; i++) ht[i] = (T)hf[i];
    T* d; cudaMalloc(&d, (size_t)n * sizeof(T));
    cudaMemcpy(d, ht, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
    free(hf); free(ht);
    return d;
}

// Print a device vector at round-trip precision (bit-equal T => identical text).
template <typename T>
static void print_device_vec_rt(const T* d, int n) {
    T* h = (T*)malloc((size_t)n * sizeof(T));
    cudaMemcpy(h, d, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if constexpr (sizeof(T) == 8) printf("%.17g", (double)h[i]);
        else                          printf("%.9g",  (double)h[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
    free(h);
}

template <typename T>
static int run_eigh(int n, int threads, const char* abin) {
    T* dA = read_device_vec_w<T>(abin, n * n);
    T* dW; cudaMalloc(&dW, (size_t)n * sizeof(T));
    T* dV; cudaMalloc(&dV, (size_t)n * (size_t)n * sizeof(T));
    bool ok = false;
    #define DN(N_) if (!ok && n == N_) { \
        k_eigh_ct<T, N_><<<1, threads, (int)glass::eigh_scratch_bytes<T, N_>()>>>(dA, dW, dV); ok = true; }
    EIGH_SIZES(DN)
    #undef DN
    if (!ok) { fprintf(stderr, "eigh: unsupported n=%d\n", n); return 2; }
    cudaDeviceSynchronize();
    print_device_vec_rt<T>(dW, n);
    print_device_vec_rt<T>(dV, n * n);
    return 0;
}

template <typename T>
static int run_psd_project(int n, int threads, T eps, const char* abin) {
    T* dA = read_device_vec_w<T>(abin, n * n);
    bool ok = false;
    #define DN(N_) if (!ok && n == N_) { \
        k_psd_project_ct<T, N_><<<1, threads, (int)glass::psd_project_scratch_bytes<T, N_>()>>>(dA, eps); ok = true; }
    EIGH_SIZES(DN)
    #undef DN
    if (!ok) { fprintf(stderr, "psd_project: unsupported n=%d\n", n); return 2; }
    cudaDeviceSynchronize();
    print_device_vec_rt<T>(dA, n * n);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "usage: %s syev|syev_ct <version> <n> <threads> <A.bin>\n"
            "       %s eig_clamp    <version> <n> <threads> <eps> <A.bin>\n",
            argv[0], argv[0]);
        return 1;
    }
    const char* op = argv[1];               // argv[2] = version ("simple"), unused
    int n       = atoi(argv[3]);
    int threads = atoi(argv[4]);

    if (strcmp(op, "syev") == 0 || strcmp(op, "syev_ct") == 0) {
        float* dA = read_device_vec(argv[5], n * n);
        float* dW = alloc_device_vec(n);
        float* dV = alloc_device_vec(n * n);
        int sm = (int)glass::syev_scratch_bytes<float>((uint32_t)n);
        if (strcmp(op, "syev") == 0) {
            k_syev<<<1, threads, sm>>>((uint32_t)n, dA, dW, dV);
        } else {
            bool ok = false;
            #define DN(N_) if (!ok && n == N_) { k_syev_ct<N_><<<1, threads, sm>>>(dA, dW, dV); ok = true; }
            SYEV_SIZES(DN)
            #undef DN
            if (!ok) { fprintf(stderr, "syev_ct: unsupported n=%d\n", n); return 2; }
        }
        cudaDeviceSynchronize();
        print_device_vec(dW, n);
        print_device_vec(dV, n * n);
    } else if (strcmp(op, "eig_clamp") == 0) {
        if (argc < 7) { fprintf(stderr, "eig_clamp needs <eps> <A.bin>\n"); return 1; }
        float eps = (float)atof(argv[5]);
        float* dA = read_device_vec(argv[6], n * n);
        int sm = (int)glass::eig_clamp_scratch_bytes<float>((uint32_t)n);
        k_eig_clamp<<<1, threads, sm>>>((uint32_t)n, dA, eps);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);
    } else if (strcmp(op, "eigh") == 0) {
        if (argc < 7) { fprintf(stderr, "eigh needs <dtype> <A.bin>\n"); return 1; }
        const char* dt = argv[5];
        int rc = strcmp(dt, "f64") == 0 ? run_eigh<double>(n, threads, argv[6])
                                        : run_eigh<float >(n, threads, argv[6]);
        if (rc) return rc;
    } else if (strcmp(op, "psd_project") == 0) {
        if (argc < 8) { fprintf(stderr, "psd_project needs <dtype> <eps> <A.bin>\n"); return 1; }
        const char* dt = argv[5];
        double eps = atof(argv[6]);
        int rc = strcmp(dt, "f64") == 0 ? run_psd_project<double>(n, threads, (double)eps, argv[7])
                                        : run_psd_project<float >(n, threads, (float)eps,  argv[7]);
        if (rc) return rc;
    } else {
        fprintf(stderr, "unknown op %s\n", op);
        return 1;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 3; }
    return 0;
}
