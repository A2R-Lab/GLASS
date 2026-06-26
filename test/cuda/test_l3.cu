// test_l3.cu — dispatch L3 GLASS operations and print float32 results to stdout
// Usage: ./test_l3 <op> <cg|simple> <m> [n] [k] [args...] [files...]
//
// Versions:
//   cg     — glass::cgrps:: (cooperative groups)
//   simple — glass::        (threadIdx, default)
//
// GEMM uses the standard BLAS convention: C is M×N, contraction K;
// op(A) is M×K (TRANSPOSE_A ⇒ A is K×M), op(B) is K×N (TRANSPOSE_B ⇒ B is N×K),
// ROW_MAJOR_C selects row-major output. The gemm_rt / gemm_ct ops below take the
// three layout flags as 0/1 ints so the Python side can sweep the full matrix.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

static int THREADS = 256;

// Read n float32 values from a .bin, round to int, upload to device int array.
static int* read_device_ivec(const char* path, int n) {
    float* h = read_host_vec(path, n);
    int* hi = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) hi[i] = (int)lroundf(h[i]);
    int* d; cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, hi, n * sizeof(int), cudaMemcpyHostToDevice);
    free(h); free(hi);
    return d;
}

// ─── gemm_batched_indexed kernel (DIM=4) ─────────────────────────────────────
__global__ void k_indexed_bgemm_4(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                   float* A, float* B, float* C) {
    glass::gemm_batched_indexed<float, 4>(pairs, a_idx, b_idx, c_idx, A, B, C);
}

// ─── gemm_batched_indexed: TRANSPOSE_A / TRANSPOSE_B / ATOMIC_C variants ──────
__global__ void k_indexed_bgemm_4_ta(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                      float* A, float* B, float* C) {
    glass::gemm_batched_indexed<float, 4, /*TA*/true, /*TB*/false, /*ATOMIC*/false>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}
__global__ void k_indexed_bgemm_4_tb(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                      float* A, float* B, float* C) {
    glass::gemm_batched_indexed<float, 4, /*TA*/false, /*TB*/true, /*ATOMIC*/false>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}
__global__ void k_indexed_bgemm_4_atomic(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                         float* A, float* B, float* C) {
    glass::gemm_batched_indexed<float, 4, /*TA*/false, /*TB*/false, /*ATOMIC*/true>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}
__global__ void k_indexed_bgemm_4_ta_atomic(uint32_t pairs, int* a_idx, int* b_idx, int* c_idx,
                                            float* A, float* B, float* C) {
    glass::gemm_batched_indexed<float, 4, /*TA*/true, /*TB*/false, /*ATOMIC*/true>(
        pairs, a_idx, b_idx, c_idx, A, B, C);
}

// ─── GEMM kernels (standard convention) ──────────────────────────────────────
// `nb` (no-beta): when 1, call the overload that overwrites C and never reads it
// (so a NaN-poisoned C must survive); when 0, the beta overload.
template <bool TA, bool TB, bool RMC>
__global__ void k_gemm_rt(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C, int nb) {
    if (nb) glass::gemm<float, TA, TB, RMC>(m, n, k, alpha, A, B, C);
    else    glass::gemm<float, TA, TB, RMC>(m, n, k, alpha, A, B, beta, C);
}
template <bool TA, bool TB, bool RMC>
__global__ void k_gemm_rt_cg(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C, int nb) {
    if (nb) glass::cgrps::gemm<float, TA, TB, RMC>(m, n, k, alpha, A, B, C);
    else    glass::cgrps::gemm<float, TA, TB, RMC>(m, n, k, alpha, A, B, beta, C);
}
// Compile-time-size (magic-multiply path).
template <int M, int N, int K, bool TA, bool TB, bool RMC>
__global__ void k_gemm_ct(float alpha, float* A, float* B, float beta, float* C, int nb) {
    if (nb) glass::gemm<float, M, N, K, TA, TB, RMC>(alpha, A, B, C);
    else    glass::gemm<float, M, N, K, TA, TB, RMC>(alpha, A, B, beta, C);
}
// Single-warp compile-time (run <<<1,32>>>).
template <int M, int N, int K, bool TA, bool TB, bool RMC>
__global__ void k_gemm_warp(float alpha, float* A, float* B, float beta, float* C, int nb) {
    if (nb) glass::warp::gemm<float, M, N, K, TA, TB, RMC>(alpha, A, B, C);
    else    glass::warp::gemm<float, M, N, K, TA, TB, RMC>(alpha, A, B, beta, C);
}

// Runtime flag dispatch (8 combos) for a launcher LAUNCH(TA,TB,RMC).
#define GEMM_FLAG_DISPATCH(ta, tb, rmc, LAUNCH)         \
    do {                                                \
        int _f = ((ta) << 2) | ((tb) << 1) | (rmc);     \
        switch (_f) {                                   \
            case 0: LAUNCH(false,false,false); break;   \
            case 1: LAUNCH(false,false,true ); break;   \
            case 2: LAUNCH(false,true ,false); break;   \
            case 3: LAUNCH(false,true ,true ); break;   \
            case 4: LAUNCH(true ,false,false); break;   \
            case 5: LAUNCH(true ,false,true ); break;   \
            case 6: LAUNCH(true ,true ,false); break;   \
            case 7: LAUNCH(true ,true ,true ); break;   \
        }                                               \
    } while (0)

// Compile-time shape table for gemm_ct / gemm_warp, selected by shape_id
// (deliberately includes non-square + all-distinct dims, plus M=1/N=1 edges).
static void ct_shape_dims(int id, int* M, int* N, int* K) {
    switch (id) {
        case 0: *M=1; *N=1; *K=1; break;
        case 1: *M=2; *N=3; *K=4; break;
        case 2: *M=4; *N=2; *K=3; break;
        case 3: *M=5; *N=7; *K=3; break;
        case 4: *M=8; *N=1; *K=5; break;
        case 5: *M=9; *N=4; *K=7; break;
        case 6: *M=7; *N=5; *K=6; break;
        default: *M=16; *N=16; *K=16; break;
    }
}

// Compile-time-size launcher: 8-way flag switch for a fixed (M,N,K).
template <int M, int N, int K>
static void launch_gemm_ct(bool warp, int th, int ta, int tb, int rmc, int nb,
                           float alpha, float* dA, float* dB, float beta, float* dC) {
    int f = (ta << 2) | (tb << 1) | rmc;
    #define _CT_DOIT(TA,TB,RMC)                                                          \
        do { if (warp) k_gemm_warp<M,N,K,TA,TB,RMC><<<1,32>>>(alpha,dA,dB,beta,dC,nb);   \
             else      k_gemm_ct  <M,N,K,TA,TB,RMC><<<1,th>>>(alpha,dA,dB,beta,dC,nb); } while (0)
    switch (f) {
        case 0: _CT_DOIT(false,false,false); break;
        case 1: _CT_DOIT(false,false,true ); break;
        case 2: _CT_DOIT(false,true ,false); break;
        case 3: _CT_DOIT(false,true ,true ); break;
        case 4: _CT_DOIT(true ,false,false); break;
        case 5: _CT_DOIT(true ,false,true ); break;
        case 6: _CT_DOIT(true ,true ,false); break;
        case 7: _CT_DOIT(true ,true ,true ); break;
    }
    #undef _CT_DOIT
}

// ─── inv kernels ──────────────────────────────────────────────────────────────
__global__ void k_inv_cg(int n, float* A, float* scratch) {
    glass::cgrps::invertMatrix(n, A, scratch);
}
__global__ void k_inv_simple(int n, float* A, float* scratch) {
    glass::invertMatrix(n, A, scratch);
}
__global__ void k_inv_pivot_simple(int n, float* A, float* scratch) {
    glass::invertMatrix_pivoted(n, A, scratch);
}
__global__ void k_inv2_simple(int dimA, int dimB, int maxd, float* A, float* B, float* scratch) {
    glass::invertMatrix(dimA, dimB, maxd, A, B, scratch);
}
__global__ void k_inv3_simple(int dimA, int dimB, int dimC, int maxd, float* A, float* B, float* C, float* scratch) {
    glass::invertMatrix(dimA, dimB, dimC, maxd, A, B, C, scratch);
}

// ─── chol kernels ─────────────────────────────────────────────────────────────
__global__ void k_chol_cg(int n, float* A) {
    glass::cgrps::cholDecomp_InPlace(n, A);
}
__global__ void k_chol_simple(int n, float* A) {
    glass::cholDecomp_InPlace(n, A);
}

// ─── trsm kernels ─────────────────────────────────────────────────────────────
__global__ void k_trsm_cg(int n, float* L, float* b) {
    glass::cgrps::trsm(n, L, b);
}
__global__ void k_trsm_simple(int n, float* L, float* b) {
    glass::trsm(n, L, b);
}

// ─── warp SPD solve (N=7) ─────────────────────────────────────────────────────
__global__ void k_posv_warp_7(float* A, float* b) {
    glass::warp::cholDecomp_InPlace<float, 7>(A);
    glass::warp::trsm<float, 7>(A, b);
    glass::warp::trsm_transpose<float, 7>(A, b);
}

// ─── gemm_strided kernels (standard convention: A M×K lead A_RS, B K×N lead B_RS) ──
__global__ void k_rsgemm_6x6x6_6_6(float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm_strided<float, 6, 6, 6, 6, 6>(alpha, A, B, beta, C);
}
__global__ void k_rsgemm_6x6x6_8_8(float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm_strided<float, 6, 6, 6, 8, 8>(alpha, A, B, beta, C);
}
__global__ void k_rsgemm_4x4x4_4_4(float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm_strided<float, 4, 4, 4, 4, 4>(alpha, A, B, beta, C);
}
__global__ void k_rsgemm_4x4x4_6_6(float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm_strided<float, 4, 4, 4, 6, 6>(alpha, A, B, beta, C);
}
// Non-square strided: C 5×7, contract 3; A 5×3 lead 8, B 3×7 lead 6.
__global__ void k_rsgemm_5x7x3_8_6(float alpha, float* A, float* B, float beta, float* C) {
    glass::gemm_strided<float, 5, 7, 3, 8, 6>(alpha, A, B, beta, C);
}

// ─── packed GEMM CT kernels (4×4×{16,32,48,64}) ──────────────────────────────
#define DEFINE_PACKED_GEMM_KERNEL(M, N, K)                                              \
    __global__ void k_packed_gemm_##M##x##N##x##K(                                     \
            float alpha, float* A, float* B, float beta, float* C) {                   \
        glass::gemm<float, M, N, K>(alpha, A, B, beta, C);                             \
    }
DEFINE_PACKED_GEMM_KERNEL(4, 4, 16)
DEFINE_PACKED_GEMM_KERNEL(4, 4, 32)
DEFINE_PACKED_GEMM_KERNEL(4, 4, 48)
DEFINE_PACKED_GEMM_KERNEL(4, 4, 64)

// ─── gemm_tiled kernel (no transpose; A m×k, B k×n, C m×n) ───────────────────
__global__ void k_gemm_tiled(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_B = smem + m * 8;
    glass::gemm_tiled<float, 8>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <op> <cg|simple> <dims...> [args...] [files...]\n", argv[0]);
        return 1;
    }
    const char* op  = argv[1];
    const char* ver = argv[2];
    bool cg = (strcmp(ver, "cg") == 0);

    if (strcmp(op, "gemm_rt") == 0) {
        // gemm_rt <cg|simple> <threads> <m> <n> <k> <ta> <tb> <rmc> <nb> <alpha> <beta> <A> <B> <C>
        int th = atoi(argv[3]);
        int m = atoi(argv[4]), n = atoi(argv[5]), k = atoi(argv[6]);
        int ta = atoi(argv[7]), tb = atoi(argv[8]), rmc = atoi(argv[9]), nb = atoi(argv[10]);
        float alpha = atof(argv[11]), beta = atof(argv[12]);
        float* dA = read_device_vec(argv[13], m * k);   // op(A) is m×k (MK either layout)
        float* dB = read_device_vec(argv[14], k * n);   // op(B) is k×n (KN either layout)
        float* dC = read_device_vec(argv[15], m * n);
        if (cg) {
            #define L_RT_CG(TA,TB,RMC) k_gemm_rt_cg<TA,TB,RMC><<<1,th>>>(m,n,k,alpha,dA,dB,beta,dC,nb)
            GEMM_FLAG_DISPATCH(ta, tb, rmc, L_RT_CG);
        } else {
            #define L_RT(TA,TB,RMC) k_gemm_rt<TA,TB,RMC><<<1,th>>>(m,n,k,alpha,dA,dB,beta,dC,nb)
            GEMM_FLAG_DISPATCH(ta, tb, rmc, L_RT);
        }
        cudaDeviceSynchronize();
        print_device_vec(dC, m * n);

    } else if (strcmp(op, "gemm_ct") == 0 || strcmp(op, "gemm_warp") == 0) {
        // gemm_ct|gemm_warp <unused> <threads> <shape_id> <ta> <tb> <rmc> <nb> <alpha> <beta> <A> <B> <C>
        bool warp = (strcmp(op, "gemm_warp") == 0);
        int th  = warp ? 32 : atoi(argv[3]);
        int id  = atoi(argv[4]);
        int ta  = atoi(argv[5]), tb = atoi(argv[6]), rmc = atoi(argv[7]), nb = atoi(argv[8]);
        float alpha = atof(argv[9]), beta = atof(argv[10]);
        int M, N, K; ct_shape_dims(id, &M, &N, &K);
        float* dA = read_device_vec(argv[11], M * K);
        float* dB = read_device_vec(argv[12], K * N);
        float* dC = read_device_vec(argv[13], M * N);
        switch (id) {
            case 0: launch_gemm_ct< 1, 1, 1>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            case 1: launch_gemm_ct< 2, 3, 4>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            case 2: launch_gemm_ct< 4, 2, 3>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            case 3: launch_gemm_ct< 5, 7, 3>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            case 4: launch_gemm_ct< 8, 1, 5>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            case 5: launch_gemm_ct< 9, 4, 7>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            case 6: launch_gemm_ct< 7, 5, 6>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
            default:launch_gemm_ct<16,16,16>(warp,th,ta,tb,rmc,nb,alpha,dA,dB,beta,dC); break;
        }
        cudaDeviceSynchronize();
        print_device_vec(dC, M * N);

    } else if (strcmp(op, "inv") == 0) {
        int threads = atoi(argv[3]);
        int n = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], 2 * n * n);
        float* scratch; cudaMalloc(&scratch, (2 * n + 1) * sizeof(float));
        if (cg) k_inv_cg<<<1, threads>>>(n, dA, scratch);
        else    k_inv_simple<<<1, threads>>>(n, dA, scratch);
        cudaDeviceSynchronize();
        print_device_vec(dA + n * n, n * n);
        cudaFree(scratch);

    } else if (strcmp(op, "inv_pivot") == 0) {
        int threads = atoi(argv[3]);
        int n = atoi(argv[4]);
        float* dA = read_device_vec(argv[5], 2 * n * n);
        float* scratch; cudaMalloc(&scratch, (3 * n + 1) * sizeof(float));
        k_inv_pivot_simple<<<1, threads>>>(n, dA, scratch);
        cudaDeviceSynchronize();
        print_device_vec(dA + n * n, n * n);
        cudaFree(scratch);

    } else if (strcmp(op, "inv2") == 0) {
        int dimA = atoi(argv[3]); int dimB = atoi(argv[4]); int maxd = atoi(argv[5]);
        float* dA = read_device_vec(argv[6], 2 * dimA * dimA);
        float* dB = read_device_vec(argv[7], 2 * dimB * dimB);
        float* scratch; cudaMalloc(&scratch, (2*dimA + 2*dimB + 2) * sizeof(float));
        k_inv2_simple<<<1, THREADS>>>(dimA, dimB, maxd, dA, dB, scratch);
        cudaDeviceSynchronize();
        print_device_vec(dA + dimA * dimA, dimA * dimA);
        print_device_vec(dB + dimB * dimB, dimB * dimB);
        cudaFree(scratch);

    } else if (strcmp(op, "inv3") == 0) {
        int dimA = atoi(argv[3]); int dimB = atoi(argv[4]); int dimC = atoi(argv[5]); int maxd = atoi(argv[6]);
        float* dA = read_device_vec(argv[7], 2 * dimA * dimA);
        float* dB = read_device_vec(argv[8], 2 * dimB * dimB);
        float* dC = read_device_vec(argv[9], 2 * dimC * dimC);
        float* scratch; cudaMalloc(&scratch, (2*dimA + 2*dimB + 2*dimC + 3) * sizeof(float));
        k_inv3_simple<<<1, THREADS>>>(dimA, dimB, dimC, maxd, dA, dB, dC, scratch);
        cudaDeviceSynchronize();
        print_device_vec(dA + dimA * dimA, dimA * dimA);
        print_device_vec(dB + dimB * dimB, dimB * dimB);
        print_device_vec(dC + dimC * dimC, dimC * dimC);
        cudaFree(scratch);

    } else if (strcmp(op, "chol") == 0) {
        int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], n * n);
        if (cg) k_chol_cg<<<1, THREADS>>>(n, dA);
        else    k_chol_simple<<<1, THREADS>>>(n, dA);
        cudaDeviceSynchronize();
        print_device_vec(dA, n * n);

    } else if (strcmp(op, "trsm") == 0) {
        int n = atoi(argv[3]);
        float* dL = read_device_vec(argv[4], n * n);
        float* db = read_device_vec(argv[5], n);
        if (cg) k_trsm_cg<<<1, THREADS>>>(n, dL, db);
        else    k_trsm_simple<<<1, THREADS>>>(n, dL, db);
        cudaDeviceSynchronize();
        print_device_vec(db, n);

    } else if (strcmp(op, "posv_warp") == 0) {
        int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], n * n);
        float* db = read_device_vec(argv[5], n);
        k_posv_warp_7<<<1, 32>>>(dA, db);
        cudaDeviceSynchronize();
        print_device_vec(db, n);

    } else if (strcmp(op, "gemm_tiled") == 0) {
        // gemm_tiled <cg|simple> <m> <n> <k> <alpha> <beta> <A> <B> <C>
        int m = atoi(argv[3]);
        int n = atoi(argv[4]);
        int k = atoi(argv[5]);
        float alpha = atof(argv[6]);
        float beta  = atof(argv[7]);
        float* dA = read_device_vec(argv[8], m * k);    // A is m×k
        float* dB = read_device_vec(argv[9], k * n);    // B is k×n
        float* dC = read_device_vec(argv[10], m * n);
        int smem_bytes = (m * 8 + 8 * n) * sizeof(float);
        k_gemm_tiled<<<1, THREADS, smem_bytes>>>(m, n, k, alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, m * n);

    } else if (strncmp(op, "rsgemm_", 7) == 0) {
        // rsgemm_<MxNxK_ARSxBRS> <unused> <threads> <alpha> <beta> <A> <B> <C>
        int th = atoi(argv[3]);
        float alpha = atof(argv[4]);
        float beta  = atof(argv[5]);
        int M, N, K, ARS, BRS;
        if      (strcmp(op, "rsgemm_6x6x6_6_6") == 0) { M=6;N=6;K=6;ARS=6;BRS=6; }
        else if (strcmp(op, "rsgemm_6x6x6_8_8") == 0) { M=6;N=6;K=6;ARS=8;BRS=8; }
        else if (strcmp(op, "rsgemm_4x4x4_4_4") == 0) { M=4;N=4;K=4;ARS=4;BRS=4; }
        else if (strcmp(op, "rsgemm_4x4x4_6_6") == 0) { M=4;N=4;K=4;ARS=6;BRS=6; }
        else if (strcmp(op, "rsgemm_5x7x3_8_6") == 0) { M=5;N=7;K=3;ARS=8;BRS=6; }
        else { fprintf(stderr, "bad rsgemm op %s\n", op); return 1; }
        float* dA = read_device_vec(argv[6], ARS * K);   // A is M×K, lead ARS
        float* dB = read_device_vec(argv[7], BRS * N);   // B is K×N, lead BRS
        float* dC = read_device_vec(argv[8], M * N);
        if      (strcmp(op, "rsgemm_6x6x6_6_6") == 0) k_rsgemm_6x6x6_6_6<<<1,th>>>(alpha,dA,dB,beta,dC);
        else if (strcmp(op, "rsgemm_6x6x6_8_8") == 0) k_rsgemm_6x6x6_8_8<<<1,th>>>(alpha,dA,dB,beta,dC);
        else if (strcmp(op, "rsgemm_4x4x4_4_4") == 0) k_rsgemm_4x4x4_4_4<<<1,th>>>(alpha,dA,dB,beta,dC);
        else if (strcmp(op, "rsgemm_4x4x4_6_6") == 0) k_rsgemm_4x4x4_6_6<<<1,th>>>(alpha,dA,dB,beta,dC);
        else                                          k_rsgemm_5x7x3_8_6<<<1,th>>>(alpha,dA,dB,beta,dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, M * N);

    } else if (strncmp(op, "packed_gemm_4x4x", 16) == 0) {
        int K = atoi(op + 16);   // 16/32/48/64
        float alpha = atof(argv[3]);
        float beta  = atof(argv[4]);
        float* dA = read_device_vec(argv[5], 4 * K);   // A is 4×K
        float* dB = read_device_vec(argv[6], K * 4);   // B is K×4
        float* dC = read_device_vec(argv[7], 4 * 4);
        if      (K == 16) k_packed_gemm_4x4x16<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        else if (K == 32) k_packed_gemm_4x4x32<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        else if (K == 48) k_packed_gemm_4x4x48<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        else              k_packed_gemm_4x4x64<<<1, THREADS>>>(alpha, dA, dB, beta, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, 4 * 4);

    } else if (strcmp(op, "indexed_bgemm_4") == 0) {
        uint32_t pairs = (uint32_t)atoi(argv[6]);
        int A_mats = atoi(argv[7]);
        int B_mats = atoi(argv[8]);
        int C_mats = atoi(argv[9]);
        int MAT = 4 * 4;
        int* a_idx = read_device_ivec(argv[10], pairs);
        int* b_idx = read_device_ivec(argv[11], pairs);
        int* c_idx = read_device_ivec(argv[12], pairs);
        float* dA = read_device_vec(argv[13], A_mats * MAT);
        float* dB = read_device_vec(argv[14], B_mats * MAT);
        float* dC = alloc_device_vec(C_mats * MAT);
        k_indexed_bgemm_4<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, C_mats * MAT);

    } else if (strcmp(op, "indexed_bgemm_4_ta") == 0 ||
               strcmp(op, "indexed_bgemm_4_tb") == 0 ||
               strcmp(op, "indexed_bgemm_4_atomic") == 0 ||
               strcmp(op, "indexed_bgemm_4_ta_atomic") == 0) {
        uint32_t pairs = (uint32_t)atoi(argv[6]);
        int A_mats = atoi(argv[7]);
        int B_mats = atoi(argv[8]);
        int C_mats = atoi(argv[9]);
        int MAT = 4 * 4;
        int* a_idx = read_device_ivec(argv[10], pairs);
        int* b_idx = read_device_ivec(argv[11], pairs);
        int* c_idx = read_device_ivec(argv[12], pairs);
        float* dA = read_device_vec(argv[13], A_mats * MAT);
        float* dB = read_device_vec(argv[14], B_mats * MAT);
        float* dC = alloc_device_vec(C_mats * MAT);
        if (strcmp(op, "indexed_bgemm_4_ta") == 0)
            k_indexed_bgemm_4_ta<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        else if (strcmp(op, "indexed_bgemm_4_tb") == 0)
            k_indexed_bgemm_4_tb<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        else if (strcmp(op, "indexed_bgemm_4_atomic") == 0)
            k_indexed_bgemm_4_atomic<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        else
            k_indexed_bgemm_4_ta_atomic<<<1, THREADS>>>(pairs, a_idx, b_idx, c_idx, dA, dB, dC);
        cudaDeviceSynchronize();
        print_device_vec(dC, C_mats * MAT);

    } else {
        fprintf(stderr, "Unknown op: %s\n", op);
        return 1;
    }

    return 0;
}
