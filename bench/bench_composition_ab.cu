// A/B harness for composition-level performance changes.
//
// This harness intentionally does not serve as a correctness test. The same
// public paths are covered by the signed pytest receipt. It times:
//   * riccati_gain: former P*A algebra vs current reuse of P*B;
//   * pcg: former redundant trailing barriers vs current coalesced barriers;
//   * bdsv: former runtime-dimension inner calls vs compile-time inner calls.
//
// Build only while the machine is shared; run only through the quiet timing
// window. Output is ns/problem, min of three trials with trial spread.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "timing_common.cuh"
#include "../glass.cuh"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(3); } } while (0)

static int NPROB = 4096;

template <typename T, uint32_t NX, uint32_t NU>
__device__ void riccati_legacy(const T* P, const T* A, const T* B, const T* R,
                               T* K, T* scratch, int* fail) {
    T* S = scratch;
    T* work = scratch + NU*NU;
    for (uint32_t i = threadIdx.x; i < NU*NU; i += blockDim.x) S[i] = R[i];
    __syncthreads();
    glass::block::congruence_sym<T, NX, NU, true>((T)1, B, P, (T)1, S, work);
    glass::block::bilinear<T, NX, NU, NX>((T)1, B, P, A, (T)0, K, work);
    glass::block::posv<T, NU, NX, false, true>(S, K, (T)0, fail);
    __syncthreads();
}

template <typename T, uint32_t NX, uint32_t NU, bool LEGACY>
__global__ void k_riccati(T* out) {
    extern __shared__ double raw[];
    T* P = reinterpret_cast<T*>(raw);
    T* A = P + NX*NX;
    T* B = A + NX*NX;
    T* R = B + NX*NU;
    T* K = R + NU*NU;
    T* scratch = K + NU*NX;
    for (uint32_t i = threadIdx.x; i < NX*NX; i += blockDim.x) {
        uint32_t r = i % NX, c = i / NX;
        P[i] = (r == c) ? (T)2 : (T)(0.01 * ((r + c) % 5));
        A[i] = (r == c) ? (T)1 : (T)(0.002 * ((r + 2*c) % 7));
    }
    for (uint32_t i = threadIdx.x; i < NX*NU; i += blockDim.x)
        B[i] = (T)(0.01 * (1 + i % 11));
    for (uint32_t i = threadIdx.x; i < NU*NU; i += blockDim.x)
        R[i] = (i % NU == i / NU) ? (T)1 : (T)0;
    __syncthreads();
    __shared__ int fail;
    if (threadIdx.x == 0) fail = 0;
    __syncthreads();
    if constexpr (LEGACY) riccati_legacy<T, NX, NU>(P, A, B, R, K, scratch, &fail);
    else glass::block::riccati_gain<T, NX, NU>(P, A, B, R, K, scratch, (T)0, &fail);
    if (threadIdx.x == 0) out[blockIdx.x] = K[0] + (T)fail;
}

template <typename T, uint32_t SS, uint32_t KP>
__device__ void pcg_legacy(T* x, T* S, T* Pinv, T* b, T* mem, uint32_t* iters) {
    constexpr uint32_t V = (KP + 2) * SS;
    T* Ap = mem; T* sx = Ap + V; T* r = sx + V; T* z = r + V;
    T* p = z + V; T* scr = p + V;
    __shared__ T rho, rho_new, alpha, beta, rho_init;
    glass::block::set_const<T, 5*V>((T)0, mem); __syncthreads();
    glass::block::copy<T, V>(x, sx); __syncthreads();
    glass::block::bdmv<T, KP, SS, false>(r, S, sx); __syncthreads();
    glass::block::axpby<T, V>((T)1, b, (T)-1, r, r); __syncthreads();
    glass::block::bdmv<T, KP, SS, false>(z, p, Pinv, r); __syncthreads();
    glass::block::dot_fast<T, V>(r, z, &rho, scr); __syncthreads();
    if (threadIdx.x == 0) rho_init = rho < (T)0 ? -rho : rho;
    __syncthreads();
    glass::block::bdmv<T, KP, SS, false>(Ap, S, p); __syncthreads();
    glass::block::dot_fast<T, V>(p, Ap, &alpha, scr); __syncthreads();
    if (threadIdx.x == 0) alpha = rho / alpha;
    __syncthreads();
    glass::block::axpy<T, V>(alpha, p, sx);
    glass::block::axpy<T, V>(-alpha, Ap, r);
    __syncthreads();
    glass::block::bdmv<T, KP, SS, false>(z, Pinv, r); __syncthreads();
    glass::block::dot_fast<T, V>(r, z, &rho_new, scr); __syncthreads();
    if (threadIdx.x == 0) { beta = rho_new / rho; rho = rho_new; }
    __syncthreads();
    glass::block::axpby<T, V>((T)1, z, beta, p, p); __syncthreads();
    if (threadIdx.x == 0) *iters = 1;
    glass::block::copy<T, V>(sx, x);
}

template <typename T, uint32_t SS, uint32_t KP, bool LEGACY>
__global__ void k_pcg(T* S, T* Pinv, T* b, T* x, uint32_t* iters) {
    extern __shared__ double raw[];
    T* mem = reinterpret_cast<T*>(raw);
    constexpr uint32_t V = (KP + 2) * SS;
    size_t p = blockIdx.x;
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x) x[p*V + i] = (T)0;
    __syncthreads();
    if constexpr (LEGACY)
        pcg_legacy<T, SS, KP>(x + p*V, S, Pinv, b, mem, iters + p);
    else
        glass::block::pcg<T, SS, KP>(x + p*V, S, Pinv, b, mem,
                                     1, (T)0, (T)0, iters + p);
}

template <typename T, uint32_t SS, uint32_t KP>
__global__ void k_init_pcg(T* S, T* Pinv, T* b) {
    constexpr uint32_t BAND = KP * 3 * SS * SS;
    constexpr uint32_t V = (KP + 2) * SS;
    for (uint32_t i = threadIdx.x; i < BAND; i += blockDim.x) {
        uint32_t local = i % (3 * SS * SS);
        uint32_t row = local / (3 * SS);
        uint32_t col = local % (3 * SS);
        T v = (col >= SS && col < 2 * SS && row == col - SS) ? (T)1 : (T)0;
        S[i] = v;
        Pinv[i] = v;
    }
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x)
        b[i] = (i >= SS && i < (KP + 1) * SS) ? (T)1 : (T)0;
}

template <typename T, uint32_t KP, uint32_t BS>
__device__ void bdsv_legacy(T* matrix, T* vec, T* scratch) {
    constexpr uint32_t BRL = 3 * BS;
    T* s1 = scratch;
    T* s2 = scratch + BS*BS;
    for (uint32_t k = 0; k < KP; ++k) {
        T* strip = matrix + k*BRL*BS;
        if (k > 0) {
            T* prev = matrix + (k-1)*BRL*BS;
            glass::block::load_block<T, BS, BRL, true>(s1, prev, glass::block::BandSlot::MAIN);
            glass::block::load_block<T, BS, BRL, false>(s2, strip, glass::block::BandSlot::LEFT);
            glass::block::trsm<T, BS, BS>(s1, s2);
            glass::block::store_block<T, BS, BRL, false>(strip, glass::block::BandSlot::LEFT, s2);
            glass::block::load_block<T, BS, BRL, true>(s1, strip, glass::block::BandSlot::MAIN);
            glass::block::syrk<T, glass::block::FillMode::Lower, true>(
                BS, BS, (T)-1, s2, (T)1, s1);
        } else {
            glass::block::load_block<T, BS, BRL, true>(s1, strip, glass::block::BandSlot::MAIN);
        }
        glass::block::potrf<T>(BS, s1);
        glass::block::store_block<T, BS, BRL, true>(strip, glass::block::BandSlot::MAIN, s1);
    }
    for (uint32_t k = 0; k < KP; ++k) {
        T* strip = matrix + k*BRL*BS; T* xk = vec + (k+1)*BS;
        if (k > 0) {
            glass::block::load_block<T, BS, BRL, true>(s1, strip, glass::block::BandSlot::LEFT);
            glass::block::gemv<T>(BS, BS, (T)-1, s1, xk-BS, (T)1, xk);
        }
        glass::block::load_block<T, BS, BRL, true>(s1, strip, glass::block::BandSlot::MAIN);
        glass::block::trsv<T>(BS, s1, xk);
    }
    for (uint32_t step = 0; step < KP; ++step) {
        uint32_t k = KP-1-step; T* strip = matrix + k*BRL*BS; T* xk = vec + (k+1)*BS;
        if (k+1 < KP) {
            T* next = matrix + (k+1)*BRL*BS;
            glass::block::load_block<T, BS, BRL, false>(s1, next, glass::block::BandSlot::LEFT);
            glass::block::gemv<T>(BS, BS, (T)-1, s1, xk+BS, (T)1, xk);
        }
        glass::block::load_block<T, BS, BRL, true>(s1, strip, glass::block::BandSlot::MAIN);
        glass::block::trsv<T, glass::block::FillMode::Lower,
                           glass::block::Diag::NonUnit, true>(BS, s1, xk);
    }
}

template <typename T, uint32_t KP, uint32_t BS, bool LEGACY>
__global__ void k_bdsv(T* out) {
    extern __shared__ double raw[];
    T* matrix = reinterpret_cast<T*>(raw);
    constexpr uint32_t BAND = KP*3*BS*BS, V = (KP+2)*BS;
    T* vec = matrix + BAND; T* scratch = vec + V;
    for (uint32_t i = threadIdx.x; i < BAND; i += blockDim.x) {
        uint32_t col = i % (3*BS), row = (i / (3*BS)) % BS;
        matrix[i] = (col >= BS && col < 2*BS && row == col-BS) ? (T)4 : (T)0;
    }
    for (uint32_t i = threadIdx.x; i < V; i += blockDim.x)
        vec[i] = (i >= BS && i < (KP+1)*BS) ? (T)1 : (T)0;
    __syncthreads();
    if constexpr (LEGACY) bdsv_legacy<T, KP, BS>(matrix, vec, scratch);
    else glass::block::bdsv<T, KP, BS>(matrix, vec, scratch);
    if (threadIdx.x == 0) out[blockIdx.x] = vec[BS];
}

template <typename LaunchA, typename LaunchB>
static void report(const char* name, LaunchA legacy, LaunchB current, int reps) {
    static bool current_first = false;
    current_first = !current_first;
    double a, as, b, bs;
    if (current_first) {
        b = tc_time_ns_per_prob(current, reps, NPROB); bs = tc_last_spread_pct();
        a = tc_time_ns_per_prob(legacy, reps, NPROB); as = tc_last_spread_pct();
    } else {
        a = tc_time_ns_per_prob(legacy, reps, NPROB); as = tc_last_spread_pct();
        b = tc_time_ns_per_prob(current, reps, NPROB); bs = tc_last_spread_pct();
    }
    printf("AB %-22s legacy=%.3f spread=%.2f%% current=%.3f spread=%.2f%% ratio=%.3f\n",
           name, a, as, b, bs, a/b);
}

template <typename T>
static void run_dtype(const char* dtype, int reps) {
    T* out; uint32_t* iters;
    CK(cudaMalloc(&out, (size_t)NPROB*sizeof(T)));
    CK(cudaMalloc(&iters, (size_t)NPROB*sizeof(uint32_t)));

    constexpr uint32_t NX=36, NU=12;
    constexpr size_t base = (2*NX*NX + NX*NU + NU*NU + NU*NX)*sizeof(T);
    constexpr size_t old_work = (NU*NU + NX*NX)*sizeof(T);
    constexpr size_t new_work = glass::block::riccati_scratch_bytes<T,NX,NU>();
    report((sizeof(T)==4 ? "riccati36x12_f32" : "riccati36x12_f64"),
           [&]{ k_riccati<T,NX,NU,true><<<NPROB,128,base+old_work>>>(out); },
           [&]{ k_riccati<T,NX,NU,false><<<NPROB,128,base+new_work>>>(out); }, reps);

    constexpr uint32_t SS=6, PK=32, V=(PK+2)*SS, BAND=PK*3*SS*SS;
    T *S, *Pinv, *b, *x;
    CK(cudaMalloc(&S, BAND*sizeof(T))); CK(cudaMalloc(&Pinv, BAND*sizeof(T)));
    CK(cudaMalloc(&b, V*sizeof(T))); CK(cudaMalloc(&x, (size_t)NPROB*V*sizeof(T)));
    k_init_pcg<T,SS,PK><<<1,128>>>(S, Pinv, b);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    // Identity S/Pinv and a nonzero RHS keep arithmetic finite while both
    // variants execute the same fixed one-iteration path.
    size_t pcg_smem = glass::block::pcg_scratch_bytes<T,SS,PK>(128);
    report((sizeof(T)==4 ? "pcg1iter_f32" : "pcg1iter_f64"),
           [&]{ k_pcg<T,SS,PK,true><<<NPROB,128,pcg_smem>>>(S,Pinv,b,x,iters); },
           [&]{ k_pcg<T,SS,PK,false><<<NPROB,128,pcg_smem>>>(S,Pinv,b,x,iters); }, reps);
    cudaFree(S); cudaFree(Pinv); cudaFree(b); cudaFree(x);

    constexpr uint32_t BK=16, BS=6;
    constexpr size_t bd_smem = (BK*3*BS*BS + (BK+2)*BS)*sizeof(T)
                             + glass::block::bdsv_scratch_bytes<T,BS>();
    report((sizeof(T)==4 ? "bdsv6x16_f32" : "bdsv6x16_f64"),
           [&]{ k_bdsv<T,BK,BS,true><<<NPROB,128,bd_smem>>>(out); },
           [&]{ k_bdsv<T,BK,BS,false><<<NPROB,128,bd_smem>>>(out); }, reps);
    cudaFree(out); cudaFree(iters);
    printf("# completed dtype=%s\n", dtype);
}

int main(int argc, char** argv) {
    NPROB = argc > 1 ? atoi(argv[1]) : 4096;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    const char* dtype = argc > 3 ? argv[3] : "both";
    printf("# composition A/B NPROB=%d reps=%d dtype=%s\n", NPROB, reps, dtype);
    tc_warm_gpu();
    if (!strcmp(dtype,"f64") || !strcmp(dtype,"both")) run_dtype<double>("f64", reps);
    if (!strcmp(dtype,"f32") || !strcmp(dtype,"both")) run_dtype<float>("f32", reps);
    return 0;
}
