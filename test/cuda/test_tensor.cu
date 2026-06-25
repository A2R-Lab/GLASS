// test_tensor.cu — dispatch glass::tensor_vec_contract / glass::vec_tensor_vec
// (block / warp / cgrps) and print float32 output.
//
// Usage:
//   tvc <surface> <THREADS> <K> <A> <B> <CONTRACT> <SYMMETRIC> <ACCUMULATE> <TIN_ROW_MAJOR> <T.bin> <v.bin> <M.bin>
//   vtv <surface> <THREADS> <K> <A> <B> <ACCUMULATE> <TIN_ROW_MAJOR> <T.bin> <u.bin> <w.bin> <s.bin>
//     surface   : block | warp | cgrps
//     CONTRACT  : 0=K 1=A 2=B   (tvc only; output is the other two axes, column-major)
//
// tvc prints Mout (OUT0*OUT1); vtv prints s (K).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"   // glass.cuh (block+warp) + glass::cgrps

enum { SURF_BLOCK = 0, SURF_WARP = 1, SURF_CGRPS = 2 };

// ── tensor_vec_contract ──────────────────────────────────────────────────────
template <int SURF, uint32_t K, uint32_t A, uint32_t B, glass::TensorAxis C,
          bool SYM, bool ACC, bool RM>
__global__ void k_tvc(const float* T, const float* v, float* M) {
    if (SURF == SURF_BLOCK)      glass::tensor_vec_contract<float, K, A, B, C, SYM, ACC, RM>(T, v, M);
    else if (SURF == SURF_WARP)  glass::warp::tensor_vec_contract<float, K, A, B, C, SYM, ACC, RM>(T, v, M);
    else                         glass::cgrps::tensor_vec_contract<float, K, A, B, C, SYM, ACC, RM>(T, v, M);
}

template <uint32_t K, uint32_t A, uint32_t B, glass::TensorAxis C, bool SYM, bool ACC, bool RM>
static void tvc_final(int surf, int th, const float* dT, const float* dv, float* dM) {
    if constexpr (!SYM || (C == glass::TensorAxis::K && A == B)) {   // skip invalid SYMMETRIC combos
        if      (surf == SURF_BLOCK) k_tvc<SURF_BLOCK, K, A, B, C, SYM, ACC, RM><<<1, th>>>(dT, dv, dM);
        else if (surf == SURF_WARP)  k_tvc<SURF_WARP,  K, A, B, C, SYM, ACC, RM><<<1, th>>>(dT, dv, dM);
        else                         k_tvc<SURF_CGRPS, K, A, B, C, SYM, ACC, RM><<<1, th>>>(dT, dv, dM);
    }
}

template <uint32_t K, uint32_t A, uint32_t B, glass::TensorAxis C, bool SYM, bool ACC>
static void tvc_rm(int surf, int th, bool rm, const float* dT, const float* dv, float* dM) {
    if (rm) tvc_final<K, A, B, C, SYM, ACC, true >(surf, th, dT, dv, dM);
    else    tvc_final<K, A, B, C, SYM, ACC, false>(surf, th, dT, dv, dM);
}
template <uint32_t K, uint32_t A, uint32_t B, glass::TensorAxis C, bool SYM>
static void tvc_acc(int surf, int th, bool acc, bool rm, const float* dT, const float* dv, float* dM) {
    if (acc) tvc_rm<K, A, B, C, SYM, true >(surf, th, rm, dT, dv, dM);
    else     tvc_rm<K, A, B, C, SYM, false>(surf, th, rm, dT, dv, dM);
}
template <uint32_t K, uint32_t A, uint32_t B, glass::TensorAxis C>
static void tvc_sym(int surf, int th, bool sym, bool acc, bool rm, const float* dT, const float* dv, float* dM) {
    if (sym) tvc_acc<K, A, B, C, true >(surf, th, acc, rm, dT, dv, dM);
    else     tvc_acc<K, A, B, C, false>(surf, th, acc, rm, dT, dv, dM);
}
template <uint32_t K, uint32_t A, uint32_t B>
static void tvc_contract(int surf, int th, int c, bool sym, bool acc, bool rm,
                         const float* dT, const float* dv, float* dM) {
    if (c == 0)      tvc_sym<K, A, B, glass::TensorAxis::K>(surf, th, sym, acc, rm, dT, dv, dM);
    else if (c == 1) tvc_sym<K, A, B, glass::TensorAxis::A>(surf, th, sym, acc, rm, dT, dv, dM);
    else             tvc_sym<K, A, B, glass::TensorAxis::B>(surf, th, sym, acc, rm, dT, dv, dM);
}

// ── vec_tensor_vec ───────────────────────────────────────────────────────────
template <int SURF, uint32_t K, uint32_t A, uint32_t B, bool ACC, bool RM>
__global__ void k_vtv(const float* T, const float* u, const float* w, float* s) {
    if (SURF == SURF_BLOCK)      glass::vec_tensor_vec<float, K, A, B, ACC, RM>(T, u, w, s);
    else if (SURF == SURF_WARP)  glass::warp::vec_tensor_vec<float, K, A, B, ACC, RM>(T, u, w, s);
    else                         glass::cgrps::vec_tensor_vec<float, K, A, B, ACC, RM>(T, u, w, s);
}
template <uint32_t K, uint32_t A, uint32_t B, bool ACC>
static void vtv_rm(int surf, int th, bool rm, const float* dT, const float* du, const float* dw, float* ds) {
    if (rm) { if (surf==SURF_BLOCK) k_vtv<SURF_BLOCK,K,A,B,ACC,true><<<1,th>>>(dT,du,dw,ds);
              else if (surf==SURF_WARP) k_vtv<SURF_WARP,K,A,B,ACC,true><<<1,th>>>(dT,du,dw,ds);
              else k_vtv<SURF_CGRPS,K,A,B,ACC,true><<<1,th>>>(dT,du,dw,ds); }
    else    { if (surf==SURF_BLOCK) k_vtv<SURF_BLOCK,K,A,B,ACC,false><<<1,th>>>(dT,du,dw,ds);
              else if (surf==SURF_WARP) k_vtv<SURF_WARP,K,A,B,ACC,false><<<1,th>>>(dT,du,dw,ds);
              else k_vtv<SURF_CGRPS,K,A,B,ACC,false><<<1,th>>>(dT,du,dw,ds); }
}
template <uint32_t K, uint32_t A, uint32_t B>
static void vtv_acc(int surf, int th, bool acc, bool rm, const float* dT, const float* du, const float* dw, float* ds) {
    if (acc) vtv_rm<K, A, B, true >(surf, th, rm, dT, du, dw, ds);
    else     vtv_rm<K, A, B, false>(surf, th, rm, dT, du, dw, ds);
}

// Shapes (K,A,B): consumer dims (Hxx 14/14/14, Hux 14/7/14), rectangular, and
// shapes whose contracted axis spans the 32-lane boundary (33, 64).
#define SHAPES(_) \
    _(14,14,14) _(8,8,8) _(5,5,5) _(14,7,14) _(3,4,5) \
    _(33,4,4)   _(3,33,4) _(3,4,33) _(64,3,3) _(4,8,8)

static uint32_t out0(int c, uint32_t K, uint32_t A, uint32_t B) { return c==0 ? A : K; }
static uint32_t out1(int c, uint32_t K, uint32_t A, uint32_t B) { return c==0 ? B : (c==1 ? B : A); }
static uint32_t cdim(int c, uint32_t K, uint32_t A, uint32_t B) { return c==0 ? K : (c==1 ? A : B); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "need op (tvc|vtv)\n"); return 1; }
    const char* op = argv[1];
    int surf = (strcmp(argv[2], "warp") == 0) ? SURF_WARP
             : (strcmp(argv[2], "cgrps") == 0) ? SURF_CGRPS : SURF_BLOCK;
    int th  = atoi(argv[3]);
    uint32_t K = (uint32_t)atoi(argv[4]);
    uint32_t A = (uint32_t)atoi(argv[5]);
    uint32_t B = (uint32_t)atoi(argv[6]);

    if (strcmp(op, "tvc") == 0) {
        // tvc <surface> <THREADS> <K> <A> <B> <CONTRACT> <SYM> <ACC> <RM> <T> <v> <M>
        int c    = atoi(argv[7]);
        bool sym = atoi(argv[8]) != 0;
        bool acc = atoi(argv[9]) != 0;
        bool rm  = atoi(argv[10]) != 0;
        uint32_t mlen = out0(c,K,A,B) * out1(c,K,A,B);
        float* dT = read_device_vec(argv[11], K*A*B);
        float* dv = read_device_vec(argv[12], cdim(c,K,A,B));
        float* dM = read_device_vec(argv[13], mlen);
        bool ok = false;
        #define DTVC(KK,AA,BB) if (!ok && K==KK && A==AA && B==BB) { tvc_contract<KK,AA,BB>(surf,th,c,sym,acc,rm,dT,dv,dM); ok=true; }
        SHAPES(DTVC)
        #undef DTVC
        if (!ok) { fprintf(stderr, "unsupported shape\n"); return 1; }
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { fprintf(stderr, "kernel err: %s\n", cudaGetErrorString(e)); return 1; }
        print_device_vec(dM, mlen);
    } else if (strcmp(op, "vtv") == 0) {
        // vtv <surface> <THREADS> <K> <A> <B> <ACC> <RM> <T> <u> <w> <s>
        bool acc = atoi(argv[7]) != 0;
        bool rm  = atoi(argv[8]) != 0;
        float* dT = read_device_vec(argv[9], K*A*B);
        float* du = read_device_vec(argv[10], A);
        float* dw = read_device_vec(argv[11], B);
        float* ds = read_device_vec(argv[12], K);
        bool ok = false;
        #define DVTV(KK,AA,BB) if (!ok && K==KK && A==AA && B==BB) { vtv_acc<KK,AA,BB>(surf,th,acc,rm,dT,du,dw,ds); ok=true; }
        SHAPES(DVTV)
        #undef DVTV
        if (!ok) { fprintf(stderr, "unsupported shape\n"); return 1; }
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { fprintf(stderr, "kernel err: %s\n", cudaGetErrorString(e)); return 1; }
        print_device_vec(ds, K);
    } else { fprintf(stderr, "bad op %s\n", op); return 1; }
    return 0;
}
