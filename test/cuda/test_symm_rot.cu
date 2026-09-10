// test_symm_rot.cu — dispatch glass::symm / glass::trmm (L3) and
// glass::rot / glass::warp::rot / glass::rotg (L1), print float32 results.
//
// Usage:
//   symm <threads> <fill l|u> <ct 0|1> <hasbeta 0|1> <n> <m> <alpha> <beta> <A.bin> <B.bin> <C.bin>
//       A.bin: n*n column-major (only the FILL triangle is read — the oracle
//       poisons the other triangle with NaN). B.bin: n*m. C.bin: n*m initial C
//       (read only when hasbeta; the hasbeta=0 overwrite overload must never
//       read it, so the oracle poisons it too).  Prints C (n*m, column-major).
//       Shapes (n, m) in {(4,1), (4,5), (7,1), (7,5), (16,5)}.
//   trmm <threads> <fill l|u> <diag n|u> <trans 0|1> <ct 0|1> <n> <m> <alpha> <A.bin> <B.bin>
//       A.bin: n*n column-major triangular (only the FILL triangle read; the
//       diagonal is not read under Diag::Unit). B.bin: n*m.  Prints C (n*m).
//       Shapes (n, m) in {(4,5), (7,3)}.
//   rot <block|warp> <threads> <ct 0|1> <n> <c> <s> <x.bin> <y.bin>
//       Prints two lines: x then y (each length n).  n in {5, 33}.
//   rotg <a> <b>
//       Prints two lines "c s r": device-computed, then host-computed.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "glass.cuh"

using glass::block::FillMode;
using glass::block::Diag;

// ─── symm ─────────────────────────────────────────────────────────────────────

template <FillMode FILL, bool CT, bool HASBETA, uint32_t N, uint32_t M>
__global__ void k_symm(uint32_t n, uint32_t m, float alpha, float beta,
                       const float* A, const float* B, float* C) {
    if constexpr (CT) {
        if constexpr (HASBETA) glass::block::symm<float, N, M, FILL>(alpha, A, B, beta, C);
        else                   glass::block::symm<float, N, M, FILL>(alpha, A, B, C);
    } else {
        if constexpr (HASBETA) glass::block::symm<float, FILL>(n, m, alpha, A, B, beta, C);
        else                   glass::block::symm<float, FILL>(n, m, alpha, A, B, C);
    }
}

template <uint32_t N, uint32_t M>
static void launch_symm(bool upper, bool ct, bool hasbeta, int th, uint32_t n, uint32_t m,
                        float alpha, float beta, const float* A, const float* B, float* C) {
    if (upper) {
        if (ct)  { if (hasbeta) k_symm<FillMode::Upper, true,  true,  N, M><<<1, th>>>(n, m, alpha, beta, A, B, C);
                   else         k_symm<FillMode::Upper, true,  false, N, M><<<1, th>>>(n, m, alpha, beta, A, B, C); }
        else     { if (hasbeta) k_symm<FillMode::Upper, false, true,  N, M><<<1, th>>>(n, m, alpha, beta, A, B, C);
                   else         k_symm<FillMode::Upper, false, false, N, M><<<1, th>>>(n, m, alpha, beta, A, B, C); }
    } else {
        if (ct)  { if (hasbeta) k_symm<FillMode::Lower, true,  true,  N, M><<<1, th>>>(n, m, alpha, beta, A, B, C);
                   else         k_symm<FillMode::Lower, true,  false, N, M><<<1, th>>>(n, m, alpha, beta, A, B, C); }
        else     { if (hasbeta) k_symm<FillMode::Lower, false, true,  N, M><<<1, th>>>(n, m, alpha, beta, A, B, C);
                   else         k_symm<FillMode::Lower, false, false, N, M><<<1, th>>>(n, m, alpha, beta, A, B, C); }
    }
}

// ─── trmm ─────────────────────────────────────────────────────────────────────

template <FillMode FILL, Diag DIAG, bool TRANS, bool CT, uint32_t N, uint32_t M>
__global__ void k_trmm(uint32_t n, uint32_t m, float alpha,
                       const float* A, const float* B, float* C) {
    if constexpr (CT) glass::block::trmm<float, N, M, FILL, DIAG, TRANS>(alpha, A, B, C);
    else              glass::block::trmm<float, FILL, DIAG, TRANS>(n, m, alpha, A, B, C);
}

template <uint32_t N, uint32_t M>
static void launch_trmm(bool upper, bool unit, bool trans, bool ct, int th,
                        uint32_t n, uint32_t m, float alpha,
                        const float* A, const float* B, float* C) {
    #define T_GO(F, D, TR)                                                        \
        { if (ct) k_trmm<F, D, TR, true,  N, M><<<1, th>>>(n, m, alpha, A, B, C); \
          else    k_trmm<F, D, TR, false, N, M><<<1, th>>>(n, m, alpha, A, B, C); \
          return; }
    if (upper) {
        if (unit) { if (trans) T_GO(FillMode::Upper, Diag::Unit,    true)
                    else       T_GO(FillMode::Upper, Diag::Unit,    false) }
        else      { if (trans) T_GO(FillMode::Upper, Diag::NonUnit, true)
                    else       T_GO(FillMode::Upper, Diag::NonUnit, false) }
    } else {
        if (unit) { if (trans) T_GO(FillMode::Lower, Diag::Unit,    true)
                    else       T_GO(FillMode::Lower, Diag::Unit,    false) }
        else      { if (trans) T_GO(FillMode::Lower, Diag::NonUnit, true)
                    else       T_GO(FillMode::Lower, Diag::NonUnit, false) }
    }
    #undef T_GO
}

// ─── rot / rotg ───────────────────────────────────────────────────────────────

template <bool WARP, bool CT, uint32_t N>
__global__ void k_rot(uint32_t n, float c, float s, float* x, float* y) {
    if constexpr (WARP) {
        if constexpr (CT) glass::warp::rot<float, N>(x, y, c, s);
        else              glass::warp::rot<float>(n, x, y, c, s);
    } else {
        if constexpr (CT) glass::block::rot<float, N>(x, y, c, s);
        else              glass::block::rot<float>(n, x, y, c, s);
    }
}

template <uint32_t N>
static void launch_rot(bool warp, bool ct, int th, uint32_t n, float c, float s,
                       float* x, float* y) {
    if (warp) { if (ct) k_rot<true,  true,  N><<<1, th>>>(n, c, s, x, y);
                else    k_rot<true,  false, N><<<1, th>>>(n, c, s, x, y); }
    else      { if (ct) k_rot<false, true,  N><<<1, th>>>(n, c, s, x, y);
                else    k_rot<false, false, N><<<1, th>>>(n, c, s, x, y); }
}

__global__ void k_rotg(float a, float b, float* out) {
    float c, s, r;
    glass::block::rotg<float>(a, b, c, s, r);
    out[0] = c; out[1] = s; out[2] = r;
}

// ─── main ─────────────────────────────────────────────────────────────────────

static int check_sync() {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "err %s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <symm|trmm|rot|rotg> ...\n", argv[0]); return 1; }
    const char* op = argv[1];

    if (strcmp(op, "symm") == 0) {
        if (argc < 13) { fprintf(stderr, "symm args\n"); return 1; }
        int th = atoi(argv[2]);
        bool upper = (argv[3][0] == 'u');
        bool ct = atoi(argv[4]) != 0;
        bool hasbeta = atoi(argv[5]) != 0;
        uint32_t n = atoi(argv[6]), m = atoi(argv[7]);
        float alpha = atof(argv[8]), beta = atof(argv[9]);
        float* dA = read_device_vec(argv[10], n * n);
        float* dB = read_device_vec(argv[11], n * m);
        float* dC = read_device_vec(argv[12], n * m);
        bool ok = true;
        if      (n == 4  && m == 1) launch_symm<4, 1>(upper, ct, hasbeta, th, n, m, alpha, beta, dA, dB, dC);
        else if (n == 4  && m == 5) launch_symm<4, 5>(upper, ct, hasbeta, th, n, m, alpha, beta, dA, dB, dC);
        else if (n == 7  && m == 1) launch_symm<7, 1>(upper, ct, hasbeta, th, n, m, alpha, beta, dA, dB, dC);
        else if (n == 7  && m == 5) launch_symm<7, 5>(upper, ct, hasbeta, th, n, m, alpha, beta, dA, dB, dC);
        else if (n == 16 && m == 5) launch_symm<16, 5>(upper, ct, hasbeta, th, n, m, alpha, beta, dA, dB, dC);
        else ok = false;
        if (!ok) { fprintf(stderr, "symm: bad shape %ux%u\n", n, m); return 1; }
        if (check_sync()) return 1;
        print_device_vec(dC, n * m);
        return 0;
    }

    if (strcmp(op, "trmm") == 0) {
        if (argc < 12) { fprintf(stderr, "trmm args\n"); return 1; }
        int th = atoi(argv[2]);
        bool upper = (argv[3][0] == 'u');
        bool unit  = (argv[4][0] == 'u');
        bool trans = atoi(argv[5]) != 0;
        bool ct = atoi(argv[6]) != 0;
        uint32_t n = atoi(argv[7]), m = atoi(argv[8]);
        float alpha = atof(argv[9]);
        float* dA = read_device_vec(argv[10], n * n);
        float* dB = read_device_vec(argv[11], n * m);
        float* dC = alloc_device_vec(n * m);
        bool ok = true;
        if      (n == 4 && m == 5) launch_trmm<4, 5>(upper, unit, trans, ct, th, n, m, alpha, dA, dB, dC);
        else if (n == 7 && m == 3) launch_trmm<7, 3>(upper, unit, trans, ct, th, n, m, alpha, dA, dB, dC);
        else ok = false;
        if (!ok) { fprintf(stderr, "trmm: bad shape %ux%u\n", n, m); return 1; }
        if (check_sync()) return 1;
        print_device_vec(dC, n * m);
        return 0;
    }

    if (strcmp(op, "rot") == 0) {
        if (argc < 10) { fprintf(stderr, "rot args\n"); return 1; }
        bool warp = (strcmp(argv[2], "warp") == 0);
        int th = atoi(argv[3]);
        bool ct = atoi(argv[4]) != 0;
        uint32_t n = atoi(argv[5]);
        float c = atof(argv[6]), s = atof(argv[7]);
        float* dx = read_device_vec(argv[8], n);
        float* dy = read_device_vec(argv[9], n);
        if      (n == 5)  launch_rot<5>(warp, ct, th, n, c, s, dx, dy);
        else if (n == 33) launch_rot<33>(warp, ct, th, n, c, s, dx, dy);
        else { fprintf(stderr, "rot: bad n=%u\n", n); return 1; }
        if (check_sync()) return 1;
        print_device_vec(dx, n);
        print_device_vec(dy, n);
        return 0;
    }

    if (strcmp(op, "rotg") == 0) {
        if (argc < 4) { fprintf(stderr, "rotg args\n"); return 1; }
        float a = atof(argv[2]), b = atof(argv[3]);
        float* dout = alloc_device_vec(3);
        k_rotg<<<1, 1>>>(a, b, dout);
        if (check_sync()) return 1;
        print_device_vec(dout, 3);          // device line: c s r
        float hc, hs, hr;
        glass::block::rotg<float>(a, b, hc, hs, hr);
        float host[3] = {hc, hs, hr};
        print_host_vec(host, 3);            // host line:   c s r
        return 0;
    }

    fprintf(stderr, "unknown op %s\n", op);
    return 1;
}
