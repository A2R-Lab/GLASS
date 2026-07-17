// test_thread.cu — driver for the glass::thread:: surface (one problem per THREAD).
//
// Two launch models over the SAME inputs, selected by the <model> argument:
//
//   thread  <<<ceil(P/TPB), TPB>>>  — thread p owns problem p; operands are copied
//                                     into thread-local arrays, computed on, copied
//                                     back. This is the tier under test.
//   block1  <<<P, 1>>>              — block p owns problem p but runs ONE thread, so
//                                     the block-scoped glass:: op degenerates to the
//                                     same sequential algorithm. This is the ORACLE.
//
// Why block1 is the oracle: every thread:: op delegates to the same *_impl body
// via ThreadBarrier{rank=0,size=1,no-op sync}, so thread and block1 run the same
// algorithm over the same operand order. They are still DIFFERENT template
// instantiations — the no-op sync removes the optimization fences __syncthreads()
// gives the block build, so nvcc may contract FMA chains differently and the two
// can disagree by a last ULP on borderline-rounding inputs (measured on
// sm_120/nvcc 13.2: f64 potrf/posv, ~0.1% of elements, 1 ulp; -fmad=false
// restores bit-identity). test_thread.py therefore asserts a tight ULP bound,
// not bit-equality. (`dot` is looser still by design: block-scoped dot reduces
// with a halving TREE, thread::dot accumulates serially, so they agree only to
// float tolerance.)
//
// P is deliberately >32 in the pytest driver so problems span multiple warps and
// several blocks with a RAGGED tail — the configuration that catches a stray
// block-wide __syncthreads() inside a thread:: op (divergent participation once
// the tail block's out-of-range threads have returned ⇒ UB/hang).
//
// DTYPE: templated on the scalar type; <dtype> = f32|f64 picks the instantiation.
// Operand .bin files are always float32 (what the Python harness writes); the
// driver widens them to T on load, so the f64 path exercises the tier's "BOTH
// dtypes" register-residency claim (see CLAUDE.md) over the same inputs.
//
// FLAGS: trsv and gemv carry their compile-time flag surface so the sweep hits it
// rather than trusting the thread:: overloads to forward it correctly —
//   gemv <trans> <rowmajor>              (each 0/1)
//   trsv <lower> <unit>  <trans>         (each 0/1)
//   trsm <lower> <unit>  <trans>         (each 0/1; B is N x TRHS, TRHS=3)
// both are instantiated for the block1 oracle too, so the ULP-bounded check
// covers every flag combination.
//
// Usage: ./test_thread <op> <model> <dtype> <N> <P> [flags...] <files...>
//   ops:    dot gemv gemm potrf trsv posv potrs reduce nrm2 asum nrm1_diff
//           axpy scal copy rot symmetrize axpy_strided copy_strided
//           trsm ldlt ldlt_solve inv
//           syrk syr2k tvc vtv congr bilinear caccum riccati
//   models: thread block1     dtype: f32 f64

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "helpers.cuh"
#include "../../glass.cuh"

#define TPB 64   // threads per block for the `thread` model (ragged tail when P%TPB)

// trsm right-hand-side width: compile-time like N, fixed at 3 so the sweep
// exercises a non-square (N x NRHS) B without widening the driver's CLI.
#define TRHS 3

using glass::FillMode;
using glass::Diag;
using glass::TensorAxis;

// ─── dtype-generic I/O (helpers.cuh is float32-only) ─────────────────────────
// Read a float32 .bin and widen to T; print with round-trip precision so that
// two bit-equal T values always render to identical text (keeps the thread vs
// block1 comparison exact even in f64).

template <typename T>
static T* read_dev(const char* path, int n) {
    float* h = read_host_vec(path, n);
    T* hT = (T*)malloc(n * sizeof(T));
    for (int i = 0; i < n; i++) hT[i] = (T)h[i];
    free(h);
    T* d; cudaMalloc(&d, n * sizeof(T));
    cudaMemcpy(d, hT, n * sizeof(T), cudaMemcpyHostToDevice);
    free(hT);
    return d;
}
template <typename T>
static T* alloc_dev(int n) { T* d; cudaMalloc(&d, n * sizeof(T)); cudaMemset(d, 0, n * sizeof(T)); return d; }

template <typename T> __global__ void print_kernelT(const T* d, int n) {
    for (int i = 0; i < n; i++) {
        if constexpr (sizeof(T) == 8) printf("%.17g", (double)d[i]);
        else                          printf("%.9g",  (double)d[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");
}
template <typename T> static void print_dev(const T* d, int n) {
    print_kernelT<T><<<1,1>>>(d, n); cudaDeviceSynchronize();
}

// ─── THREAD model: thread p owns problem p, operands thread-local ─────────────

template <typename T, int N> __global__ void kt_dot(int P, T* x, T* y, T* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    out[p] = glass::thread::dot<T, N>(x + (size_t)p*N, y + (size_t)p*N);
}
template <typename T, int N, bool TR, bool RM> __global__ void kt_gemv(int P, T alpha, T* A, T* x, T* y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], xv[N], yv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::gemv<T, N, N, TR, RM>(alpha, a, xv, yv);
    for (int i = 0; i < N;   i++) y[(size_t)p*N + i] = yv[i];
}
template <typename T, int N> __global__ void kt_gemm(int P, T alpha, T* A, T* B, T* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], b[N*N], c[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    for (int i = 0; i < N*N; i++) b[i] = B[(size_t)p*N*N + i];
    glass::thread::gemm<T, N, N, N>(alpha, a, b, c);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N + i] = c[i];
}
template <typename T, int N> __global__ void kt_potrf(int P, T* A) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    glass::thread::potrf<T, N>(a);
    for (int i = 0; i < N*N; i++) A[(size_t)p*N*N + i] = a[i];
}
template <typename T, int N, FillMode F, Diag D, bool TR> __global__ void kt_trsv(int P, T* A, T* x) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], xv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::trsv<T, N, F, D, TR>(a, xv);
    for (int i = 0; i < N;   i++) x[(size_t)p*N + i] = xv[i];
}
template <typename T, int N> __global__ void kt_posv(int P, T* A, T* b) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], bv[N];
    for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[(size_t)p*N + i];
    glass::thread::posv<T, N>(a, bv);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = bv[i];
}
template <typename T, int N> __global__ void kt_potrs(int P, T* L, T* b) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T l[N*N], bv[N];
    for (int i = 0; i < N*N; i++) l[i]  = L[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[(size_t)p*N + i];
    glass::thread::potrs<T, N>(l, bv);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = bv[i];
}

template <typename T, int N> __global__ void kt_reduce(int P, T* x, T* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N];
    for (int i = 0; i < N; i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::reduce<T, N>(xv);          // in-place: sum lands in xv[0]
    out[p] = xv[0];
}
template <typename T, int N> __global__ void kt_nrm2(int P, T* x, T* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N];
    for (int i = 0; i < N; i++) xv[i] = x[(size_t)p*N + i];
    out[p] = glass::thread::nrm2<T, N>(xv);
}
template <typename T, int N> __global__ void kt_asum(int P, T* x, T* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N];
    for (int i = 0; i < N; i++) xv[i] = x[(size_t)p*N + i];
    out[p] = glass::thread::asum<T, N>(xv);
}
template <typename T, int N> __global__ void kt_nrm1_diff(int P, T* x, T* y, T* out) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N], yv[N];
    for (int i = 0; i < N; i++) { xv[i] = x[(size_t)p*N + i]; yv[i] = y[(size_t)p*N + i]; }
    out[p] = glass::thread::nrm1_diff<T, N>(xv, yv);
}
template <typename T, int N> __global__ void kt_axpy(int P, T alpha, T* x, T* y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N], yv[N];
    for (int i = 0; i < N; i++) { xv[i] = x[(size_t)p*N + i]; yv[i] = y[(size_t)p*N + i]; }
    glass::thread::axpy<T, N>(alpha, xv, yv);
    for (int i = 0; i < N; i++) y[(size_t)p*N + i] = yv[i];
}
template <typename T, int N> __global__ void kt_scal(int P, T alpha, T* x) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N];
    for (int i = 0; i < N; i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::scal<T, N>(alpha, xv);
    for (int i = 0; i < N; i++) x[(size_t)p*N + i] = xv[i];
}
template <typename T, int N> __global__ void kt_copy(int P, T* x, T* y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N], yv[N];
    for (int i = 0; i < N; i++) xv[i] = x[(size_t)p*N + i];
    glass::thread::copy<T, N>(xv, yv);
    for (int i = 0; i < N; i++) y[(size_t)p*N + i] = yv[i];
}
// rot mutates BOTH vectors; results land in out = [all x' | all y'] so one
// buffer carries both back to the harness (print_dev emits a single line).
template <typename T, int N> __global__ void kt_rot(int P, T* x, T* y, T* out, T c, T s) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xv[N], yv[N];
    for (int i = 0; i < N; i++) { xv[i] = x[(size_t)p*N + i]; yv[i] = y[(size_t)p*N + i]; }
    glass::thread::rot<T, N>(xv, yv, c, s);
    for (int i = 0; i < N; i++) {
        out[(size_t)p*N + i]              = xv[i];
        out[(size_t)P*N + (size_t)p*N + i] = yv[i];
    }
}
template <typename T, int N> __global__ void kt_symmetrize(int P, T* A) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    glass::thread::symmetrize<T, N>(a);
    for (int i = 0; i < N*N; i++) A[(size_t)p*N*N + i] = a[i];
}
// strided ops: M=N, X at lead N+1, Y at lead N+2; whole padded buffers are
// copied thread-local and written back, so the untouched pads are compared too.
template <typename T, int N> __global__ void kt_axpy_strided(int P, T alpha, T* X, T* Y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xl[(N+1)*N], yl[(N+2)*N];
    for (int i = 0; i < (N+1)*N; i++) xl[i] = X[(size_t)p*(N+1)*N + i];
    for (int i = 0; i < (N+2)*N; i++) yl[i] = Y[(size_t)p*(N+2)*N + i];
    glass::thread::axpy_strided<T, N, N, N+2, N+1>(alpha, xl, yl);
    for (int i = 0; i < (N+2)*N; i++) Y[(size_t)p*(N+2)*N + i] = yl[i];
}
template <typename T, int N> __global__ void kt_copy_strided(int P, T alpha, T* X, T* Y) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T xl[(N+1)*N], yl[(N+2)*N];
    for (int i = 0; i < (N+1)*N; i++) xl[i] = X[(size_t)p*(N+1)*N + i];
    for (int i = 0; i < (N+2)*N; i++) yl[i] = Y[(size_t)p*(N+2)*N + i];
    glass::thread::copy_strided<T, N, N, N+2, N+1>(alpha, xl, yl);
    for (int i = 0; i < (N+2)*N; i++) Y[(size_t)p*(N+2)*N + i] = yl[i];
}

template <typename T, int N, FillMode F, Diag D, bool TR> __global__ void kt_trsm(int P, T* A, T* B) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], bm[N*TRHS];
    for (int i = 0; i < N*N;    i++) a[i]  = A[(size_t)p*N*N + i];
    for (int i = 0; i < N*TRHS; i++) bm[i] = B[(size_t)p*N*TRHS + i];
    glass::thread::trsm<T, N, TRHS, F, D, TR>(a, bm);
    for (int i = 0; i < N*TRHS; i++) B[(size_t)p*N*TRHS + i] = bm[i];
}
template <typename T, int N> __global__ void kt_ldlt(int P, T* A) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N];
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    glass::thread::ldlt<T, N>(a);
    for (int i = 0; i < N*N; i++) A[(size_t)p*N*N + i] = a[i];
}
template <typename T, int N> __global__ void kt_ldlt_solve(int P, T* LD, T* b) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T ld[N*N], bv[N];
    for (int i = 0; i < N*N; i++) ld[i] = LD[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) bv[i] = b[(size_t)p*N + i];
    glass::thread::ldlt_solve<T, N>(ld, bv);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = bv[i];
}
template <typename T, int N> __global__ void kt_inv(int P, T* A) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T aug[2*N*N], sc[2*N + 1];               // thread-local scratch: the tier's intended use
    for (int i = 0; i < 2*N*N; i++) aug[i] = A[(size_t)p*2*N*N + i];
    glass::thread::inv<T, N>(aug, sc);
    for (int i = 0; i < 2*N*N; i++) A[(size_t)p*2*N*N + i] = aug[i];
}

// Deterministic ACCUMULATE seed, identical across models and in the numpy oracle.
template <typename T> __device__ __host__ inline T acc_pat(int i) { return (T)0.25 * (T)(i % 7); }

template <typename T, int N, FillMode F, bool TR> __global__ void kt_syrk(int P, T* A, T* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], c[N*N];
    for (int i = 0; i < N*N; i++) { a[i] = A[(size_t)p*N*N + i]; c[i] = (T)0; }
    glass::thread::syrk<T, N, N, F, TR>((T)1, a, c);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N + i] = c[i];
}
template <typename T, int N, FillMode F, bool TR> __global__ void kt_syr2k(int P, T* A, T* B, T* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T a[N*N], b[N*N], c[N*N];
    for (int i = 0; i < N*N; i++) { a[i] = A[(size_t)p*N*N + i]; b[i] = B[(size_t)p*N*N + i]; c[i] = (T)0; }
    glass::thread::syr2k<T, N, N, F, TR>((T)1, a, b, c);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N + i] = c[i];
}
template <typename T, int N, TensorAxis C, bool SYM, bool ACC> __global__ void kt_tvc(int P, T* Tns, T* v, T* M) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T t[N*N*N], vv[N], m[N*N];
    for (int i = 0; i < N*N*N; i++) t[i]  = Tns[(size_t)p*N*N*N + i];
    for (int i = 0; i < N;     i++) vv[i] = v[(size_t)p*N + i];
    for (int i = 0; i < N*N;   i++) m[i]  = ACC ? acc_pat<T>(i) : (T)0;
    glass::thread::tensor_vec_contract<T, N, N, N, C, SYM, ACC>(t, vv, m);
    for (int i = 0; i < N*N; i++) M[(size_t)p*N*N + i] = m[i];
}
template <typename T, int N, bool ACC> __global__ void kt_vtv(int P, T* Tns, T* u, T* w, T* s) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T t[N*N*N], uv[N], wv[N], sv[N];
    for (int i = 0; i < N*N*N; i++) t[i]  = Tns[(size_t)p*N*N*N + i];
    for (int i = 0; i < N;     i++) { uv[i] = u[(size_t)p*N + i]; wv[i] = w[(size_t)p*N + i]; }
    for (int i = 0; i < N;     i++) sv[i] = ACC ? acc_pat<T>(i) : (T)0;
    glass::thread::vec_tensor_vec<T, N, N, N, ACC>(t, uv, wv, sv);
    for (int i = 0; i < N; i++) s[(size_t)p*N + i] = sv[i];
}
template <typename T, int N, bool ACC> __global__ void kt_congr(int P, T* X, T* M, T* Q) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T x[N*N], m[N*N], q[N*N], scr[N*N];               // scr: thread-local workspace (M·X)
    for (int i = 0; i < N*N; i++) { x[i] = X[(size_t)p*N*N + i]; m[i] = M[(size_t)p*N*N + i]; }
    for (int i = 0; i < N*N; i++) q[i] = ACC ? acc_pat<T>(i) : (T)0;
    glass::thread::congruence_sym<T, N, N, ACC>((T)1, x, m, (T)1, q, scr);
    for (int i = 0; i < N*N; i++) Q[(size_t)p*N*N + i] = q[i];
}
template <typename T, int N> __global__ void kt_bilinear(int P, T* X, T* M, T* Y, T* R) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T x[N*N], m[N*N], yv[N*N], r[N*N], scr[N*N];      // scr: thread-local workspace (M·Y)
    for (int i = 0; i < N*N; i++) { x[i] = X[(size_t)p*N*N + i]; m[i] = M[(size_t)p*N*N + i]; yv[i] = Y[(size_t)p*N*N + i]; }
    glass::thread::bilinear<T, N, N, N, false>((T)1, x, m, yv, (T)0, r, scr);
    for (int i = 0; i < N*N; i++) R[(size_t)p*N*N + i] = r[i];
}
template <typename T, int N, bool ACC> __global__ void kt_caccum(int P, T* G, T* M, T* C) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P) return;
    T g[N*N], m[N*N], c[N*N], scr[2*N*N];             // scr: Gᵀ then M·Gᵀ
    for (int i = 0; i < N*N; i++) { g[i] = G[(size_t)p*N*N + i]; m[i] = M[(size_t)p*N*N + i]; }
    for (int i = 0; i < N*N; i++) c[i] = ACC ? acc_pat<T>(i) : (T)0;
    glass::thread::congruence_accum<T, N, N, ACC>((T)1, g, m, (T)1, c, scr);
    for (int i = 0; i < N*N; i++) C[(size_t)p*N*N + i] = c[i];
}
template <typename T, int N, bool REG> __global__ void kt_riccati(int P_, T* Pm, T* Am, T* Bm, T* Rm, T* K) {
    int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= P_) return;
    T pp[N*N], aa[N*N], bb[N*N], rr[N*N], kk[N*N], scr[2*N*N];  // riccati_scratch NX=NU=N
    for (int i = 0; i < N*N; i++) { pp[i] = Pm[(size_t)p*N*N + i]; aa[i] = Am[(size_t)p*N*N + i];
                                    bb[i] = Bm[(size_t)p*N*N + i]; rr[i] = Rm[(size_t)p*N*N + i]; }
    glass::thread::riccati_gain<T, N, N, REG>(pp, aa, bb, rr, kk, scr, (T)0.05, nullptr);
    for (int i = 0; i < N*N; i++) K[(size_t)p*N*N + i] = kk[i];
}

// ─── BLOCK-1 oracle: block p owns problem p, launched with ONE thread ─────────

template <typename T, int N> __global__ void kb1_dot(T* x, T* y, T* out) {
    int p = blockIdx.x;
    glass::dot<T, N>(x + (size_t)p*N, y + (size_t)p*N);   // destructive: result in y[0]
    out[p] = y[(size_t)p*N];
}
template <typename T, int N, bool TR, bool RM> __global__ void kb1_gemv(T alpha, T* A, T* x, T* y) {
    int p = blockIdx.x;
    glass::gemv<T, N, N, TR, RM>(alpha, A + (size_t)p*N*N, x + (size_t)p*N, y + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_gemm(T alpha, T* A, T* B, T* C) {
    int p = blockIdx.x;
    glass::gemm<T, N, N, N>(alpha, A + (size_t)p*N*N, B + (size_t)p*N*N, C + (size_t)p*N*N);
}
template <typename T, int N> __global__ void kb1_potrf(T* A) {
    int p = blockIdx.x;
    glass::potrf<T, N>(A + (size_t)p*N*N);
}
template <typename T, int N, FillMode F, Diag D, bool TR> __global__ void kb1_trsv(T* A, T* x) {
    int p = blockIdx.x;
    glass::trsv<T, N, F, D, TR>(A + (size_t)p*N*N, x + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_posv(T* A, T* b) {
    int p = blockIdx.x;
    glass::posv<T, N>(A + (size_t)p*N*N, b + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_potrs(T* L, T* b) {
    int p = blockIdx.x;
    glass::potrs<T, N>(L + (size_t)p*N*N, b + (size_t)p*N);
}

template <typename T, int N> __global__ void kb1_reduce(T* x, T* out) {
    int p = blockIdx.x;
    glass::reduce<T, N>(x + (size_t)p*N);     // destructive halving tree; sum in x[0]
    out[p] = x[(size_t)p*N];
}
// block1 oracle = runtime-n nrm2_fast (block surface has no plain nrm2; _fast is
// the standard block-scoped variant). Destructive: result lands in x[0].
template <typename T, int N> __global__ void kb1_nrm2(T* x, T* out) {
    int p = blockIdx.x;
    __shared__ T scr[1];                       // ceil(1 thread / 32) = 1 slot
    glass::nrm2_fast<T>((uint32_t)N, x + (size_t)p*N, scr);
    out[p] = x[(size_t)p*N];
}
// block1 oracle = runtime-n asum_fast (matches semantics; destructive into x[0]).
template <typename T, int N> __global__ void kb1_asum(T* x, T* out) {
    int p = blockIdx.x;
    __shared__ T scr[1];
    glass::asum_fast<T>((uint32_t)N, x + (size_t)p*N, scr);
    out[p] = x[(size_t)p*N];
}
// block1 oracle = runtime-n nrm1_diff_fast (matches semantics; result in out[0]).
template <typename T, int N> __global__ void kb1_nrm1_diff(T* x, T* y, T* out) {
    int p = blockIdx.x;
    __shared__ T scr[1];
    glass::nrm1_diff_fast<T>((uint32_t)N, x + (size_t)p*N, y + (size_t)p*N, out + p, scr);
}
template <typename T, int N> __global__ void kb1_axpy(T alpha, T* x, T* y) {
    int p = blockIdx.x;
    glass::axpy<T, N>(alpha, x + (size_t)p*N, y + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_scal(T alpha, T* x) {
    int p = blockIdx.x;
    glass::scal<T, N>(alpha, x + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_copy(T* x, T* y) {
    int p = blockIdx.x;
    glass::copy<T, N>(x + (size_t)p*N, y + (size_t)p*N);
}
template <typename T, int N> __global__ void kb1_rot(int P, T* x, T* y, T* out, T c, T s) {
    int p = blockIdx.x;
    glass::rot<T, N>(x + (size_t)p*N, y + (size_t)p*N, c, s);
    for (int i = 0; i < N; i++) {              // same [all x' | all y'] packing as kt_rot
        out[(size_t)p*N + i]              = x[(size_t)p*N + i];
        out[(size_t)P*N + (size_t)p*N + i] = y[(size_t)p*N + i];
    }
}
template <typename T, int N> __global__ void kb1_symmetrize(T* A) {
    int p = blockIdx.x;
    glass::symmetrize<T, N>(A + (size_t)p*N*N);
}
template <typename T, int N> __global__ void kb1_axpy_strided(T alpha, T* X, T* Y) {
    int p = blockIdx.x;
    glass::axpy_strided<T, N, N, N+2, N+1>(alpha, X + (size_t)p*(N+1)*N, Y + (size_t)p*(N+2)*N);
}
template <typename T, int N> __global__ void kb1_copy_strided(T alpha, T* X, T* Y) {
    int p = blockIdx.x;
    glass::copy_strided<T, N, N, N+2, N+1>(alpha, X + (size_t)p*(N+1)*N, Y + (size_t)p*(N+2)*N);
}

template <typename T, int N, FillMode F, Diag D, bool TR> __global__ void kb1_trsm(T* A, T* B) {
    int p = blockIdx.x;
    glass::trsm<T, N, TRHS, F, D, TR>(A + (size_t)p*N*N, B + (size_t)p*N*TRHS);
}
template <typename T, int N> __global__ void kb1_ldlt(T* A) {
    int p = blockIdx.x;
    glass::ldlt<T, N>(A + (size_t)p*N*N, nullptr);   // NON-pivoted block call = the oracle
}
template <typename T, int N> __global__ void kb1_ldlt_solve(T* LD, T* b) {
    int p = blockIdx.x;
    glass::ldlt_solve<T, N>(LD + (size_t)p*N*N, b + (size_t)p*N);   // piv=nullptr (non-pivoted)
}
template <typename T, int N> __global__ void kb1_inv(T* A) {
    int p = blockIdx.x;
    __shared__ T sc[2*N + 1];                 // block-scoped scratch, per the glass:: contract
    glass::inv<T, N>(A + (size_t)p*2*N*N, sc);
}

template <typename T, int N, FillMode F, bool TR> __global__ void kb1_syrk(T* A, T* C) {
    int p = blockIdx.x;
    glass::syrk<T, N, N, F, TR>((T)1, A + (size_t)p*N*N, C + (size_t)p*N*N);   // C pre-zeroed by alloc_dev
}
template <typename T, int N, FillMode F, bool TR> __global__ void kb1_syr2k(T* A, T* B, T* C) {
    int p = blockIdx.x;
    glass::syr2k<T, N, N, F, TR>((T)1, A + (size_t)p*N*N, B + (size_t)p*N*N, C + (size_t)p*N*N);
}
template <typename T, int N, TensorAxis C, bool SYM, bool ACC> __global__ void kb1_tvc(T* Tns, T* v, T* M) {
    int p = blockIdx.x;
    T* m = M + (size_t)p*N*N;
    for (int i = 0; i < N*N; i++) m[i] = ACC ? acc_pat<T>(i) : (T)0;
    __syncthreads();
    glass::tensor_vec_contract<T, N, N, N, C, SYM, ACC>(Tns + (size_t)p*N*N*N, v + (size_t)p*N, m);
}
template <typename T, int N, bool ACC> __global__ void kb1_vtv(T* Tns, T* u, T* w, T* s) {
    int p = blockIdx.x;
    T* sv = s + (size_t)p*N;
    for (int i = 0; i < N; i++) sv[i] = ACC ? acc_pat<T>(i) : (T)0;
    __syncthreads();
    glass::vec_tensor_vec<T, N, N, N, ACC>(Tns + (size_t)p*N*N*N, u + (size_t)p*N, w + (size_t)p*N, sv);
}
template <typename T, int N, bool ACC> __global__ void kb1_congr(T* X, T* M, T* Q, T* scr) {
    int p = blockIdx.x;
    T* q = Q + (size_t)p*N*N;
    for (int i = 0; i < N*N; i++) q[i] = ACC ? acc_pat<T>(i) : (T)0;
    __syncthreads();
    glass::congruence_sym<T, N, N, ACC>((T)1, X + (size_t)p*N*N, M + (size_t)p*N*N, (T)1, q, scr + (size_t)p*2*N*N);
}
template <typename T, int N> __global__ void kb1_bilinear(T* X, T* M, T* Y, T* R, T* scr) {
    int p = blockIdx.x;
    glass::bilinear<T, N, N, N, false>((T)1, X + (size_t)p*N*N, M + (size_t)p*N*N, Y + (size_t)p*N*N,
                                       (T)0, R + (size_t)p*N*N, scr + (size_t)p*2*N*N);
}
template <typename T, int N, bool ACC> __global__ void kb1_caccum(T* G, T* M, T* C, T* scr) {
    int p = blockIdx.x;
    T* c = C + (size_t)p*N*N;
    for (int i = 0; i < N*N; i++) c[i] = ACC ? acc_pat<T>(i) : (T)0;
    __syncthreads();
    glass::congruence_accum<T, N, N, ACC>((T)1, G + (size_t)p*N*N, M + (size_t)p*N*N, (T)1, c, scr + (size_t)p*2*N*N);
}
template <typename T, int N, bool REG> __global__ void kb1_riccati(T* Pm, T* Am, T* Bm, T* Rm, T* K, T* scr) {
    int p = blockIdx.x;
    glass::riccati_gain<T, N, N, REG>(Pm + (size_t)p*N*N, Am + (size_t)p*N*N, Bm + (size_t)p*N*N,
                                      Rm + (size_t)p*N*N, K + (size_t)p*N*N,
                                      scr + (size_t)p*2*N*N, (T)0.05, nullptr);
}

// ─── dispatch ────────────────────────────────────────────────────────────────

static int  g_P;
static bool g_thread;          // true => thread model, false => block1 oracle
static int  g_f0, g_f1, g_f2;  // op-specific compile-time flags (parsed in main)

// grid/block for the model under test
static inline dim3 grid()  { return g_thread ? dim3((g_P + TPB - 1) / TPB) : dim3(g_P); }
static inline dim3 block() { return g_thread ? dim3(TPB) : dim3(1); }

// Runtime flag ints -> compile-time template args, instantiating both models.
template <typename T, int N, bool TR, bool RM>
static void launch_gemv(T alpha, T* A, T* x, T* y) {
    if (g_thread) kt_gemv<T,N,TR,RM><<<grid(),block()>>>(g_P, alpha, A, x, y);
    else          kb1_gemv<T,N,TR,RM><<<grid(),block()>>>(alpha, A, x, y);
}
template <typename T, int N>
static void dispatch_gemv(T alpha, T* A, T* x, T* y, bool tr, bool rm) {
    if (tr)  { if (rm) launch_gemv<T,N,true ,true >(alpha,A,x,y); else launch_gemv<T,N,true ,false>(alpha,A,x,y); }
    else     { if (rm) launch_gemv<T,N,false,true >(alpha,A,x,y); else launch_gemv<T,N,false,false>(alpha,A,x,y); }
}

template <typename T, int N, FillMode F, Diag D, bool TR>
static void launch_trsv(T* A, T* x) {
    if (g_thread) kt_trsv<T,N,F,D,TR><<<grid(),block()>>>(g_P, A, x);
    else          kb1_trsv<T,N,F,D,TR><<<grid(),block()>>>(A, x);
}
template <typename T, int N, FillMode F, Diag D>
static void dispatch_trsv_t(T* A, T* x, bool tr) {
    if (tr) launch_trsv<T,N,F,D,true>(A,x); else launch_trsv<T,N,F,D,false>(A,x);
}
template <typename T, int N>
static void dispatch_trsv(T* A, T* x, bool lower, bool unit, bool tr) {
    if (lower) {
        if (unit) dispatch_trsv_t<T,N,FillMode::Lower,Diag::Unit   >(A,x,tr);
        else      dispatch_trsv_t<T,N,FillMode::Lower,Diag::NonUnit>(A,x,tr);
    } else {
        if (unit) dispatch_trsv_t<T,N,FillMode::Upper,Diag::Unit   >(A,x,tr);
        else      dispatch_trsv_t<T,N,FillMode::Upper,Diag::NonUnit>(A,x,tr);
    }
}

template <typename T, int N, FillMode F, Diag D, bool TR>
static void launch_trsm(T* A, T* B) {
    if (g_thread) kt_trsm<T,N,F,D,TR><<<grid(),block()>>>(g_P, A, B);
    else          kb1_trsm<T,N,F,D,TR><<<grid(),block()>>>(A, B);
}
template <typename T, int N, FillMode F, Diag D>
static void dispatch_trsm_t(T* A, T* B, bool tr) {
    if (tr) launch_trsm<T,N,F,D,true>(A,B); else launch_trsm<T,N,F,D,false>(A,B);
}
template <typename T, int N>
static void dispatch_trsm(T* A, T* B, bool lower, bool unit, bool tr) {
    if (lower) {
        if (unit) dispatch_trsm_t<T,N,FillMode::Lower,Diag::Unit   >(A,B,tr);
        else      dispatch_trsm_t<T,N,FillMode::Lower,Diag::NonUnit>(A,B,tr);
    } else {
        if (unit) dispatch_trsm_t<T,N,FillMode::Upper,Diag::Unit   >(A,B,tr);
        else      dispatch_trsm_t<T,N,FillMode::Upper,Diag::NonUnit>(A,B,tr);
    }
}

template <typename T, int N, FillMode F, bool TR>
static void launch_syrk(T* A, T* C) {
    if (g_thread) kt_syrk<T,N,F,TR><<<grid(),block()>>>(g_P, A, C);
    else          kb1_syrk<T,N,F,TR><<<grid(),block()>>>(A, C);
}
template <typename T, int N>
static void dispatch_syrk(T* A, T* C, int fill, bool tr) {
    if (fill == 0)      { if (tr) launch_syrk<T,N,FillMode::Lower,true>(A,C); else launch_syrk<T,N,FillMode::Lower,false>(A,C); }
    else if (fill == 1) { if (tr) launch_syrk<T,N,FillMode::Upper,true>(A,C); else launch_syrk<T,N,FillMode::Upper,false>(A,C); }
    else                { if (tr) launch_syrk<T,N,FillMode::Full ,true>(A,C); else launch_syrk<T,N,FillMode::Full ,false>(A,C); }
}
template <typename T, int N, FillMode F, bool TR>
static void launch_syr2k(T* A, T* B, T* C) {
    if (g_thread) kt_syr2k<T,N,F,TR><<<grid(),block()>>>(g_P, A, B, C);
    else          kb1_syr2k<T,N,F,TR><<<grid(),block()>>>(A, B, C);
}
template <typename T, int N>
static void dispatch_syr2k(T* A, T* B, T* C, int fill, bool tr) {
    if (fill == 0)      { if (tr) launch_syr2k<T,N,FillMode::Lower,true>(A,B,C); else launch_syr2k<T,N,FillMode::Lower,false>(A,B,C); }
    else if (fill == 1) { if (tr) launch_syr2k<T,N,FillMode::Upper,true>(A,B,C); else launch_syr2k<T,N,FillMode::Upper,false>(A,B,C); }
    else                { if (tr) launch_syr2k<T,N,FillMode::Full ,true>(A,B,C); else launch_syr2k<T,N,FillMode::Full ,false>(A,B,C); }
}
template <typename T, int N, TensorAxis C, bool SYM, bool ACC>
static void launch_tvc(T* Tns, T* v, T* M) {
    if constexpr (!SYM || C == TensorAxis::K) {     // SYMMETRIC requires CONTRACT==K (square here)
        if (g_thread) kt_tvc<T,N,C,SYM,ACC><<<grid(),block()>>>(g_P, Tns, v, M);
        else          kb1_tvc<T,N,C,SYM,ACC><<<grid(),block()>>>(Tns, v, M);
    }
}
template <typename T, int N, TensorAxis C>
static void dispatch_tvc_sa(T* Tns, T* v, T* M, bool sym, bool acc) {
    if (sym) { if (acc) launch_tvc<T,N,C,true ,true>(Tns,v,M); else launch_tvc<T,N,C,true ,false>(Tns,v,M); }
    else     { if (acc) launch_tvc<T,N,C,false,true>(Tns,v,M); else launch_tvc<T,N,C,false,false>(Tns,v,M); }
}
template <typename T, int N>
static void dispatch_tvc(T* Tns, T* v, T* M, int contract, bool sym, bool acc) {
    if (contract == 0)      dispatch_tvc_sa<T,N,TensorAxis::K>(Tns, v, M, sym, acc);
    else if (contract == 1) dispatch_tvc_sa<T,N,TensorAxis::A>(Tns, v, M, sym, acc);
    else                    dispatch_tvc_sa<T,N,TensorAxis::B>(Tns, v, M, sym, acc);
}
template <typename T, int N>
static void dispatch_vtv(T* Tns, T* u, T* w, T* s, bool acc) {
    if (g_thread) { if (acc) kt_vtv<T,N,true ><<<grid(),block()>>>(g_P, Tns, u, w, s);
                    else     kt_vtv<T,N,false><<<grid(),block()>>>(g_P, Tns, u, w, s); }
    else          { if (acc) kb1_vtv<T,N,true ><<<grid(),block()>>>(Tns, u, w, s);
                    else     kb1_vtv<T,N,false><<<grid(),block()>>>(Tns, u, w, s); }
}
// block1 scratch: one device buffer, 2*N*N elements per problem (covers
// congruence_scratch (N*N), congruence_accum (2*N*N) and riccati (2*N*N)).
template <typename T, int N> static T* b1_scr() { return g_thread ? nullptr : alloc_dev<T>((size_t)g_P * 2*N*N); }

template <typename T, int N>
static void dispatch_congr(T* X, T* M, T* Q, bool acc) {
    T* scr = b1_scr<T,N>();
    if (g_thread) { if (acc) kt_congr<T,N,true ><<<grid(),block()>>>(g_P, X, M, Q);
                    else     kt_congr<T,N,false><<<grid(),block()>>>(g_P, X, M, Q); }
    else          { if (acc) kb1_congr<T,N,true ><<<grid(),block()>>>(X, M, Q, scr);
                    else     kb1_congr<T,N,false><<<grid(),block()>>>(X, M, Q, scr); }
}
template <typename T, int N>
static void dispatch_bilinear(T* X, T* M, T* Y, T* R) {
    T* scr = b1_scr<T,N>();
    if (g_thread) kt_bilinear<T,N><<<grid(),block()>>>(g_P, X, M, Y, R);
    else          kb1_bilinear<T,N><<<grid(),block()>>>(X, M, Y, R, scr);
}
template <typename T, int N>
static void dispatch_caccum(T* G, T* M, T* C, bool acc) {
    T* scr = b1_scr<T,N>();
    if (g_thread) { if (acc) kt_caccum<T,N,true ><<<grid(),block()>>>(g_P, G, M, C);
                    else     kt_caccum<T,N,false><<<grid(),block()>>>(g_P, G, M, C); }
    else          { if (acc) kb1_caccum<T,N,true ><<<grid(),block()>>>(G, M, C, scr);
                    else     kb1_caccum<T,N,false><<<grid(),block()>>>(G, M, C, scr); }
}
template <typename T, int N>
static void dispatch_riccati(T* Pm, T* Am, T* Bm, T* Rm, T* K, bool reg) {
    T* scr = b1_scr<T,N>();
    if (g_thread) { if (reg) kt_riccati<T,N,true ><<<grid(),block()>>>(g_P, Pm, Am, Bm, Rm, K);
                    else     kt_riccati<T,N,false><<<grid(),block()>>>(g_P, Pm, Am, Bm, Rm, K); }
    else          { if (reg) kb1_riccati<T,N,true ><<<grid(),block()>>>(Pm, Am, Bm, Rm, K, scr);
                    else     kb1_riccati<T,N,false><<<grid(),block()>>>(Pm, Am, Bm, Rm, K, scr); }
}

template <typename T, int N>
static void dispatch(const char* op, T* A, T* B, T* x, T* y, T* out)
{
    const T alpha = (T)1;
    if (!strcmp(op, "dot")) {
        if (g_thread) kt_dot<T,N><<<grid(),block()>>>(g_P, x, y, out);
        else          kb1_dot<T,N><<<grid(),block()>>>(x, y, out);
    } else if (!strcmp(op, "gemv")) {
        dispatch_gemv<T,N>(alpha, A, x, y, g_f0, g_f1);
    } else if (!strcmp(op, "gemm")) {
        if (g_thread) kt_gemm<T,N><<<grid(),block()>>>(g_P, alpha, A, B, y);
        else          kb1_gemm<T,N><<<grid(),block()>>>(alpha, A, B, y);
    } else if (!strcmp(op, "potrf")) {
        if (g_thread) kt_potrf<T,N><<<grid(),block()>>>(g_P, A);
        else          kb1_potrf<T,N><<<grid(),block()>>>(A);
    } else if (!strcmp(op, "trsv")) {
        dispatch_trsv<T,N>(A, x, g_f0, g_f1, g_f2);
    } else if (!strcmp(op, "posv")) {
        if (g_thread) kt_posv<T,N><<<grid(),block()>>>(g_P, A, x);
        else          kb1_posv<T,N><<<grid(),block()>>>(A, x);
    } else if (!strcmp(op, "potrs")) {
        if (g_thread) kt_potrs<T,N><<<grid(),block()>>>(g_P, A, x);
        else          kb1_potrs<T,N><<<grid(),block()>>>(A, x);
    } else if (!strcmp(op, "reduce")) {
        if (g_thread) kt_reduce<T,N><<<grid(),block()>>>(g_P, x, out);
        else          kb1_reduce<T,N><<<grid(),block()>>>(x, out);
    } else if (!strcmp(op, "nrm2")) {
        if (g_thread) kt_nrm2<T,N><<<grid(),block()>>>(g_P, x, out);
        else          kb1_nrm2<T,N><<<grid(),block()>>>(x, out);
    } else if (!strcmp(op, "asum")) {
        if (g_thread) kt_asum<T,N><<<grid(),block()>>>(g_P, x, out);
        else          kb1_asum<T,N><<<grid(),block()>>>(x, out);
    } else if (!strcmp(op, "nrm1_diff")) {
        if (g_thread) kt_nrm1_diff<T,N><<<grid(),block()>>>(g_P, x, y, out);
        else          kb1_nrm1_diff<T,N><<<grid(),block()>>>(x, y, out);
    } else if (!strcmp(op, "axpy")) {
        const T a2 = (T)2;                     // alpha != 1 so the scale is exercised
        if (g_thread) kt_axpy<T,N><<<grid(),block()>>>(g_P, a2, x, y);
        else          kb1_axpy<T,N><<<grid(),block()>>>(a2, x, y);
    } else if (!strcmp(op, "scal")) {
        const T a2 = (T)2;
        if (g_thread) kt_scal<T,N><<<grid(),block()>>>(g_P, a2, x);
        else          kb1_scal<T,N><<<grid(),block()>>>(a2, x);
    } else if (!strcmp(op, "copy")) {
        if (g_thread) kt_copy<T,N><<<grid(),block()>>>(g_P, x, y);
        else          kb1_copy<T,N><<<grid(),block()>>>(x, y);
    } else if (!strcmp(op, "rot")) {
        const T rc = (T)0.6, rs = (T)0.8;      // c^2 + s^2 = 1 (exact-arithmetic rotation)
        if (g_thread) kt_rot<T,N><<<grid(),block()>>>(g_P, x, y, out, rc, rs);
        else          kb1_rot<T,N><<<grid(),block()>>>(g_P, x, y, out, rc, rs);
    } else if (!strcmp(op, "symmetrize")) {
        if (g_thread) kt_symmetrize<T,N><<<grid(),block()>>>(g_P, A);
        else          kb1_symmetrize<T,N><<<grid(),block()>>>(A);
    } else if (!strcmp(op, "axpy_strided")) {
        const T a2 = (T)2;
        if (g_thread) kt_axpy_strided<T,N><<<grid(),block()>>>(g_P, a2, x, y);
        else          kb1_axpy_strided<T,N><<<grid(),block()>>>(a2, x, y);
    } else if (!strcmp(op, "copy_strided")) {
        const T a2 = (T)2;
        if (g_thread) kt_copy_strided<T,N><<<grid(),block()>>>(g_P, a2, x, y);
        else          kb1_copy_strided<T,N><<<grid(),block()>>>(a2, x, y);
    } else if (!strcmp(op, "trsm")) {
        dispatch_trsm<T,N>(A, B, g_f0, g_f1, g_f2);
    } else if (!strcmp(op, "ldlt")) {
        if (g_thread) kt_ldlt<T,N><<<grid(),block()>>>(g_P, A);
        else          kb1_ldlt<T,N><<<grid(),block()>>>(A);
    } else if (!strcmp(op, "ldlt_solve")) {
        if (g_thread) kt_ldlt_solve<T,N><<<grid(),block()>>>(g_P, A, x);
        else          kb1_ldlt_solve<T,N><<<grid(),block()>>>(A, x);
    } else if (!strcmp(op, "inv")) {
        if (g_thread) kt_inv<T,N><<<grid(),block()>>>(g_P, A);
        else          kb1_inv<T,N><<<grid(),block()>>>(A);
    } else if (!strcmp(op, "syrk")) {
        dispatch_syrk<T,N>(A, y, g_f0, g_f1);
    } else if (!strcmp(op, "syr2k")) {
        dispatch_syr2k<T,N>(A, B, y, g_f0, g_f1);
    } else if (!strcmp(op, "tvc")) {
        dispatch_tvc<T,N>(A, x, y, g_f0, g_f1, g_f2);
    } else if (!strcmp(op, "vtv")) {
        dispatch_vtv<T,N>(A, x, B, y, g_f0);
    } else if (!strcmp(op, "congr")) {
        dispatch_congr<T,N>(A, B, y, g_f0);
    } else if (!strcmp(op, "bilinear")) {
        dispatch_bilinear<T,N>(A, B, x, y);
    } else if (!strcmp(op, "caccum")) {
        dispatch_caccum<T,N>(A, B, y, g_f0);
    } else if (!strcmp(op, "riccati")) {
        dispatch_riccati<T,N>(A, B, x, y, out, g_f0);
    } else {
        fprintf(stderr, "unknown op %s\n", op); exit(1);
    }
}

template <typename T>
static int run_all(const char* op, char** argv, int f)
{
    const int N = atoi(argv[4]);
    const int mm = g_P * N * N, vv = g_P * N;
    T *A = nullptr, *B = nullptr, *x = nullptr, *y = nullptr, *out = nullptr;

    // Operand files, in the order each op consumes them.
    if (!strcmp(op, "dot")) {
        x = read_dev<T>(argv[f++], vv);
        y = read_dev<T>(argv[f++], vv);
        out = alloc_dev<T>(g_P);
    } else if (!strcmp(op, "gemv")) {
        A = read_dev<T>(argv[f++], mm);
        x = read_dev<T>(argv[f++], vv);
        y = alloc_dev<T>(vv);
    } else if (!strcmp(op, "gemm")) {
        A = read_dev<T>(argv[f++], mm);
        B = read_dev<T>(argv[f++], mm);
        y = alloc_dev<T>(mm);   // C reuses the `y` slot
    } else if (!strcmp(op, "potrf")) {
        A = read_dev<T>(argv[f++], mm);
    } else if (!strcmp(op, "trsv") || !strcmp(op, "posv") || !strcmp(op, "potrs")) {
        A = read_dev<T>(argv[f++], mm);   // trsv: triangular A; posv: SPD A; potrs: lower factor L
        x = read_dev<T>(argv[f++], vv);
    } else if (!strcmp(op, "reduce") || !strcmp(op, "nrm2") || !strcmp(op, "asum")) {
        x = read_dev<T>(argv[f++], vv);
        out = alloc_dev<T>(g_P);              // one scalar per problem
    } else if (!strcmp(op, "nrm1_diff")) {
        x = read_dev<T>(argv[f++], vv);
        y = read_dev<T>(argv[f++], vv);
        out = alloc_dev<T>(g_P);
    } else if (!strcmp(op, "axpy")) {
        x = read_dev<T>(argv[f++], vv);
        y = read_dev<T>(argv[f++], vv);
    } else if (!strcmp(op, "scal")) {
        x = read_dev<T>(argv[f++], vv);
    } else if (!strcmp(op, "copy")) {
        x = read_dev<T>(argv[f++], vv);
        y = alloc_dev<T>(vv);
    } else if (!strcmp(op, "rot")) {
        x = read_dev<T>(argv[f++], vv);
        y = read_dev<T>(argv[f++], vv);
        out = alloc_dev<T>(2 * vv);           // [all x' | all y']
    } else if (!strcmp(op, "symmetrize")) {
        A = read_dev<T>(argv[f++], mm);
    } else if (!strcmp(op, "axpy_strided") || !strcmp(op, "copy_strided")) {
        x = read_dev<T>(argv[f++], g_P * (N+1) * N);   // X at lead N+1
        y = read_dev<T>(argv[f++], g_P * (N+2) * N);   // Y at lead N+2 (pads ride through)
    }
    else if (!strcmp(op, "trsm")) {
        A = read_dev<T>(argv[f++], mm);              // triangular A
        B = read_dev<T>(argv[f++], g_P * N * TRHS);  // N x TRHS right-hand sides
    } else if (!strcmp(op, "ldlt")) {
        A = read_dev<T>(argv[f++], mm);              // symmetric A (in/out)
    } else if (!strcmp(op, "ldlt_solve")) {
        A = read_dev<T>(argv[f++], mm);              // LDLt factor (LD)
        x = read_dev<T>(argv[f++], vv);              // rhs b
    } else if (!strcmp(op, "inv")) {
        A = read_dev<T>(argv[f++], 2 * mm);          // augmented [A | I], N x 2N col-major
    }
    else if (!strcmp(op, "syrk")) {
        A = read_dev<T>(argv[f++], mm);
        y = alloc_dev<T>(mm);                        // C
    } else if (!strcmp(op, "syr2k")) {
        A = read_dev<T>(argv[f++], mm);
        B = read_dev<T>(argv[f++], mm);
        y = alloc_dev<T>(mm);                        // C
    } else if (!strcmp(op, "tvc")) {
        A = read_dev<T>(argv[f++], g_P * N * N * N); // (N,N,N) tensor, col-major slabs
        x = read_dev<T>(argv[f++], vv);              // v
        y = alloc_dev<T>(mm);                        // Mout
    } else if (!strcmp(op, "vtv")) {
        A = read_dev<T>(argv[f++], g_P * N * N * N); // tensor
        x = read_dev<T>(argv[f++], vv);              // u
        B = read_dev<T>(argv[f++], vv);              // w (rides the B slot)
        y = alloc_dev<T>(vv);                        // s
    } else if (!strcmp(op, "congr")) {
        A = read_dev<T>(argv[f++], mm);              // X
        B = read_dev<T>(argv[f++], mm);              // M (symmetric)
        y = alloc_dev<T>(mm);                        // Q
    } else if (!strcmp(op, "bilinear")) {
        A = read_dev<T>(argv[f++], mm);              // X
        B = read_dev<T>(argv[f++], mm);              // M
        x = read_dev<T>(argv[f++], mm);              // Y (rides the x slot)
        y = alloc_dev<T>(mm);                        // R
    } else if (!strcmp(op, "caccum")) {
        A = read_dev<T>(argv[f++], mm);              // G
        B = read_dev<T>(argv[f++], mm);              // M (symmetric)
        y = alloc_dev<T>(mm);                        // C
    } else if (!strcmp(op, "riccati")) {
        A = read_dev<T>(argv[f++], mm);              // P (SPD)
        B = read_dev<T>(argv[f++], mm);              // A
        x = read_dev<T>(argv[f++], mm);              // B (rides the x slot; NX=NU=N)
        y = read_dev<T>(argv[f++], mm);              // R (SPD; rides the y slot)
        out = alloc_dev<T>(mm);                      // Kgain
    }

    switch (N) {
        case 4: dispatch<T,4>(op, A, B, x, y, out); break;
        case 5: dispatch<T,5>(op, A, B, x, y, out); break;
        case 6: dispatch<T,6>(op, A, B, x, y, out); break;
        case 7: dispatch<T,7>(op, A, B, x, y, out); break;
        case 8: dispatch<T,8>(op, A, B, x, y, out); break;
        default: fprintf(stderr, "N=%d not instantiated (want 4..8)\n", N); return 1;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    // Emit the op's result buffer.
    if      (!strcmp(op, "dot"))    print_dev<T>(out, g_P);
    else if (!strcmp(op, "gemv"))   print_dev<T>(y, vv);
    else if (!strcmp(op, "gemm"))   print_dev<T>(y, mm);
    else if (!strcmp(op, "potrf"))  print_dev<T>(A, mm);
    else if (!strcmp(op, "reduce") || !strcmp(op, "nrm2") ||
             !strcmp(op, "asum")   || !strcmp(op, "nrm1_diff"))
                                    print_dev<T>(out, g_P);
    else if (!strcmp(op, "axpy") || !strcmp(op, "copy"))
                                    print_dev<T>(y, vv);
    else if (!strcmp(op, "scal"))   print_dev<T>(x, vv);
    else if (!strcmp(op, "rot"))    print_dev<T>(out, 2 * vv);
    else if (!strcmp(op, "symmetrize")) print_dev<T>(A, mm);
    else if (!strcmp(op, "axpy_strided") || !strcmp(op, "copy_strided"))
                                    print_dev<T>(y, g_P * (N+2) * N);
    else if (!strcmp(op, "trsm"))   print_dev<T>(B, g_P * N * TRHS);
    else if (!strcmp(op, "ldlt"))   print_dev<T>(A, mm);
    else if (!strcmp(op, "inv"))    print_dev<T>(A, 2 * mm);   // full augmented buffer
    else if (!strcmp(op, "vtv"))     print_dev<T>(y, vv);
    else if (!strcmp(op, "riccati")) print_dev<T>(out, mm);
    else if (!strcmp(op, "syrk") || !strcmp(op, "syr2k") || !strcmp(op, "tvc") ||
             !strcmp(op, "congr") || !strcmp(op, "bilinear") || !strcmp(op, "caccum"))
                                     print_dev<T>(y, mm);
    else                            print_dev<T>(x, vv);  // trsv, posv, potrs, ldlt_solve
    cudaDeviceSynchronize();
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 6) { fprintf(stderr, "usage: %s <op> <model> <dtype> <N> <P> [flags...] <files...>\n", argv[0]); return 1; }
    const char* op    = argv[1];
    const char* model = argv[2];
    const char* dtype = argv[3];
    // argv[4] = N (parsed in run_all), argv[5] = P
    g_P = atoi(argv[5]);
    g_thread = !strcmp(model, "thread");
    if (!g_thread && strcmp(model, "block1")) { fprintf(stderr, "model must be thread|block1\n"); return 1; }

    // Op-specific compile-time flags follow P; files follow the flags.
    int f = 6;
    g_f0 = g_f1 = g_f2 = 0;
    if (!strcmp(op, "gemv")) {                       // <trans> <rowmajor>
        g_f0 = atoi(argv[f++]); g_f1 = atoi(argv[f++]);
    } else if (!strcmp(op, "trsv") || !strcmp(op, "trsm")) {  // <lower> <unit> <trans>
        g_f0 = atoi(argv[f++]); g_f1 = atoi(argv[f++]); g_f2 = atoi(argv[f++]);
    }
    else if (!strcmp(op, "syrk") || !strcmp(op, "syr2k")) {   // <fill:0=L,1=U,2=Full> <trans>
        g_f0 = atoi(argv[f++]); g_f1 = atoi(argv[f++]);
    } else if (!strcmp(op, "tvc")) {                          // <contract:0=K,1=A,2=B> <sym> <acc>
        g_f0 = atoi(argv[f++]); g_f1 = atoi(argv[f++]); g_f2 = atoi(argv[f++]);
    } else if (!strcmp(op, "vtv") || !strcmp(op, "congr") || !strcmp(op, "caccum")) {  // <acc>
        g_f0 = atoi(argv[f++]);
    } else if (!strcmp(op, "riccati")) {                      // <reg> (rho fixed at 0.05)
        g_f0 = atoi(argv[f++]);
    }

    if      (!strcmp(dtype, "f32")) return run_all<float >(op, argv, f);
    else if (!strcmp(dtype, "f64")) return run_all<double>(op, argv, f);
    fprintf(stderr, "dtype must be f32|f64\n");
    return 1;
}
