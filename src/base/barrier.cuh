#pragma once
#include <cstdint>

/**
 * @file barrier.cuh
 * @brief Barrier policy for the shared `*_impl` bodies (cgrps dedup + TRAILING_SYNC).
 *
 * Every public single-block op is written ONCE as a `*_impl(Bar bar, ...)` body
 * templated on a *barrier policy* that carries the thread `rank()`/`size()` and a
 * `sync()`. The plain `glass::` surface passes `BlockBarrier` (threadIdx/blockDim
 * + `__syncthreads()`); the `glass::cgrps::` surface passes a `GroupBarrier`
 * (cooperative-groups handle + `cooperative_groups::sync`, defined in
 * `src/cgrps/`). Routing BOTH the internal and the trailing barrier through
 * `bar.sync()` is what lets one body serve both surfaces — that barrier
 * primitive is the only thing that ever differed between them.
 *
 * `BlockBarrier` names no cooperative-groups type, so `glass.cuh` stays
 * dependency-free; the `GroupBarrier` twin is only compiled when a caller
 * includes `<cooperative_groups.h>` via `glass-cgrps.cuh`.
 *
 * UNIFORM TRAILING_SYNC CONTRACT (retrofit completed 2026-08-12): every
 * block-scope public op takes `bool TRAILING_SYNC = true`. At the default the
 * op ends on a barrier, so its result is valid for ALL threads on return —
 * always safe to compose. A caller that owns the following barrier (or feeds
 * a consumer that opens with one) passes `false`:
 *   - where the tail barrier is SEPARABLE (the L1/reduce families, gemm/gemv,
 *     trsv/trsm/trmv, posv/potrs, potrf, inv/inv_dense, syev/eig_clamp,
 *     eigh/psd_project, pcg, bdmv, …) it is genuinely elided;
 *   - where the last barrier is FUSED into the final algorithm step
 *     (ldlt/ldlt_solve, getrf, inv_pivoted — pivoted/multi-exit control flow)
 *     the parameter is accepted but is a documented NO-OP: uniform interface,
 *     no pretended perf win (each such op says so at its declaration).
 * One deliberate exception: the compile-time single-RHS `posv<T,N>`/
 * `potrs<T,N>` carry no third template parameter (it would be ambiguous
 * against the `<T,N,NRHS>` multi-RHS overloads) — the elidable compile-time
 * spelling is the NRHS=1 multi-RHS form.
 * The knob is block-scope only: `warp::`/`thread::` tiers have no block
 * barrier to elide (warp ops are lockstep/shuffle-based; the ThreadBarrier
 * sync is already a no-op).
 */
struct BlockBarrier {
    __device__ __forceinline__ uint32_t rank() const {
        return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    }
    __device__ __forceinline__ uint32_t size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
};

/**
 * @brief Barrier policy for the `glass::thread::`

 this exists as a no-op for single threaded synchronization: if there is only one thread,
 then nothing needs to be synchronized. 
 */
struct ThreadBarrier {
    __device__ __forceinline__ uint32_t rank() const { return 0u; }
    __device__ __forceinline__ uint32_t size() const { return 1u; }
    __device__ __forceinline__ void sync() const { }
};

// Flat block rank/size — THE prologue for every public block-face op. One
// definition instead of the hand-written threadIdx arithmetic at each site
// (233 copies removed 2026-08-11); identical instructions after inlining.
__device__ __forceinline__ uint32_t flat_rank() { return BlockBarrier{}.rank(); }
__device__ __forceinline__ uint32_t flat_size() { return BlockBarrier{}.size(); }

// ─────────────────────────────────────────────────────────────────────────────
// ct_size — compile-time size carrier for the factor/solve `*_impl` bodies.
//
// The shared impls take their dimension through a deduced `SizeT` template
// parameter instead of a hard `uint32_t`. The runtime public overloads pass a
// plain `uint32_t` (unchanged behavior); the compile-time-size overloads pass
// `ct_size<N>{}`, whose constexpr conversion makes every use of the dimension a
// compile-time constant after inlining — so nvcc unrolls the loops and
// strength-reduces the `%`/`/` indexing with NO body duplication (the same
// effect `gemm_impl_ct` gets from its dedicated body). A self-made carrier
// rather than std::integral_constant so no system header lands inside
// `namespace glass` (this file is included inside the namespace by glass.cuh).
// Defined here (not in trsv.cuh, its original home) because barrier.cuh is the
// FIRST header glass.cuh pulls in and the one downstream vendoring already
// carries — every factor/solve user (trsv/inv/syev/ldlt/…) sees it regardless
// of which subset of headers a consumer embeds.
// ─────────────────────────────────────────────────────────────────────────────
template <uint32_t V>
struct ct_size {
    __host__ __device__ constexpr operator uint32_t() const { return V; }
};

// ─────────────────────────────────────────────────────────────────────────────
// beta_blend — BLAS beta==0 semantics for every beta-taking op.
//
// BLAS (and cuBLAS) guarantee that when `beta == 0` the destination is
// WRITE-ONLY: it need not hold a valid value on input. A naive
// `alpha*res + beta*dst` breaks that guarantee — `0 * NaN == NaN`, so cold
// scratch left as NaN by a previous kernel poisons the result even though
// beta is zero (found 2026-07-08: GRiD's generated RNEA passes beta=0 gemvs
// over uninitialized s_vaf smem; diverged rollouts leave NaN on the SM and
// the "zero" blend propagates it). The beta != 0 arm is an EXPLICIT fused
// multiply-add intrinsic, not `acc + beta*dst`: a plain expression leaves the
// FMA-contraction choice to ptxas, which decides per kernel and broke
// warp-vs-block bit-identity (test_syrk_warp); a single FFMA/DFMA is the same
// instruction in every instantiation. The compiler cannot fold the select
// away, because `beta*dst == 0` does not hold for floats. Every beta-form op
// must route its blend through this helper. (Bit-note vs the pre-helper code:
// results are identical at beta ∈ {0, 1} — every consumer's case — and may
// differ in the last ULP at fractional beta, where the old code's contraction
// was unspecified anyway.)
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float beta_blend_fma(float acc, float beta, float dst) {
    return __fmaf_rn(beta, dst, acc);
}
__device__ __forceinline__ double beta_blend_fma(double acc, double beta, double dst) {
    return __fma_rn(beta, dst, acc);
}
template <typename T>
__device__ __forceinline__ T beta_blend_fma(T acc, T beta, T dst) {
    return acc + beta * dst;
}

template <typename T>
__device__ __forceinline__ T beta_blend(T acc, T beta, const T &dst) {
    return (beta != static_cast<T>(0)) ? beta_blend_fma(acc, beta, dst) : acc;
}
