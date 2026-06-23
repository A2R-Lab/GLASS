# GLASS

**GPU Linear Algebra Simple Subroutines.** Composable `__device__` BLAS/LAPACK-style subroutines that run inside a single CUDA block. Now expanding to warp-level primitives for packing many small problems into one block.

## Overview

GLASS functions are `__device__` helpers that operate on data in shared or device memory. Every function assumes it runs within **one CUDA block** — the caller is responsible for launching one block per independent data item. This design enables composable GPU kernels for applications like model-predictive control and rigid-body dynamics.

GLASS started as a small set of hand-rolled SIMT subroutines (`glass::`, `glass::cgrps::`) tuned for very small matrices where the launch and dispatch overhead of a vendor library would dominate the actual work. It has since grown into a **unified single-block linear-algebra surface** that wraps NVIDIA's state-of-the-art device-side libraries — CUB (L1 reductions), cuBLASDx (L2/L3 GEMV/GEMM, including batched), and cuSOLVERDx (LAPACK: Cholesky, LU, QR, triangular and least-squares solves) — under the same `__device__` calling convention. The intent is to give callers one consistent API across the full block-size compute scale: pure-SIMT for tiny matrices where the vendor path can't beat unrolled SIMT, and tensor-core-tuned vendor kernels for everything large enough to benefit from them.

### Call surfaces

GLASS primitives are **block-scoped** by default — one block per problem — and come in three numerically-interchangeable backends, plus a **warp-scoped** surface for kernels that pack many small independent problems into one block (one per warp):

| Namespace | Scope | Backend | Header |
|-----------|-------|---------|--------|
| `glass::` | block | Hand-rolled SIMT, `threadIdx.{x,y,z}` / `blockDim.*` — no cgrps dep | `glass.cuh` |
| `glass::cgrps::` | block | Hand-rolled SIMT, `g.thread_rank()` / `g.size()` — cooperative groups | `glass-cgrps.cuh` |
| `glass::nvidia::` | block | CUB (L1) + cuBLASDx (L2/L3, batched) + cuSOLVERDx (LAPACK) — compile-time sizes only | `glass-nvidia.cuh` |
| `glass::warp::` | **warp** | Single-warp SIMT via `__shfl_*_sync` — *selected* L1/L2/L3 ops, no `__syncthreads`/shared | inline in the base L1/L2/L3 headers (via `glass.cuh`) |

The three **block-scoped** backends provide the full L1/L2/L3 surface and are interchangeable — switch by changing the namespace prefix when profiling shows one is faster at a given size. They preserve the same one-block, `__device__` calling convention, so a single kernel can mix hand-rolled and vendor-backed primitives without leaving the block. `glass::warp::` is a **warp-per-problem** variant covering a selected set (`dot`, `axpy`, `copy`, `scal`, `reduce`, `gemv`, `gemm`, `cholDecomp_InPlace`, `trsv`/`trsm`, and the composed `posv` SPD solve): the warps run independently (no block barrier), turning the block into a vehicle for intra-block parallelism. It requires a full 32-lane warp.

Both `glass::` and `glass::cgrps::` offer **runtime** (size as function arg) and **compile-time** (size as template arg) overloads for every function. Reduction operations additionally offer `glass::low_memory::` (no scratch, thread 0 accumulates) and `glass::high_speed::` (warp-shuffle + shared-memory inter-warp reduction) sub-namespaces.

The dense BLAS/LAPACK surface includes, alongside `gemm`/`gemv`/`ger`: `iamax` (index of max-abs, the pivot primitive); `trsv`/`trmv` (triangular solve / matvec, `LOWER`/`UNIT`/`TRANS` flags); `syrk`/`syr2k` (symmetric rank-k/2k — both `AAᵀ` and `AᵀA` via a `TRANS` flag, writing a `Lower`/`Upper`/`Full` triangle); `inv` and `cholDecomp_InPlace` (single-matrix **and K-way fused** multi-matrix, for packing many small factorizations into one block); `ldlt`/`ldlt_solve` (symmetric-indefinite LDLᵀ for KKT/saddle systems); and `posv`/`potrs` (SPD solve = Cholesky + two triangular solves).

### Higher-level solvers

Built on the primitives above (and likewise single-block) for the block-tridiagonal SPD systems of trajectory optimization / MPC:

| Function | What | Header |
|----------|------|--------|
| `glass::bdmv` | block-tridiagonal matvec (`[L\|D\|R]` strips, padded vectors) | `src/base/banded/bdmv.cuh` |
| `glass::pcg` | single-block preconditioned conjugate gradient (`S x = b`) | `src/base/pcg/solve.cuh` |

An internal box-constrained QP solver, `glass::internal::box_qp` (`src/L3/box_qp.cuh`), also lives in the tree but is **not** part of the public surface — QP is optimization rather than linear algebra; see `docs/open-tasks/qp_solver_scope.md`.

---

## Choosing the right backend

First pick the **execution scope**: one block per problem (the default `glass::` /
`glass::cgrps::` / `glass::nvidia::` surface) or one warp per problem
(`glass::warp::`, when you're packing many tiny independent problems into a block).
### Measured defaults (batched throughput)

From the three-contender scaling sweep (`bench/MEGA_SWEEP_RESULTS.md`, **RTX 5090 /
sm_120** — breakevens shift on other GPUs, so re-run `bench/run_mega_sweep.sh` on yours).

**Most builds don't link MathDx — start here (warp vs block, no dependency):**

| op | default | block `TB` | warp `WPB` | why |
|----|---------|-----------|-----------|-----|
| **dot** | **warp** at every N | 64 | 8–16 | a reduction can't use a block; warp wins 2–6× |
| **gemv** | **warp** ≤ N≈32, **block** ≥ N≈48 | 64–128 | 2–4 | |
| **gemm** | **warp** ≤ N≈8, else **block** | scale 64→256 with N | 2–4 | block's extra threads earn their keep once N≥12 |
| **chol / trsv / posv** | **warp**, block fallback **TB=32** | **32** | 2–4 | extra threads idle on the serial pivot — TB>32 *hurts* |

One rule: **warp-per-problem by default**; gemv → block past N≈48, gemm → block once
it's non-tiny. Pack **2–4 warps/block** (8–16 for dot).

**If you link MathDx** (`glass::nvidia::`), the vendor path wins a *middle band* (f32):
gemm **N≈16–64** (block above — cuBLASDx is smem-capped past 64 here), chol/posv
**N≥16 through 128** (cuSOLVERDx, 1.5–2.7×), trsv only **N≈16–32** (warp wins above —
cuSOLVERDx's triangular solve scales poorly). **f64** is narrower (≈ N=16–64; the double
descriptors hit the smem cap at 64). For a *single* large problem (batch≈1), the vendor
path wins factor/solve/gemm from N≈32 (up to ~8×). Full per-op × per-precision tables:
[`bench/MEGA_SWEEP_RESULTS.md`](bench/MEGA_SWEEP_RESULTS.md).

### Decision procedure

For the block-scoped backends, three questions decide which to call:

1. **Are sizes known at compile time?**
2. **Is matrix size large enough that vendor-tuned tensor-core kernels matter?**
3. **Can you launch with the thread count the backend wants?**

| Scenario | Use | Reason |
|----------|-----|--------|
| Sizes only known at runtime | `glass::gemm(m, n, k, ...)` | Pure-SIMT, accepts dynamic args |
| Compile-time sizes, small matrices (≤ ~8×8), simple kernel | `glass::gemm<float, M, N, K>(...)` | Compiler unrolls inner loops; ~1 µs/op overhead is hard to beat for tiny sizes |
| Compile-time sizes, larger matrices, tensor-core hardware | `glass::nvidia::gemm<float, M, N, K>(...)` | cuBLASDx generates SM-specific tensor-core code |
| Compile-time sizes inside an existing kernel that uses a different thread count (e.g. GRiD's 352-thread launches) | `glass::nvidia::gemm<float, M, N, K, TC>(...)` with `DEFINE_NVIDIA_GEMM_BLOCKDIM(M,N,K,TC)` | Pins cuBLASDx's `BlockDim<TC,1,1>`; lets you launch with any thread count ≥ TC ([P0-1 in the design doc](glass-rfc-batched-1d.md)) |
| Need a transposed B (or row-major A/B/C) in the NVIDIA path | `glass::nvidia::gemm<...,LA,LB,LC>` with `DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT(...)` or `_TRANSB` alias | cuBLASDx Arrangement; pure-SIMT fallback no longer needed |
| Linear solve (`Mx = b` for SPD `M`) | `glass::nvidia::posv<float, N, NRHS>(...)` | cuSOLVERDx fused factor + solve; faster than chol+trsm at N ≥ 8 |
| Cholesky alone (factor only, custom solve) | `glass::nvidia::chol_inplace<float, N>(...)` | cuSOLVERDx potrf |
| General linear solve (non-SPD) | `glass::nvidia::gesv_no_pivot<float, N, NRHS>(...)` | cuSOLVERDx LU + solve |
| Least-squares / over- or under-determined | `glass::nvidia::gels<float, M, N, NRHS>(...)` | cuSOLVERDx QR (or LQ for under-det) + solve |
| `BATCH` independent GEMMs of the same shape, want to amortize launch | `glass::nvidia::gemm_batched<...,BATCH,TC>` | Single block, all batches active via `threadIdx.y` |

**Auto-dispatch.** As of round-2, `glass::nvidia::gemm<>`, `gemv<>`,
`row_strided_*`, and `gemm_batched_1d<>` are **auto-dispatching primary
templates** that route to the pure-SIMT path (`::glass::*`) for shapes
where SIMT wins, and to cuBLASDx via the `DEFINE_NVIDIA_*` macros for
shapes where the vendor library wins. The decision is made at compile
time by `should_use_cublasdx*<T,M,N,K,SM>()` (see `src/nvidia/query.cuh`),
which consults — in order — a per-build local table, the shipped global
table (`src/nvidia/tuning_table.cuh`), and a fallback static heuristic.

This means: calling `glass::nvidia::gemm<float, 6, 6, 6>(...)` "just works"
without any DEFINE macro — small shapes route to SIMT automatically.
Calling `glass::nvidia::gemm<float, 32, 32, 32>(...)` still requires a
`DEFINE_NVIDIA_GEMM(32, 32, 32)` in scope, but produces a clean
compile-time message when missing.

**Heuristic baseline** (when no tuning-table entry matches): for `gemm`,
`max(M,N,K) >= 16 AND min(M,N,K) >= 4` → cuBLASDx; otherwise SIMT. For
`gemv`, `max(M,N) >= 32` → cuBLASDx. For `gemm_batched_1d`,
`BATCH >= 8 AND max(M,N,K) >= 8` → cuBLASDx. The benchmark suite
(`bench/run_bench.py`) measures both side by side on your hardware so
you can override these defaults via the tuning table — see
"Tuning your local build" below.

> **As of P1-4** (the auto-fallback PR), you can always write
> `glass::nvidia::gemm<float, M, N, K>(...)` regardless of size. The primary
> template consults `should_use_cublasdx<T,M,N,K,SM>()` (see [Backend dispatch](#backend-dispatch--when-does-glassnvidiagemm-run-cublasdx-vs-simt)
> below) and silently falls through to `glass::gemm<...>` for shapes the
> heuristic flags as SIMT territory. Only larger shapes still need a
> `DEFINE_NVIDIA_GEMM*` macro.

**When NOT to use `glass::nvidia`**:
- Sizes only known at runtime (the templates require compile-time `M`, `N`, `K`).
- You can't add a `DEFINE_NVIDIA_GEMM*` macro for the size you need (e.g. you want every conceivable `(M, N, K)` triple — the macro instantiation cost grows fast).
- You're stuck on an SM cuBLASDx doesn't tune for (it falls back to a generic config; the pure-SIMT compile-time path is often competitive there).

---

## Backend dispatch — when does `glass::nvidia::gemm` run cuBLASDx vs SIMT?

As of [P1-4](glass-rfc-batched-1d.md), the primary `glass::nvidia::gemm<T,M,N,K,...>` template auto-dispatches:

```
                  caller writes:  glass::nvidia::gemm<float, M, N, K>(...)
                                         │
                                         ▼
                       should_use_cublasdx<float, M, N, K, SMS>()
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
                false                                          true
                  │                                             │
       ────────── ▼ ──────────              ────────── ▼ ──────────
       SIMT fallback:                       Need a DEFINE_NVIDIA_GEMM*
       ::glass::gemm<T,M,N,K>(...)          to specialize for cuBLASDx;
       (no DEFINE needed; no smem)          else: static_assert error.
```

**The decision lives in `src/nvidia/tuning_table.cuh`.** The shipped table covers `sm_86` (Ampere consumer) and `sm_120` (Blackwell-class) for square shapes from 3×3×3 up to 64×64×64. For unmeasured shapes a conservative heuristic kicks in:

```
should_use_cublasdx = (T == float) && (max(M,N,K) >= 16) && (min(M,N,K) >= 4)
```

#### Inspecting at runtime

```cpp
glass::nvidia::print_dispatch<float, 4, 4, 4>();
// → glass::nvidia::gemm<T,4,4,4,SM=860>: SIMT fallback

glass::nvidia::print_dispatch<float, 32, 32, 32>();
// → glass::nvidia::gemm<T,32,32,32,SM=860>: cuBLASDx (needs DEFINE_NVIDIA_GEMM*)
```

#### Overriding the dispatch

| Goal | How |
|------|-----|
| Force cuBLASDx for a shape the heuristic puts in SIMT | Add `DEFINE_NVIDIA_GEMM(M,N,K)` in your `.cu` file. The explicit specialization always overrides the primary template. |
| Force SIMT for a shape the heuristic puts in cuBLASDx | Call `::glass::gemm<T,M,N,K>(...)` directly (skip the `nvidia::` path). |
| Per-host tuning without editing source | Run `python bench/autotune.py` to generate `bench/tuning/<hostname>.cuh`, then compile with `-DGLASS_TUNING_TABLE_LOCAL='"bench/tuning/<hostname>.cuh"'`. See `bench/TUNING.md`. |
| Different SM in-tree (for a PR) | `python bench/autotune.py --in-tree` rewrites the marker-delimited specializations section inside `src/nvidia/tuning_table.cuh`, preserving the round-2 primaries above it. |

#### Auto-tune for your hardware

The shipped values are sensible defaults but small-GEMM perf is highly SM-dependent. The autotune covers all five round-2 primaries (`gemm`, `gemv`, `row_strided_gemv`, `row_strided_gemm`, `gemm_batched_1d`).

```bash
# Detect SM, measure all default shapes for all 5 APIs, write per-host
# overrides to bench/tuning/<hostname>.cuh. Does NOT modify src/.
python bench/autotune.py --sm AUTO

# Restrict to one API + custom shapes; --dry-run reports without writing
python bench/autotune.py --apis gemv --shapes '6,6;14,14;32,32' --iters 20000 --dry-run

# Update the in-tree shipped table (for upstream PRs only)
python bench/autotune.py --sm AUTO --in-tree
```

The script compiles and runs a small SIMT-vs-cuBLASDx microbench per (API, shape), times each leg over `--iters` iterations, and emits one `template <> constexpr bool ...` specialization per measured combination plus a human-readable `*_results.md` alongside. Ties (within `--margin`, default ±5 %) default to SIMT. Requires `MATHDX_ROOT` set. Per-host outputs under `bench/tuning/` are gitignored.

See `bench/TUNING.md` for the full contributor workflow including how the override mechanism layers on top of the in-tree table.

---

## Trailing synchronization (`TRAILING_SYNC`)

GLASS L3 functions (`gemm`, `gemv`, `row_strided_*`, `gemm_batched_1d`,
`gemm_strided_batched_1d`, ...) take a final boolean template parameter
**`TRAILING_SYNC` that defaults to `true`** and controls whether the function
emits a `__syncthreads()` before returning:

```cpp
// Default — function returns with all threads at a block-wide barrier.
// Safe to read the result from any thread in the block immediately after.
glass::nvidia::gemm_strided_batched_1d<float, 4, 4, 4, BATCH, TC>(
    1.f, A, B, 0.f, C);

// Opt-out — caller is responsible for syncing before reading any output
// not written by the current thread. Pass false when fusing the GEMM with
// subsequent block-wide work that ALREADY does its own barrier (e.g. a
// `parallel_loop` that begins with `__syncthreads()`), so two back-to-back
// syncs collapse to one.
glass::nvidia::gemm_strided_batched_1d<
    float, 4, 4, 4, BATCH, TC,
    /*B_STRIDE=*/N*K, /*C_STRIDE=*/M*K,
    layout::col_major, layout::col_major, layout::col_major,
    /*TRAILING_SYNC=*/false>(1.f, A, B, 0.f, C);
__syncthreads();  // emit your own here, fused with any other barrier
```

The default of `true` makes the common case correct without thinking —
GLASS functions act as if they were standalone kernels. The opt-out exists
for hot kernels (e.g. GRiD's `end_effector_pose_gradient_inner`) that chain
a GEMM with a SIMT `parallel_loop` and want to collapse the two syncs.

Where it applies (full surface as of 2026-05):

- **L1** (`l1.cuh`): `reduce`, `dot`, `l2norm`. The leading sync that
  protects CUB BlockReduce TempStorage from prior writes is **not**
  gated — it's required for correctness. Only the trailing barrier
  (after the thread-0 write of the reduction result) is.
- **L2** (`l2.cuh`): `gemv`, `row_strided_gemv` — primary templates and
  every `_GLASS_GEMV_NO_BD` / `_GLASS_GEMV_BD` macro-emitted
  specialization. The cuBLASDx-backed specializations emit BOTH
  `TRAILING_SYNC=true` and `=false` variants from a single
  `DEFINE_NVIDIA_GEMV*` invocation.
- **L3** (`l3.cuh`): `gemm`, `gemm_batched`, `row_strided_gemm` — same
  dual-specialization pattern via `_GLASS_GEMM_NO_BD` /
  `_GLASS_GEMM_BD` / `_GLASS_GEMM_BATCHED_BD`.
- **L3 SIMT** (`l3_simt.cuh`): `gemm_batched_1d`,
  `gemm_strided_batched_1d`.
- **LAPACK** (`lapack.cuh`): `posv`, `chol_inplace`, `gesv_no_pivot`,
  `gels`, `trsm` — primary templates and macro-emitted
  specializations. Interior factor / solve / writeback syncs are NOT
  gated; only the final barrier before return is.

The interior `__syncthreads()` inside `gemm` / `gemm_batched` / LAPACK
factor-solve flows (between phases) is required for correctness and is
**not** gated on `TRAILING_SYNC` — only the final trailing barrier is.

Tests for the surface live at
`test/cuda/test_trailing_sync.cu` + `test/test_trailing_sync.py`. They
verify (a) both `TRAILING_SYNC=true` and `=false` specializations
compile + link, and (b) the two variants produce numerically identical
output when the `=false` caller emits its own trailing
`__syncthreads()`. The cuBLASDx-backed L3 case is covered (gated on
`GLASS_BENCH_CUBLASDX` / `MATHDX_ROOT`).

---

## Quick Start

```cpp
#include "glass.cuh"

__global__ void my_kernel(float* A, float* B, float* C, int m, int n, int k) {
    // Runtime size: all threads in the block cooperate
    glass::gemm(m, n, k, 1.f, A, B, 0.f, C);
}
```

Launch with one block per data item:
```cpp
my_kernel<<<num_items, 256>>>(A, B, C, m, n, k);
```

---

## Usage Examples

### 1. Runtime sizes — default `glass::` (threadIdx-based)

```cpp
#include "glass.cuh"

__global__ void k(float* A, float* B, float* C, int m, int n, int k) {
    glass::gemm(m, n, k, 1.f, A, B, 0.f, C);         // GEMM
    glass::gemv(m, n, 1.f, A, B, 0.f, C);             // GEMV
    glass::axpy(n, 1.5f, A, B);                        // y = alpha*x + y
}
```

### 2. Compile-time sizes — loop unrolling via template args

```cpp
#include "glass.cuh"

__global__ void k(float* A, float* B, float* C) {
    // Sizes baked in as template params — compiler can unroll loops
    glass::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);
    glass::gemv<float, 6, 6>(1.f, A, B, 0.f, C);
    glass::axpy<float, 36>(1.5f, A, B);
}
```

### 3. Cooperative groups — sub-block tiling with `glass::cgrps::`

```cpp
#include "glass-cgrps.cuh"
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

__global__ void k(float* A, float* B, float* C) {
    // Whole block (default)
    glass::cgrps::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);

    // Warp-level tiling: each warp independently computes a 4×4×4 GEMM
    auto warp = cgrps::tiled_partition<32>(cgrps::this_thread_block());
    glass::cgrps::gemm<float, 4, 4, 4>(1.f, A, B, 0.f, C, warp);
}
```

### 4. High-speed reductions (warp-shuffle)

```cpp
#include "glass.cuh"

__global__ void k(float* x, int n) {
    // Scratch: ceil(blockDim.x / 32) * sizeof(float) bytes
    extern __shared__ float scratch[];
    glass::high_speed::reduce(n, x, scratch);     // result in x[0]
    glass::high_speed::l2norm(n, x, scratch);     // ‖x‖₂ in x[0]
}
```

### 5. NVIDIA-optimized path (`glass::nvidia::`)

#### 5a. Default — let cuBLASDx pick the thread count

```cpp
#include "glass-nvidia.cuh"

// Host: query required smem and thread count (both constexpr)
constexpr auto smem    = glass::nvidia::gemm_smem_size<float, 6, 6, 6>();
constexpr auto threads = glass::nvidia::gemm_threads<float, 6, 6, 6>();

__global__ void k(float* A, float* B, float* C) {
    extern __shared__ __align__(16) char smem_buf[];
    glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, smem_buf);
}

// Launch with the EXACT thread count cuBLASDx wants — mismatch deadlocks.
k<<<1, threads, smem>>>(dA, dB, dC);
```

#### 5b. Caller-pinned `BlockDim<TC>` — launch with any TC ≥ what cuBLASDx needs

The default form forces every kernel that touches `glass::nvidia::gemm` to launch with the exact thread count cuBLASDx picked for that GEMM size. When the rest of your kernel needs a different launch (e.g. GRiD codegen launches with 352 threads for the surrounding RNEA/CRBA work), use the BlockDim form:

```cpp
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM_BLOCKDIM(6, 6, 6, 352)   // pin BlockDim<352,1,1>
}}

__global__ void k(float* A, float* B, float* C) {
    extern __shared__ __align__(16) char smem_buf[];
    // OK to launch with 352 threads — extras go idle inside the GEMM.
    glass::nvidia::gemm<float, 6, 6, 6, 352>(1.f, A, B, 0.f, C, smem_buf);
}

constexpr auto smem = glass::nvidia::gemm_smem_size<float, 6, 6, 6, 352>();
k<<<1, 352, smem>>>(dA, dB, dC);
```

The query API tells you the smallest TC cuBLASDx will accept for a `(T, M, N, K, SM)` tuple:

```cpp
static_assert(glass::nvidia::gemm_block_threads_valid<float, 6, 6, 6, 352>(),
              "352 threads should be enough for 6x6x6 on this SM");

constexpr uint32_t MIN = glass::nvidia::gemm_min_block_threads<float, 6, 6, 6>();
// MIN is the natural BlockDim cuBLASDx picks; pin TC >= MIN.
```

#### 5c. Layout / transpose (NVIDIA path)

`glass::nvidia::gemm` accepts `layout LA`, `LB`, `LC` template parameters — these mirror cuBLASDx's `Arrangement<>` and let you express transpose / row-major storage without falling back to the pure-SIMT path:

```cpp
namespace glass { namespace nvidia {
    // A · Bᵀ  (B is row-major, equivalent to "transposed col-major")
    DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB(6, 6, 6, 352)        // alias for LB=row_major
    // Or fully explicit (LA=row, LB=col, LC=col):
    DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT(6, 6, 6, 352, 1, 0, 0)
}}

__global__ void k(float* A, float* B, float* C) {
    extern __shared__ __align__(16) char smem_buf[];
    using L = glass::nvidia::layout;
    glass::nvidia::gemm<float, 6, 6, 6, 352,
                        L::col_major, L::row_major, L::col_major>(
        1.f, A, B, 0.f, C, smem_buf);
}
```

#### 5d. Multi-arch builds (per-SM dispatch)

Pin the SM at the call site so a single fatbinary can ship tuned code for multiple architectures:

```cpp
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM_BLOCKDIM_SM(6, 6, 6, 352, 890)
    DEFINE_NVIDIA_GEMM_BLOCKDIM_SM(6, 6, 6, 352, 1200)
}}

__global__ void k(float* A, float* B, float* C) {
    extern __shared__ __align__(16) char smem_buf[];
    using L = glass::nvidia::layout;
    glass::nvidia::gemm<float, 6, 6, 6, 352,
        L::col_major, L::col_major, L::col_major,
        #if __CUDA_ARCH__ >= 1200
            1200
        #else
            890
        #endif
    >(1.f, A, B, 0.f, C, smem_buf);
}
```

#### 5e. Linear solvers (cuSOLVERDx — `chol_inplace`, `trsm`, `posv`, …)

```cpp
#include "glass-nvidia.cuh"

namespace glass { namespace nvidia {
    DEFINE_NVIDIA_POSV_BLOCKDIM(7, 1, 256)   // 7×7 SPD, 1 RHS, BlockDim<256>
}}

__global__ void k(float* A, float* b) {
    extern __shared__ __align__(16) char smem_buf[];
    // Solves A·x = b in place: A := L (lower Cholesky), b := x
    glass::nvidia::posv<float, 7, 1, 256>(A, b, smem_buf);
}

constexpr auto smem = glass::nvidia::posv_smem_size<float, 7, 1, 256>();
k<<<1, 256, smem>>>(dA, db);
```

Available cuSOLVERDx wrappers (all follow the same `DEFINE_NVIDIA_<NAME>` / `_BLOCKDIM` / `_SM` / `_BLOCKDIM_SM` macro pattern):

| Function | Signature | NumPy / SciPy equivalent |
|----------|-----------|--------------------------|
| `chol_inplace<T, N>` | `(A, smem)` | `np.linalg.cholesky(A)` (lower) |
| `trsm<T, M, N>` | `(alpha, L, B, smem)` | `scipy.linalg.solve_triangular(L, alpha*B, lower=True)` |
| `posv<T, N, NRHS>` | `(A, B, smem)` | `np.linalg.solve(A, B)` (SPD A; A is destroyed) |
| `potrs<T, N, NRHS>` | `(L, B, smem)` | `scipy.linalg.cho_solve((L, True), B)` |
| `getrf_no_pivot<T, N>` | `(A, smem)` | `scipy.linalg.lu_factor(A)` (no pivoting; A := LU) |
| `getrs_no_pivot<T, N, NRHS>` | `(LU, B, smem)` | `scipy.linalg.lu_solve((LU, ...), B)` |
| `gesv_no_pivot<T, N, NRHS>` | `(A, B, smem)` | `np.linalg.solve(A, B)` (general A; A is destroyed) |
| `geqrf<T, M, N>` | `(A, tau, smem)` | `scipy.linalg.qr(A, mode='raw')` |
| `gels<T, M, N, NRHS>` | `(A, tau, B, smem)` | `np.linalg.lstsq(A, B)` |

Linking note: cuSOLVERDx ships a precompiled device library. Add `-rdc=true -dlto -L$MATHDX_ROOT/lib -lcusolverdx -lcublas -lcusolver -lcudart` to your nvcc command (the `bench/run_bench.py` driver does this automatically when `cusolverdx.hpp` is present).

#### 5f. Batched GEMM (single block, BATCH GEMMs in parallel)

```cpp
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(6, 6, 6, /*BATCH=*/16, /*TC=*/64)
}}

__global__ void k(float* const* A, float* const* B, float* const* C) {
    extern __shared__ __align__(16) char smem_buf[];
    glass::nvidia::gemm_batched<float, 6, 6, 6, 16, 64>(
        1.f, A, B, 0.f, C, smem_buf);
}

constexpr auto smem = glass::nvidia::gemm_batched_smem_size<float, 6, 6, 6, 16, 64>();
k<<<1, dim3(64, 16), smem>>>(dA_ptrs, dB_ptrs, dC_ptrs);
```

Each batch element is identified by `threadIdx.y`; each per-batch GEMM uses TC threads in `threadIdx.x`. Total launch is `dim3(TC, BATCH)`. See `bench/bench_gemm_batched.cu` for a full working example with `T**` pointer arrays.

#### 5g. Batched GEMM with **1D launch** (`gemm_batched_1d`)

If the surrounding kernel was launched 1D (e.g. `dim3(TC*BATCH, 1, 1)` because every other block-level helper uses `threadIdx.x`), the cuBLASDx-backed `gemm_batched` above won't work — it requires the 2D launch. The SIMT-only `gemm_batched_1d` (and the shared-A variant `gemm_strided_batched_1d`) partition a single 1D block of `TC*BATCH` threads into BATCH groups of TC threads each:

```cpp
__global__ void k(float* const* A, float* const* B, float* const* C) {
    // No DEFINE macro needed — fully templated on T.
    glass::nvidia::gemm_batched_1d<float, 4, 4, 4, /*BATCH=*/8, /*TC=*/32>(
        1.f, A, B, 0.f, C);
}
k<<<1, dim3(32 * 8, 1, 1)>>>(dA_ptrs, dB_ptrs, dC_ptrs);   // no smem

// Shared-A variant: one A applied to BATCH packed (B,C) pairs.
__global__ void k_shared(float* A_shared, float* B_base, float* C_base) {
    glass::nvidia::gemm_strided_batched_1d<float, 4, 4, 4, 8, 32>(
        1.f, A_shared, B_base, 0.f, C_base);   // tightly packed: B_STRIDE=N*K, C_STRIDE=M*K
}
```

These run pure SIMT; they need no shared memory and no `DEFINE_NVIDIA_GEMM_BATCHED_*` macro. Best for the small shapes (`max(M,N,K) ≲ 8`) where cuBLASDx's tile-load overhead dominates anyway. See `bench/bench_gemm_batched_1d.cu`.

### 6. Tiled GEMM (scratch in shared memory)

```cpp
#include "glass.cuh"

__global__ void k(float* A, float* B, float* C, int m, int n, int k) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_B = smem + m * 8;
    glass::gemm_tiled<float, 8>(m, n, k, 1.f, A, B, 0.f, C, s_A, s_B);
}

// Host:
size_t smem = (m * 8 + 8 * k) * sizeof(float);
k<<<1, 256, smem>>>(A, B, C, m, n, k);
```

---

## Requirements

| Component | C++ Standard | Optional deps |
|-----------|-------------|---------------|
| `glass.cuh`, `glass-cgrps.cuh` | C++17 (`nvcc -std=c++17`) | — |
| `glass-nvidia.cuh` (L1 only) | C++17 | CUB (bundled with CUDA 11+) |
| `glass-nvidia.cuh` (L2/L3 GEMM/GEMV/batched) | C++17 + `--expt-relaxed-constexpr` | cuBLASDx |
| `glass-nvidia.cuh` (LAPACK: chol/trsm/posv/getrf/gesv/geqrf/gels) | C++17 + `--expt-relaxed-constexpr` + `-rdc=true -dlto -lcusolverdx -lcublas -lcusolver -lcudart` | cuSOLVERDx |
| Test suite | C++17 | — |
| Benchmark suite | C++17 | (matches the variants you opt in) |

The wrappers gate themselves with `GLASS_HAVE_CUBLASDX` / `GLASS_HAVE_CUSOLVERDX`, both auto-detected from include order. To force-enable when including `glass-nvidia.cuh` from a TU that hasn't pre-included the headers, define `GLASS_BENCH_CUBLASDX` and/or `GLASS_BENCH_CUSOLVERDX` (the bench harness sets these). See [`bench/INSTALL.md`](bench/INSTALL.md) for download + linking instructions.

---

## API Reference

### L1 — Vector Operations

All L1 functions accept `uint32_t n` (runtime) **or** `<T, N>` template args (compile-time).

| Function | Description | NumPy equivalent | Scratch |
|----------|-------------|------------------|---------|
| `axpy(n, alpha, x, y)` | `y = alpha*x + y` | `y += alpha*x` | none |
| `axpy(n, alpha, x, y, z)` | `z = alpha*x + y` | `z = alpha*x + y` | none |
| `axpby(n, alpha, x, beta, y, z)` | `z = alpha*x + beta*y` | `z = alpha*x + beta*y` | none |
| `copy(n, x, y)` | `y = x` | `y = x.copy()` | none |
| `copy(n, alpha, x, y)` | `y = alpha*x` | `y = alpha*x` | none |
| `scal(n, alpha, x)` | `x = alpha*x` (in-place) | `x *= alpha` | none |
| `scal(n, alpha, x, y)` | `y = alpha*x` | `y = alpha*x` | none |
| `swap(n, x, y)` | swap `x` and `y` | — | none |
| `dot(n, x, y)` | `y[0] = x·y` (in-place, uses `y` as scratch) | `np.dot(x,y)` | none |
| `iamax(n, x, out)` | `out[0] = argmax\|xᵢ\|` (index of max-abs; pivot primitive) | `np.argmax(np.abs(x))` | none |
| `reduce(n, x)` | `x[0] = sum(x)` (in-place) | `np.sum(x)` | none |
| `l2norm(n, x)` | `x[0] = ‖x‖₂` (in-place, destructive) | `np.linalg.norm(x)` | none |
| `infnorm(n, x)` | `x[0] = ‖x‖∞` (in-place) | `np.max(np.abs(x))` | none |
| `asum(n, x, out)` | `out[0] = Σ|xᵢ|` | `np.sum(np.abs(x))` | `n*sizeof(T)` |
| `clip(n, x, l, u)` | `x = clamp(x, l, u)` | `np.clip(x, l, u)` | none |
| `set_const(n, alpha, x)` | `x = [alpha, …]` | `np.full(n, alpha)` | none |
| `loadIdentity(n, A)` | `A = I_n` (column-major) | `np.eye(n)` | none |
| `addI(n, A, alpha)` | `A += alpha*I` | `A += alpha*np.eye(n)` | none |
| `transpose(N, M, a, b)` | `b = aᵀ` (col-major) | `b = a.T` | none |
| `transpose(N, a)` | in-place transpose `N×N` | — | none |
| `elementwise_add(N, a, b, c)` | `c = a + b` | `c = a + b` | none |
| `elementwise_sub(N, a, b, c)` | `c = a - b` | `c = a - b` | none |
| `elementwise_mult(N, a, b, c)` | `c = a ⊙ b` | `c = a * b` | none |
| `elementwise_abs(N, a, b)` | `b = |a|` | `b = np.abs(a)` | none |
| `elementwise_max(N, a, b, c)` | `c = max(a,b)` | `np.maximum(a,b)` | none |
| `elementwise_min(N, a, b, c)` | `c = min(a,b)` | `np.minimum(a,b)` | none |
| `prefix_sum_exclusive(x, out, n)` | exclusive scan | — | none |
| `prefix_sum_inclusive(x, out, n)` | inclusive scan | `np.cumsum(x)` | none |

#### Reduction sub-namespaces

```cpp
glass::reduce(n, x);                         // default: halving reduce
glass::low_memory::reduce(n, x);             // thread 0 accumulates; no scratch
glass::high_speed::reduce(n, x, scratch);    // warp-shuffle; faster for large n
```

Scratch required for `high_speed`: `ceil(blockDim.x / 32) * sizeof(T)` bytes.

#### `glass::nvidia::` L1 (CUB BlockReduce)

```cpp
#include "glass-nvidia.cuh"
extern __shared__ float scratch[];  // sizeof(cub::BlockReduce<T,THREADS>::TempStorage)

glass::nvidia::reduce<float, N, THREADS>(x, scratch);
glass::nvidia::dot<float, N, THREADS>(x, y, out, scratch);
glass::nvidia::l2norm<float, N, THREADS>(x, out, scratch);

// Query scratch size (host-callable):
constexpr auto bytes = glass::nvidia::reduce_smem_size<float, THREADS>();
```

---

### L2 — Matrix-Vector Operations

Matrices default to **column-major** order. Pass `ROW_MAJOR=true` to use row-major (C-style).

| Function | Description | NumPy equivalent |
|----------|-------------|------------------|
| `gemv<T>(m, n, alpha, A, x, beta, y)` | `y = alpha*A*x + beta*y` (col-major A) | `y = alpha*A@x + beta*y` |
| `gemv<T,true>(m, n, alpha, A, x, beta, y)` | `y = alpha*Aᵀ*x + beta*y` | `y = alpha*A.T@x + beta*y` |
| `gemv<T,false,true>(m, n, alpha, A, x, beta, y)` | same, but A is row-major | `y = alpha*A@x + beta*y` |
| `gemv_ex<T,TRANSPOSE,ROW_MAJOR_A>(...)` | per-matrix layout control | |
| `ger(m, n, alpha, x, y, A)` | `A += alpha*x*yᵀ` | `A += alpha*np.outer(x,y)` |
| `trsv<T,LOWER,UNIT,TRANS>(n, A, x)` | triangular solve `A*x = b` in-place (`LOWER`/`UNIT`/`TRANS` flags) | `scipy.linalg.solve_triangular(A,x,lower=...)` |
| `trmv<T,LOWER,UNIT,TRANS>(n, A, x)` | triangular matvec `x = A*x` (`LOWER`/`UNIT`/`TRANS` flags) | `A @ x` (triangular A) |

Compile-time overloads omit the `m, n` args:
```cpp
glass::gemv<float, 6, 6>(1.f, A, x, 0.f, y);           // runtime: glass::gemv(6, 6, 1.f, A, x, 0.f, y)
glass::cgrps::gemv<float, 6, 6>(1.f, A, x, 0.f, y, g); // with explicit group
```

---

### L3 — Matrix Operations

Matrices default to **column-major** order.

| Function | Description | NumPy equivalent | Scratch |
|----------|-------------|------------------|---------|
| `gemm<T>(m,n,k, alpha, A, B, beta, C)` | `C = alpha*A*B + beta*C` (col-major) | `C = alpha*A@B + beta*C` | none |
| `gemm<T,true>(m,n,k, alpha, A, B, beta, C)` | `C = alpha*A*Bᵀ + beta*C` | — | none |
| `gemm<T,false,true>(m,n,k, alpha, A, B, beta, C)` | same, all matrices row-major | | none |
| `gemm_ex<T,TRANSPOSE_B,ROW_A,ROW_B,ROW_C>(...)` | per-matrix layout control | | none |
| `gemm_tiled<T,TILE>(m,n,k, alpha, A, B, beta, C, s_A, s_B)` | tiled gemm using shared memory | | `(m*TILE + TILE*k)*sizeof(T)` |
| `gemm_dispatch<T>(m,n,k, alpha, A, B, beta, C, s_A, s_B)` | auto-selects tiled or plain | | see below |
| `invertMatrix(n, A, s_temp)` | `A = A⁻¹` in-place (Gauss-Jordan) | `np.linalg.inv(A)` | `(2n+1)*sizeof(T)` |
| `invertMatrix_pivoted(n, A, s_temp)` | `A = A⁻¹` in-place, partial (row) pivoting (robust on small/zero leading pivots) | `np.linalg.inv(A)` | `(3n+1)*sizeof(T)` |
| `invertMatrix(K, ns, As, s_temp)` | K-way fused: invert K independent matrices interleaved over one block | `[np.linalg.inv(A) for A in As]` | per-matrix `(2n+1)` |
| `cholDecomp_InPlace(n, A)` | Cholesky `A → L` (lower triangular) | `np.linalg.cholesky(A)` | none |
| `cholDecomp_InPlace(K, ns, As)` | K-way fused: factor K independent matrices interleaved over one block | `[np.linalg.cholesky(A) for A in As]` | none |
| `syrk<T,TRANS,FILL>(n, k, alpha, A, beta, C)` | symmetric rank-k: `C = alpha*A*Aᵀ + beta*C` (or `AᵀA` with `TRANS`); `FILL` Lower/Upper/Full | `C = alpha*A@A.T + beta*C` | none |
| `syr2k<T,TRANS,FILL>(n, k, alpha, A, B, beta, C)` | symmetric rank-2k: `C = alpha*(A*Bᵀ + B*Aᵀ) + beta*C` (or `AᵀB+BᵀA`) | `C = alpha*(A@B.T+B@A.T) + beta*C` | none |
| `ldlt(n, A)` | symmetric-indefinite `A → L·D·Lᵀ` in-place (no sqrt; KKT/saddle systems) | `scipy.linalg.ldl(A)` | `(n+1)*sizeof(T)` |
| `ldlt_solve(n, A, b)` | solve `A*x = b` from a prior `ldlt` factorization | — | `(n+1)*sizeof(T)` |
| `posv(n, A, b)` | SPD solve `A*x = b` in-place (Cholesky + two triangular solves); multi-RHS `(n, nrhs, A, B)` | `np.linalg.solve(A, b)` (SPD A) | none |
| `potrs(n, L, b)` | solve `A*x = b` from a prior Cholesky `L`; multi-RHS `(n, nrhs, L, B)` | `scipy.linalg.cho_solve((L,True), b)` | none |
| `trsm(n, L, b)` | Solve `Lx=b` in-place (forward substitution) | `scipy.linalg.solve_triangular(L,b,lower=True)` | none |

Compile-time overloads omit the `m, n, k` args:
```cpp
glass::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);
glass::cgrps::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, g);
```

#### Auto-dispatch: `gemm_dispatch`

`glass::gemm_dispatch` selects tiled or plain gemm at runtime based on whether scratch pointers are provided and `m*k ≤ blockDim`:

```cpp
extern __shared__ float scratch[];
float *s_A = scratch, *s_B = scratch + m * 8;
glass::gemm_dispatch(m, n, k, 1.f, A, B, 0.f, C,
                     (smem > 0) ? s_A : nullptr,
                     (smem > 0) ? s_B : nullptr);
```

Use the host helper to compute required scratch at launch:
```cpp
#include "glass.cuh"
std::size_t smem = glass_gemm_dispatch_smem(m, k, /*block_threads=*/256);
my_kernel<<<grid, 256, smem>>>(m, n, k, A, B, C);
```

#### Row-Major Storage Order

```cpp
// Column-major (default): A[row + col*m]
glass::gemm<float>(m, n, k, 1.f, A, B, 0.f, C);

// Row-major (all matrices): A[row*cols + col]
glass::gemm<float, false, true>(m, n, k, 1.f, A, B, 0.f, C);

// Mixed layouts (row-major A and C, column-major B):
glass::gemm_ex<float, false, true, false, true>(m, n, k, 1.f, A, B, 0.f, C);
```

---

### `glass::nvidia::` L2/L3 (cuBLASDx + cuSOLVERDx)

cuBLASDx computes at warp/tensor-core level and requires compile-time matrix sizes. cuSOLVERDx (factorizations + solves) likewise requires compile-time sizes and additionally requires linking against a precompiled device library. Include `glass-nvidia.cuh` to get pre-instantiated sizes; both backends are auto-detected.

**Pre-instantiated GEMM/GEMV sizes** (square): `4, 6, 8, 12, 14, 24, 64`. cuSOLVERDx wrappers are not pre-instantiated; you must call the appropriate `DEFINE_NVIDIA_*` macro per size in your `.cu` file (inside `namespace glass::nvidia`).

#### Public template parameters

```cpp
// Common parameter pack for gemm / gemv (and gemm_batched, with extra BATCH):
template <typename T,
          uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS = 0,                      // 0 = let cuBLASDx pick
          layout LA = layout::col_major,                   // arrangement of A
          layout LB = layout::col_major,                   // arrangement of B
          layout LC = layout::col_major,                   // arrangement of C
          uint32_t SM_VAL = SMS>                            // SM arch (default = SMS macro)
__device__ void gemm(T alpha, T* A, T* B, T beta, T* C, char* smem);
```

`SMS` defaults to `860` and may be overridden with `-DSMS=XXX` at compile time. See sections 5b–5d above for usage examples (BlockDim, layout, multi-arch).

#### DEFINE-macro family (cuBLASDx wrappers)

Every wrapper exposes the same family of compile-time DEFINE macros. Substitute `GEMM` / `GEMV` / `GEMM_BATCHED` / `CHOL` / `TRSM` / `POSV` / `POTRS` / `GETRF` / `GETRS` / `GESV` / `GEQRF` / `GELS`:

| Macro form | Specializes |
|------------|------------|
| `DEFINE_NVIDIA_<NAME>(...)` | default block_dim, all `col_major`, SM = `SMS` |
| `DEFINE_NVIDIA_<NAME>_BLOCKDIM(..., TC)` | pinned `BlockDim<TC,1,1>`, all `col_major`, SM = `SMS` |
| `DEFINE_NVIDIA_<NAME>_LAYOUT(..., LA, LB, LC)` *(GEMM only)* | default block_dim, custom layouts, SM = `SMS` |
| `DEFINE_NVIDIA_<NAME>_BLOCKDIM_LAYOUT(..., TC, LA, LB, LC)` | pinned + custom layouts |
| `DEFINE_NVIDIA_<NAME>_SM(..., SM)` | explicit SM, default block_dim, all `col_major` |
| `DEFINE_NVIDIA_<NAME>_BLOCKDIM_SM(..., TC, SM)` | pinned + explicit SM |
| `DEFINE_NVIDIA_<NAME>_LAYOUT_SM(..., LA, LB, LC, SM)` *(GEMM only)* | custom layouts + explicit SM |
| `DEFINE_NVIDIA_<NAME>_BLOCKDIM_LAYOUT_SM(..., TC, LA, LB, LC, SM)` | every parameter explicit |

Layout arguments (`LA, LB, LC`) are integer literals: `0` = `col_major`, `1` = `row_major`. GRiD-flag aliases for the common transpose case:

```cpp
#define DEFINE_NVIDIA_GEMM_TRANSB(M, N, K)               // alias for LAYOUT(M, N, K, 0, 1, 0)
#define DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB(M, N, K, TC)  // alias for BLOCKDIM_LAYOUT(M, N, K, TC, 0, 1, 0)
```

To add a custom size, place the macro inside `namespace glass::nvidia` in your `.cu` file:

```cpp
#include "glass-nvidia.cuh"
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM(16, 16, 16)
    DEFINE_NVIDIA_GEMV(16, 16)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(16, 16, 16, 256)         // pinned thread count
    DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB(16, 16, 16, 256)  // pinned + B is row-major
    DEFINE_NVIDIA_POSV_BLOCKDIM(16, 4, 256)              // SPD solve, 4 RHS
}}
```

#### Host-side query API

```cpp
// Smallest BlockDim cuBLASDx accepts for (T, M, N, K, SM). All constexpr.
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM_VAL = SMS>
constexpr uint32_t glass::nvidia::gemm_min_block_threads();

// True iff BLOCK_THREADS >= the minimum.
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BT, uint32_t SM_VAL = SMS>
constexpr bool glass::nvidia::gemm_block_threads_valid();

// Same for gemv (Size<M, 1, N>):
constexpr uint32_t glass::nvidia::gemv_min_block_threads<T, M, N, SM_VAL>();
constexpr bool     glass::nvidia::gemv_block_threads_valid<T, M, N, BT, SM_VAL>();
```

These do **not** require a `DEFINE_NVIDIA_*` macro — they construct the GEMM type inline and read `block_dim` directly. Useful for codegen that wants to pick `MAX_PERF_LEVEL_THREADS` at generation time.

#### Debug assertions (P1-4)

Compile without `-DNDEBUG` and the wrappers `assert(blockDim >= GEMM::block_dim)` inside every `run()`. Misconfigured launches now fail with a clean assertion message rather than silently deadlocking. The assertions compile out under `-DNDEBUG`.

---

## Running Tests

```bash
cd test
pip install -r requirements.txt
pytest -v
```

The first run compiles the CUDA test binaries using `nvcc`. Subsequent runs skip recompilation unless source files have changed (cached by content hash). Compiled binaries are placed in `test/build/`.

```bash
pytest test_l1.py -v
pytest test_l1.py -k "simple"      # glass:: threadIdx variants only
pytest test_l1.py -k "cg"          # glass::cgrps:: cooperative groups variants
pytest test_l1.py -k "simple_hs"   # high_speed warp-shuffle variants
```

---

## Benchmarks

The benchmark suite compares GLASS variants against block-level CUDA library baselines AND against each other:

| File | Comparison |
|------|-----------|
| `bench_reduce.cu` | `glass::*::reduce/dot/l2norm` (plain, low_memory, high_speed, compile-time) vs CUB `BlockReduce` vs `glass::nvidia::reduce` (CUB-backed) |
| `bench_gemv.cu` | `glass::gemv` (runtime + compile-time) vs raw cuBLASDx vs `glass::nvidia::gemv` (default block_dim + caller-pinned `BlockDim<256>`) |
| `bench_gemm.cu` | `glass::gemm` (plain, tiled, compile-time) vs raw cuBLASDx vs `glass::nvidia::gemm` (default + caller-pinned) |
| `bench_blockdim.cu` | `glass::nvidia::gemm` with cuBLASDx-chosen block_dim vs caller-pinned `BlockDim<128>` vs `BlockDim<352>` (the GRiD iiwa14 launch that pre-fix would have deadlocked) |
| `bench_gemm_batched.cu` | `glass::nvidia::gemm_batched<...,BATCH>` vs naive `for(b)` loop calling `gemm` BATCH times, for BATCH ∈ {4, 8, 16, 32} |
| `bench_gemm_batched_1d.cu` | New 1D-launch `glass::nvidia::gemm_batched_1d` (SIMT vs cuBLASDx), for BATCH ∈ {4, 8, 16, 32} — feeds the autotune table |
| `bench_lapack.cu` *(needs cuSOLVERDx)* | pure-SIMT `glass::cholDecomp_InPlace` / `glass::trsm` vs `glass::nvidia::chol_inplace` / `trsm` / `posv` (fused) |

CUB is bundled with CUDA 11+. cuBLASDx and cuSOLVERDx ship together in NVIDIA MathDx (see [`bench/INSTALL.md`](bench/INSTALL.md)).

```bash
# Set MATHDX_ROOT to your MathDx installation, then:
python3 bench/run_bench.py

# Custom iteration count (default: 10000):
python3 bench/run_bench.py --iters 50000

# Skip cuBLASDx (bench_gemv/gemm/blockdim/batched/lapack will not run):
python3 bench/run_bench.py --no-cublasdx
```

`bench_lapack` is automatically skipped if `cusolverdx.hpp` is not present under `$MATHDX_ROOT/include/`.

**Anti-optimization safeguards** are baked into every bench loop: per-iteration writes to a `volatile` sink defeat dead-store elimination; destructive inputs (Cholesky, LU, QR all overwrite their input) are reloaded from a master copy each iteration; `nvcc -Xptxas -O1` is enforced. Numbers below ~0.1 µs/op for a non-trivial kernel almost always indicate the bench was elided — recheck the safeguards if you see that.

### Tuning your local build

The `glass::nvidia::*` auto-dispatch picks SIMT-vs-cuBLASDx per shape via
the heuristic in `should_use_cublasdx*<>()`. To override that heuristic
with **actual measurements on your hardware**:

```bash
python3 bench/autotune.py
# → measures all 5 round-2 primaries, writes bench/tuning/<hostname>.cuh.
# → does NOT modify src/nvidia/tuning_table.cuh.
#
# Compile your project with:
#   nvcc ... -DGLASS_TUNING_TABLE_LOCAL='"bench/tuning/<hostname>.cuh"' ...
#
# Per-host files are gitignored — only the shipped global table is
# tracked in version control. Want to PR your measurements upstream?
# See bench/TUNING.md (covers --in-tree mode for shipped-table updates).
```

The shipped global table (`src/nvidia/tuning_table.cuh`) is grown
collaboratively — contributions are welcomed via PR. See
[`bench/TUNING.md`](bench/TUNING.md) for the contributor workflow,
including `autotune.py --in-tree` (writes specializations directly
into the shipped table, preserving the round-2 primaries above it).

Timing uses the GRiD pattern — the iteration loop runs inside the kernel to amortize launch overhead. Results are printed as a Markdown table and saved to `bench/results/bench_<hostname>.json`.

---

## Notes

- Matrices default to **column-major** (Fortran) order, consistent with cuBLAS. Pure-SIMT `glass::` uses a `ROW_MAJOR=true` template parameter; `glass::nvidia::` uses the `layout` enum (`col_major` / `row_major`) per matrix via `LA`, `LB`, `LC` template arguments.
- `dot` and reduction variants modify the input array in-place (result in `x[0]`); `l2norm` squares elements before reducing.
- `cholDecomp_InPlace` only fills the **lower triangle** of the matrix; the upper triangle retains input values.
- `glass::gemm` (pure-SIMT) with `TRANSPOSE_B=true` currently requires B to be square (n×n). The `glass::nvidia::gemm` path has no such restriction — use `LB = layout::row_major` (or `DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB`).
- `glass::nvidia::*` (default form) requires exactly the thread count returned by `gemm_threads<T,M,N,K>()`. Use the `BLOCK_THREADS` template parameter (with `DEFINE_NVIDIA_<NAME>_BLOCKDIM`) to launch with any thread count `≥ gemm_min_block_threads<T,M,N,K>()` — extras go idle inside the GEMM. Compile without `-DNDEBUG` to get a clean assertion if the launch is too small instead of a silent deadlock.
- `glass::nvidia::trsm` does not accept a non-1.0 `alpha` natively (cuSOLVERDx's `trsm` has no alpha); the wrapper pre-multiplies `B` by `alpha` in shared memory before calling execute.
