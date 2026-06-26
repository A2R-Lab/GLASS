# GLASS

**GLASS is a comprehensive, header-only CUDA C++ `__device__` template library for
block-local linear algebra on GPUs** — BLAS, LAPACK-style factorizations and triangular
solves, dense linear-system solvers, and related algorithms, all under one single-block
calling convention. It is the foundational linear-algebra layer underneath
[GRiD](https://github.com/A2R-Lab/GRiD),
[MPCGPU](https://a2r-lab.org/publication/mpcgpu/),
[GATO](http://a2r-lab.org/GATO/),
[HJCD-IK](https://a2r-lab.org/publication/hjcdik/), and other A2R Lab GPU solvers.

📖 **Full documentation: <https://a2r-lab.github.io/GLASS/>** (source under [`docs/source/`](docs/source/)).

## Overview

GLASS functions are `__device__` helpers that operate on data already in shared or device
memory. Every function assumes it runs within **one CUDA block** — the caller launches one
block per independent data item — which makes GLASS a composable layer for kernels in
model-predictive control, trajectory optimization, and rigid-body dynamics. It is the
linear-algebra layer underneath [GRiD](https://github.com/A2R-Lab/GRiD).

It began as hand-rolled SIMT subroutines tuned for the very small matrices where vendor
launch/dispatch overhead dominates, and has grown into a unified single-block surface that
also wraps NVIDIA's device-side libraries — CUB (L1), cuBLASDx (L2/L3), cuSOLVERDx (LAPACK) —
under one `__device__` calling convention, so one kernel can mix hand-rolled and vendor-backed
primitives without leaving the block.

### Interfaces

GLASS exposes **three primary interfaces** — pick the one that matches how your problem maps
onto the GPU. Two are **block-scoped** (one block per problem), one is **warp-scoped** (one
warp per problem, for packing many tiny problems into a block); all share the same operations:

| Interface | Scope | What it is / when to choose it | Header |
|-----------|-------|--------------------------------|--------|
| `glass::` (**Block**) | block | Hand-rolled SIMT, `threadIdx` / `blockDim` — no deps. The default; one moderate-to-large problem per block | `glass.cuh` |
| `glass::warp::` (**Warp**) | **warp** | Single-warp SIMT via `__shfl_*_sync` (*selected* L1/L2/L3 ops, no `__syncthreads`). Pack many small independent problems into one block | inline in the base headers (via `glass.cuh`) |
| `glass::nvidia::` (**Nvidia**) | block | CUB + cuBLASDx + cuSOLVERDx, auto-dispatched against SIMT by size (compile-time sizes). When a vendor tensor-core kernel wins at your size | `glass-nvidia.cuh` |

> **Note:** `glass::cgrps::` (header `glass-cgrps.cuh`) is a convenience cooperative-groups
> *alias* of the Block interface — the same SIMT loop indexed via a
> `cooperative_groups::thread_group`, numerically identical and **not** a separately-tuned
> backend.

Both `glass::` and `glass::cgrps::` offer **runtime** (size as arg) and **compile-time** (size
as template arg) overloads. Reductions additionally offer `_lowmem` (no scratch)
and `_fast` (warp-shuffle) suffixed forms (e.g. `glass::reduce_lowmem` / `glass::reduce_fast`). The dense surface covers `gemm`/`gemv`/`ger`,
`iamax`, `trsv`/`trmv`, `syrk`/`syr2k`, `inv`/`cholDecomp_InPlace` (single **and K-way fused**),
`ldlt`/`ldlt_solve`, and `posv`/`potrs`; plus contraction-parallel `*_reduced`, `tensor_*`, and
`congruence_*` families. See the [namespace & naming guide](docs/source/user_guide/concepts/namespaces.rst).

### Higher-level solvers

Built on the primitives above (single-block) for the block-tridiagonal SPD systems of
trajectory optimization / MPC:

| Function | What | Header |
|----------|------|--------|
| `glass::bdmv` | block-tridiagonal matvec (`[L\|D\|R]` strips, padded vectors) | `src/base/banded/bdmv.cuh` |
| `glass::pcg` | single-block preconditioned conjugate gradient (`S x = b`) | `src/base/pcg/solve.cuh` |

An internal box-constrained QP solver, `glass::internal::box_qp`, also lives in the tree but is
**not** part of the public surface (QP is optimization, not linear algebra).

## Quick start

```cpp
#include "glass.cuh"

__global__ void my_kernel(float* A, float* B, float* C, int m, int n, int k) {
    glass::gemm(m, n, k, 1.f, A, B, 0.f, C);   // all block threads cooperate on one problem
}

my_kernel<<<num_items, 256>>>(A, B, C, m, n, k);   // one block per data item
```

Runnable, self-contained programs (one concept each) live in [`examples/`](examples/). GEMM
follows the **standard BLAS convention** — `C` is M×N, contraction K (`A` is M×K, `B` is K×N) —
with `TRANSPOSE_A` / `TRANSPOSE_B` operand flags and a single `ROW_MAJOR_C` output flag; a
row-major operand is just a transpose. See [`examples/10_gemm_basics.cu`](examples/10_gemm_basics.cu).

## Installation

GLASS is **header-only** — add the repo root to your include path and `#include "glass.cuh"`.
The pure-SIMT surface needs only `nvcc -std=c++17`. The `glass::nvidia::` paths additionally
need NVIDIA MathDx (cuBLASDx / cuSOLVERDx) and extra flags:

| Surface | Build requirements |
|---------|--------------------|
| `glass.cuh`, `glass-cgrps.cuh` | C++17 — no extra deps |
| `glass-nvidia.cuh` (L1) | C++17 + CUB (bundled with CUDA 11+) |
| `glass-nvidia.cuh` (L2/L3 GEMM/GEMV/batched) | C++17 + `--expt-relaxed-constexpr` + cuBLASDx |
| `glass-nvidia.cuh` (LAPACK) | C++17 + `--expt-relaxed-constexpr` + `-rdc=true -dlto -lcusolverdx -lcublas -lcusolver -lcudart` + cuSOLVERDx |

The nvidia wrappers auto-detect availability (`GLASS_HAVE_CUBLASDX` / `GLASS_HAVE_CUSOLVERDX`).
Full setup, linking, and the MathDx download are in [`bench/INSTALL.md`](bench/INSTALL.md) and
the [installation guide](docs/source/user_guide/getting_started/installation.rst).

## Build & test

```bash
pip install -r test/requirements.txt
pytest test/                 # compiles test/cuda/*.cu once, caches by source hash
```

`rm -rf test/build` forces a clean rebuild. Details in
[`docs/source/user_guide/tutorials/running_tests.rst`](docs/source/user_guide/tutorials/running_tests.rst).

## Documentation map

The README is a landing page; the deep reference lives in the
[hosted docs](https://a2r-lab.github.io/GLASS/) (sources in [`docs/source/`](docs/source/)):

| Topic | Page |
|-------|------|
| API reference (L1 / L2 / L3 / nvidia / warp / banded) | [`api_reference/`](docs/source/api_reference/) |
| Namespaces, naming rules, and the two-axis taxonomy | [`concepts/namespaces.rst`](docs/source/user_guide/concepts/namespaces.rst) |
| Choosing a backend + tuning for your hardware | [`concepts/tuning.rst`](docs/source/user_guide/concepts/tuning.rst) |
| `glass::nvidia::gemm` cuBLASDx-vs-SIMT dispatch | [`concepts/backend_dispatch.rst`](docs/source/user_guide/concepts/backend_dispatch.rst) |
| `TRAILING_SYNC` and barrier conventions | [`concepts/trailing_sync.rst`](docs/source/user_guide/concepts/trailing_sync.rst) |
| Contraction-parallel (`*_reduced`) family | [`concepts/contraction_parallel.rst`](docs/source/user_guide/concepts/contraction_parallel.rst) |
| Block-tridiagonal layout (`bdmv` / `pcg`) | [`concepts/block_tridiagonal.rst`](docs/source/user_guide/concepts/block_tridiagonal.rst) |
| Worked examples + quickstart | [`tutorials/`](docs/source/user_guide/tutorials/) · [`examples/`](examples/) |
| Benchmarks + measured sweep results | [`tutorials/benchmarks.rst`](docs/source/user_guide/tutorials/benchmarks.rst) · [`tutorials/sweep_results.rst`](docs/source/user_guide/tutorials/sweep_results.rst) |

## Notes / gotchas

- **One block per problem.** Every function runs inside a single block; launch `<<<num_items, threads>>>`.
- **Column-major by default** (Fortran order, matching cuBLAS). GEMM uses `TRANSPOSE_A` /
  `TRANSPOSE_B` + `ROW_MAJOR_C` (a row-major operand is just a transpose); GEMV keeps a
  per-matrix `ROW_MAJOR` flag (its transpose changes the math op); `glass::nvidia::` uses the
  `layout` enum per matrix (`LA`/`LB`/`LC`).
- **Reductions are destructive.** `dot` / `nrm2` / reduction variants write the result to `x[0]`
  and may consume the input as scratch; `nrm2` squares elements before reducing. The
  `glass::warp::` forms return the value instead.
- `cholDecomp_InPlace` fills only the **lower triangle**; the upper retains input values.
- `glass::nvidia::*` (default form) requires exactly `gemm_threads<T,M,N,K>()` threads; use the
  `BLOCK_THREADS` template parameter (with `DEFINE_NVIDIA_<NAME>_BLOCKDIM`) to launch any count
  `≥ gemm_min_block_threads<T,M,N,K>()`. Compile without `-DNDEBUG` for a clean assertion instead
  of a silent deadlock if the launch is too small.
- `glass::nvidia::trsm` has no native non-1.0 `alpha` (cuSOLVERDx limitation); the wrapper
  pre-scales `B` in shared memory before `execute`.
