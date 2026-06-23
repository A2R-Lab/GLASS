# CLAUDE.md — orientation for AI agents (and humans) working on GLASS

GLASS is **GPU Linear Algebra Simple Subroutines**: a header-only CUDA
library of BLAS/LAPACK-style `__device__` routines that run **inside one CUDA
block**. You launch one block per independent problem; the block's threads
cooperate over data already in shared/global memory. Block-scoped by default,
now expanding to warp-level primitives (`glass::warp::`) for packing many small
problems into one block. GLASS is the linear-algebra layer under
[GRiD](https://github.com/A2R-Lab/GRiD).

**Before changing any primitive, read `docs/agent_debugging_guide.md`** — it is
the runbook for the recurring single-block CUDA bug classes (missing
`__syncthreads()`, thread-count non-invariance, `beta=0` reads C, layout flags).

## The mental model

One block per problem. Every public function strides over its data with
`for (i = rank; i < n; i += size)` and must be **thread-count invariant** —
identical output at 1 thread, 32, a partial warp, or many warps. The #1 bug is a
missing barrier between a write phase and a dependent read: invisible at 32
threads (one warp runs lockstep), a race at 64+.

## Call surfaces

Primitives are **block-scoped** by default in three numerically-interchangeable
backends, plus a **warp-scoped** surface for warp-per-problem kernels:

| Namespace | Scope | What it is | Header |
|-----------|-------|------------|--------|
| `glass::` | block | Hand-rolled pure-SIMT (`threadIdx`/`blockDim`). No deps. | `glass.cuh` |
| `glass::cgrps::` | block | Convenience alias of `glass::` — identical numerics (same SIMT loop, indexed via a `thread_group`); for cooperative-groups callers / arbitrary sub-block tiles. NOT a separately-tuned backend. | `glass-cgrps.cuh` |
| `glass::nvidia::` | block | CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size. Needs MathDx (`MATHDX_ROOT`). | `glass-nvidia.cuh` |
| `glass::warp::` | warp | Single-warp SIMT (`__shfl_*_sync`), *selected* L1/L2/L3 ops; `glass::warp::posv` is the composed warp-per-problem SPD solve (chol → forward/back `trsv`). Lives inline in the base L1/L2/L3 headers. | via `glass.cuh` |

Convention: **namespace = scope/backend; function name = operation.** So a warp
band matvec would be `glass::warp::bdmv`, never a `banded::` namespace.

Also in the base headers: `glass::high_speed::` / `glass::low_memory::` (perf vs
scratch trade-offs of reductions/dots), and the block-tridiagonal **functions**
`glass::bdmv` (matvec) and `glass::pcg` (preconditioned conjugate gradient, with
`glass::pcg_smem_size`). An internal `glass::internal::box_qp` lives in the tree
but is not part of the public surface (see `docs/open-tasks/qp_solver_scope.md`).

Recent L1/L2/L3 additions (all single-block, thread-count invariant): `iamax`
(L1, BLAS i_amax pivot primitive); `trsv` / `trmv` (L2 triangular solve / matvec,
`LOWER`/`UNIT`/`TRANS` template flags); `syrk` / `syr2k` (L3 symmetric rank-k/2k,
both `AAᵀ` and `AᵀA` via a `TRANS` flag, `FillMode` Lower/Upper/Full); `ldlt` /
`ldlt_solve` (L3 symmetric-indefinite LDLᵀ, non-pivoted, signature reserves
`bool pivot`/`piv` for a future Bunch-Kaufman path); `posv` / `potrs` (L3 SPD
solve = chol + 2×`trsv`); and **K-way fused** `invertMatrix` / `cholDecomp_InPlace`
(invert/factor K independent matrices interleaved over one block — `inv2`/`inv3`,
the 2-/3-matrix `invertMatrix` wrappers, are now thin wrappers). The warp surface adds `warp::{dot,axpy,copy,scal,gemv,trsv,iamax}`
+ the composed `warp::posv`.

Robust/perf variants (perf user vs robustness user): `invertMatrix_pivoted`
(partial-pivoting Gauss-Jordan, robust on small/zero leading pivots), `ldlt(...,
pivot=true, piv)` (symmetric 1×1 diagonal pivoting; full Bunch-Kaufman 2×2 still
deferred), and multi-RHS `posv`/`potrs` (`(n, nrhs, A, B)` — factor once, solve N
columns; B column-major).

## Source layout

- `src/base/{L1,L2,L3}/` — **the live public API** (pulled into `namespace glass`
  by `glass.cuh` via an `#include` trick — the functions are written at file
  scope and the namespace wraps the includes). The `glass::warp::` variants live
  *inline* in these base headers (e.g. `reduce.cuh`, `gemm.cuh`), not a separate dir.
- `src/base/banded/bdmv.cuh`, `src/base/pcg/solve.cuh` — block-tridiagonal matvec
  + PCG solver (public; `glass::bdmv` / `glass::pcg`). Block-tridiagonal
  `[L|D|R]` strips + padded `(knot_points+2)*state_size` vectors.
- `src/cgrps/{l1,l2,l3}.cuh` — cooperative-groups variants.
- `src/nvidia/*.cuh` — vendor-backed paths + host-side query/size helpers.
- `src/L1`, `src/L2`, `src/L3` (non-base) were **removed** as legacy duplicates
  (superseded by the May-2026 `base/` refactor). Do not reintroduce them.
- `src/L3/box_qp.cuh` is a **validated but INTERNAL** box-constrained QP solver
  (`glass::internal::box_qp`) — deliberately NOT in `glass.cuh` or the public API
  (QP is optimization, not linear algebra). Tested by `test/test_qp.py`. See
  `docs/open-tasks/qp_solver_scope.md`.

## Build & test

GLASS is header-only — there is nothing to build to *use* it (just add the repo
root to your include path and `#include "glass.cuh"`). To run the tests:

```bash
pip install -r test/requirements.txt
pytest test/                 # compiles test/cuda/*.cu once, caches by source hash
```

`test/conftest.py` auto-detects the GPU arch (`nvidia-smi`) and caches compiled
test binaries keyed on a source hash — **if you add a new test source file, it
must be registered in that hash list or the cache won't rebuild** (see the
debugging guide). Optional cuBLASDx/cuSOLVERDx tests skip gracefully when MathDx
is absent. Force a clean rebuild with `rm -rf test/build`.

## Docs

Sphinx + Doxygen + Breathe under `docs/` (`cd docs && make all`). The API
reference is generated from the header `/** */` doc-comments — **new public
functions need a doc-comment and a `.. doxygenfile::` line** in
`docs/source/api_reference/`. Published to GitHub Pages on push to `main`.

## Conventions

- Short, single-line commit messages; no `Co-Authored-By` footer.
- Don't gate/skip an op per problem-size without saying so.
- Preserve thread-count invariance and the single-block model — never split a
  primitive across blocks.
```
