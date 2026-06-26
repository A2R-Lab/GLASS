# CLAUDE.md — orientation for AI agents (and humans) working on GLASS

GLASS is a **comprehensive, header-only CUDA C++ `__device__` template library
for block-local linear algebra on GPUs** — BLAS, LAPACK-style factorizations and
triangular solves, dense linear-system solvers, and related algorithms under one
calling convention. Routines run **inside one CUDA block**: you launch one block
per independent problem and the block's threads cooperate over data already in
shared/global memory. Three primary interfaces — **Block** (`glass::`), **Warp**
(`glass::warp::`, for packing many small problems into one block), and **Nvidia**
(`glass::nvidia::`, vendor-backed). GLASS is the foundational linear-algebra layer
under [GRiD](https://github.com/A2R-Lab/GRiD), MPCGPU, GATO, HJCD-IK, and other
A2R Lab GPU solvers.

**Before changing any primitive, read `docs/agent_debugging_guide.md`** — it is
the runbook for the recurring single-block CUDA bug classes (missing
`__syncthreads()`, thread-count non-invariance, `beta=0` reads C, layout flags).

## The mental model

One block per problem. Every public function strides over its data with
`for (i = rank; i < n; i += size)` and must be **thread-count invariant** —
identical output at 1 thread, 32, a partial warp, or many warps. The #1 bug is a
missing barrier between a write phase and a dependent read: invisible at 32
threads (one warp runs lockstep), a race at 64+.

## Interfaces

Three **primary interfaces** — **Block** (`glass::`), **Warp** (`glass::warp::`),
and **Nvidia** (`glass::nvidia::`) — picked by how the problem maps onto the GPU.
Block and Nvidia are block-scoped (one block per problem); Warp is warp-scoped (one
warp per problem, for packing many small problems into a block):

| Interface | Scope | What it is | Header |
|-----------|-------|------------|--------|
| `glass::` (Block) | block | Hand-rolled pure-SIMT (`threadIdx`/`blockDim`). No deps. | `glass.cuh` |
| `glass::warp::` (Warp) | warp | Single-warp SIMT (`__shfl_*_sync`), *selected* L1/L2/L3 ops; `glass::warp::posv` is the composed warp-per-problem SPD solve (chol → forward/back `trsv`). Inline in the base L1/L2/L3 headers. | via `glass.cuh` |
| `glass::nvidia::` (Nvidia) | block | CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size. Needs MathDx (`MATHDX_ROOT`). | `glass-nvidia.cuh` |

`glass::cgrps::` (header `glass-cgrps.cuh`) is a **convenience alias** of the Block
interface — identical numerics (the same SIMT loop, indexed via a `thread_group`),
for cooperative-groups callers / arbitrary sub-block tiles. NOT a separately-tuned
backend.

Convention: **namespace = scope/backend; function name = operation.** So a warp
band matvec would be `glass::warp::bdmv`, never a `banded::` namespace.

Also in the base headers: the `_fast` (warp-shuffle) / `_lowmem` (thread-0 serial)
reduction-strategy suffixes on the reduction family (`reduce`/`dot`/`nrm2`/`asum`/
`vector_norm`/`nrm1_diff`/`iamax`) — a strategy rides on the function name, not a
namespace (these were `high_speed::`/`low_memory::` until the 2026-06 convergence).
Plus the block-tridiagonal **functions** `glass::bdmv` (matvec) and `glass::pcg`
(preconditioned conjugate gradient, with `glass::pcg_scratch_bytes`). An internal `glass::internal::box_qp` lives in the tree
but is not part of the public surface (see `docs/open-tasks/qp_solver_scope.md`).

Recent L1/L2/L3 additions (all single-block, thread-count invariant): `iamax`
(L1, BLAS i_amax pivot primitive); `trsv` / `trmv` (L2 triangular solve / matvec,
`LOWER`/`UNIT`/`TRANSPOSE` template flags); `syrk` / `syr2k` (L3 symmetric rank-k/2k,
both `AAᵀ` and `AᵀA` via a `TRANSPOSE` flag, `FillMode` Lower/Upper/Full); `ldlt` /
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

Contraction-parallel + higher-level families (block + `warp::` + `cgrps::`, all
single-block, thread-count invariant; see
`docs/source/user_guide/concepts/contraction_parallel.rst`): the **`*_reduced`**
ops `gemm_reduced` / `gemv_reduced` / `syrk_reduced` map one warp to one output
and split the contraction across its lanes; **tensor** ops `tensor_vec_contract`
(`CONTRACT` axis enum + `SYMMETRIC`) / `vec_tensor_vec`; **congruence** forms
`congruence_sym` (XᵀMX) / `bilinear` (XᵀMY); and `riccati_gain`
(= congruence + bilinear + checked `posv`). Robustness rides as **compile-out
`bool` flags** (default-false, byte-identical PTX when off): `CHECK` on
`cholDecomp_InPlace` / `ldlt` (+ `inertia`), `REGULARIZE`+`CHECK` on multi-RHS
`posv` (the fused regularize→factor→solve). **Naming rule:** namespace = scope,
different decomposition = a name suffix (`_reduced`), additive behavior = a
compile-out flag (see `concepts/namespaces.rst`). **Perf caveat (measured, sm_120,
`bench/REDUCED_SWEEP_RESULTS.md`):** `*_reduced` is *slower* than serial in almost
every shape — `glass::suggested_use_reduced<n_out,K,blockDim>()` returns true only
in a narrow corner. The tensor/congruence families are for **expressiveness +
fusion**, not for beating a tight serial loop. The shared 32-way invariance
primitive `reduced_tree32` lives in `L1/reduce.cuh`.

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
