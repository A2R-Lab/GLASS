# CLAUDE.md — orientation for AI agents (and humans) working on GLASS

GLASS is a **comprehensive, header-only CUDA C++ `__device__` template library
for block-local linear algebra on GPUs** — BLAS, LAPACK-style factorizations and
triangular solves, dense linear-system solvers, and related algorithms under one
calling convention. Routines run **inside one CUDA block**: you launch one block
per independent problem and the block's threads cooperate over data already in
shared/global memory. Four primary interfaces — **Block** (`glass::`), **Warp**
(`glass::warp::`, for packing many small problems into one block), **Thread**
(`glass::thread::`, one problem per thread for low-DOF packing), and **Nvidia**
(`glass::nvidia::`, vendor-backed). GLASS is the foundational linear-algebra layer
under [GRiD](https://github.com/A2R-Lab/GRiD), MPCGPU, GATO, HJCD-IK, and other
A2R Lab GPU solvers.

**Pushing to main: any commit that touches `src/`, the `glass*.cuh` roots, or
`test/` MUST end with a fresh signed receipt** — run `./test/run_gpu_proof.sh`
(full GPU suite, ~40 min) and include the regenerated `test/gpu-proof.json` in
the push, or the `verify-gpu-proof` gate goes red (the receipt fingerprints the
source tree, so an un-attested source change can't verify). Library headers are
hashed into the test-binary cache key by GLOB (`src/**/*.cuh` + the `glass*.cuh`
roots) — new headers bust the cache automatically; only a NEW test/cuda driver
needs registering in `test/conftest.py` (compile target + hash entry).

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

Four **primary interfaces** — **Block** (`glass::`), **Warp** (`glass::warp::`),
**Thread** (`glass::thread::`), and **Nvidia** (`glass::nvidia::`) — picked by how the
problem maps onto the GPU. Block and Nvidia are block-scoped (one block per problem);
Warp is warp-scoped (one warp per problem); Thread is thread-scoped (one problem per
THREAD, 32 packed per warp). The ladder runs most→least problem packing:
thread → warp → block → nvidia.

| Interface | Scope | What it is | Header |
|-----------|-------|------------|--------|
| `glass::` (Block) | block | Hand-rolled pure-SIMT (`threadIdx`/`blockDim`). No deps. | `glass.cuh` |
| `glass::warp::` (Warp) | warp | Single-warp SIMT (`__shfl_*_sync`) mirroring most of the block surface — L1 reductions/vector ops, gemv/gemm/syrk, the factor/solve chain, tensor/congruence/riccati. Inline in the base L1/L2/L3 headers. | via `glass.cuh` |
| `glass::thread::` (Thread) | thread | One problem per thread, for LOW-DOF packing (N≲7: a warp-per-problem factor leaves ~26/32 lanes idle; this packs 32 problems in the warp instead). Sequential — no barriers, no shuffles, no `threadIdx` read. The `ThreadBarrier` (rank=0, size=1, no-op sync) collapses the SAME `*_impl` bodies, so each op runs the identical algorithm as its `glass::` twin on one thread (`test/test_thread.py` asserts agreement to a few ULP — bit-identity across the two instantiations is NOT guaranteed; the no-op sync frees nvcc to contract FMAs differently). Mirrors the branch-free warp surface: L1 `dot`/`reduce`/`nrm2`/`asum`/`nrm1_diff`/`axpy`(+`_strided`)/`scal`/`copy`(+`_strided`)/`rot`/`symmetrize`; L2 `gemv`/`trsv`; L3 `gemm`/`syrk`/`syr2k`/`trsm`/`potrf`/`posv`/`potrs`/`ldlt`(non-pivoted)/`ldlt_solve`/`inv`(non-pivoted) + the fused `tensor_vec_contract`/`vec_tensor_vec`/`congruence_sym`/`bilinear`/`congruence_accum`/`riccati_gain`. Inline in the base headers. | via `glass.cuh` |
| `glass::nvidia::` (Nvidia) | block | CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size. Needs MathDx (`MATHDX_ROOT`). | `glass-nvidia.cuh` |

**`glass::thread::` constraints (read before extending it):**
- **Compile-time `N` only.** The tier's value is a register-resident `T A[N*N]`; a
  runtime-`n` overload would silently spill and be strictly worse than `warp::`.
- **Measured ceiling `N <= 7`** (BOTH dtypes; nvcc 12.0/sm_86). Not the 255-register
  cap — an element-count threshold in nvcc's local-array promotion (49 promotes, 64
  does not). `#pragma unroll` does NOT lift it. Past N=7 `A` lands in local memory and
  the sweep shows the cliff directly (f32 gemv: 1.11ns at N=6 → 3.23ns at N=8).
- **Branch-free ops only.** Every lane owns a DIFFERENT problem, so a data-dependent
  branch diverges across the warp. Pivoted `ldlt`/`getrf`/`inv_pivoted`/`iamax` are
  therefore excluded on purpose — the *robust* variants are the wrong ones here.
- **No `_fast`/`_lowmem` twins.** Those name reduction STRATEGIES; one thread has none.
  `thread::dot` returns a `T` (serial accumulate) instead of reducing in place.
- **Barriers must route through `Bar`.** A raw `__syncthreads()` in a shared `*_impl`
  is block-wide, so with one problem per thread (and a ragged tail block whose
  out-of-range threads returned) it is a barrier with divergent participation ⇒ UB.
  This is why `trsv_impl` grew a defaulted `Bar` param.

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
`FillMode`/`Diag` enums + `TRANSPOSE` flag) and `trsm` (L3 multi-RHS triangular solve,
same flags); `syrk` / `syr2k` (L3 symmetric rank-k/2k,
both `AAᵀ` and `AᵀA` via a `TRANSPOSE` flag, `FillMode` Lower/Upper/Full); `ldlt` /
`ldlt_solve` (L3 symmetric-indefinite LDLᵀ, non-pivoted by default, opt-in
Bunch-Kaufman pivoting via `bool pivot`/`int32_t* piv`); `posv` / `potrs` (L3 SPD
solve = chol + 2×`trsv`); and **K-way fused** `inv` / `potrf`
(invert/factor K independent matrices interleaved over one block — `inv2`/`inv3`,
the 2-/3-matrix `inv` wrappers, are now thin wrappers); and `eigh` /
`psd_project` (fixed-sweep round-robin cyclic Jacobi + eigenvalue-clip
reconstruction — the DETERMINISTIC sibling of `syev`/`eig_clamp`: no
convergence check, unsorted spectrum, bit-identical across thread counts;
built for GATO's batched stage-Hessian PSD projection; oracle =
jacobi_study.py in the GATO so_sqp prototype). The `warp::` surface mirrors most of the block L1
reduction/vector family plus `gemv`/`gemm`/`syrk`/`syr2k`, the factor/solve chain
(`potrf`/`trsv`/`trsm`/`posv`/`potrs`/`ldlt`/`ldlt_solve`/`inv`), and the tensor/congruence/riccati families.

Robust/perf variants (perf user vs robustness user): `inv_pivoted`
(partial-pivoting Gauss-Jordan, robust on small/zero leading pivots), `ldlt(...,
pivot=true, piv)` (full Bunch-Kaufman 1×1/2×2 partial pivoting — LAPACK `sytf2`;
`piv` is int32, negative entries mark 2×2 blocks; handles zero diagonals like
`[[0,1],[1,0]]`; block path only — `warp::ldlt` stays non-pivoted), and multi-RHS
`posv`/`potrs` (`(n, nrhs, A, B)` — factor once, solve N columns; B column-major).

Contraction-parallel + higher-level families (block + `warp::` + `cgrps::`, all
single-block, thread-count invariant; see
`docs/source/user_guide/concepts/contraction_parallel.rst`): the **`*_reduced`**
ops `gemm_reduced` / `gemv_reduced` / `syrk_reduced` map one warp to one output
and split the contraction across its lanes; **tensor** ops `tensor_vec_contract`
(`CONTRACT` axis enum + `SYMMETRIC`) / `vec_tensor_vec`; **congruence** forms
`congruence_sym` (XᵀMX) / `bilinear` (XᵀMY); and `riccati_gain`
(= congruence + bilinear + checked `posv`). Robustness rides as **compile-out
`bool` flags** (default-false, byte-identical PTX when off): `CHECK` on
`potrf` / `ldlt` (+ `inertia`), `REGULARIZE`+`CHECK` on multi-RHS
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

## Benchmarking & tuning

`bench/tune.py --sm auto [--margin 0.05] [--quick] [--dry-run]` is the **one**
entry point that remeasures this GPU and regenerates every shipped defaults
table under a single noise margin: the thread/warp/block/nvidia ladder
(`glass-defaults.cuh` — per-arch `ideal_sm*` tables + SM dispatch; a first-time arch
gets its own table and dispatch case, other arches' tables stay untouched), the per-(M,N,K) cuBLASDx-vs-SIMT table
(`src/nvidia/tuning_table.cuh`, via the `bench/autotune.py` engine it drives),
and the `suggested_use_reduced<>` predicate. The shared tie rule lives in
**`bench/tune_pick.py::pick`** — a dependency impl (nvidia/cublasdx/reduced)
wins only if it beats the simplest no-dependency impl by more than the margin;
ties stay on the launchable-everywhere path. **Run perf sweeps on a quiet GPU**
(isolated timing); use `--dry-run` to diff a regeneration before committing it.
Compilation dominates the wall clock, so `bench/tune.py --prebuild` compiles
every binary into a persistent hash-keyed cache (`bench/.tune_cache/`, gitignored)
with no timing — run it anytime (even while the GPU is busy) so the later quiet-GPU
sweep is execute-only. Building isn't timed, so parallelize it with
`--build-jobs N` (size to free_RAM/7 — each cuBLASDx compile needs ~6-7GB; e.g.
`--build-jobs 6` on a 64GB box); the timed legs always run serially for clean
measurement. Details: `bench/TUNING.md`.

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
