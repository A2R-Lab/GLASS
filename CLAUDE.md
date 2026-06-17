# CLAUDE.md — orientation for AI agents (and humans) working on GLASS

GLASS is **GPU Linear Algebra for Single-block Systems**: a header-only CUDA
library of BLAS/LAPACK-style `__device__` routines that run **inside one CUDA
thread block**. You launch one block per independent problem; the block's
threads cooperate over data already in shared/global memory. GLASS is the
linear-algebra layer under [GRiD](https://github.com/A2R-Lab/GRiD).

**Before changing any primitive, read `docs/agent_debugging_guide.md`** — it is
the runbook for the recurring single-block CUDA bug classes (missing
`__syncthreads()`, thread-count non-invariance, `beta=0` reads C, layout flags).

## The mental model

One block per problem. Every public function strides over its data with
`for (i = rank; i < n; i += size)` and must be **thread-count invariant** —
identical output at 1 thread, 32, a partial warp, or many warps. The #1 bug is a
missing barrier between a write phase and a dependent read: invisible at 32
threads (one warp runs lockstep), a race at 64+.

## Three namespaces / three umbrella headers

| Header | Namespace | What it is |
|--------|-----------|------------|
| `glass.cuh` | `glass::` | Hand-rolled pure-SIMT (`threadIdx`/`blockDim`). No deps. |
| `glass-cgrps.cuh` | `glass::cgrps::` | Same surface via cooperative groups. |
| `glass-nvidia.cuh` | `glass::nvidia::` | CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size. Needs NVIDIA MathDx (`MATHDX_ROOT`). |

Scoped sub-namespaces also live in the base headers (pulled in by `glass.cuh`):
`glass::warp::` (single-warp SIMT variants for warp-per-problem kernels),
`glass::high_speed::` / `glass::low_memory::` (perf vs scratch trade-offs of
reductions/dots), and the block-tridiagonal solver families `glass::banded::`
(matvec) and `glass::pcg::` (preconditioned conjugate gradient).

## Source layout

- `src/base/{L1,L2,L3}/` — **the live public API** (pulled into `namespace glass`
  by `glass.cuh` via an `#include` trick — the functions are written at file
  scope and the namespace wraps the includes). The `glass::warp::` variants live
  *inline* in these base headers (e.g. `reduce.cuh`, `gemm.cuh`), not a separate dir.
- `src/base/banded/bdmv.cuh`, `src/base/pcg/solve.cuh` — block-tridiagonal matvec
  + PCG solver (public; `glass::banded::` / `glass::pcg::`). Block-tridiagonal
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
