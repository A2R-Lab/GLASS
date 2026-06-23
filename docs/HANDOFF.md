# GLASS handoff

Living status doc. Update the top as work lands. For onboarding read
`docs/STARTUP_PROMPT.md` first.

## 2026-06-23 — mega sweep (warp/block/nvidia) + fp64 nvidia wrappers

Built a three-contender scaling bench and extended the nvidia wrappers to double.

- **`bench/bench_mega_sweep.cu`** (was `bench_warp_vs_block.cu`) — warp vs block(SIMT) vs
  **nvidia** (cuBLASDx/cuSOLVERDx) on one ns/problem axis, N to 128, f32+f64, NPROB
  1→32768. nvidia is FORCED at every N (explicit DEFINEs bypass the size-heuristic) to
  draw the full vendor curve. Launches are **error-checked** — a descriptor whose smem
  exceeds the device opt-in cap (99 KB on RTX 5090) is skipped, not mis-timed as a "win".
  Runner: `run_mega_sweep.sh`. Results: **`bench/MEGA_SWEEP_RESULTS.md`** (raw:
  `mega_sweep_20260623_0917.txt`). Headline: the warp→SIMT→MathDx ladder is *per-op* —
  dot/gemv stay warp(+block); gemm/chol/posv hand off warp→**nvidia (N≥16)**→block (when
  nvidia runs out of smem); trsv only mid-band (cuSOLVERDx trsm scales poorly). f64
  narrows the nvidia band (caps at N=64).
- **fp64 nvidia** — parameterized the cuBLASDx/cuSOLVERDx wrapper macros on the scalar
  type (`src/nvidia/{l2,l3,lapack}.cuh`): chol/trsm/posv/gemm/gemv now have
  `DEFINE_NVIDIA_*_PREC(..., double)` variants alongside the float defaults (back-compat
  unchanged). potrs/getrf/getrs/gesv/geqrf/gels left float-only (out of scope). dot is
  already `template<T>` (CUB).
- **Validation** — new `test/test_nvidia_f64.py` + `test/cuda/test_nvidia_f64.cu`
  (registered in conftest, gated on cuSOLVERDx): posv/gemm/gemv double vs numpy, 9 pass
  at f64 tol (posv residual ~1e-15). Float regression: 75 existing nvidia/posv tests pass.
- **Known limit**: f64 nvidia is smem-capped at N≤64 on sm_120 (99 KB opt-in); higher f64
  N would need a tiled descriptor. Logged as a finding, not fixed.

## 2026-06-21 — follow-up wave: pivoting + multi-RHS + warp::iamax

Four file-isolated follow-ups (parallel worktree agents → I verified + merged; zero
shared-file edits so the merges were conflict-free). Deferred-but-API-frozen items from
the expansion are now implemented except full Bunch-Kaufman 2×2.

- **`invertMatrix_pivoted`** (inv.cuh) — robust partial-pivoting (row) Gauss-Jordan;
  full-width elimination (the unpivoted column-reduced window is invalid under row
  swaps); scratch `3*dimA+1`. Verified incl. near-singular leading-pivot cases (27).
- **`ldlt(..., pivot=true, piv)`** (ldlt.cuh) — symmetric 1×1 diagonal pivoting via
  `low_memory::iamax`; `ldlt_solve` applies `P`/`Pᵀ`; scratch stays `(n+1)`. Full
  Bunch-Kaufman 2×2 still deferred (a structurally-zero diagonal block still can't
  factor — documented limitation). Verified (204).
- **Multi-RHS `posv`/`potrs`** (posv.cuh) — `(n, nrhs, A, B)` column-major B, factor
  once / solve per column. Verified (64).
- **`warp::iamax`** (iamax.cuh) — single-warp argmax-abs, register-broadcast, lowest-index
  tie-break; multi-warp test. Verified (63).

Still deferred: full Bunch-Kaufman 2×2 pivoting; the batching-uniformity cleanup pass
(cross-cutting — held for a focused solo pass). Perf characterization backlogged in
`docs/open-tasks/perf_autotune_breakeven.md`.

## 2026-06-21 — BLAS/LAPACK expansion (8 new ops) + warp buildout

Orchestrated explore → gate → implement (worktree-isolated agents) → verify → merge.
Roadmap + gate decisions: `docs/open-tasks/expansion_roadmap_2026-06.md`. Every op is
single-block, thread-count invariant (1/7/33/256 sweeps), with a numpy/scipy oracle and
a dedicated `test_<op>.cu`/`test_<op>.py`. **Full integrated suite: 1405 passed.**
racecheck (syrk, ldlt, fused K-way, warp posv) — 0 hazards.

- **`syrk` / `syr2k`** (L3) — symmetric rank-k/2k; both `AAᵀ` (Schur) and `AᵀA`
  (Gram/Hessian JᵀJ) via a `TRANS` flag; `FillMode` Lower/Upper/Full; no-barrier
  symmetric mirror write.
- **`trsv` / `trmv`** (L2) — triangular solve / matvec; `LOWER`/`UNIT`/`TRANS` template
  flags; trailing-sync contract (so `posv` composes barrier-free). trmv = out-of-place
  core + in-place(+scratch) wrapper.
- **`posv` / `potrs`** (L3) — pure-SIMT SPD solve = `cholDecomp_InPlace` + 2×`trsv`.
- **`ldlt` / `ldlt_solve`** (L3) — symmetric-**indefinite** LDLᵀ (no sqrt) for KKT/saddle
  systems; non-pivoted now; signature reserves `bool pivot` / `uint32_t* piv` so
  Bunch-Kaufman slots in later with no API change. `iamax` is the pivot primitive.
- **`iamax`** (L1) — BLAS i_amax; deterministic lowest-index tie-break = the mechanism of
  thread-invariance; default / `high_speed::` / `low_memory::` variants; skips NaN
  (documented divergence from numpy).
- **K-way fused** `invertMatrix` / `cholDecomp_InPlace` (L3) — invert/factor K independent
  matrices interleaved over one block (prefix-sum scratch offsets); `inv2`/`inv3`, the
  2-/3-matrix `invertMatrix` wrappers, are now thin wrappers over the K-way form (output
  identical, regression-clean).
- **Warp buildout** — `warp::{dot,axpy,copy,scal,gemv,trsv}` + the composed `warp::posv`,
  closing the L1/L2 glue gap so a complete warp-per-problem solve composes; multi-warp
  (`<<<1,dim3(32,WARPS)>>>`) tests. Also fixed a real `_hash_sources` cache gap (5 L1/L2
  headers weren't hashed → edits silently tested stale binaries).
- **Deferred (API frozen, non-blocking):** Bunch-Kaufman pivoting for `ldlt`/`inv`
  (perf-vs-robust variant), multi-RHS `posv`, `warp::iamax`, the batching-uniformity
  cleanup pass. See the roadmap doc.

## 2026-06-17 — rename + flatten banded/pcg + docs audit

- **Renamed** GLASS = *GPU Linear Algebra Simple Subroutines* (was "for Single-block
  Systems") — decouples the name from "single-block" now that warp-scope is a thing.
  Subheading frames block-scoped-default + the warp expansion.
- **Flattened** the block-tridiagonal API to match the library convention
  (*namespace = scope/backend; function name = operation*):
  `glass::banded::bdmv` → **`glass::bdmv`**, `glass::pcg::solve` → **`glass::pcg`**,
  `glass::pcg::smem_elems` → **`glass::pcg_smem_size`**. Removed the `banded`/`pcg`
  namespaces. (GATO not yet wired to these — confirmed safe; no consumer break.)
- **Docs audit:** reframed the stale "three namespaces" model to **four call
  surfaces** (block-scoped `glass::`/`cgrps::`/`nvidia::` + warp-scoped `glass::warp::`)
  across README + landing + library_overview + agent files; added the solvers as flat
  functions everywhere. Full suite **382 passed**; example + docs build verified.

## 2026-06-17 — warp primitives (PR #15) + GATO banded/PCG merged, docs added

Reconciled two diverged lines onto `main` and documented the new public solvers.
- **PR #15 (`glass::warp::`)** — single-warp SIMT variants (reduce, gemm,
  chol/trsm) for warp-per-problem kernels; live *inline* in the base L1/L3 headers.
- **GATO unification (`glass::bdmv` + `glass::pcg`)** — block-tridiagonal
  matvec (`src/base/banded/bdmv.cuh`) and single-block preconditioned conjugate
  gradient (`src/base/pcg/solve.cuh`) for the block-tridiagonal SPD KKT systems of
  trajectory optimization / MPC. Public (in `glass.cuh`). This is GATO's solver
  re-homed as native GLASS single-block primitives.
- Rebased the banded/pcg branch onto PR #15 (clean, no conflicts), full suite
  **382 passed**, compute-sanitizer clean, merged to `main` (`f904b86`).
- **Docs added (this pass):** API-reference pages `banded.rst` + `pcg.rst`, the
  `concepts/block_tridiagonal.rst` walkthrough (layout + matvec + PCG), an
  `examples/08_pcg_solve.cu`, and agent-file updates (CLAUDE/STARTUP/this).
- Note: the cooperative grid-wide PCG (`glass::cgrps::grid`) is future work.

## 2026-06-15 — box-QP solver validated (internal)

Redesigned, fixed, and validated the orphaned QP solver.
- `src/L3/cpqp.cuh` → **`src/L3/box_qp.cuh`**: honest box-QP
  (`min 0.5xᵀPx+qᵀx s.t. l≤x≤u`), clean struct API (`QPParams`/`QPResult`/
  `box_qp_scratch_size`), solution in-place, `A`/`s_tmp` dropped, proper
  projected-gradient KKT stopping test.
- **Found + fixed a real OOB bug:** the dot accumulators were single scalars but
  `low_memory::dot` needs a length-`n` reduction buffer → out-of-bounds writes
  corrupted the objective. Arena is now `5n`; compute-sanitizer clean.
- **`test/test_qp.py`** (27 cases): KKT optimality, SciPy cross-check, closed-form
  cases, thread-count invariance (1..256), f32 + f64. Full suite **358 passed**.
- Kept **internal** — not in `glass.cuh`/public docs/examples. Whether to promote
  is gated on `docs/open-tasks/qp_solver_scope.md` (QP is optimization, not LA).
- Standalone `src/L3/test_cpqp.cu` removed (replaced by the harness runner
  `test/cuda/test_qp.cu`).

## 2026-06-15 — developer-experience parity + legacy cleanup

Brought GLASS's developer experience up to parity with the GRiD repo
(docs site, website deploy, agent files, examples, project-info) and removed a
stale duplicate source tree.

**Docs site (new).** Sphinx (`pydata-sphinx-theme`) under `docs/`, with the API
reference generated from header Doxygen doc-comments via Breathe.
- `docs/{Makefile,Doxyfile,requirements.txt}`, `docs/source/conf.py`.
- Narrative pages under `docs/source/user_guide/` (getting_started, concepts,
  tutorials) ported from `README.md`, `bench/INSTALL.md`, `bench/TUNING.md`, and
  the batched-1D RFC.
- API reference under `docs/source/api_reference/` (`.. doxygenfile::` per
  header).
- Build: `cd docs && make all` (needs `doxygen` + `pip install -r docs/requirements.txt`).

**Doc-comments (new).** Added Doxygen `/** */` blocks to the public entry points
across `src/base/**`, `src/cgrps/**`, `src/nvidia/**`, and the top-level headers.
`Doxyfile` uses `EXTRACT_ALL = NO`, so undocumented internals are excluded.

**Website (new, LIVE).** `.github/workflows/gh-pages.yml` builds the docs
(installs Doxygen) and deploys to GitHub Pages on push to `main`. Pages is
enabled (Settings → Pages → Source = GitHub Actions) and serving.

**Agent files (new).** `CLAUDE.md`, `docs/agent_debugging_guide.md`,
`docs/STARTUP_PROMPT.md`, this file, and `docs/open-tasks/`.

**Examples (new).** `examples/` — six standalone compilable `.cu` programs
(01–05 pure-SIMT; 06 requires MathDx) + a README.

**Project-info (new).** `CONTRIBUTING.md`, `.clang-format` (advisory),
`test/README.md`, `test/TESTING_STRATEGY.md`, root `.gitignore`.

**CI.** Docs deploy is the only active workflow. `.github/workflows/test.yml` is
a disabled self-hosted-GPU stub (GitHub-hosted runners have no GPU) — enable it
once a `gpu`-labelled runner exists.

**Legacy cleanup (code change).** Deleted the pre-refactor duplicate dirs
`src/L1`, `src/L2`, and the duplicate primitives in `src/L3` (24 files) — they
were superseded by the May-2026 `src/base/**` refactor and nothing included them.
The orphaned `src/L3/cpqp.cuh` box-QP solver (+ `test_cpqp.cu`) was the only
dependent; it was **rewired onto the `base/` API** and now compiles
(`nvcc ... -c src/L3/test_cpqp.cu`, sm_120, clean). It remains **unvalidated and
undocumented** — see `docs/open-tasks/cpqp_validation.md`.

**Untouched (already strong):** the `test/` suite and `bench/` were left as-is
(only added `test/README.md` + `test/TESTING_STRATEGY.md`). The 799-line
`README.md` remains the canonical user-facing reference.

### Verification done
- `nvcc` compile of `test_cpqp.cu` against `base/` (pre- and post-deletion): clean.
- Docs build (`cd docs && make all`) and `pytest test/`: see the latest run notes.

### Next / open
- ~~Enable GitHub Pages in repo settings~~ — DONE; site is live.
- `docs/open-tasks/cpqp_validation.md` — validate or remove cpqp.
- `docs/open-tasks/doc_comment_coverage.md` — optional long-tail doc-comments.
