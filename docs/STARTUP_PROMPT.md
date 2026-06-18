# Startup prompt — joining GLASS

A 60-second primer for an agent (or human) picking up work on GLASS.

**What GLASS is:** a header-only CUDA library of BLAS/LAPACK-style `__device__`
routines that run inside a single thread block (one block per problem). It backs
[GRiD](https://github.com/A2R-Lab/GRiD).

**Read these first, in order:**
1. `CLAUDE.md` (repo root) — the mental model, layout, build/test commands.
2. `docs/agent_debugging_guide.md` — the single-block CUDA bug-class runbook.
   **Read before changing any primitive.**
3. `README.md` — the full user-facing API tour (L1/L2/L3, backend dispatch,
   `TRAILING_SYNC`).

**Where things live:**
- Public API: `src/base/{L1,L2,L3}/` (→ `glass::`, plus inline `glass::warp::`),
  `src/base/banded/` + `src/base/pcg/` (→ `glass::bdmv` / `glass::pcg`),
  `src/cgrps/` (→ `glass::cgrps::`), `src/nvidia/` (→ `glass::nvidia::`).
- Internal (not in `glass.cuh`): `src/L3/box_qp.cuh` (validated box-QP solver).
- Tests: `test/` (pytest) driving `test/cuda/*.cu`; `pytest test/`.
- Benchmarks: `bench/` (`run_bench.py`, `autotune.py`).
- Docs: `docs/` (Sphinx + Doxygen + Breathe; `cd docs && make all`).
- Backlog: `docs/open-tasks/`.

**The one rule that catches most bugs:** primitives must be thread-count
invariant — test at 1 / 32 / partial-warp / many-warp block sizes, and put a
`__syncthreads()` between any write phase and a dependent read.

**Current state (2026-06-17):** `main` carries the docs-parity infra, the
validated (internal) `box_qp` solver, the `glass::warp::` single-warp primitives
(PR #15), and the GATO-unification `glass::bdmv` + `glass::pcg`
block-tridiagonal solvers. Full suite **382 passing**. See `docs/HANDOFF.md` for
the latest entry.
