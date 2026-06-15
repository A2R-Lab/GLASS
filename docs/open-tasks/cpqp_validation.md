# Validate (or remove) the cpqp box-QP solver

**Status:** UNVALIDATED. Compiles, not numerically verified, not in the test suite.

`src/L3/cpqp.cuh` is a single-block box-constrained QP solver (projected
gradient + Armijo line search) with a standalone harness `src/L3/test_cpqp.cu`.
It is **not** included by `glass.cuh` and **not** covered by `pytest test/`.

## What happened (2026-06-15)

The legacy `src/L1`, `src/L2`, `src/L3` duplicate header dirs were deleted (they
were pre-`base/`-refactor duplicates that nothing included). cpqp was the only
thing depending on them, so it was rewired onto the current `src/base/**` API:

- Includes repointed to `../base/...` (added `../base/L1/reduce.cuh`,
  `../base/L1/axpy.cuh`).
- `dot` calls → `low_memory::dot(n, x, y, out)` (output now LAST, no thread-group
  arg; `low_memory` is non-destructive — required since inputs are reused).
- `reduce(n, x, g)` → `reduce(n, x)` (no group arg).
- `vector_norm` → `low_memory::vector_norm(dim, x, out)`.
- `matrixAlphaAdd(1, tmp3, q, tmp1, 1, dim)` → `axpy(dim, 1, tmp3, q, tmp1)`
  (`z = 1*x + y`). Numerically equivalent.
- `gemm_v2<T,true,false>(...)` (no base equivalent) → `gemm(dim, dim, 1, 1, P, x, obj_tmp1)`.

**Verified:** `nvcc -std=c++17 -arch=sm_120 -c -I src/L3 src/L3/test_cpqp.cu`
compiles cleanly (object produced).

## What a validator must check

1. **`gemm_v2 → gemm` symmetry assumption (MEDIUM RISK).** The legacy call
   computed `obj_tmp1 = xᵀP`; the replacement computes `P@x`. The objective
   `xᵀPx` is identical **only if P is symmetric** (true for a QP Hessian). If
   cpqp is ever used with non-symmetric `P`, the quadratic term is wrong — add an
   assert/comment or symmetrize.
2. **Pre-existing bug (not introduced by the port):** the convergence check reads
   `tmp6[1]` but `vector_norm` writes the norm to `tmp6[0]`. Likely should be
   `tmp6[0]`.
3. **Dead scratch:** `s_tmp` in `objective_function` is now unused (the base
   `gemm` doesn't need it). Prune the param + its device buffer if kept.
4. **End-to-end numerics:** the solver was never validated even before the port.
   Run `test_cpqp.cu` on a GPU against a reference QP (e.g. scipy/OSQP) for the
   `P`/`q`/box case in the harness; confirm convergence and box feasibility.

## Decision

Either (a) validate it, add it to the pytest suite, and document it; or
(b) remove `src/L3/cpqp.cuh` + `test_cpqp.cu` entirely (recoverable from git).
Until then it stays undocumented in the public API.
