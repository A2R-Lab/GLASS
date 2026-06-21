# Backlog: perf autotune / breakeven characterization (quiet GPU)

**Filed 2026-06-21.** Run a dedicated benchmarking + autotune sweep on a **quiet GPU**
(no other load — matches the isolate-timing rule: one config at a time, GPU idle) to
produce *guidance* on performance and **breakeven points across batch size and problem
size** for the various API options. This is measurement/guidance, not a code change.

## Why
The library now has four call surfaces (block `glass::` SIMT, `glass::cgrps::`,
`glass::nvidia::` CUB/cuBLASDx/cuSOLVERDx, and warp `glass::warp::`) plus structured/
fused options. Callers currently choose by rule-of-thumb (the README "Choosing the right
backend" table + the `src/nvidia/tuning_table*` heuristics). We have no measured map of
**where each surface actually wins** as a function of problem size and batch (block) count.

## What to sweep
- **Axes:** problem size (matrix dim / vector length) × batch = number of blocks (1 → many,
  to fill the GPU) × dtype (f32, f64) × call surface.
- **Op families:** `gemm`, `gemv`, `dot`/`reduce`, `chol`, `trsv`/`trsm`, `inv`, `syrk`/`syr2k`,
  `ldlt`, `posv`/`potrs`, `pcg`/`bdmv`, and the **K-way fused** invert/chol (vs N separate
  single-matrix calls) + **warp-per-problem** packing (one warp per problem, many per block)
  vs one block per problem.
- **Surfaces per op:** `glass::` vs `glass::cgrps::` vs `glass::nvidia::` (where it exists)
  vs `glass::warp::` (where it exists).

## Deliverables
1. **Breakeven tables/plots:** for each op, the size/batch crossover where the winning
   surface changes (e.g. SIMT → cuBLASDx as dim grows; one-block-per-problem → warp-per-problem
   as problems get small and numerous; single-matrix → K-way fused as batch grows).
2. Feed results into the README **"Choosing the right backend"** guidance and the
   `src/nvidia/tuning_table*` thresholds (the existing autotune covers the round-2 nvidia
   primaries; extend the guidance to the new ops + the warp/fused decisions).
3. Note the GPU/SM the numbers were measured on (small-size perf is highly SM-dependent).

## Methodology guardrails
- Quiet GPU, isolate timing (one config at a time), warm up / discard first iters, fixed
  clocks if possible, report median + spread. Do NOT run other compiles/benches concurrently
  (contention skews results). Precompile/JIT-warm any vendor path before timing it.
