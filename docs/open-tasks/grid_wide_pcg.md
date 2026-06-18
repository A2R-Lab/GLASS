# Grid-wide cooperative PCG (`glass::cgrps::grid`)

**Status:** future work (not started).

`glass::pcg` (`src/base/pcg/solve.cuh`) is the **single-block** PCG: one
CUDA block solves one block-tridiagonal SPD system, synchronizing with
`__syncthreads()`. Its file doc-comment points here for the cooperative
**grid-wide** variant — one solve spanning the *whole grid* (a cooperative-groups
`grid.sync()` kernel), which lets a single large system use all SMs.

This is GATO's multi-block PCG (the form used for big horizons where one block's
shared memory / occupancy can't hold the whole problem). It would live under a
`glass::cgrps::grid` namespace and require a cooperative kernel launch
(`cudaLaunchCooperativeKernel`).

## When to pick it up

- When GLASS/GATO needs horizons large enough that the single-block PCG runs out
  of shared memory or under-uses the GPU.
- Reuse the same `[L|D|R]` banded layout and `glass::bdmv`; the dot
  reductions become grid-wide instead of block-wide.

Until then, `glass::pcg` (single-block, one-problem-per-block, batched over
the grid) is the supported path.
