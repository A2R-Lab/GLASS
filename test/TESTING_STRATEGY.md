# GLASS testing strategy

Why the GLASS tests are shaped the way they are. If you are adding a primitive or
a test, read this first — a test that passes while a real bug hides is worse than
no test. (The companion runbook is `docs/agent_debugging_guide.md`.)

GLASS primitives are **single-block** kernels: one thread block, threads
cooperating over shared/global data with strided loops
`for (i = rank; i < n; i += size)` and `__syncthreads()` between dependent
phases. Almost every GLASS bug is a violation of one invariant — **thread-count
invariance** — so the tests are built to attack it.

## Principle 1 — Sweep block thread counts, including a non-multiple of 32

A single-block kernel must produce **identical output for any block size**. The
most common bug is a **missing `__syncthreads()` between a write phase and a
later read/accumulate phase**, or multiple threads `+=`-ing into one shared
destination. Both are **invisible at 32 threads**: one warp runs in lockstep
(warp-synchronous), so the kernel "accidentally" behaves as if synchronized. The
same kernel races at 2+ warps.

Therefore tests must launch each op at **more than one** block size:

- `1` thread (the serial baseline),
- `32` (one warp — the warp-synchronous case),
- a multi-warp count (e.g. 256, the production-like case),
- and ideally a count that is **not a multiple of 32**, so a trailing partial
  warp is always present.

A kernel that disagrees across thread counts has a race or a missing barrier —
fix the kernel, never "pick a thread count that passes."

## Principle 2 — Compare against an independent reference, with scale-aware tolerance

Each CUDA result is checked against a NumPy/SciPy computation of the same
operation (not against GLASS itself). Because the GPU runs `float32` and the
reference runs `float64`, use a tolerance that **scales with the array**: floor
the absolute tolerance at `rtol * max(|expected|)` so a structurally-zero entry
carrying `~rtol·scale` round-off counts as close, while a genuine error
(`O(scale)`) still fails. Never widen a tolerance to hide a discrepancy that
grows with input magnitude — that is a real bug.

## Principle 3 — Test both storage orders and the `beta` edges

- **Layout flags** (`ROW_MAJOR_A/B/C`, `TRANSPOSE_B`) are compile-time and
  silently produce wrong numbers (not a crash) if data is passed in the wrong
  order. Test both row- and column-major where a function supports them.
- **`beta = 0`** GEMM/GEMV still *reads* `C` in some paths (`beta*C` with
  `beta=0` is `0*C`, but `0*NaN = NaN`). Test `beta = 0` with an initialized `C`,
  and remember the caller must initialize `C` even when `beta = 0`.

## Principle 4 — Gate genuinely ill-defined inputs instead of loosening tolerance

Inverse / Cholesky / solves are undefined for singular or non-SPD inputs and
`float32` will overflow to `NaN`. Skip such configurations via an explicit
condition (e.g. an SPD / invertibility check on the generated input) rather than
relaxing the tolerance for everyone.

## Checklist for adding a primitive + its test

1. Implement it; keep it thread-count invariant (strided loops, barriers between
   dependent phases).
2. Add a CUDA test source under `test/cuda/` and a pytest wrapper. **Register a
   new `.cu`'s headers in the compile-cache list in `conftest.py`** or the cache
   won't rebuild.
3. Launch it at ≥3 thread counts including 1 and a multi-warp count.
4. Compare to a NumPy/SciPy reference with a scale-aware tolerance; test both
   layouts and `beta = 0` where relevant.
5. If a mismatch appears only above one warp → a `__syncthreads`/accumulation
   race. Fix the kernel; do not mask it.
