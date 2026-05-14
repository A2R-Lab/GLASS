# Tuning GLASS for your hardware

GLASS's `glass::nvidia::*` wrappers — `gemm`, `gemv`, `row_strided_*`,
`gemm_batched_1d` — auto-dispatch between a pure-SIMT path and cuBLASDx
at compile time. The decision lives in
`src/nvidia/query.cuh::should_use_cublasdx*<>()` and consults, in order:

1. The local override table (`kLocalTable`, opt-in via `GLASS_TUNING_TABLE_LOCAL`).
2. The shipped global table (`kGlobalTable` in `src/nvidia/tuning_table.cuh`).
3. A static heuristic (e.g. `max(M,N,K) >= 16 AND min(M,N,K) >= 4` for `gemm`).

The shipped global table grows over time as community contributors run
the autotune script and PR their measurements. Until your shape is in
that table, you get the heuristic — which is reasonable on average but
often suboptimal for any one (shape, SM) combination.

## Why bother?

A representative real measurement (RTX 3080, sm_120, `--iters 100`):

| Shape        | Heuristic says | Measured winner | Speedup |
|--------------|----------------|-----------------|---------|
| gemm 14×14×14  | SIMT          | **cuBLASDx**    | 1.14×   |
| gemm 24×24×24  | cuBLASDx      | **SIMT**        | 1.19×   |
| gemv 4×4       | SIMT          | **cuBLASDx**    | 1.40×   |
| gemv 64×64     | cuBLASDx      | **SIMT**        | 2.85×   |

Free wins, for as long as you keep using that hardware.

## Quick start

```bash
cd GLASS
python3 bench/autotune.py
```

The script:
1. Runs `bench/run_bench.py` end-to-end (takes a few minutes).
2. Parses the resulting JSON.
3. Picks the faster path per `(api, shape, SM)`.
4. Writes a header at `bench/tuning/tuning_<hostname>_sm<NN>.cuh`.
5. Prints a one-line incantation to add to your nvcc command.

Then add the printed `-DGLASS_TUNING_TABLE_LOCAL=...` flag to your build.
Your local override takes effect — `should_use_cublasdx*<>()` checks it
first, before the shipped global table or the heuristic.

`bench/tuning/` is gitignored by default — your per-hardware files don't
pollute version control.

## Options

```bash
# Use a smaller iteration count for a faster pass (results are noisier).
python3 bench/autotune.py --iters 1000

# Skip running the bench again; parse an existing results JSON.
python3 bench/autotune.py --reuse-results bench/results/bench_<hostname>.json

# Also print a "patch" suggesting which entries to PR upstream.
python3 bench/autotune.py --emit-pr-diff

# Tighten the noise threshold — only emit entries with > 10% gap.
python3 bench/autotune.py --threshold-pct 10
```

The default 5% threshold filters out measurements where SIMT and cuBLASDx
are within 5% of each other — those decisions could flip on a different
machine with the same SM, and we'd rather rely on the heuristic.

## Contributing back to the global table

If you ran autotune and found entries that *disagree* with the shipped
global table (or filled in shapes the global table doesn't yet cover),
you can upstream them via PR:

1. Run with the diff flag:
   ```bash
   python3 bench/autotune.py --iters 10000 --emit-pr-diff > my_patch.txt
   ```
2. Inspect `my_patch.txt`. Entries marked `[new]` go straight in;
   entries marked `[DISAGREES]` deserve a closer look — they imply the
   shipped table is wrong for your hardware (or for yours-and-similar).
3. Open a PR adding the entries to the `kGlobalTable` initializer in
   `src/nvidia/tuning_table.cuh`. Include a comment block above your
   entries with:
   ```
   // <hostname> sm_X.Y, GPU model, CUDA toolkit version, MathDx version,
   // bench iter count. Optional: `nvidia-smi` GPU-clock setting.
   ```
4. Maintainers will sanity-check the entry against existing measurements
   and merge.

### What NOT to contribute

- Entries below 5% relative gap (the default autotune threshold already
  filters these — don't lower it before contributing).
- Measurements taken with `--iters` below ~5000 (high variance).
- Measurements on a thermally throttled GPU. Run `nvidia-smi -q -d CLOCK`
  before benching; if the GPU is at its peak boost clock, you're good.
- Entries for shapes that are unrealistic for real workloads (e.g.
  `M=N=K=2`). The tuning table is finite — keep it useful.

## Schema

Both tables (`kGlobalTable` and `kLocalTable`) use the same `entry` struct
defined in `src/nvidia/tuning_table.cuh`:

```cpp
struct entry {
    api      op;            // gemm, gemv, row_strided_gemm, row_strided_gemv, gemm_batched_1d
    uint16_t M, N, K, BATCH, SM;
    bool     use_cublasdx;
};
```

- `BATCH = 1` for non-batched APIs.
- `SM` is the compute capability times ten (`8.6` → `86`, `12.0` → `120`).
- A row is matched only when ALL of `(op, M, N, K, BATCH, SM)` agree
  exactly — there is no nearest-neighbor fallback. Shapes not in either
  table fall through to the per-API static heuristic.

## Implementation notes

- The bench measures `glass::gemm<CT> (shared)` against `cuBLASDx gemm (shared)`.
  These are the closest comparable variants — both read pre-staged inputs
  from shared memory, which matches how `glass::nvidia::gemm<>` operates
  inside a real kernel.
- `bench_gemm_batched_1d.cu` measures the new round-2 1D-launch batched
  API. It's separate from `bench_gemm_batched.cu` (the older 2D-launch
  variant), and `autotune.py` only consumes the 1D labels.
- Adding a new API to autotuning requires three changes: a new entry in
  the `api` enum in `tuning_table.cuh`, a matching `should_use_cublasdx_*`
  in `query.cuh`, and a classifier branch in `autotune.py::classify()`.
