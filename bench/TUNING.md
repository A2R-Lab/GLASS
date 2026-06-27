# Tuning GLASS for your hardware

## One command — `bench/tune.py`

GLASS ships three measured defaults tables: the warp/block/nvidia **backend
ladder** (`glass-defaults.cuh`, consumed by `glass::suggested_backend<>`), the
per-(M,N,K) **cuBLASDx-vs-SIMT table** (`src/nvidia/tuning_table.cuh`, this
document's main subject), and the serial-vs-reduced **`suggested_use_reduced<>`**
predicate. `bench/tune.py` remeasures all of them on your GPU and regenerates
them under **one shared noise margin**, so nothing bakes sub-noise jitter:

```bash
python bench/tune.py --sm auto --prebuild --build-jobs 6   # compile everything in parallel (no GPU needed)
python bench/tune.py --sm auto              # all legs, ±5% margin (reuses the prebuilt cache)
python bench/tune.py --sm auto --quick      # ladder throughput point only (faster)
python bench/tune.py --legs ladder,reduced  # pick legs; --margin 0.05 to retune the tie band
python bench/tune.py --sm auto --dry-run    # regenerate + diff, write nothing
```

**Prebuild so the sweep is fast.** Compilation — not timing — dominates the wall
clock (the `shapes` leg alone compiles ~66 separate cuBLASDx microbenches).
`--prebuild` compiles every binary the selected legs need into a persistent,
hash-keyed cache (`bench/.tune_cache/sm<sms>/`, gitignored) and runs nothing.
Run it **anytime — even while the GPU is busy**, since compilation is CPU-bound.
Building isn't timed, so fan it out with **`--build-jobs N`** (size to free_RAM/7
— each cuBLASDx compile needs ~6-7GB; `--build-jobs 6` on a 64GB box cut a full
re-sweep's compile wall from ~45min serial to ~10min). The later timed sweep on a
quiet GPU then finds every binary cached and is **execute-only** — and always runs
serially for clean measurement, regardless of `--build-jobs`. The cache is keyed
on the rendered source + a digest of the whole header library + the SM, so any
library edit transparently rebuilds the affected binaries; a cuBLASDx-rejected
shape is remembered so it isn't retried.

The shared rule (`bench/tune_pick.py::pick`): a dependency-carrying impl
(`nvidia`/`cublasdx`/`reduced`) wins **only if it beats the simplest impl by more
than the margin** — otherwise the no-dependency path (always launchable, no
MathDx) stays. Every op is measured and recorded; a dispatch picker is
regenerated only where ≥2 impls genuinely compete. **Run on a quiet GPU** — perf
timing must be isolated from other CPU/GPU load. Use `--dry-run` first to confirm
a re-run only moves dispatch inside the tie band before committing a regenerated
table. The `shapes` leg below is the per-shape engine `tune.py` drives.

## The cuBLASDx-vs-SIMT table

GLASS's `glass::nvidia::*` wrappers — `gemm`, `gemv`, `row_strided_*`,
`gemm_batched_1d` — auto-dispatch between a pure-SIMT path and cuBLASDx at
compile time. The decision lives in
`src/nvidia/query_simt.cuh::should_use_cublasdx*<>()` and consults, in order:

1. A per-shape specialization in `src/nvidia/tuning_table.cuh` if one exists
   (compile-time template specialization — zero runtime cost).
2. A per-build local override included by `tuning_table.cuh` when
   `GLASS_TUNING_TABLE_LOCAL` is defined.
3. A static per-API heuristic for unmeasured shapes.

Five per-API decision templates live in `_glass_tuning` (gemm, gemv,
gemm_batched_1d, gemm_strided, gemv_strided). Each can be
specialized independently for a given (shape, SM).

## Why bother?

A representative measurement (RTX 3080, sm_120):

| Shape          | Heuristic says | Measured winner | Speedup |
|----------------|----------------|-----------------|---------|
| gemm 14×14×14  | SIMT           | SIMT            | matches |
| gemm 24×24×24  | cuBLASDx       | **cuBLASDx**    | 2.4×    |
| gemm 6×6×6     | SIMT           | **SIMT**        | 2.3×    |
| gemv 5×5       | SIMT           | **SIMT**        | matches |

For shapes well-covered by the in-tree table this is "free perf". For
unmeasured shapes, you trust the heuristic; once you bench it, you can
specialize it and either keep it local or PR upstream.

## Quick start

```bash
cd GLASS
python3 bench/autotune.py
# → measures all 5 round-2 primaries (gemm, gemv, gemv_strided,
#   gemm_strided, gemm_batched_1d) across each one's default shape grid
# → writes bench/tuning/<hostname>.cuh with the per-host specializations
```

The script:
1. Detects your local SM via `nvidia-smi`.
2. For each requested API, measures both backends across that API's
   shape grid.
3. Picks the faster path per (shape, SM).
4. Emits one explicit specialization per measured shape into
   `bench/tuning/<hostname>.cuh`.

The shipped `src/nvidia/tuning_table.cuh` is **never overwritten** by
the default flow — it carries the per-API primary templates, the
default heuristics, and a small curated set of in-tree specializations,
and must stay stable so consumers can rely on it as the baseline.

## Consuming your per-host overrides

The default per-host output file is designed to be included via the
round-2 `GLASS_TUNING_TABLE_LOCAL` macro:

```bash
nvcc ... -DGLASS_TUNING_TABLE_LOCAL='"bench/tuning/<hostname>.cuh"' ...
```

The named header is `#include`d at the bottom of `_glass_tuning` and may
add specializations for shapes **not already specialized in the shipped
table**. (C++ disallows re-specialization; to override a shape the
shipped table already covers, edit `tuning_table.cuh` directly or
remove the in-tree entry first.)

Per-host files under `bench/tuning/` are gitignored by `bench/.gitignore`.

## Per-API templates

Each API has its own primary template in `_glass_tuning`. Default
heuristics reflect the API's arithmetic intensity:

| Template                                                            | Default heuristic |
|---------------------------------------------------------------------|-------------------|
| `cublasdx_wins<M, N, K, SM>`                                        | `max(M,N,K)>=16 AND min(M,N,K)>=4` |
| `cublasdx_wins_gemv<M, N, SM>`                                      | `max(M,N) >= 32` |
| `cublasdx_wins_batched<M, N, K, BATCH, SM>`                         | `BATCH>=8 AND max(M,N,K)>=8` |
| `cublasdx_wins_gemm_strided<M, N, K, A_RS, B_RS, SM>`           | delegates to `cublasdx_wins<>` |
| `cublasdx_wins_gemv_strided<M, N, ROW_STRIDE, SM>`              | delegates to `cublasdx_wins_gemv<>` |

`bench/autotune.py` (round-2 rewrite) covers all five. To restrict to a
subset:

```bash
python3 bench/autotune.py --apis gemm,gemv
python3 bench/autotune.py --apis gemv_strided --shapes "6,6,8;14,14,16"
```

The `--shapes` flag passes a `;`-separated tuple list; the arity has to
match the chosen API (3 values for `gemm`, 2 for `gemv`, etc.). If you
list multiple APIs and `--shapes` matches one but not all, the
non-matching APIs are skipped with a one-line note (the matching APIs
still run).

## Debugging dispatch decisions

Use the per-API `print_dispatch_*` host helpers from `query_simt.cuh`:

```cpp
#include "glass-nvidia.cuh"

int main() {
    glass::nvidia::print_dispatch<float, 6, 6, 6>();
    // → "glass::nvidia::gemm<T,6,6,6,SM=860>: SIMT fallback"
    glass::nvidia::print_dispatch_gemv<float, 64, 64>();
    // → "glass::nvidia::gemv<T,64,64,SM=860>: cuBLASDx (needs DEFINE_NVIDIA_GEMV*)"
}
```

These are `__host__ __device__` so you can drop one into a kernel for
runtime diagnostics, or call from main for build-time confirmation.

## Contributing upstream

If your measurements would meaningfully improve the shipped table
(e.g. SM not yet covered, or a shape range the curated entries miss),
contribute back. Two routes:

### Option A — submit your per-host file unchanged

The simplest contribution: rerun autotune, then attach the contents of
`bench/tuning/<hostname>.cuh` to a PR. Reviewers will spot-check and
merge specific specializations into `src/nvidia/tuning_table.cuh` as
appropriate.

### Option B — update the shipped table directly

For maintainers or contributors who want to commit specializations
straight into the shipped file:

```bash
python3 bench/autotune.py --sm AUTO --in-tree
```

`--in-tree` writes the new specializations into a marker-delimited
section inside `src/nvidia/tuning_table.cuh` while preserving the round-2
primary templates, default heuristics, and the `GLASS_TUNING_TABLE_LOCAL`
hook. The markers are:

```
// === BEGIN: autotune-generated specializations ===
// ...
// === END: autotune-generated specializations ===
```

Re-running `--in-tree` replaces the section in-place; running without
`--in-tree` writes only to `bench/tuning/<hostname>.cuh` and leaves the
shipped table alone.

### What NOT to contribute

- Entries within 5% of each other (autotune marks these "tie within ±5%
  → SIMT default" — don't second-guess that filter).
- Measurements from a thermally throttled GPU. Run `nvidia-smi -q -d CLOCK`
  before; if the GPU is at its peak boost, you're good.
- Measurements with `--iters` below ~5000 (high variance for sub-microsecond ops).
- Entries for shapes that aren't realistic for any workload (`M=N=K=2` etc.).
