# Tuning GLASS for your hardware

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
gemm_batched_1d, row_strided_gemm, row_strided_gemv). Each can be
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
# → writes src/nvidia/tuning_table.cuh with measurements for the local SM
```

The script:
1. Measures both backends across the (M,N,K) grid for `gemm` at your local SM.
2. Picks the faster path per shape.
3. Emits one explicit `cublasdx_wins<M,N,K,SM>()` specialization per measured shape.
4. Writes the result to `src/nvidia/tuning_table.cuh`.

This **overwrites** the shipped defaults with your machine's measurements.
For a shared checkout where you don't want to commit local tuning, use
the per-build override instead (below).

## Per-build local override

If you want to keep the shipped `tuning_table.cuh` clean (e.g. shared
checkout, CI that builds for multiple targets), supply a per-build
override file:

```bash
nvcc ... -DGLASS_TUNING_TABLE_LOCAL='"path/to/my_overrides.cuh"' ...
```

The named header is `#include`d at the bottom of `_glass_tuning` and may
add specializations for shapes **not already specialized in the in-tree
table**. (C++ disallows re-specialization; to override a shape the in-tree
table already covers, edit `tuning_table.cuh` directly or remove the
in-tree entry first.)

Example `my_overrides.cuh`:

```cpp
// gemm 7×7×7 on sm_8.6 — force cuBLASDx for our compute kernel.
template <> constexpr bool cublasdx_wins< 7,  7,  7, 860>() { return true; }
// gemv 17×17 on sm_8.6 — measured SIMT win not yet upstreamed.
template <> constexpr bool cublasdx_wins_gemv<17, 17, 860>() { return false; }
// batched 4×4×4 × 8 on sm_8.6 — force cuBLASDx (default heuristic agrees,
// but pinning makes the intent explicit even if the heuristic changes).
template <> constexpr bool cublasdx_wins_batched< 4,  4,  4,  8, 860>() { return true; }
```

The override file is *not* expected to live under version control. Put it
wherever your build system finds it (often `bench/tuning/`, which is
gitignored by `bench/.gitignore`).

## Per-API templates

Each API has its own primary template in `_glass_tuning`. Default
heuristics reflect the API's arithmetic intensity:

| Template                                                            | Default heuristic |
|---------------------------------------------------------------------|-------------------|
| `cublasdx_wins<M, N, K, SM>`                                        | `max(M,N,K)>=16 AND min(M,N,K)>=4` |
| `cublasdx_wins_gemv<M, N, SM>`                                      | `max(M,N) >= 32` |
| `cublasdx_wins_batched<M, N, K, BATCH, SM>`                         | `BATCH>=8 AND max(M,N,K)>=8` |
| `cublasdx_wins_row_strided_gemm<M, N, K, A_RS, B_RS, SM>`           | delegates to `cublasdx_wins<>` |
| `cublasdx_wins_row_strided_gemv<M, N, ROW_STRIDE, SM>`              | delegates to `cublasdx_wins_gemv<>` |

`autotune.py` today populates `cublasdx_wins` (gemm). The other variants
fall back to the default heuristic unless you specialize them by hand.
Extending autotune to cover the other APIs is straightforward — see the
existing pattern in `bench/autotune.py`.

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

## Contributing back

If your measurements differ materially from the shipped table:

1. Re-run autotune for the gemm grid:
   ```bash
   python3 bench/autotune.py
   ```
   This overwrites `src/nvidia/tuning_table.cuh`.
2. Inspect the diff vs `git`:
   ```bash
   git diff src/nvidia/tuning_table.cuh
   ```
3. Open a PR with the diff. Include in the description: hostname, GPU
   model from `nvidia-smi`, CUDA toolkit version, MathDx version, bench
   `--iters` count.

### What NOT to contribute

- Entries within 5% of each other (autotune marks these "tie within ±5%
  → SIMT default" — don't second-guess that filter).
- Measurements from a thermally throttled GPU. Run `nvidia-smi -q -d CLOCK`
  before; if the GPU is at its peak boost, you're good.
- Measurements with `--iters` below ~5000 (high variance for sub-microsecond ops).
- Entries for shapes that aren't realistic for any workload (`M=N=K=2` etc.).
