# `*_reduced` crossover sweep — measured results

Source: `bench/bench_reduced.cu` (200k iters/config, A/B/C shared-resident, pure
compute). Hardware: **RTX 5090 / sm_120**, CUDA 13.2. `ratio = serial_us /
reduced_us` — **> 1 means the contraction-parallel `gemm_reduced` beats serial
`gemm`**; < 1 means serial wins.

## Headline finding

**The contraction-parallel path is almost always slower than serial on this
hardware** — 47 of 48 configurations have ratio < 1, frequently by 10–100×. The
serial `gemm` (one thread per output, tight per-element loop over shared memory)
is very hard to beat at these sizes; the reduced path pays a ~5-step warp-shuffle
latency per output and, for the typical short contraction (K = 14–21), leaves
most of a warp's 32 lanes idle.

The **only** win in the swept space is the extreme corner where the output count
is tiny and the contraction is long:

| M | N | K | n_out | blockDim | serial_us | reduced_us | ratio |
|---|---|---|-------|----------|-----------|------------|-------|
| 2 | 64| 2 |   4   |   128    |  0.1725   |  0.1548    | **1.11** |
| 2 | 64| 2 |   4   |   256    |  0.1734   |  0.1549    | **1.12** |

Everywhere else reduced loses: e.g. the consumer-shaped `14×14×14` (n_out=196)
runs 0.12 µs serial vs 3.68 µs reduced at 256 threads (**32× slower**); the
long-contraction-but-wide `4×4×64` (n_out=16) is still ~50× slower because
n_out (16) exceeds the warp count (8) so each warp serializes several outputs on
top of the shuffle cost.

## Win-condition (what the picker encodes)

**Retired to constant `false` (2026-07-08).** The quiet-GPU resweep measured
**0 of 48** configurations where reduced clears the ±5% tie margin — even the
former long-contraction corner (`n_out ≤ blockDim/32 && K_contract ≥ 32`, the
2×64×2 cells that measured 1.11–1.12× on the earlier shared-GPU sweep)
collapsed into the noise band. `glass::suggested_use_reduced<>()` therefore
returns `false` unconditionally on sm_120; its signature is kept as the seam
for a future data-derived corner on different hardware (e.g. Jetson Orin).
Treat the `*_reduced` family as an expressiveness tool, never a speed default.

## Caveat for the tensor / congruence families

`tensor_vec_contract`, `vec_tensor_vec`, `congruence_sym`, `bilinear` are built on
the same warp-reduce engine, so they inherit the same overhead profile. Their
value is **expressiveness and fusion** (operations the serial BLAS surface cannot
express in one call, fewer launches / barriers), **not** beating a hand-tuned
serial contraction. A consumer optimizing for latency should benchmark against
their own serial code before adopting them for speed.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `reduced_sweep_20260718_0130.txt` · tie margin ±5% (reduced must clear it) · 0 of 48 configs pick reduced._

Predicate `suggested_use_reduced<n_out,K_contract,blockDim>()` = `(n_out <= blockDim/32) && (K_contract >= 32)` (K_contract is the N column here).

✅ Measurement matches the predicate for every swept config — the formula needs no change.

<!-- END tune.py -->

## Reproduce

The `reduced` leg of the unified autotuner remeasures this and refreshes the
"Latest measured run" block above (flagging any config where measurement
disagrees with the `suggested_use_reduced<>` predicate):

```bash
python bench/tune.py --legs reduced --sm auto   # on a quiet GPU
```

Or run the harness directly:

```bash
cd bench && nvcc -std=c++17 -arch=sm_XX -O3 -I.. -I../src bench_reduced.cu -o bench_reduced
./bench_reduced 200000   # run on a quiet GPU
```
