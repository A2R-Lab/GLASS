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

Reduced is competitive only when **both** hold:

1. `n_out ≤ blockDim / 32` — every output gets its own warp, so there is no
   serial output loop layered on top of the shuffle; and
2. `K_contract ≥ 32` — the contraction is long enough to (a) use a full warp's
   lanes and (b) amortize the shuffle tail.

`glass::suggested_use_reduced<n_out, K_contract, blockDim>()` returns `true` only
in that corner and `false` otherwise — i.e. it recommends **serial almost
always**. Treat the `*_reduced` family as a niche tool, not a default.

## Caveat for the tensor / congruence families

`tensor_vec_contract`, `vec_tensor_vec`, `congruence_sym`, `bilinear` are built on
the same warp-reduce engine, so they inherit the same overhead profile. Their
value is **expressiveness and fusion** (operations the serial BLAS surface cannot
express in one call, fewer launches / barriers), **not** beating a hand-tuned
serial contraction. A consumer optimizing for latency should benchmark against
their own serial code before adopting them for speed.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `reduced_sweep_20260626_2135.txt` · tie margin ±5% (reduced must clear it) · 0 of 48 configs pick reduced._

Predicate `suggested_use_reduced<n_out,K_contract,blockDim>()` = `(n_out <= blockDim/32) && (K_contract >= 32)` (K_contract is the N column here).

⚠️ **2 config(s) disagree** with the predicate — review before trusting the formula on this GPU:

- 2×64×2 bd=128 (n_out=4): measured **None**, predicate **reduced**
- 2×64×2 bd=256 (n_out=4): measured **None**, predicate **reduced**

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
