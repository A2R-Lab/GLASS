# Warp-per-problem vs one-block-per-problem — scaling results

Measured by `bench/bench_warp_vs_block.cu` (throughput: NPROB=8192 independent
problems, ns/problem, min of 3 trials × 500 reps). **RTX 5090, sm_120, 2026-06-21.**
Throughput harness (many problems filling the GPU), NOT the single-block latency benches.

Two packing models compared as a function of problem size N:
- **BLOCK** — one block per problem, `<<<NPROB, TB>>>`, TB ∈ {32,64,128,256}
- **WARP** — one warp per problem, `<<<NPROB/WPB, dim3(32,WPB)>>>`, WPB ∈ {1,2,4,8,16,32}

## Crossover summary (best-config each model, ns/problem, winner)

| op   | N=4 | N=6 | N=8 | N=12 | N=16 | N=24 | N=32 | crossover |
|------|-----|-----|-----|------|------|------|------|-----------|
| gemm | **W** 2.5× | **W** 2.25× | **W** 1.9× | B 1.09× | B 1.14× | B 1.11× | B 1.63× | **warp ≤ 8, block ≥ 12** |
| chol | **W** 1.73× | **W** 1.62× | **W** 1.16× | **W** 1.32× | **W** 1.28× | **W** 1.14× | ~tie | **warp ≤ 24, tie at 32** |
| posv | **W** 1.24× | **W** 1.56× | **W** 1.49× | **W** 1.43× | **W** 1.40× | **W** 1.20× | **W** 1.32× | **warp wins everywhere ≤ 32** |

(W = warp-per-problem wins, B = block-per-problem wins; the factor is best-warp vs best-block.)

## Findings / recommended defaults

1. **Factor & solve ops (chol, posv) → default to warp-per-problem** across the whole
   small-matrix regime (N ≤ ~24–32), by 1.2–1.7×. Their serial pivot loops leave most
   of a block's threads idle, so packing one problem per warp (many warps/block) keeps
   the SMs full. posv (chol + 2 trsv, all serial) favors warp at *every* tested size.

2. **GEMM → warp only for tiny N (≤ 8); block for N ≥ 12.** GEMM is embarrassingly
   parallel per problem, so once N is big enough to use a block's threads (N≥12, ~144+
   output cells), one-block-per-problem with a larger thread count wins (up to 1.63× at
   N=32). For N≤8 the block is underutilized and warp packing wins up to 2.5×.

3. **Block thread-count default is op-class-dependent** (the other half of "good defaults"):
   - **Factor/solve (chol, posv): use TB=32.** More threads *hurt* — TB=256 is 2–3×
     slower than TB=32 (serial pivot loop + barrier cost dominates; extra warps just wait).
   - **GEMM: TB grows with N** — TB=64 best at N≤12, TB=128–256 best at N≥16.

4. **Best warps-per-block is small: WPB = 2–8** (not 16–32). 2–4 warps/block is the
   sweet spot for gemm/chol/posv; beyond ~8 the per-block scheduling/occupancy gains
   flatten or regress.

## How this feeds the API

- For consumers packing many small SPD factor/solves (MPC/RBD — exactly GLASS's target),
  the **warp surface is the throughput default**, not one-block-per-problem. Worth saying
  so in the "Choosing the right backend" docs.
- The block-thread-count guidance (factor/solve → 32; gemm → scale with N) is a concrete
  default the wrappers / launch helpers can bake in.

## Reproduce / extend

```bash
cd bench && nvcc -std=c++17 -arch=sm_120 -O3 -Xptxas -O1 -I.. -I../src \
    bench_warp_vs_block.cu -o bench_warp_vs_block && ./bench_warp_vs_block 500
```

The harness covers gemm/chol/posv (matmul / factor / composed-solve). Adding trsv, dot,
gemv, syrk follows the same template (a `kb_<op>`/`kw_<op>` pair + a dispatch case);
expected pattern from the above: serial-ish ops (trsv) track posv (warp-favored), and
elementwise/L1 (dot) favor warp even more strongly at small N. The nvidia-SIMT-vs-cuBLASDx
breakeven axis is the separate, existing `bench/autotune.py` flow (gemm/gemv/lapack).
See also `docs/open-tasks/perf_autotune_breakeven.md` for the full planned matrix.
