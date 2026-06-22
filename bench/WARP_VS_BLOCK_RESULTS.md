# Warp-per-problem vs one-block-per-problem — scaling results

Measured by `bench/bench_warp_vs_block.cu` (run via `run_warp_block_sweep.sh`).
**RTX 5090, sm_120, 2026-06-21** (boost 3090 MHz). Metric: ns/problem, min of 3
trials. Two packing models across problem size N and batch count NPROB:

- **BLOCK** — one block per problem, `<<<NPROB, TB>>>`, TB ∈ {32,64,128,256}
- **WARP** — one warp per problem, `<<<ceil(NPROB/WPB), dim3(32,WPB)>>>`, WPB ∈ {1..32}

Swept NPROB ∈ {1, 64, 1024, 8192, 32768} (single-problem latency → GPU-saturating
throughput) × N ∈ {4,6,8,12,16,24,32} × ops {dot, gemv, gemm, chol, trsv, posv}.
Raw data: `bench/warp_block_sweep_20260621_2147.txt`.

## Recommended defaults (the headline)

| op | class | throughput default | block `TB` | warp `WPB` | notes |
|------|-------|--------------------|-----------|-----------|-------|
| **dot**  | L1 reduce | **WARP** (huge) | 64 | 8–32 | 2.6× at 8K, **5.6× at 32K problems**; warp scales hardest |
| **gemv** | L2 matvec | **WARP** for N≤24 | 128 | 4–8 | ~tie at N=32; warp 2–5× small N |
| **gemm** | L3 matmul | **warp N≤8 / BLOCK N≥12** | scales 64→256 with N | 2–8 (small N) | the ONLY op that flips to block; block 1.6–1.7× at N=32 |
| **chol** | L3 factor | **WARP** | **32** | 2–4 | warp 1.1–1.8× (esp. small N + big batch); block: TB=32, more *hurts* |
| **trsv** | L3 solve | **WARP** everywhere | **32** | 2–4 | warp 1.5–2.5× across all N and batch sizes |
| **posv** | L3 solve | **WARP** everywhere | **32** | 2 | warp 1.2–1.7× across all N≤32 |

**One-line rule:** for batched small problems, **default to warp-per-problem for
everything except GEMM at N≥12** (the one embarrassingly-parallel-per-problem op that
prefers a whole block once it's big enough). Pack **2–8 warps/block** (8–32 for dot).

## Single problem (NPROB=1) — "if there's only one op, warp or block?"

Launch-overhead-bound (~1.78 µs floor) so for **small N it's a wash** (warp ≈ block).
For **non-tiny N, BLOCK wins** — a lone warp is slow on a big problem while a block
parallelizes it: **gemm block 1.7×→5.75× (N=12→32)**, chol/gemv likewise. Exceptions:
trsv/posv (serial-pivot solves) marginally favor warp even at NPROB=1. **→ Your
hypothesis confirmed: block unless tiny** (and "tiny" = launch-bound, where it doesn't
matter). If the single op is *inside* a larger kernel (no separate launch), use the
throughput-regime guidance above instead.

## Why these splits

- **Reductions / matvec / factor / solve (dot, gemv, chol, trsv, posv)** leave most of
  a block's threads idle — chol/trsv/posv have serial pivot loops, dot is a 1D reduction.
  One warp does ~as much useful work per problem, so packing one problem per warp keeps
  the SMs full. dot scales most dramatically (5.6× at 32K) because it's the cheapest op,
  most dominated by per-block overhead in the block model.
- **GEMM** is fully parallel per problem (N² output cells), so once N≥12 a block's extra
  threads earn their keep and one-block-per-problem wins. Below N≈8–12 the block is
  underutilized and warp packing wins (up to ~5× at huge batch).
- **Block thread-count**: factor/solve want **TB=32** — extra warps just wait on the
  serial pivot loop and add barrier cost (TB=256 is 2–3× slower than TB=32 at NPROB=8192).
  GEMM wants TB to grow with N (tb64 small → tb128/256 at N≥16). dot/gemv: tb64–128.
- **Throughput grows the warp advantage**: warp wins widen from NPROB=1 (launch-bound
  tie) → 1024 → 8192 → 32768 as more problems expose the occupancy difference.

## Feeding the API / next steps

- The MPC/RBD workload GLASS targets is *many small SPD factor/solves* — squarely the
  warp-favored regime. The "Choosing the right backend" docs and any launch helpers
  should make **warp-per-problem the throughput default** (with the GEMM-N≥12 and
  single-problem exceptions called out).
- Block-thread-count defaults (factor/solve → 32; gemm → scale with N) are concrete
  numbers the wrappers can bake in.
- Not yet measured (block-only ops, no warp variant): syrk/ldlt/inv thread-count
  scaling, and the nvidia-SIMT-vs-cuBLASDx axis (that's the separate `bench/autotune.py`).
  See `docs/open-tasks/perf_autotune_breakeven.md` for the full planned matrix.

## Reproduce

This 2-way (warp vs block) data is the MathDx-free build of `bench_mega_sweep.cu`
(formerly `bench_warp_vs_block.cu`). The full **three-way** sweep that adds the
`glass::nvidia` (cuBLASDx/cuSOLVERDx) leg lives in `run_mega_sweep.sh` /
`MEGA_SWEEP_RESULTS.md`.

```bash
# 2-way (warp vs block), no MathDx needed:
nvcc -std=c++17 -arch=sm_120 -O3 -Xptxas -O1 -I.. -I../src bench_mega_sweep.cu -o bms
./bms 8192 500 f32     # throughput ; ./bms 1 3000 f32 → single-problem latency ; f64 also valid

# 3-way (adds nvidia/MathDx), full graduated sweep, ~tens of min, idle GPU:
cd bench && ./run_mega_sweep.sh sm_120
```
