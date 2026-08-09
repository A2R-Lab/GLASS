# Warp-scope vendor A/B: `glass::warp::` vs `glass::nvidia::warp::` (CUB WarpReduce)

Harness: `bench_nvwarp_l1.cu` — the three ops the vendor warp tier ships
(`reduce` / `dot` / `nrm2`), both legs at the identical launch shape (8 warps
per block, ONE problem per warp, full warps), correctness-gated against a host
reference before any timing. N ∈ {4…256}, f32+f64, NPROB ∈ {64, 1024, 8192}.

## sm_87 (Jetson AGX Orin, 50 W, clocks pinned, 2026-08-04)

Raw capture: `glass-paper/data/jetson/nvwarp_l1_50w_20260804_nvwarp.txt`.

**110 / 126 cells tie within ±2%.** The 16 non-ties are all ≤10%: CUB takes 12
(clustered at small-N f64 `dot`, NPROB=1024 — worst 10.2% at N=16, typical
2–6%; plausibly our broadcast `dot` pays one extra `__shfl_sync` that CUB's
lane-0-only reduction skips), SIMT takes 4 (f32 `nrm2` N=64/128, ≤4%). The
NPROB=64 section is launch-latency-bound (~80 ns floor) and ties everywhere
by construction.

**Verdict: the two warp tiers are the same algorithm and measure like it.**
No cell moves outside the ladder's tie band in a way that would justify a
warp-scope vendor dispatch target — which is the measured justification for
the dispatch ladder not descending below block scope for the `nvidia` tier
(`glass::nvidia::warp::` remains an explicit, contract-tier choice).

## sm_120 (RTX 5090, quiet window, 2026-08-06)

Raw capture: `glass-paper/data/sm120/nvwarp_l1_sm120_20260806_0146.txt`.

**119 / 126 cells tie within ±2%** — even more tie-heavy than sm_87. The 7
non-ties: CUB takes 4 (small-N f32 `dot` at NPROB=8192, ≤3.8%), SIMT takes 3
(≤3% at throughput; one 17% `reduce` N=4 outlier in the launch-latency-bound
NPROB=64 section). Same verdict on both measured architectures: the warp
tiers are the same algorithm, and no cell justifies a warp-scope vendor
dispatch target.
