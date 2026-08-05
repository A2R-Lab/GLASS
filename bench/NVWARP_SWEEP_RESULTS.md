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

## sm_120 (RTX 5090) — PENDING

Binary compiles clean; run blocked on a quiet-GPU window.
