# Thread vs warp vs block — the low-DOF packing corner

`glass::thread::` (one problem per THREAD, 32 packed per warp) only overlaps with
warp/block over its measured range (compile-time `N`, ceiling `N<=7`; see
`CLAUDE.md`). `bench_mega_sweep.cu` extends the THREAD column up to N=16 anyway —
its own header comment flags this explicitly: *"THREAD stages operands
global->registers->global in the per-problem-contiguous layout (uncoalesced; the
layout tax is IN the timing)"*. So N=12/16 numbers below are penalized by that
uncoalesced staging cost and are not a fair read of the primitive itself — only
N=4/6/8 sit inside `thread::`'s documented sweet spot.

Data: `bench/mega_sweep_20260716_1101.txt` (RTX A5000, sm_86), ops
{dot, gemv, gemm, chol, trsv, posv} × N ∈ {4,6,8,12,16} × {f32, f64}. The NVIDIA
column in this particular run never produced a valid timing (a separate MathDx
runtime issue on this box — see the mega-sweep doc/investigation, not a THREAD
concern), so this note is BLOCK/WARP/THREAD only.

## NPROB=1 and NPROB=64 — the only regimes with usable resolution

The sweep prints ns/problem at 2 decimal places. At NPROB=1 (~80ns/problem) and
NPROB=64 (~1.2-1.35ns/problem) that gives enough significant figures to compare
backends. At NPROB=1024 and above, all three backends round to the *same*
printed value (e.g. `0.09`/`0.09`/`0.09` at NPROB=1024, `0.01`/`0.01`/`0.01` at
NPROB=8192) — the metric has quantized away, so those regimes say nothing about
which backend actually wins and are excluded here. Re-running with a raw
(un-rounded) output format would be needed to compare backends in the
throughput regime.

### Win counts, N ∈ {4,6,8,12,16} (includes the two over-range N's above)

| NPROB | dtype | BLOCK | WARP | THREAD |
|-------|-------|-------|------|--------|
| 1     | f32   | 9     | 9    | 12     |
| 1     | f64   | 12    | 9    | 9      |
| 64    | f32   | 7     | 15   | 8      |
| 64    | f64   | 12    | 12   | 6      |

No backend dominates — wins split roughly evenly across all three, op-by-op and
N-by-N, which is the expected signature of a **launch-overhead-bound regime**:
at NPROB=1/64 none of these ops do enough work to separate the three SIMT
strategies; the "winner" per cell is mostly noise.

### Thread vs best-of-{block,warp}, restricted to the in-spec N ∈ {4,6}

Average ratio of `min(block,warp) / thread` across all six ops (>1 means thread
was faster):

| NPROB | dtype | ratio | reading |
|-------|-------|-------|---------|
| 1     | f32   | 0.99  | tied within ~1% |
| 1     | f64   | 1.00  | tied |
| 64    | f32   | 0.98  | tied within ~2% |
| 64    | f64   | 0.98  | tied within ~2% |

## Takeaway

At batch counts where `thread::`'s 32-per-warp packing should matter most for
latency-bound work (NPROB=1, NPROB=64), THREAD is statistically indistinguishable
from WARP/BLOCK in this run — all differences are within ~1-2%, i.e. sweep noise,
not a real performance edge either way at N≤7. This is consistent with the
ops here being tiny enough that kernel-launch and memory-round-trip overhead
dominate regardless of which SIMT tier runs the math; `thread::`'s real value
proposition (packing 32 independent low-DOF problems into one warp instead of
leaving ~26/32 lanes idle) shows up in *throughput*, not this launch-bound
corner — which is exactly the regime this run couldn't resolve (see above).
A rerun with higher NPROB and unrounded output is needed to see the THREAD
throughput story the way `bench_reduced.cu`/`REDUCED_SWEEP_RESULTS.md` does for
`gemm_reduced`.
