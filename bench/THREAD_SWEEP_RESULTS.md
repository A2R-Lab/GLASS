# Thread vs warp vs block vs nvidia — the low-DOF packing corner

**RTX 5090 (sm_120), 2026-07-18, quiet GPU** — the first ladder sweep with the
`glass::thread::` contender (one problem per THREAD, 32 packed per warp).
Raw data: `bench/mega_sweep_20260719_0234.txt` (ops × N ∈ {4..128} ×
NPROB ∈ {64, 1024, 8192} × {f32, f64}; ALL FOUR contenders over the SAME full
size domain — thread cells past the local-memory ceiling print `FAIL`:
chol/gemv/posv/trsv at N=128 and gemm at N ≥ 96 exceed the per-thread local
limit, and gemm N=16 w16/w32 remain register-infeasible). The dispatch
tables in `glass-defaults.cuh` were regenerated from this capture — the
verdicts below ARE the shipped `ideal_sm120` ladder.

## The headline: thread owns the low-DOF factor/solve corner

Throughput regime (NPROB=8192), thread vs the best other tier
(ratio > 1 = thread faster):

| op | dtype | N=4 | N=6 | N=8 | N=12 | N=16 | N=24 | N=32 | shipped band |
|------|-----|------|------|------|------|------|------|------|--------------|
| posv | f64 | **6.17×** | **7.46×** | **5.03×** | **3.05×** | **2.26×** | **1.19×** | 0.40 | thread ≤ 24 |
| chol | f64 | **5.61×** | **5.79×** | **2.88×** | **2.18×** | **1.71×** | **1.04×** | 0.41 | thread ≤ 24 |
| trsv | f64 | **4.59×** | **4.59×** | **1.97×** | **1.52×** | **1.15×** | 0.23 | 0.11 | thread ≤ 16 |
| dot  | f64 | **2.32×** | **2.33×** | **2.31×** | **1.96×** | **1.53×** | **1.30×** | **1.08×** | thread ≤ 32 |
| posv | f32 | **3.21×** | **3.44×** | **1.73×** | **1.16×** | 0.66 | 0.35 | 0.17 | thread ≤ 12 |
| trsv | f32 | **1.53×** | **2.16×** | **2.03×** | **2.33×** | **1.71×** | 0.21 | 0.10 | thread ≤ 16 |
| chol | f32 | **1.75×** | **1.43×** | 0.72 | 0.53 | 0.35 | — | — | thread ≤ 6 |
| gemv | f32 | **1.06×** | **1.16×** | 0.66 | 0.63 | 0.34 | — | — | thread ≤ 6 |
| gemv | f64 | **1.27×** | **1.07×** | 0.79 | 0.64 | 0.50 | — | — | thread ≤ 6 |
| dot  | f32 | **1.13×** | **1.15×** | **1.16×** | **1.16×** | 0.92 | — | — | thread ≤ 12 |
| gemm | f32/f64 | 0.88/0.61 | 0.41/0.51 | — | — | — | — | — | (never) |

(Full-domain re-sweep 2026-07-19: the previously-unmeasured N=24/32 thread
cells extended three f64 bands — chol/posv to 24, dot to 32; every f32 band
and every ratio at N ≤ 16 reproduced the 07-18 capture. The N ≥ 24 f32 and
N ≥ 48 f64 thread columns document the spill catastrophe: down to 0.1× —
measured, never shipped.)

Readings:
- **The factor/solve chain is where the tier earns its keep** — exactly the
  warp-packing thesis: a warp-per-problem `potrf`/`trsv`/`posv` at N ≤ 7 idles
  most of its lanes on the serial pivot; one-problem-per-thread keeps all 32
  busy. `posv` compounds it (factor + two substitutions, all serial-heavy).
- **f64 amplifies the win and extends it past the register ceiling**: the
  block/warp f64 paths are slow enough at small N that even the SPILLED
  thread path still wins — up to 7.5×, holding through N=24 for chol/posv
  and N=32 for dot.
- **gemm is the anti-case** (as the PR #20 analysis predicted): its
  work-per-element is high enough that the parallel tiers win at every size;
  thread never ships for gemm.
- **f32 dot at tiny N is a mild win (~1.15×)**, gemv f32 a wash at N≤6 —
  launch/memory-bound territory; nothing is lost either way there.
- **The harness now marks infeasible launch configs `FAIL`** instead of timing
  an empty launch (a poisoned first capture briefly credited gemm N=16 warp
  configs with fake sub-ns wins — caught by a total-time floor scan, fixed
  same night; `time_ns_per_prob` checks `cudaGetLastError`).
- The f32 sub-ceiling numbers (N ≤ 6, register-resident) are the fair read of
  the primitive; N ≥ 8 columns price the local-memory spill + the uncoalesced
  per-problem-contiguous staging (deliberately IN the timing — see the harness
  note in `bench_mega_sweep.cu`).

History: the 2026-07-16 A5000 interim analysis (this file's previous content)
could not resolve the throughput regime (%.2f ns prints quantized it, and that
box had no working MathDx). Both harness issues were fixed before this sweep
(%.4f prints; nvidia column live).
