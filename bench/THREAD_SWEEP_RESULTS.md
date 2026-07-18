# Thread vs warp vs block vs nvidia — the low-DOF packing corner

**RTX 5090 (sm_120), 2026-07-18, quiet GPU** — the first ladder sweep with the
`glass::thread::` contender (one problem per THREAD, 32 packed per warp).
Raw data: `bench/mega_sweep_20260718_1021.txt` (ops × N ∈ {4..128} ×
NPROB ∈ {64, 1024, 8192} × {f32, f64}; thread column at N ≤ 16). The dispatch
tables in `glass-defaults.cuh` were regenerated from this capture — the
verdicts below ARE the shipped `ideal_sm120` ladder.

## The headline: thread owns the low-DOF factor/solve corner

Throughput regime (NPROB=8192), thread vs the best other tier
(ratio > 1 = thread faster):

| op | dtype | N=4 | N=6 | N=8 | N=12 | N=16 | shipped band |
|------|-----|------|------|------|------|------|--------------|
| posv | f64 | **6.20×** | **7.46×** | **5.04×** | **3.03×** | **2.27×** | thread ≤ 16 |
| chol | f64 | **5.63×** | **5.78×** | **2.97×** | **2.18×** | **1.70×** | thread ≤ 16 |
| trsv | f64 | **4.60×** | **4.60×** | **1.97×** | **1.51×** | **1.15×** | thread ≤ 16 |
| dot  | f64 | **2.66×** | **2.66×** | **2.48×** | **2.07×** | **1.58×** | thread ≤ 16 |
| posv | f32 | **3.18×** | **3.44×** | **1.73×** | **1.24×** | 0.67 | thread ≤ 12 |
| trsv | f32 | **1.63×** | **2.15×** | **2.03×** | **2.36×** | **1.70×** | thread ≤ 16 |
| chol | f32 | **1.75×** | **1.43×** | 0.72 | 0.53 | 0.35 | thread ≤ 6 |
| gemv | f32 | **1.06×** | **1.16×** | 0.66 | 0.63 | 0.34 | thread ≤ 6 |
| gemv | f64 | **1.27×** | **1.07×** | 0.79 | 0.64 | 0.50 | thread ≤ 6 |
| dot  | f32 | **1.13×** | **1.15×** | **1.16×** | **1.16×** | 0.92 | thread ≤ 12 |
| gemm | f32/f64 | 0.88/0.61 | 0.41/0.51 | — | — | — | (never) |

Readings:
- **The factor/solve chain is where the tier earns its keep** — exactly the
  warp-packing thesis: a warp-per-problem `potrf`/`trsv`/`posv` at N ≤ 7 idles
  most of its lanes on the serial pivot; one-problem-per-thread keeps all 32
  busy. `posv` compounds it (factor + two substitutions, all serial-heavy).
- **f64 amplifies the win and extends it past the register ceiling**: the
  block/warp f64 paths are slow enough at small N that even the SPILLED
  thread path (N = 8..16 > the N≤7 register ceiling) still wins — up to 7.5×.
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
