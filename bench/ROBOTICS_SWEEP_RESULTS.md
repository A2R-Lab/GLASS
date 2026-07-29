# Robotics micro-op sweep — verdicts

Capture: `robotics_sweep_20260729_0037.txt` (plancher-omen-26, RTX 5080 class
sm_120, quiet GPU verified empty; harness `bench_robotics.cu`, prebuilt
`build/bench_robotics`). Grid: NPROB ∈ {256, 1024, 4096, 16384, 32768} ×
{f32, f64}; reps 500 (200 at NPROB ≥ 16384 — per-launch time is ms-scale
there, reps stop mattering); each point = min of 3 trials; plus a full repeat
pass at NPROB=4096/f32 for the noise floor.

**Noise floor** (repeat pass, 160 point pairs): median 0.40%, p90 0.87%,
worst 7.6% (one thread-tier softmax cell). Sub-5% deltas below are reported
as wash unless consistent across configs.

**Launch-overhead floor**: at NPROB=256 every op times ≈6.9 ns/problem f32
(= ~1.76 µs/launch) — batch counts below ~1k are LAUNCH-BOUND and tier choice
is irrelevant there. All tier verdicts below are from the saturated regime.

## Verdict 1 — fused vs composed: a WASH at the best tier (the intended result)

Best-config composed/fused ratio (f32; >1 = fused faster):

| NPROB | mcross | fcross | mxform | sinertia |
|---|---|---|---|---|
| 1024 | 1.05× | 1.06× | 1.24× | 1.23× |
| 4096 | 1.00× | 1.07× | 1.11× | 1.11× |
| 16384 | 0.96× | 1.03× | 0.96× | 0.96× |
| 32768 | 0.98× | 1.00× | 0.92× | 0.92× |

Split by tier (f32): at BLOCK scope fused wins modestly and consistently
(mcross 1.28–1.42×, sinertia 1.04–1.26× — materializing the 6x6 costs shared
memory traffic + a barrier); at THREAD scope the two are identical-to-8%-
either-way (the 6x6 lives in registers regardless; the ≤8% composed edge on
mxform/sinertia at NPROB=32k is order-of-noise-worst-case and NOT a reason to
compose). **Conclusion, as the paper claims: the fused forms' value is
correctness economics (no 36-element scratch, one less barrier, pinned
convention) — hand-rolling or composing matches the speed but not the
guarantees.**

## Verdict 2 — tier packing: THREAD tier dominates every fixed-size op at batch

ns/problem at NPROB=32768 f32, best config per tier:

| op | BLOCK | WARP | THREAD | block/thread |
|---|---|---|---|---|
| quat_retract | 0.512 | 0.219 | **0.078** | 6.6× |
| se3_retract | 0.523 | 0.343 | **0.117** | 4.5× |
| quat_error | 0.508 | 0.176 | **0.064** | 7.9× |
| motion_cross_mul | 0.511 | 0.256 | **0.093** | 5.5× |
| eig3 | 5.25 | 5.18 | **0.244** | 21.5× |
| svd3 | 2.10 | 1.62 | **0.379** | 5.6× |
| closest_rotation | 1.98 | 1.60 | **0.254** | 7.8× |
| argmax (n=16) | 0.525 | 0.132 | **0.048** | 11.0× |
| softmax (n=16) | 0.523 | **0.130** | 0.368 | 1.4× |

The redundant-core construction makes this mechanical: at block/warp scope
every thread computes the whole small result and the tier only strides the
copy-out, so wider tiers buy NOTHING for these shapes — pack one problem per
thread when you have the batch to fill the machine. The 21.5× on eig3 is the
extreme case (a serial fixed-sweep Jacobi repeated per thread at block scope).
**Exception: softmax** — a genuine n-length reduction, where the warp tier's
shuffle tree wins; use `warp::softmax` for weight updates, thread tier for
everything else at batch. f64 shows the same ordering throughout (~1.2–2×
absolute slowdown). Crossover: block/warp become competitive only below
NPROB ≈ 1–4k, where the launch floor flattens everything anyway.

## Verdict 3 — argmax_fast vs argmax (block tier, n=16)

Wash at tb32/64 (0.520 vs 0.525), ~10–13% win at tb128/256 (0.507 vs 0.571,
0.921 vs 1.059). As expected: the shuffle strategy pays off only when the
serial thread-0 fold spans many warps. Keep the default for narrow blocks;
`_fast` for ≥128-thread blocks (full-warp sizes only, per its contract).

## Methodology notes

- FAIL-guarded `time_ns_per_prob` (empty-launch poison lesson); no FAIL cells
  appeared in this capture.
- Inputs are valid-by-construction where semantics demand (unit quats,
  orthonormal E); est-kit inputs are well-scaled finite values (timing only).
- The composed legs share the tier of their fused counterpart (block:
  materialize into smem + `glass::gemv`; thread: registers +
  `thread::gemv`); warp has no composed leg (no natural scratch home).
