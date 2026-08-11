# Measured results — the machine-refreshed archive

One file, one section per sweep leg. The blocks between
`<!-- BEGIN tune.py <leg> -->` markers are **rewritten by `bench/tune.py`** on
every timed run of that leg; the surrounding prose is curated. Division of
labor (2026-08-11 consolidation):

- **Analysis & narrative** → the docs site
  (`docs/source/user_guide/tutorials/sweep_results.rst`).
- **Raw captures** → the `glass-paper` repo (`data/desktop/`, `data/jetson/`,
  `data/sm120/`) — nothing raw is tracked here.
- **Shipped decisions** → `glass-defaults.cuh` (ladder/blas2/rect tables),
  `glass-dispatch.cuh` (body dispatch), `src/nvidia/tuning_table.cuh`
  (per-shape cuBLASDx-vs-SIMT).
- Rows carry `spread<=X%` (worst 3-trial jitter) since 2026-08-11;
  `tune.py` warns when a row's spread exceeds the margin it resolves.

## ladder (mega sweep — thread/warp/block/nvidia)

Captures: 3-way 2026-06-23, 4-way (thread tier) 2026-07-19, both quiet RTX
5090/sm_120 (archived in glass-paper `data/desktop/`). The winners ARE the
shipped `ideal_sm120` table; sm_87 (Jetson Orin, three power modes,
byte-identical tables) landed 2026-08-03. Ladder analysis, thread-tier
verdict tables, and figures live on the docs site; the winner-per-(op,N)
table renders from `docs/source/_static/sweep_winners.txt`.

## blas2 (warp vs block for syrk/syr2k/ldlt/ldltsv/inv/trmv/ger)

Ops the ladder misses. No nvidia counterparts (2-way); inv/trmv/ger are
block-only (reported, never picked). The 2-impl ops regenerate the shipped
per-arch `blas2_sm*` table in glass-defaults.cuh (since 2026-08-06).

<!-- BEGIN tune.py blas2 -->
<!-- END tune.py blas2 -->

## rect (warp vs block for rectangular gemv/gemm)

Tall/wide gemv + rectangular gemm shapes (consumers' Jacobians are
rectangular; the ladder is square-only). nvidia leg skipped — per-shape
cuBLASDx decisions live in the `shapes` leg. Measured shapes regenerate the
shipped exact-shape `rect_*_sm*` pickers (`suggested_backend_rect_gemv/gemm<>`).

> ⚠ **2026-08-11 audit:** the 2026-07-18 capture below predates
> `bench_rect`'s launch-FAIL guard; its five `warp 0.04 ns` gemm cells
> (f64 8x32x8 / 32x8x32 / 64x6x6 / 64x16x16, f32 64x16x16) are empty-launch
> poison. The shipped table cells were flipped to conservative `block`;
> a FAIL-guarded re-measure is staged for the next quiet window. The gemv
> rows and the honest gemm rows are unaffected.

<!-- BEGIN tune.py rect -->
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `rect_sweep_20260718_0328.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 19 of 24 cells._

nvidia leg skipped for rectangular shapes (needs new per-shape DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in the `shapes` leg). Measured shapes regenerate the shipped exact-shape `rect_*_sm*` pickers in glass-defaults.cuh (`suggested_backend_rect_gemv/gemm<>`, since 2026-08-06); unmeasured shapes stay block.

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| gemv | 8x64 | f32 | 1.12 | 0.99 | **warp** | warp wins (0.990 vs block 1.120, 13.1%) |
| gemv | 8x64 | f64 | 2.49 | 2.47 | **warp** | warp wins (2.470 vs block 2.490, 0.8%) |
| gemv | 16x128 | f32 | 2.35 | 1.87 | **warp** | warp wins (1.870 vs block 2.350, 25.7%) |
| gemv | 16x128 | f64 | 12.11 | 10.75 | **warp** | warp wins (10.750 vs block 12.110, 12.7%) |
| gemv | 32x256 | f32 | 23.48 | 20.86 | **warp** | warp wins (20.860 vs block 23.480, 12.6%) |
| gemv | 32x256 | f64 | 42.55 | 41.32 | **warp** | warp wins (41.320 vs block 42.550, 3.0%) |
| gemv | 64x8 | f32 | 0.64 | 0.48 | **warp** | warp wins (0.480 vs block 0.640, 33.3%) |
| gemv | 64x8 | f64 | 0.89 | 0.87 | **warp** | warp wins (0.870 vs block 0.890, 2.3%) |
| gemv | 128x16 | f32 | 1.37 | 1.59 | **block** | block wins (1.370 vs warp 1.590, 16.1%) |
| gemv | 128x16 | f64 | 10.48 | 10.57 | **warp** | warp kept (10.570); block faster by 0.9% but inside ±2% SIMT tie |
| gemv | 256x32 | f32 | 20.36 | 20.42 | **warp** | warp kept (20.420); block faster by 0.3% but inside ±2% SIMT tie |
| gemv | 256x32 | f64 | 40.31 | 40.78 | **warp** | warp kept (40.780); block faster by 1.2% but inside ±2% SIMT tie |
| gemm | 6x6x64 | f32 | 0.88 | 0.92 | **block** | block wins (0.880 vs warp 0.920, 4.5%) |
| gemm | 6x6x64 | f64 | 2.74 | 3.02 | **block** | block wins (2.740 vs warp 3.020, 10.2%) |
| gemm | 8x32x8 | f32 | 0.92 | 0.88 | **warp** | warp wins (0.880 vs block 0.920, 4.5%) |
| gemm | 8x32x8 | f64 | 2.42 | 0.04 | **warp** | POISONED (empty launch) — see audit note above |
| gemm | 16x64x16 | f32 | 3.75 | 3.94 | **block** | block wins (3.750 vs warp 3.940, 5.1%) |
| gemm | 16x64x16 | f64 | 18.50 | 19.15 | **block** | block wins (18.500 vs warp 19.150, 3.5%) |
| gemm | 32x8x32 | f32 | 1.71 | 1.53 | **warp** | warp wins (1.530 vs block 1.710, 11.8%) |
| gemm | 32x8x32 | f64 | 9.63 | 0.04 | **warp** | POISONED (empty launch) — see audit note above |
| gemm | 64x6x6 | f32 | 0.77 | 0.72 | **warp** | warp wins (0.720 vs block 0.770, 6.9%) |
| gemm | 64x6x6 | f64 | 3.59 | 0.04 | **warp** | POISONED (empty launch) — see audit note above |
| gemm | 64x16x16 | f32 | 2.49 | 0.04 | **warp** | POISONED (empty launch) — see audit note above |
| gemm | 64x16x16 | f64 | 18.24 | 0.04 | **warp** | POISONED (empty launch) — see audit note above |

<!-- END tune.py rect -->

## solvers (characterization only — never picked)

bdsv-vs-pcg on identical block-tridiagonal SPD input (the crossover moves
with conditioning — read the iters column before generalizing); gesv/posv/
inv+gemv robustness-and-anti-pattern pricing; syev/eig_clamp timing.
Restore-outside-timing protocol (these ops mutate their input): pristine
copies restored between reps outside the timed window, cudaEvent timing, and
a host-oracle correctness gate before any timing.

<!-- BEGIN tune.py solvers -->
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `solvers_sweep_20260718_0329.txt` · NPROB=8192 ns/problem (best swept TB, min of 3 trials, restore-outside-timing protocol) · characterization only — no dispatch table is regenerated._

### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input

bdsv is faster in 1 of 12 cells **on this well-conditioned test system** (see the iters column — pcg's cost scales with the iteration count, so the crossover moves with conditioning).

| BlockSize | Knots | dtype | bdsv ns | pcg ns | pcg iters | pcg/bdsv |
|-----------|-------|-------|---------|--------|-----------|----------|
| 2 | 8 | f32 | 7.00 | 2.73 | 3 | 0.39 |
| 2 | 8 | f64 | 30.09 | 9.50 | 3 | 0.32 |
| 2 | 32 | f32 | 27.55 | 3.99 | 3 | 0.14 |
| 2 | 32 | f64 | 121.15 | 12.23 | 3 | 0.10 |
| 6 | 8 | f32 | 21.00 | 6.50 | 3 | 0.31 |
| 6 | 8 | f64 | 99.82 | 31.31 | 3 | 0.31 |
| 6 | 32 | f32 | 92.22 | 30.51 | 3 | 0.33 |
| 6 | 32 | f64 | 391.36 | 130.44 | 3 | 0.33 |
| 6 | 64 | f32 | 178.10 | 83.55 | 3 | 0.47 |
| 6 | 64 | f64 | 791.46 | 252.34 | 3 | 0.32 |
| 12 | 16 | f32 | 109.70 | 197.92 | 2 | 1.80 |
| 12 | 16 | f64 | 462.31 | 230.11 | 2 | 0.50 |

### gesv vs posv vs inv+gemv — same SPD system, single RHS

posv (Cholesky) is the intended SPD path; gesv prices the pivoted-LU robustness fallback, inv+gemv the invert-then-multiply anti-pattern.

The `thr-posv` column is the **thread-tier** `glass::thread::posv` (one problem per thread, 32 packed per warp) — measured only below the N<=7 register-residency ceiling. Where `thr/posv` < 1 the thread tier beats the block Cholesky solve on that low-DOF shape.

| N | dtype | gesv ns | posv ns | inv+gemv ns | thr-posv ns | gesv/posv | inv/posv | thr/posv |
|---|-------|---------|---------|-------------|-------------|-----------|----------|----------|
| 4 | f32 | 1.24 | 1.22 | 1.00 | 0.49 | 1.02 | 0.82 | 0.40 |
| 4 | f64 | 3.74 | 5.51 | 1.99 | 0.98 | 0.68 | 0.36 | 0.18 |
| 8 | f32 | 2.50 | 2.48 | 2.25 | — | 1.01 | 0.91 | — |
| 8 | f64 | 9.00 | 12.50 | 5.59 | — | 0.72 | 0.45 | — |
| 16 | f32 | 6.50 | 5.53 | 10.56 | — | 1.18 | 1.91 | — |
| 16 | f64 | 25.53 | 30.36 | 23.75 | — | 0.84 | 0.78 | — |
| 32 | f32 | 27.84 | 15.96 | 57.73 | — | 1.74 | 3.62 | — |
| 32 | f64 | 86.11 | 78.69 | 151.90 | — | 1.09 | 1.93 | — |
| 64 | f32 | 162.24 | 77.13 | 359.54 | — | 2.10 | 4.66 | — |
| 64 | f64 | 420.32 | 266.83 | 1072.80 | — | 1.58 | 4.02 | — |

### syev + eig_clamp — timing only (no contender)

| N | dtype | syev ns | eig_clamp ns |
|---|-------|---------|--------------|
| 4 | f32 | 3.91 | 3.99 |
| 4 | f64 | 58.38 | 59.03 |
| 8 | f32 | 25.43 | 25.50 |
| 8 | f64 | 388.32 | 390.09 |
| 16 | f32 | 115.70 | 116.32 |
| 16 | f64 | 1781.17 | 1754.45 |
| 32 | f32 | 881.79 | 1019.54 |
| 32 | f64 | 8849.49 | 9305.33 |

<!-- END tune.py solvers -->

## reduced (`*_reduced` contraction-parallel crossover)

Retired to constant `false` on sm_120 (2026-07-08): the quiet-GPU resweep
measured 0 of 48 configs where reduced clears the ±5% margin.
`suggested_use_reduced<>()`'s signature is kept as the seam for a future
data-derived corner on different hardware. Treat the `*_reduced` and
tensor/congruence families as expressiveness tools, never speed defaults
(full analysis on the docs site).

<!-- BEGIN tune.py reduced -->
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `reduced_sweep_20260718_0130.txt` · tie margin ±5% (reduced must clear it) · 0 of 48 configs pick reduced._

Predicate `suggested_use_reduced<n_out,K_contract,blockDim>()` = `(n_out <= blockDim/32) && (K_contract >= 32)` (K_contract is the N column here).

✅ Measurement matches the predicate for every swept config — the formula needs no change.

<!-- END tune.py reduced -->

## nvwarp (warp-scope vendor A/B — hand-run, `bench_nvwarp_l1.cu`)

`glass::warp::` vs `glass::nvidia::warp::` (CUB WarpReduce) at identical
launch shape, correctness-gated. **sm_87 (Orin, 50 W, 2026-08-04): 110/126
cells tie within ±2%. sm_120 (5090, 2026-08-06): 119/126.** All non-ties
≤10%. Verdict on both arches: the two warp tiers are the same algorithm and
measure like it — the measured justification for the dispatch ladder not
descending below block scope for the nvidia tier. Raw captures:
glass-paper `data/jetson/` and `data/sm120/`; full analysis on the docs site.

## robotics (micro-op tier/fusion verdicts — hand-run, `bench_robotics.cu`)

Capture 2026-07-29 (quiet sm_120, archived glass-paper `data/desktop/`);
noise floor median 0.40%, p90 0.87% (repeat pass). Verdicts: (1) fused vs
composed spatial ops = wash at the best tier — the fused forms' value is
correctness economics, exactly as the paper claims; (2) the THREAD tier
dominates every fixed-size robotics op at batch (4.5–21.5× vs block;
exception `softmax`, a genuine reduction → warp); (3) `argmax_fast` pays off
only at ≥128-thread blocks. Full tables on the docs site.

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild            # compile everything (GPU may be busy)
python3 bench/tune.py --sm auto                       # all legs, timed — QUIET GPU only
python3 bench/tune.py --sm auto --legs blas2,rect     # subset
python3 bench/tune.py --legs solvers --from-solvers <capture.txt> --dry-run   # replay, no GPU
```

Per-harness direct invocations and flags: `bench/TUNING.md`.
