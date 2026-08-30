# Measured results — the machine-refreshed archive

One file, one section per sweep leg. The blocks between
`<!-- BEGIN tune.py <leg> -->` markers are **rewritten by `bench/tune.py`** on
every timed run of that leg; the surrounding prose is curated. Division of
labor (2026-08-11 consolidation):

- **Analysis & narrative** → the docs site
  (`docs/source/user_guide/tutorials/sweep_results.rst`).
- **Raw captures** → archived externally with the paper materials — nothing
  raw is tracked here.
- **Shipped decisions** → `glass-defaults.cuh` (ladder/blas2/rect tables),
  `glass-dispatch.cuh` (body dispatch), `src/nvidia/tuning_table.cuh`
  (per-shape cuBLASDx-vs-SIMT).
- Rows carry `spread<=X%` (worst 3-trial jitter) since 2026-08-11;
  `tune.py` warns when a row's spread exceeds the margin it resolves.
- Historical rows remain evidence for their named source capture only. A result
  is current only when its source digest, environment, timing date, and proximate
  signed correctness receipt are recorded together.

## ladder (mega sweep — native thread/warp/block plus NVIDIA block/thread)

The shipped `ideal_sm120` table was re-gated 2026-08-15 from the pinned quiet
capture `mega_sweep_20260815_205919.txt` (archived externally), taken with the
drift-fixed v2 harness (untimed per-trial input reset in `timing_common.cuh`);
it is the cleanest capture recorded (40/396 rows jitter-flagged, each verified
cell-by-cell at re-gate). Earlier captures (3-way 2026-06-23, 4-way thread
tier 2026-07-19) are superseded for table generation. sm_87 (Jetson Orin,
three power modes, byte-identical tables) landed 2026-08-03. Ladder analysis,
thread-tier verdict tables, and figures live on the docs site; the
winner-per-(op,N) table renders from `docs/source/_static/sweep_winners.txt`.

## blas2 (warp vs block for syrk/syr2k/ldlt/ldltsv/inv/trmv/ger)

Ops the ladder misses. No nvidia counterparts (2-way); inv/trmv/ger are
block-only (reported, never picked). The 2-impl ops regenerate the shipped
per-arch `blas2_sm*` table in glass-defaults.cuh (since 2026-08-06).

<!-- BEGIN tune.py blas2 -->
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `blas2_sweep_20260814_011659.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 30 of 154 cells._

inv/trmv/ger are BLOCK-ONLY (no `glass::warp::` variant, so nothing competes — reported, never picked); none of these ops has a `glass::nvidia::` counterpart. The 2-impl ops (syrk/syr2k/ldlt/ldltsv) regenerate the shipped per-arch `blas2_sm*` table in glass-defaults.cuh (since 2026-08-06).

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| syrk | N=4 | f32 | 0.56 | 0.17 | **warp** | warp wins (0.170 vs block 0.560, 229.4%) |
| syrk | N=4 | f64 | 0.58 | 0.28 | **warp** | warp wins (0.280 vs block 0.580, 107.1%) |
| syrk | N=6 | f32 | 0.58 | 0.25 | **warp** | warp wins (0.250 vs block 0.580, 132.0%) |
| syrk | N=6 | f64 | 0.64 | 0.57 | **warp** | warp wins (0.570 vs block 0.640, 12.3%) |
| syrk | N=8 | f32 | 0.59 | 0.33 | **warp** | warp wins (0.330 vs block 0.590, 78.8%) |
| syrk | N=8 | f64 | 0.72 | 0.72 | **warp** | warp kept (0.720); block faster by 0.0% but inside ±2% SIMT tie |
| syrk | N=12 | f32 | 0.75 | 0.73 | **warp** | warp wins (0.730 vs block 0.750, 2.7%) |
| syrk | N=12 | f64 | 2.28 | 2.35 | **block** | block wins (2.280 vs warp 2.350, 3.1%) |
| syrk | N=16 | f32 | 1.37 | 1.40 | **block** | block wins (1.370 vs warp 1.400, 2.2%) |
| syrk | N=16 | f64 | 4.63 | 4.78 | **block** | block wins (4.630 vs warp 4.780, 3.2%) |
| syrk | N=24 | f32 | 4.10 | 4.52 | **block** | block wins (4.100 vs warp 4.520, 10.2%) |
| syrk | N=24 | f64 | 15.14 | 16.15 | **block** | block wins (15.140 vs warp 16.150, 6.7%) |
| syrk | N=32 | f32 | 10.77 | 13.94 | **block** | block wins (10.770 vs warp 13.940, 29.4%) |
| syrk | N=32 | f64 | 35.49 | 39.27 | **block** | block wins (35.490 vs warp 39.270, 10.7%) |
| syrk | N=48 | f32 | 30.37 | 62.95 | **block** | block wins (30.370 vs warp 62.950, 107.3%) |
| syrk | N=48 | f64 | 105.63 | 139.24 | **block** | block wins (105.630 vs warp 139.240, 31.8%) |
| syrk | N=64 | f32 | 62.49 | 127.33 | **block** | block wins (62.490 vs warp 127.330, 103.8%) |
| syrk | N=64 | f64 | 212.79 | 562.74 | **block** | block wins (212.790 vs warp 562.740, 164.5%) |
| syrk | N=96 | f32 | 193.80 | 1458.50 | **block** | block wins (193.800 vs warp 1458.500, 652.6%) |
| syrk | N=96 | f64 | 648.16 | 2774.44 | **block** | block wins (648.160 vs warp 2774.440, 328.0%) |
| syrk | N=128 | f32 | 506.97 | 3609.24 | **block** | block wins (506.970 vs warp 3609.240, 611.9%) |
| syrk | N=128 | f64 | 1565.54 | 6682.96 | **block** | block wins (1565.540 vs warp 6682.960, 326.9%) |
| syr2k | N=4 | f32 | 0.58 | 0.21 | **warp** | warp wins (0.210 vs block 0.580, 176.2%) |
| syr2k | N=4 | f64 | 0.64 | 0.61 | **warp** | warp wins (0.610 vs block 0.640, 4.9%) |
| syr2k | N=6 | f32 | 0.60 | 0.37 | **warp** | warp wins (0.370 vs block 0.600, 62.2%) |
| syr2k | N=6 | f64 | 1.53 | 1.56 | **warp** | warp kept (1.560); block faster by 2.0% but inside ±2% SIMT tie |
| syr2k | N=8 | f32 | 0.61 | 0.43 | **warp** | warp wins (0.430 vs block 0.610, 41.9%) |
| syr2k | N=8 | f64 | 1.98 | 2.03 | **block** | block wins (1.980 vs warp 2.030, 2.5%) |
| syr2k | N=12 | f32 | 1.27 | 1.30 | **block** | block wins (1.270 vs warp 1.300, 2.4%) |
| syr2k | N=12 | f64 | 7.05 | 7.29 | **block** | block wins (7.050 vs warp 7.290, 3.4%) |
| syr2k | N=16 | f32 | 2.49 | 2.62 | **block** | block wins (2.490 vs warp 2.620, 5.2%) |
| syr2k | N=16 | f64 | 14.79 | 15.25 | **block** | block wins (14.790 vs warp 15.250, 3.1%) |
| syr2k | N=24 | f32 | 7.77 | 10.78 | **block** | block wins (7.770 vs warp 10.780, 38.7%) |
| syr2k | N=24 | f64 | 49.99 | 51.32 | **block** | block wins (49.990 vs warp 51.320, 2.7%) |
| syr2k | N=32 | f32 | 19.75 | 31.09 | **block** | block wins (19.750 vs warp 31.090, 57.4%) |
| syr2k | N=32 | f64 | 117.84 | 126.73 | **block** | block wins (117.840 vs warp 126.730, 7.5%) |
| syr2k | N=48 | f32 | 57.62 | 132.22 | **block** | block wins (57.620 vs warp 132.220, 129.5%) |
| syr2k | N=48 | f64 | 353.29 | 579.43 | **block** | block wins (353.290 vs warp 579.430, 64.0%) |
| syr2k | N=64 | f32 | 119.67 | 719.81 | **block** | block wins (119.670 vs warp 719.810, 501.5%) |
| syr2k | N=64 | f64 | 706.00 | 1555.61 | **block** | block wins (706.000 vs warp 1555.610, 120.3%) |
| syr2k | N=96 | f32 | 363.45 | 3250.30 | **block** | block wins (363.450 vs warp 3250.300, 794.3%) |
| syr2k | N=96 | f64 | 2118.74 | 5431.08 | **block** | block wins (2118.740 vs warp 5431.080, 156.3%) |
| syr2k | N=128 | f32 | 1256.06 | 6988.99 | **block** | block wins (1256.060 vs warp 6988.990, 456.4%) |
| syr2k | N=128 | f64 | 5321.81 | 12650.99 | **block** | block wins (5321.810 vs warp 12650.990, 137.7%) |
| ldlt | N=4 | f32 | 0.64 | 0.39 | **warp** | warp wins (0.390 vs block 0.640, 64.1%) |
| ldlt | N=4 | f64 | 1.95 | 3.36 | **block** | block wins (1.950 vs warp 3.360, 72.3%) |
| ldlt | N=6 | f32 | 1.03 | 0.82 | **warp** | warp wins (0.820 vs block 1.030, 25.6%) |
| ldlt | N=6 | f64 | 4.50 | 6.44 | **block** | block wins (4.500 vs warp 6.440, 43.1%) |
| ldlt | N=8 | f32 | 1.59 | 1.29 | **warp** | warp wins (1.290 vs block 1.590, 23.3%) |
| ldlt | N=8 | f64 | 7.12 | 10.04 | **block** | block wins (7.120 vs warp 10.040, 41.0%) |
| ldlt | N=12 | f32 | 3.00 | 2.43 | **warp** | warp wins (2.430 vs block 3.000, 23.5%) |
| ldlt | N=12 | f64 | 14.24 | 19.04 | **block** | block wins (14.240 vs warp 19.040, 33.7%) |
| ldlt | N=16 | f32 | 4.65 | 3.86 | **warp** | warp wins (3.860 vs block 4.650, 20.5%) |
| ldlt | N=16 | f64 | 24.23 | 30.36 | **block** | block wins (24.230 vs warp 30.360, 25.3%) |
| ldlt | N=24 | f32 | 9.07 | 7.96 | **warp** | warp wins (7.960 vs block 9.070, 13.9%) |
| ldlt | N=24 | f64 | 52.19 | 60.62 | **block** | block wins (52.190 vs warp 60.620, 16.2%) |
| ldlt | N=32 | f32 | 16.20 | 14.13 | **warp** | warp wins (14.130 vs block 16.200, 14.6%) |
| ldlt | N=32 | f64 | 90.60 | 100.98 | **block** | block wins (90.600 vs warp 100.980, 11.5%) |
| ldlt | N=48 | f32 | 44.04 | 43.48 | **warp** | warp wins (43.480 vs block 44.040, 1.3%) |
| ldlt | N=48 | f64 | 215.87 | 235.38 | **block** | block wins (215.870 vs warp 235.380, 9.0%) |
| ldlt | N=64 | f32 | 93.78 | 93.86 | **warp** | warp kept (93.860); block faster by 0.1% but inside ±2% SIMT tie |
| ldlt | N=64 | f64 | 403.88 | 433.02 | **block** | block wins (403.880 vs warp 433.020, 7.2%) |
| ldlt | N=96 | f32 | 280.23 | 304.60 | **block** | block wins (280.230 vs warp 304.600, 8.7%) |
| ldlt | N=96 | f64 | 1022.24 | 1075.16 | **block** | block wins (1022.240 vs warp 1075.160, 5.2%) |
| ldlt | N=128 | f32 | 665.47 | 807.69 | **block** | block wins (665.470 vs warp 807.690, 21.4%) |
| ldlt | N=128 | f64 | 2038.77 | 2462.17 | **block** | block wins (2038.770 vs warp 2462.170, 20.8%) |
| ldltsv | N=4 | f32 | 0.73 | 0.58 | **warp** | warp wins (0.580 vs block 0.730, 25.9%) |
| ldltsv | N=4 | f64 | 2.98 | 4.39 | **block** | block wins (2.980 vs warp 4.390, 47.3%) |
| ldltsv | N=6 | f32 | 1.43 | 1.02 | **warp** | warp wins (1.020 vs block 1.430, 40.2%) |
| ldltsv | N=6 | f64 | 5.76 | 7.61 | **block** | block wins (5.760 vs warp 7.610, 32.1%) |
| ldltsv | N=8 | f32 | 2.13 | 1.54 | **warp** | warp wins (1.540 vs block 2.130, 38.3%) |
| ldltsv | N=8 | f64 | 8.56 | 11.34 | **block** | block wins (8.560 vs warp 11.340, 32.5%) |
| ldltsv | N=12 | f32 | 3.92 | 2.83 | **warp** | warp wins (2.830 vs block 3.920, 38.5%) |
| ldltsv | N=12 | f64 | 16.02 | 20.66 | **block** | block wins (16.020 vs warp 20.660, 29.0%) |
| ldltsv | N=16 | f32 | 5.81 | 4.43 | **warp** | warp wins (4.430 vs block 5.810, 31.2%) |
| ldltsv | N=16 | f64 | 26.48 | 32.44 | **block** | block wins (26.480 vs warp 32.440, 22.5%) |
| ldltsv | N=24 | f32 | 10.87 | 8.82 | **warp** | warp wins (8.820 vs block 10.870, 23.2%) |
| ldltsv | N=24 | f64 | 54.94 | 63.42 | **block** | block wins (54.940 vs warp 63.420, 15.4%) |
| ldltsv | N=32 | f32 | 19.77 | 15.91 | **warp** | warp wins (15.910 vs block 19.770, 24.3%) |
| ldltsv | N=32 | f64 | 94.23 | 104.42 | **block** | block wins (94.230 vs warp 104.420, 10.8%) |
| ldltsv | N=48 | f32 | 52.09 | 48.92 | **warp** | warp wins (48.920 vs block 52.090, 6.5%) |
| ldltsv | N=48 | f64 | 223.69 | 243.07 | **block** | block wins (223.690 vs warp 243.070, 8.7%) |
| ldltsv | N=64 | f32 | 105.38 | 103.67 | **warp** | warp wins (103.670 vs block 105.380, 1.6%) |
| ldltsv | N=64 | f64 | 416.28 | 444.67 | **block** | block wins (416.280 vs warp 444.670, 6.8%) |
| ldltsv | N=96 | f32 | 300.68 | 321.69 | **block** | block wins (300.680 vs warp 321.690, 7.0%) |
| ldltsv | N=96 | f64 | 1047.14 | 1097.29 | **block** | block wins (1047.140 vs warp 1097.290, 4.8%) |
| ldltsv | N=128 | f32 | 734.13 | 863.41 | **block** | block wins (734.130 vs warp 863.410, 17.6%) |
| ldltsv | N=128 | f64 | 2050.26 | 2534.78 | **block** | block wins (2050.260 vs warp 2534.780, 23.6%) |
| inv | N=4 | f32 | 0.71 | — | **block** | block only impl measured (0.710) |
| inv | N=4 | f64 | 1.44 | — | **block** | block only impl measured (1.440) |
| inv | N=6 | f32 | 1.12 | — | **block** | block only impl measured (1.120) |
| inv | N=6 | f64 | 2.94 | — | **block** | block only impl measured (2.940) |
| inv | N=8 | f32 | 1.83 | — | **block** | block only impl measured (1.830) |
| inv | N=8 | f64 | 4.99 | — | **block** | block only impl measured (4.990) |
| inv | N=12 | f32 | 4.54 | — | **block** | block only impl measured (4.540) |
| inv | N=12 | f64 | 10.57 | — | **block** | block only impl measured (10.570) |
| inv | N=16 | f32 | 9.30 | — | **block** | block only impl measured (9.300) |
| inv | N=16 | f64 | 22.31 | — | **block** | block only impl measured (22.310) |
| inv | N=24 | f32 | 26.43 | — | **block** | block only impl measured (26.430) |
| inv | N=24 | f64 | 63.79 | — | **block** | block only impl measured (63.790) |
| inv | N=32 | f32 | 52.09 | — | **block** | block only impl measured (52.090) |
| inv | N=32 | f64 | 139.88 | — | **block** | block only impl measured (139.880) |
| inv | N=48 | f32 | 171.97 | — | **block** | block only impl measured (171.970) |
| inv | N=48 | f64 | 453.52 | — | **block** | block only impl measured (453.520) |
| inv | N=64 | f32 | 357.96 | — | **block** | block only impl measured (357.960) |
| inv | N=64 | f64 | 1025.26 | — | **block** | block only impl measured (1025.260) |
| inv | N=96 | f32 | 1455.32 | — | **block** | block only impl measured (1455.320) |
| inv | N=96 | f64 | 3409.91 | — | **block** | block only impl measured (3409.910) |
| inv | N=128 | f32 | 3472.58 | — | **block** | block only impl measured (3472.580) |
| inv | N=128 | f64 | 18683.98 | — | **block** | block only impl measured (18683.980) |
| trmv | N=4 | f32 | 0.59 | — | **block** | block only impl measured (0.590) |
| trmv | N=4 | f64 | 0.63 | — | **block** | block only impl measured (0.630) |
| trmv | N=6 | f32 | 0.60 | — | **block** | block only impl measured (0.600) |
| trmv | N=6 | f64 | 0.68 | — | **block** | block only impl measured (0.680) |
| trmv | N=8 | f32 | 0.62 | — | **block** | block only impl measured (0.620) |
| trmv | N=8 | f64 | 0.83 | — | **block** | block only impl measured (0.830) |
| trmv | N=12 | f32 | 0.64 | — | **block** | block only impl measured (0.640) |
| trmv | N=12 | f64 | 0.92 | — | **block** | block only impl measured (0.920) |
| trmv | N=16 | f32 | 0.75 | — | **block** | block only impl measured (0.750) |
| trmv | N=16 | f64 | 1.55 | — | **block** | block only impl measured (1.550) |
| trmv | N=24 | f32 | 0.83 | — | **block** | block only impl measured (0.830) |
| trmv | N=24 | f64 | 1.61 | — | **block** | block only impl measured (1.610) |
| trmv | N=32 | f32 | 1.14 | — | **block** | block only impl measured (1.140) |
| trmv | N=32 | f64 | 2.24 | — | **block** | block only impl measured (2.240) |
| trmv | N=48 | f32 | 1.93 | — | **block** | block only impl measured (1.930) |
| trmv | N=48 | f64 | 5.39 | — | **block** | block only impl measured (5.390) |
| trmv | N=64 | f32 | 3.50 | — | **block** | block only impl measured (3.500) |
| trmv | N=64 | f64 | 12.39 | — | **block** | block only impl measured (12.390) |
| trmv | N=96 | f32 | 15.08 | — | **block** | block only impl measured (15.080) |
| trmv | N=96 | f64 | 26.30 | — | **block** | block only impl measured (26.300) |
| trmv | N=128 | f32 | 24.89 | — | **block** | block only impl measured (24.890) |
| trmv | N=128 | f64 | 44.24 | — | **block** | block only impl measured (44.240) |
| ger | N=4 | f32 | 0.55 | — | **block** | block only impl measured (0.550) |
| ger | N=4 | f64 | 0.55 | — | **block** | block only impl measured (0.550) |
| ger | N=6 | f32 | 0.57 | — | **block** | block only impl measured (0.570) |
| ger | N=6 | f64 | 0.57 | — | **block** | block only impl measured (0.570) |
| ger | N=8 | f32 | 0.57 | — | **block** | block only impl measured (0.570) |
| ger | N=8 | f64 | 0.57 | — | **block** | block only impl measured (0.570) |
| ger | N=12 | f32 | 0.60 | — | **block** | block only impl measured (0.600) |
| ger | N=12 | f64 | 0.62 | — | **block** | block only impl measured (0.620) |
| ger | N=16 | f32 | 0.61 | — | **block** | block only impl measured (0.610) |
| ger | N=16 | f64 | 0.83 | — | **block** | block only impl measured (0.830) |
| ger | N=24 | f32 | 0.97 | — | **block** | block only impl measured (0.970) |
| ger | N=24 | f64 | 1.71 | — | **block** | block only impl measured (1.710) |
| ger | N=32 | f32 | 1.56 | — | **block** | block only impl measured (1.560) |
| ger | N=32 | f64 | 2.88 | — | **block** | block only impl measured (2.880) |
| ger | N=48 | f32 | 3.47 | — | **block** | block only impl measured (3.470) |
| ger | N=48 | f64 | 25.11 | — | **block** | block only impl measured (25.110) |
| ger | N=64 | f32 | 22.41 | — | **block** | block only impl measured (22.410) |
| ger | N=64 | f64 | 44.68 | — | **block** | block only impl measured (44.680) |
| ger | N=96 | f32 | 50.37 | — | **block** | block only impl measured (50.370) |
| ger | N=96 | f64 | 99.89 | — | **block** | block only impl measured (99.890) |
| ger | N=128 | f32 | 89.16 | — | **block** | block only impl measured (89.160) |
| ger | N=128 | f64 | 176.57 | — | **block** | block only impl measured (176.570) |

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

_Source: `rect_sweep_20260814_031430.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 16 of 24 cells._

nvidia leg skipped for rectangular shapes (needs new per-shape DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in the `shapes` leg). Measured shapes regenerate the shipped exact-shape `rect_*_sm*` pickers in glass-defaults.cuh (`suggested_backend_rect_gemv/gemm<>`, since 2026-08-06); unmeasured shapes stay block.

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| gemv | 8x64 | f32 | 1.13 | 0.99 | **warp** | warp wins (0.990 vs block 1.130, 14.1%) |
| gemv | 8x64 | f64 | 2.49 | 2.47 | **warp** | warp wins (2.470 vs block 2.490, 0.8%) |
| gemv | 16x128 | f32 | 2.35 | 1.87 | **warp** | warp wins (1.870 vs block 2.350, 25.7%) |
| gemv | 16x128 | f64 | 12.12 | 10.76 | **warp** | warp wins (10.760 vs block 12.120, 12.6%) |
| gemv | 32x256 | f32 | 23.44 | 20.82 | **warp** | warp wins (20.820 vs block 23.440, 12.6%) |
| gemv | 32x256 | f64 | 42.50 | 41.41 | **warp** | warp wins (41.410 vs block 42.500, 2.6%) |
| gemv | 64x8 | f32 | 0.64 | 0.47 | **warp** | warp wins (0.470 vs block 0.640, 36.2%) |
| gemv | 64x8 | f64 | 0.81 | 0.80 | **warp** | warp wins (0.800 vs block 0.810, 1.2%) |
| gemv | 128x16 | f32 | 1.36 | 1.54 | **block** | block wins (1.360 vs warp 1.540, 13.2%) |
| gemv | 128x16 | f64 | 10.47 | 10.55 | **warp** | warp kept (10.550); block faster by 0.8% but inside ±2% SIMT tie |
| gemv | 256x32 | f32 | 20.33 | 20.43 | **warp** | warp kept (20.430); block faster by 0.5% but inside ±2% SIMT tie |
| gemv | 256x32 | f64 | 40.30 | 40.82 | **warp** | warp kept (40.820); block faster by 1.3% but inside ±2% SIMT tie |
| gemm | 6x6x64 | f32 | 0.89 | 0.93 | **block** | block wins (0.890 vs warp 0.930, 4.5%) |
| gemm | 6x6x64 | f64 | 2.74 | 3.00 | **block** | block wins (2.740 vs warp 3.000, 9.5%) |
| gemm | 8x32x8 | f32 | 0.92 | 0.87 | **warp** | warp wins (0.870 vs block 0.920, 5.7%) |
| gemm | 8x32x8 | f64 | 2.42 | 2.59 | **block** | block wins (2.420 vs warp 2.590, 7.0%) |
| gemm | 16x64x16 | f32 | 3.75 | 3.93 | **block** | block wins (3.750 vs warp 3.930, 4.8%) |
| gemm | 16x64x16 | f64 | 18.51 | 19.15 | **block** | block wins (18.510 vs warp 19.150, 3.5%) |
| gemm | 32x8x32 | f32 | 1.71 | 1.53 | **warp** | warp wins (1.530 vs block 1.710, 11.8%) |
| gemm | 32x8x32 | f64 | 9.68 | 10.02 | **block** | block wins (9.680 vs warp 10.020, 3.5%) |
| gemm | 64x6x6 | f32 | 0.77 | 0.72 | **warp** | warp wins (0.720 vs block 0.770, 6.9%) |
| gemm | 64x6x6 | f64 | 3.59 | 3.43 | **warp** | warp wins (3.430 vs block 3.590, 4.7%) |
| gemm | 64x16x16 | f32 | 2.39 | 2.03 | **warp** | warp wins (2.030 vs block 2.390, 17.7%) |
| gemm | 64x16x16 | f64 | 18.26 | 20.18 | **block** | block wins (18.260 vs warp 20.180, 10.5%) |

<!-- END tune.py rect -->

## solvers (characterization only — never picked)

bdsv-vs-pcg on identical block-tridiagonal SPD input at PCG's configured
``rho = rᵀz`` tolerance (not matched final accuracy; the crossover moves
with tolerance and conditioning — read the iters column before generalizing); gesv/posv/
inv+gemv robustness-and-anti-pattern pricing; syev/eig_clamp timing.
Restore-outside-timing protocol (these ops mutate their input): pristine
copies restored between reps outside the timed window, cudaEvent timing, and
a host-oracle correctness gate before any timing.

<!-- BEGIN tune.py solvers -->
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `solvers_sweep_20260814_031505.txt` · NPROB=8192 ns/problem (best swept TB, min of 3 trials, restore-outside-timing protocol) · characterization only — no dispatch table is regenerated._

### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input

bdsv is faster in 1 of 12 cells **on this well-conditioned test system at PCG's `rho = rᵀz` relative tolerance of 1e-6**. This is an approximate-solve comparison, not matched final residual accuracy; PCG cost and the crossover move with tolerance, conditioning, and iteration count.

| BlockSize | Knots | dtype | bdsv ns | pcg ns | pcg iters | pcg/bdsv |
|-----------|-------|-------|---------|--------|-----------|----------|
| 2 | 8 | f32 | 6.20 | 2.24 | 3 | 0.36 |
| 2 | 8 | f64 | 26.26 | 8.25 | 3 | 0.31 |
| 2 | 32 | f32 | 24.46 | 3.48 | 3 | 0.14 |
| 2 | 32 | f64 | 107.32 | 11.60 | 3 | 0.11 |
| 6 | 8 | f32 | 18.92 | 6.47 | 3 | 0.34 |
| 6 | 8 | f64 | 94.59 | 31.23 | 3 | 0.33 |
| 6 | 32 | f32 | 86.37 | 30.52 | 3 | 0.35 |
| 6 | 32 | f64 | 392.15 | 130.82 | 3 | 0.33 |
| 6 | 64 | f32 | 180.40 | 83.44 | 3 | 0.46 |
| 6 | 64 | f64 | 792.40 | 252.34 | 3 | 0.32 |
| 12 | 16 | f32 | 110.51 | 198.22 | 2 | 1.79 |
| 12 | 16 | f64 | 462.95 | 229.96 | 2 | 0.50 |

### gesv vs posv vs inv+gemv — same SPD system, single RHS

posv (Cholesky) is the intended SPD path; gesv prices the pivoted-LU robustness fallback, inv+gemv the invert-then-multiply anti-pattern.

The `thr-posv` column is the **thread-tier** `glass::thread::posv` (one problem per thread, 32 packed per warp) — measured only below the N<=7 register-residency ceiling. Where `thr/posv` < 1 the thread tier beats the block Cholesky solve on that low-DOF shape.

| N | dtype | gesv ns | posv ns | inv+gemv ns | thr-posv ns | gesv/posv | inv/posv | thr/posv |
|---|-------|---------|---------|-------------|-------------|-----------|----------|----------|
| 4 | f32 | 1.24 | 1.06 | 1.00 | 0.46 | 1.17 | 0.94 | 0.43 |
| 4 | f64 | 3.74 | 5.54 | 1.99 | 0.97 | 0.68 | 0.36 | 0.18 |
| 8 | f32 | 2.48 | 2.32 | 2.24 | — | 1.07 | 0.97 | — |
| 8 | f64 | 9.01 | 12.50 | 5.61 | — | 0.72 | 0.45 | — |
| 16 | f32 | 6.53 | 5.57 | 10.56 | — | 1.17 | 1.90 | — |
| 16 | f64 | 25.58 | 29.72 | 23.26 | — | 0.86 | 0.78 | — |
| 32 | f32 | 27.84 | 15.54 | 57.81 | — | 1.79 | 3.72 | — |
| 32 | f64 | 85.98 | 79.15 | 151.85 | — | 1.09 | 1.92 | — |
| 64 | f32 | 162.82 | 77.54 | 360.04 | — | 2.10 | 4.64 | — |
| 64 | f64 | 420.82 | 266.99 | 1072.50 | — | 1.58 | 4.02 | — |

### Adaptive vs fixed-sweep symmetric eigensolvers

| N | dtype | syev ns | eig_clamp ns | eigh ns | psd_project ns |
|---|-------|---------|--------------|---------|----------------|
| 4 | f32 | 3.98 | 4.00 | 4.36 | 4.49 |
| 4 | f64 | 58.37 | 59.06 | 85.30 | 86.08 |
| 8 | f32 | 25.88 | 25.86 | 9.37 | 9.49 |
| 8 | f64 | 388.43 | 390.46 | 198.38 | 200.60 |
| 16 | f32 | 116.81 | 117.54 | 55.76 | 57.26 |
| 16 | f64 | 1781.99 | 1754.48 | 753.56 | 715.33 |
| 32 | f32 | 885.50 | 1026.89 | 972.29 | 986.43 |
| 32 | f64 | 8848.46 | 9331.09 | 3887.03 | 5241.45 |

<!-- END tune.py solvers -->

## reduced (`*_reduced` contraction-parallel crossover)

The dtype-independent picker remains constant `false` on sm_120. The
2026-08-14 quiet run found no f32 wins in 48 cells and two f64 wins in 48
cells, both for 4×4×64 at 128/256 threads. That is evidence for an explicit
f64 specialization experiment, not for enabling a type-blind default that
would also affect f32. Callers may still select the explicit `*_reduced`
family.

<!-- BEGIN tune.py reduced -->
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `reduced_sweep_20260814_011659.txt` · tie margin ±5% (reduced must clear it) · 2 of 96 configs pick reduced._

| dtype | M | N | K | n_out | blockDim | serial_us | reduced_us | ratio |
|:------|---|---|---|-------|----------|-----------|------------|-------|
| f64 | 4 | 4 | 64 | 256 | 128 | 1.9115 | 1.3604 | **1.41** |
| f64 | 4 | 4 | 64 | 256 | 256 | 1.9163 | 0.9705 | **1.97** |

Predicate `suggested_use_reduced<n_out,K_contract,blockDim>()` = `false` on every cell (K_contract is the N column here).

⚠️ **2 config(s) disagree** with the predicate — review before trusting the formula on this GPU:

- f64 4×4×64 bd=128 (n_out=256): measured **reduced**, predicate **serial**
- f64 4×4×64 bd=256 (n_out=256): measured **reduced**, predicate **serial**

<!-- END tune.py reduced -->

## nvwarp (audited characterization; no dispatch decision)

The 2026-08-14 sm_120 quiet run (production `-O3` flags, post-`-O1`-revert)
compared `glass::warp::` with `glass::nvidia::warp::` (CUB WarpReduce) at
identical launch shape, using the shared warm-up, spread, provenance, and
isolation protocol. Of 126 cells, 123 tie within ±2%. Three high-throughput
(NPROB=8192) f64 `dot` cells favor CUB by 2.2–3.3%; no cell produced a SIMT
verdict. Six of 252 contender samples exceeded 5% spread, but no shipped
dispatch depends on this characterization. Raw capture:
`perf_nvwarp_20260814_171251.txt` (archived externally).

## robotics (audited characterization; no dispatch decision)

The 2026-08-14 sm_120 quiet run (production `-O3` flags, post-`-O1`-revert)
used 8192 problems and 1000 reps per trial with the shared audited protocol.
The thread tier won every fixed-size row for both dtypes. `softmax_n16` split
by dtype: warp won f32, thread won f64. At the best thread launch, fused and
composed spatial forms are nearly tied in f32; in f64 the fused forms are
8–20% faster (mcross 1.11×, fcross 1.20×, mxform 1.19×, sinertia 1.08×).
16 of 304 individual launch samples exceeded 5% spread, none changing a clear
row verdict. `argmax_fast` is a block-only characterization and does not beat
the ordinary thread-tier `argmax`. Raw capture:
`perf_robotics_20260814_170956.txt` (archived externally).

## Composition & mapping A/B (characterization; no dispatch decision)

2026-08-14 sm_120 quiet run, production `-O3` flags, `bench/perf_sweeps.py
--profile overnight` (composition: NPROB=4096, 100 reps; mapping: NPROB=8192,
100 reps; min of 3 trials). The composition harness cross-checks each
legacy/current pair to tolerance before timing (riccati agreed to 1.0e-7 rel
in f32 / 2.4e-16 in f64; pcg and bdsv bit-exact).

- **`riccati_gain` P·B reuse** (36×12): current beats the former P·A algebra
  **1.19× (f64)** / **1.35× (f32)**. The ratio bundles the algebraic saving
  with the occupancy gain from the smaller scratch (`NU²+NX·NU` vs
  `NU²+NX²`); both variants' spreads ≤1.7%.
- **`pcg` barrier coalescing** (SS=6, KP=32, fixed 1 iteration): 1.06× (f64)
  / 1.01× (f32).
- **`bdsv` compile-time inner calls** (6×16): a wash (1.00× both dtypes).
- **Compile-time `ger` flat one-thread-per-output remap** (bit-identical by
  construction): large wins where the former column loop serialized —
  8.2×/6.5× at 4×128 (f32/f64), 5.6×/3.7× at 8×64, 2.3×/1.09× at 32×32,
  1.34×/1.04× at 64×8.
- The mapping harness's transpose/prefix/getrf rows are exploratory
  candidates, not shipped changes.

Raw captures: `perf_composition_20260814_170927.txt`,
`perf_mapping_20260814_170931.txt` (archived externally).

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild            # compile everything (GPU may be busy)
python3 bench/tune.py --sm auto                       # all legs, timed — QUIET GPU only
python3 bench/tune.py --sm auto --legs blas2,rect     # subset
python3 bench/tune.py --legs solvers --from-solvers <capture.txt> --dry-run   # replay, no GPU
```

Per-harness direct invocations and flags: `bench/TUNING.md`.
