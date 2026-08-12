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
### Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `blas2_sweep_20260812_0134.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 30 of 154 cells._

inv/trmv/ger are BLOCK-ONLY (no `glass::warp::` variant, so nothing competes — reported, never picked); none of these ops has a `glass::nvidia::` counterpart. The 2-impl ops (syrk/syr2k/ldlt/ldltsv) regenerate the shipped per-arch `blas2_sm*` table in glass-defaults.cuh (since 2026-08-06).

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| syrk | N=4 | f32 | 0.56 | 0.17 | **warp** | warp wins (0.170 vs block 0.560, 229.4%) |
| syrk | N=4 | f64 | 0.58 | 0.27 | **warp** | warp wins (0.270 vs block 0.580, 114.8%) |
| syrk | N=6 | f32 | 0.58 | 0.25 | **warp** | warp wins (0.250 vs block 0.580, 132.0%) |
| syrk | N=6 | f64 | 0.64 | 0.57 | **warp** | warp wins (0.570 vs block 0.640, 12.3%) |
| syrk | N=8 | f32 | 0.59 | 0.33 | **warp** | warp wins (0.330 vs block 0.590, 78.8%) |
| syrk | N=8 | f64 | 0.72 | 0.72 | **warp** | warp kept (0.720); block faster by 0.0% but inside ±2% SIMT tie |
| syrk | N=12 | f32 | 0.74 | 0.73 | **warp** | warp wins (0.730 vs block 0.740, 1.4%) |
| syrk | N=12 | f64 | 2.28 | 2.35 | **block** | block wins (2.280 vs warp 2.350, 3.1%) |
| syrk | N=16 | f32 | 1.37 | 1.40 | **block** | block wins (1.370 vs warp 1.400, 2.2%) |
| syrk | N=16 | f64 | 4.63 | 4.78 | **block** | block wins (4.630 vs warp 4.780, 3.2%) |
| syrk | N=24 | f32 | 4.10 | 4.52 | **block** | block wins (4.100 vs warp 4.520, 10.2%) |
| syrk | N=24 | f64 | 15.14 | 16.15 | **block** | block wins (15.140 vs warp 16.150, 6.7%) |
| syrk | N=32 | f32 | 10.77 | 13.91 | **block** | block wins (10.770 vs warp 13.910, 29.2%) |
| syrk | N=32 | f64 | 35.50 | 39.03 | **block** | block wins (35.500 vs warp 39.030, 9.9%) |
| syrk | N=48 | f32 | 30.50 | 63.69 | **block** | block wins (30.500 vs warp 63.690, 108.8%) |
| syrk | N=48 | f64 | 105.64 | 139.19 | **block** | block wins (105.640 vs warp 139.190, 31.8%) |
| syrk | N=64 | f32 | 62.33 | 127.13 | **block** | block wins (62.330 vs warp 127.130, 104.0%) |
| syrk | N=64 | f64 | 212.79 | 565.68 | **block** | block wins (212.790 vs warp 565.680, 165.8%) |
| syrk | N=96 | f32 | 193.75 | 1457.87 | **block** | block wins (193.750 vs warp 1457.870, 652.4%) |
| syrk | N=96 | f64 | 647.98 | 2774.11 | **block** | block wins (647.980 vs warp 2774.110, 328.1%) |
| syrk | N=128 | f32 | 506.50 | 3611.30 | **block** | block wins (506.500 vs warp 3611.300, 613.0%) |
| syrk | N=128 | f64 | 1565.48 | 6677.22 | **block** | block wins (1565.480 vs warp 6677.220, 326.5%) |
| syr2k | N=4 | f32 | 0.58 | 0.22 | **warp** | warp wins (0.220 vs block 0.580, 163.6%) |
| syr2k | N=4 | f64 | 0.64 | 0.61 | **warp** | warp wins (0.610 vs block 0.640, 4.9%) |
| syr2k | N=6 | f32 | 0.60 | 0.37 | **warp** | warp wins (0.370 vs block 0.600, 62.2%) |
| syr2k | N=6 | f64 | 1.53 | 1.56 | **warp** | warp kept (1.560); block faster by 2.0% but inside ±2% SIMT tie |
| syr2k | N=8 | f32 | 0.61 | 0.43 | **warp** | warp wins (0.430 vs block 0.610, 41.9%) |
| syr2k | N=8 | f64 | 1.98 | 2.03 | **block** | block wins (1.980 vs warp 2.030, 2.5%) |
| syr2k | N=12 | f32 | 1.27 | 1.30 | **block** | block wins (1.270 vs warp 1.300, 2.4%) |
| syr2k | N=12 | f64 | 7.05 | 7.29 | **block** | block wins (7.050 vs warp 7.290, 3.4%) |
| syr2k | N=16 | f32 | 2.49 | 2.63 | **block** | block wins (2.490 vs warp 2.630, 5.6%) |
| syr2k | N=16 | f64 | 14.79 | 15.25 | **block** | block wins (14.790 vs warp 15.250, 3.1%) |
| syr2k | N=24 | f32 | 7.77 | 10.76 | **block** | block wins (7.770 vs warp 10.760, 38.5%) |
| syr2k | N=24 | f64 | 49.99 | 51.34 | **block** | block wins (49.990 vs warp 51.340, 2.7%) |
| syr2k | N=32 | f32 | 19.79 | 31.12 | **block** | block wins (19.790 vs warp 31.120, 57.3%) |
| syr2k | N=32 | f64 | 117.84 | 126.16 | **block** | block wins (117.840 vs warp 126.160, 7.1%) |
| syr2k | N=48 | f32 | 57.87 | 129.88 | **block** | block wins (57.870 vs warp 129.880, 124.4%) |
| syr2k | N=48 | f64 | 353.33 | 579.48 | **block** | block wins (353.330 vs warp 579.480, 64.0%) |
| syr2k | N=64 | f32 | 119.74 | 719.57 | **block** | block wins (119.740 vs warp 719.570, 500.9%) |
| syr2k | N=64 | f64 | 705.94 | 1555.43 | **block** | block wins (705.940 vs warp 1555.430, 120.3%) |
| syr2k | N=96 | f32 | 363.53 | 3248.50 | **block** | block wins (363.530 vs warp 3248.500, 793.6%) |
| syr2k | N=96 | f64 | 2118.46 | 5431.75 | **block** | block wins (2118.460 vs warp 5431.750, 156.4%) |
| syr2k | N=128 | f32 | 1259.93 | 6999.18 | **block** | block wins (1259.930 vs warp 6999.180, 455.5%) |
| syr2k | N=128 | f64 | 5322.71 | 12652.39 | **block** | block wins (5322.710 vs warp 12652.390, 137.7%) |
| ldlt | N=4 | f32 | 0.64 | 0.39 | **warp** | warp wins (0.390 vs block 0.640, 64.1%) |
| ldlt | N=4 | f64 | 1.95 | 3.35 | **block** | block wins (1.950 vs warp 3.350, 71.8%) |
| ldlt | N=6 | f32 | 1.04 | 0.82 | **warp** | warp wins (0.820 vs block 1.040, 26.8%) |
| ldlt | N=6 | f64 | 4.50 | 6.45 | **block** | block wins (4.500 vs warp 6.450, 43.3%) |
| ldlt | N=8 | f32 | 1.59 | 1.29 | **warp** | warp wins (1.290 vs block 1.590, 23.3%) |
| ldlt | N=8 | f64 | 7.12 | 10.04 | **block** | block wins (7.120 vs warp 10.040, 41.0%) |
| ldlt | N=12 | f32 | 3.00 | 2.43 | **warp** | warp wins (2.430 vs block 3.000, 23.5%) |
| ldlt | N=12 | f64 | 14.24 | 19.03 | **block** | block wins (14.240 vs warp 19.030, 33.6%) |
| ldlt | N=16 | f32 | 4.65 | 3.86 | **warp** | warp wins (3.860 vs block 4.650, 20.5%) |
| ldlt | N=16 | f64 | 24.20 | 30.36 | **block** | block wins (24.200 vs warp 30.360, 25.5%) |
| ldlt | N=24 | f32 | 9.07 | 7.96 | **warp** | warp wins (7.960 vs block 9.070, 13.9%) |
| ldlt | N=24 | f64 | 52.20 | 60.65 | **block** | block wins (52.200 vs warp 60.650, 16.2%) |
| ldlt | N=32 | f32 | 16.23 | 14.14 | **warp** | warp wins (14.140 vs block 16.230, 14.8%) |
| ldlt | N=32 | f64 | 90.60 | 101.01 | **block** | block wins (90.600 vs warp 101.010, 11.5%) |
| ldlt | N=48 | f32 | 44.00 | 43.68 | **warp** | warp wins (43.680 vs block 44.000, 0.7%) |
| ldlt | N=48 | f64 | 215.94 | 235.46 | **block** | block wins (215.940 vs warp 235.460, 9.0%) |
| ldlt | N=64 | f32 | 93.78 | 93.86 | **warp** | warp kept (93.860); block faster by 0.1% but inside ±2% SIMT tie |
| ldlt | N=64 | f64 | 404.17 | 433.06 | **block** | block wins (404.170 vs warp 433.060, 7.1%) |
| ldlt | N=96 | f32 | 279.89 | 305.11 | **block** | block wins (279.890 vs warp 305.110, 9.0%) |
| ldlt | N=96 | f64 | 1022.12 | 1075.51 | **block** | block wins (1022.120 vs warp 1075.510, 5.2%) |
| ldlt | N=128 | f32 | 664.65 | 807.90 | **block** | block wins (664.650 vs warp 807.900, 21.6%) |
| ldlt | N=128 | f64 | 2039.70 | 2460.05 | **block** | block wins (2039.700 vs warp 2460.050, 20.6%) |
| ldltsv | N=4 | f32 | 0.74 | 0.59 | **warp** | warp wins (0.590 vs block 0.740, 25.4%) |
| ldltsv | N=4 | f64 | 2.99 | 4.39 | **block** | block wins (2.990 vs warp 4.390, 46.8%) |
| ldltsv | N=6 | f32 | 1.43 | 1.02 | **warp** | warp wins (1.020 vs block 1.430, 40.2%) |
| ldltsv | N=6 | f64 | 5.75 | 7.62 | **block** | block wins (5.750 vs warp 7.620, 32.5%) |
| ldltsv | N=8 | f32 | 2.13 | 1.54 | **warp** | warp wins (1.540 vs block 2.130, 38.3%) |
| ldltsv | N=8 | f64 | 8.56 | 11.34 | **block** | block wins (8.560 vs warp 11.340, 32.5%) |
| ldltsv | N=12 | f32 | 3.92 | 2.83 | **warp** | warp wins (2.830 vs block 3.920, 38.5%) |
| ldltsv | N=12 | f64 | 16.02 | 20.66 | **block** | block wins (16.020 vs warp 20.660, 29.0%) |
| ldltsv | N=16 | f32 | 5.81 | 4.43 | **warp** | warp wins (4.430 vs block 5.810, 31.2%) |
| ldltsv | N=16 | f64 | 26.51 | 32.43 | **block** | block wins (26.510 vs warp 32.430, 22.3%) |
| ldltsv | N=24 | f32 | 10.88 | 8.85 | **warp** | warp wins (8.850 vs block 10.880, 22.9%) |
| ldltsv | N=24 | f64 | 54.97 | 63.41 | **block** | block wins (54.970 vs warp 63.410, 15.4%) |
| ldltsv | N=32 | f32 | 19.80 | 15.91 | **warp** | warp wins (15.910 vs block 19.800, 24.5%) |
| ldltsv | N=32 | f64 | 94.25 | 104.40 | **block** | block wins (94.250 vs warp 104.400, 10.8%) |
| ldltsv | N=48 | f32 | 52.12 | 49.74 | **warp** | warp wins (49.740 vs block 52.120, 4.8%) |
| ldltsv | N=48 | f64 | 223.74 | 243.21 | **block** | block wins (223.740 vs warp 243.210, 8.7%) |
| ldltsv | N=64 | f32 | 106.13 | 104.58 | **warp** | warp wins (104.580 vs block 106.130, 1.5%) |
| ldltsv | N=64 | f64 | 416.40 | 444.58 | **block** | block wins (416.400 vs warp 444.580, 6.8%) |
| ldltsv | N=96 | f32 | 301.55 | 319.27 | **block** | block wins (301.550 vs warp 319.270, 5.9%) |
| ldltsv | N=96 | f64 | 1047.05 | 1097.53 | **block** | block wins (1047.050 vs warp 1097.530, 4.8%) |
| ldltsv | N=128 | f32 | 734.54 | 863.48 | **block** | block wins (734.540 vs warp 863.480, 17.6%) |
| ldltsv | N=128 | f64 | 2050.75 | 2537.43 | **block** | block wins (2050.750 vs warp 2537.430, 23.7%) |
| inv | N=4 | f32 | 0.71 | — | **block** | block only impl measured (0.710) |
| inv | N=4 | f64 | 1.44 | — | **block** | block only impl measured (1.440) |
| inv | N=6 | f32 | 1.12 | — | **block** | block only impl measured (1.120) |
| inv | N=6 | f64 | 2.94 | — | **block** | block only impl measured (2.940) |
| inv | N=8 | f32 | 1.83 | — | **block** | block only impl measured (1.830) |
| inv | N=8 | f64 | 4.99 | — | **block** | block only impl measured (4.990) |
| inv | N=12 | f32 | 4.56 | — | **block** | block only impl measured (4.560) |
| inv | N=12 | f64 | 10.56 | — | **block** | block only impl measured (10.560) |
| inv | N=16 | f32 | 9.33 | — | **block** | block only impl measured (9.330) |
| inv | N=16 | f64 | 22.32 | — | **block** | block only impl measured (22.320) |
| inv | N=24 | f32 | 26.45 | — | **block** | block only impl measured (26.450) |
| inv | N=24 | f64 | 63.81 | — | **block** | block only impl measured (63.810) |
| inv | N=32 | f32 | 52.06 | — | **block** | block only impl measured (52.060) |
| inv | N=32 | f64 | 139.92 | — | **block** | block only impl measured (139.920) |
| inv | N=48 | f32 | 171.74 | — | **block** | block only impl measured (171.740) |
| inv | N=48 | f64 | 453.72 | — | **block** | block only impl measured (453.720) |
| inv | N=64 | f32 | 358.10 | — | **block** | block only impl measured (358.100) |
| inv | N=64 | f64 | 1025.52 | — | **block** | block only impl measured (1025.520) |
| inv | N=96 | f32 | 1457.55 | — | **block** | block only impl measured (1457.550) |
| inv | N=96 | f64 | 3411.06 | — | **block** | block only impl measured (3411.060) |
| inv | N=128 | f32 | 3474.03 | — | **block** | block only impl measured (3474.030) |
| inv | N=128 | f64 | 18684.17 | — | **block** | block only impl measured (18684.170) |
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
| trmv | N=32 | f32 | 1.15 | — | **block** | block only impl measured (1.150) |
| trmv | N=32 | f64 | 2.24 | — | **block** | block only impl measured (2.240) |
| trmv | N=48 | f32 | 1.93 | — | **block** | block only impl measured (1.930) |
| trmv | N=48 | f64 | 5.39 | — | **block** | block only impl measured (5.390) |
| trmv | N=64 | f32 | 3.47 | — | **block** | block only impl measured (3.470) |
| trmv | N=64 | f64 | 12.38 | — | **block** | block only impl measured (12.380) |
| trmv | N=96 | f32 | 15.07 | — | **block** | block only impl measured (15.070) |
| trmv | N=96 | f64 | 26.30 | — | **block** | block only impl measured (26.300) |
| trmv | N=128 | f32 | 24.89 | — | **block** | block only impl measured (24.890) |
| trmv | N=128 | f64 | 44.26 | — | **block** | block only impl measured (44.260) |
| ger | N=4 | f32 | 0.59 | — | **block** | block only impl measured (0.590) |
| ger | N=4 | f64 | 0.62 | — | **block** | block only impl measured (0.620) |
| ger | N=6 | f32 | 0.63 | — | **block** | block only impl measured (0.630) |
| ger | N=6 | f64 | 0.67 | — | **block** | block only impl measured (0.670) |
| ger | N=8 | f32 | 0.83 | — | **block** | block only impl measured (0.830) |
| ger | N=8 | f64 | 0.89 | — | **block** | block only impl measured (0.890) |
| ger | N=12 | f32 | 1.08 | — | **block** | block only impl measured (1.080) |
| ger | N=12 | f64 | 1.15 | — | **block** | block only impl measured (1.150) |
| ger | N=16 | f32 | 1.50 | — | **block** | block only impl measured (1.500) |
| ger | N=16 | f64 | 1.61 | — | **block** | block only impl measured (1.610) |
| ger | N=24 | f32 | 1.99 | — | **block** | block only impl measured (1.990) |
| ger | N=24 | f64 | 2.32 | — | **block** | block only impl measured (2.320) |
| ger | N=32 | f32 | 2.76 | — | **block** | block only impl measured (2.760) |
| ger | N=32 | f64 | 3.47 | — | **block** | block only impl measured (3.470) |
| ger | N=48 | f32 | 4.83 | — | **block** | block only impl measured (4.830) |
| ger | N=48 | f64 | 26.79 | — | **block** | block only impl measured (26.790) |
| ger | N=64 | f32 | 23.65 | — | **block** | block only impl measured (23.650) |
| ger | N=64 | f64 | 45.57 | — | **block** | block only impl measured (45.570) |
| ger | N=96 | f32 | 55.02 | — | **block** | block only impl measured (55.020) |
| ger | N=96 | f64 | 101.38 | — | **block** | block only impl measured (101.380) |
| ger | N=128 | f32 | 90.75 | — | **block** | block only impl measured (90.750) |
| ger | N=128 | f64 | 177.45 | — | **block** | block only impl measured (177.450) |

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

_Source: `rect_sweep_20260812_0331.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 16 of 24 cells._

nvidia leg skipped for rectangular shapes (needs new per-shape DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in the `shapes` leg). Measured shapes regenerate the shipped exact-shape `rect_*_sm*` pickers in glass-defaults.cuh (`suggested_backend_rect_gemv/gemm<>`, since 2026-08-06); unmeasured shapes stay block.

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| gemv | 8x64 | f32 | 1.12 | 0.99 | **warp** | warp wins (0.990 vs block 1.120, 13.1%) |
| gemv | 8x64 | f64 | 2.49 | 2.47 | **warp** | warp wins (2.470 vs block 2.490, 0.8%) |
| gemv | 16x128 | f32 | 2.35 | 1.87 | **warp** | warp wins (1.870 vs block 2.350, 25.7%) |
| gemv | 16x128 | f64 | 12.14 | 10.78 | **warp** | warp wins (10.780 vs block 12.140, 12.6%) |
| gemv | 32x256 | f32 | 23.47 | 20.79 | **warp** | warp wins (20.790 vs block 23.470, 12.9%) |
| gemv | 32x256 | f64 | 42.75 | 41.28 | **warp** | warp wins (41.280 vs block 42.750, 3.6%) |
| gemv | 64x8 | f32 | 0.64 | 0.47 | **warp** | warp wins (0.470 vs block 0.640, 36.2%) |
| gemv | 64x8 | f64 | 0.81 | 0.80 | **warp** | warp wins (0.800 vs block 0.810, 1.2%) |
| gemv | 128x16 | f32 | 1.36 | 1.54 | **block** | block wins (1.360 vs warp 1.540, 13.2%) |
| gemv | 128x16 | f64 | 10.44 | 10.57 | **warp** | warp kept (10.570); block faster by 1.2% but inside ±2% SIMT tie |
| gemv | 256x32 | f32 | 20.33 | 20.42 | **warp** | warp kept (20.420); block faster by 0.4% but inside ±2% SIMT tie |
| gemv | 256x32 | f64 | 40.29 | 40.75 | **warp** | warp kept (40.750); block faster by 1.1% but inside ±2% SIMT tie |
| gemm | 6x6x64 | f32 | 0.89 | 0.92 | **block** | block wins (0.890 vs warp 0.920, 3.4%) |
| gemm | 6x6x64 | f64 | 2.74 | 3.02 | **block** | block wins (2.740 vs warp 3.020, 10.2%) |
| gemm | 8x32x8 | f32 | 0.92 | 0.89 | **warp** | warp wins (0.890 vs block 0.920, 3.4%) |
| gemm | 8x32x8 | f64 | 2.42 | 2.61 | **block** | block wins (2.420 vs warp 2.610, 7.9%) |
| gemm | 16x64x16 | f32 | 3.76 | 3.94 | **block** | block wins (3.760 vs warp 3.940, 4.8%) |
| gemm | 16x64x16 | f64 | 18.52 | 19.16 | **block** | block wins (18.520 vs warp 19.160, 3.5%) |
| gemm | 32x8x32 | f32 | 1.71 | 1.52 | **warp** | warp wins (1.520 vs block 1.710, 12.5%) |
| gemm | 32x8x32 | f64 | 9.61 | 9.83 | **block** | block wins (9.610 vs warp 9.830, 2.3%) |
| gemm | 64x6x6 | f32 | 0.77 | 0.72 | **warp** | warp wins (0.720 vs block 0.770, 6.9%) |
| gemm | 64x6x6 | f64 | 3.59 | 3.43 | **warp** | warp wins (3.430 vs block 3.590, 4.7%) |
| gemm | 64x16x16 | f32 | 2.46 | 2.04 | **warp** | warp wins (2.040 vs block 2.460, 20.6%) |
| gemm | 64x16x16 | f64 | 18.24 | 20.18 | **block** | block wins (18.240 vs warp 20.180, 10.6%) |

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

_Source: `solvers_sweep_20260812_0332.txt` · NPROB=8192 ns/problem (best swept TB, min of 3 trials, restore-outside-timing protocol) · characterization only — no dispatch table is regenerated._

### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input

bdsv is faster in 1 of 12 cells **on this well-conditioned test system** (see the iters column — pcg's cost scales with the iteration count, so the crossover moves with conditioning).

| BlockSize | Knots | dtype | bdsv ns | pcg ns | pcg iters | pcg/bdsv |
|-----------|-------|-------|---------|--------|-----------|----------|
| 2 | 8 | f32 | 6.24 | 2.48 | 3 | 0.40 |
| 2 | 8 | f64 | 26.25 | 8.25 | 3 | 0.31 |
| 2 | 32 | f32 | 24.50 | 3.49 | 3 | 0.14 |
| 2 | 32 | f64 | 107.38 | 11.71 | 3 | 0.11 |
| 6 | 8 | f32 | 18.99 | 6.40 | 3 | 0.34 |
| 6 | 8 | f64 | 94.66 | 31.33 | 3 | 0.33 |
| 6 | 32 | f32 | 86.40 | 30.62 | 3 | 0.35 |
| 6 | 32 | f64 | 391.92 | 130.64 | 3 | 0.33 |
| 6 | 64 | f32 | 180.66 | 83.42 | 3 | 0.46 |
| 6 | 64 | f64 | 792.62 | 252.35 | 3 | 0.32 |
| 12 | 16 | f32 | 110.68 | 198.28 | 2 | 1.79 |
| 12 | 16 | f64 | 463.21 | 230.33 | 2 | 0.50 |

### gesv vs posv vs inv+gemv — same SPD system, single RHS

posv (Cholesky) is the intended SPD path; gesv prices the pivoted-LU robustness fallback, inv+gemv the invert-then-multiply anti-pattern.

The `thr-posv` column is the **thread-tier** `glass::thread::posv` (one problem per thread, 32 packed per warp) — measured only below the N<=7 register-residency ceiling. Where `thr/posv` < 1 the thread tier beats the block Cholesky solve on that low-DOF shape.

| N | dtype | gesv ns | posv ns | inv+gemv ns | thr-posv ns | gesv/posv | inv/posv | thr/posv |
|---|-------|---------|---------|-------------|-------------|-----------|----------|----------|
| 4 | f32 | 1.28 | 1.08 | 0.99 | 0.39 | 1.19 | 0.92 | 0.36 |
| 4 | f64 | 3.74 | 5.58 | 1.99 | 0.97 | 0.67 | 0.36 | 0.17 |
| 8 | f32 | 2.45 | 2.34 | 2.25 | — | 1.05 | 0.96 | — |
| 8 | f64 | 9.02 | 12.51 | 5.65 | — | 0.72 | 0.45 | — |
| 16 | f32 | 6.52 | 5.54 | 10.57 | — | 1.18 | 1.91 | — |
| 16 | f64 | 25.56 | 29.69 | 23.27 | — | 0.86 | 0.78 | — |
| 32 | f32 | 27.79 | 15.36 | 57.72 | — | 1.81 | 3.76 | — |
| 32 | f64 | 86.07 | 78.93 | 151.81 | — | 1.09 | 1.92 | — |
| 64 | f32 | 162.99 | 77.67 | 360.32 | — | 2.10 | 4.64 | — |
| 64 | f64 | 420.75 | 267.05 | 1073.04 | — | 1.58 | 4.02 | — |

### syev + eig_clamp — timing only (no contender)

| N | dtype | syev ns | eig_clamp ns |
|---|-------|---------|--------------|
| 4 | f32 | 3.99 | 3.99 |
| 4 | f64 | 58.34 | 59.14 |
| 8 | f32 | 25.89 | 25.81 |
| 8 | f64 | 388.20 | 390.60 |
| 16 | f32 | 116.85 | 117.70 |
| 16 | f64 | 1780.10 | 1751.59 |
| 32 | f32 | 884.79 | 1026.92 |
| 32 | f64 | 8848.90 | 9332.54 |

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
