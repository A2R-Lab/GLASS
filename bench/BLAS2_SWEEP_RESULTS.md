# blas2 sweep — warp/block picks for the ladder's blind-spot ops

Source: `bench/bench_blas2.cu`, driven by `bench/tune.py --legs blas2`. Ops the
mega-sweep ladder does not cover: `syrk`, `syr2k`, `ldlt`, `ldltsv`
(= `ldlt` + `ldlt_solve`, the LDLᵀ analogue of the ladder's `posv` row), `inv`
(Gauss-Jordan on the augmented `[A | I]` layout), `trmv`, and `ger`. Square N
over the ladder's N set {4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}, f32 + f64,
one problem per block (BLOCK, TB ∈ {32, 64, 128, 256}) vs one problem per warp
(WARP, WPB ∈ {1..32}).

Two contenders only:

- **No nvidia leg** — none of these ops has a `glass::nvidia::` counterpart.
- `inv`, `trmv`, `ger` are **block-only** (no `glass::warp::` variant exists);
  they are measured and recorded, not picked.

These measurements do **not** regenerate a shipped header table yet — the
`suggested_backend<>` defaults-table extension for these ops is a follow-up.
Until then this file is the record of the measured winner per (op, N, dtype);
the block between the markers below is auto-refreshed by `bench/tune.py`
through the shared `tune_pick` margin rule.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `blas2_sweep_20260718_0327.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 33 of 154 cells._

inv/trmv/ger are BLOCK-ONLY (no `glass::warp::` variant, so nothing competes — reported, never picked); none of these ops has a `glass::nvidia::` counterpart. The 2-impl ops (syrk/syr2k/ldlt/ldltsv) regenerate the shipped per-arch `blas2_sm*` table in glass-defaults.cuh (since 2026-08-06).

| op | shape | dtype | block ns | warp ns | pick | note |
|----|-------|-------|----------|---------|------|------|
| syrk | N=4 | f32 | 0.57 | 0.18 | **warp** | warp wins (0.180 vs block 0.570, 216.7%) |
| syrk | N=4 | f64 | 0.59 | 0.30 | **warp** | warp wins (0.300 vs block 0.590, 96.7%) |
| syrk | N=6 | f32 | 0.59 | 0.28 | **warp** | warp wins (0.280 vs block 0.590, 110.7%) |
| syrk | N=6 | f64 | 0.70 | 0.67 | **warp** | warp wins (0.670 vs block 0.700, 4.5%) |
| syrk | N=8 | f32 | 0.60 | 0.35 | **warp** | warp wins (0.350 vs block 0.600, 71.4%) |
| syrk | N=8 | f64 | 0.83 | 0.84 | **warp** | warp kept (0.840); block faster by 1.2% but inside ±2% SIMT tie |
| syrk | N=12 | f32 | 0.85 | 0.85 | **warp** | warp kept (0.850); block faster by 0.0% but inside ±2% SIMT tie |
| syrk | N=12 | f64 | 2.66 | 2.68 | **warp** | warp kept (2.680); block faster by 0.8% but inside ±2% SIMT tie |
| syrk | N=16 | f32 | 1.57 | 1.54 | **warp** | warp wins (1.540 vs block 1.570, 1.9%) |
| syrk | N=16 | f64 | 5.06 | 5.15 | **warp** | warp kept (5.150); block faster by 1.8% but inside ±2% SIMT tie |
| syrk | N=24 | f32 | 4.50 | 4.86 | **block** | block wins (4.500 vs warp 4.860, 8.0%) |
| syrk | N=24 | f64 | 15.15 | 16.16 | **block** | block wins (15.150 vs warp 16.160, 6.7%) |
| syrk | N=32 | f32 | 10.78 | 13.82 | **block** | block wins (10.780 vs warp 13.820, 28.2%) |
| syrk | N=32 | f64 | 35.48 | 39.01 | **block** | block wins (35.480 vs warp 39.010, 9.9%) |
| syrk | N=48 | f32 | 30.36 | 63.01 | **block** | block wins (30.360 vs warp 63.010, 107.5%) |
| syrk | N=48 | f64 | 105.64 | 137.84 | **block** | block wins (105.640 vs warp 137.840, 30.5%) |
| syrk | N=64 | f32 | 62.25 | 131.38 | **block** | block wins (62.250 vs warp 131.380, 111.1%) |
| syrk | N=64 | f64 | 212.70 | 591.51 | **block** | block wins (212.700 vs warp 591.510, 178.1%) |
| syrk | N=96 | f32 | 193.40 | 1469.61 | **block** | block wins (193.400 vs warp 1469.610, 659.9%) |
| syrk | N=96 | f64 | 647.91 | 2775.73 | **block** | block wins (647.910 vs warp 2775.730, 328.4%) |
| syrk | N=128 | f32 | 506.36 | 3603.21 | **block** | block wins (506.360 vs warp 3603.210, 611.6%) |
| syrk | N=128 | f64 | 1565.42 | 6686.33 | **block** | block wins (1565.420 vs warp 6686.330, 327.1%) |
| syr2k | N=4 | f32 | 0.58 | 0.22 | **warp** | warp wins (0.220 vs block 0.580, 163.6%) |
| syr2k | N=4 | f64 | 0.64 | 0.61 | **warp** | warp wins (0.610 vs block 0.640, 4.9%) |
| syr2k | N=6 | f32 | 0.60 | 0.37 | **warp** | warp wins (0.370 vs block 0.600, 62.2%) |
| syr2k | N=6 | f64 | 1.52 | 1.56 | **block** | block wins (1.520 vs warp 1.560, 2.6%) |
| syr2k | N=8 | f32 | 0.61 | 0.43 | **warp** | warp wins (0.430 vs block 0.610, 41.9%) |
| syr2k | N=8 | f64 | 2.04 | 2.08 | **warp** | warp kept (2.080); block faster by 2.0% but inside ±2% SIMT tie |
| syr2k | N=12 | f32 | 1.27 | 1.30 | **block** | block wins (1.270 vs warp 1.300, 2.4%) |
| syr2k | N=12 | f64 | 7.10 | 7.36 | **block** | block wins (7.100 vs warp 7.360, 3.7%) |
| syr2k | N=16 | f32 | 2.56 | 2.69 | **block** | block wins (2.560 vs warp 2.690, 5.1%) |
| syr2k | N=16 | f64 | 14.79 | 15.25 | **block** | block wins (14.790 vs warp 15.250, 3.1%) |
| syr2k | N=24 | f32 | 7.77 | 10.80 | **block** | block wins (7.770 vs warp 10.800, 39.0%) |
| syr2k | N=24 | f64 | 49.98 | 51.53 | **block** | block wins (49.980 vs warp 51.530, 3.1%) |
| syr2k | N=32 | f32 | 19.75 | 31.12 | **block** | block wins (19.750 vs warp 31.120, 57.6%) |
| syr2k | N=32 | f64 | 117.86 | 127.04 | **block** | block wins (117.860 vs warp 127.040, 7.8%) |
| syr2k | N=48 | f32 | 57.89 | 124.27 | **block** | block wins (57.890 vs warp 124.270, 114.7%) |
| syr2k | N=48 | f64 | 353.29 | 574.34 | **block** | block wins (353.290 vs warp 574.340, 62.6%) |
| syr2k | N=64 | f32 | 119.70 | 726.68 | **block** | block wins (119.700 vs warp 726.680, 507.1%) |
| syr2k | N=64 | f64 | 705.95 | 1549.77 | **block** | block wins (705.950 vs warp 1549.770, 119.5%) |
| syr2k | N=96 | f32 | 362.45 | 3254.67 | **block** | block wins (362.450 vs warp 3254.670, 798.0%) |
| syr2k | N=96 | f64 | 2118.30 | 5432.56 | **block** | block wins (2118.300 vs warp 5432.560, 156.5%) |
| syr2k | N=128 | f32 | 1218.67 | 6994.90 | **block** | block wins (1218.670 vs warp 6994.900, 474.0%) |
| syr2k | N=128 | f64 | 5318.70 | 12656.18 | **block** | block wins (5318.700 vs warp 12656.180, 138.0%) |
| ldlt | N=4 | f32 | 0.64 | 0.39 | **warp** | warp wins (0.390 vs block 0.640, 64.1%) |
| ldlt | N=4 | f64 | 1.95 | 3.36 | **block** | block wins (1.950 vs warp 3.360, 72.3%) |
| ldlt | N=6 | f32 | 1.02 | 0.81 | **warp** | warp wins (0.810 vs block 1.020, 25.9%) |
| ldlt | N=6 | f64 | 4.50 | 6.44 | **block** | block wins (4.500 vs warp 6.440, 43.1%) |
| ldlt | N=8 | f32 | 1.58 | 1.29 | **warp** | warp wins (1.290 vs block 1.580, 22.5%) |
| ldlt | N=8 | f64 | 7.12 | 10.04 | **block** | block wins (7.120 vs warp 10.040, 41.0%) |
| ldlt | N=12 | f32 | 3.01 | 2.41 | **warp** | warp wins (2.410 vs block 3.010, 24.9%) |
| ldlt | N=12 | f64 | 14.24 | 19.03 | **block** | block wins (14.240 vs warp 19.030, 33.6%) |
| ldlt | N=16 | f32 | 4.63 | 3.84 | **warp** | warp wins (3.840 vs block 4.630, 20.6%) |
| ldlt | N=16 | f64 | 24.21 | 30.34 | **block** | block wins (24.210 vs warp 30.340, 25.3%) |
| ldlt | N=24 | f32 | 9.04 | 7.95 | **warp** | warp wins (7.950 vs block 9.040, 13.7%) |
| ldlt | N=24 | f64 | 52.19 | 60.65 | **block** | block wins (52.190 vs warp 60.650, 16.2%) |
| ldlt | N=32 | f32 | 16.20 | 14.12 | **warp** | warp wins (14.120 vs block 16.200, 14.7%) |
| ldlt | N=32 | f64 | 90.66 | 101.02 | **block** | block wins (90.660 vs warp 101.020, 11.4%) |
| ldlt | N=48 | f32 | 43.95 | 43.19 | **warp** | warp wins (43.190 vs block 43.950, 1.8%) |
| ldlt | N=48 | f64 | 215.78 | 235.33 | **block** | block wins (215.780 vs warp 235.330, 9.1%) |
| ldlt | N=64 | f32 | 94.15 | 91.89 | **warp** | warp wins (91.890 vs block 94.150, 2.5%) |
| ldlt | N=64 | f64 | 403.83 | 432.63 | **block** | block wins (403.830 vs warp 432.630, 7.1%) |
| ldlt | N=96 | f32 | 275.06 | 303.61 | **block** | block wins (275.060 vs warp 303.610, 10.4%) |
| ldlt | N=96 | f64 | 1021.88 | 1073.74 | **block** | block wins (1021.880 vs warp 1073.740, 5.1%) |
| ldlt | N=128 | f32 | 664.25 | 810.02 | **block** | block wins (664.250 vs warp 810.020, 21.9%) |
| ldlt | N=128 | f64 | 2040.17 | 2461.27 | **block** | block wins (2040.170 vs warp 2461.270, 20.6%) |
| ldltsv | N=4 | f32 | 0.74 | 0.59 | **warp** | warp wins (0.590 vs block 0.740, 25.4%) |
| ldltsv | N=4 | f64 | 2.98 | 4.39 | **block** | block wins (2.980 vs warp 4.390, 47.3%) |
| ldltsv | N=6 | f32 | 1.42 | 1.01 | **warp** | warp wins (1.010 vs block 1.420, 40.6%) |
| ldltsv | N=6 | f64 | 5.76 | 7.61 | **block** | block wins (5.760 vs warp 7.610, 32.1%) |
| ldltsv | N=8 | f32 | 2.12 | 1.52 | **warp** | warp wins (1.520 vs block 2.120, 39.5%) |
| ldltsv | N=8 | f64 | 8.56 | 11.34 | **block** | block wins (8.560 vs warp 11.340, 32.5%) |
| ldltsv | N=12 | f32 | 3.90 | 2.81 | **warp** | warp wins (2.810 vs block 3.900, 38.8%) |
| ldltsv | N=12 | f64 | 16.03 | 20.65 | **block** | block wins (16.030 vs warp 20.650, 28.8%) |
| ldltsv | N=16 | f32 | 5.76 | 4.41 | **warp** | warp wins (4.410 vs block 5.760, 30.6%) |
| ldltsv | N=16 | f64 | 26.49 | 32.43 | **block** | block wins (26.490 vs warp 32.430, 22.4%) |
| ldltsv | N=24 | f32 | 10.82 | 8.76 | **warp** | warp wins (8.760 vs block 10.820, 23.5%) |
| ldltsv | N=24 | f64 | 54.86 | 63.37 | **block** | block wins (54.860 vs warp 63.370, 15.5%) |
| ldltsv | N=32 | f32 | 19.69 | 15.90 | **warp** | warp wins (15.900 vs block 19.690, 23.8%) |
| ldltsv | N=32 | f64 | 94.24 | 104.46 | **block** | block wins (94.240 vs warp 104.460, 10.8%) |
| ldltsv | N=48 | f32 | 51.91 | 48.68 | **warp** | warp wins (48.680 vs block 51.910, 6.6%) |
| ldltsv | N=48 | f64 | 223.39 | 242.91 | **block** | block wins (223.390 vs warp 242.910, 8.7%) |
| ldltsv | N=64 | f32 | 105.76 | 103.58 | **warp** | warp wins (103.580 vs block 105.760, 2.1%) |
| ldltsv | N=64 | f64 | 416.09 | 444.49 | **block** | block wins (416.090 vs warp 444.490, 6.8%) |
| ldltsv | N=96 | f32 | 300.98 | 316.91 | **block** | block wins (300.980 vs warp 316.910, 5.3%) |
| ldltsv | N=96 | f64 | 1045.04 | 1093.18 | **block** | block wins (1045.040 vs warp 1093.180, 4.6%) |
| ldltsv | N=128 | f32 | 735.22 | 865.94 | **block** | block wins (735.220 vs warp 865.940, 17.8%) |
| ldltsv | N=128 | f64 | 2049.87 | 2531.37 | **block** | block wins (2049.870 vs warp 2531.370, 23.5%) |
| inv | N=4 | f32 | 0.70 | — | **block** | block only impl measured (0.700) |
| inv | N=4 | f64 | 1.44 | — | **block** | block only impl measured (1.440) |
| inv | N=6 | f32 | 1.11 | — | **block** | block only impl measured (1.110) |
| inv | N=6 | f64 | 2.94 | — | **block** | block only impl measured (2.940) |
| inv | N=8 | f32 | 1.82 | — | **block** | block only impl measured (1.820) |
| inv | N=8 | f64 | 4.99 | — | **block** | block only impl measured (4.990) |
| inv | N=12 | f32 | 4.58 | — | **block** | block only impl measured (4.580) |
| inv | N=12 | f64 | 10.55 | — | **block** | block only impl measured (10.550) |
| inv | N=16 | f32 | 9.43 | — | **block** | block only impl measured (9.430) |
| inv | N=16 | f64 | 22.27 | — | **block** | block only impl measured (22.270) |
| inv | N=24 | f32 | 26.36 | — | **block** | block only impl measured (26.360) |
| inv | N=24 | f64 | 63.78 | — | **block** | block only impl measured (63.780) |
| inv | N=32 | f32 | 52.00 | — | **block** | block only impl measured (52.000) |
| inv | N=32 | f64 | 139.74 | — | **block** | block only impl measured (139.740) |
| inv | N=48 | f32 | 171.85 | — | **block** | block only impl measured (171.850) |
| inv | N=48 | f64 | 453.44 | — | **block** | block only impl measured (453.440) |
| inv | N=64 | f32 | 357.48 | — | **block** | block only impl measured (357.480) |
| inv | N=64 | f64 | 1024.77 | — | **block** | block only impl measured (1024.770) |
| inv | N=96 | f32 | 1453.96 | — | **block** | block only impl measured (1453.960) |
| inv | N=96 | f64 | 3409.66 | — | **block** | block only impl measured (3409.660) |
| inv | N=128 | f32 | 3471.64 | — | **block** | block only impl measured (3471.640) |
| inv | N=128 | f64 | 18666.99 | — | **block** | block only impl measured (18666.990) |
| trmv | N=4 | f32 | 0.59 | — | **block** | block only impl measured (0.590) |
| trmv | N=4 | f64 | 0.63 | — | **block** | block only impl measured (0.630) |
| trmv | N=6 | f32 | 0.60 | — | **block** | block only impl measured (0.600) |
| trmv | N=6 | f64 | 0.68 | — | **block** | block only impl measured (0.680) |
| trmv | N=8 | f32 | 0.62 | — | **block** | block only impl measured (0.620) |
| trmv | N=8 | f64 | 0.83 | — | **block** | block only impl measured (0.830) |
| trmv | N=12 | f32 | 0.64 | — | **block** | block only impl measured (0.640) |
| trmv | N=12 | f64 | 0.93 | — | **block** | block only impl measured (0.930) |
| trmv | N=16 | f32 | 0.75 | — | **block** | block only impl measured (0.750) |
| trmv | N=16 | f64 | 1.55 | — | **block** | block only impl measured (1.550) |
| trmv | N=24 | f32 | 0.83 | — | **block** | block only impl measured (0.830) |
| trmv | N=24 | f64 | 1.73 | — | **block** | block only impl measured (1.730) |
| trmv | N=32 | f32 | 1.21 | — | **block** | block only impl measured (1.210) |
| trmv | N=32 | f64 | 2.42 | — | **block** | block only impl measured (2.420) |
| trmv | N=48 | f32 | 1.96 | — | **block** | block only impl measured (1.960) |
| trmv | N=48 | f64 | 5.67 | — | **block** | block only impl measured (5.670) |
| trmv | N=64 | f32 | 3.51 | — | **block** | block only impl measured (3.510) |
| trmv | N=64 | f64 | 12.42 | — | **block** | block only impl measured (12.420) |
| trmv | N=96 | f32 | 15.24 | — | **block** | block only impl measured (15.240) |
| trmv | N=96 | f64 | 26.18 | — | **block** | block only impl measured (26.180) |
| trmv | N=128 | f32 | 24.88 | — | **block** | block only impl measured (24.880) |
| trmv | N=128 | f64 | 44.28 | — | **block** | block only impl measured (44.280) |
| ger | N=4 | f32 | 0.59 | — | **block** | block only impl measured (0.590) |
| ger | N=4 | f64 | 0.62 | — | **block** | block only impl measured (0.620) |
| ger | N=6 | f32 | 0.63 | — | **block** | block only impl measured (0.630) |
| ger | N=6 | f64 | 0.67 | — | **block** | block only impl measured (0.670) |
| ger | N=8 | f32 | 0.83 | — | **block** | block only impl measured (0.830) |
| ger | N=8 | f64 | 0.88 | — | **block** | block only impl measured (0.880) |
| ger | N=12 | f32 | 1.07 | — | **block** | block only impl measured (1.070) |
| ger | N=12 | f64 | 1.14 | — | **block** | block only impl measured (1.140) |
| ger | N=16 | f32 | 1.48 | — | **block** | block only impl measured (1.480) |
| ger | N=16 | f64 | 1.70 | — | **block** | block only impl measured (1.700) |
| ger | N=24 | f32 | 2.07 | — | **block** | block only impl measured (2.070) |
| ger | N=24 | f64 | 2.40 | — | **block** | block only impl measured (2.400) |
| ger | N=32 | f32 | 2.87 | — | **block** | block only impl measured (2.870) |
| ger | N=32 | f64 | 3.58 | — | **block** | block only impl measured (3.580) |
| ger | N=48 | f32 | 4.66 | — | **block** | block only impl measured (4.660) |
| ger | N=48 | f64 | 26.84 | — | **block** | block only impl measured (26.840) |
| ger | N=64 | f32 | 23.53 | — | **block** | block only impl measured (23.530) |
| ger | N=64 | f64 | 45.58 | — | **block** | block only impl measured (45.580) |
| ger | N=96 | f32 | 55.24 | — | **block** | block only impl measured (55.240) |
| ger | N=96 | f64 | 101.56 | — | **block** | block only impl measured (101.560) |
| ger | N=128 | f32 | 91.00 | — | **block** | block only impl measured (91.000) |
| ger | N=128 | f64 | 177.47 | — | **block** | block only impl measured (177.470) |

<!-- END tune.py -->

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild --legs blas2   # compile (GPU may be busy)
python3 bench/tune.py --sm auto --legs blas2              # timed run — QUIET GPU only
# or run the harness directly:
#   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 \
#        -I.. -I../src bench_blas2.cu -o bench_blas2
#   ./bench_blas2 [nprob=8192] [reps=500] [dtype=f32|f64]
```
