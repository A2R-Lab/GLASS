# rect sweep — warp/block picks for rectangular gemv/gemm

Source: `bench/bench_rect.cu`, driven by `bench/tune.py --legs rect`. The
mega-sweep ladder is square-only, but consumers' Jacobians are rectangular;
this leg measures:

- **gemv** `(M, N)`: tall {64×8, 128×16, 256×32} + wide {8×64, 16×128, 32×256}
  — `y(M) = A(M×N)·x(N)`.
- **gemm** `(M, K, N)`: {(32,8,32), (8,32,8), (64,16,16), (16,64,16), (6,6,64),
  (64,6,6)} — `C(M×N) = A(M×K)·B(K×N)` (glass template order is
  `gemm<T,M,N,K>`).

f32 + f64, one problem per block (BLOCK, TB ∈ {32, 64, 128, 256}) vs one
problem per warp (WARP, WPB ∈ {1..32}). **The nvidia leg is skipped** for
rectangular shapes: forcing cuBLASDx here would need new per-(M,N,K)/(M,N)
`DEFINE_NVIDIA_*` descriptor instantiations, and per-shape cuBLASDx-vs-SIMT
decisions already live in the `shapes` leg (`bench/autotune.py` →
`src/nvidia/tuning_table.cuh`) — rectangular vendor coverage belongs there.

These measurements do **not** regenerate a shipped header table yet (the
square-N `ideal_sm120` ladder stays authoritative for dispatch); the block
between the markers below is auto-refreshed by `bench/tune.py` through the
shared `tune_pick` margin rule.

<!-- BEGIN tune.py: latest measured run -->
## Latest measured run (auto-refreshed by `bench/tune.py`)

_Source: `rect_sweep_20260718_0328.txt` · NPROB=8192 ns/problem · margin ±5% (warp/block are both dependency-free; pick = cheapest, note flags sub-margin gaps) · warp picked in 16 of 24 cells._

nvidia leg skipped for rectangular shapes (needs new per-shape DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in the `shapes` leg).

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
| gemv | 128x16 | f64 | 10.48 | 10.57 | **block** | block wins (10.480 vs warp 10.570, 0.9%) |
| gemv | 256x32 | f32 | 20.36 | 20.42 | **block** | block wins (20.360 vs warp 20.420, 0.3%) |
| gemv | 256x32 | f64 | 40.31 | 40.78 | **block** | block wins (40.310 vs warp 40.780, 1.2%) |
| gemm | 6x6x64 | f32 | 0.88 | 0.92 | **block** | block wins (0.880 vs warp 0.920, 4.5%) |
| gemm | 6x6x64 | f64 | 2.74 | 3.02 | **block** | block wins (2.740 vs warp 3.020, 10.2%) |
| gemm | 8x32x8 | f32 | 0.92 | 0.88 | **warp** | warp wins (0.880 vs block 0.920, 4.5%) |
| gemm | 8x32x8 | f64 | 2.42 | 0.04 | **warp** | warp wins (0.040 vs block 2.420, 5950.0%) |
| gemm | 16x64x16 | f32 | 3.75 | 3.94 | **block** | block wins (3.750 vs warp 3.940, 5.1%) |
| gemm | 16x64x16 | f64 | 18.50 | 19.15 | **block** | block wins (18.500 vs warp 19.150, 3.5%) |
| gemm | 32x8x32 | f32 | 1.71 | 1.53 | **warp** | warp wins (1.530 vs block 1.710, 11.8%) |
| gemm | 32x8x32 | f64 | 9.63 | 0.04 | **warp** | warp wins (0.040 vs block 9.630, 23975.0%) |
| gemm | 64x6x6 | f32 | 0.77 | 0.72 | **warp** | warp wins (0.720 vs block 0.770, 6.9%) |
| gemm | 64x6x6 | f64 | 3.59 | 0.04 | **warp** | warp wins (0.040 vs block 3.590, 8875.0%) |
| gemm | 64x16x16 | f32 | 2.49 | 0.04 | **warp** | warp wins (0.040 vs block 2.490, 6125.0%) |
| gemm | 64x16x16 | f64 | 18.24 | 0.04 | **warp** | warp wins (0.040 vs block 18.240, 45500.0%) |

<!-- END tune.py -->

## Reproduce

```bash
python3 bench/tune.py --sm auto --prebuild --legs rect    # compile (GPU may be busy)
python3 bench/tune.py --sm auto --legs rect               # timed run — QUIET GPU only
# or run the harness directly:
#   nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 \
#        -I.. -I../src bench_rect.cu -o bench_rect
#   ./bench_rect [nprob=8192] [reps=500] [dtype=f32|f64]
```
