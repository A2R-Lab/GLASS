# Mega sweep — warp vs block(SIMT) vs nvidia(MathDx), the full ladder

Three contenders on one ns/problem axis, measured by `bench/bench_mega_sweep.cu`
(run via `run_mega_sweep.sh`). **RTX 5090, sm_120, 2026-06-23** (boost 3090 MHz,
idle GPU, no concurrent load). Metric: ns/problem, min of 3 trials.

- **WARP** — one warp per problem, `<<<ceil(NPROB/WPB), dim3(32,WPB)>>>`, WPB ∈ {1..32}
- **BLOCK** — one block per problem, `<<<NPROB, TB>>>`, TB ∈ {32,64,128,256} (pure-SIMT `glass::`)
- **NVIDIA** — cuBLASDx/cuSOLVERDx, `<<<NPROB, descriptor_threads>>>`, FORCED at every N
  (explicit `DEFINE_NVIDIA_*` specializations bypass the shipped size-heuristic so we see
  the *full* vendor curve). f32 reaches N=128; **f64 caps at N=64** (the cuBLASDx/cuSOLVERDx
  double descriptors exceed the 99 KB opt-in smem cap above 64 — gemm f64 especially).

Swept NPROB ∈ {1, 64, 1024, 8192, 32768} × N ∈ {4,6,8,12,16,24,32,48,64,96,128} × ops
{dot, gemv, gemm, chol, trsv(nvidia=trsm), posv} × {f32, f64}.
Raw data: `bench/mega_sweep_20260623_0917.txt`.

## The headline — f32 throughput ladder (NPROB=8192)

There is **no single crossover**; the ladder is *per-op*. Best ns/problem and winner:

| op | tiny (N≤8) | small (N=12–16) | mid (N=24–64) | large (N≥96) | shape |
|------|-----------|-----------------|---------------|--------------|-------|
| **dot**  | **WARP** | WARP | WARP | **WARP** | warp wins *everywhere* (2–3.7×); reduction can't use a block |
| **gemv** | **WARP** | WARP | WARP≤32 → **BLOCK**≥48 | **BLOCK** | nvidia never wins gemv (cuBLASDx gemv overhead > SIMT) |
| **gemm** | **WARP** | WARP→BLOCK @12 | **NVIDIA** 16–64 (1.6–3.3×) | **BLOCK** (nv smem-capped >64) | the full warp→block→**MathDx**→block ladder |
| **chol** | **WARP** | WARP→**NVIDIA**@16 | **NVIDIA** (1.5–2.4×) | **NVIDIA** to 128 (2.7×) | cuSOLVERDx dominates from N=16 up |
| **trsv** | **WARP** | WARP→**NVIDIA**@16 | NVIDIA 16–32 → **WARP**≥48 | **WARP** (2.9×) | nvidia wins only a mid-band; trsm degrades at large N |
| **posv** | **WARP** | WARP→**NVIDIA**@16 | **NVIDIA** (1.1–1.8×) | **NVIDIA** to 128 (1.2×) | chol+2trsv; tracks chol |

**One-paragraph rule (f32, batched/throughput):** reductions and matvec
(**dot, gemv**) stay on **warp** (gemv flips to block past N≈48); the
factor/solve/matmul ops (**gemm, chol, posv**) hand off **warp (tiny) → NVIDIA
(N≥16) → block (only when nvidia runs out of smem, N>64 for gemm / >128 for the f32
solves)**; **trsv** is the exception — nvidia wins only N=16–32, then **warp**
reclaims it because cuSOLVERDx's triangular solve scales poorly.

## Tuning knobs (f32 throughput)

- **block TB**: factor/solve (chol/trsv/posv) want **TB=32** (more threads idle on the
  serial pivot and add barrier cost); gemm scales **64→256** with N; dot/gemv **64–128**.
- **warp WPB**: **dot 8–16**, everything else **2–4** warps/block.
- **nvidia block dim** is fixed by the descriptor (we pin cuSOLVERDx at 256; cuBLASDx picks).

## f64 — narrower nvidia band (NPROB=8192)

Double shifts the ladder *toward the SIMT models*: cuSOLVERDx/cuBLASDx are relatively
less dominant in double, and the smem cap ends the nvidia leg at N=64.

| op | f64 verdict | vs f32 |
|------|-------------|--------|
| **gemm** | NVIDIA N≤32 (tiny 1.02–1.05× edge) → **BLOCK** N≥64 | nvidia band shrinks (f32 won 16–64) |
| **chol** | **BLOCK** N≤16 → NVIDIA 32–64 (1.1–1.5×) → **BLOCK** N≥96 | nvidia loses tiny N (f64 cuSOLVERDx overhead) and the >64 cap hands large N to block |
| **posv** | NVIDIA N=8–64 (1.1–1.8×) → **BLOCK** N≥96 | similar; capped at 64 |

Notable: **f64 chol N=8 → BLOCK** (nvidia 18.0 ns *loses* to block 7.5 ns) — at tiny N
double cuSOLVERDx carries real fixed overhead. The f32 equivalent had nvidia competitive
by N=16; in f64 it doesn't pull ahead until N=32.

## Latency — single problem (NPROB=1)

Launch-bound (~1.4–1.8 µs floor), so **tiny N is a wash**. For **non-tiny N the story
flips from the throughput regime**: a lone cuSOLVERDx/cuBLASDx block beats naive SIMT
massively on one big problem —

- **gemm**: NVIDIA **2.6× (N=32) → 7.6× (N=64)**; at N=128 nvidia is smem-capped → BLOCK.
- **chol**: NVIDIA **6.1× (N=32) → 7.8× (N=64) → 8.6× (N=128)**.

So if you have *one* big factor/solve/gemm and you're calling MathDx anyway, the vendor
path is the right call even at batch=1. (Below N≈16 it's launch-bound and doesn't matter;
warp ≈ block ≈ nv.)

## Peak throughput (NPROB=32768) — same ladder, sharper

The verdicts match NPROB=8192; the warp advantage on **dot widens to 5.9×** as more
problems expose the occupancy gap. No crossover moved between 8192 and 32768 — the
8192 table is the stable throughput recommendation.

## What feeds the API / next steps

- GLASS's target workload (**many small SPD factor/solves**, N≈7–30) sits right where the
  f32 ladder hands chol/posv to **nvidia (cuSOLVERDx)** — a 1.5–2.4× win the 2-way bench
  was blind to. Where MathDx isn't linked, **warp** is the throughput default (block TB=32).
- **trsv** should NOT route to cuSOLVERDx trsm above N≈32 — warp wins; worth encoding in
  any dispatch heuristic.
- **f64** users get a narrower nvidia band (N≈16–64); below/above that, SIMT (warp small,
  block large) is the call.
- The nvidia f64 leg is **smem-capped at N=64 on this GPU** (99 KB opt-in). Reaching higher
  f64 N would need a tiled/streamed descriptor — out of scope here, logged as a finding.

## Reproduce

```bash
cd bench && ./run_mega_sweep.sh sm_120     # full 3-way sweep, ~30–40 min, idle GPU
# single regime (3-way needs MathDx):
nvcc -std=c++17 -arch=sm_120 -O3 --expt-relaxed-constexpr -Xptxas -O1 -I.. -I../src \
  -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include \
  -DGLASS_BENCH_CUBLASDX -DGLASS_BENCH_CUSOLVERDX -DSMS=1200 \
  -DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT -rdc=true -dlto \
  -L$MATHDX_ROOT/lib -lcusolverdx -lcublas -lcusolver -lcudart bench_mega_sweep.cu -o bms
./bms 8192 500 f32     # throughput ; ./bms 1 3000 f32 → latency ; f64 also valid
```
