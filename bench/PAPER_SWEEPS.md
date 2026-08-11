# Paper sweeps — GLASS vs host-batched vendor + fusion case study

Harnesses feeding the GLASS paper's evaluation figures (the paper lives in the
sibling `glass-paper` repository, which also archives all raw captures). These are
**characterization only** — nothing here regenerates library dispatch tables
(that is `bench/tune.py`'s job).

| Leg | Harness | Paper figure | What it measures |
|-----|---------|--------------|------------------|
| hostblas | `bench_paper_hostblas.cu` | F2 (throughput vs batch), F4 (batch=1 latency) | glass block/warp vs cuBLAS `gemmStridedBatched` / cuSOLVER `potrfBatched(+potrsBatched)`, ops {gemm, potrf, posv(nrhs=1)}, N {4…64} × B {1…8192} × {f32,f64}; plus a `vendor_tf32` gemm-f32 contender (handle with `CUBLAS_TF32_TENSOR_OP_MATH` — tensor cores ALLOWED at relaxed numerics) whose report-only CHECK line records the measured TF32 rounding cost |
| fusion | `bench_paper_fusion.cu` | F3 (fusion speedup vs batch) | fused `glass::riccati_gain` kernel (intermediates in smem) vs the same math as 7 host-batched cuBLAS/cuSOLVER calls, (NX,NU) {(12,4),(14,7),(36,12),(48,16)} × B {1…4096} |

The nvidia-interface (cuBLASDx/cuSOLVERDx) curves for F1 come from the
existing mega-sweep leg (`bench/tune.py --legs ladder`, results in
`RESULTS.md`) — the paper harnesses deliberately need **no MathDx**
so they build anywhere (Jetson included) with just the CUDA toolkit.

## Running

```bash
python3 bench/paper_sweeps.py --build-only    # prep — safe while the GPU is busy
python3 bench/paper_sweeps.py                 # timed run — QUIET GPU ONLY
```

The driver auto-detects the arch (`nvidia-smi compute_cap`), links
`-lcublas -lcusolver`, refuses to time on a busy GPU (`--force` overrides),
runs legs serially, and writes `paper_<leg>_<timestamp>.txt` (gitignored, like
the other raw sweep captures).

Methodology matches `bench_solvers.cu`: correctness-guarded against a host
double reference before any timing (abort on mismatch); each rep is one
event-bracketed launch/API-chain spanning all B problems, mutated state
restored outside the event window; ns/problem = min of 3 trials. The latency
section wall-clocks batch=1 calls (sync per call, 190 timed calls on pristine
per-call data) against the NON-batched vendor calls a user would actually
write. GPU-event timing excludes host API overhead — conservative toward the
vendor. No silent caps: cells skipped for memory or smem limits print `SKIP`
lines.

## Jetson / Orin runbook (when the box lands)

1. Clone + `python3 bench/paper_sweeps.py --build-only` — arch auto-detects
   (e.g. sm_87), no MathDx needed.
2. Quiet box (no desktop compositor spikes; `sudo jetson_clocks` to pin
   clocks), then `python3 bench/paper_sweeps.py`.
3. Ladder retune for the portability section: `python3 bench/tune.py --sm auto
   --prebuild` first, then the timed legs — see `TUNING.md`. Diff the
   regenerated ladder vs sm_120's for the "what moves" discussion.
4. Optional energy figure: run `tegrastats --interval 100` alongside a repeat
   of the fusion leg only (it's short); joules/solve = mean power × time.
   Do NOT record ns/problem numbers from that interleaved run — power capture
   and timing capture are separate passes.

## Results

sm_120 timing landed in the 2026-07-08 quiet window (captures
`paper_hostblas_20260708_0054.txt` / `paper_fusion_20260708_0055.txt`; F2/F3/F4
in the paper render from them). Jetson/Orin legs of THESE harnesses
(hostblas/fusion at sm_87, optional tegrastats energy) remain un-run — the
Orin runbook below is still pending for that leg only. (Original smoke
validation 2026-07-06 on a shared box; those numbers were discarded.)

**Numerics finding (valid despite shared load — maxerr is deterministic):**
the `vendor_tf32` CHECK column doubles as a tensor-core ENGAGEMENT detector.
On CUDA 13.2 / sm_120, with TF32 allowed, cuBLAS still runs FP32-FFMA kernels
for N ≤ 16 (maxerr ~1e-7, bit-matching the plain vendor row) and engages TF32
only from N = 24 up, where maxerr jumps ~1000× to ~2e-4 (unit-scale data).
Caveat: the error probe runs at B=4; heuristics could differ at other batch
sizes — the timed sweep detects that case as a vendor_tf32-vs-vendor speed
divergence at fixed N.
