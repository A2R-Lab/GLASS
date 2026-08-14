# Paper sweeps — swept GLASS configurations vs default host APIs

Harnesses feeding the GLASS paper's evaluation figures (the paper lives in the
sibling `glass-paper` repository, which also archives all raw captures). These are
**characterization only** — nothing here regenerates library dispatch tables
(that is `bench/tune.py`'s job).

| Leg | Harness | Paper figure | What it measures |
|-----|---------|--------------|------------------|
| hostblas | `bench_paper_hostblas.cu` | F2 (throughput vs batch), F4 (batch=1 latency) | Best of the explicitly swept GLASS block32/block128/warp8 configurations vs one documented default cuBLAS/cuSOLVER host-API configuration, for {gemm, potrf, posv(nrhs=1)}, N {4…64} × B {1…8192} × {f32,f64}; plus a `vendor_tf32` gemm-f32 contender with relaxed numerics |
| fusion | `bench_paper_fusion.cu` | F3 (fusion speedup vs batch) | fused `glass::riccati_gain` kernel (intermediates in smem) vs the same math as 7 host-batched cuBLAS/cuSOLVER calls, (NX,NU) {(12,4),(14,7),(36,12),(48,16)} × B {1…4096} |

The nvidia-interface (cuBLASDx/cuSOLVERDx) curves for F1 come from the
existing mega-sweep leg (`bench/tune.py --legs ladder`, results in
`RESULTS.md`) — the paper harnesses deliberately need **no MathDx**
so they build anywhere (Jetson included) with just the CUDA toolkit.

## Running

```bash
python3 bench/paper_sweeps.py --build-only    # prep — safe while the GPU is busy
python3 bench/paper_sweeps.py                 # timed run — QUIET GPU ONLY
python3 bench/paper_sweeps.py --reps 100      # longer release confirmation
```

The driver auto-detects the arch (`nvidia-smi compute_cap`), links
`-lcublas -lcusolver`, refuses to start on a busy GPU (`--force` overrides),
and invalidates a run if another compute PID appears. Legs run serially and
write `paper_<leg>_<timestamp>.txt`. Each capture begins with the source
revision and digest, dirty state, device target, toolchain, UTC date, and the
date/fingerprint of the proximate signed correctness receipt.
Provenance schema 2 records the receipt artifact hash and source fingerprint
as separate fields. Older schema-1 captures mislabeled the fingerprint digest
as `correctness_receipt_sha256`; the value remains useful as a source identity,
but it is not a hash of the JSON artifact.

The harness warms the GPU to steady boost before the first cell and randomizes
contender order within each cell. Each rep is one event-bracketed launch or API
chain spanning all B problems; mutated state is restored outside the event
window. Reports use the minimum of three trials and include worst/best spread.
The latency section repeats its 190 synchronized batch=1 calls in three trials
on pristine data. GPU-event throughput excludes host API overhead, which is
conservative toward the vendor. Cells skipped for memory or shared-memory
limits print `SKIP` lines.

Correctness remains a separate release gate through the signed GPU receipt. The
paper harness also retains a cheap host-double guard to catch a malformed
benchmark configuration before it consumes a timing window; its output is not
counted as project correctness coverage.

This comparison is intentionally narrow: it selects the best of several GLASS
launch configurations against a single default vendor API configuration. Do not
summarize it as a universal “GLASS vs vendor” result.

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

The audited 2026-08-14 rerun validated the refreshed harnesses (fusion capture
`paper_fusion_20260813_231910.txt`; 500-rep host confirmation
`paper_hostblas_20260814_111514.txt`). Fusion was stable (1 of 240 contender
samples above 5% spread, with a stable winning contender). The host confirmation reduced noisy
selected pairs from 24/378 to 8/378; all eight retained decisive 1.56×–2.19×
gaps. The latency section had 0/117 samples above 5% spread. One near-tie
(`potrf` f64, N=8, B=4) changed winner between the independent 100- and
500-rep captures despite low within-run spread. Publication policy is therefore
conservative: small deltas and cross-run winner changes are ties; only claims
that survive both quiet captures are reportable. These characterization cells
do not drive library dispatch.

**Numerics finding (valid despite shared load — maxerr is deterministic):**
the `vendor_tf32` CHECK column doubles as a tensor-core ENGAGEMENT detector.
On CUDA 13.2 / sm_120, with TF32 allowed, cuBLAS still runs FP32-FFMA kernels
for N ≤ 16 (maxerr ~1e-7, bit-matching the plain vendor row) and engages TF32
only from N = 24 up, where maxerr jumps ~1000× to ~2e-4 (unit-scale data).
Caveat: the error probe runs at B=4; heuristics could differ at other batch
sizes — the timed sweep detects that case as a vendor_tf32-vs-vendor speed
divergence at fixed N.
