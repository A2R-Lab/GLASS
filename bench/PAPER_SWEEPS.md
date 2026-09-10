# Paper sweeps — swept GLASS configurations vs default host APIs

Harnesses feeding the GLASS paper's evaluation figures (raw captures are
archived externally with the paper materials). These are
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
summarize it as a universal "GLASS vs vendor" result.

### New-baseline legs (2026-08-16, paper "other device-side alternatives")

`--legs eigen` runs `bench_eigen_ladder.cu`: Eigen-in-kernel thread-serial
baseline vs `glass::thread::` (identical operand staging — the delta isolates
the math library) plus a `glass::block::` anchor; ops dot/gemv/gemm only and
N ≤ 32 by design (Eigen device code has no cooperative path and essentially no
device-side decompositions — that absence is the paper's point, disclosed in
the harness header). Sweeps NPROB ∈ {64, 1024, 8192} internally; every
(op, N, dtype) contender is cross-checked against a host double reference
before any timed trial. Eigen is a BENCH-ONLY dependency (`libeigen3-dev`, or
`EIGEN_ROOT`); it is never linked into the library or tests. The leg is
opt-in: it does not run under the default `--legs`.

`--legs kokkos` runs `bench_kokkos_ladder.cu`: Kokkos Kernels team-scope
baseline — kk_serial (thread-per-problem), kk_team (TeamPolicy, ts swept,
Unblocked + Blocked tags for gemm), kk_teamvector (ts × 32 lanes), vs
`glass::block::` and `glass::warp::` anchors in the same TU. Ops
gemm/gemv/trsv only (the KokkosBatched team set has no Cholesky/posv —
measured-where-comparable, prose-where-absent); N ≤ 64. Notes from
validation: `KokkosBatched::SerialGemv` is deprecated (device-aborts) and
their Team/TeamVector gemv takes a rank-3 multi-problem-per-team view, so
the gemv paths use the single-matrix `KokkosBlas::{Serial,Team,TeamVector}Gemv`
forms (their own batch-1 delegation target); `TeamVectorGemm<Blocked>` is
declared but unimplemented upstream. Kokkos + kokkos-kernels are BENCH-ONLY
installs (`KOKKOS_ROOT`/`KOKKOSKERNELS_ROOT`, default `~/opt`). Opt-in leg.

`--legs hostblas_magma` runs the hostblas harness with `-DGLASS_BENCH_MAGMA`
columns: `magma` gemm (`gemm_batched_strided`, exact mirror of the cuBLAS
strided call), `magma_pa` gemm (pointer-array `gemm_batched`, which dispatches
to MAGMA's tuned small-square kernels), `magma` potrf (`potrf_batched`) and
`magma` posv (MAGMA's fused one-call `posv_batched` — favorable to MAGMA vs
the vendor's potrf+potrs pair; disclosed). The MAGMA queue is created on the
default stream so the shared event bracketing times it identically. MAGMA
(master, Blackwell/sm_120-native) + a static no-Fortran OpenBLAS (C LAPACK)
are BENCH-ONLY deps (`MAGMA_ROOT`/`OPENBLAS_ROOT`, defaults
`~/opt/src/magma-git` and `~/opt/openblas`). Same source file as `hostblas`;
binaries are named by LEG so the two never collide. Disclosure: MAGMA is a
research library measured in addition to our documented default-vendor scope.
Opt-in leg.

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
rendered from them until the 2026-08-15 recapture below). (Original smoke
validation 2026-07-06 on a shared box; those numbers were discarded.)

The Jetson legs ran 2026-08-17 on BOTH embedded boards, closing the runbook:
Orin AGX (sm_87, CUDA 13.2, MODE_50W + pinned clocks) captures
`paper_hostblas_20260817_171835.txt` / `paper_fusion_20260817_173005.txt`,
and Xavier AGX (sm_72, CUDA 11.4/JetPack 5, MODE_30W_ALL + pinned clocks,
explicit `--arch sm_72` — auto-detect has no nvidia-smi there) captures
`paper_hostblas_20260817_172417.txt` / `paper_fusion_20260817_174837.txt`.
All four: full row counts, zero FAIL/INVALID — the same source builds and
passes its host-double cross-checks from CUDA 11.4 through 13.2 without a
single source change, which doubles as the long-pending Xavier run-proof.
Spreads are wider than desktop (Orin max 22%, Xavier max 31%); only >=1.2x
deltas are publishable from these captures and near-parity cells are ties.
An Orin tegrastats energy companion (100 ms, GPU/CPU rails) was captured
against a separate fusion repeat; Xavier's power rails are root-locked on
JetPack 5, so its energy pass awaits a sudo run. Archived externally with
the paper materials (shas recorded).

The audited 2026-08-14 rerun validated the refreshed harnesses (fusion capture
`paper_fusion_20260813_231910.txt`; 500-rep host confirmation
`paper_hostblas_20260814_111514.txt`). Those two validation captures were
compiled with a then-global `-Xptxas -O1` on the GLASS side (since reverted
to plain `-O3` for these harnesses); the vendor side is prebuilt cuBLAS and
unaffected, so GLASS-win margins in them are conservative lower bounds. The
shipped F2/F3/F4 figures still render from the 2026-07-08 plain-`-O3`
captures. Fusion was stable (1 of 240 contender
samples above 5% spread, with a stable winning contender). The host confirmation reduced noisy
selected pairs from 24/378 to 8/378; all eight retained decisive 1.56×–2.19×
gaps. The latency section had 0/117 samples above 5% spread. One near-tie
(`potrf` f64, N=8, B=4) changed winner between the independent 100- and
500-rep captures despite low within-run spread. Publication policy is therefore
conservative: small deltas and cross-run winner changes are ties; only claims
that survive both quiet captures are reportable. These characterization cells
do not drive library dispatch.

The 2026-08-15 quiet window re-ran both legs at plain `-O3` on the GLASS side
(post-revert flags): `paper_hostblas_20260815_135938.txt` /
`paper_fusion_20260815_140027.txt`. These are the captures the paper's F2/F3/F4
figures and prose numbers regenerate from (superseding the 2026-07-08 pair);
the -O1-era 08-13/08-14 captures above remain archived as conservative
validation runs only.

Baseline-leg captures (2026-08-16/17, all archived externally): eigen
`paper_eigen_20260816_171111.txt` (378/378 checks; parity vs glass::thread,
block anchor 2--12x at N>=8 — the execution-model finding); magma v2
`paper_hostblas_magma_20260817_021730.txt` (279/279 checks; thread-tier rows
+ MAGMA latency rows — supersedes 005316, which compared factorizations
against the wrong GLASS tier; posv small-N is GLASS 1.5--5x at every batch,
potrf N>=8 is MAGMA 1.2--3.2x, MAGMA's measured latency floor ~6.5--11us);
kokkos v2 `paper_kokkos_20260817_011717.txt` (972/972 checks, glass_thread
anchor; supersedes 002516 — same wrong-tier flaw on trsv). RULE adopted
after that flaw bit twice: an external-baseline harness MUST include every
GLASS tier the shipped dispatch routes to in the measured domain. Both v2
captures are jitterier than the pinned ones (ambient load); overlapping rows
agree with their independent v1 twins (median ratios 0.995/1.001), paper
claims use >=1.2x deltas, and near-parity cells are reported as ties.
Quiet confirmation captures (2026-08-17, idle box) reproduce both v2
captures and close the single-capture caveat:
`paper_hostblas_magma_20260817_110407.txt` (reps=100, 279/279 checks,
median ratio 1.000 over all 2556 common rows; thread and MAGMA rows
0.992--1.000 with zero >20% drifts outside four MAGMA gemm-latency cells)
and `paper_kokkos_20260817_110746.txt` (reps=500, 972/972 checks, only
19/3672 rows above 5% spread — the cleanest kokkos capture; glass_thread
rows median 0.997, zero >20% drifts). Paper numbers are the conservative
envelope of the v2 + confirmation pairs.

Same-evening companions (all archived externally alongside the pair above):
a 500-rep hostblas confirmation `paper_hostblas_20260815_173550.txt` (agrees
with the 135938 primary within jitter — including the single-call latency
lane), the robotics sweep `perf_robotics_paper_20260815_140911.txt` (drives
the paper's robotics panel and tables; correctness-receipt fingerprint in its
header), and the pinned ladder capture `mega_sweep_20260815_205919.txt`
(drift-fixed v2 harness; source of the 2026-08-15 dispatch-table re-gate and
the paper's ladder/heatmap figures — see `bench/RESULTS.md`).

**Numerics finding (valid despite shared load — maxerr is deterministic):**
the `vendor_tf32` CHECK column doubles as a tensor-core ENGAGEMENT detector.
On CUDA 13.2 / sm_120, with TF32 allowed, cuBLAS still runs FP32-FFMA kernels
for N ≤ 16 (maxerr ~1e-7, bit-matching the plain vendor row) and engages TF32
only from N = 24 up, where maxerr jumps ~1000× to ~2e-4 (unit-scale data).
Caveat: the error probe runs at B=4; heuristics could differ at other batch
sizes — the timed sweep detects that case as a vendor_tf32-vs-vendor speed
divergence at fixed N.
