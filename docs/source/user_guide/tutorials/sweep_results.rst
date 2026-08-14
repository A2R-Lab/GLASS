Backend Sweep Results
=====================

GLASS ships overlapping thread-, warp-, block-, and vendor-backed execution
tiers. For operations present in more than one tier, the fastest choice depends
on operation, size, dtype, launch shape, and batch regime. The **mega sweep**
(``bench/tune.py``'s ladder leg) compares supported contenders — this is
exactly the measurement behind ``glass-defaults.cuh``'s ``suggested_backend<>()``
(see :doc:`../../api_reference/defaults`).

.. warning::

   This page is a dated measurement archive, not a universal performance
   promise. Several legacy sections predate the current warm-up, trial-spread,
   source-digest, and signed-receipt protocol and are marked accordingly. Use
   the source capture named beside a claim, and rerun on the target machine.

The figures and table below are from an RTX 5090 / sm_120 run, shown across three
batch regimes — **NPROB=64** (low batch, latency-leaning), **NPROB=1024** (mid),
and **NPROB=8192** (the throughput regime that feeds the dispatch tables). The
winner can shift with batch size: at low batch the vendor (``nvidia``) path often
wins the factor/solve ops on launch-amortized latency, while at high batch the
hand-rolled SIMT paths scale back in. They are committed static assets —
regenerate them for your own hardware with::

   python bench/tune.py --sm auto      # remeasures + regenerates tables AND figures
   # or just the figures from an existing sweep .txt:
   python bench/export_sweep_figures.py bench/mega_sweep_*.txt

``bench/explore_sweep.ipynb`` is the interactive version of the same analysis.

The ladder — ns/problem vs N, per backend
------------------------------------------

Lower is faster. Each subplot is one op; the curves are ``warp`` (green),
``block`` (blue), ``thread`` (orange, N≤16 — one problem per thread, 32 packed
per warp), and ``nvidia`` / MathDx (red). The crossover points are where
``suggested_backend`` switches tiers — the 2026-07-18 sweep hands thread the
low-DOF corner of every op except ``gemm`` (up to 7.5× on ``posv`` f64 at N≤6;
verdict tables in the thread-tier section below). Where a ``thread`` curve
stops short of N=128 the remaining launches are *infeasible*, not unmeasured —
the per-thread local-memory footprint exceeds the launch limit (those cells are
``FAIL``-marked in the capture); the ``nvidia`` f64 curves cap at N=64 for the
same reason on the shared-memory side. ``suggested_backend<>()`` is keyed on the
**NPROB=8192** throughput regime; the 64/1024 figures show how the crossovers
move at smaller batch.

float32
~~~~~~~

.. image:: /_static/mega_sweep_ladder_f32_n64.png
   :alt: f32 ladder, NPROB=64
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f32_n1024.png
   :alt: f32 ladder, NPROB=1024
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f32_n8192.png
   :alt: f32 ladder, NPROB=8192 (throughput — feeds suggested_backend)
   :width: 100%

float64
~~~~~~~

.. image:: /_static/mega_sweep_ladder_f64_n64.png
   :alt: f64 ladder, NPROB=64
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f64_n1024.png
   :alt: f64 ladder, NPROB=1024
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f64_n8192.png
   :alt: f64 ladder, NPROB=8192 (throughput — feeds suggested_backend)
   :width: 100%

Winner per (op, N), per regime
------------------------------

The backend with the lowest ns/problem at each ``(op, N)``, listed for all three
NPROB regimes — the ``NPROB=8192`` block is what ``suggested_backend<>()``
encodes. The broad shape at high batch: tiny ``N`` favors ``warp``; mid sizes
favor ``nvidia`` for the factor/solve ops (chol/posv/trsv) once MathDx amortizes;
``gemv`` crosses to ``block`` early; ``dot`` stays ``warp`` throughout. At
``NPROB=64`` the ``nvidia`` band widens (launch latency dominates, so the vendor
kernels win sooner).

.. literalinclude:: /_static/sweep_winners.txt
   :language: text

vs. host-batched cuBLAS/cuSOLVER (and TF32)
-------------------------------------------

The ladder above compares *device-side* backends. A separate question is how
one-block-per-problem GLASS compares to the standard host-side recipe — a
single ``cublas<t>gemmStridedBatched`` / ``cusolverDn<t>potrfBatched``
(+ ``potrsBatched``) call over the whole batch.
``bench/bench_paper_hostblas.cu`` measures exactly that: gemm / potrf / posv,
``N`` = 4–64, batch ``B`` = 1–8192, both precisions (raw capture committed as
``paper_hostblas_20260708_0054.txt`` (archived in the glass-paper repo); RTX 5090 / sm_120, quiet GPU).

.. image:: /_static/hostblas_speedup.png
   :alt: host-batched vendor time divided by best GLASS time, vs batch size
   :width: 100%

Above 1.0 = the best swept GLASS configuration faster (fp32 shown; GLASS is
best of block32, block128, and warp8 where available). The vendor side is one
default host-API configuration, so this is not a symmetric tuner comparison or
a universal “GLASS vs vendor” statement. In this historical capture, at robot
sizes, host batching never catches up: gemm at ``N`` ≤ 24 and the full
factor-and-solve (posv) through ``N`` = 64 are GLASS wins at **every** batch
size, reaching 2.9–6.3× at saturation. The vendor's best regime is mid-batch
(``B`` ≈ 64–1024), where it briefly leads standalone potrf at mid sizes; only
gemm at ``N`` ≥ 32 is an outright vendor win at scale — the same mid-band the
ladder already routes to ``glass::nvidia::``.

An audited 2026-08-14 confirmation used 500 reps. Only 8/378 selected
GLASS/vendor pairs had a contender above 5% within-run spread, and all eight
had decisive 1.56×–2.19× gaps. One small f64 ``potrf`` comparison changed
winner between two quiet captures despite low within-run spread. Accordingly,
small deltas and cross-run winner changes are treated as ties; publication
claims must survive both independent captures.

Permitting TF32 tensor cores (dashed) does not change the story: cuBLAS
*declines to engage them* below ``N`` = 24 (results bit-identical to FP32),
and where they do engage the speed is a wash against FP32 cuBLAS while max
error jumps three orders of magnitude (~1e-7 → ~2e-4) — unusable for the
Cholesky-chain ops, which have no TF32 cuSOLVER path at all.

Fusion: ``riccati_gain`` vs. a 7-call vendor chain
--------------------------------------------------

``glass::riccati_gain`` computes the LQR feedback gain
``K = (R + BᵀPB)⁻¹(BᵀPA)`` in one kernel with all intermediates in shared
memory; the host-batched equivalent is seven vendor calls (four gemms, a
batched Cholesky, two triangular solves) with intermediates in global memory.
``bench/bench_paper_fusion.cu`` compares them (capture
``paper_fusion_20260708_0055.txt`` (archived in the glass-paper repo)):

.. image:: /_static/fusion_speedup.png
   :alt: fused riccati_gain vs 7-call host-batched vendor chain
   :width: 75%
   :align: center

In this historical implementation/capture, fusion wins at every batch size at
quadrotor/manipulator scale — 2.5–2.8× at
``(nx,nu)`` = (12,4) and 1.6–1.9× at (14,7) in fp32, more in fp64 — but the
chain wins at (36,12) fp32 and (48,16), where the staged operands outgrow what
one block overlaps profitably. Fusion is a measured choice, not a default:
GLASS composes both forms from the same primitives.

Single-call latency
-------------------

For a batch of **one** (a high-rate MPC tick), wall-clock per-call latency is
what matters. The non-batched vendor calls pin an essentially flat API floor —
~7–10 µs (``cublasSgemm``), ~15.5 µs (potrf), ~23–30 µs (posv) — while a GLASS
call starts at 5.2 µs and grows with compute, so GLASS wins single-call
latency through ``N`` = 32 (gemm), 12 (potrf), and 24 (posv); at ``N`` = 8 the
full factor-and-solve is 2.4× faster (9.6 vs 23.5 µs). And inside your own
kernel, composed GLASS calls never pay the API floor again.

Both harnesses live in ``bench/`` and rerun via
``python3 bench/paper_sweeps.py`` (see ``bench/PAPER_SWEEPS.md``).

Choosing among the dense solve paths (measured guidance)
---------------------------------------------------------

GLASS deliberately ships **no** auto-dispatch between its linear-system
solvers — the right choice depends on structure and conditioning, which a
compile-time table cannot see. Instead, ``bench/tune.py --legs solvers``
measures the trade-offs on your GPU and records them in
``bench/RESULTS.md`` (solvers section). The RTX 5090 numbers (NPROB=8192,
ns/problem):

**SPD single solve — use ``posv``; the alternatives price as follows.**
``gesv`` (pivoted LU, the robustness fallback) costs 1.0–2.1× ``posv`` in
fp32 at ``N`` ≥ 16 (1.8–2.1× at 32–64); the invert-then-multiply
anti-pattern (``inv`` + ``gemv``) costs 1.9–4.7× at ``N`` ≥ 16. **Below**
``N`` = 16 all three paths are within a few nanoseconds of each other (and
``inv``/``gesv`` can even edge out ``posv``, especially in fp64) — there,
choose by *numerics*, not speed: Cholesky is backward-stable on SPD input and
fails loudly (with ``CHECK``) on indefinite input, while an explicit inverse
amplifies conditioning error silently. Speed only ever argues *for* ``posv``,
never against it.

**Block-tridiagonal chains — ``bdsv`` (direct) vs ``pcg`` (iterative) is
problem-dependent; do not hard-code either.** On the benchmark's synthetic
diagonally-dominant SPD system, PCG converges in about three iterations and
wins 11 of 12 cells (up to 9×) at the harness's ``rho = rᵀz`` relative
tolerance of 1e-6. This is an approximate-solve comparison, not matched final
residual accuracy: the direct f64 solves reach near-machine precision while
PCG stops at its requested tolerance. At
(BlockSize=12, Knots=16) fp32 the direct sweep wins 1.8×. PCG's cost scales
linearly with its iteration count, so an ill-conditioned Riccati/KKT chain
(10–100× more iterations) moves the crossover proportionally toward
``bdsv`` — read the ``pcg iters`` column of your own sweep before
generalizing, or measure with your actual matrices.

**``syev`` / ``eig_clamp``** — the historical harness compared only this serial
Jacobi family. It did not include the parallel fixed-sweep ``eigh`` /
``psd_project`` family, so it cannot support a recommendation between them.

The thread tier — where one-problem-per-thread wins
---------------------------------------------------

The 2026-07-19 full-domain sweep (quiet RTX 5090) added the ``thread``
contender at every ``(op, N)`` point. Throughput regime (NPROB=8192), thread
vs the best other tier (ratio > 1 = thread faster; **bold** cells shipped):

.. list-table::
   :header-rows: 1

   * - op / dtype
     - N=4
     - N=6
     - N=8
     - N=12
     - N=16
     - N=24
     - shipped band
   * - posv f64
     - **6.17×**
     - **7.46×**
     - **5.03×**
     - **3.05×**
     - **2.26×**
     - **1.19×**
     - thread ≤ 24
   * - chol f64
     - **5.61×**
     - **5.79×**
     - **2.88×**
     - **2.18×**
     - **1.71×**
     - **1.04×**
     - thread ≤ 24
   * - trsv f32/f64
     - **1.5/4.6×**
     - **2.2/4.6×**
     - **2.0/2.0×**
     - **2.3/1.5×**
     - **1.7/1.2×**
     - 0.2×
     - thread ≤ 16
   * - dot f64
     - **2.32×**
     - **2.33×**
     - **2.31×**
     - **1.96×**
     - **1.53×**
     - **1.30×**
     - thread ≤ 32
   * - posv f32
     - **3.21×**
     - **3.44×**
     - **1.73×**
     - **1.16×**
     - 0.66×
     - 0.35×
     - thread ≤ 12
   * - gemm (both)
     - 0.6–0.9×
     - 0.4–0.5×
     - —
     - —
     - —
     - —
     - never

The factor/solve chain is where the tier earns its keep — a warp-per-problem
``potrf`` at N ≤ 7 idles most lanes on the serial pivot; one-problem-per-thread
keeps all 32 busy. f64 amplifies and extends the win past the register ceiling
(even the spilled thread path beats the alternatives through N=24). ``gemm``
is the anti-case: enough work per element that the parallel tiers always win.
Columns past the ceiling price the local-memory spill honestly (down to 0.1× —
measured, never shipped).

Warp-scope vendor A/B (audited characterization)
-------------------------------------------------

``bench_nvwarp_l1.cu`` A/Bs ``glass::warp::`` against ``glass::nvidia::warp::``
(CUB WarpReduce) for the three ops the vendor warp tier ships (reduce/dot/nrm2)
at identical launch shape. The 2026-08-14 sm_120 quiet run used the shared
warm-up, spread, provenance, and isolation protocol. Of 126 cells, 121 tie
within ±2%. Four high-throughput f64 ``dot`` cells favor CUB by 2.0–3.0%; the
sole SIMT verdict is a tiny latency cell with 17.45% SIMT spread and is not
actionable. Five individual contender samples exceeded 5% spread, but no
shipped dispatch decision depends on this characterization.

Robotics micro-ops — audited characterization
----------------------------------------------

The 2026-08-14 sm_120 quiet run measured 8192 problems and 1000 reps per trial
through the shared audited protocol:

- The thread tier wins every fixed-size row for both dtypes.
- ``softmax_n16`` is dtype-sensitive: warp wins f32, thread wins f64.
- At the best thread launch, fused and composed spatial forms are nearly tied
  in f32; the fused f64 forms are about 15–22% faster.
- Only 2 of 136 contender samples exceed 5% spread, neither changing a clear
  row verdict. ``argmax_fast`` is block-only and does not beat the ordinary
  thread-tier ``argmax``.

These are characterization results, not generated dispatch inputs.

The full measured-results archive
----------------------------------

The machine-refreshed verdict tables for every sweep live in one file,
``bench/RESULTS.md``. The ladder, body, blas2, rect, solvers, and reduced marker
blocks are rewritten by ``bench/tune.py``; nvwarp and robotics are hand-run
audited characterizations and do not rewrite dispatch.
Raw captures are archived in the ``glass-paper`` repository (``data/desktop/``,
``data/jetson/``, ``data/sm120/``); the paper harnesses are documented in
``bench/PAPER_SWEEPS.md``.

See :doc:`../concepts/tuning` for how to emit a per-host override table from a
sweep, and :doc:`../../api_reference/defaults` for the picker API.
