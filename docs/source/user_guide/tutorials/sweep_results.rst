Backend Sweep Results
=====================

GLASS ships four interchangeable execution tiers — thread-, warp-, and
block-scoped SIMT plus the vendor-backed ``nvidia`` path — and which one is
fastest depends on the operation, the matrix size ``N``, and the dtype. The **mega sweep** (``bench/tune.py``'s ladder leg) times all
of them head-to-head so the choice is data-driven rather than guessed — this is
exactly the measurement behind ``glass-defaults.cuh``'s ``suggested_backend<>()``
(see :doc:`../../api_reference/defaults`).

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
verdict tables in ``bench/THREAD_SWEEP_RESULTS.md``). Where a ``thread`` curve
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
``bench/paper_hostblas_20260708_0054.txt``; RTX 5090 / sm_120, quiet GPU).

.. image:: /_static/hostblas_speedup.png
   :alt: host-batched vendor time divided by best GLASS time, vs batch size
   :width: 100%

Above 1.0 = GLASS faster (fp32 shown; GLASS = best of block/warp). At robot
sizes, host batching never catches up: gemm at ``N`` ≤ 24 and the full
factor-and-solve (posv) through ``N`` = 64 are GLASS wins at **every** batch
size, reaching 2.9–6.3× at saturation. The vendor's best regime is mid-batch
(``B`` ≈ 64–1024), where it briefly leads standalone potrf at mid sizes; only
gemm at ``N`` ≥ 32 is an outright vendor win at scale — the same mid-band the
ladder already routes to ``glass::nvidia::``.

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
``bench/paper_fusion_20260708_0055.txt``):

.. image:: /_static/fusion_speedup.png
   :alt: fused riccati_gain vs 7-call host-batched vendor chain
   :width: 75%
   :align: center

Fusion wins at every batch size at quadrotor/manipulator scale — 2.5–2.8× at
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
``bench/SOLVERS_SWEEP_RESULTS.md``. The RTX 5090 numbers (NPROB=8192,
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
problem-dependent; do not hard-code either.** On our diagonally-dominant test
system PCG converges in ~3 iterations and wins 11 of 12 cells (up to 9×); at
(BlockSize=12, Knots=16) fp32 the direct sweep wins 1.8×. PCG's cost scales
linearly with its iteration count, so an ill-conditioned Riccati/KKT chain
(10–100× more iterations) moves the crossover proportionally toward
``bdsv`` — read the ``pcg iters`` column of your own sweep before
generalizing, or measure with your actual matrices.

**``syev`` / ``eig_clamp``** — the decompose–clamp–reconstruct op costs the
same as the bare eigensolve (the clamp epilogue is free); budget ~0.9 µs
fp32 / ~8.8 µs fp64 per 32×32 problem at saturation.

The full measured-results archive
----------------------------------

Every measured family keeps its spliced verdict tables in a results file under
``bench/`` in the repo:

- ``MEGA_SWEEP_RESULTS.md`` — the thread/warp/block/nvidia ladder (feeds
  ``glass-defaults.cuh``).
- ``THREAD_SWEEP_RESULTS.md`` — the thread-tier contender verdicts.
- ``BLAS2_SWEEP_RESULTS.md`` — warp-vs-block for syrk/syr2k/ldlt/ldltsv/inv/
  trmv/ger.
- ``RECT_SWEEP_RESULTS.md`` — warp-vs-block for rectangular gemv/gemm shapes.
- ``SOLVERS_SWEEP_RESULTS.md`` — bdsv-vs-pcg, gesv/posv/inv-solve, syev
  (characterization, never auto-picked).
- ``REDUCED_SWEEP_RESULTS.md`` — the serial-vs-``*_reduced`` corner.
- ``NVWARP_SWEEP_RESULTS.md`` — ``glass::warp`` vs the CUB warp tier (the
  measured case for a block-scope-only vendor dispatch).
- ``PAPER_SWEEPS.md`` — the hostblas/fusion/latency paper harnesses.

See :doc:`../concepts/tuning` for how to emit a per-host override table from a
sweep, and :doc:`../../api_reference/defaults` for the picker API.
