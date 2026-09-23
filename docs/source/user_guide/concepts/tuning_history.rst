:orphan:

Historical Orin Power and Tuning Measurements
============================================================

.. warning::

   This page preserves the earlier four-backend power study for provenance.
   Its counts, ratios, and broad conclusions describe that historical campaign.
   For the paper's 29%/91% runtime increases and 9/396 changes per mode, see
   :doc:`tuning` and :doc:`../tutorials/paper_results`.

**Historically, no material retune was needed per power mode.** Before the
NVIDIA-thread contender was added, the same Orin measured at all three standard
``nvpmodel`` modes slows by a median 1.49× (30 W → 15 W), 1.31× (50 W →
30 W), 1.95× end to end — but the picks barely move: 8 of 396 cells differ
between 15 W and 30 W, 11 between 30 W and 50 W, 7 across the full span.
For scale, two independent 50 W captures of the *same* board disagree on 5
cells, so power-mode disagreement is close to plain re-measurement noise,
while the architecture change (125 cells) is an order of magnitude beyond
it. Regenerating the table from the 15 W or 30 W capture instead of 50 W
changes 2 and 4 lines of emitted code, against 2 for a same-mode
re-measurement. **A dispatch table is a property of the silicon, not of the
power budget** — retune once per architecture and every deployment power
mode is covered.

Two practical notes from the Orin bring-up:

* NVIDIA ships no native MathDx host package for Tegra, but the cuSOLVERDx
  **LTO-IR fatbins are
  architecture-neutral**: ``tune.py`` detects a non-x86 host and stages a
  separate-compilation device link against the fatbin, so Jetson runs the full
  native/NVIDIA ladder. In the current capture, the NVIDIA block and thread
  tiers take 87 and 52 of 396 raw ladder cells respectively; the shipped
  legacy throughput table retained NVIDIA thread in 15 of 132 cells. New
  captures replace that asymmetric confirmation with the unified fresh-input
  solver ladder described above; these historical counts are retained only to
  document the earlier release.
* The ``nvpmodel`` labels are ceilings, not draws. Sampling the board rails at
  1 Hz with the GPU ≥98.6 % busy, the whole ladder pulls 9.2 W in the 15 W
  mode, 13.4 W in the 30 W mode and 16.0 W in the 50 W mode. Small
  block-resident linear algebra is clock-bound long before it is power-bound,
  so the fastest standard mode is also the most efficient, monotonically: per
  problem solved, 30 W costs 1.09× and 15 W costs 1.12× the energy of 50 W.
  **Race to idle** — run the highest standard mode your thermals allow and let
  the board idle between control cycles.

**In that four-backend power-mode study, how reproducible was a retune?** Two independent 50 W captures of the same
board (different sessions, hours apart) crown the same winner in 391 of 396
cells (98.7 %), and originally generated tables differing in exactly one
line: ``gemm`` f64 near N=48, where the block and warp tiers land within 1 %
of each other.

That single flip exposed a rough edge the generator has since closed. The
±5 % tie rule always governed whether a *dependency* tier
(cuBLASDx/cuSOLVERDx) may take a cell from the no-dependency SIMT tiers — but
between two SIMT tiers it originally took the raw minimum, so sub-1 %
run-to-run noise could change an emitted line without anything real having
changed. The generator now applies a **±2 % SIMT tie band** as well: any
dependency-free tier within 2 % of the fastest takes the cell if it is
*simpler* (thread ≻ warp ≻ block — sequential beats shuffles beats barriers;
``bench/tune_pick.py``). Under the fixed generator all four Orin captures —
the two independent 50 W sessions, the 30 W and even the 15 W — emit a
byte-identical ``ideal_sm87``: the generated table really is a property of
the silicon, invariant across re-measurement *and* the whole power envelope. The rule also cleans up noise
artifacts frozen into earlier tables — e.g. the sm_120 ``gemm`` f32 line
interleaved warp and block below N=24 on gaps under 2 %, and on sm_87 the
``gemm`` f64 block/warp boundary sat at N=16 when the two tiers are actually
within 1 % of each other all the way to N=96 (block's one real win, 24 %, is
at N=128 — where the boundary now lands).

Raw captures, provenance bundles and the analysis scripts behind these numbers
live in the paper repository (``data/jetson/``), not here — this repo ships the
generated tables and the harnesses, not the measurement archive.
