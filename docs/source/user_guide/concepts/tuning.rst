Tuning for Your Hardware
========================

One command — ``bench/tune.py``
-------------------------------

GLASS ships three measured defaults tables: the thread/warp/block/nvidia **backend
ladder** (``glass-defaults.cuh``, consumed by ``glass::suggested_backend<>``),
the per-(M,N,K) **cuBLASDx-vs-SIMT table** (``src/nvidia/tuning_table.cuh``, the
main subject below), and the serial-vs-reduced **``suggested_use_reduced<>``**
predicate. ``bench/tune.py`` remeasures all of them on your GPU and regenerates
them under **one shared noise margin**, so nothing bakes sub-noise jitter and a
pure-noise re-run reproduces the same tables:

.. code-block:: bash

   python bench/tune.py --sm auto --prebuild --build-jobs 6   # compile everything in parallel (no GPU)
   python bench/tune.py --sm auto              # all legs, ±5% margin (reuses the prebuilt cache)
   python bench/tune.py --sm auto --quick      # ladder throughput point only (faster)
   python bench/tune.py --legs ladder,reduced  # pick legs; --margin to retune the tie band
   python bench/tune.py --sm auto --dry-run    # regenerate + diff, write nothing

**Prebuild so the sweep is fast.** Compilation — not timing — dominates the wall
clock (the ``shapes`` leg alone compiles ~66 separate cuBLASDx microbenches).
``--prebuild`` compiles every binary the selected legs need into a persistent,
hash-keyed cache (``bench/.tune_cache/sm<sms>/``) and runs nothing — so you can
run it **anytime, even while the GPU is busy** (compilation is CPU-bound). Because
building isn't timed, fan it out with ``--build-jobs N`` (size to free_RAM/7 —
each cuBLASDx compile needs ~6-7GB). The later timed sweep on a quiet GPU is then
**execute-only**, and always runs serially for clean measurement. The cache is
keyed on the rendered source + a digest of the whole header library + the SM, so a
library edit transparently rebuilds only the affected binaries.

The shared rule (``bench/tune_pick.py::pick``): a dependency-carrying impl
(``nvidia`` / ``cublasdx`` / ``reduced``) wins **only if it beats the simplest
impl by more than the margin** — otherwise the no-dependency path (always
launchable, no MathDx) stays. Every op is measured and recorded; a dispatch
picker is regenerated only where ≥2 impls genuinely compete. **Run on a quiet
GPU** — perf timing must be isolated from other CPU/GPU load. Use ``--dry-run``
first to confirm a re-run only moves dispatch inside the tie band before
committing. The sections below describe the two tables ``tune.py`` drives — the
cuBLASDx-vs-SIMT table (its ``shapes`` leg, also runnable standalone as
``bench/autotune.py``) and the backend ladder (its ``ladder`` leg).

The cuBLASDx-vs-SIMT table
--------------------------

GLASS's ``glass::nvidia::*`` wrappers — ``gemm``, ``gemv``, ``row_strided_*``,
``gemm_batched_1d`` — auto-dispatch between a pure-SIMT path and cuBLASDx at
compile time (see :doc:`backend_dispatch`). The decision lives in
``src/nvidia/query_simt.cuh::should_use_cublasdx*<>()`` and consults, in order:

1. A per-shape specialization in ``src/nvidia/tuning_table.cuh`` if one exists
   (compile-time template specialization — zero runtime cost).
2. A per-build local override included by ``tuning_table.cuh`` when
   ``GLASS_TUNING_TABLE_LOCAL`` is defined.
3. A static per-API heuristic for unmeasured shapes.

Five per-API decision templates live in ``_glass_tuning`` (gemm, gemv,
gemm_batched_1d, gemm_strided, gemv_strided); each can be specialized
independently for a given (shape, SM).

Picking a backend: measured defaults
------------------------------------

Before the nvidia dispatch table (below), the higher-level question is *warp vs
block vs nvidia* for your op and size. The three-contender sweep
(``bench/tune.py --legs ladder`` → ``bench/MEGA_SWEEP_RESULTS.md``) measures all three on
one ns/problem axis. Numbers below are **RTX 5090 / sm_120**; breakevens shift on other
GPUs, so re-run the sweep on yours.

**Most builds don't link MathDx — start with warp vs block (no dependency):**

.. list-table::
   :header-rows: 1
   :widths: 26 34 20 20

   * - op
     - default (batched throughput)
     - block ``TB``
     - warp ``WPB``
   * - ``dot``
     - **warp** at every N (2–6×)
     - 64
     - 8–16
   * - ``gemv``
     - **warp** ≤ N≈32, **block** ≥ N≈48
     - 64–128
     - 2–4
   * - ``gemm``
     - **warp** ≤ N≈8, else **block**
     - scale 64→256 with N
     - 2–4
   * - ``chol`` / ``trsv`` / ``posv``
     - **warp**; block fallback **TB=32**
     - 32
     - 2–4

Rule of thumb: **warp-per-problem by default**; ``gemv`` → block past N≈48, ``gemm`` →
block once non-tiny. Factor/solve want block ``TB=32`` — extra threads idle on the
serial pivot and TB>32 *hurts*.

**If you link MathDx** (``glass::nvidia::``), the vendor path wins a middle band (f32):
``gemm`` N≈16–64 (block above; cuBLASDx is smem-capped past 64 here), ``chol``/``posv``
N≥16 through 128 (cuSOLVERDx, 1.5–2.7×), ``trsv`` only N≈16–32 (warp wins above). In
**f64** the band is narrower (≈ N=16–64; the double descriptors hit the ~99 KB opt-in
smem cap at 64). For a *single* large problem (batch≈1), the vendor path wins
factor/solve/gemm from N≈32 (up to ~8×). See ``bench/MEGA_SWEEP_RESULTS.md`` for the full
per-op × per-precision tables.

These defaults are also exposed as ``constexpr`` helpers in ``glass-defaults.cuh`` —
``glass::suggested_backend<op, N, T>()``, ``suggested_block_threads<>()`` and
``suggested_warps_per_block<>()`` — so callers and codegen can pick a backend + launch
config without hand-copying the table. Include it after ``glass.cuh`` (and after
``glass-nvidia.cuh`` to make the ``nvidia`` tier eligible; otherwise it collapses to the
warp/block runner-up). The pick is host-/codegen-side because the tiers need
different ``<<<grid, block>>>`` launches. (The sm_120 tables include the ``thread``
tier as of the 2026-07-18 sweep — see the note in
:doc:`../../api_reference/defaults`.) Tables are per-arch (``ideal_sm120`` today)
behind an SM dispatch; ``bench/tune.py --sm auto`` adds or refreshes your GPU's table
(and the tables below) in-tree, leaving other arches' tables untouched.

Note that ``suggested_backend<>`` advises **launch-level** packing — the caller
changes the ``<<<grid, block>>>``. Distinct from it, ``glass::dispatch_body()``
(``glass-dispatch.cuh``) picks the **in-block body** behind the bare
``glass::op`` face under a *fixed* block-scope calling contract — the launch
does not change. The ``body`` leg (``tune.py --legs body``, harness
``bench_body_dispatch.cu``) measures three bodies per (op, N, dtype) cell —
full-block SIMT / warp 0 / thread 0, each + block sync — across block widths
32–256 and regenerates the per-arch table under a deliberately stricter rule
than the ladder's: a body takes a cell only if it is never worse than block by
more than the margin at *any* measured (batch, width) point AND better by more
than the margin at ≥1 width in the throughput regime; verdicts are bounded at
the largest measured N and unmeasured arches stay block. A moved cell matches
block to reduction-order tolerance, not bit-exactly — the retune is a
receipt-attested event, never a silent change. Consumers that need
bit-stability across retunes pin ``glass::block::`` explicitly (see
:doc:`namespaces`).

.. _tuning-per-arch-results:

What a retune actually changes (sm_120 vs sm_87)
------------------------------------------------

GLASS ships two measured architectures today: ``sm_120`` (RTX 5090, 170 SMs)
and ``sm_87`` (Jetson AGX Orin, 16 SMs, integrated memory). Comparing them is
the clearest answer to "do I need to retune?".

**Yes, per architecture.** Of the 396 (op, N, precision, batch) cells measured
on both, **125 (32 %) crown a different tier** — and systematically toward more
problem packing on the smaller part: 34 cells move warp → thread, 27 block →
warp, 17 nvidia → thread. The thread tier's share nearly doubles (64 → 118
cells). With far fewer SMs to fill, packing more problems per warp beats
spreading one problem across more lanes. No library source differs between the
two machines; a third of the dispatch decisions do.

**No, per power mode.** The same Orin measured at 15 W, 30 W and 50 W (the
standard ``nvpmodel`` modes) changes only **4 of 330** cells between 30 W and
15 W, all near-tie boundaries, even though everything slows by a median
1.49× (P10–P90 1.46–1.51×). Regenerating the table from the 30 W capture
instead of the 50 W one changes two lines of emitted code out of twelve
(operation, precision) groups. **A dispatch table is a property of the
silicon, not of the power budget** — retune once per architecture and every
deployment power mode is covered.

Two practical notes from the Orin bring-up:

* NVIDIA ships no MathDx for Tegra, but the cuSOLVERDx **LTO-IR fatbins are
  architecture-neutral**: ``tune.py`` detects a non-x86 host and stages a
  separate-compilation device link against the fatbin, so Jetson runs the full
  four-tier ladder. It is worth having — the vendor tier wins 118 of 396 cells
  on sm_87 (Cholesky up to 3.7× over the best SIMT tier at small N).
* The ``nvpmodel`` labels are ceilings, not draws. Sampling the board rails at
  1 Hz with the GPU ≥98.7 % busy, the whole ladder pulls 13.4 W in the 30 W
  mode and 16.0 W in the 50 W mode. Small block-resident linear algebra is
  clock-bound long before it is power-bound, so the fastest standard mode is
  also the most efficient: 50 W is 1.31× faster than 30 W for 1.20× the power,
  i.e. **0.91× the energy per problem**. Race to idle.

**How reproducible is a retune?** Two independent 50 W captures of the same
board (different sessions, hours apart) crown the same winner in 391 of 396
cells (98.7 %), and generate tables differing in exactly one line: ``gemm``
f64 near N=48, where the block and warp tiers land within 1 % of each other.

That single flip exposes a rough edge worth knowing about. The ±5 % tie rule
governs whether a *dependency* tier (cuBLASDx/cuSOLVERDx) may take a cell from
the no-dependency SIMT tiers — but **between two SIMT tiers the generator takes
the raw minimum**, so sub-1 % run-to-run noise can change an emitted line
without anything real having changed. If you regenerate and see a one-line
diff in a near-tie band, that is what you are looking at, not a hardware
finding. (Applying a margin between SIMT tiers as well — preferring the
simpler tier on a tie — would make the tables reproducible across
re-measurements; it is not done today because it would perturb every shipped
table and so wants its own attested regeneration.)

Raw captures, provenance bundles and the analysis scripts behind these numbers
live in the paper repository (``data/jetson/``), not here — this repo ships the
generated tables and the harnesses, not the measurement archive.

Why bother?
-----------

Small-GEMM performance is highly SM-dependent, so the shipped heuristic is only
a default. A representative measurement (RTX 3080, sm_120):

.. list-table::
   :header-rows: 1
   :widths: 28 22 28 22

   * - Shape
     - Heuristic says
     - Measured winner
     - Speedup
   * - gemm 14×14×14
     - SIMT
     - SIMT
     - matches
   * - gemm 24×24×24
     - cuBLASDx
     - **cuBLASDx**
     - 2.4×
   * - gemm 6×6×6
     - SIMT
     - **SIMT**
     - 2.3×
   * - gemv 5×5
     - SIMT
     - **SIMT**
     - matches

For shapes well-covered by the in-tree table this is "free perf". For unmeasured
shapes you trust the heuristic; once you bench it, you can specialize it and
either keep it local or PR it upstream.

Quick start
-----------

.. code-block:: bash

   cd GLASS
   python3 bench/autotune.py
   # → measures all 5 auto-dispatching primaries (gemm, gemv, gemv_strided,
   #   gemm_strided, gemm_batched_1d) across each one's default shape grid
   # → writes bench/tuning/<hostname>.cuh with the per-host specializations

The script:

1. Detects your local SM via ``nvidia-smi``.
2. For each requested API, measures both backends across that API's shape grid.
3. Picks the faster path per (shape, SM).
4. Emits one explicit specialization per measured shape into
   ``bench/tuning/<hostname>.cuh``, plus a human-readable ``*_results.md``.

Ties (within ``--margin``, default ±5 %) default to SIMT. ``MATHDX_ROOT`` must
be set. The shipped ``src/nvidia/tuning_table.cuh`` is **never** overwritten by
the default flow — it carries the per-API primaries, default heuristics, and a
curated set of in-tree specializations, and stays stable as the baseline.

Restricting the run:

.. code-block:: bash

   python3 bench/autotune.py --apis gemm,gemv
   python3 bench/autotune.py --apis gemv_strided --shapes "6,6,8;14,14,16"
   python3 bench/autotune.py --apis gemv --shapes '6,6;14,14;32,32' --iters 20000 --dry-run

``--shapes`` takes a ``;``-separated tuple list; the arity must match the chosen
API (3 values for ``gemm``, 2 for ``gemv``, etc.). ``--dry-run`` reports without
writing.

Consuming your per-host overrides
---------------------------------

The per-host file is included via the ``GLASS_TUNING_TABLE_LOCAL`` macro:

.. code-block:: bash

   nvcc ... -DGLASS_TUNING_TABLE_LOCAL='"bench/tuning/<hostname>.cuh"' ...

The named header is ``#include``d at the bottom of ``_glass_tuning`` and may add
specializations for shapes **not already specialized in the shipped table**.
(C++ disallows re-specialization; to override a shape the shipped table already
covers, edit ``tuning_table.cuh`` directly or remove the in-tree entry first.)
Per-host files under ``bench/tuning/`` are gitignored.

Debugging dispatch decisions
----------------------------

.. code-block:: cpp

   #include "glass-nvidia.cuh"

   int main() {
       glass::nvidia::print_dispatch<float, 6, 6, 6>();
       // → "glass::nvidia::gemm<T,6,6,6,SM=860>: SIMT fallback"
       glass::nvidia::print_dispatch_gemv<float, 64, 64>();
       // → "glass::nvidia::gemv<T,64,64,SM=860>: cuBLASDx (needs DEFINE_NVIDIA_GEMV*)"
   }

These are ``__host__ __device__`` so you can call them from ``main`` for
build-time confirmation or drop one into a kernel for runtime diagnostics.

Contributing upstream
---------------------

If your measurements would meaningfully improve the shipped table (a new SM, or
a shape range the curated entries miss), contribute back. Two routes:

**Option A — submit your per-host file unchanged.** Rerun autotune and attach
the contents of ``bench/tuning/<hostname>.cuh`` to a PR. Reviewers spot-check
and merge specific specializations into ``src/nvidia/tuning_table.cuh``.

**Option B — update the shipped table directly:**

.. code-block:: bash

   python3 bench/autotune.py --sm AUTO --in-tree

``--in-tree`` writes the new specializations into a marker-delimited section
inside ``src/nvidia/tuning_table.cuh`` while preserving the primary templates,
default heuristics, and the ``GLASS_TUNING_TABLE_LOCAL`` hook. The markers are:

.. code-block:: text

   // === BEGIN: autotune-generated specializations ===
   // ...
   // === END: autotune-generated specializations ===

Re-running ``--in-tree`` replaces the section in place; running without it
writes only to ``bench/tuning/<hostname>.cuh``.

What **not** to contribute:

* Entries within 5 % of each other (autotune marks these "tie within ±5 % →
  SIMT default" — don't second-guess that filter).
* Measurements from a thermally throttled GPU. Run ``nvidia-smi -q -d CLOCK``
  first; you want the GPU at peak boost.
* Measurements with ``--iters`` below ~5000 (high variance for sub-microsecond
  ops).
* Entries for shapes that aren't realistic for any workload (``M=N=K=2`` etc.).
