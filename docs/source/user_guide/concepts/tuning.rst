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

   python bench/tune.py --sm auto --prebuild --build-jobs 1   # shared-host safe default
   python bench/tune.py --sm auto              # all legs, ±5% margin (reuses the prebuilt cache)
   python bench/tune.py --sm auto --quick      # ladder throughput point only (faster)
   python bench/tune.py --legs ladder,reduced  # pick legs; --margin to retune the tie band
   python bench/tune.py --sm auto --dry-run    # regenerate + diff, write nothing

**Prebuild so the sweep is fast.** Compilation — not timing — dominates the wall
clock (the ``shapes`` leg alone compiles ~66 separate cuBLASDx microbenches).
``--prebuild`` compiles every binary the selected legs need into a persistent,
hash-keyed cache (``bench/.tune_cache/sm<sms>/``) and runs nothing — so you can
run it **anytime, even while the GPU is busy** (compilation is CPU-bound). Because
building isn't timed, a dedicated host may fan it out with ``--build-jobs N``
(size to free_RAM/7 — each cuBLASDx compile needs ~6-7GB); keep one job on a
shared host. The later timed sweep on a quiet GPU is then
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

Every new capture records its UTC start, commit, dirty-source digest,
architecture, compiler, and nearest signed correctness-receipt fingerprint.
Timed drivers refuse a busy GPU and invalidate a leg if a foreign compute PID
appears. Correctness remains a separate signed gate. Candidate-only A/Bs use
``bench/perf_sweeps.py``; ``--build-only`` is safe while the host is shared and
the run without it is reserved for a quiet window. Release confirmation uses
``--profile overnight``. The complete capture-first sequence is
``bench/run_quiet_audit.sh``; after an interrupted first pass, ``--resume``
repairs the changed MathDx shard and carries only fingerprint-identical shards
before continuing.

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
(``bench/tune.py --legs ladder`` → ``bench/RESULTS.md``) measures all three on
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
factor/solve/gemm from N≈32 (up to ~8×). See ``bench/RESULTS.md`` for the full
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

**No, per power mode.** The same Orin measured at all three standard
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

* NVIDIA ships no MathDx for Tegra, but the cuSOLVERDx **LTO-IR fatbins are
  architecture-neutral**: ``tune.py`` detects a non-x86 host and stages a
  separate-compilation device link against the fatbin, so Jetson runs the full
  four-tier ladder. It is worth having — the vendor tier wins 118 of 396 cells
  on sm_87 (Cholesky up to 3.7× over the best SIMT tier at small N).
* The ``nvpmodel`` labels are ceilings, not draws. Sampling the board rails at
  1 Hz with the GPU ≥98.6 % busy, the whole ladder pulls 9.2 W in the 15 W
  mode, 13.4 W in the 30 W mode and 16.0 W in the 50 W mode. Small
  block-resident linear algebra is clock-bound long before it is power-bound,
  so the fastest standard mode is also the most efficient, monotonically: per
  problem solved, 30 W costs 1.09× and 15 W costs 1.12× the energy of 50 W.
  **Race to idle** — run the highest standard mode your thermals allow and let
  the board idle between control cycles.

**How reproducible is a retune?** Two independent 50 W captures of the same
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

Why bother?
-----------

Small-GEMM performance is highly SM-dependent, so the shipped heuristic is only
a default. An illustrative legacy measurement (undated early capture whose
device metadata was not recorded — kept for the *shape* of the crossover, not
the numbers; regenerate with ``bench/autotune.py`` for current hardware):

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

Running a tune (the operational runbook)
----------------------------------------

The step-by-step commands — per-leg invocations, prebuild/quiet-window
separation, the measurement methodology (min-of-3, spread capture, warmup,
telemetry), per-API shape grids, and the contribution checklist — live in the
repository runbook `bench/TUNING.md
<https://github.com/A2R-Lab/GLASS/blob/main/bench/TUNING.md>`_. This page
stays conceptual so the two never drift: the runbook says *how to run*, this
page says *what the machinery is and what a retune changes*.

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

Solver-level calibration: the cutoff recipe
-------------------------------------------

The measured-selection flow extends one level above operations, to *solver*
choice — and here the library deliberately ships a contract and a recipe, not
a policy. ``glass::pcg`` (iterative, warm-start friendly) and ``glass::bdsv``
(direct, flat cost) consume **bit-identical** ``[L|D|R]`` strips and padded
vectors; that layout compatibility is a declared behavioral obligation
(``linsys_layout_compatibility`` in ``test/coverage-obligations.json``),
checked by the signed receipt, so switching solvers per solve costs the caller
nothing but a mode flag. Which solver to run — and where the cutoff sits — is
workload evidence the application owns, because the useful switching signals
(a warm-start quality estimate, a disturbance flag, an iteration budget) live
above the algebra layer.

The recipe mirrors ``tune.py``:

#. **Probe** — run your real workload once per candidate policy on a quiet
   GPU, recording per-solve wall times (not just means: keep the traces, the
   interesting differences are in the tail percentiles).
#. **Fit** — pick the cutoff that optimizes the statistic your deployment
   actually bounds (p99/max for deadlines, mean for throughput). Cutoff bands
   are usually wide; prefer the band center over the razor edge.
#. **Persist** — ship the decision as a default in your configuration with
   the capture it came from, and re-measure per problem class and per GPU,
   exactly as GLASS re-measures its dispatch tables per architecture.

A worked end-to-end example lives downstream: GATO's ``linsys="auto"``
controller switches per solve on prediction error, with a probe/fit/persist
autotune script (``tools/autotune_linsys.py``) calibrating the threshold per
robot — a few application lines, enabled by the layout contract above.

Contributing measurements upstream
----------------------------------

See the "Contributing upstream" section of `bench/TUNING.md
<https://github.com/A2R-Lab/GLASS/blob/main/bench/TUNING.md>`_ for the two
routes (per-host override file vs ``--in-tree``) and the what-not-to-contribute
checklist.
