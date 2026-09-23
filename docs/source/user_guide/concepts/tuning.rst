Tuning for Your Hardware
========================

One command — ``bench/tune.py``
-------------------------------

GLASS ships measured native/NVIDIA execution plans, a vendor implementation
table, and explicit-algorithm characterization. The native and NVIDIA
thread/warp/block **backend ladder** (``glass-defaults.cuh``, consumed by
``glass::recommend<>``),
the per-(M,N,K) **cuBLASDx-vs-SIMT table** (``src/nvidia/tuning_table.cuh``, the
main subject below), and the serial-vs-reduced characterization are all driven
by ``bench/tune.py``. It remeasures them on your GPU and regenerates
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
(``nvidia`` / ``nvidia_thread`` / ``cublasdx`` / ``reduced``) wins **only if it beats the simplest
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

GLASS's ``glass::nvidia::block::*`` wrappers — ``gemm``, ``gemv``, ``row_strided_*``,
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

Before the nvidia dispatch table (below), the higher-level question is *thread
vs warp vs block vs NVIDIA block vs NVIDIA thread* for your op and size. The
five-contender sweep (``bench/tune.py --legs ladder``) measures them on
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
   * - ``potrf`` / ``trsv`` / ``posv``
     - **warp**; block fallback **TB=32**
     - 32
     - 2–4

Rule of thumb: **warp-per-problem by default**; ``gemv`` → block past N≈48, ``gemm`` →
block once non-tiny. Factor/solve want block ``TB=32`` — extra threads idle on the
serial pivot and TB>32 *hurts*.

**If you link MathDx**, both ``glass::nvidia::block`` and
``glass::nvidia::thread``
interfaces enter the ladder where supported. Which one wins is not monotonic:
the current sm_120 and sm_87 tables select NVIDIA thread for some small
``potrf``/``trsv``/``posv`` cells, NVIDIA block elsewhere, and native tiers in
the remaining bands. (The sm_72 Xavier table is native-only by construction —
its CUDA 11.4 toolchain predates the device-callable MathDx libraries.) See
:doc:`../tutorials/sweep_results` and
``bench/RESULTS.md`` for the dated per-op × per-precision results.

In-place solver timing uses a separate authoritative ladder. The general
ladder's back-to-back launches are suitable for non-destructive operations,
but an in-place solver would consume its own output after the first launch.
``bench_solver_ladder.cu`` therefore remeasures POTRF, TRSV, and POSV for every
supported native block/warp/thread and NVIDIA block/thread launch plan. Each
timed launch consumes a fresh valid system from a bounded ring, while input
initialization remains outside the timed region. Plans are randomized within
nine paired rounds and every raw sample is recorded. Any contender can win
under the same 5% dependency and ±2% SIMT tie rules. A selected NVIDIA plan is
additionally interval-confirmed: it is kept only when its slowest raw round
still clears every native plan's fastest round by the margin, and an ambiguous
vendor pick keeps the capture's native winner instead — the same documented
preference for dependency-free code that the margin itself encodes. Missing
solver cells make regeneration fail closed. Numerical correctness remains a separate signed-
receipt requirement, not something inferred from timing agreement.

The ``constexpr`` ``glass::recommend<op, T, dims...>()`` query returns one
``execution_plan`` containing family, scope, and launch packing.
Pass ``dependency_set::mathdx`` explicitly to admit NVIDIA candidates;
``native_only`` is the default. Each measured architecture stores both the
full winner and the measured native-only winner for every cell. The pick is
host-/codegen-side because
the tiers need different ``<<<grid, block>>>`` launches. Tables are per-arch
(``ideal_sm120``, ``ideal_sm87``, and ``ideal_sm72`` today)
behind an SM dispatch; ``bench/tune.py --sm auto`` adds or refreshes your GPU's table
(and the tables below) in-tree, leaving other arches' tables untouched.

Note that ``recommend<>`` advises **launch-level** packing — the caller
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

What a retune actually changes (sm_120 vs sm_87 vs sm_72)
---------------------------------------------------------

GLASS ships three measured architectures today: ``sm_120`` (RTX 5090, 170
SMs), ``sm_87`` (Jetson AGX Orin, 16 SMs, integrated memory), and ``sm_72``
(Jetson AGX Xavier, 8 SMs, native-only — CUDA 11.4 predates the
device-callable MathDx libraries). Comparing the paired 2026-09-02
fresh-input A/B captures that generated the shipped tables is the clearest
answer to "do I need to retune?".

**Yes, per architecture.** Of the 396 (op, N, precision, batch) cells
measured on every machine, the Orin's recommended placements differ from the
RTX 5090's in **145** and from the Xavier's in **162**. Part of the Xavier
gap is structural (no MathDx tier available), but restricting all three GPUs
to their common native thread/warp/block candidates still changes 107
(RTX–Orin), 94 (RTX–Xavier), and 66 (Orin–Xavier) recommendations. The
stakes are real: across 132 Orin cells at NPROB=8192 spanning both precisions, the best
and worst placements for a cell differ by a **median of 4.9× (up to 81×)**,
and no library source differs between the machines — only the measured
tables do. With far fewer SMs to fill, the Jetsons pack more problems per
warp or thread where the RTX spreads one problem across more lanes, but the
movement is not one-directional.

**Power mode in the paper campaign.** Relative to Orin's 50 W mode, 30 W and
15 W increase geometric-mean runtime by 29% and 91%, respectively. Each changes
9/396 recommendations, with 5–6 of those within the dispatch margin. This
supports reusing the table across the measured power modes, while absolute
throughput still falls. It does not establish invariance for every power or
thermal configuration.

The complete offline tuning campaigns took 5.5 hours on Orin, 9 hours on
Xavier, and 2 hours on RTX 5090. Retune for a new architecture and toolchain
configuration. For the older power measurements and tuning history, see
:doc:`tuning_history`; for the release comparisons, see
:doc:`../tutorials/paper_results`.

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
       glass::nvidia::block::print_dispatch<float, 6, 6, 6>();
       // → "glass::nvidia::block::gemm<T,6,6,6,SM=860>: SIMT fallback"
       glass::nvidia::block::print_dispatch_gemv<float, 64, 64>();
       // → "glass::nvidia::block::gemv<T,64,64,SM=860>: cuBLASDx"
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
