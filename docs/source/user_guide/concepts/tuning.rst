Tuning for Your Hardware
========================

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
gemm_batched_1d, row_strided_gemm, row_strided_gemv); each can be specialized
independently for a given (shape, SM).

Picking a backend: measured defaults
------------------------------------

Before the nvidia dispatch table (below), the higher-level question is *warp vs
block vs nvidia* for your op and size. The three-contender sweep
(``bench/run_mega_sweep.sh`` → ``bench/MEGA_SWEEP_RESULTS.md``) measures all three on
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
warp/block runner-up). The pick is host-/codegen-side because warp/block/nvidia need
different ``<<<grid, block>>>`` launches. Numbers are sm_120-seeded; ``bench/autotune.py``
regenerates a per-host table.

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
   # → measures all 5 auto-dispatching primaries (gemm, gemv, row_strided_gemv,
   #   row_strided_gemm, gemm_batched_1d) across each one's default shape grid
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
   python3 bench/autotune.py --apis row_strided_gemv --shapes "6,6,8;14,14,16"
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
