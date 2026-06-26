Backend Dispatch
================

The ``glass::nvidia::gemm`` / ``gemv`` / ``row_strided_*`` / ``gemm_batched_1d``
primary templates **auto-dispatch** at compile time: for shapes where pure-SIMT
wins they fall through to ``::glass::*``; for shapes where the vendor library
wins they route to cuBLASDx via the ``DEFINE_NVIDIA_*`` macros.

This means ``glass::nvidia::gemm<float, 6, 6, 6>(...)`` "just works" without any
DEFINE macro — small shapes route to SIMT automatically. Larger shapes such as
``glass::nvidia::gemm<float, 32, 32, 32>(...)`` still require a
``DEFINE_NVIDIA_GEMM(32, 32, 32)`` in scope, but produce a clean compile-time
message when it is missing.

The dispatch flow
-----------------

.. code-block:: text

                     caller writes:  glass::nvidia::gemm<float, M, N, K>(...)
                                            │
                                            ▼
                          should_use_cublasdx<float, M, N, K, SMS>()
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
                   false                                          true
                     │                                             │
          ────────── ▼ ──────────              ────────── ▼ ──────────
          SIMT fallback:                       Need a DEFINE_NVIDIA_GEMM*
          ::glass::gemm<T,M,N,K>(...)          to specialize for cuBLASDx;
          (no DEFINE needed; no smem)          else: static_assert error.

The decision is made at compile time by ``should_use_cublasdx*<T,M,N,K,SM>()``
(see ``src/nvidia/query.cuh`` / ``query_simt.cuh``), which consults — in order:

1. A per-build **local override** table (when ``GLASS_TUNING_TABLE_LOCAL`` is
   defined).
2. The shipped **global table** (``src/nvidia/tuning_table.cuh``).
3. A fallback **static heuristic**.

The shipped table covers ``sm_86`` (Ampere consumer) and ``sm_120``
(Blackwell-class) for square shapes from 3×3×3 up to 64×64×64.

The size heuristic
------------------

When no tuning-table entry matches, each API falls back to a per-API heuristic
that reflects its arithmetic intensity:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Template
     - Default heuristic
   * - ``cublasdx_wins<M, N, K, SM>`` (gemm)
     - ``max(M,N,K) >= 16 AND min(M,N,K) >= 4``
   * - ``cublasdx_wins_gemv<M, N, SM>``
     - ``max(M,N) >= 32``
   * - ``cublasdx_wins_batched<M, N, K, BATCH, SM>``
     - ``BATCH >= 8 AND max(M,N,K) >= 8``
   * - ``cublasdx_wins_gemm_strided<...>``
     - delegates to ``cublasdx_wins<>``
   * - ``cublasdx_wins_gemv_strided<...>``
     - delegates to ``cublasdx_wins_gemv<>``

Restricted to ``float``: the heuristic returns SIMT for non-float types.

Inspecting the decision
-----------------------

The ``print_dispatch*`` host helpers (``__host__ __device__``, so you can also
drop one into a kernel for runtime diagnostics) report the chosen path:

.. code-block:: cpp

   glass::nvidia::print_dispatch<float, 4, 4, 4>();
   // → glass::nvidia::gemm<T,4,4,4,SM=860>: SIMT fallback

   glass::nvidia::print_dispatch<float, 32, 32, 32>();
   // → glass::nvidia::gemm<T,32,32,32,SM=860>: cuBLASDx (needs DEFINE_NVIDIA_GEMM*)

   glass::nvidia::print_dispatch_gemv<float, 64, 64>();
   // → glass::nvidia::gemv<T,64,64,SM=860>: cuBLASDx (needs DEFINE_NVIDIA_GEMV*)

Overriding the dispatch
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Goal
     - How
   * - Force cuBLASDx for a shape the heuristic puts in SIMT
     - Add ``DEFINE_NVIDIA_GEMM(M,N,K)`` in your ``.cu`` file — the explicit specialization always overrides the primary template.
   * - Force SIMT for a shape the heuristic puts in cuBLASDx
     - Call ``::glass::gemm<T,M,N,K>(...)`` directly (skip the ``nvidia::`` path).
   * - Per-host tuning without editing source
     - Run ``python bench/autotune.py`` to generate ``bench/tuning/<hostname>.cuh``, then compile with ``-DGLASS_TUNING_TABLE_LOCAL='"bench/tuning/<hostname>.cuh"'``.
   * - Different SM in-tree (for a PR)
     - ``python bench/autotune.py --in-tree`` rewrites the marker-delimited specializations inside ``src/nvidia/tuning_table.cuh``.

The shipped values are sensible defaults, but small-GEMM performance is highly
SM-dependent. See :doc:`tuning` for the full per-host autotuning workflow that
measures both paths side by side on your hardware so you can override these
defaults.
