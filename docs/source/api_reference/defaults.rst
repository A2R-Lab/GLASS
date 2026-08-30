Backend Picker (``glass-defaults.cuh``)
=======================================

Queryable backend-selection defaults — the measured native and NVIDIA
thread / warp / block
ladder (``bench/RESULTS.md``) exposed as ``constexpr`` helpers, so callers
and GRiD-style codegen pick a backend + launch config instead of hand-copying a table.

The pick **cannot** be a device function: the tiers need different
``<<<grid, block>>>`` launches, so the decision happens host-side / at codegen time. See
:doc:`../user_guide/concepts/tuning` for the underlying numbers.

Distinct from this launch-level advisor, the tiny ``glass-dispatch.cuh``
header (included by both ``glass.cuh`` and this one; shared ``glass::op``
enum) carries ``glass::dispatch_body()`` — the measured **in-block body**
table behind the bare ``glass::op`` face, applied automatically by the
wrappers in ``src/base/dispatch.cuh`` under a *fixed* launch. See
:doc:`../user_guide/concepts/namespaces`.

.. note::

   ``backend::thread`` and ``backend::nvidia_thread`` are measured launch-level
   choices in the sm_120 and sm_87 tables. Either pick means one problem per
   thread: ``<<<ceil(P/TPB), TPB>>>`` with
   ``suggested_threads_per_block<>()``. The NVIDIA thread choice requires
   cuSOLVERDx 0.4+ and is currently eligible only for ``chol``, ``trsv``, and
   ``posv`` through ``N=32``. The ``ideal_generic`` fallback for unswept
   architectures remains warp/block/nvidia-only.

Include order
-------------

Include ``glass-defaults.cuh`` **after** ``glass.cuh``, and after ``glass-nvidia.cuh`` if you
want NVIDIA tiers to be eligible (it reads ``GLASS_HAVE_CUBLASDX``,
``GLASS_HAVE_CUSOLVERDX``, and ``GLASS_HAVE_CUSOLVERDX_THREAD``). With only
``glass.cuh`` linked, dependency-backed picks collapse to a native runner-up,
so a no-MathDx caller always gets a backend it can launch.

Helpers
-------

.. code-block:: cuda

   enum class glass::op      { dot, gemv, gemm, chol, trsv, posv };
   enum class glass::backend { warp, block, nvidia, thread, nvidia_thread };  // append-only

   // Which backend for (op, N, T) on this SM? (NVIDIA tiers only when MathDx is linked)
   template <op Op, uint32_t N, typename T, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr backend  glass::suggested_backend();

   // For the `block` backend: factor/solve want 32; gemm grows with N; dot/gemv 64–128.
   template <op Op, uint32_t N, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr uint32_t glass::suggested_block_threads();

   // For the `warp` backend: dot packs 8; others 2 warps/block.
   template <op Op, uint32_t N = 0, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr uint32_t glass::suggested_warps_per_block();

   // For the `thread` backend: launch <<<ceil(P/TPB), TPB>>>, one problem per thread.
   // Seed heuristic (shrinks as N*N registers/thread grow), NOT measured by the ladder leg.
   template <op Op, uint32_t N = 0, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr uint32_t glass::suggested_threads_per_block();

Example
-------

.. code-block:: cuda

   #include "glass.cuh"
   #include "glass-defaults.cuh"   // (after glass-nvidia.cuh too, to allow NVIDIA tiers)

   constexpr auto be = glass::suggested_backend<glass::op::chol, N, float>();
   if      constexpr (be == glass::backend::nvidia) { /* cuSOLVERDx launch */ }
   else if constexpr (be == glass::backend::nvidia_thread) {
       constexpr int TPB = glass::suggested_threads_per_block<glass::op::chol, N, float>();
       /* glass::nvidia::thread::potrf<float,N> in <<<ceil(P/TPB),TPB>>> */
   }
   else if constexpr (be == glass::backend::warp)   { /* <<<ceil(P/WPB), {32,WPB}>>> */ }
   else if constexpr (be == glass::backend::thread) {
       constexpr int TPB = glass::suggested_threads_per_block<glass::op::chol, N, float>();
       /* <<<ceil(P/TPB), TPB>>> */
   } else /* block */ {
       constexpr int TB = glass::suggested_block_threads<glass::op::chol, N, float>();
       /* <<<P, TB>>> */
   }

A runnable version is ``examples/08_backend_picker.cu``.

Per-host override
-----------------

The shipped tables are per-arch: each swept SM has its own ``constexpr`` ladder
(``ideal_sm120`` today, measured on an RTX 5090) behind an SM dispatch, and running
``bench/tune.py --sm auto`` on a new GPU (e.g. a Jetson Orin, sm_87) adds that arch's
table + dispatch case in-tree without touching the others; unmeasured SMs fall back to
a coarse heuristic. Alternatively, for a host-local override that leaves the shipped
tables alone, regenerate a table from a sweep run and point
``GLASS_DEFAULTS_TABLE_LOCAL`` at it:

.. code-block:: bash

   cd bench && ./run_mega_sweep.sh sm_XX
   python3 autotune.py --emit-defaults mega_sweep_<ts>.txt        # -> bench/tuning/<host>_defaults.cuh
   nvcc ... -DGLASS_DEFAULTS_TABLE_LOCAL='"bench/tuning/<host>_defaults.cuh"' ...

``bench/explore_sweep.ipynb`` visualizes a sweep (ladder plot + winner table);
:doc:`../user_guide/tutorials/sweep_results` shows the rendered ladder + winner
table the defaults are seeded from.
