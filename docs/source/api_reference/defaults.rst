Backend Picker (``glass-defaults.cuh``)
=======================================

Queryable backend-selection defaults — the measured warp / block / nvidia ladder
(``bench/MEGA_SWEEP_RESULTS.md``) exposed as ``constexpr`` helpers, so callers and
GRiD-style codegen pick a backend + launch config instead of hand-copying a table.

The pick **cannot** be a device function: warp, block, and ``nvidia`` need different
``<<<grid, block>>>`` launches, so the decision happens host-side / at codegen time. See
:doc:`../user_guide/concepts/tuning` for the underlying numbers.

Include order
-------------

Include ``glass-defaults.cuh`` **after** ``glass.cuh``, and after ``glass-nvidia.cuh`` if you
want the ``nvidia`` tier to be eligible (it reads ``GLASS_HAVE_CUBLASDX`` /
``GLASS_HAVE_CUSOLVERDX``). With only ``glass.cuh`` linked, the ``nvidia`` tier collapses to its
warp/block runner-up, so a no-MathDx caller always gets a backend it can launch.

Helpers
-------

.. code-block:: cuda

   enum class glass::op      { dot, gemv, gemm, chol, trsv, posv };
   enum class glass::backend { warp, block, nvidia };

   // Which backend for (op, N, T) on this SM? (nvidia only when the vendor lib is linked)
   template <op Op, uint32_t N, typename T, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr backend  glass::suggested_backend();

   // For the `block` backend: factor/solve want 32; gemm grows with N; dot/gemv 64–128.
   template <op Op, uint32_t N, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr uint32_t glass::suggested_block_threads();

   // For the `warp` backend: dot packs 8; others 2 warps/block.
   template <op Op, uint32_t N = 0, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
   constexpr uint32_t glass::suggested_warps_per_block();

Example
-------

.. code-block:: cuda

   #include "glass.cuh"
   #include "glass-defaults.cuh"   // (after glass-nvidia.cuh too, to allow the nvidia tier)

   constexpr auto be = glass::suggested_backend<glass::op::chol, N, float>();
   if      constexpr (be == glass::backend::nvidia) { /* cuSOLVERDx launch */ }
   else if constexpr (be == glass::backend::warp)   { /* <<<ceil(P/WPB), {32,WPB}>>> */ }
   else /* block */ {
       constexpr int TB = glass::suggested_block_threads<glass::op::chol, N, float>();
       /* <<<P, TB>>> */
   }

A runnable version is ``examples/09_backend_picker.cu``.

Per-host override
-----------------

Numbers are seeded from RTX 5090 / sm_120. For another GPU, regenerate a per-host table from a
sweep run and point ``GLASS_DEFAULTS_TABLE_LOCAL`` at it:

.. code-block:: bash

   cd bench && ./run_mega_sweep.sh sm_XX
   python3 autotune.py --emit-defaults mega_sweep_<ts>.txt        # -> bench/tuning/<host>_defaults.cuh
   nvcc ... -DGLASS_DEFAULTS_TABLE_LOCAL='"bench/tuning/<host>_defaults.cuh"' ...

``bench/explore_sweep.ipynb`` visualizes a sweep (ladder plot + winner table);
:doc:`../user_guide/tutorials/sweep_results` shows the rendered ladder + winner
table the defaults are seeded from.
