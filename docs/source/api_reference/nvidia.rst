NVIDIA Backend (``glass::nvidia::``)
====================================

Vendor-accelerated paths built on CUB (reductions), cuBLASDx (GEMM/GEMV), and
cuSOLVERDx (LAPACK). The block-scope ops live in ``glass::nvidia::block::``
(the contract tier; bare ``glass::nvidia::`` re-exports them as the
measured-default face). The entry points auto-dispatch **at compile time**
between a pure-SIMT implementation and the vendor backend based on a size
heuristic / tuning table (a ``constexpr`` decision — no runtime branching);
see :doc:`../user_guide/concepts/backend_dispatch`. The L2/L3/LAPACK paths
require NVIDIA MathDx (``MATHDX_ROOT``) — see
:doc:`../user_guide/getting_started/installation`.

Each call has a companion **host-side** query helper (``*_scratch_bytes``,
``*_threads``, ``*_block_threads_valid``) used to size the launch.

L1 (CUB-backed reductions)
--------------------------

``reduce`` / ``dot`` / ``nrm2`` at block scope (``cub::BlockReduce``) and, in
``glass::nvidia::warp::``, at warp scope (``cub::WarpReduce`` — one FULL
32-lane warp per problem, per-warp scratch via
``warp_reduce_scratch_bytes<T>()``, ``TRAILING_SYNC`` emits ``__syncwarp()``).

.. doxygenfile:: src/nvidia/l1.cuh
   :no-link:

L2 (gemv)
---------

.. doxygenfile:: src/nvidia/l2.cuh
   :no-link:

L3 (gemm)
---------

.. doxygenfile:: src/nvidia/l3.cuh
   :no-link:

L3 SIMT batched (no cuBLASDx)
-----------------------------

.. doxygenfile:: src/nvidia/l3_simt.cuh
   :no-link:

LAPACK (cuSOLVERDx)
-------------------

.. doxygenfile:: src/nvidia/lapack.cuh
   :no-link:

Dispatch & query helpers
------------------------

.. doxygenfile:: src/nvidia/query.cuh
   :no-link:

.. doxygenfile:: src/nvidia/query_simt.cuh
   :no-link:

.. doxygenfile:: src/nvidia/types.cuh
   :no-link:

Host helpers
------------

The umbrella headers also expose host-callable helpers for sizing dynamic
shared memory at launch time.

.. doxygenfunction:: glass_gemm_dispatch_smem

.. doxygenfile:: glass-nvidia.cuh
   :no-link:
