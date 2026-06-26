NVIDIA Backend (``glass::nvidia::``)
====================================

Vendor-accelerated paths built on CUB (reductions), cuBLASDx (GEMM/GEMV), and
cuSOLVERDx (LAPACK). The entry points auto-dispatch between a pure-SIMT
implementation and the vendor backend based on a size heuristic / tuning table;
see :doc:`../user_guide/concepts/backend_dispatch`. The L2/L3/LAPACK paths
require NVIDIA MathDx (``MATHDX_ROOT``) — see
:doc:`../user_guide/getting_started/installation`.

Each call has a companion **host-side** query helper (``*_scratch_bytes``,
``*_threads``, ``*_block_threads_valid``) used to size the launch.

L1 (CUB-backed reductions)
--------------------------

.. doxygenfile:: src/nvidia/l1.cuh

L2 (gemv)
---------

.. doxygenfile:: src/nvidia/l2.cuh

L3 (gemm)
---------

.. doxygenfile:: src/nvidia/l3.cuh

L3 SIMT batched (no cuBLASDx)
-----------------------------

.. doxygenfile:: src/nvidia/l3_simt.cuh

LAPACK (cuSOLVERDx)
-------------------

.. doxygenfile:: src/nvidia/lapack.cuh

Dispatch & query helpers
------------------------

.. doxygenfile:: src/nvidia/query.cuh

.. doxygenfile:: src/nvidia/query_simt.cuh

.. doxygenfile:: src/nvidia/types.cuh

Host helpers
------------

The umbrella headers also expose host-callable helpers for sizing dynamic
shared memory at launch time.

.. doxygenfile:: glass.cuh

.. doxygenfile:: glass-nvidia.cuh
