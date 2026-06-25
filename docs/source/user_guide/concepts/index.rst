Concepts
========

These pages explain the cross-cutting ideas that show up across the GLASS API:
how the ``glass::nvidia::`` wrappers decide between cuBLASDx and pure-SIMT, what
the ``TRAILING_SYNC`` template parameter controls, how to autotune the dispatch
for your own hardware, the batched-1D GEMM APIs designed for kernels with a
single 1D thread block, and the block-tridiagonal layout used by the
``glass::bdmv`` / ``glass::pcg`` solvers.

.. toctree::
   :maxdepth: 1

   backend_dispatch
   contraction_parallel
   namespaces
   trailing_sync
   tuning
   batched_1d
   block_tridiagonal
