Concepts
========

These pages explain the cross-cutting ideas that show up across the GLASS API:
how the ``glass::nvidia::`` wrappers decide between cuBLASDx and pure-SIMT, what
the ``TRAILING_SYNC`` template parameter controls, how to autotune the dispatch
for your own hardware, and the batched-1D GEMM APIs designed for kernels with a
single 1D thread block.

.. toctree::
   :maxdepth: 1

   backend_dispatch
   trailing_sync
   tuning
   batched_1d
