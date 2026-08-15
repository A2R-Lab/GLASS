API Reference
=============

These pages are generated directly from the Doxygen doc-comments in the GLASS
headers via `Breathe <https://breathe.readthedocs.io/>`_. Only the public,
documented entry points appear here — internal ``*_impl`` helpers are
intentionally excluded.

The reference is organized by BLAS level and by backend:

* **L1** — vector operations (axpy, copy, dot, reduce, norms, elementwise, …).
* **L2** — matrix-vector operations (gemv, ger, trsv, trmv, strided/segmented gemv).
* **L3** — matrix operations (gemm and variants, inverse, Cholesky, trsm, syrk/syr2k, ldlt, posv/potrs).
* **NVIDIA backend** — the ``glass::nvidia::block::`` CUB / cuBLASDx /
  cuSOLVERDx paths and their host-side query/size helpers, plus the
  ``glass::nvidia::warp::`` CUB ``WarpReduce`` reductions (one full 32-lane
  warp per problem).
* **Warp-scoped** — the ``glass::warp::`` single-warp SIMT variants for
  warp-per-problem kernels.
* **Thread-scoped** — the ``glass::thread::`` sequential variants for
  thread-per-problem low-DOF packing.
* **Block-tridiagonal solvers** — the ``glass::bdmv`` matvec and the
  ``glass::pcg`` preconditioned conjugate-gradient solver for the
  block-tridiagonal SPD systems of trajectory optimization / MPC.
* **Robotics operators** — the spatial 6-D, Lie/quaternion, projection/cone,
  geometry-distance, and sampling-reduction families (all three SIMT tiers;
  see :doc:`../user_guide/concepts/robotics_conventions`).
* **Backend picker** — ``glass-defaults.cuh`` ``constexpr`` helpers
  (``suggested_backend`` / ``suggested_block_threads`` / ``suggested_warps_per_block`` / ``suggested_threads_per_block``)
  that pick a backend + launch config from the measured ladder.

.. note::

   Every operation typically ships several overloads — runtime-sized and
   compile-time-sized (``<T, N, ...>``), with and without a ``beta`` term, and
   pure-SIMT vs cooperative-groups (``glass::cgrps::``) variants. The pages
   below list them per header.

.. note::

   The block-scope SIMT entries on these pages are the ``glass::block::``
   **contract tier** (bit-exact, thread-count invariant for
   deterministic-order ops; never re-dispatched);
   the bare ``glass::`` spellings are the **measured-default face**. Measured
   cells can use a warp-0 or thread-0 body while preserving the block-scope
   calling contract; unmoved operations remain re-exports. See
   :doc:`../user_guide/concepts/namespaces`.

.. toctree::
   :maxdepth: 2

   l1
   l2
   l3
   nvidia
   warp
   thread
   banded
   pcg
   robotics
   defaults
