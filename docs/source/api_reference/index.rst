API Reference
=============

These pages are generated directly from the Doxygen doc-comments in the GLASS
headers via `Breathe <https://breathe.readthedocs.io/>`_. Only the public,
documented entry points appear here — internal ``*_impl`` helpers are
intentionally excluded.

The reference is organized by BLAS level and by backend:

* **L1** — vector operations (axpy, copy, dot, reduce, norms, elementwise, …).
* **L2** — matrix-vector operations (gemv, ger, strided/segmented gemv).
* **L3** — matrix operations (gemm and variants, inverse, Cholesky, trsm).
* **NVIDIA backend** — the ``glass::nvidia::`` CUB / cuBLASDx / cuSOLVERDx paths
  and their host-side query/size helpers.
* **Warp-scoped** — the ``glass::warp::`` single-warp SIMT variants for
  warp-per-problem kernels.
* **Block-tridiagonal solvers** — the ``glass::banded::`` matvec and the
  ``glass::pcg::`` preconditioned conjugate-gradient solver for the
  block-tridiagonal SPD systems of trajectory optimization / MPC.

.. note::

   Every operation typically ships several overloads — runtime-sized and
   compile-time-sized (``<T, N, ...>``), with and without a ``beta`` term, and
   pure-SIMT vs cooperative-groups (``glass::cgrps::``) variants. The pages
   below list them per header.

.. toctree::
   :maxdepth: 2

   l1
   l2
   l3
   nvidia
   warp
   banded
   pcg
