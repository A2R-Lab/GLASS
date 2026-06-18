User Guide
==========

**GLASS** — *GPU Linear Algebra Simple Subroutines*. Composable ``__device__``
BLAS/LAPACK-style subroutines that run inside a single CUDA block, now expanding
to warp-level primitives. This guide walks you from installation through the core
concepts and hands-on tutorials.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: getting_started/installation
      :link-type: doc

      Header-only include model, CUDA toolkit requirement, and optional MathDx
      setup for the ``glass::nvidia::`` backend.

   .. grid-item-card:: Library Overview
      :link: getting_started/library_overview
      :link-type: doc

      What GLASS is, the single-block execution model, and the four call
      surfaces — with a guide to choosing the right backend.

   .. grid-item-card:: Concepts
      :link: concepts/index
      :link-type: doc

      Backend dispatch, ``TRAILING_SYNC`` semantics, per-host tuning, and the
      batched-1D GEMM APIs.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      A minimal quickstart kernel, compiling against the NVIDIA backend, running
      the tests, and running the benchmarks.
