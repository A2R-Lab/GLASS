GLASS: GPU Linear Algebra Simple Subroutines
============================================

**Composable** ``__device__`` **BLAS/LAPACK-style subroutines that run inside a
single CUDA block. Now expanding to warp-level primitives for packing many small
problems into one block.**

GLASS is a header-only CUDA library of BLAS/LAPACK-style ``__device__`` routines.
You launch one block per independent problem; the block's threads cooperate over
data already resident in shared or global memory. It is the linear-algebra layer
underneath `GRiD <https://github.com/A2R-Lab/GRiD>`_ and other A2R Lab GPU
solvers.

Call surfaces
-------------

GLASS primitives are **block-scoped** by default (one block per problem) in three
numerically-interchangeable backends, plus a **warp-scoped** surface for kernels
that pack many small independent problems into one block — one per warp:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ``glass::`` — block, SIMT
      :link: user_guide/getting_started/library_overview
      :link-type: doc

      Hand-rolled pure-SIMT (``threadIdx``/``blockDim``), runtime- and
      compile-time-sized. **No dependencies** — ``#include "glass.cuh"``.

   .. grid-item-card:: ``glass::cgrps::`` — block, coop groups
      :link: user_guide/getting_started/library_overview
      :link-type: doc

      The same surface via cooperative groups (``g.thread_rank()`` /
      ``g.size()``). ``#include "glass-cgrps.cuh"``.

   .. grid-item-card:: ``glass::nvidia::`` — block, vendor
      :link: user_guide/concepts/backend_dispatch
      :link-type: doc

      CUB / cuBLASDx / cuSOLVERDx, auto-dispatched against SIMT by size.
      Needs NVIDIA MathDx.

   .. grid-item-card:: ``glass::warp::`` — warp-per-problem
      :link: api_reference/warp
      :link-type: doc

      Single-warp SIMT variants of selected L1/L3 ops (``reduce``, ``gemm``,
      ``cholDecomp_InPlace``, ``trsm``) via ``__shfl_*_sync`` — no
      ``__syncthreads``, so warps run independently for **intra-block
      parallelism**.

The three block-scoped backends cover the full L1/L2/L3 surface and are
interchangeable (switch by namespace prefix). ``glass::warp::`` is a selected,
warp-scoped set; it requires a full 32-lane warp.

Higher-level solvers
--------------------

Built on those primitives (and likewise single-block): ``glass::bdmv``
(block-tridiagonal matvec) and ``glass::pcg`` (preconditioned conjugate
gradient) for the block-tridiagonal SPD systems of trajectory optimization /
MPC — see :doc:`user_guide/concepts/block_tridiagonal`.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Get started
      :link: user_guide/getting_started/installation
      :link-type: doc

      Header-only install, the single-block execution model, and an optional
      MathDx setup for the ``glass::nvidia::`` backend.

   .. grid-item-card:: API reference
      :link: api_reference/index
      :link-type: doc

      The L1 / L2 / L3 and NVIDIA device functions, generated from the header
      doc-comments via Doxygen + Breathe.

Quick start
-----------

.. code-block:: cpp

   #include "glass.cuh"

   // One block solves one problem; threads stride over the data.
   __global__ void saxpy_kernel(uint32_t n, float a, float *x, float *y) {
       glass::axpy(n, a, x, y);          // y = a*x + y
   }

   saxpy_kernel<<<1, 256>>>(n, 2.0f, d_x, d_y);

See :doc:`user_guide/tutorials/quickstart` for a complete, compilable example,
and the :doc:`examples in the repository <user_guide/tutorials/quickstart>`.

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/landing_page
   user_guide/getting_started/installation
   user_guide/getting_started/library_overview

.. toctree::
   :hidden:
   :caption: Concepts

   user_guide/concepts/index

.. toctree::
   :hidden:
   :caption: Tutorials

   user_guide/tutorials/index

.. toctree::
   :hidden:
   :caption: API Reference

   api_reference/index

.. toctree::
   :hidden:
   :caption: Project info

   contribution_guidelines
   sphinx_edit_guide
