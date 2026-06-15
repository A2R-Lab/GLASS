GLASS: GPU Linear Algebra for Single-block Systems
==================================================

GLASS is a header-only CUDA library of BLAS/LAPACK-style ``__device__`` routines
designed to run **inside a single CUDA thread block**. You launch one block per
independent problem; the block's threads cooperate over data already resident in
shared or global memory. It is the linear-algebra layer underneath
`GRiD <https://github.com/A2R-Lab/GRiD>`_ and other A2R Lab GPU solvers.

Three namespaces, one mental model
----------------------------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: ``glass::``
      :link: user_guide/getting_started/library_overview
      :link-type: doc

      Hand-rolled, pure-SIMT (``threadIdx``/``blockDim``) primitives with
      runtime- and compile-time-sized overloads. **No dependencies** — just
      ``#include "glass.cuh"``.

   .. grid-item-card:: ``glass::cgrps::``
      :link: user_guide/getting_started/library_overview
      :link-type: doc

      The same surface expressed with cooperative groups
      (``g.thread_rank()`` / ``g.size()``). ``#include "glass-cgrps.cuh"``.

   .. grid-item-card:: ``glass::nvidia::``
      :link: user_guide/concepts/backend_dispatch
      :link-type: doc

      Vendor-accelerated paths (CUB, cuBLASDx, cuSOLVERDx) that auto-dispatch
      between SIMT and the vendor backend by size. Needs NVIDIA MathDx.

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
