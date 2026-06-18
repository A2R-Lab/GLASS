Quickstart
==========

A minimal end-to-end example: include the umbrella header, write a kernel that
calls a GLASS function, and launch one block per data item.

The kernel
----------

.. code-block:: cpp

   #include "glass.cuh"

   __global__ void my_kernel(float* A, float* B, float* C, int m, int n, int k) {
       // Runtime size: all threads in the block cooperate
       glass::gemm(m, n, k, 1.f, A, B, 0.f, C);
   }

Launch with one block per data item:

.. code-block:: cpp

   my_kernel<<<num_items, 256>>>(A, B, C, m, n, k);

That's the whole contract: every GLASS function assumes it runs inside **one
CUDA block**, and you launch one block per independent problem.

Compiling
---------

The pure-SIMT path is header-only — just add the repository root to your include
path:

.. code-block:: bash

   nvcc -std=c++17 -I /path/to/GLASS -arch=sm_86 my_kernel.cu -o my_kernel

Compile-time sizes
------------------

Passing the sizes as template arguments lets the compiler unroll the inner
loops — the best choice for small fixed-size matrices:

.. code-block:: cpp

   #include "glass.cuh"

   __global__ void k(float* A, float* B, float* C) {
       // Sizes baked in as template params — compiler can unroll loops
       glass::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);
       glass::gemv<float, 6, 6>(1.f, A, B, 0.f, C);
       glass::axpy<float, 36>(1.5f, A, B);
   }

A few more vector/matrix calls (runtime sizes shown):

.. code-block:: cpp

   glass::gemm(m, n, k, 1.f, A, B, 0.f, C);   // C = alpha*A*B + beta*C
   glass::gemv(m, n, 1.f, A, x, 0.f, y);       // y = alpha*A*x + beta*y
   glass::axpy(n, 1.5f, x, y);                 // y = alpha*x + y

.. note::

   Matrices default to **column-major** (Fortran) order, consistent with cuBLAS.
   The pure-SIMT ``glass::`` API takes a ``ROW_MAJOR=true`` template parameter
   to switch storage order.

Next steps
----------

* :doc:`using_nvidia_backend` — route to cuBLASDx / cuSOLVERDx for larger
  shapes.
* :doc:`../getting_started/library_overview` — the call surfaces (block-scoped
  backends + warp-scoped) and when to use each.
* :doc:`../concepts/index` — backend dispatch, ``TRAILING_SYNC``, tuning, and
  batched-1D APIs.
