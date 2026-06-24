Worked Examples
===============

Minimal, self-contained **compile-and-run** programs — one concept each. Every
file is a complete program: a ``__global__`` kernel that calls a GLASS device
function, plus a ``main`` that allocates device memory, launches **one block**,
copies the result back, and prints it. They are deliberately tiny (~30–60 lines)
— the point is clarity, not features. The sources live in ``examples/`` in the
repository.

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Example
     - Shows
     - Backend / deps
   * - ``01_axpy_simt``
     - L1 vector op ``axpy`` (``y = αx + y``), runtime size
     - pure SIMT
   * - ``02_gemm``
     - ``gemm`` (``C = αAB + βC``), runtime + compile-time overloads, column-major
     - pure SIMT
   * - ``03_reduce``
     - block reduction: ``glass::reduce`` and warp-shuffle ``glass::high_speed::reduce``
     - pure SIMT
   * - ``04_cgrps``
     - the cooperative-groups variant ``glass::cgrps::gemm``
     - pure SIMT
   * - ``05_gemm_dispatch``
     - ``glass::gemm_dispatch`` + dynamic shared memory helper
     - pure SIMT
   * - ``06_nvidia_gemm``
     - the cuBLASDx-backed ``glass::nvidia::gemm`` path
     - **requires MathDx**
   * - ``07_warp_ops``
     - single-warp ``glass::warp::`` ops, launched ``<<<1,32>>>``
     - pure SIMT
   * - ``08_pcg_solve``
     - block-tridiagonal PCG solve ``glass::pcg``
     - pure SIMT
   * - ``09_backend_picker``
     - pick a backend + launch config with ``glass-defaults.cuh``, then dispatch
     - pure SIMT

Building
--------

All examples ``#include`` the GLASS headers from the repo root, so build with
``-I..`` from inside ``examples/``. Pick the ``-arch`` that matches your GPU
(``sm_75`` Turing, ``sm_86`` Ampere, ``sm_89`` Ada, ``sm_120`` Blackwell, …):

.. code-block:: bash

   cd examples
   nvcc -std=c++17 -arch=sm_75 -I.. 01_axpy_simt.cu -o axpy && ./axpy

Examples 01–05, 07, 08, 09 are pure SIMT (plain ``nvcc``, no external libraries).
Only ``06_nvidia_gemm`` needs MathDx — see :doc:`using_nvidia_backend` and
:doc:`../getting_started/installation` for the cuBLASDx flags.

.. note::

   **One block per data item.** Every GLASS function runs inside a single CUDA
   block; these examples all launch ``<<<1, threads>>>``. To process many
   independent items, launch one block each (``<<<num_items, threads>>>``).
   Matrices are **column-major** by default (``A[row + col*m]``), matching cuBLAS.

L1 / L2 / L3 device ops
-----------------------

01 — axpy (L1, runtime size)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/01_axpy_simt.cu
   :language: cuda

02 — gemm (runtime + compile-time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/02_gemm.cu
   :language: cuda

03 — reduce
~~~~~~~~~~~

.. literalinclude:: ../../../../examples/03_reduce.cu
   :language: cuda

Backends
--------

04 — cooperative-groups variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/04_cgrps.cu
   :language: cuda

05 — gemm dispatch + dynamic smem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/05_gemm_dispatch.cu
   :language: cuda

06 — NVIDIA / cuBLASDx (needs MathDx)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/06_nvidia_gemm.cu
   :language: cuda

07 — warp-per-problem ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/07_warp_ops.cu
   :language: cuda

Solvers & dispatch
------------------

08 — block-tridiagonal PCG solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/08_pcg_solve.cu
   :language: cuda

09 — backend picker
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/09_backend_picker.cu
   :language: cuda

See also :doc:`sweep_results` for the measured ladder behind the picker, and
:doc:`../../api_reference/defaults` for the picker helper reference.
