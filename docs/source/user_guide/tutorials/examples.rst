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
     - block reduction: ``glass::reduce`` and warp-shuffle ``glass::reduce_fast``
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
   * - ``10_gemm_basics``
     - the standard-BLAS GEMM convention (``C`` is M×N, contraction K) and all
       four ``TRANSPOSE_A`` / ``TRANSPOSE_B`` combos, at a non-square shape
     - pure SIMT
   * - ``11_rowmajor_is_transpose``
     - "row-major is just a transpose" — a row-major operand read with
       ``TRANSPOSE_*`` gives bit-identical output
     - pure SIMT
   * - ``12_nrm2``
     - Euclidean norm ``nrm2`` (BLAS name), block + warp forms
     - pure SIMT
   * - ``13_gemm_strided``
     - ``gemm_strided`` — GEMM on column-major sub-blocks with explicit
       leading dims
     - pure SIMT
   * - ``14_ldlt_solve``
     - symmetric-**indefinite** solve: ``ldlt`` + ``ldlt_solve``, plus the
       ``CHECK=true`` failure-flag + inertia reporting path
     - pure SIMT
   * - ``15_riccati_gain``
     - LQR feedback gain ``K = (R + BᵀPB)⁻¹(BᵀPA)`` via ``riccati_gain`` with
       ``riccati_scratch_bytes`` dynamic smem
     - pure SIMT
   * - ``16_inv``
     - matrix inversion on the augmented ``[A | I]`` layout: ``inv``, and the
       robust ``inv_pivoted`` on a zero leading pivot
     - pure SIMT

Building
--------

All examples ``#include`` the GLASS headers from the repo root, so build with
``-I..`` from inside ``examples/``. Pick the ``-arch`` that matches your GPU
(``sm_75`` Turing, ``sm_86`` Ampere, ``sm_89`` Ada, ``sm_120`` Blackwell, …):

.. code-block:: bash

   cd examples
   nvcc -std=c++17 -arch=sm_75 -I.. 01_axpy_simt.cu -o axpy && ./axpy

Examples 01–05 and 07–16 are pure SIMT (plain ``nvcc``, no external libraries).
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

10 — GEMM basics (BLAS convention + transpose flags)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/10_gemm_basics.cu
   :language: cuda

11 — row-major is just a transpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/11_rowmajor_is_transpose.cu
   :language: cuda

12 — nrm2 (Euclidean norm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/12_nrm2.cu
   :language: cuda

13 — gemm_strided (sub-block GEMM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/13_gemm_strided.cu
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

14 — symmetric-indefinite LDLᵀ solve (+ CHECK / inertia)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/14_ldlt_solve.cu
   :language: cuda

15 — LQR feedback gain (riccati_gain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/15_riccati_gain.cu
   :language: cuda

16 — matrix inversion (inv / inv_pivoted, augmented [A | I])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/16_inv.cu
   :language: cuda

17 — thread-tier packing (32 low-DOF problems per warp)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/17_thread_pack.cu
   :language: cuda

18 — spatial dynamics (6-D cross products, transforms, inertia)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/18_spatial_dynamics.cu
   :language: cuda

19 — floating-base retract (SE(3)/quaternion Lie ops)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/19_floating_base_retract.cu
   :language: cuda

20 — MPPI weights (softmax over rollout costs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/20_mppi_weights.cu
   :language: cuda

21 — cone projection (AL / friction-cone operators)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/21_cone_projection.cu
   :language: cuda

22 — collision spheres (pairwise distance checks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/22_collision_spheres.cu
   :language: cuda

23 — best-fit rotation (eig3 / svd3 / closest_rotation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/23_best_fit_rotation.cu
   :language: cuda

See also :doc:`sweep_results` for the measured ladder behind the picker, and
:doc:`../../api_reference/defaults` for the picker helper reference.
