Worked Examples
===============

Minimal, self-contained **compile-and-run** programs — one concept each. Every
file is a complete program: a ``__global__`` kernel that calls a GLASS device
function, plus a ``main`` that allocates device memory, launches **one block**
(or a batch of blocks for the batched demos), copies the result back, and
verifies it — examples return non-zero on a numeric mismatch, and the test
suite compiles and runs every one of them on hardware. The sources live in
``examples/`` in the repository; ``examples/README.md`` carries the same table
plus build instructions (keep the two in sync).

All examples are pure SIMT (no external dependencies) except ``05_nvidia_gemm``,
which requires MathDx.

.. list-table::
   :header-rows: 1
   :widths: 26 56 18

   * - Example
     - Shows
     - Backend / deps
   * - ``01_axpy_simt``
     - L1 vector op ``axpy`` (``y = αx + y``), runtime size
     - pure SIMT
   * - ``02_gemm_conventions``
     - THE GEMM example: standard-BLAS convention, both size overloads, all
       four transpose combos, row-major-is-a-transpose (bit-identical), and
       the ``glass::cgrps::`` spelling
     - pure SIMT
   * - ``03_reductions_norms``
     - ``reduce`` / warp-shuffle ``reduce_fast`` (+ scratch sizing) and the
       ``nrm2`` family across block + warp tiers
     - pure SIMT
   * - ``04_gemm_dispatch``
     - ``glass::gemm_dispatch`` auto-tiling + the ``glass_gemm_dispatch_smem``
       host helper
     - pure SIMT
   * - ``05_nvidia_gemm``
     - the cuBLASDx-backed ``glass::nvidia::block::gemm`` path
     - **requires MathDx**
   * - ``06_warp_ops``
     - single-warp ``glass::warp::`` ops (reduce, 4×4 gemm, potrf+trsv),
       launched ``<<<1,32>>>``
     - pure SIMT
   * - ``07_pcg_solve``
     - block-tridiagonal PCG solve ``glass::pcg`` (``[L|D|R]`` strips)
     - pure SIMT
   * - ``08_backend_picker``
     - ``recommend<>`` driving a warp/block/thread ``posv`` launch
     - pure SIMT
   * - ``09_gemm_strided``
     - GEMM on sub-blocks with explicit leading dims (``gemm_strided``)
     - pure SIMT
   * - ``10_ldlt_solve``
     - symmetric-indefinite ``ldlt`` + ``ldlt_solve``, the ``CHECK`` fail flag
       and inertia
     - pure SIMT
   * - ``11_riccati_gain``
     - the fused LQR gain ``(R+BᵀPB)⁻¹(BᵀPA)`` in one call
     - pure SIMT
   * - ``12_inv``
     - augmented ``[A|I]`` ``inv``, and ``inv_pivoted`` recovering a zero
       leading pivot
     - pure SIMT
   * - ``13_thread_pack``
     - the ``glass::thread::`` tier — 32 N=6 SPD solves packed per warp
     - pure SIMT
   * - ``14_spatial_dynamics``
     - Featherstone ``motion_cross_mul``/``force_cross_mul``, fused vs
       materialize-and-``gemv``
     - pure SIMT
   * - ``15_floating_base_retract``
     - batched ``se3_retract`` per thread; unit-quaternion invariants
     - pure SIMT
   * - ``16_mppi_weights``
     - MPPI weight update = ``softmax`` + ``argmin``, bit-identical across
       block sizes
     - pure SIMT
   * - ``17_cone_projection``
     - friction-cone AL: ``soc_project`` + the AL value/derivative chain
     - pure SIMT
   * - ``18_collision_spheres``
     - narrow-phase ``transform_sphere`` → ``sphere_box_dist`` →
       ``smooth_hinge`` (+ gradient)
     - pure SIMT
   * - ``19_best_fit_rotation``
     - batched Wahba/Kabsch via ``thread::closest_rotation``
     - pure SIMT

Build and run
-------------

.. code-block:: bash

   cd examples
   make -j ARCH=sm_120                    # auto-detects ARCH if omitted
   make run                               # runs every built example
   # 05_nvidia_gemm builds only when MATHDX_ROOT is set

Sources
-------

01_axpy_simt
~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/01_axpy_simt.cu
   :language: cuda

02_gemm_conventions
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/02_gemm_conventions.cu
   :language: cuda

03_reductions_norms
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/03_reductions_norms.cu
   :language: cuda

04_gemm_dispatch
~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/04_gemm_dispatch.cu
   :language: cuda

05_nvidia_gemm
~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/05_nvidia_gemm.cu
   :language: cuda

06_warp_ops
~~~~~~~~~~~

.. literalinclude:: ../../../../examples/06_warp_ops.cu
   :language: cuda

07_pcg_solve
~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/07_pcg_solve.cu
   :language: cuda

08_backend_picker
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/08_backend_picker.cu
   :language: cuda

09_gemm_strided
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/09_gemm_strided.cu
   :language: cuda

10_ldlt_solve
~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/10_ldlt_solve.cu
   :language: cuda

11_riccati_gain
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/11_riccati_gain.cu
   :language: cuda

12_inv
~~~~~~

.. literalinclude:: ../../../../examples/12_inv.cu
   :language: cuda

13_thread_pack
~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/13_thread_pack.cu
   :language: cuda

14_spatial_dynamics
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/14_spatial_dynamics.cu
   :language: cuda

15_floating_base_retract
~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/15_floating_base_retract.cu
   :language: cuda

16_mppi_weights
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/16_mppi_weights.cu
   :language: cuda

17_cone_projection
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/17_cone_projection.cu
   :language: cuda

18_collision_spheres
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/18_collision_spheres.cu
   :language: cuda

19_best_fit_rotation
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../../examples/19_best_fit_rotation.cu
   :language: cuda
