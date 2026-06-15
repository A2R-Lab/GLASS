Using the NVIDIA Backend
========================

The ``glass::nvidia::`` namespace routes to NVIDIA's device-side libraries —
CUB (L1), cuBLASDx (L2/L3 GEMM/GEMV/batched), and cuSOLVERDx (LAPACK) — while
preserving the same one-block ``__device__`` calling convention. These wrappers
require **compile-time** matrix sizes.

Make sure MathDx is installed and ``MATHDX_ROOT`` is set first — see
:doc:`../getting_started/installation`.

Compiling against MathDx
------------------------

cuBLASDx is header-only; cuSOLVERDx links a precompiled device fatbin. The flags
differ by which level you use.

**cuBLASDx (GEMM / GEMV / batched):**

.. code-block:: bash

   nvcc -std=c++17 -arch=sm_86 \
        -I /path/to/GLASS \
        -I $MATHDX_ROOT/include \
        -I $MATHDX_ROOT/external/cutlass/include \
        --expt-relaxed-constexpr \
        -Xptxas -O1 \
        -DGLASS_BENCH_CUBLASDX \
        my_kernel.cu -o my_kernel

* ``--expt-relaxed-constexpr`` is **required** for cuBLASDx headers.
* ``-Xptxas -O1`` works around the CUDA 12.9 cuBLASDx miscompilation (and
  doubles as anti-DSE).
* ``-DGLASS_BENCH_CUBLASDX`` force-enables the wrappers when the TU hasn't
  pre-included the cuBLASDx header. (GLASS otherwise auto-detects via
  ``GLASS_HAVE_CUBLASDX`` from include order.)

**cuSOLVERDx (LAPACK) additionally needs the device-link flags:**

.. code-block:: bash

   nvcc -std=c++17 -arch=sm_86 \
        -I /path/to/GLASS -I $MATHDX_ROOT/include \
        --expt-relaxed-constexpr -Xptxas -O1 \
        -rdc=true -dlto \
        -L $MATHDX_ROOT/lib \
        -lcusolverdx -lcublas -lcusolver -lcudart \
        my_kernel.cu -o my_kernel

``SMS`` defaults to ``860`` and can be overridden with ``-DSMS=XXX`` so the
dispatch heuristic and cuBLASDx code-gen target your arch.

Calling ``glass::nvidia::`` — default form
------------------------------------------

In the default form cuBLASDx picks the thread count; you query the required
shared memory and thread count (both ``constexpr``) on the host and launch with
the **exact** thread count — a mismatch deadlocks.

.. code-block:: cpp

   #include "glass-nvidia.cuh"

   constexpr auto smem    = glass::nvidia::gemm_smem_size<float, 6, 6, 6>();
   constexpr auto threads = glass::nvidia::gemm_threads<float, 6, 6, 6>();

   __global__ void k(float* A, float* B, float* C) {
       extern __shared__ __align__(16) char smem_buf[];
       glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, smem_buf);
   }

   // Launch with the EXACT thread count cuBLASDx wants.
   k<<<1, threads, smem>>>(dA, dB, dC);

Caller-pinned ``BlockDim<TC>``
------------------------------

To launch with a thread count your *surrounding* kernel needs (e.g. GRiD's
352-thread launches), pin ``BlockDim<TC>`` with a DEFINE macro. Extra threads go
idle inside the GEMM:

.. code-block:: cpp

   namespace glass { namespace nvidia {
       DEFINE_NVIDIA_GEMM_BLOCKDIM(6, 6, 6, 352)   // pin BlockDim<352,1,1>
   }}

   __global__ void k(float* A, float* B, float* C) {
       extern __shared__ __align__(16) char smem_buf[];
       glass::nvidia::gemm<float, 6, 6, 6, 352>(1.f, A, B, 0.f, C, smem_buf);
   }

   constexpr auto smem = glass::nvidia::gemm_smem_size<float, 6, 6, 6, 352>();
   k<<<1, 352, smem>>>(dA, dB, dC);

Query the smallest valid ``TC`` for a ``(T, M, N, K, SM)`` tuple:

.. code-block:: cpp

   constexpr uint32_t MIN = glass::nvidia::gemm_min_block_threads<float, 6, 6, 6>();
   static_assert(glass::nvidia::gemm_block_threads_valid<float, 6, 6, 6, 352>(),
                 "352 threads should be enough for 6x6x6 on this SM");

.. tip::

   Compile **without** ``-DNDEBUG`` and the wrappers ``assert(blockDim >=
   GEMM::block_dim)`` inside every ``run()`` — a misconfigured launch fails with
   a clean assertion instead of a silent deadlock. The asserts compile out under
   ``-DNDEBUG``.

Layout / transpose
------------------

``glass::nvidia::gemm`` accepts ``layout LA``, ``LB``, ``LC`` template
parameters (mirroring cuBLASDx's ``Arrangement<>``) so you can express transpose
/ row-major storage without falling back to SIMT:

.. code-block:: cpp

   namespace glass { namespace nvidia {
       // A · Bᵀ  (B row-major); alias for LB=row_major
       DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB(6, 6, 6, 352)
       // Or fully explicit (LA=row, LB=col, LC=col):
       DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT(6, 6, 6, 352, 1, 0, 0)
   }}

Layout arguments are integer literals: ``0`` = ``col_major``, ``1`` =
``row_major``.

Adding a custom size
--------------------

Pre-instantiated square GEMM/GEMV sizes are ``4, 6, 8, 12, 14, 24, 64``. For any
other size, place the DEFINE macro inside ``namespace glass::nvidia`` in your
``.cu`` file:

.. code-block:: cpp

   #include "glass-nvidia.cuh"
   namespace glass { namespace nvidia {
       DEFINE_NVIDIA_GEMM(16, 16, 16)
       DEFINE_NVIDIA_GEMV(16, 16)
       DEFINE_NVIDIA_GEMM_BLOCKDIM(16, 16, 16, 256)
       DEFINE_NVIDIA_POSV_BLOCKDIM(16, 4, 256)   // SPD solve, 4 RHS
   }}

Every wrapper exposes the same macro family — substitute ``GEMM`` / ``GEMV`` /
``GEMM_BATCHED`` / ``CHOL`` / ``TRSM`` / ``POSV`` / ``POTRS`` / ``GETRF`` /
``GETRS`` / ``GESV`` / ``GEQRF`` / ``GELS`` and append ``_BLOCKDIM`` /
``_LAYOUT`` / ``_SM`` / ``_BLOCKDIM_SM`` etc. as needed.

Linear solvers (cuSOLVERDx)
---------------------------

.. code-block:: cpp

   #include "glass-nvidia.cuh"

   namespace glass { namespace nvidia {
       DEFINE_NVIDIA_POSV_BLOCKDIM(7, 1, 256)   // 7×7 SPD, 1 RHS, BlockDim<256>
   }}

   __global__ void k(float* A, float* b) {
       extern __shared__ __align__(16) char smem_buf[];
       // Solves A·x = b in place: A := L (lower Cholesky), b := x
       glass::nvidia::posv<float, 7, 1, 256>(A, b, smem_buf);
   }

   constexpr auto smem = glass::nvidia::posv_smem_size<float, 7, 1, 256>();
   k<<<1, 256, smem>>>(dA, db);

Available cuSOLVERDx wrappers: ``chol_inplace``, ``trsm``, ``posv``, ``potrs``,
``getrf_no_pivot``, ``getrs_no_pivot``, ``gesv_no_pivot``, ``geqrf``, ``gels``.
All follow the same ``DEFINE_NVIDIA_<NAME>`` macro pattern and are **not**
pre-instantiated — call the macro per size you need.

See :doc:`../concepts/backend_dispatch` for how the auto-dispatch decides
between cuBLASDx and SIMT, and :doc:`../concepts/batched_1d` for the 1D-launch
batched GEMM APIs.
