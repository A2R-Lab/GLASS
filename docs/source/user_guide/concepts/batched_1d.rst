Batched-1D GEMM APIs
====================

GLASS provides batched GEMM primitives that run **inside a single 1D thread
block**. They exist because the cuBLASDx-backed ``glass::nvidia::gemm_batched``
requires a **2D launch** (``dim3(TC, BATCH)``) — it gives each batch element a
``threadIdx.y`` slot. Kernels that were launched 1D (``dim3(TC*BATCH, 1, 1)``,
because every other block-level helper uses ``threadIdx.x``) cannot use it
without rewriting their launch geometry.

The batched-1D APIs solve this for the small shapes — ``max(M,N,K) ≲ 8`` — where
cuBLASDx's tile-load overhead dominates and SIMT wins anyway.

Why a SIMT implementation
-------------------------

cuBLASDx's collective operations (``cublasdx::copy``, ``GEMM().execute()``) read
``threadIdx.x`` directly and assume the *whole* block participates; there is no
built-in way to tell cuBLASDx "use only threads ``[b*TC, (b+1)*TC)`` for this
batch element". A clean 1D-launch batched GEMM therefore cannot use cuBLASDx
unchanged.

For the entire small-shape dynamics workload (M,N,K ≤ ~8), cuBLASDx is not
faster than SIMT anyway, so these APIs are **pure SIMT**: a single 1D block of
``TC*BATCH`` threads is partitioned into ``BATCH`` groups of ``TC`` threads, and
each group computes one independent GEMM. They need **no shared memory** and
**no** ``DEFINE_NVIDIA_*`` macro — they are fully templated on ``T``.

The two APIs
------------

``gemm_batched_1d``
~~~~~~~~~~~~~~~~~~~

``BATCH`` independent GEMMs of the same shape, each with its own A/B/C pointer
(passed as pointer arrays):

.. code-block:: cpp

   __global__ void k(float* const* A, float* const* B, float* const* C) {
       // No DEFINE macro needed — fully templated on T.
       glass::nvidia::gemm_batched_1d<float, 4, 4, 4, /*BATCH=*/8, /*TC=*/32>(
           1.f, A, B, 0.f, C);
   }
   k<<<1, dim3(32 * 8, 1, 1)>>>(dA_ptrs, dB_ptrs, dC_ptrs);   // no smem

Threads required: ``gemm_batched_1d_threads<T,M,N,K,BATCH,TC>() == TC * BATCH``.

``gemm_strided_batched_1d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **shared-A** variant: a single A matrix broadcast across ``BATCH`` packed
``(B, C)`` pairs. The caller passes one A pointer plus base B/C pointers; GLASS
indexes ``B[b * B_STRIDE]`` and ``C[b * C_STRIDE]`` internally — no pointer
arrays to set up:

.. code-block:: cpp

   __global__ void k_shared(float* A_shared, float* B_base, float* C_base) {
       // tightly packed: B_STRIDE = N*K, C_STRIDE = M*K (the defaults)
       glass::nvidia::gemm_strided_batched_1d<float, 4, 4, 4, 8, 32>(
           1.f, A_shared, B_base, 0.f, C_base);
   }

This is the GRiD end-effector-pose-gradient case, where ``&s_Xhom[16*parent]``
is the shared A for every batch element. ``B_STRIDE`` and ``C_STRIDE`` are
template parameters (defaults: ``N*K`` and ``M*K`` for tight packing).

.. note::

   A related primitive, ``glass::indexed_batched_gemm``
   (``src/base/L3/gemm_batched_indexed.cuh``), gathers per-batch operands
   through an index array at irregular offsets. It lives in ``glass::`` (not
   ``glass::nvidia::``) and uses a *different* model — a block stride over the
   flattened output elements with atomic/transpose flags — **not** the TC-group
   partitioning of the two 1D-partition APIs above. It is documented with the
   other L3 ops, not here.

When to use which
-----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - API
     - Use when
   * - ``gemm_batched_1d``
     - Each batch element has its own independent A, B, and C pointer.
   * - ``gemm_strided_batched_1d``
     - One shared A applied to many B/C blocks at a regular stride (cleanest codegen — no pointer arrays).

Both run pure SIMT, need no shared memory, and are best for small shapes
(``max(M,N,K) ≲ 8``) where cuBLASDx's tile-load overhead dominates. They respect
the ``TRAILING_SYNC`` template parameter (see :doc:`trailing_sync`).

.. note::

   For **large** batched shapes (M,N,K ≥ 16) where cuBLASDx would win, use the
   2D-launch ``glass::nvidia::gemm_batched<...,BATCH,TC>`` instead and pay the
   ``dim3(TC, BATCH)`` launch geometry. The batched-1D path deliberately does
   not attempt to wrap cuBLASDx.

See ``bench/bench_gemm_batched_1d.cu`` for a working example and the
microbenchmark that feeds the autotune table.
