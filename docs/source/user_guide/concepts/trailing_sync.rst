Trailing Synchronization (``TRAILING_SYNC``)
============================================

GLASS L3 functions (``gemm``, ``gemv``, ``row_strided_*``, ``gemm_batched_1d``,
``gemm_strided_batched_1d``, ...) take a final boolean template parameter
**``TRAILING_SYNC`` that defaults to ``true``**. It controls whether the
function emits a ``__syncthreads()`` before returning.

Default vs. opt-out
-------------------

.. code-block:: cpp

   // Default — function returns with all threads at a block-wide barrier.
   // Safe to read the result from any thread in the block immediately after.
   glass::nvidia::gemm_strided_batched_1d<float, 4, 4, 4, BATCH, TC>(
       1.f, A, B, 0.f, C);

   // Opt-out — caller is responsible for syncing before reading any output
   // not written by the current thread. Pass false when fusing the GEMM with
   // subsequent block-wide work that ALREADY does its own barrier (e.g. a
   // parallel_loop that begins with __syncthreads()), so two back-to-back
   // syncs collapse into one.
   glass::nvidia::gemm_strided_batched_1d<
       float, 4, 4, 4, BATCH, TC,
       /*B_STRIDE=*/N*K, /*C_STRIDE=*/M*K,
       layout::col_major, layout::col_major, layout::col_major,
       /*TRAILING_SYNC=*/false>(1.f, A, B, 0.f, C);
   __syncthreads();  // emit your own here, fused with any other barrier

The default of ``true`` makes the common case correct without thinking — GLASS
functions act as if they were standalone kernels. The opt-out exists for hot
kernels (e.g. GRiD's ``end_effector_pose_gradient_inner``) that chain a GEMM
with a SIMT ``parallel_loop`` and want to collapse the two syncs into one.

Where it applies
----------------

The full surface as of 2026-05:

* **L1** (``l1.cuh``): ``reduce``, ``dot``, ``l2norm``. The *leading* sync that
  protects CUB ``BlockReduce`` ``TempStorage`` from prior writes is **not**
  gated — it is required for correctness. Only the *trailing* barrier (after the
  thread-0 write of the reduction result) is.
* **L2** (``l2.cuh``): ``gemv``, ``row_strided_gemv`` — primary templates and
  every ``_GLASS_GEMV_NO_BD`` / ``_GLASS_GEMV_BD`` macro-emitted
  specialization. The cuBLASDx-backed specializations emit **both**
  ``TRAILING_SYNC=true`` and ``=false`` variants from a single
  ``DEFINE_NVIDIA_GEMV*`` invocation.
* **L3** (``l3.cuh``): ``gemm``, ``gemm_batched``, ``row_strided_gemm`` — same
  dual-specialization pattern via ``_GLASS_GEMM_NO_BD`` / ``_GLASS_GEMM_BD`` /
  ``_GLASS_GEMM_BATCHED_BD``.
* **L3 SIMT** (``l3_simt.cuh``): ``gemm_batched_1d``,
  ``gemm_strided_batched_1d``.
* **LAPACK** (``lapack.cuh``): ``posv``, ``chol_inplace``, ``gesv_no_pivot``,
  ``gels``, ``trsm`` — primary templates and macro-emitted specializations.

.. note::

   Interior ``__syncthreads()`` inside ``gemm`` / ``gemm_batched`` / LAPACK
   factor-solve flows (between phases) is **required for correctness and is not
   gated** on ``TRAILING_SYNC`` — only the final trailing barrier before return
   is.

Testing
-------

Tests for the surface live at ``test/cuda/test_trailing_sync.cu`` +
``test/test_trailing_sync.py``. They verify that:

1. Both ``TRAILING_SYNC=true`` and ``=false`` specializations compile and link.
2. The two variants produce numerically identical output when the ``=false``
   caller emits its own trailing ``__syncthreads()``.

The cuBLASDx-backed L3 case is covered (gated on ``GLASS_BENCH_CUBLASDX`` /
``MATHDX_ROOT``).
