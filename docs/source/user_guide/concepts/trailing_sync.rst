Trailing Synchronization (``TRAILING_SYNC``)
============================================

**Uniformity rule (project-wide design decision):** *every* public GLASS op takes
a final boolean template parameter **``TRAILING_SYNC`` that defaults to ``true``**,
on both the ``glass::`` and ``glass::cgrps::`` surfaces. It controls whether the
function ends on a barrier (``__syncthreads()`` for the block surface,
``cooperative_groups::sync`` for the cgrps surface) so the result is valid for ALL
threads. One consistent mental model: a GLASS op acts like a standalone kernel
unless you opt out.

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

As of the 2026-06-25 uniformity wave, ``TRAILING_SYNC`` is on **every** public op
that has a *separable* trailing barrier — all of L1 (``reduce``/``dot``/elementwise/
``axpy``/``copy``/``scal``/``swap``/``clip``/``set_const``/``ident``/``transpose``/
``infnorm``/``asum``/``nrm2`` + the ``high_speed::``/``low_memory::`` reductions),
L2 (``gemv``/``ger`` + ``gemv_strided``/``gemv_segmented``/``gemv_reduced``), and L3
(``gemm`` + ``gemm_strided``/``gemm_batched_indexed``/``gemm_reduced``/``syrk``/
``syr2k``/``syrk_reduced``/tensor/congruence) — plus the cuBLASDx-backed
``glass::nvidia::`` paths.

Interior barriers (between algorithm phases) are **required for correctness and are
never gated** — only the final trailing barrier is.

Documented exceptions (the flag is intentionally absent):

* **Algorithm-terminated-by-a-barrier ops** — ``cholDecomp_InPlace`` /
  ``invertMatrix`` / ``trsm`` / ``trsv`` / ``posv`` / ``ldlt``. Their final step is
  *itself* a barrier inside the factorization loop, so there is no separable
  trailing barrier to gate; they always end synced.
* **``glass::warp::`` lockstep ops** — a single warp runs in lockstep, so a trailing
  "sync" is a no-op (``__syncwarp`` at most); the flag would be vacuous.

Testing
-------

Tests for the surface live at ``test/cuda/test_trailing_sync.cu`` +
``test/test_trailing_sync.py``. They verify that:

1. Both ``TRAILING_SYNC=true`` and ``=false`` specializations compile and link.
2. The two variants produce numerically identical output when the ``=false``
   caller emits its own trailing ``__syncthreads()``.

The cuBLASDx-backed L3 case is covered (gated on ``GLASS_BENCH_CUBLASDX`` /
``MATHDX_ROOT``).
