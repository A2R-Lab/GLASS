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
   glass::nvidia::block::gemm_strided_batched_1d<float, 4, 4, 4, BATCH, TC>(
       1.f, A, B, 0.f, C);

   // Opt-out — caller is responsible for syncing before reading any output
   // not written by the current thread. Pass false when fusing the GEMM with
   // subsequent block-wide work that ALREADY does its own barrier (e.g. a
   // parallel_loop that begins with __syncthreads()), so two back-to-back
   // syncs collapse into one.
   glass::nvidia::block::gemm_strided_batched_1d<
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

Every public block-scope operation accepts ``TRAILING_SYNC``. For operations
with a separable final barrier—including the factor/solve and eigensolver
families—``false`` genuinely elides it. For pivoted or multi-exit algorithms
whose last barrier is fused into the final algorithm step (currently
``ldlt``/``ldlt_solve``, ``getrf``, and ``inv_pivoted``), the parameter is
accepted for interface uniformity but documented as a no-op at the declaration.
The compile-time single-RHS ``posv<T,N>`` and ``potrs<T,N>`` spellings are the
one template-ambiguity exception; use their ``NRHS=1`` overload to select
``TRAILING_SYNC=false``.

Interior barriers (between algorithm phases) are **required for correctness and are
never gated** — only the final trailing barrier is.

``glass::thread::`` operations do not expose the knob at all — a single thread
has no cooperating peers to synchronize with. ``glass::warp::`` operations that
end on a warp-level publish point DO take ``TRAILING_SYNC`` and gate their
final ``__syncwarp()`` on it (``gemm_reduced``, ``gemv_reduced``,
``syrk_reduced``, ``tensor_contract``, ``axpy_strided``, ``copy_strided``,
banded ``load_block``/``store_block``, among others); purely
lockstep/shuffle-based warp ops need no trailing barrier and omit the
parameter. ``warp::riccati_gain`` accepts it for interface uniformity but
documents it as a no-op — the composed warp solver always ends at a sync
boundary.

Testing
-------

Tests for the surface live at ``test/cuda/test_trailing_sync.cu`` +
``test/test_trailing_sync.py``. They verify that:

1. Both ``TRAILING_SYNC=true`` and ``=false`` specializations compile and link.
2. The two variants produce numerically identical output when the ``=false``
   caller emits its own trailing ``__syncthreads()``.

The cuBLASDx-backed L3 case is covered (gated on ``GLASS_BENCH_CUBLASDX`` /
``MATHDX_ROOT``).
