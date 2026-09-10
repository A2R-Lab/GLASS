Batching models
===============

GLASS is single-block by design: you launch **one block per independent
problem**, and the block's threads cooperate over data already in shared/global
memory. But many callers have *many* small problems to run at once, and GLASS
offers several distinct ways to pack them. They differ in how problems map onto
threads/warps/blocks and in what stride/offset metadata they take — pick by how
your problems are laid out in memory and how many threads you can give each one.

.. list-table:: The batching models
   :header-rows: 1
   :widths: 22 30 48

   * - Model
     - Functions
     - When to use
   * - **One block per problem** (the base model)
     - every plain op (``gemm`` / ``gemv`` / ``syrk`` / ``posv`` / …)
     - The default. One ``<<<num_problems, threads>>>`` launch; each block owns
       one problem. Use when each problem is big enough to keep a block busy.
   * - **Block-per-problem, strided sub-blocks**
     - ``gemm_strided`` / ``gemv_strided`` (and ``axpy_strided`` /
       ``copy_strided`` for L1)
     - One block still owns one problem, but the operands are *sub-blocks* of a
       larger column-major matrix addressed with explicit leading dimensions
       (row strides). Use to operate on a tile in place without copying it out.
   * - **Flattened-batch, block-stride indexed**
     - ``gemm_batched_indexed``
     - A flat batch of GEMMs whose operands live at arbitrary offsets given by an
       index array; blocks stride over the batch. Use for gather-style batches
       where problem *i*'s pointers are not a fixed stride apart.
   * - **K-way fused, interleaved in one block**
     - ``inv`` / ``potrf`` (K-way overloads; ``inv2`` /
       ``inv3`` wrappers)
     - A *single* block factors/inverts K independent matrices at once by
       interleaving their sweeps over one shared row loop. Use when K is small
       and each matrix is too small to fill a block alone — fills the block by
       fusing across problems instead of across rows.
   * - **TC-group SIMT-1D batched**
     - the batched-1D GEMM APIs (see :doc:`batched_1d`)
     - Designed for kernels with a single 1D thread block that must run a batch
       of tiny GEMMs; threads are partitioned into per-problem groups. Use inside
       an existing 1D-block kernel that can't relaunch.
   * - **Warp-per-problem**
     - ``glass::warp::`` ops (``dot`` / ``gemv`` / ``trsv`` / ``potrf``
       / ``posv`` / …)
     - One 32-lane warp owns one problem; pack many warps into a block to run
       many problems concurrently (``threadIdx.y`` selects the warp). Use for the
       smallest problems (N ≈ 7–32) where a whole block per problem would waste
       lanes. See :doc:`contraction_parallel`.
   * - **Segmented**
     - ``gemv_segmented``
     - One matvec whose output rows are partitioned into contiguous segments with
       per-segment row counts. Use for ragged/block-structured matvecs.

Naming convention
-----------------

All batching variants follow ``operation_qualifier`` — the operation first, the
batching qualifier as a suffix (``gemm_strided``, ``gemv_segmented``,
``gemm_batched_indexed``), and the file name matches the function name. The
qualifier names *how the problems are packed*, never the scope (scope is the
namespace — ``glass::`` vs ``glass::warp::``; see :doc:`namespaces`).
