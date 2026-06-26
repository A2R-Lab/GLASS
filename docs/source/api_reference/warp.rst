Warp-scoped operations (``glass::warp::``)
==========================================

Single-warp SIMT variants of selected primitives: one 32-lane warp cooperates
using raw ``__shfl_*_sync`` intrinsics, with **no shared scratch, no inter-warp
combine, and no** ``__syncthreads``. They target *warp-per-problem* kernels — a
block that solves many independent problems, one per warp — where the
block-scoped ``glass::`` surface would serialize across warps and the
cooperative-groups / vendor paths add overhead at these tiny sizes.

Contract: the caller must run a full 32-lane warp (mask ``0xffffffff``);
partial-warp callers pass ``0`` from inactive lanes. These live in the same base
headers as their block-scoped siblings (under ``namespace warp``), so their
rendered signatures appear on the :doc:`l1`, :doc:`l2`, and :doc:`l3` pages.

**L1 / L2 glue** (the building blocks for a full warp-per-problem solve):

* ``glass::warp::dot`` — single-warp dot product, broadcast to every lane;
  ``glass::warp::axpy`` / ``glass::warp::copy`` / ``glass::warp::scal`` —
  elementwise vector ops; ``glass::warp::reduce`` — single-warp sum (array and
  register-partial forms); ``glass::warp::iamax`` — single-warp index of max-abs
  (register-returned, lowest-index tie-break). See :doc:`l1`.
* ``glass::warp::gemv`` — one output row per lane (reuses the block ``gemv``
  inner kernel). See :doc:`l2`.

**L3 factor / solve:**

* ``glass::warp::gemm`` — compile-time-size GEMM across one warp (e.g. 4×4
  homogeneous-transform multiplies).
* ``glass::warp::cholDecomp_InPlace`` — small SPD Cholesky factor.
* ``glass::warp::trsv`` — flagged triangular solve (``LOWER`` / ``UNIT`` /
  ``TRANSPOSE``), subsuming the lower ``glass::warp::trsm`` /
  ``glass::warp::trsm_transpose``.
* ``glass::warp::posv`` — the **composed warp-per-problem SPD solve**
  (Cholesky → forward/back ``trsv``), proving the L1/L2/L3 glue composes into a
  complete per-warp linear solve. See :doc:`l3`.

All cross-lane scalars (reduction results, solve pivots) are broadcast via
``__shfl_sync`` from a lane register — never a shared re-read — to avoid the
``__restrict__`` stale-cache miscompile class.

See ``examples/07_warp_ops.cu`` for a runnable demonstration.
