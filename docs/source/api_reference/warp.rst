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
rendered signatures appear on the :doc:`l1` and :doc:`l3` pages:

* ``glass::warp::reduce`` — single-warp sum (array and register-partial forms);
  see :doc:`l1`.
* ``glass::warp::gemm`` — compile-time-size GEMM across one warp (e.g. 4×4
  homogeneous-transform multiplies); see :doc:`l3`.
* ``glass::warp::cholDecomp_InPlace`` / ``glass::warp::trsm`` /
  ``glass::warp::trsm_transpose`` — small SPD factor + forward/transpose solves;
  compose them for a warp-scoped normal-equations solve. See :doc:`l3`.

See ``examples/07_warp_ops.cu`` for a runnable demonstration.
