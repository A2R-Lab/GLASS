Thread-scoped operations (``glass::thread::``)
==============================================

Sequential, one-problem-per-**thread** variants of the branch-free primitives:
a single thread owns the whole operation — **no barriers, no shuffles, no**
``threadIdx`` **read** — so 32 independent problems pack into one warp. They
target the *low-DOF corner* (robot DOF ≲ 7) where even a warp per problem
leaves most lanes idle: a thread-per-problem launch
(``<<<ceil(P/TPB), TPB>>>``, with packing from ``glass::recommend()``)
keeps every lane busy on its own problem.

Contract: **compile-time sizes only.** The tier's value is operands that nvcc
keeps register-resident, which requires fully-unrolled, compile-time-resolvable
indexing. Around ``N ≤ 7`` is a useful register-residency guideline, not an API
or performance ceiling: larger ``N`` still computes correctly, may spill to
local memory, and remains in the tuning ladder while it is feasible. Operands
may be thread-local register arrays; nothing is read
from ``threadIdx``, so the functions are launch-shape-agnostic.

Every op delegates to the same ``*_impl`` body its block-scoped sibling uses,
collapsed through ``ThreadBarrier`` (rank 0, size 1, no-op sync) — the same
algorithm and operand order as ``glass::block::`` on one thread, agreeing to within a
few ULP (FMA contraction may differ between the two instantiations;
``test/test_thread.py`` pins the bound). They live in the same base headers as
their block-scoped siblings (under ``namespace thread``), so their rendered
signatures appear on the :doc:`l1`, :doc:`l2`, and :doc:`l3` pages.

**Deliberately absent** (see the ``glass::thread::`` constraints block in
``CLAUDE.md``): the ``_fast`` / ``_lowmem`` reduction twins (they name reduction
*strategies*; one thread has none — ``thread::dot`` simply returns a serially
accumulated ``T``), the contraction-parallel ``*_reduced`` family, and every
data-dependent / pivoted op (``iamax``, pivoted ``ldlt``, ``getrf`` / ``gesv``,
``inv_pivoted``, ``syev``) — when every lane owns a *different* problem, a
data-dependent branch diverges the whole warp, which is exactly the cost the
tier exists to avoid.

**Surface** (mirrors the branch-free warp surface):

* **L1**: ``dot`` (returns ``T``), the serial reductions ``reduce`` / ``nrm2`` /
  ``asum`` / ``nrm1_diff``, and the maps ``axpy`` / ``axpy_strided`` / ``scal``
  / ``copy`` / ``copy_strided`` / ``rot`` / ``symmetrize``. See :doc:`l1`.
* **L2**: ``gemv`` (flags: ``TRANSPOSE`` / ``ROW_MAJOR``), ``trsv``
  (``FillMode`` / ``Diag`` / ``TRANSPOSE``). See :doc:`l2`.
* **L3**: ``gemm``, ``syrk`` / ``syr2k``, ``trsm``, the factor/solve chain
  ``potrf`` / ``posv`` / ``potrs`` / ``ldlt`` (non-pivoted) / ``ldlt_solve`` /
  ``inv`` (non-pivoted), and the fused families ``tensor_vec_contract`` /
  ``vec_tensor_vec``, ``congruence_sym`` / ``bilinear`` / ``congruence_accum``,
  ``riccati_gain``. See :doc:`l3`.

The dispatch ladder (:doc:`defaults`) contends the tier alongside native warp /
block and NVIDIA block / thread implementations. The dated measurements and
generated verdicts are on the :doc:`sweep-results page
<../user_guide/tutorials/sweep_results>` and in ``bench/RESULTS.md``. Run
``bench/tune.py --sm auto`` to contend it on your own GPU.
