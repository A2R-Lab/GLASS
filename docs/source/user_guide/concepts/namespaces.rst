Namespaces, suffixes, and flags
===============================

GLASS names encode two **orthogonal** axes. Knowing which axis a given decoration
lives on tells you what to expect from it.

Axis A — scope / backend (the namespace)
----------------------------------------

The namespace says **who cooperates and how**, never *what* the operation is.
There are **four primary interfaces** — Block (``glass::block::``), Warp
(``glass::warp::``), Thread (``glass::thread::``), and Nvidia
(``glass::nvidia::block::`` / ``glass::nvidia::warp::``) — plus
``glass::cgrps::``, a convenience alias of the Block interface, and the
**bare** ``glass::`` face described below. The ladder runs most→least problem
packing: thread (1 problem/thread, 32 per warp) → warp (1/warp) → block
(1/block) → nvidia (1/block, vendor):

================================  ======  =================================================
Namespace                         Scope   What it is
================================  ======  =================================================
``glass::block::``                block   **Block** — hand-rolled pure-SIMT (``threadIdx`` / ``blockDim``). The **contract tier**: bit-exact, thread-count invariant, never re-dispatched.
``glass::warp::``                 warp    **Warp** — single-warp SIMT (``__shfl_*_sync``), warp-per-problem. (Namespace alias of ``block::warp`` — the warp mirrors live inline in the base headers.)
``glass::thread::``               thread  **Thread** — sequential, thread-per-problem, for low-DOF packing (compile-time sizes; register-resident up to ``N≤7``). No barriers, no shuffles, no ``threadIdx`` read. (Alias of ``block::thread``.)
``glass::nvidia::block::``        block   **Nvidia** — CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size.
``glass::nvidia::warp::``         warp    **Nvidia-warp** — CUB ``WarpReduce`` L1 reductions (``reduce`` / ``dot`` / ``nrm2``), one FULL 32-lane warp per problem; per-warp scratch sized by ``warp_reduce_scratch_bytes<T>()``; ``TRAILING_SYNC`` emits ``__syncwarp()``.
``glass::`` *(bare)*              block   **Measured default** — block-scope calling contract, body chosen by ``glass::dispatch_body()``; see below. (Likewise bare ``glass::nvidia::``.)
``glass::cgrps::``                block   *Convenience alias* of Block via a cooperative-groups handle (same numerics; not a separately-tuned backend).
================================  ======  =================================================

``glass::thread::`` mirrors the branch-free surface only: reduction *strategy*
twins (``_fast`` / ``_lowmem``), the contraction-parallel ``*_reduced`` family,
and the data-dependent/pivoted ops (``iamax``, pivoted ``ldlt``, ``getrf``,
``syev``) are deliberately absent — one thread has no reduction strategy to
choose, and with a different problem on every lane a data-dependent branch
diverges the whole warp. See the ``glass::thread::`` constraints block in
``CLAUDE.md`` for the full policy.

The convention is **namespace = scope, function name = operation**. So a warp
band-matvec is ``glass::warp::bdmv`` — never a ``glass::banded::`` namespace.

Contract tier vs performance tier — the bare ``glass::`` face
-------------------------------------------------------------

The 2026-07-30 restructure splits each spelling along one more line:

- **Explicit namespaces are the contract tier.** ``glass::block::gemm`` (and
  likewise ``glass::warp::`` / ``glass::thread::`` / ``glass::nvidia::block::``
  / ``glass::nvidia::warp::``) names a specific implementation: bit-exact,
  thread-count invariant, never re-dispatched. Emit these from codegen and
  anywhere determinism is load-bearing.
- **Bare** ``glass::gemm`` (and bare ``glass::nvidia::gemm``) **is the
  measured-default face**: the same block-scope *calling* contract — all block
  threads enter, the result is valid after return — with the implementation
  *body* chosen per (op, size, dtype) by ``glass::dispatch_body()`` in
  ``glass-defaults.cuh``.

**Phase 1 (today): the two faces are identical.** Every ``dispatch_body()``
cell pins to the block body, so the bare names resolve to the *same entities*
as ``glass::block::`` — re-exported by a ``using namespace block;`` directive
in the umbrella headers, with function-pointer identity pinned by
``test/cuda/test_defaults.cu``. All pre-restructure spellings still compile
with identical meaning; Phase 1 is bit-identical by construction. A future
measured **in-block body sweep** (a ``bench/tune.py`` leg) may move cells to a
warp- or thread-body executed under the same block-scope contract — an
attested, receipt-gated retune, never a silent change (see :doc:`tuning`).

Rule of thumb: **explicit namespace = contract tier; bare namespace =
performance tier.**

Axis B — reduction strategy (function-name suffixes, vector reductions only)
---------------------------------------------------------------------------

A second axis lives on the reduction primitives only: bare ``glass::reduce``
(halving), ``glass::reduce_fast`` (warp-shuffle + shared inter-warp), and
``glass::reduce_lowmem`` (thread-0 serial, no scratch) — performance/scratch
trade-offs of the **same** result. The same ``_fast`` / ``_lowmem`` suffixes
apply across the reduction family (``dot``, ``nrm2``, ``asum``, ``vector_norm``,
``nrm1_diff``, ``iamax``). This is a strategy, not a scope, so it rides on the
*function name*, never a sub-namespace — keeping **namespace = scope** true
everywhere. (These were ``glass::high_speed::`` / ``glass::low_memory::``
sub-namespaces until the 2026-06 convergence; that clean break is done.)

The naming rule for new code
----------------------------

When you add an operation, decide what kind of variation it is:

- **A different algorithm or decomposition → its own function name (a suffix).**
  The contraction-parallel gemm is :cpp:func:`glass::gemm_reduced`, not a
  ``glass::reduced::`` namespace — matching the existing ``gemm_tiled`` /
  ``gemm_dispatch`` precedent. Same scope, different name.
- **Optional, additive behavior → a compile-time** ``bool`` **flag that compiles
  out.** ``potrf<T, N, CHECK>``, ``ldlt<T, N, CHECK>``,
  ``posv<T, N, NRHS, REGULARIZE, CHECK>`` all default the flag to ``false`` and
  guard the extra work behind ``if constexpr`` — so the unflagged instantiation is
  **byte-identical** to the original (no PTX change, no perf cost). This is how the
  robustness features (non-PD detection, inertia, Levenberg shift) attach to the
  existing factor/solve ops instead of forking new functions.
- **A different scope → a different namespace** (Axis A), with the *same* function
  name.

So: *scope* picks the namespace, *additive behavior* picks a flag, and *a genuinely
different computation* picks a new name. Following that keeps the surface
predictable — you can guess the spelling of an op you have not seen.

As of the 2026-06 convergence, namespace means *scope* everywhere — the
former ``high_speed::`` / ``low_memory::`` reduction sub-namespaces are now the
``_fast`` / ``_lowmem`` suffixes described above.
