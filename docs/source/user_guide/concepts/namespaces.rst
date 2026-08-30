Namespaces, suffixes, and flags
===============================

GLASS names encode two **orthogonal** axes. Knowing which axis a given decoration
lives on tells you what to expect from it.

Axis A — scope / backend (the namespace)
----------------------------------------

The namespace says **who cooperates and how**, never *what* the operation is.
There are three execution scopes (block, warp, thread) and two implementation
families (dependency-free GLASS and optional NVIDIA, where supported), plus
``glass::cgrps::``, a convenience alias of the Block interface, and the
**bare** ``glass::`` face described below. Scope determines placement; the
family determines the implementation and dependency contract:

================================  ======  =================================================
Namespace                         Scope   What it is
================================  ======  =================================================
``glass::block::``                block   **Block** — explicit hand-rolled pure-SIMT implementation (``threadIdx`` / ``blockDim``). CONTRACT tier: bit-exact, thread-count invariant for deterministic-order ops (the ``_fast`` shuffle reductions and ``pcg`` are oracle-close with documented ``blockDim``-dependent summation order), never re-dispatched.
``glass::warp::``                 warp    **Warp** — single-warp SIMT (``__shfl_*_sync``), warp-per-problem. (Namespace alias of ``block::warp`` — the warp mirrors live inline in the base headers.)
``glass::thread::``               thread  **Thread** — sequential, thread-per-problem, for low-DOF packing (compile-time sizes; register-resident up to ``N≤7``). No barriers, no shuffles, no ``threadIdx`` read. (Alias of ``block::thread``.)
``glass::nvidia::block::``        block   **Nvidia** — CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size at compile time.
``glass::nvidia::warp::``         warp    **Nvidia-warp** — CUB ``WarpReduce`` L1 reductions (``reduce`` / ``dot`` / ``nrm2``), one FULL 32-lane warp per problem; per-warp scratch sized by ``warp_reduce_scratch_bytes<T>()``; ``TRAILING_SYNC`` emits ``__syncwarp()``.
``glass::nvidia::thread::``       thread  **Nvidia-thread** — cuSOLVERDx 0.4+ LAPACK, one packed problem per CUDA thread; smem-less signatures and no block-wide synchronization.
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

The bare and explicit spellings make a deliberate implementation choice:

- **Explicit namespaces pin an implementation.** ``glass::block::gemm`` (and
  likewise ``glass::warp::`` / ``glass::thread::`` / ``glass::nvidia::block::``
  / ``glass::nvidia::warp::`` / ``glass::nvidia::thread::``) is never
  re-dispatched. Use these from codegen
  and wherever implementation or reduction order is load-bearing.
- **Bare** ``glass::gemm`` (and bare ``glass::nvidia::gemm``) **is the
  measured-default face**: the same block-scope *calling* contract — all block
  threads enter, any thread count, the result is valid after return — with
  the implementation *body* chosen per
  (op, size, dtype) by ``glass::dispatch_body()`` in ``glass-dispatch.cuh``.
  The choice is a ``constexpr`` selection made **inside the device function**
  — resolved per call site at compile time, not by a host-side dispatcher and
  not by any runtime branch.

The body sweep (``bench/tune.py --legs body``) compares full-block, warp-0, and
thread-0 bodies while preserving the one-problem-per-block calling contract.
Only cells meeting the configured robustness margin are moved; unmeasured
architectures and out-of-range sizes stay on the block body. A moved cell agrees
with the block result to its documented tolerance, not necessarily bit for bit.
Retuning is therefore a receipt-gated source change; see :doc:`tuning`.

Rule of thumb: **explicit namespace = contract tier; bare namespace =
performance tier.**

Axis B — reduction strategy (function-name suffixes, vector reductions only)
-----------------------------------------------------------------------------

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
  The contraction-parallel gemm is ``glass::gemm_reduced``, not a
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
