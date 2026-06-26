Namespaces, suffixes, and flags
===============================

GLASS names encode two **orthogonal** axes. Knowing which axis a given decoration
lives on tells you what to expect from it.

Axis A — scope / backend (the namespace)
----------------------------------------

The namespace says **who cooperates and how**, never *what* the operation is.
There are **three primary interfaces** — Block (``glass::``), Warp
(``glass::warp::``), and Nvidia (``glass::nvidia::``) — plus ``glass::cgrps::``, a
convenience alias of the Block interface:

================================  ======  =================================================
Namespace                         Scope   What it is
================================  ======  =================================================
``glass::``                       block   **Block** — hand-rolled pure-SIMT (``threadIdx`` / ``blockDim``).
``glass::warp::``                 warp    **Warp** — single-warp SIMT (``__shfl_*_sync``), warp-per-problem.
``glass::nvidia::``               block   **Nvidia** — CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size.
``glass::cgrps::``                block   *Convenience alias* of Block via a cooperative-groups handle (same numerics; not a separately-tuned backend).
================================  ======  =================================================

The convention is **namespace = scope, function name = operation**. So a warp
band-matvec is ``glass::warp::bdmv`` — never a ``glass::banded::`` namespace.

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
  out.** ``cholDecomp_InPlace<T, N, CHECK>``, ``ldlt<T, N, CHECK>``,
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
