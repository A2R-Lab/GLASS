Namespaces, suffixes, and flags
===============================

GLASS names encode two **orthogonal** axes. Knowing which axis a given decoration
lives on tells you what to expect from it.

Axis A — scope / backend (the namespace)
----------------------------------------

The namespace says **who cooperates and how**, never *what* the operation is:

================================  ======  =================================================
Namespace                         Scope   What it is
================================  ======  =================================================
``glass::``                       block   Hand-rolled pure-SIMT (``threadIdx`` / ``blockDim``).
``glass::cgrps::``                block   Same SIMT loop via a cooperative-groups handle.
``glass::nvidia::``               block   CUB / cuBLASDx / cuSOLVERDx, auto-dispatched by size.
``glass::warp::``                 warp    Single-warp SIMT (``__shfl_*_sync``), warp-per-problem.
================================  ======  =================================================

The convention is **namespace = scope, function name = operation**. So a warp
band-matvec is ``glass::warp::bdmv`` — never a ``glass::banded::`` namespace.

Axis B — reduction strategy (sub-namespaces, vector reductions only)
--------------------------------------------------------------------

A second, *historical* axis lives on the reduction primitives only: bare
``glass::reduce`` (halving), ``glass::high_speed::reduce`` (warp-shuffle + shared
inter-warp), ``glass::low_memory::reduce`` (thread-0 serial, no scratch). These
are performance/scratch trade-offs of the **same** result. They are the one place
where a sub-namespace encodes something other than scope — a wrinkle the
convergence proposal (below) proposes to retire.

The naming rule for new code
----------------------------

When you add an operation, decide what kind of variation it is:

- **A different algorithm or decomposition → its own function name (a suffix).**
  The contraction-parallel gemm is :cpp:func:`glass::gemm_reduced`, not a
  ``glass::reduced::`` namespace — matching the existing ``gemm_tiled`` /
  ``gemm_dispatch`` / ``gemm_ex`` precedent. Same scope, different name.
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

See also the convergence proposal in ``docs/open-tasks/`` for the planned
clean-break that folds the Axis-B reduction sub-namespaces into name suffixes, so
that namespace means *scope* everywhere.
