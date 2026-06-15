Contribution Guidelines
=======================

GLASS is a small, header-only CUDA library, and its value depends on every
primitive being **correct for any block size**. These guidelines keep that
invariant true. For the day-to-day developer workflow (build, test, benchmark)
see :doc:`user_guide/tutorials/running_tests`; for the deeper debugging runbook
read ``docs/agent_debugging_guide.md`` in the repository.

Before you open a pull request
------------------------------

#. **Add (or update) a test.** Every public function needs a CUDA equivalence
   test in ``test/`` that compares it against a NumPy/SciPy reference. See the
   existing ``test/test_l1.py`` / ``test_l2.py`` / ``test_l3.py``.
#. **Sweep thread counts.** A single-block kernel must produce identical results
   at 1 thread, 32 threads (one warp), a partial warp, and many warps. The most
   common GLASS bug is a missing ``__syncthreads()`` between a write phase and a
   later read — invisible at 32 threads, a race at 64+. Never test only at one
   warp.
#. **Test both storage orders** where a function takes layout flags
   (``ROW_MAJOR_*`` / ``TRANSPOSE_B``).
#. **Run the suite:** ``pytest test/`` must be green (the harness compiles the
   CUDA sources once and caches by source hash).
#. **Document the public API.** New public functions get a Doxygen ``/** */``
   block (``@brief``, ``@tparam``, ``@param``, NumPy equivalent) so they appear
   in the :doc:`api_reference/index`.

Coding conventions
------------------

* Public functions live under ``src/base/**`` (pulled into ``namespace glass``
  by ``glass.cuh``), with cooperative-groups variants in ``src/cgrps/**`` and
  vendor-backed variants in ``src/nvidia/**``.
* Keep primitives **thread-count invariant**: stride with
  ``for (i = rank; i < n; i += size)`` and place a ``__syncthreads()`` between
  any write phase and a dependent read.
* Provide both runtime-sized and compile-time-sized (``<T, N, ...>``) overloads
  where it makes sense.
* A formatting baseline lives in ``.clang-format`` (advisory — no enforced
  reformat pass).

Adding a new primitive — checklist
----------------------------------

#. Implement it under the right ``src/`` directory; add the ``#include`` to the
   relevant umbrella header (``glass.cuh`` / ``glass-cgrps.cuh`` /
   ``glass-nvidia.cuh``).
#. Add a Doxygen doc-comment block.
#. Add a CUDA test source under ``test/cuda/`` and a pytest wrapper under
   ``test/`` — and register any new source file in the compile-cache list in
   ``test/conftest.py`` so edits trigger a rebuild.
#. Add an API-reference entry (a ``.. doxygenfile::`` line) on the appropriate
   page under ``docs/source/api_reference/``.

Documentation
-------------

The docs build with Sphinx + Doxygen + Breathe. See
:doc:`sphinx_edit_guide` for how to build and preview locally.
