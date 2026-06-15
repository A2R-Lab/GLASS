# Doxygen doc-comment coverage

**Status:** public entry points done; long tail optional.

The 2026-06-15 docs pass added Doxygen `/** */` blocks to the **public,
user-facing** functions across `src/base/**`, `src/cgrps/**`, `src/nvidia/**`,
and the top-level headers — enough for a complete API reference of the surface
users call. `Doxyfile` sets `EXTRACT_ALL = NO`, so undocumented internals simply
don't appear (no broken Breathe references).

## Optional follow-ups

- Doc-comment any remaining public overloads that slipped through (grep for
  `__device__` functions in `src/base/**` without a preceding `/**`).
- Decide whether the `simple::` / `low_memory::` / `high_speed::` sub-namespace
  variants warrant their own API-reference grouping or a short note on each page.
- If internal `*_impl` helpers ever become useful to document for contributors,
  add them under a separate "internals" toctree rather than the public reference.

## How to add coverage

1. Add a `/** @brief ... @tparam ... @param ... */` block above the function.
2. Ensure its header file is referenced by a `.. doxygenfile::` line in the
   matching `docs/source/api_reference/*.rst` page.
3. `cd docs && make all` and confirm it renders with no Breathe warnings.
