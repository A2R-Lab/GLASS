# Contributing to GLASS

GLASS is a header-only CUDA library whose value depends on every primitive being
**correct for any block size**. Thanks for helping keep it that way.

New to the codebase? Start with [`CLAUDE.md`](CLAUDE.md) and
[`docs/agent_debugging_guide.md`](docs/agent_debugging_guide.md).

## Dev setup

GLASS is header-only — nothing to build to *use* it. For development you need an
NVIDIA GPU + CUDA toolkit (and, for the `glass::nvidia::` paths, NVIDIA MathDx).

```bash
pip install -r test/requirements.txt   # numpy, pytest, scipy
pytest test/                           # compiles test/cuda/*.cu once, runs them
```

The harness (`test/conftest.py`) auto-detects the GPU arch and caches compiled
test binaries by source hash. Force a clean rebuild with `rm -rf test/build`.

## Before you open a PR

1. **Add or update a test** in `test/` comparing your function to a NumPy/SciPy
   reference (see `test/test_l1.py` / `l2` / `l3`).
2. **Sweep thread counts** — a single-block kernel must give identical results at
   1, 32, a partial warp, and many warps. The most common bug is a missing
   `__syncthreads()` between a write phase and a dependent read (invisible at 32
   threads, a race at 64+). See `test/TESTING_STRATEGY.md`.
3. **Test both storage orders** for anything with `ROW_MAJOR_*` / `TRANSPOSE_B`.
4. **`pytest test/` is green.**
5. **Document new public functions** with a Doxygen `/** */` block and add a
   `.. doxygenfile::` line to the matching `docs/source/api_reference/*.rst`.

## Adding a primitive

Implement under the right `src/` dir, add its `#include` to the relevant umbrella
header (`glass.cuh` / `glass-cgrps.cuh` / `glass-nvidia.cuh`), add a CUDA test
source under `test/cuda/` **and register it in the compile-cache list in
`test/conftest.py`**, then add a pytest wrapper and the API-reference entry.

## Style

A `.clang-format` baseline is provided (advisory — we don't run an enforced
reformat). Match the surrounding code: 4-space indent, single-block strided
loops, `__syncthreads()` between dependent phases. Commit messages are short and
single-line.

## Docs

Sphinx + Doxygen + Breathe under `docs/`. Build locally with
`cd docs && make all` (see `docs/source/sphinx_edit_guide.rst`). The site
deploys to GitHub Pages on push to `main`.
