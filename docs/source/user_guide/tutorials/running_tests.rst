Running Tests
=============

GLASS ships a pytest suite that compiles small CUDA binaries and checks their
output against NumPy/SciPy references.

Running the suite
-----------------

.. code-block:: bash

   cd test
   pip install -r requirements.txt
   pytest -v

The first run compiles the CUDA test binaries with ``nvcc``. Subsequent runs
skip recompilation unless the source changed (see the compile cache below).
Compiled binaries land in ``test/build/``.

Selecting tests
---------------

.. code-block:: bash

   pytest test_l1.py -v
   pytest test_l1.py -k "simple"      # glass:: threadIdx variants only
   pytest test_l1.py -k "cg"          # glass::cgrps:: cooperative-groups variants
   pytest test_l1.py -k "simple_hs"   # high_speed warp-shuffle variants

The compile cache
-----------------

Compilation is driven by ``test/conftest.py``. Binaries are compiled **once per
pytest session** (a session-scoped ``bins`` fixture) and cached by a **SHA-256
hash of the source set**.

For each test binary, ``_hash_sources()`` hashes the ``.cu`` file together with
every GLASS header it can pull in — ``glass.cuh``, ``glass-cgrps.cuh``,
``glass-nvidia.cuh``, the ``src/base/**`` and ``src/nvidia/**`` headers
(including ``tuning_table.cuh`` and ``query_simt.cuh``), and the test helpers.
The hash is written to ``test/build/<name>.hash``. On the next run, if the hash
file and binary both exist and the hash matches, compilation is **skipped**;
otherwise the binary is rebuilt. Editing **any** hashed header therefore
invalidates the cache and forces a recompile.

GPU-arch detection
------------------

``detect_arch()`` queries ``nvidia-smi --query-gpu=compute_cap`` and turns the
reported compute capability into an ``nvcc`` arch flag (e.g. ``8.6`` → ``sm_86``).
If ``nvidia-smi`` is unavailable, it falls back to ``sm_75``. Every binary is
compiled with ``-std=c++17 -arch=<detected>``.

Optional binaries (graceful skips)
----------------------------------

Some test binaries need extra dependencies, and the fixtures **skip** rather
than fail when those aren't present:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Binary
     - Requirement / behavior
   * - ``test_l3_nvidia``
     - Exercises the SIMT-only batched APIs (``gemm_batched_1d``,
       ``gemm_strided_batched_1d``). Does **not** need cuBLASDx; tests skip only
       if compilation fails for some toolchain reason.
   * - ``test_nvidia_dispatch``
     - Round-2 auto-dispatch features. Needs ``MATHDX_ROOT`` (cuBLASDx) to
       compile; skipped otherwise.
   * - ``test_trailing_sync``
     - The ``TRAILING_SYNC`` surface. Compiles with or without cuBLASDx (it
       internally skips the cuBLASDx op when ``GLASS_BENCH_CUBLASDX`` isn't
       defined).

When ``MATHDX_ROOT`` is set and ``cublasdx.hpp`` is found, the cuBLASDx-gated
binaries are compiled with ``--expt-relaxed-constexpr -DGLASS_BENCH_CUBLASDX``
plus the MathDx include paths automatically.
