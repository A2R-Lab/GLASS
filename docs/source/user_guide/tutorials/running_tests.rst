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

Compilation is driven by ``test/conftest.py``. The session-scoped ``bins``
mapping compiles a binary lazily, only when a selected test first accesses it,
and caches it by a **SHA-256 hash of its source set**.

For each test binary, ``_hash_sources()`` hashes its ``.cu`` file, shared helper,
umbrella headers, and implementation headers for the binary's operation family
(vector, dense/factor, solver, robotics, or Nvidia). The hash is written to
``test/build/<name>.hash``. A local change therefore rebuilds the selected
family without rebuilding unrelated binaries. New binaries default to a
whole-source-tree hash until assigned to a family.

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
     - Round-2 compile-time auto-dispatch features. Needs ``MATHDX_ROOT`` (cuBLASDx) to
       compile; skipped otherwise.
   * - ``test_trailing_sync``
     - The ``TRAILING_SYNC`` surface. Compiles with or without cuBLASDx (it
       internally skips the cuBLASDx op when ``GLASS_BENCH_CUBLASDX`` isn't
       defined).

When ``MATHDX_ROOT`` is set and ``cublasdx.hpp`` is found, the cuBLASDx-gated
binaries are compiled with ``--expt-relaxed-constexpr -DGLASS_BENCH_CUBLASDX``
plus the MathDx include paths automatically.

Environment knobs
-----------------

``CUDA_ARCH`` overrides architecture detection (set it on Jetson, where
``nvidia-smi`` is absent and the fallback would be ``sm_75``).
``GLASS_RUN_TIMEOUT`` (seconds, default 600) bounds every driver subprocess,
so a hung kernel fails its test instead of stalling the run; raise it on slow
devices rather than removing it. ``GLASS_COMPILE_TIMEOUT`` (default 3600)
bounds each ``nvcc`` invocation. ``GLASS_JIT_ARCH`` (default ``compute_75``)
selects the virtual architecture of the PTX-JIT lane described below; that
lane skips only on a device older than its target. ``GLASS_RUN_PREFIX`` wraps
every launch, for example with ``compute-sanitizer --error-exitcode 1``.

Validated lanes
---------------

Correctness evidence is recorded per lane: the device, the toolkit that
compiled the binaries, and how the code reached that device.
:doc:`../getting_started/supported_platforms` lists every lane at its
evidence level (attested, exercised, compiles); this section covers how a
lane is recorded and verified.

The desktop lane is the receipt in ``test/gpu-proof.json``: all eight shards
on the RTX 5090, signed by the keyholder, refreshed with every change to a
fingerprinted path and verified by CI on every push. Its ``integration``
shard also runs the restricted-caller regression through the driver's PTX
JIT for ``compute_75`` (``test_restrict.py`` ``*_jit`` tests), a second code
generator on every receipt.

A device lane is a second receipt file, ``test/gpu-proof-<lane>.json``. The
merge refuses shards from different environments, so a Jetson never folds
into the desktop receipt. The device records the eight shards unsigned, from
a clean clone at the commit to attest, with the same runner:

.. code-block:: bash

   GLASS_PY=~/glass-venv/bin/python GLASS_SKIP_PIP=1 CUDA_ARCH=sm_87 \
       ./test/run_gpu_proof.sh --lane sm87 --unsigned

The shard receipts land in ``test/receipts/sm87/``; copied to the
keyholder's box, they are merged and signed there:

.. code-block:: bash

   gpu-proof merge --out test/gpu-proof-sm87.json --github-user plancherb1 \
       test/receipts/sm87/*.json

``GLASS_PY`` names the interpreter (the Jetsons have no ``.venv``),
``GLASS_SKIP_PIP`` leaves its packages alone, ``GLASS_ROOT`` points a copy of
the runner at another checkout, and ``CUDA_ARCH`` is needed where
``nvidia-smi`` is absent. A lane that cannot reach every shard records its
skips, and CI pins that set exactly through ``test/expected_skips_<lane>.txt``
(the Xavier has no MathDx for CUDA 11.4 and cannot JIT ``compute_75``).

CI verifies each lane in a worktree at the commit the receipt names, so the
fingerprint recomputes against the tree the device ran, and requires that
commit to be an ancestor of the commit under test. The lane policy
(``test/gpu-proof-policy-lane.json``) allows no carried shards and rejects a
lane older than 180 days, which is the refresh horizon for the Jetsons.

A reload-to-plain-load mutant of ``cooperative_detail::load`` (the 2026-10
handoff fix) is caught by the restricted-caller regression on all four code
generators above: CUDA 13.2 for ``sm_120`` (native and ``compute_75`` JIT),
CUDA 13.2 for ``sm_87`` and CUDA 11.4 for ``sm_72``. The ``reduce_fast`` and
``argreduce`` reloads survive that mutant everywhere; they are there for
consistency of the handoff pattern, not because a miscompile was demonstrated.

The selected-operand regression (``test_select.py``) ran on the same three
GPUs: it fails on the header that still carried ``__restrict__`` and passes on
the current one, in both precisions, under CUDA 13.2 for ``sm_120`` and
``sm_87`` and CUDA 11.4 for ``sm_72``; since 2026-11-02 it is part of the
``integration`` shard of all three lane receipts.

Device ``printf`` is not used for test output: its FIFO defaults to 1 MB on a
Jetson AGX Orin against 8 MB on an RTX 5090, and a long dump is silently cut
when it fills (2048 values, then nothing). Drivers copy results to the host
and print there.

CUDA 13 cannot compile for ``sm_72``, so the Xavier lane always uses the
toolkit installed on the board. Hopper and desktop Ampere parts have compile
coverage only; building ``sm_86`` or ``sm_90`` SASS on an RTX 5090 says
nothing about running it, and the PTX-JIT lane is a second compiler, not an
emulation of older hardware.
