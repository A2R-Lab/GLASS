Benchmarks
==========

The benchmark suite under ``bench/`` compares GLASS variants against block-level
CUDA library baselines **and** against each other, so you can see which path
wins for a given shape on your hardware.

What's in ``bench/``
--------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Comparison
   * - ``bench_reduce.cu``
     - ``glass::*::reduce/dot/l2norm`` (plain, low_memory, high_speed,
       compile-time) vs CUB ``BlockReduce`` vs ``glass::nvidia::reduce``
   * - ``bench_gemv.cu``
     - ``glass::gemv`` (runtime + compile-time) vs raw cuBLASDx vs
       ``glass::nvidia::gemv`` (default + caller-pinned ``BlockDim<256>``)
   * - ``bench_gemm.cu``
     - ``glass::gemm`` (plain, tiled, compile-time) vs raw cuBLASDx vs
       ``glass::nvidia::gemm`` (default + caller-pinned)
   * - ``bench_blockdim.cu``
     - ``glass::nvidia::gemm`` cuBLASDx-chosen block_dim vs caller-pinned
       ``BlockDim<128>`` vs ``BlockDim<352>``
   * - ``bench_gemm_batched.cu``
     - ``glass::nvidia::gemm_batched<...,BATCH>`` vs a naive ``for(b)`` loop, for
       BATCH ∈ {4, 8, 16, 32}
   * - ``bench_gemm_batched_1d.cu``
     - 1D-launch ``gemm_batched_1d`` (SIMT vs cuBLASDx) — feeds the autotune table
   * - ``bench_lapack.cu`` *(needs cuSOLVERDx)*
     - pure-SIMT ``glass::cholDecomp_InPlace`` / ``trsm`` vs
       ``glass::nvidia::chol_inplace`` / ``trsm`` / ``posv`` (fused)

CUB ships with CUDA 11+. cuBLASDx and cuSOLVERDx ship together in NVIDIA MathDx
— see :doc:`../getting_started/installation`.

Running the suite
-----------------

Set ``MATHDX_ROOT`` to your MathDx install, then:

.. code-block:: bash

   cd /path/to/GLASS
   python3 bench/run_bench.py

   # Custom iteration count (default: 10000)
   python3 bench/run_bench.py --iters 50000

   # Skip cuBLASDx (only bench_reduce will run)
   python3 bench/run_bench.py --no-cublasdx

The driver auto-detects dependencies and prints what it found at startup:

.. code-block:: text

   === GLASS Benchmark Suite ===
   GPU arch: sm_120 (SM1200)
   cuBLASDx: enabled (/opt/nvidia/mathdx/25.12)
   cuSOLVERDx: enabled
   Iterations: 10000

``bench_lapack`` is skipped automatically if ``cusolverdx.hpp`` is not present
under ``$MATHDX_ROOT/include/``. When cuSOLVERDx is enabled, the driver adds the
``-rdc=true -dlto -lcusolverdx ...`` device-link flags for you.

Results are printed as a Markdown table and saved to
``bench/results/bench_<hostname>.json``. Timing uses the GRiD pattern — the
iteration loop runs *inside* the kernel to amortize launch overhead.

.. note::

   **Anti-optimization safeguards** are baked into every bench loop:
   per-iteration writes to a ``volatile`` sink defeat dead-store elimination;
   destructive inputs (Cholesky, LU, QR overwrite their input) are reloaded from
   a master copy each iteration; ``nvcc -Xptxas -O1`` is enforced. Numbers below
   ~0.1 µs/op for a non-trivial kernel almost always mean the bench was elided —
   recheck the safeguards.

Autotuning from the bench
-------------------------

The same harness backs ``bench/autotune.py``, which measures SIMT vs cuBLASDx
per shape and writes a per-host override table so the ``glass::nvidia::*``
auto-dispatch picks the measured winner instead of the static heuristic. See
:doc:`../concepts/tuning` for that workflow.
