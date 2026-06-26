Library Overview
================

**GLASS** — *GPU Linear Algebra Simple Subroutines*. Composable ``__device__``
BLAS/LAPACK-style subroutines that run inside a single CUDA block. Now expanding
to warp-level primitives for packing many small problems into one block.

What GLASS is
-------------

GLASS functions are ``__device__`` helpers that operate on data in shared or
device memory. It began as a small set of hand-rolled SIMT subroutines tuned
for very small matrices — sizes where the launch and dispatch overhead of a
vendor library would dominate the actual work — and has since grown into a
**unified single-block linear-algebra surface** that also wraps NVIDIA's
state-of-the-art device-side libraries (CUB, cuBLASDx, cuSOLVERDx) under the
same calling convention.

The intent is one consistent API across the full block-size compute scale:
pure-SIMT for tiny matrices where unrolled SIMT can't be beaten, and
tensor-core-tuned vendor kernels for everything large enough to benefit.

The single-block execution model
---------------------------------

Every GLASS function assumes it runs within **one CUDA thread block**. The
caller is responsible for launching **one block per independent data item**:

.. code-block:: cpp

   my_kernel<<<num_items, 256>>>(A, B, C, m, n, k);

Inside the kernel, all threads of the block cooperate on a single problem. This
design enables composable GPU kernels for applications such as model-predictive
control and rigid-body dynamics, where many small independent linear-algebra
problems run in parallel — one per block.

Call surfaces
-------------

Primitives are **block-scoped** by default (one block per problem) in three
numerically-interchangeable backends, plus a **warp-scoped** surface
(``glass::warp::``) for kernels that pack many small independent problems into one
block — one per warp:

.. list-table::
   :header-rows: 1
   :widths: 20 14 40 26

   * - Namespace
     - Scope
     - Backend
     - Header
   * - ``glass::``
     - block
     - Hand-rolled SIMT, ``threadIdx.{x,y,z}`` / ``blockDim.*`` (no cooperative-groups dependency)
     - ``glass.cuh``
   * - ``glass::cgrps::``
     - block
     - Hand-rolled SIMT, ``g.thread_rank()`` / ``g.size()`` (cooperative groups)
     - ``glass-cgrps.cuh``
   * - ``glass::nvidia::``
     - block
     - CUB (L1) + cuBLASDx (L2/L3, batched) + cuSOLVERDx (LAPACK) — compile-time sizes only
     - ``glass-nvidia.cuh``
   * - ``glass::warp::``
     - warp
     - Single-warp SIMT via ``__shfl_*_sync`` — selected L1/L2/L3 ops, no ``__syncthreads`` / shared
     - inline in the base L1/L2/L3 headers

The three **block-scoped** backends cover the full L1/L2/L3 surface and are
interchangeable — switch by changing the namespace prefix when profiling shows
one is faster at a given size. They preserve the same one-block ``__device__``
calling convention, so a single kernel can mix hand-rolled and vendor-backed
primitives without leaving the block. ``glass::warp::`` is a **warp-per-problem**
variant covering a selected set (``dot``, ``axpy``, ``copy``, ``scal``,
``reduce``, ``gemv``, ``iamax``, ``gemm``, ``cholDecomp_InPlace``, ``trsv`` /
``trsm`` / ``trsm_transpose``, and the composed ``posv`` SPD solve); the warps
run independently for intra-block parallelism, and it requires a full 32-lane
warp.

Both ``glass::`` and ``glass::cgrps::`` offer **runtime** (size as a function
argument) and **compile-time** (size as a template argument) overloads for every
function. Reduction operations additionally offer ``_lowmem`` (no
scratch, thread 0 accumulates) and ``_fast`` (warp-shuffle plus shared-memory
inter-warp reduction) suffixed forms — e.g. ``glass::reduce_lowmem`` /
``glass::reduce_fast`` — keeping namespace = scope.

**Higher-level solvers** build on these primitives (and are likewise
single-block): ``glass::bdmv`` (block-tridiagonal matvec) and ``glass::pcg``
(preconditioned conjugate gradient) for the block-tridiagonal SPD systems of
trajectory optimization / MPC — see :doc:`../concepts/block_tridiagonal`.

Choosing the right backend
--------------------------

First pick the **execution scope** — one block per problem (the default
``glass::`` / ``glass::cgrps::`` / ``glass::nvidia::`` surface) or one warp per
problem (``glass::warp::``). For the block-scoped backends, three questions decide
which to call:

1. **Are sizes known at compile time?**
2. **Is the matrix large enough that vendor-tuned tensor-core kernels matter?**
3. **Can you launch with the thread count the backend wants?**

.. list-table::
   :header-rows: 1
   :widths: 40 32 28

   * - Scenario
     - Use
     - Reason
   * - Sizes only known at runtime
     - ``glass::gemm(m, n, k, ...)``
     - Pure-SIMT, accepts dynamic args
   * - Compile-time sizes, small matrices (≤ ~8×8), simple kernel
     - ``glass::gemm<float, M, N, K>(...)``
     - Compiler unrolls inner loops; ~1 µs/op overhead is hard to beat for tiny sizes
   * - Compile-time sizes, larger matrices, tensor-core hardware
     - ``glass::nvidia::gemm<float, M, N, K>(...)``
     - cuBLASDx generates SM-specific tensor-core code
   * - Compile-time sizes inside a kernel using a different thread count
     - ``glass::nvidia::gemm<float, M, N, K, TC>(...)`` with ``DEFINE_NVIDIA_GEMM_BLOCKDIM(M,N,K,TC)``
     - Pins cuBLASDx's ``BlockDim<TC,1,1>``; lets you launch with any thread count ≥ TC
   * - Need a transposed B / row-major storage in the NVIDIA path
     - ``glass::nvidia::gemm<...,LA,LB,LC>`` with ``DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT(...)``
     - cuBLASDx Arrangement; no SIMT fallback needed
   * - Linear solve ``Mx = b`` for SPD ``M``
     - ``glass::nvidia::posv<float, N, NRHS>(...)``
     - cuSOLVERDx fused factor + solve; faster than chol+trsm at N ≥ 8
   * - General linear solve (non-SPD)
     - ``glass::nvidia::gesv_no_pivot<float, N, NRHS>(...)``
     - cuSOLVERDx LU + solve
   * - Least-squares / over- or under-determined
     - ``glass::nvidia::gels<float, M, N, NRHS>(...)``
     - cuSOLVERDx QR (or LQ) + solve
   * - ``BATCH`` independent GEMMs of the same shape, amortize launch
     - ``glass::nvidia::gemm_batched<...,BATCH,TC>``
     - Single block, all batches active via ``threadIdx.y``

When **not** to use ``glass::nvidia::``:

* Sizes only known at runtime (the templates require compile-time ``M``, ``N``,
  ``K``).
* You can't add a ``DEFINE_NVIDIA_GEMM*`` macro for the size you need (the macro
  instantiation cost grows fast if you want every conceivable triple).
* You're on an SM cuBLASDx doesn't tune for — it falls back to a generic config,
  and the pure-SIMT compile-time path is often competitive there.

The ``glass::nvidia::gemm<>`` / ``gemv<>`` / ``row_strided_*`` /
``gemm_batched_1d<>`` primary templates **auto-dispatch**: small shapes route to
SIMT automatically without any DEFINE macro. See
:doc:`../concepts/backend_dispatch` for the full decision logic.

Next steps
----------

* :doc:`installation` — set up the headers and the optional MathDx backend.
* :doc:`../tutorials/quickstart` — a minimal end-to-end kernel.
* :doc:`../concepts/index` — backend dispatch, ``TRAILING_SYNC``, tuning, and
  batched-1D APIs.
