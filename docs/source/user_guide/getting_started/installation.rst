Installation
============

GLASS is a **header-only** CUDA library. There is nothing to build or link for
the pure-SIMT path — you add the repository root to your include path and
``#include`` the umbrella header. Only the optional ``glass::nvidia::`` backend
pulls in external dependencies (NVIDIA MathDx).

Header-only include model
-------------------------

Clone (or vendor) the repository, add its root to your compiler's include path,
and include the header you need:

.. code-block:: cpp

   #include "glass.cuh"        // pure-SIMT glass:: API
   // or
   #include "glass-cgrps.cuh"  // cooperative-groups glass::cgrps:: API
   // or
   #include "glass-nvidia.cuh" // CUB / cuBLASDx / cuSOLVERDx backend

A minimal ``nvcc`` invocation for the pure-SIMT path:

.. code-block:: bash

   nvcc -std=c++17 -I /path/to/GLASS -arch=sm_86 my_kernel.cu -o my_kernel

CUDA Toolkit requirement
------------------------

* **CUDA Toolkit 11.0 or newer.** CUB (used by the ``glass::nvidia::`` L1
  reductions) ships bundled with every CUDA 11+ install — no separate download.
* **C++17** (``nvcc -std=c++17``). The pure-SIMT core compiles cleanly under
  C++17; the ``glass::nvidia::`` path *requires* C++17 (mandated by cuBLASDx /
  cuSOLVERDx).

.. list-table:: Per-component requirements
   :header-rows: 1
   :widths: 40 25 35

   * - Component
     - C++ Standard
     - Optional deps
   * - ``glass.cuh``, ``glass-cgrps.cuh``
     - C++17
     - —
   * - ``glass-nvidia.cuh`` (L1 only)
     - C++17
     - CUB (bundled with CUDA 11+)
   * - ``glass-nvidia.cuh`` (L2/L3 GEMM/GEMV/batched)
     - C++17 + ``--expt-relaxed-constexpr``
     - cuBLASDx
   * - ``glass-nvidia.cuh`` (LAPACK)
     - C++17 + ``--expt-relaxed-constexpr`` + cuSOLVERDx link flags
     - cuSOLVERDx

Optional: MathDx setup for the ``glass::nvidia::`` backend
----------------------------------------------------------

The ``glass::nvidia::`` GEMM/GEMV/batched and LAPACK wrappers route to NVIDIA's
device-side libraries cuBLASDx and cuSOLVERDx, which ship together in the
**MathDx** package. They are **not** distributed via ``apt`` — installation is a
manual download from the NVIDIA Developer portal.

.. list-table:: MathDx libraries
   :header-rows: 1
   :widths: 20 50 30

   * - Library
     - Used by
     - Header-only?
   * - CUB
     - ``glass::nvidia::reduce`` / ``dot`` / ``nrm2``
     - Yes (bundled with CUDA)
   * - cuBLASDx
     - ``gemv``, ``gemm``, batched GEMM
     - Yes
   * - cuSOLVERDx
     - LAPACK (Cholesky / TRSM / posv / ...)
     - **No** — links a precompiled device fatbin

Download
~~~~~~~~

1. Go to https://developer.nvidia.com/cublasdx-downloads (a free NVIDIA
   Developer account is required).
2. Choose **MathDx for CUDA 12, Linux x86_64** (``.tar.gz``). Version 25.12.x
   or later is recommended — that is the version the GLASS wrappers are tested
   against.

Install
~~~~~~~

.. code-block:: bash

   # Extract to /opt (or any directory you prefer)
   tar -xzf MathDx_*.tar.gz -C /opt

   # Confirm the version directory name
   ls /opt/nvidia/mathdx/

   # Set the environment variable (add to ~/.bashrc to persist)
   export MATHDX_ROOT=/opt/nvidia/mathdx/25.12   # adjust version as needed

Verify
~~~~~~

.. code-block:: bash

   ls $MATHDX_ROOT/include/cublasdx.hpp           # cuBLASDx header
   ls $MATHDX_ROOT/include/cusolverdx.hpp         # cuSOLVERDx header
   ls $MATHDX_ROOT/include/cusolverdx_io.hpp      # cuSOLVERDx IO helpers
   ls $MATHDX_ROOT/lib/libcusolverdx.a            # cuSOLVERDx device library
   echo $MATHDX_ROOT

Linking notes
~~~~~~~~~~~~~

cuBLASDx is **header-only** — ``#include <cublasdx.hpp>`` is enough; no extra
link flags.

cuSOLVERDx is **not** header-only: part of the implementation ships as a
precompiled device fatbin (``libcusolverdx.a``). Any ``.cu`` file that uses
cuSOLVERDx must add:

.. code-block:: bash

   -rdc=true                      # relocatable device code (required for device link)
   -dlto                          # link-time optimization (the fatbin is LTO-compiled)
   -L$MATHDX_ROOT/lib
   -lcusolverdx                   # the device library
   -lcublas -lcusolver -lcudart   # host-side CUDA libs

The benchmark driver ``bench/run_bench.py`` adds these automatically when
``cusolverdx.hpp`` is present.

Required nvcc flags and known issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* cuBLASDx headers require ``--expt-relaxed-constexpr`` (otherwise many
  ``#20013-D`` constexpr/``__host__``/``__device__`` warnings fire).
* **CUDA 12.9 cuBLASDx miscompilation:** apply ``-Xptxas -O1`` (this flag also
  doubles as anti-DSE for bench loops). If results are still wrong, add
  ``-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT``.
* **cuSOLVERDx NVBUG 5288270** affects SM 1200 (Blackwell consumer) for some
  real-precision configs on CUDA ≤ 13.0. Define
  ``CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT`` to silence the static asserts, but
  verify correctness on your target arch first.

Next steps
----------

* :doc:`library_overview` — the single-block model and the call surfaces.
* :doc:`../tutorials/quickstart` — a minimal end-to-end kernel.
* :doc:`../tutorials/using_nvidia_backend` — a full compile-and-call walkthrough
  for the MathDx backend.

See :doc:`../tutorials/using_nvidia_backend` for a full compile-and-call
walkthrough.
