# Benchmark Dependencies

The GLASS benchmark suite uses up to three external libraries. CUB is mandatory
(comes with CUDA); cuBLASDx and cuSOLVERDx are optional but unlock most of the
benches. All three are auto-detected by `bench/run_bench.py`.

| Library | Required for | Header-only? |
|---------|-------------|--------------|
| CUB    | `bench_reduce` (CUB baseline + `glass::nvidia::reduce` variant) | Yes (bundled with CUDA) |
| cuBLASDx | `bench_gemv`, `bench_gemm`, `bench_blockdim`, `bench_gemm_batched`, `bench_lapack` | Yes |
| cuSOLVERDx | `bench_lapack` (Cholesky / TRSM / posv / etc.) | **No** — links a precompiled device fatbin |

---

## 1. CUB (block-level reductions — L1 baseline)

CUB is included with every CUDA Toolkit installation (11.0+). No extra installation
is needed. The header is at `<cub/cub.cuh>`.

**Verify:**
```bash
ls /usr/local/cuda/include/cub/cub.cuh
```

---

## 2. cuBLASDx + cuSOLVERDx (NVIDIA MathDx)

Both libraries ship together in NVIDIA's **MathDx** package. They are **not**
distributed via apt — installation is a manual download from the NVIDIA Developer
portal.

### Download

1. Go to: https://developer.nvidia.com/cublasdx-downloads
   (Free NVIDIA Developer account required.)

2. Choose: **MathDx for CUDA 12, Linux x86_64** (`.tar.gz` format).
   Version 25.12.x or later is recommended (this is the version the GLASS
   wrappers are tested against).

### Install

```bash
# Extract to /opt (or any directory you prefer)
tar -xzf MathDx_*.tar.gz -C /opt

# Confirm the version directory name
ls /opt/nvidia/mathdx/

# Set the environment variable (add to ~/.bashrc to persist)
export MATHDX_ROOT=/opt/nvidia/mathdx/25.12   # adjust version as needed
```

### Verify

```bash
ls $MATHDX_ROOT/include/cublasdx.hpp           # cuBLASDx header
ls $MATHDX_ROOT/include/cusolverdx.hpp         # cuSOLVERDx header
ls $MATHDX_ROOT/include/cusolverdx_io.hpp      # cuSOLVERDx IO helpers (copy_2d)
ls $MATHDX_ROOT/lib/libcusolverdx.a            # cuSOLVERDx device library
echo $MATHDX_ROOT
```

### Linking notes (cuBLASDx vs cuSOLVERDx)

cuBLASDx is **header-only** — `#include <cublasdx.hpp>` is enough. No extra link flags.

cuSOLVERDx is **NOT header-only**: a portion of the implementation is shipped as a
precompiled device fatbin (`libcusolverdx.a`). Any `.cu` file that uses cuSOLVERDx
must be compiled and linked with:

```
-rdc=true                      # relocatable device code (required for device link)
-dlto                          # link-time optimization (the fatbin is LTO-compiled)
-L$MATHDX_ROOT/lib
-lcusolverdx                   # the device library
-lcublas -lcusolver -lcudart   # host-side CUDA libs
```

`bench/run_bench.py` adds these automatically when `cusolverdx.hpp` is present.
Build times for cuSOLVERDx-linked binaries are noticeably longer than cuBLASDx-only
ones because the device link step has to inline the library kernels.

### Required nvcc flags for cuBLASDx headers

cuBLASDx triggers many `#20013-D` warnings about constexpr/`__host__`/`__device__`
qualifiers without the relaxed-constexpr flag. The bench harness adds:

```
--expt-relaxed-constexpr
```

on every cuBLASDx-enabled compile.

### CUDA 12.9 known issues

1. **cuBLASDx miscompilation bug.** The benchmark harness automatically applies
   the workaround:

   ```
   -Xptxas -O1
   ```

   This same flag also serves as anti-DSE for the bench loops — keep it on.

   If you still see incorrect results, try also:
   ```
   -DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT
   ```

2. **cuSOLVERDx NVBUG 5288270** affects SM 1200 (Blackwell consumer) for some
   real-precision configurations on CUDA ≤ 13.0. The bench harness defines
   `CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT` so the static asserts don't fire.
   Verify correctness on your target arch before relying on cuSOLVERDx in
   production at SM 1200.

---

## C++ Standard

| Component | C++ Standard |
|-----------|-------------|
| GLASS pure-SIMT (`glass.cuh`, `glass-cgrps.cuh`, `src/base/**`, `src/cgrps/**`) | C++14 (compiles cleanly under `-std=c++17`) |
| GLASS NVIDIA path (`glass-nvidia.cuh`, `src/nvidia/**`) | **C++17** (required by cuBLASDx / cuSOLVERDx) |
| Test suite (`test/cuda/*.cu`) | C++14 |
| Benchmark suite (`bench/*.cu`) | **C++17** |

The benchmark harness compiles with `-std=c++17` automatically.

---

## Running Benchmarks

```bash
# Once MATHDX_ROOT is set:
cd /path/to/GLASS
python3 bench/run_bench.py

# With custom iteration count (default: 10000):
python3 bench/run_bench.py --iters 50000

# Skip cuBLASDx (only bench_reduce will run):
python3 bench/run_bench.py --no-cublasdx

# Results are also saved to bench/results/bench_<hostname>.json
```

The script prints which dependencies it found at startup:

```
=== GLASS Benchmark Suite ===
GPU arch: sm_120 (SM1200)
cuBLASDx: enabled (/opt/nvidia/mathdx/25.12)
cuSOLVERDx: enabled
Iterations: 10000
```

If `cuSOLVERDx: disabled` appears, `bench_lapack` is skipped automatically; the
other benches still run.
