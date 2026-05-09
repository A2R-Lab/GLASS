# Benchmark Dependencies

The GLASS benchmark suite requires two external libraries. Both must be installed
before running `bench/run_bench.py`.

---

## 1. CUB (block-level reductions — L1 baseline)

CUB is included with every CUDA Toolkit installation (11.0+). No extra installation
is needed. The header is at `<cub/cub.cuh>`.

**Verify:**
```bash
ls /usr/local/cuda/include/cub/cub.cuh
```

---

## 2. cuBLASDx / NVIDIA MathDx (block-level GEMM/GEMV — L2 and L3 baseline)

cuBLASDx is part of NVIDIA's **MathDx** package. It is **not** distributed via apt —
it requires a manual download from the NVIDIA Developer portal.

### Download

1. Go to: https://developer.nvidia.com/cublasdx-downloads  
   (Free NVIDIA Developer account required.)

2. Choose: **MathDx for CUDA 12, Linux x86_64** (`.tar.gz` format).  
   Version 25.12.x or later is recommended.

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
ls $MATHDX_ROOT/include/cuda/blas/device/cublasdx.hpp   # or cublasdx.hpp
echo $MATHDX_ROOT
```

### CUDA 12.9 known issue

cuBLASDx on CUDA 12.9 may trigger a compiler miscompilation bug. The benchmark
harness automatically applies the workaround flag:

```
-Xptxas -O1
```

If you encounter incorrect results, also try:
```
-DCUBLASDX_IGNORE_NVBUG_5218000_ASSERT
```

---

## C++ Standard Note

| Component | C++ Standard |
|-----------|-------------|
| GLASS library (`glass.cuh`, `src/**`) | C++14 |
| Test suite (`test/cuda/*.cu`) | C++14 |
| **Benchmark suite** (`bench/*.cu`) | **C++17** (required by cuBLASDx) |

The benchmark harness compiles with `-std=c++17` automatically.

---

## Running Benchmarks

```bash
# Once MATHDX_ROOT is set:
cd /path/to/GLASS
python3 bench/run_bench.py

# With custom iteration count (default: 10000):
python3 bench/run_bench.py --iters 50000

# Results also saved to bench/results/bench_<hostname>.json
```
