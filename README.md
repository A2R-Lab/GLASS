# GLASS

**GPU Linear Algebra for Single-block Systems** — a header-only CUDA library of BLAS-like device functions designed for use within a single thread block.

## Overview

GLASS functions are `__device__` helpers that operate on data in shared or device memory. Every function assumes it runs within **one CUDA block** — the caller is responsible for launching one block per independent data item. This design enables composable GPU kernels for applications like model-predictive control and rigid-body dynamics.

Two API styles are provided for every function:
- **`glass::` (cooperative groups)** — takes an optional `cgrps::thread_group g = cgrps::this_thread_block()` argument for flexible sub-warp tiling.
- **`glass::simple::` (threadIdx-based)** — no cooperative groups dependency; thread rank/size derived directly from `threadIdx.{x,y,z}` and `blockDim.{x,y,z}`.

Reduction operations additionally offer `glass::simple::low_memory::` (no scratch, thread 0 accumulates) and `glass::simple::high_speed::` (warp-shuffle + shared-memory inter-warp reduction) variants.

## Usage

```cpp
#include "glass.cuh"

__global__ void my_kernel(float* x, float* y, float alpha, int n) {
    // y = alpha * x + y  (all threads in this block cooperate)
    glass::axpy(n, alpha, x, y);
    // or without cooperative groups:
    glass::simple::axpy(n, alpha, x, y);
}
```

Launch with one block per data item:
```cpp
my_kernel<<<num_items, 256>>>(x, y, 1.5f, n);
```

## Requirements

- CUDA 11.0+ with cooperative groups support

| Component | C++ Standard |
|-----------|-------------|
| GLASS library (`glass.cuh`, `src/**`) | C++14 (`nvcc -std=c++14`) |
| Test suite (`test/cuda/*.cu`) | C++14 |
| Benchmark suite (`bench/*.cu`) | **C++17** (required by cuBLASDx) |

## API Reference

### L1 — Vector Operations

| Function | Description | NumPy equivalent | Scratch |
|----------|-------------|------------------|---------|
| `axpy(n, alpha, x, y)` | `y = alpha*x + y` | `y += alpha*x` | none |
| `axpy(n, alpha, x, y, z)` | `z = alpha*x + y` | `z = alpha*x + y` | none |
| `axpby(n, alpha, x, beta, y, z)` | `z = alpha*x + beta*y` | `z = alpha*x + beta*y` | none |
| `copy(n, x, y)` | `y = x` | `y = x.copy()` | none |
| `copy(n, alpha, x, y)` | `y = alpha*x` | `y = alpha*x` | none |
| `scal(n, alpha, x)` | `x = alpha*x` (in-place) | `x *= alpha` | none |
| `scal(n, alpha, x, y)` | `y = alpha*x` | `y = alpha*x` | none |
| `swap(n, x, y)` | swap `x` and `y` | — | none |
| `dot(n, x, y)` | `y[0] = x·y` (in-place) | `np.dot(x,y)` | none (uses `y`) |
| `reduce(n, x, g)` | `x[0] = sum(x)` (in-place) | `np.sum(x)` | none |
| `l2norm(n, x)` | `x[0] = ‖x‖₂` (in-place) | `np.linalg.norm(x)` | none |
| `infnorm(n, x)` | `x[0] = ‖x‖∞` (in-place) | `np.max(np.abs(x))` | none |
| `asum(n, x, out)` | `out[0] = Σ|xᵢ|` | `np.sum(np.abs(x))` | `n*sizeof(T)` |
| `vector_norm(n, a, out)` | `out[0] = ‖a‖₂` | `np.linalg.norm(a)` | `n*sizeof(T)` |
| `clip(n, x, l, u)` | `x = clamp(x, l, u)` | `np.clip(x, l, u)` | none |
| `set_const(n, alpha, x)` | `x = [alpha, …]` | `np.full(n, alpha)` | none |
| `loadIdentity(n, A)` | `A = I_n` (column-major) | `np.eye(n)` | none |
| `addI(n, A, alpha)` | `A += alpha*I` | `A += alpha*np.eye(n)` | none |
| `transpose(N, M, a, b)` | `b = aᵀ` (col-major) | `b = a.T` | none |
| `transpose(N, a)` | in-place transpose `N×N` | — | none |
| `elementwise_add(N, a, b, c)` | `c = a + b` | `c = a + b` | none |
| `elementwise_sub(N, a, b, c)` | `c = a - b` | `c = a - b` | none |
| `elementwise_mult(N, a, b, c)` | `c = a ⊙ b` | `c = a * b` | none |
| `elementwise_abs(N, a, b)` | `b = |a|` | `b = np.abs(a)` | none |
| `elementwise_max(N, a, b, c)` | `c = max(a,b)` | `np.maximum(a,b)` | none |
| `elementwise_min(N, a, b, c)` | `c = min(a,b)` | `np.minimum(a,b)` | none |
| `prefix_sum_exclusive(x, out, n)` | exclusive scan | — | none |
| `prefix_sum_inclusive(x, out, n)` | inclusive scan | `np.cumsum(x)` | none |

### L2 — Matrix-Vector Operations

Matrices default to **column-major** order. Pass `ROW_MAJOR=true` to use row-major (C-style) layout.

| Function | Description | NumPy equivalent |
|----------|-------------|------------------|
| `gemv<T>(m, n, alpha, A, x, beta, y)` | `y = alpha*A*x + beta*y` (col-major A) | `y = alpha*A@x + beta*y` |
| `gemv<T,true>(m, n, alpha, A, x, beta, y)` | `y = alpha*Aᵀ*x + beta*y` | `y = alpha*A.T@x + beta*y` |
| `gemv<T,false,true>(m, n, alpha, A, x, beta, y)` | same, but A is row-major | `y = alpha*A@x + beta*y` |
| `gemv_ex<T,TRANSPOSE,ROW_MAJOR_A>(...)` | per-matrix layout control | |
| `ger(m, n, alpha, x, y, A)` | `A += alpha*x*yᵀ` | `A += alpha*np.outer(x,y)` |

### L3 — Matrix Operations

Matrices default to **column-major** order. Use the `ROW_MAJOR` template parameter for row-major (C-style) layout.

| Function | Description | NumPy equivalent | Scratch |
|----------|-------------|------------------|---------|
| `gemm<T>(m,n,k, alpha, A, B, beta, C, g)` | `C = alpha*A*B + beta*C` (col-major) | `C = alpha*A@B + beta*C` | none |
| `gemm<T,true>(m,n,k, alpha, A, B, beta, C, g)` | `C = alpha*A*Bᵀ + beta*C` | — | none |
| `gemm<T,false,true>(m,n,k, alpha, A, B, beta, C, g)` | same, all matrices row-major | `C = alpha*A@B + beta*C` | none |
| `gemm_ex<T,TRANSPOSE_B,ROW_A,ROW_B,ROW_C>(...)` | per-matrix layout control | | none |
| `simple::gemm_tiled<T,TILE>(m,n,k, alpha, A, B, beta, C, s_A, s_B)` | tiled gemm using shared memory | | `(m*TILE + TILE*k)*sizeof(T)` |
| `simple::gemm_dispatch<T>(m,n,k, alpha, A, B, beta, C, s_A, s_B)` | auto-selects tiled or plain | | see below |
| `invertMatrix(n, A, s_temp)` | `A = A⁻¹` in-place (Gauss-Jordan) | `np.linalg.inv(A)` | `(2n+1)*sizeof(T)` |
| `cholDecomp_InPlace(n, A)` | Cholesky `A → L` (lower triangular) | `np.linalg.cholesky(A)` | none |
| `trsm(n, L, b)` | Solve `Lx=b` in-place (forward substitution) | `scipy.linalg.solve_triangular(L,b,lower=True)` | none |

#### Auto-dispatch: `gemm_dispatch`

`glass::simple::gemm_dispatch` selects tiled or plain gemm at runtime based on whether scratch pointers are provided and `m*k ≤ blockDim`:

```cpp
// Device: pass scratch pointers when available
extern __shared__ float scratch[];
float *s_A = scratch, *s_B = scratch + m * 8;
glass::simple::gemm_dispatch(m, n, k, 1.f, A, B, 0.f, C,
                              (smem > 0) ? s_A : nullptr,
                              (smem > 0) ? s_B : nullptr);
```

Use the host helper to compute required scratch size at kernel launch:
```cpp
#include "glass.cuh"
// Returns (m*TILE + TILE*k)*sizeof(T) if tiling is warranted, else 0
std::size_t smem = glass_gemm_dispatch_smem(m, k, /*block_threads=*/256);
my_kernel<<<grid, 256, smem>>>(m, n, k, A, B, C);
```

Tiling is used when `s_A != nullptr` and `m*k ≤ block_threads` (default threshold: `m < 32`).

#### Row-Major Storage Order

All three template bools are compile-time constants — the compiler eliminates unused index branches with zero runtime cost.

```cpp
// Column-major (default): A[row + col*m]
glass::simple::gemm<float>(m, n, k, 1.f, A, B, 0.f, C);

// Row-major (all matrices): A[row*cols + col]
glass::simple::gemm<float, false, true>(m, n, k, 1.f, A, B, 0.f, C);

// Mixed layouts (row-major A and C, column-major B):
glass::simple::gemm_ex<float, false, true, false, true>(m, n, k, 1.f, A, B, 0.f, C);
```

The same `ROW_MAJOR` template parameter is available on `glass::simple::gemv` and `glass::simple::gemv_ex`.

### Scratch memory for `glass::simple` reduction variants

| Variant | Scratch required |
|---------|-----------------|
| `glass::simple::low_memory::reduce/dot/l2norm/asum` | None (thread 0 accumulates) |
| `glass::simple::high_speed::reduce/dot/l2norm/asum` | `ceil(blockDim/32) * sizeof(T)` bytes |

## Running Tests

```bash
cd test
pip install -r requirements.txt
pytest -v
```

The first run compiles the CUDA test binaries automatically using `nvcc`. Subsequent runs skip recompilation unless source files have changed (cached by content hash). Compiled binaries are placed in `test/build/`.

To run a specific level:
```bash
pytest test_l1.py -v
pytest test_l1.py -k "simple"      # simple-namespace variants only
pytest test_l1.py -k "simple_hs"   # high_speed warp-shuffle variants
```

## Benchmarks

The benchmark suite compares GLASS against block-level CUDA library baselines:

| Level | GLASS | Baseline |
|-------|-------|----------|
| L1 reduce/dot/l2norm | `glass::simple::high_speed::*` | CUB `BlockReduce` |
| L2 gemv | `glass::simple::gemv` | cuBLASDx |
| L3 gemm | `glass::simple::gemm` (plain + tiled) | cuBLASDx |

Both CUB and cuBLASDx are **required** to run benchmarks. See [`bench/INSTALL.md`](bench/INSTALL.md) for installation instructions.

```bash
# Set MATHDX_ROOT to your MathDx installation, then:
python3 bench/run_bench.py

# Custom iteration count (default: 10000):
python3 bench/run_bench.py --iters 50000
```

Timing uses the GRiD pattern — the iteration loop runs inside the kernel to measure raw compute with no launch-overhead amortization. Results are printed as a Markdown table and saved to `bench/results/bench_<hostname>.json`.

> **Note**: Benchmark binaries require C++17 (cuBLASDx dependency). The GLASS library itself (`glass.cuh` and all `src/**`) remains C++14 compatible.

## Notes

- Matrices default to **column-major** (Fortran) order, consistent with cuBLAS. Use `ROW_MAJOR=true` template parameter for row-major.
- `dot` and reduction variants modify the input array in-place (result in `x[0]`).
- `cholDecomp_InPlace` only fills the **lower triangle** of the matrix; the upper triangle retains input values.
- `gemm` with `TRANSPOSE_B=true` currently requires B to be square (n×n) — see `gemm.cuh`.
