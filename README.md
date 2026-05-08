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
- C++14 or later (`nvcc -std=c++14`)

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

All matrices are stored in **column-major** order.

| Function | Description | NumPy equivalent |
|----------|-------------|------------------|
| `gemv<T>(m, n, alpha, A, x, beta, y)` | `y = alpha*A*x + beta*y` | `y = alpha*A@x + beta*y` |
| `gemv<T,true>(m, n, alpha, A, x, beta, y)` | `y = alpha*Aᵀ*x + beta*y` | `y = alpha*A.T@x + beta*y` |
| `ger(m, n, alpha, x, y, A)` | `A += alpha*x*yᵀ` | `A += alpha*np.outer(x,y)` |

### L3 — Matrix Operations

| Function | Description | NumPy equivalent | Scratch |
|----------|-------------|------------------|---------|
| `gemm<T>(m,n,k, alpha, A, B, beta, C, g)` | `C = alpha*A*B + beta*C` | `C = alpha*A@B + beta*C` | none |
| `gemm<T,true>(m,n,k, alpha, A, B, beta, C, g)` | `C = alpha*A*Bᵀ + beta*C` (B must be n×n) | — | none |
| `invertMatrix(n, A, s_temp)` | `A = A⁻¹` in-place (Gauss-Jordan) | `np.linalg.inv(A)` | `(2n+1)*sizeof(T)` |
| `cholDecomp_InPlace(n, A)` | Cholesky `A → L` (lower triangular) | `np.linalg.cholesky(A)` | none |
| `trsm(n, L, b)` | Solve `Lx=b` in-place (forward substitution) | `scipy.linalg.solve_triangular(L,b,lower=True)` | none |

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

## Notes

- All matrices are in **column-major** (Fortran) order, consistent with cuBLAS.
- `dot` and reduction variants modify the input array in-place (result in `x[0]`).
- `cholDecomp_InPlace` only fills the **lower triangle** of the matrix; the upper triangle retains input values.
- `gemm` with `TRANSPOSE_B=true` currently requires B to be square (n×n) — see `gemm.cuh`.
