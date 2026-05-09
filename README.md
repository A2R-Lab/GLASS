# GLASS

**GPU Linear Algebra for Single-block Systems** — a header-only CUDA library of BLAS-like device functions designed for use within a single thread block.

## Overview

GLASS functions are `__device__` helpers that operate on data in shared or device memory. Every function assumes it runs within **one CUDA block** — the caller is responsible for launching one block per independent data item. This design enables composable GPU kernels for applications like model-predictive control and rigid-body dynamics.

Three namespaces are provided:

| Namespace | Thread source | Header |
|-----------|--------------|--------|
| `glass::` | `threadIdx.{x,y,z}` / `blockDim.*` — no cgrps dep | `glass.cuh` |
| `glass::cgrps::` | `g.thread_rank()` / `g.size()` — cooperative groups | `glass-cgrps.cuh` |
| `glass::nvidia::` | CUB (L1) or cuBLASDx (L2/L3) — compile-time sizes only | `glass-nvidia.cuh` |

Both `glass::` and `glass::cgrps::` offer **runtime** (size as function arg) and **compile-time** (size as template arg) overloads for every function.

Reduction operations additionally offer `glass::low_memory::` (no scratch, thread 0 accumulates) and `glass::high_speed::` (warp-shuffle + shared-memory inter-warp reduction) sub-namespaces.

---

## Quick Start

```cpp
#include "glass.cuh"

__global__ void my_kernel(float* A, float* B, float* C, int m, int n, int k) {
    // Runtime size: all threads in the block cooperate
    glass::gemm(m, n, k, 1.f, A, B, 0.f, C);
}
```

Launch with one block per data item:
```cpp
my_kernel<<<num_items, 256>>>(A, B, C, m, n, k);
```

---

## Usage Examples

### 1. Runtime sizes — default `glass::` (threadIdx-based)

```cpp
#include "glass.cuh"

__global__ void k(float* A, float* B, float* C, int m, int n, int k) {
    glass::gemm(m, n, k, 1.f, A, B, 0.f, C);         // GEMM
    glass::gemv(m, n, 1.f, A, B, 0.f, C);             // GEMV
    glass::axpy(n, 1.5f, A, B);                        // y = alpha*x + y
}
```

### 2. Compile-time sizes — loop unrolling via template args

```cpp
#include "glass.cuh"

__global__ void k(float* A, float* B, float* C) {
    // Sizes baked in as template params — compiler can unroll loops
    glass::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);
    glass::gemv<float, 6, 6>(1.f, A, B, 0.f, C);
    glass::axpy<float, 36>(1.5f, A, B);
}
```

### 3. Cooperative groups — sub-block tiling with `glass::cgrps::`

```cpp
#include "glass-cgrps.cuh"
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

__global__ void k(float* A, float* B, float* C) {
    // Whole block (default)
    glass::cgrps::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);

    // Warp-level tiling: each warp independently computes a 4×4×4 GEMM
    auto warp = cgrps::tiled_partition<32>(cgrps::this_thread_block());
    glass::cgrps::gemm<float, 4, 4, 4>(1.f, A, B, 0.f, C, warp);
}
```

### 4. High-speed reductions (warp-shuffle)

```cpp
#include "glass.cuh"

__global__ void k(float* x, int n) {
    // Scratch: ceil(blockDim.x / 32) * sizeof(float) bytes
    extern __shared__ float scratch[];
    glass::high_speed::reduce(n, x, scratch);     // result in x[0]
    glass::high_speed::l2norm(n, x, scratch);     // ‖x‖₂ in x[0]
}
```

### 5. NVIDIA-optimized path (`glass::nvidia::`)

```cpp
#include "glass-nvidia.cuh"

// Host: query required smem and thread count (all constexpr)
constexpr auto smem    = glass::nvidia::gemm_smem_size<float, 6, 6, 6>();
constexpr auto threads = glass::nvidia::gemm_threads<float, 6, 6, 6>();

__global__ void k(float* A, float* B, float* C) {
    glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C,
        reinterpret_cast<char*>(smem_ptr));
}

// Launch:
k<<<1, threads, smem>>>(dA, dB, dC);
```

### 6. Tiled GEMM (scratch in shared memory)

```cpp
#include "glass.cuh"

__global__ void k(float* A, float* B, float* C, int m, int n, int k) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_B = smem + m * 8;
    glass::gemm_tiled<float, 8>(m, n, k, 1.f, A, B, 0.f, C, s_A, s_B);
}

// Host:
size_t smem = (m * 8 + 8 * k) * sizeof(float);
k<<<1, 256, smem>>>(A, B, C, m, n, k);
```

---

## Requirements

| Component | C++ Standard |
|-----------|-------------|
| `glass.cuh`, `glass-cgrps.cuh` | C++17 (`nvcc -std=c++17`) |
| `glass-nvidia.cuh` | C++17 + CUB + cuBLASDx |
| Test suite | C++17 |
| Benchmark suite | C++17 |

CUB is bundled with CUDA 11+. cuBLASDx requires a separate installation (see [`bench/INSTALL.md`](bench/INSTALL.md)).

---

## API Reference

### L1 — Vector Operations

All L1 functions accept `uint32_t n` (runtime) **or** `<T, N>` template args (compile-time).

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
| `dot(n, x, y)` | `y[0] = x·y` (in-place, uses `y` as scratch) | `np.dot(x,y)` | none |
| `reduce(n, x)` | `x[0] = sum(x)` (in-place) | `np.sum(x)` | none |
| `l2norm(n, x)` | `x[0] = ‖x‖₂` (in-place, destructive) | `np.linalg.norm(x)` | none |
| `infnorm(n, x)` | `x[0] = ‖x‖∞` (in-place) | `np.max(np.abs(x))` | none |
| `asum(n, x, out)` | `out[0] = Σ|xᵢ|` | `np.sum(np.abs(x))` | `n*sizeof(T)` |
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

#### Reduction sub-namespaces

```cpp
glass::reduce(n, x);                         // default: halving reduce
glass::low_memory::reduce(n, x);             // thread 0 accumulates; no scratch
glass::high_speed::reduce(n, x, scratch);    // warp-shuffle; faster for large n
```

Scratch required for `high_speed`: `ceil(blockDim.x / 32) * sizeof(T)` bytes.

#### `glass::nvidia::` L1 (CUB BlockReduce)

```cpp
#include "glass-nvidia.cuh"
extern __shared__ float scratch[];  // sizeof(cub::BlockReduce<T,THREADS>::TempStorage)

glass::nvidia::reduce<float, N, THREADS>(x, scratch);
glass::nvidia::dot<float, N, THREADS>(x, y, out, scratch);
glass::nvidia::l2norm<float, N, THREADS>(x, out, scratch);

// Query scratch size (host-callable):
constexpr auto bytes = glass::nvidia::reduce_smem_size<float, THREADS>();
```

---

### L2 — Matrix-Vector Operations

Matrices default to **column-major** order. Pass `ROW_MAJOR=true` to use row-major (C-style).

| Function | Description | NumPy equivalent |
|----------|-------------|------------------|
| `gemv<T>(m, n, alpha, A, x, beta, y)` | `y = alpha*A*x + beta*y` (col-major A) | `y = alpha*A@x + beta*y` |
| `gemv<T,true>(m, n, alpha, A, x, beta, y)` | `y = alpha*Aᵀ*x + beta*y` | `y = alpha*A.T@x + beta*y` |
| `gemv<T,false,true>(m, n, alpha, A, x, beta, y)` | same, but A is row-major | `y = alpha*A@x + beta*y` |
| `gemv_ex<T,TRANSPOSE,ROW_MAJOR_A>(...)` | per-matrix layout control | |
| `ger(m, n, alpha, x, y, A)` | `A += alpha*x*yᵀ` | `A += alpha*np.outer(x,y)` |

Compile-time overloads omit the `m, n` args:
```cpp
glass::gemv<float, 6, 6>(1.f, A, x, 0.f, y);           // runtime: glass::gemv(6, 6, 1.f, A, x, 0.f, y)
glass::cgrps::gemv<float, 6, 6>(1.f, A, x, 0.f, y, g); // with explicit group
```

---

### L3 — Matrix Operations

Matrices default to **column-major** order.

| Function | Description | NumPy equivalent | Scratch |
|----------|-------------|------------------|---------|
| `gemm<T>(m,n,k, alpha, A, B, beta, C)` | `C = alpha*A*B + beta*C` (col-major) | `C = alpha*A@B + beta*C` | none |
| `gemm<T,true>(m,n,k, alpha, A, B, beta, C)` | `C = alpha*A*Bᵀ + beta*C` | — | none |
| `gemm<T,false,true>(m,n,k, alpha, A, B, beta, C)` | same, all matrices row-major | | none |
| `gemm_ex<T,TRANSPOSE_B,ROW_A,ROW_B,ROW_C>(...)` | per-matrix layout control | | none |
| `gemm_tiled<T,TILE>(m,n,k, alpha, A, B, beta, C, s_A, s_B)` | tiled gemm using shared memory | | `(m*TILE + TILE*k)*sizeof(T)` |
| `gemm_dispatch<T>(m,n,k, alpha, A, B, beta, C, s_A, s_B)` | auto-selects tiled or plain | | see below |
| `invertMatrix(n, A, s_temp)` | `A = A⁻¹` in-place (Gauss-Jordan) | `np.linalg.inv(A)` | `(2n+1)*sizeof(T)` |
| `cholDecomp_InPlace(n, A)` | Cholesky `A → L` (lower triangular) | `np.linalg.cholesky(A)` | none |
| `trsm(n, L, b)` | Solve `Lx=b` in-place (forward substitution) | `scipy.linalg.solve_triangular(L,b,lower=True)` | none |

Compile-time overloads omit the `m, n, k` args:
```cpp
glass::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C);
glass::cgrps::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, g);
```

#### Auto-dispatch: `gemm_dispatch`

`glass::gemm_dispatch` selects tiled or plain gemm at runtime based on whether scratch pointers are provided and `m*k ≤ blockDim`:

```cpp
extern __shared__ float scratch[];
float *s_A = scratch, *s_B = scratch + m * 8;
glass::gemm_dispatch(m, n, k, 1.f, A, B, 0.f, C,
                     (smem > 0) ? s_A : nullptr,
                     (smem > 0) ? s_B : nullptr);
```

Use the host helper to compute required scratch at launch:
```cpp
#include "glass.cuh"
std::size_t smem = glass_gemm_dispatch_smem(m, k, /*block_threads=*/256);
my_kernel<<<grid, 256, smem>>>(m, n, k, A, B, C);
```

#### Row-Major Storage Order

```cpp
// Column-major (default): A[row + col*m]
glass::gemm<float>(m, n, k, 1.f, A, B, 0.f, C);

// Row-major (all matrices): A[row*cols + col]
glass::gemm<float, false, true>(m, n, k, 1.f, A, B, 0.f, C);

// Mixed layouts (row-major A and C, column-major B):
glass::gemm_ex<float, false, true, false, true>(m, n, k, 1.f, A, B, 0.f, C);
```

---

### `glass::nvidia::` L2/L3 (cuBLASDx)

cuBLASDx computes at warp/tensor-core level and requires compile-time matrix sizes. Include `glass-nvidia.cuh` (which includes `cublasdx.hpp`) to get pre-instantiated sizes.

**Pre-instantiated sizes** (square): `4, 6, 8, 12, 14, 24, 64`

```cpp
#include "glass-nvidia.cuh"

// Host: query required smem and thread count
constexpr auto smem    = glass::nvidia::gemm_smem_size<float, 6, 6, 6>();
constexpr auto threads = glass::nvidia::gemm_threads<float, 6, 6, 6>();

__global__ void k(float* A, float* B, float* C) {
    extern __shared__ __align__(16) char smem_buf[];
    glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, smem_buf);
}

// Launch:
k<<<1, threads, smem>>>(dA, dB, dC);
```

To add a custom size, call `DEFINE_NVIDIA_GEMM(M, N, K)` in your `.cu` file **inside `namespace glass::nvidia`** before first use:

```cpp
#include "glass-nvidia.cuh"
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM(16, 16, 16)
    DEFINE_NVIDIA_GEMV(16, 16)
} }
```

---

## Running Tests

```bash
cd test
pip install -r requirements.txt
pytest -v
```

The first run compiles the CUDA test binaries using `nvcc`. Subsequent runs skip recompilation unless source files have changed (cached by content hash). Compiled binaries are placed in `test/build/`.

```bash
pytest test_l1.py -v
pytest test_l1.py -k "simple"      # glass:: threadIdx variants only
pytest test_l1.py -k "cg"          # glass::cgrps:: cooperative groups variants
pytest test_l1.py -k "simple_hs"   # high_speed warp-shuffle variants
```

---

## Benchmarks

The benchmark suite compares GLASS against block-level CUDA library baselines:

| Level | GLASS | Baseline |
|-------|-------|----------|
| L1 reduce/dot/l2norm | `glass::high_speed::*` | CUB `BlockReduce` |
| L2 gemv | `glass::gemv` | cuBLASDx |
| L3 gemm | `glass::gemm` (plain + tiled) | cuBLASDx |

CUB is bundled with CUDA 11+. cuBLASDx requires a separate installation (see [`bench/INSTALL.md`](bench/INSTALL.md)).

```bash
# Set MATHDX_ROOT to your MathDx installation, then:
python3 bench/run_bench.py

# Custom iteration count (default: 10000):
python3 bench/run_bench.py --iters 50000
```

Timing uses the GRiD pattern — the iteration loop runs inside the kernel to amortize launch overhead. Results are printed as a Markdown table and saved to `bench/results/bench_<hostname>.json`.

---

## Notes

- Matrices default to **column-major** (Fortran) order, consistent with cuBLAS. Use `ROW_MAJOR=true` template parameter for row-major.
- `dot` and reduction variants modify the input array in-place (result in `x[0]`); `l2norm` squares elements before reducing.
- `cholDecomp_InPlace` only fills the **lower triangle** of the matrix; the upper triangle retains input values.
- `gemm` with `TRANSPOSE_B=true` currently requires B to be square (n×n).
- `glass::nvidia::` functions require exactly the thread count returned by `gemm_threads<T,M,N,K>()`. Launching with the wrong thread count produces incorrect results.
