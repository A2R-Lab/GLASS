# GLASS examples

Minimal, self-contained, **compile-and-run** CUDA programs — one concept each.
Every file is a complete program: a `__global__` kernel that calls a GLASS
device function, plus a `main` that allocates device memory, launches **one
block**, copies the result back, and prints it so you can eyeball correctness.
They are deliberately tiny (~30–60 lines) — the point is clarity, not features.

These are the same snippets shown in the [top-level README](../README.md)
"Usage Examples" section, extracted into standalone files you can actually
build. For the full API surface and the backend-choice guide, read that README.

## Which example shows what

| File | Shows | Backend / deps |
|------|-------|----------------|
| [`01_axpy_simt.cu`](01_axpy_simt.cu) | L1 vector op `axpy` (`y = αx + y`), **runtime size** | pure SIMT — no extra deps |
| [`02_gemm.cu`](02_gemm.cu) | single-block `gemm` (`C = αAB + βC`), runtime + **compile-time** size overloads, column-major | pure SIMT — no extra deps |
| [`03_reduce.cu`](03_reduce.cu) | block reduction: `glass::reduce` and the warp-shuffle `glass::high_speed::reduce` (with scratch) | pure SIMT — no extra deps |
| [`04_cgrps.cu`](04_cgrps.cu) | the **cooperative-groups** variant `glass::cgrps::gemm` (whole-block or warp-tile) | pure SIMT — no extra deps |
| [`05_gemm_dispatch.cu`](05_gemm_dispatch.cu) | `glass::gemm_dispatch` + dynamic shared memory via the `glass_gemm_dispatch_smem` host helper (tiled path) | pure SIMT — no extra deps |
| [`06_nvidia_gemm.cu`](06_nvidia_gemm.cu) | the cuBLASDx-backed `glass::nvidia::gemm` path | **requires NVIDIA MathDx** |
| [`07_warp_ops.cu`](07_warp_ops.cu) | single-warp `glass::warp::` ops (`reduce`, 4×4 `gemm`, SPD `cholDecomp_InPlace`+`trsm`+`trsm_transpose`), launched `<<<1,32>>>` | pure SIMT — no extra deps |

**Examples 01–05 and 07 are pure SIMT** — they build with plain `nvcc` and need
no external libraries. **Only `06_nvidia_gemm.cu` needs MathDx** (cuBLASDx); skip
it if you don't have MathDx installed.

## Building

All examples `#include` the GLASS headers from the repo root. The build commands
below use `-I..` so the relative `#include "glass.cuh"` resolves from this
`examples/` directory; run them **from inside `examples/`**. Pick the `-arch`
that matches your GPU (`sm_75` Turing, `sm_86` Ampere, `sm_89` Ada, `sm_120`
Blackwell, …).

### Pure-SIMT examples (01–05)

```bash
nvcc -std=c++17 -arch=sm_75 -I.. 01_axpy_simt.cu    -o axpy     && ./axpy
nvcc -std=c++17 -arch=sm_75 -I.. 02_gemm.cu         -o gemm     && ./gemm
nvcc -std=c++17 -arch=sm_75 -I.. 03_reduce.cu       -o reduce   && ./reduce
nvcc -std=c++17 -arch=sm_75 -I.. 04_cgrps.cu        -o cgrps    && ./cgrps
nvcc -std=c++17 -arch=sm_75 -I.. 05_gemm_dispatch.cu -o dispatch && ./dispatch
nvcc -std=c++17 -arch=sm_75 -I.. 07_warp_ops.cu     -o warp_ops && ./warp_ops
```

### NVIDIA / cuBLASDx example (06) — requires MathDx

This one needs NVIDIA MathDx (which ships cuBLASDx). Install it and set
`MATHDX_ROOT` first — see [`../bench/INSTALL.md`](../bench/INSTALL.md) for the
download + setup steps. Then:

```bash
nvcc -std=c++17 -arch=sm_86 -I.. \
     -DGLASS_BENCH_CUBLASDX -DSMS=860 \
     --expt-relaxed-constexpr -Xptxas -O1 \
     -I$MATHDX_ROOT/include \
     -I$MATHDX_ROOT/external/cutlass/include \
     06_nvidia_gemm.cu -o nvidia_gemm && ./nvidia_gemm
```

The MathDx-specific flags:

| Flag | Why |
|------|-----|
| `-DGLASS_BENCH_CUBLASDX` | force-includes `<cublasdx.hpp>` from `glass-nvidia.cuh` (it is otherwise gated on include order) |
| `-DSMS=860` | selects the cuBLASDx-tuned config + pre-instantiated GEMM table; **must match `-arch`** (860↔sm_86, 1200↔sm_120, …) |
| `--expt-relaxed-constexpr` | required by cuBLASDx's constexpr `__host__`/`__device__` helpers |
| `-Xptxas -O1` | works around a cuBLASDx miscompilation on recent CUDA (see `INSTALL.md`) |
| `-I$MATHDX_ROOT/include` | cuBLASDx headers |
| `-I$MATHDX_ROOT/external/cutlass/include` | CUTLASS headers cuBLASDx depends on |

> The cuSOLVERDx (LAPACK: `chol_inplace` / `posv` / `gels` / …) path is **not**
> covered by these examples — it additionally requires linking a precompiled
> device library (`-rdc=true -dlto -L$MATHDX_ROOT/lib -lcusolverdx -lcublas
> -lcusolver -lcudart`). See the top-level README section 5e and
> [`../bench/INSTALL.md`](../bench/INSTALL.md).

## Notes

- **One block per data item.** Every GLASS function assumes it runs inside a
  single CUDA block; these examples all launch `<<<1, threads>>>`. To process
  many independent items, launch one block each (`<<<num_items, threads>>>`).
- **Column-major by default.** Matrices are Fortran/column-major (`A[row +
  col*m]`), matching cuBLAS. The pure-SIMT path takes a `ROW_MAJOR=true`
  template parameter; the `glass::nvidia::` path uses the `layout` enum.
- Reductions (`reduce`, `dot`, `l2norm`) write their result **in place** to
  `x[0]` and may consume the input as scratch.
