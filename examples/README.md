# GLASS examples

Minimal, self-contained, **compile-and-run** CUDA programs — one concept each.
Every file is a complete program: a `__global__` kernel that calls a GLASS
device function, plus a `main` that allocates device memory, launches **one
block** (or a batch of blocks for the batched demos), copies the result back,
and **verifies it** — examples return non-zero on a numeric mismatch, and the
test suite (`test/test_examples.py`) compiles and runs every one of them on
hardware.

> This table is the CANONICAL per-example description (the hosted docs page
> `docs/source/user_guide/tutorials/examples.rst` mirrors it and carries the
> full code listings — keep the two in sync).

## Which example shows what

| File | Shows | Backend / deps |
|------|-------|----------------|
| [`01_axpy_simt.cu`](01_axpy_simt.cu) | L1 vector op `axpy` (`y = αx + y`), **runtime size** | pure SIMT |
| [`02_gemm_conventions.cu`](02_gemm_conventions.cu) | THE GEMM example: the **standard-BLAS convention** at a non-square shape, runtime + compile-time size overloads, all four `TRANSPOSE_A`/`TRANSPOSE_B` combos, **"row-major is just a transpose"** (bit-identical), and the `glass::cgrps::` spelling | pure SIMT |
| [`03_reductions_norms.cu`](03_reductions_norms.cu) | block reductions + norms: `reduce`, warp-shuffle `reduce_fast` (+ `reduce_fast_scratch_bytes`), and the `nrm2` family across block + warp tiers | pure SIMT |
| [`04_gemm_dispatch.cu`](04_gemm_dispatch.cu) | `glass::block::gemm_dispatch` auto-tiling + dynamic shared memory via the `glass_gemm_dispatch_smem` host helper | pure SIMT |
| [`05_nvidia_gemm.cu`](05_nvidia_gemm.cu) | the cuBLASDx-backed `glass::nvidia::block::gemm` path | **requires NVIDIA MathDx** |
| [`06_warp_ops.cu`](06_warp_ops.cu) | single-warp `glass::warp::` ops (`reduce`, 4×4 `gemm`, SPD `potrf`+`trsm`+`trsm_transpose`), launched `<<<1,32>>>` | pure SIMT |
| [`07_pcg_solve.cu`](07_pcg_solve.cu) | block-tridiagonal PCG solve `glass::block::pcg` (`[L\|D\|R]` strips, padded vectors, block-Jacobi preconditioner) | pure SIMT |
| [`08_backend_picker.cu`](08_backend_picker.cu) | choose a backend + launch config with `glass-defaults.cuh` (`suggested_backend` / `suggested_block_threads` / `suggested_warps_per_block`), then dispatch a real SPD solve to the picked launch | pure SIMT |
| [`09_gemm_strided.cu`](09_gemm_strided.cu) | `gemm_strided` — GEMM on column-major sub-blocks with explicit leading dims | pure SIMT |
| [`10_ldlt_solve.cu`](10_ldlt_solve.cu) | symmetric-**indefinite** solve `ldlt` + `ldlt_solve`, plus the `CHECK=true` failure flag + **inertia** reporting (`ldlt_scratch_bytes`) | pure SIMT |
| [`11_riccati_gain.cu`](11_riccati_gain.cu) | LQR feedback gain `K = (R + BᵀPB)⁻¹(BᵀPA)` via `riccati_gain`, smem sized by `riccati_scratch_bytes<T,NX,NU>()` | pure SIMT |
| [`12_inv.cu`](12_inv.cu) | matrix inversion on the augmented `[A \| I]` layout: `inv` (+ `inv_scratch_bytes`), and the robust `inv_pivoted` recovering a zero leading pivot | pure SIMT |
| [`13_thread_pack.cu`](13_thread_pack.cu) | the `glass::thread::` tier: 4096 N=6 SPD solves, one problem per THREAD (32 packed per warp), launch shape from `suggested_threads_per_block<>` | pure SIMT |
| [`14_spatial_dynamics.cu`](14_spatial_dynamics.cu) | Featherstone spatial cross products (the RNEA inner loop): fused `motion_cross_mul`/`force_cross_mul` vs materialize-6×6 + `gemv` | pure SIMT |
| [`15_floating_base_retract.cu`](15_floating_base_retract.cu) | batched SE(3) manifold integration at thread scope (`se3_retract`): unit-norm drift-free, one-parameter-subgroup check | pure SIMT |
| [`16_mppi_weights.cu`](16_mppi_weights.cu) | the MPPI weight update: `softmax` + `argmin` per controller block, bit-identical across block sizes | pure SIMT |
| [`17_cone_projection.cu`](17_cone_projection.cu) | friction-cone AL constraint step: `soc_project` + `al_soc_value`/`al_hinge_value`, projection-orthogonality checks | pure SIMT |
| [`18_collision_spheres.cu`](18_collision_spheres.cu) | sphere narrow phase: `transform_sphere` → `sphere_box_dist` → `smooth_hinge` cost chain | pure SIMT |
| [`19_best_fit_rotation.cu`](19_best_fit_rotation.cu) | batched Wahba/Kabsch alignment: cross-covariance → `thread::closest_rotation`, verified vs ground-truth rotations | pure SIMT |

**Every example is pure SIMT except `05_nvidia_gemm.cu`**, which needs MathDx
(cuBLASDx); skip it if you don't have MathDx installed.

## Building

The easy way (auto-detects your GPU arch; skips 05 unless `MATHDX_ROOT` is set):

```bash
cd examples
make -j            # or: make -j ARCH=sm_120  /  make -j MATHDX_ROOT=/opt/nvidia/mathdx/26.03
make run           # runs every built example
```

Or by hand — all examples `#include` the GLASS headers from the repo root, so
build **from inside `examples/`** with `-I..` and the `-arch` matching your GPU:

```bash
nvcc -std=c++17 -arch=sm_75 -I.. 02_gemm_conventions.cu -o gemm && ./gemm
```

### The MathDx example (05) — extra flags

Install MathDx and set `MATHDX_ROOT` first — see
[`../bench/INSTALL.md`](../bench/INSTALL.md). Then:

```bash
nvcc -std=c++17 -arch=sm_86 -I.. \
     -DGLASS_BENCH_CUBLASDX -DSMS=860 \
     --expt-relaxed-constexpr -Xptxas -O1 \
     -I$MATHDX_ROOT/include \
     -I$MATHDX_ROOT/external/cutlass/include \
     05_nvidia_gemm.cu -o nvidia_gemm && ./nvidia_gemm
```

| Flag | Why |
|------|-----|
| `-DGLASS_BENCH_CUBLASDX` | force-includes `<cublasdx.hpp>` from `glass-nvidia.cuh` (otherwise gated on include order) |
| `-DSMS=860` | selects the cuBLASDx-tuned config + pre-instantiated GEMM table; **must match `-arch`** (860↔sm_86, 1200↔sm_120, …) |
| `--expt-relaxed-constexpr` | required by cuBLASDx's constexpr `__host__`/`__device__` helpers |
| `-Xptxas -O1` | works around a cuBLASDx miscompilation on recent CUDA (see `INSTALL.md`) |
| `-I$MATHDX_ROOT/include` | cuBLASDx headers |
| `-I$MATHDX_ROOT/external/cutlass/include` | CUTLASS headers cuBLASDx depends on |

> The cuSOLVERDx (LAPACK: `potrf` / `posv` / `gels` / …) path is **not**
> covered by these examples — it additionally requires linking a precompiled
> device library (`-rdc=true -dlto -L$MATHDX_ROOT/lib -lcusolverdx -lcublas
> -lcusolver -lcudart`). See [`../bench/INSTALL.md`](../bench/INSTALL.md).

## Notes

- **One block per data item.** Every GLASS function assumes it runs inside a
  single CUDA block; most examples launch `<<<1, threads>>>`. To process many
  independent items, launch one block each (`<<<num_items, threads>>>`) — the
  batched demos (13, 15–19) do exactly that.
- **Column-major by default.** Matrices are Fortran/column-major (`A[row +
  col*m]`), matching cuBLAS / Eigen. GEMM follows the standard BLAS convention
  (`C` is M×N, contraction K) with `TRANSPOSE_A` / `TRANSPOSE_B` / `ROW_MAJOR_C`
  flags; a row-major operand is just a transpose (`02_gemm_conventions.cu`).
  The `glass::nvidia::` path uses the `layout` enum.
- Reductions (`reduce`, `dot`, `nrm2`) write their result **in place** to
  `x[0]` and may consume the input as scratch (the `glass::warp::` forms return
  the value instead).
