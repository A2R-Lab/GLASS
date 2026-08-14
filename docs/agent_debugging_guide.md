# GLASS agent guide — debugging, bug classes, and refactor traps

Hard-won institutional knowledge for working on **GLASS** (*GPU Linear Algebra Simple
Subroutines*) — the comprehensive, header-only, single-block GPU linear-algebra library.
The public execution tiers are explicit **Block** `glass::block::`, **Warp**
`glass::warp::`, **Thread** `glass::thread::`, and **Nvidia**
`glass::nvidia::block::`/`warp::`; bare `glass::` is the measured-default
block-scope face, while `glass::cgrps::` is the cooperative-groups twin. Plus the
block-tridiagonal `glass::bdmv` / `glass::pcg`. **Read this before you change any
primitive or do a refactor.** Every GLASS function is a `__device__` helper that assumes it
runs inside **one CUDA block**, cooperating across `threadIdx`/`blockDim` (or a cooperative
group). That single-block, multi-thread, shared-data model is the source of essentially every
recurring bug below — they are races, thread-count assumptions, and uninitialized-scratch
reads, not algebra mistakes. Tone of this doc is a runbook: do X, check Y.

Source map you will reference constantly:
- Pure-SIMT surface: `glass.cuh` → `src/base/L1/*.cuh`, `src/base/L2/*.cuh`, `src/base/L3/*.cuh`.
- Cooperative-groups surface: `glass-cgrps.cuh`.
- Vendor backends: `glass-nvidia.cuh` → `src/nvidia/{l1,l2,l3,l3_simt,lapack,query_simt,tuning_table,types}.cuh`.
- Warp-scoped variants: inline in the base L1/L2/L3 headers (`src/base/L1/{reduce,dot,axpy,copy,scal,iamax}.cuh`, `src/base/L2/gemv.cuh`, `src/base/L3/{gemm,potrf,trsv,trsm,posv}.cuh`), under `namespace warp`.
- Block-tridiagonal: `glass::bdmv` (`src/base/banded/bdmv.cuh`), `glass::pcg` + `glass::pcg_scratch_bytes` (`src/base/pcg/solve.cuh`).
- Host smem helper: `glass_gemm_dispatch_smem` in `glass.cuh`.
- Tests: `test/conftest.py` (compile + cache harness), `test/test_l{1,2,3}.py`,
  `test/test_nvidia_dispatch.py`, `test/test_trailing_sync.py`, and the CUDA drivers under
  `test/cuda/*.cu` (`helpers.cuh`, `test_l3.cu`, …).

---

## 0. Validation checklist (do these EVERY time before committing a primitive change)

1. **Build the CUDA tests and run the suite.**
   ```bash
   cd test && pytest -v
   ```
   The first run compiles `test_l1/l2/l3` (and, if available, `test_l3_nvidia`,
   `test_nvidia_dispatch`, `test_trailing_sync`) via `nvcc` into `test/build/`; later runs reuse
   the cache. The Python tests diff GPU output against a NumPy/SciPy reference, so a green suite
   means your numbers match `numpy`/`scipy` at `rtol=atol=1e-3`.
2. **Confirm your edit actually recompiled.** Binaries compile lazily and the
   cache hashes the selected CUDA source, umbrella headers, shared helper, and
   that binary's operation-family headers (see §4). Add new binaries to the
   right family; unassigned binaries conservatively hash the full source tree.
3. **Check thread-count invariance.** A correct single-block GLASS function must produce
   **identical output for any block size.** The shipped tests sweep representative
   counts, including serial, single-warp, partial-final-warp, and multi-warp
   launches. Extend that sweep for a new primitive rather than introducing a
   one-off fixed thread count. A discrepancy indicates a missing barrier or a
   hard-coded thread assumption (§1).
4. **Check both layouts where relevant.** Storage order is a compile-time flag, not a runtime
   crash — wrong layout = silently wrong numbers. For any L2/L3 change, validate **column-major
   (default) AND all transpose combos** (`TRANSPOSE_A`/`TRANSPOSE_B` for `gemm`; a row-major
   operand is a transpose, so there is no per-operand row-major flag — only `ROW_MAJOR_C`).
   `test_l3.py` already has `test_gemm_rt`, `test_gemm_ct`, `test_gemm_warp`,
   `test_gemm_rowmajor_is_transpose` (the non-square + all-distinct-`M,N,K` net that catches the
   silent-on-square dim mapping) — extend them, don't bypass.
5. **Check the `beta = 0` path and uninitialized destinations** (§1c). If your op writes a
   destination via a beta-form GEMM, make sure the caller initializes C.
6. **MathDx-optional paths must still build and skip gracefully without it** (§3). Run the suite
   once with `MATHDX_ROOT` unset to confirm the SIMT paths and the skip logic are intact.
7. **Confirm you touched only the files you expect** (`git diff --stat`). The block-scoped
   namespaces — and the inline `glass::warp::` variants — share base impls via an include trick,
   so a one-line change in a base header can have library-wide blast radius (§5).

---

## 1. Recurring CUDA bug classes for single-block kernels

These are the failure modes specific to GLASS's "one block, many threads, shared data" model.
Check them first; they are far more likely than an algebra error.

### 1a. Missing `__syncthreads()` between a write phase and a later read/accumulate phase

**This is THE central single-block bug class.** Any primitive with two phases — write a partial
result to shared/device memory, then read or accumulate across it — needs a `__syncthreads()`
(or the cooperative-group equivalent) between the phases. Without it, thread *i* may read a slot
that thread *j* has not yet written.

- **Why a 32-thread test hides it:** within a single warp, threads execute in lockstep
  (warp-synchronous), so a phase-1 write by lane *j* is visible to lane *i*'s phase-2 read
  **without** a barrier. The race only appears once two or more warps run, because warp B can race
  ahead of warp A's writes. A test that only ever launches ≤ 32 threads will pass a kernel that is
  missing a barrier.
- **The fix / the rule:** every phase boundary that crosses threads needs a barrier. Look at the
  correct examples already in-tree: `reduce` in `src/base/L1/reduce.cuh` syncs once per halving
  round (`__syncthreads()` after each `for (i=rank; i<left; i+=size) x[i] += x[i+left];`);
  `reduce_fast` syncs after the inter-warp `s_scratch` write; `gemm_tiled` in
  `src/base/L3/gemm.cuh` syncs around each tile load (`__syncthreads()` before and after the
  inner-product over the tile). The matrix factor/solve flows (`inv.cuh`, `potrf.cuh`,
  `trsm.cuh`, `ldlt.cuh`, `posv.cuh`) sync between elimination steps. `inv.cuh` and
  `potrf.cuh` also have **K-way fused overloads** — `inv(K, dims, MAX_DIM, mats,
  s_temp)` and `potrf(K, dims, MAX_DIM, mats)` — with thin 2-/3-matrix wrappers (the
  3-matrix invert is GATO's Schur kernel, Q_k/Q_kp1/R_k). They interleave K independent matrices'
  sweeps over one shared `MAX_DIM = max(dims)` pivot/row loop (matrix `m` idles once `pivRC >=
  dims[m]`), keeping the **same augmented `[V | I]` convention (invert) / in-place lower `L·Lᵀ`
  convention (chol) and the same per-step save→update barrier pair** as the single-matrix paths —
  so the barrier audit is identical, just replicated per matrix. The fused invert gives each matrix
  a contiguous scratch span at the **prefix-sum offset `Σ_{j<m}(2*dims[j]+1)`** (total
  `Σ_m (2*dims[m]+1)`), recomputed locally per thread by scanning `dims[]` so there is no shared
  write to race on. The fused chol needs **no scratch** and distributes the K diagonals of each step
  across threads (`for (m = rank; m < K; m += size)`) before the parallel trailing-column update.
- **Counter-note (do not over-add barriers):** the *plain* `gemm_impl` (`gemm.cuh`) needs **no**
  interior sync — each thread owns a disjoint output element `C[cidx]` and never reads another
  thread's partial. Adding a barrier there is wrong-headed bloat. The rule is precise: sync only
  where one thread READS a slot another thread WROTE this kernel.
- **Tests MUST sweep multiple thread counts including a non-multiple of 32** (e.g. 1, 32, 48,
  256). 48 threads = 1.5 warps catches "works at exact warp multiples" bugs. The shipped suite
  pins 256, so add a sweep when validating a sync-sensitive change.

### 1b. Thread-count NON-invariance

A correct single-block function produces the **same output for any block size** — 1 thread,
32, a partial warp, or hundreds. The canonical correct pattern is the strided loop:
```cpp
uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
uint32_t size = blockDim.x * blockDim.y * blockDim.z;
for (uint32_t i = rank; i < n; i += size) { /* work on element i */ }
```
Every base GLASS op computes `rank`/`size` exactly this way (`reduce.cuh`, `gemm.cuh`, …) so it
self-adapts to the launch.

- **Symptoms of a bug:** output changes with the thread count, or is correct only at certain
  thread counts (e.g. only when `blockDim.x` divides `n`, or only at 1 thread).
- **Usual causes:** (1) a hard-coded `threadIdx`/`blockDim` assumption (e.g. "thread `i` handles
  element `i`" without the `+= size` stride, so elements past `blockDim` are never touched, or a
  partial last warp is skipped); (2) a missing barrier (§1a) — a sync bug presents *as* a
  thread-count-dependent result. **A discrepancy that appears only at >32 threads and vanishes at
  1 thread is almost always a missing sync or an uninitialized-scratch read, not a hardware blip.**
- **How to localize:** run at `THREADS=1` (forces full serialization — if it now passes, you have
  a race or a coverage gap) versus many threads, and bisect.

### 1c. `beta = 0` semantics — FIXED library-wide (2026-07-08): destinations are write-only

**Historical bug class, now closed at the primitive level.** GLASS's beta-form ops used to compute
`C[cidx] = alpha*res + beta*C[cidx]` unconditionally — reading `C` even when `beta == 0`, so an
**uninitialized scratch slot** holding leftover `NaN`/`Inf` poisoned the result (`0 * NaN == NaN`).
This is exactly how GRiD's generated RNEA NaN'd (2026-07-08): the v/a forward-pass gemvs write cold
`s_vaf` shared memory with `beta = 0`, and a diverged rollout kernel had left NaN on the SM.

- **The guarantee (BLAS/cuBLAS convention, now enforced):** when `beta == 0` the destination is
  **write-only** — it need not hold a valid value on input. Every beta-taking blend routes through
  `beta_blend(acc, beta, dst)` in `src/base/barrier.cuh`, which reads `dst` only under
  `beta != 0`. Covered: `gemv` / `gemv_strided` / `gemv_segmented` / `gemv_reduced`, `gemm`
  (untiled + tile4 + tiled) / `gemm_strided` / `gemm_reduced`, `syrk` / `syrk_reduced`, `symm`,
  `congruence`, `axpby` — block, `warp::`, and `cgrps::` surfaces (shared impls).
- **When you write a NEW beta-taking op:** use `beta_blend`, never a raw `alpha*res + beta*dst`.
  The no-beta overloads remain preferred for write-into-fresh-scratch (they also skip the
  runtime compare).
- **The regression net:** `test_l2.py::test_gemv*_beta0_poisoned_y_no_read` and
  `test_l3.py::test_gemm_rt_betaform_beta0_no_read` call the BETA overloads with `beta = 0` into a
  NaN-poisoned destination and require a clean result; `test_gemm_rt_beta0_no_read` covers the
  no-beta overloads. Keep poisoned-destination beta=0 cases in any new beta-form test.
- **The `nvidia::` surface honors the same guarantee** — audited 2026-07-08: `l3_simt.cuh`
  forwards to the fixed base cores, and the cuBLASDx route was probed empirically (poison ∈
  {0, 1e30, NaN} → bit-identical output, f32+f64): `cublasdx::execute` does not blend a
  `beta == 0` destination on MathDx 26.03. That vendor behavior is PINNED by
  `test_nvidia_dispatch.py::test_dispatch_op[beta0_poison]` — if a future MathDx bump breaks it,
  that test fails and the fix is to zero-fill `c_smem` instead of staging `C` when `beta == 0`.
- **The historical tell (if you ever see it again):** a discrepancy that is thread-count-dependent,
  nondeterministic across launches, and vanishes at 1 thread is the signature of reading cold
  scratch. Sanitizers do NOT catch read-of-uninitialized **shared** memory — qNaN-poison the
  arena to prove it (see GRiD/PDDP heavy-robustness notes).

### 1d. Storage-order / layout-flag mistakes

`TRANSPOSE_A` / `TRANSPOSE_B` and `ROW_MAJOR_C` are **compile-time template
flags** (default column-major). Passing data in the wrong order does **not crash** — it silently
indexes the wrong elements and produces wrong numbers.

- Column-major (default): `A[row + col*m]`. Row-major: `A[row*cols + col]`. `gemm_impl` selects the
  operand index via the `TRANSPOSE_A`/`TRANSPOSE_B` flags (a row-major operand == its
  transposed column-major view), and the output index via `ROW_MAJOR_C` (see `gemm.cuh`). On the
  `glass::nvidia::` path the equivalent is the `layout` enum (`LA`, `LB`, `LC`) mapping to
  cuBLASDx `Arrangement`.
- **Catch it by testing both layouts.** Mirror `test_l3.py`: `test_gemm_rt`/`test_gemm_ct` (all transpose combos), and
  `test_gemm_rowmajor_is_transpose` feeds a row-major operand (C-order) vs
  `np.asfortranarray(B).ravel(order='F')` (F-order) deliberately, and reshape the result with the
  matching order. If you add a layout, add the paired test; a layout bug is invisible to a
  same-layout round-trip.
- **No squareness restriction (since the BLAS-convention standardization):** `glass::gemm`
  follows standard BLAS — `C` is `M×N`, contraction `K`, with independent `TRANSPOSE_A`
  (`A` is `K×M`) and `TRANSPOSE_B` (`B` is `N×K`) flags. All four transpose combos work at any
  rectangular `M,N,K` (the old "`TRANSPOSE_B` requires `B` square" limitation is gone). The
  `test_gemm_rt`/`test_gemm_ct` matrices exercise all combos at non-square, all-distinct dims.

### 1e. Shared-memory sizing — dynamic scratch must match the host-side size helper

Functions that take `extern __shared__` scratch (or an explicit scratch pointer) require the
**host launch to allocate exactly what the device side reads**, or you get out-of-bounds access.

- **`gemm_tiled` / `gemm_dispatch`:** the tiled path needs `(m*TILE + TILE*k)*sizeof(T)` bytes,
  split into `s_A = smem` and `s_B = smem + m*TILE`. The host size is computed by
  `glass_gemm_dispatch_smem<T>(m, k, block_threads, tile)` in `glass.cuh` (returns 0 when tiling
  is not warranted: `m >= 32` or `m*k > block_threads`). `test_l3.cu`'s `gemm_tiled` op hard-codes
  the matching `(m*8 + 8*n)*sizeof(float)` with `TILE=8` — if you change the tile size on one side
  you MUST change it on both. A too-small allocation overruns `s_B`.
- **`reduce_fast` / `dot` / `nrm2`:** need `ceil(blockDim/32)*sizeof(T)` bytes of
  `s_scratch` (one slot per warp). Under-sizing this overflows when `blockDim > 32*available`.
- **`glass::nvidia::` (CUB) L1:** scratch is `sizeof(cub::BlockReduce<T,THREADS>::TempStorage)`;
  query it with `glass::nvidia::reduce_smem_size<T,THREADS>()`. For the cuBLASDx/cuSOLVERDx
  paths, query with `gemm_smem_size<T,M,N,K[,TC]>()` / `posv_smem_size<...>()` etc. and pass the
  EXACT value to the launch — too small = OOB, and (for the default form) a wrong thread count
  deadlocks. Always query, never hard-code a guessed byte count.

### 1f. Block-tridiagonal layout contracts (`glass::bdmv` / `glass::pcg`)

The banded matvec and PCG solver carry layout/launch preconditions that fail *silently*
(wrong numbers, not a crash) if violated:

- **Pre-zero the vector pads.** Vectors are padded `(knot_points+2)*state_size` with one
  `state_size` pad block on each end. `glass::bdmv` relies on the first/last block-rows multiplying
  their absent `L`/`R` against **zero pad** — if the pads hold garbage, the edge rows are wrong.
  `glass::pcg` zeroes its internal vectors (`set_const`), but a hand-rolled `glass::bdmv` caller must
  zero the pads itself.
- **Use a multiple of 32 PCG threads.** Its fast dot reduction uses a full-warp
  shuffle mask; a partial final warp violates that launch contract. This is the
  ONE documented exception to GLASS's thread-count-invariance contract — do not
  add new ones.
- **Dynamic shared memory must be at least
  `glass::pcg_scratch_bytes<T,state_size,knot_points>(threads)` bytes** — pass
  the helper result directly as the launch byte count. Under-sizing overruns;
  see 1e.
- **`[L|D|R]` is row-major per block-row**, strip `br` at `s_matrix + br*(3*state_size)*state_size`.
  Mixing this up with a dense or column-major layout silently produces wrong results — validate a
  new caller against the dense reference in `test/test_banded.py` / `test/test_pcg.py`.

### 1g. Warp-scoped broadcast via shared re-read → `__restrict__` stale-cache miscompile

**Never broadcast a warp value by writing it to shared and re-reading that same location.** Use
`__shfl_sync` from the computing lane's register instead. The bad pattern:

```cpp
if (lane == 0) s_A[k*N+k] = sqrt(val - sum);   // lane 0 writes shared
__syncwarp();
T diag = s_A[k*N+k];                            // every lane re-reads shared  <-- MISCOMPILES
```

When the buffer is reached through a caller `__restrict__` pointer under aggressive optimization
(observed: sm_120 / CUDA 13.2, `-O3`), nvcc can **cache that shared load stale** — the non-writing
lanes read a previous value, so an in-place warp solve returns wrong results for a fraction of
inputs. `__syncwarp()` guarantees *execution* convergence, not that the compiler reloads the
shared address. The fix (`1df6e40`, in `warp::potrf` / `warp::trsv`):

```cpp
T diag = static_cast<T>(0);
if (lane == 0) { diag = sqrt(val - sum); s_A[k*N+k] = diag; }  // still write the result
diag = __shfl_sync(0xffffffffu, diag, 0);                      // broadcast from REGISTER
```

Keep the result write to shared (it's the actual output); only the *broadcast* moves to `__shfl`.
Retain any **trailing** `__syncwarp()` — that one orders this iteration's shared writes before the
next iteration reads them, and is unrelated to the broadcast.

Nasty properties, so check for it deliberately:

- **GLASS's own warp tests will NOT catch it** — `test_l3.cu`'s `k_posv_warp_7` calls the solve on
  a non-`__restrict__` pointer, and the miscompile only fires through a `__restrict__` caller *and*
  typically only inside a larger kernel (more inlining / register pressure). It was found via a
  downstream warp-per-candidate IK solver: correct standalone, wrong only in the full kernel.
- **The tell:** glass-on-a-fresh-copy correct, in-place-under-`__restrict__` wrong,
  in-place-without-`__restrict__` correct ⇒ a `__restrict__` aliasing miscompile, not your algebra.
  Don't "fix" it by dropping `__restrict__` (you lose the optimization) — kill the shared re-read.
- **`glass::warp::reduce` is already safe** (shfl-based); `warp::gemm` is flat (no broadcast). The
  block-scoped `cgrps`/`__syncthreads` twins are safe *today* (full-block fence) but carry a note to
  prefer `g.shfl(pivot, 0)` if ever called with a warp-tiled group.

### 1h. Loop-unroll copies contract FMAs differently → thread-count NON-invariance inside ONE kernel

Found 2026-07-28 landing the robotics ops. A `for (i = rank; i < n; i += size)` loop whose body
contains a float multiply-add chain is NOT automatically bit-invariant across block sizes, even
though the arithmetic per element is "the same code": ptxas may UNROLL the loop, and different
unroll copies of the body can contract the multiply-adds differently. At `tpb=1` one thread runs
elements through the copy sequence (prologue copy, then main-loop copies); at `tpb=32` each thread
executes only the FIRST copy — so the same element computes through differently-contracted code
and the outputs differ by 1 ulp on a handful of elements. Caught by the tpb∈{1,32,64,256}
bit-equality gate on `se3_retract_hessian` (3/864 f64 elements, 1 ulp, uniform-looking until you
diff bits).

- **Fix:** `#pragma unroll 1` on every rank-strided loop whose body does FP arithmetic
  (see `spatial/cross.cuh` fused applies, `lie/se3.cuh` hessian k-loop, `L1/softmax.cuh` exp
  loops). Pure copy-out loops (`out[i] = tmp[i]`) and comparison-only loops (argreduce combine)
  are immune — don't pessimize those.
- **Not the same bug as** the cross-TIER 1-ulp jitter (test_thread's `-fmad=false` finding):
  that one is across template instantiations and is handled by test tolerance; THIS one is within
  a single compiled kernel and breaks the hard bit-invariance contract, so it must be fixed in
  the source, not tolerated in tests.
- **The tell:** block output differs between two block sizes by ≤1-2 ulp on a SPARSE subset of
  elements (the elements whose iteration parity lands in a different unroll copy), while
  everything is deterministic run-to-run at any fixed size.

---

## 2. Debugging methodology (what localizes a bug fast here)

- **Run at 1 thread vs many threads to bisect sync bugs.** `THREADS=1` serializes the block. If a
  failing op passes at 1 thread but fails at 256, you have a missing barrier (§1a), a cold-scratch
  read (§1c), or a thread-count assumption (§1b) — not an algebra error. If it fails at 1 thread
  too, the math/index logic is wrong.
- **`compute-sanitizer`.** Run the built test binary under it:
  - `compute-sanitizer --tool racecheck ./build/test_l3 gemm_tiled simple ...` flags missing
    barriers between shared-memory write/read phases.
  - `compute-sanitizer --tool memcheck ./build/test_l3 ...` flags OOB from a smem/scratch
    mis-size (§1e) or a wrong-layout index walking off the array.
- **Compare against the NumPy/SciPy reference in the Python tests.** Every `test_l*.py` op has a
  one-line `numpy`/`scipy` ground truth (e.g. `alpha*A@B + beta*C0`, `np.linalg.cholesky`,
  `scipy.linalg.solve_triangular`). Reproduce a failure in Python first: build the same inputs,
  compute the reference, and diff — it tells you *which* output element is wrong, which often
  pinpoints a layout or index bug immediately.
- **Shrink to the smallest failing size.** Drop to a 2×2 or 3×3 with simple integer inputs and
  print the full matrix (`helpers.cuh::print_device_vec`). A single wrong cell at a known
  `(row,col)` localizes a stride/transpose bug far faster than a 64×64 `allclose` failure.
- **A slot that should be 0 but isn't is the highest-signal lead** — it means a stale or
  mis-strided read (cold scratch, wrong layout offset, or skipped element from a thread-coverage
  gap).

---

## 3. Optional-dependency gotchas (`glass::nvidia::` + MathDx)

The `glass::nvidia::` L2/L3/LAPACK backends need **MathDx (cuBLASDx + cuSOLVERDx)** and a
`MATHDX_ROOT` env var. The pure-SIMT `glass::` / `glass::cgrps::` surface and the L1 CUB path do
NOT. Rules:

- **SIMT paths must work without MathDx.** `l3_simt.cuh` (the `gemm_batched_1d` /
  `gemm_strided_batched_1d` SIMT-only batched APIs) has no cuBLASDx dependency — `conftest.py`
  compiles `test_l3_nvidia` for exactly this reason and only skips it on a generic toolchain
  failure, not on missing MathDx. Keep new SIMT-only additions cuBLASDx-free so they build on a
  bare CUDA install.
- **Tests that need cuBLASDx/cuSOLVERDx must skip gracefully when it's absent.** See how
  `conftest.py` does it: it checks `os.environ.get("MATHDX_ROOT")` and
  `(<root>/include/cublasdx.hpp).exists()`, and only then compiles `test_nvidia_dispatch` with
  `--expt-relaxed-constexpr -DGLASS_BENCH_CUBLASDX` and the MathDx include dirs. The
  `bin_nvidia_dispatch` fixture calls `pytest.skip("test_nvidia_dispatch needs MATHDX_ROOT
  (cuBLASDx)")` when the binary wasn't built. Mirror this for any new MathDx-dependent test —
  guard at the fixture, never let a missing dep turn into a hard compile error in the session
  fixture.
- **Dual-build pattern for partially-optional binaries.** `test_trailing_sync` compiles with
  cuBLASDx flags *if available* and without them otherwise; the binary itself prints `SKIP` for
  the cuBLASDx op when built without `-DGLASS_BENCH_CUBLASDX`, and
  `test_trailing_sync_cublasdx_gemm` turns that into `pytest.skip(...)`. Use the same
  "compile both ways, runtime-skip the vendor op" shape when a test mixes SIMT and vendor cases.
- The headers auto-detect via `GLASS_HAVE_CUBLASDX` / `GLASS_HAVE_CUSOLVERDX` (include-order
  driven); the bench/test harness force-enables with `GLASS_BENCH_CUBLASDX` /
  `GLASS_BENCH_CUSOLVERDX`. cuSOLVERDx additionally needs link flags (`-rdc=true -dlto
  -lcusolverdx -lcublas -lcusolver -lcudart`).

---

## 4. Test-infra gotchas (the compile cache)

- **The compile cache is keyed on a whole-tree source hash plus build
  identity.** `conftest.py::_hash_sources` hashes the selected `.cu`, shared
  helpers, umbrella headers, and EVERY `src/**/*.cuh` by glob (an edited or
  new header can never be forgotten; the narrower per-family key was tried and
  reverted 2026-08-14 — it reintroduced the 2026-07-17 "edited header never
  rebuilds" bug). The cache fingerprint additionally folds in `-arch`, extra
  nvcc flags, and the nvcc version, so a GPU/toolchain change rebuilds too.
- **Force a clean rebuild** by deleting the build artifacts: `rm -rf test/build` (or just the
  `<name>.hash` files). `test/build/` is **gitignored** — it's a scratch dir, never commit it.
- The `bins` fixture is **lazy**: selecting one pytest module compiles only the binaries it
  accesses. Per-binary skips are surfaced as fixture `pytest.skip`s (§3), so a single missing
  optional dep doesn't fail the run.

---

## 5. Refactor traps (looks-fine-but-isn't)

- **Converting a serial loop to parallel without adding the needed barrier.** The most common
  refactor regression: you split a thread-0 serial pass into a strided multi-thread loop for
  speed, but the *next* phase reads what the loop wrote — and you didn't add a `__syncthreads()`
  between them (§1a). It will pass at 32 threads and race at 256. Any serial→parallel change must
  be re-validated with the thread-count sweep (§0.3), not just the default 256-thread test.
- **Touching a shared base impl has block-wide blast radius.** `glass::`, `glass::cgrps::`, and
  `glass::nvidia::` all pull the SAME base headers in via the **include trick**: `glass.cuh`
  literally `#include`s `src/base/L1/*.cuh` etc. *inside* `namespace glass { ... }`, and the other
  umbrellas do likewise into their namespaces. So editing `src/base/L3/gemm.cuh` changes
  `glass::gemm` AND `glass::cgrps::gemm` AND the inline `glass::warp::gemm` AND the SIMT fallback
  that `glass::nvidia::gemm` auto-dispatches to (and `glass::pcg`, which composes `glass::bdmv` +
  the base dot/axpy). After touching a base impl, run the FULL suite (`test_l1` + `test_l2` +
  `test_l3` + banded/pcg + the nvidia dispatch/trailing-sync tests), not just the namespace you were thinking
  about. Validate that all the surfaces sharing that base impl still produce identical numbers.
- **Changing a default thread group / launch contract.** `glass::nvidia::` (default form) requires
  **exactly** `gemm_threads<T,M,N,K>()` threads — a mismatch silently deadlocks (or, without
  `-DNDEBUG`, asserts via the P1-4 `assert(blockDim >= GEMM::block_dim)`). Don't change a wrapper's
  default `BLOCK_THREADS` or its `TRAILING_SYNC` default without auditing every caller's launch
  geometry; the `BlockDim<TC>` form exists precisely so callers can launch with a different thread
  count, and the trailing-sync default of `true` is what makes the common case correct.
- **Touching `TRAILING_SYNC` plumbing.** The cuBLASDx-backed L2/L3 macros emit BOTH
  `TRAILING_SYNC=true` and `=false` specializations from one `DEFINE_NVIDIA_*` invocation — drop
  one and the `=false`-variant kernel fails to LINK. `test_trailing_sync.py` exists to catch
  exactly that (both variants link + run, and produce numerically identical output when the
  `=false` caller emits its own trailing `__syncthreads()`). The INTERIOR syncs inside
  `gemm`/`gemm_tiled`/LAPACK factor-solve flows are required for correctness and are **not** gated
  on `TRAILING_SYNC` — only the final trailing barrier is. Do not gate an interior sync on it.
- **Reordering template parameters on a base/vendored signature.** Callers (including downstream
  consumers that vendor GLASS) pass template args positionally. Appending a new flag at the END is
  usually safe; inserting one before an existing positional arg silently shifts meaning. Keep new
  optional template params at the tail with defaults.

---

## 6. Vendor-descriptor + precision gotchas (2026-06-23 mega-sweep / fp64 work)

- **Dynamic-smem opt-in cap is GPU-specific and lower than you think.** RTX 5090 / sm_120
  caps opt-in dynamic shared memory at **99 KB** (`cudaDevAttrMaxSharedMemoryPerBlockOptin`),
  not the 227 KB of data-center Blackwell. A cuBLASDx/cuSOLVERDx descriptor whose
  `shared_memory_size` exceeds that fails to launch. Query the cap at runtime; don't assume.
- **An unchecked failed launch reads as a *win* in a timing bench.** When a launch fails
  (e.g. smem > opt-in cap, or `cudaFuncSetAttribute` rejected), the kernel never runs and the
  timing loop measures ~1 ns/problem — which the "lower is better" comparison reports as a
  huge speedup. ALWAYS check `cudaGetLastError()` / the sync return after a vendor launch and
  drop the data point (return -1) on failure. See `bench/bench_mega_sweep.cu::nv_timed`.
- **Parameterizing the float-only nvidia macros on a scalar type.** The cuBLASDx/cuSOLVERDx
  wrapper macros were `float`-hardcoded. To add `double`: within each core macro, every literal
  `float` → a `CT` token (safe — no other `f`-word collides: `1.0f`/`FillMode`/`fill_mode` don't
  contain "float"), thread `CT` before the always-last `ARCH` param, and add a `##CT##` segment
  to the impl-namespace name so float/double instances don't collide. Keep the existing
  `DEFINE_NVIDIA_*` as `_PREC(..., float)` aliases (back-compat). Validate BOTH precisions
  (`*_smem_size` doubling for f64 is the cheap first signal; a numpy-oracle residual is the real one).
- **fp64 vendor descriptors hit the smem cap sooner.** f64 doubles every byte, so the nvidia
  leg caps at a smaller N than f32 (gemm f64 ~N64 vs f32 ~N64-but-via-different-bytes; chol/posv
  f64 to ~96). The backend picker (`glass-defaults.cuh`) encodes a narrower f64 nvidia band.

## 7. Contraction-parallel `*_reduced` engine + flagged ops (2026-06-24 feature expansion)

- **Sub-warp invariance requires reproducing the warp-shuffle tree EXACTLY, not a serial sum.**
  The `*_reduced` engines reduce each output across a warp's lanes via `glass::warp::reduce`
  (a `__shfl_down` tree, offsets 16/8/4/2/1). A block with `< 32` threads can't run that shuffle,
  so the fallback must combine 32 register partials in the **same pairwise grouping** —
  `glass::reduced_tree32` (`L1/reduce.cuh`) does exactly that, so results are bit-identical across
  the 32-thread boundary. A naive serial `for k: sum += p[k]` fallback would round differently and
  FAIL the 1-vs-32 invariance check on non-integer inputs. Verify with the thread sweep
  `1/7/31/32/33/57/64/...` using `np.array_equal` (not allclose).
- **Trailing partial warp must IDLE; never call `__shfl_*_sync(0xffffffff,...)` from a partial warp.**
  Use `n_warps = blockDim >> 5` (full warps only); warps `>= n_warps` skip the loop and fall through
  to the trailing barrier. A partial warp calling the full-mask shuffle is UB.
- **Cross-surface is NOT bit-identical for composed (two-step) ops — that's FMA, not a bug.**
  Single-step engines (`gemm_reduced`, `tensor_vec_contract`) match bit-for-bit across block/warp/cgrps.
  But `congruence_sym`/`bilinear`/`riccati_gain` run a separate gemm in step 1 whose `a*b` the compiler
  may fuse into an FMA differently per instantiation → ~1 ULP cross-surface drift. Confirm benign with
  `nvcc --fmad=false` (block==warp exactly); test cross-surface with `allclose`, reserve `array_equal`
  for WITHIN-surface thread-invariance.
- **Compile-out flags must be byte-identical when off.** `CHECK`/`REGULARIZE`/`SYMMETRIC` etc. guard their
  extra work behind `if constexpr (FLAG)` with the flag defaulted false. Prove the off-path unchanged by
  the EXISTING golden suite (`test_l3` chol, `test_ldlt`, `test_posv`) still passing — not just the new flag test.
- **Adding a bool flag to one of two same-named overloads can make calls ambiguous.** `posv` has a 2-template-param
  single-RHS and a 3-param multi-RHS overload. Putting `REGULARIZE`/`CHECK` (with the same trailing arg shape)
  on BOTH makes `posv<T,N>(A,b)` ambiguous. Fix: flag only the multi-RHS form (single-RHS = `NRHS=1`); the
  single-RHS overload stays untouched. See `posv.cuh`.
- **`std::size_t` inside `namespace glass` resolves to `glass::std`.** The base headers are `#include`d *inside*
  `namespace glass {}`, so a `#include <cstddef>` there nests `std`, and `std::size_t` fails to compile. Use a
  plain `uint32_t` return (these are small single-block byte counts) or pre-include the system header at global scope.
- **A `*_reduced` op is a measured LOSS on sm_120 — don't assume parallelizing the contraction helps.** The
  serial one-thread-per-output loop over shared memory is very hard to beat; the warp-shuffle path is 10–100×
  slower except in a tiny corner (`n_out <= blockDim/32` AND `K >= 32`). ALWAYS bench before claiming a speedup
  (`bench/RESULTS.md`, reduced section); the picker `glass::suggested_use_reduced<>()` recommends serial almost always.
- **Regenerating the dispatch defaults: one tool, one margin.** The three shipped tables — the warp/block/nvidia
  ladder (`glass-defaults.cuh`, per-arch `ideal_sm*` tables + SM dispatch — tune.py replaces only the measured arch's marker block), the per-(M,N,K) cuBLASDx-vs-SIMT table (`src/nvidia/tuning_table.cuh`),
  and the `suggested_use_reduced<>` predicate — are all regenerated by `bench/tune.py` (legs `ladder`/`shapes`/
  `reduced`/`figures`). Every verdict routes through the single tie rule in `bench/tune_pick.py::pick`: a
  dependency impl (nvidia/cublasdx/reduced) wins ONLY if it beats the simplest no-dependency impl by more than the
  margin (default 5%), else the launchable-everywhere path stays — this is what stops sub-noise jitter (e.g. a
  cuBLASDx pick 1% faster than warp) from churning the tables. Don't hand-edit these tables or re-derive the margin
  per-tool; run `tune.py` on a **quiet GPU** (perf timing must be isolated) and `--dry-run` to diff first.

---

### Quick reference: which bug class fits the symptom

| Symptom | Most-likely class |
|---|---|
| Correct at 32 threads, wrong at 256 | Missing `__syncthreads()` (§1a) |
| Output changes with block size | Thread-count non-invariance / sync (§1b/§1a) |
| `NaN` in the result for `beta=0` into scratch | Uninitialized C read (§1c) |
| Wrong numbers, no crash, transposed/permuted look | Layout flag mistake (§1d) |
| `memcheck` OOB, sporadic garbage | Smem/scratch undersize vs host helper (§1e) |
| Flaky, vanishes in isolation / at 1 thread | Cold-scratch read or missing sync (§1c/§1a) |
| Hard compile error when MathDx absent | Optional-dep guard missing (§3) |
| Edits seem to have no effect | Stale compile cache / header not hashed (§4) |
| Warp op correct standalone, wrong only in a larger `__restrict__` kernel | Shared-reread broadcast miscompile — use `__shfl` (§1g) |
| Block-tridiagonal `glass::bdmv`/`pcg` wrong at edges or non-warp thread count | Layout/launch contract (§1f) |
| Vendor op times ~1 ns / "wins" by 100×+ | Failed launch (smem > opt-in cap) read as a win — error-check it (§6) |
| f64 vendor op won't launch above some N | f64 descriptor smem > opt-in cap; cap is lower for double (§6) |
| `*_reduced` op correct at 32+ threads but differs at 1/7/31 | Sub-warp fallback not reproducing the shuffle tree — use `reduced_tree32` (§7) |
| Two-step `congruence`/`bilinear`/`riccati` differs warp-vs-block by ~1 ULP | Benign FMA-context drift; compare cross-surface with `allclose` (§7) |
| `posv<T,N>(A,b)` suddenly ambiguous after adding a flag | Flagged both posv overloads — flag only multi-RHS (§7) |
| `std::size_t` undefined / `glass::std` error in a base header | `std` nested by an in-namespace include — return `uint32_t` (§7) |
| `*_reduced` op far slower than serial | Expected on sm_120 — use `suggested_use_reduced<>()`, prefer serial (§7) |

## Bare-face body dispatch (2026-07-30, Phase 2)

Three bug classes introduced/dodged by the dispatch wrappers (`src/base/dispatch.cuh`):

1. **Name shadowing is per-NAME, not per-overload.** A single declaration of
   `dot` directly in `namespace glass` hides the ENTIRE `block::dot` overload
   set from the bare face (qualified lookup stops before using-directives).
   Every public block overload of a shadowed op must be restated (dispatched
   or forwarded) — a new block:: overload on dot/gemv/potrf/trsv/posv/eig3/
   softmax that isn't restated silently vanishes from `glass::`.
2. **Partial-scope bodies need a full first warp + a block-wide sync.** The
   warp body is only valid when `blockDim.x >= 32` (x-major first warp is a
   real hardware warp); narrower launches must fall back to the block body.
   The trailing `__syncthreads()` after `if (tid<32) warp::op(...)` is the
   publish barrier — dropping it is the classic write-then-read race, invisible
   at 32 threads, racy at 64+. `test_dispatch.cu` runs TB=16 deliberately.
3. **Never extrapolate a single-point sweep verdict.** softmax was measured at
   n=16 only; an unbounded table entry would have routed n=4096 onto one warp.
   tune.py's `_body_expr` bounds every verdict at the largest measured N
   (beyond it -> block) — keep that property if you touch the emitter.
