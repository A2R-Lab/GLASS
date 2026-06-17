# GLASS agent guide — debugging, bug classes, and refactor traps

Hard-won institutional knowledge for working on **GLASS** — the header-only, single-block
GPU linear-algebra library (`glass::` SIMT, `glass::cgrps::` cooperative-groups, and
`glass::nvidia::` CUB/cuBLASDx/cuSOLVERDx backends). **Read this before you change any
primitive or do a refactor.** Every GLASS function is a `__device__` helper that assumes it
runs inside **one CUDA block**, cooperating across `threadIdx`/`blockDim` (or a cooperative
group). That single-block, multi-thread, shared-data model is the source of essentially every
recurring bug below — they are races, thread-count assumptions, and uninitialized-scratch
reads, not algebra mistakes. Tone of this doc is a runbook: do X, check Y.

Source map you will reference constantly:
- Pure-SIMT surface: `glass.cuh` → `src/base/L1/*.cuh`, `src/base/L2/*.cuh`, `src/base/L3/*.cuh`.
- Cooperative-groups surface: `glass-cgrps.cuh`.
- Vendor backends: `glass-nvidia.cuh` → `src/nvidia/{l1,l2,l3,l3_simt,lapack,query_simt,tuning_table,types}.cuh`.
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
2. **Confirm your edit actually recompiled.** The compile cache is keyed on a **source hash** of a
   curated file list (see §4). If you edited a header that the hash covers, the next `pytest` run
   rebuilds automatically. If you touched a header NOT in that list, the cache is stale — force a
   clean rebuild (delete `test/build/`) or add the file to `_hash_sources` in `conftest.py`.
3. **Check thread-count invariance.** A correct single-block GLASS function must produce
   **identical output for any block size.** The shipped tests launch a fixed `THREADS = 256`
   (see `test_l3.cu`), so they do NOT exercise this on their own. Before trusting a change to a
   multi-phase primitive, run it at **1 thread, 32 (one warp), a partial warp (e.g. 48), and many
   warps (e.g. 256)** and diff the outputs. A discrepancy = a missing barrier or a hard-coded
   thread assumption (§1).
4. **Check both layouts where relevant.** Storage order is a compile-time flag, not a runtime
   crash — wrong layout = silently wrong numbers. For any L2/L3 change, validate **column-major
   (default) AND row-major** (and `TRANSPOSE_B` for `gemm`). `test_l3.py` already has
   `test_gemm`, `test_gemm_t`, `test_gemm_rowmajor`, `test_gemm_ex` — extend them, don't bypass.
5. **Check the `beta = 0` path and uninitialized destinations** (§1c). If your op writes a
   destination via a beta-form GEMM, make sure the caller initializes C.
6. **MathDx-optional paths must still build and skip gracefully without it** (§3). Run the suite
   once with `MATHDX_ROOT` unset to confirm the SIMT paths and the skip logic are intact.
7. **Confirm you touched only the files you expect** (`git diff --stat`). The three namespaces
   share base impls via an include trick — a one-line change can have block-wide blast radius (§5).

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
  `high_speed::reduce` syncs after the inter-warp `s_scratch` write; `gemm_tiled` in
  `src/base/L3/gemm.cuh` syncs around each tile load (`__syncthreads()` before and after the
  inner-product over the tile). The matrix factor/solve flows (`inv.cuh`, `chol_InPlace.cuh`,
  `trsm.cuh`) sync between elimination steps.
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

### 1c. `beta = 0` GEMM still READS C → `0 * NaN = NaN` poisoning

GLASS's beta-form GEMM computes `C[cidx] = alpha*res + beta*C[cidx]` — it **reads `C` even when
`beta == 0`** (see `gemm_impl` in `src/base/L3/gemm.cuh`, and the docstrings explicitly say
"C is read; caller must initialize it"). If `C` is an **uninitialized scratch slot**, its leftover
bit pattern can be `NaN`/`Inf`, and `0 * NaN == NaN` poisons the result even though beta is 0.

- **Where this applies:** every beta-taking path — `glass::gemm` / `gemm_ex` / `gemm_impl`,
  `gemm_tiled`, `row_strided_gemm`, and the `_1d` batched variants — when called with
  `beta = 0` into a destination the caller did not initialize.
- **The fix:** the **caller must initialize `C`** before a `beta = 0` write into cold scratch
  (a tiny `for (i=rank; i<size; i+=size) C[i] = 0;` + `__syncthreads()`), OR use a GLASS
  overload that has no `beta*C` term. `gemm.cuh` provides implicit-`beta=0` overloads (the
  `gemm`/`gemm_ex` forms documented "with implicit `beta = 0`") that write `C` directly and never
  read it — prefer those for write-into-fresh-scratch.
- **The tell:** a discrepancy that is **thread-count-dependent and vanishes when isolated / run at
  1 thread** is the signature of reading cold scratch (the leftover bytes are nondeterministic
  across launches; at 1 thread the slot is more reliably overwritten before re-read). Reproduce
  flakes by running the check many times, not once. `test_gemm_strided` already parametrizes
  `(alpha,beta) = (1.0, 0.0)` for exactly this reason — keep beta=0 cases in any new GEMM test.

### 1d. Storage-order / layout-flag mistakes

`ROW_MAJOR_A` / `ROW_MAJOR_B` / `ROW_MAJOR_C` and `TRANSPOSE_B` are **compile-time template
flags** (default column-major). Passing data in the wrong order does **not crash** — it silently
indexes the wrong elements and produces wrong numbers.

- Column-major (default): `A[row + col*m]`. Row-major: `A[row*cols + col]`. `gemm_impl` selects
  per-matrix via `ROW_MAJOR_? ? A[row*n+ind] : A[ind*m+row]` etc. (see `gemm.cuh`). On the
  `glass::nvidia::` path the equivalent is the `layout` enum (`LA`, `LB`, `LC`) mapping to
  cuBLASDx `Arrangement`.
- **Catch it by testing both layouts.** Mirror `test_l3.py`: `test_gemm` (col-major), and
  `test_gemm_rowmajor` / `test_gemm_ex` feed `A.ravel()` (C-order) vs
  `np.asfortranarray(B).ravel(order='F')` (F-order) deliberately, and reshape the result with the
  matching order. If you add a layout, add the paired test; a layout bug is invisible to a
  same-layout round-trip.
- **Known restriction:** pure-SIMT `glass::gemm` with `TRANSPOSE_B=true` requires `B` square
  (n×n) — see `test_gemm_t`'s comment. The `glass::nvidia::` path (`LB=row_major`) has no such
  restriction. Don't "fix" the SIMT path to accept non-square B without updating that test.

### 1e. Shared-memory sizing — dynamic scratch must match the host-side size helper

Functions that take `extern __shared__` scratch (or an explicit scratch pointer) require the
**host launch to allocate exactly what the device side reads**, or you get out-of-bounds access.

- **`gemm_tiled` / `gemm_dispatch`:** the tiled path needs `(m*TILE + TILE*k)*sizeof(T)` bytes,
  split into `s_A = smem` and `s_B = smem + m*TILE`. The host size is computed by
  `glass_gemm_dispatch_smem<T>(m, k, block_threads, tile)` in `glass.cuh` (returns 0 when tiling
  is not warranted: `m >= 32` or `m*k > block_threads`). `test_l3.cu`'s `gemm_tiled` op hard-codes
  the matching `(m*8 + 8*k)*sizeof(float)` with `TILE=8` — if you change the tile size on one side
  you MUST change it on both. A too-small allocation overruns `s_B`.
- **`high_speed::reduce` / `dot` / `l2norm`:** need `ceil(blockDim/32)*sizeof(T)` bytes of
  `s_scratch` (one slot per warp). Under-sizing this overflows when `blockDim > 32*available`.
- **`glass::nvidia::` (CUB) L1:** scratch is `sizeof(cub::BlockReduce<T,THREADS>::TempStorage)`;
  query it with `glass::nvidia::reduce_smem_size<T,THREADS>()`. For the cuBLASDx/cuSOLVERDx
  paths, query with `gemm_smem_size<T,M,N,K[,TC]>()` / `posv_smem_size<...>()` etc. and pass the
  EXACT value to the launch — too small = OOB, and (for the default form) a wrong thread count
  deadlocks. Always query, never hard-code a guessed byte count.

### 1f. Block-tridiagonal layout contracts (`glass::banded::` / `glass::pcg::`)

The banded matvec and PCG solver carry layout/launch preconditions that fail *silently*
(wrong numbers, not a crash) if violated:

- **Pre-zero the vector pads.** Vectors are padded `(knot_points+2)*state_size` with one
  `state_size` pad block on each end. `bdmv` relies on the first/last block-rows multiplying
  their absent `L`/`R` against **zero pad** — if the pads hold garbage, the edge rows are wrong.
  `pcg::solve` zeroes its internal vectors (`set_const`), but a hand-rolled `bdmv` caller must
  zero the pads itself.
- **`pcg::solve` needs `blockDim.x` a multiple of 32.** Its inner dot is `high_speed::dot`
  (warp-shuffle); a non-warp-multiple thread count drops the partial warp's contribution. (This
  is the one place the usual "any thread count" invariance does **not** hold — it's a documented
  contract, asserted-by-convention.)
- **Shared-mem must equal `pcg::smem_elems<T,state_size,knot_points>(threads)`** — five padded
  work vectors + `ceil(threads/32)` warp-dot scratch. Under-sizing overruns; see 1e.
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
shared address. The fix (`1df6e40`, in `warp::cholDecomp_InPlace` / `warp::trsm` /
`warp::trsm_transpose`):

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

- **The compile cache is keyed on a source hash.** `conftest.py::_hash_sources` SHA-256s a
  **curated list** of files — the `.cu` under test, `cuda/helpers.cuh`, `glass.cuh`,
  `glass-cgrps.cuh`, `glass-nvidia.cuh`, and the specific `src/base/*` and `src/nvidia/*` headers
  it enumerates. `compile_binary` skips `nvcc` iff the stored `<name>.hash` matches the current
  hash AND the binary exists. So **editing a hashed header self-invalidates the cache** and the
  next `pytest` rebuilds.
- **The trap:** if you add a NEW header (e.g. a new `src/base/L3/<thing>.cuh`) and `glass.cuh`
  pulls it in, but you do NOT add it to `_hash_sources`, the cache will NOT notice your edits to
  that file — you'll test stale binaries and see phantom pass/fail. **Add new headers to the
  `_hash_sources` list** (or keep additions inside an already-hashed file).
- **Force a clean rebuild** by deleting the build artifacts: `rm -rf test/build` (or just the
  `<name>.hash` files). `test/build/` is **gitignored** — it's a scratch dir, never commit it.
- The session `bins` fixture compiles all binaries once per `pytest` session; per-binary skips are
  surfaced as fixture `pytest.skip`s (§3), so a single missing optional dep doesn't fail the run.

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
  `glass::gemm` AND `glass::cgrps::gemm` AND the SIMT fallback that `glass::nvidia::gemm`
  auto-dispatches to. After touching a base impl, run the FULL suite (`test_l1` + `test_l2` +
  `test_l3` + the nvidia dispatch/trailing-sync tests), not just the namespace you were thinking
  about. Validate that all three namespaces still produce identical numbers.
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
| Block-tridiagonal `bdmv`/`pcg` wrong at edges or non-warp thread count | Layout/launch contract (§1f) |
