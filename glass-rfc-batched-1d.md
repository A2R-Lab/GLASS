# GLASS feature handoff — features needed for GRiD-A2R's `glass-nvidia` path

Target audience: an agent picking up implementation work on the GLASS repo
(submodule of GRiD-A2R, also independently usable). This doc explains what's
needed, why, and how to implement each piece. Read end-to-end before starting
any single item — a few are interdependent.

---

## 0. Context

**GLASS** is a header-only CUDA library for small in-block linear algebra.
It vendors two paths:
- `glass::gemm<T,M,N,K>` — **SIMT** compile-time-unrolled GEMM for small
  matrices.
- `glass::nvidia::gemm<T,M,N,K,BLOCK_THREADS,...>` — **cuBLASDx**-backed
  packed GEMM for shapes that win on tensor cores.

Source layout:
```
GLASS/
  src/nvidia/
    types.cuh       — layout enum, public type forward decls
    sizes.cuh       — smem-size computations
    query.cuh       — host-side `gemm_min_block_threads`, validity checks
    l1.cuh          — dot products, vector reductions
    l2.cuh          — GEMV, row-strided GEMV
    l3.cuh          — GEMM, transpose-B GEMM, row_strided_gemm, gemm_batched
    lapack.cuh      — cuSOLVERDx wrappers (chol, posv, trsm, ...)
  bench/
    bench_gemm.cu, bench_gemm_batched.cu, bench_blockdim.cu, bench_gemv.cu,
    bench_lapack.cu, bench_reduce.cu, run_bench.py
  glass.cuh, glass-nvidia.cuh  — public entry points
```

**GRiD-A2R** is the consumer. Its codegen generates per-robot CUDA headers
that call `grid_linalg_gemm<T,M,N,K,...>` (thin wrappers around
`glass::gemm` or `glass::nvidia::gemm`). Every kernel in GRiD launches
with **1D thread blocks** (`dim3(MAX_PERF_LEVEL_THREADS, 1, 1)`).

**The problem.** Today's `glass::nvidia::gemm_batched` requires a **2D launch
(`dim3(TC, BATCH)`)** to give each batch element a `threadIdx.y` slot. GRiD
can't use it without rewriting every kernel's launch geometry. The features
below remove or work around that requirement and add several adjacent wins.

---

## 1. Motivating bug (P0)

In commit `<TBD>` of GRiD-A2R, the codegen for `end_effector_pose_gradient`
tried to express "one shared `A` × `N` consecutive 4×4 `B` blocks" as a
single `glass::nvidia::gemm<T,4,4,4*n*run_len>` call (where `n =
NUM_JOINTS`, `run_len = number of consecutive equal-parent end-effectors`).

For go2 (n=12, run_len=1) this expanded to a 4×4×48 GEMM with A pointing
into `s_Xhom[16*parent_jid]`. Problem: cuBLASDx interprets that as one big
GEMM that **sums over K=48** (one 4×4 result), not as the intended **48
independent 4×4 results sharing one A**. The math is wrong AND it reads
past `s_Xhom` (which is only `n*16 = 192` elements) — runtime
`cudaErrorIllegalAddress`.

**Current workaround** in
[GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py](../GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py):
the entire `#if GRID_CUDA_USE_GLASS_NVIDIA` block was deleted; codegen now
always emits the pure-SIMT loop with per-element `dot_prod<T,4,4,1>`. This
unblocks builds but leaves perf on the table on humanoid robots (g1, atlas)
where consecutive same-parent runs are common.

**Proper fix needs**: a batched GEMM API that works inside GRiD's 1D launch.
That's P0-1 below.

---

## 2. Priorities

| ID | Item | Priority | Effort | Unblocks |
|---|---|---|---|---|
| P0-1 | 1D-launch `gemm_batched_1d` for small shapes | P0 | 1-2 days | EE-pose-gradient on g1/atlas |
| P0-2 | Stride-0 / shared-A convenience helper | P0 | 0.5 day | Codegen ergonomics |
| P1-3 | `should_use_cublasdx<M,N,K,SM>()` auto-select | P1 | 1 day | Removes tuning knob in GRiD |
| P1-4 | Small-GEMM SIMT fast path under `glass::nvidia::gemm` | P1 | 0.5-1 day | All GRiD GEMM call sites win on glass-nvidia |
| P1-5 | 1D-launch variants of `row_strided_gemv` if needed | P1 | 0.5 day | Maybe a no-op (check) |
| P2-6 | Warp-resident no-sync GEMM for ≤6×6 | P2 | 1-2 days | Single-call timings |
| P2-7 | Static asserts / clear errors on misuse | P2 | 0.5 day | Debugging |

P0-1 + P0-2 should ship together since they share an implementation.

---

## 3. P0-1: 1D-launch `gemm_batched_1d`

### 3.1 The fundamental challenge

`glass::nvidia::gemm_batched` (the 2D-launch version) works because
cuBLASDx's `Block()` policy reads `threadIdx.x` directly to assign work
within a single GEMM. The 2D launch puts BATCH on `threadIdx.y`, so each
batch element gets a fresh `threadIdx.x` range from 0 to TC-1.
[`GLASS/src/nvidia/l3.cuh:412-460`](../GLASS/src/nvidia/l3.cuh#L412):

```cpp
// 2D-launch existing version:
const uint32_t b = threadIdx.y;        // batch index from y-dim
char* my_smem = smem + b * per_batch_smem;
float* a = A[b]; ...
GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);
```

For **1D** launch, all threads share `threadIdx.y == 0` and have
`threadIdx.x ∈ [0, TC*BATCH)`. cuBLASDx's collective operations
(`cublasdx::copy`, `GEMM().execute()`) read `threadIdx.x` directly and
assume the full block participates — there's no built-in way to tell
cuBLASDx "use only threads `[b*TC, (b+1)*TC)` for this batch element".

So a clean 1D-launch batched GEMM **cannot use cuBLASDx unchanged**. Two
viable implementation strategies:

### 3.2 Strategy A — SIMT-batched (recommended for small shapes)

For M,N,K ≤ ~8 (our entire dynamics workload), cuBLASDx is not faster than
SIMT anyway (see your README's "Choosing the right backend"). For these
shapes, write a **pure-SIMT batched GEMM** that runs `BATCH` GEMMs in
parallel via thread-block partitioning:

```cpp
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
__device__ void gemm_batched_1d(T alpha, T* const* A, T* const* B,
                                T beta,  T* const* C)
{
    // Each batch element gets TC threads; total block has TC*BATCH threads.
    // Each thread computes M*N/TC output elements (round up).
    const uint32_t b   = threadIdx.x / TC;          // which batch
    if (b >= BATCH) return;                          // safety
    const uint32_t tx  = threadIdx.x - b * TC;       // local thread-id within batch
    const uint32_t mn  = M * N;
    // distribute mn output cells across TC threads
    for (uint32_t i = tx; i < mn; i += TC) {
        uint32_t row = i % M, col = i / M;
        T acc = T(0);
        #pragma unroll
        for (uint32_t k = 0; k < K; k++) {
            // A access — column-major: A[row + k*M]; if A_RS != M, adjust
            T a_val = A[b][row + k * /*LDA*/ M];
            // B access — column-major: B[k + col*K]
            T b_val = B[b][k + col * /*LDB*/ K];
            acc += a_val * b_val;
        }
        C[b][row + col * /*LDC*/ M] = alpha * acc + beta * C[b][row + col * /*LDC*/ M];
    }
}
```

**No `__syncthreads()` needed** because threads within one batch don't
exchange data. **No shared memory needed** for the basic version (each
thread writes its own output cells directly).

For layouts: replicate the `_GLASS_CUBLAS_LAYOUT` mapping for SIMT — handle
row_major vs col_major in the inner loop.

**Why this is fast for small shapes:**
- No cuBLASDx tile-loading overhead (which dominates for 4×4).
- Compile-time `#pragma unroll` over M, N, K makes the inner loop pure
  arithmetic.
- All BATCH batches run truly in parallel (different threads).

### 3.3 Strategy B — cuBLASDx with sub-block dispatch via cooperative groups

If we want to support **large** batched shapes (M,N,K ≥ 16 where cuBLASDx
wins), we'd need cooperative groups partitioning OR a custom cuBLASDx
plumb-through that accepts a thread offset. This is significantly harder
because cuBLASDx's collectives don't accept thread-group arguments.

Recommendation: **don't implement Strategy B**. For the cases where cuBLASDx
wins, GRiD callers can pay the kernel-launch refactor cost (a future PR3
item). Strategy A covers GRiD's actual needs today.

### 3.4 API surface to add to `GLASS/src/nvidia/l3.cuh`

```cpp
// 1D-launch batched GEMM (Strategy A: SIMT). Each batch element gets TC
// threads from a single 1D thread block of TC*BATCH total. Use for small
// shapes where cuBLASDx loses to SIMT anyway (M,N,K ≤ ~8).
//
// Launch: kernel<<<grid, dim3(TC*BATCH, 1, 1), 0>>>(...)
// No shared memory required.
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
__device__ void gemm_batched_1d(T alpha, T* const* A, T* const* B,
                                T beta,  T* const* C);

// Threads required for the 1D-launch batched variant.
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t TC>
constexpr uint32_t gemm_batched_1d_threads() { return TC * BATCH; }
```

No `_smem_size` helper needed (uses no shared memory in the basic version).
No `DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM` macro needed (no cuBLASDx
specialization, so no per-shape pre-instantiation required) — but you may
still want one for consistency with the 2D-launch API.

### 3.5 Testing

Add `GLASS/bench/bench_gemm_batched_1d.cu` modeled on the existing
`bench_gemm_batched.cu`:

- For each (M, N, K, BATCH) in `{(4,4,4,1), (4,4,4,2), (4,4,4,4), (4,4,4,8),
  (6,6,6,1), (6,6,6,2), (6,6,6,4)}`, compare:
  - **naive_1d**: `BATCH` sequential `glass::gemm<T,M,N,K>` calls in one block
  - **batched_1d**: one `glass::nvidia::gemm_batched_1d<T,M,N,K,BATCH,TC>` call
- For correctness, also run a CPU reference (loop of M*N*K mults).
- Expected: `batched_1d` should be roughly `1×` of naive for BATCH=1 and
  scale up to BATCH× speedup as BATCH increases (limited by available
  threads).

### 3.6 GRiD-A2R consumer-side validation

After P0-1 lands in GLASS, update the GRiD codegen in
[_eepose_gradient_hessian.py](../GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py):

```python
# Reinstate the broken nvidia path, fixed:
self.gen_add_code_line("#if GRID_CUDA_USE_GLASS_NVIDIA")
if packed_parent_runs:
    self.gen_add_code_line("// Batched 1D GEMM for consecutive equal-parent runs.")
    for run_start, run_len, parent_jid in packed_parent_runs:
        # Build pointer arrays in shared memory (or registers if small).
        self.gen_add_code_line(f"const T* _a_ptrs[{run_len}];")
        self.gen_add_code_line(f"T* _b_ptrs[{run_len}];")
        self.gen_add_code_line(f"T* _c_ptrs[{run_len}];")
        for i in range(run_len):
            self.gen_add_code_line(f"_a_ptrs[{i}] = &s_Xhom[16*{parent_jid}];")
            self.gen_add_code_line(f"_b_ptrs[{i}] = &s_eeTemp[{tempSrcOffset + 16*n*(run_start+i)}];")
            self.gen_add_code_line(f"_c_ptrs[{i}] = &s_eeTemp[{tempDstOffset + 16*n*(run_start+i)}];")
        self.gen_add_code_line(
            f"glass::nvidia::gemm_batched_1d<T,4,4,4,{run_len},MAX_PERF_LEVEL_THREADS>("
            f"static_cast<T>(1), _a_ptrs, _b_ptrs, static_cast<T>(0), _c_ptrs);"
        )
# ... SIMT path stays as the #else fallback
```

Bench on iiwa14 / go2 / g1 EE_POSE_GRADIENT compute-only at N=256 on sm_120
and sm_86. Expected wins on g1 (where `run_len > 1`); roughly neutral on
iiwa14/go2 (where `run_len = 1` and the batched call degenerates to one
SIMT GEMM).

---

## 4. P0-2: Stride-0 / shared-A convenience helper

GRiD's common case is "single shared A across all BATCH ops" (the EE-pose
gradient case literally has `&s_Xhom[16*parent_jid]` as A for every batch
element). Constructing 3 pointer arrays and broadcasting A through them is
ergonomically annoying in codegen.

### 4.1 API

```cpp
// Stride-0 A: single shared matrix broadcast across BATCH ops.
// B_stride, C_stride: element stride between consecutive batch elements'
// matrices in B/C respectively. M*K for tightly packed.
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t TC,
          uint32_t B_STRIDE = M * K,      // tight-packed default
          uint32_t C_STRIDE = M * N,
          /* layouts */>
__device__ void gemm_strided_batched_1d(T alpha, const T* A_shared,
                                         const T* B, T beta, T* C);
```

Caller passes a single A pointer + base B/C pointers; GLASS internally
indexes `B[b * B_STRIDE]`, `C[b * C_STRIDE]`. No pointer-array setup.

### 4.2 Implementation

Trivial layer over P0-1's Strategy A:

```cpp
// (Strategy A inner loop with these replacements)
T a_val = A_shared[row + k * M];                                  // no [b] index
T b_val = B[b * B_STRIDE + k + col * K];                          // strided B
C[b * C_STRIDE + row + col * M] = alpha * acc + beta * C[...];   // strided C
```

### 4.3 Consumer-side win

The GRiD codegen for the EE-pose case becomes one line:

```python
self.gen_add_code_line(
    f"glass::nvidia::gemm_strided_batched_1d<T,4,4,4,{run_len},MAX_PERF_LEVEL_THREADS>("
    f"static_cast<T>(1), &s_Xhom[16*{parent_jid}], "
    f"&s_eeTemp[{tempSrcOffset + 16*n*run_start}], static_cast<T>(0), "
    f"&s_eeTemp[{tempDstOffset + 16*n*run_start}]);"
)
```

No pointer arrays. Much cleaner generated code.

---

## 5. P1-3: `should_use_cublasdx<M,N,K,SM>()` constexpr auto-select

### 5.1 Motivation

GRiD currently picks at codegen time via `linalg_smem_for(M, N, K)` with a
runtime-tunable threshold (`GRID_BENCH_NVIDIA_MIN_DIM` env var, default 16).
The heuristic lives in GRiD and has to be re-tuned per arch when GLASS
changes its internal tile sizes. The heuristic should live in **GLASS**,
which knows its own tile sizes per SM.

### 5.2 API

```cpp
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx() {
    // SOM machine-readable heuristic. Returns true iff cuBLASDx is expected
    // to win for this (T, M, N, K) on SM_VAL. Today: T==float && max(M,N,K)
    // >= 16 (plus floored: min(M,N,K) >= 4 to avoid cuBLASDx's tile bugs on
    // K=1). Implementation can be refined per-SM as data lands.
}
```

### 5.3 Implementation

Phase 1: literal hardcoded heuristic matching GRiD's current
`linalg_smem_for`:

```cpp
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM_VAL>
constexpr bool should_use_cublasdx() {
    constexpr uint32_t mx = (M > N ? (M > K ? M : K) : (N > K ? N : K));
    constexpr uint32_t mn = (M < N ? (M < K ? M : K) : (N < K ? N : K));
    return std::is_same<T, float>::value && mx >= 16 && mn >= 4;
}
```

Phase 2 (future): consult an internal benchmark table (e.g. `query.cuh`
loads a constexpr lookup from compiled-in data per SM). GRiD just calls
the helper; it benefits automatically when GLASS updates.

### 5.4 Consumer side

GRiD's [_lin_alg_helpers.py:133-162](../GRiDCodeGenerator/helpers/_lin_alg_helpers.py#L133-L162)
`linalg_smem_for(m, n, k)` calls `glass::nvidia::should_use_cublasdx<T,M,N,K>`
at codegen time (via a Python-side reimplementation that mirrors the C++
heuristic, OR via a small generated stub that codegen emits and then reads
back at GRiD compile time). Drop `GRID_BENCH_NVIDIA_MIN_DIM` env var.

---

## 6. P1-4: Small-GEMM SIMT fast path inside `glass::nvidia::gemm`

### 6.1 Motivation

GRiD's data on sm_120 shows `glass::nvidia::gemm` ≈ `glass::gemm` for
iiwa14 (all 6×6×6). On smaller (4×4×4) shapes cuBLASDx is materially slower
than SIMT (our bug above is one symptom; the broader pattern is
constants-dominated cuBLASDx tile init).

### 6.2 Proposed change

In `glass::nvidia::gemm<T,M,N,K,...>`, add a compile-time dispatch:

```cpp
template <typename T, uint32_t M, uint32_t N, uint32_t K, ...>
__device__ void gemm(T alpha, T* A, T* B, T beta, T* C, char* smem) {
    if constexpr (!should_use_cublasdx<T, M, N, K, SM_VAL>()) {
        // Fall back to SIMT path internally — no caller change needed.
        glass::gemm<T, M, N, K, /* layouts */>(alpha, A, B, beta, C);
        return;
    }
    // ... existing cuBLASDx path
}
```

### 6.3 Result

GRiD codegen no longer needs to think about cutover — it always emits
`glass::nvidia::gemm<...>` for the `glass-nvidia` build, and GLASS
internally picks the right impl per shape. Removes a tuning knob.

---

## 7. P1-5: 1D-launch GEMV variant audit

`glass::nvidia::row_strided_gemv` is used by GRiD's ABA forward pass +
several other call sites. Confirm in the existing GLASS code:
[GLASS/src/nvidia/l2.cuh](../GLASS/src/nvidia/l2.cuh) — does
`row_strided_gemv` require 2D launch like the batched GEMM does, or does
it natively work in 1D? If 2D, add a 1D variant. If 1D already, this item
is a no-op; just document.

GRiD call sites to check that they're consistent with whatever 1D contract
exists: [_aba.py:83, 240, 352, 602](../GRiDCodeGenerator/algorithms/_aba.py)
and similar in `_inverse_dynamics.py`.

---

## 8. P2-6 / P2-7

Lower priority; do after P0/P1 land and ship.

- **P2-6 — Warp-resident no-sync GEMM** for ≤6×6 — speculation, may not
  beat compiler auto-vectorization. Bench first; only land if measurably
  wins.
- **P2-7 — Diagnostics** — extend the existing `gemm_block_threads_valid<>`
  static_assert pattern to batched + row_strided + small-GEMM paths. Goal:
  any misuse triggers a compile error or a runtime `printf` before the
  illegal memory access.

---

## 9. Execution order

| Phase | Items | Why |
|---|---|---|
| 1 | P0-1 SIMT-batched gemm_batched_1d + bench | Unblocks the actual bug shipped today |
| 2 | P0-2 strided-batched (one-line layer over P0-1) | Codegen ergonomics |
| 3 | GRiD-A2R consumer update + benchmark validation | Closes the loop |
| 4 | P1-3 `should_use_cublasdx` constexpr | Cleanup; removes GRiD tuning knob |
| 5 | P1-4 internal SIMT fallback under `nvidia::gemm` | Lets GRiD always use the nvidia path |
| 6 | P1-5 GEMV audit | Probably no-op |
| 7 | P2-6, P2-7 | Polish |

---

## 10. Reference materials

For implementers picking this up:

- **Existing 2D-launch `gemm_batched`** in
  [GLASS/src/nvidia/l3.cuh:412-487](../GLASS/src/nvidia/l3.cuh#L412-L487) —
  the precise template specialization pattern and macro structure to
  mirror.
- **Existing bench** `bench_gemm_batched.cu` — exact test structure.
- **The GRiD-A2R bug** that motivated this:
  [GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py:516](../GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py#L516)
  (the broken line was deleted; see Section 1).
- **GRiD-A2R consumer side wrappers** (where the new GLASS APIs will be
  called from): [GRiDCodeGenerator/helpers/_lin_alg_helpers.py](../GRiDCodeGenerator/helpers/_lin_alg_helpers.py).
- **cuBLASDx docs**: https://docs.nvidia.com/cuda/cublasdx/

---

## 11. Validation criteria for hand-back

The implementing agent should be done when:

1. P0-1 + P0-2 land in GLASS with new bench files (and the existing bench
   suite still passes — no regression on the 2D-launch path).
2. GRiD-A2R consumer-side codegen is updated, builds clean, runs without
   `cudaErrorIllegalAddress` on go2/g1 fixed + floating.
3. Numerical equivalence verified: regenerated EE-pose-gradient output
   matches the current SIMT-only output to float32 precision on iiwa14,
   go2, g1.
4. Bench numbers reported for EE-pose-gradient on g1 fixed glass vs
   glass-nvidia at N=256 on sm_120: glass-nvidia should be ≥ 1.0× glass
   (no regression; ideally a small win from the batched call).
5. Optional: P1-3 + P1-4 land and `GRID_BENCH_NVIDIA_MIN_DIM` is dropped
   from GRiD-A2R (cleanup PR).

---

## 12. Open questions for review before implementation

- Is **Strategy A (SIMT-batched)** good enough, or do we want the harder
  Strategy B (cuBLASDx-batched with cooperative-group partitioning) for
  larger shapes? Recommend Strategy A only unless GRiD's data shows we
  need batched cuBLASDx for some shape.
- For the strided variant (P0-2), should we expose `A_stride` too (allowing
  a strided-A use case) or only `B_stride`/`C_stride` (shared-A only)?
  Recommend shared-A only — it's GRiD's only use case and the API is
  cleaner. Add A_stride later if needed.
- Should P1-3 / P1-4 (auto-select + internal SIMT fallback) be combined
  into one PR? They're conceptually paired. Recommend yes.
