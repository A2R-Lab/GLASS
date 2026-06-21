# GLASS expansion roadmap (2026-06) — DRAFT, pending scope sign-off

Plan for the BLAS/LAPACK gaps + custom-pattern extensions + warp buildout discussed
2026-06-21. Orchestrated as explore → **plan-review gate (main agent)** → implement
(worktree-isolated sub-agents) → **verify+merge gate (main agent)**. Sub-agents MUST
read `CLAUDE.md` and `docs/agent_debugging_guide.md` (and the named sibling header)
before writing anything — style, single-block model, thread-count invariance, barrier
rules, `beta=0` must-not-read-C, layout flags.

## Charter test for every addition
Does it serve single-block, small-matrix robotics/MPC/QP, preserve **thread-count
invariance** (identical at 1 / partial-warp / non-mult-32 / many warps), and keep the
single-block model? If not, it doesn't go in.

---

## Work packages

Each WP = one new primitive header + its **own** `test/cuda/test_<name>.cu` runner +
its **own** `test/test_<name>.py` (zero-contention, mirrors the banded/pcg/qp dedicated-
runner precedent). Sub-agents do **not** touch shared integration files.

| WP | Primitive(s) | File(s) owned by agent | Depends on | Notes |
|----|--------------|------------------------|-----------|-------|
| A | `syrk` / `syr2k` | `src/base/L3/syrk.cuh` | — | symmetric rank-k AAᵀ; exploit symmetry; decide named-triangle vs full-mirror |
| B | `trsv` / `trmv` | `src/base/L2/trsv.cuh` | — | triangular single-vec solve/mult; prereq for C + G |
| C | `posv` / `potrs` (base SIMT) | `src/base/L3/posv.cuh` | B, chol | SPD solve = chol + 2×trsv; pure-SIMT companion to `nvidia::posv` |
| D | `ldlt` (+ solve) | `src/base/L3/ldlt.cuh` | (B-like) | symmetric-indefinite → KKT/saddle; pivoting scope = decision |
| E | `iamax` | `src/base/L1/iamax.cuh` | — | argmax\|x\|; unblocks optional pivoting |
| F | fused-multi **K-way** invert + fused-multi **chol** | `src/base/L3/inv.cuh`, `src/base/L3/chol_InPlace.cuh` | — | generalize inv2/inv3 → K-way array form; same trick for chol |
| G | warp glue: `warp::{dot,axpy,copy,scal,gemv,trsv}` + warp complete-solve | inline in those L1/L2 headers + `test/cuda/test_warp.cu` | B | **GATED on a confirmed GRiD/GATO consumer** |

### Proposed signatures (sub-agents refine in Phase 1, I approve)
```cpp
// A
template <typename T> __device__ void syrk(uint32_t n, uint32_t k, T alpha,
        const T* A, T beta, T* C, bool lower=true, bool transA=false);
template <typename T, uint32_t N, uint32_t K> __device__ void syrk(T alpha, const T* A, T beta, T* C);
// B
template <typename T> __device__ void trsv(uint32_t n, const T* A, T* x,
        bool lower, bool unit, bool trans);   // x: in=b, out=solution
template <typename T> __device__ void trmv(uint32_t n, const T* A, T* x, bool lower, bool unit, bool trans);
// C
template <typename T> __device__ void posv(uint32_t n, T* A, T* b, T* s_temp);  // factor+solve, both in place
template <typename T> __device__ void potrs(uint32_t n, const T* L, T* b);
// D
template <typename T> __device__ void ldlt(uint32_t n, T* A, T* s_temp);        // -> L,D in place
template <typename T> __device__ void ldlt_solve(uint32_t n, const T* LD, T* b);
// E
template <typename T> __device__ void iamax(uint32_t n, const T* x, uint32_t* out, T* s_temp);
// F
template <typename T> __device__ void invertMatrix(uint32_t K, const uint32_t* dims,
        uint32_t MAX_DIM, T** mats, T* s_temp);                                 // keep inv2/inv3 as wrappers
template <typename T> __device__ void cholDecomp_InPlace(uint32_t K, const uint32_t* dims,
        uint32_t MAX_DIM, T** mats);                                            // fused multi-chol
```

### Batching-uniformity cleanup (separate, LAST, solo by main agent)
`gemm` has `batched_indexed`+`strided`; `gemv` has `strided`+`segmented`; L1 has none.
Cross-cutting + conflict-prone → not parallelized; done by main agent after additions land.

---

## File ownership (contention control)
- **Sub-agent (per WP, in its own git worktree):** its new header(s), its dedicated
  `test/cuda/test_<name>.cu`, its `test/test_<name>.py`, the API doc-comment in the header.
- **Main agent only (never a sub-agent):** `glass.cuh` includes, `test/conftest.py`
  (`bins` + `_hash_sources`), `docs/source/api_reference/*`, `CLAUDE.md`, `docs/HANDOFF.md`,
  `docs/agent_debugging_guide.md`. WP-F is the sole editor of `inv.cuh`/`chol_InPlace.cuh`;
  WP-G is the sole editor of the L1/L2 headers it extends — no two agents share a file.

## Orchestration protocol
1. **Phase 1 — explore + sub-plan (read-only, parallel).** One agent per WP reads the
   agent docs + sibling header, returns a detailed sub-plan: exact signature, algorithm,
   scratch-element formula, per-step barrier argument, thread-invariance argument, test
   design (numpy oracle + thread-count sweep), files touched. **No code.**
2. **Phase 1 gate (main).** I review every sub-plan, surface issues (signature
   consistency, scratch sizing, barrier placement, symmetry/`beta=0`/layout handling,
   naming), and approve or send back.
3. **Phase 2 — implement (parallel, worktree-isolated, concurrency ≤ 4–6 for RAM).**
   Approved agents implement header + dedicated runner + pytest; iterate until their
   own tests pass in-worktree.
4. **Phase 2 gate + merge (main).** Per WP: review diff, **re-run the tests myself**
   (don't trust "done"), run `racecheck` + a 1/7/32/33/64/256 thread-invariance check,
   then I do the shared-file integration (glass.cuh/conftest/docs) and merge serially,
   one commit per primitive.
5. **Phase 3 (main).** Batching-uniformity sweep, full suite, `docs make all`, HANDOFF.

## Dependency ordering
B (trsv) before C (posv) and before G (warp solve). E (iamax) before any pivoting.
A, D, E, F independent → can run first wave in parallel. G last (and gated).

## Phase 1 gate review — LOCKED decisions (2026-06-21, main agent)

Cross-cutting:
- **Shared-file model:** each impl agent MAY make its own *additive* `glass.cuh` include +
  `conftest.py` (`bins` + `_hash_sources`) entries in its worktree so its slice compiles and
  tests in isolation. Main agent resolves merge-order conflicts on those two files serially and
  is the SOLE editor of human-facing shared docs (`CLAUDE.md`, `docs/source/api_reference/*`,
  `agent_debugging_guide.md`) — agents hand over exact insert text, they do not edit them.
- **`_hash_sources` bug (WP-G catch):** `dot/axpy/copy/scal/gemv.cuh` are NOT in the hash list
  today → edits don't bust the test cache. WP-G adds those 5; every agent adds its own header.
- **trsv flag semantics are normative** (set by WP-B, mirrored by C & G): template
  `<LOWER,UNIT,TRANS>`; `LOWER`=which stored triangle holds data; `TRANS`=solve `Aᵀx=b` against
  that same-stored triangle; in-place into the RHS; **MUST end with a trailing `__syncthreads()`**
  (so posv composes without a defensive barrier). `A` is `const`.
- **Dependency / sequencing:** `posv` (C) calls `glass::trsv`, so its worktree needs WP-B's
  `trsv.cuh` to compile → **run C AFTER B merges.** A, B, D, E, F, G are mutually independent
  (G decoupled from B, see below) → run in parallel, concurrency capped ~4 for RAM (~5GB/compile).
- **WP-G ↔ WP-B decouple:** warp ops can't share the block impl (`__syncwarp`/`__shfl` vs
  `__syncthreads`). `warp::trsv` is its OWN thin wrapper over the existing
  `warp::trsm`/`trsm_transpose` + 2 new upper variants — it does NOT depend on B's core. No
  shared file: B owns new `src/base/L2/trsv.cuh`; G owns the warp block in `src/base/L3/trsm.cuh`.

Per-WP required changes (conditions of approval):
- **A (syrk):** ADD a transpose flag so BOTH `C=αAAᵀ` (n×n, Schur) AND `C=αAᵀA` (k×k, Gram/Hessian
  JᵀJ) are supported — the Hessian use needs AᵀA. Keep `FillMode{Lower,Upper,Full}`, default `Full`.
- **B (trsv):** template flags; trailing-sync contract (above); trmv = out-of-place core +
  in-place(`+scratch`) wrapper; lock flag semantics as normative.
- **C (posv):** rely on trsv trailing-sync (drop the defensive barrier); single-RHS now, NRHS
  deferred (additive); `const` L. Runs after B.
- **D (ldlt):** freeze signature `ldlt(n,A,s_temp,bool pivot=false,uint32_t* piv=nullptr)` +
  `(n+1)` advertised scratch so Bunch-Kaufman slots in later; non-pivoted now; zero-pivot
  documented-limitation test (non-strict).
- **E (iamax):** mirror `reduce.cuh` variant set (low_memory + high_speed + default) + a
  value-returning overload; document NaN-skip divergence from numpy and exclude NaN from the
  oracle; NO `warp::iamax` this round.
- **F (fused):** K-way array form + rewrite inv2/inv3 as thin wrappers (output identical);
  fused multi-chol with the diagonal distributed across K; dedicated `test_fused.cu` with a
  device `T**`; SOLE editor of `inv.cuh` + `chol_InPlace.cuh`.
- **G (warp):** decoupled from B (above); add the 5 headers to `_hash_sources`; multi-warp test
  runner (`<<<1,dim3(32,WARPS)>>>`, WARPS≥2) — the existing single-warp tests can't catch
  cross-warp bugs; SOLE editor of `dot/axpy/copy/scal.cuh`, `gemv.cuh`, and the warp block of
  `trsm.cuh`.

## Decisions (signed off 2026-06-21)
- **Scope:** FULL set A–F in wave 1.
- **Warp (WP-G):** BUILD NOW alongside the rest.
- **Pivoting:** we want **both a pivoting and a non-pivoting variant** across the board
  (perf users vs robustness users) — design every factorization/solve API to admit both
  (e.g. a `bool pivot` / template flag or paired functions). Implement the **non-pivoted**
  path now; **pivoting may be deferred** to a fast-follow if it's too heavy this round, but
  the API must not have to change to add it later. `iamax` (WP-E) is the shared pivot
  primitive.
- **Standing requirements for every WP:** (1) **sufficient tests** — numpy oracle +
  parametrized sizes + dtype (f32/f64 where the runner supports it) + **thread-count
  sweep** (1 / partial-warp / non-mult-32 / many warps) + edge cases (singular-ish,
  active constraints, beta=0); (2) **sufficient docs** — full Doxygen `/** */` on every
  public function + an API-reference entry; (3) **max parallel performance per block** —
  justify the parallelization of every loop, minimize serial sections and barriers,
  exploit structure (symmetry, triangular sparsity) to cut work.
