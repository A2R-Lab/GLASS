# Backlog: API uniformity cleanup (deferred from the 2026-06-21 sweep)

Three read-only audits (signature/naming, batching idioms, dead-code/doc) ran after the
BLAS/LAPACK expansion. The **bugs + doc drift were fixed** (chol `sqrtf`→`sqrt` for double;
`invertMatrix_pivoted<T,N>` scratch doc; README/STARTUP/library_overview/agent-guide/
api-index/batched_1d doc drift). The items below are **deferred** because they are API
breaks (GRiD/consumer coordination) or behavior changes to core primitives — batch them
into a deliberate clean-break pass, not a pre-autotune hotfix.

## 1. Batching-idiom naming uniformity (API break — needs consumer coordination)
Adopt one canonical rule `op_qualifier` (operation first), aligning function names with
their file names:
- `row_strided_gemv` → `gemv_strided` (file is `gemv_strided.cuh`)
- `row_strided_gemm` → `gemm_strided` (file is `gemm_strided.cuh`)
- `indexed_batched_gemm` → `gemm_batched_indexed` (file is `gemm_batched_indexed.cuh`)
- `segmented_row_strided_gemv` → `gemv_segmented` (or a unified `*_batched` noun)
These are `glass::` public API that **GRiD calls** — ship with `[[deprecated]]` inline
aliases for one release. Keep the essential mechanism differences (TRANSPOSE vs
TRANSPOSE_A/_B; offset-arrays vs slot-index-arrays; per-thread vs block-stride vs K-way
vs warp-per-problem) — only the *names* are accidental.

## 2. const-correctness sweep (safe but cross-op — do together)
Read-only matrix/vector inputs are `const` in the newest ops (`trsv`/`trmv`/`potrs`/
`ldlt_solve`) but non-`const` in `syrk`/`syr2k`/`gemm`/`gemv` (A/B/x are pure inputs).
Add `const` to those inputs across all four so callers holding `const T*` data can use
them. Additive/backward-compatible; do as one sweep for uniformity (doing syrk-only would
itself be non-uniform).

## 3. Scratch-size helper naming + missing helpers
Standardize on `<op>_scratch_size` (currently a mix of `_temp_size` (iamax), `_scratch_size`
(trmv, invertMatrix_pivoted), `_smem_size` (pcg)). Add missing helpers (OOB-risk per guide
§1e): `invertMatrix_scratch_size` (=`2*dimA+1`), `invertMatrix_dense` (=`3*dimA`), fused
K-way invert (=`Σ(2*dims[m]+1)`, host helper over `dims[]`), `ldlt` (=`n+1`),
`high_speed::dot`/`reduce` (=`ceil(blockDim/32)`). Rename `iamax_temp_size`→
`iamax_scratch_size` (just shipped; cheap clean-break).

## 4. Default `reduce`/`dot` trailing-sync contract
Default `reduce`/`dot` end on a rank-0 serial fold with **no** trailing `__syncthreads()`
(result valid only when read from rank 0); every newer op and the `high_speed::`/
`low_memory::` variants DO trail-sync. Either add the trailing sync (uniform, safe — it's
already block-collective) or explicitly document the rank-0-only contract. Behavior change
to a hot core primitive → decide deliberately, verify no internal caller relies on the
current shape, A/B the perf.

## 5. Transpose/layout flag-token uniformity
"Transpose this operand" is spelled `TRANS` (trsv/syrk), `TRANSPOSE` (gemv), `TRANSPOSE_B`
(gemm). Layout is `ROW_MAJOR` (syrk, public gemm/gemv) vs `ROW_MAJOR_A/_B/_C` (the `_ex`
forms). Standardize the token (recommend `TRANS`, keeping operand-specific suffixes only
where layouts genuinely differ per matrix). Document `FillMode` as the canonical 3-way
selector (the lone enum-class flag).

## 6. Warp-surface coverage + symmetry
`warp::dot`/`warp::iamax` accept runtime `n`; the warp matrix ops are compile-time-`N`
only — pick one rule and document it. No `warp::syrk` yet (the L3 op most analogous to
`warp::gemm`). Add or document-as-intentional.

## 7. Batching menu doc page (net-new)
No single place explains "how do I batch many small problems in GLASS?" Add
`docs/source/user_guide/concepts/batching.rst`: a table of the 6 essential models
(per-thread / one-block-per-problem strided / flattened-batch block-stride / K-way fused
interleaved / TC-group SIMT 1D / warp-per-problem), when to use each, and which functions
implement each. Cross-link from CLAUDE.md "Call surfaces".
