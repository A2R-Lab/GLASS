# GLASS handoff

Living status doc. Update the top as work lands. For onboarding read
`docs/STARTUP_PROMPT.md` first.

## 2026-06-15 — box-QP solver validated (internal)

Redesigned, fixed, and validated the orphaned QP solver.
- `src/L3/cpqp.cuh` → **`src/L3/box_qp.cuh`**: honest box-QP
  (`min 0.5xᵀPx+qᵀx s.t. l≤x≤u`), clean struct API (`QPParams`/`QPResult`/
  `box_qp_scratch_size`), solution in-place, `A`/`s_tmp` dropped, proper
  projected-gradient KKT stopping test.
- **Found + fixed a real OOB bug:** the dot accumulators were single scalars but
  `low_memory::dot` needs a length-`n` reduction buffer → out-of-bounds writes
  corrupted the objective. Arena is now `5n`; compute-sanitizer clean.
- **`test/test_qp.py`** (27 cases): KKT optimality, SciPy cross-check, closed-form
  cases, thread-count invariance (1..256), f32 + f64. Full suite **358 passed**.
- Kept **internal** — not in `glass.cuh`/public docs/examples. Whether to promote
  is gated on `docs/open-tasks/qp_solver_scope.md` (QP is optimization, not LA).
- Standalone `src/L3/test_cpqp.cu` removed (replaced by the harness runner
  `test/cuda/test_qp.cu`).

## 2026-06-15 — developer-experience parity + legacy cleanup

Brought GLASS's developer experience up to parity with the GRiD repo
(docs site, website deploy, agent files, examples, project-info) and removed a
stale duplicate source tree.

**Docs site (new).** Sphinx (`pydata-sphinx-theme`) under `docs/`, with the API
reference generated from header Doxygen doc-comments via Breathe.
- `docs/{Makefile,Doxyfile,requirements.txt}`, `docs/source/conf.py`.
- Narrative pages under `docs/source/user_guide/` (getting_started, concepts,
  tutorials) ported from `README.md`, `bench/INSTALL.md`, `bench/TUNING.md`, and
  the batched-1D RFC.
- API reference under `docs/source/api_reference/` (`.. doxygenfile::` per
  header).
- Build: `cd docs && make all` (needs `doxygen` + `pip install -r docs/requirements.txt`).

**Doc-comments (new).** Added Doxygen `/** */` blocks to the public entry points
across `src/base/**`, `src/cgrps/**`, `src/nvidia/**`, and the top-level headers.
`Doxyfile` uses `EXTRACT_ALL = NO`, so undocumented internals are excluded.

**Website (new, LIVE).** `.github/workflows/gh-pages.yml` builds the docs
(installs Doxygen) and deploys to GitHub Pages on push to `main`. Pages is
enabled (Settings → Pages → Source = GitHub Actions) and serving.

**Agent files (new).** `CLAUDE.md`, `docs/agent_debugging_guide.md`,
`docs/STARTUP_PROMPT.md`, this file, and `docs/open-tasks/`.

**Examples (new).** `examples/` — six standalone compilable `.cu` programs
(01–05 pure-SIMT; 06 requires MathDx) + a README.

**Project-info (new).** `CONTRIBUTING.md`, `.clang-format` (advisory),
`test/README.md`, `test/TESTING_STRATEGY.md`, root `.gitignore`.

**CI.** Docs deploy is the only active workflow. `.github/workflows/test.yml` is
a disabled self-hosted-GPU stub (GitHub-hosted runners have no GPU) — enable it
once a `gpu`-labelled runner exists.

**Legacy cleanup (code change).** Deleted the pre-refactor duplicate dirs
`src/L1`, `src/L2`, and the duplicate primitives in `src/L3` (24 files) — they
were superseded by the May-2026 `src/base/**` refactor and nothing included them.
The orphaned `src/L3/cpqp.cuh` box-QP solver (+ `test_cpqp.cu`) was the only
dependent; it was **rewired onto the `base/` API** and now compiles
(`nvcc ... -c src/L3/test_cpqp.cu`, sm_120, clean). It remains **unvalidated and
undocumented** — see `docs/open-tasks/cpqp_validation.md`.

**Untouched (already strong):** the `test/` suite and `bench/` were left as-is
(only added `test/README.md` + `test/TESTING_STRATEGY.md`). The 799-line
`README.md` remains the canonical user-facing reference.

### Verification done
- `nvcc` compile of `test_cpqp.cu` against `base/` (pre- and post-deletion): clean.
- Docs build (`cd docs && make all`) and `pytest test/`: see the latest run notes.

### Next / open
- ~~Enable GitHub Pages in repo settings~~ — DONE; site is live.
- `docs/open-tasks/cpqp_validation.md` — validate or remove cpqp.
- `docs/open-tasks/doc_comment_coverage.md` — optional long-tail doc-comments.
