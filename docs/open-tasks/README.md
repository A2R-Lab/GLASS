# Open tasks / backlog

Tracked, non-urgent work for GLASS. One file per topic. Keep entries concise and
delete them when done. This mirrors the convention used in the GRiD repo.

- [`cpqp_validation.md`](cpqp_validation.md) — **DONE.** The box-QP solver
  (`src/L3/box_qp.cuh`, formerly `cpqp.cuh`) was redesigned, bug-fixed, and
  validated by `test/test_qp.py`. Kept internal.
- [`qp_solver_scope.md`](qp_solver_scope.md) — open architectural question: should
  GLASS host QP/optimization solvers at all (QP is optimization, not linear
  algebra)? Gates whether `box_qp` is ever promoted to the public API.
- [`grid_wide_pcg.md`](grid_wide_pcg.md) — future work: the cooperative grid-wide
  PCG variant (`glass::cgrps::grid`) for systems too large for the single-block
  `glass::pcg::solve`.
- [`doc_comment_coverage.md`](doc_comment_coverage.md) — extend Doxygen
  doc-comments from the public entry points to remaining overloads / internals as
  desired; track API-reference completeness.
