# Should GLASS host QP / optimization solvers at all?

**Status:** open question. Decision for now: **keep `box_qp` integrated but
unexposed.**

GLASS's charter is **single-block linear-algebra primitives** (BLAS/LAPACK-style
`__device__` routines). **Linear-system solvers are explicitly in scope** — both
direct (`chol`/`trsm`/`invertMatrix`) and iterative (`glass::pcg`); these
are public, first-class GLASS. (Confirmed 2026-06-17.)

The open question is narrower: **constrained-optimization solvers.** The `box_qp`
solver (`src/L3/box_qp.cuh`) is *optimization, not linear algebra* — it loops to
convergence with a projected-gradient line search. It reuses GLASS L1/L3
primitives and is convenient to colocate, but it arguably doesn't belong in
GLASS's public surface.

There are also sibling QP/optimization efforts in the repo's history/branches
(`qp_line_search`, `admm`, `cpddp-updates`), which makes the scope question real:
is there an intent to grow a *family* of solvers?

## The decision to make later

- **(a) Grow a deliberate optimization module** — if more solvers are wanted
  (projected-Newton/active-set box-QP, ADMM, interior-point, …), give them a home:
  either a clearly-separated `glass`-adjacent namespace/module, or a **separate
  library** that depends on GLASS for its linear algebra. Then promote `box_qp`
  into it with a public API + docs + examples.
- **(b) Keep QP out of GLASS entirely** — host solvers in the consumer
  (GRiD / GATO / PDDP) and keep GLASS purely linear algebra. `box_qp` would then
  move out of this repo.

## Why it matters

This gates whether `box_qp` (now validated — see `cpqp_validation.md`) ever gets
**promoted to the public API**. Promotion is blocked on resolving *this* scope
question, not on validation. Until then `box_qp` stays an internal, tested
utility: not in `glass.cuh`, not in the API reference, not in `examples/`.
