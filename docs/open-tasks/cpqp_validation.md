# box-QP solver — VALIDATED (internal)

**Status:** DONE (2026-06-15). The former `cpqp.cuh` is now
`src/L3/box_qp.cuh` — redesigned, bug-fixed, and validated by `test/test_qp.py`.
It remains **internal** (not exported via `glass.cuh`, not in the public API
docs). Whether it should ever be promoted is gated on
[`qp_solver_scope.md`](qp_solver_scope.md).

## What it is

`glass::internal::box_qp<T>` — single-block solver for
`min 0.5 xᵀP x + qᵀx  s.t.  l ≤ x ≤ u` (P symmetric PD), via projected-gradient
descent with Armijo backtracking line search. Clean API:

```cpp
QPParams<T>  { max_iter, tol, alpha0, c, beta };   // was #defines
QPResult<T>  { converged, iters, grad_norm };
box_qp_scratch_size<T>(n)  -> element count (= 5n)
box_qp<T>(n, P, q, l, u, x /*in/out*/, scratch, params) -> QPResult
```

## What was fixed in the redesign

- **Dropped the vestigial `A`** (the old solver only gave a correct result for
  `A = I`); committed to the box `l ≤ x ≤ u`. Solution is returned **in-place in
  `x`**, not in a scratch buffer.
- **Replaced the dead `tmp6[1]` convergence test** with the proper box-QP
  optimality measure: projected-gradient inf-norm `‖x − clip(x − (Px+q), l, u)‖∞`.
- **Collapsed the 11 caller scratch buffers + `s_tmp`** into a single arena sliced
  internally; added `box_qp_scratch_size`.
- **Fixed a real scratch bug found during validation:** `low_memory::dot(n,x,y,out)`
  uses `out` as a length-`n` reduction buffer (result in `out[0]`), but the dots
  were handed single-element scalar slots → out-of-bounds writes that corrupted the
  objective and stalled the line search. The arena now gives the dots a full
  length-`n` `tmp` buffer. (compute-sanitizer memcheck/racecheck now clean.)
- Tunable params (`max_iter/tol/alpha0/c/beta`) are struct fields, not `#define`s.
- `gemm` symmetry assumption documented (`xᵀPx == xᵀ(Px)` holds for symmetric P).

## How it's validated (`test/test_qp.py`, 27 cases)

- **KKT optimality** — projected-gradient inf-norm below tolerance + box feasibility.
- **SciPy cross-check** — matches `scipy.optimize.minimize(L-BFGS-B, bounds=...)`.
- **Closed-form** — wide bounds ⇒ `x = −P⁻¹q`; diagonal P ⇒ `x = clip(−P⁻¹q, l, u)`.
- **Thread-count invariance** — identical solution at 1 / 32 / 64 / 96 / 256 threads.
- **dtypes** — float64 (tol 1e-6) and float32 (tol 1e-3).
- **Convergence flag** — reports `converged` within budget for well-conditioned P.

## Known characteristics / possible future work

- **Linear convergence** — plain projected gradient is slow to squeeze below
  ~1e-6; fine for the small, well-conditioned control QPs it targets. An
  accelerated (Nesterov) or active-set / projected-Newton method would converge
  faster. Not required for current use.
