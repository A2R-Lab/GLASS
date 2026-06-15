"""Validation tests for the internal box-constrained QP solver
(``glass::internal::box_qp`` in ``src/L3/box_qp.cuh``).

Problem:  minimize 0.5 xᵀP x + qᵀx   subject to   l ≤ x ≤ u.

The solver is validated three ways: against a self-contained KKT / projected-
gradient optimality condition, against a SciPy bounded-minimization reference,
and against closed-form special cases (unconstrained = −P⁻¹q, separable =
clip(−P⁻¹q)). We also check thread-count invariance (the core GLASS single-block
invariant) and both float32 / float64.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest
from scipy.optimize import minimize

# Thread counts swept for the invariance test: 1, one warp, two warps, the
# production count, and a deliberate non-multiple of 32 (trailing partial warp).
THREAD_COUNTS = [1, 32, 64, 256, 96]


# --- harness ----------------------------------------------------------------
def run_box_qp(binary, P, q, l, u, x0, *, threads=64, max_iter=5000,
               tol=1e-6, dtype="f64"):
    """Invoke the test_qp runner; return (solution, converged, iters, grad_norm)."""
    n = len(q)
    npdt = np.float64 if dtype == "f64" else np.float32
    arrays = [np.asfortranarray(P).ravel(order="F"), q, l, u, x0]
    files = []
    try:
        for arr in arrays:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            np.asarray(arr, dtype=npdt).tofile(f)
            f.close()
            files.append(f.name)
        cmd = [str(binary), "box_qp", dtype, str(n), str(threads),
               str(max_iter), repr(tol)] + files
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        lines = [ln for ln in res.stdout.strip().split("\n") if ln.strip()]
        sol = np.fromstring(lines[0], sep=" ").astype(np.float64)
        info = np.fromstring(lines[1], sep=" ").astype(np.float64)
        return sol, bool(info[0]), int(info[1]), info[2]
    finally:
        for fn in files:
            os.unlink(fn)


def make_coupled_spd(rng, n):
    """Random symmetric positive-definite (well-conditioned) Hessian."""
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def scipy_box_qp(P, q, l, u):
    """Reference box-QP solution via SciPy L-BFGS-B."""
    n = len(q)
    res = minimize(lambda x: 0.5 * x @ P @ x + q @ x, np.zeros(n),
                   jac=lambda x: P @ x + q, bounds=list(zip(l, u)),
                   method="L-BFGS-B",
                   options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000))
    return res.x


def kkt_residual(P, q, l, u, x):
    """Projected-gradient inf-norm — zero iff x is the box-QP optimum (convex)."""
    g = P @ x + q
    return np.max(np.abs(x - np.clip(x - g, l, u)))


# --- closed-form special cases ----------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 7, 16])
def test_unconstrained(bins, n):
    """Wide bounds ⇒ solution is the unconstrained minimizer −P⁻¹q."""
    rng = np.random.default_rng(10 + n)
    P = make_coupled_spd(rng, n)
    q = rng.standard_normal(n)
    big = 1e6 * np.ones(n)
    sol, conv, _, _ = run_box_qp(bins["qp"], P, q, -big, big, np.zeros(n),
                                 tol=1e-7, max_iter=20000)
    ref = np.linalg.solve(P, -q)
    assert np.allclose(sol, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [3, 7, 16])
def test_separable_active_set(bins, n):
    """Diagonal P ⇒ separable ⇒ solution is clip(−P⁻¹q, l, u)."""
    rng = np.random.default_rng(20 + n)
    P = np.diag(rng.uniform(1.0, 5.0, n))
    q = rng.standard_normal(n)
    l = -0.1 * np.ones(n)
    u = 0.1 * np.ones(n)
    sol, conv, _, gn = run_box_qp(bins["qp"], P, q, l, u, np.zeros(n),
                                  tol=1e-7, max_iter=20000)
    ref = np.clip(np.linalg.solve(P, -q), l, u)
    assert np.allclose(sol, ref, atol=1e-5)
    assert conv and gn <= 1e-7


# --- general coupled box-QP vs SciPy + KKT ----------------------------------
@pytest.mark.parametrize("n", [2, 3, 7, 16, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coupled_vs_scipy(bins, n, seed):
    rng = np.random.default_rng(100 + seed * 17 + n)
    P = make_coupled_spd(rng, n)
    q = rng.standard_normal(n)
    l = -rng.uniform(0.1, 1.0, n)
    u = rng.uniform(0.1, 1.0, n)
    sol, conv, _, _ = run_box_qp(bins["qp"], P, q, l, u, np.zeros(n),
                                 tol=1e-6, max_iter=10000)
    ref = scipy_box_qp(P, q, l, u)
    assert np.allclose(sol, ref, atol=1e-4, rtol=1e-4)
    assert kkt_residual(P, q, l, u, sol) < 1e-4          # optimality
    assert np.all(sol >= l - 1e-6) and np.all(sol <= u + 1e-6)  # feasibility


def test_convergence_flag(bins):
    """A well-conditioned problem reports converged within budget."""
    rng = np.random.default_rng(5)
    n = 8
    P = np.diag(rng.uniform(2.0, 4.0, n))
    q = rng.standard_normal(n)
    sol, conv, _, gn = run_box_qp(bins["qp"], P, q, -np.ones(n), np.ones(n),
                                  np.zeros(n), tol=1e-6, max_iter=20000)
    assert conv and gn <= 1e-6


# --- single-block thread-count invariance -----------------------------------
def test_thread_invariance(bins):
    rng = np.random.default_rng(7)
    n = 12
    P = make_coupled_spd(rng, n)
    q = rng.standard_normal(n)
    l = -0.5 * np.ones(n)
    u = 0.5 * np.ones(n)
    base = None
    for t in THREAD_COUNTS:
        sol, _, _, _ = run_box_qp(bins["qp"], P, q, l, u, np.zeros(n),
                                  threads=t, tol=1e-7, max_iter=10000)
        if base is None:
            base = sol
        else:
            assert np.array_equal(sol, base), f"thread count {t} changed the result"


# --- float32 ----------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 7, 16])
def test_float32(bins, n):
    rng = np.random.default_rng(200 + n)
    P = make_coupled_spd(rng, n)
    q = rng.standard_normal(n)
    l = -rng.uniform(0.2, 1.0, n)
    u = rng.uniform(0.2, 1.0, n)
    sol, conv, _, _ = run_box_qp(bins["qp"], P, q, l, u, np.zeros(n),
                                 dtype="f32", tol=1e-3, max_iter=5000)
    ref = scipy_box_qp(P, q, l, u)
    assert np.allclose(sol, ref, atol=3e-3, rtol=3e-3)
    assert kkt_residual(P, q, l, u, sol) < 5e-3
