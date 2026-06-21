"""L2 triangular solve / matvec (trsv/trmv) tests — GPU vs SciPy/NumPy reference.

Covers all 8 (lower, unit, trans) flag combinations across several sizes, plus a
thread-count sweep asserting thread-count invariance AND oracle agreement.
"""

import numpy as np
import pytest
import scipy.linalg
from conftest import run_op

RNG = np.random.default_rng(7)

ATOL = 1e-4
RTOL = 1e-3

FLAGS = [(lo, un, tr) for lo in (0, 1) for un in (0, 1) for tr in (0, 1)]
SIZES = [1, 2, 5, 8, 16, 33]


def _make_A(n, lower):
    """Well-conditioned (diagonally dominant) n×n triangular matrix.

    Returns the FULL dense matrix already masked to the selected triangle so the
    ignored triangle holds garbage the kernel must not read. Column-major flat is
    produced by the caller.
    """
    A = (RNG.random((n, n)).astype(np.float32) - 0.5)
    # diagonally dominant diagonal for good conditioning
    A[np.diag_indices(n)] = n + 1.0 + RNG.random(n).astype(np.float32)
    if lower:
        tri = np.tril(A)
        # poison the unused (strict upper) triangle to catch stray reads
        tri[np.triu_indices(n, k=1)] = np.nan
        return tri
    else:
        tri = np.triu(A)
        tri[np.tril_indices(n, k=-1)] = np.nan
        return tri


def _oracle_trsv(A, b, lower, unit, trans):
    # Build a clean (NaN-free) triangular matrix for the oracle.
    clean = np.where(np.isnan(A), 0.0, A).astype(np.float64)
    return scipy.linalg.solve_triangular(
        clean, b.astype(np.float64), lower=bool(lower),
        unit_diagonal=bool(unit), trans=(1 if trans else 0))


def _oracle_trmv(A, x, lower, unit, trans):
    clean = np.where(np.isnan(A), 0.0, A).astype(np.float64).copy()
    if unit:
        clean[np.diag_indices(clean.shape[0])] = 1.0
    M = clean.T if trans else clean
    return M @ x.astype(np.float64)


def _run(bins, op, threads, n, lower, unit, trans, A, x):
    # Write the REAL (NaN-bearing) matrix so the kernel must avoid reading the
    # dead triangle — any stray read poisons the result to NaN and fails the
    # allclose against the (clean) oracle.
    A_dev = np.asfortranarray(A).ravel(order='F').astype(np.float32)
    return run_op(bins["trsv"], op, str(threads),
                  args=[n, lower, unit, trans], inputs=[A_dev, x])


# ─── trsv ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lower,unit,trans", FLAGS)
@pytest.mark.parametrize("n", SIZES)
def test_trsv(bins, n, lower, unit, trans):
    A = _make_A(n, lower)
    b = (RNG.random(n).astype(np.float32) - 0.5)
    result = _run(bins, "trsv", 256, n, lower, unit, trans, A, b)
    expected = _oracle_trsv(A, b, lower, unit, trans).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL), \
        f"trsv n={n} lower={lower} unit={unit} trans={trans}\n{result}\nvs\n{expected}"


# ─── trmv ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lower,unit,trans", FLAGS)
@pytest.mark.parametrize("n", SIZES)
def test_trmv(bins, n, lower, unit, trans):
    A = _make_A(n, lower)
    x = (RNG.random(n).astype(np.float32) - 0.5)
    result = _run(bins, "trmv", 256, n, lower, unit, trans, A, x)
    expected = _oracle_trmv(A, x, lower, unit, trans).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL), \
        f"trmv n={n} lower={lower} unit={unit} trans={trans}\n{result}\nvs\n{expected}"


# ─── thread-count invariance sweep ────────────────────────────────────────────

@pytest.mark.parametrize("lower,unit,trans", FLAGS)
@pytest.mark.parametrize("op", ["trsv", "trmv"])
def test_thread_invariance(bins, op, lower, unit, trans):
    n = 33
    A = _make_A(n, lower)
    v = (RNG.random(n).astype(np.float32) - 0.5)
    if op == "trsv":
        expected = _oracle_trsv(A, v, lower, unit, trans).astype(np.float32)
    else:
        expected = _oracle_trmv(A, v, lower, unit, trans).astype(np.float32)
    ref = None
    for threads in (1, 7, 33, 256):
        result = _run(bins, op, threads, n, lower, unit, trans, A, v)
        assert np.allclose(result, expected, rtol=RTOL, atol=ATOL), \
            f"{op} threads={threads} lower={lower} unit={unit} trans={trans}: oracle mismatch"
        if ref is None:
            ref = result
        else:
            assert np.allclose(result, ref, rtol=0, atol=0) or \
                   np.allclose(result, ref, rtol=RTOL, atol=ATOL), \
                f"{op} threads={threads}: output differs from threads-1 baseline"
