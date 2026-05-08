"""L2 GLASS function tests — compare GPU results to NumPy reference."""

import numpy as np
import pytest
from conftest import run_op

RNG = np.random.default_rng(42)

ATOL = 1e-4
RTOL = 1e-3

CG_SIMPLE = ["cg", "simple"]


# ─── gemv ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n", [(4, 6), (8, 8), (16, 12)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemv(bins, m, n, version):
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(m).astype(np.float32)
    y0 = y.copy()
    # A stored column-major
    A_col = np.asfortranarray(A)
    result = run_op(bins["l2"], "gemv", version,
                    args=[m, n, alpha, beta],
                    inputs=[A_col.ravel(order='F'), x, y])
    expected = (alpha * A @ x + beta * y0).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n", [(4, 6), (8, 8), (16, 12)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemv_t(bins, m, n, version):
    # y = alpha * A^T * x + beta * y  (A: mxn, x: m, y: n)
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    x = RNG.random(m).astype(np.float32)
    y = RNG.random(n).astype(np.float32)
    y0 = y.copy()
    A_col = np.asfortranarray(A)
    result = run_op(bins["l2"], "gemv_t", version,
                    args=[m, n, alpha, beta],
                    inputs=[A_col.ravel(order='F'), x, y])
    expected = (alpha * A.T @ x + beta * y0).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── ger ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n", [(4, 6), (8, 8), (16, 12)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_ger(bins, m, n, version):
    alpha = 0.5
    x = RNG.random(m).astype(np.float32)
    y = RNG.random(n).astype(np.float32)
    A = RNG.random((m, n)).astype(np.float32)
    A0 = A.copy()
    A_col = np.asfortranarray(A)
    result = run_op(bins["l2"], "ger", version,
                    args=[m, n, alpha],
                    inputs=[x, y, A_col.ravel(order='F')])
    expected = (A0 + alpha * np.outer(x, y)).astype(np.float32)
    mat = result.reshape(m, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)
