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


# ─── gemv row-major ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n", [(8, 6), (12, 4), (16, 12)])
def test_gemv_rowmajor(bins, m, n):
    """Row-major A (C-contiguous): y = alpha*A*x + beta*y."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)   # C-order = row-major
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(m).astype(np.float32)
    y0 = y.copy()
    result = run_op(bins["l2"], "gemv_rowmajor", "simple",
                    args=[m, n, alpha, beta],
                    inputs=[A.ravel(), x, y])    # ravel of C-order is row-major flat
    expected = (alpha * A @ x + beta * y0).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n", [(8, 6), (12, 4), (16, 12)])
def test_gemv_ex(bins, m, n):
    """gemv_ex per-matrix flag (ROW_MAJOR_A=true) — same result as gemv_rowmajor."""
    alpha, beta = 2.0, 0.5
    A = RNG.random((m, n)).astype(np.float32)
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(m).astype(np.float32)
    y0 = y.copy()
    result = run_op(bins["l2"], "gemv_ex", "simple",
                    args=[m, n, alpha, beta],
                    inputs=[A.ravel(), x, y])
    expected = (alpha * A @ x + beta * y0).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── gemv_strided ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
@pytest.mark.parametrize("op,m,n,rs", [
    ("gemv_strided_6x6_6", 6, 6, 6),
    ("gemv_strided_6x6_8", 6, 6, 8),
    ("gemv_strided_4x4_4", 4, 4, 4),
    ("gemv_strided_4x4_6", 4, 4, 6),
])
def test_gemv_strided(bins, op, m, n, rs, alpha, beta):
    # A stored column-major with LDA=rs: A[i][j] = A_flat[i + j*rs]
    # Allocate rs×n storage; only first m rows are used by the kernel.
    A_storage = np.zeros((rs, n), dtype=np.float32)
    A_storage[:m, :] = RNG.random((m, n)).astype(np.float32)
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(m).astype(np.float32)
    y0 = y.copy()
    A_flat = np.asfortranarray(A_storage).ravel(order='F')
    result = run_op(bins["l2"], op, "simple",
                    args=[m, n, alpha, beta], inputs=[A_flat, x, y])
    expected = (alpha * A_storage[:m, :] @ x + beta * y0).astype(np.float32)
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
