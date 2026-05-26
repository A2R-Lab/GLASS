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


# ─── segmented_row_strided_gemv ──────────────────────────────────────────────
# `segments` independent 6x6 col-major (LDA=6) GEMVs computed concurrently.
# Descriptor arrays give per-segment base element offsets into packed buffers.
# numpy does each GEMV independently as the reference.

def _build_segmented(segments, m, n, rs, rng, with_S=False):
    """Pack `segments` GEMVs into contiguous A/x/y buffers with per-seg offsets."""
    A_blocks, x_blocks, y_blocks, S_blocks = [], [], [], []
    a_off, x_off, y_off, s_off = [], [], [], []
    A_cur = x_cur = y_cur = s_cur = 0
    A_list, x_list, y_list, S_list = [], [], [], []
    for _ in range(segments):
        Aseg = rng.random((rs, n)).astype(np.float32)   # rs x n storage, LDA=rs
        Aseg[m:, :] = 0.0
        xseg = rng.random(n).astype(np.float32)
        yseg = rng.random(m).astype(np.float32)
        a_off.append(A_cur); x_off.append(x_cur); y_off.append(y_cur)
        A_list.append(np.asfortranarray(Aseg).ravel(order='F'))
        x_list.append(xseg); y_list.append(yseg)
        A_cur += rs * n; x_cur += n; y_cur += m
        A_blocks.append(Aseg[:m, :]); x_blocks.append(xseg); y_blocks.append(yseg)
        if with_S:
            Sseg = rng.random(m).astype(np.float32)
            s_off.append(s_cur); S_list.append(Sseg); s_cur += m
            S_blocks.append(Sseg)
    A = np.concatenate(A_list).astype(np.float32)
    x = np.concatenate(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    out = dict(A=A, x=x, y=y, a_off=np.array(a_off, np.float32),
               x_off=np.array(x_off, np.float32), y_off=np.array(y_off, np.float32),
               A_blocks=A_blocks, x_blocks=x_blocks, y_blocks=y_blocks)
    if with_S:
        out["S"] = np.concatenate(S_list).astype(np.float32)
        out["s_off"] = np.array(s_off, np.float32)
        out["S_blocks"] = S_blocks
    return out


@pytest.mark.parametrize("segments", [1, 3, 5])
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
def test_segmented_row_strided_gemv_nofuse(bins, segments, alpha, beta):
    m, n, rs = 6, 6, 6
    d = _build_segmented(segments, m, n, rs, RNG)
    y0 = d["y"].copy()
    result = run_op(
        bins["l2"], "seg_gemv_6x6_nofuse", "simple",
        args=[m, n, segments, alpha, beta, d["A"].size, d["x"].size, d["y"].size],
        inputs=[d["a_off"], d["x_off"], d["y_off"], d["A"], d["x"], d["y"]])
    expected = []
    for s in range(segments):
        yseg0 = d["y_blocks"][s]
        expected.append(alpha * d["A_blocks"][s] @ d["x_blocks"][s] + beta * yseg0)
    expected = np.concatenate(expected).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("segments", [1, 3, 5])
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
def test_segmented_row_strided_gemv_fuse(bins, segments, alpha, beta):
    m, n, rs = 6, 6, 6
    d = _build_segmented(segments, m, n, rs, RNG, with_S=True)
    scalar = (RNG.random(segments) - 0.5).astype(np.float32)
    result = run_op(
        bins["l2"], "seg_gemv_6x6_fuse", "simple",
        args=[m, n, segments, alpha, beta,
              d["A"].size, d["x"].size, d["y"].size, d["S"].size],
        inputs=[d["a_off"], d["x_off"], d["y_off"], d["A"], d["x"], d["y"],
                d["s_off"], d["S"], scalar])
    expected = []
    for s in range(segments):
        gemv = alpha * d["A_blocks"][s] @ d["x_blocks"][s] + beta * d["y_blocks"][s]
        gemv = gemv + d["S_blocks"][s] * scalar[s]
        expected.append(gemv)
    expected = np.concatenate(expected).astype(np.float32)
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
