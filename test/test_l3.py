"""L3 GLASS function tests — compare GPU results to NumPy/SciPy reference."""

import numpy as np
import pytest
from conftest import run_op

RNG = np.random.default_rng(42)

ATOL = 1e-3
RTOL = 1e-3

CG_SIMPLE = ["cg", "simple"]


def make_spd(n):
    """Generate a random n x n symmetric positive definite matrix."""
    A = RNG.random((n, n)).astype(np.float32)
    return (A @ A.T + n * np.eye(n, dtype=np.float32))


def make_lower_triangular(n):
    """Generate a random n x n lower triangular matrix with positive diagonal."""
    L = np.tril(RNG.random((n, n)).astype(np.float32))
    np.fill_diagonal(L, np.abs(L.diagonal()) + 0.5)
    return L


# ─── gemm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemm(bins, m, n, k, version):
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm", version,
                    args=[m, n, k, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n", [(4, 6), (8, 8)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemm_t(bins, m, n, version):
    # GLASS gemm_t computes C(m×n) = alpha * A(m×n) * B(n×n)^T + beta * C(m×n)
    # B must be square n×n (this is how the GLASS coop-groups implementation works).
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, n)).astype(np.float32)
    C = RNG.random((m, n)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_t", version,
                    args=[m, n, n, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B.T + beta * C0).astype(np.float32)
    mat = result.reshape(m, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm row-major ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6)])
def test_gemm_rowmajor(bins, m, n, k):
    """All matrices row-major (C-contiguous): C = alpha*A*B + beta*C."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)   # C-order = row-major
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_rowmajor", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[A.ravel(), B.ravel(), C.ravel()])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    # result is row-major flat: reshape with C order
    mat = result.reshape(m, k)
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8)])
def test_gemm_ex(bins, m, n, k):
    """gemm_ex mixed layout: row-major A, col-major B, row-major C."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)       # row-major
    B_col = RNG.random((n, k)).astype(np.float32)   # will be passed col-major
    C = RNG.random((m, k)).astype(np.float32)       # row-major
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_ex", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[A.ravel(),
                            np.asfortranarray(B_col).ravel(order='F'),  # col-major flat
                            C.ravel()])
    expected = (alpha * A @ B_col + beta * C0).astype(np.float32)
    mat = result.reshape(m, k)
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_tiled ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6), (6, 10, 7), (4, 3, 4)])
def test_gemm_tiled(bins, m, n, k):
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_tiled", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_strided ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
@pytest.mark.parametrize("op,m,n,k,a_rs,b_rs", [
    ("gemm_strided_6x6x6_6_6", 6, 6, 6, 6, 6),
    ("gemm_strided_6x6x6_8_8", 6, 6, 6, 8, 8),
    ("gemm_strided_4x4x4_4_4", 4, 4, 4, 4, 4),
    ("gemm_strided_4x4x4_6_6", 4, 4, 4, 6, 6),
])
def test_gemm_strided(bins, op, m, n, k, a_rs, b_rs, alpha, beta):
    # A[i][j] = A_flat[i + j*a_rs]; B[j][l] = B_flat[j + l*b_rs]; C standard col-major.
    A_storage = np.zeros((a_rs, n), dtype=np.float32)
    A_storage[:m, :] = RNG.random((m, n)).astype(np.float32)
    B_storage = np.zeros((b_rs, k), dtype=np.float32)
    B_storage[:n, :] = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    A_flat = np.asfortranarray(A_storage).ravel(order='F')
    B_flat = np.asfortranarray(B_storage).ravel(order='F')
    C_flat = np.asfortranarray(C).ravel(order='F')
    # gemm_strided dispatch uses args=[alpha, beta] directly (no m/n/k positional args)
    result = run_op(bins["l3"], op, "simple",
                    args=[alpha, beta], inputs=[A_flat, B_flat, C_flat])
    expected = (alpha * A_storage[:m, :] @ B_storage[:n, :] + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── packed_gemm ──────────────────────────────────────────────────────────────

def _make_packed_vec(size, case):
    if case == "positive": return RNG.random(size).astype(np.float32)
    if case == "negative": return -RNG.random(size).astype(np.float32)
    if case == "mixed":    return (RNG.random(size) - 0.5).astype(np.float32)
    if case == "zero":     return np.zeros(size, dtype=np.float32)
    if case == "tiny":     return (RNG.random(size) * 1e-6).astype(np.float32)
    raise ValueError(case)


@pytest.mark.parametrize("case", ["positive", "negative", "mixed", "zero", "tiny"])
@pytest.mark.parametrize("k", [16, 32, 48, 64])
def test_packed_gemm(bins, k, case):
    # glass::gemm<float,4,4,K>: C(4×K) = alpha * A(4×4) * B(4×K) + beta * C(4×K), col-major
    m, n = 4, 4
    alpha, beta = 1.5, 0.3
    A = _make_packed_vec(m * n, case).reshape(m, n)
    B = _make_packed_vec(n * k, case).reshape(n, k)
    C = _make_packed_vec(m * k, case).reshape(m, k)
    C0 = C.copy()
    result = run_op(bins["l3"], f"packed_gemm_4x4x{k}", "simple",
                    args=[alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── inv ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_inv(bins, n, version):
    A = make_spd(n)
    # invertMatrix expects [A | I] layout (n*2n) column-major
    AI = np.hstack([A, np.eye(n, dtype=np.float32)])   # row layout (n x 2n)
    AI_col = np.asfortranarray(AI).ravel(order='F')
    result = run_op(bins["l3"], "inv", version, args=[n], inputs=[AI_col])
    Ainv = result.reshape(n, n, order='F')
    expected = np.linalg.inv(A).astype(np.float32)
    assert np.allclose(Ainv, expected, rtol=1e-2, atol=1e-3)


# ─── chol ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_chol(bins, n, version):
    A = make_spd(n)
    A_col = np.asfortranarray(A).ravel(order='F')
    result = run_op(bins["l3"], "chol", version, args=[n], inputs=[A_col])
    L_gpu = result.reshape(n, n, order='F')
    # Extract lower triangle
    L_gpu = np.tril(L_gpu)
    L_ref = np.linalg.cholesky(A).astype(np.float32)
    assert np.allclose(L_gpu, L_ref, rtol=1e-2, atol=1e-3)


# ─── trsm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6, 8])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_trsm(bins, n, version):
    L = make_lower_triangular(n)
    b = RNG.random(n).astype(np.float32)
    L_col = np.asfortranarray(L).ravel(order='F')
    result = run_op(bins["l3"], "trsm", version, args=[n], inputs=[L_col, b])
    # Verify L @ result == b (residual check avoids scipy dependency)
    residual = L.astype(np.float64) @ result.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(residual, 0, atol=1e-3)
