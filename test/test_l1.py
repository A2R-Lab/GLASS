"""L1 GLASS function tests — compare GPU results to NumPy reference."""

import numpy as np
import pytest
from conftest import run_op

RNG = np.random.default_rng(42)

ATOL = 1e-5
RTOL = 1e-4

SIZES = [8, 64, 256]

CG_SIMPLE    = ["cg", "simple"]
CG_LM_HS     = ["cg", "simple_lm", "simple_hs"]


# ─── axpy ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_axpy(bins, n, version):
    alpha = 1.5
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(n).astype(np.float32)
    y0 = y.copy()
    result = run_op(bins["l1"], "axpy", version, args=[n, alpha], inputs=[x, y])
    assert np.allclose(result, alpha * x + y0, rtol=RTOL, atol=ATOL)


# ─── axpby ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_axpby(bins, n, version):
    alpha, beta = 1.5, 0.3
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "axpby", version, args=[n, alpha, beta], inputs=[x, y])
    assert np.allclose(result, alpha * x + beta * y, rtol=RTOL, atol=ATOL)


# ─── copy ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_copy(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "copy", version, args=[n], inputs=[x])
    assert np.allclose(result, x, rtol=RTOL, atol=ATOL)


# ─── scal ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_scal(bins, n, version):
    alpha = 2.5
    x = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "scal", version, args=[n, alpha], inputs=[x])
    assert np.allclose(result, alpha * x, rtol=RTOL, atol=ATOL)


# ─── swap ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_swap(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(n).astype(np.float32)
    x0, y0 = x.copy(), y.copy()
    result = run_op(bins["l1"], "swap", version, args=[n], inputs=[x, y])
    assert np.allclose(result[0], y0, rtol=RTOL, atol=ATOL)
    assert np.allclose(result[1], x0, rtol=RTOL, atol=ATOL)


# ─── dot ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_dot(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    y = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "dot", version, args=[n], inputs=[x, y])
    expected = np.dot(x.astype(np.float64), y.astype(np.float64))
    assert np.allclose(float(result[0]), expected, rtol=1e-3, atol=1e-4)


# ─── reduce ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_reduce(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "reduce", version, args=[n], inputs=[x])
    expected = float(np.sum(x.astype(np.float64)))
    assert np.allclose(float(result[0]), expected, rtol=1e-3, atol=1e-4)


# ─── l2norm ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_l2norm(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "l2norm", version, args=[n], inputs=[x])
    expected = float(np.linalg.norm(x.astype(np.float64)))
    assert np.allclose(float(result[0]), expected, rtol=1e-3, atol=1e-4)


# ─── infnorm ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_infnorm(bins, n, version):
    x = RNG.random(n).astype(np.float32) - 0.5   # allow negatives
    result = run_op(bins["l1"], "infnorm", version, args=[n], inputs=[x])
    expected = float(np.max(np.abs(x)))
    assert np.allclose(float(result[0]), expected, rtol=RTOL, atol=ATOL)


# ─── asum ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_asum(bins, n, version):
    x = RNG.random(n).astype(np.float32) - 0.5
    result = run_op(bins["l1"], "asum", version, args=[n], inputs=[x])
    expected = float(np.sum(np.abs(x).astype(np.float64)))
    assert np.allclose(float(result[0]), expected, rtol=1e-3, atol=1e-4)


# ─── clip ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_clip(bins, n, version):
    x = RNG.random(n).astype(np.float32) * 2 - 1   # [-1, 1]
    lo = np.full(n, -0.5, dtype=np.float32)
    hi = np.full(n,  0.5, dtype=np.float32)
    result = run_op(bins["l1"], "clip", version, args=[n], inputs=[x, lo, hi])
    assert np.allclose(result, np.clip(x, lo, hi), rtol=RTOL, atol=ATOL)


# ─── set_const ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_set_const(bins, n, version):
    alpha = 3.14
    result = run_op(bins["l1"], "set_const", version, args=[n, alpha], inputs=[])
    assert np.allclose(result, np.full(n, alpha, dtype=np.float32), rtol=RTOL, atol=ATOL)


# ─── loadIdentity ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_loadIdentity(bins, n, version):
    result = run_op(bins["l1"], "loadIdentity", version, args=[n], inputs=[])
    # column-major identity: reshape as (n, n) Fortran order
    mat = result.reshape(n, n, order='F')
    assert np.allclose(mat, np.eye(n, dtype=np.float32), rtol=RTOL, atol=ATOL)


# ─── addI ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_addI(bins, n, version):
    alpha = 0.5
    A = RNG.random((n, n)).astype(np.float32)
    A_col = np.asfortranarray(A)
    result = run_op(bins["l1"], "addI", version, args=[n, alpha], inputs=[A_col.ravel(order='F')])
    expected = A + alpha * np.eye(n, dtype=np.float32)
    mat = result.reshape(n, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── transpose ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N,M", [(4, 6), (8, 8), (12, 4)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_transpose(bins, N, M, version):
    A = RNG.random((N, M)).astype(np.float32)
    A_col = np.asfortranarray(A)  # column-major
    result = run_op(bins["l1"], "transpose", version, args=[N, M], inputs=[A_col.ravel(order='F')])
    # Result is MxN in column-major order
    mat = result.reshape(M, N, order='F')
    assert np.allclose(mat, A.T, rtol=RTOL, atol=ATOL)


# ─── elementwise ops ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_add(bins, n, version):
    a = RNG.random(n).astype(np.float32)
    b = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "elementwise_add", version, args=[n], inputs=[a, b])
    assert np.allclose(result, a + b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_sub(bins, n, version):
    a = RNG.random(n).astype(np.float32)
    b = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "elementwise_sub", version, args=[n], inputs=[a, b])
    assert np.allclose(result, a - b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_mult(bins, n, version):
    a = RNG.random(n).astype(np.float32)
    b = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "elementwise_mult", version, args=[n], inputs=[a, b])
    assert np.allclose(result, a * b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_abs(bins, n, version):
    a = RNG.random(n).astype(np.float32) - 0.5
    result = run_op(bins["l1"], "elementwise_abs", version, args=[n], inputs=[a])
    assert np.allclose(result, np.abs(a), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_max(bins, n, version):
    a = RNG.random(n).astype(np.float32)
    b = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "elementwise_max", version, args=[n], inputs=[a, b])
    assert np.allclose(result, np.maximum(a, b), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_min(bins, n, version):
    a = RNG.random(n).astype(np.float32)
    b = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "elementwise_min", version, args=[n], inputs=[a, b])
    assert np.allclose(result, np.minimum(a, b), rtol=RTOL, atol=ATOL)


# ─── dot_strided ──────────────────────────────────────────────────────────────

DOT_STRIDED_SHAPES = [
    (4, 4, 1),   # x_size=16, y_size=4
    (6, 1, 1),   # x_size=6,  y_size=6
    (6, 6, 1),   # x_size=36, y_size=6
    (6, 6, 6),   # x_size=36, y_size=36
]


def _make_test_vec(size, case):
    if case == "positive": return RNG.random(size).astype(np.float32)
    if case == "negative": return -RNG.random(size).astype(np.float32)
    if case == "mixed":    return (RNG.random(size) - 0.5).astype(np.float32)
    if case == "zero":     return np.zeros(size, dtype=np.float32)
    if case == "tiny":     return (RNG.random(size) * 1e-6).astype(np.float32)
    raise ValueError(case)


@pytest.mark.parametrize("n,sx,sy", DOT_STRIDED_SHAPES)
@pytest.mark.parametrize("case", ["positive", "negative", "mixed", "zero", "tiny"])
def test_dot_strided(bins, n, sx, sy, case):
    x = _make_test_vec(n * sx, case)
    y = _make_test_vec(n * sy, case)
    # argv[3] is the required <n> positional arg in test_l1; unused by dot_strided dispatch
    result = run_op(bins["l1"], f"dot_strided_{n}_{sx}_{sy}", "simple",
                    args=[0], inputs=[x, y])
    expected = sum(float(x[i * sx]) * float(y[i * sy]) for i in range(n))
    assert np.isclose(float(result[0]), expected, rtol=RTOL, atol=ATOL)


# ─── dot_strided_coalesced ──────────────────────────────────────────────────
# Block-cooperative sibling of dot_strided: same value, coalesced global loads.
# "simple" version dispatches the per-thread dot_strided reference (thread 0
# writes); "simple_hs" dispatches the coalesced block-reduction primitive.
# Equivalence test asserts both agree (and match numpy) for a large stride.

DOT_COALESCED_SHAPES = [
    (64, 64, 64),   # x,y = 4096 each; stride 64 (column of a 64-wide row-major mat)
    (256, 256, 1),  # x = 65536, y = 256; large x stride, unit y stride
]


@pytest.mark.parametrize("n,sx,sy", DOT_COALESCED_SHAPES)
def test_dot_strided_coalesced(bins, n, sx, sy):
    x = (RNG.random(n * sx) - 0.5).astype(np.float32)
    y = (RNG.random(n * sy) - 0.5).astype(np.float32)
    op = f"dot_coalesced_{n}_{sx}_{sy}"
    coalesced = run_op(bins["l1"], op, "simple_hs", args=[0], inputs=[x, y])
    reference = run_op(bins["l1"], op, "simple", args=[0], inputs=[x, y])
    expected = sum(float(x[i * sx]) * float(y[i * sy]) for i in range(n))
    assert np.isclose(float(coalesced[0]), float(reference[0]), rtol=1e-4, atol=1e-4)
    assert np.isclose(float(coalesced[0]), expected, rtol=1e-3, atol=1e-4)


# ─── prefix sum ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [8, 32, 64])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_prefix_sum_excl(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "prefix_sum_excl", version, args=[n], inputs=[x])
    expected = np.concatenate([[0], np.cumsum(x)[:-1]]).astype(np.float32)
    assert np.allclose(result, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("n", [8, 32, 64])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_prefix_sum_incl(bins, n, version):
    x = RNG.random(n).astype(np.float32)
    result = run_op(bins["l1"], "prefix_sum_incl", version, args=[n], inputs=[x])
    expected = np.cumsum(x).astype(np.float32)
    assert np.allclose(result, expected, rtol=1e-3, atol=1e-4)
