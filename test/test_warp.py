"""glass::warp:: surface tests — MULTI-warp.

Each warp (threadIdx.y) owns a DISTINCT problem packed contiguously; we assert
EVERY warp's output slice independently against a per-warp numpy/scipy oracle.
Single-warp tests can't catch cross-warp bugs (a stray __syncthreads, a shared
re-read leaking across warps, a lane-mask leak), so we always run WARPS>=2 and
also WARPS=1 as a single==multi identity check.

float32, rtol=atol=1e-3. Sizes include non-multiples of 32.
"""

import numpy as np
import pytest
import scipy.linalg
from conftest import run_op

RNG = np.random.default_rng(7)

RTOL = 1e-3
ATOL = 1e-3

SIZES = [5, 7, 16, 33, 40, 64]
WARP_COUNTS = [1, 2, 4]


def _per_warp(arr, W, n):
    """Split a flat length-(W*n) result into a list of W length-n slices."""
    a = np.asarray(arr, dtype=np.float32).ravel()
    return [a[w * n:(w + 1) * n] for w in range(W)]


def _spd(n):
    """A well-conditioned SPD matrix (column-major-friendly: symmetric)."""
    M = RNG.standard_normal((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


# ─── dot ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_dot(bins, n, W):
    xs = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    ys = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # run_op builds argv = [bin, op, version, *args, *files]; the driver reads
    # argv[2]=n, argv[3]=W, so version carries n and args starts with W.
    # result is one scalar per warp (length W)
    result = run_op(bins["warp"], "dot", str(n), args=[W], inputs=[x, y])
    result = np.asarray(result, dtype=np.float32).ravel()
    for w in range(W):
        assert np.allclose(result[w], np.dot(xs[w], ys[w]), rtol=RTOL, atol=ATOL), \
            f"warp {w} dot mismatch (n={n}, W={W})"


# ─── axpy ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_axpy(bins, n, W):
    alpha = 1.7
    xs = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    ys = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    result = run_op(bins["warp"], "axpy", str(n), args=[W, alpha], inputs=[x, y])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], alpha * xs[w] + ys[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} axpy mismatch (n={n}, W={W})"


# ─── copy ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_copy(bins, n, W):
    xs = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    x = np.concatenate(xs)
    result = run_op(bins["warp"], "copy", str(n), args=[W], inputs=[x])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], xs[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} copy mismatch (n={n}, W={W})"


# ─── scal ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_scal(bins, n, W):
    alpha = 2.3
    xs = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    x = np.concatenate(xs)
    result = run_op(bins["warp"], "scal", str(n), args=[W, alpha], inputs=[x])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], alpha * xs[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} scal mismatch (n={n}, W={W})"


# ─── gemv (y = alpha*A@x, implicit beta=0) ────────────────────────────────────
# A is column-major: feed np.asfortranarray(A).ravel(order='F').

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_gemv(bins, n, W):
    alpha = 1.4
    As = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(W)]
    xs = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    Aflat = np.concatenate([np.asfortranarray(A).ravel(order="F") for A in As])
    x = np.concatenate(xs)
    result = run_op(bins["warp"], "gemv", str(n), args=[W, alpha], inputs=[Aflat, x])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], alpha * As[w] @ xs[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} gemv mismatch (n={n}, W={W})"


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_gemv_t(bins, n, W):
    alpha = 1.4
    As = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(W)]
    xs = [RNG.standard_normal(n).astype(np.float32) for _ in range(W)]
    Aflat = np.concatenate([np.asfortranarray(A).ravel(order="F") for A in As])
    x = np.concatenate(xs)
    result = run_op(bins["warp"], "gemv_t", str(n), args=[W, alpha], inputs=[Aflat, x])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], alpha * As[w].T @ xs[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} gemv_t mismatch (n={n}, W={W})"


# ─── trsv (all {lower,unit,trans} combos) ─────────────────────────────────────
# A column-major; oracle scipy.linalg.solve_triangular per flag combo.

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("unit", [False, True])
@pytest.mark.parametrize("trans", [False, True])
def test_trsv(bins, n, W, lower, unit, trans):
    As, bs, oracles = [], [], []
    for _ in range(W):
        # build a well-conditioned triangular matrix of the requested triangle
        M = RNG.standard_normal((n, n)).astype(np.float32)
        if lower:
            T = np.tril(M)
        else:
            T = np.triu(M)
        # strong diagonal for conditioning; unit case overwrites it with 1s
        np.fill_diagonal(T, np.abs(np.diag(T)) + n)
        if unit:
            np.fill_diagonal(T, 1.0)
        b = RNG.standard_normal(n).astype(np.float32)
        x = scipy.linalg.solve_triangular(
            T, b, lower=lower, trans=(1 if trans else 0), unit_diagonal=unit)
        As.append(T)
        bs.append(b)
        oracles.append(x.astype(np.float32))
    Aflat = np.concatenate([np.asfortranarray(A).ravel(order="F") for A in As])
    bflat = np.concatenate(bs)
    result = run_op(bins["warp"], "trsv", str(n),
                    args=[W, int(lower), int(unit), int(trans)],
                    inputs=[Aflat, bflat])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], oracles[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} trsv mismatch (n={n}, W={W}, lower={lower}, unit={unit}, trans={trans})"


# ─── posv (SPD solve A x = b) ─────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_posv(bins, n, W):
    As, bs, oracles = [], [], []
    for _ in range(W):
        A = _spd(n)
        b = RNG.standard_normal(n).astype(np.float32)
        As.append(A)
        bs.append(b)
        oracles.append(np.linalg.solve(A, b).astype(np.float32))
    # A symmetric => column-major == row-major; ravel order is irrelevant but be explicit.
    Aflat = np.concatenate([np.asfortranarray(A).ravel(order="F") for A in As])
    bflat = np.concatenate(bs)
    result = run_op(bins["warp"], "posv", str(n), args=[W], inputs=[Aflat, bflat])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], oracles[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} posv mismatch (n={n}, W={W})"
