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
WARP_COUNTS = [1, 2, 3, 4, 8]  # odd + large; partial warps are FORBIDDEN by the warp contract


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


# ─── gemm (C = alpha*A@B, implicit beta=0) ────────────────────────────────────
# A, B column-major; C returned column-major (n*n per warp).
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_gemm(bins, n, W):
    alpha = 1.4
    As = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(W)]
    Bs = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(W)]
    Aflat = np.concatenate([np.asfortranarray(A).ravel(order="F") for A in As])
    Bflat = np.concatenate([np.asfortranarray(B).ravel(order="F") for B in Bs])
    result = run_op(bins["warp"], "gemm", str(n), args=[W, alpha], inputs=[Aflat, Bflat])
    for w in range(W):
        Cw = result[w*n*n:(w+1)*n*n].reshape((n, n), order="F")
        assert np.allclose(Cw, alpha * As[w] @ Bs[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} gemm mismatch (n={n}, W={W})"


# ─── potrs (SPD solve from a precomputed Cholesky factor) ─────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_potrs(bins, n, W):
    Ls, bs, oracles = [], [], []
    for _ in range(W):
        A = _spd(n)
        L = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
        b = RNG.standard_normal(n).astype(np.float32)
        Ls.append(L)
        bs.append(b)
        # Oracle against the SAME float32-rounded factor the device consumes:
        # potrs solves L Lt x = b, which differs from A x = b at factor precision.
        oracles.append(
            scipy.linalg.cho_solve((L.astype(np.float64), True),
                                   b.astype(np.float64)).astype(np.float32))
    Lflat = np.concatenate([np.asfortranarray(L).ravel(order="F") for L in Ls])
    bflat = np.concatenate(bs)
    result = run_op(bins["warp"], "potrs", str(n), args=[W], inputs=[Lflat, bflat])
    slices = _per_warp(result, W, n)
    for w in range(W):
        assert np.allclose(slices[w], oracles[w], rtol=RTOL, atol=ATOL), \
            f"warp {w} potrs mismatch (n={n}, W={W})"


# ─── transform_points (homogeneous 4x4 on packed xyz points) ──────────────────
# Two assertions per case: (1) BIT-identity vs the HJCD-IK placement expression
# the primitive replaces (compared on device — the driver prints a per-warp
# mismatch count on its first line); (2) tolerance vs a numpy oracle computed in
# the transform's precision. Transforms are random proper rigid motions.

XPTS_SIZES = [1, 5, 31, 32, 33, 64, 97]


def _rigid(rng):
    """Random SE(3) as a column-major 4x4 (flattened order='F')."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    X = np.eye(4)
    X[:3, :3] = q
    X[:3, 3] = rng.standard_normal(3) * 2.0
    return X.astype(np.float32)   # float32 file; the f64 path promotes exactly


def _xpts_oracle(X32, pts, f64):
    dt = np.float64 if f64 else np.float32
    X = X32.astype(dt)
    P = pts.reshape(-1, 3).astype(dt)
    return (P @ X[:3, :3].T + X[:3, 3]).astype(np.float32).ravel()


@pytest.mark.parametrize("n", XPTS_SIZES)
@pytest.mark.parametrize("W", [1, 2, 4])
@pytest.mark.parametrize("f64", [False, True], ids=["Xf32", "Xf64"])
def test_transform_points(bins, n, W, f64):
    Xs = [_rigid(RNG) for _ in range(W)]
    pts = [RNG.standard_normal(3 * n).astype(np.float32) for _ in range(W)]
    Xflat = np.concatenate([X.ravel(order="F") for X in Xs])
    lines = run_op(bins["warp"], "xpts64" if f64 else "xpts", str(n), args=[W],
                   inputs=[Xflat, np.concatenate(pts)])
    mism, out = lines
    assert np.all(mism == 0), f"bit-mismatch vs HJCD expression per warp: {mism}"
    slices = _per_warp(out, W, 3 * n)
    tol = 1e-6 if f64 else 1e-5
    for w in range(W):
        np.testing.assert_allclose(slices[w], _xpts_oracle(Xs[w], pts[w], f64),
                                   rtol=tol, atol=tol, err_msg=f"warp {w} (n={n})")


@pytest.mark.parametrize("n", [1, 7, 40, 97])
@pytest.mark.parametrize("W", [1, 3])
@pytest.mark.parametrize("f64", [False, True], ids=["Xf32", "Xf64"])
def test_transform_points_indexed(bins, n, W, f64):
    NX = 6   # transforms per warp (a short kinematic chain); every point picks one
    Xs = [[_rigid(RNG) for _ in range(NX)] for _ in range(W)]
    idx = [RNG.integers(0, NX, size=n).astype(np.float32) for _ in range(W)]
    pts = [RNG.standard_normal(3 * n).astype(np.float32) for _ in range(W)]
    Xflat = np.concatenate([X.ravel(order="F") for chain in Xs for X in chain])
    lines = run_op(bins["warp"], "xpts_idx64" if f64 else "xpts_idx", str(n),
                   args=[W, NX], inputs=[Xflat, np.concatenate(idx), np.concatenate(pts)])
    mism, out = lines
    assert np.all(mism == 0), f"bit-mismatch vs HJCD expression per warp: {mism}"
    slices = _per_warp(out, W, 3 * n)
    tol = 1e-6 if f64 else 1e-5
    for w in range(W):
        P = pts[w].reshape(-1, 3)
        want = np.concatenate([_xpts_oracle(Xs[w][int(k)], P[i], f64)
                               for i, k in enumerate(idx[w])])
        np.testing.assert_allclose(slices[w], want, rtol=tol, atol=tol,
                                   err_msg=f"warp {w} (n={n})")


# ─── warp::any / warp::all ────────────────────────────────────────────────────
# Lanes diverge (odd lanes take a data-dependent detour) and reconverge before
# voting. Truth table over all-false, all-true, single-lane and random masks,
# with several warps voting independently in one block.

def _vote_masks(rng, W):
    pool = [np.zeros(32), np.ones(32), np.eye(32)[0], np.eye(32)[31],
            (rng.random(32) < 0.5).astype(float), np.ones(32) - np.eye(32)[17]]
    return [pool[(w + i) % len(pool)] for i, w in enumerate(range(W))]


@pytest.mark.parametrize("W", [1, 2, 3, 6])
@pytest.mark.parametrize("seed", [0, 1])
def test_vote_any_all(bins, W, seed):
    rng = np.random.default_rng(seed)
    masks = _vote_masks(rng, W)
    rng.shuffle(masks)
    pred = np.concatenate(masks).astype(np.float32)
    out = np.asarray(run_op(bins["warp"], "vote", "0", args=[W], inputs=[pred]),
                     dtype=np.float32).ravel()
    for w in range(W):
        m = masks[w].astype(bool)
        assert out[2 * w] == float(m.any()), f"warp {w} any: mask={m.astype(int)}"
        assert out[2 * w + 1] == float(m.all()), f"warp {w} all: mask={m.astype(int)}"
