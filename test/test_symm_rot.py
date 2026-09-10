"""glass::symm / glass::trmm (L3) and glass::rot / glass::rotg (L1).

symm: C = alpha*A_sym*B + beta*C with only the FILL triangle of A stored — the
unstored triangle of the input buffer is NaN-POISONED so any read of it (a
mirror-logic bug) poisons the output. trmm: C = alpha*op(A_tri)*B out-of-place,
all {FILL, DIAG, TRANSPOSE} combos, with the unstored triangle (and, under
Diag::Unit, the diagonal) poisoned the same way. rot: the BLAS Givens apply
(block + warp). rotg: the __host__ __device__ Givens generator (stable scaled
form), device vs host vs reference.

Both matrix ops use gemm's flat one-output-per-thread loop (serial ascending-k
chain per element), so outputs must be BYTE-identical across the full thread
sweep; rot is elementwise and likewise byte-identical.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, make_general, make_vec

WARP = 32

RTOL = 1e-4
ATOL = 1e-5

SYMM_SHAPES = [(4, 1), (4, 5), (7, 1), (7, 5), (16, 5)]
TRMM_SHAPES = [(4, 5), (7, 3)]
AB_GRID = [(1.0, 0.0), (0.5, 2.0), (-1.5, 0.3)]


# ─── harness ───────────────────────────────────────────────────────────────────

def _write(arr):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    np.asarray(arr, dtype=np.float32).ravel(order="F").tofile(f)
    f.close()
    return f.name


def _invoke(bins, args, files):
    fns = [_write(a) for a in files]
    try:
        cmd = [str(bins["symm_rot"])] + [str(a) for a in args] + fns
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
        return [np.fromstring(l, sep=" ").astype(np.float32) for l in lines]
    finally:
        for fn in fns:
            os.unlink(fn)


def _run_symm(bins, threads, fill, ct, hasbeta, n, m, alpha, beta, A, B, C):
    out = _invoke(bins, ["symm", threads, fill, int(ct), int(hasbeta), n, m, alpha, beta],
                  [A, B, C])
    return out[0].reshape(n, m, order="F")


def _run_trmm(bins, threads, fill, diag, trans, ct, n, m, alpha, A, B):
    out = _invoke(bins, ["trmm", threads, fill, diag, int(trans), int(ct), n, m, alpha],
                  [A, B])
    return out[0].reshape(n, m, order="F")


def _run_rot(bins, surface, threads, ct, n, c, s, x, y):
    out = _invoke(bins, ["rot", surface, threads, int(ct), n, c, s], [x, y])
    return out[0], out[1]


def _run_rotg(bins, a, b):
    out = _invoke(bins, ["rotg", a, b], [])
    return out[0], out[1]   # device (c, s, r), host (c, s, r)


# ─── references + poisoned inputs ──────────────────────────────────────────────

def _sym_inputs(n, fill, seed=0, scale=1.0):
    """(poisoned stored-triangle buffer, full symmetric reference matrix)."""
    A = make_general(n, seed=seed, scale=scale)
    if fill == "l":
        S = np.tril(A) + np.tril(A, -1).T
        buf = np.where(np.tril(np.ones((n, n), bool)), S, np.float32(np.nan))
    else:
        S = np.triu(A) + np.triu(A, 1).T
        buf = np.where(np.triu(np.ones((n, n), bool)), S, np.float32(np.nan))
    return buf.astype(np.float32), S.astype(np.float32)


def _tri_inputs(n, fill, diag, seed=0, scale=1.0):
    """(poisoned triangular buffer, full effective triangular reference)."""
    A = make_general(n, seed=seed, scale=scale)
    tri = np.tril(A) if fill == "l" else np.triu(A)
    mask = np.tril(np.ones((n, n), bool)) if fill == "l" else np.triu(np.ones((n, n), bool))
    buf = np.where(mask, tri, np.float32(np.nan))
    if diag == "u":
        np.fill_diagonal(buf, np.float32(np.nan))   # Unit diag: never read
        np.fill_diagonal(tri, np.float32(1.0))
    return buf.astype(np.float32), tri.astype(np.float32)


def _symm_ref(S, B, alpha, beta, C0):
    return (np.float32(alpha) * (S @ B) + np.float32(beta) * C0).astype(np.float32)


def _trmm_ref(T, B, alpha, trans):
    op = T.T if trans else T
    return (np.float32(alpha) * (op @ B)).astype(np.float32)


def _rotg_ref(a, b):
    a, b = np.float64(a), np.float64(b)
    if a == 0.0 and b == 0.0:
        return 1.0, 0.0, 0.0
    roe = a if abs(a) > abs(b) else b
    r = np.sign(roe) * np.hypot(a, b)
    return a / r, b / r, r


# ─── symm ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fill", ["l", "u"])
@pytest.mark.parametrize("n,m", SYMM_SHAPES)
@pytest.mark.parametrize("alpha,beta", AB_GRID)
def test_symm_vs_numpy(bins, fill, n, m, alpha, beta):
    buf, S = _sym_inputs(n, fill, seed=10 * n + m)
    B = make_general(n, m, seed=3 * n + m)
    C0 = make_general(n, m, seed=5 * n + m)
    out = _run_symm(bins, 64, fill, 0, 1, n, m, alpha, beta, buf, B, C0)
    ref = _symm_ref(S, B, alpha, beta, C0)
    assert np.isfinite(out).all(), f"NaN leak: unstored {fill} triangle was read (n={n},m={m})"
    assert np.allclose(out, ref, rtol=RTOL, atol=ATOL), \
        f"symm mismatch fill={fill} n={n} m={m} a={alpha} b={beta}"


@pytest.mark.parametrize("fill", ["l", "u"])
def test_symm_overwrite_never_reads_c(bins, fill):
    """The implicit-beta=0 overload: C input is ALL NaN and must not poison."""
    n, m = 7, 5
    buf, S = _sym_inputs(n, fill, seed=71)
    B = make_general(n, m, seed=72)
    C0 = np.full((n, m), np.nan, dtype=np.float32)
    out = _run_symm(bins, 64, fill, 0, 0, n, m, 0.5, 0.0, buf, B, C0)
    ref = _symm_ref(S, B, 0.5, 0.0, np.zeros((n, m), np.float32))
    assert np.isfinite(out).all(), "overwrite overload read C"
    assert np.allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_symm_thread_invariance(bins):
    """(7,5) over the FULL thread sweep, byte-identical, 2 input kinds."""
    n, m = 7, 5
    for seed, scale in [(11, 1.0), (12, 1e3)]:
        buf, S = _sym_inputs(n, "l", seed=seed, scale=scale)
        B = make_general(n, m, seed=seed + 100, scale=scale)
        C0 = make_general(n, m, seed=seed + 200)
        ref = _symm_ref(S, B, 0.5, 2.0, C0)
        outs = [_run_symm(bins, t, "l", 0, 1, n, m, 0.5, 2.0, buf, B, C0) for t in THREAD_SWEEP]
        assert np.allclose(outs[0], ref, rtol=RTOL, atol=ATOL * scale), f"seed={seed}"
        for t, r in zip(THREAD_SWEEP[1:], outs[1:]):
            assert np.array_equal(outs[0], r), f"symm non-invariant at {t} threads (seed={seed})"


@pytest.mark.parametrize("fill", ["l", "u"])
@pytest.mark.parametrize("n,m", SYMM_SHAPES)
def test_symm_ct_matches_runtime(bins, fill, n, m):
    buf, S = _sym_inputs(n, fill, seed=21)
    B = make_general(n, m, seed=22)
    C0 = make_general(n, m, seed=23)
    for t in (1, 33, 256):
        ct = _run_symm(bins, t, fill, 1, 1, n, m, -1.5, 0.3, buf, B, C0)
        rt = _run_symm(bins, t, fill, 0, 1, n, m, -1.5, 0.3, buf, B, C0)
        assert np.array_equal(ct, rt), f"symm CT != runtime (fill={fill} n={n} m={m} t={t})"
        assert np.allclose(ct, _symm_ref(S, B, -1.5, 0.3, C0), rtol=RTOL, atol=ATOL)


# ─── trmm ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fill", ["l", "u"])
@pytest.mark.parametrize("diag", ["n", "u"])
@pytest.mark.parametrize("trans", [0, 1])
@pytest.mark.parametrize("n,m", TRMM_SHAPES)
@pytest.mark.parametrize("alpha", [1.0, -0.5])
def test_trmm_vs_numpy(bins, fill, diag, trans, n, m, alpha):
    buf, T = _tri_inputs(n, fill, diag, seed=10 * n + m)
    B = make_general(n, m, seed=4 * n + m)
    out = _run_trmm(bins, 64, fill, diag, trans, 0, n, m, alpha, buf, B)
    ref = _trmm_ref(T, B, alpha, trans)
    assert np.isfinite(out).all(), \
        f"NaN leak: unstored triangle/diag read (fill={fill} diag={diag} trans={trans})"
    assert np.allclose(out, ref, rtol=RTOL, atol=ATOL), \
        f"trmm mismatch fill={fill} diag={diag} trans={trans} n={n} m={m} a={alpha}"


def test_trmm_thread_invariance(bins):
    """(7,3) over the FULL thread sweep, byte-identical, 2 input kinds."""
    n, m = 7, 3
    for seed, scale in [(31, 1.0), (32, 1e3)]:
        buf, T = _tri_inputs(n, "l", "n", seed=seed, scale=scale)
        B = make_general(n, m, seed=seed + 100, scale=scale)
        ref = _trmm_ref(T, B, 0.5, False)
        outs = [_run_trmm(bins, t, "l", "n", 0, 0, n, m, 0.5, buf, B) for t in THREAD_SWEEP]
        assert np.allclose(outs[0], ref, rtol=RTOL, atol=ATOL * scale * scale), f"seed={seed}"
        for t, r in zip(THREAD_SWEEP[1:], outs[1:]):
            assert np.array_equal(outs[0], r), f"trmm non-invariant at {t} threads (seed={seed})"


@pytest.mark.parametrize("fill", ["l", "u"])
@pytest.mark.parametrize("diag", ["n", "u"])
@pytest.mark.parametrize("trans", [0, 1])
def test_trmm_ct_matches_runtime(bins, fill, diag, trans):
    n, m = 4, 5
    buf, T = _tri_inputs(n, fill, diag, seed=41)
    B = make_general(n, m, seed=42)
    for t in (1, 33, 256):
        ct = _run_trmm(bins, t, fill, diag, trans, 1, n, m, 1.0, buf, B)
        rt = _run_trmm(bins, t, fill, diag, trans, 0, n, m, 1.0, buf, B)
        assert np.array_equal(ct, rt), f"trmm CT != runtime ({fill},{diag},{trans},t={t})"
        assert np.allclose(ct, _trmm_ref(T, B, 1.0, trans), rtol=RTOL, atol=ATOL)


# ─── rot ───────────────────────────────────────────────────────────────────────

def _rot_ref(x, y, c, s):
    c, s = np.float32(c), np.float32(s)
    return (c * x + s * y).astype(np.float32), (c * y - s * x).astype(np.float32)


def test_rot_block_sweep(bins):
    """n=33 (crosses a warp) over the FULL thread sweep, 2 input kinds."""
    n = 33
    c, s = float(np.cos(0.3)), float(np.sin(0.3))
    for seed, kind in [(51, "normal"), (52, "mixed")]:
        x = make_vec(n, seed=seed, kind=kind)
        y = make_vec(n, seed=seed + 100, kind=kind)
        rx, ry = _rot_ref(x, y, c, s)
        outs = [_run_rot(bins, "block", t, 0, n, c, s, x, y) for t in THREAD_SWEEP]
        assert np.allclose(outs[0][0], rx, rtol=1e-5, atol=1e-6), f"x mismatch kind={kind}"
        assert np.allclose(outs[0][1], ry, rtol=1e-5, atol=1e-6), f"y mismatch kind={kind}"
        for t, (ox, oy) in zip(THREAD_SWEEP[1:], outs[1:]):
            assert np.array_equal(outs[0][0], ox) and np.array_equal(outs[0][1], oy), \
                f"rot non-invariant at {t} threads (kind={kind})"


@pytest.mark.parametrize("n", [5, 33])
@pytest.mark.parametrize("ct", [0, 1])
def test_rot_warp(bins, n, ct):
    """warp::rot at one 32-lane warp == block rot at 32 threads, byte-identical."""
    c, s = float(np.cos(1.1)), float(np.sin(1.1))
    x = make_vec(n, seed=61)
    y = make_vec(n, seed=62)
    wx, wy = _run_rot(bins, "warp", WARP, ct, n, c, s, x, y)
    bx, by = _run_rot(bins, "block", WARP, ct, n, c, s, x, y)
    rx, ry = _rot_ref(x, y, c, s)
    assert np.allclose(wx, rx, rtol=1e-5, atol=1e-6) and np.allclose(wy, ry, rtol=1e-5, atol=1e-6)
    assert np.array_equal(wx, bx) and np.array_equal(wy, by), f"warp != block (n={n}, ct={ct})"


@pytest.mark.parametrize("n", [5, 33])
def test_rot_ct_matches_runtime(bins, n):
    c, s = float(np.cos(0.7)), float(np.sin(0.7))
    x = make_vec(n, seed=63)
    y = make_vec(n, seed=64)
    for t in (1, 33, 256):
        ct_out = _run_rot(bins, "block", t, 1, n, c, s, x, y)
        rt_out = _run_rot(bins, "block", t, 0, n, c, s, x, y)
        assert np.array_equal(ct_out[0], rt_out[0]) and np.array_equal(ct_out[1], rt_out[1]), \
            f"rot CT != runtime (n={n}, t={t})"


# ─── rotg ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    (3.0, 4.0), (-3.0, 4.0), (3.0, -4.0), (4.0, 3.0),
    (5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (0.0, 0.0),
    (3e30, 4e30),      # naive a*a overflows float32; the scaled form must not
    (1e-30, 1e-30),    # naive a*a underflows to 0
])
def test_rotg(bins, a, b):
    dev, host = _run_rotg(bins, a, b)
    assert np.isfinite(dev).all() and np.isfinite(host).all(), f"rotg overflow a={a} b={b}"
    # device and host paths agree (same scalar helper compiled twice)
    assert np.allclose(dev, host, rtol=1e-6, atol=0), f"device != host: {dev} vs {host}"
    c, s, r = (float(v) for v in dev)
    rc, rs, rr = _rotg_ref(a, b)
    assert np.allclose([c, s, r], [rc, rs, rr], rtol=1e-5, atol=1e-30), \
        f"rotg vs reference: {(c, s, r)} vs {(rc, rs, rr)}"
    # defining identities: G @ [a, b] = [r, 0], and G is a rotation
    fa, fb = np.float64(a), np.float64(b)
    assert np.isclose(c * fa + s * fb, r, rtol=1e-5, atol=1e-30)
    assert np.isclose(c * fb - s * fa, 0.0, atol=max(1e-30, 1e-5 * abs(r)))
    assert np.isclose(c * c + s * s, 1.0, rtol=1e-5) or (a == 0.0 and b == 0.0 and (c, s) == (1.0, 0.0))
