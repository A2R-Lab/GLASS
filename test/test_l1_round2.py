"""Round-2 L1 ops: nrm1_diff (‖x−y‖₁), warp norms (asum / nrm2), and the
row-strided AXPY / COPY block movers.

Thread discipline (see test_hardening_thread_sweep_2026-06-24.md):
  • block elementwise / serial-tail ops  → full THREAD_SWEEP, byte-identical.
  • block warp-shuffle reductions (nrm1_hs) → full sweep, oracle-close (the
    reduction order varies with blockDim, so not byte-identical across counts).
  • warp ops                              → exactly one 32-lane warp.
Varied inputs: normal (mixed sign), mixed-magnitude (stresses the 1-norm/L2
accumulation), and strictly-positive draws.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, make_vec

RTOL, ATOL = 1e-4, 1e-4
WARP = 32


def _write(arr):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    np.asarray(arr, dtype=np.float32).tofile(f)
    f.close()
    return f.name


def _run(bins, args):
    cmd = [str(bins["l1_round2"])] + [str(a) for a in args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"runner failed:\n{res.stderr}")
    return [np.fromstring(l, sep=" ").astype(np.float32)
            for l in res.stdout.strip().split("\n") if l.strip()]


# ─── nrm1_diff ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["normal", "mixed", "pos"])
@pytest.mark.parametrize("n", [7, 20, 65])
def test_nrm1_diff_lowmem(bins, n, kind):
    """low_memory ‖x−y‖₁: oracle-correct and byte-identical across the full sweep."""
    x, y = make_vec(n, seed=1, kind=kind), make_vec(n, seed=2, kind=kind)
    xf, yf = _write(x), _write(y)
    try:
        expected = np.abs(x - y).sum()
        ref = None
        for t in THREAD_SWEEP:
            out = _run(bins, ["nrm1_lm", t, n, xf, yf])[0][0]
            assert np.allclose(out, expected, rtol=RTOL, atol=ATOL), f"t={t}: {out} vs {expected}"
            if ref is None:
                ref = out
            else:
                assert out == ref, f"t={t}: non-invariant ({out} vs {ref})"
    finally:
        os.unlink(xf); os.unlink(yf)


@pytest.mark.parametrize("kind", ["normal", "mixed"])
@pytest.mark.parametrize("n", [7, 20, 65])
def test_nrm1_diff_highspeed(bins, n, kind):
    """high_speed ‖x−y‖₁: oracle-correct across the full sweep (warp-shuffle reduce
    order varies with blockDim → close, not byte-identical)."""
    x, y = make_vec(n, seed=3, kind=kind), make_vec(n, seed=4, kind=kind)
    xf, yf = _write(x), _write(y)
    try:
        expected = np.abs(x - y).sum()
        for t in THREAD_SWEEP:
            out = _run(bins, ["nrm1_hs", t, n, xf, yf])[0][0]
            assert np.allclose(out, expected, rtol=1e-3, atol=1e-3), f"t={t}: {out} vs {expected}"
    finally:
        os.unlink(xf); os.unlink(yf)


@pytest.mark.parametrize("kind", ["normal", "mixed", "pos"])
@pytest.mark.parametrize("n", [7, 33, 65])
def test_nrm1_diff_warp(bins, n, kind):
    """warp ‖x−y‖₁ at one warp, matching the oracle."""
    x, y = make_vec(n, seed=5, kind=kind), make_vec(n, seed=6, kind=kind)
    xf, yf = _write(x), _write(y)
    try:
        out = _run(bins, ["nrm1_warp", WARP, n, xf, yf])[0][0]
        assert np.allclose(out, np.abs(x - y).sum(), rtol=1e-3, atol=1e-3)
    finally:
        os.unlink(xf); os.unlink(yf)


# ─── warp norms ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["normal", "mixed", "pos"])
@pytest.mark.parametrize("n", [7, 33, 65])
def test_warp_asum(bins, n, kind):
    x = make_vec(n, seed=7, kind=kind)
    xf = _write(x)
    try:
        out = _run(bins, ["warp_asum", WARP, n, xf])[0][0]
        assert np.allclose(out, np.abs(x).sum(), rtol=1e-3, atol=1e-3)
    finally:
        os.unlink(xf)


@pytest.mark.parametrize("kind", ["normal", "mixed", "pos"])
@pytest.mark.parametrize("n", [7, 33, 65])
def test_warp_nrm2(bins, n, kind):
    x = make_vec(n, seed=8, kind=kind)
    xf = _write(x)
    try:
        out = _run(bins, ["warp_nrm2", WARP, n, xf])[0][0]
        assert np.allclose(out, np.linalg.norm(x), rtol=1e-3, atol=1e-3)
    finally:
        os.unlink(xf)


# ─── row-strided AXPY / COPY ──────────────────────────────────────────────────
# Shapes match the harness table: 0=(14,14,21,14) PDDP, 1=(6,5,8,6), 2=(4,4,4,4).
SHAPES = {0: (14, 14, 21, 14), 1: (6, 5, 8, 6), 2: (4, 4, 4, 4)}


def _rs_io(shape_id, seed):
    M, N, YRS, XRS = SHAPES[shape_id]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((XRS, N)).astype(np.float32)
    Y = rng.standard_normal((YRS, N)).astype(np.float32)
    return M, N, YRS, XRS, X, Y


@pytest.mark.parametrize("shape_id", [0, 1, 2])
@pytest.mark.parametrize("op,copy", [("rsaxpy", False), ("rscopy", True)])
def test_row_strided_block(bins, op, copy, shape_id):
    """Block row-strided AXPY/COPY: oracle-correct, byte-identical across the full
    sweep (each (r,c) written once → no reduction, exact invariance)."""
    M, N, YRS, XRS, X, Y = _rs_io(shape_id, seed=10 + shape_id)
    alpha = 0.5
    expected = Y.copy()
    if copy:
        expected[:M, :N] = alpha * X[:M, :N]
    else:
        expected[:M, :N] += alpha * X[:M, :N]
    Xf = _write(np.asfortranarray(X).ravel(order="F"))
    try:
        ref = None
        for t in THREAD_SWEEP:
            Yf = _write(np.asfortranarray(Y).ravel(order="F"))
            try:
                flat = _run(bins, [op, t, shape_id, alpha, Xf, Yf])[0]
                got = flat.reshape((YRS, N), order="F")
                assert np.allclose(got, expected, rtol=1e-5, atol=1e-5), f"{op} shape={shape_id} t={t}"
                if ref is None:
                    ref = flat
                else:
                    assert np.array_equal(flat, ref), f"{op} t={t}: non-invariant"
            finally:
                os.unlink(Yf)
    finally:
        os.unlink(Xf)


@pytest.mark.parametrize("shape_id", [0, 1, 2])
@pytest.mark.parametrize("op,copy", [("rsaxpy_warp", False), ("rscopy_warp", True)])
def test_row_strided_warp(bins, op, copy, shape_id):
    """Warp row-strided AXPY/COPY at one warp, matching the block/oracle."""
    M, N, YRS, XRS, X, Y = _rs_io(shape_id, seed=20 + shape_id)
    alpha = -1.5
    expected = Y.copy()
    if copy:
        expected[:M, :N] = alpha * X[:M, :N]
    else:
        expected[:M, :N] += alpha * X[:M, :N]
    Xf = _write(np.asfortranarray(X).ravel(order="F"))
    Yf = _write(np.asfortranarray(Y).ravel(order="F"))
    try:
        flat = _run(bins, [op, WARP, shape_id, alpha, Xf, Yf])[0]
        got = flat.reshape((YRS, N), order="F")
        assert np.allclose(got, expected, rtol=1e-5, atol=1e-5), f"{op} shape={shape_id}"
    finally:
        os.unlink(Xf); os.unlink(Yf)
