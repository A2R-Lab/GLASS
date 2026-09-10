"""glass::symmetrize — in-place A := 0.5*(A + Aᵀ) (block / warp / cgrps).

The Schur-cleanup primitive MPC-SOLVER hand-rolls after its gemm sequences.
Each strictly-lower mirror pair is owned by exactly one thread (it reads and
writes BOTH mirror elements), the diagonal is untouched — so the op is
byte-identical across the full thread sweep. Warp forms run at one 32-lane
warp; cgrps must match block exactly (same impl body, GroupBarrier).
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, make_general

WARP = 32
SIZES = [1, 4, 7, 16]


def _write(arr):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    np.asarray(arr, dtype=np.float32).ravel(order="F").tofile(f)
    f.close()
    return f.name


def _run(bins, surface, threads, n, ct, A):
    fn = _write(A)
    try:
        cmd = [str(bins["symmetrize"]), surface, str(threads), str(n), str(int(ct)), fn]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        return np.fromstring(res.stdout.strip(), sep=" ").astype(np.float32).reshape(n, n, order="F")
    finally:
        os.unlink(fn)


def _expected(A):
    A = A.astype(np.float32)
    return (np.float32(0.5) * (A + A.T)).astype(np.float32)


# ─── block, runtime n: sizes × full thread sweep, byte-identical invariance ───

@pytest.mark.parametrize("n", SIZES)
def test_symmetrize_block_sweep(bins, n):
    for seed, scale in [(11, 1.0), (12, 1e3)]:
        A = make_general(n, seed=seed, scale=scale)
        expected = _expected(A)
        outs = [_run(bins, "block", t, n, 0, A) for t in THREAD_SWEEP]
        assert np.allclose(outs[0], expected, rtol=1e-6, atol=1e-6), f"n={n} seed={seed}\n{outs[0]}\nvs\n{expected}"
        assert np.allclose(outs[0], outs[0].T), "result not symmetric"
        for t, r in zip(THREAD_SWEEP[1:], outs[1:]):
            assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t} (n={n})"


# ─── compile-time <T, N> overload matches runtime ──────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_symmetrize_ct(bins, n):
    A = make_general(n, seed=21)
    expected = _expected(A)
    for t in (1, 33, 256):
        out = _run(bins, "block", t, n, 1, A)
        assert np.allclose(out, expected, rtol=1e-6, atol=1e-6), f"CT n={n} t={t}"
        assert np.array_equal(out, _run(bins, "block", t, n, 0, A)), f"CT != runtime (n={n}, t={t})"


# ─── warp variant at <<<1,32>>> (runtime + compile-time) ───────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("ct", [0, 1])
def test_symmetrize_warp(bins, n, ct):
    A = make_general(n, seed=31 + n)
    expected = _expected(A)
    out = _run(bins, "warp", WARP, n, ct, A)
    assert np.allclose(out, expected, rtol=1e-6, atol=1e-6), f"warp n={n} ct={ct}"
    assert np.array_equal(out, _run(bins, "block", WARP, n, ct, A)), "warp != block"


# ─── cgrps delegation matches block byte-identically ──────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_symmetrize_cgrps(bins, n):
    A = make_general(n, seed=41, scale=5.0)
    for t in (1, 57, 128):
        block = _run(bins, "block", t, n, 0, A)
        cg = _run(bins, "cgrps", t, n, 0, A)
        assert np.array_equal(block, cg), f"cgrps != block (n={n}, t={t})"


# ─── diagonal untouched + already-symmetric input is a fixed point ─────────────

def test_symmetrize_diagonal_and_fixed_point(bins):
    n = 7
    A = make_general(n, seed=51)
    out = _run(bins, "block", 64, n, 0, A)
    # out went through %.8g text formatting (not an exact float32 round trip),
    # so compare the untouched diagonal with a tight allclose, not array_equal.
    assert np.allclose(np.diag(out), np.diag(A.astype(np.float32)), rtol=1e-6, atol=0), "diagonal modified"
    S = _expected(make_general(n, seed=52))
    assert np.allclose(_run(bins, "block", 64, n, 0, S), S, rtol=1e-6, atol=1e-6), "symmetric input not a fixed point"
