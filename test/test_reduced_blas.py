"""Contraction-parallel GEMV / SYRK (glass::gemv_reduced / glass::syrk_reduced).

Validates against NumPy. Within a surface the result is thread-count invariant
(bit-identical across block sizes); across surfaces it agrees to tolerance (FMA
may fuse differently per instantiation), so cross-surface checks use allclose.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, THREAD_SWEEP_CORE, make_vec

RNG = np.random.default_rng(53)
RTOL, ATOL = 2e-2, 2e-3

GEMV = [(14, 14), (7, 21), (8, 3), (33, 5), (5, 33), (64, 3), (3, 7)]
SYRK = [(14, 7), (8, 8), (5, 3), (33, 4), (7, 14), (64, 2)]
AL, BE = 1.5, 0.3


def _w(a):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    a.astype(np.float32).ravel(order="F").tofile(f)
    f.close()
    return f.name


def _gemv(binary, s, th, M, N, tr, A, x, y):
    t = [_w(A), _w(x), _w(y)]
    try:
        r = subprocess.run([str(binary), "gemv", s, str(th), str(M), str(N), str(int(tr)),
                            str(AL), str(BE)] + t, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return np.fromstring(r.stdout.strip(), sep=" ").astype(np.float32)
    finally:
        for f in t:
            os.unlink(f)


def _syrk(binary, s, th, R, C, tr, A, Cm):
    t = [_w(A), _w(Cm)]
    try:
        r = subprocess.run([str(binary), "syrk", s, str(th), str(R), str(C), str(int(tr)),
                            str(AL), str(BE)] + t, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        OUT = C if tr else R
        return np.fromstring(r.stdout.strip(), sep=" ").astype(np.float32).reshape(OUT, OUT, order="F")
    finally:
        for f in t:
            os.unlink(f)


# ─── gemv_reduced ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", GEMV)
@pytest.mark.parametrize("tr", [False, True])
def test_gemv_reduced(bins, M, N, tr):
    A = RNG.random((M, N)).astype(np.float32)
    xl, yl = (M if tr else N), (N if tr else M)
    x = RNG.random(xl).astype(np.float32)
    y0 = RNG.random(yl).astype(np.float32)
    expected = (AL * ((A.T @ x) if tr else (A @ x)) + BE * y0).astype(np.float32)
    outs = [_gemv(bins["reduced_blas"], "block", t, M, N, tr, A, x, y0.copy())
            for t in THREAD_SWEEP_CORE]
    assert np.allclose(outs[0], expected, rtol=RTOL, atol=ATOL)
    for t, r in zip(THREAD_SWEEP_CORE[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"
    for s, th in [("cgrps", 128), ("warp", 32)]:
        r = _gemv(bins["reduced_blas"], s, th, M, N, tr, A, x, y0.copy())
        assert np.allclose(r, outs[0], rtol=RTOL, atol=ATOL), f"{s} disagrees with block"


@pytest.mark.parametrize("M,N", [(14, 14), (33, 5), (64, 3)])
@pytest.mark.parametrize("kind", ["normal", "mixed"])
def test_gemv_thread_invariance(bins, M, N, kind):
    A = RNG.random((M, N)).astype(np.float32)
    x = make_vec(N, seed=M + N, kind=kind)
    y0 = RNG.random(M).astype(np.float32)
    outs = [_gemv(bins["reduced_blas"], "block", t, M, N, False, A, x, y0.copy())
            for t in THREAD_SWEEP]
    for t, r in zip(THREAD_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"


# ─── syrk_reduced ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("R,C", SYRK)
@pytest.mark.parametrize("tr", [False, True])
def test_syrk_reduced(bins, R, C, tr):
    A = RNG.random((R, C)).astype(np.float32)
    OUT = C if tr else R
    C0 = RNG.random((OUT, OUT)).astype(np.float32)
    C0 = (C0 + C0.T).astype(np.float32)
    expected = (AL * ((A.T @ A) if tr else (A @ A.T)) + BE * C0).astype(np.float32)
    outs = [_syrk(bins["reduced_blas"], "block", t, R, C, tr, A, C0.copy())
            for t in THREAD_SWEEP_CORE]
    assert np.allclose(outs[0], expected, rtol=RTOL, atol=ATOL)
    assert np.allclose(outs[0], outs[0].T, rtol=1e-4, atol=1e-5), "C not symmetric"
    for t, r in zip(THREAD_SWEEP_CORE[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"
    for s, th in [("cgrps", 96), ("warp", 32)]:
        r = _syrk(bins["reduced_blas"], s, th, R, C, tr, A, C0.copy())
        assert np.allclose(r, outs[0], rtol=RTOL, atol=ATOL), f"{s} disagrees with block"


@pytest.mark.parametrize("R,C", [(14, 7), (33, 4), (8, 8)])
@pytest.mark.parametrize("kind", ["normal", "scaled"])
def test_syrk_thread_invariance(bins, R, C, kind):
    A = RNG.random((R, C)).astype(np.float32)
    if kind == "scaled":   # alternating huge/tiny columns stress the lane partials
        A[:, 0::2] *= 1e3
        A[:, 1::2] *= 1e-3
    C0 = RNG.random((R, R)).astype(np.float32)
    C0 = (C0 + C0.T).astype(np.float32)
    outs = [_syrk(bins["reduced_blas"], "block", t, R, C, False, A, C0.copy())
            for t in THREAD_SWEEP]
    for t, r in zip(THREAD_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"
