"""Congruence / bilinear forms (glass::congruence_sym / glass::bilinear).

Validates the fused two-step products against NumPy. Within a surface the result
is thread-count invariant (bit-identical across block sizes); across surfaces it
agrees to floating-point tolerance (step 1's gemm may fuse FMA differently per
instantiation — see the header note), so cross-surface checks use allclose.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(37)
RTOL, ATOL = 2e-2, 2e-3

CONG = [(14, 21), (8, 8), (5, 3), (7, 14), (33, 5), (64, 3), (14, 14), (3, 4)]
BIL = [(14, 7, 21), (8, 5, 3), (5, 5, 5), (33, 4, 6), (7, 14, 7)]
THREADS_SWEEP = (1, 7, 31, 32, 33, 57, 64, 96, 128, 256)
ALPHA, BETA = 1.5, 0.3


def _w(a):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    a.astype(np.float32).ravel(order="F").tofile(f)
    f.close()
    return f.name


def _run(binary, args, files, rows, cols):
    tmp = [_w(a) for a in files]
    try:
        r = subprocess.run([str(binary)] + args + tmp, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return np.fromstring(r.stdout.strip(), sep=" ").astype(np.float32).reshape(rows, cols, order="F")
    finally:
        for f in tmp:
            os.unlink(f)


def _cong(binary, surf, th, N, K, acc, X, M, Q):
    return _run(binary, ["cong", surf, str(th), str(N), str(K), str(int(acc)), str(ALPHA), str(BETA)],
                [X, M, Q], K, K)


def _bil(binary, surf, th, N, P, Qd, acc, X, M, Y, R):
    return _run(binary, ["bil", surf, str(th), str(N), str(P), str(Qd), str(int(acc)), str(ALPHA), str(BETA)],
                [X, M, Y, R], P, Qd)


def _sym(n):
    A = RNG.random((n, n)).astype(np.float32)
    return (A + A.T).astype(np.float32)


# ─── congruence_sym: Q = alpha X^T M X + beta Q (symmetric) ───────────────────

@pytest.mark.parametrize("N,K", CONG)
@pytest.mark.parametrize("acc", [False, True])
def test_congruence_sym(bins, N, K, acc):
    X = RNG.random((N, K)).astype(np.float32)
    M = _sym(N)
    Q0 = _sym(K)
    expected = (ALPHA * (X.T @ M @ X) + (BETA * Q0 if acc else 0)).astype(np.float32)
    block = _cong(bins["congruence"], "block", 128, N, K, acc, X, M, Q0.copy())
    assert np.allclose(block, expected, rtol=RTOL, atol=ATOL), f"\n{block}\nvs\n{expected}"
    assert np.allclose(block, block.T, rtol=1e-4, atol=1e-5), "Q not symmetric"
    for surf, th in [("cgrps", 96), ("warp", 32)]:
        r = _cong(bins["congruence"], surf, th, N, K, acc, X, M, Q0.copy())
        assert np.allclose(r, block, rtol=RTOL, atol=ATOL), f"{surf} disagrees with block"


@pytest.mark.parametrize("N,K", [(14, 21), (33, 5), (64, 3), (8, 8)])
def test_congruence_thread_invariance(bins, N, K):
    X = RNG.random((N, K)).astype(np.float32)
    M = _sym(N)
    Q0 = _sym(K)
    outs = [_cong(bins["congruence"], "block", t, N, K, True, X, M, Q0.copy()) for t in THREADS_SWEEP]
    for t, r in zip(THREADS_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"


# ─── bilinear: R = alpha X^T M Y + beta R (general) ──────────────────────────

@pytest.mark.parametrize("N,P,Qd", BIL)
@pytest.mark.parametrize("acc", [False, True])
def test_bilinear(bins, N, P, Qd, acc):
    X = RNG.random((N, P)).astype(np.float32)
    M = RNG.random((N, N)).astype(np.float32)
    Y = RNG.random((N, Qd)).astype(np.float32)
    R0 = RNG.random((P, Qd)).astype(np.float32)
    expected = (ALPHA * (X.T @ M @ Y) + (BETA * R0 if acc else 0)).astype(np.float32)
    block = _bil(bins["congruence"], "block", 128, N, P, Qd, acc, X, M, Y, R0.copy())
    assert np.allclose(block, expected, rtol=RTOL, atol=ATOL)
    for surf, th in [("cgrps", 128), ("warp", 32)]:
        r = _bil(bins["congruence"], surf, th, N, P, Qd, acc, X, M, Y, R0.copy())
        assert np.allclose(r, block, rtol=RTOL, atol=ATOL), f"{surf} disagrees with block"


@pytest.mark.parametrize("N,P,Qd", [(14, 7, 21), (33, 4, 6)])
def test_bilinear_thread_invariance(bins, N, P, Qd):
    X = RNG.random((N, P)).astype(np.float32)
    M = RNG.random((N, N)).astype(np.float32)
    Y = RNG.random((N, Qd)).astype(np.float32)
    R0 = RNG.random((P, Qd)).astype(np.float32)
    outs = [_bil(bins["congruence"], "block", t, N, P, Qd, True, X, M, Y, R0.copy()) for t in THREADS_SWEEP]
    for t, r in zip(THREADS_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"
