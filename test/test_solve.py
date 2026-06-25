"""posv REGULARIZE/CHECK flags and riccati_gain.

The posv off-path (REGULARIZE=CHECK=false) byte-identity is covered by the
existing test_posv suite still passing. Here: the regularized shift solves
`(A+rho I) X = B`, CHECK flags a non-PD A, and riccati_gain reproduces
`K = (R + BᵀPB)⁻¹ BᵀPA` and is thread-count invariant.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(41)
RTOL, ATOL = 3e-2, 3e-3

POSV = [(7, 7), (8, 5), (14, 7), (5, 1), (3, 3), (7, 14)]
RIC = [(14, 7), (8, 4), (6, 3), (10, 5), (4, 2)]
THREADS_SWEEP = (1, 7, 32, 33, 64, 128)


def _w(a):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    a.astype(np.float32).ravel(order="F").tofile(f)
    f.close()
    return f.name


def _posv(binary, th, N, NRHS, reg, rho, A, B):
    t = [_w(A), _w(B)]
    try:
        r = subprocess.run([str(binary), "posvreg", str(th), str(N), str(NRHS),
                            str(int(reg)), str(rho)] + t, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        L = r.stdout.strip().split("\n")
        return int(L[0]), np.fromstring(L[1], sep=" ").reshape(N, NRHS, order="F")
    finally:
        for f in t:
            os.unlink(f)


def _ricc(binary, th, NX, NU, reg, rho, P, A, B, R):
    t = [_w(P), _w(A), _w(B), _w(R)]
    try:
        r = subprocess.run([str(binary), "riccati", str(th), str(NX), str(NU),
                            str(int(reg)), str(rho)] + t, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        L = r.stdout.strip().split("\n")
        return int(L[0]), np.fromstring(L[1], sep=" ").reshape(NU, NX, order="F")
    finally:
        for f in t:
            os.unlink(f)


def _spd(n):
    M = RNG.random((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


@pytest.mark.parametrize("N,NRHS", POSV)
@pytest.mark.parametrize("reg", [False, True])
def test_posv_flags(bins, N, NRHS, reg):
    A = _spd(N)
    B = RNG.random((N, NRHS)).astype(np.float32)
    rho = 0.5 if reg else 0.0
    fail, X = _posv(bins["solve"], 128, N, NRHS, reg, rho, A.copy(), B.copy())
    expected = np.linalg.solve(A + rho * np.eye(N), B)
    assert fail == 0, "SPD must not flag"
    assert np.allclose(X, expected, rtol=RTOL, atol=ATOL), f"err {np.abs(X-expected).max()}"


def test_posv_check_non_pd(bins):
    ND = (-_spd(7)).astype(np.float32)
    B = RNG.random((7, 7)).astype(np.float32)
    fail, _ = _posv(bins["solve"], 128, 7, 7, False, 0.0, ND.copy(), B.copy())
    assert fail == 1, "non-PD A must set fail=1"


@pytest.mark.parametrize("NX,NU", RIC)
def test_riccati_gain(bins, NX, NU):
    P = _spd(NX)
    A = RNG.random((NX, NX)).astype(np.float32)
    B = RNG.random((NX, NU)).astype(np.float32)
    R = _spd(NU)
    fail, K = _ricc(bins["solve"], 128, NX, NU, False, 0.0, P, A, B, R)
    S = R + B.T @ P @ B
    expK = np.linalg.solve(S, B.T @ P @ A)
    assert fail == 0
    assert np.allclose(K, expK, rtol=RTOL, atol=ATOL), f"err {np.abs(K-expK).max()}"


@pytest.mark.parametrize("NX,NU", [(14, 7), (8, 4)])
def test_riccati_thread_invariance(bins, NX, NU):
    P = _spd(NX)
    A = RNG.random((NX, NX)).astype(np.float32)
    B = RNG.random((NX, NU)).astype(np.float32)
    R = _spd(NU)
    outs = [_ricc(bins["solve"], t, NX, NU, False, 0.0, P, A, B, R)[1] for t in THREADS_SWEEP]
    for t, r in zip(THREADS_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"
