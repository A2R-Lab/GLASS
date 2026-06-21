"""glass::posv / potrs tests — SPD solve via Cholesky + two triangular solves.

The runner CLI is `<posv|potrs> <n> <threads> <A_or_L.bin> <b.bin>`, so we invoke
it directly rather than via conftest.run_op.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(11)
RTOL, ATOL = 1e-2, 1e-3


def _make_spd(n):
    A = RNG.standard_normal((n, n)).astype(np.float32)
    return (A @ A.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)


def _run(binary, op, n, threads, M, b):
    """Write M (n*n col-major) and b as float32 .bin, run, parse x."""
    tmp = []
    try:
        for arr in (np.asfortranarray(M).ravel(order="F"), b):
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f); f.close(); tmp.append(f.name)
        cmd = [str(binary), op, str(n), str(threads), tmp[0], tmp[1]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        return np.fromstring(res.stdout.strip().split("\n")[0], sep=" ").astype(np.float32)
    finally:
        for f in tmp:
            os.unlink(f)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_posv(bins, n):
    A = _make_spd(n)
    b = RNG.standard_normal(n).astype(np.float32)
    x = _run(bins["posv"], "posv", n, 256, A, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=RTOL, atol=ATOL)
    # residual backstop
    assert np.allclose(A @ x, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_potrs(bins, n):
    """Precomputed-factor path: pass the host Cholesky L, never the full A."""
    A = _make_spd(n)
    L = np.linalg.cholesky(A).astype(np.float32)   # lower
    b = RNG.standard_normal(n).astype(np.float32)
    x = _run(bins["posv"], "potrs", n, 256, L, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("op", ["posv", "potrs"])
@pytest.mark.parametrize("n", [4, 6, 8])
def test_posv_thread_invariance(bins, op, n):
    """Barrier correctness: identical x at 1/7/33/256 threads, matching the oracle."""
    A = _make_spd(n)
    b = RNG.standard_normal(n).astype(np.float32)
    M = np.linalg.cholesky(A).astype(np.float32) if op == "potrs" else A
    expected = np.linalg.solve(A, b)
    outs = []
    for threads in (1, 7, 33, 256):
        x = _run(bins["posv"], op, n, threads, M, b)
        assert np.allclose(x, expected, rtol=RTOL, atol=ATOL), f"threads={threads} mismatch"
        outs.append(x)
    for x in outs[1:]:
        assert np.array_equal(outs[0], x), "thread-count non-invariance"
