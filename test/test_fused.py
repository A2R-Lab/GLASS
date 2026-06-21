"""Fused K-way invertMatrix / cholDecomp_InPlace tests.

Dedicated runner (test/cuda/test_fused.cu) to avoid contention on test_l3.
Each (K, dims) case is checked per-matrix against a NumPy oracle and swept over
several thread counts (including non-multiples of 32) for thread-invariance.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(7)

RTOL = 1e-2
ATOL = 1e-3

THREAD_SWEEP = [1, 7, 33, 256]

# (K, dims):
#   (1,[4])         == single-matrix path (degenerate K=1)
#   (2,[6,4])       == fused-2, ragged
#   (3,[12,12,6])   == GATO Schur (Q_k, Q_kp1, R_k)
#   (5,[8,3,8,2,5]) == ragged K=5
CASES = [
    (1, [4]),
    (2, [6, 4]),
    (3, [12, 12, 6]),
    (5, [8, 3, 8, 2, 5]),
]


def make_spd(n):
    A = RNG.random((n, n)).astype(np.float32)
    return (A @ A.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)


def _aug(M, d):
    # [M | I] column-major augmented buffer for invertMatrix
    return np.asfortranarray(np.hstack([M, np.eye(d, dtype=np.float32)])).ravel(order="F")


def _run(binary, op, threads, dims, mats):
    """Invoke test_fused: <op> <threads> K d0..d_{K-1} MAX_DIM <files...>.

    `mats` is a list of flattened float32 column-major buffers (augmented for inv,
    plain for chol). Returns a list of K float32 arrays (one printed line each).
    """
    K = len(dims)
    tmpfiles = []
    try:
        for arr in mats:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f)
            f.close()
            tmpfiles.append(f.name)
        cmd = [str(binary), op, str(threads), str(K)]
        cmd += [str(d) for d in dims]
        cmd += [str(max(dims))]
        cmd += tmpfiles
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Binary failed:\n{result.stderr}")
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        return [np.fromstring(l, sep=" ").astype(np.float32) for l in lines]
    finally:
        for f in tmpfiles:
            os.unlink(f)


@pytest.fixture(scope="session")
def fused_bin(bins):
    return bins["fused"]


@pytest.mark.parametrize("K,dims", CASES)
@pytest.mark.parametrize("threads", THREAD_SWEEP)
def test_fused_inv(fused_bin, K, dims, threads):
    mats = [make_spd(d) for d in dims]
    inputs = [_aug(M, d) for M, d in zip(mats, dims)]
    res = _run(fused_bin, "inv", threads, dims, inputs)
    assert len(res) == K
    for M, d, r in zip(mats, dims, res):
        gpu = r.reshape(d, d, order="F")
        ref = np.linalg.inv(M).astype(np.float32)
        assert np.allclose(gpu, ref, rtol=RTOL, atol=ATOL), \
            f"inv mismatch K={K} dims={dims} d={d} threads={threads}"


@pytest.mark.parametrize("K,dims", CASES)
@pytest.mark.parametrize("threads", THREAD_SWEEP)
def test_fused_chol(fused_bin, K, dims, threads):
    mats = [make_spd(d) for d in dims]
    inputs = [np.asfortranarray(M).ravel(order="F") for M in mats]
    res = _run(fused_bin, "chol", threads, dims, inputs)
    assert len(res) == K
    for M, d, r in zip(mats, dims, res):
        gpu = np.tril(r.reshape(d, d, order="F"))
        ref = np.linalg.cholesky(M).astype(np.float32)
        assert np.allclose(gpu, ref, rtol=RTOL, atol=ATOL), \
            f"chol mismatch K={K} dims={dims} d={d} threads={threads}"
