"""Fused K-way inv / potrf tests, plus the single-warp glass::warp::inv.

Dedicated runner (test/cuda/test_fused.cu) to avoid contention on test_l3.
Each (K, dims) case is checked per-matrix against a NumPy oracle and swept over
several thread counts (including non-multiples of 32) for thread-invariance.

warp::inv ("winv") launches ONE block of dim3(32, W) with warp w inverting its
OWN matrix from its own shared-scratch span — the TRAJOPT-SOLVER/MPC-SOLVER warp-packed
Schur-inversion use-case. Each result is checked against np.linalg.inv AND
byte-compared against the single-matrix block glass::inv ("binv") at exactly 32
threads: the warp body mirrors inv_impl's phases and arithmetic bit-for-bit.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import make_spd  # shared; pass rng=RNG for varied draws

RNG = np.random.default_rng(7)

RTOL = 1e-2
ATOL = 1e-3

THREAD_SWEEP = [1, 7, 33, 256]

# (K, dims):
#   (1,[4])         == single-matrix path (degenerate K=1)
#   (2,[6,4])       == fused-2, ragged
#   (3,[12,12,6])   == TRAJOPT-SOLVER Schur (Q_k, Q_kp1, R_k)
#   (5,[8,3,8,2,5]) == ragged K=5
CASES = [
    (1, [4]),
    (2, [6, 4]),
    (3, [12, 12, 6]),
    (5, [8, 3, 8, 2, 5]),
]


def _aug(M, d):
    # [M | I] column-major augmented buffer for inv
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
    mats = [make_spd(d, rng=RNG) for d in dims]
    inputs = [_aug(M, d) for M, d in zip(mats, dims)]
    res = _run(fused_bin, "inv", threads, dims, inputs)
    assert len(res) == K
    for M, d, r in zip(mats, dims, res):
        gpu = r.reshape(d, d, order="F")
        ref = np.linalg.inv(M).astype(np.float32)
        assert np.allclose(gpu, ref, rtol=RTOL, atol=ATOL), \
            f"inv mismatch K={K} dims={dims} d={d} threads={threads}"


# ─── warp::inv — warp-packed, one matrix per warp ──────────────────────────────

# (W, d): W warps in one block, all matrices the same size d (the packing
# use-case: many identical small Schur blocks). d must be in the runner's
# compile-time set {4, 8, 12}.
WARP_CASES = [
    (1, 4),
    (4, 4),
    (2, 8),
    (4, 12),
]


@pytest.mark.parametrize("W,d", WARP_CASES)
def test_warp_inv(fused_bin, W, d):
    """Each warp inverts a DIFFERENT matrix: correct vs np.linalg.inv, and
    byte-identical to the block glass::inv run at exactly 32 threads (the warp
    port mirrors inv_impl's two-phase arithmetic bit-for-bit)."""
    mats = [make_spd(d, rng=RNG) for _ in range(W)]
    inputs = [_aug(M, d) for M in mats]
    res = _run(fused_bin, "winv", 32, [d] * W, inputs)
    assert len(res) == W
    for M, aug_in, r in zip(mats, inputs, res):
        gpu = r.reshape(d, d, order="F")
        ref = np.linalg.inv(M).astype(np.float32)
        assert np.allclose(gpu, ref, rtol=RTOL, atol=ATOL), \
            f"warp::inv mismatch vs numpy W={W} d={d}"
        block = _run(fused_bin, "binv", 32, [d], [aug_in])[0].reshape(d, d, order="F")
        assert np.array_equal(gpu, block), \
            f"warp::inv != block inv@32 threads (W={W}, d={d})"


@pytest.mark.parametrize("threads", THREAD_SWEEP)
def test_block_inv_baseline(fused_bin, threads):
    """The binv baseline op itself is thread-count invariant and correct."""
    d = 8
    M = make_spd(d, rng=RNG)
    outs = [_run(fused_bin, "binv", t, [d], [_aug(M, d)])[0] for t in (threads, 256)]
    ref = np.linalg.inv(M).astype(np.float32)
    assert np.allclose(outs[0].reshape(d, d, order="F"), ref, rtol=RTOL, atol=ATOL)
    assert np.array_equal(outs[0], outs[1]), f"binv non-invariant at {threads}"


@pytest.mark.parametrize("K,dims", CASES)
@pytest.mark.parametrize("threads", THREAD_SWEEP)
def test_fused_chol(fused_bin, K, dims, threads):
    mats = [make_spd(d, rng=RNG) for d in dims]
    inputs = [np.asfortranarray(M).ravel(order="F") for M in mats]
    res = _run(fused_bin, "chol", threads, dims, inputs)
    assert len(res) == K
    for M, d, r in zip(mats, dims, res):
        gpu = np.tril(r.reshape(d, d, order="F"))
        ref = np.linalg.cholesky(M).astype(np.float32)
        assert np.allclose(gpu, ref, rtol=RTOL, atol=ATOL), \
            f"chol mismatch K={K} dims={dims} d={d} threads={threads}"
