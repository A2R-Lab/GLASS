"""Contraction-parallel GEMM (glass::gemm_reduced) tests — block / warp / cgrps.

Validates the `*_reduced` engine against a NumPy oracle and, crucially, that it
is thread-count invariant: bit-identical across the 32-thread boundary (the
register fallback below 32 threads reproduces the warp-shuffle reduction order
exactly), and that the warp / cgrps surfaces match the block surface bit-for-bit.

The CUDA runner (test_reduced.cu) has its own CLI:
    <surface> <THREADS> <M> <N> <K> <TRANSPOSE_B> <ROW_MAJOR> <alpha> <beta> <A> <B> <C>
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(11)

RTOL = 1e-2
ATOL = 1e-3

# (M, N, K) shapes the runner compiles — consumer dims, partial-warp output
# counts, and contraction lengths spanning the 32-lane boundary (N = 33, 64).
SHAPES = [
    (1, 1, 1), (3, 5, 4), (5, 3, 6), (7, 2, 9), (8, 8, 8),
    (14, 14, 14), (21, 21, 21), (14, 21, 7), (21, 14, 21), (7, 14, 7),
    (2, 33, 2), (4, 64, 4),
]
# Square shapes only for TRANSPOSE_B (B must be N x N).
SQUARE = [(m, n, k) for (m, n, k) in SHAPES if n == k]

# Full thread-invariance sweep for the block surface — spans the 32 boundary;
# 31 & 57 are the load-bearing partial-warp cases.
THREADS_SWEEP = (1, 7, 31, 32, 33, 57, 64, 96, 128, 256)


def _ravel(mat, row_major):
    if row_major:
        return np.ascontiguousarray(mat).ravel(order="C")
    return np.asfortranarray(mat).ravel(order="F")


def _reshape(flat, m, ccols, row_major):
    return flat.reshape(m, ccols, order="C" if row_major else "F")


def _run(binary, surface, threads, M, N, K, tb, rm, alpha, beta, A, B, C):
    ccols = N if tb else K
    tmp = []
    try:
        for arr in (_ravel(A, rm), _ravel(B, rm), _ravel(C, rm)):
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f)
            f.close()
            tmp.append(f.name)
        cmd = [str(binary), surface, str(threads), str(M), str(N), str(K),
               str(int(tb)), str(int(rm)), str(alpha), str(beta)] + tmp
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        line = res.stdout.strip().split("\n")[0]
        flat = np.fromstring(line, sep=" ").astype(np.float32)
        return _reshape(flat, M, ccols, rm)
    finally:
        for f in tmp:
            os.unlink(f)


def _oracle(alpha, A, B, beta, C0, tb):
    prod = (A @ B.T) if tb else (A @ B)
    return (alpha * prod + beta * C0).astype(np.float32)


def _make(M, N, K, tb):
    A = RNG.random((M, N)).astype(np.float32)
    B = RNG.random((N, N) if tb else (N, K)).astype(np.float32)
    ccols = N if tb else K
    C = RNG.random((M, ccols)).astype(np.float32)
    return A, B, C


# ─── correctness vs NumPy oracle, all 3 surfaces ─────────────────────────────

@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
@pytest.mark.parametrize("row_major", [False, True])
def test_reduced_block(bins, M, N, K, alpha, beta, row_major):
    A, B, C = _make(M, N, K, False)
    expected = _oracle(alpha, A, B, beta, C, False)
    r = _run(bins["reduced"], "block", 256, M, N, K, False, row_major, alpha, beta, A, B, C.copy())
    assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), f"\nresult=\n{r}\nexpected=\n{expected}"


@pytest.mark.parametrize("M,N,K", SQUARE)
@pytest.mark.parametrize("row_major", [False, True])
def test_reduced_transpose_b(bins, M, N, K, row_major):
    alpha, beta = 1.5, 0.3
    A, B, C = _make(M, N, K, True)
    expected = _oracle(alpha, A, B, beta, C, True)
    r = _run(bins["reduced"], "block", 256, M, N, K, True, row_major, alpha, beta, A, B, C.copy())
    assert np.allclose(r, expected, rtol=RTOL, atol=ATOL)


# ─── thread-count invariance: bit-identical across the 32 boundary ───────────

@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (14, 14, 14), (7, 2, 9), (2, 33, 2), (4, 64, 4)])
def test_reduced_thread_invariance(bins, M, N, K):
    """Block surface: bit-identical at 1/7/31/32/33/57/64/96/128/256 threads
    (the <32 register path must match the warp-shuffle rounding), all matching
    the oracle."""
    alpha, beta = 1.5, 0.3
    A, B, C = _make(M, N, K, False)
    expected = _oracle(alpha, A, B, beta, C, False)
    outs = []
    for t in THREADS_SWEEP:
        r = _run(bins["reduced"], "block", t, M, N, K, False, False, alpha, beta, A, B, C.copy())
        assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), f"threads={t} mismatch vs oracle"
        outs.append(r)
    for t, r in zip(THREADS_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"


# ─── cross-surface agreement: warp / cgrps == block, bit-for-bit ─────────────

@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (14, 14, 14), (21, 14, 21), (4, 64, 4)])
def test_reduced_surfaces_agree(bins, M, N, K):
    alpha, beta = 1.5, 0.3
    A, B, C = _make(M, N, K, False)
    block = _run(bins["reduced"], "block", 256, M, N, K, False, False, alpha, beta, A, B, C.copy())
    warp = _run(bins["reduced"], "warp", 32, M, N, K, False, False, alpha, beta, A, B, C.copy())
    assert np.array_equal(block, warp), "warp surface != block surface"
    for t in (32, 64, 128, 256):
        cg = _run(bins["reduced"], "cgrps", t, M, N, K, False, False, alpha, beta, A, B, C.copy())
        assert np.array_equal(block, cg), f"cgrps surface != block surface at {t} threads"


# ─── beta=0 must not read C (NaN-poison proof) ───────────────────────────────

@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (14, 14, 14)])
def test_reduced_beta0_skips_C(bins, M, N, K):
    alpha = 1.0
    A, B, _ = _make(M, N, K, False)
    C = np.full((M, K), np.nan, dtype=np.float32)
    r = _run(bins["reduced"], "block", 256, M, N, K, False, False, alpha, 0.0, A, B, C)
    expected = (alpha * (A @ B)).astype(np.float32)
    assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), "beta=0 path read NaN-poisoned C"
