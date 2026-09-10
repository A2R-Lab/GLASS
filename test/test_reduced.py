"""Contraction-parallel GEMM (glass::gemm_reduced) tests — block / warp / cgrps.

Validates the `*_reduced` engine against a NumPy oracle and, crucially, that it
is thread-count invariant: bit-identical across the 32-thread boundary (the
register fallback below 32 threads reproduces the warp-shuffle reduction order
exactly), and that the warp / cgrps surfaces match the block surface bit-for-bit.

Standard BLAS convention: C is M×N, contraction K; op(A) is M×K (TRANSPOSE_A ⇒ A
stored K×M), op(B) is K×N (TRANSPOSE_B ⇒ B stored N×K). The CUDA runner CLI:
    <surface> <THREADS> <M> <N> <K> <TRANSPOSE_A> <TRANSPOSE_B> <alpha> <beta> <A> <B> <C>
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, THREAD_SWEEP_CORE

RNG = np.random.default_rng(11)

RTOL = 1e-2
ATOL = 1e-3

# (M, N, K) shapes the runner compiles — consumer dims, partial-warp output
# counts, and CONTRACTION lengths spanning the 32-lane boundary (K = 33, 64).
SHAPES = [
    (1, 1, 1), (3, 5, 4), (5, 3, 6), (7, 2, 9), (8, 8, 8),
    (14, 14, 14), (21, 21, 21), (14, 7, 21), (21, 21, 14), (7, 7, 14),
    (2, 2, 33), (4, 4, 64),
]
TRANSPOSE_COMBOS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _storage(M, N, K, ta, tb, seed, kind="normal"):
    """Logical op(A) (M×K), op(B) (K×N) + physical col-major storage (transpose
    stores op(_)ᵀ col-major). kind='mixed' alternates huge/tiny contraction
    slices (1e3/1e-3) to stress the lane-partial accumulation."""
    rng = np.random.default_rng(seed)
    opA = rng.standard_normal((M, K)).astype(np.float32)
    opB = rng.standard_normal((K, N)).astype(np.float32)
    if kind == "mixed":
        opA[:, 0::2] *= 1e3
        opA[:, 1::2] *= 1e-3
    A_phys = opA.T if ta else opA
    B_phys = opB.T if tb else opB
    return opA, opB, A_phys, B_phys


def _run(binary, surface, threads, M, N, K, ta, tb, alpha, beta, A_phys, B_phys, C):
    tmp = []
    try:
        for arr in (np.asfortranarray(A_phys).ravel(order="F"),
                    np.asfortranarray(B_phys).ravel(order="F"),
                    np.asfortranarray(C).ravel(order="F")):
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f)
            f.close()
            tmp.append(f.name)
        cmd = [str(binary), surface, str(threads), str(M), str(N), str(K),
               str(int(ta)), str(int(tb)), str(alpha), str(beta)] + tmp
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        line = res.stdout.strip().split("\n")[0]
        flat = np.fromstring(line, sep=" ").astype(np.float32)
        return flat.reshape(M, N, order="F")
    finally:
        for f in tmp:
            os.unlink(f)


def _oracle(alpha, opA, opB, beta, C0):
    return (alpha * (opA @ opB) + beta * C0).astype(np.float32)


# ─── correctness vs NumPy oracle, all transpose combos ───────────────────────

@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
def test_reduced_block(bins, M, N, K, ta, tb, alpha, beta):
    """Oracle-correct AND bit-identical over the core sweep (the *_reduced
    engine is engineered to be thread-count invariant across the 32 boundary)."""
    opA, opB, A_phys, B_phys = _storage(M, N, K, ta, tb, seed=M * 100 + N * 10 + K + ta + tb)
    C = RNG.random((M, N)).astype(np.float32)
    expected = _oracle(alpha, opA, opB, beta, C)
    outs = []
    for t in THREAD_SWEEP_CORE:
        outs.append(_run(bins["reduced"], "block", t, M, N, K, ta, tb, alpha, beta,
                         A_phys, B_phys, C.copy()))
    assert np.allclose(outs[0], expected, rtol=RTOL, atol=ATOL), \
        f"\nresult=\n{outs[0]}\nexpected=\n{expected}"
    for t, r in zip(THREAD_SWEEP_CORE[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"


# ─── thread-count invariance: bit-identical across the 32 boundary ───────────

@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (14, 14, 14), (7, 2, 9), (2, 2, 33), (4, 4, 64)])
@pytest.mark.parametrize("kind", ["normal", "mixed"])
def test_reduced_thread_invariance(bins, M, N, K, kind):
    """Block surface: bit-identical at 1/7/31/32/33/57/64/96/128/256 threads
    (the <32 register path must match the warp-shuffle rounding), all matching
    the oracle. 'mixed' alternates huge/tiny contraction slices."""
    alpha, beta = 1.5, 0.3
    opA, opB, A_phys, B_phys = _storage(M, N, K, 0, 0, seed=M + N + K, kind=kind)
    C = RNG.random((M, N)).astype(np.float32)
    expected = _oracle(alpha, opA, opB, beta, C)
    # mixed magnitudes ~1e3 → rtol-dominated closeness
    atol = ATOL * (1e3 if kind == "mixed" else 1.0)
    outs = []
    for t in THREAD_SWEEP:
        r = _run(bins["reduced"], "block", t, M, N, K, 0, 0, alpha, beta, A_phys, B_phys, C.copy())
        assert np.allclose(r, expected, rtol=RTOL, atol=atol), f"threads={t} mismatch vs oracle"
        outs.append(r)
    for t, r in zip(THREAD_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"


# ─── cross-surface agreement: warp / cgrps == block, bit-for-bit ─────────────

@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (14, 14, 14), (21, 21, 14), (4, 4, 64)])
def test_reduced_surfaces_agree(bins, M, N, K):
    alpha, beta = 1.5, 0.3
    opA, opB, A_phys, B_phys = _storage(M, N, K, 0, 0, seed=M * 7 + N * 3 + K)
    C = RNG.random((M, N)).astype(np.float32)
    block = _run(bins["reduced"], "block", 256, M, N, K, 0, 0, alpha, beta, A_phys, B_phys, C.copy())
    warp = _run(bins["reduced"], "warp", 32, M, N, K, 0, 0, alpha, beta, A_phys, B_phys, C.copy())
    assert np.array_equal(block, warp), "warp surface != block surface"
    for t in (32, 64, 128, 256):
        cg = _run(bins["reduced"], "cgrps", t, M, N, K, 0, 0, alpha, beta, A_phys, B_phys, C.copy())
        assert np.array_equal(block, cg), f"cgrps surface != block surface at {t} threads"


# ─── beta=0 must not read C (NaN-poison proof) ───────────────────────────────

@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (14, 14, 14)])
def test_reduced_beta0_skips_C(bins, M, N, K):
    alpha = 1.0
    opA, opB, A_phys, B_phys = _storage(M, N, K, 0, 0, seed=M + N + K + 1)
    C = np.full((M, N), np.nan, dtype=np.float32)
    r = _run(bins["reduced"], "block", 256, M, N, K, 0, 0, alpha, 0.0, A_phys, B_phys, C)
    expected = (alpha * (opA @ opB)).astype(np.float32)
    assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), "beta=0 path read NaN-poisoned C"
