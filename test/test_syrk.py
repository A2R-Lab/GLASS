"""SYRK / SYR2K GLASS function tests — compare GPU results to a NumPy oracle.

The CUDA runner (test_syrk.cu) has its own CLI:
    <op> <THREADS> <n> <k> <FILL> <TRANSPOSE> <ROW_MAJOR> <alpha> <beta> <A.bin> [<B.bin>] <C.bin>
so we invoke it directly rather than via conftest.run_op (whose fixed
`op version args... files...` shape doesn't match).

Thread discipline: syrk/syr2k assign one C entry per thread with a serial
contraction, so block results must be BYTE-IDENTICAL across block sizes.
The main matrix sweeps THREAD_SWEEP_CORE; test_syrk_thread_invariance covers
the full canonical THREAD_SWEEP across FillMode/TRANSPOSE and two input kinds.
warp:: forms always run one 32-lane warp.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, THREAD_SWEEP_CORE, make_general

RNG = np.random.default_rng(7)

RTOL = 1e-2
ATOL = 1e-3

# FillMode enum ints matching syrk.cuh: Lower=0, Upper=1, Full=2.
LOWER, UPPER, FULL = 0, 1, 2
FILL_IDS = {LOWER: "Lower", UPPER: "Upper", FULL: "Full"}


def _ravel(mat, row_major):
    """Flatten a 2D array in the storage order the kernel expects."""
    if row_major:
        return np.ascontiguousarray(mat).ravel(order="C")
    return np.asfortranarray(mat).ravel(order="F")


def _reshape(flat, n, row_major):
    return flat.reshape(n, n, order="C" if row_major else "F")


def _run(binary, op, threads, n, k, fill, trans, row_major,
         alpha, beta, A, B, C):
    """Write inputs as .bin, invoke the runner, parse the printed C (n*n)."""
    tmp = []
    try:
        arrays = [_ravel(A, row_major)]
        if op.startswith("syr2k"):
            arrays.append(_ravel(B, row_major))
        arrays.append(_ravel(C, row_major))
        for arr in arrays:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f)
            f.close()
            tmp.append(f.name)
        cmd = [str(binary), op, str(threads), str(n), str(k),
               str(fill), str(int(trans)), str(int(row_major)),
               str(alpha), str(beta)] + tmp
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        line = res.stdout.strip().split("\n")[0]
        flat = np.fromstring(line, sep=" ").astype(np.float32)
        return _reshape(flat, n, row_major)
    finally:
        for f in tmp:
            os.unlink(f)


def _oracle_syrk(alpha, A, beta, C0, trans):
    if trans:
        return alpha * (A.T @ A) + beta * C0
    return alpha * (A @ A.T) + beta * C0


def _oracle_syr2k(alpha, A, B, beta, C0, trans):
    if trans:
        return alpha * (A.T @ B + B.T @ A) + beta * C0
    return alpha * (A @ B.T + B @ A.T) + beta * C0


def _shapes(n, k, trans):
    """Shape of A (and B) given n,k and TRANSPOSE. TRANSPOSE=false → n x k; true → k x n."""
    return (k, n) if trans else (n, k)


def _check(result, expected, n, fill, C0):
    """Compare per FILL semantics; for triangular fills assert the OTHER
    triangle equals the input C (untouched)."""
    ri, ci = np.indices((n, n))
    if fill == FULL:
        assert np.allclose(result, expected, rtol=RTOL, atol=ATOL), \
            f"\nresult=\n{result}\nexpected=\n{expected}"
    elif fill == LOWER:
        lo = ri >= ci
        assert np.allclose(result[lo], expected[lo], rtol=RTOL, atol=ATOL)
        # untouched triangle must be byte-for-byte the input C — equal_nan so the
        # NaN-poisoned (beta=0) cells prove the kernel never read/wrote them.
        up = ri < ci
        assert np.allclose(result[up], C0[up], rtol=RTOL, atol=ATOL, equal_nan=True), \
            "upper triangle must be left untouched"
    else:  # UPPER
        up = ri <= ci
        assert np.allclose(result[up], expected[up], rtol=RTOL, atol=ATOL)
        lo = ri > ci
        assert np.allclose(result[lo], C0[lo], rtol=RTOL, atol=ATOL, equal_nan=True), \
            "lower triangle must be left untouched"


PAIRS = [(1, 1), (3, 5), (5, 3), (4, 4), (7, 2), (8, 8)]
FILLS = [LOWER, UPPER, FULL]
TRANSES = [False, True]
ALPHA_BETA = [(1.5, 0.3), (1.0, 0.0), (0.0, 0.3)]


@pytest.mark.parametrize("op", ["syrk", "syr2k"])
@pytest.mark.parametrize("n,k", PAIRS)
@pytest.mark.parametrize("fill", FILLS, ids=lambda f: FILL_IDS[f])
@pytest.mark.parametrize("trans", TRANSES)
@pytest.mark.parametrize("alpha,beta", ALPHA_BETA)
@pytest.mark.parametrize("row_major", [False, True])
def test_syrk(bins, op, n, k, fill, trans, alpha, beta, row_major):
    sh = _shapes(n, k, trans)
    A = RNG.random(sh).astype(np.float32)
    B = RNG.random(sh).astype(np.float32) if op == "syr2k" else None
    C = RNG.random((n, n)).astype(np.float32)
    C0 = C.copy()

    # For beta=0 + triangular fill, poison the untouched triangle with NaN to
    # PROVE the kernel never reads it (NaN in → NaN out only if it's read).
    if beta == 0.0 and fill != FULL:
        ri, ci = np.indices((n, n))
        if fill == LOWER:
            C[ri < ci] = np.nan
        else:
            C[ri > ci] = np.nan
        C0 = C.copy()

    outs = []
    for threads in THREAD_SWEEP_CORE:
        result = _run(bins["syrk"], op, threads, n, k, fill, trans, row_major,
                      alpha, beta, A, B, C)
        outs.append(result)

    if op == "syrk":
        expected = _oracle_syrk(alpha, A, beta, C0, trans)
    else:
        expected = _oracle_syr2k(alpha, A, B, beta, C0, trans)
    expected = expected.astype(np.float32)
    _check(outs[0], expected, n, fill, C0)
    for threads, r in zip(THREAD_SWEEP_CORE[1:], outs[1:]):
        # equal_nan: beta=0 + triangular fill NaN-poisons the untouched triangle
        assert np.array_equal(outs[0], r, equal_nan=True), \
            f"thread-count non-invariance at {threads} threads"


# ─── thread-count invariance: FULL sweep × FillMode × TRANSPOSE × input kind ──

def _syrk_input(sh, seed, kind):
    if kind == "colscaled":   # alternating huge/tiny columns stress accumulation
        A = make_general(*sh, seed=seed).copy()
        A[:, 0::2] *= 1e3
        A[:, 1::2] *= 1e-3
        return A.astype(np.float32)
    return make_general(*sh, seed=seed)


@pytest.mark.parametrize("op", ["syrk", "syr2k"])
@pytest.mark.parametrize("n,k", [(7, 2), (8, 8)])
@pytest.mark.parametrize("fill", FILLS, ids=lambda f: FILL_IDS[f])
@pytest.mark.parametrize("trans", TRANSES)
@pytest.mark.parametrize("kind", ["normal", "colscaled"])
def test_syrk_thread_invariance(bins, op, n, k, fill, trans, kind):
    """Byte-identical output over the FULL canonical THREAD_SWEEP for every
    FillMode/TRANSPOSE combo and two input kinds, matching the oracle."""
    alpha, beta = 1.5, 0.3
    sh = _shapes(n, k, trans)
    A = _syrk_input(sh, 40 + n + k, kind)
    B = _syrk_input(sh, 41 + n + k, kind) if op == "syr2k" else None
    C = make_general(n, n, seed=42 + n + k)
    C0 = C.copy()

    if op == "syrk":
        expected = _oracle_syrk(alpha, A, beta, C0, trans).astype(np.float32)
    else:
        expected = _oracle_syr2k(alpha, A, B, beta, C0, trans).astype(np.float32)

    outs = []
    for threads in THREAD_SWEEP:
        outs.append(_run(bins["syrk"], op, threads, n, k, fill, trans, False,
                         alpha, beta, A, B, C.copy()))
    for threads, r in zip(THREAD_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), \
            f"thread-count non-invariance at {threads} threads"
    # oracle check per FILL semantics (rtol-dominated: colscaled entries ~1e6)
    _check(outs[0], expected, n, fill, C0)


# ─── warp forms (compile-time N,K; one 32-lane warp) ──────────────────────────
# warp::syrk/syr2k reuse the SAME validated syrk_impl_ct/syr2k_impl_ct as the block
# form (just (lane,32) striping), so the warp output must equal the block output
# bit-for-bit AND satisfy the oracle. Harness instantiates these (n,k) shapes only.
WARP_PAIRS = [(4, 6), (6, 4), (7, 6)]
WARP = 32


@pytest.mark.parametrize("op", ["syrk_warp", "syr2k_warp"])
@pytest.mark.parametrize("n,k", WARP_PAIRS)
@pytest.mark.parametrize("fill", FILLS, ids=lambda f: FILL_IDS[f])
@pytest.mark.parametrize("trans", TRANSES)
@pytest.mark.parametrize("row_major", [False, True])
def test_syrk_warp(bins, op, n, k, fill, trans, row_major):
    block_op = op.replace("_warp", "")
    alpha, beta = 1.5, 0.3
    sh = _shapes(n, k, trans)
    A = RNG.random(sh).astype(np.float32)
    B = RNG.random(sh).astype(np.float32) if block_op == "syr2k" else None
    C = RNG.random((n, n)).astype(np.float32)
    C0 = C.copy()
    warp = _run(bins["syrk"], op, WARP, n, k, fill, trans, row_major, alpha, beta, A, B, C)
    block = _run(bins["syrk"], block_op, 64, n, k, fill, trans, row_major, alpha, beta, A, B, C)
    # warp == block bit-for-bit (same impl, restricted to one warp)
    assert np.array_equal(warp, block), f"{op} n={n} k={k} fill={fill}: warp != block"
    # and matches the oracle per FILL semantics
    expected = (_oracle_syrk(alpha, A, beta, C0, trans) if block_op == "syrk"
                else _oracle_syr2k(alpha, A, B, beta, C0, trans))
    _check(warp, expected, n, fill, C0)
