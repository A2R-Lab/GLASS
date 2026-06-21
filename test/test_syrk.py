"""SYRK / SYR2K GLASS function tests — compare GPU results to a NumPy oracle.

The CUDA runner (test_syrk.cu) has its own CLI:
    <op> <THREADS> <n> <k> <FILL> <TRANS> <ROW_MAJOR> <alpha> <beta> <A.bin> [<B.bin>] <C.bin>
so we invoke it directly rather than via conftest.run_op (whose fixed
`op version args... files...` shape doesn't match).
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

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
        if op == "syr2k":
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
    """Shape of A (and B) given n,k and TRANS. TRANS=false → n x k; true → k x n."""
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

    result = _run(bins["syrk"], op, 256, n, k, fill, trans, row_major,
                  alpha, beta, A, B, C)

    if op == "syrk":
        expected = _oracle_syrk(alpha, A, beta, C0, trans)
    else:
        expected = _oracle_syr2k(alpha, A, B, beta, C0, trans)
    expected = expected.astype(np.float32)
    _check(result, expected, n, fill, C0)


# ─── thread-count invariance sweep (Full, beta != 0) ─────────────────────────

@pytest.mark.parametrize("op", ["syrk", "syr2k"])
@pytest.mark.parametrize("n,k", [(8, 8), (7, 2), (5, 3)])
@pytest.mark.parametrize("trans", [False, True])
def test_syrk_thread_invariance(bins, op, n, k, trans):
    """Full / beta != 0 must give identical output at 1, 7, 33, 256 threads,
    all matching the oracle."""
    alpha, beta = 1.5, 0.3
    fill = FULL
    sh = _shapes(n, k, trans)
    A = RNG.random(sh).astype(np.float32)
    B = RNG.random(sh).astype(np.float32) if op == "syr2k" else None
    C = RNG.random((n, n)).astype(np.float32)
    C0 = C.copy()

    if op == "syrk":
        expected = _oracle_syrk(alpha, A, beta, C0, trans).astype(np.float32)
    else:
        expected = _oracle_syr2k(alpha, A, B, beta, C0, trans).astype(np.float32)

    outs = []
    for threads in (1, 7, 33, 256):
        r = _run(bins["syrk"], op, threads, n, k, fill, trans, False,
                 alpha, beta, A, B, C.copy())
        assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), \
            f"threads={threads} mismatch vs oracle"
        outs.append(r)
    for r in outs[1:]:
        assert np.array_equal(outs[0], r), "thread-count non-invariance"
