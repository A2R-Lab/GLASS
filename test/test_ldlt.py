"""LDLᵀ GLASS tests — factor + solve a symmetric (possibly indefinite) matrix.

Oracle: split the factored buffer into unit-L + diagonal-D and assert
L @ diag(D) @ L.T ≈ A; the solve asserts x ≈ np.linalg.solve(A, b) and the
residual A@x - b ≈ 0. Covered for SPD and genuinely indefinite-but-nonsingular
symmetric matrices (mixed-sign D — the differentiator vs Cholesky), across a
thread sweep to pin thread-count invariance.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(7)

ATOL = 1e-3
RTOL = 1e-3

NS = [1, 2, 3, 4, 6, 8]
THREADS = [1, 7, 33, 256]


def make_spd(n):
    A = RNG.random((n, n)).astype(np.float32)
    return (A @ A.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)


def make_indefinite(n):
    """Symmetric, nonsingular, but indefinite (mixed-sign eigenvalues → mixed-sign D).

    Built from a random orthonormal basis with a deliberately sign-mixed
    spectrum bounded away from zero, so the non-pivoted factorization has
    nonzero pivots yet D is genuinely indefinite.
    """
    if n == 1:
        return np.array([[-2.0]], dtype=np.float32)
    Q, _ = np.linalg.qr(RNG.standard_normal((n, n)))
    # eigenvalues: alternate sign, magnitudes in [1, n]
    mags = np.linspace(1.0, float(n), n)
    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    # guarantee at least one of each sign for n >= 2
    signs[0], signs[1] = 1.0, -1.0
    D = np.diag(signs * mags)
    A = (Q @ D @ Q.T).astype(np.float32)
    return ((A + A.T) / 2).astype(np.float32)


# ─── runner ──────────────────────────────────────────────────────────────────

def _run(binary, op, n, threads, mats):
    tmp = []
    try:
        for arr in mats:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            np.asfortranarray(arr).astype(np.float32).ravel(order="F").tofile(f)
            f.close()
            tmp.append(f.name)
        cmd = [str(binary), op, str(n), str(threads)] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"binary failed:\n{r.stderr}")
        lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
        return [np.fromstring(l, sep=" ").astype(np.float32) for l in lines]
    finally:
        for f in tmp:
            os.unlink(f)


def _split_LD(buf, n):
    """Factored column-major n*n buffer → (unit-L, D-vector)."""
    M = buf.reshape(n, n, order="F")
    L = np.tril(M, -1) + np.eye(n)
    D = np.diag(M).copy()
    return L, D


@pytest.fixture(scope="session")
def ldlt_bin(bins):
    return bins["ldlt"]


# ─── factor ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["spd", "indef"])
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("threads", THREADS)
def test_ldlt_factor(ldlt_bin, kind, n, threads):
    A = make_spd(n) if kind == "spd" else make_indefinite(n)
    out = _run(ldlt_bin, "ldlt", n, threads, [A])[0]
    L, D = _split_LD(out, n)
    recon = L @ np.diag(D) @ L.T
    assert np.allclose(recon, A, rtol=RTOL, atol=ATOL), (
        f"L@D@L.T mismatch (kind={kind}, n={n}, threads={threads})\n"
        f"D={D}")
    if kind == "indef" and n >= 2:
        # the differentiator: genuinely mixed-sign pivots
        assert (D > 0).any() and (D < 0).any(), f"expected mixed-sign D, got {D}"


@pytest.mark.parametrize("kind", ["spd", "indef"])
@pytest.mark.parametrize("n", NS)
def test_ldlt_factor_thread_invariant(ldlt_bin, kind, n):
    A = make_spd(n) if kind == "spd" else make_indefinite(n)
    ref = _run(ldlt_bin, "ldlt", n, THREADS[0], [A])[0]
    for t in THREADS[1:]:
        cur = _run(ldlt_bin, "ldlt", n, t, [A])[0]
        assert np.allclose(cur, ref, rtol=RTOL, atol=ATOL), (
            f"thread-count non-invariant (kind={kind}, n={n}, threads={t} vs {THREADS[0]})")


# ─── solve ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["spd", "indef"])
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("threads", THREADS)
def test_ldlt_solve(ldlt_bin, kind, n, threads):
    A = make_spd(n) if kind == "spd" else make_indefinite(n)
    b = RNG.random(n).astype(np.float32)
    x = _run(ldlt_bin, "ldlt_solve", n, threads, [A, b])[0]
    x_ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    assert np.allclose(x, x_ref, rtol=RTOL, atol=ATOL), (
        f"x mismatch (kind={kind}, n={n}, threads={threads})\n"
        f"got {x}\nref {x_ref}")
    resid = A.astype(np.float64) @ x.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(resid, 0.0, atol=ATOL), f"residual not zero: {resid}"


# ─── documented zero-pivot limitation (pins the motivation for pivoting) ──────

def test_ldlt_zero_pivot_limitation(ldlt_bin):
    """Non-pivoted LDLᵀ on a zero-(1,1)-block saddle matrix is EXPECTED to fail.

    [[0, 1],[1, 0]] is symmetric and nonsingular (det = -1) but its first pivot
    D_0 = A_00 = 0, so the non-pivoted factor divides by zero and the solve
    blows up. This is the documented Limitation that motivates the (frozen,
    not-yet-implemented) Bunch–Kaufman pivot path. Expected, non-strict.
    """
    A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    x = _run(ldlt_bin, "ldlt_solve", 2, 64, [A, b])[0]
    bad = (not np.all(np.isfinite(x))) or (
        not np.allclose(A.astype(np.float64) @ x.astype(np.float64),
                        b.astype(np.float64), atol=ATOL))
    # the true solution is x = [1, 1]; the non-pivoted path should NOT recover it
    assert bad, (
        f"non-pivoted LDLᵀ unexpectedly handled a zero pivot: x={x} "
        "(if this ever passes cleanly, pivoting may have landed — update the test)")
