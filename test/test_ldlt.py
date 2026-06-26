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

from conftest import make_spd  # shared; pass rng=RNG for varied draws

RNG = np.random.default_rng(7)

ATOL = 1e-3
RTOL = 1e-3

NS = [1, 2, 3, 4, 6, 8]
THREADS = [1, 7, 33, 256]


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

def _run(binary, op, n, threads, mats, pivot=0):
    tmp = []
    try:
        for arr in mats:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            np.asfortranarray(arr).astype(np.float32).ravel(order="F").tofile(f)
            f.close()
            tmp.append(f.name)
        cmd = [str(binary), op, str(n), str(threads), str(pivot)] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"binary failed:\n{r.stderr}")
        lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
        return [np.fromstring(l, sep=" ").astype(np.float32) for l in lines]
    finally:
        for f in tmp:
            os.unlink(f)


def _perm_from_piv(piv):
    """Rebuild the permutation vector applied by ldlt: forward sweep of swaps.

    Factor applies, at step k, swap(k, piv[k]) to the working matrix, so
    P A Pᵀ = L D Lᵀ where P is that ordered product. Applying P to the identity
    index vector reproduces the row permutation: perm[i] = (P @ arange(n))[i].
    """
    piv = [int(round(x)) for x in piv]
    perm = list(range(len(piv)))
    for k, p in enumerate(piv):
        perm[k], perm[p] = perm[p], perm[k]
    return np.array(perm, dtype=int)


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
    A = make_spd(n, rng=RNG) if kind == "spd" else make_indefinite(n)
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
    A = make_spd(n, rng=RNG) if kind == "spd" else make_indefinite(n)
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
    A = make_spd(n, rng=RNG) if kind == "spd" else make_indefinite(n)
    b = RNG.random(n).astype(np.float32)
    x = _run(ldlt_bin, "ldlt_solve", n, threads, [A, b])[0]
    x_ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    assert np.allclose(x, x_ref, rtol=RTOL, atol=ATOL), (
        f"x mismatch (kind={kind}, n={n}, threads={threads})\n"
        f"got {x}\nref {x_ref}")
    resid = A.astype(np.float64) @ x.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(resid, 0.0, atol=ATOL), f"residual not zero: {resid}"


# ─── pivoted (symmetric 1×1) factor + solve ──────────────────────────────────

def make_indef_small_leading(n):
    """Indefinite symmetric with a NEAR-ZERO leading diagonal block.

    The non-pivoted path divides by the tiny leading pivot and blows up; the
    symmetric 1×1 pivot path swaps a large-magnitude diagonal to the front and
    succeeds. Built by taking a well-conditioned indefinite matrix and shrinking
    the top-left diagonal entries toward zero (kept nonzero so the matrix stays
    nonsingular but the leading pivot is catastrophic without pivoting).
    """
    A = make_indefinite(n).astype(np.float64)
    # drive the first (and for n>=4 the second) diagonal entries to ~1e-6
    A[0, 0] = 1e-6
    if n >= 4:
        A[1, 1] = -1e-6
    return ((A + A.T) / 2).astype(np.float32)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("threads", THREADS)
def test_ldlt_pivot_factor_reconstruction(ldlt_bin, n, threads):
    """Pivoted factor: L @ diag(D) @ L.T == P A Pᵀ using the recorded piv."""
    A = make_indef_small_leading(n)
    out = _run(ldlt_bin, "ldlt", n, threads, [A], pivot=1)
    factor, piv = out[0], out[1]
    L, D = _split_LD(factor, n)
    perm = _perm_from_piv(piv)
    recon = L @ np.diag(D) @ L.T
    PAPt = A[np.ix_(perm, perm)]
    assert np.allclose(recon, PAPt, rtol=RTOL, atol=ATOL), (
        f"L@D@L.T != P A Pᵀ (n={n}, threads={threads})\n"
        f"piv={piv} perm={perm}\nD={D}\nrecon={recon}\nPAPt={PAPt}")


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("threads", THREADS)
def test_ldlt_pivot_solve(ldlt_bin, n, threads):
    """Pivoted solve matches np.linalg.solve even with a near-zero leading pivot
    (the case the non-pivoted path cannot handle)."""
    A = make_indef_small_leading(n)
    b = RNG.random(n).astype(np.float32)
    x = _run(ldlt_bin, "ldlt_solve", n, threads, [A, b], pivot=1)[0]
    x_ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    assert np.allclose(x, x_ref, rtol=RTOL, atol=ATOL), (
        f"pivoted x mismatch (n={n}, threads={threads})\ngot {x}\nref {x_ref}")
    resid = A.astype(np.float64) @ x.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(resid, 0.0, atol=ATOL), f"residual not zero: {resid}"


@pytest.mark.parametrize("kind", ["spd", "indef"])
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("threads", THREADS)
def test_ldlt_pivot_solve_general(ldlt_bin, kind, n, threads):
    """Pivoted solve also works on the ordinary SPD / indefinite matrices."""
    A = make_spd(n, rng=RNG) if kind == "spd" else make_indefinite(n)
    b = RNG.random(n).astype(np.float32)
    x = _run(ldlt_bin, "ldlt_solve", n, threads, [A, b], pivot=1)[0]
    x_ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    assert np.allclose(x, x_ref, rtol=RTOL, atol=ATOL), (
        f"pivoted x mismatch (kind={kind}, n={n}, threads={threads})\n"
        f"got {x}\nref {x_ref}")


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
def test_ldlt_pivot_thread_invariant(ldlt_bin, n):
    """Pivoted factor (and recorded piv) identical across the thread sweep."""
    A = make_indef_small_leading(n)
    ref = _run(ldlt_bin, "ldlt", n, THREADS[0], [A], pivot=1)
    for t in THREADS[1:]:
        cur = _run(ldlt_bin, "ldlt", n, t, [A], pivot=1)
        assert np.allclose(cur[0], ref[0], rtol=RTOL, atol=ATOL), (
            f"pivoted factor thread non-invariant (n={n}, threads={t} vs {THREADS[0]})")
        assert np.array_equal(
            [int(round(v)) for v in cur[1]], [int(round(v)) for v in ref[1]]), (
            f"piv thread non-invariant (n={n}, threads={t}): {cur[1]} vs {ref[1]}")


def test_ldlt_pivot_leading_zero_fails_unpivoted(ldlt_bin):
    """Sanity: the near-zero-leading-pivot matrix DOES break the non-pivoted path
    (so the pivoted success above is meaningful, not a trivially-easy matrix)."""
    n = 4
    A = make_indef_small_leading(n)
    b = RNG.random(n).astype(np.float32)
    x = _run(ldlt_bin, "ldlt_solve", n, 64, [A, b], pivot=0)[0]
    x_ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    failed = (not np.all(np.isfinite(x))) or (
        not np.allclose(x, x_ref, rtol=RTOL, atol=ATOL))
    assert failed, (
        f"non-pivoted path unexpectedly handled the near-zero leading pivot: x={x}")


# ─── documented zero-pivot limitation (pins the motivation for pivoting) ──────

def test_ldlt_zero_pivot_limitation(ldlt_bin):
    """Non-pivoted LDLᵀ on a zero-(1,1)-block saddle matrix is EXPECTED to fail.

    [[0, 1],[1, 0]] is symmetric and nonsingular (det = -1) but its first pivot
    D_0 = A_00 = 0, so the non-pivoted factor divides by zero and the solve
    blows up. This is the documented Limitation that motivates pivoting. Expected,
    non-strict.
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


def test_ldlt_zero_diag_block_needs_2x2(ldlt_bin):
    """[[0,1],[1,0]] still fails EVEN WITH symmetric 1×1 pivoting — it has a
    structurally-zero remaining DIAGONAL (both D_eff are 0 at step 0), so no 1×1
    pivot helps; it requires a 2×2 (Bunch–Kaufman) pivot, which is NOT
    implemented. Documented known limitation; expected, non-strict.
    """
    A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    x = _run(ldlt_bin, "ldlt_solve", 2, 64, [A, b], pivot=1)[0]
    bad = (not np.all(np.isfinite(x))) or (
        not np.allclose(A.astype(np.float64) @ x.astype(np.float64),
                        b.astype(np.float64), atol=ATOL))
    assert bad, (
        f"1×1-pivoted LDLᵀ unexpectedly handled a zero diagonal BLOCK: x={x} "
        "(would require 2×2 Bunch–Kaufman — if this passes, 2×2 may have landed)")


# ─── warp forms (non-pivoted; one 32-lane warp) ───────────────────────────────

WARP = 32
WARP_NS = [3, 4, 5, 6, 7, 8]   # harness instantiates warp::ldlt for these N


@pytest.mark.parametrize("kind", ["spd", "indef"])
@pytest.mark.parametrize("n", WARP_NS)
def test_ldlt_warp_factor(ldlt_bin, kind, n):
    """warp::ldlt non-pivoted factor reconstructs A = L·diag(D)·Lᵀ and matches the
    block factor (run at one warp)."""
    A = make_spd(n, rng=RNG) if kind == "spd" else make_indefinite(n)
    buf = _run(ldlt_bin, "ldlt_warp", n, WARP, [A])[0]
    L, D = _split_LD(buf, n)
    recon = (L @ np.diag(D) @ L.T)
    assert np.allclose(recon, A, rtol=RTOL, atol=ATOL), f"{kind} n={n}: A != L D Lᵀ"
    # cross-check vs the block factor (same algorithm, different scope)
    block = _run(ldlt_bin, "ldlt", n, 64, [A])[0]
    assert np.allclose(buf, block, rtol=1e-2, atol=1e-3), f"{kind} n={n}: warp != block factor"


@pytest.mark.parametrize("kind", ["spd", "indef"])
@pytest.mark.parametrize("n", WARP_NS)
def test_ldlt_warp_solve(ldlt_bin, kind, n):
    """warp::ldlt + warp::ldlt_solve recovers x = A⁻¹ b (one warp)."""
    A = make_spd(n, rng=RNG) if kind == "spd" else make_indefinite(n)
    b = RNG.standard_normal(n).astype(np.float32)
    x = _run(ldlt_bin, "ldlt_solve_warp", n, WARP, [A, b])[0]
    expected = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
    assert np.allclose(x, expected, rtol=RTOL, atol=ATOL), f"{kind} n={n}: solve mismatch"
    assert np.allclose(A.astype(np.float64) @ x.astype(np.float64), b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", WARP_NS)
def test_ldlt_warp_inertia(ldlt_bin, n):
    """CHECK path: inertia {n_pos, n_neg, n_zero} matches the eigenvalue signs of A;
    s_fail stays 0 for a nonsingular indefinite matrix."""
    A = make_indefinite(n)
    lines = _run(ldlt_bin, "ldlt_warp_check", n, WARP, [A])
    s_fail, n_pos, n_neg, n_zero = [int(round(v)) for v in lines[1]]
    ev = np.linalg.eigvalsh(A.astype(np.float64))
    assert (n_pos, n_neg, n_zero) == (int((ev > 0).sum()), int((ev < 0).sum()), 0), \
        f"n={n}: inertia {(n_pos, n_neg, n_zero)} != eig signs"
    assert s_fail == 0, f"n={n}: nonsingular indefinite flagged s_fail=1"
