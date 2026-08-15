"""LU with partial pivoting (getrf / getrs / gesv + laswp) vs SciPy/NumPy references.

getrf follows the LAPACK/SciPy `lu_factor` convention exactly: A overwritten
with L\\U (unit-lower L below the diagonal, U on/above), piv[k] = row swapped
with row k at step k (0-based ipiv). The net covers: LU reconstruction
(P@A == L@U and drop-in use of our (LU,piv) inside scipy.linalg.lu_solve),
gesv solve correctness on general NON-symmetric matrices including a matrix
that REQUIRES pivoting (zero leading pivot — the case no-pivot LU fails),
multi-RHS, TRANSPOSE getrs vs scipy.linalg.lu_solve(trans=1), byte-identical
thread-count invariance over the full THREAD_SWEEP at two shapes, CHECK on a
singular matrix, and the laswp forward/reverse round-trip.
"""

import numpy as np
import pytest
import scipy.linalg
from conftest import run_op, THREAD_SWEEP, make_general

RTOL = 1e-3
ATOL = 1e-3

NS = [1, 3, 4, 7, 16]


def _col(M):
    return np.asfortranarray(M).ravel(order='F')


# ─── input kinds (≥2 kinds; every maker deterministic per (kind, n)) ───────────

def _make_A(kind, n):
    """'general': mixed-sign random; 'scaled': same but 50× magnitudes;
    'needs_pivot': invertible with a ZERO leading pivot — the matrix no-pivot
    LU fails on (partial pivoting must swap the large row up)."""
    seed = 9000 + 17 * n
    if kind == "general":
        return make_general(n, seed=seed)
    if kind == "scaled":
        return make_general(n, seed=seed + 1, scale=50.0)
    if kind == "needs_pivot":
        A = make_general(n, seed=seed + 2)
        A[0, 0] = 0.0            # zero leading pivot: unpivoted LU divides by 0
        A[n - 1, 0] = 3.0        # large entry below → pivot row must swap up
        return A.astype(np.float32)
    raise ValueError(kind)


def _make_singular(n):
    """Exactly singular: one all-zero column (stays exactly zero through the
    trailing updates, so the pivot at that step is exactly 0 → CHECK fires)."""
    A = make_general(n, seed=77)
    A[:, n // 2] = 0.0
    return A.astype(np.float32)


def _apply_swaps(M, piv, reverse=False):
    """Apply LAPACK ipiv row interchanges sequentially (reference for laswp/P)."""
    out = M.copy()
    ks = range(len(piv) - 1, -1, -1) if reverse else range(len(piv))
    for k in ks:
        p = int(piv[k])
        out[[k, p], :] = out[[p, k], :]
    return out


# ─── getrf: LU reconstruction vs scipy.linalg.lu_factor convention ─────────────

# needs_pivot requires n >= 2: a 1x1 matrix with a zero leading pivot is
# singular (no row to swap to), so that combination is never generated —
# excluded at parametrization rather than skipped, keeping the suite skip-free.
@pytest.mark.parametrize("n,kind",
                         [(n, k) for n in NS
                          for k in ("general", "scaled", "needs_pivot")
                          if not (k == "needs_pivot" and n < 2)])
def test_getrf_reconstruction(bins, n, kind):
    """P@A == L@U with valid 0-based LAPACK ipiv, and our (LU, piv) is drop-in
    for scipy.linalg.lu_solve — the lu_factor-compatibility contract."""
    A = _make_A(kind, n)
    LU_flat, piv_f = run_op(bins["getrf"], "getrf", "simple",
                            args=[128, n], inputs=[_col(A)])
    LU = LU_flat.reshape(n, n, order='F')
    piv = piv_f.astype(np.int64)
    # Valid LAPACK ipiv: piv[k] in [k, n).
    assert np.all(piv >= np.arange(n)) and np.all(piv < n), f"bad ipiv {piv}"
    if kind == "needs_pivot":
        assert piv[0] != 0, "zero leading pivot was not swapped away"
    L = np.tril(LU.astype(np.float64), -1) + np.eye(n)
    U = np.triu(LU.astype(np.float64))
    PA = _apply_swaps(A.astype(np.float64), piv)
    assert np.allclose(L @ U, PA, rtol=1e-2, atol=1e-2 * max(1.0, np.abs(A).max())), \
        f"L@U != P@A (n={n}, kind={kind})"
    # Drop-in for scipy.linalg.lu_solve((lu, piv), b): relative residual small.
    b = np.arange(1, n + 1, dtype=np.float64)
    x = scipy.linalg.lu_solve((LU.astype(np.float64), piv), b)
    res = np.linalg.norm(A.astype(np.float64) @ x - b)
    assert res <= 1e-3 * (np.linalg.norm(b) + np.linalg.norm(A) * np.linalg.norm(x))


# ─── gesv: general non-symmetric solve (incl. the pivot-REQUIRING case) ────────

# same n >= 2 exclusion as test_getrf_reconstruction (vacuous at n=1)
@pytest.mark.parametrize("n,kind",
                         [(n, k) for n in NS
                          for k in ("general", "needs_pivot")
                          if not (k == "needs_pivot" and n < 2)])
def test_gesv(bins, n, kind):
    """gesv == np.linalg.solve on general non-symmetric A, single RHS —
    including a zero leading pivot, where no-pivot LU produces garbage."""
    A = _make_A(kind, n)
    b = make_general(n, 1, seed=31 + n)
    X_flat, _LU = run_op(bins["getrf"], "gesv", "simple",
                         args=[128, n, 1], inputs=[_col(A), _col(b)])
    x = X_flat.astype(np.float64)
    x_ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64)).ravel()
    assert np.allclose(x, x_ref, rtol=1e-2, atol=1e-2), f"n={n} kind={kind}"
    res = np.linalg.norm(A.astype(np.float64) @ x - b.ravel())
    assert res <= 1e-3 * (np.linalg.norm(b) + np.linalg.norm(A) * np.linalg.norm(x))


@pytest.mark.parametrize("n", [4, 7, 16])
def test_gesv_multirhs(bins, n):
    """Multi-RHS gesv (nrhs=3): every column solved, X == np.linalg.solve."""
    nrhs = 3
    A = _make_A("general", n)
    B = make_general(n, nrhs, seed=57 + n)
    X_flat, _LU = run_op(bins["getrf"], "gesv", "simple",
                         args=[128, n, nrhs], inputs=[_col(A), _col(B)])
    X = X_flat.reshape(n, nrhs, order='F').astype(np.float64)
    X_ref = np.linalg.solve(A.astype(np.float64), B.astype(np.float64))
    assert np.allclose(X, X_ref, rtol=1e-2, atol=1e-2)
    res = A.astype(np.float64) @ X - B.astype(np.float64)
    assert np.linalg.norm(res) <= 1e-3 * (np.linalg.norm(B) + np.linalg.norm(A) * np.linalg.norm(X))


# ─── getrs: both transpose modes vs scipy.linalg.lu_solve on scipy's OWN factor ─

@pytest.mark.parametrize("n,nrhs", [(3, 1), (4, 3), (7, 3)])
@pytest.mark.parametrize("transpose", [0, 1])
def test_getrs(bins, n, nrhs, transpose):
    """getrs consumes a scipy.linalg.lu_factor (lu, piv) pair directly —
    independent of our getrf — and must match lu_solve(trans=0/1)."""
    A = make_general(n, seed=400 + n)
    B = make_general(n, nrhs, seed=500 + n)
    lu, piv = scipy.linalg.lu_factor(A)          # float32 in → sgetrf out
    X_flat = run_op(bins["getrf"], "getrs", "simple",
                    args=[128, n, nrhs, transpose],
                    inputs=[_col(lu), piv.astype(np.float32), _col(B)])
    X = X_flat.reshape(n, nrhs, order='F')
    X_ref = scipy.linalg.lu_solve((lu, piv), B, trans=transpose)
    assert np.allclose(X, X_ref, rtol=1e-2, atol=1e-3), f"trans={transpose}"
    opA = A.T if transpose else A
    res = opA.astype(np.float64) @ X.astype(np.float64) - B.astype(np.float64)
    assert np.allclose(res, 0, atol=1e-2)


# ─── compile-time overloads (getrf<T,N> / getrs<T,N,NRHS> / gesv<T,N,NRHS>) ────

@pytest.mark.parametrize("split", [0, 1])   # 0: gesv CT; 1: getrf CT + getrs CT
def test_gesv_ct(bins, split):
    n, nrhs = 4, 3
    A = _make_A("needs_pivot", n)
    B = make_general(n, nrhs, seed=61)
    X_flat = run_op(bins["getrf"], "gesv_ct", "simple",
                    args=[128, split], inputs=[_col(A), _col(B)])
    X = X_flat.reshape(n, nrhs, order='F').astype(np.float64)
    X_ref = np.linalg.solve(A.astype(np.float64), B.astype(np.float64))
    assert np.allclose(X, X_ref, rtol=1e-2, atol=1e-3)


def test_gesv_ct_split_matches_fused(bins):
    """getrf<T,N> + getrs<T,N,NRHS> must be byte-identical to gesv<T,N,NRHS>."""
    A = _make_A("general", 4)
    B = make_general(4, 3, seed=62)
    r0 = run_op(bins["getrf"], "gesv_ct", "simple", args=[128, 0], inputs=[_col(A), _col(B)])
    r1 = run_op(bins["getrf"], "gesv_ct", "simple", args=[128, 1], inputs=[_col(A), _col(B)])
    assert np.array_equal(r0, r1)


# ─── thread-count invariance: byte-identical over the FULL sweep, 2 shapes ─────

@pytest.mark.parametrize("kind,n", [("general", 7), ("needs_pivot", 4)])
def test_getrf_thread_invariance(bins, kind, n):
    """getrf output (LU AND piv) byte-identical at every block size in
    THREAD_SWEEP (1..256, incl. partial/odd/non-warp counts)."""
    A = _make_A(kind, n)
    ref = None
    for th in THREAD_SWEEP:
        LU, piv = run_op(bins["getrf"], "getrf", "simple",
                         args=[th, n], inputs=[_col(A)])
        if ref is None:
            ref = (LU, piv)
        else:
            assert np.array_equal(LU, ref[0]), f"LU non-invariant at th={th} (kind={kind})"
            assert np.array_equal(piv, ref[1]), f"piv non-invariant at th={th} (kind={kind})"


@pytest.mark.parametrize("n,nrhs", [(7, 3), (16, 1)])
def test_gesv_thread_invariance(bins, n, nrhs):
    """Composed gesv (X and LU) byte-identical across the full sweep."""
    A = _make_A("general", n)
    B = make_general(n, nrhs, seed=71 + n)
    ref = None
    for th in THREAD_SWEEP:
        X, LU = run_op(bins["getrf"], "gesv", "simple",
                       args=[th, n, nrhs], inputs=[_col(A), _col(B)])
        if ref is None:
            ref = (X, LU)
        else:
            assert np.array_equal(X, ref[0]), f"X non-invariant at th={th}"
            assert np.array_equal(LU, ref[1]), f"LU non-invariant at th={th}"


# ─── CHECK flag: singular → fail=1, well-conditioned → fail=0 ──────────────────

def test_getrf_check_singular(bins):
    """An exactly-zero pivot column sets *s_fail = 1 (the divide is skipped)."""
    n = 4
    A = _make_singular(n)
    for th in (1, 33, 256):
        _LU, _piv, fail = run_op(bins["getrf"], "getrf_check", "simple",
                                 args=[th, n], inputs=[_col(A)])
        assert int(fail[0]) == 1, f"singular matrix not flagged (th={th})"


def test_getrf_check_ok(bins):
    """A well-conditioned general matrix leaves *s_fail = 0 (poisoned to -1
    before launch, so 0 proves getrf wrote it)."""
    n = 6
    A = _make_A("general", n)
    _LU, piv, fail = run_op(bins["getrf"], "getrf_check", "simple",
                            args=[128, n], inputs=[_col(A)])
    assert int(fail[0]) == 0
    # Checked path must still produce a correct factorization.
    LU = _LU.reshape(n, n, order='F').astype(np.float64)
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    PA = _apply_swaps(A.astype(np.float64), piv.astype(np.int64))
    assert np.allclose(L @ U, PA, rtol=1e-2, atol=1e-2)


# ─── laswp: forward matches the sequential reference; reverse undoes it ────────

def _valid_ipiv(n, seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.integers(k, n) for k in range(n)], dtype=np.int64)


def _int_valued(shape, seed):
    """Integer-valued float32 draw: laswp only MOVES values, so exact equality
    is the right check — integer values survive the binary's %.8g print
    round-trip exactly (arbitrary float32 can lose its 9th significant digit)."""
    rng = np.random.default_rng(seed)
    return rng.integers(-999, 999, size=shape).astype(np.float32)


@pytest.mark.parametrize("th", [1, 33, 256])
def test_laswp_vector_roundtrip(bins, th):
    n = 8
    piv = _valid_ipiv(n, seed=5)
    x = _int_valued(n, seed=6)
    fwd = run_op(bins["getrf"], "laswp_vec", "simple",
                 args=[th, n, 0], inputs=[piv.astype(np.float32), x])
    ref = _apply_swaps(x.reshape(-1, 1), piv).ravel()
    assert np.array_equal(fwd, ref.astype(np.float32))
    # REVERSE application is the inverse permutation: round-trips to x.
    back = run_op(bins["getrf"], "laswp_vec", "simple",
                  args=[th, n, 1], inputs=[piv.astype(np.float32), fwd])
    assert np.array_equal(back, x.astype(np.float32))


@pytest.mark.parametrize("th", [1, 33, 256])
def test_laswp_matrix(bins, th):
    n = 6
    piv = _valid_ipiv(n, seed=15)
    A = _int_valued((n, n), seed=16)
    out = run_op(bins["getrf"], "laswp_mat", "simple",
                 args=[th, n, 0], inputs=[piv.astype(np.float32), _col(A)])
    ref = _apply_swaps(A, piv)
    assert np.array_equal(out.reshape(n, n, order='F'), ref.astype(np.float32))
    back = run_op(bins["getrf"], "laswp_mat", "simple",
                  args=[th, n, 1], inputs=[piv.astype(np.float32), out])
    assert np.array_equal(back.reshape(n, n, order='F'), A.astype(np.float32))
