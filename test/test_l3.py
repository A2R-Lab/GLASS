"""L3 GLASS function tests — compare GPU results to NumPy/SciPy reference."""

import numpy as np
import pytest
from conftest import run_op

# bin_l3_nvidia fixture is defined in conftest.py — re-import for pytest discovery.
from conftest import bin_l3_nvidia  # noqa: F401

RNG = np.random.default_rng(42)

ATOL = 1e-3
RTOL = 1e-3

CG_SIMPLE = ["cg", "simple"]


def make_spd(n):
    """Generate a random n x n symmetric positive definite matrix."""
    A = RNG.random((n, n)).astype(np.float32)
    return (A @ A.T + n * np.eye(n, dtype=np.float32))


def make_lower_triangular(n):
    """Generate a random n x n lower triangular matrix with positive diagonal."""
    L = np.tril(RNG.random((n, n)).astype(np.float32))
    np.fill_diagonal(L, np.abs(L.diagonal()) + 0.5)
    return L


# ─── gemm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemm(bins, m, n, k, version):
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm", version,
                    args=[m, n, k, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n", [(4, 6), (8, 8)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemm_t(bins, m, n, version):
    # GLASS gemm_t computes C(m×n) = alpha * A(m×n) * B(n×n)^T + beta * C(m×n)
    # B must be square n×n (this is how the GLASS coop-groups implementation works).
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, n)).astype(np.float32)
    C = RNG.random((m, n)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_t", version,
                    args=[m, n, n, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B.T + beta * C0).astype(np.float32)
    mat = result.reshape(m, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm row-major ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6)])
def test_gemm_rowmajor(bins, m, n, k):
    """All matrices row-major (C-contiguous): C = alpha*A*B + beta*C."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)   # C-order = row-major
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_rowmajor", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[A.ravel(), B.ravel(), C.ravel()])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    # result is row-major flat: reshape with C order
    mat = result.reshape(m, k)
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8)])
def test_gemm_ex(bins, m, n, k):
    """gemm_ex mixed layout: row-major A, col-major B, row-major C."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)       # row-major
    B_col = RNG.random((n, k)).astype(np.float32)   # will be passed col-major
    C = RNG.random((m, k)).astype(np.float32)       # row-major
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_ex", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[A.ravel(),
                            np.asfortranarray(B_col).ravel(order='F'),  # col-major flat
                            C.ravel()])
    expected = (alpha * A @ B_col + beta * C0).astype(np.float32)
    mat = result.reshape(m, k)
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_tiled ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6), (6, 10, 7), (4, 3, 4)])
def test_gemm_tiled(bins, m, n, k):
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_tiled", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_strided ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
@pytest.mark.parametrize("op,m,n,k,a_rs,b_rs", [
    ("gemm_strided_6x6x6_6_6", 6, 6, 6, 6, 6),
    ("gemm_strided_6x6x6_8_8", 6, 6, 6, 8, 8),
    ("gemm_strided_4x4x4_4_4", 4, 4, 4, 4, 4),
    ("gemm_strided_4x4x4_6_6", 4, 4, 4, 6, 6),
])
def test_gemm_strided(bins, op, m, n, k, a_rs, b_rs, alpha, beta):
    # A[i][j] = A_flat[i + j*a_rs]; B[j][l] = B_flat[j + l*b_rs]; C standard col-major.
    A_storage = np.zeros((a_rs, n), dtype=np.float32)
    A_storage[:m, :] = RNG.random((m, n)).astype(np.float32)
    B_storage = np.zeros((b_rs, k), dtype=np.float32)
    B_storage[:n, :] = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    A_flat = np.asfortranarray(A_storage).ravel(order='F')
    B_flat = np.asfortranarray(B_storage).ravel(order='F')
    C_flat = np.asfortranarray(C).ravel(order='F')
    # gemm_strided dispatch uses args=[alpha, beta] directly (no m/n/k positional args)
    result = run_op(bins["l3"], op, "simple",
                    args=[alpha, beta], inputs=[A_flat, B_flat, C_flat])
    expected = (alpha * A_storage[:m, :] @ B_storage[:n, :] + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── packed_gemm ──────────────────────────────────────────────────────────────

def _make_packed_vec(size, case):
    if case == "positive": return RNG.random(size).astype(np.float32)
    if case == "negative": return -RNG.random(size).astype(np.float32)
    if case == "mixed":    return (RNG.random(size) - 0.5).astype(np.float32)
    if case == "zero":     return np.zeros(size, dtype=np.float32)
    if case == "tiny":     return (RNG.random(size) * 1e-6).astype(np.float32)
    raise ValueError(case)


@pytest.mark.parametrize("case", ["positive", "negative", "mixed", "zero", "tiny"])
@pytest.mark.parametrize("k", [16, 32, 48, 64])
def test_packed_gemm(bins, k, case):
    # glass::gemm<float,4,4,K>: C(4×K) = alpha * A(4×4) * B(4×K) + beta * C(4×K), col-major
    m, n = 4, 4
    alpha, beta = 1.5, 0.3
    A = _make_packed_vec(m * n, case).reshape(m, n)
    B = _make_packed_vec(n * k, case).reshape(n, k)
    C = _make_packed_vec(m * k, case).reshape(m, k)
    C0 = C.copy()
    result = run_op(bins["l3"], f"packed_gemm_4x4x{k}", "simple",
                    args=[alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── inv ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_inv(bins, n, version):
    A = make_spd(n)
    # invertMatrix expects [A | I] layout (n*2n) column-major
    AI = np.hstack([A, np.eye(n, dtype=np.float32)])   # row layout (n x 2n)
    AI_col = np.asfortranarray(AI).ravel(order='F')
    result = run_op(bins["l3"], "inv", version, args=[n], inputs=[AI_col])
    Ainv = result.reshape(n, n, order='F')
    expected = np.linalg.inv(A).astype(np.float32)
    assert np.allclose(Ainv, expected, rtol=1e-2, atol=1e-3)


def _aug(M, d):
    # [M | I] column-major augmented buffer for invertMatrix
    return np.asfortranarray(np.hstack([M, np.eye(d, dtype=np.float32)])).ravel(order='F')


# ─── inv_pivot (robust partial-pivoting) ──────────────────────────────────────

def _run_inv_pivot(bins, A, n, threads):
    """Run the partial-pivoting inverse on A (n x n) with a given block size."""
    AI = np.hstack([A, np.eye(n, dtype=np.float32)])      # row layout (n x 2n)
    AI_col = np.asfortranarray(AI).ravel(order='F')
    result = run_op(bins["l3"], "inv_pivot", "simple",
                    args=[threads, n], inputs=[AI_col])
    return result.reshape(n, n, order='F')


def _near_singular_leading(n):
    """Invertible matrix whose UNPIVOTED Gauss-Jordan hits a tiny/zero leading
    pivot: a tiny A[0,0] with a large entry lower in column 0 (the partial-pivot
    path swaps it up; the plain path divides by ~0)."""
    A = make_spd(n)
    A[0, 0] = 1e-7                 # tiny leading pivot
    A[n - 1, 0] = 5.0             # large later entry in column 0 → must pivot up
    A[0, n - 1] = 5.0            # keep it reasonably conditioned / nonsymmetric
    return A.astype(np.float32)


def _zero_diagonal_perm(n):
    """Invertible matrix with a literal ZERO on the (0,0) diagonal — the plain
    path divides by exactly 0; partial pivoting swaps a nonzero row up."""
    A = make_spd(n)
    # Swap rows 0 and 1 of an SPD matrix then zero the (0,0) entry: still
    # invertible, but A[0,0] == 0 breaks the unpivoted divide.
    A[[0, 1], :] = A[[1, 0], :]
    A[0, 0] = 0.0
    return A.astype(np.float32)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
def test_inv_pivot(bins, n):
    """Robust partial-pivoting inverse matches np.linalg.inv on well-conditioned
    SPD matrices, across a thread-count sweep (1, 7, 33, 256) with identical
    output at every block size."""
    A = make_spd(n)
    expected = np.linalg.inv(A).astype(np.float32)
    ref = None
    for threads in (1, 7, 33, 256):
        Ainv = _run_inv_pivot(bins, A, n, threads)
        assert np.allclose(Ainv, expected, rtol=1e-2, atol=1e-3), \
            f"n={n} threads={threads}: mismatch vs np.linalg.inv"
        if ref is None:
            ref = Ainv
        else:
            # Thread-count invariance: byte-for-byte identical across block sizes.
            assert np.array_equal(Ainv, ref), \
                f"n={n} threads={threads}: output differs from threads=1"


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("maker", [_near_singular_leading, _zero_diagonal_perm])
def test_inv_pivot_near_singular(bins, n, maker):
    """Partial pivoting is CORRECT on matrices whose leading pivot is tiny or
    exactly zero (the unpivoted path would divide by ~0 / produce garbage).
    Same thread-count sweep, identical output across block sizes."""
    A = maker(n)
    expected = np.linalg.inv(A).astype(np.float32)
    ref = None
    for threads in (1, 7, 33, 256):
        Ainv = _run_inv_pivot(bins, A, n, threads)
        assert np.allclose(Ainv, expected, rtol=1e-2, atol=1e-3), \
            f"n={n} threads={threads} maker={maker.__name__}: mismatch vs np.linalg.inv"
        if ref is None:
            ref = Ainv
        else:
            assert np.array_equal(Ainv, ref), \
                f"n={n} threads={threads} maker={maker.__name__}: output differs from threads=1"


@pytest.mark.parametrize("dimA,dimB", [(4, 4), (6, 4), (3, 6)])
def test_inv2(bins, dimA, dimB):
    # fused 2-matrix invert: same augmented [A|I] convention, interleaved sweep
    A = make_spd(dimA); B = make_spd(dimB)
    res = run_op(bins["l3"], "inv2", "simple", args=[dimA, dimB, max(dimA, dimB)],
                 inputs=[_aug(A, dimA), _aug(B, dimB)])
    assert np.allclose(res[0].reshape(dimA, dimA, order='F'), np.linalg.inv(A), rtol=1e-2, atol=1e-3)
    assert np.allclose(res[1].reshape(dimB, dimB, order='F'), np.linalg.inv(B), rtol=1e-2, atol=1e-3)


# (12,12,6) mirrors GATO's Schur fused-3 (STATE_SIZE=12, CONTROL_SIZE=6 for indy7)
@pytest.mark.parametrize("dimA,dimB,dimC", [(12, 12, 6), (6, 6, 6), (4, 6, 3)])
def test_inv3(bins, dimA, dimB, dimC):
    A = make_spd(dimA); B = make_spd(dimB); C = make_spd(dimC)
    res = run_op(bins["l3"], "inv3", "simple", args=[dimA, dimB, dimC, max(dimA, dimB, dimC)],
                 inputs=[_aug(A, dimA), _aug(B, dimB), _aug(C, dimC)])
    for M, d, r in [(A, dimA, res[0]), (B, dimB, res[1]), (C, dimC, res[2])]:
        assert np.allclose(r.reshape(d, d, order='F'), np.linalg.inv(M), rtol=1e-2, atol=1e-3)


# ─── chol ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_chol(bins, n, version):
    A = make_spd(n)
    A_col = np.asfortranarray(A).ravel(order='F')
    result = run_op(bins["l3"], "chol", version, args=[n], inputs=[A_col])
    L_gpu = result.reshape(n, n, order='F')
    # Extract lower triangle
    L_gpu = np.tril(L_gpu)
    L_ref = np.linalg.cholesky(A).astype(np.float32)
    assert np.allclose(L_gpu, L_ref, rtol=1e-2, atol=1e-3)


# ─── trsm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6, 8])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_trsm(bins, n, version):
    L = make_lower_triangular(n)
    b = RNG.random(n).astype(np.float32)
    L_col = np.asfortranarray(L).ravel(order='F')
    result = run_op(bins["l3"], "trsm", version, args=[n], inputs=[L_col, b])
    # Verify L @ result == b (residual check avoids scipy dependency)
    residual = L.astype(np.float64) @ result.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(residual, 0, atol=1e-3)


# ─── warp:: (single warp, launched <<<1,32>>>) ────────────────────────────────

def test_gemm_warp_4x4x4(bins):
    # glass::warp::gemm<float,4,4,4>: C = alpha*A@B + beta*C, col-major, one warp.
    m, n, k = 4, 4, 4
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    B = RNG.random((n, k)).astype(np.float32)
    C = RNG.random((m, k)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_warp", "warp",
                    args=[m, n, k, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, k, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


def test_posv_warp_7(bins):
    # Single-warp SPD solve A x = b via warp:: chol + trsm + trsm_transpose (N=7).
    n = 7
    A = make_spd(n)
    b = RNG.random(n).astype(np.float32)
    A_col = np.asfortranarray(A).ravel(order='F')
    result = run_op(bins["l3"], "posv_warp", "warp", args=[n], inputs=[A_col, b])
    residual = A.astype(np.float64) @ result.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(residual, 0, atol=1e-3)


# ─── nvidia::gemm_batched_1d (SIMT, 1D launch) ────────────────────────────────

def _flatten_col(mats):
    """Concatenate a list of 2D arrays into a flat F-order buffer."""
    return np.concatenate([np.asfortranarray(m).ravel(order='F') for m in mats])


def _flatten_row(mats):
    return np.concatenate([np.ascontiguousarray(m).ravel() for m in mats])


@pytest.mark.parametrize("op,m,n,k,batch", [
    ("gemm_batched_1d_4x4x4_b1_col", 4, 4, 4, 1),
    ("gemm_batched_1d_4x4x4_b4_col", 4, 4, 4, 4),
    ("gemm_batched_1d_6x6x6_b2_col", 6, 6, 6, 2),
    ("gemm_batched_1d_3x5x7_b3_col", 3, 5, 7, 3),
])
def test_gemm_batched_1d_colmajor(bin_l3_nvidia, op, m, n, k, batch):
    """SIMT batched 1D-launch GEMM: BATCH independent (M×N)·(N×K) GEMMs, col-major."""
    alpha, beta = 1.5, 0.3
    As = [RNG.random((m, n)).astype(np.float32) for _ in range(batch)]
    Bs = [RNG.random((n, k)).astype(np.float32) for _ in range(batch)]
    Cs = [RNG.random((m, k)).astype(np.float32) for _ in range(batch)]
    C0s = [c.copy() for c in Cs]
    result = run_op(bin_l3_nvidia, op, "simple",
                    args=[alpha, beta],
                    inputs=[_flatten_col(As), _flatten_col(Bs), _flatten_col(Cs)])
    # result is one flat F-order buffer of length BATCH*M*K.
    expected = np.concatenate([
        np.asfortranarray((alpha * As[b] @ Bs[b] + beta * C0s[b]).astype(np.float32)).ravel(order='F')
        for b in range(batch)
    ])
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("op,m,n,k,batch", [
    ("gemm_batched_1d_4x4x4_b4_row", 4, 4, 4, 4),
    ("gemm_batched_1d_3x5x7_b3_row", 3, 5, 7, 3),
])
def test_gemm_batched_1d_rowmajor(bin_l3_nvidia, op, m, n, k, batch):
    """SIMT batched 1D-launch GEMM: row-major layout for all of A, B, C."""
    alpha, beta = 1.5, 0.3
    As = [RNG.random((m, n)).astype(np.float32) for _ in range(batch)]
    Bs = [RNG.random((n, k)).astype(np.float32) for _ in range(batch)]
    Cs = [RNG.random((m, k)).astype(np.float32) for _ in range(batch)]
    C0s = [c.copy() for c in Cs]
    result = run_op(bin_l3_nvidia, op, "simple",
                    args=[alpha, beta],
                    inputs=[_flatten_row(As), _flatten_row(Bs), _flatten_row(Cs)])
    expected = np.concatenate([
        (alpha * As[b] @ Bs[b] + beta * C0s[b]).astype(np.float32).ravel()
        for b in range(batch)
    ])
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── nvidia::gemm_strided_batched_1d (shared A across BATCH ops) ──────────────

@pytest.mark.parametrize("op,m,n,k,batch", [
    ("gemm_strided_batched_1d_4x4x4_b1", 4, 4, 4, 1),
    ("gemm_strided_batched_1d_4x4x4_b4", 4, 4, 4, 4),
    ("gemm_strided_batched_1d_6x6x6_b2", 6, 6, 6, 2),
    ("gemm_strided_batched_1d_3x5x7_b3", 3, 5, 7, 3),
])
def test_gemm_strided_batched_1d(bin_l3_nvidia, op, m, n, k, batch):
    """Shared-A batched GEMM: one A applied to BATCH packed (B,C) pairs."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)
    Bs = [RNG.random((n, k)).astype(np.float32) for _ in range(batch)]
    Cs = [RNG.random((m, k)).astype(np.float32) for _ in range(batch)]
    C0s = [c.copy() for c in Cs]
    result = run_op(bin_l3_nvidia, op, "simple",
                    args=[alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            _flatten_col(Bs), _flatten_col(Cs)])
    expected = np.concatenate([
        np.asfortranarray((alpha * A @ Bs[b] + beta * C0s[b]).astype(np.float32)).ravel(order='F')
        for b in range(batch)
    ])
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("op,m,n,k,batch,b_stride,c_stride", [
    # 4×4×4, B padded N*K=16 -> 24 (8 floats slack), C padded M*K=16 -> 20 (4 floats slack)
    ("gemm_strided_padded_4x4x4_b4_bs24_cs20", 4, 4, 4, 4, 24, 20),
    # 3×5×7, B packed N*K=35 -> 50, C packed M*K=21 -> 28
    ("gemm_strided_padded_3x5x7_b3_bs50_cs28", 3, 5, 7, 3, 50, 28),
])
def test_gemm_strided_batched_1d_padded(bin_l3_nvidia, op, m, n, k, batch,
                                         b_stride, c_stride):
    """Strided variant with non-default B_STRIDE / C_STRIDE — verifies the
    `b * STRIDE` indexing inside the kernel, not just the tightly-packed default."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, n)).astype(np.float32)

    # Build padded B: each batch occupies b_stride floats; only first n*k matter.
    B_padded = RNG.random(batch * b_stride).astype(np.float32)
    # Build padded C: each batch occupies c_stride floats; only first m*k matter.
    C_padded = RNG.random(batch * c_stride).astype(np.float32)

    # Extract packed col-major B[b] and C[b] for the reference computation.
    # The kernel reads col-major M×N B from B[b*b_stride : b*b_stride + n*k]
    # and writes col-major M×K C to C[b*c_stride : b*c_stride + m*k].
    Bs = [B_padded[b*b_stride : b*b_stride + n*k].reshape(n, k, order='F')
          for b in range(batch)]
    Cs0 = [C_padded[b*c_stride : b*c_stride + m*k].reshape(m, k, order='F').copy()
           for b in range(batch)]

    result = run_op(bin_l3_nvidia, op, "simple",
                    args=[alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            B_padded, C_padded])

    # Build expected output with padding bytes preserved.
    expected = C_padded.copy()
    for b in range(batch):
        new_C = (alpha * A @ Bs[b] + beta * Cs0[b]).astype(np.float32)
        expected[b*c_stride : b*c_stride + m*k] = \
            np.asfortranarray(new_C).ravel(order='F')

    # Result has length batch*c_stride (we print the full padded buffer so we
    # can verify the kernel did NOT write to the padding slots either).
    assert len(result) == batch * c_stride
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── indexed_batched_gemm ─────────────────────────────────────────────────────
# C[c_idx[p]] = A[a_idx[p]] @ B[b_idx[p]], 4x4 col-major, selected by index lists.
# numpy does each indexed product independently as the reference.

@pytest.mark.parametrize("pairs,a_mats,b_mats,c_mats", [
    (1, 1, 1, 1),
    (4, 2, 3, 4),    # repeated/aliased a_idx,b_idx; distinct c_idx
    (8, 5, 5, 8),
])
def test_indexed_batched_gemm(bins, pairs, a_mats, b_mats, c_mats):
    DIM = 4
    rng = RNG
    A_mats = [rng.random((DIM, DIM)).astype(np.float32) for _ in range(a_mats)]
    B_mats = [rng.random((DIM, DIM)).astype(np.float32) for _ in range(b_mats)]
    a_idx = rng.integers(0, a_mats, size=pairs).astype(np.int64)
    b_idx = rng.integers(0, b_mats, size=pairs).astype(np.int64)
    c_idx = rng.permutation(c_mats)[:pairs].astype(np.int64)  # distinct c slots

    A_flat = np.concatenate([np.asfortranarray(M).ravel(order='F') for M in A_mats]).astype(np.float32)
    B_flat = np.concatenate([np.asfortranarray(M).ravel(order='F') for M in B_mats]).astype(np.float32)

    result = run_op(
        bins["l3"], "indexed_bgemm_4", "simple",
        args=[DIM, DIM, DIM, pairs, a_mats, b_mats, c_mats],
        inputs=[a_idx.astype(np.float32), b_idx.astype(np.float32),
                c_idx.astype(np.float32), A_flat, B_flat])

    MAT = DIM * DIM
    expected = np.zeros(c_mats * MAT, dtype=np.float32)
    for p in range(pairs):
        prod = A_mats[a_idx[p]] @ B_mats[b_idx[p]]
        base = int(c_idx[p]) * MAT
        expected[base:base + MAT] = np.asfortranarray(prod).ravel(order='F')
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── indexed_batched_gemm: TRANSPOSE_A / TRANSPOSE_B ──────────────────────────
# Distinct c_idx, plain overwrite, but the left and/or right factor is read
# transposed. Reference applies .T to the corresponding numpy operand.

@pytest.mark.parametrize("op,ta,tb", [
    ("indexed_bgemm_4_ta", True, False),
    ("indexed_bgemm_4_tb", False, True),
])
@pytest.mark.parametrize("pairs,a_mats,b_mats,c_mats", [
    (1, 1, 1, 1),
    (4, 2, 3, 4),
    (8, 5, 5, 8),
])
def test_indexed_batched_gemm_transpose(bins, op, ta, tb, pairs, a_mats, b_mats, c_mats):
    DIM = 4
    rng = RNG
    A_mats = [rng.random((DIM, DIM)).astype(np.float32) for _ in range(a_mats)]
    B_mats = [rng.random((DIM, DIM)).astype(np.float32) for _ in range(b_mats)]
    a_idx = rng.integers(0, a_mats, size=pairs).astype(np.int64)
    b_idx = rng.integers(0, b_mats, size=pairs).astype(np.int64)
    c_idx = rng.permutation(c_mats)[:pairs].astype(np.int64)

    A_flat = np.concatenate([np.asfortranarray(M).ravel(order='F') for M in A_mats]).astype(np.float32)
    B_flat = np.concatenate([np.asfortranarray(M).ravel(order='F') for M in B_mats]).astype(np.float32)

    result = run_op(
        bins["l3"], op, "simple",
        args=[DIM, DIM, DIM, pairs, a_mats, b_mats, c_mats],
        inputs=[a_idx.astype(np.float32), b_idx.astype(np.float32),
                c_idx.astype(np.float32), A_flat, B_flat])

    MAT = DIM * DIM
    expected = np.zeros(c_mats * MAT, dtype=np.float32)
    for p in range(pairs):
        Am = A_mats[a_idx[p]].T if ta else A_mats[a_idx[p]]
        Bm = B_mats[b_idx[p]].T if tb else B_mats[b_idx[p]]
        prod = Am @ Bm
        base = int(c_idx[p]) * MAT
        expected[base:base + MAT] = np.asfortranarray(prod).ravel(order='F')
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── indexed_batched_gemm: ATOMIC_C (overlapping c_idx) ───────────────────────
# Several pairs SHARE a c_idx slot (a parent block); the atomic path must
# scatter-ADD their products. Caller pre-zeros C; reference is a numpy
# scatter-add into the shared C slots. parent_of maps pair -> c slot.

@pytest.mark.parametrize("op,ta", [
    ("indexed_bgemm_4_atomic", False),     # C += A · B
    ("indexed_bgemm_4_ta_atomic", True),   # C += Aᵀ · B  (backward Xᵀ·M·X→parent)
])
def test_indexed_batched_gemm_atomic(bins, op, ta):
    DIM = 4
    rng = RNG
    # 6 child pairs accumulating into 3 shared parent C slots (overlap by design).
    parent_of = [0, 1, 1, 2, 2, 2]
    c_mats = 3
    pairs = len(parent_of)
    a_mats = b_mats = pairs
    A_mats = [rng.random((DIM, DIM)).astype(np.float32) for _ in range(a_mats)]
    B_mats = [rng.random((DIM, DIM)).astype(np.float32) for _ in range(b_mats)]
    a_idx = np.arange(pairs).astype(np.int64)
    b_idx = np.arange(pairs).astype(np.int64)
    c_idx = np.array(parent_of, dtype=np.int64)

    A_flat = np.concatenate([np.asfortranarray(M).ravel(order='F') for M in A_mats]).astype(np.float32)
    B_flat = np.concatenate([np.asfortranarray(M).ravel(order='F') for M in B_mats]).astype(np.float32)

    result = run_op(
        bins["l3"], op, "simple",
        args=[DIM, DIM, DIM, pairs, a_mats, b_mats, c_mats],
        inputs=[a_idx.astype(np.float32), b_idx.astype(np.float32),
                c_idx.astype(np.float32), A_flat, B_flat])

    MAT = DIM * DIM
    expected = np.zeros(c_mats * MAT, dtype=np.float64)
    for p in range(pairs):
        Am = A_mats[a_idx[p]].T if ta else A_mats[a_idx[p]]
        prod = Am @ B_mats[b_idx[p]]
        base = int(c_idx[p]) * MAT
        expected[base:base + MAT] += np.asfortranarray(prod).ravel(order='F')
    expected = expected.astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)
