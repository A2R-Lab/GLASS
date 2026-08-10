"""L3 GLASS function tests — compare GPU results to NumPy/SciPy reference.

GEMM uses the standard BLAS convention: C is M×N, contraction K;
op(A) is M×K (TRANSPOSE_A ⇒ A stored K×M), op(B) is K×N (TRANSPOSE_B ⇒ B stored
N×K), ROW_MAJOR_C selects row-major output. The test net is deliberately wide:
non-square + all-distinct M,N,K (the contraction→last-dim convention swap is
SILENT on square matrices), all four transpose combos, ROW_MAJOR_C, an
alpha/beta grid, the beta=0 no-read (NaN-poison) overload, a row-major==transpose
equivalence check, a hand-written triple-loop reference, and the thread sweep.
"""

import numpy as np
import pytest
from conftest import run_op, THREAD_SWEEP, THREAD_SWEEP_CORE
from conftest import make_spd, make_lower_triangular  # shared; pass rng=RNG for varied draws


RNG = np.random.default_rng(42)

ATOL = 1e-3
RTOL = 1e-3

CG_SIMPLE = ["cg", "simple"]

# ─── gemm shapes: deliberately non-square + all-distinct M,N,K + edge rows/cols ──
GEMM_SHAPES = [
    (1, 1, 1), (2, 3, 4), (4, 2, 3), (3, 4, 2), (5, 7, 3), (7, 5, 6),
    (8, 1, 5), (1, 8, 5), (5, 8, 1), (6, 6, 6), (9, 4, 7), (16, 16, 16),
]
# Compile-time shape ids in test_l3.cu's ct_shape_dims() table.
GEMM_CT_IDS = {0: (1, 1, 1), 1: (2, 3, 4), 2: (4, 2, 3), 3: (5, 7, 3),
               4: (8, 1, 5), 5: (9, 4, 7), 6: (7, 5, 6), 7: (16, 16, 16)}
TRANSPOSE_COMBOS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _gemm_storage(m, n, k, ta, tb, seed):
    """Return (opA, opB, A_flat, B_flat): the LOGICAL op(A) (m×k) and op(B) (k×n)
    plus the PHYSICAL column-major storage the kernel reads. A row-major operand
    is a transposed column-major operand, so TRANSPOSE stores op(_)ᵀ col-major."""
    rng = np.random.default_rng(seed)
    opA = rng.standard_normal((m, k)).astype(np.float32)
    opB = rng.standard_normal((k, n)).astype(np.float32)
    A_phys = opA.T if ta else opA          # (k×m) when transposed, else (m×k)
    B_phys = opB.T if tb else opB          # (n×k) when transposed, else (k×n)
    A_flat = np.asfortranarray(A_phys).ravel(order='F')
    B_flat = np.asfortranarray(B_phys).ravel(order='F')
    return opA, opB, A_flat, B_flat


def _c_flat(C0, rmc):
    return np.ascontiguousarray(C0).ravel() if rmc else np.asfortranarray(C0).ravel(order='F')


def _decode(result, m, n, rmc):
    return result.reshape(m, n) if rmc else result.reshape(m, n, order='F')


def _gemm_ref_triple(alpha, opA, opB, beta, C0):
    """Hand-written triple-loop reference, independent of numpy `@` — the
    migration-safety net that would catch a wrong dim mapping even if `@` and the
    kernel shared a bug."""
    m, k = opA.shape
    _, n = opB.shape
    out = np.array(C0, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += float(opA[i, t]) * float(opB[t, j])
            out[i, j] = alpha * s + beta * float(C0[i, j])
    return out.astype(np.float32)


# ─── gemm: runtime, full shape × transpose × ROW_MAJOR_C matrix ────────────────

@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
@pytest.mark.parametrize("rmc", [0, 1])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemm_rt(bins, m, n, k, ta, tb, rmc, version):
    """Runtime gemm across non-square shapes, all 4 transpose combos, both C
    layouts, oracle = alpha*op(A)@op(B)+beta*C. Thread-invariant (byte-identical)
    over the core sweep."""
    alpha, beta = 1.5, 0.3
    seed = 1000 + m * 100 + n * 10 + k + ta * 3 + tb * 7 + rmc * 11
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, ta, tb, seed)
    C0 = np.random.default_rng(seed + 1).standard_normal((m, n)).astype(np.float32)
    expected = (alpha * (opA @ opB) + beta * C0).astype(np.float32)
    ref = None
    for th in THREAD_SWEEP_CORE:
        result = run_op(bins["l3"], "gemm_rt", version,
                        args=[th, m, n, k, ta, tb, rmc, 0, alpha, beta],
                        inputs=[A_flat, B_flat, _c_flat(C0, rmc)])
        got = _decode(result, m, n, rmc)
        assert np.allclose(got, expected, rtol=RTOL, atol=ATOL), \
            f"m,n,k={m},{n},{k} ta,tb={ta},{tb} rmc={rmc} th={th}"
        if ref is None:
            ref = result
        else:
            assert np.array_equal(result, ref), \
                f"non-invariant: m,n,k={m},{n},{k} ta,tb={ta},{tb} rmc={rmc} th={th}"


@pytest.mark.parametrize("m,n,k", [(5, 7, 3), (7, 5, 6), (8, 1, 5), (9, 4, 7)])
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
def test_gemm_rt_full_sweep(bins, m, n, k, ta, tb):
    """Dedicated thread-invariance over the FULL sweep (1,7,31,32,33,57,64,96,128,256)
    at representative non-square shapes — byte-identical at every block size."""
    alpha, beta = -1.25, 0.5
    seed = 2000 + m * 13 + n * 5 + k + ta + tb
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, ta, tb, seed)
    C0 = np.random.default_rng(seed + 1).standard_normal((m, n)).astype(np.float32)
    expected = (alpha * (opA @ opB) + beta * C0).astype(np.float32)
    ref = None
    for th in THREAD_SWEEP:
        result = run_op(bins["l3"], "gemm_rt", "simple",
                        args=[th, m, n, k, ta, tb, 0, 0, alpha, beta],
                        inputs=[A_flat, B_flat, _c_flat(C0, 0)])
        assert np.allclose(result.reshape(m, n, order='F'), expected, rtol=RTOL, atol=ATOL)
        if ref is None:
            ref = result
        else:
            assert np.array_equal(result, ref), f"non-invariant th={th}"


@pytest.mark.parametrize("alpha,beta", [(1, 0), (1, 1), (0, 1), (1.5, 0.3), (-1.25, 0.5), (0, 0)])
def test_gemm_rt_alpha_beta(bins, alpha, beta):
    """alpha/beta grid at a non-square shape (beta!=0 reads C)."""
    m, n, k = 5, 7, 3
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, 0, 0, seed=77)
    C0 = np.random.default_rng(78).standard_normal((m, n)).astype(np.float32)
    expected = (alpha * (opA @ opB) + beta * C0).astype(np.float32)
    result = run_op(bins["l3"], "gemm_rt", "simple",
                    args=[64, m, n, k, 0, 0, 0, 0, alpha, beta],
                    inputs=[A_flat, B_flat, _c_flat(C0, 0)])
    assert np.allclose(result.reshape(m, n, order='F'), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n,k", [(5, 7, 3), (8, 1, 5), (6, 6, 6)])
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
def test_gemm_rt_beta0_no_read(bins, m, n, k, ta, tb):
    """The no-beta overload (nb=1) must OVERWRITE C and never read it: a
    NaN-poisoned C must not contaminate the result."""
    alpha = 1.5
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, ta, tb, seed=303)
    C_poison = np.full((m, n), np.nan, dtype=np.float32)
    expected = (alpha * (opA @ opB)).astype(np.float32)
    result = run_op(bins["l3"], "gemm_rt", "simple",
                    args=[64, m, n, k, ta, tb, 0, 1, alpha, 0.0],
                    inputs=[A_flat, B_flat, _c_flat(C_poison, 0)])
    assert not np.any(np.isnan(result)), "beta=0 no-read overload read the NaN-poisoned C"
    assert np.allclose(result.reshape(m, n, order='F'), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n,k", [(5, 7, 3), (8, 1, 5), (6, 6, 6), (16, 4, 8)])
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
def test_gemm_rt_betaform_beta0_no_read(bins, m, n, k, ta, tb):
    """BLAS beta==0 semantics: the BETA overload (nb=0) called with beta=0 must
    also treat C as write-only — 0*NaN must not poison the result. Regression
    for the GRiD s_vaf uninit-smem NaN (2026-07-08); the (16,4,8) shape takes
    the tile4 fast path."""
    alpha = 1.5
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, ta, tb, seed=304)
    C_poison = np.full((m, n), np.nan, dtype=np.float32)
    expected = (alpha * (opA @ opB)).astype(np.float32)
    result = run_op(bins["l3"], "gemm_rt", "simple",
                    args=[64, m, n, k, ta, tb, 0, 0, alpha, 0.0],
                    inputs=[A_flat, B_flat, _c_flat(C_poison, 0)])
    assert not np.any(np.isnan(result)), "beta form at beta=0 read the NaN-poisoned C"
    assert np.allclose(result.reshape(m, n, order='F'), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n,k", [(5, 7, 3), (8, 1, 5), (9, 4, 7)])
def test_gemm_rowmajor_is_transpose(bins, m, n, k):
    """Row-major == transpose, bit-for-bit: feeding op(A) as a row-major M×K buffer
    (read with TRANSPOSE_A over its col-major K×M view) yields output IDENTICAL to
    the plain col-major NN call. Proves pruning per-operand ROW_MAJOR is loss-free."""
    rng = np.random.default_rng(404)
    opA = rng.standard_normal((m, k)).astype(np.float32)
    opB = rng.standard_normal((k, n)).astype(np.float32)
    C0 = rng.standard_normal((m, n)).astype(np.float32)
    alpha, beta = 1.3, 0.4
    # Path 1: NN — A stored col-major (m×k).
    A_nn = np.asfortranarray(opA).ravel(order='F')
    r_nn = run_op(bins["l3"], "gemm_rt", "simple",
                  args=[64, m, n, k, 0, 0, 0, 0, alpha, beta],
                  inputs=[A_nn, np.asfortranarray(opB).ravel(order='F'), _c_flat(C0, 0)])
    # Path 2: TA — same opA but stored row-major (== col-major (k×m)), read transposed.
    A_ta = np.ascontiguousarray(opA).ravel()        # row-major M×K bytes
    r_ta = run_op(bins["l3"], "gemm_rt", "simple",
                  args=[64, m, n, k, 1, 0, 0, 0, alpha, beta],
                  inputs=[A_ta, np.asfortranarray(opB).ravel(order='F'), _c_flat(C0, 0)])
    assert np.array_equal(r_nn, r_ta), "row-major-via-transpose differs from col-major NN"


# ─── gemm: compile-time (magic-multiply) path + hand-written reference ─────────

@pytest.mark.parametrize("shape_id", list(GEMM_CT_IDS.keys()))
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
def test_gemm_ct(bins, shape_id, ta, tb):
    """Compile-time-size gemm across the non-square shape table, all transpose
    combos, vs a hand-written triple-loop reference; thread-invariant."""
    m, n, k = GEMM_CT_IDS[shape_id]
    alpha, beta = 1.5, 0.3
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, ta, tb, seed=500 + shape_id)
    C0 = np.random.default_rng(600 + shape_id).standard_normal((m, n)).astype(np.float32)
    expected = _gemm_ref_triple(alpha, opA, opB, beta, C0)
    ref = None
    for th in THREAD_SWEEP_CORE:
        result = run_op(bins["l3"], "gemm_ct", "simple",
                        args=[th, shape_id, ta, tb, 0, 0, alpha, beta],
                        inputs=[A_flat, B_flat, _c_flat(C0, 0)])
        assert np.allclose(result.reshape(m, n, order='F'), expected, rtol=RTOL, atol=ATOL), \
            f"shape_id={shape_id} ({m},{n},{k}) ta,tb={ta},{tb} th={th}"
        if ref is None:
            ref = result
        else:
            assert np.array_equal(result, ref), f"non-invariant shape_id={shape_id} th={th}"


# ─── gemm: single-warp (one warp, launched <<<1,32>>>) ────────────────────────

@pytest.mark.parametrize("shape_id", [0, 1, 2, 3, 4, 5, 6])  # skip 16×16×16 (warp = 256 outputs serial-K, slow but fine)
@pytest.mark.parametrize("ta,tb", TRANSPOSE_COMBOS)
def test_gemm_warp(bins, shape_id, ta, tb):
    """Single-warp compile-time gemm == hand reference at one warp."""
    m, n, k = GEMM_CT_IDS[shape_id]
    alpha, beta = 1.5, 0.3
    opA, opB, A_flat, B_flat = _gemm_storage(m, n, k, ta, tb, seed=700 + shape_id)
    C0 = np.random.default_rng(800 + shape_id).standard_normal((m, n)).astype(np.float32)
    expected = _gemm_ref_triple(alpha, opA, opB, beta, C0)
    result = run_op(bins["l3"], "gemm_warp", "warp",
                    args=[0, shape_id, ta, tb, 0, 0, alpha, beta],
                    inputs=[A_flat, B_flat, _c_flat(C0, 0)])
    assert np.allclose(result.reshape(m, n, order='F'), expected, rtol=RTOL, atol=ATOL)


# ─── gemm_tiled (no transpose; A m×k, B k×n, C m×n) ──────────────────────────

@pytest.mark.parametrize("m,n,k", [(4, 6, 5), (8, 8, 8), (12, 4, 6), (6, 10, 7), (4, 3, 4)])
def test_gemm_tiled(bins, m, n, k):
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, k)).astype(np.float32)   # A is m×k
    B = RNG.random((k, n)).astype(np.float32)   # B is k×n
    C = RNG.random((m, n)).astype(np.float32)
    C0 = C.copy()
    result = run_op(bins["l3"], "gemm_tiled", "simple",
                    args=[m, n, k, alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_strided (standard convention, explicit leading dims) ────────────

@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
@pytest.mark.parametrize("op,m,n,k,a_rs,b_rs", [
    ("rsgemm_6x6x6_6_6", 6, 6, 6, 6, 6),
    ("rsgemm_6x6x6_8_8", 6, 6, 6, 8, 8),
    ("rsgemm_4x4x4_4_4", 4, 4, 4, 4, 4),
    ("rsgemm_4x4x4_6_6", 4, 4, 4, 6, 6),
    ("rsgemm_5x7x3_8_6", 5, 7, 3, 8, 6),
])
def test_gemm_strided(bins, op, m, n, k, a_rs, b_rs, alpha, beta):
    # A is M×K, lead A_RS: A[m][k] = A_flat[m + k*a_rs].
    # B is K×N, lead B_RS: B[k][n] = B_flat[k + n*b_rs].  C standard col-major.
    A_storage = np.zeros((a_rs, k), dtype=np.float32)
    A_storage[:m, :] = RNG.random((m, k)).astype(np.float32)
    B_storage = np.zeros((b_rs, n), dtype=np.float32)
    B_storage[:k, :] = RNG.random((k, n)).astype(np.float32)
    C = RNG.random((m, n)).astype(np.float32)
    C0 = C.copy()
    A_flat = np.asfortranarray(A_storage).ravel(order='F')
    B_flat = np.asfortranarray(B_storage).ravel(order='F')
    C_flat = np.asfortranarray(C).ravel(order='F')
    ref = None
    for th in THREAD_SWEEP_CORE:
        result = run_op(bins["l3"], op, "simple",
                        args=[th, alpha, beta], inputs=[A_flat, B_flat, C_flat])
        expected = (alpha * A_storage[:m, :] @ B_storage[:k, :] + beta * C0).astype(np.float32)
        mat = result.reshape(m, n, order='F')
        assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL), f"{op} th={th}"
        if ref is None:
            ref = result
        else:
            assert np.array_equal(result, ref), f"{op} non-invariant th={th}"


# ─── packed_gemm (compile-time 4×K) ───────────────────────────────────────────

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
    # glass::gemm<float,4,4,K>: C(4×4) = alpha * A(4×K) * B(K×4) + beta * C(4×4).
    m, n = 4, 4
    alpha, beta = 1.5, 0.3
    A = _make_packed_vec(m * k, case).reshape(m, k)   # A is 4×K
    B = _make_packed_vec(k * n, case).reshape(k, n)   # B is K×4
    C = _make_packed_vec(m * n, case).reshape(m, n)
    C0 = C.copy()
    result = run_op(bins["l3"], f"packed_gemm_4x4x{k}", "simple",
                    args=[alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            np.asfortranarray(B).ravel(order='F'),
                            np.asfortranarray(C).ravel(order='F')])
    expected = (alpha * A @ B + beta * C0).astype(np.float32)
    mat = result.reshape(m, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)




# ─── inv ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("cond", [None, 1e4])  # well-conditioned + ill-conditioned
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_inv(bins, n, cond, version):
    """inv across the canonical thread sweep (incl. partial/odd/non-warp
    counts), two input conditionings. The augmented-row save loop must be
    grid-strided (`ind += size`); a non-strided `ind++` is a write-write race that
    compute-sanitizer racecheck flags even though every racing write stores the same
    value (so it is invisible at 32/256 — see test_hardening doc). Output must be
    byte-for-byte identical at every block size."""
    A = make_spd(n, seed=n, cond=cond)
    # inv expects [A | I] layout (n*2n) column-major
    AI = np.hstack([A, np.eye(n, dtype=np.float32)])   # row layout (n x 2n)
    AI_col = np.asfortranarray(AI).ravel(order='F')
    expected = np.linalg.inv(A).astype(np.float32)
    rtol, atol = (1e-2, 1e-3) if cond is None else (5e-2, 5e-3)
    ref = None
    for threads in THREAD_SWEEP:
        result = run_op(bins["l3"], "inv", version, args=[threads, n], inputs=[AI_col])
        Ainv = result.reshape(n, n, order='F')
        assert np.allclose(Ainv, expected, rtol=rtol, atol=atol), \
            f"n={n} cond={cond} threads={threads}: mismatch vs np.linalg.inv"
        if ref is None:
            ref = result
        else:
            assert np.array_equal(result, ref), \
                f"n={n} cond={cond} threads={threads}: output differs from threads=1"


def _aug(M, d):
    # [M | I] column-major augmented buffer for inv
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
    A = make_spd(n, rng=RNG)
    A[0, 0] = 1e-7                 # tiny leading pivot
    A[n - 1, 0] = 5.0             # large later entry in column 0 → must pivot up
    A[0, n - 1] = 5.0            # keep it reasonably conditioned / nonsymmetric
    return A.astype(np.float32)


def _zero_diagonal_perm(n):
    """Invertible matrix with a literal ZERO on the (0,0) diagonal — the plain
    path divides by exactly 0; partial pivoting swaps a nonzero row up."""
    A = make_spd(n, rng=RNG)
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
    A = make_spd(n, rng=RNG)
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
    A = make_spd(dimA, rng=RNG); B = make_spd(dimB, rng=RNG)
    res = run_op(bins["l3"], "inv2", "simple", args=[dimA, dimB, max(dimA, dimB)],
                 inputs=[_aug(A, dimA), _aug(B, dimB)])
    assert np.allclose(res[0].reshape(dimA, dimA, order='F'), np.linalg.inv(A), rtol=1e-2, atol=1e-3)
    assert np.allclose(res[1].reshape(dimB, dimB, order='F'), np.linalg.inv(B), rtol=1e-2, atol=1e-3)


# (12,12,6) mirrors GATO's Schur fused-3 (STATE_SIZE=12, CONTROL_SIZE=6 for indy7)
@pytest.mark.parametrize("dimA,dimB,dimC", [(12, 12, 6), (6, 6, 6), (4, 6, 3)])
def test_inv3(bins, dimA, dimB, dimC):
    A = make_spd(dimA, rng=RNG); B = make_spd(dimB, rng=RNG); C = make_spd(dimC, rng=RNG)
    res = run_op(bins["l3"], "inv3", "simple", args=[dimA, dimB, dimC, max(dimA, dimB, dimC)],
                 inputs=[_aug(A, dimA), _aug(B, dimB), _aug(C, dimC)])
    for M, d, r in [(A, dimA, res[0]), (B, dimB, res[1]), (C, dimC, res[2])]:
        assert np.allclose(r.reshape(d, d, order='F'), np.linalg.inv(M), rtol=1e-2, atol=1e-3)


# ─── chol ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_chol(bins, n, version):
    A = make_spd(n, rng=RNG)
    A_col = np.asfortranarray(A).ravel(order='F')
    result = run_op(bins["l3"], "chol", version, args=[n], inputs=[A_col])
    L_gpu = result.reshape(n, n, order='F')
    # Extract lower triangle
    L_gpu = np.tril(L_gpu)
    L_ref = np.linalg.cholesky(A).astype(np.float32)
    assert np.allclose(L_gpu, L_ref, rtol=1e-2, atol=1e-3)


# ─── trsm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 4, 6, 8])
@pytest.mark.parametrize("nrhs", [1, 3])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_trsm(bins, n, nrhs, transpose, version):
    L = make_lower_triangular(n, rng=RNG)
    B = RNG.random((n, nrhs)).astype(np.float32)
    L_col = np.asfortranarray(L).ravel(order='F')
    B_col = np.asfortranarray(B).ravel(order='F')
    result = run_op(bins["l3"], "trsm", version,
                    args=[n, nrhs, int(transpose)], inputs=[L_col, B_col])
    X = result.reshape(n, nrhs, order='F')
    # Verify op(L) @ X == B (residual check avoids scipy dependency)
    opL = L.T if transpose else L
    residual = opL.astype(np.float64) @ X.astype(np.float64) - B.astype(np.float64)
    assert np.allclose(residual, 0, atol=1e-3)


@pytest.mark.parametrize("transpose", [False, True])
def test_trsm_warp_7_3(bins, transpose):
    # Single-warp multi-RHS trsm (compile-time N=7, NRHS=3), forward + transpose.
    n, nrhs = 7, 3
    L = make_lower_triangular(n, rng=RNG)
    B = RNG.random((n, nrhs)).astype(np.float32)
    L_col = np.asfortranarray(L).ravel(order='F')
    B_col = np.asfortranarray(B).ravel(order='F')
    result = run_op(bins["l3"], "trsm_warp", "warp",
                    args=[int(transpose)], inputs=[L_col, B_col])
    X = result.reshape(n, nrhs, order='F')
    opL = L.T if transpose else L
    residual = opL.astype(np.float64) @ X.astype(np.float64) - B.astype(np.float64)
    assert np.allclose(residual, 0, atol=1e-3)


# ─── warp:: (single warp, launched <<<1,32>>>) ────────────────────────────────


def test_posv_warp_7(bins):
    # Single-warp SPD solve A x = b via warp:: potrf + trsv forward/back (N=7).
    n = 7
    A = make_spd(n, rng=RNG)
    b = RNG.random(n).astype(np.float32)
    A_col = np.asfortranarray(A).ravel(order='F')
    result = run_op(bins["l3"], "posv_warp", "warp", args=[n], inputs=[A_col, b])
    residual = A.astype(np.float64) @ result.astype(np.float64) - b.astype(np.float64)
    assert np.allclose(residual, 0, atol=1e-3)
