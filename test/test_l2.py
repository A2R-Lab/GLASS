"""L2 GLASS function tests — NumPy oracle + canonical thread-count sweep.

Thread discipline (see docs/open-tasks/test_hardening_thread_sweep_2026-06-24.md):
every L2 op here assigns one OUTPUT element per thread with a serial
contraction, so results must be BYTE-IDENTICAL across block sizes — except the
ATOMIC_Y segmented paths, whose float atomicAdd ordering is nondeterministic
(those check oracle-closeness at every swept count instead).

The CUDA runner takes the block size as a runtime arg:
    ./test_l2 <op> <ver> <threads> <m> <n> [args...] [files...]
Most tests sweep THREAD_SWEEP_CORE; gemv/gemv_t/ger get a FULL-sweep test.
Input variety: 'normal' A (mean-zero general) and 'colscaled' A (alternating
1e3/1e-3 column magnitudes — stresses the accumulation) via conftest makers.
"""

import numpy as np
import pytest
from conftest import run_op, THREAD_SWEEP, THREAD_SWEEP_CORE, make_general, make_vec

RNG = np.random.default_rng(42)

ATOL = 1e-4
RTOL = 1e-3

CG_SIMPLE = ["cg", "simple"]

A_KINDS = ["normal", "colscaled"]


def _make_mat(m, n, seed, kind="normal"):
    """General m x n float32 matrix. 'colscaled' alternates huge/tiny column
    magnitudes (1e3 / 1e-3) to stress the per-output serial accumulation."""
    A = make_general(m, n, seed=seed)
    if kind == "colscaled":
        A = A.copy()
        A[:, 0::2] *= 1e3
        A[:, 1::2] *= 1e-3
    return A.astype(np.float32)


def _rel_close(got, expected, rtol=2e-3):
    """Elementwise closeness scaled by the expected magnitude (colscaled inputs
    make a fixed atol meaningless)."""
    tol = rtol * np.maximum(1.0, np.abs(expected))
    return np.all(np.abs(got.astype(np.float64) - expected.astype(np.float64)) <= tol)


def sweep_exact(binary, op, version, args, inputs, sweep=THREAD_SWEEP_CORE):
    """Run `op` at every thread count in `sweep`; assert BYTE-IDENTICAL output
    and return the first result."""
    ref = None
    for th in sweep:
        r = run_op(binary, op, version, [th] + list(args), inputs)
        if ref is None:
            ref = r
        else:
            assert np.array_equal(ref, r), \
                f"{op}/{version}: thread-count non-invariance at {th} threads"
    return ref


# ─── gemv ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n", [(4, 6), (8, 8), (16, 12)])
@pytest.mark.parametrize("kind", A_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemv(bins, m, n, kind, version):
    alpha, beta = 1.5, 0.3
    A = _make_mat(m, n, seed=m * 10 + n, kind=kind)
    x = make_vec(n, seed=m + n)
    y = make_vec(m, seed=m + n + 1)
    # A stored column-major
    A_col = np.asfortranarray(A)
    result = sweep_exact(bins["l2"], "gemv", version,
                         [m, n, alpha, beta],
                         [A_col.ravel(order='F'), x, y])
    expected = (alpha * A.astype(np.float64) @ x + beta * y).astype(np.float32)
    assert _rel_close(result, expected)


@pytest.mark.parametrize("m,n", [(4, 6), (8, 8), (16, 12)])
@pytest.mark.parametrize("kind", A_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemv_t(bins, m, n, kind, version):
    # y = alpha * A^T * x + beta * y  (A: mxn, x: m, y: n)
    alpha, beta = 1.5, 0.3
    A = _make_mat(m, n, seed=m * 10 + n + 2, kind=kind)
    x = make_vec(m, seed=m + n + 3)
    y = make_vec(n, seed=m + n + 4)
    A_col = np.asfortranarray(A)
    result = sweep_exact(bins["l2"], "gemv_t", version,
                         [m, n, alpha, beta],
                         [A_col.ravel(order='F'), x, y])
    expected = (alpha * A.T.astype(np.float64) @ x + beta * y).astype(np.float32)
    assert _rel_close(result, expected)


@pytest.mark.parametrize("op,trans", [("gemv", False), ("gemv_t", True)])
@pytest.mark.parametrize("kind", A_KINDS)
def test_gemv_full_sweep(bins, op, trans, kind):
    """gemv / gemvᵀ over the FULL canonical sweep at a non-square shape."""
    m, n = 13, 7
    alpha, beta = -1.25, 0.5
    A = _make_mat(m, n, seed=99, kind=kind)
    xl, yl = (m, n) if trans else (n, m)
    x = make_vec(xl, seed=100)
    y = make_vec(yl, seed=101)
    result = sweep_exact(bins["l2"], op, "simple", [m, n, alpha, beta],
                         [np.asfortranarray(A).ravel(order='F'), x, y],
                         sweep=THREAD_SWEEP)
    opA = A.T if trans else A
    expected = (alpha * opA.astype(np.float64) @ x + beta * y).astype(np.float32)
    assert _rel_close(result, expected)


# BLAS beta==0 semantics: the beta overload called with beta=0 must treat y as
# WRITE-ONLY — a NaN-poisoned y must not contaminate the result (an unguarded
# blend computes 0*NaN=NaN). Regression for the GRiD s_vaf uninit-smem NaN
# (2026-07-08): generated RNEA issues beta=0 gemvs over cold shared memory.
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_gemv_beta0_poisoned_y_no_read(bins, version):
    m, n = 8, 6
    alpha = 1.5
    A = _make_mat(m, n, seed=77)
    x = make_vec(n, seed=78)
    y = np.full(m, np.nan, dtype=np.float32)
    result = sweep_exact(bins["l2"], "gemv", version,
                         [m, n, alpha, 0.0],
                         [np.asfortranarray(A).ravel(order='F'), x, y])
    assert not np.any(np.isnan(result)), "beta=0 gemv read the NaN-poisoned y"
    expected = (alpha * A.astype(np.float64) @ x).astype(np.float32)
    assert _rel_close(result, expected)


# ─── gemv row-major ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n", [(8, 6), (12, 4), (16, 12)])
@pytest.mark.parametrize("kind", A_KINDS)
def test_gemv_rowmajor(bins, m, n, kind):
    """Row-major A (C-contiguous): y = alpha*A*x + beta*y."""
    alpha, beta = 1.5, 0.3
    A = _make_mat(m, n, seed=m * 10 + n + 5, kind=kind)   # C-order = row-major
    x = make_vec(n, seed=m + n + 6)
    y = make_vec(m, seed=m + n + 7)
    result = sweep_exact(bins["l2"], "gemv_rowmajor", "simple",
                         [m, n, alpha, beta],
                         [np.ascontiguousarray(A).ravel(), x, y])
    expected = (alpha * A.astype(np.float64) @ x + beta * y).astype(np.float32)
    assert _rel_close(result, expected)


# ─── gemv_strided ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
@pytest.mark.parametrize("op,m,n,rs", [
    ("gemv_strided_6x6_6", 6, 6, 6),
    ("gemv_strided_6x6_8", 6, 6, 8),
    ("gemv_strided_4x4_4", 4, 4, 4),
    ("gemv_strided_4x4_6", 4, 4, 6),
])
def test_gemv_strided(bins, op, m, n, rs, alpha, beta):
    # A stored column-major with LDA=rs: A[i][j] = A_flat[i + j*rs]
    # Allocate rs×n storage; only first m rows are used by the kernel.
    A_storage = np.zeros((rs, n), dtype=np.float32)
    A_storage[:m, :] = make_general(m, n, seed=rs * 10 + m)
    x = make_vec(n, seed=rs + m)
    y = make_vec(m, seed=rs + m + 1)
    A_flat = np.asfortranarray(A_storage).ravel(order='F')
    result = sweep_exact(bins["l2"], op, "simple",
                         [m, n, alpha, beta], [A_flat, x, y])
    expected = (alpha * A_storage[:m, :].astype(np.float64) @ x + beta * y).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# beta==0 write-only semantics for the strided beta form (see the gemv poison test).
def test_gemv_strided_beta0_poisoned_y_no_read(bins):
    m, n, rs = 6, 6, 8
    A_storage = np.zeros((rs, n), dtype=np.float32)
    A_storage[:m, :] = make_general(m, n, seed=79)
    x = make_vec(n, seed=80)
    y = np.full(m, np.nan, dtype=np.float32)
    result = sweep_exact(bins["l2"], "gemv_strided_6x6_8", "simple",
                         [m, n, 1.0, 0.0],
                         [np.asfortranarray(A_storage).ravel(order='F'), x, y])
    assert not np.any(np.isnan(result)), "beta=0 gemv_strided read the NaN-poisoned y"
    expected = (A_storage[:m, :].astype(np.float64) @ x).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── gemv_segmented ──────────────────────────────────────────────
# `segments` independent 6x6 col-major (LDA=6) GEMVs computed concurrently.
# Descriptor arrays give per-segment base element offsets into packed buffers.
# numpy does each GEMV independently as the reference. One output row per
# thread, serial contraction → byte-identical across the sweep (non-atomic).

def _build_segmented(segments, m, n, rs, rng, with_S=False):
    """Pack `segments` GEMVs into contiguous A/x/y buffers with per-seg offsets."""
    A_blocks, x_blocks, y_blocks, S_blocks = [], [], [], []
    a_off, x_off, y_off, s_off = [], [], [], []
    A_cur = x_cur = y_cur = s_cur = 0
    A_list, x_list, y_list, S_list = [], [], [], []
    for _ in range(segments):
        Aseg = rng.standard_normal((rs, n)).astype(np.float32)
        Aseg[m:, :] = 0.0
        xseg = rng.standard_normal(n).astype(np.float32)
        yseg = rng.standard_normal(m).astype(np.float32)
        a_off.append(A_cur); x_off.append(x_cur); y_off.append(y_cur)
        A_list.append(np.asfortranarray(Aseg).ravel(order='F'))
        x_list.append(xseg); y_list.append(yseg)
        A_cur += rs * n; x_cur += n; y_cur += m
        A_blocks.append(Aseg[:m, :]); x_blocks.append(xseg); y_blocks.append(yseg)
        if with_S:
            Sseg = rng.standard_normal(m).astype(np.float32)
            s_off.append(s_cur); S_list.append(Sseg); s_cur += m
            S_blocks.append(Sseg)
    A = np.concatenate(A_list).astype(np.float32)
    x = np.concatenate(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    out = dict(A=A, x=x, y=y, a_off=np.array(a_off, np.float32),
               x_off=np.array(x_off, np.float32), y_off=np.array(y_off, np.float32),
               A_blocks=A_blocks, x_blocks=x_blocks, y_blocks=y_blocks)
    if with_S:
        out["S"] = np.concatenate(S_list).astype(np.float32)
        out["s_off"] = np.array(s_off, np.float32)
        out["S_blocks"] = S_blocks
    return out


# beta==0 write-only semantics for the segmented beta form — THE GRiD RNEA shape
# (v/a forward-pass gemvs write cold s_vaf slots with beta=0; see the gemv poison test).
@pytest.mark.parametrize("segments", [1, 3])
def test_gemv_segmented_beta0_poisoned_y_no_read(bins, segments):
    m, n, rs = 6, 6, 6
    d = _build_segmented(segments, m, n, rs, RNG)
    y_poison = np.full_like(d["y"], np.nan)
    result = sweep_exact(
        bins["l2"], "seg_gemv_6x6_nofuse", "simple",
        [m, n, segments, 1.0, 0.0, d["A"].size, d["x"].size, d["y"].size],
        [d["a_off"], d["x_off"], d["y_off"], d["A"], d["x"], y_poison])
    assert not np.any(np.isnan(result)), "beta=0 gemv_segmented read the NaN-poisoned y"
    expected = np.concatenate(
        [d["A_blocks"][s] @ d["x_blocks"][s] for s in range(segments)]).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("segments", [1, 3, 5])
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
def test_gemv_segmented_nofuse(bins, segments, alpha, beta):
    m, n, rs = 6, 6, 6
    d = _build_segmented(segments, m, n, rs, RNG)
    y0 = d["y"].copy()
    result = sweep_exact(
        bins["l2"], "seg_gemv_6x6_nofuse", "simple",
        [m, n, segments, alpha, beta, d["A"].size, d["x"].size, d["y"].size],
        [d["a_off"], d["x_off"], d["y_off"], d["A"], d["x"], d["y"]])
    expected = []
    for s in range(segments):
        yseg0 = d["y_blocks"][s]
        expected.append(alpha * d["A_blocks"][s] @ d["x_blocks"][s] + beta * yseg0)
    expected = np.concatenate(expected).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("segments", [1, 3, 5])
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
def test_gemv_segmented_fuse(bins, segments, alpha, beta):
    m, n, rs = 6, 6, 6
    d = _build_segmented(segments, m, n, rs, RNG, with_S=True)
    scalar = (RNG.random(segments) - 0.5).astype(np.float32)
    result = sweep_exact(
        bins["l2"], "seg_gemv_6x6_fuse", "simple",
        [m, n, segments, alpha, beta,
         d["A"].size, d["x"].size, d["y"].size, d["S"].size],
        [d["a_off"], d["x_off"], d["y_off"], d["A"], d["x"], d["y"],
         d["s_off"], d["S"], scalar])
    expected = []
    for s in range(segments):
        gemv = alpha * d["A_blocks"][s] @ d["x_blocks"][s] + beta * d["y_blocks"][s]
        gemv = gemv + d["S_blocks"][s] * scalar[s]
        expected.append(gemv)
    expected = np.concatenate(expected).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── gemv_segmented: TRANSPOSE ───────────────────────────────────
# Per segment y_seg(N) = alpha * Aᵀ_seg(N×M) * x_seg(M) + beta*y_seg(N).
# A_seg is M×N col-major LDA=rs; the kernel binds M=6,N=4,ROW_STRIDE=6.
# Segments keep DISJOINT y ranges (non-atomic), checked vs a per-segment Aᵀ·x.

@pytest.mark.parametrize("segments", [1, 3, 5])
@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (1.0, 0.0)])
def test_gemv_segmented_transpose(bins, segments, alpha, beta):
    m, n, rs = 6, 4, 6
    A_list, x_list, y_list = [], [], []
    a_off, x_off, y_off = [], [], []
    A_blocks, x_blocks, y_blocks = [], [], []
    A_cur = x_cur = y_cur = 0
    for _ in range(segments):
        Aseg = RNG.standard_normal((rs, n)).astype(np.float32)
        Aseg[m:, :] = 0.0
        xseg = RNG.standard_normal(m).astype(np.float32)   # transpose: x has M values
        yseg = RNG.standard_normal(n).astype(np.float32)   # transpose: y has N values
        a_off.append(A_cur); x_off.append(x_cur); y_off.append(y_cur)
        A_list.append(np.asfortranarray(Aseg).ravel(order='F'))
        x_list.append(xseg); y_list.append(yseg)
        A_cur += rs * n; x_cur += m; y_cur += n
        A_blocks.append(Aseg[:m, :]); x_blocks.append(xseg); y_blocks.append(yseg)
    A = np.concatenate(A_list).astype(np.float32)
    x = np.concatenate(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    result = sweep_exact(
        bins["l2"], "seg_gemv_transpose", "simple",
        [m, n, segments, alpha, beta, A.size, x.size, y.size],
        [np.array(a_off, np.float32), np.array(x_off, np.float32),
         np.array(y_off, np.float32), A, x, y])
    expected = []
    for s in range(segments):
        expected.append(alpha * A_blocks[s].T @ x_blocks[s] + beta * y_blocks[s])
    expected = np.concatenate(expected).astype(np.float32)
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── gemv_segmented: ATOMIC_Y (overlapping y) ─────────────────────
# Per segment y_seg(M) += alpha * A_seg(M×N) * x_seg(N). Multiple segments share
# the SAME y range (a parent), so the atomic path must SCATTER-ADD them. Caller
# pre-zeros y; reference is a numpy scatter-add. M=6,N=6,rs=6.
# Float atomicAdd ordering is nondeterministic → oracle-close at every count,
# NOT byte-identical across the sweep.

@pytest.mark.parametrize("alpha", [1.0, 1.5])
def test_gemv_segmented_atomic(bins, alpha):
    m, n, rs = 6, 6, 6
    # 3 parents, several child segments each accumulating into a shared parent.
    parent_of = [0, 1, 1, 2, 2, 2]   # seg -> parent index (overlap by design)
    n_parents = 3
    segments = len(parent_of)
    A_list, x_list = [], []
    a_off, x_off, y_off = [], [], []
    A_blocks, x_blocks = [], []
    A_cur = x_cur = 0
    for s in range(segments):
        Aseg = RNG.standard_normal((rs, n)).astype(np.float32)
        Aseg[m:, :] = 0.0
        xseg = RNG.standard_normal(n).astype(np.float32)
        a_off.append(A_cur); x_off.append(x_cur); y_off.append(parent_of[s] * m)
        A_list.append(np.asfortranarray(Aseg).ravel(order='F'))
        x_list.append(xseg)
        A_cur += rs * n; x_cur += n
        A_blocks.append(Aseg[:m, :]); x_blocks.append(xseg)
    A = np.concatenate(A_list).astype(np.float32)
    x = np.concatenate(x_list).astype(np.float32)
    y = np.zeros(n_parents * m, np.float32)   # pre-zeroed accumulator
    expected = np.zeros(n_parents * m, np.float64)
    for s in range(segments):
        p = parent_of[s]
        expected[p*m:(p+1)*m] += alpha * (A_blocks[s] @ x_blocks[s])
    expected = expected.astype(np.float32)
    for th in THREAD_SWEEP_CORE:
        result = run_op(
            bins["l2"], "seg_gemv_atomic", "simple",
            args=[th, m, n, segments, alpha, A.size, x.size, y.size],
            inputs=[np.array(a_off, np.float32), np.array(x_off, np.float32),
                    np.array(y_off, np.float32), A, x, y])
        assert np.allclose(result, expected, rtol=RTOL, atol=ATOL), f"threads={th}"


# ─── gemv_segmented: TRANSPOSE + ATOMIC_Y (backward pass) ─────────
# The leaf→root case: each child segment computes Aᵀ_seg·x_seg (the Xᵀ·f map)
# and atomically accumulates it into a SHARED parent y range. M=6,N=6,rs=6 so
# y_seg also has length 6 (square X). Reference is a transposed scatter-add.

@pytest.mark.parametrize("alpha", [1.0, 1.5])
def test_gemv_segmented_transpose_atomic(bins, alpha):
    m, n, rs = 6, 6, 6
    parent_of = [0, 1, 1, 2, 2, 2]
    n_parents = 3
    segments = len(parent_of)
    A_list, x_list = [], []
    a_off, x_off, y_off = [], [], []
    A_blocks, x_blocks = [], []
    A_cur = x_cur = 0
    for s in range(segments):
        Aseg = RNG.standard_normal((rs, n)).astype(np.float32)
        Aseg[m:, :] = 0.0
        xseg = RNG.standard_normal(m).astype(np.float32)   # transpose: x has M values
        a_off.append(A_cur); x_off.append(x_cur); y_off.append(parent_of[s] * n)
        A_list.append(np.asfortranarray(Aseg).ravel(order='F'))
        x_list.append(xseg)
        A_cur += rs * n; x_cur += m
        A_blocks.append(Aseg[:m, :]); x_blocks.append(xseg)
    A = np.concatenate(A_list).astype(np.float32)
    x = np.concatenate(x_list).astype(np.float32)
    y = np.zeros(n_parents * n, np.float32)
    expected = np.zeros(n_parents * n, np.float64)
    for s in range(segments):
        p = parent_of[s]
        expected[p*n:(p+1)*n] += alpha * (A_blocks[s].T @ x_blocks[s])
    expected = expected.astype(np.float32)
    for th in THREAD_SWEEP_CORE:
        result = run_op(
            bins["l2"], "seg_gemv_transpose_atomic", "simple",
            args=[th, m, n, segments, alpha, A.size, x.size, y.size],
            inputs=[np.array(a_off, np.float32), np.array(x_off, np.float32),
                    np.array(y_off, np.float32), A, x, y])
        assert np.allclose(result, expected, rtol=RTOL, atol=ATOL), f"threads={th}"


# ─── ger ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,n", [(4, 6), (8, 8), (16, 12)])
@pytest.mark.parametrize("kind", ["normal", "mixed"])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_ger(bins, m, n, kind, version):
    alpha = 0.5
    x = make_vec(m, seed=m * 10 + n + 8, kind=kind)
    y = make_vec(n, seed=m * 10 + n + 9, kind=kind)
    A = make_general(m, n, seed=m * 10 + n + 10)
    A_col = np.asfortranarray(A)
    result = sweep_exact(bins["l2"], "ger", version,
                         [m, n, alpha],
                         [x, y, A_col.ravel(order='F')])
    expected = (A.astype(np.float64) + alpha * np.outer(x, y)).astype(np.float32)
    mat = result.reshape(m, n, order='F')
    assert _rel_close(mat, expected)


@pytest.mark.parametrize("kind", ["normal", "mixed"])
def test_ger_full_sweep(bins, kind):
    m, n = 13, 7
    alpha = -0.75
    x = make_vec(m, seed=110, kind=kind)
    y = make_vec(n, seed=111, kind=kind)
    A = make_general(m, n, seed=112)
    result = sweep_exact(bins["l2"], "ger", "simple", [m, n, alpha],
                         [x, y, np.asfortranarray(A).ravel(order='F')],
                         sweep=THREAD_SWEEP)
    expected = (A.astype(np.float64) + alpha * np.outer(x, y)).astype(np.float32)
    assert _rel_close(result.reshape(m, n, order='F'), expected)


@pytest.mark.parametrize("m,n", [(4, 128), (64, 8)])
def test_ger_compile_time_rectangular_full_sweep(bins, m, n):
    """The flat compile-time mapping covers short-wide and tall-thin shapes."""
    alpha = -0.75
    x = make_vec(m, seed=210 + m)
    y = make_vec(n, seed=211 + n)
    A = make_general(m, n, seed=212 + m + n)
    result = sweep_exact(
        bins["l2"], "ger", "ct", [m, n, alpha],
        [x, y, np.asfortranarray(A).ravel(order="F")],
        sweep=THREAD_SWEEP,
    )
    expected = (A.astype(np.float64) + alpha * np.outer(x, y)).astype(np.float32)
    assert _rel_close(result.reshape(m, n, order="F"), expected)
