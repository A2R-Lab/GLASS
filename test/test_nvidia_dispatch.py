"""Round-2 auto-dispatch correctness tests.

Companion to test_l3_nvidia.py (which tests the SIMT batched APIs from
l3_simt.cuh). This module targets the round-2 additions:

  * Gap A — glass::nvidia::block::gemv<> auto-dispatches SIMT vs cuBLASDx
  * Gap B — gemv_strided<>        auto-dispatches; SIMT uses stride directly
  * Gap C — gemm_strided<>        auto-dispatches; SIMT skips compact-pack
  * Gap D — gemm<...,LB=row_major,...> maps onto SIMT TRANSPOSE_B=true

The test binary requires cuBLASDx for the gemm_cublas op (compiles via DEFINE),
so the whole module skips when MATHDX_ROOT isn't set (see conftest.py).
"""

import subprocess

import pytest


def _run(nvidia_bin, op):
    res = subprocess.run([str(nvidia_bin), op], capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


@pytest.mark.parametrize("op", [
    "gemm_simt",      # 6x6x6 auto-routes to SIMT (no DEFINE); matches CPU
    "gemm_cublas",    # 16x16x16 routes to cuBLASDx via shipped DEFINE; bit-parity
    "gemm_transb",    # Gap D — LB=row_major maps to SIMT TRANSPOSE_B=true
    "gemv_simt",      # Gap A — gemv 6x6 SIMT
    "strided_gemv",   # Gap B — non-tight stride
    "strided_gemm",   # Gap C — non-tight A_RS/B_RS
    "beta0_poison",   # beta==0 write-only through BOTH routes (NaN-poisoned C);
                      # pins the cuBLASDx vendor behavior (MathDx 26.03) + SIMT beta_blend
])
def test_dispatch_op(bin_nvidia_dispatch, op):
    rc, stdout, stderr = _run(bin_nvidia_dispatch, op)
    assert rc == 0, f"{op} returned {rc}\nstdout: {stdout}\nstderr: {stderr}"
    assert "PASS" in stdout, f"{op} did not print PASS:\nstdout: {stdout}"


def test_dispatch_query(bin_nvidia_dispatch):
    """print_dispatch<> reports SIMT for small shapes and cuBLASDx for large."""
    rc, stdout, _ = _run(bin_nvidia_dispatch, "dispatch_q")
    assert rc == 0
    # 6x6x6 → SIMT, 16x16x16 → cuBLASDx (matches the shipped tuning + heuristic).
    assert "glass::nvidia::block::gemm<T,6,6,6" in stdout and "SIMT" in stdout
    assert "glass::nvidia::block::gemm<T,16,16,16" in stdout and "cuBLASDx" in stdout

# ─── moved from test_l3.py 2026-08-06 (shard partition: these ride the nvidia TU) ───
import numpy as np
from conftest import bin_l3_nvidia, run_op, THREAD_SWEEP_CORE  # noqa: F401
RNG = np.random.default_rng(20260806)
RTOL, ATOL = 1e-4, 1e-5

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
    As = [RNG.random((m, k)).astype(np.float32) for _ in range(batch)]
    Bs = [RNG.random((k, n)).astype(np.float32) for _ in range(batch)]
    Cs = [RNG.random((m, n)).astype(np.float32) for _ in range(batch)]
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
    As = [RNG.random((m, k)).astype(np.float32) for _ in range(batch)]
    Bs = [RNG.random((k, n)).astype(np.float32) for _ in range(batch)]
    Cs = [RNG.random((m, n)).astype(np.float32) for _ in range(batch)]
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
    A = RNG.random((m, k)).astype(np.float32)
    Bs = [RNG.random((k, n)).astype(np.float32) for _ in range(batch)]
    Cs = [RNG.random((m, n)).astype(np.float32) for _ in range(batch)]
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
    # 4×4×4, B padded K*N=16 -> 24 (8 floats slack), C padded M*N=16 -> 20 (4 floats slack)
    ("gemm_strided_padded_4x4x4_b4_bs24_cs20", 4, 4, 4, 4, 24, 20),
    # 3×5×7 (standard: A 3×7, B 7×5, C 3×5), B packed K*N=35 -> 50, C packed M*N=15 -> 28
    ("gemm_strided_padded_3x5x7_b3_bs50_cs28", 3, 5, 7, 3, 50, 28),
])
def test_gemm_strided_batched_1d_padded(bin_l3_nvidia, op, m, n, k, batch,
                                         b_stride, c_stride):
    """Strided variant with non-default B_STRIDE / C_STRIDE — verifies the
    `b * STRIDE` indexing inside the kernel, not just the tightly-packed default.
    Standard convention: shared A is m×k, each B is k×n, each C is m×n."""
    alpha, beta = 1.5, 0.3
    A = RNG.random((m, k)).astype(np.float32)

    # Build padded B: each batch occupies b_stride floats; only first k*n matter.
    B_padded = RNG.random(batch * b_stride).astype(np.float32)
    # Build padded C: each batch occupies c_stride floats; only first m*n matter.
    C_padded = RNG.random(batch * c_stride).astype(np.float32)

    # Extract packed col-major B[b] (k×n) and C[b] (m×n) for the reference.
    Bs = [B_padded[b*b_stride : b*b_stride + k*n].reshape(k, n, order='F')
          for b in range(batch)]
    Cs0 = [C_padded[b*c_stride : b*c_stride + m*n].reshape(m, n, order='F').copy()
           for b in range(batch)]

    result = run_op(bin_l3_nvidia, op, "simple",
                    args=[alpha, beta],
                    inputs=[np.asfortranarray(A).ravel(order='F'),
                            B_padded, C_padded])

    # Build expected output with padding bytes preserved.
    expected = C_padded.copy()
    for b in range(batch):
        new_C = (alpha * A @ Bs[b] + beta * Cs0[b]).astype(np.float32)
        expected[b*c_stride : b*c_stride + m*n] = \
            np.asfortranarray(new_C).ravel(order='F')

    # Result has length batch*c_stride (we print the full padded buffer so we
    # can verify the kernel did NOT write to the padding slots either).
    assert len(result) == batch * c_stride
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_batched_indexed ─────────────────────────────────────────────────────
# C[c_idx[p]] = A[a_idx[p]] @ B[b_idx[p]], 4x4 col-major, selected by index lists.
# numpy does each indexed product independently as the reference.

@pytest.mark.parametrize("threads", THREAD_SWEEP_CORE)
@pytest.mark.parametrize("pairs,a_mats,b_mats,c_mats", [
    (1, 1, 1, 1),
    (4, 2, 3, 4),    # repeated/aliased a_idx,b_idx; distinct c_idx
    (8, 5, 5, 8),
])
def test_gemm_batched_indexed(bins, pairs, a_mats, b_mats, c_mats, threads):
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
        args=[threads, DIM, DIM, pairs, a_mats, b_mats, c_mats],
        inputs=[a_idx.astype(np.float32), b_idx.astype(np.float32),
                c_idx.astype(np.float32), A_flat, B_flat])

    MAT = DIM * DIM
    expected = np.zeros(c_mats * MAT, dtype=np.float32)
    for p in range(pairs):
        prod = A_mats[a_idx[p]] @ B_mats[b_idx[p]]
        base = int(c_idx[p]) * MAT
        expected[base:base + MAT] = np.asfortranarray(prod).ravel(order='F')
    assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)


# ─── gemm_batched_indexed: TRANSPOSE_A / TRANSPOSE_B ──────────────────────────
# Distinct c_idx, plain overwrite, but the left and/or right factor is read
# transposed. Reference applies .T to the corresponding numpy operand.

@pytest.mark.parametrize("threads", THREAD_SWEEP_CORE)
@pytest.mark.parametrize("op,ta,tb", [
    ("indexed_bgemm_4_ta", True, False),
    ("indexed_bgemm_4_tb", False, True),
])
@pytest.mark.parametrize("pairs,a_mats,b_mats,c_mats", [
    (1, 1, 1, 1),
    (4, 2, 3, 4),
    (8, 5, 5, 8),
])
def test_gemm_batched_indexed_transpose(bins, op, ta, tb, pairs, a_mats, b_mats, c_mats, threads):
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
        args=[threads, DIM, DIM, pairs, a_mats, b_mats, c_mats],
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


# ─── gemm_batched_indexed: ATOMIC_C (overlapping c_idx) ───────────────────────
# Several pairs SHARE a c_idx slot (a parent block); the atomic path must
# scatter-ADD their products. Caller pre-zeros C; reference is a numpy
# scatter-add into the shared C slots. parent_of maps pair -> c slot.

@pytest.mark.parametrize("threads", THREAD_SWEEP_CORE)
@pytest.mark.parametrize("op,ta", [
    ("indexed_bgemm_4_atomic", False),     # C += A · B
    ("indexed_bgemm_4_ta_atomic", True),   # C += Aᵀ · B  (backward Xᵀ·M·X→parent)
])
def test_gemm_batched_indexed_atomic(bins, op, ta, threads):
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
        args=[threads, DIM, DIM, pairs, a_mats, b_mats, c_mats],
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


def test_gemm_batched(bin_nvidia_dispatch):
    """Pointer-array batched GEMM (BATCH=4, 8x8x8) vs in-driver host reference."""
    rc, stdout, stderr = _run(bin_nvidia_dispatch, "gemm_batched")
    assert rc == 0 and "PASS" in stdout, stderr
