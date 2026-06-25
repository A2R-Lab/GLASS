"""Tensor ⊗ vector contractions (glass::tensor_vec_contract / vec_tensor_vec).

Validates the engine consumers against NumPy einsum oracles across the CONTRACT
axis (K/A/B), ACCUMULATE, TIN_ROW_MAJOR and the SYMMETRIC fast path, on all
three surfaces (block / warp / cgrps), plus thread-count invariance.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(23)
RTOL, ATOL = 1e-2, 1e-3

# (K, A, B): consumer dims (Hxx 14/14/14, Hux 14/7/14), rectangular, and shapes
# whose contracted axis spans the 32-lane boundary (33, 64).
SHAPES = [(14, 14, 14), (8, 8, 8), (5, 5, 5), (14, 7, 14), (3, 4, 5),
          (33, 4, 4), (3, 33, 4), (3, 4, 33), (64, 3, 3), (4, 8, 8)]
SQUARE = [(k, a, b) for (k, a, b) in SHAPES if a == b]  # SYMMETRIC needs A==B
THREADS_SWEEP = (1, 7, 31, 32, 33, 57, 64, 96, 128, 256)


def _w(arr):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    arr.astype(np.float32).ravel(order="C").tofile(f)
    f.close()
    return f.name


def _flatT(T, row_major):
    """(K,A,B) tensor -> flat, each slab in row- or column-major order."""
    K, A, B = T.shape
    out = np.empty(K * A * B, np.float32)
    for k in range(K):
        slab = T[k]
        out[k*A*B:(k+1)*A*B] = (np.ascontiguousarray(slab).ravel("C") if row_major
                                else np.asfortranarray(slab).ravel("F"))
    return out


def _dims(c, K, A, B):
    o0 = A if c == 0 else K
    o1 = B if c == 0 else (B if c == 1 else A)
    return o0, o1


def _run_tvc(binary, surf, th, K, A, B, c, sym, acc, rm, T, v, M):
    tmp = [_w(_flatT(T, rm)), _w(v), _w(np.asfortranarray(M).ravel("F"))]
    try:
        cmd = [str(binary), "tvc", surf, str(th), str(K), str(A), str(B),
               str(c), str(int(sym)), str(int(acc)), str(int(rm))] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        o0, o1 = _dims(c, K, A, B)
        return np.fromstring(r.stdout.strip(), sep=" ").astype(np.float32).reshape(o0, o1, order="F")
    finally:
        for f in tmp:
            os.unlink(f)


def _run_vtv(binary, surf, th, K, A, B, acc, rm, T, u, w, s):
    tmp = [_w(_flatT(T, rm)), _w(u), _w(w), _w(s)]
    try:
        cmd = [str(binary), "vtv", surf, str(th), str(K), str(A), str(B),
               str(int(acc)), str(int(rm))] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return np.fromstring(r.stdout.strip(), sep=" ").astype(np.float32)
    finally:
        for f in tmp:
            os.unlink(f)


def _orc_tvc(T, v, c):
    # lazy: each branch's v has a different length (the contracted axis), so only
    # evaluate the selected einsum — building all three eagerly would mis-size v.
    if c == 0:
        return np.einsum('k,kab->ab', v, T)
    if c == 1:
        return np.einsum('a,kab->kb', v, T)
    return np.einsum('b,kab->ka', v, T)


# ─── tensor_vec_contract: CONTRACT axis × ACCUMULATE × layout, 3 surfaces ─────

@pytest.mark.parametrize("K,A,B", SHAPES)
@pytest.mark.parametrize("c", [0, 1, 2])
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("rm", [False, True])
def test_tvc(bins, K, A, B, c, acc, rm):
    T = RNG.random((K, A, B)).astype(np.float32)
    clen = (K, A, B)[c]
    v = RNG.random(clen).astype(np.float32)
    o0, o1 = _dims(c, K, A, B)
    M0 = RNG.random((o0, o1)).astype(np.float32)
    expected = (_orc_tvc(T, v, c) + (M0 if acc else 0)).astype(np.float32)
    block = _run_tvc(bins["tensor"], "block", 128, K, A, B, c, False, acc, rm, T, v, M0.copy())
    assert np.allclose(block, expected, rtol=RTOL, atol=ATOL), f"\n{block}\nvs\n{expected}"
    # cgrps matches block bit-for-bit; warp matches at 32 threads
    cg = _run_tvc(bins["tensor"], "cgrps", 128, K, A, B, c, False, acc, rm, T, v, M0.copy())
    assert np.array_equal(block, cg), "cgrps != block"
    wp = _run_tvc(bins["tensor"], "warp", 32, K, A, B, c, False, acc, rm, T, v, M0.copy())
    assert np.array_equal(block, wp), "warp != block"


# ─── SYMMETRIC fast path: symmetric slabs → symmetric output ─────────────────

@pytest.mark.parametrize("K,A,B", SQUARE)
@pytest.mark.parametrize("acc", [False, True])
def test_tvc_symmetric(bins, K, A, B, acc):
    Ts = RNG.random((K, A, A)).astype(np.float32)
    Ts = Ts + Ts.transpose(0, 2, 1)            # symmetric in (a,b)
    v = RNG.random(K).astype(np.float32)
    M0 = RNG.random((A, A)).astype(np.float32)
    M0 = M0 + M0.T
    expected = (np.einsum('k,kab->ab', v, Ts) + (M0 if acc else 0)).astype(np.float32)
    for surf, th in [("block", 128), ("cgrps", 96), ("warp", 32)]:
        r = _run_tvc(bins["tensor"], surf, th, K, A, A, 0, True, acc, False, Ts, v, M0.copy())
        assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), f"{surf} value mismatch"
        assert np.allclose(r, r.T, rtol=RTOL, atol=ATOL), f"{surf} output not symmetric"


# ─── thread-count invariance (block), bit-identical across the 32 boundary ───

@pytest.mark.parametrize("K,A,B", [(14, 14, 14), (33, 4, 4), (64, 3, 3), (14, 7, 14)])
def test_tvc_thread_invariance(bins, K, A, B):
    T = RNG.random((K, A, B)).astype(np.float32)
    v = RNG.random(K).astype(np.float32)
    M0 = RNG.random((A, B)).astype(np.float32)
    expected = (np.einsum('k,kab->ab', v, T) + M0).astype(np.float32)
    outs = []
    for t in THREADS_SWEEP:
        r = _run_tvc(bins["tensor"], "block", t, K, A, B, 0, False, True, False, T, v, M0.copy())
        assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), f"threads={t} vs oracle"
        outs.append(r)
    for t, r in zip(THREADS_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"


# ─── vec_tensor_vec: s[k] = u^T T_k w ────────────────────────────────────────

@pytest.mark.parametrize("K,A,B", SHAPES)
@pytest.mark.parametrize("rm", [False, True])
def test_vtv(bins, K, A, B, rm):
    T = RNG.random((K, A, B)).astype(np.float32)
    u = RNG.random(A).astype(np.float32)
    w = RNG.random(B).astype(np.float32)
    expected = np.einsum('a,kab,b->k', u, T, w).astype(np.float32)
    block = _run_vtv(bins["tensor"], "block", 128, K, A, B, False, rm, T, u, w, np.zeros(K, np.float32))
    assert np.allclose(block, expected, rtol=RTOL, atol=ATOL)
    cg = _run_vtv(bins["tensor"], "cgrps", 128, K, A, B, False, rm, T, u, w, np.zeros(K, np.float32))
    assert np.array_equal(block, cg), "cgrps != block"
    wp = _run_vtv(bins["tensor"], "warp", 32, K, A, B, False, rm, T, u, w, np.zeros(K, np.float32))
    assert np.array_equal(block, wp), "warp != block"


@pytest.mark.parametrize("K,A,B", [(14, 14, 14), (4, 8, 8), (64, 3, 3)])
def test_vtv_thread_invariance(bins, K, A, B):
    T = RNG.random((K, A, B)).astype(np.float32)
    u = RNG.random(A).astype(np.float32)
    w = RNG.random(B).astype(np.float32)
    expected = np.einsum('a,kab,b->k', u, T, w).astype(np.float32)
    outs = []
    for t in THREADS_SWEEP:
        r = _run_vtv(bins["tensor"], "block", t, K, A, B, False, False, T, u, w, np.zeros(K, np.float32))
        assert np.allclose(r, expected, rtol=RTOL, atol=ATOL), f"threads={t} vs oracle"
        outs.append(r)
    for t, r in zip(THREADS_SWEEP[1:], outs[1:]):
        assert np.array_equal(outs[0], r), f"thread-count non-invariance at {t}"
