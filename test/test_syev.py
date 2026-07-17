"""test_syev.py — glass::syev (cyclic-Jacobi symmetric eigendecomposition) and
glass::eig_clamp (eigenvalue clamp + reconstruct) vs numpy.

Coverage: syev vs np.linalg.eigh across sizes (eigenvalues, orthogonality,
V diag(W) Vᵀ reconstruction), per-column eigenvector match up to SIGN on a
well-separated spectrum, the compile-time-N overload, repeated eigenvalues
(identity + rank-1), ill-conditioned SPD (cond=1e6), negative-definite and
indefinite inputs, eig_clamp SPD-ness + numpy-clamped reconstruction, and the
THREAD_SWEEP byte-identical invariance rule.
"""
import numpy as np
import pytest

from conftest import run_op, THREAD_SWEEP, make_spd

SIZES = [1, 2, 3, 4, 7, 12, 16, 32]


def make_sym(n, seed=0):
    """Random symmetric n x n (generally indefinite), float32."""
    G = np.random.default_rng(seed).standard_normal((n, n))
    return ((G + G.T) / 2).astype(np.float32)


def make_sym_spectrum(n, eigs, seed=0):
    """Symmetric matrix with a prescribed spectrum: Q diag(eigs) Qᵀ."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q @ np.diag(np.asarray(eigs, dtype=np.float64)) @ Q.T).astype(np.float32)


def run_syev(bins, n, threads, A, op="syev"):
    out = run_op(bins["syev"], op, "simple", args=[n, threads],
                 inputs=[np.asfortranarray(A).ravel(order="F")])
    W, V = out[0], out[1].reshape(n, n, order="F")
    return W, V


def run_eig_clamp(bins, n, threads, A, eps):
    out = run_op(bins["syev"], "eig_clamp", "simple", args=[n, threads, eps],
                 inputs=[np.asfortranarray(A).ravel(order="F")])
    return out.reshape(n, n, order="F")


def check_eig(A, W, V, rtol=1e-3):
    """Shared correctness net: ascending order, eigenvalues vs eigh,
    orthogonality of V, and the V diag(W) Vᵀ reconstruction."""
    n = A.shape[0]
    wref = np.linalg.eigvalsh(A.astype(np.float64))
    scale = max(np.abs(wref).max(), 1e-6)
    assert np.all(np.diff(W) >= 0), "eigenvalues not ascending"
    assert np.allclose(W, wref, rtol=rtol, atol=rtol * scale), \
        f"eigenvalues off: {W} vs {wref}"
    V64 = V.astype(np.float64)
    assert np.allclose(V64.T @ V64, np.eye(n), atol=5e-3), "V not orthogonal"
    R = (V64 * W.astype(np.float64)) @ V64.T
    assert np.allclose(R, A, atol=rtol * scale, rtol=rtol), "V W Vᵀ != A"


@pytest.mark.parametrize("n", SIZES)
def test_syev_vs_numpy(bins, n):
    A = make_sym(n, seed=11 * n + 1)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)


@pytest.mark.parametrize("n", SIZES)
def test_syev_eigenvectors_match_up_to_sign(bins, n):
    # Well-separated spectrum 1..n so the eigh comparison is stable: each GPU
    # column must line up with the reference column up to SIGN, i.e.
    # diag(|V_refᵀ V_gpu|) ≈ 1.
    A = make_sym_spectrum(n, np.arange(1, n + 1), seed=7 * n + 3)
    W, V = run_syev(bins, n, 128, A)
    wref, vref = np.linalg.eigh(A.astype(np.float64))
    assert np.allclose(W, wref, rtol=1e-3, atol=1e-3 * n)
    d = np.abs(np.diag(vref.T @ V.astype(np.float64)))
    assert np.allclose(d, 1.0, atol=1e-2), f"eigenvector columns misaligned: {d}"


@pytest.mark.parametrize("n", SIZES)
def test_syev_compile_time_overload(bins, n):
    A = make_sym(n, seed=5 * n + 2)
    W, V = run_syev(bins, n, 64, A, op="syev_ct")
    check_eig(A, W, V)


def test_syev_repeated_eigenvalues(bins):
    # identity + rank-1: eigenvalues {1 (x n-1), 1 + ||u||²} — a degenerate
    # cluster. Per-column vector match is ill-posed here; the reconstruction +
    # orthogonality net still fully pins correctness.
    n = 12
    u = np.random.default_rng(3).standard_normal(n)
    A = (np.eye(n) + np.outer(u, u)).astype(np.float32)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    wref = np.sort(np.concatenate([np.ones(n - 1), [1 + u @ u]]))
    assert np.allclose(W, wref, rtol=1e-3, atol=1e-3)


def test_syev_ill_conditioned(bins):
    n = 16
    A = make_spd(n, seed=9, cond=1e6)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    assert np.all(W > 0), "SPD input must give positive eigenvalues"


def test_syev_negative_definite(bins):
    n = 12
    A = (-make_spd(n, seed=21)).astype(np.float32)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    assert np.all(W < 0), "negative-definite input must give negative eigenvalues"


def test_syev_indefinite(bins):
    n = 16
    eigs = np.linspace(-5.0, 5.0, n)          # mixed signs, well separated
    A = make_sym_spectrum(n, eigs, seed=13)
    W, V = run_syev(bins, n, 128, A)
    check_eig(A, W, V)
    assert (W < 0).sum() == (eigs < 0).sum(), "inertia mismatch"


@pytest.mark.parametrize("n", [4, 12, 16])
def test_eig_clamp_indefinite_becomes_spd(bins, n):
    eps = 0.05
    A = make_sym(n, seed=17 * n + 5)          # generally indefinite
    out = run_eig_clamp(bins, n, 128, A, eps)
    # exact symmetry by construction (canonical FMA operand order)
    assert np.array_equal(out, out.T), "clamped output not bit-symmetric"
    w_out = np.linalg.eigvalsh(out.astype(np.float64))
    assert np.all(w_out >= eps - 1e-3), f"not SPD at floor eps: {w_out}"
    # equals the numpy-clamped reconstruction
    w, v = np.linalg.eigh(A.astype(np.float64))
    ref = (v * np.maximum(w, eps)) @ v.T
    scale = max(np.abs(w).max(), 1.0)
    assert np.allclose(out, ref, rtol=1e-2, atol=1e-2 * scale)


def test_eig_clamp_spd_passthrough(bins):
    # eigenvalues already above the floor: the clamp must return A (round-off only).
    n = 7
    A = make_spd(n, seed=8)                   # eigenvalues >= n by construction
    out = run_eig_clamp(bins, n, 64, A, 1e-3)
    assert np.allclose(out, A, rtol=1e-3, atol=1e-3 * np.abs(A).max())


@pytest.mark.parametrize("n", [7, 16])
def test_syev_thread_invariance(bins, n):
    A = make_sym(n, seed=100 + n)
    refW = refV = None
    for th in THREAD_SWEEP:
        W, V = run_syev(bins, n, th, A)
        if refW is None:
            refW, refV = W, V
        else:
            assert np.array_equal(W, refW), f"W non-invariant at threads={th}"
            assert np.array_equal(V, refV), f"V non-invariant at threads={th}"


def test_eig_clamp_thread_invariance(bins):
    n = 7
    A = make_sym(n, seed=42)
    ref = None
    for th in THREAD_SWEEP:
        out = run_eig_clamp(bins, n, th, A, 0.05)
        if ref is None:
            ref = out
        else:
            assert np.array_equal(out, ref), f"non-invariant at threads={th}"


# ═══ eigh + psd_project (fixed-sweep Jacobi; see src/base/L3/eigh.cuh) ═══════
#
# The deterministic sibling of syev: FIXED sweeps, round-robin schedule,
# UNSORTED spectrum, dtype-templated driver ops. Gates per the 2026-07-17
# handoff: (1) A·V = V·diag(W); (2) V orthonormal; (3) parity vs the
# jacobi_study.py oracle at the SAME sweep count (near-bitwise f64, ~2e-6 f32);
# (4) psd_project symmetric PSD + matches the oracle clip-reconstruct;
# (5) thread-count bit-invariance + run-twice determinism.
#
# The oracle below is jacobi_study.py's round_robin_rounds()/jacobi_eigh()
# ported VERBATIM (GATO so_sqp_prototype). One known divergence, documented in
# eigh.cuh: the device applies a round's disjoint pairs PHASED (rows then
# cols), the oracle serially — identical in exact arithmetic, so parity is
# tolerance-gated, not bitwise.

import subprocess as _sp
import tempfile as _tf
import os as _os

EIGH_SIZES = [4, 7, 12, 14, 18, 21, 33]      # must match EIGH_SIZES in the driver
_EIGH_SWEEPS = {"f32": 6, "f64": 12}          # must match eigh_sweeps<T>()
_EIGH_NPDT = {"f32": np.float32, "f64": np.float64}


def _rr_rounds(n):
    """jacobi_study.py round_robin_rounds(), verbatim."""
    m = n + (n % 2)
    idx = list(range(m))
    rounds = []
    for _ in range(m - 1):
        pairs = []
        for i in range(m // 2):
            p, q = idx[i], idx[m - 1 - i]
            if p < n and q < n:
                pairs.append((min(p, q), max(p, q)))
        rounds.append(pairs)
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return rounds


def _jacobi_eigh(A, sweeps, dtype=np.float64):
    """jacobi_study.py jacobi_eigh(), verbatim (minus the off-norm history)."""
    A = np.array(A, dtype=dtype)
    n = A.shape[0]
    V = np.eye(n, dtype=dtype)
    for _ in range(sweeps):
        for pairs in _rr_rounds(n):
            for (p, q) in pairs:
                apq = A[p, q]
                if apq == 0.0:
                    continue
                theta = (A[q, q] - A[p, p]) / (dtype(2.0) * apq)
                t = np.sign(theta) / (abs(theta) + np.sqrt(dtype(1.0) + theta * theta))
                if theta == 0.0:
                    t = dtype(1.0)
                c = dtype(1.0) / np.sqrt(dtype(1.0) + t * t)
                s = t * c
                rp, rq = A[p, :].copy(), A[q, :].copy()
                A[p, :] = c * rp - s * rq
                A[q, :] = s * rp + c * rq
                cp, cq = A[:, p].copy(), A[:, q].copy()
                A[:, p] = c * cp - s * cq
                A[:, q] = s * cp + c * cq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
    return np.diag(A).copy(), V


def _run_eigh_raw(bins, op, n, threads, dtype, extra, A):
    """Dtype-aware driver invocation (run_op parses float32 only); returns the
    raw stdout lines parsed at the native dtype — exact text comparison happens
    on the parsed bits via the invariance/determinism gates."""
    fh = _tf.NamedTemporaryFile(suffix=".bin", delete=False)
    np.asfortranarray(A).ravel(order="F").astype(np.float32).tofile(fh)
    fh.close()
    try:
        cmd = [str(bins["syev"]), op, "simple", str(n), str(threads), dtype] +               [str(x) for x in extra] + [fh.name]
        r = _sp.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Binary failed:\n{r.stdout}\n{r.stderr}")
        return [np.fromstring(l, sep=" ").astype(_EIGH_NPDT[dtype])
                for l in r.stdout.strip().split("\n")]
    finally:
        _os.unlink(fh.name)


def run_eigh(bins, n, threads, dtype, A):
    W, Vf = _run_eigh_raw(bins, "eigh", n, threads, dtype, [], A)
    return W, Vf.reshape(n, n, order="F")


def run_psd_project(bins, n, threads, dtype, eps, A):
    (Af,) = _run_eigh_raw(bins, "psd_project", n, threads, dtype, [eps], A)
    return Af.reshape(n, n, order="F")


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("n", EIGH_SIZES)
def test_eigh_decomposition(bins, n, dtype):
    """Gate 1+2: A·V = V·diag(W) and V orthonormal (W is unsorted by design)."""
    A = make_sym(n, seed=n)
    W, V = run_eigh(bins, n, 128, dtype, A)
    A64, V64 = A.astype(np.float64), V.astype(np.float64)
    scale = max(np.abs(W).max(), 1e-6)
    tol = 2e-5 if dtype == "f32" else 1e-11
    assert np.allclose(A64 @ V64, V64 * W.astype(np.float64), atol=tol * scale), \
        "A V != V diag(W)"
    assert np.allclose(V64.T @ V64, np.eye(n), atol=tol * 10), "V not orthonormal"
    # And the spectrum matches numpy's (sorted, since ours is unsorted).
    wref = np.linalg.eigvalsh(A64)
    assert np.allclose(np.sort(W), wref, atol=tol * scale, rtol=tol), "spectrum off"


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("n", [12, 14, 18, 21])
def test_eigh_oracle_parity(bins, n, dtype):
    """Gate 3: parity vs the jacobi_study.py oracle at the SAME sweep count —
    same schedule, same formulas, so W (unsorted!) and V match column-for-column
    including sign. f64 near-bitwise; f32 to rounding (the handoff's ~2e-6)."""
    A = make_sym(n, seed=100 + n)
    W, V = run_eigh(bins, n, 128, dtype, A)
    npdt = _EIGH_NPDT[dtype]
    Wo, Vo = _jacobi_eigh(A.astype(np.float64), _EIGH_SWEEPS[dtype], dtype=npdt)
    scale = max(np.abs(Wo).max(), 1e-6)
    tol = 2e-6 if dtype == "f32" else 1e-9
    assert np.allclose(W, Wo, atol=tol * scale, rtol=tol), \
        f"eigenvalue parity vs oracle: max diff {np.abs(W - Wo).max():.3e}"
    assert np.allclose(V, Vo, atol=tol * 10), \
        f"eigenvector parity vs oracle: max diff {np.abs(V - Vo).max():.3e}"


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("n", EIGH_SIZES)
def test_psd_project(bins, n, dtype):
    """Gate 4: output symmetric (bit-exact, canonical-order reconstruction),
    PSD at the floor, and matches the oracle's clip-reconstruct."""
    A = make_sym(n, seed=200 + n)          # indefinite in general
    eps = 1e-6 * (1.0 + float(np.abs(np.diag(A)).max()))   # the consumer's rule
    P = run_psd_project(bins, n, 128, dtype, eps, A)
    assert np.array_equal(P, P.T), "psd_project output not bit-exactly symmetric"
    w = np.linalg.eigvalsh(P.astype(np.float64))
    scale = max(np.abs(w).max(), 1e-6)
    slack = (2e-5 if dtype == "f32" else 1e-11) * scale
    assert w.min() >= eps - slack, f"not PSD at the floor: min eig {w.min():.3e} < {eps:.3e}"
    # Oracle: clip-reconstruct from the SAME fixed-sweep Jacobi.
    npdt = _EIGH_NPDT[dtype]
    Wo, Vo = _jacobi_eigh(A.astype(np.float64), _EIGH_SWEEPS[dtype], dtype=npdt)
    Po = (Vo.astype(np.float64) * np.maximum(Wo.astype(np.float64), eps)) @ Vo.T.astype(np.float64)
    tol = 2e-6 if dtype == "f32" else 1e-9
    assert np.allclose(P.astype(np.float64), Po, atol=tol * scale, rtol=tol), \
        f"psd_project vs oracle: max diff {np.abs(P - Po).max():.3e}"


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("op,extra", [("eigh", []), ("psd_project", [1e-5])])
@pytest.mark.parametrize("n", [12, 21])
def test_eigh_thread_invariance_and_determinism(bins, n, op, extra, dtype):
    """Gate 5: byte-identical output across the thread sweep AND across two
    runs at the same thread count (fixed schedule/sweeps, no reductions)."""
    A = make_sym(n, seed=300 + n)
    ref = None
    for threads in list(THREAD_SWEEP) + [32]:   # full sweep incl. 1 thread; 32 repeats for run-twice
        outs = _run_eigh_raw(bins, op, n, threads, dtype, extra, A)
        flat = np.concatenate([o.ravel() for o in outs])
        if ref is None:
            ref = flat
        else:
            assert np.array_equal(ref, flat), \
                f"{op}: output not bit-identical at threads={threads}"
