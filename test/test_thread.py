"""glass::thread:: surface tests — one problem per THREAD.

The tier's oracle is the shared-body construction: every `thread::` op delegates
to the same `*_impl` body as `glass::`, via `ThreadBarrier{rank=0, size=1, no-op
sync}`. So the `thread` model and a `block1` model (`<<<P, 1>>>`, the block-scoped
op degenerated to one thread) run the identical algorithm over the identical
operand order and must agree to within FMA-contraction jitter — a few ULP, not a
loose tolerance (see `_assert_ulp_equal` for why bit-exactness across the two
template instantiations is NOT guaranteed, and the -fmad=false experiment that
pins the residual on contraction). We assert that tight ULP bound, and separately
check both against a numpy/scipy oracle so a shared bug in both surfaces can't
pass silently.

`dot` is the deliberate exception: block-scoped `dot` reduces with a halving
TREE while `thread::dot` accumulates serially, so the two differ in summation
ORDER by design and are compared to float tolerance only. That asymmetry is why
the tier ships no `_fast`/`_lowmem` twins (a single thread has no reduction
strategy to choose) — see the `thread::` block in src/base/L1/dot.cuh.

P is >32 and NOT a multiple of the driver's TPB, so problems span multiple warps
and several blocks with a ragged tail — the shape that catches a stray
block-wide `__syncthreads()` inside a thread:: op (once the tail block's
out-of-range threads return, such a barrier has divergent participation ⇒
UB/hang). That is a real bug this suite found in `trsv_impl`.

COVERAGE:
  * Both dtypes (f32/f64) — the tier claims a register-residency ceiling for
    BOTH (CLAUDE.md); each op runs under both instantiations.
  * Sizes bracket the measured N<=7 ceiling (N=8 still computes correctly, it
    just spills `A` to local memory).
  * trsv sweeps its full flag surface (Lower/Upper × Unit/NonUnit × trans) and
    gemv sweeps (trans × row-major) — the thread:: overloads only forward these
    to the shared `*_impl`, but we exercise the combinations rather than trust
    the forwarding.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest
import scipy.linalg

RNG = np.random.default_rng(11)   # reseeded per test by _seed_rng below


@pytest.fixture(autouse=True)
def _seed_rng(request):
    """Reseed the module RNG from the test's nodeid so every test draws the SAME
    inputs whether it runs alone or in the full suite — a failure reproduces in
    isolation. (crc32, not hash(): hash() is salted per process.)"""
    global RNG
    import zlib
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))

# 4..7 register-resident; 8 is past the measured ceiling (correct, just spilled).
SIZES = [4, 5, 6, 7, 8]
DTYPES = ["f32", "f64"]
# >32 (multi-warp) and not a multiple of TPB=64 (ragged tail block).
NPROB = 100

# Per-dtype tolerances: f32 accumulates visibly, f64 is near-exact.
_TOL = {"f32": dict(rtol=1e-3, atol=1e-3), "f64": dict(rtol=1e-10, atol=1e-10)}
_NPDT = {"f32": np.float32, "f64": np.float64}


# ─── local runner (dtype- and flag-aware; conftest.run_op is float32-only) ────

def _run(binary, op, model, dtype, n, P, inputs, flags=()):
    """Write float32 .bin inputs, invoke the driver, parse stdout at `dtype`.

    Inputs are always float32 on disk (the driver widens to T on load); the
    output is parsed at the native dtype so the bit-identical check stays exact
    even in f64.
    """
    tmp = []
    try:
        for arr in inputs:
            fh = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(fh)
            fh.close()
            tmp.append(fh.name)
        cmd = [str(binary), op, model, dtype, str(n), str(P)] + [str(x) for x in flags] + tmp
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Binary failed:\n{r.stderr}")
        return np.fromstring(r.stdout.strip(), sep=" ").astype(_NPDT[dtype])
    finally:
        for t in tmp:
            os.unlink(t)


def _both(bins, op, dtype, n, P, inputs, flags=()):
    """Run the same inputs through both models; return (thread_out, block1_out)."""
    t = _run(bins["thread"], op, "thread", dtype, n, P, inputs, flags)
    b = _run(bins["thread"], op, "block1", dtype, n, P, inputs, flags)
    return t, b


def _colmajor_batch(mats):
    """Flatten a list of matrices column-major and concatenate (`.T.ravel()`)."""
    return np.concatenate([m.T.ravel() for m in mats]).astype(np.float32)


def _assert_ulp_equal(t, b, tag, max_ulp=4):
    """thread:: and a 1-thread block run the same *algorithm* — same C++ body,
    same operand order — but they are DIFFERENT template instantiations
    (ThreadBarrier's no-op sync removes the optimization fences BlockBarrier
    has), so nvcc may contract the multiply-add chains differently and the two
    can disagree in the LAST ULP on borderline-rounding inputs. Measured on
    sm_120/nvcc 13.2: potrf/posv f64 differ on ~0.1% of elements, all by
    exactly 1 ulp; compiling both with -fmad=false restores bit-identity,
    proving FMA contraction is the sole source. So the invariant we can and do
    hold is ULP-bounded agreement, not bit-equality across tiers. BOUNDED-CHAIN
    ops only (elementwise maps and fixed-length contractions: gemv/gemm/syrk/
    tensor/congruence/L1): ops whose chains divide by computed pivots or sqrt
    computed values (potrf/ldlt and every solve: trsv/trsm/posv/potrs/
    ldlt_solve/inv/riccati) amplify the seed condition- and BUILD-dependently
    (adding an unrelated header to glass.cuh moved which op tripped a fixed
    bound), so they use _assert_close_tight instead."""
    assert t.shape == b.shape, f"{tag}: shape mismatch {t.shape} vs {b.shape}"
    if np.array_equal(t, b):
        return
    ib, it = b.view(f"i{b.itemsize}"), t.view(f"i{t.itemsize}")
    ulp = np.abs(np.where(ib == it, 0, ib - it))
    if int(ulp.max()) > max_ulp:
        i = int(np.argmax(ulp))
        raise AssertionError(
            f"{tag}: thread:: differs from a 1-thread block by {int(ulp.max())} ulp "
            f"(> {max_ulp}) at element {i}: {t[i]!r} vs {b[i]!r}. More than FMA-"
            f"contraction jitter — the thread tier has diverged from the shared "
            f"*_impl body (or the block path is not thread-count invariant)."
        )


def _assert_close_tight(t, b, tag, dtype):
    """Cross-model gate for SOLVE-COMPOSED ops (trsv legs, substitutions,
    eliminations): the few-ulp contraction seed (see _assert_ulp_equal) is
    amplified by the solve's conditioning, and the amplification SHIFTS BETWEEN
    BUILDS (nvcc's contraction choices depend on inlining context — adding an
    unrelated header to glass.cuh moved riccati's f64 divergence past a fixed
    1024-ulp bound). A fixed ULP budget is therefore build-fragile here; the
    robust invariant is a tight relative tolerance — still 5-9 orders below the
    oracle tolerances, while real divergence (wrong operand order/algorithm)
    is at 1e-3+ relative."""
    tol = {"f32": 1e-4, "f64": 1e-11}[dtype]
    scale = max(float(np.abs(b).max()), 1.0)
    np.testing.assert_allclose(
        t, b, rtol=tol, atol=tol * scale,
        err_msg=f"{tag}: thread:: vs 1-thread block beyond solve-amplified "
                f"contraction jitter — the thread tier has diverged from the "
                f"shared *_impl body")


def _spd(n):
    """A well-conditioned SPD matrix (symmetric ⇒ layout-agnostic)."""
    M = RNG.standard_normal((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


def _tri(n, lower):
    """A well-conditioned triangular matrix with a strong diagonal."""
    A = RNG.standard_normal((n, n)).astype(np.float32)
    A = np.tril(A) if lower else np.triu(A)
    A[np.diag_indices(n)] = np.abs(A[np.diag_indices(n)]) + n
    return A.astype(np.float32)


# ─── dot (tolerance only: tree reduce vs serial accumulate — different by design) ──

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_dot(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    y = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "dot", dtype, n, NPROB, [x, y])

    # Oracle in float64: the device widens the float32 inputs to T, so the f64
    # path is more accurate than a float32-precision reference would be.
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    want = np.array([x64[p*n:(p+1)*n] @ y64[p*n:(p+1)*n] for p in range(NPROB)])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    # NOT bit-identical to block1 by design; both must still hit the same answer.
    np.testing.assert_allclose(t, b, **_TOL[dtype])


# ─── gemv (sweeps trans × row-major) ──────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("rowmajor", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_gemv(bins, n, rowmajor, trans, dtype):
    mats = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    # ROW_MAJOR is a STORAGE flag: same logical matrix, different byte order.
    ravel = (lambda m: m.ravel()) if rowmajor else (lambda m: m.T.ravel())
    A = np.concatenate([ravel(m) for m in mats]).astype(np.float32)
    flags = (int(trans), int(rowmajor))
    t, b = _both(bins, "gemv", dtype, n, NPROB, [A, x], flags)

    x64 = x.astype(np.float64)
    want = np.concatenate([
        (mats[p].T if trans else mats[p]).astype(np.float64) @ x64[p*n:(p+1)*n]
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, f"gemv(trans={trans},rowmajor={rowmajor})")


# ─── gemm ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_gemm(bins, n, dtype):
    ma = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    mb = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    A, B = _colmajor_batch(ma), _colmajor_batch(mb)
    t, b = _both(bins, "gemm", dtype, n, NPROB, [A, B])

    want = np.concatenate([
        (ma[p].astype(np.float64) @ mb[p].astype(np.float64)).T.ravel()
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "gemm")


# ─── potrf ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_potrf(bins, n, dtype):
    mats = [_spd(n) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    t, b = _both(bins, "potrf", dtype, n, NPROB, [A])

    for p in range(NPROB):
        got = t[p*n*n:(p+1)*n*n].reshape(n, n).T      # column-major -> row-major view
        want = np.linalg.cholesky(mats[p].astype(np.float64))
        np.testing.assert_allclose(np.tril(got), np.tril(want), **_TOL[dtype])
    _assert_close_tight(t, b, "potrf", dtype)


# ─── trsv (sweeps Lower/Upper × Unit/NonUnit × trans) ─────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("unit", [False, True])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", SIZES)
def test_trsv(bins, n, lower, unit, trans, dtype):
    mats = [_tri(n, lower) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    flags = (int(lower), int(unit), int(trans))
    t, b = _both(bins, "trsv", dtype, n, NPROB, [A, x], flags)

    want = np.concatenate([
        scipy.linalg.solve_triangular(
            mats[p].astype(np.float64), x[p*n:(p+1)*n].astype(np.float64),
            lower=lower, trans=(1 if trans else 0), unit_diagonal=unit)
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_close_tight(t, b, f"trsv(lower={lower},unit={unit},trans={trans})", dtype)


# ─── posv (the pyroffi-relevant op: factor + both solves, layout-consistent) ───

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_posv(bins, n, dtype):
    mats = [_spd(n) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    bvec = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "posv", dtype, n, NPROB, [A, bvec])

    want = np.concatenate([
        np.linalg.solve(mats[p].astype(np.float64), bvec[p*n:(p+1)*n].astype(np.float64))
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_close_tight(t, b, "posv", dtype)


# ─── potrs (reusable-factor path: solve from a precomputed Cholesky factor) ────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_potrs(bins, n, dtype):
    mats = [_spd(n) for _ in range(NPROB)]
    facts = [np.linalg.cholesky(m.astype(np.float64)).astype(np.float32) for m in mats]
    L = _colmajor_batch(facts)                       # lower factor, column-major
    bvec = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "potrs", dtype, n, NPROB, [L, bvec])

    # Solve against the SAME (float32-rounded) factor the device sees, not the
    # exact A: potrs consumes L, so `L Lᵀ x = b` is the reference, not `A x = b`
    # (they differ at float32-factor precision, ~1e-7 — below the f64 tolerance).
    want = np.concatenate([
        scipy.linalg.cho_solve((facts[p].astype(np.float64), True),
                               bvec[p*n:(p+1)*n].astype(np.float64))
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_close_tight(t, b, "potrs", dtype)


# ═══ L1 ops (reduce/nrm2/asum/nrm1_diff/axpy/scal/copy/rot/symmetrize/*_strided) ═══
#
# Comparison policy (matches the module's existing rules):
#   * Reduction-family ops (reduce/nrm2/asum/nrm1_diff) accumulate SERIALLY on
#     the thread tier while the block1 twin reduces with a tree (reduce) or a
#     lane-strided shuffle (_fast forms) — summation ORDER differs by design, so
#     thread-vs-block1 uses allclose, exactly like test_dot. No _fast/_lowmem
#     thread twins exist (one thread has no reduction strategy).
#   * nrm2: the block sqrtf-in-f64 wart was FIXED (2026-07-17, type-generic
#     sqrt everywhere), so the cross-model check uses the native tolerance.
#   * Elementwise ops (axpy/scal/copy/rot/symmetrize/axpy_strided/copy_strided)
#     run the identical per-element arithmetic in both models (shared *_impl body
#     or identical serial loop), so they use _assert_ulp_equal (ULP-bounded, not
#     bit-equality — see the helper's docstring).
#
# Driver-side constants these tests must mirror (see test_thread.cu dispatch):
#   alpha = 2 for axpy/scal/axpy_strided/copy_strided; (c, s) = (0.6, 0.8) for
#   rot; strided ops use M = N = n, X lead n+1, Y lead n+2, and emit the FULL
#   padded Y buffer (pads ride through both models and are compared too).


# ─── reduce (allclose only: serial accumulate vs block halving tree) ───────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_reduce(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "reduce", dtype, n, NPROB, [x])

    x64 = x.astype(np.float64)
    want = np.array([x64[p*n:(p+1)*n].sum() for p in range(NPROB)])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    # NOT bit-identical to block1 by design (serial vs tree order), like dot.
    np.testing.assert_allclose(t, b, **_TOL[dtype])


# ─── nrm2 (allclose only: serial vs shuffle-strided summation order) ──────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_nrm2(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "nrm2", dtype, n, NPROB, [x])

    x64 = x.astype(np.float64)
    want = np.array([np.linalg.norm(x64[p*n:(p+1)*n]) for p in range(NPROB)])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    # NOT bit-identical to block1 by design (summation order), like dot.
    np.testing.assert_allclose(t, b, **_TOL[dtype])


# ─── asum (allclose only: serial vs shuffle-strided order) ─────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_asum(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "asum", dtype, n, NPROB, [x])

    x64 = x.astype(np.float64)
    want = np.array([np.sum(np.abs(x64[p*n:(p+1)*n])) for p in range(NPROB)])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    np.testing.assert_allclose(t, b, **_TOL[dtype])


# ─── nrm1_diff (allclose only: serial vs shuffle-strided order) ────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_nrm1_diff(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    y = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "nrm1_diff", dtype, n, NPROB, [x, y])

    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    want = np.array([np.sum(np.abs(x64[p*n:(p+1)*n] - y64[p*n:(p+1)*n]))
                     for p in range(NPROB)])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    np.testing.assert_allclose(t, b, **_TOL[dtype])


# ─── axpy (elementwise: shared axpy_impl body -> ULP-equal to block1) ──────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_axpy(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    y = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "axpy", dtype, n, NPROB, [x, y])

    want = 2.0 * x.astype(np.float64) + y.astype(np.float64)   # driver alpha = 2
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "axpy")


# ─── scal (elementwise: shared scal_impl body -> ULP-equal) ────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_scal(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "scal", dtype, n, NPROB, [x])

    want = 2.0 * x.astype(np.float64)                          # driver alpha = 2
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "scal")


# ─── copy (elementwise: shared copy_impl body -> ULP-equal) ────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_copy(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "copy", dtype, n, NPROB, [x])

    want = x.astype(np.float64)
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "copy")


# ─── rot (elementwise pair update: shared rot_impl body -> ULP-equal).
#     Driver emits [all x' | all y'] in one buffer; (c, s) = (0.6, 0.8). ────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_rot(bins, n, dtype):
    x = RNG.standard_normal(NPROB * n).astype(np.float32)
    y = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "rot", dtype, n, NPROB, [x, y])
    assert t.size == 2 * NPROB * n

    c, s = 0.6, 0.8
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    want = np.concatenate([c*x64 + s*y64, c*y64 - s*x64])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "rot")


# ─── symmetrize (each mirror pair owned by one thread -> ULP-equal) ────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_symmetrize(bins, n, dtype):
    mats = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    t, b = _both(bins, "symmetrize", dtype, n, NPROB, [A])

    # 0.5*(A + A.T) leaves the diagonal bit-identical (0.5*(a+a) == a), so the
    # full-matrix oracle is exact for the untouched diagonal too.
    want = np.concatenate([
        (0.5 * (m.astype(np.float64) + m.astype(np.float64).T)).T.ravel()
        for m in mats
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "symmetrize")


# ─── axpy_strided / copy_strided (identical per-element arithmetic -> ULP-equal;
#     M = N = n, X lead n+1, Y lead n+2; the full padded Y buffer is compared,
#     so untouched pads must ride through both models bit-clean) ────────────────

def _strided_oracle(X, Y, n, add):
    """f64 oracle for the strided block ops: block region gets alpha=2 applied
    (accumulate when add=True, overwrite when add=False), pads pass through."""
    lx, ly = (n + 1) * n, (n + 2) * n
    want = Y.astype(np.float64).copy()
    for p in range(NPROB):
        for c in range(n):
            for r in range(n):
                xv = 2.0 * np.float64(X[p*lx + r + c*(n+1)])
                i = p*ly + r + c*(n+2)
                want[i] = want[i] + xv if add else xv
    return want


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_axpy_strided(bins, n, dtype):
    lx, ly = (n + 1) * n, (n + 2) * n
    X = RNG.standard_normal(NPROB * lx).astype(np.float32)
    Y = RNG.standard_normal(NPROB * ly).astype(np.float32)
    t, b = _both(bins, "axpy_strided", dtype, n, NPROB, [X, Y])
    assert t.size == NPROB * ly

    np.testing.assert_allclose(t, _strided_oracle(X, Y, n, add=True), **_TOL[dtype])
    _assert_ulp_equal(t, b, "axpy_strided")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_copy_strided(bins, n, dtype):
    lx, ly = (n + 1) * n, (n + 2) * n
    X = RNG.standard_normal(NPROB * lx).astype(np.float32)
    Y = RNG.standard_normal(NPROB * ly).astype(np.float32)
    t, b = _both(bins, "copy_strided", dtype, n, NPROB, [X, Y])
    assert t.size == NPROB * ly

    np.testing.assert_allclose(t, _strided_oracle(X, Y, n, add=False), **_TOL[dtype])
    _assert_ulp_equal(t, b, "copy_strided")


# ═══ solver ops (trsm / ldlt / ldlt_solve / inv) ═══════════════════════════════

# trsm right-hand-side width — must match the driver's compile-time TRHS.
TRSM_NRHS = 3


def _ldl_nopivot64(A):
    """Float64 NON-pivoted LDLt oracle (scipy.linalg.ldl may pivot even on nice
    input — Bunch–Kaufman is its algorithm — so roll the textbook recurrence,
    which is exactly the algorithm the non-pivoted device path implements)."""
    A = A.astype(np.float64)
    n = A.shape[0]
    L = np.eye(n)
    d = np.zeros(n)
    for j in range(n):
        d[j] = A[j, j] - (L[j, :j] ** 2) @ d[:j]
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - (L[i, :j] * L[j, :j]) @ d[:j]) / d[j]
    return L, d


def _sym_indef(n):
    """A well-conditioned symmetric INDEFINITE matrix that is still safe for the
    NON-pivoted LDLt: strictly diagonally dominant (=> nonzero pivots at every
    step) with mixed-sign diagonal (=> genuinely indefinite, exercising the
    no-sqrt property Cholesky lacks)."""
    M = RNG.standard_normal((n, n)).astype(np.float32)
    S = ((M + M.T) / 2).astype(np.float32)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float32)
    S[np.diag_indices(n)] = signs * (np.abs(S).sum(axis=1) + n)
    return S.astype(np.float32)


# ─── trsm (multi-RHS; sweeps Lower/Upper × Unit/NonUnit × trans) ──────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("unit", [False, True])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", SIZES)
def test_trsm(bins, n, lower, unit, trans, dtype):
    mats = [_tri(n, lower) for _ in range(NPROB)]
    rhss = [RNG.standard_normal((n, TRSM_NRHS)).astype(np.float32) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    B = _colmajor_batch(rhss)                       # n x NRHS column-major per problem
    flags = (int(lower), int(unit), int(trans))
    t, b = _both(bins, "trsm", dtype, n, NPROB, [A, B], flags)

    want = np.concatenate([
        scipy.linalg.solve_triangular(
            mats[p].astype(np.float64), rhss[p].astype(np.float64),
            lower=lower, trans=(1 if trans else 0), unit_diagonal=unit
        ).T.ravel()                                  # back to column-major
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_close_tight(t, b, f"trsm(lower={lower},unit={unit},trans={trans})", dtype)


# ─── ldlt (non-pivoted; block1 oracle = non-pivoted glass::ldlt) ──────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_ldlt(bins, n, dtype):
    # Half SPD, half genuinely indefinite — both non-pivoted-safe.
    mats = [_spd(n) if p % 2 == 0 else _sym_indef(n) for p in range(NPROB)]
    A = _colmajor_batch(mats)
    t, b = _both(bins, "ldlt", dtype, n, NPROB, [A])

    for p in range(NPROB):
        got = t[p*n*n:(p+1)*n*n].reshape(n, n).T     # column-major -> row-major view
        L, d = _ldl_nopivot64(mats[p])
        # diagonal slots hold D, strict lower holds unit-L (upper is untouched input).
        np.testing.assert_allclose(np.diag(got), d, **_TOL[dtype])
        np.testing.assert_allclose(np.tril(got, -1), np.tril(L, -1), **_TOL[dtype])
    _assert_close_tight(t, b, "ldlt", dtype)


# ─── ldlt_solve (reusable-factor path, non-pivoted) ───────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_ldlt_solve(bins, n, dtype):
    mats = [_spd(n) if p % 2 == 0 else _sym_indef(n) for p in range(NPROB)]
    packs = []
    for m in mats:
        L, d = _ldl_nopivot64(m)
        LD = np.tril(L, -1) + np.diag(d)             # packed factor: strict-lower L + D diag
        packs.append(LD.astype(np.float32))
    LDs = _colmajor_batch(packs)
    bvec = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "ldlt_solve", dtype, n, NPROB, [LDs, bvec])

    # Solve against the SAME (float32-rounded) packed factor the device sees
    # (cf. test_potrs): L D Lt x = b with unit-L from the packed strict lower.
    want = []
    for p in range(NPROB):
        LD = packs[p].astype(np.float64)
        L = np.tril(LD, -1) + np.eye(n)
        d = np.diag(LD)
        y = scipy.linalg.solve_triangular(L, bvec[p*n:(p+1)*n].astype(np.float64),
                                          lower=True, unit_diagonal=True)
        want.append(scipy.linalg.solve_triangular(L.T, y / d,
                                                  lower=False, unit_diagonal=True))
    want = np.concatenate(want)
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_close_tight(t, b, "ldlt_solve", dtype)


# ─── inv (unpivoted Gauss-Jordan on the augmented [A | I] layout) ─────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_inv(bins, n, dtype):
    # SPD => every leading principal minor is positive: safe for the UNPIVOTED
    # sweep (the data-dependent inv_pivoted is excluded from the tier on purpose).
    mats = [_spd(n) for _ in range(NPROB)]
    eye = np.eye(n, dtype=np.float32)
    # augmented [A | I], column-major: A's n columns then I's n columns.
    A = np.concatenate([np.concatenate([m.T.ravel(), eye.T.ravel()]) for m in mats]
                       ).astype(np.float32)
    t, b = _both(bins, "inv", dtype, n, NPROB, [A])

    nn = n * n
    for p in range(NPROB):
        got = t[p*2*nn + nn : (p+1)*2*nn].reshape(n, n).T   # right half, col-major -> view
        want = np.linalg.inv(mats[p].astype(np.float64))
        np.testing.assert_allclose(got, want, **_TOL[dtype])
    _assert_close_tight(t, b, "inv", dtype)


# ═══ fused L3 ops (syrk/syr2k/tvc/vtv/congr/bilinear/caccum/riccati) ═══════════
#
# Conventions mirrored from the driver:
#   * all shapes square: K=N, Kdim=N, NX=NU=N;
#   * ACCUMULATE seeds the output with 0.25*(i%7) in-kernel (both models) — the
#     numpy oracle adds the same pattern;
#   * riccati uses rho=0.05 when REG (hardcoded in the driver);
#   * every op asserts the tight thread-vs-block1 ULP bound AND a float-tolerance
#     numpy oracle (composed the same way test_solve.py / test_congruence-style
#     suites compose theirs), so a shared bug in both surfaces can't pass.


def _acc_pat(length):
    """The driver's deterministic ACCUMULATE seed: 0.25*(i%7) (exact in f32/f64)."""
    return 0.25 * (np.arange(length) % 7)


def _tensor_file(tens):
    """Flatten a list of (K,A,B) tensors the way the impl indexes them:
    outer k slabs, each A x B slab column-major (`k*A*B + a + b*A`)."""
    return np.concatenate([
        np.concatenate([t3[k].ravel(order="F") for k in range(t3.shape[0])])
        for t3 in tens
    ]).astype(np.float32)


# ─── syrk (sweeps FillMode × TRANSPOSE; untouched triangle stays zero) ────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("fill", [0, 1, 2])   # 0=Lower 1=Upper 2=Full
@pytest.mark.parametrize("n", SIZES)
def test_syrk(bins, n, fill, trans, dtype):
    mats = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    A = _colmajor_batch(mats)
    t, b = _both(bins, "syrk", dtype, n, NPROB, [A], (fill, int(trans)))

    outs = []
    for p in range(NPROB):
        m = mats[p].astype(np.float64)
        full = (m.T @ m) if trans else (m @ m.T)
        if fill == 0:   full = np.tril(full)   # driver zero-inits C: off-triangle stays 0
        elif fill == 1: full = np.triu(full)
        outs.append(full.ravel(order="F"))
    np.testing.assert_allclose(t, np.concatenate(outs), **_TOL[dtype])
    _assert_ulp_equal(t, b, f"syrk(fill={fill},trans={trans})")


# ─── syr2k ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("fill", [0, 1, 2])
@pytest.mark.parametrize("n", SIZES)
def test_syr2k(bins, n, fill, trans, dtype):
    ma = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    mb = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    A, B = _colmajor_batch(ma), _colmajor_batch(mb)
    t, b = _both(bins, "syr2k", dtype, n, NPROB, [A, B], (fill, int(trans)))

    outs = []
    for p in range(NPROB):
        x, y = ma[p].astype(np.float64), mb[p].astype(np.float64)
        full = (x.T @ y + y.T @ x) if trans else (x @ y.T + y @ x.T)
        if fill == 0:   full = np.tril(full)
        elif fill == 1: full = np.triu(full)
        outs.append(full.ravel(order="F"))
    np.testing.assert_allclose(t, np.concatenate(outs), **_TOL[dtype])
    _assert_ulp_equal(t, b, f"syr2k(fill={fill},trans={trans})")


# ─── tensor_vec_contract (sweeps CONTRACT axis, SYMMETRIC, ACCUMULATE) ────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("contract,sym", [(0, False), (1, False), (2, False), (0, True)])
@pytest.mark.parametrize("n", SIZES)
def test_tensor_vec_contract(bins, n, contract, sym, acc, dtype):
    tens = [RNG.standard_normal((n, n, n)).astype(np.float32) for _ in range(NPROB)]
    if sym:  # SYMMETRIC requires symmetric (a,b) slabs (and CONTRACT==K, A==B)
        tens = [((t3 + t3.transpose(0, 2, 1)) / 2).astype(np.float32) for t3 in tens]
    v = RNG.standard_normal(NPROB * n).astype(np.float32)
    Tfile = _tensor_file(tens)
    t, b = _both(bins, "tvc", dtype, n, NPROB, [Tfile, v],
                 (contract, int(sym), int(acc)))

    init = _acc_pat(n * n)
    outs = []
    for p in range(NPROB):
        t3, vp = tens[p].astype(np.float64), v[p*n:(p+1)*n].astype(np.float64)
        if contract == 0:   M = np.einsum("k,kab->ab", vp, t3)
        elif contract == 1: M = np.einsum("a,kab->kb", vp, t3)
        else:               M = np.einsum("b,kab->ka", vp, t3)
        out = M.ravel(order="F")             # OUT0 x OUT1, column-major
        outs.append(out + init if acc else out)
    np.testing.assert_allclose(t, np.concatenate(outs), **_TOL[dtype])
    _assert_ulp_equal(t, b, f"tvc(contract={contract},sym={sym},acc={acc})")


# ─── vec_tensor_vec (sweeps ACCUMULATE) ───────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_vec_tensor_vec(bins, n, acc, dtype):
    tens = [RNG.standard_normal((n, n, n)).astype(np.float32) for _ in range(NPROB)]
    u = RNG.standard_normal(NPROB * n).astype(np.float32)
    w = RNG.standard_normal(NPROB * n).astype(np.float32)
    t, b = _both(bins, "vtv", dtype, n, NPROB, [_tensor_file(tens), u, w], (int(acc),))

    init = _acc_pat(n)
    outs = []
    for p in range(NPROB):
        t3 = tens[p].astype(np.float64)
        up, wp = u[p*n:(p+1)*n].astype(np.float64), w[p*n:(p+1)*n].astype(np.float64)
        s = np.einsum("a,kab,b->k", up, t3, wp)
        outs.append(s + init if acc else s)
    np.testing.assert_allclose(t, np.concatenate(outs), **_TOL[dtype])
    _assert_ulp_equal(t, b, f"vtv(acc={acc})")


# ─── congruence_sym (Q = XᵀMX, M symmetric; sweeps ACCUMULATE/beta=1) ─────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_congruence_sym(bins, n, acc, dtype):
    xs = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    ms = [_spd(n) for _ in range(NPROB)]   # symmetric M so Q = XᵀMX is symmetric
    t, b = _both(bins, "congr", dtype, n, NPROB,
                 [_colmajor_batch(xs), _colmajor_batch(ms)], (int(acc),))

    init = _acc_pat(n * n)
    outs = []
    for p in range(NPROB):
        X, M = xs[p].astype(np.float64), ms[p].astype(np.float64)
        Q = (X.T @ M @ X).ravel(order="F")
        outs.append(Q + init if acc else Q)   # ACC: beta=1 on the seeded Q
    np.testing.assert_allclose(t, np.concatenate(outs), **_TOL[dtype])
    _assert_ulp_equal(t, b, f"congruence_sym(acc={acc})")


# ─── bilinear (R = XᵀMY, general M/Y) ─────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_bilinear(bins, n, dtype):
    xs = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    ms = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    ys = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    t, b = _both(bins, "bilinear", dtype, n, NPROB,
                 [_colmajor_batch(xs), _colmajor_batch(ms), _colmajor_batch(ys)])

    want = np.concatenate([
        (xs[p].astype(np.float64).T @ ms[p].astype(np.float64) @ ys[p].astype(np.float64)).ravel(order="F")
        for p in range(NPROB)
    ])
    np.testing.assert_allclose(t, want, **_TOL[dtype])
    _assert_ulp_equal(t, b, "bilinear")


# ─── congruence_accum (C = G·M·Gᵀ, M symmetric; sweeps ACCUMULATE) ────────────

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_congruence_accum(bins, n, acc, dtype):
    gs = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    ms = [_spd(n) for _ in range(NPROB)]
    t, b = _both(bins, "caccum", dtype, n, NPROB,
                 [_colmajor_batch(gs), _colmajor_batch(ms)], (int(acc),))

    init = _acc_pat(n * n)
    outs = []
    for p in range(NPROB):
        G, M = gs[p].astype(np.float64), ms[p].astype(np.float64)
        C = (G @ M @ G.T).ravel(order="F")
        outs.append(C + init if acc else C)
    np.testing.assert_allclose(t, np.concatenate(outs), **_TOL[dtype])
    _assert_ulp_equal(t, b, f"congruence_accum(acc={acc})")


# ─── riccati_gain (K = (R+BᵀPB+ρI)⁻¹ BᵀPA; the house test_solve.py oracle) ────

# The composed solve amplifies f32 rounding a little beyond the module _TOL
# (test_solve.py uses 3e-2 for the same op); f64 stays near-exact.
_RIC_TOL = {"f32": dict(rtol=1e-2, atol=1e-2), "f64": dict(rtol=1e-9, atol=1e-9)}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reg", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_riccati_gain(bins, n, reg, dtype):
    Ps = [_spd(n) for _ in range(NPROB)]
    As = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    Bs = [RNG.standard_normal((n, n)).astype(np.float32) for _ in range(NPROB)]
    Rs = [_spd(n) for _ in range(NPROB)]
    t, b = _both(bins, "riccati", dtype, n, NPROB,
                 [_colmajor_batch(Ps), _colmajor_batch(As),
                  _colmajor_batch(Bs), _colmajor_batch(Rs)], (int(reg),))

    rho = 0.05 if reg else 0.0            # matches the driver's hardcoded rho
    outs = []
    for p in range(NPROB):
        P = Ps[p].astype(np.float64); A = As[p].astype(np.float64)
        B = Bs[p].astype(np.float64); R = Rs[p].astype(np.float64)
        S = R + B.T @ P @ B + rho * np.eye(n)          # test_solve.py's composition
        K = np.linalg.solve(S, B.T @ P @ A)            # NU x NX (= n x n)
        outs.append(K.ravel(order="F"))
    np.testing.assert_allclose(t, np.concatenate(outs), **_RIC_TOL[dtype])
    _assert_close_tight(t, b, f"riccati(reg={reg})", dtype)
