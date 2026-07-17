"""L1 GLASS function tests — NumPy oracle + canonical thread-count sweep.

Thread discipline (see docs/open-tasks/test_hardening_thread_sweep_2026-06-24.md):
  • elementwise / halving-tree / serial-tail ops → BYTE-IDENTICAL output across
    the sweep (cg, simple, simple_lm versions).
  • warp-shuffle reductions (simple_hs / *_fast) → oracle-close at every count;
    the per-thread partial grouping varies with blockDim, so byte-identity
    across counts is not the contract there.
  • warp:: ops → exactly one 32-lane warp (never swept).
  • prefix sums document `n <= blockDim.x`, so they sweep only counts >= n.

The CUDA runner takes the block size as a runtime arg:
    ./test_l1 <op> <ver> <threads> <n> [args...] [files...]
Most tests sweep THREAD_SWEEP_CORE; the *_full_sweep tests cover the full
canonical THREAD_SWEEP at a representative odd non-boundary size (57).
Inputs vary between 'normal' (mixed sign) and 'mixed' (alternating 1e3/1e-3
magnitudes — stresses reductions/1-norms) via conftest.make_vec.
"""

import numpy as np
import pytest
from conftest import run_op, THREAD_SWEEP, THREAD_SWEEP_CORE, make_vec

RNG = np.random.default_rng(42)

ATOL = 1e-5
RTOL = 1e-4

# (n, input kind) pairs: every op sees >= 2 input kinds across its size matrix.
SIZED_KINDS = [(8, "normal"), (64, "mixed"), (256, "normal")]
SIZES = [8, 64, 256]

CG_SIMPLE = ["cg", "simple"]
CG_LM_HS  = ["cg", "simple_lm", "simple_hs"]

# versions whose result is required to be byte-identical across block sizes
# (halving-tree / serial-tail); simple_hs is the warp-shuffle strategy whose
# summation grouping legitimately changes with blockDim.
EXACT_VERSIONS = ("cg", "simple", "simple_lm")


def _rel_ok(got, expected, rtol=2e-3):
    """Scalar closeness with a relative-dominant tolerance (mixed-magnitude
    inputs make absolute tolerances meaningless)."""
    return abs(float(got) - float(expected)) <= rtol * max(1.0, abs(float(expected)))


def sweep_exact(binary, op, version, args, inputs, sweep=THREAD_SWEEP_CORE):
    """Run `op` at every thread count in `sweep`; assert the output is
    BYTE-IDENTICAL at every count and return the first result."""
    ref = None
    for th in sweep:
        r = run_op(binary, op, version, [th] + list(args), inputs)
        if ref is None:
            ref = r
        elif isinstance(ref, list):
            for a, b in zip(ref, r):
                assert np.array_equal(a, b), \
                    f"{op}/{version}: thread-count non-invariance at {th} threads"
        else:
            assert np.array_equal(ref, r), \
                f"{op}/{version}: thread-count non-invariance at {th} threads"
    return ref


def sweep_scalar(binary, op, version, args, inputs, expected,
                 sweep=THREAD_SWEEP_CORE, exact=True, rtol=2e-3):
    """Scalar-reduction sweep: oracle-close at every count; byte-identical
    across counts when `exact` (halving/serial strategies)."""
    ref = None
    for th in sweep:
        r = run_op(binary, op, version, [th] + list(args), inputs)
        got = r[0] if not isinstance(r, list) else r[0][0]
        assert _rel_ok(got, expected, rtol), \
            f"{op}/{version} threads={th}: {got} vs {expected}"
        if exact:
            if ref is None:
                ref = got
            else:
                assert got == ref, \
                    f"{op}/{version}: thread-count non-invariance at {th} threads"


# ─── axpy ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_axpy(bins, n, kind, version):
    alpha = 1.5
    x = make_vec(n, seed=n + 1, kind=kind)
    y = make_vec(n, seed=n + 2, kind=kind)
    result = sweep_exact(bins["l1"], "axpy", version, [n, alpha], [x, y])
    assert np.allclose(result, alpha * x + y, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kind", ["normal", "mixed"])
def test_axpy_full_sweep(bins, kind):
    """Representative elementwise op over the FULL canonical sweep."""
    n, alpha = 57, -1.25
    x = make_vec(n, seed=3, kind=kind)
    y = make_vec(n, seed=4, kind=kind)
    result = sweep_exact(bins["l1"], "axpy", "simple", [n, alpha], [x, y],
                         sweep=THREAD_SWEEP)
    assert np.allclose(result, alpha * x + y, rtol=RTOL, atol=ATOL)


# ─── axpby ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_axpby(bins, n, kind, version):
    alpha, beta = 1.5, 0.3
    x = make_vec(n, seed=n + 5, kind=kind)
    y = make_vec(n, seed=n + 6, kind=kind)
    result = sweep_exact(bins["l1"], "axpby", version, [n, alpha, beta], [x, y])
    assert np.allclose(result, alpha * x + beta * y, rtol=RTOL, atol=ATOL)


# ─── copy ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_copy(bins, n, kind, version):
    x = make_vec(n, seed=n + 7, kind=kind)
    result = sweep_exact(bins["l1"], "copy", version, [n], [x])
    assert np.allclose(result, x, rtol=RTOL, atol=ATOL)


# ─── scal ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_scal(bins, n, kind, version):
    alpha = 2.5
    x = make_vec(n, seed=n + 8, kind=kind)
    result = sweep_exact(bins["l1"], "scal", version, [n, alpha], [x])
    assert np.allclose(result, alpha * x, rtol=RTOL, atol=ATOL)


# ─── swap ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_swap(bins, n, kind, version):
    x = make_vec(n, seed=n + 9, kind=kind)
    y = make_vec(n, seed=n + 10, kind=kind)
    result = sweep_exact(bins["l1"], "swap", version, [n], [x, y])
    assert np.allclose(result[0], y, rtol=RTOL, atol=ATOL)
    assert np.allclose(result[1], x, rtol=RTOL, atol=ATOL)


# ─── dot ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_dot(bins, n, kind, version):
    x = make_vec(n, seed=n + 11, kind=kind)
    y = make_vec(n, seed=n + 12, kind=kind)
    expected = np.dot(x.astype(np.float64), y.astype(np.float64))
    sweep_scalar(bins["l1"], "dot", version, [n], [x, y], expected,
                 exact=(version in EXACT_VERSIONS))


# ─── reduce ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_reduce(bins, n, kind, version):
    x = make_vec(n, seed=n + 13, kind=kind)
    expected = float(np.sum(x.astype(np.float64)))
    sweep_scalar(bins["l1"], "reduce", version, [n], [x], expected,
                 exact=(version in EXACT_VERSIONS))


# ─── reduce_fast register-partial overload ─────────────────────────────
# Each thread forms a per-thread partial (strided slice of x) and passes it
# directly to reduce(partial, scratch); the returned block total is broadcast to
# every thread. Partial grouping varies with blockDim → oracle-close per count.

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
def test_reduce_partial(bins, n, kind):
    x = make_vec(n, seed=n + 14, kind=kind)
    expected = float(np.sum(x.astype(np.float64)))
    sweep_scalar(bins["l1"], "reduce_partial", "simple_hs", [n], [x], expected,
                 exact=False)


# ─── reduce_fast_min register-partial overload ─────────────────────────
# Min twin of test_reduce_partial. Exact: min selects an input element, so no
# summation-order rounding — unlike the sum overload it must match the oracle
# bit-for-bit at every thread count.

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
def test_reduce_min_partial(bins, n, kind):
    x = make_vec(n, seed=n + 17, kind=kind)
    expected = float(np.min(x.astype(np.float64)))
    # exact=True: min SELECTS an input element rather than accumulating, so unlike
    # the sum overload there is no grouping-dependent rounding — it must hit the
    # oracle and stay byte-identical across every thread count.
    sweep_scalar(bins["l1"], "reduce_min_partial", "simple_hs", [n], [x], expected,
                 exact=True)


# ─── warp::reduce (single warp, always launched <<<1,32>>>) ───────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
def test_reduce_warp(bins, n, kind):
    x = make_vec(n, seed=n + 15, kind=kind)
    result = run_op(bins["l1"], "reduce", "warp", args=[32, n], inputs=[x])
    expected = float(np.sum(x.astype(np.float64)))
    assert _rel_ok(result[0], expected)


@pytest.mark.parametrize("n,kind", SIZED_KINDS)
def test_reduce_partial_warp(bins, n, kind):
    x = make_vec(n, seed=n + 16, kind=kind)
    result = run_op(bins["l1"], "reduce_partial", "warp", args=[32, n], inputs=[x])
    expected = float(np.sum(x.astype(np.float64)))
    assert _rel_ok(result[0], expected)


# ─── nrm2 ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_nrm2(bins, n, kind, version):
    x = make_vec(n, seed=n + 17, kind=kind)
    expected = float(np.linalg.norm(x.astype(np.float64)))
    sweep_scalar(bins["l1"], "nrm2", version, [n], [x], expected,
                 exact=(version in EXACT_VERSIONS))


# ─── vector_norm (non-destructive ‖a‖₂ into out[0]) ───────────────────────────
# All three strategies: bare default (halving tree), _lowmem (thread-0 serial),
# _fast (warp shuffle + inter-warp scratch, sized per reduce_fast_scratch_bytes
# in the runner). The runner prints out[0] AND the input vector afterwards so
# we can prove the input is untouched (the destructive nrm2's whole point).

VN_VERSIONS = ["simple", "simple_lm", "simple_hs"]


@pytest.mark.parametrize("n,kind", SIZED_KINDS + [(57, "pos")])
@pytest.mark.parametrize("version", VN_VERSIONS)
def test_vector_norm(bins, n, kind, version):
    x = make_vec(n, seed=n + 18, kind=kind)
    expected = float(np.linalg.norm(x.astype(np.float64)))
    exact = version in EXACT_VERSIONS
    ref = None
    for th in THREAD_SWEEP_CORE:
        norm_line, a_after = run_op(bins["l1"], "vector_norm", version,
                                    args=[th, n], inputs=[x])
        assert _rel_ok(norm_line[0], expected), \
            f"vector_norm/{version} threads={th}: {norm_line[0]} vs {expected}"
        # non-destructive: input must come back unchanged (%.8g print tolerance)
        assert np.allclose(a_after, x, rtol=1e-6, atol=0.0), \
            f"vector_norm/{version} threads={th}: input vector was clobbered"
        if exact:
            if ref is None:
                ref = norm_line[0]
            else:
                assert norm_line[0] == ref, \
                    f"vector_norm/{version}: non-invariance at {th} threads"


@pytest.mark.parametrize("kind", ["normal", "mixed"])
@pytest.mark.parametrize("version", VN_VERSIONS)
def test_vector_norm_full_sweep(bins, version, kind):
    n = 57
    x = make_vec(n, seed=19, kind=kind)
    expected = float(np.linalg.norm(x.astype(np.float64)))
    exact = version in EXACT_VERSIONS
    ref = None
    for th in THREAD_SWEEP:
        norm_line, a_after = run_op(bins["l1"], "vector_norm", version,
                                    args=[th, n], inputs=[x])
        assert _rel_ok(norm_line[0], expected), \
            f"vector_norm/{version} threads={th}: {norm_line[0]} vs {expected}"
        assert np.allclose(a_after, x, rtol=1e-6, atol=0.0), \
            f"vector_norm/{version} threads={th}: input vector was clobbered"
        if exact:
            if ref is None:
                ref = norm_line[0]
            else:
                assert norm_line[0] == ref, \
                    f"vector_norm/{version}: non-invariance at {th} threads"


# ─── infnorm (halving max-reduce → byte-identical) ────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_infnorm(bins, n, kind, version):
    x = make_vec(n, seed=n + 20, kind=kind)
    expected = float(np.max(np.abs(x)))
    sweep_scalar(bins["l1"], "infnorm", version, [n], [x], expected, exact=True)


# ─── asum ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_LM_HS)
def test_asum(bins, n, kind, version):
    x = make_vec(n, seed=n + 21, kind=kind)
    expected = float(np.sum(np.abs(x).astype(np.float64)))
    sweep_scalar(bins["l1"], "asum", version, [n], [x], expected,
                 exact=(version in EXACT_VERSIONS))


# ─── reductions: FULL canonical sweep at an odd non-boundary size ─────────────

REDUCTION_ORACLES = {
    "dot":    (2, lambda x, y: float(np.dot(x.astype(np.float64), y.astype(np.float64)))),
    "reduce": (1, lambda x: float(np.sum(x.astype(np.float64)))),
    "nrm2":   (1, lambda x: float(np.linalg.norm(x.astype(np.float64)))),
    "asum":   (1, lambda x: float(np.sum(np.abs(x).astype(np.float64)))),
}


@pytest.mark.parametrize("op", sorted(REDUCTION_ORACLES))
@pytest.mark.parametrize("version", ["simple_lm", "simple_hs"])
@pytest.mark.parametrize("kind", ["normal", "mixed"])
def test_reduction_full_sweep(bins, op, version, kind):
    """dot/reduce/nrm2/asum over the FULL THREAD_SWEEP: oracle-close at every
    count; the serial-tail (_lowmem) strategy additionally byte-identical."""
    n = 57
    nin, oracle = REDUCTION_ORACLES[op]
    vecs = [make_vec(n, seed=30 + i, kind=kind) for i in range(nin)]
    expected = oracle(*vecs)
    sweep_scalar(bins["l1"], op, version, [n], vecs, expected,
                 sweep=THREAD_SWEEP, exact=(version in EXACT_VERSIONS))


# ─── clip ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_clip(bins, n, kind, version):
    x = make_vec(n, seed=n + 22, kind=kind)
    lo = np.full(n, -0.5, dtype=np.float32)
    hi = np.full(n,  0.5, dtype=np.float32)
    result = sweep_exact(bins["l1"], "clip", version, [n], [x, lo, hi])
    assert np.allclose(result, np.clip(x, lo, hi), rtol=RTOL, atol=ATOL)


# ─── set_const ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_set_const(bins, n, version):
    alpha = 3.14
    result = sweep_exact(bins["l1"], "set_const", version, [n, alpha], [])
    assert np.allclose(result, np.full(n, alpha, dtype=np.float32), rtol=RTOL, atol=ATOL)


# ─── set_identity ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_loadIdentity(bins, n, version):
    result = sweep_exact(bins["l1"], "set_identity", version, [n], [])
    # column-major identity: reshape as (n, n) Fortran order
    mat = result.reshape(n, n, order='F')
    assert np.allclose(mat, np.eye(n, dtype=np.float32), rtol=RTOL, atol=ATOL)


# ─── add_identity ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_addI(bins, n, version):
    alpha = 0.5
    A = RNG.random((n, n)).astype(np.float32)
    A_col = np.asfortranarray(A)
    result = sweep_exact(bins["l1"], "add_identity", version, [n, alpha],
                         [A_col.ravel(order='F')])
    expected = A + alpha * np.eye(n, dtype=np.float32)
    mat = result.reshape(n, n, order='F')
    assert np.allclose(mat, expected, rtol=RTOL, atol=ATOL)


# ─── transpose ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N,M", [(4, 6), (8, 8), (12, 4)])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_transpose(bins, N, M, version):
    A = (RNG.standard_normal((N, M))).astype(np.float32)
    A_col = np.asfortranarray(A)  # column-major
    result = sweep_exact(bins["l1"], "transpose", version, [N, M],
                         [A_col.ravel(order='F')])
    # Result is MxN in column-major order
    mat = result.reshape(M, N, order='F')
    assert np.allclose(mat, A.T, rtol=RTOL, atol=ATOL)


# ─── elementwise ops ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
@pytest.mark.parametrize("op,ref", [
    ("elementwise_add",  lambda a, b: a + b),
    ("elementwise_sub",  lambda a, b: a - b),
    ("elementwise_mult", lambda a, b: a * b),
    ("elementwise_max",  np.maximum),
    ("elementwise_min",  np.minimum),
])
def test_elementwise_binary(bins, n, kind, version, op, ref):
    a = make_vec(n, seed=n + 23, kind=kind)
    b = make_vec(n, seed=n + 24, kind=kind)
    result = sweep_exact(bins["l1"], op, version, [n], [a, b])
    assert np.allclose(result, ref(a, b), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_abs(bins, n, kind, version):
    a = make_vec(n, seed=n + 25, kind=kind)
    result = sweep_exact(bins["l1"], "elementwise_abs", version, [n], [a])
    assert np.allclose(result, np.abs(a), rtol=RTOL, atol=ATOL)


# ─── dot_strided ──────────────────────────────────────────────────────────────
# Per-thread primitive launched <<<1,1>>> by design — no thread sweep.

DOT_STRIDED_SHAPES = [
    (4, 4, 1),   # x_size=16, y_size=4
    (6, 1, 1),   # x_size=6,  y_size=6
    (6, 6, 1),   # x_size=36, y_size=6
    (6, 6, 6),   # x_size=36, y_size=36
]


def _make_test_vec(size, case):
    if case == "positive": return RNG.random(size).astype(np.float32)
    if case == "negative": return -RNG.random(size).astype(np.float32)
    if case == "mixed":    return (RNG.random(size) - 0.5).astype(np.float32)
    if case == "zero":     return np.zeros(size, dtype=np.float32)
    if case == "tiny":     return (RNG.random(size) * 1e-6).astype(np.float32)
    raise ValueError(case)


@pytest.mark.parametrize("n,sx,sy", DOT_STRIDED_SHAPES)
@pytest.mark.parametrize("case", ["positive", "negative", "mixed", "zero", "tiny"])
def test_dot_strided(bins, n, sx, sy, case):
    x = _make_test_vec(n * sx, case)
    y = _make_test_vec(n * sy, case)
    # args: <threads> then the required <n> positional (unused by this dispatch)
    result = run_op(bins["l1"], f"dot_strided_{n}_{sx}_{sy}", "simple",
                    args=[1, 0], inputs=[x, y])
    expected = sum(float(x[i * sx]) * float(y[i * sy]) for i in range(n))
    assert np.isclose(float(result[0]), expected, rtol=RTOL, atol=ATOL)


# ─── dot_strided_coalesced ──────────────────────────────────────────────────
# Block-cooperative sibling of dot_strided: same value, coalesced global loads.
# "simple" version dispatches the per-thread dot_strided reference (thread 0
# writes); "simple_hs" dispatches the coalesced block-reduction primitive.
# The block reduction's partial grouping varies with blockDim → oracle-close
# at every swept count (not byte-identical).

DOT_COALESCED_SHAPES = [
    (64, 64, 64),   # x,y = 4096 each; stride 64 (column of a 64-wide row-major mat)
    (256, 256, 1),  # x = 65536, y = 256; large x stride, unit y stride
]


@pytest.mark.parametrize("n,sx,sy", DOT_COALESCED_SHAPES)
def test_dot_strided_coalesced(bins, n, sx, sy):
    x = (RNG.random(n * sx) - 0.5).astype(np.float32)
    y = (RNG.random(n * sy) - 0.5).astype(np.float32)
    op = f"dot_coalesced_{n}_{sx}_{sy}"
    expected = sum(float(x[i * sx]) * float(y[i * sy]) for i in range(n))
    reference = run_op(bins["l1"], op, "simple", args=[64, 0], inputs=[x, y])
    assert np.isclose(float(reference[0]), expected, rtol=1e-3, atol=1e-4)
    for th in THREAD_SWEEP_CORE:
        coalesced = run_op(bins["l1"], op, "simple_hs", args=[th, 0], inputs=[x, y])
        assert np.isclose(float(coalesced[0]), expected, rtol=1e-3, atol=1e-4), \
            f"{op} threads={th}"


# ─── prefix sum ───────────────────────────────────────────────────────────────
# Documented contract: n must not exceed blockDim.x (tid-indexed Hillis-Steele
# scan, no grid stride) — so the sweep uses only thread counts >= n.

@pytest.mark.parametrize("n,kind", [(8, "normal"), (32, "mixed"), (64, "normal")])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_prefix_sum_excl(bins, n, kind, version):
    x = make_vec(n, seed=n + 26, kind=kind)
    sweep = [t for t in THREAD_SWEEP if t >= n]
    result = sweep_exact(bins["l1"], "prefix_sum_excl", version, [n], [x], sweep=sweep)
    expected = np.concatenate([[0], np.cumsum(x.astype(np.float64))[:-1]])
    assert np.allclose(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("n,kind", [(8, "normal"), (32, "mixed"), (64, "normal")])
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_prefix_sum_incl(bins, n, kind, version):
    x = make_vec(n, seed=n + 27, kind=kind)
    sweep = [t for t in THREAD_SWEEP if t >= n]
    result = sweep_exact(bins["l1"], "prefix_sum_incl", version, [n], [x], sweep=sweep)
    expected = np.cumsum(x.astype(np.float64))
    assert np.allclose(result, expected, rtol=1e-3, atol=1e-3)


# ── comparison / logic / scalar elementwise ──────────────────────────────────

@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
@pytest.mark.parametrize("op,ref", [
    ("elementwise_less_than",       lambda a, b: (a < b).astype(np.float32)),
    ("elementwise_more_than",       lambda a, b: (a > b).astype(np.float32)),
    ("elementwise_less_than_or_eq", lambda a, b: (a <= b).astype(np.float32)),
])
def test_elementwise_compare(bins, n, kind, version, op, ref):
    a = make_vec(n, seed=n + 28, kind=kind)
    b = make_vec(n, seed=n + 29, kind=kind)
    result = sweep_exact(bins["l1"], op, version, [n], [a, b])
    assert np.allclose(result, ref(a, b), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_and(bins, n, version):
    # logical AND needs zeros to exercise both branches
    a = RNG.integers(0, 2, n).astype(np.float32)
    b = RNG.integers(0, 2, n).astype(np.float32)
    result = sweep_exact(bins["l1"], "elementwise_and", version, [n], [a, b])
    assert np.allclose(result, ((a != 0) & (b != 0)).astype(np.float32), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("version", CG_SIMPLE)
def test_elementwise_not(bins, n, version):
    a = RNG.integers(0, 2, n).astype(np.float32)
    result = sweep_exact(bins["l1"], "elementwise_not", version, [n], [a])
    assert np.allclose(result, (a == 0).astype(np.float32), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n,kind", SIZED_KINDS)
@pytest.mark.parametrize("version", CG_SIMPLE)
@pytest.mark.parametrize("op,ref", [
    ("elementwise_mult_scalar", lambda a, s: a * s),
    ("elementwise_max_scalar",  lambda a, s: np.maximum(a, s)),
    ("elementwise_min_scalar",  lambda a, s: np.minimum(a, s)),
])
def test_elementwise_scalar(bins, n, kind, version, op, ref):
    a = make_vec(n, seed=n + 30, kind=kind)
    s = np.float32(0.5)
    result = sweep_exact(bins["l1"], op, version, [n, float(s)], [a])
    assert np.allclose(result, ref(a, s), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n,kind", SIZED_KINDS)
def test_elementwise_less_than_scalar(bins, n, kind):
    a = make_vec(n, seed=n + 31, kind=kind)
    s = np.float32(0.5)
    result = sweep_exact(bins["l1"], "elementwise_less_than_scalar", "simple",
                         [n, float(s)], [a])
    assert np.allclose(result, (a < s).astype(np.float32), rtol=RTOL, atol=ATOL)
