"""iamax (BLAS i_amax) tests — GPU index-of-max-abs vs NumPy argmax(abs).

Oracle: int(np.argmax(np.abs(x))) — which returns the LOWEST index on ties,
matching the BLAS i_amax tie-break implemented in GLASS. NaN inputs are
deliberately excluded: GLASS skips NaN (IEEE compares false), whereas
np.argmax(np.abs) would propagate NaN.
"""

import numpy as np
import pytest
from conftest import run_op

RNG = np.random.default_rng(42)

# All three block-scoped variants.
VARIANTS = ["iamax", "iamax_lm", "iamax_hs"]

SIZES = [1, 8, 64, 256]

DEFAULT_THREADS = 256


def _gpu_idx(bins, op, x, threads=DEFAULT_THREADS):
    """Run an iamax variant and return the integer index it reported."""
    result = run_op(bins["iamax"], op, "simple", args=[len(x), threads], inputs=[x])
    # single-line output -> a length-1 float array
    return int(round(float(np.asarray(result).ravel()[0])))


def _oracle(x):
    return int(np.argmax(np.abs(x.astype(np.float64))))


# ─── random distinct-ish vectors (with negatives) ─────────────────────────────

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", VARIANTS)
def test_iamax_random(bins, n, op):
    x = (RNG.random(n) - 0.5).astype(np.float32)
    assert _gpu_idx(bins, op, x) == _oracle(x)


# ─── deliberately-tied vectors → lowest tied index must win ───────────────────

TIED_CASES = [
    np.ones(8, dtype=np.float32),                       # all equal -> 0
    np.full(16, -2.0, dtype=np.float32),                # all equal (neg) -> 0
    np.array([1, 3, 3, 2, 3], dtype=np.float32),        # max 3 tied at idx 1
]


@pytest.mark.parametrize("op", VARIANTS)
@pytest.mark.parametrize("x", TIED_CASES, ids=["ones", "neg2", "13323"])
def test_iamax_ties(bins, op, x):
    expected = _oracle(x)            # np.argmax already returns the lowest tie
    assert _gpu_idx(bins, op, x) == expected


# ─── all-zero vector → index 0 ────────────────────────────────────────────────

@pytest.mark.parametrize("op", VARIANTS)
def test_iamax_all_zero(bins, op):
    x = np.zeros(32, dtype=np.float32)
    assert _gpu_idx(bins, op, x) == 0


# ─── negative max picks the right (negative) element ──────────────────────────

@pytest.mark.parametrize("op", VARIANTS)
def test_iamax_neg_max(bins, op):
    x = np.array([0.1, -5.0, 2.0], dtype=np.float32)   # |x| max at idx 1
    assert _gpu_idx(bins, op, x) == 1 == _oracle(x)


# ─── thread-count invariance: identical index across block sizes ──────────────
# Sweep 1 / 32 / 48 (partial warp) / 256 on BOTH a distinct-max and a tied
# vector. Mismatch across counts is the §1a/§1b sync / invariance hazard.

THREAD_SWEEP = [1, 32, 48, 256]

SWEEP_VECS = {
    "distinct": (RNG.random(200) - 0.5).astype(np.float32),
    "tied":     np.array([1, 3, 3, 2, 3] * 13, dtype=np.float32),  # len 65, max ties
}


@pytest.mark.parametrize("op", VARIANTS)
@pytest.mark.parametrize("vec_name", list(SWEEP_VECS))
def test_iamax_thread_invariance(bins, op, vec_name):
    x = SWEEP_VECS[vec_name]
    expected = _oracle(x)
    seen = {t: _gpu_idx(bins, op, x, threads=t) for t in THREAD_SWEEP}
    # all thread counts agree...
    assert len(set(seen.values())) == 1, f"non-invariant: {seen}"
    # ...and they agree with the oracle.
    assert next(iter(seen.values())) == expected, f"{seen} != oracle {expected}"


# ─── value-returning overload: out_val == max|x| ─────────────────────────────

@pytest.mark.parametrize("n", [8, 64, 256])
def test_iamax_value(bins, n):
    x = (RNG.random(n) - 0.5).astype(np.float32)
    result = run_op(bins["iamax"], "iamax_val", "simple",
                    args=[n, DEFAULT_THREADS], inputs=[x])
    # two lines: [index_array, value_array]
    idx = int(round(float(np.asarray(result[0]).ravel()[0])))
    val = float(np.asarray(result[1]).ravel()[0])
    assert idx == _oracle(x)
    assert np.isclose(val, float(np.max(np.abs(x))), rtol=1e-5, atol=1e-6)


# ─── warp-scoped glass::warp::iamax: WARPS independent vectors, one warp each ──
# Launched <<<1, dim3(32, WARPS)>>>; warp w owns x[w]. Each warp returns
# argmax(|x_w|) (broadcast to its lanes), written to out[w]. Per-warp oracle is
# int(np.argmax(np.abs(x_w))) — single-warp must equal multi-warp (independence),
# and the answer must match the block-scoped variants and the lowest-index rule.

WARP_SIZES = [5, 7, 33, 40, 64, 256]   # incl. non-multiples of 32
WARP_COUNTS = [1, 2, 4]                 # WARPS=1 (single) == WARPS>1 (independent)


def _gpu_warp_idx(bins, x_stack, n, W):
    """Run glass::warp::iamax on W stacked length-n vectors; return W indices."""
    flat = np.ascontiguousarray(x_stack.reshape(-1)).astype(np.float32)
    # runner argv: <op> <version(ignored)> <n> <W> <file> — n and W are args.
    result = run_op(bins["iamax"], "iamax_warp", "simple", args=[n, W], inputs=[flat])
    arr = np.asarray(result).ravel()
    return [int(round(float(v))) for v in arr]


@pytest.mark.parametrize("n", WARP_SIZES)
@pytest.mark.parametrize("W", WARP_COUNTS)
def test_iamax_warp_random(bins, n, W):
    # Each warp gets a distinct random vector (with negatives).
    x = (RNG.random((W, n)) - 0.5).astype(np.float32)
    got = _gpu_warp_idx(bins, x, n, W)
    expected = [int(np.argmax(np.abs(x[w].astype(np.float64)))) for w in range(W)]
    assert got == expected, f"n={n} W={W}: {got} != {expected}"


@pytest.mark.parametrize("n", WARP_SIZES)
def test_iamax_warp_single_eq_multi(bins, n):
    # The SAME vector replicated across warps must give the SAME index in every
    # warp (multi-warp independence) and equal the WARPS=1 single-warp result.
    base = (RNG.random(n) - 0.5).astype(np.float32)
    expected = int(np.argmax(np.abs(base.astype(np.float64))))
    single = _gpu_warp_idx(bins, base[None, :], n, 1)
    multi = _gpu_warp_idx(bins, np.tile(base, (4, 1)), n, 4)
    assert single == [expected]
    assert multi == [expected] * 4


def test_iamax_warp_ties(bins):
    # Deliberately-tied vectors: lowest tied index must win, per warp.
    vecs = [
        np.ones(7, dtype=np.float32),                           # all equal -> 0
        np.array([1, 3, 3, 2, 3], dtype=np.float32),            # max 3 tied at idx 1
        np.array([2, 1, 2, 2, 0, 2, 1], dtype=np.float32),      # max 2 tied at idx 0
    ]
    n = 7
    stack = np.zeros((len(vecs), n), dtype=np.float32)
    for w, v in enumerate(vecs):
        stack[w, :len(v)] = v
    got = _gpu_warp_idx(bins, stack, n, len(vecs))
    expected = [int(np.argmax(np.abs(stack[w].astype(np.float64)))) for w in range(len(vecs))]
    assert got == expected == [0, 1, 0]


def test_iamax_warp_all_zero(bins):
    # All-zero vector -> index 0; mixed with a non-zero warp to confirm independence.
    stack = np.zeros((2, 33), dtype=np.float32)
    stack[1, 10] = -4.0   # warp 1 has its max at idx 10
    got = _gpu_warp_idx(bins, stack, 33, 2)
    assert got == [0, 10]


def test_iamax_warp_neg_max(bins):
    # Negative element is the abs-max; per-warp distinct max locations.
    stack = np.array([
        [0.1, -5.0, 2.0, 0.0, 0.0],    # |x| max at idx 1 (negative)
        [0.0, 0.0, 0.0, -7.0, 3.0],    # |x| max at idx 3 (negative)
    ], dtype=np.float32)
    got = _gpu_warp_idx(bins, stack, 5, 2)
    assert got == [1, 3]
