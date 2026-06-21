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
