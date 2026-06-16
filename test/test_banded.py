"""test_banded.py — glass::banded::bdmv (block-tridiagonal matvec) vs numpy.

Validates the single + dual-output bdmv against a dense block-tridiagonal
reference. Uses ASYMMETRIC L/R off-diagonal blocks so an L/R swap is caught,
exercises the edge block-rows (row 0 with a zero L-pad, row N-1 with a zero
R-pad), and checks thread-count invariance (1 / 32 / 256 threads).
"""
import numpy as np
import pytest

from conftest import run_op


def make_banded(BS, NBR, seed):
    """Random [L|D|R] row-major strips + a padded vector (pads zeroed)."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((NBR, BS, 3 * BS)).astype(np.float32)   # asymmetric L/R
    vec = rng.standard_normal((NBR + 2) * BS).astype(np.float32)
    vec[:BS] = 0.0          # leading pad  (block-row 0's absent L)
    vec[-BS:] = 0.0         # trailing pad (block-row N-1's absent R)
    return mat, vec


def reference(mat, vec, BS, NBR):
    """Dense block-tridiagonal matvec into the padded output layout."""
    out = np.zeros((NBR + 2) * BS, dtype=np.float64)
    v = vec.astype(np.float64)
    for br in range(NBR):
        strip = mat[br].astype(np.float64)          # BS x 3BS
        window = v[br * BS: br * BS + 3 * BS]        # [prev | cur | next]
        out[(br + 1) * BS:(br + 2) * BS] = strip @ window
    return out


@pytest.mark.parametrize("BS,NBR", [(2, 3), (6, 4)])
@pytest.mark.parametrize("threads", [1, 32, 256])
def test_bdmv(bins, BS, NBR, threads):
    mat, vec = make_banded(BS, NBR, seed=BS * 100 + NBR)
    ref = reference(mat, vec, BS, NBR)
    got = run_op(bins["banded"], "bdmv", "simple",
                 [BS, NBR, threads], [mat.ravel(), vec])
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("BS,NBR", [(2, 3), (6, 4)])
def test_bdmv_dual(bins, BS, NBR):
    mat, vec = make_banded(BS, NBR, seed=BS * 100 + NBR + 7)
    ref = reference(mat, vec, BS, NBR)
    out = run_op(bins["banded"], "bdmv_dual", "simple",
                 [BS, NBR, 256], [mat.ravel(), vec])
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_allclose(out[0], ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(out[1], ref, rtol=1e-3, atol=1e-3)


def test_bdmv_asymmetry_guard():
    """Sanity: the reference distinguishes L from R, so the test would catch a swap."""
    BS, NBR = 2, 3
    mat, vec = make_banded(BS, NBR, seed=999)
    ref = reference(mat, vec, BS, NBR)
    sw = mat.copy()
    sw[:, :, :BS], sw[:, :, 2 * BS:] = mat[:, :, 2 * BS:].copy(), mat[:, :, :BS].copy()
    ref_sw = reference(sw, vec, BS, NBR)
    assert not np.allclose(ref, ref_sw)
