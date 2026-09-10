"""test_bdsv.py — glass::bdsv (direct block-tridiagonal SPD solve) vs numpy.

Reuses test_pcg's banded builders: SPD block-tridiagonal system in the [L|D|R]
strip layout + padded vectors. Covers the fused bdsv, factor-once/solve-twice
reuse, the THREAD_SWEEP invariance rule, and the CHECK non-SPD net.
"""
import numpy as np
import pytest

from conftest import run_op, THREAD_SWEEP
from test_pcg import make_spd_banded, to_padded, from_padded

RNG = np.random.default_rng(7)

SHAPES = [(2, 3), (6, 4), (3, 1), (1, 5), (4, 7)]   # (SS, KP); KP=1 and SS=1 edges


def _run(bins, op, SS, KP, threads, band, vecs):
    inputs = [band.ravel().astype(np.float32)] + [v.astype(np.float32) for v in vecs]
    return run_op(bins["bdsv"], op, "simple", args=[SS, KP, threads], inputs=inputs)


@pytest.mark.parametrize("SS,KP", SHAPES)
def test_bdsv_vs_dense(bins, SS, KP):
    Sd, band, _ = make_spd_banded(SS, KP, seed=SS * 13 + KP)
    b = RNG.standard_normal(SS * KP)
    xref = np.linalg.solve(Sd, b)
    out = _run(bins, "bdsv", SS, KP, 128, band, [to_padded(b, SS, KP)])
    x = from_padded(out, SS, KP)
    assert np.allclose(x, xref, rtol=1e-3, atol=1e-3), f"SS={SS} KP={KP}"


@pytest.mark.parametrize("SS,KP", [(2, 3), (6, 4)])
def test_bdsv_thread_invariance(bins, SS, KP):
    _, band, _ = make_spd_banded(SS, KP, seed=99)
    b = np.random.default_rng(100).standard_normal(SS * KP)
    ref = None
    for th in THREAD_SWEEP:
        out = _run(bins, "bdsv", SS, KP, th, band, [to_padded(b, SS, KP)])
        if ref is None:
            ref = out
        else:
            assert np.array_equal(out, ref), f"non-invariant at threads={th}"


@pytest.mark.parametrize("SS,KP", [(2, 3), (4, 7)])
def test_bdsv_factor_reuse_two_rhs(bins, SS, KP):
    Sd, band, _ = make_spd_banded(SS, KP, seed=SS + 31 * KP)
    rng = np.random.default_rng(5)
    b1, b2 = rng.standard_normal(SS * KP), rng.standard_normal(SS * KP)
    out = _run(bins, "two_rhs", SS, KP, 64, band,
               [to_padded(b1, SS, KP), to_padded(b2, SS, KP)])
    x1, x2 = from_padded(out[0], SS, KP), from_padded(out[1], SS, KP)
    assert np.allclose(x1, np.linalg.solve(Sd, b1), rtol=1e-3, atol=1e-3)
    assert np.allclose(x2, np.linalg.solve(Sd, b2), rtol=1e-3, atol=1e-3)


def test_bdsv_check_non_spd(bins):
    SS, KP = 6, 4
    _, band, _ = make_spd_banded(SS, KP, seed=17)
    band[2, :, SS:2 * SS] = -np.eye(SS)          # knot 2 diagonal block ← -I (non-SPD)
    b = np.zeros(SS * KP)
    out = _run(bins, "check", SS, KP, 64, band, [to_padded(b, SS, KP)])
    fail = int(out[0][0])
    assert fail == 1, "CHECK failed to flag a non-SPD knot"


def test_bdsv_check_good_spd(bins):
    SS, KP = 2, 3
    Sd, band, _ = make_spd_banded(SS, KP, seed=23)
    b = RNG.standard_normal(SS * KP)
    out = _run(bins, "check", SS, KP, 64, band, [to_padded(b, SS, KP)])
    fail, x = int(out[0][0]), from_padded(out[1], SS, KP)
    assert fail == 0
    assert np.allclose(x, np.linalg.solve(Sd, b), rtol=1e-3, atol=1e-3)
