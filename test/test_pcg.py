"""test_pcg.py — glass::pcg::solve (block-wide PCG) vs numpy.linalg.solve.

Builds an SPD block-tridiagonal system in the [L|D|R] banded layout, a block-
Jacobi preconditioner (inverse of each diagonal block), solves it with the
single-block PCG, and checks the solution against the dense numpy solve plus a
residual bound and a warm-start early-out. Padded vector layout matches
glass::banded::bdmv (one state_size pad block on each end).
"""
import numpy as np
import pytest

from conftest import run_op


def make_spd_banded(SS, KP, seed):
    """Return (dense S, banded S [L|D|R], banded block-Jacobi Pinv)."""
    rng = np.random.default_rng(seed)
    n = SS * KP
    Sd = np.zeros((n, n))
    for k in range(KP):                                   # SPD, diagonally dominant blocks
        M = rng.standard_normal((SS, SS))
        Sd[k * SS:(k + 1) * SS, k * SS:(k + 1) * SS] = M @ M.T + SS * np.eye(SS)
    for k in range(KP - 1):                               # small symmetric off-diagonals
        R = rng.standard_normal((SS, SS)) * 0.1
        Sd[k * SS:(k + 1) * SS, (k + 1) * SS:(k + 2) * SS] = R
        Sd[(k + 1) * SS:(k + 2) * SS, k * SS:(k + 1) * SS] = R.T
    Sd = 0.5 * (Sd + Sd.T)

    band = np.zeros((KP, SS, 3 * SS))
    pinv = np.zeros((KP, SS, 3 * SS))
    for k in range(KP):
        if k > 0:
            band[k, :, 0:SS] = Sd[k * SS:(k + 1) * SS, (k - 1) * SS:k * SS]        # L
        band[k, :, SS:2 * SS] = Sd[k * SS:(k + 1) * SS, k * SS:(k + 1) * SS]       # D
        if k < KP - 1:
            band[k, :, 2 * SS:3 * SS] = Sd[k * SS:(k + 1) * SS, (k + 1) * SS:(k + 2) * SS]  # R
        pinv[k, :, SS:2 * SS] = np.linalg.inv(Sd[k * SS:(k + 1) * SS, k * SS:(k + 1) * SS])
    return Sd, band, pinv


def to_padded(v, SS, KP):
    p = np.zeros((KP + 2) * SS)
    p[SS:(KP + 1) * SS] = v
    return p


def from_padded(p, SS, KP):
    return p[SS:(KP + 1) * SS]


@pytest.mark.parametrize("SS,KP", [(2, 3), (6, 4)])
@pytest.mark.parametrize("threads", [32, 128, 256])
def test_pcg_solve(bins, SS, KP, threads):
    Sd, band, pinv = make_spd_banded(SS, KP, seed=SS * 10 + KP)
    n = SS * KP
    bvec = np.random.default_rng(SS + KP).standard_normal(n)
    xref = np.linalg.solve(Sd, bvec)

    out = run_op(bins["pcg"], "solve", "simple",
                 [SS, KP, threads, 100, 1e-6, 1e-12],
                 [band.ravel().astype(np.float32),
                  pinv.ravel().astype(np.float32),
                  to_padded(bvec, SS, KP).astype(np.float32)])
    assert isinstance(out, list) and len(out) == 2
    xgot = from_padded(out[0], SS, KP)
    iters = int(round(out[1][0]))

    np.testing.assert_allclose(xgot, xref, rtol=1e-2, atol=1e-3)
    res = np.linalg.norm(Sd @ xgot - bvec)
    assert res < 1e-2 * (np.linalg.norm(bvec) + 1.0)
    assert 0 < iters <= 100


def test_pcg_warm_start_early_out(bins):
    """Seeding the exact solution should converge in 0 iterations."""
    SS, KP = 2, 3
    Sd, band, pinv = make_spd_banded(SS, KP, seed=123)
    # b = 0 with a zero initial guess => preconditioned residual is 0 at entry.
    out = run_op(bins["pcg"], "solve", "simple",
                 [SS, KP, 64, 100, 1e-6, 1e-9],
                 [band.ravel().astype(np.float32),
                  pinv.ravel().astype(np.float32),
                  np.zeros((KP + 2) * SS, dtype=np.float32)])
    iters = int(round(out[1][0]))
    assert iters == 0
    np.testing.assert_allclose(from_padded(out[0], SS, KP), np.zeros(SS * KP), atol=1e-5)
