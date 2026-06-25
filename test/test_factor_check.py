"""CHECK flag on glass::cholDecomp_InPlace (block/warp/cgrps) and glass::ldlt.

The off-path (CHECK=false) byte-identity is covered by the existing test_l3
(chol) / test_ldlt suites still passing — this file exercises the new reporting:
a PD matrix factors with fail=0 and the correct L; a non-PD matrix sets fail=1;
ldlt reports the inertia (pivot-sign counts, == eigenvalue signs by Sylvester)
and flags a zero-pivot breakdown.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

RNG = np.random.default_rng(31)
RTOL, ATOL = 1e-2, 1e-3

CHOL_N = [1, 3, 4, 5, 7, 8, 14]
LDLT_N = [3, 4, 5, 7, 8]


def _w(a):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    a.astype(np.float32).ravel(order="F").tofile(f)
    f.close()
    return f.name


def _cholchk(binary, surf, th, n, A):
    f = _w(A)
    try:
        r = subprocess.run([str(binary), "cholchk", surf, str(th), str(n), f],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        lines = r.stdout.strip().split("\n")
        return int(lines[0]), np.fromstring(lines[1], sep=" ").reshape(n, n, order="F")
    finally:
        os.unlink(f)


def _ldltchk(binary, th, n, A):
    f = _w(A)
    try:
        r = subprocess.run([str(binary), "ldltchk", str(th), str(n), f],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        lines = r.stdout.strip().split("\n")
        sc = [int(x) for x in lines[0].split()]
        return sc, np.fromstring(lines[1], sep=" ").reshape(n, n, order="F")
    finally:
        os.unlink(f)


def _spd(n):
    M = RNG.random((n, n)).astype(np.float32)
    return (M @ M.T + n * np.eye(n)).astype(np.float32)


@pytest.mark.parametrize("n", CHOL_N)
@pytest.mark.parametrize("surf,th", [("block", 128), ("cgrps", 96), ("warp", 32)])
def test_chol_check_pd(bins, n, surf, th):
    P = _spd(n)
    fail, L = _cholchk(bins["factor_check"], surf, th, n, P.copy())
    assert fail == 0, "PD matrix should not flag"
    Lr = np.tril(L)
    assert np.allclose(Lr @ Lr.T, P, rtol=RTOL, atol=ATOL), "L L^T != A"


@pytest.mark.parametrize("n", CHOL_N)
@pytest.mark.parametrize("surf,th", [("block", 128), ("cgrps", 96), ("warp", 32)])
def test_chol_check_non_pd(bins, n, surf, th):
    ND = (-_spd(n)).astype(np.float32)      # negative-definite => first pivot <= 0
    fail, _ = _cholchk(bins["factor_check"], surf, th, n, ND.copy())
    assert fail == 1, "non-PD matrix must set fail=1"


@pytest.mark.parametrize("n", LDLT_N)
def test_ldlt_check_inertia(bins, n):
    M = RNG.random((n, n)).astype(np.float64)
    S = M + M.T + np.diag(RNG.choice([-2.0, 2.0], size=n)) * n   # indefinite, nonsingular
    sc, _ = _ldltchk(bins["factor_check"], 128, n, S.astype(np.float32).copy())
    fail, npos, nneg, nzero = sc
    ev = np.linalg.eigvalsh(S)
    epos, eneg = int((ev > 1e-6).sum()), int((ev < -1e-6).sum())
    assert fail == 0 and nzero == 0, f"nonsingular should not flag: {sc}"
    assert (npos, nneg) == (epos, eneg), f"inertia {npos,nneg} != eig signs {epos,eneg}"


def test_ldlt_check_zero_pivot(bins):
    # leading zero diagonal => non-pivoted recurrence hits a zero pivot at step 0.
    Z = np.array([[0, 1, 0], [1, 2, 0], [0, 0, 3]], dtype=np.float32)
    sc, _ = _ldltchk(bins["factor_check"], 64, 3, Z.copy())
    assert sc[0] == 1, "zero pivot must set fail=1"
