"""glass::posv / potrs tests — SPD solve via Cholesky + two triangular solves.

The runner CLI is `<posv|potrs> <n> <threads> <A_or_L.bin> <b.bin>`, so we invoke
it directly rather than via conftest.run_op.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import THREAD_SWEEP, make_spd

RNG = np.random.default_rng(11)
RTOL, ATOL = 1e-2, 1e-3

# warp ops occupy exactly one 32-lane warp on shared per-problem data; launching
# more than a warp would have every warp race on the same A/b, fewer would break
# the full-mask shfl. So warp solves are validated at exactly 32 threads.
WARP = 32


def _run(binary, op, n, threads, M, b):
    """Write M (n*n col-major) and b as float32 .bin, run, parse x."""
    tmp = []
    try:
        for arr in (np.asfortranarray(M).ravel(order="F"), b):
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f); f.close(); tmp.append(f.name)
        cmd = [str(binary), op, str(n), str(threads), tmp[0], tmp[1]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        return np.fromstring(res.stdout.strip().split("\n")[0], sep=" ").astype(np.float32)
    finally:
        for f in tmp:
            os.unlink(f)


def _run_m(binary, op, n, nrhs, threads, M, B):
    """Multi-RHS runner: M (n*n col-major), B (n*nrhs col-major) -> X (n*nrhs).

    Returns X as an (n, nrhs) float32 array (un-flattened from col-major).
    """
    tmp = []
    try:
        for arr in (np.asfortranarray(M).ravel(order="F"),
                    np.asfortranarray(B).ravel(order="F")):
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f); f.close(); tmp.append(f.name)
        cmd = [str(binary), op, str(n), str(nrhs), str(threads), tmp[0], tmp[1]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        flat = np.fromstring(res.stdout.strip().split("\n")[0], sep=" ").astype(np.float32)
        return flat.reshape((n, nrhs), order="F")
    finally:
        for f in tmp:
            os.unlink(f)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_posv(bins, n):
    A = make_spd(n, rng=RNG)
    b = RNG.standard_normal(n).astype(np.float32)
    x = _run(bins["posv"], "posv", n, 256, A, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=RTOL, atol=ATOL)
    # residual backstop
    assert np.allclose(A @ x, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_potrs(bins, n):
    """Precomputed-factor path: pass the host Cholesky L, never the full A."""
    A = make_spd(n, rng=RNG)
    L = np.linalg.cholesky(A).astype(np.float32)   # lower
    b = RNG.standard_normal(n).astype(np.float32)
    x = _run(bins["posv"], "potrs", n, 256, L, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
@pytest.mark.parametrize("nrhs", [1, 2, 3, 5])
def test_posv_multirhs(bins, n, nrhs):
    """Multi-RHS factor+solve: X (n×nrhs col-major) == np.linalg.solve(A, B)."""
    A = make_spd(n, rng=RNG)
    B = RNG.standard_normal((n, nrhs)).astype(np.float32)
    X = _run_m(bins["posv"], "posv_m", n, nrhs, 256, A, B)
    expected = np.linalg.solve(A, B)
    assert np.allclose(X, expected, rtol=RTOL, atol=ATOL)
    # residual backstop
    assert np.allclose(A @ X, B, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
@pytest.mark.parametrize("nrhs", [1, 2, 3, 5])
def test_potrs_multirhs(bins, n, nrhs):
    """Multi-RHS precomputed-factor path: pass host Cholesky L, solve n×nrhs B."""
    A = make_spd(n, rng=RNG)
    L = np.linalg.cholesky(A).astype(np.float32)   # lower
    B = RNG.standard_normal((n, nrhs)).astype(np.float32)
    X = _run_m(bins["posv"], "potrs_m", n, nrhs, 256, L, B)
    expected = np.linalg.solve(A, B)
    assert np.allclose(X, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("op", ["posv_m", "potrs_m"])
@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("nrhs", [1, 3, 5])
def test_posv_multirhs_thread_invariance(bins, op, n, nrhs):
    """Barrier correctness: identical X at 1/7/33/256 threads, matching the oracle."""
    A = make_spd(n, rng=RNG)
    B = RNG.standard_normal((n, nrhs)).astype(np.float32)
    M = np.linalg.cholesky(A).astype(np.float32) if op == "potrs_m" else A
    expected = np.linalg.solve(A, B)
    outs = []
    for threads in THREAD_SWEEP:
        X = _run_m(bins["posv"], op, n, nrhs, threads, M, B)
        assert np.allclose(X, expected, rtol=RTOL, atol=ATOL), f"threads={threads} mismatch"
        outs.append(X)
    for X in outs[1:]:
        assert np.array_equal(outs[0], X), "thread-count non-invariance"


@pytest.mark.parametrize("op", ["posv", "potrs"])
@pytest.mark.parametrize("n", [4, 6, 8])
def test_posv_thread_invariance(bins, op, n):
    """Barrier correctness: identical x at 1/7/33/256 threads, matching the oracle."""
    A = make_spd(n, rng=RNG)
    b = RNG.standard_normal(n).astype(np.float32)
    M = np.linalg.cholesky(A).astype(np.float32) if op == "potrs" else A
    expected = np.linalg.solve(A, b)
    outs = []
    for threads in THREAD_SWEEP:
        x = _run(bins["posv"], op, n, threads, M, b)
        assert np.allclose(x, expected, rtol=RTOL, atol=ATOL), f"threads={threads} mismatch"
        outs.append(x)
    for x in outs[1:]:
        assert np.array_equal(outs[0], x), "thread-count non-invariance"


# ─── flagged solves: REGULARIZE (rho·I) / REG_DIAG (Levenberg) / CHECK ─────────
#
# The flagged kernels are compiled for n=7 (HJCD's LM DIM); b is an n×1 RHS routed
# through the multi-RHS overload (NRHS=1). The runner prints x on line 1 and the
# CHECK s_fail flag (0/1) on line 2.
FN = 7


def _run_flag(binary, op, threads, A, b, rho):
    """Run a flagged posv (block or warp). Returns (x[7], s_fail:int)."""
    tmp = []
    try:
        for arr in (np.asfortranarray(A).ravel(order="F"), b):
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f); f.close(); tmp.append(f.name)
        cmd = [str(binary), op, str(FN), str(threads), tmp[0], tmp[1], repr(float(rho))]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed:\n{res.stderr}")
        lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
        x = np.fromstring(lines[0], sep=" ").astype(np.float32)
        s_fail = int(lines[1].strip())
        return x, s_fail
    finally:
        for f in tmp:
            os.unlink(f)


@pytest.mark.parametrize("mode", ["reg", "regdiag"])
@pytest.mark.parametrize("seed", [0, 1])
def test_posv_flag_block(bins, mode, seed):
    """Block REGULARIZE (rho·I) / REG_DIAG (rho·diag(A)) shift, swept over the full
    thread set with byte-identical output. Oracle solves the shifted system."""
    rho = 0.5
    A = make_spd(FN, seed=seed)
    b = np.random.default_rng(100 + seed).standard_normal(FN).astype(np.float32)
    if mode == "reg":
        Ashift = A + rho * np.eye(FN, dtype=np.float32)
    else:  # Levenberg: A + rho*diag(A)
        Ashift = A + rho * np.diag(np.diag(A)).astype(np.float32)
    expected = np.linalg.solve(Ashift, b)
    op = f"posv_{mode}"
    ref = None
    for threads in THREAD_SWEEP:
        x, s_fail = _run_flag(bins["posv"], op, threads, A, b, rho)
        assert np.allclose(x, expected, rtol=RTOL, atol=ATOL), \
            f"{op} seed={seed} threads={threads}: mismatch vs shifted-solve"
        assert s_fail == 0, "no CHECK flag requested; s_fail must stay 0"
        if ref is None:
            ref = x
        else:
            assert np.array_equal(x, ref), f"{op} threads={threads}: non-invariant"


def test_posv_flag_check_block(bins):
    """Block CHECK: s_fail=0 on SPD (and x correct), s_fail=1 on a non-PD matrix."""
    b = RNG.standard_normal(FN).astype(np.float32)
    # SPD → flag stays 0, solve is correct.
    A = make_spd(FN, seed=7)
    for threads in THREAD_SWEEP:
        x, s_fail = _run_flag(bins["posv"], "posv_check", threads, A, b, 0.0)
        assert s_fail == 0, f"SPD threads={threads}: s_fail should be 0"
        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL, atol=ATOL)
    # Non-PD (negative-definite) → first pivot ≤ 0 → flag 1.
    Aneg = (-make_spd(FN, seed=8)).astype(np.float32)
    for threads in THREAD_SWEEP:
        _, s_fail = _run_flag(bins["posv"], "posv_check", threads, Aneg, b, 0.0)
        assert s_fail == 1, f"non-PD threads={threads}: s_fail should be 1"


@pytest.mark.parametrize("mode", ["reg", "regdiag", "check"])
def test_posv_flag_warp(bins, mode):
    """Warp parity: warp::posv flagged forms match the block/oracle at one warp."""
    b = RNG.standard_normal(FN).astype(np.float32)
    rho = 0.5
    if mode == "check":
        A = make_spd(FN, seed=9)
        x, s_fail = _run_flag(bins["posv"], "posv_warp_check", WARP, A, b, 0.0)
        assert s_fail == 0
        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL, atol=ATOL)
        Aneg = (-make_spd(FN, seed=10)).astype(np.float32)
        _, s_fail = _run_flag(bins["posv"], "posv_warp_check", WARP, Aneg, b, 0.0)
        assert s_fail == 1
        return
    A = make_spd(FN, seed=3)
    if mode == "reg":
        Ashift = A + rho * np.eye(FN, dtype=np.float32)
    else:
        Ashift = A + rho * np.diag(np.diag(A)).astype(np.float32)
    expected = np.linalg.solve(Ashift, b)
    x, s_fail = _run_flag(bins["posv"], f"posv_warp_{mode}", WARP, A, b, rho)
    assert s_fail == 0
    assert np.allclose(x, expected, rtol=RTOL, atol=ATOL), f"warp {mode}: mismatch"
