"""Selected-operand regression: a thread-tier product whose read operand is
chosen at runtime between a thread-local array and shared memory.

`test/cuda/test_select.cu` replays the lane-parallel forward-kinematics suffix
that exposed the second `__restrict__` miscompile class (2026-10-31): with the
qualifier on the read operand of the compile-time gemm path, nvcc dropped the
inlined writer's stores to the local array, and the result was nondeterministic
between two repeats of one launch. Two checks, both precisions:

* the two repeats are bit-identical (no oracle needed for the fault itself);
* every lane matches an independent numpy chain of the same transforms, built
  from the fixture's geometry (fixed rpy .1 0 .03 and origin .02 .01 .08, joint
  axes y, -z, x prismatic, y, -z, x, y) rather than from the generated writer.
"""
import os
import subprocess
import tempfile

import numpy as np
import pytest

from conftest import RUN_TIMEOUT

N = 7
DTYPES = ["f32", "f64"]
_NPDT = {"f32": np.float32, "f64": np.float64}
_TOL = {"f32": dict(rtol=2e-5, atol=5e-5), "f64": dict(rtol=1e-12, atol=1e-12)}


def _rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_RFIX = _rot("z", 0.03) @ _rot("y", 0.0) @ _rot("x", 0.1)
_P = np.array([0.02, 0.01, 0.08])


def joint_transform(j, th):
    """Column-major-agnostic 4x4 of joint j at coordinate th (fixture geometry)."""
    T = np.eye(4)
    if j % 3 == 2 and j % 5 == 2:          # j = 2: prismatic along local x
        T[:3, :3] = _RFIX
        T[:3, 3] = _P + _RFIX @ np.array([th, 0.0, 0.0])
        return T
    axis, sign = [("y", 1.0), ("z", -1.0), ("x", 1.0)][j % 3]
    T[:3, :3] = _RFIX @ _rot(axis, sign * th)
    T[:3, 3] = _P
    return T


def oracle(anchors, q):
    """anchors: (2, 32*N) float; q: (2, N). Returns (2, N, 16) column-major results."""
    out = np.zeros((2, N, 16))
    for warp in range(2):
        world = [anchors[warp, 16*j:16*j+16].reshape(4, 4, order="F") for j in range(N)]
        local = [anchors[warp, 16*(N+j):16*(N+j)+16].reshape(4, 4, order="F") for j in range(N)]
        for changed in range(N):
            W = world[changed-1] if changed else np.eye(4)
            for j in range(changed, N):
                W = W @ (joint_transform(j, q[warp, changed]) if j == changed else local[j])
            out[warp, changed] = W.reshape(16, order="F")
    return out


def _run(binary, dtype, anchors, q):
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    try:
        np.concatenate([anchors.ravel(), q.ravel()]).astype(np.float32).tofile(tmp.name)
        tmp.close()
        result = subprocess.run([str(binary), dtype, tmp.name], capture_output=True,
                                text=True, timeout=RUN_TIMEOUT)
    finally:
        os.unlink(tmp.name)
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr)
    out = np.array(result.stdout.split(), dtype=np.float64)
    assert out.size == 2*64*16, out.size
    return out.reshape(2, 2, 32, 16)      # repeat, warp, lane, element


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
def test_selected_operand_thread_gemm(bins, dtype, seed):
    rng = np.random.default_rng(100 + seed)
    anchors = rng.uniform(-1.0, 1.0, size=(2, 32*N)).astype(np.float32)
    q = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    out = _run(bins["select"], dtype, anchors, q)
    lanes = out[:, :, :N, :]
    assert np.isfinite(lanes).all()
    assert np.array_equal(lanes[0], lanes[1]), "nondeterministic between two repeats of one launch"
    assert (out[:, :, N:, :] == 789).all()
    expected = oracle(anchors.astype(np.float64), q.astype(np.float64))
    np.testing.assert_allclose(lanes[0], expected, **_TOL[dtype])
