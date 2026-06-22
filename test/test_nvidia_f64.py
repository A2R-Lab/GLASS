"""Double-precision validation of the glass::nvidia cuSOLVERDx/cuBLASDx wrappers.

Exercises the fp64 path added to the nvidia macros (posv = chol+trsm, gemm, gemv).
The CUDA runner (test_nvidia_f64.cu) builds a DETERMINISTIC problem; we rebuild the
identical problem here in float64 numpy and compare. Tolerances are tight (double).

Skipped unless MATHDX_ROOT + cuSOLVERDx are available (the binary won't compile
otherwise — see the bin_nvidia_f64 fixture in conftest.py).
"""
import subprocess
import numpy as np
import pytest


def _problem(N):
    """Mirror test_nvidia_f64.cu's build() exactly (column-major logical matrices)."""
    i = np.arange(N).reshape(-1, 1)
    j = np.arange(N).reshape(1, -1)
    M = ((i + 2 * j) % 5) * 0.1                      # M[i,j]
    A = M @ M.T + N * np.eye(N)                      # SPD
    B = ((i + 3 * j) % 4) * 0.1                      # gemm RHS
    b = 1.0 + 0.1 * np.arange(N)                     # posv/gemv RHS
    return A, B, b


def _run(binary, op, N):
    out = subprocess.run([str(binary), op, str(N)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"runner failed ({op} N={N}):\n{out.stderr}")
    return np.array([float(x) for x in out.stdout.split()], dtype=np.float64)


@pytest.mark.parametrize("N", [8, 16, 32])
def test_posv_f64(bin_nvidia_f64, N):
    A, _, b = _problem(N)
    x = _run(bin_nvidia_f64, "posv", N)
    assert x.shape == (N,)
    ref = np.linalg.solve(A, b)
    # SPD, well-conditioned (diag-dominant) → expect near machine precision.
    assert np.allclose(x, ref, rtol=1e-10, atol=1e-10), f"max err {np.max(np.abs(x-ref)):.2e}"
    # residual is the strongest end-to-end check (chol + 2 trsv in double)
    assert np.max(np.abs(A @ x - b)) < 1e-10


@pytest.mark.parametrize("N", [8, 16, 32])
def test_gemm_f64(bin_nvidia_f64, N):
    A, B, _ = _problem(N)
    C = _run(bin_nvidia_f64, "gemm", N)
    assert C.shape == (N * N,)
    ref = (A @ B).flatten("F")                       # runner prints column-major storage
    assert np.allclose(C, ref, rtol=1e-11, atol=1e-11), f"max err {np.max(np.abs(C-ref)):.2e}"


@pytest.mark.parametrize("N", [8, 16, 32])
def test_gemv_f64(bin_nvidia_f64, N):
    A, _, b = _problem(N)
    y = _run(bin_nvidia_f64, "gemv", N)
    assert y.shape == (N,)
    ref = A @ b
    assert np.allclose(y, ref, rtol=1e-11, atol=1e-11), f"max err {np.max(np.abs(y-ref)):.2e}"
