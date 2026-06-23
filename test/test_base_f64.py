"""Double-precision validation of the pure-SIMT base ops (``glass::``) and warp ops
(``glass::warp::``). The rest of the suite is f32-only; these ops are templated on the
scalar type, so this covers the f64 instantiations (correctness + f64 thread-invariance).

Oracle-free: residual / reconstruction checks at f64 tolerance.
"""
import subprocess
import numpy as np
import pytest

OPS = ["dot", "gemv", "gemm", "chol", "trsv", "posv"]
SURFACES = ["block", "warp"]
SIZES = [8, 16, 32]


def _problem(N):
    """Mirror test_base_f64.cu's build() exactly (column-major logical matrices)."""
    i = np.arange(N).reshape(-1, 1)
    j = np.arange(N).reshape(1, -1)
    M = ((i + 2 * j) % 5) * 0.1
    A = M @ M.T + N * np.eye(N)
    B = ((i + 3 * j) % 4) * 0.1
    L = np.tril(((i + 2 * j) % 5) * 0.1, -1) + N * np.eye(N)   # lower-tri, dominant diag
    b = 1.0 + 0.1 * np.arange(N)
    return A, B, L, b


def _run(binary, op, surface, N, threads=64):
    out = subprocess.run([str(binary), op, surface, str(N), str(threads)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"runner failed ({op}/{surface}/N={N}):\n{out.stderr}")
    return np.array([float(x) for x in out.stdout.split()], dtype=np.float64)


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("N", SIZES)
def test_base_f64_correctness(bin_base_f64, op, surface, N):
    A, B, L, b = _problem(N)
    out = _run(bin_base_f64, op, surface, N)
    if op == "dot":
        assert abs(out[0] - b @ b) < 1e-10
    elif op == "gemv":
        assert np.max(np.abs(out - A @ b)) < 1e-10
    elif op == "gemm":
        assert np.max(np.abs(out - (A @ B).flatten("F"))) < 1e-9
    elif op == "chol":
        Lc = np.tril(out.reshape(N, N, order="F"))          # GPU writes L in the lower triangle
        assert np.max(np.abs(Lc @ Lc.T - A)) < 1e-9         # reconstruct A
    elif op == "trsv":
        assert np.max(np.abs(L @ out - b)) < 1e-10          # residual of L x = b
    else:  # posv
        assert np.max(np.abs(A @ out - b)) < 1e-10          # residual of A x = b


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("N", SIZES)
def test_base_f64_thread_invariance(bin_base_f64, op, N):
    """Block-scoped ops must be bit-identical across thread counts (the barrier-bug
    class) — verified here in f64, including a non-multiple-of-32."""
    ref = _run(bin_base_f64, op, "block", N, threads=1)
    for tb in (7, 33, 64, 256):
        got = _run(bin_base_f64, op, "block", N, threads=tb)
        assert np.array_equal(got, ref), f"{op} N={N}: tb={tb} differs from tb=1"
