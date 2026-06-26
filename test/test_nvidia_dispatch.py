"""Round-2 auto-dispatch correctness tests.

Companion to test_l3_nvidia.py (which tests the SIMT batched APIs from
l3_simt.cuh). This module targets the round-2 additions:

  * Gap A — glass::nvidia::gemv<>     auto-dispatches SIMT vs cuBLASDx
  * Gap B — gemv_strided<>        auto-dispatches; SIMT uses stride directly
  * Gap C — gemm_strided<>        auto-dispatches; SIMT skips compact-pack
  * Gap D — gemm<...,LB=row_major,...> maps onto SIMT TRANSPOSE_B=true

The test binary requires cuBLASDx for the gemm_cublas op (compiles via DEFINE),
so the whole module skips when MATHDX_ROOT isn't set (see conftest.py).
"""

import subprocess

import pytest


def _run(nvidia_bin, op):
    res = subprocess.run([str(nvidia_bin), op], capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


@pytest.mark.parametrize("op", [
    "gemm_simt",      # 6x6x6 auto-routes to SIMT (no DEFINE); matches CPU
    "gemm_cublas",    # 16x16x16 routes to cuBLASDx via shipped DEFINE; bit-parity
    "gemm_transb",    # Gap D — LB=row_major maps to SIMT TRANSPOSE_B=true
    "gemv_simt",      # Gap A — gemv 6x6 SIMT
    "strided_gemv",   # Gap B — non-tight stride
    "strided_gemm",   # Gap C — non-tight A_RS/B_RS
])
def test_dispatch_op(bin_nvidia_dispatch, op):
    rc, stdout, stderr = _run(bin_nvidia_dispatch, op)
    assert rc == 0, f"{op} returned {rc}\nstdout: {stdout}\nstderr: {stderr}"
    assert "PASS" in stdout, f"{op} did not print PASS:\nstdout: {stdout}"


def test_dispatch_query(bin_nvidia_dispatch):
    """print_dispatch<> reports SIMT for small shapes and cuBLASDx for large."""
    rc, stdout, _ = _run(bin_nvidia_dispatch, "dispatch_q")
    assert rc == 0
    # 6x6x6 → SIMT, 16x16x16 → cuBLASDx (matches the shipped tuning + heuristic).
    assert "glass::nvidia::gemm<T,6,6,6" in stdout and "SIMT" in stdout
    assert "glass::nvidia::gemm<T,16,16,16" in stdout and "cuBLASDx" in stdout
