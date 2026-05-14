"""Tests for the glass::nvidia auto-dispatch (Phase 1 + 2 + 3 of round-1/2).

Each op in test_nvidia_dispatch.cu writes "PASS" or "FAIL" plus a max-error
to stdout. Returncode is 0 on PASS, 1 on FAIL.

Skips entire module if MATHDX_ROOT is not configured.
"""

import subprocess

import pytest


@pytest.fixture(scope="module")
def nvidia_bin(bins):
    if bins["nvidia"] is None:
        pytest.skip("MATHDX_ROOT not configured — cuBLASDx tests skipped")
    return bins["nvidia"]


def _run(nvidia_bin, op):
    res = subprocess.run([str(nvidia_bin), op], capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


@pytest.mark.parametrize("op", [
    "gemm_simt",     # 6x6x6 auto-routes to SIMT; matches CPU reference
    "gemm_cublas",   # 8x8x8 routes to cuBLASDx via DEFINE; bit-parity with SIMT
    "gemm_transb",   # LB=row_major routes SIMT through TRANSPOSE_B=true
    "strided_gemv",  # row_strided_gemv SIMT with non-tight stride
    "batched_simt",  # gemm_batched_1d (BATCH=4) SIMT vs loop
    "batched_dx",    # gemm_batched_1d (BATCH=16,8x8x8) cuBLASDx vs SIMT
])
def test_dispatch_op(nvidia_bin, op):
    rc, stdout, stderr = _run(nvidia_bin, op)
    assert rc == 0, f"{op} returned {rc}\nstdout: {stdout}\nstderr: {stderr}"
    assert "PASS" in stdout, f"{op} did not print PASS:\nstdout: {stdout}"


def test_dispatch_query(nvidia_bin):
    """print_dispatch_full prints one line per shape with source info."""
    rc, stdout, _ = _run(nvidia_bin, "dispatch_q")
    assert rc == 0
    # Spot-check: 6x6x6 should report SIMT, 24x24x24 should report cuBLASDx.
    assert "gemm<float,6,6,6" in stdout and "SIMT" in stdout
    assert "gemm<float,24,24,24" in stdout and "cuBLASDx" in stdout
