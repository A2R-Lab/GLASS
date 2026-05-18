"""TRAILING_SYNC API surface tests.

Verifies that every L1/L2/L3 GLASS function exposes a `bool TRAILING_SYNC`
template parameter with both `true` and `false` specializations, and that
the two variants produce numerically identical output (when the caller
emits its own __syncthreads() after the false variant, as the docs
require).

What this catches:
  * The cuBLASDx-backed macros emit BOTH dual specializations (a missing
    one would cause link failure on the false-variant kernel).
  * The SIMT-side templates allow both compile-time values.
  * `if constexpr (TRAILING_SYNC)` gating is correct (kernel runs to
    completion either way; output matches).

Companion to test_l3_nvidia.py (existing SIMT API tests) and
test_nvidia_dispatch.py (auto-dispatch correctness).
"""

import subprocess

import pytest


def _run(binary, op):
    res = subprocess.run([str(binary), op], capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


@pytest.mark.parametrize("op", [
    "l1_dot",
    "l3_simt_batched",
    "l3_simt_strided_batched",
])
def test_trailing_sync_surface(bin_trailing_sync, op):
    """Default (true) and opt-out (false) produce identical results.

    Both kernels run; both write to separate output buffers. The test
    binary computes max_abs_diff and prints PASS only if it's below
    tolerance.
    """
    rc, stdout, stderr = _run(bin_trailing_sync, op)
    assert rc == 0, f"{op} returned {rc}\nstdout: {stdout}\nstderr: {stderr}"
    assert "PASS" in stdout, f"{op} did not print PASS:\nstdout: {stdout}"


def test_trailing_sync_cublasdx_gemm(bin_trailing_sync):
    """cuBLASDx-backed gemm: validate that BOTH macro-emitted
    specializations (TRAILING_SYNC=true and =false) link + run.

    Skips gracefully when the binary was built without
    -DGLASS_BENCH_CUBLASDX (no MathDx available).
    """
    rc, stdout, stderr = _run(bin_trailing_sync, "l3_cublasdx_gemm")
    assert rc == 0, f"l3_cublasdx_gemm returned {rc}\nstderr: {stderr}"
    if "SKIP" in stdout:
        pytest.skip("test_trailing_sync built without cuBLASDx")
    assert "PASS" in stdout, f"l3_cublasdx_gemm did not print PASS:\nstdout: {stdout}"
