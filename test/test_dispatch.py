"""Bare-face body-dispatch functional tests (2026-07-30 Phase 2).

The bare ``glass::op`` face routes measured (op, N, dtype) cells to a
warp-in-block or thread-in-block body (src/base/dispatch.cuh, table in
glass-dispatch.cuh). These tests run the bare spelling against the explicit
``glass::block::`` contract twin on identical inputs across block widths
{16, 32, 64, 256} and require agreement within reduction-order tolerance:

  * TB=16 exercises the narrow-block fallback (warp bodies need a full
    x-major first warp; the wrapper must fall back to the block body).
  * TB=256 is where the moved bodies win — and where a missing
    __syncthreads() after a partial-scope body would race.
  * softmax additionally checks the bounded table: n beyond the largest
    measured point stays on the block body.

Compile-time table pins (which cells moved, boundedness, unmeasured-arch
fallback, alias identity) live in cuda/test_defaults.cu.
"""

import subprocess

import pytest


OPS = ["dot_f32", "dot_f64", "gemv_f32", "chol_f32", "trsv_f32", "posv_f32",
       "eig3_f64", "softmax_f32"]


@pytest.mark.parametrize("op", OPS)
def test_bare_face_matches_block(bin_dispatch, op):
    res = subprocess.run([str(bin_dispatch), op], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"{op} rc={res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    assert "FAIL" not in res.stdout, f"{op}:\n{res.stdout}"
    assert "PASS" in res.stdout, f"{op} produced no verdicts:\n{res.stdout}"
