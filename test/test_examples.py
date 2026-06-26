"""Smoke test: every pure-SIMT example in examples/ compiles and runs (exit 0).

Keeps the worked examples (which document the standard BLAS/Eigen GEMM
convention, `row-major == transpose`, nrm2, strided GEMM, …) from bit-rotting as
the API evolves. The host-verified examples return non-zero on a numeric
mismatch, so a clean exit is a real correctness check, not just "it linked".

The one MathDx/cuBLASDx example (06_nvidia_gemm) is skipped unless MATHDX_ROOT is
present, mirroring how conftest gates the nvidia binaries.
"""

import os
import pathlib
import subprocess

import pytest

from conftest import GLASS_DIR, CUDA_ARCH

EXAMPLES_DIR = GLASS_DIR / "examples"

# Pure-SIMT examples (no MathDx). 06_nvidia_gemm is handled separately.
SIMT_EXAMPLES = sorted(
    p.name for p in EXAMPLES_DIR.glob("*.cu") if not p.name.startswith("06_")
)


def _compile_and_run(tmp_path, src_name, extra_flags=None):
    src = EXAMPLES_DIR / src_name
    out = tmp_path / (src.stem)
    cmd = ["nvcc", "-std=c++17", f"-arch={CUDA_ARCH}", "-I", str(GLASS_DIR),
           str(src), "-o", str(out)] + (extra_flags or [])
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        pytest.fail(f"compile failed for {src_name}:\n{cp.stderr}")
    rp = subprocess.run([str(out)], capture_output=True, text=True)
    assert rp.returncode == 0, f"{src_name} exited {rp.returncode}:\n{rp.stdout}\n{rp.stderr}"


@pytest.mark.parametrize("src_name", SIMT_EXAMPLES)
def test_simt_example(tmp_path, src_name):
    _compile_and_run(tmp_path, src_name)


def test_nvidia_example(tmp_path):
    mathdx = os.environ.get("MATHDX_ROOT")
    if not (mathdx and (pathlib.Path(mathdx) / "include" / "cublasdx.hpp").exists()):
        pytest.skip("06_nvidia_gemm needs MATHDX_ROOT (cuBLASDx)")
    _compile_and_run(tmp_path, "06_nvidia_gemm.cu", extra_flags=[
        "--expt-relaxed-constexpr", "-DGLASS_BENCH_CUBLASDX", "-DSMS=860",
        "-I", str(pathlib.Path(mathdx) / "include"),
        "-I", str(pathlib.Path(mathdx) / "external" / "cutlass" / "include"),
    ])
