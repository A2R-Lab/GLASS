"""
conftest.py — pytest configuration for GLASS CUDA tests.

Compiles test_l1/l2/l3 CUDA binaries once per session, caching by source hash.
Each binary is compiled with nvcc against the local glass.cuh.
"""

import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

# ─── paths ────────────────────────────────────────────────────────────────────

TEST_DIR  = pathlib.Path(__file__).parent
GLASS_DIR = TEST_DIR.parent
CUDA_DIR  = TEST_DIR / "cuda"
BUILD_DIR = TEST_DIR / "build"


# ─── GPU architecture detection ───────────────────────────────────────────────

def detect_arch() -> str:
    """Return nvcc arch flag like 'sm_86' by querying nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0].strip()
        major, minor = out.split(".")
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_75"  # safe fallback


CUDA_ARCH = detect_arch()


# ─── source hashing ───────────────────────────────────────────────────────────

def _hash_sources(cu_path: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in [cu_path, CUDA_DIR / "helpers.cuh",
              GLASS_DIR / "glass.cuh", GLASS_DIR / "glass-cgrps.cuh"]:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ─── compilation ──────────────────────────────────────────────────────────────

def _mathdx_root() -> pathlib.Path | None:
    root = os.environ.get("MATHDX_ROOT")
    if not root:
        return None
    root = pathlib.Path(root)
    if (root / "include" / "cublasdx.hpp").exists():
        return root
    return None


def compile_binary(name: str, build_dir: pathlib.Path, arch: str,
                   needs_cublasdx: bool = False) -> pathlib.Path | None:
    """Compile a CUDA test binary, skipping if the source hash is unchanged.

    If needs_cublasdx is True and MATHDX_ROOT is not set / invalid, returns
    None so callers can pytest.skip() the relevant tests.
    """
    cu_src    = CUDA_DIR / f"{name}.cu"
    out_bin   = build_dir / name
    hash_file = build_dir / f"{name}.hash"

    mathdx = _mathdx_root() if needs_cublasdx else None
    if needs_cublasdx and mathdx is None:
        return None

    current_hash = _hash_sources(cu_src)
    if hash_file.exists() and out_bin.exists():
        if hash_file.read_text().strip() == current_hash:
            return out_bin

    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc",
        "-std=c++17",
        f"-arch={arch}",
        "-I", str(GLASS_DIR),
        "-I", str(GLASS_DIR / "src"),
        "-I", str(CUDA_DIR),
    ]
    if needs_cublasdx:
        cmd += [
            "--expt-relaxed-constexpr",
            "-DGLASS_BENCH_CUBLASDX",
            "-I", str(mathdx / "include"),
            "-I", str(mathdx / "external" / "cutlass" / "include"),
        ]
    cmd += ["-o", str(out_bin), str(cu_src)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nCompilation failed for {name}:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"nvcc failed for {name}")
    hash_file.write_text(current_hash)
    return out_bin


# ─── session fixture ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def bins(tmp_path_factory):
    """Compile all test binaries once per pytest session. The nvidia binary
    is None when MATHDX_ROOT is not configured; tests that depend on it
    pytest.skip()."""
    build_dir = BUILD_DIR
    return {
        "l1": compile_binary("test_l1", build_dir, CUDA_ARCH),
        "l2": compile_binary("test_l2", build_dir, CUDA_ARCH),
        "l3": compile_binary("test_l3", build_dir, CUDA_ARCH),
        "nvidia": compile_binary("test_nvidia_dispatch", build_dir, CUDA_ARCH,
                                 needs_cublasdx=True),
    }


# ─── run_op helper ────────────────────────────────────────────────────────────

def run_op(binary: pathlib.Path, op: str, version: str, args: list, inputs: list):
    """
    Write numpy arrays to tempfiles, invoke the CUDA binary, parse stdout.

    Parameters
    ----------
    binary  : path to compiled CUDA binary
    op      : operation name (e.g. 'axpy')
    version : 'cg', 'simple', 'simple_lm', or 'simple_hs'
    args    : list of scalar arguments (int/float) inserted after version
    inputs  : list of numpy float32 arrays to write as .bin files
    """
    import numpy as np

    tmpfiles = []
    try:
        for arr in inputs:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            arr.astype(np.float32).tofile(f)
            f.close()
            tmpfiles.append(f.name)

        cmd = [str(binary), op, version] + [str(a) for a in args] + tmpfiles
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Binary failed:\n{result.stderr}")

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if len(lines) == 1:
            return np.fromstring(lines[0], sep=" ").astype(np.float32)
        else:
            return [np.fromstring(l, sep=" ").astype(np.float32) for l in lines]
    finally:
        for f in tmpfiles:
            os.unlink(f)
