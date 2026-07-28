"""
conftest.py — pytest configuration for GLASS CUDA tests.

Compiles test_l1/l2/l3 CUDA binaries once per session, caching by source hash.
Each binary is compiled with nvcc against the local glass.cuh.
"""

import hashlib
import os
import pathlib
import shlex
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


# ─── thread-count sweeps (single source of truth) ─────────────────────────────
#
# GLASS's #1 invariant is thread-count invariance: every single-block op must
# produce BIT-IDENTICAL output at any block size. The bugs this catches hide at
# the counts most tests historically used (a full warp / 256), so the canonical
# sweep deliberately spans four regimes:
#
#   • 1            — single thread (serializes every grid-stride loop; exposes
#                    "loop wrote to the wrong place because size==1" bugs).
#   • 7            — low partial warp, ODD (fewer threads than most problem dims;
#                    exposes missing tail handling).
#   • 31           — one BELOW a warp boundary, odd (off-by-one at the warp edge).
#   • 32, 64       — exact warp boundaries. NOTE: races between same-value writers
#                    are INVISIBLE here (a warp runs lockstep) — these are the
#                    counts that let the inv.cuh `ind++` race pass for months.
#                    Kept so a real *value* divergence at the boundary still shows.
#   • 33, 57       — just ABOVE a warp / mid, both ODD and NON-warp-boundary
#                    (partial trailing warp + a full warp; the load-bearing cases).
#   • 96, 128, 256 — multi-warp (3,4,8 warps); exercises cross-warp reductions.
#
# Use THREAD_SWEEP for an op's dedicated thread-invariance test (one representative
# input). For tests that already fan out over a large parameter matrix, use
# THREAD_SWEEP_CORE — a cheap 4-count subset that still includes a low partial (7),
# an odd non-boundary count (33), and a multi-warp count (256). Runtime cost of a
# binary invocation is ~ms; the only real cost is the one-time nvcc recompile, so
# prefer the full sweep wherever a single input suffices.
THREAD_SWEEP      = [1, 7, 31, 32, 33, 57, 64, 96, 128, 256]
THREAD_SWEEP_CORE = [1, 7, 33, 256]


# ─── input variety (single source of truth) ───────────────────────────────────
#
# A correctness test that only ever sees one well-conditioned random matrix can
# miss sign-handling, conditioning, and pivot bugs. These makers give every test
# the same vocabulary of "kinds of input". All return float32, seedable so a
# failure reproduces. Pass distinct `seed`s to vary the draw within a sweep.

def make_spd(n, seed=0, cond=None, rng=None):
    """Random n x n symmetric positive-definite matrix (float32).

    cond=None: well-conditioned (A Aᵀ + n·I). Pass a float `cond` to force an
    approximate 2-norm condition number (eigenvalues geometrically spaced in
    [1, cond]) for ill-conditioned / near-singular factorization tests.

    Pass an existing `rng` (np.random.Generator) to draw from a caller-owned
    stream — lets a test module advance one RNG across calls for varied draws;
    otherwise a fresh `default_rng(seed)` is used (deterministic per seed)."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(seed)
    if cond is None:
        A = rng.standard_normal((n, n)).astype(np.float32)
        return (A @ A.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)
    # Q Λ Qᵀ with a controlled spectrum.
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.geomspace(1.0, float(cond), n)
    return (Q @ np.diag(eig) @ Q.T).astype(np.float32)


def make_general(m, n=None, seed=0, scale=1.0, rng=None):
    """Random m x n general matrix (float32), mean-zero so signs are mixed.

    Pass an existing `rng` to draw from a caller-owned stream (see make_spd)."""
    import numpy as np
    n = m if n is None else n
    if rng is None:
        rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal((m, n))).astype(np.float32)


def make_lower_triangular(n, seed=0, rng=None):
    """Random n x n lower-triangular matrix with a positive diagonal (float32).

    Pass an existing `rng` to draw from a caller-owned stream (see make_spd)."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(seed)
    L = np.tril(rng.standard_normal((n, n)).astype(np.float32))
    np.fill_diagonal(L, np.abs(L.diagonal()) + 0.5)
    return L.astype(np.float32)


def make_vec(n, seed=0, kind="normal"):
    """Random length-n vector (float32). kind: 'normal' (mixed sign), 'pos'
    (strictly positive), 'mixed' (alternating large/small magnitudes to stress
    reductions and 1-norms)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if kind == "pos":
        return (np.abs(rng.standard_normal(n)) + 0.1).astype(np.float32)
    if kind == "mixed":
        v = rng.standard_normal(n).astype(np.float32)
        v[::2] *= 1e3
        v[1::2] *= 1e-3
        return v
    return rng.standard_normal(n).astype(np.float32)


# ─── source hashing ───────────────────────────────────────────────────────────

def _hash_sources(cu_path: pathlib.Path) -> str:
    h = hashlib.sha256()
    # EVERY library header is hashed via the glob — a new/edited .cuh can never be
    # forgotten again (the old explicit list silently omitted 12 headers, so e.g.
    # a prefix_sum.cuh fix never rebuilt test_l1; found 2026-07-17). The named
    # test/cuda drivers stay explicit: they are shared-fixture inputs whose edits
    # must bust every binary, but globbing ALL drivers would rebust the world on
    # any single-driver edit.
    paths = [cu_path, CUDA_DIR / "helpers.cuh",
             GLASS_DIR / "glass.cuh", GLASS_DIR / "glass-cgrps.cuh",
             GLASS_DIR / "glass-defaults.cuh", GLASS_DIR / "glass-nvidia.cuh"]
    paths += sorted((GLASS_DIR / "src").rglob("*.cuh"))
    paths += [
              CUDA_DIR / "test_iamax.cu",
              CUDA_DIR / "test_symmetrize.cu",
              CUDA_DIR / "test_symm_rot.cu",
              CUDA_DIR / "test_l1_round2.cu",
              CUDA_DIR / "test_reduced_blas.cu",
              CUDA_DIR / "test_warp.cu",
              CUDA_DIR / "test_thread.cu",
              CUDA_DIR / "test_reduced.cu",
              CUDA_DIR / "test_tensor.cu",
              CUDA_DIR / "test_congruence.cu",
              CUDA_DIR / "test_syrk.cu",
              CUDA_DIR / "test_fused.cu",
              CUDA_DIR / "test_factor_check.cu",
              CUDA_DIR / "test_getrf.cu",
              CUDA_DIR / "test_ldlt.cu",
              CUDA_DIR / "test_trsv.cu",
              CUDA_DIR / "test_posv.cu",
              CUDA_DIR / "test_syev.cu",
              CUDA_DIR / "test_solve.cu",
              CUDA_DIR / "test_bdsv.cu",
              CUDA_DIR / "test_block_access.cu",
              CUDA_DIR / "test_robotics.cu",
    ]
    for p in paths:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ─── compilation ──────────────────────────────────────────────────────────────

def compile_binary(name: str, build_dir: pathlib.Path, arch: str,
                   extra_flags: list = None) -> pathlib.Path:
    """Compile a CUDA test binary, skipping if the source hash is unchanged."""
    cu_src    = CUDA_DIR / f"{name}.cu"
    out_bin   = build_dir / name
    hash_file = build_dir / f"{name}.hash"

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
        "-o", str(out_bin),
        str(cu_src),
    ]
    if extra_flags:
        cmd += extra_flags
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nCompilation failed for {name}:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"nvcc failed for {name}")
    hash_file.write_text(current_hash)
    return out_bin


# ─── session fixture ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def bins(tmp_path_factory):
    """Compile all test binaries once per pytest session."""
    build_dir = BUILD_DIR
    out = {
        "l1": compile_binary("test_l1", build_dir, CUDA_ARCH),
        "l2": compile_binary("test_l2", build_dir, CUDA_ARCH),
        "l3": compile_binary("test_l3", build_dir, CUDA_ARCH),
        "qp": compile_binary("test_qp", build_dir, CUDA_ARCH),
        "banded": compile_binary("test_banded", build_dir, CUDA_ARCH),
        "bdsv": compile_binary("test_bdsv", build_dir, CUDA_ARCH),
        "pcg": compile_binary("test_pcg", build_dir, CUDA_ARCH),
        "syrk": compile_binary("test_syrk", build_dir, CUDA_ARCH),
        "trsv": compile_binary("test_trsv", build_dir, CUDA_ARCH),
        "ldlt": compile_binary("test_ldlt", build_dir, CUDA_ARCH),
        "getrf": compile_binary("test_getrf", build_dir, CUDA_ARCH),
        "iamax": compile_binary("test_iamax", build_dir, CUDA_ARCH),
        "fused": compile_binary("test_fused", build_dir, CUDA_ARCH),
        "warp": compile_binary("test_warp", build_dir, CUDA_ARCH),
        "thread": compile_binary("test_thread", build_dir, CUDA_ARCH),
        "posv": compile_binary("test_posv", build_dir, CUDA_ARCH),
        "reduced": compile_binary("test_reduced", build_dir, CUDA_ARCH),
        "tensor": compile_binary("test_tensor", build_dir, CUDA_ARCH),
        "factor_check": compile_binary("test_factor_check", build_dir, CUDA_ARCH),
        "congruence": compile_binary("test_congruence", build_dir, CUDA_ARCH),
        "solve": compile_binary("test_solve", build_dir, CUDA_ARCH),
        "reduced_blas": compile_binary("test_reduced_blas", build_dir, CUDA_ARCH),
        "base_f64": compile_binary("test_base_f64", build_dir, CUDA_ARCH),
        "defaults": compile_binary("test_defaults", build_dir, CUDA_ARCH),
        "l1_round2": compile_binary("test_l1_round2", build_dir, CUDA_ARCH),
        "block_access": compile_binary("test_block_access", build_dir, CUDA_ARCH),
        "symmetrize": compile_binary("test_symmetrize", build_dir, CUDA_ARCH),
        "symm_rot": compile_binary("test_symm_rot", build_dir, CUDA_ARCH),
        "syev": compile_binary("test_syev", build_dir, CUDA_ARCH),
        "robotics": compile_binary("test_robotics", build_dir, CUDA_ARCH),
    }
    # test_l3_nvidia.cu includes glass-nvidia.cuh and exercises the SIMT-only
    # batched APIs (gemm_batched_1d, gemm_strided_batched_1d). It does NOT
    # require cuBLASDx — l3_simt.cuh has no cuBLASDx dependency. If compilation
    # fails for any reason (e.g. a non-cuBLASDx-related toolchain issue), tests
    # that depend on it will be skipped via the `bin_l3_nvidia` fixture below.
    try:
        out["l3_nvidia"] = compile_binary("test_l3_nvidia", build_dir, CUDA_ARCH)
    except Exception as e:
        print(f"\nSkipping test_l3_nvidia (compile failed): {e}", file=sys.stderr)

    # test_nvidia_dispatch.cu exercises round-2 auto-dispatch features
    # (Gap A/B/C/D, gemv + row_strided_* + gemm-with-TRANSPOSE_B, dispatch
    # query helpers). Requires cuBLASDx for the gemm cuBLASDx-route test,
    # so it needs MATHDX_ROOT to compile; otherwise skipped at the fixture.
    mathdx = os.environ.get("MATHDX_ROOT")
    cublasdx_available = bool(mathdx) and (pathlib.Path(mathdx) / "include" / "cublasdx.hpp").exists() if mathdx else False
    if cublasdx_available:
        try:
            out["nvidia_dispatch"] = compile_binary(
                "test_nvidia_dispatch", build_dir, CUDA_ARCH,
                extra_flags=[
                    "--expt-relaxed-constexpr",
                    "-DGLASS_BENCH_CUBLASDX",
                    "-I", str(pathlib.Path(mathdx) / "include"),
                    "-I", str(pathlib.Path(mathdx) / "external" / "cutlass" / "include"),
                ])
        except Exception as e:
            print(f"\nSkipping test_nvidia_dispatch (compile failed): {e}", file=sys.stderr)

    # test_trailing_sync.cu exercises the bool TRAILING_SYNC template
    # parameter across the L1/L2/L3 surface. Compile both variants
    # (with and without cuBLASDx) — the binary internally skips the
    # cuBLASDx op when GLASS_BENCH_CUBLASDX isn't defined.
    try:
        flags = []
        if cublasdx_available:
            flags = [
                "--expt-relaxed-constexpr",
                "-DGLASS_BENCH_CUBLASDX",
                "-I", str(pathlib.Path(mathdx) / "include"),
                "-I", str(pathlib.Path(mathdx) / "external" / "cutlass" / "include"),
            ]
        out["trailing_sync"] = compile_binary(
            "test_trailing_sync", build_dir, CUDA_ARCH, extra_flags=flags)
    except Exception as e:
        print(f"\nSkipping test_trailing_sync (compile failed): {e}", file=sys.stderr)

    # test_nvidia_f64.cu validates the DOUBLE-precision nvidia path (cuSOLVERDx
    # posv / cuBLASDx gemm,gemv). Needs cuSOLVERDx (headers + libcusolverdx.a,
    # which is LTO so the link wants -rdc=true -dlto). Skipped if absent.
    cusolverdx_available = (
        cublasdx_available
        and (pathlib.Path(mathdx) / "include" / "cusolverdx.hpp").exists()
        and (pathlib.Path(mathdx) / "include" / "cusolverdx_io.hpp").exists()
        and (pathlib.Path(mathdx) / "lib" / "libcusolverdx.a").exists()
    )
    if cusolverdx_available:
        sms = CUDA_ARCH.replace("sm_", "") + "0"   # sm_120 -> 1200
        try:
            out["nvidia_f64"] = compile_binary(
                "test_nvidia_f64", build_dir, CUDA_ARCH,
                extra_flags=[
                    "--expt-relaxed-constexpr",
                    "-DGLASS_BENCH_CUBLASDX", "-DGLASS_BENCH_CUSOLVERDX",
                    "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", f"-DSMS={sms}",
                    "-I", str(pathlib.Path(mathdx) / "include"),
                    "-I", str(pathlib.Path(mathdx) / "external" / "cutlass" / "include"),
                    "-rdc=true", "-dlto",
                    "-L", str(pathlib.Path(mathdx) / "lib"),
                    "-lcusolverdx", "-lcublas", "-lcusolver", "-lcudart",
                ])
        except Exception as e:
            print(f"\nSkipping test_nvidia_f64 (compile failed): {e}", file=sys.stderr)
    return out


@pytest.fixture(scope="session")
def bin_l3_nvidia(bins):
    """Yield the test_l3_nvidia binary, or skip the test if it didn't compile."""
    if "l3_nvidia" not in bins:
        pytest.skip("test_l3_nvidia.cu failed to compile")
    return bins["l3_nvidia"]


@pytest.fixture(scope="session")
def bin_nvidia_dispatch(bins):
    """Round-2 auto-dispatch tests; skip if MATHDX_ROOT isn't configured or
    the test failed to compile."""
    if "nvidia_dispatch" not in bins:
        pytest.skip("test_nvidia_dispatch needs MATHDX_ROOT (cuBLASDx)")
    return bins["nvidia_dispatch"]


@pytest.fixture(scope="session")
def bin_trailing_sync(bins):
    """TRAILING_SYNC surface tests; skip if the binary didn't compile."""
    if "trailing_sync" not in bins:
        pytest.skip("test_trailing_sync failed to compile")
    return bins["trailing_sync"]


@pytest.fixture(scope="session")
def bin_nvidia_f64(bins):
    """Double-precision nvidia path; skip if cuSOLVERDx isn't available."""
    if "nvidia_f64" not in bins:
        pytest.skip("test_nvidia_f64 needs MATHDX_ROOT + cuSOLVERDx")
    return bins["nvidia_f64"]


@pytest.fixture(scope="session")
def bin_base_f64(bins):
    """Double-precision base (glass::) + warp (glass::warp::) ops."""
    return bins["base_f64"]


@pytest.fixture(scope="session")
def bin_defaults(bins):
    """Compile-time backend-defaults helpers (static_asserts validate at compile)."""
    return bins["defaults"]


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

        # GLASS_RUN_PREFIX lets a wrapper (e.g. compute-sanitizer) wrap every
        # kernel launch. Sanitizer diagnostics go to stderr, so parsed stdout
        # stays clean; pair with --error-exitcode so a finding trips returncode.
        prefix = shlex.split(os.environ.get("GLASS_RUN_PREFIX", ""))
        cmd = prefix + [str(binary), op, version] + [str(a) for a in args] + tmpfiles
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Binary failed:\n{result.stdout}\n{result.stderr}")

        # Drop compute-sanitizer's own "=========" banner lines (stdout) so the
        # numeric parse below is unaffected when GLASS_RUN_PREFIX wraps the launch.
        lines = [l.strip() for l in result.stdout.strip().split("\n")
                 if l.strip() and not l.lstrip().startswith("=========")]
        if len(lines) == 1:
            return np.fromstring(lines[0], sep=" ").astype(np.float32)
        else:
            return [np.fromstring(l, sep=" ").astype(np.float32) for l in lines]
    finally:
        for f in tmpfiles:
            os.unlink(f)


# ─── gpu-proof integration (pytest-gpu-proof, pinned in test/requirements.txt) ──
# test/run_gpu_proof.sh runs this suite with --gpu-proof-enable to emit a signed
# receipt (test/gpu-proof.json) that the CPU-only verify-gpu-proof CI checks.
# The plugin only records tests carrying the gpu_proof marker, so mark every
# collected test. The marker is registered here too so plain runs (plugin not
# installed) stay warning-free.

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu_proof: recorded in the signed GPU-run receipt")


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.gpu_proof)
