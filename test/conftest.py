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
from collections.abc import Mapping

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

VECTOR_BINS = {"test_l1", "test_l1_round2", "test_iamax", "test_symmetrize", "test_api_vector"}
DENSE_BINS = {
    "test_l2", "test_l3", "test_syrk", "test_reduced", "test_reduced_blas",
    "test_tensor", "test_congruence", "test_block_access", "test_symm_rot", "test_api_dense",
}
FACTOR_BINS = {
    "test_fused", "test_factor_check", "test_getrf", "test_ldlt", "test_trsv",
    "test_posv", "test_syev", "test_solve", "test_base_f64", "test_api_factor",
}
TIER_BINS = {"test_warp", "test_thread", "test_defaults", "test_dispatch", "test_trailing_sync"}
SOLVER_BINS = {"test_qp", "test_banded", "test_bdsv", "test_pcg"}
NVIDIA_BINS = {"test_l3_nvidia", "test_nvidia_dispatch", "test_nvidia_f64"}


def _family_headers(binary_name: str) -> list[pathlib.Path]:
    common = [
        GLASS_DIR / "src/base/barrier.cuh",
        GLASS_DIR / "src/base/dispatch.cuh",
        GLASS_DIR / "src/base/flags.cuh",
    ]
    l1 = sorted((GLASS_DIR / "src/base/L1").glob("*.cuh"))
    l2 = sorted((GLASS_DIR / "src/base/L2").glob("*.cuh"))
    l3 = sorted((GLASS_DIR / "src/base/L3").glob("*.cuh"))
    cgrps = sorted((GLASS_DIR / "src/cgrps").glob("*.cuh"))
    if binary_name in VECTOR_BINS:
        return common + l1 + [GLASS_DIR / "src/cgrps/l1.cuh"]
    if binary_name in DENSE_BINS or binary_name in FACTOR_BINS or binary_name in TIER_BINS:
        return common + l1 + l2 + l3 + cgrps
    if binary_name in SOLVER_BINS:
        return (common + l1 + l2 + l3 + cgrps
                + sorted((GLASS_DIR / "src/base/banded").glob("*.cuh"))
                + sorted((GLASS_DIR / "src/base/pcg").glob("*.cuh"))
                + sorted((GLASS_DIR / "src/internal").glob("*.cuh")))
    if binary_name in {"test_robotics", "test_api_robotics"}:
        return common + sorted((GLASS_DIR / "src/base").rglob("*.cuh")) + cgrps
    if binary_name in NVIDIA_BINS:
        return common + l1 + l2 + l3 + cgrps + sorted((GLASS_DIR / "src/nvidia").glob("*.cuh"))
    # New binaries start conservative until assigned to a family above.
    return sorted((GLASS_DIR / "src").rglob("*.cuh"))


def _hash_sources(cu_path: pathlib.Path) -> str:
    h = hashlib.sha256()
    # Cache scopes mirror receipt shards. Compile-smoke still parses the complete
    # umbrella on every PR; this narrower key controls numerical-test rebuilds.
    paths = [cu_path, pathlib.Path(__file__), CUDA_DIR / "helpers.cuh",
             GLASS_DIR / "glass.cuh", GLASS_DIR / "glass-cgrps.cuh",
             GLASS_DIR / "glass-defaults.cuh", GLASS_DIR / "glass-dispatch.cuh",
             GLASS_DIR / "glass-nvidia.cuh"]
    paths += _family_headers(cu_path.stem)
    paths = list(dict.fromkeys(paths))
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

    # A source-only key can silently reuse sm_XX SASS on another GPU or retain
    # a binary across compiler/MathDx flag changes. Include the complete build
    # identity in the persistent cache fingerprint.
    identity = hashlib.sha256()
    identity.update(_hash_sources(cu_src).encode())
    identity.update(arch.encode())
    identity.update("\0".join(extra_flags or []).encode())
    try:
        identity.update(subprocess.check_output(["nvcc", "--version"]))
    except Exception:
        pass
    current_hash = identity.hexdigest()[:16]
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

class LazyBins(Mapping):
    """Compile only the binary a selected test actually requests."""

    names = {
        "l1", "l2", "l3", "qp", "banded", "bdsv", "pcg", "syrk", "trsv",
        "ldlt", "getrf", "iamax", "fused", "warp", "thread", "posv", "reduced",
        "tensor", "factor_check", "congruence", "solve", "reduced_blas", "base_f64",
        "defaults", "dispatch", "l1_round2", "block_access", "symmetrize", "symm_rot",
        "syev", "robotics", "api_vector", "api_dense", "api_factor", "api_robotics", "l3_nvidia", "nvidia_dispatch", "trailing_sync", "nvidia_f64",
    }
    optional = {"l3_nvidia", "nvidia_dispatch", "trailing_sync", "nvidia_f64"}

    def __init__(self):
        self.cache: dict[str, pathlib.Path] = {}
        self.failed: set[str] = set()

    @staticmethod
    def _mathdx():
        root = os.environ.get("MATHDX_ROOT")
        path = pathlib.Path(root) if root else None
        available = bool(path and (path / "include/cublasdx.hpp").exists())
        return path, available

    def _compile(self, key: str) -> pathlib.Path:
        mathdx, cublasdx = self._mathdx()
        flags: list[str] = []
        if key == "nvidia_dispatch" and not cublasdx:
            raise KeyError(key)
        if key in {"nvidia_dispatch", "trailing_sync"} and cublasdx:
            flags = ["--expt-relaxed-constexpr", "-DGLASS_BENCH_CUBLASDX",
                     "-I", str(mathdx / "include"),
                     "-I", str(mathdx / "external/cutlass/include")]
        if key == "nvidia_f64":
            solver = bool(cublasdx and (mathdx / "include/cusolverdx.hpp").exists()
                          and (mathdx / "include/cusolverdx_io.hpp").exists()
                          and (mathdx / "lib/libcusolverdx.a").exists())
            if not solver:
                raise KeyError(key)
            sms = CUDA_ARCH.replace("sm_", "") + "0"
            flags = ["--expt-relaxed-constexpr", "-DGLASS_BENCH_CUBLASDX",
                     "-DGLASS_BENCH_CUSOLVERDX",
                     "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", f"-DSMS={sms}",
                     "-I", str(mathdx / "include"),
                     "-I", str(mathdx / "external/cutlass/include"),
                     "-rdc=true", "-dlto", "-L", str(mathdx / "lib"),
                     "-lcusolverdx", "-lcublas", "-lcusolver", "-lcudart"]
        return compile_binary(f"test_{key}", BUILD_DIR, CUDA_ARCH, flags)

    def __getitem__(self, key: str) -> pathlib.Path:
        if key not in self.names or key in self.failed:
            raise KeyError(key)
        if key not in self.cache:
            try:
                self.cache[key] = self._compile(key)
            except Exception as exc:
                if key in self.optional:
                    self.failed.add(key)
                    print(f"\nSkipping test_{key} (compile unavailable): {exc}", file=sys.stderr)
                    raise KeyError(key) from exc
                raise
        return self.cache[key]

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __contains__(self, key):
        if key not in self.names:
            return False
        if key not in self.optional:
            return True
        try:
            self[key]
            return True
        except KeyError:
            return False


@pytest.fixture(scope="session")
def bins():
    return LazyBins()


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


@pytest.fixture(scope="session")
def bin_dispatch(bins):
    """Bare-face body-dispatch functional tests (bare glass::op vs glass::block::)."""
    return bins["dispatch"]


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
