#!/usr/bin/env python3
"""GLASS benchmark driver.

Compiles bench_reduce.cu, bench_gemv.cu, bench_gemm.cu and runs them,
printing a Markdown timing table.

Requirements:
  - CUB: included with every CUDA Toolkit (verified automatically)
  - cuBLASDx: set MATHDX_ROOT env var to the MathDx installation directory
              (see bench/INSTALL.md for download instructions)

Usage:
    python3 bench/run_bench.py [--iters N] [--no-cublasdx]
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time

BENCH_DIR   = pathlib.Path(__file__).parent.resolve()
GLASS_DIR   = BENCH_DIR.parent
BUILD_DIR   = BENCH_DIR / "build"
RESULTS_DIR = BENCH_DIR / "results"

BENCH_NAMES = ["bench_reduce", "bench_gemv", "bench_gemm"]


# ─── GPU detection ────────────────────────────────────────────────────────────

def detect_arch() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0].strip()
        major, minor = out.split(".")
        return f"sm_{major}{minor}", int(f"{major}{minor}0")
    except Exception:
        return "sm_75", 750


# ─── Dependency checks ────────────────────────────────────────────────────────

def check_cub() -> pathlib.Path:
    """Find CUB header. Raises SystemExit if not found."""
    cuda_root = pathlib.Path(
        subprocess.check_output(["nvcc", "--version"], text=True)
    )  # just to verify nvcc exists
    cub_header = pathlib.Path("/usr/local/cuda/include/cub/cub.cuh")
    if cub_header.exists():
        return cub_header.parent.parent.parent  # /usr/local/cuda
    sys.exit("ERROR: CUB not found at /usr/local/cuda/include/cub/cub.cuh\n"
             "CUB is part of the CUDA Toolkit — ensure CUDA 11.0+ is installed.")


def check_cublasdx() -> pathlib.Path | None:
    """Return MATHDX_ROOT path or None if not set/invalid."""
    root = os.environ.get("MATHDX_ROOT")
    if not root:
        return None
    root = pathlib.Path(root)
    # Look for the main cuBLASDx header in common locations
    candidates = [
        root / "include" / "cublasdx.hpp",
        root / "include" / "cuda" / "blas" / "device" / "cublasdx.hpp",
    ]
    for c in candidates:
        if c.exists():
            return root
    print(f"WARNING: MATHDX_ROOT={root} set but cublasdx.hpp not found there.")
    return None


# ─── Compilation ──────────────────────────────────────────────────────────────

def source_hash(cu_file: pathlib.Path, extra_files: list[pathlib.Path]) -> str:
    h = hashlib.sha256()
    for f in [cu_file] + extra_files:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def compile_binary(name: str, arch: str, sms: int, mathdx_root, cuda_root: pathlib.Path) -> pathlib.Path:
    cu_src    = BENCH_DIR / f"{name}.cu"
    out_bin   = BUILD_DIR / name
    hash_file = BUILD_DIR / f"{name}.hash"

    watch_files = [
        cu_src,
        GLASS_DIR / "glass.cuh",
        BENCH_DIR / "INSTALL.md",
    ]
    cur_hash = source_hash(cu_src, watch_files)

    if hash_file.exists() and out_bin.exists() and hash_file.read_text().strip() == cur_hash:
        return out_bin

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Compiling {name} ...", flush=True)

    cmd = [
        "nvcc",
        "-std=c++17",
        f"-arch={arch}",
        "-O3",
        f"-I{GLASS_DIR}",
        f"-I{GLASS_DIR / 'src'}",
        "-o", str(out_bin),
        str(cu_src),
    ]

    if mathdx_root is not None:
        cmd += [
            f"-I{mathdx_root / 'include'}",
            f"-I{mathdx_root / 'external' / 'cutlass' / 'include'}",
            "-DGLASS_BENCH_CUBLASDX",
            f"-DSMS={sms}",
            "-Xptxas", "-O1",   # workaround for CUDA 12.9 bug
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE ERROR for {name}:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    hash_file.write_text(cur_hash)
    return out_bin


# ─── Running and parsing ──────────────────────────────────────────────────────

def run_binary(binary: pathlib.Path, args: list[str]) -> list[dict]:
    result = subprocess.run(
        [str(binary)] + args,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"RUNTIME ERROR: {result.stderr}", file=sys.stderr)
        return []

    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected format: "label   m=X n=Y k=Z   T.TTT us/op"
        # rsplit into 3 tokens so parts[-2] is the number, parts[-1] is "us/op"
        try:
            parts = line.rsplit(None, 2)
            if parts[-1] != "us/op":
                continue
            us = float(parts[-2])
            label = parts[0].strip()
            rows.append({"label": label, "us_per_op": us, "raw": line})
        except (ValueError, IndexError):
            pass
    return rows


# ─── Markdown table ───────────────────────────────────────────────────────────

def print_table(title: str, rows: list[dict]):
    if not rows:
        return
    print(f"\n### {title}\n")
    print(f"{'Operation':<45} {'us/op':>8}")
    print(f"{'-'*45} {'-'*8}")
    for r in rows:
        print(f"{r['label']:<45} {r['us_per_op']:>8.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLASS benchmark driver")
    parser.add_argument("--iters", type=int, default=10000,
                        help="Iterations per kernel invocation (default: 10000)")
    parser.add_argument("--no-cublasdx", action="store_true",
                        help="Skip cuBLASDx even if MATHDX_ROOT is set")
    args = parser.parse_args()

    print("=== GLASS Benchmark Suite ===\n")

    # Check dependencies
    check_cub()
    mathdx_root = None if args.no_cublasdx else check_cublasdx()

    if mathdx_root is None and not args.no_cublasdx:
        print("ERROR: MATHDX_ROOT is not set or cuBLASDx header not found.")
        print("  Set MATHDX_ROOT to the MathDx installation directory.")
        print("  See bench/INSTALL.md for download instructions.")
        print("  To run without cuBLASDx: --no-cublasdx")
        sys.exit(1)

    arch, sms = detect_arch()
    print(f"GPU arch: {arch} (SM{sms})")
    print(f"cuBLASDx: {'enabled (' + str(mathdx_root) + ')' if mathdx_root else 'disabled'}")
    print(f"Iterations: {args.iters}\n")

    # Compile
    print("Compiling benchmarks...")
    binaries = {}
    for name in BENCH_NAMES:
        binaries[name] = compile_binary(name, arch, sms, mathdx_root, pathlib.Path("/usr/local/cuda"))
    print("Done.\n")

    sizes = [4, 6, 8, 12, 14, 24, 64]
    all_results = {}

    # bench_reduce (L1): powers-of-2 vector sizes up to block limit (256)
    print("Running L1 reductions (reduce, dot, l2norm vs CUB)...")
    reduce_rows = []
    for n in [8, 16, 32, 64, 128, 256]:
        print(f"  n={n} ...", end=" ", flush=True)
        reduce_rows += run_binary(binaries["bench_reduce"], [str(n), str(args.iters)])
        print("done")
    print_table("L1 Reductions", reduce_rows)
    all_results["reduce"] = reduce_rows

    # bench_gemv (L2): square sizes
    print("\nRunning L2 GEMV (glass vs cuBLASDx)...")
    gemv_rows = []
    for s in sizes:
        print(f"  {s}x{s} ...", end=" ", flush=True)
        gemv_rows += run_binary(binaries["bench_gemv"], [str(s), str(s), str(args.iters)])
        print("done")
    print_table("L2 GEMV", gemv_rows)
    all_results["gemv"] = gemv_rows

    # bench_gemm (L3): square sizes
    print("\nRunning L3 GEMM (glass plain + tiled vs cuBLASDx)...")
    gemm_rows = []
    for s in sizes:
        print(f"  {s}x{s}x{s} ...", end=" ", flush=True)
        gemm_rows += run_binary(binaries["bench_gemm"], [str(s), str(s), str(s), str(args.iters)])
        print("done")
    print_table("L3 GEMM", gemm_rows)
    all_results["gemm"] = gemm_rows

    # Save JSON
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"bench_{platform.node()}.json"
    with open(out_file, "w") as f:
        json.dump({
            "host": platform.node(),
            "arch": arch,
            "iters": args.iters,
            "cublasdx": str(mathdx_root) if mathdx_root else None,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
