#!/usr/bin/env python3
"""Paper-figure sweep driver: builds and runs the bench_paper_* harnesses.

These feed the GLASS paper's host-batched-baseline (F2), latency (F4) and
fusion (F3) figures — see docs/open-tasks/paper_glass_smallblock_2026-07-06.md
and bench/PAPER_SWEEPS.md. They are SEPARATE from bench/tune.py (which
regenerates the shipped dispatch tables); nothing here writes library headers.

Usage:
    python3 bench/paper_sweeps.py --build-only          # prep (any time, busy GPU OK)
    python3 bench/paper_sweeps.py                        # build + run all legs (QUIET GPU)
    python3 bench/paper_sweeps.py --legs hostblas        # one leg
    python3 bench/paper_sweeps.py --reps 100 --dtype f64

Timed legs run serially and REFUSE to start if the GPU looks busy
(>5% utilization) unless --force is given. Requires cuBLAS/cuSOLVER from the
CUDA toolkit; MathDx is NOT needed (the nvidia-interface curves for F1 come
from bench/tune.py's mega-sweep leg instead).
"""

import argparse
import datetime
import pathlib
import subprocess
import sys

BENCH_DIR = pathlib.Path(__file__).parent.resolve()
BUILD_DIR = BENCH_DIR / "build"

LEGS = {
    # name -> (source, extra nvcc flags, runtime args builder)
    "hostblas": ("bench_paper_hostblas.cu", ["-lcublas", "-lcusolver"]),
    "fusion":   ("bench_paper_fusion.cu",   ["-lcublas", "-lcusolver"]),
}


# Jetson boards don't reliably answer nvidia-smi's compute_cap query; map the
# device-tree model string instead. All Orin-class boards are sm_87.
TEGRA_MODELS = (("orin", "sm_87"), ("xavier", "sm_72"), ("tx2", "sm_62"),
                ("tx1", "sm_53"), ("nano", "sm_53"))


def detect_tegra():
    try:
        model = pathlib.Path("/proc/device-tree/model").read_text().lower()
    except OSError:
        return None
    for key, sm in TEGRA_MODELS:
        if key in model:
            return sm
    return None


def detect_arch():
    try:
        cap = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        major, minor = cap.split(".")
        return f"sm_{major}{minor}"
    except Exception:
        tegra = detect_tegra()
        if tegra:
            print(f"arch: nvidia-smi query failed; Tegra device tree -> {tegra}")
            return tegra
        # A silently wrong arch produces SASS that won't run (or worse, a stale
        # binary that does) — fail hard instead of guessing.
        sys.exit("could not detect GPU arch (nvidia-smi compute_cap query "
                 "failed, no Tegra device tree); pass --arch sm_XX explicitly")


def gpu_busy():
    try:
        util = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(util) > 5
    except Exception:
        return False


def build(leg, arch):
    src, libs = LEGS[leg]
    BUILD_DIR.mkdir(exist_ok=True)
    out = BUILD_DIR / f"{src.replace('.cu', '')}_{arch}"
    # skip if fresh: binary newer than the harness source AND every library
    # header (precompile via --build-only, then the quiet run starts instantly).
    # arch is baked into the name so a build dir synced from another GPU
    # (5090 -> Jetson) can never be mistaken for fresh.
    if out.exists():
        deps = [BENCH_DIR / src] + \
               list((BENCH_DIR.parent / "src").rglob("*.cuh")) + \
               list(BENCH_DIR.parent.glob("glass*.cuh"))
        if out.stat().st_mtime > max(d.stat().st_mtime for d in deps):
            print(f"[build] {leg}: up to date, skipping")
            return out
    cmd = ["nvcc", "-std=c++17", f"-arch={arch}", "-O3", "--expt-relaxed-constexpr",
           "-I..", "-I../src", src, "-o", str(out)] + libs
    print(f"[build] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=BENCH_DIR)
    if r.returncode != 0:
        sys.exit(f"build failed: {leg}")
    return out


def run(leg, binary, reps, dtype):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_txt = BENCH_DIR / f"paper_{leg}_{ts}.txt"
    args = [str(binary), str(reps), dtype]
    print(f"[run] {' '.join(args)} -> {out_txt.name}")
    with open(out_txt, "w") as f:
        r = subprocess.run(args, cwd=BENCH_DIR, stdout=f, stderr=subprocess.STDOUT)
    print(f"[run] {leg} exit={r.returncode}")
    if r.returncode != 0:
        tail = out_txt.read_text().splitlines()[-15:]
        print("\n".join(tail))
        sys.exit(f"leg failed: {leg} (see {out_txt.name})")
    return out_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="hostblas,fusion",
                    help="comma-separated subset of: " + ",".join(LEGS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--dtype", default="both", choices=["f32", "f64", "both"])
    ap.add_argument("--build-only", action="store_true",
                    help="compile everything, run nothing (safe on a busy GPU)")
    ap.add_argument("--force", action="store_true",
                    help="run timed legs even if the GPU looks busy")
    ap.add_argument("--arch", default=None,
                    help="override GPU arch (e.g. sm_87); default: auto-detect")
    args = ap.parse_args()

    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    for l in legs:
        if l not in LEGS:
            sys.exit(f"unknown leg {l!r}; choose from {list(LEGS)}")

    arch = args.arch or detect_arch()
    print(f"GPU arch: {arch}")
    binaries = {l: build(l, arch) for l in legs}
    if args.build_only:
        print("[build-only] done — run again without --build-only on a quiet GPU")
        return

    if gpu_busy() and not args.force:
        sys.exit("GPU is busy (>5% util) — perf timing must be isolated. "
                 "Re-run when quiet, or pass --force.")

    for l in legs:   # serial, one timed leg at a time
        run(l, binaries[l], args.reps, args.dtype)
    print("all legs done")


if __name__ == "__main__":
    main()
