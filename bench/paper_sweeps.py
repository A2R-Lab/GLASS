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
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

BENCH_DIR = pathlib.Path(__file__).parent.resolve()
BUILD_DIR = BENCH_DIR / "build"

LEGS = {
    # name -> (source, extra nvcc flags, runtime args builder)
    "hostblas": ("bench_paper_hostblas.cu", ["-lcublas", "-lcusolver"]),
    "fusion":   ("bench_paper_fusion.cu",   ["-lcublas", "-lcusolver"]),
    # opt-in (--legs eigen): Eigen-in-kernel thread-serial baseline; Eigen is a
    # BENCH-ONLY header dep (apt libeigen3-dev, or EIGEN_ROOT). -diag-suppress
    # 20012 quiets Eigen's known defaulted-function annotation warnings.
    "eigen":    ("bench_eigen_ladder.cu",
                 [f"-I{os.environ.get('EIGEN_ROOT', '/usr/include/eigen3')}",
                  "-DEIGEN_DONT_VECTORIZE", "-DEIGEN_NO_DEBUG",
                  "-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int",
                  "-diag-suppress", "20012"]),
    # opt-in (--legs hostblas_magma): the hostblas harness with MAGMA columns
    # (gemm strided + pointer-array, potrf, fused posv). Needs MAGMA built at
    # MAGMA_ROOT (default ~/opt/src/magma-git: headers + build/lib/libmagma.a)
    # and a static OpenBLAS at OPENBLAS_ROOT (default ~/opt/openblas) — both
    # BENCH-ONLY deps. Same source file as hostblas; the define adds columns.
    "hostblas_magma": ("bench_paper_hostblas.cu",
                 ["-DGLASS_BENCH_MAGMA",
                  f"-I{os.environ.get('MAGMA_ROOT', os.path.expanduser('~/opt/src/magma-git'))}/include",
                  f"-I{os.environ.get('MAGMA_ROOT', os.path.expanduser('~/opt/src/magma-git'))}/build/include",
                  f"-L{os.environ.get('MAGMA_ROOT', os.path.expanduser('~/opt/src/magma-git'))}/build/lib",
                  "-lmagma",
                  os.path.expanduser(os.environ.get('OPENBLAS_ROOT', '~/opt/openblas')) + "/lib/libopenblas.a",
                  "-lcublas", "-lcusolver", "-lcusparse", "-lgomp", "-lpthread"]),
    # opt-in (--legs kokkos): Kokkos Kernels team-scope baseline; needs Kokkos
    # core + kokkos-kernels (BATCHED component) installed — BENCH-ONLY deps
    # (KOKKOS_ROOT / KOKKOSKERNELS_ROOT override the ~/opt defaults).
    "kokkos":   ("bench_kokkos_ladder.cu",
                 ["--expt-extended-lambda",
                  f"-I{os.environ.get('KOKKOS_ROOT', os.path.expanduser('~/opt/kokkos'))}/include",
                  f"-I{os.environ.get('KOKKOSKERNELS_ROOT', os.path.expanduser('~/opt/kokkos-kernels'))}/include",
                  f"-L{os.environ.get('KOKKOS_ROOT', os.path.expanduser('~/opt/kokkos'))}/lib",
                  f"-L{os.environ.get('KOKKOSKERNELS_ROOT', os.path.expanduser('~/opt/kokkos-kernels'))}/lib",
                  "-lkokkoskernels", "-lkokkoscore", "-lkokkoscontainers",
                  "-lkokkossimd", "-lcuda", "-ldl"]),
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


# Quiet-GPU discipline and provenance stamps are shared across the four
# benchmark drivers — one copy in bench_common (they drifted as four copies).
from bench_common import (compute_pids, require_quiet_gpu as wait_for_quiet_gpu,
                          watch_process)
import bench_common


def provenance(leg, arch, reps, dtype):
    return bench_common.provenance(
        leg, f"arch={arch} reps={reps} dtype={dtype}",
        comparison_line=("comparison=best swept GLASS configurations versus "
                         "documented default vendor APIs"))


def build(leg, arch):
    src, libs = LEGS[leg]
    BUILD_DIR.mkdir(exist_ok=True)
    # Named by LEG (not source): hostblas and hostblas_magma share a source
    # file and must never collide on the binary path.
    out = BUILD_DIR / f"bench_{leg}_{arch}"
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


def run(leg, binary, arch, reps, dtype, baseline=frozenset()):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = BENCH_DIR / f"paper_{leg}_{ts}.txt"
    args = [str(binary), str(reps), dtype]
    print(f"[run] {' '.join(args)} -> {out_txt.name}")
    with open(out_txt, "w") as f:
        f.write("\n".join(provenance(leg, arch, reps, dtype)) + "\n")
        f.flush()
        proc = subprocess.Popen(args, cwd=BENCH_DIR, stdout=f, stderr=subprocess.STDOUT)
        returncode, foreign = watch_process(proc, baseline, poll_s=2)
        if foreign:
            f.write(f"# INVALID foreign_compute_pids={','.join(map(str, sorted(foreign)))}\n")
    print(f"[run] {leg} exit={returncode}")
    if foreign:
        sys.exit(f"leg invalidated: another compute process appeared ({sorted(foreign)})")
    if returncode != 0:
        tail = out_txt.read_text().splitlines()[-15:]
        print("\n".join(tail))
        sys.exit(f"leg failed: {leg} (see {out_txt.name})")
    spread_by_section = {}
    for line in out_txt.read_text().splitlines():
        match = re.search(r"RESULT section=(\w+).* spread=([0-9.]+)%", line)
        if match:
            spread_by_section.setdefault(match.group(1), []).append(float(match.group(2)))
    for section, spreads in sorted(spread_by_section.items()):
        over5 = sum(value > 5.0 for value in spreads)
        over10 = sum(value > 10.0 for value in spreads)
        print(f"[spread] {leg}/{section}: {over5}/{len(spreads)} >5%, "
              f"{over10}/{len(spreads)} >10%, max={max(spreads):.2f}%")
        if over5:
            print("[spread] WARNING: do not publish small deltas from this capture; "
                  "compare an independent quiet run")
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

    wait_for_quiet_gpu(args.force)
    # Under --force, pre-existing PIDs are tolerated by design — subtract them
    # so the mid-run watch only trips on NEW processes.
    baseline = compute_pids() if args.force else frozenset()

    for l in legs:   # serial, one timed leg at a time
        run(l, binaries[l], arch, args.reps, args.dtype, baseline)
    print("all legs done")


if __name__ == "__main__":
    main()
