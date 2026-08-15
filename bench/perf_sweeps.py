#!/usr/bin/env python3
"""Build or run the audit-only performance sweeps.

These harnesses characterize optimization candidates; they do not regenerate
dispatch tables. Correctness remains a separate signed GPU receipt. Every
capture binds its source digest and the nearest receipt, refuses a busy GPU,
and is invalidated if another compute PID appears during the run.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

BENCH_DIR = pathlib.Path(__file__).parent.resolve()
ROOT = BENCH_DIR.parent
BUILD_DIR = BENCH_DIR / "build"

# robotics_paper: the paper's batch sweep — the same robotics micro-op grid
# re-run across the concurrency axis (one capture, one invocation per NPROB).
BATCH_GRID = ("256", "1024", "4096", "8192", "32768")

LEGS = {
    "composition": ("bench_composition_ab.cu", ["4096", "20", "both"]),
    "mapping": ("bench_mapping_ab.cu", ["8192", "20", "both"]),
    "robotics": ("bench_robotics.cu", ["8192", "500", "both"]),
    "robotics_paper": ("bench_robotics.cu",
                       [[n, "500", "both"] for n in BATCH_GRID]),
    "nvwarp": ("bench_nvwarp_l1.cu", []),
}

OVERNIGHT_ARGS = {
    "composition": ["4096", "100", "both"],
    "mapping": ["8192", "100", "both"],
    "robotics": ["8192", "1000", "both"],
    "robotics_paper": [[n, "1000", "both"] for n in BATCH_GRID],
    "nvwarp": [],
}


def detect_arch():
    try:
        cap = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip().splitlines()[0]
        major, minor = cap.split(".")
        return f"sm_{major}{minor}"
    except Exception:
        sys.exit("could not detect GPU architecture; pass --arch sm_XX")


# Quiet-GPU discipline and provenance stamps are shared across the four
# benchmark drivers — one copy in bench_common (they drifted as four copies).
from bench_common import (compute_pids, require_quiet_gpu as wait_for_quiet_gpu,
                          watch_process)
import bench_common


def provenance(leg, arch, argv):
    return bench_common.provenance(leg, f"arch={arch} argv={' '.join(argv)}")


def build(leg, arch):
    source, _ = LEGS[leg]
    BUILD_DIR.mkdir(exist_ok=True)
    binary = BUILD_DIR / f"{source.removesuffix('.cu')}_{arch}"
    deps = [BENCH_DIR / source, BENCH_DIR / "timing_common.cuh"]
    deps += list((ROOT / "src").rglob("*.cuh")) + list(ROOT.glob("glass*.cuh"))
    if binary.exists() and binary.stat().st_mtime > max(p.stat().st_mtime for p in deps):
        print(f"[build] {leg}: up to date")
        return binary
    cmd = ["nvcc", "-std=c++17", f"-arch={arch}", "-O3",
           "--expt-relaxed-constexpr", "-I..", "-I../src", source,
           "-o", str(binary)]
    print("[build]", " ".join(cmd))
    if subprocess.run(cmd, cwd=BENCH_DIR).returncode:
        sys.exit(f"build failed: {leg}")
    return binary


def run_leg(leg, binary, arch, force, runtime_args):
    wait_for_quiet_gpu(force)
    # Under --force, pre-existing PIDs are tolerated by design — subtract them
    # so the mid-run watch only trips on NEW processes.
    baseline = compute_pids() if force else frozenset()
    # A leg is either one invocation (flat argv list) or several appended into
    # the same capture (list of argv lists — the robotics_paper batch grid).
    runs = (runtime_args if runtime_args and isinstance(runtime_args[0], list)
            else [runtime_args])
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    capture = BENCH_DIR / f"perf_{leg}_{stamp}.txt"
    with capture.open("w") as stream:
        stream.write("\n".join(provenance(
            leg, arch, [" ".join(r) for r in runs])) + "\n")
        stream.flush()
        for runtime in runs:
            argv = [str(binary), *runtime]
            print(f"[run] {' '.join(argv)} -> {capture.name}")
            proc = subprocess.Popen(argv, cwd=BENCH_DIR, stdout=stream,
                                    stderr=subprocess.STDOUT)
            returncode, foreign = watch_process(proc, baseline)
            if foreign:
                stream.write(
                    f"# INVALID foreign_compute_pids={sorted(foreign)}\n")
            if foreign:
                sys.exit(f"{leg} invalidated by foreign compute PIDs "
                         f"{sorted(foreign)}")
            if returncode:
                sys.exit(f"{leg} exited {returncode}; see {capture.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legs", default=",".join(LEGS))
    parser.add_argument("--arch", help="target architecture, e.g. sm_120")
    parser.add_argument("--profile", choices=("screen", "overnight"),
                        default="screen",
                        help="overnight uses longer confirmation sampling")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="ignore only the initial utilization/PID check")
    args = parser.parse_args()
    legs = [item.strip() for item in args.legs.split(",") if item.strip()]
    unknown = [leg for leg in legs if leg not in LEGS]
    if unknown:
        sys.exit(f"unknown legs {unknown}; choose from {list(LEGS)}")
    arch = args.arch or detect_arch()
    binaries = {leg: build(leg, arch) for leg in legs}  # intentionally serial
    if args.build_only:
        print("[build-only] complete; no timing was executed")
        return
    runtime = OVERNIGHT_ARGS if args.profile == "overnight" else {
        leg: spec[1] for leg, spec in LEGS.items()
    }
    for leg in legs:
        run_leg(leg, binaries[leg], arch, args.force, runtime[leg])


if __name__ == "__main__":
    main()
