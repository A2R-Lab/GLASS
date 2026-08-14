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


def compute_pids():
    """Return active compute PIDs when nvidia-smi supports the query."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return {int(line.strip()) for line in out.splitlines() if line.strip().isdigit()}
    except Exception:
        return set()


def wait_for_quiet_gpu(force=False, timeout=30):
    """Wait out trailing telemetry from a just-finished serial benchmark."""
    if force:
        return
    deadline = time.monotonic() + timeout
    while True:
        active = compute_pids()
        busy = gpu_busy()
        if not active and not busy:
            return
        if time.monotonic() >= deadline:
            sys.exit("GPU is not isolated after 30s; active compute PIDs="
                     f"{sorted(active)}, utilization_busy={busy}")
        time.sleep(1)


def source_digest():
    """Hash every tracked or untracked, non-ignored benchmark/library source."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "glass*.cuh", "src", "bench"],
        cwd=BENCH_DIR.parent, capture_output=True, check=True).stdout
    digest = hashlib.sha256()
    for raw in sorted(p for p in listed.split(b"\0") if p):
        path = BENCH_DIR.parent / os.fsdecode(raw)
        if not path.is_file() or path.suffix not in {".cuh", ".cu", ".py"}:
            continue
        digest.update(raw + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(leg, arch, reps, dtype):
    root = BENCH_DIR.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--",
         "glass*.cuh", "src", "bench"], cwd=root, capture_output=True,
        text=True, check=True).stdout.strip())
    nvcc = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()[-1]
    receipt_path = root / "test/gpu-proof.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path.exists() else "missing"
    session = receipt.get("session", {})
    fingerprint = receipt.get("fingerprint", {})
    return [
        "# provenance_schema=2",
        f"# benchmark_leg={leg}",
        f"# timing_started_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"# git_commit={commit}",
        f"# source_dirty={str(dirty).lower()}",
        f"# source_sha256={source_digest()}",
        f"# arch={arch} reps={reps} dtype={dtype}",
        f"# toolchain={nvcc}",
        f"# correctness_receipt_ended_utc={session.get('ended_at', 'missing')}",
        f"# correctness_fingerprint_sha256={fingerprint.get('digest', 'missing')}",
        f"# correctness_receipt_sha256={receipt_sha}",
        "# comparison=best swept GLASS configurations versus documented default vendor APIs",
    ]


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
           "-Xptxas", "-O1",
           "-I..", "-I../src", src, "-o", str(out)] + libs
    print(f"[build] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=BENCH_DIR)
    if r.returncode != 0:
        sys.exit(f"build failed: {leg}")
    return out


def run(leg, binary, arch, reps, dtype):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = BENCH_DIR / f"paper_{leg}_{ts}.txt"
    args = [str(binary), str(reps), dtype]
    print(f"[run] {' '.join(args)} -> {out_txt.name}")
    with open(out_txt, "w") as f:
        f.write("\n".join(provenance(leg, arch, reps, dtype)) + "\n")
        f.flush()
        proc = subprocess.Popen(args, cwd=BENCH_DIR, stdout=f, stderr=subprocess.STDOUT)
        foreign = set()
        while proc.poll() is None:
            foreign |= compute_pids() - {proc.pid}
            if foreign:
                proc.terminate()
                break
            time.sleep(2)
        returncode = proc.wait()
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

    for l in legs:   # serial, one timed leg at a time
        run(l, binaries[l], arch, args.reps, args.dtype)
    print("all legs done")


if __name__ == "__main__":
    main()
