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

LEGS = {
    "composition": ("bench_composition_ab.cu", ["4096", "20", "both"]),
    "mapping": ("bench_mapping_ab.cu", ["8192", "20", "both"]),
    "robotics": ("bench_robotics.cu", ["8192", "500", "both"]),
    "nvwarp": ("bench_nvwarp_l1.cu", []),
}

OVERNIGHT_ARGS = {
    "composition": ["4096", "100", "both"],
    "mapping": ["8192", "100", "both"],
    "robotics": ["8192", "1000", "both"],
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


def compute_pids():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True).stdout
        return {int(line) for line in out.splitlines() if line.strip().isdigit()}
    except Exception:
        return set()


def gpu_busy():
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True).stdout.strip().splitlines()[0]
        return int(value) > 5
    except Exception:
        return False


def wait_for_quiet_gpu(force=False, timeout=30):
    """Let telemetry from our preceding serial leg drain before failing."""
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
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "glass*.cuh", "src", "bench"], cwd=ROOT,
        capture_output=True, check=True).stdout
    digest = hashlib.sha256()
    for raw in sorted(p for p in listed.split(b"\0") if p):
        path = ROOT / os.fsdecode(raw)
        # Captures such as perf_*.txt are outputs, not executable inputs.
        if path.is_file() and path.suffix in {".cuh", ".cu", ".py"}:
            digest.update(raw + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(leg, arch, argv):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--",
         "glass*.cuh", "src", "bench"], cwd=ROOT, capture_output=True,
        text=True, check=True).stdout.strip())
    nvcc = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()[-1]
    receipt_path = ROOT / "test/gpu-proof.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path.exists() else "missing"
    session, fingerprint = receipt.get("session", {}), receipt.get("fingerprint", {})
    return [
        "# provenance_schema=2",
        f"# benchmark_leg={leg}",
        f"# timing_started_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"# git_commit={commit}",
        f"# source_dirty={str(dirty).lower()}",
        f"# source_sha256={source_digest()}",
        f"# arch={arch} argv={' '.join(argv)}",
        f"# toolchain={nvcc}",
        f"# correctness_receipt_ended_utc={session.get('ended_at', 'missing')}",
        f"# correctness_fingerprint_sha256={fingerprint.get('digest', 'missing')}",
        f"# correctness_receipt_sha256={receipt_sha}",
        "# correctness=separate signed GPU receipt; timing harness retains launch/finite sentinels",
    ]


def build(leg, arch):
    source, _ = LEGS[leg]
    BUILD_DIR.mkdir(exist_ok=True)
    binary = BUILD_DIR / f"{source.removesuffix('.cu')}_{arch}"
    deps = [BENCH_DIR / source, BENCH_DIR / "timing_common.cuh"]
    deps += list((ROOT / "src").rglob("*.cuh")) + list(ROOT.glob("glass*.cuh"))
    if binary.exists() and binary.stat().st_mtime > max(p.stat().st_mtime for p in deps):
        print(f"[build] {leg}: up to date")
        return binary
    cmd = ["nvcc", "-std=c++17", f"-arch={arch}", "-O3", "-Xptxas", "-O1",
           "--expt-relaxed-constexpr", "-I..", "-I../src", source,
           "-o", str(binary)]
    print("[build]", " ".join(cmd))
    if subprocess.run(cmd, cwd=BENCH_DIR).returncode:
        sys.exit(f"build failed: {leg}")
    return binary


def run_leg(leg, binary, arch, force, runtime_args):
    wait_for_quiet_gpu(force)
    argv = [str(binary), *runtime_args]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    capture = BENCH_DIR / f"perf_{leg}_{stamp}.txt"
    print(f"[run] {' '.join(argv)} -> {capture.name}")
    with capture.open("w") as stream:
        stream.write("\n".join(provenance(leg, arch, argv[1:])) + "\n")
        stream.flush()
        proc = subprocess.Popen(argv, cwd=BENCH_DIR, stdout=stream,
                                stderr=subprocess.STDOUT)
        foreign = set()
        while proc.poll() is None:
            foreign |= compute_pids() - {proc.pid}
            if foreign:
                proc.terminate()
                break
            time.sleep(1)
        returncode = proc.wait()
        if foreign:
            stream.write(f"# INVALID foreign_compute_pids={sorted(foreign)}\n")
    if foreign:
        sys.exit(f"{leg} invalidated by foreign compute PIDs {sorted(foreign)}")
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
