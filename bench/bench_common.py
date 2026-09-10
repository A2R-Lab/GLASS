"""Shared helpers for the benchmark drivers (tune, autotune, paper/perf sweeps).

One copy of the quiet-GPU discipline and the capture-provenance stamp. The four
drivers each carried a private copy of these; they drifted (autotune's
provenance format diverged, and the mid-run foreign-PID check made --force
self-defeating). Import from here instead of re-pasting.

Pure-Python logic (baseline subtraction in watch_process) is unit-tested in
test/test_bench_common.py — no GPU needed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def compute_pids() -> set[int]:
    """Active GPU compute PIDs when nvidia-smi supports the query."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True).stdout
        return {int(line.strip()) for line in out.splitlines()
                if line.strip().isdigit()}
    except Exception:
        return set()


def gpu_busy() -> bool:
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True).stdout.strip().splitlines()[0]
        return int(value) > 5
    except Exception:
        return False


def require_quiet_gpu(force: bool = False, timeout: float = 30) -> None:
    """Block until the GPU is idle (lets a preceding serial leg's telemetry
    drain); exit if it is still busy after `timeout`. --force skips the
    preflight for deliberate diagnostic runs."""
    if force:
        return
    deadline = time.monotonic() + timeout
    while True:
        active = compute_pids()
        busy = gpu_busy()
        if not active and not busy:
            return
        if time.monotonic() >= deadline:
            sys.exit("ERROR: GPU is not isolated after "
                     f"{timeout:.0f}s; active compute PIDs={sorted(active)}, "
                     "utilization_busy="
                     f"{busy}. --force is for deliberate diagnostic runs only.")
        time.sleep(1)


def foreign_pids(current: set[int], own_pid: int,
                 baseline: frozenset[int] | set[int]) -> set[int]:
    """PIDs that invalidate a timed run: active now, not ours, not pre-existing.

    Subtracting `baseline` (the compute PIDs present BEFORE the harness
    launched) is what makes --force usable: a deliberately-tolerated busy GPU
    must not re-trip the mid-run check, while a NEW process appearing during
    the leg still invalidates it.
    """
    return current - {own_pid} - set(baseline)


def watch_process(proc: subprocess.Popen,
                  baseline: frozenset[int] | set[int] = frozenset(),
                  poll_s: float = 1.0) -> tuple[int, set[int]]:
    """Poll `proc` to completion; terminate it if a foreign compute PID
    appears. Returns (returncode, foreign_pid_set)."""
    foreign: set[int] = set()
    while proc.poll() is None:
        foreign |= foreign_pids(compute_pids(), proc.pid, baseline)
        if foreign:
            proc.terminate()
            break
        time.sleep(poll_s)
    return proc.wait(), foreign


def source_digest(root: pathlib.Path = ROOT) -> str:
    """Hash tracked and untracked, non-ignored library/benchmark sources.
    Captures such as *_sweep_*.txt are outputs, not executable inputs."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "glass*.cuh", "src", "bench"], cwd=root,
        capture_output=True, check=True).stdout
    digest = hashlib.sha256()
    for raw in sorted(p for p in listed.split(b"\0") if p):
        path = root / os.fsdecode(raw)
        if path.is_file() and path.suffix in {".cuh", ".cu", ".py"}:
            digest.update(raw + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(leg: str, config_line: str,
               comparison_line: str = ("correctness=separate signed GPU "
                                       "receipt; timing harness retains "
                                       "launch/finite sentinels"),
               root: pathlib.Path = ROOT) -> list[str]:
    """The standard schema-2 capture header, one format for every driver."""
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
    receipt = (json.loads(receipt_path.read_text())
               if receipt_path.exists() else {})
    receipt_sha = (hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                   if receipt_path.exists() else "missing")
    session = receipt.get("session", {})
    fingerprint = receipt.get("fingerprint", {})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return [
        "# provenance_schema=2",
        f"# benchmark_leg={leg}",
        f"# timing_started_utc={now}",
        f"# git_commit={commit}",
        f"# source_dirty={str(dirty).lower()}",
        f"# source_sha256={source_digest(root)}",
        f"# {config_line}",
        f"# toolchain={nvcc}",
        f"# correctness_receipt_ended_utc={session.get('ended_at', 'missing')}",
        f"# correctness_fingerprint_sha256={fingerprint.get('digest', 'missing')}",
        f"# correctness_receipt_sha256={receipt_sha}",
        f"# {comparison_line}",
    ]
