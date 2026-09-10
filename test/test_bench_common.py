"""Pure-Python tests for bench/bench_common.py — no GPU, no nvcc.

Covers the --force baseline-PID subtraction: the mid-run foreign-process
check must tolerate compute PIDs that were present (and deliberately
accepted) before the harness launched, while still tripping on NEW ones.
This logic drifted once across four driver copies; it is unit-pinned here
and runs in the GPU-less contract-coverage CI job.
"""

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import bench_common  # noqa: E402


def test_foreign_pids_subtracts_own_and_baseline():
    assert bench_common.foreign_pids({10, 20, 30}, own_pid=30,
                                     baseline={10}) == {20}


def test_foreign_pids_all_tolerated():
    assert bench_common.foreign_pids({10, 20}, own_pid=20,
                                     baseline={10}) == set()


def test_watch_process_tolerates_baseline_pids(monkeypatch):
    # The GPU "stays busy" with exactly the pre-launch PIDs → run completes.
    monkeypatch.setattr(bench_common, "compute_pids", lambda: {111, 222})
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    rc, foreign = bench_common.watch_process(proc, baseline={111, 222},
                                             poll_s=0.05)
    assert rc == 0
    assert foreign == set()


def test_watch_process_trips_on_new_pid(monkeypatch):
    # A NEW PID (not in baseline) appears mid-run → terminate + report it.
    monkeypatch.setattr(bench_common, "compute_pids", lambda: {111, 999})
    proc = subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(30)"])
    rc, foreign = bench_common.watch_process(proc, baseline={111},
                                             poll_s=0.05)
    assert foreign == {999}
    assert rc != 0  # terminated, not a clean exit
