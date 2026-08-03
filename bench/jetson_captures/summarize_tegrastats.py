#!/usr/bin/env python3
"""Condense a tegrastats run log into the few numbers the paper needs.

Raw tegrastats is ~1 sample/s (13 MB for a 9 h capture) — too big to ship, but
it carries the board-power story: what an "N-watt" nvpmodel mode actually draws
while the benchmark saturates the GPU, plus thermals and GPU busy fraction.
Emits a small text summary that CAN ship in-tree next to the timings.

    ./summarize_tegrastats.py <tegrastats_run.txt> [--label 30W]
    ./summarize_tegrastats.py <log> --after 11:16   # skip a contended window
"""
import argparse, pathlib, re, statistics, sys

TSTAMP = re.compile(r"^\d\d-\d\d-\d{4} (\d\d:\d\d:\d\d)")

RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
TJ = re.compile(r"tj@([\d.]+)C")
GPUT = re.compile(r"gpu@([\d.]+)C")


def rail(line, name):
    m = re.search(rf"{name} (\d+)mW", line)  # first field = instantaneous
    return int(m.group(1)) if m else None


def summarize(path, label, after=None, before=None):
    busy, tj, gput = [], [], []
    rails = {r: [] for r in RAILS}
    n = 0
    for ln in path.read_text(errors="replace").splitlines():
        if "GR3D_FREQ" not in ln:
            continue
        if after or before:
            m = TSTAMP.match(ln)
            if not m:
                continue
            t = m.group(1)
            if after and t < after:
                continue
            if before and t > before:
                continue
        n += 1
        m = GR3D.search(ln)
        if m:
            busy.append(int(m.group(1)))
        m = TJ.search(ln)
        if m:
            tj.append(float(m.group(1)))
        m = GPUT.search(ln)
        if m:
            gput.append(float(m.group(1)))
        for r in RAILS:
            v = rail(ln, r)
            if v is not None:
                rails[r].append(v)
    if not n:
        sys.exit(f"ERROR: no tegrastats samples parsed from {path}")

    # Board power while actually benchmarking: restrict to samples with the GPU
    # busy, so idle gaps between legs don't drag the mean down.
    busy_idx = [i for i, b in enumerate(busy) if b >= 50]
    def stat(vals, idx=None):
        v = [vals[i] for i in idx] if idx is not None else vals
        v = [x for x in v if x is not None]
        return (statistics.mean(v), max(v)) if v else (float("nan"),) * 2

    win = f" | window {after or 'start'}..{before or 'end'}" if (after or before) else ""
    out = [f"# tegrastats summary | {label}{win} | {path.name}",
           f"# samples: {n} (~{n/3600:.1f} h at 1 Hz); GPU-busy (>=50%) samples: {len(busy_idx)}",
           f"GPU busy      mean {statistics.mean(busy):5.1f} %   median {statistics.median(busy):5.1f} %"]
    tot_mean = 0.0
    for r in RAILS:
        if not rails[r]:
            continue
        mean_b, max_b = stat(rails[r], busy_idx)
        tot_mean += mean_b
        out.append(f"{r:<13} mean {mean_b/1000:6.2f} W   peak {max_b/1000:6.2f} W   (GPU-busy samples)")
    out.append(f"{'BOARD TOTAL':<13} mean {tot_mean/1000:6.2f} W   (sum of rails, GPU-busy samples)")
    if tj:
        out.append(f"tj            mean {statistics.mean(tj):5.1f} C   peak {max(tj):5.1f} C")
    if gput:
        out.append(f"gpu temp      mean {statistics.mean(gput):5.1f} C   peak {max(gput):5.1f} C")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--after", help="HH:MM:SS — ignore samples before this clock time")
    ap.add_argument("--before", help="HH:MM:SS — ignore samples after this clock time")
    a = ap.parse_args()
    print(summarize(a.log, a.label or a.log.parent.name, a.after, a.before))
