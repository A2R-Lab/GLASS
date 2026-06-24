#!/usr/bin/env python3
"""Render the warp/block/nvidia sweep ladder into static docs assets.

Reads a ``bench/mega_sweep_*.txt`` run (the same data behind
``glass-defaults.cuh``'s ``suggested_backend<>()``) and writes, into
``docs/source/_static/``:

* ``mega_sweep_ladder_f32.png`` / ``mega_sweep_ladder_f64.png`` — ns/problem vs N
  per backend, one subplot per op (the throughput regime, NPROB=8192);
* ``sweep_winners.txt`` — the data-driven winning backend per (op, N), f32 + f64.

The docs embed these committed assets statically (``docs/source/user_guide/
tutorials/sweep_results.rst``), so the site needs no GPU and no sweep ``.txt`` at
build time. Regenerate after a fresh sweep::

    cd bench && ./run_mega_sweep.sh && python export_sweep_figures.py

This is the script form of ``bench/explore_sweep.ipynb`` (kept as the interactive
explorer). Needs numpy + matplotlib.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

OPS = ["dot", "gemv", "gemm", "chol", "trsv", "posv"]
_HDR = re.compile(r"NPROB=(\d+).*dtype=(f32|f64)")
_ROW = re.compile(
    r"^(dot|gemv|gemm|chol|trsv|posv)\s+N=(\d+).*\|\|\s*"
    r"block\s+tb\d+=([\d.]+)\s+warp\s+w\d+=([\d.]+)(?:\s+nv=([\d.]+))?"
)


def parse(text):
    """(dtype, op, N) -> {block, warp, nvidia} best ns/problem at NPROB=8192."""
    data, dt, nprob = {}, None, None
    for line in text.splitlines():
        if line.startswith("####"):
            m = _HDR.search(line)
            if m:
                nprob, dt = int(m.group(1)), m.group(2)
            continue
        if nprob != 8192:  # the throughput regime
            continue
        m = _ROW.match(line.strip())
        if m:
            op, N = m.group(1), int(m.group(2))
            d = {"block": float(m.group(3)), "warp": float(m.group(4))}
            if m.group(5):
                d["nvidia"] = float(m.group(5))
            data[(dt, op, N)] = d
    return data


def _series(data, dt, op, key):
    pts = sorted(
        (N, data[(dt, op, N)][key])
        for (d, o, N) in data
        if d == dt and o == op and key in data[(dt, op, N)]
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_ladder(data, dt, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    drew = False
    for ax, op in zip(axes, OPS):
        for key, c in [("warp", "tab:green"), ("block", "tab:blue"), ("nvidia", "tab:red")]:
            xs, ys = _series(data, dt, op, key)
            if xs:
                ax.plot(xs, ys, "o-", color=c, label=key, ms=4)
                drew = True
        ax.set_title(f"{op} ({dt}, NPROB=8192)")
        ax.set_xlabel("N")
        ax.set_ylabel("ns/problem")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return drew


def winners_text(data):
    lines = []
    for dt in ("f32", "f64"):
        Ns = sorted({N for (d, o, N) in data if d == dt})
        if not Ns:
            continue
        lines.append(f"{dt} winner by op x N:")
        lines.append("op     " + "".join(f"{N:>7}" for N in Ns))
        for op in OPS:
            cells = []
            for N in Ns:
                d = data.get((dt, op, N))
                cells.append(f"{min(d, key=d.get):>7}" if d else f"{'-':>7}")
            lines.append(f"{op:6} " + "".join(cells))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    default_static = os.path.join(repo, "docs", "source", "_static")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", nargs="?", help="path to a mega_sweep_*.txt (default: latest in bench/)")
    ap.add_argument("--out", default=default_static, help="output dir (default: docs/source/_static)")
    args = ap.parse_args()

    sweep = args.sweep
    if not sweep:
        cands = sorted(glob.glob(os.path.join(here, "mega_sweep_*.txt")))
        if not cands:
            sys.exit("no mega_sweep_*.txt found in bench/ — run ./run_mega_sweep.sh first")
        sweep = cands[-1]
    print("sweep file:", os.path.relpath(sweep, repo))

    data = parse(open(sweep).read())
    print("parsed", len(data), "cells")
    os.makedirs(args.out, exist_ok=True)

    for dt in ("f32", "f64"):
        png = os.path.join(args.out, f"mega_sweep_ladder_{dt}.png")
        if plot_ladder(data, dt, png):
            print("wrote", os.path.relpath(png, repo))

    winners = os.path.join(args.out, "sweep_winners.txt")
    with open(winners, "w") as fh:
        fh.write(winners_text(data))
    print("wrote", os.path.relpath(winners, repo))


if __name__ == "__main__":
    main()
