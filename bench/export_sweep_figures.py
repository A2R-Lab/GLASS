#!/usr/bin/env python3
"""Render the warp/block/nvidia sweep ladder into static docs assets.

Reads a ``bench/mega_sweep_*.txt`` run (the same data behind
``glass-defaults.cuh``'s ``suggested_backend<>()``) and writes, into
``docs/source/_static/``:

* ``mega_sweep_ladder_<dt>_n<nprob>.png`` — one figure per (dtype, NPROB regime)
  for NPROB in {64, 1024, 8192}, ns/problem vs N per backend, one subplot per op;
* ``mega_sweep_ladder_<dt>.png`` — the headline alias (= the NPROB=8192 throughput
  regime), embedded on the landing page;
* ``sweep_winners.txt`` — the data-driven winning backend per (op, N), per regime.

The docs embed these committed assets statically (``docs/source/user_guide/
tutorials/sweep_results.rst``), so the site needs no GPU and no sweep ``.txt`` at
build time. Regenerate after a fresh sweep (``bench/tune.py`` runs this as its
figures leg, or run it standalone)::

    python bench/export_sweep_figures.py bench/mega_sweep_*.txt

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


# NPROB regimes to render, low → high: 64 ≈ low-batch latency, 1024 mid,
# 8192 the throughput regime that feeds the dispatch tables.
REGIMES = (64, 1024, 8192)
HEADLINE_NPROB = 8192  # also written to the bare mega_sweep_ladder_<dt>.png


def parse(text, regimes=REGIMES):
    """(nprob, dtype, op, N) -> {block, warp, nvidia} ns/problem, for each NPROB
    in `regimes` present in the sweep."""
    data, dt, nprob = {}, None, None
    keep = set(regimes)
    for line in text.splitlines():
        if line.startswith("####"):
            m = _HDR.search(line)
            if m:
                nprob, dt = int(m.group(1)), m.group(2)
            continue
        if nprob not in keep:
            continue
        m = _ROW.match(line.strip())
        if m:
            op, N = m.group(1), int(m.group(2))
            d = {"block": float(m.group(3)), "warp": float(m.group(4))}
            if m.group(5):
                d["nvidia"] = float(m.group(5))
            data[(nprob, dt, op, N)] = d
    return data


def regimes_present(data):
    return sorted({np for (np, _d, _o, _N) in data})


def _series(data, nprob, dt, op, key):
    pts = sorted(
        (N, v[key])
        for (np, d, o, N), v in data.items()
        if np == nprob and d == dt and o == op and key in v
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_ladder(data, nprob, dt, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    drew = False
    for ax, op in zip(axes, OPS):
        for key, c in [("warp", "tab:green"), ("block", "tab:blue"), ("nvidia", "tab:red")]:
            xs, ys = _series(data, nprob, dt, op, key)
            if xs:
                ax.plot(xs, ys, "o-", color=c, label=key, ms=4)
                drew = True
        ax.set_title(f"{op} ({dt}, NPROB={nprob})")
        ax.set_xlabel("N")
        ax.set_ylabel("ns/problem")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{dt} warp/block/nvidia ladder — NPROB={nprob} "
                 f"({'throughput' if nprob >= 8192 else 'low-batch' if nprob <= 64 else 'mid-batch'})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return drew


def winners_text(data):
    lines = []
    for nprob in regimes_present(data):
        for dt in ("f32", "f64"):
            Ns = sorted({N for (np, d, o, N) in data if np == nprob and d == dt})
            if not Ns:
                continue
            lines.append(f"NPROB={nprob}  {dt} winner by op x N:")
            lines.append("op     " + "".join(f"{N:>7}" for N in Ns))
            for op in OPS:
                cells = []
                for N in Ns:
                    d = data.get((nprob, dt, op, N))
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
    present = regimes_present(data)
    print("parsed", len(data), "cells across NPROB", present)
    os.makedirs(args.out, exist_ok=True)

    # One figure per (dtype, NPROB regime): mega_sweep_ladder_<dt>_n<nprob>.png.
    for dt in ("f32", "f64"):
        for nprob in present:
            png = os.path.join(args.out, f"mega_sweep_ladder_{dt}_n{nprob}.png")
            if plot_ladder(data, nprob, dt, png):
                print("wrote", os.path.relpath(png, repo))
        # Headline alias (bare name) = the throughput regime, for index.rst.
        headline = HEADLINE_NPROB if HEADLINE_NPROB in present else (present[-1] if present else None)
        if headline is not None:
            bare = os.path.join(args.out, f"mega_sweep_ladder_{dt}.png")
            if plot_ladder(data, headline, dt, bare):
                print("wrote", os.path.relpath(bare, repo), f"(headline NPROB={headline})")

    winners = os.path.join(args.out, "sweep_winners.txt")
    with open(winners, "w") as fh:
        fh.write(winners_text(data))
    print("wrote", os.path.relpath(winners, repo))


if __name__ == "__main__":
    main()
