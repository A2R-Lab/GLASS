#!/usr/bin/env python3
"""Which sections of the 2026-08-02 50W capture did the zombie actually spoil?

The 08-02 50W run started 07:56; a stray 15W-launched tune.py sweep kept
running on the same GPU until 11:16. 50W has strictly higher clocks than 30W,
so an UNcontended 50W section must be FASTER than the 30W reference for the
same cell. A section whose cells are at-or-slower than 30W was sharing the GPU.
"""
import re, statistics, pathlib

CAP = pathlib.Path(__file__).parent
W30 = CAP / "jetson_localhost.localdomain_20260801_1226/mega_sweep_20260801_1226.txt"
W50 = CAP / "50w_20260802_partial/mega_sweep_20260802_0756.txt"

SEC = re.compile(r"#+ NPROB=(\d+)\s+reps=\d+\s+dtype=(f\d+)")
ROW = re.compile(r"^(\w+)\s+N=(\d+)\s+\|")


def parse(path):
    out, key, order = {}, None, []
    for ln in path.read_text().splitlines():
        m = SEC.search(ln)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key not in order:
                order.append(key)
            continue
        m = ROW.match(ln)
        if not m or key is None or "->" not in ln:
            continue
        tail = ln.split("||")[1]
        cands = [float(v) for _, v in re.findall(r"(block|warp|thread) \S+=([\d.]+)", tail)]
        mnv = re.search(r"nv=([\d.]+)", tail)
        if mnv:
            cands.append(float(mnv.group(1)))
        if cands:
            out[(key[0], key[1], m.group(1), int(m.group(2)))] = min(cands)
    return out, order


a30, _ = parse(W30)
a50, order50 = parse(W50)
common = set(a30) & set(a50)
print(f"cells: 30W={len(a30)} 50W={len(a50)} common={len(common)}\n")
print("section              n    median 50W/30W   verdict")
print("-" * 58)
for sec in order50:
    r = sorted(a50[k] / a30[k] for k in common if (k[0], k[1]) == sec)
    if not r:
        continue
    med = statistics.median(r)
    # clean 50W should be clearly faster than 30W (ratio well under 1)
    verdict = "CLEAN (faster than 30W)" if med < 0.92 else \
              "SUSPECT (no 50W speedup)" if med < 1.05 else \
              "CONTENDED (slower than 30W)"
    print(f"NPROB={sec[0]:<6d} {sec[1]}  {len(r):4d}   {med:6.3f}          {verdict}")
