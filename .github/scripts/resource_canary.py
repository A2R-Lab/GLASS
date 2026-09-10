#!/usr/bin/env python3
"""Static resource canary: pin per-kernel registers/stack/spill/smem.

Compiles resource_canary.cu (GPU-less — only nvcc is needed) with
`-Xptxas -v`, parses the per-kernel resource report, and diffs it against the
committed baseline (resource_canary_baseline.json). Any drift fails loudly:
a register-count or shared-memory change is a real code change that must be
acknowledged by regenerating the baseline in the same commit; spill bytes or
stack bytes above zero fail regardless of the baseline (the library's whole
premise is register/smem-resident math).

Usage:
  resource_canary.py --arch sm_120            # check against baseline
  resource_canary.py --arch sm_120 --update   # regenerate baseline for arch

The baseline records the nvcc release it was generated with; a differing
nvcc release fails with instructions (resource counts are toolkit-dependent,
so a toolkit bump legitimately regenerates the baseline). On mismatch the
measured JSON is printed in full so the baseline can be updated from CI logs.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TU = HERE / "resource_canary.cu"
BASELINE = HERE / "resource_canary_baseline.json"


def nvcc_release():
    out = subprocess.run(["nvcc", "--version"], check=True,
                         capture_output=True, text=True).stdout
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1)


def compile_and_parse(arch):
    cmd = ["nvcc", "-std=c++17", f"-arch={arch}", f"-I{ROOT}",
           "-Xptxas", "-v", "-c", str(TU), "-o", "/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(f"FATAL: canary TU failed to compile for {arch}")
    kernels = {}
    cur = None
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            cur = m.group(1)
            kernels[cur] = {"stack": 0, "spill_stores": 0, "spill_loads": 0,
                            "regs": 0, "smem": 0}
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                      r"(\d+) bytes spill loads", line)
        if m:
            kernels[cur]["stack"] = int(m.group(1))
            kernels[cur]["spill_stores"] = int(m.group(2))
            kernels[cur]["spill_loads"] = int(m.group(3))
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m:
            kernels[cur]["regs"] = int(m.group(1))
            sm = re.search(r"(\d+) bytes smem", line)
            kernels[cur]["smem"] = int(sm.group(1)) if sm else 0
    if not kernels:
        sys.exit("FATAL: parsed no kernels from ptxas output")
    return kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="e.g. sm_90, sm_120")
    ap.add_argument("--update", action="store_true",
                    help="regenerate the baseline for this arch")
    args = ap.parse_args()

    release = nvcc_release()
    measured = compile_and_parse(args.arch)

    # Hard invariant, baseline-independent: no spills, no stack, anywhere.
    hard_fail = [k for k, v in measured.items()
                 if v["spill_stores"] or v["spill_loads"] or v["stack"]]
    if hard_fail:
        print(json.dumps(measured, indent=2, sort_keys=True))
        sys.exit(f"FAIL: spill/stack bytes nonzero in: {', '.join(hard_fail)}")

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}

    if args.update:
        baseline[args.arch] = {"nvcc": release, "kernels": measured}
        BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        print(f"baseline updated: {args.arch} (nvcc {release}, "
              f"{len(measured)} kernels, zero spill/stack)")
        return

    if args.arch not in baseline:
        print(json.dumps({args.arch: {"nvcc": release, "kernels": measured}},
                         indent=2, sort_keys=True))
        sys.exit(f"FAIL: no baseline for {args.arch} — run with --update and "
                 f"commit resource_canary_baseline.json")
    entry = baseline[args.arch]
    if entry["nvcc"] != release:
        sys.exit(f"FAIL: baseline for {args.arch} was generated with nvcc "
                 f"{entry['nvcc']} but this is {release} — regenerate with "
                 f"--update (resource counts are toolkit-dependent)")

    diffs = []
    for k in sorted(set(entry["kernels"]) | set(measured)):
        want, got = entry["kernels"].get(k), measured.get(k)
        if want != got:
            diffs.append(f"  {k}: baseline={want} measured={got}")
    if diffs:
        print(json.dumps(measured, indent=2, sort_keys=True))
        print("\n".join(["FAIL: resource drift vs baseline "
                         f"({args.arch}, nvcc {release}):"] + diffs))
        sys.exit("If the change is intended, regenerate: "
                 f"resource_canary.py --arch {args.arch} --update "
                 "and commit the baseline in the same PR.")
    print(f"OK: {args.arch} matches baseline exactly "
          f"({len(measured)} kernels, zero spill/stack, nvcc {release})")


if __name__ == "__main__":
    main()
