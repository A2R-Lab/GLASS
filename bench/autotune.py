#!/usr/bin/env python3
"""GLASS autotune — generate a per-hardware tuning override for the
should_use_cublasdx*<> dispatch decisions.

What it does
------------
1. Invokes bench/run_bench.py to measure GLASS SIMT vs cuBLASDx across the
   shape grid for gemm, gemv, gemm_batched (if available).
2. For each (api, shape) where both SIMT and cuBLASDx measurements exist,
   compares and picks the faster path as the dispatch decision.
3. Writes `bench/tuning/tuning_<hostname>_sm<NN>.cuh` with a
   `constexpr std::array<entry, N> kLocalTable = {{ ... }};` that overrides
   the shipped global table for the local build.
4. Computes a diff against `src/nvidia/tuning_table.cuh`'s `kGlobalTable`
   (entries where local disagrees, plus new entries the global table lacks)
   and prints a summary. With `--emit-pr-diff` it also dumps a suggested
   patch to stdout so contributors can open a PR.

Usage:
    python3 bench/autotune.py
    python3 bench/autotune.py --iters 10000
    python3 bench/autotune.py --emit-pr-diff
    python3 bench/autotune.py --reuse-results bench/results/bench_<host>.json

See bench/TUNING.md for the full contributor workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import sys


BENCH_DIR   = pathlib.Path(__file__).parent.resolve()
GLASS_DIR   = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
TUNING_DIR  = BENCH_DIR / "tuning"
GLOBAL_TBL  = GLASS_DIR / "src" / "nvidia" / "tuning_table.cuh"


# ─── shape parsing ──────────────────────────────────────────────────────────

# Matches "m= 4 n= 4 k= 4" or "m=64 n=64 k=64" (or with extra spaces).
_RE_MNK   = re.compile(r"\bm=\s*(\d+)\s+n=\s*(\d+)\s+k=\s*(\d+)")
_RE_MN    = re.compile(r"\bm=\s*(\d+)\s+n=\s*(\d+)(?!\s*k)")  # no k
_RE_BATCH = re.compile(r"\bBATCH=\s*(\d+)|\bbatch=\s*(\d+)")


def classify(label: str):
    """Return (api, M, N, K, BATCH, backend) or None.

    api      ∈ {"gemm", "gemv", "gemm_batched_1d"}
    backend  ∈ {"simt", "cublasdx"}
    BATCH    = 1 for non-batched

    Maps the closest comparable variants:
      "glass::gemm<CT> (shared)"   → SIMT  (compile-time SIMT, shared-mem inputs)
      "cuBLASDx gemm (shared)"     → cuBLASDx
      "glass::gemv<CT>"            → SIMT gemv
      "cuBLASDx gemv (shared)"     → cuBLASDx gemv
    """
    s = label.strip()

    # gemv first (looser regex pattern after gemm filtering).
    if "gemv<CT>" in s or ("gemv" in s and "CT" in s and "cuBLASDx" not in s):
        m = _RE_MN.search(s)
        if not m: return None
        return ("gemv", int(m.group(1)), int(m.group(2)), 1, 1, "simt")
    if "cuBLASDx gemv (shared)" in s:
        m = _RE_MN.search(s)
        if not m: return None
        return ("gemv", int(m.group(1)), int(m.group(2)), 1, 1, "cublasdx")

    # gemm (compile-time SIMT with shared-memory inputs)
    if "glass::gemm<CT> (shared)" in s:
        m = _RE_MNK.search(s)
        if not m: return None
        return ("gemm", int(m.group(1)), int(m.group(2)), int(m.group(3)), 1, "simt")
    if "cuBLASDx gemm (shared)" in s:
        m = _RE_MNK.search(s)
        if not m: return None
        return ("gemm", int(m.group(1)), int(m.group(2)), int(m.group(3)), 1, "cublasdx")

    # gemm_batched_1d (the round-2 1D-launch variant — bench_gemm_batched_1d.cu).
    # We intentionally do NOT classify the older 2D-launch gemm_batched here:
    # its tuning curve differs and the wrapper does not auto-dispatch yet.
    if "gemm_batched_1d" in s:
        mnk = _RE_MNK.search(s)
        if not mnk: return None
        bm = _RE_BATCH.search(s)
        if not bm: return None
        batch = int(bm.group(1) or bm.group(2))
        M, N, K = int(mnk.group(1)), int(mnk.group(2)), int(mnk.group(3))
        if "glass::nvidia::gemm_batched_1d" in s:
            return ("gemm_batched_1d", M, N, K, batch, "cublasdx")
        if "naive" in s:
            return ("gemm_batched_1d", M, N, K, batch, "simt")

    return None


# ─── results aggregation ────────────────────────────────────────────────────

def parse_results(json_path: pathlib.Path):
    """Return dict: {(api, M, N, K, BATCH): {"simt": us, "cublasdx": us}}."""
    data = json.loads(json_path.read_text())
    sm = data.get("arch", "sm_75")
    m = re.match(r"sm_(\d+)", sm)
    sm_val = int(m.group(1)) * 10 if m else 750

    by_shape: dict[tuple, dict[str, float]] = {}
    for category, rows in data["results"].items():
        for row in rows:
            tag = classify(row["label"])
            if tag is None: continue
            api, M, N, K, BATCH, backend = tag
            key = (api, M, N, K, BATCH)
            by_shape.setdefault(key, {})[backend] = float(row["us_per_op"])
    return by_shape, sm_val


def decide_entries(by_shape, threshold_pct=5.0):
    """Decide use_cublasdx for each shape where both backends measured.

    Skips entries where the relative gap is below `threshold_pct` to avoid
    noise-driven flips.
    """
    out = []
    skipped_noise = 0
    for (api, M, N, K, BATCH), times in sorted(by_shape.items()):
        if "simt" not in times or "cublasdx" not in times:
            continue
        simt = times["simt"]; cdx = times["cublasdx"]
        if simt <= 0 or cdx <= 0:
            continue
        ratio = abs(simt - cdx) / min(simt, cdx) * 100.0
        if ratio < threshold_pct:
            skipped_noise += 1
            continue
        use_cublasdx = cdx < simt
        out.append({
            "api": api, "M": M, "N": N, "K": K, "BATCH": BATCH,
            "use_cublasdx": use_cublasdx,
            "simt_us": simt, "cublasdx_us": cdx, "ratio_pct": ratio,
        })
    return out, skipped_noise


# ─── kLocalTable header generation ──────────────────────────────────────────

API_ENUM = {
    "gemm":              "api::gemm",
    "gemv":              "api::gemv",
    "row_strided_gemm":  "api::row_strided_gemm",
    "row_strided_gemv":  "api::row_strided_gemv",
    "gemm_batched_1d":   "api::gemm_batched_1d",
}


def write_local_override(entries, sm_val, host, gpu_model, mathdx_root,
                         out_path: pathlib.Path):
    """Write a tuning_<host>_smNN.cuh file with the entries.

    The header is included INSIDE namespace glass::nvidia::tuning by
    tuning_table.cuh, so it just defines `kLocalTable` in that scope.
    """
    lines = [
        "// Autotune override for the local build.",
        "// Generated by bench/autotune.py — do not edit by hand.",
        "//",
        f"//   host:        {host}",
        f"//   sm:          {sm_val}",
        f"//   gpu_model:   {gpu_model}",
        f"//   mathdx_root: {mathdx_root}",
        "//",
        "// Compile your project with:",
        f"//   -DGLASS_TUNING_TABLE_LOCAL='\"{out_path.relative_to(GLASS_DIR)}\"'",
        "// to make these entries take precedence over the shipped global table.",
        "//",
        "// See bench/TUNING.md for the contributor workflow if you'd like to",
        "// upstream these measurements to the global table via PR.",
        "",
        f"constexpr std::array<entry, {len(entries)}> kLocalTable = {{{{",
    ]
    for e in entries:
        api_field = f"{API_ENUM[e['api']]},"
        lines.append(
            f"    {{{api_field:<28} "
            f"{e['M']:>3}, {e['N']:>3}, {e['K']:>3}, {e['BATCH']:>3}, "
            f"{sm_val:>4}, {str(e['use_cublasdx']).lower()}}}, "
            f"// simt={e['simt_us']:.3f}us  cublasdx={e['cublasdx_us']:.3f}us  "
            f"({e['ratio_pct']:.1f}% gap)"
        )
    lines.append("}};")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ─── diff against shipped global table ──────────────────────────────────────

def parse_global_table():
    """Very lightweight regex parse of kGlobalTable[] entries.

    Returns a dict {(api, M, N, K, BATCH, SM): use_cublasdx}. Empty if the
    table is empty (which is the shipped default).
    """
    text = GLOBAL_TBL.read_text()
    # Find `constexpr std::array<entry, N> kGlobalTable = {{ ... }};`
    m = re.search(r"kGlobalTable\s*=\s*\{\{(.*?)\}\};", text, re.DOTALL)
    if not m:
        return {}
    body = m.group(1).strip()
    if not body:
        return {}
    out: dict[tuple, bool] = {}
    for entry_match in re.finditer(
        r"\{\s*api::(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(true|false)\s*\}",
        body,
    ):
        api, M, N, K, B, SM, flag = entry_match.groups()
        out[(api, int(M), int(N), int(K), int(B), int(SM))] = (flag == "true")
    return out


def diff_against_global(local_entries, global_tbl, sm_val):
    """Compute (agree, disagree, new) lists."""
    agree, disagree, new = [], [], []
    for e in local_entries:
        key = (e["api"], e["M"], e["N"], e["K"], e["BATCH"], sm_val)
        if key in global_tbl:
            if global_tbl[key] == e["use_cublasdx"]:
                agree.append(e)
            else:
                disagree.append(e)
        else:
            new.append(e)
    return agree, disagree, new


# ─── main ───────────────────────────────────────────────────────────────────

def detect_gpu() -> tuple[str, str]:
    """Return (compute_cap "8.6", gpu_model "RTX 3060")."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap,name",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0]
        cc, name = [p.strip() for p in out.split(",", 1)]
        return cc, name
    except Exception:
        return "7.5", "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a local tuning override based on bench measurements.")
    parser.add_argument("--iters", type=int, default=10000,
                        help="Iterations per bench kernel (default: 10000).")
    parser.add_argument("--reuse-results", type=pathlib.Path, default=None,
                        help="Skip running the bench, parse this JSON instead.")
    parser.add_argument("--emit-pr-diff", action="store_true",
                        help="Also dump a global-table patch suggestion to stdout.")
    parser.add_argument("--threshold-pct", type=float, default=5.0,
                        help="Skip entries where the SIMT/cuBLASDx gap is "
                             "below this pct (default: 5%%).")
    args = parser.parse_args()

    # Find or run the bench.
    if args.reuse_results:
        results_path = args.reuse_results
        if not results_path.exists():
            sys.exit(f"--reuse-results path does not exist: {results_path}")
    else:
        print("Running bench/run_bench.py ...")
        rc = subprocess.run(
            [sys.executable, str(BENCH_DIR / "run_bench.py"),
             "--iters", str(args.iters)],
            cwd=GLASS_DIR,
        ).returncode
        if rc != 0:
            sys.exit("bench/run_bench.py failed")
        results_path = RESULTS_DIR / f"bench_{platform.node()}.json"
        if not results_path.exists():
            sys.exit(f"bench results not found: {results_path}")

    # Parse and decide.
    print(f"Parsing {results_path} ...")
    by_shape, sm_val = parse_results(results_path)
    print(f"  found {len(by_shape)} (api, shape) measurement groups")

    entries, skipped = decide_entries(by_shape, threshold_pct=args.threshold_pct)
    print(f"  {len(entries)} entries above {args.threshold_pct}% gap threshold "
          f"({skipped} skipped as noise)")

    # Write the override.
    cc, gpu_model = detect_gpu()
    host = platform.node()
    sm_tag = f"sm{sm_val}"
    out_path = TUNING_DIR / f"tuning_{host}_{sm_tag}.cuh"
    mathdx_root = os.environ.get("MATHDX_ROOT", "")
    write_local_override(entries, sm_val, host, gpu_model, mathdx_root, out_path)
    print(f"\nWrote local override: {out_path}")
    print(f"To use it, compile your project with:")
    rel = out_path.relative_to(GLASS_DIR)
    print(f"  -DGLASS_TUNING_TABLE_LOCAL='\"{rel}\"'")

    # Diff against global.
    global_tbl = parse_global_table()
    agree, disagree, new = diff_against_global(entries, global_tbl, sm_val)
    print("\nDiff vs shipped kGlobalTable:")
    print(f"  {len(agree)} entries agree with the global table")
    print(f"  {len(disagree)} entries DISAGREE (your hardware says otherwise)")
    print(f"  {len(new)} entries new (not in global table for this SM)")

    if args.emit_pr_diff:
        print("\n--- suggested global-table patch ---")
        candidates = sorted(disagree + new,
                            key=lambda e: (e["api"], e["M"], e["N"], e["K"], e["BATCH"]))
        if not candidates:
            print("(nothing to contribute — your measurements match the global table)")
        else:
            print(f"// Contributed measurements from {host} (sm_{sm_val // 10}.{sm_val % 10}, "
                  f"{gpu_model}). See bench/TUNING.md.")
            for e in candidates:
                tag = "DISAGREES" if (e["api"], e["M"], e["N"], e["K"], e["BATCH"], sm_val) in global_tbl else "new"
                api_field = f"{API_ENUM[e['api']]},"
                print(
                    f"{{{api_field:<28} "
                    f"{e['M']:>3}, {e['N']:>3}, {e['K']:>3}, {e['BATCH']:>3}, "
                    f"{sm_val:>4}, {str(e['use_cublasdx']).lower()}}}, "
                    f"// [{tag}] simt={e['simt_us']:.3f}us cublasdx={e['cublasdx_us']:.3f}us"
                )

    print("\nSee bench/TUNING.md for the contribution workflow.")


if __name__ == "__main__":
    main()
