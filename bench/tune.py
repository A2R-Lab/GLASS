#!/usr/bin/env python3
"""GLASS unified autotuner — one command to remeasure this GPU and regenerate
every shipped defaults table + figure under a single noise margin.

    python bench/tune.py --sm auto [--margin 0.05] [--quick] [--legs ...]

It drives the three measurement harnesses and routes every verdict through the
one shared tie rule in ``bench/tune_pick.py`` (a dependency-carrying impl wins
only if it clears the margin), so no table bakes sub-noise jitter and a
pure-noise re-run reproduces the same tables. The legs:

  ladder   bench_mega_sweep.cu  → warp/block/nvidia ladder in glass-defaults.cuh
                                  (constexpr ideal_sm120 / local_ideal)
  shapes   bench/autotune.py    → per-(M,N,K) cuBLASDx-vs-SIMT table in
                                  src/nvidia/tuning_table.cuh  (needs MATHDX_ROOT)
  reduced  bench_reduced.cu     → validates serial-vs-reduced crossover against
                                  suggested_use_reduced<>; rewrites REDUCED_SWEEP_RESULTS.md
  figures  export_sweep_figures → docs _static/*.png ladders + sweep_winners.txt

All ops are *measured and recorded*; a dispatch picker is regenerated only for
ops with ≥2 genuinely-competing impls (the 6 ladder ops, the per-shape cuBLASDx
table, and the reduced corner). Single-impl families are reported, not picked.

EXECUTION DISCIPLINE: perf timing must be ISOLATED — run on a quiet GPU with no
concurrent CPU/GPU load. Build/iterate the tool offline with the ``--from-*``
hooks (feed an existing sweep .txt; no GPU touched). ``--dry-run`` regenerates
into memory and diffs against the in-tree tables WITHOUT writing — use it to
confirm a re-run only moves dispatch inside the tie band before committing.
"""
import argparse
import glob
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import time

import tune_pick as tp
from autotune import lib_digest  # shared library-content hash for cache keys

BENCH_DIR = pathlib.Path(__file__).parent.resolve()
GLASS_DIR = BENCH_DIR.parent
DEFAULTS  = GLASS_DIR / "glass-defaults.cuh"
STATIC    = GLASS_DIR / "docs" / "source" / "_static"
REDUCED_MD = BENCH_DIR / "REDUCED_SWEEP_RESULTS.md"
CACHE_ROOT = BENCH_DIR / ".tune_cache"

ALL_LEGS = ("ladder", "shapes", "reduced", "figures")


def cache_dir(sms):
    d = CACHE_ROOT / f"sm{sms}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_build(label, cu_name, flags, sms):
    """Compile `flags` (an nvcc argv WITHOUT -o; `cu_name` is its input) into the
    persistent cache, hash-keyed on the source + library digest + flags. Returns
    ``(bin_path, status)`` with status ∈ {cached, built, fail}. A cache hit skips
    nvcc entirely — so a prebuilt sweep is execute-only."""
    src = (BENCH_DIR / cu_name).read_bytes()
    key = hashlib.sha256(src + lib_digest().encode()
                         + " ".join(flags).encode()).hexdigest()[:12]
    binp = cache_dir(sms) / f"{label}_{key}"
    if binp.exists():
        return binp, "cached"
    res = run(flags + ["-o", str(binp)], cwd=BENCH_DIR)
    return (binp, "built") if res.returncode == 0 else (None, "fail")


# ─── environment ──────────────────────────────────────────────────────────────

def detect_sm():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip().split("\n")[0].strip()
        major, minor = out.split(".")
        return int(f"{major}{minor}0")
    except Exception:
        sys.exit("ERROR: could not detect SM via nvidia-smi; pass --sm <e.g. 1200>.")


def mathdx_root():
    root = os.environ.get("MATHDX_ROOT")
    if root and (pathlib.Path(root) / "include" / "cublasdx.hpp").exists():
        return pathlib.Path(root)
    return None


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


# ─── ladder leg: bench_mega_sweep → ideal_sm120 ───────────────────────────────

_SIG = "constexpr backend ideal_sm120(op o, uint32_t N, bool f64) {"
_SENTINEL = "// Coarse fallback for unmeasured SMs"
# NPROB schedule mirrors run_mega_sweep.sh; --quick trims to the throughput point.
_FULL_SCHED  = [("1", "3000"), ("64", "1500"), ("1024", "700"),
                ("8192", "500"), ("32768", "200")]
_QUICK_SCHED = [("8192", "300")]


def build_mega_sweep(sms, mdx):
    if mdx is None:
        sys.exit("ERROR: ladder leg needs MATHDX_ROOT (the nvidia contender). "
                 "Set it, or run with --legs reduced (no MathDx) / --from-ladder.")
    flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
             "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
             f"-I{mdx/'include'}", f"-I{mdx/'external'/'cutlass'/'include'}",
             "-DGLASS_BENCH_CUBLASDX", "-DGLASS_BENCH_CUSOLVERDX", f"-DSMS={sms}",
             "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", "-rdc=true", "-dlto",
             f"-L{mdx/'lib'}", "-lcusolverdx", "-lcublas", "-lcusolver", "-lcudart",
             "bench_mega_sweep.cu"]
    binp, status = cached_build("mega_sweep", "bench_mega_sweep.cu", flags, sms)
    if status == "fail":
        sys.exit("ERROR: bench_mega_sweep compile failed.")
    print(f"  bench_mega_sweep: {status} ({binp.name})")
    return binp


def run_mega_sweep(binp, quick):
    sched = _QUICK_SCHED if quick else _FULL_SCHED
    out = [f"# mega sweep  {time.strftime('%c')}  (bench/tune.py)"]
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.sm,temperature.gpu",
             "--format=csv,noheader"], text=True).strip()
        out.append(smi)
    except Exception:
        pass
    out.append("")
    for nprob, reps in sched:
        for dt in ("f32", "f64"):
            print(f"  -> mega_sweep NPROB={nprob} reps={reps} {dt}")
            out.append(f"################ NPROB={nprob}  reps={reps}  dtype={dt} ################")
            r = subprocess.run([str(binp), nprob, reps, dt], text=True,
                               capture_output=True, cwd=BENCH_DIR)
            out.append(r.stdout)
            out.append("")
    path = BENCH_DIR / f"mega_sweep_{time.strftime('%Y%m%d_%H%M')}.txt"
    path.write_text("\n".join(out))
    print(f"==> wrote {path.relative_to(GLASS_DIR)}")
    return path


def _ladder_expr(winners, dtype, op):
    """Collapse {N: backend} into a C++ 'N <= hi ? backend::x : ...' expr."""
    picks = winners.get((dtype, op))
    if not picks:
        return None
    runs = []  # (hi_N, backend)
    for N in sorted(picks):
        be = picks[N]
        if runs and runs[-1][1] == be:
            runs[-1] = (N, be)
        else:
            runs.append((N, be))
    if len(runs) == 1:
        return f"backend::{runs[0][1]}"
    parts = [f"N <= {hi}u ? backend::{be}" for hi, be in runs[:-1]]
    return " : ".join(parts) + f" : backend::{runs[-1][1]}"


def winners_from_sweep(text, margin):
    """(dtype, op) -> {N: backend} under the shared margin (nvidia is the dep)."""
    cells = tp.parse_mega_sweep(text, nprob=8192)
    winners = {}
    for (dt, op, N), times in cells.items():
        win = tp.pick(times, margin, {"nvidia"})
        if win:
            winners.setdefault((dt, op), {})[N] = win
    return winners


def emit_ideal_body(winners):
    lines = [_SIG, "    switch (o) {"]
    for op in tp.LADDER_OPS:
        f32 = _ladder_expr(winners, "f32", op)
        f64 = _ladder_expr(winners, "f64", op)
        if f32 is None and f64 is None:
            continue
        f32 = f32 or "backend::block"
        f64 = f64 or f32
        if f32 == f64:
            lines.append(f"        case op::{op}: return {f32};")
        else:
            lines.append(f"        case op::{op}:")
            lines.append(f"            if (!f64) return {f32};")
            lines.append(f"            else      return {f64};")
    lines += ["    }", "    return backend::block;", "}"]
    return "\n".join(lines)


def regen_ladder(sweep_text, margin, src_name):
    winners = winners_from_sweep(sweep_text, margin)
    if not winners:
        sys.exit("ERROR: no NPROB=8192 verdicts parsed from the sweep.")
    body = ("// ── sm_120 measured ladder (auto-regenerated by bench/tune.py) ──\n"
            f"// Source sweep: {src_name}   tie margin: ±{margin*100:.0f}% "
            "(nvidia must clear it)\n"
            "// Returns the *ideal* tier assuming nvidia is linked; "
            "nv_available() filters after.\n" + emit_ideal_body(winners))
    text = DEFAULTS.read_text()
    si = text.index(_SIG)
    head_lines = text[:si].splitlines(keepends=True)
    while head_lines and head_lines[-1].lstrip().startswith("//"):
        head_lines.pop()
    head = "".join(head_lines)
    tail = text[text.index(_SENTINEL, si):]
    return head + body + "\n\n" + tail, len(winners)


# ─── reduced leg: bench_reduced → validate suggested_use_reduced<> ────────────

def build_reduced(sms):
    flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms//10}", "-O3", "-I..", "-I../src",
             "bench_reduced.cu"]
    binp, status = cached_build("reduced", "bench_reduced.cu", flags, sms)
    if status == "fail":
        sys.exit("ERROR: bench_reduced compile failed.")
    print(f"  bench_reduced: {status} ({binp.name})")
    return binp


def predicate_use_reduced(n_out, K, blockDim):
    """Mirror of glass::suggested_use_reduced<n_out,K_contract,blockDim>()."""
    return (n_out <= blockDim // 32) and (K >= 32)


def analyze_reduced(text, margin):
    rows = tp.parse_reduced(text)
    wins, mism = [], []
    for r in rows:
        win = tp.pick({"serial": r["serial"], "reduced": r["reduced"]},
                      margin, {"reduced"})
        measured_reduced = (win == "reduced")
        # bench_reduced computes C(M,K)=A(M,N)·B(N,K): the contracted dim is N
        # (n_out = M*K), so the predicate's K_contract is the N column.
        predicted = predicate_use_reduced(r["n_out"], r["N"], r["blockDim"])
        r["winner"] = win
        if measured_reduced:
            wins.append(r)
        if measured_reduced != predicted:
            mism.append(r)
    return rows, wins, mism


_RED_BEGIN = "<!-- BEGIN tune.py: latest measured run -->"
_RED_END   = "<!-- END tune.py -->"


def gen_reduced_block(rows, wins, mism, margin, src):
    """The auto-refreshed measured-data block (between markers). The surrounding
    curated narrative in REDUCED_SWEEP_RESULTS.md is preserved."""
    L = [_RED_BEGIN,
         f"## Latest measured run (auto-refreshed by `bench/tune.py`)", "",
         f"_Source: `{src}` · tie margin ±{margin*100:.0f}% (reduced must clear "
         f"it) · {len(wins)} of {len(rows)} configs pick reduced._", ""]
    if wins:
        L += ["| M | N | K | n_out | blockDim | serial_us | reduced_us | ratio |",
              "|---|---|---|-------|----------|-----------|------------|-------|"]
        for r in wins:
            L.append(f"| {r['M']} | {r['N']} | {r['K']} | {r['n_out']} | "
                     f"{r['blockDim']} | {r['serial']:.4f} | {r['reduced']:.4f} | "
                     f"**{r['serial']/r['reduced']:.2f}** |")
        L.append("")
    L.append("Predicate `suggested_use_reduced<n_out,K_contract,blockDim>()` = "
             "`(n_out <= blockDim/32) && (K_contract >= 32)` "
             "(K_contract is the N column here).")
    if mism:
        L += ["", f"⚠️ **{len(mism)} config(s) disagree** with the predicate — "
              "review before trusting the formula on this GPU:", ""]
        for r in mism:
            pred = "reduced" if predicate_use_reduced(r['n_out'], r['N'], r['blockDim']) else "serial"
            L.append(f"- {r['M']}×{r['N']}×{r['K']} bd={r['blockDim']} "
                     f"(n_out={r['n_out']}): measured **{r['winner']}**, predicate **{pred}**")
    else:
        L += ["", "✅ Measurement matches the predicate for every swept config — "
              "the formula needs no change."]
    L += ["", _RED_END]
    return "\n".join(L)


def splice_reduced_md(existing, block):
    """Replace the marker region in `existing` with `block`; if absent, insert it
    just before '## Reproduce' (else append). Curated prose stays intact."""
    if _RED_BEGIN in existing and _RED_END in existing:
        pre, _, rest = existing.partition(_RED_BEGIN)
        _, _, post = rest.partition(_RED_END)
        return pre + block + post
    anchor = "## Reproduce"
    if anchor in existing:
        pre, _, post = existing.partition(anchor)
        return pre + block + "\n\n" + anchor + post
    return existing.rstrip() + "\n\n" + block + "\n"


# ─── diff helper ──────────────────────────────────────────────────────────────

def show_diff(path, new_text, label):
    old = path.read_text() if path.exists() else ""
    if old == new_text:
        print(f"  [{label}] no change.")
        return False
    import difflib
    diff = difflib.unified_diff(old.splitlines(), new_text.splitlines(),
                                fromfile=f"{label} (in-tree)",
                                tofile=f"{label} (regenerated)", lineterm="")
    print("\n".join(diff))
    return True


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sm", default="auto", help="SM (e.g. 1200) or auto (nvidia-smi)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="shared tie margin; a dependency impl wins only if it "
                        "beats the simplest impl by more than this (default 0.05)")
    p.add_argument("--legs", default=",".join(ALL_LEGS),
                   help=f"comma list of legs to run. default all: {','.join(ALL_LEGS)}")
    p.add_argument("--quick", action="store_true",
                   help="ladder: throughput point only (NPROB=8192), fewer reps")
    p.add_argument("--prebuild", action="store_true",
                   help="compile every binary the selected legs need into the "
                        "build cache and exit — no timing. Run this ANYTIME (even "
                        "while the GPU is busy; compilation is CPU-bound), so the "
                        "later sweep on a quiet GPU is execute-only and fast.")
    p.add_argument("--iters", type=int, default=200000, help="bench_reduced iters")
    p.add_argument("--dry-run", action="store_true",
                   help="regenerate + diff against in-tree tables, write nothing")
    p.add_argument("--from-ladder", metavar="TXT",
                   help="skip ladder build/run; regenerate from this mega_sweep .txt")
    p.add_argument("--from-reduced", metavar="TXT",
                   help="skip reduced build/run; analyze from this bench_reduced .txt")
    args = p.parse_args()

    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    bad = [l for l in legs if l not in ALL_LEGS]
    if bad:
        sys.exit(f"unknown leg(s) {bad}; choose from {ALL_LEGS}")
    offline = bool(args.from_ladder or args.from_reduced)
    sms = None if (args.sm == "auto" and offline) else (
        detect_sm() if args.sm == "auto" else int(args.sm))
    mdx = mathdx_root()

    print(f"=== GLASS unified autotune ===")
    print(f"  SM:      {('(offline)' if sms is None else 'sm_'+str(sms//10))}")
    print(f"  margin:  ±{args.margin*100:.0f}%   legs: {','.join(legs)}"
          f"{'   [PREBUILD]' if args.prebuild else '   [DRY RUN]' if args.dry_run else ''}")
    print(f"  MathDx:  {mdx or 'absent'}\n")
    changed = {}

    # ── prebuild: compile everything the legs need, run nothing ──
    if args.prebuild:
        if sms is None:
            sys.exit("ERROR: --prebuild needs a concrete SM (pass --sm 1200 or "
                     "ensure nvidia-smi works); it compiles for a target arch.")
        if "ladder" in legs:
            print("── prebuild: ladder ──────────────────────────────────────")
            build_mega_sweep(sms, mdx)
        if "shapes" in legs:
            print("── prebuild: shapes (cuBLASDx microbenches) ──────────────")
            if mdx is None:
                print("  [skip] shapes needs MATHDX_ROOT (cuBLASDx).")
            else:
                run([sys.executable, "autotune.py", "--sm", str(sms),
                     "--build-only", "--build-dir", str(cache_dir(sms))], cwd=BENCH_DIR)
        if "reduced" in legs:
            print("── prebuild: reduced ─────────────────────────────────────")
            build_reduced(sms)
        if "figures" in legs:
            print("── prebuild: figures ─────────────────────────────────────")
            print("  [n/a] figures is pure Python (matplotlib) — nothing to compile.")
        print(f"\n==> prebuild done. Cache: {cache_dir(sms)}")
        print("    Run the timed sweep on a quiet GPU; cached binaries skip nvcc.")
        return

    # ── ladder ──
    if "ladder" in legs:
        print("── ladder ───────────────────────────────────────────────")
        if args.from_ladder:
            sweep_path = pathlib.Path(args.from_ladder)
            sweep_text = sweep_path.read_text()
        else:
            binp = build_mega_sweep(sms, mdx)
            sweep_path = run_mega_sweep(binp, args.quick)
            sweep_text = sweep_path.read_text()
        new_defaults, n = regen_ladder(sweep_text, args.margin, sweep_path.name)
        print(f"  regenerated ideal_sm120 from {n} (dtype,op) groups")
        if args.dry_run:
            changed["ladder"] = show_diff(DEFAULTS, new_defaults, "glass-defaults.cuh")
        else:
            DEFAULTS.write_text(new_defaults)
            print(f"  wrote {DEFAULTS.relative_to(GLASS_DIR)}")

    # ── shapes (delegate to the mature per-shape engine; shares tune_pick) ──
    if "shapes" in legs:
        print("── shapes (cuBLASDx vs SIMT per (M,N,K)) ─────────────────")
        if offline and sms is None:
            print("  [skip] shapes needs a GPU; not available in offline mode.")
        elif mdx is None:
            print("  [skip] shapes needs MATHDX_ROOT (cuBLASDx).")
        else:
            cmd = [sys.executable, "autotune.py", "--sm", str(sms),
                   "--margin", str(args.margin), "--in-tree",
                   "--build-dir", str(cache_dir(sms))]
            if args.dry_run:
                cmd.append("--dry-run")
            run(cmd, cwd=BENCH_DIR)

    # ── reduced ──
    if "reduced" in legs:
        print("── reduced (serial vs gemm_reduced) ──────────────────────")
        if args.from_reduced:
            rtxt_path = pathlib.Path(args.from_reduced)
            rtxt = rtxt_path.read_text()
        elif sms is None:
            print("  [skip] reduced needs a GPU or --from-reduced.")
            rtxt = None
        else:
            binp = build_reduced(sms)
            print(f"  -> bench_reduced {args.iters}")
            rtxt = subprocess.run([str(binp), str(args.iters)], text=True,
                                  capture_output=True, cwd=BENCH_DIR).stdout
            rtxt_path = BENCH_DIR / f"reduced_sweep_{time.strftime('%Y%m%d_%H%M')}.txt"
            rtxt_path.write_text(rtxt)
        if rtxt:
            rows, wins, mism = analyze_reduced(rtxt, args.margin)
            print(f"  {len(rows)} configs, reduced wins {len(wins)}, "
                  f"predicate mismatches {len(mism)}")
            if mism:
                print("  ⚠️ predicate disagrees with measurement — see REDUCED_SWEEP_RESULTS.md")
            block = gen_reduced_block(rows, wins, mism, args.margin, rtxt_path.name)
            md = splice_reduced_md(REDUCED_MD.read_text(), block)
            if args.dry_run:
                changed["reduced"] = show_diff(REDUCED_MD, md, "REDUCED_SWEEP_RESULTS.md")
            else:
                REDUCED_MD.write_text(md)
                print(f"  wrote {REDUCED_MD.relative_to(GLASS_DIR)}")

    # ── figures ──
    if "figures" in legs:
        print("── figures ───────────────────────────────────────────────")
        sweep_for_fig = args.from_ladder
        if not sweep_for_fig:
            cands = sorted(glob.glob(str(BENCH_DIR / "mega_sweep_*.txt")))
            sweep_for_fig = cands[-1] if cands else None
        if not sweep_for_fig:
            print("  [skip] no mega_sweep_*.txt to plot.")
        elif args.dry_run:
            print(f"  [dry-run] would render figures from {pathlib.Path(sweep_for_fig).name}")
        else:
            run([sys.executable, "export_sweep_figures.py", sweep_for_fig], cwd=BENCH_DIR)

    if args.dry_run:
        moved = [k for k, v in changed.items() if v]
        print(f"\n[dry run] tables that would change: {moved or 'none'}")
    print("\n==> done.")


if __name__ == "__main__":
    main()
