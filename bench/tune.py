#!/usr/bin/env python3
"""GLASS unified autotuner — one command to remeasure this GPU and regenerate
every shipped defaults table + figure under a single noise margin.

    python bench/tune.py --sm auto [--margin 0.05] [--quick] [--legs ...]

It drives the three measurement harnesses and routes every verdict through the
one shared tie rule in ``bench/tune_pick.py`` (a dependency-carrying impl wins
only if it clears the margin; between SIMT tiers, any tier within the ±2% SIMT
tie band of the fastest takes the cell if it is simpler — thread ≻ warp ≻
block), so no table bakes sub-noise jitter and a pure-noise re-run reproduces
the same tables. The legs:

  ladder   bench_mega_sweep.cu  → thread/warp/block/nvidia ladder in glass-defaults.cuh
                                  (thread — one problem/thread, N<=7 — is a
                                  dependency-free contender alongside warp/block:
                                  the shared pick takes the cheapest SIMT tier —
                                  with ties inside the ±2% SIMT band resolving to
                                  the simpler tier — so a fresh sweep emits
                                  `backend::thread` wherever the low-DOF packing
                                  actually wins)
                                  (per-arch constexpr ideal_sm* tables + the SM
                                  dispatch switch; a first-time arch — e.g. sm_87
                                  on a Jetson Orin — gets a new table + case,
                                  other arches' tables are left untouched)
  shapes   bench/autotune.py    → per-(M,N,K) cuBLASDx-vs-SIMT table in
                                  src/nvidia/tuning_table.cuh  (needs MATHDX_ROOT)
  reduced  bench_reduced.cu     → validates serial-vs-reduced crossover against
                                  suggested_use_reduced<>; rewrites REDUCED_SWEEP_RESULTS.md
  blas2    bench_blas2.cu       → warp/block sweep of the ops the ladder misses
                                  (syrk/syr2k/ldlt/ldltsv/inv/trmv/ger); reports picks
                                  into BLAS2_SWEEP_RESULTS.md (no header table yet)
  rect     bench_rect.cu        → warp/block sweep of rectangular gemv/gemm shapes;
                                  reports picks into RECT_SWEEP_RESULTS.md (no table yet)
  solvers  bench_solvers.cu     → solver characterization (bdsv-vs-pcg crossover,
                                  gesv/posv/inv-solve on SPD, syev/eig_clamp) into
                                  SOLVERS_SWEEP_RESULTS.md — measured, never picked
                                  (the bdsv/pcg choice is conditioning-dependent)
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
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

import tune_pick as tp
from autotune import lib_digest  # shared library-content hash for cache keys

BENCH_DIR = pathlib.Path(__file__).parent.resolve()
GLASS_DIR = BENCH_DIR.parent
DEFAULTS  = GLASS_DIR / "glass-defaults.cuh"
DISPATCH_HDR = GLASS_DIR / "glass-dispatch.cuh"
STATIC    = GLASS_DIR / "docs" / "source" / "_static"
REDUCED_MD = BENCH_DIR / "REDUCED_SWEEP_RESULTS.md"
BLAS2_MD   = BENCH_DIR / "BLAS2_SWEEP_RESULTS.md"
RECT_MD    = BENCH_DIR / "RECT_SWEEP_RESULTS.md"
SOLVERS_MD = BENCH_DIR / "SOLVERS_SWEEP_RESULTS.md"
CACHE_ROOT = BENCH_DIR / ".tune_cache"

ALL_LEGS = ("ladder", "body", "shapes", "reduced", "blas2", "rect", "solvers",
            "figures")


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


# ─── ladder leg: bench_mega_sweep → per-arch ideal_sm* + dispatch ────────────

# glass-defaults.cuh regions this leg owns: one table block per swept arch, plus
# the case-list of the SM dispatch switch inside defaults::ideal().
_LAD_BEGIN = "// === BEGIN tune.py ladder sm_{a} ==="
_LAD_END   = "// === END tune.py ladder sm_{a} ==="
_LAD_RE    = re.compile(r"// === BEGIN tune\.py ladder sm_(\d+) ===")
_LAD_END_RE = re.compile(r"// === END tune\.py ladder sm_\d+ ===")
_DIS_BEGIN = "// === BEGIN tune.py ladder dispatch ==="
_DIS_END   = "// === END tune.py ladder dispatch ==="
# NPROB schedule. 8192 is the throughput regime every regenerated table reads;
#64 and 1024 are collected for inspection. NPROB=1 (single-problem latency) and
# 32768 (slow tail, feeds no table) are intentionally dropped. --quick = 8192 only.
_FULL_SCHED  = [("64", "1000"), ("1024", "500"), ("8192", "250")]
_QUICK_SCHED = [("8192", "300")]


def _fatbin_build_mega(sms, mdx):
    """4-tier build for hosts where libcusolverdx.a is foreign (the MathDx
    tarball ships x86-64 objects only — e.g. Jetson/aarch64). cuSOLVERDx also
    ships an LTO-IR `libcusolverdx.fatbin`, which is host-arch-independent but
    only legal as a DEVICE-LINK input, so the build is staged:
    -dc (LTO) -> -dlto -dlink with the fatbin -> final host link.
    Verified on AGX Orin (sm_87, CUDA 13.2) 2026-07-31."""
    common = ["-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
              "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
              f"-I{mdx/'include'}", f"-I{mdx/'external'/'cutlass'/'include'}",
              "-DGLASS_BENCH_CUBLASDX", "-DGLASS_BENCH_CUSOLVERDX", f"-DSMS={sms}",
              "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT"]
    src = (BENCH_DIR / "bench_mega_sweep.cu").read_bytes()
    key = hashlib.sha256(src + lib_digest().encode()
                         + " ".join(common + ["fatbin"]).encode()).hexdigest()[:12]
    binp = cache_dir(sms) / f"mega_sweep_fatbin_{key}"
    if binp.exists():
        return binp, "cached"
    obj, dlk = str(binp) + ".o", str(binp) + "_dlink.o"
    steps = [
        ["nvcc"] + common + ["-rdc=true", "-dlto", "-dc",
                             "bench_mega_sweep.cu", "-o", obj],
        ["nvcc", f"-arch=sm_{sms // 10}", "-dlto", "-dlink", obj,
         str(mdx / "lib" / "libcusolverdx.fatbin"), "-o", dlk],
        ["nvcc", f"-arch=sm_{sms // 10}", obj, dlk,
         "-lcublas", "-lcusolver", "-lcudart", "-o", str(binp)],
    ]
    for cmd in steps:
        if run(cmd, cwd=BENCH_DIR).returncode != 0:
            return None, "fail"
    return binp, "built"


def build_mega_sweep(sms, mdx, allow_no_mathdx=False):
    if mdx is None and not allow_no_mathdx:
        sys.exit("ERROR: ladder leg needs MATHDX_ROOT (the nvidia contender). "
                 "Set it (works on Tegra too — see the fatbin note in "
                 "bench/JETSON.md), run with --legs reduced (no MathDx) / "
                 "--from-ladder, or pass --allow-no-mathdx for a 3-tier "
                 "SIMT-only ladder.")
    if mdx is None:
        # 3-tier ladder: bench_mega_sweep.cu compiles out its vendor legs when
        # the GLASS_BENCH_CUBLASDX/CUSOLVERDX macros are absent, so the sweep
        # and the regenerated table simply lack the nvidia contender.
        print("  bench_mega_sweep: MathDx absent -> 3-tier (thread/warp/block)")
        flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
                 "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
                 f"-DSMS={sms}", "bench_mega_sweep.cu"]
        binp, status = cached_build("mega_sweep_simt", "bench_mega_sweep.cu",
                                    flags, sms)
    elif platform.machine() != "x86_64":
        # The tarball's libcusolverdx.a is x86-64-only; use the fatbin path.
        print("  bench_mega_sweep: non-x86 host -> 4-tier via cusolverdx FATBIN")
        binp, status = _fatbin_build_mega(sms, mdx)
    else:
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


def run_mega_sweep(binp, quick, prefix="mega_sweep", sched=None):
    """Run a ladder-style harness (mega/blas2/rect/solvers share the CLI +
    section grammar) over the NPROB schedule x {f32,f64}; write
    <prefix>_<ts>.txt. ``sched`` overrides the NPROB/reps schedule (the solvers
    leg times per-launch with restores, so its reps budget is much smaller)."""
    if sched is None:
        sched = _QUICK_SCHED if quick else _FULL_SCHED
    # INCREMENTAL PERSIST: the file is created up front and appended after
    # EVERY section, so a killed run (power cut, reboot, orchestrator bail)
    # keeps all completed sections — hours of Orin data died to the old
    # write-at-the-end behavior (2026-07-31).
    path = BENCH_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M')}.txt"
    hdr = [f"# {prefix}  {time.strftime('%c')}  (bench/tune.py)"]
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.sm,temperature.gpu",
             "--format=csv,noheader"], text=True).strip()
        hdr.append(smi)
    except Exception:
        pass
    hdr.append("")
    path.write_text("\n".join(hdr) + "\n")
    for nprob, reps in sched:
        for dt in ("f32", "f64"):
            print(f"  -> {prefix} NPROB={nprob} reps={reps} {dt}")
            r = subprocess.run([str(binp), nprob, reps, dt], text=True,
                               capture_output=True, cwd=BENCH_DIR)
            with open(path, "a") as f:
                f.write(f"################ NPROB={nprob}  reps={reps}  dtype={dt} ################\n")
                f.write(r.stdout)
                f.write("\n")
            print(f"     section saved -> {path.name}")
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


def emit_ideal_body(winners, fname):
    lines = [f"constexpr backend {fname}(op o, uint32_t N, bool f64) {{",
             "    switch (o) {"]
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


def regen_ladder(sweep_text, margin, src_name, sms):
    """Regenerate this arch's table block in glass-defaults.cuh (replace if the
    arch was swept before, insert after the last table block if it's new) and
    rebuild the SM dispatch case-list from every table block present."""
    arch = sms // 10
    winners = winners_from_sweep(sweep_text, margin)
    if not winners:
        sys.exit("ERROR: no NPROB=8192 verdicts parsed from the sweep.")
    begin, end = _LAD_BEGIN.format(a=arch), _LAD_END.format(a=arch)
    region = "\n".join([
        begin,
        f"// Source sweep: {src_name}   tie margin: ±{margin*100:.0f}% "
        "(nvidia must clear it; SIMT ties ±2% prefer thread>warp>block)",
        "// Returns the *ideal* tier assuming nvidia is linked; "
        "nv_available() filters after.",
        emit_ideal_body(winners, f"ideal_sm{arch}"),
        end])
    text = DEFAULTS.read_text()
    if begin in text:
        pre, _, rest = text.partition(begin)
        _, _, post = rest.partition(end)
        text = pre + region + post
    else:
        ends = list(_LAD_END_RE.finditer(text))
        if not ends:
            sys.exit("ERROR: no '=== tune.py ladder sm_* ===' marker blocks in "
                     f"{DEFAULTS.name}; the file layout has drifted.")
        at = ends[-1].end()
        text = text[:at] + "\n\n" + region + text[at:]
    # dispatch switch: one case per table block, ascending SM.
    arches = sorted(int(a) for a in _LAD_RE.findall(text))
    if _DIS_BEGIN not in text or _DIS_END not in text:
        sys.exit(f"ERROR: dispatch markers missing from {DEFAULTS.name}.")
    cases = "".join(f"        case {a * 10}u: return ideal_sm{a}(o, N, f64);\n"
                    for a in arches)
    pre, _, rest = text.partition(_DIS_BEGIN)
    _, _, post = rest.partition(_DIS_END)
    text = pre + _DIS_BEGIN + "\n" + cases + "        " + _DIS_END + post
    return text, len(winners)


# ─── body leg: bench_body_dispatch → dispatch_body() for the bare face ───────
#
# The bare `glass::op` face keeps a BLOCK-SCOPE calling contract; this leg only
# picks the implementation BODY per (op, N, dtype): full-block SIMT, warp 0
# only, or thread 0 only (each followed by a block sync). Because the caller's
# launch shape is fixed, a body may only take a cell if it is robust across
# every thread count the caller might launch with:
#
#   RULE: a candidate body wins a cell iff (a) it is never slower than the
#   block body by more than the margin at ANY measured (NPROB, TB) point, and
#   (b) it beats the block body by more than the margin at >=1 TB in the
#   NPROB=8192 throughput section. Ties/instability stay block (the
#   launchable-everywhere Phase-1 identity). Unmeasured N interpolate like the
#   ladder tables (each verdict extends up to the next measured N).

_BODY_SM_BEGIN = "// === BEGIN tune.py body sm_{a} ==="
_BODY_SM_END   = "// === END tune.py body sm_{a} ==="
_BODY_SM_RE    = re.compile(r"// === BEGIN tune\.py body sm_(\d+) ===")
_BODY_DIS_BEGIN = "// === BEGIN tune.py body dispatch ==="
_BODY_DIS_END   = "// === END tune.py body dispatch ==="
BODY_OPS = ("dot", "gemv", "gemm", "chol", "trsv", "posv", "eig3", "softmax")
_BODY_HDR_RE = re.compile(r"NPROB=(\d+)\s+reps=\d+\s+dtype=(f32|f64)")
_BODY_SEG_RE = re.compile(r"(BLOCKBODY|WARPBODY|THREADBODY)((?:\s+tb\d+=[0-9.]+)+)")
_BODY_NAME = {"BLOCKBODY": "block", "WARPBODY": "warp_in_block",
              "THREADBODY": "thread_in_block"}


def build_body(sms):
    flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms//10}", "-O3", "-I..", "-I../src",
             f"-DSMS={sms}", "bench_body_dispatch.cu"]
    binp, status = cached_build("body_dispatch", "bench_body_dispatch.cu", flags, sms)
    if status == "fail":
        sys.exit("ERROR: bench_body_dispatch compile failed.")
    print(f"  bench_body_dispatch: {status} ({binp.name})")
    return binp


def parse_body_sweep(text):
    """-> {(dtype, op, N): {body_name: {(nprob, tb): ns}}} (FAIL rows dropped)."""
    cells, nprob, dt = {}, None, None
    for line in text.splitlines():
        hdr = _BODY_HDR_RE.search(line)
        if hdr and "|" not in line:
            nprob, dt = int(hdr.group(1)), hdr.group(2)
            continue
        if "||" not in line or nprob is None or "FAIL" in line:
            continue
        m = re.match(r"\s*(\w+?)_n(\d+)\s*\|", line)
        if not m or m.group(1) not in BODY_OPS:
            continue
        key = (dt, m.group(1), int(m.group(2)))
        for bname, tbs in _BODY_SEG_RE.findall(line.split("||")[0]):
            pts = cells.setdefault(key, {}).setdefault(_BODY_NAME[bname], {})
            for tb, ns in re.findall(r"tb(\d+)=([0-9.]+)", tbs):
                pts[(nprob, int(tb))] = float(ns)
    return cells


def body_picks(cells, margin, top_nprob=8192):
    """(dtype, op) -> {N: body_name} for EVERY measured cell (block included)."""
    picks = {}
    for (dtc, opn, N), bodies in sorted(cells.items()):
        blk = bodies.get("block")
        if not blk or not any(pt[0] == top_nprob for pt in blk):
            continue
        best = None  # (geomean ratio over top-NPROB points, body_name)
        for cand in ("warp_in_block", "thread_in_block"):
            times = bodies.get(cand)
            if not times:
                continue
            common = [pt for pt in blk if pt in times]
            if len(common) != len(blk):          # partial coverage: not trusted
                continue
            if any(times[pt] > blk[pt] * (1 + margin) for pt in common):
                continue                          # (a) regression somewhere
            top = [pt for pt in common if pt[0] == top_nprob]
            if not any(times[pt] < blk[pt] * (1 - margin) for pt in top):
                continue                          # (b) no real win where it counts
            gm = math.exp(sum(math.log(times[pt] / blk[pt]) for pt in top) / len(top))
            if best is None or gm < best[0]:
                best = (gm, cand)
        picks.setdefault((dtc, opn), {})[N] = best[1] if best else "block"
    return picks


def _body_expr(picks, dtype, opn):
    """Collapse {N: body} into a bounded C++ conditional. Unlike the ladder
    tables (which extrapolate their last run to all larger N), the bare face
    NEVER extrapolates a partial-scope body past the largest measured N —
    running an unmeasured huge problem on one warp/thread would be
    catastrophic, while block is merely default. Beyond max-measured N: block."""
    ps = picks.get((dtype, opn))
    if not ps:
        return None
    runs = []  # (hi_N, body)
    for N in sorted(ps):
        b = ps[N]
        if runs and runs[-1][1] == b:
            runs[-1] = (N, b)
        else:
            runs.append((N, b))
    if runs[-1][1] == "block":
        runs[-1] = (None, "block")       # trailing block run: unbounded default
    else:
        runs.append((None, "block"))     # bound the last non-block run
    if len(runs) == 1:
        return "body::block"
    parts = [f"N <= {hi}u ? body::{b}" for hi, b in runs[:-1]]
    return " : ".join(parts) + f" : body::{runs[-1][1]}"


def emit_body_fn(picks, fname):
    lines = [f"GLASS_DISPATCH_HD constexpr body {fname}(op o, uint32_t N, bool f64) {{",
             "    switch (o) {"]
    for opn in BODY_OPS:
        f32 = _body_expr(picks, "f32", opn)
        f64 = _body_expr(picks, "f64", opn)
        if f32 is None and f64 is None:
            continue
        f32 = f32 or "body::block"
        f64 = f64 or f32
        if f32 == f64:
            lines.append(f"        case op::{opn}: return {f32};")
        else:
            lines.append(f"        case op::{opn}:")
            lines.append(f"            if (!f64) return {f32};")
            lines.append(f"            else      return {f64};")
    lines += ["        default: break;", "    }", "    return body::block;", "}"]
    return "\n".join(lines)


def regen_body(sweep_text, margin, src_name, sms):
    """Insert/replace this arch's body_sm* block and rebuild dispatch_body()
    from every body table present. Mirrors regen_ladder's marker discipline;
    the dispatch block always exists (Phase-1 stub or a prior regen)."""
    arch = sms // 10
    cells = parse_body_sweep(sweep_text)
    picks = body_picks(cells, margin)
    if not picks:
        sys.exit("ERROR: no NPROB=8192 body verdicts parsed from the sweep.")
    moved = sum(1 for ps in picks.values() for b in ps.values() if b != "block")
    begin, end = _BODY_SM_BEGIN.format(a=arch), _BODY_SM_END.format(a=arch)
    region = "\n".join([
        begin,
        f"// Source sweep: {src_name}   margin: ±{margin*100:.0f}%",
        "// RULE: never slower than block by >margin at ANY measured (NPROB, TB);",
        "// faster by >margin at >=1 TB in the NPROB=8192 section. Else block.",
        "// Verdicts are BOUNDED: N beyond the largest measured point stays block.",
        emit_body_fn(picks, f"body_sm{arch}"),
        end])
    text = DISPATCH_HDR.read_text()
    if _BODY_DIS_BEGIN not in text or _BODY_DIS_END not in text:
        sys.exit(f"ERROR: body dispatch markers missing from {DISPATCH_HDR.name}.")
    if begin in text:
        pre, _, rest = text.partition(begin)
        _, _, post = rest.partition(end)
        text = pre + region + post
    else:
        pre, _, rest = text.partition(_BODY_DIS_BEGIN)
        text = pre + region + "\n\n" + _BODY_DIS_BEGIN + rest
    arches = sorted(int(a) for a in _BODY_SM_RE.findall(text))
    cases = "".join(f"        case {a * 10}u: return body_sm{a}(o, N, f64);\n"
                    for a in arches)
    dispatch = "\n".join([
        _BODY_DIS_BEGIN,
        "// Bodies for the bare block-scope face; unmeasured arches stay block.",
        "GLASS_DISPATCH_HD constexpr body dispatch_body(op o, uint32_t N, bool f64,",
        "                                               uint32_t sm = GLASS_DEFAULTS_SM) {",
        "    switch (sm) {",
        cases.rstrip("\n"),
        "        default: break;",
        "    }",
        "    return body::block;",
        "}",
        _BODY_DIS_END])
    pre, _, rest = text.partition(_BODY_DIS_BEGIN)
    _, _, post = rest.partition(_BODY_DIS_END)
    text = pre + dispatch + post
    return text, moved


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
    """Mirror of glass::suggested_use_reduced<n_out,K_contract,blockDim>().

    Retired to constant False 2026-07-08: the quiet-GPU resweep measured 0/48
    reduced wins under the ±5% margin, so the predicate's true-corner was
    emptied. If a sweep on new hardware finds wins, derive a new corner from
    that data and update BOTH this mirror and the .cuh predicate."""
    return False


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


# ─── blas2 + rect legs: warp/block picks, reported (no header table yet) ─────
# These legs measure ops/shapes with no shipped defaults table: blas2 covers the
# ladder's blind-spot ops (syrk/syr2k/ldlt/ldltsv/inv/trmv/ger; no nvidia
# counterparts, so 2-way), rect covers rectangular gemv/gemm (nvidia skipped —
# per-shape cuBLASDx decisions live in the `shapes` leg). Verdicts route through
# tune_pick just like the tables; the results land in a marker-delimited block
# of BLAS2_SWEEP_RESULTS.md / RECT_SWEEP_RESULTS.md until the defaults-table
# extension is designed.

def _build_simt_harness(label, cu_name, sms):
    """Compile a no-MathDx warp/block harness with the mega sweep's opt flags."""
    flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms//10}", "-O3",
             "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
             cu_name]
    binp, status = cached_build(label, cu_name, flags, sms)
    if status == "fail":
        sys.exit(f"ERROR: {cu_name} compile failed.")
    print(f"  {cu_name.removesuffix('.cu')}: {status} ({binp.name})")
    return binp


def build_blas2(sms):
    return _build_simt_harness("blas2", "bench_blas2.cu", sms)


def build_rect(sms):
    return _build_simt_harness("rect", "bench_rect.cu", sms)


def build_solvers(sms):
    return _build_simt_harness("solvers", "bench_solvers.cu", sms)


def _shape_str(shape):
    return f"N={shape}" if isinstance(shape, int) else "x".join(str(d) for d in shape)


# Row order for the report tables: harness op order, then numeric shape, dtype.
_OP_ORDER = {op: i for i, op in enumerate(tp.BLAS2_OPS + ("gemv", "gemm"))}


def _cell_key(key):
    dt, op, shape = key
    return (_OP_ORDER.get(op, len(_OP_ORDER)),
            (shape,) if isinstance(shape, int) else tuple(shape), dt)


def gen_pick_block(cells, margin, src, block_only_note=""):
    """Marker-delimited measured-run block for the blas2/rect legs.

    ``cells``: ``(dtype, op, shape) -> {block[, warp]}`` ns/problem. warp and
    block are both dependency-free, so :func:`tune_pick.pick` resolves each cell
    to the cheapest impl; the verdict note records the gap vs the ±margin band.
    """
    warp_wins = sum(1 for t in cells.values()
                    if tp.pick(t, margin) == "warp")
    L = [_RED_BEGIN,
         "## Latest measured run (auto-refreshed by `bench/tune.py`)", "",
         f"_Source: `{src}` · NPROB=8192 ns/problem · margin ±{margin*100:.0f}% "
         f"(warp/block are both dependency-free; pick = cheapest, note flags "
         f"sub-margin gaps) · warp picked in {warp_wins} of {len(cells)} cells._", ""]
    if block_only_note:
        L += [block_only_note, ""]
    L += ["| op | shape | dtype | block ns | warp ns | pick | note |",
          "|----|-------|-------|----------|---------|------|------|"]
    for (dt, op, shape) in sorted(cells, key=_cell_key):
        t = cells[(dt, op, shape)]
        win, note = tp.verdict(t, margin)
        warp_s = f"{t['warp']:.2f}" if "warp" in t else "—"
        L.append(f"| {op} | {_shape_str(shape)} | {dt} | {t['block']:.2f} | "
                 f"{warp_s} | **{win}** | {note} |")
    L += ["", _RED_END]
    return "\n".join(L)


def report_pick_leg(name, txt, txt_name, md_path, margin, parse, dry_run,
                    changed, block_only_note=""):
    """Parse a blas2/rect sweep, generate the pick block, splice into md_path."""
    cells = parse(txt)
    if not cells:
        print(f"  ⚠️ no NPROB=8192 rows parsed from {txt_name}; nothing written.")
        return
    print(f"  {len(cells)} (dtype,op,shape) cells parsed")
    block = gen_pick_block(cells, margin, txt_name, block_only_note)
    existing = md_path.read_text() if md_path.exists() else f"# {name} sweep — measured results\n"
    md = splice_reduced_md(existing, block)
    if dry_run:
        changed[name] = show_diff(md_path, md, md_path.name)
    else:
        md_path.write_text(md)
        print(f"  wrote {md_path.relative_to(GLASS_DIR)}")


_BLAS2_NOTE = ("inv/trmv/ger are BLOCK-ONLY (no `glass::warp::` variant); "
               "none of these ops has a `glass::nvidia::` counterpart.")
_RECT_NOTE = ("nvidia leg skipped for rectangular shapes (needs new per-shape "
              "DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in "
              "the `shapes` leg).")


# ─── solvers leg: characterization only (measured + reported, NEVER picked) ──
# bdsv and pcg ARE two impls of the same block-tridiagonal SPD solve, but the
# right choice is problem-dependent (pcg's cost scales with the iteration count,
# i.e. with conditioning; bdsv is exact in one serial-over-knots sweep) — so this
# leg records the measured crossover on the harness's well-conditioned test
# system instead of regenerating any dispatch table. gesv/posv/inv-solve and
# syev/eig_clamp rows are pure characterization (what does the robustness /
# anti-pattern path cost where Cholesky suffices).

# The solvers harness restores mutated state between reps (untimed), so each rep
# is a full restore+launch round-trip — budget far fewer reps than the ladder.
_SOLVERS_SCHED = [("8192", "50")]

_SLV_HDR_RE = re.compile(r"NPROB=(\d+).*dtype=(f32|f64)")
_SLV_A_RE = re.compile(
    r"^bdsv_pcg\s+BS=(\d+)\s+KP=(\d+)\b.*?it=(\d+)\s*\|\|\s*"
    r"bdsv tb\d+=([\d.]+)\s+pcg tb\d+=([\d.]+)")
_SLV_B_RE = re.compile(
    r"^spdsv\s+N=(\d+)\b.*\|\|\s*gesv=([\d.]+)\s+posv=([\d.]+)\s+invsv=([\d.]+)"
    r"(?:\s+thr-posv=([\d.]+))?")
# The thr-posv group is OPTIONAL: it is absent from archived solver sweeps (every
# run before the thread tier existed — replayed via --from-solvers) and from live
# rows at N>7, where bench_solvers skips the tier (register-residency ceiling).
# Keep it optional or old sweeps stop parsing and the section-B table drops.
_SLV_C_RE = re.compile(r"^(syev|eig_clamp)\s+N=(\d+)\b.*\|\|\s*best tb\d+=([\d.]+)")


def parse_solvers(text, nprob=8192):
    """Parse a bench_solvers sweep at ``nprob`` into three keyed dicts:

    ``bdsv_pcg``: ``(dtype, BS, KP) -> {bdsv, pcg, iters}`` (best-TB ns/problem
    from the ``||`` summary + pcg's converged iteration count);
    ``spdsv``:    ``(dtype, N) -> {gesv, posv, invsv[, thrposv]}`` (``thrposv`` is
    the thread-tier ``glass::thread::posv``, present only for N<=7);
    ``eig``:      ``(dtype, N) -> {syev, eig_clamp}``.
    """
    data = {"bdsv_pcg": {}, "spdsv": {}, "eig": {}}
    dtype, cur = None, None
    for line in text.splitlines():
        if line.startswith("####"):
            m = _SLV_HDR_RE.search(line)
            if m:
                cur, dtype = int(m.group(1)), m.group(2)
            continue
        if cur != nprob:
            continue
        s = line.strip()
        m = _SLV_A_RE.match(s)
        if m:
            data["bdsv_pcg"][(dtype, int(m.group(1)), int(m.group(2)))] = dict(
                iters=int(m.group(3)), bdsv=float(m.group(4)), pcg=float(m.group(5)))
            continue
        m = _SLV_B_RE.match(s)
        if m:
            rec = dict(gesv=float(m.group(2)), posv=float(m.group(3)),
                       invsv=float(m.group(4)))
            if m.group(5):                       # thread-tier posv (N<=7 only)
                rec["thrposv"] = float(m.group(5))
            data["spdsv"][(dtype, int(m.group(1)))] = rec
            continue
        m = _SLV_C_RE.match(s)
        if m:
            data["eig"].setdefault((dtype, int(m.group(2))), {})[m.group(1)] = \
                float(m.group(3))
    return data


def gen_solvers_block(data, src):
    """Marker-delimited measured block for SOLVERS_SWEEP_RESULTS.md (no picks)."""
    L = [_RED_BEGIN,
         "## Latest measured run (auto-refreshed by `bench/tune.py`)", "",
         f"_Source: `{src}` · NPROB=8192 ns/problem (best swept TB, min of 3 "
         f"trials, restore-outside-timing protocol) · characterization only — "
         f"no dispatch table is regenerated._", ""]
    A = data["bdsv_pcg"]
    if A:
        bw = sum(1 for v in A.values() if v["bdsv"] <= v["pcg"])
        L += ["### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input", "",
              f"bdsv is faster in {bw} of {len(A)} cells **on this well-conditioned "
              f"test system** (see the iters column — pcg's cost scales with the "
              f"iteration count, so the crossover moves with conditioning).", "",
              "| BlockSize | Knots | dtype | bdsv ns | pcg ns | pcg iters | pcg/bdsv |",
              "|-----------|-------|-------|---------|--------|-----------|----------|"]
        for key in sorted(A, key=lambda k: (k[1], k[2], k[0])):
            v = A[key]
            L.append(f"| {key[1]} | {key[2]} | {key[0]} | {v['bdsv']:.2f} | "
                     f"{v['pcg']:.2f} | {v['iters']} | {v['pcg']/v['bdsv']:.2f} |")
        L.append("")
    B = data["spdsv"]
    if B:
        has_thr = any("thrposv" in v for v in B.values())
        L += ["### gesv vs posv vs inv+gemv — same SPD system, single RHS", "",
              "posv (Cholesky) is the intended SPD path; gesv prices the pivoted-LU "
              "robustness fallback, inv+gemv the invert-then-multiply anti-pattern."]
        if has_thr:
            L += ["", "The `thr-posv` column is the **thread-tier** `glass::thread::posv` "
                  "(one problem per thread, 32 packed per warp) — measured only below the "
                  "N<=7 register-residency ceiling. Where `thr/posv` < 1 the thread tier "
                  "beats the block Cholesky solve on that low-DOF shape."]
        if has_thr:
            L += ["",
                  "| N | dtype | gesv ns | posv ns | inv+gemv ns | thr-posv ns | gesv/posv | inv/posv | thr/posv |",
                  "|---|-------|---------|---------|-------------|-------------|-----------|----------|----------|"]
            for key in sorted(B, key=lambda k: (k[1], k[0])):
                v = B[key]
                thr = f"{v['thrposv']:.2f}" if "thrposv" in v else "—"
                thr_ratio = f"{v['thrposv']/v['posv']:.2f}" if "thrposv" in v else "—"
                L.append(f"| {key[1]} | {key[0]} | {v['gesv']:.2f} | {v['posv']:.2f} | "
                         f"{v['invsv']:.2f} | {thr} | {v['gesv']/v['posv']:.2f} | "
                         f"{v['invsv']/v['posv']:.2f} | {thr_ratio} |")
        else:
            L += ["",
                  "| N | dtype | gesv ns | posv ns | inv+gemv ns | gesv/posv | inv/posv |",
                  "|---|-------|---------|---------|-------------|-----------|----------|"]
            for key in sorted(B, key=lambda k: (k[1], k[0])):
                v = B[key]
                L.append(f"| {key[1]} | {key[0]} | {v['gesv']:.2f} | {v['posv']:.2f} | "
                         f"{v['invsv']:.2f} | {v['gesv']/v['posv']:.2f} | "
                         f"{v['invsv']/v['posv']:.2f} |")
        L.append("")
    C = data["eig"]
    if C:
        L += ["### syev + eig_clamp — timing only (no contender)", "",
              "| N | dtype | syev ns | eig_clamp ns |",
              "|---|-------|---------|--------------|"]
        for key in sorted(C, key=lambda k: (k[1], k[0])):
            v = C[key]
            sy = f"{v['syev']:.2f}" if "syev" in v else "—"
            ec = f"{v['eig_clamp']:.2f}" if "eig_clamp" in v else "—"
            L.append(f"| {key[1]} | {key[0]} | {sy} | {ec} |")
        L.append("")
    L.append(_RED_END)
    return "\n".join(L)


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
    p.add_argument("--sched", default=None, metavar="N:R[,N:R...]",
                   help="override the ladder/body NPROB:reps schedule, e.g. "
                        "'64:200,1024:100,8192:50' — the Tegra profile; keep "
                        "an 8192 section, the regenerated tables read it")
    p.add_argument("--quick", action="store_true",
                   help="ladder: throughput point only (NPROB=8192), fewer reps")
    p.add_argument("--prebuild", action="store_true",
                   help="compile every binary the selected legs need into the "
                        "build cache and exit — no timing. Run this ANYTIME (even "
                        "while the GPU is busy; compilation is CPU-bound), so the "
                        "later sweep on a quiet GPU is execute-only and fast.")
    p.add_argument("--build-jobs", type=int, default=1,
                   help="parallel nvcc compiles for --prebuild (default 1). Each "
                        "cuBLASDx compile needs ~6-7GB RAM, so size to free_RAM/7 "
                        "(e.g. 6 on a 64GB box). The timed legs always run serially.")
    p.add_argument("--iters", type=int, default=200000, help="bench_reduced iters")
    p.add_argument("--allow-no-mathdx", action="store_true",
                   help="let the ladder leg run without MATHDX_ROOT as a 3-tier "
                        "(thread/warp/block) SIMT sweep — the regenerated table "
                        "simply lacks the nvidia contender. Required on Tegra/"
                        "Jetson, where MathDx does not ship.")
    p.add_argument("--dry-run", action="store_true",
                   help="regenerate + diff against in-tree tables, write nothing")
    p.add_argument("--from-ladder", metavar="TXT",
                   help="skip ladder build/run; regenerate from this mega_sweep .txt")
    p.add_argument("--from-body", metavar="TXT",
                   help="skip body build/run; regenerate dispatch_body() from "
                        "this body_dispatch_sweep .txt")
    p.add_argument("--from-reduced", metavar="TXT",
                   help="skip reduced build/run; analyze from this bench_reduced .txt")
    p.add_argument("--from-blas2", metavar="TXT",
                   help="skip blas2 build/run; report from this bench_blas2 sweep .txt")
    p.add_argument("--from-rect", metavar="TXT",
                   help="skip rect build/run; report from this bench_rect sweep .txt")
    p.add_argument("--from-solvers", metavar="TXT",
                   help="skip solvers build/run; report from this bench_solvers sweep .txt")
    args = p.parse_args()
    user_sched = None
    if args.sched:
        try:
            user_sched = [tuple(c.split(":")) for c in args.sched.split(",")]
            assert all(len(t) == 2 and t[0].isdigit() and t[1].isdigit()
                       for t in user_sched)
        except (AssertionError, ValueError):
            sys.exit(f"ERROR: bad --sched {args.sched!r}; want 'N:R,N:R,...'")

    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    bad = [l for l in legs if l not in ALL_LEGS]
    if bad:
        sys.exit(f"unknown leg(s) {bad}; choose from {ALL_LEGS}")
    offline = bool(args.from_ladder or args.from_body or args.from_reduced
                   or args.from_blas2 or args.from_rect or args.from_solvers)
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
            build_mega_sweep(sms, mdx, args.allow_no_mathdx)
        if "body" in legs:
            print("── prebuild: body ────────────────────────────────────────")
            build_body(sms)
        if "shapes" in legs:
            print("── prebuild: shapes (cuBLASDx microbenches) ──────────────")
            if mdx is None:
                print("  [skip] shapes needs MATHDX_ROOT (cuBLASDx).")
            else:
                run([sys.executable, "autotune.py", "--sm", str(sms),
                     "--build-only", "--build-jobs", str(args.build_jobs),
                     "--build-dir", str(cache_dir(sms))], cwd=BENCH_DIR)
        if "reduced" in legs:
            print("── prebuild: reduced ─────────────────────────────────────")
            build_reduced(sms)
        if "blas2" in legs:
            print("── prebuild: blas2 ───────────────────────────────────────")
            build_blas2(sms)
        if "rect" in legs:
            print("── prebuild: rect ────────────────────────────────────────")
            build_rect(sms)
        if "solvers" in legs:
            print("── prebuild: solvers ─────────────────────────────────────")
            build_solvers(sms)
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
            if sms is None:
                sys.exit("ERROR: --from-ladder needs an explicit --sm (the arch "
                         "the sweep was MEASURED on — auto-detecting this host "
                         "could mislabel a foreign sweep, e.g. a Jetson capture "
                         "regenerated on the desktop).")
            sweep_path = pathlib.Path(args.from_ladder)
            sweep_text = sweep_path.read_text()
        else:
            binp = build_mega_sweep(sms, mdx, args.allow_no_mathdx)
            sweep_path = run_mega_sweep(binp, args.quick, sched=user_sched)
            sweep_text = sweep_path.read_text()
        new_defaults, n = regen_ladder(sweep_text, args.margin, sweep_path.name, sms)
        print(f"  regenerated ideal_sm{sms // 10} from {n} (dtype,op) groups")
        if args.dry_run:
            changed["ladder"] = show_diff(DEFAULTS, new_defaults, "glass-defaults.cuh")
        else:
            DEFAULTS.write_text(new_defaults)
            print(f"  wrote {DEFAULTS.relative_to(GLASS_DIR)}")

    # ── body (in-block body dispatch for the bare face) ──
    if "body" in legs:
        print("── body (bare-face in-block body dispatch) ───────────────")
        if args.from_body:
            if sms is None:
                sys.exit("ERROR: --from-body needs an explicit --sm (the arch "
                         "the sweep was MEASURED on — see --from-ladder).")
            body_path = pathlib.Path(args.from_body)
            body_text = body_path.read_text()
        elif sms is None:
            print("  [skip] body needs a GPU or --from-body.")
            body_text = None
        else:
            binp = build_body(sms)
            body_path = run_mega_sweep(binp, args.quick,
                                       prefix="body_dispatch_sweep",
                                       sched=user_sched)
            body_text = body_path.read_text()
        if body_text is not None:
            new_defaults, moved = regen_body(body_text, args.margin,
                                             body_path.name, sms)
            print(f"  regenerated body_sm{sms // 10}: {moved} cells moved off "
                  "the block body")
            if args.dry_run:
                changed["body"] = show_diff(DISPATCH_HDR, new_defaults,
                                            "glass-dispatch.cuh")
            else:
                DISPATCH_HDR.write_text(new_defaults)
                print(f"  wrote {DISPATCH_HDR.relative_to(GLASS_DIR)}")

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

    # ── blas2 / rect (warp-vs-block picks, reported; no header table yet) ──
    for leg, from_arg, builder, prefix, md_path, parse, note in (
            ("blas2", args.from_blas2, build_blas2, "blas2_sweep",
             BLAS2_MD, tp.parse_blas2, _BLAS2_NOTE),
            ("rect", args.from_rect, build_rect, "rect_sweep",
             RECT_MD, tp.parse_rect, _RECT_NOTE)):
        if leg not in legs:
            continue
        print(f"── {leg} (warp vs block, ns/problem) ─────────────────────")
        if from_arg:
            txt_path = pathlib.Path(from_arg)
            txt = txt_path.read_text()
        elif sms is None:
            print(f"  [skip] {leg} needs a GPU or --from-{leg}.")
            continue
        else:
            binp = builder(sms)
            txt_path = run_mega_sweep(binp, args.quick, prefix)
            txt = txt_path.read_text()
        report_pick_leg(leg, txt, txt_path.name, md_path, args.margin,
                        parse, args.dry_run, changed, note)

    # ── solvers (characterization only — measured + reported, never picked) ──
    if "solvers" in legs:
        print("── solvers (bdsv vs pcg, gesv/posv/inv-solve, syev) ──────")
        txt = None
        if args.from_solvers:
            txt_path = pathlib.Path(args.from_solvers)
            txt = txt_path.read_text()
        elif sms is None:
            print("  [skip] solvers needs a GPU or --from-solvers.")
        else:
            binp = build_solvers(sms)
            txt_path = run_mega_sweep(binp, args.quick, "solvers_sweep",
                                      sched=_SOLVERS_SCHED)
            txt = txt_path.read_text()
        if txt:
            data = parse_solvers(txt)
            if not any(data.values()):
                print(f"  ⚠️ no NPROB=8192 rows parsed from {txt_path.name}; "
                      "nothing written.")
            else:
                print(f"  parsed: {len(data['bdsv_pcg'])} bdsv-vs-pcg, "
                      f"{len(data['spdsv'])} spd-solve, {len(data['eig'])} "
                      "syev/eig_clamp cells")
                block = gen_solvers_block(data, txt_path.name)
                existing = (SOLVERS_MD.read_text() if SOLVERS_MD.exists()
                            else "# solvers sweep — measured results\n")
                md = splice_reduced_md(existing, block)
                if args.dry_run:
                    changed["solvers"] = show_diff(SOLVERS_MD, md, SOLVERS_MD.name)
                else:
                    SOLVERS_MD.write_text(md)
                    print(f"  wrote {SOLVERS_MD.relative_to(GLASS_DIR)}")

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
            r = run([sys.executable, "export_sweep_figures.py", sweep_for_fig], cwd=BENCH_DIR)
            if r.returncode != 0:
                print("  ⚠️ figures leg failed (needs matplotlib: `pip install matplotlib` "
                      "into the env running tune.py). Tables above are unaffected.")

    if args.dry_run:
        moved = [k for k, v in changed.items() if v]
        print(f"\n[dry run] tables that would change: {moved or 'none'}")
    print("\n==> done.")


if __name__ == "__main__":
    main()
