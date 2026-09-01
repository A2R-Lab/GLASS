#!/usr/bin/env python3
"""GLASS unified autotuner — one command to remeasure this GPU and regenerate
every shipped defaults table + figure under a single noise margin.

    python bench/tune.py --sm auto [--margin 0.05] [--quick] [--legs ...]

It drives the measurement harnesses and routes every verdict through the
one shared tie rule in ``bench/tune_pick.py`` (a dependency-carrying impl wins
only if it clears the margin; between SIMT tiers, any tier within the ±2% SIMT
tie band of the fastest takes the cell if it is simpler — thread ≻ warp ≻
block), so no table bakes sub-noise jitter and a pure-noise re-run reproduces
the same tables. The legs:

  ladder   bench_mega_sweep.cu + bench_solver_ladder.cu
                                → native/NVIDIA thread/warp/block ladder in glass-defaults.cuh
                                  (native thread — one problem/thread — is a
                                  dependency-free contender alongside warp/block:
                                  the shared pick takes the cheapest SIMT tier —
                                  with ties inside the ±2% SIMT band resolving to
                                  the simpler tier — so a fresh sweep emits
                                  `backend::thread` wherever the low-DOF packing
                                  actually wins; POTRF/TRSV/POSV are replaced by
                                  a symmetric fresh-valid-input sweep of every
                                  native and NVIDIA contender)
                                  (paired per-arch MathDx/native-only tables + the SM
                                  dispatch switch; a first-time arch — e.g. sm_87
                                  on a Jetson Orin — gets a new table + case,
                                  other arches' tables are left untouched)
  body     bench_body_dispatch.cu → compatible thread-0/warp-0/block bodies
                                  behind the fixed block-scope bare interface
  shapes   bench/autotune.py    → per-(M,N,K) cuBLASDx-vs-SIMT table in
                                  src/nvidia/tuning_table.cuh  (needs MATHDX_ROOT)
  reduced  bench_reduced.cu     → characterizes serial-vs-reduced crossover and
                                  validates the conservative standard policy; rewrites RESULTS.md
  blas2    bench_blas2.cu       → warp/block sweep of the ops the ladder misses
                                  (syrk/syr2k/ldlt/ldlt_solve/inv/trmv/ger); reports picks
                                  into RESULTS.md (blas2 section) + regenerates the
                                  per-arch blas2_sm* table (2-impl ops only)
  rect     bench_rect.cu        → warp/block sweep of rectangular gemv/gemm shapes;
                                  reports picks into RESULTS.md (rect section) +
                                  regenerates the exact-shape rect_*_sm* pickers
  solvers  bench_solvers.cu     → solver characterization (bdsv-vs-pcg crossover,
                                  gesv/posv/inv-solve on SPD, syev/eig_clamp,
                                  eigh/psd_project) into
                                  RESULTS.md (solvers section) — measured, never picked
                                  (the bdsv/pcg choice is conditioning-dependent)
  figures  export_sweep_figures → docs _static/*.png ladders + sweep_winners.txt

All ops are *measured and recorded*; a dispatch picker is regenerated only for
ops with ≥2 genuinely competing implementations. Reduced GEMM remains an
explicit opt-in because its isolated wins do not justify another public advisor
axis. Single-implementation families are reported, not picked.

EXECUTION DISCIPLINE: perf timing must be ISOLATED — run on a quiet GPU with no
concurrent CPU/GPU load. Build/iterate the tool offline with the ``--from-*``
hooks (feed an existing sweep .txt; no GPU touched). ``--dry-run`` regenerates
into memory and diffs against the in-tree tables WITHOUT writing — use it to
confirm a re-run only moves dispatch inside the tie band before committing.
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time

import tune_pick as tp
from autotune import lib_digest  # shared library-content hash for cache keys

BENCH_DIR = pathlib.Path(__file__).parent.resolve()
GLASS_DIR = BENCH_DIR.parent
DEFAULTS  = GLASS_DIR / "glass-defaults.cuh"
DISPATCH_HDR = GLASS_DIR / "glass-dispatch.cuh"
STATIC    = GLASS_DIR / "docs" / "source" / "_static"
# Single measured-results archive (consolidated 2026-08-11): every leg's
# "latest measured run" splices into its own marker-delimited section of
# bench/RESULTS.md. Curated analysis lives on the docs site; raw captures are
# archived externally.
RESULTS_MD = BENCH_DIR / "RESULTS.md"
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


# Quiet-GPU discipline and provenance stamps are shared across the four
# benchmark drivers — one copy in bench_common (they drifted as four copies).
from bench_common import (compute_pids, gpu_busy, require_quiet_gpu,
                          source_digest, watch_process)
import bench_common


def provenance(leg, sms, margin):
    return bench_common.provenance(
        leg, f"arch=sm_{sms // 10} decision_margin={margin:.6f}")


def run_isolated(argv, force=False):
    """Run one timed process and invalidate it if another compute PID appears."""
    require_quiet_gpu(force)
    # Under --force, pre-existing PIDs were deliberately tolerated — subtract
    # them so the mid-run check catches only NEW processes (previously the
    # very PIDs --force accepted re-tripped it, making the flag self-defeating).
    baseline = compute_pids() if force else frozenset()
    # watch_process() polls the child before communicate() is called.  A PIPE
    # therefore deadlocks once a verbose harness (notably the raw-sample solver
    # ladder) fills the kernel pipe buffer.  A seekable temporary file retains
    # the same all-or-nothing capture semantics without requiring a second
    # reader thread or buffering output in the child.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as capture:
        proc = subprocess.Popen([str(x) for x in argv], cwd=BENCH_DIR, text=True,
                                stdout=capture, stderr=subprocess.STDOUT)
        _, foreign = watch_process(proc, baseline)
        capture.seek(0)
        stdout = capture.read()
    if foreign:
        sys.exit(f"ERROR: timing invalidated; foreign compute PIDs appeared: "
                 f"{sorted(foreign)}")
    if proc.returncode != 0:
        sys.exit(f"ERROR: timed harness exited {proc.returncode}: {' '.join(argv)}\n"
                 + "\n".join(stdout.splitlines()[-20:]))
    return stdout


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
    """MathDx ladder build for hosts where libcusolverdx.a is foreign (the
    tarball ships x86-64 objects only — e.g. Jetson/aarch64). cuSOLVERDx also
    ships an LTO-IR `libcusolverdx.fatbin`, which is host-arch-independent but
    only legal as a DEVICE-LINK input, so the build is staged:
    -dc (LTO) -> -dlto -dlink with the fatbin -> final host link.
    Verified on AGX Orin (sm_87, CUDA 13.2) 2026-07-31."""
    common = ["-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
              "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
              f"-I{mdx/'include'}", f"-I{mdx/'external'/'cutlass'/'include'}",
              "-DGLASS_BENCH_CUBLASDX", "-DGLASS_BENCH_CUSOLVERDX",
              f"-DGLASS_TARGET_SM={sms}",
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
                 f"-DGLASS_TARGET_SM={sms}", "bench_mega_sweep.cu"]
        binp, status = cached_build("mega_sweep_simt", "bench_mega_sweep.cu",
                                    flags, sms)
    elif platform.machine() != "x86_64":
        # The tarball's libcusolverdx.a is x86-64-only; use the fatbin path.
        print("  bench_mega_sweep: non-x86 host -> NVIDIA block/thread via cusolverdx FATBIN")
        binp, status = _fatbin_build_mega(sms, mdx)
    else:
        flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
                 "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
                 f"-I{mdx/'include'}", f"-I{mdx/'external'/'cutlass'/'include'}",
                 "-DGLASS_BENCH_CUBLASDX", "-DGLASS_BENCH_CUSOLVERDX",
                 f"-DGLASS_TARGET_SM={sms}",
                 "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", "-rdc=true", "-dlto",
                 f"-L{mdx/'lib'}", "-lcusolverdx", "-lcublas", "-lcusolver", "-lcudart",
                 "bench_mega_sweep.cu"]
        binp, status = cached_build("mega_sweep", "bench_mega_sweep.cu", flags, sms)
    if status == "fail":
        sys.exit("ERROR: bench_mega_sweep compile failed.")
    print(f"  bench_mega_sweep: {status} ({binp.name})")
    return binp


def _fatbin_build_solver_ladder(sms, mdx):
    """Build the fresh-input solver ladder on non-x86 MathDx hosts."""
    common = ["-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
              "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
              f"-I{mdx/'include'}", f"-I{mdx/'external'/'cutlass'/'include'}",
              "-DGLASS_BENCH_CUSOLVERDX",
              f"-DGLASS_TARGET_SM={sms}",
              "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT"]
    src = (BENCH_DIR / "bench_solver_ladder.cu").read_bytes()
    key = hashlib.sha256(src + lib_digest().encode()
                         + " ".join(common + ["fatbin"]).encode()).hexdigest()[:12]
    binp = cache_dir(sms) / f"solver_ladder_fatbin_{key}"
    if binp.exists():
        return binp, "cached"
    obj, dlk = str(binp) + ".o", str(binp) + "_dlink.o"
    steps = [
        ["nvcc"] + common + ["-rdc=true", "-dlto", "-dc",
                             "bench_solver_ladder.cu", "-o", obj],
        ["nvcc", f"-arch=sm_{sms // 10}", "-dlto", "-dlink", obj,
         str(mdx / "lib" / "libcusolverdx.fatbin"), "-o", dlk],
        ["nvcc", f"-arch=sm_{sms // 10}", obj, dlk,
         "-lcublas", "-lcusolver", "-lcudart", "-o", str(binp)],
    ]
    for cmd in steps:
        if run(cmd, cwd=BENCH_DIR).returncode != 0:
            return None, "fail"
    return binp, "built"


def build_solver_ladder(sms, mdx):
    if mdx is None:
        print("  bench_solver_ladder: MathDx absent -> native contenders only")
        flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
                 "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
                 f"-DGLASS_TARGET_SM={sms}", "bench_solver_ladder.cu"]
        binp, status = cached_build("solver_ladder_simt",
                                    "bench_solver_ladder.cu", flags, sms)
    elif platform.machine() != "x86_64":
        binp, status = _fatbin_build_solver_ladder(sms, mdx)
    else:
        flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms // 10}", "-O3",
                 "--expt-relaxed-constexpr", "-Xptxas", "-O1", "-I..", "-I../src",
                 f"-I{mdx/'include'}", f"-I{mdx/'external'/'cutlass'/'include'}",
                 "-DGLASS_BENCH_CUSOLVERDX",
                 f"-DGLASS_TARGET_SM={sms}",
                 "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", "-rdc=true", "-dlto",
                 f"-L{mdx/'lib'}", "-lcusolverdx", "-lcublas", "-lcusolver", "-lcudart",
                 "bench_solver_ladder.cu"]
        binp, status = cached_build("solver_ladder", "bench_solver_ladder.cu",
                                    flags, sms)
    if status == "fail":
        sys.exit("ERROR: bench_solver_ladder compile failed.")
    print(f"  bench_solver_ladder: {status} ({binp.name})")
    return binp


def run_solver_ladder(binp, quick, sms, margin, sched=None, force=False,
                      rounds=9, seed=1):
    """Capture every solver contender on fresh valid inputs.

    Input generation/restoration is outside the timed region.  Implementations
    and launch shapes are randomized within each paired round; all raw samples
    are retained so table generation and paper analysis need not reuse a
    min-of-three summary.
    """
    path = BENCH_DIR / f"solver_ladder_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    lines = provenance("solver_ladder", sms, margin)
    lines += [f"# solver_ladder  {time.strftime('%c')}  (bench/tune.py)",
              f"# paired_rounds={rounds} seed={seed} inputs=fresh_valid", ""]
    path.write_text("\n".join(lines) + "\n")
    if sched is None:
        sched = _QUICK_SCHED if quick else _FULL_SCHED
    slots = {64: 2048, 1024: 256, 8192: 64}
    for nprob_text, _ in sched:
        nprob = int(nprob_text)
        requested_slots = slots.get(nprob, max(8, min(2048, 131072 // nprob)))
        for dtype in ("f32", "f64"):
            print(f"  -> solver_ladder NPROB={nprob} slots={requested_slots} "
                  f"rounds={rounds} {dtype}")
            stdout = run_isolated(
                [str(binp), nprob, requested_slots, dtype, rounds, seed], force)
            with open(path, "a") as stream:
                stream.write(stdout)
                stream.write("\n")
    print(f"==> wrote {path.relative_to(GLASS_DIR)}")
    return path


def run_mega_sweep(binp, quick, sms, margin, prefix="mega_sweep", sched=None,
                   force=False):
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
    path = BENCH_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    def _telemetry():
        """One `# telemetry ...` line (SM clock + temp) — stamped per SECTION,
        not just once per file, so thermal/clock drift across a multi-hour sweep
        is visible in the capture itself (clocks are deliberately NOT locked:
        users run at stock boost, so we tune at stock boost — see TUNING.md)."""
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.sm,temperature.gpu",
                 "--format=csv,noheader"], text=True).strip()
            return f"# telemetry {smi}"
        except Exception:
            return "# telemetry n/a (no nvidia-smi query support on this host)"

    hdr = provenance(prefix, sms, margin)
    hdr += [f"# {prefix}  {time.strftime('%c')}  (bench/tune.py)", _telemetry(), ""]
    path.write_text("\n".join(hdr) + "\n")
    for nprob, reps in sched:
        for dt in ("f32", "f64"):
            print(f"  -> {prefix} NPROB={nprob} reps={reps} {dt}")
            stdout = run_isolated([str(binp), nprob, reps, dt], force)
            with open(path, "a") as f:
                f.write(f"################ NPROB={nprob}  reps={reps}  dtype={dt} ################\n")
                f.write(_telemetry() + "\n")
                f.write(stdout)
                f.write(_telemetry() + "  [section end]\n")
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
    def cpp_backend(name):
        return "nvidia_block" if name == "nvidia" else name
    if len(runs) == 1:
        return f"backend::{cpp_backend(runs[0][1])}"
    parts = [f"N <= {hi}u ? backend::{cpp_backend(be)}" for hi, be in runs[:-1]]
    return " : ".join(parts) + f" : backend::{cpp_backend(runs[-1][1])}"


# Ladder-style rows print `spread<=X%` (row-max over cells); the reduced and
# solvers/AB harnesses print per-measurement `spread=X%`. Match both.
_SPREAD_RE = re.compile(r"spread<?=+([\d.]+)%")


def warn_jittery_rows(text, margin, label):
    """Audit hook (2026-08-11): sweep rows carry `spread<=X%` — the worst
    (max−min)/min over the timer's 3 trials across the row's cells. A row whose
    spread exceeds the decision margin cannot cleanly resolve that margin, so
    warn rather than silently bake jitter into a shipped table. Pre-2026-08
    captures carry no spread tokens → nothing to check (replays stay valid)."""
    jittery = []
    for line in text.splitlines():
        m = _SPREAD_RE.search(line)
        if m and float(m.group(1)) > margin * 100.0:
            jittery.append((float(m.group(1)), line.split("||")[0].strip()[:48]))
    if jittery:
        jittery.sort(reverse=True)
        print(f"  !! {label}: {len(jittery)} row(s) with trial spread > "
              f"{margin*100:.0f}% margin (worst {jittery[0][0]:.1f}% @ "
              f"'{jittery[0][1]}') — treat those verdicts as unresolved; "
              f"re-capture on a quieter GPU if any of them matters.")
    return len(jittery)


def winners_from_sweep(text, margin, native_only=False, solver_text=None):
    """(dtype, op) -> {N: backend} under the shared dependency margin.

    ``native_only`` removes both vendor measurements before applying the same
    native SIMT tie rule. This emits an actual measured runner-up table instead
    of approximating vendor cells with a size heuristic.
    """
    cells = tp.parse_mega_sweep(text, nprob=8192)
    if solver_text is not None:
        solver_cells = tp.parse_solver_ladder(solver_text, nprob=8192)
        expected = {key for key in cells if key[1] in {"potrf", "trsv", "posv"}}
        missing = sorted(expected - set(solver_cells))
        if missing:
            summary = ", ".join(f"{dt}/{op}/N={N}" for dt, op, N in missing[:6])
            sys.exit("ERROR: fresh-input solver capture is incomplete at "
                     f"NPROB=8192 ({summary}{' ...' if len(missing) > 6 else ''}).")
        for key in expected:
            cells[key] = {impl: row["ns"]
                          for impl, row in solver_cells[key].items()}
    winners = {}
    for (dt, op, N), times in cells.items():
        if native_only:
            times = {name: value for name, value in times.items()
                     if name not in {"nvidia", "nvidia_thread"}}
        win = tp.pick(times, margin, {"nvidia", "nvidia_thread"})
        if win:
            winners.setdefault((dt, op), {})[N] = win
    return winners


def require_stable_solver_picks(solver_text, winners, native_winners, margin):
    """Fail closed when an emitted solver plan is noisy at decision scale."""
    cells = tp.parse_solver_ladder(solver_text, nprob=8192)
    unstable = []
    for policy_name, policy in (("full", winners), ("native", native_winners)):
        for (dtype, op), sizes in policy.items():
            if op not in {"potrf", "trsv", "posv"}:
                continue
            for N, impl in sizes.items():
                row = cells[(dtype, op, N)][impl]
                if row["spread"] > margin * 100.0:
                    unstable.append((policy_name, dtype, op, N, impl,
                                     row["cfg"], row["spread"]))
    if unstable:
        detail = "; ".join(
            f"{policy} {dtype}/{op}/N={N} {impl}/{cfg} spread={spread:.2f}%"
            for policy, dtype, op, N, impl, cfg, spread in unstable[:8])
        sys.exit("ERROR: selected fresh-input solver plan has round spread "
                 f"above the ±{margin*100:.0f}% decision margin: {detail}. "
                 "Recapture on a quieter GPU.")


def emit_ideal_body(winners, fname, ops=tp.LADDER_OPS):
    lines = [f"constexpr backend {fname}(op o, uint32_t N, bool f64) {{",
             "    switch (o) {"]
    for op in ops:
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


def regen_ladder(sweep_text, margin, src_name, sms,
                 solver_text=None, solver_name=None):
    """Regenerate this arch's table block in glass-defaults.cuh (replace if the
    arch was swept before, insert after the last table block if it's new) and
    rebuild the SM dispatch case-list from every table block present."""
    arch = sms // 10
    warn_jittery_rows(sweep_text, margin, "ladder")
    if solver_text is None:
        sys.exit("ERROR: ladder regeneration requires a symmetric fresh-input "
                 "solver capture; pass --from-solver-ladder or run online.")
    winners = winners_from_sweep(sweep_text, margin, solver_text=solver_text)
    native_winners = winners_from_sweep(
        sweep_text, margin, native_only=True, solver_text=solver_text)
    if not winners:
        sys.exit("ERROR: no NPROB=8192 verdicts parsed from the sweep.")
    require_stable_solver_picks(solver_text, winners, native_winners, margin)
    solver_rows = tp.parse_solver_configs(solver_text, nprob=8192)
    begin, end = _LAD_BEGIN.format(a=arch), _LAD_END.format(a=arch)
    region = "\n".join([
        begin,
        f"// Source sweep: {src_name}   tie margin: ±{margin*100:.0f}% "
        "(NVIDIA block/thread must clear it; SIMT ties ±2% prefer thread>warp>block)",
        f"// Fresh-input solver sweep: {solver_name} "
        f"({len(solver_rows)} measured execution plans; symmetric selection)",
        "// Paired tables preserve the measured native runner-up for callers that",
        "// do not opt into MathDx; both use the same capture and SIMT tie rule.",
        emit_ideal_body(winners, f"ideal_sm{arch}"),
        "",
        emit_ideal_body(native_winners, f"native_sm{arch}"),
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
    cases = "".join(f"        case {a * 10}u: return allow_nvidia "
                    f"? ideal_sm{a}(o, N, f64) : native_sm{a}(o, N, f64);\n"
                    for a in arches)
    pre, _, rest = text.partition(_DIS_BEGIN)
    _, _, post = rest.partition(_DIS_END)
    text = pre + _DIS_BEGIN + "\n" + cases + "        " + _DIS_END + post
    return text, len(winners), len(solver_rows)


# ─── blas2 + rect header tables (glass-defaults.cuh; warp-vs-block only) ─────
#
# Same marker-block model as the ladder: each swept arch owns a spliced
# `blas2_sm<A>` / `rect_*_sm<A>` block and a dispatch case; other arches are
# untouched and unmeasured arches fall to block (the incumbent). Only the
# 2-impl blas2 ops are tabled — inv/trmv/ger are block-only and stay
# report-only by the "single-impl families are reported, not picked" rule.

B2_TABLE_OPS = ("syrk", "syr2k", "ldlt", "ldlt_solve")

_B2_BEGIN = "// === BEGIN tune.py blas2 sm_{a} ==="
_B2_END   = "// === END tune.py blas2 sm_{a} ==="
_B2_END_RE = re.compile(r"// === END tune\.py blas2 sm_\d+ ===")
_B2_RE     = re.compile(r"// === BEGIN tune\.py blas2 sm_(\d+) ===")
_B2_DIS_BEGIN = "// === BEGIN tune.py blas2 dispatch ==="
_B2_DIS_END   = "// === END tune.py blas2 dispatch ==="

_RECT_BEGIN = "// === BEGIN tune.py rect sm_{a} ==="
_RECT_END   = "// === END tune.py rect sm_{a} ==="
_RECT_END_RE = re.compile(r"// === END tune\.py rect sm_\d+ ===")
_RECT_RE     = re.compile(r"// === BEGIN tune\.py rect sm_(\d+) ===")
_RECT_DIS = (("// === BEGIN tune.py rect dispatch ===",
              "// === END tune.py rect dispatch ===", "rect_gemv_sm"),
             ("// === BEGIN tune.py rect gemm dispatch ===",
              "// === END tune.py rect gemm dispatch ===", "rect_gemm_sm"))


def _splice_arch_block(text, begin, end, end_re, region):
    """Replace an arch's marker block, or insert after the family's last one."""
    if begin in text:
        pre, _, rest = text.partition(begin)
        _, _, post = rest.partition(end)
        return pre + region + post
    ends = list(end_re.finditer(text))
    if not ends:
        sys.exit(f"ERROR: marker family for {begin!r} missing from {DEFAULTS.name}.")
    at = ends[-1].end()
    return text[:at] + "\n\n" + region + text[at:]


def _rebuild_dispatch(text, dis_begin, dis_end, arches, fn_prefix, argstr):
    cases = "".join(f"        case {a * 10}u: return {fn_prefix}{a}({argstr});\n"
                    for a in arches)
    pre, _, rest = text.partition(dis_begin)
    _, _, post = rest.partition(dis_end)
    return pre + dis_begin + "\n" + cases + "        " + dis_end + post


def regen_blas2_table(sweep_text, margin, src_name, sms):
    """Regenerate this arch's blas2_sm<A> block + dispatch in glass-defaults.cuh."""
    arch = sms // 10
    warn_jittery_rows(sweep_text, margin, "blas2")
    cells = tp.parse_blas2(sweep_text, nprob=8192)
    winners = {}
    for (dt, op, N), times in cells.items():
        if op not in B2_TABLE_OPS:
            continue                      # single-impl families: report, don't pick
        win = tp.pick(times, margin)
        if win:
            winners.setdefault((dt, op), {})[N] = win
    if not winners:
        sys.exit("ERROR: no NPROB=8192 blas2 verdicts parsed from the sweep.")
    begin, end = _B2_BEGIN.format(a=arch), _B2_END.format(a=arch)
    region = "\n".join([
        begin,
        f"// Source sweep: {src_name}   tie margin: ±{margin*100:.0f}% "
        "(SIMT ties ±2% prefer the simpler tier)",
        emit_ideal_body(winners, f"blas2_sm{arch}", ops=B2_TABLE_OPS),
        end])
    text = _splice_arch_block(DEFAULTS.read_text(), begin, end, _B2_END_RE, region)
    arches = sorted(int(a) for a in _B2_RE.findall(text))
    if _B2_DIS_BEGIN not in text or _B2_DIS_END not in text:
        sys.exit(f"ERROR: blas2 dispatch markers missing from {DEFAULTS.name}.")
    text = _rebuild_dispatch(text, _B2_DIS_BEGIN, _B2_DIS_END, arches,
                             "blas2_sm", "o, N, f64")
    return text, len(winners)


def _emit_rect_fn(cells, op_name, fname, dims_names, margin):
    """Exact-shape constexpr: measured shapes emit their winner; rest → block."""
    args = ", ".join(f"uint32_t {d}" for d in dims_names)
    lines = [f"constexpr backend {fname}({args}, bool f64) {{"]
    for f64, dt in ((False, "f32"), (True, "f64")):
        rows = sorted((dims, be) for (d, o, dims), t in cells.items()
                      if d == dt and o == op_name
                      for be in [tp.pick(t, margin)] if be)
        if not rows:
            continue
        lines.append(f"    if ({'f64' if f64 else '!f64'}) {{")
        for dims, be in rows:
            # The benchmark grammar retains its historical (M,K,N) tuple;
            # the public C++ API is uniformly (M,N,K).
            if op_name == "gemm":
                dims = (dims[0], dims[2], dims[1])
            cond = " && ".join(f"{n} == {v}u" for n, v in zip(dims_names, dims))
            cpp_backend = "nvidia_block" if be == "nvidia" else be
            lines.append(f"        if ({cond}) return backend::{cpp_backend};")
        lines.append("    }")
    lines += ["    return backend::block;", "}"]
    return "\n".join(lines)


def regen_rect_table(sweep_text, margin, src_name, sms):
    """Regenerate this arch's rect_gemv/gemm_sm<A> blocks + dispatch cases."""
    arch = sms // 10
    warn_jittery_rows(sweep_text, margin, "rect")
    cells = tp.parse_rect(sweep_text, nprob=8192)
    if not cells:
        sys.exit("ERROR: no NPROB=8192 rect rows parsed from the sweep.")
    begin, end = _RECT_BEGIN.format(a=arch), _RECT_END.format(a=arch)
    region = "\n".join([
        begin,
        f"// Source sweep: {src_name}   tie margin: ±{margin*100:.0f}% "
        "(SIMT ties ±2% prefer the simpler tier); exact shapes only",
        _emit_rect_fn(cells, "gemv", f"rect_gemv_sm{arch}", ("M", "N"), margin),
        _emit_rect_fn(cells, "gemm", f"rect_gemm_sm{arch}", ("M", "N", "K"), margin),
        end])
    text = _splice_arch_block(DEFAULTS.read_text(), begin, end, _RECT_END_RE, region)
    arches = sorted(int(a) for a in _RECT_RE.findall(text))
    for dis_begin, dis_end, fn_prefix in _RECT_DIS:
        if dis_begin not in text or dis_end not in text:
            sys.exit(f"ERROR: rect dispatch markers missing from {DEFAULTS.name}.")
        argstr = "M, N, f64" if "gemv" in fn_prefix else "M, N, K, f64"
        text = _rebuild_dispatch(text, dis_begin, dis_end, arches, fn_prefix, argstr)
    return text, len(cells)


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
BODY_OPS = ("dot", "gemv", "gemm", "potrf", "trsv", "posv", "eig3", "softmax")
_BODY_HDR_RE = re.compile(r"NPROB=(\d+)\s+reps=\d+\s+dtype=(f32|f64)")
_BODY_SEG_RE = re.compile(r"(BLOCKBODY|WARPBODY|THREADBODY)((?:\s+tb\d+=[0-9.]+)+)")
_BODY_NAME = {"BLOCKBODY": "block", "WARPBODY": "warp_in_block",
              "THREADBODY": "thread_in_block"}


def build_body(sms):
    flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms//10}", "-O3", "-I..", "-I../src",
             f"-DGLASS_TARGET_SM={sms}", "bench_body_dispatch.cu"]
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
        if not m:
            continue
        opn = "potrf" if m.group(1) == "chol" else m.group(1)
        if opn not in BODY_OPS:
            continue
        key = (dt, opn, int(m.group(2)))
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
    warn_jittery_rows(sweep_text, margin, "body")
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
        "                                               uint32_t sm = GLASS_TARGET_SM) {",
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


# ─── reduced leg: bench_reduced → characterize the explicit reduced path ───

def build_reduced(sms):
    flags = ["nvcc", "-std=c++17", f"-arch=sm_{sms//10}", "-O3", "-I..", "-I../src",
             "bench_reduced.cu"]
    binp, status = cached_build("reduced", "bench_reduced.cu", flags, sms)
    if status == "fail":
        sys.exit("ERROR: bench_reduced compile failed.")
    print(f"  bench_reduced: {status} ({binp.name})")
    return binp


def plan_uses_reduced(n_out, K, blockDim):
    """Conservative policy for the measured standard/reduced leg.

    Kept false after the 2026-08-14 sm_120 sweep: f32 had 0/48 wins and f64
    had 2/48 at one shape. A future plan may promote a repeatable region."""
    return False


def analyze_reduced(text, margin):
    rows = tp.parse_reduced(text)
    wins, mism = [], []
    for r in rows:
        win = tp.pick({"serial": r["serial"], "reduced": r["reduced"]},
                      margin, {"reduced"})
        measured_reduced = (win == "reduced")
        # bench_reduced computes C(M,K)=A(M,N)·B(N,K): the contracted dim is N
        # (n_out = M*K); keep these arguments if the plan gains a typed region.
        predicted = plan_uses_reduced(r["n_out"], r["N"], r["blockDim"])
        r["winner"] = win
        if measured_reduced:
            wins.append(r)
        if measured_reduced != predicted:
            mism.append(r)
    return rows, wins, mism


def _md_begin(leg):
    return f"<!-- BEGIN tune.py {leg} -->"


def _md_end(leg):
    return f"<!-- END tune.py {leg} -->"


def gen_reduced_block(rows, wins, mism, margin, src):
    """The auto-refreshed measured-data block (between markers). The surrounding
    curated narrative in RESULTS.md is preserved."""
    L = [_md_begin("reduced"),
         f"### Latest measured run (auto-refreshed by `bench/tune.py`)", "",
         f"_Source: `{src}` · tie margin ±{margin*100:.0f}% (reduced must clear "
         f"it) · {len(wins)} of {len(rows)} configs pick reduced._", ""]
    if wins:
        L += ["| dtype | M | N | K | n_out | blockDim | serial_us | reduced_us | ratio |",
              "|:------|---|---|---|-------|----------|-----------|------------|-------|"]
        for r in wins:
            L.append(f"| {r['dtype']} | {r['M']} | {r['N']} | {r['K']} | {r['n_out']} | "
                     f"{r['blockDim']} | {r['serial']:.4f} | {r['reduced']:.4f} | "
                     f"**{r['serial']/r['reduced']:.2f}** |")
        L.append("")
    L.append("The public advisor stays two-axis; reduced GEMM remains an explicit opt-in.")
    if mism:
        L += ["", f"⚠️ **{len(mism)} config(s) favor the explicit reduced "
              "variant while the conservative plan stays standard:**", ""]
        for r in mism:
            pred = "reduced" if plan_uses_reduced(r['n_out'], r['N'], r['blockDim']) else "serial"
            L.append(f"- {r['dtype']} {r['M']}×{r['N']}×{r['K']} bd={r['blockDim']} "
                     f"(n_out={r['n_out']}): measured **{r['winner']}**, plan **{pred}**")
    else:
        L += ["", "✅ Measurement matches the plan for every swept config."]
    L += ["", _md_end("reduced")]
    return "\n".join(L)


def splice_results_md(existing, block, leg):
    """Replace `leg`'s marker region in `existing` with `block`; if the leg has
    no markers yet, insert before '## Reproduce' (else append). Curated prose
    stays intact; each leg owns its own marker pair in bench/RESULTS.md."""
    begin, end = _md_begin(leg), _md_end(leg)
    if begin in existing and end in existing:
        pre, _, rest = existing.partition(begin)
        _, _, post = rest.partition(end)
        return pre + block + post
    anchor = "## Reproduce"
    if anchor in existing:
        pre, _, post = existing.partition(anchor)
        return pre + block + "\n\n" + anchor + post
    return existing.rstrip() + "\n\n" + block + "\n"


# ─── blas2 + rect legs: warp/block picks → md report + header tables ─────────
# These legs measure ops/shapes with no shipped defaults table: blas2 covers the
# ladder's blind-spot ops (syrk/syr2k/ldlt/ldltsv/inv/trmv/ger; no nvidia
# counterparts, so 2-way), rect covers rectangular gemv/gemm (nvidia skipped —
# per-shape cuBLASDx decisions live in the `shapes` leg). Verdicts route through
# tune_pick just like the tables; the results land in a marker-delimited block
# of bench/RESULTS.md until the defaults-table
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


def gen_pick_block(leg, cells, margin, src, block_only_note=""):
    """Marker-delimited measured-run block for the blas2/rect legs.

    ``cells``: ``(dtype, op, shape) -> {block[, warp]}`` ns/problem. warp and
    block are both dependency-free, so :func:`tune_pick.pick` resolves each cell
    to the cheapest impl; the verdict note records the gap vs the ±margin band.
    """
    warp_wins = sum(1 for t in cells.values()
                    if tp.pick(t, margin) == "warp")
    L = [_md_begin(leg),
         "### Latest measured run (auto-refreshed by `bench/tune.py`)", "",
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
    L += ["", _md_end(leg)]
    return "\n".join(L)


def report_pick_leg(name, txt, txt_name, md_path, margin, parse, dry_run,
                    changed, block_only_note=""):
    """Parse a blas2/rect sweep, generate the pick block, splice into md_path."""
    cells = parse(txt)
    if not cells:
        print(f"  ⚠️ no NPROB=8192 rows parsed from {txt_name}; nothing written.")
        return
    print(f"  {len(cells)} (dtype,op,shape) cells parsed")
    block = gen_pick_block(name, cells, margin, txt_name, block_only_note)
    existing = md_path.read_text() if md_path.exists() else f"# measured results\n"
    md = splice_results_md(existing, block, name)
    if dry_run:
        changed[name] = show_diff(md_path, md, md_path.name)
    else:
        md_path.write_text(md)
        print(f"  wrote {md_path.relative_to(GLASS_DIR)}")


_BLAS2_NOTE = ("inv/trmv/ger are BLOCK-ONLY (no `glass::warp::` variant, so "
               "nothing competes — reported, never picked); none of these ops "
               "has an NVIDIA counterpart. The 2-impl ops "
               "(syrk/syr2k/ldlt/ldlt_solve) regenerate the shipped per-arch "
               "`blas2_sm*` table in glass-defaults.cuh (since 2026-08-06).")
_RECT_NOTE = ("nvidia leg skipped for rectangular shapes (needs new per-shape "
              "DEFINE_NVIDIA_* machinery; cuBLASDx-vs-SIMT per (M,N,K) lives in "
              "the `shapes` leg). Measured shapes regenerate the shipped "
              "exact-shape `rect_*_sm*` pickers in glass-defaults.cuh "
              "(`recommend<op::gemv/gemm,T,dims...>`, since 2026-08-06); "
              "unmeasured shapes stay block.")


# ─── solvers leg: characterization only (measured + reported, NEVER picked) ──
# bdsv and pcg ARE two impls of the same block-tridiagonal SPD solve, but the
# right choice is problem-dependent (pcg's cost scales with the iteration count,
# i.e. with conditioning; bdsv is exact in one serial-over-knots sweep) — so this
# leg records the measured crossover on the harness's well-conditioned test
# system instead of regenerating any dispatch table. gesv/posv/inv-solve and
# adaptive/fixed-sweep eigensolver rows are pure characterization.

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
_SLV_C_RE = re.compile(
    r"^(syev|eig_clamp|eigh|psd_project)\s+N=(\d+)\b.*\|\|\s*best tb\d+=([\d.]+)")


def parse_solvers(text, nprob=8192):
    """Parse a bench_solvers sweep at ``nprob`` into three keyed dicts:

    ``bdsv_pcg``: ``(dtype, BS, KP) -> {bdsv, pcg, iters}`` (best-TB ns/problem
    from the ``||`` summary + pcg's converged iteration count);
    ``spdsv``:    ``(dtype, N) -> {gesv, posv, invsv[, thrposv]}`` (``thrposv`` is
    the thread-tier ``glass::thread::posv``, present only for N<=7);
    ``eig``:      ``(dtype, N) -> {syev, eig_clamp, eigh, psd_project}``.
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
    """Marker-delimited measured block for RESULTS.md, solvers section (no picks)."""
    L = [_md_begin("solvers"),
         "### Latest measured run (auto-refreshed by `bench/tune.py`)", "",
         f"_Source: `{src}` · NPROB=8192 ns/problem (best swept TB, min of 3 "
         f"trials, restore-outside-timing protocol) · characterization only — "
         f"no dispatch table is regenerated._", ""]
    A = data["bdsv_pcg"]
    if A:
        bw = sum(1 for v in A.values() if v["bdsv"] <= v["pcg"])
        L += ["### bdsv (direct) vs pcg (iterative) — identical block-tridiagonal SPD input", "",
              f"bdsv is faster in {bw} of {len(A)} cells **on this well-conditioned "
              f"test system at PCG's `rho = rᵀz` relative tolerance of 1e-6**. "
              f"This is an approximate-solve comparison, not matched final residual "
              f"accuracy; PCG cost and the crossover move with tolerance, conditioning, "
              f"and iteration count.", "",
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
        L += ["### Adaptive vs fixed-sweep symmetric eigensolvers", "",
              "| N | dtype | syev ns | eig_clamp ns | eigh ns | psd_project ns |",
              "|---|-------|---------|--------------|---------|----------------|"]
        for key in sorted(C, key=lambda k: (k[1], k[0])):
            v = C[key]
            sy = f"{v['syev']:.2f}" if "syev" in v else "—"
            ec = f"{v['eig_clamp']:.2f}" if "eig_clamp" in v else "—"
            eh = f"{v['eigh']:.2f}" if "eigh" in v else "—"
            pp = f"{v['psd_project']:.2f}" if "psd_project" in v else "—"
            L.append(f"| {key[1]} | {key[0]} | {sy} | {ec} | {eh} | {pp} |")
        L.append("")
    L.append(_md_end("solvers"))
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
    p.add_argument("--solver-rounds", type=int, default=9,
                   help="paired rounds per fresh-input solver plan (default 9)")
    p.add_argument("--solver-seed", type=int, default=1,
                   help="candidate-order seed for the solver ladder; use a "
                        "different seed for held-out Capture B")
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
                        "simply lacks NVIDIA contenders. Use only when MathDx "
                        "headers/fatbins are unavailable.")
    p.add_argument("--dry-run", action="store_true",
                   help="regenerate + diff against in-tree tables, write nothing")
    p.add_argument("--force", action="store_true",
                   help="allow timed runs despite a busy-GPU preflight; foreign "
                        "compute PIDs appearing during a leg still invalidate it")
    p.add_argument("--from-ladder", metavar="TXT",
                   help="reuse this mega_sweep .txt; with --from-solver-ladder, "
                        "regenerate fully offline, otherwise capture only the "
                        "missing fresh-input solver companion")
    p.add_argument("--from-solver-ladder", metavar="TXT",
                   help="symmetric fresh-valid-input POTRF/TRSV/POSV companion "
                        "capture required with --from-ladder")
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
    if args.solver_rounds < 3:
        p.error("--solver-rounds must be at least 3")
    if args.from_solver_ladder and not args.from_ladder:
        p.error("--from-solver-ladder is a companion to --from-ladder")
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
    offline = bool(args.from_ladder or args.from_solver_ladder or args.from_body or args.from_reduced
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
            build_solver_ladder(sms, mdx)
        if "body" in legs:
            print("── prebuild: body ────────────────────────────────────────")
            build_body(sms)
        if "shapes" in legs:
            print("── prebuild: shapes (cuBLASDx microbenches) ──────────────")
            if mdx is None:
                print("  [skip] shapes needs MATHDX_ROOT (cuBLASDx).")
            else:
                result = run([sys.executable, "autotune.py", "--sm", str(sms),
                              "--build-only", "--build-jobs", str(args.build_jobs),
                              "--build-dir", str(cache_dir(sms))], cwd=BENCH_DIR)
                if result.returncode:
                    sys.exit("ERROR: shapes prebuild failed")
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
            if args.from_solver_ladder:
                solver_path = pathlib.Path(args.from_solver_ladder)
            else:
                solver_bin = build_solver_ladder(sms, mdx)
                solver_path = run_solver_ladder(
                    solver_bin, args.quick, sms, args.margin, sched=user_sched,
                    force=args.force, rounds=args.solver_rounds,
                    seed=args.solver_seed)
        else:
            binp = build_mega_sweep(sms, mdx, args.allow_no_mathdx)
            sweep_path = run_mega_sweep(
                binp, args.quick, sms, args.margin, sched=user_sched,
                force=args.force)
            sweep_text = sweep_path.read_text()
            solver_bin = build_solver_ladder(sms, mdx)
            solver_path = run_solver_ladder(
                solver_bin, args.quick, sms, args.margin, sched=user_sched,
                force=args.force, rounds=args.solver_rounds,
                seed=args.solver_seed)
        solver_text = (solver_path.read_text()
                       if solver_path is not None else None)
        new_defaults, n, solver_plans = regen_ladder(
            sweep_text, args.margin, sweep_path.name, sms, solver_text,
            solver_path.name if solver_path is not None else None)
        print(f"  regenerated ideal_sm{sms // 10} from {n} (dtype,op) groups; "
              f"fresh-input solver plans={solver_plans}")
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
            body_path = run_mega_sweep(
                binp, args.quick, sms, args.margin,
                prefix="body_dispatch_sweep", sched=user_sched,
                force=args.force)
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
            if args.force:
                cmd.append("--force")
            if run(cmd, cwd=BENCH_DIR).returncode:
                sys.exit("ERROR: shapes autotune failed")

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
            rtxt = run_isolated([str(binp), str(args.iters)], args.force)
            rtxt_path = BENCH_DIR / f"reduced_sweep_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            rtxt_path.write_text(
                "\n".join(provenance("reduced_sweep", sms, args.margin))
                + "\n" + rtxt)
        if rtxt:
            warn_jittery_rows(rtxt, args.margin, "reduced")
            rows, wins, mism = analyze_reduced(rtxt, args.margin)
            print(f"  {len(rows)} configs, reduced wins {len(wins)}, "
                  f"conservative-plan exceptions {len(mism)}")
            if mism:
                print("  ⚠️ explicit reduced wins remain outside the conservative plan")
            block = gen_reduced_block(rows, wins, mism, args.margin, rtxt_path.name)
            md = splice_results_md(RESULTS_MD.read_text(), block, "reduced")
            if args.dry_run:
                changed["reduced"] = show_diff(RESULTS_MD, md, "RESULTS.md")
            else:
                RESULTS_MD.write_text(md)
                print(f"  wrote {RESULTS_MD.relative_to(GLASS_DIR)}")

    # ── blas2 / rect (warp-vs-block picks → md report + header table) ──
    for leg, from_arg, builder, prefix, md_path, parse, note, regen in (
            ("blas2", args.from_blas2, build_blas2, "blas2_sweep",
             RESULTS_MD, tp.parse_blas2, _BLAS2_NOTE, regen_blas2_table),
            ("rect", args.from_rect, build_rect, "rect_sweep",
             RESULTS_MD, tp.parse_rect, _RECT_NOTE, regen_rect_table)):
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
            txt_path = run_mega_sweep(
                binp, args.quick, sms, args.margin, prefix,
                force=args.force)
            txt = txt_path.read_text()
        report_pick_leg(leg, txt, txt_path.name, md_path, args.margin,
                        parse, args.dry_run, changed, note)
        # header table: same marker-block model as the ladder (per-arch splice).
        if sms is None:
            print(f"  [skip] {leg} header table needs an explicit --sm "
                  "(the arch the capture came from).")
        else:
            new_text, n = regen(txt, args.margin, txt_path.name, sms)
            print(f"  regenerated {leg} table from {n} groups")
            if args.dry_run:
                changed[f"{leg}-table"] = show_diff(DEFAULTS, new_text,
                                                    DEFAULTS.name)
            else:
                DEFAULTS.write_text(new_text)
                print(f"  wrote {DEFAULTS.relative_to(GLASS_DIR)}")

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
            txt_path = run_mega_sweep(
                binp, args.quick, sms, args.margin, "solvers_sweep",
                sched=_SOLVERS_SCHED, force=args.force)
            txt = txt_path.read_text()
        if txt:
            warn_jittery_rows(txt, args.margin, "solvers")
            data = parse_solvers(txt)
            if not any(data.values()):
                print(f"  ⚠️ no NPROB=8192 rows parsed from {txt_path.name}; "
                      "nothing written.")
            else:
                print(f"  parsed: {len(data['bdsv_pcg'])} bdsv-vs-pcg, "
                      f"{len(data['spdsv'])} spd-solve, {len(data['eig'])} "
                      "syev/eig_clamp cells")
                block = gen_solvers_block(data, txt_path.name)
                existing = (RESULTS_MD.read_text() if RESULTS_MD.exists()
                            else "# measured results\n")
                md = splice_results_md(existing, block, "solvers")
                if args.dry_run:
                    changed["solvers"] = show_diff(RESULTS_MD, md, RESULTS_MD.name)
                else:
                    RESULTS_MD.write_text(md)
                    print(f"  wrote {RESULTS_MD.relative_to(GLASS_DIR)}")

    # ── figures ──
    if "figures" in legs:
        print("── figures ───────────────────────────────────────────────")
        sweep_for_fig = (str(pathlib.Path(args.from_ladder).resolve())
                         if args.from_ladder else None)
        if not sweep_for_fig:
            cands = sorted(glob.glob(str(BENCH_DIR / "mega_sweep_*.txt")))
            sweep_for_fig = cands[-1] if cands else None
        solver_for_fig = (str(pathlib.Path(args.from_solver_ladder).resolve())
                          if args.from_solver_ladder else None)
        if sweep_for_fig and not solver_for_fig:
            cands = sorted(glob.glob(str(BENCH_DIR / "solver_ladder_*.txt")))
            solver_for_fig = cands[-1] if cands else None
        if not sweep_for_fig:
            print("  [skip] no mega_sweep_*.txt to plot.")
        elif not solver_for_fig:
            print("  [skip] no solver_ladder_*.txt companion to plot destructive rows.")
        elif args.dry_run:
            print(f"  [dry-run] would render figures from "
                  f"{pathlib.Path(sweep_for_fig).name} + "
                  f"{pathlib.Path(solver_for_fig).name}")
        else:
            r = run([sys.executable, "export_sweep_figures.py", sweep_for_fig,
                     "--solver", solver_for_fig], cwd=BENCH_DIR)
            if r.returncode != 0:
                print("  ⚠️ figures leg failed; inspect the renderer error above "
                      "(matplotlib is one possible missing dependency). Tables are unaffected.")

    if args.dry_run:
        moved = [k for k, v in changed.items() if v]
        print(f"\n[dry run] tables that would change: {moved or 'none'}")
    print("\n==> done.")


if __name__ == "__main__":
    main()
