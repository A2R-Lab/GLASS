#!/usr/bin/env python3
"""Shared tie-margin + sweep parsers for the unified autotuner (bench/tune.py).

This is the *one* place the noise margin lives. Every defaults table GLASS ships
— the warp/block/nvidia ladder (``glass-defaults.cuh``), the per-shape
cuBLASDx-vs-SIMT table (``src/nvidia/tuning_table.cuh``), and the
serial-vs-reduced picker (``suggested_use_reduced<>``) — routes its verdict
through :func:`pick` so none of them bakes sub-noise jitter, and so a pure-noise
re-run reproduces the same table.

`pick` encodes the project rule: **a higher-complexity / dependency-carrying
impl wins only if it beats the best dependency-free impl by more than the
margin.** Ties go to the simpler impl. That single rule subsumes all three
legacy decisions:

* ladder:  ``dependency={"nvidia"}`` — block/warp (pure SIMT, no MathDx) is the
           incumbent; nvidia must clear the margin to be chosen.
* shapes:  ``dependency={"cublasdx"}`` — SIMT is the incumbent (autotune.py's
           original pairwise ±margin rule, generalized).
* reduced: ``dependency={"reduced"}`` — serial ``gemm`` is the incumbent.
"""
import re


def pick(timings, margin=0.05, dependency=()):
    """Return the winning impl name from ``timings`` (name -> time, lower=better).

    Entries that are ``None`` or non-positive (failed / un-measured legs) are
    ignored. The cheapest **dependency-free** impl is the incumbent; an impl in
    ``dependency`` is chosen only if it beats that incumbent by *more* than
    ``margin`` (fractional, e.g. 0.05 = 5%). With no dependency-free impl
    measured, the cheapest measured impl wins. Returns ``None`` if nothing was
    measured.

    The rule is intentionally asymmetric: it makes near-ties resolve to the
    simpler / no-dependency code path, which is both the safer default (always
    launchable, no MathDx) and what keeps regenerated tables stable across the
    timer noise of a re-run.
    """
    valid = {k: float(v) for k, v in timings.items() if v is not None and v > 0}
    if not valid:
        return None
    dep = set(dependency)
    simple = {k: v for k, v in valid.items() if k not in dep}
    if not simple:
        # Only dependency impls were measured — cheapest of them.
        return min(valid, key=valid.get)
    base_name = min(simple, key=simple.get)
    base = simple[base_name]
    deps = {k: v for k, v in valid.items() if k in dep}
    if deps:
        best_dep = min(deps, key=deps.get)
        if deps[best_dep] < base * (1.0 - margin):
            return best_dep
    return base_name


def verdict(timings, margin=0.05, dependency=(), noise_floor=0.0):
    """Like :func:`pick`, but also return a human note explaining the call.

    Returns ``(winner_name, note)``. ``noise_floor`` (in the same time unit as
    ``timings``) forces the incumbent / first-dependency-free impl when *every*
    measured leg is below the floor — sub-granularity timings can't be trusted
    to resolve a margin.
    """
    valid = {k: float(v) for k, v in timings.items() if v is not None and v > 0}
    if not valid:
        return None, "no measurement"
    if noise_floor and all(v < noise_floor for v in valid.values()):
        dep = set(dependency)
        simple = [k for k in valid if k not in dep]
        fallback = min(simple, key=valid.get) if simple else min(valid, key=valid.get)
        return fallback, (f"all legs < {noise_floor:g} noise floor "
                          f"({', '.join(f'{k}={v:.3f}' for k, v in valid.items())}) "
                          f"→ {fallback}")
    win = pick(valid, margin, dependency)
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    best, second = ordered[0], ordered[1] if len(ordered) > 1 else None
    if second is None:
        note = f"{win} only impl measured ({best[1]:.3f})"
    elif win == ordered[0][0]:
        gap = (second[1] / best[1] - 1.0) * 100.0
        note = f"{win} wins ({best[1]:.3f} vs {second[0]} {second[1]:.3f}, {gap:.1f}%)"
    else:
        win_t = valid[win]
        gap = (win_t / best[1] - 1.0) * 100.0
        note = (f"{win} kept ({win_t:.3f}); {best[0]} faster by {gap:.1f}% "
                f"but inside ±{margin*100:.0f}% margin")
    return win, note


# ─── parsers ────────────────────────────────────────────────────────────────

LADDER_OPS = ("dot", "gemv", "gemm", "chol", "trsv", "posv")

_HDR_RE = re.compile(r"NPROB=(\d+).*dtype=(f32|f64)")
# Raw per-backend ns from a mega_sweep row:
#   "<op>  N=<N> | BLOCK ... | WARP ... [| THREAD ...] || block tb<TB>=<ns>
#    warp w<WPB>=<ns> [thread t<TPB>=<ns>] [nv=<ns>] -> ..."
# The thread group is OPTIONAL on purpose: it is absent both from archived sweep
# .txt files (every run before the tier existed — `tune.py --from-ladder` replays
# them) and from live rows at N>16, where the harness skips the tier rather than
# instantiate an absurd per-thread T[N*N]. Keep it optional or old sweeps stop
# parsing and regen silently drops every op.
_ROW_RE = re.compile(
    r"^(dot|gemv|gemm|chol|trsv|posv)\s+N=(\d+)\b.*\|\|\s*"
    r"block\s+tb\d+=([\d.]+)\s+warp\s+w\d+=([\d.]+)"
    r"(?:\s+thread\s+t\d+=([\d.]+))?(?:\s+nv=([\d.]+))?")


def parse_mega_sweep(text, nprob=8192):
    """``(dtype, op, N) -> {block, warp[, thread][, nvidia]}`` raw ns/problem at ``nprob``.

    Reads the raw per-backend numbers (NOT the harness's ``-> WINNER`` verdict,
    which is a bare argmin with no margin) so :func:`pick` can re-decide the
    winner under the shared margin.
    """
    data, dtype, cur = {}, None, None
    for line in text.splitlines():
        if line.startswith("####"):
            m = _HDR_RE.search(line)
            if m:
                cur, dtype = int(m.group(1)), m.group(2)
            continue
        if cur != nprob:
            continue
        m = _ROW_RE.match(line.strip())
        if m:
            op, N = m.group(1), int(m.group(2))
            d = {"block": float(m.group(3)), "warp": float(m.group(4))}
            if m.group(5):
                d["thread"] = float(m.group(5))   # absent in pre-tier sweeps and at N>16
            if m.group(6):
                d["nvidia"] = float(m.group(6))
            data[(dtype, op, N)] = d
    return data


BLAS2_OPS = ("syrk", "syr2k", "ldlt", "ldltsv", "inv", "trmv", "ger")

# Raw per-backend ns from a bench_blas2 row (same grammar as the mega sweep, but
# 2-way: the warp leg is absent for the block-only ops inv/trmv/ger):
#   "<op>  N=<N> | BLOCK ... [| WARP ...] || block tb<TB>=<ns> [warp w<WPB>=<ns>] -> ..."
_B2_ROW_RE = re.compile(
    r"^(syr2k|syrk|ldltsv|ldlt|inv|trmv|ger)\s+N=(\d+)\b.*\|\|\s*"
    r"block\s+tb\d+=([\d.]+)(?:\s+warp\s+w\d+=([\d.]+))?")


def parse_blas2(text, nprob=8192):
    """``(dtype, op, N) -> {block[, warp]}`` raw ns/problem at ``nprob``.

    Reads the raw per-backend numbers (NOT the harness's ``-> WINNER`` verdict,
    a bare argmin) so :func:`pick`/:func:`verdict` re-decide under the shared
    margin. Ops without a warp:: variant yield a block-only dict.
    """
    data, dtype, cur = {}, None, None
    for line in text.splitlines():
        if line.startswith("####"):
            m = _HDR_RE.search(line)
            if m:
                cur, dtype = int(m.group(1)), m.group(2)
            continue
        if cur != nprob:
            continue
        m = _B2_ROW_RE.match(line.strip())
        if m:
            op, N = m.group(1), int(m.group(2))
            d = {"block": float(m.group(3))}
            if m.group(4):
                d["warp"] = float(m.group(4))
            data[(dtype, op, N)] = d
    return data


# Raw per-backend ns from a bench_rect row. Shapes are rectangular, so rows are
# keyed by the full dim tuple: gemv "M=<M> N=<N>", gemm "M=<M> K=<K> N=<N>".
_RECT_GEMV_RE = re.compile(
    r"^gemv\s+M=(\d+)\s+N=(\d+)\b.*\|\|\s*block\s+tb\d+=([\d.]+)\s+warp\s+w\d+=([\d.]+)")
_RECT_GEMM_RE = re.compile(
    r"^gemm\s+M=(\d+)\s+K=(\d+)\s+N=(\d+)\b.*\|\|\s*block\s+tb\d+=([\d.]+)\s+warp\s+w\d+=([\d.]+)")


def parse_rect(text, nprob=8192):
    """``(dtype, op, dims) -> {block, warp}`` raw ns/problem at ``nprob``.

    ``dims`` is ``(M, N)`` for gemv and ``(M, K, N)`` for gemm (C is MxN,
    contraction K).
    """
    data, dtype, cur = {}, None, None
    for line in text.splitlines():
        if line.startswith("####"):
            m = _HDR_RE.search(line)
            if m:
                cur, dtype = int(m.group(1)), m.group(2)
            continue
        if cur != nprob:
            continue
        s = line.strip()
        m = _RECT_GEMV_RE.match(s)
        if m:
            data[(dtype, "gemv", (int(m.group(1)), int(m.group(2))))] = \
                {"block": float(m.group(3)), "warp": float(m.group(4))}
            continue
        m = _RECT_GEMM_RE.match(s)
        if m:
            data[(dtype, "gemm", (int(m.group(1)), int(m.group(2)), int(m.group(3))))] = \
                {"block": float(m.group(4)), "warp": float(m.group(5))}
    return data


_REDUCED_RE = re.compile(
    r"^REDUCED\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)")


def parse_reduced(text):
    """Parse bench_reduced rows → list of dicts with M,N,K,blockDim,n_out,serial,reduced."""
    rows = []
    for line in text.splitlines():
        m = _REDUCED_RE.match(line.strip())
        if m:
            M, N, K, bd, n_out = (int(m.group(i)) for i in range(1, 6))
            rows.append(dict(M=M, N=N, K=K, blockDim=bd, n_out=n_out,
                             serial=float(m.group(6)), reduced=float(m.group(7))))
    return rows


# ─── self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # pick(): dependency must clear the margin.
    assert pick({"block": 100, "warp": 110, "nvidia": 99}, 0.05, {"nvidia"}) == "block", \
        "nvidia 1% faster than block → inside margin → block"
    assert pick({"block": 150, "warp": 110, "nvidia": 99}, 0.05, {"nvidia"}) == "nvidia", \
        "nvidia 10% faster than warp → clears margin"
    assert pick({"block": 100, "warp": 99}, 0.05, {"nvidia"}) == "warp", \
        "warp/block both simple → cheapest"
    # thread tier: dependency-free SIMT alongside warp/block — cheapest of the
    # three wins with no margin (a structural launch change, like warp, is not
    # margin-gated); nvidia still must clear the margin over the SIMT incumbent.
    assert pick({"block": 100, "warp": 90, "thread": 60}, 0.05, {"nvidia"}) == "thread", \
        "thread cheapest SIMT tier → wins (low-DOF packing)"
    assert pick({"block": 100, "warp": 90, "thread": 91}, 0.05, {"nvidia"}) == "warp", \
        "thread not cheapest → warp keeps it"
    assert pick({"block": 100, "warp": 90, "thread": 60, "nvidia": 59}, 0.05, {"nvidia"}) == "thread", \
        "nvidia inside margin of the thread incumbent → thread"
    assert pick({"nvidia": 50}, 0.05, {"nvidia"}) == "nvidia", "only nvidia measured"
    assert pick({"block": 0, "warp": None}, 0.05) is None, "nothing valid"
    assert pick({"serial": 1.0, "reduced": 0.90}, 0.05, {"reduced"}) == "reduced", \
        "reduced 10% faster → wins"
    assert pick({"serial": 1.0, "reduced": 0.97}, 0.05, {"reduced"}) == "serial", \
        "reduced 3% faster → inside margin → serial"
    # verdict noise floor.
    w, note = verdict({"simt": 0.02, "cublasdx": 0.01}, 0.05, {"cublasdx"}, noise_floor=0.05)
    assert w == "simt" and "noise floor" in note, note
    # parse_mega_sweep(): thread column optional — present at low N, absent at
    # high N and in pre-tier archived sweeps.
    _mg = "\n".join([
        "################ NPROB=8192  reps=250  dtype=f32 ################",
        "dot   N=4   | BLOCK  tb32=1.00 | WARP  w1=0.80 | THREAD  t256=0.30"
        "  || block tb32=1.00  warp w1=0.80  thread t256=0.30  nv=2.00 -> THREAD",
        "posv  N=64  | BLOCK  tb32=5.00 | WARP  w1=6.00"
        "  || block tb32=5.00  warp w1=6.00  nv=3.00 -> NV",
    ])
    _mc = parse_mega_sweep(_mg)
    assert _mc[("f32", "dot", 4)] == {"block": 1.0, "warp": 0.8, "thread": 0.3, "nvidia": 2.0}, _mc
    assert _mc[("f32", "posv", 64)] == {"block": 5.0, "warp": 6.0, "nvidia": 3.0}, \
        "thread absent at high N parses as block/warp/nvidia only"
    assert pick(_mc[("f32", "dot", 4)], 0.05, {"nvidia"}) == "thread", "dot N=4 → thread wins"
    # parse_blas2(): 2-way rows, warp leg optional, ldlt/ldltsv disambiguation.
    _b2 = "\n".join([
        "################ NPROB=8192  reps=250  dtype=f32 ################",
        "syrk   N=8   | BLOCK  tb32=1.10  tb64=1.00  | WARP  w1=0.80  w2=0.90"
        "  || block tb64=1.00  warp w1=0.80  -> WARP (1.25x)",
        "ldltsv N=16  | BLOCK  tb32=3.00  | WARP  w1=2.00"
        "  || block tb32=3.00  warp w1=2.00  -> WARP (1.50x)",
        "ldlt   N=16  | BLOCK  tb32=2.50  | WARP  w1=2.40"
        "  || block tb32=2.50  warp w1=2.40  -> WARP (1.04x)",
        "inv    N=8   | BLOCK  tb32=5.00  tb64=4.00  || block tb64=4.00  -> BLOCK (1.00x)",
        "################ NPROB=64  reps=1000  dtype=f32 ################",
        "ger    N=8   | BLOCK  tb32=9.00  || block tb32=9.00  -> BLOCK (1.00x)",
    ])
    _c = parse_blas2(_b2)
    assert _c[("f32", "syrk", 8)] == {"block": 1.00, "warp": 0.80}, _c
    assert _c[("f32", "ldltsv", 16)] == {"block": 3.00, "warp": 2.00}, _c
    assert _c[("f32", "ldlt", 16)] == {"block": 2.50, "warp": 2.40}, _c
    assert _c[("f32", "inv", 8)] == {"block": 4.00}, "block-only op parses without warp"
    assert ("f32", "ger", 8) not in _c, "NPROB=64 section must be filtered out"
    # parse_rect(): gemv (M,N) + gemm (M,K,N) keys.
    _rc = parse_rect("\n".join([
        "################ NPROB=8192  reps=250  dtype=f64 ################",
        "gemv  M=64  N=8   | BLOCK  tb32=1.5  | WARP  w1=1.2"
        "  || block tb32=1.50  warp w1=1.20  -> WARP (1.25x)",
        "gemm  M=32  K=8   N=32  | BLOCK  tb32=2.5  | WARP  w1=3.0"
        "  || block tb32=2.50  warp w1=3.00  -> BLOCK (1.20x)",
    ]))
    assert _rc[("f64", "gemv", (64, 8))] == {"block": 1.50, "warp": 1.20}, _rc
    assert _rc[("f64", "gemm", (32, 8, 32))] == {"block": 2.50, "warp": 3.00}, _rc
    print("tune_pick self-test OK")
