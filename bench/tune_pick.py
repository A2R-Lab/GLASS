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
#   "<op>  N=<N> | BLOCK ... | WARP ... || block tb<TB>=<ns> warp w<WPB>=<ns> [nv=<ns>] -> ..."
_ROW_RE = re.compile(
    r"^(dot|gemv|gemm|chol|trsv|posv)\s+N=(\d+)\b.*\|\|\s*"
    r"block\s+tb\d+=([\d.]+)\s+warp\s+w\d+=([\d.]+)(?:\s+nv=([\d.]+))?")


def parse_mega_sweep(text, nprob=8192):
    """``(dtype, op, N) -> {block, warp[, nvidia]}`` raw ns/problem at ``nprob``.

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
                d["nvidia"] = float(m.group(5))
            data[(dtype, op, N)] = d
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
    assert pick({"nvidia": 50}, 0.05, {"nvidia"}) == "nvidia", "only nvidia measured"
    assert pick({"block": 0, "warp": None}, 0.05) is None, "nothing valid"
    assert pick({"serial": 1.0, "reduced": 0.90}, 0.05, {"reduced"}) == "reduced", \
        "reduced 10% faster → wins"
    assert pick({"serial": 1.0, "reduced": 0.97}, 0.05, {"reduced"}) == "serial", \
        "reduced 3% faster → inside margin → serial"
    # verdict noise floor.
    w, note = verdict({"simt": 0.02, "cublasdx": 0.01}, 0.05, {"cublasdx"}, noise_floor=0.05)
    assert w == "simt" and "noise floor" in note, note
    print("tune_pick self-test OK")
