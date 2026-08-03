#!/usr/bin/env python3
"""Build one canonical 50W ladder capture from the clean parts of two runs.

The 2026-08-02 50W run had its first three sections spoiled by a stray sweep
(see check_zombie_contamination.py); topup_50w_targeted.sh re-measured exactly
those three. tune.py --from-ladder reads whole sections, so this emits a single
file with each (NPROB,dtype) section taken from the run that measured it clean.

  ./merge_50w_sections.py                 # write mega_sweep_50W_merged.txt
  ./merge_50w_sections.py --selftest      # round-trip the 30W file, no writes

Section provenance is recorded as a comment above each section so the merge is
auditable from the capture alone.
"""
import argparse, pathlib, re, sys

HERE = pathlib.Path(__file__).parent
OLD = HERE / "50w_20260802_partial/mega_sweep_20260802_0756.txt"
REF30 = HERE / "jetson_localhost.localdomain_20260801_1226/mega_sweep_20260801_1226.txt"
OUT = HERE / "mega_sweep_50W_merged.txt"

HDR = re.compile(r"^#+ NPROB=(\d+)\s+reps=(\d+)\s+dtype=(f\d+)\s+#+\s*$")
# sections proven clean in the 08-02 run; the rest come from the recut
CLEAN_IN_OLD = {(1024, "f64"), (8192, "f32"), (8192, "f64")}


def split_sections(path):
    """-> [((nprob, dtype), header_line, [body lines])] in file order."""
    secs, cur = [], None
    for ln in path.read_text().splitlines():
        m = HDR.match(ln)
        if m:
            cur = ((int(m.group(1)), m.group(3)), ln, [])
            secs.append(cur)
        elif cur is not None:
            cur[2].append(ln)
    return secs


def selftest():
    secs = split_sections(REF30)
    keys = [s[0] for s in secs]
    assert len(secs) == 6, f"expected 6 sections, got {len(secs)}"
    assert len(set(keys)) == 6, f"duplicate sections: {keys}"
    # round-trip: header + body must reproduce the original byte-for-byte from
    # the first header onward (the preamble above section 1 is metadata only)
    orig = REF30.read_text().splitlines()
    first = next(i for i, l in enumerate(orig) if HDR.match(l))
    rebuilt = []
    for _, hdr, body in secs:
        rebuilt.append(hdr)
        rebuilt.extend(body)
    assert rebuilt == orig[first:], "round-trip mismatch"
    rows = {k: sum(1 for l in b if "||" in l) for k, _, b in secs}
    print(f"selftest OK: 6 sections, round-trip exact, rows/section={sorted(set(rows.values()))}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recut", type=pathlib.Path,
                    help="mega_sweep_50w_recut_*.txt from topup_50w_targeted.sh")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    recut = a.recut
    if recut is None:
        cands = sorted(HERE.glob("**/mega_sweep_50w_recut_*.txt"))
        if not cands:
            sys.exit("ERROR: no mega_sweep_50w_recut_*.txt found — pass --recut "
                     "(the targeted 50W top-up must land first).")
        recut = cands[-1]

    old = {k: (h, b) for k, h, b in split_sections(OLD)}
    new = {k: (h, b) for k, h, b in split_sections(recut)}
    want = [(64, "f32"), (64, "f64"), (1024, "f32"),
            (1024, "f64"), (8192, "f32"), (8192, "f64")]

    out, missing = ["# mega_sweep 50W (merged: clean sections per source, see "
                    "check_zombie_contamination.py)",
                    f"# recut source: {recut.name}", f"# kept source: {OLD.name}", ""], []
    for k in want:
        src, tag = ((old, OLD.name) if k in CLEAN_IN_OLD else (new, recut.name))
        if k not in src:
            missing.append((k, tag)); continue
        hdr, body = src[k]
        rows = sum(1 for l in body if "||" in l)
        if rows < 60:
            missing.append((k, f"{tag} (only {rows} rows)")); continue
        out += [f"# section source: {tag}", hdr, *body, ""]
        print(f"  NPROB={k[0]:<5d} {k[1]}  <- {tag}  ({rows} rows)")
    if missing:
        sys.exit("ERROR: missing/short sections: " +
                 ", ".join(f"NPROB={k[0]} {k[1]} from {t}" for k, t in missing))
    OUT.write_text("\n".join(out) + "\n")
    print(f"\nwrote {OUT}  ({len(want)} sections)")
    print("next: cd ~/Desktop/GLASS && .venv/bin/python bench/tune.py --legs ladder "
          f"--from-ladder {OUT} --sm 870 --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
