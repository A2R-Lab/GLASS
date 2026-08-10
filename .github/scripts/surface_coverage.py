#!/usr/bin/env python3
"""API-surface coverage: % of public documented entry points exercised by test/cuda/.

Honest metric for a CUDA header library (device-code line coverage does not
exist: nvcc emits no instrumentation for __device__ code, and instrumenting
the Python drivers would measure the harness, not the library). Instead we
enumerate the public surface — every doc-commented function in the library
headers — and check which names are referenced by at least one GPU test TU.

Usage: python .github/scripts/surface_coverage.py [--json OUT] [--list-gaps]
Exit 0 always (reporting tool); the number lands in a shields.io endpoint
badge JSON when --json is given.
"""
import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADERS = [*ROOT.glob("glass*.cuh"), *(ROOT / "src").rglob("*.cuh")]
TESTS = [*(ROOT / "test" / "cuda").glob("*.cu"),
         *(ROOT / "examples").glob("*.cu")]  # examples run on GPU via test_examples.py

# A public entry point = a `/** ... */` doc comment whose following declaration
# names a function. Captures free functions and namespaced ones alike.
_FN = re.compile(
    r"/\*\*.*?\*/\s*(?:template\s*<[^;{]*?>\s*)?"
    r"(?:__device__|__host__|inline|constexpr|static|\s)*[\w:<>,*&\s]+?"
    r"\b([a-z_][a-zA-Z0-9_]*)\s*\(", re.S)
_SKIP = {"if", "for", "while", "switch", "return", "sizeof", "defined"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--list-gaps", action="store_true")
    args = ap.parse_args()

    surface = {}  # name -> first header
    for h in HEADERS:
        for m in _FN.finditer(h.read_text(errors="ignore")):
            n = m.group(1)
            if n not in _SKIP and not n.endswith("_impl") and not n.startswith("_"):
                surface.setdefault(n, h.relative_to(ROOT))
    test_text = "\n".join(t.read_text(errors="ignore") for t in TESTS)
    covered = {n for n in surface if re.search(rf"\b{re.escape(n)}\b", test_text)}
    gaps = sorted(set(surface) - covered)
    pct = 100.0 * len(covered) / max(1, len(surface))
    print(f"surface: {len(surface)} public entry points, "
          f"{len(covered)} exercised by test/cuda + examples -> {pct:.1f}%")
    if args.list_gaps and gaps:
        print("uncovered:")
        for n in gaps:
            print(f"  {n:32s} ({surface[n]})")
    if args.json:
        color = "brightgreen" if pct >= 100 else "red"  # 100% is the contract (ruled 2026-08-06)
        pathlib.Path(args.json).write_text(json.dumps({
            "schemaVersion": 1, "label": "API surface tested",
            "message": f"{len(covered)}/{len(surface)} ({pct:.0f}%)",
            "color": color}))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
