#!/usr/bin/env python3
"""Verify every receipt shard fingerprints the API surface its tests CALL.

Each shard in test/run_gpu_proof.sh carries a NARROW fingerprint (SHARD_PATHS
plus its own test files/drivers). Textual #include closure cannot be the
invariant — every driver includes the glass.cuh umbrella, so each TU compiles
the whole library. The property that keeps DAG-scoped carry-forward sound is
behavioral: a header that DEFINES an operation a shard's tests call must be
inside that shard's fingerprint, or an edit to it could leave a stale shard
carrying forward silently (this exact hole shipped once: the vector shard
called glass::cgrps:: ops via test_api_vector.cu without fingerprinting
src/cgrps).

Mechanization: harvest every glass::-qualified call token from each shard's
CUDA drivers, resolve it to its defining header(s) through the generated
overload manifest test/api-contracts.json, and require those headers to fall
under the shard's fingerprint paths. Runs GPU-less; wired into the
contract-coverage CI job.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "test" / "run_gpu_proof.sh"
MANIFEST = ROOT / "test" / "api-contracts.json"
CALL_RE = re.compile(r"\bglass((?:\s*::\s*\w+)+)\s*[(<]")


def shard_tables() -> dict[str, dict[str, str]]:
    """Evaluate the runner's assignment block in bash and dump the arrays."""
    lines = RUNNER.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ROOTS_BASE="))
    end = next(i for i, l in enumerate(lines) if l.startswith("ALL_SHARDS="))
    prog = "\n".join(lines[start:end + 1]) + """
for s in $ALL_SHARDS; do
    printf '%s\\x1f%s\\x1f%s\\x1f%s\\x1e' \
        "$s" "${SHARD_PATHS[$s]}" "${SHARD_FILES[$s]}" "${SHARD_DRIVERS[$s]}"
done
"""
    out = subprocess.check_output(["bash", "-c", prog], text=True)
    tables: dict[str, dict[str, str]] = {}
    for rec in filter(None, out.split("\x1e")):
        name, paths, files, drivers = rec.split("\x1f")
        tables[name] = {"paths": paths, "files": files, "drivers": drivers}
    return tables


def manifest_maps() -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    contracts = json.loads(MANIFEST.read_text())["contracts"]
    by_surface: dict[tuple[str, str], set[str]] = {}
    by_name: dict[str, set[str]] = {}
    for c in contracts:
        by_surface.setdefault((c["surface"], c["name"]), set()).add(c["file"])
        by_name.setdefault(c["name"], set()).add(c["file"])
    return by_surface, by_name


def in_scope(rel: str, scope_paths: list[str]) -> bool:
    return any(rel == p or rel.startswith(p.rstrip("/") + "/")
               for p in scope_paths)


def _tier_filter(surface: str, files: set[str]) -> set[str]:
    """Restrict candidate defining headers to the call's backend directory.

    The measured-default bare face (and the SIMT tiers) never route into the
    nvidia or cgrps backends, and vice versa — so a glass::block::dot call
    can only be satisfied by non-backend headers even when the name also
    exists as glass::nvidia::block::dot.
    """
    if surface.startswith("glass::nvidia"):
        return {f for f in files
                if f.startswith("src/nvidia") or f == "glass-nvidia.cuh"}
    if surface.startswith("glass::cgrps"):
        return {f for f in files if f.startswith("src/cgrps")}
    return {f for f in files
            if not f.startswith(("src/nvidia", "src/cgrps"))
            and f != "glass-nvidia.cuh"}


def resolve_files(surface: str, name: str,
                  by_surface: dict[tuple[str, str], set[str]],
                  by_name: dict[str, set[str]]) -> set[str]:
    if name not in by_name:
        return set()  # type/enum/detail token — not a public overload
    if surface == "glass::":
        # The bare face dispatches per (op, size, dtype) into ANY SIMT tier
        # body — behaviorally the shard depends on every candidate body
        # header, not just dispatch.cuh where the face is declared.
        return _tier_filter(surface, by_name[name])
    if (surface, name) in by_surface:
        return by_surface[(surface, name)]
    # Call token absent from the manifest under this exact surface (e.g. a
    # nested tier spelling): fall back to all same-backend definitions.
    return _tier_filter(surface, by_name[name])


def main() -> int:
    by_surface, by_name = manifest_maps()
    failures: list[str] = []
    for shard, t in shard_tables().items():
        # Mirror the runner: narrow fingerprint = SHARD_PATHS + own files/drivers.
        scope = [p for p in t["paths"].split(",") if p]
        scope += t["files"].split() + t["drivers"].split()
        needed: dict[str, set[str]] = {}  # header -> call tokens that need it
        for drv in t["drivers"].split():
            src = (ROOT / drv).read_text(errors="replace")
            for m in CALL_RE.finditer(src):
                parts = re.sub(r"\s", "", m.group(1)).lstrip(":").split("::")
                name = parts[-1]
                surface = "glass::" + "".join(p + "::" for p in parts[:-1])
                for f in resolve_files(surface, name, by_surface, by_name):
                    needed.setdefault(f, set()).add(surface + name)
        for f in sorted(needed):
            if not in_scope(f, scope):
                calls = ", ".join(sorted(needed[f])[:4])
                failures.append(
                    f"shard '{shard}' calls {calls} defined in {f}, "
                    f"which its fingerprint does not cover")
    if failures:
        print("shard-scope closure FAILED:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        print("fix: extend SHARD_PATHS in test/run_gpu_proof.sh", file=sys.stderr)
        return 1
    print("shard-scope closure OK: every called overload's defining header "
          "is fingerprinted by its shard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
