#!/usr/bin/env python3
"""Check the declared behavioral obligations (test/coverage-obligations.json)
have passing evidence in collection and the signed receipt. This is an
obligation-presence check, not line or semantic coverage."""

import argparse
import fnmatch
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def collected_nodes() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.startswith("test/") and "::" in line}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=pathlib.Path, default=ROOT / "test/coverage-obligations.json")
    parser.add_argument("--receipt", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    nodes = collected_nodes()
    passed: set[str] | None = None
    node_shards: dict[str, str] = {}
    if args.receipt:
        receipt = json.loads(args.receipt.read_text())
        passed = {test["node_id"] for test in receipt["tests"] if test["outcome"] == "passed"}
        for shard in receipt["shards"]:
            for node in shard["node_ids"]:
                node_shards[node] = shard["name"]

    ok = 0
    failures: list[str] = []
    for obligation in policy["obligations"]:
        obligation_ok = True
        for pattern in obligation["evidence"]:
            matches = {node for node in nodes if fnmatch.fnmatchcase(node, pattern)}
            if not matches:
                failures.append(f"{obligation['id']}: no collected test matches {pattern}")
                obligation_ok = False
                continue
            if passed is not None:
                passed_matches = matches & passed
                if not passed_matches:
                    failures.append(f"{obligation['id']}: no passing receipted test matches {pattern}")
                    obligation_ok = False
                    continue
                expected = obligation["shard"]
                if expected != "mixed" and not any(node_shards.get(node) == expected for node in passed_matches):
                    failures.append(f"{obligation['id']}: passing evidence is not in shard {expected}")
                    obligation_ok = False
        if obligation_ok:
            ok += 1

    total = len(policy["obligations"])
    pct = 100.0 * ok / max(1, total)
    print(f"required correctness obligations: {ok}/{total} ({pct:.1f}%)")
    for failure in failures:
        print(f"  {failure}")
    if args.json:
        args.json.write_text(json.dumps({
            "schemaVersion": 1,
            "label": "correctness obligations",
            "message": f"{ok}/{total} ({pct:.0f}%)",
            "color": "brightgreen" if ok == total else "red",
        }) + "\n")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
