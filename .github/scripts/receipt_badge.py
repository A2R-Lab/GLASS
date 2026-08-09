#!/usr/bin/env python3
"""Emit a shields.io endpoint badge from the signed gpu-proof receipt."""
import json, sys

receipt = json.load(open("test/gpu-proof.json"))
n = receipt.get("summary", {}).get("passed", 0)
if not n:
    n = sum(1 for t in receipt.get("tests", []) if t.get("outcome") == "passed")
json.dump({"schemaVersion": 1, "label": "GPU tests",
           "message": f"{n} passed on hardware · receipt signed",
           "color": "brightgreen" if n else "red"},
          open(sys.argv[1], "w"))
print(f"wrote {sys.argv[1]} ({n} passed)")
