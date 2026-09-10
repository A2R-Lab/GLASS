#!/usr/bin/env python3
"""Compatibility entry point for the overload-aware API contract checker."""

import os
import pathlib
import sys


SCRIPT = pathlib.Path(__file__).with_name("api_contract_coverage.py")

print(
    "surface_coverage.py is deprecated; using api_contract_coverage.py",
    file=sys.stderr,
)
os.execv(sys.executable, [sys.executable, str(SCRIPT), *sys.argv[1:]])
