#!/usr/bin/env bash
# Run the GLASS test suite with every kernel launch wrapped in compute-sanitizer.
#
# Each test binary invocation (see run_op in conftest.py) is prefixed via
# GLASS_RUN_PREFIX with `compute-sanitizer`. Sanitizer diagnostics go to stderr;
# --error-exitcode=1 makes any finding (racecheck/memcheck/etc.) return nonzero,
# which run_op already turns into a failing test whose message carries the report.
#
# This is SLOW: compute-sanitizer adds seconds of startup per launch and the
# suite launches kernels thousands of times. Narrow it down with pytest args,
# e.g. only the L3 gemm race hunt:
#
#   TOOL=racecheck ./test/run_sanitizer.sh test/test_l3.py -k gemm_rt
#   ./test/run_sanitizer.sh test/test_l3.py::test_gemm_rt -k cg
#
# Usage:  ./test/run_sanitizer.sh [pytest args...]
# Env:    TOOL=memcheck|racecheck|initcheck|synccheck   (default: memcheck)
#         PY=path/to/python                             (default: .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
TOOL="${TOOL:-memcheck}"

command -v compute-sanitizer >/dev/null 2>&1 || {
    echo "compute-sanitizer not found on PATH (expected under \$CUDA_HOME/bin)." >&2
    exit 1
}

# Wrap each binary invocation. --error-exitcode=1 so a finding fails the test;
# --exitcode-on-non-clean-output false unavailable across versions, so we rely
# on the child's own return path staying 0 when clean.
export GLASS_RUN_PREFIX="compute-sanitizer --tool ${TOOL} --error-exitcode 1"

echo "compute-sanitizer --tool ${TOOL} wrapping every kernel launch"
echo "GLASS_RUN_PREFIX=${GLASS_RUN_PREFIX}"
echo

# -p no:randomly / no xdist: sanitizer needs serial, deterministic execution.
# -x stops at the first offending launch so its report isn't buried.
exec "$PY" -m pytest test/ -q -p no:cacheprovider "$@"
