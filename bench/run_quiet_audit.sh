#!/usr/bin/env bash
# Complete audit timing/attestation sequence for a quiet sm_120 GPU.
#
# Prebuild first. Timing is capture-first: no dispatch header changes until
# every measured leg finishes. Regeneration then consumes immutable captures
# offline; commit the reviewed diffs and run ./test/run_gpu_proof.sh for the
# second receipt (the runner refuses dirty trees by design).
set -euo pipefail
cd "$(dirname "$0")/.."

RESUME=false
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=true
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--resume]" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
SMS=1200
ARCH=sm_120

echo "── quiet-GPU preflight ──────────────────────────────────"
nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits

echo "── prebuild every timed binary (hash-cached, compile-only) ─"
# Cold caches would otherwise burn quiet-window time on multi-GB MathDx
# compiles mid-sequence. These are no-ops when the caches are warm, so the
# quiet window itself stays execute-only. (Run this script's prebuild ahead
# of time on a busy box if you want: it performs no timed GPU work.)
python3 bench/tune.py --sm "$SMS" --prebuild --build-jobs 4
python3 bench/perf_sweeps.py --arch "$ARCH" --build-only
python3 bench/paper_sweeps.py --arch "$ARCH" --build-only

if $RESUME; then
    echo "── repair NVIDIA shard; carry prior completed shards ────"
    [[ -f test/gpu-proof.json ]] || {
        echo "FATAL: --resume requires test/gpu-proof.json" >&2
        exit 2
    }
    ./test/run_gpu_proof.sh --shards mathdx \
        --carry-from test/gpu-proof.json
else
    echo "── correctness receipt for measured source ──────────────"
    ./test/run_gpu_proof.sh
fi

echo "── audit-only and paper captures (no table writes) ───────"
python3 bench/perf_sweeps.py --arch "$ARCH" --profile overnight
python3 bench/paper_sweeps.py --arch "$ARCH" --reps 100

marker="$(mktemp)"
trap 'rm -f "$marker"' EXIT

echo "── tuner captures (no table writes) ──────────────────────"
python3 bench/tune.py --sm "$SMS" \
    --legs ladder,body,reduced,blas2,rect,solvers --dry-run

one_new_capture() {
    local pattern="$1"
    local -a files=()
    mapfile -t files < <(find bench -maxdepth 1 -type f -name "$pattern" \
        -newer "$marker" -print)
    if [[ "${#files[@]}" -ne 1 ]]; then
        echo "FATAL: expected one new $pattern capture, found ${#files[@]}" >&2
        return 1
    fi
    printf '%s' "${files[0]}"
}

ladder="$(one_new_capture 'mega_sweep_*.txt')"
body="$(one_new_capture 'body_dispatch_sweep_*.txt')"
reduced="$(one_new_capture 'reduced_sweep_*.txt')"
blas2="$(one_new_capture 'blas2_sweep_*.txt')"
rect="$(one_new_capture 'rect_sweep_*.txt')"
solvers="$(one_new_capture 'solvers_sweep_*.txt')"

echo "── shape capture + table write (last timed mutation) ─────"
python3 bench/tune.py --sm "$SMS" --legs shapes

echo "── offline regeneration from immutable captures ──────────"
python3 bench/tune.py --sm "$SMS" \
    --legs ladder,body,reduced,blas2,rect,solvers,figures \
    --from-ladder "$ladder" \
    --from-body "$body" \
    --from-reduced "$reduced" \
    --from-blas2 "$blas2" \
    --from-rect "$rect" \
    --from-solvers "$solvers"

echo
echo "Quiet audit complete. Review captures and generated diffs, COMMIT them,"
echo "then run ./test/run_gpu_proof.sh for the final receipt — the runner"
echo "refuses dirty trees because a dirty-tree receipt can never verify"
echo "against the commit that would carry it."
