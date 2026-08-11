#!/usr/bin/env bash
# Run the GPU suite as FIVE schema-2 shards and merge into one signed receipt
# (test/gpu-proof.json). Shards carry DAG-scoped fingerprints so an untouched
# family can carry forward from the last committed receipt instead of re-running:
#
#   core     — L1/L2/L3 + roots + cgrps (the trunk; everything depends on it)
#   solvers  — banded/pcg/bdsv + box_qp          (scope = its files + core's)
#   robotics — spatial/lie/proj/geom/est         (scope = its files + core's)
#   mathdx   — src/nvidia + glass-nvidia.cuh     (scope = its files + core's)
#   cross    — examples + trailing-sync (deliberately whole-tree scope: these
#              tests span every family, so they only carry when NOTHING changed)
#
# Usage:
#   ./test/run_gpu_proof.sh                       # run all five shards
#   ./test/run_gpu_proof.sh --shards solvers      # run some; carry the rest
#                                                 # from the committed receipt
#   (extra args after -- go to pytest)
#
# Merge always attempts --carry-from the last committed test/gpu-proof.json
# when a subset ran: freshly-run shards win; absent shards graft in iff their
# narrow fingerprint recomputes identical AND the old commit is an ancestor
# (schema-2 rules — a stale family can never ride along silently). CI verifies
# carried shards under the policy in verify-gpu-proof.yml (allow_carried +
# carried_max_age_days); see pytest-gpu-proof docs/sharding.md.
#
# The 2 permanent documented skips (test_getrf n=1) are pinned as an EXACT set
# by CI via test/expected_skips.txt. The signer must be the human KEYHOLDER
# (--gpu-proof-github-user), not the org.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
$PY -m pip install -q -r test/requirements.txt

ROOTS_CORE="glass.cuh,glass-cgrps.cuh,glass-defaults.cuh,glass-dispatch.cuh"
SRC_CORE="src/base/L1,src/base/L2,src/base/L3,src/base/barrier.cuh,src/base/dispatch.cuh,src/base/flags.cuh,src/cgrps"
TEST_CORE="test/conftest.py,test/cuda,test/expected_skips.txt"
CORE_SCOPE="$ROOTS_CORE,$SRC_CORE,$TEST_CORE"

declare -A SHARD_FILES SHARD_PATHS
SHARD_FILES[core]="test/test_l1.py test/test_l1_round2.py test/test_l2.py test/test_l3.py \
 test/test_thread.py test/test_warp.py test/test_defaults.py test/test_dispatch.py \
 test/test_iamax.py test/test_symmetrize.py test/test_symm_rot.py test/test_syrk.py \
 test/test_fused.py test/test_factor_check.py test/test_getrf.py test/test_ldlt.py \
 test/test_trsv.py test/test_posv.py test/test_syev.py test/test_solve.py \
 test/test_congruence.py test/test_tensor.py test/test_reduced.py test/test_reduced_blas.py \
 test/test_base_f64.py test/test_block_access.py"
SHARD_PATHS[core]="$CORE_SCOPE"

SHARD_FILES[solvers]="test/test_banded.py test/test_bdsv.py test/test_pcg.py test/test_qp.py"
SHARD_PATHS[solvers]="$CORE_SCOPE,src/base/banded,src/base/pcg,src/internal/box_qp.cuh"

SHARD_FILES[robotics]="test/test_robotics.py"
SHARD_PATHS[robotics]="$CORE_SCOPE,src/base/spatial,src/base/lie,src/base/proj,src/base/geom,src/base/est"

SHARD_FILES[mathdx]="test/test_nvidia_dispatch.py test/test_nvidia_f64.py"
SHARD_PATHS[mathdx]="$CORE_SCOPE,src/nvidia,glass-nvidia.cuh"

SHARD_FILES[cross]="test/test_examples.py test/test_trailing_sync.py"
SHARD_PATHS[cross]="$ROOTS_CORE,glass-nvidia.cuh,src,test/conftest.py,test/cuda,test/expected_skips.txt,examples"

ALL_SHARDS="core solvers robotics mathdx cross"
RUN_SHARDS="$ALL_SHARDS"
if [[ "${1:-}" == "--shards" ]]; then
    RUN_SHARDS="${2//,/ }"; shift 2
fi
[[ "${1:-}" == "--" ]] && shift

mkdir -p test/receipts
FRESH=()
for s in $RUN_SHARDS; do
    echo "── shard: $s ──────────────────────────────────────────"
    # A shard's own .py test files belong in its fingerprint: without them an
    # edited test could carry forward stale results (hole closed 2026-08-11).
    FP_PATHS="${SHARD_PATHS[$s]},$(echo ${SHARD_FILES[$s]} | tr -s ' ' ',')"
    # shellcheck disable=SC2086
    $PY -m pytest ${SHARD_FILES[$s]} -q "$@" \
        --gpu-proof-enable \
        --gpu-proof-out "test/receipts/$s.json" \
        --gpu-proof-github-user plancherb1 \
        --gpu-proof-shard "$s" \
        --gpu-proof-shard-fingerprint-paths "$FP_PATHS" \
        --gpu-proof-fingerprint-paths "glass.cuh,glass-cgrps.cuh,glass-nvidia.cuh,glass-defaults.cuh,glass-dispatch.cuh,src,test/cuda,test/conftest.py"
    FRESH+=("test/receipts/$s.json")
done

CARRY=()
if [[ "$RUN_SHARDS" != "$ALL_SHARDS" ]] && git cat-file -e HEAD:test/gpu-proof.json 2>/dev/null; then
    git show HEAD:test/gpu-proof.json > test/receipts/_last_green.json
    CARRY=(--carry-from test/receipts/_last_green.json --repo .)
fi
.venv/bin/gpu-proof merge --out test/gpu-proof.json "${CARRY[@]}" \
    --github-user plancherb1 "${FRESH[@]}"

echo
echo "Signed receipt: test/gpu-proof.json — 'git add test/gpu-proof.json' to attest this run."
