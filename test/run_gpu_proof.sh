#!/usr/bin/env bash
# Run the GPU suite as EIGHT schema-3 shards and merge into one signed receipt
# (test/gpu-proof.json). Shards carry DAG-scoped fingerprints so an untouched
# family can carry forward from the last committed receipt instead of re-running:
#
#   vector      — L1, iamax, symmetrize
#   dense       — L2/L3 products and storage helpers
#   factor      — factorizations, triangular solves, and dense solvers
#   tiers       — thread/warp/default/dispatch contracts
#   solvers     — banded/pcg/bdsv + box_qp
#   robotics    — spatial/lie/proj/geom/est
#   mathdx      — src/nvidia + glass-nvidia.cuh
#   integration — examples + trailing-sync; deliberately whole-tree scoped
#
# Usage:
#   ./test/run_gpu_proof.sh                       # run all eight shards
#   ./test/run_gpu_proof.sh --shards solvers      # run some; carry the rest
#                                                 # from the committed receipt
#   ./test/run_gpu_proof.sh --shards mathdx \
#       --carry-from test/gpu-proof.json           # repair a current receipt
#   ./test/run_gpu_proof.sh --shards dense,tiers \
#       --reuse-shards factor,solvers \
#       --carry-from test/gpu-proof.json           # merge prior fresh shard files
#   (--allow-dirty permits a dirty-tree run for local iteration — the result
#    can never verify post-commit, so it must not be committed)
#   (extra args after -- go to pytest)
#
# Merge always attempts --carry-from the last committed test/gpu-proof.json
# when a subset ran: freshly-run shards win; absent shards graft in iff their
# narrow fingerprint recomputes identical AND the old commit is an ancestor
# (schema-3 rules — a stale family can never ride along silently). CI verifies
# carried shards under the policy in test/gpu-proof-policy.json (allow_carried +
# carried_max_age_days); see pytest-gpu-proof docs/sharding.md.
#
# The 2 permanent documented skips (test_getrf n=1) are pinned as an EXACT set
# by CI via test/expected_skips.txt. The signer must be the human KEYHOLDER
# (--gpu-proof-github-user), not the org.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
$PY -m pip install -q -r test/requirements.txt

ROOTS_BASE="glass.cuh,glass-cgrps.cuh,glass-defaults.cuh,glass-dispatch.cuh"
SYNC_BASE="src/base/barrier.cuh,src/base/dispatch.cuh,src/base/flags.cuh"
L1_BASE="src/base/L1,$SYNC_BASE"
DENSE_BASE="$L1_BASE,src/base/L2,src/base/L3,src/cgrps"
TEST_BASE="test/conftest.py,test/cuda/helpers.cuh,test/expected_skips.txt"

declare -A SHARD_FILES SHARD_DRIVERS SHARD_PATHS
SHARD_FILES[vector]="test/test_l1.py test/test_l1_round2.py test/test_iamax.py test/test_symmetrize.py test/test_api_vector.py"
SHARD_DRIVERS[vector]="test/cuda/test_l1.cu test/cuda/test_l1_round2.cu test/cuda/test_iamax.cu test/cuda/test_symmetrize.cu test/cuda/test_api_vector.cu"
SHARD_PATHS[vector]="$ROOTS_BASE,$L1_BASE,$TEST_BASE,src/cgrps"

SHARD_FILES[dense]="test/test_l2.py test/test_l3.py test/test_symm_rot.py test/test_syrk.py \
 test/test_congruence.py test/test_tensor.py test/test_reduced.py test/test_reduced_blas.py \
 test/test_block_access.py test/test_api_dense.py"
SHARD_DRIVERS[dense]="test/cuda/test_l2.cu test/cuda/test_l3.cu test/cuda/test_symm_rot.cu \
 test/cuda/test_syrk.cu test/cuda/test_congruence.cu test/cuda/test_tensor.cu \
 test/cuda/test_reduced.cu test/cuda/test_reduced_blas.cu test/cuda/test_block_access.cu"
SHARD_DRIVERS[dense]+=" test/cuda/test_api_dense.cu"
# block_access.cuh: test_block_access exercises banded load/store_block.
SHARD_PATHS[dense]="$ROOTS_BASE,$DENSE_BASE,$TEST_BASE,src/base/banded/block_access.cuh"

SHARD_FILES[factor]="test/test_fused.py test/test_factor_check.py test/test_getrf.py \
 test/test_ldlt.py test/test_trsv.py test/test_posv.py test/test_syev.py \
 test/test_solve.py test/test_base_f64.py test/test_api_factor.py"
SHARD_DRIVERS[factor]="test/cuda/test_fused.cu test/cuda/test_factor_check.cu \
 test/cuda/test_getrf.cu test/cuda/test_ldlt.cu test/cuda/test_trsv.cu \
 test/cuda/test_posv.cu test/cuda/test_syev.cu test/cuda/test_solve.cu \
 test/cuda/test_base_f64.cu test/cuda/test_api_factor.cu"
SHARD_PATHS[factor]="$ROOTS_BASE,$DENSE_BASE,$TEST_BASE"

SHARD_FILES[tiers]="test/test_thread.py test/test_warp.py test/test_defaults.py test/test_dispatch.py \
 test/test_api_hygiene.py test/test_tuning_tools.py test/test_bench_common.py \
 test/test_tune_isolation.py"
SHARD_DRIVERS[tiers]="test/cuda/test_thread.cu test/cuda/test_warp.cu \
 test/cuda/test_defaults.cu test/cuda/test_dispatch.cu"
# svd3.cuh: the dispatch face routes eig3 into its est/ body.
SHARD_PATHS[tiers]="$ROOTS_BASE,$DENSE_BASE,$TEST_BASE,src/base/est/svd3.cuh,bench/bench_common.py,bench/tune.py,bench/tune_pick.py,bench/autotune.py"

SHARD_FILES[solvers]="test/test_banded.py test/test_bdsv.py test/test_pcg.py test/test_qp.py"
SHARD_DRIVERS[solvers]="test/cuda/test_banded.cu test/cuda/test_bdsv.cu \
 test/cuda/test_pcg.cu test/cuda/test_qp.cu"
SHARD_PATHS[solvers]="$ROOTS_BASE,$DENSE_BASE,$TEST_BASE,src/base/banded,src/base/pcg,src/internal/box_qp.cuh"

SHARD_FILES[robotics]="test/test_robotics.py test/test_api_robotics.py"
SHARD_DRIVERS[robotics]="test/cuda/test_robotics.cu test/cuda/test_api_robotics.cu"
SHARD_PATHS[robotics]="$ROOTS_BASE,$DENSE_BASE,$TEST_BASE,src/base/spatial,src/base/lie,src/base/proj,src/base/geom,src/base/est"

SHARD_FILES[mathdx]="test/test_nvidia_dispatch.py test/test_nvidia_f64.py test/test_nvidia_thread.py"
SHARD_DRIVERS[mathdx]="test/cuda/test_nvidia_dispatch.cu test/cuda/test_nvidia_f64.cu \
 test/cuda/test_nvidia_thread.cu test/cuda/test_l3_nvidia.cu"
SHARD_PATHS[mathdx]="$ROOTS_BASE,$DENSE_BASE,$TEST_BASE,src/nvidia,glass-nvidia.cuh"

SHARD_FILES[integration]="test/test_examples.py test/test_trailing_sync.py"
SHARD_DRIVERS[integration]="test/cuda/test_trailing_sync.cu"
SHARD_PATHS[integration]="$ROOTS_BASE,glass-nvidia.cuh,src,$TEST_BASE,examples"

ALL_SHARDS="vector dense factor tiers solvers robotics mathdx integration"
RUN_SHARDS="$ALL_SHARDS"
REUSE_SHARDS=""
CARRY_FROM=""
ALLOW_DIRTY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards)
            RUN_SHARDS="${2//,/ }"; shift 2 ;;
        --carry-from)
            CARRY_FROM="$2"; shift 2 ;;
        --reuse-shards)
            REUSE_SHARDS="${2//,/ }"; shift 2 ;;
        --allow-dirty)
            ALLOW_DIRTY=1; shift ;;
        --)
            shift; break ;;
        *)
            break ;;
    esac
done
[[ "${1:-}" == "--" ]] && shift

# Order-insensitive "did every shard run" test — a reordered or duplicated
# --shards list must still count as a full run (and vice versa).
is_full_run() {
    [[ "$(tr ' ' '\n' <<< "$RUN_SHARDS" | sort -u | tr '\n' ' ')" == \
       "$(tr ' ' '\n' <<< "$ALL_SHARDS" | sort -u | tr '\n' ' ')" ]]
}

# A receipt fingerprinted on a dirty tree can NEVER verify once committed:
# the commit_sha it records predates the very changes it hashed. Refuse
# early instead of discovering it in CI (--allow-dirty for local iteration).
if [[ -n "$(git status --porcelain)" && "$ALLOW_DIRTY" != 1 ]]; then
    echo "FATAL: working tree is dirty — a receipt generated now can never" >&2
    echo "verify against the commit that eventually carries it. Commit (or" >&2
    echo "stash) first, or pass --allow-dirty for a local-only run." >&2
    exit 2
fi

for s in $RUN_SHARDS $REUSE_SHARDS; do
    [[ " $ALL_SHARDS " == *" $s "* ]] || {
        echo "FATAL: unknown shard '$s'" >&2; exit 2;
    }
done
HEAD_SHA=$(git rev-parse HEAD)
for s in $REUSE_SHARDS; do
    [[ " $RUN_SHARDS " != *" $s "* ]] || {
        echo "FATAL: shard '$s' appears in both --shards and --reuse-shards" >&2; exit 2;
    }
    [[ -f "test/receipts/$s.json" ]] || {
        echo "FATAL: reusable receipt test/receipts/$s.json does not exist" >&2; exit 2;
    }
    # Reused shard files merge as FRESH (no carry-time fingerprint recheck),
    # so require they were produced at exactly this commit on a clean tree.
    read -r R_SHA R_DIRTY < <($PY -c "
import json, sys
r = json.load(open('test/receipts/$s.json'))['repo']
print(r.get('commit_sha', '?'), r.get('dirty', True))")
    [[ "$R_SHA" == "$HEAD_SHA" && "$R_DIRTY" == "False" ]] || {
        echo "FATAL: test/receipts/$s.json was generated at $R_SHA (dirty=$R_DIRTY)," >&2
        echo "not at clean HEAD $HEAD_SHA — rerun the shard instead of reusing it." >&2
        exit 2
    }
done

mkdir -p test/receipts
if [[ -n "$CARRY_FROM" ]]; then
    if is_full_run; then
        echo "FATAL: --carry-from is only meaningful with --shards" >&2
        exit 2
    fi
    cp "$CARRY_FROM" test/receipts/_carry_explicit.json
fi
FRESH=()
for s in $RUN_SHARDS; do
    echo "── shard: $s ──────────────────────────────────────────"
    # A shard's own .py test files belong in its fingerprint: without them an
    # edited test could carry forward stale results (hole closed 2026-08-11).
    FP_PATHS="${SHARD_PATHS[$s]},$(echo ${SHARD_FILES[$s]} ${SHARD_DRIVERS[$s]} | tr -s ' ' ',')"
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
for s in $REUSE_SHARDS; do
    echo "── reuse fresh shard receipt: $s ───────────────────────"
    FRESH+=("test/receipts/$s.json")
done

CARRY=()
if [[ -n "$CARRY_FROM" ]]; then
    CARRY=(--carry-from test/receipts/_carry_explicit.json --repo .)
elif ! is_full_run && git cat-file -e HEAD:test/gpu-proof.json 2>/dev/null; then
    git show HEAD:test/gpu-proof.json > test/receipts/_last_green.json
    CARRY=(--carry-from test/receipts/_last_green.json --repo .)
fi
.venv/bin/gpu-proof merge --out test/gpu-proof.json "${CARRY[@]}" \
    --github-user plancherb1 "${FRESH[@]}"

echo
echo "Signed receipt: test/gpu-proof.json — 'git add test/gpu-proof.json' to attest this run."
