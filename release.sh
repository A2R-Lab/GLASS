#!/usr/bin/env bash
# release.sh vX.Y.Z — the coordinated release gate. Nothing ships stale:
#
#   1. clean-tree + main-branch preconditions; CHANGELOG.md must have a
#      section for the version being released;
#   2. 100% Doxygen-derived overload compile-coverage gate; the generated API
#      manifest must also be current;
#   3. FULL eight-shard GPU receipt — carry-forward is deliberately NOT used
#      for a release: every shard re-runs at the release commit;
#   4. local verify of the merged receipt and every declared correctness
#      obligation against passing evidence in its assigned shard;
#   5. commit the receipt if it changed, annotated tag, push branch + tag —
#      the push regenerates the coverage/receipt badges via the Documentation
#      workflow, so the public numbers can never go stale relative to a tag.
#
# Usage: ./release.sh v0.1.0
set -euo pipefail
cd "$(dirname "$0")"

VER="${1:?usage: ./release.sh vX.Y.Z}"
[[ "$VER" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "FATAL: version must look like v0.1.0"; exit 1; }

echo "── preconditions ──────────────────────────────────────"
[[ -z "$(git status --porcelain)" ]] || { echo "FATAL: working tree not clean"; exit 1; }
[[ "$(git branch --show-current)" == "main" ]] || { echo "FATAL: not on main"; exit 1; }
git tag | grep -qx "$VER" && { echo "FATAL: tag $VER already exists"; exit 1; }
grep -q "^## \[${VER#v}\]\|^## $VER" CHANGELOG.md || { echo "FATAL: CHANGELOG.md has no section for $VER"; exit 1; }

echo "── coverage gate (100% is the contract) ───────────────"
.venv/bin/python .github/scripts/api_contract_coverage.py \
    --check-manifest test/api-contracts.json --require-100
.venv/bin/python .github/scripts/coverage_obligations.py

echo "── full eight-shard GPU receipt (no carry-forward) ────"
./test/run_gpu_proof.sh

echo "── verify ─────────────────────────────────────────────"
# Release policy forbids carried shards outright (allow_carried:false) so the
# full-fresh property is ENFORCED, not just incidental to running all shards.
.venv/bin/gpu-proof verify --receipt test/gpu-proof.json --require-gpu \
    --expected-skips test/expected_skips.txt \
    --policy test/gpu-proof-release-policy.json
.venv/bin/python .github/scripts/coverage_obligations.py \
    --receipt test/gpu-proof.json

if [[ -n "$(git status --porcelain test/gpu-proof.json)" ]]; then
    git add test/gpu-proof.json
    git commit -m "receipt: $VER release attestation"
fi

echo "── tag + push ─────────────────────────────────────────"
git tag -a "$VER" -m "GLASS $VER"
git push origin main "$VER"

echo
echo "Released $VER. The Documentation workflow now regenerates the coverage"
echo "and receipt badges from this exact tree — nothing public is stale."
