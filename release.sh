#!/usr/bin/env bash
# release.sh vX.Y.Z — the coordinated release gate. Nothing ships stale:
#
#   1. clean-tree + main-branch preconditions; CHANGELOG.md must have a
#      section for the version being released;
#   2. 100% API-surface coverage gate (.github/scripts/surface_coverage.py —
#      the badge contract; a release cannot regress it);
#   3. FULL five-shard GPU receipt — carry-forward is deliberately NOT used
#      for a release: every shard re-runs at the release commit;
#   4. local verify of the merged receipt (same seven checks CI runs);
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
python3 .github/scripts/surface_coverage.py | tee /tmp/glass_release_cov.txt
grep -q "100.0%" /tmp/glass_release_cov.txt || { echo "FATAL: API-surface coverage below 100%"; exit 1; }

echo "── full five-shard GPU receipt (no carry-forward) ─────"
./test/run_gpu_proof.sh

echo "── verify ─────────────────────────────────────────────"
.venv/bin/gpu-proof verify --receipt test/gpu-proof.json --require-gpu \
    --expected-skips test/expected_skips.txt --policy test/gpu-proof-policy.json

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
