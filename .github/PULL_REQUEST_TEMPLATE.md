<!-- Thanks for contributing to GLASS! -->

## What & why

<!-- One paragraph: what changes, and why. Link the issue if there is one. -->

## Checklist

- [ ] **Receipt**: this PR touches `src/`, a `glass*.cuh` root, or `test/` —
      so the tip commit includes a fresh signed `test/gpu-proof.json` from
      `./test/run_gpu_proof.sh` (full GPU suite; the `verify-gpu-proof` gate
      is red otherwise). Doc-only / bench-only changes can skip this.
- [ ] **Thread-count invariance**: any new/changed block-scope op produces
      identical results at 1, 32, 33, 64, … threads (the tests sweep this).
- [ ] **Doc-comment**: new public functions carry a `/** */` doc-comment and
      are pulled into `docs/source/api_reference/` (the API-coverage check
      counts every documented name; the badge goes red below 100%).
- [ ] **Tier uniformity**: where it makes sense, the op is provided across
      block / warp / thread from one shared core (see existing families).
- [ ] **No silent gating**: no per-size or per-robot skips without a loud
      comment and a tracked issue.

<!-- New GPU tuning tables: include the signed receipt + capture provenance
     (see bench/TUNING.md) so the table can be verified without a maintainer
     owning your hardware. -->
