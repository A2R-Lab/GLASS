# Changelog

All notable changes to GLASS will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `glass::block::riccati_gain` reuses the symmetric `P·B` intermediate
  (`BᵀPA = (PB)ᵀA`), shrinking its shared-scratch requirement from
  `NU²+NX²` to `NU²+NX·NU` scalars and speeding up the fused gain solve.
  Results can shift at ULP level relative to earlier releases;
  `riccati_scratch_bytes<T,NX,NU>()` reflects the smaller footprint.
- `glass::block::pcg` elides trailing barriers that were immediately followed
  by an explicit `__syncthreads()` between composed sub-operations
  (numerically identical, fewer barriers per iteration).
- Compile-time `ger` uses a flat one-thread-per-output work mapping
  (bit-identical results by construction).
- Documented-overload coverage is measured per overload contract — 662
  contracts extracted from Doxygen XML with call-shape evidence
  (`test/api-contracts.json`) — instead of per public name, plus 19 declared
  behavioral correctness obligations checked against the signed receipt.
- The signed GPU receipt is split into eight dependency-scoped shards with
  lazy test-binary compilation; development reruns only affected shards, and
  releases still require a fresh all-shard receipt (now enforced by a
  no-carry verify policy in `release.sh`).
- `test/test_pcg.py` sweeps only multiple-of-32 thread counts, matching
  `pcg`'s documented full-warp launch contract (the removed 1/7/33-thread
  passes exercised undefined behavior, not coverage).

### Fixed

- `glass-dispatch.cuh` is included in the CMake install file list (previously
  a CMake-installed GLASS could not compile the bare measured-default face).
- CMake propagates the cuBLASDx CUTLASS include path when
  `GLASS_ENABLE_MATHDX` is on.
- `pcg` documentation states the scratch size in BYTES and the dynamic
  shared-memory launch requirement explicitly (the old wording said
  "elements", understating the allocation).

[Unreleased]: https://github.com/A2R-Lab/GLASS/commits/main
