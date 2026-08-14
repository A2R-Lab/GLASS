# Changelog

All notable changes to GLASS will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Audit the public API, documentation, correctness coverage, benchmark
  methodology, and release process for consistency and reproducibility.
- Reuse the symmetric Riccati ``P*B`` intermediate, remove redundant
  composition barriers, and flatten compile-time GER work mapping; the
  accompanying sm_120 A/B sweeps cover both ``float`` and ``double``.
- Split the signed GPU proof into dependency-scoped shards with lazy binary
  compilation while retaining a mandatory fresh all-shard release receipt.
