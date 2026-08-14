# GLASS contributor and agent guide

GLASS is a header-only CUDA C++ library of `__device__` routines for small,
block-local linear algebra and robotics math. Most callers launch one CUDA block
per independent problem and compose GLASS operations inside a larger kernel.

This file contains working rules, not project history. For user-facing API
documentation, start with `README.md`; for current work, see `docs/HANDOFF.md`.

## Read before editing

1. `docs/agent_debugging_guide.md` before changing a primitive.
2. `docs/source/user_guide/concepts/namespaces.rst` before changing public API.
3. `docs/source/user_guide/concepts/testing_oracles.rst` before changing tests.
4. `bench/TUNING.md` before changing measured defaults or benchmark results.

## Execution and API contracts

- `glass::block::` is the explicit pure-SIMT block implementation.
- Bare `glass::op` is the measured-default block-scoped face. It may select a
  block, warp-0, or thread-0 body for measured compile-time cells.
- `glass::warp::` owns one problem per full 32-lane warp.
- `glass::thread::` owns one problem per thread. It is compile-time-only,
  branch-free, and intended for register-resident sizes up to `N <= 7`.
- `glass::nvidia::block::` and `glass::nvidia::warp::` expose vendor-backed
  implementations. MathDx is optional; CUB ships with CUDA.
- `glass::cgrps::` is a cooperative-groups spelling of the block algorithm, not
  an independently tuned backend.

Use an explicit namespace when the implementation or deterministic reduction
order is part of the caller's contract. Use the bare face when measured-default
dispatch is desired.

The naming rule is:

- namespace = execution scope or backend;
- suffix = a distinct algorithm, such as `_fast`, `_lowmem`, or `_reduced`;
- template flag = optional behavior that compiles out when disabled.

Do not infer that every operation exists in every namespace. The generated
public-overload manifest, `test/api-contracts.json`, is the inventory of what is
actually documented and compile-covered.

## Primitive invariants

- A block primitive must be entered by every participating block thread.
- Results must be invariant across supported block sizes, including partial
  warps, unless the API documents a stricter launch contract.
- Put a synchronization boundary between a cooperative write phase and any
  dependent read phase.
- Route shared implementations through their barrier object. A raw
  `__syncthreads()` is invalid in a thread-per-problem implementation.
- `TRAILING_SYNC=true` publishes results before return. With
  `TRAILING_SYNC=false`, the caller owns the next required synchronization.
- `beta == 0` overloads must not read the destination.
- Matrices are column-major unless the API explicitly exposes a layout flag.
- Robotics conventions are load-bearing: spatial vectors are angular-first
  `[omega; v]`, SE(3) tangent blocks are linear-first `[rho; phi]`, and
  quaternion layout is selected by `QuatLayout`.

## Repository map

- `glass.cuh`, `glass-cgrps.cuh`, `glass-nvidia.cuh`: installed umbrella headers.
- `glass-defaults.cuh`, `glass-dispatch.cuh`: generated measured defaults.
- `src/base/L1`, `L2`, `L3`: pure-SIMT primitives and inline warp/thread forms.
- `src/base/banded`, `src/base/pcg`: public structured-system solvers.
- `src/cgrps`: cooperative-groups overloads.
- `src/nvidia`: CUB, cuBLASDx, and cuSOLVERDx wrappers.
- `src/internal`: tested implementation details that are not public API.
- `test`: pytest oracles and CUDA drivers.
- `bench`: benchmark harnesses, tuning policy, and generated reports.
- `docs/source`: published Sphinx documentation.
- `docs/open-tasks`: dated design records and archived plans; not current policy.

## Testing workflow

Install the Python tools once:

```bash
python3 -m venv .venv
.venv/bin/pip install -r test/requirements.txt
```

Run the smallest relevant test first. The `bins` fixture compiles binaries on
first access and caches them by functional header family, so a selected test no
longer builds the entire suite.

```bash
.venv/bin/python -m pytest test/test_l3.py -k gemm -q
.venv/bin/python .github/scripts/api_contract_coverage.py \
  --check-manifest test/api-contracts.json --require-100
.venv/bin/python .github/scripts/coverage_obligations.py
```

The signed GPU receipt is split into eight shards: `vector`, `dense`, `factor`,
`tiers`, `solvers`, `robotics`, `mathdx`, and `integration`. During development,
rerun only affected shards and carry forward unchanged fresh shards:

```bash
./test/run_gpu_proof.sh vector dense
```

A release always runs all eight shards from scratch. Any tracked change under a
fingerprinted library or test path invalidates the old receipt. Stage newly
created files before the final receipt so the fingerprint includes them.

## Benchmark workflow

Correctness and timing are separate gates. A published timing artifact must
identify the source revision, device, toolchain, benchmark configuration, date,
and proximate signed correctness receipt.

- Author and compile benchmark binaries while the machine is shared.
- Compile serially when a MathDx translation unit may consume several GB of RAM.
- Run timed sweeps only on a quiet GPU.
- Do not edit generated tables or result blocks by hand; use their owning script.
- Treat a comparison as narrowly as its harness: disclose candidate selection,
  vendor configuration, batch regime, precision, and statistic.

See `bench/TUNING.md` for the authoritative commands and selection policy.

## Documentation and release checks

Public functions need Doxygen comments and inclusion in the relevant Sphinx API
page. Keep landing pages task-oriented; put detailed contracts in one canonical
concept page and link to it instead of repeating them.

Before handoff:

```bash
git diff --check
cd docs && make all SPHINXOPTS="-W --keep-going"
```

`release.sh` requires a clean `main`, a changelog entry, a current 100% overload
manifest, 100% declared correctness obligations, a fresh full GPU receipt, and
local receipt verification before it tags and pushes.
