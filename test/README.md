# GLASS tests

CUDA equivalence tests: each GLASS primitive is run on the GPU and compared
against a NumPy/SciPy reference. The harness is Python (pytest) driving compiled
CUDA programs.

## Running

```bash
pip install -r test/requirements.txt   # numpy, pytest, scipy
pytest test/                           # from the repo root
pytest test/test_l3.py -k gemm -v      # a subset
```

You need an NVIDIA GPU + CUDA toolkit (`nvcc`). The `glass::nvidia::` /
cuBLASDx tests additionally need NVIDIA MathDx (`MATHDX_ROOT`) and **skip
gracefully** when it is absent.

## Layout

| File | Covers |
|------|--------|
| `test_l1.py` | L1 vector ops (axpy, copy, dot, reduce, norms, elementwise, …) |
| `test_l2.py` | L2 matrix-vector ops (gemv, ger, strided/segmented) |
| `test_l3.py` | L3 matrix ops (gemm family, inv, chol, trsm, batched-1D) |
| `test_nvidia_dispatch.py` | `glass::nvidia::` SIMT-vs-cuBLASDx auto-dispatch (needs MathDx) |
| `test_trailing_sync.py` | the `TRAILING_SYNC` template parameter across the surface |
| `cuda/*.cu` | the CUDA programs the Python tests invoke (`helpers.cuh` shared) |
| `conftest.py` | arch detection, compile caching, fixtures, `run_op` harness |

## How the harness works (`conftest.py`)

- **GPU arch detection** — `detect_arch()` queries `nvidia-smi --query-gpu=compute_cap`
  and builds with `-arch=sm_XX` (falls back to `sm_75`).
- **Compile-once, cache by source hash** — `compile_binary()` compiles each
  `cuda/test_*.cu` once per session and caches it; it recompiles only when
  `_hash_sources()` changes. **Gotcha:** that hash covers a *curated* list of
  headers (`glass.cuh`, the nvidia headers, a few base files), not every header.
  If you edit a header that isn't in the list and the cache looks stale,
  `rm -rf test/build` to force a clean rebuild.
- **Optional-dependency skips** — `test_l3_nvidia`, `test_nvidia_dispatch`, and
  `test_trailing_sync` are compiled in `try/except`; their fixtures
  (`bin_l3_nvidia`, `bin_nvidia_dispatch`, `bin_trailing_sync`) `pytest.skip`
  when the binary didn't build (e.g. no MathDx).
- **Data path** — `run_op()` writes NumPy `float32` arrays to temp `.bin` files,
  invokes the CUDA binary with the op/version/args, and parses stdout back into
  arrays.

`test/build/` (compiled binaries + `.hash` files) is gitignored.

See [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) for *why* the tests are shaped
this way — especially the thread-count sweep that catches single-block races.
