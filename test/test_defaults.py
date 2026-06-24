"""Backend-defaults query helpers (glass-defaults.cuh).

The helpers are constexpr, so correctness is enforced by static_asserts in
test/cuda/test_defaults.cu — if the binary compiled, the picks match the sweep
(bench/MEGA_SWEEP_RESULTS.md). This just confirms it built and runs.
"""
import subprocess


def test_defaults_compile_and_run(bin_defaults):
    out = subprocess.run([str(bin_defaults)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
