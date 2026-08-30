"""CPU-only contract tests for the measured execution-plan generator."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import autotune  # noqa: E402
import tune  # noqa: E402


SYNTHETIC_SWEEP = """\
################ NPROB=8192  reps=300  dtype=f32 ################
potrf  N=8 | BLOCK | WARP | THREAD || block tb32=10.0 warp w2=8.0 thread t32=7.0 nv=6.0 nvt t32=3.0 -> NVIDIA_THREAD
"""


def test_ladder_preserves_measured_native_runner_up():
    full = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05)
    native = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05, native_only=True)

    assert full[("f32", "potrf")][8] == "nvidia_thread"
    assert native[("f32", "potrf")][8] == "thread"


def test_local_override_emits_both_dependency_policies(tmp_path):
    capture = tmp_path / "capture.txt"
    output = tmp_path / "defaults.cuh"
    capture.write_text(SYNTHETIC_SWEEP)

    autotune.emit_defaults_table(capture, output, 1200, margin=0.05)
    generated = output.read_text()

    assert "bool allow_nvidia" in generated
    assert "if (allow_nvidia)" in generated
    assert "backend::nvidia_thread" in generated
    assert "if (!allow_nvidia)" in generated
    assert "backend::thread" in generated


def test_local_ladder_override_does_not_mask_other_tables():
    defaults = (ROOT / "glass-defaults.cuh").read_text()
    blas2_dispatch = "if (is_blas2(o)) return blas2_ideal"
    override = "#ifdef GLASS_DEFAULTS_HAVE_LOCAL"

    assert defaults.index(blas2_dispatch) < defaults.index(override)
