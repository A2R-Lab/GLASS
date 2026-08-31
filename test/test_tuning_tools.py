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

SYNTHETIC_NVT_VALID = """\
NVT_VALID op=potrf N=8 dtype=f32 nprob=8192 slots=64 block=8.0000 block_shape=128 block_spread=0.50 warp=7.5000 warp_shape=8 warp_spread=0.60 thread=7.0000 thread_shape=64 thread_spread=0.40 nvidia_thread=7.2000 nvt_shape=64 nvt_spread=0.70
"""


def test_ladder_preserves_measured_native_runner_up():
    full = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05)
    native = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05, native_only=True)

    assert full[("f32", "potrf")][8] == "nvidia_thread"
    assert native[("f32", "potrf")][8] == "thread"


def test_valid_input_confirmation_is_a_veto_not_a_promotion():
    full = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05)
    gated, vetoes = tune.apply_nvt_valid_veto(full, SYNTHETIC_NVT_VALID, 0.05)

    assert gated[("f32", "potrf")][8] == "thread"
    assert vetoes == 1

    native_main = (SYNTHETIC_SWEEP
                   .replace("nv=6.0", "nv=7.5")
                   .replace("nvt t32=3.0", "nvt t32=7.5"))
    already_native = tune.winners_from_sweep(native_main, 0.05)
    gated, vetoes = tune.apply_nvt_valid_veto(
        already_native,
        SYNTHETIC_NVT_VALID.replace("nvidia_thread=7.2000", "nvidia_thread=2.0000"),
        0.05,
    )
    assert gated[("f32", "potrf")][8] == "thread"
    assert vetoes == 0


def test_valid_input_confirmation_requires_every_selected_cell():
    full = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05)
    try:
        tune.apply_nvt_valid_veto(full, "", 0.05)
    except SystemExit as error:
        assert "lacks valid-input confirmation" in str(error)
    else:
        raise AssertionError("missing confirmation must fail closed")


def test_valid_input_confirmation_rejects_decision_scale_jitter():
    full = tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05)
    noisy = (SYNTHETIC_NVT_VALID
             .replace("nvidia_thread=7.2000", "nvidia_thread=6.5000")
             .replace("nvt_spread=0.70", "nvt_spread=5.10"))
    try:
        tune.apply_nvt_valid_veto(full, noisy, 0.05)
    except SystemExit as error:
        assert "cannot resolve" in str(error)
    else:
        raise AssertionError("decision-scale confirmation jitter must fail closed")


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
