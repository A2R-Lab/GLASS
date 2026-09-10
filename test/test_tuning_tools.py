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

SYNTHETIC_SOLVER = """\
SOLVER_RESULT op=potrf N=8 dtype=f32 nprob=8192 slots=64 impl=block cfg=tb32 ns=8.0000 spread=0.50 samples=8.00,8.04,8.02
SOLVER_RESULT op=potrf N=8 dtype=f32 nprob=8192 slots=64 impl=warp cfg=w2 ns=7.5000 spread=0.60 samples=7.50,7.52,7.54
SOLVER_RESULT op=potrf N=8 dtype=f32 nprob=8192 slots=64 impl=thread cfg=t32 ns=7.0000 spread=0.40 samples=7.00,7.01,7.02
SOLVER_RESULT op=potrf N=8 dtype=f32 nprob=8192 slots=64 impl=nvidia cfg=tb256 ns=6.0000 spread=0.70 samples=6.00,6.02,6.04
SOLVER_RESULT op=potrf N=8 dtype=f32 nprob=8192 slots=64 impl=nvidia_thread cfg=t32 ns=7.2000 spread=0.70 samples=7.20,7.22,7.24
"""


def test_ladder_preserves_measured_native_runner_up():
    full = tune.winners_from_sweep(
        SYNTHETIC_SWEEP, 0.05, solver_text=SYNTHETIC_SOLVER)
    native = tune.winners_from_sweep(
        SYNTHETIC_SWEEP, 0.05, native_only=True,
        solver_text=SYNTHETIC_SOLVER)

    assert full[("f32", "potrf")][8] == "nvidia"
    assert native[("f32", "potrf")][8] == "thread"


def test_fresh_input_solver_capture_is_symmetric_and_authoritative():
    promoted = SYNTHETIC_SOLVER.replace(
        "impl=nvidia_thread cfg=t32 ns=7.2000 spread=0.70 samples=7.20,7.22,7.24",
        "impl=nvidia_thread cfg=t32 ns=2.0000 spread=0.70 samples=2.00,2.01,2.02",
    )
    winners = tune.winners_from_sweep(
        SYNTHETIC_SWEEP, 0.05, solver_text=promoted)
    assert winners[("f32", "potrf")][8] == "nvidia_thread"

    # The old destructive-input row says NVIDIA thread=3ns, but the unified
    # solver row can select any other contender; it is not an asymmetric veto.
    winners = tune.winners_from_sweep(
        SYNTHETIC_SWEEP, 0.05, solver_text=SYNTHETIC_SOLVER)
    assert winners[("f32", "potrf")][8] == "nvidia"


def test_fresh_input_solver_capture_requires_every_solver_cell():
    try:
        tune.winners_from_sweep(SYNTHETIC_SWEEP, 0.05, solver_text="")
    except SystemExit as error:
        assert "capture is incomplete" in str(error)
    else:
        raise AssertionError("missing solver measurement must fail closed")


def test_solver_parser_retains_launch_plan_and_raw_rounds():
    rows = tune.tp.parse_solver_configs(SYNTHETIC_SOLVER)
    row = rows[("f32", "potrf", 8, "nvidia_thread", "t32")]

    assert row["ns"] == 7.2
    assert row["samples"] == (7.2, 7.22, 7.24)
    assert row["slots"] == 64


def test_solver_parser_rejects_too_few_rounds():
    malformed = SYNTHETIC_SOLVER.replace(
        "samples=7.20,7.22,7.24", "samples=7.20,7.22")
    try:
        tune.tp.parse_solver_configs(malformed)
    except ValueError as error:
        assert "at least three" in str(error)
    else:
        raise AssertionError("undersampled solver row must fail closed")


def test_ladder_regeneration_names_symmetric_solver_evidence():
    generated, groups, plans = tune.regen_ladder(
        SYNTHETIC_SWEEP, 0.05, "mega.txt", 1200,
        SYNTHETIC_SOLVER, "solver.txt")

    assert groups == 1
    assert plans == 5
    assert "Fresh-input solver sweep: solver.txt" in generated
    assert "veto" not in generated[generated.index("// === BEGIN tune.py ladder sm_120 ==="):
                                   generated.index("// === END tune.py ladder sm_120 ===")]


def test_ambiguous_vendor_pick_is_demoted_to_native_winner():
    # Median rule selects nvidia (6.0 clears thread 7.0 by >5%), but its
    # slowest round (6.90) cannot beat thread's fastest round (7.00) by the
    # margin, so the cell keeps the capture's native winner.
    ambiguous = SYNTHETIC_SOLVER.replace(
        "impl=nvidia cfg=tb256 ns=6.0000 spread=0.70 samples=6.00,6.02,6.04",
        "impl=nvidia cfg=tb256 ns=6.0000 spread=15.00 samples=6.00,6.02,6.90")
    generated, groups, plans = tune.regen_ladder(
        SYNTHETIC_SWEEP, 0.05, "mega.txt", 1200, ambiguous, "solver.txt")

    assert groups == 1 and plans == 5
    assert "demoted to the capture's native winner" in generated
    region = generated[generated.index("// === BEGIN tune.py ladder sm_120 ==="):
                       generated.index("// === END tune.py ladder sm_120 ===")]
    assert "backend::nvidia_block" not in region.split("native_sm120")[0]


def test_guaranteed_vendor_win_survives_a_wide_spread():
    # 10% round spread, but even the slowest round (6.60) clears thread's
    # fastest round (7.00) by more than 5% — the decision is stable.
    wide = SYNTHETIC_SOLVER.replace(
        "impl=nvidia cfg=tb256 ns=6.0000 spread=0.70 samples=6.00,6.02,6.04",
        "impl=nvidia cfg=tb256 ns=6.0000 spread=10.00 samples=6.00,6.30,6.60")
    generated, _, _ = tune.regen_ladder(
        SYNTHETIC_SWEEP, 0.05, "mega.txt", 1200, wide, "solver.txt")
    assert "demoted" not in generated
    assert "backend::nvidia_block" in generated


def test_noisy_native_pick_never_fails_the_capture():
    # A native winner above the margin is reported, not fatal: the
    # dependency margin and SIMT tie band absorb noise in that direction.
    noisy_native = (SYNTHETIC_SOLVER
                    .replace("impl=nvidia cfg=tb256 ns=6.0000 spread=0.70 "
                             "samples=6.00,6.02,6.04",
                             "impl=nvidia cfg=tb256 ns=7.6000 spread=0.70 "
                             "samples=7.60,7.62,7.64")
                    .replace("impl=thread cfg=t32 ns=7.0000 spread=0.40 "
                             "samples=7.00,7.01,7.02",
                             "impl=thread cfg=t32 ns=7.0000 spread=6.30 "
                             "samples=7.00,7.05,7.44"))
    generated, _, _ = tune.regen_ladder(
        SYNTHETIC_SWEEP, 0.05, "mega.txt", 1200, noisy_native, "solver.txt")
    assert "backend::thread" in generated


def test_local_override_emits_both_dependency_policies(tmp_path):
    capture = tmp_path / "capture.txt"
    output = tmp_path / "defaults.cuh"
    capture.write_text(SYNTHETIC_SWEEP)

    solver = tmp_path / "solver.txt"
    solver.write_text(SYNTHETIC_SOLVER)
    autotune.emit_defaults_table(capture, solver, output, 1200, margin=0.05)
    generated = output.read_text()

    assert "bool allow_nvidia" in generated
    assert "if (allow_nvidia)" in generated
    assert "backend::nvidia_block" in generated
    assert "if (!allow_nvidia)" in generated
    assert "backend::thread" in generated


def test_local_ladder_override_does_not_mask_other_tables():
    defaults = (ROOT / "glass-defaults.cuh").read_text()
    blas2_dispatch = "if (is_blas2(o)) return blas2_ideal"
    override = "#ifdef GLASS_DEFAULTS_HAVE_LOCAL"

    assert defaults.index(blas2_dispatch) < defaults.index(override)
