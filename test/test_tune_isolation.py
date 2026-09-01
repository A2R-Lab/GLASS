"""Contract tests for the unified tuning process wrapper (no GPU required)."""

import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import tune  # noqa: E402


def test_run_isolated_uses_seekable_capture_not_pipe(monkeypatch):
    """Raw solver samples must not fill an unread stdout pipe and deadlock."""
    monkeypatch.setattr(tune, "require_quiet_gpu", lambda force: None)
    monkeypatch.setattr(tune, "compute_pids", lambda: set())

    class FakeProc:
        returncode = 0

        def __init__(self, argv, **kwargs):
            del argv
            capture = kwargs["stdout"]
            assert capture is not subprocess.PIPE
            assert capture.seekable()
            capture.write("raw-solver-output\n")
            capture.flush()

    monkeypatch.setattr(tune.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(tune, "watch_process",
                        lambda proc, baseline: (0, set()))

    assert tune.run_isolated(["fake-harness"]) == "raw-solver-output\n"


def test_from_ladder_without_companion_resumes_solver_only(
        monkeypatch, tmp_path):
    mega = tmp_path / "mega.txt"
    mega.write_text("mega rows\n")
    solver = tmp_path / "solver.txt"
    solver.write_text("solver rows\n")
    calls = []

    monkeypatch.setattr(tune, "mathdx_root", lambda: None)
    monkeypatch.setattr(tune, "build_solver_ladder",
                        lambda sms, mdx: "solver-binary")

    def fake_run(binary, quick, sms, margin, **kwargs):
        calls.append((binary, quick, sms, margin, kwargs))
        return solver

    monkeypatch.setattr(tune, "run_solver_ladder", fake_run)
    monkeypatch.setattr(tune, "regen_ladder",
                        lambda *args: ("defaults\n", 1, 1))
    monkeypatch.setattr(tune, "show_diff", lambda *args: False)
    monkeypatch.setattr(
        sys, "argv",
        ["tune.py", "--sm", "870", "--legs", "ladder", "--dry-run",
         "--from-ladder", str(mega), "--sched", "64:200,8192:50"])

    tune.main()

    assert len(calls) == 1
    assert calls[0][0:3] == ("solver-binary", False, 870)
    assert calls[0][4]["sched"] == [("64", "200"), ("8192", "50")]
    assert calls[0][4]["seed"] == 1
