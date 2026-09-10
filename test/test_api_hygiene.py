"""Cheap source-level guards for the public naming contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nvidia_scope_is_always_explicit():
    umbrella = (ROOT / "glass-nvidia.cuh").read_text()
    assert "using namespace block;" not in umbrella


def test_removed_advisors_do_not_reenter_public_headers():
    public = "\n".join(
        (ROOT / name).read_text()
        for name in ("glass.cuh", "glass-defaults.cuh", "glass-nvidia.cuh")
    )
    removed = (
        "suggested_backend",
        "suggested_block_threads",
        "suggested_warps_per_block",
        "suggested_threads_per_block",
        "suggested_use_reduced",
    )
    assert not [name for name in removed if name in public]


def test_one_target_architecture_selector():
    dispatch = (ROOT / "glass-dispatch.cuh").read_text()
    assert "GLASS_TARGET_SM" in dispatch
    assert "#define SMS GLASS_TARGET_SM" in dispatch
    nvidia_headers = "\n".join(
        path.read_text() for path in (ROOT / "src" / "nvidia").glob("*.cuh")
    )
    assert "#define SMS" not in nvidia_headers
