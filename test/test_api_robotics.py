import subprocess


def test_robotics_overload_compile_canary(bins):
    result = subprocess.run(
        [str(bins["api_robotics"])], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "1"
