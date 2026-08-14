import subprocess


def test_dense_overload_compile_canary(bins):
    result = subprocess.run(
        [str(bins["api_dense"])], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "1"
