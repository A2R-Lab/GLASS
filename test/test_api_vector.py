import subprocess
from conftest import RUN_TIMEOUT


def test_vector_overload_compile_canary(bins):
    result = subprocess.run(
        [str(bins["api_vector"])], capture_output=True, text=True, check=True, timeout=RUN_TIMEOUT
    )
    assert result.stdout.strip() == "1"
