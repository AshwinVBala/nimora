import os
import subprocess
import sys
from pathlib import Path


def test_serialization_imports_in_clean_process():
    root = Path(__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from nimora.serialization import canonical_json; "
                "from nimora.agent import AgentRuntime; "
                "assert canonical_json({'ok': True}) == '{\"ok\":true}'; "
                "assert AgentRuntime.__name__ == 'AgentRuntime'"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
