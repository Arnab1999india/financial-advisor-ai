import os
import subprocess
import sys

import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_eval_script_exits_zero():
    result = subprocess.run(
        [sys.executable, "evaluation/run_eval.py"],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "refusal_accuracy" in result.stdout
    assert "groundedness_rate" in result.stdout
