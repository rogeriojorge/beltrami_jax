from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def test_examples_run() -> None:
    scripts = [
        ROOT / "examples" / "solve_spec_fixture.py",
        ROOT / "examples" / "parameter_scan.py",
        ROOT / "examples" / "autodiff_mu.py",
    ]
    for script in scripts:
        completed = subprocess.run([str(PYTHON), str(script)], cwd=ROOT, check=True, capture_output=True, text=True)
        assert "[beltrami_jax]" in completed.stdout
