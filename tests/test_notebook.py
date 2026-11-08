"""Notebook hygiene and a clean-interpreter, top-to-bottom run.

The run regenerates raw data (deterministic) and rewrites outputs/, exactly as the notebook is meant to.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "Customer_Intelligence_Support_Copilot.ipynb"


def test_notebook_is_valid_and_has_no_stale_outputs():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    assert not any("C:\\" in "".join(c["source"]) for c in nb["cells"]), "no hard-coded Windows paths"


def test_notebook_runs_top_to_bottom_from_notebooks_dir():
    """cwd=notebooks/ exercises the ROOT auto-detection as well."""
    proc = subprocess.run([sys.executable, str(ROOT / "tests" / "run_notebook.py"), str(NB)],
                          cwd=ROOT / "notebooks", capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "NOTEBOOK RAN TOP TO BOTTOM WITHOUT ERRORS" in proc.stdout
    assert "ALL CHECKPOINTS PASSED" in proc.stdout
