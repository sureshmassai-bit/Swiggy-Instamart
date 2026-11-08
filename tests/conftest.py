"""Shared fixtures. Tests run from anywhere; they build missing artifacts in dependency order."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = [ROOT / "data/raw/orders.csv", ROOT / "data/instamart.db", ROOT / "data/clean/orders.csv",
            ROOT / "outputs/customer_segments.csv", ROOT / "outputs/lapse_model.joblib"]


@pytest.fixture(scope="session", autouse=True)
def pipeline_built():
    """Build raw data, Part A and Part B outputs if they are missing (the later parts depend on them)."""
    if not REQUIRED[0].exists():
        subprocess.run([sys.executable, "generate_data.py"], cwd=ROOT, check=True)
    if not all(p.exists() for p in REQUIRED[1:3]):
        subprocess.run([sys.executable, "-m", "src.data_pipeline"], cwd=ROOT, check=True)
    if not all(p.exists() for p in REQUIRED[3:]):
        subprocess.run([sys.executable, "-m", "src.customer_models"], cwd=ROOT, check=True)


@pytest.fixture
def agent_sandbox(tmp_path):
    """Redirect the refund ledger and checkpoints to a temp dir so tests never touch real artifacts."""
    from src import refund_agent as ra
    with ra.isolated(tmp_path / "ledger.jsonl", tmp_path / "state"):
        yield ra
