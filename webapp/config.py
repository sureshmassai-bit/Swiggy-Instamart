"""Paths and settings for the web app. Every path can be overridden (tests use a temp dir)."""
from dataclasses import dataclass
from pathlib import Path
import os
import secrets

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    app_db: Path = ROOT / "data" / "app.db"                  # users, web orders, tickets (git-ignored; dump is committed)
    core_db: Path = ROOT / "data" / "instamart.db"           # cleaned Part A database, opened READ-ONLY
    segments_csv: Path = ROOT / "outputs" / "customer_segments.csv"
    outputs_dir: Path = ROOT / "outputs"                     # Parts A-D artifacts shown on the analytics pages (read)
    ledger_path: Path = ROOT / "outputs" / "refund_ledger.jsonl"   # refund agent run from the admin console (write)
    run_log_path: Path = ROOT / "outputs" / "agent_runs.jsonl"
    state_dir: Path = ROOT / "state"                         # agent checkpoints
    excel_path: Path = ROOT / "data" / "records" / "instamart_web_records.xlsx"
    dump_dir: Path = ROOT / "data" / "dump"
    credentials_dir: Path = ROOT / "credentials"             # git-ignored
    secret_key_path: Path = ROOT / "data" / ".secret_key"    # git-ignored
    pw_iterations: int = 600_000                             # PBKDF2-SHA256 work factor
    max_failed_logins: int = 5                               # per username ...
    lockout_minutes: int = 10                                # ... within this window
    page_size: int = 25

    def secret_key(self):
        env = os.environ.get("INSTAMART_SECRET_KEY")
        if env:
            return env
        if not self.secret_key_path.exists():
            self.secret_key_path.parent.mkdir(parents=True, exist_ok=True)
            self.secret_key_path.write_text(secrets.token_hex(32))
            try:
                self.secret_key_path.chmod(0o600)
            except OSError:
                pass
        return self.secret_key_path.read_text().strip()
