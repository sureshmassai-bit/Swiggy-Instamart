"""Management commands (run from the repository root):

    python -m webapp.manage init              create schema, 1 admin + 10 demo users, credential files, Excel, dumps
    python -m webapp.manage restore           rebuild data/app.db from data/dump/app_dump.sql (fresh clone)
    python -m webapp.manage dump              rewrite data/dump/*.sql and the Excel record
    python -m webapp.manage reset-password USERNAME   generate a new random password and print it once
"""
import argparse
import secrets
import sys

from webapp import records
from webapp.config import Config
from webapp.db import connect, hash_password, now

ADMIN_USERNAME = "admin"
DEMO_PLAN = [("Premium", 3), ("Loyal Frequent", 3), ("Casual", 2), ("At Risk / Dormant", 2)]   # 10 demo customers


def new_password():
    return secrets.token_urlsafe(10)          # ~14 URL-safe characters, ~80 bits


def pick_demo_customers(con):
    """Deterministic: the lowest customer_ids of each segment, so the demo users cover every segment."""
    picks = []
    for seg, n in DEMO_PLAN:
        picks += [r[0] for r in con.execute("SELECT customer_id FROM seg WHERE segment_name = ? ORDER BY customer_id LIMIT ?", (seg, n))]
    if len(picks) < 10:                       # segments file missing: fall back to the first customers
        picks = [r[0] for r in con.execute("SELECT customer_id FROM core.customers ORDER BY customer_id LIMIT 10")]
    return picks


def write_credentials(cfg, admin, users):
    cfg.credentials_dir.mkdir(parents=True, exist_ok=True)
    admin_file, user_file = cfg.credentials_dir / "ADMIN_CREDENTIALS.txt", cfg.credentials_dir / "USER_CREDENTIALS.txt"
    admin_file.write_text(
        "INSTAMART COPILOT - ADMIN LOGIN (keep private; do not commit)\n"
        "URL      : http://127.0.0.1:5000/login\n"
        f"Username : {admin[0]}\nPassword : {admin[1]}\n\n"
        "Can: view every customer, order and ticket; reply to and close tickets; update order status; download the Excel record.\n",
        encoding="utf-8")
    lines = ["INSTAMART COPILOT - CUSTOMER LOGINS (demo accounts; keep private; do not commit)",
             "URL: http://127.0.0.1:5000/login", "",
             f"{'Username':<10} {'Password':<18} {'Customer ID':<12} Segment", "-" * 60]
    lines += [f"{u:<10} {p:<18} {c:<12} {s}" for u, p, c, s in users]
    lines += ["", "Can: order products, view own orders, raise and follow up on own tickets.",
              "New customers can also register at /register."]
    user_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for f in (admin_file, user_file):
        try:
            f.chmod(0o600)
        except OSError:
            pass
    return admin_file, user_file


def cmd_init(cfg):
    con = connect(cfg)
    if con.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
        print("Users already exist; nothing created. To get a new password run: python -m webapp.manage reset-password USERNAME")
        records.refresh(cfg)
        records.dump_all(cfg)
        return 0
    ts = now()
    admin_pw = new_password()
    con.execute("INSERT INTO users (username, role, customer_id, full_name, email, password_hash, created_at) "
                "VALUES (?, 'admin', NULL, 'Support Admin', 'admin@instamart.local', ?, ?)", (ADMIN_USERNAME, hash_password(cfg, admin_pw), ts))
    users = []
    for i, cid in enumerate(pick_demo_customers(con), 1):
        username, pw = f"user{i:02d}", new_password()
        seg = (con.execute("SELECT segment_name FROM seg WHERE customer_id = ?", (cid,)).fetchone() or ["Not segmented"])[0]
        con.execute("INSERT INTO users (username, role, customer_id, full_name, email, password_hash, created_at) VALUES (?, 'user', ?, ?, ?, ?, ?)",
                    (username, cid, f"Customer {cid}", f"{username}@instamart.local", hash_password(cfg, pw), ts))
        users.append((username, pw, cid, seg))
    con.commit()
    con.close()
    files = write_credentials(cfg, (ADMIN_USERNAME, admin_pw), users)
    records.refresh(cfg)
    records.dump_all(cfg)
    print("Created 1 admin and", len(users), "customer logins.")
    print("Credentials written (git-ignored):", *[str(f) for f in files], sep="\n  ")
    print("Excel record:", cfg.excel_path, "\nSQL dumps   :", cfg.dump_dir)
    return 0


def cmd_reset(cfg, username):
    con = connect(cfg, with_core=False)
    pw = new_password()
    cur = con.execute("UPDATE users SET password_hash = ?, active = 1 WHERE username = ?", (hash_password(cfg, pw), username.lower()))
    con.commit()
    con.close()
    if cur.rowcount == 0:
        print("No such user:", username)
        return 1
    records.refresh(cfg)
    print(f"New password for {username}: {pw}\n(shown once; it is not stored anywhere in plain text)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m webapp.manage")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    r = sub.add_parser("restore")
    r.add_argument("--force", action="store_true")
    sub.add_parser("dump")
    rp = sub.add_parser("reset-password")
    rp.add_argument("username")
    a = p.parse_args(argv)
    cfg = Config()
    if a.cmd == "init":
        return cmd_init(cfg)
    if a.cmd == "restore":
        records.restore_app(cfg, force=a.force)
        print("Restored", cfg.app_db, "from", cfg.dump_dir / "app_dump.sql")
        return 0
    if a.cmd == "dump":
        connect(cfg).close()
        records.refresh(cfg)
        print(*records.dump_all(cfg), cfg.excel_path, sep="\n")
        return 0
    return cmd_reset(cfg, a.username)


if __name__ == "__main__":
    sys.exit(main())
