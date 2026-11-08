"""Excel mirror and SQL dumps of the web app data.

* Excel: data/records/instamart_web_records.xlsx (users, customers, orders, tickets, messages).
  It deliberately has NO password column: hashes stay in the database and SQL dump only.
* SQL dump: data/dump/app_dump.sql (web app data, including password HASHES) and
  data/dump/instamart_dump.sql (the cleaned Part A database).
"""
import os
import sqlite3
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

SHEETS = {
    "Users": "SELECT user_id, username, role, customer_id, full_name, email, active, created_at FROM users ORDER BY user_id",
    "WebCustomers": "SELECT * FROM web_customers ORDER BY customer_id",
    "Orders": "SELECT 'WEB-' || printf('%06d', order_no) AS order_ref, customer_id, user_id, product_id, product_name, "
              "quantity, unit_price, amount, status, created_at, updated_at FROM web_orders ORDER BY order_no",
    "Tickets": "SELECT 'TKT-' || printf('%06d', ticket_no) AS ticket_ref, customer_id, user_id, order_ref, subject, body, "
               "category, triage_status, route, priority, status, handled_by, bot_replies, created_at, updated_at FROM tickets ORDER BY ticket_no",
    "ChatLog": "SELECT chat_no, user_id, customer_id, turns, outcome, created_at FROM chat_log ORDER BY chat_no",
    "TicketMessages": "SELECT msg_no, 'TKT-' || printf('%06d', ticket_no) AS ticket_ref, author_id, author_role, message, created_at "
                      "FROM ticket_messages ORDER BY msg_no",
}


def _plain(db_path):
    con = sqlite3.connect(Path(db_path).as_uri(), uri=True)
    con.row_factory = sqlite3.Row
    from webapp.db import migrate
    migrate(con)                       # older dumps lack the support-bot ticket columns
    return con


def _excel_safe(v):
    """Neutralise spreadsheet formula injection: a text cell starting with = + - @ must not become a formula."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def sync_excel(cfg):
    con = _plain(cfg.app_db)
    wb = Workbook()
    wb.remove(wb.active)
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="2F4858")
    for name, sql in SHEETS.items():
        ws = wb.create_sheet(name)
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        ws.append(cols)
        for row in cur:
            ws.append([_excel_safe(v) for v in tuple(row)])
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
        ws.freeze_panes = "A2"
        for i, col in enumerate(cols, 1):
            width = max([len(str(col))] + [len(str(c.value)) for c in ws[get_column_letter(i)][1:50] if c.value is not None])
            ws.column_dimensions[get_column_letter(i)].width = min(width + 2, 60)
    con.close()
    cfg.excel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.excel_path.with_suffix(".tmp")
    wb.save(tmp)
    os.replace(tmp, cfg.excel_path)                       # atomic: readers never see a half-written file
    return cfg.excel_path


def dump_sql(db_path, out_path):
    con = _plain(db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for line in con.iterdump():
            f.write(line + "\n")
    con.close()
    os.replace(tmp, out_path)
    return out_path


def dump_app(cfg):
    return dump_sql(cfg.app_db, cfg.dump_dir / "app_dump.sql")


def dump_all(cfg):
    paths = [dump_app(cfg)]
    if cfg.core_db.exists():
        paths.append(dump_sql(cfg.core_db, cfg.dump_dir / "instamart_dump.sql"))
    return paths


def refresh(cfg):
    """Called after every write: keep the Excel record and the app dump in step with the database."""
    sync_excel(cfg)
    dump_app(cfg)


def restore_app(cfg, force=False):
    """Rebuild app.db from data/dump/app_dump.sql (for a fresh clone)."""
    src = cfg.dump_dir / "app_dump.sql"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    if cfg.app_db.exists():
        if not force:
            raise FileExistsError(f"{cfg.app_db} already exists; pass force=True to replace it")
        cfg.app_db.unlink()
    con = sqlite3.connect(cfg.app_db)
    con.executescript(src.read_text(encoding="utf-8"))
    from webapp.db import migrate
    migrate(con)                       # a dump written before the support bot existed gets the new columns
    con.close()
