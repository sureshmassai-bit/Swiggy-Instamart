"""SQLite access for the web app: schema, connection (with the read-only core DB attached), helpers."""
import csv
import sqlite3
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS web_customers (
    customer_id TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    city        TEXT NOT NULL,
    age         INTEGER CHECK (age IS NULL OR age BETWEEN 10 AND 100),
    gender      TEXT NOT NULL CHECK (gender IN ('male','female','unknown')),
    created_at  TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role          TEXT NOT NULL CHECK (role IN ('admin','user')),
    customer_id   TEXT,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    created_at    TEXT NOT NULL,
    CHECK ((role = 'admin' AND customer_id IS NULL) OR (role = 'user' AND customer_id IS NOT NULL)));
CREATE TABLE IF NOT EXISTS web_orders (
    order_no    INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    product_id  TEXT NOT NULL,
    product_name TEXT NOT NULL,
    quantity    INTEGER NOT NULL CHECK (quantity BETWEEN 1 AND 20),
    unit_price  REAL NOT NULL CHECK (unit_price > 0),
    amount      REAL NOT NULL CHECK (amount > 0),
    status      TEXT NOT NULL CHECK (status IN ('placed','packed','out_for_delivery','delivered','cancelled')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_no   INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    order_ref   TEXT,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT '',
    triage_status TEXT NOT NULL DEFAULT '',
    route       TEXT NOT NULL DEFAULT '',
    priority    TEXT NOT NULL DEFAULT 'NORMAL',
    status      TEXT NOT NULL CHECK (status IN ('open','answered','closed')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    handled_by  TEXT NOT NULL DEFAULT 'human' CHECK (handled_by IN ('bot','human')),
    bot_replies INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS ticket_messages (
    msg_no      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_no   INTEGER NOT NULL REFERENCES tickets(ticket_no),
    author_id   INTEGER NOT NULL REFERENCES users(user_id),
    author_role TEXT NOT NULL CHECK (author_role IN ('admin','user')),
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chat_log (
    chat_no     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    customer_id TEXT NOT NULL,
    turns       INTEGER NOT NULL CHECK (turns BETWEEN 0 AND 99),
    outcome     TEXT NOT NULL CHECK (outcome IN ('solved','ticket')),
    created_at  TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS login_failures (
    username TEXT NOT NULL COLLATE NOCASE,
    ts       TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_tickets_customer ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON web_orders(customer_id);
"""

ORDER_STATUSES = ["placed", "packed", "out_for_delivery", "delivered", "cancelled"]
TICKET_STATUSES = ["open", "answered", "closed"]
CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def hash_password(cfg, password):
    return generate_password_hash(password, method=f"pbkdf2:sha256:{cfg.pw_iterations}")


def verify_password(stored, password):
    return check_password_hash(stored, password)


_SEG_CACHE = {}


def _segments(cfg):
    path = cfg.segments_csv
    key = (str(path), path.stat().st_mtime if path.exists() else None)
    if key not in _SEG_CACHE:
        rows = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                rows = [(r["customer_id"], r["segment_name"], 1 if r["is_premium"] == "True" else 0) for r in csv.DictReader(f)]
        _SEG_CACHE.clear()
        _SEG_CACHE[key] = rows
    return _SEG_CACHE[key]


def connect(cfg, with_core=True):
    """Open app.db. With `with_core`, attach the Part A database READ-ONLY and create helper views."""
    cfg.app_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(cfg.app_db.as_uri(), uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    migrate(con)
    if with_core:
        if not cfg.core_db.exists():
            raise FileNotFoundError(f"{cfg.core_db} not found. Run Part A first: python -m src.data_pipeline")
        con.execute("ATTACH DATABASE ? AS core", (cfg.core_db.as_uri() + "?mode=ro",))
        con.execute("CREATE TEMP TABLE seg (customer_id TEXT PRIMARY KEY, segment_name TEXT, is_premium INTEGER)")
        con.executemany("INSERT OR IGNORE INTO seg VALUES (?,?,?)", _segments(cfg))
        con.executescript("""
            CREATE TEMP VIEW all_customers AS
                SELECT customer_id, NULL AS full_name, city, age, gender, 'core' AS source FROM core.customers
                UNION ALL
                SELECT customer_id, full_name, city, age, gender, 'web' FROM web_customers;
            CREATE TEMP VIEW all_orders AS
                SELECT order_id AS order_ref, customer_id, product_id, quantity, order_ts AS created_at,
                       status, order_amount AS amount, delivery_minutes, 'history' AS source FROM core.orders
                UNION ALL
                SELECT 'WEB-' || printf('%06d', order_no), customer_id, product_id, quantity, created_at,
                       status, amount, NULL, 'web' FROM web_orders;
        """)
    return con


def migrate(con):
    """Databases made before the support bot existed lack two ticket columns; add them in place (idempotent)."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(tickets)")}
    if not cols:
        return                                           # not the app database (e.g. the Part A core DB)
    if "handled_by" not in cols:
        con.execute("ALTER TABLE tickets ADD COLUMN handled_by TEXT NOT NULL DEFAULT 'human' CHECK (handled_by IN ('bot','human'))")
    if "bot_replies" not in cols:
        con.execute("ALTER TABLE tickets ADD COLUMN bot_replies INTEGER NOT NULL DEFAULT 0")
    con.commit()


BOT_USERNAME = "copilot_bot"


def bot_user_id(con):
    """The disabled system account that authors bot replies (cannot log in: active=0, unusable hash)."""
    row = con.execute("SELECT user_id FROM users WHERE username = ?", (BOT_USERNAME,)).fetchone()
    if row:
        return row[0]
    cur = con.execute("INSERT INTO users (username, role, customer_id, full_name, email, password_hash, active, created_at) "
                      "VALUES (?, 'admin', NULL, 'Instamart Copilot Assistant', 'bot@instamart.invalid', '!disabled', 0, ?)",
                      (BOT_USERNAME, now()))
    return cur.lastrowid


def customer_of(con, customer_id):
    return con.execute("SELECT * FROM all_customers WHERE customer_id = ?", (customer_id,)).fetchone()


def segment_of(con, customer_id):
    row = con.execute("SELECT segment_name, is_premium FROM seg WHERE customer_id = ?", (customer_id,)).fetchone()
    return (row["segment_name"], bool(row["is_premium"])) if row else ("Not segmented yet", False)
