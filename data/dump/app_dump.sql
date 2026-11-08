BEGIN TRANSACTION;
CREATE TABLE chat_log (
    chat_no     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    customer_id TEXT NOT NULL,
    turns       INTEGER NOT NULL CHECK (turns BETWEEN 0 AND 99),
    outcome     TEXT NOT NULL CHECK (outcome IN ('solved','ticket')),
    created_at  TEXT NOT NULL);
CREATE TABLE login_failures (
    username TEXT NOT NULL COLLATE NOCASE,
    ts       TEXT NOT NULL);
INSERT INTO "login_failures" VALUES('admin','2026-11-08 18:54:13');
CREATE TABLE ticket_messages (
    msg_no      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_no   INTEGER NOT NULL REFERENCES tickets(ticket_no),
    author_id   INTEGER NOT NULL REFERENCES users(user_id),
    author_role TEXT NOT NULL CHECK (author_role IN ('admin','user')),
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL);
CREATE TABLE tickets (
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
    updated_at  TEXT NOT NULL, handled_by TEXT NOT NULL DEFAULT 'human' CHECK (handled_by IN ('bot','human')), bot_replies INTEGER NOT NULL DEFAULT 0);
CREATE TABLE users (
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
INSERT INTO "users" VALUES(1,'admin','admin',NULL,'Support Admin','admin@instamart.local','pbkdf2:sha256:600000$IRDxtkn7Zm74kLge$2791071ae37682e780a0b5cd385dd8f7446fa90d7c64fdfcc55e6d756ab2957a',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(2,'user01','user','C0002','Customer C0002','user01@instamart.local','pbkdf2:sha256:600000$wcniMwK8CJcohKEq$ecb83f8da5bc401391006498146ce7b21ad0dd26f09d3182a5dac31a9bba1977',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(3,'user02','user','C0009','Customer C0009','user02@instamart.local','pbkdf2:sha256:600000$57YWduX5j9RZAB7q$845c7a9f2b307385d5ff74beded5e48dcbccaaf006811139b2bcc378bcc5d6ac',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(4,'user03','user','C0014','Customer C0014','user03@instamart.local','pbkdf2:sha256:600000$7NTQnCNYbCCNZVg8$d8f100ebb18256346d6176c14bfa59de7b3ea76ed0f2c0bf79ec90f9db180bd2',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(5,'user04','user','C0003','Customer C0003','user04@instamart.local','pbkdf2:sha256:600000$SDs2wS6VpZY7sTHU$dbf92cb45aa996b07aae55f196d9fd64fe7b959f5bf0191e7239a7b0f961d18c',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(6,'user05','user','C0007','Customer C0007','user05@instamart.local','pbkdf2:sha256:600000$fBMsiZeqyJtIxhB8$93c2a98bc91c53ff3c15f7c6f9ff2a92e1f38cef8b1c610fb17b8988af469257',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(7,'user06','user','C0008','Customer C0008','user06@instamart.local','pbkdf2:sha256:600000$5OoWre4yEZwex8z8$cf5e49041ca8592f45e924ee7f1a570dae11d8fd01a662fb70e9e2f803ec47f0',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(8,'user07','user','C0001','Customer C0001','user07@instamart.local','pbkdf2:sha256:600000$WPtdz6kVJeGJeeWu$42746e9026fd08d2cab8bc2a9833f559bc1e24837a26bdf894ff3bfa946318a9',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(9,'user08','user','C0004','Customer C0004','user08@instamart.local','pbkdf2:sha256:600000$Nkd4uwxCvMR0xItv$4a17b60704105ee74e042743a7efed78ccef3b02e6feb8d58a48c07b5ed2c681',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(10,'user09','user','C0005','Customer C0005','user09@instamart.local','pbkdf2:sha256:600000$91a9tSLnDaLp2zwG$9fd753f3ee9952dd9b35f60236075797b674a381dc6df00ce97e2cbea31afcc4',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(11,'user10','user','C0012','Customer C0012','user10@instamart.local','pbkdf2:sha256:600000$7F8V8qK1KHGgxbkB$c4832cdd8b52e0f2233cdbc43c9ec2996651a3c545d97ed53a58a325dca3b27e',1,'2026-11-07 18:07:06');
CREATE TABLE web_customers (
    customer_id TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    city        TEXT NOT NULL,
    age         INTEGER CHECK (age IS NULL OR age BETWEEN 10 AND 100),
    gender      TEXT NOT NULL CHECK (gender IN ('male','female','unknown')),
    created_at  TEXT NOT NULL);
CREATE TABLE web_orders (
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
CREATE INDEX ix_tickets_customer ON tickets(customer_id);
CREATE INDEX ix_orders_customer ON web_orders(customer_id);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('users',11);
COMMIT;
