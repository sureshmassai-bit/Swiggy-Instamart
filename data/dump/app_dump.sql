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
INSERT INTO "users" VALUES(1,'admin','admin',NULL,'Support Admin','admin@instamart.local','pbkdf2:sha256:600000$kL1pJNBIoao9DCzL$1bc79f193eae686360cf01016323ecfa60bd738d78caf0b46a9abc77a07163db',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(2,'user01','user','C0002','Customer C0002','user01@instamart.local','pbkdf2:sha256:600000$FTs5gURvNmOKy2dH$40c81db5179470d3f7a630e6518c3e591b1578119c3ab3497fe8f7dfa8007eed',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(3,'user02','user','C0009','Customer C0009','user02@instamart.local','pbkdf2:sha256:600000$HvX6GRgFeUwXD0bh$9941f1852e526b06f4a7f28c35d9de3792678e315ce8c7d09fa36d0eb0179f38',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(4,'user03','user','C0014','Customer C0014','user03@instamart.local','pbkdf2:sha256:600000$eivt4dYO2KMz9xEO$41274bb648232651545f38249909af2cdbe82f9044a9c0086bcd378e4c6b0f91',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(5,'user04','user','C0003','Customer C0003','user04@instamart.local','pbkdf2:sha256:600000$sS4FWv3OVKZSIw8x$e433ae32b899651fe7bf675ec8ef53a62be561cc59165e202625b492e03778b2',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(6,'user05','user','C0007','Customer C0007','user05@instamart.local','pbkdf2:sha256:600000$RNQ77pMaOJFZXp7S$65584bf9fc5efc653360bb784a8ba19f0c1c51084bb24466ca77950e48503638',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(7,'user06','user','C0008','Customer C0008','user06@instamart.local','pbkdf2:sha256:600000$nN2dSwz5kHUiD3sD$3d955b138850528d76d793a918e36e52c78502e26af5a6a98776a897164919db',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(8,'user07','user','C0001','Customer C0001','user07@instamart.local','pbkdf2:sha256:600000$v3ctesO23zCgcm9D$0d31fbae0b44858e798fecdd63ea5b80d73b2925ba04d40e5bebd5d4bfe7e803',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(9,'user08','user','C0004','Customer C0004','user08@instamart.local','pbkdf2:sha256:600000$l6cIsLgv433Ep4aF$1fe3c43bcd016ff103268790b7c753d740a6d6cba42c0b223d613eefd839169a',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(10,'user09','user','C0005','Customer C0005','user09@instamart.local','pbkdf2:sha256:600000$xCQcg2Q6m13WDk5J$0d3c01879e5407fac125ccc5f44cc86a18d8ccf6abdb6bbac19c75862e97a3c7',1,'2026-11-07 18:07:06');
INSERT INTO "users" VALUES(11,'user10','user','C0012','Customer C0012','user10@instamart.local','pbkdf2:sha256:600000$RQ3ZFxFdKljYiePr$6bd18a203e28891ac29878eaa239a7447f60d909c7200e6fa277a4e5ac17b47c',1,'2026-11-07 18:07:06');
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
