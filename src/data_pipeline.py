"""Part A: raw-data audit, cleaning, SQLite load and checkpoint validation.

Run from the repository root:
    python -m src.data_pipeline
"""
from pathlib import Path
import json
import re
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"

TABLES = ["customers", "products", "orders"]

# Explicit schema so keys, NOT NULLs and CHECKs are enforced by SQLite itself.
# delivery_minutes is deliberately nullable: a cancelled order was never delivered.
DDL = {
    "customers": """CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        city        TEXT NOT NULL,
        signup_date TEXT NOT NULL,
        age         INTEGER NOT NULL CHECK (age BETWEEN 10 AND 100),
        gender      TEXT NOT NULL CHECK (gender IN ('male','female','unknown')))""",
    "products": """CREATE TABLE products (
        product_id   TEXT PRIMARY KEY,
        product_name TEXT NOT NULL,
        category     TEXT NOT NULL,
        price_inr    REAL NOT NULL CHECK (price_inr > 0))""",
    "orders": """CREATE TABLE orders (
        order_id         TEXT PRIMARY KEY,
        customer_id      TEXT NOT NULL REFERENCES customers(customer_id),
        product_id       TEXT NOT NULL REFERENCES products(product_id),
        quantity         INTEGER NOT NULL CHECK (quantity > 0),
        order_ts         TEXT NOT NULL,
        status           TEXT NOT NULL CHECK (status IN ('completed','cancelled')),
        discount_pct     REAL NOT NULL CHECK (discount_pct BETWEEN 0 AND 100),
        order_amount     REAL NOT NULL CHECK (order_amount > 0),
        delivery_minutes INTEGER CHECK (delivery_minutes IS NULL OR delivery_minutes >= 0))""",
}

# Values observed from this repo's deterministic data (generate_data.py, seed 42) with the
# cleaning rules below. They are regression checkpoints for THIS implementation. They were
# NOT copied from the capstone brief, which was not available in the workspace.
OBSERVED = {
    "customers_rows": 1200,
    "products_rows": 40,
    "orders_raw_rows": 13084,
    "orders_clean_rows": 12309,
    "orders_completed": 11110,
    "orders_cancelled": 1199,
    "completed_revenue": 2998668.80,
}


def _read_raw_strings(name):
    """Read a raw CSV as strings with NO NA inference, so blanks stay visible."""
    return pd.read_csv(RAW / f"{name}.csv", dtype=str, keep_default_na=False)


def _blank_to_nan(df):
    return df.replace(r"^\s*$", np.nan, regex=True)


# --------------------------------------------------------------------------- audit
def audit(verbose=True):
    """Report shape, dtypes, blank/missing counts and percentages for the four raw tables."""
    report = {}
    for name in ["customers", "products", "orders", "tickets"]:
        naive = pd.read_csv(RAW / f"{name}.csv")            # what a naive load would infer
        strings = _blank_to_nan(_read_raw_strings(name))     # blanks -> NaN for counting
        missing = strings.isna().sum()
        report[name] = {
            "shape": list(naive.shape),
            "dtypes": {c: str(t) for c, t in naive.dtypes.items()},
            "missing_count": {c: int(v) for c, v in missing.items()},
            "missing_pct": {c: round(float(v) / len(strings) * 100, 2) for c, v in missing.items()},
        }
        if verbose:
            print(f"\n{name}: shape={naive.shape}")
            print(pd.DataFrame({"dtype": naive.dtypes.astype(str), "missing_count": missing,
                                "missing_pct": (missing / len(strings) * 100).round(2)}).to_string())

    orders = _read_raw_strings("orders")
    cust = _read_raw_strings("customers")
    amount = pd.to_numeric(orders.order_amount, errors="coerce")
    delivery = pd.to_numeric(orders.delivery_minutes, errors="coerce")
    age = pd.to_numeric(cust.age, errors="coerce")
    report["orders_checks"] = {
        "rows": len(orders),
        "exact_duplicate_rows": int(orders.duplicated().sum()),
        "pending_rows": int(orders.status.eq("pending").sum()),
        "distinct_order_id": int(orders.order_id.nunique()),
        "non_numeric_or_blank_amount": int(amount.isna().sum()),
        "non_numeric_amount_tokens": sorted(set(orders.order_amount[amount.isna() & orders.order_amount.str.strip().ne("")])),
        "negative_delivery": int(delivery.lt(0).sum()),
        "blank_delivery": int(orders.delivery_minutes.str.strip().eq("").sum()),
        "blank_delivery_on_completed": int((orders.delivery_minutes.str.strip().eq("") & orders.status.eq("completed")).sum()),
        "blank_delivery_on_cancelled": int((orders.delivery_minutes.str.strip().eq("") & orders.status.eq("cancelled")).sum()),
    }
    report["customers_checks"] = {
        "blank_age": int(cust.age.str.strip().eq("").sum()),
        "age_outside_10_100": int(((age < 10) | (age > 100)).sum()),
        "raw_city_spellings": int(cust.city.nunique()),
        "clean_city_spellings": int(cust.city.str.strip().str.title().nunique()),
        "raw_gender_spellings": int(cust.loc[cust.gender.str.strip().ne(""), "gender"].nunique()),
        "blank_gender": int(cust.gender.str.strip().eq("").sum()),
    }
    if verbose:
        print("\nOrders checks:", json.dumps(report["orders_checks"], indent=1))
        print("Customers checks:", json.dumps(report["customers_checks"], indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# -------------------------------------------------------------------------- cleaning
def clean_customers(raw):
    """Return (clean_df, change_counts). `raw` is the all-string raw table."""
    cust = raw.copy()
    changes = {}
    city = cust.city.str.strip().str.title()
    changes["city_rows_changed"] = int((city != raw.city).sum())
    cust["city"] = city

    gender = cust.gender.str.strip().str.lower().map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}).fillna("unknown")
    changes["gender_rows_changed"] = int((gender != raw.gender).sum())
    cust["gender"] = gender

    age = pd.to_numeric(cust.age, errors="coerce")
    invalid = ~age.between(10, 100) & age.notna()
    changes["age_impossible_set_missing"] = int(invalid.sum())
    age[invalid] = np.nan
    changes["age_missing_before_fill"] = int(age.isna().sum())
    # Median age of the customer's own city, via groupby + transform (per the brief).
    age = age.fillna(age.groupby(cust["city"]).transform("median"))
    assert age.notna().all(), "a city had no valid ages to take a median from"
    cust["age"] = age.round().astype(int)

    cust["signup_date"] = pd.to_datetime(cust.signup_date).dt.strftime("%Y-%m-%d")
    return cust, changes


def clean_orders(raw, products):
    """Apply the order-cleaning sequence in the documented order. Returns (df, counts)."""
    changes = {"raw_rows": len(raw)}
    orders = raw.copy()

    # 1. order_ts -> datetime
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    # 2. exact duplicate rows
    before = len(orders)
    orders = orders.drop_duplicates()
    changes["exact_duplicates_removed"] = before - len(orders)
    # 3. latest record per order_id (a pending row precedes its final row in time)
    before = len(orders)
    orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last").copy()
    changes["superseded_order_id_rows_removed"] = before - len(orders)
    changes["clean_rows"] = len(orders)

    # 4. order_amount -> numeric ("N/A", "error" and blanks become NaN)
    orders["quantity"] = pd.to_numeric(orders.quantity, errors="coerce")
    orders["discount_pct"] = pd.to_numeric(orders.discount_pct, errors="coerce").fillna(0)
    orders["order_amount"] = pd.to_numeric(orders.order_amount, errors="coerce")
    # 5. fill invalid amounts: price * quantity * (1 - discount/100)
    price = products.set_index("product_id").price_inr.astype(float)
    bad = orders.order_amount.isna()
    changes["invalid_amounts_filled"] = int(bad.sum())
    orders.loc[bad, "order_amount"] = (orders.loc[bad, "product_id"].map(price)
                                       * orders.loc[bad, "quantity"]
                                       * (1 - orders.loc[bad, "discount_pct"] / 100)).round(2)
    # 6. negative delivery times -> missing
    orders["delivery_minutes"] = pd.to_numeric(orders.delivery_minutes, errors="coerce")
    neg = orders.delivery_minutes < 0
    changes["negative_delivery_set_missing"] = int(neg.sum())
    orders.loc[neg, "delivery_minutes"] = np.nan
    # 7. completed orders only: fill missing delivery with the product-category median
    orders["_category"] = orders.product_id.map(products.set_index("product_id").category)
    completed = orders.status.eq("completed")
    cat_median = orders[completed].groupby("_category").delivery_minutes.median()
    fill = completed & orders.delivery_minutes.isna()
    changes["completed_delivery_filled_with_category_median"] = int(fill.sum())
    orders.loc[fill, "delivery_minutes"] = orders.loc[fill, "_category"].map(cat_median)
    # 8. cancelled orders keep a missing delivery time (never delivered; nothing to impute)
    cancelled = orders.status.eq("cancelled")
    changes["cancelled_delivery_left_missing"] = int((cancelled & orders.delivery_minutes.isna()).sum())
    assert not orders.loc[cancelled, "delivery_minutes"].notna().any(), \
        "cancelled orders must not carry a delivery time"
    orders["delivery_minutes"] = orders.delivery_minutes.round().astype("Int64")
    orders["quantity"] = orders.quantity.astype(int)
    orders["order_ts"] = orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    orders = orders.drop(columns="_category").reset_index(drop=True)
    return orders, changes


def load_db(frames):
    """(Re)create the SQLite tables from explicit DDL and load the clean frames."""
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA foreign_keys = ON")
        for name in reversed(TABLES):
            con.execute(f"DROP TABLE IF EXISTS {name}")
        for name in TABLES:
            con.execute(DDL[name])
            frames[name].to_sql(name, con, if_exists="append", index=False)


def clean_data():
    """Clean all tables, write data/clean/*.csv, data/instamart.db and a change report."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    cust_raw = _read_raw_strings("customers")
    orders_raw = _read_raw_strings("orders")
    products = pd.read_csv(RAW / "products.csv")

    cust, cust_changes = clean_customers(cust_raw)
    orders, order_changes = clean_orders(orders_raw, products)
    frames = {"customers": cust, "products": products, "orders": orders}
    for name, df in frames.items():
        df.to_csv(CLEAN / f"{name}.csv", index=False)
    load_db(frames)

    OUT.mkdir(exist_ok=True)
    (OUT / "cleaning_report.json").write_text(
        json.dumps({"customers": cust_changes, "orders": order_changes}, indent=2), encoding="utf-8")
    print("Customer cleaning:", cust_changes)
    print("Order cleaning:", order_changes)
    return frames


# ------------------------------------------------------------------------ validation
def run_checkpoints():
    """Return a list of (name, passed, detail). Reads the CSVs and the SQLite file."""
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    c = pd.read_csv(CLEAN / "customers.csv")
    o = pd.read_csv(CLEAN / "orders.csv")
    p = pd.read_csv(CLEAN / "products.csv")
    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA foreign_keys = ON")
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        check("1 SQLite has customers, products, orders tables", set(TABLES) <= tables, str(sorted(tables)))
        n = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
        check("2 customers rows == 1200 and CSV == DB", n["customers"] == OBSERVED["customers_rows"] == len(c), str(n))
        check("3 products rows == 40 and CSV == DB", n["products"] == OBSERVED["products_rows"] == len(p), str(n))
        check("4 orders rows == 12309 (distinct raw order_id) and CSV == DB",
              n["orders"] == OBSERVED["orders_clean_rows"] == len(o)
              == _read_raw_strings("orders").order_id.nunique(), str(n))
        st = dict(con.execute("SELECT status, COUNT(*) FROM orders GROUP BY status").fetchall())
        check("5 completed == 11110, cancelled == 1199, no pending",
              st == {"completed": OBSERVED["orders_completed"], "cancelled": OBSERVED["orders_cancelled"]}, str(st))
        rev = con.execute("SELECT ROUND(SUM(order_amount),2) FROM orders WHERE status='completed'").fetchone()[0]
        check("6 completed-order revenue == 2998668.80 (DB and CSV agree)",
              abs(rev - OBSERVED["completed_revenue"]) < 0.005
              and abs(o.loc[o.status == "completed", "order_amount"].sum() - rev) < 0.01, f"{rev}")
        nulls = con.execute("SELECT SUM(delivery_minutes IS NULL AND status='completed'), "
                            "SUM(delivery_minutes IS NULL AND status='cancelled'), "
                            "SUM(delivery_minutes IS NOT NULL AND status='cancelled') FROM orders").fetchone()
        check("7 completed delivery never null; ALL cancelled delivery null (not imputed)",
              nulls == (0, OBSERVED["orders_cancelled"], 0), str(nulls))
        check("8 no negative delivery and no null/non-positive amount",
              con.execute("SELECT COUNT(*) FROM orders WHERE delivery_minutes<0 OR order_amount IS NULL OR order_amount<=0").fetchone()[0] == 0)
        check("9 order_id unique and foreign keys valid",
              o.order_id.is_unique and len(con.execute("PRAGMA foreign_key_check").fetchall()) == 0)
        fmt = con.execute("SELECT COUNT(*) FROM orders WHERE order_ts NOT GLOB "
                          "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'").fetchone()[0]
        check("10 order_ts stored as YYYY-MM-DD HH:MM:SS", fmt == 0 and all(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", s) for s in o.order_ts))
    check("11 six title-cased cities, gender in {male,female,unknown}",
          c.city.nunique() == 6 and (c.city == c.city.str.strip().str.title()).all()
          and set(c.gender) <= {"male", "female", "unknown"}, str(sorted(c.city.unique())))
    check("12 ages all present and within 10-100", c.age.notna().all() and c.age.between(10, 100).all())
    return results


def validate():
    """Print every checkpoint; print exactly ALL CHECKPOINTS PASSED only if all pass."""
    results = run_checkpoints()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail and not ok else ""))
    if all(ok for _, ok, _ in results):
        print("ALL CHECKPOINTS PASSED")
        return True
    print(f"{sum(not ok for _, ok, _ in results)} CHECKPOINT(S) FAILED")
    return False


if __name__ == "__main__":
    audit()
    clean_data()
    raise SystemExit(0 if validate() else 1)
