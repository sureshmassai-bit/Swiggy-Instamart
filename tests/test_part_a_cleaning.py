"""Part A: data-cleaning rules and database checkpoints."""
import sqlite3

import numpy as np
import pandas as pd

from src import data_pipeline as dp


def _customers(rows):
    return pd.DataFrame(rows, columns=["customer_id", "city", "signup_date", "age", "gender"], dtype=str)


def test_customer_rules():
    raw = _customers([
        ["C1", " chennai ", "2024-01-05", "30", "M"],
        ["C2", "CHENNAI", "2024-01-06", "40", "female"],
        ["C3", "Chennai", "2024-01-07", "999", "x"],      # impossible age -> missing -> city median (35)
        ["C4", "mumbai", "2024-02-01", "", ""],           # blank age -> Mumbai median
        ["C5", "MUMBAI", "2024-02-02", "20", "FEMALE"],
        ["C6", "Mumbai", "2024-02-03", "5", "m"],         # age 5 is outside 10-100
    ])
    out, changes = dp.clean_customers(raw)
    assert set(out.city) == {"Chennai", "Mumbai"}
    assert list(out.gender) == ["male", "female", "unknown", "unknown", "female", "male"]
    assert out.loc[2, "age"] == 35                         # median(30, 40) of Chennai, not the global median
    assert out.loc[3, "age"] == 20 and out.loc[5, "age"] == 20   # Mumbai median of the only valid age (20)
    assert out.age.between(10, 100).all() and out.age.notna().all()
    assert changes["age_impossible_set_missing"] == 2
    assert str(out.signup_date.iloc[0]) == "2024-01-05"


def _products():
    return pd.DataFrame({"product_id": ["P1", "P2"], "category": ["Dairy", "Snacks"], "price_inr": [100, 50]})


def _orders(rows):
    cols = ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
            "discount_pct", "order_amount", "delivery_minutes"]
    return pd.DataFrame(rows, columns=cols, dtype=str)


def test_order_cleaning_sequence():
    t = "2025-05-01 10:00:00"
    raw = _orders([
        ["ORD-1", "C1", "P1", "2", t, "completed", "10", "180.0", "30"],
        ["ORD-1", "C1", "P1", "2", t, "completed", "10", "180.0", "30"],                    # exact duplicate
        ["ORD-2", "C1", "P1", "1", "2025-05-02 09:55:00", "pending", "0", "100.0", ""],     # superseded by the next row
        ["ORD-2", "C1", "P1", "1", "2025-05-02 10:00:00", "completed", "0", "100.0", "50"],
        ["ORD-3", "C1", "P2", "3", t, "completed", "20", "N/A", "20"],                      # invalid amount -> 50*3*0.8
        ["ORD-4", "C1", "P1", "1", t, "completed", "0", "100.0", "-10"],                    # negative delivery -> missing -> filled
        ["ORD-5", "C1", "P1", "1", t, "completed", "0", "100.0", ""],                       # missing delivery -> category median
        ["ORD-6", "C1", "P1", "1", t, "cancelled", "0", "100.0", ""],                       # cancelled: stays missing
        ["ORD-7", "C1", "P2", "1", t, "completed", "0", "error", "10"],                     # 'error' amount -> 50
    ])
    out, ch = dp.clean_orders(raw, _products())
    o = out.set_index("order_id")
    assert list(out.order_id) == sorted(set(raw.order_id))            # unique, one row per order
    assert ch["exact_duplicates_removed"] == 1 and ch["superseded_order_id_rows_removed"] == 1
    assert o.loc["ORD-2", "status"] == "completed"                    # latest record wins
    assert o.loc["ORD-3", "order_amount"] == 120.0 and o.loc["ORD-7", "order_amount"] == 50.0
    assert ch["invalid_amounts_filled"] == 2 and ch["negative_delivery_set_missing"] == 1
    # Dairy completed deliveries after negatives are removed: 30, 50 -> median 40
    assert o.loc["ORD-4", "delivery_minutes"] == 40 and o.loc["ORD-5", "delivery_minutes"] == 40
    assert pd.isna(o.loc["ORD-6", "delivery_minutes"]), "cancelled order must NOT get an invented delivery time"
    assert ch["cancelled_delivery_left_missing"] == 1
    assert o.order_ts.str.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d").all()


def test_database_checkpoints_all_pass():
    results = dp.run_checkpoints()
    assert len(results) >= 8
    failed = [n for n, ok, _ in results if not ok]
    assert not failed, failed


def test_cancelled_delivery_is_null_in_csv_and_db():
    csv = pd.read_csv(dp.CLEAN / "orders.csv")
    assert csv.loc[csv.status == "cancelled", "delivery_minutes"].isna().all()
    assert csv.loc[csv.status == "completed", "delivery_minutes"].notna().all()
    with sqlite3.connect(dp.DB) as con:
        assert con.execute("SELECT COUNT(*) FROM orders WHERE status='cancelled' AND delivery_minutes IS NOT NULL").fetchone()[0] == 0


def test_schema_enforces_keys():
    with sqlite3.connect(dp.DB) as con:
        con.execute("PRAGMA foreign_keys = ON")
        pk = {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})") if r[5]] for t in dp.TABLES}
        assert pk == {"customers": ["customer_id"], "products": ["product_id"], "orders": ["order_id"]}
        try:
            con.execute("INSERT INTO orders VALUES ('ORD-X','NOPE','P001',1,'2025-01-01 00:00:00','completed',0,10,5)")
            raise AssertionError("foreign key was not enforced")
        except sqlite3.IntegrityError:
            pass


def test_pipeline_is_reproducible():
    """Cleaning the same raw data twice gives identical frames."""
    raw = dp._read_raw_strings("orders")
    prod = pd.read_csv(dp.RAW / "products.csv")
    a, _ = dp.clean_orders(raw, prod)
    b, _ = dp.clean_orders(raw, prod)
    pd.testing.assert_frame_equal(a, b)
    assert not np.any(a.order_id.duplicated())
