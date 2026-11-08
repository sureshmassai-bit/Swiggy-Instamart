"""Part A: run the seven SQL queries with pandas and cross-check them against pandas.

Run from the repository root:
    python -m src.sql_analysis
"""
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"
OUT = ROOT / "outputs"
RESULTS = OUT / "sql_results"


def load_queries(path=SQL_FILE):
    """Split queries.sql on its '-- Qn:' header comments -> {'Q1': sql, ...}."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^-- (Q\d+):.*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_queries(con):
    queries = load_queries()
    assert sorted(queries) == [f"Q{i}" for i in range(1, 8)], f"expected Q1..Q7, got {sorted(queries)}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, sql in queries.items():
        df = pd.read_sql_query(sql, con)
        df.to_csv(RESULTS / f"{name}.csv", index=False)
        results[name] = df
        print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(df.head(5).to_string(index=False))
    return results


def cross_check(results, orders, customers, products):
    """Recompute each SQL answer independently in pandas. Returns {check: bool}."""
    done = orders[orders.status == "completed"]
    checks = {}

    spend = done[done.order_amount > 500].groupby("customer_id").order_amount.sum()
    q1 = spend[spend > 8000]
    checks["Q1 rows match pandas"] = set(results["Q1"].customer_id) == set(q1.index)

    q2 = (done.merge(customers[["customer_id", "city"]], on="customer_id")
              .merge(products[["product_id", "category"]], on="product_id")
              .groupby(["city", "category"]).order_amount.sum().round(2))
    checks["Q2 revenue by city/category matches pandas"] = np.allclose(
        results["Q2"].set_index(["city", "category"]).revenue.sort_index().values, q2.sort_index().values)
    checks["Q2 grand total == completed revenue"] = abs(results["Q2"].revenue.sum() - done.order_amount.sum()) < 0.05

    checks["Q3 products above mean price"] = set(results["Q3"].product_id) == set(
        products.loc[products.price_inr > products.price_inr.mean(), "product_id"])
    no_done = set(customers.customer_id) - set(done.customer_id)
    checks["Q4 customers with no completed order"] = set(results["Q4"].customer_id) == no_done
    tot = done.groupby("customer_id").order_amount.sum()
    checks["Q5 above-average spenders"] = set(results["Q5"].customer_id) == set(tot[tot > tot.mean()].index)
    q6 = results["Q6"]
    checks["Q6 one customer per row, rank 1 exists in every city"] = (
        q6.customer_id.is_unique and (q6.groupby("city").spend_rank.min() == 1).all()
        and (q6.groupby("city").row_num.max() == q6.groupby("city").size()).all())
    q7 = results["Q7"]
    checks["Q7 cumulative total == completed revenue and is non-decreasing"] = (
        abs(q7.cumulative_revenue.iloc[-1] - done.order_amount.sum()) < 0.05 and q7.cumulative_revenue.is_monotonic_increasing)
    return checks


def pandas_analysis(orders, customers, products):
    done = orders[orders.status == "completed"].copy()
    done = (done.merge(customers[["customer_id", "city"]], on="customer_id")
                .merge(products[["product_id", "category"]], on="product_id"))

    pivot = pd.pivot_table(done, index="city", columns="category", values="order_amount",
                           aggfunc="sum", margins=True, margins_name="Total").round(2)
    print("\nQ2 rebuilt with pd.pivot_table(margins=True):\n", pivot.to_string())

    city_stats = done.groupby("city").order_amount.agg(["sum", "mean", "count"]).round(2)
    print("\nCity-level completed order amount (sum, mean, count):\n", city_stats.to_string())

    x = done.order_amount
    q1, q3 = x.quantile(.25), x.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = {"mean": x.mean(), "median": x.median(), "skewness": x.skew(),
             "Q1": q1, "Q3": q3, "IQR": iqr, "lower_fence": lo, "upper_fence": hi,
             "n_outliers_high": int((x > hi).sum()), "n_outliers_low": int((x < lo).sum())}
    print("\norder_amount distribution (completed orders):")
    for k, v in stats.items():
        print(f"  {k:>16}: {v:,.3f}")

    corr = done[["quantity", "discount_pct", "order_amount", "delivery_minutes"]].astype(float).corr().round(3)
    print("\nCorrelation matrix (completed orders):\n", corr.to_string())
    return pivot, city_stats, stats, corr, done


def write_eda_notes(stats, city_stats, corr, done):
    """Answer the four exploratory questions in <= 200 words (word count is asserted)."""
    top_city = city_stats["sum"].idxmax()
    cat_rev = done.groupby("category").order_amount.sum()
    top_cat = cat_rev.idxmax()
    share = cat_rev.max() / cat_rev.sum() * 100
    text = (
        "**1. Is spend symmetric?** No. Skewness is "
        f"{stats['skewness']:.2f}, mean INR {stats['mean']:.0f} against median INR {stats['median']:.0f}: "
        "a few expensive electronics orders pull the mean up, so the median describes a typical order better.\n\n"
        f"**2. Are there outliers?** Yes: {stats['n_outliers_high']} orders exceed the IQR upper fence "
        f"(INR {stats['upper_fence']:.0f}). They are genuine high-value purchases, not errors, so they are kept.\n\n"
        f"**3. Where does revenue come from?** {top_city} earns the most overall (INR {city_stats['sum'].max():,.0f}); "
        f"{top_cat} contributes {share:.0f}% of revenue.\n\n"
        "**4. What drives order value?** Linear relationships are weak: quantity "
        f"(r={corr.loc['quantity', 'order_amount']:.2f}), discount (r={corr.loc['discount_pct', 'order_amount']:.2f}) "
        f"and delivery time (r={corr.loc['delivery_minutes', 'order_amount']:.2f}) barely track amount. "
        "Product price, not captured here, likely dominates.\n\n"
        "Correlation does not establish causation."
    )
    words = len(re.sub(r"[*]", "", text).split())
    assert words <= 200, f"EDA notes are {words} words; limit is 200"
    OUT.mkdir(exist_ok=True)
    (OUT / "eda_notes.md").write_text(text + "\n", encoding="utf-8")
    print(f"\nEDA notes ({words} words) written to outputs/eda_notes.md")
    return text


def main():
    with sqlite3.connect(DB) as con:
        results = run_queries(con)
        orders = pd.read_sql_query("SELECT * FROM orders", con)
        customers = pd.read_sql_query("SELECT * FROM customers", con)
        products = pd.read_sql_query("SELECT * FROM products", con)
    checks = cross_check(results, orders, customers, products)
    print("\nSQL vs pandas cross-checks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    pivot, city_stats, stats, corr, done = pandas_analysis(orders, customers, products)
    sql_q2 = results["Q2"].pivot(index="city", columns="category", values="revenue")
    same = np.allclose(sql_q2.values, pivot.drop(index="Total", columns="Total").reindex_like(sql_q2).values, equal_nan=True)
    print(f"\nPivot (margins=True) body equals SQL Q2: {'PASS' if same else 'FAIL'}")
    write_eda_notes(stats, city_stats, corr, done)
    return all(checks.values()) and same


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
