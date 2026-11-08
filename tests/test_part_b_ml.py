"""Part B: segmentation sanity and data-leakage guarantees for the lapse model."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import customer_models as cm


def _data():
    return cm.load()


def test_features_use_only_pre_cutoff_orders():
    customers, orders = _data()
    X1, y1, _ = cm.build_lapse_dataset(customers, orders)
    # Tamper with EVERYTHING dated on/after the cutoff: huge amounts, slow deliveries, cancelled status.
    tampered = orders.copy()
    post = tampered.order_ts >= cm.LAPSE_CUTOFF
    tampered.loc[post, "order_amount"] *= 100
    tampered.loc[post, "delivery_minutes"] = 999
    tampered.loc[post, "discount_pct"] = 99
    X2, _, _ = cm.build_lapse_dataset(customers, tampered)
    pd.testing.assert_frame_equal(X1, X2)                              # features are blind to the future
    # Removing the future entirely leaves the features unchanged and makes every customer 'lapsed'.
    X3, y3, _ = cm.build_lapse_dataset(customers, orders[orders.order_ts < cm.LAPSE_CUTOFF])
    pd.testing.assert_frame_equal(X1, X3)
    assert y3.eq(1).all() and y1.mean() < 1


def test_feature_contract():
    customers, orders = _data()
    X, y, _ = cm.build_lapse_dataset(customers, orders)
    assert list(X.columns) == cm.FEATURES and len(cm.FEATURES) == 11
    assert not X.isna().any().any()
    assert not {"segment_id", "segment_name", "is_premium", "lapsed"} & set(X.columns)
    assert set(y.unique()) == {0, 1}


def test_preprocessing_is_fitted_on_training_rows_only():
    customers, orders = _data()
    X, y, _ = cm.build_lapse_dataset(customers, orders)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    from sklearn.linear_model import LogisticRegression
    pipe = cm.make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X_tr[cm.NUMERIC].mean().values)
    assert not np.allclose(scaler.mean_, X[cm.NUMERIC].mean().values), "scaler must not have seen the test rows"
    # Unseen categories at predict time must not crash (encoder fitted on train only).
    unseen = X_te.copy()
    unseen["city"] = "Atlantis"
    assert len(pipe.predict(unseen)) == len(X_te)


def test_split_is_stratified_and_reproducible():
    customers, orders = _data()
    X, y, _ = cm.build_lapse_dataset(customers, orders)
    a = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    b = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    assert list(a[1].index) == list(b[1].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.01          # class balance preserved in both splits
    assert len(a[1]) == int(np.ceil(len(X) * .2))         # sklearn rounds the test size up


def test_oversampling_touches_training_data_only():
    customers, orders = _data()
    X, y, _ = cm.build_lapse_dataset(customers, orders)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    X_os, y_os = cm.oversample_train(X_tr, y_tr)
    assert set(X_tr.index) <= set(X_os.index) and len(X_os) > len(X_tr)           # all originals kept
    assert (y_os == 1).sum() == (y_os == 0).sum()                                 # minority raised to parity
    assert (y_os == 0).sum() == (y_tr == 0).sum()                                 # majority untouched
    extra = X_os.iloc[len(X_tr):]
    assert set(extra.index) <= set(X_tr[y_tr == 1].index), "only minority training rows may be duplicated"
    assert not set(extra.index) & set(X_te.index), "no test row may enter the oversampled set"
    X_os2, _ = cm.oversample_train(X_tr, y_tr)
    assert list(X_os.index) == list(X_os2.index)                                  # seeded, reproducible


def test_segments_file_and_premium_rule():
    seg = pd.read_csv(cm.OUT / "customer_segments.csv")
    assert list(seg.columns) == ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    assert seg.customer_id.is_unique and seg.is_premium.dtype == bool
    assert seg.groupby("segment_id").segment_name.nunique().eq(1).all()
    assert seg.groupby("segment_name").segment_id.nunique().eq(1).all(), "names must be unique per cluster"
    assert 3 <= seg.segment_id.nunique() <= 5
    top = seg.groupby("segment_id").monetary.mean().idxmax()
    assert set(seg.loc[seg.is_premium, "segment_id"]) == {top}                   # cluster number is a label, not a rank
    customers, orders = _data()
    assert set(seg.customer_id) == set(orders.loc[orders.status == "completed", "customer_id"])


def test_rfm_matches_definition():
    _, orders = _data()
    rfm = cm.build_rfm(orders).set_index("customer_id")
    done = orders[orders.status == "completed"]
    cid = done.customer_id.iloc[0]
    mine = done[done.customer_id == cid]
    assert rfm.loc[cid, "frequency"] == mine.order_id.nunique()
    assert np.isclose(rfm.loc[cid, "monetary"], mine.order_amount.sum())
    assert rfm.loc[cid, "recency"] == (cm.RFM_SNAPSHOT - mine.order_ts.max()).days


def test_saved_model_meets_requirements():
    m = json.loads((cm.OUT / "lapse_metrics.json").read_text())
    sel = m["models"][m["selected_model"]]
    assert sel["f1"] >= cm.MIN_LAPSED_F1
    assert m["reload_predictions_identical"] and m["latest_feature_order_ts"] < "2025-09-01"
    assert m["oversampling"]["train_rows_after"] > m["oversampling"]["train_rows_before"]
    ret = pd.read_csv(cm.OUT / "retention_list.csv")
    assert list(ret.columns) == ["customer_id", "lapse_probability", "actually_lapsed"] and len(ret) <= 10
    seg = pd.read_csv(cm.OUT / "customer_segments.csv")
    assert set(ret.customer_id) <= set(seg.loc[seg.is_premium, "customer_id"])
    assert ret.lapse_probability.is_monotonic_decreasing
