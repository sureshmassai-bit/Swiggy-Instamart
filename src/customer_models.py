"""Part B: RFM segmentation (K-means + PCA) and a leakage-safe lapse model.

Run from the repository root (needs Part A outputs first):
    python -m src.customer_models
"""
from pathlib import Path
import json
import tempfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"

SEED = 42
RFM_SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")   # RFM snapshot date
LAPSE_CUTOFF = pd.Timestamp("2025-09-01")             # features use orders strictly BEFORE this
LAPSE_END = pd.Timestamp("2025-10-01")                # label window is [cutoff, end)
MIN_LAPSED_F1 = 0.55

NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
           "avg_delivery", "age", "tenure_days"]
CATEGORICAL = ["favourite_category", "city", "gender"]
FEATURES = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
            "avg_delivery", "favourite_category", "city", "age", "gender", "tenure_days"]

SEGMENT_ACTIONS = {
    "Premium": "VIP treatment: early access, premium support desk, protect with proactive retention offers.",
    "Loyal Frequent": "Reward habit: loyalty points and bundles; nudge toward higher-value baskets.",
    "At Risk / Dormant": "Win-back: time-limited discount or reminder before they churn completely.",
    "Casual": "Grow engagement: category recommendations and first-repeat incentives.",
}


def load():
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    return customers, orders


# ------------------------------------------------------------------ RFM + clustering
def build_rfm(orders, snapshot=RFM_SNAPSHOT):
    """RFM for customers with at least one completed order."""
    done = orders[orders.status == "completed"]
    rfm = done.groupby("customer_id").agg(last_order=("order_ts", "max"),
                                          frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum")).reset_index()
    rfm["recency"] = (snapshot - rfm.last_order).dt.days
    return rfm.drop(columns="last_order")[["customer_id", "recency", "frequency", "monetary"]]


def kmeans_diagnostics(X, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def choose_k(diagnostics, candidates=(3, 4, 5)):
    """Best silhouette among the allowed candidates (business reading is documented in README)."""
    sub = diagnostics[diagnostics.k.isin(candidates)]
    return int(sub.loc[sub.silhouette.idxmax(), "k"])


def name_segments(profile):
    """Give each cluster a business name from its profile. Cluster ids are labels, not ranks."""
    p = profile.set_index("segment_id")
    names = {p.monetary.idxmax(): "Premium"}
    rest = p.drop(index=list(names))
    if len(rest):
        loyal = rest.frequency.idxmax()
        names[loyal] = "Loyal Frequent"
        rest = rest.drop(index=loyal)
    if len(rest):
        risk = rest.recency.idxmax()
        names[risk] = "At Risk / Dormant"
        rest = rest.drop(index=risk)
    for i, sid in enumerate(rest.sort_values("monetary", ascending=False).index, 1):
        names[sid] = "Casual" if len(rest) == 1 else f"Casual {i}"
    return names


def rfm_and_segments(k=None, save=True):
    """Returns (rfm_with_segments, diagnostics, profile, pca_explained_variance_ratio)."""
    _, orders = load()
    rfm = build_rfm(orders)
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    diagnostics = kmeans_diagnostics(X)
    k = k or choose_k(diagnostics)

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    rfm["segment_id"] = km.fit_predict(X)
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "size"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"),
                                            monetary=("monetary", "mean")).reset_index()
    names = name_segments(profile)
    profile["segment_name"] = profile.segment_id.map(names)
    profile["action"] = profile.segment_name.map(lambda n: SEGMENT_ACTIONS.get(n, SEGMENT_ACTIONS["Casual"]))
    rfm["segment_name"] = rfm.segment_id.map(names)
    rfm["is_premium"] = rfm.segment_name.eq("Premium")     # exactly the highest-monetary cluster

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    coords = pca.transform(X)
    if save:
        OUT.mkdir(exist_ok=True)
        rfm[["customer_id", "recency", "frequency", "monetary", "segment_id",
             "segment_name", "is_premium"]].round({"monetary": 2}).to_csv(OUT / "customer_segments.csv", index=False)
        diagnostics.assign(selected=diagnostics.k.eq(k)).to_csv(OUT / "kmeans_diagnostics.csv", index=False)
        profile.to_csv(OUT / "segment_profile.csv", index=False)
        rfm[["recency", "frequency", "monetary"]].describe().round(3).to_csv(OUT / "rfm_summary.csv")
        _plot_pca(coords, rfm, pca.explained_variance_ratio_)
    return rfm, diagnostics, profile, pca.explained_variance_ratio_


def _plot_pca(coords, rfm, var):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, grp in rfm.groupby("segment_name"):
        ax.scatter(coords[grp.index, 0], coords[grp.index, 1], s=10, alpha=.6, label=f"{name} (n={len(grp)})")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} of variance)")
    ax.set_title("RFM segments in PCA space (scaled R, F, M)")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pca_segments.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- lapse prediction
def build_lapse_dataset(customers, orders, cutoff=LAPSE_CUTOFF, end=LAPSE_END):
    """Cohort, features and label.

    Cohort: customers with >= 1 completed order BEFORE the cutoff.
    Label : lapsed = 1 if the customer has NO completed order in [cutoff, end).
    Every feature is computed only from orders with order_ts < cutoff (no future information).
    """
    history = orders[orders.order_ts < cutoff]
    done = history[history.status == "completed"]
    cohort = done.groupby("customer_id").agg(last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
                                             monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                             avg_delivery=("delivery_minutes", "mean")).reset_index()
    cohort["recency"] = (cutoff - cohort.last_order).dt.days
    final = history[history.status.isin(["completed", "cancelled"])]
    canc = final.groupby("customer_id").status.apply(lambda s: (s == "cancelled").mean()).rename("cancellation_rate")
    cohort = cohort.merge(canc, on="customer_id", how="left")

    products = pd.read_csv(CLEAN / "products.csv")[["product_id", "category"]]
    fav = (done.merge(products, on="product_id").groupby(["customer_id", "category"]).size().rename("n").reset_index()
           .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
           .drop_duplicates("customer_id").rename(columns={"category": "favourite_category"}))
    cohort = cohort.merge(fav[["customer_id", "favourite_category"]], on="customer_id", how="left")
    cohort = cohort.merge(customers, on="customer_id", how="left")
    cohort["tenure_days"] = (cutoff - pd.to_datetime(cohort.signup_date)).dt.days

    window = orders[(orders.order_ts >= cutoff) & (orders.order_ts < end) & (orders.status == "completed")]
    cohort["lapsed"] = (~cohort.customer_id.isin(window.customer_id)).astype(int)
    cohort = cohort.set_index("customer_id", drop=False).rename_axis(None)
    return cohort[FEATURES], cohort["lapsed"], cohort


def make_pipeline(estimator):
    """All preprocessing lives inside the Pipeline, so it is fitted on training folds only."""
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]), CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", estimator)])


def oversample_train(X_train, y_train, random_state=SEED):
    """Keep every original training row, then duplicate minority rows (with replacement) up to parity."""
    counts = y_train.value_counts()
    minority = counts.idxmin()
    n_extra = int(counts.max() - counts.min())
    extra = resample(X_train[y_train == minority], replace=True, n_samples=n_extra, random_state=random_state)
    X_os = pd.concat([X_train, extra])
    y_os = pd.concat([y_train, pd.Series(minority, index=extra.index)])
    return X_os, y_os


def _metrics(y_true, pred):
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "accuracy": accuracy_score(y_true, pred),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}  # rows=true, cols=pred


def feature_importance(model, X_test, y_test):
    """Which inputs drive the lapse prediction.

    * permutation importance (model-agnostic): drop in test F1 when one ORIGINAL feature is shuffled
      (10 repeats, random_state=42). It is measured on the held-out test set purely for interpretation;
      it plays no part in training or model selection.
    * model-native importance summed back to the original feature: impurity decrease for a random
      forest, |coefficient| on the scaled/one-hot inputs for logistic regression.
    """
    perm = permutation_importance(model, X_test, y_test, scoring="f1", n_repeats=10, random_state=SEED, n_jobs=1)
    names = model.named_steps["prep"].get_feature_names_out()
    est = model.named_steps["model"]
    raw = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_[0])
    native = {}
    for name, value in zip(names, raw):
        col = name.split("__", 1)[1]
        base = next((c for c in CATEGORICAL if col.startswith(c + "_")), col)   # one-hot column -> source feature
        native[base] = native.get(base, 0.0) + float(value)
    total = sum(native.values()) or 1.0
    df = pd.DataFrame({"feature": list(X_test.columns),
                       "permutation_importance_mean": perm.importances_mean,
                       "permutation_importance_std": perm.importances_std})
    df["model_importance_share"] = df.feature.map(lambda f: native.get(f, 0.0) / total)
    return df.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def build_lapse_model(save=True, segments=None):
    customers, orders = load()
    X, y, cohort = build_lapse_dataset(customers, orders)
    assert not X.isna().any().any(), "feature columns must not contain missing values"
    assert not any(c in X.columns for c in ("segment_id", "segment_name", "is_premium", "lapsed")), \
        "segment labels / target must not be features"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    specs = {
        "logistic": (LogisticRegression(max_iter=1000), {"model__C": [.01, .1, 1, 10, 100]}),
        "forest": (RandomForestClassifier(n_estimators=100, random_state=SEED),
                   {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}),
    }
    models, searches = {}, {}
    for name, (est, grid) in specs.items():
        search = GridSearchCV(make_pipeline(est), grid, cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search
        models[name] = {"best_params": search.best_params_, "cv_f1": float(search.best_score_),
                        **_metrics(y_test, search.predict(X_test))}

    baselines = {"majority_class (predict never lapsed)": _metrics(y_test, np.zeros(len(y_test), dtype=int)),
                 "always_lapsed": _metrics(y_test, np.ones(len(y_test), dtype=int))}

    # Model selection uses CROSS-VALIDATION on the training set only; the test set is never used to choose.
    selected_name = max(models, key=lambda m: models[m]["cv_f1"])
    selected = searches[selected_name].best_estimator_
    sel_f1 = models[selected_name]["f1"]
    assert sel_f1 >= MIN_LAPSED_F1, f"selected model lapsed-class test F1 {sel_f1:.3f} < {MIN_LAPSED_F1}"

    # Class-balance experiment: oversample the TRAINING set only; evaluate on the unchanged test set.
    X_os, y_os = oversample_train(X_train, y_train)
    oversampling = {"train_rows_before": len(X_train), "train_rows_after": len(X_os),
                    "train_lapsed_before": int(y_train.sum()), "train_lapsed_after": int(y_os.sum())}
    for name, search in searches.items():
        refit = clone(search.best_estimator_).fit(X_os, y_os)   # same tuned params, no CV on duplicated rows
        oversampling[name] = _metrics(y_test, refit.predict(X_test))

    # Persistence round-trip
    OUT.mkdir(exist_ok=True)
    model_path = OUT / "lapse_model.joblib" if save else Path(tempfile.mkdtemp()) / "lapse_model.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    identical = bool(np.array_equal(selected.predict(X_test), reloaded.predict(X_test)))
    assert identical, "reloaded model predictions differ"

    importance = feature_importance(selected, X_test, y_test)
    if save:
        importance.round(6).to_csv(OUT / "feature_importance.csv", index=False)

    # Illustrative retention list. Premium is a whole-window (Apr-Sep) RFM label, i.e. post-hoc.
    if segments is None:
        segments, _, _, _ = rfm_and_segments(save=save)
    premium = set(segments.loc[segments.is_premium, "customer_id"])
    prob = selected.predict_proba(X_test)[:, list(selected.classes_).index(1)]
    ret = pd.DataFrame({"customer_id": X_test.index, "lapse_probability": prob, "actually_lapsed": y_test.values})
    ret = ret[ret.customer_id.isin(premium)].sort_values("lapse_probability", ascending=False).head(10)
    if save:
        ret.to_csv(OUT / "retention_list.csv", index=False)

    summary = {"cohort_rows": len(cohort), "lapsed": int(y.sum()), "lapse_rate": float(y.mean()),
               "train_rows": len(X_train), "test_rows": len(X_test),
               "latest_feature_order_ts": str(orders[orders.order_ts < LAPSE_CUTOFF].order_ts.max()),
               "models": models, "baselines": baselines, "selected_model": selected_name,
               "selection_rule": "highest 5-fold CV F1 on the training set", "oversampling": oversampling,
               "reload_predictions_identical": identical, "retention_list_rows": len(ret),
               "feature_importance": importance.round(4).to_dict("records")}
    if save:
        (OUT / "lapse_metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    summary["retention_list"] = ret
    return summary


def main():
    rfm, diagnostics, profile, var = rfm_and_segments()
    print("RFM summary:\n", rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())
    print("\nK-means diagnostics (StandardScaler, n_init=10, random_state=42):\n", diagnostics.round(4).to_string(index=False))
    print("\nSelected K:", profile.shape[0])
    print(profile.round(2).drop(columns="action").to_string(index=False))
    print("\nPCA explained variance:", np.round(var, 4), "total", round(float(var.sum()), 4))
    s = build_lapse_model(segments=rfm)
    print(f"\nLapse cohort={s['cohort_rows']} lapsed={s['lapsed']} ({s['lapse_rate']:.1%}) "
          f"train={s['train_rows']} test={s['test_rows']}")
    for k, v in {**s["models"], **s["baselines"]}.items():
        print(f"{k:>38}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} CM={v['confusion_matrix']}")
    print("Selected:", s["selected_model"], "|", s["selection_rule"])
    print("Oversampling:", json.dumps(s["oversampling"], default=float))
    print("Reload predictions identical:", s["reload_predictions_identical"])
    print("\nFeature importance (permutation on test F1; model-native share):")
    print(pd.DataFrame(s["feature_importance"]).to_string(index=False))
    print(s["retention_list"].to_string(index=False))


if __name__ == "__main__":
    main()
