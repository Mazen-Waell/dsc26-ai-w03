from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"

FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "created_at", "promised_slot",
    "cod_amount", "n_items", "address_text",
]

HISTORY_COLUMNS = ["customer_prior_parcels", "customer_prior_fail_rate"]
CATEGORICAL = ["area", "merchant_id", "promised_slot"]
NUMERIC = ["n_items", "customer_prior_parcels", "customer_prior_fail_rate"]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame({
        "cust": df["customer_id"].to_numpy(),
        "t": df["created_at"].to_numpy(),
        "f": df["failed"].to_numpy(),
    })
    groups = (work.groupby(["cust", "t"], sort=True, dropna=False)
              .agg(n=("f", "size"), s=("f", "sum"))
              .reset_index())
    by_cust = groups.groupby("cust", sort=False, dropna=False)
    groups["prior_n"] = by_cust["n"].cumsum() - groups["n"]
    groups["prior_s"] = by_cust["s"].cumsum() - groups["s"]

    merged = work.merge(groups[["cust", "t", "prior_n", "prior_s"]],
                        on=["cust", "t"], how="left")
    prior_n = merged["prior_n"].to_numpy()
    prior_s = merged["prior_s"].to_numpy(dtype=float)
    rate = np.where(prior_n > 0, prior_s / np.maximum(prior_n, 1), np.nan)

    out = df.copy()
    out["customer_prior_parcels"] = prior_n.astype(int)
    out["customer_prior_fail_rate"] = rate
    return out


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    ordered = df.sort_values(time_col, kind="stable")
    n_test = math.ceil(len(df) * test_fraction)
    boundary = ordered[time_col].iloc[len(df) - n_test]
    test = df[df[time_col] >= boundary]
    train = df[df[time_col] < boundary]
    return train, test


def _as_object(X):
    return X.astype(object).where(X.notna(), np.nan)


def _log1p(X):
    return np.log1p(np.clip(X, 0, None))


def _fill_text(X):
    return pd.Series(X).fillna("").astype(str)


def _empty_flag(X):
    col = X.iloc[:, 0].fillna("").astype(str).str.strip()
    return (col == "").to_numpy(dtype=float).reshape(-1, 1)


def build_pipeline(seed: int = 0) -> Pipeline:
    categorical = Pipeline([
        ("to_object", FunctionTransformer(_as_object)),
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cod = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
        ("log", FunctionTransformer(_log1p)),
        ("scale", StandardScaler()),
    ])
    address = Pipeline([
        ("fill", FunctionTransformer(_fill_text)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)),
    ])

    prep = ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("num", numeric, NUMERIC),
        ("cod", cod, ["cod_amount"]),
        ("addr", address, "address_text"),
        ("addr_empty", FunctionTransformer(_empty_flag), ["address_text"]),
    ], remainder="drop")

    return Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y = test_df[TARGET]
    proba = model.predict_proba(test_df)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "base_rate": float(y.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    model = build_pipeline(seed).fit(train_df, train_df[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(train_df, train_df[TARGET])
    model_scores = evaluate(model, test_df)
    dummy_scores = evaluate(dummy, test_df)
    return {
        "model": model_scores,
        "dummy": dummy_scores,
        "lift_pr_auc": model_scores["pr_auc"] - dummy_scores["pr_auc"],
    }
