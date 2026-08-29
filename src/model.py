"""Train / load the crop-recommendation model and derive per-crop agronomic stats.

The model is a ``RandomForestClassifier`` inside a ``Pipeline`` (scaler + forest).
Alongside it we compute, for every crop, the distribution of each feature
(min / low-quartile / median / high-quartile / max).  Those ranges power the
rule-based **management advisor**, which compares a field's current readings to
the crop's learned comfort zone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import load_dataset


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def compute_crop_stats(df: pd.DataFrame) -> dict:
    """Per-crop, per-feature summary statistics used by the advisor & UI."""
    stats: dict[str, dict] = {}
    for crop, group in df.groupby(config.TARGET):
        stats[crop] = {"count": int(len(group))}
        for feat in config.FEATURES:
            s = group[feat]
            stats[crop][feat] = {
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
    return stats


def train(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train the model, persist artifacts, and return a metrics dict."""
    df, source = load_dataset()

    X = df[config.FEATURES]
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_samples": int(len(df)),
        "n_classes": int(y.nunique()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "source": source,
        "features": config.FEATURES,
        "classes": sorted(y.unique().tolist()),
    }

    # Feature importances (averaged across the forest).
    forest: RandomForestClassifier = model.named_steps["forest"]
    metrics["feature_importances"] = {
        feat: float(imp) for feat, imp in zip(config.FEATURES, forest.feature_importances_)
    }

    # Persist everything.
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, config.MODEL_PATH)
    config.STATS_PATH.write_text(json.dumps(compute_crop_stats(df), indent=2))
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


# --------------------------------------------------------------------------
# Loading / prediction
# --------------------------------------------------------------------------
def load_model():
    """Load the trained pipeline, training it first if no artifact exists."""
    import joblib

    if not config.MODEL_PATH.exists():
        train()
    return joblib.load(config.MODEL_PATH)


def load_stats() -> dict:
    if not config.STATS_PATH.exists():
        train()
    return json.loads(config.STATS_PATH.read_text())


def load_metrics() -> dict:
    if not config.METRICS_PATH.exists():
        train()
    return json.loads(config.METRICS_PATH.read_text())


def predict(model, features: dict, top_k: int = 3) -> list[tuple[str, float]]:
    """Return the ``top_k`` ``(crop, probability)`` pairs, most likely first."""
    row = pd.DataFrame([[features[f] for f in config.FEATURES]], columns=config.FEATURES)
    proba = model.predict_proba(row)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1][:top_k]
    return [(str(classes[i]), float(proba[i])) for i in order]
