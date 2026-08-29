"""Load the Kaggle *Crop Recommendation* dataset.

Resolution order (first that succeeds wins):
    1. Local cached CSV  (``data/Crop_recommendation.csv``)
    2. Kaggle API via ``kagglehub`` — used automatically when credentials exist
    3. Public GitHub mirror of the same Kaggle dataset (no credentials needed)

Every path funnels through :func:`_validate` so the rest of the app can rely on
a clean frame with exactly the expected columns.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pandas as pd
import requests

from . import config


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep expected columns, drop nulls, normalise crop labels."""
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df = df[config.FEATURES + [config.TARGET]].copy()
    df = df.dropna()
    df[config.TARGET] = df[config.TARGET].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def _from_kaggle() -> Path | None:
    """Download via kagglehub if it's installed and credentials are configured."""
    try:
        import kagglehub  # optional dependency
    except ImportError:
        return None
    try:
        folder = kagglehub.dataset_download(config.KAGGLE_DATASET)
    except Exception:
        return None
    for csv in Path(folder).rglob("*.csv"):
        return csv
    return None


def _from_mirror() -> pd.DataFrame | None:
    """Fetch the dataset from a public mirror as a last resort."""
    for url in config.DATASET_MIRRORS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
        except Exception:
            continue
    return None


def load_dataset(force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source_description)``.

    The resolved dataset is cached to :data:`config.DATASET_PATH` so subsequent
    runs are instant and fully offline.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.DATASET_PATH.exists() and not force_refresh:
        return _validate(pd.read_csv(config.DATASET_PATH)), "local cache"

    kaggle_csv = _from_kaggle()
    if kaggle_csv is not None:
        df = _validate(pd.read_csv(kaggle_csv))
        df.to_csv(config.DATASET_PATH, index=False)
        return df, "Kaggle API"

    mirror_df = _from_mirror()
    if mirror_df is not None:
        df = _validate(mirror_df)
        df.to_csv(config.DATASET_PATH, index=False)
        return df, "public mirror (Kaggle dataset)"

    raise RuntimeError(
        "Could not obtain the dataset from the local cache, Kaggle API, or any "
        "mirror. Check your internet connection, or manually place "
        f"'Crop_recommendation.csv' in {config.DATA_DIR}."
    )
