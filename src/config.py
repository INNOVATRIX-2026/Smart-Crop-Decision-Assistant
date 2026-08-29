"""Central configuration: file paths, feature schema, data sources, crop metadata."""

from __future__ import annotations

from pathlib import Path

# --- Project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATASET_PATH = DATA_DIR / "Crop_recommendation.csv"
MODEL_PATH = MODEL_DIR / "crop_model.joblib"
STATS_PATH = MODEL_DIR / "crop_stats.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --- Data sources ---------------------------------------------------------
# Official Kaggle dataset (used automatically when Kaggle API credentials exist).
KAGGLE_DATASET = "atharvaingle/crop-recommendation-dataset"

# Public mirrors of the *same* Kaggle dataset, used as a no-credentials fallback
# so the app works out of the box. Both carry the canonical 8-column schema.
DATASET_MIRRORS = [
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv",
    "https://raw.githubusercontent.com/dheerajreddy71/Design_project/main/Crop_recommendation.csv",
]

# --- Feature schema -------------------------------------------------------
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

SOIL_FEATURES = ["N", "P", "K", "ph"]
CLIMATE_FEATURES = ["temperature", "humidity", "rainfall"]

# UI metadata for each input: label, unit, slider bounds, and default value
# (defaults are a realistic rice sample from the dataset).
FEATURE_META = {
    "N":           {"label": "Nitrogen (N)",   "unit": "kg/ha", "min": 0.0,  "max": 140.0, "step": 1.0,  "default": 90.0,   "help": "Ratio of nitrogen content in soil"},
    "P":           {"label": "Phosphorus (P)", "unit": "kg/ha", "min": 0.0,  "max": 145.0, "step": 1.0,  "default": 42.0,   "help": "Ratio of phosphorus content in soil"},
    "K":           {"label": "Potassium (K)",  "unit": "kg/ha", "min": 0.0,  "max": 205.0, "step": 1.0,  "default": 43.0,   "help": "Ratio of potassium content in soil"},
    "temperature": {"label": "Temperature",    "unit": "°C",    "min": -5.0, "max": 55.0,  "step": 0.1,  "default": 20.9,   "help": "Average temperature"},
    "humidity":    {"label": "Humidity",       "unit": "%",     "min": 0.0,  "max": 100.0, "step": 0.5,  "default": 82.0,   "help": "Relative humidity"},
    "ph":          {"label": "Soil pH",        "unit": "",      "min": 0.0,  "max": 14.0,  "step": 0.1,  "default": 6.5,    "help": "Soil pH value (0-14)"},
    "rainfall":    {"label": "Rainfall",       "unit": "mm",    "min": 0.0,  "max": 1200.0, "step": 1.0,  "default": 202.9,  "help": "Rainfall over the growing period. NOTE: the live-weather button fills this with the last 30 days' total, which is a different quantity — monsoon totals can exceed the dataset's 300 mm range."},
}

# The synthetic training dataset's `rainfall` column spans roughly 20-300 mm. Live
# 30-day monsoon totals routinely exceed that (Jabalpur in August: 465 mm), which is
# both out of the model's training distribution and — before the max above was
# raised — a hard crash when written into the slider. Values beyond this are flagged
# in the UI as extrapolation rather than silently accepted.
DATASET_RAINFALL_MAX = 300.0

# Display niceties: emoji + category per crop (falls back gracefully if missing).
CROP_META = {
    "rice":        ("🌾", "Cereal"),
    "maize":       ("🌽", "Cereal"),
    "chickpea":    ("🫛", "Pulse"),
    "kidneybeans": ("🫘", "Pulse"),
    "pigeonpeas":  ("🫘", "Pulse"),
    "mothbeans":   ("🫘", "Pulse"),
    "mungbean":    ("🫛", "Pulse"),
    "blackgram":   ("🫘", "Pulse"),
    "lentil":      ("🫛", "Pulse"),
    "pomegranate": ("🔴", "Fruit"),
    "banana":      ("🍌", "Fruit"),
    "mango":       ("🥭", "Fruit"),
    "grapes":      ("🍇", "Fruit"),
    "watermelon":  ("🍉", "Fruit"),
    "muskmelon":   ("🍈", "Fruit"),
    "apple":       ("🍎", "Fruit"),
    "orange":      ("🍊", "Fruit"),
    "papaya":      ("🌴", "Fruit"),
    "coconut":     ("🥥", "Plantation"),
    "cotton":      ("☁️", "Fiber"),
    "jute":        ("🧵", "Fiber"),
    "coffee":      ("☕", "Plantation"),
}


def crop_emoji(name: str) -> str:
    return CROP_META.get(name, ("🌱", "Crop"))[0]


def crop_category(name: str) -> str:
    return CROP_META.get(name, ("🌱", "Crop"))[1]


def crop_label(name: str) -> str:
    """'rice' -> '🌾 Rice'."""
    return f"{crop_emoji(name)} {name.title()}"
