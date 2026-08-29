"""CLI entry point to (re)train the model and print evaluation metrics.

Usage
-----
    python -m scripts.train            # train, caching dataset + model
    python -m scripts.train --refresh  # re-download the dataset first
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows consoles default to cp1252, which can't encode the bar glyphs below.
# Force UTF-8 so the report prints cleanly on every platform.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

from src import config
from src.data_loader import load_dataset
from src.model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crop-recommendation model.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download of the dataset before training.",
    )
    args = parser.parse_args()

    if args.refresh:
        print("Refreshing dataset from source...")
        _, source = load_dataset(force_refresh=True)
        print(f"  dataset source: {source}")

    print("Training model (RandomForest)...")
    metrics = train()

    print("\n=== Training complete ===")
    print(f"  source          : {metrics['source']}")
    print(f"  samples         : {metrics['n_samples']}  "
          f"({metrics['n_train']} train / {metrics['n_test']} test)")
    print(f"  crops (classes) : {metrics['n_classes']}")
    print(f"  test accuracy   : {metrics['accuracy']:.4f}")
    print(f"  test f1 (macro) : {metrics['f1_macro']:.4f}")

    print("\n  feature importances:")
    for feat, imp in sorted(
        metrics["feature_importances"].items(), key=lambda kv: kv[1], reverse=True
    ):
        bar = "█" * int(round(imp * 40))
        print(f"    {feat:<12} {imp:6.3f}  {bar}")

    print(f"\nArtifacts written to: {config.MODEL_DIR}")
    print(json.dumps(
        {k: metrics[k] for k in ("accuracy", "f1_macro", "n_classes")}, indent=2
    ))


if __name__ == "__main__":
    main()
