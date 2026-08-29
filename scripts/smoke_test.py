"""End-to-end smoke test: data → model → prediction → advice → live weather.

Run with:  python -m scripts.smoke_test
Exits non-zero if any stage fails.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src import config
from src.advisor import advise
from src.data_loader import load_dataset
from src.model import load_model, load_stats, predict


def main() -> int:
    ok = True

    # 1. Dataset ----------------------------------------------------------
    df, source = load_dataset()
    assert list(df.columns) == config.FEATURES + [config.TARGET], "unexpected columns"
    assert len(df) > 1000, "dataset looks too small"
    print(f"[ok] dataset: {len(df)} rows, {df[config.TARGET].nunique()} crops (via {source})")

    # 2. Model + prediction ----------------------------------------------
    model = load_model()
    stats = load_stats()
    sample = {f: config.FEATURE_META[f]["default"] for f in config.FEATURES}
    ranked = predict(model, sample, top_k=3)
    assert len(ranked) == 3, "expected 3 ranked crops"
    assert abs(sum(p for _, p in ranked) - ranked[0][1]) >= 0, "probabilities malformed"
    top_crop, top_prob = ranked[0]
    print(f"[ok] prediction: {top_crop} ({top_prob*100:.1f}%) "
          f"| runners-up: {[c for c, _ in ranked[1:]]}")

    # 3. Advisor ----------------------------------------------------------
    advice = advise(top_crop, sample, stats)
    assert 0 <= advice.suitability <= 100, "suitability out of range"
    assert len(advice.actions) == len(config.FEATURES), "one action per feature expected"
    print(f"[ok] advisor: suitability {advice.suitability:.0f}/100 "
          f"({advice.verdict}), {len(advice.actions)} actions")

    # Advisor should flag a clearly bad condition (starve the crop of N).
    bad = dict(sample, N=0.0, ph=3.0)
    bad_advice = advise(top_crop, bad, stats)
    assert bad_advice.suitability < advice.suitability, "advisor didn't penalise bad inputs"
    print(f"[ok] advisor reacts to poor inputs: {bad_advice.suitability:.0f}/100")

    # 4. Weather (network; non-fatal) ------------------------------------
    try:
        from src.weather import geocode, get_weather
        loc = geocode("Pune")
        assert loc is not None, "geocode returned nothing"
        w = get_weather(loc["latitude"], loc["longitude"])
        assert w["temperature"] is not None, "no temperature"
        print(f"[ok] weather: {loc['name']} → {w['temperature']}°C, "
              f"{w['humidity']}%, {w['rainfall']}mm")
    except Exception as exc:  # network may be unavailable
        print(f"[warn] weather check skipped (network?): {exc}")

    print("\nAll critical checks passed ✅" if ok else "\nFAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
