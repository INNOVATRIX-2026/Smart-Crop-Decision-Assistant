"""Smart Crop Decision Assistant — Streamlit application.

Two decision engines, one workflow:
  1. **Recommend a crop**  — ML model ranks the best crops for the field's
     soil + weather readings.
  2. **Management actions** — for any chosen crop, a rule-based advisor turns the
     same readings into prioritized, actionable guidance and a suitability score.

Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.advisor import SEVERITY_ICON, advise
from src.data_loader import load_dataset
from src.model import load_metrics, load_model, load_stats, predict
from src.weather import geocode, get_weather

st.set_page_config(
    page_title="Smart Crop Decision Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Cached resources (loaded once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing model & data (first run trains it)…")
def _bootstrap():
    """Ensure dataset + model exist, then load model, stats, metrics, dataframe."""
    df, source = load_dataset()
    model = load_model()
    stats = load_stats()
    metrics = load_metrics()
    return model, stats, metrics, df, source


@st.cache_data(show_spinner=False)
def _fetch_weather(city: str):
    loc = geocode(city)
    if loc is None:
        return None, None
    weather = get_weather(loc["latitude"], loc["longitude"])
    return loc, weather


try:
    MODEL, STATS, METRICS, DATA, SOURCE = _bootstrap()
except Exception as exc:  # pragma: no cover - surfaced in UI
    st.error(f"Failed to initialise the app: {exc}")
    st.stop()

CROPS = sorted(STATS.keys())


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
for feat, meta in config.FEATURE_META.items():
    st.session_state.setdefault(feat, meta["default"])

def _apply_weather(weather: dict) -> None:
    """Push fetched weather into the input widgets safely."""

    for key in ("temperature", "humidity", "rainfall"):

        value = weather.get(key)

        if value is None:
            continue

        value = float(value)

        # Keep the actual API value
        st.session_state[f"actual_{key}"] = value

        # Get the existing slider limits
        meta = config.FEATURE_META[key]

        # Put a value within the existing UI range
        value_for_slider = max(
            meta["min"],
            min(value, meta["max"])
        )

        st.session_state[key] = value_for_slider

# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 Field Inputs")
    st.caption(
        "Enter soil test values and current weather, then explore the tabs."
    )

    # ============================================================
    # AUTO-FILL WEATHER
    # ============================================================

    with st.expander(
        "🌦️ Auto-fill weather by location",
        expanded=True
    ):
        city = st.text_input(
            "City / place name",
            value="Pune",
            placeholder="e.g. Pune, Nairobi, Iowa",
            help=(
                "Fetches live temperature, humidity, "
                "and recent rainfall via Open-Meteo."
            ),
        )

        if st.button("Fetch live weather", width="stretch"):

            try:
                loc, weather = _fetch_weather(city.strip())

                if loc is None:
                    st.warning(
                        "Location not found. Try a different name."
                    )

                else:
                    # Store the REAL weather values separately
                    st.session_state["actual_temperature"] = (
                        weather["temperature"]
                    )

                    st.session_state["actual_humidity"] = (
                        weather["humidity"]
                    )

                    st.session_state["actual_rainfall"] = (
                        weather["rainfall"]
                    )

                    # ------------------------------------------------
                    # Apply values to the existing sliders.
                    # Values are clamped so Streamlit doesn't crash.
                    # ------------------------------------------------
                    _apply_weather(weather)

                    where = ", ".join(
                        x
                        for x in (
                            loc["name"],
                            loc["admin1"],
                            loc["country"],
                        )
                        if x
                    )

                    st.success(
                        f"📍 {where}\n\n"
                        f"🌡️ {weather['temperature']} °C  •  "
                        f"💧 {weather['humidity']}%  •  "
                        f"🌧️ {weather['rainfall']} mm (30-day)"
                    )

            except Exception as exc:
                st.error(
                    f"Weather fetch failed: {exc}"
                )

    # ============================================================
    # SOIL NUTRIENTS
    # ============================================================

    st.subheader("🧪 Soil nutrients")

    for feat in ("N", "P", "K", "ph"):

        m = config.FEATURE_META[feat]

        st.slider(
            f"{m['label']}" + (
                f" ({m['unit']})"
                if m["unit"]
                else ""
            ),
            min_value=m["min"],
            max_value=m["max"],
            step=m["step"],
            key=feat,
            help=m["help"],
        )

    # ============================================================
    # CLIMATE
    # ============================================================

    st.subheader("🌡️ Climate")

    for feat in (
        "temperature",
        "humidity",
        "rainfall",
    ):

        m = config.FEATURE_META[feat]

        # Get current value
        current_value = st.session_state.get(
            feat,
            m["min"]
        )

        # Make sure it is numeric
        try:
            current_value = float(current_value)
        except (TypeError, ValueError):
            current_value = float(m["min"])

        # Clamp value to slider's existing range
        current_value = max(
            float(m["min"]),
            min(
                current_value,
                float(m["max"])
            )
        )

        st.slider(
            f"{m['label']}" + (
                f" ({m['unit']})"
                if m["unit"]
                else ""
            ),
            min_value=m["min"],
            max_value=m["max"],
            step=m["step"],
            value=current_value,
            key=feat,
            help=m["help"],
        )


# ================================================================
# DATASET / MODEL INFORMATION
# ================================================================

st.divider()

st.caption(
    f"📊 Dataset: **{SOURCE}** · "
    f"{METRICS['n_samples']} samples · "
    f"{METRICS['n_classes']} crops"
)

st.caption(
    f"🤖 Model test accuracy: "
    f"**{METRICS['accuracy'] * 100:.1f}%**"
)
# Collect current feature values.
FEATURES = {feat: float(st.session_state[feat]) for feat in config.FEATURES}


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Smart Crop Decision Assistant")
st.markdown(
    "Recommends **suitable crops** and **management actions** from your soil "
    "conditions, crop type, and live weather — powered by machine learning on a "
    "real agricultural dataset."
)

tab_reco, tab_manage, tab_explore, tab_model = st.tabs(
    ["🌱 Crop Recommendation", "🧭 Management Actions", "📊 Explore Data", "🤖 Model Insights"]
)


# --------------------------------------------------------------------------
# Tab 1 — Crop recommendation
# --------------------------------------------------------------------------
with tab_reco:
    st.subheader("Best crops for your field")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Current readings**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Value": [f"{FEATURES[f]:g} {config.FEATURE_META[f]['unit']}".strip()
                              for f in config.FEATURES],
                },
                index=[config.FEATURE_META[f]["label"] for f in config.FEATURES],
            ),
            width="stretch",
        )

    ranked = predict(MODEL, FEATURES, top_k=5)
    top_crop, top_prob = ranked[0]

    with right:
        st.markdown("**Top recommendation**")
        st.markdown(
            f"<div style='font-size:2.4rem;font-weight:700;line-height:1.2'>"
            f"{config.crop_label(top_crop)}</div>"
            f"<div style='color:#2e7d32;font-size:1.1rem'>"
            f"{top_prob*100:.1f}% model confidence · {config.crop_category(top_crop)}</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, top_prob))

    st.markdown("#### Ranked candidates")
    reco_df = pd.DataFrame(
        [
            {
                "Crop": config.crop_label(c),
                "Category": config.crop_category(c),
                "Confidence": p,
            }
            for c, p in ranked
        ]
    )
    st.dataframe(
        reco_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.1f%%", min_value=0.0, max_value=1.0
            )
        },
    )

    st.info(
        f"👉 Head to **Management Actions** to see what it takes to grow "
        f"{config.crop_label(top_crop)} (or any other crop) on this field.",
        icon="🧭",
    )
    # Offer the top crop as the default for the management tab.
    st.session_state.setdefault("manage_crop", top_crop)


# --------------------------------------------------------------------------
# Tab 2 — Management actions
# --------------------------------------------------------------------------
with tab_manage:
    st.subheader("Crop management recommendations")

    # `manage_crop` is seeded in the recommendation tab (which always runs
    # first), so the widget reads its value from session state via `key`.
    chosen = st.selectbox(
        "Select the crop you intend to grow",
        options=CROPS,
        format_func=config.crop_label,
        key="manage_crop",
    )

    advice = advise(chosen, FEATURES, STATS)

    c1, c2, c3 = st.columns(3)
    c1.metric("Suitability score", f"{advice.suitability:.0f}/100")
    c2.metric("Verdict", advice.verdict)
    n_urgent = sum(1 for a in advice.actions if a.severity in ("critical", "warning"))
    c3.metric("Actions needed", n_urgent)

    st.progress(advice.suitability / 100)

    st.markdown("#### Prioritized actions")
    for a in advice.actions:
        icon = SEVERITY_ICON[a.severity]
        with st.container(border=True):
            head, tgt = st.columns([3, 2])
            head.markdown(f"**{icon} {a.title}**")
            if a.target:
                tgt.markdown(
                    f"<div style='text-align:right;color:#555'>"
                    f"now <b>{a.current:g}</b> · target {a.target}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(a.detail)

    with st.expander("How is this computed?"):
        st.markdown(
            "Each reading is compared to the **learned comfort zone** for the "
            "selected crop — the 25th–75th percentile range observed for that crop "
            "in the dataset. Readings outside the zone generate an action whose "
            "urgency scales with how far outside they fall. The **suitability score** "
            "is the average closeness to the ideal across all seven factors."
        )


# --------------------------------------------------------------------------
# Tab 3 — Explore data
# --------------------------------------------------------------------------
with tab_explore:
    st.subheader("Explore the training dataset")

    focus = st.selectbox(
        "Inspect a crop's ideal conditions",
        options=CROPS, format_func=config.crop_label, key="explore_crop",
    )
    s = STATS[focus]
    st.markdown(f"**Typical growing conditions for {config.crop_label(focus)}** "
                f"({s['count']} samples)")

    grid = st.columns(len(config.FEATURES))
    for col, feat in zip(grid, config.FEATURES):
        fs = s[feat]
        unit = config.FEATURE_META[feat]["unit"]
        col.metric(
            config.FEATURE_META[feat]["label"],
            f"{fs['median']:g}{(' ' + unit) if unit else ''}",
            help=f"range {fs['min']:g}–{fs['max']:g}",
        )

    st.markdown("#### Median conditions across all crops")
    summary = (
        DATA.groupby(config.TARGET)[config.FEATURES]
        .median()
        .round(1)
        .reset_index()
        .rename(columns={config.TARGET: "crop"})
    )
    summary["crop"] = summary["crop"].map(config.crop_label)
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown("#### Raw dataset")
    st.caption(f"Source: {SOURCE} · {len(DATA)} rows")
    st.dataframe(DATA, width="stretch", height=280)
    st.download_button(
        "⬇️ Download dataset (CSV)",
        DATA.to_csv(index=False).encode(),
        file_name="Crop_recommendation.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Tab 4 — Model insights
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Model insights")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", f"{METRICS['accuracy']*100:.1f}%")
    m2.metric("F1 (macro)", f"{METRICS['f1_macro']*100:.1f}%")
    m3.metric("Training samples", f"{METRICS['n_train']}")
    m4.metric("Crops", f"{METRICS['n_classes']}")

    st.markdown("#### What drives the recommendation?")
    st.caption("Relative importance of each input in the Random Forest model.")
    imp = (
        pd.DataFrame(
            {
                "feature": [config.FEATURE_META[f]["label"] for f in METRICS["feature_importances"]],
                "importance": list(METRICS["feature_importances"].values()),
            }
        )
        .sort_values("importance", ascending=False)
        .set_index("feature")
    )
    st.bar_chart(imp, horizontal=True, color="#2e7d32")

    st.markdown("#### Pipeline")
    st.markdown(
        "- **Data**: Kaggle *Crop Recommendation* dataset "
        f"(loaded via {SOURCE}).\n"
        "- **Model**: `StandardScaler` → `RandomForestClassifier` (300 trees).\n"
        "- **Target**: one of "
        f"{METRICS['n_classes']} crops from soil (N, P, K, pH) and climate "
        "(temperature, humidity, rainfall).\n"
        "- **Weather**: live values from the free Open-Meteo API."
    )
    with st.expander("Full metrics (JSON)"):
        st.json(METRICS)


st.caption(
    "Built with Streamlit · scikit-learn · Open-Meteo · Kaggle Crop Recommendation "
    "dataset. Guidance is decision-support, not a substitute for local agronomic advice."
)
