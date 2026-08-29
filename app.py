"""Smart Crop Decision Assistant — Streamlit application.

**Primary function — management actions.** Given crop type, soil conditions and
weather, the agronomic engine (``src/engine/``) prescribes *how much* water and
fertiliser and *when*, with the arithmetic on show. This is what the problem
statement asks for: crop type is an **input**, and the output is a quantity and a
date.

**Secondary — field suitability.** A ranking of which crops suit the field's
readings. This solves the *inverse* problem and is kept as an exploratory view, with
its synthetic training data stated plainly rather than presented as a headline result.

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
from ui import advisory

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
    """Push fetched weather into the input widgets, safely.

    Two things happen here and both matter:

    1. **The true API value is preserved** under ``actual_<key>`` in session state, so
       the real measurement is never silently replaced by a clamped stand-in.
    2. **The slider value is clamped** to the widget's declared bounds. Without this a
       live fetch over a monsoon region raises ``StreamlitValueAboveMaxError`` —
       Jabalpur in August returns a 30-day total of ~465 mm against a slider that used
       to stop at 320.

    The underlying problem is a units mismatch, not a bounds problem: the dataset's
    ``rainfall`` column means *growing-period* rainfall, while ``get_weather`` returns
    the *last 30 days*. Different quantities sharing a name. Anything past the
    dataset's range is therefore extrapolation for the suitability model, and is
    reported as such rather than quietly accepted.
    """
    notes: list[str] = []

    for key in ("temperature", "humidity", "rainfall"):
        raw = weather.get(key)
        if raw is None:
            continue
        value = float(raw)

        # Keep the real measurement, separate from whatever the slider can display.
        st.session_state[f"actual_{key}"] = value

        meta = config.FEATURE_META[key]
        clamped = max(float(meta["min"]), min(value, float(meta["max"])))
        if clamped != value:
            notes.append(
                f"{meta['label']} {value:g} {meta['unit']} is outside the input range "
                f"and was clamped to {clamped:g} for display."
            )
        st.session_state[key] = clamped

    rain = weather.get("rainfall")
    if rain is not None and float(rain) > config.DATASET_RAINFALL_MAX:
        notes.append(
            f"{float(rain):g} mm over 30 days is beyond the suitability model's "
            f"training range (~{config.DATASET_RAINFALL_MAX:g} mm), so its ranking is "
            f"extrapolating. The Management Actions tab is unaffected — it uses a "
            f"dated rainfall series from Open-Meteo, not this figure."
        )
    st.session_state["_weather_notes"] = notes

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
        "🌦️ give your current location",
        expanded=True
    ):
        city = st.text_input(
            "City / place name",
            value="Pune",
            placeholder="e.g. Pune, warangal, nagpur",
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
                    for note in st.session_state.get("_weather_notes", []):
                        st.info(note, icon="ℹ️")

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

        # Defensively repair a stale session value before building the widget.
        # A value left over from an earlier run (or an older, narrower slider range)
        # would otherwise crash Streamlit on instantiation.
        #
        # Note: the clamped value is written back to session_state rather than passed
        # as `value=`. Passing both `value` and `key` to a Streamlit widget makes it
        # fight its own session-state binding and can reset the slider on every rerun.
        try:
            current_value = float(st.session_state.get(feat, m["min"]))
        except (TypeError, ValueError):
            current_value = float(m["min"])
        st.session_state[feat] = max(
            float(m["min"]), min(current_value, float(m["max"]))
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
            key=feat,
            help=m["help"],
        )

        # Show the true measurement when the slider had to clamp it.
        actual = st.session_state.get(f"actual_{feat}")
        if actual is not None and abs(float(actual) - st.session_state[feat]) > 1e-9:
            st.caption(f"Measured: **{float(actual):g} {m['unit']}** (outside slider range)")

    st.divider()
    st.caption(f"📊 Dataset: **{SOURCE}** · {METRICS['n_samples']} samples · "
               f"{METRICS['n_classes']} crops")
    st.caption(
        f"🤖 Suitability model accuracy: **{METRICS['accuracy']*100:.1f}%** — "
        "inflated by synthetic data, see the Field Suitability tab"
    )


# Collect current feature values.
FEATURES = {feat: float(st.session_state[feat]) for feat in config.FEATURES}


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Smart Crop Decision Assistant")
st.markdown(
    "Given **crop type, soil conditions and weather**, this prescribes the "
    "**management actions** a farmer needs — how much water and fertiliser, and "
    "when — computed from FAO-56 and validated agronomic packages, with every "
    "number's derivation on show."
)

tab_manage, tab_reco, tab_explore, tab_model = st.tabs(
    ["🧭 Management Actions", "🌱 Field Suitability", "📊 Explore Data", "🤖 Model Insights"]
)


# --------------------------------------------------------------------------
# Tab 1 — Management actions (the point of the platform)
# --------------------------------------------------------------------------
with tab_manage:
    advisory.render()


# --------------------------------------------------------------------------
# Tab 2 — Field suitability (the inverse problem; secondary)
# --------------------------------------------------------------------------
with tab_reco:
    st.subheader("Which crops suit this field?")
    st.info(
        "**This answers the inverse question** and is not the platform's main output. "
        "The brief supplies crop type as an *input*; this tab instead ranks crops from "
        "soil and climate readings. Useful for exploring a field, but read the caveat "
        "below before quoting the confidence figures.",
        icon="ℹ️",
    )
    st.warning(
        "The model behind this ranking is trained on the Kaggle *Crop Recommendation* "
        "dataset: 2200 rows, **exactly 100 per crop across 22 crops**. That perfect "
        "balance means the data is generated, not observed — which is why test accuracy "
        f"reads {METRICS['accuracy']*100:.1f}%. Treat the ranking as indicative only. "
        "The Management Actions tab uses no part of this dataset.",
        icon="⚠️",
    )

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Current readings** (set in the sidebar)")
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
        st.markdown("**Best match**")
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

    st.markdown("#### Dataset-relative guidance")
    st.caption(
        "The comparison below is against this crop's quartiles **in the synthetic "
        "dataset** — not agronomic thresholds. It cannot give dosages; for real "
        "quantities use the Management Actions tab."
    )
    chosen = st.selectbox(
        "Compare readings against",
        options=CROPS, format_func=config.crop_label, key="manage_crop",
    )
    advice = advise(chosen, FEATURES, STATS)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset-relative fit", f"{advice.suitability:.0f}/100")
    c2.metric("Verdict", advice.verdict)
    n_urgent = sum(1 for a in advice.actions if a.severity in ("critical", "warning"))
    c3.metric("Readings outside range", n_urgent)

    for a in advice.actions:
        icon = SEVERITY_ICON[a.severity]
        with st.container(border=True):
            head, tgt = st.columns([3, 2])
            head.markdown(f"**{icon} {a.title}**")
            if a.target:
                tgt.markdown(
                    f"<div style='text-align:right;color:#555'>"
                    f"now <b>{a.current:g}</b> · dataset range {a.target}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(a.detail)


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
