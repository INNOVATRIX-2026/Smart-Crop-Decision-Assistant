"""Streamlit view: quantified management actions from the agronomic engine.

This is the primary tab and the point of the whole platform: given a location, a
crop, and a sowing date, it prescribes **how much water and fertiliser, and when** —
with the arithmetic on show.

It renders only what the engine produced. No thresholds, formulas, or dosages are
recomputed here, so the explanation panel cannot drift out of sync with the decision
it claims to explain.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.crops import available_crops, get_crop
from src.engine import nutrients as nut
from src.engine import water as wtr
from src.soil import DEFAULT_TEXTURE, TEXTURE_PRESETS, fetch_soil, saxton_rawls
from src.weather import get_daily_series, rain_forecast_map

URGENCY_ICON = {"critical": "🔴", "warning": "🟠", "good": "🟢", "info": "🔵"}

ACTION_TITLE = {
    wtr.ACTION_IRRIGATE_NOW: "Irrigate now",
    wtr.ACTION_IRRIGATE_SCHEDULED: "Irrigation scheduled",
    wtr.ACTION_RAIN_EXPECTED: "Hold off — rain expected",
    wtr.ACTION_NO_ACTION: "No irrigation needed",
    wtr.ACTION_DRAINAGE: "Ensure drainage",
}

# A few reference locations so the demo does not depend on typing coordinates.
PRESET_LOCATIONS = {
    "Karnal, Haryana": (29.40, 76.30),
    "Ludhiana, Punjab": (30.90, 75.85),
    "Bhopal, Madhya Pradesh": (23.20, 77.50),
    "Thanjavur, Tamil Nadu": (11.00, 78.00),
    "Nashik, Maharashtra": (20.00, 73.80),
}


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _soil(lat: float, lon: float, texture: str):
    return fetch_soil(lat, lon, texture_fallback=texture)


@st.cache_data(show_spinner=False, ttl=1800)
def _weather(lat: float, lon: float, sowing: date, today: date):
    return get_daily_series(lat, lon, sowing, today)


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def _explanation(exp, key: str) -> None:
    """Render a computation trace exactly as the engine emitted it."""
    with st.expander("Show the working", expanded=False):
        if exp.inputs:
            st.markdown("**Inputs used**")
            st.dataframe(
                pd.DataFrame({"Value": list(exp.inputs.values())}, index=list(exp.inputs)),
                width="stretch",
            )
        if exp.steps:
            st.markdown("**Computation**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Step": s.label,
                            "How": s.expression,
                            "Result": f"{s.value:,.1f} {s.unit}".strip(),
                        }
                        for s in exp.steps
                    ]
                ),
                width="stretch", hide_index=True,
            )
        if exp.threshold:
            st.info(f"**Threshold** — {exp.threshold}", icon="📏")
        if exp.sources:
            st.caption("Sources: " + " · ".join(exp.sources))


def _irrigation_card(plan: wtr.IrrigationPlan) -> None:
    icon = URGENCY_ICON.get(plan.urgency, "🔵")
    title = ACTION_TITLE.get(plan.action, plan.action)

    with st.container(border=True):
        head, when = st.columns([3, 2])
        head.markdown(f"### {icon} {title}")
        if plan.day:
            rel = (
                "today" if (plan.days_from_today or 0) <= 0
                else "tomorrow" if plan.days_from_today == 1
                else f"in {plan.days_from_today} days"
            )
            when.markdown(
                f"<div style='text-align:right;padding-top:.6rem'>"
                f"<b>{plan.day:%d %b %Y}</b><br><span style='color:#666'>{rel}</span></div>",
                unsafe_allow_html=True,
            )

        if plan.gross_depth_mm > 0:
            a, b, c = st.columns(3)
            a.metric("Apply", f"{plan.gross_depth_mm:.0f} mm")
            b.metric("Volume", f"{plan.cubic_metres_per_ha:,.0f} m³/ha")
            c.metric("Net to root zone", f"{plan.net_depth_mm:.0f} mm")

        if plan.current:
            cur = plan.current
            st.progress(
                min(1.0, max(0.0, cur.available_fraction)),
                text=(
                    f"Soil water now: {cur.available_fraction * 100:.0f}% of available "
                    f"({cur.depletion_mm:.0f} mm depleted of {cur.taw_mm:.0f} mm) · "
                    f"{cur.stage} stage, day {cur.das}"
                ),
            )

        st.markdown(plan.explanation.conclusion)
        for w in plan.warnings:
            st.warning(w, icon="⚠️")
        _explanation(plan.explanation, "irrig")


def _fertiliser_card(plan: nut.NutrientPlan, crop_label: str) -> None:
    nd = plan.next_dose
    icon = URGENCY_ICON.get(plan.urgency, "🔵")

    with st.container(border=True):
        head, when = st.columns([3, 2])
        head.markdown(f"### {icon} Fertiliser")
        if nd:
            when.markdown(
                f"<div style='text-align:right;padding-top:.6rem'>"
                f"<b>{nd.day:%d %b %Y}</b><br>"
                f"<span style='color:#666'>{nd.stage} · day {nd.das}</span></div>",
                unsafe_allow_html=True,
            )

        if nd and nd.products:
            cols = st.columns(max(2, len(nd.products)))
            for col, p in zip(cols, nd.products):
                col.metric(p.product, f"{p.kg_ha:.0f} kg/ha")

        st.markdown(plan.explanation.conclusion)

        st.markdown("**Season plan**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When": f"{d.day:%d %b}",
                        "Day": d.das,
                        "Stage": d.stage,
                        "Status": d.status,
                        "Product": " + ".join(p.render() for p in d.products),
                        "Supplies": " + ".join(
                            f"{v:.0f} kg {k}" for k, v in d.nutrients.items() if v > 0.5
                        ),
                    }
                    for d in plan.doses
                ]
            ),
            width="stretch", hide_index=True,
        )

        cols = st.columns(3)
        for col, (k, v) in zip(cols, plan.season_dose.items()):
            col.metric(
                f"Season {k}", f"{v:.0f} kg/ha",
                help=f"Soil tested {plan.fertility_classes[k]} for {k}",
            )

        for w in plan.warnings:
            st.warning(w, icon="⚠️")
        _explanation(plan.explanation, "fert")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def render() -> None:
    st.subheader("Crop management actions")
    st.caption(
        "Quantified prescriptions from soil, weather and crop stage — how much and "
        "when, with the arithmetic shown."
    )

    # ---- inputs ---------------------------------------------------------
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        place = st.selectbox("Location", list(PRESET_LOCATIONS), key="adv_place")
        lat, lon = PRESET_LOCATIONS[place]
        if st.checkbox("Enter coordinates manually", key="adv_manual"):
            lat = st.number_input("Latitude", -90.0, 90.0, float(lat), 0.01, key="adv_lat")
            lon = st.number_input("Longitude", -180.0, 180.0, float(lon), 0.01, key="adv_lon")

    with c2:
        crop_name = st.selectbox(
            "Crop", available_crops(),
            format_func=lambda c: get_crop(c).label(), key="adv_crop",
        )
        crop = get_crop(crop_name)
        method = st.selectbox(
            "Irrigation method", list(crop.water.application_efficiency),
            format_func=lambda m: f"{m} ({crop.water.application_efficiency[m]:.0%} efficient)",
            key="adv_method",
        )

    with c3:
        today = date.today()
        sowing = st.date_input(
            "Sowing date", value=today - timedelta(days=45),
            max_value=today, key="adv_sowing",
        )
        das = (today - sowing).days
        st.caption(
            f"Day **{das}** of a {crop.water.total_days}-day season · "
            f"**{crop.water.stage_at(das)}** stage"
        )

    if das > crop.water.total_days:
        st.warning(
            f"That sowing date is {das} days ago, past this crop's "
            f"{crop.water.total_days}-day season — the advisory below assumes the crop "
            f"is at harvest.",
            icon="⚠️",
        )

    # ---- soil -----------------------------------------------------------
    with st.spinner("Fetching soil and weather…"):
        texture_fallback = st.session_state.get("adv_texture", DEFAULT_TEXTURE)
        soil_profile = _soil(lat, lon, texture_fallback)
        days, provenance = _weather(lat, lon, sowing, today)

    badge = (
        ":green[**LIVE**]" if provenance == "LIVE"
        else f":orange[**{provenance}**]" if provenance.startswith("CACHED")
        else ":red[**NO DATA**]"
    )
    st.markdown(
        f"Soil: **{soil_profile.texture_class}** "
        f"({soil_profile.sand_pct:.0f}% sand, {soil_profile.clay_pct:.0f}% clay, "
        f"pH {soil_profile.ph:.1f}) · source *{soil_profile.source}* — "
        f"Weather: {len(days)} days {badge}"
    )

    if "no SoilGrids coverage" in soil_profile.source:
        st.warning(
            "SoilGrids has no data at this location — it has genuine gaps over parts "
            "of India. Soil properties below come from a texture preset, so pick the "
            "texture that matches your field.",
            icon="⚠️",
        )
        st.selectbox(
            "Soil texture", list(TEXTURE_PRESETS), key="adv_texture",
            help="Used to derive water-holding capacity when no survey data exists.",
        )

    with st.expander("Soil water limits (derived, not measured)"):
        fc, wp = saxton_rawls(soil_profile.sand_pct, soil_profile.clay_pct,
                              soil_profile.om_pct)
        a, b, c = st.columns(3)
        a.metric("Field capacity", f"{fc:.3f} m³/m³")
        b.metric("Wilting point", f"{wp:.3f} m³/m³")
        c.metric("Available water", f"{fc - wp:.3f} m³/m³")
        st.caption(
            "Derived from texture and organic matter via the Saxton & Rawls (2006) "
            "pedotransfer functions. No API supplies these directly."
        )

    if not days:
        st.error(
            "No weather data could be fetched or recovered from cache, so soil "
            "moisture cannot be reconstructed. Check the network and retry.",
            icon="🚫",
        )
        return

    # ---- irrigation history --------------------------------------------
    with st.expander("Record irrigations already applied (improves accuracy)"):
        st.caption(
            "Without this the balance assumes nothing has been applied since sowing, "
            "so it accumulates deficit and over-prescribes."
        )
        n_events = st.number_input("How many?", 0, 6, 0, key="adv_n_irrig")
        irrigations: dict[date, float] = {}
        for i in range(int(n_events)):
            ec1, ec2 = st.columns(2)
            d = ec1.date_input(f"Date {i + 1}", value=sowing + timedelta(days=21 * (i + 1)),
                               min_value=sowing, max_value=today, key=f"adv_id_{i}")
            mm = ec2.number_input(f"Net depth {i + 1} (mm)", 0.0, 200.0, 50.0, 5.0,
                                  key=f"adv_im_{i}")
            irrigations[d] = mm

    # ---- run the engine -------------------------------------------------
    soil_water = soil_profile.water()
    irrigation = wtr.recommend_irrigation(
        crop, soil_water, days, sowing, today, method=method, irrigations=irrigations,
    )
    fertiliser = nut.recommend_fertiliser(
        crop, soil_profile.nutrients(), sowing, today,
        rain_forecast=rain_forecast_map(days),
    )

    st.divider()
    _irrigation_card(irrigation)
    _fertiliser_card(fertiliser, crop.label())

    # ---- water balance chart -------------------------------------------
    with st.expander("Soil water balance since sowing"):
        bal = pd.DataFrame(
            [
                {
                    "Date": b.day,
                    "Depletion (mm)": round(b.depletion_mm, 1),
                    "Trigger RAW (mm)": round(b.raw_mm, 1),
                    "Capacity TAW (mm)": round(b.taw_mm, 1),
                }
                for b in irrigation.balance
            ]
        ).set_index("Date")
        st.line_chart(bal, height=280)
        st.caption(
            "Depletion is reconstructed from rainfall and evapotranspiration since "
            "sowing — no soil-moisture sensor involved. Irrigation is due wherever "
            "the depletion line meets the trigger line."
        )

    st.caption(
        "Decision support, not a substitute for local agronomic advice. Values marked "
        "estimated or assumed carry real uncertainty — see the working on each card."
    )
