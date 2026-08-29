"""Every shipped crop spec must be internally consistent and agronomically sane.

Parametrised across all five crops, so adding a sixth automatically inherits the
whole suite. These are the checks that catch a bad reference-data commit — the risk
the plan names as larger than any code bug, because a wrong Kc or uptake figure
produces a confidently wrong dosage.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.crops import NUTRIENTS, available_crops, get_crop
from src.engine.nutrients import SoilNutrients, recommend_fertiliser
from src.engine.water import SoilWater, recommend_irrigation

from .conftest import make_weather

ALL_CROPS = available_crops()
SOWING = date(2025, 6, 15)

EXPECTED = {"cotton", "maize", "rice", "sugarcane", "wheat"}


def test_all_five_crops_are_present():
    assert set(ALL_CROPS) == EXPECTED


@pytest.fixture(params=ALL_CROPS)
def crop(request):
    return get_crop(request.param)


# --------------------------------------------------------------------------
# Structural consistency
# --------------------------------------------------------------------------
def test_splits_sum_to_one(crop):
    """A drift here silently under- or over-applies the whole season dose."""
    for nut in NUTRIENTS:
        total = sum(s.fraction.get(nut, 0.0) for s in crop.nutrients.splits)
        assert total == pytest.approx(1.0), f"{crop.crop}/{nut} sums to {total}"


def test_split_days_are_within_the_season(crop):
    for s in crop.nutrients.splits:
        assert 0 <= s.day <= crop.water.total_days


def test_split_stage_names_are_real_stages(crop):
    names = {s.name for s in crop.water.stages} | {"basal"}
    for s in crop.nutrients.splits:
        assert s.stage in names, f"{crop.crop}: unknown split stage {s.stage!r}"


def test_has_hindi_label(crop):
    """The i18n layer depends on this being present for every crop."""
    hi = crop.display_name.get("hi")
    assert hi and hi != crop.crop


def test_yield_basis_is_declared(crop):
    """Cotton is seed cotton, sugarcane is cane — mislabelling misleads by 10x."""
    assert crop.nutrients.yield_basis
    if crop.crop == "cotton":
        assert crop.nutrients.yield_basis == "seed cotton"
    if crop.crop == "sugarcane":
        assert crop.nutrients.yield_basis == "millable cane"


# --------------------------------------------------------------------------
# Agronomic plausibility
# --------------------------------------------------------------------------
def test_kc_curve_shape(crop):
    """Kc must peak mid-season — that is what the FAO-56 curve means."""
    w = crop.water
    mid_start = w.stage_start_day("mid")
    assert w.kc_at(mid_start + 1) >= w.kc_at(1)
    assert w.kc_at(mid_start + 1) >= w.kc_at(w.total_days)
    assert w.kc_at(mid_start + 1) == pytest.approx(w.kc_mid)


def test_kc_values_are_plausible(crop):
    for das in range(0, crop.water.total_days + 1):
        assert 0.1 <= crop.water.kc_at(das) <= 1.4


def test_root_depth_is_monotonic_and_bounded(crop):
    depths = [crop.water.root_depth_at(d) for d in range(0, crop.water.total_days + 1)]
    assert all(b >= a - 1e-9 for a, b in zip(depths, depths[1:]))
    assert depths[-1] == pytest.approx(crop.water.root_depth_max_m)
    assert 0.3 <= crop.water.root_depth_max_m <= 2.0


def test_season_length_is_plausible(crop):
    assert 90 <= crop.water.total_days <= 400


def test_depletion_fraction_is_plausible(crop):
    assert 0.15 <= crop.water.depletion_fraction_p <= 0.75


def test_rice_is_the_most_moisture_sensitive():
    """Sanity anchor: rice's p must be far below the others (FAO-56 gives 0.20)."""
    rice_p = get_crop("rice").water.depletion_fraction_p
    others = [get_crop(c).water.depletion_fraction_p for c in ALL_CROPS if c != "rice"]
    assert rice_p < min(others)


def test_rice_tolerates_standing_water_and_maize_does_not():
    """Rice is grown flooded; treating it like maize would spam drainage alerts."""
    rice = get_crop("rice").limits["waterlogging_tolerance_days"]
    maize = get_crop("maize").limits["waterlogging_tolerance_days"]
    assert rice > 30
    assert maize <= 3


def test_efficiency_ordering(crop):
    """Drip must beat sprinkler must beat surface, for every crop."""
    eff = crop.water.application_efficiency
    assert eff["surface"] < eff["sprinkler"] < eff["drip"]


def test_uptake_scale_matches_yield_scale(crop):
    """Guards the units trap: high-tonnage crops need small per-tonne figures.

    Season nitrogen uptake at attainable yield should land in a plausible band for
    any field crop. Sugarcane at 80 t/ha and wheat at 5 t/ha both pass only if their
    per-tonne figures are on the right scale.
    """
    n_per_t = crop.nutrients.uptake_per_tonne_grain["N"]
    y = crop.nutrients.attainable_yield_t_ha.get(
        "irrigated", crop.nutrients.reference_yield_t_ha
    )
    season_uptake = n_per_t * y
    assert 40 <= season_uptake <= 350, (
        f"{crop.crop}: {season_uptake:.0f} kg N/ha uptake at {y} t/ha looks off-scale"
    )


def test_reference_dose_is_plausible(crop):
    n = crop.nutrients.reference_dose_kg_ha["N"]
    assert 40 <= n <= 400, f"{crop.crop}: reference N dose {n} kg/ha implausible"


# --------------------------------------------------------------------------
# End-to-end: every crop produces a usable advisory
# --------------------------------------------------------------------------
def test_every_crop_produces_an_irrigation_plan(crop):
    soil = SoilWater(theta_fc=0.30, theta_wp=0.15)
    today = SOWING + timedelta(days=min(45, crop.water.total_days - 1))
    weather = make_weather(SOWING, (today - SOWING).days + 16, eto_mm=5.0, rain_mm=0.0)

    plan = recommend_irrigation(crop, soil, weather, SOWING, today)
    assert plan.action
    assert plan.net_depth_mm >= 0
    assert plan.explanation.conclusion
    if plan.day:
        trigger = next(b for b in plan.balance if b.day == plan.day)
        assert plan.net_depth_mm <= trigger.taw_mm + 1e-9


def test_every_crop_produces_a_fertiliser_plan(crop):
    soil = SoilNutrients(
        available={"N": 350.0, "P2O5": 15.0, "K2O": 150.0},
        provenance={n: "measured" for n in NUTRIENTS},
    )
    today = SOWING + timedelta(days=min(45, crop.water.total_days - 1))
    plan = recommend_fertiliser(crop, soil, SOWING, today)

    assert plan.doses, f"{crop.crop}: no dose events generated"
    assert plan.explanation.conclusion
    for nut in NUTRIENTS:
        applied = sum(d.nutrients[nut] for d in plan.doses)
        assert applied == pytest.approx(plan.season_dose[nut])
    for d in plan.doses:
        assert d.products, f"{crop.crop}/{d.stage}: nutrients but no product"


def test_every_crop_reproduces_its_reference_package(crop):
    """At reference yield on a medium soil, output must equal the validated package."""
    soil = SoilNutrients(
        available={"N": 350.0, "P2O5": 15.0, "K2O": 150.0},
        provenance={n: "measured" for n in NUTRIENTS},
    )
    plan = recommend_fertiliser(
        crop, soil, SOWING, SOWING,
        target_yield_t_ha=crop.nutrients.reference_yield_t_ha,
    )
    for nut in NUTRIENTS:
        assert plan.season_dose[nut] == pytest.approx(
            crop.nutrients.reference_dose_kg_ha[nut]
        ), f"{crop.crop}/{nut} does not reproduce its reference dose"
