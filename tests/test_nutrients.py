"""Fertiliser prescription: dose arithmetic, product conversion, weather coupling.

The tests that matter most here are the ones that would have caught the original
bug — a dose derived from a dataset quartile and printed as kg/ha. Every dose below
traces to a validated reference package scaled by explicit factors.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.crops import NUTRIENTS
from src.engine.nutrients import (
    LEACHING_RAIN_MM,
    PRODUCTS,
    STATUS_DEFERRED,
    STATUS_DUE,
    SoilNutrients,
    allocate_products,
    recommend_fertiliser,
)

from .conftest import SOWING


# --------------------------------------------------------------------------
# Soil nutrient inputs
# --------------------------------------------------------------------------
def test_missing_nutrient_is_rejected():
    with pytest.raises(ValueError, match="K2O"):
        SoilNutrients(available={"N": 300.0, "P2O5": 15.0})


def test_negative_nutrient_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        SoilNutrients(available={"N": -1.0, "P2O5": 15.0, "K2O": 150.0})


def test_fertility_classification(wheat):
    """Soil Health Card bands: N low <280, medium 280-560, high >560 kg/ha."""
    low = SoilNutrients(available={"N": 200.0, "P2O5": 5.0, "K2O": 50.0})
    med = SoilNutrients(available={"N": 350.0, "P2O5": 15.0, "K2O": 150.0})
    high = SoilNutrients(available={"N": 700.0, "P2O5": 40.0, "K2O": 400.0})
    assert low.fertility_class("N", wheat) == "low"
    assert med.fertility_class("N", wheat) == "medium"
    assert high.fertility_class("N", wheat) == "high"


def test_value_above_the_top_band_is_high(wheat):
    huge = SoilNutrients(available={"N": 99_999.0, "P2O5": 15.0, "K2O": 150.0})
    assert huge.fertility_class("N", wheat) == "high"


def test_provenance_defaults_to_assumed():
    s = SoilNutrients(available={"N": 300.0, "P2O5": 15.0, "K2O": 150.0})
    assert s.provenance_of("P2O5") == "assumed"


# --------------------------------------------------------------------------
# Season dose = reference × yield ratio × fertility factor
# --------------------------------------------------------------------------
def test_dose_equals_reference_at_reference_yield(wheat, medium_soil_nutrients):
    """Medium soil at the reference yield must reproduce the validated package."""
    plan = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, SOWING,
        target_yield_t_ha=wheat.nutrients.reference_yield_t_ha,
    )
    for nut in NUTRIENTS:
        assert plan.season_dose[nut] == pytest.approx(
            wheat.nutrients.reference_dose_kg_ha[nut]
        )


def test_dose_scales_linearly_with_yield_target(wheat, medium_soil_nutrients):
    ref_yield = wheat.nutrients.reference_yield_t_ha
    half = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, SOWING, target_yield_t_ha=ref_yield / 2
    )
    full = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, SOWING, target_yield_t_ha=ref_yield
    )
    for nut in NUTRIENTS:
        assert half.season_dose[nut] == pytest.approx(full.season_dose[nut] / 2)


def test_poor_soil_gets_more_and_rich_soil_gets_less(wheat):
    """The invariant from the plan: dose falls as soil supply rises."""
    ref_yield = wheat.nutrients.reference_yield_t_ha
    doses = {}
    for label, n_value in (("low", 200.0), ("medium", 350.0), ("high", 700.0)):
        soil = SoilNutrients(available={"N": n_value, "P2O5": 15.0, "K2O": 150.0})
        plan = recommend_fertiliser(
            wheat, soil, SOWING, SOWING, target_yield_t_ha=ref_yield
        )
        doses[label] = plan.season_dose["N"]
    assert doses["low"] > doses["medium"] > doses["high"]


def test_fertility_class_is_reported(wheat, medium_soil_nutrients):
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    assert plan.fertility_classes["N"] == "medium"


def test_target_yield_defaults_to_attainable(wheat, medium_soil_nutrients):
    """The plan must work with no yield model trained."""
    irrigated = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, SOWING, irrigated=True
    )
    rainfed = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, SOWING, irrigated=False
    )
    assert irrigated.target_yield_t_ha > rainfed.target_yield_t_ha
    assert irrigated.season_dose["N"] > rainfed.season_dose["N"]


def test_non_positive_yield_is_rejected(wheat, medium_soil_nutrients):
    with pytest.raises(ValueError, match="target yield"):
        recommend_fertiliser(
            wheat, medium_soil_nutrients, SOWING, SOWING, target_yield_t_ha=0.0
        )


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def test_split_doses_sum_to_the_season_requirement(wheat, medium_soil_nutrients):
    """The invariant from the plan — a drift here silently mis-applies fertiliser."""
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    for nut in NUTRIENTS:
        applied = sum(d.nutrients[nut] for d in plan.doses)
        assert applied == pytest.approx(plan.season_dose[nut])


def test_doses_are_scheduled_in_order(wheat, medium_soil_nutrients):
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    days = [d.day for d in plan.doses]
    assert days == sorted(days)


def test_basal_dose_lands_on_the_sowing_date(wheat, medium_soil_nutrients):
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    basal = plan.doses[0]
    assert basal.day == SOWING
    assert basal.das == 0


# --------------------------------------------------------------------------
# Product allocation
# --------------------------------------------------------------------------
def test_phosphorus_comes_from_dap():
    products = {p.product: p for p in allocate_products(n=0.0, p2o5=46.0, k2o=0.0)}
    assert "DAP" in products
    assert products["DAP"].kg_ha == pytest.approx(46.0 / PRODUCTS["DAP"]["P2O5"])


def test_dap_nitrogen_is_credited_against_the_urea_top_up():
    """DAP carries 18% N — failing to credit it would over-apply nitrogen."""
    p2o5, n_target = 46.0, 60.0
    products = {p.product: p for p in allocate_products(n_target, p2o5, 0.0)}
    dap_kg = p2o5 / PRODUCTS["DAP"]["P2O5"]
    n_from_dap = dap_kg * PRODUCTS["DAP"]["N"]
    expected_urea = (n_target - n_from_dap) / PRODUCTS["Urea"]["N"]
    assert products["Urea"].kg_ha == pytest.approx(expected_urea)


def test_total_nitrogen_delivered_matches_the_request():
    n_target = 60.0
    products = allocate_products(n_target, 46.0, 0.0)
    delivered = sum(p.supplies.get("N", 0.0) for p in products)
    assert delivered == pytest.approx(n_target)


def test_no_urea_when_dap_already_covers_nitrogen():
    """A small N target beside a large P target needs no urea at all."""
    products = {p.product: p for p in allocate_products(n=1.0, p2o5=100.0, k2o=0.0)}
    assert "Urea" not in products


def test_potassium_comes_from_mop():
    products = {p.product: p for p in allocate_products(0.0, 0.0, 60.0)}
    assert products["MOP"].kg_ha == pytest.approx(60.0 / PRODUCTS["MOP"]["K2O"])


def test_zero_request_yields_no_products():
    assert allocate_products(0.0, 0.0, 0.0) == ()


def test_every_dose_event_has_products(wheat, medium_soil_nutrients):
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    for d in plan.doses:
        assert d.products, f"{d.stage} dose has nutrients but no product"
        assert d.total_kg_ha > 0


# --------------------------------------------------------------------------
# Weather coupling
# --------------------------------------------------------------------------
def test_heavy_rain_defers_a_due_top_dress(wheat, medium_soil_nutrients):
    """Surface-applied N before heavy rain is washed away — hold it."""
    split = wheat.nutrients.splits[1]          # first N top-dress
    dose_day = SOWING + timedelta(days=split.day)
    rain = {dose_day: LEACHING_RAIN_MM + 20.0}

    plan = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, today=dose_day, rain_forecast=rain
    )
    assert plan.next_dose is not None
    assert plan.next_dose.status == STATUS_DEFERRED
    assert "leaching" in plan.next_dose.defer_reason


def test_dry_forecast_leaves_the_dose_due(wheat, medium_soil_nutrients):
    split = wheat.nutrients.splits[1]
    dose_day = SOWING + timedelta(days=split.day)
    rain = {dose_day: 0.0}

    plan = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, today=dose_day, rain_forecast=rain
    )
    assert plan.next_dose.status == STATUS_DUE
    assert plan.urgency == "critical"


def test_deferral_suggests_a_dry_day(wheat, medium_soil_nutrients):
    split = wheat.nutrients.splits[1]
    dose_day = SOWING + timedelta(days=split.day)
    rain = {dose_day: 60.0, dose_day + timedelta(days=1): 40.0}
    for d in range(2, 8):
        rain[dose_day + timedelta(days=d)] = 0.0

    plan = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, today=dose_day, rain_forecast=rain
    )
    assert plan.next_dose.defer_until is not None
    assert plan.next_dose.defer_until > dose_day


# --------------------------------------------------------------------------
# Honesty: cross-check and provenance
# --------------------------------------------------------------------------
def test_uptake_cross_check_is_reported(wheat, medium_soil_nutrients):
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    for nut in NUTRIENTS:
        assert plan.uptake_cross_check[nut] >= 0


def test_cross_check_is_shown_in_the_trace_not_warned_on(wheat, medium_soil_nutrients):
    """The uptake budget is visible but must not raise a warning.

    For wheat it lands ~2x the validated dose, and it diverges for every crop —
    the two methods are systematically incommensurable, not occasionally so. A
    warning that always fires is noise, and it would train the reader to skip the
    provenance warnings that genuinely matter. So it belongs in the trace.
    """
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)

    labels = " ".join(s.label for s in plan.explanation.steps).lower()
    assert "cross-check" in labels
    assert "not the recommendation" in labels
    assert not any("cross-check" in w for w in plan.warnings)


def test_cross_check_divergence_is_real_and_the_package_still_wins(wheat, medium_soil_nutrients):
    """Guard the decision: the validated dose is reported, not the budget."""
    plan = recommend_fertiliser(
        wheat, medium_soil_nutrients, SOWING, SOWING,
        target_yield_t_ha=wheat.nutrients.reference_yield_t_ha,
    )
    assert plan.uptake_cross_check["N"] > plan.season_dose["N"] * 1.5
    assert plan.season_dose["N"] == pytest.approx(
        wheat.nutrients.reference_dose_kg_ha["N"]
    )


def test_assumed_soil_values_are_flagged(wheat):
    """SoilGrids has no available P or K — the UI must not imply otherwise."""
    soil = SoilNutrients(
        available={"N": 350.0, "P2O5": 15.0, "K2O": 150.0},
        provenance={"N": "estimated"},   # P and K default to assumed
    )
    plan = recommend_fertiliser(wheat, soil, SOWING, SOWING)
    assert any("P2O5" in w and "assumed" in w for w in plan.warnings)
    assert any("K2O" in w and "assumed" in w for w in plan.warnings)


def test_measured_soil_values_are_not_flagged(wheat):
    soil = SoilNutrients(
        available={"N": 350.0, "P2O5": 15.0, "K2O": 150.0},
        provenance={n: "measured" for n in NUTRIENTS},
    )
    plan = recommend_fertiliser(wheat, soil, SOWING, SOWING)
    assert not any("assumed" in w for w in plan.warnings)


def test_explanation_shows_the_dose_derivation(wheat, medium_soil_nutrients):
    plan = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    exp = plan.explanation
    assert exp.inputs and exp.steps and exp.conclusion and exp.sources
    labels = " ".join(s.label for s in exp.steps)
    assert "season dose" in labels.lower()
    assert "cross-check" in labels.lower()


def test_recommendation_is_deterministic(wheat, medium_soil_nutrients):
    a = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    b = recommend_fertiliser(wheat, medium_soil_nutrients, SOWING, SOWING)
    assert a.season_dose == b.season_dose
    assert a.explanation.conclusion == b.explanation.conclusion
