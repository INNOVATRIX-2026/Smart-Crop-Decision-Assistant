"""Crop reference data loads, validates, and reconstructs the FAO-56 curves.

These tests guard the data layer that replaced dataset-derived quartiles. A
malformed spec must fail loudly here rather than silently producing a wrong
fertiliser dose in the field.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src import crops as crops_mod
from src.crops import NUTRIENTS, CropSpecError, _parse, available_crops, get_crop


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def test_wheat_loads():
    c = get_crop("wheat")
    assert c.crop == "wheat"
    assert c.category == "cereal"
    assert c.label("hi")  # Hindi label present for the i18n layer


def test_lookup_is_case_insensitive():
    assert get_crop("WHEAT").crop == "wheat"
    assert get_crop("  Wheat ").crop == "wheat"


def test_unknown_crop_names_the_alternatives():
    with pytest.raises(CropSpecError, match="available"):
        get_crop("dragonfruit")


def test_available_crops_is_sorted():
    got = available_crops()
    assert got == sorted(got)
    assert "wheat" in got


# --------------------------------------------------------------------------
# FAO-56 Kc curve (Ch. 6)
# --------------------------------------------------------------------------
# wheat.yaml stages: initial 20 / development 30 / mid 50 / late 25  => 125 days
# Control points: (0,0.30) (20,0.30) (50,1.15) (100,1.15) (125,0.35)
def test_kc_flat_through_initial_stage(wheat):
    assert wheat.water.kc_at(0) == pytest.approx(0.30)
    assert wheat.water.kc_at(10) == pytest.approx(0.30)
    assert wheat.water.kc_at(20) == pytest.approx(0.30)


def test_kc_ramps_linearly_across_development(wheat):
    # Halfway through development (day 35 of 20->50) is halfway from 0.30 to 1.15.
    assert wheat.water.kc_at(35) == pytest.approx(0.30 + 0.5 * (1.15 - 0.30))


def test_kc_flat_through_mid_season(wheat):
    assert wheat.water.kc_at(50) == pytest.approx(1.15)
    assert wheat.water.kc_at(75) == pytest.approx(1.15)
    assert wheat.water.kc_at(100) == pytest.approx(1.15)


def test_kc_declines_across_late_stage(wheat):
    assert wheat.water.kc_at(125) == pytest.approx(0.35)
    assert wheat.water.kc_at(112.5) == pytest.approx(1.15 + 0.5 * (0.35 - 1.15))


def test_kc_is_clamped_outside_the_season(wheat):
    assert wheat.water.kc_at(-5) == pytest.approx(0.30)
    assert wheat.water.kc_at(500) == pytest.approx(0.35)


def test_kc_never_leaves_the_span_of_its_control_points(wheat):
    lo, hi = 0.30, 1.15
    for das in range(0, 126):
        assert lo - 1e-9 <= wheat.water.kc_at(das) <= hi + 1e-9


# --------------------------------------------------------------------------
# Stages and rooting depth
# --------------------------------------------------------------------------
def test_stage_boundaries(wheat):
    assert wheat.water.stage_at(0) == "initial"
    assert wheat.water.stage_at(19) == "initial"
    assert wheat.water.stage_at(20) == "development"
    assert wheat.water.stage_at(49) == "development"
    assert wheat.water.stage_at(50) == "mid"
    assert wheat.water.stage_at(99) == "mid"
    assert wheat.water.stage_at(100) == "late"


def test_stage_is_defined_outside_the_season(wheat):
    """Callers should never have to special-case pre-sowing or post-harvest."""
    assert wheat.water.stage_at(-10) == "initial"
    assert wheat.water.stage_at(9999) == "late"


def test_total_days_matches_stage_sum(wheat):
    assert wheat.water.total_days == sum(s.days for s in wheat.water.stages) == 125


def test_root_depth_grows_then_holds(wheat):
    w = wheat.water
    assert w.root_depth_at(0) == pytest.approx(w.root_depth_initial_m)
    assert w.root_depth_at(50) == pytest.approx(w.root_depth_max_m)
    assert w.root_depth_at(100) == pytest.approx(w.root_depth_max_m)
    # Monotonic non-decreasing — roots do not retract.
    depths = [w.root_depth_at(d) for d in range(0, 126)]
    assert all(b >= a - 1e-9 for a, b in zip(depths, depths[1:]))


def test_stage_start_day(wheat):
    assert wheat.water.stage_start_day("initial") == 0
    assert wheat.water.stage_start_day("development") == 20
    assert wheat.water.stage_start_day("mid") == 50


# --------------------------------------------------------------------------
# Nutrient invariants
# --------------------------------------------------------------------------
def test_splits_sum_to_one_per_nutrient(wheat):
    for nut in NUTRIENTS:
        total = sum(s.fraction.get(nut, 0.0) for s in wheat.nutrients.splits)
        assert total == pytest.approx(1.0), f"{nut} splits sum to {total}"


def test_fertility_adjustment_decreases_with_fertility(wheat):
    fa = wheat.nutrients.fertility_adjustment
    assert fa["low"] > fa["medium"] > fa["high"]


def test_reference_dose_covers_all_nutrients(wheat):
    for nut in NUTRIENTS:
        assert wheat.nutrients.reference_dose_kg_ha[nut] > 0
    assert wheat.nutrients.reference_yield_t_ha > 0


def test_no_split_falls_after_harvest(wheat):
    for s in wheat.nutrients.splits:
        assert 0 <= s.day <= wheat.water.total_days


# --------------------------------------------------------------------------
# Validation rejects bad specs
# --------------------------------------------------------------------------
MINIMAL = textwrap.dedent(
    """
    crop: testcrop
    water:
      kc: {initial: 0.3, mid: 1.1, end: 0.4}
      stages:
        - {name: initial, days: 10}
        - {name: development, days: 20}
        - {name: mid, days: 30}
        - {name: late, days: 10}
      root_depth_m: {initial: 0.1, max: 1.0}
      depletion_fraction_p: 0.5
      application_efficiency: {surface: 0.6}
    nutrients:
      uptake_per_tonne_grain: {N: 20, P2O5: 10, K2O: 25}
      use_efficiency: {N: 0.45, P2O5: 0.2, K2O: 0.6}
      splits:
        - {stage: basal, day: 0, fraction: {N: 1.0, P2O5: 1.0, K2O: 1.0}}
      attainable_yield_t_ha: {irrigated: 4.0}
      reference_dose_kg_ha: {N: 100, P2O5: 50, K2O: 40, at_yield_t_ha: 4.0}
      fertility_adjustment: {low: 1.25, medium: 1.0, high: 0.75}
    soil_fertility_classes:
      available_N_kg_ha: {low: [0, 280], medium: [280, 560], high: [560, 2000]}
    mineralisation_fraction: 0.02
    """
)


def _parse_yaml(text: str):
    import yaml
    return _parse(yaml.safe_load(text), Path("testcrop.yaml"))


def test_minimal_spec_is_accepted():
    spec = _parse_yaml(MINIMAL)
    assert spec.crop == "testcrop"
    assert spec.water.total_days == 70


def test_splits_that_do_not_sum_to_one_are_rejected():
    """The most dangerous silent error: under- or over-applying the season dose."""
    bad = MINIMAL.replace("fraction: {N: 1.0,", "fraction: {N: 0.8,")
    with pytest.raises(CropSpecError, match="sum to"):
        _parse_yaml(bad)


def test_missing_required_stage_is_rejected():
    bad = MINIMAL.replace("- {name: mid, days: 30}", "- {name: peak, days: 30}")
    with pytest.raises(CropSpecError, match="mid"):
        _parse_yaml(bad)


def test_non_positive_stage_length_is_rejected():
    bad = MINIMAL.replace("days: 20", "days: 0")
    with pytest.raises(CropSpecError, match="non-positive"):
        _parse_yaml(bad)


def test_out_of_range_depletion_fraction_is_rejected():
    bad = MINIMAL.replace("depletion_fraction_p: 0.5", "depletion_fraction_p: 1.5")
    with pytest.raises(CropSpecError, match="depletion_fraction_p"):
        _parse_yaml(bad)


def test_implausible_kc_is_rejected():
    bad = MINIMAL.replace("mid: 1.1", "mid: 12.0")
    with pytest.raises(CropSpecError, match="plausible"):
        _parse_yaml(bad)


def test_root_depth_initial_above_max_is_rejected():
    bad = MINIMAL.replace("{initial: 0.1, max: 1.0}", "{initial: 1.5, max: 1.0}")
    with pytest.raises(CropSpecError, match="root depth"):
        _parse_yaml(bad)


def test_inverted_fertility_adjustment_is_rejected():
    """A richer soil must never be told to apply more fertiliser."""
    bad = MINIMAL.replace(
        "{low: 1.25, medium: 1.0, high: 0.75}", "{low: 0.75, medium: 1.0, high: 1.25}"
    )
    with pytest.raises(CropSpecError, match="richer soil"):
        _parse_yaml(bad)


def test_split_after_harvest_is_rejected():
    bad = MINIMAL.replace("day: 0, fraction", "day: 500, fraction")
    with pytest.raises(CropSpecError, match="after harvest"):
        _parse_yaml(bad)


def test_implausible_mineralisation_fraction_is_rejected():
    bad = MINIMAL.replace("mineralisation_fraction: 0.02", "mineralisation_fraction: 0.9")
    with pytest.raises(CropSpecError, match="implausible"):
        _parse_yaml(bad)


def test_missing_nutrient_is_rejected():
    bad = MINIMAL.replace("{N: 20, P2O5: 10, K2O: 25}", "{N: 20, P2O5: 10}")
    with pytest.raises(CropSpecError, match="K2O"):
        _parse_yaml(bad)


def test_every_shipped_spec_validates():
    """Loading the real data directory must not raise — catches a bad commit."""
    crops_mod.load_crops.cache_clear()
    specs = crops_mod.load_crops()
    assert specs, "no crop specs shipped"
    for name, spec in specs.items():
        assert spec.crop == name
        assert spec.water.total_days > 0
