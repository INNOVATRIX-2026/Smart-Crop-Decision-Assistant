# Smart Crop Decision Assistant — Project Plan

**SIH 2026 · Problem Statement**
> Develop a platform that recommends suitable crop management actions using crop type, soil
> conditions, and weather information provided as input.

**Status:** planning complete, implementation not started.

---

## 1. Reading the problem correctly

The single most important observation about this problem statement:

> **Crop type is an INPUT, not an output.**

This is *not* the familiar "which crop should I plant?" classification exercise that most crop
datasets and tutorials solve. The farmer has already sown a crop. What they need is a
**management advisory**: when to irrigate and how much, when to fertilize and how much, and what
stresses are coming.

That distinction drives every design decision that follows:

| | "Which crop?" (not our problem) | **"How do I manage it?" (our problem)** |
|---|---|---|
| Output type | a category (rice / wheat / …) | **a quantity + a date** (38 mm on 2 Sep) |
| Right method | classification model | **agronomic computation** |
| Labels needed | crop labels (widely available) | dosage labels (**do not exist publicly**) |

Because the output is a *quantity*, not a category, it must be **computed** from established
agronomic models — not guessed by a classifier. There is no public dataset mapping soil + weather
to correct fertilizer dosages, so any system claiming to "predict the right amount" with ML alone
is either overfitting to a proxy or quietly inventing its labels.

**Consequence:** the recommendation backbone is a transparent rules/physics engine, and machine
learning plays a well-defined supporting role. This is not a retreat from ML — it is what makes
every number on screen defensible when a farmer (or a judge) asks *"why that amount?"*

---

## 2. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Stack | Python only — **FastAPI** backend + **Streamlit** UI | One language; fastest path to a working demo; API boundary preserved for a future mobile client |
| Engine | **Hybrid** — agronomic rules backbone + ML support | Explainable *and* data-driven |
| Crops | 5 — rice, wheat, maize, cotton, sugarcane | Covers most of India; reference data stays hand-verifiable |
| Soil | Automatic from lat/lon via **SoilGrids** | Zero data entry (see §7 risk 2) |
| Weather | **Open-Meteo**, live + cached fallback | Free, no API key, no signup; demo survives network loss |
| Extras | Hindi/English UI · explainability panel · saved plots + season calendar · yield-prediction ML | All four in scope |

---

## 3. Competitive landscape — what makes this better, not just different

> ⚠️ **Verification pending.** Web research tools were unavailable during planning, so the VISTAAR
> and Krishi-DSS rows below rest on prior knowledge. **Confirm at `https://vistaar.da.gov.in/`
> and for Krishi-DSS before repeating these claims in a submission or pitch.** The rest of this
> section is high-confidence.

### Is this the same as VISTAAR?

**Adjacent, but a different layer.** VISTAAR is a Department of Agriculture & Farmers' Welfare
initiative positioned as digital public infrastructure for agricultural **extension** — unifying
fragmented advisory content and connecting farmers with extension services, within the Digital
Agriculture Mission / AgriStack ecosystem. That is a **content delivery and aggregation** layer.

This project is a **computation** layer: it derives plot-specific quantities. The two are
complementary rather than competing — a platform like VISTAAR is a realistic *distribution
channel* for an engine like this one, which is a stronger story than claiming to replace it.

### The real incumbent to beat

Not VISTAAR — the **IMD / ICAR Agromet Advisory Service (GKMS, "Meghdoot" app)**. It already
issues crop-specific, weather-based advisories twice weekly, nationwide, through District Agromet
Units. Any pitch that ignores this loses credibility immediately.

**The exploitable gap:** those advisories are **qualitative and block-level** — *"light irrigation
may be given"* — not *"38 mm on 2 September, for your plot."*

| System | Quantified dose | Weather-coupled | Plot-level | No hardware / lab | Shows its working |
|---|---|---|---|---|---|
| IMD Agromet / Meghdoot | ✗ qualitative | ✓ | ✗ block-level | ✓ | partial |
| Soil Health Card portal | ✓ fertilizer only | ✗ | ✓ | ✗ needs lab test | ✓ |
| Nutrient Expert | ✓ fertilizer only | ✗ | ✓ | ✓ | ✓ |
| CROPWAT / AquaCrop (FAO) | ✓ water only | partial | ✓ | ✓ | ✓ but expert-only |
| Fasal / IoT advisory services | ✓ | ✓ | ✓ | ✗ needs field sensors | ✗ black box |
| Plantix | n/a (pest ID) | ✗ | ✓ | ✓ | ✗ |
| VISTAAR (extension DPI) | ✗ | — | ✗ | ✓ | — |
| **This project** | **✓ water *and* nutrients** | **✓** | **✓** | **✓** | **✓** |

### The five differentiators to lead with

1. **Both levers from one coupled model.** Every incumbent does irrigation *or* fertilizer. We
   compute both against the same forecast window — which is what enables advice like *"delay the
   top-dress, 40 mm of rain is coming"* instead of treating the two decisions independently.
2. **Numbers, not adjectives.** "Apply 38 mm on 2 Sep" vs "light irrigation may be given." The
   sharpest and most demonstrable contrast with the incumbent bulletins.
3. **No hardware, no soil test.** Works from lat/lon + crop + sowing date alone. Fasal needs
   sensors in the ground; Soil Health Card needs a lab. **This is the adoption argument.**
4. **Plot-level, not block-level.** Derived from *this* plot's sowing date and rainfall history,
   not a district bulletin shared by thousands of farms.
5. **It shows its arithmetic.** IoT advisory services are black boxes. We ship the trace — the
   trust argument for farmers and the defensibility argument for judges.

Differentiators 3 and 4 are only *simultaneously* possible because of the technique in §4. **That
is the actual technical contribution and should lead the pitch.**

---

## 4. The core technique: soil moisture without a sensor

**The problem.** Irrigation advice needs current soil moisture. We have no sensor, and SoilGrids
returns only *static* soil properties (texture, bulk density, pH, carbon) — never today's
moisture. Assuming a value would make every irrigation number arbitrary.

**The solution.** Replay the water balance forward from the sowing date using the historical
weather archive, then continue into the 16-day forecast. Rainfall and evapotranspiration since
sowing determine current root-zone depletion.

```
sowing date ──── archive rain + ETo ────▶ today ──── forecast ────▶ +16 days
                 (reconstruct depletion)          (project next action)
```

This converts *"what is the soil moisture right now?"* from an unanswerable input into a
**computed state**, using only data we can actually obtain for any plot in the country.

---

## 5. Architecture

```
scda/
  models.py            Pydantic contracts: SoilProfile, WeatherSeries, CropSpec,
                       PlotInput, Recommendation, Explanation, Advisory
  config.py            paths, API base URLs, cache TTLs
  providers/
    base.py            @cached_provider — live fetch, snapshot, fallback on failure
    weather.py         Open-Meteo forecast + archive
    soil.py            SoilGrids + Saxton–Rawls pedotransfer functions
  engine/              PURE functions over the models — no network, no I/O
    crops.py           load + validate crop reference data
    phenology.py       sowing date -> growth stage -> Kc, root depth, MAD
    water.py           FAO-56 water balance -> irrigation prescription
    nutrients.py       nutrient budget -> split fertilizer schedule
    stress.py          heat / cold / waterlogging / pest-risk alerts
    explain.py         Explanation + Step trace objects
    recommend.py       orchestrator -> Advisory
  ml/
    predictor.py       Predictor Protocol; DefaultYield (ships now) | TrainedYield
    features.py        season weather aggregation from archive
    train.py           CLI: train + temporal/region-split evaluation
  store.py             SQLite: plots, advisory history
  i18n.py              en/hi message catalog
  api.py               FastAPI app
ui/
  app.py               Streamlit entry
  views/               input_form · advisory_cards · explain_panel · calendar
data/
  crops/*.yaml         agronomic reference data, one file per crop, every value sourced
  cache/               weather + soil snapshots (fallback)
  training/            district yield CSV
  app.db               SQLite
tests/                 water · nutrients · phenology · providers (fixtures, no network)
```

Two invariants keep this maintainable:

1. **`engine/` is pure.** Providers fetch, engine computes, api/ui present. The engine takes
   `SoilProfile` + `WeatherSeries` + `CropSpec` and returns an `Advisory`, with no network calls
   inside — so it is fully unit-testable against FAO-56 worked examples.
2. **The engine emits message IDs and numbers, never prose.** `irrigate.now` +
   `{depth_mm: 38}`. This is what makes Hindi nearly free and keeps the UI a pure renderer.

---

## 6. The two prescription computations

### 6.1 Irrigation — `engine/water.py`

FAO-56 root-zone depletion tracking:

```
θ_FC, θ_WP  ← Saxton–Rawls pedotransfer from SoilGrids sand / clay / organic carbon
TAW   = (θ_FC − θ_WP) × root_depth(stage)     total available water, mm
RAW   = p × TAW                                p = crop MAD, adjusted upward for high ETc
ETc   = ETo × Kc(stage)                        ETo from Open-Meteo daily field
Dr[i] = Dr[i−1] − P_eff[i] − I[i] + ETc[i]     daily depletion, replayed from sowing
```

**Prescription:** walk forward from today; the first day `Dr ≥ RAW` is the irrigation date, and the
depth is `Dr` at that point (refill to field capacity) divided by application efficiency.

**Suppress the event if forecast effective rainfall covers the deficit.** This is the weather input
doing genuine work, and "don't irrigate, it's going to rain" is among the most valuable things the
system can say.

### 6.2 Fertilizer — `engine/nutrients.py`

```
N_req    = target_yield × N_uptake_per_tonne / N_use_efficiency
N_credit = SoilGrids total N × mineralization_fraction × bulk_density × depth
N_dose   = max(0, N_req − N_credit)
```

Split across growth stages per the crop spec (rice ≈ 50 % basal / 25 % tillering / 25 % panicle),
then convert nutrient → product: urea 46 % N, DAP 18-46-0, MOP 60 % K₂O.

Weather couples in here too: **defer a top-dress when heavy rain is forecast within 48 h**
(leaching and runoff loss), and coordinate fertilizer timing with irrigation events.

`target_yield` is the single value supplied by ML — see §6.3.

### 6.3 Where ML actually sits

`ml/predictor.py` defines a `Predictor` Protocol with two implementations:

- **`DefaultYieldPredictor`** — attainable yield from the crop spec, scaled by a soil-suitability
  factor (pH, organic carbon, texture). No training required, so **the platform is fully
  functional from day one.**
- **`TrainedYieldPredictor`** — loads a fitted model if present, else falls back.

Training (`ml/train.py`):

- **Labels:** district-wise crop yield (kg/ha) from public Indian crop production statistics.
- **Features:** SoilGrids properties + season weather aggregates from the Open-Meteo *archive*
  (growing degree days, total and distribution of rainfall, heat-stress day count, mean ETo).
- **Split: temporal (train ≤ y−2, validate y−1, test y) AND grouped by district — never random.**
  Weather is heavily autocorrelated in time and space; a random split leaks near-duplicate rows
  into the test set and produces a flattering, meaningless score.
- **Always report against a baseline** (district-mean predictor). If the model does not beat
  baseline MAE, we say so in the README and keep the default predictor. That is a legitimate
  finding, not something to bury.

Because only `target_yield` flows from ML into the nutrient budget, a weak model degrades dosing
slightly — **it can never break the app or corrupt the irrigation logic.**

### 6.4 Explainability — designed in, not bolted on

Every `Recommendation` carries an `Explanation` holding the inputs used, the ordered computation
steps (label, expression, value, unit), the threshold compared against, and the conclusion. The
explain panel **renders data the engine already produced** — it never re-derives the logic, so it
cannot drift out of sync with the actual decision. That is why it is nearly free to build.

Target output:

```
IRRIGATION  ·  act within 48 h
  Apply 38 mm  (≈380 m³/ha)
  Why: available soil water 42 %; wheat MAD 55 %
       ETc next 3 days = 14 mm, forecast rain = 2 mm

FERTILIZER  ·  12 Sep, at tillering
  Urea 57 kg/ha   (split 2 of 3)
  Why: target 4.5 t/ha → 120 kg N/ha required
       soil N credit 18 kg/ha; 40 kg N already applied basal
```

### 6.5 Data provenance and graceful degradation

`providers/base.py` wraps each fetch: on success write a JSON snapshot keyed by rounded
lat/lon + date; on failure return the most recent snapshot tagged `CACHED` with its age. Every
`Advisory` carries a `data_provenance` field and the UI badges it `[LIVE]` / `[CACHED · 2d]`.
**A dead network degrades the demo; it does not kill it.**

---

## 7. Implementation phases

Each phase leaves the system in a runnable state.

| # | Phase | Output |
|---|---|---|
| 0 | **Documentation + verification (no code)** | this `PLAN.md`; API contracts and competitor claims confirmed |
| 1 | Scaffold: `requirements.txt`, package skeleton, Pydantic models | importable package |
| 2 | Providers + caching | `/weather` and `/soil` return real data for a lat/lon |
| 3 | Crop reference data — 5 crops | `data/crops/*.yaml`, every value sourced in a comment |
| 4 | Engine: phenology, water, nutrients, stress, explain | full `Advisory` from fixtures; unit tests green |
| 5 | FastAPI | `POST /recommend` working end to end |
| 6 | Streamlit UI + i18n + explain panel | demoable app, Hindi/English toggle |
| 7 | SQLite plots + season calendar | save a plot, see upcoming actions on a timeline |
| 8 | ML yield model | `train.py` with honest baseline comparison |
| 9 | README + demo script | judge-ready |

### Open verification items (Phase 0)

**Competitive positioning** — confirm what VISTAAR offers today and revise §3 if it has absorbed a
quantitative advisory module. Also examine **Krishi-DSS**, the DA&FW geospatial decision-support
system: on name alone it is the nearest neighbour to "crop decision support" in the government
stack and deserves a direct look.

**API contracts** — the field names below are from established knowledge and **must be confirmed
against the live endpoints before the engine is built on them:**

- Open-Meteo forecast `https://api.open-meteo.com/v1/forecast`, archive
  `https://archive-api.open-meteo.com/v1/archive` — no API key. Daily fields expected:
  `et0_fao_evapotranspiration`, `precipitation_sum`, `temperature_2m_max`/`_min`,
  `shortwave_radiation_sum`, `wind_speed_10m_max`.
  *If `et0_fao_evapotranspiration` is available, we get FAO-56 reference evapotranspiration
  directly and skip implementing Penman-Monteith — check this first, with PM as fallback.*
- SoilGrids
  `https://rest.isric.org/soilgrids/v2.0/properties/query?lon=&lat=&property=&depth=&value=`
  — no key, rate-limited. Properties expected: `phh2o`, `soc`, `nitrogen`, `clay`, `sand`,
  `silt`, `bdod`, `cec`.
  **Note the scaling factors** (pH is ×10, texture in g/kg, …) — a missed conversion here
  silently corrupts every dosage downstream.

---

## 8. Risks

1. **Agronomic reference data is the real risk — not the code.** A wrong Kc curve or N-uptake
   figure yields confidently wrong dosages, which is worse than no advice at all.
   *Mitigation:* every number in `data/crops/*.yaml` carries a source comment, values stay within
   published FAO-56 / ICAR ranges, and `tests/` checks the water balance against FAO-56 worked
   examples.

2. **SoilGrids has no plant-available P or K.** This is the known cost of fully-automatic soil
   input. Total N converts to a seasonal credit via a mineralization fraction, but P and K must
   fall back to a fertility class.
   *Recommendation:* auto-fill from location as decided, but keep the soil fields **editable**, so
   a farmer holding a Soil Health Card can override with measured values and get accurate dosing —
   roughly 15 lines of UI for a large accuracy gain. The UI must state plainly which values are
   **measured** and which are **estimated**.

3. **The yield model may not beat baseline** on coarse district data. Handled by design: the
   default predictor ships regardless and the comparison is reported honestly.

4. **This advice has financial consequences.** Water and fertilizer cost money, and a mistake can
   cost a season. The UI frames output as decision *support* with the numbers exposed — not as an
   authoritative instruction — and states its assumptions.

---

## 9. Verification

- **Unit tests, no network** — `pytest tests/`: water balance against FAO-56 worked examples;
  nutrient budget arithmetic; phenology stage boundaries at sowing and harvest edges; providers
  against saved fixtures.
- **Engine determinism** — identical inputs produce an identical advisory, proving no hidden I/O
  or clock dependence.
- **Live end-to-end** — start API + UI, request an advisory for a real lat/lon (e.g. a Punjab
  wheat plot with a past sowing date), confirm the `[LIVE]` badge and non-empty irrigation and
  fertilizer cards with populated explanation traces.
- **Fallback path** — disable the network, re-request, confirm `[CACHED · Nd]` and that the
  advisory still renders.
- **Sanity checks that catch real bugs:**
  - irrigation depth never exceeds TAW
  - forecast heavy rain suppresses an irrigation event
  - N dose falls as soil N credit rises
  - split doses sum to the season requirement
- **ML** — `python -m scda.ml.train` prints a test-set MAE table against the district-mean
  baseline, under both temporal and region-held-out splits.
