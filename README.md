# Smart Crop Decision Assistant — Project Plan

**SIH 2026 · Problem Statement**
> Develop a platform that recommends suitable crop management actions using crop type, soil
> conditions, and weather information provided as input.

**Status:** working end to end. All 5 crops, live soil + weather, quantified prescriptions in the
UI, **177 passing tests**. ML yield model, i18n and season calendar still to do — see §7.

```bash
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501
python -m pytest tests/ -q    # 177 passed in ~0.3 s, no network needed
```

---

## 0. Build status

| Component | State |
|---|---|
| `data/crops/*.yaml` — 5 crops, every value source-tagged | ✅ rice · wheat · maize · cotton · sugarcane |
| `src/crops.py` — spec loader, validation, FAO-56 curve reconstruction | ✅ |
| `src/engine/water.py` — FAO-56 balance + irrigation prescription | ✅ |
| `src/engine/nutrients.py` — scaled reference dose + product allocation | ✅ |
| `src/engine/explain.py` — computation traces | ✅ |
| `src/soil.py` — SoilGrids + Saxton–Rawls pedotransfer + gap fallback | ✅ |
| `src/weather.py` — archive + forecast ETo series, cached, provenance-badged | ✅ |
| `ui/advisory.py` + `app.py` — advisory is now the primary tab | ✅ |
| `tests/` — 177 tests, no network | ✅ |
| Yield-prediction ML, i18n (Hindi UI), season calendar, SQLite plots | ⬜ |

**Verified working on live data.** For a real lat/lon the engine fetches soil and weather,
reconstructs soil moisture from the sowing date, and prescribes a dated irrigation depth plus a
full-season fertiliser schedule — each with its arithmetic on show. At reference yield on a medium
soil, every crop reproduces its validated ICAR package exactly (wheat and maize 120:60:40, rice and
cotton 120:60:60, sugarcane 250:115:115).

### The five crops

| Crop | Season | Kc mid | Root m | p | Reference NPK | at | Yield basis |
|---|---|---|---|---|---|---|---|
| Rice | 150 d | 1.20 | 0.60 | **0.20** | 120:60:60 | 5.5 t/ha | grain |
| Wheat | 125 d | 1.15 | 1.10 | 0.55 | 120:60:40 | 5.0 t/ha | grain |
| Maize | 125 d | 1.20 | 1.20 | 0.55 | 120:60:40 | 5.0 t/ha | grain |
| Cotton | 195 d | 1.15 | 1.30 | 0.65 | 120:60:60 | 2.5 t/ha | **seed cotton** |
| Sugarcane | 360 d | 1.25 | 1.50 | 0.65 | 250:115:115 | 80 t/ha | **millable cane** |

Two things that table is doing deliberately:

- **`yield_basis` is not cosmetic.** Cotton yield is seed cotton (kapas), sugarcane is millable
  cane. Uptake figures are *per tonne of that*, so sugarcane's 1.3 kg N/t against wheat's 26 kg N/t
  is correct, not a typo. Reading cane figures as if they were grain would understate the nitrogen
  dose more than tenfold. A test asserts season uptake lands in a plausible 40–350 kg N/ha band for
  every crop, which catches exactly this class of error.
- **Rice's `p = 0.20`** is far below every other crop, per FAO-56 Table 22 — and rice's
  waterlogging tolerance is 90 days against maize's 2. Rice is grown *in* standing water; treating
  it like a dryland cereal would spam drainage alerts on every paddy. See the caveat in §6.1.

### What the teammate's code contributed, and what had to change

A working Streamlit app arrived mid-build. Its module structure, docstring conventions, Open-Meteo
integration, theming, and the Windows UTF-8 console fix in `scripts/train.py` are all kept. Three
things had to be replaced:

1. **`src/advisor.py` invented fertiliser dosages.** It computed `deficit = median − value` over
   dataset quartiles and printed the result as "Add roughly N kg/ha via urea". The dataset's N/P/K
   columns are dimensionless indices, not kg/ha — false precision on exactly the question this
   platform exists to answer.
2. **Its agronomic thresholds came from synthetic data.** Confirmed empirically: the Kaggle crop
   dataset is 2200 rows, **exactly 100 per crop across 22 crops**. Real survey data is never
   perfectly balanced. Test accuracy reads **99.55%** — an artifact of that separability, and a
   liability if quoted as a headline.
3. **`src/weather.py` fed a mismatched variable**, which crashed the app in production — see §6.6.

Crop recommendation is **demoted, not deleted** — reframed as a secondary "Field Suitability" tab
with its provenance stated inline, since it answers a real question, just not the one on the brief.

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

Actual layout (✅ built · ⬜ planned). The engine lives under `src/` alongside the existing modules
rather than in a separate `scda/` package, so it composes with the teammate's app instead of
forking it.

```
data/
  crops/wheat.yaml     ✅ agronomic reference data, every value source-tagged
  cache/               ⬜ weather + soil snapshots (fallback)
src/
  crops.py             ✅ load + validate specs; reconstruct FAO-56 Kc and root curves
  engine/              ✅ PURE functions — no network, no file I/O, no clock reads
    explain.py         ✅ Explanation + Step computation traces
    water.py           ✅ FAO-56 balance -> irrigation prescription
    nutrients.py       ✅ scaled reference dose -> split schedule -> products
    stress.py          ⬜ heat / cold / pest-risk alerts
    recommend.py       ⬜ orchestrator -> combined Advisory
  soil.py              ⬜ SoilGrids + Saxton-Rawls pedotransfer
  weather.py           ⚠️ exists (current weather only) — needs archive + ETo + cache
  config.py            ⚠️ exists — N/P/K mislabelled kg/ha, rainfall slider caps at 320 mm
  advisor.py           ⚠️ exists — superseded by engine/, to be retired
  model.py             ⚠️ exists — crop classifier, to be repointed at yield regression
  data_loader.py       ⚠️ exists — synthetic Kaggle dataset
  i18n.py              ⬜ en/hi message catalog
  store.py             ⬜ SQLite: plots, advisory history
app.py                 ⚠️ exists — tabs to be reordered, management actions first
tests/                 ✅ 93 tests, fixtures only, no network
```

Two invariants keep this maintainable:

1. **`engine/` is pure.** Providers fetch, engine computes, UI presents. No network calls inside —
   which is why the whole suite runs in 0.2 s with no fixtures beyond plain dataclasses, and why
   `today` is a parameter rather than a clock read.
2. **The engine emits numbers and structured traces, never prose baked into logic.** This is what
   makes Hindi nearly free and keeps the UI a pure renderer.

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

Two behaviours worth recording, both found by running the engine rather than reasoning about it:

- **Waterlogging is a warning, not a competing action.** A first implementation checked saturation
  risk *before* the irrigation trigger and returned early — so any plot whose deficit was about to
  be relieved by heavy rain was told only "arrange drainage", never the more useful "don't irrigate,
  rain is coming". Because the check fires precisely when rain refills the root zone, it was
  systematically stealing the rain-expected case. Saturation is now attached as a warning, and only
  becomes the headline when there is no deficit to report.
- **Recording past irrigation delays the next one but makes it larger.** Logging a 50 mm application
  pushed the next irrigation nine days later — and from 80 mm to 92 mm, because by then the root
  zone is deeper, so TAW and RAW are both larger. Counter-intuitive but correct; the tests assert
  the delay, not a smaller depth.

A single application above ~75 mm net is flagged rather than silently capped: it usually means
earlier irrigations were never recorded, so the balance has been accumulating deficit since sowing.
Under-reporting a real deficit would be the worse failure. The UI therefore offers a "record
irrigations already applied" input.

> ### ⚠️ Rice is not a deficit-irrigation crop
>
> The depletion model above assumes **aerobic soil under deficit irrigation**: the root zone dries,
> and we refill it. Traditional Indian rice is grown **puddled and flooded**, holding 2–5 cm of
> standing water, where the soil never dries between events. That is not the same regime.
>
> Rice's `p = 0.20` approximates **Alternate Wetting and Drying (AWD)** — the water-saving practice
> ICAR and IRRI promote, where the field drains to near-saturation before re-flooding. Under AWD the
> depths this engine reports are meaningful. **Under continuous flooding, read them as supplementary
> to ponding management, not a replacement for it.** Do not present rice irrigation figures without
> saying which regime is assumed. Recorded in `data/crops/rice.yaml`.

### 6.2 Fertilizer — `src/engine/nutrients.py`

**Method: scaled reference dose** — as used by Soil Health Card and Nutrient Expert.

```
dose = reference_dose × (target_yield / reference_yield) × fertility_factor
```

where `fertility_factor` is 1.25 / 1.00 / 0.75 for a low / medium / high soil test class. For wheat
this reproduces the validated 120:60:40 package exactly at 5 t/ha on a medium soil.

> ### ⚠️ Correction to an earlier draft of this plan
>
> This section originally specified a from-scratch uptake budget:
> `(target_yield × uptake_per_tonne − soil_credit) / use_efficiency`.
>
> **Implemented and run, that gives ~242 kg N/ha for wheat against a validated 120 kg/ha — a 2×
> overshoot.** Reconciling backwards, `130 = soil_supply + 120 × 0.45` implies the soil supplies
> ~76 kg N/ha, far more than a 2 % mineralisation of the topsoil yields. The mechanistic budget
> carries too many uncertain parameters — use efficiency, mineralisation rate, effective soil
> depth — to size a dose on its own.
>
> So the field-validated package now leads, and the uptake budget survives only as a **displayed
> cross-check**. Anchoring on decades of trial data beats anchoring on a chain of parameter guesses.

The cross-check is **shown in the trace but never warned on.** It diverges for *every* crop — the
two methods are systematically incommensurable, not occasionally divergent: the uptake budget says
"replace what the crop removes, adjusted for recovery", while validated packages account for soil
reserves built up over years and for realistic field losses. A warning that always fires is noise,
and it trains the reader to skip the provenance warnings that genuinely matter. (Sugarcane is the
interesting exception: its budget lands within a few percent of the 250 kg N/ha package, a useful
signal that its per-tonne figures are sound.)

Products are then allocated the standard Indian way: meet P with **DAP** (crediting its 18 % N —
failing to credit it would over-apply nitrogen), top the remaining N up with **urea**, meet K with
**MOP**. Split across growth stages per the crop spec — three splits for the cereals, four for
cotton and sugarcane, whose longer indeterminate seasons need feeding for longer.

**Weather coupling:** a due top-dress is **deferred when ≥30 mm of rain is forecast within 48 h**
(leaching and runoff loss), and the plan names the next dry day instead.

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

`src/weather.py` and `src/soil.py` each wrap their fetch: on success write a JSON snapshot under
`data/cache/`; on failure return the most recent snapshot tagged `CACHED · Nd`. The UI badges every
advisory `LIVE` / `CACHED · Nd` / `NO DATA`. **A dead network degrades the demo; it does not kill it.**

> ### ⚠️ SoilGrids has real coverage gaps over India
>
> Found by testing, not by reading docs. Queries over **Punjab return HTTP 200 with every value
> `null`** — both Ludhiana city *and* surrounding farmland — while Haryana, Madhya Pradesh and Tamil
> Nadu return data. Punjab is one of India's most important wheat regions, so a null response is a
> **normal case to design for, not an error path.**
>
> Resolution order is therefore: **exact point → offsets at ~5 and ~11 km → texture preset chosen by
> the user.** Ludhiana now resolves via `SoilGrids ~6 km offset`. When even that fails, the UI says
> so plainly and asks the farmer to pick a texture class, rather than inventing soil properties.

Field capacity and wilting point are **derived, never fetched** — no API supplies them. `src/soil.py`
applies the **Saxton & Rawls (2006)** pedotransfer functions to texture and organic matter. Sanity
anchor: a loam (40 % sand, 20 % clay, 1.5 % OM) gives FC 0.269 / WP 0.131, so ~0.14 m³/m³ available
water — squarely in the published 0.13–0.17 range.

### 6.6 A units mismatch that crashed the app

Worth recording because it is the archetype of the bug class this rewrite targets, and it reached
production.

Fetching live weather for **Jabalpur** raised:

```
streamlit.errors.StreamlitValueAboveMaxError:
  The value 465.7 is greater than the max_value 320.0
```

The dataset's `rainfall` column means **growing-period** rainfall (roughly 20–300 mm). `get_weather`
fills it with the **last 30 days'** precipitation. Different quantities sharing a name — and in a
monsoon month Jabalpur returns 465.7 mm, past a slider that stopped at 320.

Fixed at three levels rather than one:

1. The slider range now reaches 1200 mm, covering real monsoon totals.
2. `_apply_weather` **clamps every fetched value to its widget bounds**, so no future field can crash
   the app this way regardless of range.
3. Anything past the dataset's ~300 mm training range is reported as **extrapolation** for the
   suitability model, with a note that the Management Actions tab is unaffected — it consumes a dated
   rainfall series from Open-Meteo, never this scalar.

Raising the cap alone would have hidden the real problem: the suitability model was being fed a
variable it was never trained on.

---

## 7. Implementation phases

Each phase leaves the system in a runnable state.

| # | Phase | Output | State |
|---|---|---|---|
| 1 | Crop reference data — schema + wheat | `data/crops/wheat.yaml`, `src/crops.py`, validation | ✅ |
| 2 | Engine: water, nutrients, explain | prescriptions from fixtures | ✅ |
| 3 | Providers: `soil.py` + extend `weather.py` | real lat/lon → soil + archive + forecast, cached | ✅ |
| 4 | Rework `app.py` + `ui/advisory.py` | management actions primary, traces on show | ✅ |
| 5 | Remaining 4 crops | rice, maize, cotton, sugarcane | ✅ |
| 6 | i18n (Hindi UI) | engine already emits numbers + structured traces, so this is a catalog | ⬜ |
| 7 | Season calendar + SQLite saved plots | timeline of upcoming actions | ⬜ |
| 8 | Yield model repointed at regression | `train.py` with honest baseline comparison | ⬜ |
| 9 | `stress.py` — heat / cold / pest-risk alerts | third card alongside water and nutrients | ⬜ |

Phase 5 deliberately came *after* the engine. Replicating the reference-data schema across five
crops before proving it on one would have meant rewriting five files when the schema changed — and it
changed twice: once when the fertiliser method was corrected (§6.2), and again when `yield_basis` was
added because cotton and sugarcane are not measured in grain.

### Verified live (Aug 2026)

Both API contracts were confirmed against the real endpoints before the engine was built on them:

- **`et0_fao_evapotranspiration` exists as a daily field on BOTH the forecast and archive endpoints,
  in mm.** So FAO-56 reference evapotranspiration comes straight from the API and **Penman-Monteith is
  never implemented.** Forecast endpoint: `https://api.open-meteo.com/v1/forecast` (16 days ahead, up
  to 92 `past_days`). Archive: `https://archive-api.open-meteo.com/v1/archive`, used for sowing dates
  older than 92 days — which sugarcane, at 360 days, always needs. No API key for either.
- **SoilGrids scaling factors confirmed exactly as feared**, from the API's own `d_factor`:
  `phh2o` ×10 · `sand`/`clay`/`silt` g/kg (÷10 for %) · `bdod` cg/cm³ (÷100) · `soc` dg/kg (÷10) ·
  `nitrogen` cg/kg (÷100). A missed conversion here silently corrupts every dosage downstream.
- **SoilGrids coverage gaps over Punjab** — see §6.5.

### Still open

**Competitive positioning** — confirm what VISTAAR offers today and revise §3 if it has absorbed a
quantitative advisory module. Also examine **Krishi-DSS**, the DA&FW geospatial decision-support
system: on name alone it is the nearest neighbour in the government stack and deserves a direct look.
Web research tools were unavailable throughout the build, so §3 remains flagged rather than asserted.

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

### What is verified

**177 tests, no network, ~0.3 s.** The engine has no I/O to mock, so fixtures are plain dataclasses
and `today` is a parameter rather than a clock read.

Parametrised across **all five crops**, so a sixth automatically inherits the whole suite:

- **FAO-56 Kc curve** — flat through initial, linear ramp to `kc_mid`, flat through mid-season,
  declining to `kc_end`. Asserted at stage boundaries, interpolation midpoints, and clamped outside
  the season. Kc must peak mid-season for every crop.
- **Physical invariants** — `0 ≤ Dr ≤ TAW` always · depletion monotonic without rain · irrigation
  depth never exceeds TAW · gross = net ÷ efficiency · drip beats sprinkler beats surface · balance
  independent of input ordering · root depth monotonic and bounded 0.3–2.0 m.
- **Dose invariants** — every crop reproduces its validated reference package at reference yield ·
  scales linearly with yield target · falls as soil fertility rises · split doses sum to the season
  requirement · DAP's 18 % N credited against the urea top-up · splits sum to 1.0 per nutrient.
- **Units guards** — season N uptake must land in 40–350 kg/ha for every crop, which catches the
  cane-vs-grain scale error · `yield_basis` asserted for cotton and sugarcane.
- **Agronomic sanity anchors** — rice's depletion fraction must be the lowest of all crops · rice
  tolerates >30 days of saturation while maize tolerates ≤3.
- **Weather coupling** — forecast rain suppresses a triggered irrigation · ≥30 mm within 48 h defers a
  top-dress and names the next dry day · waterlogging is a warning beside an irrigation decision, and
  only becomes the headline when there is no deficit.
- **Determinism** — identical inputs give identical output, including explanation text.
- **Honesty** — assumed vs measured soil values flagged · the uptake cross-check appears in the trace
  and raises no warning.
- **Spec validation rejects bad data** — splits not summing to 1.0, missing stages, implausible Kc,
  inverted fertility adjustment (a richer soil demanding *more* fertiliser), doses after harvest,
  implausible mineralisation fraction.

Plus verified by hand against live APIs: soil and weather fetch for real coordinates, the Punjab
offset fallback, the `LIVE` badge, and a headless full-app render with **zero exceptions**.

### Not yet verified — read before trusting output

- **`src/soil.py`, `src/weather.py` and `ui/` have no automated tests.** They are network-dependent;
  the 177 tests cover the pure engine only. This is the largest gap in the suite.
- **The cached-fallback path has not been exercised** — no test kills the network and asserts a
  `CACHED` badge.
- **`[REVIEW]`-tagged values in `data/crops/*.yaml` are working defaults** drawn from published
  ranges, not cross-checked against primary sources. The ones that matter most:
  - `use_efficiency` and `mineralisation_fraction` (drive the nutrient budget)
  - `AVAILABLE_N_FRACTION = 0.08` in `src/soil.py` — this single number decides the N fertility class
    and therefore the nitrogen dose
  - stage lengths, which are variety- and sowing-date-dependent
  **An agronomist should sign these off before the output guides a real field.**
- **Rice under continuous flooding** — see the caveat in §6.1.
- **VISTAAR / Krishi-DSS positioning** (§3) — still unverified.

### A bug found in this codebase's own history, worth keeping in mind

While building `src/soil.py` I made *precisely* the mistake this rewrite exists to remove: comparing
SoilGrids **total** nitrogen (~4200 kg/ha over 30 cm) against Soil Health Card bands that describe
**plant-available** nitrogen (280–560 kg/ha). Every soil classified as "high", and every nitrogen dose
was cut by 25 %. Caught by noticing an implausible fertility class in real output, not by a test.

The lesson generalises: **units mismatches do not announce themselves.** They produce numbers that
look reasonable. The defences that actually work are sanity anchors on the *output* — "does a Punjab
loam really test high for nitrogen?", "is 1.3 kg N per tonne plausible for cane?" — which is why those
now exist as tests.
