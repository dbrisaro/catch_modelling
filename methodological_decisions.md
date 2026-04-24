# Methodological Decisions

## Peru Anchoveta Catch Modeling — Suyana

---

## 1. Variables Considered

The following variables were extracted from oceanographic sources and matched to each fishing event (cala) via nearest-neighbor interpolation in space and time (step 05):

| Variable | Source | Description |
|---|---|---|
| `modis_sst_anom` | MODIS AQUA | Daily SST anomaly (SST minus DOY climatology) |
| `modis_sst` | MODIS AQUA | Daily raw SST |
| `hycom_temp` | HYCOM | Daily water temperature (subsurface) |
| `hycom_temp_anom` | HYCOM | Daily temperature anomaly (subsurface) |
| `hycom_sal` | HYCOM | Daily salinity |
| `hycom_sal_anom_ref35` | HYCOM | Salinity anomaly relative to reference (35.1 PSU) |
| `chl` | MODIS AQUA | Daily chlorophyll-a concentration |
| `chl_anom` | MODIS AQUA | Daily chlorophyll-a anomaly |
| `nino12_anom` | NOAA CPC | Monthly Nino 1+2 SST anomaly index (0-10S, 80-90W) |

---

## 2. Variables Excluded from Final Model and Rationale

| Variable | Decision | Reason |
|---|---|---|
| `modis_sst` | Excluded | Raw SST conflates location effects with anomalies; anomaly form is preferred to control for seasonal and spatial baselines. |
| `hycom_temp` / `hycom_temp_anom` | Excluded | High collinearity with MODIS SST anomaly (same thermal signal); MODIS provides direct surface measurement with longer record and finer spatial coverage for the fishing area. |
| `hycom_sal` / `hycom_sal_anom_ref35` | Excluded | Salinity has a secondary physical link to anchoveta abundance relative to temperature; preliminary regressions showed weaker and less consistent signal than SST anomaly across aggregation levels. |
| `chl` / `chl_anom` | Excluded | Chlorophyll is a proxy for prey availability (phytoplankton) but has high spatial and temporal patchiness and frequent cloud gaps in MODIS coverage, limiting sample size and reliability at the cala level. |
| `nino12_anom` | Excluded from regression | Used as a descriptive context variable (step 09 visualization) rather than a regressor because it is a large-scale index that is itself driven by the same SST anomalies already included in the model; including it would introduce multicollinearity and reduce interpretability. |
| Quadratic SST term | Excluded | Step 06 explored `modis_sst_anom + modis_sst_anom^2`; the nonlinear specification did not improve fit meaningfully at seasonal aggregation level and added interpretive complexity for the parametric insurance application. |

**Final model (M1):** `log(catch) ~ modis_sst_anom` (OLS, semi-log specification)

**Parallel model (M2):** `log(CPUE) ~ modis_sst_anom` (OLS, semi-log specification) — see Section 7.

---

## 3. Temporal Scope

| Decision | Value | Rationale |
|---|---|---|
| Catch data period | 2015-2024 | IHMA fishing logbook data available from 2015 onward. |
| SST anomaly period (AEP analysis) | 2002-2026 | MODIS AQUA SST available from July 2002; extends the record well beyond the catch data period for a more stable seasonal SST distribution. |
| SST DOY climatology baseline | 2015-2025 | The day-of-year mean SST (used to compute anomalies) was calculated from the 2015-2025 daily files. Anomalies for 2002-2014 are therefore expressed relative to the 2015-2025 mean, not the full-period mean. This was a pipeline implementation decision (the climatology file was built before the 2002-2014 data was added and was not recomputed to avoid invalidating the enriched calas dataset). The practical effect is a slight warm bias in the 2015-2025 reference, which may modestly underestimate anomaly magnitudes in early El Nino events (2002-2003, 2009-2010). |
| Seasons modeled | T1 (Apr-Jul, DOY 91-212) and T2 (Nov-Dec, DOY 305-365) | Corresponds to official Peruvian anchoveta fishing seasons (primera and segunda temporada) as regulated by IMARPE/PRODUCE. |
| Off-season periods | Excluded | No fishing activity occurs outside T1 and T2 by regulation; including off-season observations would inflate zero-catch records without a physical basis for loss estimation. |

---

## 4. Spatial Scope

| Decision | Value | Rationale |
|---|---|---|
| Region Norte | lat > -7.1° | Northern fishing zone as defined by IMARPE regional boundaries. |
| Region Centro | -15.8° < lat ≤ -7.1° | Central fishing zone; primary focus of the parametric insurance product. |
| Longitude range (AEP analysis) | -81.5° to -74.7° | Covers the active fishing corridor along the Peruvian continental shelf where MODIS SST data intersects with observed cala locations. |
| Southern limit | -15.8° | IMARPE southern boundary of the central zone; fishing activity south of this latitude is sparse and not part of the modeled product. |
| Spatial aggregation | not applied | Spatial grid cells (1°×1°) were evaluated as an earlier M2 specification but removed; the CPUE normalization (Section 7) addresses vessel heterogeneity without spatial binning. |

---

## 5. Model Specification Decisions

| Decision | Value | Rationale |
|---|---|---|
| Dependent variable | log(catch_tm) | Semi-log specification gives beta the interpretation of a proportional change in catch per unit SST anomaly; normalizes the heavy right-tail of catch tonnage distributions. |
| Aggregation levels tested | Individual calas, empresa×diario, empresa×semanal, empresa×mensual, empresa×temporada | Tests whether the SST-catch relationship strengthens with aggregation (attenuation bias from measurement error is expected to decrease). |
| Fishing quotas | Excluded by design | IMARPE/PRODUCE fishing quotas are not used as a covariate or filter. Including quotas would require conditioning on quota utilization, which introduces a regulatory variable outside the control of the insured and breaks the fully parametric structure of the product. The model treats catch as a function of SST only; quota effects are absorbed into the residuals. |
| Outlier removal | IQR filter applied at individual cala level before regression | Catch observations outside Q1 - 1.5×IQR and Q1 + 1.5×IQR are excluded prior to fitting all regression models (step 06). Removes data entry errors and extreme single-haul events that would otherwise distort OLS estimates. Applied before aggregation. |
| Beta used for loss estimation | -0.849 | OLS M1, empresa×temporada level, Centro region. Seasonal aggregation minimizes attenuation bias. M2 (CPUE) yields -0.538 at the same level for 2017-2022 seasons (see Section 6). |
| Loss formula | `loss = baseline × (1 - exp(beta × SST_anom))` when SST_anom > 0 | Derived directly from the semi-log OLS specification; cold anomalies are treated as no-loss events since the insurance product covers warm-event risk only. |

---

## 6. CPUE Methodology (M2)

### 6.1 Data source: SISESAT VMS

Fishing effort is measured from the SISESAT vessel monitoring system (VMS), which records GPS pings for each vessel at approximately 9-minute intervals. Files are provided by IHMA per season (primera/segunda temporada de anchoveta).

| Decision | Value | Rationale |
|---|---|---|
| Effort unit | VMS pings × (9/60) hours | Median observed ping interval is 9 minutes; each ping represents one 9-minute time unit of vessel activity. |
| Effort definition | Total accumulated VMS time per vessel per season | No velocity filter applied; total sea time is used as a proportional proxy for fishing effort. Velocity-based filtering (e.g., < 5 knots) was considered but adds an arbitrary threshold and is less stable at fine temporal aggregation. |
| Temporal coverage | 2017-2022 (9 seasons with clean data) | SISESAT files are not available for 2015-2016. Files for 2023 (4 days coverage) and 2024 (corrupt timestamps) were excluded. Minimum coverage threshold: 30 days of clean pings within the season DOY window. |
| Date filtering | Pings filtered to season DOY window AND calendar year from filename | Some SISESAT files span multiple seasons or contain pings from adjacent years. Each file is filtered to T1 (DOY 91-212) or T2 (DOY 305-365) of the year indicated in the file path to avoid cross-season contamination. |

### 6.2 Vessel-to-company matching

SISESAT records vessel positions by vessel code (`Cod_Barco`). The calas (fishing event) data records the same vessels under the field `nave`, but with inconsistent naming for some fleets:

| Issue | Rule applied |
|---|---|
| TASA fleet naming | SISESAT uses codes like `T17`, `T21`; calas uses `TASA 17`, `TASA 21`. Normalized via regex: `T(\d+)` -> `TASA \1`. |
| Non-breaking spaces | Some vessel names contain `\xa0`; replaced with regular space before matching. |
| Case | All names upper-cased before comparison. |

Match rate: approximately 66% of vessels in calas data have a corresponding SISESAT record. The unmatched 34% consists primarily of TASA fleet vessels where the naming normalization did not fully resolve the mismatch.

Company assignment: the vessel-to-company (`empresa`) mapping is derived from the calas data (which records both vessel name and company). This lookup is applied to SISESAT vessels to aggregate effort at the company level.

### 6.3 CPUE computation

CPUE (catch per unit effort) is computed at the empresa x time aggregation level:

```
effort_empresa_period = sum of effort_hours for all vessels of that company in that period
catch_empresa_period  = sum of catch_tm from calas_enriched for that company in that period
CPUE = catch_empresa_period / effort_empresa_period
```

The dependent variable in M2 is `log(CPUE)`. The SST anomaly predictor is the mean `modis_sst_anom` across all calas of that company in that period (same as M1).

| Aggregation level | M2 available? | Note |
|---|---|---|
| Calas individuales | No | A single VMS ping cannot be matched to a specific haul (cala); individual CPUE is not computable. |
| Empresa x Diario | Yes | Daily effort summed per company from daily ping counts. |
| Empresa x Semanal | Yes | Weekly effort summed from daily records. |
| Empresa x Mensual | Yes | Monthly effort summed from daily records. |
| Empresa x Temporada | Yes | Seasonal effort summed from daily records. |

### 6.4 M2 regression results (Centro region)

| Aggregation | M1 beta (log catch) | M2 beta (log CPUE) | M2 p-value | M2 N |
|---|---|---|---|---|
| Empresa x Diario | -0.087 | -0.086 | 0.027 | 1,196 |
| Empresa x Semanal | -0.222 | -0.173 | 0.032 | 341 |
| Empresa x Mensual | -0.415 | -0.252 | 0.205 | 111 |
| Empresa x Temporada | -0.849 | -0.538 | 0.075 | 44 |

At daily and weekly resolution, M1 and M2 betas are nearly identical, indicating that at fine temporal scales effort tracks catch closely. At monthly and seasonal resolution, M2 betas are less negative than M1, suggesting that part of the catch decline in warm years is explained by companies reducing fishing hours - not solely by lower fish availability.

The M2 seasonal beta (-0.538) is estimated on 2017-2022 only (no major El Nino events). The M1 full-period beta (-0.849) includes 2015-2016, which drives much of the magnitude. Restricting M1 to the same SISESAT seasons gives beta = -0.410, comparable in magnitude to M2 = -0.538.

---

## 7. AEP Methodology Decisions

| Decision | Value | Rationale |
|---|---|---|
| Bootstrap method | Parametric (Normal fit) | With only ~23 years of MODIS data, block bootstrap cannot generate events more extreme than historical observations. A Normal distribution fitted to observed seasonal means allows tail extrapolation. |
| Distribution fitted | Normal | Starting point; symmetric tails are conservative for the warm tail but defensible given the limited sample. GEV or lognormal could be explored if heavier warm tails are warranted. |
| N simulations | 4,000 | Sufficient for a stable AEP curve down to ~0.025% exceedance probability. |
| SST triggers | p90, p95, p99 of historical seasonal distribution | Correspond to anomalies of 0.96°C, 1.38°C, and 2.75°C respectively; provide reference points for contract trigger design. |
| Baseline catch | Mean observed T1 and T2 catch per season (Centro, all companies) | T1 baseline: ~0.97M tn, T2 baseline: ~1.27M tn, annual total: ~2.24M tn. |

---

## 8. Trigger Design Decisions

### 8.1 Spatial scale selection

The SST-catch regression was run at empresa x temporada level for five spatial scopes to determine which region provides the strongest signal for trigger design:

| Region | Lat bounds | N | Beta | R² | p-value |
|---|---|---|---|---|---|
| All (Norte+Centro) | lat > -15.8 | 146 | -0.846 | 0.195 | <0.001 |
| Norte | lat > -7.1 | 94 | -0.352 | 0.039 | 0.058 |
| Centro Norte | -11.0 < lat <= -7.1 | 142 | -0.816 | **0.261** | <0.001 |
| Centro Sur | -15.8 < lat <= -11.0 | 137 | +0.107 | 0.002 | 0.653 |
| Centro | -15.8 < lat <= -7.1 | 146 | -0.849 | 0.198 | <0.001 |

**Decision: Centro Norte (-11° to -7.1°) is the recommended trigger region.** It has the highest R² (0.261) and the tightest standard error (se=0.116). Centro Sur shows no significant relationship (R²=0.002), indicating that SST anomalies in the southern sub-zone do not reliably predict catch there. Including Centro Sur in the trigger domain dilutes the signal without adding predictive power.

### 8.2 Payout structure

Two payout structures were evaluated using beta = -0.816 (Centro Norte, M1 empresa x temporada):

**Option A - Linear ramp (recommended):**
- Entry point: SST anomaly >= 0.5°C (payout begins)
- Exit point: SST anomaly >= 2.5°C (maximum payout reached)
- Payout between entry and exit: linear interpolation from 0% to 100% of maximum
- Rationale: proportional to the severity of the warm event; reduces basis risk relative to the step design; aligns with the OLS exponential loss curve at both endpoints.

**Option B - Step design:**
- Single trigger at p90 = 0.96°C
- Payout: 100% of maximum if SST_anom >= trigger, 0% otherwise
- Rationale: simpler to explain and verify; used as a benchmark comparison.

**Payout comparison at key SST levels:**

| SST anomaly | OLS exponential | Linear ramp | Step (p90) |
|---|---|---|---|
| 0.0°C | 0% | 0% | 0% |
| 0.5°C (entry) | 38% | 0% | 0% |
| 0.96°C (p90) | 62% | 23% | 100% |
| 1.38°C (p95) | 78% | 44% | 100% |
| 2.0°C | 93% | 75% | 100% |
| 2.5°C (exit) | 100% | 100% | 100% |
| 2.75°C (p99) | 100% | 100% | 100% |

The step design overpays at low anomalies (100% payout at p90, where OLS implies only 62% loss) and underpays between 0 and p90. The linear ramp more closely tracks the OLS-implied loss curve across the full anomaly range, minimizing basis risk.

**Recommended payout formula (linear ramp, normalized):**
```
payout_fraction = clip((SST_anom - 0.5) / (2.5 - 0.5), 0, 1)
payout_tons     = baseline_catch × payout_fraction × coverage_ratio
```

---

## 9. SST Spatial Homogeneity (step 16)

### 9.1 Objective

Validate that SST warming events are spatially coherent along the Peruvian fishing corridor, justifying the use of a single area-wide SST anomaly index as the parametric insurance trigger rather than requiring spatially disaggregated triggers per zone.

### 9.2 Method

| Parameter | Value |
|---|---|
| Data source | MODIS AQUA SST anomaly (daily, 2002-2026) |
| Spatial domain | lat -16° to -6°, lon -80° to -75° (inner continental shelf) |
| Temporal aggregation | Monthly means (285 months) |
| Spatial aggregation | Mean over longitude -> 10 bands of 1° latitude each |
| Band labels | 16-15°S, 15-14°S, ..., 7-6°S (southern to northern) |
| Correlation method | Pairwise Pearson r between monthly anomaly time series |

### 9.3 Results

Pairwise Pearson correlations between 1-degree latitude bands (monthly SST anomaly, 2015-2025):

|         | 16-15°S | 15-14°S | 14-13°S | 13-12°S | 12-11°S | 11-10°S | 10-9°S | 9-8°S | 8-7°S | 7-6°S |
|---------|---------|---------|---------|---------|---------|---------|--------|-------|-------|-------|
| 16-15°S | 1.00 | 0.93 | 0.84 | 0.79 | 0.75 | 0.74 | 0.69 | 0.65 | 0.62 | 0.54 |
| 15-14°S | 0.93 | 1.00 | 0.94 | 0.87 | 0.82 | 0.82 | 0.78 | 0.75 | 0.71 | 0.61 |
| 14-13°S | 0.84 | 0.94 | 1.00 | 0.93 | 0.86 | 0.85 | 0.82 | 0.79 | 0.76 | 0.65 |
| 13-12°S | 0.79 | 0.87 | 0.93 | 1.00 | 0.91 | 0.88 | 0.85 | 0.81 | 0.75 | 0.65 |
| 12-11°S | 0.75 | 0.82 | 0.86 | 0.91 | 1.00 | 0.93 | 0.86 | 0.81 | 0.74 | 0.64 |
| 11-10°S | 0.74 | 0.82 | 0.85 | 0.88 | 0.93 | 1.00 | 0.93 | 0.87 | 0.79 | 0.69 |
| 10-9°S  | 0.69 | 0.78 | 0.82 | 0.85 | 0.86 | 0.93 | 1.00 | 0.96 | 0.88 | 0.77 |
| 9-8°S   | 0.65 | 0.75 | 0.79 | 0.81 | 0.81 | 0.87 | 0.96 | 1.00 | 0.94 | 0.84 |
| 8-7°S   | 0.62 | 0.71 | 0.76 | 0.75 | 0.74 | 0.79 | 0.88 | 0.94 | 1.00 | 0.93 |
| 7-6°S   | 0.54 | 0.61 | 0.65 | 0.65 | 0.64 | 0.69 | 0.77 | 0.84 | 0.93 | 1.00 |

**Summary statistics:** median r = 0.81, min r = 0.54, max r = 0.96, 80% of pairs r > 0.70.

### 9.4 Interpretation and Decision

**Decision: the area-wide SST anomaly index is justified for the trigger design.**

Key observations:

- Correlations within the core trigger zone (Centro Norte, 11°S to 7°S, bands 11-10°S through 8-7°S) are uniformly high: r >= 0.79 for all pairs within that zone. Adjacent 1-degree bands within Centro Norte show r > 0.87.
- The weakest correlations involve the northernmost band (7-6°S), which lies near the equatorial transition zone and is less dominated by the Humboldt Current dynamics that drive the central fishing corridor. Even so, the minimum correlation (r = 0.54) still indicates a common underlying signal across the full 24-year record.
- The signal structure is consistent with large-scale ENSO-driven warm events (El Nino) that affect the entire Peruvian coast simultaneously, rather than localized SST perturbations that would require separate triggers per zone.
- A single area-wide SST anomaly index (mean over the trigger zone) captures the shared warming signal while averaging out local noise. The high inter-band correlations confirm that no systematic asymmetry exists between sub-zones that would require differentiated triggers.

**Conclusion:** A single SST anomaly index computed as the spatial mean over the Centro Norte trigger zone (lat -11° to -7.1°, lon -81.5° to -74.7°) is a valid and parsimonious parametric trigger for the insurance product. Disaggregated triggers per latitude band would not materially improve precision and would substantially increase contract complexity.

---

## 10. Company-Level Analysis (step 18)

### 10.1 Company name matching

Company names in the raw IHMA data are inconsistent across years (e.g. "CFG-COPEINCA", "CFG/COPEINCA", "COPEINCA", "copeinca" for the same fleet; "PESQUERA EXALMAR S.A.A.", "EXALMAR-CENTINELA", "EXALMAR" for Exalmar). The analysis uses case-insensitive regex substring matching (e.g. `r"copeinca"`, `r"exalmar"`) rather than exact string matching, so all naming variants are captured automatically. The code logs which specific variants were found at runtime.

### 10.2 Trigger SST for company analysis

The SST trigger applied to each company is the **area-wide mean SST anomaly** computed from all companies' calas in the Centro Norte zone (lat -11° to -7.1°) for that season - not the company's own calas. This is deliberate: the parametric trigger must be an objective, independently verifiable index that no single company can influence. Using the company's own SST would create a moral hazard (a company could inflate its trigger by fishing in warmer micro-zones) and reduce auditability.

### 10.3 Company baseline catch

The baseline catch used in the payout formula is the **historical mean seasonal catch per company per season type** (T1 and T2 separately), computed over all available seasons in calas_enriched.csv. This represents the "normal" production level the insurance product is designed to protect. No outlier exclusion is applied to the baseline calculation - all seasons including extreme years are included so the baseline reflects true long-run average exposure.

### 10.4 Payout formula at company level

The payout formula is identical to the product-wide formula (Section 8): linear ramp with entry=0.5°C and exit=2.5°C. The company-specific beta (fitted via OLS on empresa×temporada data) is computed and displayed for informational purposes only - to show how each company's catch sensitivity compares to the area-wide beta (-0.816). The company beta is **not** used in the payout calculation. Using company-specific betas would make the product non-standard and introduce negotiation risk.

### 10.5 Basis risk observations (COPEINCA and EXALMAR)

| Pattern | Explanation |
|---|---|
| Cold-year losses (2016 T1, 2017 T2, 2019 T2 for both companies) | Significant catch shortfalls in cold/neutral SST seasons. These are not covered by the product by design - likely driven by IMARPE quota reductions or low biomass assessments independent of surface temperature. |
| T2-2018 false trigger | SST = +0.94°C (above entry); both companies caught well above baseline (COPEINCA +38%, EXALMAR +71%). Product would have paid despite no loss. Reflects noise in the SST-catch relationship at mild warming levels. The linear ramp limits this to a 22% payout (not full coverage). |
| T1-2023 El Nino | Both companies captured near zero (COPEINCA 6% of baseline, EXALMAR 10%). Product covers 80-83% of the loss. The residual uncovered loss exists because SST = 2.01°C does not reach the exit point (2.5°C). |
