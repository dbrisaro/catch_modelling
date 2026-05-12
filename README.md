# Peru Catch Modeling

Parametric insurance design for Peruvian anchoveta fisheries. The pipeline links
oceanographic data (SST, chlorophyll, HYCOM) to observed catch records to estimate
the relationship between warm SST anomalies and catch losses, then uses that
relationship to price a parametric trigger.

## Running the pipeline

```bash
cd scripts
python3 run_pipeline.py              # all steps
python3 run_pipeline.py --steps 5 6  # specific steps
python3 run_pipeline.py --list       # show all steps
```

## Stages

**Wrangling - oceanographic (01-04)**
Processes raw HYCOM and MODIS AQUA files into daily netCDF grids for water
temperature, SST, and chlorophyll-a. Computes day-of-year climatologies and
daily anomalies for each variable.

**Wrangling - fishing records (05-08)**
Loads raw IHMA anchoveta cala records (2015-2025), aggregates them to daily
totals per vessel, matches each event to the nearest oceanographic grid point,
and produces a catch time series by region and season.

**Analysis (09-14)**
Cross-validates HYCOM vs MODIS SST, maps vessel density and seasonal SST
anomalies, tests spatial homogeneity of the SST signal across the fishing
corridor, and fits OLS regressions of log(catch) on SST anomaly by region.

**Pricing (15-23)**
Designs parametric triggers (entry/exit SST percentiles), runs a bootstrap
AEP for the aggregated portfolio, generates baseline and per-company pricing
reports, and produces reinsurance-ready outputs (AEP curve, region map,
payout curves).

## Key outputs

| File | Description |
|------|-------------|
| `outputs/calas_enriched.csv` | Fishing events with matched oceanographic variables |
| `outputs/catch_summary_by_region.csv` | Seasonal catch totals by region (2015-2025) |
| `outputs/betas_by_region.csv` | Conservative OLS betas used for pricing |
| `outputs/report_baseline.md` | SST trigger percentiles and catch baselines |
| `outputs/report_pricing_document.md` | Full pricing reference document |
| `outputs/21_reinsurance_analysis.md` | Reinsurance portfolio summary |

## Data sources

- **HYCOM**: 3-hourly ocean model output (temperature, salinity)
- **MODIS AQUA**: Daily SST and chlorophyll-a (2002-2026)
- **IHMA calas**: Individual fishing event records from the Peruvian fishing authority
